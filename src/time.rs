pub mod address;
mod http_date;
#[cfg(target_os = "windows")]
mod windows_input;
use self::{
    address::{UrlScheme, parse_server},
    http_date::{civil_from_days, parse_http_date_to_systemtime},
};
use crate::{
    buffmt::{ByteCursor, DIGITS, TWO_DIGITS, write_zero_err},
    numeric::low_u8_from_u32,
    write_line_ignored,
};
use alloc::{borrow::Cow, fmt, str};
use core::{mem, ops::Mul as _, result as stdresult, time::Duration};
use std::{
    io::{self, BufRead as _, BufReader, Result as ioResult, Write as _},
    net::{self, TcpStream},
    process::{Command, Stdio},
    sync::{LazyLock, mpsc},
    thread,
    time::{Instant, SystemTime, SystemTimeError, UNIX_EPOCH},
};
const FULL_SYNC_INTERVAL: Duration = Duration::from_mins(5);
const RETRY_DELAY: Duration = Duration::from_secs(10);
const TCP_TIMEOUT_SECS: u64 = 5;
pub const NUM_SAMPLES: usize = 10;
const FINAL_COUNTDOWN_RTT_ALPHA_NUM: u32 = 7;
const FINAL_COUNTDOWN_RTT_ALPHA_DENOM: u32 = 10;
const MAX_CALIBRATION_FAILURES: u32 = 100;
pub const KST_OFFSET_SECS_U64: u64 = 9 * 3600;
const KST_OFFSET_SECS: i64 = KST_OFFSET_SECS_U64.cast_signed();
const DAY_OF_WEEK_KO: [&str; 7] = ["일", "월", "화", "수", "목", "금", "토"];
const DISPLAY_INTERVAL: Duration = Duration::from_millis(16);
const ADAPTIVE_POLL_INTERVAL: Duration = Duration::from_millis(10);
const DISPLAY_UPDATE_INTERVAL: Duration = Duration::from_millis(45);
#[cfg(target_os = "windows")]
const TIMERR_NOERROR: u32 = 0;
#[cfg(target_os = "windows")]
const TARGET_PERIOD_MS: u32 = 1;
pub static CURL_AVAILABLE: LazyLock<bool> = LazyLock::new(|| {
    Command::new("curl")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
});
#[cfg(target_os = "linux")]
pub static XDO_TOOL_AVAILABLE: LazyLock<bool> = LazyLock::new(|| {
    Command::new("xdotool")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
});
#[cfg(target_os = "windows")]
pub struct HighResTimerGuard;
#[cfg(target_os = "windows")]
#[link(name = "winmm")]
unsafe extern "system" {
    fn timeBeginPeriod(u_period: u32) -> u32;
    fn timeEndPeriod(u_period: u32) -> u32;
}
#[cfg(target_os = "windows")]
impl Drop for HighResTimerGuard {
    fn drop(&mut self) {
        // SAFETY: This releases the timer period requested when the guard was created
        // using the same value on the same process.
        unsafe {
            timeEndPeriod(TARGET_PERIOD_MS);
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TimeErrorKind {
    Curl,
    HeaderNotFound,
    Io,
    Parse,
    SyncFailed,
    Time,
}
#[derive(Debug)]
pub struct TimeError {
    detail: Cow<'static, str>,
    io_kind: Option<io::ErrorKind>,
    kind: TimeErrorKind,
}
impl TimeError {
    fn header_not_found(detail: impl Into<Cow<'static, str>>) -> Self {
        Self::new(TimeErrorKind::HeaderNotFound, detail)
    }
    pub const fn io_kind(&self) -> Option<io::ErrorKind> {
        self.io_kind
    }
    fn new(kind: TimeErrorKind, detail: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind,
            detail: detail.into(),
            io_kind: None,
        }
    }
    fn parse(detail: impl Into<Cow<'static, str>>) -> Self {
        Self::new(TimeErrorKind::Parse, detail)
    }
    fn sync_failed(detail: impl Into<Cow<'static, str>>) -> Self {
        Self::new(TimeErrorKind::SyncFailed, detail)
    }
}
impl From<io::Error> for TimeError {
    fn from(err: io::Error) -> Self {
        let io_kind = err.kind();
        Self {
            kind: TimeErrorKind::Io,
            detail: Cow::Owned(err.to_string()),
            io_kind: Some(io_kind),
        }
    }
}
impl From<SystemTimeError> for TimeError {
    fn from(err: SystemTimeError) -> Self {
        Self {
            kind: TimeErrorKind::Time,
            detail: Cow::Owned(err.to_string()),
            io_kind: None,
        }
    }
}
impl fmt::Display for TimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            TimeErrorKind::Io => write!(f, "I/O 오류: {}", self.detail),
            TimeErrorKind::Time => write!(f, "시스템 시간 오류: {}", self.detail),
            TimeErrorKind::Parse => write!(f, "파싱 오류: {}", self.detail),
            TimeErrorKind::HeaderNotFound => write!(f, "{} 헤더를 찾을 수 없음", self.detail),
            TimeErrorKind::Curl => write!(f, "curl 실행 실패: {}", self.detail),
            TimeErrorKind::SyncFailed => write!(f, "서버 시간 확인 실패: {}", self.detail),
        }
    }
}
type Result<T> = stdresult::Result<T, TimeError>;
#[derive(Clone, Copy, Debug)]
pub struct TimeSample {
    pub response_received_inst: Instant,
    pub rtt: Duration,
    pub server_time: SystemTime,
}
#[derive(Debug)]
pub struct ServerTime {
    anchor_instant: Instant,
    anchor_time: SystemTime,
    baseline_rtt: Duration,
}
struct DisplayableTime {
    day_of_month: u32,
    day_of_week_str: &'static str,
    hour: u32,
    millis: u32,
    minute: u32,
    month: u32,
    second: u32,
    year: i32,
}
struct SliceCursor<'buffer> {
    inner: ByteCursor<'buffer>,
}
impl SliceCursor<'_> {
    fn checked_add_index(value: usize, amount: usize) -> ioResult<usize> {
        value.checked_add(amount).ok_or_else(write_zero_err)
    }
    fn checked_sub_index(value: usize, amount: usize) -> ioResult<usize> {
        value.checked_sub(amount).ok_or_else(write_zero_err)
    }
    fn digit_byte(index: usize) -> ioResult<u8> {
        DIGITS.get(index).copied().ok_or_else(write_zero_err)
    }
    fn two_digits(index: usize) -> ioResult<&'static [u8; 2]> {
        TWO_DIGITS.get(index).ok_or_else(write_zero_err)
    }
    fn write_byte(&mut self, byte: u8) -> ioResult<()> {
        self.inner.write_byte(byte)
    }
    fn write_bytes(&mut self, bytes: &[u8]) -> ioResult<()> {
        self.inner.write_bytes(bytes)
    }
    fn write_u32_2digits(&mut self, value: u32) -> ioResult<()> {
        let idx = usize::from(low_u8_from_u32(value));
        self.inner.take(2)?.copy_from_slice(Self::two_digits(idx)?);
        Ok(())
    }
    fn write_u32_3digits(&mut self, value: u32) -> ioResult<()> {
        let hundreds = usize::from(low_u8_from_u32(value.div_euclid(100)));
        let rem = usize::from(low_u8_from_u32(value.rem_euclid(100)));
        let head = self.inner.take(3)?;
        let (hundreds_digit, remaining_digits) = head.split_at_mut(1);
        if let Some(slot) = hundreds_digit.first_mut() {
            *slot = Self::digit_byte(hundreds)?;
        }
        remaining_digits.copy_from_slice(Self::two_digits(rem)?);
        Ok(())
    }
    fn write_u32_dec(&mut self, mut n: u32) -> ioResult<()> {
        let mut tmp = [0_u8; 10];
        let mut i = tmp.len();
        while n >= 100 {
            let rem = usize::from(low_u8_from_u32(n.rem_euclid(100)));
            n /= 100;
            i = Self::checked_sub_index(i, 2)?;
            let end = Self::checked_add_index(i, 2)?;
            tmp.get_mut(i..end)
                .ok_or_else(write_zero_err)?
                .copy_from_slice(Self::two_digits(rem)?);
        }
        if n >= 10 {
            let rem = usize::from(low_u8_from_u32(n));
            i = Self::checked_sub_index(i, 2)?;
            let end = Self::checked_add_index(i, 2)?;
            tmp.get_mut(i..end)
                .ok_or_else(write_zero_err)?
                .copy_from_slice(Self::two_digits(rem)?);
        } else {
            i = Self::checked_sub_index(i, 1)?;
            let digit = usize::from(low_u8_from_u32(n));
            *tmp.get_mut(i).ok_or_else(write_zero_err)? = Self::digit_byte(digit)?;
        }
        self.write_bytes(tmp.get(i..).ok_or_else(write_zero_err)?)
    }
    fn write_year_padded4(&mut self, year: i32) -> ioResult<()> {
        if year >= 0_i32 {
            let year_value = year.cast_unsigned();
            if year_value < 10_000 {
                let hi = usize::from(low_u8_from_u32(year_value.div_euclid(100)));
                let lo = usize::from(low_u8_from_u32(year_value.rem_euclid(100)));
                let head = self.inner.take(4)?;
                let (hi_digits, lo_digits) = head.split_at_mut(2);
                hi_digits.copy_from_slice(Self::two_digits(hi)?);
                lo_digits.copy_from_slice(Self::two_digits(lo)?);
                return Ok(());
            }
            return self.write_u32_dec(year_value);
        }
        self.write_byte(b'-')?;
        let abs = year.unsigned_abs();
        if abs < 1000 {
            let hundreds = usize::from(low_u8_from_u32(abs.div_euclid(100)));
            let rem = usize::from(low_u8_from_u32(abs.rem_euclid(100)));
            let head = self.inner.take(3)?;
            let (hundreds_digit, remaining_digits) = head.split_at_mut(1);
            if let Some(slot) = hundreds_digit.first_mut() {
                *slot = Self::digit_byte(hundreds)?;
            }
            remaining_digits.copy_from_slice(Self::two_digits(rem)?);
            return Ok(());
        }
        self.write_u32_dec(abs)
    }
    const fn written_len(&self) -> usize {
        self.inner.written_len()
    }
}
impl ServerTime {
    fn calculate_display_time(&self) -> Result<DisplayableTime> {
        let current_time = self.current_server_time();
        let since_epoch = current_time.duration_since(UNIX_EPOCH)?;
        let total_seconds = parse_result_with_context(
            i64::try_from(since_epoch.as_secs()),
            "초 계산 중 범위 오류",
        )?;
        let total_seconds_kst = total_seconds
            .checked_add(KST_OFFSET_SECS)
            .ok_or_else(|| TimeError::parse("초 계산 중 범위 오류"))?;
        let millis = since_epoch.subsec_millis();
        let days_since_epoch = total_seconds_kst.div_euclid(86400);
        let day_of_week_num = days_since_epoch
            .checked_add(4)
            .ok_or_else(|| TimeError::parse("요일 계산 중 범위 오류"))?
            .rem_euclid(7);
        let day_of_week_idx =
            parse_result_with_context(usize::try_from(day_of_week_num), "요일 계산 중 범위 오류")?;
        let day_of_week_str = DAY_OF_WEEK_KO
            .get(day_of_week_idx)
            .copied()
            .ok_or_else(|| TimeError::parse("요일 계산 중 범위 오류"))?;
        let sec_of_day = total_seconds_kst.rem_euclid(86400);
        let hour = parse_result_with_context(
            u32::try_from(sec_of_day.div_euclid(3600)),
            "시 계산 중 범위 오류",
        )?;
        let minute = parse_result_with_context(
            u32::try_from(sec_of_day.rem_euclid(3600).div_euclid(60)),
            "분 계산 중 범위 오류",
        )?;
        let second = parse_result_with_context(
            u32::try_from(sec_of_day.rem_euclid(60)),
            "초 계산 중 범위 오류",
        )?;
        let day_index =
            parse_result_with_context(i32::try_from(days_since_epoch), "일자 계산 중 범위 오류")?;
        let (year, month, day_of_month) =
            civil_from_days(day_index).ok_or_else(|| TimeError::parse("일자 계산 중 범위 오류"))?;
        Ok(DisplayableTime {
            day_of_month,
            day_of_week_str,
            hour,
            millis,
            minute,
            month,
            second,
            year,
        })
    }
    fn current_server_time(&self) -> SystemTime {
        let elapsed_since_anchor = self.anchor_instant.elapsed();
        self.anchor_time
            .checked_add(elapsed_since_anchor)
            .unwrap_or(self.anchor_time)
    }
    fn recalibrate_with_rtt(&self, new_rtt: Duration) -> Self {
        let old_rtt_nanos = self.baseline_rtt.as_nanos();
        let new_rtt_nanos = new_rtt.as_nanos();
        let smoothed_rtt_nanos =
            old_rtt_nanos
                .checked_mul(u128::from(7_u8))
                .map_or(new_rtt_nanos, |weighted_old| {
                    new_rtt_nanos.checked_mul(u128::from(3_u8)).map_or(
                        new_rtt_nanos,
                        |weighted_new| {
                            weighted_old.checked_add(weighted_new).map_or(
                                new_rtt_nanos,
                                |weighted_sum| {
                                    weighted_sum
                                        .checked_div(u128::from(10_u8))
                                        .unwrap_or(new_rtt_nanos)
                                },
                            )
                        },
                    )
                });
        let smoothed_rtt = Duration::from_nanos_u128(smoothed_rtt_nanos);
        Self {
            anchor_time: self.anchor_time,
            anchor_instant: self.anchor_instant,
            baseline_rtt: smoothed_rtt,
        }
    }
    fn write_current_display_time_buf(
        &self,
        cur: &mut SliceCursor<'_>,
        show_millis: bool,
    ) -> Result<()> {
        let dt = self.calculate_display_time()?;
        cur.write_year_padded4(dt.year).map_err(TimeError::from)?;
        cur.write_byte(b'-').map_err(TimeError::from)?;
        cur.write_u32_2digits(dt.month).map_err(TimeError::from)?;
        cur.write_byte(b'-').map_err(TimeError::from)?;
        cur.write_u32_2digits(dt.day_of_month)
            .map_err(TimeError::from)?;
        cur.write_byte(b'(').map_err(TimeError::from)?;
        cur.write_bytes(dt.day_of_week_str.as_bytes())
            .map_err(TimeError::from)?;
        cur.write_bytes(b") ").map_err(TimeError::from)?;
        cur.write_u32_2digits(dt.hour).map_err(TimeError::from)?;
        cur.write_byte(b':').map_err(TimeError::from)?;
        cur.write_u32_2digits(dt.minute).map_err(TimeError::from)?;
        cur.write_byte(b':').map_err(TimeError::from)?;
        cur.write_u32_2digits(dt.second).map_err(TimeError::from)?;
        if show_millis {
            cur.write_byte(b'.').map_err(TimeError::from)?;
            cur.write_u32_3digits(dt.millis).map_err(TimeError::from)?;
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub enum TriggerAction {
    F5Press,
    LeftClick,
}
pub struct AppState {
    pub baseline_rtt: Option<Duration>,
    pub baseline_rtt_attempts: usize,
    pub baseline_rtt_next_sample_at: Instant,
    pub baseline_rtt_samples: [TimeSample; NUM_SAMPLES],
    pub baseline_rtt_valid_count: usize,
    pub calibration_failure_count: u32,
    #[cfg(target_os = "windows")]
    pub high_res_timer_guard: Option<HighResTimerGuard>,
    pub host: String,
    pub last_sample: Option<TimeSample>,
    pub live_rtt: Option<Duration>,
    pub next_full_sync_at: Instant,
    pub server_time: Option<ServerTime>,
    pub target_time: Option<SystemTime>,
    pub trigger_action: Option<TriggerAction>,
}
struct NetworkContext {
    curl_stderr_buf: String,
    tcp_line_buffer: Vec<u8>,
}
#[derive(Clone, Copy, Debug)]
enum Activity {
    CalibrateOnTick,
    FinalCountdown { target_time: SystemTime },
    Finished,
    MeasureBaselineRtt,
    Predicting,
    Retrying { retry_at: Instant },
}
trait AppStateRunLoopExt {
    fn display_server_time_line(server_time: &ServerTime, out: &mut dyn io::Write) -> Result<()>;
    fn next_activity<'message>(
        &mut self,
        activity: &Activity,
        message_buffer: &'message mut String,
        network_context: &mut NetworkContext,
        out: &mut dyn io::Write,
        err: &mut dyn io::Write,
    ) -> (Activity, Option<&'message str>);
}
impl AppStateRunLoopExt for AppState {
    fn display_server_time_line(server_time: &ServerTime, out: &mut dyn io::Write) -> Result<()> {
        let mut line_buf = [0_u8; 80];
        let mut cur = SliceCursor {
            inner: ByteCursor::new(&mut line_buf),
        };
        match (|| -> Result<()> {
            cur.write_bytes("\r서버 시간: ".as_bytes())
                .map_err(TimeError::from)?;
            server_time.write_current_display_time_buf(&mut cur, true)?;
            cur.write_bytes(b" \r").map_err(TimeError::from)?;
            Ok(())
        })() {
            Ok(()) => {
                let used = cur.written_len();
                out.write_all(
                    line_buf
                        .get(..used)
                        .ok_or_else(|| io::Error::other("표시 버퍼 사용 길이 계산 실패"))?,
                )?;
                out.flush()?;
            }
            Err(display_err) => {
                out.write_all("\r서버 시간: ".as_bytes())?;
                write!(out, "표시 버퍼 오류: {display_err}")?;
                out.write_all(b" \r")?;
                out.flush()?;
            }
        }
        Ok(())
    }
    fn next_activity<'message>(
        &mut self,
        activity: &Activity,
        message_buffer: &'message mut String,
        network_context: &mut NetworkContext,
        out: &mut dyn io::Write,
        err: &mut dyn io::Write,
    ) -> (Activity, Option<&'message str>) {
        match *activity {
            Activity::MeasureBaselineRtt => {
                self.handle_measure_baseline_rtt(message_buffer, network_context, out)
            }
            Activity::CalibrateOnTick => {
                self.handle_calibrate_on_tick(message_buffer, network_context)
            }
            Activity::Predicting => self.handle_predicting(message_buffer),
            Activity::FinalCountdown { target_time } => {
                self.handle_final_countdown(target_time, message_buffer, network_context, err)
            }
            Activity::Finished => (Activity::Predicting, Some("액션 완료. 예측 모드 전환.")),
            Activity::Retrying { retry_at } => {
                if Instant::now() >= retry_at {
                    (
                        Activity::MeasureBaselineRtt,
                        Some("[재시도] 동기화를 다시 시작합니다."),
                    )
                } else {
                    (Activity::Retrying { retry_at }, None)
                }
            }
        }
    }
}
impl AppState {
    fn begin_baseline_rtt_measurement(&mut self, now: Instant, out: &mut dyn io::Write) {
        if self.last_sample.is_none() {
            write_line_ignored(out, format_args!("1단계: RTT 기준값 측정을 시작합니다..."));
        }
        let placeholder = TimeSample {
            response_received_inst: now,
            rtt: Duration::ZERO,
            server_time: UNIX_EPOCH,
        };
        self.baseline_rtt_samples = [placeholder; NUM_SAMPLES];
        self.baseline_rtt_valid_count = 0;
        self.baseline_rtt_next_sample_at = now;
    }
    fn finish_baseline_rtt_measurement<'message>(
        &mut self,
        sample_count: usize,
        msg_buf: &'message mut String,
    ) -> (Activity, Option<&'message str>) {
        self.baseline_rtt_attempts = 0;
        self.baseline_rtt_valid_count = 0;
        if sample_count == 0 {
            return transition_to_retry("유효한 RTT 샘플을 얻지 못했습니다.");
        }
        let mut rtt_nanos = [0_u128; NUM_SAMPLES];
        let mut filled = 0_usize;
        for sample in &self.baseline_rtt_samples {
            if sample.rtt > Duration::ZERO {
                let Some(slot) = rtt_nanos.get_mut(filled) else {
                    return transition_to_retry("RTT 샘플 저장 범위 계산 실패.");
                };
                *slot = sample.rtt.as_nanos();
                let Some(next_filled) = filled.checked_add(1) else {
                    return transition_to_retry("RTT 샘플 개수 계산 실패.");
                };
                filled = next_filled;
                if filled >= sample_count {
                    break;
                }
            }
        }
        let Some(rtts) = rtt_nanos.get_mut(..filled) else {
            return transition_to_retry("RTT 샘플 범위 계산 실패.");
        };
        rtts.sort_unstable();
        let trim = filled.div_euclid(5);
        let Some(window_end) = filled.checked_sub(trim) else {
            return transition_to_retry("RTT 샘플 윈도우 계산 실패.");
        };
        let Some(window) = rtts.get(trim..window_end) else {
            return transition_to_retry("RTT 샘플 윈도우 계산 실패.");
        };
        let sum_nanos: u128 = window.iter().sum();
        let window_len = u128::try_from(window.len()).unwrap_or_default();
        let Some(avg_nanos) = sum_nanos.checked_div(window_len) else {
            return transition_to_retry("RTT 기준값 계산 실패.");
        };
        let baseline_rtt = Duration::from_nanos_u128(avg_nanos);
        self.baseline_rtt = Some(baseline_rtt);
        self.calibration_failure_count = 0;
        msg_buf.push_str("[완료] RTT 기준값: ");
        msg_buf.push_str(baseline_rtt.as_millis().to_string().as_str());
        msg_buf.push_str("ms. 2단계: 정밀 보정을 시작합니다.");
        (Activity::CalibrateOnTick, Some(msg_buf))
    }
    fn handle_calibrate_on_tick<'message>(
        &mut self,
        _msg_buf: &'message mut str,
        net_ctx: &mut NetworkContext,
    ) -> (Activity, Option<&'message str>) {
        let current_sample = if let Ok(sample) = fetch_server_time_sample(&self.host, net_ctx) {
            self.calibration_failure_count = 0;
            sample
        } else {
            self.calibration_failure_count = self.calibration_failure_count.saturating_add(1);
            if self.calibration_failure_count >= MAX_CALIBRATION_FAILURES {
                return transition_to_retry(
                    "정밀 보정 중 서버 응답을 지속적으로 받지 못했습니다. 전체 보정을 다시 시작합니다.",
                );
            }
            return (Activity::CalibrateOnTick, None);
        };
        if let Some(prev_sample) = self.last_sample
            && let Some(baseline_rtt) = self.baseline_rtt
            && let Ok(prev_dur) = prev_sample.server_time.duration_since(UNIX_EPOCH)
            && let Ok(current_dur) = current_sample.server_time.duration_since(UNIX_EPOCH)
            && prev_dur.as_secs() != current_dur.as_secs()
            && current_dur.as_secs().wrapping_sub(prev_dur.as_secs()) == 1
        {
            let calibrated_server_time = (|| -> Result<ServerTime> {
                let mut server_time_at_tick = current_sample.server_time;
                let since_epoch = server_time_at_tick.duration_since(UNIX_EPOCH)?;
                let nanos_to_subtract = since_epoch.subsec_nanos();
                let nanos_to_subtract_duration = Duration::from_nanos(u64::from(nanos_to_subtract));
                server_time_at_tick = server_time_at_tick
                    .checked_sub(nanos_to_subtract_duration)
                    .ok_or_else(|| TimeError::parse("서버 시각 보정 중 범위 오류"))?;
                let one_way_delay = current_sample
                    .rtt
                    .checked_div(2)
                    .ok_or_else(|| TimeError::parse("RTT 지연 계산 중 범위 오류"))?;
                Ok(ServerTime {
                    anchor_time: server_time_at_tick,
                    anchor_instant: current_sample
                        .response_received_inst
                        .checked_sub(one_way_delay)
                        .unwrap_or(current_sample.response_received_inst),
                    baseline_rtt,
                })
            })();
            if let Ok(server_time) = calibrated_server_time {
                self.server_time = Some(server_time);
                let Some(next_full_sync_at) = Instant::now().checked_add(FULL_SYNC_INTERVAL) else {
                    return transition_to_retry("다음 전체 동기화 시각 계산 실패.");
                };
                self.next_full_sync_at = next_full_sync_at;
                return (Activity::Predicting, Some("[성공] 정밀 보정 완료!"));
            }
        }
        self.last_sample = Some(current_sample);
        (Activity::CalibrateOnTick, None)
    }
    fn handle_final_countdown<'message>(
        &mut self,
        target_time: SystemTime,
        msg_buf: &'message mut String,
        net_ctx: &mut NetworkContext,
        err: &mut dyn io::Write,
    ) -> (Activity, Option<&'message str>) {
        let sample = match fetch_server_time_sample(&self.host, net_ctx) {
            Ok(sample_value) => sample_value,
            Err(fetch_err) => {
                return self.handle_final_countdown_fetch_error(
                    target_time,
                    msg_buf,
                    &fetch_err,
                    err,
                );
            }
        };
        let Some(st) = self.server_time.as_mut() else {
            return (
                Activity::MeasureBaselineRtt,
                Some("[오류] 내부 상태 불일치: server_time 없음"),
            );
        };
        *st = st.recalibrate_with_rtt(sample.rtt);
        let current_server_time = st.current_server_time();
        let old_rtt = self.live_rtt.unwrap_or(sample.rtt);
        let sample_rtt_nanos = sample.rtt.as_nanos();
        let new_rtt_nanos = old_rtt
            .as_nanos()
            .checked_mul(u128::from(
                FINAL_COUNTDOWN_RTT_ALPHA_DENOM - FINAL_COUNTDOWN_RTT_ALPHA_NUM,
            ))
            .map_or(sample_rtt_nanos, |weighted_old| {
                sample_rtt_nanos
                    .checked_mul(u128::from(FINAL_COUNTDOWN_RTT_ALPHA_NUM))
                    .map_or(sample_rtt_nanos, |weighted_new| {
                        weighted_old.checked_add(weighted_new).map_or(
                            sample_rtt_nanos,
                            |weighted_sum| {
                                weighted_sum
                                    .checked_div(u128::from(FINAL_COUNTDOWN_RTT_ALPHA_DENOM))
                                    .unwrap_or(sample_rtt_nanos)
                            },
                        )
                    })
            });
        let live_rtt = Duration::from_nanos_u128(new_rtt_nanos);
        self.live_rtt = Some(live_rtt);
        let effective_rtt = live_rtt.max(sample.rtt);
        let Some(one_way_delay) = effective_rtt.checked_div(2) else {
            msg_buf.push_str("카운트다운 지연 계산 실패");
            return (Activity::FinalCountdown { target_time }, Some(msg_buf));
        };
        match target_time.duration_since(current_server_time) {
            Ok(duration_until_target) if duration_until_target <= one_way_delay => {
                self.trigger_and_finish(
     msg_buf,
     format_args!(
         "\n>>> 액션 실행! (목표 도달까지 {:.1}ms 남음) (지연 예측: {:.1}ms, 실측 RTT: {:.1}ms)",
         duration_until_target.as_secs_f64().mul(1000.0),
         one_way_delay.as_secs_f64().mul(1000.0),
         sample.rtt.as_secs_f64().mul(1000.0)
     ),
     err
)
            }
            Err(_) => {
                self.trigger_and_finish(
     msg_buf,
     format_args!(
         "\n>>> 액션 실행! (시간 초과) (지연 예측: {:.1}ms, 실측 RTT: {:.1}ms)",
         one_way_delay.as_secs_f64().mul(1000.0),
         sample.rtt.as_secs_f64().mul(1000.0)
     ),
     err
)
            }
            _ => (Activity::FinalCountdown { target_time }, None),
        }
    }
    fn handle_final_countdown_fetch_error<'message>(
        &self,
        target_time: SystemTime,
        msg_buf: &'message mut String,
        err: &TimeError,
        stderr: &mut dyn io::Write,
    ) -> (Activity, Option<&'message str>) {
        if let Some(st) = self.server_time.as_ref() {
            let current_server_time = st.current_server_time();
            let effective_rtt = self.live_rtt.unwrap_or(st.baseline_rtt);
            let Some(one_way_delay) = effective_rtt.checked_div(2) else {
                msg_buf.push_str("카운트다운 지연 계산 실패");
                return (Activity::FinalCountdown { target_time }, Some(msg_buf));
            };
            match target_time.duration_since(current_server_time) {
                Ok(duration_until_target) if duration_until_target <= one_way_delay => {
                    return self.trigger_and_finish(
                        msg_buf,
                        format_args!(
                            "\n>>> 액션 실행! (예측값 기준 강제 실행, 목표 도달까지 {:.1}ms 남음) (지연 예측: {:.1}ms, 원인: 카운트다운 샘플 실패: {err})",
                            duration_until_target.as_secs_f64().mul(1000.0),
                            one_way_delay.as_secs_f64().mul(1000.0)
                        ),
                        stderr,
                    );
                }
                Err(_) => {
                    return self.trigger_and_finish(
                        msg_buf,
                        format_args!(
                            "\n>>> 액션 실행! (예측값 기준 강제 실행, 시간 초과) (지연 예측: {:.1}ms, 원인: 카운트다운 샘플 실패: {err})",
                            one_way_delay.as_secs_f64().mul(1000.0)
                        ),
                        stderr,
                    );
                }
                _ => {}
            }
        }
        msg_buf.push_str("카운트다운 샘플 획득 실패: ");
        msg_buf.push_str(err.to_string().as_str());
        (Activity::FinalCountdown { target_time }, Some(msg_buf))
    }
    fn handle_measure_baseline_rtt<'message>(
        &mut self,
        msg_buf: &'message mut String,
        net_ctx: &mut NetworkContext,
        out: &mut dyn io::Write,
    ) -> (Activity, Option<&'message str>) {
        let now = Instant::now();
        if self.baseline_rtt_attempts == 0 {
            self.begin_baseline_rtt_measurement(now, out);
        }
        if now < self.baseline_rtt_next_sample_at {
            return (Activity::MeasureBaselineRtt, None);
        }
        let attempt_index = self.baseline_rtt_attempts;
        match fetch_server_time_sample(&self.host, net_ctx) {
            Ok(sample) if sample.rtt > Duration::ZERO => {
                let Some(slot) = self.baseline_rtt_samples.get_mut(attempt_index) else {
                    return transition_to_retry("RTT 샘플 인덱스 계산 실패.");
                };
                *slot = sample;
                let Some(next_valid_count) = self.baseline_rtt_valid_count.checked_add(1) else {
                    return transition_to_retry("RTT 샘플 개수 계산 실패.");
                };
                self.baseline_rtt_valid_count = next_valid_count;
                self.last_sample = Some(sample);
            }
            Ok(_) => {}
            Err(err) => {
                self.baseline_rtt_attempts = 0;
                self.baseline_rtt_valid_count = 0;
                msg_buf.push_str("RTT 샘플 수집 실패: ");
                msg_buf.push_str(err.to_string().as_str());
                return transition_to_retry(msg_buf);
            }
        }
        let Some(next_attempt_index) = attempt_index.checked_add(1) else {
            return transition_to_retry("RTT 시도 횟수 계산 실패.");
        };
        self.baseline_rtt_attempts = next_attempt_index;
        let Some(next_sample_at) = Instant::now().checked_add(ADAPTIVE_POLL_INTERVAL) else {
            return transition_to_retry("다음 RTT 샘플 시각 계산 실패.");
        };
        self.baseline_rtt_next_sample_at = next_sample_at;
        if self.baseline_rtt_attempts < NUM_SAMPLES {
            return (Activity::MeasureBaselineRtt, None);
        }
        self.finish_baseline_rtt_measurement(self.baseline_rtt_valid_count, msg_buf)
    }
    fn handle_predicting<'message>(
        &mut self,
        _msg_buf: &'message mut String,
    ) -> (Activity, Option<&'message str>) {
        let Some(server_time) = self.server_time.as_ref() else {
            return (Activity::MeasureBaselineRtt, None);
        };
        let estimated_server_time = server_time.current_server_time();
        if let Some(target_time) = self.target_time.take_if(|target| {
            target
                .duration_since(estimated_server_time)
                .is_ok_and(|duration_until_target| duration_until_target <= Duration::from_secs(10))
        }) {
            self.live_rtt = Some(server_time.baseline_rtt);
            return (
                Activity::FinalCountdown { target_time },
                Some("최종 카운트다운 시작!"),
            );
        }
        if Instant::now() >= self.next_full_sync_at {
            self.server_time = None;
            self.baseline_rtt = None;
            self.live_rtt = None;
            (
                Activity::MeasureBaselineRtt,
                Some("서버 시간 보정 주기 도래, 재보정 시작."),
            )
        } else {
            (Activity::Predicting, None)
        }
    }
    pub(crate) fn run_loop(
        &mut self,
        out: &mut dyn io::Write,
        err: &mut dyn io::Write,
    ) -> Result<()> {
        out.write_all("\n서버 시간 확인을 시작합니다... (Enter를 누르면 종료)\n".as_bytes())?;
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || -> ioResult<()> {
            let mut line = String::new();
            io::stdin().read_line(&mut line)?;
            match tx.send(()) {
                Ok(()) | Err(_) => {}
            }
            Ok(())
        });
        let mut activity = Activity::MeasureBaselineRtt;
        let mut last_display_update = Instant::now();
        let mut message_buffer = String::with_capacity(256);
        let mut network_context = NetworkContext {
            tcp_line_buffer: Vec::with_capacity(256),
            curl_stderr_buf: String::with_capacity(1024),
        };
        loop {
            let activity_poll = match activity {
                Activity::MeasureBaselineRtt
                | Activity::CalibrateOnTick
                | Activity::FinalCountdown { .. } => ADAPTIVE_POLL_INTERVAL,
                Activity::Predicting | Activity::Finished | Activity::Retrying { .. } => {
                    DISPLAY_UPDATE_INTERVAL
                }
            };
            let elapsed = last_display_update.elapsed();
            let remaining_display = if elapsed >= DISPLAY_INTERVAL {
                Duration::ZERO
            } else {
                DISPLAY_INTERVAL.saturating_sub(elapsed)
            };
            let poll_timeout = activity_poll.min(remaining_display);
            match rx.recv_timeout(poll_timeout) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            let now = Instant::now();
            if now.duration_since(last_display_update) >= DISPLAY_INTERVAL
                && let Some(st) = self.server_time.as_ref()
            {
                <Self as AppStateRunLoopExt>::display_server_time_line(st, out)?;
                last_display_update = now;
            }
            message_buffer.clear();
            let (next_activity, log_opt_msg) = <Self as AppStateRunLoopExt>::next_activity(
                self,
                &activity,
                &mut message_buffer,
                &mut network_context,
                out,
                err,
            );
            #[cfg(target_os = "windows")]
            self.sync_high_res_timer_state(&next_activity, err);
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            Self::sync_high_res_timer_state(&next_activity, err);
            if let Some(console_msg) = log_opt_msg {
                writeln!(out, "\n{console_msg}")?;
            }
            activity = next_activity;
        }
        Ok(())
    }
    #[cfg(target_os = "windows")]
    fn sync_high_res_timer_state(&mut self, next_activity: &Activity, err: &mut dyn io::Write) {
        if matches!(next_activity, Activity::FinalCountdown { .. }) {
            if self.high_res_timer_guard.is_none() {
                // SAFETY: `timeBeginPeriod` is a WinMM FFI call with a plain integer input
                // and does not impose additional aliasing or lifetime requirements.
                if unsafe { timeBeginPeriod(TARGET_PERIOD_MS) } == TIMERR_NOERROR {
                    self.high_res_timer_guard = Some(HighResTimerGuard);
                } else {
                    write_line_ignored(
                        err,
                        format_args!(
                            "[경고] Windows 타이머 해상도 {TARGET_PERIOD_MS}ms 요청에 실패했습니다. 카운트다운 정확도가 저하될 수 있습니다."
                        ),
                    );
                }
            }
        } else {
            self.high_res_timer_guard = None;
        }
    }
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    const fn sync_high_res_timer_state(_next_activity: &Activity, _err: &mut dyn io::Write) {}
    fn trigger_and_finish<'message>(
        &self,
        msg_buf: &'message mut String,
        log_message: fmt::Arguments,
        err: &mut dyn io::Write,
    ) -> (Activity, Option<&'message str>) {
        if let Some(action) = self.trigger_action {
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            let run_external_command = |program: &str, args: &[&str]| match Command::new(program)
                .args(args)
                .status()
            {
                Ok(status) if status.success() => {}
                Ok(status) => {
                    write_line_ignored(
                        err,
                        format_args!("[경고] 명령 실행 실패: {program} {args:?} (상태: {status})"),
                    );
                }
                Err(err) => {
                    write_line_ignored(
                        err,
                        format_args!("[경고] 명령 실행 실패: {program} {args:?} ({err})"),
                    );
                }
            };
            match action {
                TriggerAction::LeftClick => {
                    #[cfg(target_os = "linux")]
                    run_external_command("xdotool", &["click", "1"]);
                    #[cfg(target_os = "macos")]
                    run_external_command(
                        "osascript",
                        &["-e", r#"tell application "System Events" to click"#],
                    );
                    #[cfg(target_os = "windows")]
                    windows_input::send_action(windows_input::InputAction::MouseClick, err);
                }
                TriggerAction::F5Press => {
                    #[cfg(target_os = "linux")]
                    run_external_command("xdotool", &["key", "F5"]);
                    #[cfg(target_os = "macos")]
                    run_external_command(
                        "osascript",
                        &["-e", r#"tell application "System Events" to key code 96"#],
                    );
                    #[cfg(target_os = "windows")]
                    windows_input::send_action(windows_input::InputAction::F5Press, err);
                }
            }
        }
        msg_buf.push_str(log_message.to_string().as_str());
        (Activity::Finished, Some(msg_buf))
    }
}
fn parse_err_with_source(context: &'static str, err: impl fmt::Display) -> TimeError {
    TimeError::parse(format!("{context}: {err}"))
}
fn parse_result_with_context<T, E>(
    result: stdresult::Result<T, E>,
    context: &'static str,
) -> Result<T>
where
    E: fmt::Display,
{
    result.map_err(|err| parse_err_with_source(context, err))
}
pub fn get_validated_input<T, F>(
    prompt: &str,
    input_buf: &mut String,
    out: &mut dyn io::Write,
    mut validator: F,
) -> ioResult<T>
where
    F: FnMut(&str) -> stdresult::Result<T, &'static str>,
{
    loop {
        out.write_all(prompt.as_bytes())?;
        out.flush()?;
        input_buf.clear();
        let bytes_read = io::stdin().read_line(input_buf)?;
        if bytes_read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "표준 입력이 종료되었습니다.",
            ));
        }
        let trimmed = input_buf.trim();
        match validator(trimmed) {
            Ok(value) => return Ok(value),
            Err(err) => {
                if !err.is_empty() {
                    writeln!(out, "{err}")?;
                }
            }
        }
    }
}
fn transition_to_retry(msg: &str) -> (Activity, Option<&str>) {
    let retry_at = Instant::now()
        .checked_add(RETRY_DELAY)
        .unwrap_or_else(Instant::now);
    (Activity::Retrying { retry_at }, Some(msg))
}
fn find_date_header_value(line: &[u8]) -> Option<&str> {
    let prefix = line.get(..5)?;
    if prefix.eq_ignore_ascii_case(b"date:") {
        str::from_utf8(line.get(5..)?).ok().map(str::trim_ascii)
    } else {
        None
    }
}
fn fetch_server_time_sample_curl(
    url_str: &str,
    context: &str,
    net_ctx: &mut NetworkContext,
) -> Result<TimeSample> {
    net_ctx.curl_stderr_buf.clear();
    let time_before_curl_call_inst = Instant::now();
    let timeout_str = TCP_TIMEOUT_SECS.to_string();
    let output = Command::new("curl")
        .args([
            "-sI",
            "--ssl-no-revoke",
            "-L",
            "--max-time",
            timeout_str.as_str(),
            "--connect-timeout",
            timeout_str.as_str(),
            "-w",
            "\n%{time_starttransfer}",
            url_str,
        ])
        .output()?;
    let stdout_bytes = output.stdout;
    let stderr_bytes = output.stderr;
    if !output.status.success() {
        net_ctx
            .curl_stderr_buf
            .push_str(&String::from_utf8_lossy(&stderr_bytes));
        if net_ctx.curl_stderr_buf.trim().is_empty() {
            net_ctx.curl_stderr_buf.clear();
            net_ctx.curl_stderr_buf.push_str("curl ");
            net_ctx.curl_stderr_buf.push_str(context);
            net_ctx.curl_stderr_buf.push_str(" 실패, 상태: ");
            net_ctx
                .curl_stderr_buf
                .push_str(output.status.to_string().as_str());
        }
        return Err(TimeError::new(
            TimeErrorKind::Curl,
            mem::take(&mut net_ctx.curl_stderr_buf),
        ));
    }
    let mut end = stdout_bytes.len();
    while end > 0
        && end
            .checked_sub(1)
            .and_then(|idx| stdout_bytes.get(idx))
            .is_some_and(u8::is_ascii_whitespace)
    {
        let Some(next_end) = end.checked_sub(1) else {
            break;
        };
        end = next_end;
    }
    let trimmed_stdout = stdout_bytes
        .get(..end)
        .ok_or_else(|| TimeError::parse("curl 응답 경계 계산 실패"))?;
    let pos = trimmed_stdout
        .iter()
        .rposition(|&byte| byte == b'\n')
        .ok_or_else(|| TimeError::parse("curl 응답에서 time_starttransfer 정보 누락"))?;
    let headers_part = trimmed_stdout
        .get(..pos)
        .ok_or_else(|| TimeError::parse("curl 응답 헤더 범위 계산 실패"))?;
    let time_starttransfer_start = pos
        .checked_add(1)
        .ok_or_else(|| TimeError::parse("curl 응답 time_starttransfer 범위 계산 실패"))?;
    let time_starttransfer_part = trimmed_stdout
        .get(time_starttransfer_start..)
        .ok_or_else(|| TimeError::parse("curl 응답 time_starttransfer 범위 계산 실패"))?;
    let time_starttransfer_str = parse_result_with_context(
        str::from_utf8(time_starttransfer_part),
        "curl time_starttransfer 파싱 실패",
    )?;
    let time_starttransfer_secs: f64 = time_starttransfer_str
        .trim_ascii()
        .parse()
        .map_err(|err| parse_err_with_source("curl time_starttransfer 파싱 실패", err))?;
    let rtt_reported_by_curl = Duration::from_secs_f64(time_starttransfer_secs.max(0.000_001));
    let date_header_str_slice = headers_part
        .split(|&byte| byte == b'\n')
        .rev()
        .find_map(|line| find_date_header_value(line))
        .ok_or_else(|| TimeError::header_not_found("curl 응답에서 Date 헤더를 찾을 수 없음"))?;
    let server_time = parse_http_date_to_systemtime(date_header_str_slice)?;
    let response_received_inst = time_before_curl_call_inst
        .checked_add(rtt_reported_by_curl)
        .ok_or_else(|| TimeError::parse("응답 수신 시각 계산 실패"))?;
    Ok(TimeSample {
        response_received_inst,
        rtt: rtt_reported_by_curl,
        server_time,
    })
}
fn fetch_server_time_sample(host: &str, net_ctx: &mut NetworkContext) -> Result<TimeSample> {
    let parsed_address = parse_server(host)?;
    if parsed_address.scheme() == Some(UrlScheme::Https) {
        let https_url = parsed_address.curl_url(UrlScheme::Https);
        return fetch_server_time_sample_curl(&https_url, "HTTPS (explicit)", net_ctx);
    }
    let tcp_attempt_result = {
        let request_start_inst = Instant::now();
        let tcp_timeout = Duration::from_secs(TCP_TIMEOUT_SECS);
        let socket_addr_result: ioResult<net::SocketAddr> = if let Ok(ip_addr) =
            parsed_address.host().parse::<net::IpAddr>()
        {
            Ok(net::SocketAddr::new(ip_addr, parsed_address.port()))
        } else {
            net::ToSocketAddrs::to_socket_addrs(&(parsed_address.host(), parsed_address.port()))?
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Host not found"))
        };
        let socket_addr = socket_addr_result?;
        let mut stream = TcpStream::connect_timeout(&socket_addr, tcp_timeout)?;
        stream.set_read_timeout(Some(tcp_timeout))?;
        stream.set_write_timeout(Some(tcp_timeout))?;
        stream.write_all(b"HEAD / HTTP/1.1\r\nHost: ")?;
        let host_header = parsed_address.tcp_host_header_value();
        stream.write_all(host_header.as_bytes())?;
        stream.write_all(b"\r\nConnection: close\r\nUser-Agent: Rust-Time-Sync\r\n\r\n")?;
        let mut stream_reader = BufReader::new(&stream);
        loop {
            net_ctx.tcp_line_buffer.clear();
            let bytes_read = stream_reader.read_until(b'\n', &mut net_ctx.tcp_line_buffer)?;
            if bytes_read == 0 {
                break Err(TimeError::header_not_found("Date (TCP)"));
            }
            if let Some(date_str) = find_date_header_value(&net_ctx.tcp_line_buffer) {
                let response_received_inst = Instant::now();
                let rtt_for_sample = response_received_inst.duration_since(request_start_inst);
                let server_time = parse_http_date_to_systemtime(date_str)?;
                break Ok(TimeSample {
                    response_received_inst,
                    rtt: rtt_for_sample,
                    server_time,
                });
            }
            if net_ctx.tcp_line_buffer == b"\r\n" {
                break Err(TimeError::header_not_found("Date (TCP)"));
            }
        }
    };
    tcp_attempt_result.or_else(|_| {
        if !*CURL_AVAILABLE {
            return Err(TimeError::sync_failed(
                "TCP 연결에 실패했고 curl을 사용할 수 없습니다.",
            ));
        }
        let mut last_error = None;
        for (scheme, context_str) in [
            (UrlScheme::Https, "HTTPS (fallback)"),
            (UrlScheme::Http, "HTTP (fallback)"),
        ] {
            let url = parsed_address.curl_url(scheme);
            match fetch_server_time_sample_curl(&url, context_str, net_ctx) {
                Ok(sample) => return Ok(sample),
                Err(err) => last_error = Some(err),
            }
        }
        Err(last_error
            .unwrap_or_else(|| TimeError::sync_failed("Curl 폴백 시도 중 알 수 없는 오류")))
    })
}
fn parse_u32_digits(raw: &str) -> Option<u32> {
    let mut value = 0_u32;
    if raw.is_empty() {
        return None;
    }
    for &byte in raw.as_bytes() {
        if !byte.is_ascii_digit() {
            return None;
        }
        let digit = byte.checked_sub(b'0')?;
        value = value.checked_mul(10)?.checked_add(u32::from(digit))?;
    }
    Some(value)
}
