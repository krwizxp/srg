mod address;
#[cfg(target_os = "windows")]
mod high_res_timer;
mod http_date;
#[cfg(target_os = "windows")]
mod windows_input;
use self::{
    address::{ParsedServerAddress, UrlScheme, parse_server_address},
    http_date::{civil_from_days, parse_http_date_to_systemtime},
};
use crate::{
    buffmt::{ByteCursor, DIGITS, TWO_DIGITS, write_zero_err},
    numeric::low_u8_from_u32,
};
use std::{
    borrow::Cow,
    error,
    fmt::{self, Write as _},
    io::{self, BufRead as _, BufReader, Result as ioResult, Write as _},
    mem,
    net::{self, TcpStream},
    process::{Command, Stdio},
    result as stdresult, str,
    sync::{LazyLock, mpsc},
    thread,
    time::{Duration, Instant, SystemTime, SystemTimeError, UNIX_EPOCH},
};
const FULL_SYNC_INTERVAL: Duration = Duration::from_mins(5);
const RETRY_DELAY: Duration = Duration::from_secs(10);
const TCP_TIMEOUT_SECS: u64 = 5;
const NUM_SAMPLES: usize = 10;
const FINAL_COUNTDOWN_RTT_ALPHA_NUM: u32 = 7;
const FINAL_COUNTDOWN_RTT_ALPHA_DENOM: u32 = 10;
const MAX_CALIBRATION_FAILURES: u32 = 100;
const KST_OFFSET_SECS_U64: u64 = 9 * 3600;
const KST_OFFSET_SECS: i64 = KST_OFFSET_SECS_U64.cast_signed();
const DAY_OF_WEEK_KO: [&str; 7] = ["일", "월", "화", "수", "목", "금", "토"];
const DISPLAY_INTERVAL: Duration = Duration::from_millis(16);
const ADAPTIVE_POLL_INTERVAL: Duration = Duration::from_millis(10);
const DISPLAY_UPDATE_INTERVAL: Duration = Duration::from_millis(45);
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TimeErrorKind {
    Io,
    Time,
    Parse,
    HeaderNotFound,
    Curl,
    SyncFailed,
}
#[derive(Debug)]
pub struct TimeError {
    kind: TimeErrorKind,
    detail: Cow<'static, str>,
    source: Option<Box<dyn error::Error + Send + Sync + 'static>>,
    io_kind: Option<io::ErrorKind>,
}
impl TimeError {
    fn new(kind: TimeErrorKind, detail: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind,
            detail: detail.into(),
            source: None,
            io_kind: None,
        }
    }

    fn parse(detail: impl Into<Cow<'static, str>>) -> Self {
        Self::new(TimeErrorKind::Parse, detail)
    }

    fn header_not_found(detail: impl Into<Cow<'static, str>>) -> Self {
        Self::new(TimeErrorKind::HeaderNotFound, detail)
    }

    fn curl(detail: impl Into<Cow<'static, str>>) -> Self {
        Self::new(TimeErrorKind::Curl, detail)
    }

    fn sync_failed(detail: impl Into<Cow<'static, str>>) -> Self {
        Self::new(TimeErrorKind::SyncFailed, detail)
    }

    pub const fn io_kind(&self) -> Option<io::ErrorKind> {
        self.io_kind
    }
}
impl From<io::Error> for TimeError {
    fn from(err: io::Error) -> Self {
        let io_kind = err.kind();
        Self {
            kind: TimeErrorKind::Io,
            detail: Cow::Owned(err.to_string()),
            source: Some(Box::new(err)),
            io_kind: Some(io_kind),
        }
    }
}
impl From<SystemTimeError> for TimeError {
    fn from(err: SystemTimeError) -> Self {
        Self {
            kind: TimeErrorKind::Time,
            detail: Cow::Owned(err.to_string()),
            source: Some(Box::new(err)),
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
impl error::Error for TimeError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| -> &(dyn error::Error + 'static) { source })
    }
}
type Result<T> = stdresult::Result<T, TimeError>;
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
#[derive(Clone, Copy, Debug)]
struct TimeSample {
    response_received_inst: Instant,
    rtt: Duration,
    server_time: SystemTime,
}
#[derive(Debug)]
struct ServerTime {
    anchor_time: SystemTime,
    anchor_instant: Instant,
    baseline_rtt: Duration,
}
struct DisplayableTime {
    year: i32,
    month: u32,
    day_of_month: u32,
    day_of_week_str: &'static str,
    hour: u32,
    minute: u32,
    second: u32,
    millis: u32,
}
struct SliceCursor<'buffer> {
    inner: ByteCursor<'buffer>,
}
impl<'buffer> SliceCursor<'buffer> {
    const fn new(buf: &'buffer mut [u8]) -> Self {
        Self {
            inner: ByteCursor::new(buf),
        }
    }
    const fn written_len(&self) -> usize {
        self.inner.written_len()
    }
    fn write_bytes(&mut self, bytes: &[u8]) -> ioResult<()> {
        self.inner.write_bytes(bytes)
    }
    fn write_byte(&mut self, b: u8) -> ioResult<()> {
        self.inner.write_byte(b)
    }
    fn digit_byte(index: usize) -> ioResult<u8> {
        DIGITS.get(index).copied().ok_or_else(write_zero_err)
    }
    fn two_digits(index: usize) -> ioResult<&'static [u8; 2]> {
        TWO_DIGITS.get(index).ok_or_else(write_zero_err)
    }
    fn write_u32_dec(&mut self, mut n: u32) -> ioResult<()> {
        let mut tmp = [0_u8; 10];
        let mut i = tmp.len();
        while n >= 100 {
            let rem = usize::from(low_u8_from_u32(n % 100));
            n /= 100;
            i -= 2;
            tmp.get_mut(i..i + 2)
                .ok_or_else(write_zero_err)?
                .copy_from_slice(Self::two_digits(rem)?);
        }
        if n >= 10 {
            let rem = usize::from(low_u8_from_u32(n));
            i -= 2;
            tmp.get_mut(i..i + 2)
                .ok_or_else(write_zero_err)?
                .copy_from_slice(Self::two_digits(rem)?);
        } else {
            i -= 1;
            let digit = low_u8_from_u32(n);
            *tmp.get_mut(i).ok_or_else(write_zero_err)? = b'0' + digit;
        }
        self.write_bytes(tmp.get(i..).ok_or_else(write_zero_err)?)
    }
    fn write_year_padded4(&mut self, year: i32) -> ioResult<()> {
        if year >= 0_i32 {
            let y = year.cast_unsigned();
            if y < 10_000 {
                let hi = usize::from(low_u8_from_u32(y / 100));
                let lo = usize::from(low_u8_from_u32(y % 100));
                let head = self.inner.take(4)?;
                let (hi_digits, lo_digits) = head.split_at_mut(2);
                hi_digits.copy_from_slice(Self::two_digits(hi)?);
                lo_digits.copy_from_slice(Self::two_digits(lo)?);
                return Ok(());
            }
            return self.write_u32_dec(y);
        }
        self.write_byte(b'-')?;
        let abs = if year == i32::MIN {
            i32::MAX.cast_unsigned() + 1
        } else {
            (-year).cast_unsigned()
        };
        if abs < 1000 {
            let hundreds = usize::from(low_u8_from_u32(abs / 100));
            let rem = usize::from(low_u8_from_u32(abs % 100));
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
    fn write_u32_2digits(&mut self, v: u32) -> ioResult<()> {
        let idx = usize::from(low_u8_from_u32(v));
        self.inner.take(2)?.copy_from_slice(Self::two_digits(idx)?);
        Ok(())
    }
    fn write_u32_3digits(&mut self, v: u32) -> ioResult<()> {
        let hundreds = usize::from(low_u8_from_u32(v / 100));
        let rem = usize::from(low_u8_from_u32(v % 100));
        let head = self.inner.take(3)?;
        let (hundreds_digit, remaining_digits) = head.split_at_mut(1);
        if let Some(slot) = hundreds_digit.first_mut() {
            *slot = Self::digit_byte(hundreds)?;
        }
        remaining_digits.copy_from_slice(Self::two_digits(rem)?);
        Ok(())
    }
}
impl ServerTime {
    fn from_tick_sample(sample: TimeSample, baseline_rtt: Duration) -> Result<Self> {
        let mut server_time_at_tick = sample.server_time;
        let since_epoch = server_time_at_tick.duration_since(UNIX_EPOCH)?;
        let nanos_to_subtract = since_epoch.subsec_nanos();
        server_time_at_tick -= Duration::from_nanos(u64::from(nanos_to_subtract));
        let one_way_delay = sample.rtt / 2;
        let anchor_time = server_time_at_tick;
        let anchor_instant = sample
            .response_received_inst
            .checked_sub(one_way_delay)
            .unwrap_or(sample.response_received_inst);
        Ok(Self {
            anchor_time,
            anchor_instant,
            baseline_rtt,
        })
    }
    const fn recalibrate_with_rtt(&self, new_rtt: Duration) -> Self {
        let old_rtt_nanos = self.baseline_rtt.as_nanos();
        let new_rtt_nanos = new_rtt.as_nanos();
        let smoothed_rtt_nanos = (old_rtt_nanos * 7 + new_rtt_nanos * 3) / 10;
        let smoothed_rtt = Duration::from_nanos_u128(smoothed_rtt_nanos);
        Self {
            anchor_time: self.anchor_time,
            anchor_instant: self.anchor_instant,
            baseline_rtt: smoothed_rtt,
        }
    }
    fn current_server_time(&self) -> SystemTime {
        let elapsed_since_anchor = self.anchor_instant.elapsed();
        self.anchor_time + elapsed_since_anchor
    }
    fn calculate_display_time(&self) -> Result<DisplayableTime> {
        let current_time = self.current_server_time();
        let since_epoch = current_time.duration_since(UNIX_EPOCH)?;
        let total_seconds_kst = since_epoch.as_secs().cast_signed() + KST_OFFSET_SECS;
        let millis = since_epoch.subsec_millis();
        let days_since_epoch = total_seconds_kst.div_euclid(86400);
        let day_of_week_num = (days_since_epoch + 4).rem_euclid(7);
        let day_of_week_idx =
            parse_result_with_context(usize::try_from(day_of_week_num), "요일 계산 중 범위 오류")?;
        let day_of_week_str = DAY_OF_WEEK_KO
            .get(day_of_week_idx)
            .copied()
            .ok_or_else(|| TimeError::parse("요일 계산 중 범위 오류"))?;
        let sec_of_day = total_seconds_kst.rem_euclid(86400);
        let hour =
            parse_result_with_context(u32::try_from(sec_of_day / 3600), "시 계산 중 범위 오류")?;
        let minute = parse_result_with_context(
            u32::try_from((sec_of_day % 3600) / 60),
            "분 계산 중 범위 오류",
        )?;
        let second =
            parse_result_with_context(u32::try_from(sec_of_day % 60), "초 계산 중 범위 오류")?;
        let day_index =
            parse_result_with_context(i32::try_from(days_since_epoch), "일자 계산 중 범위 오류")?;
        let (year, month, day_of_month) = civil_from_days(day_index);
        Ok(DisplayableTime {
            year,
            month,
            day_of_month,
            day_of_week_str,
            hour,
            minute,
            second,
            millis,
        })
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
enum TriggerAction {
    LeftClick,
    F5Press,
}
struct AppState {
    host: String,
    target_time: Option<SystemTime>,
    server_time: Option<ServerTime>,
    baseline_rtt: Option<Duration>,
    baseline_rtt_samples: [TimeSample; NUM_SAMPLES],
    baseline_rtt_attempts: usize,
    baseline_rtt_valid_count: usize,
    baseline_rtt_next_sample_at: Instant,
    next_full_sync_at: Instant,
    last_sample: Option<TimeSample>,
    trigger_action: Option<TriggerAction>,
    live_rtt: Option<Duration>,
    calibration_failure_count: u32,
    #[cfg(target_os = "windows")]
    high_res_timer_guard: Option<high_res_timer::HighResTimerGuard>,
}
#[derive(Clone, Debug)]
enum Activity {
    MeasureBaselineRtt,
    CalibrateOnTick,
    Predicting,
    FinalCountdown { target_time: SystemTime },
    Finished,
    Retrying { retry_at: Instant },
}
struct NetworkContext {
    tcp_line_buffer: Vec<u8>,
    curl_stderr_buf: String,
}
static CURL_AVAILABLE: LazyLock<bool> = LazyLock::new(|| is_command_available("curl"));
#[cfg(target_os = "linux")]
static XDO_TOOL_AVAILABLE: LazyLock<bool> = LazyLock::new(|| is_command_available("xdotool"));
fn has_ignored_address_suffix(raw_input: &str) -> bool {
    let input_bytes = raw_input.as_bytes();
    let after_scheme = if input_bytes
        .get(..8)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"https://"))
    {
        raw_input.get(8..).unwrap_or(raw_input)
    } else if input_bytes
        .get(..7)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"http://"))
    {
        raw_input.get(7..).unwrap_or(raw_input)
    } else {
        raw_input
    };
    after_scheme
        .bytes()
        .any(|byte| matches!(byte, b'/' | b'?' | b'#'))
}
impl AppState {
    fn new() -> Result<Self> {
        let mut user_input_buf = String::with_capacity(256);
        let host = get_validated_input(
            "확인할 서버 주소를 입력하세요 (예: www.example.com): ",
            &mut user_input_buf,
            |raw_input| {
                if raw_input.is_empty() {
                    Err("서버 주소를 비워둘 수 없습니다.")
                } else {
                    if has_ignored_address_suffix(raw_input) {
                        eprintln!(
                            "[안내] 서버 주소의 경로/쿼리/프래그먼트는 무시되고 호스트만 사용됩니다."
                        );
                    }
                    Ok(raw_input.to_owned())
                }
            },
        )?;
        let target_time = get_validated_input(
            "액션 실행 목표 시간을 입력하세요 (예: 20:00:00 / 건너뛰려면 Enter): ",
            &mut user_input_buf,
            |raw_input| {
                if raw_input.is_empty() {
                    return Ok(None);
                }
                let mut parts = raw_input.split(':');
                if let (Some(hour_str), Some(minute_str), Some(second_str)) =
                    (parts.next(), parts.next(), parts.next())
                    && parts.next().is_none()
                    && let (Ok(hour), Ok(minute), Ok(second)) = (
                        hour_str.parse::<u32>(),
                        minute_str.parse::<u32>(),
                        second_str.parse::<u32>(),
                    )
                    && hour <= 23
                    && minute <= 59
                    && second <= 59
                {
                    let now_local = SystemTime::now();
                    let since_epoch = match now_local.duration_since(UNIX_EPOCH) {
                        Ok(duration) => duration,
                        Err(_duration_err) => {
                            return Err("시간 계산 오류: 시스템 시간이 UNIX EPOCH보다 이전입니다.");
                        }
                    };
                    let today_start_secs_utc =
                        ((since_epoch.as_secs() + KST_OFFSET_SECS_U64) / 86400 * 86400)
                            - KST_OFFSET_SECS_U64;
                    let target_secs_of_day = u64::from(hour * 3600 + minute * 60 + second);
                    let mut target_time =
                        UNIX_EPOCH + Duration::from_secs(today_start_secs_utc + target_secs_of_day);
                    if now_local > target_time {
                        target_time += Duration::from_hours(24);
                    }
                    Ok(Some(target_time))
                } else {
                    Err("잘못된 형식, 숫자 또는 시간 범위입니다 (HH:MM:SS, 0-23:0-59:0-59).")
                }
            },
        )?;
        let trigger_action: Option<TriggerAction> = if target_time.is_some() {
            Some(get_validated_input(
                "수행할 동작을 선택하세요 (1: 마우스 왼쪽 클릭, 2: F5 입력): ",
                &mut user_input_buf,
                |s| match s {
                    "1" => Ok(TriggerAction::LeftClick),
                    "2" => Ok(TriggerAction::F5Press),
                    _ => Err("잘못된 입력입니다. 1 또는 2를 입력해주세요."),
                },
            )?)
        } else {
            None
        };
        let baseline_placeholder = TimeSample {
            response_received_inst: Instant::now(),
            rtt: Duration::ZERO,
            server_time: UNIX_EPOCH,
        };
        Ok(Self {
            host,
            target_time,
            trigger_action,
            server_time: None,
            baseline_rtt: None,
            baseline_rtt_samples: [baseline_placeholder; NUM_SAMPLES],
            baseline_rtt_attempts: 0,
            baseline_rtt_valid_count: 0,
            baseline_rtt_next_sample_at: Instant::now(),
            next_full_sync_at: Instant::now(),
            last_sample: None,
            live_rtt: None,
            calibration_failure_count: 0,
            #[cfg(target_os = "windows")]
            high_res_timer_guard: None,
        })
    }
    #[cfg(target_os = "windows")]
    fn sync_high_res_timer_state(&mut self, next_activity: &Activity) {
        if matches!(next_activity, Activity::FinalCountdown { .. }) {
            if self.high_res_timer_guard.is_none() {
                match high_res_timer::HighResTimerGuard::new() {
                    Ok(guard) => {
                        self.high_res_timer_guard = Some(guard);
                    }
                    Err(e) => {
                        eprintln!("[경고] {e}. 카운트다운 정확도가 저하될 수 있습니다.");
                    }
                }
            }
        } else {
            self.high_res_timer_guard = None;
        }
    }
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    const fn sync_high_res_timer_state(_next_activity: &Activity) {}
    fn run_loop(&mut self) -> Result<()> {
        println!("\n서버 시간 확인을 시작합니다... (Enter를 누르면 종료)");
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || -> ioResult<()> {
            let mut line = String::new();
            io::stdin().read_line(&mut line)?;
            let _send_result = tx.send(());
            Ok(())
        });
        let mut activity = Activity::MeasureBaselineRtt;
        let mut last_display_update = Instant::now();
        let mut message_buffer = String::with_capacity(256);
        let mut network_context = NetworkContext {
            tcp_line_buffer: Vec::with_capacity(256),
            curl_stderr_buf: String::with_capacity(1024),
        };
        let stdout = io::stdout();
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
                let mut line_buf = [0_u8; 80];
                let mut cur = SliceCursor::new(&mut line_buf);
                match (|| -> Result<()> {
                    cur.write_bytes("\r서버 시간: ".as_bytes())
                        .map_err(TimeError::from)?;
                    st.write_current_display_time_buf(&mut cur, true)?;
                    cur.write_bytes(b" \r").map_err(TimeError::from)?;
                    Ok(())
                })() {
                    Ok(()) => {
                        let used = cur.written_len();
                        let mut out = stdout.lock();
                        out.write_all(
                            line_buf
                                .get(..used)
                                .ok_or_else(|| io::Error::other("표시 버퍼 사용 길이 계산 실패"))?,
                        )?;
                        out.flush()?;
                    }
                    Err(e) => {
                        let mut out = stdout.lock();
                        out.write_all("\r서버 시간: ".as_bytes())?;
                        write!(out, "시간 표시 오류: {e}")?;
                        out.write_all(b" \r")?;
                        out.flush()?;
                    }
                }
                last_display_update = now;
            }
            message_buffer.clear();
            let (next_activity, log_opt_msg) = match activity {
                Activity::MeasureBaselineRtt => {
                    self.handle_measure_baseline_rtt(&mut message_buffer, &mut network_context)
                }
                Activity::CalibrateOnTick => {
                    self.handle_calibrate_on_tick(&mut message_buffer, &mut network_context)
                }
                Activity::Predicting => self.handle_predicting(&mut message_buffer),
                Activity::FinalCountdown { target_time } => self.handle_final_countdown(
                    target_time,
                    &mut message_buffer,
                    &mut network_context,
                ),
                Activity::Finished => Self::handle_finished(),
                Activity::Retrying { retry_at } => Self::handle_retrying(retry_at),
            };
            #[cfg(target_os = "windows")]
            self.sync_high_res_timer_state(&next_activity);
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            Self::sync_high_res_timer_state(&next_activity);
            if let Some(console_msg) = log_opt_msg {
                println!("\n{console_msg}");
            }
            activity = next_activity;
        }
        Ok(())
    }
    fn handle_measure_baseline_rtt<'message>(
        &mut self,
        msg_buf: &'message mut String,
        net_ctx: &mut NetworkContext,
    ) -> (Activity, Option<&'message str>) {
        let now = Instant::now();
        if self.baseline_rtt_attempts == 0 {
            if self.last_sample.is_none() {
                println!("1단계: RTT 기준값 측정을 시작합니다...");
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
            Err(e) => {
                self.baseline_rtt_attempts = 0;
                self.baseline_rtt_valid_count = 0;
                let _write_result = write!(msg_buf, "RTT 샘플 수집 실패: {e}");
                return transition_to_retry(msg_buf);
            }
        }
        let Some(next_attempt_index) = attempt_index.checked_add(1) else {
            return transition_to_retry("RTT 시도 횟수 계산 실패.");
        };
        self.baseline_rtt_attempts = next_attempt_index;
        self.baseline_rtt_next_sample_at = Instant::now() + ADAPTIVE_POLL_INTERVAL;
        if self.baseline_rtt_attempts < NUM_SAMPLES {
            return (Activity::MeasureBaselineRtt, None);
        }
        let sample_count = self.baseline_rtt_valid_count;
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
        let trim = filled / 5;
        let Some(window_end) = filled.checked_sub(trim) else {
            return transition_to_retry("RTT 샘플 윈도우 계산 실패.");
        };
        let Some(window) = rtts.get(trim..window_end) else {
            return transition_to_retry("RTT 샘플 윈도우 계산 실패.");
        };
        let sum_nanos: u128 = window.iter().sum();
        let window_len = u128::try_from(window.len()).unwrap_or_default();
        let baseline_rtt = Duration::from_nanos_u128(sum_nanos / window_len);
        self.baseline_rtt = Some(baseline_rtt);
        self.calibration_failure_count = 0;
        let _write_result = write!(
            msg_buf,
            "[완료] RTT 기준값: {rtt_ms}ms. 2단계: 정밀 보정을 시작합니다.",
            rtt_ms = baseline_rtt.as_millis()
        );
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
            self.calibration_failure_count += 1;
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
            && let Ok(st) = ServerTime::from_tick_sample(current_sample, baseline_rtt)
        {
            self.server_time = Some(st);
            self.next_full_sync_at = Instant::now() + FULL_SYNC_INTERVAL;
            return (Activity::Predicting, Some("[성공] 정밀 보정 완료!"));
        }
        self.last_sample = Some(current_sample);
        (Activity::CalibrateOnTick, None)
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
                .is_ok_and(|d| d <= Duration::from_secs(10))
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
    fn trigger_and_finish<'message>(
        &self,
        msg_buf: &'message mut String,
        log_message: fmt::Arguments,
    ) -> (Activity, Option<&'message str>) {
        if let Some(action) = self.trigger_action {
            trigger_action(action);
        }
        let _format_result = msg_buf.write_fmt(log_message);
        (Activity::Finished, Some(msg_buf))
    }
    fn handle_final_countdown<'message>(
        &mut self,
        target_time: SystemTime,
        msg_buf: &'message mut String,
        net_ctx: &mut NetworkContext,
    ) -> (Activity, Option<&'message str>) {
        let sample = match fetch_server_time_sample(&self.host, net_ctx) {
            Ok(s) => s,
            Err(e) => {
                if let Some(st) = self.server_time.as_ref() {
                    let current_server_time = st.current_server_time();
                    let effective_rtt = self.live_rtt.unwrap_or(st.baseline_rtt);
                    let one_way_delay = effective_rtt / 2;
                    match target_time.duration_since(current_server_time) {
                        Ok(duration_until_target) if duration_until_target <= one_way_delay => {
                            return self.trigger_and_finish(
                                msg_buf,
                                format_args!(
                                    "\n>>> 액션 실행! (예측값 기준 강제 실행, 목표 도달까지 {:.1}ms 남음) (지연 예측: {:.1}ms, 원인: 카운트다운 샘플 실패: {e})",
                                    duration_until_target.as_secs_f64() * 1000.0,
                                    one_way_delay.as_secs_f64() * 1000.0
                                ),
                            );
                        }
                        Err(_) => {
                            return self.trigger_and_finish(
                                msg_buf,
                                format_args!(
                                    "\n>>> 액션 실행! (예측값 기준 강제 실행, 시간 초과) (지연 예측: {:.1}ms, 원인: 카운트다운 샘플 실패: {e})",
                                    one_way_delay.as_secs_f64() * 1000.0
                                ),
                            );
                        }
                        _ => {}
                    }
                }
                let _write_result = write!(msg_buf, "카운트다운 샘플 획득 실패: {e}");
                return (Activity::FinalCountdown { target_time }, Some(msg_buf));
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
        let new_rtt_nanos = (old_rtt.as_nanos()
            * u128::from(FINAL_COUNTDOWN_RTT_ALPHA_DENOM - FINAL_COUNTDOWN_RTT_ALPHA_NUM)
            + sample.rtt.as_nanos() * u128::from(FINAL_COUNTDOWN_RTT_ALPHA_NUM))
            / u128::from(FINAL_COUNTDOWN_RTT_ALPHA_DENOM);
        let live_rtt = Duration::from_nanos_u128(new_rtt_nanos);
        self.live_rtt = Some(live_rtt);
        let effective_rtt = live_rtt.max(sample.rtt);
        let one_way_delay = effective_rtt / 2;
        match target_time.duration_since(current_server_time) {
            Ok(duration_until_target) if duration_until_target <= one_way_delay => {
                self.trigger_and_finish(
     msg_buf,
     format_args!(
         "\n>>> 액션 실행! (목표 도달까지 {:.1}ms 남음) (지연 예측: {:.1}ms, 실측 RTT: {:.1}ms)",
         duration_until_target.as_secs_f64() * 1000.0,
         one_way_delay.as_secs_f64() * 1000.0,
         sample.rtt.as_secs_f64() * 1000.0
     )
)
            }
            Err(_) => {
                self.trigger_and_finish(
     msg_buf,
     format_args!(
         "\n>>> 액션 실행! (시간 초과) (지연 예측: {:.1}ms, 실측 RTT: {:.1}ms)",
         one_way_delay.as_secs_f64() * 1000.0,
         sample.rtt.as_secs_f64() * 1000.0
     )
)
            }
            _ => (Activity::FinalCountdown { target_time }, None),
        }
    }
    const fn handle_finished() -> (Activity, Option<&'static str>) {
        (Activity::Predicting, Some("액션 완료. 예측 모드 전환."))
    }
    fn handle_retrying(retry_at: Instant) -> (Activity, Option<&'static str>) {
        let now = Instant::now();
        if now >= retry_at {
            (
                Activity::MeasureBaselineRtt,
                Some("[재시도] 동기화를 다시 시작합니다."),
            )
        } else {
            (Activity::Retrying { retry_at }, None)
        }
    }
}
fn get_validated_input<T, F>(prompt: &str, input_buf: &mut String, validator: F) -> ioResult<T>
where
    F: Fn(&str) -> stdresult::Result<T, &'static str>,
{
    loop {
        print!("{prompt}");
        io::stdout().flush()?;
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
            Err(e) => {
                if !e.is_empty() {
                    println!("{e}");
                }
            }
        }
    }
}
pub fn run() -> Result<()> {
    #[cfg(target_os = "windows")]
    if !*CURL_AVAILABLE {
        eprintln!(
            "[경고] 'curl' 명령어를 찾을 수 없습니다. TCP 연결 실패 시 대체 수단이 없습니다."
        );
    }
    #[cfg(target_os = "linux")]
    if !*XDO_TOOL_AVAILABLE {
        eprintln!(
            "[경고] 'xdotool'이 설치되지 않았습니다. 액션 기능이 동작하지 않습니다.\n(설치 방법: sudo apt-get install xdotool 또는 유사한 패키지 관리자 명령어)"
        );
    }
    let mut app_state = AppState::new()?;
    app_state.run_loop()?;
    println!("\n프로그램을 종료합니다.");
    Ok(())
}
fn transition_to_retry(msg: &str) -> (Activity, Option<&str>) {
    (
        Activity::Retrying {
            retry_at: Instant::now() + RETRY_DELAY,
        },
        Some(msg),
    )
}
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn run_external_command(program: &str, args: &[&str]) {
    match Command::new(program).args(args).status() {
        Ok(status) if status.success() => {}
        Ok(status) => {
            eprintln!("[경고] 외부 명령 실행 실패: {program} {args:?} (상태: {status})");
        }
        Err(e) => {
            eprintln!("[경고] 외부 명령 실행 실패: {program} {args:?} ({e})");
        }
    }
}
fn trigger_action(action: TriggerAction) {
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
            windows_input::send_mouse_click();
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
            windows_input::send_f5_press();
        }
    }
}
fn is_command_available(command: &str) -> bool {
    Command::new(command)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
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
            let _write_result = write!(
                &mut net_ctx.curl_stderr_buf,
                "curl {context} 실패, 상태: {status}",
                status = output.status
            );
        }
        return Err(TimeError::curl(mem::take(&mut net_ctx.curl_stderr_buf)));
    }
    let mut end = stdout_bytes.len();
    while end > 0
        && stdout_bytes
            .get(end - 1)
            .is_some_and(u8::is_ascii_whitespace)
    {
        end -= 1;
    }
    let trimmed_stdout = stdout_bytes
        .get(..end)
        .ok_or_else(|| TimeError::parse("curl 응답 경계 계산 실패"))?;
    let pos = trimmed_stdout
        .iter()
        .rposition(|&b| b == b'\n')
        .ok_or_else(|| TimeError::parse("curl 응답에서 time_starttransfer 정보 누락"))?;
    let headers_part = trimmed_stdout
        .get(..pos)
        .ok_or_else(|| TimeError::parse("curl 응답 헤더 범위 계산 실패"))?;
    let time_starttransfer_part = trimmed_stdout
        .get(pos + 1..)
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
        .split(|&b| b == b'\n')
        .rev()
        .find_map(|line| find_date_header_value(line))
        .ok_or_else(|| TimeError::header_not_found("curl 응답에서 Date 헤더를 찾을 수 없음"))?;
    let server_time = parse_http_date_to_systemtime(date_header_str_slice)?;
    let response_received_inst = time_before_curl_call_inst + rtt_reported_by_curl;
    Ok(TimeSample {
        response_received_inst,
        rtt: rtt_reported_by_curl,
        server_time,
    })
}
fn tcp_attempt(line_buffer: &mut Vec<u8>, address: &ParsedServerAddress) -> Result<TimeSample> {
    let request_start_inst = Instant::now();
    let tcp_timeout = Duration::from_secs(TCP_TIMEOUT_SECS);
    let socket_addr_result: ioResult<net::SocketAddr> =
        if let Ok(ip_addr) = address.host().parse::<net::IpAddr>() {
            Ok(net::SocketAddr::new(ip_addr, address.port()))
        } else {
            net::ToSocketAddrs::to_socket_addrs(&(address.host(), address.port()))?
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Host not found"))
        };
    let socket_addr = socket_addr_result?;
    let mut stream = TcpStream::connect_timeout(&socket_addr, tcp_timeout)?;
    stream.set_read_timeout(Some(tcp_timeout))?;
    stream.set_write_timeout(Some(tcp_timeout))?;
    stream.write_all(b"HEAD / HTTP/1.1\r\nHost: ")?;
    let host_header = address.tcp_host_header_value();
    stream.write_all(host_header.as_bytes())?;
    stream.write_all(b"\r\nConnection: close\r\nUser-Agent: Rust-Time-Sync\r\n\r\n")?;
    let mut stream_reader = BufReader::new(&stream);
    loop {
        line_buffer.clear();
        let bytes_read = stream_reader.read_until(b'\n', line_buffer)?;
        if bytes_read == 0 {
            break;
        }
        if let Some(date_str) = find_date_header_value(line_buffer) {
            let response_received_inst = Instant::now();
            let rtt_for_sample = response_received_inst.duration_since(request_start_inst);
            let server_time = parse_http_date_to_systemtime(date_str)?;
            return Ok(TimeSample {
                response_received_inst,
                rtt: rtt_for_sample,
                server_time,
            });
        }
        if line_buffer == b"\r\n" {
            break;
        }
    }
    Err(TimeError::header_not_found("Date (TCP)"))
}
fn fetch_server_time_sample(host: &str, net_ctx: &mut NetworkContext) -> Result<TimeSample> {
    let parsed_address = parse_server_address(host)?;
    if parsed_address.scheme() == Some(UrlScheme::Https) {
        let https_url = parsed_address.curl_url(UrlScheme::Https);
        return fetch_server_time_sample_curl(&https_url, "HTTPS (explicit)", net_ctx);
    }
    tcp_attempt(&mut net_ctx.tcp_line_buffer, &parsed_address).or_else(|_| {
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
                Err(e) => last_error = Some(e),
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
        value = value.checked_mul(10)?.checked_add(u32::from(byte - b'0'))?;
    }
    Some(value)
}
