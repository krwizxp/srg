use crate::numeric::low_u8_from_u32;
use std::{
    borrow::Cow,
    error,
    fmt::{self, Write},
    io::{self, BufRead, BufReader, Result as ioResult, Write as ioWrite},
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
const DIGITS: [u8; 10] = *b"0123456789";
const fn make_two_digits_table() -> [[u8; 2]; 100] {
    let mut table = [[0u8; 2]; 100];
    let mut idx = 0usize;
    let mut value = 0u8;
    while idx < 100 {
        table[idx] = [b'0' + value / 10, b'0' + value % 10];
        idx += 1;
        value += 1;
    }
    table
}
const TWO_DIGITS: [[u8; 2]; 100] = make_two_digits_table();
const DISPLAY_INTERVAL: Duration = Duration::from_millis(16);
const ADAPTIVE_POLL_INTERVAL: Duration = Duration::from_millis(10);
const DISPLAY_UPDATE_INTERVAL: Duration = Duration::from_millis(45);
#[cfg(target_os = "windows")]
mod high_res_timer;
#[cfg(target_os = "windows")]
mod windows_input;
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Time(SystemTimeError),
    Parse(Cow<'static, str>),
    HeaderNotFound(Cow<'static, str>),
    Curl(String),
    SyncFailed(Cow<'static, str>),
}
impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}
impl From<SystemTimeError> for Error {
    fn from(err: SystemTimeError) -> Self {
        Self::Time(err)
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O 오류: {e}"),
            Self::Time(e) => write!(f, "시스템 시간 오류: {e}"),
            Self::Parse(msg) => write!(f, "파싱 오류: {msg}"),
            Self::HeaderNotFound(header) => write!(f, "{header} 헤더를 찾을 수 없음"),
            Self::Curl(stderr) => write!(f, "curl 실행 실패: {stderr}"),
            Self::SyncFailed(msg) => write!(f, "서버 시간 확인 실패: {msg}"),
        }
    }
}
impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Time(e) => Some(e),
            _ => None,
        }
    }
}
pub type Result<T> = stdresult::Result<T, Error>;
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
#[inline(never)]
#[cold]
fn write_zero_err() -> io::Error {
    io::Error::new(io::ErrorKind::WriteZero, "failed to write whole buffer")
}
struct SliceCursor<'a> {
    buf: &'a mut [u8],
    pos: usize,
}
impl<'a> SliceCursor<'a> {
    const fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    const fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
    const fn written_len(&self) -> usize {
        self.pos
    }
    fn write_bytes(&mut self, bytes: &[u8]) -> ioResult<()> {
        let len = bytes.len();
        if self.remaining() < len {
            return Err(write_zero_err());
        }
        let end = self.pos + len;
        self.buf[self.pos..end].copy_from_slice(bytes);
        self.pos = end;
        Ok(())
    }
    fn write_byte(&mut self, b: u8) -> ioResult<()> {
        if self.remaining() < 1 {
            return Err(write_zero_err());
        }
        self.buf[self.pos] = b;
        self.pos += 1;
        Ok(())
    }
    fn write_u32_dec(&mut self, mut n: u32) -> ioResult<()> {
        let mut tmp = [0u8; 10];
        let mut i = tmp.len();
        while n >= 100 {
            let rem = usize::from(low_u8_from_u32(n % 100));
            n /= 100;
            i -= 2;
            tmp[i..i + 2].copy_from_slice(&TWO_DIGITS[rem]);
        }
        if n >= 10 {
            let rem = usize::from(low_u8_from_u32(n));
            i -= 2;
            tmp[i..i + 2].copy_from_slice(&TWO_DIGITS[rem]);
        } else {
            i -= 1;
            let digit = low_u8_from_u32(n);
            tmp[i] = b'0' + digit;
        }
        self.write_bytes(&tmp[i..])
    }
    fn write_year_padded4(&mut self, year: i32) -> ioResult<()> {
        if year >= 0 {
            let y = year.cast_unsigned();
            if y < 10_000 {
                if self.remaining() < 4 {
                    return Err(write_zero_err());
                }
                let hi = usize::from(low_u8_from_u32(y / 100));
                let lo = usize::from(low_u8_from_u32(y % 100));
                let start = self.pos;
                self.buf[start..start + 2].copy_from_slice(&TWO_DIGITS[hi]);
                self.buf[start + 2..start + 4].copy_from_slice(&TWO_DIGITS[lo]);
                self.pos = start + 4;
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
            if self.remaining() < 3 {
                return Err(write_zero_err());
            }
            let hundreds = usize::from(low_u8_from_u32(abs / 100));
            let rem = usize::from(low_u8_from_u32(abs % 100));
            let start = self.pos;
            self.buf[start] = DIGITS[hundreds];
            self.buf[start + 1..start + 3].copy_from_slice(&TWO_DIGITS[rem]);
            self.pos = start + 3;
            return Ok(());
        }
        self.write_u32_dec(abs)
    }
    fn write_u32_2digits(&mut self, v: u32) -> ioResult<()> {
        if self.remaining() < 2 {
            return Err(write_zero_err());
        }
        let start = self.pos;
        let idx = usize::from(low_u8_from_u32(v));
        self.buf[start..start + 2].copy_from_slice(&TWO_DIGITS[idx]);
        self.pos = start + 2;
        Ok(())
    }
    fn write_u32_3digits(&mut self, v: u32) -> ioResult<()> {
        if self.remaining() < 3 {
            return Err(write_zero_err());
        }
        let hundreds = usize::from(low_u8_from_u32(v / 100));
        let rem = usize::from(low_u8_from_u32(v % 100));
        let start = self.pos;
        self.buf[start] = DIGITS[hundreds];
        self.buf[start + 1..start + 3].copy_from_slice(&TWO_DIGITS[rem]);
        self.pos = start + 3;
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
        let day_of_week_idx = usize::try_from(day_of_week_num)
            .map_err(|_| Error::Parse(Cow::Borrowed("요일 계산 중 범위 오류")))?;
        let day_of_week_str = DAY_OF_WEEK_KO[day_of_week_idx];
        let sec_of_day = total_seconds_kst.rem_euclid(86400);
        let hour = u32::try_from(sec_of_day / 3600)
            .map_err(|_| Error::Parse(Cow::Borrowed("시 계산 중 범위 오류")))?;
        let minute = u32::try_from((sec_of_day % 3600) / 60)
            .map_err(|_| Error::Parse(Cow::Borrowed("분 계산 중 범위 오류")))?;
        let second = u32::try_from(sec_of_day % 60)
            .map_err(|_| Error::Parse(Cow::Borrowed("초 계산 중 범위 오류")))?;
        let day_index = i32::try_from(days_since_epoch)
            .map_err(|_| Error::Parse(Cow::Borrowed("일자 계산 중 범위 오류")))?;
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
        cur.write_year_padded4(dt.year).map_err(Error::Io)?;
        cur.write_byte(b'-').map_err(Error::Io)?;
        cur.write_u32_2digits(dt.month).map_err(Error::Io)?;
        cur.write_byte(b'-').map_err(Error::Io)?;
        cur.write_u32_2digits(dt.day_of_month).map_err(Error::Io)?;
        cur.write_byte(b'(').map_err(Error::Io)?;
        cur.write_bytes(dt.day_of_week_str.as_bytes())
            .map_err(Error::Io)?;
        cur.write_bytes(b") ").map_err(Error::Io)?;
        cur.write_u32_2digits(dt.hour).map_err(Error::Io)?;
        cur.write_byte(b':').map_err(Error::Io)?;
        cur.write_u32_2digits(dt.minute).map_err(Error::Io)?;
        cur.write_byte(b':').map_err(Error::Io)?;
        cur.write_u32_2digits(dt.second).map_err(Error::Io)?;
        if show_millis {
            cur.write_byte(b'.').map_err(Error::Io)?;
            cur.write_u32_3digits(dt.millis).map_err(Error::Io)?;
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
        &raw_input[8..]
    } else if input_bytes
        .get(..7)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"http://"))
    {
        &raw_input[7..]
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
            |s| {
                if s.is_empty() {
                    Err("서버 주소를 비워둘 수 없습니다.")
                } else {
                    if has_ignored_address_suffix(s) {
                        eprintln!(
                            "[안내] 서버 주소의 경로/쿼리/프래그먼트는 무시되고 호스트만 사용됩니다."
                        );
                    }
                    Ok(s.to_string())
                }
            },
        )?;
        let target_time = get_validated_input(
            "액션 실행 목표 시간을 입력하세요 (예: 20:00:00 / 건너뛰려면 Enter): ",
            &mut user_input_buf,
            |s| {
                if s.is_empty() {
                    return Ok(None);
                }
                let mut parts = s.split(':');
                if let (Some(h_str), Some(m_str), Some(s_str)) =
                    (parts.next(), parts.next(), parts.next())
                    && parts.next().is_none()
                    && let (Ok(h), Ok(m), Ok(s)) = (
                        h_str.parse::<u32>(),
                        m_str.parse::<u32>(),
                        s_str.parse::<u32>(),
                    )
                    && h <= 23
                    && m <= 59
                    && s <= 59
                {
                    let now_local = SystemTime::now();
                    let since_epoch = now_local
                        .duration_since(UNIX_EPOCH)
                        .map_err(|_| "시간 계산 오류: 시스템 시간이 UNIX EPOCH보다 이전입니다.")?;
                    let today_start_secs_utc =
                        ((since_epoch.as_secs() + KST_OFFSET_SECS_U64) / 86400 * 86400)
                            - KST_OFFSET_SECS_U64;
                    let target_secs_of_day = u64::from(h * 3600 + m * 60 + s);
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
    #[cfg(not(target_os = "windows"))]
    const fn sync_high_res_timer_state(_next_activity: &Activity) {}
    fn run_loop(&mut self) -> Result<()> {
        println!("\n서버 시간 확인을 시작합니다... (Enter를 누르면 종료)");
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || -> ioResult<()> {
            let mut line = String::new();
            io::stdin().read_line(&mut line)?;
            let _ = tx.send(());
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
                _ => DISPLAY_UPDATE_INTERVAL,
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
            if now.duration_since(last_display_update) >= DISPLAY_INTERVAL {
                if let Some(st) = &self.server_time {
                    let mut line_buf = [0u8; 80];
                    let mut cur = SliceCursor::new(&mut line_buf);
                    match (|| -> Result<()> {
                        cur.write_bytes("\r서버 시간: ".as_bytes())
                            .map_err(Error::Io)?;
                        st.write_current_display_time_buf(&mut cur, true)?;
                        cur.write_bytes(b" \r").map_err(Error::Io)?;
                        Ok(())
                    })() {
                        Ok(()) => {
                            let used = cur.written_len();
                            let mut out = stdout.lock();
                            out.write_all(&line_buf[..used])?;
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
            #[cfg(not(target_os = "windows"))]
            Self::sync_high_res_timer_state(&next_activity);
            if let Some(console_msg) = log_opt_msg {
                println!("\n{console_msg}");
            }
            activity = next_activity;
        }
        Ok(())
    }
    fn handle_measure_baseline_rtt<'a>(
        &mut self,
        msg_buf: &'a mut String,
        net_ctx: &mut NetworkContext,
    ) -> (Activity, Option<&'a str>) {
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
                self.baseline_rtt_samples[attempt_index] = sample;
                self.baseline_rtt_valid_count += 1;
                self.last_sample = Some(sample);
            }
            Ok(_) => {}
            Err(e) => {
                self.baseline_rtt_attempts = 0;
                self.baseline_rtt_valid_count = 0;
                let _ = write!(msg_buf, "RTT 샘플 수집 실패: {e}");
                return transition_to_retry(msg_buf);
            }
        }
        self.baseline_rtt_attempts = attempt_index + 1;
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
        let mut rtt_nanos = [0u128; NUM_SAMPLES];
        let mut filled = 0usize;
        for sample in &self.baseline_rtt_samples {
            if sample.rtt > Duration::ZERO {
                rtt_nanos[filled] = sample.rtt.as_nanos();
                filled += 1;
                if filled >= sample_count {
                    break;
                }
            }
        }
        let rtts = &mut rtt_nanos[..filled];
        rtts.sort_unstable();
        let trim = filled / 5;
        let window = &rtts[trim..(filled - trim)];
        let sum_nanos: u128 = window.iter().sum();
        let window_len = window.len() as u128;
        let baseline_rtt = Duration::from_nanos_u128(sum_nanos / window_len);
        self.baseline_rtt = Some(baseline_rtt);
        self.calibration_failure_count = 0;
        let _ = write!(
            msg_buf,
            "[완료] RTT 기준값: {rtt_ms}ms. 2단계: 정밀 보정을 시작합니다.",
            rtt_ms = baseline_rtt.as_millis()
        );
        (Activity::CalibrateOnTick, Some(msg_buf))
    }
    fn handle_calibrate_on_tick<'a>(
        &mut self,
        _msg_buf: &'a mut str,
        net_ctx: &mut NetworkContext,
    ) -> (Activity, Option<&'a str>) {
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
    fn handle_predicting<'a>(&mut self, _msg_buf: &'a mut String) -> (Activity, Option<&'a str>) {
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
    fn trigger_and_finish<'a>(
        &self,
        msg_buf: &'a mut String,
        log_message: fmt::Arguments,
    ) -> (Activity, Option<&'a str>) {
        if let Some(action) = self.trigger_action {
            trigger_action(action);
        }
        let _ = msg_buf.write_fmt(log_message);
        (Activity::Finished, Some(msg_buf))
    }
    fn handle_final_countdown<'a>(
        &mut self,
        target_time: SystemTime,
        msg_buf: &'a mut String,
        net_ctx: &mut NetworkContext,
    ) -> (Activity, Option<&'a str>) {
        let sample = match fetch_server_time_sample(&self.host, net_ctx) {
            Ok(s) => s,
            Err(e) => {
                let _ = write!(msg_buf, "카운트다운 샘플 획득 실패: {e}");
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
        io::stdin().read_line(input_buf)?;
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
#[cfg(not(target_os = "windows"))]
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
    if line.len() > 5 && line[..5].eq_ignore_ascii_case(b"date:") {
        str::from_utf8(&line[5..]).ok().map(str::trim_ascii)
    } else {
        None
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UrlScheme {
    Http,
    Https,
}
impl UrlScheme {
    const fn default_port(self) -> u16 {
        match self {
            Self::Http => 80,
            Self::Https => 443,
        }
    }
    const fn prefix(self) -> &'static str {
        match self {
            Self::Http => "http://",
            Self::Https => "https://",
        }
    }
}
#[derive(Debug)]
struct ParsedServerAddress {
    scheme: Option<UrlScheme>,
    host: String,
    host_for_header: String,
    port: u16,
    explicit_port: bool,
}
impl ParsedServerAddress {
    fn curl_url(&self, scheme: UrlScheme) -> String {
        let mut url = String::with_capacity(self.host_for_header.len() + 16);
        url.push_str(scheme.prefix());
        url.push_str(&self.host_for_header);
        if self.explicit_port {
            url.push(':');
            url.push_str(&self.port.to_string());
        }
        url
    }
    fn tcp_host_header_value(&self) -> String {
        if self.port == UrlScheme::Http.default_port() {
            return self.host_for_header.clone();
        }
        let mut host_header = String::with_capacity(self.host_for_header.len() + 8);
        host_header.push_str(&self.host_for_header);
        host_header.push(':');
        host_header.push_str(&self.port.to_string());
        host_header
    }
}
fn parse_port(port_str: &str) -> Result<u16> {
    const ERR_PORT: &str = "서버 주소 파싱 실패: 포트 번호가 유효하지 않습니다 (1~65535).";
    if port_str.is_empty() {
        return Err(Error::Parse(Cow::Borrowed(ERR_PORT)));
    }
    let port_num = parse_u32_digits(port_str).ok_or(Error::Parse(Cow::Borrowed(ERR_PORT)))?;
    let port = u16::try_from(port_num).map_err(|_| Error::Parse(Cow::Borrowed(ERR_PORT)))?;
    if port == 0 {
        return Err(Error::Parse(Cow::Borrowed(ERR_PORT)));
    }
    Ok(port)
}
fn parse_authority_host_port(authority: &str, default_port: u16) -> Result<(String, u16, bool)> {
    const ERR_HOST: &str = "서버 주소 파싱 실패: 호스트 값이 비어있거나 형식이 올바르지 않습니다.";
    if let Some(bracketed) = authority.strip_prefix('[') {
        let close_idx = bracketed
            .find(']')
            .ok_or(Error::Parse(Cow::Borrowed(ERR_HOST)))?;
        let host_part = &bracketed[..close_idx];
        if host_part.is_empty() {
            return Err(Error::Parse(Cow::Borrowed(ERR_HOST)));
        }
        let rem = &bracketed[close_idx + 1..];
        if rem.is_empty() {
            return Ok((host_part.to_string(), default_port, false));
        }
        let port_part = rem
            .strip_prefix(':')
            .ok_or(Error::Parse(Cow::Borrowed(ERR_HOST)))?;
        let port = parse_port(port_part)?;
        return Ok((host_part.to_string(), port, true));
    }
    let colon_count = authority.bytes().filter(|&b| b == b':').count();
    if colon_count == 1 {
        let (host_part, port_part) = authority
            .rsplit_once(':')
            .ok_or(Error::Parse(Cow::Borrowed(ERR_HOST)))?;
        if host_part.is_empty() {
            return Err(Error::Parse(Cow::Borrowed(ERR_HOST)));
        }
        let port = parse_port(port_part)?;
        return Ok((host_part.to_string(), port, true));
    }
    if authority.is_empty() {
        return Err(Error::Parse(Cow::Borrowed(ERR_HOST)));
    }
    Ok((authority.to_string(), default_port, false))
}
fn parse_server_address(raw_input: &str) -> Result<ParsedServerAddress> {
    const ERR_EMPTY: &str = "서버 주소를 비워둘 수 없습니다.";
    const ERR_HOST: &str = "서버 주소 파싱 실패: 호스트 값이 비어있거나 형식이 올바르지 않습니다.";
    let input = raw_input.trim();
    if input.is_empty() {
        return Err(Error::Parse(Cow::Borrowed(ERR_EMPTY)));
    }
    let input_bytes = input.as_bytes();
    let (scheme, after_scheme) = if input_bytes
        .get(..8)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"https://"))
    {
        (Some(UrlScheme::Https), &input[8..])
    } else if input_bytes
        .get(..7)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"http://"))
    {
        (Some(UrlScheme::Http), &input[7..])
    } else {
        (None, input)
    };
    let authority_end = after_scheme
        .bytes()
        .position(|b| matches!(b, b'/' | b'?' | b'#'))
        .unwrap_or(after_scheme.len());
    let authority = &after_scheme[..authority_end];
    if authority.is_empty() || authority.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return Err(Error::Parse(Cow::Borrowed(ERR_HOST)));
    }
    let default_port = scheme.map_or(UrlScheme::Http.default_port(), UrlScheme::default_port);
    let (host, port, explicit_port) = parse_authority_host_port(authority, default_port)?;
    let host_for_header = if host.contains(':') {
        format!("[{host}]")
    } else {
        host.clone()
    };
    Ok(ParsedServerAddress {
        scheme,
        host,
        host_for_header,
        port,
        explicit_port,
    })
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
            let _ = write!(
                &mut net_ctx.curl_stderr_buf,
                "curl {context} 실패, 상태: {}",
                output.status
            );
        }
        return Err(Error::Curl(mem::take(&mut net_ctx.curl_stderr_buf)));
    }
    let mut end = stdout_bytes.len();
    while end > 0 && stdout_bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    let pos = stdout_bytes[..end]
        .iter()
        .rposition(|&b| b == b'\n')
        .ok_or_else(|| Error::Parse("curl 응답에서 time_starttransfer 정보 누락".into()))?;
    let headers_part = &stdout_bytes[..pos];
    let time_starttransfer_part = &stdout_bytes[pos + 1..end];
    let time_starttransfer_str = std::str::from_utf8(time_starttransfer_part)
        .map_err(|_| Error::Parse("curl time_starttransfer 파싱 실패".into()))?;
    let time_starttransfer_secs: f64 = time_starttransfer_str
        .trim_ascii()
        .parse()
        .map_err(|_| Error::Parse("curl time_starttransfer 파싱 실패".into()))?;
    let rtt_reported_by_curl = Duration::from_secs_f64(time_starttransfer_secs.max(0.000_001));
    let date_header_str_slice = headers_part
        .split(|&b| b == b'\n')
        .rev()
        .find_map(|line| find_date_header_value(line))
        .ok_or_else(|| Error::HeaderNotFound("curl 응답에서 Date 헤더를 찾을 수 없음".into()))?;
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
        if let Ok(ip_addr) = address.host.parse::<net::IpAddr>() {
            Ok(net::SocketAddr::new(ip_addr, address.port))
        } else {
            net::ToSocketAddrs::to_socket_addrs(&(address.host.as_str(), address.port))?
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
    Err(Error::HeaderNotFound("Date (TCP)".into()))
}
fn fetch_server_time_sample(host: &str, net_ctx: &mut NetworkContext) -> Result<TimeSample> {
    let parsed_address = parse_server_address(host)?;
    if parsed_address.scheme == Some(UrlScheme::Https) {
        let https_url = parsed_address.curl_url(UrlScheme::Https);
        return fetch_server_time_sample_curl(&https_url, "HTTPS (explicit)", net_ctx);
    }
    tcp_attempt(&mut net_ctx.tcp_line_buffer, &parsed_address).or_else(|_| {
        if !*CURL_AVAILABLE {
            return Err(Error::SyncFailed(
                "TCP 연결에 실패했고 curl을 사용할 수 없습니다.".into(),
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
            .unwrap_or_else(|| Error::SyncFailed("Curl 폴백 시도 중 알 수 없는 오류".into())))
    })
}
fn parse_u32_digits(raw: &str) -> Option<u32> {
    let mut value = 0u32;
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
fn parse_two_digits(d0: u8, d1: u8) -> Option<u32> {
    if d0.is_ascii_digit() && d1.is_ascii_digit() {
        Some(u32::from(d0 - b'0') * 10 + u32::from(d1 - b'0'))
    } else {
        None
    }
}
#[derive(Clone, Copy)]
struct HttpDateComponents {
    weekday: u32,
    day: u32,
    month: u32,
    year: i32,
    hour: u32,
    minute: u32,
    second: u32,
}
fn parse_http_time_components(time_str: &str) -> Result<(u32, u32, u32)> {
    const ERR_TIME_FMT: &str = "HTTP Date 파싱 실패: 시간 형식이 올바르지 않습니다 (HH:MM:SS)";
    const ERR_TIME_RANGE: &str = "HTTP Date 파싱 실패: 시간 값 범위가 올바르지 않습니다.";
    let time_array = time_str
        .as_bytes()
        .as_array::<8>()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_TIME_FMT)))?;
    if time_array[2] != b':' || time_array[5] != b':' {
        return Err(Error::Parse(Cow::Borrowed(ERR_TIME_FMT)));
    }
    let hour = parse_two_digits(time_array[0], time_array[1])
        .ok_or(Error::Parse(Cow::Borrowed(ERR_TIME_FMT)))?;
    let minute = parse_two_digits(time_array[3], time_array[4])
        .ok_or(Error::Parse(Cow::Borrowed(ERR_TIME_FMT)))?;
    let second = parse_two_digits(time_array[6], time_array[7])
        .ok_or(Error::Parse(Cow::Borrowed(ERR_TIME_FMT)))?;
    if hour > 23 || minute > 59 || second > 59 {
        return Err(Error::Parse(Cow::Borrowed(ERR_TIME_RANGE)));
    }
    Ok((hour, minute, second))
}
fn parse_http_month(month_str: &str) -> Result<u32> {
    const ERR_MONTH: &str = "HTTP Date 파싱 실패: 알 수 없는 월 형식";
    match month_str {
        "Jan" => Ok(1),
        "Feb" => Ok(2),
        "Mar" => Ok(3),
        "Apr" => Ok(4),
        "May" => Ok(5),
        "Jun" => Ok(6),
        "Jul" => Ok(7),
        "Aug" => Ok(8),
        "Sep" => Ok(9),
        "Oct" => Ok(10),
        "Nov" => Ok(11),
        "Dec" => Ok(12),
        _ => Err(Error::Parse(Cow::Borrowed(ERR_MONTH))),
    }
}
fn parse_http_weekday_short(weekday_str: &str) -> Option<u32> {
    match weekday_str {
        "Sun" => Some(0),
        "Mon" => Some(1),
        "Tue" => Some(2),
        "Wed" => Some(3),
        "Thu" => Some(4),
        "Fri" => Some(5),
        "Sat" => Some(6),
        _ => None,
    }
}
fn parse_http_weekday_long(weekday_str: &str) -> Option<u32> {
    match weekday_str {
        "Sunday" => Some(0),
        "Monday" => Some(1),
        "Tuesday" => Some(2),
        "Wednesday" => Some(3),
        "Thursday" => Some(4),
        "Friday" => Some(5),
        "Saturday" => Some(6),
        _ => None,
    }
}
const fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}
const fn days_in_month(year: i32, month: u32) -> Option<u32> {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => Some(31),
        4 | 6 | 9 | 11 => Some(30),
        2 => Some(if is_leap_year(year) { 29 } else { 28 }),
        _ => None,
    }
}
fn current_utc_year() -> i32 {
    let day_index_i64 = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => {
            i64::try_from(duration.as_secs() / 86_400).unwrap_or_else(|_| i64::from(i32::MAX))
        }
        Err(err) => {
            let secs_before_epoch = err.duration().as_secs();
            let days_before_epoch = secs_before_epoch.saturating_add(86_399) / 86_400;
            let days_before_epoch_i64 =
                i64::try_from(days_before_epoch).unwrap_or_else(|_| i64::from(i32::MAX));
            -days_before_epoch_i64
        }
    };
    let day_index = i32::try_from(day_index_i64).unwrap_or_else(|_| {
        if day_index_i64.is_negative() {
            i32::MIN
        } else {
            i32::MAX
        }
    });
    civil_from_days(day_index).0
}
fn expand_rfc850_year(two_digit_year: u32) -> Result<i32> {
    const ERR_YEAR: &str = "HTTP Date 파싱 실패: rfc850 2자리 연도 변환에 실패했습니다.";
    let year_2 =
        i32::try_from(two_digit_year).map_err(|_| Error::Parse(Cow::Borrowed(ERR_YEAR)))?;
    let current_year = current_utc_year();
    let century_base = current_year.div_euclid(100) * 100;
    let mut expanded = century_base + year_2;
    if expanded > current_year + 50 {
        expanded -= 100;
    }
    Ok(expanded)
}
fn parse_http_date_imf_fixdate(raw_date: &str) -> Result<HttpDateComponents> {
    const ERR_FORMAT: &str = "HTTP Date 파싱 실패: IMF-fixdate 형식이 아닙니다.";
    const ERR_NUM: &str = "HTTP Date 파싱 실패: IMF-fixdate 숫자 변환에 실패했습니다.";
    let mut parts = raw_date.split_ascii_whitespace();
    let weekday_token = parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let day_token = parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let month_token = parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let year_token = parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let time_token = parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let tz_token = parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    if parts.next().is_some() || day_token.len() != 2 || year_token.len() != 4 || tz_token != "GMT"
    {
        return Err(Error::Parse(Cow::Borrowed(ERR_FORMAT)));
    }
    let weekday_name = weekday_token
        .strip_suffix(',')
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let weekday =
        parse_http_weekday_short(weekday_name).ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let day = parse_u32_digits(day_token).ok_or(Error::Parse(Cow::Borrowed(ERR_NUM)))?;
    let month = parse_http_month(month_token)?;
    let year_u32 = parse_u32_digits(year_token).ok_or(Error::Parse(Cow::Borrowed(ERR_NUM)))?;
    let year = i32::try_from(year_u32).map_err(|_| Error::Parse(Cow::Borrowed(ERR_NUM)))?;
    let (hour, minute, second) = parse_http_time_components(time_token)?;
    Ok(HttpDateComponents {
        weekday,
        day,
        month,
        year,
        hour,
        minute,
        second,
    })
}
fn parse_http_date_rfc850(raw_date: &str) -> Result<HttpDateComponents> {
    const ERR_FORMAT: &str = "HTTP Date 파싱 실패: rfc850-date 형식이 아닙니다.";
    const ERR_NUM: &str = "HTTP Date 파싱 실패: rfc850-date 숫자 변환에 실패했습니다.";
    let mut parts = raw_date.split_ascii_whitespace();
    let weekday_token = parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let date_token = parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let time_token = parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let tz_token = parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    if parts.next().is_some() || tz_token != "GMT" {
        return Err(Error::Parse(Cow::Borrowed(ERR_FORMAT)));
    }
    let weekday_name = weekday_token
        .strip_suffix(',')
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let weekday =
        parse_http_weekday_long(weekday_name).ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let mut date_parts = date_token.split('-');
    let day_token = date_parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let month_token = date_parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let year2_token = date_parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    if date_parts.next().is_some() || day_token.len() != 2 || year2_token.len() != 2 {
        return Err(Error::Parse(Cow::Borrowed(ERR_FORMAT)));
    }
    let day = parse_u32_digits(day_token).ok_or(Error::Parse(Cow::Borrowed(ERR_NUM)))?;
    let month = parse_http_month(month_token)?;
    let year2 = parse_u32_digits(year2_token).ok_or(Error::Parse(Cow::Borrowed(ERR_NUM)))?;
    let year = expand_rfc850_year(year2)?;
    let (hour, minute, second) = parse_http_time_components(time_token)?;
    Ok(HttpDateComponents {
        weekday,
        day,
        month,
        year,
        hour,
        minute,
        second,
    })
}
fn parse_http_date_asctime(raw_date: &str) -> Result<HttpDateComponents> {
    const ERR_FORMAT: &str = "HTTP Date 파싱 실패: asctime-date 형식이 아닙니다.";
    const ERR_NUM: &str = "HTTP Date 파싱 실패: asctime-date 숫자 변환에 실패했습니다.";
    let mut parts = raw_date.split_ascii_whitespace();
    let weekday_token = parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let month_token = parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let day_token = parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let time_token = parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let year_token = parts
        .next()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    if parts.next().is_some() || !(1..=2).contains(&day_token.len()) || year_token.len() != 4 {
        return Err(Error::Parse(Cow::Borrowed(ERR_FORMAT)));
    }
    let weekday =
        parse_http_weekday_short(weekday_token).ok_or(Error::Parse(Cow::Borrowed(ERR_FORMAT)))?;
    let day = parse_u32_digits(day_token).ok_or(Error::Parse(Cow::Borrowed(ERR_NUM)))?;
    let month = parse_http_month(month_token)?;
    let year_u32 = parse_u32_digits(year_token).ok_or(Error::Parse(Cow::Borrowed(ERR_NUM)))?;
    let year = i32::try_from(year_u32).map_err(|_| Error::Parse(Cow::Borrowed(ERR_NUM)))?;
    let (hour, minute, second) = parse_http_time_components(time_token)?;
    Ok(HttpDateComponents {
        weekday,
        day,
        month,
        year,
        hour,
        minute,
        second,
    })
}
fn unix_timestamp_to_system_time(timestamp_secs: i64) -> Result<SystemTime> {
    const ERR_TIMESTAMP: &str = "HTTP Date 변환 실패: 유효하지 않은 타임스탬프입니다.";
    let secs_i128 = i128::from(timestamp_secs);
    if secs_i128 >= 0 {
        let secs_u64 =
            u64::try_from(secs_i128).map_err(|_| Error::Parse(Cow::Borrowed(ERR_TIMESTAMP)))?;
        UNIX_EPOCH
            .checked_add(Duration::from_secs(secs_u64))
            .ok_or(Error::Parse(Cow::Borrowed(ERR_TIMESTAMP)))
    } else {
        let abs_i128 = secs_i128
            .checked_abs()
            .ok_or(Error::Parse(Cow::Borrowed(ERR_TIMESTAMP)))?;
        let abs_secs =
            u64::try_from(abs_i128).map_err(|_| Error::Parse(Cow::Borrowed(ERR_TIMESTAMP)))?;
        UNIX_EPOCH
            .checked_sub(Duration::from_secs(abs_secs))
            .ok_or(Error::Parse(Cow::Borrowed(ERR_TIMESTAMP)))
    }
}
const fn validate_http_date_components(components: HttpDateComponents) -> Result<()> {
    const ERR_DAY: &str = "HTTP Date 파싱 실패: 날짜 값이 유효하지 않습니다.";
    let Some(max_day) = days_in_month(components.year, components.month) else {
        return Err(Error::Parse(Cow::Borrowed(ERR_DAY)));
    };
    if components.day == 0 || components.day > max_day {
        return Err(Error::Parse(Cow::Borrowed(ERR_DAY)));
    }
    Ok(())
}
fn http_date_components_to_systemtime(components: HttpDateComponents) -> Result<SystemTime> {
    const ERR_WEEKDAY: &str = "HTTP Date 파싱 실패: 요일이 날짜와 일치하지 않습니다.";
    validate_http_date_components(components)?;
    let days = days_from_civil(components.year, components.month, components.day);
    let actual_weekday = (days + 4).rem_euclid(7).cast_unsigned();
    if actual_weekday != components.weekday {
        return Err(Error::Parse(Cow::Borrowed(ERR_WEEKDAY)));
    }
    let timestamp_secs = i64::from(days) * 86_400
        + i64::from(components.hour) * 3_600
        + i64::from(components.minute) * 60
        + i64::from(components.second);
    unix_timestamp_to_system_time(timestamp_secs)
}
fn parse_http_date_to_systemtime(raw_date: &str) -> Result<SystemTime> {
    const ERR_FORMAT: &str = "HTTP Date 파싱 실패: RFC 9110 HTTP-date의 3개 형식(IMF-fixdate/rfc850/asctime) 중 하나가 아닙니다.";
    if raw_date.as_bytes().get(3).is_some_and(|ch| *ch == b',') {
        return parse_http_date_imf_fixdate(raw_date).and_then(http_date_components_to_systemtime);
    }
    if raw_date.contains(',') {
        return parse_http_date_rfc850(raw_date).and_then(http_date_components_to_systemtime);
    }
    if raw_date.contains("GMT") {
        return parse_http_date_rfc850(raw_date).and_then(http_date_components_to_systemtime);
    }
    if raw_date
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_alphabetic)
    {
        return parse_http_date_asctime(raw_date).and_then(http_date_components_to_systemtime);
    }
    Err(Error::Parse(Cow::Borrowed(ERR_FORMAT)))
}
const fn days_from_civil(y: i32, m: u32, d: u32) -> i32 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let shifted_month: i32 = if m > 2 {
        (m - 3).cast_signed()
    } else {
        (m + 9).cast_signed()
    };
    let doy = (153 * shifted_month + 2) / 5 + d.cast_signed() - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}
fn civil_from_days(z: i32) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1).cast_unsigned();
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }).cast_unsigned();
    (y + i32::from(m <= 2), m, d)
}
