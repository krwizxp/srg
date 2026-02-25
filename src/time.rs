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
mod high_res_timer {
    #[link(name = "winmm")]
    unsafe extern "system" {
        fn timeBeginPeriod(uPeriod: u32) -> u32;
        fn timeEndPeriod(uPeriod: u32) -> u32;
    }
    const TIMERR_NOERROR: u32 = 0;
    const TARGET_PERIOD_MS: u32 = 1;
    pub struct HighResTimerGuard;
    impl HighResTimerGuard {
        pub fn new() -> Result<Self, String> {
            unsafe {
                if timeBeginPeriod(TARGET_PERIOD_MS) == TIMERR_NOERROR {
                    Ok(Self)
                } else {
                    Err(format!(
                        "타이머 해상도를 {TARGET_PERIOD_MS}ms로 설정하는 데 실패했습니다."
                    ))
                }
            }
        }
    }
    impl Drop for HighResTimerGuard {
        fn drop(&mut self) {
            unsafe {
                timeEndPeriod(TARGET_PERIOD_MS);
            }
        }
    }
}
#[cfg(target_os = "windows")]
mod windows_input {
    const INPUT_MOUSE: u32 = 0;
    const INPUT_KEYBOARD: u32 = 1;
    const MOUSEEVENTF_LEFTDOWN: u32 = 0x0002;
    const MOUSEEVENTF_LEFTUP: u32 = 0x0004;
    const KEYEVENTF_KEYUP: u32 = 0x0002;
    const VK_F5: u16 = 0x74;
    #[repr(C)]
    #[derive(Copy, Clone, Default)]
    struct MouseInput {
        dx: i32,
        dy: i32,
        mouse_data: u32,
        dw_flags: u32,
        time: u32,
        dw_extra_info: usize,
    }
    #[repr(C)]
    #[derive(Copy, Clone, Default)]
    struct KeybdInput {
        w_vk: u16,
        w_scan: u16,
        dw_flags: u32,
        time: u32,
        dw_extra_info: usize,
    }
    #[repr(C)]
    #[derive(Copy, Clone)]
    union InputUnion {
        mi: MouseInput,
        ki: KeybdInput,
    }
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct Input {
        r#type: u32,
        union: InputUnion,
    }
    impl Input {
        const fn mouse(mi: MouseInput) -> Self {
            Self {
                r#type: INPUT_MOUSE,
                union: InputUnion { mi },
            }
        }
        const fn keyboard(ki: KeybdInput) -> Self {
            Self {
                r#type: INPUT_KEYBOARD,
                union: InputUnion { ki },
            }
        }
    }
    #[cfg(target_pointer_width = "64")]
    const _: [(); 40] = [(); std::mem::size_of::<Input>()];
    #[cfg(target_pointer_width = "32")]
    const _: [(); 28] = [(); std::mem::size_of::<Input>()];
    #[link(name = "user32")]
    unsafe extern "system" {
        fn SendInput(cInputs: u32, pInputs: *const Input, cbSize: i32) -> u32;
    }
    fn send_input_events(inputs: &[Input]) {
        unsafe {
            let input_count = u32::try_from(inputs.len())
                .expect("SendInput input count fits within u32 on supported targets");
            let input_size = i32::try_from(std::mem::size_of::<Input>())
                .expect("INPUT struct size always fits in i32");
            SendInput(input_count, inputs.as_ptr(), input_size);
        }
    }
    pub fn send_mouse_click() {
        let inputs = [
            Input::mouse(MouseInput {
                dw_flags: MOUSEEVENTF_LEFTDOWN,
                ..Default::default()
            }),
            Input::mouse(MouseInput {
                dw_flags: MOUSEEVENTF_LEFTUP,
                ..Default::default()
            }),
        ];
        send_input_events(&inputs);
    }
    pub fn send_f5_press() {
        let inputs = [
            Input::keyboard(KeybdInput {
                w_vk: VK_F5,
                ..Default::default()
            }),
            Input::keyboard(KeybdInput {
                w_vk: VK_F5,
                dw_flags: KEYEVENTF_KEYUP,
                ..Default::default()
            }),
        ];
        send_input_events(&inputs);
    }
}
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
            let rem = usize::try_from(n % 100).expect("remainder is in 0..100");
            n /= 100;
            i -= 2;
            tmp[i..i + 2].copy_from_slice(&TWO_DIGITS[rem]);
        }
        if n >= 10 {
            let rem = usize::try_from(n).expect("n is in 10..100");
            i -= 2;
            tmp[i..i + 2].copy_from_slice(&TWO_DIGITS[rem]);
        } else {
            i -= 1;
            let digit = u8::try_from(n).expect("n is in 0..10");
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
                let hi = usize::try_from(y / 100).expect("year chunk fits usize");
                let lo = usize::try_from(y % 100).expect("year chunk fits usize");
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
            let hundreds = usize::try_from(abs / 100).expect("hundreds fits usize");
            let rem = usize::try_from(abs % 100).expect("remainder fits usize");
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
        let idx = usize::try_from(v).expect("v is in 0..100");
        self.buf[start..start + 2].copy_from_slice(&TWO_DIGITS[idx]);
        self.pos = start + 2;
        Ok(())
    }
    fn write_u32_3digits(&mut self, v: u32) -> ioResult<()> {
        if self.remaining() < 3 {
            return Err(write_zero_err());
        }
        let hundreds = usize::try_from(v / 100).expect("hundreds is in 0..10");
        let rem = usize::try_from(v % 100).expect("remainder is in 0..100");
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
const TCP_TIMEOUT_SECS_STR: &str = "5";
#[cfg(target_os = "linux")]
static XDO_TOOL_AVAILABLE: LazyLock<bool> = LazyLock::new(|| is_command_available("xdotool"));
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
        })
    }
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
        let window_len = u128::try_from(window.len()).expect("slice length fits u128");
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
    if let Err(e) = high_res_timer::HighResTimerGuard::new() {
        eprintln!("[경고] {e}. 시간 오차가 클 수 있습니다.");
    }
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
        )
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
    let _ = Command::new(program).args(args).status();
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
fn fetch_server_time_sample_curl(
    url_str: &str,
    context: &str,
    net_ctx: &mut NetworkContext,
) -> Result<TimeSample> {
    net_ctx.curl_stderr_buf.clear();
    let time_before_curl_call_inst = Instant::now();
    let timeout_str = TCP_TIMEOUT_SECS_STR;
    let output = Command::new("curl")
        .args([
            "-sI",
            "--ssl-no-revoke",
            "-L",
            "--max-time",
            timeout_str,
            "--connect-timeout",
            timeout_str,
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
fn tcp_attempt(line_buffer: &mut Vec<u8>, host: &str) -> Result<TimeSample> {
    let request_start_inst = Instant::now();
    let tcp_host_uri = host.strip_prefix("http://").unwrap_or(host);
    let (tcp_host_no_port, _) = tcp_host_uri.split_once(':').unwrap_or((tcp_host_uri, ""));
    let tcp_timeout = Duration::from_secs(TCP_TIMEOUT_SECS);
    let socket_addr_result: ioResult<net::SocketAddr> =
        if let Ok(ip_addr) = tcp_host_no_port.parse::<net::IpAddr>() {
            Ok(net::SocketAddr::new(ip_addr, 80))
        } else {
            net::ToSocketAddrs::to_socket_addrs(&(tcp_host_no_port, 80))?
                .next()
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Host not found"))
        };
    let socket_addr = socket_addr_result?;
    let mut stream = TcpStream::connect_timeout(&socket_addr, tcp_timeout)?;
    stream.set_read_timeout(Some(tcp_timeout))?;
    stream.set_write_timeout(Some(tcp_timeout))?;
    stream.write_all(b"HEAD / HTTP/1.1\r\nHost: ")?;
    stream.write_all(tcp_host_no_port.as_bytes())?;
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
    if host
        .as_bytes()
        .get(..8)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"https://"))
    {
        return fetch_server_time_sample_curl(host, "HTTPS (explicit)", net_ctx);
    }
    tcp_attempt(&mut net_ctx.tcp_line_buffer, host).or_else(|_| {
        if !*CURL_AVAILABLE {
            return Err(Error::SyncFailed(
                "TCP 연결에 실패했고 curl을 사용할 수 없습니다.".into(),
            ));
        }
        let base_host = host.strip_prefix("http://").unwrap_or(host);
        let mut url_buf = [0u8; 512];
        let mut last_error = None;
        for (protocol, context_str) in [
            ("https://", "HTTPS (fallback)"),
            ("http://", "HTTP (fallback)"),
        ] {
            let protocol_bytes = protocol.as_bytes();
            let host_bytes = base_host.as_bytes();
            let len = protocol_bytes.len() + host_bytes.len();
            if len > url_buf.len() {
                return Err(Error::Parse("URL 생성 버퍼가 작습니다".into()));
            }
            url_buf[..protocol_bytes.len()].copy_from_slice(protocol_bytes);
            url_buf[protocol_bytes.len()..len].copy_from_slice(host_bytes);
            let url_str = str::from_utf8(&url_buf[..len])
                .map_err(|_| Error::Parse("URL 생성 중 UTF-8 변환 실패".into()))?;
            match fetch_server_time_sample_curl(url_str, context_str, net_ctx) {
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
        value = value * 10 + u32::from(byte - b'0');
    }
    Some(value)
}
fn parse_i32_digits(raw: &str) -> Option<i32> {
    let bytes = raw.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let mut idx = 0usize;
    let mut negative = false;
    match bytes[0] {
        b'+' => idx = 1,
        b'-' => {
            negative = true;
            idx = 1;
        }
        _ => {}
    }
    if idx == bytes.len() {
        return None;
    }
    let mut value: i32 = 0;
    while idx < bytes.len() {
        let byte = bytes[idx];
        if !byte.is_ascii_digit() {
            return None;
        }
        let digit = i32::from(byte - b'0');
        value = if negative {
            value.checked_mul(10)?.checked_sub(digit)?
        } else {
            value.checked_mul(10)?.checked_add(digit)?
        };
        idx += 1;
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
fn parse_http_time_components(time_str: &str) -> Result<(u32, u32, u32)> {
    const ERR_TIME_FMT: &str = "HTTP Date 파싱 실패: 시간 형식이 올바르지 않습니다 (HH:MM:SS)";
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
    Ok((hour, minute, second))
}
fn parse_http_month(month_str: &str) -> Result<u32> {
    const ERR_MONTH: &str = "HTTP Date 파싱 실패: 알 수 없는 월 형식";
    let month_array = month_str
        .as_bytes()
        .as_array::<3>()
        .ok_or(Error::Parse(Cow::Borrowed(ERR_MONTH)))?;
    let first = month_array[0].to_ascii_lowercase();
    let second = month_array[1].to_ascii_lowercase();
    let third = month_array[2].to_ascii_lowercase();
    match (first, second, third) {
        (b'j', b'a', b'n') => Ok(1),
        (b'f', b'e', b'b') => Ok(2),
        (b'm', b'a', b'r') => Ok(3),
        (b'a', b'p', b'r') => Ok(4),
        (b'm', b'a', b'y') => Ok(5),
        (b'j', b'u', b'n') => Ok(6),
        (b'j', b'u', b'l') => Ok(7),
        (b'a', b'u', b'g') => Ok(8),
        (b's', b'e', b'p') => Ok(9),
        (b'o', b'c', b't') => Ok(10),
        (b'n', b'o', b'v') => Ok(11),
        (b'd', b'e', b'c') => Ok(12),
        _ => Err(Error::Parse(Cow::Borrowed(ERR_MONTH))),
    }
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
fn parse_http_date_to_systemtime(raw_date: &str) -> Result<SystemTime> {
    const ERR_BAD_FORMAT: &str = "HTTP Date 파싱 실패: 형식이 올바르지 않습니다.";
    const ERR_NUM_CONV: &str = "HTTP Date 파싱 실패: 날짜 또는 시간의 숫자 변환에 실패했습니다.";
    let mut parts = raw_date.split_ascii_whitespace();
    let mut expect_part = |msg: &'static str| parts.next().ok_or(Error::Parse(Cow::Borrowed(msg)));
    let _weekday = expect_part(ERR_BAD_FORMAT)?;
    let day_str = expect_part(ERR_BAD_FORMAT)?;
    let month_str = expect_part(ERR_BAD_FORMAT)?;
    let year_str = expect_part(ERR_BAD_FORMAT)?;
    let time_str = expect_part(ERR_BAD_FORMAT)?;
    let day = parse_u32_digits(day_str).ok_or(Error::Parse(Cow::Borrowed(ERR_NUM_CONV)))?;
    let year = parse_i32_digits(year_str).ok_or(Error::Parse(Cow::Borrowed(ERR_NUM_CONV)))?;
    let month = parse_http_month(month_str)?;
    let (hour, minute, second) = parse_http_time_components(time_str)?;
    let days = days_from_civil(year, month, day);
    let timestamp_secs = i64::from(days) * 86_400
        + i64::from(hour) * 3_600
        + i64::from(minute) * 60
        + i64::from(second);
    unix_timestamp_to_system_time(timestamp_secs)
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
