use self::{
    activity::{ADAPTIVE_POLL_INTERVAL, Activity, CountdownTrigger},
    sample::{TCP_LINE_BUFFER_CAPACITY, fetch_server_time_sample},
    util::{NewSampleWeight, blend_rtt, parse_u32_digits},
};
cfg_select! {
    target_os = "windows" => {
        use self::timer_resolution::{
            CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, SYNCHRONIZE_ACCESS, TIMER_MODIFY_STATE_ACCESS,
            sys,
        };
        use core::{ffi::c_void, ptr::{NonNull, null}};
    }
    _ => {}
}
use crate::{buffmt::ByteCursor, write_line_best_effort};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use alloc::ffi::CString;
use alloc::{borrow::Cow, sync::Arc};
use core::{
    error::Error,
    fmt::{self, Write as FmtWrite},
    hint::spin_loop,
    ops::Mul as NumericMul,
    result::Result as CoreResult,
    str::FromStr,
    time::Duration,
};
use std::{
    io::{self, BufRead as IoBufRead, Error as IoError, Read as IoRead, Result as IoResult},
    net,
    sync::{Mutex, TryLockError, mpsc},
    thread,
    time::{Instant, SystemTime, SystemTimeError, UNIX_EPOCH},
};
mod activity;
mod address;
mod display;
mod http_date;
cfg_select! {
    target_os = "linux" => {
        mod wayland_input;
    }
    target_os = "macos" => {
        mod macos_input;
    }
    target_os = "windows" => {
        mod windows_input;
    }
    _ => {
        compile_error!("SRG native input supports only Windows, Linux, and macOS.");
    }
}
mod native_http;
mod sample;
mod util;
cfg_select! {
    target_os = "windows" => {
        mod timer_resolution;
    }
    _ => {}
}
const FULL_SYNC_INTERVAL: Duration = Duration::from_mins(5);
const RETRY_DELAY: Duration = Duration::from_secs(10);
const SAMPLE_WORKER_RESTARTED_MESSAGE: &str = "서버 시간 샘플 worker가 종료되어 다시 시작했습니다.";
const TCP_TIMEOUT: Duration = Duration::from_secs(5);
const ENTER_BUFFER_CAPACITY: usize = 8;
const ENTER_BUFFER_READ_LIMIT_BYTES: usize = ENTER_BUFFER_CAPACITY + 1;
const ENTER_INPUT_TOO_LONG: &str = "서버 시간 종료 입력이 너무 깁니다.";
const ENTER_THREAD_PANIC: &str = "입력 대기 스레드 패닉 발생";
const NUM_SAMPLES: usize = 10;
const CALIBRATION_TIMEOUT: Duration = Duration::from_secs(5);
const CALIBRATION_TIMEOUT_MESSAGE: &str = "정밀 보정 제한 시간 안에 유효한 서버 Date tick을 관측하지 못했습니다. 전체 보정을 다시 시작합니다.";
const HTTP_SCHEME_PREFIX: &str = "http://";
const HTTP_SCHEME_PREFIX_LEN: usize = HTTP_SCHEME_PREFIX.len();
const HTTPS_SCHEME_PREFIX: &str = "https://";
const HTTPS_SCHEME_PREFIX_LEN: usize = HTTPS_SCHEME_PREFIX.len();
const KST_OFFSET: Duration = Duration::from_hours(9);
const KST_OFFSET_SECS_U64: u64 = KST_OFFSET.as_secs();
const KST_OFFSET_SECS: i64 = KST_OFFSET_SECS_U64.cast_signed();
const KST_SECONDS_PER_MINUTE_U32: u32 = 60;
const KST_SECONDS_PER_HOUR_U32: u32 = 60 * KST_SECONDS_PER_MINUTE_U32;
const KST_SECONDS_PER_DAY_U64: u64 = 86_400;
const CLOCK_COMPONENT_LEN: usize = 2;
const DISPLAY_INTERVAL: Duration = Duration::from_millis(16);
const DISPLAY_STATUS_PREFIX: &str = "\r서버 시간: ";
const FINAL_COUNTDOWN_FREEZE_WINDOW: Duration = Duration::from_millis(200);
const FINAL_COUNTDOWN_SAMPLE_ERROR_MESSAGE_INTERVAL: Duration = Duration::from_secs(1);
const FINAL_COUNTDOWN_SAMPLE_FAST_INTERVAL: Duration = Duration::from_millis(50);
const FINAL_COUNTDOWN_SAMPLE_FINAL_INTERVAL: Duration = Duration::from_millis(25);
const FINAL_COUNTDOWN_SAMPLE_NORMAL_INTERVAL: Duration = Duration::from_millis(200);
const FINAL_COUNTDOWN_SAMPLE_WARMUP_INTERVAL: Duration = Duration::from_millis(500);
const FINAL_COUNTDOWN_SLEEP_MARGIN: Duration = Duration::from_millis(2);
const FINAL_COUNTDOWN_WARMUP_WINDOW: Duration = Duration::from_mins(1);
const FINAL_COUNTDOWN_WINDOW: Duration = Duration::from_secs(10);
const HALF_RTT_DIVISOR: u32 = 2;
const MESSAGE_BUFFER_CAPACITY: usize = 256;
const MILLIS_PER_SECOND_F64: f64 = 1000.0;
const MIN_TRANSFER_TIME: Duration = Duration::from_micros(1);
const RTT_TRIM_DIVISOR: usize = 5;
type BoxError = Box<dyn Error + Send + Sync>;
type Result<T> = CoreResult<T, TimeError>;
#[derive(Clone, Copy, Debug)]
pub(super) enum TriggerAction {
    F5Press,
    LeftClick,
}
enum NativeInputSendStatus {
    FailedBeforeSend,
    PartialOrUnknown,
    Sent,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TimeErrorKind {
    HeaderNotFound,
    Io,
    NativeHttp,
    Parse,
    Time,
}
#[derive(Clone, Copy)]
enum CountdownTriggerSource {
    Estimated,
    Sampled { rtt: Duration },
}
#[cfg(target_os = "windows")]
struct HighResTimerGuard {
    handle: NonNull<c_void>,
}
#[derive(Debug)]
pub(super) struct TimeError {
    detail: Cow<'static, str>,
    kind: TimeErrorKind,
    source: Option<BoxError>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UrlScheme {
    Http,
    Https,
}
#[derive(Debug)]
pub(super) struct ParsedServer {
    host: String,
    literal_tcp_socket_addr: Option<net::SocketAddr>,
    port: u16,
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    request_target: CString,
    #[cfg(target_os = "windows")]
    request_target: String,
    scheme: UrlScheme,
}
pub(super) struct TargetTimeOfDay {
    seconds_after_midnight: u32,
}
pub(super) struct ServerTimeSession {
    pub host: ParsedServer,
    pub scheduled_trigger: Option<(TargetTimeOfDay, TriggerAction)>,
    pub stop_after: Option<Duration>,
}
#[derive(Default)]
pub(super) struct ServerTimeRuntime {
    sample_worker: Option<SampleWorker>,
}
#[derive(Clone, Copy, Debug)]
struct CivilDate {
    day: u32,
    month: u32,
    year: i32,
}
struct ActivityTransition<'message> {
    activity: Activity,
    message: Option<&'message str>,
}
#[derive(Clone, Copy, Debug)]
struct TimeSample {
    response_received_inst: Instant,
    rtt: Duration,
    server_time: SystemTime,
}
#[derive(Clone, Copy, Debug)]
struct ServerTime {
    anchor_instant: Instant,
    anchor_time: SystemTime,
    baseline_rtt: Duration,
}
impl FromStr for TargetTimeOfDay {
    type Err = &'static str;
    fn from_str(raw_input: &str) -> CoreResult<Self, Self::Err> {
        const INVALID_TIME_INPUT_ERR: &str =
            "잘못된 형식, 숫자 또는 시간 범위입니다 (HH:MM:SS, 0-23:0-59:0-59).";
        let Some((hour_str, minute_second)) = raw_input.split_once(':') else {
            return Err(INVALID_TIME_INPUT_ERR);
        };
        let Some((minute_str, second_str)) = minute_second.split_once(':') else {
            return Err(INVALID_TIME_INPUT_ERR);
        };
        if second_str.contains(':') {
            return Err(INVALID_TIME_INPUT_ERR);
        }
        let parse_component = |component: &str| -> CoreResult<u32, &'static str> {
            if component.len() != CLOCK_COMPONENT_LEN {
                return Err(INVALID_TIME_INPUT_ERR);
            }
            parse_u32_digits(component).ok_or(INVALID_TIME_INPUT_ERR)
        };
        let (hour, minute, second) = (
            parse_component(hour_str)?,
            parse_component(minute_str)?,
            parse_component(second_str)?,
        );
        if !(hour <= 23 && minute <= 59 && second <= 59) {
            return Err(INVALID_TIME_INPUT_ERR);
        }
        let seconds_after_midnight = hour
            .saturating_mul(KST_SECONDS_PER_HOUR_U32)
            .saturating_add(minute.saturating_mul(KST_SECONDS_PER_MINUTE_U32))
            .saturating_add(second);
        Ok(Self {
            seconds_after_midnight,
        })
    }
}
impl TimeError {
    fn header_not_found(detail: impl Into<Cow<'static, str>>) -> Self {
        Self::new(TimeErrorKind::HeaderNotFound, detail)
    }
    pub(super) const fn is_io(&self) -> bool {
        matches!(self.kind, TimeErrorKind::Io)
    }
    fn new(kind: TimeErrorKind, detail: impl Into<Cow<'static, str>>) -> Self {
        Self {
            detail: detail.into(),
            kind,
            source: None,
        }
    }
    fn new_with_source<E>(
        kind: TimeErrorKind,
        detail: impl Into<Cow<'static, str>>,
        source: E,
    ) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self {
            detail: detail.into(),
            kind,
            source: Some(Box::new(source)),
        }
    }
    fn parse(detail: impl Into<Cow<'static, str>>) -> Self {
        Self::new(TimeErrorKind::Parse, detail)
    }
    fn parse_with_source<E>(detail: impl Into<Cow<'static, str>>, source: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        Self::new_with_source(TimeErrorKind::Parse, detail, source)
    }
}
impl From<io::Error> for TimeError {
    fn from(err: io::Error) -> Self {
        Self::new_with_source(TimeErrorKind::Io, "", err)
    }
}
impl From<SystemTimeError> for TimeError {
    fn from(err: SystemTimeError) -> Self {
        Self::new_with_source(TimeErrorKind::Time, "", err)
    }
}
impl fmt::Display for TimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            TimeErrorKind::Io => match self.source.as_ref() {
                Some(source) => write!(f, "I/O 오류: {source}"),
                None => write!(f, "I/O 오류: {}", self.detail),
            },
            TimeErrorKind::Time => match self.source.as_ref() {
                Some(source) => write!(f, "시스템 시간 오류: {source}"),
                None => write!(f, "시스템 시간 오류: {}", self.detail),
            },
            TimeErrorKind::Parse => match self.source.as_ref() {
                Some(source) => write!(f, "파싱 오류: {}: {source}", self.detail),
                None => write!(f, "파싱 오류: {}", self.detail),
            },
            TimeErrorKind::HeaderNotFound => write!(f, "{} 헤더를 찾을 수 없음", self.detail),
            TimeErrorKind::NativeHttp => match self.source.as_ref() {
                Some(source) => write!(f, "native HTTP 요청 실패: {}: {source}", self.detail),
                None => write!(f, "native HTTP 요청 실패: {}", self.detail),
            },
        }
    }
}
impl Error for TimeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|source| {
            let source_ref: &(dyn Error + 'static) = source;
            source_ref
        })
    }
}
impl SampleWorker {
    fn fetch(&mut self, kind: SampleRequestKind, host: Arc<ParsedServer>) -> Result<Option<u64>> {
        let generation = self.generation.wrapping_add(1);
        match self.command_sender.try_send(SampleWorkerRequest {
            generation,
            host,
            kind,
        }) {
            Ok(()) => {
                self.generation = generation;
                Ok(Some(generation))
            }
            Err(mpsc::TrySendError::Full(_)) => Ok(None),
            Err(mpsc::TrySendError::Disconnected(_)) => {
                Err(TimeError::parse("서버 시간 샘플 worker 요청 실패"))
            }
        }
    }
}
impl ServerTimeRuntime {
    fn sample_worker(&mut self, host: Arc<ParsedServer>) -> Result<&mut SampleWorker> {
        let worker = match self.sample_worker.take() {
            Some(worker) => worker,
            None => spawn_sample_worker(host)?,
        };
        Ok(self.sample_worker.insert(worker))
    }
}
struct AppState<'worker> {
    baseline_rtt: Option<Duration>,
    baseline_rtt_attempts: usize,
    baseline_rtt_next_sample_at: Instant,
    baseline_rtt_samples: [Option<TimeSample>; NUM_SAMPLES],
    calibration_started_at: Option<Instant>,
    final_countdown_last_sample_error_message_at: Option<Instant>,
    final_countdown_sampler: Option<FinalCountdownSamplerActive>,
    #[cfg(target_os = "windows")]
    high_res_timer_guard: Option<HighResTimerGuard>,
    host: Arc<ParsedServer>,
    last_full_sync_at: Instant,
    last_sample: Option<TimeSample>,
    live_rtt: Option<Duration>,
    pending_sample_generation: Option<u64>,
    pending_target_time: Option<TargetTimeOfDay>,
    sample_worker: &'worker mut SampleWorker,
    target_time: Option<SystemTime>,
    trigger_action: Option<TriggerAction>,
}
struct LoopRuntime<'runtime> {
    err: &'runtime mut dyn io::Write,
    out: &'runtime mut dyn io::Write,
    #[cfg(target_os = "linux")]
    prepared_input: &'runtime mut wayland_input::PreparedInput,
}
#[derive(Clone, Copy)]
enum LoopStop<'receiver> {
    Deadline(Instant),
    Receiver(&'receiver mpsc::Receiver<()>),
}
struct SampleWorker {
    command_sender: mpsc::SyncSender<SampleWorkerRequest>,
    generation: u64,
    join_handle: thread::JoinHandle<()>,
    response_receiver: mpsc::Receiver<SampleWorkerResponse>,
}
struct FinalCountdownSamplerActive {
    command_sender: mpsc::Sender<Duration>,
    join_handle: thread::JoinHandle<()>,
    sample_interval: Duration,
    sample_slot: Arc<Mutex<Option<Result<TimeSample>>>>,
}
struct FinalCountdownDeadline {
    one_way_delay: Duration,
    server_time: ServerTime,
    source: CountdownTriggerSource,
    target_time: SystemTime,
    trigger_instant: Instant,
}
struct SampleWorkerResponse {
    generation: u64,
    kind: SampleRequestKind,
    result: Result<TimeSample>,
}
#[derive(Clone, Copy)]
enum SampleRequestKind {
    Baseline { attempt_index: usize },
    Calibration,
}
enum SampleRequestPoll {
    Empty,
    Sample {
        kind: SampleRequestKind,
        result: Result<TimeSample>,
    },
}
struct SampleWorkerRequest {
    generation: u64,
    host: Arc<ParsedServer>,
    kind: SampleRequestKind,
}
enum FinalCountdownSamplePoll {
    Disconnected,
    Empty,
    Sample(Result<TimeSample>),
}
struct NetworkContext {
    cached_tcp_connection: Option<io::BufReader<net::TcpStream>>,
    cached_tcp_socket_addr: Option<net::SocketAddr>,
    host: Arc<ParsedServer>,
    native_http: native_http::Client,
    tcp_line_buffer: Vec<u8>,
    tcp_request_buffer: Vec<u8>,
}
impl NetworkContext {
    fn new(host: Arc<ParsedServer>, (line, request): (Vec<u8>, Vec<u8>)) -> Self {
        Self {
            cached_tcp_connection: None,
            cached_tcp_socket_addr: None,
            host,
            native_http: native_http::Client::default(),
            tcp_line_buffer: line,
            tcp_request_buffer: request,
        }
    }
    fn reset_host(&mut self, host: Arc<ParsedServer>) {
        self.cached_tcp_connection = None;
        self.cached_tcp_socket_addr = None;
        self.host = host;
        self.native_http = native_http::Client::default();
    }
}
impl ServerTimeSession {
    pub(super) fn run_loop(
        self,
        runtime: &mut ServerTimeRuntime,
        out: &mut dyn io::Write,
        err: &mut dyn io::Write,
    ) -> Result<()> {
        let now = Instant::now();
        let host = Arc::new(self.host);
        let sample_worker = runtime.sample_worker(Arc::clone(&host))?;
        let (pending_target_time, trigger_action) = self.scheduled_trigger.unzip();
        cfg_select! {
            target_os = "macos" => {
                if trigger_action.is_some()
                    && !macos_input::post_event_access_granted(true)
                {
                    write_line_best_effort(
                        err,
                        format_args!("[경고] macOS 입력 제어 권한이 허용되지 않았습니다."),
                    );
                }
            }
            _ => {}
        }
        if let Some(stop_after) = self.stop_after {
            writeln!(
                out,
                "\n서버 시간 확인을 시작합니다… ({}초 후 종료)",
                stop_after.as_secs()
            )?;
        } else {
            out.write_all("\n서버 시간 확인을 시작합니다… (Enter를 누르면 종료)\n".as_bytes())?;
        }
        let mut app_state = AppState {
            baseline_rtt: None,
            baseline_rtt_attempts: 0,
            baseline_rtt_next_sample_at: now,
            baseline_rtt_samples: [None; NUM_SAMPLES],
            calibration_started_at: None,
            final_countdown_last_sample_error_message_at: None,
            final_countdown_sampler: None,
            #[cfg(target_os = "windows")]
            high_res_timer_guard: None,
            host,
            last_full_sync_at: now,
            last_sample: None,
            live_rtt: None,
            pending_sample_generation: None,
            pending_target_time,
            sample_worker,
            target_time: None,
            trigger_action,
        };
        let run_result = app_state.run_loop(self.stop_after, out, err);
        let worker_reusable =
            run_result.is_ok() && !app_state.sample_worker.join_handle.is_finished();
        drop(app_state);
        if !worker_reusable {
            runtime.sample_worker = None;
        }
        run_result
    }
}
impl AppState<'_> {
    fn accept_calibration_tick(&mut self, current_sample: TimeSample) -> Option<ServerTime> {
        let server_time = self.build_calibrated_server_time(current_sample)?;
        self.calibration_started_at = None;
        self.last_full_sync_at = current_sample.response_received_inst;
        Some(server_time)
    }
    fn append_final_countdown_sample_error(
        &mut self,
        now: Instant,
        msg_buf: &mut String,
        err: impl fmt::Display,
    ) -> bool {
        if self
            .final_countdown_last_sample_error_message_at
            .is_some_and(|last_message_at| {
                now.saturating_duration_since(last_message_at)
                    < FINAL_COUNTDOWN_SAMPLE_ERROR_MESSAGE_INTERVAL
            })
        {
            return false;
        }
        self.final_countdown_last_sample_error_message_at = Some(now);
        append_error_detail(msg_buf, "카운트다운 샘플 획득 실패: ", err);
        true
    }
    fn begin_baseline_rtt_measurement(&mut self, now: Instant, out: &mut dyn io::Write) {
        if self.last_sample.is_none() {
            write_line_best_effort(out, format_args!("1단계: RTT 기준값 측정을 시작합니다…"));
        }
        self.baseline_rtt_samples = [None; NUM_SAMPLES];
        self.baseline_rtt_next_sample_at = now;
        self.calibration_started_at = None;
    }
    fn begin_final_countdown_sampling(&mut self, initial_sample_interval: Duration) -> Result<()> {
        if self.final_countdown_sampler.is_none() {
            let network_buffers = try_network_buffers()?;
            let sample_slot = Arc::new(Mutex::new(None));
            let (command_sender, command_receiver) = mpsc::channel();
            let worker_slot = Arc::clone(&sample_slot);
            let host = Arc::clone(&self.host);
            let join_handle = thread::Builder::new()
                .name(String::from("srg-final-countdown-sampler"))
                .spawn(move || {
                    let mut network_context = NetworkContext::new(host, network_buffers);
                    let mut current_interval = initial_sample_interval;
                    loop {
                        loop {
                            match command_receiver.try_recv() {
                                Ok(interval) => current_interval = interval,
                                Err(mpsc::TryRecvError::Disconnected) => return,
                                Err(mpsc::TryRecvError::Empty) => break,
                            }
                        }
                        let sample_result = fetch_server_time_sample(&mut network_context);
                        let Ok(mut slot) = worker_slot.lock() else {
                            return;
                        };
                        *slot = Some(sample_result);
                        drop(slot);
                        match command_receiver.recv_timeout(current_interval) {
                            Ok(interval) => current_interval = interval,
                            Err(mpsc::RecvTimeoutError::Disconnected) => return,
                            Err(mpsc::RecvTimeoutError::Timeout) => {}
                        }
                    }
                })
                .map_err(TimeError::from)?;
            self.final_countdown_sampler = Some(FinalCountdownSamplerActive {
                command_sender,
                join_handle,
                sample_interval: initial_sample_interval,
                sample_slot,
            });
        } else {
            self.set_final_countdown_sample_interval(initial_sample_interval);
        }
        Ok(())
    }
    fn build_calibrated_server_time(&self, current_sample: TimeSample) -> Option<ServerTime> {
        let prev_sample = self.last_sample?;
        let baseline_rtt = self.baseline_rtt?;
        let prev_dur = prev_sample.server_time.duration_since(UNIX_EPOCH).ok()?;
        let current_dur = current_sample.server_time.duration_since(UNIX_EPOCH).ok()?;
        if current_dur.as_secs().checked_sub(prev_dur.as_secs()) != Some(1) {
            return None;
        }
        let mut server_time_at_tick = current_sample.server_time;
        let since_epoch = server_time_at_tick.duration_since(UNIX_EPOCH).ok()?;
        let nanos_to_subtract = since_epoch.subsec_nanos();
        let nanos_to_subtract_duration = Duration::from_nanos(u64::from(nanos_to_subtract));
        server_time_at_tick = server_time_at_tick.checked_sub(nanos_to_subtract_duration)?;
        let one_way_delay = effective_one_way_delay(current_sample.rtt);
        let anchor_instant = current_sample
            .response_received_inst
            .checked_sub(one_way_delay)
            .unwrap_or(current_sample.response_received_inst);
        Some(ServerTime {
            anchor_time: server_time_at_tick,
            anchor_instant,
            baseline_rtt,
        })
    }
    fn confirm_pending_target_time(
        &mut self,
        server_time: ServerTime,
        target_second: u32,
        now: Instant,
        msg_buf: &mut String,
    ) -> Result<()> {
        let current_server_time = server_time.current_server_time_at(now);
        let since_epoch = current_server_time.duration_since(UNIX_EPOCH)?;
        let kst_epoch_secs = since_epoch
            .as_secs()
            .checked_add(KST_OFFSET_SECS_U64)
            .ok_or_else(|| TimeError::parse("KST 현재 시각 계산 중 overflow가 발생했습니다."))?;
        let current_kst_second = u32::try_from(kst_epoch_secs.rem_euclid(KST_SECONDS_PER_DAY_U64))
            .map_err(|source| TimeError::parse_with_source("KST 초 변환 실패", source))?;
        let current_kst_day = kst_epoch_secs.div_euclid(KST_SECONDS_PER_DAY_U64);
        let target_day = if target_second < current_kst_second {
            current_kst_day.checked_add(1).ok_or_else(|| {
                TimeError::parse("다음날 목표 날짜 계산 중 overflow가 발생했습니다.")
            })?
        } else {
            current_kst_day
        };
        let target_kst_epoch_secs = target_day
            .checked_mul(KST_SECONDS_PER_DAY_U64)
            .and_then(|day_start| day_start.checked_add(u64::from(target_second)))
            .ok_or_else(|| TimeError::parse("목표 시각 계산 중 overflow가 발생했습니다."))?;
        let target_utc_epoch_secs = target_kst_epoch_secs
            .checked_sub(KST_OFFSET_SECS_U64)
            .ok_or_else(|| TimeError::parse("목표 UTC 시각 계산 중 underflow가 발생했습니다."))?;
        let target_system_time = UNIX_EPOCH
            .checked_add(Duration::from_secs(target_utc_epoch_secs))
            .ok_or_else(|| TimeError::parse("목표 절대 시각 계산 중 범위 오류가 발생했습니다."))?;
        let day_index = i32::try_from(target_day)
            .map_err(|source| TimeError::parse_with_source("목표 날짜 변환 실패", source))?;
        let CivilDate { day, month, year } = http_date::civil_from_days(day_index)
            .ok_or_else(|| TimeError::parse("목표 날짜 계산 중 범위 오류가 발생했습니다."))?;
        let hour = target_second.div_euclid(KST_SECONDS_PER_HOUR_U32);
        let minute = target_second
            .rem_euclid(KST_SECONDS_PER_HOUR_U32)
            .div_euclid(KST_SECONDS_PER_MINUTE_U32);
        let second = target_second.rem_euclid(KST_SECONDS_PER_MINUTE_U32);
        self.pending_target_time = None;
        self.target_time = Some(target_system_time);
        append_fmt(
            msg_buf,
            format_args!(
                "\n목표 시각 확정(KST): {year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}"
            ),
        );
        Ok(())
    }
    const fn discard_partial_baseline_measurement(&mut self) {
        self.baseline_rtt_attempts = 0;
        self.calibration_started_at = None;
        self.pending_sample_generation = None;
    }
    fn end_final_countdown_sampling(&mut self) {
        self.final_countdown_sampler = None;
        self.final_countdown_last_sample_error_message_at = None;
    }
    fn ensure_sample_request(&mut self, kind: SampleRequestKind) -> Result<()> {
        if self.pending_sample_generation.is_some() {
            return Ok(());
        }
        match self.sample_worker.fetch(kind, Arc::clone(&self.host)) {
            Ok(request) => {
                self.pending_sample_generation = request;
                Ok(())
            }
            Err(fetch_err) => {
                if let Err(respawn_err) = self.respawn_sample_worker() {
                    return Err(TimeError::parse_with_source(
                        format!("서버 시간 샘플 요청 실패: {fetch_err}; worker 재시작 실패"),
                        respawn_err,
                    ));
                }
                Err(TimeError::parse(format!(
                    "서버 시간 샘플 요청 실패: {fetch_err}; worker를 다시 시작했습니다."
                )))
            }
        }
    }
    fn finish_baseline_rtt_measurement<'message>(
        &mut self,
        msg_buf: &'message mut String,
    ) -> ActivityTransition<'message> {
        self.baseline_rtt_attempts = 0;
        let sample_count = self.baseline_rtt_samples.iter().flatten().count();
        if sample_count == 0 {
            return transition_to_retry("유효한 RTT 샘플을 얻지 못했습니다.");
        }
        let mut rtt_nanos = [u128::MAX; NUM_SAMPLES];
        for (slot, sample) in rtt_nanos
            .iter_mut()
            .zip(self.baseline_rtt_samples.iter().flatten())
        {
            *slot = sample.rtt.as_nanos();
        }
        rtt_nanos.sort_unstable();
        let trim = sample_count.div_euclid(RTT_TRIM_DIVISOR);
        let trimmed_sample_count = sample_count.saturating_sub(trim.saturating_mul(2));
        let (sum_nanos, averaged_sample_count) = rtt_nanos
            .iter()
            .skip(trim)
            .take(trimmed_sample_count)
            .fold((0_u128, 0_u128), |(sum, count), &sample_nanos| {
                (sum.saturating_add(sample_nanos), count.saturating_add(1))
            });
        let avg_nanos = sum_nanos.div_euclid(averaged_sample_count);
        let baseline_rtt = Duration::from_nanos_u128(avg_nanos);
        self.baseline_rtt = Some(baseline_rtt);
        self.calibration_started_at = None;
        append_fmt(
            msg_buf,
            format_args!(
                "[완료] RTT 기준값: {}ms. 2단계: 정밀 보정을 시작합니다.",
                baseline_rtt.as_millis()
            ),
        );
        ActivityTransition {
            activity: Activity::CalibrateOnTick,
            message: Some(msg_buf),
        }
    }
    fn handle_calibrate_on_tick<'message>(
        &mut self,
        msg_buf: &'message mut String,
    ) -> ActivityTransition<'message> {
        let now = Instant::now();
        let started_at = *self.calibration_started_at.get_or_insert(now);
        if now.saturating_duration_since(started_at) >= CALIBRATION_TIMEOUT {
            return transition_to_retry(CALIBRATION_TIMEOUT_MESSAGE);
        }
        let sample_result = match self.poll_sample_request() {
            Ok(SampleRequestPoll::Sample {
                kind: SampleRequestKind::Calibration,
                result,
            }) => result,
            Ok(SampleRequestPoll::Sample { .. } | SampleRequestPoll::Empty) => {
                if let Err(start_err) = self.ensure_sample_request(SampleRequestKind::Calibration) {
                    append_error_detail(msg_buf, "정밀 보정 샘플 요청 실패: ", start_err);
                    return transition_to_retry(msg_buf);
                }
                return ActivityTransition {
                    activity: Activity::CalibrateOnTick,
                    message: None,
                };
            }
            Err(worker_err) => {
                append_error_detail(msg_buf, "서버 시간 샘플 worker 오류: ", worker_err);
                return transition_to_retry(msg_buf);
            }
        };
        let completed_at = Instant::now();
        if completed_at.saturating_duration_since(started_at) >= CALIBRATION_TIMEOUT {
            return transition_to_retry(CALIBRATION_TIMEOUT_MESSAGE);
        }
        let Ok(current_sample) = sample_result else {
            if let Err(start_err) = self.ensure_sample_request(SampleRequestKind::Calibration) {
                append_error_detail(msg_buf, "정밀 보정 샘플 요청 실패: ", start_err);
                return transition_to_retry(msg_buf);
            }
            return ActivityTransition {
                activity: Activity::CalibrateOnTick,
                message: None,
            };
        };
        if let Some(server_time) = self.accept_calibration_tick(current_sample) {
            if let Some(target_second) = self
                .pending_target_time
                .as_ref()
                .map(|target_time| target_time.seconds_after_midnight)
            {
                msg_buf.push_str("[성공] 정밀 보정 완료!");
                if let Err(target_err) = self.confirm_pending_target_time(
                    server_time,
                    target_second,
                    Instant::now(),
                    msg_buf,
                ) {
                    append_error_detail(msg_buf, "\n목표 시각 확정 실패: ", target_err);
                    return transition_to_retry(msg_buf);
                }
                return ActivityTransition {
                    activity: Activity::Predicting { server_time },
                    message: Some(msg_buf),
                };
            }
            return ActivityTransition {
                activity: Activity::Predicting { server_time },
                message: Some("[성공] 정밀 보정 완료!"),
            };
        }
        self.last_sample = Some(current_sample);
        if Instant::now().saturating_duration_since(started_at) >= CALIBRATION_TIMEOUT {
            return transition_to_retry(CALIBRATION_TIMEOUT_MESSAGE);
        }
        if let Err(start_err) = self.ensure_sample_request(SampleRequestKind::Calibration) {
            append_error_detail(msg_buf, "정밀 보정 샘플 요청 실패: ", start_err);
            return transition_to_retry(msg_buf);
        }
        ActivityTransition {
            activity: Activity::CalibrateOnTick,
            message: None,
        }
    }
    fn handle_final_countdown<'message>(
        &mut self,
        server_time: ServerTime,
        target_time: SystemTime,
        msg_buf: &'message mut String,
        runtime: &mut LoopRuntime<'_>,
    ) -> ActivityTransition<'message> {
        let now = Instant::now();
        let current_server_time = server_time.current_server_time_at(now);
        let sample_poll = self.poll_final_countdown_sample();
        let sample_error_reported = match sample_poll {
            FinalCountdownSamplePoll::Sample(Ok(sample)) => {
                return self.handle_final_countdown_sample(
                    server_time,
                    target_time,
                    msg_buf,
                    sample,
                    runtime,
                );
            }
            FinalCountdownSamplePoll::Sample(Err(fetch_err)) => {
                self.append_final_countdown_sample_error(now, msg_buf, fetch_err)
            }
            FinalCountdownSamplePoll::Disconnected => self.append_final_countdown_sample_error(
                now,
                msg_buf,
                "카운트다운 샘플러가 종료되었습니다. 기존 RTT 예측으로 계속 진행합니다.",
            ),
            FinalCountdownSamplePoll::Empty => false,
        };
        let estimated_rtt = self.live_rtt.unwrap_or(server_time.baseline_rtt);
        let estimated_one_way_delay = effective_one_way_delay(estimated_rtt);
        let Some(trigger_instant) =
            trigger_instant_for_target(server_time, target_time, estimated_one_way_delay, now)
        else {
            msg_buf.push_str("카운트다운 실행 시각 계산 실패");
            return ActivityTransition {
                activity: Activity::FinalCountdown {
                    server_time,
                    target_time,
                },
                message: Some(msg_buf),
            };
        };
        if trigger_instant.saturating_duration_since(now) <= FINAL_COUNTDOWN_FREEZE_WINDOW {
            let deadline = FinalCountdownDeadline {
                one_way_delay: estimated_one_way_delay,
                server_time,
                source: CountdownTriggerSource::Estimated,
                target_time,
                trigger_instant,
            };
            return self.trigger_final_countdown_deadline(&deadline, msg_buf, runtime);
        }
        if let Ok(duration_until_target) = target_time.duration_since(current_server_time) {
            self.set_final_countdown_sample_interval(final_countdown_sample_interval(
                duration_until_target,
            ));
        }
        if sample_error_reported {
            return ActivityTransition {
                activity: Activity::FinalCountdown {
                    server_time,
                    target_time,
                },
                message: Some(msg_buf),
            };
        }
        let sample_interval = target_time.duration_since(current_server_time).map_or(
            FINAL_COUNTDOWN_SAMPLE_FINAL_INTERVAL,
            final_countdown_sample_interval,
        );
        if let Err(start_err) = self.begin_final_countdown_sampling(sample_interval) {
            self.end_final_countdown_sampling();
            append_error_detail(msg_buf, "카운트다운 샘플러 시작 실패: ", start_err);
            return ActivityTransition {
                activity: Activity::FinalCountdown {
                    server_time,
                    target_time,
                },
                message: Some(msg_buf),
            };
        }
        ActivityTransition {
            activity: Activity::FinalCountdown {
                server_time,
                target_time,
            },
            message: None,
        }
    }
    fn handle_final_countdown_sample<'message>(
        &mut self,
        server_time: ServerTime,
        target_time: SystemTime,
        msg_buf: &'message mut String,
        sample: TimeSample,
        runtime: &mut LoopRuntime<'_>,
    ) -> ActivityTransition<'message> {
        let sample_rtt = sample.rtt;
        let calibrated_server_time = server_time.recalibrate_with_rtt(sample_rtt);
        let sample_now = Instant::now();
        let old_rtt = self.live_rtt.unwrap_or(sample_rtt);
        let live_rtt = blend_rtt(old_rtt, sample_rtt, NewSampleWeight::SeventyPercent);
        self.live_rtt = Some(live_rtt);
        let effective_rtt = live_rtt.max(sample_rtt);
        let one_way_delay = effective_one_way_delay(effective_rtt);
        let Some(trigger_instant) = trigger_instant_for_target(
            calibrated_server_time,
            target_time,
            one_way_delay,
            sample_now,
        ) else {
            msg_buf.push_str("카운트다운 실행 시각 계산 실패");
            return ActivityTransition {
                activity: Activity::FinalCountdown {
                    server_time: calibrated_server_time,
                    target_time,
                },
                message: Some(msg_buf),
            };
        };
        if trigger_instant.saturating_duration_since(sample_now) <= FINAL_COUNTDOWN_FREEZE_WINDOW {
            let deadline = FinalCountdownDeadline {
                one_way_delay,
                server_time: calibrated_server_time,
                source: CountdownTriggerSource::Sampled { rtt: sample_rtt },
                target_time,
                trigger_instant,
            };
            return self.trigger_final_countdown_deadline(&deadline, msg_buf, runtime);
        }
        ActivityTransition {
            activity: Activity::FinalCountdown {
                server_time: calibrated_server_time,
                target_time,
            },
            message: None,
        }
    }
    fn handle_measure_baseline_rtt<'message>(
        &mut self,
        msg_buf: &'message mut String,
        now: Instant,
        out: &mut dyn io::Write,
    ) -> ActivityTransition<'message> {
        if self.baseline_rtt_attempts == 0 && self.pending_sample_generation.is_none() {
            self.begin_baseline_rtt_measurement(now, out);
        }
        let had_pending_request = self.pending_sample_generation.is_some();
        let sample_result = match self.poll_sample_request() {
            Ok(SampleRequestPoll::Sample {
                kind: SampleRequestKind::Baseline { attempt_index },
                result,
            }) => (attempt_index, result),
            Ok(SampleRequestPoll::Sample { .. } | SampleRequestPoll::Empty) => {
                if had_pending_request || now < self.baseline_rtt_next_sample_at {
                    return ActivityTransition {
                        activity: Activity::MeasureBaselineRtt,
                        message: None,
                    };
                }
                let attempt_index = self.baseline_rtt_attempts;
                let start_result =
                    self.ensure_sample_request(SampleRequestKind::Baseline { attempt_index });
                if let Err(start_err) = start_result {
                    append_error_detail(msg_buf, "RTT 샘플 요청 실패: ", start_err);
                    return transition_to_retry(msg_buf);
                }
                return ActivityTransition {
                    activity: Activity::MeasureBaselineRtt,
                    message: None,
                };
            }
            Err(worker_err) => {
                append_error_detail(msg_buf, "서버 시간 샘플 worker 오류: ", worker_err);
                return transition_to_retry(msg_buf);
            }
        };
        let (attempt_index, result) = sample_result;
        let next_sample_base = match result {
            Ok(sample) => {
                if sample.rtt > Duration::ZERO {
                    let Some(slot) = self.baseline_rtt_samples.get_mut(attempt_index) else {
                        return transition_to_retry("RTT 샘플 인덱스 계산 실패.");
                    };
                    *slot = Some(sample);
                    self.last_sample = Some(sample);
                }
                sample.response_received_inst
            }
            Err(err) => {
                self.baseline_rtt_attempts = 0;
                append_error_detail(msg_buf, "RTT 샘플 수집 실패: ", err);
                return transition_to_retry(msg_buf);
            }
        };
        self.baseline_rtt_attempts = attempt_index.saturating_add(1);
        let Some(next_sample_at) = next_sample_base.checked_add(ADAPTIVE_POLL_INTERVAL) else {
            return transition_to_retry("다음 RTT 샘플 시각 계산 실패.");
        };
        self.baseline_rtt_next_sample_at = next_sample_at;
        if self.baseline_rtt_attempts < NUM_SAMPLES {
            return ActivityTransition {
                activity: Activity::MeasureBaselineRtt,
                message: None,
            };
        }
        self.finish_baseline_rtt_measurement(msg_buf)
    }
    fn handle_predicting<'message>(
        &mut self,
        server_time: ServerTime,
        msg_buf: &'message mut String,
        now: Instant,
    ) -> ActivityTransition<'message> {
        let estimated_server_time = server_time.current_server_time_at(now);
        let target_remaining = self
            .target_time
            .and_then(|target| target.duration_since(estimated_server_time).ok());
        let protect_target =
            target_remaining.is_some_and(|remaining| remaining <= FINAL_COUNTDOWN_WARMUP_WINDOW);
        if let Some(warmup_remaining) = target_remaining
            && warmup_remaining <= FINAL_COUNTDOWN_WARMUP_WINDOW
        {
            let sample_interval = final_countdown_sample_interval(warmup_remaining);
            if let Err(start_err) = self.begin_final_countdown_sampling(sample_interval) {
                self.end_final_countdown_sampling();
                append_error_detail(msg_buf, "카운트다운 샘플러 시작 실패: ", start_err);
                return ActivityTransition {
                    activity: Activity::Predicting { server_time },
                    message: Some(msg_buf),
                };
            }
        }
        if now.saturating_duration_since(self.last_full_sync_at) >= FULL_SYNC_INTERVAL
            && !protect_target
        {
            self.end_final_countdown_sampling();
            self.baseline_rtt = None;
            self.live_rtt = None;
            ActivityTransition {
                activity: Activity::MeasureBaselineRtt,
                message: Some("서버 시간 보정 주기 도래, 재보정 시작."),
            }
        } else {
            ActivityTransition {
                activity: Activity::Predicting { server_time },
                message: None,
            }
        }
    }
    fn maybe_start_final_countdown(
        &mut self,
        server_time: ServerTime,
        now: Instant,
    ) -> Option<ActivityTransition<'static>> {
        let estimated_server_time = server_time.current_server_time_at(now);
        let target_time = self.target_time.take_if(|target| {
            target
                .duration_since(estimated_server_time)
                .ok()
                .is_none_or(|remaining| remaining <= FINAL_COUNTDOWN_WINDOW)
        })?;
        if self.live_rtt.is_none() {
            self.live_rtt = Some(server_time.baseline_rtt);
        }
        self.discard_partial_baseline_measurement();
        Some(ActivityTransition {
            activity: Activity::FinalCountdown {
                server_time,
                target_time,
            },
            message: Some("최종 카운트다운 시작!"),
        })
    }
    fn next_activity<'message>(
        &mut self,
        activity: &Activity,
        message_buffer: &'message mut String,
        now: Instant,
        runtime: &mut LoopRuntime<'_>,
    ) -> ActivityTransition<'message> {
        if let Activity::Predicting { server_time } = *activity
            && let Some(transition) = self.maybe_start_final_countdown(server_time, now)
        {
            return transition;
        }
        match *activity {
            Activity::MeasureBaselineRtt => {
                self.handle_measure_baseline_rtt(message_buffer, now, runtime.out)
            }
            Activity::CalibrateOnTick => self.handle_calibrate_on_tick(message_buffer),
            Activity::Predicting { server_time } => {
                self.handle_predicting(server_time, message_buffer, now)
            }
            Activity::FinalCountdown {
                server_time,
                target_time,
            } => self.handle_final_countdown(server_time, target_time, message_buffer, runtime),
            Activity::Retrying { started_at } => {
                if now.saturating_duration_since(started_at) >= RETRY_DELAY {
                    ActivityTransition {
                        activity: Activity::MeasureBaselineRtt,
                        message: Some("[재시도] 동기화를 다시 시작합니다."),
                    }
                } else {
                    ActivityTransition {
                        activity: Activity::Retrying { started_at },
                        message: None,
                    }
                }
            }
        }
    }
    fn poll_final_countdown_sample(&mut self) -> FinalCountdownSamplePoll {
        let Some(active) = self.final_countdown_sampler.as_mut() else {
            return FinalCountdownSamplePoll::Empty;
        };
        let sampler_finished = active.join_handle.is_finished();
        let (latest_result, poisoned) = match active.sample_slot.try_lock() {
            Ok(mut slot) => (slot.take(), false),
            Err(TryLockError::WouldBlock) => {
                if sampler_finished {
                    (None, false)
                } else {
                    return FinalCountdownSamplePoll::Empty;
                }
            }
            Err(TryLockError::Poisoned(_)) => (None, true),
        };
        if sampler_finished || poisoned {
            self.final_countdown_sampler = None;
        }
        if poisoned {
            return FinalCountdownSamplePoll::Sample(Err(TimeError::parse(
                "카운트다운 샘플 상태 잠금 실패",
            )));
        }
        if let Some(sample_result) = latest_result {
            return FinalCountdownSamplePoll::Sample(sample_result);
        }
        if sampler_finished {
            return FinalCountdownSamplePoll::Disconnected;
        }
        FinalCountdownSamplePoll::Empty
    }
    fn poll_sample_request(&mut self) -> Result<SampleRequestPoll> {
        let Some(pending_generation) = self.pending_sample_generation else {
            loop {
                match self.sample_worker.response_receiver.try_recv() {
                    Ok(_) => {}
                    Err(mpsc::TryRecvError::Empty) => return Ok(SampleRequestPoll::Empty),
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.respawn_sample_worker()?;
                        return Err(TimeError::parse(SAMPLE_WORKER_RESTARTED_MESSAGE));
                    }
                }
            }
        };
        loop {
            match self.sample_worker.response_receiver.try_recv() {
                Ok(response) => {
                    if response.generation == pending_generation {
                        self.pending_sample_generation = None;
                        return Ok(SampleRequestPoll::Sample {
                            kind: response.kind,
                            result: response.result,
                        });
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    if self.sample_worker.join_handle.is_finished() {
                        self.pending_sample_generation = None;
                        self.respawn_sample_worker()?;
                        return Err(TimeError::parse(SAMPLE_WORKER_RESTARTED_MESSAGE));
                    }
                    return Ok(SampleRequestPoll::Empty);
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.pending_sample_generation = None;
                    self.respawn_sample_worker()?;
                    return Err(TimeError::parse(SAMPLE_WORKER_RESTARTED_MESSAGE));
                }
            }
        }
    }
    fn respawn_sample_worker(&mut self) -> Result<()> {
        *self.sample_worker = spawn_sample_worker(Arc::clone(&self.host))?;
        Ok(())
    }
    fn run_loop(
        &mut self,
        stop_after: Option<Duration>,
        out: &mut dyn io::Write,
        err: &mut dyn io::Write,
    ) -> Result<()> {
        if let Some(duration) = stop_after {
            let stop_at = Instant::now()
                .checked_add(duration)
                .ok_or_else(|| TimeError::parse("관찰 종료 시각 계산 실패"))?;
            #[cfg(target_os = "linux")]
            let (mut prepared_input, stop_requested) = {
                let mut prepared_input = wayland_input::PreparedInput::EMPTY;
                let stop_requested =
                    prepared_input.prepare(self.trigger_action, err, || Instant::now() >= stop_at);
                (prepared_input, stop_requested)
            };
            #[cfg(not(target_os = "linux"))]
            let stop_requested = false;
            if !stop_requested {
                cfg_select! {
                    target_os = "linux" => {
                        self.run_loop_active(LoopStop::Deadline(stop_at), out, err, &mut prepared_input)?;
                    }
                    _ => {
                        self.run_loop_active(LoopStop::Deadline(stop_at), out, err)?;
                    }
                }
            }
            return Ok(());
        }
        let (tx, rx) = mpsc::channel();
        let input_thread = thread::spawn(move || -> IoResult<()> {
            let mut line = Vec::new();
            line.try_reserve_exact(ENTER_BUFFER_READ_LIMIT_BYTES)
                .map_err(io::Error::other)?;
            let mut stdin_lock = io::stdin().lock();
            let read_limit =
                u64::try_from(ENTER_BUFFER_READ_LIMIT_BYTES).map_err(IoError::other)?;
            {
                let mut limited_stdin = IoRead::take(&mut stdin_lock, read_limit);
                IoBufRead::read_until(&mut limited_stdin, b'\n', &mut line)?;
            }
            if line.len() > ENTER_BUFFER_CAPACITY && !line.ends_with(b"\n") {
                IoBufRead::skip_until(&mut stdin_lock, b'\n')?;
                return Err(IoError::new(
                    io::ErrorKind::InvalidInput,
                    ENTER_INPUT_TOO_LONG,
                ));
            }
            tx.send(())
                .map_err(|_source| IoError::other("서버 시간 루프 종료 알림 실패"))?;
            Ok(())
        });
        #[cfg(target_os = "linux")]
        let (mut prepared_input, stop_requested) = {
            let mut prepared_input = wayland_input::PreparedInput::EMPTY;
            let stop_requested = prepared_input.prepare(self.trigger_action, err, || {
                !matches!(rx.try_recv(), Err(mpsc::TryRecvError::Empty))
            });
            (prepared_input, stop_requested)
        };
        #[cfg(not(target_os = "linux"))]
        let stop_requested = false;
        if !stop_requested {
            cfg_select! {
                target_os = "linux" => {
                    self.run_loop_active(LoopStop::Receiver(&rx), out, err, &mut prepared_input)?;
                }
                _ => {
                    self.run_loop_active(LoopStop::Receiver(&rx), out, err)?;
                }
            }
        }
        input_thread
            .join()
            .map_err(|_panic_payload| TimeError::parse(ENTER_THREAD_PANIC))?
            .map_err(|source| TimeError::parse_with_source("종료 입력 실패", source))?;
        Ok(())
    }
    fn run_loop_active(
        &mut self,
        stop: LoopStop<'_>,
        out: &mut dyn io::Write,
        err: &mut dyn io::Write,
        #[cfg(target_os = "linux")] prepared_input: &mut wayland_input::PreparedInput,
    ) -> Result<()> {
        let mut message_buffer = String::new();
        message_buffer
            .try_reserve_exact(MESSAGE_BUFFER_CAPACITY)
            .map_err(|source| TimeError::parse_with_source("buffer 메모리 확보 실패", source))?;
        let mut activity = Activity::MeasureBaselineRtt;
        let mut last_display_update = Instant::now();
        let mut line_buf = [0_u8; display::DISPLAY_LINE_BUF_LEN];
        loop {
            let pre_wait_now = Instant::now();
            if matches!(stop, LoopStop::Deadline(deadline) if pre_wait_now >= deadline) {
                break;
            }
            let activity_poll_timeout = activity.poll_interval();
            let mut poll_timeout = if self.should_update_display(&activity, pre_wait_now) {
                let elapsed = pre_wait_now.saturating_duration_since(last_display_update);
                let remaining_display = DISPLAY_INTERVAL.saturating_sub(elapsed);
                activity_poll_timeout.min(remaining_display)
            } else {
                activity_poll_timeout
            };
            if let LoopStop::Deadline(deadline) = stop {
                poll_timeout = poll_timeout.min(deadline.saturating_duration_since(pre_wait_now));
            }
            match stop {
                LoopStop::Receiver(receiver) => match receiver.recv_timeout(poll_timeout) {
                    Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                },
                LoopStop::Deadline(_) => thread::sleep(poll_timeout),
            }
            cfg_select! {
                target_os = "linux" => {
                    prepared_input.maintain(err);
                }
                _ => {}
            }
            let now = Instant::now();
            let timing_sensitive_activity =
                self.should_prioritize_activity_transition(&activity, now);
            if !timing_sensitive_activity {
                self.run_loop_write_display_if_due(
                    &activity,
                    now,
                    &mut last_display_update,
                    out,
                    line_buf.as_mut_slice(),
                )?;
            }
            message_buffer.clear();
            let transition = {
                let mut runtime = LoopRuntime {
                    err,
                    out,
                    #[cfg(target_os = "linux")]
                    prepared_input,
                };
                self.next_activity(&activity, &mut message_buffer, now, &mut runtime)
            };
            let next_activity = transition.activity;
            self.sync_sample_request_state(&next_activity);
            cfg_select! {
                windows => {
                    self.sync_high_res_timer_state(&next_activity, err);
                }
                _ => {}
            }
            let transition_had_message = if let Some(console_msg) = transition.message {
                writeln!(out, "\n{console_msg}")?;
                true
            } else {
                false
            };
            activity = next_activity;
            if timing_sensitive_activity && !transition_had_message {
                self.run_loop_write_display_if_due(
                    &activity,
                    Instant::now(),
                    &mut last_display_update,
                    out,
                    line_buf.as_mut_slice(),
                )?;
            }
        }
        Ok(())
    }
    fn run_loop_write_display_if_due(
        &self,
        display_activity: &Activity,
        now: Instant,
        last_update: &mut Instant,
        output: &mut dyn io::Write,
        buffer: &mut [u8],
    ) -> Result<()> {
        if now.saturating_duration_since(*last_update) >= DISPLAY_INTERVAL
            && self.should_update_display(display_activity, now)
            && let Some(server_time) = display_activity.server_time()
        {
            let mut cur = ByteCursor::new(buffer);
            let display_result = (|| -> Result<()> {
                cur.write_bytes(DISPLAY_STATUS_PREFIX.as_bytes())?;
                server_time.write_current_display_time_buf_at(&mut cur, now)?;
                cur.write_bytes(b" \r")?;
                Ok(())
            })();
            match display_result {
                Ok(()) => output.write_all(cur.written_slice()?)?,
                Err(display_err) => {
                    output.write_all(DISPLAY_STATUS_PREFIX.as_bytes())?;
                    write!(output, "표시 버퍼 오류: {display_err}")?;
                    output.write_all(b" \r")?;
                }
            }
            output.flush()?;
            *last_update = now;
        }
        Ok(())
    }
    fn set_final_countdown_sample_interval(&mut self, sample_interval: Duration) {
        let Some(active) = self.final_countdown_sampler.as_mut() else {
            return;
        };
        if active.sample_interval != sample_interval {
            active.sample_interval = sample_interval;
            if active.command_sender.send(sample_interval).is_err() {
                self.final_countdown_sampler = None;
            }
        }
    }
    fn should_prioritize_activity_transition(&self, activity: &Activity, now: Instant) -> bool {
        if activity.is_final_countdown() {
            return true;
        }
        let Activity::Predicting { server_time } = *activity else {
            return false;
        };
        self.target_time.is_some_and(|target| {
            target
                .duration_since(server_time.current_server_time_at(now))
                .ok()
                .is_none_or(|remaining| remaining <= FINAL_COUNTDOWN_WINDOW)
        })
    }
    fn should_update_display(&self, activity: &Activity, now: Instant) -> bool {
        let Some(server_time) = activity.server_time() else {
            return false;
        };
        let Activity::FinalCountdown { target_time, .. } = *activity else {
            return true;
        };
        let rtt = self.live_rtt.unwrap_or(server_time.baseline_rtt);
        let one_way_delay = effective_one_way_delay(rtt);
        let Some(trigger_instant) =
            trigger_instant_for_target(server_time, target_time, one_way_delay, now)
        else {
            return true;
        };
        trigger_instant.saturating_duration_since(now) > FINAL_COUNTDOWN_FREEZE_WINDOW
    }
    cfg_select! {
        windows => {
            fn sync_high_res_timer_state(&mut self, next_activity: &Activity, err: &mut dyn io::Write) {
                if !next_activity.is_final_countdown() {
                    self.high_res_timer_guard = None;
                    return;
                }
                if self.high_res_timer_guard.is_some() {
                    return;
                }
                // SAFETY: No security attributes or name are needed, and a successful handle is
                // transferred to the guard.
                let timer_handle = unsafe {
                    sys::create_waitable_timer_ex_w(
                        null(),
                        null(),
                        CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                        SYNCHRONIZE_ACCESS | TIMER_MODIFY_STATE_ACCESS,
                    )
                };
                let Some(handle) = NonNull::new(timer_handle) else {
                    write_line_best_effort(
                        err,
                        format_args!(concat!(
                            "[경고] Windows 고해상도 대기 타이머 생성에 실패했습니다. ",
                            "카운트다운 정확도가 저하될 수 있습니다."
                        )),
                    );
                    return;
                };
                let timer_guard = HighResTimerGuard { handle };
                self.high_res_timer_guard = Some(timer_guard);
            }
        }
        _ => {}
    }
    const fn sync_sample_request_state(&mut self, next_activity: &Activity) {
        if !matches!(
            *next_activity,
            Activity::MeasureBaselineRtt | Activity::CalibrateOnTick
        ) {
            self.pending_sample_generation = None;
        }
        if !matches!(*next_activity, Activity::CalibrateOnTick) {
            self.calibration_started_at = None;
        }
    }
    fn trigger_and_finish<'message>(
        &self,
        server_time: ServerTime,
        msg_buf: &'message mut String,
        timing_detail: fmt::Arguments<'_>,
        runtime: &mut LoopRuntime<'_>,
    ) -> ActivityTransition<'message> {
        let send_status = cfg_select! {
            target_os = "linux" => {
                if self.trigger_action.is_some() {
                    runtime.prepared_input.send(runtime.err)
                } else {
                    NativeInputSendStatus::Sent
                }
            }
            any(target_os = "windows", target_os = "macos") => {
                self.trigger_action.map_or(NativeInputSendStatus::Sent, |action| {
                    action.send(runtime.err)
                })
            }
            _ => {
                compile_error!("SRG native input supports only Windows, Linux, and macOS.")
            }
        };
        match send_status {
            NativeInputSendStatus::Sent => msg_buf.push_str("\n>>> 액션 실행! "),
            NativeInputSendStatus::FailedBeforeSend => msg_buf.push_str("\n>>> 액션 실행 실패! "),
            NativeInputSendStatus::PartialOrUnknown => {
                msg_buf.push_str("\n>>> 액션 전송 상태 불확실! ");
            }
        }
        append_fmt(msg_buf, timing_detail);
        ActivityTransition {
            activity: Activity::Predicting { server_time },
            message: Some(msg_buf),
        }
    }
    fn trigger_countdown<'message>(
        &self,
        trigger: CountdownTrigger,
        server_time: ServerTime,
        msg_buf: &'message mut String,
        one_way_delay: Duration,
        source: CountdownTriggerSource,
        runtime: &mut LoopRuntime<'_>,
    ) -> ActivityTransition<'message> {
        match (trigger, source) {
            (
                CountdownTrigger::WithRemaining(remaining),
                CountdownTriggerSource::Estimated,
            ) => self.trigger_and_finish(
                server_time,
                msg_buf,
                format_args!(
                    "(예측값 기준, 목표 도달까지 {:.1}ms 남음) (지연 예측: {:.1}ms)",
                    duration_millis_f64(remaining),
                    duration_millis_f64(one_way_delay),
                ),
                runtime,
            ),
            (CountdownTrigger::Late(late_by), CountdownTriggerSource::Estimated) => self
                .trigger_and_finish(
                    server_time,
                    msg_buf,
                    format_args!(
                        "(예측값 기준, 목표 초가 이미 시작되어 즉시 액션을 실행했습니다. ({:.1}ms 지연, 지연 예측: {:.1}ms))",
                        duration_millis_f64(late_by),
                        duration_millis_f64(one_way_delay),
                    ),
                    runtime,
                ),
            (
                CountdownTrigger::WithRemaining(remaining),
                CountdownTriggerSource::Sampled { rtt },
            ) => self.trigger_and_finish(
                server_time,
                msg_buf,
                format_args!(
                    "(목표 도달까지 {:.1}ms 남음) (지연 예측: {:.1}ms, 실측 RTT: {:.1}ms)",
                    duration_millis_f64(remaining),
                    duration_millis_f64(one_way_delay),
                    duration_millis_f64(rtt)
                ),
                runtime,
            ),
            (CountdownTrigger::Late(late_by), CountdownTriggerSource::Sampled { rtt }) => {
                self.trigger_and_finish(
                    server_time,
                    msg_buf,
                    format_args!(
                        "(목표 초가 이미 시작되어 즉시 액션을 실행했습니다. ({:.1}ms 지연, 지연 예측: {:.1}ms, 실측 RTT: {:.1}ms))",
                        duration_millis_f64(late_by),
                        duration_millis_f64(one_way_delay),
                        duration_millis_f64(rtt)
                    ),
                    runtime,
                )
            }
        }
    }
    fn trigger_final_countdown_deadline<'message>(
        &mut self,
        deadline: &FinalCountdownDeadline,
        msg_buf: &'message mut String,
        runtime: &mut LoopRuntime<'_>,
    ) -> ActivityTransition<'message> {
        self.end_final_countdown_sampling();
        let now = Instant::now();
        if deadline.trigger_instant > now {
            if let Some(sleep_until) = deadline
                .trigger_instant
                .checked_sub(FINAL_COUNTDOWN_SLEEP_MARGIN)
                && sleep_until > now
            {
                let sleep_duration = sleep_until.duration_since(now);
                cfg_select! {
                    target_os = "windows" => {
                        if let Some(guard) = self.high_res_timer_guard.as_ref() {
                            guard.sleep(sleep_duration);
                        } else {
                            thread::sleep(sleep_duration);
                        }
                    }
                    _ => {
                        thread::sleep(sleep_duration);
                    }
                }
            }
            while Instant::now() < deadline.trigger_instant {
                spin_loop();
            }
        }
        let trigger_now = Instant::now();
        let trigger_server_time = deadline.server_time.current_server_time_at(trigger_now);
        let trigger = match deadline.target_time.duration_since(trigger_server_time) {
            Ok(remaining) => CountdownTrigger::WithRemaining(remaining),
            Err(late) => CountdownTrigger::Late(late.duration()),
        };
        self.trigger_countdown(
            trigger,
            deadline.server_time,
            msg_buf,
            deadline.one_way_delay,
            deadline.source,
            runtime,
        )
    }
}
fn spawn_sample_worker(initial_host: Arc<ParsedServer>) -> Result<SampleWorker> {
    let network_buffers = try_network_buffers()?;
    let (command_sender, command_receiver) = mpsc::sync_channel(0);
    let (response_sender, response_receiver) = mpsc::channel();
    let join_handle = thread::Builder::new()
        .name(String::from("srg-sample-worker"))
        .spawn(move || {
            let mut network_context = NetworkContext::new(initial_host, network_buffers);
            while let Ok(SampleWorkerRequest {
                generation,
                host: request_host,
                kind,
            }) = command_receiver.recv()
            {
                if !Arc::ptr_eq(&network_context.host, &request_host) {
                    network_context.reset_host(request_host);
                }
                let result = fetch_server_time_sample(&mut network_context);
                if response_sender
                    .send(SampleWorkerResponse {
                        generation,
                        kind,
                        result,
                    })
                    .is_err()
                {
                    return;
                }
            }
        })?;
    Ok(SampleWorker {
        command_sender,
        generation: 0,
        join_handle,
        response_receiver,
    })
}
fn try_network_buffers() -> Result<(Vec<u8>, Vec<u8>)> {
    let mut line = Vec::new();
    line.try_reserve_exact(TCP_LINE_BUFFER_CAPACITY)
        .map_err(|source| TimeError::parse_with_source("buffer 메모리 확보 실패", source))?;
    let mut request = Vec::new();
    request
        .try_reserve_exact(TCP_LINE_BUFFER_CAPACITY)
        .map_err(|source| TimeError::parse_with_source("buffer 메모리 확보 실패", source))?;
    Ok((line, request))
}
fn duration_millis_f64(duration: Duration) -> f64 {
    NumericMul::mul(duration.as_secs_f64(), MILLIS_PER_SECOND_F64)
}
fn append_error_detail(target: &mut String, prefix: &str, err: impl fmt::Display) {
    append_fmt(target, format_args!("{prefix}{err}"));
}
fn effective_one_way_delay(rtt: Duration) -> Duration {
    Duration::from_nanos_u128(rtt.as_nanos().div_euclid(u128::from(HALF_RTT_DIVISOR)))
}
fn final_countdown_sample_interval(duration_until_target: Duration) -> Duration {
    if duration_until_target <= Duration::from_secs(1) {
        FINAL_COUNTDOWN_SAMPLE_FINAL_INTERVAL
    } else if duration_until_target <= Duration::from_secs(3) {
        FINAL_COUNTDOWN_SAMPLE_FAST_INTERVAL
    } else if duration_until_target <= FINAL_COUNTDOWN_WINDOW {
        FINAL_COUNTDOWN_SAMPLE_NORMAL_INTERVAL
    } else {
        FINAL_COUNTDOWN_SAMPLE_WARMUP_INTERVAL
    }
}
fn trigger_instant_for_target(
    server_time: ServerTime,
    target_time: SystemTime,
    one_way_delay: Duration,
    now: Instant,
) -> Option<Instant> {
    let Ok(target_delta) = target_time.duration_since(server_time.anchor_time) else {
        return Some(now);
    };
    let target_instant = server_time.anchor_instant.checked_add(target_delta)?;
    Some(target_instant.checked_sub(one_way_delay).unwrap_or(now))
}
fn append_fmt(target: &mut String, args: fmt::Arguments<'_>) {
    match FmtWrite::write_fmt(target, args) {
        Ok(()) | Err(_) => {}
    }
}
fn transition_to_retry(msg: &str) -> ActivityTransition<'_> {
    ActivityTransition {
        activity: Activity::Retrying {
            started_at: Instant::now(),
        },
        message: Some(msg),
    }
}
