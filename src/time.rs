use self::{
    activity::{Activity, CountdownDecision},
    sample::TCP_LINE_BUFFER_CAPACITY,
    util::blend_weighted_nanos,
    worker::{spawn_final_countdown_sampler, spawn_sample_worker},
};
cfg_select! {
    target_os = "windows" => {
        use self::timer_resolution::{
            CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, HighResTimerGuard, SYNCHRONIZE_ACCESS,
            TARGET_PERIOD_MS, TIMER_MODIFY_STATE_ACCESS, TIMERR_NOERROR, sys,
        };
        use core::ptr::{NonNull, null};
    }
    _ => {}
}
use crate::{buffmt::ByteCursor, write_line_best_effort};
use alloc::{borrow::Cow, sync::Arc};
use core::{
    error::Error,
    fmt::{self, Write as FmtWrite},
    hint::spin_loop,
    ops::Mul as NumericMul,
    range::Range,
    result::Result as CoreResult,
    str::FromStr,
    time::Duration,
};
use std::{
    io::{
        self, BufRead as IoBufRead, Error as IoError, ErrorKind, Read as IoRead, Result as IoResult,
    },
    net,
    sync::{Mutex, TryLockError, mpsc},
    thread,
    time::{Instant, SystemTime, SystemTimeError, UNIX_EPOCH},
};
cfg_select! {
    target_os = "linux" => {
        use self::linux_input as native_input;
    }
    target_os = "macos" => {
        use self::macos_input as native_input;
    }
    target_os = "windows" => {
        use self::windows_input as native_input;
    }
    _ => {
        mod native_input {
            use super::NativeInputSendStatus;
            use std::io;
            #[derive(Clone, Copy)]
            enum InputAction {
                F5Press,
                MouseClick,
            }
            struct PreparedInput;
            impl PreparedInput {
                const EMPTY: Self = Self;
                fn prepare(
                    &mut self,
                    _action: Option<InputAction>,
                    _err: &mut dyn io::Write,
                ) {
                    *self = Self;
                }
                const fn reset(&mut self) {
                    *self = Self;
                }
                fn send(
                    &mut self,
                    _action: InputAction,
                    _err: &mut dyn io::Write,
                ) -> NativeInputSendStatus {
                    *self = Self;
                    NativeInputSendStatus::FailedBeforeSend
                }
            }
        }
    }
}
mod activity;
mod address;
mod display;
mod http_date;
cfg_select! {
    target_os = "linux" => {
        mod linux_input;
    }
    target_os = "macos" => {
        mod macos_input;
    }
    target_os = "windows" => {
        mod windows_input;
    }
    _ => {}
}
mod native_http;
mod sample;
mod util;
mod worker;
cfg_select! {
    target_os = "windows" => {
        mod timer_resolution;
    }
    _ => {}
}
const FULL_SYNC_INTERVAL: Duration = Duration::from_mins(5);
const RETRY_DELAY: Duration = Duration::from_secs(10);
const TCP_TIMEOUT: Duration = Duration::from_secs(5);
const ENTER_BUFFER_CAPACITY: usize = 8;
const ENTER_BUFFER_READ_LIMIT_BYTES: usize = ENTER_BUFFER_CAPACITY + 1;
const ENTER_INPUT_TOO_LONG: &str = "서버 시간 종료 입력이 너무 깁니다.";
const ENTER_THREAD_PANIC: &str = "입력 대기 스레드 패닉 발생";
const NUM_SAMPLES: usize = 10;
const CALIBRATION_TIMEOUT: Duration = Duration::from_secs(5);
const CALIBRATION_TIMEOUT_MESSAGE: &str = "정밀 보정 제한 시간 안에 유효한 서버 Date tick을 관측하지 못했습니다. 전체 보정을 다시 시작합니다.";
const FINAL_COUNTDOWN_RTT_ALPHA_NUM: u32 = 7;
const FINAL_COUNTDOWN_RTT_ALPHA_DENOM: u32 = 10;
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
const ADAPTIVE_POLL_INTERVAL: Duration = Duration::from_millis(10);
const COUNTDOWN_DELAY_ERROR: &str = "카운트다운 지연 계산 실패";
const DISPLAY_STATUS_PREFIX: &str = "\r서버 시간: ";
const DISPLAY_UPDATE_INTERVAL: Duration = Duration::from_millis(45);
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
#[derive(Debug)]
pub(super) struct TimeError {
    detail: Cow<'static, str>,
    io_kind: Option<ErrorKind>,
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
    scheme: UrlScheme,
    secure_url: String,
    tcp_host_header: String,
}
pub(super) struct TargetTimeOfDay {
    seconds_after_midnight: u32,
}
pub(super) struct ServerTimeSession {
    host: ParsedServer,
    now: Instant,
    target_time: Option<TargetTimeOfDay>,
    trigger_action: Option<TriggerAction>,
}
pub(super) struct ServerTimeSessionParts {
    pub host: ParsedServer,
    pub now: Instant,
    pub target_time: Option<TargetTimeOfDay>,
    pub trigger_action: Option<TriggerAction>,
}
impl From<ServerTimeSessionParts> for ServerTimeSession {
    fn from(parts: ServerTimeSessionParts) -> Self {
        Self {
            host: parts.host,
            now: parts.now,
            target_time: parts.target_time,
            trigger_action: parts.trigger_action,
        }
    }
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
            let mut value = 0_u32;
            for byte in component.bytes() {
                if !byte.is_ascii_digit() {
                    return Err(INVALID_TIME_INPUT_ERR);
                }
                let digit = u32::from(byte.wrapping_sub(b'0'));
                value = value
                    .checked_mul(10)
                    .and_then(|current| current.checked_add(digit))
                    .ok_or(INVALID_TIME_INPUT_ERR)?;
            }
            Ok(value)
        };
        let (hour, minute, second) = (
            parse_component(hour_str)?,
            parse_component(minute_str)?,
            parse_component(second_str)?,
        );
        if !(hour <= 23 && minute <= 59 && second <= 59) {
            return Err(INVALID_TIME_INPUT_ERR);
        }
        let Some(hour_secs) = hour.checked_mul(KST_SECONDS_PER_HOUR_U32) else {
            return Err(INVALID_TIME_INPUT_ERR);
        };
        let Some(minute_secs) = minute.checked_mul(KST_SECONDS_PER_MINUTE_U32) else {
            return Err(INVALID_TIME_INPUT_ERR);
        };
        let Some(seconds_after_midnight) = hour_secs
            .checked_add(minute_secs)
            .and_then(|value| value.checked_add(second))
        else {
            return Err(INVALID_TIME_INPUT_ERR);
        };
        if seconds_after_midnight >= 86_400 {
            return Err(INVALID_TIME_INPUT_ERR);
        }
        Ok(Self {
            seconds_after_midnight,
        })
    }
}
impl TimeError {
    fn header_not_found(detail: impl Into<Cow<'static, str>>) -> Self {
        Self::new(TimeErrorKind::HeaderNotFound, detail)
    }
    pub(super) const fn is_unexpected_eof(&self) -> bool {
        matches!(self.io_kind, Some(ErrorKind::UnexpectedEof))
    }
    fn new(kind: TimeErrorKind, detail: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind,
            detail: detail.into(),
            io_kind: None,
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
            kind,
            detail: detail.into(),
            io_kind: None,
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
        let io_kind = err.kind();
        Self {
            kind: TimeErrorKind::Io,
            detail: Cow::Borrowed(""),
            io_kind: Some(io_kind),
            source: Some(Box::new(err)),
        }
    }
}
impl From<SystemTimeError> for TimeError {
    fn from(err: SystemTimeError) -> Self {
        Self {
            kind: TimeErrorKind::Time,
            detail: Cow::Borrowed(""),
            io_kind: None,
            source: Some(Box::new(err)),
        }
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
impl Drop for SampleWorker {
    fn drop(&mut self) {
        let _send_result = self.command_sender.send(SampleWorkerCommand::Stop);
    }
}
impl SampleWorker {
    fn fetch(&mut self, kind: SampleRequestKind) -> Result<PendingSampleRequest> {
        let Some(generation) = self.generation.checked_add(1) else {
            return Err(TimeError::parse(
                "서버 시간 샘플 요청 세대 계산 중 overflow가 발생했습니다.",
            ));
        };
        self.generation = generation;
        self.command_sender
            .send(SampleWorkerCommand::Fetch { generation, kind })
            .map_err(|source| {
                TimeError::parse_with_source("서버 시간 샘플 worker 요청 실패", source)
            })?;
        Ok(PendingSampleRequest { generation })
    }
}
struct AppState {
    baseline_rtt: Option<Duration>,
    baseline_rtt_attempts: usize,
    baseline_rtt_next_sample_at: Instant,
    baseline_rtt_samples: [TimeSample; NUM_SAMPLES],
    baseline_rtt_valid_count: usize,
    calibration_deadline: Option<Instant>,
    final_countdown_next_sample_error_message_at: Option<Instant>,
    final_countdown_sampler: FinalCountdownSampler,
    #[cfg(target_os = "windows")]
    high_res_timer_guard: Option<HighResTimerGuard>,
    host: Arc<ParsedServer>,
    last_sample: Option<TimeSample>,
    live_rtt: Option<Duration>,
    next_full_sync_at: Instant,
    pending_sample_request: Option<PendingSampleRequest>,
    pending_target_time: Option<TargetTimeOfDay>,
    sample_worker: SampleWorker,
    server_time: Option<ServerTime>,
    target_time: Option<SystemTime>,
    trigger_action: Option<TriggerAction>,
}
struct LoopRuntime<'runtime> {
    err: &'runtime mut dyn io::Write,
    out: &'runtime mut dyn io::Write,
    prepared_input: &'runtime mut native_input::PreparedInput,
}
struct SampleWorker {
    command_sender: mpsc::Sender<SampleWorkerCommand>,
    generation: u64,
    join_handle: thread::JoinHandle<()>,
    response_receiver: mpsc::Receiver<SampleWorkerResponse>,
}
struct FinalCountdownSampler {
    command_sender: Option<mpsc::Sender<FinalCountdownSamplerCommand>>,
    generation: u64,
    join_handle: Option<thread::JoinHandle<()>>,
    sample_interval: Option<Duration>,
    sample_slot: Arc<Mutex<FinalCountdownSampleSlot>>,
    startup_error: Option<TimeError>,
    unavailable: bool,
}
struct FinalCountdownDeadline {
    one_way_delay: Duration,
    server_time: ServerTime,
    source: CountdownTriggerSource,
    target_time: SystemTime,
    trigger_instant: Instant,
}
struct FinalCountdownSampleSlot {
    generation: u64,
    latest_result: Option<Result<TimeSample>>,
}
struct PendingSampleRequest {
    generation: u64,
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
    Disconnected,
    Empty,
    Sample {
        kind: SampleRequestKind,
        result: Result<TimeSample>,
    },
}
enum SampleWorkerCommand {
    Fetch {
        generation: u64,
        kind: SampleRequestKind,
    },
    Stop,
}
enum FinalCountdownSamplerCommand {
    SetInterval { generation: u64, interval: Duration },
    Shutdown,
    StartPeriodic { generation: u64, interval: Duration },
    StopPeriodic,
}
enum FinalCountdownSamplerCommandFlow {
    Continue,
    Shutdown,
}
enum FinalCountdownSamplePoll {
    Disconnected,
    Empty,
    Sample(Result<TimeSample>),
}
struct CachedTcpSocketAddr {
    addr: net::SocketAddr,
}
struct CachedTcpConnection {
    reader: io::BufReader<net::TcpStream>,
}
struct NetworkContext {
    cached_tcp_connection: Option<CachedTcpConnection>,
    cached_tcp_socket_addr: Option<CachedTcpSocketAddr>,
    host: Arc<ParsedServer>,
    native_http: native_http::NativeHttp,
    tcp_line_buffer: Vec<u8>,
    tcp_request_buffer: Vec<u8>,
}
impl FinalCountdownSamplerCommand {
    fn apply(
        self,
        active_generation: &mut Option<u64>,
        sample_interval: &mut Duration,
    ) -> FinalCountdownSamplerCommandFlow {
        match self {
            Self::SetInterval {
                generation,
                interval,
            } => {
                if *active_generation == Some(generation) {
                    *sample_interval = interval;
                }
                FinalCountdownSamplerCommandFlow::Continue
            }
            Self::Shutdown => FinalCountdownSamplerCommandFlow::Shutdown,
            Self::StartPeriodic {
                generation,
                interval,
            } => {
                *active_generation = Some(generation);
                *sample_interval = interval;
                FinalCountdownSamplerCommandFlow::Continue
            }
            Self::StopPeriodic => {
                *active_generation = None;
                FinalCountdownSamplerCommandFlow::Continue
            }
        }
    }
}
impl Drop for FinalCountdownSampler {
    fn drop(&mut self) {
        if let Some(command_sender) = self.command_sender.as_ref() {
            let _send_result = command_sender.send(FinalCountdownSamplerCommand::Shutdown);
        }
    }
}
impl FinalCountdownSampler {
    fn disable(&mut self) {
        self.stop_periodic();
        self.unavailable = true;
        self.command_sender = None;
    }
    fn poll(&mut self) -> FinalCountdownSamplePoll {
        let sampler_finished = self
            .join_handle
            .as_ref()
            .is_some_and(thread::JoinHandle::is_finished);
        let mut should_disable = sampler_finished;
        let mut poisoned = false;
        let latest_result = match self.sample_slot.try_lock() {
            Ok(mut slot) => {
                if slot.generation == self.generation {
                    slot.latest_result.take()
                } else {
                    None
                }
            }
            Err(TryLockError::WouldBlock) => {
                if sampler_finished {
                    None
                } else {
                    return FinalCountdownSamplePoll::Empty;
                }
            }
            Err(TryLockError::Poisoned(_)) => {
                should_disable = true;
                poisoned = true;
                None
            }
        };
        if should_disable {
            self.disable();
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
    fn set_interval(&mut self, sample_interval: Duration) {
        if self.sample_interval != Some(sample_interval) {
            self.sample_interval = Some(sample_interval);
            let Some(command_sender) = self.command_sender.as_ref() else {
                self.disable();
                return;
            };
            if command_sender
                .send(FinalCountdownSamplerCommand::SetInterval {
                    generation: self.generation,
                    interval: sample_interval,
                })
                .is_err()
            {
                self.disable();
            }
        }
    }
    fn start(&mut self, sample_interval: Duration) -> Result<()> {
        if self.unavailable {
            if let Some(startup_error) = self.startup_error.take() {
                return Err(startup_error);
            }
            return Ok(());
        }
        if self.sample_interval.is_some() {
            self.set_interval(sample_interval);
            return Ok(());
        }
        let Some(next_generation) = self.generation.checked_add(1) else {
            return Err(TimeError::parse(
                "카운트다운 샘플 세대 계산 중 overflow가 발생했습니다.",
            ));
        };
        self.generation = next_generation;
        self.sample_interval = Some(sample_interval);
        {
            let mut slot = self
                .sample_slot
                .lock()
                .map_err(|_poisoned| TimeError::parse("카운트다운 샘플 상태 잠금 실패"))?;
            slot.generation = self.generation;
            slot.latest_result = None;
        }
        let Some(command_sender) = self.command_sender.as_ref() else {
            self.disable();
            return Err(TimeError::parse("카운트다운 샘플러가 종료되었습니다."));
        };
        command_sender
            .send(FinalCountdownSamplerCommand::StartPeriodic {
                generation: self.generation,
                interval: sample_interval,
            })
            .map_err(|source| {
                self.disable();
                TimeError::parse_with_source("카운트다운 샘플러 시작 실패", source)
            })
    }
    fn stop_periodic(&mut self) {
        if self.sample_interval.take().is_none() {
            return;
        }
        let Some(next_generation) = self.generation.checked_add(1) else {
            self.disable();
            return;
        };
        self.generation = next_generation;
        if let Ok(mut slot) = self.sample_slot.lock() {
            slot.generation = self.generation;
            slot.latest_result = None;
        }
        if let Some(command_sender) = self.command_sender.as_ref()
            && command_sender
                .send(FinalCountdownSamplerCommand::StopPeriodic)
                .is_err()
        {
            self.unavailable = true;
            self.command_sender = None;
        }
    }
}
impl NetworkContext {
    fn new(host: Arc<ParsedServer>) -> Result<Self> {
        let mut tcp_line_buffer = Vec::new();
        tcp_line_buffer
            .try_reserve_exact(TCP_LINE_BUFFER_CAPACITY)
            .map_err(|source| TimeError::parse_with_source("buffer 메모리 확보 실패", source))?;
        let mut tcp_request_buffer = Vec::new();
        tcp_request_buffer
            .try_reserve_exact(TCP_LINE_BUFFER_CAPACITY)
            .map_err(|source| TimeError::parse_with_source("buffer 메모리 확보 실패", source))?;
        Ok(Self {
            cached_tcp_connection: None,
            cached_tcp_socket_addr: None,
            host,
            native_http: native_http::NativeHttp::default(),
            tcp_line_buffer,
            tcp_request_buffer,
        })
    }
}
impl ServerTimeSession {
    pub(super) fn run_loop(self, out: &mut dyn io::Write, err: &mut dyn io::Write) -> Result<()> {
        let host = Arc::new(self.host);
        let sample_worker = spawn_sample_worker(Arc::clone(&host))?;
        let final_countdown_sampler = spawn_final_countdown_sampler(Arc::clone(&host));
        let baseline_placeholder = TimeSample {
            response_received_inst: self.now,
            rtt: Duration::ZERO,
            server_time: UNIX_EPOCH,
        };
        let mut app_state = AppState {
            baseline_rtt: None,
            baseline_rtt_attempts: 0,
            baseline_rtt_next_sample_at: self.now,
            baseline_rtt_samples: [baseline_placeholder; NUM_SAMPLES],
            baseline_rtt_valid_count: 0,
            calibration_deadline: None,
            final_countdown_next_sample_error_message_at: None,
            final_countdown_sampler,
            #[cfg(target_os = "windows")]
            high_res_timer_guard: None,
            host,
            last_sample: None,
            live_rtt: None,
            next_full_sync_at: self.now,
            pending_sample_request: None,
            pending_target_time: self.target_time,
            sample_worker,
            server_time: None,
            target_time: None,
            trigger_action: self.trigger_action,
        };
        app_state.run_loop(out, err)
    }
}
impl AppState {
    fn accept_calibration_tick(
        &mut self,
        current_sample: TimeSample,
    ) -> Option<Result<ServerTime>> {
        let server_time = self.build_calibrated_server_time(current_sample)?;
        self.server_time = Some(server_time);
        self.calibration_deadline = None;
        let Some(next_full_sync_at) = current_sample
            .response_received_inst
            .checked_add(FULL_SYNC_INTERVAL)
        else {
            return Some(Err(TimeError::parse("다음 전체 동기화 시각 계산 실패.")));
        };
        self.next_full_sync_at = next_full_sync_at;
        Some(Ok(server_time))
    }
    fn append_final_countdown_sample_error(
        &mut self,
        now: Instant,
        msg_buf: &mut String,
        err: impl fmt::Display,
    ) -> bool {
        if self
            .final_countdown_next_sample_error_message_at
            .is_some_and(|next_message_at| now < next_message_at)
        {
            return false;
        }
        self.final_countdown_next_sample_error_message_at =
            now.checked_add(FINAL_COUNTDOWN_SAMPLE_ERROR_MESSAGE_INTERVAL);
        append_error_detail(msg_buf, "카운트다운 샘플 획득 실패: ", err);
        true
    }
    fn begin_baseline_rtt_measurement(&mut self, now: Instant, out: &mut dyn io::Write) {
        if self.last_sample.is_none() {
            write_line_best_effort(out, format_args!("1단계: RTT 기준값 측정을 시작합니다…"));
        }
        let placeholder = TimeSample {
            response_received_inst: now,
            rtt: Duration::ZERO,
            server_time: UNIX_EPOCH,
        };
        self.baseline_rtt_samples = [placeholder; NUM_SAMPLES];
        self.baseline_rtt_valid_count = 0;
        self.baseline_rtt_next_sample_at = now;
        self.calibration_deadline = None;
    }
    fn begin_final_countdown_sampling(&mut self, initial_sample_interval: Duration) -> Result<()> {
        if self.final_countdown_sampler.unavailable {
            self.final_countdown_sampler = spawn_final_countdown_sampler(Arc::clone(&self.host));
        }
        self.final_countdown_sampler.start(initial_sample_interval)
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
        let one_way_delay = effective_one_way_delay(current_sample.rtt)?;
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
    const fn cancel_sample_request(&mut self) {
        self.pending_sample_request = None;
    }
    fn confirm_pending_target_time(
        &mut self,
        server_time: ServerTime,
        now: Instant,
        msg_buf: &mut String,
    ) -> Result<()> {
        let Some(target_time_of_day) = self.pending_target_time.as_ref() else {
            return Ok(());
        };
        let current_server_time = server_time.current_server_time_at(now);
        let since_epoch = current_server_time.duration_since(UNIX_EPOCH)?;
        let kst_epoch_secs = since_epoch
            .as_secs()
            .checked_add(KST_OFFSET_SECS_U64)
            .ok_or_else(|| TimeError::parse("KST 현재 시각 계산 중 overflow가 발생했습니다."))?;
        let current_kst_second = u32::try_from(kst_epoch_secs.rem_euclid(KST_SECONDS_PER_DAY_U64))
            .map_err(|source| TimeError::parse_with_source("KST 초 변환 실패", source))?;
        let current_kst_day = kst_epoch_secs.div_euclid(KST_SECONDS_PER_DAY_U64);
        let target_second = target_time_of_day.seconds_after_midnight;
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
    fn disable_final_countdown_sampling(&mut self) {
        self.final_countdown_sampler.disable();
    }
    const fn discard_partial_baseline_measurement(&mut self) {
        self.baseline_rtt_attempts = 0;
        self.baseline_rtt_valid_count = 0;
        self.calibration_deadline = None;
        self.pending_sample_request = None;
    }
    fn end_final_countdown_sampling(&mut self) {
        self.final_countdown_sampler.stop_periodic();
        self.final_countdown_next_sample_error_message_at = None;
    }
    fn ensure_sample_request(&mut self, kind: SampleRequestKind) -> Result<()> {
        if self.pending_sample_request.is_some() {
            return Ok(());
        }
        match self.sample_worker.fetch(kind) {
            Ok(request) => {
                self.pending_sample_request = Some(request);
                Ok(())
            }
            Err(fetch_err) => {
                if !self.respawn_sample_worker() {
                    return Err(fetch_err);
                }
                self.pending_sample_request = Some(self.sample_worker.fetch(kind)?);
                Ok(())
            }
        }
    }
    fn finish_baseline_rtt_measurement<'message>(
        &mut self,
        sample_count: usize,
        msg_buf: &'message mut String,
    ) -> ActivityTransition<'message> {
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
        let Some(rtts) = rtt_nanos.get_mut(Range {
            start: 0,
            end: filled,
        }) else {
            return transition_to_retry("RTT 샘플 범위 계산 실패.");
        };
        rtts.sort_unstable();
        let trim = filled.div_euclid(RTT_TRIM_DIVISOR);
        let Some(window_end) = filled.checked_sub(trim) else {
            return transition_to_retry("RTT 샘플 윈도우 계산 실패.");
        };
        let Some(window) = rtts.get(Range {
            start: trim,
            end: window_end,
        }) else {
            return transition_to_retry("RTT 샘플 윈도우 계산 실패.");
        };
        let sum_nanos: u128 = window.iter().sum();
        let Ok(window_len) = u128::try_from(window.len()) else {
            return transition_to_retry("RTT 샘플 수 변환 실패.");
        };
        let Some(avg_nanos) = sum_nanos.checked_div(window_len) else {
            return transition_to_retry("RTT 기준값 계산 실패.");
        };
        let baseline_rtt = Duration::from_nanos_u128(avg_nanos);
        self.baseline_rtt = Some(baseline_rtt);
        self.calibration_deadline = None;
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
        let deadline = if let Some(deadline) = self.calibration_deadline {
            deadline
        } else {
            let Some(deadline) = now.checked_add(CALIBRATION_TIMEOUT) else {
                return transition_to_retry("정밀 보정 제한 시간 계산 실패.");
            };
            self.calibration_deadline = Some(deadline);
            deadline
        };
        if now >= deadline {
            return transition_to_retry(CALIBRATION_TIMEOUT_MESSAGE);
        }
        let sample_result = match self.poll_sample_request() {
            SampleRequestPoll::Sample {
                kind: SampleRequestKind::Calibration,
                result,
            } => result,
            SampleRequestPoll::Sample { .. } | SampleRequestPoll::Empty => {
                if let Err(start_err) = self.ensure_sample_request(SampleRequestKind::Calibration) {
                    append_error_detail(msg_buf, "정밀 보정 샘플 요청 실패: ", start_err);
                    return transition_to_retry(msg_buf);
                }
                return ActivityTransition {
                    activity: Activity::CalibrateOnTick,
                    message: None,
                };
            }
            SampleRequestPoll::Disconnected => {
                return transition_to_retry("서버 시간 샘플 worker가 종료되었습니다.");
            }
        };
        let completed_at = Instant::now();
        if completed_at >= deadline {
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
        if let Some(server_time_result) = self.accept_calibration_tick(current_sample) {
            let server_time = match server_time_result {
                Ok(server_time) => server_time,
                Err(err) => {
                    append_error_detail(msg_buf, "", err);
                    return transition_to_retry(msg_buf);
                }
            };
            if self.pending_target_time.is_some() {
                msg_buf.push_str("[성공] 정밀 보정 완료!");
                if let Err(target_err) =
                    self.confirm_pending_target_time(server_time, Instant::now(), msg_buf)
                {
                    append_error_detail(msg_buf, "\n목표 시각 확정 실패: ", target_err);
                    return transition_to_retry(msg_buf);
                }
                return ActivityTransition {
                    activity: Activity::Predicting,
                    message: Some(msg_buf),
                };
            }
            return ActivityTransition {
                activity: Activity::Predicting,
                message: Some("[성공] 정밀 보정 완료!"),
            };
        }
        self.last_sample = Some(current_sample);
        if Instant::now() >= deadline {
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
        target_time: SystemTime,
        msg_buf: &'message mut String,
        prepared_input: &mut native_input::PreparedInput,
        err: &mut dyn io::Write,
    ) -> ActivityTransition<'message> {
        let Some(st) = self.server_time.as_ref().copied() else {
            return ActivityTransition {
                activity: Activity::MeasureBaselineRtt,
                message: Some("[오류] 내부 상태 불일치: server_time 없음"),
            };
        };
        let now = Instant::now();
        let current_server_time = st.current_server_time_at(now);
        let sample_error_reported = match self.final_countdown_sampler.poll() {
            FinalCountdownSamplePoll::Sample(Ok(sample)) => {
                return self.handle_final_countdown_sample(
                    target_time,
                    msg_buf,
                    sample,
                    prepared_input,
                    err,
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
        let estimated_rtt = self.live_rtt.unwrap_or(st.baseline_rtt);
        let Some(estimated_one_way_delay) = effective_one_way_delay(estimated_rtt) else {
            msg_buf.push_str(COUNTDOWN_DELAY_ERROR);
            return ActivityTransition {
                activity: Activity::FinalCountdown { target_time },
                message: Some(msg_buf),
            };
        };
        let Some(trigger_instant) =
            trigger_instant_for_target(st, target_time, estimated_one_way_delay, now)
        else {
            msg_buf.push_str("카운트다운 실행 시각 계산 실패");
            return ActivityTransition {
                activity: Activity::FinalCountdown { target_time },
                message: Some(msg_buf),
            };
        };
        if trigger_instant.saturating_duration_since(now) <= FINAL_COUNTDOWN_FREEZE_WINDOW {
            let deadline = FinalCountdownDeadline {
                one_way_delay: estimated_one_way_delay,
                server_time: st,
                source: CountdownTriggerSource::Estimated,
                target_time,
                trigger_instant,
            };
            return self.trigger_final_countdown_deadline(&deadline, msg_buf, prepared_input, err);
        }
        if let Ok(duration_until_target) = target_time.duration_since(current_server_time) {
            self.set_final_countdown_sample_interval(final_countdown_sample_interval(
                duration_until_target,
            ));
        }
        let decision =
            decide_countdown_action(target_time, current_server_time, estimated_one_way_delay);
        if !matches!(decision, CountdownDecision::Wait) {
            return self.trigger_countdown_decision(
                decision,
                msg_buf,
                estimated_one_way_delay,
                CountdownTriggerSource::Estimated,
                prepared_input,
                err,
            );
        }
        if sample_error_reported {
            return ActivityTransition {
                activity: Activity::FinalCountdown { target_time },
                message: Some(msg_buf),
            };
        }
        let sample_interval = target_time.duration_since(current_server_time).map_or(
            FINAL_COUNTDOWN_SAMPLE_FINAL_INTERVAL,
            final_countdown_sample_interval,
        );
        if let Err(start_err) = self.begin_final_countdown_sampling(sample_interval) {
            self.disable_final_countdown_sampling();
            append_error_detail(msg_buf, "카운트다운 샘플러 시작 실패: ", start_err);
            return ActivityTransition {
                activity: Activity::FinalCountdown { target_time },
                message: Some(msg_buf),
            };
        }
        ActivityTransition {
            activity: Activity::FinalCountdown { target_time },
            message: None,
        }
    }
    fn handle_final_countdown_sample<'message>(
        &mut self,
        target_time: SystemTime,
        msg_buf: &'message mut String,
        sample: TimeSample,
        prepared_input: &mut native_input::PreparedInput,
        err: &mut dyn io::Write,
    ) -> ActivityTransition<'message> {
        let Some(server_time) = self.server_time.as_mut() else {
            return ActivityTransition {
                activity: Activity::MeasureBaselineRtt,
                message: Some("[오류] 내부 상태 불일치: server_time 없음"),
            };
        };
        let sample_rtt = sample.rtt;
        *server_time = server_time.recalibrate_with_rtt(sample_rtt);
        let sample_now = Instant::now();
        let sampled_server_time = server_time.current_server_time_at(sample_now);
        let old_rtt = self.live_rtt.unwrap_or(sample_rtt);
        let new_rtt_nanos = blend_weighted_nanos(
            old_rtt.as_nanos(),
            sample_rtt.as_nanos(),
            FINAL_COUNTDOWN_RTT_ALPHA_NUM,
            FINAL_COUNTDOWN_RTT_ALPHA_DENOM,
        );
        let live_rtt = Duration::from_nanos_u128(new_rtt_nanos);
        self.live_rtt = Some(live_rtt);
        let effective_rtt = live_rtt.max(sample_rtt);
        let Some(one_way_delay) = effective_one_way_delay(effective_rtt) else {
            msg_buf.push_str(COUNTDOWN_DELAY_ERROR);
            return ActivityTransition {
                activity: Activity::FinalCountdown { target_time },
                message: Some(msg_buf),
            };
        };
        let Some(trigger_instant) =
            trigger_instant_for_target(*server_time, target_time, one_way_delay, sample_now)
        else {
            msg_buf.push_str("카운트다운 실행 시각 계산 실패");
            return ActivityTransition {
                activity: Activity::FinalCountdown { target_time },
                message: Some(msg_buf),
            };
        };
        if trigger_instant.saturating_duration_since(sample_now) <= FINAL_COUNTDOWN_FREEZE_WINDOW {
            let deadline = FinalCountdownDeadline {
                one_way_delay,
                server_time: *server_time,
                source: CountdownTriggerSource::Sampled { rtt: sample_rtt },
                target_time,
                trigger_instant,
            };
            return self.trigger_final_countdown_deadline(&deadline, msg_buf, prepared_input, err);
        }
        let sampled_decision =
            decide_countdown_action(target_time, sampled_server_time, one_way_delay);
        if matches!(sampled_decision, CountdownDecision::Wait) {
            return ActivityTransition {
                activity: Activity::FinalCountdown { target_time },
                message: None,
            };
        }
        self.trigger_countdown_decision(
            sampled_decision,
            msg_buf,
            one_way_delay,
            CountdownTriggerSource::Sampled { rtt: sample_rtt },
            prepared_input,
            err,
        )
    }
    fn handle_measure_baseline_rtt<'message>(
        &mut self,
        msg_buf: &'message mut String,
        now: Instant,
        out: &mut dyn io::Write,
    ) -> ActivityTransition<'message> {
        if self.baseline_rtt_attempts == 0
            && self.baseline_rtt_valid_count == 0
            && self.pending_sample_request.is_none()
        {
            self.begin_baseline_rtt_measurement(now, out);
        }
        let had_pending_request = self.pending_sample_request.is_some();
        let sample_result = match self.poll_sample_request() {
            SampleRequestPoll::Sample {
                kind: SampleRequestKind::Baseline { attempt_index },
                result,
            } => (attempt_index, result),
            SampleRequestPoll::Sample { .. } | SampleRequestPoll::Empty => {
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
            SampleRequestPoll::Disconnected => {
                return transition_to_retry("서버 시간 샘플 worker가 종료되었습니다.");
            }
        };
        let (attempt_index, result) = sample_result;
        let next_sample_base = match result {
            Ok(sample) => {
                if sample.rtt > Duration::ZERO {
                    let Some(slot) = self.baseline_rtt_samples.get_mut(attempt_index) else {
                        return transition_to_retry("RTT 샘플 인덱스 계산 실패.");
                    };
                    *slot = sample;
                    let Some(next_valid_count) = self.baseline_rtt_valid_count.checked_add(1)
                    else {
                        return transition_to_retry("RTT 샘플 개수 계산 실패.");
                    };
                    self.baseline_rtt_valid_count = next_valid_count;
                    self.last_sample = Some(sample);
                }
                sample.response_received_inst
            }
            Err(err) => {
                self.baseline_rtt_attempts = 0;
                self.baseline_rtt_valid_count = 0;
                append_error_detail(msg_buf, "RTT 샘플 수집 실패: ", err);
                return transition_to_retry(msg_buf);
            }
        };
        let Some(next_attempt_index) = attempt_index.checked_add(1) else {
            return transition_to_retry("RTT 시도 횟수 계산 실패.");
        };
        self.baseline_rtt_attempts = next_attempt_index;
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
        self.finish_baseline_rtt_measurement(self.baseline_rtt_valid_count, msg_buf)
    }
    fn handle_predicting<'message>(
        &mut self,
        msg_buf: &'message mut String,
        now: Instant,
    ) -> ActivityTransition<'message> {
        let Some(server_time) = self.server_time.as_ref() else {
            return ActivityTransition {
                activity: Activity::MeasureBaselineRtt,
                message: None,
            };
        };
        let estimated_server_time = server_time.current_server_time_at(now);
        let target_remaining = self
            .target_time
            .and_then(|target| target.duration_since(estimated_server_time).ok());
        let protect_target =
            target_remaining.is_some_and(|remaining| remaining <= FINAL_COUNTDOWN_WARMUP_WINDOW);
        if let Some(target_time) = self.target_time.take_if(|target| {
            target
                .duration_since(estimated_server_time)
                .ok()
                .is_none_or(|remaining| remaining <= FINAL_COUNTDOWN_WINDOW)
        }) {
            if self.live_rtt.is_none() {
                self.live_rtt = Some(server_time.baseline_rtt);
            }
            self.discard_partial_baseline_measurement();
            return ActivityTransition {
                activity: Activity::FinalCountdown { target_time },
                message: Some("최종 카운트다운 시작!"),
            };
        }
        if let Some(warmup_remaining) = target_remaining
            && warmup_remaining <= FINAL_COUNTDOWN_WARMUP_WINDOW
        {
            let sample_interval = final_countdown_sample_interval(warmup_remaining);
            if self.final_countdown_sampler.sample_interval.is_some() {
                self.set_final_countdown_sample_interval(sample_interval);
            } else {
                match self.begin_final_countdown_sampling(sample_interval) {
                    Ok(()) => {}
                    Err(start_err) => {
                        self.disable_final_countdown_sampling();
                        append_error_detail(msg_buf, "카운트다운 샘플러 시작 실패: ", start_err);
                        return ActivityTransition {
                            activity: Activity::Predicting,
                            message: Some(msg_buf),
                        };
                    }
                }
            }
        }
        if now >= self.next_full_sync_at && !protect_target {
            self.end_final_countdown_sampling();
            ActivityTransition {
                activity: Activity::MeasureBaselineRtt,
                message: Some("서버 시간 보정 주기 도래, 재보정 시작."),
            }
        } else {
            ActivityTransition {
                activity: Activity::Predicting,
                message: None,
            }
        }
    }
    fn maybe_start_final_countdown(&mut self, now: Instant) -> Option<ActivityTransition<'static>> {
        let server_time = self.server_time.as_ref().copied()?;
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
            activity: Activity::FinalCountdown { target_time },
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
        if !matches!(
            *activity,
            Activity::FinalCountdown { .. } | Activity::Finished
        ) && let Some(transition) = self.maybe_start_final_countdown(now)
        {
            return transition;
        }
        match *activity {
            Activity::MeasureBaselineRtt => {
                self.handle_measure_baseline_rtt(message_buffer, now, runtime.out)
            }
            Activity::CalibrateOnTick => self.handle_calibrate_on_tick(message_buffer),
            Activity::Predicting => self.handle_predicting(message_buffer, now),
            Activity::FinalCountdown { target_time } => self.handle_final_countdown(
                target_time,
                message_buffer,
                runtime.prepared_input,
                runtime.err,
            ),
            Activity::Finished => ActivityTransition {
                activity: Activity::Predicting,
                message: Some("액션 완료. 예측 모드 전환."),
            },
            Activity::Retrying { retry_at } => {
                if now >= retry_at {
                    ActivityTransition {
                        activity: Activity::MeasureBaselineRtt,
                        message: Some("[재시도] 동기화를 다시 시작합니다."),
                    }
                } else {
                    ActivityTransition {
                        activity: Activity::Retrying { retry_at },
                        message: None,
                    }
                }
            }
        }
    }
    fn poll_sample_request(&mut self) -> SampleRequestPoll {
        let Some(pending_generation) = self
            .pending_sample_request
            .as_ref()
            .map(|request| request.generation)
        else {
            loop {
                match self.sample_worker.response_receiver.try_recv() {
                    Ok(_) => {}
                    Err(mpsc::TryRecvError::Empty) => return SampleRequestPoll::Empty,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        return if self.respawn_sample_worker() {
                            SampleRequestPoll::Empty
                        } else {
                            SampleRequestPoll::Disconnected
                        };
                    }
                }
            }
        };
        loop {
            match self.sample_worker.response_receiver.try_recv() {
                Ok(response) => {
                    if response.generation == pending_generation {
                        self.pending_sample_request = None;
                        return SampleRequestPoll::Sample {
                            kind: response.kind,
                            result: response.result,
                        };
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    if self.sample_worker.join_handle.is_finished() {
                        self.pending_sample_request = None;
                        return if self.respawn_sample_worker() {
                            SampleRequestPoll::Empty
                        } else {
                            SampleRequestPoll::Disconnected
                        };
                    }
                    return SampleRequestPoll::Empty;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.pending_sample_request = None;
                    return if self.respawn_sample_worker() {
                        SampleRequestPoll::Empty
                    } else {
                        SampleRequestPoll::Disconnected
                    };
                }
            }
        }
    }
    fn respawn_sample_worker(&mut self) -> bool {
        match spawn_sample_worker(Arc::clone(&self.host)) {
            Ok(sample_worker) => {
                self.sample_worker = sample_worker;
                true
            }
            Err(_) => false,
        }
    }
    fn run_loop(&mut self, out: &mut dyn io::Write, err: &mut dyn io::Write) -> Result<()> {
        out.write_all("\n서버 시간 확인을 시작합니다… (Enter를 누르면 종료)\n".as_bytes())?;
        let (tx, rx) = mpsc::channel();
        let input_thread = thread::spawn(move || -> IoResult<()> {
            let mut line = Vec::new();
            line.try_reserve_exact(ENTER_BUFFER_READ_LIMIT_BYTES)
                .map_err(io::Error::other)?;
            let mut stdin_lock = io::stdin().lock();
            let read_limit =
                u64::try_from(ENTER_BUFFER_READ_LIMIT_BYTES).map_err(IoError::other)?;
            let mut limited_stdin = IoRead::take(&mut stdin_lock, read_limit);
            IoBufRead::read_until(&mut limited_stdin, b'\n', &mut line)?;
            if line.len() > ENTER_BUFFER_CAPACITY && !line.ends_with(b"\n") {
                return Err(IoError::new(ErrorKind::InvalidInput, ENTER_INPUT_TOO_LONG));
            }
            let _send_result = tx.send(());
            Ok(())
        });
        let mut activity = Activity::MeasureBaselineRtt;
        let mut last_display_update = Instant::now();
        let mut message_buffer = String::new();
        message_buffer
            .try_reserve_exact(MESSAGE_BUFFER_CAPACITY)
            .map_err(|source| TimeError::parse_with_source("buffer 메모리 확보 실패", source))?;
        let mut prepared_input = native_input::PreparedInput::EMPTY;
        let mut line_buf = [0_u8; display::DISPLAY_LINE_BUF_LEN];
        loop {
            let activity_poll = match activity {
                Activity::Predicting | Activity::Finished | Activity::Retrying { .. } => {
                    DISPLAY_UPDATE_INTERVAL
                }
                Activity::MeasureBaselineRtt
                | Activity::CalibrateOnTick
                | Activity::FinalCountdown { .. } => ADAPTIVE_POLL_INTERVAL,
            };
            let pre_wait_now = Instant::now();
            let elapsed = pre_wait_now.saturating_duration_since(last_display_update);
            let remaining_display = DISPLAY_INTERVAL.saturating_sub(elapsed);
            let poll_timeout = activity_poll.min(remaining_display);
            match rx.recv_timeout(poll_timeout) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
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
                    prepared_input: &mut prepared_input,
                };
                self.next_activity(&activity, &mut message_buffer, now, &mut runtime)
            };
            let next_activity = transition.activity;
            self.sync_sample_request_state(&next_activity);
            self.sync_prepared_input_state(&activity, &next_activity, &mut prepared_input, err);
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
        input_thread
            .join()
            .map_err(|_panic_payload| TimeError::parse(ENTER_THREAD_PANIC))??;
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
            && let Some(st) = self.server_time.as_ref()
        {
            if let Err(display_err) = (|| -> Result<()> {
                let mut cur = ByteCursor::new(buffer);
                cur.write_bytes(DISPLAY_STATUS_PREFIX.as_bytes())?;
                st.write_current_display_time_buf_at(&mut cur, now)?;
                cur.write_bytes(b" \r")?;
                output.write_all(cur.written_slice()?)?;
                output.flush()?;
                Ok(())
            })() {
                output.write_all(DISPLAY_STATUS_PREFIX.as_bytes())?;
                write!(output, "표시 버퍼 오류: {display_err}")?;
                output.write_all(b" \r")?;
                output.flush()?;
            }
            *last_update = now;
        }
        Ok(())
    }
    fn set_final_countdown_sample_interval(&mut self, sample_interval: Duration) {
        self.final_countdown_sampler.set_interval(sample_interval);
    }
    fn should_prioritize_activity_transition(&self, activity: &Activity, now: Instant) -> bool {
        if activity.is_final_countdown() {
            return true;
        }
        let Some(server_time) = self.server_time.as_ref() else {
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
        let Activity::FinalCountdown { target_time } = *activity else {
            return true;
        };
        let Some(server_time) = self.server_time else {
            return true;
        };
        let rtt = self.live_rtt.unwrap_or(server_time.baseline_rtt);
        let Some(one_way_delay) = effective_one_way_delay(rtt) else {
            return true;
        };
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
                // SAFETY: No security attributes or name are needed. On OS versions without
                // high-resolution waitable timers this returns null and the code falls back to
                // WinMM + thread::sleep.
                let wait_timer_handle = unsafe {
                    sys::create_waitable_timer_ex_w(
                        null(),
                        null(),
                        CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                        SYNCHRONIZE_ACCESS | TIMER_MODIFY_STATE_ACCESS,
                    )
                };
                let high_res_wait_timer =
                    NonNull::new(wait_timer_handle).map(HighResTimerGuard::WaitTimer);
                let acquired_timer_guard = high_res_wait_timer.map_or_else(
                    || {
                        // SAFETY: This fallback requests a process-wide WinMM timer period only
                        // when the modern high-resolution waitable timer path is unavailable. A
                        // successful request is paired with exactly one `timeEndPeriod` call from
                        // `Drop`.
                        let status = unsafe { sys::time_begin_period(TARGET_PERIOD_MS) };
                        (status == TIMERR_NOERROR).then_some(HighResTimerGuard::PeriodAcquired)
                    },
                    Some,
                );
                let Some(timer_guard) = acquired_timer_guard else {
                    write_line_best_effort(
                        err,
                        format_args!(
                            concat!(
                                "[경고] Windows 타이머 해상도 {}ms 요청에 실패했습니다. ",
                                "카운트다운 정확도가 저하될 수 있습니다."
                            ),
                            TARGET_PERIOD_MS
                        ),
                    );
                    return;
                };
                self.high_res_timer_guard = Some(timer_guard);
            }
        }
        _ => {}
    }
    fn sync_prepared_input_state(
        &self,
        previous_activity: &Activity,
        next_activity: &Activity,
        prepared_input: &mut native_input::PreparedInput,
        err: &mut dyn io::Write,
    ) {
        let entering_final_countdown =
            next_activity.is_final_countdown() && !previous_activity.is_final_countdown();
        if entering_final_countdown {
            prepared_input.prepare(
                self.trigger_action.map(|action| match action {
                    TriggerAction::LeftClick => native_input::InputAction::MouseClick,
                    TriggerAction::F5Press => native_input::InputAction::F5Press,
                }),
                err,
            );
            return;
        }
        if !next_activity.is_final_countdown() {
            prepared_input.reset();
        }
    }
    const fn sync_sample_request_state(&mut self, next_activity: &Activity) {
        if !matches!(
            *next_activity,
            Activity::MeasureBaselineRtt | Activity::CalibrateOnTick
        ) {
            self.cancel_sample_request();
        }
        if !matches!(*next_activity, Activity::CalibrateOnTick) {
            self.calibration_deadline = None;
        }
    }
    fn trigger_and_finish<'message>(
        &mut self,
        msg_buf: &'message mut String,
        timing_detail: fmt::Arguments<'_>,
        prepared_input: &mut native_input::PreparedInput,
        err: &mut dyn io::Write,
    ) -> ActivityTransition<'message> {
        self.end_final_countdown_sampling();
        let send_status = self
            .trigger_action
            .map_or(NativeInputSendStatus::Sent, |action| {
                let input_action = match action {
                    TriggerAction::LeftClick => native_input::InputAction::MouseClick,
                    TriggerAction::F5Press => native_input::InputAction::F5Press,
                };
                prepared_input.send(input_action, err)
            });
        match send_status {
            NativeInputSendStatus::Sent => msg_buf.push_str("\n>>> 액션 실행! "),
            NativeInputSendStatus::FailedBeforeSend => msg_buf.push_str("\n>>> 액션 실행 실패! "),
            NativeInputSendStatus::PartialOrUnknown => {
                msg_buf.push_str("\n>>> 액션 전송 상태 불확실! ");
            }
        }
        append_fmt(msg_buf, timing_detail);
        ActivityTransition {
            activity: Activity::Finished,
            message: Some(msg_buf),
        }
    }
    fn trigger_countdown_decision<'message>(
        &mut self,
        decision: CountdownDecision,
        msg_buf: &'message mut String,
        one_way_delay: Duration,
        source: CountdownTriggerSource,
        prepared_input: &mut native_input::PreparedInput,
        err: &mut dyn io::Write,
    ) -> ActivityTransition<'message> {
        match (decision, source) {
            (
                CountdownDecision::TriggerWithRemaining(remaining),
                CountdownTriggerSource::Estimated,
            ) => self.trigger_and_finish(
                msg_buf,
                format_args!(
                    "(예측값 기준, 목표 도달까지 {:.1}ms 남음) (지연 예측: {:.1}ms)",
                    duration_millis_f64(remaining),
                    duration_millis_f64(one_way_delay),
                ),
                prepared_input,
                err,
            ),
            (CountdownDecision::TriggerLate(late_by), CountdownTriggerSource::Estimated) => self
                .trigger_and_finish(
                    msg_buf,
                    format_args!(
                        "(예측값 기준, 목표 초가 이미 시작되어 즉시 액션을 실행했습니다. ({:.1}ms 지연, 지연 예측: {:.1}ms))",
                        duration_millis_f64(late_by),
                        duration_millis_f64(one_way_delay),
                    ),
                    prepared_input,
                    err,
                ),
            (
                CountdownDecision::TriggerWithRemaining(remaining),
                CountdownTriggerSource::Sampled { rtt },
            ) => self.trigger_and_finish(
                msg_buf,
                format_args!(
                    "(목표 도달까지 {:.1}ms 남음) (지연 예측: {:.1}ms, 실측 RTT: {:.1}ms)",
                    duration_millis_f64(remaining),
                    duration_millis_f64(one_way_delay),
                    duration_millis_f64(rtt)
                ),
                prepared_input,
                err,
            ),
            (CountdownDecision::TriggerLate(late_by), CountdownTriggerSource::Sampled { rtt }) => {
                self.trigger_and_finish(
                    msg_buf,
                    format_args!(
                        "(목표 초가 이미 시작되어 즉시 액션을 실행했습니다. ({:.1}ms 지연, 지연 예측: {:.1}ms, 실측 RTT: {:.1}ms))",
                        duration_millis_f64(late_by),
                        duration_millis_f64(one_way_delay),
                        duration_millis_f64(rtt)
                    ),
                    prepared_input,
                    err,
                )
            }
            (CountdownDecision::Wait, _) => ActivityTransition {
                activity: Activity::Finished,
                message: Some("카운트다운 판단 상태 오류"),
            },
        }
    }
    fn trigger_final_countdown_deadline<'message>(
        &mut self,
        deadline: &FinalCountdownDeadline,
        msg_buf: &'message mut String,
        prepared_input: &mut native_input::PreparedInput,
        err: &mut dyn io::Write,
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
        let decision = match decide_countdown_action(
            deadline.target_time,
            trigger_server_time,
            deadline.one_way_delay,
        ) {
            CountdownDecision::Wait => {
                let remaining = match deadline.target_time.duration_since(trigger_server_time) {
                    Ok(remaining) => remaining,
                    Err(_past_target) => Duration::ZERO,
                };
                CountdownDecision::TriggerWithRemaining(remaining)
            }
            CountdownDecision::TriggerLate(late_by) => CountdownDecision::TriggerLate(late_by),
            CountdownDecision::TriggerWithRemaining(remaining) => {
                CountdownDecision::TriggerWithRemaining(remaining)
            }
        };
        self.trigger_countdown_decision(
            decision,
            msg_buf,
            deadline.one_way_delay,
            deadline.source,
            prepared_input,
            err,
        )
    }
}
fn duration_millis_f64(duration: Duration) -> f64 {
    NumericMul::mul(duration.as_secs_f64(), MILLIS_PER_SECOND_F64)
}
fn append_error_detail(target: &mut String, prefix: &str, err: impl fmt::Display) {
    append_fmt(target, format_args!("{prefix}{err}"));
}
const fn effective_one_way_delay(rtt: Duration) -> Option<Duration> {
    rtt.checked_div(HALF_RTT_DIVISOR)
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
fn decide_countdown_action(
    target_time: SystemTime,
    current_server_time: SystemTime,
    one_way_delay: Duration,
) -> CountdownDecision {
    match target_time.duration_since(current_server_time) {
        Ok(duration_until_target) if duration_until_target <= one_way_delay => {
            CountdownDecision::TriggerWithRemaining(duration_until_target)
        }
        Err(late) => CountdownDecision::TriggerLate(late.duration()),
        Ok(_) => CountdownDecision::Wait,
    }
}
fn append_fmt(target: &mut String, args: fmt::Arguments<'_>) {
    match FmtWrite::write_fmt(target, args) {
        Ok(()) | Err(_) => {}
    }
}
fn transition_to_retry(msg: &str) -> ActivityTransition<'_> {
    let now = Instant::now();
    let Some(retry_at) = now.checked_add(RETRY_DELAY) else {
        return ActivityTransition {
            activity: Activity::Retrying { retry_at: now },
            message: Some(msg),
        };
    };
    ActivityTransition {
        activity: Activity::Retrying { retry_at },
        message: Some(msg),
    }
}
