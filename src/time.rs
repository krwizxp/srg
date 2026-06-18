use self::{
    activity::{Activity, CountdownDecision},
    sample::{TCP_LINE_BUFFER_CAPACITY, fetch_server_time_sample},
    util::blend_weighted_nanos,
};
cfg_select! {
    target_os = "windows" => {
        use self::timer_resolution::{
            HighResTimerGuard, TARGET_PERIOD_MS, TIMERR_NOERROR, time_begin_period,
        };
    }
    _ => {}
}
use crate::{buffmt::ByteCursor, write_line_best_effort};
use alloc::{borrow::Cow, fmt, sync::Arc};
use core::{
    error::Error, fmt::Write as FmtWrite, ops::Mul as NumericMul, range::Range,
    result::Result as CoreResult, time::Duration,
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
mod activity;
mod address;
mod display;
mod http_date;
mod native_http;
mod sample;
mod util;
cfg_select! {
    target_os = "windows" => {
        mod timer_resolution;
    }
    _ => {}
}
cfg_select! {
    target_os = "linux" => {
        mod linux_input;
        mod native_input {
            pub(super) use super::linux_input::{InputAction, PreparedInput};
        }
    }
    target_os = "macos" => {
        mod macos_input;
        mod native_input {
            pub(super) use super::macos_input::{InputAction, PreparedInput};
        }
    }
    target_os = "windows" => {
        mod windows_input;
        mod native_input {
            pub(super) use super::windows_input::{InputAction, PreparedInput};
        }
    }
    _ => {
        mod native_input {
            use std::io;
            #[derive(Clone, Copy)]
            pub(super) enum InputAction {
                F5Press,
                MouseClick,
            }
            pub(super) struct PreparedInput;
            impl PreparedInput {
                pub(super) const EMPTY: Self = Self;
                pub(super) fn prepare(
                    &mut self,
                    _action: Option<InputAction>,
                    _err: &mut dyn io::Write,
                ) {
                    *self = Self;
                }
                pub(super) const fn reset(&mut self) {
                    *self = Self;
                }
                pub(super) fn send(
                    &mut self,
                    _action: InputAction,
                    _err: &mut dyn io::Write,
                ) {
                    *self = Self;
                }
            }
        }
    }
}
const FULL_SYNC_INTERVAL: Duration = Duration::from_mins(5);
const RETRY_DELAY: Duration = Duration::from_secs(10);
const TCP_TIMEOUT: Duration = Duration::from_secs(5);
const ENTER_BUFFER_CAPACITY: usize = 8;
const ENTER_BUFFER_READ_LIMIT: u64 = 9;
const ENTER_INPUT_TOO_LONG: &str = "서버 시간 종료 입력이 너무 깁니다.";
const ENTER_THREAD_PANIC: &str = "입력 대기 스레드 패닉 발생";
const NUM_SAMPLES: usize = 10;
const FINAL_COUNTDOWN_RTT_ALPHA_NUM: u32 = 7;
const FINAL_COUNTDOWN_RTT_ALPHA_DENOM: u32 = 10;
const MAX_CALIBRATION_FAILURES: u32 = 100;
const HTTP_SCHEME_PREFIX: &str = "http://";
const HTTP_SCHEME_PREFIX_LEN: usize = HTTP_SCHEME_PREFIX.len();
const HTTPS_SCHEME_PREFIX: &str = "https://";
const HTTPS_SCHEME_PREFIX_LEN: usize = HTTPS_SCHEME_PREFIX.len();
const KST_OFFSET: Duration = Duration::from_hours(9);
pub const KST_OFFSET_SECS_U64: u64 = KST_OFFSET.as_secs();
const KST_OFFSET_SECS: i64 = KST_OFFSET_SECS_U64.cast_signed();
const DISPLAY_INTERVAL: Duration = Duration::from_millis(16);
const ADAPTIVE_POLL_INTERVAL: Duration = Duration::from_millis(10);
const COUNTDOWN_DELAY_ERROR: &str = "카운트다운 지연 계산 실패";
const DISPLAY_STATUS_PREFIX: &str = "\r서버 시간: ";
const DISPLAY_UPDATE_INTERVAL: Duration = Duration::from_millis(45);
const FINAL_COUNTDOWN_SAMPLE_ERROR_MESSAGE_INTERVAL: Duration = Duration::from_secs(1);
const FINAL_COUNTDOWN_WINDOW: Duration = Duration::from_secs(10);
const HALF_RTT_DIVISOR: u32 = 2;
const MESSAGE_BUFFER_CAPACITY: usize = 256;
const MILLIS_PER_SECOND_F64: f64 = 1000.0;
const MIN_TRANSFER_TIME: Duration = Duration::from_micros(1);
const RTT_TRIM_DIVISOR: usize = 5;
type BoxError = Box<dyn Error + Send + Sync>;
type Result<T> = CoreResult<T, TimeError>;
#[derive(Clone, Copy, Debug)]
pub enum TriggerAction {
    F5Press,
    LeftClick,
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
pub struct TimeError {
    detail: Cow<'static, str>,
    io_kind: Option<io::ErrorKind>,
    kind: TimeErrorKind,
    source: Option<BoxError>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UrlScheme {
    Http,
    Https,
}
#[derive(Clone, Debug)]
pub struct ParsedServer {
    host: String,
    http_url: String,
    literal_tcp_socket_addr: Option<net::SocketAddr>,
    port: u16,
    scheme: Option<UrlScheme>,
    secure_url: String,
    tcp_host_header: String,
}
pub struct ServerTimeSession {
    pub host: ParsedServer,
    pub now: Instant,
    pub target_time: Option<SystemTime>,
    pub trigger_action: Option<TriggerAction>,
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
            source: None,
        }
    }
    fn parse(detail: impl Into<Cow<'static, str>>) -> Self {
        Self::new(TimeErrorKind::Parse, detail)
    }
}
impl From<io::Error> for TimeError {
    fn from(err: io::Error) -> Self {
        let io_kind = err.kind();
        let detail = owned_time_error_detail(&err);
        Self {
            kind: TimeErrorKind::Io,
            detail,
            io_kind: Some(io_kind),
            source: Some(Box::new(err)),
        }
    }
}
impl From<SystemTimeError> for TimeError {
    fn from(err: SystemTimeError) -> Self {
        let detail = owned_time_error_detail(&err);
        Self {
            kind: TimeErrorKind::Time,
            detail,
            io_kind: None,
            source: Some(Box::new(err)),
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
            TimeErrorKind::NativeHttp => write!(f, "native HTTP 요청 실패: {}", self.detail),
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
impl TriggerAction {
    const fn native_input_action(self) -> native_input::InputAction {
        match self {
            Self::LeftClick => native_input::InputAction::MouseClick,
            Self::F5Press => native_input::InputAction::F5Press,
        }
    }
}
struct AppState {
    baseline_rtt: Option<Duration>,
    baseline_rtt_attempts: usize,
    baseline_rtt_next_sample_at: Instant,
    baseline_rtt_samples: [TimeSample; NUM_SAMPLES],
    baseline_rtt_valid_count: usize,
    calibration_failure_count: u32,
    final_countdown_next_sample_error_message_at: Option<Instant>,
    final_countdown_sampler: Option<FinalCountdownSampler>,
    final_countdown_sampler_unavailable: bool,
    #[cfg(target_os = "windows")]
    high_res_timer_guard: Option<HighResTimerGuard>,
    host: ParsedServer,
    last_sample: Option<TimeSample>,
    live_rtt: Option<Duration>,
    next_full_sync_at: Instant,
    server_time: Option<ServerTime>,
    target_time: Option<SystemTime>,
    trigger_action: Option<TriggerAction>,
}
struct LoopRuntime<'runtime> {
    err: &'runtime mut dyn io::Write,
    network_context: &'runtime mut NetworkContext,
    out: &'runtime mut dyn io::Write,
    prepared_input: &'runtime mut native_input::PreparedInput,
}
struct FinalCountdownSampler {
    join_handle: thread::JoinHandle<()>,
    sample_slot: Arc<Mutex<FinalCountdownSampleSlot>>,
    stop_sender: mpsc::Sender<()>,
}
struct FinalCountdownSampleSlot {
    latest_error: Option<TimeError>,
    latest_success: Option<TimeSample>,
}
enum FinalCountdownSamplePoll {
    Disconnected,
    Empty,
    Sample(Result<TimeSample>),
}
struct CachedTcpSocketAddr {
    addr: net::SocketAddr,
    host: String,
    port: u16,
}
struct NetworkContext {
    cached_tcp_socket_addr: Option<CachedTcpSocketAddr>,
    native_http: native_http::NativeHttp,
    tcp_line_buffer: Vec<u8>,
}
impl Drop for FinalCountdownSampler {
    fn drop(&mut self) {
        let _send_result = self.stop_sender.send(());
    }
}
impl NetworkContext {
    fn new() -> Result<Self> {
        let mut tcp_line_buffer = Vec::new();
        tcp_line_buffer
            .try_reserve(TCP_LINE_BUFFER_CAPACITY)
            .map_err(|source| TimeError::parse(format!("buffer 메모리 확보 실패: {source}")))?;
        Ok(Self {
            cached_tcp_socket_addr: None,
            native_http: native_http::NativeHttp::default(),
            tcp_line_buffer,
        })
    }
}
impl ServerTimeSession {
    pub(super) fn run_loop(self, out: &mut dyn io::Write, err: &mut dyn io::Write) -> Result<()> {
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
            calibration_failure_count: 0,
            final_countdown_next_sample_error_message_at: None,
            final_countdown_sampler: None,
            final_countdown_sampler_unavailable: false,
            #[cfg(target_os = "windows")]
            high_res_timer_guard: None,
            host: self.host,
            last_sample: None,
            live_rtt: None,
            next_full_sync_at: self.now,
            server_time: None,
            target_time: self.target_time,
            trigger_action: self.trigger_action,
        };
        app_state.run_loop(out, err)
    }
}
impl AppState {
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
            write_line_best_effort(out, format_args!("1단계: RTT 기준값 측정을 시작합니다..."));
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
    fn begin_final_countdown_sampling(&mut self) -> Result<()> {
        if self.final_countdown_sampler.is_some() || self.final_countdown_sampler_unavailable {
            return Ok(());
        }
        let shared_slot = Arc::new(Mutex::new(FinalCountdownSampleSlot {
            latest_error: None,
            latest_success: None,
        }));
        let worker_slot = Arc::clone(&shared_slot);
        let (stop_sender, stop_receiver) = mpsc::channel();
        let host = self.host.clone();
        let join_handle = thread::Builder::new()
            .name(String::from("srg-final-countdown-sampler"))
            .spawn(move || {
                let mut network_context = match NetworkContext::new() {
                    Ok(context) => context,
                    Err(fetch_err) => {
                        if let Ok(mut slot) = worker_slot.lock() {
                            slot.latest_error = Some(fetch_err);
                        }
                        return;
                    }
                };
                loop {
                    match stop_receiver.try_recv() {
                        Ok(()) | Err(mpsc::TryRecvError::Disconnected) => return,
                        Err(mpsc::TryRecvError::Empty) => {}
                    }
                    let sample_result = fetch_server_time_sample(&host, &mut network_context);
                    let Ok(mut slot) = worker_slot.lock() else {
                        return;
                    };
                    match sample_result {
                        Ok(sample) => slot.latest_success = Some(sample),
                        Err(fetch_err) => slot.latest_error = Some(fetch_err),
                    }
                    match stop_receiver.recv_timeout(ADAPTIVE_POLL_INTERVAL) {
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => return,
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                }
            })?;
        self.final_countdown_sampler = Some(FinalCountdownSampler {
            join_handle,
            sample_slot: shared_slot,
            stop_sender,
        });
        Ok(())
    }
    fn disable_final_countdown_sampling(&mut self) {
        self.final_countdown_sampler = None;
        self.final_countdown_sampler_unavailable = true;
    }
    fn end_final_countdown_sampling(&mut self) {
        self.final_countdown_sampler = None;
        self.final_countdown_next_sample_error_message_at = None;
        self.final_countdown_sampler_unavailable = false;
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
        self.calibration_failure_count = 0;
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
        _msg_buf: &'message mut str,
        net_ctx: &mut NetworkContext,
    ) -> ActivityTransition<'message> {
        let Ok(current_sample) = fetch_server_time_sample(&self.host, net_ctx) else {
            let Some(next_failure_count) = self.calibration_failure_count.checked_add(1) else {
                return transition_to_retry(
                    "정밀 보정 실패 횟수 계산 중 overflow가 발생했습니다. 전체 보정을 다시 시작합니다.",
                );
            };
            self.calibration_failure_count = next_failure_count;
            if self.calibration_failure_count >= MAX_CALIBRATION_FAILURES {
                return transition_to_retry(
                    "정밀 보정 중 서버 응답을 지속적으로 받지 못했습니다. 전체 보정을 다시 시작합니다.",
                );
            }
            return ActivityTransition {
                activity: Activity::CalibrateOnTick,
                message: None,
            };
        };
        self.calibration_failure_count = 0;
        if let Some(prev_sample) = self.last_sample
            && let Some(baseline_rtt) = self.baseline_rtt
            && let Ok(prev_dur) = prev_sample.server_time.duration_since(UNIX_EPOCH)
            && let Ok(current_dur) = current_sample.server_time.duration_since(UNIX_EPOCH)
            && current_dur.as_secs().checked_sub(prev_dur.as_secs()) == Some(1)
        {
            let calibrated_server_time = (|| -> Result<ServerTime> {
                let mut server_time_at_tick = current_sample.server_time;
                let since_epoch = server_time_at_tick.duration_since(UNIX_EPOCH)?;
                let nanos_to_subtract = since_epoch.subsec_nanos();
                let nanos_to_subtract_duration = Duration::from_nanos(u64::from(nanos_to_subtract));
                server_time_at_tick = server_time_at_tick
                    .checked_sub(nanos_to_subtract_duration)
                    .ok_or_else(|| TimeError::parse("서버 시각 보정 중 범위 오류"))?;
                let one_way_delay = effective_one_way_delay(current_sample.rtt)
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
                let Some(next_full_sync_at) = current_sample
                    .response_received_inst
                    .checked_add(FULL_SYNC_INTERVAL)
                else {
                    return transition_to_retry("다음 전체 동기화 시각 계산 실패.");
                };
                self.next_full_sync_at = next_full_sync_at;
                return ActivityTransition {
                    activity: Activity::Predicting,
                    message: Some("[성공] 정밀 보정 완료!"),
                };
            }
        }
        self.last_sample = Some(current_sample);
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
        let Some(st) = self.server_time.as_ref() else {
            return ActivityTransition {
                activity: Activity::MeasureBaselineRtt,
                message: Some("[오류] 내부 상태 불일치: server_time 없음"),
            };
        };
        let now = Instant::now();
        let current_server_time = st.current_server_time_at(now);
        let estimated_rtt = self.live_rtt.unwrap_or(st.baseline_rtt);
        let sample_error_reported = match self.poll_final_countdown_sample() {
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
        let Some(estimated_one_way_delay) = effective_one_way_delay(estimated_rtt) else {
            msg_buf.push_str(COUNTDOWN_DELAY_ERROR);
            return ActivityTransition {
                activity: Activity::FinalCountdown { target_time },
                message: Some(msg_buf),
            };
        };
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
        if let Err(start_err) = self.begin_final_countdown_sampling() {
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
        *server_time = server_time.recalibrate_with_rtt(sample.rtt);
        let sample_now = Instant::now();
        let sampled_server_time = server_time.current_server_time_at(sample_now);
        let old_rtt = self.live_rtt.unwrap_or(sample.rtt);
        let new_rtt_nanos = blend_weighted_nanos(
            old_rtt.as_nanos(),
            sample.rtt.as_nanos(),
            FINAL_COUNTDOWN_RTT_ALPHA_NUM,
            FINAL_COUNTDOWN_RTT_ALPHA_DENOM,
        );
        let live_rtt = Duration::from_nanos_u128(new_rtt_nanos);
        self.live_rtt = Some(live_rtt);
        let effective_rtt = live_rtt.max(sample.rtt);
        let Some(one_way_delay) = effective_one_way_delay(effective_rtt) else {
            msg_buf.push_str(COUNTDOWN_DELAY_ERROR);
            return ActivityTransition {
                activity: Activity::FinalCountdown { target_time },
                message: Some(msg_buf),
            };
        };
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
            CountdownTriggerSource::Sampled { rtt: sample.rtt },
            prepared_input,
            err,
        )
    }
    fn handle_measure_baseline_rtt<'message>(
        &mut self,
        msg_buf: &'message mut String,
        net_ctx: &mut NetworkContext,
        now: Instant,
        out: &mut dyn io::Write,
    ) -> ActivityTransition<'message> {
        if self.baseline_rtt_attempts == 0 {
            self.begin_baseline_rtt_measurement(now, out);
        }
        if now < self.baseline_rtt_next_sample_at {
            return ActivityTransition {
                activity: Activity::MeasureBaselineRtt,
                message: None,
            };
        }
        let attempt_index = self.baseline_rtt_attempts;
        let next_sample_base = match fetch_server_time_sample(&self.host, net_ctx) {
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
        _msg_buf: &'message mut String,
        now: Instant,
    ) -> ActivityTransition<'message> {
        let Some(server_time) = self.server_time.as_ref() else {
            return ActivityTransition {
                activity: Activity::MeasureBaselineRtt,
                message: None,
            };
        };
        let estimated_server_time = server_time.current_server_time_at(now);
        if let Some(target_time) = self.target_time.take_if(|target| {
            target
                .duration_since(estimated_server_time)
                .map_or(true, |duration_until_target| {
                    duration_until_target <= FINAL_COUNTDOWN_WINDOW
                })
        }) {
            self.live_rtt = Some(server_time.baseline_rtt);
            return ActivityTransition {
                activity: Activity::FinalCountdown { target_time },
                message: Some("최종 카운트다운 시작!"),
            };
        }
        if now >= self.next_full_sync_at {
            self.server_time = None;
            self.baseline_rtt = None;
            self.live_rtt = None;
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
    fn next_activity<'message>(
        &mut self,
        activity: &Activity,
        message_buffer: &'message mut String,
        now: Instant,
        runtime: &mut LoopRuntime<'_>,
    ) -> ActivityTransition<'message> {
        match *activity {
            Activity::MeasureBaselineRtt => self.handle_measure_baseline_rtt(
                message_buffer,
                runtime.network_context,
                now,
                runtime.out,
            ),
            Activity::CalibrateOnTick => {
                self.handle_calibrate_on_tick(message_buffer, runtime.network_context)
            }
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
    fn poll_final_countdown_sample(&mut self) -> FinalCountdownSamplePoll {
        let Some(sampler) = self.final_countdown_sampler.as_ref() else {
            return FinalCountdownSamplePoll::Empty;
        };
        let sampler_finished = sampler.join_handle.is_finished();
        let sample_slot = Arc::clone(&sampler.sample_slot);
        let mut slot = match sample_slot.try_lock() {
            Ok(value) => value,
            Err(TryLockError::WouldBlock) => {
                if sampler_finished {
                    self.disable_final_countdown_sampling();
                    return FinalCountdownSamplePoll::Disconnected;
                }
                return FinalCountdownSamplePoll::Empty;
            }
            Err(TryLockError::Poisoned(_)) => {
                self.disable_final_countdown_sampling();
                return FinalCountdownSamplePoll::Sample(Err(TimeError::parse(
                    "카운트다운 샘플 상태 잠금 실패",
                )));
            }
        };
        let latest_success = slot.latest_success.take();
        let latest_error = slot.latest_error.take();
        drop(slot);
        if sampler_finished {
            self.disable_final_countdown_sampling();
        }
        if let Some(sample) = latest_success {
            return FinalCountdownSamplePoll::Sample(Ok(sample));
        }
        if let Some(fetch_err) = latest_error {
            return FinalCountdownSamplePoll::Sample(Err(fetch_err));
        }
        if sampler_finished {
            return FinalCountdownSamplePoll::Disconnected;
        }
        FinalCountdownSamplePoll::Empty
    }
    fn run_loop(&mut self, out: &mut dyn io::Write, err: &mut dyn io::Write) -> Result<()> {
        out.write_all("\n서버 시간 확인을 시작합니다... (Enter를 누르면 종료)\n".as_bytes())?;
        let (tx, rx) = mpsc::channel();
        let input_thread = thread::spawn(move || -> IoResult<()> {
            let mut line = Vec::new();
            line.try_reserve(ENTER_BUFFER_CAPACITY)
                .map_err(io::Error::other)?;
            let mut stdin_lock = io::stdin().lock();
            let mut limited_stdin = IoRead::take(&mut stdin_lock, ENTER_BUFFER_READ_LIMIT);
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
            .try_reserve(MESSAGE_BUFFER_CAPACITY)
            .map_err(|source| TimeError::parse(format!("buffer 메모리 확보 실패: {source}")))?;
        let mut network_context = NetworkContext::new()?;
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
            let elapsed = pre_wait_now.duration_since(last_display_update);
            let remaining_display = DISPLAY_INTERVAL.saturating_sub(elapsed);
            let poll_timeout = activity_poll.min(remaining_display);
            match rx.recv_timeout(poll_timeout) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            let now = Instant::now();
            if now.duration_since(last_display_update) >= DISPLAY_INTERVAL
                && let Some(st) = self.server_time.as_ref()
            {
                if let Err(display_err) = (|| -> Result<()> {
                    let mut cur = ByteCursor::new(line_buf.as_mut_slice());
                    cur.write_bytes(DISPLAY_STATUS_PREFIX.as_bytes())?;
                    st.write_current_display_time_buf_at(&mut cur, now)?;
                    cur.write_bytes(b" \r")?;
                    out.write_all(cur.written_slice()?)?;
                    out.flush()?;
                    Ok(())
                })() {
                    out.write_all(DISPLAY_STATUS_PREFIX.as_bytes())?;
                    write!(out, "표시 버퍼 오류: {display_err}")?;
                    out.write_all(b" \r")?;
                    out.flush()?;
                }
                last_display_update = now;
            }
            message_buffer.clear();
            let transition = {
                let mut runtime = LoopRuntime {
                    err,
                    network_context: &mut network_context,
                    out,
                    prepared_input: &mut prepared_input,
                };
                self.next_activity(&activity, &mut message_buffer, now, &mut runtime)
            };
            let next_activity = transition.activity;
            self.sync_prepared_input_state(&activity, &next_activity, &mut prepared_input, err);
            cfg_select! {
                windows => {
                    self.sync_high_res_timer_state(&next_activity, err);
                }
                _ => {}
            }
            if let Some(console_msg) = transition.message {
                writeln!(out, "\n{console_msg}")?;
            }
            activity = next_activity;
        }
        input_thread
            .join()
            .map_err(|_panic_payload| TimeError::parse(ENTER_THREAD_PANIC))??;
        Ok(())
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
                // SAFETY: `time_begin_period` is a WinMM FFI call with a plain integer input.
                if unsafe { time_begin_period(TARGET_PERIOD_MS) } != TIMERR_NOERROR {
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
                }
                let guard = HighResTimerGuard;
                self.high_res_timer_guard = Some(guard);
            }
        }
        _ => {}
    }
    fn sync_prepared_input_state(
        &mut self,
        previous_activity: &Activity,
        next_activity: &Activity,
        prepared_input: &mut native_input::PreparedInput,
        err: &mut dyn io::Write,
    ) {
        let entering_final_countdown =
            next_activity.is_final_countdown() && !previous_activity.is_final_countdown();
        if entering_final_countdown {
            prepared_input.prepare(
                self.trigger_action.map(TriggerAction::native_input_action),
                err,
            );
            return;
        }
        if !next_activity.is_final_countdown() {
            self.end_final_countdown_sampling();
            prepared_input.reset();
        }
    }
    fn trigger_and_finish<'message>(
        &self,
        msg_buf: &'message mut String,
        log_message: fmt::Arguments,
        prepared_input: &mut native_input::PreparedInput,
        err: &mut dyn io::Write,
    ) -> ActivityTransition<'message> {
        if let Some(action) = self.trigger_action.map(TriggerAction::native_input_action) {
            prepared_input.send(action, err);
        }
        append_fmt(msg_buf, log_message);
        ActivityTransition {
            activity: Activity::Finished,
            message: Some(msg_buf),
        }
    }
    fn trigger_countdown_decision<'message>(
        &self,
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
                    "\n>>> 액션 실행! (예측값 기준, 목표 도달까지 {:.1}ms 남음) (지연 예측: {:.1}ms)",
                    duration_millis_f64(remaining),
                    duration_millis_f64(one_way_delay),
                ),
                prepared_input,
                err,
            ),
            (CountdownDecision::TriggerLate, CountdownTriggerSource::Estimated) => {
                self.trigger_and_finish(
                    msg_buf,
                    format_args!(
                        "\n>>> 액션 실행! (예측값 기준, 시간 초과) (지연 예측: {:.1}ms)",
                        duration_millis_f64(one_way_delay),
                    ),
                    prepared_input,
                    err,
                )
            }
            (
                CountdownDecision::TriggerWithRemaining(remaining),
                CountdownTriggerSource::Sampled { rtt },
            ) => self.trigger_and_finish(
                msg_buf,
                format_args!(
                    "\n>>> 액션 실행! (목표 도달까지 {:.1}ms 남음) (지연 예측: {:.1}ms, 실측 RTT: {:.1}ms)",
                    duration_millis_f64(remaining),
                    duration_millis_f64(one_way_delay),
                    duration_millis_f64(rtt)
                ),
                prepared_input,
                err,
            ),
            (CountdownDecision::TriggerLate, CountdownTriggerSource::Sampled { rtt }) => {
                self.trigger_and_finish(
                    msg_buf,
                    format_args!(
                        "\n>>> 액션 실행! (시간 초과) (지연 예측: {:.1}ms, 실측 RTT: {:.1}ms)",
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
}
fn owned_time_error_detail(err: impl fmt::Display) -> Cow<'static, str> {
    Cow::Owned(err.to_string())
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
fn decide_countdown_action(
    target_time: SystemTime,
    current_server_time: SystemTime,
    one_way_delay: Duration,
) -> CountdownDecision {
    match target_time.duration_since(current_server_time) {
        Ok(duration_until_target) if duration_until_target <= one_way_delay => {
            CountdownDecision::TriggerWithRemaining(duration_until_target)
        }
        Err(_) => CountdownDecision::TriggerLate,
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
