use self::util::{blend_rtt, parse_u32_digits};
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
    sync::mpsc,
    thread,
    time::{Instant, SystemTime, SystemTimeError, UNIX_EPOCH},
};
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
mod util;
cfg_select! {
    target_os = "windows" => {
        mod timer_resolution;
    }
    _ => {}
}
const FULL_SYNC_INTERVAL: Duration = Duration::from_mins(5);
const ADAPTIVE_POLL_INTERVAL: Duration = Duration::from_millis(10);
const PASSIVE_POLL_INTERVAL: Duration = Duration::from_millis(45);
const RETRY_DELAY: Duration = Duration::from_secs(10);
const SAMPLE_WORKER_RESTARTED_MESSAGE: &str = "서버 시간 샘플 worker가 종료되어 다시 시작했습니다.";
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
    #[cfg(target_os = "windows")]
    host_wide: Vec<u16>,
    #[cfg(target_os = "windows")]
    port: u16,
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    request_target: CString,
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
impl<'message> ActivityTransition<'message> {
    const fn message(activity: Activity, message: &'message str) -> Self {
        Self {
            activity,
            message: Some(message),
        }
    }
    const fn stay(activity: Activity) -> Self {
        Self {
            activity,
            message: None,
        }
    }
}
#[derive(Clone, Copy, Debug)]
struct TimeSample {
    response_received_inst: Instant,
    rtt: Duration,
    server_time: SystemTime,
}
#[derive(Debug)]
struct BaselineRttState {
    attempts: usize,
    had_previous_sample: bool,
    next_sample_at: Instant,
    pending_generation: Option<u64>,
    samples: [Option<TimeSample>; NUM_SAMPLES],
    started: bool,
}
#[derive(Debug)]
struct CalibrationState {
    baseline_rtt: Duration,
    pending_generation: Option<u64>,
    previous_sample: TimeSample,
    started_at: Instant,
}
#[derive(Clone, Copy, Debug)]
struct FinalCountdownState {
    last_sample_error_message_at: Option<Instant>,
    live_rtt: Duration,
    server_time: ServerTime,
    target_time: SystemTime,
}
struct FinalSamplingState {
    interval: Duration,
    next_sample_at: Instant,
    pending_generation: Option<u64>,
}
#[derive(Clone, Copy, Debug)]
struct ServerTime {
    anchor_instant: Instant,
    anchor_time: SystemTime,
    baseline_rtt: Duration,
}
#[derive(Debug)]
enum Activity {
    CalibrateOnTick(CalibrationState),
    FinalCountdown(FinalCountdownState),
    MeasureBaselineRtt(Box<BaselineRttState>),
    Predicting {
        server_time: ServerTime,
    },
    Retrying {
        had_previous_sample: bool,
        started_at: Instant,
    },
}
enum CountdownTrigger {
    Late(Duration),
    WithRemaining(Duration),
}
impl Activity {
    const fn is_final_countdown(&self) -> bool {
        matches!(self, Self::FinalCountdown(_))
    }
    fn measure_baseline(now: Instant, had_previous_sample: bool) -> Self {
        Self::MeasureBaselineRtt(Box::new(BaselineRttState {
            attempts: 0,
            had_previous_sample,
            next_sample_at: now,
            pending_generation: None,
            samples: [None; NUM_SAMPLES],
            started: false,
        }))
    }
    const fn poll_interval(&self) -> Duration {
        match self {
            &Self::CalibrateOnTick(_) | &Self::FinalCountdown(_) | &Self::MeasureBaselineRtt(_) => {
                ADAPTIVE_POLL_INTERVAL
            }
            &Self::Predicting { .. } | &Self::Retrying { .. } => PASSIVE_POLL_INTERVAL,
        }
    }
    const fn server_time(&self) -> Option<ServerTime> {
        match *self {
            Self::FinalCountdown(state) => Some(state.server_time),
            Self::Predicting { server_time } => Some(server_time),
            Self::CalibrateOnTick(_) | Self::MeasureBaselineRtt(_) | Self::Retrying { .. } => None,
        }
    }
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
        let prefix = match self.kind {
            TimeErrorKind::HeaderNotFound => {
                return write!(f, "{} 헤더를 찾을 수 없음", self.detail);
            }
            TimeErrorKind::Io => "I/O 오류",
            TimeErrorKind::NativeHttp => "native HTTP 요청 실패",
            TimeErrorKind::Parse => "파싱 오류",
            TimeErrorKind::Time => "시스템 시간 오류",
        };
        f.write_str(prefix)?;
        if !self.detail.is_empty() {
            write!(f, ": {}", self.detail)?;
        }
        if let Some(source) = self.source.as_ref() {
            write!(f, ": {source}")?;
        }
        Ok(())
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
    fn ensure_fetch(&mut self, pending_generation: &mut Option<u64>) -> Result<()> {
        if pending_generation.is_none() {
            *pending_generation = self.fetch()?;
        }
        Ok(())
    }
    fn fetch(&mut self) -> Result<Option<u64>> {
        let generation = self.generation.wrapping_add(1);
        match self.command_sender.try_send(generation) {
            Ok(()) => {
                self.generation = generation;
                Ok(Some(generation))
            }
            Err(mpsc::TrySendError::Full(_)) => Ok(None),
            Err(mpsc::TrySendError::Disconnected(_)) => {
                self.respawn()?;
                Err(TimeError::parse(SAMPLE_WORKER_RESTARTED_MESSAGE))
            }
        }
    }
    fn poll_fetch(
        &mut self,
        pending_generation: &mut Option<u64>,
    ) -> Result<Option<Result<TimeSample>>> {
        let Some(expected_generation) = *pending_generation else {
            loop {
                match self.response_receiver.try_recv() {
                    Ok(_) => {}
                    Err(mpsc::TryRecvError::Empty) => return Ok(None),
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.respawn()?;
                        return Err(TimeError::parse(SAMPLE_WORKER_RESTARTED_MESSAGE));
                    }
                }
            }
        };
        loop {
            match self.response_receiver.try_recv() {
                Ok(response) => {
                    if response.generation == expected_generation {
                        *pending_generation = None;
                        return Ok(Some(response.result));
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    return Ok(None);
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    *pending_generation = None;
                    self.respawn()?;
                    return Err(TimeError::parse(SAMPLE_WORKER_RESTARTED_MESSAGE));
                }
            }
        }
    }
    fn respawn(&mut self) -> Result<()> {
        let (command_sender, response_receiver) = sample_worker_channels(Arc::clone(&self.host))?;
        self.command_sender = command_sender;
        self.generation = 0;
        self.response_receiver = response_receiver;
        Ok(())
    }
}
struct AppState<'worker> {
    final_sampling: Option<FinalSamplingState>,
    #[cfg(target_os = "windows")]
    high_res_timer_guard: Option<HighResTimerGuard>,
    last_full_sync_at: Instant,
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
    command_sender: mpsc::SyncSender<u64>,
    generation: u64,
    host: Arc<ParsedServer>,
    response_receiver: mpsc::Receiver<SampleWorkerResult>,
}
struct FinalCountdownDeadline {
    one_way_delay: Duration,
    server_time: ServerTime,
    source: CountdownTriggerSource,
    target_time: SystemTime,
    trigger_instant: Instant,
}
struct SampleWorkerResult {
    generation: u64,
    result: Result<TimeSample>,
}
impl ServerTimeSession {
    pub(super) fn run_loop(self, out: &mut dyn io::Write, err: &mut dyn io::Write) -> Result<()> {
        let now = Instant::now();
        let shared_host = Arc::new(self.host);
        let (command_sender, response_receiver) = sample_worker_channels(Arc::clone(&shared_host))?;
        let mut sample_worker = SampleWorker {
            command_sender,
            generation: 0,
            host: shared_host,
            response_receiver,
        };
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
            final_sampling: None,
            #[cfg(target_os = "windows")]
            high_res_timer_guard: None,
            last_full_sync_at: now,
            pending_target_time,
            sample_worker: &mut sample_worker,
            target_time: None,
            trigger_action,
        };
        app_state.run_loop(self.stop_after, out, err)
    }
}
impl AppState<'_> {
    fn accept_calibration_tick(
        &mut self,
        calibration: &CalibrationState,
        current_sample: TimeSample,
    ) -> Option<ServerTime> {
        let prev_dur = calibration
            .previous_sample
            .server_time
            .duration_since(UNIX_EPOCH)
            .ok()?;
        let current_dur = current_sample.server_time.duration_since(UNIX_EPOCH).ok()?;
        if current_dur.as_secs().checked_sub(prev_dur.as_secs()) != Some(1) {
            return None;
        }
        let since_epoch = current_sample.server_time.duration_since(UNIX_EPOCH).ok()?;
        let nanos_to_subtract = Duration::from_nanos(u64::from(since_epoch.subsec_nanos()));
        let anchor_time = current_sample.server_time.checked_sub(nanos_to_subtract)?;
        let one_way_delay = effective_one_way_delay(current_sample.rtt);
        let anchor_instant = current_sample
            .response_received_inst
            .checked_sub(one_way_delay)
            .unwrap_or(current_sample.response_received_inst);
        self.last_full_sync_at = current_sample.response_received_inst;
        Some(ServerTime {
            anchor_instant,
            anchor_time,
            baseline_rtt: calibration.baseline_rtt,
        })
    }
    fn begin_final_countdown_sampling(
        &mut self,
        interval: Duration,
    ) -> Result<Option<Result<TimeSample>>> {
        let now = Instant::now();
        let next_sample_at = now
            .checked_add(interval)
            .ok_or_else(|| TimeError::parse("카운트다운 다음 샘플 시각 계산 실패"))?;
        let sampling = self.final_sampling.get_or_insert(FinalSamplingState {
            interval,
            next_sample_at: now,
            pending_generation: None,
        });
        if sampling.interval != interval {
            sampling.interval = interval;
            if sampling.pending_generation.is_none() {
                sampling.next_sample_at = next_sample_at;
            }
        }
        let latest_sample = self
            .sample_worker
            .poll_fetch(&mut sampling.pending_generation)?;
        if latest_sample.is_some() {
            sampling.next_sample_at = next_sample_at;
        }
        if sampling.pending_generation.is_none() && now >= sampling.next_sample_at {
            self.sample_worker
                .ensure_fetch(&mut sampling.pending_generation)?;
        }
        Ok(latest_sample)
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
        let CivilDate { day, month, year } = http_date::civil_from_days(day_index);
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
    const fn end_final_countdown_sampling(&mut self) {
        self.final_sampling = None;
    }
    fn handle_calibrate_on_tick<'message>(
        &mut self,
        mut calibration: CalibrationState,
        msg_buf: &'message mut String,
    ) -> ActivityTransition<'message> {
        let now = Instant::now();
        if now.saturating_duration_since(calibration.started_at) >= CALIBRATION_TIMEOUT {
            return transition_to_retry(CALIBRATION_TIMEOUT_MESSAGE, true);
        }
        let sample_poll = match self
            .sample_worker
            .poll_fetch(&mut calibration.pending_generation)
        {
            Ok(sample) => sample,
            Err(worker_err) => {
                append_error_detail(msg_buf, "서버 시간 샘플 worker 오류: ", worker_err);
                return transition_to_retry(msg_buf, true);
            }
        };
        let Some(sample_result) = sample_poll else {
            if let Err(start_err) = self
                .sample_worker
                .ensure_fetch(&mut calibration.pending_generation)
            {
                append_error_detail(msg_buf, "정밀 보정 샘플 요청 실패: ", start_err);
                return transition_to_retry(msg_buf, true);
            }
            return ActivityTransition::stay(Activity::CalibrateOnTick(calibration));
        };
        let completed_at = Instant::now();
        if completed_at.saturating_duration_since(calibration.started_at) >= CALIBRATION_TIMEOUT {
            return transition_to_retry(CALIBRATION_TIMEOUT_MESSAGE, true);
        }
        let Ok(current_sample) = sample_result else {
            if let Err(start_err) = self
                .sample_worker
                .ensure_fetch(&mut calibration.pending_generation)
            {
                append_error_detail(msg_buf, "정밀 보정 샘플 요청 실패: ", start_err);
                return transition_to_retry(msg_buf, true);
            }
            return ActivityTransition::stay(Activity::CalibrateOnTick(calibration));
        };
        if let Some(server_time) = self.accept_calibration_tick(&calibration, current_sample) {
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
                    return transition_to_retry(msg_buf, true);
                }
                return ActivityTransition::message(Activity::Predicting { server_time }, msg_buf);
            }
            return ActivityTransition::message(
                Activity::Predicting { server_time },
                "[성공] 정밀 보정 완료!",
            );
        }
        calibration.previous_sample = current_sample;
        if Instant::now().saturating_duration_since(calibration.started_at) >= CALIBRATION_TIMEOUT {
            return transition_to_retry(CALIBRATION_TIMEOUT_MESSAGE, true);
        }
        if let Err(start_err) = self
            .sample_worker
            .ensure_fetch(&mut calibration.pending_generation)
        {
            append_error_detail(msg_buf, "정밀 보정 샘플 요청 실패: ", start_err);
            return transition_to_retry(msg_buf, true);
        }
        ActivityTransition::stay(Activity::CalibrateOnTick(calibration))
    }
    fn handle_final_countdown<'message>(
        &mut self,
        mut countdown: FinalCountdownState,
        msg_buf: &'message mut String,
        runtime: &mut LoopRuntime<'_>,
    ) -> ActivityTransition<'message> {
        let now = Instant::now();
        let server_time = countdown.server_time;
        let target_time = countdown.target_time;
        let current_server_time = server_time.current_server_time_at(now);
        let sample_interval = target_time.duration_since(current_server_time).map_or(
            FINAL_COUNTDOWN_SAMPLE_FINAL_INTERVAL,
            final_countdown_sample_interval,
        );
        let sample_error_reported = match self.begin_final_countdown_sampling(sample_interval) {
            Ok(Some(Ok(sample))) => {
                return self.handle_final_countdown_sample(countdown, msg_buf, sample, runtime);
            }
            Ok(Some(Err(fetch_err))) => append_final_countdown_sample_error(
                &mut countdown.last_sample_error_message_at,
                now,
                msg_buf,
                fetch_err,
            ),
            Err(sampling_err) => append_final_countdown_sample_error(
                &mut countdown.last_sample_error_message_at,
                now,
                msg_buf,
                sampling_err,
            ),
            Ok(None) => false,
        };
        let estimated_one_way_delay = effective_one_way_delay(countdown.live_rtt);
        let Some(trigger_instant) =
            trigger_instant_for_target(server_time, target_time, estimated_one_way_delay, now)
        else {
            msg_buf.push_str("카운트다운 실행 시각 계산 실패");
            return ActivityTransition::message(Activity::FinalCountdown(countdown), msg_buf);
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
        if sample_error_reported {
            return ActivityTransition::message(Activity::FinalCountdown(countdown), msg_buf);
        }
        ActivityTransition::stay(Activity::FinalCountdown(countdown))
    }
    fn handle_final_countdown_sample<'message>(
        &mut self,
        mut countdown: FinalCountdownState,
        msg_buf: &'message mut String,
        sample: TimeSample,
        runtime: &mut LoopRuntime<'_>,
    ) -> ActivityTransition<'message> {
        let sample_rtt = sample.rtt;
        let calibrated_server_time = countdown.server_time.recalibrate_with_rtt(sample_rtt);
        let sample_now = Instant::now();
        let live_rtt = blend_rtt::<7>(countdown.live_rtt, sample_rtt);
        countdown.live_rtt = live_rtt;
        countdown.server_time = calibrated_server_time;
        let effective_rtt = live_rtt.max(sample_rtt);
        let one_way_delay = effective_one_way_delay(effective_rtt);
        let Some(trigger_instant) = trigger_instant_for_target(
            calibrated_server_time,
            countdown.target_time,
            one_way_delay,
            sample_now,
        ) else {
            msg_buf.push_str("카운트다운 실행 시각 계산 실패");
            return ActivityTransition::message(Activity::FinalCountdown(countdown), msg_buf);
        };
        if trigger_instant.saturating_duration_since(sample_now) <= FINAL_COUNTDOWN_FREEZE_WINDOW {
            let deadline = FinalCountdownDeadline {
                one_way_delay,
                server_time: calibrated_server_time,
                source: CountdownTriggerSource::Sampled { rtt: sample_rtt },
                target_time: countdown.target_time,
                trigger_instant,
            };
            return self.trigger_final_countdown_deadline(&deadline, msg_buf, runtime);
        }
        ActivityTransition::stay(Activity::FinalCountdown(countdown))
    }
    fn handle_measure_baseline_rtt<'message>(
        &mut self,
        mut baseline: Box<BaselineRttState>,
        msg_buf: &'message mut String,
        now: Instant,
        out: &mut dyn io::Write,
    ) -> ActivityTransition<'message> {
        if !baseline.started {
            if !baseline.had_previous_sample {
                write_line_best_effort(out, format_args!("1단계: RTT 기준값 측정을 시작합니다…"));
            }
            baseline.started = true;
        }
        let had_pending_request = baseline.pending_generation.is_some();
        let sample_poll = match self
            .sample_worker
            .poll_fetch(&mut baseline.pending_generation)
        {
            Ok(sample) => sample,
            Err(worker_err) => {
                append_error_detail(msg_buf, "서버 시간 샘플 worker 오류: ", worker_err);
                return transition_to_retry(msg_buf, baseline.had_previous_sample);
            }
        };
        let Some(result) = sample_poll else {
            if had_pending_request || now < baseline.next_sample_at {
                return ActivityTransition::stay(Activity::MeasureBaselineRtt(baseline));
            }
            if let Err(start_err) = self
                .sample_worker
                .ensure_fetch(&mut baseline.pending_generation)
            {
                append_error_detail(msg_buf, "RTT 샘플 요청 실패: ", start_err);
                return transition_to_retry(msg_buf, baseline.had_previous_sample);
            }
            return ActivityTransition::stay(Activity::MeasureBaselineRtt(baseline));
        };
        let attempt_index = baseline.attempts;
        let next_sample_base = match result {
            Ok(sample) => {
                if sample.rtt > Duration::ZERO {
                    let Some(slot) = baseline.samples.get_mut(attempt_index) else {
                        return transition_to_retry(
                            "RTT 샘플 인덱스 계산 실패.",
                            baseline.had_previous_sample,
                        );
                    };
                    *slot = Some(sample);
                    baseline.had_previous_sample = true;
                }
                sample.response_received_inst
            }
            Err(err) => {
                append_error_detail(msg_buf, "RTT 샘플 수집 실패: ", err);
                return transition_to_retry(msg_buf, baseline.had_previous_sample);
            }
        };
        baseline.attempts = attempt_index.saturating_add(1);
        let Some(next_sample_at) = next_sample_base.checked_add(ADAPTIVE_POLL_INTERVAL) else {
            return transition_to_retry(
                "다음 RTT 샘플 시각 계산 실패.",
                baseline.had_previous_sample,
            );
        };
        baseline.next_sample_at = next_sample_at;
        if baseline.attempts < NUM_SAMPLES {
            return ActivityTransition::stay(Activity::MeasureBaselineRtt(baseline));
        }
        let Some(previous_sample) = baseline.samples.iter().rev().flatten().next().copied() else {
            return transition_to_retry(
                "유효한 RTT 샘플을 얻지 못했습니다.",
                baseline.had_previous_sample,
            );
        };
        let sample_count = baseline.samples.iter().flatten().count();
        let mut rtt_nanos = [u128::MAX; NUM_SAMPLES];
        for (slot, sample) in rtt_nanos.iter_mut().zip(baseline.samples.iter().flatten()) {
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
        append_fmt(
            msg_buf,
            format_args!(
                "[완료] RTT 기준값: {}ms. 2단계: 정밀 보정을 시작합니다.",
                baseline_rtt.as_millis()
            ),
        );
        ActivityTransition::message(
            Activity::CalibrateOnTick(CalibrationState {
                baseline_rtt,
                pending_generation: None,
                previous_sample,
                started_at: Instant::now(),
            }),
            msg_buf,
        )
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
                return ActivityTransition::message(Activity::Predicting { server_time }, msg_buf);
            }
        }
        if now.saturating_duration_since(self.last_full_sync_at) >= FULL_SYNC_INTERVAL
            && !protect_target
        {
            self.end_final_countdown_sampling();
            ActivityTransition::message(
                Activity::measure_baseline(now, true),
                "서버 시간 보정 주기 도래, 재보정 시작.",
            )
        } else {
            ActivityTransition::stay(Activity::Predicting { server_time })
        }
    }
    fn next_activity<'message>(
        &mut self,
        activity: Activity,
        message_buffer: &'message mut String,
        now: Instant,
        runtime: &mut LoopRuntime<'_>,
    ) -> ActivityTransition<'message> {
        match activity {
            Activity::MeasureBaselineRtt(baseline) => {
                self.handle_measure_baseline_rtt(baseline, message_buffer, now, runtime.out)
            }
            Activity::CalibrateOnTick(calibration) => {
                self.handle_calibrate_on_tick(calibration, message_buffer)
            }
            Activity::Predicting { server_time } => {
                let estimated_server_time = server_time.current_server_time_at(now);
                if let Some(target_time) = self.target_time.take_if(|target| {
                    target
                        .duration_since(estimated_server_time)
                        .ok()
                        .is_none_or(|remaining| remaining <= FINAL_COUNTDOWN_WINDOW)
                }) {
                    return ActivityTransition::message(
                        Activity::FinalCountdown(FinalCountdownState {
                            last_sample_error_message_at: None,
                            live_rtt: server_time.baseline_rtt,
                            server_time,
                            target_time,
                        }),
                        "최종 카운트다운 시작!",
                    );
                }
                self.handle_predicting(server_time, message_buffer, now)
            }
            Activity::FinalCountdown(countdown) => {
                self.handle_final_countdown(countdown, message_buffer, runtime)
            }
            Activity::Retrying {
                had_previous_sample,
                started_at,
            } => {
                if now.saturating_duration_since(started_at) >= RETRY_DELAY {
                    ActivityTransition::message(
                        Activity::measure_baseline(now, had_previous_sample),
                        "[재시도] 동기화를 다시 시작합니다.",
                    )
                } else {
                    ActivityTransition::stay(Activity::Retrying {
                        had_previous_sample,
                        started_at,
                    })
                }
            }
        }
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
        let mut last_display_update = Instant::now();
        let mut activity = Activity::measure_baseline(last_display_update, false);
        let mut line_buf = [0_u8; display::DISPLAY_LINE_BUF_LEN];
        loop {
            let pre_wait_now = Instant::now();
            if matches!(stop, LoopStop::Deadline(deadline) if pre_wait_now >= deadline) {
                break;
            }
            let activity_poll_timeout = activity.poll_interval();
            let mut poll_timeout = if Self::should_update_display(&activity, pre_wait_now) {
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
            let timing_sensitive_activity = activity.is_final_countdown()
                || matches!(
                    &activity,
                    Activity::Predicting { server_time }
                        if self.target_time.is_some_and(|target| {
                            target
                                .duration_since(server_time.current_server_time_at(now))
                                .ok()
                                .is_none_or(|remaining| remaining <= FINAL_COUNTDOWN_WINDOW)
                        })
                );
            if !timing_sensitive_activity {
                Self::run_loop_write_display_if_due(
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
                self.next_activity(activity, &mut message_buffer, now, &mut runtime)
            };
            let next_activity = transition.activity;
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
                Self::run_loop_write_display_if_due(
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
        display_activity: &Activity,
        now: Instant,
        last_update: &mut Instant,
        output: &mut dyn io::Write,
        buffer: &mut [u8],
    ) -> Result<()> {
        if now.saturating_duration_since(*last_update) >= DISPLAY_INTERVAL
            && Self::should_update_display(display_activity, now)
            && let Some(server_time) = display_activity.server_time()
        {
            let mut cur = ByteCursor::new(buffer);
            cur.write_bytes(DISPLAY_STATUS_PREFIX.as_bytes())?;
            server_time.write_current_display_time_buf_at(&mut cur, now)?;
            cur.write_bytes(b" \r")?;
            output.write_all(cur.written_slice()?)?;
            output.flush()?;
            *last_update = now;
        }
        Ok(())
    }
    fn should_update_display(activity: &Activity, now: Instant) -> bool {
        let Some(server_time) = activity.server_time() else {
            return false;
        };
        let Activity::FinalCountdown(countdown) = *activity else {
            return true;
        };
        let one_way_delay = effective_one_way_delay(countdown.live_rtt);
        let Some(trigger_instant) =
            trigger_instant_for_target(server_time, countdown.target_time, one_way_delay, now)
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
        ActivityTransition::message(Activity::Predicting { server_time }, msg_buf)
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
fn sample_worker_channels(
    host: Arc<ParsedServer>,
) -> Result<(mpsc::SyncSender<u64>, mpsc::Receiver<SampleWorkerResult>)> {
    let (command_sender, command_receiver) = mpsc::sync_channel(1);
    let (response_sender, response_receiver) = mpsc::channel();
    drop(
        thread::Builder::new()
            .name(String::from("srg-sample-worker"))
            .spawn(move || {
                let mut native_http = native_http::Client::default();
                while let Ok(generation) = command_receiver.recv() {
                    let context = match host.scheme {
                        UrlScheme::Http => "HTTP",
                        UrlScheme::Https => "HTTPS",
                    };
                    let result = native_http.fetch_head(&host, context);
                    if response_sender
                        .send(SampleWorkerResult { generation, result })
                        .is_err()
                    {
                        return;
                    }
                }
            })?,
    );
    Ok((command_sender, response_receiver))
}
fn duration_millis_f64(duration: Duration) -> f64 {
    NumericMul::mul(duration.as_secs_f64(), MILLIS_PER_SECOND_F64)
}
fn append_error_detail(target: &mut String, prefix: &str, err: impl fmt::Display) {
    append_fmt(target, format_args!("{prefix}{err}"));
}
fn append_final_countdown_sample_error(
    last_message_at: &mut Option<Instant>,
    now: Instant,
    msg_buf: &mut String,
    err: impl fmt::Display,
) -> bool {
    if last_message_at.is_some_and(|last| {
        now.saturating_duration_since(last) < FINAL_COUNTDOWN_SAMPLE_ERROR_MESSAGE_INTERVAL
    }) {
        return false;
    }
    *last_message_at = Some(now);
    append_error_detail(msg_buf, "카운트다운 샘플 획득 실패: ", err);
    true
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
fn transition_to_retry(msg: &str, had_previous_sample: bool) -> ActivityTransition<'_> {
    ActivityTransition::message(
        Activity::Retrying {
            had_previous_sample,
            started_at: Instant::now(),
        },
        msg,
    )
}
