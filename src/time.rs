use crate::{
    IS_TERMINAL,
    buffmt::{ByteCursor, digit_byte, two_digits},
    diagnostic::or_abort,
    numeric::{low_u8_from_u32, low_u32_from_u64},
    stop_input::StopInput,
};
use alloc::borrow::Cow;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use alloc::ffi::CString;
use core::{
    error::Error,
    fmt::{self, NumBuffer, Write as FmtWrite},
    hint::spin_loop,
    ops::Div as NumericDiv,
    result::Result as CoreResult,
    str::FromStr,
    time::Duration,
};
#[cfg(target_os = "windows")]
use core::{
    ffi::c_void,
    ptr::{NonNull, null},
};
use std::{
    io,
    panic::resume_unwind,
    process,
    sync::mpsc,
    thread,
    time::{Instant, SystemTime, SystemTimeError, UNIX_EPOCH},
};
mod address;
mod http_date;
mod native_http;
cfg_select! {
    target_os = "windows" => {
        mod timer_resolution;
        mod windows_input;
    }
    target_os = "linux" => {
        mod wayland_input;
    }
    target_os = "macos" => {
        mod macos_input;
    }
    _ => {
        compile_error!("SRG native input supports only Windows, Linux, and macOS.");
    }
}
const ADAPTIVE_POLL_INTERVAL: Duration = Duration::from_millis(10);
const NUM_SAMPLES: usize = 10;
const CALIBRATION_TIMEOUT: Duration = Duration::from_secs(5);
const CALIBRATION_TIMEOUT_MESSAGE: &str = "정밀 보정 제한 시간 안에 유효한 서버 Date tick을 관측하지 못했습니다. 전체 보정을 다시 시작합니다.";
const HTTP_SCHEME_PREFIX: &str = "http://";
const HTTPS_SCHEME_PREFIX: &str = "https://";
const KST_OFFSET_SECS: u64 = 9 * 3_600;
const SECS_PER_DAY_U64: u64 = 86_400;
const SECS_PER_DAY_I64: i64 = 86_400;
const SECS_PER_HOUR_U32: u32 = 3_600;
const SECS_PER_MINUTE_U32: u32 = 60;
const DAYS_PER_WEEK_I64: i64 = 7;
const DAY_OF_WEEK_KO: &[[u8; 3]] = "일월화수목금토".as_bytes().as_chunks::<3>().0;
const FINAL_COUNTDOWN_FREEZE_WINDOW: Duration = Duration::from_millis(200);
const FINAL_COUNTDOWN_WINDOW: Duration = Duration::from_secs(10);
const MAX_FINAL_SAMPLE_RTT: Duration = Duration::from_secs(2);
const MIN_TRANSFER_TIME: Duration = Duration::from_micros(1);
type ActivityTransition<'message> = (Activity, Option<&'message str>);
type BoxError = Box<dyn Error + Send + Sync>;
type Result<T> = CoreResult<T, TimeError>;
type SampleWorkerResponse = (u64, Result<TimeSample>);
#[derive(Clone, Copy)]
pub(super) enum TriggerAction {
    F5Press,
    LeftClick,
}
enum NativeInputSendStatus {
    FailedBeforeSend,
    PartialOrUnknown,
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    Sent,
}
#[derive(Clone, Copy, Debug)]
enum TimeErrorKind {
    HeaderNotFound,
    Io,
    NativeHttp,
    Parse,
    Time,
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
#[derive(Clone, Copy)]
enum UrlScheme {
    Http,
    Https,
}
pub(super) struct ParsedServer {
    #[cfg(target_os = "windows")]
    host_wide: Vec<u16>,
    #[cfg(target_os = "windows")]
    port: u16,
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    request_target: CString,
    scheme: UrlScheme,
}
#[derive(Clone, Copy)]
pub(super) struct TargetTimeOfDay {
    seconds_after_midnight: u32,
}
pub(super) struct ServerTimeSession {
    pub host: ParsedServer,
    pub scheduled_trigger: Option<(TargetTimeOfDay, TriggerAction)>,
    pub stop_after: Option<Duration>,
}
struct CivilDate {
    day: u32,
    month: u32,
    year: i32,
}
#[derive(Clone, Copy)]
struct TimeSample {
    response_received_inst: Instant,
    rtt: Duration,
    server_time: SystemTime,
}
struct BaselineRttState {
    attempts: usize,
    had_previous_sample: bool,
    next_sample_at: Instant,
    pending_generation: Option<u64>,
    previous_sample: Option<TimeSample>,
    rtt_nanos: [u128; NUM_SAMPLES],
    started: bool,
}
struct CalibrationState {
    baseline_rtt: Duration,
    pending_generation: Option<u64>,
    previous_sample: TimeSample,
    started_at: Instant,
}
#[derive(Clone, Copy)]
struct FinalCountdownState {
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    action: TriggerAction,
    http_guard: Option<HttpActionGuard>,
    last_sample_error_message_at: Option<Instant>,
    live_rtt: Duration,
    next_sample_at: Instant,
    pending_sample_generation: Option<u64>,
    sample_interval: Duration,
    server_time: ServerTime,
    target_time: SystemTime,
}
#[derive(Clone, Copy)]
enum ScheduledTarget {
    Confirmed(SystemTime, Option<HttpActionGuard>),
    Pending(TargetTimeOfDay),
}
#[derive(Clone, Copy)]
struct ScheduledTrigger {
    action: TriggerAction,
    target: ScheduledTarget,
}
#[derive(Clone, Copy)]
struct HttpActionGuard {
    last_valid_sample_at: Option<Instant>,
    reference_server_deadline: Instant,
}
#[derive(Clone, Copy)]
struct ServerTime {
    anchor_instant: Instant,
    anchor_time: SystemTime,
    baseline_rtt: Duration,
}
enum Activity {
    CalibrateOnTick(CalibrationState),
    FinalCountdown(FinalCountdownState),
    MeasureBaselineRtt(Box<BaselineRttState>),
    Predicting(ServerTime),
    Retrying {
        had_previous_sample: bool,
        started_at: Instant,
    },
}
enum ObservationStop<'input> {
    Deadline(Instant),
    Input(&'input mut StopInput),
}
struct SampleWorker {
    command_sender: mpsc::SyncSender<u64>,
    generation: u64,
    response_receiver: mpsc::Receiver<SampleWorkerResponse>,
    scheme: UrlScheme,
    worker: Option<thread::JoinHandle<()>>,
}
struct AppState<'worker> {
    #[cfg(target_os = "windows")]
    high_res_timer_attempted: bool,
    #[cfg(target_os = "windows")]
    high_res_timer_guard: Option<HighResTimerGuard>,
    last_full_sync_at: Instant,
    sample_worker: &'worker mut SampleWorker,
    scheduled_trigger: Option<ScheduledTrigger>,
}
impl FinalCountdownState {
    fn accept_sample(
        &mut self,
        sample: TimeSample,
        effective_rtt: Duration,
        server_time: ServerTime,
    ) -> Result<()> {
        (sample.rtt <= MAX_FINAL_SAMPLE_RTT).ok_or("최종 샘플 RTT가 허용 범위를 벗어났습니다.")?;
        (effective_rtt <= MAX_FINAL_SAMPLE_RTT)
            .ok_or("액션에 사용하는 RTT가 허용 범위를 벗어났습니다.")?;
        if let Some(guard) = self.http_guard.as_mut() {
            let expected = server_time.current_server_time_at(sample.response_received_inst)?;
            let offset = match sample.server_time.duration_since(expected) {
                Ok(offset) => offset,
                Err(source) => source.duration(),
            };
            (offset.saturating_sub(sample.rtt) <= Duration::from_secs(1))
                .ok_or("HTTP 최종 샘플 서버 시각이 기준 시각과 일치하지 않습니다.")?;
            guard.last_valid_sample_at = Some(sample.response_received_inst);
        }
        Ok(())
    }
}
impl ServerTime {
    fn current_server_time_at(&self, now: Instant) -> Result<SystemTime> {
        let elapsed = now
            .checked_duration_since(self.anchor_instant)
            .ok_or("현재 시각이 서버 시간 기준 시각보다 앞섭니다.")?;
        Ok(self
            .anchor_time
            .checked_add(elapsed)
            .ok_or("현재 서버 시간 계산 범위를 초과했습니다.")?)
    }
}
impl Activity {
    fn measure_baseline(now: Instant, had_previous_sample: bool) -> Self {
        Self::MeasureBaselineRtt(Box::new(BaselineRttState {
            attempts: 0,
            had_previous_sample,
            next_sample_at: now,
            pending_generation: None,
            previous_sample: None,
            rtt_nanos: [0; NUM_SAMPLES],
            started: false,
        }))
    }
    const fn server_time(&self) -> Option<ServerTime> {
        match *self {
            Self::FinalCountdown(state) => Some(state.server_time),
            Self::Predicting(server_time) => Some(server_time),
            Self::CalibrateOnTick(_)
            | Self::MeasureBaselineRtt(_)
            | Self::Retrying {
                had_previous_sample: _,
                started_at: _,
            } => None,
        }
    }
}
impl TargetTimeOfDay {
    fn resolve(
        self,
        server_time: ServerTime,
        now: Instant,
        msg_buf: &mut String,
    ) -> Result<SystemTime> {
        let target_second = self.seconds_after_midnight;
        let kst_epoch_secs = server_time
            .current_server_time_at(now)?
            .duration_since(UNIX_EPOCH)?
            .as_secs()
            .checked_add(KST_OFFSET_SECS)
            .ok_or("KST 현재 시각 계산 중 overflow가 발생했습니다.")?;
        let current_kst_second = u32::try_from(kst_epoch_secs.rem_euclid(SECS_PER_DAY_U64))
            .map_err(|source| TimeError::parse_with_source("KST 초 변환 실패", source))?;
        let current_kst_day = kst_epoch_secs.div_euclid(SECS_PER_DAY_U64);
        let target_day = if target_second < current_kst_second {
            current_kst_day
                .checked_add(1)
                .ok_or("다음날 목표 날짜 계산 중 overflow가 발생했습니다.")?
        } else {
            current_kst_day
        };
        let target_utc_epoch_secs = target_day
            .checked_mul(SECS_PER_DAY_U64)
            .and_then(|day_start| day_start.checked_add(u64::from(target_second)))
            .ok_or("목표 시각 계산 중 overflow가 발생했습니다.")?
            .checked_sub(KST_OFFSET_SECS)
            .ok_or("목표 UTC 시각 계산 중 underflow가 발생했습니다.")?;
        let target_system_time = UNIX_EPOCH
            .checked_add(Duration::from_secs(target_utc_epoch_secs))
            .ok_or("목표 절대 시각 계산 중 범위 오류가 발생했습니다.")?;
        let day_index = i32::try_from(target_day)
            .map_err(|source| TimeError::parse_with_source("목표 날짜 변환 실패", source))?;
        let CivilDate { day, month, year } = http_date::civil_from_days(day_index);
        let (hour, minute, second) = hms_from_second_of_day(target_second);
        append_fmt(
            msg_buf,
            format_args!(
                "\n목표 시각 확정(KST): {year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}"
            ),
        );
        Ok(target_system_time)
    }
}
impl FromStr for TargetTimeOfDay {
    type Err = &'static str;
    fn from_str(raw_input: &str) -> CoreResult<Self, Self::Err> {
        const INVALID_TIME_INPUT_ERR: &str =
            "잘못된 형식, 숫자 또는 시간 범위입니다 (HH:MM:SS, 0-23:0-59:0-59).";
        let (hour, minute, second) = parse_hms(raw_input).ok_or(INVALID_TIME_INPUT_ERR)?;
        (hour <= 23 && minute <= 59 && second <= 59).ok_or(INVALID_TIME_INPUT_ERR)?;
        Ok(Self {
            seconds_after_midnight: hour
                .strict_mul(SECS_PER_HOUR_U32)
                .strict_add(minute.strict_mul(SECS_PER_MINUTE_U32))
                .strict_add(second),
        })
    }
}
impl TimeError {
    fn new(kind: TimeErrorKind, detail: impl Into<Cow<'static, str>>) -> Self {
        Self {
            detail: detail.into(),
            kind,
            source: None,
        }
    }
    fn new_with_source(
        kind: TimeErrorKind,
        detail: impl Into<Cow<'static, str>>,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            detail: detail.into(),
            kind,
            source: Some(Box::new(source)),
        }
    }
    fn parse(detail: &'static str) -> Self {
        Self::new(TimeErrorKind::Parse, detail)
    }
    fn parse_with_source(
        detail: impl Into<Cow<'static, str>>,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self::new_with_source(TimeErrorKind::Parse, detail, source)
    }
}
impl From<&'static str> for TimeError {
    fn from(detail: &'static str) -> Self {
        Self::parse(detail)
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
        self.source
            .as_deref()
            .map(|source| -> &(dyn Error + 'static) { source })
    }
}
impl SampleWorker {
    fn ensure_fetch(&mut self, pending_generation: &mut Option<u64>) {
        if pending_generation.is_some() {
            return;
        }
        let generation = self.generation.strict_add(1);
        match self.command_sender.try_send(generation) {
            Ok(()) => {
                self.generation = generation;
                *pending_generation = Some(generation);
            }
            Err(mpsc::TrySendError::Full(_)) => {}
            Err(mpsc::TrySendError::Disconnected(_)) => self.resume_panic(),
        }
    }
    fn poll_fetch(&mut self, pending_generation: &mut Option<u64>) -> Option<Result<TimeSample>> {
        let expected_generation = *pending_generation;
        loop {
            match self.response_receiver.try_recv() {
                Ok((generation, result)) if Some(generation) == expected_generation => {
                    *pending_generation = None;
                    return Some(result);
                }
                Ok(_) => {}
                Err(mpsc::TryRecvError::Empty) => return None,
                Err(mpsc::TryRecvError::Disconnected) => self.resume_panic(),
            }
        }
    }
    fn resume_panic(&mut self) -> ! {
        match or_abort(self.worker.take()).join() {
            Ok(()) => process::abort(),
            Err(payload) => resume_unwind(payload),
        }
    }
}
impl ObservationStop<'_> {
    fn poll(&mut self, now: Instant) -> io::Result<bool> {
        match *self {
            Self::Deadline(stop_at) => Ok(now >= stop_at),
            Self::Input(ref mut input) => input.poll(),
        }
    }
    fn remaining(&self, now: Instant) -> Option<Duration> {
        match *self {
            Self::Deadline(stop_at) => Some(stop_at.saturating_duration_since(now)),
            Self::Input(_) => None,
        }
    }
}
impl ServerTimeSession {
    pub(super) fn run_loop(self, out: &mut dyn io::Write, err: &mut dyn io::Write) -> Result<()> {
        let now = Instant::now();
        let host = self.host;
        let scheme = host.scheme;
        let (command_sender, command_receiver) = mpsc::sync_channel(1);
        let (response_sender, response_receiver) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name(String::from("srg-sample-worker"))
            .spawn(move || {
                let mut native_http = native_http::Client::default();
                let context = match host.scheme {
                    UrlScheme::Http => "HTTP",
                    UrlScheme::Https => "HTTPS",
                };
                while let Ok(generation) = command_receiver.recv() {
                    if response_sender
                        .send((generation, native_http.fetch_head(&host, context)))
                        .is_err()
                    {
                        return;
                    }
                }
            })?;
        let mut sample_worker = SampleWorker {
            command_sender,
            generation: 0,
            response_receiver,
            scheme,
            worker: Some(worker),
        };
        let scheduled_trigger =
            self.scheduled_trigger
                .map(|(target_time, action)| ScheduledTrigger {
                    action,
                    target: ScheduledTarget::Pending(target_time),
                });
        if scheduled_trigger.is_some() && matches!(sample_worker.scheme, UrlScheme::Http) {
            err.write_all(
                concat!(
                    "[경고] HTTP 서버 시간은 네트워크에서 위조될 수 있습니다.\n",
                    "이상 보정, 과도한 지연 또는 최신 샘플 부재가 감지되면 예약 액션을 자동 취소합니다.\n"
                )
                .as_bytes(),
            )?;
        }
        #[cfg(target_os = "macos")]
        if scheduled_trigger.is_some() && !macos_input::post_event_access_granted(true) {
            writeln!(err, "[경고] macOS 입력 제어 권한이 허용되지 않았습니다.")?;
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
            #[cfg(target_os = "windows")]
            high_res_timer_attempted: false,
            #[cfg(target_os = "windows")]
            high_res_timer_guard: None,
            last_full_sync_at: now,
            sample_worker: &mut sample_worker,
            scheduled_trigger,
        };
        app_state.run_loop(self.stop_after, out, err)?;
        writeln!(out, "\n서버 시간 확인을 종료합니다.").map_err(Into::into)
    }
}
impl AppState<'_> {
    fn accept_calibration_tick(
        &mut self,
        calibration: &CalibrationState,
        current_sample: TimeSample,
    ) -> Result<Option<ServerTime>> {
        let prev_dur = calibration
            .previous_sample
            .server_time
            .duration_since(UNIX_EPOCH)?;
        let current_dur = current_sample.server_time.duration_since(UNIX_EPOCH)?;
        if current_dur.as_secs().checked_sub(prev_dur.as_secs()) != Some(1) {
            return Ok(None);
        }
        let anchor_instant = current_sample
            .response_received_inst
            .checked_sub(one_way_delay(None, current_sample.rtt))
            .ok_or("정밀 보정 기준 단조 시각 계산 실패")?;
        self.last_full_sync_at = current_sample.response_received_inst;
        Ok(Some(ServerTime {
            anchor_instant,
            anchor_time: current_sample.server_time,
            baseline_rtt: calibration.baseline_rtt,
        }))
    }
    fn cancel_final_countdown(
        server_time: ServerTime,
        msg_buf: &mut String,
        reason: impl fmt::Display,
    ) -> ActivityTransition<'_> {
        append_error_detail(msg_buf, "[경고] 예약 액션 취소: ", reason);
        (Activity::Predicting(server_time), Some(msg_buf))
    }
    fn fetch_final_countdown_sample(
        &mut self,
        countdown: &mut FinalCountdownState,
        interval: Duration,
        now: Instant,
    ) -> Result<Option<TimeSample>> {
        let next_sample_at = now
            .checked_add(interval)
            .ok_or("카운트다운 다음 샘플 시각 계산 실패")?;
        if countdown.sample_interval != interval {
            countdown.sample_interval = interval;
            if countdown.pending_sample_generation.is_none() {
                countdown.next_sample_at = next_sample_at;
            }
        }
        let latest_sample = self
            .sample_worker
            .poll_fetch(&mut countdown.pending_sample_generation)
            .transpose()?;
        if latest_sample.is_some() {
            countdown.next_sample_at = next_sample_at;
        }
        if countdown.pending_sample_generation.is_none() && now >= countdown.next_sample_at {
            self.sample_worker
                .ensure_fetch(&mut countdown.pending_sample_generation);
        }
        Ok(latest_sample)
    }
    fn handle_calibrate_on_tick<'message>(
        &mut self,
        mut calibration: CalibrationState,
        msg_buf: &'message mut String,
    ) -> ActivityTransition<'message> {
        if Instant::now().saturating_duration_since(calibration.started_at) >= CALIBRATION_TIMEOUT {
            return transition_to_retry(CALIBRATION_TIMEOUT_MESSAGE, true);
        }
        if let Some(sample_result) = self
            .sample_worker
            .poll_fetch(&mut calibration.pending_generation)
        {
            let completed_at = Instant::now();
            if completed_at.saturating_duration_since(calibration.started_at) >= CALIBRATION_TIMEOUT
            {
                return transition_to_retry(CALIBRATION_TIMEOUT_MESSAGE, true);
            }
            let current_sample = match sample_result {
                Ok(sample) => sample,
                Err(sample_err) => {
                    append_error_detail(msg_buf, "정밀 보정 샘플 수집 실패: ", sample_err);
                    return transition_to_retry(msg_buf, true);
                }
            };
            let accepted_server_time =
                match self.accept_calibration_tick(&calibration, current_sample) {
                    Ok(server_time) => server_time,
                    Err(calibration_err) => {
                        append_error_detail(msg_buf, "정밀 보정 시각 계산 실패: ", calibration_err);
                        return transition_to_retry(msg_buf, true);
                    }
                };
            if let Some(server_time) = accepted_server_time {
                msg_buf.push_str("[성공] 정밀 보정 완료!");
                if let Some(schedule) = self.scheduled_trigger.as_mut() {
                    let updated = match schedule.target {
                        ScheduledTarget::Pending(target_time) => {
                            let resolved =
                                match target_time.resolve(server_time, completed_at, msg_buf) {
                                    Ok(resolved) => resolved,
                                    Err(target_err) => {
                                        append_error_detail(
                                            msg_buf,
                                            "\n목표 시각 확정 실패: ",
                                            target_err,
                                        );
                                        return transition_to_retry(msg_buf, true);
                                    }
                                };
                            match self.sample_worker.scheme {
                                UrlScheme::Https => Ok(None),
                                UrlScheme::Http => strict_server_deadline(server_time, resolved)
                                    .map(|reference_server_deadline| {
                                        Some(HttpActionGuard {
                                            last_valid_sample_at: None,
                                            reference_server_deadline,
                                        })
                                    }),
                            }
                            .map(|guard| ScheduledTarget::Confirmed(resolved, guard))
                        }
                        ScheduledTarget::Confirmed(target_time, http_guard) => http_guard
                            .map_or_else(
                                || Ok(()),
                                |guard| {
                                    let deadline = strict_server_deadline(server_time, target_time)?;
                                    let shift = if deadline >= guard.reference_server_deadline {
                                        deadline.duration_since(guard.reference_server_deadline)
                                    } else {
                                        guard.reference_server_deadline.duration_since(deadline)
                                    };
                                    Ok((shift <= Duration::from_secs(2)).ok_or(
                                        "HTTP 서버 deadline 누적 이동량이 허용 범위를 초과했습니다.",
                                    )?)
                                },
                            )
                            .map(|()| ScheduledTarget::Confirmed(target_time, http_guard)),
                    };
                    match updated {
                        Ok(updated_target) => schedule.target = updated_target,
                        Err(reason) => {
                            self.scheduled_trigger = None;
                            append_error_detail(msg_buf, "\n[경고] HTTP 예약 액션 취소: ", reason);
                        }
                    }
                }
                return (Activity::Predicting(server_time), Some(msg_buf.as_str()));
            }
            calibration.previous_sample = current_sample;
        }
        self.sample_worker
            .ensure_fetch(&mut calibration.pending_generation);
        (Activity::CalibrateOnTick(calibration), None)
    }
    fn handle_final_countdown<'message>(
        &mut self,
        mut countdown: FinalCountdownState,
        msg_buf: &'message mut String,
        err: &mut dyn io::Write,
        #[cfg(target_os = "linux")] prepared_input: &mut wayland_input::PreparedInput,
    ) -> ActivityTransition<'message> {
        let now = Instant::now();
        let current_server_time = match countdown.server_time.current_server_time_at(now) {
            Ok(current) => current,
            Err(time_err) => {
                return Self::cancel_final_countdown(countdown.server_time, msg_buf, time_err);
            }
        };
        let sample_interval = final_countdown_sample_interval(
            countdown
                .target_time
                .duration_since(current_server_time)
                .ok(),
        );
        let (check_now, delay, sampled_rtt, sample_error_reported) = match self
            .fetch_final_countdown_sample(&mut countdown, sample_interval, now)
        {
            Ok(Some(sample)) => {
                let calibrated_server_time = ServerTime {
                    anchor_time: countdown.server_time.anchor_time,
                    anchor_instant: countdown.server_time.anchor_instant,
                    baseline_rtt: blend_rtt::<3>(countdown.server_time.baseline_rtt, sample.rtt),
                };
                let sample_now = Instant::now();
                let live_rtt = blend_rtt::<7>(countdown.live_rtt, sample.rtt);
                let effective_rtt = live_rtt.max(sample.rtt);
                if let Err(policy_err) =
                    countdown.accept_sample(sample, effective_rtt, calibrated_server_time)
                {
                    return Self::cancel_final_countdown(
                        calibrated_server_time,
                        msg_buf,
                        policy_err,
                    );
                }
                countdown.live_rtt = live_rtt;
                countdown.server_time = calibrated_server_time;
                (
                    sample_now,
                    one_way_delay(countdown.http_guard, effective_rtt),
                    Some(sample.rtt),
                    false,
                )
            }
            Err(fetch_err) => {
                if countdown.http_guard.is_some() {
                    return Self::cancel_final_countdown(countdown.server_time, msg_buf, fetch_err);
                }
                let reported = if countdown.last_sample_error_message_at.is_some_and(|last| {
                    now.saturating_duration_since(last) < Duration::from_secs(1)
                }) {
                    false
                } else {
                    countdown.last_sample_error_message_at = Some(now);
                    append_error_detail(msg_buf, "카운트다운 샘플 획득 실패: ", fetch_err);
                    true
                };
                (
                    now,
                    one_way_delay(countdown.http_guard, countdown.live_rtt),
                    None,
                    reported,
                )
            }
            Ok(None) => (
                now,
                one_way_delay(countdown.http_guard, countdown.live_rtt),
                None,
                false,
            ),
        };
        let server_time = countdown.server_time;
        let trigger_instant =
            match trigger_instant_for_target(server_time, countdown.target_time, delay) {
                Ok(trigger) => trigger,
                Err(time_err) => {
                    return Self::cancel_final_countdown(server_time, msg_buf, time_err);
                }
            };
        if trigger_instant.saturating_duration_since(check_now) > FINAL_COUNTDOWN_FREEZE_WINDOW {
            return (
                Activity::FinalCountdown(countdown),
                sample_error_reported.then_some(msg_buf.as_str()),
            );
        }
        #[cfg(target_os = "windows")]
        let mut high_res_wait_error = None;
        let wait_now = Instant::now();
        if trigger_instant > wait_now {
            if let Some(sleep_until) = trigger_instant.checked_sub(Duration::from_millis(2))
                && sleep_until > wait_now
            {
                let sleep_duration = sleep_until.duration_since(wait_now);
                cfg_select! {
                    target_os = "windows" => {
                        if let Some(guard) = self.high_res_timer_guard.as_ref() {
                            high_res_wait_error = guard.sleep(sleep_duration).err();
                        } else {
                            thread::sleep(sleep_duration);
                        }
                    }
                    _ => {
                        thread::sleep(sleep_duration);
                    }
                }
            }
            while Instant::now() < trigger_instant {
                spin_loop();
            }
        }
        let trigger_now = Instant::now();
        let trigger_server_time = match server_time.current_server_time_at(trigger_now) {
            Ok(current) => current,
            Err(time_err) => return Self::cancel_final_countdown(server_time, msg_buf, time_err),
        };
        let timing = match countdown.target_time.duration_since(trigger_server_time) {
            Ok(remaining) => Ok(remaining),
            Err(late) if late.duration() <= Duration::from_secs(1) => Err(late.duration()),
            Err(_) => {
                return Self::cancel_final_countdown(
                    server_time,
                    msg_buf,
                    "목표 시각을 1초 넘게 지나 예약 액션을 전송하지 않습니다.",
                );
            }
        };
        if let Some(guard) = countdown.http_guard {
            let freshness = guard
                .last_valid_sample_at
                .ok_or("HTTP 최신 유효 샘플이 없습니다.")
                .and_then(|last_sample_at| {
                    let age = trigger_now
                        .checked_duration_since(last_sample_at)
                        .ok_or("HTTP 샘플 단조 시각 순서가 유효하지 않습니다.")?;
                    (age <= Duration::from_secs(2)).ok_or("HTTP 최신 샘플이 만료되었습니다.")
                });
            if let Err(time_err) = freshness {
                return Self::cancel_final_countdown(server_time, msg_buf, time_err);
            }
        }
        let send_status = cfg_select! {
            target_os = "linux" => prepared_input.send(err),
            any(target_os = "windows", target_os = "macos") => countdown.action.send(err),
            _ => compile_error!("SRG native input supports only Windows, Linux, and macOS."),
        };
        msg_buf.push_str(match send_status {
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            NativeInputSendStatus::Sent => "\n>>> 액션 실행! ",
            NativeInputSendStatus::FailedBeforeSend => "\n>>> 액션 실행 실패! ",
            NativeInputSendStatus::PartialOrUnknown => "\n>>> 액션 전송 상태 불확실! ",
        });
        msg_buf.push_str(if sampled_rtt.is_none() {
            "(예측값 기준, "
        } else {
            "("
        });
        let final_parenthesis = match timing {
            Ok(remaining) => {
                append_fmt(
                    msg_buf,
                    format_args!(
                        "목표 도달까지 {:.1}ms 남음) (지연 예측: {:.1}ms",
                        duration_millis_f64(remaining),
                        duration_millis_f64(delay),
                    ),
                );
                ")"
            }
            Err(late_by) => {
                append_fmt(
                    msg_buf,
                    format_args!(
                        "목표 초가 이미 시작되어 즉시 액션을 실행했습니다. ({:.1}ms 지연, 지연 예측: {:.1}ms",
                        duration_millis_f64(late_by),
                        duration_millis_f64(delay),
                    ),
                );
                "))"
            }
        };
        if let Some(rtt) = sampled_rtt {
            append_fmt(
                msg_buf,
                format_args!(", 실측 RTT: {:.1}ms", duration_millis_f64(rtt)),
            );
        }
        msg_buf.push_str(final_parenthesis);
        #[cfg(target_os = "windows")]
        if let Some(wait_error) = high_res_wait_error {
            append_fmt(
                msg_buf,
                format_args!(
                    "\n[경고] Windows 고해상도 대기 타이머 실패로 표준 대기를 사용했습니다: {wait_error}"
                ),
            );
        }
        (Activity::Predicting(server_time), Some(msg_buf))
    }
    fn handle_measure_baseline_rtt<'message>(
        &mut self,
        mut baseline: Box<BaselineRttState>,
        msg_buf: &'message mut String,
        now: Instant,
    ) -> ActivityTransition<'message> {
        if !baseline.started {
            baseline.started = true;
            if !baseline.had_previous_sample {
                return (
                    Activity::MeasureBaselineRtt(baseline),
                    Some("1단계: RTT 기준값 측정을 시작합니다…"),
                );
            }
        }
        let had_pending_request = baseline.pending_generation.is_some();
        let Some(result) = self
            .sample_worker
            .poll_fetch(&mut baseline.pending_generation)
        else {
            if !had_pending_request && now >= baseline.next_sample_at {
                self.sample_worker
                    .ensure_fetch(&mut baseline.pending_generation);
            }
            return (Activity::MeasureBaselineRtt(baseline), None);
        };
        let attempt_index = baseline.attempts;
        let next_sample_base = match result {
            Ok(sample) => {
                let Some(slot) = baseline.rtt_nanos.get_mut(attempt_index) else {
                    process::abort();
                };
                *slot = sample.rtt.as_nanos();
                baseline.previous_sample = Some(sample);
                baseline.had_previous_sample = true;
                sample.response_received_inst
            }
            Err(err) => {
                append_error_detail(msg_buf, "RTT 샘플 수집 실패: ", err);
                return transition_to_retry(msg_buf, baseline.had_previous_sample);
            }
        };
        baseline.attempts = attempt_index.strict_add(1);
        let Some(next_sample_at) = next_sample_base.checked_add(ADAPTIVE_POLL_INTERVAL) else {
            return transition_to_retry(
                "다음 RTT 샘플 시각 계산 실패.",
                baseline.had_previous_sample,
            );
        };
        baseline.next_sample_at = next_sample_at;
        if baseline.attempts < NUM_SAMPLES {
            return (Activity::MeasureBaselineRtt(baseline), None);
        }
        let previous_sample = or_abort(baseline.previous_sample);
        baseline.rtt_nanos.sort_unstable();
        let trim = NUM_SAMPLES.div_euclid(5);
        let trimmed_sample_count = NUM_SAMPLES.strict_sub(trim.strict_mul(2));
        let sum_nanos = baseline
            .rtt_nanos
            .iter()
            .skip(trim)
            .take(trimmed_sample_count)
            .fold(0_u128, |sum, &sample_nanos| sum.strict_add(sample_nanos));
        let baseline_rtt =
            Duration::from_nanos_u128(sum_nanos.div_euclid(trimmed_sample_count as u128));
        append_fmt(
            msg_buf,
            format_args!(
                "[완료] RTT 기준값: {}ms. 2단계: 정밀 보정을 시작합니다.",
                baseline_rtt.as_millis()
            ),
        );
        (
            Activity::CalibrateOnTick(CalibrationState {
                baseline_rtt,
                pending_generation: None,
                previous_sample,
                started_at: Instant::now(),
            }),
            Some(msg_buf),
        )
    }
    fn handle_predicting(
        &mut self,
        server_time: ServerTime,
        scheduled_trigger: Option<ScheduledTrigger>,
        now: Instant,
    ) -> Result<ActivityTransition<'static>> {
        let estimated_server_time = server_time.current_server_time_at(now)?;
        let protect_target = if let Some(schedule) = scheduled_trigger
            && let ScheduledTarget::Confirmed(target_time, http_guard) = schedule.target
        {
            let remaining = target_time.duration_since(estimated_server_time).ok();
            match remaining {
                Some(duration) if duration > FINAL_COUNTDOWN_WINDOW => {
                    duration <= Duration::from_mins(1)
                }
                _ => {
                    self.scheduled_trigger = None;
                    return Ok((
                        Activity::FinalCountdown(FinalCountdownState {
                            #[cfg(any(target_os = "windows", target_os = "macos"))]
                            action: schedule.action,
                            http_guard,
                            last_sample_error_message_at: None,
                            live_rtt: server_time.baseline_rtt,
                            next_sample_at: now,
                            pending_sample_generation: None,
                            sample_interval: final_countdown_sample_interval(remaining),
                            server_time,
                            target_time,
                        }),
                        Some("최종 카운트다운 시작!"),
                    ));
                }
            }
        } else {
            false
        };
        if now.saturating_duration_since(self.last_full_sync_at) >= Duration::from_mins(5)
            && !protect_target
        {
            return Ok((
                Activity::measure_baseline(now, true),
                Some("서버 시간 보정 주기 도래, 재보정 시작."),
            ));
        }
        Ok((Activity::Predicting(server_time), None))
    }
    fn run_loop(
        &mut self,
        stop_after: Option<Duration>,
        out: &mut dyn io::Write,
        err: &mut dyn io::Write,
    ) -> Result<()> {
        #[cfg(target_os = "linux")]
        let trigger_action = self
            .scheduled_trigger
            .as_ref()
            .map(|schedule| schedule.action);
        #[cfg(target_os = "linux")]
        if trigger_action.is_some() {
            writeln!(err, "[안내] Wayland 입력 권한 승인을 기다립니다.")?;
        }
        let mut stop_input;
        let mut stop = if let Some(duration) = stop_after {
            ObservationStop::Deadline(
                Instant::now()
                    .checked_add(duration)
                    .ok_or("관찰 종료 시각 계산 실패")?,
            )
        } else {
            stop_input = StopInput::try_from(Some(8))
                .map_err(|source| TimeError::parse_with_source("종료 입력 준비 실패", source))?;
            ObservationStop::Input(&mut stop_input)
        };
        #[cfg(target_os = "linux")]
        let mut prepared_input = wayland_input::PreparedInput::EMPTY;
        #[cfg(target_os = "linux")]
        {
            let mut stop_error = None;
            let stop_requested =
                prepared_input.prepare(trigger_action, err, || match stop.poll(Instant::now()) {
                    Ok(requested) => requested,
                    Err(source) => {
                        stop_error = Some(source);
                        true
                    }
                });
            if let Some(source) = stop_error {
                return Err(TimeError::parse_with_source("종료 입력 실패", source));
            }
            if stop_requested {
                return Ok(());
            }
        }
        let mut message_buffer = String::new();
        message_buffer
            .try_reserve_exact(256)
            .map_err(|source| TimeError::parse_with_source("buffer 메모리 확보 실패", source))?;
        let mut last_display_update = Instant::now();
        let mut activity = Activity::measure_baseline(last_display_update, false);
        let mut line_buf = [0_u8; 80];
        let display_interval = if *IS_TERMINAL {
            Duration::from_millis(16)
        } else {
            Duration::from_secs(1)
        };
        loop {
            let pre_wait_now = Instant::now();
            if stop
                .poll(pre_wait_now)
                .map_err(|source| TimeError::parse_with_source("종료 입력 실패", source))?
            {
                break;
            }
            let activity_poll_timeout = match activity {
                Activity::CalibrateOnTick(_)
                | Activity::FinalCountdown(_)
                | Activity::MeasureBaselineRtt(_) => ADAPTIVE_POLL_INTERVAL,
                Activity::Predicting(_)
                | Activity::Retrying {
                    had_previous_sample: _,
                    started_at: _,
                } => Duration::from_millis(45),
            };
            let mut poll_timeout = if Self::should_update_display(&activity, pre_wait_now) {
                let remaining_display = display_interval
                    .saturating_sub(pre_wait_now.saturating_duration_since(last_display_update));
                activity_poll_timeout.min(remaining_display)
            } else {
                activity_poll_timeout
            };
            if let Some(remaining) = stop.remaining(pre_wait_now) {
                poll_timeout = poll_timeout.min(remaining);
            }
            thread::sleep(poll_timeout);
            #[cfg(target_os = "linux")]
            prepared_input.maintain(err);
            let now = Instant::now();
            let scheduled_trigger = self.scheduled_trigger;
            let timing_sensitive_activity = if matches!(activity, Activity::FinalCountdown(_)) {
                true
            } else if let (
                &Activity::Predicting(server_time),
                Some(ScheduledTrigger {
                    action: _,
                    target: ScheduledTarget::Confirmed(target_time, _),
                }),
            ) = (&activity, scheduled_trigger)
            {
                target_time
                    .duration_since(server_time.current_server_time_at(now)?)
                    .ok()
                    .is_none_or(|remaining| remaining <= FINAL_COUNTDOWN_WINDOW)
            } else {
                false
            };
            if !timing_sensitive_activity {
                Self::run_loop_write_display_if_due(
                    &activity,
                    now,
                    &mut last_display_update,
                    display_interval,
                    out,
                    line_buf.as_mut_slice(),
                )?;
            }
            message_buffer.clear();
            let (next_activity, transition_message) = match activity {
                Activity::MeasureBaselineRtt(baseline) => {
                    self.handle_measure_baseline_rtt(baseline, &mut message_buffer, now)
                }
                Activity::CalibrateOnTick(calibration) => {
                    self.handle_calibrate_on_tick(calibration, &mut message_buffer)
                }
                Activity::Predicting(server_time) => {
                    self.handle_predicting(server_time, scheduled_trigger, now)?
                }
                Activity::FinalCountdown(countdown) => self.handle_final_countdown(
                    countdown,
                    &mut message_buffer,
                    err,
                    #[cfg(target_os = "linux")]
                    &mut prepared_input,
                ),
                Activity::Retrying {
                    had_previous_sample,
                    started_at,
                } => {
                    if now.saturating_duration_since(started_at) >= Duration::from_secs(10) {
                        (
                            Activity::measure_baseline(now, had_previous_sample),
                            Some("[재시도] 동기화를 다시 시작합니다."),
                        )
                    } else {
                        (
                            Activity::Retrying {
                                had_previous_sample,
                                started_at,
                            },
                            None,
                        )
                    }
                }
            };
            #[cfg(target_os = "windows")]
            self.sync_high_res_timer_state(&next_activity, err)?;
            let transition_had_message = if let Some(console_msg) = transition_message {
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
                    display_interval,
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
        display_interval: Duration,
        output: &mut dyn io::Write,
        buffer: &mut [u8],
    ) -> Result<()> {
        if now.saturating_duration_since(*last_update) >= display_interval
            && Self::should_update_display(display_activity, now)
            && let Some(server_time) = display_activity.server_time()
        {
            let mut cur = ByteCursor::new(buffer);
            cur.write_bytes("\r서버 시간: ".as_bytes());
            let since_epoch = server_time
                .current_server_time_at(now)?
                .duration_since(UNIX_EPOCH)?;
            let total_seconds_kst = i64::try_from(since_epoch.as_secs())
                .unwrap_or_else(|_| process::abort())
                .strict_add(KST_OFFSET_SECS.cast_signed());
            let days_since_epoch = total_seconds_kst.div_euclid(SECS_PER_DAY_I64);
            let day_of_week_idx =
                usize::try_from(days_since_epoch.strict_add(4).rem_euclid(DAYS_PER_WEEK_I64))
                    .unwrap_or_else(|_| process::abort());
            let day_of_week_bytes = or_abort(DAY_OF_WEEK_KO.get(day_of_week_idx).copied());
            let (hour, minute, second) = hms_from_second_of_day(low_u32_from_u64(
                total_seconds_kst
                    .rem_euclid(SECS_PER_DAY_I64)
                    .cast_unsigned(),
            ));
            let CivilDate { day, month, year } = http_date::civil_from_days(
                i32::try_from(days_since_epoch).unwrap_or_else(|_| process::abort()),
            );
            let year_value = year.cast_unsigned();
            if year_value < 10_000 {
                let [h0, h1] = two_digits(low_u8_from_u32(year_value.div_euclid(100)));
                let [l0, l1] = two_digits(low_u8_from_u32(year_value.rem_euclid(100)));
                *cur.take_array::<4>() = [h0, h1, l0, l1];
            } else {
                let mut year_buf = NumBuffer::new();
                cur.write_bytes(year_value.format_into(&mut year_buf).as_bytes());
            }
            for val in [month, day] {
                cur.write_byte(b'-');
                write_2digits(&mut cur, val);
            }
            cur.write_byte(b'(');
            *cur.take_array::<3>() = day_of_week_bytes;
            cur.write_bytes(b") ");
            write_2digits(&mut cur, hour);
            for val in [minute, second] {
                cur.write_byte(b':');
                write_2digits(&mut cur, val);
            }
            cur.write_byte(b'.');
            let millis = since_epoch.subsec_millis();
            let [tens, ones] = two_digits(low_u8_from_u32(millis.rem_euclid(100)));
            *cur.take_array::<3>() = [
                digit_byte(low_u8_from_u32(millis.div_euclid(100))),
                tens,
                ones,
            ];
            cur.write_bytes(b" \r");
            let written_len = cur.written_len();
            output.write_all(buffer.split_at(written_len).0)?;
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
        let delay = one_way_delay(countdown.http_guard, countdown.live_rtt);
        trigger_instant_for_target(server_time, countdown.target_time, delay)
            .ok()
            .is_none_or(|trigger_instant| {
                trigger_instant.saturating_duration_since(now) > FINAL_COUNTDOWN_FREEZE_WINDOW
            })
    }
    #[cfg(target_os = "windows")]
    fn sync_high_res_timer_state(
        &mut self,
        next_activity: &Activity,
        err: &mut dyn io::Write,
    ) -> Result<()> {
        if !matches!(next_activity, Activity::FinalCountdown(_)) {
            self.high_res_timer_attempted = false;
            self.high_res_timer_guard = None;
            return Ok(());
        }
        if self.high_res_timer_attempted {
            return Ok(());
        }
        self.high_res_timer_attempted = true;
        // SAFETY: No security attributes or name are needed, and a successful handle is transferred to the guard.
        let timer_handle = unsafe {
            timer_resolution::create_waitable_timer_ex_w(null(), null(), 0x2, 0x0010_0000 | 0x2)
        };
        let Some(handle) = NonNull::new(timer_handle) else {
            writeln!(
                err,
                concat!(
                    "[경고] Windows 고해상도 대기 타이머 생성에 실패했습니다. ",
                    "카운트다운 정확도가 저하될 수 있습니다."
                )
            )?;
            return Ok(());
        };
        self.high_res_timer_guard = Some(HighResTimerGuard { handle });
        Ok(())
    }
}
fn one_way_delay(http_guard: Option<HttpActionGuard>, rtt: Duration) -> Duration {
    let delay = NumericDiv::div(rtt, 2_u32);
    if http_guard.is_some() {
        delay.min(Duration::from_millis(500))
    } else {
        delay
    }
}
const fn blend_rtt<const NEW_WEIGHT: u128>(old_value: Duration, new_value: Duration) -> Duration {
    let weighted_sum = old_value
        .as_nanos()
        .strict_mul(10_u128.strict_sub(NEW_WEIGHT))
        .strict_add(new_value.as_nanos().strict_mul(NEW_WEIGHT));
    Duration::from_nanos_u128(weighted_sum.div_euclid(10))
}
const fn hms_from_second_of_day(second_of_day: u32) -> (u32, u32, u32) {
    (
        second_of_day.div_euclid(SECS_PER_HOUR_U32),
        second_of_day
            .rem_euclid(SECS_PER_HOUR_U32)
            .div_euclid(SECS_PER_MINUTE_U32),
        second_of_day.rem_euclid(SECS_PER_MINUTE_U32),
    )
}
fn parse_hms(raw: &str) -> Option<(u32, u32, u32)> {
    let (hour_str, minute_second) = raw.split_once(':')?;
    let (minute_str, second_str) = minute_second.split_once(':')?;
    let parse_two = |part: &str| (part.len() == 2).then(|| parse_u32_digits(part)).flatten();
    Some((
        parse_two(hour_str)?,
        parse_two(minute_str)?,
        parse_two(second_str)?,
    ))
}
fn parse_u32_digits(raw: &str) -> Option<u32> {
    (!raw.starts_with('+')).then(|| raw.parse().ok()).flatten()
}
fn write_2digits(cur: &mut ByteCursor<'_>, value: u32) {
    *cur.take_array::<2>() = two_digits(low_u8_from_u32(value));
}
const fn duration_millis_f64(duration: Duration) -> f64 {
    duration.as_secs_f64().algebraic_mul(1_000.0)
}
fn append_error_detail(target: &mut String, prefix: &str, err: impl fmt::Display) {
    append_fmt(target, format_args!("{prefix}{err}"));
}
fn final_countdown_sample_interval(remaining: Option<Duration>) -> Duration {
    match remaining {
        Some(duration) if duration > Duration::from_secs(3) => Duration::from_millis(200),
        Some(duration) if duration > Duration::from_secs(1) => Duration::from_millis(50),
        _ => Duration::from_millis(25),
    }
}
fn strict_server_deadline(server_time: ServerTime, target_time: SystemTime) -> Result<Instant> {
    Ok(server_time
        .anchor_instant
        .checked_add(target_time.duration_since(server_time.anchor_time)?)
        .ok_or("서버 deadline 계산 범위 오류")?)
}
fn trigger_instant_for_target(
    server_time: ServerTime,
    target_time: SystemTime,
    delay: Duration,
) -> Result<Instant> {
    Ok(strict_server_deadline(server_time, target_time)?
        .checked_sub(delay)
        .ok_or("액션 실행 deadline 계산 범위 오류")?)
}
fn append_fmt(target: &mut String, args: fmt::Arguments<'_>) {
    assert!(
        FmtWrite::write_fmt(target, args).is_ok(),
        "String formatting must be infallible"
    );
}
fn transition_to_retry(msg: &str, had_previous_sample: bool) -> ActivityTransition<'_> {
    (
        Activity::Retrying {
            had_previous_sample,
            started_at: Instant::now(),
        },
        Some(msg),
    )
}
