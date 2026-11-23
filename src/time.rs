use crate::{
    IS_TERMINAL,
    buffmt::{ByteCursor, digit_byte, two_digits},
    diagnostic::{AppError, Result, or_abort},
    numeric::{low_u8_from_u32, low_u32_from_u64},
    stop_input::StopInput,
};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use alloc::ffi::CString;
#[cfg(target_os = "windows")]
use core::{
    ffi::c_void,
    ptr::{NonNull, null},
};
use core::{
    fmt::{self, NumBuffer, Write as FmtWrite},
    hint::spin_loop,
    ops::Div as NumericDiv,
    result::Result as CoreResult,
    str::FromStr,
    time::Duration,
};
use std::{
    io,
    panic::resume_unwind,
    process,
    sync::mpsc,
    thread,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
mod http_date;
mod native_http;
cfg_select! {
    target_os = "windows" => {
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
const CALIBRATION_TIMEOUT: Duration = Duration::from_secs(5);
const CALIBRATION_TIMEOUT_MSG: &str = "정밀 보정 제한 시간 안에 유효한 서버 Date tick을 관측하지 못했습니다. 전체 보정을 다시 시작합니다.";
const DAY_OF_WEEK_KO: &[[u8; 3]] = "일월화수목금토".as_bytes().as_chunks::<3>().0;
const FINAL_COUNTDOWN_FREEZE_WINDOW: Duration = Duration::from_millis(200);
const FINAL_COUNTDOWN_WINDOW: Duration = Duration::from_secs(10);
const HTTP_SCHEME_PREFIX: &str = "http://";
const HTTPS_SCHEME_PREFIX: &str = "https://";
const KST_OFFSET_SECS: u64 = 9 * 3_600;
const MAX_FINAL_SAMPLE_RTT: Duration = Duration::from_secs(2);
const NUM_SAMPLES: usize = 10;
const SECS_PER_DAY_U64: u64 = 86_400;
type ActivityTransition<'message> = (Activity, Option<&'message str>);
type SampleWorkerResponse = (u64, Result<TimeSample>);
enum Activity {
    CalibrateOnTick(CalibrationState),
    FinalCountdown(FinalCountdownState),
    MeasureBaselineRtt(Box<BaselineRttState>),
    Predicting(ServerTime),
    Retrying(bool, Instant),
}
struct AppState {
    display_interval: Duration,
    #[cfg(target_os = "windows")]
    high_res_timer_attempted: bool,
    #[cfg(target_os = "windows")]
    high_res_timer_guard: Option<HighResTimerGuard>,
    last_display: Instant,
    last_full_sync_at: Instant,
    line_buf: [u8; 80],
    sample_worker: SampleWorker,
    scheduled_trigger: Option<ScheduledTrigger>,
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
#[cfg(target_os = "windows")]
struct HighResTimerGuard {
    handle: NonNull<c_void>,
}
#[derive(Clone, Copy)]
struct HttpActionGuard {
    last_valid_sample_at: Option<Instant>,
    reference_server_deadline: Instant,
}
enum NativeInputSendStatus {
    FailedBeforeSend,
    PartialOrUnknown,
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    Sent,
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
struct SampleWorker {
    command_sender: mpsc::SyncSender<u64>,
    generation: u64,
    response_receiver: mpsc::Receiver<SampleWorkerResponse>,
    scheme: UrlScheme,
    worker: Option<thread::JoinHandle<()>>,
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
struct ServerTime {
    anchor_instant: Instant,
    anchor_time: SystemTime,
    baseline_rtt: Duration,
}
pub(super) struct ServerTimeSession {
    pub host: ParsedServer,
    pub scheduled_trigger: Option<(TargetTimeOfDay, TriggerAction)>,
    pub stop_after: Option<Duration>,
}
#[derive(Clone, Copy)]
pub(super) struct TargetTimeOfDay {
    seconds_after_midnight: u32,
}
#[derive(Clone, Copy)]
struct TimeSample {
    response_received_inst: Instant,
    rtt: Duration,
    server_time: SystemTime,
}
#[derive(Clone, Copy)]
pub(super) enum TriggerAction {
    F5Press,
    LeftClick,
}
#[derive(Clone, Copy)]
enum UrlScheme {
    Http,
    Https,
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
            Self::Predicting(st) => Some(st),
            Self::CalibrateOnTick(_) | Self::MeasureBaselineRtt(_) | Self::Retrying(_, _) => None,
        }
    }
}
impl AppState {
    fn cancel_final_countdown(
        st: ServerTime,
        msg_buf: &mut String,
        reason: impl fmt::Display,
    ) -> ActivityTransition<'_> {
        append_error_detail(msg_buf, "[경고] 예약 액션 취소: ", reason);
        (Activity::Predicting(st), Some(msg_buf))
    }
    fn handle_calibrate_on_tick<'message>(
        &mut self,
        mut cal: CalibrationState,
        msg_buf: &'message mut String,
    ) -> ActivityTransition<'message> {
        if Instant::now().saturating_duration_since(cal.started_at) >= CALIBRATION_TIMEOUT {
            return transition_to_retry(CALIBRATION_TIMEOUT_MSG, true);
        }
        if let Some(sample_res) = self.sample_worker.poll_fetch(&mut cal.pending_generation) {
            let done_at = Instant::now();
            if done_at.saturating_duration_since(cal.started_at) >= CALIBRATION_TIMEOUT {
                return transition_to_retry(CALIBRATION_TIMEOUT_MSG, true);
            }
            let tick = (|| -> Result<Option<ServerTime>> {
                let curr =
                    sample_res.map_err(|err| AppError::context("정밀 보정 샘플 수집 실패", err))?;
                let prev_s = cal.previous_sample.server_time.duration_since(UNIX_EPOCH)?;
                let curr_s = curr.server_time.duration_since(UNIX_EPOCH)?;
                if curr_s.as_secs().checked_sub(prev_s.as_secs()) != Some(1) {
                    cal.previous_sample = curr;
                    return Ok(None);
                }
                let anchor_instant = curr
                    .response_received_inst
                    .checked_sub(one_way_delay(None, curr.rtt))
                    .ok_or("정밀 보정 기준 단조 시각 계산 실패")?;
                self.last_full_sync_at = curr.response_received_inst;
                Ok(Some(ServerTime {
                    anchor_instant,
                    anchor_time: curr.server_time,
                    baseline_rtt: cal.baseline_rtt,
                }))
            })();
            match tick {
                Err(err) => {
                    append_error_detail(msg_buf, "정밀 보정 실패: ", err);
                    return transition_to_retry(msg_buf, true);
                }
                Ok(None) => {}
                Ok(Some(st)) => {
                    msg_buf.push_str("[성공] 정밀 보정 완료!");
                    if let Some(sched) = self.scheduled_trigger.as_mut() {
                        let updated: Result<ScheduledTarget> = match sched.target {
                            ScheduledTarget::Pending(target) => {
                                let res = match target.resolve(st, done_at, msg_buf) {
                                    Ok(val) => val,
                                    Err(err) => {
                                        append_error_detail(
                                            msg_buf,
                                            "\n목표 시각 확정 실패: ",
                                            err,
                                        );
                                        return transition_to_retry(msg_buf, true);
                                    }
                                };
                                match self.sample_worker.scheme {
                                    UrlScheme::Https => Ok(ScheduledTarget::Confirmed(res, None)),
                                    UrlScheme::Http => strict_server_deadline(st, res).map(|dl| {
                                        ScheduledTarget::Confirmed(
                                            res,
                                            Some(HttpActionGuard {
                                                last_valid_sample_at: None,
                                                reference_server_deadline: dl,
                                            }),
                                        )
                                    }),
                                }
                            }
                            ScheduledTarget::Confirmed(target, hg) => (|| {
                                if let Some(guard) = hg {
                                    let dl = strict_server_deadline(st, target)?;
                                    let shift = dl
                                        .checked_duration_since(guard.reference_server_deadline)
                                        .unwrap_or_else(|| {
                                            guard.reference_server_deadline.duration_since(dl)
                                        });
                                    (shift <= Duration::from_secs(2)).ok_or(
                                        "HTTP 서버 deadline 누적 이동량이 허용 범위를 초과했습니다.",
                                    )?;
                                }
                                Ok(ScheduledTarget::Confirmed(target, hg))
                            })(
                            ),
                        };
                        match updated {
                            Ok(target) => sched.target = target,
                            Err(reason) => {
                                self.scheduled_trigger = None;
                                append_error_detail(
                                    msg_buf,
                                    "\n[경고] HTTP 예약 액션 취소: ",
                                    reason,
                                );
                            }
                        }
                    }
                    return (Activity::Predicting(st), Some(msg_buf.as_str()));
                }
            }
        }
        self.sample_worker.ensure_fetch(&mut cal.pending_generation);
        (Activity::CalibrateOnTick(cal), None)
    }
    fn handle_final_countdown<'message>(
        &mut self,
        mut cd: FinalCountdownState,
        msg_buf: &'message mut String,
        err_out: &mut dyn io::Write,
        #[cfg(target_os = "linux")] prepared_input: &mut wayland_input::PreparedInput,
    ) -> ActivityTransition<'message> {
        let now = Instant::now();
        let step = (|| -> Result<(Instant, Duration, Option<Duration>, bool, Instant)> {
            let cur_st = cd.server_time.current_server_time_at(now)?;
            let interval =
                final_countdown_sample_interval(cd.target_time.duration_since(cur_st).ok());
            let next_at = now
                .checked_add(interval)
                .ok_or("카운트다운 다음 샘플 시각 계산 실패")?;
            if cd.sample_interval != interval {
                cd.sample_interval = interval;
                if cd.pending_sample_generation.is_none() {
                    cd.next_sample_at = next_at;
                }
            }
            let polled = self
                .sample_worker
                .poll_fetch(&mut cd.pending_sample_generation)
                .transpose();
            if matches!(polled, Ok(Some(_))) {
                cd.next_sample_at = next_at;
            }
            if cd.pending_sample_generation.is_none() && now >= cd.next_sample_at {
                self.sample_worker
                    .ensure_fetch(&mut cd.pending_sample_generation);
            }
            let (check_now, delay, sampled_rtt, reported) = match polled {
                Ok(Some(sample)) => {
                    let cal = ServerTime {
                        baseline_rtt: blend_rtt::<3>(cd.server_time.baseline_rtt, sample.rtt),
                        ..cd.server_time
                    };
                    let sample_now = Instant::now();
                    let live_rtt = blend_rtt::<7>(cd.live_rtt, sample.rtt);
                    let eff_rtt = live_rtt.max(sample.rtt);
                    (sample.rtt <= MAX_FINAL_SAMPLE_RTT && eff_rtt <= MAX_FINAL_SAMPLE_RTT)
                        .ok_or("최종 샘플 또는 액션 RTT가 허용 범위를 벗어났습니다.")?;
                    if let Some(guard) = cd.http_guard.as_mut() {
                        let exp = cal.current_server_time_at(sample.response_received_inst)?;
                        let off = sample
                            .server_time
                            .duration_since(exp)
                            .unwrap_or_else(|src| src.duration());
                        (off.saturating_sub(sample.rtt) <= Duration::from_secs(1))
                            .ok_or("HTTP 최종 샘플 서버 시각이 기준 시각과 일치하지 않습니다.")?;
                        guard.last_valid_sample_at = Some(sample.response_received_inst);
                    }
                    cd.live_rtt = live_rtt;
                    cd.server_time = cal;
                    (
                        sample_now,
                        one_way_delay(cd.http_guard, eff_rtt),
                        Some(sample.rtt),
                        false,
                    )
                }
                Err(fetch_err) => {
                    if cd.http_guard.is_some() {
                        return Err(fetch_err);
                    }
                    let rep = !cd.last_sample_error_message_at.is_some_and(|last| {
                        now.saturating_duration_since(last) < Duration::from_secs(1)
                    });
                    if rep {
                        cd.last_sample_error_message_at = Some(now);
                        append_error_detail(msg_buf, "카운트다운 샘플 획득 실패: ", fetch_err);
                    }
                    (now, one_way_delay(cd.http_guard, cd.live_rtt), None, rep)
                }
                Ok(None) => (now, one_way_delay(cd.http_guard, cd.live_rtt), None, false),
            };
            let trig = trigger_instant_for_target(cd.server_time, cd.target_time, delay)?;
            Ok((check_now, delay, sampled_rtt, reported, trig))
        })();
        let (check_now, delay, sampled_rtt, reported, trig_inst) = match step {
            Ok(vals) => vals,
            Err(err) => return Self::cancel_final_countdown(cd.server_time, msg_buf, err),
        };
        if trig_inst.saturating_duration_since(check_now) > FINAL_COUNTDOWN_FREEZE_WINDOW {
            return (
                Activity::FinalCountdown(cd),
                reported.then_some(msg_buf.as_str()),
            );
        }
        #[cfg(target_os = "windows")]
        let mut high_res_err = None;
        let wait_now = Instant::now();
        if trig_inst > wait_now {
            if let Some(until) = trig_inst.checked_sub(Duration::from_millis(2))
                && until > wait_now
            {
                let dur = until.duration_since(wait_now);
                cfg_select! {
                    target_os = "windows" => {
                        if let Some(guard) = self.high_res_timer_guard.as_ref() {
                            high_res_err = guard.sleep(dur).err();
                        } else {
                            thread::sleep(dur);
                        }
                    }
                    _ => {
                        thread::sleep(dur);
                    }
                }
            }
            while Instant::now() < trig_inst {
                spin_loop();
            }
        }
        let st = cd.server_time;
        let pre_send = (|| -> Result<CoreResult<Duration, Duration>> {
            let trig_now = Instant::now();
            let cur = st.current_server_time_at(trig_now)?;
            let timing = match cd.target_time.duration_since(cur) {
                Ok(rem) => Ok(rem),
                Err(late) if late.duration() <= Duration::from_secs(1) => Err(late.duration()),
                Err(_) => {
                    return Err("목표 시각을 1초 넘게 지나 예약 액션을 전송하지 않습니다.".into());
                }
            };
            if let Some(guard) = cd.http_guard {
                let last = guard
                    .last_valid_sample_at
                    .ok_or("HTTP 최신 유효 샘플이 없습니다.")?;
                let age = trig_now
                    .checked_duration_since(last)
                    .ok_or("HTTP 샘플 단조 시각 순서가 유효하지 않습니다.")?;
                (age <= Duration::from_secs(2)).ok_or("HTTP 최신 샘플이 만료되었습니다.")?;
            }
            Ok(timing)
        })();
        let timing = match pre_send {
            Ok(val) => val,
            Err(err) => return Self::cancel_final_countdown(st, msg_buf, err),
        };
        let status = cfg_select! {
            target_os = "linux" => prepared_input.send(err_out),
            any(target_os = "windows", target_os = "macos") => cd.action.send(err_out),
            _ => compile_error!("SRG native input supports only Windows, Linux, and macOS."),
        };
        let prefix = match status {
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            NativeInputSendStatus::Sent => "\n>>> 액션 실행! ",
            NativeInputSendStatus::FailedBeforeSend => "\n>>> 액션 실행 실패! ",
            NativeInputSendStatus::PartialOrUnknown => "\n>>> 액션 전송 상태 불확실! ",
        };
        let mode = if sampled_rtt.is_none() {
            "(예측값 기준, "
        } else {
            "("
        };
        let delay_ms = duration_millis_f64(delay);
        let (detail_pre, detail_ms, detail_mid, closing) = match timing {
            Ok(rem) => (
                "목표 도달까지 ",
                duration_millis_f64(rem),
                "ms 남음) (",
                ")",
            ),
            Err(late) => (
                "목표 초가 이미 시작되어 즉시 액션을 실행했습니다. (",
                duration_millis_f64(late),
                "ms 지연, ",
                "))",
            ),
        };
        append_fmt(
            msg_buf,
            format_args!(
                "{prefix}{mode}{detail_pre}{detail_ms:.1}{detail_mid}지연 예측: {delay_ms:.1}ms"
            ),
        );
        if let Some(rtt) = sampled_rtt {
            append_fmt(
                msg_buf,
                format_args!(", 실측 RTT: {:.1}ms", duration_millis_f64(rtt)),
            );
        }
        msg_buf.push_str(closing);
        #[cfg(target_os = "windows")]
        if let Some(wait_err) = high_res_err {
            append_fmt(
                msg_buf,
                format_args!(
                    "\n[경고] Windows 고해상도 대기 타이머 실패로 표준 대기를 사용했습니다: {wait_err}"
                ),
            );
        }
        (Activity::Predicting(st), Some(msg_buf))
    }
    fn handle_measure_baseline_rtt<'message>(
        &mut self,
        mut base: Box<BaselineRttState>,
        msg_buf: &'message mut String,
        now: Instant,
    ) -> ActivityTransition<'message> {
        if !base.started {
            base.started = true;
            if !base.had_previous_sample {
                return (
                    Activity::MeasureBaselineRtt(base),
                    Some("1단계: RTT 기준값 측정을 시작합니다…"),
                );
            }
        }
        let had_pending = base.pending_generation.is_some();
        let Some(res) = self.sample_worker.poll_fetch(&mut base.pending_generation) else {
            if !had_pending && now >= base.next_sample_at {
                self.sample_worker
                    .ensure_fetch(&mut base.pending_generation);
            }
            return (Activity::MeasureBaselineRtt(base), None);
        };
        let idx = base.attempts;
        let next_base = match res {
            Ok(sample) => {
                *or_abort(base.rtt_nanos.get_mut(idx)) = sample.rtt.as_nanos();
                base.previous_sample = Some(sample);
                base.had_previous_sample = true;
                sample.response_received_inst
            }
            Err(err) => {
                append_error_detail(msg_buf, "RTT 샘플 수집 실패: ", err);
                return transition_to_retry(msg_buf, base.had_previous_sample);
            }
        };
        base.attempts = idx.strict_add(1);
        let Some(next_at) = next_base.checked_add(ADAPTIVE_POLL_INTERVAL) else {
            return transition_to_retry("다음 RTT 샘플 시각 계산 실패.", base.had_previous_sample);
        };
        base.next_sample_at = next_at;
        if base.attempts < NUM_SAMPLES {
            return (Activity::MeasureBaselineRtt(base), None);
        }
        let previous_sample = or_abort(base.previous_sample);
        base.rtt_nanos.sort_unstable();
        let sum = base
            .rtt_nanos
            .iter()
            .skip(2)
            .take(6)
            .fold(0_u128, |acc, &ns| acc.strict_add(ns));
        let baseline_rtt = Duration::from_nanos_u128(sum.div_euclid(6));
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
        st: ServerTime,
        sched: Option<ScheduledTrigger>,
        now: Instant,
    ) -> Result<ActivityTransition<'static>> {
        let est = st.current_server_time_at(now)?;
        let protect = if let Some(schedule) = sched
            && let ScheduledTarget::Confirmed(target_time, http_guard) = schedule.target
        {
            let rem = target_time.duration_since(est).ok();
            match rem {
                Some(dur) if dur > FINAL_COUNTDOWN_WINDOW => dur <= Duration::from_mins(1),
                _ => {
                    self.scheduled_trigger = None;
                    return Ok((
                        Activity::FinalCountdown(FinalCountdownState {
                            #[cfg(any(target_os = "windows", target_os = "macos"))]
                            action: schedule.action,
                            http_guard,
                            last_sample_error_message_at: None,
                            live_rtt: st.baseline_rtt,
                            next_sample_at: now,
                            pending_sample_generation: None,
                            sample_interval: final_countdown_sample_interval(rem),
                            server_time: st,
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
            && !protect
        {
            return Ok((
                Activity::measure_baseline(now, true),
                Some("서버 시간 보정 주기 도래, 재보정 시작."),
            ));
        }
        Ok((Activity::Predicting(st), None))
    }
    fn run_loop(
        &mut self,
        stop_after: Option<Duration>,
        out: &mut dyn io::Write,
        err: &mut dyn io::Write,
    ) -> Result<()> {
        #[cfg(target_os = "linux")]
        let trigger_action = self.scheduled_trigger.as_ref().map(|sched| sched.action);
        #[cfg(target_os = "linux")]
        if trigger_action.is_some() {
            writeln!(err, "[안내] Wayland 입력 권한 승인을 기다립니다.")?;
        }
        let mut stop: CoreResult<Instant, StopInput> = if let Some(dur) = stop_after {
            Ok(Instant::now()
                .checked_add(dur)
                .ok_or("관찰 종료 시각 계산 실패")?)
        } else {
            Err(StopInput::try_from(Some(8))
                .map_err(|src| AppError::context("종료 입력 준비 실패", src))?)
        };
        let poll_stop = |st: &mut CoreResult<Instant, StopInput>, now: Instant| match *st {
            Ok(at) => Ok(now >= at),
            Err(ref mut inp) => inp.poll(),
        };
        #[cfg(target_os = "linux")]
        let mut prepared_input = wayland_input::PreparedInput::EMPTY;
        #[cfg(target_os = "linux")]
        {
            let mut stop_err = None;
            let stop_req = prepared_input.prepare(trigger_action, err, || {
                match poll_stop(&mut stop, Instant::now()) {
                    Ok(req) => req,
                    Err(src) => {
                        stop_err = Some(src);
                        true
                    }
                }
            });
            if let Some(src) = stop_err {
                return Err(AppError::context("종료 입력 실패", src));
            }
            if stop_req {
                return Ok(());
            }
        }
        let mut msg_buf = String::new();
        msg_buf
            .try_reserve_exact(256)
            .map_err(|src| AppError::context("buffer 메모리 확보 실패", src))?;
        let mut activity = Activity::measure_baseline(self.last_display, false);
        loop {
            let pre_wait = Instant::now();
            if poll_stop(&mut stop, pre_wait)
                .map_err(|src| AppError::context("종료 입력 실패", src))?
            {
                break;
            }
            let act_poll = match activity {
                Activity::CalibrateOnTick(_)
                | Activity::FinalCountdown(_)
                | Activity::MeasureBaselineRtt(_) => ADAPTIVE_POLL_INTERVAL,
                Activity::Predicting(_) | Activity::Retrying(_, _) => Duration::from_millis(45),
            };
            let mut timeout = if Self::should_update_display(&activity, pre_wait) {
                act_poll.min(
                    self.display_interval
                        .saturating_sub(pre_wait.saturating_duration_since(self.last_display)),
                )
            } else {
                act_poll
            };
            if let Ok(stop_at) = stop {
                timeout = timeout.min(stop_at.saturating_duration_since(pre_wait));
            }
            thread::sleep(timeout);
            #[cfg(target_os = "linux")]
            prepared_input.maintain(err);
            let now = Instant::now();
            let sched = self.scheduled_trigger;
            let timing_sensitive = matches!(activity, Activity::FinalCountdown(_))
                || matches!(
                    (&activity, sched),
                    (
                        &Activity::Predicting(st),
                        Some(ScheduledTrigger {
                            action: _,
                            target: ScheduledTarget::Confirmed(target_time, _),
                        }),
                    ) if target_time
                        .duration_since(st.current_server_time_at(now)?)
                        .ok()
                        .is_none_or(|rem| rem <= FINAL_COUNTDOWN_WINDOW)
                );
            if !timing_sensitive {
                self.write_display_if_due(&activity, now, out)?;
            }
            msg_buf.clear();
            let (next_act, transition_msg) = match activity {
                Activity::MeasureBaselineRtt(base) => {
                    self.handle_measure_baseline_rtt(base, &mut msg_buf, now)
                }
                Activity::CalibrateOnTick(cal) => self.handle_calibrate_on_tick(cal, &mut msg_buf),
                Activity::Predicting(st) => self.handle_predicting(st, sched, now)?,
                Activity::FinalCountdown(cd) => self.handle_final_countdown(
                    cd,
                    &mut msg_buf,
                    err,
                    #[cfg(target_os = "linux")]
                    &mut prepared_input,
                ),
                Activity::Retrying(had_prev, started_at) => {
                    if now.saturating_duration_since(started_at) >= Duration::from_secs(10) {
                        (
                            Activity::measure_baseline(now, had_prev),
                            Some("[재시도] 동기화를 다시 시작합니다."),
                        )
                    } else {
                        (Activity::Retrying(had_prev, started_at), None)
                    }
                }
            };
            #[cfg(target_os = "windows")]
            self.sync_high_res_timer_state(&next_act, err)?;
            let wrote_msg = if let Some(msg) = transition_msg {
                writeln!(out, "\n{msg}")?;
                true
            } else {
                false
            };
            activity = next_act;
            if timing_sensitive && !wrote_msg {
                self.write_display_if_due(&activity, Instant::now(), out)?;
            }
        }
        Ok(())
    }
    fn should_update_display(activity: &Activity, now: Instant) -> bool {
        let Some(st) = activity.server_time() else {
            return false;
        };
        let Activity::FinalCountdown(cd) = *activity else {
            return true;
        };
        trigger_instant_for_target(
            st,
            cd.target_time,
            one_way_delay(cd.http_guard, cd.live_rtt),
        )
        .ok()
        .is_none_or(|trig| trig.saturating_duration_since(now) > FINAL_COUNTDOWN_FREEZE_WINDOW)
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
        let raw = unsafe {
            windows_input::create_waitable_timer_ex_w(null(), null(), 0x2, 0x0010_0000 | 0x2)
        };
        if let Some(handle) = NonNull::new(raw) {
            self.high_res_timer_guard = Some(HighResTimerGuard { handle });
        } else {
            writeln!(
                err,
                "[경고] Windows 고해상도 대기 타이머 생성에 실패했습니다. 카운트다운 정확도가 저하될 수 있습니다."
            )?;
        }
        Ok(())
    }
    fn write_display_if_due(
        &mut self,
        activity: &Activity,
        now: Instant,
        output: &mut dyn io::Write,
    ) -> Result<()> {
        if now.saturating_duration_since(self.last_display) >= self.display_interval
            && Self::should_update_display(activity, now)
            && let Some(st) = activity.server_time()
        {
            let mut cur = ByteCursor::new(&mut self.line_buf);
            cur.write_bytes("\r서버 시간: ".as_bytes());
            let since_epoch = st.current_server_time_at(now)?.duration_since(UNIX_EPOCH)?;
            let kst_secs = since_epoch.as_secs().strict_add(KST_OFFSET_SECS);
            let days = kst_secs.div_euclid(SECS_PER_DAY_U64);
            let dow_idx = usize::try_from(days.strict_add(4).rem_euclid(7))
                .unwrap_or_else(|_| process::abort());
            let (hour, minute, second) =
                hms_from_second_of_day(low_u32_from_u64(kst_secs.rem_euclid(SECS_PER_DAY_U64)));
            let (year, month, day) = http_date::civil_from_days(
                i32::try_from(days).unwrap_or_else(|_| process::abort()),
            );
            let mut year_buf = NumBuffer::new();
            cur.write_bytes(year.cast_unsigned().format_into(&mut year_buf).as_bytes());
            for val in [month, day] {
                cur.write_byte(b'-');
                write_2digits(&mut cur, val);
            }
            cur.write_byte(b'(');
            *cur.take_array::<3>() = or_abort(DAY_OF_WEEK_KO.get(dow_idx).copied());
            cur.write_bytes(b") ");
            write_2digits(&mut cur, hour);
            for val in [minute, second] {
                cur.write_byte(b':');
                write_2digits(&mut cur, val);
            }
            cur.write_byte(b'.');
            let millis = since_epoch.subsec_millis();
            cur.write_byte(digit_byte(low_u8_from_u32(millis.div_euclid(100))));
            write_2digits(&mut cur, millis.rem_euclid(100));
            cur.write_bytes(b" \r");
            let written_len = cur.written_len();
            output.write_all(self.line_buf.split_at(written_len).0)?;
            output.flush()?;
            self.last_display = now;
        }
        Ok(())
    }
}
impl SampleWorker {
    fn ensure_fetch(&mut self, pending_generation: &mut Option<u64>) {
        if pending_generation.is_some() {
            return;
        }
        let gen_id = self.generation.strict_add(1);
        match self.command_sender.try_send(gen_id) {
            Ok(()) => {
                self.generation = gen_id;
                *pending_generation = Some(gen_id);
            }
            Err(mpsc::TrySendError::Full(_)) => {}
            Err(mpsc::TrySendError::Disconnected(_)) => self.resume_panic(),
        }
    }
    fn poll_fetch(&mut self, pending_generation: &mut Option<u64>) -> Option<Result<TimeSample>> {
        let expected = *pending_generation;
        loop {
            match self.response_receiver.try_recv() {
                Ok((gen_id, result)) if Some(gen_id) == expected => {
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
impl ServerTimeSession {
    pub(super) fn run_loop(self, out: &mut dyn io::Write, err: &mut dyn io::Write) -> Result<()> {
        let (now, host) = (Instant::now(), self.host);
        let scheme = host.scheme;
        let (command_sender, command_receiver) = mpsc::sync_channel(1);
        let (response_sender, response_receiver) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name(String::from("srg-sample-worker"))
            .spawn(move || {
                let mut client = native_http::Client::default();
                let ctx = match host.scheme {
                    UrlScheme::Http => "HTTP",
                    UrlScheme::Https => "HTTPS",
                };
                while let Ok(gen_id) = command_receiver.recv() {
                    if response_sender
                        .send((gen_id, client.fetch_head(&host, ctx)))
                        .is_err()
                    {
                        return;
                    }
                }
            })?;
        let scheduled_trigger = self
            .scheduled_trigger
            .map(|(target, action)| ScheduledTrigger {
                action,
                target: ScheduledTarget::Pending(target),
            });
        if scheduled_trigger.is_some() && matches!(scheme, UrlScheme::Http) {
            err.write_all(
                "[경고] HTTP 서버 시간은 네트워크에서 위조될 수 있습니다.\n이상 보정, 과도한 지연 또는 최신 샘플 부재가 감지되면 예약 액션을 자동 취소합니다.\n"
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
        AppState {
            display_interval: if *IS_TERMINAL {
                Duration::from_millis(16)
            } else {
                Duration::from_secs(1)
            },
            #[cfg(target_os = "windows")]
            high_res_timer_attempted: false,
            #[cfg(target_os = "windows")]
            high_res_timer_guard: None,
            last_display: now,
            last_full_sync_at: now,
            line_buf: [0_u8; 80],
            sample_worker: SampleWorker {
                command_sender,
                generation: 0,
                response_receiver,
                scheme,
                worker: Some(worker),
            },
            scheduled_trigger,
        }
        .run_loop(self.stop_after, out, err)?;
        writeln!(out, "\n서버 시간 확인을 종료합니다.").map_err(Into::into)
    }
}
impl TargetTimeOfDay {
    fn resolve(self, st: ServerTime, now: Instant, msg_buf: &mut String) -> Result<SystemTime> {
        let target_sec = self.seconds_after_midnight;
        let kst_secs = st
            .current_server_time_at(now)?
            .duration_since(UNIX_EPOCH)?
            .as_secs()
            .checked_add(KST_OFFSET_SECS)
            .ok_or("KST 현재 시각 계산 중 overflow가 발생했습니다.")?;
        let current_sec = u32::try_from(kst_secs.rem_euclid(SECS_PER_DAY_U64))
            .map_err(|src| AppError::context("KST 초 변환 실패", src))?;
        let current_day = kst_secs.div_euclid(SECS_PER_DAY_U64);
        let target_day = if target_sec < current_sec {
            current_day
                .checked_add(1)
                .ok_or("다음날 목표 날짜 계산 중 overflow가 발생했습니다.")?
        } else {
            current_day
        };
        let target_utc_secs = target_day
            .checked_mul(SECS_PER_DAY_U64)
            .and_then(|day_start| day_start.checked_add(u64::from(target_sec)))
            .ok_or("목표 시각 계산 중 overflow가 발생했습니다.")?
            .checked_sub(KST_OFFSET_SECS)
            .ok_or("목표 UTC 시각 계산 중 underflow가 발생했습니다.")?;
        let target_st = UNIX_EPOCH
            .checked_add(Duration::from_secs(target_utc_secs))
            .ok_or("목표 절대 시각 계산 중 범위 오류가 발생했습니다.")?;
        let (year, month, day) = http_date::civil_from_days(
            i32::try_from(target_day)
                .map_err(|src| AppError::context("목표 날짜 변환 실패", src))?,
        );
        let (hour, minute, second) = hms_from_second_of_day(target_sec);
        append_fmt(
            msg_buf,
            format_args!(
                "\n목표 시각 확정(KST): {year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}"
            ),
        );
        Ok(target_st)
    }
}
impl FromStr for TargetTimeOfDay {
    type Err = &'static str;
    fn from_str(raw_input: &str) -> CoreResult<Self, Self::Err> {
        const ERR: &str = "잘못된 형식, 숫자 또는 시간 범위입니다 (HH:MM:SS, 0-23:0-59:0-59).";
        let (hour, minute, second) = parse_hms(raw_input).ok_or(ERR)?;
        (hour <= 23 && minute <= 59 && second <= 59).ok_or(ERR)?;
        Ok(Self {
            seconds_after_midnight: hour
                .strict_mul(3_600)
                .strict_add(minute.strict_mul(60))
                .strict_add(second),
        })
    }
}
fn append_error_detail(target: &mut String, prefix: &str, err: impl fmt::Display) {
    append_fmt(target, format_args!("{prefix}{err}"));
}
fn append_fmt(target: &mut String, args: fmt::Arguments<'_>) {
    assert!(
        FmtWrite::write_fmt(target, args).is_ok(),
        "String formatting must be infallible"
    );
}
const fn blend_rtt<const NEW_WEIGHT: u128>(old_value: Duration, new_value: Duration) -> Duration {
    Duration::from_nanos_u128(
        old_value
            .as_nanos()
            .strict_mul(10_u128.strict_sub(NEW_WEIGHT))
            .strict_add(new_value.as_nanos().strict_mul(NEW_WEIGHT))
            .div_euclid(10),
    )
}
const fn duration_millis_f64(duration: Duration) -> f64 {
    duration.as_secs_f64().algebraic_mul(1_000.0)
}
fn final_countdown_sample_interval(remaining: Option<Duration>) -> Duration {
    match remaining {
        Some(dur) if dur > Duration::from_secs(3) => Duration::from_millis(200),
        Some(dur) if dur > Duration::from_secs(1) => Duration::from_millis(50),
        _ => Duration::from_millis(25),
    }
}
const fn hms_from_second_of_day(second_of_day: u32) -> (u32, u32, u32) {
    (
        second_of_day.div_euclid(3_600),
        second_of_day.rem_euclid(3_600).div_euclid(60),
        second_of_day.rem_euclid(60),
    )
}
fn one_way_delay(http_guard: Option<HttpActionGuard>, rtt: Duration) -> Duration {
    let delay = NumericDiv::div(rtt, 2_u32);
    if http_guard.is_some() {
        delay.min(Duration::from_millis(500))
    } else {
        delay
    }
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
fn strict_server_deadline(st: ServerTime, target_time: SystemTime) -> Result<Instant> {
    Ok(st
        .anchor_instant
        .checked_add(target_time.duration_since(st.anchor_time)?)
        .ok_or("서버 deadline 계산 범위 오류")?)
}
fn transition_to_retry(msg: &str, had_previous_sample: bool) -> ActivityTransition<'_> {
    (
        Activity::Retrying(had_previous_sample, Instant::now()),
        Some(msg),
    )
}
fn trigger_instant_for_target(
    st: ServerTime,
    target_time: SystemTime,
    delay: Duration,
) -> Result<Instant> {
    Ok(strict_server_deadline(st, target_time)?
        .checked_sub(delay)
        .ok_or("액션 실행 deadline 계산 범위 오류")?)
}
fn write_2digits(cur: &mut ByteCursor<'_>, value: u32) {
    *cur.take_array::<2>() = two_digits(low_u8_from_u32(value));
}
