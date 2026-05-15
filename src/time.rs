use self::{
    address::ParsedServer,
    diagnostic::{Result, TimeError, TimeErrorKind},
    sample::{NetworkContext, NetworkContextRequest, fetch_server_time_sample},
    util::blend_weighted_nanos,
};
use crate::write_line_ignored;
use alloc::fmt;
#[cfg(target_os = "windows")]
use core::result::Result as StdResult;
use core::{fmt::Write as _, ops::Mul as _, time::Duration};
use std::{
    io::{self, BufRead as _, Result as IoResult},
    sync::mpsc,
    thread,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
pub mod address;
pub mod diagnostic;
mod display;
mod http_date;
mod native_http;
mod sample;
mod util;
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
pub const NUM_SAMPLES: usize = 10;
const FINAL_COUNTDOWN_RTT_ALPHA_NUM: u32 = 7;
const FINAL_COUNTDOWN_RTT_ALPHA_DENOM: u32 = 10;
const MAX_CALIBRATION_FAILURES: u32 = 100;
const KST_OFFSET: Duration = Duration::from_hours(9);
pub const KST_OFFSET_SECS_U64: u64 = KST_OFFSET.as_secs();
const KST_OFFSET_SECS: i64 = KST_OFFSET_SECS_U64.cast_signed();
const DISPLAY_INTERVAL: Duration = Duration::from_millis(16);
const ADAPTIVE_POLL_INTERVAL: Duration = Duration::from_millis(10);
const COUNTDOWN_DELAY_ERROR: &str = "카운트다운 지연 계산 실패";
const DISPLAY_STATUS_PREFIX: &str = "\r서버 시간: ";
const DISPLAY_UPDATE_INTERVAL: Duration = Duration::from_millis(45);
const FINAL_COUNTDOWN_WINDOW: Duration = Duration::from_secs(10);
const HALF_RTT_DIVISOR: u32 = 2;
const MESSAGE_BUFFER_CAPACITY: usize = 256;
const MILLIS_PER_SECOND_F64: f64 = 1000.0;
const MIN_TRANSFER_TIME: Duration = Duration::from_micros(1);
const RTT_TRIM_DIVISOR: usize = 5;
cfg_select! {
    target_os = "windows" => {
        const TIMERR_NOERROR: u32 = 0;
        const TARGET_PERIOD_MS: u32 = 1;
        pub struct HighResTimerGuard;
        struct HighResTimerRequest;
        #[link(name = "winmm")]
        unsafe extern "system" {
            fn timeBeginPeriod(u_period: u32) -> u32;
            fn timeEndPeriod(u_period: u32) -> u32;
        }
        impl TryFrom<HighResTimerRequest> for HighResTimerGuard {
            type Error = ();
            fn try_from(_value: HighResTimerRequest) -> StdResult<Self, Self::Error> {
                // SAFETY: `timeBeginPeriod` is a WinMM FFI call with a plain
                // integer input and does not impose additional aliasing or
                // lifetime requirements.
                if unsafe { timeBeginPeriod(TARGET_PERIOD_MS) } == TIMERR_NOERROR {
                    Ok(Self)
                } else {
                    Err(())
                }
            }
        }
        impl Drop for HighResTimerGuard {
            fn drop(&mut self) {
                // SAFETY: This releases the timer period requested when the guard was
                // created using the same value on the same process.
                unsafe { timeEndPeriod(TARGET_PERIOD_MS) };
            }
        }
    }
    _ => {}
}
#[derive(Clone, Copy, Debug)]
pub struct TimeSample {
    pub response_received_inst: Instant,
    pub rtt: Duration,
    pub server_time: SystemTime,
}
#[derive(Clone, Copy, Debug)]
pub struct ServerTime {
    anchor_instant: Instant,
    anchor_time: SystemTime,
    baseline_rtt: Duration,
}
#[derive(Clone, Copy, Debug)]
pub enum TriggerAction {
    F5Press,
    LeftClick,
}
impl From<TriggerAction> for native_input::InputAction {
    fn from(action: TriggerAction) -> Self {
        match action {
            TriggerAction::LeftClick => Self::MouseClick,
            TriggerAction::F5Press => Self::F5Press,
        }
    }
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
    pub host: ParsedServer,
    pub last_sample: Option<TimeSample>,
    pub live_rtt: Option<Duration>,
    pub next_full_sync_at: Instant,
    pub server_time: Option<ServerTime>,
    pub target_time: Option<SystemTime>,
    pub trigger_action: Option<TriggerAction>,
}
struct LoopRuntime<'a> {
    err: &'a mut dyn io::Write,
    network_context: &'a mut NetworkContext,
    out: &'a mut dyn io::Write,
    prepared_input: &'a mut native_input::PreparedInput,
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
enum CountdownDecision {
    TriggerLate,
    TriggerWithRemaining(Duration),
    Wait,
}
impl Activity {
    const fn is_final_countdown(&self) -> bool {
        matches!(self, Self::FinalCountdown { .. })
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
    fn finish_baseline_rtt_measurement<'a>(
        &mut self,
        sample_count: usize,
        msg_buf: &'a mut String,
    ) -> (Activity, Option<&'a str>) {
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
        let trim = filled.div_euclid(RTT_TRIM_DIVISOR);
        let Some(window_end) = filled.checked_sub(trim) else {
            return transition_to_retry("RTT 샘플 윈도우 계산 실패.");
        };
        let Some(window) = rtts.get(trim..window_end) else {
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
                return (Activity::Predicting, Some("[성공] 정밀 보정 완료!"));
            }
        }
        self.last_sample = Some(current_sample);
        (Activity::CalibrateOnTick, None)
    }
    fn handle_final_countdown<'a>(
        &mut self,
        target_time: SystemTime,
        msg_buf: &'a mut String,
        net_ctx: &mut NetworkContext,
        prepared_input: &mut native_input::PreparedInput,
        err: &mut dyn io::Write,
    ) -> (Activity, Option<&'a str>) {
        let sample = match fetch_server_time_sample(&self.host, net_ctx) {
            Ok(sample_value) => sample_value,
            Err(fetch_err) => {
                return self.handle_final_countdown_fetch_error(
                    target_time,
                    msg_buf,
                    &fetch_err,
                    prepared_input,
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
        let now = Instant::now();
        let current_server_time = st.current_server_time_at(now);
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
            return (Activity::FinalCountdown { target_time }, Some(msg_buf));
        };
        match decide_countdown_action(target_time, current_server_time, one_way_delay) {
            CountdownDecision::TriggerWithRemaining(duration_until_target) => {
                self.trigger_and_finish(
                    msg_buf,
                    format_args!(
                        "\n>>> 액션 실행! (목표 도달까지 {:.1}ms 남음) (지연 예측: {:.1}ms, 실측 RTT: {:.1}ms)",
                        duration_millis_f64(duration_until_target),
                        duration_millis_f64(one_way_delay),
                        duration_millis_f64(sample.rtt)
                    ),
                    prepared_input,
                    err,
                )
            }
            CountdownDecision::TriggerLate => self.trigger_and_finish(
                msg_buf,
                format_args!(
                    "\n>>> 액션 실행! (시간 초과) (지연 예측: {:.1}ms, 실측 RTT: {:.1}ms)",
                    duration_millis_f64(one_way_delay),
                    duration_millis_f64(sample.rtt)
                ),
                prepared_input,
                err,
            ),
            CountdownDecision::Wait => (Activity::FinalCountdown { target_time }, None),
        }
    }
    fn handle_final_countdown_fetch_error<'a>(
        &self,
        target_time: SystemTime,
        msg_buf: &'a mut String,
        err: &TimeError,
        prepared_input: &mut native_input::PreparedInput,
        stderr: &mut dyn io::Write,
    ) -> (Activity, Option<&'a str>) {
        if let Some(st) = self.server_time.as_ref() {
            let now = Instant::now();
            let current_server_time = st.current_server_time_at(now);
            let effective_rtt = self.live_rtt.unwrap_or(st.baseline_rtt);
            let Some(one_way_delay) = effective_one_way_delay(effective_rtt) else {
                msg_buf.push_str(COUNTDOWN_DELAY_ERROR);
                return (Activity::FinalCountdown { target_time }, Some(msg_buf));
            };
            match decide_countdown_action(target_time, current_server_time, one_way_delay) {
                CountdownDecision::TriggerWithRemaining(duration_until_target) => {
                    return self.trigger_and_finish(
                        msg_buf,
                        format_args!(
                            concat!(
                                "\n>>> 액션 실행! (예측값 기준 강제 실행, 목표 도달까지 {:.1}ms 남음) ",
                                "(지연 예측: {:.1}ms, 원인: 카운트다운 샘플 실패: {})"
                            ),
                            duration_millis_f64(duration_until_target),
                            duration_millis_f64(one_way_delay),
                            err
                        ),
                        prepared_input,
                        stderr,
                    );
                }
                CountdownDecision::TriggerLate => {
                    return self.trigger_and_finish(
                        msg_buf,
                        format_args!(
                            concat!(
                                "\n>>> 액션 실행! (예측값 기준 강제 실행, 시간 초과) ",
                                "(지연 예측: {:.1}ms, 원인: 카운트다운 샘플 실패: {})"
                            ),
                            duration_millis_f64(one_way_delay),
                            err
                        ),
                        prepared_input,
                        stderr,
                    );
                }
                CountdownDecision::Wait => {}
            }
        }
        append_error_detail(msg_buf, "카운트다운 샘플 획득 실패: ", err);
        (Activity::FinalCountdown { target_time }, Some(msg_buf))
    }
    fn handle_measure_baseline_rtt<'a>(
        &mut self,
        msg_buf: &'a mut String,
        net_ctx: &mut NetworkContext,
        now: Instant,
        out: &mut dyn io::Write,
    ) -> (Activity, Option<&'a str>) {
        if self.baseline_rtt_attempts == 0 {
            self.begin_baseline_rtt_measurement(now, out);
        }
        if now < self.baseline_rtt_next_sample_at {
            return (Activity::MeasureBaselineRtt, None);
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
            return (Activity::MeasureBaselineRtt, None);
        }
        self.finish_baseline_rtt_measurement(self.baseline_rtt_valid_count, msg_buf)
    }
    fn handle_predicting<'a>(
        &mut self,
        _msg_buf: &'a mut String,
        now: Instant,
    ) -> (Activity, Option<&'a str>) {
        let Some(server_time) = self.server_time.as_ref() else {
            return (Activity::MeasureBaselineRtt, None);
        };
        let estimated_server_time = server_time.current_server_time_at(now);
        if let Some(target_time) = self.target_time.take_if(|target| {
            target
                .duration_since(estimated_server_time)
                .is_ok_and(|duration_until_target| duration_until_target <= FINAL_COUNTDOWN_WINDOW)
        }) {
            self.live_rtt = Some(server_time.baseline_rtt);
            return (
                Activity::FinalCountdown { target_time },
                Some("최종 카운트다운 시작!"),
            );
        }
        if now >= self.next_full_sync_at {
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
    fn next_activity<'a>(
        &mut self,
        activity: &Activity,
        message_buffer: &'a mut String,
        now: Instant,
        runtime: &mut LoopRuntime<'_>,
    ) -> (Activity, Option<&'a str>) {
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
                runtime.network_context,
                runtime.prepared_input,
                runtime.err,
            ),
            Activity::Finished => (Activity::Predicting, Some("액션 완료. 예측 모드 전환.")),
            Activity::Retrying { retry_at } => {
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
    }
    pub(super) fn run_loop(
        &mut self,
        out: &mut dyn io::Write,
        err: &mut dyn io::Write,
    ) -> Result<()> {
        out.write_all("\n서버 시간 확인을 시작합니다... (Enter를 누르면 종료)\n".as_bytes())?;
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || -> IoResult<()> {
            let mut line = Vec::new();
            line.try_reserve(ENTER_BUFFER_CAPACITY)
                .map_err(io::Error::other)?;
            io::stdin().lock().read_until(b'\n', &mut line)?;
            match tx.send(()) {
                Ok(()) | Err(_) => {}
            }
            Ok(())
        });
        let mut activity = Activity::MeasureBaselineRtt;
        let mut last_display_update = Instant::now();
        let mut message_buffer = String::new();
        message_buffer
            .try_reserve(MESSAGE_BUFFER_CAPACITY)
            .map_err(io::Error::other)?;
        let mut network_context = NetworkContext::try_from(NetworkContextRequest)?;
        let mut prepared_input = native_input::PreparedInput::EMPTY;
        let mut line_buf = [0_u8; display::DISPLAY_LINE_BUF_LEN];
        loop {
            let activity_poll = match activity {
                Activity::MeasureBaselineRtt
                | Activity::CalibrateOnTick
                | Activity::FinalCountdown { .. } => ADAPTIVE_POLL_INTERVAL,
                Activity::Predicting | Activity::Finished | Activity::Retrying { .. } => {
                    DISPLAY_UPDATE_INTERVAL
                }
            };
            let pre_wait_now = Instant::now();
            let elapsed = pre_wait_now.duration_since(last_display_update);
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
                let mut cur = display::SliceCursor {
                    inner: crate::buffmt::ByteCursor::new(&mut line_buf),
                };
                match (|| -> Result<()> {
                    cur.write_bytes(DISPLAY_STATUS_PREFIX.as_bytes())
                        .map_err(TimeError::from)?;
                    st.write_current_display_time_buf_at(&mut cur, true, now)?;
                    cur.write_bytes(b" \r").map_err(TimeError::from)?;
                    Ok(())
                })() {
                    Ok(()) => {
                        out.write_all(cur.inner.written_slice()?)?;
                        out.flush()?;
                    }
                    Err(display_err) => {
                        out.write_all(DISPLAY_STATUS_PREFIX.as_bytes())?;
                        write!(out, "표시 버퍼 오류: {display_err}")?;
                        out.write_all(b" \r")?;
                        out.flush()?;
                    }
                }
                last_display_update = now;
            }
            message_buffer.clear();
            let (next_activity, log_opt_msg) = {
                let mut runtime = LoopRuntime {
                    err,
                    network_context: &mut network_context,
                    out,
                    prepared_input: &mut prepared_input,
                };
                self.next_activity(&activity, &mut message_buffer, now, &mut runtime)
            };
            self.sync_prepared_input_state(&activity, &next_activity, &mut prepared_input, err);
            cfg_select! {
                windows => {
                    self.sync_high_res_timer_state(&next_activity, err);
                }
                _ => {}
            }
            if let Some(console_msg) = log_opt_msg {
                writeln!(out, "\n{console_msg}")?;
            }
            activity = next_activity;
        }
        Ok(())
    }
    #[cfg(windows)]
    fn sync_high_res_timer_state(&mut self, next_activity: &Activity, err: &mut dyn io::Write) {
        if next_activity.is_final_countdown() {
            if self.high_res_timer_guard.is_none() {
                if let Ok(guard) = HighResTimerGuard::try_from(HighResTimerRequest) {
                    self.high_res_timer_guard = Some(guard);
                } else {
                    write_line_ignored(
                        err,
                        format_args!(
                            concat!(
                                "[경고] Windows 타이머 해상도 {}ms 요청에 실패했습니다. ",
                                "카운트다운 정확도가 저하될 수 있습니다."
                            ),
                            TARGET_PERIOD_MS
                        ),
                    );
                }
            }
        } else {
            self.high_res_timer_guard = None;
        }
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
                self.trigger_action.map(native_input::InputAction::from),
                err,
            );
            return;
        }
        if !next_activity.is_final_countdown() {
            prepared_input.reset();
        }
    }
    fn trigger_and_finish<'a>(
        &self,
        msg_buf: &'a mut String,
        log_message: fmt::Arguments,
        prepared_input: &mut native_input::PreparedInput,
        err: &mut dyn io::Write,
    ) -> (Activity, Option<&'a str>) {
        if let Some(action) = self.trigger_action.map(native_input::InputAction::from) {
            prepared_input.send(action, err);
        }
        append_fmt(msg_buf, log_message);
        (Activity::Finished, Some(msg_buf))
    }
}
fn duration_millis_f64(duration: Duration) -> f64 {
    duration.as_secs_f64().mul(MILLIS_PER_SECOND_F64)
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
    let result = target.write_fmt(args);
    debug_assert!(result.is_ok(), "writing to String should not fail");
}
fn transition_to_retry(msg: &str) -> (Activity, Option<&str>) {
    let now = Instant::now();
    let Some(retry_at) = now.checked_add(RETRY_DELAY) else {
        return (Activity::Retrying { retry_at: now }, Some(msg));
    };
    (Activity::Retrying { retry_at }, Some(msg))
}
