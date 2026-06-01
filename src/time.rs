use self::{
    activity::{Activity, CountdownDecision},
    address::ParsedServer,
    diagnostic::{Result, TimeError, TimeErrorKind},
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
use crate::{ServerTimeSession, buffmt::ByteCursor, write_line_ignored};
use alloc::fmt;
use core::{fmt::Write as FmtWrite, ops::Mul as NumericMul, range::Range, time::Duration};
use std::{
    io::{self, BufRead as IoBufRead, Result as IoResult},
    net,
    sync::mpsc,
    thread,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
mod activity;
pub mod address;
pub mod diagnostic;
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
#[derive(Clone, Copy, Debug)]
pub enum TriggerAction {
    F5Press,
    LeftClick,
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
struct CachedTcpSocketAddr {
    addr: net::SocketAddr,
    host: String,
    port: u16,
}
struct NetworkContext {
    cached_tcp_socket_addr: Option<CachedTcpSocketAddr>,
    tcp_line_buffer: Vec<u8>,
}
impl ServerTimeSession {
    pub fn run_loop(self, out: &mut dyn io::Write, err: &mut dyn io::Write) -> Result<()> {
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
    fn finish_baseline_rtt_measurement<'message>(
        &mut self,
        sample_count: usize,
        msg_buf: &'message mut String,
    ) -> (Activity, Option<&'message str>) {
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
        (Activity::CalibrateOnTick, Some(msg_buf))
    }
    fn handle_calibrate_on_tick<'message>(
        &mut self,
        _msg_buf: &'message mut str,
        net_ctx: &mut NetworkContext,
    ) -> (Activity, Option<&'message str>) {
        let Ok(current_sample) = fetch_server_time_sample(&self.host, net_ctx) else {
            self.calibration_failure_count = self.calibration_failure_count.saturating_add(1);
            if self.calibration_failure_count >= MAX_CALIBRATION_FAILURES {
                return transition_to_retry(
                    "정밀 보정 중 서버 응답을 지속적으로 받지 못했습니다. 전체 보정을 다시 시작합니다.",
                );
            }
            return (Activity::CalibrateOnTick, None);
        };
        self.calibration_failure_count = 0;
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
    fn handle_final_countdown<'message>(
        &mut self,
        target_time: SystemTime,
        msg_buf: &'message mut String,
        net_ctx: &mut NetworkContext,
        prepared_input: &mut native_input::PreparedInput,
        err: &mut dyn io::Write,
    ) -> (Activity, Option<&'message str>) {
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
    fn handle_final_countdown_fetch_error<'message>(
        &self,
        target_time: SystemTime,
        msg_buf: &'message mut String,
        err: &TimeError,
        prepared_input: &mut native_input::PreparedInput,
        stderr: &mut dyn io::Write,
    ) -> (Activity, Option<&'message str>) {
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
    fn handle_measure_baseline_rtt<'message>(
        &mut self,
        msg_buf: &'message mut String,
        net_ctx: &mut NetworkContext,
        now: Instant,
        out: &mut dyn io::Write,
    ) -> (Activity, Option<&'message str>) {
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
    fn handle_predicting<'message>(
        &mut self,
        _msg_buf: &'message mut String,
        now: Instant,
    ) -> (Activity, Option<&'message str>) {
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
    fn next_activity<'message>(
        &mut self,
        activity: &Activity,
        message_buffer: &'message mut String,
        now: Instant,
        runtime: &mut LoopRuntime<'_>,
    ) -> (Activity, Option<&'message str>) {
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
    fn run_loop(&mut self, out: &mut dyn io::Write, err: &mut dyn io::Write) -> Result<()> {
        out.write_all("\n서버 시간 확인을 시작합니다... (Enter를 누르면 종료)\n".as_bytes())?;
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || -> IoResult<()> {
            let mut line = Vec::new();
            line.try_reserve(ENTER_BUFFER_CAPACITY)
                .map_err(io::Error::other)?;
            let mut stdin_lock = io::stdin().lock();
            IoBufRead::read_until(&mut stdin_lock, b'\n', &mut line)?;
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
        let mut tcp_line_buffer = Vec::new();
        tcp_line_buffer
            .try_reserve(TCP_LINE_BUFFER_CAPACITY)
            .map_err(|source| {
                TimeError::parse(format!("TCP line buffer 메모리 확보 실패: {source}"))
            })?;
        let mut network_context = NetworkContext {
            cached_tcp_socket_addr: None,
            tcp_line_buffer,
        };
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
                let mut cur = display::SliceCursor {
                    inner: ByteCursor::new(line_buf.as_mut_slice()),
                };
                match (|| -> Result<()> {
                    cur.write_bytes(DISPLAY_STATUS_PREFIX.as_bytes())
                        .map_err(TimeError::from)?;
                    st.write_current_display_time_buf_at(&mut cur, now)?;
                    cur.write_bytes(b" \r").map_err(TimeError::from)?;
                    Ok(())
                })() {
                    Ok(()) => {
                        out.write_all(cur.written_slice()?)?;
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
                    return;
                }
                let guard = HighResTimerGuard;
                self.high_res_timer_guard = Some(guard);
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
                self.trigger_action.map(TriggerAction::native_input_action),
                err,
            );
            return;
        }
        if !next_activity.is_final_countdown() {
            prepared_input.reset();
        }
    }
    fn trigger_and_finish<'message>(
        &self,
        msg_buf: &'message mut String,
        log_message: fmt::Arguments,
        prepared_input: &mut native_input::PreparedInput,
        err: &mut dyn io::Write,
    ) -> (Activity, Option<&'message str>) {
        if let Some(action) = self.trigger_action.map(TriggerAction::native_input_action) {
            prepared_input.send(action, err);
        }
        append_fmt(msg_buf, log_message);
        (Activity::Finished, Some(msg_buf))
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
fn transition_to_retry(msg: &str) -> (Activity, Option<&str>) {
    let now = Instant::now();
    let Some(retry_at) = now.checked_add(RETRY_DELAY) else {
        return (Activity::Retrying { retry_at: now }, Some(msg));
    };
    (Activity::Retrying { retry_at }, Some(msg))
}
