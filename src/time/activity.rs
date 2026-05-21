use core::time::Duration;
use std::time::{Instant, SystemTime};
#[derive(Clone, Copy, Debug)]
pub(super) enum Activity {
    CalibrateOnTick,
    FinalCountdown { target_time: SystemTime },
    Finished,
    MeasureBaselineRtt,
    Predicting,
    Retrying { retry_at: Instant },
}
pub(super) enum CountdownDecision {
    TriggerLate,
    TriggerWithRemaining(Duration),
    Wait,
}
impl Activity {
    pub(super) const fn is_final_countdown(&self) -> bool {
        matches!(self, Self::FinalCountdown { .. })
    }
}
