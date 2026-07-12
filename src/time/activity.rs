use core::time::Duration;
use std::time::{Instant, SystemTime};
pub(super) const ADAPTIVE_POLL_INTERVAL: Duration = Duration::from_millis(10);
const PASSIVE_POLL_INTERVAL: Duration = Duration::from_millis(45);
#[derive(Clone, Copy, Debug)]
pub(super) enum Activity {
    CalibrateOnTick,
    FinalCountdown { target_time: SystemTime },
    MeasureBaselineRtt,
    Predicting,
    Retrying { retry_at: Instant },
}
pub(super) enum CountdownTrigger {
    Late(Duration),
    WithRemaining(Duration),
}
impl Activity {
    pub(super) const fn is_final_countdown(&self) -> bool {
        matches!(self, Self::FinalCountdown { .. })
    }
    pub(super) const fn poll_interval(&self) -> Duration {
        match *self {
            Self::CalibrateOnTick | Self::FinalCountdown { .. } | Self::MeasureBaselineRtt => {
                ADAPTIVE_POLL_INTERVAL
            }
            Self::Predicting | Self::Retrying { .. } => PASSIVE_POLL_INTERVAL,
        }
    }
}
