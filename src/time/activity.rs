use super::ServerTime;
use core::time::Duration;
use std::time::{Instant, SystemTime};
pub(super) const ADAPTIVE_POLL_INTERVAL: Duration = Duration::from_millis(10);
const PASSIVE_POLL_INTERVAL: Duration = Duration::from_millis(45);
#[derive(Clone, Copy, Debug)]
pub(super) enum Activity {
    CalibrateOnTick,
    FinalCountdown {
        server_time: ServerTime,
        target_time: SystemTime,
    },
    MeasureBaselineRtt,
    Predicting {
        server_time: ServerTime,
    },
    Retrying {
        started_at: Instant,
    },
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
            Self::Predicting { .. } | Self::Retrying { .. } => PASSIVE_POLL_INTERVAL,
        }
    }
    pub(super) const fn server_time(&self) -> Option<ServerTime> {
        match *self {
            Self::FinalCountdown { server_time, .. } | Self::Predicting { server_time } => {
                Some(server_time)
            }
            Self::CalibrateOnTick | Self::MeasureBaselineRtt | Self::Retrying { .. } => None,
        }
    }
}
