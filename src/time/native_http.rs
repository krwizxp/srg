use super::{Result, TimeError, TimeErrorKind, TimeSample};
#[cfg(target_os = "windows")]
use core::error::Error;
use core::{fmt::Display, time::Duration};
use std::time::{Instant, SystemTime};
cfg_select! {
    any(target_os = "linux", target_os = "macos", target_os = "windows") => {
        use super::MIN_TRANSFER_TIME;
    }
    _ => {}
}
cfg_select! {
    any(target_os = "linux", target_os = "macos") => {
        use self::libcurl as platform;
        mod libcurl;
    }
    target_os = "windows" => {
        use self::winhttp as platform;
        mod winhttp;
    }
    _ => {
        compile_error!("SRG native HTTP supports only Windows, Linux, and macOS.");
    }
}
struct HeadResponse {
    response_received_inst: Instant,
    rtt: Duration,
    server_time: SystemTime,
}
#[derive(Default)]
pub(super) struct NativeHttp {
    platform: platform::Client,
}
impl NativeHttp {
    pub(super) fn fetch_head_sample(&mut self, url: &str, context: &str) -> Result<TimeSample> {
        let response = self.platform.fetch_head(url, context)?;
        Ok(TimeSample {
            response_received_inst: response.response_received_inst,
            rtt: response.rtt,
            server_time: response.server_time,
        })
    }
}
fn error(context: &str, detail: impl Display) -> TimeError {
    TimeError::new(TimeErrorKind::NativeHttp, format!("{context}: {detail}"))
}
#[cfg(target_os = "windows")]
fn error_with_source<E>(context: &str, detail: impl Display, source: E) -> TimeError
where
    E: Error + Send + Sync + 'static,
{
    TimeError::new_with_source(
        TimeErrorKind::NativeHttp,
        format!("{context}: {detail}"),
        source,
    )
}
