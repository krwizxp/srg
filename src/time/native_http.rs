use super::{
    Result, TimeError, TimeErrorKind, TimeSample, http_date::parse_http_date_to_systemtime,
};
#[cfg(target_os = "windows")]
use core::error::Error;
use core::{fmt::Display, time::Duration};
use std::time::{Instant, SystemTime};
cfg_select! {
    any(target_os = "linux", target_os = "macos", target_os = "windows") => {
        use super::{MIN_TRANSFER_TIME, TCP_TIMEOUT};
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
        use self::unsupported as platform;
        mod unsupported;
    }
}
struct HeadResponse {
    response_received_inst: Instant,
    rtt: Duration,
    server_time: SystemTime,
}
pub(super) struct NativeHttp {
    parse_http_date: fn(&str) -> Result<SystemTime>,
    platform: platform::Client,
}
type ParseHttpDate = fn(&str) -> Result<SystemTime>;
impl NativeHttp {
    pub(super) fn fetch_head_sample(&mut self, url: &str, context: &str) -> Result<TimeSample> {
        let response = self
            .platform
            .fetch_head(url, context, self.parse_http_date)?;
        Ok(TimeSample {
            response_received_inst: response.response_received_inst,
            rtt: response.rtt,
            server_time: response.server_time,
        })
    }
}
impl Default for NativeHttp {
    fn default() -> Self {
        Self {
            parse_http_date: parse_http_date_to_systemtime,
            platform: platform::Client::default(),
        }
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
