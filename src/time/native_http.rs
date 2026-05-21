pub(super) use super::sample::find_date_header_value;
use super::{
    MIN_TRANSFER_TIME, Result, TCP_TIMEOUT, TimeError, TimeErrorKind, TimeSample,
    http_date::parse_http_date_to_systemtime,
};
use core::{fmt::Display, time::Duration};
use std::time::{Instant, SystemTime};
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
pub(super) const NATIVE_HTTP: NativeHttp = NativeHttp {
    parse_http_date: parse_http_date_to_systemtime,
};
struct HeadResponse {
    response_received_inst: Instant,
    rtt: Duration,
    server_time: SystemTime,
}
pub(super) struct NativeHttp {
    parse_http_date: fn(&str) -> Result<SystemTime>,
}
type ParseHttpDate = fn(&str) -> Result<SystemTime>;
impl NativeHttp {
    pub(super) fn fetch_head_sample(&self, url: &str, context: &str) -> Result<TimeSample> {
        let response = platform::CLIENT.fetch_head(url, context, self.parse_http_date)?;
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
