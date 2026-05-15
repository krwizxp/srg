use super::{
    MIN_TRANSFER_TIME, Result, TCP_TIMEOUT, TimeError, TimeErrorKind, TimeSample,
    http_date::parse_http_date_to_systemtime, sample::find_date_header_value,
};
use alloc::string::String;
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
    date_header: String,
    response_received_inst: Instant,
    rtt: Duration,
}
pub(super) struct NativeHttp {
    parse_http_date: fn(&str) -> Result<SystemTime>,
}
impl NativeHttp {
    pub(super) fn fetch_head_sample(&self, url: &str, context: &str) -> Result<TimeSample> {
        let response = platform::CLIENT.fetch_head(url, context)?;
        let server_time = (self.parse_http_date)(&response.date_header)?;
        Ok(TimeSample {
            response_received_inst: response.response_received_inst,
            rtt: response.rtt,
            server_time,
        })
    }
}
fn error(context: &str, detail: impl Display) -> TimeError {
    TimeError::new(TimeErrorKind::NativeHttp, format!("{context}: {detail}"))
}
fn missing_date(context: &str) -> TimeError {
    TimeError::header_not_found(format!("{context} 응답에서 Date"))
}
