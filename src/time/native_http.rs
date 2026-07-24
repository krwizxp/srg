use super::{MIN_TRANSFER_TIME, Result, TimeError, TimeErrorKind, TimeSample, http_date::HttpDate};
#[cfg(target_os = "windows")]
use core::error::Error;
use core::{fmt::Display, result::Result as CoreResult};
use std::time::{Instant, SystemTime};
cfg_select! {
    any(target_os = "linux", target_os = "macos") => {
        pub(super) use self::libcurl::Client;
        mod libcurl;
    }
    target_os = "windows" => {
        pub(super) use self::winhttp::Client;
        mod winhttp;
    }
    _ => {
        compile_error!("SRG native HTTP supports only Windows, Linux, and macOS.");
    }
}
const AGE_HEADER_PREFIX: &[u8; 4] = b"age:";
const DATE_HEADER_PREFIX: &[u8; 5] = b"date:";
#[derive(Default)]
struct FreshTimeHeaders {
    age_result: Option<CoreResult<(), &'static str>>,
    date_result: Option<Result<(SystemTime, Instant)>>,
}
impl FreshTimeHeaders {
    fn capture_age(&mut self, raw: &str) {
        self.age_result = Some(if self.age_result.is_some() {
            Err("Age 헤더가 여러 개입니다.")
        } else {
            let trimmed = raw.trim_ascii();
            if trimmed.is_empty() {
                Err("Age 헤더 값이 비어 있습니다.")
            } else if trimmed.bytes().any(|byte| !byte.is_ascii_digit()) {
                Err("Age 헤더 값이 숫자가 아닙니다.")
            } else if trimmed.bytes().any(|byte| byte != b'0') {
                Err("Age 헤더가 0보다 커 캐시된 응답입니다.")
            } else {
                Ok(())
            }
        });
    }
    fn capture_date(&mut self, raw: &str, received_at: Instant) {
        self.date_result = Some(if self.date_result.is_some() {
            Err(TimeError::parse("Date 헤더가 여러 개입니다."))
        } else {
            raw.parse::<HttpDate>()
                .map(|HttpDate(server_time)| (server_time, received_at))
        });
    }
    fn finish(self, context: &str) -> Result<(SystemTime, Instant)> {
        if let Some(age_result) = self.age_result {
            age_result.map_err(|message| error(context, message))?;
        }
        self.date_result
            .ok_or_else(|| TimeError::header_not_found(format!("{context} 응답에서 Date")))?
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
