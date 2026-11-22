use super::{MIN_TRANSFER_TIME, Result, TimeError, TimeErrorKind, TimeSample, http_date::HttpDate};
#[cfg(target_os = "windows")]
use core::error::Error;
use core::result::Result as CoreResult;
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
#[derive(Default)]
struct FreshTimeHeaders {
    age_result: Option<CoreResult<(), &'static str>>,
    date_result: Option<Result<(SystemTime, Instant)>>,
}
impl FreshTimeHeaders {
    fn capture_age(&mut self, raw: &str) {
        self.age_result = Some(if self.age_result.is_some() {
            Err("Age 헤더가 여러 개입니다.")
        } else if raw.is_empty() {
            Err("Age 헤더 값이 비어 있습니다.")
        } else if let Some(nonzero) = raw.bytes().try_fold(false, |nonzero, byte| {
            byte.is_ascii_digit().then_some(nonzero || byte != b'0')
        }) {
            (!nonzero).ok_or("Age 헤더가 0보다 커 캐시된 응답입니다.")
        } else {
            Err("Age 헤더 값이 숫자가 아닙니다.")
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
    fn finish(self, request_start: Instant, context: &str) -> Result<TimeSample> {
        self.age_result
            .transpose()
            .map_err(|message| error(context, message))?;
        let (server_time, response_received_inst) = self.date_result.ok_or_else(|| {
            TimeError::new(
                TimeErrorKind::HeaderNotFound,
                format!("{context} 응답에서 Date"),
            )
        })??;
        let rtt = response_received_inst
            .checked_duration_since(request_start)
            .ok_or_else(|| TimeError::parse("HTTP 응답 시각이 요청 시작 시각보다 앞섭니다."))?
            .max(MIN_TRANSFER_TIME);
        Ok(TimeSample {
            response_received_inst,
            rtt,
            server_time,
        })
    }
}
#[cold]
#[inline(never)]
fn error(context: &str, detail: &str) -> TimeError {
    let mut message = String::with_capacity(context.len().strict_add(detail.len()).strict_add(2));
    message.push_str(context);
    message.push_str(": ");
    message.push_str(detail);
    TimeError::new(TimeErrorKind::NativeHttp, message)
}
#[cfg(target_os = "windows")]
#[cold]
#[inline(never)]
fn error_with_source<E>(context: &str, detail: &str, source: E) -> TimeError
where
    E: Error + Send + Sync + 'static,
{
    let mut message = String::with_capacity(context.len().strict_add(detail.len()).strict_add(2));
    message.push_str(context);
    message.push_str(": ");
    message.push_str(detail);
    TimeError::new_with_source(TimeErrorKind::NativeHttp, message, source)
}
