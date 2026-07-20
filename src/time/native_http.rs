use super::{MIN_TRANSFER_TIME, Result, TimeError, TimeErrorKind, TimeSample};
#[cfg(target_os = "windows")]
use core::error::Error;
use core::fmt::Display;
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
