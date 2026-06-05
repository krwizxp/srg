use crate::time::TimeError;
use alloc::borrow::Cow;
use core::{error::Error, fmt, fmt::Display, result::Result as CoreResult};
use std::io::{Error as IoError, ErrorKind};
type BoxError = Box<dyn Error + Send + Sync>;
pub type Result<T> = CoreResult<T, AppError>;
pub struct AppError {
    io_kind: Option<ErrorKind>,
    message: Cow<'static, str>,
    source: Option<BoxError>,
    time_source: Option<TimeError>,
}
impl AppError {
    pub fn context<M, E>(context: M, source: E) -> Self
    where
        M: Into<Cow<'static, str>>,
        E: Error + Send + Sync + 'static,
    {
        let source_ref: &dyn Error = &source;
        let io_kind = source_ref.downcast_ref::<IoError>().map(IoError::kind);
        Self {
            io_kind,
            message: context.into(),
            source: Some(Box::new(source)),
            time_source: None,
        }
    }
    pub const fn io_error_kind(&self) -> Option<ErrorKind> {
        self.io_kind
    }
    pub fn message<M>(message: M) -> Self
    where
        M: Into<Cow<'static, str>>,
    {
        Self {
            io_kind: None,
            message: message.into(),
            source: None,
            time_source: None,
        }
    }
    cfg_select! {
        target_arch = "x86_64" => {
            pub fn prepend_context<M>(self, context: M) -> Self
            where
                M: Into<Cow<'static, str>>,
            {
                Self {
                    io_kind: self.io_kind,
                    message: Cow::Owned(format!("{}: {}", context.into(), self.message)),
                    source: self.source,
                    time_source: self.time_source,
                }
            }
        }
        _ => {}
    }
}
impl Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(source) = self.source.as_ref() {
            write!(f, "{}: {source}", self.message)
        } else if let Some(source) = self.time_source.as_ref() {
            write!(f, "{}: {source}", self.message)
        } else {
            f.write_str(self.message.as_ref())
        }
    }
}
impl fmt::Debug for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}
impl Error for AppError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map_or_else(
            || {
                self.time_source.as_ref().map(|source| {
                    let source_ref: &(dyn Error + 'static) = source;
                    source_ref
                })
            },
            |source| {
                let source_ref: &(dyn Error + 'static) = source;
                Some(source_ref)
            },
        )
    }
}
impl From<Cow<'static, str>> for AppError {
    fn from(value: Cow<'static, str>) -> Self {
        Self::message(value)
    }
}
impl From<String> for AppError {
    fn from(value: String) -> Self {
        Self::message(value)
    }
}
impl From<&'static str> for AppError {
    fn from(value: &'static str) -> Self {
        Self::message(value)
    }
}
impl From<IoError> for AppError {
    fn from(source: IoError) -> Self {
        Self::context("I/O 오류", source)
    }
}
impl From<fmt::Error> for AppError {
    fn from(source: fmt::Error) -> Self {
        Self::context("format 오류", source)
    }
}
impl From<TimeError> for AppError {
    fn from(source: TimeError) -> Self {
        let io_kind = source.io_kind();
        Self {
            io_kind,
            message: Cow::Borrowed("시간 처리 오류"),
            source: None,
            time_source: Some(source),
        }
    }
}
