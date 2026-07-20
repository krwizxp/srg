use crate::time::TimeError;
use alloc::borrow::Cow;
use core::{error::Error, fmt, fmt::Display, result::Result as CoreResult};
use std::io::{self, Error as IoError};
type BoxError = Box<dyn Error + Send + Sync>;
pub(super) type Result<T> = CoreResult<T, AppError>;
pub(super) struct AppError {
    message: Cow<'static, str>,
    source: Option<BoxError>,
}
impl AppError {
    pub(super) fn context<M, E>(context: M, source: E) -> Self
    where
        M: Into<Cow<'static, str>>,
        E: Error + Send + Sync + 'static,
    {
        Self {
            message: context.into(),
            source: Some(Box::new(source)),
        }
    }
    pub(super) fn message<M>(message: M) -> Self
    where
        M: Into<Cow<'static, str>>,
    {
        Self {
            message: message.into(),
            source: None,
        }
    }
}
impl Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(source) = self.source.as_ref() {
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
        self.source.as_deref().map(|source| {
            let source_ref: &(dyn Error + 'static) = source;
            source_ref
        })
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
        Self::context("시간 처리 오류", source)
    }
}
pub(super) fn is_unexpected_eof(error: &(dyn Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(source) = current {
        if source
            .downcast_ref::<IoError>()
            .is_some_and(|io_error| io_error.kind() == io::ErrorKind::UnexpectedEof)
        {
            return true;
        }
        current = source.source();
    }
    false
}
