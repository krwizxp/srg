use crate::time::TimeError;
use alloc::borrow::Cow;
use core::{
    error::Error,
    fmt::{self, Display, Write as _},
    result::Result as CoreResult,
};
use std::io::Error as IoError;
type BoxError = Box<dyn Error + Send + Sync>;
pub(super) type Result<T> = CoreResult<T, AppError>;
pub(super) struct AppError {
    message: Cow<'static, str>,
    source: Option<BoxError>,
}
struct ControlEscapingWriter<'formatter, 'output>(&'formatter mut fmt::Formatter<'output>);
impl AppError {
    pub(super) fn context(
        context: impl Into<Cow<'static, str>>,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            message: context.into(),
            source: Some(Box::new(source)),
        }
    }
    pub(super) fn message(message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }
}
impl Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", terminal_safe(self.message.as_ref()))?;
        if let Some(source) = self.source.as_ref() {
            f.write_str(": ")?;
            write!(f, "{}", terminal_safe(source))?;
        }
        Ok(())
    }
}
impl fmt::Write for ControlEscapingWriter<'_, '_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for character in s.chars() {
            if character.is_control()
                || matches!(
                    character,
                    '\u{061c}'
                        | '\u{200e}'
                        | '\u{200f}'
                        | '\u{202a}'..='\u{202e}'
                        | '\u{2066}'..='\u{2069}'
                )
            {
                for escaped in character.escape_debug() {
                    self.0.write_char(escaped)?;
                }
            } else {
                self.0.write_char(character)?;
            }
        }
        Ok(())
    }
}
impl fmt::Debug for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}
impl Error for AppError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| -> &(dyn Error + 'static) { source })
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
impl From<TimeError> for AppError {
    fn from(source: TimeError) -> Self {
        Self::context("시간 처리 오류", source)
    }
}
pub(super) const fn terminal_safe<T>(value: &T) -> impl Display + '_
where
    T: Display + ?Sized,
{
    fmt::from_fn(move |formatter| write!(&mut ControlEscapingWriter(formatter), "{value}"))
}
