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
        let mut remaining = s;
        let mut consumed = 0;
        for (index, character) in s.char_indices() {
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
                let (plain, escaped_tail) = remaining.split_at(index.strict_sub(consumed));
                if !plain.is_empty() {
                    self.0.write_str(plain)?;
                }
                write!(self.0, "{}", character.escape_debug())?;
                let char_len = character.len_utf8();
                remaining = escaped_tail.split_at(char_len).1;
                consumed = index.strict_add(char_len);
            }
        }
        if remaining.is_empty() {
            Ok(())
        } else {
            self.0.write_str(remaining)
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
