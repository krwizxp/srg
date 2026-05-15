use alloc::{borrow::Cow, fmt};
use core::result::Result as StdResult;
use std::{io, time::SystemTimeError};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TimeErrorKind {
    HeaderNotFound,
    Io,
    NativeHttp,
    Parse,
    SyncFailed,
    Time,
}
#[derive(Debug)]
pub struct TimeError {
    detail: Cow<'static, str>,
    io_kind: Option<io::ErrorKind>,
    kind: TimeErrorKind,
}
impl TimeError {
    pub(super) fn header_not_found(detail: impl Into<Cow<'static, str>>) -> Self {
        Self::new(TimeErrorKind::HeaderNotFound, detail)
    }
    pub const fn io_kind(&self) -> Option<io::ErrorKind> {
        self.io_kind
    }
    pub(super) fn new(kind: TimeErrorKind, detail: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind,
            detail: detail.into(),
            io_kind: None,
        }
    }
    pub(super) fn parse(detail: impl Into<Cow<'static, str>>) -> Self {
        Self::new(TimeErrorKind::Parse, detail)
    }
}
impl From<io::Error> for TimeError {
    fn from(err: io::Error) -> Self {
        let io_kind = err.kind();
        Self {
            kind: TimeErrorKind::Io,
            detail: owned_detail(err),
            io_kind: Some(io_kind),
        }
    }
}
impl From<SystemTimeError> for TimeError {
    fn from(err: SystemTimeError) -> Self {
        Self {
            kind: TimeErrorKind::Time,
            detail: owned_detail(err),
            io_kind: None,
        }
    }
}
impl fmt::Display for TimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            TimeErrorKind::Io => write!(f, "I/O 오류: {}", self.detail),
            TimeErrorKind::Time => write!(f, "시스템 시간 오류: {}", self.detail),
            TimeErrorKind::Parse => write!(f, "파싱 오류: {}", self.detail),
            TimeErrorKind::HeaderNotFound => write!(f, "{} 헤더를 찾을 수 없음", self.detail),
            TimeErrorKind::NativeHttp => write!(f, "native HTTP 요청 실패: {}", self.detail),
            TimeErrorKind::SyncFailed => write!(f, "서버 시간 확인 실패: {}", self.detail),
        }
    }
}
pub(super) type Result<T> = StdResult<T, TimeError>;
fn owned_detail(err: impl fmt::Display) -> Cow<'static, str> {
    Cow::Owned(err.to_string())
}
