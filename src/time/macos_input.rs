use super::{NativeInputSendStatus, TriggerAction};
use crate::{diagnostic::terminal_safe, write_line_best_effort};
use core::{
    ffi::c_void,
    mem::offset_of,
    ptr::{NonNull, null_mut},
};
use std::io::Write;
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: *const c_void);
}
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGEventCreate(source: CGEventSourceRef) -> CGEventRef;
    fn CGEventCreateKeyboardEvent(
        source: CGEventSourceRef,
        virtual_key: CGKeyCode,
        key_down: bool,
    ) -> CGEventRef;
    fn CGEventCreateMouseEvent(
        source: CGEventSourceRef,
        mouse_type: CGEventType,
        mouse_cursor_position: CGPoint,
        mouse_button: CGMouseButton,
    ) -> CGEventRef;
    fn CGEventGetLocation(event: CGEventRef) -> CGPoint;
    fn CGEventPost(tap: CGEventTapLocation, event: CGEventRef);
    fn CGPreflightPostEventAccess() -> bool;
    fn CGRequestPostEventAccess() -> bool;
}
type CGEventRef = *mut c_void;
type CGEventSourceRef = *mut c_void;
type CGEventTapLocation = u32;
type CGEventType = u32;
type CGKeyCode = u16;
type CGMouseButton = u32;
type InputResult<T> = Result<T, String>;
#[repr(C)]
#[derive(Clone, Copy)]
struct CGPoint {
    x: f64,
    y: f64,
}
const _: () = assert!(
    size_of::<CGPoint>() == 16
        && align_of::<CGPoint>() == 8
        && offset_of!(CGPoint, x) == 0
        && offset_of!(CGPoint, y) == 8,
    "CoreGraphics CGPoint layout mismatch"
);
struct Event(NonNull<c_void>);
impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: self.0 is a CoreFoundation object returned by CGEventCreate* and released once here.
        unsafe {
            CFRelease(self.0.as_ptr());
        }
    }
}
impl Event {
    fn from_raw(raw_ptr: CGEventRef, context: &str) -> InputResult<Self> {
        NonNull::new(raw_ptr)
            .map(Self)
            .ok_or_else(|| format!("{context}: CGEvent 생성 실패"))
    }
    fn post(&self) {
        // SAFETY: self.0 is a live CGEventRef retained by self.
        unsafe {
            CGEventPost(0, self.0.as_ptr());
        }
    }
}
impl TriggerAction {
    pub(super) fn send(self, err: &mut dyn Write) -> NativeInputSendStatus {
        if !post_event_access_granted(false) {
            write_line_best_effort(err, format_args!("[경고] macOS 입력 제어 권한이 없습니다."));
            return NativeInputSendStatus::FailedBeforeSend;
        }
        let result: InputResult<()> = (|| {
            let events = match self {
                Self::LeftClick => {
                    // SAFETY: null asks CoreGraphics to use the default source.
                    let current = Event::from_raw(
                        unsafe { CGEventCreate(null_mut()) },
                        "현재 마우스 위치 조회",
                    )?;
                    // SAFETY: current.0 is a live CGEventRef retained by current.
                    let point = unsafe { CGEventGetLocation(current.0.as_ptr()) };
                    [
                        Event::from_raw(
                            // SAFETY: null uses default source and point came from CoreGraphics.
                            unsafe { CGEventCreateMouseEvent(null_mut(), 1, point, 0) },
                            "마우스 누름",
                        )?,
                        Event::from_raw(
                            // SAFETY: null uses default source and point came from CoreGraphics.
                            unsafe { CGEventCreateMouseEvent(null_mut(), 2, point, 0) },
                            "마우스 뗌",
                        )?,
                    ]
                }
                Self::F5Press => [
                    Event::from_raw(
                        // SAFETY: null uses default source and 96 is the F5 key code.
                        unsafe { CGEventCreateKeyboardEvent(null_mut(), 96, true) },
                        "F5 누름",
                    )?,
                    Event::from_raw(
                        // SAFETY: null uses default source and 96 is the F5 key code.
                        unsafe { CGEventCreateKeyboardEvent(null_mut(), 96, false) },
                        "F5 뗌",
                    )?,
                ],
            };
            for event in events {
                event.post();
            }
            Ok(())
        })();
        match result {
            Ok(()) => NativeInputSendStatus::PartialOrUnknown,
            Err(source) => {
                write_line_best_effort(
                    err,
                    format_args!("[경고] macOS native 입력 실패: {}", terminal_safe(&source)),
                );
                NativeInputSendStatus::FailedBeforeSend
            }
        }
    }
}
pub(super) fn post_event_access_granted(request_if_missing: bool) -> bool {
    // SAFETY: CGPreflightPostEventAccess has no preconditions.
    if unsafe { CGPreflightPostEventAccess() } {
        return true;
    }
    // SAFETY: CGRequestPostEventAccess has no preconditions.
    request_if_missing && unsafe { CGRequestPostEventAccess() }
}
