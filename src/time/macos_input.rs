use crate::write_line_best_effort;
use super::NativeInputSendStatus;
use alloc::borrow::Cow;
use core::{
    ffi::c_void,
    mem::{align_of, offset_of, size_of},
    ptr::{NonNull, null_mut},
};
use std::io::Write;
mod sys;
const EVENT_LEFT_MOUSE_DOWN: CGEventType = 1;
const EVENT_LEFT_MOUSE_UP: CGEventType = 2;
const HID_EVENT_TAP: CGEventTapLocation = 0;
const KEY_CODE_F5: CGKeyCode = 96;
const MOUSE_BUTTON_LEFT: CGMouseButton = 0;
type CGEventRef = *mut c_void;
type CGEventSourceRef = *mut c_void;
type CGEventTapLocation = u32;
type CGEventType = u32;
type CGKeyCode = u16;
type CGMouseButton = u32;
type InputError = Cow<'static, str>;
type InputResult<T> = Result<T, InputError>;
#[repr(C)]
#[derive(Clone, Copy)]
struct CGPoint {
    x: f64,
    y: f64,
}
const _: () = assert!(size_of::<CGPoint>() == 16, "CoreGraphics CGPoint size mismatch");
const _: () = assert!(align_of::<CGPoint>() == 8, "CoreGraphics CGPoint align mismatch");
const _: () = assert!(offset_of!(CGPoint, x) == 0, "CoreGraphics CGPoint x offset mismatch");
const _: () = assert!(offset_of!(CGPoint, y) == 8, "CoreGraphics CGPoint y offset mismatch");
#[derive(Clone, Copy)]
pub(super) enum InputAction {
    F5Press,
    MouseClick,
}
struct Event {
    raw: NonNull<c_void>,
}
#[derive(Clone, Copy)]
enum KeyDirection {
    Down,
    Up,
}
impl KeyDirection {
    const fn is_down(self) -> bool {
        matches!(self, Self::Down)
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: raw is a CoreFoundation object returned by CGEventCreate* and released exactly once here.
        unsafe {
            sys::CFRelease(self.raw.as_ptr());
        }
    }
}
impl Event {
    fn from_raw(raw_ptr: CGEventRef, context: &str) -> InputResult<Self> {
        let Some(raw) = NonNull::new(raw_ptr) else {
            return Err(Cow::Owned(format!("{context}: CGEvent 생성 실패")));
        };
        Ok(Self { raw })
    }
    fn keyboard(
        virtual_key: CGKeyCode,
        direction: KeyDirection,
        context: &str,
    ) -> InputResult<Self> {
        // SAFETY: null asks CoreGraphics to use the default source and the key code is a validated constant.
        let raw_ptr =
            unsafe { sys::CGEventCreateKeyboardEvent(null_mut(), virtual_key, direction.is_down()) };
        Self::from_raw(raw_ptr, context)
    }
    fn location(&self) -> CGPoint {
        // SAFETY: raw is a live CGEventRef retained by self.
        unsafe { sys::CGEventGetLocation(self.raw.as_ptr()) }
    }
    fn mouse(
        mouse_type: CGEventType,
        mouse_cursor_position: CGPoint,
        mouse_button: CGMouseButton,
        context: &str,
    ) -> InputResult<Self> {
        // SAFETY: null asks CoreGraphics to use the default source and the point comes from CoreGraphics.
        let raw_ptr = unsafe {
            sys::CGEventCreateMouseEvent(
                null_mut(),
                mouse_type,
                mouse_cursor_position,
                mouse_button,
            )
        };
        Self::from_raw(raw_ptr, context)
    }
    fn post(&self) {
        // SAFETY: raw is a live CGEventRef retained by self.
        unsafe {
            sys::CGEventPost(HID_EVENT_TAP, self.raw.as_ptr());
        }
    }
}
impl InputAction {
    pub(super) fn send(self, err: &mut dyn Write) -> NativeInputSendStatus {
        if !post_event_access_granted(false) {
            write_line_best_effort(
                err,
                format_args!("[경고] macOS 입력 제어 권한이 없습니다."),
            );
            return NativeInputSendStatus::FailedBeforeSend;
        }
        let result: InputResult<()> = (|| {
            match self {
                Self::MouseClick => {
                    // SAFETY: null asks CoreGraphics to use the default source.
                    let current = Event::from_raw(
                        unsafe { sys::CGEventCreate(null_mut()) },
                        "현재 마우스 위치 조회",
                    )?;
                    let point = current.location();
                    let mouse_down = Event::mouse(
                        EVENT_LEFT_MOUSE_DOWN,
                        point,
                        MOUSE_BUTTON_LEFT,
                        "마우스 누름",
                    )?;
                    let mouse_up = Event::mouse(
                        EVENT_LEFT_MOUSE_UP,
                        point,
                        MOUSE_BUTTON_LEFT,
                        "마우스 뗌",
                    )?;
                    mouse_down.post();
                    mouse_up.post();
                }
                Self::F5Press => {
                    let key_down = Event::keyboard(KEY_CODE_F5, KeyDirection::Down, "F5 누름")?;
                    let key_up = Event::keyboard(KEY_CODE_F5, KeyDirection::Up, "F5 뗌")?;
                    key_down.post();
                    key_up.post();
                }
            }
            Ok(())
        })();
        match result {
            Ok(()) => NativeInputSendStatus::PartialOrUnknown,
            Err(source) => {
                write_line_best_effort(
                    err,
                    format_args!("[경고] macOS native 입력 실패: {source}"),
                );
                NativeInputSendStatus::FailedBeforeSend
            }
        }
    }
}
pub(super) fn post_event_access_granted(request_if_missing: bool) -> bool {
    // SAFETY: CGPreflightPostEventAccess has no preconditions.
    if unsafe { sys::CGPreflightPostEventAccess() } {
        return true;
    }
    if !request_if_missing {
        return false;
    }
    // SAFETY: CGRequestPostEventAccess has no preconditions.
    unsafe { sys::CGRequestPostEventAccess() }
}
