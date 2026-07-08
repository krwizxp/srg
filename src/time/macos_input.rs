use crate::write_line_best_effort;
use super::NativeInputSendStatus;
use alloc::borrow::Cow;
use core::{
    ffi::c_void,
    mem::{align_of, offset_of, size_of},
    ptr::{NonNull, null_mut},
};
use std::io::Write;
mod sys {
    use super::{
        CGEventRef, CGEventSourceRef, CGEventTapLocation, CGEventType, CGKeyCode, CGMouseButton,
        CGPoint, c_void,
    };
    #[link(name = "ApplicationServices", kind = "framework")]
    unsafe extern "C" {
        pub(super) fn CFRelease(cf: *const c_void);
        pub(super) fn CGEventCreate(source: CGEventSourceRef) -> CGEventRef;
        pub(super) fn CGEventCreateKeyboardEvent(
            source: CGEventSourceRef,
            virtual_key: CGKeyCode,
            key_down: bool,
        ) -> CGEventRef;
        pub(super) fn CGEventCreateMouseEvent(
            source: CGEventSourceRef,
            mouse_type: CGEventType,
            mouse_cursor_position: CGPoint,
            mouse_button: CGMouseButton,
        ) -> CGEventRef;
        pub(super) fn CGEventGetLocation(event: CGEventRef) -> CGPoint;
        pub(super) fn CGEventPost(tap: CGEventTapLocation, event: CGEventRef);
    }
}
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
pub(super) struct PreparedInput;
struct Event {
    raw: NonNull<c_void>,
}
#[derive(Clone, Copy)]
enum EventRequest {
    Current,
    Keyboard {
        direction: KeyDirection,
        virtual_key: CGKeyCode,
    },
    Mouse {
        mouse_button: CGMouseButton,
        mouse_cursor_position: CGPoint,
        mouse_type: CGEventType,
    },
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
    fn create(request: EventRequest, context: &str) -> InputResult<Self> {
        let raw_ptr = match request {
            // SAFETY: null asks CoreGraphics to use the default source.
            EventRequest::Current => unsafe { sys::CGEventCreate(null_mut()) },
            EventRequest::Keyboard {
                direction,
                virtual_key,
            } => {
                // SAFETY: null asks CoreGraphics to use the default source and the key code is a validated constant.
                unsafe { sys::CGEventCreateKeyboardEvent(null_mut(), virtual_key, direction.is_down()) }
            }
            EventRequest::Mouse {
                mouse_button,
                mouse_cursor_position,
                mouse_type,
            } => {
                // SAFETY: null asks CoreGraphics to use the default source and the point comes from CoreGraphics.
                unsafe {
                    sys::CGEventCreateMouseEvent(
                        null_mut(),
                        mouse_type,
                        mouse_cursor_position,
                        mouse_button,
                    )
                }
            }
        };
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
        Self::create(
            EventRequest::Keyboard {
                direction,
                virtual_key,
            },
            context,
        )
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
        Self::create(
            EventRequest::Mouse {
                mouse_button,
                mouse_cursor_position,
                mouse_type,
            },
            context,
        )
    }
    fn post(&self) {
        // SAFETY: raw is a live CGEventRef retained by self.
        unsafe {
            sys::CGEventPost(HID_EVENT_TAP, self.raw.as_ptr());
        }
    }
}
impl PreparedInput {
    pub(super) const EMPTY: Self = Self;
    pub(super) fn prepare(&mut self, _action: Option<InputAction>, _err: &mut dyn Write) {
        *self = Self;
    }
    pub(super) const fn reset(&mut self) {
        *self = Self;
    }
    pub(super) fn send(
        &mut self,
        action: InputAction,
        err: &mut dyn Write,
    ) -> NativeInputSendStatus {
        *self = Self;
        let result: InputResult<()> = (|| {
            match action {
                InputAction::MouseClick => {
                    let current = Event::create(EventRequest::Current, "현재 마우스 위치 조회")?;
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
                InputAction::F5Press => {
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
