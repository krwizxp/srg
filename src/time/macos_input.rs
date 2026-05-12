use crate::write_line_ignored;
use core::{
    ffi::c_void,
    ptr::{NonNull, null_mut},
};
use std::io::Write;

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

#[repr(C)]
#[derive(Clone, Copy)]
struct CGPoint {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy)]
pub enum InputAction {
    F5Press,
    MouseClick,
}

pub(super) struct PreparedInput {
    active: bool,
}

struct Event {
    raw: NonNull<c_void>,
}

#[derive(Clone, Copy)]
enum EventRequest {
    Current,
    Keyboard {
        key_down: bool,
        virtual_key: CGKeyCode,
    },
    Mouse {
        mouse_button: CGMouseButton,
        mouse_cursor_position: CGPoint,
        mouse_type: CGEventType,
    },
}

#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: *const c_void);
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
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: raw is a CoreFoundation object returned by CGEventCreate* and released exactly once here.
        unsafe {
            CFRelease(self.raw.as_ptr().cast_const());
        }
    }
}

impl Event {
    fn create(request: EventRequest, context: &str) -> Result<Self, String> {
        let raw_ptr = match request {
            // SAFETY: null asks CoreGraphics to use the default source.
            EventRequest::Current => unsafe { CGEventCreate(null_mut()) },
            EventRequest::Keyboard {
                key_down,
                virtual_key,
            } => {
                // SAFETY: null asks CoreGraphics to use the default source and the key code is a validated constant.
                unsafe { CGEventCreateKeyboardEvent(null_mut(), virtual_key, key_down) }
            }
            EventRequest::Mouse {
                mouse_button,
                mouse_cursor_position,
                mouse_type,
            } => {
                // SAFETY: null asks CoreGraphics to use the default source and the point comes from CoreGraphics.
                unsafe {
                    CGEventCreateMouseEvent(
                        null_mut(),
                        mouse_type,
                        mouse_cursor_position,
                        mouse_button,
                    )
                }
            }
        };
        let Some(raw) = NonNull::new(raw_ptr) else {
            return Err(format!("{context}: CGEvent 생성 실패"));
        };
        Ok(Self { raw })
    }
    fn keyboard(virtual_key: CGKeyCode, key_down: bool, context: &str) -> Result<Self, String> {
        Self::create(
            EventRequest::Keyboard {
                key_down,
                virtual_key,
            },
            context,
        )
    }
    fn mouse(
        mouse_type: CGEventType,
        mouse_cursor_position: CGPoint,
        mouse_button: CGMouseButton,
        context: &str,
    ) -> Result<Self, String> {
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
            CGEventPost(HID_EVENT_TAP, self.raw.as_ptr());
        }
    }
}

impl PreparedInput {
    pub(super) const EMPTY: Self = Self { active: false };

    pub(super) fn prepare(&mut self, action: Option<InputAction>, _err: &mut dyn Write) {
        self.active = action.is_some();
    }

    pub(super) const fn reset(&mut self) {
        self.active = false;
    }

    pub(super) fn send(&mut self, action: InputAction, err: &mut dyn Write) {
        self.active = false;
        let result: Result<(), String> = (|| {
        match action {
            InputAction::MouseClick => {
                let current = Event::create(EventRequest::Current, "현재 마우스 위치 조회")?;
                // SAFETY: current.raw is a live CGEventRef.
                let point = unsafe { CGEventGetLocation(current.raw.as_ptr()) };
                Event::mouse(
                    EVENT_LEFT_MOUSE_DOWN,
                    point,
                    MOUSE_BUTTON_LEFT,
                    "마우스 누름",
                )?
                .post();
                Event::mouse(
                    EVENT_LEFT_MOUSE_UP,
                    point,
                    MOUSE_BUTTON_LEFT,
                    "마우스 뗌",
                )?
                .post();
            }
            InputAction::F5Press => {
                Event::keyboard(KEY_CODE_F5, true, "F5 누름")?.post();
                Event::keyboard(KEY_CODE_F5, false, "F5 뗌")?.post();
            }
        }
        Ok(())
    })();
        if let Err(source) = result {
            write_line_ignored(err, format_args!("[경고] macOS native 입력 실패: {source}"));
        }
    }
}
