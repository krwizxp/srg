use crate::write_line_ignored;
use core::{ffi::c_void, ptr::null_mut};
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

struct Event {
    raw: CGEventRef,
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
        if !self.raw.is_null() {
            // SAFETY: raw is a CoreFoundation object returned by CGEventCreate* and released exactly once here.
            unsafe {
                CFRelease(self.raw.cast_const());
            }
        }
    }
}

impl Event {
    fn new(raw: CGEventRef, context: &str) -> Result<Self, String> {
        if raw.is_null() {
            Err(format!("{context}: CGEvent 생성 실패"))
        } else {
            Ok(Self { raw })
        }
    }

    fn post(&self) {
        // SAFETY: raw is a live CGEventRef retained by self.
        unsafe {
            CGEventPost(HID_EVENT_TAP, self.raw);
        }
    }
}

pub fn send_action(action: InputAction, err: &mut dyn Write) {
    let result: Result<(), String> = (|| {
        match action {
            InputAction::MouseClick => {
                // SAFETY: null source asks CoreGraphics to create the event with the default source.
                let current = Event::new(
                    unsafe { CGEventCreate(null_mut()) },
                    "현재 마우스 위치 조회",
                )?;
                // SAFETY: current.raw is a live CGEventRef.
                let point = unsafe { CGEventGetLocation(current.raw) };
                post_event(
                    // SAFETY: null source and current point/button values satisfy CGEventCreateMouseEvent's contract.
                    unsafe {
                        CGEventCreateMouseEvent(
                            null_mut(),
                            EVENT_LEFT_MOUSE_DOWN,
                            point,
                            MOUSE_BUTTON_LEFT,
                        )
                    },
                    "마우스 누름",
                )?;
                post_event(
                    // SAFETY: null source and current point/button values satisfy CGEventCreateMouseEvent's contract.
                    unsafe {
                        CGEventCreateMouseEvent(
                            null_mut(),
                            EVENT_LEFT_MOUSE_UP,
                            point,
                            MOUSE_BUTTON_LEFT,
                        )
                    },
                    "마우스 뗌",
                )?;
            }
            InputAction::F5Press => {
                post_event(
                    // SAFETY: KEY_CODE_F5 is the macOS virtual key code for F5.
                    unsafe { CGEventCreateKeyboardEvent(null_mut(), KEY_CODE_F5, true) },
                    "F5 누름",
                )?;
                post_event(
                    // SAFETY: KEY_CODE_F5 is the macOS virtual key code for F5.
                    unsafe { CGEventCreateKeyboardEvent(null_mut(), KEY_CODE_F5, false) },
                    "F5 뗌",
                )?;
            }
        }
        Ok(())
    })();
    if let Err(source) = result {
        write_line_ignored(err, format_args!("[경고] macOS native 입력 실패: {source}"));
    }
}

fn post_event(raw: CGEventRef, context: &str) -> Result<(), String> {
    let event = Event::new(raw, context)?;
    event.post();
    Ok(())
}
