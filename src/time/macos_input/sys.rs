use super::{
    CGEventRef, CGEventSourceRef, CGEventTapLocation, CGEventType, CGKeyCode, CGMouseButton, CGPoint,
    c_void,
};
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    pub(super) fn CFRelease(cf: *const c_void);
}
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
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
    pub(super) fn CGPreflightPostEventAccess() -> bool;
    pub(super) fn CGRequestPostEventAccess() -> bool;
}
