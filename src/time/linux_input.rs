use crate::write_line_ignored;
use alloc::string::String;
use core::{
    ffi::{CStr, c_char, c_int, c_uint, c_ulong, c_void},
    mem,
    ptr::{NonNull, null},
};
use std::io;
const BUTTON_LEFT: c_uint = 1;
const DL_NOW: c_int = 2;
const KEY_PRESS: c_int = 1;
const KEY_RELEASE: c_int = 0;
const XK_F5: c_ulong = 0xffc2;
const X11_LIBS: [&CStr; 2] = [c"libX11.so.6", c"libX11.so"];
const XTST_LIBS: [&CStr; 2] = [c"libXtst.so.6", c"libXtst.so"];
const SYM_X_CLOSE_DISPLAY: &CStr = c"XCloseDisplay";
const SYM_X_FLUSH: &CStr = c"XFlush";
const SYM_X_KEYS_PRESS: &CStr = c"XKeysymToKeycode";
const SYM_X_OPEN_DISPLAY: &CStr = c"XOpenDisplay";
const SYM_X_TEST_BUTTON: &CStr = c"XTestFakeButtonEvent";
const SYM_X_TEST_KEY: &CStr = c"XTestFakeKeyEvent";
type Display = c_void;
type XCloseDisplay = unsafe extern "C" fn(*mut Display) -> c_int;
type XFlush = unsafe extern "C" fn(*mut Display) -> c_int;
type XKeysymToKeycode = unsafe extern "C" fn(*mut Display, c_ulong) -> c_uint;
type XOpenDisplay = unsafe extern "C" fn(*const c_char) -> *mut Display;
type XTestFakeButtonEvent = unsafe extern "C" fn(*mut Display, c_uint, c_int, c_ulong) -> c_int;
type XTestFakeKeyEvent = unsafe extern "C" fn(*mut Display, c_uint, c_int, c_ulong) -> c_int;
#[derive(Clone, Copy, Debug)]
pub enum InputAction {
    F5Press,
    MouseClick,
}
struct Library {
    handle: NonNull<c_void>,
}
struct X11Api {
    _x11: Library,
    _xtst: Library,
    close_display: XCloseDisplay,
    flush: XFlush,
    keycode: XKeysymToKeycode,
    open_display: XOpenDisplay,
    test_button: XTestFakeButtonEvent,
    test_key: XTestFakeKeyEvent,
}
struct DisplayHandle<'api> {
    api: &'api X11Api,
    display: NonNull<Display>,
}
#[link(name = "dl")]
unsafe extern "C" {
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: handle was returned by dlopen and is closed exactly once here.
        unsafe {
            dlclose(self.handle.as_ptr());
        }
    }
}
impl Library {
    fn open(candidates: &[&CStr]) -> Result<Self, String> {
        let mut last_error = String::new();
        for candidate in candidates {
            // SAFETY: candidate values are static NUL-terminated library names.
            let handle_ptr = unsafe { dlopen(candidate.as_ptr(), DL_NOW) };
            if let Some(handle) = NonNull::new(handle_ptr) {
                return Ok(Self { handle });
            }
            last_error = dl_error_message();
        }
        Err(last_error)
    }
    fn symbol_address(&self, name: &CStr) -> Result<NonNull<c_void>, String> {
        // SAFETY: self.handle is a live dlopen handle and name is NUL-terminated.
        let symbol = unsafe { dlsym(self.handle.as_ptr(), name.as_ptr()) };
        NonNull::new(symbol).ok_or_else(dl_error_message)
    }
    fn x_close_display(&self) -> Result<XCloseDisplay, String> {
        let symbol = self.symbol_address(SYM_X_CLOSE_DISPLAY)?;
        // SAFETY: symbol was resolved from XCloseDisplay with the matching function pointer type.
        Ok(unsafe { mem::transmute::<*mut c_void, XCloseDisplay>(symbol.as_ptr()) })
    }
    fn x_flush(&self) -> Result<XFlush, String> {
        let symbol = self.symbol_address(SYM_X_FLUSH)?;
        // SAFETY: symbol was resolved from XFlush with the matching function pointer type.
        Ok(unsafe { mem::transmute::<*mut c_void, XFlush>(symbol.as_ptr()) })
    }
    fn x_keycode(&self) -> Result<XKeysymToKeycode, String> {
        let symbol = self.symbol_address(SYM_X_KEYS_PRESS)?;
        // SAFETY: symbol was resolved from XKeysymToKeycode with the matching function pointer type.
        Ok(unsafe { mem::transmute::<*mut c_void, XKeysymToKeycode>(symbol.as_ptr()) })
    }
    fn x_open_display(&self) -> Result<XOpenDisplay, String> {
        let symbol = self.symbol_address(SYM_X_OPEN_DISPLAY)?;
        // SAFETY: symbol was resolved from XOpenDisplay with the matching function pointer type.
        Ok(unsafe { mem::transmute::<*mut c_void, XOpenDisplay>(symbol.as_ptr()) })
    }
    fn x_test_button(&self) -> Result<XTestFakeButtonEvent, String> {
        let symbol = self.symbol_address(SYM_X_TEST_BUTTON)?;
        // SAFETY: symbol was resolved from XTestFakeButtonEvent with the matching function pointer type.
        Ok(unsafe { mem::transmute::<*mut c_void, XTestFakeButtonEvent>(symbol.as_ptr()) })
    }
    fn x_test_key(&self) -> Result<XTestFakeKeyEvent, String> {
        let symbol = self.symbol_address(SYM_X_TEST_KEY)?;
        // SAFETY: symbol was resolved from XTestFakeKeyEvent with the matching function pointer type.
        Ok(unsafe { mem::transmute::<*mut c_void, XTestFakeKeyEvent>(symbol.as_ptr()) })
    }
}
impl Drop for DisplayHandle<'_> {
    fn drop(&mut self) {
        // SAFETY: display was returned by XOpenDisplay and is closed exactly once here.
        unsafe {
            (self.api.close_display)(self.display.as_ptr());
        }
    }
}
impl X11Api {
    fn click_left(&self, display: NonNull<Display>) -> Result<(), String> {
        self.fake_button(display, KEY_PRESS)?;
        self.fake_button(display, KEY_RELEASE)
    }
    fn fake_button(&self, display: NonNull<Display>, state: c_int) -> Result<(), String> {
        // SAFETY: display is a live X11 Display and the button/state values match XTest's contract.
        let ok = unsafe { (self.test_button)(display.as_ptr(), BUTTON_LEFT, state, 0) };
        if ok == 0 {
            Err(String::from("XTestFakeButtonEvent 실패"))
        } else {
            Ok(())
        }
    }
    fn fake_key(&self, display: NonNull<Display>, keycode: c_uint, state: c_int) -> Result<(), String> {
        // SAFETY: display is a live X11 Display and keycode/state come from X11/XTest APIs.
        let ok = unsafe { (self.test_key)(display.as_ptr(), keycode, state, 0) };
        if ok == 0 {
            Err(String::from("XTestFakeKeyEvent 실패"))
        } else {
            Ok(())
        }
    }
    fn flush(&self, display: NonNull<Display>) -> Result<(), String> {
        // SAFETY: display is a live X11 Display.
        let ok = unsafe { (self.flush)(display.as_ptr()) };
        if ok == 0 {
            Err(String::from("XFlush 실패"))
        } else {
            Ok(())
        }
    }
    fn open_display(&self) -> Result<DisplayHandle<'_>, String> {
        // SAFETY: null asks XOpenDisplay to read DISPLAY from the process environment.
        let display_ptr = unsafe { (self.open_display)(null()) };
        let Some(display) = NonNull::new(display_ptr) else {
            return Err(String::from(
                "XOpenDisplay 실패: DISPLAY 환경 또는 X11 세션을 확인하세요.",
            ));
        };
        Ok(DisplayHandle { api: self, display })
    }
    fn press_f5(&self, display: NonNull<Display>) -> Result<(), String> {
        // SAFETY: display is a live X11 Display and XK_F5 is a valid keysym constant.
        let keycode = unsafe { (self.keycode)(display.as_ptr(), XK_F5) };
        if keycode == 0 {
            return Err(String::from("F5 keycode 조회 실패"));
        }
        self.fake_key(display, keycode, KEY_PRESS)?;
        self.fake_key(display, keycode, KEY_RELEASE)
    }
}
pub fn send_action(action: InputAction, err: &mut dyn io::Write) {
    let result: Result<(), String> = (|| {
        let x11 =
            Library::open(&X11_LIBS).map_err(|source| format!("libX11 로드 실패: {source}"))?;
        let xtst =
            Library::open(&XTST_LIBS).map_err(|source| format!("libXtst 로드 실패: {source}"))?;
        let close_display = x11.x_close_display()?;
        let flush = x11.x_flush()?;
        let keycode = x11.x_keycode()?;
        let open_display = x11.x_open_display()?;
        let test_button = xtst.x_test_button()?;
        let test_key = xtst.x_test_key()?;
        let api = X11Api {
            _x11: x11,
            _xtst: xtst,
            close_display,
            flush,
            keycode,
            open_display,
            test_button,
            test_key,
        };
        let display = api.open_display()?;
        match action {
            InputAction::MouseClick => api.click_left(display.display)?,
            InputAction::F5Press => api.press_f5(display.display)?,
        }
        api.flush(display.display)?;
        Ok(())
    })();
    if let Err(source) = result {
        write_line_ignored(err, format_args!("[경고] Linux native 입력 실패: {source}"));
    }
}
fn dl_error_message() -> String {
    // SAFETY: dlerror has no preconditions and returns a thread-local C string or null.
    let raw_error = unsafe { dlerror() };
    NonNull::new(raw_error.cast_mut()).map_or_else(
        || String::from("알 수 없는 dynamic loader 오류"),
        |error_ptr| {
            // SAFETY: dlerror returned a non-null NUL-terminated C string.
            unsafe { CStr::from_ptr(error_ptr.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        },
    )
}
