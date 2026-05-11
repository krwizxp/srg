use crate::write_line_ignored;
use alloc::string::String;
use core::{
    ffi::{CStr, c_char, c_int, c_uint, c_ulong, c_void},
    mem,
    ptr::null,
};
use std::io;
const BUTTON_LEFT: c_uint = 1;
const DL_NOW: c_int = 2;
const KEY_PRESS: c_int = 1;
const KEY_RELEASE: c_int = 0;
const XK_F5: c_ulong = 0xffc2;
const X11_LIBS: [&[u8]; 2] = [b"libX11.so.6\0", b"libX11.so\0"];
const XTST_LIBS: [&[u8]; 2] = [b"libXtst.so.6\0", b"libXtst.so\0"];
const SYM_X_CLOSE_DISPLAY: &[u8] = b"XCloseDisplay\0";
const SYM_X_FLUSH: &[u8] = b"XFlush\0";
const SYM_X_KEYS_PRESS: &[u8] = b"XKeysymToKeycode\0";
const SYM_X_OPEN_DISPLAY: &[u8] = b"XOpenDisplay\0";
const SYM_X_TEST_BUTTON: &[u8] = b"XTestFakeButtonEvent\0";
const SYM_X_TEST_KEY: &[u8] = b"XTestFakeKeyEvent\0";
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
    handle: *mut c_void,
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
    display: *mut Display,
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
        if !self.handle.is_null() {
            // SAFETY: handle was returned by dlopen and is closed exactly once here.
            unsafe {
                dlclose(self.handle);
            }
        }
    }
}
impl Library {
    fn open(candidates: &[&[u8]]) -> Result<Self, String> {
        let mut last_error = String::new();
        for candidate in candidates {
            // SAFETY: candidate values are static NUL-terminated library names.
            let handle = unsafe { dlopen(candidate.as_ptr().cast::<c_char>(), DL_NOW) };
            if !handle.is_null() {
                return Ok(Self { handle });
            }
            last_error = dl_error_message();
        }
        Err(last_error)
    }
    fn symbol<T>(&self, name: &[u8]) -> Result<T, String>
    where
        T: Copy,
    {
        // SAFETY: self.handle is a live dlopen handle and name is NUL-terminated.
        let symbol = unsafe { dlsym(self.handle, name.as_ptr().cast::<c_char>()) };
        if symbol.is_null() {
            Err(dl_error_message())
        } else {
            // SAFETY: each caller requests the concrete function pointer type matching the named X11 symbol.
            Ok(unsafe { mem::transmute_copy::<*mut c_void, T>(&symbol) })
        }
    }
}
impl Drop for DisplayHandle<'_> {
    fn drop(&mut self) {
        if !self.display.is_null() {
            // SAFETY: display was returned by XOpenDisplay and is closed exactly once here.
            unsafe {
                (self.api.close_display)(self.display);
            }
        }
    }
}
impl X11Api {
    fn click_left(&self, display: *mut Display) -> Result<(), String> {
        self.fake_button(display, KEY_PRESS)?;
        self.fake_button(display, KEY_RELEASE)
    }
    fn fake_button(&self, display: *mut Display, state: c_int) -> Result<(), String> {
        // SAFETY: display is a live X11 Display and the button/state values match XTest's contract.
        let ok = unsafe { (self.test_button)(display, BUTTON_LEFT, state, 0) };
        if ok == 0 {
            Err(String::from("XTestFakeButtonEvent 실패"))
        } else {
            Ok(())
        }
    }
    fn fake_key(&self, display: *mut Display, keycode: c_uint, state: c_int) -> Result<(), String> {
        // SAFETY: display is a live X11 Display and keycode/state come from X11/XTest APIs.
        let ok = unsafe { (self.test_key)(display, keycode, state, 0) };
        if ok == 0 {
            Err(String::from("XTestFakeKeyEvent 실패"))
        } else {
            Ok(())
        }
    }
    fn flush(&self, display: *mut Display) -> Result<(), String> {
        // SAFETY: display is a live X11 Display.
        let ok = unsafe { (self.flush)(display) };
        if ok == 0 {
            Err(String::from("XFlush 실패"))
        } else {
            Ok(())
        }
    }
    fn open_display(&self) -> Result<DisplayHandle<'_>, String> {
        // SAFETY: null asks XOpenDisplay to read DISPLAY from the process environment.
        let display = unsafe { (self.open_display)(null()) };
        if display.is_null() {
            Err(String::from(
                "XOpenDisplay 실패: DISPLAY 환경 또는 X11 세션을 확인하세요.",
            ))
        } else {
            Ok(DisplayHandle { api: self, display })
        }
    }
    fn press_f5(&self, display: *mut Display) -> Result<(), String> {
        // SAFETY: display is a live X11 Display and XK_F5 is a valid keysym constant.
        let keycode = unsafe { (self.keycode)(display, XK_F5) };
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
        let close_display = x11.symbol(SYM_X_CLOSE_DISPLAY)?;
        let flush = x11.symbol(SYM_X_FLUSH)?;
        let keycode = x11.symbol(SYM_X_KEYS_PRESS)?;
        let open_display = x11.symbol(SYM_X_OPEN_DISPLAY)?;
        let test_button = xtst.symbol(SYM_X_TEST_BUTTON)?;
        let test_key = xtst.symbol(SYM_X_TEST_KEY)?;
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
    let error = unsafe { dlerror() };
    if error.is_null() {
        String::from("알 수 없는 dynamic loader 오류")
    } else {
        // SAFETY: dlerror returned a non-null NUL-terminated C string.
        unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}
