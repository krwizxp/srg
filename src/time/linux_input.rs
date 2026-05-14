use crate::write_line_ignored;
use alloc::string::String;
use core::{
    ffi::{CStr, c_char, c_int, c_uint, c_ulong, c_void},
    mem,
    num::NonZero,
    ptr::{NonNull, null},
};
use std::io;
macro_rules! load_x11_symbol {
    ($library:expr, $symbol_name:expr, $symbol_type:ty) => {{
        let symbol = $library.symbol_address($symbol_name)?;
        // SAFETY: the requested symbol name matches the concrete X11/XTest function pointer type.
        unsafe { mem::transmute::<*mut c_void, $symbol_type>(symbol.as_ptr()) }
    }};
}
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
type XKeycode = NonZero<c_uint>;
#[derive(Clone, Copy, Debug)]
pub(super) enum InputAction {
    F5Press,
    MouseClick,
}
pub(super) struct PreparedInput {
    prepared: Option<PreparedX11Input>,
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
struct PreparedX11Input {
    api: X11Api,
    display: NonNull<Display>,
    f5_keycode: Option<XKeycode>,
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
        let mut last_error = None;
        for candidate in candidates {
            // SAFETY: candidate values are static NUL-terminated library names.
            let handle_ptr = unsafe { dlopen(candidate.as_ptr(), DL_NOW) };
            if let Some(handle) = NonNull::new(handle_ptr) {
                return Ok(Self { handle });
            }
            last_error = Some(dl_error_message());
        }
        Err(last_error.unwrap_or_else(|| String::from("dynamic loader 후보가 없습니다.")))
    }
    fn symbol_address(&self, name: &CStr) -> Result<NonNull<c_void>, String> {
        // SAFETY: self.handle is a live dlopen handle and name is NUL-terminated.
        let symbol = unsafe { dlsym(self.handle.as_ptr(), name.as_ptr()) };
        NonNull::new(symbol).ok_or_else(dl_error_message)
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
    fn lookup_f5_keycode(&self, display: NonNull<Display>) -> Result<XKeycode, String> {
        // SAFETY: display is a live X11 Display and XK_F5 is a valid keysym constant.
        let keycode = unsafe { (self.keycode)(display.as_ptr(), XK_F5) };
        XKeycode::new(keycode).ok_or_else(|| String::from("F5 keycode 조회 실패"))
    }
    fn open_display(&self) -> Result<NonNull<Display>, String> {
        // SAFETY: null asks XOpenDisplay to read DISPLAY from the process environment.
        let display_ptr = unsafe { (self.open_display)(null()) };
        NonNull::new(display_ptr).ok_or_else(|| {
            String::from("XOpenDisplay 실패: DISPLAY 환경 또는 X11 세션을 확인하세요.")
        })
    }
    fn press_f5_with_keycode(
        &self,
        display: NonNull<Display>,
        keycode: XKeycode,
    ) -> Result<(), String> {
        self.fake_key(display, keycode.get(), KEY_PRESS)?;
        self.fake_key(display, keycode.get(), KEY_RELEASE)
    }
}
impl Drop for PreparedX11Input {
    fn drop(&mut self) {
        // SAFETY: display was returned by XOpenDisplay and is closed exactly once here,
        // while the libX11 handle owned by self.api is still alive.
        unsafe {
            (self.api.close_display)(self.display.as_ptr());
        }
    }
}
impl PreparedX11Input {
    fn f5_keycode(&mut self) -> Result<XKeycode, String> {
        if let Some(keycode) = self.f5_keycode {
            return Ok(keycode);
        }
        let keycode = self.api.lookup_f5_keycode(self.display)?;
        self.f5_keycode = Some(keycode);
        Ok(keycode)
    }
    fn open(preload_action: InputAction) -> Result<Self, String> {
        let x11 = Library::open(&X11_LIBS).map_err(|source| format!("libX11 로드 실패: {source}"))?;
        let xtst =
            Library::open(&XTST_LIBS).map_err(|source| format!("libXtst 로드 실패: {source}"))?;
        let close_display = load_x11_symbol!(x11, SYM_X_CLOSE_DISPLAY, XCloseDisplay);
        let flush = load_x11_symbol!(x11, SYM_X_FLUSH, XFlush);
        let keycode = load_x11_symbol!(x11, SYM_X_KEYS_PRESS, XKeysymToKeycode);
        let open_display = load_x11_symbol!(x11, SYM_X_OPEN_DISPLAY, XOpenDisplay);
        let test_button = load_x11_symbol!(xtst, SYM_X_TEST_BUTTON, XTestFakeButtonEvent);
        let test_key = load_x11_symbol!(xtst, SYM_X_TEST_KEY, XTestFakeKeyEvent);
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
        let mut prepared = Self {
            api,
            display,
            f5_keycode: None,
        };
        if matches!(preload_action, InputAction::F5Press) {
            prepared.f5_keycode = Some(prepared.api.lookup_f5_keycode(prepared.display)?);
        }
        Ok(prepared)
    }
    fn send(&mut self, action: InputAction) -> Result<(), String> {
        match action {
            InputAction::MouseClick => self.api.click_left(self.display)?,
            InputAction::F5Press => {
                let keycode = self.f5_keycode()?;
                self.api.press_f5_with_keycode(self.display, keycode)?;
            }
        }
        self.api.flush(self.display)
    }
}
impl PreparedInput {
    pub(super) const EMPTY: Self = Self { prepared: None };
    pub(super) fn prepare(&mut self, action: Option<InputAction>, err: &mut dyn io::Write) {
        self.prepared = None;
        let Some(input_action) = action else {
            return;
        };
        match PreparedX11Input::open(input_action) {
            Ok(prepared) => self.prepared = Some(prepared),
            Err(source) => write_line_ignored(
                err,
                format_args!("[경고] Linux native 입력 사전 준비 실패: {source}"),
            ),
        }
    }
    pub(super) fn reset(&mut self) {
        self.prepared = None;
    }
    pub(super) fn send(&mut self, action: InputAction, err: &mut dyn io::Write) {
        if let Some(mut prepared) = self.prepared.take() {
            match prepared.send(action) {
                Ok(()) => return,
                Err(source) => write_line_ignored(
                    err,
                    format_args!("[경고] Linux native 입력 사전 준비 경로 실패: {source}"),
                ),
            }
            return;
        }
        match PreparedX11Input::open(action).and_then(|mut prepared| prepared.send(action)) {
            Ok(()) => {}
            Err(source) => write_line_ignored(
                err,
                format_args!("[경고] Linux native 입력 실패: {source}"),
            ),
        }
    }
}
fn dl_error_message() -> String {
    // SAFETY: dlerror has no preconditions and returns a thread-local C string or null.
    let raw_error = unsafe { dlerror() };
    if raw_error.is_null() {
        return String::from("알 수 없는 dynamic loader 오류");
    }
    // SAFETY: dlerror returned a non-null NUL-terminated C string.
    unsafe { CStr::from_ptr(raw_error) }
        .to_string_lossy()
        .into_owned()
}
