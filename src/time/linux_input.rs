use crate::write_line_ignored;
use alloc::{borrow::Cow, string::String};
use core::{
    ffi::{CStr, c_char, c_int, c_uint, c_ulong, c_void},
    mem,
    num::NonZero,
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
type XKeycode = NonZero<c_uint>;
type InputError = Cow<'static, str>;
type InputResult<T> = Result<T, InputError>;
#[derive(Clone, Copy, Debug)]
pub(super) enum InputAction {
    F5Press,
    MouseClick,
}
pub(super) struct PreparedInput {
    prepared: Option<PreparedX11Input>,
}
enum PreparedSendError {
    Partial(InputError),
    RetrySafe(InputError),
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
    fn open(candidates: &[&CStr]) -> InputResult<Self> {
        let mut last_error = None;
        for candidate in candidates {
            // SAFETY: candidate values are static NUL-terminated library names.
            let handle_ptr = unsafe { dlopen(candidate.as_ptr(), DL_NOW) };
            if let Some(handle) = NonNull::new(handle_ptr) {
                return Ok(Self { handle });
            }
            last_error = Some(dl_error_message());
        }
        match last_error {
            Some(err) => Err(err),
            None => Err(Cow::Borrowed("dynamic loader 후보가 없습니다.")),
        }
    }
    fn symbol_address(&self, name: &CStr) -> InputResult<NonNull<c_void>> {
        // SAFETY: self.handle is a live dlopen handle and name is NUL-terminated.
        let symbol = unsafe { dlsym(self.handle.as_ptr(), name.as_ptr()) };
        NonNull::new(symbol).ok_or_else(dl_error_message)
    }
}
macro_rules! load_x11_symbol {
    ($library:expr, $symbol_name:expr, $symbol_type:ty) => {{
        let symbol = $library.symbol_address($symbol_name)?;
        // SAFETY: the requested X11/XTest symbol name matches this concrete function pointer type.
        Ok::<$symbol_type, InputError>(unsafe {
            mem::transmute::<*mut c_void, $symbol_type>(symbol.as_ptr())
        })
    }};
}
impl X11Api {
    fn fake_button(&self, display: NonNull<Display>, state: c_int) -> InputResult<()> {
        // SAFETY: display is a live X11 Display and the button/state values match XTest's contract.
        let ok = unsafe { (self.test_button)(display.as_ptr(), BUTTON_LEFT, state, 0) };
        if ok == 0 {
            Err(Cow::Borrowed("XTestFakeButtonEvent 실패"))
        } else {
            Ok(())
        }
    }
    fn fake_key(&self, display: NonNull<Display>, keycode: c_uint, state: c_int) -> InputResult<()> {
        // SAFETY: display is a live X11 Display and keycode/state come from X11/XTest APIs.
        let ok = unsafe { (self.test_key)(display.as_ptr(), keycode, state, 0) };
        if ok == 0 {
            Err(Cow::Borrowed("XTestFakeKeyEvent 실패"))
        } else {
            Ok(())
        }
    }
    fn flush(&self, display: NonNull<Display>) -> InputResult<()> {
        // SAFETY: display is a live X11 Display.
        let ok = unsafe { (self.flush)(display.as_ptr()) };
        if ok == 0 {
            Err(Cow::Borrowed("XFlush 실패"))
        } else {
            Ok(())
        }
    }
    fn lookup_f5_keycode(&self, display: NonNull<Display>) -> InputResult<XKeycode> {
        // SAFETY: display is a live X11 Display and XK_F5 is a valid keysym constant.
        let keycode = unsafe { (self.keycode)(display.as_ptr(), XK_F5) };
        XKeycode::new(keycode).ok_or_else(|| Cow::Borrowed("F5 keycode 조회 실패"))
    }
    fn open_display(&self) -> InputResult<NonNull<Display>> {
        // SAFETY: null asks XOpenDisplay to read DISPLAY from the process environment.
        let display_ptr = unsafe { (self.open_display)(null()) };
        NonNull::new(display_ptr).ok_or_else(|| {
            Cow::Borrowed("XOpenDisplay 실패: DISPLAY 환경 또는 X11 세션을 확인하세요.")
        })
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
    fn f5_keycode(&mut self) -> InputResult<XKeycode> {
        if let Some(keycode) = self.f5_keycode {
            return Ok(keycode);
        }
        let keycode = self.api.lookup_f5_keycode(self.display)?;
        self.f5_keycode = Some(keycode);
        Ok(keycode)
    }
    fn open(preload_action: InputAction) -> InputResult<Self> {
        let x11 = Library::open(&X11_LIBS).map_err(|source| format!("libX11 로드 실패: {source}"))?;
        let xtst =
            Library::open(&XTST_LIBS).map_err(|source| format!("libXtst 로드 실패: {source}"))?;
        let close_display = load_x11_symbol!(x11, SYM_X_CLOSE_DISPLAY, XCloseDisplay)?;
        let flush = load_x11_symbol!(x11, SYM_X_FLUSH, XFlush)?;
        let keycode = load_x11_symbol!(x11, SYM_X_KEYS_PRESS, XKeysymToKeycode)?;
        let open_display = load_x11_symbol!(x11, SYM_X_OPEN_DISPLAY, XOpenDisplay)?;
        let test_button = load_x11_symbol!(xtst, SYM_X_TEST_BUTTON, XTestFakeButtonEvent)?;
        let test_key = load_x11_symbol!(xtst, SYM_X_TEST_KEY, XTestFakeKeyEvent)?;
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
    fn send(&mut self, action: InputAction) -> Result<(), PreparedSendError> {
        match action {
            InputAction::MouseClick => {
                self.api
                    .fake_button(self.display, KEY_PRESS)
                    .map_err(PreparedSendError::RetrySafe)?;
                self.api
                    .fake_button(self.display, KEY_RELEASE)
                    .map_err(PreparedSendError::Partial)?;
            }
            InputAction::F5Press => {
                let keycode = self.f5_keycode().map_err(PreparedSendError::RetrySafe)?;
                self.api
                    .fake_key(self.display, keycode.get(), KEY_PRESS)
                    .map_err(PreparedSendError::RetrySafe)?;
                self.api
                    .fake_key(self.display, keycode.get(), KEY_RELEASE)
                    .map_err(PreparedSendError::Partial)?;
            }
        }
        self.api.flush(self.display).map_err(PreparedSendError::Partial)
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
                Err(PreparedSendError::RetrySafe(prepared_error)) => {
                    match send_fresh(action) {
                        Ok(()) => write_line_ignored(
                            err,
                            format_args!(
                                "[경고] Linux native 입력 사전 준비 경로 실패, 즉시 재시도 완료: {prepared_error}"
                            ),
                        ),
                        Err(fresh_error) => write_line_ignored(
                            err,
                            format_args!(
                                "[경고] Linux native 입력 사전 준비 경로 실패: {prepared_error}; 즉시 재시도 실패: {fresh_error}"
                            ),
                        ),
                    }
                    return;
                }
                Err(PreparedSendError::Partial(prepared_error)) => {
                    write_line_ignored(
                        err,
                        format_args!(
                            "[경고] Linux native 입력 사전 준비 경로 부분 실패, 중복 입력 방지를 위해 재시도 생략: {prepared_error}"
                        ),
                    );
                }
            }
            return;
        }
        match send_fresh(action) {
            Ok(()) => {}
            Err(source) => write_line_ignored(
                err,
                format_args!("[경고] Linux native 입력 실패: {source}"),
            ),
        }
    }
}
fn send_fresh(action: InputAction) -> InputResult<()> {
    PreparedX11Input::open(action).and_then(|mut prepared| {
        prepared.send(action).map_err(|error| match error {
            PreparedSendError::Partial(source) | PreparedSendError::RetrySafe(source) => source,
        })
    })
}
fn dl_error_message() -> InputError {
    // SAFETY: dlerror has no preconditions and returns a thread-local C string or null.
    let raw_error = unsafe { dlerror() };
    if raw_error.is_null() {
        return Cow::Borrowed("알 수 없는 dynamic loader 오류");
    }
    // SAFETY: dlerror returned a non-null NUL-terminated C string.
    unsafe { CStr::from_ptr(raw_error) }
        .to_string_lossy()
        .into_owned()
        .into()
}
