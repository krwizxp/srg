use crate::write_line_best_effort;
use alloc::borrow::Cow;
use core::{
    ffi::{CStr, c_char, c_int, c_uchar, c_uint, c_ulong, c_void},
    mem::size_of,
    num::NonZero,
    ptr::{NonNull, null},
};
use super::NativeInputSendStatus;
use std::io;
mod sys {
    use super::{c_char, c_int, c_void};
    #[link(name = "dl")]
    unsafe extern "C" {
        pub(super) fn dlclose(handle: *mut c_void) -> c_int;
        pub(super) fn dlerror() -> *const c_char;
        pub(super) fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        pub(super) fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
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
const SYM_X_KEYCODE: &CStr = c"XKeysymToKeycode";
const SYM_X_OPEN_DISPLAY: &CStr = c"XOpenDisplay";
const SYM_X_TEST_BUTTON: &CStr = c"XTestFakeButtonEvent";
const SYM_X_TEST_KEY: &CStr = c"XTestFakeKeyEvent";
type Display = c_void;
type XCloseDisplay = unsafe extern "C" fn(*mut Display) -> c_int;
type XFlush = unsafe extern "C" fn(*mut Display) -> c_int;
type XKeysymToKeycode = unsafe extern "C" fn(*mut Display, c_ulong) -> c_uchar;
type XOpenDisplay = unsafe extern "C" fn(*const c_char) -> *mut Display;
type XTestFakeButtonEvent = unsafe extern "C" fn(*mut Display, c_uint, c_int, c_ulong) -> c_int;
type XTestFakeKeyEvent = unsafe extern "C" fn(*mut Display, c_uint, c_int, c_ulong) -> c_int;
const _: () = assert!(size_of::<XCloseDisplay>() == size_of::<*mut c_void>(), "XCloseDisplay pointer size mismatch");
const _: () = assert!(size_of::<XFlush>() == size_of::<*mut c_void>(), "XFlush pointer size mismatch");
const _: () = assert!(size_of::<XKeysymToKeycode>() == size_of::<*mut c_void>(), "XKeysymToKeycode pointer size mismatch");
const _: () = assert!(size_of::<XOpenDisplay>() == size_of::<*mut c_void>(), "XOpenDisplay pointer size mismatch");
const _: () = assert!(size_of::<XTestFakeButtonEvent>() == size_of::<*mut c_void>(), "XTestFakeButtonEvent pointer size mismatch");
const _: () = assert!(size_of::<XTestFakeKeyEvent>() == size_of::<*mut c_void>(), "XTestFakeKeyEvent pointer size mismatch");
type XKeycode = NonZero<c_uchar>;
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
#[repr(C)]
union DlsymSymbol<F: Copy> {
    raw: *mut c_void,
    typed: F,
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
#[derive(Clone, Copy)]
enum XTestInput {
    Button(c_uint),
    Key(XKeycode),
}
impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: handle was returned by dlopen and is closed exactly once here.
        unsafe {
            sys::dlclose(self.handle.as_ptr());
        }
    }
}
impl Library {
    fn open(candidates: &[&CStr]) -> InputResult<Self> {
        let mut last_error = None;
        for candidate in candidates {
            // SAFETY: candidate values are static NUL-terminated library names.
            let handle_ptr = unsafe { sys::dlopen(candidate.as_ptr(), DL_NOW) };
            if let Some(handle) = NonNull::new(handle_ptr) {
                return Ok(Self { handle });
            }
            last_error = Some(dl_error_message());
        }
        let Some(err) = last_error else {
            return Err(Cow::Borrowed("dynamic loader 후보가 없습니다."));
        };
        Err(err)
    }
    fn symbol_address(&self, name: &CStr) -> InputResult<NonNull<c_void>> {
        // SAFETY: dlerror has no preconditions; this clears any previous loader error.
        unsafe {
            sys::dlerror();
        }
        // SAFETY: self.handle is a live dlopen handle and name is NUL-terminated.
        let symbol = unsafe { sys::dlsym(self.handle.as_ptr(), name.as_ptr()) };
        NonNull::new(symbol).ok_or_else(dl_error_message)
    }
    fn typed_symbol<F: Copy>(&self, name: &CStr) -> InputResult<F> {
        if size_of::<F>() != size_of::<*mut c_void>() {
            return Err(Cow::Borrowed(
                "dynamic loader symbol 크기가 함수 포인터와 다릅니다.",
            ));
        }
        let symbol = self.symbol_address(name)?;
        // SAFETY: the requested symbol name is paired with its concrete C function
        // pointer type, the size is checked above, and X11Api keeps the dlopen
        // handle alive while the function pointer is stored.
        Ok(unsafe {
            DlsymSymbol::<F> {
                raw: symbol.as_ptr(),
            }
            .typed
        })
    }
}
impl X11Api {
    fn fake_button(&self, display: NonNull<Display>, button: c_uint, state: c_int) -> InputResult<()> {
        // SAFETY: display is a live X11 Display and the button/state values match XTest's contract.
        let ok = unsafe { (self.test_button)(display.as_ptr(), button, state, 0) };
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
        XKeycode::new(keycode).ok_or(Cow::Borrowed("F5 keycode 조회 실패"))
    }
    fn open_display(&self) -> InputResult<NonNull<Display>> {
        // SAFETY: null asks XOpenDisplay to read DISPLAY from the process environment.
        let display_ptr = unsafe { (self.open_display)(null()) };
        NonNull::new(display_ptr).ok_or(Cow::Borrowed(
            "XOpenDisplay 실패: DISPLAY 환경 또는 X11 세션을 확인하세요.",
        ))
    }
}
impl XTestInput {
    fn press(self, api: &X11Api, display: NonNull<Display>) -> InputResult<()> {
        self.send(api, display, KEY_PRESS)
    }
    fn release(self, api: &X11Api, display: NonNull<Display>) -> InputResult<()> {
        self.send(api, display, KEY_RELEASE)
    }
    fn send(self, api: &X11Api, display: NonNull<Display>, state: c_int) -> InputResult<()> {
        match self {
            Self::Button(button) => api.fake_button(display, button, state),
            Self::Key(keycode) => {
                let expanded_keycode = NonZero::<c_uint>::from(keycode).get();
                api.fake_key(display, expanded_keycode, state)
            }
        }
    }
}
impl Drop for PreparedX11Input {
    fn drop(&mut self) {
        // SAFETY: display came from XOpenDisplay and closes once here while self.api keeps libX11 alive.
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
        let close_display = x11.typed_symbol::<XCloseDisplay>(SYM_X_CLOSE_DISPLAY)?;
        let flush = x11.typed_symbol::<XFlush>(SYM_X_FLUSH)?;
        let keycode = x11.typed_symbol::<XKeysymToKeycode>(SYM_X_KEYCODE)?;
        let open_display = x11.typed_symbol::<XOpenDisplay>(SYM_X_OPEN_DISPLAY)?;
        let test_button = xtst.typed_symbol::<XTestFakeButtonEvent>(SYM_X_TEST_BUTTON)?;
        let test_key = xtst.typed_symbol::<XTestFakeKeyEvent>(SYM_X_TEST_KEY)?;
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
        let input = match action {
            InputAction::MouseClick => XTestInput::Button(BUTTON_LEFT),
            InputAction::F5Press => {
                XTestInput::Key(self.f5_keycode().map_err(PreparedSendError::RetrySafe)?)
            }
        };
        input
            .press(&self.api, self.display)
            .map_err(PreparedSendError::RetrySafe)?;
        if let Err(release_error) = input.release(&self.api, self.display) {
            let cleanup_error = input
                .release(&self.api, self.display)
                .and_then(|()| self.api.flush(self.display))
                .err();
            let error_message = cleanup_error.map_or_else(
                || Cow::Owned(format!("{release_error}; release 재시도 완료")),
                |retry_error| {
                    Cow::Owned(format!("{release_error}; release 재시도 실패: {retry_error}"))
                },
            );
            return Err(PreparedSendError::Partial(error_message));
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
            Err(source) => write_line_best_effort(
                err,
                format_args!("[경고] Linux native 입력 사전 준비 실패: {source}"),
            ),
        }
    }
    pub(super) fn reset(&mut self) {
        self.prepared = None;
    }
    pub(super) fn send(
        &mut self,
        action: InputAction,
        err: &mut dyn io::Write,
    ) -> NativeInputSendStatus {
        if let Some(mut prepared) = self.prepared.take() {
            match prepared.send(action) {
                Ok(()) => return NativeInputSendStatus::Sent,
                Err(PreparedSendError::RetrySafe(prepared_error)) => {
                    match send_fresh(action) {
                        Ok(()) => write_line_best_effort(
                            err,
                            format_args!(
                                "[경고] Linux native 입력 사전 준비 경로 실패, 즉시 재시도 완료: {prepared_error}"
                            ),
                        ),
                        Err(PreparedSendError::RetrySafe(fresh_error)) => {
                            write_line_best_effort(
                                err,
                                format_args!(
                                    "[경고] Linux native 입력 사전 준비 경로 실패: {prepared_error}; 즉시 재시도 실패: {fresh_error}"
                                ),
                            );
                            return NativeInputSendStatus::FailedBeforeSend;
                        }
                        Err(PreparedSendError::Partial(fresh_error)) => {
                            write_line_best_effort(
                                err,
                                format_args!(
                                    "[경고] Linux native 입력 사전 준비 경로 실패: {prepared_error}; 즉시 재시도 중 부분 실패: {fresh_error}"
                                ),
                            );
                            return NativeInputSendStatus::PartialOrUnknown;
                        }
                    }
                    return NativeInputSendStatus::Sent;
                }
                Err(PreparedSendError::Partial(prepared_error)) => {
                    write_line_best_effort(
                        err,
                        format_args!(
                            "[경고] Linux native 입력 사전 준비 경로 부분 실패, 중복 입력 방지를 위해 재시도 생략: {prepared_error}"
                        ),
                    );
                    return NativeInputSendStatus::PartialOrUnknown;
                }
            }
        }
        match send_fresh(action) {
            Ok(()) => NativeInputSendStatus::Sent,
            Err(PreparedSendError::RetrySafe(source)) => {
                write_line_best_effort(
                    err,
                    format_args!("[경고] Linux native 입력 실패: {source}"),
                );
                NativeInputSendStatus::FailedBeforeSend
            }
            Err(PreparedSendError::Partial(source)) => {
                write_line_best_effort(
                    err,
                    format_args!("[경고] Linux native 입력 부분 실패: {source}"),
                );
                NativeInputSendStatus::PartialOrUnknown
            }
        }
    }
}
fn send_fresh(action: InputAction) -> Result<(), PreparedSendError> {
    let mut prepared = PreparedX11Input::open(action).map_err(PreparedSendError::RetrySafe)?;
    prepared.send(action)
}
fn dl_error_message() -> InputError {
    // SAFETY: dlerror has no preconditions and returns a thread-local C string or null.
    let raw_error = unsafe { sys::dlerror() };
    if raw_error.is_null() {
        return Cow::Borrowed("알 수 없는 dynamic loader 오류");
    }
    // SAFETY: dlerror returned a non-null NUL-terminated C string.
    unsafe { CStr::from_ptr(raw_error) }
        .to_string_lossy()
        .into_owned()
        .into()
}
