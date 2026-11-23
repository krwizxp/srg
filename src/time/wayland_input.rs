use super::{NativeInputSendStatus, TriggerAction};
use crate::{
    diagnostic::terminal_safe,
    stop_input::{POLLERR, POLLHUP, POLLIN, POLLNVAL, PollDescriptorCount, PollFd, poll_stdin},
    write_line_best_effort,
};
use alloc::borrow::Cow;
use core::{
    ffi::{CStr, c_char, c_int, c_short, c_uint, c_void},
    mem::DropGuard,
    ptr::{NonNull, null_mut},
    slice,
    time::Duration,
};
use std::{
    io,
    os::fd::{FromRawFd as _, IntoRawFd as _, OwnedFd},
    time::Instant,
};
#[link(name = "dl")]
unsafe extern "C" {
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *mut c_char;
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
macro_rules! dynamic_api {
    ($api:ident, {$($field:ident: $field_type:ty = $symbol:expr),+ $(,)?}) => {
        struct $api {
            _library: Library,
            $($field: $field_type,)+
        }
        impl TryFrom<Library> for $api {
            type Error = InputError;
            fn try_from(library: Library) -> InputResult<Self> {
                const SYMBOLS: &[&CStr] = &[$($symbol,)+];
                let mut raw = [core::ptr::null_mut(); SYMBOLS.len()];
                library.load_symbols(SYMBOLS, &mut raw)?;
                let mut iter = raw.into_iter();
                $(
                    const {
                        assert!(
                            size_of::<$field_type>() == size_of::<*mut c_void>()
                                && align_of::<$field_type>() == align_of::<*mut c_void>(),
                            "dynamic loader symbol ABI가 함수 포인터와 다릅니다."
                        );
                    }
                    // SAFETY: symbol matches its exact C function pointer ABI and `_library` keeps it loaded.
                    let $field = unsafe {
                        DlsymSymbol::<$field_type> {
                            raw: iter.next().unwrap_or_else(|| std::process::abort()),
                        }
                        .typed
                    };
                )+
                Ok(Self { _library: library, $($field,)+ })
            }
        }
    };
}
macro_rules! opaque_ffi_types {
    ($($name:ident),+ $(,)?) => { $(enum $name {})+ };
}
const EI_DEVICE_CAP_BUTTON: c_int = 1 << 5;
const EI_DEVICE_CAP_KEYBOARD: c_int = 1 << 2;
const EI_DEVICE_CAP_POINTER: c_int = 1;
const POLL_READ_EVENTS: c_short = POLLIN | POLLERR | POLLHUP | POLLNVAL;
opaque_ffi_types!(Ei, EiDevice, EiEvent, EiPing, EiSeat, Oeffis);
type InputError = Cow<'static, str>;
type InputResult<T> = Result<T, InputError>;
#[derive(Clone, Copy)]
struct DeviceState {
    active: bool,
    raw: NonNull<EiDevice>,
}
#[repr(C)]
union DlsymSymbol<F: Copy> {
    raw: *mut c_void,
    typed: F,
}
dynamic_api!(EiApi, {
    configure_name: unsafe extern "C" fn(*mut Ei, *const c_char) = c"ei_configure_name",
    device_button_button: unsafe extern "C" fn(*mut EiDevice, c_uint, bool) = c"ei_device_button_button",
    device_close: unsafe extern "C" fn(*mut EiDevice) = c"ei_device_close",
    device_frame: unsafe extern "C" fn(*mut EiDevice, u64) = c"ei_device_frame",
    device_has_capability: unsafe extern "C" fn(*mut EiDevice, c_int) -> bool = c"ei_device_has_capability",
    device_keyboard_key: unsafe extern "C" fn(*mut EiDevice, c_uint, bool) = c"ei_device_keyboard_key",
    device_ref: unsafe extern "C" fn(*mut EiDevice) -> *mut EiDevice = c"ei_device_ref",
    device_start_emulating: unsafe extern "C" fn(*mut EiDevice, c_uint) = c"ei_device_start_emulating",
    device_stop_emulating: unsafe extern "C" fn(*mut EiDevice) = c"ei_device_stop_emulating",
    device_unref: unsafe extern "C" fn(*mut EiDevice) -> *mut EiDevice = c"ei_device_unref",
    dispatch: unsafe extern "C" fn(*mut Ei) = c"ei_dispatch",
    event_get_device: unsafe extern "C" fn(*mut EiEvent) -> *mut EiDevice = c"ei_event_get_device",
    event_get_seat: unsafe extern "C" fn(*mut EiEvent) -> *mut EiSeat = c"ei_event_get_seat",
    event_get_type: unsafe extern "C" fn(*mut EiEvent) -> c_int = c"ei_event_get_type",
    event_pong_get_ping: unsafe extern "C" fn(*mut EiEvent) -> *mut EiPing = c"ei_event_pong_get_ping",
    event_unref: unsafe extern "C" fn(*mut EiEvent) -> *mut EiEvent = c"ei_event_unref",
    get_event: unsafe extern "C" fn(*mut Ei) -> *mut EiEvent = c"ei_get_event",
    get_fd: unsafe extern "C" fn(*mut Ei) -> c_int = c"ei_get_fd",
    new_ping: unsafe extern "C" fn(*mut Ei) -> *mut EiPing = c"ei_new_ping",
    new_sender: unsafe extern "C" fn(*mut c_void) -> *mut Ei = c"ei_new_sender",
    now: unsafe extern "C" fn(*mut Ei) -> u64 = c"ei_now",
    ping: unsafe extern "C" fn(*mut EiPing) = c"ei_ping",
    ping_unref: unsafe extern "C" fn(*mut EiPing) -> *mut EiPing = c"ei_ping_unref",
    seat_bind_capabilities: unsafe extern "C" fn(*mut EiSeat, ...) = c"ei_seat_bind_capabilities",
    setup_backend_fd: unsafe extern "C" fn(*mut Ei, c_int) -> c_int = c"ei_setup_backend_fd",
    unref: unsafe extern "C" fn(*mut Ei) -> *mut Ei = c"ei_unref",
});
struct Library(NonNull<c_void>);
dynamic_api!(PortalApi, {
    create_session: unsafe extern "C" fn(*mut Oeffis, c_uint) = c"oeffis_create_session",
    dispatch: unsafe extern "C" fn(*mut Oeffis) = c"oeffis_dispatch",
    get_eis_fd: unsafe extern "C" fn(*mut Oeffis) -> c_int = c"oeffis_get_eis_fd",
    get_error_message: unsafe extern "C" fn(*mut Oeffis) -> *const c_char = c"oeffis_get_error_message",
    get_event: unsafe extern "C" fn(*mut Oeffis) -> c_int = c"oeffis_get_event",
    get_fd: unsafe extern "C" fn(*mut Oeffis) -> c_int = c"oeffis_get_fd",
    new: unsafe extern "C" fn(*mut c_void) -> *mut Oeffis = c"oeffis_new",
    unref: unsafe extern "C" fn(*mut Oeffis) -> *mut Oeffis = c"oeffis_unref",
});
pub(super) struct PreparedInput(Option<WaylandInput>);
struct WaylandInput {
    action: TriggerAction,
    device: Option<DeviceState>,
    ei_api: EiApi,
    ei_context: NonNull<Ei>,
    portal_api: PortalApi,
    portal_context: NonNull<Oeffis>,
    sequence: c_uint,
}
impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: self.0 was returned by dlopen and is closed exactly once here.
        unsafe {
            dlclose(self.0.as_ptr());
        }
    }
}
impl Drop for WaylandInput {
    fn drop(&mut self) {
        if let Some(dev) = self.device.take() {
            if dev.active {
                // SAFETY: dev is retained by this session and emulation was started.
                unsafe {
                    (self.ei_api.device_stop_emulating)(dev.raw.as_ptr());
                }
            }
            // SAFETY: dev is retained by this session and is closed once here.
            unsafe {
                (self.ei_api.device_close)(dev.raw.as_ptr());
            }
            // SAFETY: dev is retained by this session and is unreferenced once here.
            unsafe {
                (self.ei_api.device_unref)(dev.raw.as_ptr());
            }
        }
        // SAFETY: ei_context was returned by ei_new_sender and is unreferenced once here.
        unsafe {
            (self.ei_api.unref)(self.ei_context.as_ptr());
        }
        // SAFETY: portal_context was returned by oeffis_new and is unreferenced once here.
        unsafe {
            (self.portal_api.unref)(self.portal_context.as_ptr());
        }
    }
}
impl Library {
    fn load(name: &CStr, label: &str) -> InputResult<Self> {
        // SAFETY: name is NUL-terminated and remains valid for the call.
        NonNull::new(unsafe { dlopen(name.as_ptr(), 2) })
            .map(Self)
            .ok_or_else(|| format!("{label} 로드 실패: {}", dl_error_message()).into())
    }
    #[inline(never)]
    fn load_symbols(&self, names: &[&CStr], out: &mut [*mut c_void]) -> InputResult<()> {
        for (&name, slot) in names.iter().zip(out.iter_mut()) {
            // SAFETY: dlerror clears any previous loader error.
            unsafe {
                dlerror();
            }
            // SAFETY: self.0 is live and name is NUL-terminated.
            let sym = NonNull::new(unsafe { dlsym(self.0.as_ptr(), name.as_ptr()) })
                .ok_or_else(dl_error_message)?;
            *slot = sym.as_ptr();
        }
        Ok(())
    }
}
impl PreparedInput {
    pub(super) const EMPTY: Self = Self(None);
    pub(super) fn maintain(&mut self, err: &mut dyn io::Write) {
        if let Some(prep) = self.0.as_mut()
            && let Err(src) = prep.dispatch(0)
        {
            warn_wayland(err, "Wayland 입력 세션 유지 실패", &src);
            self.0 = None;
        }
    }
    pub(super) fn prepare<F>(
        &mut self,
        action: Option<TriggerAction>,
        err: &mut dyn io::Write,
        mut should_cancel: F,
    ) -> bool
    where
        F: FnMut() -> bool,
    {
        let Some(input_action) = action else {
            return false;
        };
        let result = (|| -> InputResult<Option<WaylandInput>> {
            if should_cancel() {
                return Ok(None);
            }
            let portal_api = PortalApi::try_from(Library::load(c"liboeffis.so.1", "liboeffis")?)?;
            let ei_api = EiApi::try_from(Library::load(c"libei.so.1", "libei")?)?;
            let deadline = Instant::now()
                .checked_add(Duration::from_mins(2))
                .ok_or("Wayland 입력 준비 제한 시간 계산 실패")?;
            // SAFETY: null user data is permitted by oeffis_new.
            let portal_context =
                NonNull::new(unsafe { (portal_api.new)(null_mut()) }).ok_or("oeffis_new 실패")?;
            // SAFETY: null user data is permitted by ei_new_sender.
            let ei_context = NonNull::new(unsafe { (ei_api.new_sender)(null_mut()) })
                .ok_or("ei_new_sender 실패")?;
            let mut prep = WaylandInput {
                action: input_action,
                device: None,
                ei_api,
                ei_context,
                portal_api,
                portal_context,
                sequence: 0,
            };
            prep.portal_poll_fd()?;
            let devices = match input_action {
                TriggerAction::F5Press => 1_u32,
                TriggerAction::LeftClick => 2_u32,
            };
            // SAFETY: portal_context is new and devices is a valid liboeffis device mask.
            unsafe {
                (prep.portal_api.create_session)(prep.portal_context.as_ptr(), devices);
            }
            let Some(portal_eis_fd) = wait_with_deadline(
                deadline,
                "Wayland 입력 권한 준비 시간이 초과되었습니다.",
                &mut should_cancel,
                |timeout, cancel| {
                    let revents = poll_one(
                        prep.portal_poll_fd()?,
                        timeout,
                        "liboeffis poll descriptor가 무효화되었습니다.",
                    )?;
                    if cancel() {
                        return Ok(Some(None));
                    }
                    if revents & POLL_READ_EVENTS != 0 && prep.dispatch_portal()? {
                        // SAFETY: called after OEFFIS_EVENT_CONNECTED_TO_EIS.
                        let raw_fd =
                            unsafe { (prep.portal_api.get_eis_fd)(prep.portal_context.as_ptr()) };
                        let eis_fd = (raw_fd >= 0_i32).then_some(raw_fd).ok_or_else(|| {
                            format!("oeffis_get_eis_fd 실패: {}", io::Error::last_os_error())
                        })?;
                        return Ok(Some(Some(eis_fd)));
                    }
                    check_poll(
                        revents,
                        POLLERR | POLLHUP,
                        "liboeffis poll 연결이 종료되었습니다.",
                    )?;
                    Ok(None)
                },
            )?
            .flatten() else {
                return Ok(None);
            };
            // SAFETY: liboeffis returned a new caller-owned duplicated descriptor.
            let eis_fd = unsafe { OwnedFd::from_raw_fd(portal_eis_fd) };
            // SAFETY: ei_context is new and configure_name must be called before backend setup.
            unsafe {
                (prep.ei_api.configure_name)(prep.ei_context.as_ptr(), c"srg".as_ptr());
            }
            // SAFETY: ei_context is new and libei takes ownership of the descriptor on every result.
            let setup_res = unsafe {
                (prep.ei_api.setup_backend_fd)(prep.ei_context.as_ptr(), eis_fd.into_raw_fd())
            };
            (setup_res >= 0_i32).ok_or_else(|| format!("ei_setup_backend_fd 실패: {setup_res}"))?;
            prep.ei_poll_fd()?;
            let ready = wait_with_deadline(
                deadline,
                "Wayland 가상 입력 장치 준비 시간이 초과되었습니다.",
                &mut should_cancel,
                |timeout, cancel| {
                    prep.dispatch(timeout)?;
                    Ok(if prep.device.is_some_and(|dev| dev.active) {
                        Some(true)
                    } else {
                        cancel().then_some(false)
                    })
                },
            )?;
            Ok((ready == Some(true)).then_some(prep))
        })();
        match result {
            Ok(Some(prep)) => {
                self.0 = Some(prep);
                false
            }
            Ok(None) => true,
            Err(src) => {
                warn_wayland(err, "Wayland 입력 사전 준비 실패", &src);
                false
            }
        }
    }
    pub(super) fn send(&mut self, err: &mut dyn io::Write) -> NativeInputSendStatus {
        let Some(mut prep) = self.0.take() else {
            write_line_best_effort(
                err,
                format_args!("[경고] 준비된 Wayland 입력 세션이 없습니다."),
            );
            return NativeInputSendStatus::FailedBeforeSend;
        };
        let send_result = (|| -> Result<(), (bool, InputError)> {
            let before = |msg: InputError| (false, msg);
            prep.dispatch(0).map_err(before)?;
            let dev = prep
                .device
                .filter(|dev| dev.active)
                .ok_or_else(|| before("활성 libei 입력 장치가 없습니다.".into()))?;
            let deadline = Instant::now()
                .checked_add(Duration::from_secs(2))
                .ok_or_else(|| before("Wayland 입력 전달 제한 시간 계산 실패".into()))?;
            // SAFETY: ei_context is a live libei sender context.
            let raw_ping =
                NonNull::new(unsafe { (prep.ei_api.new_ping)(prep.ei_context.as_ptr()) })
                    .ok_or_else(|| before("ei_new_ping 실패".into()))?;
            let ping_unref = prep.ei_api.ping_unref;
            let ping = DropGuard::new(raw_ping, |owned| {
                // SAFETY: owned was returned by ei_new_ping and is unreferenced once here.
                unsafe {
                    ping_unref(owned.as_ptr());
                }
            });
            let (emit, code) = match prep.action {
                TriggerAction::F5Press => (prep.ei_api.device_keyboard_key, 63_u32),
                TriggerAction::LeftClick => (prep.ei_api.device_button_button, 0x0110_u32),
            };
            for pressed in [true, false] {
                // SAFETY: dev is active and has the capability corresponding to emit.
                unsafe {
                    emit(dev.raw.as_ptr(), code, pressed);
                }
                // SAFETY: ei_context is the live context that owns dev.
                let ts = unsafe { (prep.ei_api.now)(prep.ei_context.as_ptr()) };
                // SAFETY: dev is active and ts comes from its live context.
                unsafe {
                    (prep.ei_api.device_frame)(dev.raw.as_ptr(), ts);
                }
            }
            // SAFETY: ping is a live synchronization object owned by this send operation.
            unsafe {
                (prep.ei_api.ping)(ping.as_ptr());
            }
            let expected_ping = *ping;
            let (mut closed, mut never_cancel) = (false, || false);
            wait_with_deadline(
                deadline,
                "Wayland 입력 전달 확인 시간이 초과되었습니다.",
                &mut never_cancel,
                |timeout, _| {
                    if let Some(pong) = prep.dispatch_ei()? {
                        (pong == expected_ping)
                            .ok_or("예상하지 못한 libei PONG 이벤트가 발생했습니다.")?;
                        return Ok(Some(()));
                    }
                    (!closed).ok_or("libei poll 연결이 종료되었습니다.")?;
                    let revents = poll_one(
                        prep.ei_poll_fd()?,
                        timeout,
                        "libei poll descriptor가 무효화되었습니다.",
                    )?;
                    closed = revents & (POLLERR | POLLHUP) != 0;
                    Ok(None)
                },
            )
            .map(|_| ())
            .map_err(|msg| (true, msg))
        })();
        match send_result {
            Ok(()) => NativeInputSendStatus::Sent,
            Err((false, src)) => {
                warn_wayland(err, "Wayland 입력 실패", &src);
                NativeInputSendStatus::FailedBeforeSend
            }
            Err((true, src)) => {
                warn_wayland(err, "Wayland 입력 전송 상태 불확실", &src);
                NativeInputSendStatus::PartialOrUnknown
            }
        }
    }
}
impl WaylandInput {
    fn dispatch(&mut self, timeout: c_int) -> InputResult<()> {
        let mut fds = [
            make_poll_fd(self.portal_poll_fd()?),
            make_poll_fd(self.ei_poll_fd()?),
        ];
        poll_fds(&mut fds, timeout)?;
        let [portal_poll, ei_poll] = fds;
        check_poll(
            portal_poll.revents,
            POLLNVAL,
            "liboeffis poll descriptor가 무효화되었습니다.",
        )?;
        check_poll(
            ei_poll.revents,
            POLLNVAL,
            "libei poll descriptor가 무효화되었습니다.",
        )?;
        if portal_poll.revents & POLL_READ_EVENTS != 0 {
            (!self.dispatch_portal()?).ok_or("예상하지 못한 추가 EIS 연결이 발생했습니다.")?;
        }
        check_poll(
            portal_poll.revents,
            POLLERR | POLLHUP,
            "liboeffis poll 연결이 종료되었습니다.",
        )?;
        if ei_poll.revents & POLL_READ_EVENTS != 0 {
            self.dispatch_ei()?
                .is_none()
                .ok_or("예상하지 못한 libei PONG 이벤트가 발생했습니다.")?;
        }
        check_poll(
            ei_poll.revents,
            POLLERR | POLLHUP,
            "libei poll 연결이 종료되었습니다.",
        )
    }
    fn dispatch_ei(&mut self) -> InputResult<Option<NonNull<EiPing>>> {
        // SAFETY: ei_context is a live libei sender context.
        unsafe {
            (self.ei_api.dispatch)(self.ei_context.as_ptr());
        }
        let mut pong = None;
        loop {
            // SAFETY: ei_context is live and ei_get_event returns an owned event reference or null.
            let Some(raw_ev) =
                NonNull::new(unsafe { (self.ei_api.get_event)(self.ei_context.as_ptr()) })
            else {
                return Ok(pong);
            };
            let event_unref = self.ei_api.event_unref;
            let event = DropGuard::new(raw_ev, |owned| {
                // SAFETY: owned is owned by this loop iteration and is unreferenced once.
                unsafe {
                    event_unref(owned.as_ptr());
                }
            });
            if let Some(ev_pong) = self.handle_ei_event(*event)?
                && pong.replace(ev_pong).is_some()
            {
                return Err("중복 libei PONG 이벤트가 발생했습니다.".into());
            }
        }
    }
    fn dispatch_portal(&mut self) -> InputResult<bool> {
        // SAFETY: portal_context is a live liboeffis context whose fd signaled readiness.
        unsafe {
            (self.portal_api.dispatch)(self.portal_context.as_ptr());
        }
        let mut connected = false;
        loop {
            // SAFETY: portal_context is live and oeffis_get_event returns an enum value.
            match unsafe { (self.portal_api.get_event)(self.portal_context.as_ptr()) } {
                0 => return Ok(connected),
                1 => {
                    (!connected).ok_or("중복 EIS 연결 이벤트가 발생했습니다.")?;
                    connected = true;
                }
                2 => return Err("Wayland 입력 권한 세션이 종료되었습니다.".into()),
                3 => {
                    // SAFETY: portal_context is live and owns the returned NUL-terminated error string.
                    let raw = unsafe {
                        (self.portal_api.get_error_message)(self.portal_context.as_ptr())
                    };
                    // SAFETY: raw is either null or a valid NUL-terminated C string owned by portal_context.
                    return Err(unsafe {
                        cstr_or_default(raw, "liboeffis 연결이 종료되었습니다.")
                    });
                }
                other => return Err(format!("알 수 없는 liboeffis 이벤트: {other}").into()),
            }
        }
    }
    fn ei_poll_fd(&self) -> InputResult<c_int> {
        // SAFETY: ei_context is a live libei sender context.
        let fd = unsafe { (self.ei_api.get_fd)(self.ei_context.as_ptr()) };
        Ok((fd >= 0_i32).then_some(fd).ok_or("ei_get_fd 실패")?)
    }
    fn handle_ei_event(&mut self, event: NonNull<EiEvent>) -> InputResult<Option<NonNull<EiPing>>> {
        // SAFETY: event is live for this call.
        let ev_type = unsafe { (self.ei_api.event_get_type)(event.as_ptr()) };
        match ev_type {
            5_i32..=8_i32 => {
                // SAFETY: called only for event types 5..=8 that carry an ei_device pointer.
                let dev = NonNull::new(unsafe { (self.ei_api.event_get_device)(event.as_ptr()) })
                    .ok_or("libei device 이벤트에 device가 없습니다.")?;
                match ev_type {
                    5_i32 => {
                        let cap = match self.action {
                            TriggerAction::F5Press => EI_DEVICE_CAP_KEYBOARD,
                            TriggerAction::LeftClick => EI_DEVICE_CAP_BUTTON,
                        };
                        // SAFETY: dev came from EI_EVENT_DEVICE_ADDED and is live for this event.
                        let has_cap =
                            unsafe { (self.ei_api.device_has_capability)(dev.as_ptr(), cap) };
                        if self.device.is_some() || !has_cap {
                            // SAFETY: dev is not retained by this session.
                            unsafe {
                                (self.ei_api.device_close)(dev.as_ptr());
                            }
                            return Ok(None);
                        }
                        // SAFETY: dev came from a live event and is retained for the session.
                        let raw = NonNull::new(unsafe { (self.ei_api.device_ref)(dev.as_ptr()) })
                            .ok_or("libei device 참조에 실패했습니다.")?;
                        self.device = Some(DeviceState { active: false, raw });
                    }
                    6_i32 => {
                        if let Some(cur) = self.device.take_if(|cur| cur.raw == dev) {
                            // SAFETY: cur is the retained reference for the removed device.
                            unsafe {
                                (self.ei_api.device_unref)(cur.raw.as_ptr());
                            }
                        }
                    }
                    7_i32 => {
                        if let Some(cur) = self.device.as_mut()
                            && cur.raw == dev
                        {
                            cur.active = false;
                        }
                    }
                    _ => {
                        if let Some(cur) = self.device.as_mut()
                            && cur.raw == dev
                            && !cur.active
                        {
                            self.sequence = self.sequence.wrapping_add(1);
                            // SAFETY: dev was resumed and is retained by this session.
                            unsafe {
                                (self.ei_api.device_start_emulating)(
                                    cur.raw.as_ptr(),
                                    self.sequence,
                                );
                            }
                            cur.active = true;
                        }
                    }
                }
            }
            2_i32 => return Err("libei 연결이 종료되었습니다.".into()),
            90_i32 => {
                // SAFETY: event is an EI_EVENT_PONG event.
                let ping =
                    NonNull::new(unsafe { (self.ei_api.event_pong_get_ping)(event.as_ptr()) })
                        .ok_or("libei PONG 이벤트에 ping이 없습니다.")?;
                return Ok(Some(ping));
            }
            3_i32 => {
                // SAFETY: event is an EI_EVENT_SEAT_ADDED event.
                let seat = NonNull::new(unsafe { (self.ei_api.event_get_seat)(event.as_ptr()) })
                    .ok_or("libei seat 추가 이벤트에 seat가 없습니다.")?;
                match self.action {
                    TriggerAction::F5Press => {
                        // SAFETY: seat came from EI_EVENT_SEAT_ADDED and variadic list ends with null.
                        unsafe {
                            (self.ei_api.seat_bind_capabilities)(
                                seat.as_ptr(),
                                EI_DEVICE_CAP_KEYBOARD,
                                null_mut::<c_void>(),
                            );
                        }
                    }
                    TriggerAction::LeftClick => {
                        // SAFETY: seat came from EI_EVENT_SEAT_ADDED and variadic list ends with null.
                        unsafe {
                            (self.ei_api.seat_bind_capabilities)(
                                seat.as_ptr(),
                                EI_DEVICE_CAP_POINTER,
                                EI_DEVICE_CAP_BUTTON,
                                null_mut::<c_void>(),
                            );
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(None)
    }
    fn portal_poll_fd(&self) -> InputResult<c_int> {
        // SAFETY: portal_context is a live liboeffis context.
        let fd = unsafe { (self.portal_api.get_fd)(self.portal_context.as_ptr()) };
        Ok((fd >= 0_i32).then_some(fd).ok_or("oeffis_get_fd 실패")?)
    }
}
fn check_poll(revents: c_short, mask: c_short, msg: &'static str) -> InputResult<()> {
    Ok((revents & mask == 0).ok_or(msg)?)
}
unsafe fn cstr_or_default(raw_error: *const c_char, fallback: &'static str) -> InputError {
    if raw_error.is_null() {
        return Cow::Borrowed(fallback);
    }
    // SAFETY: caller guarantees raw_error is a valid NUL-terminated C string.
    unsafe { CStr::from_ptr(raw_error) }
        .to_string_lossy()
        .into_owned()
        .into()
}
fn dl_error_message() -> InputError {
    // SAFETY: dlerror returns a thread-local C string or null.
    let raw_error = unsafe { dlerror() };
    // SAFETY: raw_error is either null or a valid NUL-terminated thread-local C string.
    unsafe { cstr_or_default(raw_error, "알 수 없는 dynamic loader 오류") }
}
const fn make_poll_fd(fd: c_int) -> PollFd {
    PollFd {
        fd,
        events: POLLIN,
        revents: 0,
    }
}
fn poll_fds(fds: &mut [PollFd], timeout: c_int) -> InputResult<()> {
    let count = PollDescriptorCount::try_from(fds.len())
        .map_err(|src| format!("poll descriptor 개수 변환 실패: {src}"))?;
    // SAFETY: fds is a writable pollfd array with the exact element count supplied.
    if unsafe { poll_stdin(fds.as_mut_ptr(), count, timeout) } >= 0_i32 {
        return Ok(());
    }
    let src = io::Error::last_os_error();
    if src.kind() == io::ErrorKind::Interrupted {
        for pfd in fds {
            pfd.revents = 0;
        }
        return Ok(());
    }
    Err(format!("poll 실패: {src}").into())
}
fn poll_one(fd: c_int, timeout: c_int, nval_err: &'static str) -> InputResult<c_short> {
    let mut pfd = make_poll_fd(fd);
    poll_fds(slice::from_mut(&mut pfd), timeout)?;
    check_poll(pfd.revents, POLLNVAL, nval_err)?;
    Ok(pfd.revents)
}
fn wait_with_deadline<T, C, S>(
    deadline: Instant,
    timeout_err: &'static str,
    should_cancel: &mut C,
    mut step: S,
) -> InputResult<Option<T>>
where
    C: FnMut() -> bool,
    S: FnMut(c_int, &mut C) -> InputResult<Option<T>>,
{
    loop {
        if should_cancel() {
            return Ok(None);
        }
        let now = Instant::now();
        (now < deadline).ok_or(timeout_err)?;
        let timeout = c_int::from(
            u16::try_from(
                deadline
                    .saturating_duration_since(now)
                    .as_millis()
                    .clamp(1, 1_000),
            )
            .unwrap_or(1_000),
        );
        if let Some(done) = step(timeout, should_cancel)? {
            return Ok(Some(done));
        }
    }
}
#[cold]
#[inline(never)]
fn warn_wayland(err: &mut dyn io::Write, prefix: &str, source: &str) {
    write_line_best_effort(
        err,
        format_args!("[경고] {prefix}: {}", terminal_safe(source)),
    );
}
