use super::{NativeInputSendStatus, TriggerAction};
use crate::{
    diagnostic::terminal_safe,
    stop_input::{POLLERR, POLLHUP, POLLIN, POLLNVAL, PollDescriptorCount, PollFd, poll_stdin},
    write_line_best_effort,
};
use alloc::borrow::Cow;
use core::{
    ffi::{CStr, c_char, c_int, c_short, c_uint, c_void},
    marker::{PhantomData, PhantomPinned},
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
                    // SAFETY: each symbol name is paired with its exact C function pointer type,
                    // ABI size/alignment are verified above, and `_library` keeps it loaded.
                    let $field = unsafe {
                        DlsymSymbol::<$field_type> {
                            raw: iter.next().unwrap_or_else(|| std::process::abort()),
                        }
                        .typed
                    };
                )+
                Ok(Self {
                    _library: library,
                    $($field,)+
                })
            }
        }
    };
}
macro_rules! opaque_ffi_types {
    ($($name:ident),+ $(,)?) => {
        $(
            #[repr(C)]
            struct $name {
                _data: (),
                _marker: PhantomData<(*mut u8, PhantomPinned)>,
            }
        )+
    };
}
const EI_DEVICE_CAP_BUTTON: c_int = 1 << 5;
const EI_DEVICE_CAP_KEYBOARD: c_int = 1 << 2;
const EI_DEVICE_CAP_POINTER: c_int = 1;
const EI_EVENT_DEVICE_ADDED: c_int = 5;
const EI_EVENT_DEVICE_PAUSED: c_int = 7;
const EI_EVENT_DEVICE_REMOVED: c_int = 6;
const EI_EVENT_DEVICE_RESUMED: c_int = 8;
const EI_EVENT_DISCONNECT: c_int = 2;
const EI_EVENT_PONG: c_int = 90;
const EI_EVENT_SEAT_ADDED: c_int = 3;
const OEFFIS_EVENT_CLOSED: c_int = 2;
const OEFFIS_EVENT_CONNECTED_TO_EIS: c_int = 1;
const OEFFIS_EVENT_DISCONNECTED: c_int = 3;
const OEFFIS_EVENT_NONE: c_int = 0;
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
struct Library {
    handle: NonNull<c_void>,
}
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
struct PortalSession {
    api: PortalApi,
    context: NonNull<Oeffis>,
}
pub(super) struct PreparedInput {
    prepared: Option<WaylandInput>,
}
enum SendError {
    Before(InputError),
    Partial(InputError),
}
struct WaylandInput {
    action: TriggerAction,
    api: EiApi,
    context: NonNull<Ei>,
    device: Option<DeviceState>,
    portal: PortalSession,
    sequence: c_uint,
}
impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: handle was returned by dlopen and is closed exactly once here.
        unsafe {
            dlclose(self.handle.as_ptr());
        }
    }
}
impl Drop for PortalSession {
    fn drop(&mut self) {
        // SAFETY: context was returned by oeffis_new and is unreferenced exactly once here.
        unsafe {
            (self.api.unref)(self.context.as_ptr());
        }
    }
}
impl Drop for WaylandInput {
    fn drop(&mut self) {
        if let Some(device) = self.device.take() {
            if device.active {
                // SAFETY: device is retained by this session and emulation was started after a resume event.
                unsafe {
                    (self.api.device_stop_emulating)(device.raw.as_ptr());
                }
            }
            // SAFETY: device is retained by this session and is closed and unreferenced exactly once here.
            unsafe {
                (self.api.device_close)(device.raw.as_ptr());
            }
            // SAFETY: device is retained by this session and is unreferenced exactly once here.
            unsafe {
                (self.api.device_unref)(device.raw.as_ptr());
            }
        }
        // SAFETY: context was returned by ei_new_sender and is unreferenced exactly once here.
        unsafe {
            (self.api.unref)(self.context.as_ptr());
        }
    }
}
impl Library {
    fn load(name: &CStr, label: &str) -> InputResult<Self> {
        // SAFETY: name is NUL-terminated and remains valid for the call.
        let Some(handle) = NonNull::new(unsafe { dlopen(name.as_ptr(), 2) }) else {
            return Err(format!("{label} 로드 실패: {}", dl_error_message()).into());
        };
        Ok(Self { handle })
    }
    #[inline(never)]
    fn load_symbols(&self, names: &[&CStr], out: &mut [*mut c_void]) -> InputResult<()> {
        for (&name, slot) in names.iter().zip(out.iter_mut()) {
            // SAFETY: dlerror has no preconditions and clears any previous loader error.
            unsafe {
                dlerror();
            }
            // SAFETY: self.handle is live and name is NUL-terminated.
            let symbol = NonNull::new(unsafe { dlsym(self.handle.as_ptr(), name.as_ptr()) })
                .ok_or_else(dl_error_message)?;
            *slot = symbol.as_ptr();
        }
        Ok(())
    }
}
impl PortalSession {
    fn dispatch(&mut self) -> InputResult<bool> {
        // SAFETY: context is a live liboeffis context whose fd signaled readiness.
        unsafe {
            (self.api.dispatch)(self.context.as_ptr());
        }
        let mut connected = false;
        loop {
            // SAFETY: context is live and oeffis_get_event returns an enum value.
            let event = unsafe { (self.api.get_event)(self.context.as_ptr()) };
            match event {
                OEFFIS_EVENT_NONE => return Ok(connected),
                OEFFIS_EVENT_CONNECTED_TO_EIS => {
                    (!connected).ok_or("중복 EIS 연결 이벤트가 발생했습니다.")?;
                    connected = true;
                }
                OEFFIS_EVENT_CLOSED => {
                    return Err("Wayland 입력 권한 세션이 종료되었습니다.".into());
                }
                OEFFIS_EVENT_DISCONNECTED => {
                    // SAFETY: context is live and owns the returned NUL-terminated error string.
                    let raw = unsafe { (self.api.get_error_message)(self.context.as_ptr()) };
                    // SAFETY: raw is either null or a valid NUL-terminated C string owned by context.
                    return Err(unsafe {
                        cstr_or_default(raw, "liboeffis 연결이 종료되었습니다.")
                    });
                }
                _ => return Err(format!("알 수 없는 liboeffis 이벤트: {event}").into()),
            }
        }
    }
    fn poll_fd(&self) -> InputResult<c_int> {
        // SAFETY: context is a live liboeffis context.
        let fd = unsafe { (self.api.get_fd)(self.context.as_ptr()) };
        Ok((fd >= 0).then_some(fd).ok_or("oeffis_get_fd 실패")?)
    }
}
impl PreparedInput {
    pub(super) const EMPTY: Self = Self { prepared: None };
    pub(super) fn maintain(&mut self, err: &mut dyn io::Write) {
        if let Some(prepared) = self.prepared.as_mut()
            && let Err(source) = prepared.dispatch(0)
        {
            warn_wayland(err, "Wayland 입력 세션 유지 실패", &source);
            self.prepared = None;
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
            let mut portal = PortalSession {
                api: portal_api,
                context: portal_context,
            };
            portal.poll_fd()?;
            let portal_devices = match input_action {
                TriggerAction::F5Press => 1_u32,
                TriggerAction::LeftClick => 2_u32,
            };
            // SAFETY: context is new and the requested device mask is defined by liboeffis.
            unsafe {
                (portal.api.create_session)(portal.context.as_ptr(), portal_devices);
            }
            let Some(portal_eis_fd) = wait_with_deadline(
                deadline,
                "Wayland 입력 권한 준비 시간이 초과되었습니다.",
                &mut should_cancel,
                |timeout, cancel| {
                    let mut pfd = make_poll_fd(portal.poll_fd()?);
                    poll_fds(slice::from_mut(&mut pfd), timeout)?;
                    (pfd.revents & POLLNVAL == 0)
                        .ok_or("liboeffis poll descriptor가 무효화되었습니다.")?;
                    if cancel() {
                        return Ok(Some(None));
                    }
                    if pfd.revents & POLL_READ_EVENTS != 0 && portal.dispatch()? {
                        // SAFETY: this is called after OEFFIS_EVENT_CONNECTED_TO_EIS.
                        let fd = unsafe { (portal.api.get_eis_fd)(portal.context.as_ptr()) };
                        let eis_fd = (fd >= 0_i32).then_some(fd).ok_or_else(|| {
                            format!("oeffis_get_eis_fd 실패: {}", io::Error::last_os_error())
                        })?;
                        return Ok(Some(Some(eis_fd)));
                    }
                    (pfd.revents & (POLLERR | POLLHUP) == 0)
                        .ok_or("liboeffis poll 연결이 종료되었습니다.")?;
                    Ok(None)
                },
            )?
            .flatten() else {
                return Ok(None);
            };
            // SAFETY: liboeffis returned a new caller-owned duplicated descriptor.
            let eis_fd = unsafe { OwnedFd::from_raw_fd(portal_eis_fd) };
            // SAFETY: null user data is permitted by ei_new_sender.
            let ei_context = NonNull::new(unsafe { (ei_api.new_sender)(null_mut()) })
                .ok_or("ei_new_sender 실패")?;
            let mut prepared = WaylandInput {
                action: input_action,
                api: ei_api,
                context: ei_context,
                device: None,
                portal,
                sequence: 0,
            };
            // SAFETY: context is new and configure_name must be called before backend setup.
            unsafe {
                (prepared.api.configure_name)(prepared.context.as_ptr(), c"srg".as_ptr());
            }
            // SAFETY: context is new and libei takes ownership of the descriptor on every result.
            let setup_result = unsafe {
                (prepared.api.setup_backend_fd)(prepared.context.as_ptr(), eis_fd.into_raw_fd())
            };
            (setup_result >= 0_i32)
                .ok_or_else(|| format!("ei_setup_backend_fd 실패: {setup_result}"))?;
            prepared.ei_poll_fd()?;
            let ready = wait_with_deadline(
                deadline,
                "Wayland 가상 입력 장치 준비 시간이 초과되었습니다.",
                &mut should_cancel,
                |timeout, cancel| {
                    prepared.dispatch(timeout)?;
                    if prepared.device.is_some_and(|device| device.active) {
                        return Ok(Some(true));
                    }
                    Ok(cancel().then_some(false))
                },
            )?;
            Ok((ready == Some(true)).then_some(prepared))
        })();
        match result {
            Ok(Some(prepared)) => {
                self.prepared = Some(prepared);
                false
            }
            Ok(None) => true,
            Err(source) => {
                warn_wayland(err, "Wayland 입력 사전 준비 실패", &source);
                false
            }
        }
    }
    pub(super) fn send(&mut self, err: &mut dyn io::Write) -> NativeInputSendStatus {
        let Some(mut prepared) = self.prepared.take() else {
            write_line_best_effort(
                err,
                format_args!("[경고] 준비된 Wayland 입력 세션이 없습니다."),
            );
            return NativeInputSendStatus::FailedBeforeSend;
        };
        let send_result = match prepared.dispatch(0) {
            Ok(()) => prepared.send(),
            Err(source) => Err(SendError::Before(source)),
        };
        match send_result {
            Ok(()) => NativeInputSendStatus::Sent,
            Err(SendError::Before(source)) => {
                warn_wayland(err, "Wayland 입력 실패", &source);
                NativeInputSendStatus::FailedBeforeSend
            }
            Err(SendError::Partial(source)) => {
                warn_wayland(err, "Wayland 입력 전송 상태 불확실", &source);
                NativeInputSendStatus::PartialOrUnknown
            }
        }
    }
}
impl WaylandInput {
    fn dispatch(&mut self, timeout: c_int) -> InputResult<()> {
        let mut fds = [
            make_poll_fd(self.portal.poll_fd()?),
            make_poll_fd(self.ei_poll_fd()?),
        ];
        poll_fds(&mut fds, timeout)?;
        let [portal_poll, ei_poll] = fds;
        (portal_poll.revents & POLLNVAL == 0)
            .ok_or("liboeffis poll descriptor가 무효화되었습니다.")?;
        (ei_poll.revents & POLLNVAL == 0).ok_or("libei poll descriptor가 무효화되었습니다.")?;
        if portal_poll.revents & POLL_READ_EVENTS != 0 {
            (!self.portal.dispatch()?).ok_or("예상하지 못한 추가 EIS 연결이 발생했습니다.")?;
        }
        (portal_poll.revents & (POLLERR | POLLHUP) == 0)
            .ok_or("liboeffis poll 연결이 종료되었습니다.")?;
        if ei_poll.revents & POLL_READ_EVENTS != 0 {
            self.dispatch_ei()?
                .is_none()
                .ok_or("예상하지 못한 libei PONG 이벤트가 발생했습니다.")?;
        }
        Ok((ei_poll.revents & (POLLERR | POLLHUP) == 0)
            .ok_or("libei poll 연결이 종료되었습니다.")?)
    }
    fn dispatch_ei(&mut self) -> InputResult<Option<NonNull<EiPing>>> {
        // SAFETY: context is a live libei sender context.
        unsafe {
            (self.api.dispatch)(self.context.as_ptr());
        }
        let mut pong = None;
        loop {
            // SAFETY: context is live and ei_get_event returns an owned event reference or null.
            let Some(raw_event) =
                NonNull::new(unsafe { (self.api.get_event)(self.context.as_ptr()) })
            else {
                return Ok(pong);
            };
            let event_unref = self.api.event_unref;
            let event = DropGuard::new(raw_event, |owned_event| {
                // SAFETY: owned_event is owned by this loop iteration and is unreferenced exactly once.
                unsafe {
                    event_unref(owned_event.as_ptr());
                }
            });
            if let Some(event_pong) = self.handle_ei_event(*event)?
                && pong.replace(event_pong).is_some()
            {
                return Err("중복 libei PONG 이벤트가 발생했습니다.".into());
            }
        }
    }
    fn ei_poll_fd(&self) -> InputResult<c_int> {
        // SAFETY: context is a live libei sender context.
        let fd = unsafe { (self.api.get_fd)(self.context.as_ptr()) };
        Ok((fd >= 0).then_some(fd).ok_or("ei_get_fd 실패")?)
    }
    fn handle_ei_event(&mut self, event: NonNull<EiEvent>) -> InputResult<Option<NonNull<EiPing>>> {
        // SAFETY: event is live for this call.
        let event_type = unsafe { (self.api.event_get_type)(event.as_ptr()) };
        let event_device = || -> InputResult<NonNull<EiDevice>> {
            // SAFETY: the closure is called only for event types that carry an ei_device pointer.
            Ok(
                NonNull::new(unsafe { (self.api.event_get_device)(event.as_ptr()) })
                    .ok_or("libei device 이벤트에 device가 없습니다.")?,
            )
        };
        match event_type {
            EI_EVENT_DEVICE_ADDED => {
                let device = event_device()?;
                let required_capability = match self.action {
                    TriggerAction::F5Press => EI_DEVICE_CAP_KEYBOARD,
                    TriggerAction::LeftClick => EI_DEVICE_CAP_BUTTON,
                };
                // SAFETY: device came from EI_EVENT_DEVICE_ADDED and is live for this event.
                let has_cap = unsafe {
                    (self.api.device_has_capability)(device.as_ptr(), required_capability)
                };
                if self.device.is_some() || !has_cap {
                    // SAFETY: device came from EI_EVENT_DEVICE_ADDED and is not retained by this session.
                    unsafe {
                        (self.api.device_close)(device.as_ptr());
                    }
                    return Ok(None);
                }
                // SAFETY: device came from a live event and is retained for the session.
                let retained = NonNull::new(unsafe { (self.api.device_ref)(device.as_ptr()) })
                    .ok_or("libei device 참조에 실패했습니다.")?;
                self.device = Some(DeviceState {
                    active: false,
                    raw: retained,
                });
            }
            EI_EVENT_DEVICE_PAUSED | EI_EVENT_DEVICE_REMOVED | EI_EVENT_DEVICE_RESUMED => {
                let device = event_device()?;
                match event_type {
                    EI_EVENT_DEVICE_REMOVED => {
                        if let Some(current) = self.device.take_if(|current| current.raw == device)
                        {
                            // SAFETY: current is the retained reference corresponding to the removed device.
                            unsafe {
                                (self.api.device_unref)(current.raw.as_ptr());
                            }
                        }
                    }
                    EI_EVENT_DEVICE_PAUSED => {
                        if let Some(current) = self.device.as_mut()
                            && current.raw == device
                        {
                            current.active = false;
                        }
                    }
                    _ => {
                        if let Some(current) = self.device.as_mut()
                            && current.raw == device
                            && !current.active
                        {
                            self.sequence = self.sequence.wrapping_add(1);
                            // SAFETY: device was resumed and is retained by this session.
                            unsafe {
                                (self.api.device_start_emulating)(
                                    current.raw.as_ptr(),
                                    self.sequence,
                                );
                            }
                            current.active = true;
                        }
                    }
                }
            }
            EI_EVENT_DISCONNECT => return Err("libei 연결이 종료되었습니다.".into()),
            EI_EVENT_PONG => {
                // SAFETY: event is an EI_EVENT_PONG event.
                let ping = NonNull::new(unsafe { (self.api.event_pong_get_ping)(event.as_ptr()) })
                    .ok_or("libei PONG 이벤트에 ping이 없습니다.")?;
                return Ok(Some(ping));
            }
            EI_EVENT_SEAT_ADDED => {
                // SAFETY: event is an EI_EVENT_SEAT_ADDED event.
                let seat = NonNull::new(unsafe { (self.api.event_get_seat)(event.as_ptr()) })
                    .ok_or("libei seat 추가 이벤트에 seat가 없습니다.")?;
                match self.action {
                    TriggerAction::F5Press => {
                        // SAFETY: seat came from an EI_EVENT_SEAT_ADDED event and the variadic capability list ends with zero.
                        unsafe {
                            (self.api.seat_bind_capabilities)(
                                seat.as_ptr(),
                                EI_DEVICE_CAP_KEYBOARD,
                                null_mut::<c_void>(),
                            );
                        }
                    }
                    TriggerAction::LeftClick => {
                        // SAFETY: seat came from an EI_EVENT_SEAT_ADDED event and the variadic capability list ends with zero.
                        unsafe {
                            (self.api.seat_bind_capabilities)(
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
    fn send(&mut self) -> Result<(), SendError> {
        let Some(device) = self.device.filter(|device| device.active) else {
            return Err(SendError::Before("활성 libei 입력 장치가 없습니다.".into()));
        };
        let deadline = Instant::now()
            .checked_add(Duration::from_secs(2))
            .ok_or_else(|| SendError::Before("Wayland 입력 전달 제한 시간 계산 실패".into()))?;
        // SAFETY: context is a live libei sender context.
        let raw_ping = NonNull::new(unsafe { (self.api.new_ping)(self.context.as_ptr()) })
            .ok_or_else(|| SendError::Before("ei_new_ping 실패".into()))?;
        let ping_unref = self.api.ping_unref;
        let ping = DropGuard::new(raw_ping, |owned_ping| {
            // SAFETY: owned_ping was returned by ei_new_ping and is unreferenced exactly once here.
            unsafe {
                ping_unref(owned_ping.as_ptr());
            }
        });
        let (emit, code) = match self.action {
            TriggerAction::F5Press => (self.api.device_keyboard_key, 63_u32),
            TriggerAction::LeftClick => (self.api.device_button_button, 0x0110_u32),
        };
        for pressed in [true, false] {
            // SAFETY: device is active and has the capability corresponding to emit.
            unsafe {
                emit(device.raw.as_ptr(), code, pressed);
            }
            // SAFETY: context is the live context that owns device.
            let timestamp = unsafe { (self.api.now)(self.context.as_ptr()) };
            // SAFETY: device is active and timestamp comes from its live context.
            unsafe {
                (self.api.device_frame)(device.raw.as_ptr(), timestamp);
            }
        }
        // SAFETY: ping is a live synchronization object owned by this send operation.
        unsafe {
            (self.api.ping)(ping.as_ptr());
        }
        let expected_ping = *ping;
        let mut connection_closed = false;
        let mut never_cancel = || false;
        wait_with_deadline(
            deadline,
            "Wayland 입력 전달 확인 시간이 초과되었습니다.",
            &mut never_cancel,
            |timeout, _| {
                if let Some(pong) = self.dispatch_ei()? {
                    (pong == expected_ping)
                        .ok_or("예상하지 못한 libei PONG 이벤트가 발생했습니다.")?;
                    return Ok(Some(()));
                }
                (!connection_closed).ok_or("libei poll 연결이 종료되었습니다.")?;
                let mut pfd = make_poll_fd(self.ei_poll_fd()?);
                poll_fds(slice::from_mut(&mut pfd), timeout)?;
                (pfd.revents & POLLNVAL == 0).ok_or("libei poll descriptor가 무효화되었습니다.")?;
                connection_closed = pfd.revents & (POLLERR | POLLHUP) != 0;
                Ok(None)
            },
        )
        .map(|_| ())
        .map_err(SendError::Partial)
    }
}
const fn make_poll_fd(fd: c_int) -> PollFd {
    PollFd {
        fd,
        events: POLLIN,
        revents: 0,
    }
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
    // SAFETY: dlerror has no preconditions and returns a thread-local C string or null.
    let raw_error = unsafe { dlerror() };
    // SAFETY: raw_error is either null or a valid NUL-terminated thread-local C string.
    unsafe { cstr_or_default(raw_error, "알 수 없는 dynamic loader 오류") }
}
fn poll_fds(fds: &mut [PollFd], timeout: c_int) -> InputResult<()> {
    let descriptor_count = PollDescriptorCount::try_from(fds.len())
        .map_err(|source| format!("poll descriptor 개수 변환 실패: {source}"))?;
    // SAFETY: fds is a writable pollfd array with the exact element count supplied.
    let result = unsafe { poll_stdin(fds.as_mut_ptr(), descriptor_count, timeout) };
    if result >= 0_i32 {
        return Ok(());
    }
    let source = io::Error::last_os_error();
    if source.kind() == io::ErrorKind::Interrupted {
        for poll_fd in fds {
            poll_fd.revents = 0;
        }
        return Ok(());
    }
    Err(format!("poll 실패: {source}").into())
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
        let remaining = deadline
            .saturating_duration_since(now)
            .min(Duration::from_secs(1));
        let timeout = if remaining == Duration::from_secs(1) {
            1_000
        } else {
            remaining.subsec_millis().cast_signed().max(1)
        };
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
