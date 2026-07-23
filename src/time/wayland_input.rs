use super::{NativeInputSendStatus, TriggerAction};
use crate::write_line_best_effort;
use alloc::borrow::Cow;
use core::{
    ffi::{CStr, c_char, c_int, c_short, c_uint, c_ulong, c_void},
    marker::{PhantomData, PhantomPinned},
    mem::{align_of, size_of},
    ptr::{NonNull, null_mut},
    slice,
    time::Duration,
};
use std::{
    io,
    os::fd::{FromRawFd as _, IntoRawFd as _, OwnedFd},
    time::Instant,
};
mod sys;
macro_rules! dynamic_api {
    ($api:ident, {$($field:ident: $field_type:ty = $symbol:expr),+ $(,)?}) => {
        struct $api {
            _library: Library,
            $($field: $field_type,)+
        }
        impl TryFrom<Library> for $api {
            type Error = InputError;
            fn try_from(library: Library) -> InputResult<Self> {
                $(let $field = library.typed_symbol::<$field_type>($symbol)?;)+
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
const BTN_LEFT: c_uint = 0x0110;
const DELIVERY_TIMEOUT: Duration = Duration::from_secs(2);
const DL_NOW: c_int = 2;
const EI_CLIENT_NAME: &CStr = c"srg";
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
const F5_KEY_CODE: c_uint = 63;
const LIBEI: &CStr = c"libei.so.1";
const LIBOEFFIS: &CStr = c"liboeffis.so.1";
const OEFFIS_DEVICE_KEYBOARD: c_uint = 1;
const OEFFIS_DEVICE_POINTER: c_uint = 1 << 1;
const OEFFIS_EVENT_CLOSED: c_int = 2;
const OEFFIS_EVENT_CONNECTED_TO_EIS: c_int = 1;
const OEFFIS_EVENT_DISCONNECTED: c_int = 3;
const OEFFIS_EVENT_NONE: c_int = 0;
const POLL_ERROR_EVENTS: c_short = POLLERR | POLLHUP | POLLNVAL;
const POLL_INTERVAL: Duration = Duration::from_secs(1);
const POLL_READ_EVENTS: c_short = POLLIN | POLL_ERROR_EVENTS;
const POLLERR: c_short = 0x0008;
const POLLHUP: c_short = 0x0010;
const POLLIN: c_short = 0x0001;
const POLLNVAL: c_short = 0x0020;
const PREPARE_TIMEOUT: Duration = Duration::from_mins(2);
opaque_ffi_types!(Ei, EiDevice, EiEvent, EiPing, EiSeat, Oeffis);
#[repr(C)]
#[derive(Clone, Copy)]
struct PollFd {
    fd: c_int,
    events: c_short,
    revents: c_short,
}
const _: () = assert!(size_of::<PollFd>() == 8, "pollfd size mismatch");
const _: () = assert!(align_of::<PollFd>() == 4, "pollfd align mismatch");
type PollDescriptorCount = c_ulong;
type InputError = Cow<'static, str>;
type InputResult<T> = Result<T, InputError>;
#[derive(Clone, Copy)]
struct DeviceState {
    active: bool,
    raw: NonNull<EiDevice>,
}
#[derive(Clone, Copy)]
enum DeviceEventType {
    Added,
    Paused,
    Removed,
    Resumed,
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
struct EiSession {
    action: TriggerAction,
    api: EiApi,
    context: NonNull<Ei>,
    device: Option<DeviceState>,
    sequence: c_uint,
}
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
    ei: EiSession,
    portal: PortalSession,
}
impl Drop for EiSession {
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
impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: handle was returned by dlopen and is closed exactly once here.
        unsafe {
            sys::dlclose(self.handle.as_ptr());
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
impl EiApi {
    fn bind_action(&self, action: TriggerAction, seat: NonNull<EiSeat>) {
        match action {
            TriggerAction::F5Press => {
                // SAFETY: seat came from an EI_EVENT_SEAT_ADDED event and the variadic capability list ends with zero.
                unsafe {
                    (self.seat_bind_capabilities)(
                        seat.as_ptr(),
                        EI_DEVICE_CAP_KEYBOARD,
                        null_mut::<c_void>(),
                    );
                }
            }
            TriggerAction::LeftClick => {
                // SAFETY: seat came from an EI_EVENT_SEAT_ADDED event and the variadic capability list ends with zero.
                unsafe {
                    (self.seat_bind_capabilities)(
                        seat.as_ptr(),
                        EI_DEVICE_CAP_POINTER,
                        EI_DEVICE_CAP_BUTTON,
                        null_mut::<c_void>(),
                    );
                }
            }
        }
    }
    fn button(&self, device: NonNull<EiDevice>, is_press: bool) {
        // SAFETY: device is a retained active device with button capability.
        unsafe {
            (self.device_button_button)(device.as_ptr(), BTN_LEFT, is_press);
        }
    }
    fn frame(&self, context: NonNull<Ei>, device: NonNull<EiDevice>) {
        // SAFETY: context is the live context that owns device.
        let timestamp = unsafe { (self.now)(context.as_ptr()) };
        // SAFETY: device is active and timestamp comes from its live context.
        unsafe {
            (self.device_frame)(device.as_ptr(), timestamp);
        }
    }
    fn keyboard_key(&self, device: NonNull<EiDevice>, is_press: bool) {
        // SAFETY: device is a retained active device with keyboard capability.
        unsafe {
            (self.device_keyboard_key)(device.as_ptr(), F5_KEY_CODE, is_press);
        }
    }
}
impl EiSession {
    fn dispatch(&mut self) -> InputResult<Option<NonNull<EiPing>>> {
        // SAFETY: context is a live libei sender context.
        unsafe {
            (self.api.dispatch)(self.context.as_ptr());
        }
        let mut pong = None;
        loop {
            // SAFETY: context is live and ei_get_event returns an owned event reference or null.
            let event_ptr = unsafe { (self.api.get_event)(self.context.as_ptr()) };
            let Some(event) = NonNull::new(event_ptr) else {
                return Ok(pong);
            };
            let result = self.handle_event(event);
            // SAFETY: event is owned by this loop iteration and is unreferenced exactly once.
            unsafe {
                (self.api.event_unref)(event.as_ptr());
            }
            if let Some(event_pong) = result?
                && pong.replace(event_pong).is_some()
            {
                return Err(Cow::Borrowed("중복 libei PONG 이벤트가 발생했습니다."));
            }
        }
    }
    fn handle_device_event(
        &mut self,
        event_type: DeviceEventType,
        device: NonNull<EiDevice>,
    ) -> InputResult<()> {
        match event_type {
            DeviceEventType::Added => {
                let required_capability = self.action.required_capability();
                // SAFETY: device came from EI_EVENT_DEVICE_ADDED and is live for this event.
                let has_required_capability = unsafe {
                    (self.api.device_has_capability)(device.as_ptr(), required_capability)
                };
                if self.device.is_some() || !has_required_capability {
                    // SAFETY: device came from EI_EVENT_DEVICE_ADDED and is not retained by this session.
                    unsafe {
                        (self.api.device_close)(device.as_ptr());
                    }
                    return Ok(());
                }
                // SAFETY: device came from a live event and is retained for the session.
                let retained_ptr = unsafe { (self.api.device_ref)(device.as_ptr()) };
                let retained = NonNull::new(retained_ptr)
                    .ok_or(Cow::Borrowed("libei device 참조에 실패했습니다."))?;
                self.device = Some(DeviceState {
                    active: false,
                    raw: retained,
                });
                Ok(())
            }
            DeviceEventType::Paused => {
                if let Some(current) = self.device.as_mut()
                    && current.raw == device
                {
                    current.active = false;
                }
                Ok(())
            }
            DeviceEventType::Removed => {
                if let Some(current) = self
                    .device
                    .take_if(|current| current.raw == device)
                {
                    // SAFETY: current is the retained reference corresponding to the removed device.
                    unsafe {
                        (self.api.device_unref)(current.raw.as_ptr());
                    }
                }
                Ok(())
            }
            DeviceEventType::Resumed => {
                if let Some(current) = self.device.as_mut()
                    && current.raw == device
                    && !current.active
                {
                    self.sequence = self.sequence.wrapping_add(1);
                    // SAFETY: device was resumed and is retained by this session.
                    unsafe {
                        (self.api.device_start_emulating)(current.raw.as_ptr(), self.sequence);
                    }
                    current.active = true;
                }
                Ok(())
            }
        }
    }
    fn handle_event(&mut self, event: NonNull<EiEvent>) -> InputResult<Option<NonNull<EiPing>>> {
        // SAFETY: event is live for this call.
        let event_type = unsafe { (self.api.event_get_type)(event.as_ptr()) };
        let device_event_type = match event_type {
            EI_EVENT_DEVICE_ADDED => DeviceEventType::Added,
            EI_EVENT_DEVICE_PAUSED => DeviceEventType::Paused,
            EI_EVENT_DEVICE_REMOVED => DeviceEventType::Removed,
            EI_EVENT_DEVICE_RESUMED => DeviceEventType::Resumed,
            EI_EVENT_DISCONNECT => {
                return Err(Cow::Borrowed("libei 연결이 종료되었습니다."));
            }
            EI_EVENT_PONG => {
                // SAFETY: event is an EI_EVENT_PONG event.
                let ping_ptr = unsafe { (self.api.event_pong_get_ping)(event.as_ptr()) };
                let ping = NonNull::new(ping_ptr)
                    .ok_or(Cow::Borrowed("libei PONG 이벤트에 ping이 없습니다."))?;
                return Ok(Some(ping));
            }
            EI_EVENT_SEAT_ADDED => {
                // SAFETY: event is an EI_EVENT_SEAT_ADDED event.
                let seat_ptr = unsafe { (self.api.event_get_seat)(event.as_ptr()) };
                let seat = NonNull::new(seat_ptr)
                    .ok_or(Cow::Borrowed("libei seat 추가 이벤트에 seat가 없습니다."))?;
                self.api.bind_action(self.action, seat);
                return Ok(None);
            }
            _ => return Ok(None),
        };
        // SAFETY: device event types carry an ei_device pointer.
        let device_ptr = unsafe { (self.api.event_get_device)(event.as_ptr()) };
        let device = NonNull::new(device_ptr)
            .ok_or(Cow::Borrowed("libei device 이벤트에 device가 없습니다."))?;
        self.handle_device_event(device_event_type, device)?;
        Ok(None)
    }
    fn poll_fd(&self) -> InputResult<c_int> {
        // SAFETY: context is a live libei sender context.
        let fd = unsafe { (self.api.get_fd)(self.context.as_ptr()) };
        if fd < 0 {
            Err(Cow::Borrowed("ei_get_fd 실패"))
        } else {
            Ok(fd)
        }
    }
    fn send(&mut self) -> Result<(), SendError> {
        let Some(device) = self.device.filter(|device| device.active) else {
            return Err(SendError::Before(Cow::Borrowed(
                "활성 libei 입력 장치가 없습니다.",
            )));
        };
        let deadline = Instant::now()
            .checked_add(DELIVERY_TIMEOUT)
            .ok_or(SendError::Before(Cow::Borrowed(
                "Wayland 입력 전달 제한 시간 계산 실패",
            )))?;
        // SAFETY: context is a live libei sender context.
        let ping_ptr = unsafe { (self.api.new_ping)(self.context.as_ptr()) };
        let ping = NonNull::new(ping_ptr).ok_or(SendError::Before(Cow::Borrowed(
            "ei_new_ping 실패",
        )))?;
        match self.action {
            TriggerAction::F5Press => {
                self.api.keyboard_key(device.raw, true);
                self.api.frame(self.context, device.raw);
                self.api.keyboard_key(device.raw, false);
                self.api.frame(self.context, device.raw);
            }
            TriggerAction::LeftClick => {
                self.api.button(device.raw, true);
                self.api.frame(self.context, device.raw);
                self.api.button(device.raw, false);
                self.api.frame(self.context, device.raw);
            }
        }
        // SAFETY: ping is a live synchronization object owned by this send operation.
        unsafe {
            (self.api.ping)(ping.as_ptr());
        }
        let result = self.wait_for_delivery(ping, deadline);
        // SAFETY: ping was returned by ei_new_ping and is unreferenced exactly once here.
        unsafe {
            (self.api.ping_unref)(ping.as_ptr());
        }
        result.map_err(SendError::Partial)
    }
    fn wait_for_delivery(
        &mut self,
        expected_ping: NonNull<EiPing>,
        deadline: Instant,
    ) -> InputResult<()> {
        loop {
            if let Some(pong) = self.dispatch()? {
                if pong == expected_ping {
                    return Ok(());
                }
                return Err(Cow::Borrowed("예상하지 못한 libei PONG 이벤트가 발생했습니다."));
            }
            if Instant::now() >= deadline {
                return Err(Cow::Borrowed("Wayland 입력 전달 확인 시간이 초과되었습니다."));
            }
            let mut poll_fd = PollFd::new(self.poll_fd()?);
            poll_fds(
                slice::from_mut(&mut poll_fd),
                poll_timeout_until(deadline)?,
            )?;
            if poll_fd.is_invalid() {
                return Err(Cow::Borrowed("libei poll descriptor가 무효화되었습니다."));
            }
            if poll_fd.has_terminal_error() {
                if let Some(pong) = self.dispatch()? {
                    if pong == expected_ping {
                        return Ok(());
                    }
                    return Err(Cow::Borrowed(
                        "예상하지 못한 libei PONG 이벤트가 발생했습니다.",
                    ));
                }
                return Err(Cow::Borrowed("libei poll 연결이 종료되었습니다."));
            }
        }
    }
}
impl TriggerAction {
    const fn portal_devices(self) -> c_uint {
        match self {
            Self::F5Press => OEFFIS_DEVICE_KEYBOARD,
            Self::LeftClick => OEFFIS_DEVICE_POINTER,
        }
    }
    const fn required_capability(self) -> c_int {
        match self {
            Self::F5Press => EI_DEVICE_CAP_KEYBOARD,
            Self::LeftClick => EI_DEVICE_CAP_BUTTON,
        }
    }
}
impl Library {
    fn load(name: &CStr, label: &str) -> InputResult<Self> {
        // SAFETY: name is NUL-terminated and remains valid for the call.
        let handle_ptr = unsafe { sys::dlopen(name.as_ptr(), DL_NOW) };
        let Some(handle) = NonNull::new(handle_ptr) else {
            let source = dl_error_message();
            return Err(format!("{label} 로드 실패: {source}").into());
        };
        Ok(Self { handle })
    }
    fn typed_symbol<F>(&self, name: &CStr) -> InputResult<F>
    where
        F: Copy,
    {
        if size_of::<F>() != size_of::<*mut c_void>()
            || align_of::<F>() != align_of::<*mut c_void>()
        {
            return Err(Cow::Borrowed(
                "dynamic loader symbol ABI가 함수 포인터와 다릅니다.",
            ));
        }
        // SAFETY: dlerror has no preconditions and clears any previous loader error.
        unsafe {
            sys::dlerror();
        }
        // SAFETY: self.handle is live and name is NUL-terminated.
        let symbol_ptr = unsafe { sys::dlsym(self.handle.as_ptr(), name.as_ptr()) };
        let symbol = NonNull::new(symbol_ptr).ok_or_else(dl_error_message)?;
        // SAFETY: each symbol name is paired with its exact C function pointer type, ABI size and
        // alignment are checked above, and the owning API keeps this library loaded.
        Ok(unsafe {
            DlsymSymbol::<F> {
                raw: symbol.as_ptr(),
            }
            .typed
        })
    }
}
impl PollFd {
    const fn has_events(self) -> bool {
        self.revents & POLL_READ_EVENTS != 0
    }
    const fn has_terminal_error(self) -> bool {
        self.revents & (POLLERR | POLLHUP) != 0
    }
    const fn is_invalid(self) -> bool {
        self.revents & POLLNVAL != 0
    }
    const fn new(fd: c_int) -> Self {
        Self {
            fd,
            events: POLLIN,
            revents: 0,
        }
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
                    if connected {
                        return Err(Cow::Borrowed("중복 EIS 연결 이벤트가 발생했습니다."));
                    }
                    connected = true;
                }
                OEFFIS_EVENT_CLOSED => {
                    return Err(Cow::Borrowed("Wayland 입력 권한 세션이 종료되었습니다."));
                }
                OEFFIS_EVENT_DISCONNECTED => return Err(self.error_message()),
                _ => return Err(format!("알 수 없는 liboeffis 이벤트: {event}").into()),
            }
        }
    }
    fn eis_fd(&self) -> InputResult<c_int> {
        // SAFETY: this is called after OEFFIS_EVENT_CONNECTED_TO_EIS.
        let fd = unsafe { (self.api.get_eis_fd)(self.context.as_ptr()) };
        if fd < 0 {
            Err(format!("oeffis_get_eis_fd 실패: {}", io::Error::last_os_error()).into())
        } else {
            Ok(fd)
        }
    }
    fn error_message(&self) -> InputError {
        // SAFETY: context is live and owns the returned NUL-terminated error string.
        let raw_error = unsafe { (self.api.get_error_message)(self.context.as_ptr()) };
        if raw_error.is_null() {
            return Cow::Borrowed("liboeffis 연결이 종료되었습니다.");
        }
        // SAFETY: liboeffis returned a non-null NUL-terminated string owned by the context.
        unsafe { CStr::from_ptr(raw_error) }
            .to_string_lossy()
            .into_owned()
            .into()
    }
    fn poll_fd(&self) -> InputResult<c_int> {
        // SAFETY: context is a live liboeffis context.
        let fd = unsafe { (self.api.get_fd)(self.context.as_ptr()) };
        if fd < 0 {
            Err(Cow::Borrowed("oeffis_get_fd 실패"))
        } else {
            Ok(fd)
        }
    }
    fn wait_for_eis_fd<F>(
        &mut self,
        deadline: Instant,
        should_cancel: &mut F,
    ) -> InputResult<Option<c_int>>
    where
        F: FnMut() -> bool,
    {
        loop {
            if should_cancel() {
                return Ok(None);
            }
            if Instant::now() >= deadline {
                return Err(Cow::Borrowed("Wayland 입력 권한 준비 시간이 초과되었습니다."));
            }
            let mut poll_fd = PollFd::new(self.poll_fd()?);
            poll_fds(
                slice::from_mut(&mut poll_fd),
                poll_timeout_until(deadline)?,
            )?;
            if poll_fd.is_invalid() {
                return Err(Cow::Borrowed("liboeffis poll descriptor가 무효화되었습니다."));
            }
            if should_cancel() {
                return Ok(None);
            }
            if poll_fd.has_events() {
                let connected = self.dispatch()?;
                if poll_fd.has_terminal_error() {
                    return Err(Cow::Borrowed("liboeffis poll 연결이 종료되었습니다."));
                }
                if connected {
                    return self.eis_fd().map(Some);
                }
            }
        }
    }
}
impl PreparedInput {
    pub(super) const EMPTY: Self = Self { prepared: None };
    pub(super) fn maintain(&mut self, err: &mut dyn io::Write) {
        if let Some(prepared) = self.prepared.as_mut()
            && let Err(source) = prepared.dispatch(0)
        {
            write_line_best_effort(
                err,
                format_args!("[경고] Wayland 입력 세션 유지 실패: {source}"),
            );
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
        write_line_best_effort(err, format_args!("[안내] Wayland 입력 권한 승인을 기다립니다."));
        let result = (|| -> InputResult<Option<WaylandInput>> {
            if should_cancel() {
                return Ok(None);
            }
            let portal_api = PortalApi::try_from(Library::load(LIBOEFFIS, "liboeffis")?)?;
            let ei_api = EiApi::try_from(Library::load(LIBEI, "libei")?)?;
            let deadline = Instant::now()
                .checked_add(PREPARE_TIMEOUT)
                .ok_or(Cow::Borrowed("Wayland 입력 준비 제한 시간 계산 실패"))?;
            // SAFETY: null user data is permitted by oeffis_new.
            let portal_context_ptr = unsafe { (portal_api.new)(null_mut()) };
            let portal_context = NonNull::new(portal_context_ptr)
                .ok_or(Cow::Borrowed("oeffis_new 실패"))?;
            let mut portal = PortalSession {
                api: portal_api,
                context: portal_context,
            };
            portal.poll_fd()?;
            // SAFETY: context is new and the requested device mask is defined by liboeffis.
            unsafe {
                (portal.api.create_session)(
                    portal.context.as_ptr(),
                    input_action.portal_devices(),
                );
            }
            let Some(portal_eis_fd) = portal.wait_for_eis_fd(deadline, &mut should_cancel)? else {
                return Ok(None);
            };
            // SAFETY: liboeffis returned a new caller-owned duplicated descriptor.
            let eis_fd = unsafe { OwnedFd::from_raw_fd(portal_eis_fd) };
            // SAFETY: null user data is permitted by ei_new_sender.
            let ei_context_ptr = unsafe { (ei_api.new_sender)(null_mut()) };
            let Some(ei_context) = NonNull::new(ei_context_ptr) else {
                return Err(Cow::Borrowed("ei_new_sender 실패"));
            };
            let ei = EiSession {
                action: input_action,
                api: ei_api,
                context: ei_context,
                device: None,
                sequence: 0,
            };
            // SAFETY: context is new and configure_name must be called before backend setup.
            unsafe {
                (ei.api.configure_name)(ei.context.as_ptr(), EI_CLIENT_NAME.as_ptr());
            }
            // SAFETY: context is new and libei takes ownership of the descriptor on every result.
            let setup_result = unsafe {
                (ei.api.setup_backend_fd)(ei.context.as_ptr(), eis_fd.into_raw_fd())
            };
            if setup_result < 0_i32 {
                return Err(format!("ei_setup_backend_fd 실패: {setup_result}").into());
            }
            ei.poll_fd()?;
            let mut prepared = WaylandInput { ei, portal };
            if !prepared.wait_until_ready(deadline, &mut should_cancel)? {
                return Ok(None);
            }
            Ok(Some(prepared))
        })();
        match result {
            Ok(Some(prepared)) => {
                self.prepared = Some(prepared);
                false
            }
            Ok(None) => true,
            Err(source) => {
                write_line_best_effort(
                    err,
                    format_args!("[경고] Wayland 입력 사전 준비 실패: {source}"),
                );
                false
            }
        }
    }
    pub(super) fn send(
        &mut self,
        err: &mut dyn io::Write,
    ) -> NativeInputSendStatus {
        let Some(mut prepared) = self.prepared.take() else {
            write_line_best_effort(err, format_args!("[경고] 준비된 Wayland 입력 세션이 없습니다."));
            return NativeInputSendStatus::FailedBeforeSend;
        };
        let send_result = match prepared.dispatch(0) {
            Ok(()) => prepared.ei.send(),
            Err(source) => Err(SendError::Before(source)),
        };
        match send_result {
            Ok(()) => NativeInputSendStatus::Sent,
            Err(SendError::Before(source)) => {
                write_line_best_effort(err, format_args!("[경고] Wayland 입력 실패: {source}"));
                NativeInputSendStatus::FailedBeforeSend
            }
            Err(SendError::Partial(source)) => {
                write_line_best_effort(
                    err,
                    format_args!("[경고] Wayland 입력 전송 상태 불확실: {source}"),
                );
                NativeInputSendStatus::PartialOrUnknown
            }
        }
    }
}
impl WaylandInput {
    fn dispatch(&mut self, timeout: c_int) -> InputResult<()> {
        let mut fds = [
            PollFd::new(self.portal.poll_fd()?),
            PollFd::new(self.ei.poll_fd()?),
        ];
        poll_fds(&mut fds, timeout)?;
        let [portal_poll, ei_poll] = fds;
        if portal_poll.is_invalid() {
            return Err(Cow::Borrowed("liboeffis poll descriptor가 무효화되었습니다."));
        }
        if ei_poll.is_invalid() {
            return Err(Cow::Borrowed("libei poll descriptor가 무효화되었습니다."));
        }
        if portal_poll.has_events() {
            let connected = self.portal.dispatch()?;
            if portal_poll.has_terminal_error() {
                return Err(Cow::Borrowed("liboeffis poll 연결이 종료되었습니다."));
            }
            if connected {
                return Err(Cow::Borrowed("예상하지 못한 추가 EIS 연결이 발생했습니다."));
            }
        }
        if ei_poll.has_events() {
            let pong = self.ei.dispatch()?;
            if ei_poll.has_terminal_error() {
                return Err(Cow::Borrowed("libei poll 연결이 종료되었습니다."));
            }
            if pong.is_some() {
                return Err(Cow::Borrowed("예상하지 못한 libei PONG 이벤트가 발생했습니다."));
            }
        }
        Ok(())
    }
    fn wait_until_ready<F>(
        &mut self,
        deadline: Instant,
        should_cancel: &mut F,
    ) -> InputResult<bool>
    where
        F: FnMut() -> bool,
    {
        while Instant::now() < deadline {
            if should_cancel() {
                return Ok(false);
            }
            self.dispatch(poll_timeout_until(deadline)?)?;
            if self.ei.device.is_some_and(|device| device.active) {
                return Ok(true);
            }
            if should_cancel() {
                return Ok(false);
            }
        }
        Err(Cow::Borrowed(
            "Wayland 가상 입력 장치 준비 시간이 초과되었습니다.",
        ))
    }
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
fn poll_timeout_until(deadline: Instant) -> InputResult<c_int> {
    let timeout_millis = deadline
        .saturating_duration_since(Instant::now())
        .min(POLL_INTERVAL)
        .as_millis()
        .max(1);
    c_int::try_from(timeout_millis)
        .map_err(|source| format!("poll 제한 시간 변환 실패: {source}").into())
}
fn poll_fds(fds: &mut [PollFd], timeout: c_int) -> InputResult<()> {
    let descriptor_count = PollDescriptorCount::try_from(fds.len())
        .map_err(|source| format!("poll descriptor 개수 변환 실패: {source}"))?;
    // SAFETY: fds is a writable pollfd array with the exact element count supplied.
    let result = unsafe { sys::poll(fds.as_mut_ptr(), descriptor_count, timeout) };
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
