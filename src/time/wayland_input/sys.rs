use super::{PollDescriptorCount, PollFd, c_char, c_int, c_void};
unsafe extern "C" {
    pub(super) fn poll(
        fds: *mut PollFd,
        nfds: PollDescriptorCount,
        timeout: c_int,
    ) -> c_int;
}
#[link(name = "dl")]
unsafe extern "C" {
    pub(super) fn dlclose(handle: *mut c_void) -> c_int;
    pub(super) fn dlerror() -> *mut c_char;
    pub(super) fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    pub(super) fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
