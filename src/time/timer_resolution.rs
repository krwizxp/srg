use super::HighResTimerGuard;
use core::{ffi::c_void, ptr::null, time::Duration};
use std::{io, thread::sleep, time::Instant};
unsafe extern "system" {
    #[link_name = "CloseHandle"]
    fn close_handle(h_object: *mut c_void) -> i32;
    #[link_name = "CreateWaitableTimerExW"]
    pub(super) fn create_waitable_timer_ex_w(
        timer_attributes: *const c_void,
        timer_name: *const u16,
        flags: u32,
        desired_access: u32,
    ) -> *mut c_void;
    #[link_name = "SetWaitableTimerEx"]
    fn set_waitable_timer_ex(
        timer: *mut c_void,
        due_time: *const i64,
        period: i32,
        completion_routine: Option<unsafe extern "system" fn(*const c_void, u32, u32)>,
        arg_to_completion_routine: *const c_void,
        wake_context: *const c_void,
        tolerable_delay: u32,
    ) -> i32;
    #[link_name = "WaitForSingleObject"]
    fn wait_for_single_object(handle: *mut c_void, milliseconds: u32) -> u32;
}
impl HighResTimerGuard {
    pub(super) fn sleep(&self, duration: Duration) -> io::Result<()> {
        let started_at = Instant::now();
        (|| -> io::Result<()> {
            let due_time = i64::try_from(duration.as_nanos().div_ceil(100).max(1))
                .map_err(|source| io::Error::new(io::ErrorKind::InvalidInput, source))?
                .wrapping_neg();
            // SAFETY: handle is a live waitable timer handle and due_time points to a valid relative
            // due-time value for this call.
            let set_ok = unsafe {
                set_waitable_timer_ex(
                    self.handle.as_ptr(),
                    &raw const due_time,
                    0,
                    None,
                    null(),
                    null(),
                    0,
                )
            };
            (set_ok != 0_i32).ok_or_else(|| {
                io::Error::other(format!(
                    "SetWaitableTimerEx 실패: {}",
                    io::Error::last_os_error()
                ))
            })?;
            // SAFETY: handle remains valid while waiting.
            let wait_status = unsafe { wait_for_single_object(self.handle.as_ptr(), u32::MAX) };
            (wait_status == 0_u32).ok_or_else(|| {
                io::Error::other(format!(
                    "WaitForSingleObject 상태 {wait_status:#010x}: {}",
                    io::Error::last_os_error()
                ))
            })
        })()
        .inspect_err(|_| sleep(duration.saturating_sub(started_at.elapsed())))
    }
}
impl Drop for HighResTimerGuard {
    fn drop(&mut self) {
        // SAFETY: handle was returned by CreateWaitableTimerExW and is closed exactly once here.
        unsafe {
            close_handle(self.handle.as_ptr());
        }
    }
}
