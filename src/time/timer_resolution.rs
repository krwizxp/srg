use super::HighResTimerGuard;
use core::{
    ffi::c_void,
    ptr::null,
    time::Duration,
};
use std::{thread::sleep, time::Instant};
pub(super) mod sys;
pub(super) const CREATE_WAITABLE_TIMER_HIGH_RESOLUTION: u32 = 0x0000_0002;
const INFINITE: u32 = u32::MAX;
pub(super) const SYNCHRONIZE_ACCESS: u32 = 0x0010_0000;
pub(super) const TIMER_MODIFY_STATE_ACCESS: u32 = 0x0000_0002;
const WAIT_OBJECT_0: u32 = 0;
impl HighResTimerGuard {
    pub(super) fn sleep(&self, duration: Duration) {
        let started_at = Instant::now();
        let Ok(due_time_units) = i64::try_from(duration.as_nanos().div_ceil(100).max(1))
        else {
            sleep_remaining(started_at, duration);
            return;
        };
        let due_time = due_time_units.wrapping_neg();
        // SAFETY: handle is a live waitable timer handle and due_time points to a valid relative
        // due-time value for this call.
        let set_ok = unsafe {
            sys::set_waitable_timer_ex(
                self.handle.as_ptr(),
                &raw const due_time,
                0,
                None,
                null(),
                null(),
                0,
            )
        };
        if set_ok == 0_i32 {
            sleep_remaining(started_at, duration);
            return;
        }
        // SAFETY: handle remains valid while waiting.
        if unsafe {
            sys::wait_for_single_object(self.handle.as_ptr(), INFINITE) != WAIT_OBJECT_0
        } {
            sleep_remaining(started_at, duration);
        }
    }
}
impl Drop for HighResTimerGuard {
    fn drop(&mut self) {
        // SAFETY: handle was returned by CreateWaitableTimerExW and is closed exactly once here.
        unsafe {
            sys::close_handle(self.handle.as_ptr());
        }
    }
}
fn sleep_remaining(started_at: Instant, duration: Duration) {
    sleep(duration.saturating_sub(started_at.elapsed()));
}
