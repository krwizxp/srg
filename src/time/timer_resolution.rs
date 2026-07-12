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
        let Some(due_time) = duration
            .as_nanos()
            .checked_add(99)
            .map(|rounded_nanos| rounded_nanos.div_euclid(100))
            .map(|units| units.max(1))
            .and_then(|units| i64::try_from(units).ok())
            .and_then(i64::checked_neg)
        else {
            sleep_remaining(started_at, duration);
            return;
        };
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
