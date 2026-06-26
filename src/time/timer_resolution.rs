use core::{
    ffi::c_void,
    ptr::{NonNull, null},
    time::Duration,
};
use std::thread::sleep;
pub(super) mod sys {
    use super::c_void;
    #[link(name = "winmm")]
    unsafe extern "system" {
        #[link_name = "timeBeginPeriod"]
        pub(in crate::time) fn time_begin_period(u_period: u32) -> u32;
        #[link_name = "timeEndPeriod"]
        pub(super) fn time_end_period(u_period: u32) -> u32;
    }
    unsafe extern "system" {
        #[link_name = "CloseHandle"]
        pub(super) fn close_handle(h_object: *mut c_void) -> i32;
        #[link_name = "CreateWaitableTimerExW"]
        pub(in crate::time) fn create_waitable_timer_ex_w(
            timer_attributes: *const c_void,
            timer_name: *const u16,
            flags: u32,
            desired_access: u32,
        ) -> *mut c_void;
        #[link_name = "SetWaitableTimerEx"]
        pub(super) fn set_waitable_timer_ex(
            timer: *mut c_void,
            due_time: *const i64,
            period: i32,
            completion_routine: *const c_void,
            arg_to_completion_routine: *const c_void,
            wake_context: *const c_void,
            tolerable_delay: u32,
        ) -> i32;
        #[link_name = "WaitForSingleObject"]
        pub(super) fn wait_for_single_object(handle: *mut c_void, milliseconds: u32) -> u32;
    }
}
pub(super) const CREATE_WAITABLE_TIMER_HIGH_RESOLUTION: u32 = 0x0000_0002;
const INFINITE: u32 = u32::MAX;
pub(super) const SYNCHRONIZE_ACCESS: u32 = 0x0010_0000;
pub(super) const TIMERR_NOERROR: u32 = 0;
pub(super) const TIMER_MODIFY_STATE_ACCESS: u32 = 0x0000_0002;
const WAIT_OBJECT_0: u32 = 0;
pub(super) const TARGET_PERIOD_MS: u32 = 1;
pub(super) enum HighResTimerGuard {
    PeriodAcquired,
    WaitTimer(NonNull<c_void>),
}
impl HighResTimerGuard {
    pub(super) fn sleep(&self, duration: Duration) {
        let &Self::WaitTimer(handle) = self else {
            sleep(duration);
            return;
        };
        let Some(due_time) = duration
            .as_nanos()
            .checked_add(99)
            .and_then(|rounded_nanos| rounded_nanos.checked_div(100))
            .map(|units| units.max(1))
            .and_then(|units| i64::try_from(units).ok())
            .and_then(i64::checked_neg)
        else {
            sleep(duration);
            return;
        };
        // SAFETY: handle is a live waitable timer handle and due_time points to a valid relative
        // due-time value for this call.
        let set_ok = unsafe {
            sys::set_waitable_timer_ex(
                handle.as_ptr(),
                &raw const due_time,
                0,
                null(),
                null(),
                null(),
                0,
            )
        };
        if set_ok == 0_i32 {
            sleep(duration);
            return;
        }
        // SAFETY: handle remains valid while waiting.
        if unsafe { sys::wait_for_single_object(handle.as_ptr(), INFINITE) != WAIT_OBJECT_0 } {
            sleep(duration);
        }
    }
}
impl Drop for HighResTimerGuard {
    fn drop(&mut self) {
        match *self {
            Self::PeriodAcquired => {
                // SAFETY: This releases the timer period requested when the guard was created.
                unsafe {
                    sys::time_end_period(TARGET_PERIOD_MS);
                }
            }
            Self::WaitTimer(handle) => {
                // SAFETY: handle was returned by CreateWaitableTimerExW and is closed exactly once here.
                unsafe {
                    sys::close_handle(handle.as_ptr());
                }
            }
        }
    }
}
