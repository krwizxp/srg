pub(super) use self::sys::time_begin_period;
mod sys {
    #[link(name = "winmm")]
    unsafe extern "system" {
        #[link_name = "timeBeginPeriod"]
        pub(in crate::time) fn time_begin_period(u_period: u32) -> u32;
        #[link_name = "timeEndPeriod"]
        pub(super) fn time_end_period(u_period: u32) -> u32;
    }
}
pub(super) const TIMERR_NOERROR: u32 = 0;
pub(super) const TARGET_PERIOD_MS: u32 = 1;
pub(super) struct HighResTimerGuard;
impl Drop for HighResTimerGuard {
    fn drop(&mut self) {
        // SAFETY: This releases the timer period requested when the guard was created.
        unsafe {
            sys::time_end_period(TARGET_PERIOD_MS);
        }
    }
}
