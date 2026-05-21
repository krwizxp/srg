use core::result::Result as StdResult;
const TIMERR_NOERROR: u32 = 0;
pub(super) const TARGET_PERIOD_MS: u32 = 1;
pub(super) struct HighResTimerGuard;
pub(super) struct HighResTimerRequest;
#[link(name = "winmm")]
unsafe extern "system" {
    fn timeBeginPeriod(u_period: u32) -> u32;
    fn timeEndPeriod(u_period: u32) -> u32;
}
impl TryFrom<HighResTimerRequest> for HighResTimerGuard {
    type Error = ();
    fn try_from(_value: HighResTimerRequest) -> StdResult<Self, Self::Error> {
        // SAFETY: `timeBeginPeriod` is a WinMM FFI call with a plain integer input.
        if unsafe { timeBeginPeriod(TARGET_PERIOD_MS) } == TIMERR_NOERROR {
            Ok(Self)
        } else {
            Err(())
        }
    }
}
impl Drop for HighResTimerGuard {
    fn drop(&mut self) {
        // SAFETY: This releases the timer period requested when the guard was created.
        unsafe {
            timeEndPeriod(TARGET_PERIOD_MS);
        }
    }
}
