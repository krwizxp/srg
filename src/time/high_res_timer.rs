#[link(name = "winmm")]
unsafe extern "system" {
    fn timeBeginPeriod(u_period: u32) -> u32;
    fn timeEndPeriod(u_period: u32) -> u32;
}
const TIMERR_NOERROR: u32 = 0;
const TARGET_PERIOD_MS: u32 = 1;
pub struct HighResTimerGuard;
impl HighResTimerGuard {
    pub fn new() -> Result<Self, String> {
        // SAFETY: `timeBeginPeriod` is a WinMM FFI call with a plain integer input
        // and does not impose additional aliasing or lifetime requirements.
        unsafe {
            if timeBeginPeriod(TARGET_PERIOD_MS) == TIMERR_NOERROR {
                Ok(Self)
            } else {
                Err(format!(
                    "타이머 해상도를 {TARGET_PERIOD_MS}ms로 설정하는 데 실패했습니다."
                ))
            }
        }
    }
}
impl Drop for HighResTimerGuard {
    fn drop(&mut self) {
        // SAFETY: This releases the timer period requested by `new` using the same value.
        unsafe {
            timeEndPeriod(TARGET_PERIOD_MS);
        }
    }
}
