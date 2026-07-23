use crate::diagnostic::Result;
use core::{
    arch::x86_64::{_rdrand64_step, _rdseed64_step},
    hint::spin_loop,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use std::{
    io::Write,
    is_x86_feature_detected,
    sync::LazyLock,
    thread::yield_now,
    time::Instant,
};
const HARDWARE_RANDOM_RETRY_COUNT: u8 = 10;
const RDSEED_SPIN_BURST_RETRY_COUNT: u16 = 256;
const RDSEED_TIMEOUT: Duration = Duration::from_mins(5);
static INITIAL_RNG_SUPPORT: LazyLock<(bool, bool)> = LazyLock::new(|| {
    (
        is_x86_feature_detected!("rdseed"),
        is_x86_feature_detected!("rdrand"),
    )
});
pub(super) struct HardwareRng {
    fallback_notice_pending: AtomicBool,
    rdrand_supported: bool,
    rdseed_active: AtomicBool,
}
#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum HardwareRandomSource {
    None,
    RdRand,
    RdSeed,
}
impl HardwareRng {
    pub(super) fn new() -> Self {
        let &(rdseed_supported, rdrand_supported) = &*INITIAL_RNG_SUPPORT;
        Self {
            fallback_notice_pending: AtomicBool::new(false),
            rdrand_supported,
            rdseed_active: AtomicBool::new(rdseed_supported),
        }
    }
    pub(super) fn next_u64(&self) -> Result<u64> {
        match self.source() {
            HardwareRandomSource::RdSeed => self.rdseed_random(),
            HardwareRandomSource::RdRand => Self::rdrand_random(),
            HardwareRandomSource::None => Err("RDSEED·RDRAND 모두 미지원합니다.".into()),
        }
    }
    fn rdrand_random() -> Result<u64> {
        let mut value = 0_u64;
        for _ in 0_u8..HARDWARE_RANDOM_RETRY_COUNT {
            // SAFETY: callers only use this after confirming `rdrand` support.
            if unsafe { _rdrand64_step(&mut value) } == 1_i32 {
                return Ok(value);
            }
            spin_loop();
        }
        Err("RDRAND 실패".into())
    }
    fn rdseed_random(&self) -> Result<u64> {
        let mut value = 0_u64;
        if Self::try_rdseed(&mut value) {
            return Ok(value);
        }
        match self.source() {
            HardwareRandomSource::RdSeed => {}
            HardwareRandomSource::RdRand => return Self::rdrand_random(),
            HardwareRandomSource::None => return Err("RDSEED·RDRAND 모두 미지원합니다.".into()),
        }
        let retry_started_at = Instant::now();
        while Instant::now().saturating_duration_since(retry_started_at) < RDSEED_TIMEOUT {
            if Self::try_rdseed(&mut value) {
                return Ok(value);
            }
            match self.source() {
                HardwareRandomSource::RdSeed => yield_now(),
                HardwareRandomSource::RdRand => return Self::rdrand_random(),
                HardwareRandomSource::None => {
                    return Err("RDSEED/RDRAND 사용 불가 상태입니다.".into());
                }
            }
        }
        self.rdseed_active.store(false, Ordering::Relaxed);
        if self.rdrand_supported {
            self.fallback_notice_pending
                .store(true, Ordering::Relaxed);
            return Self::rdrand_random();
        }
        Err("RDSEED 5분 타임아웃, RDRAND 미지원".into())
    }
    pub(super) fn source(&self) -> HardwareRandomSource {
        if self.rdseed_active.load(Ordering::Relaxed) {
            HardwareRandomSource::RdSeed
        } else if self.rdrand_supported {
            HardwareRandomSource::RdRand
        } else {
            HardwareRandomSource::None
        }
    }
    fn try_rdseed(value: &mut u64) -> bool {
        for _ in 0_u16..RDSEED_SPIN_BURST_RETRY_COUNT {
            // SAFETY: callers only use this after confirming `rdseed` support.
            if unsafe { _rdseed64_step(value) } == 1_i32 {
                return true;
            }
            spin_loop();
        }
        false
    }
    pub(super) fn write_initial_source_notice(&self, err: &mut dyn Write) -> Result<()> {
        if self.source() == HardwareRandomSource::RdRand {
            writeln!(err, "RDSEED를 미지원하여 RDRAND를 사용합니다.")?;
        }
        Ok(())
    }
    pub(super) fn write_rdseed_fallback_notice(&self, err: &mut dyn Write) -> Result<()> {
        if self.fallback_notice_pending.swap(false, Ordering::Relaxed) {
            writeln!(err, "RDSEED 5분 타임아웃으로 RDRAND로 전환했습니다.")?;
        }
        Ok(())
    }
}
