use crate::diagnostic::Result;
use core::{
    arch::x86_64::{_rdrand64_step, _rdseed64_step},
    hint::spin_loop,
    sync::atomic::{AtomicBool, AtomicU8, Ordering},
    time::Duration,
};
use std::{
    is_x86_feature_detected,
    sync::LazyLock,
    thread::yield_now,
    time::Instant,
};
const HARDWARE_RANDOM_RETRY_COUNT: u8 = 10;
const RDSEED_SPIN_BURST_RETRY_COUNT: u16 = 256;
const RDSEED_TIMEOUT: Duration = Duration::from_mins(5);
const RNG_SOURCE_NONE: u8 = 0;
const RNG_SOURCE_RDRAND: u8 = 1;
const RNG_SOURCE_RDSEED: u8 = 2;
static INITIAL_RNG_SOURCE: LazyLock<HardwareRandomSource> = LazyLock::new(|| {
    if is_x86_feature_detected!("rdseed") {
        HardwareRandomSource::RdSeed
    } else if is_x86_feature_detected!("rdrand") {
        HardwareRandomSource::RdRand
    } else {
        HardwareRandomSource::None
    }
});
pub(super) struct HardwareRng {
    fallback_notice_pending: AtomicBool,
    source: AtomicU8,
}
impl Default for HardwareRng {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum HardwareRandomSource {
    None,
    RdRand,
    RdSeed,
}
impl HardwareRandomSource {
    const fn code(self) -> u8 {
        match self {
            Self::None => RNG_SOURCE_NONE,
            Self::RdRand => RNG_SOURCE_RDRAND,
            Self::RdSeed => RNG_SOURCE_RDSEED,
        }
    }
}
impl HardwareRng {
    fn current_source(&self) -> HardwareRandomSource {
        match self.source.load(Ordering::Relaxed) {
            RNG_SOURCE_RDRAND => HardwareRandomSource::RdRand,
            RNG_SOURCE_RDSEED => HardwareRandomSource::RdSeed,
            _ => HardwareRandomSource::None,
        }
    }
    pub(super) fn new() -> Self {
        let source = *INITIAL_RNG_SOURCE;
        Self {
            fallback_notice_pending: AtomicBool::new(false),
            source: AtomicU8::new(source.code()),
        }
    }
    pub(super) fn next_u64(&self) -> Result<u64> {
        match self.current_source() {
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
        match self.current_source() {
            HardwareRandomSource::RdSeed => {}
            HardwareRandomSource::RdRand => return Self::rdrand_random(),
            HardwareRandomSource::None => return Err("RDSEED·RDRAND 모두 미지원합니다.".into()),
        }
        let retry_started_at = Instant::now();
        while Instant::now().saturating_duration_since(retry_started_at) < RDSEED_TIMEOUT {
            if Self::try_rdseed(&mut value) {
                return Ok(value);
            }
            match self.current_source() {
                HardwareRandomSource::RdSeed => yield_now(),
                HardwareRandomSource::RdRand => return Self::rdrand_random(),
                HardwareRandomSource::None => {
                    return Err("RDSEED/RDRAND 사용 불가 상태입니다.".into());
                }
            }
        }
        if is_x86_feature_detected!("rdrand") {
            self.store_source(HardwareRandomSource::RdRand);
            self.fallback_notice_pending
                .store(true, Ordering::Relaxed);
            return Self::rdrand_random();
        }
        self.store_source(HardwareRandomSource::None);
        Err("RDSEED 5분 타임아웃, RDRAND 미지원".into())
    }
    pub(super) fn source(&self) -> HardwareRandomSource {
        self.current_source()
    }
    fn store_source(&self, source: HardwareRandomSource) {
        self.source.store(source.code(), Ordering::Relaxed);
    }
    pub(super) fn take_rdseed_fallback_notice(&self) -> bool {
        self.fallback_notice_pending
            .swap(false, Ordering::Relaxed)
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
}
