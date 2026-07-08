use crate::diagnostic::Result;
use alloc::sync::Arc;
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
static RNG_SUPPORT: LazyLock<HardwareRandomSupport> = LazyLock::new(|| {
    let rdseed = is_x86_feature_detected!("rdseed");
    let rdrand = is_x86_feature_detected!("rdrand");
    let initial_source = if rdseed {
        HardwareRandomSource::RdSeed
    } else if rdrand {
        HardwareRandomSource::RdRand
    } else {
        HardwareRandomSource::None
    };
    HardwareRandomSupport {
        initial_source,
        rdrand,
    }
});
struct HardwareRandomSupport {
    initial_source: HardwareRandomSource,
    rdrand: bool,
}
struct HardwareRandomState {
    fallback_notice_pending: AtomicBool,
    source: AtomicU8,
}
pub struct HardwareRng {
    source: HardwareRandomSource,
    state: Arc<HardwareRandomState>,
}
impl Default for HardwareRng {
    fn default() -> Self {
        Self::new()
    }
}
impl HardwareRandomState {
    fn mark_rdseed_fallback_notice_pending(&self) {
        self.fallback_notice_pending.store(true, Ordering::Relaxed);
    }
    fn source(&self) -> HardwareRandomSource {
        match self.source.load(Ordering::Relaxed) {
            RNG_SOURCE_RDRAND => HardwareRandomSource::RdRand,
            RNG_SOURCE_RDSEED => HardwareRandomSource::RdSeed,
            _ => HardwareRandomSource::None,
        }
    }
    fn store_source(&self, source: HardwareRandomSource) {
        self.source.store(source.code(), Ordering::Relaxed);
    }
    fn take_rdseed_fallback_notice(&self) -> bool {
        self.fallback_notice_pending.swap(false, Ordering::Relaxed)
    }
}
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum HardwareRandomSource {
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
    pub fn new() -> Self {
        let source = RNG_SUPPORT.initial_source;
        Self::with_state(Arc::new(HardwareRandomState {
            fallback_notice_pending: AtomicBool::new(false),
            source: AtomicU8::new(source.code()),
        }), source)
    }
    pub fn next_u64(&mut self) -> Result<u64> {
        match self.source {
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
    fn rdseed_random(&mut self) -> Result<u64> {
        let mut value = 0_u64;
        for _ in 0_u16..RDSEED_SPIN_BURST_RETRY_COUNT {
            // SAFETY: callers only use this after confirming `rdseed` support.
            if unsafe { _rdseed64_step(&mut value) } == 1_i32 {
                return Ok(value);
            }
            spin_loop();
        }
        self.sync_source_from_shared();
        match self.source {
            HardwareRandomSource::RdSeed => {}
            HardwareRandomSource::RdRand => return Self::rdrand_random(),
            HardwareRandomSource::None => return Err("RDSEED·RDRAND 모두 미지원합니다.".into()),
        }
        let deadline = Instant::now()
            .checked_add(RDSEED_TIMEOUT)
            .ok_or("RDSEED timeout 계산 실패")?;
        while Instant::now() < deadline {
            for _ in 0_u16..RDSEED_SPIN_BURST_RETRY_COUNT {
                // SAFETY: callers only use this after confirming `rdseed` support.
                if unsafe { _rdseed64_step(&mut value) } == 1_i32 {
                    return Ok(value);
                }
                spin_loop();
            }
            self.sync_source_from_shared();
            match self.source {
                HardwareRandomSource::RdSeed => yield_now(),
                HardwareRandomSource::RdRand => return Self::rdrand_random(),
                HardwareRandomSource::None => {
                    return Err("RDSEED/RDRAND 사용 불가 상태입니다.".into());
                }
            }
        }
        if RNG_SUPPORT.rdrand {
            self.source = HardwareRandomSource::RdRand;
            self.state.store_source(HardwareRandomSource::RdRand);
            self.state.mark_rdseed_fallback_notice_pending();
            return Self::rdrand_random();
        }
        self.source = HardwareRandomSource::None;
        self.state.store_source(HardwareRandomSource::None);
        Err("RDSEED 5분 타임아웃, RDRAND 미지원".into())
    }
    pub fn shared_source_rng(&self) -> Self {
        Self::with_state(Arc::clone(&self.state), self.state.source())
    }
    pub fn source(&mut self) -> HardwareRandomSource {
        self.sync_source_from_shared();
        self.source
    }
    pub(crate) fn sync_source_from_shared(&mut self) {
        self.source = self.state.source();
    }
    pub fn take_rdseed_fallback_notice(&mut self) -> bool {
        self.sync_source_from_shared();
        self.state.take_rdseed_fallback_notice()
    }
    const fn with_state(state: Arc<HardwareRandomState>, source: HardwareRandomSource) -> Self {
        Self { source, state }
    }
}
