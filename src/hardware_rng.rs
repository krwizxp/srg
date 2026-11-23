use crate::diagnostic::Result;
use core::{
    arch::x86_64::{_rdrand64_step, _rdseed64_step},
    hint::spin_loop,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use std::{io::Write, is_x86_feature_detected, thread::yield_now, time::Instant};
pub(super) struct HardwareRng {
    fallback_notice_pending: AtomicBool,
    rdrand_supported: bool,
    rdseed_active: AtomicBool,
    stop_pending: AtomicBool,
}
#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum HardwareRandomSource {
    None,
    RdRand,
    RdSeed,
}
impl HardwareRng {
    pub(super) fn new() -> Self {
        Self {
            fallback_notice_pending: AtomicBool::new(false),
            rdrand_supported: is_x86_feature_detected!("rdrand"),
            rdseed_active: AtomicBool::new(is_x86_feature_detected!("rdseed")),
            stop_pending: AtomicBool::new(false),
        }
    }
    pub(super) fn next_u64(&self) -> Result<u64> {
        match self.source() {
            HardwareRandomSource::RdSeed => {
                let mut value = 0_u64;
                // SAFETY: callers only use this after confirming `rdseed` support.
                if unsafe { _rdseed64_step(&mut value) } == 1_i32 {
                    return Ok(value);
                }
                self.rdseed_random_slow()
            }
            HardwareRandomSource::RdRand => Self::rdrand_random(),
            HardwareRandomSource::None => Err("RDSEED·RDRAND 모두 미지원합니다.".into()),
        }
    }
    fn rdrand_random() -> Result<u64> {
        let mut value = 0_u64;
        for _ in 0_u8..10 {
            // SAFETY: callers only use this after confirming `rdrand` support.
            if unsafe { _rdrand64_step(&mut value) } == 1_i32 {
                return Ok(value);
            }
            spin_loop();
        }
        Err("RDRAND 실패".into())
    }
    #[cold]
    #[inline(never)]
    fn rdseed_random_slow(&self) -> Result<u64> {
        let mut value = 0_u64;
        if let Some(fallback) = self.try_rdrand_fallback() {
            return Ok(fallback);
        }
        if Self::try_rdseed(&mut value) {
            return Ok(value);
        }
        match self.source() {
            HardwareRandomSource::RdSeed => {}
            HardwareRandomSource::RdRand => return Self::rdrand_random(),
            HardwareRandomSource::None => return Err("RDSEED·RDRAND 모두 미지원합니다.".into()),
        }
        let started_at = Instant::now();
        while Instant::now().saturating_duration_since(started_at) < Duration::from_mins(5) {
            if self.stop_pending.load(Ordering::Relaxed) {
                return if self.rdrand_supported {
                    Self::rdrand_random()
                } else {
                    Err("RDSEED 재시도 중단, RDRAND 미지원".into())
                };
            }
            if Self::try_rdseed(&mut value) {
                return Ok(value);
            }
            if let Some(fallback) = self.try_rdrand_fallback() {
                return Ok(fallback);
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
            self.fallback_notice_pending.store(true, Ordering::Release);
            return Self::rdrand_random();
        }
        Err("RDSEED 5분 타임아웃, RDRAND 미지원".into())
    }
    pub(super) fn set_pending_read_interruption(&self, pending: bool) {
        self.stop_pending.store(pending, Ordering::Relaxed);
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
    fn try_rdrand_fallback(&self) -> Option<u64> {
        let fallback = self.rdrand_supported.then(Self::rdrand_random)?.ok()?;
        self.fallback_notice_pending.store(true, Ordering::Release);
        Some(fallback)
    }
    fn try_rdseed(value: &mut u64) -> bool {
        for _ in 0_u16..256 {
            // SAFETY: callers only use this after confirming `rdseed` support.
            if unsafe { _rdseed64_step(value) } == 1_i32 {
                return true;
            }
            spin_loop();
        }
        false
    }
    pub(super) fn write_initial_source_notice(&self, err: &mut dyn Write) -> Result<()> {
        match self.source() {
            HardwareRandomSource::RdSeed => {
                writeln!(err, "CPU가 RDSEED를 지원하여 RDSEED를 사용합니다.")?;
            }
            HardwareRandomSource::RdRand => {
                writeln!(err, "RDSEED를 미지원하여 RDRAND를 사용합니다.")?;
            }
            HardwareRandomSource::None => {}
        }
        Ok(())
    }
    pub(super) fn write_rdseed_fallback_notice(&self, err: &mut dyn Write) -> Result<()> {
        if self.fallback_notice_pending.swap(false, Ordering::Acquire) {
            writeln!(
                err,
                "{}",
                if self.source() == HardwareRandomSource::RdSeed {
                    "RDSEED 일시 재시도 중 RDRAND를 사용했습니다. 다음 호출은 RDSEED를 우선 시도합니다."
                } else {
                    "RDSEED 5분 타임아웃으로 RDRAND로 전환했습니다."
                }
            )?;
        }
        Ok(())
    }
}
