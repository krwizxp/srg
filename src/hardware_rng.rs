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
const RDSEED_TIMEOUT: Duration = Duration::from_mins(10);
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
static RNG_SOURCE: LazyLock<AtomicU8> = LazyLock::new(|| {
    AtomicU8::new(RNG_SUPPORT.initial_source.code())
});
static RDSEED_FALLBACK_NOTICE_PENDING: AtomicBool = AtomicBool::new(false);
struct HardwareRandomSupport {
    initial_source: HardwareRandomSource,
    rdrand: bool,
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
    fn try_hardware_value(self) -> Option<u64> {
        let mut value = 0_u64;
        match self {
            Self::RdSeed => {
                // SAFETY: callers only use this after confirming `rdseed` support.
                (unsafe { _rdseed64_step(&mut value) } == 1_i32).then_some(value)
            }
            Self::RdRand => {
                // SAFETY: callers only use this after confirming `rdrand` support.
                (unsafe { _rdrand64_step(&mut value) } == 1_i32).then_some(value)
            }
            Self::None => None,
        }
    }
}
pub fn get_hardware_random() -> Result<u64> {
    match hardware_random_source() {
        HardwareRandomSource::RdSeed => {
            let deadline = Instant::now()
                .checked_add(RDSEED_TIMEOUT)
                .ok_or("RDSEED timeout 계산 실패")?;
            while Instant::now() < deadline {
                for _ in 0_u16..RDSEED_SPIN_BURST_RETRY_COUNT {
                    if let Some(value) = HardwareRandomSource::RdSeed.try_hardware_value() {
                        return Ok(value);
                    }
                    spin_loop();
                }
                match hardware_random_source() {
                    HardwareRandomSource::RdSeed => yield_now(),
                    HardwareRandomSource::RdRand => return rdrand_random(),
                    HardwareRandomSource::None => {
                        return Err("RDSEED/RDRAND 사용 불가 상태입니다.".into());
                    }
                }
            }
            if RNG_SUPPORT.rdrand {
                RNG_SOURCE.store(HardwareRandomSource::RdRand.code(), Ordering::Relaxed);
                RDSEED_FALLBACK_NOTICE_PENDING.store(true, Ordering::Relaxed);
                return rdrand_random();
            }
            RNG_SOURCE.store(HardwareRandomSource::None.code(), Ordering::Relaxed);
            Err("RDSEED 10분 타임아웃, RDRAND 미지원".into())
        }
        HardwareRandomSource::RdRand => rdrand_random(),
        HardwareRandomSource::None => Err("RDSEED·RDRAND 모두 미지원합니다.".into()),
    }
}
pub fn hardware_random_source() -> HardwareRandomSource {
    match RNG_SOURCE.load(Ordering::Relaxed) {
        RNG_SOURCE_RDRAND => HardwareRandomSource::RdRand,
        RNG_SOURCE_RDSEED => HardwareRandomSource::RdSeed,
        _ => HardwareRandomSource::None,
    }
}
pub fn take_rdseed_fallback_notice() -> bool {
    RDSEED_FALLBACK_NOTICE_PENDING.swap(false, Ordering::Relaxed)
}
fn rdrand_random() -> Result<u64> {
    for _ in 0_u8..HARDWARE_RANDOM_RETRY_COUNT {
        if let Some(value) = HardwareRandomSource::RdRand.try_hardware_value() {
            return Ok(value);
        }
        spin_loop();
    }
    Err("RDRAND 실패".into())
}
