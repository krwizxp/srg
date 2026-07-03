use crate::diagnostic::Result;
use core::{
    arch::x86_64::{_rdrand64_step, _rdseed64_step},
    hint::spin_loop,
    sync::atomic::{AtomicBool, AtomicU8, Ordering},
    time::Duration,
};
use std::{
    is_x86_feature_detected,
    sync::{LazyLock, Mutex},
    thread::yield_now,
    time::Instant,
};
const HARDWARE_RANDOM_RETRY_COUNT: u8 = 10;
const HARDWARE_RANDOM_REPETITION_LIMIT: u8 = 8;
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
static RNG_SOURCE: LazyLock<AtomicU8> = LazyLock::new(|| {
    AtomicU8::new(RNG_SUPPORT.initial_source.code())
});
static RNG_HEALTH: LazyLock<Mutex<HardwareRandomHealth>> = LazyLock::new(|| {
    Mutex::new(HardwareRandomHealth::default())
});
static RDSEED_FALLBACK_NOTICE_PENDING: AtomicBool = AtomicBool::new(false);
struct HardwareRandomSupport {
    initial_source: HardwareRandomSource,
    rdrand: bool,
}
#[derive(Default)]
struct HardwareRandomHealth {
    rdrand: SourceHealth,
    rdseed: SourceHealth,
}
#[derive(Default)]
struct SourceHealth {
    last_value: Option<u64>,
    repetition_count: u8,
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
    const fn label(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::RdRand => "RDRAND",
            Self::RdSeed => "RDSEED",
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
impl HardwareRandomHealth {
    fn check(&mut self, source: HardwareRandomSource, value: u64) -> Result<u64> {
        let Some(health) = self.source_health_mut(source) else {
            return Err("하드웨어 RNG source 상태가 올바르지 않습니다.".into());
        };
        if health.last_value == Some(value) {
            health.repetition_count = health.repetition_count.saturating_add(1);
        } else {
            health.last_value = Some(value);
            health.repetition_count = 1;
        }
        if health.repetition_count >= HARDWARE_RANDOM_REPETITION_LIMIT {
            return Err(format!(
                "{} 반복값 이상 감지: 동일한 64비트 값이 {}회 연속 반환되었습니다.",
                source.label(),
                health.repetition_count,
            )
            .into());
        }
        Ok(value)
    }
    const fn source_health_mut(
        &mut self,
        source: HardwareRandomSource,
    ) -> Option<&mut SourceHealth> {
        match source {
            HardwareRandomSource::RdRand => Some(&mut self.rdrand),
            HardwareRandomSource::RdSeed => Some(&mut self.rdseed),
            HardwareRandomSource::None => None,
        }
    }
}
fn health_checked_value(source: HardwareRandomSource, value: u64) -> Result<u64> {
    let mut health = RNG_HEALTH
        .lock()
        .map_err(|_lock_error| "하드웨어 RNG health check 상태 잠금 실패")?;
    health.check(source, value)
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
                        return health_checked_value(HardwareRandomSource::RdSeed, value);
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
            Err("RDSEED 5분 타임아웃, RDRAND 미지원".into())
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
            return health_checked_value(HardwareRandomSource::RdRand, value);
        }
        spin_loop();
    }
    Err("RDRAND 실패".into())
}
