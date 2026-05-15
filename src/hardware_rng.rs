use super::Result;
use core::{
    arch::x86_64::{_rdrand64_step, _rdseed64_step},
    hint::spin_loop,
};
use std::{is_x86_feature_detected, sync::LazyLock};
const HARDWARE_RANDOM_RETRY_COUNT: u8 = 10;
pub static RNG_SOURCE: LazyLock<RngSource> = LazyLock::new(|| {
    if is_x86_feature_detected!("rdseed") {
        RngSource::RdSeed
    } else if is_x86_feature_detected!("rdrand") {
        RngSource::RdRand
    } else {
        RngSource::None
    }
});
#[derive(Clone, Copy)]
pub enum RngSource {
    None,
    RdRand,
    RdSeed,
}
impl RngSource {
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
    match *RNG_SOURCE {
        RngSource::RdSeed => loop {
            if let Some(value) = RngSource::RdSeed.try_hardware_value() {
                break Ok(value);
            }
            spin_loop();
        },
        RngSource::RdRand => {
            for _ in 0_u8..HARDWARE_RANDOM_RETRY_COUNT {
                if let Some(value) = RngSource::RdRand.try_hardware_value() {
                    return Ok(value);
                }
                spin_loop();
            }
            Err("RDRAND 실패".into())
        }
        RngSource::None => Err("RDSEED·RDRAND 모두 미지원합니다.".into()),
    }
}
