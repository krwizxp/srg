use super::hardware_rng::HardwareRng;
use crate::diagnostic::Result;
use std::io::Write;
pub(super) const FLOAT_INPUT_ERROR: &str =
    "유효한 정규 실수 값을 입력해야 합니다 (NaN, 무한대, 비정규 값 제외).";
pub(super) const MIN_ALLOWED_INTEGER_VALUE: i64 = i64::MIN + 1;
const RANDOM_BOUNDED_RETRY_LIMIT: usize = 1024;
const TWO_POW_32_F64: f64 = 4_294_967_296.0;
const U64_UNIT_SCALE: f64 = 1.0 / (TWO_POW_32_F64 * TWO_POW_32_F64);
pub(super) fn generate_random_integer(
    min_value: i64,
    max_value: i64,
    seed_modifier: u64,
    out: &mut dyn Write,
    rng: &HardwareRng,
) -> Result<()> {
    let rand_offset = random_bounded_inclusive(max_value.abs_diff(min_value), seed_modifier, rng)?;
    let result = min_value.strict_add_unsigned(rand_offset);
    writeln!(
        out,
        "무작위 정수({min_value} ~ {max_value}): {result} (0x{result:X})"
    )
    .map_err(Into::into)
}
pub(super) fn generate_random_float(
    min_value: f64,
    max_value: f64,
    seed_modifier: u64,
    out: &mut dyn Write,
    rng: &HardwareRng,
) -> Result<()> {
    let [b0, b1, b2, b3, b4, b5, b6, b7] = (rng.next_u64()? ^ seed_modifier).to_be_bytes();
    let upper_32 = u32::from_be_bytes([b0, b1, b2, b3]);
    let lower_32 = u32::from_be_bytes([b4, b5, b6, b7]);
    let scale = f64::from(upper_32).mul_add(TWO_POW_32_F64, f64::from(lower_32)) * U64_UNIT_SCALE;
    let result = if min_value.to_bits() == max_value.to_bits() {
        min_value
    } else {
        scale.mul_add(max_value - min_value, min_value)
    };
    writeln!(out, "무작위 실수({min_value} ~ {max_value}): {result}").map_err(Into::into)
}
pub(super) fn random_bounded_inclusive(
    inclusive_max: u64,
    seed_mod: u64,
    rng: &HardwareRng,
) -> Result<u64> {
    let range_value = inclusive_max.strict_add(1);
    let threshold = range_value.wrapping_neg().rem_euclid(range_value);
    for _ in 0..RANDOM_BOUNDED_RETRY_LIMIT {
        let (low_bits, high_bits) = (rng.next_u64()? ^ seed_mod).carrying_mul(range_value, 0_u64);
        if low_bits >= threshold {
            return Ok(high_bits);
        }
    }
    Err("bounded random rejection sampling 시도 횟수를 초과했습니다.".into())
}
