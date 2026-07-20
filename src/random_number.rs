use super::{
    hardware_rng::HardwareRng,
    input::parse_regular_f64, input::read_parsed_value,
};
use crate::diagnostic::{AppError, Result};
use core::{
    num::NonZeroU64,
    ops::{Mul as NumericMul, Sub as NumericSub},
};
use std::io::Write;
const FLOAT_INPUT_ERROR: &str =
    "유효한 정규 실수 값을 입력해야 합니다 (NaN, 무한대, 비정규 값 제외).";
const INTEGER_MODE_TITLE: &str =
    "\n무작위 정수 생성기(지원 범위: -9223372036854775807 ~ 9223372036854775807)";
const MIN_ALLOWED_INTEGER_VALUE: i64 = i64::MIN + 1;
const RANDOM_BOUNDED_RETRY_LIMIT: usize = 1024;
const TWO_POW_32_F64: f64 = 4_294_967_296.0;
const U64_UNIT_SCALE: f64 = 1.0 / (TWO_POW_32_F64 * TWO_POW_32_F64);
#[derive(Clone, Copy)]
pub(super) enum RandomNumberMode {
    Float,
    Integer,
}
pub(super) fn generate_random_number(
    mode: RandomNumberMode,
    seed_modifier: u64,
    input_buffer: &mut String,
    out: &mut dyn Write,
    err: &mut dyn Write,
    rng: &HardwareRng,
) -> Result<()> {
    match mode {
        RandomNumberMode::Integer => {
            writeln!(out, "{INTEGER_MODE_TITLE}")?;
            let min_value = loop {
                let value = read_parsed_value(
                    format_args!("최솟값을 입력해 주세요 ({MIN_ALLOWED_INTEGER_VALUE} 이상): "),
                    input_buffer,
                    out,
                    err,
                    "유효한 정수 형식이 아닙니다.",
                    |line| line.parse::<i64>().ok(),
                )?;
                if value >= MIN_ALLOWED_INTEGER_VALUE {
                    break value;
                }
                writeln!(err, "{MIN_ALLOWED_INTEGER_VALUE} 이상의 값을 입력해 주세요.")?;
            };
            let max_value = loop {
                let value = read_parsed_value(
                    format_args!("최댓값을 입력해 주세요: "),
                    input_buffer,
                    out,
                    err,
                    "유효한 정수 형식이 아닙니다.",
                    |line| line.parse::<i64>().ok(),
                )?;
                if value >= min_value {
                    break value;
                }
                writeln!(err, "최댓값은 최솟값보다 크거나 같아야 합니다.")?;
            };
            generate_random_integer(min_value, max_value, seed_modifier, out, rng)?;
        }
        RandomNumberMode::Float => {
            writeln!(out, "\n무작위 실수 생성기")?;
            let min_value = read_parsed_value(
                format_args!("최솟값을 입력해 주세요: "),
                input_buffer,
                out,
                err,
                FLOAT_INPUT_ERROR,
                parse_regular_f64,
            )?;
            let max_value = loop {
                let value = read_parsed_value(
                    format_args!("최댓값을 입력해 주세요: "),
                    input_buffer,
                    out,
                    err,
                    FLOAT_INPUT_ERROR,
                    parse_regular_f64,
                )?;
                if value >= min_value {
                    break value;
                }
                writeln!(err, "최댓값은 최솟값보다 크거나 같아야 합니다.")?;
            };
            generate_random_float(min_value, max_value, seed_modifier, out, rng)?;
        }
    }
    Ok(())
}
pub(super) fn generate_random_integer(
    min_value: i64,
    max_value: i64,
    seed_modifier: u64,
    out: &mut dyn Write,
    rng: &HardwareRng,
) -> Result<()> {
    validate_random_integer_range(min_value, max_value)?;
    let range_size = NonZeroU64::MIN.saturating_add(max_value.abs_diff(min_value));
    let rand_offset = random_bounded(range_size, seed_modifier, rng)?;
    let result = min_value.wrapping_add_unsigned(rand_offset);
    writeln!(
        out,
        "무작위 정수({min_value} ~ {max_value}): {result} (0x{result:X})"
    )?;
    Ok(())
}
pub(super) fn validate_random_integer_range(min_value: i64, max_value: i64) -> Result<()> {
    if min_value < MIN_ALLOWED_INTEGER_VALUE {
        return Err(AppError::message(format!(
            "최솟값은 {MIN_ALLOWED_INTEGER_VALUE} 이상이어야 합니다."
        )));
    }
    if max_value < min_value {
        return Err("최댓값은 최솟값보다 크거나 같아야 합니다.".into());
    }
    Ok(())
}
pub(super) fn generate_random_float(
    min_value: f64,
    max_value: f64,
    seed_modifier: u64,
    out: &mut dyn Write,
    rng: &HardwareRng,
) -> Result<()> {
    validate_random_float_range(min_value, max_value)?;
    let random_u64 = rng.next_u64()? ^ seed_modifier;
    let [b0, b1, b2, b3, b4, b5, b6, b7] = random_u64.to_be_bytes();
    let upper_32 = u32::from_be_bytes([b0, b1, b2, b3]);
    let lower_32 = u32::from_be_bytes([b4, b5, b6, b7]);
    let scale = NumericMul::mul(
        f64::from(upper_32).mul_add(TWO_POW_32_F64, f64::from(lower_32)),
        U64_UNIT_SCALE,
    );
    let result = if min_value.to_bits() == max_value.to_bits() {
        min_value
    } else {
        scale.mul_add(NumericSub::sub(max_value, min_value), min_value)
    };
    if !result.is_finite() {
        return Err("실수 난수 결과가 유한하지 않습니다.".into());
    }
    writeln!(out, "무작위 실수({min_value} ~ {max_value}): {result}")?;
    Ok(())
}
pub(super) fn validate_random_float_range(min_value: f64, max_value: f64) -> Result<()> {
    if !min_value.is_finite()
        || min_value.is_subnormal()
        || !max_value.is_finite()
        || max_value.is_subnormal()
    {
        return Err(FLOAT_INPUT_ERROR.into());
    }
    if max_value < min_value {
        return Err("최댓값은 최솟값보다 크거나 같아야 합니다.".into());
    }
    if !NumericSub::sub(max_value, min_value).is_finite() {
        return Err("실수 범위가 너무 커서 안전하게 계산할 수 없습니다.".into());
    }
    Ok(())
}
pub(super) fn random_bounded(
    range_size: NonZeroU64,
    seed_mod: u64,
    rng: &HardwareRng,
) -> Result<u64> {
    let range_value = range_size.get();
    let threshold = range_value.wrapping_neg().rem_euclid(range_value);
    for _ in 0..RANDOM_BOUNDED_RETRY_LIMIT {
        let (low_bits, high_bits) =
            (rng.next_u64()? ^ seed_mod).carrying_mul(range_value, 0_u64);
        if low_bits >= threshold {
            return Ok(high_bits);
        }
    }
    Err("bounded random rejection sampling 시도 횟수를 초과했습니다.".into())
}
