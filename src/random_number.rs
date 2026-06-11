use super::{
    hardware_rng::get_hardware_random,
    input::parse_regular_f64, input::read_parsed_value,
};
use crate::{
    constants::{TWO_POW_32_F64, U64_UNIT_SCALE},
    diagnostic::{AppError, Result},
};
use core::{
    num::{NonZeroU64, NonZeroU128},
    ops::{Mul as NumericMul, Sub as NumericSub},
};
use std::io::Write;
const FLOAT_INPUT_ERROR: &str =
    "유효한 정규 실수 값을 입력해야 합니다 (NaN, 무한대, 비정규 값 제외).";
const INTEGER_MODE_TITLE: &str =
    "\n무작위 정수 생성기(지원 범위: -9223372036854775807 ~ 9223372036854775807)";
const MIN_ALLOWED_INTEGER_VALUE: i64 = i64::MIN + 1;
const RANDOM_BOUNDED_RETRY_LIMIT: usize = 1024;
const U64_MODULUS: u128 = 1_u128 << 64;
#[derive(Clone, Copy)]
pub enum RandomNumberMode {
    Float,
    Integer,
}
pub fn generate_random_number(
    mode: RandomNumberMode,
    seed_modifier: u64,
    input_buffer: &mut String,
    out: &mut dyn Write,
    err: &mut dyn Write,
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
            let range_size_i128 = i128::from(max_value)
                .checked_sub(i128::from(min_value))
                .and_then(|range| range.checked_add(1))
                .ok_or("난수 범위 계산 실패")?;
            let range_size_raw = u64::try_from(range_size_i128)
                .map_err(|source| AppError::context("난수 범위 변환 실패", source))?;
            let range_size =
                NonZeroU64::new(range_size_raw).ok_or("난수 범위가 비어 있습니다.")?;
            let rand_offset = random_bounded(range_size, seed_modifier)?;
            let result_i128 = i128::from(min_value)
                .checked_add(i128::from(rand_offset))
                .ok_or("난수 결과 계산 실패")?;
            let result = i64::try_from(result_i128)
                .map_err(|source| AppError::context("난수 결과 변환 실패", source))?;
            writeln!(
                out,
                "무작위 정수({min_value} ~ {max_value}): {result} (0x{result:X})"
            )?;
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
            let random_u64 = get_hardware_random()? ^ seed_modifier;
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
                let span = NumericSub::sub(max_value, min_value);
                if !span.is_finite() {
                    return Err("실수 범위가 너무 커서 안전하게 계산할 수 없습니다.".into());
                }
                let value = scale.mul_add(span, min_value);
                if !value.is_finite() {
                    return Err("실수 난수 결과가 유한하지 않습니다.".into());
                }
                value
            };
            writeln!(out, "무작위 실수({min_value} ~ {max_value}): {result}")?;
        }
    }
    Ok(())
}
pub fn random_bounded(range_size: NonZeroU64, seed_mod: u64) -> Result<u64> {
    let range_size128 = u128::from(NonZeroU128::from(range_size));
    let threshold128 = U64_MODULUS
        .checked_sub(range_size128)
        .and_then(|value| value.checked_rem(range_size128))
        .ok_or("난수 범위 계산 실패")?;
    let threshold = u64::try_from(threshold128)
        .map_err(|source| AppError::context("난수 범위 변환 실패", source))?;
    for _ in 0..RANDOM_BOUNDED_RETRY_LIMIT {
        let product = u128::from(get_hardware_random()? ^ seed_mod)
            .checked_mul(range_size128)
            .ok_or("난수 곱셈 계산 실패")?;
        let low_bits = u64::try_from(product & u128::from(u64::MAX))
            .map_err(|source| AppError::context("난수 하위 비트 변환 실패", source))?;
        if low_bits >= threshold {
            return u64::try_from(product >> 64)
                .map_err(|source| AppError::context("난수 상위 비트 변환 실패", source));
        }
    }
    Err("bounded random rejection sampling 시도 횟수를 초과했습니다.".into())
}
