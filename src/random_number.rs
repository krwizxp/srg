use super::{
    Result, TWO_POW_32_F64, U64_UNIT_SCALE, hardware_rng::get_hardware_random,
    input::parse_regular_f64, input::read_parsed_value, random_util::split_u64_to_u32_pair,
};
use core::ops::{Mul as _, Sub as _};
use std::io::{Error as IoError, Write};
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
            const MIN_ALLOWED_VALUE: i64 = i64::MIN + 1;
            writeln!(
                out,
                "\n무작위 정수 생성기(지원 범위: {MIN_ALLOWED_VALUE} ~ {max_i64})",
                max_i64 = i64::MAX
            )?;
            let min_value = loop {
                let value = read_parsed_value(
                    format_args!("최솟값을 입력해 주세요 ({MIN_ALLOWED_VALUE} 이상): "),
                    input_buffer,
                    out,
                    err,
                    "유효한 정수 형식이 아닙니다.",
                    |line| line.parse::<i64>().ok(),
                )?;
                if value >= MIN_ALLOWED_VALUE {
                    break value;
                }
                writeln!(err, "{MIN_ALLOWED_VALUE} 이상의 값을 입력해 주세요.")?;
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
            let range_size = max_value
                .wrapping_sub(min_value)
                .wrapping_add(1)
                .cast_unsigned();
            let rand_offset = if range_size == 0 {
                (get_hardware_random()? ^ seed_modifier).cast_signed()
            } else {
                random_bounded(range_size, seed_modifier)?.cast_signed()
            };
            let result = min_value.wrapping_add(rand_offset);
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
                "유효한 정규 실수 값을 입력해야 합니다 (NaN, 무한대, 비정규 값 제외).",
                parse_regular_f64,
            )?;
            let max_value = loop {
                let value = read_parsed_value(
                    format_args!("최댓값을 입력해 주세요: "),
                    input_buffer,
                    out,
                    err,
                    "유효한 정규 실수 값을 입력해야 합니다 (NaN, 무한대, 비정규 값 제외).",
                    parse_regular_f64,
                )?;
                if value >= min_value {
                    break value;
                }
                writeln!(err, "최댓값은 최솟값보다 크거나 같아야 합니다.")?;
            };
            let random_u64 = get_hardware_random()? ^ seed_modifier;
            let (upper_32, lower_32) = split_u64_to_u32_pair(random_u64);
            let scale = f64::from(upper_32)
                .mul_add(TWO_POW_32_F64, f64::from(lower_32))
                .mul(U64_UNIT_SCALE);
            let result = if min_value.to_bits() == max_value.to_bits() {
                min_value
            } else {
                scale.mul_add(max_value.sub(min_value), min_value)
            };
            writeln!(out, "무작위 실수({min_value} ~ {max_value}): {result}")?;
        }
    }
    Ok(())
}
pub fn random_bounded(range_size: u64, seed_mod: u64) -> Result<u64> {
    let threshold = range_size
        .wrapping_neg()
        .checked_rem(range_size)
        .ok_or_else(|| IoError::other("난수 범위 계산 실패"))?;
    let range_size128 = u128::from(range_size);
    loop {
        let Some(product) =
            u128::from(get_hardware_random()? ^ seed_mod).checked_mul(range_size128)
        else {
            continue;
        };
        let Ok(low_bits) = u64::try_from(product & u128::from(u64::MAX)) else {
            continue;
        };
        if low_bits >= threshold {
            let Ok(high_bits) = u64::try_from(product >> 64) else {
                continue;
            };
            return Ok(high_bits);
        }
    }
}
