use super::{Result, TimeError};
use core::{error::Error, result::Result as CoreResult};
pub(super) fn blend_weighted_nanos(
    old_value: u128,
    new_value: u128,
    new_weight: u32,
    total_weight: u32,
) -> u128 {
    let Some(old_weight) = total_weight.checked_sub(new_weight) else {
        return new_value;
    };
    let Some(weighted_old) = old_value.checked_mul(u128::from(old_weight)) else {
        return new_value;
    };
    let Some(weighted_new) = new_value.checked_mul(u128::from(new_weight)) else {
        return new_value;
    };
    let Some(weighted_sum) = weighted_old.checked_add(weighted_new) else {
        return new_value;
    };
    let Some(weighted_average) = weighted_sum.checked_div(u128::from(total_weight)) else {
        return new_value;
    };
    weighted_average
}
pub(super) fn parse_result_with_context<T, E>(
    result: CoreResult<T, E>,
    context: &'static str,
) -> Result<T>
where
    E: Error + Send + Sync + 'static,
{
    result.map_err(|err| TimeError::parse_with_source(context, err))
}
pub(super) fn parse_u32_digits(raw: &str) -> Option<u32> {
    if raw.is_empty() {
        return None;
    }
    raw.bytes().try_fold(0_u32, |value, byte| {
        if !byte.is_ascii_digit() {
            return None;
        }
        let digit = byte.wrapping_sub(b'0');
        value.checked_mul(10)?.checked_add(u32::from(digit))
    })
}
