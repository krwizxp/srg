use super::{Result, TimeError};
use core::{error::Error, result::Result as CoreResult, time::Duration};
#[derive(Clone, Copy)]
pub(super) enum NewSampleWeight {
    SeventyPercent,
    ThirtyPercent,
}
pub(super) const fn blend_rtt(
    old_value: Duration,
    new_value: Duration,
    weight: NewSampleWeight,
) -> Duration {
    let (old_weight, new_weight_value) = match weight {
        NewSampleWeight::SeventyPercent => (3_u128, 7_u128),
        NewSampleWeight::ThirtyPercent => (7_u128, 3_u128),
    };
    let weighted_sum = old_value
        .as_nanos()
        .saturating_mul(old_weight)
        .saturating_add(new_value.as_nanos().saturating_mul(new_weight_value));
    Duration::from_nanos_u128(weighted_sum.div_euclid(10))
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
