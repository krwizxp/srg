use super::{Result, TimeError};
use core::{error::Error, result::Result as CoreResult, time::Duration};
pub(super) const fn blend_rtt<const NEW_WEIGHT: u128>(
    old_value: Duration,
    new_value: Duration,
) -> Duration {
    let weighted_sum = old_value
        .as_nanos()
        .strict_mul(10_u128.strict_sub(NEW_WEIGHT))
        .strict_add(new_value.as_nanos().strict_mul(NEW_WEIGHT));
    Duration::from_nanos_u128(weighted_sum.div_euclid(10))
}
pub(super) fn parse_result_with_context<T>(
    result: CoreResult<T, impl Error + Send + Sync + 'static>,
    context: &'static str,
) -> Result<T> {
    result.map_err(|err| TimeError::parse_with_source(context, err))
}
pub(super) fn parse_u32_digits(raw: &str) -> Option<u32> {
    if raw.starts_with('+') {
        return None;
    }
    raw.parse::<u32>().ok()
}
