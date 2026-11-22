use super::{Result, TimeError};
use core::{error::Error, result::Result as CoreResult, time::Duration};
pub(super) const DAYS_PER_WEEK_I64: i64 = 7;
pub(super) const SECS_PER_DAY_I64: i64 = 86_400;
pub(super) const SECS_PER_DAY_U64: u64 = 86_400;
pub(super) const SECS_PER_HOUR_I64: i64 = 3_600;
pub(super) const SECS_PER_HOUR_U32: u32 = 3_600;
pub(super) const SECS_PER_MINUTE_I64: i64 = 60;
pub(super) const SECS_PER_MINUTE_U32: u32 = 60;
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
pub(super) const fn hms_from_second_of_day(second_of_day: u32) -> (u32, u32, u32) {
    (
        second_of_day.div_euclid(SECS_PER_HOUR_U32),
        second_of_day
            .rem_euclid(SECS_PER_HOUR_U32)
            .div_euclid(SECS_PER_MINUTE_U32),
        second_of_day.rem_euclid(SECS_PER_MINUTE_U32),
    )
}
pub(super) fn parse_hms(raw: &str) -> Option<(u32, u32, u32)> {
    let (hour_str, minute_second) = raw.split_once(':')?;
    let (minute_str, second_str) = minute_second.split_once(':')?;
    let parse_two = |part: &str| (part.len() == 2).then(|| parse_u32_digits(part)).flatten();
    Some((
        parse_two(hour_str)?,
        parse_two(minute_str)?,
        parse_two(second_str)?,
    ))
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
