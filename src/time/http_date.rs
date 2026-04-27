use super::{Result, TimeError, parse_result_with_context, parse_u32_digits};
use core::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
const SECS_PER_DAY_I64: i64 = 86_400;
const SECS_PER_DAY_U64: u64 = 86_400;
const SECS_PER_HOUR_I64: i64 = 3_600;
const SECS_PER_MINUTE_I64: i64 = 60;
const DAYS_PER_WEEK_I64: i64 = 7;
const DAYS_PER_100_YEARS_I64: i64 = 36_524;
const DAYS_PER_400_YEARS_I64: i64 = 146_097;
const DAYS_PER_4_YEARS_I64: i64 = 1_460;
const DAYS_PER_COMMON_YEAR_I64: i64 = 365;
const DAYS_UNTIL_UNIX_EPOCH_I64: i64 = 719_468;
const DECIMAL_BASE_I64: i64 = 10;
const DECIMAL_BASE_U32: u32 = 10;
const FEBRUARY_DAY_LEAP: u32 = 29;
const FEBRUARY_DAY_NORMAL: u32 = 28;
const IMF_FIXDATE_WEEKDAY_COMMA_INDEX: usize = 3;
const HTTP_MONTH_DAY_MAX: [u32; 12] = [
    31,
    FEBRUARY_DAY_NORMAL,
    31,
    30,
    31,
    30,
    31,
    31,
    30,
    31,
    30,
    31,
];
const LEAP_YEAR_CENTURY_DIVISOR_I32: i32 = 100;
const LEAP_YEAR_DIVISOR_I32: i32 = 4;
const LEAP_YEAR_ERA_DIVISOR_I32: i32 = 400;
const MARCH_BASE_MONTH_OFFSET_I64: i64 = 3;
const PRE_MARCH_MONTH_OFFSET_I64: i64 = 9;
const RFC850_CENTURY_CUTOFF_OFFSET_I64: i64 = 50;
const TWO_DIGIT_LEN: usize = 2;
const FOUR_DIGIT_LEN: usize = 4;
const MAX_HTTP_HOUR: u32 = 23;
const MAX_HTTP_MINUTE_OR_SECOND: u32 = 59;
const MARCH_MONTH_THRESHOLD: u32 = 2;
const MONTH_TERM_DIVISOR_I64: i64 = 5;
const MONTH_TERM_MULTIPLIER_I64: i64 = 153;
const MONTH_TERM_OFFSET_I64: i64 = 2;
const TIME_COMPONENT_LEN: usize = 8;
const UNIX_EPOCH_WEEKDAY_OFFSET_I64: i64 = 4;
#[derive(Clone, Copy)]
struct HttpDateComponents {
    day: u32,
    hour: u32,
    minute: u32,
    month: u32,
    second: u32,
    weekday: u32,
    year: i32,
}
impl HttpDateComponents {
    fn days_since_epoch(self) -> Result<i64> {
        const ERR_DAY: &str = "HTTP Date 파싱 실패: 날짜 값이 유효하지 않습니다.";
        const ERR_TIMESTAMP: &str = "HTTP Date 변환 실패: 타임스탬프 계산 중 범위 오류입니다.";
        let leap_year = (self.year.rem_euclid(LEAP_YEAR_DIVISOR_I32) == 0_i32
            && self.year.rem_euclid(LEAP_YEAR_CENTURY_DIVISOR_I32) != 0_i32)
            || self.year.rem_euclid(LEAP_YEAR_ERA_DIVISOR_I32) == 0_i32;
        let month_index = usize::try_from(
            self.month
                .checked_sub(1)
                .ok_or_else(|| TimeError::parse(ERR_DAY))?,
        )
        .ok()
        .ok_or_else(|| TimeError::parse(ERR_DAY))?;
        let max_day = HTTP_MONTH_DAY_MAX.get(month_index).copied().map_or_else(
            || Err(TimeError::parse(ERR_DAY)),
            |day| {
                if self.month == 2 && leap_year {
                    Ok(FEBRUARY_DAY_LEAP)
                } else {
                    Ok(day)
                }
            },
        )?;
        if self.day == 0 || self.day > max_day {
            return Err(TimeError::parse(ERR_DAY));
        }
        let year = i64::from(self.year);
        let adjusted_year = if self.month <= MARCH_MONTH_THRESHOLD {
            year.checked_sub(1_i64)
                .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))?
        } else {
            year
        };
        let (era, yoe) = (
            adjusted_year.div_euclid(LEAP_YEAR_ERA_DIVISOR_I32.into()),
            adjusted_year.rem_euclid(LEAP_YEAR_ERA_DIVISOR_I32.into()),
        );
        let shifted_month = if self.month > MARCH_MONTH_THRESHOLD {
            i64::from(self.month)
                .checked_sub(MARCH_BASE_MONTH_OFFSET_I64)
                .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))?
        } else {
            i64::from(self.month)
                .checked_add(PRE_MARCH_MONTH_OFFSET_I64)
                .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))?
        };
        let month_term = MONTH_TERM_MULTIPLIER_I64
            .checked_mul(shifted_month)
            .and_then(|value| value.checked_add(MONTH_TERM_OFFSET_I64))
            .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))?;
        let day_offset = i64::from(self.day)
            .checked_sub(1_i64)
            .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))?;
        let doy = month_term
            .checked_div(MONTH_TERM_DIVISOR_I64)
            .and_then(|value| value.checked_add(day_offset))
            .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))?;
        let doe = DAYS_PER_COMMON_YEAR_I64
            .checked_mul(yoe)
            .and_then(|value| value.checked_add(yoe.div_euclid(LEAP_YEAR_DIVISOR_I32.into())))
            .and_then(|value| {
                value.checked_sub(yoe.div_euclid(LEAP_YEAR_CENTURY_DIVISOR_I32.into()))
            })
            .and_then(|value| value.checked_add(doy))
            .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))?;
        era.checked_mul(DAYS_PER_400_YEARS_I64)
            .and_then(|value| value.checked_add(doe))
            .and_then(|value| value.checked_sub(DAYS_UNTIL_UNIX_EPOCH_I64))
            .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))
    }
    fn timestamp_secs(self, days: i64) -> Result<i64> {
        const ERR_TIMESTAMP: &str = "HTTP Date 변환 실패: 타임스탬프 계산 중 범위 오류입니다.";
        days.checked_mul(SECS_PER_DAY_I64)
            .and_then(|value| {
                let hour_secs = i64::from(self.hour).checked_mul(SECS_PER_HOUR_I64)?;
                value.checked_add(hour_secs)
            })
            .and_then(|value| {
                let minute_secs = i64::from(self.minute).checked_mul(SECS_PER_MINUTE_I64)?;
                value.checked_add(minute_secs)
            })
            .and_then(|value| value.checked_add(i64::from(self.second)))
            .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))
    }
    fn validate_weekday(self, days: i64) -> Result<()> {
        const ERR_WEEKDAY: &str = "HTTP Date 파싱 실패: 요일이 날짜와 일치하지 않습니다.";
        let actual_weekday = parse_result_with_context(
            u32::try_from(
                days.checked_add(UNIX_EPOCH_WEEKDAY_OFFSET_I64)
                    .ok_or_else(|| TimeError::parse(ERR_WEEKDAY))?
                    .rem_euclid(DAYS_PER_WEEK_I64),
            ),
            ERR_WEEKDAY,
        )?;
        if actual_weekday == self.weekday {
            Ok(())
        } else {
            Err(TimeError::parse(ERR_WEEKDAY))
        }
    }
}
fn parse_two_digits(d0: u8, d1: u8) -> Option<u32> {
    if !(d0.is_ascii_digit() && d1.is_ascii_digit()) {
        return None;
    }
    let tens = (d0.checked_sub(b'0'))?;
    let ones = (d1.checked_sub(b'0'))?;
    (u32::from(tens).checked_mul(DECIMAL_BASE_U32))?.checked_add(u32::from(ones))
}
fn parse_http_month(month_str: &str) -> Result<u32> {
    const ERR_MONTH: &str = "HTTP Date 파싱 실패: 알 수 없는 월 형식";
    match month_str {
        "Jan" => Ok(1),
        "Feb" => Ok(2),
        "Mar" => Ok(3),
        "Apr" => Ok(4),
        "May" => Ok(5),
        "Jun" => Ok(6),
        "Jul" => Ok(7),
        "Aug" => Ok(8),
        "Sep" => Ok(9),
        "Oct" => Ok(10),
        "Nov" => Ok(11),
        "Dec" => Ok(12),
        _ => Err(TimeError::parse(ERR_MONTH)),
    }
}
fn parse_http_weekday(weekday_str: &str) -> Option<u32> {
    match weekday_str {
        "Sun" | "Sunday" => Some(0),
        "Mon" | "Monday" => Some(1),
        "Tue" | "Tuesday" => Some(2),
        "Wed" | "Wednesday" => Some(3),
        "Thu" | "Thursday" => Some(4),
        "Fri" | "Friday" => Some(5),
        "Sat" | "Saturday" => Some(6),
        _ => None,
    }
}
fn parse_i32_token(raw: &str, err: &'static str) -> Result<i32> {
    let value = parse_u32_digits(raw).ok_or_else(|| TimeError::parse(err))?;
    parse_result_with_context(i32::try_from(value), err)
}
fn parse_u32_token(raw: &str, err: &'static str) -> Result<u32> {
    parse_u32_digits(raw).ok_or_else(|| TimeError::parse(err))
}
fn parse_with_time(
    day: u32,
    month: u32,
    year: i32,
    weekday: u32,
    time_token: &str,
) -> Result<HttpDateComponents> {
    const ERR_TIME_FMT: &str = "HTTP Date 파싱 실패: 시간 형식이 올바르지 않습니다 (HH:MM:SS)";
    const ERR_TIME_RANGE: &str = "HTTP Date 파싱 실패: 시간 값 범위가 올바르지 않습니다.";
    let &[
        hour_tens,
        hour_ones,
        b':',
        minute_tens,
        minute_ones,
        b':',
        second_tens,
        second_ones,
    ] = time_token
        .as_bytes()
        .as_array::<TIME_COMPONENT_LEN>()
        .ok_or_else(|| TimeError::parse(ERR_TIME_FMT))?
    else {
        return Err(TimeError::parse(ERR_TIME_FMT));
    };
    let (hour, minute, second) = (
        parse_two_digits(hour_tens, hour_ones).ok_or_else(|| TimeError::parse(ERR_TIME_FMT))?,
        parse_two_digits(minute_tens, minute_ones).ok_or_else(|| TimeError::parse(ERR_TIME_FMT))?,
        parse_two_digits(second_tens, second_ones).ok_or_else(|| TimeError::parse(ERR_TIME_FMT))?,
    );
    if hour > MAX_HTTP_HOUR
        || minute > MAX_HTTP_MINUTE_OR_SECOND
        || second > MAX_HTTP_MINUTE_OR_SECOND
    {
        return Err(TimeError::parse(ERR_TIME_RANGE));
    }
    Ok(HttpDateComponents {
        day,
        hour,
        minute,
        month,
        second,
        weekday,
        year,
    })
}
fn parse_http_date_rfc850(raw_date: &str) -> Result<HttpDateComponents> {
    const ERR_FORMAT: &str = "HTTP Date 파싱 실패: rfc850-date 형식이 아닙니다.";
    const ERR_NUM: &str = "HTTP Date 파싱 실패: rfc850-date 숫자 변환에 실패했습니다.";
    const ERR_YEAR: &str = "HTTP Date 파싱 실패: rfc850 2자리 연도 변환에 실패했습니다.";
    let mut parts = raw_date.split_ascii_whitespace();
    let (weekday_token, date_token, time_token, tz_token) = (
        parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?,
        parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?,
        parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?,
        parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?,
    );
    if parts.next().is_some() || tz_token != "GMT" {
        return Err(TimeError::parse(ERR_FORMAT));
    }
    let weekday_name = (weekday_token
        .strip_suffix(',')
        .ok_or_else(|| TimeError::parse(ERR_FORMAT)))?;
    let weekday = parse_http_weekday(weekday_name).ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let mut date_parts = date_token.split('-');
    let day_token = (date_parts
        .next()
        .ok_or_else(|| TimeError::parse(ERR_FORMAT)))?;
    let month_token = (date_parts
        .next()
        .ok_or_else(|| TimeError::parse(ERR_FORMAT)))?;
    let year2_token = (date_parts
        .next()
        .ok_or_else(|| TimeError::parse(ERR_FORMAT)))?;
    if date_parts.next().is_some()
        || day_token.len() != TWO_DIGIT_LEN
        || year2_token.len() != TWO_DIGIT_LEN
    {
        return Err(TimeError::parse(ERR_FORMAT));
    }
    let (day, month, year2) = (
        parse_u32_token(day_token, ERR_NUM)?,
        parse_http_month(month_token)?,
        parse_u32_token(year2_token, ERR_NUM)?,
    );
    let year_two_digits = (parse_result_with_context(i32::try_from(year2), ERR_YEAR))?;
    let current_year = {
        let day_index_i64 = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => i64::try_from(duration.as_secs().div_euclid(SECS_PER_DAY_U64))
                .unwrap_or_else(|_| i64::from(i32::MAX)),
            Err(err) => {
                let secs_before_epoch = err.duration().as_secs();
                let days_before_epoch = secs_before_epoch
                    .saturating_add(SECS_PER_DAY_U64 - 1)
                    .div_euclid(SECS_PER_DAY_U64);
                let days_before_epoch_i64 =
                    i64::try_from(days_before_epoch).unwrap_or_else(|_| i64::from(i32::MAX));
                days_before_epoch_i64.checked_neg().unwrap_or(i64::MIN)
            }
        };
        let day_index = i32::try_from(day_index_i64).unwrap_or_else(|_| {
            if day_index_i64.is_negative() {
                i32::MIN
            } else {
                i32::MAX
            }
        });
        civil_from_days(day_index).map_or_else(
            || {
                if day_index.is_negative() {
                    i32::MIN
                } else {
                    i32::MAX
                }
            },
            |date| date.0,
        )
    };
    let year = {
        let century_base = (i64::from(current_year)
            .div_euclid(i64::from(LEAP_YEAR_CENTURY_DIVISOR_I32))
            .checked_mul(i64::from(LEAP_YEAR_CENTURY_DIVISOR_I32))
            .ok_or_else(|| TimeError::parse(ERR_YEAR)))?;
        let mut expanded = (century_base
            .checked_add(i64::from(year_two_digits))
            .ok_or_else(|| TimeError::parse(ERR_YEAR)))?;
        let cutoff = (i64::from(current_year)
            .checked_add(RFC850_CENTURY_CUTOFF_OFFSET_I64)
            .ok_or_else(|| TimeError::parse(ERR_YEAR)))?;
        if expanded > cutoff {
            expanded = (expanded
                .checked_sub(i64::from(LEAP_YEAR_CENTURY_DIVISOR_I32))
                .ok_or_else(|| TimeError::parse(ERR_YEAR)))?;
        }
        parse_result_with_context(i32::try_from(expanded), ERR_YEAR)?
    };
    parse_with_time(day, month, year, weekday, time_token)
}
fn http_date_components_to_systemtime(components: HttpDateComponents) -> Result<SystemTime> {
    const ERR_UNIX_TIMESTAMP: &str = "HTTP Date 변환 실패: 유효하지 않은 타임스탬프입니다.";
    let days = components.days_since_epoch()?;
    components.validate_weekday(days)?;
    let timestamp_secs = components.timestamp_secs(days)?;
    let secs_i128 = i128::from(timestamp_secs);
    if secs_i128 >= 0 {
        let secs_u64 = (parse_result_with_context(u64::try_from(secs_i128), ERR_UNIX_TIMESTAMP))?;
        return UNIX_EPOCH
            .checked_add(Duration::from_secs(secs_u64))
            .ok_or_else(|| TimeError::parse(ERR_UNIX_TIMESTAMP));
    }
    let abs_i128 = (secs_i128
        .checked_abs()
        .ok_or_else(|| TimeError::parse(ERR_UNIX_TIMESTAMP)))?;
    let abs_secs = (parse_result_with_context(u64::try_from(abs_i128), ERR_UNIX_TIMESTAMP))?;
    UNIX_EPOCH
        .checked_sub(Duration::from_secs(abs_secs))
        .ok_or_else(|| TimeError::parse(ERR_UNIX_TIMESTAMP))
}
pub fn parse_http_date_to_systemtime(raw_date: &str) -> Result<SystemTime> {
    const ERR_FORMAT: &str = concat!(
        "HTTP Date 파싱 실패: RFC 9110 HTTP-date의 3개 형식",
        "(IMF-fixdate/rfc850/asctime) 중 하나가 아닙니다."
    );
    let raw_bytes = raw_date.as_bytes();
    let has_comma = raw_bytes.contains(&b',');
    let has_gmt = raw_bytes.windows(3).any(|window| window == b"GMT");
    if raw_bytes
        .get(IMF_FIXDATE_WEEKDAY_COMMA_INDEX)
        .is_some_and(|ch| *ch == b',')
    {
        const ERR_IMF_FORMAT: &str = "HTTP Date 파싱 실패: IMF-fixdate 형식이 아닙니다.";
        const ERR_IMF_NUM: &str = "HTTP Date 파싱 실패: IMF-fixdate 숫자 변환에 실패했습니다.";
        let mut parts = raw_date.split_ascii_whitespace();
        let weekday_token = (parts.next().ok_or_else(|| TimeError::parse(ERR_IMF_FORMAT)))?;
        let day_token = (parts.next().ok_or_else(|| TimeError::parse(ERR_IMF_FORMAT)))?;
        let month_token = (parts.next().ok_or_else(|| TimeError::parse(ERR_IMF_FORMAT)))?;
        let year_token = (parts.next().ok_or_else(|| TimeError::parse(ERR_IMF_FORMAT)))?;
        let time_token = (parts.next().ok_or_else(|| TimeError::parse(ERR_IMF_FORMAT)))?;
        let tz_token = (parts.next().ok_or_else(|| TimeError::parse(ERR_IMF_FORMAT)))?;
        if parts.next().is_some()
            || day_token.len() != TWO_DIGIT_LEN
            || year_token.len() != FOUR_DIGIT_LEN
            || tz_token != "GMT"
        {
            return Err(TimeError::parse(ERR_IMF_FORMAT));
        }
        let weekday_name = (weekday_token
            .strip_suffix(',')
            .ok_or_else(|| TimeError::parse(ERR_IMF_FORMAT)))?;
        let weekday =
            parse_http_weekday(weekday_name).ok_or_else(|| TimeError::parse(ERR_IMF_FORMAT))?;
        let day = parse_u32_token(day_token, ERR_IMF_NUM)?;
        let month = parse_http_month(month_token)?;
        let year = parse_i32_token(year_token, ERR_IMF_NUM)?;
        return parse_with_time(day, month, year, weekday, time_token)
            .and_then(http_date_components_to_systemtime);
    }
    if has_comma {
        return parse_http_date_rfc850(raw_date).and_then(http_date_components_to_systemtime);
    }
    if has_gmt {
        return parse_http_date_rfc850(raw_date).and_then(http_date_components_to_systemtime);
    }
    if raw_bytes.first().is_some_and(u8::is_ascii_alphabetic) {
        const ERR_ASCTIME_FORMAT: &str = "HTTP Date 파싱 실패: asctime-date 형식이 아닙니다.";
        const ERR_ASCTIME_NUM: &str = "HTTP Date 파싱 실패: asctime-date 숫자 변환에 실패했습니다.";
        let mut parts = raw_date.split_ascii_whitespace();
        let weekday_token = (parts
            .next()
            .ok_or_else(|| TimeError::parse(ERR_ASCTIME_FORMAT)))?;
        let month_token = (parts
            .next()
            .ok_or_else(|| TimeError::parse(ERR_ASCTIME_FORMAT)))?;
        let day_token = (parts
            .next()
            .ok_or_else(|| TimeError::parse(ERR_ASCTIME_FORMAT)))?;
        let time_token = (parts
            .next()
            .ok_or_else(|| TimeError::parse(ERR_ASCTIME_FORMAT)))?;
        let year_token = (parts
            .next()
            .ok_or_else(|| TimeError::parse(ERR_ASCTIME_FORMAT)))?;
        if parts.next().is_some()
            || !(1..=TWO_DIGIT_LEN).contains(&day_token.len())
            || year_token.len() != FOUR_DIGIT_LEN
        {
            return Err(TimeError::parse(ERR_ASCTIME_FORMAT));
        }
        let weekday = (parse_http_weekday(weekday_token)
            .ok_or_else(|| TimeError::parse(ERR_ASCTIME_FORMAT)))?;
        let day = parse_u32_token(day_token, ERR_ASCTIME_NUM)?;
        let month = parse_http_month(month_token)?;
        let year = parse_i32_token(year_token, ERR_ASCTIME_NUM)?;
        return parse_with_time(day, month, year, weekday, time_token)
            .and_then(http_date_components_to_systemtime);
    }
    Err(TimeError::parse(ERR_FORMAT))
}
pub fn civil_from_days(z: i32) -> Option<(i32, u32, u32)> {
    let shifted_days = (i64::from(z).checked_add(DAYS_UNTIL_UNIX_EPOCH_I64))?;
    let era = shifted_days.div_euclid(DAYS_PER_400_YEARS_I64);
    let doe = shifted_days.rem_euclid(DAYS_PER_400_YEARS_I64);
    let yoe_after_first = (doe.checked_sub((doe.checked_div(DAYS_PER_4_YEARS_I64))?))?;
    let yoe_after_second =
        (yoe_after_first.checked_add((doe.checked_div(DAYS_PER_100_YEARS_I64))?))?;
    let yoe_numerator =
        (yoe_after_second.checked_sub((doe.checked_div(DAYS_PER_400_YEARS_I64 - 1_i64))?))?;
    let yoe = (yoe_numerator.checked_div(DAYS_PER_COMMON_YEAR_I64))?;
    let y = (yoe.checked_add((era.checked_mul(i64::from(LEAP_YEAR_ERA_DIVISOR_I32)))?))?;
    let year_days = (DAYS_PER_COMMON_YEAR_I64.checked_mul(yoe))?;
    let leap_days = (yoe.checked_div(i64::from(LEAP_YEAR_DIVISOR_I32)))?;
    let skipped_centuries = (yoe.checked_div(i64::from(LEAP_YEAR_CENTURY_DIVISOR_I32)))?;
    let doy = (doe.checked_sub(
        (year_days
            .checked_add(leap_days)?
            .checked_sub(skipped_centuries))?,
    ))?;
    let mp = (MONTH_TERM_DIVISOR_I64
        .checked_mul(doy)?
        .checked_add(MONTH_TERM_OFFSET_I64)?
        .checked_div(MONTH_TERM_MULTIPLIER_I64))?;
    let month_term = (MONTH_TERM_MULTIPLIER_I64
        .checked_mul(mp)?
        .checked_add(MONTH_TERM_OFFSET_I64)?
        .checked_div(MONTH_TERM_DIVISOR_I64))?;
    let day = (u32::try_from((doy.checked_sub(month_term)?.checked_add(1_i64))?).ok())?;
    let month_i64 = if mp < DECIMAL_BASE_I64 {
        (mp.checked_add(MARCH_BASE_MONTH_OFFSET_I64))?
    } else {
        (mp.checked_sub(PRE_MARCH_MONTH_OFFSET_I64))?
    };
    let month = (u32::try_from(month_i64).ok())?;
    let year = (i32::try_from((y.checked_add(i64::from(month <= MARCH_MONTH_THRESHOLD)))?).ok())?;
    Some((year, month, day))
}
