use super::{
    CivilDate, Result, TimeError,
    util::{parse_result_with_context, parse_u32_digits},
};
use core::{
    str::{FromStr, SplitAsciiWhitespace},
    time::Duration,
};
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(test)]
mod tests;
const SECS_PER_DAY_I64: i64 = 86_400;
const SECS_PER_DAY_U64: u64 = 86_400;
const SECS_PER_HOUR_I64: i64 = 3_600;
const SECS_PER_MINUTE_I64: i64 = 60;
const DAYS_PER_WEEK_I64: i64 = 7;
const DAYS_PER_400_YEARS_I64: i64 = 146_097;
const DAYS_PER_COMMON_YEAR_I64: i64 = 365;
const DAYS_UNTIL_UNIX_EPOCH_I64: i64 = 719_468;
const DECIMAL_BASE_U32: u32 = 10;
const ERR_ASCTIME_FORMAT: &str = "HTTP Date 파싱 실패: asctime-date 형식이 아닙니다.";
const ERR_ASCTIME_NUM: &str = "HTTP Date 파싱 실패: asctime-date 숫자 변환에 실패했습니다.";
const ERR_HTTP_DATE_FORMAT: &str = concat!(
    "HTTP Date 파싱 실패: RFC 9110 HTTP-date의 3개 형식",
    "(IMF-fixdate/rfc850/asctime) 중 하나가 아닙니다."
);
const ERR_IMF_FORMAT: &str = "HTTP Date 파싱 실패: IMF-fixdate 형식이 아닙니다.";
const ERR_IMF_NUM: &str = "HTTP Date 파싱 실패: IMF-fixdate 숫자 변환에 실패했습니다.";
const ERR_LOCAL_YEAR: &str = "HTTP Date 파싱 실패: 현재 연도 계산에 실패했습니다.";
const ERR_RFC850_FORMAT: &str = "HTTP Date 파싱 실패: rfc850-date 형식이 아닙니다.";
const ERR_RFC850_NUM: &str = "HTTP Date 파싱 실패: rfc850-date 숫자 변환에 실패했습니다.";
const FEBRUARY_DAY_LEAP: u32 = 29;
const IMF_FIXDATE_WEEKDAY_COMMA_INDEX: usize = 3;
const LEAP_YEAR_CENTURY_DIVISOR_I32: i32 = 100;
const LEAP_YEAR_DIVISOR_I32: i32 = 4;
const LEAP_YEAR_ERA_DIVISOR_I32: i32 = 400;
const MARCH_BASE_MONTH_OFFSET_I64: i64 = 3;
const PRE_MARCH_MONTH_OFFSET_I64: i64 = 9;
const RFC850_CENTURY_CUTOFF_OFFSET: i32 = 50;
const TWO_DIGIT_LEN: usize = 2;
const FOUR_DIGIT_LEN: usize = 4;
const MAX_HTTP_HOUR: u32 = 23;
const MAX_HTTP_MINUTE: u32 = 59;
const MAX_HTTP_SECOND: u32 = 60;
const MARCH_MONTH_THRESHOLD: u32 = 2;
const MONTH_TERM_DIVISOR_I64: i64 = 5;
const MONTH_TERM_MULTIPLIER_I64: i64 = 153;
const MONTH_TERM_OFFSET_I64: i64 = 2;
const UNIX_EPOCH_WEEKDAY_OFFSET_I64: i64 = 4;
pub(super) struct HttpDate(pub SystemTime);
#[derive(Clone, Copy)]
enum HttpDateFormat {
    Asctime,
    ImfFixdate,
}
#[derive(Clone, Copy)]
struct HttpMonth {
    max_day: u32,
    number: u32,
}
impl FromStr for HttpDate {
    type Err = TimeError;
    fn from_str(raw_date: &str) -> Result<Self> {
        let raw_bytes = raw_date.as_bytes();
        if raw_bytes.get(IMF_FIXDATE_WEEKDAY_COMMA_INDEX) == Some(&b',') {
            return parse_date(raw_date, HttpDateFormat::ImfFixdate).map(Self);
        }
        if !(raw_date.contains(',') || raw_date.contains("GMT")) {
            if raw_bytes.first().is_some_and(u8::is_ascii_alphabetic) {
                return parse_date(raw_date, HttpDateFormat::Asctime).map(Self);
            }
            return Err(TimeError::parse(ERR_HTTP_DATE_FORMAT));
        }
        let (day_index_i64, current_second) = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => {
                let seconds = duration.as_secs();
                (
                    seconds.div_euclid(SECS_PER_DAY_U64).cast_signed(),
                    seconds.rem_euclid(SECS_PER_DAY_U64),
                )
            }
            Err(error) => {
                let duration = error.duration();
                let remainder = duration.as_secs().rem_euclid(SECS_PER_DAY_U64);
                let fraction = u64::from(duration.subsec_nanos() != 0);
                let days_before = duration
                    .as_secs()
                    .div_euclid(SECS_PER_DAY_U64)
                    .strict_add(u64::from(remainder != 0 || fraction != 0));
                (
                    parse_result_with_context(i64::try_from(days_before), ERR_LOCAL_YEAR)?
                        .strict_neg(),
                    SECS_PER_DAY_U64
                        .strict_sub(remainder.strict_add(fraction))
                        .rem_euclid(SECS_PER_DAY_U64),
                )
            }
        };
        let day_index = parse_result_with_context(i32::try_from(day_index_i64), ERR_LOCAL_YEAR)?;
        let current_year = civil_from_days(day_index).year;
        let mut parts = raw_date.split_ascii_whitespace();
        let (weekday_token, date_token, time_token, tz_token) = (
            next_date_part(&mut parts, ERR_RFC850_FORMAT)?,
            next_date_part(&mut parts, ERR_RFC850_FORMAT)?,
            next_date_part(&mut parts, ERR_RFC850_FORMAT)?,
            next_date_part(&mut parts, ERR_RFC850_FORMAT)?,
        );
        ensure_parts_exhausted(&mut parts, ERR_RFC850_FORMAT)?;
        (tz_token == "GMT").ok_or_else(|| TimeError::parse(ERR_RFC850_FORMAT))?;
        let weekday_name = weekday_token
            .strip_suffix(',')
            .ok_or_else(|| TimeError::parse(ERR_RFC850_FORMAT))?;
        let weekday =
            parse_http_weekday(weekday_name).ok_or_else(|| TimeError::parse(ERR_RFC850_FORMAT))?;
        let mut date_parts = date_token.split('-');
        let (Some(day_token), Some(month_token), Some(year2_token), None) = (
            date_parts.next(),
            date_parts.next(),
            date_parts.next(),
            date_parts.next(),
        ) else {
            return Err(TimeError::parse(ERR_RFC850_FORMAT));
        };
        (day_token.len() == TWO_DIGIT_LEN && year2_token.len() == TWO_DIGIT_LEN)
            .ok_or_else(|| TimeError::parse(ERR_RFC850_FORMAT))?;
        let (day, month, year2) = (
            parse_u32_token(day_token, ERR_RFC850_NUM)?,
            parse_http_month(month_token)?,
            parse_u32_token(year2_token, ERR_RFC850_NUM)?,
        );
        let century_base = current_year
            .div_euclid(LEAP_YEAR_CENTURY_DIVISOR_I32)
            .strict_mul(LEAP_YEAR_CENTURY_DIVISOR_I32);
        let mut year = century_base.strict_add(year2.cast_signed());
        let cutoff_year = current_year.strict_add(RFC850_CENTURY_CUTOFF_OFFSET);
        let past_cutoff = year > cutoff_year
            || year == cutoff_year && {
                let current_date = civil_from_days(day_index);
                let date_order = (month.number, day).cmp(&(current_date.month, current_date.day));
                if date_order.is_eq() {
                    let clock = [
                        time_token.get(..2),
                        time_token.get(3..5),
                        time_token.get(6..8),
                    ]
                    .map(|part| part.and_then(parse_u32_digits));
                    let [Some(hour), Some(minute), Some(second)] = clock else {
                        return Err(TimeError::parse(
                            "HTTP Date 파싱 실패: 시간 형식이 올바르지 않습니다 (HH:MM:SS)",
                        ));
                    };
                    let second_of_day = u64::from(hour)
                        .strict_mul(3600)
                        .strict_add(u64::from(minute).strict_mul(60))
                        .strict_add(u64::from(second));
                    second_of_day > current_second
                } else {
                    date_order.is_gt()
                }
            };
        if past_cutoff {
            year = year.strict_sub(LEAP_YEAR_CENTURY_DIVISOR_I32);
        }
        parse_http_date_time(day, month, year, weekday, time_token).map(Self)
    }
}
fn parse_two_digits(d0: u8, d1: u8) -> Option<u32> {
    (d0.is_ascii_digit() && d1.is_ascii_digit()).then(|| {
        u32::from(d0.strict_sub(b'0'))
            .strict_mul(DECIMAL_BASE_U32)
            .strict_add(u32::from(d1.strict_sub(b'0')))
    })
}
fn parse_http_month(month_str: &str) -> Result<HttpMonth> {
    const ERR_MONTH: &str = "HTTP Date 파싱 실패: 알 수 없는 월 형식";
    let (number, max_day) = match month_str {
        "Jan" => (1, 31),
        "Feb" => (2, 28),
        "Mar" => (3, 31),
        "Apr" => (4, 30),
        "May" => (5, 31),
        "Jun" => (6, 30),
        "Jul" => (7, 31),
        "Aug" => (8, 31),
        "Sep" => (9, 30),
        "Oct" => (10, 31),
        "Nov" => (11, 30),
        "Dec" => (12, 31),
        _ => return Err(TimeError::parse(ERR_MONTH)),
    };
    Ok(HttpMonth { max_day, number })
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
fn parse_u32_token(raw: &str, err: &'static str) -> Result<u32> {
    parse_u32_digits(raw).ok_or_else(|| TimeError::parse(err))
}
fn next_date_part<'part>(
    parts: &mut impl Iterator<Item = &'part str>,
    err: &'static str,
) -> Result<&'part str> {
    parts.next().ok_or_else(|| TimeError::parse(err))
}
fn ensure_parts_exhausted(parts: &mut SplitAsciiWhitespace<'_>, err: &'static str) -> Result<()> {
    parts.next().is_none().ok_or_else(|| TimeError::parse(err))
}
fn parse_http_date_time(
    day: u32,
    month: HttpMonth,
    year: i32,
    weekday: u32,
    time_token: &str,
) -> Result<SystemTime> {
    const ERR_DAY: &str = "HTTP Date 파싱 실패: 날짜 값이 유효하지 않습니다.";
    const ERR_TIME_FMT: &str = "HTTP Date 파싱 실패: 시간 형식이 올바르지 않습니다 (HH:MM:SS)";
    const ERR_TIME_RANGE: &str = "HTTP Date 파싱 실패: 시간 값 범위가 올바르지 않습니다.";
    const ERR_UNIX_TIMESTAMP: &str = "HTTP Date 변환 실패: 유효하지 않은 타임스탬프입니다.";
    const ERR_WEEKDAY: &str = "HTTP Date 파싱 실패: 요일이 날짜와 일치하지 않습니다.";
    let &[
        hour_tens,
        hour_ones,
        b':',
        minute_tens,
        minute_ones,
        b':',
        second_tens,
        second_ones,
    ] = time_token.as_bytes()
    else {
        return Err(TimeError::parse(ERR_TIME_FMT));
    };
    let (hour, minute, second) = (
        parse_two_digits(hour_tens, hour_ones).ok_or_else(|| TimeError::parse(ERR_TIME_FMT))?,
        parse_two_digits(minute_tens, minute_ones).ok_or_else(|| TimeError::parse(ERR_TIME_FMT))?,
        parse_two_digits(second_tens, second_ones).ok_or_else(|| TimeError::parse(ERR_TIME_FMT))?,
    );
    (hour <= MAX_HTTP_HOUR && minute <= MAX_HTTP_MINUTE && second <= MAX_HTTP_SECOND)
        .ok_or_else(|| TimeError::parse(ERR_TIME_RANGE))?;
    let leap_year = year.rem_euclid(LEAP_YEAR_DIVISOR_I32) == 0_i32
        && (year.rem_euclid(LEAP_YEAR_CENTURY_DIVISOR_I32) != 0_i32
            || year.rem_euclid(LEAP_YEAR_ERA_DIVISOR_I32) == 0_i32);
    let HttpMonth {
        mut max_day,
        number: month_number,
    } = month;
    if month_number == 2 && leap_year {
        max_day = FEBRUARY_DAY_LEAP;
    }
    (day != 0 && day <= max_day).ok_or_else(|| TimeError::parse(ERR_DAY))?;
    let adjusted_year =
        i64::from(year).strict_sub(i64::from(month_number <= MARCH_MONTH_THRESHOLD));
    let (era, year_of_era) = (
        adjusted_year.div_euclid(LEAP_YEAR_ERA_DIVISOR_I32.into()),
        adjusted_year.rem_euclid(LEAP_YEAR_ERA_DIVISOR_I32.into()),
    );
    let shifted_month = if month_number > MARCH_MONTH_THRESHOLD {
        i64::from(month_number).strict_sub(MARCH_BASE_MONTH_OFFSET_I64)
    } else {
        i64::from(month_number).strict_add(PRE_MARCH_MONTH_OFFSET_I64)
    };
    let month_term = MONTH_TERM_MULTIPLIER_I64
        .strict_mul(shifted_month)
        .strict_add(MONTH_TERM_OFFSET_I64);
    let day_of_year = month_term
        .div_euclid(MONTH_TERM_DIVISOR_I64)
        .strict_add(i64::from(day).strict_sub(1_i64));
    let day_of_era = DAYS_PER_COMMON_YEAR_I64
        .strict_mul(year_of_era)
        .strict_add(year_of_era.div_euclid(LEAP_YEAR_DIVISOR_I32.into()))
        .strict_sub(year_of_era.div_euclid(LEAP_YEAR_CENTURY_DIVISOR_I32.into()))
        .strict_add(day_of_year);
    let days = era
        .strict_mul(DAYS_PER_400_YEARS_I64)
        .strict_add(day_of_era)
        .strict_sub(DAYS_UNTIL_UNIX_EPOCH_I64);
    let actual_weekday = days
        .rem_euclid(DAYS_PER_WEEK_I64)
        .strict_add(UNIX_EPOCH_WEEKDAY_OFFSET_I64)
        .rem_euclid(DAYS_PER_WEEK_I64);
    (actual_weekday == i64::from(weekday)).ok_or_else(|| TimeError::parse(ERR_WEEKDAY))?;
    let timestamp_secs = days
        .strict_mul(SECS_PER_DAY_I64)
        .strict_add(i64::from(hour).strict_mul(SECS_PER_HOUR_I64))
        .strict_add(i64::from(minute).strict_mul(SECS_PER_MINUTE_I64))
        // SystemTime has no separate leap-second value; :60 normalizes to the next second.
        .strict_add(i64::from(second));
    let duration = Duration::from_secs(timestamp_secs.unsigned_abs());
    if timestamp_secs >= 0 {
        return UNIX_EPOCH
            .checked_add(duration)
            .ok_or_else(|| TimeError::parse(ERR_UNIX_TIMESTAMP));
    }
    UNIX_EPOCH
        .checked_sub(duration)
        .ok_or_else(|| TimeError::parse(ERR_UNIX_TIMESTAMP))
}
fn parse_date(raw: &str, format: HttpDateFormat) -> Result<SystemTime> {
    let (format_error, numeric_error) = match format {
        HttpDateFormat::Asctime => (ERR_ASCTIME_FORMAT, ERR_ASCTIME_NUM),
        HttpDateFormat::ImfFixdate => (ERR_IMF_FORMAT, ERR_IMF_NUM),
    };
    let mut parts = raw.split_ascii_whitespace();
    let weekday_token = next_date_part(&mut parts, format_error)?;
    let first = next_date_part(&mut parts, format_error)?;
    let second = next_date_part(&mut parts, format_error)?;
    let third = next_date_part(&mut parts, format_error)?;
    let fourth = next_date_part(&mut parts, format_error)?;
    let (month_token, day_token, time_token, year_token, weekday_name) = match format {
        HttpDateFormat::Asctime => (first, second, third, fourth, weekday_token),
        HttpDateFormat::ImfFixdate => {
            (next_date_part(&mut parts, format_error)? == "GMT")
                .ok_or_else(|| TimeError::parse(format_error))?;
            let weekday_name = weekday_token
                .strip_suffix(',')
                .ok_or_else(|| TimeError::parse(format_error))?;
            (second, first, fourth, third, weekday_name)
        }
    };
    ensure_parts_exhausted(&mut parts, format_error)?;
    let valid_day_length = day_token.len() == TWO_DIGIT_LEN
        || matches!(format, HttpDateFormat::Asctime) && day_token.len() == 1;
    (valid_day_length && year_token.len() == FOUR_DIGIT_LEN)
        .ok_or_else(|| TimeError::parse(format_error))?;
    let weekday = parse_http_weekday(weekday_name).ok_or_else(|| TimeError::parse(format_error))?;
    let day = parse_u32_token(day_token, numeric_error)?;
    let month = parse_http_month(month_token)?;
    let year = parse_u32_token(year_token, numeric_error)?.cast_signed();
    parse_http_date_time(day, month, year, weekday, time_token)
}
pub(super) fn civil_from_days(z: i32) -> CivilDate {
    const DAYS_PER_400_YEARS: i32 = 146_097;
    const DAYS_UNTIL_UNIX_EPOCH: i32 = 719_468;
    let z_remainder = z.rem_euclid(DAYS_PER_400_YEARS);
    let shifted_remainder = z_remainder.strict_add(DAYS_UNTIL_UNIX_EPOCH);
    let era = z
        .div_euclid(DAYS_PER_400_YEARS)
        .strict_add(shifted_remainder.div_euclid(DAYS_PER_400_YEARS));
    let doe = shifted_remainder.rem_euclid(DAYS_PER_400_YEARS);
    let yoe = doe
        .strict_sub(doe.div_euclid(1_460))
        .strict_add(doe.div_euclid(36_524))
        .strict_sub(doe.div_euclid(DAYS_PER_400_YEARS.strict_sub(1)))
        .div_euclid(365);
    let y = yoe.strict_add(era.strict_mul(LEAP_YEAR_ERA_DIVISOR_I32));
    let doy = doe.strict_sub(
        365_i32
            .strict_mul(yoe)
            .strict_add(yoe.div_euclid(LEAP_YEAR_DIVISOR_I32))
            .strict_sub(yoe.div_euclid(LEAP_YEAR_CENTURY_DIVISOR_I32)),
    );
    let mp = 5_i32.strict_mul(doy).strict_add(2).div_euclid(153);
    let month_term = 153_i32.strict_mul(mp).strict_add(2).div_euclid(5);
    let day = doy.strict_sub(month_term).strict_add(1).cast_unsigned();
    let month = mp
        .strict_add(if mp < 10_i32 { 3 } else { -9 })
        .cast_unsigned();
    let year = y.strict_add(i32::from(month <= MARCH_MONTH_THRESHOLD));
    CivilDate { day, month, year }
}
