use super::{
    CivilDate, Result, TimeError,
    util::{parse_result_with_context, parse_u32_digits},
};
use core::{
    str::{FromStr, SplitAsciiWhitespace},
    time::Duration,
};
use std::time::{SystemTime, UNIX_EPOCH};
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
const RFC850_CENTURY_CUTOFF_OFFSET: i32 = 50;
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
pub(super) struct HttpDate(pub SystemTime);
#[derive(Clone, Copy)]
enum HttpDateFormat {
    Asctime,
    ImfFixdate,
    Rfc850 { current_year: i32 },
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
        let day_index_i64 = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => duration
                .as_secs()
                .div_euclid(SECS_PER_DAY_U64)
                .cast_signed(),
            Err(err) => {
                let secs_before_epoch = err.duration().as_secs();
                let days_before_epoch = secs_before_epoch.div_ceil(SECS_PER_DAY_U64);
                days_before_epoch.cast_signed().wrapping_neg()
            }
        };
        let day_index = parse_result_with_context(i32::try_from(day_index_i64), ERR_LOCAL_YEAR)?;
        let current_year = civil_from_days(day_index).year;
        parse_date(raw_date, HttpDateFormat::Rfc850 { current_year }).map(Self)
    }
}
fn parse_two_digits(d0: u8, d1: u8) -> Option<u32> {
    if !(d0.is_ascii_digit() && d1.is_ascii_digit()) {
        return None;
    }
    let tens = d0.wrapping_sub(b'0');
    let ones = d1.wrapping_sub(b'0');
    Some(
        u32::from(tens)
            .wrapping_mul(DECIMAL_BASE_U32)
            .wrapping_add(u32::from(ones)),
    )
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
fn next_date_part<'part>(
    parts: &mut impl Iterator<Item = &'part str>,
    err: &'static str,
) -> Result<&'part str> {
    parts.next().ok_or_else(|| TimeError::parse(err))
}
fn strip_date_suffix<'date>(
    raw: &'date str,
    suffix: char,
    err: &'static str,
) -> Result<&'date str> {
    raw.strip_suffix(suffix)
        .ok_or_else(|| TimeError::parse(err))
}
fn ensure_parts_exhausted(parts: &mut SplitAsciiWhitespace<'_>, err: &'static str) -> Result<()> {
    if parts.next().is_none() {
        Ok(())
    } else {
        Err(TimeError::parse(err))
    }
}
fn parse_http_date_time(
    day: u32,
    month: u32,
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
    let leap_year = (year.rem_euclid(LEAP_YEAR_DIVISOR_I32) == 0_i32
        && year.rem_euclid(LEAP_YEAR_CENTURY_DIVISOR_I32) != 0_i32)
        || year.rem_euclid(LEAP_YEAR_ERA_DIVISOR_I32) == 0_i32;
    let month_index = usize::try_from(
        month
            .checked_sub(1)
            .ok_or_else(|| TimeError::parse(ERR_DAY))?,
    )
    .ok()
    .ok_or_else(|| TimeError::parse(ERR_DAY))?;
    let Some(mut max_day) = HTTP_MONTH_DAY_MAX.get(month_index).copied() else {
        return Err(TimeError::parse(ERR_DAY));
    };
    if month == 2 && leap_year {
        max_day = FEBRUARY_DAY_LEAP;
    }
    if day == 0 || day > max_day {
        return Err(TimeError::parse(ERR_DAY));
    }
    let year_i64 = i64::from(year);
    let adjusted_year = if month <= MARCH_MONTH_THRESHOLD {
        year_i64.wrapping_sub(1_i64)
    } else {
        year_i64
    };
    let (era, year_of_era) = (
        adjusted_year.div_euclid(LEAP_YEAR_ERA_DIVISOR_I32.into()),
        adjusted_year.rem_euclid(LEAP_YEAR_ERA_DIVISOR_I32.into()),
    );
    let shifted_month = if month > MARCH_MONTH_THRESHOLD {
        i64::from(month).wrapping_sub(MARCH_BASE_MONTH_OFFSET_I64)
    } else {
        i64::from(month).wrapping_add(PRE_MARCH_MONTH_OFFSET_I64)
    };
    let month_term = MONTH_TERM_MULTIPLIER_I64
        .wrapping_mul(shifted_month)
        .wrapping_add(MONTH_TERM_OFFSET_I64);
    let day_of_year = month_term
        .div_euclid(MONTH_TERM_DIVISOR_I64)
        .wrapping_add(i64::from(day).wrapping_sub(1_i64));
    let day_of_era = DAYS_PER_COMMON_YEAR_I64
        .wrapping_mul(year_of_era)
        .wrapping_add(year_of_era.div_euclid(LEAP_YEAR_DIVISOR_I32.into()))
        .wrapping_sub(year_of_era.div_euclid(LEAP_YEAR_CENTURY_DIVISOR_I32.into()))
        .wrapping_add(day_of_year);
    let days = era
        .wrapping_mul(DAYS_PER_400_YEARS_I64)
        .wrapping_add(day_of_era)
        .wrapping_sub(DAYS_UNTIL_UNIX_EPOCH_I64);
    let actual_weekday = parse_result_with_context(
        u32::try_from(
            days.checked_add(UNIX_EPOCH_WEEKDAY_OFFSET_I64)
                .ok_or_else(|| TimeError::parse(ERR_WEEKDAY))?
                .rem_euclid(DAYS_PER_WEEK_I64),
        ),
        ERR_WEEKDAY,
    )?;
    if actual_weekday != weekday {
        return Err(TimeError::parse(ERR_WEEKDAY));
    }
    let timestamp_secs = days
        .wrapping_mul(SECS_PER_DAY_I64)
        .wrapping_add(i64::from(hour).wrapping_mul(SECS_PER_HOUR_I64))
        .wrapping_add(i64::from(minute).wrapping_mul(SECS_PER_MINUTE_I64))
        .wrapping_add(i64::from(second));
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
    match format {
        HttpDateFormat::Asctime => {
            let mut parts = raw.split_ascii_whitespace();
            let weekday_token = next_date_part(&mut parts, ERR_ASCTIME_FORMAT)?;
            let month_token = next_date_part(&mut parts, ERR_ASCTIME_FORMAT)?;
            let day_token = next_date_part(&mut parts, ERR_ASCTIME_FORMAT)?;
            let time_token = next_date_part(&mut parts, ERR_ASCTIME_FORMAT)?;
            let year_token = next_date_part(&mut parts, ERR_ASCTIME_FORMAT)?;
            ensure_parts_exhausted(&mut parts, ERR_ASCTIME_FORMAT)?;
            if !(1..=TWO_DIGIT_LEN).contains(&day_token.len()) || year_token.len() != FOUR_DIGIT_LEN
            {
                return Err(TimeError::parse(ERR_ASCTIME_FORMAT));
            }
            let weekday = parse_http_weekday(weekday_token)
                .ok_or_else(|| TimeError::parse(ERR_ASCTIME_FORMAT))?;
            let day = parse_u32_token(day_token, ERR_ASCTIME_NUM)?;
            let month = parse_http_month(month_token)?;
            let year = parse_i32_token(year_token, ERR_ASCTIME_NUM)?;
            parse_http_date_time(day, month, year, weekday, time_token)
        }
        HttpDateFormat::ImfFixdate => {
            let mut parts = raw.split_ascii_whitespace();
            let weekday_token = next_date_part(&mut parts, ERR_IMF_FORMAT)?;
            let day_token = next_date_part(&mut parts, ERR_IMF_FORMAT)?;
            let month_token = next_date_part(&mut parts, ERR_IMF_FORMAT)?;
            let year_token = next_date_part(&mut parts, ERR_IMF_FORMAT)?;
            let time_token = next_date_part(&mut parts, ERR_IMF_FORMAT)?;
            let tz_token = next_date_part(&mut parts, ERR_IMF_FORMAT)?;
            ensure_parts_exhausted(&mut parts, ERR_IMF_FORMAT)?;
            if day_token.len() != TWO_DIGIT_LEN
                || year_token.len() != FOUR_DIGIT_LEN
                || tz_token != "GMT"
            {
                return Err(TimeError::parse(ERR_IMF_FORMAT));
            }
            let weekday_name = strip_date_suffix(weekday_token, ',', ERR_IMF_FORMAT)?;
            let weekday =
                parse_http_weekday(weekday_name).ok_or_else(|| TimeError::parse(ERR_IMF_FORMAT))?;
            let day = parse_u32_token(day_token, ERR_IMF_NUM)?;
            let month = parse_http_month(month_token)?;
            let year = parse_i32_token(year_token, ERR_IMF_NUM)?;
            parse_http_date_time(day, month, year, weekday, time_token)
        }
        HttpDateFormat::Rfc850 { current_year } => {
            let mut parts = raw.split_ascii_whitespace();
            let (weekday_token, date_token, time_token, tz_token) = (
                next_date_part(&mut parts, ERR_RFC850_FORMAT)?,
                next_date_part(&mut parts, ERR_RFC850_FORMAT)?,
                next_date_part(&mut parts, ERR_RFC850_FORMAT)?,
                next_date_part(&mut parts, ERR_RFC850_FORMAT)?,
            );
            ensure_parts_exhausted(&mut parts, ERR_RFC850_FORMAT)?;
            if tz_token != "GMT" {
                return Err(TimeError::parse(ERR_RFC850_FORMAT));
            }
            let weekday_name = strip_date_suffix(weekday_token, ',', ERR_RFC850_FORMAT)?;
            let weekday = parse_http_weekday(weekday_name)
                .ok_or_else(|| TimeError::parse(ERR_RFC850_FORMAT))?;
            let mut date_parts = date_token.split('-');
            let (Some(day_token), Some(month_token), Some(year2_token), None) = (
                date_parts.next(),
                date_parts.next(),
                date_parts.next(),
                date_parts.next(),
            ) else {
                return Err(TimeError::parse(ERR_RFC850_FORMAT));
            };
            if day_token.len() != TWO_DIGIT_LEN || year2_token.len() != TWO_DIGIT_LEN {
                return Err(TimeError::parse(ERR_RFC850_FORMAT));
            }
            let (day, month, year2) = (
                parse_u32_token(day_token, ERR_RFC850_NUM)?,
                parse_http_month(month_token)?,
                parse_u32_token(year2_token, ERR_RFC850_NUM)?,
            );
            let century_base = current_year
                .div_euclid(LEAP_YEAR_CENTURY_DIVISOR_I32)
                .wrapping_mul(LEAP_YEAR_CENTURY_DIVISOR_I32);
            let mut year = century_base.wrapping_add(year2.cast_signed());
            let cutoff = current_year.wrapping_add(RFC850_CENTURY_CUTOFF_OFFSET);
            if year > cutoff {
                year = year.wrapping_sub(LEAP_YEAR_CENTURY_DIVISOR_I32);
            }
            parse_http_date_time(day, month, year, weekday, time_token)
        }
    }
}
pub(super) fn civil_from_days(z: i32) -> CivilDate {
    const DAYS_PER_400_YEARS: i32 = 146_097;
    const DAYS_UNTIL_UNIX_EPOCH: i32 = 719_468;
    let z_remainder = z.rem_euclid(DAYS_PER_400_YEARS);
    let shifted_remainder = z_remainder.wrapping_add(DAYS_UNTIL_UNIX_EPOCH);
    let era = z
        .div_euclid(DAYS_PER_400_YEARS)
        .wrapping_add(shifted_remainder.div_euclid(DAYS_PER_400_YEARS));
    let doe = shifted_remainder.rem_euclid(DAYS_PER_400_YEARS);
    let yoe = doe
        .wrapping_sub(doe.div_euclid(1_460))
        .wrapping_add(doe.div_euclid(36_524))
        .wrapping_sub(doe.div_euclid(DAYS_PER_400_YEARS.wrapping_sub(1)))
        .div_euclid(365);
    let y = yoe.wrapping_add(era.wrapping_mul(LEAP_YEAR_ERA_DIVISOR_I32));
    let doy = doe.wrapping_sub(
        365_i32
            .wrapping_mul(yoe)
            .wrapping_add(yoe.div_euclid(LEAP_YEAR_DIVISOR_I32))
            .wrapping_sub(yoe.div_euclid(LEAP_YEAR_CENTURY_DIVISOR_I32)),
    );
    let mp = 5_i32.wrapping_mul(doy).wrapping_add(2).div_euclid(153);
    let month_term = 153_i32.wrapping_mul(mp).wrapping_add(2).div_euclid(5);
    let day = doy.wrapping_sub(month_term).wrapping_add(1).cast_unsigned();
    let month_i32 = if mp < 10_i32 {
        mp.wrapping_add(3)
    } else {
        mp.wrapping_sub(9)
    };
    let month = month_i32.cast_unsigned();
    let year = y.wrapping_add(i32::from(month <= MARCH_MONTH_THRESHOLD));
    CivilDate { day, month, year }
}
