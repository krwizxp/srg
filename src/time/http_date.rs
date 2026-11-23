use super::{
    CivilDate, Result, TimeError,
    util::{
        DAYS_PER_WEEK_I64, SECS_PER_DAY_I64, SECS_PER_DAY_U64, SECS_PER_HOUR_I64,
        SECS_PER_MINUTE_I64, parse_hms, parse_u32_digits,
    },
};
use core::{
    str::{FromStr, SplitAsciiWhitespace},
    time::Duration,
};
use std::time::{SystemTime, UNIX_EPOCH};
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
const ERR_TIME_FMT: &str = "HTTP Date 파싱 실패: 시간 형식이 올바르지 않습니다 (HH:MM:SS)";
const FOUR_DIGIT_LEN: usize = 4;
const IMF_FIXDATE_WEEKDAY_COMMA_INDEX: usize = 3;
const LEAP_YEAR_CENTURY_DIVISOR_I32: i32 = 100;
const RFC850_CENTURY_CUTOFF_OFFSET: i32 = 50;
const TWO_DIGIT_LEN: usize = 2;
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
                    i64::try_from(days_before)
                        .map_err(|source| TimeError::parse_with_source(ERR_LOCAL_YEAR, source))?
                        .strict_neg(),
                    SECS_PER_DAY_U64
                        .strict_sub(remainder.strict_add(fraction))
                        .rem_euclid(SECS_PER_DAY_U64),
                )
            }
        };
        let day_index = i32::try_from(day_index_i64)
            .map_err(|source| TimeError::parse_with_source(ERR_LOCAL_YEAR, source))?;
        let current_date = civil_from_days(day_index);
        let current_year = current_date.year;
        let mut parts = raw_date.split_ascii_whitespace();
        let (weekday_token, date_token, time_token, tz_token) = (
            next_date_part(&mut parts, ERR_RFC850_FORMAT)?,
            next_date_part(&mut parts, ERR_RFC850_FORMAT)?,
            next_date_part(&mut parts, ERR_RFC850_FORMAT)?,
            next_date_part(&mut parts, ERR_RFC850_FORMAT)?,
        );
        ensure_parts_exhausted(&mut parts, ERR_RFC850_FORMAT)?;
        (tz_token == "GMT").ok_or(ERR_RFC850_FORMAT)?;
        let weekday_name = weekday_token.strip_suffix(',').ok_or(ERR_RFC850_FORMAT)?;
        let weekday = parse_http_weekday(weekday_name).ok_or(ERR_RFC850_FORMAT)?;
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
            .ok_or(ERR_RFC850_FORMAT)?;
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
                let date_order = (month.number, day).cmp(&(current_date.month, current_date.day));
                if date_order.is_eq() {
                    let (hour, minute, second) = parse_hms(time_token).ok_or(ERR_TIME_FMT)?;
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
fn parse_http_month(month_str: &str) -> Result<HttpMonth> {
    const NAMES: &[u8; 36] = b"JanFebMarAprMayJunJulAugSepOctNovDec";
    const DAYS: [u8; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let idx = NAMES
        .as_chunks::<3>()
        .0
        .iter()
        .position(|name| name.as_slice() == month_str.as_bytes())
        .ok_or("HTTP Date 파싱 실패: 알 수 없는 월 형식")?;
    let max_day = *DAYS
        .get(idx)
        .ok_or("HTTP Date 파싱 실패: 알 수 없는 월 형식")?;
    Ok(HttpMonth {
        max_day: u32::from(max_day),
        number: u32::try_from(idx).unwrap_or(0).strict_add(1),
    })
}
fn parse_http_weekday(weekday_str: &str) -> Option<u32> {
    const NAMES: [&str; 7] = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];
    NAMES.iter().enumerate().find_map(|(idx, &name)| {
        (name == weekday_str || name.get(..3) == Some(weekday_str))
            .then(|| u32::try_from(idx).unwrap_or(0))
    })
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
    const ERR_TIME_RANGE: &str = "HTTP Date 파싱 실패: 시간 값 범위가 올바르지 않습니다.";
    const ERR_UNIX_TIMESTAMP: &str = "HTTP Date 변환 실패: 유효하지 않은 타임스탬프입니다.";
    const ERR_WEEKDAY: &str = "HTTP Date 파싱 실패: 요일이 날짜와 일치하지 않습니다.";
    let (hour, minute, second) = parse_hms(time_token).ok_or(ERR_TIME_FMT)?;
    (hour <= 23_u32 && minute <= 59_u32 && second <= 60_u32).ok_or(ERR_TIME_RANGE)?;
    let leap_year = year.rem_euclid(4_i32) == 0_i32
        && (year.rem_euclid(LEAP_YEAR_CENTURY_DIVISOR_I32) != 0_i32
            || year.rem_euclid(400_i32) == 0_i32);
    let HttpMonth {
        mut max_day,
        number: month_number,
    } = month;
    if month_number == 2_u32 && leap_year {
        max_day = 29_u32;
    }
    (day != 0 && day <= max_day).ok_or(ERR_DAY)?;
    let adjusted_year = i64::from(year).strict_sub(i64::from(month_number <= 2_u32));
    let (era, year_of_era) = (
        adjusted_year.div_euclid(400_i64),
        adjusted_year.rem_euclid(400_i64),
    );
    let shifted_month = if month_number > 2_u32 {
        i64::from(month_number).strict_sub(3_i64)
    } else {
        i64::from(month_number).strict_add(9_i64)
    };
    let day_of_year = 153_i64
        .strict_mul(shifted_month)
        .strict_add(2_i64)
        .div_euclid(5_i64)
        .strict_add(i64::from(day).strict_sub(1_i64));
    let day_of_era = 365_i64
        .strict_mul(year_of_era)
        .strict_add(year_of_era.div_euclid(4_i64))
        .strict_sub(year_of_era.div_euclid(100_i64))
        .strict_add(day_of_year);
    let days = era
        .strict_mul(146_097_i64)
        .strict_add(day_of_era)
        .strict_sub(719_468_i64);
    let actual_weekday = days.strict_add(4_i64).rem_euclid(DAYS_PER_WEEK_I64);
    (actual_weekday == i64::from(weekday)).ok_or(ERR_WEEKDAY)?;
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
            (next_date_part(&mut parts, format_error)? == "GMT").ok_or(format_error)?;
            let weekday_name = weekday_token.strip_suffix(',').ok_or(format_error)?;
            (second, first, fourth, third, weekday_name)
        }
    };
    ensure_parts_exhausted(&mut parts, format_error)?;
    let valid_day_length = day_token.len() == TWO_DIGIT_LEN
        || matches!(format, HttpDateFormat::Asctime) && day_token.len() == 1;
    (valid_day_length && year_token.len() == FOUR_DIGIT_LEN).ok_or(format_error)?;
    let weekday = parse_http_weekday(weekday_name).ok_or(format_error)?;
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
    let y = yoe.strict_add(era.strict_mul(400_i32));
    let doy = doe.strict_sub(
        365_i32
            .strict_mul(yoe)
            .strict_add(yoe.div_euclid(4_i32))
            .strict_sub(yoe.div_euclid(LEAP_YEAR_CENTURY_DIVISOR_I32)),
    );
    let mp = 5_i32.strict_mul(doy).strict_add(2).div_euclid(153);
    let month_term = 153_i32.strict_mul(mp).strict_add(2).div_euclid(5);
    let day = doy.strict_sub(month_term).strict_add(1).cast_unsigned();
    let month = mp
        .strict_add(if mp < 10_i32 { 3 } else { -9 })
        .cast_unsigned();
    let year = y.strict_add(i32::from(month <= 2_u32));
    CivilDate { day, month, year }
}
