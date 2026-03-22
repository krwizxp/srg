use super::{Result, TimeError, parse_result_with_context, parse_u32_digits};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy)]
struct HttpDateComponents {
    weekday: u32,
    day: u32,
    month: u32,
    year: i32,
    hour: u32,
    minute: u32,
    second: u32,
}

fn parse_two_digits(d0: u8, d1: u8) -> Option<u32> {
    (d0.is_ascii_digit() && d1.is_ascii_digit())
        .then(|| u32::from(d0 - b'0') * 10 + u32::from(d1 - b'0'))
}

fn parse_http_time_components(time_str: &str) -> Result<(u32, u32, u32)> {
    const ERR_TIME_FMT: &str = "HTTP Date 파싱 실패: 시간 형식이 올바르지 않습니다 (HH:MM:SS)";
    const ERR_TIME_RANGE: &str = "HTTP Date 파싱 실패: 시간 값 범위가 올바르지 않습니다.";
    let time_array = time_str
        .as_bytes()
        .as_array::<8>()
        .ok_or_else(|| TimeError::parse(ERR_TIME_FMT))?;
    if time_array[2] != b':' || time_array[5] != b':' {
        return Err(TimeError::parse(ERR_TIME_FMT));
    }
    let hour = parse_two_digits(time_array[0], time_array[1])
        .ok_or_else(|| TimeError::parse(ERR_TIME_FMT))?;
    let minute = parse_two_digits(time_array[3], time_array[4])
        .ok_or_else(|| TimeError::parse(ERR_TIME_FMT))?;
    let second = parse_two_digits(time_array[6], time_array[7])
        .ok_or_else(|| TimeError::parse(ERR_TIME_FMT))?;
    if hour > 23 || minute > 59 || second > 59 {
        return Err(TimeError::parse(ERR_TIME_RANGE));
    }
    Ok((hour, minute, second))
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

fn parse_http_weekday_short(weekday_str: &str) -> Option<u32> {
    match weekday_str {
        "Sun" => Some(0),
        "Mon" => Some(1),
        "Tue" => Some(2),
        "Wed" => Some(3),
        "Thu" => Some(4),
        "Fri" => Some(5),
        "Sat" => Some(6),
        _ => None,
    }
}

fn parse_http_weekday_long(weekday_str: &str) -> Option<u32> {
    match weekday_str {
        "Sunday" => Some(0),
        "Monday" => Some(1),
        "Tuesday" => Some(2),
        "Wednesday" => Some(3),
        "Thursday" => Some(4),
        "Friday" => Some(5),
        "Saturday" => Some(6),
        _ => None,
    }
}

const fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

const fn days_in_month(year: i32, month: u32) -> Option<u32> {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => Some(31),
        4 | 6 | 9 | 11 => Some(30),
        2 => Some(if is_leap_year(year) { 29 } else { 28 }),
        _ => None,
    }
}

fn current_utc_year() -> i32 {
    let day_index_i64 = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => {
            i64::try_from(duration.as_secs() / 86_400).unwrap_or_else(|_| i64::from(i32::MAX))
        }
        Err(err) => {
            let secs_before_epoch = err.duration().as_secs();
            let days_before_epoch = secs_before_epoch.saturating_add(86_399) / 86_400;
            let days_before_epoch_i64 =
                i64::try_from(days_before_epoch).unwrap_or_else(|_| i64::from(i32::MAX));
            -days_before_epoch_i64
        }
    };
    let day_index = i32::try_from(day_index_i64).unwrap_or_else(|_| {
        if day_index_i64.is_negative() {
            i32::MIN
        } else {
            i32::MAX
        }
    });
    civil_from_days(day_index).0
}

fn expand_rfc850_year(two_digit_year: u32) -> Result<i32> {
    const ERR_YEAR: &str = "HTTP Date 파싱 실패: rfc850 2자리 연도 변환에 실패했습니다.";
    let year_2 = parse_result_with_context(i32::try_from(two_digit_year), ERR_YEAR)?;
    let current_year = current_utc_year();
    let century_base = current_year.div_euclid(100_i32) * 100_i32;
    let mut expanded = century_base + year_2;
    if expanded > current_year + 50_i32 {
        expanded -= 100_i32;
    }
    Ok(expanded)
}

fn parse_http_date_imf_fixdate(raw_date: &str) -> Result<HttpDateComponents> {
    const ERR_FORMAT: &str = "HTTP Date 파싱 실패: IMF-fixdate 형식이 아닙니다.";
    const ERR_NUM: &str = "HTTP Date 파싱 실패: IMF-fixdate 숫자 변환에 실패했습니다.";
    let mut parts = raw_date.split_ascii_whitespace();
    let weekday_token = parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let day_token = parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let month_token = parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let year_token = parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let time_token = parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let tz_token = parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    if parts.next().is_some() || day_token.len() != 2 || year_token.len() != 4 || tz_token != "GMT"
    {
        return Err(TimeError::parse(ERR_FORMAT));
    }
    let weekday_name = weekday_token
        .strip_suffix(',')
        .ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let weekday =
        parse_http_weekday_short(weekday_name).ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let day = parse_u32_digits(day_token).ok_or_else(|| TimeError::parse(ERR_NUM))?;
    let month = parse_http_month(month_token)?;
    let year_u32 = parse_u32_digits(year_token).ok_or_else(|| TimeError::parse(ERR_NUM))?;
    let year = parse_result_with_context(i32::try_from(year_u32), ERR_NUM)?;
    let (hour, minute, second) = parse_http_time_components(time_token)?;
    Ok(HttpDateComponents {
        weekday,
        day,
        month,
        year,
        hour,
        minute,
        second,
    })
}

fn parse_http_date_rfc850(raw_date: &str) -> Result<HttpDateComponents> {
    const ERR_FORMAT: &str = "HTTP Date 파싱 실패: rfc850-date 형식이 아닙니다.";
    const ERR_NUM: &str = "HTTP Date 파싱 실패: rfc850-date 숫자 변환에 실패했습니다.";
    let mut parts = raw_date.split_ascii_whitespace();
    let weekday_token = parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let date_token = parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let time_token = parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let tz_token = parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    if parts.next().is_some() || tz_token != "GMT" {
        return Err(TimeError::parse(ERR_FORMAT));
    }
    let weekday_name = weekday_token
        .strip_suffix(',')
        .ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let weekday =
        parse_http_weekday_long(weekday_name).ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let mut date_parts = date_token.split('-');
    let day_token = date_parts
        .next()
        .ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let month_token = date_parts
        .next()
        .ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let year2_token = date_parts
        .next()
        .ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    if date_parts.next().is_some() || day_token.len() != 2 || year2_token.len() != 2 {
        return Err(TimeError::parse(ERR_FORMAT));
    }
    let day = parse_u32_digits(day_token).ok_or_else(|| TimeError::parse(ERR_NUM))?;
    let month = parse_http_month(month_token)?;
    let year2 = parse_u32_digits(year2_token).ok_or_else(|| TimeError::parse(ERR_NUM))?;
    let year = expand_rfc850_year(year2)?;
    let (hour, minute, second) = parse_http_time_components(time_token)?;
    Ok(HttpDateComponents {
        weekday,
        day,
        month,
        year,
        hour,
        minute,
        second,
    })
}

fn parse_http_date_asctime(raw_date: &str) -> Result<HttpDateComponents> {
    const ERR_FORMAT: &str = "HTTP Date 파싱 실패: asctime-date 형식이 아닙니다.";
    const ERR_NUM: &str = "HTTP Date 파싱 실패: asctime-date 숫자 변환에 실패했습니다.";
    let mut parts = raw_date.split_ascii_whitespace();
    let weekday_token = parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let month_token = parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let day_token = parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let time_token = parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let year_token = parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    if parts.next().is_some() || !(1..=2).contains(&day_token.len()) || year_token.len() != 4 {
        return Err(TimeError::parse(ERR_FORMAT));
    }
    let weekday =
        parse_http_weekday_short(weekday_token).ok_or_else(|| TimeError::parse(ERR_FORMAT))?;
    let day = parse_u32_digits(day_token).ok_or_else(|| TimeError::parse(ERR_NUM))?;
    let month = parse_http_month(month_token)?;
    let year_u32 = parse_u32_digits(year_token).ok_or_else(|| TimeError::parse(ERR_NUM))?;
    let year = parse_result_with_context(i32::try_from(year_u32), ERR_NUM)?;
    let (hour, minute, second) = parse_http_time_components(time_token)?;
    Ok(HttpDateComponents {
        weekday,
        day,
        month,
        year,
        hour,
        minute,
        second,
    })
}

fn unix_timestamp_to_system_time(timestamp_secs: i64) -> Result<SystemTime> {
    const ERR_TIMESTAMP: &str = "HTTP Date 변환 실패: 유효하지 않은 타임스탬프입니다.";
    let secs_i128 = i128::from(timestamp_secs);
    if secs_i128 >= 0 {
        let secs_u64 = parse_result_with_context(u64::try_from(secs_i128), ERR_TIMESTAMP)?;
        UNIX_EPOCH
            .checked_add(Duration::from_secs(secs_u64))
            .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))
    } else {
        let abs_i128 = secs_i128
            .checked_abs()
            .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))?;
        let abs_secs = parse_result_with_context(u64::try_from(abs_i128), ERR_TIMESTAMP)?;
        UNIX_EPOCH
            .checked_sub(Duration::from_secs(abs_secs))
            .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))
    }
}

fn validate_http_date_components(components: HttpDateComponents) -> Result<()> {
    const ERR_DAY: &str = "HTTP Date 파싱 실패: 날짜 값이 유효하지 않습니다.";
    let Some(max_day) = days_in_month(components.year, components.month) else {
        return Err(TimeError::parse(ERR_DAY));
    };
    if components.day == 0 || components.day > max_day {
        return Err(TimeError::parse(ERR_DAY));
    }
    Ok(())
}

fn http_date_components_to_systemtime(components: HttpDateComponents) -> Result<SystemTime> {
    const ERR_WEEKDAY: &str = "HTTP Date 파싱 실패: 요일이 날짜와 일치하지 않습니다.";
    validate_http_date_components(components)?;
    let days = days_from_civil(components.year, components.month, components.day);
    let actual_weekday = (days + 4).rem_euclid(7).cast_unsigned();
    if actual_weekday != components.weekday {
        return Err(TimeError::parse(ERR_WEEKDAY));
    }
    let timestamp_secs = i64::from(days) * 86_400
        + i64::from(components.hour) * 3_600
        + i64::from(components.minute) * 60
        + i64::from(components.second);
    unix_timestamp_to_system_time(timestamp_secs)
}

pub fn parse_http_date_to_systemtime(raw_date: &str) -> Result<SystemTime> {
    const ERR_FORMAT: &str = "HTTP Date 파싱 실패: RFC 9110 HTTP-date의 3개 형식(IMF-fixdate/rfc850/asctime) 중 하나가 아닙니다.";
    if raw_date.as_bytes().get(3).is_some_and(|ch| *ch == b',') {
        return parse_http_date_imf_fixdate(raw_date).and_then(http_date_components_to_systemtime);
    }
    if raw_date.contains(',') {
        return parse_http_date_rfc850(raw_date).and_then(http_date_components_to_systemtime);
    }
    if raw_date.contains("GMT") {
        return parse_http_date_rfc850(raw_date).and_then(http_date_components_to_systemtime);
    }
    if raw_date
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_alphabetic)
    {
        return parse_http_date_asctime(raw_date).and_then(http_date_components_to_systemtime);
    }
    Err(TimeError::parse(ERR_FORMAT))
}

pub const fn days_from_civil(y: i32, m: u32, d: u32) -> i32 {
    let adjusted_year = if m <= 2 { y - 1_i32 } else { y };
    let era = adjusted_year.div_euclid(400);
    let yoe = adjusted_year.rem_euclid(400);
    let shifted_month: i32 = if m > 2 {
        (m - 3).cast_signed()
    } else {
        (m + 9).cast_signed()
    };
    let doy = (153_i32 * shifted_month + 2_i32) / 5_i32 + d.cast_signed() - 1_i32;
    let doe = yoe * 365_i32 + yoe / 4_i32 - yoe / 100_i32 + doy;
    era * 146_097 + doe - 719_468
}

pub fn civil_from_days(z: i32) -> (i32, u32, u32) {
    let shifted_days = z + 719_468_i32;
    let era = shifted_days.div_euclid(146_097);
    let doe = shifted_days.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460_i32 + doe / 36_524_i32 - doe / 146_096_i32) / 365_i32;
    let y = yoe + era * 400_i32;
    let doy = doe - (365_i32 * yoe + yoe / 4_i32 - yoe / 100_i32);
    let mp = (5_i32 * doy + 2_i32) / 153_i32;
    let d = (doy - (153 * mp + 2) / 5 + 1).cast_unsigned();
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }).cast_unsigned();
    (y + i32::from(m <= 2), m, d)
}
