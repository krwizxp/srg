use super::{Result, TimeError, parse_result_with_context, parse_u32_digits};
use core::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
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
        let leap_year = (self.year.rem_euclid(4_i32) == 0_i32
            && self.year.rem_euclid(100_i32) != 0_i32)
            || self.year.rem_euclid(400_i32) == 0_i32;
        let max_day = match self.month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => {
                if leap_year {
                    29
                } else {
                    28
                }
            }
            _ => return Err(TimeError::parse(ERR_DAY)),
        };
        if self.day == 0 || self.day > max_day {
            return Err(TimeError::parse(ERR_DAY));
        }
        let year = i64::from(self.year);
        let adjusted_year = if self.month <= 2 {
            year.checked_sub(1_i64)
                .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))?
        } else {
            year
        };
        let (era, yoe) = (
            adjusted_year.div_euclid(400_i64),
            adjusted_year.rem_euclid(400_i64),
        );
        let shifted_month = if self.month > 2 {
            i64::from(self.month)
                .checked_sub(3_i64)
                .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))?
        } else {
            i64::from(self.month)
                .checked_add(9_i64)
                .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))?
        };
        let month_term = 153_i64
            .checked_mul(shifted_month)
            .and_then(|value| value.checked_add(2_i64))
            .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))?;
        let day_offset = i64::from(self.day)
            .checked_sub(1_i64)
            .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))?;
        let doy = month_term
            .checked_div(5_i64)
            .and_then(|value| value.checked_add(day_offset))
            .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))?;
        let doe = 365_i64
            .checked_mul(yoe)
            .and_then(|value| value.checked_add(yoe.div_euclid(4_i64)))
            .and_then(|value| value.checked_sub(yoe.div_euclid(100_i64)))
            .and_then(|value| value.checked_add(doy))
            .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))?;
        era.checked_mul(146_097_i64)
            .and_then(|value| value.checked_add(doe))
            .and_then(|value| value.checked_sub(719_468_i64))
            .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))
    }
    fn timestamp_secs(self, days: i64) -> Result<i64> {
        const ERR_TIMESTAMP: &str = "HTTP Date 변환 실패: 타임스탬프 계산 중 범위 오류입니다.";
        days.checked_mul(86_400_i64)
            .and_then(|value| {
                let hour_secs = i64::from(self.hour).checked_mul(3_600_i64)?;
                value.checked_add(hour_secs)
            })
            .and_then(|value| {
                let minute_secs = i64::from(self.minute).checked_mul(60_i64)?;
                value.checked_add(minute_secs)
            })
            .and_then(|value| value.checked_add(i64::from(self.second)))
            .ok_or_else(|| TimeError::parse(ERR_TIMESTAMP))
    }
    fn validate_weekday(self, days: i64) -> Result<()> {
        const ERR_WEEKDAY: &str = "HTTP Date 파싱 실패: 요일이 날짜와 일치하지 않습니다.";
        let actual_weekday = parse_result_with_context(
            u32::try_from(
                days.checked_add(4_i64)
                    .ok_or_else(|| TimeError::parse(ERR_WEEKDAY))?
                    .rem_euclid(7_i64),
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
    (u32::from(tens).checked_mul(10))?.checked_add(u32::from(ones))
}
fn parse_http_time_components(time_str: &str) -> Result<(u32, u32, u32)> {
    const ERR_TIME_FMT: &str = "HTTP Date 파싱 실패: 시간 형식이 올바르지 않습니다 (HH:MM:SS)";
    const ERR_TIME_RANGE: &str = "HTTP Date 파싱 실패: 시간 값 범위가 올바르지 않습니다.";
    let time_array = (time_str
        .as_bytes()
        .as_array::<8>()
        .ok_or_else(|| TimeError::parse(ERR_TIME_FMT)))?;
    if time_array[2] != b':' || time_array[5] != b':' {
        return Err(TimeError::parse(ERR_TIME_FMT));
    }
    let hour = (parse_two_digits(time_array[0], time_array[1])
        .ok_or_else(|| TimeError::parse(ERR_TIME_FMT)))?;
    let minute = (parse_two_digits(time_array[3], time_array[4])
        .ok_or_else(|| TimeError::parse(ERR_TIME_FMT)))?;
    let second = (parse_two_digits(time_array[6], time_array[7])
        .ok_or_else(|| TimeError::parse(ERR_TIME_FMT)))?;
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
fn parse_http_date_rfc850(raw_date: &str) -> Result<HttpDateComponents> {
    const ERR_FORMAT: &str = "HTTP Date 파싱 실패: rfc850-date 형식이 아닙니다.";
    const ERR_NUM: &str = "HTTP Date 파싱 실패: rfc850-date 숫자 변환에 실패했습니다.";
    const ERR_YEAR: &str = "HTTP Date 파싱 실패: rfc850 2자리 연도 변환에 실패했습니다.";
    let mut parts = raw_date.split_ascii_whitespace();
    let weekday_token = (parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT)))?;
    let date_token = (parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT)))?;
    let time_token = (parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT)))?;
    let tz_token = (parts.next().ok_or_else(|| TimeError::parse(ERR_FORMAT)))?;
    if parts.next().is_some() || tz_token != "GMT" {
        return Err(TimeError::parse(ERR_FORMAT));
    }
    let weekday_name = (weekday_token
        .strip_suffix(',')
        .ok_or_else(|| TimeError::parse(ERR_FORMAT)))?;
    let weekday = (match weekday_name {
        "Sunday" => Some(0),
        "Monday" => Some(1),
        "Tuesday" => Some(2),
        "Wednesday" => Some(3),
        "Thursday" => Some(4),
        "Friday" => Some(5),
        "Saturday" => Some(6),
        _ => None,
    }
    .ok_or_else(|| TimeError::parse(ERR_FORMAT)))?;
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
    if date_parts.next().is_some() || day_token.len() != 2 || year2_token.len() != 2 {
        return Err(TimeError::parse(ERR_FORMAT));
    }
    let day = (parse_u32_digits(day_token).ok_or_else(|| TimeError::parse(ERR_NUM)))?;
    let month = (parse_http_month(month_token))?;
    let year2 = (parse_u32_digits(year2_token).ok_or_else(|| TimeError::parse(ERR_NUM)))?;
    let year_two_digits = (parse_result_with_context(i32::try_from(year2), ERR_YEAR))?;
    let current_year = {
        let day_index_i64 = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => i64::try_from(duration.as_secs().div_euclid(86_400))
                .unwrap_or_else(|_| i64::from(i32::MAX)),
            Err(err) => {
                let secs_before_epoch = err.duration().as_secs();
                let days_before_epoch = secs_before_epoch.saturating_add(86_399).div_euclid(86_400);
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
            .div_euclid(100_i64)
            .checked_mul(100_i64)
            .ok_or_else(|| TimeError::parse(ERR_YEAR)))?;
        let mut expanded = (century_base
            .checked_add(i64::from(year_two_digits))
            .ok_or_else(|| TimeError::parse(ERR_YEAR)))?;
        let cutoff = (i64::from(current_year)
            .checked_add(50_i64)
            .ok_or_else(|| TimeError::parse(ERR_YEAR)))?;
        if expanded > cutoff {
            expanded = (expanded
                .checked_sub(100_i64)
                .ok_or_else(|| TimeError::parse(ERR_YEAR)))?;
        }
        parse_result_with_context(i32::try_from(expanded), ERR_YEAR)?
    };
    let (hour, minute, second) = (parse_http_time_components(time_token))?;
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
    const ERR_FORMAT: &str = "HTTP Date 파싱 실패: RFC 9110 HTTP-date의 3개 형식(IMF-fixdate/rfc850/asctime) 중 하나가 아닙니다.";
    if raw_date.as_bytes().get(3).is_some_and(|ch| *ch == b',') {
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
            || day_token.len() != 2
            || year_token.len() != 4
            || tz_token != "GMT"
        {
            return Err(TimeError::parse(ERR_IMF_FORMAT));
        }
        let weekday_name = (weekday_token
            .strip_suffix(',')
            .ok_or_else(|| TimeError::parse(ERR_IMF_FORMAT)))?;
        let weekday = (parse_http_weekday_short(weekday_name)
            .ok_or_else(|| TimeError::parse(ERR_IMF_FORMAT)))?;
        let day = (parse_u32_digits(day_token).ok_or_else(|| TimeError::parse(ERR_IMF_NUM)))?;
        let month = (parse_http_month(month_token))?;
        let year_u32 = (parse_u32_digits(year_token).ok_or_else(|| TimeError::parse(ERR_IMF_NUM)))?;
        let year = (parse_result_with_context(i32::try_from(year_u32), ERR_IMF_NUM))?;
        let (hour, minute, second) = (parse_http_time_components(time_token))?;
        return http_date_components_to_systemtime(HttpDateComponents {
            day,
            hour,
            minute,
            month,
            second,
            weekday,
            year,
        });
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
        if parts.next().is_some() || !(1..=2).contains(&day_token.len()) || year_token.len() != 4 {
            return Err(TimeError::parse(ERR_ASCTIME_FORMAT));
        }
        let weekday = (parse_http_weekday_short(weekday_token)
            .ok_or_else(|| TimeError::parse(ERR_ASCTIME_FORMAT)))?;
        let day = (parse_u32_digits(day_token).ok_or_else(|| TimeError::parse(ERR_ASCTIME_NUM)))?;
        let month = (parse_http_month(month_token))?;
        let year_u32 =
            (parse_u32_digits(year_token).ok_or_else(|| TimeError::parse(ERR_ASCTIME_NUM)))?;
        let year = (parse_result_with_context(i32::try_from(year_u32), ERR_ASCTIME_NUM))?;
        let (hour, minute, second) = (parse_http_time_components(time_token))?;
        return http_date_components_to_systemtime(HttpDateComponents {
            day,
            hour,
            minute,
            month,
            second,
            weekday,
            year,
        });
    }
    Err(TimeError::parse(ERR_FORMAT))
}
pub fn civil_from_days(z: i32) -> Option<(i32, u32, u32)> {
    let shifted_days = (i64::from(z).checked_add(719_468_i64))?;
    let era = shifted_days.div_euclid(146_097_i64);
    let doe = shifted_days.rem_euclid(146_097_i64);
    let yoe_after_first = (doe.checked_sub((doe.checked_div(1_460_i64))?))?;
    let yoe_after_second = (yoe_after_first.checked_add((doe.checked_div(36_524_i64))?))?;
    let yoe_numerator = (yoe_after_second.checked_sub((doe.checked_div(146_096_i64))?))?;
    let yoe = (yoe_numerator.checked_div(365_i64))?;
    let y = (yoe.checked_add((era.checked_mul(400_i64))?))?;
    let year_days = (365_i64.checked_mul(yoe))?;
    let leap_days = (yoe.checked_div(4_i64))?;
    let skipped_centuries = (yoe.checked_div(100_i64))?;
    let doy = (doe.checked_sub(
        (year_days
            .checked_add(leap_days)?
            .checked_sub(skipped_centuries))?,
    ))?;
    let mp = (5_i64
        .checked_mul(doy)?
        .checked_add(2_i64)?
        .checked_div(153_i64))?;
    let month_term = (153_i64
        .checked_mul(mp)?
        .checked_add(2_i64)?
        .checked_div(5_i64))?;
    let day = (u32::try_from((doy.checked_sub(month_term)?.checked_add(1_i64))?).ok())?;
    let month_i64 = if mp < 10_i64 {
        (mp.checked_add(3_i64))?
    } else {
        (mp.checked_sub(9_i64))?
    };
    let month = (u32::try_from(month_i64).ok())?;
    let year = (i32::try_from((y.checked_add(i64::from(month <= 2)))?).ok())?;
    Some((year, month, day))
}
