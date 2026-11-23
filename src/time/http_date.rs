use super::{
    CivilDate, DAYS_PER_WEEK_I64, Result, SECS_PER_DAY_I64, SECS_PER_DAY_U64, TimeError, parse_hms,
    parse_u32_digits,
};
use core::{str::FromStr, time::Duration};
use std::time::{SystemTime, UNIX_EPOCH};
const ERR_HTTP_DATE_FORMAT: &str = concat!(
    "HTTP Date 파싱 실패: RFC 9110 HTTP-date의 3개 형식",
    "(IMF-fixdate/rfc850/asctime) 중 하나가 아닙니다."
);
const ERR_LOCAL_YEAR: &str = "HTTP Date 파싱 실패: 현재 연도 계산에 실패했습니다.";
const ERR_TIME_FMT: &str = "HTTP Date 파싱 실패: 시간 형식이 올바르지 않습니다 (HH:MM:SS)";
pub(super) struct HttpDate(pub SystemTime);
#[derive(Clone, Copy)]
enum HttpDateFormat {
    Asctime,
    ImfFixdate,
    Rfc850,
}
impl FromStr for HttpDate {
    type Err = TimeError;
    fn from_str(raw_date: &str) -> Result<Self> {
        let raw_bytes = raw_date.as_bytes();
        let format = if raw_bytes.get(3) == Some(&b',') {
            HttpDateFormat::ImfFixdate
        } else if raw_date.contains(',') || raw_date.contains("GMT") {
            HttpDateFormat::Rfc850
        } else if raw_bytes.first().is_some_and(u8::is_ascii_alphabetic) {
            HttpDateFormat::Asctime
        } else {
            return Err(TimeError::parse(ERR_HTTP_DATE_FORMAT));
        };
        let (fmt_err, num_err) = match format {
            HttpDateFormat::Asctime => (
                "HTTP Date 파싱 실패: asctime-date 형식이 아닙니다.",
                "HTTP Date 파싱 실패: asctime-date 숫자 변환에 실패했습니다.",
            ),
            HttpDateFormat::ImfFixdate => (
                "HTTP Date 파싱 실패: IMF-fixdate 형식이 아닙니다.",
                "HTTP Date 파싱 실패: IMF-fixdate 숫자 변환에 실패했습니다.",
            ),
            HttpDateFormat::Rfc850 => (
                "HTTP Date 파싱 실패: rfc850-date 형식이 아닙니다.",
                "HTTP Date 파싱 실패: rfc850-date 숫자 변환에 실패했습니다.",
            ),
        };
        let mut parts = raw_date.split_ascii_whitespace();
        let mut next = || parts.next().ok_or(fmt_err);
        let (weekday_name, day_token, month_token, year_token, time_token) = match format {
            HttpDateFormat::Asctime => {
                let (wday, mon, day, time, year) = (next()?, next()?, next()?, next()?, next()?);
                (wday, day, mon, year, time)
            }
            HttpDateFormat::ImfFixdate => {
                let (wday, day, mon, year, time, tz) =
                    (next()?, next()?, next()?, next()?, next()?, next()?);
                (tz == "GMT").ok_or(fmt_err)?;
                (wday.strip_suffix(',').ok_or(fmt_err)?, day, mon, year, time)
            }
            HttpDateFormat::Rfc850 => {
                let (wday, date, time, tz) = (next()?, next()?, next()?, next()?);
                (tz == "GMT").ok_or(fmt_err)?;
                let mut date_parts = date.split('-');
                let (Some(day), Some(mon), Some(year2), None) = (
                    date_parts.next(),
                    date_parts.next(),
                    date_parts.next(),
                    date_parts.next(),
                ) else {
                    return Err(TimeError::parse(fmt_err));
                };
                (
                    wday.strip_suffix(',').ok_or(fmt_err)?,
                    day,
                    mon,
                    year2,
                    time,
                )
            }
        };
        parts.next().is_none().ok_or(fmt_err)?;
        let valid_day_len = day_token.len() == 2
            || matches!(format, HttpDateFormat::Asctime) && day_token.len() == 1;
        let expected_year_len = if matches!(format, HttpDateFormat::Rfc850) {
            2
        } else {
            4
        };
        (valid_day_len && year_token.len() == expected_year_len).ok_or(fmt_err)?;
        let weekday = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ]
        .iter()
        .position(|&name| name == weekday_name || name.get(..3) == Some(weekday_name))
        .ok_or(fmt_err)?;
        let month_idx = b"JanFebMarAprMayJunJulAugSepOctNovDec"
            .as_chunks::<3>()
            .0
            .iter()
            .position(|name| name.as_slice() == month_token.as_bytes())
            .ok_or("HTTP Date 파싱 실패: 알 수 없는 월 형식")?;
        let month_number = u32::try_from(month_idx).unwrap_or(0).strict_add(1);
        let mut max_day = u32::from(
            *[31_u8, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                .get(month_idx)
                .ok_or("HTTP Date 파싱 실패: 알 수 없는 월 형식")?,
        );
        let day = parse_u32_digits(day_token).ok_or(num_err)?;
        let mut year = parse_u32_digits(year_token).ok_or(num_err)?.cast_signed();
        let (hour, minute, second) = parse_hms(time_token).ok_or(ERR_TIME_FMT)?;
        if matches!(format, HttpDateFormat::Rfc850) {
            let (day_index_i64, current_second) = match SystemTime::now().duration_since(UNIX_EPOCH)
            {
                Ok(dur) => (
                    dur.as_secs().div_euclid(SECS_PER_DAY_U64).cast_signed(),
                    dur.as_secs().rem_euclid(SECS_PER_DAY_U64),
                ),
                Err(error) => {
                    let dur = error.duration();
                    let rem = dur.as_secs().rem_euclid(SECS_PER_DAY_U64);
                    let frac = u64::from(dur.subsec_nanos() != 0);
                    let days_before = dur
                        .as_secs()
                        .div_euclid(SECS_PER_DAY_U64)
                        .strict_add(u64::from(rem != 0 || frac != 0));
                    (
                        i64::try_from(days_before)
                            .map_err(|source| TimeError::parse_with_source(ERR_LOCAL_YEAR, source))?
                            .strict_neg(),
                        SECS_PER_DAY_U64
                            .strict_sub(rem.strict_add(frac))
                            .rem_euclid(SECS_PER_DAY_U64),
                    )
                }
            };
            let current_date = civil_from_days(
                i32::try_from(day_index_i64)
                    .map_err(|source| TimeError::parse_with_source(ERR_LOCAL_YEAR, source))?,
            );
            year = current_date
                .year
                .div_euclid(100)
                .strict_mul(100)
                .strict_add(year);
            let cutoff_year = current_date.year.strict_add(50);
            let past_cutoff = year > cutoff_year
                || year == cutoff_year && {
                    let date_order =
                        (month_number, day).cmp(&(current_date.month, current_date.day));
                    if date_order.is_eq() {
                        u64::from(hour)
                            .strict_mul(3_600)
                            .strict_add(u64::from(minute).strict_mul(60))
                            .strict_add(u64::from(second))
                            > current_second
                    } else {
                        date_order.is_gt()
                    }
                };
            if past_cutoff {
                year = year.strict_sub(100);
            }
        }
        (hour <= 23 && minute <= 59 && second <= 60)
            .ok_or("HTTP Date 파싱 실패: 시간 값 범위가 올바르지 않습니다.")?;
        if month_number == 2
            && year.rem_euclid(4) == 0_i32
            && (year.rem_euclid(100) != 0_i32 || year.rem_euclid(400) == 0_i32)
        {
            max_day = 29;
        }
        (day != 0 && day <= max_day).ok_or("HTTP Date 파싱 실패: 날짜 값이 유효하지 않습니다.")?;
        let adj_year = i64::from(year).strict_sub(i64::from(month_number <= 2));
        let (era, yoe) = (adj_year.div_euclid(400), adj_year.rem_euclid(400));
        let shifted_month =
            i64::from(month_number).strict_add(if month_number > 2 { -3 } else { 9 });
        let doy = 153_i64
            .strict_mul(shifted_month)
            .strict_add(2)
            .div_euclid(5)
            .strict_add(i64::from(day).strict_sub(1));
        let doe = 365_i64
            .strict_mul(yoe)
            .strict_add(yoe.div_euclid(4))
            .strict_sub(yoe.div_euclid(100))
            .strict_add(doy);
        let days = era.strict_mul(146_097).strict_add(doe).strict_sub(719_468);
        (days.strict_add(4).rem_euclid(DAYS_PER_WEEK_I64) == i64::try_from(weekday).unwrap_or(-1))
            .ok_or("HTTP Date 파싱 실패: 요일이 날짜와 일치하지 않습니다.")?;
        let timestamp_secs = days
            .strict_mul(SECS_PER_DAY_I64)
            .strict_add(i64::from(hour).strict_mul(3_600))
            .strict_add(i64::from(minute).strict_mul(60))
            // SystemTime has no separate leap-second value; :60 normalizes to the next second.
            .strict_add(i64::from(second));
        let duration = Duration::from_secs(timestamp_secs.unsigned_abs());
        let ts = if timestamp_secs >= 0 {
            UNIX_EPOCH.checked_add(duration)
        } else {
            UNIX_EPOCH.checked_sub(duration)
        };
        Ok(Self(ts.ok_or(
            "HTTP Date 변환 실패: 유효하지 않은 타임스탬프입니다.",
        )?))
    }
}
pub(super) fn civil_from_days(z: i32) -> CivilDate {
    let shifted = z.rem_euclid(146_097).strict_add(719_468);
    let era = z
        .div_euclid(146_097)
        .strict_add(shifted.div_euclid(146_097));
    let doe = shifted.rem_euclid(146_097);
    let yoe = doe
        .strict_sub(doe.div_euclid(1_460))
        .strict_add(doe.div_euclid(36_524))
        .strict_sub(doe.div_euclid(146_096))
        .div_euclid(365);
    let doy = doe.strict_sub(
        365_i32
            .strict_mul(yoe)
            .strict_add(yoe.div_euclid(4))
            .strict_sub(yoe.div_euclid(100)),
    );
    let mp = 5_i32.strict_mul(doy).strict_add(2).div_euclid(153);
    let day = doy
        .strict_sub(153_i32.strict_mul(mp).strict_add(2).div_euclid(5))
        .strict_add(1)
        .cast_unsigned();
    let month = mp.strict_add(if mp < 10 { 3 } else { -9 }).cast_unsigned();
    let year = yoe
        .strict_add(era.strict_mul(400))
        .strict_add(i32::from(month <= 2));
    CivilDate { day, month, year }
}
