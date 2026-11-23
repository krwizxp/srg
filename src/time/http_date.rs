use super::{
    HTTP_SCHEME_PREFIX, HTTPS_SCHEME_PREFIX, ParsedServer, SECS_PER_DAY_U64, UrlScheme, parse_hms,
    parse_u32_digits,
};
use crate::diagnostic::{AppError, Result};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use alloc::ffi::CString;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use core::fmt::NumBuffer;
use core::{num::IntErrorKind, str::FromStr, time::Duration};
use std::{
    net,
    time::{SystemTime, UNIX_EPOCH},
};
const ERR_HOST: &str = "서버 주소 파싱 실패: 호스트 값이 비어있거나 형식이 올바르지 않습니다.";
const ERR_LOCAL_YEAR: &str = "HTTP Date 파싱 실패: 현재 연도 계산에 실패했습니다.";
const ERR_PORT: &str = "서버 주소 파싱 실패: 포트 번호가 유효하지 않습니다 (1~65535).";
pub(super) struct HttpDate(pub SystemTime);
#[derive(Clone, Copy)]
enum HttpDateFormat {
    Asctime,
    ImfFixdate,
    Rfc850,
}
impl FromStr for HttpDate {
    type Err = AppError;
    fn from_str(raw_date: &str) -> Result<Self> {
        let raw_bytes = raw_date.as_bytes();
        let format = if raw_bytes.get(3) == Some(&b',') {
            HttpDateFormat::ImfFixdate
        } else if raw_date.contains(',') || raw_date.contains("GMT") {
            HttpDateFormat::Rfc850
        } else if raw_bytes.first().is_some_and(u8::is_ascii_alphabetic) {
            HttpDateFormat::Asctime
        } else {
            return Err("HTTP Date 파싱 실패: RFC 9110 HTTP-date의 3개 형식(IMF-fixdate/rfc850/asctime) 중 하나가 아닙니다.".into());
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
        let (weekday_name, day_tok, mon_tok, year_tok, time_tok) = match format {
            HttpDateFormat::Asctime => {
                let (wd, mon, day, tm, yr) = (next()?, next()?, next()?, next()?, next()?);
                (wd, day, mon, yr, tm)
            }
            HttpDateFormat::ImfFixdate => {
                let (wd, day, mon, yr, tm, tz) =
                    (next()?, next()?, next()?, next()?, next()?, next()?);
                (tz == "GMT").ok_or(fmt_err)?;
                (wd.strip_suffix(',').ok_or(fmt_err)?, day, mon, yr, tm)
            }
            HttpDateFormat::Rfc850 => {
                let (wd, date, tm, tz) = (next()?, next()?, next()?, next()?);
                (tz == "GMT").ok_or(fmt_err)?;
                let mut dp = date.split('-');
                let (Some(day), Some(mon), Some(yr2), None) =
                    (dp.next(), dp.next(), dp.next(), dp.next())
                else {
                    return Err(fmt_err.into());
                };
                (wd.strip_suffix(',').ok_or(fmt_err)?, day, mon, yr2, tm)
            }
        };
        parts.next().is_none().ok_or(fmt_err)?;
        let valid_day_len =
            day_tok.len() == 2 || matches!(format, HttpDateFormat::Asctime) && day_tok.len() == 1;
        let exp_yr_len = if matches!(format, HttpDateFormat::Rfc850) {
            2
        } else {
            4
        };
        (valid_day_len && year_tok.len() == exp_yr_len).ok_or(fmt_err)?;
        let weekday = "Sunday,Monday,Tuesday,Wednesday,Thursday,Friday,Saturday"
            .split(',')
            .position(|name| name == weekday_name || name.get(..3) == Some(weekday_name))
            .ok_or(fmt_err)?;
        let mon_idx = b"JanFebMarAprMayJunJulAugSepOctNovDec"
            .as_chunks::<3>()
            .0
            .iter()
            .position(|name| name.as_slice() == mon_tok.as_bytes())
            .ok_or("HTTP Date 파싱 실패: 알 수 없는 월 형식")?;
        let mon_num = u32::try_from(mon_idx).unwrap_or(0).strict_add(1);
        let mut max_day = u32::from(
            *[31_u8, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                .get(mon_idx)
                .ok_or("HTTP Date 파싱 실패: 알 수 없는 월 형식")?,
        );
        let day = parse_u32_digits(day_tok).ok_or(num_err)?;
        let mut year = parse_u32_digits(year_tok).ok_or(num_err)?.cast_signed();
        let (hour, minute, second) = parse_hms(time_tok)
            .ok_or("HTTP Date 파싱 실패: 시간 형식이 올바르지 않습니다 (HH:MM:SS)")?;
        if matches!(format, HttpDateFormat::Rfc850) {
            let (day_i64, cur_sec) = match SystemTime::now().duration_since(UNIX_EPOCH) {
                Ok(dur) => (
                    dur.as_secs().div_euclid(SECS_PER_DAY_U64).cast_signed(),
                    dur.as_secs().rem_euclid(SECS_PER_DAY_U64),
                ),
                Err(error) => {
                    let dur = error.duration();
                    let rem = dur.as_secs().rem_euclid(SECS_PER_DAY_U64);
                    let frac = u64::from(dur.subsec_nanos() != 0);
                    let before = dur
                        .as_secs()
                        .div_euclid(SECS_PER_DAY_U64)
                        .strict_add(u64::from(rem != 0 || frac != 0));
                    (
                        i64::try_from(before)
                            .map_err(|src| AppError::context(ERR_LOCAL_YEAR, src))?
                            .strict_neg(),
                        SECS_PER_DAY_U64
                            .strict_sub(rem.strict_add(frac))
                            .rem_euclid(SECS_PER_DAY_U64),
                    )
                }
            };
            let (cur_year, cur_month, cur_day) = civil_from_days(
                i32::try_from(day_i64).map_err(|src| AppError::context(ERR_LOCAL_YEAR, src))?,
            );
            year = cur_year.div_euclid(100).strict_mul(100).strict_add(year);
            let sec_of_day = u64::from(hour)
                .strict_mul(3_600)
                .strict_add(u64::from(minute).strict_mul(60))
                .strict_add(u64::from(second));
            if (year, mon_num, day, sec_of_day)
                > (cur_year.strict_add(50), cur_month, cur_day, cur_sec)
            {
                year = year.strict_sub(100);
            }
        }
        (hour <= 23 && minute <= 59 && second <= 60)
            .ok_or("HTTP Date 파싱 실패: 시간 값 범위가 올바르지 않습니다.")?;
        if mon_num == 2
            && year.rem_euclid(4) == 0_i32
            && (year.rem_euclid(100) != 0_i32 || year.rem_euclid(400) == 0_i32)
        {
            max_day = 29;
        }
        (day != 0 && day <= max_day).ok_or("HTTP Date 파싱 실패: 날짜 값이 유효하지 않습니다.")?;
        let adj_year = i64::from(year).strict_sub(i64::from(mon_num <= 2));
        let (era, yoe) = (adj_year.div_euclid(400), adj_year.rem_euclid(400));
        let shifted_mon = i64::from(mon_num).strict_add(if mon_num > 2 { -3 } else { 9 });
        let doy = 153_i64
            .strict_mul(shifted_mon)
            .strict_add(2)
            .div_euclid(5)
            .strict_add(i64::from(day).strict_sub(1));
        let doe = 365_i64
            .strict_mul(yoe)
            .strict_add(yoe.div_euclid(4))
            .strict_sub(yoe.div_euclid(100))
            .strict_add(doy);
        let days = era.strict_mul(146_097).strict_add(doe).strict_sub(719_468);
        (days.strict_add(4).rem_euclid(7) == i64::try_from(weekday).unwrap_or(-1))
            .ok_or("HTTP Date 파싱 실패: 요일이 날짜와 일치하지 않습니다.")?;
        let ts_secs = days
            .strict_mul(86_400)
            .strict_add(i64::from(hour).strict_mul(3_600))
            .strict_add(i64::from(minute).strict_mul(60))
            .strict_add(i64::from(second));
        let dur = Duration::from_secs(ts_secs.unsigned_abs());
        let ts = if ts_secs >= 0 {
            UNIX_EPOCH.checked_add(dur)
        } else {
            UNIX_EPOCH.checked_sub(dur)
        };
        Ok(Self(ts.ok_or(
            "HTTP Date 변환 실패: 유효하지 않은 타임스탬프입니다.",
        )?))
    }
}
impl FromStr for ParsedServer {
    type Err = AppError;
    fn from_str(host: &str) -> Result<Self> {
        let trimmed = host.trim();
        (!trimmed.is_empty()).ok_or("서버 주소를 비워둘 수 없습니다.")?;
        let (scheme, after_scheme) = if let Some((prefix, rest)) =
            trimmed.split_at_checked(HTTPS_SCHEME_PREFIX.len())
            && prefix.eq_ignore_ascii_case(HTTPS_SCHEME_PREFIX)
        {
            (UrlScheme::Https, rest)
        } else if let Some((prefix, rest)) = trimmed.split_at_checked(HTTP_SCHEME_PREFIX.len())
            && prefix.eq_ignore_ascii_case(HTTP_SCHEME_PREFIX)
        {
            (UrlScheme::Http, rest)
        } else {
            (UrlScheme::Https, trimmed)
        };
        let (has_path, invalid_char, colon_count) =
            after_scheme
                .chars()
                .fold((false, false, 0_usize), |(path, invalid, colons), ch| {
                    (
                        path || matches!(ch, '/' | '\\' | '?' | '#'),
                        invalid || matches!(ch, '@' | '%') || ch.is_control() || ch.is_whitespace(),
                        colons.strict_add(usize::from(ch == ':')),
                    )
                });
        (!has_path).ok_or("서버 주소에는 path/query/fragment를 사용할 수 없습니다.")?;
        (!after_scheme.is_empty() && !invalid_char).ok_or(ERR_HOST)?;
        let (host_part, explicit_port, bracketed) =
            if let Some(bracketed_host) = after_scheme.strip_prefix('[') {
                let (host_part, rem) = bracketed_host.split_once(']').ok_or(ERR_HOST)?;
                let port = (!rem.is_empty())
                    .then(|| parse_port(rem.strip_prefix(':').ok_or(ERR_HOST)?))
                    .transpose()?;
                (host_part, port, true)
            } else if colon_count == 1
                && let Some((host_part, port_part)) = after_scheme.split_once(':')
            {
                (host_part, Some(parse_port(port_part)?), false)
            } else {
                (after_scheme, None, false)
            };
        let port = explicit_port.unwrap_or(match scheme {
            UrlScheme::Http => 80,
            UrlScheme::Https => 443,
        });
        (!host_part.is_empty() && !host_part.contains(['[', ']'])).ok_or(ERR_HOST)?;
        let host_is_ipv6 = colon_count > 1 && host_part.parse::<net::Ipv6Addr>().is_ok();
        (!bracketed && (explicit_port.is_some() || colon_count == 0) || host_is_ipv6)
            .ok_or(ERR_HOST)?;
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        let request_target = {
            let prefix = match scheme {
                UrlScheme::Http => HTTP_SCHEME_PREFIX,
                UrlScheme::Https => HTTPS_SCHEME_PREFIX,
            };
            let (open, close) = if host_is_ipv6 { ("[", "]") } else { ("", "") };
            let mut port_buf = NumBuffer::new();
            let (sep, port_text) = if explicit_port.is_some() {
                (":", port.format_into(&mut port_buf))
            } else {
                ("", "")
            };
            let parts = [prefix, open, host_part, close, sep, port_text];
            let mut text = String::with_capacity(
                parts
                    .iter()
                    .fold(1_usize, |len, part| len.strict_add(part.len())),
            );
            text.extend(parts);
            CString::new(text)
                .map_err(|source| AppError::context("서버 HTTP 요청 대상 변환 실패", source))?
        };
        #[cfg(target_os = "windows")]
        let host_wide = {
            let mut value = Vec::new();
            value
                .try_reserve_exact(host_part.len().strict_add(1))
                .map_err(|source| AppError::context("서버 host UTF-16 메모리 확보 실패", source))?;
            value.extend(host_part.encode_utf16());
            value.push(0);
            value
        };
        Ok(Self {
            #[cfg(target_os = "windows")]
            host_wide,
            #[cfg(target_os = "windows")]
            port,
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            request_target,
            scheme,
        })
    }
}
pub(super) fn civil_from_days(z: i32) -> (i32, u32, u32) {
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
    (year, month, day)
}
fn parse_port(port_part: &str) -> Result<u16> {
    let port = match port_part.parse::<u16>() {
        Ok(port) if port != 0 && port_part.as_bytes().first().is_some_and(u8::is_ascii_digit) => {
            port
        }
        Err(source)
            if matches!(source.kind(), IntErrorKind::PosOverflow)
                && port_part.bytes().all(|byte| byte.is_ascii_digit()) =>
        {
            return Err(AppError::context(ERR_PORT, source));
        }
        Ok(_) | Err(_) => return Err(AppError::message(ERR_PORT)),
    };
    Ok(port)
}
