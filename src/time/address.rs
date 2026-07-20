use super::{
    HTTP_SCHEME_PREFIX, HTTP_SCHEME_PREFIX_LEN, HTTPS_SCHEME_PREFIX, HTTPS_SCHEME_PREFIX_LEN,
    ParsedServer, Result, TimeError, UrlScheme, util::parse_result_with_context,
};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use alloc::ffi::CString;
use core::str::FromStr;
use std::net;
const ERR_EMPTY: &str = "서버 주소를 비워둘 수 없습니다.";
const ERR_HOST: &str = "서버 주소 파싱 실패: 호스트 값이 비어있거나 형식이 올바르지 않습니다.";
const ERR_PATH: &str = "서버 주소에는 path/query/fragment를 사용할 수 없습니다.";
const ERR_PORT: &str = "서버 주소 파싱 실패: 포트 번호가 유효하지 않습니다 (1~65535).";
const DEFAULT_HTTP_PORT: u16 = 80;
const DEFAULT_HTTPS_PORT: u16 = 443;
impl FromStr for ParsedServer {
    type Err = TimeError;
    fn from_str(host: &str) -> Result<Self> {
        let trimmed_input = host.trim();
        if trimmed_input.is_empty() {
            return Err(TimeError::parse(ERR_EMPTY));
        }
        let (scheme, after_scheme) = if let Some((prefix, rest)) =
            trimmed_input.split_at_checked(HTTPS_SCHEME_PREFIX_LEN)
            && prefix.eq_ignore_ascii_case(HTTPS_SCHEME_PREFIX)
        {
            (UrlScheme::Https, rest)
        } else if let Some((prefix, rest)) = trimmed_input.split_at_checked(HTTP_SCHEME_PREFIX_LEN)
            && prefix.eq_ignore_ascii_case(HTTP_SCHEME_PREFIX)
        {
            (UrlScheme::Http, rest)
        } else {
            (UrlScheme::Https, trimmed_input)
        };
        if after_scheme.contains(['/', '\\', '?', '#']) {
            return Err(TimeError::parse(ERR_PATH));
        }
        if after_scheme.is_empty()
            || after_scheme.contains(['@', '%'])
            || after_scheme.contains(|ch: char| ch.is_control() || ch.is_whitespace())
        {
            return Err(TimeError::parse(ERR_HOST));
        }
        let default_port = match scheme {
            UrlScheme::Http => DEFAULT_HTTP_PORT,
            UrlScheme::Https => DEFAULT_HTTPS_PORT,
        };
        let (host_part, explicit_port, bracketed) =
            if let Some(bracketed_host) = after_scheme.strip_prefix('[') {
                let (host_part, rem) = bracketed_host
                    .split_once(']')
                    .ok_or_else(|| TimeError::parse(ERR_HOST))?;
                if rem.is_empty() {
                    (host_part, None, true)
                } else {
                    let port_part = rem
                        .strip_prefix(':')
                        .ok_or_else(|| TimeError::parse(ERR_HOST))?;
                    (host_part, Some(parse_port(port_part)?), true)
                }
            } else if let Some((host_part, port_part)) = after_scheme.split_once(':')
                && !port_part.contains(':')
            {
                (host_part, Some(parse_port(port_part)?), false)
            } else {
                (after_scheme, None, false)
            };
        let port = explicit_port.unwrap_or(default_port);
        if host_part.is_empty() || host_part.contains(['[', ']']) {
            return Err(TimeError::parse(ERR_HOST));
        }
        let literal_ip_addr = host_part.parse::<net::IpAddr>().ok();
        if bracketed && !matches!(literal_ip_addr, Some(net::IpAddr::V6(_))) {
            return Err(TimeError::parse(ERR_HOST));
        }
        if host_part.contains(':') && !matches!(literal_ip_addr, Some(net::IpAddr::V6(_))) {
            return Err(TimeError::parse(ERR_HOST));
        }
        let build_host_port_text =
            |prefix: &str, include_port: bool| match (host_part.contains(':'), include_port) {
                (true, true) => format!("{prefix}[{host_part}]:{port}"),
                (true, false) => format!("{prefix}[{host_part}]"),
                (false, true) => format!("{prefix}{host_part}:{port}"),
                (false, false) => format!("{prefix}{host_part}"),
            };
        let request_target_text = match scheme {
            #[cfg(target_os = "windows")]
            UrlScheme::Https => String::new(),
            UrlScheme::Http => {
                build_host_port_text("", port != DEFAULT_HTTP_PORT || explicit_port.is_some())
            }
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            UrlScheme::Https => build_host_port_text(HTTPS_SCHEME_PREFIX, explicit_port.is_some()),
        };
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        let request_target = CString::new(request_target_text).map_err(|source| {
            TimeError::parse_with_source("서버 HTTP 요청 대상 변환 실패", source)
        })?;
        Ok(Self {
            host: host_part.to_owned(),
            literal_tcp_socket_addr: literal_ip_addr.map(|ip| net::SocketAddr::new(ip, port)),
            port,
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            request_target,
            #[cfg(target_os = "windows")]
            request_target: request_target_text,
            scheme,
        })
    }
}
fn parse_port(port_part: &str) -> Result<u16> {
    if port_part.is_empty() || !port_part.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(TimeError::parse(ERR_PORT));
    }
    let port = parse_result_with_context(port_part.parse::<u16>(), ERR_PORT)?;
    if port == 0 {
        return Err(TimeError::parse(ERR_PORT));
    }
    Ok(port)
}
