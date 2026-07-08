use super::{
    HTTP_SCHEME_PREFIX, HTTP_SCHEME_PREFIX_LEN, HTTPS_SCHEME_PREFIX, HTTPS_SCHEME_PREFIX_LEN,
    ParsedServer, Result, TimeError, UrlScheme,
    util::{parse_result_with_context, parse_u32_digits},
};
use core::str::{self, FromStr};
use std::net;
const ERR_EMPTY: &str = "서버 주소를 비워둘 수 없습니다.";
const ERR_HOST: &str = "서버 주소 파싱 실패: 호스트 값이 비어있거나 형식이 올바르지 않습니다.";
const ERR_PATH: &str = "서버 주소에는 path/query/fragment를 사용할 수 없습니다.";
const ERR_PORT: &str = "서버 주소 파싱 실패: 포트 번호가 유효하지 않습니다 (1~65535).";
const DEFAULT_HTTP_PORT: u16 = 80;
const DEFAULT_HTTPS_PORT: u16 = 443;
const PORT_SUFFIX_CAPACITY: usize = 1 + U16_DECIMAL_MAX_LEN;
const SECURE_URL_FORMAT: HostPortFormat = HostPortFormat {
    capacity_context: "URL 용량 계산 실패",
    port_context: "URL port 작성 실패",
    prefix: HTTPS_SCHEME_PREFIX,
    reserve_context: "URL 메모리 확보 실패",
};
const TCP_HOST_HEADER_FORMAT: HostPortFormat = HostPortFormat {
    capacity_context: "TCP host header 용량 계산 실패",
    port_context: "TCP host header port 작성 실패",
    prefix: "",
    reserve_context: "TCP host header 메모리 확보 실패",
};
const U16_DECIMAL_MAX_LEN: usize = 5;
#[derive(Clone, Copy)]
enum PortSource {
    Default,
    Explicit,
}
impl PortSource {
    const fn is_explicit(self) -> bool {
        matches!(self, Self::Explicit)
    }
}
struct AuthorityParts<'input> {
    bracketed: bool,
    host: &'input str,
    port: u16,
    port_source: PortSource,
}
struct HostPortFormat {
    capacity_context: &'static str,
    port_context: &'static str,
    prefix: &'static str,
    reserve_context: &'static str,
}
impl<'input> AuthorityParts<'input> {
    const fn default_port(host: &'input str, port: u16, bracketed: bool) -> Self {
        Self {
            bracketed,
            host,
            port,
            port_source: PortSource::Default,
        }
    }
    const fn explicit_port(host: &'input str, port: u16, bracketed: bool) -> Self {
        Self {
            bracketed,
            host,
            port,
            port_source: PortSource::Explicit,
        }
    }
}
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
        if after_scheme.contains(['/', '?', '#']) {
            return Err(TimeError::parse(ERR_PATH));
        }
        if after_scheme.is_empty() || after_scheme.contains(|ch: char| ch.is_ascii_whitespace()) {
            return Err(TimeError::parse(ERR_HOST));
        }
        let default_port = match scheme {
            UrlScheme::Http => DEFAULT_HTTP_PORT,
            UrlScheme::Https => DEFAULT_HTTPS_PORT,
        };
        let authority_parts = if let Some(bracketed) = after_scheme.strip_prefix('[') {
            let (host_part, rem) = bracketed
                .split_once(']')
                .ok_or_else(|| TimeError::parse(ERR_HOST))?;
            if rem.is_empty() {
                AuthorityParts::default_port(host_part, default_port, true)
            } else {
                let port_part = rem
                    .strip_prefix(':')
                    .ok_or_else(|| TimeError::parse(ERR_HOST))?;
                AuthorityParts::explicit_port(host_part, parse_port(port_part)?, true)
            }
        } else if let Some((host_part, port_part)) = after_scheme.split_once(':')
            && !port_part.contains(':')
        {
            AuthorityParts::explicit_port(host_part, parse_port(port_part)?, false)
        } else {
            AuthorityParts::default_port(after_scheme, default_port, false)
        };
        let bracketed = authority_parts.bracketed;
        let host_part = authority_parts.host;
        let port = authority_parts.port;
        let port_source = authority_parts.port_source;
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
        let host_for_header = if host_part.contains(':') {
            let mut out = String::new();
            let capacity = checked_capacity(host_part.len(), 2, "서버 host header 용량 계산 실패")?;
            reserve_string(&mut out, capacity, "서버 host header 메모리 확보 실패")?;
            out.push('[');
            out.push_str(host_part);
            out.push(']');
            out
        } else {
            copy_str(host_part, "서버 host header 메모리 확보 실패")?
        };
        let secure_url = build_host_port_text(
            &SECURE_URL_FORMAT,
            &host_for_header,
            port,
            port_source.is_explicit(),
        )?;
        let tcp_host_header = if port == DEFAULT_HTTP_PORT && !port_source.is_explicit() {
            host_for_header
        } else {
            build_host_port_text(&TCP_HOST_HEADER_FORMAT, &host_for_header, port, true)?
        };
        let host_owned = copy_str(host_part, "서버 host 메모리 확보 실패")?;
        Ok(Self {
            scheme,
            host: host_owned,
            literal_tcp_socket_addr: literal_ip_addr.map(|ip| net::SocketAddr::new(ip, port)),
            port,
            secure_url,
            tcp_host_header,
        })
    }
}
fn copy_str(value: &str, context: &'static str) -> Result<String> {
    let mut out = String::new();
    reserve_string(&mut out, value.len(), context)?;
    out.push_str(value);
    Ok(out)
}
fn build_host_port_text(
    format: &HostPortFormat,
    host_for_header: &str,
    port: u16,
    include_port: bool,
) -> Result<String> {
    let port_len = if include_port {
        PORT_SUFFIX_CAPACITY
    } else {
        0
    };
    let prefix_host_len = checked_capacity(
        format.prefix.len(),
        host_for_header.len(),
        format.capacity_context,
    )?;
    let capacity = checked_capacity(prefix_host_len, port_len, format.capacity_context)?;
    let mut out = String::new();
    reserve_string(&mut out, capacity, format.reserve_context)?;
    out.push_str(format.prefix);
    out.push_str(host_for_header);
    if include_port {
        out.push(':');
        let mut digits = [0_u8; U16_DECIMAL_MAX_LEN];
        let mut index = digits.len();
        let mut value = port;
        loop {
            let digit = u8::try_from(value.rem_euclid(10))
                .map_err(|source| TimeError::parse_with_source(format.port_context, source))?;
            index = index
                .checked_sub(1)
                .ok_or_else(|| TimeError::parse(format.port_context))?;
            let byte = b'0'
                .checked_add(digit)
                .ok_or_else(|| TimeError::parse(format.port_context))?;
            let Some(slot) = digits.get_mut(index) else {
                return Err(TimeError::parse(format.port_context));
            };
            *slot = byte;
            value = value.div_euclid(10);
            if value == 0 {
                break;
            }
        }
        let Some(port_digits) = digits.get(index..) else {
            return Err(TimeError::parse(format.port_context));
        };
        let port_text = str::from_utf8(port_digits)
            .map_err(|source| TimeError::parse_with_source(format.port_context, source))?;
        out.push_str(port_text);
    }
    Ok(out)
}
fn checked_capacity(base: usize, additional: usize, context: &'static str) -> Result<usize> {
    base.checked_add(additional)
        .ok_or_else(|| TimeError::parse(context))
}
fn reserve_string(out: &mut String, capacity: usize, context: &'static str) -> Result<()> {
    out.try_reserve_exact(capacity)
        .map_err(|source| TimeError::parse_with_source(context, source))
}
fn parse_port(port_part: &str) -> Result<u16> {
    if port_part.is_empty() {
        return Err(TimeError::parse(ERR_PORT));
    }
    let port_num = parse_u32_digits(port_part).ok_or_else(|| TimeError::parse(ERR_PORT))?;
    let port = parse_result_with_context(u16::try_from(port_num), ERR_PORT)?;
    if port == 0 {
        return Err(TimeError::parse(ERR_PORT));
    }
    Ok(port)
}
