use super::{
    Result, TimeError,
    util::{parse_result_with_context, parse_u32_digits},
};
use core::{fmt::Write as FmtWrite, str::FromStr};
use std::net;
const ERR_EMPTY: &str = "서버 주소를 비워둘 수 없습니다.";
const ERR_HOST: &str = "서버 주소 파싱 실패: 호스트 값이 비어있거나 형식이 올바르지 않습니다.";
const ERR_PORT: &str = "서버 주소 파싱 실패: 포트 번호가 유효하지 않습니다 (1~65535).";
const DEFAULT_HTTP_PORT: u16 = 80;
const DEFAULT_HTTPS_PORT: u16 = 443;
const HTTP_SCHEME_PREFIX: &str = "http://";
const HTTP_SCHEME_PREFIX_LEN: usize = HTTP_SCHEME_PREFIX.len();
const HTTPS_SCHEME_PREFIX: &str = "https://";
const HTTPS_SCHEME_PREFIX_LEN: usize = HTTPS_SCHEME_PREFIX.len();
const U16_DECIMAL_MAX_LEN: usize = 5;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UrlScheme {
    Http,
    Https,
}
#[derive(Debug)]
pub struct ParsedServer {
    host: String,
    http_url: String,
    literal_tcp_socket_addr: Option<net::SocketAddr>,
    port: u16,
    scheme: Option<UrlScheme>,
    secure_url: String,
    tcp_host_header: String,
}
impl ParsedServer {
    pub fn host(&self) -> &str {
        &self.host
    }
    pub const fn literal_tcp_socket_addr(&self) -> Option<net::SocketAddr> {
        self.literal_tcp_socket_addr
    }
    pub const fn port(&self) -> u16 {
        self.port
    }
    pub const fn scheme(&self) -> Option<UrlScheme> {
        self.scheme
    }
    pub fn tcp_host_header_value(&self) -> &str {
        &self.tcp_host_header
    }
    pub fn url(&self, scheme: UrlScheme) -> &str {
        match scheme {
            UrlScheme::Http => &self.http_url,
            UrlScheme::Https => &self.secure_url,
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
        let after_scheme = strip_scheme_prefix(trimmed_input);
        let scheme = if after_scheme.len() == trimmed_input.len() {
            None
        } else if trimmed_input
            .split_at_checked(HTTPS_SCHEME_PREFIX_LEN)
            .is_some_and(|(prefix, _)| prefix.eq_ignore_ascii_case(HTTPS_SCHEME_PREFIX))
        {
            Some(UrlScheme::Https)
        } else {
            Some(UrlScheme::Http)
        };
        let authority = match after_scheme.split_once(['/', '?', '#']) {
            Some((authority, _)) => authority,
            None => after_scheme,
        };
        if authority.is_empty() || authority.contains(|ch: char| ch.is_ascii_whitespace()) {
            return Err(TimeError::parse(ERR_HOST));
        }
        let default_port = if scheme.is_some_and(|scheme_value| scheme_value == UrlScheme::Https) {
            DEFAULT_HTTPS_PORT
        } else {
            DEFAULT_HTTP_PORT
        };
        let colon_count = authority.matches(':').count();
        let (host_part, port, explicit_port) = if let Some(bracketed) = authority.strip_prefix('[')
        {
            let (host_part, rem) = bracketed
                .split_once(']')
                .ok_or_else(|| TimeError::parse(ERR_HOST))?;
            if rem.is_empty() {
                (host_part, default_port, false)
            } else {
                let port_part = rem
                    .strip_prefix(':')
                    .ok_or_else(|| TimeError::parse(ERR_HOST))?;
                (host_part, parse_port(port_part)?, true)
            }
        } else if colon_count == 1 {
            let (host_part, port_part) = authority
                .rsplit_once(':')
                .ok_or_else(|| TimeError::parse(ERR_HOST))?;
            (host_part, parse_port(port_part)?, true)
        } else {
            (authority, default_port, false)
        };
        if host_part.is_empty() {
            return Err(TimeError::parse(ERR_HOST));
        }
        let literal_tcp_socket_addr = host_part
            .parse::<net::IpAddr>()
            .ok()
            .map(|ip_addr| net::SocketAddr::new(ip_addr, port));
        let host_for_header = if host_part.contains(':') {
            let mut out = String::new();
            let capacity = host_part.len().saturating_add(2);
            out.try_reserve(capacity).map_err(|source| {
                TimeError::parse(format!("서버 host header 메모리 확보 실패: {source}"))
            })?;
            out.push('[');
            out.push_str(host_part);
            out.push(']');
            out
        } else {
            copy_str(host_part, "서버 host header 메모리 확보 실패")?
        };
        let plain_url = build_url(&host_for_header, explicit_port, port, UrlScheme::Http)?;
        let secure_url = build_url(&host_for_header, explicit_port, port, UrlScheme::Https)?;
        let tcp_host_header = if port == DEFAULT_HTTP_PORT {
            host_for_header
        } else {
            let mut out = String::new();
            let capacity = host_for_header
                .len()
                .saturating_add(":".len())
                .saturating_add(U16_DECIMAL_MAX_LEN);
            out.try_reserve(capacity).map_err(|source| {
                TimeError::parse(format!("TCP host header 메모리 확보 실패: {source}"))
            })?;
            out.push_str(&host_for_header);
            FmtWrite::write_fmt(&mut out, format_args!(":{port}")).map_err(|error| {
                TimeError::parse(format!("TCP host header port 작성 실패: {error}"))
            })?;
            out
        };
        let host_owned = copy_str(host_part, "서버 host 메모리 확보 실패")?;
        Ok(Self {
            scheme,
            host: host_owned,
            http_url: plain_url,
            literal_tcp_socket_addr,
            port,
            secure_url,
            tcp_host_header,
        })
    }
}
pub const fn strip_scheme_prefix(input: &str) -> &str {
    if let Some((prefix, rest)) = input.split_at_checked(HTTPS_SCHEME_PREFIX_LEN)
        && prefix.eq_ignore_ascii_case(HTTPS_SCHEME_PREFIX)
    {
        rest
    } else if let Some((prefix, rest)) = input.split_at_checked(HTTP_SCHEME_PREFIX_LEN)
        && prefix.eq_ignore_ascii_case(HTTP_SCHEME_PREFIX)
    {
        rest
    } else {
        input
    }
}
fn build_url(
    host_for_header: &str,
    explicit_port: bool,
    port: u16,
    scheme: UrlScheme,
) -> Result<String> {
    let scheme_prefix = match scheme {
        UrlScheme::Http => HTTP_SCHEME_PREFIX,
        UrlScheme::Https => HTTPS_SCHEME_PREFIX,
    };
    let mut out = String::new();
    let capacity = scheme_prefix
        .len()
        .saturating_add(host_for_header.len())
        .saturating_add(if explicit_port {
            ":".len().saturating_add(U16_DECIMAL_MAX_LEN)
        } else {
            0
        });
    out.try_reserve(capacity)
        .map_err(|source| TimeError::parse(format!("URL 메모리 확보 실패: {source}")))?;
    out.push_str(scheme_prefix);
    out.push_str(host_for_header);
    if explicit_port {
        FmtWrite::write_fmt(&mut out, format_args!(":{port}"))
            .map_err(|error| TimeError::parse(format!("URL port 작성 실패: {error}")))?;
    }
    Ok(out)
}
fn copy_str(value: &str, context: &'static str) -> Result<String> {
    let mut out = String::new();
    out.try_reserve(value.len())
        .map_err(|source| TimeError::parse(format!("{context}: {source}")))?;
    out.push_str(value);
    Ok(out)
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
