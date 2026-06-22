use super::{
    HTTP_SCHEME_PREFIX, HTTP_SCHEME_PREFIX_LEN, HTTPS_SCHEME_PREFIX, HTTPS_SCHEME_PREFIX_LEN,
    ParsedServer, Result, TimeError, UrlScheme,
    util::{parse_result_with_context, parse_u32_digits},
};
use core::{fmt::Write as FmtWrite, str::FromStr};
use std::net;
const ERR_EMPTY: &str = "서버 주소를 비워둘 수 없습니다.";
const ERR_HOST: &str = "서버 주소 파싱 실패: 호스트 값이 비어있거나 형식이 올바르지 않습니다.";
const ERR_PATH: &str = "서버 주소에는 path/query/fragment를 사용할 수 없습니다.";
const ERR_PORT: &str = "서버 주소 파싱 실패: 포트 번호가 유효하지 않습니다 (1~65535).";
const DEFAULT_HTTP_PORT: u16 = 80;
const DEFAULT_HTTPS_PORT: u16 = 443;
const PORT_SUFFIX_CAPACITY: usize = 1 + U16_DECIMAL_MAX_LEN;
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
    host: &'input str,
    port: u16,
    port_source: PortSource,
}
impl<'input> AuthorityParts<'input> {
    const fn default_port(host: &'input str, port: u16) -> Self {
        Self {
            host,
            port,
            port_source: PortSource::Default,
        }
    }
    const fn explicit_port(host: &'input str, port: u16) -> Self {
        Self {
            host,
            port,
            port_source: PortSource::Explicit,
        }
    }
}
impl ParsedServer {
    pub(super) fn host(&self) -> &str {
        &self.host
    }
    pub(super) const fn literal_tcp_socket_addr(&self) -> Option<net::SocketAddr> {
        self.literal_tcp_socket_addr
    }
    pub(super) const fn port(&self) -> u16 {
        self.port
    }
    pub(super) const fn scheme(&self) -> UrlScheme {
        match self.scheme {
            Some(scheme) => scheme,
            None => UrlScheme::Https,
        }
    }
    pub(super) fn tcp_host_header_value(&self) -> &str {
        &self.tcp_host_header
    }
    pub(super) fn url(&self, scheme: UrlScheme) -> &str {
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
        let (scheme, after_scheme) = if let Some((prefix, rest)) =
            trimmed_input.split_at_checked(HTTPS_SCHEME_PREFIX_LEN)
            && prefix.eq_ignore_ascii_case(HTTPS_SCHEME_PREFIX)
        {
            (Some(UrlScheme::Https), rest)
        } else if let Some((prefix, rest)) = trimmed_input.split_at_checked(HTTP_SCHEME_PREFIX_LEN)
            && prefix.eq_ignore_ascii_case(HTTP_SCHEME_PREFIX)
        {
            (Some(UrlScheme::Http), rest)
        } else {
            (None, trimmed_input)
        };
        if after_scheme.contains(['/', '?', '#']) {
            return Err(TimeError::parse(ERR_PATH));
        }
        if after_scheme.is_empty() || after_scheme.contains(|ch: char| ch.is_ascii_whitespace()) {
            return Err(TimeError::parse(ERR_HOST));
        }
        let default_port = if scheme == Some(UrlScheme::Http) {
            DEFAULT_HTTP_PORT
        } else {
            DEFAULT_HTTPS_PORT
        };
        let authority_parts = if let Some(bracketed) = after_scheme.strip_prefix('[') {
            let (host_part, rem) = bracketed
                .split_once(']')
                .ok_or_else(|| TimeError::parse(ERR_HOST))?;
            if rem.is_empty() {
                AuthorityParts::default_port(host_part, default_port)
            } else {
                let port_part = rem
                    .strip_prefix(':')
                    .ok_or_else(|| TimeError::parse(ERR_HOST))?;
                AuthorityParts::explicit_port(host_part, parse_port(port_part)?)
            }
        } else if let Some((host_part, port_part)) = after_scheme.split_once(':')
            && !port_part.contains(':')
        {
            AuthorityParts::explicit_port(host_part, parse_port(port_part)?)
        } else {
            AuthorityParts::default_port(after_scheme, default_port)
        };
        let AuthorityParts {
            host: host_part,
            port,
            port_source,
        } = authority_parts;
        if host_part.is_empty() {
            return Err(TimeError::parse(ERR_HOST));
        }
        let host_for_header = if host_part.contains(':') {
            let mut out = String::new();
            let capacity = checked_capacity(host_part.len(), 2, "서버 host header 용량 계산 실패")?;
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
        let plain_url = build_url(&host_for_header, port_source, port, UrlScheme::Http)?;
        let secure_url = build_url(&host_for_header, port_source, port, UrlScheme::Https)?;
        let tcp_host_header = if port == DEFAULT_HTTP_PORT {
            host_for_header
        } else {
            let mut out = String::new();
            let capacity = checked_capacity(
                host_for_header.len(),
                PORT_SUFFIX_CAPACITY,
                "TCP host header 용량 계산 실패",
            )?;
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
            literal_tcp_socket_addr: host_part
                .parse::<net::IpAddr>()
                .ok()
                .map(|ip_addr| net::SocketAddr::new(ip_addr, port)),
            port,
            secure_url,
            tcp_host_header,
        })
    }
}
fn build_url(
    host_for_header: &str,
    port_source: PortSource,
    port: u16,
    scheme: UrlScheme,
) -> Result<String> {
    let scheme_prefix = match scheme {
        UrlScheme::Http => HTTP_SCHEME_PREFIX,
        UrlScheme::Https => HTTPS_SCHEME_PREFIX,
    };
    let mut out = String::new();
    let explicit_port_len = if port_source.is_explicit() {
        PORT_SUFFIX_CAPACITY
    } else {
        0
    };
    let prefix_host_len = checked_capacity(
        scheme_prefix.len(),
        host_for_header.len(),
        "URL 용량 계산 실패",
    )?;
    let capacity = checked_capacity(prefix_host_len, explicit_port_len, "URL 용량 계산 실패")?;
    out.try_reserve(capacity)
        .map_err(|source| TimeError::parse(format!("URL 메모리 확보 실패: {source}")))?;
    out.push_str(scheme_prefix);
    out.push_str(host_for_header);
    if port_source.is_explicit() {
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
fn checked_capacity(base: usize, additional: usize, context: &'static str) -> Result<usize> {
    base.checked_add(additional)
        .ok_or_else(|| TimeError::parse(context))
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
