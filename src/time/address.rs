use super::{Result, TimeError, parse_result_with_context, parse_u32_digits};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UrlScheme {
    Http,
    Https,
}

impl UrlScheme {
    pub const fn default_port(self) -> u16 {
        match self {
            Self::Http => 80,
            Self::Https => 443,
        }
    }

    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Http => "http://",
            Self::Https => "https://",
        }
    }
}

#[derive(Debug)]
pub struct ParsedServerAddress {
    scheme: Option<UrlScheme>,
    host: String,
    host_for_header: String,
    port: u16,
    explicit_port: bool,
}

impl ParsedServerAddress {
    pub const fn scheme(&self) -> Option<UrlScheme> {
        self.scheme
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    pub const fn port(&self) -> u16 {
        self.port
    }

    pub fn curl_url(&self, scheme: UrlScheme) -> String {
        let mut url = String::with_capacity(self.host_for_header.len() + 16);
        url.push_str(scheme.prefix());
        url.push_str(&self.host_for_header);
        if self.explicit_port {
            url.push(':');
            url.push_str(&self.port.to_string());
        }
        url
    }

    pub fn tcp_host_header_value(&self) -> String {
        if self.port == UrlScheme::Http.default_port() {
            return self.host_for_header.clone();
        }
        let mut host_header = String::with_capacity(self.host_for_header.len() + 8);
        host_header.push_str(&self.host_for_header);
        host_header.push(':');
        host_header.push_str(&self.port.to_string());
        host_header
    }
}

fn parse_port(port_str: &str) -> Result<u16> {
    const ERR_PORT: &str = "서버 주소 파싱 실패: 포트 번호가 유효하지 않습니다 (1~65535).";
    if port_str.is_empty() {
        return Err(TimeError::parse(ERR_PORT));
    }
    let port_num = parse_u32_digits(port_str).ok_or_else(|| TimeError::parse(ERR_PORT))?;
    let port = parse_result_with_context(u16::try_from(port_num), ERR_PORT)?;
    if port == 0 {
        return Err(TimeError::parse(ERR_PORT));
    }
    Ok(port)
}

fn parse_authority_host_port(authority: &str, default_port: u16) -> Result<(String, u16, bool)> {
    const ERR_HOST: &str = "서버 주소 파싱 실패: 호스트 값이 비어있거나 형식이 올바르지 않습니다.";
    if let Some(bracketed) = authority.strip_prefix('[') {
        let close_idx = bracketed
            .find(']')
            .ok_or_else(|| TimeError::parse(ERR_HOST))?;
        let host_part = bracketed
            .get(..close_idx)
            .ok_or_else(|| TimeError::parse(ERR_HOST))?;
        if host_part.is_empty() {
            return Err(TimeError::parse(ERR_HOST));
        }
        let rem = bracketed
            .get(close_idx + 1..)
            .ok_or_else(|| TimeError::parse(ERR_HOST))?;
        if rem.is_empty() {
            return Ok((host_part.to_owned(), default_port, false));
        }
        let port_part = rem
            .strip_prefix(':')
            .ok_or_else(|| TimeError::parse(ERR_HOST))?;
        let port = parse_port(port_part)?;
        return Ok((host_part.to_owned(), port, true));
    }
    let colon_count = authority.bytes().filter(|&b| b == b':').count();
    if colon_count == 1 {
        let (host_part, port_part) = authority
            .rsplit_once(':')
            .ok_or_else(|| TimeError::parse(ERR_HOST))?;
        if host_part.is_empty() {
            return Err(TimeError::parse(ERR_HOST));
        }
        let port = parse_port(port_part)?;
        return Ok((host_part.to_owned(), port, true));
    }
    if authority.is_empty() {
        return Err(TimeError::parse(ERR_HOST));
    }
    Ok((authority.to_owned(), default_port, false))
}

pub fn parse_server_address(raw_input: &str) -> Result<ParsedServerAddress> {
    const ERR_EMPTY: &str = "서버 주소를 비워둘 수 없습니다.";
    const ERR_HOST: &str = "서버 주소 파싱 실패: 호스트 값이 비어있거나 형식이 올바르지 않습니다.";
    let input = raw_input.trim();
    if input.is_empty() {
        return Err(TimeError::parse(ERR_EMPTY));
    }
    let input_bytes = input.as_bytes();
    let (scheme, after_scheme) = if input_bytes
        .get(..8)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"https://"))
    {
        (Some(UrlScheme::Https), input.get(8..).unwrap_or(input))
    } else if input_bytes
        .get(..7)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"http://"))
    {
        (Some(UrlScheme::Http), input.get(7..).unwrap_or(input))
    } else {
        (None, input)
    };
    let authority_end = after_scheme
        .bytes()
        .position(|b| matches!(b, b'/' | b'?' | b'#'))
        .unwrap_or(after_scheme.len());
    let authority = after_scheme.get(..authority_end).unwrap_or(after_scheme);
    if authority.is_empty() || authority.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return Err(TimeError::parse(ERR_HOST));
    }
    let default_port = scheme.map_or(UrlScheme::Http.default_port(), UrlScheme::default_port);
    let (host, port, explicit_port) = parse_authority_host_port(authority, default_port)?;
    let host_for_header = if host.contains(':') {
        format!("[{host}]")
    } else {
        host.clone()
    };
    Ok(ParsedServerAddress {
        scheme,
        host,
        host_for_header,
        port,
        explicit_port,
    })
}
