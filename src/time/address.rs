use super::{Result, TimeError, parse_result_with_context, parse_u32_digits};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UrlScheme {
    Http,
    Https,
}
#[derive(Debug)]
pub struct ParsedServer {
    pub explicit_port: bool,
    pub host: String,
    pub host_for_header: String,
    pub port: u16,
    pub scheme: Option<UrlScheme>,
}
impl ParsedServer {
    pub fn curl_url(&self, scheme: UrlScheme) -> String {
        let capacity = self
            .host_for_header
            .len()
            .checked_add(16)
            .unwrap_or(self.host_for_header.len());
        let mut url = String::with_capacity(capacity);
        url.push_str(match scheme {
            UrlScheme::Http => "http://",
            UrlScheme::Https => "https://",
        });
        url.push_str(&self.host_for_header);
        if self.explicit_port {
            url.push(':');
            url.push_str(&self.port.to_string());
        }
        url
    }
    pub fn host(&self) -> &str {
        &self.host
    }
    pub const fn port(&self) -> u16 {
        self.port
    }
    pub const fn scheme(&self) -> Option<UrlScheme> {
        self.scheme
    }
    pub fn tcp_host_header_value(&self) -> String {
        if self.port == 80 {
            return self.host_for_header.clone();
        }
        let capacity = self
            .host_for_header
            .len()
            .checked_add(8)
            .unwrap_or(self.host_for_header.len());
        let mut host_header = String::with_capacity(capacity);
        host_header.push_str(&self.host_for_header);
        host_header.push(':');
        host_header.push_str(&self.port.to_string());
        host_header
    }
}
pub fn parse_server(host: &str) -> Result<ParsedServer> {
    const ERR_EMPTY: &str = "서버 주소를 비워둘 수 없습니다.";
    const ERR_HOST: &str = "서버 주소 파싱 실패: 호스트 값이 비어있거나 형식이 올바르지 않습니다.";
    const ERR_PORT: &str = "서버 주소 파싱 실패: 포트 번호가 유효하지 않습니다 (1~65535).";
    let input = host.trim();
    if input.is_empty() {
        return Err(TimeError::parse(ERR_EMPTY));
    }
    let input_bytes = input.as_bytes();
    let (scheme, after_scheme) = if input_bytes
        .get(..8)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"https://"))
    {
        (Some(UrlScheme::Https), input.get(8..).unwrap_or(input))
    } else if input_bytes
        .get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"http://"))
    {
        (Some(UrlScheme::Http), input.get(7..).unwrap_or(input))
    } else {
        (None, input)
    };
    let authority_end = after_scheme
        .bytes()
        .position(|byte| matches!(byte, b'/' | b'?' | b'#'))
        .unwrap_or(after_scheme.len());
    let authority = after_scheme.get(..authority_end).unwrap_or(after_scheme);
    if authority.is_empty() || authority.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return Err(TimeError::parse(ERR_HOST));
    }
    let default_port = match scheme {
        Some(UrlScheme::Https) => 443,
        Some(UrlScheme::Http) | None => 80,
    };
    let (host_name, port, explicit_port) = if let Some(bracketed) = authority.strip_prefix('[') {
        let close_idx = bracketed
            .find(']')
            .ok_or_else(|| TimeError::parse(ERR_HOST))?;
        let host_part = bracketed
            .get(..close_idx)
            .ok_or_else(|| TimeError::parse(ERR_HOST))?;
        if host_part.is_empty() {
            return Err(TimeError::parse(ERR_HOST));
        }
        let rem_start = close_idx
            .checked_add(1)
            .ok_or_else(|| TimeError::parse(ERR_HOST))?;
        let rem = bracketed
            .get(rem_start..)
            .ok_or_else(|| TimeError::parse(ERR_HOST))?;
        if rem.is_empty() {
            (host_part.to_owned(), default_port, false)
        } else {
            let port_part = rem
                .strip_prefix(':')
                .ok_or_else(|| TimeError::parse(ERR_HOST))?;
            if port_part.is_empty() {
                return Err(TimeError::parse(ERR_PORT));
            }
            let port_num = (parse_u32_digits(port_part).ok_or_else(|| TimeError::parse(ERR_PORT)))?;
            let port = (parse_result_with_context(u16::try_from(port_num), ERR_PORT))?;
            if port == 0 {
                return Err(TimeError::parse(ERR_PORT));
            }
            (host_part.to_owned(), port, true)
        }
    } else {
        let colon_count = authority.bytes().filter(|&byte| byte == b':').count();
        if colon_count == 1 {
            let (host_part, port_part) = (authority
                .rsplit_once(':')
                .ok_or_else(|| TimeError::parse(ERR_HOST)))?;
            if host_part.is_empty() {
                return Err(TimeError::parse(ERR_HOST));
            }
            if port_part.is_empty() {
                return Err(TimeError::parse(ERR_PORT));
            }
            let port_num = (parse_u32_digits(port_part).ok_or_else(|| TimeError::parse(ERR_PORT)))?;
            let port = (parse_result_with_context(u16::try_from(port_num), ERR_PORT))?;
            if port == 0 {
                return Err(TimeError::parse(ERR_PORT));
            }
            (host_part.to_owned(), port, true)
        } else {
            (authority.to_owned(), default_port, false)
        }
    };
    let host_for_header = if host_name.contains(':') {
        format!("[{host_name}]")
    } else {
        host_name.clone()
    };
    Ok(ParsedServer {
        scheme,
        host: host_name,
        host_for_header,
        port,
        explicit_port,
    })
}
