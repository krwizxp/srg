use super::{
    CachedTcpConnection, CachedTcpSocketAddr, NetworkContext, ParsedServer, Result, TCP_TIMEOUT,
    TimeError, TimeSample, UrlScheme, http_date::parse_http_date_to_systemtime,
};
use alloc::str;
use core::result::Result as CoreResult;
use std::{
    io::{self, BufRead as IoBufRead, BufReader, Read as IoRead, Write as IoWrite},
    net::{TcpStream, ToSocketAddrs as _},
    time::{Instant, SystemTime},
};
const TCP_HEAD_REQUEST_PREFIX: &[u8] = b"HEAD / HTTP/1.1\r\nHost: ";
const TCP_HEAD_REQUEST_SUFFIX: &[u8] = b"\r\nUser-Agent: Rust-Time-Sync\r\n\r\n";
const TCP_MAX_HEADER_BYTES: usize = 64 * 1024;
const TCP_MAX_HEADER_LINE_BYTES: usize = 8192;
const TCP_MAX_HEADER_LINE_READ_BYTES: u64 = 8193;
const CONNECTION_HEADER_NAME: &[u8; 10] = b"connection";
const DATE_HEADER_PREFIX: &[u8; 5] = b"date:";
const HTTP_STATUS_PREFIX: &[u8; 5] = b"HTTP/";
const HTTP_VERSION_1_0: &[u8] = b"HTTP/1.0";
const HTTP_VERSION_1_1: &[u8] = b"HTTP/1.1";
const CONNECTION_CLOSE: &[u8] = b"close";
const CONNECTION_KEEP_ALIVE: &[u8] = b"keep-alive";
pub(super) const TCP_LINE_BUFFER_CAPACITY: usize = 256;
enum HttpVersion {
    Http10,
    Http11,
    Other,
}
struct TcpStatusLine {
    code: u16,
    version: HttpVersion,
}
struct TcpHeaderBlock {
    date_sample: Option<CoreResult<(SystemTime, Instant), TimeError>>,
    has_connection_close: bool,
    has_connection_keep_alive: bool,
}
struct TcpResponseSample {
    response_received_inst: Instant,
    reusable: bool,
    server_time: SystemTime,
}
struct TcpAttemptError {
    error: TimeError,
    retryable: bool,
}
struct TcpResponseReader<'io, 'ctx> {
    header_bytes_seen: usize,
    net_ctx: &'ctx mut NetworkContext,
    reader: &'io mut BufReader<TcpStream>,
}
impl TcpAttemptError {
    const fn non_retryable(error: TimeError) -> Self {
        Self {
            error,
            retryable: false,
        }
    }
    const fn retryable(error: TimeError) -> Self {
        Self {
            error,
            retryable: true,
        }
    }
}
impl TcpHeaderBlock {
    const fn reusable(&self, version: &HttpVersion) -> bool {
        if self.has_connection_close {
            return false;
        }
        match *version {
            HttpVersion::Http11 => true,
            HttpVersion::Http10 => self.has_connection_keep_alive,
            HttpVersion::Other => false,
        }
    }
}
impl TcpResponseReader<'_, '_> {
    fn read_header_block(&mut self) -> CoreResult<TcpHeaderBlock, TcpAttemptError> {
        let mut header_block = TcpHeaderBlock {
            date_sample: None,
            has_connection_close: false,
            has_connection_keep_alive: false,
        };
        loop {
            let bytes_read = self.read_line()?;
            if bytes_read == 0 {
                return Err(TcpAttemptError::retryable(missing_tcp_date()));
            }
            if self.net_ctx.tcp_line_buffer == b"\r\n" || self.net_ctx.tcp_line_buffer == b"\n" {
                return Ok(header_block);
            }
            if header_block.date_sample.is_none() {
                let maybe_date_str = find_date_header_value(&self.net_ctx.tcp_line_buffer)
                    .map_err(|source| {
                        TcpAttemptError::non_retryable(TimeError::parse(format!(
                            "TCP Date 헤더 UTF-8 변환 실패: {source}"
                        )))
                    })?;
                if let Some(date_str) = maybe_date_str {
                    let response_received_inst = Instant::now();
                    header_block.date_sample = Some(
                        parse_http_date_to_systemtime(date_str)
                            .map(|server_time| (server_time, response_received_inst)),
                    );
                }
            }
            self.update_connection_tokens(&mut header_block);
        }
    }
    fn read_line(&mut self) -> CoreResult<usize, TcpAttemptError> {
        self.net_ctx.tcp_line_buffer.clear();
        let bytes_read = IoBufRead::read_until(
            &mut IoRead::take(&mut *self.reader, TCP_MAX_HEADER_LINE_READ_BYTES),
            b'\n',
            &mut self.net_ctx.tcp_line_buffer,
        )
        .map_err(|source| TcpAttemptError::retryable(TimeError::from(source)))?;
        self.header_bytes_seen =
            self.header_bytes_seen
                .checked_add(bytes_read)
                .ok_or_else(|| {
                    TcpAttemptError::non_retryable(TimeError::parse("TCP HTTP 헤더 길이 계산 실패"))
                })?;
        if self.header_bytes_seen > TCP_MAX_HEADER_BYTES {
            return Err(TcpAttemptError::non_retryable(TimeError::parse(
                "TCP HTTP 헤더가 너무 큽니다.",
            )));
        }
        let line_ended = self.net_ctx.tcp_line_buffer.ends_with(b"\n");
        if self.net_ctx.tcp_line_buffer.len() > TCP_MAX_HEADER_LINE_BYTES
            || (!line_ended && self.net_ctx.tcp_line_buffer.len() == TCP_MAX_HEADER_LINE_BYTES)
        {
            return Err(TcpAttemptError::non_retryable(TimeError::parse(
                "TCP HTTP 헤더 line이 너무 깁니다.",
            )));
        }
        Ok(bytes_read)
    }
    fn read_response(&mut self) -> CoreResult<TcpResponseSample, TcpAttemptError> {
        loop {
            let status = self.read_status_line()?;
            let header_block = self.read_header_block()?;
            if (100..=199).contains(&status.code) {
                continue;
            }
            let reusable = header_block.reusable(&status.version);
            let Some(final_date_sample) = header_block.date_sample else {
                return Err(TcpAttemptError::non_retryable(missing_tcp_date()));
            };
            let (server_time, response_received_inst) =
                final_date_sample.map_err(TcpAttemptError::non_retryable)?;
            return Ok(TcpResponseSample {
                response_received_inst,
                reusable,
                server_time,
            });
        }
    }
    fn read_status_line(&mut self) -> CoreResult<TcpStatusLine, TcpAttemptError> {
        let status_bytes_read = self.read_line()?;
        if status_bytes_read == 0 {
            return Err(TcpAttemptError::retryable(missing_tcp_date()));
        }
        let status_line = trim_http_line_end(&self.net_ctx.tcp_line_buffer);
        if !status_line.starts_with(HTTP_STATUS_PREFIX) {
            return Err(TcpAttemptError::non_retryable(TimeError::parse(
                "TCP HTTP 상태 행 형식이 올바르지 않습니다.",
            )));
        }
        let Some(version_end) = status_line.iter().position(|&byte| byte == b' ') else {
            return Err(TcpAttemptError::non_retryable(TimeError::parse(
                "TCP HTTP 상태 행 형식이 올바르지 않습니다.",
            )));
        };
        let version = match status_line.get(..version_end) {
            Some(HTTP_VERSION_1_0) => HttpVersion::Http10,
            Some(HTTP_VERSION_1_1) => HttpVersion::Http11,
            Some(_) => HttpVersion::Other,
            None => {
                return Err(TcpAttemptError::non_retryable(TimeError::parse(
                    "TCP HTTP 상태 행 형식이 올바르지 않습니다.",
                )));
            }
        };
        let Some(status_tail_raw) = status_line.get(version_end..) else {
            return Err(TcpAttemptError::non_retryable(TimeError::parse(
                "TCP HTTP 상태 행 형식이 올바르지 않습니다.",
            )));
        };
        let status_tail = trim_ascii_bytes(status_tail_raw);
        let Some((status_digits, _)) = status_tail.split_first_chunk::<3>() else {
            return Err(TcpAttemptError::non_retryable(TimeError::parse(
                "TCP HTTP 상태 코드가 올바르지 않습니다.",
            )));
        };
        let [hundreds, tens, ones] = *status_digits;
        let Some(hundreds_digit) = hundreds.checked_sub(b'0') else {
            return Err(TcpAttemptError::non_retryable(TimeError::parse(
                "TCP HTTP 상태 코드가 올바르지 않습니다.",
            )));
        };
        let Some(tens_digit) = tens.checked_sub(b'0') else {
            return Err(TcpAttemptError::non_retryable(TimeError::parse(
                "TCP HTTP 상태 코드가 올바르지 않습니다.",
            )));
        };
        let Some(ones_digit) = ones.checked_sub(b'0') else {
            return Err(TcpAttemptError::non_retryable(TimeError::parse(
                "TCP HTTP 상태 코드가 올바르지 않습니다.",
            )));
        };
        if !(hundreds_digit <= 9 && tens_digit <= 9 && ones_digit <= 9) {
            return Err(TcpAttemptError::non_retryable(TimeError::parse(
                "TCP HTTP 상태 코드가 올바르지 않습니다.",
            )));
        }
        let code = u16::from(hundreds_digit)
            .checked_mul(100)
            .and_then(|hundreds_value| {
                let tens_value = u16::from(tens_digit).checked_mul(10)?;
                hundreds_value.checked_add(tens_value)
            })
            .and_then(|value| value.checked_add(u16::from(ones_digit)))
            .ok_or_else(|| {
                TcpAttemptError::non_retryable(TimeError::parse(
                    "TCP HTTP 상태 코드가 올바르지 않습니다.",
                ))
            })?;
        Ok(TcpStatusLine { code, version })
    }
    fn update_connection_tokens(&self, header_block: &mut TcpHeaderBlock) {
        let header_line = trim_http_line_end(&self.net_ctx.tcp_line_buffer);
        let Some(header_name_end) = header_line.iter().position(|&byte| byte == b':') else {
            return;
        };
        let Some(header_value_start) = header_name_end.checked_add(1) else {
            return;
        };
        let Some(name) = header_line.get(..header_name_end) else {
            return;
        };
        let Some(header_value) = header_line.get(header_value_start..) else {
            return;
        };
        if !name.eq_ignore_ascii_case(CONNECTION_HEADER_NAME) {
            return;
        }
        for raw_token in header_value.split(|&byte| byte == b',') {
            let token = trim_ascii_bytes(raw_token);
            match token {
                connection_token if connection_token.eq_ignore_ascii_case(CONNECTION_CLOSE) => {
                    header_block.has_connection_close = true;
                }
                connection_token
                    if connection_token.eq_ignore_ascii_case(CONNECTION_KEEP_ALIVE) =>
                {
                    header_block.has_connection_keep_alive = true;
                }
                _ => {}
            }
        }
    }
}
pub(super) fn find_date_header_value(line: &[u8]) -> CoreResult<Option<&str>, str::Utf8Error> {
    let Some((prefix, value)) = line.split_first_chunk::<5>() else {
        return Ok(None);
    };
    if !prefix.eq_ignore_ascii_case(DATE_HEADER_PREFIX) {
        return Ok(None);
    }
    str::from_utf8(value).map(str::trim_ascii).map(Some)
}
fn missing_tcp_date() -> TimeError {
    TimeError::header_not_found("Date (TCP)")
}
pub(super) fn fetch_server_time_sample(
    parsed_address: &ParsedServer,
    net_ctx: &mut NetworkContext,
) -> Result<TimeSample> {
    if matches!(parsed_address.scheme(), UrlScheme::Https) {
        return net_ctx
            .native_http
            .fetch_head_sample(parsed_address.url(UrlScheme::Https), "HTTPS");
    }
    let cached_connection = if net_ctx
        .cached_tcp_connection
        .as_ref()
        .is_some_and(|cached| {
            cached.host.as_str() == parsed_address.host() && cached.port == parsed_address.port()
        }) {
        net_ctx.cached_tcp_connection.take()
    } else {
        net_ctx.cached_tcp_connection = None;
        None
    };
    let connection = if let Some(connection) = cached_connection {
        match sample_from_tcp_connection(parsed_address, net_ctx, connection) {
            Ok(sample) => return Ok(sample),
            Err(attempt_error) if attempt_error.retryable => {
                connect_tcp_connection(parsed_address, net_ctx)?
            }
            Err(attempt_error) => return Err(attempt_error.error),
        }
    } else {
        connect_tcp_connection(parsed_address, net_ctx)?
    };
    sample_from_tcp_connection(parsed_address, net_ctx, connection)
        .map_err(|attempt_error| attempt_error.error)
}
fn connect_tcp_connection(
    parsed_address: &ParsedServer,
    net_ctx: &mut NetworkContext,
) -> Result<CachedTcpConnection> {
    let stream = 'connect: {
        if let Some(cached_socket_addr) = parsed_address.literal_tcp_socket_addr() {
            break 'connect TcpStream::connect_timeout(&cached_socket_addr, TCP_TIMEOUT)?;
        }
        if let Some(cached_socket_addr) = net_ctx
            .cached_tcp_socket_addr
            .as_ref()
            .filter(|cached| {
                cached.host.as_str() == parsed_address.host()
                    && cached.port == parsed_address.port()
            })
            .map(|cached| cached.addr)
        {
            if let Ok(stream) = TcpStream::connect_timeout(&cached_socket_addr, TCP_TIMEOUT) {
                break 'connect stream;
            }
            net_ctx.cached_tcp_socket_addr = None;
        }
        let mut last_connect_error = None;
        let addrs = (parsed_address.host(), parsed_address.port()).to_socket_addrs()?;
        for socket_addr in addrs {
            match TcpStream::connect_timeout(&socket_addr, TCP_TIMEOUT) {
                Ok(stream) => {
                    let mut host = String::new();
                    host.try_reserve(parsed_address.host().len())
                        .map_err(|source| {
                            TimeError::parse(format!("TCP cache host 메모리 확보 실패: {source}"))
                        })?;
                    host.push_str(parsed_address.host());
                    net_ctx.cached_tcp_socket_addr = Some(CachedTcpSocketAddr {
                        addr: socket_addr,
                        host,
                        port: parsed_address.port(),
                    });
                    break 'connect stream;
                }
                Err(source) => {
                    last_connect_error = Some(source);
                }
            }
        }
        let host_not_found = || io::Error::new(io::ErrorKind::NotFound, "Host not found");
        return Err(TimeError::from(
            last_connect_error.unwrap_or_else(host_not_found),
        ));
    };
    stream.set_read_timeout(Some(TCP_TIMEOUT))?;
    stream.set_write_timeout(Some(TCP_TIMEOUT))?;
    stream.set_nodelay(true)?;
    let mut host = String::new();
    host.try_reserve(parsed_address.host().len())
        .map_err(|source| TimeError::parse(format!("TCP cache host 메모리 확보 실패: {source}")))?;
    host.push_str(parsed_address.host());
    Ok(CachedTcpConnection {
        host,
        port: parsed_address.port(),
        reader: BufReader::new(stream),
    })
}
fn sample_from_tcp_connection(
    parsed_address: &ParsedServer,
    net_ctx: &mut NetworkContext,
    mut connection: CachedTcpConnection,
) -> CoreResult<TimeSample, TcpAttemptError> {
    net_ctx.tcp_request_buffer.clear();
    let host_header = parsed_address.tcp_host_header_value();
    let request_len = TCP_HEAD_REQUEST_PREFIX
        .len()
        .checked_add(host_header.len())
        .and_then(|value| value.checked_add(TCP_HEAD_REQUEST_SUFFIX.len()))
        .ok_or_else(|| {
            TcpAttemptError::non_retryable(TimeError::parse("TCP HTTP 요청 길이 계산 실패"))
        })?;
    net_ctx
        .tcp_request_buffer
        .try_reserve(request_len)
        .map_err(|source| {
            TcpAttemptError::non_retryable(TimeError::from(io::Error::other(source)))
        })?;
    net_ctx
        .tcp_request_buffer
        .extend_from_slice(TCP_HEAD_REQUEST_PREFIX);
    net_ctx
        .tcp_request_buffer
        .extend_from_slice(host_header.as_bytes());
    net_ctx
        .tcp_request_buffer
        .extend_from_slice(TCP_HEAD_REQUEST_SUFFIX);
    let request_start_inst = Instant::now();
    IoWrite::write_all(connection.reader.get_mut(), &net_ctx.tcp_request_buffer)
        .map_err(|source| TcpAttemptError::retryable(TimeError::from(source)))?;
    let response = TcpResponseReader {
        header_bytes_seen: 0,
        net_ctx,
        reader: &mut connection.reader,
    }
    .read_response()?;
    let rtt_for_sample = response
        .response_received_inst
        .duration_since(request_start_inst);
    if response.reusable {
        net_ctx.cached_tcp_connection = Some(connection);
    }
    Ok(TimeSample {
        response_received_inst: response.response_received_inst,
        rtt: rtt_for_sample,
        server_time: response.server_time,
    })
}
fn trim_http_line_end(mut line: &[u8]) -> &[u8] {
    if let Some(stripped) = line.strip_suffix(b"\n") {
        line = stripped;
    }
    if let Some(stripped) = line.strip_suffix(b"\r") {
        line = stripped;
    }
    line
}
const fn trim_ascii_bytes(mut bytes: &[u8]) -> &[u8] {
    while let Some((&first, rest)) = bytes.split_first()
        && first.is_ascii_whitespace()
    {
        bytes = rest;
    }
    while let Some((&last, rest)) = bytes.split_last()
        && last.is_ascii_whitespace()
    {
        bytes = rest;
    }
    bytes
}
