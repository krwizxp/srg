use super::{
    CachedTcpConnection, CachedTcpSocketAddr, NetworkContext, ParsedServer, Result, TCP_TIMEOUT,
    TimeError, TimeSample, UrlScheme, http_date::parse_http_date_to_systemtime,
};
use core::{result::Result as CoreResult, str, time::Duration};
use std::{
    io::{self, BufRead as IoBufRead, BufReader, Write as IoWrite},
    net::{TcpStream, ToSocketAddrs as _},
    time::{Instant, SystemTime},
};
const TCP_HEAD_REQUEST_PREFIX: &[u8] = b"HEAD / HTTP/1.1\r\nHost: ";
const TCP_HEAD_REQUEST_SUFFIX: &[u8] = b"\r\nUser-Agent: Rust-Time-Sync\r\n\r\n";
const TCP_MAX_HEADER_BYTES: usize = 64 * 1024;
const TCP_MAX_HEADER_LINE_BYTES: usize = 8192;
const TCP_MAX_INFORMATIONAL_RESPONSES: u8 = 16;
pub(super) const AGE_HEADER_PREFIX: &[u8; 4] = b"age:";
const CONNECTION_HEADER_NAME: &[u8; 10] = b"connection";
pub(super) const DATE_HEADER_PREFIX: &[u8; 5] = b"date:";
const HTTP_STATUS_PREFIX: &[u8; 5] = b"HTTP/";
const HTTP_VERSION_1_0: &[u8] = b"HTTP/1.0";
const HTTP_VERSION_1_1: &[u8] = b"HTTP/1.1";
const CONNECTION_CLOSE: &[u8] = b"close";
const CONNECTION_KEEP_ALIVE: &[u8] = b"keep-alive";
pub(super) const TCP_LINE_BUFFER_CAPACITY: usize = 256;
#[derive(Clone, Copy)]
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
    age_result: Option<Result<()>>,
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
    deadline: Instant,
    header_bytes_seen: usize,
    line_buffer: &'ctx mut Vec<u8>,
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
    const fn reusable(&self, version: HttpVersion) -> bool {
        if self.has_connection_close {
            return false;
        }
        match version {
            HttpVersion::Http11 => true,
            HttpVersion::Http10 => self.has_connection_keep_alive,
            HttpVersion::Other => false,
        }
    }
}
impl TcpResponseReader<'_, '_> {
    fn read_header_block(&mut self) -> CoreResult<TcpHeaderBlock, TcpAttemptError> {
        let mut header_block = TcpHeaderBlock {
            age_result: None,
            date_sample: None,
            has_connection_close: false,
            has_connection_keep_alive: false,
        };
        loop {
            let bytes_read = self.read_line()?;
            if bytes_read == 0 {
                return Err(TcpAttemptError::retryable(missing_tcp_date()));
            }
            if self.line_buffer == b"\r\n" || self.line_buffer == b"\n" {
                return Ok(header_block);
            }
            let maybe_date_str =
                find_header_value(self.line_buffer, DATE_HEADER_PREFIX).map_err(|source| {
                    TcpAttemptError::non_retryable(TimeError::parse_with_source(
                        "TCP Date 헤더 UTF-8 변환 실패",
                        source,
                    ))
                })?;
            if let Some(date_str) = maybe_date_str {
                if header_block.date_sample.is_some() {
                    return Err(TcpAttemptError::non_retryable(TimeError::parse(
                        "TCP Date 헤더가 여러 개입니다.",
                    )));
                }
                let response_received_inst = Instant::now();
                header_block.date_sample = Some(
                    parse_http_date_to_systemtime(date_str)
                        .map(|server_time| (server_time, response_received_inst)),
                );
                continue;
            }
            let maybe_age_str =
                find_header_value(self.line_buffer, AGE_HEADER_PREFIX).map_err(|source| {
                    TcpAttemptError::non_retryable(TimeError::parse_with_source(
                        "TCP Age 헤더 UTF-8 변환 실패",
                        source,
                    ))
                })?;
            if let Some(age_str) = maybe_age_str {
                header_block.age_result = Some(if header_block.age_result.is_some() {
                    Err(TimeError::parse("TCP Age 헤더가 여러 개입니다."))
                } else {
                    validate_http_age_value(age_str)
                        .map_err(|message| TimeError::parse(format!("TCP {message}")))
                });
                continue;
            }
            self.update_connection_tokens(&mut header_block);
        }
    }
    fn read_line(&mut self) -> CoreResult<usize, TcpAttemptError> {
        self.line_buffer.clear();
        loop {
            let remaining = self.remaining_timeout()?;
            self.reader
                .get_ref()
                .set_read_timeout(Some(remaining))
                .map_err(|source| TcpAttemptError::retryable(TimeError::from(source)))?;
            let available = IoBufRead::fill_buf(&mut *self.reader)
                .map_err(|source| TcpAttemptError::retryable(TimeError::from(source)))?;
            if Instant::now() >= self.deadline {
                return Err(tcp_response_timeout());
            }
            if available.is_empty() {
                break;
            }
            let (chunk_len, line_ended) =
                if let Some(newline_index) = available.iter().position(|&byte| byte == b'\n') {
                    (
                        newline_index.checked_add(1).ok_or_else(|| {
                            TcpAttemptError::non_retryable(TimeError::parse(
                                "TCP HTTP 헤더 line 길이 계산 실패",
                            ))
                        })?,
                        true,
                    )
                } else {
                    (available.len(), false)
                };
            let next_line_len = self
                .line_buffer
                .len()
                .checked_add(chunk_len)
                .ok_or_else(|| {
                    TcpAttemptError::non_retryable(TimeError::parse(
                        "TCP HTTP 헤더 line 길이 계산 실패",
                    ))
                })?;
            if next_line_len > TCP_MAX_HEADER_LINE_BYTES
                || (next_line_len == TCP_MAX_HEADER_LINE_BYTES && !line_ended)
            {
                return Err(TcpAttemptError::non_retryable(TimeError::parse(
                    "TCP HTTP 헤더 line이 너무 깁니다.",
                )));
            }
            self.line_buffer.try_reserve(chunk_len).map_err(|source| {
                TcpAttemptError::non_retryable(TimeError::parse_with_source(
                    "TCP HTTP 헤더 line 메모리 확보 실패",
                    source,
                ))
            })?;
            self.line_buffer
                .extend_from_slice(available.get(..chunk_len).ok_or_else(|| {
                    TcpAttemptError::non_retryable(TimeError::parse(
                        "TCP HTTP 헤더 line 범위 계산 실패",
                    ))
                })?);
            IoBufRead::consume(&mut *self.reader, chunk_len);
            if line_ended {
                break;
            }
        }
        let bytes_read = self.line_buffer.len();
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
        Ok(bytes_read)
    }
    fn read_response(&mut self) -> CoreResult<TcpResponseSample, TcpAttemptError> {
        let mut informational_count = 0_u8;
        loop {
            let status = self.read_status_line()?;
            let header_block = self.read_header_block()?;
            if (100..=199).contains(&status.code) {
                informational_count = informational_count.checked_add(1).ok_or_else(|| {
                    TcpAttemptError::non_retryable(TimeError::parse(
                        "TCP HTTP informational 응답이 너무 많습니다.",
                    ))
                })?;
                if informational_count > TCP_MAX_INFORMATIONAL_RESPONSES {
                    return Err(TcpAttemptError::non_retryable(TimeError::parse(
                        "TCP HTTP informational 응답이 너무 많습니다.",
                    )));
                }
                continue;
            }
            let reusable = header_block.reusable(status.version);
            if let Some(age_result) = header_block.age_result {
                age_result.map_err(TcpAttemptError::non_retryable)?;
            }
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
        let status_line = trim_http_line_end(self.line_buffer);
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
        let status_tail = status_tail_raw.trim_ascii();
        let Some((status_digits, status_rest)) = status_tail.split_first_chunk::<3>() else {
            return Err(TcpAttemptError::non_retryable(TimeError::parse(
                "TCP HTTP 상태 코드가 올바르지 않습니다.",
            )));
        };
        if status_rest
            .first()
            .is_some_and(|byte| !byte.is_ascii_whitespace())
        {
            return Err(TcpAttemptError::non_retryable(TimeError::parse(
                "TCP HTTP 상태 코드가 올바르지 않습니다.",
            )));
        }
        let [hundreds, tens, ones] = *status_digits;
        if !(hundreds.is_ascii_digit() && tens.is_ascii_digit() && ones.is_ascii_digit()) {
            return Err(TcpAttemptError::non_retryable(TimeError::parse(
                "TCP HTTP 상태 코드가 올바르지 않습니다.",
            )));
        }
        let hundreds_digit = hundreds.wrapping_sub(b'0');
        let tens_digit = tens.wrapping_sub(b'0');
        let ones_digit = ones.wrapping_sub(b'0');
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
    fn remaining_timeout(&self) -> CoreResult<Duration, TcpAttemptError> {
        let remaining = self.deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            Err(tcp_response_timeout())
        } else {
            Ok(remaining)
        }
    }
    fn update_connection_tokens(&self, header_block: &mut TcpHeaderBlock) {
        let header_line = trim_http_line_end(self.line_buffer);
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
            let token = raw_token.trim_ascii();
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
pub(super) fn find_header_value<'line, const PREFIX_LEN: usize>(
    line: &'line [u8],
    expected_prefix: &[u8; PREFIX_LEN],
) -> CoreResult<Option<&'line str>, str::Utf8Error> {
    let Some((prefix, value)) = line.split_first_chunk::<PREFIX_LEN>() else {
        return Ok(None);
    };
    if !prefix.eq_ignore_ascii_case(expected_prefix) {
        return Ok(None);
    }
    str::from_utf8(value).map(str::trim_ascii).map(Some)
}
pub(super) fn validate_http_age_value(value: &str) -> CoreResult<(), &'static str> {
    let trimmed_value = value.trim_ascii();
    if trimmed_value.is_empty() {
        return Err("Age 헤더 값이 비어 있습니다.");
    }
    let mut has_nonzero_digit = false;
    for byte in trimmed_value.bytes() {
        if !byte.is_ascii_digit() {
            return Err("Age 헤더 값이 숫자가 아닙니다.");
        }
        has_nonzero_digit |= byte != b'0';
    }
    if has_nonzero_digit {
        return Err("Age 헤더가 0보다 커 캐시된 응답입니다.");
    }
    Ok(())
}
fn missing_tcp_date() -> TimeError {
    TimeError::header_not_found("Date (TCP)")
}
fn tcp_response_timeout() -> TcpAttemptError {
    TcpAttemptError::retryable(TimeError::from(io::Error::new(
        io::ErrorKind::TimedOut,
        "TCP HTTP 응답 제한 시간 초과",
    )))
}
pub(super) fn fetch_server_time_sample(net_ctx: &mut NetworkContext) -> Result<TimeSample> {
    let parsed_address = net_ctx.host.as_ref();
    if matches!(parsed_address.scheme, UrlScheme::Https) {
        return net_ctx
            .native_http
            .fetch_head_sample(&parsed_address.secure_url, "HTTPS");
    }
    let cached_tcp_connection = &mut net_ctx.cached_tcp_connection;
    let cached_tcp_socket_addr = &mut net_ctx.cached_tcp_socket_addr;
    let tcp_line_buffer = &mut net_ctx.tcp_line_buffer;
    let tcp_request_buffer = &mut net_ctx.tcp_request_buffer;
    let connection = if let Some(connection) = cached_tcp_connection.take() {
        match sample_from_tcp_connection(
            parsed_address,
            tcp_request_buffer,
            tcp_line_buffer,
            cached_tcp_connection,
            connection,
        ) {
            Ok(sample) => return Ok(sample),
            Err(attempt_error) if attempt_error.retryable => {
                connect_tcp_connection(parsed_address, cached_tcp_socket_addr)?
            }
            Err(attempt_error) => return Err(attempt_error.error),
        }
    } else {
        connect_tcp_connection(parsed_address, cached_tcp_socket_addr)?
    };
    sample_from_tcp_connection(
        parsed_address,
        tcp_request_buffer,
        tcp_line_buffer,
        cached_tcp_connection,
        connection,
    )
    .map_err(|attempt_error| attempt_error.error)
}
fn connect_tcp_connection(
    parsed_address: &ParsedServer,
    cached_tcp_socket_addr: &mut Option<CachedTcpSocketAddr>,
) -> Result<CachedTcpConnection> {
    let stream = 'connect: {
        if let Some(cached_socket_addr) = parsed_address.literal_tcp_socket_addr {
            break 'connect TcpStream::connect_timeout(&cached_socket_addr, TCP_TIMEOUT)?;
        }
        if let Some(cached_socket_addr) = cached_tcp_socket_addr.as_ref().map(|cached| cached.addr)
        {
            if let Ok(stream) = TcpStream::connect_timeout(&cached_socket_addr, TCP_TIMEOUT) {
                break 'connect stream;
            }
            *cached_tcp_socket_addr = None;
        }
        let mut last_connect_error = None;
        let addrs = (parsed_address.host.as_str(), parsed_address.port).to_socket_addrs()?;
        for socket_addr in addrs {
            match TcpStream::connect_timeout(&socket_addr, TCP_TIMEOUT) {
                Ok(stream) => {
                    *cached_tcp_socket_addr = Some(CachedTcpSocketAddr { addr: socket_addr });
                    break 'connect stream;
                }
                Err(source) => {
                    last_connect_error = Some(source);
                }
            }
        }
        if let Some(connect_error) = last_connect_error {
            return Err(TimeError::from(connect_error));
        }
        return Err(TimeError::from(io::Error::new(
            io::ErrorKind::NotFound,
            "Host not found",
        )));
    };
    stream.set_read_timeout(Some(TCP_TIMEOUT))?;
    stream.set_write_timeout(Some(TCP_TIMEOUT))?;
    stream.set_nodelay(true)?;
    Ok(CachedTcpConnection {
        reader: BufReader::new(stream),
    })
}
fn sample_from_tcp_connection(
    parsed_address: &ParsedServer,
    tcp_request_buffer: &mut Vec<u8>,
    tcp_line_buffer: &mut Vec<u8>,
    reusable_connection: &mut Option<CachedTcpConnection>,
    mut connection: CachedTcpConnection,
) -> CoreResult<TimeSample, TcpAttemptError> {
    tcp_request_buffer.clear();
    let host_header = parsed_address.tcp_host_header.as_str();
    let request_len = TCP_HEAD_REQUEST_PREFIX
        .len()
        .checked_add(host_header.len())
        .and_then(|value| value.checked_add(TCP_HEAD_REQUEST_SUFFIX.len()))
        .ok_or_else(|| {
            TcpAttemptError::non_retryable(TimeError::parse("TCP HTTP 요청 길이 계산 실패"))
        })?;
    if tcp_request_buffer.capacity() < request_len {
        tcp_request_buffer
            .try_reserve_exact(request_len)
            .map_err(|source| {
                TcpAttemptError::non_retryable(TimeError::parse_with_source(
                    "TCP HTTP 요청 메모리 확보 실패",
                    source,
                ))
            })?;
    }
    tcp_request_buffer.extend_from_slice(TCP_HEAD_REQUEST_PREFIX);
    tcp_request_buffer.extend_from_slice(host_header.as_bytes());
    tcp_request_buffer.extend_from_slice(TCP_HEAD_REQUEST_SUFFIX);
    let request_start_inst = Instant::now();
    let response_deadline = request_start_inst.checked_add(TCP_TIMEOUT).ok_or_else(|| {
        TcpAttemptError::non_retryable(TimeError::parse("TCP HTTP 응답 기한 계산 실패"))
    })?;
    IoWrite::write_all(connection.reader.get_mut(), tcp_request_buffer)
        .map_err(|source| TcpAttemptError::retryable(TimeError::from(source)))?;
    let response = TcpResponseReader {
        deadline: response_deadline,
        header_bytes_seen: 0,
        line_buffer: tcp_line_buffer,
        reader: &mut connection.reader,
    }
    .read_response()?;
    let rtt_for_sample = response
        .response_received_inst
        .saturating_duration_since(request_start_inst);
    if response.reusable {
        *reusable_connection = Some(connection);
    }
    Ok(TimeSample {
        response_received_inst: response.response_received_inst,
        rtt: rtt_for_sample,
        server_time: response.server_time,
    })
}
fn trim_http_line_end(line: &[u8]) -> &[u8] {
    let without_lf = line.strip_suffix(b"\n").unwrap_or(line);
    without_lf.strip_suffix(b"\r").unwrap_or(without_lf)
}
