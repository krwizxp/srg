use super::{
    CachedTcpSocketAddr, NetworkContext, Result, TCP_TIMEOUT, TimeError, TimeSample,
    address::{ParsedServer, UrlScheme},
    http_date::parse_http_date_to_systemtime,
    native_http,
};
use alloc::str;
use std::{
    io::{self, BufRead as IoBufRead, BufReader, Read as IoRead, Write as IoWrite},
    net::{self, TcpStream},
    time::Instant,
};
const TCP_HEAD_REQUEST_PREFIX: &[u8] = b"HEAD / HTTP/1.1\r\nHost: ";
const TCP_HEAD_REQUEST_SUFFIX: &[u8] =
    b"\r\nConnection: close\r\nUser-Agent: Rust-Time-Sync\r\n\r\n";
const TCP_MAX_HEADER_BYTES: usize = 64 * 1024;
const TCP_MAX_HEADER_LINE_BYTES: usize = 8192;
const TCP_MAX_HEADER_LINE_READ_BYTES: u64 = 8193;
pub(super) const TCP_LINE_BUFFER_CAPACITY: usize = 256;
pub(super) fn find_date_header_value(line: &[u8]) -> Option<&str> {
    let (prefix, value) = line.split_at_checked(5)?;
    if prefix.eq_ignore_ascii_case(b"date:") {
        str::from_utf8(value).map(str::trim_ascii).ok()
    } else {
        None
    }
}
fn missing_tcp_date() -> TimeError {
    TimeError::header_not_found("Date (TCP)")
}
pub(super) fn fetch_server_time_sample(
    parsed_address: &ParsedServer,
    net_ctx: &mut NetworkContext,
) -> Result<TimeSample> {
    if matches!(parsed_address.scheme(), Some(UrlScheme::Https)) {
        return native_http::NATIVE_HTTP
            .fetch_head_sample(parsed_address.url(UrlScheme::Https), "HTTPS (explicit)");
    }
    let request_start_inst = Instant::now();
    let mut stream = 'connect: {
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
        let socket_addrs =
            net::ToSocketAddrs::to_socket_addrs(&(parsed_address.host(), parsed_address.port()))?;
        for socket_addr in socket_addrs {
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
    IoWrite::write_all(&mut stream, TCP_HEAD_REQUEST_PREFIX)?;
    let host_header = parsed_address.tcp_host_header_value();
    IoWrite::write_all(&mut stream, host_header.as_bytes())?;
    IoWrite::write_all(&mut stream, TCP_HEAD_REQUEST_SUFFIX)?;
    let mut stream_reader = BufReader::new(&stream);
    let mut header_bytes_seen = 0_usize;
    loop {
        net_ctx.tcp_line_buffer.clear();
        let bytes_read = IoBufRead::read_until(
            &mut IoRead::take(&mut stream_reader, TCP_MAX_HEADER_LINE_READ_BYTES),
            b'\n',
            &mut net_ctx.tcp_line_buffer,
        )?;
        if bytes_read == 0 {
            break Err(missing_tcp_date());
        }
        header_bytes_seen = header_bytes_seen
            .checked_add(bytes_read)
            .ok_or_else(|| TimeError::parse("TCP HTTP 헤더 길이 계산 실패"))?;
        if header_bytes_seen > TCP_MAX_HEADER_BYTES {
            return Err(TimeError::parse("TCP HTTP 헤더가 너무 큽니다."));
        }
        let line_ended = net_ctx
            .tcp_line_buffer
            .last()
            .is_some_and(|byte| *byte == b'\n');
        if net_ctx.tcp_line_buffer.len() > TCP_MAX_HEADER_LINE_BYTES
            || (!line_ended && net_ctx.tcp_line_buffer.len() == TCP_MAX_HEADER_LINE_BYTES)
        {
            return Err(TimeError::parse("TCP HTTP 헤더 line이 너무 깁니다."));
        }
        if let Some(date_str) = find_date_header_value(&net_ctx.tcp_line_buffer) {
            let response_received_inst = Instant::now();
            let rtt_for_sample = response_received_inst.duration_since(request_start_inst);
            let server_time = parse_http_date_to_systemtime(date_str)?;
            break Ok(TimeSample {
                response_received_inst,
                rtt: rtt_for_sample,
                server_time,
            });
        }
        if net_ctx.tcp_line_buffer == b"\r\n" {
            break Err(missing_tcp_date());
        }
    }
}
