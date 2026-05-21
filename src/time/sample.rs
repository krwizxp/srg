use super::{
    Result, TCP_TIMEOUT, TimeError, TimeSample,
    address::{ParsedServer, UrlScheme},
    http_date::parse_http_date_to_systemtime,
    native_http,
};
use alloc::str;
use std::{
    io::{self, BufRead as IoBufRead, BufReader, Write as IoWrite},
    net::{self, TcpStream},
    time::Instant,
};
const TCP_HEAD_REQUEST_PREFIX: &[u8] = b"HEAD / HTTP/1.1\r\nHost: ";
const TCP_HEAD_REQUEST_SUFFIX: &[u8] =
    b"\r\nConnection: close\r\nUser-Agent: Rust-Time-Sync\r\n\r\n";
const TCP_LINE_BUF_CAPACITY: usize = 256;
pub(super) struct NetworkContext {
    cached_tcp_socket_addr: Option<net::SocketAddr>,
    tcp_line_buffer: Vec<u8>,
}
pub(super) struct NetworkContextRequest;
impl TryFrom<NetworkContextRequest> for NetworkContext {
    type Error = TimeError;
    fn try_from(_value: NetworkContextRequest) -> Result<Self> {
        let mut tcp_line_buffer = Vec::new();
        tcp_line_buffer
            .try_reserve(TCP_LINE_BUF_CAPACITY)
            .map_err(|source| {
                TimeError::parse(format!("TCP line buffer 메모리 확보 실패: {source}"))
            })?;
        Ok(Self {
            cached_tcp_socket_addr: None,
            tcp_line_buffer,
        })
    }
}
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
    if parsed_address
        .scheme()
        .is_some_and(|scheme| scheme == UrlScheme::Https)
    {
        let https_url = parsed_address.url(UrlScheme::Https);
        return native_http::NATIVE_HTTP.fetch_head_sample(https_url, "HTTPS (explicit)");
    }
    let request_start_inst = Instant::now();
    let literal_socket_addr = parsed_address.literal_tcp_socket_addr();
    let had_cached_socket = net_ctx.cached_tcp_socket_addr.is_some();
    let socket_addr = resolve_tcp_socket_addr(parsed_address, net_ctx)?;
    let stream_result: Result<TcpStream> =
        match TcpStream::connect_timeout(&socket_addr, TCP_TIMEOUT) {
            Ok(stream) => Ok(stream),
            Err(connect_err) if had_cached_socket && literal_socket_addr.is_none() => {
                net_ctx.cached_tcp_socket_addr = None;
                let refreshed_socket_addr = resolve_tcp_socket_addr(parsed_address, net_ctx)?;
                if refreshed_socket_addr == socket_addr {
                    Err(TimeError::from(connect_err))
                } else {
                    TcpStream::connect_timeout(&refreshed_socket_addr, TCP_TIMEOUT)
                        .map_err(TimeError::from)
                }
            }
            Err(connect_err) => {
                if literal_socket_addr.is_none() {
                    net_ctx.cached_tcp_socket_addr = None;
                }
                Err(TimeError::from(connect_err))
            }
        };
    let mut stream = stream_result?;
    stream.set_read_timeout(Some(TCP_TIMEOUT))?;
    stream.set_write_timeout(Some(TCP_TIMEOUT))?;
    IoWrite::write_all(&mut stream, TCP_HEAD_REQUEST_PREFIX)?;
    let host_header = parsed_address.tcp_host_header_value();
    IoWrite::write_all(&mut stream, host_header.as_bytes())?;
    IoWrite::write_all(&mut stream, TCP_HEAD_REQUEST_SUFFIX)?;
    let mut stream_reader = BufReader::new(&stream);
    loop {
        net_ctx.tcp_line_buffer.clear();
        let bytes_read =
            IoBufRead::read_until(&mut stream_reader, b'\n', &mut net_ctx.tcp_line_buffer)?;
        if bytes_read == 0 {
            break Err(missing_tcp_date());
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
fn resolve_tcp_socket_addr(
    parsed_address: &ParsedServer,
    net_ctx: &mut NetworkContext,
) -> Result<net::SocketAddr> {
    if let Some(cached_socket_addr) = parsed_address
        .literal_tcp_socket_addr()
        .or(net_ctx.cached_tcp_socket_addr)
    {
        return Ok(cached_socket_addr);
    }
    let socket_addr =
        net::ToSocketAddrs::to_socket_addrs(&(parsed_address.host(), parsed_address.port()))?
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Host not found"))?;
    net_ctx.cached_tcp_socket_addr = Some(socket_addr);
    Ok(socket_addr)
}
