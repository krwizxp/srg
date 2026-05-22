use super::{
    HeadResponse, MIN_TRANSFER_TIME, ParseHttpDate, Result, TCP_TIMEOUT, TimeError, error,
    find_date_header_value,
};
use alloc::{string::String, vec::Vec};
use core::{
    cell::RefCell,
    ffi::c_void,
    ptr::{NonNull, null, null_mut},
};
use std::{
    ffi::OsStr,
    os::windows::ffi::OsStrExt as WindowsOsStrExt,
    time::{Instant, SystemTime},
};
const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
const HTTP_SCHEME_PREFIX: &str = "http://";
const HTTPS_SCHEME_PREFIX: &str = "https://";
const INTERNET_DEFAULT_HTTP_PORT: u16 = 80;
const INTERNET_DEFAULT_HTTPS_PORT: u16 = 443;
const WINHTTP_ACCESS_TYPE_DEFAULT_PROXY: u32 = 0;
const WINHTTP_FLAG_SECURE: u32 = 0x0080_0000;
const WINHTTP_OPTION_IGNORE_CERT_REVOCATION_OFFLINE: u32 = 155;
const WINHTTP_QUERY_RAW_HEADERS_CRLF: u32 = 22;
const WINHTTP_CONNECT_CACHE_LIMIT: usize = 4;
pub(super) const CLIENT: Client = Client {
    get_last_error: GetLastError,
    http_scheme_prefix: HTTP_SCHEME_PREFIX,
    https_scheme_prefix: HTTPS_SCHEME_PREFIX,
};
type HInternet = *mut c_void;
pub(super) struct Client {
    get_last_error: unsafe extern "system" fn() -> u32,
    http_scheme_prefix: &'static str,
    https_scheme_prefix: &'static str,
}
struct RequestTarget {
    host: String,
    port: u16,
    secure: bool,
}
struct Handle(NonNull<c_void>);
struct CachedConnect {
    handle: Handle,
    host: String,
    port: u16,
}
struct SessionCache {
    connects: Vec<CachedConnect>,
    session: Handle,
}
std::thread_local! {
    static SESSION_CACHE: RefCell<Option<SessionCache>> = const { RefCell::new(None) };
}
#[link(name = "winhttp")]
unsafe extern "system" {
    fn WinHttpCloseHandle(h_internet: HInternet) -> i32;
    fn WinHttpConnect(
        h_session: HInternet,
        server_name: *const u16,
        server_port: u16,
        reserved: u32,
    ) -> HInternet;
    fn WinHttpOpen(
        user_agent: *const u16,
        access_type: u32,
        proxy_name: *const u16,
        proxy_bypass: *const u16,
        flags: u32,
    ) -> HInternet;
    fn WinHttpOpenRequest(
        h_connect: HInternet,
        verb: *const u16,
        object_name: *const u16,
        version: *const u16,
        referrer: *const u16,
        accept_types: *const *const u16,
        flags: u32,
    ) -> HInternet;
    fn WinHttpQueryHeaders(
        h_request: HInternet,
        info_level: u32,
        name: *const u16,
        buffer: *mut c_void,
        buffer_length: *mut u32,
        index: *mut u32,
    ) -> i32;
    fn WinHttpReceiveResponse(h_request: HInternet, reserved: *mut c_void) -> i32;
    fn WinHttpSendRequest(
        h_request: HInternet,
        headers: *const u16,
        headers_length: u32,
        optional: *const c_void,
        optional_length: u32,
        total_length: u32,
        context: usize,
    ) -> i32;
    fn WinHttpSetOption(
        h_internet: HInternet,
        option: u32,
        buffer: *mut c_void,
        buffer_length: u32,
    ) -> i32;
    fn WinHttpSetTimeouts(
        h_internet: HInternet,
        resolve_timeout: i32,
        connect_timeout: i32,
        send_timeout: i32,
        receive_timeout: i32,
    ) -> i32;
}
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetLastError() -> u32;
}
impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: self.0 is a WinHTTP handle returned by WinHTTP and is closed exactly once here.
        unsafe {
            WinHttpCloseHandle(self.as_ptr());
        }
    }
}
impl Handle {
    const fn as_ptr(&self) -> HInternet {
        self.0.as_ptr()
    }
}
impl Client {
    fn cached_connect<'cache>(
        &self,
        cache: &'cache mut SessionCache,
        host: &str,
        host_wide: &[u16],
        port: u16,
        context: &str,
    ) -> Result<&'cache Handle> {
        if let Some(index) = cache
            .connects
            .iter()
            .position(|entry| entry.port == port && entry.host.as_str() == host)
        {
            return cache
                .connects
                .get(index)
                .map(|entry| &entry.handle)
                .ok_or_else(|| error(context, "WinHTTP connect cache 범위 오류"));
        }
        let handle = self.connect(&cache.session, host_wide, port, context)?;
        let mut host_key = String::new();
        host_key
            .try_reserve(host.len())
            .map_err(|source| error(context, format!("WinHTTP connect host key 메모리 확보 실패: {source}")))?;
        host_key.push_str(host);
        if cache.connects.len() >= WINHTTP_CONNECT_CACHE_LIMIT {
            cache.connects.remove(0);
        }
        cache
            .connects
            .try_reserve(1)
            .map_err(|source| error(context, format!("WinHTTP connect cache 메모리 확보 실패: {source}")))?;
        let entry = cache.connects.push_mut(CachedConnect {
            handle,
            host: host_key,
            port,
        });
        Ok(&entry.handle)
    }
    fn connect(&self, session: &Handle, host: &[u16], port: u16, context: &str) -> Result<Handle> {
        // SAFETY: host is NUL-terminated and session is a valid session handle.
        let raw_connect = unsafe { WinHttpConnect(session.as_ptr(), host.as_ptr(), port, 0) };
        self.non_null_handle(raw_connect, "WinHttpConnect", context)
    }
    pub(super) fn fetch_head(
        &self,
        url: &str,
        context: &str,
        parse_http_date: ParseHttpDate,
    ) -> Result<HeadResponse> {
        let target = self.request_target(url, context)?;
        let user_agent = wide(concat!("srg/", env!("CARGO_PKG_VERSION")), context)?;
        let host_wide = wide(&target.host, context)?;
        let method_wide = wide("HEAD", context)?;
        let path_wide = wide("/", context)?;
        let flags = if target.secure {
            WINHTTP_FLAG_SECURE
        } else {
            0
        };
        let (request, request_start, response_received_inst) = self.with_cached_connect(
            &user_agent,
            &target.host,
            &host_wide,
            target.port,
            context,
            |connect| {
                let request = self.open_request(connect, &method_wide, &path_wide, flags, context)?;
                if target.secure {
                    match self.set_ignore_revocation_offline(&request) {
                        Ok(()) | Err(_) => {}
                    }
                }
                let request_start = Instant::now();
                self.send_head(&request, context)?;
                self.receive_response(&request, context)?;
                let response_received_inst = Instant::now();
                Ok((request, request_start, response_received_inst))
            },
        )?;
        let server_time = self.query_server_time(&request, context, parse_http_date)?;
        let rtt = response_received_inst
            .saturating_duration_since(request_start)
            .max(MIN_TRANSFER_TIME);
        Ok(HeadResponse {
            response_received_inst,
            rtt,
            server_time,
        })
    }
    fn last_error(&self, operation: &str, context: &str) -> super::TimeError {
        let code = self.last_error_code();
        error(context, format!("{operation} 실패: Windows error {code}"))
    }
    fn last_error_code(&self) -> u32 {
        // SAFETY: GetLastError has no preconditions.
        unsafe { (self.get_last_error)() }
    }
    fn non_null_handle(&self, handle: HInternet, operation: &str, context: &str) -> Result<Handle> {
        NonNull::new(handle)
            .map(Handle)
            .ok_or_else(|| self.last_error(operation, context))
    }
    fn open_request(
        &self,
        connect: HInternet,
        method: &[u16],
        path: &[u16],
        flags: u32,
        context: &str,
    ) -> Result<Handle> {
        // SAFETY: method and path are NUL-terminated and connect is valid.
        let raw_request = unsafe {
            WinHttpOpenRequest(
                connect,
                method.as_ptr(),
                path.as_ptr(),
                null(),
                null(),
                null(),
                flags,
            )
        };
        self.non_null_handle(raw_request, "WinHttpOpenRequest", context)
    }
    fn open_session(&self, user_agent: &[u16], context: &str) -> Result<Handle> {
        // SAFETY: user_agent is NUL-terminated and optional proxy pointers are intentionally null.
        let raw_session = unsafe {
            WinHttpOpen(
                user_agent.as_ptr(),
                WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                null(),
                null(),
                0,
            )
        };
        let session = self.non_null_handle(raw_session, "WinHttpOpen", context)?;
        let Ok(timeout) = i32::try_from(TCP_TIMEOUT.as_millis()) else {
            return Err(error(context, "WinHTTP timeout 변환 실패"));
        };
        // SAFETY: session is a valid WinHTTP session handle.
        unsafe {
            WinHttpSetTimeouts(session.as_ptr(), timeout, timeout, timeout, timeout);
        }
        Ok(session)
    }
    fn query_server_time(
        &self,
        request: &Handle,
        context: &str,
        parse_http_date: ParseHttpDate,
    ) -> Result<SystemTime> {
        let mut bytes = 0_u32;
        let mut index = 0_u32;
        // SAFETY: request is valid; this first call probes the required buffer size.
        let probe_ok = unsafe {
            WinHttpQueryHeaders(
                request.as_ptr(),
                WINHTTP_QUERY_RAW_HEADERS_CRLF,
                null(),
                null_mut(),
                &raw mut bytes,
                &raw mut index,
            )
        };
        if probe_ok != 0_i32 {
            return Err(TimeError::header_not_found(format!(
                "{context} 응답에서 Date"
            )));
        }
        let last_error_code = self.last_error_code();
        if last_error_code != ERROR_INSUFFICIENT_BUFFER {
            return Err(self.last_error("WinHttpQueryHeaders", context));
        }
        let units = usize::try_from(bytes)
            .map_err(|source| error(context, format!("응답 헤더 길이 변환 실패: {source}")))?
            .checked_div(2)
            .ok_or_else(|| error(context, "응답 헤더 길이 계산 실패"))?;
        let mut buffer = Vec::new();
        buffer.try_reserve(units).map_err(|source| {
            error(
                context,
                format!("응답 헤더 버퍼 메모리 확보 실패: {source}"),
            )
        })?;
        buffer.resize(units, 0_u16);
        index = 0;
        // SAFETY: buffer has the size requested by WinHTTP and request is valid.
        let fetch_ok = unsafe {
            WinHttpQueryHeaders(
                request.as_ptr(),
                WINHTTP_QUERY_RAW_HEADERS_CRLF,
                null(),
                buffer.as_mut_ptr().cast::<c_void>(),
                &raw mut bytes,
                &raw mut index,
            )
        };
        if fetch_ok == 0_i32 {
            return Err(self.last_error("WinHttpQueryHeaders", context));
        }
        while buffer.pop_if(|value| *value == 0).is_some() {}
        for raw_line in buffer.split(|unit| *unit == u16::from(b'\n')).rev() {
            let line = raw_line
                .strip_suffix(&[u16::from(b'\r')])
                .unwrap_or(raw_line);
            let Some((prefix, _value)) = line.split_at_checked(5) else {
                continue;
            };
            if !prefix.iter().zip(b"date:").all(|(&unit, &byte)| {
                u8::try_from(unit).is_ok_and(|converted| converted.eq_ignore_ascii_case(&byte))
            }) {
                continue;
            }
            let mut line_bytes = Vec::new();
            line_bytes.try_reserve(line.len()).map_err(|source| {
                error(
                    context,
                    format!("Date 헤더 ASCII 버퍼 메모리 확보 실패: {source}"),
                )
            })?;
            for &unit in line {
                let byte = u8::try_from(unit)
                    .map_err(|source| error(context, format!("Date 헤더 ASCII 변환 실패: {source}")))?;
                line_bytes.push(byte);
            }
            if let Some(date_header_raw) = find_date_header_value(&line_bytes) {
                return parse_http_date(date_header_raw);
            }
        }
        Err(TimeError::header_not_found(format!(
            "{context} 응답에서 Date"
        )))
    }
    fn receive_response(&self, request: &Handle, context: &str) -> Result<()> {
        // SAFETY: request is a valid request handle and no reserved pointer is required.
        let received = unsafe { WinHttpReceiveResponse(request.as_ptr(), null_mut()) };
        if received == 0_i32 {
            Err(self.last_error("WinHttpReceiveResponse", context))
        } else {
            Ok(())
        }
    }
    fn request_target(&self, url: &str, context: &str) -> Result<RequestTarget> {
        let (secure, rest) = if let Some(rest) = url.strip_prefix(self.https_scheme_prefix) {
            (true, rest)
        } else if let Some(rest) = url.strip_prefix(self.http_scheme_prefix) {
            (false, rest)
        } else {
            return Err(error(context, "지원하지 않는 URL scheme입니다."));
        };
        let authority = match rest.split_once(['/', '?', '#']) {
            Some((authority, _)) => authority,
            None => rest,
        };
        if authority.is_empty() {
            return Err(error(context, "URL host가 비어 있습니다."));
        }
        let default_port = if secure {
            INTERNET_DEFAULT_HTTPS_PORT
        } else {
            INTERNET_DEFAULT_HTTP_PORT
        };
        let parse_port = |port_text: &str| {
            let port = port_text
                .parse::<u16>()
                .map_err(|source| error(context, format!("URL port 파싱 실패: {source}")))?;
            if port == 0 {
                Err(error(context, "URL port는 1 이상이어야 합니다."))
            } else {
                Ok(port)
            }
        };
        let copy_host = |host_text: &str| -> Result<String> {
            let mut host = String::new();
            host.try_reserve(host_text.len())
                .map_err(|source| error(context, format!("URL host 메모리 확보 실패: {source}")))?;
            host.push_str(host_text);
            Ok(host)
        };
        let (host, port) = if let Some(bracketed) = authority.strip_prefix('[') {
            let Some((host, bracket_rest)) = bracketed.split_once(']') else {
                return Err(error(context, "IPv6 URL host 형식이 올바르지 않습니다."));
            };
            let port = if bracket_rest.is_empty() {
                default_port
            } else {
                parse_port(
                    bracket_rest
                        .strip_prefix(':')
                    .ok_or_else(|| error(context, "URL port 형식이 올바르지 않습니다."))?,
                )?
            };
            (copy_host(host)?, port)
        } else if let Some((host, port_text)) = authority.rsplit_once(':') {
            if host.contains(':') {
                (copy_host(authority)?, default_port)
            } else {
                (copy_host(host)?, parse_port(port_text)?)
            }
        } else {
            (copy_host(authority)?, default_port)
        };
        Ok(RequestTarget { host, port, secure })
    }
    fn send_head(&self, request: &Handle, context: &str) -> Result<()> {
        // SAFETY: request is valid and no additional request body or headers are needed for HEAD.
        let sent = unsafe { WinHttpSendRequest(request.as_ptr(), null(), 0, null(), 0, 0, 0) };
        if sent == 0_i32 {
            Err(self.last_error("WinHttpSendRequest", context))
        } else {
            Ok(())
        }
    }
    fn set_ignore_revocation_offline(&self, request: &Handle) -> Result<()> {
        let mut enabled = 1_u32;
        let buffer_length = u32::try_from(size_of::<u32>())
            .map_err(|source| error("WinHTTP", format!("옵션 길이 변환 실패: {source}")))?;
        // SAFETY: request is a valid WinHTTP request handle and enabled points to a u32 option value.
        let ok = unsafe {
            WinHttpSetOption(
                request.as_ptr(),
                WINHTTP_OPTION_IGNORE_CERT_REVOCATION_OFFLINE,
                (&raw mut enabled).cast::<c_void>(),
                buffer_length,
            )
        };
        if ok == 0_i32 {
            Err(self.last_error("WinHttpSetOption IGNORE_CERT_REVOCATION_OFFLINE", "WinHTTP"))
        } else {
            Ok(())
        }
    }
    fn with_cached_connect<R>(
        &self,
        user_agent: &[u16],
        host: &str,
        host_wide: &[u16],
        port: u16,
        context: &str,
        action: impl FnOnce(HInternet) -> Result<R>,
    ) -> Result<R> {
        let connect_ptr = SESSION_CACHE
            .try_with(|cell| {
                let mut session_cache = cell.try_borrow_mut().map_err(|source| {
                    error(
                        context,
                        format!("WinHTTP session cache borrow 실패: {source}"),
                    )
                })?;
                if session_cache.is_none() {
                    *session_cache = Some(SessionCache {
                        connects: Vec::new(),
                        session: self.open_session(user_agent, context)?,
                    });
                }
                let Some(cache) = session_cache.as_mut() else {
                    return Err(error(context, "WinHTTP session cache가 비어 있습니다."));
                };
                let connect = self.cached_connect(cache, host, host_wide, port, context)?;
                Ok(connect.as_ptr())
            })
            .map_err(|source| error(context, format!("WinHTTP session cache 접근 실패: {source}")))??;
        match action(connect_ptr) {
            Ok(value) => Ok(value),
            Err(action_error) => match SESSION_CACHE.try_with(|cell| {
                let mut session_cache = cell.try_borrow_mut().map_err(|source| {
                    error(
                        context,
                        format!("WinHTTP session cache borrow 실패: {source}"),
                    )
                })?;
                *session_cache = None;
                Ok::<(), TimeError>(())
            }) {
                Ok(Ok(())) => Err(action_error),
                Ok(Err(cache_error)) => Err(error(
                    context,
                    format!("{action_error}; WinHTTP session cache 정리 실패: {cache_error}"),
                )),
                Err(access_error) => Err(error(
                    context,
                    format!("{action_error}; WinHTTP session cache 접근 실패: {access_error}"),
                )),
            },
        }
    }
}
fn wide(value: &str, context: &str) -> Result<Vec<u16>> {
    let capacity = value
        .len()
        .checked_add(1)
        .ok_or_else(|| error(context, "wide 문자열 용량 계산 실패"))?;
    let mut out = Vec::new();
    out.try_reserve(capacity)
        .map_err(|source| error(context, format!("wide 문자열 메모리 확보 실패: {source}")))?;
    out.extend(<OsStr as WindowsOsStrExt>::encode_wide(OsStr::new(value)));
    out.push(0);
    Ok(out)
}
