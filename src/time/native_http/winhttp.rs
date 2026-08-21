use super::super::{ParsedServer, UrlScheme};
use super::{
    FreshTimeHeaders, MIN_TRANSFER_TIME, Result, TimeError, TimeSample, error, error_with_source,
};
use alloc::vec::Vec;
use core::{
    ffi::c_void,
    ptr::{NonNull, null, null_mut},
    result::Result as CoreResult,
    str,
};
use std::{
    process,
    time::{Instant, SystemTime},
};
mod sys;
const DWORD_BYTE_SIZE: u32 = 4;
const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
const ERROR_WINHTTP_HEADER_NOT_FOUND: u32 = 12_150;
const WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY: u32 = 4;
const WINHTTP_ACCESS_TYPE_NO_PROXY: u32 = 1;
const WINHTTP_FLAG_SECURE: u32 = 0x0080_0000;
const WINHTTP_OPTION_DISABLE_FEATURE: u32 = 63;
const WINHTTP_OPTION_SECURE_PROTOCOLS: u32 = 84;
const WINHTTP_OPTION_MAX_RESPONSE_HEADER_SIZE: u32 = 91;
const WINHTTP_OPTION_DISABLE_SECURE_PROTOCOL_FALLBACK: u32 = 144;
const WINHTTP_OPTION_IPV6_FAST_FALLBACK: u32 = 140;
const WINHTTP_OPTION_DISABLE_GLOBAL_POOLING: u32 = 195;
const WINHTTP_SESSION_OPTIONS: [(u32, &str, Option<u32>); 3] = [
    (
        WINHTTP_OPTION_DISABLE_SECURE_PROTOCOL_FALLBACK,
        "WinHttpSetOption DISABLE_SECURE_PROTOCOL_FALLBACK",
        None,
    ),
    (
        WINHTTP_OPTION_DISABLE_GLOBAL_POOLING,
        "WinHttpSetOption DISABLE_GLOBAL_POOLING",
        Some(ERROR_WINHTTP_INVALID_OPTION),
    ),
    (
        WINHTTP_OPTION_IPV6_FAST_FALLBACK,
        "WinHttpSetOption IPV6_FAST_FALLBACK",
        None,
    ),
];
const WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2: u32 = 0x0000_0800;
const WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_3: u32 = 0x0000_2000;
const WINHTTP_SECURE_PROTOCOLS_MIN_TLS_1_2: u32 =
    WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2 | WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_3;
const WINHTTP_DISABLE_AUTHENTICATION: u32 = 0x0000_0004;
const WINHTTP_DISABLE_COOKIES: u32 = 0x0000_0001;
const WINHTTP_DISABLE_REDIRECTS: u32 = 0x0000_0002;
const ERROR_INVALID_PARAMETER: u32 = 87;
const ERROR_WINHTTP_INVALID_OPTION: u32 = 12_009;
const WINHTTP_QUERY_AGE: u32 = 48;
const WINHTTP_QUERY_DATE: u32 = 9;
const HTTP_HEAD_MAX_HEADER_BYTES: usize = 64 * 1024;
const HTTP_HEAD_MAX_HEADER_BYTES_DWORD: u32 = 64 * 1024;
const METHOD_HEAD_WIDE: [u16; 5] = [0x48, 0x45, 0x41, 0x44, 0];
const PATH_ROOT_WIDE: [u16; 2] = [0x2F, 0];
const WINHTTP_TIMEOUT_MILLIS: i32 = 5_000;
enum WinHttpHandle {}
type HInternet = *mut WinHttpHandle;
#[derive(Default)]
pub(in crate::time) struct Client {
    header_buffer: Vec<u16>,
    header_line_buffer: Vec<u8>,
    session_cache: Option<SessionCache>,
}
struct Handle(NonNull<WinHttpHandle>);
struct SessionCache {
    connect: Option<Handle>,
    session: Handle,
}
impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: self.0 is a WinHTTP handle returned by WinHTTP and is closed exactly once here.
        unsafe {
            sys::WinHttpCloseHandle(self.0.as_ptr());
        }
    }
}
impl Client {
    fn cached_connect(
        &mut self,
        server: &ParsedServer,
        context: &str,
    ) -> Result<NonNull<WinHttpHandle>> {
        let cache = if let Some(ref mut cache) = self.session_cache {
            cache
        } else {
            let user_agent_text = match server.scheme {
                UrlScheme::Http => "Rust-Time-Sync",
                UrlScheme::Https => concat!("srg/", env!("CARGO_PKG_VERSION")),
            };
            let capacity = user_agent_text.len().strict_add(1);
            let mut user_agent = Vec::new();
            user_agent.try_reserve_exact(capacity).map_err(|source| {
                error_with_source(context, "wide 문자열 메모리 확보 실패", source)
            })?;
            user_agent.extend(user_agent_text.encode_utf16());
            user_agent.push(0);
            // SAFETY: user_agent is NUL-terminated and optional proxy pointers are intentionally null.
            let raw_session = unsafe {
                sys::WinHttpOpen(
                    user_agent.as_ptr(),
                    match server.scheme {
                        UrlScheme::Http => WINHTTP_ACCESS_TYPE_NO_PROXY,
                        UrlScheme::Https => WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                    },
                    null(),
                    null(),
                    0,
                )
            };
            let session = Self::non_null_handle(raw_session, "WinHttpOpen", context)?;
            // SAFETY: session is a valid WinHTTP session handle.
            let timeout_ok = unsafe {
                sys::WinHttpSetTimeouts(
                    session.0.as_ptr(),
                    WINHTTP_TIMEOUT_MILLIS,
                    WINHTTP_TIMEOUT_MILLIS,
                    WINHTTP_TIMEOUT_MILLIS,
                    WINHTTP_TIMEOUT_MILLIS,
                )
            };
            if timeout_ok == 0_i32 {
                return Err(Self::last_error("WinHttpSetTimeouts", context));
            }
            if let Err(code) = Self::try_set_dword_option(
                &session,
                WINHTTP_OPTION_SECURE_PROTOCOLS,
                WINHTTP_SECURE_PROTOCOLS_MIN_TLS_1_2,
            ) {
                if matches!(code, ERROR_INVALID_PARAMETER | ERROR_WINHTTP_INVALID_OPTION) {
                    Self::set_dword_option(
                        &session,
                        WINHTTP_OPTION_SECURE_PROTOCOLS,
                        WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2,
                        "WinHttpSetOption SECURE_PROTOCOLS",
                        context,
                    )?;
                } else {
                    return Err(Self::windows_error(
                        "WinHttpSetOption SECURE_PROTOCOLS",
                        code,
                        context,
                    ));
                }
            }
            for (option, operation, ignored_error) in WINHTTP_SESSION_OPTIONS {
                if let Err(code) = Self::try_set_dword_option(&session, option, 1)
                    && ignored_error != Some(code)
                {
                    return Err(Self::windows_error(operation, code, context));
                }
            }
            self.session_cache.insert(SessionCache {
                connect: None,
                session,
            })
        };
        if let Some(connect) = cache.connect.as_ref() {
            return Ok(connect.0);
        }
        // SAFETY: host_wide is NUL-terminated and cache.session is a valid session handle.
        let raw_connect = unsafe {
            sys::WinHttpConnect(
                cache.session.0.as_ptr(),
                server.host_wide.as_ptr(),
                server.port,
                0,
            )
        };
        let handle = NonNull::new(raw_connect)
            .map(Handle)
            .ok_or_else(|| Self::last_error("WinHttpConnect", context))?;
        let connect = handle.0;
        cache.connect = Some(handle);
        Ok(connect)
    }
    pub(in crate::time) fn fetch_head(
        &mut self,
        server: &ParsedServer,
        context: &str,
    ) -> Result<TimeSample> {
        let connect = self.cached_connect(server, context)?;
        let (request, request_start, response_received) = (|| -> Result<_> {
            // SAFETY: method and path are NUL-terminated and connect is valid.
            let raw_request = unsafe {
                sys::WinHttpOpenRequest(
                    connect.as_ptr(),
                    METHOD_HEAD_WIDE.as_ptr(),
                    PATH_ROOT_WIDE.as_ptr(),
                    null(),
                    null(),
                    null(),
                    match server.scheme {
                        UrlScheme::Http => 0,
                        UrlScheme::Https => WINHTTP_FLAG_SECURE,
                    },
                )
            };
            let request = Self::non_null_handle(raw_request, "WinHttpOpenRequest", context)?;
            Self::set_dword_option(
                &request,
                WINHTTP_OPTION_DISABLE_FEATURE,
                WINHTTP_DISABLE_COOKIES
                    | WINHTTP_DISABLE_REDIRECTS
                    | WINHTTP_DISABLE_AUTHENTICATION,
                "WinHttpSetOption DISABLE_FEATURE",
                context,
            )?;
            Self::set_dword_option(
                &request,
                WINHTTP_OPTION_MAX_RESPONSE_HEADER_SIZE,
                HTTP_HEAD_MAX_HEADER_BYTES_DWORD,
                "WinHttpSetOption MAX_RESPONSE_HEADER_SIZE",
                context,
            )?;
            let request_start = Instant::now();
            // SAFETY: request is valid and no additional request body or headers are needed for HEAD.
            let sent =
                unsafe { sys::WinHttpSendRequest(request.0.as_ptr(), null(), 0, null(), 0, 0, 0) };
            (sent != 0_i32).ok_or_else(|| Self::last_error("WinHttpSendRequest", context))?;
            // SAFETY: request is a valid request handle and no reserved pointer is required.
            let received = unsafe { sys::WinHttpReceiveResponse(request.0.as_ptr(), null_mut()) };
            (received != 0_i32)
                .ok_or_else(|| Self::last_error("WinHttpReceiveResponse", context))?;
            let response_received = Instant::now();
            Ok((request, request_start, response_received))
        })()
        .inspect_err(|_| self.session_cache = None)?;
        let server_time = self.query_server_time(&request, response_received, context)?;
        let rtt = response_received
            .checked_duration_since(request_start)
            .ok_or_else(|| TimeError::parse("HTTP 응답 시각이 요청 시작 시각보다 앞섭니다."))?
            .max(MIN_TRANSFER_TIME);
        Ok(TimeSample {
            response_received_inst: response_received,
            rtt,
            server_time,
        })
    }
    fn last_error(operation: &str, context: &str) -> TimeError {
        let code = Self::last_error_code();
        Self::windows_error(operation, code, context)
    }
    fn last_error_code() -> u32 {
        // SAFETY: GetLastError has no preconditions.
        unsafe { sys::GetLastError() }
    }
    fn non_null_handle(handle: HInternet, operation: &str, context: &str) -> Result<Handle> {
        NonNull::new(handle)
            .map(Handle)
            .ok_or_else(|| Self::last_error(operation, context))
    }
    fn query_server_time(
        &mut self,
        request: &Handle,
        response_received: Instant,
        context: &str,
    ) -> Result<SystemTime> {
        let mut time_headers = FreshTimeHeaders::default();
        Self::read_header_values(
            request,
            WINHTTP_QUERY_DATE,
            "Date",
            context,
            (&mut self.header_buffer, &mut self.header_line_buffer),
            |value| time_headers.capture_date(value, response_received),
        )?;
        Self::read_header_values(
            request,
            WINHTTP_QUERY_AGE,
            "Age",
            context,
            (&mut self.header_buffer, &mut self.header_line_buffer),
            |value| time_headers.capture_age(value),
        )?;
        time_headers
            .finish(context)
            .map(|(server_time, _received_at)| server_time)
    }
    fn read_header_values(
        request: &Handle,
        query: u32,
        name: &str,
        context: &str,
        buffers: (&mut Vec<u16>, &mut Vec<u8>),
        mut capture: impl FnMut(&str),
    ) -> Result<()> {
        let (wide_buffer, ascii_buffer) = buffers;
        let mut index = 0_u32;
        loop {
            let current_index = index;
            let mut bytes = 0_u32;
            // SAFETY: request is valid; this call probes one indexed response header value.
            let probed = unsafe {
                sys::WinHttpQueryHeaders(
                    request.0.as_ptr(),
                    query,
                    null(),
                    null_mut(),
                    &raw mut bytes,
                    &raw mut index,
                )
            };
            if probed != 0_i32 {
                return Err(error(
                    context,
                    format!("{name} 헤더 크기 조회가 예기치 않게 성공했습니다."),
                ));
            }
            let code = Self::last_error_code();
            if code == ERROR_WINHTTP_HEADER_NOT_FOUND {
                return Ok(());
            }
            if code != ERROR_INSUFFICIENT_BUFFER {
                return Err(Self::windows_error(
                    "WinHttpQueryHeaders header size",
                    code,
                    context,
                ));
            }
            let header_bytes = usize::try_from(bytes).unwrap_or_else(|_| process::abort());
            if header_bytes > HTTP_HEAD_MAX_HEADER_BYTES {
                return Err(error(
                    context,
                    format!(
                        "{name} 헤더가 허용 한도({HTTP_HEAD_MAX_HEADER_BYTES} bytes)를 초과했습니다."
                    ),
                ));
            }
            if !header_bytes.is_multiple_of(2) {
                return Err(error(
                    context,
                    format!("{name} 헤더 UTF-16 길이가 올바르지 않습니다."),
                ));
            }
            let units = header_bytes.div_euclid(2);
            wide_buffer.clear();
            if wide_buffer.capacity() < units {
                wide_buffer.try_reserve_exact(units).map_err(|source| {
                    error_with_source(context, "응답 헤더 메모리 확보 실패", source)
                })?;
            }
            wide_buffer.resize(units, 0_u16);
            index = current_index;
            // SAFETY: wide_buffer has the probed size and request is valid.
            let fetched = unsafe {
                sys::WinHttpQueryHeaders(
                    request.0.as_ptr(),
                    query,
                    null(),
                    wide_buffer.as_mut_ptr().cast::<c_void>(),
                    &raw mut bytes,
                    &raw mut index,
                )
            };
            (fetched != 0_i32)
                .ok_or_else(|| Self::last_error("WinHttpQueryHeaders header", context))?;
            if index <= current_index {
                return Err(error(
                    context,
                    format!("{name} 헤더 index가 진행되지 않았습니다."),
                ));
            }
            while wide_buffer.pop_if(|unit| *unit == 0).is_some() {}
            ascii_buffer.clear();
            if ascii_buffer.capacity() < wide_buffer.len() {
                ascii_buffer
                    .try_reserve_exact(wide_buffer.len())
                    .map_err(|source| {
                        error_with_source(context, "응답 헤더 ASCII 메모리 확보 실패", source)
                    })?;
            }
            for &unit in wide_buffer.iter() {
                ascii_buffer.push(u8::try_from(unit).map_err(|source| {
                    error_with_source(context, format!("{name} 헤더 ASCII 변환 실패"), source)
                })?);
            }
            let value = str::from_utf8(ascii_buffer)
                .map(str::trim_ascii)
                .map_err(|source| {
                    error_with_source(context, format!("{name} 헤더 UTF-8 변환 실패"), source)
                })?;
            capture(value);
        }
    }
    fn set_dword_option(
        handle: &Handle,
        option: u32,
        value: u32,
        operation: &str,
        context: &str,
    ) -> Result<()> {
        Self::try_set_dword_option(handle, option, value)
            .map_err(|code| Self::windows_error(operation, code, context))
    }
    fn try_set_dword_option(handle: &Handle, option: u32, value: u32) -> CoreResult<(), u32> {
        // SAFETY: handle is a valid WinHTTP handle and value points to a DWORD option value.
        let ok = unsafe {
            sys::WinHttpSetOption(
                handle.0.as_ptr(),
                option,
                (&raw const value).cast::<c_void>(),
                DWORD_BYTE_SIZE,
            )
        };
        (ok != 0_i32).ok_or_else(Self::last_error_code)
    }
    fn windows_error(operation: &str, code: u32, context: &str) -> TimeError {
        error(context, format!("{operation} 실패: Windows error {code}"))
    }
}
