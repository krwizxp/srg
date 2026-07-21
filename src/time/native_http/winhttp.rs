use super::{MIN_TRANSFER_TIME, Result, TimeError, TimeSample, error, error_with_source};
use super::super::{
    ParsedServer, http_date::parse_http_date_to_systemtime, sample::validate_http_age_value,
};
use alloc::vec::Vec;
use core::{
    ffi::c_void,
    mem,
    ptr::{NonNull, null, null_mut},
    result::Result as CoreResult,
    str,
};
use std::{
    ffi::OsStr,
    os::windows::ffi::OsStrExt as WindowsOsStrExt,
    time::{Instant, SystemTime},
};
mod sys;
const DWORD_BYTE_SIZE: u32 = 4;
const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
const AGE_HEADER_PREFIX: &[u8] = b"age:";
const DATE_HEADER_PREFIX: &[u8] = b"date:";
const WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY: u32 = 4;
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
const WINHTTP_QUERY_RAW_HEADERS_CRLF: u32 = 22;
const HTTP_HEAD_MAX_HEADER_BYTES: usize = 64 * 1024;
const HTTP_HEAD_MAX_HEADER_BYTES_DWORD: u32 = 64 * 1024;
const HTTP_HEADER_LINE_STACK_BYTES: usize = 256;
const METHOD_HEAD_WIDE: [u16; 5] = [0x48, 0x45, 0x41, 0x44, 0];
const PATH_ROOT_WIDE: [u16; 2] = [0x2F, 0];
const WINHTTP_TIMEOUT_MILLIS: i32 = 5_000;
type HInternet = *mut c_void;
#[derive(Default)]
pub(in crate::time) struct Client {
    header_buffer: Vec<u16>,
    header_line_buffer: Vec<u8>,
    session_cache: Option<SessionCache>,
}
struct Handle(NonNull<c_void>);
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
    fn ascii_header_value<'line>(
        line: &'line [u8],
        prefix_len: usize,
        context: &str,
        header_name: &str,
    ) -> Result<&'line str> {
        let Some(value) = line.get(prefix_len..) else {
            return Err(error(context, format!("{header_name} 헤더 prefix 길이 오류")));
        };
        str::from_utf8(value)
            .map(str::trim_ascii)
            .map_err(|source| error_with_source(context, format!("{header_name} 헤더 UTF-8 변환 실패"), source))
    }
    fn cached_connect_ptr(
        &mut self,
        server: &ParsedServer,
        host_wide: &[u16],
        context: &str,
    ) -> Result<HInternet> {
        let cache = if let Some(ref mut cache) = self.session_cache {
            cache
        } else {
                let user_agent =
                    wide(concat!("srg/", env!("CARGO_PKG_VERSION")), context)?;
                // SAFETY: user_agent is NUL-terminated and optional proxy pointers are intentionally null.
                let raw_session = unsafe {
                    sys::WinHttpOpen(
                        user_agent.as_ptr(),
                        WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
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
            return Ok(connect.0.as_ptr());
        }
        // SAFETY: host_wide is NUL-terminated and cache.session is a valid session handle.
        let raw_connect = unsafe {
            sys::WinHttpConnect(
                cache.session.0.as_ptr(),
                host_wide.as_ptr(),
                server.port,
                0,
            )
        };
        let handle = NonNull::new(raw_connect)
            .map(Handle)
            .ok_or_else(|| Self::last_error("WinHttpConnect", context))?;
        let connect = handle.0.as_ptr();
        cache.connect = Some(handle);
        Ok(connect)
    }
    pub(in crate::time) fn fetch_head(
        &mut self,
        server: &ParsedServer,
        context: &str,
    ) -> Result<TimeSample> {
        let host_wide = wide(&server.host, context)?;
        let connect = self.cached_connect_ptr(server, &host_wide, context)?;
        let (request, request_start, response_received) = (|| -> Result<_> {
            // SAFETY: method and path are NUL-terminated and connect is valid.
            let raw_request = unsafe {
                sys::WinHttpOpenRequest(
                    connect,
                    METHOD_HEAD_WIDE.as_ptr(),
                    PATH_ROOT_WIDE.as_ptr(),
                    null(),
                    null(),
                    null(),
                    WINHTTP_FLAG_SECURE,
                )
            };
            let request =
                Self::non_null_handle(raw_request, "WinHttpOpenRequest", context)?;
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
            let sent = unsafe {
                sys::WinHttpSendRequest(request.0.as_ptr(), null(), 0, null(), 0, 0, 0)
            };
            if sent == 0_i32 {
                return Err(Self::last_error("WinHttpSendRequest", context));
            }
            // SAFETY: request is a valid request handle and no reserved pointer is required.
            let received =
                unsafe { sys::WinHttpReceiveResponse(request.0.as_ptr(), null_mut()) };
            if received == 0_i32 {
                return Err(Self::last_error("WinHttpReceiveResponse", context));
            }
            let response_received = Instant::now();
            Ok((request, request_start, response_received))
        })()
        .inspect_err(|_| self.session_cache = None)?;
        let server_time = self.query_server_time(&request, context)?;
        let rtt = response_received
            .saturating_duration_since(request_start)
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
    fn line_starts_with_ascii_ignore_case(line: &[u16], prefix: &[u8]) -> bool {
        if line.len() < prefix.len() {
            return false;
        }
        line.iter()
            .zip(prefix)
            .all(|(&unit, &byte)| u8::try_from(unit).is_ok_and(|unit_byte| unit_byte.eq_ignore_ascii_case(&byte)))
    }
    fn non_null_handle(handle: HInternet, operation: &str, context: &str) -> Result<Handle> {
        NonNull::new(handle)
            .map(Handle)
            .ok_or_else(|| Self::last_error(operation, context))
    }
    fn query_raw_headers(&mut self, request: &Handle, context: &str) -> Result<()> {
        let mut bytes = 0_u32;
        let mut index = 0_u32;
        // SAFETY: request is valid; this first call probes the required buffer size.
        let probe_ok = unsafe {
            sys::WinHttpQueryHeaders(
                request.0.as_ptr(),
                WINHTTP_QUERY_RAW_HEADERS_CRLF,
                null(),
                null_mut(),
                &raw mut bytes,
                &raw mut index,
            )
        };
        if probe_ok != 0_i32 {
            return Err(TimeError::header_not_found(format!("{context} 응답에서 Date")));
        }
        let code = Self::last_error_code();
        if code != ERROR_INSUFFICIENT_BUFFER {
            return Err(Self::windows_error("WinHttpQueryHeaders", code, context));
        }
        let header_bytes = usize::try_from(bytes)
            .map_err(|source| error_with_source(context, "응답 헤더 길이 변환 실패", source))?;
        if header_bytes > HTTP_HEAD_MAX_HEADER_BYTES {
            return Err(error(context, format!("응답 헤더가 허용 한도({HTTP_HEAD_MAX_HEADER_BYTES} bytes)를 초과했습니다.")));
        }
        if !header_bytes.is_multiple_of(2) {
            return Err(error(
                context,
                "응답 헤더 UTF-16 버퍼 길이가 2바이트 단위가 아닙니다.",
            ));
        }
        let units = header_bytes.div_euclid(2);
        let buffer = &mut self.header_buffer;
        buffer.clear();
        if buffer.capacity() < units {
            buffer.try_reserve_exact(units).map_err(|source| {
                error_with_source(context, "응답 헤더 버퍼 메모리 확보 실패", source)
            })?;
        }
        buffer.resize(units, 0_u16);
        index = 0;
        // SAFETY: buffer has the size requested by WinHTTP and request is valid.
        let fetch_ok = unsafe {
            sys::WinHttpQueryHeaders(
                request.0.as_ptr(),
                WINHTTP_QUERY_RAW_HEADERS_CRLF,
                null(),
                buffer.as_mut_ptr().cast::<c_void>(),
                &raw mut bytes,
                &raw mut index,
            )
        };
        if fetch_ok == 0_i32 {
            return Err(Self::last_error("WinHttpQueryHeaders", context));
        }
        while buffer.pop_if(|value| *value == 0).is_some() {}
        Ok(())
    }
    fn query_server_time(
        &mut self,
        request: &Handle,
        context: &str,
    ) -> Result<SystemTime> {
        self.query_raw_headers(request, context)?;
        let buffer = mem::take(&mut self.header_buffer);
        let mut line_bytes = mem::take(&mut self.header_line_buffer);
        let result = (|| {
            let mut age_seen = false;
            let mut line_stack = [0_u8; HTTP_HEADER_LINE_STACK_BYTES];
            let mut parsed_date = None;
            for raw_line in buffer.split(|unit| *unit == u16::from(b'\n')) {
                let line = raw_line
                    .strip_suffix(&[u16::from(b'\r')])
                    .unwrap_or(raw_line);
                if line.is_empty() {
                    continue;
                }
                let is_age_header =
                    Self::line_starts_with_ascii_ignore_case(line, AGE_HEADER_PREFIX);
                let is_date_header =
                    Self::line_starts_with_ascii_ignore_case(line, DATE_HEADER_PREFIX);
                if !(is_age_header || is_date_header) {
                    continue;
                }
                let line_ascii = if line.len() <= line_stack.len() {
                    let stack_line = line_stack
                        .get_mut(..line.len())
                        .ok_or_else(|| error(context, "응답 헤더 stack buffer 범위 오류"))?;
                    for (&unit, byte) in line.iter().zip(stack_line.iter_mut()) {
                        *byte = u8::try_from(unit).map_err(|source| {
                            error_with_source(context, "응답 헤더 ASCII 변환 실패", source)
                        })?;
                    }
                    stack_line
                } else {
                    line_bytes.clear();
                    if line_bytes.capacity() < line.len() {
                        line_bytes.try_reserve_exact(line.len()).map_err(|source| {
                            error_with_source(
                                context,
                                "응답 헤더 ASCII buffer 메모리 확보 실패",
                                source,
                            )
                        })?;
                    }
                    for &unit in line {
                        line_bytes.push(u8::try_from(unit).map_err(|source| {
                            error_with_source(context, "응답 헤더 ASCII 변환 실패", source)
                        })?);
                    }
                    line_bytes.as_slice()
                };
                if is_age_header {
                    let age_header_raw = Self::ascii_header_value(
                        line_ascii,
                        AGE_HEADER_PREFIX.len(),
                        context,
                        "Age",
                    )?;
                    if age_seen {
                        return Err(error(context, "Age 헤더가 여러 개입니다."));
                    }
                    age_seen = true;
                    validate_http_age_value(age_header_raw)
                        .map_err(|message| error(context, message))?;
                }
                if is_date_header {
                    let date_header_raw = Self::ascii_header_value(
                        line_ascii,
                        DATE_HEADER_PREFIX.len(),
                        context,
                        "Date",
                    )?;
                    if parsed_date.is_some() {
                        return Err(error(context, "Date 헤더가 여러 개입니다."));
                    }
                    parsed_date = Some(parse_http_date_to_systemtime(date_header_raw)?);
                }
            }
            parsed_date.ok_or_else(|| TimeError::header_not_found(format!("{context} 응답에서 Date")))
        })();
        self.header_buffer = buffer;
        self.header_line_buffer = line_bytes;
        result
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
    fn try_set_dword_option(
        handle: &Handle,
        option: u32,
        value: u32,
    ) -> CoreResult<(), u32> {
        // SAFETY: handle is a valid WinHTTP handle and value points to a DWORD option value.
        let ok = unsafe {
            sys::WinHttpSetOption(
                handle.0.as_ptr(),
                option,
                (&raw const value).cast::<c_void>(),
                DWORD_BYTE_SIZE,
            )
        };
        if ok == 0_i32 {
            Err(Self::last_error_code())
        } else {
            Ok(())
        }
    }
    fn windows_error(operation: &str, code: u32, context: &str) -> TimeError {
        error(context, format!("{operation} 실패: Windows error {code}"))
    }
}
fn wide(value: &str, context: &str) -> Result<Vec<u16>> {
    let capacity = value.len().wrapping_add(1);
    let mut out = Vec::new();
    out.try_reserve_exact(capacity)
        .map_err(|source| error_with_source(context, "wide 문자열 메모리 확보 실패", source))?;
    out.extend(<OsStr as WindowsOsStrExt>::encode_wide(OsStr::new(value)));
    out.push(0);
    Ok(out)
}
