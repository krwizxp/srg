use super::{
    HeadResponse, MIN_TRANSFER_TIME, ParseHttpDate, Result, TCP_TIMEOUT, TimeError, error,
};
use super::super::sample::{
    find_age_header_value, find_date_header_value, validate_http_age_value,
};
use alloc::{string::String, vec::Vec};
use core::{
    ffi::c_void,
    ptr::{NonNull, null, null_mut},
};
use std::{
    ffi::OsStr,
    os::windows::ffi::OsStrExt as WindowsOsStrExt,
    time::{Instant, SystemTime},
};
mod sys {
    use super::{HInternet, c_void};
    #[link(name = "winhttp")]
    unsafe extern "system" {
        pub(super) fn WinHttpCloseHandle(h_internet: HInternet) -> i32;
        pub(super) fn WinHttpConnect(
            h_session: HInternet,
            server_name: *const u16,
            server_port: u16,
            reserved: u32,
        ) -> HInternet;
        pub(super) fn WinHttpOpen(
            user_agent: *const u16,
            access_type: u32,
            proxy_name: *const u16,
            proxy_bypass: *const u16,
            flags: u32,
        ) -> HInternet;
        pub(super) fn WinHttpOpenRequest(
            h_connect: HInternet,
            verb: *const u16,
            object_name: *const u16,
            version: *const u16,
            referrer: *const u16,
            accept_types: *const *const u16,
            flags: u32,
        ) -> HInternet;
        pub(super) fn WinHttpQueryHeaders(
            h_request: HInternet,
            info_level: u32,
            name: *const u16,
            buffer: *mut c_void,
            buffer_length: *mut u32,
            index: *mut u32,
        ) -> i32;
        pub(super) fn WinHttpReceiveResponse(h_request: HInternet, reserved: *mut c_void) -> i32;
        pub(super) fn WinHttpSendRequest(
            h_request: HInternet,
            headers: *const u16,
            headers_length: u32,
            optional: *const c_void,
            optional_length: u32,
            total_length: u32,
            context: usize,
        ) -> i32;
        pub(super) fn WinHttpSetOption(
            h_internet: HInternet,
            option: u32,
            buffer: *mut c_void,
            buffer_length: u32,
        ) -> i32;
        pub(super) fn WinHttpSetTimeouts(
            h_internet: HInternet,
            resolve_timeout: i32,
            connect_timeout: i32,
            send_timeout: i32,
            receive_timeout: i32,
        ) -> i32;
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub(super) fn GetLastError() -> u32;
    }
}
const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
const HTTP_SCHEME_PREFIX: &str = "http://";
const HTTPS_SCHEME_PREFIX: &str = "https://";
const INTERNET_DEFAULT_HTTP_PORT: u16 = 80;
const INTERNET_DEFAULT_HTTPS_PORT: u16 = 443;
const WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY: u32 = 4;
const WINHTTP_FLAG_SECURE: u32 = 0x0080_0000;
const WINHTTP_OPTION_DISABLE_FEATURE: u32 = 63;
const WINHTTP_OPTION_REDIRECT_POLICY: u32 = 88;
const WINHTTP_OPTION_SECURE_PROTOCOLS: u32 = 84;
const WINHTTP_OPTION_MAX_RESPONSE_HEADER_SIZE: u32 = 91;
const WINHTTP_OPTION_DISABLE_SECURE_PROTOCOL_FALLBACK: u32 = 144;
const WINHTTP_OPTION_IPV6_FAST_FALLBACK: u32 = 140;
const WINHTTP_OPTION_DISABLE_GLOBAL_POOLING: u32 = 195;
const WINHTTP_OPTION_REDIRECT_POLICY_DISALLOW_HTTPS_TO_HTTP: u32 = 1;
const WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2: u32 = 0x0000_0800;
const WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_3: u32 = 0x0000_2000;
const WINHTTP_SECURE_PROTOCOLS_MIN_TLS_1_2: u32 =
    WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2 | WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_3;
const WINHTTP_DISABLE_AUTHENTICATION: u32 = 0x0000_0004;
const WINHTTP_DISABLE_COOKIES: u32 = 0x0000_0001;
const ERROR_INVALID_PARAMETER: u32 = 87;
const ERROR_WINHTTP_INVALID_OPTION: u32 = 12_009;
const ERROR_WINHTTP_OPTION_NOT_SETTABLE: u32 = 12_011;
const WINHTTP_QUERY_RAW_HEADERS_CRLF: u32 = 22;
const WINHTTP_CONNECT_CACHE_LIMIT: usize = 4;
const HTTP_HEAD_MAX_HEADER_BYTES: usize = 64 * 1024;
const METHOD_HEAD_WIDE: [u16; 5] = [0x48, 0x45, 0x41, 0x44, 0];
const PATH_ROOT_WIDE: [u16; 2] = [0x2F, 0];
type HInternet = *mut c_void;
pub(super) struct Client {
    error_code_label: &'static str,
    http_scheme_prefix: &'static str,
    https_scheme_prefix: &'static str,
    session_cache: Option<SessionCache>,
}
struct RequestTarget {
    host: String,
    port: u16,
    secure: bool,
}
struct Handle(NonNull<c_void>);
struct WinHttpHeadPerform {
    request: Handle,
    request_start: Instant,
    response_received: Instant,
}
struct CachedConnect {
    handle: Handle,
    host: String,
    port: u16,
}
struct SessionCache {
    connects: Vec<CachedConnect>,
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
    fn cached_connect_ptr(
        &mut self,
        host: &str,
        host_wide: &[u16],
        port: u16,
        context: &str,
    ) -> Result<HInternet> {
        if self.session_cache.is_none() {
            let user_agent = wide(concat!("srg/", env!("CARGO_PKG_VERSION")), context)?;
            self.session_cache = Some(SessionCache {
                connects: Vec::new(),
                session: self.open_session(&user_agent, context)?,
            });
        }
        let error_code_label = self.error_code_label;
        let cache = self
            .session_cache
            .as_mut()
            .ok_or_else(|| error(context, "WinHTTP session cache 상태 오류"))?;
        if let Some(index) = cache
            .connects
            .iter()
            .position(|entry| entry.port == port && entry.host.as_str() == host)
        {
            return cache
                .connects
                .get(index)
                .map(|entry| &entry.handle)
                .map(|handle| handle.0.as_ptr())
                .ok_or_else(|| error(context, "WinHTTP connect cache 범위 오류"));
        }
        // SAFETY: host_wide is NUL-terminated and cache.session is a valid session handle.
        let raw_connect =
            unsafe { sys::WinHttpConnect(cache.session.0.as_ptr(), host_wide.as_ptr(), port, 0) };
        let handle = NonNull::new(raw_connect)
            .map(Handle)
            .ok_or_else(|| Self::last_error_for(error_code_label, "WinHttpConnect", context))?;
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
        Ok(entry.handle.0.as_ptr())
    }
    fn clear_session_cache(&mut self) {
        self.session_cache = None;
    }
    pub(super) fn fetch_head(
        &mut self,
        url: &str,
        context: &str,
        parse_http_date: ParseHttpDate,
    ) -> Result<HeadResponse> {
        let target = self.request_target(url, context)?;
        let host_wide = wide(&target.host, context)?;
        let flags = if target.secure {
            WINHTTP_FLAG_SECURE
        } else {
            0
        };
        let connect = self.cached_connect_ptr(
            &target.host,
            &host_wide,
            target.port,
            context,
        )?;
        let perform: Result<WinHttpHeadPerform> = (|| {
            let request = self.open_request(connect, &METHOD_HEAD_WIDE, &PATH_ROOT_WIDE, flags, context)?;
            self.set_request_options(&request, context)?;
            let request_start = Instant::now();
            self.send_head(&request, context)?;
            self.receive_response(&request, context)?;
            let response_received = Instant::now();
            Ok(WinHttpHeadPerform {
                request,
                request_start,
                response_received,
            })
        })();
        if perform.is_err() {
            self.clear_session_cache();
        }
        let perform_result = perform?;
        let server_time = self.query_server_time(&perform_result.request, context, parse_http_date)?;
        let rtt = perform_result
            .response_received
            .saturating_duration_since(perform_result.request_start)
            .max(MIN_TRANSFER_TIME);
        Ok(HeadResponse {
            response_received_inst: perform_result.response_received,
            rtt,
            server_time,
        })
    }
    fn last_error(&self, operation: &str, context: &str) -> super::TimeError {
        Self::last_error_for(self.error_code_label, operation, context)
    }
    fn last_error_code() -> u32 {
        // SAFETY: GetLastError has no preconditions.
        unsafe { sys::GetLastError() }
    }
    fn last_error_for(
        error_code_label: &'static str,
        operation: &str,
        context: &str,
    ) -> super::TimeError {
        let code = Self::last_error_code();
        error(context, format!("{operation} 실패: {error_code_label} {code}"))
    }
    fn line_starts_with_ascii_ignore_case(line: &[u16], prefix: &[u8]) -> bool {
        if line.len() < prefix.len() {
            return false;
        }
        line.iter()
            .zip(prefix)
            .all(|(&unit, &byte)| u8::try_from(unit).is_ok_and(|unit_byte| unit_byte.eq_ignore_ascii_case(&byte)))
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
            sys::WinHttpOpenRequest(
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
            sys::WinHttpOpen(
                user_agent.as_ptr(),
                WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
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
        let timeout_ok = unsafe {
            sys::WinHttpSetTimeouts(session.0.as_ptr(), timeout, timeout, timeout, timeout)
        };
        if timeout_ok == 0_i32 {
            return Err(self.last_error("WinHttpSetTimeouts", context));
        }
        self.set_secure_protocols(&session, context)?;
        self.set_optional_dword_option(
            &session,
            WINHTTP_OPTION_DISABLE_SECURE_PROTOCOL_FALLBACK,
            1,
            "WinHttpSetOption DISABLE_SECURE_PROTOCOL_FALLBACK",
            context,
        )?;
        self.set_dword_option(
            &session,
            WINHTTP_OPTION_DISABLE_GLOBAL_POOLING,
            1,
            "WinHttpSetOption DISABLE_GLOBAL_POOLING",
            context,
        )?;
        self.set_optional_dword_option(
            &session,
            WINHTTP_OPTION_IPV6_FAST_FALLBACK,
            1,
            "WinHttpSetOption IPV6_FAST_FALLBACK",
            context,
        )?;
        Ok(session)
    }
    fn query_raw_headers(&self, request: &Handle, context: &str) -> Result<Vec<u16>> {
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
        if Self::last_error_code() != ERROR_INSUFFICIENT_BUFFER {
            return Err(self.last_error("WinHttpQueryHeaders", context));
        }
        let header_bytes = usize::try_from(bytes)
            .map_err(|source| error(context, format!("응답 헤더 길이 변환 실패: {source}")))?;
        if header_bytes > HTTP_HEAD_MAX_HEADER_BYTES {
            return Err(error(context, format!("응답 헤더가 허용 한도({HTTP_HEAD_MAX_HEADER_BYTES} bytes)를 초과했습니다.")));
        }
        if !header_bytes.is_multiple_of(2) {
            return Err(error(
                context,
                "응답 헤더 UTF-16 버퍼 길이가 2바이트 단위가 아닙니다.",
            ));
        }
        let units = header_bytes
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
            return Err(self.last_error("WinHttpQueryHeaders", context));
        }
        while buffer.pop_if(|value| *value == 0).is_some() {}
        Ok(buffer)
    }
    fn query_server_time(
        &self,
        request: &Handle,
        context: &str,
        parse_http_date: ParseHttpDate,
    ) -> Result<SystemTime> {
        let buffer = self.query_raw_headers(request, context)?;
        let mut age_seen = false;
        let mut date_seen = false;
        let mut parsed_date = None;
        for raw_line in buffer.split(|unit| *unit == u16::from(b'\n')) {
            let line = raw_line
                .strip_suffix(&[u16::from(b'\r')])
                .map_or(raw_line, |stripped| stripped);
            if line.is_empty() {
                continue;
            }
            let is_age_header = Self::line_starts_with_ascii_ignore_case(line, b"age:");
            let is_date_header = Self::line_starts_with_ascii_ignore_case(line, b"date:");
            if !(is_age_header || is_date_header) {
                continue;
            }
            let line_bytes = line
                .iter()
                .map(|&unit| {
                    u8::try_from(unit).map_err(|source| {
                        error(context, format!("응답 헤더 ASCII 변환 실패: {source}"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if is_age_header {
                match find_age_header_value(&line_bytes) {
                    Ok(Some(age_header_raw)) => {
                        if age_seen {
                            return Err(error(context, "Age 헤더가 여러 개입니다."));
                        }
                        age_seen = true;
                        validate_http_age_value(age_header_raw)
                            .map_err(|message| error(context, message))?;
                    }
                    Ok(None) => {}
                    Err(source) => {
                        return Err(error(
                            context,
                            format!("Age 헤더 UTF-8 변환 실패: {source}"),
                        ));
                    }
                }
            }
            if is_date_header {
                match find_date_header_value(&line_bytes) {
                    Ok(Some(date_header_raw)) => {
                        if date_seen {
                            return Err(error(context, "Date 헤더가 여러 개입니다."));
                        }
                        date_seen = true;
                        parsed_date = Some(parse_http_date(date_header_raw)?);
                    }
                    Ok(None) => {}
                    Err(source) => {
                        return Err(error(
                            context,
                            format!("Date 헤더 UTF-8 변환 실패: {source}"),
                        ));
                    }
                }
            }
        }
        parsed_date.ok_or_else(|| TimeError::header_not_found(format!("{context} 응답에서 Date")))
    }
    fn receive_response(&self, request: &Handle, context: &str) -> Result<()> {
        // SAFETY: request is a valid request handle and no reserved pointer is required.
        let received = unsafe { sys::WinHttpReceiveResponse(request.0.as_ptr(), null_mut()) };
        if received == 0_i32 {
            Err(self.last_error("WinHttpReceiveResponse", context))
        } else {
            Ok(())
        }
    }
    fn request_target(&self, url: &str, context: &str) -> Result<RequestTarget> {
        struct RequestScheme<'url> {
            rest: &'url str,
            secure: bool,
        }
        let scheme = if let Some(rest) = url.strip_prefix(self.https_scheme_prefix) {
            RequestScheme { rest, secure: true }
        } else if let Some(rest) = url.strip_prefix(self.http_scheme_prefix) {
            RequestScheme {
                rest,
                secure: false,
            }
        } else {
            return Err(error(context, "지원하지 않는 URL scheme입니다."));
        };
        let authority = scheme
            .rest
            .split_once(['/', '?', '#'])
            .map_or(scheme.rest, |(authority, _)| authority);
        if authority.is_empty() {
            return Err(error(context, "URL host가 비어 있습니다."));
        }
        let default_port = if scheme.secure {
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
        let target = if let Some(bracketed) = authority.strip_prefix('[') {
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
            RequestTarget {
                host: copy_host(host)?,
                port,
                secure: scheme.secure,
            }
        } else if let Some((host, port_text)) = authority.rsplit_once(':') {
            if host.contains(':') {
                RequestTarget {
                    host: copy_host(authority)?,
                    port: default_port,
                    secure: scheme.secure,
                }
            } else {
                RequestTarget {
                    host: copy_host(host)?,
                    port: parse_port(port_text)?,
                    secure: scheme.secure,
                }
            }
        } else {
            RequestTarget {
                host: copy_host(authority)?,
                port: default_port,
                secure: scheme.secure,
            }
        };
        Ok(target)
    }
    fn send_head(&self, request: &Handle, context: &str) -> Result<()> {
        // SAFETY: request is valid and no additional request body or headers are needed for HEAD.
        let sent =
            unsafe { sys::WinHttpSendRequest(request.0.as_ptr(), null(), 0, null(), 0, 0, 0) };
        if sent == 0_i32 {
            Err(self.last_error("WinHttpSendRequest", context))
        } else {
            Ok(())
        }
    }
    fn set_dword_option(
        &self,
        handle: &Handle,
        option: u32,
        value: u32,
        operation: &str,
        context: &str,
    ) -> Result<()> {
        let mut raw_value = value;
        let buffer_length = u32::try_from(size_of::<u32>())
            .map_err(|source| error(context, format!("옵션 길이 변환 실패: {source}")))?;
        // SAFETY: handle is a valid WinHTTP handle and raw_value points to a DWORD option value.
        let ok = unsafe {
            sys::WinHttpSetOption(
                handle.0.as_ptr(),
                option,
                (&raw mut raw_value).cast::<c_void>(),
                buffer_length,
            )
        };
        if ok == 0_i32 {
            Err(self.last_error(operation, context))
        } else {
            Ok(())
        }
    }
    fn set_optional_dword_option(
        &self,
        handle: &Handle,
        option: u32,
        value: u32,
        operation: &str,
        context: &str,
    ) -> Result<()> {
        let mut raw_value = value;
        let buffer_length = u32::try_from(size_of::<u32>())
            .map_err(|source| error(context, format!("옵션 길이 변환 실패: {source}")))?;
        // SAFETY: handle is a valid WinHTTP handle and raw_value points to a DWORD option value.
        let ok = unsafe {
            sys::WinHttpSetOption(
                handle.0.as_ptr(),
                option,
                (&raw mut raw_value).cast::<c_void>(),
                buffer_length,
            )
        };
        if ok != 0_i32 {
            return Ok(());
        }
        match Self::last_error_code() {
            ERROR_WINHTTP_INVALID_OPTION | ERROR_WINHTTP_OPTION_NOT_SETTABLE => Ok(()),
            _ => Err(self.last_error(operation, context)),
        }
    }
    fn set_request_options(&self, request: &Handle, context: &str) -> Result<()> {
        self.set_dword_option(
            request,
            WINHTTP_OPTION_REDIRECT_POLICY,
            WINHTTP_OPTION_REDIRECT_POLICY_DISALLOW_HTTPS_TO_HTTP,
            "WinHttpSetOption REDIRECT_POLICY",
            context,
        )?;
        self.set_dword_option(
            request,
            WINHTTP_OPTION_DISABLE_FEATURE,
            WINHTTP_DISABLE_COOKIES | WINHTTP_DISABLE_AUTHENTICATION,
            "WinHttpSetOption DISABLE_FEATURE",
            context,
        )?;
        self.set_dword_option(
            request,
            WINHTTP_OPTION_MAX_RESPONSE_HEADER_SIZE,
            u32::try_from(HTTP_HEAD_MAX_HEADER_BYTES)
                .map_err(|source| error(context, format!("WinHTTP 헤더 한도 변환 실패: {source}")))?,
            "WinHttpSetOption MAX_RESPONSE_HEADER_SIZE",
            context,
        )
    }
    fn set_secure_protocols(&self, session: &Handle, context: &str) -> Result<()> {
        let mut raw_value = WINHTTP_SECURE_PROTOCOLS_MIN_TLS_1_2;
        let buffer_length = u32::try_from(size_of::<u32>())
            .map_err(|source| error(context, format!("옵션 길이 변환 실패: {source}")))?;
        // SAFETY: session is a valid WinHTTP session handle and raw_value points to a DWORD.
        let ok = unsafe {
            sys::WinHttpSetOption(
                session.0.as_ptr(),
                WINHTTP_OPTION_SECURE_PROTOCOLS,
                (&raw mut raw_value).cast::<c_void>(),
                buffer_length,
            )
        };
        if ok != 0_i32 {
            return Ok(());
        }
        if matches!(
            Self::last_error_code(),
            ERROR_INVALID_PARAMETER | ERROR_WINHTTP_INVALID_OPTION
        ) {
            return self.set_dword_option(
                session,
                WINHTTP_OPTION_SECURE_PROTOCOLS,
                WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2,
                "WinHttpSetOption SECURE_PROTOCOLS",
                context,
            );
        }
        Err(self.last_error("WinHttpSetOption SECURE_PROTOCOLS", context))
    }
}
impl Default for Client {
    fn default() -> Self {
        Self {
            error_code_label: "Windows error",
            http_scheme_prefix: HTTP_SCHEME_PREFIX,
            https_scheme_prefix: HTTPS_SCHEME_PREFIX,
            session_cache: None,
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
