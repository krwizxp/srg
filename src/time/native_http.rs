use super::{
    MIN_TRANSFER_TIME, Result, TCP_TIMEOUT, TimeError, TimeErrorKind, TimeSample,
    find_date_header_value, http_date::parse_http_date_to_systemtime,
};
use alloc::{borrow::Cow, string::String};
use core::time::Duration;
use std::time::{Instant, SystemTime};
cfg_select! {
    any(target_os = "linux", target_os = "macos") => {
        mod platform {
    use super::{
        HeadResponse, MIN_TRANSFER_TIME, Result, TCP_TIMEOUT, TimeError, error,
        find_date_header_value,
    };
    use alloc::{ffi::CString, string::String, vec::Vec};
    use core::{
        ffi::{CStr, c_char, c_int, c_long, c_void},
        ptr::NonNull,
        slice,
    };
    use std::{sync::OnceLock, time::Instant};
    const CURLE_OK: CurlCode = 0;
    const CURL_ERROR_SIZE: usize = 256;
    const CURL_GLOBAL_DEFAULT: c_long = 3;
    const CURLOPT_CONNECTTIMEOUT: CurlOption = 78;
    const CURLOPT_ERRORBUFFER: CurlOption = 10_010;
    const CURLOPT_FOLLOWLOCATION: CurlOption = 52;
    const CURLOPT_HEADERDATA: CurlOption = 10_029;
    const CURLOPT_HEADERFUNCTION: CurlOption = 20_079;
    const CURLOPT_NOBODY: CurlOption = 44;
    const CURLOPT_NOSIGNAL: CurlOption = 99;
    const CURLOPT_TIMEOUT: CurlOption = 13;
    const CURLOPT_URL: CurlOption = 10_002;
    const CURLOPT_USERAGENT: CurlOption = 10_018;
    const CURLOPT_WRITEDATA: CurlOption = 10_001;
    const CURLOPT_WRITEFUNCTION: CurlOption = 20_011;
    pub(super) const CLIENT: Client = Client {
        user_agent: concat!("srg/", env!("CARGO_PKG_VERSION")),
    };
    type Curl = c_void;
    type CurlCode = c_int;
    type CurlOption = c_int;
    type CurlWriteCallback = unsafe extern "C" fn(*mut c_char, usize, usize, *mut c_void) -> usize;
    pub(super) struct Client {
        user_agent: &'static str,
    }
    struct EasyHandle(NonNull<Curl>);
    struct CurlBuffer {
        bytes: Vec<u8>,
    }
    #[link(name = "curl")]
    unsafe extern "C" {
        fn curl_easy_cleanup(curl: *mut Curl);
        fn curl_easy_init() -> *mut Curl;
        fn curl_easy_perform(curl: *mut Curl) -> CurlCode;
        fn curl_easy_setopt(curl: *mut Curl, option: CurlOption, ...) -> CurlCode;
        fn curl_easy_strerror(code: CurlCode) -> *const c_char;
        fn curl_global_init(flags: c_long) -> CurlCode;
    }
    impl Drop for EasyHandle {
        fn drop(&mut self) {
            // SAFETY: self.0 is an easy handle returned by libcurl and is closed exactly once here.
            unsafe {
                curl_easy_cleanup(self.0.as_ptr());
            }
        }
    }
    impl EasyHandle {
        const fn as_ptr(&self) -> *mut Curl {
            self.0.as_ptr()
        }
        fn perform(&self) -> CurlCode {
            // SAFETY: self.0 is configured with callbacks and buffers that live until the call returns.
            unsafe { curl_easy_perform(self.as_ptr()) }
        }
        fn setopt_callback(
            &self,
            option: CurlOption,
            value: CurlWriteCallback,
            context: &str,
        ) -> Result<()> {
            // SAFETY: value is a libcurl-compatible callback function pointer.
            let code = unsafe { curl_easy_setopt(self.as_ptr(), option, value) };
            if code == CURLE_OK {
                Ok(())
            } else {
                Err(error(context, curl_error("curl_easy_setopt", code)))
            }
        }
        fn setopt_long(&self, option: CurlOption, value: c_long, context: &str) -> Result<()> {
            // SAFETY: value is a scalar option value for the given CurlOption.
            let code = unsafe { curl_easy_setopt(self.as_ptr(), option, value) };
            if code == CURLE_OK {
                Ok(())
            } else {
                Err(error(context, curl_error("curl_easy_setopt", code)))
            }
        }
        fn setopt_ptr<T>(&self, option: CurlOption, value: *mut T, context: &str) -> Result<()> {
            // SAFETY: value is a pointer option that remains valid for the transfer duration.
            let code = unsafe { curl_easy_setopt(self.as_ptr(), option, value) };
            if code == CURLE_OK {
                Ok(())
            } else {
                Err(error(context, curl_error("curl_easy_setopt", code)))
            }
        }
        fn setopt_str(
            &self,
            option: CurlOption,
            value: *const c_char,
            context: &str,
        ) -> Result<()> {
            // SAFETY: value is a valid NUL-terminated string that outlives the setopt call.
            let code = unsafe { curl_easy_setopt(self.as_ptr(), option, value) };
            if code == CURLE_OK {
                Ok(())
            } else {
                Err(error(context, curl_error("curl_easy_setopt", code)))
            }
        }
    }
    impl Client {
        pub(super) fn fetch_head(&self, url: &str, context: &str) -> Result<HeadResponse> {
            static INIT: OnceLock<CurlCode> = OnceLock::new();
            // SAFETY: curl_global_init may be called once here before any easy handles are used.
            let init_code = *INIT.get_or_init(|| unsafe { curl_global_init(CURL_GLOBAL_DEFAULT) });
            if init_code != CURLE_OK {
                return Err(error(context, curl_error("curl_global_init", init_code)));
            }
            // SAFETY: curl_easy_init has no preconditions.
            let raw_handle_ptr = unsafe { curl_easy_init() };
            let Some(raw_handle) = NonNull::new(raw_handle_ptr) else {
                return Err(error(context, "curl_easy_init 실패"));
            };
            let handle = EasyHandle(raw_handle);
            let url_c = cstring("URL", url, context)?;
            let user_agent = cstring("User-Agent", self.user_agent, context)?;
            let mut error_buffer = [c_char::default(); CURL_ERROR_SIZE];
            let mut body_buffer = CurlBuffer { bytes: Vec::new() };
            let mut header_buffer = CurlBuffer { bytes: Vec::new() };
            handle.setopt_callback(CURLOPT_WRITEFUNCTION, write_vec_callback, context)?;
            handle.setopt_callback(CURLOPT_HEADERFUNCTION, write_vec_callback, context)?;
            handle.setopt_ptr(
                CURLOPT_WRITEDATA,
                (&raw mut body_buffer).cast::<c_void>(),
                context,
            )?;
            handle.setopt_ptr(
                CURLOPT_HEADERDATA,
                (&raw mut header_buffer).cast::<c_void>(),
                context,
            )?;
            handle.setopt_str(CURLOPT_URL, url_c.as_ptr(), context)?;
            handle.setopt_str(
                CURLOPT_USERAGENT,
                user_agent.as_ptr(),
                context,
            )?;
            handle.setopt_ptr(
                CURLOPT_ERRORBUFFER,
                error_buffer.as_mut_ptr(),
                context,
            )?;
            handle.setopt_long(
                CURLOPT_CONNECTTIMEOUT,
                tcp_timeout_secs(),
                context,
            )?;
            handle.setopt_long(CURLOPT_TIMEOUT, tcp_timeout_secs(), context)?;
            handle.setopt_long(CURLOPT_NOSIGNAL, 1, context)?;
            handle.setopt_long(CURLOPT_NOBODY, 1, context)?;
            handle.setopt_long(CURLOPT_FOLLOWLOCATION, 1, context)?;
            let request_start = Instant::now();
            let perform_code = handle.perform();
            let response_received_inst = Instant::now();
            if perform_code != CURLE_OK {
                let error_bytes = error_buffer.map(|ch| ch.to_le_bytes()[0]);
                let perform_error = CStr::from_bytes_until_nul(&error_bytes)
                    .ok()
                    .filter(|message| !message.to_bytes().is_empty())
                    .map_or_else(
                        || curl_error("curl_easy_perform", perform_code),
                        |message_cstr| {
                            let message = message_cstr.to_string_lossy();
                            format!("curl_easy_perform 실패: {message} ({perform_code})")
                        },
                    );
                return Err(error(context, perform_error));
            }
            let Some(date_header) = header_buffer
                .bytes
                .rsplit(|byte| *byte == b'\n')
                .find_map(find_date_header_value)
                .map(str::to_owned)
            else {
                let mut out = String::with_capacity(context.len().saturating_add(28));
                out.push_str(context);
                out.push_str(" 응답에서 Date");
                return Err(TimeError::header_not_found(out));
            };
            let rtt = response_received_inst
                .saturating_duration_since(request_start)
                .max(MIN_TRANSFER_TIME);
            Ok(HeadResponse {
                date_header,
                response_received_inst,
                rtt,
            })
        }
    }
    fn cstring(label: &str, value: &str, context: &str) -> Result<CString> {
        CString::new(value).map_err(|source| {
            error(
                context,
                format!("{label}에 NUL 문자가 포함되어 있습니다: {source}"),
            )
        })
    }
    fn curl_error(context: &str, code: CurlCode) -> String {
        // SAFETY: curl_easy_strerror returns either null or a static NUL-terminated message for code.
        let raw_ptr = unsafe { curl_easy_strerror(code) };
        let message = NonNull::new(raw_ptr.cast_mut()).map_or_else(
            || String::from("unknown curl error"),
            |message_ptr| {
                // SAFETY: libcurl guarantees a valid NUL-terminated string for non-null strerror results.
                unsafe { CStr::from_ptr(message_ptr.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            },
        );
        format!("{context} 실패: {message} ({code})")
    }
    unsafe extern "C" fn write_vec_callback(
        ptr: *mut c_char,
        size: usize,
        nmemb: usize,
        userdata: *mut c_void,
    ) -> usize {
        let Some(len) = size.checked_mul(nmemb) else {
            return 0;
        };
        if len == 0 {
            return 0;
        }
        let Some(bytes_ptr) = NonNull::new(ptr.cast::<u8>()) else {
            return 0;
        };
        let Some(mut target_ptr) = NonNull::new(userdata.cast::<CurlBuffer>()) else {
            return 0;
        };
        // SAFETY: libcurl passes a valid buffer with len bytes for the duration of this callback.
        let bytes = unsafe { slice::from_raw_parts(bytes_ptr.as_ptr(), len) };
        // SAFETY: userdata is the CurlBuffer pointer set before curl_easy_perform.
        let target = unsafe { target_ptr.as_mut() };
        if target.bytes.try_reserve(bytes.len()).is_err() {
            return 0;
        }
        target.bytes.extend_from_slice(bytes);
        len
    }
    fn tcp_timeout_secs() -> c_long {
        c_long::try_from(TCP_TIMEOUT.as_secs()).unwrap_or(5)
    }
        }
    }
    target_os = "windows" => {
        mod platform {
    use super::{
        HeadResponse, MIN_TRANSFER_TIME, Result, TCP_TIMEOUT, error, find_date_header_value,
        missing_date,
    };
    use alloc::{string::String, vec::Vec};
    use core::{
        ffi::c_void,
        ptr::{NonNull, null, null_mut},
    };
    use std::{ffi::OsStr, os::windows::ffi::OsStrExt as _, time::Instant};
    const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
    const HTTP_SCHEME_PREFIX: &str = "http://";
    const HTTPS_SCHEME_PREFIX: &str = "https://";
    const INTERNET_DEFAULT_HTTP_PORT: u16 = 80;
    const INTERNET_DEFAULT_HTTPS_PORT: u16 = 443;
    const WINHTTP_ACCESS_TYPE_DEFAULT_PROXY: u32 = 0;
    const WINHTTP_FLAG_SECURE: u32 = 0x0080_0000;
    const WINHTTP_OPTION_IGNORE_CERT_REVOCATION_OFFLINE: u32 = 155;
    const WINHTTP_QUERY_RAW_HEADERS_CRLF: u32 = 22;
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
        fn connect(
            &self,
            session: &Handle,
            host: &[u16],
            port: u16,
            context: &str,
        ) -> Result<Handle> {
            // SAFETY: host is NUL-terminated and session is a valid session handle.
            let raw_connect = unsafe { WinHttpConnect(session.as_ptr(), host.as_ptr(), port, 0) };
            self.non_null_handle(raw_connect, "WinHttpConnect", context)
        }
        pub(super) fn fetch_head(&self, url: &str, context: &str) -> Result<HeadResponse> {
            let target = self.request_target(url, context)?;
            let user_agent = wide(concat!("srg/", env!("CARGO_PKG_VERSION")));
            let host_wide = wide(&target.host);
            let method_wide = wide("HEAD");
            let path_wide = wide("/");
            let session = self.open_session(&user_agent, context)?;
            let connect = self.connect(&session, &host_wide, target.port, context)?;
            let flags = if target.secure {
                WINHTTP_FLAG_SECURE
            } else {
                0
            };
            let request = self.open_request(&connect, &method_wide, &path_wide, flags, context)?;
            if target.secure {
                match self.set_ignore_revocation_offline(&request) {
                    Ok(()) | Err(_) => {}
                }
            }
            let request_start = Instant::now();
            self.send_head(&request, context)?;
            self.receive_response(&request, context)?;
            let response_received_inst = Instant::now();
            let date_header = self.query_date_header(&request, context)?;
            let rtt = response_received_inst
                .saturating_duration_since(request_start)
                .max(MIN_TRANSFER_TIME);
            Ok(HeadResponse {
                date_header,
                response_received_inst,
                rtt,
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
        fn non_null_handle(
            &self,
            handle: HInternet,
            operation: &str,
            context: &str,
        ) -> Result<Handle> {
            NonNull::new(handle)
                .map(Handle)
                .ok_or_else(|| self.last_error(operation, context))
        }
        fn open_request(
            &self,
            connect: &Handle,
            method: &[u16],
            path: &[u16],
            flags: u32,
            context: &str,
        ) -> Result<Handle> {
            // SAFETY: method and path are NUL-terminated and connect is valid.
            let raw_request = unsafe {
                WinHttpOpenRequest(
                    connect.as_ptr(),
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
            let timeout = i32::try_from(TCP_TIMEOUT.as_millis()).unwrap_or(5_000_i32);
            // SAFETY: session is a valid WinHTTP session handle.
            unsafe { WinHttpSetTimeouts(session.as_ptr(), timeout, timeout, timeout, timeout) };
            Ok(session)
        }
        fn query_date_header(&self, request: &Handle, context: &str) -> Result<String> {
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
                return Err(missing_date(context));
            }
            let last_error_code = self.last_error_code();
            if last_error_code != ERROR_INSUFFICIENT_BUFFER {
                return Err(self.last_error("WinHttpQueryHeaders", context));
            }
            let units = usize::try_from(bytes)
                .map_err(|source| error(context, format!("응답 헤더 길이 변환 실패: {source}")))?
                .checked_div(2)
                .ok_or_else(|| error(context, "응답 헤더 길이 계산 실패"))?;
            let mut buffer = vec![0_u16; units];
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
            let raw = String::from_utf16_lossy(&buffer);
            raw.lines()
                .rev()
                .find_map(|line| find_date_header_value(line.as_bytes()).map(str::to_owned))
                .ok_or_else(|| missing_date(context))
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
            let authority_end = rest
                .bytes()
                .position(|byte| matches!(byte, b'/' | b'?' | b'#'))
                .unwrap_or(rest.len());
            let authority = rest.get(..authority_end).unwrap_or(rest);
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
                (host.to_owned(), port)
            } else if let Some((host, port_text)) = authority.rsplit_once(':') {
                if host.contains(':') {
                    (authority.to_owned(), default_port)
                } else {
                    (host.to_owned(), parse_port(port_text)?)
                }
            } else {
                (authority.to_owned(), default_port)
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
    }
    fn wide(value: &str) -> Vec<u16> {
        OsStr::new(value).encode_wide().chain([0]).collect()
    }
        }
    }
    _ => {
        mod platform {
    use super::{HeadResponse, Result, error};
    pub(super) struct Client;
    impl Client {
        pub(super) fn fetch_head(&self, _url: &str, context: &str) -> Result<HeadResponse> {
            Err(error(
                context,
                "이 플랫폼은 외부 크레이트 없이 native HTTP를 지원하지 않습니다.",
            ))
        }
    }
        }
    }
}
pub(super) const NATIVE_HTTP: NativeHttp = NativeHttp {
    parse_http_date: parse_http_date_to_systemtime,
};
struct HeadResponse {
    date_header: String,
    response_received_inst: Instant,
    rtt: Duration,
}
pub(super) struct NativeHttp {
    parse_http_date: fn(&str) -> Result<SystemTime>,
}
impl NativeHttp {
    pub(super) fn fetch_head_sample(&self, url: &str, context: &str) -> Result<TimeSample> {
        let response = platform::CLIENT.fetch_head(url, context)?;
        let server_time = (self.parse_http_date)(&response.date_header)?;
        Ok(TimeSample {
            response_received_inst: response.response_received_inst,
            rtt: response.rtt,
            server_time,
        })
    }
}
fn error(context: &str, detail: impl Into<Cow<'static, str>>) -> TimeError {
    let mut out = String::with_capacity(context.len().saturating_add(2));
    out.push_str(context);
    out.push_str(": ");
    out.push_str(&detail.into());
    TimeError::new(TimeErrorKind::NativeHttp, out)
}
#[cfg(target_os = "windows")]
fn missing_date(context: &str) -> TimeError {
    let mut out = String::with_capacity(context.len().saturating_add(28));
    out.push_str(context);
    out.push_str(" 응답에서 Date");
    TimeError::header_not_found(out)
}
