use super::{
    HeadResponse, MIN_TRANSFER_TIME, ParseHttpDate, Result, TCP_TIMEOUT, error,
};
use super::super::{TimeError, TimeErrorKind};
use super::super::sample::{
    AGE_HEADER_PREFIX, DATE_HEADER_PREFIX, find_header_value, validate_http_age_value,
};
use alloc::{borrow::Cow, string::String, vec::Vec};
use core::{
    ffi::{CStr, c_char, c_int, c_long, c_uint, c_void},
    mem::{self, align_of, offset_of, size_of},
    ptr::{NonNull, null},
    slice,
};
use std::{sync::LazyLock, time::Instant};
mod sys {
    use super::{
        Curl, CurlCode, CurlInfo, CurlOption, CurlVersion, CurlVersionInfoData, c_char, c_long,
    };
    #[link(name = "curl")]
    unsafe extern "C" {
        pub fn curl_easy_cleanup(curl: *mut Curl);
        pub fn curl_easy_getinfo(curl: *mut Curl, info: CurlInfo, ...) -> CurlCode;
        pub fn curl_easy_init() -> *mut Curl;
        pub fn curl_easy_perform(curl: *mut Curl) -> CurlCode;
        pub fn curl_easy_reset(curl: *mut Curl);
        pub fn curl_easy_setopt(curl: *mut Curl, option: CurlOption, ...) -> CurlCode;
        pub fn curl_easy_strerror(code: CurlCode) -> *const c_char;
        pub fn curl_global_init(flags: c_long) -> CurlCode;
        pub fn curl_version_info(age: CurlVersion) -> *const CurlVersionInfoData;
    }
}
const CURLE_OK: CurlCode = 0;
const CURL_ERROR_SIZE: usize = 256;
const CURL_GLOBAL_DEFAULT: c_long = 3;
const CURLINFO_SCHEME: CurlInfo = 0x10_0031;
const CURL_MIN_PROTOCOLS_STR_VERSION: c_uint = 0x07_55_00;
const CURLVERSION_NOW: CurlVersion = 11;
const CURLOPT_CONNECTTIMEOUT_MS: CurlOption = 156;
const CURLOPT_ERRORBUFFER: CurlOption = 10_010;
const CURLOPT_FOLLOWLOCATION: CurlOption = 52;
const CURLOPT_HEADERDATA: CurlOption = 10_029;
const CURLOPT_HEADERFUNCTION: CurlOption = 20_079;
const CURLOPT_MAXREDIRS: CurlOption = 68;
const CURLOPT_NOBODY: CurlOption = 44;
const CURLOPT_NOSIGNAL: CurlOption = 99;
const CURLOPT_PROTOCOLS_STR: CurlOption = 10_318;
const CURLOPT_REDIR_PROTOCOLS_STR: CurlOption = 10_319;
const CURLOPT_SSLVERSION: CurlOption = 32;
const CURLOPT_TIMEOUT_MS: CurlOption = 155;
const CURLOPT_URL: CurlOption = 10_002;
const CURLOPT_USERAGENT: CurlOption = 10_018;
const CURLOPT_WRITEDATA: CurlOption = 10_001;
const CURLOPT_WRITEFUNCTION: CurlOption = 20_011;
const CURL_SSLVERSION_MAX_DEFAULT: c_long = 1 << 16;
const CURL_SSLVERSION_TLSV1_2: c_long = 6;
const CURL_PROTOCOLS_STR_UNSUPPORTED_SUFFIX: &str =
    "은 HTTPS protocol 제한 최신 API를 지원하지 않습니다. libcurl 7.85.0 이상이 필요합니다.";
const HTTP_HEAD_MAX_BODY_BYTES: usize = 1024 * 1024;
const HTTP_HEAD_MAX_HEADER_BYTES: usize = 1024 * 1024;
const HTTPS_PROTOCOL: &CStr = c"https";
const MAX_HTTP_REDIRECTS: c_long = 5;
const USER_AGENT_C_BYTES: &[u8] = concat!("srg/", env!("CARGO_PKG_VERSION"), "\0").as_bytes();
const USER_AGENT: &CStr = {
    // SAFETY: concat! emits exactly one trailing NUL here, and Cargo package versions cannot
    // contain interior NUL bytes.
    unsafe { CStr::from_bytes_with_nul_unchecked(USER_AGENT_C_BYTES) }
};
static CURL_INIT: LazyLock<CurlCode> = LazyLock::new(|| {
    // SAFETY: LazyLock runs this initializer once before any easy handles are used.
    unsafe { sys::curl_global_init(CURL_GLOBAL_DEFAULT) }
});
static CURL_PROTOCOLS_STR_UNSUPPORTED_VERSION: LazyLock<Option<Cow<'static, str>>> =
    LazyLock::new(|| {
        // SAFETY: callers force this after curl_global_init has completed, and libcurl returns a
        // process-wide immutable version info pointer.
        NonNull::new(unsafe { sys::curl_version_info(CURLVERSION_NOW).cast_mut() }).map_or(
            Some(Cow::Borrowed("unknown")),
            |version_info| {
                // SAFETY: version_info is non-null and points to libcurl's version info.
                let version_info_ref = unsafe { version_info.as_ref() };
                if version_info_ref.version_num >= CURL_MIN_PROTOCOLS_STR_VERSION {
                    None
                } else {
                    Some(NonNull::new(version_info_ref.version.cast_mut()).map_or_else(
                        || Cow::Borrowed("unknown"),
                        |version_ptr| {
                            // SAFETY: libcurl documents version as an ASCII NUL-terminated string.
                            Cow::Owned(
                                unsafe { CStr::from_ptr(version_ptr.as_ptr()) }
                                    .to_string_lossy()
                                    .into_owned(),
                            )
                        },
                    ))
                }
            },
        )
    });
type Curl = c_void;
type CurlCode = c_int;
type CurlInfo = c_int;
type CurlOption = c_int;
type CurlVersion = c_int;
type CurlWriteCallback = unsafe extern "C" fn(*mut c_char, usize, usize, *mut c_void) -> usize;
#[repr(C)]
struct CurlVersionInfoData {
    age: CurlVersion,
    version: *const c_char,
    version_num: c_uint,
}
cfg_select! {
    target_pointer_width = "64" => {
        const _: () = assert!(
            size_of::<CurlVersionInfoData>() == 24,
            "libcurl version info prefix x64 size mismatch"
        );
        const _: () = assert!(
            align_of::<CurlVersionInfoData>() == 8,
            "libcurl version info prefix x64 align mismatch"
        );
        const _: () = assert!(
            offset_of!(CurlVersionInfoData, age) == 0,
            "libcurl version info prefix x64 age offset mismatch"
        );
        const _: () = assert!(
            offset_of!(CurlVersionInfoData, version) == 8,
            "libcurl version info prefix x64 version offset mismatch"
        );
        const _: () = assert!(
            offset_of!(CurlVersionInfoData, version_num) == 16,
            "libcurl version info prefix x64 version number offset mismatch"
        );
    }
    target_pointer_width = "32" => {
        const _: () = assert!(
            size_of::<CurlVersionInfoData>() == 12,
            "libcurl version info prefix x86 size mismatch"
        );
        const _: () = assert!(
            align_of::<CurlVersionInfoData>() == 4,
            "libcurl version info prefix x86 align mismatch"
        );
        const _: () = assert!(
            offset_of!(CurlVersionInfoData, age) == 0,
            "libcurl version info prefix x86 age offset mismatch"
        );
        const _: () = assert!(
            offset_of!(CurlVersionInfoData, version) == 4,
            "libcurl version info prefix x86 version offset mismatch"
        );
        const _: () = assert!(
            offset_of!(CurlVersionInfoData, version_num) == 8,
            "libcurl version info prefix x86 version number offset mismatch"
        );
    }
    _ => {}
}
#[derive(Default)]
pub(super) struct Client {
    easy_handle: Option<EasyHandle>,
    url_buffer: Vec<u8>,
}
struct EasyHandle(NonNull<Curl>);
#[derive(Default)]
struct CurlBodySink {
    bytes_seen: usize,
    error: Option<Cow<'static, str>>,
    limit: usize,
}
#[derive(Default)]
struct CurlHeaderCapture {
    age_error: Option<Cow<'static, str>>,
    bytes_seen: usize,
    current_block_age_error: Option<Cow<'static, str>>,
    current_block_age_seen: bool,
    current_block_date: Option<String>,
    current_block_date_error: Option<Cow<'static, str>>,
    current_block_date_received_inst: Option<Instant>,
    current_block_date_seen: bool,
    date_error: Option<Cow<'static, str>>,
    date_header: Option<String>,
    date_received_inst: Option<Instant>,
    error: Option<Cow<'static, str>>,
    in_header_block: bool,
    limit: usize,
    pending_line: Vec<u8>,
}
struct CurlHeadPerform {
    code: CurlCode,
    request_start: Instant,
    response_received: Instant,
}
enum CurlWriteTarget<'buffer> {
    Body(&'buffer mut CurlBodySink),
    Header(&'buffer mut CurlHeaderCapture),
}
impl Drop for EasyHandle {
    fn drop(&mut self) {
        // SAFETY: self.0 is an easy handle returned by libcurl and is closed exactly once here.
        unsafe {
            sys::curl_easy_cleanup(self.0.as_ptr());
        }
    }
}
impl EasyHandle {
    const fn as_ptr(&self) -> *mut Curl {
        self.0.as_ptr()
    }
    fn configure_head_request(
        &self,
        url: &CStr,
        user_agent: &CStr,
        error_buffer: &mut [c_char; CURL_ERROR_SIZE],
        context: &str,
    ) -> Result<()> {
        self.setopt_callback(CURLOPT_WRITEFUNCTION, write_callback, context)?;
        self.setopt_callback(CURLOPT_HEADERFUNCTION, write_callback, context)?;
        self.setopt_str(CURLOPT_URL, url.as_ptr(), context)?;
        self.setopt_str(CURLOPT_USERAGENT, user_agent.as_ptr(), context)?;
        self.setopt_ptr(CURLOPT_ERRORBUFFER, error_buffer.as_mut_ptr(), context)?;
        self.setopt_long(CURLOPT_CONNECTTIMEOUT_MS, tcp_timeout_millis(), context)?;
        self.setopt_long(CURLOPT_TIMEOUT_MS, tcp_timeout_millis(), context)?;
        self.setopt_long(CURLOPT_NOSIGNAL, 1, context)?;
        self.setopt_long(CURLOPT_NOBODY, 1, context)?;
        self.setopt_long(CURLOPT_FOLLOWLOCATION, 1, context)?;
        self.setopt_long(CURLOPT_MAXREDIRS, MAX_HTTP_REDIRECTS, context)?;
        self.setopt_long(
            CURLOPT_SSLVERSION,
            CURL_SSLVERSION_TLSV1_2 | CURL_SSLVERSION_MAX_DEFAULT,
            context,
        )?;
        self.setopt_str(CURLOPT_PROTOCOLS_STR, HTTPS_PROTOCOL.as_ptr(), context)?;
        self.setopt_str(CURLOPT_REDIR_PROTOCOLS_STR, HTTPS_PROTOCOL.as_ptr(), context)?;
        Ok(())
    }
    fn ensure_https_scheme(&self, context: &str) -> Result<()> {
        let mut scheme = null::<c_char>();
        // SAFETY: scheme is a valid output pointer for CURLINFO_SCHEME.
        let code = unsafe { sys::curl_easy_getinfo(self.as_ptr(), CURLINFO_SCHEME, &raw mut scheme) };
        if code != CURLE_OK {
            return Err(error(context, curl_error("curl_easy_getinfo scheme", code)));
        }
        let Some(scheme_ptr) = NonNull::new(scheme.cast_mut()) else {
            return Err(error(context, "curl 최종 scheme이 비어 있습니다."));
        };
        // SAFETY: libcurl returns a NUL-terminated scheme string owned by the easy handle.
        let scheme_bytes = unsafe { CStr::from_ptr(scheme_ptr.as_ptr()) }.to_bytes();
        if scheme_bytes.eq_ignore_ascii_case(b"https") {
            Ok(())
        } else {
            Err(error(
                context,
                format!(
                    "curl 최종 scheme이 HTTPS가 아닙니다: {}",
                    String::from_utf8_lossy(scheme_bytes)
                ),
            ))
        }
    }
    fn perform(&self) -> CurlCode {
        // SAFETY: self.0 is configured with callbacks and buffers that live until the call returns.
        unsafe { sys::curl_easy_perform(self.as_ptr()) }
    }
    fn reset(&self) {
        // SAFETY: self.0 is a valid easy handle; reset clears options while keeping libcurl caches.
        unsafe {
            sys::curl_easy_reset(self.as_ptr());
        }
    }
    fn setopt_callback(
        &self,
        option: CurlOption,
        value: CurlWriteCallback,
        context: &str,
    ) -> Result<()> {
        // SAFETY: value is a libcurl-compatible callback function pointer.
        let code = unsafe { sys::curl_easy_setopt(self.as_ptr(), option, value) };
        if code == CURLE_OK {
            Ok(())
        } else {
            Err(error(context, curl_error("curl_easy_setopt", code)))
        }
    }
    fn setopt_long(&self, option: CurlOption, value: c_long, context: &str) -> Result<()> {
        // SAFETY: value is a scalar option value for the given CurlOption.
        let code = unsafe { sys::curl_easy_setopt(self.as_ptr(), option, value) };
        if code == CURLE_OK {
            Ok(())
        } else {
            Err(error(context, curl_error("curl_easy_setopt", code)))
        }
    }
    fn setopt_ptr<T>(&self, option: CurlOption, value: *mut T, context: &str) -> Result<()> {
        // SAFETY: value is a pointer option that remains valid for the transfer duration.
        let code = unsafe { sys::curl_easy_setopt(self.as_ptr(), option, value) };
        if code == CURLE_OK {
            Ok(())
        } else {
            Err(error(context, curl_error("curl_easy_setopt", code)))
        }
    }
    fn setopt_str(&self, option: CurlOption, value: *const c_char, context: &str) -> Result<()> {
        // SAFETY: value is a valid NUL-terminated string that outlives the setopt call.
        let code = unsafe { sys::curl_easy_setopt(self.as_ptr(), option, value) };
        if code == CURLE_OK {
            Ok(())
        } else {
            Err(error(context, curl_error("curl_easy_setopt", code)))
        }
    }
}
impl Client {
    fn clear_reusable_handle(&mut self) {
        self.easy_handle = None;
    }
    pub(super) fn fetch_head(
        &mut self,
        url: &str,
        context: &str,
        parse_http_date: ParseHttpDate,
    ) -> Result<HeadResponse> {
        let mut url_buffer = mem::take(&mut self.url_buffer);
        let result = (|| {
            let url_capacity = url
                .len()
                .checked_add(1)
                .ok_or_else(|| error(context, "URL 용량 계산 실패"))?;
            url_buffer.clear();
            if url_buffer.capacity() < url_capacity {
                url_buffer.try_reserve_exact(url_capacity).map_err(|source| {
                    TimeError::new_with_source(
                        TimeErrorKind::NativeHttp,
                        format!("{context}: URL 메모리 확보 실패"),
                        source,
                    )
                })?;
            }
            url_buffer.extend_from_slice(url.as_bytes());
            url_buffer.push(0);
            let url_c = CStr::from_bytes_with_nul(&url_buffer).map_err(|source| {
                TimeError::new_with_source(
                    TimeErrorKind::NativeHttp,
                    format!("{context}: URL에 NUL 문자가 포함되어 있습니다"),
                    source,
                )
            })?;
            self.fetch_head_curl(url_c, context, parse_http_date)
        })();
        self.url_buffer = url_buffer;
        result
    }
    fn fetch_head_curl(
        &mut self,
        url_c: &CStr,
        context: &str,
        parse_http_date: ParseHttpDate,
    ) -> Result<HeadResponse> {
        let mut error_buffer = [c_char::default(); CURL_ERROR_SIZE];
        let mut body_sink = CurlBodySink {
            limit: HTTP_HEAD_MAX_BODY_BYTES,
            ..CurlBodySink::default()
        };
        let mut header_capture = CurlHeaderCapture {
            limit: HTTP_HEAD_MAX_HEADER_BYTES,
            ..CurlHeaderCapture::default()
        };
        let init_code = *CURL_INIT;
        if init_code != CURLE_OK {
            return Err(error(context, curl_error("curl_global_init", init_code)));
        }
        if self.easy_handle.is_none() {
            if let Some(version) = CURL_PROTOCOLS_STR_UNSUPPORTED_VERSION.as_ref() {
                return Err(error(
                    context,
                    format!("libcurl {version}{CURL_PROTOCOLS_STR_UNSUPPORTED_SUFFIX}"),
                ));
            }
            // SAFETY: curl_easy_init has no preconditions after global init.
            let raw_handle = NonNull::new(unsafe { sys::curl_easy_init() })
                .ok_or_else(|| error(context, "curl_easy_init 실패"))?;
            self.easy_handle = Some(EasyHandle(raw_handle));
        }
        let handle = self
            .easy_handle
            .as_ref()
            .ok_or_else(|| error(context, "curl easy handle cache 상태 오류"))?;
        handle.reset();
        handle.configure_head_request(url_c, USER_AGENT, &mut error_buffer, context)?;
        let perform_result = {
            let mut body_target = CurlWriteTarget::Body(&mut body_sink);
            let mut header_target = CurlWriteTarget::Header(&mut header_capture);
            let body_data = (&raw mut body_target).cast::<c_void>();
            let header_data = (&raw mut header_target).cast::<c_void>();
            handle.setopt_ptr(CURLOPT_WRITEDATA, body_data, context)?;
            handle.setopt_ptr(CURLOPT_HEADERDATA, header_data, context)?;
            let request_start = Instant::now();
            let code = handle.perform();
            CurlHeadPerform {
                code,
                request_start,
                response_received: Instant::now(),
            }
        };
        if !header_capture.pending_line.is_empty() {
            header_capture.capture_pending();
            header_capture.pending_line.clear();
        }
        if let Some(callback_error) = body_sink.error.take().or_else(|| header_capture.error.take()) {
            self.clear_reusable_handle();
            return Err(error(context, callback_error));
        }
        if perform_result.code != CURLE_OK {
            let error_bytes = error_buffer.map(|ch| ch.to_le_bytes()[0]);
            let perform_error: Cow<'static, str> =
                if let Ok(message_cstr) = CStr::from_bytes_until_nul(&error_bytes)
                    && !message_cstr.to_bytes().is_empty()
                {
                    Cow::Owned(format!(
                        "curl_easy_perform 실패: {} ({})",
                        message_cstr.to_string_lossy(),
                        perform_result.code
                    ))
                } else {
                    Cow::Owned(curl_error("curl_easy_perform", perform_result.code))
                };
            self.clear_reusable_handle();
            return Err(error(context, perform_error));
        }
        if let Err(scheme_error) = handle.ensure_https_scheme(context) {
            self.clear_reusable_handle();
            return Err(scheme_error);
        }
        if let Some(header_error) = header_capture
            .age_error
            .take()
            .or_else(|| header_capture.date_error.take())
        {
            return Err(error(context, header_error));
        }
        let Some(date_header_raw) = header_capture.date_header.as_deref() else {
            return Err(super::TimeError::header_not_found(format!("{context} 응답에서 Date")));
        };
        let response_received_inst = header_capture.date_received_inst.unwrap_or(perform_result.response_received);
        let http_elapsed = response_received_inst
            .saturating_duration_since(perform_result.request_start)
            .max(MIN_TRANSFER_TIME);
        Ok(HeadResponse {
            response_received_inst,
            rtt: http_elapsed,
            server_time: parse_http_date(date_header_raw)?,
        })
    }
}
impl CurlBodySink {
    fn append(&mut self, bytes: &[u8]) -> bool {
        let Some(next_len) = self.bytes_seen.checked_add(bytes.len()) else {
            self.error = Some(Cow::Borrowed("HTTP HEAD 응답 본문 크기 계산 실패"));
            return false;
        };
        if next_len > self.limit {
            self.error = Some(Cow::Owned(format!(
                "HTTP HEAD 응답 본문 크기가 허용 한도({} bytes)를 초과했습니다.",
                self.limit
            )));
            return false;
        }
        self.bytes_seen = next_len;
        true
    }
}
impl CurlHeaderCapture {
    fn append(&mut self, bytes: &[u8]) -> bool {
        let Some(next_len) = self.bytes_seen.checked_add(bytes.len()) else {
            self.error = Some(Cow::Borrowed("HTTP HEAD 응답 헤더 크기 계산 실패"));
            return false;
        };
        if next_len > self.limit {
            self.error = Some(Cow::Owned(format!(
                "HTTP HEAD 응답 헤더 크기가 허용 한도({} bytes)를 초과했습니다.",
                self.limit
            )));
            return false;
        }
        self.bytes_seen = next_len;
        let Some(pending_capacity) = self.pending_line.len().checked_add(bytes.len()) else {
            self.error = Some(Cow::Borrowed("HTTP HEAD 응답 헤더 line 길이 계산 실패"));
            return false;
        };
        if self.pending_line.capacity() < pending_capacity
            && let Err(source) = self.pending_line.try_reserve(bytes.len())
        {
            self.error = Some(Cow::Owned(format!(
                "HTTP HEAD 응답 헤더 메모리 확보 실패: {source}"
            )));
            return false;
        }
        for segment in bytes.split_inclusive(|byte| *byte == b'\n') {
            self.pending_line.extend_from_slice(segment);
            if segment.ends_with(b"\n") {
                if !self.capture_pending() {
                    return false;
                }
                self.pending_line.clear();
            }
        }
        true
    }
    fn capture_pending(&mut self) -> bool {
        let without_lf = self
            .pending_line
            .strip_suffix(b"\n")
            .unwrap_or(self.pending_line.as_slice());
        let line = without_lf.strip_suffix(b"\r").unwrap_or(without_lf);
        if line.starts_with(b"HTTP/") {
            self.current_block_age_error = None;
            self.current_block_age_seen = false;
            self.current_block_date = None;
            self.current_block_date_error = None;
            self.current_block_date_received_inst = None;
            self.current_block_date_seen = false;
            self.in_header_block = true;
            return true;
        }
        if line.is_empty() {
            if self.in_header_block {
                self.age_error = self.current_block_age_error.take();
                self.current_block_age_seen = false;
                self.date_error = self.current_block_date_error.take();
                self.current_block_date_seen = false;
                self.date_header = self.current_block_date.take();
                self.date_received_inst = self.current_block_date_received_inst.take();
                self.in_header_block = false;
            }
            return true;
        }
        if !self.in_header_block {
            return true;
        }
        match find_header_value(line, AGE_HEADER_PREFIX) {
            Ok(Some(age_header_raw)) => {
                if self.current_block_age_seen {
                    self.current_block_age_error = Some(Cow::Borrowed("Age 헤더가 여러 개입니다."));
                } else {
                    self.current_block_age_seen = true;
                    if let Err(message) = validate_http_age_value(age_header_raw) {
                        self.current_block_age_error = Some(Cow::Borrowed(message));
                    }
                }
                return true;
            }
            Ok(None) => {}
            Err(source) => {
                self.current_block_age_error = Some(Cow::Owned(format!(
                    "HTTP HEAD 응답 Age 헤더 UTF-8 변환 실패: {source}"
                )));
                return true;
            }
        }
        let date_header_raw = match find_header_value(line, DATE_HEADER_PREFIX) {
            Ok(Some(value)) => value,
            Ok(None) => return true,
            Err(source) => {
                self.error = Some(Cow::Owned(format!(
                    "HTTP HEAD 응답 Date 헤더 UTF-8 변환 실패: {source}"
                )));
                return false;
            }
        };
        if self.current_block_date_seen {
            self.current_block_date_error = Some(Cow::Borrowed("Date 헤더가 여러 개입니다."));
            return true;
        }
        self.current_block_date_seen = true;
        let mut date_header = String::new();
        if let Err(source) = date_header.try_reserve_exact(date_header_raw.len()) {
            self.error = Some(Cow::Owned(format!(
                "HTTP HEAD 응답 Date 헤더 메모리 확보 실패: {source}"
            )));
            return false;
        }
        date_header.push_str(date_header_raw);
        self.current_block_date = Some(date_header);
        self.current_block_date_received_inst = Some(Instant::now());
        true
    }
}
fn curl_error(context: &str, code: CurlCode) -> String {
    // SAFETY: curl_easy_strerror returns either null or a static NUL-terminated message for code.
    let raw_ptr = unsafe { sys::curl_easy_strerror(code) };
    let message = if raw_ptr.is_null() {
        Cow::Borrowed("unknown curl error")
    } else {
        // SAFETY: libcurl guarantees a valid NUL-terminated string for non-null strerror results.
        unsafe { CStr::from_ptr(raw_ptr) }.to_string_lossy()
    };
    format!("{context} 실패: {message} ({code})")
}
unsafe extern "C" fn write_callback(
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
    let Some(payload_head) = NonNull::new(ptr.cast::<u8>()) else {
        return 0;
    };
    let Some(mut target_ptr) = NonNull::new(userdata.cast::<CurlWriteTarget<'_>>()) else {
        return 0;
    };
    // SAFETY: len is non-zero, payload_head is non-null, and libcurl passes a readable buffer with
    // len bytes for this callback.
    let bytes = unsafe { slice::from_raw_parts(payload_head.as_ptr(), len) };
    // SAFETY: userdata is the CurlWriteTarget pointer configured before curl_easy_perform.
    let target = unsafe { target_ptr.as_mut() };
    let accepted = match *target {
        CurlWriteTarget::Body(ref mut buffer) => (*buffer).append(bytes),
        CurlWriteTarget::Header(ref mut capture) => (*capture).append(bytes),
    };
    if !accepted {
        return 0;
    }
    len
}
fn tcp_timeout_millis() -> c_long {
    let Ok(millis) = c_long::try_from(TCP_TIMEOUT.as_millis()) else {
        return 5_000;
    };
    millis
}
