use super::super::{ParsedServer, TimeError, UrlScheme};
use super::{FreshTimeHeaders, MIN_TRANSFER_TIME, Result, TimeSample, error};
use alloc::{borrow::Cow, string::String, vec::Vec};
use core::{
    ffi::{CStr, c_char, c_long, c_uint, c_void},
    marker::{PhantomData, PhantomPinned},
    ptr::{NonNull, null_mut},
    slice, str,
};
use std::{sync::LazyLock, time::Instant};
mod sys;
macro_rules! curl_setopt {
    ($handle:expr, $option:expr, $value:expr, $context:expr) => {{
        // SAFETY: call sites pair each option with a wrapper using its documented libcurl ABI type.
        let code = unsafe { sys::curl_easy_setopt($handle.as_ptr(), $option, $value) };
        (code == CURLE_OK).ok_or_else(|| error($context, curl_error("curl_easy_setopt", code)))
    }};
}
const AGE_HEADER_NAME: &[u8; 3] = b"age";
const CURLE_OK: CurlCode = 0;
const CURL_ERROR_SIZE: usize = 256;
const CURL_GLOBAL_DEFAULT: c_long = 3;
const CURLINFO_SCHEME: CurlInfo = 0x10_0031;
const CURLOPT_CONNECTTIMEOUT_MS: CurlOption = 156;
const CURLOPT_ERRORBUFFER: CurlOption = 10_010;
const CURLOPT_FOLLOWLOCATION: CurlOption = 52;
const CURLOPT_HEADERDATA: CurlOption = 10_029;
const CURLOPT_HEADERFUNCTION: CurlOption = 20_079;
const CURLOPT_NOBODY: CurlOption = 44;
const CURLOPT_NOPROXY: CurlOption = 10_177;
const CURLOPT_NOSIGNAL: CurlOption = 99;
const CURLOPT_PROTOCOLS_STR: CurlOption = 10_318;
const CURLOPT_SSLVERSION: CurlOption = 32;
const CURLOPT_TIMEOUT_MS: CurlOption = 155;
const CURLOPT_URL: CurlOption = 10_002;
const CURLOPT_USERAGENT: CurlOption = 10_018;
const CURLOPT_WRITEDATA: CurlOption = 10_001;
const CURLOPT_WRITEFUNCTION: CurlOption = 20_011;
const CURL_SSLVERSION_MAX_DEFAULT: c_long = 1 << 16;
const CURL_SSLVERSION_TLSV1_2: c_long = 6;
const DATE_HEADER_NAME: &[u8; 4] = b"date";
const HTTP_PROTOCOL: &CStr = c"http";
const HTTP_USER_AGENT: &CStr = c"Rust-Time-Sync";
const HTTP_HEAD_MAX_BODY_BYTES: usize = 1024 * 1024;
const HTTP_HEAD_MAX_HEADER_BYTES: usize = 1024 * 1024;
const HTTP_HEAD_MAX_PLAIN_HEADER_BYTES: usize = 64 * 1024;
const HTTPS_PROTOCOL: &CStr = c"https";
const HTTPS_USER_AGENT_C_BYTES: &[u8] = concat!("srg/", env!("CARGO_PKG_VERSION"), "\0").as_bytes();
const HTTPS_USER_AGENT: &CStr = {
    // SAFETY: concat! emits exactly one trailing NUL here, and Cargo package versions cannot
    // contain interior NUL bytes.
    unsafe { CStr::from_bytes_with_nul_unchecked(HTTPS_USER_AGENT_C_BYTES) }
};
const NO_PROXY: &CStr = c"*";
const TCP_TIMEOUT_MILLIS: c_long = 5_000;
static CURL_INIT: LazyLock<CurlCode> = LazyLock::new(|| {
    // SAFETY: LazyLock runs this initializer once before any easy handles are used.
    unsafe { sys::curl_global_init(CURL_GLOBAL_DEFAULT) }
});
#[repr(C)]
struct Curl {
    _data: (),
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}
type CurlCode = c_uint;
type CurlInfo = c_uint;
type CurlOption = c_uint;
#[derive(Default)]
pub(in crate::time) struct Client {
    easy_handle: Option<EasyHandle>,
    header_line_buffer: Vec<u8>,
}
struct EasyHandle(NonNull<Curl>);
#[derive(Default)]
struct CurlBodySink {
    bytes_seen: usize,
    error: Option<Cow<'static, str>>,
}
struct CurlHeaderCapture<'line> {
    bytes_seen: usize,
    completed_block: Option<FreshTimeHeaders>,
    current_block: Option<FreshTimeHeaders>,
    error: Option<Cow<'static, str>>,
    limit: usize,
    pending_line: &'line mut Vec<u8>,
}
enum CurlWriteTarget<'target, 'line> {
    Body(&'target mut CurlBodySink),
    Header(&'target mut CurlHeaderCapture<'line>),
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
    fn ensure_scheme(&self, expected_scheme: &CStr, context: &str) -> Result<()> {
        let mut scheme = null_mut::<c_char>();
        // SAFETY: scheme is a valid output pointer for CURLINFO_SCHEME.
        let code =
            unsafe { sys::curl_easy_getinfo(self.as_ptr(), CURLINFO_SCHEME, &raw mut scheme) };
        (code == CURLE_OK)
            .ok_or_else(|| error(context, curl_error("curl_easy_getinfo scheme", code)))?;
        let Some(scheme_ptr) = NonNull::new(scheme) else {
            return Err(error(context, "curl 최종 scheme이 비어 있습니다."));
        };
        // SAFETY: libcurl returns a NUL-terminated scheme string owned by the easy handle.
        let scheme_bytes = unsafe { CStr::from_ptr(scheme_ptr.as_ptr()) }.to_bytes();
        scheme_bytes
            .eq_ignore_ascii_case(expected_scheme.to_bytes())
            .ok_or_else(|| {
                error(
                    context,
                    format!(
                        "curl 최종 scheme이 {}가 아닙니다: {}",
                        expected_scheme.to_string_lossy(),
                        String::from_utf8_lossy(scheme_bytes)
                    ),
                )
            })
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
        value: unsafe extern "C" fn(*mut c_char, usize, usize, *mut c_void) -> usize,
        context: &str,
    ) -> Result<()> {
        curl_setopt!(self, option, value, context)
    }
    fn setopt_long(&self, option: CurlOption, value: c_long, context: &str) -> Result<()> {
        curl_setopt!(self, option, value, context)
    }
    fn setopt_ptr<T>(&self, option: CurlOption, value: *mut T, context: &str) -> Result<()> {
        curl_setopt!(self, option, value, context)
    }
    fn setopt_str(&self, option: CurlOption, value: *const c_char, context: &str) -> Result<()> {
        curl_setopt!(self, option, value, context)
    }
}
impl Client {
    pub(in crate::time) fn fetch_head(
        &mut self,
        server: &ParsedServer,
        context: &str,
    ) -> Result<TimeSample> {
        let mut error_buffer = [c_char::default(); CURL_ERROR_SIZE];
        let mut body_sink = CurlBodySink::default();
        self.header_line_buffer.clear();
        let mut header_capture = CurlHeaderCapture {
            bytes_seen: 0,
            completed_block: None,
            current_block: None,
            error: None,
            limit: match server.scheme {
                UrlScheme::Http => HTTP_HEAD_MAX_PLAIN_HEADER_BYTES,
                UrlScheme::Https => HTTP_HEAD_MAX_HEADER_BYTES,
            },
            pending_line: &mut self.header_line_buffer,
        };
        let init_code = *CURL_INIT;
        (init_code == CURLE_OK)
            .ok_or_else(|| error(context, curl_error("curl_global_init", init_code)))?;
        let handle = match &mut self.easy_handle {
            &mut Some(ref mut handle) => handle,
            empty @ &mut None => {
                // SAFETY: curl_easy_init has no preconditions after global init.
                let raw_handle = NonNull::new(unsafe { sys::curl_easy_init() })
                    .ok_or_else(|| error(context, "curl_easy_init 실패"))?;
                empty.insert(EasyHandle(raw_handle))
            }
        };
        handle.reset();
        handle.setopt_callback(CURLOPT_WRITEFUNCTION, write_callback, context)?;
        handle.setopt_callback(CURLOPT_HEADERFUNCTION, write_callback, context)?;
        handle.setopt_str(CURLOPT_URL, server.request_target.as_ptr(), context)?;
        handle.setopt_str(
            CURLOPT_USERAGENT,
            match server.scheme {
                UrlScheme::Http => HTTP_USER_AGENT,
                UrlScheme::Https => HTTPS_USER_AGENT,
            }
            .as_ptr(),
            context,
        )?;
        handle.setopt_ptr(CURLOPT_ERRORBUFFER, error_buffer.as_mut_ptr(), context)?;
        handle.setopt_long(CURLOPT_CONNECTTIMEOUT_MS, TCP_TIMEOUT_MILLIS, context)?;
        handle.setopt_long(CURLOPT_TIMEOUT_MS, TCP_TIMEOUT_MILLIS, context)?;
        handle.setopt_long(CURLOPT_NOSIGNAL, 1, context)?;
        handle.setopt_long(CURLOPT_NOBODY, 1, context)?;
        handle.setopt_long(CURLOPT_FOLLOWLOCATION, 0, context)?;
        let protocol = match server.scheme {
            UrlScheme::Http => {
                handle.setopt_str(CURLOPT_NOPROXY, NO_PROXY.as_ptr(), context)?;
                HTTP_PROTOCOL
            }
            UrlScheme::Https => {
                handle.setopt_long(
                    CURLOPT_SSLVERSION,
                    CURL_SSLVERSION_TLSV1_2 | CURL_SSLVERSION_MAX_DEFAULT,
                    context,
                )?;
                HTTPS_PROTOCOL
            }
        };
        handle.setopt_str(CURLOPT_PROTOCOLS_STR, protocol.as_ptr(), context)?;
        let (perform_code, request_start) = {
            let mut body_target = CurlWriteTarget::Body(&mut body_sink);
            let mut header_target = CurlWriteTarget::Header(&mut header_capture);
            let body_data = (&raw mut body_target).cast::<c_void>();
            let header_data = (&raw mut header_target).cast::<c_void>();
            handle.setopt_ptr(CURLOPT_WRITEDATA, body_data, context)?;
            handle.setopt_ptr(CURLOPT_HEADERDATA, header_data, context)?;
            let request_start = Instant::now();
            let code = handle.perform();
            (code, request_start)
        };
        if !header_capture.pending_line.is_empty() {
            header_capture.capture_pending();
            header_capture.pending_line.clear();
        }
        if let Some(callback_error) = body_sink.error.or(header_capture.error) {
            self.easy_handle = None;
            return Err(error(context, callback_error));
        }
        if perform_code != CURLE_OK {
            let error_bytes = error_buffer.map(|ch| ch.to_le_bytes()[0]);
            let perform_error: Cow<'static, str> = if let Ok(message_cstr) =
                CStr::from_bytes_until_nul(&error_bytes)
                && !message_cstr.is_empty()
            {
                Cow::Owned(format!(
                    "curl_easy_perform 실패: {} ({})",
                    message_cstr.to_string_lossy(),
                    perform_code
                ))
            } else {
                Cow::Owned(curl_error("curl_easy_perform", perform_code))
            };
            self.easy_handle = None;
            return Err(error(context, perform_error));
        }
        if let Err(scheme_error) = handle.ensure_scheme(protocol, context) {
            self.easy_handle = None;
            return Err(scheme_error);
        }
        let header_block = header_capture
            .completed_block
            .ok_or_else(|| error(context, "완료된 HTTP 응답 header block을 찾지 못했습니다."))?;
        let (server_time, response_received_inst) = header_block.finish(context)?;
        let http_elapsed = response_received_inst
            .checked_duration_since(request_start)
            .ok_or_else(|| TimeError::parse("HTTP 응답 시각이 요청 시작 시각보다 앞섭니다."))?
            .max(MIN_TRANSFER_TIME);
        Ok(TimeSample {
            response_received_inst,
            rtt: http_elapsed,
            server_time,
        })
    }
}
impl CurlBodySink {
    fn append(&mut self, bytes: &[u8]) -> bool {
        let Some(next_len) = self.bytes_seen.checked_add(bytes.len()) else {
            self.error = Some(Cow::Borrowed("HTTP HEAD 응답 본문 크기 계산 실패"));
            return false;
        };
        if next_len > HTTP_HEAD_MAX_BODY_BYTES {
            self.error = Some(Cow::Owned(format!(
                "HTTP HEAD 응답 본문 크기가 허용 한도({HTTP_HEAD_MAX_BODY_BYTES} bytes)를 초과했습니다."
            )));
            return false;
        }
        self.bytes_seen = next_len;
        true
    }
}
impl CurlHeaderCapture<'_> {
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
            && self.pending_line.try_reserve(bytes.len()).is_err()
        {
            self.error = Some(Cow::Borrowed("HTTP HEAD 응답 헤더 메모리 확보 실패"));
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
            self.current_block = Some(FreshTimeHeaders::default());
            return true;
        }
        if line.is_empty() {
            if let Some(block) = self.current_block.take() {
                self.completed_block = Some(block);
            }
            return true;
        }
        let Some(current_block) = self.current_block.as_mut() else {
            return true;
        };
        let Some(colon) = line.iter().position(|byte| *byte == b':') else {
            return true;
        };
        let (name, value_with_colon) = line.split_at(colon);
        let is_age_header = match name.len() {
            3 if name.eq_ignore_ascii_case(AGE_HEADER_NAME) => true,
            4 if name.eq_ignore_ascii_case(DATE_HEADER_NAME) => false,
            _ => return true,
        };
        let Some((_, value_bytes)) = value_with_colon.split_first() else {
            return true;
        };
        let header_raw = match str::from_utf8(value_bytes).map(str::trim_ascii) {
            Ok(header_raw) => header_raw,
            Err(source) => {
                self.error = Some(Cow::Owned(format!(
                    "HTTP HEAD 응답 헤더 UTF-8 변환 실패: {source}"
                )));
                return false;
            }
        };
        if is_age_header {
            current_block.capture_age(header_raw);
        } else {
            current_block.capture_date(header_raw, Instant::now());
        }
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
    let Some(mut target_ptr) = NonNull::new(userdata.cast::<CurlWriteTarget<'_, '_>>()) else {
        return 0;
    };
    // SAFETY: len is non-zero, payload_head is non-null, and libcurl passes a readable buffer with
    // len bytes for this callback.
    let bytes = unsafe { slice::from_raw_parts(payload_head.as_ptr(), len) };
    // SAFETY: userdata is the CurlWriteTarget pointer configured before curl_easy_perform.
    let target = unsafe { target_ptr.as_mut() };
    if !match *target {
        CurlWriteTarget::Body(ref mut buffer) => (*buffer).append(bytes),
        CurlWriteTarget::Header(ref mut capture) => (*capture).append(bytes),
    } {
        return 0;
    }
    len
}
