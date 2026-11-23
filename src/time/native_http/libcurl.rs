use super::super::TimeError;
use super::super::{ParsedServer, UrlScheme};
use super::{FreshTimeHeaders, Result, TimeSample, error};
use alloc::{borrow::Cow, string::String};
use core::{
    ffi::{CStr, c_char, c_long, c_uint, c_void},
    marker::{PhantomData, PhantomPinned},
    mem::DropGuard,
    ptr::{NonNull, null_mut},
    slice, str,
};
use std::{process, sync::LazyLock, time::Instant};
mod sys;
macro_rules! curl_setopt {
    ($handle:expr, $option:expr, $value:expr, $context:expr) => {{
        // SAFETY: call sites pair each option with a wrapper using its documented libcurl ABI type.
        let code = unsafe { sys::curl_easy_setopt($handle.as_ptr(), $option, $value) };
        (code == CURLE_OK).ok_or_else(|| setopt_error($context, code))
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
const HTTPS_USER_AGENT: &CStr = match CStr::from_bytes_with_nul(HTTPS_USER_AGENT_C_BYTES) {
    Ok(user_agent) => user_agent,
    Err(_) => c"srg",
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
}
struct EasyHandle(NonNull<Curl>);
#[derive(Default)]
struct CurlBodySink {
    bytes_seen: usize,
    error: Option<Cow<'static, str>>,
}
struct CurlHeaderCapture {
    bytes_seen: usize,
    completed_block: Option<FreshTimeHeaders>,
    current_block: Option<FreshTimeHeaders>,
    error: Option<Cow<'static, str>>,
    limit: usize,
}
enum CurlWriteTarget<'target> {
    Body(&'target mut CurlBodySink),
    Header(&'target mut CurlHeaderCapture),
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
            .ok_or_else(|| error(context, &curl_error("curl_easy_getinfo scheme", code)))?;
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
                    &format!(
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
        let mut error_buffer = [0_u8; CURL_ERROR_SIZE];
        let mut body_sink = CurlBodySink::default();
        let mut header_capture = CurlHeaderCapture {
            bytes_seen: 0,
            completed_block: None,
            current_block: None,
            error: None,
            limit: match server.scheme {
                UrlScheme::Http => HTTP_HEAD_MAX_PLAIN_HEADER_BYTES,
                UrlScheme::Https => HTTP_HEAD_MAX_HEADER_BYTES,
            },
        };
        let init_code = *CURL_INIT;
        (init_code == CURLE_OK)
            .ok_or_else(|| error(context, &curl_error("curl_global_init", init_code)))?;
        let handle = match &mut self.easy_handle {
            &mut Some(ref mut handle) => handle,
            empty @ &mut None => {
                // SAFETY: curl_easy_init has no preconditions after global init.
                let raw_handle = NonNull::new(unsafe { sys::curl_easy_init() })
                    .ok_or_else(|| error(context, "curl_easy_init 실패"))?;
                empty.insert(EasyHandle(raw_handle))
            }
        };
        let (perform_code, request_start, scheme_result) = {
            let mut body_target = CurlWriteTarget::Body(&mut body_sink);
            let mut header_target = CurlWriteTarget::Header(&mut header_capture);
            let _options = DropGuard::new(&*handle, |reset_handle| {
                // SAFETY: reset_handle is valid; reset clears options while keeping libcurl caches.
                unsafe {
                    sys::curl_easy_reset(reset_handle.as_ptr());
                }
            });
            handle.setopt_callback(CURLOPT_WRITEFUNCTION, write_callback, context)?;
            handle.setopt_callback(CURLOPT_HEADERFUNCTION, write_callback, context)?;
            for (option, value) in [
                (CURLOPT_CONNECTTIMEOUT_MS, TCP_TIMEOUT_MILLIS),
                (CURLOPT_TIMEOUT_MS, TCP_TIMEOUT_MILLIS),
                (CURLOPT_NOSIGNAL, 1),
                (CURLOPT_NOBODY, 1),
                (CURLOPT_FOLLOWLOCATION, 0),
            ] {
                handle.setopt_long(option, value, context)?;
            }
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
            let user_agent = match server.scheme {
                UrlScheme::Http => HTTP_USER_AGENT,
                UrlScheme::Https => HTTPS_USER_AGENT,
            };
            for (option, value) in [
                (CURLOPT_URL, server.request_target.as_ptr()),
                (CURLOPT_USERAGENT, user_agent.as_ptr()),
                (CURLOPT_PROTOCOLS_STR, protocol.as_ptr()),
            ] {
                handle.setopt_str(option, value, context)?;
            }
            let body_data = (&raw mut body_target).cast::<c_void>();
            let header_data = (&raw mut header_target).cast::<c_void>();
            for (option, ptr) in [
                (
                    CURLOPT_ERRORBUFFER,
                    error_buffer.as_mut_ptr().cast::<c_void>(),
                ),
                (CURLOPT_WRITEDATA, body_data),
                (CURLOPT_HEADERDATA, header_data),
            ] {
                curl_setopt!(handle, option, ptr, context)?;
            }
            let request_start = Instant::now();
            let code = handle.perform();
            let scheme_result = (code == CURLE_OK).then(|| handle.ensure_scheme(protocol, context));
            (code, request_start, scheme_result)
        };
        if let Some(callback_error) = body_sink.error.or(header_capture.error) {
            self.easy_handle = None;
            return Err(error(context, &callback_error));
        }
        if perform_code != CURLE_OK {
            let perform_error = if let Ok(message_cstr) = CStr::from_bytes_until_nul(&error_buffer)
                && !message_cstr.is_empty()
            {
                format!(
                    "curl_easy_perform 실패: {} ({})",
                    message_cstr.to_string_lossy(),
                    perform_code
                )
            } else {
                curl_error("curl_easy_perform", perform_code)
            };
            self.easy_handle = None;
            return Err(error(context, &perform_error));
        }
        if let Err(scheme_error) = scheme_result.unwrap_or_else(|| process::abort()) {
            self.easy_handle = None;
            return Err(scheme_error);
        }
        let header_block = header_capture
            .completed_block
            .ok_or_else(|| error(context, "완료된 HTTP 응답 header block을 찾지 못했습니다."))?;
        header_block.finish(request_start, context)
    }
}
impl CurlHeaderCapture {
    fn append(&mut self, bytes: &[u8]) -> bool {
        if !account_bytes(
            &mut self.bytes_seen,
            &mut self.error,
            bytes.len(),
            self.limit,
            "헤더",
        ) {
            return false;
        }
        let line = bytes.trim_suffix(b"\n").trim_suffix(b"\r");
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
        let (name, tail) = line.split_at(colon);
        let is_age_header = match name.len() {
            3 if name.eq_ignore_ascii_case(AGE_HEADER_NAME) => true,
            4 if name.eq_ignore_ascii_case(DATE_HEADER_NAME) => false,
            _ => return true,
        };
        let value_bytes = tail.get(1..).unwrap_or_else(|| process::abort());
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
fn account_bytes(
    bytes_seen: &mut usize,
    error_slot: &mut Option<Cow<'static, str>>,
    len: usize,
    limit: usize,
    label: &str,
) -> bool {
    let Some(next_len) = bytes_seen.checked_add(len) else {
        *error_slot = Some(Cow::Owned(format!("HTTP HEAD 응답 {label} 크기 계산 실패")));
        return false;
    };
    if next_len > limit {
        *error_slot = Some(Cow::Owned(format!(
            "HTTP HEAD 응답 {label} 크기가 허용 한도({limit} bytes)를 초과했습니다."
        )));
        return false;
    }
    *bytes_seen = next_len;
    true
}
#[cold]
#[inline(never)]
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
#[cold]
#[inline(never)]
fn setopt_error(context: &str, code: CurlCode) -> TimeError {
    error(context, &curl_error("curl_easy_setopt", code))
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
    if !match *target {
        CurlWriteTarget::Body(ref mut buffer) => account_bytes(
            &mut buffer.bytes_seen,
            &mut buffer.error,
            bytes.len(),
            HTTP_HEAD_MAX_BODY_BYTES,
            "본문",
        ),
        CurlWriteTarget::Header(ref mut capture) => (*capture).append(bytes),
    } {
        return 0;
    }
    len
}
