use super::super::{ParsedServer, UrlScheme};
use super::{FreshTimeHeaders, TimeSample, error};
use crate::diagnostic::{AppError, Result, or_abort};
use alloc::{borrow::Cow, string::String};
use core::{
    ffi::{CStr, c_char, c_long, c_uint, c_void},
    mem::DropGuard,
    ptr::{NonNull, null_mut},
    slice, str,
};
use std::{process, sync::LazyLock, time::Instant};
#[link(name = "curl")]
unsafe extern "C" {
    fn curl_easy_cleanup(curl: *mut Curl);
    fn curl_easy_getinfo(curl: *mut Curl, info: CurlInfo, ...) -> CurlCode;
    fn curl_easy_init() -> *mut Curl;
    fn curl_easy_perform(curl: *mut Curl) -> CurlCode;
    fn curl_easy_reset(curl: *mut Curl);
    fn curl_easy_setopt(curl: *mut Curl, option: CurlOption, ...) -> CurlCode;
    fn curl_easy_strerror(code: CurlCode) -> *const c_char;
    fn curl_global_init(flags: c_long) -> CurlCode;
}
macro_rules! curl_setopt {
    ($handle:expr, $option:expr, $value:expr, $context:expr $(,)?) => {{
        // SAFETY: call sites pair each option with its documented libcurl ABI type.
        let code = unsafe { curl_easy_setopt($handle.0.as_ptr(), $option, $value) };
        (code == CURLE_OK).ok_or_else(|| setopt_error($context, code))
    }};
}
const CURLE_OK: CurlCode = 0;
const HTTP_HEAD_MAX_BODY_BYTES: usize = 1024 * 1024;
const HTTPS_USER_AGENT: &CStr =
    match CStr::from_bytes_with_nul(concat!("srg/", env!("CARGO_PKG_VERSION"), "\0").as_bytes()) {
        Ok(user_agent) => user_agent,
        Err(_) => c"srg",
    };
static CURL_INIT: LazyLock<CurlCode> = LazyLock::new(|| {
    // SAFETY: LazyLock runs this initializer once before any easy handles are used.
    unsafe { curl_global_init(3) }
});
type CurlCode = c_uint;
type CurlInfo = c_uint;
type CurlOption = c_uint;
type WriteTarget = (*mut CurlCapture, bool);
#[derive(Default)]
pub(in crate::time) struct Client {
    easy_handle: Option<EasyHandle>,
}
enum Curl {}
struct CurlCapture {
    body_bytes: usize,
    completed_block: Option<FreshTimeHeaders>,
    current_block: Option<FreshTimeHeaders>,
    error: Option<Cow<'static, str>>,
    header_bytes: usize,
    header_limit: usize,
}
struct EasyHandle(NonNull<Curl>);
impl Drop for EasyHandle {
    fn drop(&mut self) {
        // SAFETY: self.0 is an easy handle returned by libcurl and is closed exactly once here.
        unsafe {
            curl_easy_cleanup(self.0.as_ptr());
        }
    }
}
impl Client {
    pub(in crate::time) fn fetch_head(
        &mut self,
        server: &ParsedServer,
        context: &str,
    ) -> Result<TimeSample> {
        let mut error_buffer = [0_u8; 256];
        let mut capture = CurlCapture {
            body_bytes: 0,
            completed_block: None,
            current_block: None,
            error: None,
            header_bytes: 0,
            header_limit: match server.scheme {
                UrlScheme::Http => 64 * 1024,
                UrlScheme::Https => 1024 * 1024,
            },
        };
        let init_code = *CURL_INIT;
        (init_code == CURLE_OK)
            .ok_or_else(|| error(context, &curl_error("curl_global_init", init_code)))?;
        let handle = match &mut self.easy_handle {
            &mut Some(ref mut handle) => handle,
            empty @ &mut None => {
                // SAFETY: curl_easy_init has no preconditions after global init.
                let raw = NonNull::new(unsafe { curl_easy_init() })
                    .ok_or_else(|| error(context, "curl_easy_init 실패"))?;
                empty.insert(EasyHandle(raw))
            }
        };
        let (perform_code, request_start, scheme_result) = {
            let mut body_target: WriteTarget = (&raw mut capture, false);
            let mut header_target: WriteTarget = (&raw mut capture, true);
            let _options = DropGuard::new(&*handle, |reset_handle| {
                // SAFETY: reset_handle is valid; reset clears options while keeping libcurl caches.
                unsafe {
                    curl_easy_reset(reset_handle.0.as_ptr());
                }
            });
            let callbacks: [(
                CurlOption,
                unsafe extern "C" fn(*mut c_char, usize, usize, *mut c_void) -> usize,
            ); 2] = [(20_011, write_callback), (20_079, write_callback)];
            for (option, cb) in callbacks {
                curl_setopt!(handle, option, cb, context)?;
            }
            for (option, value) in [
                (156_u32, 5_000_i64),
                (155, 5_000),
                (99, 1),
                (44, 1),
                (52, 0),
            ] {
                curl_setopt!(handle, option, c_long::from(value), context)?;
            }
            let (protocol, user_agent) = match server.scheme {
                UrlScheme::Http => {
                    curl_setopt!(handle, 10_177_u32, c"*".as_ptr(), context)?;
                    (c"http", c"Rust-Time-Sync")
                }
                UrlScheme::Https => {
                    curl_setopt!(handle, 32_u32, c_long::from(6_i32 | (1_i32 << 16)), context)?;
                    (c"https", HTTPS_USER_AGENT)
                }
            };
            for (option, value) in [
                (10_002_u32, server.request_target.as_ptr()),
                (10_018, user_agent.as_ptr()),
                (10_318, protocol.as_ptr()),
            ] {
                curl_setopt!(handle, option, value, context)?;
            }
            for (option, ptr) in [
                (10_010_u32, error_buffer.as_mut_ptr().cast::<c_void>()),
                (10_001, (&raw mut body_target).cast::<c_void>()),
                (10_029, (&raw mut header_target).cast::<c_void>()),
            ] {
                curl_setopt!(handle, option, ptr, context)?;
            }
            let request_start = Instant::now();
            // SAFETY: handle is configured with callbacks and buffers that live until the call returns.
            let code = unsafe { curl_easy_perform(handle.0.as_ptr()) };
            let scheme_result = (code == CURLE_OK).then(|| -> Result<()> {
                let mut scheme = null_mut::<c_char>();
                // SAFETY: scheme is a valid output pointer for CURLINFO_SCHEME.
                let info_code =
                    unsafe { curl_easy_getinfo(handle.0.as_ptr(), 0x10_0031, &raw mut scheme) };
                (info_code == CURLE_OK).ok_or_else(|| {
                    error(context, &curl_error("curl_easy_getinfo scheme", info_code))
                })?;
                let scheme_ptr = NonNull::new(scheme)
                    .ok_or_else(|| error(context, "curl 최종 scheme이 비어 있습니다."))?;
                // SAFETY: libcurl returns a NUL-terminated scheme string owned by the easy handle.
                let bytes = unsafe { CStr::from_ptr(scheme_ptr.as_ptr()) }.to_bytes();
                bytes
                    .eq_ignore_ascii_case(protocol.to_bytes())
                    .ok_or_else(|| {
                        error(
                            context,
                            &format!(
                                "curl 최종 scheme이 {}가 아닙니다: {}",
                                protocol.to_string_lossy(),
                                String::from_utf8_lossy(bytes)
                            ),
                        )
                    })
            });
            (code, request_start, scheme_result)
        };
        if let Some(callback_error) = capture.error {
            self.easy_handle = None;
            return Err(error(context, &callback_error));
        }
        if perform_code != CURLE_OK {
            let msg = if let Ok(cstr) = CStr::from_bytes_until_nul(&error_buffer)
                && !cstr.is_empty()
            {
                format!(
                    "curl_easy_perform 실패: {} ({perform_code})",
                    cstr.to_string_lossy()
                )
            } else {
                curl_error("curl_easy_perform", perform_code)
            };
            self.easy_handle = None;
            return Err(error(context, &msg));
        }
        if let Err(scheme_error) = scheme_result.unwrap_or_else(|| process::abort()) {
            self.easy_handle = None;
            return Err(scheme_error);
        }
        capture
            .completed_block
            .ok_or_else(|| error(context, "완료된 HTTP 응답 header block을 찾지 못했습니다."))?
            .finish(request_start, context)
    }
}
#[cold]
#[inline(never)]
fn curl_error(context: &str, code: CurlCode) -> String {
    // SAFETY: curl_easy_strerror returns either null or a static NUL-terminated message for code.
    let raw_ptr = unsafe { curl_easy_strerror(code) };
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
fn setopt_error(context: &str, code: CurlCode) -> AppError {
    error(context, &curl_error("curl_easy_setopt", code))
}
unsafe extern "C" fn write_callback(
    ptr: *mut c_char,
    size: usize,
    nmemb: usize,
    userdata: *mut c_void,
) -> usize {
    let (Some(len), Some(payload_head), Some(target_ptr)) = (
        size.checked_mul(nmemb).filter(|&len| len != 0),
        NonNull::new(ptr.cast::<u8>()),
        NonNull::new(userdata.cast::<WriteTarget>()),
    ) else {
        return 0;
    };
    // SAFETY: len is non-zero, payload_head is non-null, and libcurl passes a readable buffer with len bytes.
    let bytes = unsafe { slice::from_raw_parts(payload_head.as_ptr(), len) };
    // SAFETY: userdata is the WriteTarget pointer configured before curl_easy_perform.
    let (capture_raw, is_header) = unsafe { *target_ptr.as_ptr() };
    let Some(mut capture_ptr) = NonNull::new(capture_raw) else {
        return 0;
    };
    // SAFETY: capture_raw points to the stack CurlCapture owned by fetch_head for the duration of curl_easy_perform.
    let capture = unsafe { capture_ptr.as_mut() };
    let (slot, limit, label) = if is_header {
        (&mut capture.header_bytes, capture.header_limit, "헤더")
    } else {
        (&mut capture.body_bytes, HTTP_HEAD_MAX_BODY_BYTES, "본문")
    };
    let Some(next_len) = slot.checked_add(bytes.len()) else {
        capture.error = Some(Cow::Owned(format!("HTTP HEAD 응답 {label} 크기 계산 실패")));
        return 0;
    };
    if next_len > limit {
        capture.error = Some(Cow::Owned(format!(
            "HTTP HEAD 응답 {label} 크기가 허용 한도({limit} bytes)를 초과했습니다."
        )));
        return 0;
    }
    *slot = next_len;
    if !is_header {
        return len;
    }
    let line = bytes.trim_suffix(b"\n").trim_suffix(b"\r");
    if line.starts_with(b"HTTP/") {
        capture.current_block = Some(FreshTimeHeaders::default());
        return len;
    }
    if line.is_empty() {
        if let Some(block) = capture.current_block.take() {
            capture.completed_block = Some(block);
        }
        return len;
    }
    let (Some(block), Some(colon)) = (
        capture.current_block.as_mut(),
        line.iter().position(|&byte| byte == b':'),
    ) else {
        return len;
    };
    let (name, tail) = line.split_at(colon);
    let is_age = match name.len() {
        3 if name.eq_ignore_ascii_case(b"age") => true,
        4 if name.eq_ignore_ascii_case(b"date") => false,
        _ => return len,
    };
    let header_raw = match str::from_utf8(or_abort(tail.get(1..))).map(str::trim_ascii) {
        Ok(raw) => raw,
        Err(source) => {
            capture.error = Some(Cow::Owned(format!(
                "HTTP HEAD 응답 헤더 UTF-8 변환 실패: {source}"
            )));
            return 0;
        }
    };
    if is_age {
        block.capture_age(header_raw);
    } else {
        block.capture_date(header_raw, Instant::now());
    }
    len
}
