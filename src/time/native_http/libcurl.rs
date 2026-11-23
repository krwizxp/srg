use super::super::{ParsedServer, TimeError, UrlScheme};
use super::{FreshTimeHeaders, Result, TimeSample, error};
use crate::diagnostic::or_abort;
use alloc::{borrow::Cow, string::String};
use core::{
    ffi::{CStr, c_char, c_long, c_uint, c_void},
    marker::{PhantomData, PhantomPinned},
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
        // SAFETY: call sites pair each option with a wrapper using its documented libcurl ABI type.
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
#[repr(C)]
struct Curl {
    _data: (),
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}
type CurlCode = c_uint;
type CurlInfo = c_uint;
type CurlOption = c_uint;
type WriteTarget = (*mut CurlCapture, bool);
#[derive(Default)]
pub(in crate::time) struct Client {
    easy_handle: Option<EasyHandle>,
}
struct EasyHandle(NonNull<Curl>);
struct CurlCapture {
    body_bytes: usize,
    completed_block: Option<FreshTimeHeaders>,
    current_block: Option<FreshTimeHeaders>,
    error: Option<Cow<'static, str>>,
    header_bytes: usize,
    header_limit: usize,
}
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
                let raw_handle = NonNull::new(unsafe { curl_easy_init() })
                    .ok_or_else(|| error(context, "curl_easy_init 실패"))?;
                empty.insert(EasyHandle(raw_handle))
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
            for (option, callback) in callbacks {
                curl_setopt!(handle, option, callback, context)?;
            }
            let long_opts: [(CurlOption, c_long); 5] =
                [(156, 5_000), (155, 5_000), (99, 1), (44, 1), (52, 0)];
            for (option, value) in long_opts {
                curl_setopt!(handle, option, value, context)?;
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
                (10_018_u32, user_agent.as_ptr()),
                (10_318_u32, protocol.as_ptr()),
            ] {
                curl_setopt!(handle, option, value, context)?;
            }
            for (option, ptr) in [
                (10_010_u32, error_buffer.as_mut_ptr().cast::<c_void>()),
                (10_001_u32, (&raw mut body_target).cast::<c_void>()),
                (10_029_u32, (&raw mut header_target).cast::<c_void>()),
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
                let Some(scheme_ptr) = NonNull::new(scheme) else {
                    return Err(error(context, "curl 최종 scheme이 비어 있습니다."));
                };
                // SAFETY: libcurl returns a NUL-terminated scheme string owned by the easy handle.
                let scheme_bytes = unsafe { CStr::from_ptr(scheme_ptr.as_ptr()) }.to_bytes();
                scheme_bytes
                    .eq_ignore_ascii_case(protocol.to_bytes())
                    .ok_or_else(|| {
                        error(
                            context,
                            &format!(
                                "curl 최종 scheme이 {}가 아닙니다: {}",
                                protocol.to_string_lossy(),
                                String::from_utf8_lossy(scheme_bytes)
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
        capture
            .completed_block
            .ok_or_else(|| error(context, "완료된 HTTP 응답 header block을 찾지 못했습니다."))?
            .finish(request_start, context)
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
fn setopt_error(context: &str, code: CurlCode) -> TimeError {
    error(context, &curl_error("curl_easy_setopt", code))
}
unsafe extern "C" fn write_callback(
    ptr: *mut c_char,
    size: usize,
    nmemb: usize,
    userdata: *mut c_void,
) -> usize {
    let Some(len) = size.checked_mul(nmemb).filter(|&len| len != 0) else {
        return 0;
    };
    let (Some(payload_head), Some(target_ptr)) = (
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
    if !is_header {
        return if account_bytes(
            &mut capture.body_bytes,
            &mut capture.error,
            bytes.len(),
            HTTP_HEAD_MAX_BODY_BYTES,
            "본문",
        ) {
            len
        } else {
            0
        };
    }
    if !account_bytes(
        &mut capture.header_bytes,
        &mut capture.error,
        bytes.len(),
        capture.header_limit,
        "헤더",
    ) {
        return 0;
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
    let Some(current_block) = capture.current_block.as_mut() else {
        return len;
    };
    let Some(colon) = line.iter().position(|byte| *byte == b':') else {
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
        current_block.capture_age(header_raw);
    } else {
        current_block.capture_date(header_raw, Instant::now());
    }
    len
}
