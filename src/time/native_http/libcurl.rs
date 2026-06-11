use super::{
    HeadResponse, MIN_TRANSFER_TIME, ParseHttpDate, Result, TCP_TIMEOUT, error,
    find_date_header_value,
};
use alloc::{borrow::Cow, ffi::CString, string::String, vec::Vec};
use core::{
    ffi::{CStr, c_char, c_int, c_long, c_void},
    ptr::NonNull,
};
use std::{sync::LazyLock, time::Instant};
mod sys {
    use super::{Curl, CurlCode, CurlOption, c_char, c_long};
    #[link(name = "curl")]
    unsafe extern "C" {
        pub(super) fn curl_easy_cleanup(curl: *mut Curl);
        pub(super) fn curl_easy_init() -> *mut Curl;
        pub(super) fn curl_easy_perform(curl: *mut Curl) -> CurlCode;
        pub(super) fn curl_easy_reset(curl: *mut Curl);
        pub(super) fn curl_easy_setopt(curl: *mut Curl, option: CurlOption, ...) -> CurlCode;
        pub(super) fn curl_easy_strerror(code: CurlCode) -> *const c_char;
        pub(super) fn curl_global_init(flags: c_long) -> CurlCode;
    }
}
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
const HTTP_HEAD_MAX_BODY_BYTES: usize = 1024 * 1024;
const HTTP_HEAD_MAX_HEADER_BYTES: usize = 1024 * 1024;
const USER_AGENT_C_BYTES: &[u8] = concat!("srg/", env!("CARGO_PKG_VERSION"), "\0").as_bytes();
static CURL_INIT: LazyLock<CurlCode> = LazyLock::new(|| {
    // SAFETY: LazyLock runs this initializer once before any easy handles are used.
    unsafe { sys::curl_global_init(CURL_GLOBAL_DEFAULT) }
});
type Curl = c_void;
type CurlCode = c_int;
type CurlOption = c_int;
type CurlWriteCallback = unsafe extern "C" fn(*mut c_char, usize, usize, *mut c_void) -> usize;
#[derive(Default)]
pub(super) struct Client {
    easy_handle: Option<EasyHandle>,
}
struct EasyHandle(NonNull<Curl>);
struct CurlBodySink {
    bytes_seen: usize,
    error: Option<Cow<'static, str>>,
    limit: usize,
}
struct CurlHeaderCapture {
    bytes_seen: usize,
    date_header: Option<String>,
    error: Option<Cow<'static, str>>,
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
        let url_c = CString::new(url).map_err(|source| {
            error(
                context,
                format!("URL에 NUL 문자가 포함되어 있습니다: {source}"),
            )
        })?;
        let user_agent = CStr::from_bytes_with_nul(USER_AGENT_C_BYTES)
            .map_err(|source| error(context, format!("User-Agent C string 해석 실패: {source}")))?;
        let mut error_buffer = [c_char::default(); CURL_ERROR_SIZE];
        let mut body_sink = CurlBodySink { bytes_seen: 0, error: None, limit: HTTP_HEAD_MAX_BODY_BYTES };
        let mut header_capture = CurlHeaderCapture {
            bytes_seen: 0,
            date_header: None,
            error: None,
            limit: HTTP_HEAD_MAX_HEADER_BYTES,
            pending_line: Vec::new(),
        };
        let init_code = *CURL_INIT;
        if init_code != CURLE_OK {
            return Err(error(context, curl_error("curl_global_init", init_code)));
        }
        if self.easy_handle.is_none() {
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
        handle.setopt_callback(CURLOPT_WRITEFUNCTION, write_callback, context)?;
        handle.setopt_callback(CURLOPT_HEADERFUNCTION, write_callback, context)?;
        handle.setopt_str(CURLOPT_URL, url_c.as_ptr(), context)?;
        handle.setopt_str(CURLOPT_USERAGENT, user_agent.as_ptr(), context)?;
        handle.setopt_ptr(CURLOPT_ERRORBUFFER, error_buffer.as_mut_ptr(), context)?;
        handle.setopt_long(CURLOPT_CONNECTTIMEOUT, tcp_timeout_secs(), context)?;
        handle.setopt_long(CURLOPT_TIMEOUT, tcp_timeout_secs(), context)?;
        handle.setopt_long(CURLOPT_NOSIGNAL, 1, context)?;
        handle.setopt_long(CURLOPT_NOBODY, 1, context)?;
        handle.setopt_long(CURLOPT_FOLLOWLOCATION, 1, context)?;
        let perform_result = {
            let mut body_target = CurlWriteTarget::Body(&mut body_sink);
            let mut header_target = CurlWriteTarget::Header(&mut header_capture);
            let body_data = (&raw mut body_target).cast::<c_void>();
            let header_data = (&raw mut header_target).cast::<c_void>();
            handle.setopt_ptr(CURLOPT_WRITEDATA, body_data, context)?;
            handle.setopt_ptr(CURLOPT_HEADERDATA, header_data, context)?;
            let request_start = Instant::now();
            let perform_code = handle.perform();
            let response_received = Instant::now();
            CurlHeadPerform {
                code: perform_code,
                request_start,
                response_received,
            }
        };
        if !header_capture.pending_line.is_empty() {
            header_capture.capture_pending();
            header_capture.pending_line.clear();
        }
        if let Some(callback_error) = body_sink
            .error
            .take()
            .or_else(|| header_capture.error.take())
        {
            self.clear_reusable_handle();
            return Err(error(context, callback_error));
        }
        if perform_result.code != CURLE_OK {
            let error_bytes = error_buffer.map(|ch| ch.to_le_bytes()[0]);
            let perform_error: Cow<'static, str> = Cow::Owned(CStr::from_bytes_until_nul(&error_bytes)
                .ok()
                .filter(|message| !message.to_bytes().is_empty())
                .map_or_else(
                    || curl_error("curl_easy_perform", perform_result.code),
                    |message_cstr| {
                        let message = message_cstr.to_string_lossy();
                        format!("curl_easy_perform 실패: {message} ({})", perform_result.code)
                    },
                ));
            self.clear_reusable_handle();
            return Err(error(context, perform_error));
        }
        let Some(date_header_raw) = header_capture.date_header.as_deref() else {
            return Err(super::TimeError::header_not_found(format!("{context} 응답에서 Date")));
        };
        let server_time = parse_http_date(date_header_raw)?;
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
        if let Err(source) = self.pending_line.try_reserve(bytes.len()) {
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
        let date_header_raw = match find_date_header_value(&self.pending_line) {
            Ok(Some(value)) => value,
            Ok(None) => return true,
            Err(source) => {
                self.error = Some(Cow::Owned(format!(
                    "HTTP HEAD 응답 Date 헤더 UTF-8 변환 실패: {source}"
                )));
                return false;
            }
        };
        let mut date_header = String::new();
        if let Err(source) = date_header.try_reserve(date_header_raw.len()) {
            self.error = Some(Cow::Owned(format!(
                "HTTP HEAD 응답 Date 헤더 메모리 확보 실패: {source}"
            )));
            return false;
        }
        date_header.push_str(date_header_raw);
        self.date_header = Some(date_header);
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
    let payload_ptr = NonNull::slice_from_raw_parts(payload_head, len);
    // SAFETY: libcurl passes a readable buffer with len bytes for this callback.
    let bytes = unsafe { payload_ptr.as_ref() };
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
fn tcp_timeout_secs() -> c_long {
    let Ok(seconds) = c_long::try_from(TCP_TIMEOUT.as_secs()) else {
        return 5;
    };
    seconds
}
