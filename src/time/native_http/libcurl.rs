use super::{HeadResponse, MIN_TRANSFER_TIME, Result, TCP_TIMEOUT, error, find_date_header_value};
use alloc::{ffi::CString, string::String, vec::Vec};
use core::{
    ffi::{CStr, c_char, c_int, c_long, c_void},
    ptr::NonNull,
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
const HTTP_HEAD_MAX_BODY_BYTES: usize = 1024 * 1024;
const HTTP_HEAD_MAX_HEADER_BYTES: usize = 1024 * 1024;
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
    error: Option<String>,
    label: &'static str,
    limit: usize,
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
    fn setopt_str(&self, option: CurlOption, value: *const c_char, context: &str) -> Result<()> {
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
        let mut body_buffer = CurlBuffer::new("본문", HTTP_HEAD_MAX_BODY_BYTES);
        let mut header_buffer = CurlBuffer::new("헤더", HTTP_HEAD_MAX_HEADER_BYTES);
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
        handle.setopt_str(CURLOPT_USERAGENT, user_agent.as_ptr(), context)?;
        handle.setopt_ptr(CURLOPT_ERRORBUFFER, error_buffer.as_mut_ptr(), context)?;
        handle.setopt_long(CURLOPT_CONNECTTIMEOUT, tcp_timeout_secs(), context)?;
        handle.setopt_long(CURLOPT_TIMEOUT, tcp_timeout_secs(), context)?;
        handle.setopt_long(CURLOPT_NOSIGNAL, 1, context)?;
        handle.setopt_long(CURLOPT_NOBODY, 1, context)?;
        handle.setopt_long(CURLOPT_FOLLOWLOCATION, 1, context)?;
        let request_start = Instant::now();
        let perform_code = handle.perform();
        let response_received_inst = Instant::now();
        if let Some(callback_error) = body_buffer
            .error
            .take()
            .or_else(|| header_buffer.error.take())
        {
            return Err(error(context, callback_error));
        }
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
        let Some(date_header_raw) = header_buffer
            .bytes
            .rsplit(|byte| *byte == b'\n')
            .find_map(find_date_header_value)
        else {
            return Err(super::missing_date(context));
        };
        let mut date_header = String::new();
        date_header.try_reserve(date_header_raw.len()).map_err(|source| {
            error(
                context,
                format!("Date header 메모리 확보 실패: {source}"),
            )
        })?;
        date_header.push_str(date_header_raw);
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
impl CurlBuffer {
    fn append(&mut self, bytes: &[u8]) -> bool {
        let Some(next_len) = self.bytes.len().checked_add(bytes.len()) else {
            self.error = Some(format!("HTTP HEAD 응답 {} 크기 계산 실패", self.label));
            return false;
        };
        if next_len > self.limit {
            self.error = Some(format!(
                "HTTP HEAD 응답 {} 크기가 허용 한도({} bytes)를 초과했습니다.",
                self.label, self.limit
            ));
            return false;
        }
        if let Err(source) = self.bytes.try_reserve(bytes.len()) {
            self.error = Some(format!(
                "HTTP HEAD 응답 {} 메모리 확보 실패: {source}",
                self.label
            ));
            return false;
        }
        self.bytes.extend_from_slice(bytes);
        true
    }
    const fn new(label: &'static str, limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            error: None,
            label,
            limit,
        }
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
    let message = if raw_ptr.is_null() {
        "unknown curl error".to_owned()
    } else {
        // SAFETY: libcurl guarantees a valid NUL-terminated string for non-null strerror results.
        unsafe { CStr::from_ptr(raw_ptr) }
            .to_string_lossy()
            .into_owned()
    };
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
    // SAFETY: libcurl invokes this callback with a readable payload pointer and
    // the userdata pointer configured for the active response buffer.
    let Some((bytes, target)) = (unsafe { callback_payload(ptr, len, userdata) }) else {
        return 0;
    };
    if !target.append(bytes) {
        return 0;
    };
    len
}
unsafe fn callback_payload<'a>(
    ptr: *mut c_char,
    len: usize,
    userdata: *mut c_void,
) -> Option<(&'a [u8], &'a mut CurlBuffer)> {
    let bytes_ptr = NonNull::new(ptr.cast::<u8>())?;
    let mut target_ptr = NonNull::new(userdata.cast::<CurlBuffer>())?;
    let bytes_ptr = NonNull::slice_from_raw_parts(bytes_ptr, len);
    // SAFETY: libcurl passes a valid buffer with len bytes for the duration of this callback.
    let bytes = unsafe { bytes_ptr.as_ref() };
    // SAFETY: userdata is the target buffer pointer set before curl_easy_perform.
    let target = unsafe { target_ptr.as_mut() };
    Some((bytes, target))
}
fn tcp_timeout_secs() -> c_long {
    let Ok(seconds) = c_long::try_from(TCP_TIMEOUT.as_secs()) else {
        return 5;
    };
    seconds
}
