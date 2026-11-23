use super::super::{ParsedServer, UrlScheme};
use super::{FreshTimeHeaders, Result, TimeError, TimeSample, error, error_with_source};
use alloc::vec::Vec;
use core::{
    ffi::c_void,
    ptr::{NonNull, null, null_mut},
    result::Result as CoreResult,
    str,
};
use std::time::Instant;
#[link(name = "winhttp")]
unsafe extern "system" {
    fn WinHttpCloseHandle(h: HInternet) -> i32;
    fn WinHttpConnect(s: HInternet, host: *const u16, port: u16, res: u32) -> HInternet;
    fn WinHttpOpen(
        ua: *const u16,
        access: u32,
        proxy: *const u16,
        bypass: *const u16,
        flags: u32,
    ) -> HInternet;
    fn WinHttpOpenRequest(
        c: HInternet,
        verb: *const u16,
        path: *const u16,
        ver: *const u16,
        ref_: *const u16,
        accept: *const *const u16,
        flags: u32,
    ) -> HInternet;
    fn WinHttpQueryHeaders(
        r: HInternet,
        lvl: u32,
        name: *const u16,
        buf: *mut c_void,
        len: *mut u32,
        idx: *mut u32,
    ) -> i32;
    fn WinHttpReceiveResponse(r: HInternet, res: *mut c_void) -> i32;
    fn WinHttpSendRequest(
        r: HInternet,
        hdrs: *const u16,
        hlen: u32,
        opt: *const c_void,
        olen: u32,
        tlen: u32,
        ctx: usize,
    ) -> i32;
    fn WinHttpSetOption(h: HInternet, opt: u32, buf: *const c_void, len: u32) -> i32;
    fn WinHttpSetTimeouts(h: HInternet, res: i32, con: i32, snd: i32, rcv: i32) -> i32;
}
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetLastError() -> u32;
}
const HTTP_HEAD_MAX_HEADER_BYTES: usize = 64 * 1024;
const METHOD_HEAD_WIDE: [u16; 5] = [0x48, 0x45, 0x41, 0x44, 0];
const PATH_ROOT_WIDE: [u16; 2] = [0x2F, 0];
const USER_AGENT_CAPACITY: usize = {
    let http_len = "Rust-Time-Sync".len();
    let https_len = concat!("srg/", env!("CARGO_PKG_VERSION")).len();
    (if http_len > https_len {
        http_len
    } else {
        https_len
    })
    .strict_add(1)
};
enum WinHttpHandle {}
type HInternet = *mut WinHttpHandle;
#[derive(Default)]
pub(in crate::time) struct Client {
    connect: Option<Handle>,
    header_buffer: Vec<u16>,
    header_line_buffer: Vec<u8>,
    session: Option<Handle>,
}
struct Handle(NonNull<WinHttpHandle>);
impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: self.0 is a WinHTTP handle returned by WinHTTP and is closed exactly once here.
        unsafe {
            WinHttpCloseHandle(self.0.as_ptr());
        }
    }
}
impl Client {
    fn cached_connect(
        &mut self,
        server: &ParsedServer,
        context: &str,
    ) -> Result<NonNull<WinHttpHandle>> {
        if let Some(connect) = self.connect.as_ref() {
            return Ok(connect.0);
        }
        let session = if let Some(session) = self.session.as_ref() {
            session
        } else {
            let ua_text = match server.scheme {
                UrlScheme::Http => "Rust-Time-Sync",
                UrlScheme::Https => concat!("srg/", env!("CARGO_PKG_VERSION")),
            };
            let mut ua = [0_u16; USER_AGENT_CAPACITY];
            for (slot, unit) in ua.iter_mut().zip(ua_text.encode_utf16()) {
                *slot = unit;
            }
            let access = match server.scheme {
                UrlScheme::Http => 1,
                UrlScheme::Https => 4,
            };
            // SAFETY: ua is NUL-terminated and optional proxy pointers are intentionally null.
            let raw_session = unsafe { WinHttpOpen(ua.as_ptr(), access, null(), null(), 0) };
            let session = Self::non_null_handle(raw_session, "WinHttpOpen", context)?;
            // SAFETY: session is a valid WinHTTP session handle.
            let timeout_ok =
                unsafe { WinHttpSetTimeouts(session.0.as_ptr(), 5_000, 5_000, 5_000, 5_000) };
            (timeout_ok != 0_i32).ok_or_else(|| Self::last_error("WinHttpSetTimeouts", context))?;
            if let Err(code) = Self::try_set_dword_option(&session, 84, 0x0000_0800 | 0x0000_2000) {
                if matches!(code, 87 | 12_009) {
                    Self::try_set_dword_option(&session, 84, 0x0000_0800).map_err(|err_code| {
                        Self::windows_error("WinHttpSetOption SECURE_PROTOCOLS", err_code, context)
                    })?;
                } else {
                    return Err(Self::windows_error(
                        "WinHttpSetOption SECURE_PROTOCOLS",
                        code,
                        context,
                    ));
                }
            }
            for (option, operation, ignored) in [
                (
                    144_u32,
                    "WinHttpSetOption DISABLE_SECURE_PROTOCOL_FALLBACK",
                    None,
                ),
                (
                    195,
                    "WinHttpSetOption DISABLE_GLOBAL_POOLING",
                    Some(12_009_u32),
                ),
                (140, "WinHttpSetOption IPV6_FAST_FALLBACK", None),
            ] {
                if let Err(code) = Self::try_set_dword_option(&session, option, 1)
                    && ignored != Some(code)
                {
                    return Err(Self::windows_error(operation, code, context));
                }
            }
            self.session.insert(session)
        };
        // SAFETY: host_wide is NUL-terminated and session is a valid session handle.
        let raw_connect = unsafe {
            WinHttpConnect(
                session.0.as_ptr(),
                server.host_wide.as_ptr(),
                server.port,
                0,
            )
        };
        let handle = Self::non_null_handle(raw_connect, "WinHttpConnect", context)?;
        let connect = handle.0;
        self.connect = Some(handle);
        Ok(connect)
    }
    pub(in crate::time) fn fetch_head(
        &mut self,
        server: &ParsedServer,
        context: &str,
    ) -> Result<TimeSample> {
        let connect = self.cached_connect(server, context)?;
        let (request, request_start, response_received) = (|| -> Result<_> {
            let flags = match server.scheme {
                UrlScheme::Http => 0,
                UrlScheme::Https => 0x0080_0000,
            };
            // SAFETY: method and path are NUL-terminated and connect is valid.
            let raw_request = unsafe {
                WinHttpOpenRequest(
                    connect.as_ptr(),
                    METHOD_HEAD_WIDE.as_ptr(),
                    PATH_ROOT_WIDE.as_ptr(),
                    null(),
                    null(),
                    null(),
                    flags,
                )
            };
            let request = Self::non_null_handle(raw_request, "WinHttpOpenRequest", context)?;
            for (option, value, operation) in [
                (63_u32, 0x0007_u32, "WinHttpSetOption DISABLE_FEATURE"),
                (91, 64 * 1024, "WinHttpSetOption MAX_RESPONSE_HEADER_SIZE"),
            ] {
                Self::try_set_dword_option(&request, option, value)
                    .map_err(|code| Self::windows_error(operation, code, context))?;
            }
            let request_start = Instant::now();
            // SAFETY: request is valid and no additional request body or headers are needed for HEAD.
            let sent =
                unsafe { WinHttpSendRequest(request.0.as_ptr(), null(), 0, null(), 0, 0, 0) };
            (sent != 0_i32).ok_or_else(|| Self::last_error("WinHttpSendRequest", context))?;
            // SAFETY: request is a valid request handle and no reserved pointer is required.
            let received = unsafe { WinHttpReceiveResponse(request.0.as_ptr(), null_mut()) };
            (received != 0_i32)
                .ok_or_else(|| Self::last_error("WinHttpReceiveResponse", context))?;
            Ok((request, request_start, Instant::now()))
        })()
        .inspect_err(|_| {
            self.connect = None;
            self.session = None;
        })?;
        let mut time_headers = FreshTimeHeaders::default();
        Self::read_header_values(
            &request,
            9,
            "Date",
            context,
            (&mut self.header_buffer, &mut self.header_line_buffer),
            &mut |value| time_headers.capture_date(value, response_received),
        )?;
        Self::read_header_values(
            &request,
            48,
            "Age",
            context,
            (&mut self.header_buffer, &mut self.header_line_buffer),
            &mut |value| time_headers.capture_age(value),
        )?;
        time_headers.finish(request_start, context)
    }
    #[cold]
    #[inline(never)]
    fn last_error(operation: &str, context: &str) -> TimeError {
        // SAFETY: GetLastError has no preconditions.
        Self::windows_error(operation, unsafe { GetLastError() }, context)
    }
    fn non_null_handle(handle: HInternet, operation: &str, context: &str) -> Result<Handle> {
        NonNull::new(handle)
            .map(Handle)
            .ok_or_else(|| Self::last_error(operation, context))
    }
    fn read_header_values(
        request: &Handle,
        query: u32,
        name: &str,
        context: &str,
        (wide_buffer, ascii_buffer): (&mut Vec<u16>, &mut Vec<u8>),
        capture: &mut dyn FnMut(&str),
    ) -> Result<()> {
        let ensure_wide = |buf: &mut Vec<u16>, min_len: usize| -> Result<()> {
            if buf.len() < min_len {
                buf.try_reserve_exact(min_len.strict_sub(buf.len()))
                    .map_err(|source| {
                        error_with_source(context, "응답 헤더 메모리 확보 실패", source)
                    })?;
                buf.resize(min_len, 0);
            }
            Ok(())
        };
        ensure_wide(wide_buffer, 64)?;
        let mut index = 0_u32;
        loop {
            let current_index = index;
            let mut bytes = u32::try_from(wide_buffer.len().strict_mul(2)).unwrap_or(u32::MAX);
            // SAFETY: request is valid and wide_buffer provides bytes writable bytes for this indexed header.
            let fetched = unsafe {
                WinHttpQueryHeaders(
                    request.0.as_ptr(),
                    query,
                    null(),
                    wide_buffer.as_mut_ptr().cast::<c_void>(),
                    &raw mut bytes,
                    &raw mut index,
                )
            };
            // SAFETY: GetLastError has no preconditions.
            let fetch_error = (fetched == 0_i32).then(|| unsafe { GetLastError() });
            if fetch_error == Some(12_150) {
                return Ok(());
            }
            if let Some(code) = fetch_error
                && code != 122
            {
                return Err(Self::windows_error(
                    "WinHttpQueryHeaders header size",
                    code,
                    context,
                ));
            }
            let header_bytes = bytes as usize;
            (header_bytes <= HTTP_HEAD_MAX_HEADER_BYTES).ok_or_else(|| {
                error(
                    context,
                    &format!(
                        "{name} 헤더가 허용 한도({HTTP_HEAD_MAX_HEADER_BYTES} bytes)를 초과했습니다."
                    ),
                )
            })?;
            header_bytes.is_multiple_of(2).ok_or_else(|| {
                error(
                    context,
                    &format!("{name} 헤더 UTF-16 길이가 올바르지 않습니다."),
                )
            })?;
            let units = header_bytes.div_euclid(2);
            if fetch_error.is_some() {
                ensure_wide(wide_buffer, units)?;
                index = current_index;
                // SAFETY: wide_buffer has at least bytes writable bytes and request is valid.
                let long_fetched = unsafe {
                    WinHttpQueryHeaders(
                        request.0.as_ptr(),
                        query,
                        null(),
                        wide_buffer.as_mut_ptr().cast::<c_void>(),
                        &raw mut bytes,
                        &raw mut index,
                    )
                };
                (long_fetched != 0_i32)
                    .ok_or_else(|| Self::last_error("WinHttpQueryHeaders header", context))?;
            }
            (index > current_index).ok_or_else(|| {
                error(
                    context,
                    &format!("{name} 헤더 index가 진행되지 않았습니다."),
                )
            })?;
            let mut header = wide_buffer
                .get(..units)
                .ok_or_else(|| error(context, &format!("{name} 헤더 길이가 올바르지 않습니다.")))?;
            while let Some((&0, rest)) = header.split_last() {
                header = rest;
            }
            ascii_buffer.clear();
            ascii_buffer
                .try_reserve_exact(header.len())
                .map_err(|source| {
                    error_with_source(context, "응답 헤더 ASCII 메모리 확보 실패", source)
                })?;
            for &unit in header {
                ascii_buffer.push(u8::try_from(unit).map_err(|source| {
                    error_with_source(context, &format!("{name} 헤더 ASCII 변환 실패"), source)
                })?);
            }
            let value = str::from_utf8(ascii_buffer)
                .map(str::trim_ascii)
                .map_err(|source| {
                    error_with_source(context, &format!("{name} 헤더 UTF-8 변환 실패"), source)
                })?;
            capture(value);
        }
    }
    fn try_set_dword_option(handle: &Handle, option: u32, value: u32) -> CoreResult<(), u32> {
        // SAFETY: handle is a valid WinHTTP handle and value points to a DWORD option value.
        let ok = unsafe {
            WinHttpSetOption(
                handle.0.as_ptr(),
                option,
                (&raw const value).cast::<c_void>(),
                4,
            )
        };
        // SAFETY: GetLastError has no preconditions.
        (ok != 0_i32).ok_or_else(|| unsafe { GetLastError() })
    }
    #[cold]
    #[inline(never)]
    fn windows_error(operation: &str, code: u32, context: &str) -> TimeError {
        error(context, &format!("{operation} 실패: Windows error {code}"))
    }
}
