use super::super::{ParsedServer, UrlScheme};
use super::{FreshTimeHeaders, TimeSample, error, error_with_source};
use crate::diagnostic::{AppError, Result};
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
    fn WinHttpOpen(ua: *const u16, acc: u32, px: *const u16, by: *const u16, fl: u32) -> HInternet;
    fn WinHttpOpenRequest(
        c: HInternet,
        v: *const u16,
        p: *const u16,
        ver: *const u16,
        r: *const u16,
        a: *const *const u16,
        fl: u32,
    ) -> HInternet;
    fn WinHttpQueryHeaders(
        r: HInternet,
        l: u32,
        n: *const u16,
        b: *mut c_void,
        bl: *mut u32,
        i: *mut u32,
    ) -> i32;
    fn WinHttpReceiveResponse(r: HInternet, res: *mut c_void) -> i32;
    fn WinHttpSendRequest(
        r: HInternet,
        h: *const u16,
        hl: u32,
        o: *const c_void,
        ol: u32,
        tl: u32,
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
const USER_AGENT_CAPACITY: usize = 32;
const _: () = assert!(
    concat!("srg/", env!("CARGO_PKG_VERSION")).len() < USER_AGENT_CAPACITY,
    "USER_AGENT_CAPACITY must fit package user agent"
);
type HInternet = *mut WinHttpHandle;
#[derive(Default)]
pub(in crate::time) struct Client {
    connect: Option<Handle>,
    header_buffer: Vec<u16>,
    header_line_buffer: Vec<u8>,
    session: Option<Handle>,
}
struct Handle(NonNull<WinHttpHandle>);
enum WinHttpHandle {}
impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: self.0 is a WinHTTP handle returned by WinHTTP and is closed once here.
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
            let (ua_text, access) = match server.scheme {
                UrlScheme::Http => ("Rust-Time-Sync", 1_u32),
                UrlScheme::Https => (concat!("srg/", env!("CARGO_PKG_VERSION")), 4_u32),
            };
            let mut ua = [0_u16; USER_AGENT_CAPACITY];
            for (slot, unit) in ua.iter_mut().zip(ua_text.encode_utf16()) {
                *slot = unit;
            }
            // SAFETY: ua is NUL-terminated and optional proxy pointers are intentionally null.
            let session = Self::non_null_handle(
                unsafe { WinHttpOpen(ua.as_ptr(), access, null(), null(), 0) },
                "WinHttpOpen",
                context,
            )?;
            // SAFETY: session is a valid WinHTTP session handle.
            let ok = unsafe { WinHttpSetTimeouts(session.0.as_ptr(), 5_000, 5_000, 5_000, 5_000) };
            (ok != 0_i32).ok_or_else(|| Self::last_error("WinHttpSetTimeouts", context))?;
            if let Err(code) = Self::try_set_dword_option(&session, 84, 0x0000_0800 | 0x0000_2000) {
                let err_code = if matches!(code, 87 | 12_009) {
                    Self::try_set_dword_option(&session, 84, 0x0000_0800).err()
                } else {
                    Some(code)
                };
                if let Some(final_code) = err_code {
                    return Err(Self::windows_error(
                        "WinHttpSetOption SECURE_PROTOCOLS",
                        final_code,
                        context,
                    ));
                }
            }
            for (opt, op, ignored) in [
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
                if let Err(code) = Self::try_set_dword_option(&session, opt, 1)
                    && ignored != Some(code)
                {
                    return Err(Self::windows_error(op, code, context));
                }
            }
            self.session.insert(session)
        };
        // SAFETY: host_wide is NUL-terminated and session is a valid session handle.
        let handle = Self::non_null_handle(
            unsafe {
                WinHttpConnect(
                    session.0.as_ptr(),
                    server.host_wide.as_ptr(),
                    server.port,
                    0,
                )
            },
            "WinHttpConnect",
            context,
        )?;
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
            let req = Self::non_null_handle(
                unsafe {
                    WinHttpOpenRequest(
                        connect.as_ptr(),
                        METHOD_HEAD_WIDE.as_ptr(),
                        PATH_ROOT_WIDE.as_ptr(),
                        null(),
                        null(),
                        null(),
                        flags,
                    )
                },
                "WinHttpOpenRequest",
                context,
            )?;
            for (opt, val, op) in [
                (63_u32, 0x0007_u32, "WinHttpSetOption DISABLE_FEATURE"),
                (91, 64 * 1024, "WinHttpSetOption MAX_RESPONSE_HEADER_SIZE"),
            ] {
                Self::try_set_dword_option(&req, opt, val)
                    .map_err(|code| Self::windows_error(op, code, context))?;
            }
            let start = Instant::now();
            // SAFETY: req is valid and no additional request body or headers are needed for HEAD.
            let sent = unsafe { WinHttpSendRequest(req.0.as_ptr(), null(), 0, null(), 0, 0, 0) };
            (sent != 0_i32).ok_or_else(|| Self::last_error("WinHttpSendRequest", context))?;
            // SAFETY: req is a valid request handle and no reserved pointer is required.
            let recvd = unsafe { WinHttpReceiveResponse(req.0.as_ptr(), null_mut()) };
            (recvd != 0_i32).ok_or_else(|| Self::last_error("WinHttpReceiveResponse", context))?;
            Ok((req, start, Instant::now()))
        })()
        .inspect_err(|_| {
            self.connect = None;
            self.session = None;
        })?;
        let mut time_headers = FreshTimeHeaders::default();
        let ensure_wide = |buf: &mut Vec<u16>, min_len: usize| -> Result<()> {
            if buf.len() < min_len {
                buf.try_reserve_exact(min_len.strict_sub(buf.len()))
                    .map_err(|src| error_with_source(context, "응답 헤더 메모리 확보 실패", src))?;
                buf.resize(min_len, 0);
            }
            Ok(())
        };
        let (wide_buf, ascii_buf) = (&mut self.header_buffer, &mut self.header_line_buffer);
        ensure_wide(wide_buf, 64)?;
        for (query, name, is_date) in [(9_u32, "Date", true), (48, "Age", false)] {
            let hdr_err = |msg: &str| error(context, &format!("{name} 헤더 {msg}"));
            let (mut index, mut resized) = (0_u32, false);
            loop {
                let current_index = index;
                let mut bytes = u32::try_from(wide_buf.len().strict_mul(2)).unwrap_or(u32::MAX);
                // SAFETY: request is valid and wide_buf provides bytes writable bytes.
                let fetched = unsafe {
                    WinHttpQueryHeaders(
                        request.0.as_ptr(),
                        query,
                        null(),
                        wide_buf.as_mut_ptr().cast::<c_void>(),
                        &raw mut bytes,
                        &raw mut index,
                    )
                };
                if fetched == 0_i32 {
                    // SAFETY: GetLastError has no preconditions.
                    let code = unsafe { GetLastError() };
                    if code == 12_150 {
                        break;
                    }
                    if code != 122 || resized {
                        return Err(Self::windows_error(
                            "WinHttpQueryHeaders header",
                            code,
                            context,
                        ));
                    }
                }
                let hdr_bytes = bytes as usize;
                (hdr_bytes <= HTTP_HEAD_MAX_HEADER_BYTES).ok_or_else(|| {
                    hdr_err(&format!(
                        "가 허용 한도({HTTP_HEAD_MAX_HEADER_BYTES} bytes)를 초과했습니다."
                    ))
                })?;
                hdr_bytes
                    .is_multiple_of(2)
                    .ok_or_else(|| hdr_err("UTF-16 길이가 올바르지 않습니다."))?;
                let units = hdr_bytes.div_euclid(2);
                if fetched == 0_i32 {
                    ensure_wide(wide_buf, units)?;
                    index = current_index;
                    resized = true;
                    continue;
                }
                resized = false;
                (index > current_index).ok_or_else(|| hdr_err("index가 진행되지 않았습니다."))?;
                let mut header = wide_buf
                    .get(..units)
                    .ok_or_else(|| hdr_err("길이가 올바르지 않습니다."))?;
                while let Some((&0, rest)) = header.split_last() {
                    header = rest;
                }
                ascii_buf.clear();
                ascii_buf.try_reserve_exact(header.len()).map_err(|src| {
                    error_with_source(context, "응답 헤더 ASCII 메모리 확보 실패", src)
                })?;
                for &unit in header {
                    ascii_buf.push(u8::try_from(unit).map_err(|src| {
                        error_with_source(context, &format!("{name} 헤더 ASCII 변환 실패"), src)
                    })?);
                }
                let value = str::from_utf8(ascii_buf)
                    .map(str::trim_ascii)
                    .map_err(|src| {
                        error_with_source(context, &format!("{name} 헤더 UTF-8 변환 실패"), src)
                    })?;
                if is_date {
                    time_headers.capture_date(value, response_received);
                } else {
                    time_headers.capture_age(value);
                }
            }
        }
        time_headers.finish(request_start, context)
    }
    #[cold]
    #[inline(never)]
    fn last_error(operation: &str, context: &str) -> AppError {
        // SAFETY: GetLastError has no preconditions.
        Self::windows_error(operation, unsafe { GetLastError() }, context)
    }
    fn non_null_handle(handle: HInternet, operation: &str, context: &str) -> Result<Handle> {
        NonNull::new(handle)
            .map(Handle)
            .ok_or_else(|| Self::last_error(operation, context))
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
    fn windows_error(operation: &str, code: u32, context: &str) -> AppError {
        error(context, &format!("{operation} 실패: Windows error {code}"))
    }
}
