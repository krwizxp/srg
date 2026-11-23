#[cfg(target_os = "macos")]
use core::ffi::c_uint;
#[cfg(target_os = "linux")]
use core::ffi::c_ulong;
use std::io;
cfg_select! {
    target_os = "windows" => {
        use core::{ffi::{c_int, c_void}, ptr::null_mut};
        use std::io::{IsTerminal as _, stdin};
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetFileType(file: *mut c_void) -> u32;
            fn GetStdHandle(std_handle: i32) -> *mut c_void;
            fn PeekNamedPipe(
                pipe: *mut c_void,
                buffer: *mut c_void,
                buffer_size: u32,
                bytes_read: *mut u32,
                bytes_available: *mut u32,
                bytes_left: *mut u32,
            ) -> i32;
            fn ReadFile(
                file: *mut c_void,
                buffer: *mut c_void,
                bytes_to_read: u32,
                bytes_read: *mut u32,
                overlapped: *mut c_void,
            ) -> i32;
        }
        #[link(name = "msvcrt")]
        unsafe extern "C" {
            #[link_name = "_kbhit"]
            fn console_key_available() -> c_int;
            #[link_name = "_getch"]
            fn read_console_key() -> c_int;
        }
    }
    any(target_os = "linux", target_os = "macos") => {
        use core::ffi::{c_int, c_short, c_void};
        pub(crate) const POLLERR: c_short = 0x0008;
        pub(crate) const POLLHUP: c_short = 0x0010;
        pub(crate) const POLLIN: c_short = 0x0001;
        pub(crate) const POLLNVAL: c_short = 0x0020;
    }
    _ => {
        compile_error!("Stop input supports only Windows, Linux, and macOS.");
    }
}
cfg_select! {
    target_os = "linux" => {
        pub(crate) type PollDescriptorCount = c_ulong;
    }
    target_os = "macos" => {
        pub(crate) type PollDescriptorCount = c_uint;
    }
    _ => {}
}
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct PollFd {
    pub fd: c_int,
    pub events: c_short,
    pub revents: c_short,
}
#[cfg(any(target_os = "linux", target_os = "macos"))]
const _: () = assert!(
    size_of::<PollFd>() == 8 && align_of::<PollFd>() == 4,
    "pollfd layout must match the platform ABI"
);
#[cfg(any(target_os = "linux", target_os = "macos"))]
unsafe extern "C" {
    #[link_name = "poll"]
    pub(crate) fn poll_stdin(
        descriptors: *mut PollFd,
        descriptor_count: PollDescriptorCount,
        timeout: c_int,
    ) -> c_int;
    #[link_name = "read"]
    fn read_stdin(file: c_int, buffer: *mut c_void, count: usize) -> isize;
}
#[cfg(target_os = "windows")]
enum StopInputBackend {
    Console,
    Stream { file: *mut c_void, pipe: bool },
}
pub(crate) struct StopInput {
    #[cfg(target_os = "windows")]
    backend: StopInputBackend,
    input_bytes_remaining: Option<usize>,
}
impl TryFrom<Option<usize>> for StopInput {
    type Error = io::Error;
    fn try_from(max_bytes: Option<usize>) -> io::Result<Self> {
        #[cfg(target_os = "windows")]
        let backend = if stdin().is_terminal() {
            StopInputBackend::Console
        } else {
            // SAFETY: GetStdHandle returns the process standard input handle.
            let file = unsafe { GetStdHandle(-10) };
            (!file.is_null() && file.addr() != usize::MAX).ok_or_else(io::Error::last_os_error)?;
            // SAFETY: file is the process standard input handle.
            let file_type = unsafe { GetFileType(file) };
            matches!(file_type, 1 | 3)
                .ok_or_else(|| io::Error::other("지원하지 않는 Windows 표준 입력 유형입니다."))?;
            StopInputBackend::Stream {
                file,
                pipe: file_type == 3,
            }
        };
        Ok(Self {
            #[cfg(target_os = "windows")]
            backend,
            input_bytes_remaining: max_bytes,
        })
    }
}
impl StopInput {
    pub(crate) fn poll(&mut self) -> io::Result<bool> {
        cfg_select! {
            target_os = "windows" => {
                match self.backend {
                    StopInputBackend::Console => {
                        for _ in 0..64_usize {
                            // SAFETY: _kbhit reads only the calling process console input state.
                            if unsafe { console_key_available() } == 0_i32 {
                                return Ok(false);
                            }
                            // SAFETY: _kbhit reported an available console key for _getch to consume.
                            let key = unsafe { read_console_key() };
                            if key == 13_i32 {
                                return Ok(true);
                            }
                            if matches!(key, 0 | 224) {
                                // SAFETY: this checks whether an extended-key byte is available.
                                if unsafe { console_key_available() } != 0_i32 {
                                    // SAFETY: _kbhit reported the second byte of an extended key.
                                    unsafe {
                                        read_console_key();
                                    }
                                }
                                continue;
                            }
                            self.record_non_newline()?;
                        }
                        Ok(false)
                    }
                    StopInputBackend::Stream { file, pipe } => loop {
                        if pipe {
                            let mut available = 0_u32;
                            // SAFETY: file is a pipe handle and available is a writable DWORD.
                            if unsafe {
                                PeekNamedPipe(file, null_mut(), 0, null_mut(), &raw mut available, null_mut())
                            } == 0_i32
                            {
                                return pipe_failure(io::Error::last_os_error());
                            }
                            if available == 0 {
                                return Ok(false);
                            }
                        }
                        let mut byte = 0_u8;
                        let mut read = 0_u32;
                        // SAFETY: file is a readable standard input handle and byte is writable.
                        if unsafe {
                            ReadFile(file, (&raw mut byte).cast::<c_void>(), 1, &raw mut read, null_mut())
                        } == 0_i32
                        {
                            let source = io::Error::last_os_error();
                            return if pipe { pipe_failure(source) } else { Err(source) };
                        }
                        if read == 0 || byte == b'\n' {
                            return Ok(true);
                        }
                        self.record_non_newline()?;
                    },
                }
            }
            any(target_os = "linux", target_os = "macos") => {
                loop {
                    let mut descriptor = PollFd { fd: 0_i32, events: POLLIN, revents: 0 };
                    // SAFETY: descriptor is one writable pollfd and the count is exactly one.
                    let status = unsafe { poll_stdin(&raw mut descriptor, 1, 0_i32) };
                    if status < 0_i32 {
                        let source = io::Error::last_os_error();
                        if source.kind() == io::ErrorKind::Interrupted {
                            continue;
                        }
                        return Err(source);
                    }
                    if status == 0_i32 {
                        return Ok(false);
                    }
                    if descriptor.revents & POLLIN != 0 {
                        let mut byte = 0_u8;
                        // SAFETY: byte is a writable one-byte buffer and fd 0 is standard input.
                        let read = unsafe { read_stdin(0_i32, (&raw mut byte).cast::<c_void>(), 1) };
                        if read < 0 {
                            let source = io::Error::last_os_error();
                            if source.kind() == io::ErrorKind::Interrupted {
                                continue;
                            }
                            return Err(source);
                        }
                        if read == 0 || byte == b'\n' {
                            return Ok(true);
                        }
                        self.record_non_newline()?;
                        continue;
                    }
                    if descriptor.revents & POLLHUP != 0 {
                        return Ok(true);
                    }
                    if descriptor.revents & (POLLERR | POLLNVAL) != 0 {
                        return Err(io::Error::other(format!(
                            "표준 입력 poll 오류: revents={:#06x}",
                            descriptor.revents
                        )));
                    }
                    return Ok(false);
                }
            }
            _ => {
                compile_error!("Stop input supports only Windows, Linux, and macOS.");
            }
        }
    }
    fn record_non_newline(&mut self) -> io::Result<()> {
        let Some(remaining) = self.input_bytes_remaining.as_mut() else {
            return Ok(());
        };
        (*remaining != 0).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "서버 시간 종료 입력이 너무 깁니다.",
            )
        })?;
        *remaining = remaining.strict_sub(1);
        Ok(())
    }
}
#[cfg(target_os = "windows")]
fn pipe_failure(source: io::Error) -> io::Result<bool> {
    if source.kind() == io::ErrorKind::BrokenPipe {
        Ok(true)
    } else {
        Err(source)
    }
}
