use std::io;
cfg_select! {
    target_os = "windows" => {
        use core::{ffi::{c_int, c_void}, ptr::null_mut};
        use std::io::{IsTerminal as _, stdin};
    }
    any(target_os = "linux", target_os = "macos") => {
        use core::{
            ffi::{c_int, c_short, c_void},
            mem::size_of,
        };
    }
    _ => {}
}
cfg_select! {
    target_os = "linux" => {
        use core::ffi::c_ulong;
    }
    target_os = "macos" => {
        use core::ffi::c_uint;
    }
    _ => {}
}
cfg_select! {
    target_os = "windows" => {
        const ENTER_KEY: c_int = 13;
        const FILE_TYPE_DISK: u32 = 1;
        const FILE_TYPE_PIPE: u32 = 3;
        const MAX_CONSOLE_KEYS_PER_POLL: usize = 64;
        const STD_INPUT_HANDLE: i32 = -10;
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
        const POLL_DESCRIPTOR_COUNT: PollDescriptorCount = 1;
        const POLL_ERROR_EVENTS: c_short = 0x0038;
        const POLLHUP: c_short = 0x0010;
        const POLLIN: c_short = 0x0001;
    }
    _ => {
        compile_error!("Stop input supports only Windows, Linux, and macOS.");
    }
}
cfg_select! {
    target_os = "linux" => {
        type PollDescriptorCount = c_ulong;
    }
    target_os = "macos" => {
        type PollDescriptorCount = c_uint;
    }
    _ => {}
}
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[repr(C)]
struct PollFd {
    fd: c_int,
    events: c_short,
    revents: c_short,
}
#[cfg(any(target_os = "linux", target_os = "macos"))]
const _: () = assert!(
    size_of::<PollFd>() == 8,
    "pollfd layout must match the platform ABI"
);
#[cfg(any(target_os = "linux", target_os = "macos"))]
unsafe extern "C" {
    #[link_name = "poll"]
    fn poll_stdin(
        descriptors: *mut PollFd,
        descriptor_count: PollDescriptorCount,
        timeout: c_int,
    ) -> c_int;
    #[link_name = "read"]
    fn read_stdin(file: c_int, buffer: *mut c_void, count: usize) -> isize;
}
cfg_select! {
    target_os = "windows" => {
        enum StopInputBackend {
            Console,
            Stream { file: *mut c_void, pipe: bool },
        }
    }
    _ => {}
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
            let file = unsafe { GetStdHandle(STD_INPUT_HANDLE) };
            if file.is_null() || file.addr() == usize::MAX {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: file is the process standard input handle.
            let file_type = unsafe { GetFileType(file) };
            if !matches!(file_type, FILE_TYPE_DISK | FILE_TYPE_PIPE) {
                return Err(io::Error::other(
                    "지원하지 않는 Windows 표준 입력 유형입니다.",
                ));
            }
            StopInputBackend::Stream {
                file,
                pipe: file_type == FILE_TYPE_PIPE,
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
                self.poll_windows()
            }
            any(target_os = "linux", target_os = "macos") => {
                self.poll_unix()
            }
            _ => {
                compile_error!("Stop input supports only Windows, Linux, and macOS.");
            }
        }
    }
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn poll_unix(&mut self) -> io::Result<bool> {
        loop {
            let mut descriptor = PollFd {
                fd: 0_i32,
                events: POLLIN,
                revents: 0,
            };
            // SAFETY: descriptor is one writable pollfd and the count is exactly one.
            let status = unsafe { poll_stdin(&raw mut descriptor, POLL_DESCRIPTOR_COUNT, 0_i32) };
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
                if read == 0 {
                    return Ok(true);
                }
                if self.record_byte(byte)? {
                    return Ok(true);
                }
                continue;
            }
            if descriptor.revents & POLLHUP != 0 {
                return Ok(true);
            }
            if descriptor.revents & POLL_ERROR_EVENTS != 0 {
                return Err(io::Error::other(format!(
                    "표준 입력 poll 오류: revents={:#06x}",
                    descriptor.revents
                )));
            }
            return Ok(false);
        }
    }
    #[cfg(target_os = "windows")]
    fn poll_windows(&mut self) -> io::Result<bool> {
        match self.backend {
            StopInputBackend::Console => {
                for _ in 0..MAX_CONSOLE_KEYS_PER_POLL {
                    // SAFETY: _kbhit reads only the calling process console input state.
                    if unsafe { console_key_available() } == 0_i32 {
                        return Ok(false);
                    }
                    // SAFETY: _kbhit reported an available console key for _getch to consume.
                    let key = unsafe { read_console_key() };
                    if key == ENTER_KEY {
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
                    let status = unsafe {
                        PeekNamedPipe(
                            file,
                            null_mut(),
                            0,
                            null_mut(),
                            &raw mut available,
                            null_mut(),
                        )
                    };
                    if status == 0_i32 {
                        return pipe_failure(io::Error::last_os_error());
                    }
                    if available == 0 {
                        return Ok(false);
                    }
                }
                let mut byte = 0_u8;
                let mut read = 0_u32;
                // SAFETY: file is a readable standard input handle and byte is writable.
                let status = unsafe {
                    ReadFile(
                        file,
                        (&raw mut byte).cast::<c_void>(),
                        1,
                        &raw mut read,
                        null_mut(),
                    )
                };
                if status == 0_i32 {
                    let source = io::Error::last_os_error();
                    if pipe {
                        return pipe_failure(source);
                    }
                    return Err(source);
                }
                if read == 0 {
                    return Ok(true);
                }
                if self.record_byte(byte)? {
                    return Ok(true);
                }
            },
        }
    }
    fn record_byte(&mut self, byte: u8) -> io::Result<bool> {
        if byte == b'\n' {
            return Ok(true);
        }
        self.record_non_newline()?;
        Ok(false)
    }
    fn record_non_newline(&mut self) -> io::Result<()> {
        let Some(remaining) = self.input_bytes_remaining.as_mut() else {
            return Ok(());
        };
        if *remaining == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "서버 시간 종료 입력이 너무 깁니다.",
            ));
        }
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
