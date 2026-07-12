use crate::constants::{OUTPUT_FILE_BUFFER_CAPACITY, UTF8_BOM};
use crate::diagnostic::{AppError, Result};
use std::{
    fs::File,
    io::{BufWriter, Seek as _, SeekFrom, Write as IoWrite},
    path::Path,
    sync::{Mutex, MutexGuard},
};
cfg_select! {
    windows => {
        use core::ffi::c_void;
        use std::os::windows::fs::{
            MetadataExt as WindowsMetadataExt, OpenOptionsExt as WindowsOpenOptionsExt,
        };
        use std::{
            io::Error as IoError,
            os::windows::io::AsRawHandle as WindowsRawHandle,
        };
        mod sys;
    }
    any(target_os = "linux", target_os = "macos") => {
        use std::fs::TryLockError;
        use std::os::unix::fs::{
            MetadataExt as UnixMetadataExt, OpenOptionsExt as UnixOpenOptionsExt,
        };
    }
    _ => {}
}
cfg_select! {
    target_os = "linux" => {
        const OPEN_NOFOLLOW_FLAG: i32 = 0x2_0000;
    }
    target_os = "macos" => {
        const OPEN_NOFOLLOW_FLAG: i32 = 0x0100;
    }
    windows => {
        const ERROR_SHARING_VIOLATION_CODE: i32 = 32;
        const FILE_ATTRIBUTE_REPARSE_POINT_FLAG: u32 = 0x0000_0400;
        const FILE_FLAG_OPEN_REPARSE_POINT_FLAG: u32 = 0x0020_0000;
        const FILE_SHARE_READ_FLAG: u32 = 0x0000_0001;
        const FILE_STANDARD_INFO_CLASS: i32 = 1;
    }
    _ => {}
}
cfg_select! {
    windows => {
        const FILE_STANDARD_INFO_SIZE: u32 = 24;
        const _: () = assert!(
            size_of::<FileStandardInfo>() == 24,
            "Windows FILE_STANDARD_INFO size mismatch"
        );
        #[repr(C)]
        #[derive(Default)]
        struct FileStandardInfo {
            allocation_size: i64,
            end_of_file: i64,
            number_of_links: u32,
            delete_pending: u8,
            directory: u8,
        }
    }
    _ => {}
}
pub(super) struct OutputFile(File);
impl TryFrom<&Path> for OutputFile {
    type Error = AppError;
    fn try_from(path: &Path) -> Result<Self> {
        let mut options = File::options();
        options.read(true).write(true).create(true).truncate(false);
        cfg_select! {
            windows => {
                WindowsOpenOptionsExt::custom_flags(
                    &mut options,
                    FILE_FLAG_OPEN_REPARSE_POINT_FLAG,
                );
                WindowsOpenOptionsExt::share_mode(&mut options, FILE_SHARE_READ_FLAG);
            }
            any(target_os = "linux", target_os = "macos") => {
                UnixOpenOptionsExt::custom_flags(&mut options, OPEN_NOFOLLOW_FLAG);
            }
            _ => {
                compile_error!("Output file security checks support only Windows, Linux, and macOS.")
            }
        }
        let mut file = match options.open(path) {
            Ok(file) => file,
            Err(source) => {
                cfg_select! {
                    windows => {
                        if source.raw_os_error() == Some(ERROR_SHARING_VIOLATION_CODE) {
                            return Err(AppError::message(
                                "다른 srg 인스턴스가 출력 파일을 사용 중입니다.",
                            ));
                        }
                    }
                    _ => {}
                }
                return Err(source.into());
            }
        };
        let metadata = file.metadata()?;
        cfg_select! {
            windows => {
                if WindowsMetadataExt::file_attributes(&metadata)
                    & FILE_ATTRIBUTE_REPARSE_POINT_FLAG
                    != 0
                {
                    return Err(invalid_output_path_err(
                        "출력 파일은 일반 파일이어야 하며 리파스 포인트는 허용되지 않습니다.",
                    ));
                }
            }
            _ => {}
        }
        if !metadata.is_file() {
            return Err(invalid_output_path_err(
                "출력 경로는 일반 파일이어야 합니다.",
            ));
        }
        let has_unexpected_link_count = cfg_select! {
            windows => {{
                let mut standard_info = FileStandardInfo::default();
                // SAFETY: `standard_info` is a valid FILE_STANDARD_INFO buffer for the borrowed file handle.
                let result = unsafe {
                    sys::get_file_information_by_handle_ex(
                        WindowsRawHandle::as_raw_handle(&file),
                        FILE_STANDARD_INFO_CLASS,
                        (&raw mut standard_info).cast::<c_void>(),
                        FILE_STANDARD_INFO_SIZE,
                    )
                };
                if result == 0_i32 {
                    return Err(IoError::last_os_error().into());
                }
                standard_info.number_of_links != 1
            }}
            any(target_os = "linux", target_os = "macos") => {
                UnixMetadataExt::nlink(&metadata) != 1
            }
            _ => {
                compile_error!("Output file security checks support only Windows, Linux, and macOS.")
            }
        };
        if has_unexpected_link_count {
            return Err(invalid_output_path_err(
                "출력 파일의 하드 링크 수는 1이어야 합니다.",
            ));
        }
        cfg_select! {
            any(target_os = "linux", target_os = "macos") => {
                match file.try_lock() {
                    Ok(()) => {}
                    Err(TryLockError::WouldBlock) => {
                        return Err(AppError::message(
                            "다른 srg 인스턴스가 출력 파일을 사용 중입니다.",
                        ));
                    }
                    Err(TryLockError::Error(err)) => {
                        return Err(AppError::context("출력 파일 잠금 실패", err));
                    }
                }
            }
            _ => {}
        }
        if file.seek(SeekFrom::End(0))? == 0 {
            IoWrite::write_all(&mut file, UTF8_BOM)?;
        }
        Ok(Self(file))
    }
}
impl From<OutputFile> for BufWriter<File> {
    #[inline]
    fn from(output_file: OutputFile) -> Self {
        Self::with_capacity(OUTPUT_FILE_BUFFER_CAPACITY, output_file.0)
    }
}
fn invalid_output_path_err(message: &'static str) -> AppError {
    AppError::message(message)
}
pub(super) fn lock_mutex<'guard, T>(
    mutex: &'guard Mutex<T>,
    context_msg: &'static str,
) -> Result<MutexGuard<'guard, T>> {
    mutex
        .lock()
        .map_err(|err| AppError::message(format!("{context_msg}: {err}")))
}
