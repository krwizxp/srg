use crate::constants::{FILE_NAME, OUTPUT_FILE_BUFFER_CAPACITY, UTF8_BOM};
use crate::diagnostic::{AppError, Result};
use std::{
    fs::{self, File},
    io::{BufWriter, ErrorKind, Write as IoWrite},
    path::Path,
};
cfg_select! {
    target_arch = "x86_64" => {
        use std::sync::{Mutex, MutexGuard};
    }
    _ => {}
}
cfg_select! {
    windows => {
        use core::ffi::c_void;
        use std::os::windows::fs::{
            MetadataExt as WindowsMetadataExt, OpenOptionsExt as WindowsOpenOptionsExt,
        };
        use std::{
            io::{Error as IoError, Result as IoResult},
            os::windows::io::AsRawHandle as WindowsRawHandle,
        };
    }
    any(target_os = "linux", target_os = "macos") => {
        use std::os::unix::fs::{
            MetadataExt as UnixMetadataExt, OpenOptionsExt as UnixOpenOptionsExt,
        };
    }
    _ => {}
}
cfg_select! {
    windows => {
        mod sys {
            use super::{ByHandleFileInformation, c_void};
            unsafe extern "system" {
                #[link_name = "GetFileInformationByHandleEx"]
                pub fn get_file_information_by_handle_ex(
                    h_file: *mut c_void,
                    file_information_class: i32,
                    file_information: *mut c_void,
                    buffer_size: u32,
                ) -> i32;
                #[link_name = "GetFileInformationByHandle"]
                pub fn get_file_information_by_handle(
                    h_file: *mut c_void,
                    file_information: *mut ByHandleFileInformation,
                ) -> i32;
            }
        }
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
        const FILE_ATTRIBUTE_REPARSE_POINT_FLAG: u32 = 0x0000_0400;
        const FILE_ID_INFO_CLASS: i32 = 18;
        const FILE_FLAG_OPEN_REPARSE_POINT_FLAG: u32 = 0x0020_0000;
        const FILE_STANDARD_INFO_CLASS: i32 = 1;
    }
}
cfg_select! {
    windows => {
        #[repr(C)]
        #[derive(Default)]
        struct FileTime {
            low_date_time: u32,
            high_date_time: u32,
        }
        #[repr(C)]
        #[derive(Default)]
        struct ByHandleFileInformation {
            file_attributes: u32,
            creation_time: FileTime,
            last_access_time: FileTime,
            last_write_time: FileTime,
            volume_serial_number: u32,
            file_size_high: u32,
            file_size_low: u32,
            number_of_links: u32,
            file_index_high: u32,
            file_index_low: u32,
        }
        #[repr(C)]
        #[derive(Default)]
        struct FileStandardInfo {
            allocation_size: i64,
            end_of_file: i64,
            number_of_links: u32,
            delete_pending: u8,
            directory: u8,
        }
        #[repr(C)]
        #[derive(Default)]
        struct FileIdInfo {
            volume_serial_number: u64,
            file_id: [u8; 16],
        }
    }
    _ => {}
}
cfg_select! {
    all(target_arch = "x86_64", windows) => {
        #[derive(Eq, PartialEq)]
        enum FileIdentity {
            ByHandleInfo {
                file_index_high: u32,
                file_index_low: u32,
                volume_serial_number: u32,
            },
            FileId {
                file_id: [u8; 16],
                volume_serial_number: u64,
            },
        }
    }
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")) => {
        #[derive(Eq, PartialEq)]
        struct FileIdentity {
            device: u64,
            inode: u64,
        }
    }
    _ => {}
}
cfg_select! {
    windows => {
        fn file_has_multiple_links(file: &File) -> IoResult<bool> {
            let mut standard_info = FileStandardInfo::default();
            let buffer_size = u32::try_from(size_of::<FileStandardInfo>())
                .map_err(|source| IoError::other(format!("FILE_STANDARD_INFO 크기 변환 실패: {source}")))?;
            // SAFETY: `standard_info` is a valid FILE_STANDARD_INFO buffer for the borrowed file handle.
            let result = unsafe {
                sys::get_file_information_by_handle_ex(
                    WindowsRawHandle::as_raw_handle(file),
                    FILE_STANDARD_INFO_CLASS,
                    (&raw mut standard_info).cast::<c_void>(),
                    buffer_size,
                )
            };
            if result != 0_i32 {
                return Ok(standard_info.number_of_links > 1);
            }
            let mut file_information = ByHandleFileInformation::default();
            // SAFETY: `GetFileInformationByHandle` only writes to `file_information` and reads the borrowed file handle during the call.
            let fallback_result = unsafe {
                sys::get_file_information_by_handle(
                    WindowsRawHandle::as_raw_handle(file),
                    &raw mut file_information,
                )
            };
            if fallback_result == 0_i32 {
                Err(IoError::last_os_error())
            } else {
                Ok(file_information.number_of_links > 1)
            }
        }
    }
    _ => {}
}
cfg_select! {
    target_arch = "x86_64" => {
        pub fn ensure_file_guard_current(
            file_guard: &mut MutexGuard<'_, BufWriter<File>>,
        ) -> Result<()> {
            let path = Path::new(FILE_NAME);
            let needs_reopen = cfg_select! {
                windows => {{
                    validate_safe_output_file_path(path)?;
                    let mut options = File::options();
                    options.read(true);
                    WindowsOpenOptionsExt::custom_flags(&mut options, FILE_FLAG_OPEN_REPARSE_POINT_FLAG);
                    let current_identity = match options.open(path) {
                        Ok(file) => Some(open_file_identity(&file)?),
                        Err(err) if err.kind() == ErrorKind::NotFound => None,
                        Err(err) => return Err(err.into()),
                    };
                    current_identity.as_ref() != Some(&open_file_identity(file_guard.get_ref())?)
                }}
                any(target_os = "linux", target_os = "macos") => {{
                    validate_safe_output_file_path(path)?;
                    let mut options = File::options();
                    options.read(true);
                    UnixOpenOptionsExt::custom_flags(&mut options, OPEN_NOFOLLOW_FLAG);
                    let current_identity = match options.open(path) {
                        Ok(file) => Some(open_file_identity(&file)?),
                        Err(err) if err.kind() == ErrorKind::NotFound => None,
                        Err(err) => return Err(err.into()),
                    };
                    current_identity.as_ref() != Some(&open_file_identity(file_guard.get_ref())?)
                }}
                _ => {
                    !path.try_exists()?
                }
            };
            if !needs_reopen {
                return Ok(());
            }
            file_guard.flush()?;
            **file_guard = open_or_create_file()?;
            Ok(())
        }
        pub fn ensure_file_exists_and_reopen(
            file_mutex: &Mutex<BufWriter<File>>,
        ) -> Result<()> {
            let mut file_guard = lock_mutex(file_mutex, "Mutex 잠금 실패 (파일 상태 확인 시)")?;
            ensure_file_guard_current(&mut file_guard)
        }
    }
    _ => {}
}
cfg_select! {
    all(target_arch = "x86_64", windows) => {
        fn open_file_identity(file: &File) -> Result<FileIdentity> {
            let metadata = file.metadata()?;
            if !metadata.is_file() {
                return Err(invalid_output_path_err(
                    "출력 경로는 일반 파일이어야 합니다.",
                ));
            }
            let mut file_information = ByHandleFileInformation::default();
            // SAFETY: `GetFileInformationByHandle` only reads the borrowed file handle and writes to `file_information`.
            let result = unsafe {
                sys::get_file_information_by_handle(
                    WindowsRawHandle::as_raw_handle(file),
                    &raw mut file_information,
                )
            };
            if result == 0_i32 {
                return Err(IoError::last_os_error().into());
            }
            if file_information.file_attributes & FILE_ATTRIBUTE_REPARSE_POINT_FLAG != 0 {
                return Err(invalid_output_path_err(
                    "출력 파일은 일반 파일이어야 하며 리파스 포인트는 허용되지 않습니다.",
                ));
            }
            if file_has_multiple_links(file)? {
                return Err(invalid_output_path_err(
                    "출력 파일은 하드 링크가 아니어야 합니다.",
                ));
            }
            let mut file_id_info = FileIdInfo::default();
            let buffer_size = u32::try_from(size_of::<FileIdInfo>())
                .map_err(|source| AppError::message(format!("FILE_ID_INFO 크기 변환 실패: {source}")))?;
            // SAFETY: `file_id_info` is a valid FILE_ID_INFO buffer for the borrowed file handle.
            let file_id_result = unsafe {
                sys::get_file_information_by_handle_ex(
                    WindowsRawHandle::as_raw_handle(file),
                    FILE_ID_INFO_CLASS,
                    (&raw mut file_id_info).cast::<c_void>(),
                    buffer_size,
                )
            };
            if file_id_result != 0_i32 {
                return Ok(FileIdentity::FileId {
                    file_id: file_id_info.file_id,
                    volume_serial_number: file_id_info.volume_serial_number,
                });
            }
            Ok(FileIdentity::ByHandleInfo {
                file_index_high: file_information.file_index_high,
                file_index_low: file_information.file_index_low,
                volume_serial_number: file_information.volume_serial_number,
            })
        }
    }
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")) => {
        fn open_file_identity(file: &File) -> Result<FileIdentity> {
            let metadata = file.metadata()?;
            if !metadata.is_file() {
                return Err(invalid_output_path_err(
                    "출력 경로는 일반 파일이어야 합니다.",
                ));
            }
            if UnixMetadataExt::nlink(&metadata) > 1 {
                return Err(invalid_output_path_err(
                    "출력 파일은 하드 링크가 아니어야 합니다.",
                ));
            }
            Ok(FileIdentity {
                device: UnixMetadataExt::dev(&metadata),
                inode: UnixMetadataExt::ino(&metadata),
            })
        }
    }
    _ => {}
}
fn invalid_output_path_err(message: &'static str) -> AppError {
    AppError::message(message)
}
pub fn validate_safe_output_file_path(path: &Path) -> Result<()> {
    let maybe_metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => Some(metadata),
        Err(err) if err.kind() == ErrorKind::NotFound => None,
        Err(err) => return Err(err.into()),
    };
    let Some(metadata) = maybe_metadata else {
        return Ok(());
    };
    cfg_select! {
        windows => {
            if WindowsMetadataExt::file_attributes(&metadata) & FILE_ATTRIBUTE_REPARSE_POINT_FLAG != 0 {
                return Err(invalid_output_path_err(
                    "출력 파일은 일반 파일이어야 하며 리파스 포인트는 허용되지 않습니다.",
                ));
            }
        }
        _ => {
            if metadata.file_type().is_symlink() {
                return Err(invalid_output_path_err(
                    "출력 파일은 일반 파일이어야 하며 심볼릭 링크는 허용되지 않습니다.",
                ));
            }
        }
    }
    if !metadata.is_file() {
        return Err(invalid_output_path_err(
            "출력 경로는 일반 파일이어야 합니다.",
        ));
    }
    let has_multiple_links = cfg_select! {
        windows => {{
            let mut options = File::options();
            options.read(true);
            WindowsOpenOptionsExt::custom_flags(&mut options, FILE_FLAG_OPEN_REPARSE_POINT_FLAG);
            file_has_multiple_links(&options.open(path)?)?
        }}
        _ => {
            UnixMetadataExt::nlink(&metadata) > 1
        }
    };
    if has_multiple_links {
        return Err(invalid_output_path_err(
            "출력 파일은 하드 링크가 아니어야 합니다.",
        ));
    }
    Ok(())
}
pub fn open_or_create_file() -> Result<BufWriter<File>> {
    let path = Path::new(FILE_NAME);
    validate_safe_output_file_path(path)?;
    let mut file = cfg_select! {
        windows => {{
            let mut options = File::options();
            options.read(true).append(true).create(true);
            WindowsOpenOptionsExt::custom_flags(&mut options, FILE_FLAG_OPEN_REPARSE_POINT_FLAG);
            options.open(path)?
        }}
        _ => {{
            let mut options = File::options();
            options.read(true).append(true).create(true);
            UnixOpenOptionsExt::custom_flags(&mut options, OPEN_NOFOLLOW_FLAG);
            options.open(path)?
        }}
    };
    match file.try_lock() {
        Ok(()) => {}
        Err(fs::TryLockError::WouldBlock) => {
            return Err(invalid_output_path_err(
                "다른 srg 인스턴스가 출력 파일을 사용 중입니다.",
            ));
        }
        Err(fs::TryLockError::Error(err)) => {
            return Err(AppError::context("출력 파일 잠금 실패", err));
        }
    }
    let metadata = file.metadata()?;
    cfg_select! {
        windows => {
            if WindowsMetadataExt::file_attributes(&metadata) & FILE_ATTRIBUTE_REPARSE_POINT_FLAG != 0 {
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
    let has_multiple_links = cfg_select! {
        windows => {
            file_has_multiple_links(&file)?
        }
        _ => {
            UnixMetadataExt::nlink(&metadata) > 1
        }
    };
    if has_multiple_links {
        return Err(invalid_output_path_err(
            "출력 파일은 하드 링크가 아니어야 합니다.",
        ));
    }
    if metadata.len() == 0 {
        IoWrite::write_all(&mut file, UTF8_BOM)?;
        IoWrite::flush(&mut file)?;
    }
    Ok(BufWriter::with_capacity(OUTPUT_FILE_BUFFER_CAPACITY, file))
}
cfg_select! {
    target_arch = "x86_64" => {
        pub fn lock_mutex<'guard, T>(
            mutex: &'guard Mutex<T>,
            context_msg: &'static str,
        ) -> Result<MutexGuard<'guard, T>> {
            mutex
                .lock()
                .map_err(|err| AppError::message(format!("{context_msg}: {err}")))
        }
    }
    _ => {}
}
