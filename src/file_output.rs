use crate::constants::{OUTPUT_FILE_BUFFER_CAPACITY, UTF8_BOM};
use crate::diagnostic::{AppError, Result};
use std::{
    fs::{self, File},
    io::{BufWriter, ErrorKind, Seek as _, SeekFrom, Write as IoWrite},
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
        mod sys {
            use super::{ByHandleFileInformation, c_void};
            unsafe extern "system" {
                #[link_name = "GetFileInformationByHandleEx"]
                pub(super) fn get_file_information_by_handle_ex(
                    h_file: *mut c_void,
                    file_information_class: i32,
                    file_information: *mut c_void,
                    buffer_size: u32,
                ) -> i32;
                #[link_name = "GetFileInformationByHandle"]
                pub(super) fn get_file_information_by_handle(
                    h_file: *mut c_void,
                    file_information: *mut ByHandleFileInformation,
                ) -> i32;
            }
        }
    }
    target_family = "unix" => {
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
    target_family = "unix" => {
        const OPEN_NOFOLLOW_FLAG: i32 = 0;
    }
    windows => {
        const FILE_ATTRIBUTE_REPARSE_POINT_FLAG: u32 = 0x0000_0400;
        const FILE_FLAG_OPEN_REPARSE_POINT_FLAG: u32 = 0x0020_0000;
        const FILE_STANDARD_INFO_CLASS: i32 = 1;
    }
    _ => {}
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
    }
    _ => {}
}
pub(super) struct OutputFile(File);
impl TryFrom<&Path> for OutputFile {
    type Error = AppError;
    fn try_from(path: &Path) -> Result<Self> {
        match fs::symlink_metadata(path) {
            Ok(metadata) => validate_output_file_metadata(&metadata)?,
            Err(err) if err.kind() == ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
        let mut options = File::options();
        options.read(true).write(true).create(true).truncate(false);
        cfg_select! {
            windows => {
                WindowsOpenOptionsExt::custom_flags(
                    &mut options,
                    FILE_FLAG_OPEN_REPARSE_POINT_FLAG,
                );
            }
            target_family = "unix" => {
                UnixOpenOptionsExt::custom_flags(&mut options, OPEN_NOFOLLOW_FLAG);
            }
            _ => {}
        }
        let mut file = options.open(path)?;
        let metadata = file.metadata()?;
        validate_output_file_metadata(&metadata)?;
        let has_multiple_links = cfg_select! {
            windows => {{
                let mut standard_info = FileStandardInfo::default();
                let buffer_size =
                    u32::try_from(size_of::<FileStandardInfo>()).map_err(|source| {
                        IoError::other(format!("FILE_STANDARD_INFO 크기 변환 실패: {source}"))
                    })?;
                // SAFETY: `standard_info` is a valid FILE_STANDARD_INFO buffer for the borrowed file handle.
                let result = unsafe {
                    sys::get_file_information_by_handle_ex(
                        WindowsRawHandle::as_raw_handle(&file),
                        FILE_STANDARD_INFO_CLASS,
                        (&raw mut standard_info).cast::<c_void>(),
                        buffer_size,
                    )
                };
                if result != 0_i32 {
                    standard_info.number_of_links > 1
                } else {
                    let mut file_information = ByHandleFileInformation::default();
                    // SAFETY: `file_information` is valid and the borrowed file handle remains open during the call.
                    let fallback_result = unsafe {
                        sys::get_file_information_by_handle(
                            WindowsRawHandle::as_raw_handle(&file),
                            &raw mut file_information,
                        )
                    };
                    if fallback_result == 0_i32 {
                        return Err(IoError::last_os_error().into());
                    }
                    file_information.number_of_links > 1
                }
            }}
            target_family = "unix" => {
                UnixMetadataExt::nlink(&metadata) > 1
            }
            _ => false
        };
        if has_multiple_links {
            return Err(invalid_output_path_err(
                "출력 파일은 하드 링크가 아니어야 합니다.",
            ));
        }
        match file.try_lock() {
            Ok(()) => {}
            Err(fs::TryLockError::WouldBlock) => {
                return Err(AppError::message(
                    "다른 srg 인스턴스가 출력 파일을 사용 중입니다.",
                ));
            }
            Err(fs::TryLockError::Error(err)) => {
                return Err(AppError::context("출력 파일 잠금 실패", err));
            }
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
fn validate_output_file_metadata(metadata: &fs::Metadata) -> Result<()> {
    cfg_select! {
        windows => {
            if WindowsMetadataExt::file_attributes(metadata) & FILE_ATTRIBUTE_REPARSE_POINT_FLAG != 0
            {
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
    Ok(())
}
pub(super) fn lock_mutex<'guard, T>(
    mutex: &'guard Mutex<T>,
    context_msg: &'static str,
) -> Result<MutexGuard<'guard, T>> {
    mutex
        .lock()
        .map_err(|err| AppError::message(format!("{context_msg}: {err}")))
}
