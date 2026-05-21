use super::{FILE_NAME, OUTPUT_FILE_BUFFER_CAPACITY, Result, UTF8_BOM};
use core::{error::Error, fmt::Display};
use std::{
    fs::{self, File},
    io::{BufWriter, Error as IoError, ErrorKind, Write as IoWrite},
    path::Path,
    sync::{Mutex, MutexGuard},
};
cfg_select! {
    windows => {
        use core::ffi::c_void;
        use std::io::Result as IoResult;
        use std::os::windows::fs::{
            MetadataExt as WindowsMetadataExt, OpenOptionsExt as WindowsOpenOptionsExt,
        };
        use std::os::windows::io::AsRawHandle as WindowsRawHandle;
    }
    any(target_os = "linux", target_os = "macos") => {
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
        const FILE_ATTRIBUTE_REPARSE_POINT_FLAG: u32 = 0x0000_0400;
        const FILE_FLAG_OPEN_REPARSE_POINT_FLAG: u32 = 0x0020_0000;
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
        impl TryFrom<&File> for ByHandleFileInformation {
            type Error = IoError;
            fn try_from(file: &File) -> IoResult<Self> {
                let mut file_information = Self::default();
                // SAFETY: `GetFileInformationByHandle` only writes to the provided output
                // struct and uses the raw OS handle borrowed from `file` for the duration
                // of this call.
                let result = unsafe {
                    GetFileInformationByHandle(
                        WindowsRawHandle::as_raw_handle(file),
                        &raw mut file_information,
                    )
                };
                if result == 0_i32 {
                    return Err(IoError::last_os_error());
                }
                Ok(file_information)
            }
        }
        unsafe extern "system" {
            fn GetFileInformationByHandle(
                h_file: *mut c_void,
                file_information: *mut ByHandleFileInformation,
            ) -> i32;
        }
        fn file_has_multiple_links(file: &File) -> IoResult<bool> {
            Ok(ByHandleFileInformation::try_from(file)?.number_of_links > 1)
        }
    }
    _ => {}
}
cfg_select! {
    target_arch = "x86_64" => {
        pub fn ensure_file_exists_and_reopen(file_mutex: &Mutex<BufWriter<File>>) -> Result<()> {
            if Path::new(FILE_NAME).try_exists()? {
                return Ok(());
            }
            *lock_mutex(file_mutex, "Mutex 잠금 실패 (파일 생성 시)")? = open_or_create_file()?;
            Ok(())
        }
    }
    _ => {}
}
fn invalid_output_path_err(message: &'static str) -> Box<dyn Error + Send + Sync> {
    IoError::other(message).into()
}
pub fn validate_safe_output_file_path(path: &Path, allow_missing: bool) -> Result<()> {
    let maybe_metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => Some(metadata),
        Err(err) if allow_missing && err.kind() == ErrorKind::NotFound => None,
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
pub fn boxed_other_with_source<E>(context_msg: &str, err: E) -> Box<dyn Error + Send + Sync>
where
    E: Display,
{
    IoError::other(format!("{context_msg}: {err}")).into()
}
pub fn open_or_create_file() -> Result<BufWriter<File>> {
    let path = Path::new(FILE_NAME);
    validate_safe_output_file_path(path, true)?;
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
            return Err(boxed_other_with_source("출력 파일 잠금 실패", err));
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
pub fn lock_mutex<'guard, T>(
    mutex: &'guard Mutex<T>,
    context_msg: &str,
) -> Result<MutexGuard<'guard, T>> {
    mutex
        .lock()
        .map_err(|err| boxed_other_with_source(context_msg, err))
}
