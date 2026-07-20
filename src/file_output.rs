use crate::UTF8_BOM;
use crate::diagnostic::{AppError, Result};
use std::{
    fs::File,
    io::{BufWriter, Seek as _, SeekFrom, Write as IoWrite},
    path::Path,
};
cfg_select! {
    windows => {
        use core::{ffi::c_void, mem::size_of};
        use std::{
            io::Error as IoError,
            os::windows::{
                fs::{MetadataExt as _, OpenOptionsExt as _},
                io::AsRawHandle as _,
            },
        };
        const ERROR_SHARING_VIOLATION_CODE: i32 = 32;
        const FILE_ATTRIBUTE_REPARSE_POINT_FLAG: u32 = 0x0000_0400;
        const FILE_FLAG_OPEN_REPARSE_POINT_FLAG: u32 = 0x0020_0000;
        const FILE_SHARE_READ_FLAG: u32 = 0x0000_0001;
        const FILE_STANDARD_INFO_CLASS: i32 = 1;
        const FILE_STANDARD_INFO_SIZE: u32 = 24;
    }
    any(target_os = "linux", target_os = "macos") => {
        use std::fs::TryLockError;
        use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _};
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
    _ => {}
}
const OUTPUT_FILE_BUFFER_CAPACITY: usize = 0x0010_0000;
#[cfg(target_os = "windows")]
const _: () = assert!(
    size_of::<FileStandardInfo>() == 24,
    "Windows FILE_STANDARD_INFO size mismatch"
);
#[cfg(target_os = "windows")]
#[repr(C)]
#[derive(Default)]
struct FileStandardInfo {
    allocation_size: i64,
    end_of_file: i64,
    number_of_links: u32,
    delete_pending: u8,
    directory: u8,
}
#[cfg(target_os = "windows")]
unsafe extern "system" {
    #[link_name = "GetFileInformationByHandleEx"]
    fn get_file_information_by_handle_ex(
        file: *mut c_void,
        information_class: i32,
        information: *mut c_void,
        buffer_size: u32,
    ) -> i32;
}
pub(super) struct OutputFile {
    writer: BufWriter<File>,
}
impl TryFrom<&Path> for OutputFile {
    type Error = AppError;
    fn try_from(path: &Path) -> Result<Self> {
        let mut options = File::options();
        options.read(true).write(true).create(true).truncate(false);
        cfg_select! {
            target_os = "windows" => {
                options
                    .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT_FLAG)
                    .share_mode(FILE_SHARE_READ_FLAG);
            }
            any(target_os = "linux", target_os = "macos") => {
                options.custom_flags(OPEN_NOFOLLOW_FLAG);
            }
            _ => {}
        }
        let mut file = match options.open(path) {
            Ok(file) => file,
            Err(source) => {
                #[cfg(target_os = "windows")]
                if source.raw_os_error() == Some(ERROR_SHARING_VIOLATION_CODE) {
                    return Err(AppError::message(
                        "다른 srg 인스턴스가 출력 파일을 사용 중입니다.",
                    ));
                }
                return Err(source.into());
            }
        };
        let metadata = file.metadata()?;
        #[cfg(target_os = "windows")]
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT_FLAG != 0 {
            return Err(AppError::message(
                "출력 파일은 일반 파일이어야 하며 리파스 포인트는 허용되지 않습니다.",
            ));
        }
        if !metadata.is_file() {
            return Err(AppError::message("출력 경로는 일반 파일이어야 합니다."));
        }
        let link_count = cfg_select! {
            target_os = "windows" => {{
                let mut standard_info = FileStandardInfo::default();
                // SAFETY: standard_info is a valid FILE_STANDARD_INFO buffer for the borrowed file handle.
                let status = unsafe {
                    get_file_information_by_handle_ex(
                        file.as_raw_handle(),
                        FILE_STANDARD_INFO_CLASS,
                        (&raw mut standard_info).cast::<c_void>(),
                        FILE_STANDARD_INFO_SIZE,
                    )
                };
                if status == 0_i32 {
                    return Err(IoError::last_os_error().into());
                }
                u64::from(standard_info.number_of_links)
            }}
            any(target_os = "linux", target_os = "macos") => {
                metadata.nlink()
            }
            _ => {
                compile_error!("Output file validation supports only Windows, Linux, and macOS.")
            }
        };
        if link_count != 1 {
            return Err(AppError::message(
                "출력 파일의 하드 링크 수는 1이어야 합니다.",
            ));
        }
        cfg_select! {
            any(target_os = "linux", target_os = "macos") => {
                match file.try_lock() {
                    Ok(()) => {}
                    Err(TryLockError::WouldBlock) => {
                        return Err(AppError::message("다른 srg 인스턴스가 출력 파일을 사용 중입니다."));
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
        Ok(Self {
            writer: BufWriter::with_capacity(OUTPUT_FILE_BUFFER_CAPACITY, file),
        })
    }
}
impl OutputFile {
    pub(super) const fn writer(&mut self) -> &mut BufWriter<File> {
        &mut self.writer
    }
}
