use crate::diagnostic::{AppError, Result};
use crate::{
    BUFFER_SIZE, FILE_RECORD_FINAL_LABEL, FILE_RECORD_LINE_COUNT, FILE_RECORD_START, IS_TERMINAL,
    UTF8_BOM,
    output::{OutputTarget, format_data_into_buffer, write_slice_to_console},
    random_data::RandomDataSet,
};
use std::{
    fs::File,
    io::{Read as _, Seek as _, SeekFrom, Write as IoWrite},
    path::Path,
    process,
};
cfg_select! {
    windows => {
        use core::ffi::c_void;
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
        const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;
        const FILE_SHARE_READ_FLAG: u32 = 0x0000_0001;
        const FILE_STANDARD_INFO_CLASS: i32 = 1;
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
        unsafe extern "system" {
            #[link_name = "GetFileInformationByHandleEx"]
            fn get_file_information_by_handle_ex(
                file: *mut c_void,
                information_class: i32,
                information: *mut c_void,
                buffer_size: u32,
            ) -> i32;
        }
    }
    any(target_os = "linux", target_os = "macos") => {
        use std::fs::TryLockError;
        use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _, PermissionsExt as _};
    }
    _ => {}
}
#[cfg(target_os = "linux")]
const OPEN_NOFOLLOW_FLAG: i32 = 0x2_0000;
#[cfg(target_os = "macos")]
const OPEN_NOFOLLOW_FLAG: i32 = 0x0100;
pub(super) struct OutputFile {
    file: File,
}
impl TryFrom<&Path> for OutputFile {
    type Error = AppError;
    fn try_from(path: &Path) -> Result<Self> {
        let mut options = File::options();
        options.read(true).write(true).create(true).truncate(false);
        cfg_select! {
            target_os = "windows" => {
                options
                    .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT_FLAG | FILE_FLAG_SEQUENTIAL_SCAN)
                    .share_mode(FILE_SHARE_READ_FLAG);
            }
            any(target_os = "linux", target_os = "macos") => {
                options.custom_flags(OPEN_NOFOLLOW_FLAG).mode(0o600);
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
        metadata
            .is_file()
            .ok_or_else(|| AppError::message("출력 경로는 일반 파일이어야 합니다."))?;
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
        (link_count == 1)
            .ok_or_else(|| AppError::message("출력 파일의 하드 링크 수는 1이어야 합니다."))?;
        #[cfg(any(target_os = "linux", target_os = "macos"))]
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
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        if metadata.mode() & 0o077 != 0 {
            let mut permissions = metadata.permissions();
            permissions.set_mode(metadata.mode() & !0o077);
            file.set_permissions(permissions)?;
            if file.metadata()?.mode() & 0o077 != 0 {
                return Err(AppError::message(
                    "출력 파일의 group/other 접근 권한을 제거하지 못했습니다.",
                ));
            }
        }
        let len = file.seek(SeekFrom::End(0))?;
        if len == 0 {
            IoWrite::write_all(&mut file, UTF8_BOM)?;
        } else {
            let mut bom = [0_u8; UTF8_BOM.len()];
            file.rewind()?;
            file.read_exact(&mut bom)
                .map_err(|source| AppError::context("기존 출력 파일 BOM 읽기 실패", source))?;
            (&bom == UTF8_BOM).ok_or_else(|| {
                AppError::message("기존 출력 파일은 SRG UTF-8 형식이어야 합니다.")
            })?;
            let bom_len = u64::try_from(UTF8_BOM.len()).unwrap_or_else(|_| process::abort());
            if len != bom_len {
                let max_tail_len = u64::try_from(BUFFER_SIZE).unwrap_or_else(|_| process::abort());
                let tail_len_u64 = len.min(max_tail_len);
                let tail_len = usize::try_from(tail_len_u64).unwrap_or_else(|_| process::abort());
                let mut tail = [0_u8; BUFFER_SIZE];
                let tail_start = len.strict_sub(tail_len_u64);
                file.seek(SeekFrom::Start(tail_start))?;
                file.read_exact(tail.get_mut(..tail_len).unwrap_or_else(|| process::abort()))?;
                file.seek(SeekFrom::End(0))?;
                let content_start = if tail_start == 0 { UTF8_BOM.len() } else { 0 };
                let content = tail
                    .get(content_start..tail_len)
                    .unwrap_or_else(|| process::abort());
                let record_start = content
                    .array_windows::<{ FILE_RECORD_START.len() }>()
                    .rposition(|window| window == FILE_RECORD_START)
                    .ok_or_else(|| {
                        AppError::message("기존 출력 파일의 마지막 SRG 레코드를 찾지 못했습니다.")
                    })?;
                if record_start != 0
                    && content
                        .get(record_start.strict_sub(1))
                        .is_none_or(|&byte| byte != b'\n')
                {
                    return Err(AppError::message(
                        "기존 출력 파일의 마지막 SRG 레코드 경계가 올바르지 않습니다.",
                    ));
                }
                let record = content
                    .get(record_start..)
                    .unwrap_or_else(|| process::abort());
                str::from_utf8(record).map_err(|source| {
                    AppError::context("기존 출력 파일이 올바른 UTF-8이 아닙니다.", source)
                })?;
                let body = record.strip_suffix(b"\n").ok_or_else(|| {
                    AppError::message("기존 출력 파일의 마지막 SRG 레코드가 완전하지 않습니다.")
                })?;
                let mut lines = body.split(|&byte| byte == b'\n');
                let first = lines.next().unwrap_or_else(|| process::abort());
                let mut record_line_count = 1_usize;
                let mut last = first;
                for line in lines {
                    record_line_count = record_line_count.strict_add(1);
                    last = line;
                }
                let final_value_is_valid =
                    last.strip_prefix(FILE_RECORD_FINAL_LABEL)
                        .is_some_and(|value| {
                            let mut coordinate_count = 0_usize;
                            value.split(|&byte| byte == b':').all(|coordinate| {
                                coordinate_count = coordinate_count.strict_add(1);
                                coordinate.len() == 4
                                    && coordinate.iter().all(|&byte| {
                                        byte.is_ascii_digit() || matches!(byte, b'A'..=b'F')
                                    })
                            }) && coordinate_count == 4
                        });
                if !first.starts_with(FILE_RECORD_START)
                    || record_line_count != FILE_RECORD_LINE_COUNT
                    || !final_value_is_valid
                {
                    return Err(AppError::message(
                        "기존 출력 파일의 마지막 SRG 레코드 형식이 올바르지 않습니다.",
                    ));
                }
            }
        }
        Ok(Self { file })
    }
}
impl OutputFile {
    pub(super) fn clear(&mut self) -> Result<()> {
        self.file.set_len(0)?;
        self.file.rewind()?;
        self.file.write_all(UTF8_BOM)?;
        Ok(())
    }
    pub(super) fn persist_and_print(&mut self, data: &RandomDataSet) -> Result<()> {
        let mut buffer = [0_u8; BUFFER_SIZE];
        let file_len = format_data_into_buffer(data, &mut buffer, OutputTarget::File);
        self.file.write_all(buffer.split_at(file_len).0)?;
        let output_len = if *IS_TERMINAL {
            format_data_into_buffer(data, &mut buffer, OutputTarget::Console)
        } else {
            file_len
        };
        write_slice_to_console(buffer.split_at(output_len).0)?;
        Ok(())
    }
    #[cfg(target_arch = "x86_64")]
    pub(super) fn read_tail_into<'buffer>(
        &mut self,
        len: usize,
        buffer: &'buffer mut [u8],
    ) -> Result<&'buffer [u8]> {
        let tail = buffer.get_mut(..len).unwrap_or_else(|| process::abort());
        let offset = i64::try_from(len).unwrap_or_else(|_| process::abort());
        self.file.seek(SeekFrom::End(offset.strict_neg()))?;
        self.file.read_exact(tail)?;
        self.file.seek(SeekFrom::End(0))?;
        Ok(tail)
    }
    #[cfg(target_arch = "x86_64")]
    pub(super) const fn writer(&mut self) -> &mut File {
        &mut self.file
    }
}
