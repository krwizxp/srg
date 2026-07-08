use crate::constants::{OUTPUT_FILE_BUFFER_CAPACITY, UTF8_BOM};
use crate::diagnostic::{AppError, Result};
use std::{
    fs::{self, File},
    io::{BufWriter, Seek as _, SeekFrom, Write as IoWrite},
    path::Path,
    sync::{Mutex, MutexGuard},
};
pub struct OutputFile(File);
impl TryFrom<&Path> for OutputFile {
    type Error = AppError;
    fn try_from(path: &Path) -> Result<Self> {
        let mut file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
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
pub fn lock_mutex<'guard, T>(
    mutex: &'guard Mutex<T>,
    context_msg: &'static str,
) -> Result<MutexGuard<'guard, T>> {
    mutex
        .lock()
        .map_err(|err| AppError::message(format!("{context_msg}: {err}")))
}
