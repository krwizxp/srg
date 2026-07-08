use super::{
    file_output::lock_mutex,
    output::{OutputTarget, format_data_into_buffer, prefix_slice, write_slice_to_console},
    random_data::RandomDataSet,
};
use crate::constants::{BUFFER_SIZE, IS_TERMINAL};
use crate::diagnostic::Result;
use std::{
    fs::File,
    io::{BufWriter, Write as IoWrite},
    sync::Mutex,
};
pub fn write_random_data_to_console(
    data: &RandomDataSet,
    buffer: &mut [u8; BUFFER_SIZE],
    preformatted_file_len: Option<usize>,
) -> Result<()> {
    let output_len = if *IS_TERMINAL {
        format_data_into_buffer(data, buffer, OutputTarget::Console)?
    } else if let Some(file_len) = preformatted_file_len {
        file_len
    } else {
        format_data_into_buffer(data, buffer, OutputTarget::File)?
    };
    write_slice_to_console(prefix_slice(buffer, output_len)?)?;
    Ok(())
}
pub fn persist_and_print_random_data(
    file_mutex: &Mutex<BufWriter<File>>,
    data: &RandomDataSet,
) -> Result<()> {
    let mut buffer = [0_u8; BUFFER_SIZE];
    let file_len = format_data_into_buffer(data, &mut buffer, OutputTarget::File)?;
    {
        let mut file_guard = lock_mutex(file_mutex, "Mutex 잠금 실패 (단일 쓰기 시)")?;
        IoWrite::write_all(&mut *file_guard, prefix_slice(&buffer, file_len)?)?;
        IoWrite::flush(&mut *file_guard)?;
    };
    write_random_data_to_console(data, &mut buffer, Some(file_len))?;
    Ok(())
}
