use super::{
    file_output::OutputFile,
    output::{OutputTarget, format_data_into_buffer, prefix_slice, write_slice_to_console},
    random_data::RandomDataSet,
};
use crate::diagnostic::Result;
use crate::{BUFFER_SIZE, IS_TERMINAL};
use std::io::Write as IoWrite;
pub(super) fn write_random_data_to_console(
    data: &RandomDataSet,
    buffer: &mut [u8; BUFFER_SIZE],
    preformatted_file_len: usize,
) -> Result<()> {
    let output_len = if *IS_TERMINAL {
        format_data_into_buffer(data, buffer, OutputTarget::Console)?
    } else {
        preformatted_file_len
    };
    write_slice_to_console(prefix_slice(buffer, output_len)?)?;
    Ok(())
}
pub(super) fn persist_and_print_random_data(
    output_file: &mut OutputFile,
    data: &RandomDataSet,
) -> Result<()> {
    let mut buffer = [0_u8; BUFFER_SIZE];
    let file_len = format_data_into_buffer(data, &mut buffer, OutputTarget::File)?;
    let writer = output_file.writer();
    IoWrite::write_all(&mut *writer, prefix_slice(&buffer, file_len)?)?;
    IoWrite::flush(writer)?;
    write_random_data_to_console(data, &mut buffer, file_len)?;
    Ok(())
}
