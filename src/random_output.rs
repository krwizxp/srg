use super::{
    BUFFER_SIZE, IS_TERMINAL, Result,
    file_output::lock_mutex,
    output::{format_data_into_buffer, prefix_slice, write_slice_to_console},
    random_data::RandomDataSet,
};
use std::{
    fs::File,
    io::{BufWriter, Write as _},
    sync::Mutex,
};
pub fn write_random_data_to_console(data: &RandomDataSet) -> Result<()> {
    let mut buffer = [0_u8; BUFFER_SIZE];
    if *IS_TERMINAL {
        let console_len = format_data_into_buffer(data, &mut buffer, true)?;
        write_slice_to_console(prefix_slice(&buffer, console_len)?)?;
    } else {
        let file_len = format_data_into_buffer(data, &mut buffer, false)?;
        write_slice_to_console(prefix_slice(&buffer, file_len)?)?;
    }
    Ok(())
}
pub fn persist_and_print_random_data(
    file_mutex: &Mutex<BufWriter<File>>,
    data: &RandomDataSet,
) -> Result<()> {
    let mut buffer = [0_u8; BUFFER_SIZE];
    let file_len = format_data_into_buffer(data, &mut buffer, false)?;
    {
        let mut file_guard = lock_mutex(file_mutex, "Mutex 잠금 실패 (단일 쓰기 시)")?;
        file_guard.write_all(prefix_slice(&buffer, file_len)?)?;
        file_guard.flush()?;
    };
    write_random_data_to_console(data)
}
