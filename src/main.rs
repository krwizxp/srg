extern crate alloc;
use self::diagnostic::Result;
use self::file_output::open_or_create_file;
use self::menu_app::MenuApp;
use core::fmt;
use std::{
    io::{self},
    process::ExitCode,
    sync::Mutex,
};
cfg_select! {
    target_arch = "x86_64" => {
        use self::hardware_rng::{
            HardwareRandomSource, hardware_random_source, take_rdseed_fallback_notice,
        };
        use self::random_data::generate_random_data;
        use self::random_output::persist_and_print_random_data;
        use std::io::{Write as IoWrite, stderr};
    }
    _ => {
        use self::diagnostic::AppError;
    }
}
cfg_select! {
    target_arch = "x86_64" => {
        mod batch;
        mod hardware_rng;
        mod random_number;
        mod random_output;
    }
    _ => {}
}
mod buffmt;
mod constants;
mod diagnostic;
mod file_output;
mod input;
mod menu_app;
mod numeric;
mod output;
mod random_data;
mod random_util;
mod tables;
mod time;
#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
compile_error!("SRG currently supports only Windows, Linux, and macOS.");
const GENERIC_INPUT_BUFFER_CAPACITY: usize = 256;
#[cfg(target_arch = "x86_64")]
fn reserved_string(capacity: usize, context: &'static str) -> Result<String> {
    let mut value = String::new();
    value
        .try_reserve(capacity)
        .map_err(|source| diagnostic::AppError::context(context, source))?;
    Ok(value)
}
fn write_line_best_effort(output: &mut dyn io::Write, args: fmt::Arguments<'_>) {
    match output.write_fmt(args) {
        Ok(()) | Err(_) => {}
    }
    match output.write_all(b"\n") {
        Ok(()) | Err(_) => {}
    }
}
fn main() -> Result<ExitCode> {
    let file_mutex = Mutex::new(open_or_create_file()?);
    let input_buffer = cfg_select! {
        target_arch = "x86_64" => {{
            reserved_string(GENERIC_INPUT_BUFFER_CAPACITY, "입력 버퍼 메모리 확보 실패")?
        }}
        _ => {{
            let mut value = String::new();
            value
                .try_reserve(GENERIC_INPUT_BUFFER_CAPACITY)
                .map_err(|source| AppError::context("입력 버퍼 메모리 확보 실패", source))?;
            value
        }}
    };
    let mut app = cfg_select! {
        target_arch = "x86_64" => {{
            let num_64 = match hardware_random_source() {
                HardwareRandomSource::RdSeed => {
                    let data = generate_random_data()?;
                    if take_rdseed_fallback_notice() {
                        let mut err = stderr().lock();
                        IoWrite::write_fmt(
                            &mut err,
                            format_args!("RDSEED 5분 타임아웃으로 RDRAND로 전환했습니다.\n"),
                        )?;
                    }
                    let num_64 = data.num_64;
                    persist_and_print_random_data(&file_mutex, &data)?;
                    num_64
                }
                HardwareRandomSource::RdRand => {
                    let mut err = stderr().lock();
                    IoWrite::write_fmt(
                        &mut err,
                        format_args!("RDSEED를 미지원하여 RDRAND를 사용합니다.\n"),
                    )?;
                    let data = generate_random_data()?;
                    let num_64 = data.num_64;
                    persist_and_print_random_data(&file_mutex, &data)?;
                    num_64
                }
                HardwareRandomSource::None => {
                    let mut err = stderr().lock();
                    IoWrite::write_fmt(
                        &mut err,
                        format_args!(
                            "[경고] RDSEED/RDRAND를 지원하지 않아 하드웨어 RNG 기능(메뉴 1~4)을 비활성화합니다. 메뉴 5/7은 사용 가능합니다.\n"
                        ),
                    )?;
                    0
                }
            };
            let ladder_players_storage = reserved_string(
                GENERIC_INPUT_BUFFER_CAPACITY,
                "사다리 참여자 버퍼 메모리 확보 실패",
            )?;
            let ladder_results_storage = reserved_string(
                GENERIC_INPUT_BUFFER_CAPACITY,
                "사다리 결과 버퍼 메모리 확보 실패",
            )?;
            MenuApp {
                file_mutex,
                input_buffer,
                ladder_players_storage,
                ladder_results_storage,
                num_64,
            }
        }}
        _ => {
            MenuApp { file_mutex, input_buffer }
        }
    };
    app.run()
}
