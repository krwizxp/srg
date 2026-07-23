extern crate alloc;
use self::command::CliCommand;
use self::diagnostic::{AppError, Result};
use self::{file_output::OutputFile, menu_app::MenuApp};
use core::fmt;
use std::{
    env,
    ffi::OsStr,
    io::{self, IsTerminal as TerminalDetect, Write as _},
    path::Path,
    sync::LazyLock,
};
cfg_select! {
    target_arch = "x86_64" => {
        use self::hardware_rng::{
            HardwareRandomSource, HardwareRng,
        };
        use self::random_data::generate_random_data_with_rng;
        use self::random_output::persist_and_print_random_data;
        use std::io::stderr;
        mod batch;
        mod hardware_rng;
        mod ladder;
        mod random_number;
        mod random_output;
    }
    _ => {}
}
mod buffmt;
mod command;
mod diagnostic;
mod file_output;
mod input;
mod menu_app;
mod numeric;
mod output;
mod random_data;
mod time;
const BUFFER_SIZE: usize = 1016;
const FILE_NAME: &str = "random_data.txt";
static IS_TERMINAL: LazyLock<bool> = LazyLock::new(|| TerminalDetect::is_terminal(&io::stdout()));
const UTF8_BOM: &[u8; 3] = b"\xEF\xBB\xBF";
const APP_NAME: &str = env!("CARGO_PKG_NAME");
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
const HELP_TEXT: &str = concat!(
    env!("CARGO_PKG_NAME"),
    " ",
    env!("CARGO_PKG_VERSION"),
    "\n난수 데이터 생성 및 서버 시간 동기화 도구\n\n",
    "사용법:\n  ",
    env!("CARGO_PKG_NAME"),
    "\n  ",
    env!("CARGO_PKG_NAME"),
    " --version\n  ",
    env!("CARGO_PKG_NAME"),
    " generate <count>\n  ",
    env!("CARGO_PKG_NAME"),
    " ladder <players-csv> <results-csv>\n  ",
    env!("CARGO_PKG_NAME"),
    " random-integer <min> <max>\n  ",
    env!("CARGO_PKG_NAME"),
    " random-float <min> <max>\n  ",
    env!("CARGO_PKG_NAME"),
    " time-observe <host> <seconds>\n\n",
    "옵션:\n",
    "  -h, --help               도움말\n",
    "  --version                버전\n"
);
fn unknown_option(label: &str, option: &OsStr) -> AppError {
    format!("{label}: {}", option.to_string_lossy()).into()
}
fn write_line_best_effort(output: &mut dyn io::Write, args: fmt::Arguments<'_>) {
    match writeln!(output, "{args}") {
        Ok(()) | Err(_) => {}
    }
}
fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    if let Some(first) = args.next() {
        if first == OsStr::new("-h") || first == OsStr::new("--help") {
            if let Some(extra) = args.next() {
                return Err(unknown_option("알 수 없는 도움말 옵션", &extra));
            }
            output::write_slice_to_console(HELP_TEXT.as_bytes())?;
            return Ok(());
        }
        if first == OsStr::new("--version") {
            if let Some(extra) = args.next() {
                return Err(unknown_option("알 수 없는 --version 옵션", &extra));
            }
            let mut out = io::stdout().lock();
            writeln!(out, "{APP_NAME} {APP_VERSION}")?;
            return Ok(());
        }
        return CliCommand::try_from((first, args))?.execute();
    }
    let input_buffer = String::new();
    let mut app = cfg_select! {
        target_arch = "x86_64" => {{
            let mut output_file = OutputFile::try_from(Path::new(FILE_NAME))?;
            let rng = HardwareRng::new();
            let num_64 = match rng.source() {
                HardwareRandomSource::RdSeed | HardwareRandomSource::RdRand => {
                    rng.write_initial_source_notice(&mut stderr().lock())?;
                    let data = generate_random_data_with_rng(&rng)?;
                    rng.write_rdseed_fallback_notice(&mut stderr().lock())?;
                    let num_64 = data.num_64;
                    persist_and_print_random_data(&mut output_file, &data)?;
                    num_64
                }
                HardwareRandomSource::None => {
                    let mut err = stderr().lock();
                    err.write_all(
                        "[경고] RDSEED/RDRAND를 지원하지 않아 하드웨어 RNG 기능(메뉴 1~4)을 비활성화합니다. 메뉴 5/7은 사용 가능합니다.\n"
                            .as_bytes(),
                    )?;
                    0
                }
            };
            let ladder_results_storage = String::new();
            MenuApp {
                input_buffer,
                ladder_results_storage,
                num_64,
                output_file,
                rng,
            }
        }}
        _ => {{
            let output_file = OutputFile::try_from(Path::new(FILE_NAME))?;
            MenuApp { input_buffer, output_file }
        }}
    };
    app.run()
}
