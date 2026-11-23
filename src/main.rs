extern crate alloc;
use self::{command::CliCommand, diagnostic::Result, file_output::OutputFile, menu_app::MenuApp};
#[cfg(target_arch = "x86_64")]
use self::{
    hardware_rng::{HardwareRandomSource, HardwareRng},
    korea_boundary::BoundaryCache,
    random_data::generate_random_data_with_rng,
};
use core::fmt;
#[cfg(target_arch = "x86_64")]
use std::io::{Write as _, stderr};
use std::{
    env,
    io::{self, IsTerminal as TerminalDetect},
    path::Path,
    sync::LazyLock,
};
#[cfg(target_arch = "x86_64")]
mod batch;
mod boundary_inflate;
mod buffmt;
mod command;
mod diagnostic;
mod file_output;
mod geo_uniform;
#[cfg(target_arch = "x86_64")]
mod hardware_rng;
mod input;
mod korea_boundary;
#[cfg(target_arch = "x86_64")]
mod ladder;
mod menu_app;
mod numeric;
mod output;
mod random_data;
#[cfg(target_arch = "x86_64")]
mod random_number;
mod stop_input;
mod time;
const BUFFER_SIZE: usize = 1016;
const FILE_NAME: &str = "random_data.txt";
const FILE_RECORD_FINAL_LABEL: &[u8] = "NMS 은하 좌표: ".as_bytes();
const FILE_RECORD_LINE_COUNT: usize = 17;
const FILE_RECORD_START: &[u8] = "64비트 난수: ".as_bytes();
const HELP_TEXT: &str = concat!(
    env!("CARGO_PKG_NAME"),
    " ",
    env!("CARGO_PKG_VERSION"),
    "\n난수 데이터 생성 및 서버 시간 동기화 도구\n\n사용법:\n  ",
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
    " time-observe <host> <seconds>\n\n옵션:\n",
    "  -h, --help               도움말\n  --version                버전\n"
);
static IS_TERMINAL: LazyLock<bool> = LazyLock::new(|| TerminalDetect::is_terminal(&io::stdout()));
const UTF8_BOM: &[u8; 3] = b"\xEF\xBB\xBF";
fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    if let Some(first) = args.next() {
        let info = if first == "-h" || first == "--help" {
            Some(("알 수 없는 도움말 옵션", HELP_TEXT))
        } else if first == "--version" {
            Some((
                "알 수 없는 --version 옵션",
                concat!(env!("CARGO_PKG_NAME"), " ", env!("CARGO_PKG_VERSION"), "\n"),
            ))
        } else {
            None
        };
        if let Some((err_label, text)) = info {
            if let Some(extra) = args.next() {
                return Err(format!("{err_label}: {}", extra.display()).into());
            }
            return output::write_slice_to_console(text.as_bytes()).map_err(Into::into);
        }
        return CliCommand::try_from((first, args))?.execute();
    }
    let output_file = OutputFile::try_from(Path::new(FILE_NAME))?;
    let mut app = cfg_select! {
        target_arch = "x86_64" => {
            MenuApp {
                input_buffer: String::new(),
                ladder_results_storage: String::new(),
                output_file,
                rng: HardwareRng::new(),
            }
        }
        _ => {
            MenuApp {
                input_buffer: String::new(),
                output_file,
            }
        }
    };
    #[cfg(target_arch = "x86_64")]
    {
        let mut err = stderr().lock();
        match app.rng.source() {
            HardwareRandomSource::RdSeed | HardwareRandomSource::RdRand => {
                app.rng.write_initial_source_notice(&mut err)?;
                let mut boundary_cache = BoundaryCache::default();
                let data = generate_random_data_with_rng(&app.rng, &mut boundary_cache)?;
                app.rng.write_rdseed_fallback_notice(&mut err)?;
                app.output_file.persist_and_print(&data)?;
            }
            HardwareRandomSource::None => err.write_all(
                "[경고] RDSEED/RDRAND를 지원하지 않아 하드웨어 RNG 기능(메뉴 1~4)을 비활성화합니다. 메뉴 5/7은 사용 가능합니다.\n"
                    .as_bytes(),
            )?,
        }
    }
    app.run()
}
fn write_line_best_effort(output: &mut dyn io::Write, args: fmt::Arguments<'_>) {
    match writeln!(output, "{args}") {
        Ok(()) | Err(_) => {}
    }
}
