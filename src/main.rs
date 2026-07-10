extern crate alloc;
use self::diagnostic::{AppError, Result};
use self::{
    constants::FILE_NAME,
    file_output::OutputFile,
    menu_app::{MenuApp, MenuAppInit},
};
use core::fmt;
use std::{
    env,
    ffi::OsStr,
    io::{self, BufWriter, Write as _},
    path::Path,
    process::ExitCode,
    sync::Mutex,
};
cfg_select! {
    target_arch = "x86_64" => {
        use self::hardware_rng::{
            HardwareRandomSource, HardwareRng,
        };
        use self::random_data::generate_random_data_with_rng;
        use self::random_output::persist_and_print_random_data;
        use std::io::stderr;
    }
    _ => {}
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
mod build_info;
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
cfg_select! {
    any(target_os = "windows", target_os = "linux", target_os = "macos") => {}
    _ => {
        compile_error!("SRG currently supports only Windows, Linux, and macOS.");
    }
}
const GENERIC_INPUT_BUFFER_CAPACITY: usize = 256;
const HELP_TEXT: &str = concat!(
    env!("CARGO_PKG_NAME"),
    " ",
    env!("CARGO_PKG_VERSION"),
    "\n난수 데이터 생성 및 서버 시간 동기화 도구\n\n",
    "사용법:\n  ",
    env!("CARGO_PKG_NAME"),
    "\n  ",
    env!("CARGO_PKG_NAME"),
    " --version [--verbose]\n\n",
    "옵션:\n",
    "  -h, --help               도움말\n",
    "  --version                버전\n",
    "  --version --verbose      빌드 메타데이터 포함 버전\n"
);
fn unknown_option(label: &str, option: &OsStr) -> AppError {
    format!("{label}: {}", option.to_string_lossy()).into()
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
    let mut args = env::args_os().skip(1);
    if let Some(first) = args.next() {
        if first == OsStr::new("-h") || first == OsStr::new("--help") {
            if let Some(extra) = args.next() {
                return Err(unknown_option("알 수 없는 도움말 옵션", &extra));
            }
            let mut out = io::stdout().lock();
            out.write_all(HELP_TEXT.as_bytes())?;
            return Ok(ExitCode::SUCCESS);
        }
        if first == OsStr::new("--version") {
            let verbose = match args.next() {
                None => false,
                Some(flag) if flag == OsStr::new("--verbose") => {
                    if let Some(extra) = args.next() {
                        return Err(unknown_option("알 수 없는 --version 옵션", &extra));
                    }
                    true
                }
                Some(flag) => return Err(unknown_option("알 수 없는 --version 옵션", &flag)),
            };
            let mut out = io::stdout().lock();
            writeln!(out, "{} {}", build_info::APP_NAME, build_info::APP_VERSION)?;
            if verbose {
                writeln!(out, "target: {}", build_info::BUILD_TARGET)?;
                writeln!(out, "profile: {}", build_info::BUILD_PROFILE)?;
                writeln!(out, "rustc: {}", build_info::BUILD_RUSTC)?;
                writeln!(out, "git: {}", build_info::BUILD_GIT_SHA)?;
                writeln!(out, "dirty: {}", build_info::BUILD_GIT_DIRTY)?;
                writeln!(out, "rng backend: {}", build_info::RNG_BACKEND)?;
            }
            return Ok(ExitCode::SUCCESS);
        }
        return Err(unknown_option("알 수 없는 옵션", &first));
    }
    let output_file = OutputFile::try_from(Path::new(FILE_NAME))?;
    let file_mutex = Mutex::new(BufWriter::from(output_file));
    let reserved_string = |capacity: usize, context: &'static str| -> Result<String> {
        let mut value = String::new();
        value
            .try_reserve_exact(capacity)
            .map_err(|source| AppError::context(context, source))?;
        Ok(value)
    };
    let input_buffer =
        reserved_string(GENERIC_INPUT_BUFFER_CAPACITY, "입력 버퍼 메모리 확보 실패")?;
    let mut app = cfg_select! {
        target_arch = "x86_64" => {{
            let mut rng = HardwareRng::new();
            let num_64 = match rng.source() {
                HardwareRandomSource::RdSeed => {
                    let data = generate_random_data_with_rng(&mut rng)?;
                    if rng.take_rdseed_fallback_notice() {
                        let mut err = stderr().lock();
                        err.write_all("RDSEED 5분 타임아웃으로 RDRAND로 전환했습니다.\n".as_bytes())?;
                    }
                    let num_64 = data.num_64;
                    persist_and_print_random_data(&file_mutex, &data)?;
                    num_64
                }
                HardwareRandomSource::RdRand => {
                    let mut err = stderr().lock();
                    err.write_all("RDSEED를 미지원하여 RDRAND를 사용합니다.\n".as_bytes())?;
                    let data = generate_random_data_with_rng(&mut rng)?;
                    let num_64 = data.num_64;
                    persist_and_print_random_data(&file_mutex, &data)?;
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
            let ladder_players_storage = reserved_string(
                GENERIC_INPUT_BUFFER_CAPACITY,
                "사다리 참여자 버퍼 메모리 확보 실패",
            )?;
            let ladder_results_storage = reserved_string(
                GENERIC_INPUT_BUFFER_CAPACITY,
                "사다리 결과 버퍼 메모리 확보 실패",
            )?;
            MenuApp::from(MenuAppInit {
                file_mutex,
                input_buffer,
                ladder_players_storage,
                ladder_results_storage,
                num_64,
                rng,
            })
        }}
        _ => {
            MenuApp::from(MenuAppInit { file_mutex, input_buffer })
        }
    };
    app.run()
}
