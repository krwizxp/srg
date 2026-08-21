use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
macro_rules! required_pair_args {
    ($command:literal, $first_env:literal, $second_env:literal) => {
        vec![
            OsString::from($command),
            required_env($first_env)?,
            required_env($second_env)?,
        ]
    };
}
fn invalid_input(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
fn required_env(name: &'static str) -> io::Result<OsString> {
    env::var_os(name)
        .ok_or_else(|| invalid_input("required workflow input environment variable is missing"))
}
fn main() -> io::Result<()> {
    let action = required_env("SRG_ACTION")?;
    let (args, output_check) = match action.to_str() {
        Some("generate-single") => (vec![OsString::from("generate"), OsString::from("1")], None),
        Some("generate-multiple") => (
            vec![OsString::from("generate"), required_env("SRG_COUNT")?],
            None,
        ),
        Some("ladder") => (
            required_pair_args!("ladder", "SRG_PLAYERS", "SRG_RESULTS"),
            Some("사다리타기 결과:"),
        ),
        Some("random-integer") => (
            required_pair_args!("random-integer", "SRG_INT_MIN", "SRG_INT_MAX"),
            Some("무작위 정수("),
        ),
        Some("random-float") => (
            required_pair_args!("random-float", "SRG_FLOAT_MIN", "SRG_FLOAT_MAX"),
            Some("무작위 실수("),
        ),
        Some("time-sync-observe") => (
            required_pair_args!("time-observe", "SRG_TIME_HOST", "SRG_OBSERVE_SECONDS"),
            Some("서버 시간:"),
        ),
        Some(_) | None => return Err(invalid_input("unsupported SRG workflow action")),
    };
    let artifacts = Path::new("artifacts");
    let random_data = Path::new("random_data.txt");
    let console_log = artifacts.join("srg-result-console.log");
    let copied_random_data = artifacts.join("srg-result-random_data.txt");
    fs::create_dir_all(artifacts)?;
    for path in [random_data, copied_random_data.as_path()] {
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(source) if source.kind() == io::ErrorKind::NotFound => {}
            Err(source) => return Err(source),
        }
    }
    let log = File::create(&console_log)?;
    let mut binary = env::var_os("CARGO_TARGET_DIR")
        .map_or_else(|| PathBuf::from("target"), PathBuf::from)
        .join("release")
        .join(env!("CARGO_PKG_NAME"));
    binary.add_extension(env::consts::EXE_EXTENSION);
    let status = Command::new(binary)
        .args(args)
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .status()?;
    let log_bytes = fs::read(&console_log)?;
    let mut stdout = io::stdout().lock();
    stdout.write_all(&log_bytes)?;
    stdout.flush()?;
    if !status.success() {
        return Err(io::Error::other("selected SRG action failed"));
    }
    if let Some(expected) = output_check
        && !log_bytes
            .windows(expected.len())
            .any(|window| window == expected.as_bytes())
    {
        return Err(io::Error::other(
            "selected SRG action produced no expected output",
        ));
    }
    let source_len = match fs::metadata(random_data) {
        Ok(metadata) => metadata.len(),
        Err(source) if source.kind() == io::ErrorKind::NotFound => {
            return output_check.map_or_else(
                || Err(io::Error::other("SRG created no random data output file")),
                |_| Ok(()),
            );
        }
        Err(source) => return Err(source),
    };
    if output_check.is_none() && source_len <= 3 {
        return Err(io::Error::other("SRG generated no random data"));
    }
    if fs::copy(random_data, copied_random_data)? != source_len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "workflow output changed while copying",
        ));
    }
    Ok(())
}
