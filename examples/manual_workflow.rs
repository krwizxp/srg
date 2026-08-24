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
    let console_log = artifacts.join("srg-result-console.log");
    let random_data = artifacts.join("random_data.txt");
    let published_random_data = artifacts.join("srg-result-random_data.txt");
    fs::create_dir_all(artifacts)?;
    for path in [&random_data, &published_random_data] {
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
    let status = Command::new(fs::canonicalize(binary)?)
        .args(args)
        .current_dir(artifacts)
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .status()?;
    let log_bytes = fs::read(&console_log)?;
    let mut stdout = io::stdout().lock();
    stdout.write_all(&log_bytes)?;
    stdout.flush()?;
    status
        .success()
        .ok_or_else(|| io::Error::other("selected SRG action failed"))?;
    output_check
        .is_none_or(|expected| {
            log_bytes
                .windows(expected.len())
                .any(|window| window == expected.as_bytes())
        })
        .ok_or_else(|| io::Error::other("selected SRG action produced no expected output"))?;
    if output_check.is_none() {
        (fs::metadata(&random_data)?.len() > 3)
            .ok_or_else(|| io::Error::other("SRG generated no random data"))?;
        fs::rename(random_data, published_random_data)?;
    }
    Ok(())
}
