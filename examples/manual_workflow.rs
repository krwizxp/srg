use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Write as _};
use std::path::Path;
use std::process::{Command, Stdio};
#[derive(Clone, Copy)]
enum OutputCheck {
    Console(&'static str),
    RandomData,
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
        Some("generate-single") => (
            vec![OsString::from("generate"), OsString::from("1")],
            OutputCheck::RandomData,
        ),
        Some("generate-multiple") => (
            vec![OsString::from("generate"), required_env("SRG_COUNT")?],
            OutputCheck::RandomData,
        ),
        Some("ladder") => (
            vec![
                OsString::from("ladder"),
                required_env("SRG_PLAYERS")?,
                required_env("SRG_RESULTS")?,
            ],
            OutputCheck::Console("사다리타기 결과:"),
        ),
        Some("random-integer") => (
            vec![
                OsString::from("random-integer"),
                required_env("SRG_INT_MIN")?,
                required_env("SRG_INT_MAX")?,
            ],
            OutputCheck::Console("무작위 정수("),
        ),
        Some("random-float") => (
            vec![
                OsString::from("random-float"),
                required_env("SRG_FLOAT_MIN")?,
                required_env("SRG_FLOAT_MAX")?,
            ],
            OutputCheck::Console("무작위 실수("),
        ),
        Some("time-sync-observe") => (
            vec![
                OsString::from("time-observe"),
                required_env("SRG_TIME_HOST")?,
                required_env("SRG_OBSERVE_SECONDS")?,
            ],
            OutputCheck::Console("서버 시간:"),
        ),
        Some(_) | None => return Err(invalid_input("unsupported SRG workflow action")),
    };
    let artifacts = Path::new("artifacts");
    let random_data = Path::new("random_data.txt");
    let console_log = artifacts.join("srg-result-console.log");
    let copied_random_data = artifacts.join("srg-result-random_data.txt");
    fs::create_dir_all(artifacts)?;
    if random_data.try_exists()? {
        fs::remove_file(random_data)?;
    }
    if copied_random_data.try_exists()? {
        fs::remove_file(&copied_random_data)?;
    }
    let log = File::create(&console_log)?;
    let status = Command::new("target/release/srg")
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
    if let OutputCheck::Console(expected) = output_check
        && !log_bytes
            .windows(expected.len())
            .any(|window| window == expected.as_bytes())
    {
        return Err(io::Error::other(
            "selected SRG action produced no expected output",
        ));
    }
    if !random_data.try_exists()? {
        return match output_check {
            OutputCheck::Console(_) => Ok(()),
            OutputCheck::RandomData => {
                Err(io::Error::other("SRG created no random data output file"))
            }
        };
    }
    let source_len = random_data.metadata()?.len();
    if matches!(output_check, OutputCheck::RandomData) && source_len <= 3 {
        return Err(io::Error::other("SRG generated no random data"));
    }
    let copied = fs::copy(random_data, copied_random_data)?;
    if copied != source_len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "workflow output changed while copying",
        ));
    }
    Ok(())
}
