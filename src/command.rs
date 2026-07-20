use crate::{
    diagnostic::{AppError, Result},
    time::{ParsedServer, ServerTimeSession},
};
cfg_select! {
    target_arch = "x86_64" => {
        use crate::{
            batch::{MAX_BATCH_GENERATE_COUNT, regenerate_with_count},
            file_output::OutputFile,
            hardware_rng::{HardwareRandomSource, HardwareRng},
            ladder::{MAX_LADDER_ENTRIES, write_ladder_results},
            random_number::{generate_random_float, generate_random_integer, validate_random_float_range, validate_random_integer_range},
            FILE_NAME,
        };
        use std::{io::Write, path::Path};
    }
    _ => {
        use std::io::Write as _;
    }
}
use core::{fmt::Display, str::FromStr, time::Duration};
use std::{
    ffi::{OsStr, OsString},
    io,
    process::ExitCode,
};
pub(super) enum CliCommand {
    #[cfg(target_arch = "x86_64")]
    Generate {
        count: u64,
    },
    #[cfg(target_arch = "x86_64")]
    Ladder {
        players: String,
        results: String,
    },
    #[cfg(target_arch = "x86_64")]
    RandomFloat {
        max: f64,
        min: f64,
    },
    #[cfg(target_arch = "x86_64")]
    RandomInteger {
        max: i64,
        min: i64,
    },
    TimeObserve {
        host: ParsedServer,
        seconds: u64,
    },
}
impl CliCommand {
    pub(super) fn execute(self) -> Result<ExitCode> {
        let mut out = io::stdout().lock();
        let mut err = io::stderr().lock();
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::Generate { count } => {
                let rng = Self::hardware_rng(&mut err)?;
                let mut output_file = OutputFile::try_from(Path::new(FILE_NAME))?;
                regenerate_with_count(&mut output_file, &rng, count, &mut out)?;
                Self::write_rdseed_fallback_notice(&rng, &mut err)?;
            }
            #[cfg(target_arch = "x86_64")]
            Self::Ladder { players, results } => {
                let rng = Self::hardware_rng(&mut err)?;
                let seed = rng.next_u64()?;
                write_ladder_results(
                    players.split(',').map(str::trim),
                    results.split(',').map(str::trim),
                    seed,
                    &rng,
                    &mut out,
                )?;
                Self::write_rdseed_fallback_notice(&rng, &mut err)?;
            }
            #[cfg(target_arch = "x86_64")]
            Self::RandomFloat { max, min } => {
                let rng = Self::hardware_rng(&mut err)?;
                let seed = rng.next_u64()?;
                generate_random_float(min, max, seed, &mut out, &rng)?;
                Self::write_rdseed_fallback_notice(&rng, &mut err)?;
            }
            #[cfg(target_arch = "x86_64")]
            Self::RandomInteger { max, min } => {
                let rng = Self::hardware_rng(&mut err)?;
                let seed = rng.next_u64()?;
                generate_random_integer(min, max, seed, &mut out, &rng)?;
                Self::write_rdseed_fallback_notice(&rng, &mut err)?;
            }
            Self::TimeObserve { host, seconds } => {
                ServerTimeSession {
                    host,
                    scheduled_trigger: None,
                    stop_after: Some(Duration::from_secs(seconds)),
                }
                .run_loop(&mut out, &mut err)?;
                writeln!(out, "\n서버 시간 확인을 종료합니다.")?;
            }
        }
        Ok(ExitCode::SUCCESS)
    }
    #[cfg(target_arch = "x86_64")]
    fn hardware_rng(err: &mut dyn Write) -> Result<HardwareRng> {
        let rng = HardwareRng::new();
        match rng.source() {
            HardwareRandomSource::None => Err("RDSEED·RDRAND를 지원하지 않는 CPU입니다.".into()),
            HardwareRandomSource::RdRand => {
                writeln!(err, "RDSEED를 미지원하여 RDRAND를 사용합니다.")?;
                Ok(rng)
            }
            HardwareRandomSource::RdSeed => Ok(rng),
        }
    }
    #[cfg(target_arch = "x86_64")]
    fn owned_text(arg: OsString, label: &str) -> Result<String> {
        let Ok(text) = arg.into_string() else {
            return Err(AppError::message(format!(
                "{label} 값은 유효한 Unicode여야 합니다."
            )));
        };
        Ok(text)
    }
    fn parse_arg<T>(arg: &OsStr, label: &str) -> Result<T>
    where
        T: FromStr,
        T::Err: Display,
    {
        let text = arg
            .to_str()
            .ok_or_else(|| AppError::message(format!("{label} 값은 유효한 Unicode여야 합니다.")))?;
        text.parse::<T>().map_err(|source| {
            AppError::message(format!("{label} 값이 올바르지 않습니다: {source}"))
        })
    }
    #[cfg(target_arch = "x86_64")]
    fn take_two_args<I>(args: &mut I, usage: &str) -> Result<(OsString, OsString)>
    where
        I: Iterator<Item = OsString>,
    {
        let (Some(first), Some(second), None) = (args.next(), args.next(), args.next()) else {
            return Err(AppError::message(format!("사용법: srg {usage}")));
        };
        Ok((first, second))
    }
    #[cfg(target_arch = "x86_64")]
    fn write_rdseed_fallback_notice(rng: &HardwareRng, err: &mut dyn Write) -> Result<()> {
        if rng.take_rdseed_fallback_notice() {
            writeln!(err, "RDSEED 5분 타임아웃으로 RDRAND로 전환했습니다.")?;
        }
        Ok(())
    }
}
impl<I> TryFrom<(OsString, I)> for CliCommand
where
    I: Iterator<Item = OsString>,
{
    type Error = AppError;
    fn try_from((raw_command, mut args): (OsString, I)) -> Result<Self> {
        const INVALID_COMMAND_NAME: &str = "명령 이름은 유효한 Unicode여야 합니다.";
        let command = raw_command.to_str().ok_or(INVALID_COMMAND_NAME)?;
        match command {
            #[cfg(target_arch = "x86_64")]
            "generate" => {
                let (Some(count_arg), None) = (args.next(), args.next()) else {
                    return Err(AppError::message("사용법: srg generate <count>"));
                };
                let count = Self::parse_arg::<u64>(&count_arg, "count")?;
                if !(1..=MAX_BATCH_GENERATE_COUNT).contains(&count) {
                    return Err(AppError::message(format!(
                        "count는 1~{MAX_BATCH_GENERATE_COUNT} 범위여야 합니다."
                    )));
                }
                Ok(Self::Generate { count })
            }
            #[cfg(target_arch = "x86_64")]
            "ladder" => {
                let (players_arg, results_arg) =
                    Self::take_two_args(&mut args, "ladder <players-csv> <results-csv>")?;
                let players = Self::owned_text(players_arg, "players-csv")?;
                let results = Self::owned_text(results_arg, "results-csv")?;
                if players.contains(['\r', '\n']) || results.contains(['\r', '\n']) {
                    return Err("플레이어와 결과값은 한 줄로 입력해야 합니다.".into());
                }
                let player_count = players.split(',').count();
                if !(2..=MAX_LADDER_ENTRIES).contains(&player_count) {
                    return Err(AppError::message(format!(
                        "플레이어는 2~{MAX_LADDER_ENTRIES}명이어야 합니다."
                    )));
                }
                if player_count != results.split(',').count() {
                    return Err("결과값 개수는 플레이어 수와 같아야 합니다.".into());
                }
                if players
                    .split(',')
                    .chain(results.split(','))
                    .any(|entry| entry.trim().is_empty())
                {
                    return Err("플레이어와 결과값은 비워둘 수 없습니다.".into());
                }
                Ok(Self::Ladder { players, results })
            }
            #[cfg(target_arch = "x86_64")]
            "random-float" => {
                let (min_arg, max_arg) =
                    Self::take_two_args(&mut args, "random-float <min> <max>")?;
                let min = Self::parse_arg::<f64>(&min_arg, "min")?;
                let max = Self::parse_arg::<f64>(&max_arg, "max")?;
                validate_random_float_range(min, max)?;
                Ok(Self::RandomFloat { max, min })
            }
            #[cfg(target_arch = "x86_64")]
            "random-integer" => {
                let (min_arg, max_arg) =
                    Self::take_two_args(&mut args, "random-integer <min> <max>")?;
                let min = Self::parse_arg::<i64>(&min_arg, "min")?;
                let max = Self::parse_arg::<i64>(&max_arg, "max")?;
                validate_random_integer_range(min, max)?;
                Ok(Self::RandomInteger { max, min })
            }
            #[cfg(not(target_arch = "x86_64"))]
            "generate" | "ladder" | "random-float" | "random-integer" => {
                Err("이 명령은 x86_64의 RDSEED/RDRAND 지원 환경에서만 사용할 수 있습니다.".into())
            }
            "time-observe" => {
                let (Some(host_arg), Some(seconds_arg), None) =
                    (args.next(), args.next(), args.next())
                else {
                    return Err("사용법: srg time-observe <host> <seconds>".into());
                };
                let host = Self::parse_arg::<ParsedServer>(&host_arg, "host")?;
                let seconds = Self::parse_arg::<u64>(&seconds_arg, "seconds")?;
                if !(1..=60).contains(&seconds) {
                    return Err("seconds는 1~60 범위여야 합니다.".into());
                }
                Ok(Self::TimeObserve { host, seconds })
            }
            _ => Err(AppError::message(format!(
                "알 수 없는 옵션: {}",
                raw_command.to_string_lossy()
            ))),
        }
    }
}
