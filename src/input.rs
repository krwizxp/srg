use crate::diagnostic::Result;
#[cfg(target_arch = "x86_64")]
use crate::ladder::MAX_LADDER_ENTRIES;
use core::{fmt::Arguments, mem, result::Result as CoreResult};
use std::io::{self, BufRead as _, Error as IoError, Result as IoResult, Write, stdin};
cfg_select! {
    target_arch = "x86_64" => {
        const LADDER_INPUT_LINE_MAX_BYTES: usize = 64 * 1024;
    }
    _ => {}
}
const DEFAULT_INPUT_LINE_MAX_BYTES: usize = 4096;
const HEX_INPUT_LINE_MAX_BYTES: usize = 256;
cfg_select! {
    target_arch = "x86_64" => {
        #[derive(Clone, Copy)]
        pub(super) enum LadderEntryMode {
            Players,
            Results { expected_count: usize },
        }
    }
    _ => {}
}
pub(super) fn read_line_reuse_limited<'buffer>(
    prompt: Arguments<'_>,
    buffer: &'buffer mut String,
    out: &mut dyn Write,
    max_bytes: usize,
) -> IoResult<&'buffer str> {
    out.write_fmt(prompt)?;
    out.flush()?;
    let mut bytes = mem::take(buffer).into_bytes();
    bytes.clear();
    let mut stdin_lock = stdin().lock();
    loop {
        let available = stdin_lock.fill_buf()?;
        if available.is_empty() {
            if bytes.is_empty() {
                return Err(IoError::new(
                    io::ErrorKind::UnexpectedEof,
                    "표준 입력이 종료되었습니다.",
                ));
            }
            break;
        }
        let line_end = available.iter().position(|&byte| byte == b'\n');
        let reached_line_end = line_end.is_some();
        let take_len = line_end.map_or(available.len(), |index| index.saturating_add(1));
        let segment = available
            .get(..take_len)
            .ok_or_else(|| IoError::new(io::ErrorKind::InvalidInput, "입력 범위 계산 실패"))?;
        if segment.len() > max_bytes.saturating_sub(bytes.len()) {
            stdin_lock.consume(take_len);
            if !reached_line_end {
                stdin_lock.skip_until(b'\n')?;
            }
            return Err(IoError::new(
                io::ErrorKind::InvalidInput,
                format!("입력이 너무 깁니다. 최대 {max_bytes} bytes까지 입력할 수 있습니다."),
            ));
        }
        bytes.try_reserve(segment.len()).map_err(IoError::other)?;
        bytes.extend_from_slice(segment);
        stdin_lock.consume(take_len);
        if reached_line_end {
            break;
        }
    }
    drop(stdin_lock);
    *buffer = String::from_utf8(bytes)
        .map_err(|source| IoError::new(io::ErrorKind::InvalidData, source))?;
    Ok(buffer.trim())
}
pub(super) fn read_u64_hex_input(
    prompt: Arguments<'_>,
    input_buffer: &mut String,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<u64> {
    loop {
        let raw = read_line_reuse_limited(prompt, input_buffer, out, HEX_INPUT_LINE_MAX_BYTES)?;
        let parsed_value = raw
            .strip_prefix('0')
            .and_then(|body| body.strip_prefix(['x', 'X']))
            .map_or_else(
                || raw.parse::<u64>().ok(),
                |hex| {
                    if hex.starts_with('+') {
                        None
                    } else {
                        u64::from_str_radix(hex, 16).ok()
                    }
                },
            );
        if let Some(value) = parsed_value {
            return Ok(value);
        }
        writeln!(
            err,
            "유효한 u64 형식이 아닙니다 (최소값 예: 0 또는 0x0, 최대값 예: {max_u64} 또는 0x{max_u64:X}).",
            max_u64 = u64::MAX
        )?;
    }
}
pub(super) fn get_validated_input<T, E, F>(
    prompt: &str,
    input_buf: &mut String,
    out: &mut dyn Write,
    mut validator: F,
) -> IoResult<T>
where
    E: AsRef<str>,
    F: FnMut(&str) -> CoreResult<T, E>,
{
    loop {
        let input = read_line_reuse_limited(
            format_args!("{prompt}"),
            input_buf,
            out,
            DEFAULT_INPUT_LINE_MAX_BYTES,
        )?;
        match validator(input) {
            Ok(value) => return Ok(value),
            Err(err) => {
                let err_text = err.as_ref();
                if !err_text.is_empty() {
                    writeln!(out, "{err_text}")?;
                }
            }
        }
    }
}
cfg_select! {
    target_arch = "x86_64" => {
        pub(super) fn parse_regular_f64(raw: &str) -> Option<f64> {
            raw.parse::<f64>()
                .ok()
                .filter(|value| value.is_finite() && !value.is_subnormal())
        }
        pub(super) fn read_ladder_entries(
            prompt: Arguments<'_>,
            io: (&mut dyn Write, &mut dyn Write),
            storage: &mut String,
            mode: LadderEntryMode,
        ) -> Result<usize> {
            let (out, err) = io;
            let count = 'read: loop {
                let line =
                    read_line_reuse_limited(prompt, storage, out, LADDER_INPUT_LINE_MAX_BYTES)?;
                let mut count = 0_usize;
                for part in line.split(',') {
                    count = count.saturating_add(1);
                    match mode {
                        LadderEntryMode::Players if count > MAX_LADDER_ENTRIES => {
                            writeln!(
                                err,
                                "플레이어 수가 최대 {MAX_LADDER_ENTRIES}명을 초과했습니다."
                            )?;
                            continue 'read;
                        }
                        LadderEntryMode::Results { expected_count } if count > expected_count => break,
                        LadderEntryMode::Players | LadderEntryMode::Results { .. } => {}
                    }
                    if part.trim().is_empty() {
                        let message = match mode {
                            LadderEntryMode::Players => "플레이어 이름은 비워둘 수 없습니다.",
                            LadderEntryMode::Results { .. } => "결과값은 비워둘 수 없습니다.",
                        };
                        writeln!(err, "{message}")?;
                        continue 'read;
                    }
                }
                match mode {
                    LadderEntryMode::Players if count < 2 => {
                        writeln!(err, "플레이어 수는 최소 2명이어야 합니다.")?;
                        continue;
                    }
                    LadderEntryMode::Results { expected_count } if count != expected_count => {
                        writeln!(
                            err,
                            "결과값의 개수({count})가 플레이어 수({expected_count})와 일치하지 않습니다.\n"
                        )?;
                        continue;
                    }
                    LadderEntryMode::Players | LadderEntryMode::Results { .. } => {}
                }
                break count;
            };
            Ok(count)
        }
        pub(super) fn read_parsed_value<T, F>(
            prompt: Arguments<'_>,
            buffer: &mut String,
            out: &mut dyn Write,
            err: &mut dyn Write,
            invalid_message: &str,
            mut parse: F,
        ) -> Result<T>
        where
            F: FnMut(&str) -> Option<T>,
        {
            loop {
                let line = read_line_reuse_limited(
                    prompt,
                    buffer,
                    out,
                    DEFAULT_INPUT_LINE_MAX_BYTES,
                )?;
                if let Some(value) = parse(line) {
                    return Ok(value);
                }
                writeln!(err, "{invalid_message}")?;
            }
        }
    }
    _ => {}
}
