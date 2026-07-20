use crate::diagnostic::Result;
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
    buffer.clear();
    out.write_fmt(prompt)?;
    out.flush()?;
    read_line_limited(buffer, max_bytes)?;
    Ok(buffer.trim())
}
fn read_line_limited(buffer: &mut String, max_bytes: usize) -> IoResult<()> {
    let mut bytes = mem::take(buffer).into_bytes();
    bytes.clear();
    {
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
            let next_len = match bytes.len().checked_add(segment.len()) {
                Some(next_len) if next_len <= max_bytes => next_len,
                _ => {
                    stdin_lock.consume(take_len);
                    if !reached_line_end {
                        stdin_lock.skip_until(b'\n')?;
                    }
                    return Err(IoError::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "입력이 너무 깁니다. 최대 {max_bytes} bytes까지 입력할 수 있습니다."
                        ),
                    ));
                }
            };
            if bytes.capacity() < next_len {
                bytes.try_reserve(segment.len()).map_err(IoError::other)?;
            }
            bytes.extend_from_slice(segment);
            stdin_lock.consume(take_len);
            if reached_line_end {
                drop(stdin_lock);
                break;
            }
        }
    }
    *buffer = String::from_utf8(bytes)
        .map_err(|source| IoError::new(io::ErrorKind::InvalidData, source))?;
    Ok(())
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
                || parse_u64_digits(raw, 10),
                |hex| parse_u64_digits(hex, 16),
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
pub(super) fn parse_u64_digits(raw: &str, radix: u32) -> Option<u64> {
    match radix {
        10 => raw.parse::<u64>().ok(),
        16 if !raw.starts_with('+') => u64::from_str_radix(raw, 16).ok(),
        _ => None,
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
        out.write_all(prompt.as_bytes())?;
        out.flush()?;
        input_buf.clear();
        read_line_limited(input_buf, DEFAULT_INPUT_LINE_MAX_BYTES)?;
        let trimmed = input_buf.trim();
        match validator(trimmed) {
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
        pub(super) fn read_ladder_entries<'storage, const N: usize>(
            prompt: Arguments<'_>,
            io: (&mut dyn Write, &mut dyn Write),
            storage: &'storage mut String,
            entries: &mut [&'storage str; N],
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
                        LadderEntryMode::Players if count > N => {
                            writeln!(err, "플레이어 수가 최대 {N}명을 초과했습니다.")?;
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
            for (slot, part) in entries
                .iter_mut()
                .take(count)
                .zip(storage.trim().split(',').map(str::trim))
            {
                *slot = part;
            }
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
