use crate::diagnostic::Result;
use core::{fmt::Arguments, mem, result::Result as CoreResult};
use std::io::{BufRead as _, Error as IoError, ErrorKind, Result as IoResult, Write, stdin};
cfg_select! {
    target_arch = "x86_64" => {
        use crate::diagnostic::AppError;
        use core::range::Range;
        use super::random_util::checked_add_one_usize;
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
                        ErrorKind::UnexpectedEof,
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
                .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "입력 범위 계산 실패"))?;
            let next_len = match bytes.len().checked_add(segment.len()) {
                Some(next_len) if next_len <= max_bytes => next_len,
                _ => {
                    stdin_lock.consume(take_len);
                    if !reached_line_end {
                        stdin_lock.skip_until(b'\n')?;
                    }
                    return Err(IoError::new(
                        ErrorKind::InvalidInput,
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
    *buffer =
        String::from_utf8(bytes).map_err(|source| IoError::new(ErrorKind::InvalidData, source))?;
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
    if raw.is_empty() || !matches!(radix, 10 | 16) {
        return None;
    }
    let digits = if radix == 10 {
        raw.strip_prefix('+').unwrap_or(raw)
    } else {
        raw
    };
    if digits.is_empty() {
        return None;
    }
    let radix_value = u64::from(radix);
    digits.bytes().try_fold(0_u64, |value, byte| {
        let digit = match byte {
            b'0'..=b'9' => u64::from(byte.wrapping_sub(b'0')),
            b'a'..=b'f' if radix_value == 16 => u64::from(byte.wrapping_sub(b'a').wrapping_add(10)),
            b'A'..=b'F' if radix_value == 16 => u64::from(byte.wrapping_sub(b'A').wrapping_add(10)),
            _ => return None,
        };
        value.checked_mul(radix_value)?.checked_add(digit)
    })
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
            input_buffer: &mut String,
            io: (&mut dyn Write, &mut dyn Write),
            storage: &'storage mut String,
            entries: &mut [&'storage str; N],
            mode: LadderEntryMode,
            index_error: &'static str,
        ) -> Result<usize> {
            let (out, err) = io;
            let index_err = || AppError::message(index_error);
            let count = loop {
                let line =
                    read_line_reuse_limited(prompt, input_buffer, out, LADDER_INPUT_LINE_MAX_BYTES)?;
                let mut count = 0_usize;
                let mut has_empty_entry = false;
                let mut players_overflowed = false;
                let mut trimmed_ranges: [Range<usize>; N] = [Range { start: 0, end: 0 }; N];
                let mut segment_start = 0_usize;
                for (boundary_end, has_separator) in line
                    .match_indices(',')
                    .map(|(idx, _)| (idx, true))
                    .chain([(line.len(), false)])
                {
                    count = checked_add_one_usize(count).ok_or_else(|| {
                        AppError::message(match mode {
                            LadderEntryMode::Players => "플레이어 수 계산 실패",
                            LadderEntryMode::Results { .. } => "결과 수 계산 실패",
                        })
                    })?;
                    match mode {
                        LadderEntryMode::Players if count > N => {
                            writeln!(err, "플레이어 수가 최대 {N}명을 초과했습니다.")?;
                            players_overflowed = true;
                            break;
                        }
                        LadderEntryMode::Results { expected_count } if count > expected_count => break,
                        LadderEntryMode::Players | LadderEntryMode::Results { .. } => {
                            let part = line
                                .get(segment_start..boundary_end)
                                .ok_or_else(index_err)?;
                            let leading_whitespace = part
                                .len()
                                .checked_sub(part.trim_start().len())
                                .ok_or_else(index_err)?;
                            let trimmed = part.trim();
                            if trimmed.is_empty() {
                                has_empty_entry = true;
                                break;
                            }
                            let entry_index = count.checked_sub(1).ok_or_else(index_err)?;
                            let slot = trimmed_ranges.get_mut(entry_index).ok_or_else(index_err)?;
                            let range_start = segment_start
                                .checked_add(leading_whitespace)
                                .ok_or_else(index_err)?;
                            let range_end = range_start
                                .checked_add(trimmed.len())
                                .ok_or_else(index_err)?;
                            *slot = Range {
                                start: range_start,
                                end: range_end,
                            };
                        }
                    }
                    if has_separator {
                        segment_start =
                            checked_add_one_usize(boundary_end).ok_or_else(index_err)?;
                    }
                }
                if players_overflowed || count == 0 {
                    continue;
                }
                if has_empty_entry {
                    let message = match mode {
                        LadderEntryMode::Players => "플레이어 이름은 비워둘 수 없습니다.",
                        LadderEntryMode::Results { .. } => "결과값은 비워둘 수 없습니다.",
                    };
                    writeln!(err, "{message}")?;
                    continue;
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
                storage.clear();
                storage.push_str(line);
                for (entry_index, range) in trimmed_ranges.iter().copied().enumerate().take(count) {
                    let part = storage.get(range).ok_or_else(index_err)?;
                    let slot = entries.get_mut(entry_index).ok_or_else(index_err)?;
                    *slot = part;
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
