use crate::diagnostic::Result;
use core::{fmt::Arguments, result::Result as CoreResult};
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
        pub enum LadderEntryMode {
            Players,
            Results { expected_count: usize },
        }
    }
    _ => {}
}
pub fn read_line_reuse_limited<'buffer>(
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
    let bytes = {
        let mut stdin_lock = stdin().lock();
        let mut bytes = Vec::new();
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
            let take_len = line_end.map_or(available.len(), |index| index.saturating_add(1));
            let segment = available
                .get(..take_len)
                .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "입력 범위 계산 실패"))?;
            if bytes
                .len()
                .checked_add(segment.len())
                .is_none_or(|next_len| next_len > max_bytes)
            {
                stdin_lock.consume(take_len);
                if line_end.is_none() {
                    loop {
                        let discard_available = stdin_lock.fill_buf()?;
                        if discard_available.is_empty() {
                            break;
                        }
                        let extra_line_end =
                            discard_available.iter().position(|&byte| byte == b'\n');
                        let extra_take_len = extra_line_end
                            .map_or(discard_available.len(), |index| index.saturating_add(1));
                        stdin_lock.consume(extra_take_len);
                        if extra_line_end.is_some() {
                            break;
                        }
                    }
                }
                return Err(IoError::new(
                    ErrorKind::InvalidInput,
                    format!("입력이 너무 깁니다. 최대 {max_bytes} bytes까지 입력할 수 있습니다."),
                ));
            }
            bytes.try_reserve(segment.len()).map_err(IoError::other)?;
            bytes.extend_from_slice(segment);
            stdin_lock.consume(take_len);
            if line_end.is_some() {
                drop(stdin_lock);
                break;
            }
        }
        bytes
    };
    *buffer =
        String::from_utf8(bytes).map_err(|source| IoError::new(ErrorKind::InvalidData, source))?;
    Ok(())
}
pub fn read_u64_hex_input(
    prompt: Arguments<'_>,
    input_buffer: &mut String,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<u64> {
    loop {
        let raw = read_line_reuse_limited(prompt, input_buffer, out, HEX_INPUT_LINE_MAX_BYTES)?;
        match raw
            .strip_prefix('0')
            .and_then(|body| body.strip_prefix(['x', 'X']))
            .map_or_else(|| raw.parse::<u64>(), |hex| u64::from_str_radix(hex, 16))
        {
            Ok(value) => return Ok(value),
            Err(_) => {
                writeln!(
                    err,
                    "유효한 u64 형식이 아닙니다 (최소값 예: 0 또는 0x0, 최대값 예: {max_u64} 또는 0x{max_u64:X}).",
                    max_u64 = u64::MAX
                )?;
            }
        }
    }
}
pub fn get_validated_input<T, E, F>(
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
        pub fn parse_regular_f64(raw: &str) -> Option<f64> {
            raw.parse::<f64>()
                .ok()
                .filter(|value| value.is_finite() && !value.is_subnormal())
        }
        pub fn read_ladder_entries<'storage, const N: usize>(
            prompt: Arguments<'_>,
            input_buffer: &mut String,
            io: (&mut dyn Write, &mut dyn Write),
            storage: &'storage mut String,
            entries: &mut [&'storage str; N],
            mode: LadderEntryMode,
            index_error: &'static str,
        ) -> Result<usize> {
            struct SegmentBoundary {
                end: usize,
                has_separator: bool,
            }
            let (out, err) = io;
            let index_err = || AppError::message(index_error);
            let count = loop {
                let line =
                    read_line_reuse_limited(prompt, input_buffer, out, LADDER_INPUT_LINE_MAX_BYTES)?;
                let mut count = 0_usize;
                let mut players_overflowed = false;
                let mut trimmed_ranges: [Range<usize>; N] = [Range { start: 0, end: 0 }; N];
                let mut segment_start = 0_usize;
                for boundary in line
                    .match_indices(',')
                    .map(|(idx, _)| SegmentBoundary {
                        end: idx,
                        has_separator: true,
                    })
                    .chain([SegmentBoundary {
                        end: line.len(),
                        has_separator: false,
                    }])
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
                                .get(segment_start..boundary.end)
                                .ok_or_else(index_err)?;
                            let leading_whitespace = part
                                .len()
                                .checked_sub(part.trim_start().len())
                                .ok_or_else(index_err)?;
                            let trimmed = part.trim();
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
                    if boundary.has_separator {
                        segment_start =
                            checked_add_one_usize(boundary.end).ok_or_else(index_err)?;
                    }
                }
                if players_overflowed || count == 0 {
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
        pub fn read_parsed_value<T, F>(
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
