use super::Result;
use core::{fmt::Arguments, result::Result as CoreResult};
use std::io::{Error as IoError, ErrorKind, Result as IoResult, Write, stdin};
cfg_select! {
    target_arch = "x86_64" => {
        use super::random_util::checked_add_one_usize;
        #[derive(Clone, Copy)]
        pub enum LadderEntryMode {
            Players,
            Results { expected_count: usize },
        }
    }
    _ => {}
}
pub fn read_line_reuse<'buffer>(
    prompt: Arguments<'_>,
    buffer: &'buffer mut String,
    out: &mut dyn Write,
) -> IoResult<&'buffer str> {
    buffer.clear();
    out.write_fmt(prompt)?;
    out.flush()?;
    let bytes_read = stdin().read_line(buffer)?;
    if bytes_read == 0 {
        return Err(IoError::new(
            ErrorKind::UnexpectedEof,
            "표준 입력이 종료되었습니다.",
        ));
    }
    Ok(buffer.trim())
}
pub fn read_u64_hex_input(
    prompt: Arguments<'_>,
    input_buffer: &mut String,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<u64> {
    loop {
        let raw = read_line_reuse(prompt, input_buffer, out)?;
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
        let bytes_read = stdin().read_line(input_buf)?;
        if bytes_read == 0 {
            return Err(IoError::new(
                ErrorKind::UnexpectedEof,
                "표준 입력이 종료되었습니다.",
            ));
        }
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
            let (out, err) = io;
            let index_err = || IoError::other(index_error);
            let count = loop {
                let line = read_line_reuse(prompt, input_buffer, out)?;
                let mut count = 0_usize;
                let mut players_overflowed = false;
                let mut trimmed_ranges = [(0_usize, 0_usize); N];
                let mut segment_start = 0_usize;
                for (segment_end, separator) in line
                    .match_indices(',')
                    .map(|(idx, _)| (idx, true))
                    .chain([(line.len(), false)])
                {
                    count = checked_add_one_usize(count).ok_or_else(|| {
                        IoError::other(match mode {
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
                            let part = line.get(segment_start..segment_end).ok_or_else(index_err)?;
                            let leading_whitespace =
                                part.len().saturating_sub(part.trim_start().len());
                            let trimmed = part.trim();
                            let entry_index = count.checked_sub(1).ok_or_else(index_err)?;
                            let slot = trimmed_ranges.get_mut(entry_index).ok_or_else(index_err)?;
                            let range_start = segment_start
                                .checked_add(leading_whitespace)
                                .ok_or_else(index_err)?;
                            let range_end = range_start
                                .checked_add(trimmed.len())
                                .ok_or_else(index_err)?;
                            *slot = (range_start, range_end);
                        }
                    }
                    if separator {
                        segment_start = checked_add_one_usize(segment_end).ok_or_else(index_err)?;
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
                for (entry_index, (range_start, range_end)) in
                    trimmed_ranges.iter().copied().enumerate().take(count)
                {
                    let part = storage.get(range_start..range_end).ok_or_else(index_err)?;
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
                let line = read_line_reuse(prompt, buffer, out)?;
                if let Some(value) = parse(line) {
                    return Ok(value);
                }
                writeln!(err, "{invalid_message}")?;
            }
        }
    }
    _ => {}
}
