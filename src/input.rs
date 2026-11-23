use crate::diagnostic::Result;
#[cfg(target_arch = "x86_64")]
use crate::{
    batch::MAX_BATCH_GENERATE_COUNT,
    ladder::{MAX_LADDER_ENTRIES, MAX_LADDER_INPUT_BYTES},
};
#[cfg(target_arch = "x86_64")]
use alloc::borrow::Cow;
use core::{fmt::Arguments, mem, result::Result as CoreResult};
use std::io::{self, BufRead as _, Error as IoError, Result as IoResult, Write, stdin};
#[cfg(target_arch = "x86_64")]
pub(super) const BATCH_COUNT_INPUT_MAX_BYTES: usize = 64;
pub(super) const DEFAULT_INPUT_LINE_MAX_BYTES: usize = 4096;
pub(super) fn validate_input_line(text: &str, max_bytes: usize) -> IoResult<&str> {
    if text.len() > max_bytes {
        return Err(IoError::new(
            io::ErrorKind::InvalidInput,
            format!("입력이 너무 깁니다. 최대 {max_bytes} bytes까지 입력할 수 있습니다."),
        ));
    }
    if text.contains(['\r', '\n']) {
        return Err(IoError::new(
            io::ErrorKind::InvalidInput,
            "값은 한 줄로 입력해야 합니다.",
        ));
    }
    Ok(text.trim())
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
            (!bytes.is_empty()).ok_or_else(|| {
                IoError::new(io::ErrorKind::UnexpectedEof, "표준 입력이 종료되었습니다.")
            })?;
            break;
        }
        let line_end = available.iter().position(|&byte| byte == b'\n');
        let take_len = line_end.map_or(available.len(), |index| index.strict_add(1));
        let (segment, _) = available.split_at(take_len);
        if segment.len() > max_bytes.strict_add(2).strict_sub(bytes.len()) {
            stdin_lock.consume(take_len);
            if line_end.is_none() {
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
        if line_end.is_some() {
            break;
        }
    }
    drop(stdin_lock);
    if bytes.pop_if(|byte| *byte == b'\n').is_some() {
        bytes.pop_if(|byte| *byte == b'\r');
    }
    *buffer = String::from_utf8(bytes)
        .map_err(|source| IoError::new(io::ErrorKind::InvalidData, source))?;
    validate_input_line(buffer, max_bytes)
}
pub(super) fn read_u64_hex_input(
    prompt: Arguments<'_>,
    input_buffer: &mut String,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<u64> {
    get_validated_input(prompt, input_buffer, out, Some(err), 256, |raw| {
        let parsed = match raw
            .strip_prefix('0')
            .and_then(|body| body.strip_prefix(['x', 'X']))
        {
            Some(hex) if !hex.starts_with('+') => u64::from_str_radix(hex, 16).ok(),
            Some(_) => None,
            None => raw.parse::<u64>().ok(),
        };
        parsed.ok_or_else(|| {
            format!(
                "유효한 u64 형식이 아닙니다 (최소값 예: 0 또는 0x0, 최대값 예: {max_u64} 또는 0x{max_u64:X}).",
                max_u64 = u64::MAX
            )
        })
    })
    .map_err(Into::into)
}
pub(super) fn get_validated_input<'writer, T, E>(
    prompt: Arguments<'_>,
    input_buf: &mut String,
    out: &'writer mut dyn Write,
    mut err_out: Option<&'writer mut dyn Write>,
    max_bytes: usize,
    mut validator: impl FnMut(&str) -> CoreResult<T, E>,
) -> IoResult<T>
where
    E: AsRef<str>,
{
    loop {
        let input = read_line_reuse_limited(prompt, input_buf, out, max_bytes)?;
        match validator(input) {
            Ok(value) => return Ok(value),
            Err(err) => {
                let err_text = err.as_ref();
                if !err_text.is_empty() {
                    let target = err_out.as_mut().map_or(&mut *out, |writer| &mut **writer);
                    target.write_all(err_text.as_bytes())?;
                    target.write_all(b"\n")?;
                }
            }
        }
    }
}
#[cfg(target_arch = "x86_64")]
pub(super) fn parse_regular_f64(raw: &str) -> Option<f64> {
    raw.parse::<f64>()
        .ok()
        .filter(|value| value.is_finite() && !value.is_subnormal())
}
#[cfg(target_arch = "x86_64")]
pub(super) fn validate_batch_count(count: usize) -> CoreResult<usize, Cow<'static, str>> {
    if count == 0 {
        Err(Cow::Borrowed("1 이상의 값을 입력해 주세요."))
    } else if count > MAX_BATCH_GENERATE_COUNT {
        Err(Cow::Owned(format!(
            "대량 생성 개수는 최대 {MAX_BATCH_GENERATE_COUNT}건까지 입력할 수 있습니다."
        )))
    } else {
        Ok(count)
    }
}
#[cfg(target_arch = "x86_64")]
fn validate_ladder_csv(
    line: &str,
    expected_count: Option<usize>,
) -> CoreResult<usize, Cow<'static, str>> {
    let mut count = 0_usize;
    for part in line.split(',') {
        count = count.strict_add(1);
        match expected_count {
            None if count > MAX_LADDER_ENTRIES => {
                return Err(Cow::Owned(format!(
                    "플레이어 수가 최대 {MAX_LADDER_ENTRIES}명을 초과했습니다."
                )));
            }
            Some(expected) if count > expected => break,
            None | Some(_) => {}
        }
        if part.trim().is_empty() {
            return Err(Cow::Borrowed(if expected_count.is_none() {
                "플레이어 이름은 비워둘 수 없습니다."
            } else {
                "결과값은 비워둘 수 없습니다."
            }));
        }
    }
    match expected_count {
        None if count < 2 => Err(Cow::Borrowed("플레이어 수는 최소 2명이어야 합니다.")),
        Some(expected) if count != expected => Err(Cow::Owned(format!(
            "결과값의 개수({count})가 플레이어 수({expected})와 일치하지 않습니다.\n"
        ))),
        None | Some(_) => Ok(count),
    }
}
#[cfg(target_arch = "x86_64")]
pub(super) fn read_ladder_entries(
    prompt: Arguments<'_>,
    (out, err): (&mut dyn Write, &mut dyn Write),
    storage: &mut String,
    expected_count: Option<usize>,
) -> Result<usize> {
    get_validated_input(
        prompt,
        storage,
        out,
        Some(err),
        MAX_LADDER_INPUT_BYTES,
        |line| validate_ladder_csv(line, expected_count),
    )
    .map_err(Into::into)
}
