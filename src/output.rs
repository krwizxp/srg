use crate::{
    BAR_FULL, BAR_WIDTH_U64, BIN8_TABLE, BUFFER_SIZE, HEX_BYTE_TABLE, HEX_UPPER, INVALID_TIME,
    IS_TERMINAL, RandomDataSet, Result,
    buffmt::{ByteCursor, DIGITS, TWO_DIGITS, write_zero_err},
    numeric::{low_u8_from_u32, low_u8_from_u64, low_u8_from_u128, low_u16_from_u64},
    u64_to_be_bytes,
};
use std::{
    fmt::Arguments,
    fs::File,
    io::{BufWriter, Error as ioErr, ErrorKind, Result as IoRst, Write as _, stdout},
    sync::MutexGuard,
    time::Instant,
};

pub fn format_data_into_buffer(
    data: &RandomDataSet,
    buffer: &mut [u8; BUFFER_SIZE],
    use_colors: bool,
) -> Result<usize> {
    Ok(format_output(buffer.as_mut_slice(), data, use_colors)?)
}

type BufCursor<'buffer> = ByteCursor<'buffer>;

fn cursor_write_fmt(cur: &mut BufCursor<'_>, args: Arguments<'_>) -> IoRst<()> {
    let mut slice = cur.tail_mut();
    let before = slice.len();
    let res = slice.write_fmt(args);
    let after = slice.len();
    cur.advance_by(before - after);
    res
}

fn buf_write_bytes(cur: &mut BufCursor<'_>, bytes: &[u8]) -> IoRst<()> {
    cur.write_bytes(bytes)
}

fn buf_write_byte(cur: &mut BufCursor<'_>, byte: u8) -> IoRst<()> {
    cur.write_byte(byte)
}

pub fn prefix_slice(slice: &[u8], len: usize) -> IoRst<&[u8]> {
    slice.get(..len).ok_or_else(write_zero_err)
}

fn prefix_slice_mut(slice: &mut [u8], len: usize) -> IoRst<&mut [u8]> {
    slice.get_mut(..len).ok_or_else(write_zero_err)
}

fn range_slice_mut(slice: &mut [u8], start: usize, len: usize) -> IoRst<&mut [u8]> {
    let end = start.checked_add(len).ok_or_else(write_zero_err)?;
    slice.get_mut(start..end).ok_or_else(write_zero_err)
}

fn digit_byte(index: usize) -> IoRst<u8> {
    DIGITS.get(index).copied().ok_or_else(write_zero_err)
}

fn two_digits(index: usize) -> IoRst<&'static [u8; 2]> {
    TWO_DIGITS.get(index).ok_or_else(write_zero_err)
}

fn bin8_bytes(index: usize) -> IoRst<&'static [u8; 8]> {
    BIN8_TABLE.get(index).ok_or_else(write_zero_err)
}

fn hex_byte(index: usize) -> IoRst<&'static [u8; 2]> {
    HEX_BYTE_TABLE.get(index).ok_or_else(write_zero_err)
}

fn buf_write_chars<const N: usize>(cur: &mut BufCursor<'_>, chars: &[char; N]) -> IoRst<()> {
    let mut total = 0_usize;
    let mut i = 0_usize;
    while i < N {
        let ch = chars.get(i).ok_or_else(write_zero_err)?;
        total += ch.len_utf8();
        i += 1;
    }
    let head = cur.take(total)?;
    let mut pos = 0_usize;
    let mut j = 0_usize;
    while j < N {
        let ch = chars.get(j).ok_or_else(write_zero_err)?;
        let tail = head.get_mut(pos..).ok_or_else(write_zero_err)?;
        let written = ch.encode_utf8(tail).len();
        pos += written;
        j += 1;
    }
    Ok(())
}

fn buf_write_u8_dec(cur: &mut BufCursor<'_>, n: u8) -> IoRst<()> {
    if n >= 100 {
        let hundreds = usize::from(n / 100);
        let rem = usize::from(n % 100);
        let head = cur.take(3)?;
        let (hundreds_digit, remaining_digits) = head.split_at_mut(1);
        if let Some(slot) = hundreds_digit.first_mut() {
            *slot = digit_byte(hundreds)?;
        }
        remaining_digits.copy_from_slice(two_digits(rem)?);
        return Ok(());
    }
    if n >= 10 {
        let head = cur.take(2)?;
        head.copy_from_slice(two_digits(usize::from(n))?);
        return Ok(());
    }
    let head = cur.take(1)?;
    if let Some(slot) = head.first_mut() {
        *slot = b'0' + n;
    }
    Ok(())
}

fn buf_write_u8_array_spaced<const N: usize>(cur: &mut BufCursor<'_>, nums: &[u8; N]) -> IoRst<()> {
    let mut total = N.saturating_sub(1);
    let mut i = 0_usize;
    while i < N {
        let n = *nums.get(i).ok_or_else(write_zero_err)?;
        total += if n >= 100 {
            3
        } else if n >= 10 {
            2
        } else {
            1
        };
        i += 1;
    }
    let head = cur.take(total)?;
    let mut pos = 0_usize;
    let mut j = 0_usize;
    while j < N {
        if j != 0 {
            let slot = head.get_mut(pos).ok_or_else(write_zero_err)?;
            *slot = b' ';
            pos += 1;
        }
        let n = *nums.get(j).ok_or_else(write_zero_err)?;
        if n >= 100 {
            let hundreds = usize::from(n / 100);
            let rem = usize::from(n % 100);
            let slot = range_slice_mut(head, pos, 3)?;
            let (hundreds_digit, remaining_digits) = slot.split_at_mut(1);
            if let Some(digit_slot) = hundreds_digit.first_mut() {
                *digit_slot = digit_byte(hundreds)?;
            }
            remaining_digits.copy_from_slice(two_digits(rem)?);
            pos += 3;
        } else if n >= 10 {
            range_slice_mut(head, pos, 2)?.copy_from_slice(two_digits(usize::from(n))?);
            pos += 2;
        } else {
            let slot = head.get_mut(pos).ok_or_else(write_zero_err)?;
            *slot = b'0' + n;
            pos += 1;
        }
        j += 1;
    }
    Ok(())
}

fn buf_write_hash_hex24_from_bytes(cur: &mut BufCursor<'_>, b0: u8, b1: u8, b2: u8) -> IoRst<()> {
    let head = cur.take(7)?;
    let (prefix, hex_bytes) = head.split_at_mut(1);
    if let Some(slot) = prefix.first_mut() {
        *slot = b'#';
    }
    let (first_hex, remaining_hex_bytes) = hex_bytes.split_at_mut(2);
    first_hex.copy_from_slice(hex_byte(usize::from(b0))?);
    let (second_hex, third_hex) = remaining_hex_bytes.split_at_mut(2);
    second_hex.copy_from_slice(hex_byte(usize::from(b1))?);
    third_hex.copy_from_slice(hex_byte(usize::from(b2))?);
    Ok(())
}

fn buf_write_m_hash_hex24_from_bytes(cur: &mut BufCursor<'_>, b0: u8, b1: u8, b2: u8) -> IoRst<()> {
    let head = cur.take(8)?;
    let (prefix, hex_bytes) = head.split_at_mut(2);
    let (m_prefix, hash_prefix) = prefix.split_at_mut(1);
    if let Some(slot) = m_prefix.first_mut() {
        *slot = b'm';
    }
    if let Some(slot) = hash_prefix.first_mut() {
        *slot = b'#';
    }
    let (first_hex, remaining_hex_bytes) = hex_bytes.split_at_mut(2);
    first_hex.copy_from_slice(hex_byte(usize::from(b0))?);
    let (second_hex, third_hex) = remaining_hex_bytes.split_at_mut(2);
    second_hex.copy_from_slice(hex_byte(usize::from(b1))?);
    third_hex.copy_from_slice(hex_byte(usize::from(b2))?);
    Ok(())
}

fn buf_write_bin8_line(cur: &mut BufCursor<'_>, bytes: [u8; 8]) -> IoRst<()> {
    const PREFIX: &str = "2진수: ";
    const PREFIX_LEN: usize = PREFIX.len();
    const LINE_LEN: usize = PREFIX_LEN + 8 * 8 + 7 + 1;
    let head = cur.take(LINE_LEN)?;
    prefix_slice_mut(head, PREFIX_LEN)?.copy_from_slice(PREFIX.as_bytes());
    let mut pos = PREFIX_LEN;
    let mut i = 0_usize;
    while i < 8 {
        let b = usize::from(*bytes.get(i).ok_or_else(write_zero_err)?);
        range_slice_mut(head, pos, 8)?.copy_from_slice(bin8_bytes(b)?);
        pos += 8;
        let slot = head.get_mut(pos).ok_or_else(write_zero_err)?;
        *slot = if i == 7 { b'\n' } else { b' ' };
        pos += 1;
        i += 1;
    }
    Ok(())
}

fn buf_write_hex8_line(cur: &mut BufCursor<'_>, bytes: [u8; 8]) -> IoRst<()> {
    const PREFIX: &str = "16진수: ";
    const PREFIX_LEN: usize = PREFIX.len();
    const LINE_LEN: usize = PREFIX_LEN + 8 * 2 + 7 + 1;
    let head = cur.take(LINE_LEN)?;
    prefix_slice_mut(head, PREFIX_LEN)?.copy_from_slice(PREFIX.as_bytes());
    let mut pos = PREFIX_LEN;
    let mut i = 0_usize;
    while i < 8 {
        let b = usize::from(*bytes.get(i).ok_or_else(write_zero_err)?);
        range_slice_mut(head, pos, 2)?.copy_from_slice(hex_byte(b)?);
        pos += 2;
        let slot = head.get_mut(pos).ok_or_else(write_zero_err)?;
        *slot = if i == 7 { b'\n' } else { b' ' };
        pos += 1;
        i += 1;
    }
    Ok(())
}

fn buf_write_ascii8(cur: &mut BufCursor<'_>, chars: &[char; 8]) -> IoRst<()> {
    let head = cur.take(8)?;
    let mut i = 0_usize;
    while i < 8 {
        let ch = *chars.get(i).ok_or_else(write_zero_err)?;
        let slot = head.get_mut(i).ok_or_else(write_zero_err)?;
        *slot = u8::try_from(u32::from(ch)).map_err(|err| {
            ioErr::new(
                ErrorKind::InvalidData,
                format!("password contains non-ASCII character: {err}"),
            )
        })?;
        i += 1;
    }
    Ok(())
}

fn buf_write_u32_dec(cur: &mut BufCursor<'_>, mut n: u32) -> IoRst<()> {
    let mut tmp = [0_u8; 10];
    let mut i = tmp.len();
    while n >= 100 {
        let rem = usize::from(low_u8_from_u32(n % 100));
        n /= 100;
        i -= 2;
        range_slice_mut(&mut tmp, i, 2)?.copy_from_slice(two_digits(rem)?);
    }
    if n >= 10 {
        let rem = usize::from(low_u8_from_u32(n));
        i -= 2;
        range_slice_mut(&mut tmp, i, 2)?.copy_from_slice(two_digits(rem)?);
    } else {
        i -= 1;
        let digit = low_u8_from_u32(n);
        let slot = tmp.get_mut(i).ok_or_else(write_zero_err)?;
        *slot = b'0' + digit;
    }
    buf_write_bytes(cur, tmp.get(i..).ok_or_else(write_zero_err)?)
}

fn buf_write_u64_dec(cur: &mut BufCursor<'_>, mut n: u64) -> IoRst<()> {
    let mut tmp = [0_u8; 20];
    let mut i = tmp.len();
    while n >= 100 {
        let rem = usize::from(low_u8_from_u64(n % 100));
        n /= 100;
        i -= 2;
        range_slice_mut(&mut tmp, i, 2)?.copy_from_slice(two_digits(rem)?);
    }
    if n >= 10 {
        let rem = usize::from(low_u8_from_u64(n));
        i -= 2;
        range_slice_mut(&mut tmp, i, 2)?.copy_from_slice(two_digits(rem)?);
    } else {
        i -= 1;
        let digit = low_u8_from_u64(n);
        let slot = tmp.get_mut(i).ok_or_else(write_zero_err)?;
        *slot = b'0' + digit;
    }
    buf_write_bytes(cur, tmp.get(i..).ok_or_else(write_zero_err)?)
}

fn buf_write_i64_dec(cur: &mut BufCursor<'_>, n: i64) -> IoRst<()> {
    if n < 0 {
        buf_write_byte(cur, b'-')?;
        let abs = if n == i64::MIN {
            i64::MAX.cast_unsigned() + 1
        } else {
            (-n).cast_unsigned()
        };
        buf_write_u64_dec(cur, abs)
    } else {
        buf_write_u64_dec(cur, n.cast_unsigned())
    }
}

fn buf_write_u32_dec_0pad_6(cur: &mut BufCursor<'_>, n: u32) -> IoRst<()> {
    if n >= 1_000_000 {
        return buf_write_u32_dec(cur, n);
    }
    let hi = usize::from(low_u8_from_u32(n / 10_000));
    let rem = usize::from(low_u16_from_u64(u64::from(n % 10_000)));
    let mid = rem / 100;
    let lo = rem % 100;
    let head = cur.take(6)?;
    let (hi_digits, rest) = head.split_at_mut(2);
    hi_digits.copy_from_slice(two_digits(hi)?);
    let (mid_digits, lo_digits) = rest.split_at_mut(2);
    mid_digits.copy_from_slice(two_digits(mid)?);
    lo_digits.copy_from_slice(two_digits(lo)?);
    Ok(())
}

fn buf_write_u64_octal(cur: &mut BufCursor<'_>, mut n: u64) -> IoRst<()> {
    if n == 0 {
        return buf_write_byte(cur, b'0');
    }
    let mut tmp = [0_u8; 22];
    let mut i = tmp.len();
    while n != 0 {
        i -= 1;
        let oct_digit = low_u8_from_u64(n & 7);
        let slot = tmp.get_mut(i).ok_or_else(write_zero_err)?;
        *slot = b'0' + oct_digit;
        n >>= 3_u32;
    }
    buf_write_bytes(cur, tmp.get(i..).ok_or_else(write_zero_err)?)
}

fn buf_write_hex_u16_0pad4(cur: &mut BufCursor<'_>, v: u16) -> IoRst<()> {
    let head = cur.take(4)?;
    let upper = usize::from(low_u8_from_u32(u32::from(v >> 8_u32)));
    let lower = usize::from(low_u8_from_u32(u32::from(v)));
    let (upper_hex, lower_hex) = head.split_at_mut(2);
    upper_hex.copy_from_slice(hex_byte(upper)?);
    lower_hex.copy_from_slice(hex_byte(lower)?);
    Ok(())
}

fn buf_write_hex_u16_min3(cur: &mut BufCursor<'_>, v: u16) -> IoRst<()> {
    if v < 0x1000 {
        let head = cur.take(3)?;
        let hi = usize::from(low_u8_from_u32(u32::from(v >> 8)));
        let lo = usize::from(low_u8_from_u32(u32::from(v)));
        let (prefix, suffix) = head.split_at_mut(1);
        if let Some(slot) = prefix.first_mut() {
            *slot = HEX_UPPER.get(hi).copied().ok_or_else(write_zero_err)?;
        }
        suffix.copy_from_slice(hex_byte(lo)?);
        Ok(())
    } else {
        buf_write_hex_u16_0pad4(cur, v)
    }
}

fn format_output(buf: &mut [u8], data: &RandomDataSet, use_colors: bool) -> IoRst<usize> {
    let mut cur = BufCursor::new(buf);
    let v = data.num_64;
    let bytes = u64_to_be_bytes(v);
    let [b0, b1, b2, b3, b4, b5, _, _] = bytes;
    buf_write_bytes(&mut cur, "64비트 난수: ".as_bytes())?;
    buf_write_u64_dec(&mut cur, v)?;
    buf_write_bytes(&mut cur, " (유부호 정수: ".as_bytes())?;
    buf_write_i64_dec(&mut cur, v.cast_signed())?;
    buf_write_bytes(&mut cur, b")\n")?;
    buf_write_bin8_line(&mut cur, bytes)?;
    buf_write_bytes(&mut cur, "8진수: ".as_bytes())?;
    buf_write_u64_octal(&mut cur, v)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_hex8_line(&mut cur, bytes)?;
    buf_write_bytes(&mut cur, "Hex 코드: ".as_bytes())?;
    if use_colors {
        buf_write_bytes(&mut cur, b"\x1B[38;2;")?;
        buf_write_u8_dec(&mut cur, b0)?;
        buf_write_byte(&mut cur, b';')?;
        buf_write_u8_dec(&mut cur, b1)?;
        buf_write_byte(&mut cur, b';')?;
        buf_write_u8_dec(&mut cur, b2)?;
        buf_write_m_hash_hex24_from_bytes(&mut cur, b0, b1, b2)?;
        buf_write_bytes(&mut cur, b"\x1B[0m \x1B[38;2;")?;
        buf_write_u8_dec(&mut cur, b3)?;
        buf_write_byte(&mut cur, b';')?;
        buf_write_u8_dec(&mut cur, b4)?;
        buf_write_byte(&mut cur, b';')?;
        buf_write_u8_dec(&mut cur, b5)?;
        buf_write_m_hash_hex24_from_bytes(&mut cur, b3, b4, b5)?;
        buf_write_bytes(&mut cur, b"\x1B[0m")?;
    } else {
        buf_write_hash_hex24_from_bytes(&mut cur, b0, b1, b2)?;
        buf_write_byte(&mut cur, b' ')?;
        buf_write_hash_hex24_from_bytes(&mut cur, b3, b4, b5)?;
    }
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "바이트 배열: ".as_bytes())?;
    buf_write_u8_array_spaced(&mut cur, &bytes)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "6자리 숫자 비밀번호: ".as_bytes())?;
    buf_write_u32_dec_0pad_6(&mut cur, data.numeric_password)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "8자리 비밀번호: ".as_bytes())?;
    buf_write_ascii8(&mut cur, &data.password)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "로또 번호: ".as_bytes())?;
    buf_write_u8_array_spaced(&mut cur, &data.lotto_numbers)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "일본 로또 7 번호: ".as_bytes())?;
    buf_write_u8_array_spaced(&mut cur, &data.lotto7_numbers)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "유로밀리언 번호: ".as_bytes())?;
    buf_write_u8_array_spaced(&mut cur, &data.euro_millions_main_numbers)?;
    buf_write_bytes(&mut cur, b" + ")?;
    buf_write_u8_array_spaced(&mut cur, &data.euro_millions_lucky_stars)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "한글 음절 4글자: ".as_bytes())?;
    buf_write_chars(&mut cur, &data.hangul_syllables)?;
    buf_write_byte(&mut cur, b'\n')?;
    cursor_write_fmt(
        &mut cur,
        format_args!(
            "대한민국 위경도: {kor_lat}, {kor_lon}\n세계 위경도: {world_lat}, {world_lon}\n",
            kor_lat = data.kor_coords.0,
            kor_lon = data.kor_coords.1,
            world_lat = data.world_coords.0,
            world_lon = data.world_coords.1
        ),
    )?;
    buf_write_bytes(&mut cur, "NMS 은하 번호: ".as_bytes())?;
    buf_write_u32_dec(&mut cur, u32::from(u16::from(b0).wrapping_add(1)))?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "NMS 포탈 주소: ".as_bytes())?;
    buf_write_u8_dec(&mut cur, data.planet_number)?;
    buf_write_byte(&mut cur, b' ')?;
    buf_write_hex_u16_min3(&mut cur, data.solar_system_index)?;
    buf_write_byte(&mut cur, b' ')?;
    buf_write_bytes(&mut cur, hex_byte(usize::from(data.nms_portal_yy))?)?;
    buf_write_byte(&mut cur, b' ')?;
    buf_write_hex_u16_min3(&mut cur, data.nms_portal_zzz)?;
    buf_write_byte(&mut cur, b' ')?;
    buf_write_hex_u16_min3(&mut cur, data.nms_portal_xxx)?;
    buf_write_byte(&mut cur, b' ')?;
    buf_write_byte(&mut cur, b'(')?;
    buf_write_chars(&mut cur, &data.glyph_string)?;
    buf_write_bytes(&mut cur, b")\n")?;
    buf_write_bytes(&mut cur, "NMS 은하 좌표: ".as_bytes())?;
    buf_write_hex_u16_0pad4(&mut cur, data.galaxy_x)?;
    buf_write_byte(&mut cur, b':')?;
    buf_write_hex_u16_0pad4(&mut cur, data.galaxy_y)?;
    buf_write_byte(&mut cur, b':')?;
    buf_write_hex_u16_0pad4(&mut cur, data.galaxy_z)?;
    buf_write_byte(&mut cur, b':')?;
    buf_write_hex_u16_0pad4(&mut cur, data.solar_system_index)?;
    Ok(cur.written_len())
}

pub fn write_buffer_to_file_guard(
    file_guard: &mut MutexGuard<BufWriter<File>>,
    buffer: &[u8],
) -> IoRst<()> {
    file_guard.write_all(buffer)?;
    file_guard.write_all(b"\n")?;
    Ok(())
}

pub fn write_slice_to_console(data_slice: &[u8]) -> IoRst<()> {
    let mut stdout_lock = stdout().lock();
    stdout_lock.write_all(data_slice)?;
    stdout_lock.write_all(b"\n")?;
    stdout_lock.flush()?;
    Ok(())
}

pub fn print_progress(
    completed: u64,
    total: u64,
    start_time: &Instant,
    elapsed_buf: &mut [u8],
    eta_buf: &mut [u8],
) -> Result<()> {
    if !*IS_TERMINAL {
        return Ok(());
    }
    let elapsed_millis = start_time.elapsed().as_millis();
    let elapsed_deci = elapsed_millis / 100;
    let eta_deci = if total == 0 || completed >= total {
        Some(0)
    } else if completed == 0 {
        None
    } else {
        Some(
            elapsed_millis.saturating_mul(u128::from(total - completed))
                / (u128::from(completed) * 100),
        )
    };
    let elapsed_len = format_time_into(Some(elapsed_deci), elapsed_buf);
    let eta_len = format_time_into(eta_deci, eta_buf);
    let filled_u64 = if total == 0 {
        BAR_WIDTH_U64
    } else {
        completed.saturating_mul(BAR_WIDTH_U64) / total
    };
    let filled = usize::from(low_u8_from_u64(filled_u64.min(BAR_WIDTH_U64)));
    let bar = BAR_FULL
        .get(filled)
        .copied()
        .ok_or_else(|| ioErr::other("진행 막대 인덱스 범위 초과"))?;
    let percent_u64 = if total == 0 {
        100
    } else {
        completed.saturating_mul(100) / total
    };
    let percent = low_u8_from_u64(percent_u64.min(100));
    let mut line = [0_u8; 128];
    let mut cur = BufCursor::new(&mut line);
    buf_write_byte(&mut cur, b'\r')?;
    buf_write_bytes(&mut cur, bar.as_bytes())?;
    buf_write_byte(&mut cur, b' ')?;
    match percent {
        0..=9 => buf_write_bytes(&mut cur, b"  ")?,
        10..=99 => buf_write_byte(&mut cur, b' ')?,
        _ => {}
    }
    buf_write_u8_dec(&mut cur, percent)?;
    buf_write_byte(&mut cur, b'%')?;
    buf_write_bytes(&mut cur, b" (")?;
    buf_write_u64_dec(&mut cur, completed)?;
    buf_write_byte(&mut cur, b'/')?;
    buf_write_u64_dec(&mut cur, total)?;
    buf_write_bytes(&mut cur, ") | 소요: ".as_bytes())?;
    buf_write_bytes(&mut cur, prefix_slice(elapsed_buf, elapsed_len)?)?;
    buf_write_bytes(&mut cur, b" | ETA: ")?;
    buf_write_bytes(&mut cur, prefix_slice(eta_buf, eta_len)?)?;
    buf_write_bytes(&mut cur, b" \x1b[K")?;
    let used = cur.written_len();
    let mut out = stdout().lock();
    out.write_all(prefix_slice(&line, used)?)?;
    out.flush()?;
    Ok(())
}

fn format_time_into(deci_seconds: Option<u128>, buf: &mut [u8]) -> usize {
    let Some(head) = buf.get_mut(..7) else {
        return 0;
    };
    let Some(deci) = deci_seconds else {
        head.copy_from_slice(INVALID_TIME);
        return 7;
    };
    let minutes = usize::from(low_u8_from_u128((deci / 600).min(99)));
    let sec_whole = usize::from(low_u8_from_u128((deci / 10) % 60));
    let tenths = usize::from(low_u8_from_u128(deci % 10));
    let (minute_digits, rest) = head.split_at_mut(2);
    let Some(minute_pair) = TWO_DIGITS.get(minutes) else {
        return 0;
    };
    minute_digits.copy_from_slice(minute_pair);
    let Some((separator, remainder)) = rest.split_first_mut() else {
        return 0;
    };
    *separator = b':';
    let (second_digits, tenth_part) = remainder.split_at_mut(2);
    let Some(second_pair) = TWO_DIGITS.get(sec_whole) else {
        return 0;
    };
    second_digits.copy_from_slice(second_pair);
    let Some((decimal_point, tenths_digit)) = tenth_part.split_first_mut() else {
        return 0;
    };
    *decimal_point = b'.';
    let Some(tenth_slot) = tenths_digit.first_mut() else {
        return 0;
    };
    let Some(digit) = DIGITS.get(tenths).copied() else {
        return 0;
    };
    *tenth_slot = digit;
    7
}
