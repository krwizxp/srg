use crate::{
    BAR_FULL, BAR_WIDTH_U64, BIN8_TABLE, BUFFER_SIZE, HEX_BYTE_TABLE, HEX_UPPER, INVALID_TIME,
    IS_TERMINAL, RandomDataSet, Result,
    buffmt::{ByteCursor, DIGITS, copy_two_digits, digit_byte, write_zero_err},
    numeric::{low_u8_from_u32, low_u8_from_u64, low_u8_from_u128, low_u16_from_u64},
};
use core::fmt::Write as _;
use std::{
    fs::File,
    io::{BufWriter, Error as IoError, Result as IoResult, Write, stdout},
    sync::MutexGuard,
};
pub const PROGRESS_LINE_BUF_LEN: usize = 128;
const BYTE_GROUP_COUNT: usize = 8;
const DECI_PER_MINUTE: u128 = 600;
const DECI_PER_SECOND: u128 = 10;
const ELAPSED_MILLIS_PER_DECI: u128 = 100;
const FOUR_DIGIT_WIDTH: usize = 4;
const HASH_HEX24_LEN: usize = 7;
const HEX_U16_FULL_WIDTH: usize = FOUR_DIGIT_WIDTH;
const HEX_U16_SHORT_THRESHOLD: u16 = 0x1000;
const MAX_TIME_MINUTES: u128 = 99;
const M_HASH_HEX24_LEN: usize = 8;
const OCTAL_DIGIT_MASK: u64 = 7;
const OCTAL_SHIFT_BITS: u32 = 3;
const OCTAL_TMP_LEN: usize = 22;
const PASSWORD_FULL_WIDTH_THRESHOLD: u32 = 1_000_000;
const PASSWORD_HIGH_DIVISOR: u32 = 10_000;
const PASSWORD_WIDTH: usize = 6;
const PERCENT_WIDTH: usize = 3;
const PERCENT_SCALE_U64: u64 = 100;
const SECONDS_PER_MINUTE_U128: u128 = 60;
const THREE_DIGIT_WIDTH: usize = 3;
const TIME_BUF_LEN: usize = 7;
const TWO_DIGIT_WIDTH: usize = 2;
const U32_DEC_BUF_LEN: usize = 10;
const U64_DEC_BUF_LEN: usize = 20;
const U8_THREE_DIGIT_THRESHOLD: u8 = 100;
const U8_TWO_DIGIT_THRESHOLD: u8 = 10;
type BufCursor<'a> = ByteCursor<'a>;
struct OutputFormatter<'a, 'b, 'c> {
    bytes: [u8; 8],
    cursor: &'a mut BufCursor<'b>,
    data: &'c RandomDataSet,
    use_colors: bool,
}
impl OutputFormatter<'_, '_, '_> {
    fn write_all(&mut self) -> Result<()> {
        self.write_number_lines()?;
        self.write_random_lines()?;
        self.write_nms_lines()?;
        Ok(())
    }
    fn write_labeled_line<F>(&mut self, label: &[u8], write_value: F) -> IoResult<()>
    where
        F: FnOnce(&mut BufCursor<'_>) -> IoResult<()>,
    {
        self.cursor.write_bytes(label)?;
        write_value(self.cursor)?;
        self.cursor.write_byte(b'\n')
    }
    fn write_labeled_u8_array_line<const N: usize>(
        &mut self,
        label: &[u8],
        values: &[u8; N],
    ) -> IoResult<()> {
        self.write_labeled_line(label, |buffer_cur| {
            buf_write_u8_array_spaced(buffer_cur, values)
        })
    }
    fn write_nms_lines(&mut self) -> Result<()> {
        let [galaxy_number_byte, ..] = self.bytes;
        let data = self.data;
        self.write_labeled_line("NMS 은하 번호: ".as_bytes(), |buffer_cur| {
            buf_write_u32_dec(
                buffer_cur,
                u32::from(u16::from(galaxy_number_byte).wrapping_add(1)),
            )
        })?;
        self.write_labeled_line("NMS 포탈 주소: ".as_bytes(), |buffer_cur| {
            buf_write_u8_dec(buffer_cur, data.planet_number)?;
            buffer_cur.write_byte(b' ')?;
            buf_write_hex_u16_min3(buffer_cur, data.solar_system_index)?;
            buffer_cur.write_byte(b' ')?;
            buffer_cur.write_bytes(hex_byte(usize::from(data.nms_portal_yy))?)?;
            buffer_cur.write_byte(b' ')?;
            buf_write_hex_u16_min3(buffer_cur, data.nms_portal_zzz)?;
            buffer_cur.write_byte(b' ')?;
            buf_write_hex_u16_min3(buffer_cur, data.nms_portal_xxx)?;
            buffer_cur.write_byte(b' ')?;
            buffer_cur.write_byte(b'(')?;
            buf_write_chars(buffer_cur, &data.glyph_string)?;
            buffer_cur.write_bytes(b")")
        })?;
        self.write_labeled_line("NMS 은하 좌표: ".as_bytes(), |buffer_cur| {
            buf_write_hex_u16_0pad4(buffer_cur, data.galaxy_x)?;
            buffer_cur.write_byte(b':')?;
            buf_write_hex_u16_0pad4(buffer_cur, data.galaxy_y)?;
            buffer_cur.write_byte(b':')?;
            buf_write_hex_u16_0pad4(buffer_cur, data.galaxy_z)?;
            buffer_cur.write_byte(b':')?;
            buf_write_hex_u16_0pad4(buffer_cur, data.solar_system_index)
        })?;
        Ok(())
    }
    fn write_number_lines(&mut self) -> Result<()> {
        let number64 = self.data.num_64;
        let signed_number = number64.cast_signed();
        let bytes = self.bytes;
        let use_colors = self.use_colors;
        self.cursor.write_bytes("64비트 난수: ".as_bytes())?;
        buf_write_u64_dec(self.cursor, number64)?;
        self.cursor.write_bytes(" (유부호 정수: ".as_bytes())?;
        if signed_number < 0 {
            self.cursor.write_byte(b'-')?;
        }
        buf_write_u64_dec(self.cursor, signed_number.unsigned_abs())?;
        self.cursor.write_bytes(b")\n")?;
        self.write_prefixed_byte_groups("2진수: ", 8, |byte| {
            BIN8_TABLE
                .get(usize::from(byte))
                .map(<[u8; 8]>::as_slice)
                .ok_or_else(write_zero_err)
        })?;
        self.write_labeled_line("8진수: ".as_bytes(), |buffer_cur| {
            if number64 == 0 {
                return buffer_cur.write_byte(b'0');
            }
            let mut tmp = [0_u8; OCTAL_TMP_LEN];
            let mut index = tmp.len();
            let mut octal_number = number64;
            while octal_number != 0 {
                sub_from_index(&mut index, 1)?;
                let oct_digit = low_u8_from_u64(octal_number & OCTAL_DIGIT_MASK);
                let slot = tmp.get_mut(index).ok_or_else(write_zero_err)?;
                *slot = digit_byte(usize::from(oct_digit))?;
                octal_number >>= OCTAL_SHIFT_BITS;
            }
            buffer_cur.write_bytes(tmp.get(index..).ok_or_else(write_zero_err)?)
        })?;
        self.write_prefixed_byte_groups("16진수: ", 2, |byte| {
            hex_byte(usize::from(byte)).map(<[u8; 2]>::as_slice)
        })?;
        self.write_labeled_line("Hex 코드: ".as_bytes(), |buffer_cur| {
            let [b0, b1, b2, b3, b4, b5, _, _] = bytes;
            if use_colors {
                buffer_cur.write_bytes(b"\x1B[38;2;")?;
                buf_write_u8_dec(buffer_cur, b0)?;
                buffer_cur.write_byte(b';')?;
                buf_write_u8_dec(buffer_cur, b1)?;
                buffer_cur.write_byte(b';')?;
                buf_write_u8_dec(buffer_cur, b2)?;
                buf_write_m_hash_hex24_from_bytes(buffer_cur, b0, b1, b2)?;
                buffer_cur.write_bytes(b"\x1B[0m \x1B[38;2;")?;
                buf_write_u8_dec(buffer_cur, b3)?;
                buffer_cur.write_byte(b';')?;
                buf_write_u8_dec(buffer_cur, b4)?;
                buffer_cur.write_byte(b';')?;
                buf_write_u8_dec(buffer_cur, b5)?;
                buf_write_m_hash_hex24_from_bytes(buffer_cur, b3, b4, b5)?;
                return buffer_cur.write_bytes(b"\x1B[0m");
            }
            buf_write_hash_hex24_from_bytes(buffer_cur, b0, b1, b2)?;
            buffer_cur.write_byte(b' ')?;
            buf_write_hash_hex24_from_bytes(buffer_cur, b3, b4, b5)
        })?;
        Ok(())
    }
    fn write_prefixed_byte_groups<F>(
        &mut self,
        prefix: &'static str,
        group_width: usize,
        mut render_group: F,
    ) -> IoResult<()>
    where
        F: FnMut(u8) -> IoResult<&'static [u8]>,
    {
        let prefix_bytes = prefix.as_bytes();
        let prefix_len = prefix_bytes.len();
        let line_len = prefix_len
            .checked_add(
                group_width
                    .checked_mul(BYTE_GROUP_COUNT)
                    .ok_or_else(write_zero_err)?,
            )
            .and_then(|value| value.checked_add(BYTE_GROUP_COUNT))
            .ok_or_else(write_zero_err)?;
        let head = self.cursor.take(line_len)?;
        head.get_mut(..prefix_len)
            .ok_or_else(write_zero_err)?
            .copy_from_slice(prefix_bytes);
        let mut pos = prefix_len;
        for (index, byte) in self.bytes.into_iter().enumerate() {
            let group = render_group(byte)?;
            range_slice_mut(head, pos, group_width)?.copy_from_slice(group);
            add_to_index(&mut pos, group_width)?;
            let slot = head.get_mut(pos).ok_or_else(write_zero_err)?;
            *slot = if index == BYTE_GROUP_COUNT.saturating_sub(1) {
                b'\n'
            } else {
                b' '
            };
            add_to_index(&mut pos, 1)?;
        }
        Ok(())
    }
    fn write_random_lines(&mut self) -> Result<()> {
        let bytes = self.bytes;
        let data = self.data;
        self.write_labeled_u8_array_line("바이트 배열: ".as_bytes(), &bytes)?;
        self.write_labeled_line("6자리 숫자 비밀번호: ".as_bytes(), |buffer_cur| {
            if data.numeric_password >= PASSWORD_FULL_WIDTH_THRESHOLD {
                return buf_write_u32_dec(buffer_cur, data.numeric_password);
            }
            let hi = usize::from(low_u8_from_u32(
                data.numeric_password.div_euclid(PASSWORD_HIGH_DIVISOR),
            ));
            let rem = usize::from(low_u16_from_u64(u64::from(
                data.numeric_password.rem_euclid(PASSWORD_HIGH_DIVISOR),
            )));
            let head = buffer_cur.take(PASSWORD_WIDTH)?;
            let (hi_digits, rest) = head.split_at_mut(TWO_DIGIT_WIDTH);
            copy_two_digits(hi_digits, hi)?;
            let (mid_digits, lo_digits) = rest.split_at_mut(TWO_DIGIT_WIDTH);
            copy_two_digits(
                mid_digits,
                rem.div_euclid(usize::from(U8_THREE_DIGIT_THRESHOLD)),
            )?;
            copy_two_digits(
                lo_digits,
                rem.rem_euclid(usize::from(U8_THREE_DIGIT_THRESHOLD)),
            )?;
            Ok(())
        })?;
        self.write_labeled_line("8자리 비밀번호: ".as_bytes(), |buffer_cur| {
            buffer_cur.write_bytes(&data.password)
        })?;
        self.write_labeled_u8_array_line("로또 번호: ".as_bytes(), &data.lotto_numbers)?;
        self.write_labeled_u8_array_line("일본 로또 7 번호: ".as_bytes(), &data.lotto7_numbers)?;
        self.write_labeled_line("유로밀리언 번호: ".as_bytes(), |buffer_cur| {
            buf_write_u8_array_spaced(buffer_cur, &data.euro_millions_main_numbers)?;
            buffer_cur.write_bytes(b" + ")?;
            buf_write_u8_array_spaced(buffer_cur, &data.euro_millions_lucky_stars)
        })?;
        self.write_labeled_line("한글 음절 4글자: ".as_bytes(), |buffer_cur| {
            buf_write_chars(buffer_cur, &data.hangul_syllables)
        })?;
        self.cursor.write_bytes("대한민국 위경도: ".as_bytes())?;
        writeln!(self.cursor, "{}, {}", data.kor_coords.0, data.kor_coords.1)?;
        self.cursor.write_bytes("세계 위경도: ".as_bytes())?;
        writeln!(
            self.cursor,
            "{}, {}",
            data.world_coords.0, data.world_coords.1
        )?;
        Ok(())
    }
}
pub fn format_data_into_buffer(
    data: &RandomDataSet,
    buffer: &mut [u8; BUFFER_SIZE],
    use_colors: bool,
) -> Result<usize> {
    let mut cur = BufCursor::new(buffer.as_mut_slice());
    let mut formatter = OutputFormatter {
        bytes: data.num_64.to_be_bytes(),
        cursor: &mut cur,
        data,
        use_colors,
    };
    formatter.write_all()?;
    Ok(cur.written_len())
}
fn checked_add_index(value: usize, amount: usize) -> IoResult<usize> {
    value.checked_add(amount).ok_or_else(write_zero_err)
}
fn add_to_index(value: &mut usize, amount: usize) -> IoResult<()> {
    *value = checked_add_index(*value, amount)?;
    Ok(())
}
fn sub_from_index(value: &mut usize, amount: usize) -> IoResult<()> {
    *value = value.checked_sub(amount).ok_or_else(write_zero_err)?;
    Ok(())
}
pub fn prefix_slice(slice: &[u8], len: usize) -> IoResult<&[u8]> {
    slice.get(..len).ok_or_else(write_zero_err)
}
fn range_slice_mut(slice: &mut [u8], start: usize, len: usize) -> IoResult<&mut [u8]> {
    let end = start.checked_add(len).ok_or_else(write_zero_err)?;
    slice.get_mut(start..end).ok_or_else(write_zero_err)
}
fn hex_byte(index: usize) -> IoResult<&'static [u8; 2]> {
    HEX_BYTE_TABLE.get(index).ok_or_else(write_zero_err)
}
const fn u8_dec_len(n: u8) -> usize {
    if n >= U8_THREE_DIGIT_THRESHOLD {
        3
    } else if n >= U8_TWO_DIGIT_THRESHOLD {
        2
    } else {
        1
    }
}
fn buf_write_chars<const N: usize>(cur: &mut BufCursor<'_>, chars: &[char; N]) -> IoResult<()> {
    let mut total = 0_usize;
    for &ch in chars {
        total = checked_add_index(total, ch.len_utf8())?;
    }
    let head = cur.take(total)?;
    let mut pos = 0_usize;
    for &ch in chars {
        let tail = head.get_mut(pos..).ok_or_else(write_zero_err)?;
        let written = ch.encode_utf8(tail).len();
        pos = checked_add_index(pos, written)?;
    }
    Ok(())
}
fn buf_write_u8_dec(cur: &mut BufCursor<'_>, n: u8) -> IoResult<()> {
    let head = cur.take(u8_dec_len(n))?;
    write_u8_dec_into_slice(head, n)?;
    Ok(())
}
fn write_u8_dec_into_slice(target: &mut [u8], n: u8) -> IoResult<usize> {
    if n >= U8_THREE_DIGIT_THRESHOLD {
        let hundreds = usize::from(n.div_euclid(U8_THREE_DIGIT_THRESHOLD));
        let rem = usize::from(n.rem_euclid(U8_THREE_DIGIT_THRESHOLD));
        let Some((digit_slot, remaining_digits)) = target.split_first_mut() else {
            return Err(write_zero_err());
        };
        *digit_slot = digit_byte(hundreds)?;
        copy_two_digits(remaining_digits, rem)?;
        return Ok(3);
    }
    if n >= U8_TWO_DIGIT_THRESHOLD {
        copy_two_digits(target, usize::from(n))?;
        return Ok(2);
    }
    let Some(slot) = target.first_mut() else {
        return Err(write_zero_err());
    };
    *slot = digit_byte(usize::from(n))?;
    Ok(1)
}
fn buf_write_u8_array_spaced<const N: usize>(
    cur: &mut BufCursor<'_>,
    nums: &[u8; N],
) -> IoResult<()> {
    let mut total = N.saturating_sub(1);
    for &n in nums {
        total = checked_add_index(total, u8_dec_len(n))?;
    }
    let head = cur.take(total)?;
    let mut pos = 0_usize;
    for (index, &n) in nums.iter().enumerate() {
        if index != 0 {
            let Some(slot) = head.get_mut(pos) else {
                return Err(write_zero_err());
            };
            *slot = b' ';
            add_to_index(&mut pos, 1)?;
        }
        let width = u8_dec_len(n);
        let slot = range_slice_mut(head, pos, width)?;
        write_u8_dec_into_slice(slot, n)?;
        add_to_index(&mut pos, width)?;
    }
    Ok(())
}
fn buf_write_hash_hex24_from_bytes(
    cur: &mut BufCursor<'_>,
    b0: u8,
    b1: u8,
    b2: u8,
) -> IoResult<()> {
    let head = cur.take(HASH_HEX24_LEN)?;
    let Some((prefix, hex_bytes)) = head.split_first_mut() else {
        return Err(write_zero_err());
    };
    *prefix = b'#';
    buf_write_hex24_bytes(hex_bytes, b0, b1, b2)?;
    Ok(())
}
fn buf_write_m_hash_hex24_from_bytes(
    cur: &mut BufCursor<'_>,
    b0: u8,
    b1: u8,
    b2: u8,
) -> IoResult<()> {
    let head = cur.take(M_HASH_HEX24_LEN)?;
    let Some((m_prefix, rest)) = head.split_first_mut() else {
        return Err(write_zero_err());
    };
    *m_prefix = b'm';
    let Some((hash_prefix, hex_bytes)) = rest.split_first_mut() else {
        return Err(write_zero_err());
    };
    *hash_prefix = b'#';
    buf_write_hex24_bytes(hex_bytes, b0, b1, b2)?;
    Ok(())
}
fn buf_write_hex24_bytes(buf: &mut [u8], b0: u8, b1: u8, b2: u8) -> IoResult<()> {
    let (first_hex, remaining_hex_bytes) = buf.split_at_mut(TWO_DIGIT_WIDTH);
    first_hex.copy_from_slice(hex_byte(usize::from(b0))?);
    let (second_hex, third_hex) = remaining_hex_bytes.split_at_mut(TWO_DIGIT_WIDTH);
    second_hex.copy_from_slice(hex_byte(usize::from(b1))?);
    third_hex.copy_from_slice(hex_byte(usize::from(b2))?);
    Ok(())
}
fn buf_write_u32_dec(cur: &mut BufCursor<'_>, mut n: u32) -> IoResult<()> {
    let mut tmp = [0_u8; U32_DEC_BUF_LEN];
    let mut i = tmp.len();
    while n >= u32::from(U8_THREE_DIGIT_THRESHOLD) {
        let rem = usize::from(low_u8_from_u32(
            n.rem_euclid(u32::from(U8_THREE_DIGIT_THRESHOLD)),
        ));
        n = n.div_euclid(u32::from(U8_THREE_DIGIT_THRESHOLD));
        sub_from_index(&mut i, TWO_DIGIT_WIDTH)?;
        copy_two_digits(range_slice_mut(&mut tmp, i, TWO_DIGIT_WIDTH)?, rem)?;
    }
    if n >= u32::from(U8_TWO_DIGIT_THRESHOLD) {
        let rem = usize::from(low_u8_from_u32(n));
        sub_from_index(&mut i, TWO_DIGIT_WIDTH)?;
        copy_two_digits(range_slice_mut(&mut tmp, i, TWO_DIGIT_WIDTH)?, rem)?;
    } else {
        sub_from_index(&mut i, 1)?;
        let digit = low_u8_from_u32(n);
        let slot = tmp.get_mut(i).ok_or_else(write_zero_err)?;
        *slot = digit_byte(usize::from(digit))?;
    }
    cur.write_bytes(tmp.get(i..).ok_or_else(write_zero_err)?)
}
fn buf_write_u64_dec(cur: &mut BufCursor<'_>, mut n: u64) -> IoResult<()> {
    let mut tmp = [0_u8; U64_DEC_BUF_LEN];
    let mut i = tmp.len();
    while n >= u64::from(U8_THREE_DIGIT_THRESHOLD) {
        let rem = usize::from(low_u8_from_u64(
            n.rem_euclid(u64::from(U8_THREE_DIGIT_THRESHOLD)),
        ));
        n = n.div_euclid(u64::from(U8_THREE_DIGIT_THRESHOLD));
        sub_from_index(&mut i, TWO_DIGIT_WIDTH)?;
        copy_two_digits(range_slice_mut(&mut tmp, i, TWO_DIGIT_WIDTH)?, rem)?;
    }
    if n >= u64::from(U8_TWO_DIGIT_THRESHOLD) {
        let rem = usize::from(low_u8_from_u64(n));
        sub_from_index(&mut i, TWO_DIGIT_WIDTH)?;
        copy_two_digits(range_slice_mut(&mut tmp, i, TWO_DIGIT_WIDTH)?, rem)?;
    } else {
        sub_from_index(&mut i, 1)?;
        let digit = low_u8_from_u64(n);
        let slot = tmp.get_mut(i).ok_or_else(write_zero_err)?;
        *slot = digit_byte(usize::from(digit))?;
    }
    cur.write_bytes(tmp.get(i..).ok_or_else(write_zero_err)?)
}
fn buf_write_hex_u16_0pad4(cur: &mut BufCursor<'_>, value: u16) -> IoResult<()> {
    let head = cur.take(HEX_U16_FULL_WIDTH)?;
    let upper = usize::from(low_u8_from_u32(u32::from(value >> 8_u32)));
    let lower = usize::from(low_u8_from_u32(u32::from(value)));
    let (upper_hex, lower_hex) = head.split_at_mut(TWO_DIGIT_WIDTH);
    upper_hex.copy_from_slice(hex_byte(upper)?);
    lower_hex.copy_from_slice(hex_byte(lower)?);
    Ok(())
}
fn buf_write_hex_u16_min3(cur: &mut BufCursor<'_>, value: u16) -> IoResult<()> {
    if value < HEX_U16_SHORT_THRESHOLD {
        let head = cur.take(THREE_DIGIT_WIDTH)?;
        let hi = usize::from(low_u8_from_u32(u32::from(value >> 8)));
        let lo = usize::from(low_u8_from_u32(u32::from(value)));
        let Some((prefix, suffix)) = head.split_first_mut() else {
            return Err(write_zero_err());
        };
        *prefix = HEX_UPPER.get(hi).copied().ok_or_else(write_zero_err)?;
        suffix.copy_from_slice(hex_byte(lo)?);
        Ok(())
    } else {
        buf_write_hex_u16_0pad4(cur, value)
    }
}
fn scaled_progress_value(
    completed: u64,
    total: u64,
    scale: u64,
    zero_total_value: u64,
    err_msg: &'static str,
) -> IoResult<u64> {
    if total == 0 {
        return Ok(zero_total_value);
    }
    completed
        .saturating_mul(scale)
        .checked_div(total)
        .ok_or_else(|| IoError::other(err_msg))
}
pub fn write_buffer_to_file_guard(
    file_guard: &mut MutexGuard<BufWriter<File>>,
    buffer: &[u8],
) -> IoResult<()> {
    file_guard.write_all(buffer)
}
pub fn write_slice_to_console(data_slice: &[u8]) -> IoResult<()> {
    let mut stdout_lock = stdout().lock();
    stdout_lock.write_all(data_slice)?;
    stdout_lock.flush()
}
pub fn print_progress(
    out: &mut dyn Write,
    completed: u64,
    line_buf: &mut [u8; PROGRESS_LINE_BUF_LEN],
    total: u64,
    elapsed_millis: u128,
    elapsed_buf: &mut [u8],
    eta_buf: &mut [u8],
) -> Result<()> {
    if !*IS_TERMINAL {
        return Ok(());
    }
    let elapsed_deci = elapsed_millis.div_euclid(ELAPSED_MILLIS_PER_DECI);
    let eta_deci = if total == 0 || completed >= total {
        Some(0)
    } else if completed == 0 {
        None
    } else {
        let remaining = total
            .checked_sub(completed)
            .ok_or_else(|| IoError::other("남은 작업 수 계산 실패"))?;
        let completed_scaled = u128::from(completed)
            .checked_mul(u128::from(PERCENT_SCALE_U64))
            .ok_or_else(|| IoError::other("ETA 분모 계산 실패"))?;
        Some(
            elapsed_millis
                .saturating_mul(u128::from(remaining))
                .checked_div(completed_scaled)
                .ok_or_else(|| IoError::other("ETA 계산 실패"))?,
        )
    };
    let elapsed_len = format_time_into(Some(elapsed_deci), elapsed_buf);
    let eta_len = format_time_into(eta_deci, eta_buf);
    let filled_u64 = scaled_progress_value(
        completed,
        total,
        BAR_WIDTH_U64,
        BAR_WIDTH_U64,
        "진행 막대 계산 실패",
    )?;
    let filled = usize::from(low_u8_from_u64(filled_u64.min(BAR_WIDTH_U64)));
    let bar = BAR_FULL
        .get(filled)
        .copied()
        .ok_or_else(|| IoError::other("진행 막대 인덱스 범위 초과"))?;
    let percent_u64 = scaled_progress_value(
        completed,
        total,
        PERCENT_SCALE_U64,
        PERCENT_SCALE_U64,
        "진행률 계산 실패",
    )?;
    let percent = low_u8_from_u64(percent_u64.min(PERCENT_SCALE_U64));
    let mut cur = BufCursor::new(line_buf);
    cur.write_byte(b'\r')?;
    cur.write_bytes(bar.as_bytes())?;
    cur.write_byte(b' ')?;
    let padding = PERCENT_WIDTH.saturating_sub(u8_dec_len(percent));
    for _ in 0..padding {
        cur.write_byte(b' ')?;
    }
    buf_write_u8_dec(&mut cur, percent)?;
    cur.write_byte(b'%')?;
    cur.write_bytes(b" (")?;
    buf_write_u64_dec(&mut cur, completed)?;
    cur.write_byte(b'/')?;
    buf_write_u64_dec(&mut cur, total)?;
    cur.write_bytes(") | 소요: ".as_bytes())?;
    cur.write_bytes(prefix_slice(elapsed_buf, elapsed_len)?)?;
    cur.write_bytes(b" | ETA: ")?;
    cur.write_bytes(prefix_slice(eta_buf, eta_len)?)?;
    cur.write_bytes(b" \x1b[K")?;
    out.write_all(cur.written_slice()?)?;
    out.flush()?;
    Ok(())
}
fn format_time_into(deci_seconds: Option<u128>, buf: &mut [u8]) -> usize {
    let Some(head) = buf.get_mut(..TIME_BUF_LEN) else {
        return 0;
    };
    let Some(deci) = deci_seconds else {
        head.copy_from_slice(INVALID_TIME);
        return TIME_BUF_LEN;
    };
    let minutes = usize::from(low_u8_from_u128(
        (deci.div_euclid(DECI_PER_MINUTE)).min(MAX_TIME_MINUTES),
    ));
    let sec_whole = usize::from(low_u8_from_u128(
        deci.div_euclid(DECI_PER_SECOND)
            .rem_euclid(SECONDS_PER_MINUTE_U128),
    ));
    let tenths = usize::from(low_u8_from_u128(deci.rem_euclid(DECI_PER_SECOND)));
    let (minute_digits, rest) = head.split_at_mut(TWO_DIGIT_WIDTH);
    if copy_two_digits(minute_digits, minutes).is_err() {
        return 0;
    }
    let Some((separator, remainder)) = rest.split_first_mut() else {
        return 0;
    };
    *separator = b':';
    let (second_digits, tenth_part) = remainder.split_at_mut(TWO_DIGIT_WIDTH);
    if copy_two_digits(second_digits, sec_whole).is_err() {
        return 0;
    }
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
    TIME_BUF_LEN
}
