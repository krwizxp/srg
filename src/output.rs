use crate::{
    BAR_FULL, BAR_WIDTH_U64, BIN8_TABLE, BUFFER_SIZE, HEX_BYTE_TABLE, HEX_UPPER, INVALID_TIME,
    IS_TERMINAL, RandomDataSet, Result,
    buffmt::{ByteCursor, DIGITS, TWO_DIGITS, write_zero_err},
    numeric::{low_u8_from_u32, low_u8_from_u64, low_u8_from_u128, low_u16_from_u64},
    u64_to_be_bytes,
};
use std::{
    fs::File,
    io::{BufWriter, Error as ioErr, ErrorKind, Result as IoRst, Write, stdout},
    sync::MutexGuard,
    time::Instant,
};
type BufCursor<'buffer> = ByteCursor<'buffer>;
struct OutputFormatter<'cursor, 'buffer, 'data> {
    bytes: [u8; 8],
    cursor: &'cursor mut BufCursor<'buffer>,
    data: &'data RandomDataSet,
    number64: u64,
    signed_number: i64,
    use_colors: bool,
}
impl OutputFormatter<'_, '_, '_> {
    fn write_all(&mut self) -> Result<()> {
        self.write_number_lines()?;
        self.write_random_lines()?;
        self.write_nms_lines()?;
        Ok(())
    }
    fn write_labeled_line<F>(&mut self, label: &[u8], write_value: F) -> IoRst<()>
    where
        F: FnOnce(&mut BufCursor<'_>) -> IoRst<()>,
    {
        (buf_write_bytes(self.cursor, label))?;
        write_value(self.cursor)?;
        buf_write_byte(self.cursor, b'\n')
    }
    fn write_labeled_u8_array_line<const N: usize>(
        &mut self,
        label: &[u8],
        values: &[u8; N],
    ) -> IoRst<()> {
        self.write_labeled_line(label, |buffer_cur| {
            buf_write_u8_array_spaced(buffer_cur, values)
        })
    }
    fn write_nms_lines(&mut self) -> Result<()> {
        let bytes = self.bytes;
        let data = self.data;
        self.write_labeled_line("NMS 은하 번호: ".as_bytes(), |buffer_cur| {
            buf_write_u32_dec(buffer_cur, u32::from(u16::from(bytes[0]).wrapping_add(1)))
        })?;
        self.write_labeled_line("NMS 포탈 주소: ".as_bytes(), |buffer_cur| {
            (buf_write_u8_dec(buffer_cur, data.planet_number))?;
            (buf_write_byte(buffer_cur, b' '))?;
            (buf_write_hex_u16_min3(buffer_cur, data.solar_system_index))?;
            (buf_write_byte(buffer_cur, b' '))?;
            (buf_write_bytes(buffer_cur, (hex_byte(usize::from(data.nms_portal_yy)))?))?;
            (buf_write_byte(buffer_cur, b' '))?;
            (buf_write_hex_u16_min3(buffer_cur, data.nms_portal_zzz))?;
            (buf_write_byte(buffer_cur, b' '))?;
            (buf_write_hex_u16_min3(buffer_cur, data.nms_portal_xxx))?;
            (buf_write_byte(buffer_cur, b' '))?;
            (buf_write_byte(buffer_cur, b'('))?;
            (buf_write_chars(buffer_cur, &data.glyph_string))?;
            buf_write_bytes(buffer_cur, b")")
        })?;
        self.write_labeled_line("NMS 은하 좌표: ".as_bytes(), |buffer_cur| {
            (buf_write_hex_u16_0pad4(buffer_cur, data.galaxy_x))?;
            (buf_write_byte(buffer_cur, b':'))?;
            (buf_write_hex_u16_0pad4(buffer_cur, data.galaxy_y))?;
            (buf_write_byte(buffer_cur, b':'))?;
            (buf_write_hex_u16_0pad4(buffer_cur, data.galaxy_z))?;
            (buf_write_byte(buffer_cur, b':'))?;
            buf_write_hex_u16_0pad4(buffer_cur, data.solar_system_index)
        })?;
        Ok(())
    }
    fn write_number_lines(&mut self) -> Result<()> {
        let number64 = self.number64;
        let signed_number = self.signed_number;
        let bytes = self.bytes;
        let use_colors = self.use_colors;
        (buf_write_bytes(self.cursor, "64비트 난수: ".as_bytes()))?;
        (buf_write_u64_dec(self.cursor, number64))?;
        (buf_write_bytes(self.cursor, " (유부호 정수: ".as_bytes()))?;
        if signed_number < 0 {
            (buf_write_byte(self.cursor, b'-'))?;
        }
        (buf_write_u64_dec(self.cursor, signed_number.unsigned_abs()))?;
        (buf_write_bytes(self.cursor, b")\n"))?;
        self.write_prefixed_byte_groups("2진수: ", 8, |byte| {
            BIN8_TABLE
                .get(usize::from(byte))
                .map(<[u8; 8]>::as_slice)
                .ok_or_else(write_zero_err)
        })?;
        self.write_labeled_line("8진수: ".as_bytes(), |buffer_cur| {
            if number64 == 0 {
                return buf_write_byte(buffer_cur, b'0');
            }
            let mut tmp = [0_u8; 22];
            let mut index = tmp.len();
            let mut octal_number = number64;
            while octal_number != 0 {
                (sub_from_index(&mut index, 1))?;
                let oct_digit = low_u8_from_u64(octal_number & 7);
                let slot = (tmp.get_mut(index).ok_or_else(write_zero_err))?;
                *slot = (digit_byte(usize::from(oct_digit)))?;
                octal_number >>= 3_u32;
            }
            buf_write_bytes(buffer_cur, (tmp.get(index..).ok_or_else(write_zero_err))?)
        })?;
        self.write_prefixed_byte_groups("16진수: ", 2, |byte| {
            hex_byte(usize::from(byte)).map(<[u8; 2]>::as_slice)
        })?;
        self.write_labeled_line("Hex 코드: ".as_bytes(), |buffer_cur| {
            let [b0, b1, b2, b3, b4, b5, _, _] = bytes;
            if use_colors {
                (buf_write_bytes(buffer_cur, b"\x1B[38;2;"))?;
                (buf_write_u8_dec(buffer_cur, b0))?;
                (buf_write_byte(buffer_cur, b';'))?;
                (buf_write_u8_dec(buffer_cur, b1))?;
                (buf_write_byte(buffer_cur, b';'))?;
                (buf_write_u8_dec(buffer_cur, b2))?;
                (buf_write_m_hash_hex24_from_bytes(buffer_cur, b0, b1, b2))?;
                (buf_write_bytes(buffer_cur, b"\x1B[0m \x1B[38;2;"))?;
                (buf_write_u8_dec(buffer_cur, b3))?;
                (buf_write_byte(buffer_cur, b';'))?;
                (buf_write_u8_dec(buffer_cur, b4))?;
                (buf_write_byte(buffer_cur, b';'))?;
                (buf_write_u8_dec(buffer_cur, b5))?;
                (buf_write_m_hash_hex24_from_bytes(buffer_cur, b3, b4, b5))?;
                return buf_write_bytes(buffer_cur, b"\x1B[0m");
            }
            (buf_write_hash_hex24_from_bytes(buffer_cur, b0, b1, b2))?;
            (buf_write_byte(buffer_cur, b' '))?;
            buf_write_hash_hex24_from_bytes(buffer_cur, b3, b4, b5)
        })?;
        Ok(())
    }
    fn write_prefixed_byte_groups<F>(
        &mut self,
        prefix: &'static str,
        group_width: usize,
        mut render_group: F,
    ) -> IoRst<()>
    where
        F: FnMut(u8) -> IoRst<&'static [u8]>,
    {
        let prefix_len = prefix.len();
        let line_len = (prefix_len
            .checked_add(group_width.checked_mul(8).ok_or_else(write_zero_err)?)
            .and_then(|value| value.checked_add(8))
            .ok_or_else(write_zero_err))?;
        let head = (self.cursor.take(line_len))?;
        (head.get_mut(..prefix_len).ok_or_else(write_zero_err))?.copy_from_slice(prefix.as_bytes());
        let mut pos = prefix_len;
        let mut index = 0_usize;
        while index < 8 {
            let group = render_group(*(self.bytes.get(index).ok_or_else(write_zero_err))?)?;
            (range_slice_mut(head, pos, group_width))?.copy_from_slice(group);
            (add_to_index(&mut pos, group_width))?;
            let slot = (head.get_mut(pos).ok_or_else(write_zero_err))?;
            *slot = if index == 7 { b'\n' } else { b' ' };
            (add_to_index(&mut pos, 1))?;
            index = (checked_add_index(index, 1))?;
        }
        Ok(())
    }
    fn write_random_lines(&mut self) -> Result<()> {
        let bytes = self.bytes;
        let data = self.data;
        self.write_labeled_u8_array_line("바이트 배열: ".as_bytes(), &bytes)?;
        self.write_labeled_u8_array_line("로또 번호: ".as_bytes(), &data.lotto_numbers)?;
        self.write_labeled_u8_array_line("일본 로또 7 번호: ".as_bytes(), &data.lotto7_numbers)?;
        self.write_labeled_line("6자리 숫자 비밀번호: ".as_bytes(), |buffer_cur| {
            if data.numeric_password >= 1_000_000 {
                return buf_write_u32_dec(buffer_cur, data.numeric_password);
            }
            let hi = usize::from(low_u8_from_u32(data.numeric_password.div_euclid(10_000)));
            let rem = usize::from(low_u16_from_u64(u64::from(
                data.numeric_password.rem_euclid(10_000),
            )));
            let head = (buffer_cur.take(6))?;
            let (hi_digits, rest) = head.split_at_mut(2);
            hi_digits.copy_from_slice((two_digits(hi))?);
            let (mid_digits, lo_digits) = rest.split_at_mut(2);
            mid_digits.copy_from_slice((two_digits(rem.div_euclid(100)))?);
            lo_digits.copy_from_slice((two_digits(rem.rem_euclid(100)))?);
            Ok(())
        })?;
        self.write_labeled_line("8자리 비밀번호: ".as_bytes(), |buffer_cur| {
            let head = (buffer_cur.take(8))?;
            let mut index = 0_usize;
            while index < 8 {
                let ch = *(data.password.get(index).ok_or_else(write_zero_err))?;
                let slot = (head.get_mut(index).ok_or_else(write_zero_err))?;
                *slot = (u8::try_from(u32::from(ch)).map_err(|err| {
                    ioErr::new(
                        ErrorKind::InvalidData,
                        format!("password contains non-ASCII character: {err}"),
                    )
                }))?;
                index = (checked_add_index(index, 1))?;
            }
            Ok(())
        })?;
        self.write_labeled_line("유로밀리언 번호: ".as_bytes(), |buffer_cur| {
            (buf_write_u8_array_spaced(buffer_cur, &data.euro_millions_main_numbers))?;
            (buf_write_bytes(buffer_cur, b" + "))?;
            buf_write_u8_array_spaced(buffer_cur, &data.euro_millions_lucky_stars)
        })?;
        self.write_labeled_line("한글 음절 4글자: ".as_bytes(), |buffer_cur| {
            buf_write_chars(buffer_cur, &data.hangul_syllables)
        })?;
        let consumed = {
            let mut tail_slice = self.cursor.tail_mut();
            let before = tail_slice.len();
            tail_slice.write_fmt(format_args!(
                "대한민국 위경도: {kor_lat}, {kor_lon}\n세계 위경도: {world_lat}, {world_lon}\n",
                kor_lat = data.kor_coords.0,
                kor_lon = data.kor_coords.1,
                world_lat = data.world_coords.0,
                world_lon = data.world_coords.1
            ))?;
            let after = tail_slice.len();
            (checked_sub_index(before, after))?
        };
        self.cursor.advance_by(consumed)?;
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
        bytes: u64_to_be_bytes(data.num_64),
        cursor: &mut cur,
        data,
        number64: data.num_64,
        signed_number: data.num_64.cast_signed(),
        use_colors,
    };
    formatter.write_all()?;
    Ok(cur.written_len())
}
fn checked_add_index(value: usize, amount: usize) -> IoRst<usize> {
    value.checked_add(amount).ok_or_else(write_zero_err)
}
fn checked_sub_index(value: usize, amount: usize) -> IoRst<usize> {
    value.checked_sub(amount).ok_or_else(write_zero_err)
}
fn add_to_index(value: &mut usize, amount: usize) -> IoRst<()> {
    *value = (checked_add_index(*value, amount))?;
    Ok(())
}
fn sub_from_index(value: &mut usize, amount: usize) -> IoRst<()> {
    *value = (checked_sub_index(*value, amount))?;
    Ok(())
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
fn range_slice_mut(slice: &mut [u8], start: usize, len: usize) -> IoRst<&mut [u8]> {
    let end = (start.checked_add(len).ok_or_else(write_zero_err))?;
    slice.get_mut(start..end).ok_or_else(write_zero_err)
}
fn digit_byte(index: usize) -> IoRst<u8> {
    DIGITS.get(index).copied().ok_or_else(write_zero_err)
}
fn two_digits(index: usize) -> IoRst<&'static [u8; 2]> {
    TWO_DIGITS.get(index).ok_or_else(write_zero_err)
}
fn hex_byte(index: usize) -> IoRst<&'static [u8; 2]> {
    HEX_BYTE_TABLE.get(index).ok_or_else(write_zero_err)
}
fn buf_write_chars<const N: usize>(cur: &mut BufCursor<'_>, chars: &[char; N]) -> IoRst<()> {
    let mut total = 0_usize;
    let mut i = 0_usize;
    while i < N {
        let ch = (chars.get(i).ok_or_else(write_zero_err))?;
        total = (checked_add_index(total, ch.len_utf8()))?;
        i = (checked_add_index(i, 1))?;
    }
    let head = (cur.take(total))?;
    let mut pos = 0_usize;
    let mut j = 0_usize;
    while j < N {
        let ch = (chars.get(j).ok_or_else(write_zero_err))?;
        let tail = (head.get_mut(pos..).ok_or_else(write_zero_err))?;
        let written = ch.encode_utf8(tail).len();
        pos = (checked_add_index(pos, written))?;
        j = (checked_add_index(j, 1))?;
    }
    Ok(())
}
fn buf_write_u8_dec(cur: &mut BufCursor<'_>, n: u8) -> IoRst<()> {
    if n >= 100 {
        let hundreds = usize::from(n.div_euclid(100));
        let rem = usize::from(n.rem_euclid(100));
        let head = (cur.take(3))?;
        let (hundreds_digit, remaining_digits) = head.split_at_mut(1);
        if let Some(slot) = hundreds_digit.first_mut() {
            *slot = (digit_byte(hundreds))?;
        }
        remaining_digits.copy_from_slice((two_digits(rem))?);
        return Ok(());
    }
    if n >= 10 {
        let head = (cur.take(2))?;
        head.copy_from_slice((two_digits(usize::from(n)))?);
        return Ok(());
    }
    let head = (cur.take(1))?;
    if let Some(slot) = head.first_mut() {
        *slot = (digit_byte(usize::from(n)))?;
    }
    Ok(())
}
fn buf_write_u8_array_spaced<const N: usize>(cur: &mut BufCursor<'_>, nums: &[u8; N]) -> IoRst<()> {
    let mut total = N.saturating_sub(1);
    let mut i = 0_usize;
    while i < N {
        let n = *(nums.get(i).ok_or_else(write_zero_err))?;
        total = (checked_add_index(
            total,
            if n >= 100 {
                3
            } else if n >= 10 {
                2
            } else {
                1
            },
        ))?;
        i = (checked_add_index(i, 1))?;
    }
    let head = (cur.take(total))?;
    let mut pos = 0_usize;
    let mut j = 0_usize;
    while j < N {
        if j != 0 {
            let slot = (head.get_mut(pos).ok_or_else(write_zero_err))?;
            *slot = b' ';
            (add_to_index(&mut pos, 1))?;
        }
        let n = *(nums.get(j).ok_or_else(write_zero_err))?;
        if n >= 100 {
            let hundreds = usize::from(n.div_euclid(100));
            let rem = usize::from(n.rem_euclid(100));
            let slot = (range_slice_mut(head, pos, 3))?;
            let (hundreds_digit, remaining_digits) = slot.split_at_mut(1);
            if let Some(digit_slot) = hundreds_digit.first_mut() {
                *digit_slot = (digit_byte(hundreds))?;
            }
            remaining_digits.copy_from_slice((two_digits(rem))?);
            (add_to_index(&mut pos, 3))?;
        } else if n >= 10 {
            (range_slice_mut(head, pos, 2))?.copy_from_slice((two_digits(usize::from(n)))?);
            (add_to_index(&mut pos, 2))?;
        } else {
            let slot = (head.get_mut(pos).ok_or_else(write_zero_err))?;
            *slot = (digit_byte(usize::from(n)))?;
            (add_to_index(&mut pos, 1))?;
        }
        j = (checked_add_index(j, 1))?;
    }
    Ok(())
}
fn buf_write_hash_hex24_from_bytes(cur: &mut BufCursor<'_>, b0: u8, b1: u8, b2: u8) -> IoRst<()> {
    let head = (cur.take(7))?;
    let (prefix, hex_bytes) = head.split_at_mut(1);
    if let Some(slot) = prefix.first_mut() {
        *slot = b'#';
    }
    let (first_hex, remaining_hex_bytes) = hex_bytes.split_at_mut(2);
    first_hex.copy_from_slice((hex_byte(usize::from(b0)))?);
    let (second_hex, third_hex) = remaining_hex_bytes.split_at_mut(2);
    second_hex.copy_from_slice((hex_byte(usize::from(b1)))?);
    third_hex.copy_from_slice((hex_byte(usize::from(b2)))?);
    Ok(())
}
fn buf_write_m_hash_hex24_from_bytes(cur: &mut BufCursor<'_>, b0: u8, b1: u8, b2: u8) -> IoRst<()> {
    let head = (cur.take(8))?;
    let (prefix, hex_bytes) = head.split_at_mut(2);
    let (m_prefix, hash_prefix) = prefix.split_at_mut(1);
    if let Some(slot) = m_prefix.first_mut() {
        *slot = b'm';
    }
    if let Some(slot) = hash_prefix.first_mut() {
        *slot = b'#';
    }
    let (first_hex, remaining_hex_bytes) = hex_bytes.split_at_mut(2);
    first_hex.copy_from_slice((hex_byte(usize::from(b0)))?);
    let (second_hex, third_hex) = remaining_hex_bytes.split_at_mut(2);
    second_hex.copy_from_slice((hex_byte(usize::from(b1)))?);
    third_hex.copy_from_slice((hex_byte(usize::from(b2)))?);
    Ok(())
}
fn buf_write_u32_dec(cur: &mut BufCursor<'_>, mut n: u32) -> IoRst<()> {
    let mut tmp = [0_u8; 10];
    let mut i = tmp.len();
    while n >= 100 {
        let rem = usize::from(low_u8_from_u32(n.rem_euclid(100)));
        n = n.div_euclid(100);
        (sub_from_index(&mut i, 2))?;
        (range_slice_mut(&mut tmp, i, 2))?.copy_from_slice((two_digits(rem))?);
    }
    if n >= 10 {
        let rem = usize::from(low_u8_from_u32(n));
        (sub_from_index(&mut i, 2))?;
        (range_slice_mut(&mut tmp, i, 2))?.copy_from_slice((two_digits(rem))?);
    } else {
        (sub_from_index(&mut i, 1))?;
        let digit = low_u8_from_u32(n);
        let slot = (tmp.get_mut(i).ok_or_else(write_zero_err))?;
        *slot = (digit_byte(usize::from(digit)))?;
    }
    buf_write_bytes(cur, (tmp.get(i..).ok_or_else(write_zero_err))?)
}
fn buf_write_u64_dec(cur: &mut BufCursor<'_>, mut n: u64) -> IoRst<()> {
    let mut tmp = [0_u8; 20];
    let mut i = tmp.len();
    while n >= 100 {
        let rem = usize::from(low_u8_from_u64(n.rem_euclid(100)));
        n = n.div_euclid(100);
        (sub_from_index(&mut i, 2))?;
        (range_slice_mut(&mut tmp, i, 2))?.copy_from_slice((two_digits(rem))?);
    }
    if n >= 10 {
        let rem = usize::from(low_u8_from_u64(n));
        (sub_from_index(&mut i, 2))?;
        (range_slice_mut(&mut tmp, i, 2))?.copy_from_slice((two_digits(rem))?);
    } else {
        (sub_from_index(&mut i, 1))?;
        let digit = low_u8_from_u64(n);
        let slot = (tmp.get_mut(i).ok_or_else(write_zero_err))?;
        *slot = (digit_byte(usize::from(digit)))?;
    }
    buf_write_bytes(cur, (tmp.get(i..).ok_or_else(write_zero_err))?)
}
fn buf_write_hex_u16_0pad4(cur: &mut BufCursor<'_>, value: u16) -> IoRst<()> {
    let head = (cur.take(4))?;
    let upper = usize::from(low_u8_from_u32(u32::from(value >> 8_u32)));
    let lower = usize::from(low_u8_from_u32(u32::from(value)));
    let (upper_hex, lower_hex) = head.split_at_mut(2);
    upper_hex.copy_from_slice((hex_byte(upper))?);
    lower_hex.copy_from_slice((hex_byte(lower))?);
    Ok(())
}
fn buf_write_hex_u16_min3(cur: &mut BufCursor<'_>, value: u16) -> IoRst<()> {
    if value < 0x1000 {
        let head = (cur.take(3))?;
        let hi = usize::from(low_u8_from_u32(u32::from(value >> 8)));
        let lo = usize::from(low_u8_from_u32(u32::from(value)));
        let (prefix, suffix) = head.split_at_mut(1);
        if let Some(slot) = prefix.first_mut() {
            *slot = (HEX_UPPER.get(hi).copied().ok_or_else(write_zero_err))?;
        }
        suffix.copy_from_slice((hex_byte(lo))?);
        Ok(())
    } else {
        buf_write_hex_u16_0pad4(cur, value)
    }
}
pub fn write_buffer_to_file_guard(
    file_guard: &mut MutexGuard<BufWriter<File>>,
    buffer: &[u8],
) -> IoRst<()> {
    file_guard.write_all(buffer)
}
pub fn write_slice_to_console(data_slice: &[u8]) -> IoRst<()> {
    let mut stdout_lock = stdout().lock();
    stdout_lock.write_all(data_slice)?;
    stdout_lock.flush()
}
pub fn print_progress(
    out: &mut dyn Write,
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
    let elapsed_deci = elapsed_millis.div_euclid(100);
    let eta_deci = if total == 0 || completed >= total {
        Some(0)
    } else if completed == 0 {
        None
    } else {
        let remaining = (total
            .checked_sub(completed)
            .ok_or_else(|| ioErr::other("남은 작업 수 계산 실패")))?;
        let completed_scaled = (u128::from(completed)
            .checked_mul(100)
            .ok_or_else(|| ioErr::other("ETA 분모 계산 실패")))?;
        Some(
            (elapsed_millis
                .saturating_mul(u128::from(remaining))
                .checked_div(completed_scaled)
                .ok_or_else(|| ioErr::other("ETA 계산 실패")))?,
        )
    };
    let elapsed_len = format_time_into(Some(elapsed_deci), elapsed_buf);
    let eta_len = format_time_into(eta_deci, eta_buf);
    let filled_u64 = if total == 0 {
        BAR_WIDTH_U64
    } else {
        (completed
            .saturating_mul(BAR_WIDTH_U64)
            .checked_div(total)
            .ok_or_else(|| ioErr::other("진행 막대 계산 실패")))?
    };
    let filled = usize::from(low_u8_from_u64(filled_u64.min(BAR_WIDTH_U64)));
    let bar = (BAR_FULL
        .get(filled)
        .copied()
        .ok_or_else(|| ioErr::other("진행 막대 인덱스 범위 초과")))?;
    let percent_u64 = if total == 0 {
        100
    } else {
        (completed
            .saturating_mul(100)
            .checked_div(total)
            .ok_or_else(|| ioErr::other("진행률 계산 실패")))?
    };
    let percent = low_u8_from_u64(percent_u64.min(100));
    let mut line = [0_u8; 128];
    let mut cur = BufCursor::new(&mut line);
    (buf_write_byte(&mut cur, b'\r'))?;
    (buf_write_bytes(&mut cur, bar.as_bytes()))?;
    (buf_write_byte(&mut cur, b' '))?;
    match percent {
        0..=9 => (buf_write_bytes(&mut cur, b"  "))?,
        10..=99 => (buf_write_byte(&mut cur, b' '))?,
        _ => {}
    }
    (buf_write_u8_dec(&mut cur, percent))?;
    (buf_write_byte(&mut cur, b'%'))?;
    (buf_write_bytes(&mut cur, b" ("))?;
    (buf_write_u64_dec(&mut cur, completed))?;
    (buf_write_byte(&mut cur, b'/'))?;
    (buf_write_u64_dec(&mut cur, total))?;
    (buf_write_bytes(&mut cur, ") | 소요: ".as_bytes()))?;
    (buf_write_bytes(&mut cur, (prefix_slice(elapsed_buf, elapsed_len))?))?;
    (buf_write_bytes(&mut cur, b" | ETA: "))?;
    (buf_write_bytes(&mut cur, (prefix_slice(eta_buf, eta_len))?))?;
    (buf_write_bytes(&mut cur, b" \x1b[K"))?;
    let used = cur.written_len();
    (out.write_all((prefix_slice(&line, used))?))?;
    (out.flush())?;
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
    let minutes = usize::from(low_u8_from_u128((deci.div_euclid(600)).min(99)));
    let sec_whole = usize::from(low_u8_from_u128(deci.div_euclid(10).rem_euclid(60)));
    let tenths = usize::from(low_u8_from_u128(deci.rem_euclid(10)));
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
