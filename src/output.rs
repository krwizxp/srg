use crate::{
    BUFFER_SIZE, FILE_RECORD_FINAL_LABEL, FILE_RECORD_START,
    buffmt::{ByteCursor, digit_byte, two_digits},
    numeric::{low_u8_from_u32, low_u8_from_u64, low_u16_from_u64},
    random_data::RandomDataSet,
};
use std::io::{Result as IoResult, Write as IoWrite, stdout};
#[cfg(target_arch = "x86_64")]
pub(super) mod progress;
#[cfg(target_arch = "x86_64")]
pub(super) const PROGRESS_LINE_BUF_LEN: usize = 128;
const BYTE_GROUP_COUNT: usize = 8;
const HEX_U16_FULL_WIDTH: usize = 4;
const HEX_U16_SHORT_THRESHOLD: u16 = 0x1000;
const OCTAL_DIGIT_MASK: u64 = 7;
const OCTAL_SHIFT_BITS: u32 = 3;
const OCTAL_TMP_LEN: usize = 22;
const PASSWORD_FULL_WIDTH_THRESHOLD: u32 = 1_000_000;
const PASSWORD_HIGH_DIVISOR: u32 = 10_000;
const PASSWORD_WIDTH: usize = 6;
const TWO_DIGIT_WIDTH: usize = 2;
const U8_THREE_DIGIT_THRESHOLD: u8 = 100;
const U8_TWO_DIGIT_THRESHOLD: u8 = 10;
#[derive(Clone, Copy)]
pub(super) enum OutputTarget {
    Console,
    File,
}
struct OutputFormatter<'cursor, 'buffer, 'data> {
    bytes: [u8; 8],
    cursor: &'cursor mut ByteCursor<'buffer>,
    data: &'data RandomDataSet,
    use_colors: bool,
}
impl OutputFormatter<'_, '_, '_> {
    fn write_labeled_line<F>(&mut self, label: &[u8], write_value: F)
    where
        F: FnOnce(&mut ByteCursor<'_>),
    {
        self.cursor.write_bytes(label);
        write_value(self.cursor);
        self.cursor.write_byte(b'\n');
    }
    fn write_labeled_u8_array_line<const N: usize>(&mut self, label: &[u8], values: &[u8; N]) {
        self.write_labeled_line(label, |buffer_cur| {
            buf_write_u8_array_spaced(buffer_cur, values);
        });
    }
    fn write_nms_lines(&mut self) {
        let [galaxy_number_byte, ..] = self.bytes;
        let data = self.data;
        self.write_labeled_line("NMS 은하 번호: ".as_bytes(), |buffer_cur| {
            buffer_cur.write_u32_dec(u32::from(galaxy_number_byte).strict_add(1));
        });
        self.write_labeled_line("NMS 포탈 주소: ".as_bytes(), |buffer_cur| {
            buf_write_u8_dec(buffer_cur, data.planet_number);
            buffer_cur.write_byte(b' ');
            buf_write_hex_u16_min3(buffer_cur, data.solar_system_index);
            buffer_cur.write_byte(b' ');
            buffer_cur.write_bytes(&hex_byte(data.nms_portal_yy));
            buffer_cur.write_byte(b' ');
            buf_write_hex_u16_min3(buffer_cur, data.nms_portal_zzz);
            buffer_cur.write_byte(b' ');
            buf_write_hex_u16_min3(buffer_cur, data.nms_portal_xxx);
            buffer_cur.write_byte(b' ');
            buffer_cur.write_byte(b'(');
            buf_write_chars(buffer_cur, &data.glyph_string);
            buffer_cur.write_bytes(b")");
        });
        self.write_labeled_line(FILE_RECORD_FINAL_LABEL, |buffer_cur| {
            buffer_cur.write_bytes(&hex_u16(data.galaxy_x));
            buffer_cur.write_byte(b':');
            buffer_cur.write_bytes(&hex_u16(data.galaxy_y));
            buffer_cur.write_byte(b':');
            buffer_cur.write_bytes(&hex_u16(data.galaxy_z));
            buffer_cur.write_byte(b':');
            buffer_cur.write_bytes(&hex_u16(data.solar_system_index));
        });
    }
    fn write_number_lines(&mut self) {
        let number64 = self.data.num_64;
        let signed_number = number64.cast_signed();
        let bytes = self.bytes;
        let use_colors = self.use_colors;
        self.cursor.write_bytes(FILE_RECORD_START);
        self.cursor.write_u64_dec(number64);
        self.cursor.write_bytes(" (유부호 정수: ".as_bytes());
        if signed_number < 0 {
            self.cursor.write_byte(b'-');
        }
        self.cursor.write_u64_dec(signed_number.unsigned_abs());
        self.cursor.write_bytes(b")\n");
        self.write_prefixed_byte_groups("2진수: ", |byte| {
            [
                bit(byte, 0b1000_0000),
                bit(byte, 0b0100_0000),
                bit(byte, 0b0010_0000),
                bit(byte, 0b0001_0000),
                bit(byte, 0b0000_1000),
                bit(byte, 0b0000_0100),
                bit(byte, 0b0000_0010),
                bit(byte, 0b0000_0001),
            ]
        });
        self.write_labeled_line("8진수: ".as_bytes(), |buffer_cur| {
            if number64 == 0 {
                buffer_cur.write_byte(b'0');
                return;
            }
            let mut tmp = [0_u8; OCTAL_TMP_LEN];
            let mut index = tmp.len();
            let mut octal_number = number64;
            for slot in tmp.iter_mut().rev() {
                if octal_number == 0 {
                    break;
                }
                *slot = digit_byte(low_u8_from_u64(octal_number & OCTAL_DIGIT_MASK));
                octal_number >>= OCTAL_SHIFT_BITS;
                index = index.strict_sub(1);
            }
            buffer_cur.write_bytes(tmp.split_at(index).1);
        });
        self.write_prefixed_byte_groups("16진수: ", hex_byte);
        self.write_labeled_line("Hex 코드: ".as_bytes(), |buffer_cur| {
            let [b0, b1, b2, b3, b4, b5, _, _] = bytes;
            if use_colors {
                buffer_cur.write_bytes(b"\x1B[38;2;");
                buf_write_u8_dec(buffer_cur, b0);
                buffer_cur.write_byte(b';');
                buf_write_u8_dec(buffer_cur, b1);
                buffer_cur.write_byte(b';');
                buf_write_u8_dec(buffer_cur, b2);
                buf_write_prefixed_hex24(buffer_cur, b"m#", b0, b1, b2);
                buffer_cur.write_bytes(b"\x1B[0m \x1B[38;2;");
                buf_write_u8_dec(buffer_cur, b3);
                buffer_cur.write_byte(b';');
                buf_write_u8_dec(buffer_cur, b4);
                buffer_cur.write_byte(b';');
                buf_write_u8_dec(buffer_cur, b5);
                buf_write_prefixed_hex24(buffer_cur, b"m#", b3, b4, b5);
                buffer_cur.write_bytes(b"\x1B[0m");
                return;
            }
            buf_write_prefixed_hex24(buffer_cur, b"#", b0, b1, b2);
            buffer_cur.write_byte(b' ');
            buf_write_prefixed_hex24(buffer_cur, b"#", b3, b4, b5);
        });
    }
    fn write_prefixed_byte_groups<const WIDTH: usize>(
        &mut self,
        prefix: &'static str,
        mut render_group: impl FnMut(u8) -> [u8; WIDTH],
    ) {
        let prefix_bytes = prefix.as_bytes();
        let prefix_len = prefix_bytes.len();
        let line_len = prefix_len
            .strict_add(WIDTH.strict_mul(BYTE_GROUP_COUNT))
            .strict_add(BYTE_GROUP_COUNT);
        let head = self.cursor.take(line_len);
        let (prefix_out, groups) = head.split_at_mut(prefix_len);
        prefix_out.copy_from_slice(prefix_bytes);
        let last_index = BYTE_GROUP_COUNT.strict_sub(1);
        let mut remaining = groups;
        for (index, byte) in self.bytes.into_iter().enumerate() {
            let (group_out, tail) = remaining.split_at_mut(WIDTH.strict_add(1));
            remaining = tail;
            let group = render_group(byte);
            let (value_out, separator_out) = group_out.split_at_mut(WIDTH);
            value_out.copy_from_slice(&group);
            separator_out.fill(if index == last_index { b'\n' } else { b' ' });
        }
    }
    fn write_random_lines(&mut self) {
        let bytes = self.bytes;
        let data = self.data;
        self.write_labeled_u8_array_line("바이트 배열: ".as_bytes(), &bytes);
        self.write_labeled_line("6자리 숫자 비밀번호: ".as_bytes(), |buffer_cur| {
            if data.numeric_password >= PASSWORD_FULL_WIDTH_THRESHOLD {
                buffer_cur.write_u32_dec(data.numeric_password);
                return;
            }
            let hi = low_u8_from_u32(data.numeric_password.div_euclid(PASSWORD_HIGH_DIVISOR));
            let rem = low_u16_from_u64(u64::from(
                data.numeric_password.rem_euclid(PASSWORD_HIGH_DIVISOR),
            ));
            let [h0, h1] = two_digits(hi);
            let [m0, m1] = two_digits(low_u8_from_u32(
                u32::from(rem).div_euclid(u32::from(U8_THREE_DIGIT_THRESHOLD)),
            ));
            let [l0, l1] = two_digits(low_u8_from_u32(
                u32::from(rem).rem_euclid(u32::from(U8_THREE_DIGIT_THRESHOLD)),
            ));
            *buffer_cur.take_array::<PASSWORD_WIDTH>() = [h0, h1, m0, m1, l0, l1];
        });
        self.write_labeled_line("8자리 비밀번호: ".as_bytes(), |buffer_cur| {
            buffer_cur.write_bytes(&data.password);
        });
        self.write_labeled_u8_array_line("로또 번호: ".as_bytes(), &data.lotto_numbers);
        self.write_labeled_u8_array_line("일본 로또 7 번호: ".as_bytes(), &data.lotto7_numbers);
        self.write_labeled_line("유로밀리언 번호: ".as_bytes(), |buffer_cur| {
            buf_write_u8_array_spaced(buffer_cur, &data.euro_millions_main_numbers);
            buffer_cur.write_bytes(b" + ");
            buf_write_u8_array_spaced(buffer_cur, &data.euro_millions_lucky_stars);
        });
        self.write_labeled_line("한글 음절 4글자: ".as_bytes(), |buffer_cur| {
            buf_write_chars(buffer_cur, &data.hangul_syllables);
        });
        self.cursor.write_bytes("대한민국 위경도: ".as_bytes());
        let kor_latitude = data.kor_coords.latitude;
        let kor_longitude = data.kor_coords.longitude;
        self.cursor
            .write_format(format_args!("{kor_latitude}, {kor_longitude}\n"));
        self.cursor.write_bytes("세계 위경도: ".as_bytes());
        let world_latitude = data.world_coords.latitude;
        let world_longitude = data.world_coords.longitude;
        self.cursor
            .write_format(format_args!("{world_latitude}, {world_longitude}\n"));
    }
}
pub(super) fn format_data_into_buffer(
    data: &RandomDataSet,
    buffer: &mut [u8; BUFFER_SIZE],
    target: OutputTarget,
) -> usize {
    let mut cur = ByteCursor::new(buffer.as_mut_slice());
    let mut formatter = OutputFormatter {
        bytes: data.num_64.to_be_bytes(),
        cursor: &mut cur,
        data,
        use_colors: matches!(target, OutputTarget::Console),
    };
    formatter.write_number_lines();
    formatter.write_random_lines();
    formatter.write_nms_lines();
    cur.written_slice().len()
}
pub(super) const fn prefix_slice(slice: &[u8], len: usize) -> &[u8] {
    slice.split_at(len).0
}
const fn bit(byte: u8, mask: u8) -> u8 {
    if byte & mask == 0 { b'0' } else { b'1' }
}
const fn hex_byte(byte: u8) -> [u8; 2] {
    [hex_digit(byte >> 4_u8), hex_digit(byte & 0x0F)]
}
const fn hex_digit(nibble: u8) -> u8 {
    if nibble < 10 {
        b'0'.strict_add(nibble)
    } else {
        b'A'.strict_add(nibble.strict_sub(10))
    }
}
const fn hex_u16(value: u16) -> [u8; HEX_U16_FULL_WIDTH] {
    let [upper, lower] = value.to_be_bytes();
    let [h0, h1] = hex_byte(upper);
    let [h2, h3] = hex_byte(lower);
    [h0, h1, h2, h3]
}
fn buf_write_chars<const N: usize>(cur: &mut ByteCursor<'_>, chars: &[char; N]) {
    let total = chars
        .iter()
        .fold(0_usize, |sum, ch| sum.strict_add(ch.len_utf8()));
    let mut tail = cur.take(total);
    for &ch in chars {
        let written = ch.encode_utf8(tail).len();
        tail = tail.split_at_mut(written).1;
    }
}
fn buf_write_u8_dec(cur: &mut ByteCursor<'_>, n: u8) {
    if n >= U8_THREE_DIGIT_THRESHOLD {
        let &mut [ref mut hundreds, ref mut tens, ref mut ones] = cur.take_array::<3>();
        *hundreds = digit_byte(n.div_euclid(U8_THREE_DIGIT_THRESHOLD));
        let [tens_value, ones_value] = two_digits(n.rem_euclid(U8_THREE_DIGIT_THRESHOLD));
        *tens = tens_value;
        *ones = ones_value;
        return;
    }
    if n >= U8_TWO_DIGIT_THRESHOLD {
        *cur.take_array::<TWO_DIGIT_WIDTH>() = two_digits(n);
        return;
    }
    cur.write_byte(digit_byte(n));
}
fn buf_write_u8_array_spaced<const N: usize>(cur: &mut ByteCursor<'_>, nums: &[u8; N]) {
    for (index, &n) in nums.iter().enumerate() {
        if index != 0 {
            cur.write_byte(b' ');
        }
        buf_write_u8_dec(cur, n);
    }
}
fn buf_write_prefixed_hex24(cur: &mut ByteCursor<'_>, prefix: &[u8], b0: u8, b1: u8, b2: u8) {
    let head = cur.take(prefix.len().strict_add(6));
    let (prefix_out, hex_bytes) = head.split_at_mut(prefix.len());
    prefix_out.copy_from_slice(prefix);
    let [b00, b01] = hex_byte(b0);
    let [b10, b11] = hex_byte(b1);
    let [b20, b21] = hex_byte(b2);
    hex_bytes.copy_from_slice(&[b00, b01, b10, b11, b20, b21]);
}
fn buf_write_hex_u16_min3(cur: &mut ByteCursor<'_>, value: u16) {
    if value < HEX_U16_SHORT_THRESHOLD {
        let [_, h1, h2, h3] = hex_u16(value);
        cur.write_bytes(&[h1, h2, h3]);
    } else {
        cur.write_bytes(&hex_u16(value));
    }
}
pub(super) fn write_slice_to_console(data_slice: &[u8]) -> IoResult<()> {
    let mut stdout_lock = stdout().lock();
    IoWrite::write_all(&mut stdout_lock, data_slice)?;
    IoWrite::flush(&mut stdout_lock)
}
