use crate::{
    BUFFER_SIZE, FILE_RECORD_FINAL_LABEL, FILE_RECORD_START,
    buffmt::{ByteCursor, digit_byte, two_digits},
    numeric::{low_u8_from_u32, low_u8_from_u64},
    random_data::RandomDataSet,
};
use core::fmt::NumBuffer;
use std::{
    io::{Result as IoResult, Write as IoWrite, stdout},
    process,
};
#[cfg(target_arch = "x86_64")]
pub(super) mod progress;
#[cfg(target_arch = "x86_64")]
pub(super) const PROGRESS_LINE_BUF_LEN: usize = 128;
const BYTE_GROUP_COUNT: usize = 8;
const COLORED_HEX_BUF_LEN: usize = 64;
const HEX_CODE_LABEL: &[u8] = "Hex 코드: ".as_bytes();
const HEX_CODE_MIN_OFFSET: usize = 175;
const HEX_U16_FULL_WIDTH: usize = 4;
static HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";
const NIBBLE_MASK_U64: u64 = 0xF;
const NMS_GLYPH_COUNT: usize = 12;
const NMS_GLYPH_TOP_SHIFT: u32 = 44;
static NMS_GLYPHS: [&str; 16] = [
    "🌅", "🐦", "👫", "🦕", "🌘", "🎈", "⛵", "🕷", "🦋", "🌀", "🧊", "🐟", "⛺", "🚀", "🌳", "🔯",
];
const OCTAL_DIGIT_MASK: u64 = 7;
const OCTAL_SHIFT_BITS: u32 = 3;
const PASSWORD_HIGH_DIVISOR: u32 = 10_000;
const PASSWORD_WIDTH: usize = 6;
const PLAIN_HEX_CODE_LEN: usize = 15;
const TWO_DIGIT_WIDTH: usize = 2;
const U8_THREE_DIGIT_THRESHOLD: u8 = 100;
const U8_TWO_DIGIT_THRESHOLD: u8 = 10;
static BINARY_NIBBLES: [[u8; 4]; 16] = [
    *b"0000", *b"0001", *b"0010", *b"0011", *b"0100", *b"0101", *b"0110", *b"0111", *b"1000",
    *b"1001", *b"1010", *b"1011", *b"1100", *b"1101", *b"1110", *b"1111",
];
#[derive(Clone, Copy)]
pub(super) enum OutputTarget {
    Console(usize),
    File,
}
struct OutputFormatter<'cursor, 'buffer, 'data> {
    bytes: [u8; 8],
    cursor: &'cursor mut ByteCursor<'buffer>,
    data: &'data RandomDataSet,
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
            buffer_cur.write_u64_dec(u64::from(galaxy_number_byte).strict_add(1));
        });
        self.write_labeled_line("NMS 포탈 주소: ".as_bytes(), |buffer_cur| {
            buf_write_u8_dec(buffer_cur, data.planet_number);
            buffer_cur.write_byte(b' ');
            buf_write_hex_u16_3(buffer_cur, data.solar_system_index);
            buffer_cur.write_byte(b' ');
            buffer_cur.write_bytes(&hex_byte(data.nms_portal_yy));
            buffer_cur.write_byte(b' ');
            buf_write_hex_u16_3(buffer_cur, data.nms_portal_zzz);
            buffer_cur.write_byte(b' ');
            buf_write_hex_u16_3(buffer_cur, data.nms_portal_xxx);
            buffer_cur.write_byte(b' ');
            buffer_cur.write_byte(b'(');
            let mut glyph_bits = (u64::from(data.planet_number) << 44_u32)
                | (u64::from(data.solar_system_index) << 32_u32)
                | ((data.num_64 >> 8_u32) & u64::from(u32::MAX));
            for _ in 0..NMS_GLYPH_COUNT {
                let index = usize::from(low_u8_from_u64(
                    (glyph_bits >> NMS_GLYPH_TOP_SHIFT) & NIBBLE_MASK_U64,
                ));
                let glyph = NMS_GLYPHS
                    .get(index)
                    .copied()
                    .unwrap_or_else(|| process::abort());
                buffer_cur.write_bytes(glyph.as_bytes());
                glyph_bits <<= 4_u32;
            }
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
        self.cursor.write_bytes(FILE_RECORD_START);
        let mut number_buffer = NumBuffer::new();
        let number_text = number64.format_into(&mut number_buffer).as_bytes();
        self.cursor.write_bytes(number_text);
        self.cursor.write_bytes(" (유부호 정수: ".as_bytes());
        if signed_number < 0 {
            self.cursor.write_byte(b'-');
            self.cursor.write_u64_dec(signed_number.unsigned_abs());
        } else {
            self.cursor.write_bytes(number_text);
        }
        self.cursor.write_bytes(b")\n");
        self.write_prefixed_byte_groups("2진수: ", |byte| {
            let [[h0, h1, h2, h3], [l0, l1, l2, l3]] = [byte >> 4_u8, byte & 0x0f].map(|nibble| {
                BINARY_NIBBLES
                    .get(usize::from(nibble))
                    .copied()
                    .unwrap_or_else(|| process::abort())
            });
            [h0, h1, h2, h3, l0, l1, l2, l3]
        });
        self.write_labeled_line("8진수: ".as_bytes(), |buffer_cur| {
            let octal_len = usize::from(low_u8_from_u32(
                u64::BITS
                    .strict_sub(number64.leading_zeros())
                    .div_ceil(OCTAL_SHIFT_BITS)
                    .max(1),
            ));
            let mut octal_number = number64;
            for slot in buffer_cur.take(octal_len).iter_mut().rev() {
                *slot = digit_byte(low_u8_from_u64(octal_number & OCTAL_DIGIT_MASK));
                octal_number >>= OCTAL_SHIFT_BITS;
            }
        });
        self.write_prefixed_byte_groups("16진수: ", hex_byte);
        self.write_labeled_line(HEX_CODE_LABEL, |buffer_cur| {
            let [b0, b1, b2, b3, b4, b5, _, _] = bytes;
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
        for (group_out, byte) in groups.chunks_exact_mut(WIDTH.strict_add(1)).zip(self.bytes) {
            let (value_out, separator_out) = group_out.split_at_mut(WIDTH);
            value_out.copy_from_slice(&render_group(byte));
            separator_out.fill(b' ');
        }
        *groups.last_mut().unwrap_or_else(|| process::abort()) = b'\n';
    }
    fn write_random_lines(&mut self) {
        let bytes = self.bytes;
        let data = self.data;
        self.write_labeled_u8_array_line("바이트 배열: ".as_bytes(), &bytes);
        self.write_labeled_line("6자리 숫자 비밀번호: ".as_bytes(), |buffer_cur| {
            let hi = low_u8_from_u32(data.numeric_password.div_euclid(PASSWORD_HIGH_DIVISOR));
            let rem = data.numeric_password.rem_euclid(PASSWORD_HIGH_DIVISOR);
            let [h0, h1] = two_digits(hi);
            let [m0, m1] = two_digits(low_u8_from_u32(
                rem.div_euclid(u32::from(U8_THREE_DIGIT_THRESHOLD)),
            ));
            let [l0, l1] = two_digits(low_u8_from_u32(
                rem.rem_euclid(u32::from(U8_THREE_DIGIT_THRESHOLD)),
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
            let (chunks, _) = buffer_cur.take(12).as_chunks_mut::<3>();
            for (&ch, chunk) in data.hangul_syllables.iter().zip(chunks) {
                ch.encode_utf8(chunk);
            }
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
    let bytes = data.num_64.to_be_bytes();
    if let OutputTarget::Console(file_len) = target {
        let [b0, b1, b2, b3, b4, b5, _, _] = bytes;
        let mut colored = [0_u8; COLORED_HEX_BUF_LEN];
        let mut color_cur = ByteCursor::new(colored.as_mut_slice());
        color_cur.write_bytes(b"\x1B[38;2;");
        buf_write_u8_dec(&mut color_cur, b0);
        color_cur.write_byte(b';');
        buf_write_u8_dec(&mut color_cur, b1);
        color_cur.write_byte(b';');
        buf_write_u8_dec(&mut color_cur, b2);
        buf_write_prefixed_hex24(&mut color_cur, b"m#", b0, b1, b2);
        color_cur.write_bytes(b"\x1B[0m \x1B[38;2;");
        buf_write_u8_dec(&mut color_cur, b3);
        color_cur.write_byte(b';');
        buf_write_u8_dec(&mut color_cur, b4);
        color_cur.write_byte(b';');
        buf_write_u8_dec(&mut color_cur, b5);
        buf_write_prefixed_hex24(&mut color_cur, b"m#", b3, b4, b5);
        color_cur.write_bytes(b"\x1B[0m");
        let colored_len = color_cur.written_len();
        let value_start = buffer
            .get(HEX_CODE_MIN_OFFSET..file_len)
            .and_then(|slice| slice.iter().position(|&byte| byte == b'#'))
            .map_or_else(
                || process::abort(),
                |offset| HEX_CODE_MIN_OFFSET.strict_add(offset),
            );
        let value_end = value_start.strict_add(PLAIN_HEX_CODE_LEN);
        let extra = colored_len.strict_sub(PLAIN_HEX_CODE_LEN);
        buffer.copy_within(value_end..file_len, value_end.strict_add(extra));
        buffer
            .get_mut(value_start..value_start.strict_add(colored_len))
            .unwrap_or_else(|| process::abort())
            .copy_from_slice(colored.split_at(colored_len).0);
        return file_len.strict_add(extra);
    }
    let mut cur = ByteCursor::new(buffer.as_mut_slice());
    let mut formatter = OutputFormatter {
        bytes,
        cursor: &mut cur,
        data,
    };
    formatter.write_number_lines();
    formatter.write_random_lines();
    formatter.write_nms_lines();
    cur.written_len()
}
fn hex_byte(byte: u8) -> [u8; 2] {
    [byte >> 4_u8, byte & 0x0f].map(|nibble| {
        HEX_DIGITS
            .get(usize::from(nibble))
            .copied()
            .unwrap_or_else(|| process::abort())
    })
}
fn hex_u16(value: u16) -> [u8; HEX_U16_FULL_WIDTH] {
    let [[h0, h1], [h2, h3]] = value.to_be_bytes().map(hex_byte);
    [h0, h1, h2, h3]
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
    let Some((&first, rest)) = nums.split_first() else {
        return;
    };
    buf_write_u8_dec(cur, first);
    for &n in rest {
        cur.write_byte(b' ');
        buf_write_u8_dec(cur, n);
    }
}
fn buf_write_prefixed_hex24(cur: &mut ByteCursor<'_>, prefix: &[u8], b0: u8, b1: u8, b2: u8) {
    let head = cur.take(prefix.len().strict_add(6));
    let (prefix_out, hex_bytes) = head.split_at_mut(prefix.len());
    prefix_out.copy_from_slice(prefix);
    let hex = [b0, b1, b2].map(hex_byte);
    hex_bytes.copy_from_slice(hex.as_flattened());
}
fn buf_write_hex_u16_3(cur: &mut ByteCursor<'_>, value: u16) {
    let [_, h1, h2, h3] = hex_u16(value);
    cur.write_bytes(&[h1, h2, h3]);
}
pub(super) fn write_slice_to_console(data_slice: &[u8]) -> IoResult<()> {
    let mut stdout_lock = stdout().lock();
    IoWrite::write_all(&mut stdout_lock, data_slice)?;
    IoWrite::flush(&mut stdout_lock)
}
