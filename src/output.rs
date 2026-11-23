use crate::{
    BUFFER_SIZE, FILE_RECORD_FINAL_LABEL, FILE_RECORD_START,
    buffmt::{ByteCursor, digit_byte, two_digits},
    diagnostic::or_abort,
    numeric::{low_u8_from_u32, low_u8_from_u64},
    random_data::RandomDataSet,
};
#[cfg(target_arch = "x86_64")]
use crate::{diagnostic::Result, numeric::low_u8_from_u128};
use core::fmt::NumBuffer;
#[cfg(target_arch = "x86_64")]
use core::time::Duration;
use std::{
    io::{Result as IoResult, Write as IoWrite, stdout},
    process,
};
static BINARY_NIBBLES: [[u8; 4]; 16] = [
    *b"0000", *b"0001", *b"0010", *b"0011", *b"0100", *b"0101", *b"0110", *b"0111", *b"1000",
    *b"1001", *b"1010", *b"1011", *b"1100", *b"1101", *b"1110", *b"1111",
];
static HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";
static NMS_GLYPHS: [[u8; 4]; 16] = [
    [0xF0, 0x9F, 0x8C, 0x85],
    [0xF0, 0x9F, 0x90, 0xA6],
    [0xF0, 0x9F, 0x91, 0xAB],
    [0xF0, 0x9F, 0xA6, 0x95],
    [0xF0, 0x9F, 0x8C, 0x98],
    [0xF0, 0x9F, 0x8E, 0x88],
    [0xE2, 0x9B, 0xB5, 0x00],
    [0xF0, 0x9F, 0x95, 0xB7],
    [0xF0, 0x9F, 0xA6, 0x8B],
    [0xF0, 0x9F, 0x8C, 0x80],
    [0xF0, 0x9F, 0xA7, 0x8A],
    [0xF0, 0x9F, 0x90, 0x9F],
    [0xE2, 0x9B, 0xBA, 0x00],
    [0xF0, 0x9F, 0x9A, 0x80],
    [0xF0, 0x9F, 0x8C, 0xB3],
    [0xF0, 0x9F, 0x94, 0xAF],
];
struct OutputFormatter<'cursor, 'buffer, 'data> {
    bytes: [u8; 8],
    cursor: &'cursor mut ByteCursor<'buffer>,
    data: &'data RandomDataSet,
}
#[derive(Clone, Copy)]
pub(super) enum OutputTarget {
    Console(usize),
    File,
}
#[cfg(target_arch = "x86_64")]
pub(super) struct ProgressBuffers([u8; 128]);
impl OutputFormatter<'_, '_, '_> {
    fn write_labeled_line(&mut self, label: &[u8], write_value: impl FnOnce(&mut ByteCursor<'_>)) {
        self.cursor.write_bytes(label);
        write_value(self.cursor);
        self.cursor.write_byte(b'\n');
    }
    fn write_labeled_u8_array_line<const N: usize>(&mut self, label: &[u8], values: &[u8; N]) {
        self.write_labeled_line(label, |cur| buf_write_u8_array_spaced(cur, values));
    }
    fn write_nms_lines(&mut self) {
        let ([galaxy_byte, ..], data) = (self.bytes, self.data);
        self.write_labeled_line("NMS 은하 번호: ".as_bytes(), |cur| {
            if galaxy_byte == u8::MAX {
                cur.write_bytes(b"256");
            } else {
                buf_write_u8_dec(cur, galaxy_byte.strict_add(1));
            }
        });
        self.write_labeled_line("NMS 포탈 주소: ".as_bytes(), |cur| {
            cur.write_byte(digit_byte(data.planet_number));
            cur.write_byte(b' ');
            buf_write_hex_u16_3(cur, data.solar_system_index);
            cur.write_byte(b' ');
            cur.write_bytes(&hex_byte(data.nms_portal_yy));
            cur.write_byte(b' ');
            buf_write_hex_u16_3(cur, data.nms_portal_zzz);
            cur.write_byte(b' ');
            buf_write_hex_u16_3(cur, data.nms_portal_xxx);
            cur.write_bytes(b" (");
            let mut bits = (u64::from(data.planet_number) << 44_u32)
                | (u64::from(data.solar_system_index) << 32_u32)
                | ((data.num_64 >> 8_u32) & u64::from(u32::MAX));
            for _ in 0..12_usize {
                let glyph = or_abort(
                    NMS_GLYPHS
                        .get(usize::from(low_u8_from_u64((bits >> 44_u32) & 0xF)))
                        .copied(),
                );
                cur.write_array_advance(glyph, 3_usize.strict_add(usize::from(glyph[3] != 0)));
                bits <<= 4_u32;
            }
            cur.write_byte(b')');
        });
        self.write_labeled_line(FILE_RECORD_FINAL_LABEL, |cur| {
            for (idx, val) in [
                data.galaxy_x,
                data.galaxy_y,
                data.galaxy_z,
                data.solar_system_index,
            ]
            .into_iter()
            .enumerate()
            {
                if idx != 0 {
                    cur.write_byte(b':');
                }
                cur.write_bytes(&hex_u16(val));
            }
        });
    }
    fn write_number_lines(&mut self) {
        let (num64, bytes) = (self.data.num_64, self.bytes);
        let signed = num64.cast_signed();
        self.cursor.write_bytes(FILE_RECORD_START);
        let mut num_buf = NumBuffer::new();
        let num_text = num64.format_into(&mut num_buf).as_bytes();
        self.cursor.write_bytes(num_text);
        self.cursor.write_bytes(" (유부호 정수: ".as_bytes());
        if signed < 0 {
            self.cursor.write_byte(b'-');
            self.cursor
                .write_bytes(signed.unsigned_abs().format_into(&mut num_buf).as_bytes());
        } else {
            self.cursor.write_bytes(num_text);
        }
        self.cursor.write_bytes(b")\n");
        self.write_prefixed_byte_groups("2진수: ", |byte| {
            let [[h0, h1, h2, h3], [l0, l1, l2, l3]] = [byte >> 4_u8, byte & 0x0f]
                .map(|nib| or_abort(BINARY_NIBBLES.get(usize::from(nib)).copied()));
            [h0, h1, h2, h3, l0, l1, l2, l3]
        });
        self.write_labeled_line("8진수: ".as_bytes(), |cur| {
            let mut oct = num64;
            for slot in cur
                .take(usize::from(low_u8_from_u32(
                    num64.bit_width().div_ceil(3).max(1),
                )))
                .iter_mut()
                .rev()
            {
                *slot = digit_byte(low_u8_from_u64(oct & 7));
                oct >>= 3_u32;
            }
        });
        self.write_prefixed_byte_groups("16진수: ", hex_byte);
        self.write_labeled_line("Hex 코드: ".as_bytes(), |cur| {
            let [b0, b1, b2, b3, b4, b5, _, _] = bytes;
            buf_write_prefixed_hex24(cur, b"#", b0, b1, b2);
            cur.write_byte(b' ');
            buf_write_prefixed_hex24(cur, b"#", b3, b4, b5);
        });
    }
    fn write_prefixed_byte_groups<const WIDTH: usize>(
        &mut self,
        prefix: &'static str,
        mut render_group: impl FnMut(u8) -> [u8; WIDTH],
    ) {
        let pb = prefix.as_bytes();
        let (prefix_out, groups) = self
            .cursor
            .take(pb.len().strict_add(WIDTH.strict_mul(8)).strict_add(8))
            .split_at_mut(pb.len());
        prefix_out.copy_from_slice(pb);
        for (group_out, byte) in groups.chunks_exact_mut(WIDTH.strict_add(1)).zip(self.bytes) {
            let (val_out, sep_out) = group_out.split_at_mut(WIDTH);
            val_out.copy_from_slice(&render_group(byte));
            if let Some(sep) = sep_out.first_mut() {
                *sep = b' ';
            }
        }
        *or_abort(groups.last_mut()) = b'\n';
    }
    fn write_random_lines(&mut self) {
        let (bytes, data) = (self.bytes, self.data);
        self.write_labeled_u8_array_line("바이트 배열: ".as_bytes(), &bytes);
        self.write_labeled_line("6자리 숫자 비밀번호: ".as_bytes(), |cur| {
            let rem = data.numeric_password.rem_euclid(10_000);
            let [h0, h1] = two_digits(low_u8_from_u32(data.numeric_password.div_euclid(10_000)));
            let [m0, m1] = two_digits(low_u8_from_u32(rem.div_euclid(100)));
            let [l0, l1] = two_digits(low_u8_from_u32(rem.rem_euclid(100)));
            *cur.take_array::<6>() = [h0, h1, m0, m1, l0, l1];
        });
        self.write_labeled_line("8자리 비밀번호: ".as_bytes(), |cur| {
            cur.write_bytes(&data.password);
        });
        self.write_labeled_u8_array_line("로또 번호: ".as_bytes(), &data.lotto_numbers);
        self.write_labeled_u8_array_line("일본 로또 7 번호: ".as_bytes(), &data.lotto7_numbers);
        self.write_labeled_line("유로밀리언 번호: ".as_bytes(), |cur| {
            buf_write_u8_array_spaced(cur, &data.euro_millions_main_numbers);
            cur.write_bytes(b" + ");
            buf_write_u8_array_spaced(cur, &data.euro_millions_lucky_stars);
        });
        self.write_labeled_line("한글 음절 4글자: ".as_bytes(), |cur| {
            let (chunks, _) = cur.take(12).as_chunks_mut::<3>();
            for (&ch, chunk) in data.hangul_syllables.iter().zip(chunks) {
                let [b0, b1] = ch.to_le_bytes();
                *chunk = [
                    0xE0_u8 | (b1 >> 4_u8),
                    0x80_u8 | ((b1 & 0x0F_u8) << 2_u8) | (b0 >> 6_u8),
                    0x80_u8 | (b0 & 0x3F_u8),
                ];
            }
        });
        self.cursor.write_format(format_args!(
            "대한민국 위경도: {}, {}\n세계 위경도: {}, {}\n",
            data.kor_coords.latitude,
            data.kor_coords.longitude,
            data.world_coords.latitude,
            data.world_coords.longitude,
        ));
    }
}
#[cfg(target_arch = "x86_64")]
impl ProgressBuffers {
    pub(super) const EMPTY: Self = Self([0_u8; 128]);
    pub(super) fn print(
        &mut self,
        out: &mut dyn IoWrite,
        completed: usize,
        total: usize,
        elapsed: Duration,
    ) -> Result<()> {
        let elapsed_ms = elapsed.as_millis();
        let eta_deci = if completed == total {
            Some(0)
        } else if completed == 0 {
            None
        } else {
            Some(
                elapsed_ms
                    .strict_mul(total.strict_sub(completed) as u128)
                    .div_euclid((completed as u128).strict_mul(100)),
            )
        };
        let percent_val = completed.strict_mul(100).div_euclid(total).min(100);
        let filled = percent_val.div_euclid(10);
        let [percent, ..] = percent_val.to_le_bytes();
        let mut cur = ByteCursor::new(&mut self.0);
        cur.write_bytes(b"\r[");
        for _ in 0..filled {
            cur.write_bytes("█".as_bytes());
        }
        for _ in filled..10 {
            cur.write_byte(b' ');
        }
        cur.write_bytes(b"] ");
        if percent < 100 {
            cur.write_byte(b' ');
        }
        if percent < 10 {
            cur.write_byte(b' ');
        }
        buf_write_u8_dec(&mut cur, percent);
        cur.write_bytes(b"% (");
        let mut count_buf = NumBuffer::new();
        cur.write_bytes(completed.format_into(&mut count_buf).as_bytes());
        cur.write_byte(b'/');
        cur.write_bytes(total.format_into(&mut count_buf).as_bytes());
        cur.write_bytes(") | 소요: ".as_bytes());
        write_progress_time(&mut cur, Some(elapsed_ms.div_euclid(100)));
        cur.write_bytes(b" | ETA: ");
        write_progress_time(&mut cur, eta_deci);
        cur.write_bytes(b" \x1b[K");
        let written_len = cur.written_len();
        IoWrite::write_all(out, self.0.split_at(written_len).0)?;
        IoWrite::flush(out).map_err(Into::into)
    }
}
fn buf_write_hex_u16_3(cur: &mut ByteCursor<'_>, value: u16) {
    let [_, h1, h2, h3] = hex_u16(value);
    cur.write_bytes(&[h1, h2, h3]);
}
fn buf_write_prefixed_hex24(cur: &mut ByteCursor<'_>, prefix: &[u8], b0: u8, b1: u8, b2: u8) {
    let (prefix_out, hex_bytes) = cur
        .take(prefix.len().strict_add(6))
        .split_at_mut(prefix.len());
    prefix_out.copy_from_slice(prefix);
    hex_bytes.copy_from_slice([b0, b1, b2].map(hex_byte).as_flattened());
}
fn buf_write_u8_array_spaced<const N: usize>(cur: &mut ByteCursor<'_>, nums: &[u8; N]) {
    let Some((&first, rest)) = nums.split_first() else {
        return;
    };
    buf_write_u8_dec(cur, first);
    for &num in rest {
        cur.write_byte(b' ');
        buf_write_u8_dec(cur, num);
    }
}
fn buf_write_u8_dec(cur: &mut ByteCursor<'_>, num: u8) {
    if num >= 100 {
        let [tens, ones] = two_digits(num.rem_euclid(100));
        *cur.take_array::<3>() = [digit_byte(num.div_euclid(100)), tens, ones];
    } else if num >= 10 {
        *cur.take_array::<2>() = two_digits(num);
    } else {
        cur.write_byte(digit_byte(num));
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
        let mut colored = [0_u8; 64];
        let mut cur = ByteCursor::new(colored.as_mut_slice());
        for (prefix, [red, green, blue]) in [
            (b"\x1B[38;2;".as_slice(), [b0, b1, b2]),
            (b" \x1B[38;2;".as_slice(), [b3, b4, b5]),
        ] {
            cur.write_bytes(prefix);
            buf_write_u8_dec(&mut cur, red);
            cur.write_byte(b';');
            buf_write_u8_dec(&mut cur, green);
            cur.write_byte(b';');
            buf_write_u8_dec(&mut cur, blue);
            buf_write_prefixed_hex24(&mut cur, b"m#", red, green, blue);
            cur.write_bytes(b"\x1B[0m");
        }
        let colored_len = cur.written_len();
        let val_start = buffer
            .get(175..file_len)
            .and_then(|sl| sl.iter().position(|&byte| byte == b'#'))
            .map_or_else(|| process::abort(), |off| 175_usize.strict_add(off));
        let (val_end, extra) = (val_start.strict_add(15), colored_len.strict_sub(15));
        buffer.copy_within(val_end..file_len, val_end.strict_add(extra));
        or_abort(buffer.get_mut(val_start..val_start.strict_add(colored_len)))
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
    [byte >> 4_u8, byte & 0x0f].map(|nib| or_abort(HEX_DIGITS.get(usize::from(nib)).copied()))
}
fn hex_u16(value: u16) -> [u8; 4] {
    let [[h0, h1], [h2, h3]] = value.to_be_bytes().map(hex_byte);
    [h0, h1, h2, h3]
}
#[cfg(target_arch = "x86_64")]
fn write_progress_time(cur: &mut ByteCursor<'_>, deci_seconds: Option<u128>) {
    let Some(deci) = deci_seconds else {
        *cur.take_array::<7>() = *b"--:--.-";
        return;
    };
    let total_secs = deci.div_euclid(10);
    let [m0, m1] = two_digits(low_u8_from_u128(total_secs.div_euclid(60).min(99)));
    let [s0, s1] = two_digits(low_u8_from_u128(total_secs.rem_euclid(60)));
    *cur.take_array::<7>() = [
        m0,
        m1,
        b':',
        s0,
        s1,
        b'.',
        digit_byte(low_u8_from_u128(deci.rem_euclid(10))),
    ];
}
pub(super) fn write_slice_to_console(data_slice: &[u8]) -> IoResult<()> {
    let mut stdout_lock = stdout().lock();
    IoWrite::write_all(&mut stdout_lock, data_slice)?;
    IoWrite::flush(&mut stdout_lock)
}
