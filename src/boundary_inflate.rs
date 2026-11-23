use crate::{
    diagnostic::{AppError, Result},
    numeric::{low_u16_from_u64, low_u32_from_u64},
};
use std::sync::LazyLock;
const CODE_ORDER: [u8; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];
static FIXED_TREES: LazyLock<Result<HuffmanTrees>> = LazyLock::new(|| {
    let mut lengths = [8_u8; 288];
    lengths
        .get_mut(144..256)
        .ok_or("경계 고정 코드 오류")?
        .fill(9);
    lengths
        .get_mut(256..280)
        .ok_or("경계 고정 코드 오류")?
        .fill(7);
    Ok(HuffmanTrees {
        distance: Huffman::new(&[5; 32], false)?,
        literal: Huffman::new(&lengths, false)?,
    })
});
struct Bits<'data> {
    bit_buffer: u64,
    bit_count: u32,
    bytes: &'data [u8],
    cursor: usize,
}
struct Huffman {
    counts: [u16; 16],
    quick: [u16; 512],
    symbols: [u16; 288],
}
struct HuffmanTrees {
    distance: Huffman,
    literal: Huffman,
}
pub(super) struct InflatedPage<'output>(&'output [u8]);
impl Bits<'_> {
    const fn align_to_byte(&mut self) {
        let drop_bits = self.bit_count & 7_u32;
        self.bit_buffer >>= drop_bits;
        self.bit_count = self.bit_count.strict_sub(drop_bits);
    }
    #[inline]
    const fn consume(&mut self, count: u32) {
        self.bit_buffer >>= count;
        self.bit_count = self.bit_count.wrapping_sub(count);
    }
    #[inline]
    fn read(&mut self, count: u32) -> Result<u32> {
        if count == 0 {
            return Ok(0);
        }
        self.refill(count);
        (self.bit_count >= count).ok_or("경계 압축 데이터가 잘렸습니다.")?;
        let val = low_u32_from_u64(self.bit_buffer & (1_u64 << count).wrapping_sub(1));
        self.consume(count);
        Ok(val)
    }
    fn read_usize(&mut self, count: u32) -> Result<usize> {
        Ok(self.read(count)? as usize)
    }
    #[inline]
    fn refill(&mut self, needed: u32) {
        if self.bit_count >= needed {
            return;
        }
        if let Some(tail) = self.bytes.get(self.cursor..) {
            if self.bit_count <= 16
                && let Some(&chunk) = tail.first_chunk::<8>()
            {
                self.bit_buffer |=
                    (u64::from_le_bytes(chunk) & 0x0000_ffff_ffff_ffff) << self.bit_count;
                self.cursor = self.cursor.wrapping_add(6);
                self.bit_count = self.bit_count.wrapping_add(48);
                return;
            }
            if let Some(&chunk) = tail.first_chunk::<4>() {
                self.bit_buffer |= u64::from(u32::from_le_bytes(chunk)) << self.bit_count;
                self.cursor = self.cursor.wrapping_add(4);
                self.bit_count = self.bit_count.wrapping_add(32);
                return;
            }
        }
        while self.bit_count < needed
            && let Some(&byte) = self.bytes.get(self.cursor)
        {
            self.bit_buffer |= u64::from(byte) << self.bit_count;
            self.cursor = self.cursor.strict_add(1);
            self.bit_count = self.bit_count.strict_add(8);
        }
    }
}
impl Huffman {
    fn new(lengths: &[u8], require_complete: bool) -> Result<Self> {
        let mut res = Self {
            counts: [0; 16],
            quick: [0; 512],
            symbols: [0; 288],
        };
        (lengths.len() <= res.symbols.len()).ok_or("경계 압축 코드 개수 오류")?;
        for &len in lengths {
            let cnt = res
                .counts
                .get_mut(usize::from(len))
                .ok_or("경계 압축 코드 길이 오류")?;
            *cnt = cnt.strict_add(1);
        }
        let (mut rem, mut offsets, mut next_off) = (1_u32, [0_u16; 16], 0_u16);
        for (&len_cnt, slot) in res.counts.iter().skip(1).zip(offsets.iter_mut().skip(1)) {
            rem = rem
                .strict_mul(2)
                .checked_sub(u32::from(len_cnt))
                .ok_or("경계 압축 코드가 중복됩니다.")?;
            *slot = next_off;
            next_off = next_off.strict_add(len_cnt);
        }
        (rem == 0 || (!require_complete && lengths.iter().all(|&len| len <= 1)))
            .ok_or("경계 압축 코드가 불완전합니다.")?;
        for (sym, &len) in lengths.iter().enumerate().filter(|&(_, &len)| len != 0) {
            let off = offsets
                .get_mut(usize::from(len))
                .ok_or("경계 압축 코드 오류")?;
            let [s0, s1, ..] = sym.to_le_bytes();
            *res.symbols
                .get_mut(usize::from(*off))
                .ok_or("경계 압축 코드 위치 오류")? = u16::from_le_bytes([s0, s1]);
            *off = off.strict_add(1);
        }
        let (mut first, mut index) = (0_u16, 0_usize);
        for (len, &cnt) in (1_u16..=9).zip(res.counts.iter().skip(1)) {
            for off in 0..cnt {
                let sym = *res
                    .symbols
                    .get(index.strict_add(usize::from(off)))
                    .ok_or("경계 압축 심볼 위치 오류")?;
                let packed = (len << 9_u16) | sym;
                let rev =
                    first.strict_add(off).reverse_bits() >> u16::BITS.strict_sub(u32::from(len));
                if let Some(tail) = res.quick.get_mut(usize::from(rev)..) {
                    for slot in tail.iter_mut().step_by(1_usize << len) {
                        *slot = packed;
                    }
                }
            }
            index = index.strict_add(usize::from(cnt));
            first = first.strict_add(cnt).strict_mul(2);
        }
        Ok(res)
    }
    #[cold]
    #[inline(never)]
    fn slow_symbol(&self, bits: &mut Bits<'_>) -> Result<u16> {
        bits.refill(15);
        let (mut buf, avail) = (bits.bit_buffer, bits.bit_count);
        let (mut code, mut first, mut index) = (0_u32, 0_u32, 0_u32);
        for (len, &entry_cnt) in (1_u32..).zip(self.counts.iter().skip(1)) {
            (avail >= len).ok_or("경계 압축 데이터가 잘렸습니다.")?;
            code |= low_u32_from_u64(buf & 1_u64);
            buf >>= 1_u32;
            let count = u32::from(entry_cnt);
            if code < first.strict_add(count) {
                bits.consume(len);
                return Ok(*self
                    .symbols
                    .get(index.strict_add(code.strict_sub(first)) as usize)
                    .ok_or("경계 압축 코드 위치 오류")?);
            }
            index = index.strict_add(count);
            first = first.strict_add(count).strict_mul(2);
            code <<= 1_u32;
        }
        Err("경계 압축 허프먼 코드 오류".into())
    }
    #[inline]
    fn symbol(&self, bits: &mut Bits<'_>) -> Result<u16> {
        bits.refill(9);
        let packed = *self
            .quick
            .get(usize::from(low_u16_from_u64(bits.bit_buffer) & 0x01FF))
            .ok_or("경계 압축 조회 위치 오류")?;
        let length = u32::from(packed >> 9_u32);
        if length != 0 && length <= bits.bit_count {
            bits.consume(length);
            return Ok(packed & 0x01FF);
        }
        self.slow_symbol(bits)
    }
}
impl TryFrom<&mut Bits<'_>> for HuffmanTrees {
    type Error = AppError;
    fn try_from(bits: &mut Bits<'_>) -> Result<Self> {
        let (lit_cnt, dist_cnt, code_cnt) = (
            bits.read_usize(5)?.strict_add(257),
            bits.read_usize(5)?.strict_add(1),
            bits.read_usize(4)?.strict_add(4),
        );
        (lit_cnt <= 286).ok_or("경계 압축 코드 개수 오류")?;
        let mut code_lens = [0_u8; 19];
        for &pos in CODE_ORDER.iter().take(code_cnt) {
            *code_lens
                .get_mut(usize::from(pos))
                .ok_or("경계 압축 코드 순서 오류")? = bits.read(3)?.to_le_bytes()[0];
        }
        let codes = Huffman::new(&code_lens, true)?;
        let total = lit_cnt.strict_add(dist_cnt);
        let (mut lengths, mut filled) = ([0_u8; 318], 0_usize);
        while filled < total {
            let sym = codes.symbol(bits)?;
            let (val, cnt) = match sym {
                0..=15 => (sym.to_le_bytes()[0], 1_usize),
                16 => (
                    *lengths
                        .get(filled.checked_sub(1).ok_or("경계 압축 코드 반복 오류")?)
                        .ok_or("경계 압축 코드 반복 오류")?,
                    bits.read_usize(2)?.strict_add(3),
                ),
                17 => (0, bits.read_usize(3)?.strict_add(3)),
                18 => (0, bits.read_usize(7)?.strict_add(11)),
                _ => return Err("경계 압축 코드 길이 오류".into()),
            };
            let next = filled.strict_add(cnt);
            (next <= total).ok_or("경계 압축 코드 반복 한도 오류")?;
            lengths
                .get_mut(filled..next)
                .ok_or("경계 압축 코드 반복 오류")?
                .fill(val);
            filled = next;
        }
        lengths
            .get(256)
            .is_some_and(|len| *len != 0)
            .ok_or("경계 압축 종료 코드 오류")?;
        let (lit_lens, dist_lens) = lengths
            .get(..total)
            .ok_or("경계 압축 코드 개수 오류")?
            .split_at(lit_cnt);
        Ok(Self {
            distance: Huffman::new(dist_lens, false)?,
            literal: Huffman::new(lit_lens, false)?,
        })
    }
}
impl AsRef<[u8]> for InflatedPage<'_> {
    fn as_ref(&self) -> &[u8] {
        self.0
    }
}
impl<'output> TryFrom<(&[u8], &'output mut [u8])> for InflatedPage<'output> {
    type Error = AppError;
    fn try_from((input, output): (&[u8], &'output mut [u8])) -> Result<Self> {
        let header = input.first_chunk::<2>().ok_or("경계 압축 헤더 오류")?;
        (header[0] & 15 == 8
            && header[0] >> 4_u32 <= 7
            && header[1] & 32 == 0
            && u16::from_be_bytes(*header).is_multiple_of(31))
        .ok_or("경계 압축 형식 오류")?;
        let tail = input.len().checked_sub(4).ok_or("경계 압축 길이 오류")?;
        let mut bits = Bits {
            bit_buffer: 0,
            bit_count: 0,
            bytes: input.get(2..tail).ok_or("경계 압축 길이 오류")?,
            cursor: 0,
        };
        let mut written = 0_usize;
        loop {
            let final_block = bits.read(1)? != 0;
            match bits.read(2)? {
                0 => {
                    bits.align_to_byte();
                    let length = bits.read(16)?;
                    (length ^ bits.read(16)? == 0xFFFF).ok_or("경계 압축 저장 블록 오류")?;
                    for _ in 0..length {
                        *output.get_mut(written).ok_or("경계 압축 출력 한도 오류")? =
                            bits.read(8)?.to_le_bytes()[0];
                        written = written.strict_add(1);
                    }
                }
                kind @ (1 | 2) => {
                    let dynamic;
                    let trees = if kind == 1 {
                        FIXED_TREES
                            .as_ref()
                            .map_err(|src| AppError::prefixed_ref("", src))?
                    } else {
                        dynamic = HuffmanTrees::try_from(&mut bits)?;
                        &dynamic
                    };
                    loop {
                        let sym = trees.literal.symbol(&mut bits)?;
                        if sym == 256 {
                            break;
                        }
                        if sym < 256 {
                            *output.get_mut(written).ok_or("경계 압축 출력 한도 오류")? =
                                sym.to_le_bytes()[0];
                            written = written.wrapping_add(1);
                            continue;
                        }
                        let length = decode_extra(
                            false,
                            u32::from(sym.strict_sub(257)),
                            &mut bits,
                            "경계 압축 길이 코드 오류",
                        )?;
                        let backward = decode_extra(
                            true,
                            u32::from(trees.distance.symbol(&mut bits)?),
                            &mut bits,
                            "경계 압축 거리 코드 오류",
                        )?;
                        let src = written
                            .checked_sub(backward)
                            .ok_or("경계 압축 참조 위치 오류")?;
                        let end = written.strict_add(length);
                        (end <= output.len()).ok_or("경계 압축 출력 한도 오류")?;
                        if backward == 1 {
                            let fill = *output.get(src).ok_or("경계 압축 참조 위치 오류")?;
                            output
                                .get_mut(written..end)
                                .ok_or("경계 압축 출력 한도 오류")?
                                .fill(fill);
                            written = end;
                        } else if backward >= length {
                            output.copy_within(src..src.wrapping_add(length), written);
                            written = end;
                        } else {
                            let mut copied = 0_usize;
                            while copied < length {
                                let step = backward.max(copied).min(length.strict_sub(copied));
                                output.copy_within(src..src.strict_add(step), written);
                                written = written.strict_add(step);
                                copied = copied.strict_add(step);
                            }
                        }
                    }
                }
                _ => return Err("경계 압축 블록 형식 오류".into()),
            }
            if final_block {
                break;
            }
        }
        (bits.cursor == bits.bytes.len() && bits.bit_count < 8)
            .ok_or("경계 압축 잔여 데이터 오류")?;
        let (mut first, mut second) = (1_u32, 0_u32);
        let out_slice = output.get(..written).ok_or("경계 압축 출력 오류")?;
        for chunk in out_slice.chunks(5552) {
            let (blocks, rem) = chunk.as_chunks::<16>();
            for block in blocks {
                second = second.wrapping_add(first.wrapping_mul(16));
                let (mut sum, mut weighted) = (0_u32, 0_u32);
                for (weight, &byte) in (1_u32..=16_u32).rev().zip(block) {
                    let val = u32::from(byte);
                    sum = sum.wrapping_add(val);
                    weighted = weighted.wrapping_add(val.wrapping_mul(weight));
                }
                first = first.wrapping_add(sum);
                second = second.wrapping_add(weighted);
            }
            for &byte in rem {
                first = first.wrapping_add(u32::from(byte));
                second = second.wrapping_add(first);
            }
            first = first.rem_euclid(0xFFF1);
            second = second.rem_euclid(0xFFF1);
        }
        let expected = input
            .get(tail..)
            .and_then(<[u8]>::first_chunk::<4>)
            .ok_or("경계 압축 체크섬 길이 오류")?;
        (((second << 16_u32) | first) == u32::from_be_bytes(*expected))
            .ok_or("경계 압축 체크섬 오류")?;
        Ok(Self(out_slice))
    }
}
#[inline]
fn decode_extra(
    is_distance: bool,
    index: u32,
    bits: &mut Bits<'_>,
    code_err: &'static str,
) -> Result<usize> {
    let (base, extra) = if is_distance {
        (index < 30).ok_or(code_err)?;
        if index < 4 {
            (index.strict_add(1), 0)
        } else {
            let ex = index.strict_sub(2) >> 1_u32;
            (((2_u32.strict_add(index & 1_u32)) << ex).strict_add(1), ex)
        }
    } else {
        (index < 29).ok_or(code_err)?;
        if index < 8 {
            (index.strict_add(3), 0)
        } else if index == 28 {
            (258, 0)
        } else {
            let ex = index.strict_sub(4) >> 2_u32;
            (((4_u32.strict_add(index & 3_u32)) << ex).strict_add(3), ex)
        }
    };
    Ok((base as usize).strict_add(if extra == 0 {
        0
    } else {
        bits.read_usize(extra)?
    }))
}
