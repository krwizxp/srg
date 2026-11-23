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
    let literal = Huffman::new(&lengths, false)?;
    let distance = Huffman::new(&[5; 32], false)?;
    Ok(HuffmanTrees { distance, literal })
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
    const fn consume(&mut self, count: u32) {
        self.bit_buffer >>= count;
        self.bit_count = self.bit_count.wrapping_sub(count);
    }
    const fn is_exhausted(&self) -> bool {
        self.cursor == self.bytes.len() && self.bit_count < 8
    }
    fn read(&mut self, count: u32) -> Result<u32> {
        if count == 0 {
            return Ok(0);
        }
        self.refill(count);
        (self.bit_count >= count).ok_or("경계 압축 데이터가 잘렸습니다.")?;
        let mask = (1_u64 << count).wrapping_sub(1);
        let value = low_u32_from_u64(self.bit_buffer & mask);
        self.consume(count);
        Ok(value)
    }
    fn read_usize(&mut self, count: u32, context: &'static str) -> Result<usize> {
        usize::try_from(self.read(count)?).map_err(|source| AppError::context(context, source))
    }
    fn refill(&mut self, needed: u32) {
        if self.bit_count >= needed {
            return;
        }
        if let Some(tail) = self.bytes.get(self.cursor..) {
            if self.bit_count <= 16
                && let Some(&chunk) = tail.first_chunk::<8>()
            {
                let refill = u64::from_le_bytes(chunk) & 0x0000_ffff_ffff_ffff;
                self.bit_buffer |= refill << self.bit_count;
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
        while self.bit_count < needed {
            let Some(&byte) = self.bytes.get(self.cursor) else {
                break;
            };
            self.bit_buffer |= u64::from(byte) << self.bit_count;
            self.cursor = self.cursor.strict_add(1);
            self.bit_count = self.bit_count.strict_add(8);
        }
    }
}
impl Huffman {
    fn new(lengths: &[u8], require_complete: bool) -> Result<Self> {
        let mut result = Self {
            counts: [0; 16],
            quick: [0; 512],
            symbols: [0; 288],
        };
        (lengths.len() <= result.symbols.len()).ok_or("경계 압축 코드 개수 오류")?;
        for &length in lengths {
            let count = result
                .counts
                .get_mut(usize::from(length))
                .ok_or("경계 압축 코드 길이 오류")?;
            *count = count.strict_add(1);
        }
        let mut remaining = 1_u32;
        let mut offsets = [0_u16; 16];
        let mut next_offset = 0_u16;
        for (&length_count, offset_slot) in
            result.counts.iter().skip(1).zip(offsets.iter_mut().skip(1))
        {
            remaining = remaining
                .strict_mul(2)
                .checked_sub(u32::from(length_count))
                .ok_or("경계 압축 코드가 중복됩니다.")?;
            *offset_slot = next_offset;
            next_offset = next_offset.strict_add(length_count);
        }
        (remaining == 0 || (!require_complete && lengths.iter().all(|&length| length <= 1)))
            .ok_or("경계 압축 코드가 불완전합니다.")?;
        for (symbol, &length) in lengths.iter().enumerate() {
            if length == 0 {
                continue;
            }
            let offset = offsets
                .get_mut(usize::from(length))
                .ok_or("경계 압축 코드 오류")?;
            *result
                .symbols
                .get_mut(usize::from(*offset))
                .ok_or("경계 압축 코드 위치 오류")? = u16::try_from(symbol)
                .map_err(|source| AppError::context("경계 압축 심볼 오류", source))?;
            *offset = offset.strict_add(1);
        }
        let mut first = 0_u16;
        let mut index = 0_usize;
        for (length, &count) in (1_u16..=9).zip(result.counts.iter().skip(1)) {
            for offset in 0..count {
                let symbol = *result
                    .symbols
                    .get(index.strict_add(usize::from(offset)))
                    .ok_or("경계 압축 심볼 위치 오류")?;
                let packed = (length << 9_u16) | symbol;
                let reversed = first.strict_add(offset).reverse_bits()
                    >> u16::BITS.strict_sub(u32::from(length));
                if let Some(tail) = result.quick.get_mut(usize::from(reversed)..) {
                    for slot in tail.iter_mut().step_by(1_usize << length) {
                        *slot = packed;
                    }
                }
            }
            index = index.strict_add(usize::from(count));
            first = first.strict_add(count).strict_mul(2);
        }
        Ok(result)
    }
    #[cold]
    #[inline(never)]
    fn slow_symbol(&self, bits: &mut Bits<'_>) -> Result<u16> {
        bits.refill(15);
        let mut buf = bits.bit_buffer;
        let available = bits.bit_count;
        let mut code = 0_u32;
        let mut first = 0_u32;
        let mut index = 0_u32;
        for (len, &entry_count) in (1_u32..).zip(self.counts.iter().skip(1)) {
            (available >= len).ok_or("경계 압축 데이터가 잘렸습니다.")?;
            code |= low_u32_from_u64(buf & 1_u64);
            buf >>= 1_u32;
            let count = u32::from(entry_count);
            if code < first.strict_add(count) {
                bits.consume(len);
                let position = usize::try_from(index.strict_add(code.strict_sub(first)))
                    .map_err(|source| AppError::context("경계 압축 코드 위치 오류", source))?;
                return self
                    .symbols
                    .get(position)
                    .copied()
                    .ok_or_else(|| "경계 압축 코드 위치 오류".into());
            }
            index = index.strict_add(count);
            first = first.strict_add(count).strict_mul(2);
            code <<= 1_u32;
        }
        Err("경계 압축 허프먼 코드 오류".into())
    }
    fn symbol(&self, bits: &mut Bits<'_>) -> Result<u16> {
        bits.refill(9);
        let prefix = low_u16_from_u64(bits.bit_buffer) & 0x01FF;
        let packed = *self
            .quick
            .get(usize::from(prefix))
            .ok_or("경계 압축 조회 위치 오류")?;
        let length = u32::from(packed >> 9_u32);
        if length != 0 && length <= bits.bit_count {
            bits.consume(length);
            return Ok(packed & 0x01FF);
        }
        self.slow_symbol(bits)
    }
}
impl AsRef<[u8]> for InflatedPage<'_> {
    fn as_ref(&self) -> &[u8] {
        self.0
    }
}
impl TryFrom<&mut Bits<'_>> for HuffmanTrees {
    type Error = AppError;
    fn try_from(bits: &mut Bits<'_>) -> Result<Self> {
        let literal_count = bits
            .read_usize(5, "경계 압축 코드 개수 오류")?
            .strict_add(257);
        let distance_count = bits
            .read_usize(5, "경계 압축 코드 개수 오류")?
            .strict_add(1);
        let code_count = bits
            .read_usize(4, "경계 압축 코드 개수 오류")?
            .strict_add(4);
        (literal_count <= 286).ok_or("경계 압축 코드 개수 오류")?;
        let mut code_lengths = [0_u8; 19];
        for &position in CODE_ORDER.iter().take(code_count) {
            *code_lengths
                .get_mut(usize::from(position))
                .ok_or("경계 압축 코드 순서 오류")? = bits.read(3)?.to_le_bytes()[0];
        }
        let codes = Huffman::new(&code_lengths, true)?;
        let total = literal_count.strict_add(distance_count);
        let mut lengths = [0_u8; 318];
        let mut filled = 0_usize;
        while filled < total {
            let symbol = codes.symbol(bits)?;
            let (value, count) = match symbol {
                0..=15 => (symbol.to_le_bytes()[0], 1_usize),
                16 => (
                    *lengths
                        .get(filled.checked_sub(1).ok_or("경계 압축 코드 반복 오류")?)
                        .ok_or("경계 압축 코드 반복 오류")?,
                    bits.read_usize(2, "경계 압축 반복 오류")?.strict_add(3),
                ),
                17 => (0, bits.read_usize(3, "경계 압축 반복 오류")?.strict_add(3)),
                18 => (0, bits.read_usize(7, "경계 압축 반복 오류")?.strict_add(11)),
                _ => return Err("경계 압축 코드 길이 오류".into()),
            };
            let next = filled.strict_add(count);
            (next <= total).ok_or("경계 압축 코드 반복 한도 오류")?;
            lengths
                .get_mut(filled..next)
                .ok_or("경계 압축 코드 반복 오류")?
                .fill(value);
            filled = next;
        }
        lengths
            .get(256)
            .is_some_and(|length| *length != 0)
            .ok_or("경계 압축 종료 코드 오류")?;
        let (literal_lengths, distance_lengths) = lengths
            .get(..total)
            .ok_or("경계 압축 코드 개수 오류")?
            .split_at(literal_count);
        let literal = Huffman::new(literal_lengths, false)?;
        let distance = Huffman::new(distance_lengths, false)?;
        Ok(Self { distance, literal })
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
                            .map_err(|source| AppError::prefixed_ref("", source))?
                    } else {
                        dynamic = HuffmanTrees::try_from(&mut bits)?;
                        &dynamic
                    };
                    loop {
                        let symbol = trees.literal.symbol(&mut bits)?;
                        if symbol == 256 {
                            break;
                        }
                        if symbol < 256 {
                            *output.get_mut(written).ok_or("경계 압축 출력 한도 오류")? =
                                symbol.to_le_bytes()[0];
                            written = written.wrapping_add(1);
                            continue;
                        }
                        let length = decode_extra(
                            false,
                            u32::from(symbol.strict_sub(257)),
                            &mut bits,
                            "경계 압축 길이 코드 오류",
                            "경계 압축 길이 오류",
                        )?;
                        let distance_index = u32::from(trees.distance.symbol(&mut bits)?);
                        let backward = decode_extra(
                            true,
                            distance_index,
                            &mut bits,
                            "경계 압축 거리 코드 오류",
                            "경계 압축 거리 오류",
                        )?;
                        let source_start = written
                            .checked_sub(backward)
                            .ok_or("경계 압축 참조 위치 오류")?;
                        let end = written.strict_add(length);
                        (end <= output.len()).ok_or("경계 압축 출력 한도 오류")?;
                        if backward == 1 {
                            let repeated =
                                *output.get(source_start).ok_or("경계 압축 참조 위치 오류")?;
                            output
                                .get_mut(written..end)
                                .ok_or("경계 압축 출력 한도 오류")?
                                .fill(repeated);
                            written = end;
                        } else if backward >= length {
                            output.copy_within(
                                source_start..source_start.wrapping_add(length),
                                written,
                            );
                            written = written.wrapping_add(length);
                        } else {
                            let mut copied = 0_usize;
                            while copied < length {
                                let step = backward.max(copied).min(length.strict_sub(copied));
                                output.copy_within(
                                    source_start..source_start.strict_add(step),
                                    written,
                                );
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
        bits.is_exhausted().ok_or("경계 압축 잔여 데이터 오류")?;
        let mut first = 1_u32;
        let mut second = 0_u32;
        for chunk in output
            .get(..written)
            .ok_or("경계 압축 출력 오류")?
            .chunks(5552)
        {
            let (blocks, remainder) = chunk.as_chunks::<16>();
            for block in blocks {
                second = second.wrapping_add(first.wrapping_mul(16));
                let mut sum = 0_u32;
                let mut weighted = 0_u32;
                for (weight, &byte) in (1_u32..=16_u32).rev().zip(block) {
                    let value = u32::from(byte);
                    sum = sum.wrapping_add(value);
                    weighted = weighted.wrapping_add(value.wrapping_mul(weight));
                }
                first = first.wrapping_add(sum);
                second = second.wrapping_add(weighted);
            }
            for &byte in remainder {
                first = first.wrapping_add(u32::from(byte));
                second = second.wrapping_add(first);
            }
            first = first.rem_euclid(0xFFF1);
            second = second.rem_euclid(0xFFF1);
        }
        let expected = input
            .get(tail..)
            .and_then(|bytes| bytes.first_chunk::<4>())
            .ok_or("경계 압축 체크섬 길이 오류")?;
        (((second << 16_u32) | first) == u32::from_be_bytes(*expected))
            .ok_or("경계 압축 체크섬 오류")?;
        Ok(Self(output.get(..written).ok_or("경계 압축 출력 오류")?))
    }
}
fn decode_extra(
    is_distance: bool,
    index: u32,
    bits: &mut Bits<'_>,
    code_err: &'static str,
    range_err: &'static str,
) -> Result<usize> {
    let (base, extra_bits) = if is_distance {
        (index < 30).ok_or(code_err)?;
        if index < 4 {
            (index.strict_add(1), 0)
        } else {
            let extra = index.strict_sub(2) >> 1_u32;
            (
                ((2_u32.strict_add(index & 1_u32)) << extra).strict_add(1),
                extra,
            )
        }
    } else {
        (index < 29).ok_or(code_err)?;
        if index < 8 {
            (index.strict_add(3), 0)
        } else if index == 28 {
            (258, 0)
        } else {
            let extra = index.strict_sub(4) >> 2_u32;
            (
                ((4_u32.strict_add(index & 3_u32)) << extra).strict_add(3),
                extra,
            )
        }
    };
    let base_usize =
        usize::try_from(base).map_err(|source| AppError::context(range_err, source))?;
    if extra_bits == 0 {
        return Ok(base_usize);
    }
    Ok(base_usize.strict_add(bits.read_usize(extra_bits, range_err)?))
}
