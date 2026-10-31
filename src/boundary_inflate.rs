use crate::diagnostic::{AppError, Result};
use std::sync::LazyLock;
const CODE_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];
const DISTANCE_BASE: [usize; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 0x4001, 24577,
];
const DISTANCE_EXTRA: [u32; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const LENGTH_BASE: [usize; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
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
    bytes: &'data [u8],
    position: usize,
}
struct Huffman {
    counts: [u32; 16],
    quick: [u16; 512],
    symbols: [u16; 288],
}
struct HuffmanTrees {
    distance: Huffman,
    literal: Huffman,
}
pub(super) struct InflatedPage(Vec<u8>);
impl Bits<'_> {
    fn peek_nine(&self) -> Option<u16> {
        let bytes = self
            .bytes
            .get(self.position.strict_div(8)..)?
            .first_chunk::<2>()?;
        Some((u16::from_le_bytes(*bytes) >> (self.position & 7)) & 0x01FF)
    }
    fn read(&mut self, count: u32) -> Result<u32> {
        let bit_count = usize::try_from(count)
            .map_err(|source| format!("경계 압축 비트 개수 오류: {source}"))?;
        let mut value = 0_u32;
        let mut shift = 0_usize;
        while shift < bit_count {
            let offset = self.position & 7_usize;
            let width = bit_count.strict_sub(shift).min(8_usize.strict_sub(offset));
            let byte = *self
                .bytes
                .get(self.position.strict_div(8))
                .ok_or("경계 압축 데이터가 잘렸습니다.")?;
            let mask = (1_u32 << width).strict_sub(1);
            value |= ((u32::from(byte) >> offset) & mask) << shift;
            self.position = self.position.strict_add(width);
            shift = shift.strict_add(width);
        }
        Ok(value)
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
        let mut offsets = [0_u32; 16];
        for length in 1_usize..16 {
            remaining = remaining
                .strict_mul(2)
                .checked_sub(*result.counts.get(length).ok_or("경계 압축 코드 오류")?)
                .ok_or("경계 압축 코드가 중복됩니다.")?;
            if length > 1 {
                *offsets.get_mut(length).ok_or("경계 압축 코드 오류")? = offsets
                    .get(length.strict_sub(1))
                    .ok_or("경계 압축 코드 오류")?
                    .strict_add(
                        *result
                            .counts
                            .get(length.strict_sub(1))
                            .ok_or("경계 압축 코드 오류")?,
                    );
            }
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
            let index = usize::try_from(*offset)
                .map_err(|source| format!("경계 압축 코드 위치 오류: {source}"))?;
            *result
                .symbols
                .get_mut(index)
                .ok_or("경계 압축 코드 위치 오류")? =
                u16::try_from(symbol).map_err(|source| format!("경계 압축 심볼 오류: {source}"))?;
            *offset = offset.strict_add(1);
        }
        let mut first = 0_u32;
        let mut index = 0_usize;
        for length in 1_u32..=9 {
            let count = *result
                .counts
                .get(usize::try_from(length).map_err(|source| format!("{source}"))?)
                .ok_or("경계 압축 코드 길이 오류")?;
            for offset in 0..count {
                let symbol_index = index
                    .strict_add(usize::try_from(offset).map_err(|source| format!("{source}"))?);
                let symbol = *result
                    .symbols
                    .get(symbol_index)
                    .ok_or("경계 압축 심볼 위치 오류")?;
                let packed = u16::try_from((length << 9_u32) | u32::from(symbol))
                    .map_err(|source| format!("{source}"))?;
                let reversed =
                    first.strict_add(offset).reverse_bits() >> u32::BITS.strict_sub(length);
                let begin = usize::try_from(reversed).map_err(|source| format!("{source}"))?;
                for slot in (begin..512).step_by(1_usize << length) {
                    *result
                        .quick
                        .get_mut(slot)
                        .ok_or("경계 압축 조회 위치 오류")? = packed;
                }
            }
            index = index.strict_add(usize::try_from(count).map_err(|source| format!("{source}"))?);
            first = first.strict_add(count).strict_mul(2);
        }
        Ok(result)
    }
    fn symbol(&self, bits: &mut Bits<'_>) -> Result<u16> {
        if let Some(prefix) = bits.peek_nine() {
            let packed = *self
                .quick
                .get(usize::from(prefix))
                .ok_or("경계 압축 조회 위치 오류")?;
            if packed != 0 {
                bits.position = bits.position.strict_add(usize::from(packed >> 9_u32));
                return Ok(packed & 0x01FF);
            }
        }
        let mut code = 0_u32;
        let mut first = 0_u32;
        let mut index = 0_u32;
        for count in self.counts.iter().skip(1) {
            code |= bits.read(1)?;
            if code < first.strict_add(*count) {
                let position = usize::try_from(index.strict_add(code.strict_sub(first)))
                    .map_err(|source| format!("경계 압축 코드 위치 오류: {source}"))?;
                return self
                    .symbols
                    .get(position)
                    .copied()
                    .ok_or_else(|| "경계 압축 코드 위치 오류".into());
            }
            index = index.strict_add(*count);
            first = first.strict_add(*count).strict_mul(2);
            code <<= 1_u32;
        }
        Err("경계 압축 허프먼 코드 오류".into())
    }
}
impl AsRef<[u8]> for InflatedPage {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}
impl TryFrom<&mut Bits<'_>> for HuffmanTrees {
    type Error = AppError;
    fn try_from(bits: &mut Bits<'_>) -> Result<Self> {
        let literal_count = usize::try_from(bits.read(5)?.strict_add(257))
            .map_err(|source| format!("경계 압축 코드 개수 오류: {source}"))?;
        let distance_count = usize::try_from(bits.read(5)?.strict_add(1))
            .map_err(|source| format!("경계 압축 코드 개수 오류: {source}"))?;
        let code_count = usize::try_from(bits.read(4)?.strict_add(4))
            .map_err(|source| format!("경계 압축 코드 개수 오류: {source}"))?;
        (literal_count <= 286).ok_or("경계 압축 코드 개수 오류")?;
        let mut code_lengths = [0_u8; 19];
        for &position in CODE_ORDER.iter().take(code_count) {
            *code_lengths
                .get_mut(position)
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
                    usize::try_from(bits.read(2)?.strict_add(3))
                        .map_err(|source| format!("경계 압축 반복 오류: {source}"))?,
                ),
                17 => (
                    0,
                    usize::try_from(bits.read(3)?.strict_add(3))
                        .map_err(|source| format!("경계 압축 반복 오류: {source}"))?,
                ),
                18 => (
                    0,
                    usize::try_from(bits.read(7)?.strict_add(11))
                        .map_err(|source| format!("경계 압축 반복 오류: {source}"))?,
                ),
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
impl TryFrom<(&[u8], usize)> for InflatedPage {
    type Error = AppError;
    fn try_from((input, capacity): (&[u8], usize)) -> Result<Self> {
        let mut output = vec![0_u8; capacity];
        let header = input.first_chunk::<2>().ok_or("경계 압축 헤더 오류")?;
        (header[0] & 15 == 8
            && header[0] >> 4_u32 <= 7
            && header[1] & 32 == 0
            && u16::from_be_bytes(*header).is_multiple_of(31))
        .ok_or("경계 압축 형식 오류")?;
        let tail = input.len().checked_sub(4).ok_or("경계 압축 길이 오류")?;
        let mut bits = Bits {
            bytes: input.get(2..tail).ok_or("경계 압축 길이 오류")?,
            position: 0,
        };
        let mut written = 0_usize;
        loop {
            let final_block = bits.read(1)? != 0;
            match bits.read(2)? {
                0 => {
                    bits.position = bits.position.next_multiple_of(8);
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
                        FIXED_TREES.as_ref().map_err(|source| format!("{source}"))?
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
                            written = written.strict_add(1);
                            continue;
                        }
                        let index =
                            usize::from(symbol.checked_sub(257).ok_or("경계 압축 길이 코드 오류")?);
                        let length = LENGTH_BASE
                            .get(index)
                            .ok_or("경계 압축 길이 코드 오류")?
                            .strict_add(
                                usize::try_from(bits.read(
                                    *LENGTH_EXTRA.get(index).ok_or("경계 압축 길이 코드 오류")?,
                                )?)
                                .map_err(|source| format!("경계 압축 길이 오류: {source}"))?,
                            );
                        let distance_index = usize::from(trees.distance.symbol(&mut bits)?);
                        let backward = DISTANCE_BASE
                            .get(distance_index)
                            .ok_or("경계 압축 거리 코드 오류")?
                            .strict_add(
                                usize::try_from(
                                    bits.read(
                                        *DISTANCE_EXTRA
                                            .get(distance_index)
                                            .ok_or("경계 압축 거리 코드 오류")?,
                                    )?,
                                )
                                .map_err(|source| format!("경계 압축 거리 오류: {source}"))?,
                            );
                        for _ in 0..length {
                            let source = written
                                .checked_sub(backward)
                                .and_then(|position| output.get(position))
                                .copied()
                                .ok_or("경계 압축 참조 위치 오류")?;
                            *output.get_mut(written).ok_or("경계 압축 출력 한도 오류")? = source;
                            written = written.strict_add(1);
                        }
                    }
                }
                _ => return Err("경계 압축 블록 형식 오류".into()),
            }
            if final_block {
                break;
            }
        }
        (bits.position.div_ceil(8) == bits.bytes.len()).ok_or("경계 압축 잔여 데이터 오류")?;
        let mut first = 1_u32;
        let mut second = 0_u32;
        for chunk in output
            .get(..written)
            .ok_or("경계 압축 출력 오류")?
            .chunks(1024)
        {
            for byte in chunk {
                first = first.strict_add(u32::from(*byte));
                second = second.strict_add(first);
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
        output.truncate(written);
        Ok(Self(output))
    }
}
