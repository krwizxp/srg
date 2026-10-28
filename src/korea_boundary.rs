use self::data::{INDEX as DATA, PARTS};
use crate::{boundary_inflate::inflate, diagnostic::Result, random_data::Coordinates};
use alloc::collections::VecDeque;
mod data;
#[cfg(test)]
mod tests;
#[derive(Default)]
pub(super) struct BoundaryCache {
    pages: VecDeque<(usize, Box<[u8; MAX_PAGE_SIZE]>)>,
}
const INDEX_SIZE: usize = 24;
const MAX_PAGE_SIZE: usize = 15_876;
const RECORD_SIZE: usize = 14;
fn decode_page(compressed: &[u8], mut previous: Option<u64>) -> Result<[u8; MAX_PAGE_SIZE]> {
    let mut shuffled = [0_u8; MAX_PAGE_SIZE];
    let length = inflate(compressed, &mut shuffled)?;
    let vertices = usize::from(read_u16(&shuffled, 0)?);
    let triangles = usize::from(read_u16(&shuffled, 2)?);
    let records = 4_usize.strict_add(vertices.strict_mul(16));
    (vertices > 0
        && vertices <= 768
        && triangles > 0
        && triangles <= 256
        && length == records.strict_add(triangles.strict_mul(RECORD_SIZE)))
    .ok_or("대한민국 경계 데이터 블록 오류")?;
    let mut page = [0_u8; MAX_PAGE_SIZE];
    page.get_mut(..4)
        .ok_or("경계 블록 오류")?
        .copy_from_slice(shuffled.get(..4).ok_or("경계 블록 오류")?);
    for (word, destination) in page
        .get_mut(4..records)
        .ok_or("경계 정점 오류")?
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .enumerate()
    {
        for (lane, byte) in destination.iter_mut().enumerate() {
            let offset = 4_usize
                .strict_add(lane.strict_mul(vertices.strict_mul(2)))
                .strict_add(word);
            *byte = *shuffled.get(offset).ok_or("경계 정점 오류")?;
        }
    }
    for (row, destination) in page
        .get_mut(records..length)
        .ok_or("경계 삼각형 오류")?
        .as_chunks_mut::<RECORD_SIZE>()
        .0
        .iter_mut()
        .enumerate()
    {
        for (lane, byte) in destination.iter_mut().enumerate() {
            let offset = records
                .strict_add(lane.strict_mul(triangles))
                .strict_add(row);
            *byte = *shuffled.get(offset).ok_or("경계 삼각형 오류")?;
        }
        let width = read_u64(destination, 0)?;
        let end = previous
            .map_or_else(|| width.checked_sub(1), |prior| prior.checked_add(width))
            .ok_or("대한민국 경계 데이터 가중치 오류")?;
        destination
            .get_mut(..8)
            .ok_or("경계 가중치 오류")?
            .copy_from_slice(&end.to_le_bytes());
        previous = Some(end);
    }
    Ok(page)
}
fn float_from_u64(value: u64) -> f64 {
    let [byte0, byte1, byte2, byte3, byte4, byte5, byte6, byte7] = value.to_be_bytes();
    f64::from(u32::from_be_bytes([byte0, byte1, byte2, byte3])).mul_add(
        4_294_967_296.0,
        f64::from(u32::from_be_bytes([byte4, byte5, byte6, byte7])),
    )
}
fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let value = bytes
        .get(offset..)
        .and_then(|tail| tail.first_chunk::<2>())
        .ok_or("대한민국 경계 데이터 구조 오류")?;
    Ok(u16::from_le_bytes(*value))
}
fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let value = bytes
        .get(offset..)
        .and_then(|tail| tail.first_chunk::<4>())
        .ok_or("대한민국 경계 데이터 구조 오류")?;
    Ok(u32::from_le_bytes(*value))
}
fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let value = bytes
        .get(offset..)
        .and_then(|tail| tail.first_chunk::<8>())
        .ok_or("대한민국 경계 데이터 구조 오류")?;
    Ok(u64::from_le_bytes(*value))
}
pub(super) fn sample(value: u64, cache: Option<&mut BoundaryCache>) -> Result<Coordinates> {
    DATA.starts_with(b"SRGKR001")
        .ok_or("대한민국 경계 데이터 버전 오류")?;
    let count = usize::try_from(read_u32(DATA, 8)?)
        .map_err(|source| format!("대한민국 경계 데이터 인덱스 크기 오류: {source}"))?;
    (count > 0 && count <= DATA.len().saturating_sub(32).strict_div(INDEX_SIZE))
        .ok_or("대한민국 경계 데이터 인덱스 오류")?;
    let mut low = 0_usize;
    let mut high = count;
    while low < high {
        let middle = low.midpoint(high);
        if read_u64(DATA, 32_usize.strict_add(middle.strict_mul(INDEX_SIZE)))? < value {
            low = middle.strict_add(1);
        } else {
            high = middle;
        }
    }
    (low < count).ok_or("대한민국 경계 데이터 범위 오류")?;
    let block_index = low;
    let index = 32_usize.strict_add(block_index.strict_mul(INDEX_SIZE));
    let start = usize::try_from(read_u32(DATA, index.strict_add(8))?)
        .map_err(|source| format!("경계 블록 위치 오류: {source}"))?;
    let part = usize::try_from(read_u32(DATA, index.strict_add(12))?)
        .map_err(|source| format!("경계 파일 번호 오류: {source}"))?;
    let length = usize::try_from(read_u32(DATA, index.strict_add(20))?)
        .map_err(|source| format!("경계 블록 길이 오류: {source}"))?;
    let compressed = PARTS
        .get(part)
        .and_then(|bytes| {
            let end = start.checked_add(length)?;
            bytes.get(start..end)
        })
        .ok_or("대한민국 경계 데이터 길이 오류")?;
    let previous = if block_index > 0 {
        Some(read_u64(DATA, index.strict_sub(INDEX_SIZE))?)
    } else {
        None
    };
    if let Some(cache) = cache {
        let cache = &mut cache.pages;
        let entry = if let Some(position) = cache.iter().position(|(key, _)| *key == block_index) {
            cache.remove(position).ok_or("경계 캐시 위치 오류")?
        } else {
            let decoded = decode_page(compressed, previous)?;
            if cache.len() == 32 {
                let (_, mut page) = cache.pop_front().ok_or("경계 캐시 오류")?;
                *page = decoded;
                (block_index, page)
            } else {
                (block_index, Box::new(decoded))
            }
        };
        cache.push_back(entry);
        let page = &cache.back().ok_or("경계 캐시 오류")?.1;
        sample_page(value, page, index, block_index)
    } else {
        let page = decode_page(compressed, previous)?;
        sample_page(value, &page, index, block_index)
    }
}
fn sample_page(
    value: u64,
    page: &[u8; MAX_PAGE_SIZE],
    index: usize,
    block_index: usize,
) -> Result<Coordinates> {
    let vertices = usize::from(read_u16(page, 0)?);
    let triangles = usize::from(read_u16(page, 2)?);
    let records = 4_usize.strict_add(vertices.strict_mul(16));
    let mut low = 0;
    let mut high = triangles;
    while low < high {
        let middle = low.midpoint(high);
        if read_u64(page, records.strict_add(middle.strict_mul(RECORD_SIZE)))? < value {
            low = middle.strict_add(1);
        } else {
            high = middle;
        }
    }
    (low < triangles).ok_or("대한민국 경계 데이터 삼각형 오류")?;
    let record = records.strict_add(low.strict_mul(RECORD_SIZE));
    let end = read_u64(page, record)?;
    let begin = if low > 0 {
        read_u64(page, record.strict_sub(RECORD_SIZE))?.strict_add(1)
    } else if block_index > 0 {
        read_u64(DATA, index.strict_sub(INDEX_SIZE))?.strict_add(1)
    } else {
        0
    };
    let width = end
        .checked_sub(begin)
        .and_then(|difference| difference.checked_add(1))
        .ok_or("대한민국 경계 데이터 가중치 오류")?;
    let offset = value
        .checked_sub(begin)
        .ok_or("대한민국 경계 데이터 선택 오류")?;
    let first = ((float_from_u64(offset) + 0.5) / float_from_u64(width))
        .clamp(f64::EPSILON / 2.0, 1.0 - f64::EPSILON / 2.0);
    let second = float_from_u64(offset.reverse_bits() >> 12)
        .mul_add(1.0 / 4_503_599_627_370_496.0, 1.0 / 9_007_199_254_740_992.0);
    let (first_weight, second_weight) = if first + second > 1.0_f64 {
        (1.0_f64 - first, 1.0_f64 - second)
    } else {
        (first, second)
    };
    let weights = [
        1.0_f64 - first_weight - second_weight,
        first_weight,
        second_weight,
    ]
    .map(|weight| weight * (1.0 - 3.0 / 1_048_576.0) + 1.0 / 1_048_576.0);
    let mut latitude = [0.0_f64; 3];
    let mut longitude = [0.0_f64; 3];
    for (corner, (lat, lon)) in latitude.iter_mut().zip(&mut longitude).enumerate() {
        let vertex = usize::from(read_u16(
            page,
            record.strict_add(8).strict_add(corner.strict_mul(2)),
        )?);
        (vertex < vertices).ok_or("대한민국 경계 데이터 정점 오류")?;
        let position = 4_usize.strict_add(vertex.strict_mul(16));
        *lat = f64::from_bits(read_u64(page, position)?);
        *lon = f64::from_bits(read_u64(page, position.strict_add(8))?);
    }
    Ok(Coordinates {
        latitude: latitude
            .iter()
            .zip(weights)
            .map(|(coordinate, weight)| coordinate * weight)
            .sum(),
        longitude: longitude
            .iter()
            .zip(weights)
            .map(|(coordinate, weight)| coordinate * weight)
            .sum(),
    })
}
