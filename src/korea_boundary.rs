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
    DATA.starts_with(b"SRGKR003")
        .ok_or("대한민국 경계 데이터 버전 오류")?;
    let total = read_u64(DATA, 24)?;
    let parts = index_count(DATA, 8)?;
    (total > 0 && parts == PARTS.len() && DATA.len() == 32_usize.strict_add(parts.strict_mul(16)))
        .ok_or("대한민국 경계 인덱스 오류")?;
    let scaled = u128::from(value).strict_mul(u128::from(total)) >> 64_u32;
    let position =
        u64::try_from(scaled).map_err(|source| format!("좌표 입력 구간 오류: {source}"))?;
    let (begin, weight, vertices) = select(position, parts, cache)?;
    let first = (u128::from(begin) << 64_u32).div_ceil(u128::from(total));
    let end = (u128::from(begin.strict_add(weight)) << 64_u32).div_ceil(u128::from(total));
    let rank = u64::try_from(u128::from(value).strict_sub(first))
        .map_err(|source| format!("좌표 입력 순번 오류: {source}"))?;
    let count = u64::try_from(end.strict_sub(first))
        .map_err(|source| format!("좌표 입력 개수 오류: {source}"))?;
    crate::geo_uniform::sample(&vertices, rank, count)
}
fn index_count(bytes: &[u8], offset: usize) -> Result<usize> {
    usize::try_from(read_u32(bytes, offset)?)
        .map_err(|source| format!("경계 인덱스 크기 오류: {source}").into())
}
fn search(bytes: &[u8], start: usize, count: usize, value: u64) -> Result<usize> {
    let mut low = 0_usize;
    let mut high = count;
    while low < high {
        let middle = low.midpoint(high);
        if read_u64(bytes, start.strict_add(middle.strict_mul(16)))? < value {
            low = middle.strict_add(1);
        } else {
            high = middle;
        }
    }
    (low < count).ok_or("대한민국 경계 데이터 범위 오류")?;
    Ok(low)
}
fn select(
    value: u64,
    count: usize,
    cache: Option<&mut BoundaryCache>,
) -> Result<(u64, u64, [[f64; 2]; 3])> {
    let part_index = search(DATA, 32, count, value)?;
    let root_entry = 32_usize.strict_add(part_index.strict_mul(16));
    let part = PARTS.get(part_index).ok_or("경계 조각 번호 오류")?;
    let blocks = index_count(part, 8)?;
    let declared = index_count(DATA, root_entry.strict_add(8))?;
    (part.starts_with(b"SRGKP003")
        && blocks > 0
        && blocks == index_count(DATA, root_entry.strict_add(12))?
        && part.len() == declared
        && part.len() <= 524_288)
        .ok_or("경계 조각 구조 오류")?;
    let first = index_count(part, 12)?;
    let local = search(part, 16, blocks, value)?;
    let entry = 16_usize.strict_add(local.strict_mul(16));
    let start = index_count(part, entry.strict_add(8))?;
    let length = index_count(part, entry.strict_add(12))?;
    (start >= 16_usize.strict_add(blocks.strict_mul(16))).ok_or("경계 조각 위치 오류")?;
    let compressed = part
        .get(start..start.checked_add(length).ok_or("경계 조각 길이 오류")?)
        .ok_or("경계 조각 길이 오류")?;
    let previous = if local > 0 {
        Some(read_u64(part, entry.strict_sub(16))?)
    } else if part_index > 0 {
        Some(read_u64(DATA, root_entry.strict_sub(16))?)
    } else {
        None
    };
    let block = first.strict_add(local);
    if let Some(cache) = cache {
        let pages = &mut cache.pages;
        let item = if let Some(position) = pages.iter().position(|(key, _)| *key == block) {
            pages.remove(position).ok_or("경계 캐시 위치 오류")?
        } else {
            let decoded = decode_page(compressed, previous)?;
            if pages.len() == 32 {
                let (_, mut page) = pages.pop_front().ok_or("경계 캐시 오류")?;
                *page = decoded;
                (block, page)
            } else {
                (block, Box::new(decoded))
            }
        };
        pages.push_back(item);
        pick(value, &pages.back().ok_or("경계 캐시 오류")?.1, previous)
    } else {
        let page = decode_page(compressed, previous)?;
        pick(value, &page, previous)
    }
}
fn pick(
    value: u64,
    page: &[u8; MAX_PAGE_SIZE],
    previous: Option<u64>,
) -> Result<(u64, u64, [[f64; 2]; 3])> {
    let vertices = usize::from(read_u16(page, 0)?);
    let triangles = usize::from(read_u16(page, 2)?);
    let records = 4_usize.strict_add(vertices.strict_mul(16));
    let mut low = 0_usize;
    let mut high = triangles;
    while low < high {
        let middle = low.midpoint(high);
        if read_u64(page, records.strict_add(middle.strict_mul(RECORD_SIZE)))? < value {
            low = middle.strict_add(1);
        } else {
            high = middle;
        }
    }
    (low < triangles).ok_or("대한민국 경계 삼각형 오류")?;
    let entry = records.strict_add(low.strict_mul(RECORD_SIZE));
    let end = read_u64(page, entry)?;
    let begin = if low > 0 {
        read_u64(page, entry.strict_sub(RECORD_SIZE))?.strict_add(1)
    } else {
        previous.map_or(0, |v| v.strict_add(1))
    };
    let weight = end
        .checked_sub(begin)
        .and_then(|v| v.checked_add(1))
        .ok_or("대한민국 경계 가중치 오류")?;
    let mut points = [[0.0_f64; 2]; 3];
    for (corner, point) in points.iter_mut().enumerate() {
        let vertex = usize::from(read_u16(
            page,
            entry.strict_add(8).strict_add(corner.strict_mul(2)),
        )?);
        (vertex < vertices).ok_or("경계 정점 번호 오류")?;
        let offset = 4_usize.strict_add(vertex.strict_mul(16));
        *point = [
            f64::from_bits(read_u64(page, offset)?),
            f64::from_bits(read_u64(page, offset.strict_add(8))?),
        ];
    }
    Ok((begin, weight, points))
}
