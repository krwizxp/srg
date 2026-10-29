use self::data::{INDEX as DATA, INDEX_SHA256, PARTS};
use crate::{
    boundary_inflate::inflate,
    diagnostic::{AppError, Result},
    geo_uniform::{candidate, orientation},
    random_data::Coordinates,
};
use alloc::collections::VecDeque;
use std::sync::{LazyLock, OnceLock};
mod data;
mod hash;
#[cfg(test)]
mod tests;
const CACHE_BYTES: usize = 0x0010_0000;
const CACHE_PAGES: usize = 4096;
const HEADER_SIZE: usize = 128;
const KOREA_SUPP_RETRY_LIMIT: usize = 1024;
const KOREA_TILE_BYTES: &[u8] = include_bytes!("../data/korea-boundary-v4/tile-select-v5.bin");
const KOREA_TILE_COUNT: usize = 79113;
const KOREA_TILE_SHA256: [u8; 32] = [73, 47, 219, 238, 231, 231, 166, 72, 22, 215, 234, 144, 197, 124, 181, 119, 54, 133, 251, 249, 110, 11, 149, 145, 26, 139, 124, 220, 251, 242, 174, 139];
const MAX_PAGE_SIZE: usize = 32_768;
const MAX_PART_SIZE: usize = 0x0008_0000;
static BOUNDARY: LazyLock<Result<BoundaryIndex>> = LazyLock::new(|| {
    (hash::digest(DATA)? == INDEX_SHA256).ok_or("대한민국 경계 인덱스 SHA-256 오류")?;
    BoundaryIndex::parse(DATA, PARTS.len())
});
#[derive(Default)]
pub(super) struct BoundaryCache {
    bytes: usize,
    pages: VecDeque<(usize, DecodedPage)>,
}
struct BoundaryIndex {
    bounds: [f64; 4],
    columns: Box<[f64]>,
    rows: Box<[f64]>,
    polygon_count: usize,
    ring_count: usize,
    vertex_count: usize,
    source_hash: [u8; 32],
    tiles: Box<[Tile]>,
    sampler_tiles: Box<[u32]>,
    parts: Box<[Part]>,
}
struct Tile {
    state: u32,
    first: usize,
    count: usize,
}
struct Part {
    first: usize,
    count: usize,
    bytes: usize,
    digest: [u8; 32],
    validated: OnceLock<Result<()>>,
}
enum DecodedPage {
    Parity {
        base: bool,
        endpoints: Box<[f64]>,
    },
    Chain {
        ring: usize,
        owner: usize,
        hole: u32,
        first: usize,
        vertices: Box<[[f64; 2]]>,
    },
}
impl BoundaryIndex {
    fn parse(bytes: &[u8], expected_parts: usize) -> Result<Self> {
        (bytes.len() >= HEADER_SIZE && bytes.starts_with(b"SRGKR004"))
            .ok_or("대한민국 경계 데이터 버전 오류")?;
        let tile_count = read_count(bytes, 12)?;
        let part_count = read_count(bytes, 16)?;
        let page_count = read_count(bytes, 20)?;
        let polygon_count = read_count(bytes, 24)?;
        let ring_count = read_count(bytes, 28)?;
        let vertex_count = read_count(bytes, 32)?;
        let column_count = read_u32(bytes, 36)?;
        let row_count = read_u32(bytes, 40)?;
        let scale = read_u32(bytes, 44)?;
        (read_count(bytes, 8)? == HEADER_SIZE
            && tile_count > 0
            && tile_count <= 0x0004_0000
            && part_count > 0
            && part_count <= 4096
            && part_count == expected_parts
            && page_count > 0
            && page_count <= 1_000_000
            && polygon_count > 0
            && polygon_count <= ring_count
            && ring_count < vertex_count
            && vertex_count <= 20_000_000
            && column_count > 0
            && column_count <= 4096
            && row_count > 0
            && row_count <= 4096
            && usize::try_from(column_count.strict_mul(row_count)).ok() == Some(tile_count)
            && scale.is_power_of_two()
            && scale <= 256
            && bytes.len()
                == HEADER_SIZE
                    .strict_add(tile_count.strict_mul(12))
                    .strict_add(part_count.strict_mul(48))
            && bytes.len() <= 0x0080_0000)
            .ok_or("대한민국 경계 인덱스 구조 오류")?;
        let bounds = [
            read_f64(bytes, 64)?,
            read_f64(bytes, 72)?,
            read_f64(bytes, 80)?,
            read_f64(bytes, 88)?,
        ];
        let [west, south, east, north] = bounds;
        (bounds.into_iter().all(f64::is_finite)
            && west >= 124.0_f64
            && east <= 132.0_f64
            && west < east
            && south >= 32.0_f64
            && north <= 40.0_f64
            && south < north)
            .ok_or("대한민국 경계 범위 오류")?;
        let grid_west = read_f64(bytes, 48)?;
        let grid_south = read_f64(bytes, 56)?;
        let divisor = f64::from(scale);
        let grid_east = grid_west + f64::from(column_count) / divisor;
        let grid_north = grid_south + f64::from(row_count) / divisor;
        (grid_west.is_finite()
            && grid_south.is_finite()
            && (grid_west * divisor).fract() == 0.0_f64
            && (grid_south * divisor).fract() == 0.0_f64
            && grid_west <= west
            && west < grid_west + 1.0_f64 / divisor
            && grid_south <= south
            && south < grid_south + 1.0_f64 / divisor
            && east <= grid_east
            && east > grid_east - 1.0_f64 / divisor
            && north <= grid_north
            && north > grid_north - 1.0_f64 / divisor)
            .ok_or("대한민국 경계 타일 범위 오류")?;
        let columns = (0..=column_count)
            .map(|position| grid_west + f64::from(position) / divisor)
            .collect();
        let rows = (0..=row_count)
            .map(|position| grid_south + f64::from(position) / divisor)
            .collect();
        let mut source_hash = [0_u8; 32];
        source_hash.copy_from_slice(bytes.get(96..128).ok_or("대한민국 경계 원본 식별자 오류")?);
        let mut tiles = Vec::with_capacity(tile_count);
        let mut next_page = 0_usize;
        for tile_number in 0..tile_count {
            let offset = HEADER_SIZE.strict_add(tile_number.strict_mul(12));
            let state = read_u32(bytes, offset)?;
            let first = read_count(bytes, offset.strict_add(4))?;
            let count = read_count(bytes, offset.strict_add(8))?;
            (state <= 2
                && first == next_page
                && ((state == 2 && count > 0) || (state < 2 && count == 0)))
                .ok_or("대한민국 경계 타일 인덱스 오류")?;
            next_page = next_page
                .checked_add(count)
                .ok_or("대한민국 경계 블록 개수 오류")?;
            (next_page <= page_count).ok_or("대한민국 경계 블록 범위 오류")?;
            tiles.push(Tile {
                state,
                first,
                count,
            });
        }
        (next_page == page_count).ok_or("대한민국 경계 타일 블록 개수 오류")?;
        next_page = 0;
        let mut parts = Vec::with_capacity(part_count);
        let part_start = HEADER_SIZE.strict_add(tile_count.strict_mul(12));
        for part_number in 0..part_count {
            let offset = part_start.strict_add(part_number.strict_mul(48));
            let first = read_count(bytes, offset)?;
            let count = read_count(bytes, offset.strict_add(4))?;
            let length = read_count(bytes, offset.strict_add(8))?;
            (first == next_page
                && count > 0
                && count <= page_count
                && read_u32(bytes, offset.strict_add(12))? == 0
                && length > 64_usize.strict_add(count.strict_mul(12))
                && length <= MAX_PART_SIZE)
                .ok_or("대한민국 경계 조각 인덱스 오류")?;
            let mut digest = [0_u8; 32];
            digest.copy_from_slice(
                bytes
                    .get(offset.strict_add(16)..offset.strict_add(48))
                    .ok_or("대한민국 경계 조각 해시 오류")?,
            );
            parts.push(Part {
                first,
                count,
                bytes: length,
                digest,
                validated: OnceLock::new(),
            });
            next_page = next_page
                .checked_add(count)
                .ok_or("대한민국 경계 조각 개수 오류")?;
        }
        (next_page == page_count).ok_or("대한민국 경계 전체 블록 개수 오류")?;
        let sampler_tiles = validate_tile_index()?;
        Ok(Self {
            bounds,
            columns,
            rows,
            polygon_count,
            ring_count,
            vertex_count,
            source_hash,
            tiles: tiles.into_boxed_slice(),
            sampler_tiles,
            parts: parts.into_boxed_slice(),
        })
    }
    fn tile_bounds(&self, tile_number: usize) -> Result<[f64; 4]> {
        let width = self.columns.len().strict_sub(1);
        let column = tile_number.rem_euclid(width);
        let row = tile_number.div_euclid(width);
        Ok([
            *self.columns.get(column).ok_or("대한민국 타일 열 오류")?,
            *self.rows.get(row).ok_or("대한민국 타일 행 오류")?,
            *self
                .columns
                .get(column.strict_add(1))
                .ok_or("대한민국 타일 열 오류")?,
            *self
                .rows
                .get(row.strict_add(1))
                .ok_or("대한민국 타일 행 오류")?,
        ])
    }
    fn validate_part(&self, part: &Part, bytes: &[u8]) -> Result<()> {
        (bytes.len() == part.bytes
            && bytes.starts_with(b"SRGKP004")
            && read_count(bytes, 8)? == part.first
            && read_count(bytes, 12)? == part.count
            && read_count(bytes, 16)? == bytes.len()
            && read_u32(bytes, 20)? == 0
            && bytes.get(24..56) == Some(self.source_hash.as_slice())
            && read_u64(bytes, 56)? == 0
            && hash::digest(bytes)? == part.digest)
            .ok_or("대한민국 경계 조각 버전·길이·SHA-256 오류")?;
        let mut expected = 64_usize.strict_add(part.count.strict_mul(12));
        for local in 0..part.count {
            let entry = 64_usize.strict_add(local.strict_mul(12));
            let start = read_count(bytes, entry)?;
            let size = read_count(bytes, entry.strict_add(4))?;
            let raw_size = read_count(bytes, entry.strict_add(8))?;
            (start == expected && size > 0 && raw_size >= 32 && raw_size <= MAX_PAGE_SIZE)
                .ok_or("대한민국 경계 압축 블록 인덱스 오류")?;
            expected = start
                .checked_add(size)
                .ok_or("대한민국 경계 압축 블록 길이 오류")?;
            (expected <= bytes.len()).ok_or("대한민국 경계 압축 블록 범위 오류")?;
        }
        (expected == bytes.len()).ok_or("대한민국 경계 조각 끝 위치 오류")?;
        Ok(())
    }
    fn decode_page(
        &self,
        page_number: usize,
        tile_number: usize,
        bytes: &[u8],
        raw_size: usize,
    ) -> Result<DecodedPage> {
        let mut raw = [0_u8; MAX_PAGE_SIZE];
        let length = inflate(bytes, &mut raw)?;
        (length == raw_size
            && raw.starts_with(b"SRGKB004")
            && read_count(&raw, 8)? == tile_number
            && read_count(&raw, 12)? == page_number)
            .ok_or("대한민국 경계 블록 버전·길이·번호 오류")?;
        let count = read_count(&raw, 20)?;
        let field_a = read_count(&raw, 24)?;
        let field_b = read_count(&raw, 28)?;
        match read_u32(&raw, 16)? {
            0 => {
                (count <= 4092
                    && length == 32_usize.strict_add(count.strict_mul(8))
                    && field_a <= 1
                    && field_b == 0)
                    .ok_or("대한민국 경계 교차 인덱스 오류")?;
                let [_, south, _, north] = self.tile_bounds(tile_number)?;
                let mut previous = south;
                let mut endpoints = Vec::with_capacity(count);
                for offset in 0..count {
                    let value = read_f64(&raw, 32_usize.strict_add(offset.strict_mul(8)))?;
                    (value.is_finite() && value > previous && value <= north)
                        .ok_or("대한민국 경계 교차 범위 오류")?;
                    endpoints.push(value);
                    previous = value;
                }
                Ok(DecodedPage::Parity {
                    base: field_a == 1,
                    endpoints: endpoints.into_boxed_slice(),
                })
            }
            1 => {
                let first = read_count(&raw, 32)?;
                (count > 0
                    && count <= 1024
                    && field_a < self.ring_count
                    && field_b < self.polygon_count
                    && first
                        .checked_add(count)
                        .is_some_and(|end| end < self.vertex_count)
                    && read_u32(&raw, 36)? <= 1
                    && length == 40_usize.strict_add(count.strict_add(1).strict_mul(16)))
                .ok_or("대한민국 경계 링 인덱스 오류")?;
                let words = count.strict_add(1).strict_mul(2);
                let mut vertices = Vec::with_capacity(count.strict_add(1));
                let [west, south, east, north] = self.bounds;
                for position in 0..=count {
                    let mut point = [0.0_f64; 2];
                    for (axis, coordinate) in point.iter_mut().enumerate() {
                        let word = position.strict_mul(2).strict_add(axis);
                        let mut number = [0_u8; 8];
                        for (lane, byte) in number.iter_mut().enumerate() {
                            let offset =
                                40_usize.strict_add(lane.strict_mul(words)).strict_add(word);
                            *byte = *raw.get(offset).ok_or("대한민국 경계 정점 길이 오류")?;
                        }
                        *coordinate = f64::from_le_bytes(number);
                    }
                    (point[0] >= west
                        && point[0] <= east
                        && point[1] >= south
                        && point[1] <= north)
                        .ok_or("대한민국 경계 정점 범위 오류")?;
                    vertices.push(point);
                }
                Ok(DecodedPage::Chain {
                    ring: field_a,
                    owner: field_b,
                    hole: read_u32(&raw, 36)?,
                    first,
                    vertices: vertices.into_boxed_slice(),
                })
            }
            _ => Err("대한민국 경계 블록 종류 오류".into()),
        }
    }
    fn page(&self, page_number: usize, tile_number: usize) -> Result<DecodedPage> {
        let part_number = self
            .parts
            .partition_point(|part| part.first <= page_number)
            .checked_sub(1)
            .ok_or("대한민국 경계 조각 선택 오류")?;
        let part = self
            .parts
            .get(part_number)
            .ok_or("대한민국 경계 조각 선택 오류")?;
        let bytes = *PARTS
            .get(part_number)
            .ok_or("대한민국 경계 내장 조각 누락")?;
        part.validated
            .get_or_init(|| self.validate_part(part, bytes))
            .as_ref()
            .map_err(|source| format!("대한민국 경계 조각 검사 실패: {source}"))?;
        let local = page_number
            .checked_sub(part.first)
            .ok_or("대한민국 경계 블록 순번 오류")?;
        (local < part.count).ok_or("대한민국 경계 블록 순번 오류")?;
        let entry = 64_usize.strict_add(local.strict_mul(12));
        let start = read_count(bytes, entry)?;
        let size = read_count(bytes, entry.strict_add(4))?;
        let raw_size = read_count(bytes, entry.strict_add(8))?;
        let compressed = bytes
            .get(start..start.strict_add(size))
            .ok_or("대한민국 경계 블록 위치 오류")?;
        self.decode_page(page_number, tile_number, compressed, raw_size)
    }
    fn covers(&self, point: &Coordinates, cache: &mut BoundaryCache) -> Result<bool> {
        let [west, south, east, north] = self.bounds;
        if !(point.longitude >= west
            && point.longitude <= east
            && point.latitude >= south
            && point.latitude <= north)
        {
            (point.longitude.is_finite() && point.latitude.is_finite())
                .ok_or("대한민국 포함 판정 좌표 오류")?;
            return Ok(false);
        }
        let width = self.columns.len().strict_sub(1);
        let column = self
            .columns
            .partition_point(|edge| *edge <= point.longitude)
            .min(width)
            .strict_sub(1);
        let row = self
            .rows
            .partition_point(|edge| *edge <= point.latitude)
            .min(self.rows.len().strict_sub(1))
            .strict_sub(1);
        let tile_number = row.strict_mul(width).strict_add(column);
        let tile = self
            .tiles
            .get(tile_number)
            .ok_or("대한민국 경계 타일 선택 오류")?;
        if tile.state != 2 {
            return Ok(tile.state == 1);
        }
        let mut parity = false;
        let mut boundary = false;
        let mut chains_started = false;
        let mut last_endpoint = None;
        let mut last_chain = None;
        for offset in 0..tile.count {
            let page_number = tile.first.strict_add(offset);
            let page = cache.page(self, page_number, tile_number)?;
            match page {
                DecodedPage::Parity { base, endpoints } => {
                    (!chains_started
                        && (offset == 0 || !*base)
                        && last_endpoint
                            .zip(endpoints.first())
                            .is_none_or(|(previous, next)| previous < *next))
                    .ok_or("대한민국 경계 교차 블록 순서 오류")?;
                    parity ^= *base
                        ^ !endpoints
                            .partition_point(|edge| *edge <= point.latitude)
                            .is_multiple_of(2);
                    if let Some(last) = endpoints.last() {
                        last_endpoint = Some(*last);
                    }
                }
                DecodedPage::Chain {
                    ring,
                    owner,
                    hole,
                    first,
                    vertices,
                } => {
                    if let Some((previous_ring, previous_owner, previous_hole, previous_end)) =
                        last_chain
                    {
                        (*ring > previous_ring
                            || (*ring == previous_ring
                                && *owner == previous_owner
                                && *hole == previous_hole
                                && *first >= previous_end))
                            .ok_or("대한민국 경계 링 순서 오류")?;
                    }
                    last_chain = Some((
                        *ring,
                        *owner,
                        *hole,
                        first.strict_add(vertices.len().strict_sub(1)),
                    ));
                    (offset > 0).ok_or("대한민국 경계 교차 블록 누락")?;
                    chains_started = true;
                    for pair in vertices.windows(2) {
                        let [start, end] = pair else {
                            return Err("대한민국 경계 선분 오류".into());
                        };
                        let within = point.longitude >= start[0].min(end[0])
                            && point.longitude <= start[0].max(end[0])
                            && point.latitude >= start[1].min(end[1])
                            && point.latitude <= start[1].max(end[1]);
                        let crosses = (start[1] > point.latitude) != (end[1] > point.latitude);
                        if within || crosses {
                            let side = orientation(*start, *end, [point.longitude, point.latitude]);
                            boundary |= within && side == 0.0_f64;
                            parity ^= crosses
                                && if end[1] > start[1] {
                                    side > 0.0_f64
                                } else {
                                    side < 0.0_f64
                                };
                        }
                    }
                }
            }
        }
        chains_started.ok_or("대한민국 경계 링 누락")?;
        Ok(boundary || parity)
    }
}
impl DecodedPage {
    fn retained_bytes(&self) -> usize {
        let payload = match self {
            Self::Parity { endpoints, .. } => endpoints.len().strict_mul(8),
            Self::Chain { vertices, .. } => vertices.len().strict_mul(16),
        };
        payload.strict_add(size_of::<(usize, Self)>())
    }
}
impl BoundaryCache {
    fn page(
        &mut self,
        index: &BoundaryIndex,
        page_number: usize,
        tile_number: usize,
    ) -> Result<&DecodedPage> {
        let item = if let Some(position) = self
            .pages
            .iter()
            .position(|(number, _)| *number == page_number)
        {
            self.pages
                .remove(position)
                .ok_or("대한민국 경계 캐시 위치 오류")?
        } else {
            let decoded = index.page(page_number, tile_number)?;
            let additional = decoded.retained_bytes();
            while self.pages.len() >= CACHE_PAGES || self.bytes.strict_add(additional) > CACHE_BYTES
            {
                let (_, removed) = self
                    .pages
                    .pop_front()
                    .ok_or("대한민국 경계 캐시 용량 오류")?;
                self.bytes = self.bytes.strict_sub(removed.retained_bytes());
            }
            self.bytes = self.bytes.strict_add(additional);
            (page_number, decoded)
        };
        self.pages.push_back(item);
        Ok(&self.pages.back().ok_or("대한민국 경계 캐시 오류")?.1)
    }
}
fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let value = bytes
        .get(offset..)
        .and_then(|tail| tail.first_chunk::<4>())
        .ok_or("대한민국 경계 데이터 구조 오류")?;
    Ok(u32::from_le_bytes(*value))
}
fn read_count(bytes: &[u8], offset: usize) -> Result<usize> {
    usize::try_from(read_u32(bytes, offset)?)
        .map_err(|source| format!("대한민국 경계 길이 오류: {source}").into())
}
fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let value = bytes
        .get(offset..)
        .and_then(|tail| tail.first_chunk::<8>())
        .ok_or("대한민국 경계 데이터 구조 오류")?;
    Ok(u64::from_le_bytes(*value))
}
fn read_f64(bytes: &[u8], offset: usize) -> Result<f64> {
    Ok(f64::from_bits(read_u64(bytes, offset)?))
}
fn validate_tile_index() -> Result<Box<[u32]>> {
    (KOREA_TILE_BYTES.len() == 32_768 && hash::digest(KOREA_TILE_BYTES)? == KOREA_TILE_SHA256)
        .ok_or("대한민국 좌표 타일 인덱스 SHA-256 오류")?;
    let mut selected = Vec::with_capacity(KOREA_TILE_COUNT);
    for tile in 0..262_144_usize {
        let byte = KOREA_TILE_BYTES[tile >> 3];
        if byte & (1_u8 << (tile & 7)) != 0 {
            selected.push(u32::try_from(tile).map_err(|_| "대한민국 타일 번호 오류")?);
        }
    }
    (selected.len() == KOREA_TILE_COUNT).ok_or("대한민국 좌표 타일 개수 오류")?;
    Ok(selected.into_boxed_slice())
}
fn remap_tile(value: u64, index: &BoundaryIndex) -> Result<Option<u64>> {
    let selector = usize::try_from(value >> 46).map_err(|_| "대한민국 타일 선택값 오류")?;
    if selector >= KOREA_TILE_COUNT.strict_mul(2) {
        return Ok(None);
    }
    let tile = u64::from(*index.sampler_tiles
        .get(selector / 2).ok_or("대한민국 좌표 타일 선택 범위 오류")?);
    let vertical = ((tile >> 9) << 23) | ((value >> 23) & 0x7f_ffff);
    let horizontal = ((tile & 511) << 23) | (value & 0x7f_ffff);
    Ok(Some((vertical << 32) | horizontal))
}
pub(super) fn sample<F>(
    value: u64,
    next_supp: &mut F,
    cache: Option<&mut BoundaryCache>,
) -> Result<Coordinates>
where
    F: FnMut(&'static str) -> Result<u64>,
{
    let index = BOUNDARY
        .as_ref()
        .map_err(|source| format!("대한민국 경계 초기화 실패: {source}"))?;
    let mut local = BoundaryCache::default();
    let active_cache = cache.unwrap_or(&mut local);
    let mut proposed = value;
    for reads in 0..=KOREA_SUPP_RETRY_LIMIT {
        if let Some(mapped) = remap_tile(proposed, index)? {
            let point = candidate(mapped, index.bounds)?;
            if index.covers(&point, active_cache)? {
                return Ok(point);
            }
        }
        if reads == KOREA_SUPP_RETRY_LIMIT {
            return Err("대한민국 좌표 보완 난수 시도 횟수 초과".into());
        }
        proposed = next_supp("대한민국 좌표 재추출")
            .map_err(|source| AppError::context("대한민국 좌표 보완 난수 공급 실패", source))?;
    }
    Err("대한민국 좌표 보완 난수 시도 횟수 초과".into())
}
