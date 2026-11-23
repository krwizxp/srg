use self::data::{INDEX as DATA, PARTS};
use crate::{
    boundary_inflate::InflatedPage,
    diagnostic::{AppError, Result},
    geo_uniform::BoundaryAreaBounds,
    numeric::{low_u16_from_u64, low_u32_from_u64},
    random_data::Coordinates,
};
use alloc::collections::VecDeque;
use core::cmp::Ordering;
use std::sync::{LazyLock, OnceLock};
mod atlas;
mod data;
const CACHE_BYTES: usize = 896 * 1024;
const CACHE_PAGES: usize = 4096;
const HEADER_SIZE: usize = 128;
const IEEE_F64_MANTISSA_ALIGN: f64 = 4_503_599_627_370_496.0_f64;
const KOREA_SUPP_RETRY_LIMIT: usize = 1024;
const KOREA_TILE_BYTES: &[u8] = include_bytes!("../data/korea-boundary-v4/tile-select-v5.bin");
const KOREA_TILE_COUNT: usize = 79113;
const MAX_GRID_DIMENSION_F64: f64 = 4096.0_f64;
const MAX_PAGE_SIZE: usize = 0x8000;
const MAX_PART_SIZE: usize = 0x0008_0000;
const TILE_INSIDE: u16 = u16::MAX;
const TILE_OUTSIDE: u16 = u16::MAX - 1;
static BOUNDARY: LazyLock<Result<BoundaryIndex>> =
    LazyLock::new(|| BoundaryIndex::try_from((DATA, PARTS.len())));
#[derive(Default)]
pub(super) struct BoundaryCache {
    bytes: usize,
    pages: VecDeque<(u32, CachedPage)>,
    resident: Vec<u64>,
    scratch: Vec<u8>,
}
struct CachedPage {
    bytes: u32,
    decoded: DecodedPage,
}
struct BoundaryIndex {
    area_bounds: BoundaryAreaBounds,
    atlas: atlas::Atlas<'static>,
    bounds: [f64; 4],
    column_count: u16,
    divisor: f64,
    fine_divisor: f64,
    fine_inv_divisor: f64,
    grid_south: f64,
    grid_west: f64,
    inv_divisor: f64,
    mixed_tiles: Box<[MixedTile]>,
    page_count: usize,
    parts: Box<[Part]>,
    polygon_count: usize,
    ring_count: usize,
    row_count: u16,
    sampler_tiles: Box<[[u8; 3]]>,
    source_hash: [u8; 32],
    tile_states: Box<[u16]>,
    vertex_count: usize,
}
struct MixedTile {
    count: u32,
    first: u32,
}
struct Part {
    bytes: usize,
    count: usize,
    first: usize,
    validated: OnceLock<Result<()>>,
}
enum DecodedPage {
    Chain {
        first: u32,
        hole: u32,
        owner: u32,
        ring: u32,
        vertices: Box<[[f64; 2]]>,
    },
    Parity {
        base: bool,
        endpoints: Box<[f64]>,
    },
}
#[derive(Clone, Copy)]
enum DecodedPageRef<'page> {
    Chain {
        first: u32,
        hole: u32,
        owner: u32,
        ring: u32,
        vertices: &'page [[f64; 2]],
    },
    Parity {
        base: bool,
        endpoints: &'page [f64],
    },
}
pub(super) struct SampleState<'cache> {
    active_cache: &'cache mut BoundaryCache,
    index: &'static BoundaryIndex,
    point: Option<Coordinates>,
    supplemental_reads: usize,
}
impl BoundaryIndex {
    fn covers_sample(&self, point: &Coordinates, cache: &mut BoundaryCache) -> Result<bool> {
        let width = usize::from(self.column_count);
        let column = grid_axis_index(
            self.grid_west,
            self.divisor,
            self.inv_divisor,
            self.column_count.strict_sub(1),
            point.longitude,
        );
        let row = grid_axis_index(
            self.grid_south,
            self.divisor,
            self.inv_divisor,
            self.row_count.strict_sub(1),
            point.latitude,
        );
        let tile_number = usize::from(row)
            .strict_mul(width)
            .strict_add(usize::from(column));
        let tile_state = *self
            .tile_states
            .get(tile_number)
            .ok_or("대한민국 경계 타일 선택 오류")?;
        if tile_state >= TILE_OUTSIDE {
            return Ok(tile_state == TILE_INSIDE);
        }
        match self.atlas.state(self, point, column, row, tile_state)? {
            0 => return Ok(false),
            1 => return Ok(true),
            _ => {}
        }
        let tile = self
            .mixed_tiles
            .get(usize::from(tile_state))
            .ok_or("대한민국 경계 혼합 타일 선택 오류")?;
        let mut parity = false;
        let mut boundary = false;
        let mut chains_started = false;
        let mut last_endpoint = None;
        let mut last_chain = None;
        let first_page = tile.first;
        for offset in 0..tile.count {
            let page_number = first_page.strict_add(offset);
            match cache.page(self, page_number, tile_number)? {
                DecodedPageRef::Parity { base, endpoints } => {
                    (!chains_started
                        && (offset == 0 || !base)
                        && last_endpoint
                            .zip(endpoints.first())
                            .is_none_or(|(previous, next)| previous < *next))
                    .ok_or("대한민국 경계 교차 블록 순서 오류")?;
                    parity ^= base
                        ^ !endpoints
                            .partition_point(|edge| *edge <= point.latitude)
                            .is_multiple_of(2);
                    if let Some(&last) = endpoints.last() {
                        last_endpoint = Some(last);
                    }
                }
                DecodedPageRef::Chain {
                    first,
                    hole,
                    owner,
                    ring,
                    vertices,
                } => {
                    if let Some((previous_ring, previous_owner, previous_hole, previous_end)) =
                        last_chain
                    {
                        (ring > previous_ring
                            || (ring == previous_ring
                                && owner == previous_owner
                                && hole == previous_hole
                                && first >= previous_end))
                            .ok_or("대한민국 경계 링 순서 오류")?;
                    }
                    let [b0, b1, b2, b3, ..] = vertices.len().strict_sub(1).to_le_bytes();
                    let span = u32::from_le_bytes([b0, b1, b2, b3]);
                    last_chain = Some((ring, owner, hole, first.strict_add(span)));
                    (offset > 0).ok_or("대한민국 경계 교차 블록 누락")?;
                    chains_started = true;
                    for &[start, end] in vertices.array_windows::<2>() {
                        if (point.latitude < start[1] && point.latitude < end[1])
                            || (point.latitude > start[1] && point.latitude > end[1])
                            || (point.longitude > start[0] && point.longitude > end[0])
                        {
                            continue;
                        }
                        let crosses = (start[1] > point.latitude) != (end[1] > point.latitude);
                        if point.longitude < start[0] && point.longitude < end[0] {
                            parity ^= crosses;
                            continue;
                        }
                        let dx = end[0] - start[0];
                        let dy = end[1] - start[1];
                        let px = point.longitude - start[0];
                        let py = point.latitude - start[1];
                        let left = dx * py;
                        let right = dy * px;
                        let approximate = left - right;
                        let determinant: f64 = if approximate.abs()
                            > 3.330_669_073_875_471_6e-16_f64 * (left.abs() + right.abs())
                        {
                            approximate
                        } else {
                            let mut expansion = [0.0_f64; 4];
                            let mut length = 0_usize;
                            for mut value in
                                [dx.mul_add(py, -left), left, -dy.mul_add(px, -right), -right]
                            {
                                let mut count = 0_usize;
                                let previous = expansion;
                                for entry in previous.into_iter().take(length) {
                                    let total = value + entry;
                                    let virtual_value = total - value;
                                    let error =
                                        (value - (total - virtual_value)) + (entry - virtual_value);
                                    if error != 0.0_f64 {
                                        *expansion
                                            .get_mut(count)
                                            .ok_or("대한민국 경계 좌표 누산 범위 오류")? = error;
                                        count = count.strict_add(1);
                                    }
                                    value = total;
                                }
                                if value != 0.0_f64 || count == 0 {
                                    *expansion
                                        .get_mut(count)
                                        .ok_or("대한민국 경계 좌표 누산 범위 오류")? = value;
                                    count = count.strict_add(1);
                                }
                                length = count;
                            }
                            expansion.iter().take(length).sum()
                        };
                        let side = determinant.partial_cmp(&0.0_f64);
                        boundary |= side == Some(Ordering::Equal);
                        parity ^= crosses
                            && if end[1] > start[1] {
                                side == Some(Ordering::Greater)
                            } else {
                                side == Some(Ordering::Less)
                            };
                    }
                }
            }
        }
        chains_started.ok_or("대한민국 경계 링 누락")?;
        Ok(boundary || parity)
    }
    fn decode_page(
        &self,
        page_number: usize,
        tile_number: usize,
        bytes: &[u8],
        raw_size: usize,
        scratch: &mut Vec<u8>,
    ) -> Result<CachedPage> {
        if scratch.len() < raw_size {
            scratch.resize(raw_size, 0);
        }
        let target = scratch
            .get_mut(..raw_size)
            .ok_or("대한민국 경계 블록 버퍼 범위 오류")?;
        let decoded = InflatedPage::try_from((bytes, target))?;
        let raw = decoded.as_ref();
        let length = raw.len();
        (length == raw_size
            && raw.starts_with(b"SRGKB004")
            && read_count(raw, 8)? == tile_number
            && read_count(raw, 12)? == page_number)
            .ok_or("대한민국 경계 블록 버전·길이·번호 오류")?;
        let count = read_count(raw, 20)?;
        let field_a = read_count(raw, 24)?;
        let field_b = read_count(raw, 28)?;
        match read_u32(raw, 16)? {
            0 => {
                (count <= 4092
                    && length == 32_usize.strict_add(count.strict_mul(8))
                    && field_a <= 1
                    && field_b == 0)
                    .ok_or("대한민국 경계 교차 인덱스 오류")?;
                let width = usize::from(self.column_count);
                let row = tile_number.div_euclid(width);
                let row_u16 = u16::try_from(row)
                    .ok()
                    .filter(|&value| value < self.row_count)
                    .ok_or("대한민국 타일 행 오류")?;
                let south = axis_edge(self.grid_south, self.inv_divisor, row_u16);
                let north = axis_edge(self.grid_south, self.inv_divisor, row_u16.strict_add(1));
                let mut previous = south;
                let mut endpoints = Vec::with_capacity(count);
                for chunk in raw
                    .get(32..)
                    .ok_or("대한민국 경계 교차 길이 오류")?
                    .as_chunks::<8>()
                    .0
                {
                    let value = f64::from_le_bytes(*chunk);
                    (value.is_finite() && value > previous && value <= north)
                        .ok_or("대한민국 경계 교차 범위 오류")?;
                    endpoints.push(value);
                    previous = value;
                }
                let [b0, b1, b2, b3, ..] = endpoints
                    .len()
                    .strict_mul(8)
                    .strict_add(size_of::<(u32, CachedPage)>())
                    .to_le_bytes();
                Ok(CachedPage {
                    bytes: u32::from_le_bytes([b0, b1, b2, b3]),
                    decoded: DecodedPage::Parity {
                        base: field_a == 1,
                        endpoints: endpoints.into_boxed_slice(),
                    },
                })
            }
            1 => {
                let first_u32 = read_u32(raw, 32)?;
                let first = first_u32 as usize;
                let hole = read_u32(raw, 36)?;
                (count > 0
                    && count <= 1024
                    && field_a < self.ring_count
                    && field_b < self.polygon_count
                    && first
                        .checked_add(count)
                        .is_some_and(|end| end < self.vertex_count)
                    && hole <= 1
                    && length == 40_usize.strict_add(count.strict_add(1).strict_mul(16)))
                .ok_or("대한민국 경계 링 인덱스 오류")?;
                let words = count.strict_add(1).strict_mul(2);
                let payload = raw.get(40..).ok_or("대한민국 경계 정점 길이 오류")?;
                let mut plane_iter = payload.chunks_exact(words);
                let mut planes = [[].as_slice(); 8];
                for slot in &mut planes {
                    *slot = plane_iter.next().ok_or("대한민국 경계 정점 길이 오류")?;
                }
                let [p0, p1, p2, p3, p4, p5, p6, p7] = planes.map(|plane| plane.as_chunks::<2>().0);
                let mut vertices = Vec::with_capacity(count.strict_add(1));
                let [west, south, east, north] = self.bounds;
                for (((b0, b1), (b2, b3)), ((b4, b5), (b6, b7))) in p0
                    .iter()
                    .zip(p1)
                    .zip(p2.iter().zip(p3))
                    .zip(p4.iter().zip(p5).zip(p6.iter().zip(p7)))
                {
                    let lon = f64::from_le_bytes([
                        b0[0], b1[0], b2[0], b3[0], b4[0], b5[0], b6[0], b7[0],
                    ]);
                    let lat = f64::from_le_bytes([
                        b0[1], b1[1], b2[1], b3[1], b4[1], b5[1], b6[1], b7[1],
                    ]);
                    (lon >= west && lon <= east && lat >= south && lat <= north)
                        .ok_or("대한민국 경계 정점 범위 오류")?;
                    vertices.push([lon, lat]);
                }
                let [b0, b1, b2, b3, ..] = vertices
                    .len()
                    .strict_mul(16)
                    .strict_add(size_of::<(u32, CachedPage)>())
                    .to_le_bytes();
                let [r0, r1, r2, r3, ..] = field_a.to_le_bytes();
                let [o0, o1, o2, o3, ..] = field_b.to_le_bytes();
                Ok(CachedPage {
                    bytes: u32::from_le_bytes([b0, b1, b2, b3]),
                    decoded: DecodedPage::Chain {
                        first: first_u32,
                        hole,
                        owner: u32::from_le_bytes([o0, o1, o2, o3]),
                        ring: u32::from_le_bytes([r0, r1, r2, r3]),
                        vertices: vertices.into_boxed_slice(),
                    },
                })
            }
            _ => Err("대한민국 경계 블록 종류 오류".into()),
        }
    }
    fn page(
        &self,
        page_number: usize,
        tile_number: usize,
        scratch: &mut Vec<u8>,
    ) -> Result<CachedPage> {
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
            .map_err(|source| AppError::prefixed_ref("대한민국 경계 조각 검사 실패", source))?;
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
        self.decode_page(page_number, tile_number, compressed, raw_size, scratch)
    }
    #[cold]
    #[inline(never)]
    fn validate_part(&self, part: &Part, bytes: &[u8]) -> Result<()> {
        (bytes.len() == part.bytes
            && bytes.starts_with(b"SRGKP004")
            && read_count(bytes, 8)? == part.first
            && read_count(bytes, 12)? == part.count
            && read_count(bytes, 16)? == bytes.len()
            && read_u32(bytes, 20)? == 0
            && bytes.get(24..56) == Some(self.source_hash.as_slice())
            && read_u64(bytes, 56)? == 0)
            .ok_or("대한민국 경계 조각 버전·길이 오류")?;
        let mut expected = 64_usize.strict_add(part.count.strict_mul(12));
        for local in 0..part.count {
            let entry = 64_usize.strict_add(local.strict_mul(12));
            let start = read_count(bytes, entry)?;
            let size = read_count(bytes, entry.strict_add(4))?;
            let raw_size = read_count(bytes, entry.strict_add(8))?;
            (start == expected && size > 0 && (32..=MAX_PAGE_SIZE).contains(&raw_size))
                .ok_or("대한민국 경계 압축 블록 인덱스 오류")?;
            expected = start
                .checked_add(size)
                .ok_or("대한민국 경계 압축 블록 길이 오류")?;
            (expected <= bytes.len()).ok_or("대한민국 경계 압축 블록 범위 오류")?;
        }
        (expected == bytes.len()).ok_or("대한민국 경계 조각 끝 위치 오류")?;
        Ok(())
    }
}
impl TryFrom<(&[u8], usize)> for BoundaryIndex {
    type Error = AppError;
    fn try_from((bytes, expected_parts): (&[u8], usize)) -> Result<Self> {
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
        let inv_divisor = 1.0_f64 / divisor;
        let mut source_hash = [0_u8; 32];
        source_hash.copy_from_slice(bytes.get(96..128).ok_or("대한민국 경계 원본 식별자 오류")?);
        let mut tile_states = Vec::with_capacity(tile_count);
        let mut mixed_tiles = Vec::new();
        let mut next_page = 0_usize;
        let mut next_ordinal = 0_u16;
        for tile_number in 0..tile_count {
            let offset = HEADER_SIZE.strict_add(tile_number.strict_mul(12));
            let state_u32 = read_u32(bytes, offset)?;
            let first_u32 = read_u32(bytes, offset.strict_add(4))?;
            let count_u32 = read_u32(bytes, offset.strict_add(8))?;
            let first = first_u32 as usize;
            let count = count_u32 as usize;
            (state_u32 <= 2
                && first == next_page
                && ((state_u32 == 2 && count > 0) || (state_u32 < 2 && count == 0)))
                .ok_or("대한민국 경계 타일 인덱스 오류")?;
            next_page = next_page
                .checked_add(count)
                .ok_or("대한민국 경계 블록 개수 오류")?;
            (next_page <= page_count).ok_or("대한민국 경계 블록 범위 오류")?;
            let tile_state = match state_u32 {
                0 => TILE_OUTSIDE,
                1 => TILE_INSIDE,
                _ => {
                    (next_ordinal < TILE_OUTSIDE).ok_or("대한민국 세부 격자 개수 오류")?;
                    let assigned = next_ordinal;
                    next_ordinal = next_ordinal.strict_add(1);
                    mixed_tiles.push(MixedTile {
                        count: count_u32,
                        first: first_u32,
                    });
                    assigned
                }
            };
            tile_states.push(tile_state);
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
            parts.push(Part {
                first,
                count,
                bytes: length,
                validated: OnceLock::new(),
            });
            next_page = next_page
                .checked_add(count)
                .ok_or("대한민국 경계 조각 개수 오류")?;
        }
        (next_page == page_count).ok_or("대한민국 경계 전체 블록 개수 오류")?;
        (KOREA_TILE_BYTES.len() == 0x8000).ok_or("대한민국 좌표 타일 인덱스 길이 오류")?;
        let mut selected = Vec::with_capacity(KOREA_TILE_COUNT);
        for (word_index, chunk) in KOREA_TILE_BYTES.as_chunks::<8>().0.iter().enumerate() {
            let mut bits = u64::from_le_bytes(*chunk);
            let [w0, w1, w2, w3, ..] = word_index.to_le_bytes();
            let base = u32::from_le_bytes([w0, w1, w2, w3]).strict_mul(64);
            while bits != 0 {
                let [low, mid, high, _] = (base | bits.trailing_zeros()).to_le_bytes();
                selected.push([low, mid, high]);
                bits &= bits.wrapping_sub(1);
            }
        }
        (selected.len() == KOREA_TILE_COUNT).ok_or("대한민국 좌표 타일 개수 오류")?;
        let sampler_tiles = selected.into_boxed_slice();
        Ok(Self {
            area_bounds: BoundaryAreaBounds::try_from(bounds)?,
            atlas: atlas::Atlas::try_from((
                atlas::DATA,
                tile_states.len(),
                mixed_tiles.len(),
                source_hash,
            ))?,
            bounds,
            column_count: u16::try_from(column_count)
                .ok()
                .ok_or("대한민국 경계 격자 열 범위 오류")?,
            divisor,
            fine_divisor: divisor * 16.0_f64,
            fine_inv_divisor: inv_divisor * (1.0_f64 / 16.0_f64),
            grid_south,
            grid_west,
            inv_divisor,
            mixed_tiles: mixed_tiles.into_boxed_slice(),
            row_count: u16::try_from(row_count)
                .ok()
                .ok_or("대한민국 경계 격자 행 범위 오류")?,
            polygon_count,
            ring_count,
            vertex_count,
            source_hash,
            tile_states: tile_states.into_boxed_slice(),
            sampler_tiles,
            page_count,
            parts: parts.into_boxed_slice(),
        })
    }
}
impl BoundaryCache {
    fn page(
        &mut self,
        index: &BoundaryIndex,
        page_number: u32,
        tile_number: usize,
    ) -> Result<DecodedPageRef<'_>> {
        if self.resident.is_empty() {
            self.resident.resize(index.page_count.div_ceil(64), 0);
        }
        let page_index = page_number as usize;
        let word = page_index.div_euclid(64);
        let bit = 1_u64 << page_index.rem_euclid(64);
        let resident = *self
            .resident
            .get(word)
            .ok_or("대한민국 경계 블록 순번 오류")?;
        if resident & bit != 0 {
            if self
                .pages
                .back()
                .is_none_or(|&(number, _)| number != page_number)
            {
                let position = self
                    .pages
                    .iter()
                    .rposition(|&(number, _)| number == page_number)
                    .ok_or("대한민국 경계 캐시 위치 오류")?;
                let item = self
                    .pages
                    .remove(position)
                    .ok_or("대한민국 경계 캐시 위치 오류")?;
                self.pages.push_back(item);
            }
        } else {
            let decoded = index.page(page_index, tile_number, &mut self.scratch)?;
            let additional = decoded.bytes as usize;
            while self.pages.len() >= CACHE_PAGES || self.bytes.strict_add(additional) > CACHE_BYTES
            {
                let (removed_number, removed) = self
                    .pages
                    .pop_front()
                    .ok_or("대한민국 경계 캐시 용량 오류")?;
                let removed_idx = removed_number as usize;
                let removed_word = self
                    .resident
                    .get_mut(removed_idx.div_euclid(64))
                    .ok_or("대한민국 경계 캐시 위치 오류")?;
                *removed_word &= !(1_u64 << removed_idx.rem_euclid(64));
                self.bytes = self.bytes.strict_sub(removed.bytes as usize);
            }
            let inserted_word = self
                .resident
                .get_mut(word)
                .ok_or("대한민국 경계 블록 순번 오류")?;
            *inserted_word |= bit;
            self.bytes = self.bytes.strict_add(additional);
            self.pages.push_back((page_number, decoded));
        }
        let &mut (_, ref mut cached) = self
            .pages
            .back_mut()
            .ok_or("대한민국 경계 캐시 위치 오류")?;
        Ok(match cached.decoded {
            DecodedPage::Parity {
                base,
                ref mut endpoints,
            } => DecodedPageRef::Parity { base, endpoints },
            DecodedPage::Chain {
                first,
                hole,
                owner,
                ring,
                ref mut vertices,
            } => DecodedPageRef::Chain {
                first,
                hole,
                owner,
                ring,
                vertices,
            },
        })
    }
}
impl SampleState<'_> {
    pub(super) fn finish(self) -> Result<Coordinates> {
        Ok(self.point.ok_or("대한민국 좌표 보완 난수 시도 횟수 초과")?)
    }
    pub(super) const fn is_complete(&self) -> bool {
        self.point.is_some()
    }
    pub(super) fn offer_supplement(&mut self, value: u64) -> Result<()> {
        if self.is_complete() {
            return Ok(());
        }
        self.supplemental_reads = self.supplemental_reads.strict_add(1);
        self.try_word(value)?;
        (self.supplemental_reads != KOREA_SUPP_RETRY_LIMIT || self.is_complete())
            .ok_or("대한민국 좌표 보완 난수 시도 횟수 초과")
            .map_err(Into::into)
    }
    fn try_word(&mut self, value: u64) -> Result<()> {
        let selector = low_u32_from_u64(value >> 42_i32);
        if (selector as usize) >= KOREA_TILE_COUNT.strict_mul(53) {
            return Ok(());
        }
        let &[low, mid, high] = self
            .index
            .sampler_tiles
            .get(selector.div_euclid(53) as usize)
            .ok_or("대한민국 좌표 타일 선택 범위 오류")?;
        let tile = u64::from(u32::from_le_bytes([low, mid, high, 0]));
        let vertical = ((tile >> 9_i32) << 23_i32) | (((value >> 21_i32) & 0x1f_ffff) << 2_i32) | 2;
        let horizontal = ((tile & 511) << 23_i32) | ((value & 0x1f_ffff) << 2_i32) | 2;
        let mapped = (vertical << 32_u32) | horizontal;
        let point = Coordinates::try_from((mapped, self.index.area_bounds))?;
        if self.index.covers_sample(&point, self.active_cache)? {
            self.point = Some(point);
        }
        Ok(())
    }
}
impl<'cache> TryFrom<(u64, &'cache mut BoundaryCache)> for SampleState<'cache> {
    type Error = AppError;
    fn try_from((value, active_cache): (u64, &'cache mut BoundaryCache)) -> Result<Self> {
        let index = BOUNDARY
            .as_ref()
            .map_err(|source| AppError::prefixed_ref("대한민국 경계 초기화 실패", source))?;
        let mut state = Self {
            active_cache,
            index,
            point: None,
            supplemental_reads: 0,
        };
        state.try_word(value)?;
        Ok(state)
    }
}
fn axis_edge(origin: f64, inv_divisor: f64, index: u16) -> f64 {
    let offset = f64::from(index) * inv_divisor;
    origin + offset
}
fn grid_axis_index(
    origin: f64,
    divisor: f64,
    inv_divisor: f64,
    max_cell: u16,
    coordinate: f64,
) -> u16 {
    let scaled = ((coordinate - origin) * divisor)
        .floor()
        .clamp(0.0_f64, MAX_GRID_DIMENSION_F64);
    let aligned_bits = (scaled + IEEE_F64_MANTISSA_ALIGN).to_bits() & 0x1FFF_u64;
    let approx = low_u16_from_u64(aligned_bits).min(max_cell);
    if axis_edge(origin, inv_divisor, approx.strict_add(1)) <= coordinate {
        approx.strict_add(1).min(max_cell)
    } else if approx > 0 && axis_edge(origin, inv_divisor, approx) > coordinate {
        approx.strict_sub(1)
    } else {
        approx
    }
}
fn read_chunk<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N]> {
    bytes
        .get(offset..)
        .and_then(<[u8]>::first_chunk::<N>)
        .copied()
        .ok_or_else(|| "대한민국 경계 데이터 구조 오류".into())
}
fn read_count(bytes: &[u8], offset: usize) -> Result<usize> {
    Ok(read_u32(bytes, offset)? as usize)
}
fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    read_chunk::<4>(bytes, offset).map(u32::from_le_bytes)
}
fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    read_chunk::<8>(bytes, offset).map(u64::from_le_bytes)
}
fn read_f64(bytes: &[u8], offset: usize) -> Result<f64> {
    Ok(f64::from_bits(read_u64(bytes, offset)?))
}
