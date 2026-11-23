use crate::{
    boundary_inflate::InflatedPage,
    diagnostic::{AppError, Result, or_abort},
    geo_uniform::BoundaryAreaBounds,
    numeric::{low_u16_from_u64, low_u32_from_u64},
    random_data::Coordinates,
};
use core::{array, cmp::Ordering};
use std::sync::{LazyLock, OnceLock};
macro_rules! boundary_parts {
    ([$($ten:literal),*]; [$($tail:literal),*]) => {
        [
            $(
                include_bytes!(concat!("../data/korea-boundary-v4/part-0", $ten, "0.bin")),
                include_bytes!(concat!("../data/korea-boundary-v4/part-0", $ten, "1.bin")),
                include_bytes!(concat!("../data/korea-boundary-v4/part-0", $ten, "2.bin")),
                include_bytes!(concat!("../data/korea-boundary-v4/part-0", $ten, "3.bin")),
                include_bytes!(concat!("../data/korea-boundary-v4/part-0", $ten, "4.bin")),
                include_bytes!(concat!("../data/korea-boundary-v4/part-0", $ten, "5.bin")),
                include_bytes!(concat!("../data/korea-boundary-v4/part-0", $ten, "6.bin")),
                include_bytes!(concat!("../data/korea-boundary-v4/part-0", $ten, "7.bin")),
                include_bytes!(concat!("../data/korea-boundary-v4/part-0", $ten, "8.bin")),
                include_bytes!(concat!("../data/korea-boundary-v4/part-0", $ten, "9.bin")),
            )*
            $(include_bytes!(concat!("../data/korea-boundary-v4/part-09", $tail, ".bin")),)*
        ]
    };
}
const ATLAS_DATA: &[u8] = include_bytes!("../data/korea-boundary-v4/atlas-16.bin");
const ATLAS_INDEX_HASH: [u8; 32] = [
    0x8C, 0xB2, 0x0A, 0x5C, 0x0D, 0xA9, 0xC3, 0xD6, 0x22, 0x5E, 0xFB, 0x03, 0x23, 0xE2, 0x4A, 0x4F,
    0x99, 0x2E, 0xB3, 0x02, 0x9A, 0x77, 0x2D, 0xA3, 0xA7, 0x1A, 0x55, 0xD3, 0xF5, 0x03, 0xBF, 0x17,
];
static BOUNDARY: LazyLock<Result<BoundaryIndex>> =
    LazyLock::new(|| BoundaryIndex::try_from((DATA, PARTS.len())));
const CACHE_BYTES: usize = 896 * 1024;
const CACHE_PAGES: usize = 4096;
static DATA: &[u8] = include_bytes!("../data/korea-boundary-v4/index.bin");
const HEADER_SIZE: usize = 128;
const KOREA_TILE_BYTES: &[u8] = include_bytes!("../data/korea-boundary-v4/tile-select-v5.bin");
const KOREA_TILE_COUNT: usize = 79113;
static PARTS: [&[u8]; 96] = boundary_parts!([0, 1, 2, 3, 4, 5, 6, 7, 8]; [0, 1, 2, 3, 4, 5]);
const TILE_INSIDE: u16 = u16::MAX;
const TILE_OUTSIDE: u16 = u16::MAX - 1;
#[derive(Default)]
pub(super) struct BoundaryCache {
    bytes: usize,
    hand: usize,
    pages: Vec<(u32, bool, CachedPage)>,
    scratch: Vec<u8>,
    slots: Vec<u16>,
}
struct BoundaryIndex {
    area_bounds: BoundaryAreaBounds,
    atlas_states: &'static [u8],
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
struct CachedPage {
    bytes: u32,
    decoded: DecodedPage,
}
enum DecodedPage {
    Chain(u32, u32, u32, u32, Box<[[f64; 2]]>),
    Parity(bool, Box<[f64]>),
}
#[derive(Clone, Copy)]
enum DecodedPageRef<'page> {
    Chain(u32, u32, u32, u32, &'page [[f64; 2]]),
    Parity(bool, &'page [f64]),
}
struct ExactOrient2d([f64; 6]);
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
pub(super) struct SampleState<'cache> {
    active_cache: &'cache mut BoundaryCache,
    index: &'static BoundaryIndex,
    point: Option<Coordinates>,
    supplemental_reads: usize,
}
impl BoundaryCache {
    fn page(
        &mut self,
        index: &BoundaryIndex,
        page_num: u32,
        tile_num: usize,
    ) -> Result<DecodedPageRef<'_>> {
        if self.slots.is_empty() {
            self.slots.resize(index.page_count, 0);
        }
        let page_idx = page_num as usize;
        let existing = *self
            .slots
            .get(page_idx)
            .ok_or("대한민국 경계 블록 순번 오류")?;
        let slot_idx = if existing != 0 {
            let idx = usize::from(existing.strict_sub(1));
            let &mut (_, ref mut used, _) = self
                .pages
                .get_mut(idx)
                .ok_or("대한민국 경계 캐시 위치 오류")?;
            *used = true;
            idx
        } else {
            let decoded = index.page(page_idx, tile_num, &mut self.scratch)?;
            let add = decoded.bytes as usize;
            while self.pages.len() >= CACHE_PAGES || self.bytes.strict_add(add) > CACHE_BYTES {
                if self.hand >= self.pages.len() {
                    self.hand = 0;
                }
                let evict = loop {
                    let &mut (_, ref mut used, _) = self
                        .pages
                        .get_mut(self.hand)
                        .ok_or("대한민국 경계 캐시 용량 오류")?;
                    if *used {
                        *used = false;
                        self.hand = self.hand.strict_add(1);
                        if self.hand >= self.pages.len() {
                            self.hand = 0;
                        }
                    } else {
                        break self.hand;
                    }
                };
                let (rm_num, _, rm) = self.pages.swap_remove(evict);
                *self
                    .slots
                    .get_mut(rm_num as usize)
                    .ok_or("대한민국 경계 캐시 위치 오류")? = 0;
                self.bytes = self.bytes.strict_sub(rm.bytes as usize);
                if let Some(&(moved_num, _, _)) = self.pages.get(evict) {
                    let [s0, s1, ..] = evict.strict_add(1).to_le_bytes();
                    *self
                        .slots
                        .get_mut(moved_num as usize)
                        .ok_or("대한민국 경계 캐시 위치 오류")? = u16::from_le_bytes([s0, s1]);
                }
            }
            let idx = self.pages.len();
            let [s0, s1, ..] = idx.strict_add(1).to_le_bytes();
            *self
                .slots
                .get_mut(page_idx)
                .ok_or("대한민국 경계 블록 순번 오류")? = u16::from_le_bytes([s0, s1]);
            self.bytes = self.bytes.strict_add(add);
            self.pages.push((page_num, true, decoded));
            idx
        };
        let &mut (_, _, ref mut cached) = self
            .pages
            .get_mut(slot_idx)
            .ok_or("대한민국 경계 캐시 위치 오류")?;
        Ok(match cached.decoded {
            DecodedPage::Parity(base, ref mut eps) => DecodedPageRef::Parity(base, eps),
            DecodedPage::Chain(f0, hole, owner, ring, ref mut verts) => {
                DecodedPageRef::Chain(f0, hole, owner, ring, verts)
            }
        })
    }
}
impl BoundaryIndex {
    fn covers_sample(&self, point: &Coordinates, cache: &mut BoundaryCache) -> Result<bool> {
        let col = grid_axis_index(
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
        let tile_num = usize::from(row)
            .strict_mul(usize::from(self.column_count))
            .strict_add(usize::from(col));
        let tile_state = *self
            .tile_states
            .get(tile_num)
            .ok_or("대한민국 경계 타일 선택 오류")?;
        if tile_state >= TILE_OUTSIDE {
            return Ok(tile_state == TILE_INSIDE);
        }
        let fine_col = usize::from(grid_axis_index(
            axis_edge(self.grid_west, self.inv_divisor, col),
            self.fine_divisor,
            self.fine_inv_divisor,
            15,
            point.longitude,
        ));
        let fine_row = usize::from(grid_axis_index(
            axis_edge(self.grid_south, self.inv_divisor, row),
            self.fine_divisor,
            self.fine_inv_divisor,
            15,
            point.latitude,
        ));
        let cell = usize::from(tile_state)
            .strict_mul(16)
            .strict_add(fine_row)
            .strict_mul(16)
            .strict_add(fine_col);
        let packed = *self
            .atlas_states
            .get(cell >> 2_i32)
            .ok_or("대한민국 세부 격자 상태 범위 오류")?;
        match (packed >> ((cell & 3_usize).strict_mul(2))) & 3_u8 {
            0 => return Ok(false),
            1 => return Ok(true),
            _ => {}
        }
        let tile = self
            .mixed_tiles
            .get(usize::from(tile_state))
            .ok_or("대한민국 경계 혼합 타일 선택 오류")?;
        let (mut parity, mut boundary, mut chains_started) = (false, false, false);
        let (mut last_endpoint, mut last_chain) = (None, None);
        for offset in 0..tile.count {
            match cache.page(self, tile.first.strict_add(offset), tile_num)? {
                DecodedPageRef::Parity(base, endpoints) => {
                    (!chains_started
                        && (offset == 0 || !base)
                        && last_endpoint
                            .zip(endpoints.first())
                            .is_none_or(|(prev, next)| prev < *next))
                    .ok_or("대한민국 경계 교차 블록 순서 오류")?;
                    parity ^= base
                        ^ !endpoints
                            .partition_point(|edge| *edge <= point.latitude)
                            .is_multiple_of(2);
                    if let Some(&last) = endpoints.last() {
                        last_endpoint = Some(last);
                    }
                }
                DecodedPageRef::Chain(first, hole, owner, ring, vertices) => {
                    if let Some((pr, po, ph, pe)) = last_chain {
                        (ring > pr || (ring == pr && owner == po && hole == ph && first >= pe))
                            .ok_or("대한민국 경계 링 순서 오류")?;
                    }
                    let [b0, b1, b2, b3, ..] = vertices.len().strict_sub(1).to_le_bytes();
                    last_chain = Some((
                        ring,
                        owner,
                        hole,
                        first.strict_add(u32::from_le_bytes([b0, b1, b2, b3])),
                    ));
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
                        let (dx, dy) = (end[0] - start[0], end[1] - start[1]);
                        let (px, py) = (point.longitude - start[0], point.latitude - start[1]);
                        let (left, right) = (dx * py, dy * px);
                        let approx = left - right;
                        let det: f64 = if approx.abs()
                            > 3.330_669_073_875_471_6e-16_f64 * (left.abs() + right.abs())
                        {
                            approx
                        } else {
                            ExactOrient2d([dx, dy, px, py, left, right]).det()?
                        };
                        let side = det.partial_cmp(&0.0_f64);
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
        page_num: usize,
        tile_num: usize,
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
        let len = raw.len();
        (len == raw_size
            && raw.starts_with(b"SRGKB004")
            && read_count(raw, 8)? == tile_num
            && read_count(raw, 12)? == page_num)
            .ok_or("대한민국 경계 블록 버전·길이·번호 오류")?;
        let (cnt, fa, fb) = (
            read_count(raw, 20)?,
            read_count(raw, 24)?,
            read_count(raw, 28)?,
        );
        match read_u32(raw, 16)? {
            0 => {
                (cnt <= 4092
                    && len == 32_usize.strict_add(cnt.strict_mul(8))
                    && fa <= 1
                    && fb == 0)
                    .ok_or("대한민국 경계 교차 인덱스 오류")?;
                let row = u16::try_from(tile_num.div_euclid(usize::from(self.column_count)))
                    .ok()
                    .filter(|&val| val < self.row_count)
                    .ok_or("대한민국 타일 행 오류")?;
                let mut prev = axis_edge(self.grid_south, self.inv_divisor, row);
                let north = axis_edge(self.grid_south, self.inv_divisor, row.strict_add(1));
                let mut endpoints = Vec::with_capacity(cnt);
                for chunk in raw
                    .get(32..)
                    .ok_or("대한민국 경계 교차 길이 오류")?
                    .as_chunks::<8>()
                    .0
                {
                    let val = f64::from_le_bytes(*chunk);
                    (val.is_finite() && val > prev && val <= north)
                        .ok_or("대한민국 경계 교차 범위 오류")?;
                    endpoints.push(val);
                    prev = val;
                }
                let [b0, b1, b2, b3, ..] = endpoints
                    .len()
                    .strict_mul(8)
                    .strict_add(size_of::<(u32, CachedPage)>())
                    .to_le_bytes();
                Ok(CachedPage {
                    bytes: u32::from_le_bytes([b0, b1, b2, b3]),
                    decoded: DecodedPage::Parity(fa == 1, endpoints.into_boxed_slice()),
                })
            }
            1 => {
                let (first_u32, hole) = (read_u32(raw, 32)?, read_u32(raw, 36)?);
                ((1..=1024).contains(&cnt)
                    && fa < self.ring_count
                    && fb < self.polygon_count
                    && (first_u32 as usize)
                        .checked_add(cnt)
                        .is_some_and(|end| end < self.vertex_count)
                    && hole <= 1
                    && len == 40_usize.strict_add(cnt.strict_add(1).strict_mul(16)))
                .ok_or("대한민국 경계 링 인덱스 오류")?;
                let words = cnt.strict_add(1).strict_mul(2);
                let mut piter = raw
                    .get(40..)
                    .ok_or("대한민국 경계 정점 길이 오류")?
                    .chunks_exact(words);
                let [p0, p1, p2, p3, p4, p5, p6, p7] =
                    array::from_fn(|_| or_abort(piter.next()).as_chunks::<2>().0);
                let mut vertices = Vec::with_capacity(cnt.strict_add(1));
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
                let [r0, r1, r2, r3, ..] = fa.to_le_bytes();
                let [o0, o1, o2, o3, ..] = fb.to_le_bytes();
                Ok(CachedPage {
                    bytes: u32::from_le_bytes([b0, b1, b2, b3]),
                    decoded: DecodedPage::Chain(
                        first_u32,
                        hole,
                        u32::from_le_bytes([o0, o1, o2, o3]),
                        u32::from_le_bytes([r0, r1, r2, r3]),
                        vertices.into_boxed_slice(),
                    ),
                })
            }
            _ => Err("대한민국 경계 블록 종류 오류".into()),
        }
    }
    fn page(&self, page_num: usize, tile_num: usize, scratch: &mut Vec<u8>) -> Result<CachedPage> {
        let part_num = self
            .parts
            .partition_point(|part| part.first <= page_num)
            .checked_sub(1)
            .ok_or("대한민국 경계 조각 선택 오류")?;
        let part = self
            .parts
            .get(part_num)
            .ok_or("대한민국 경계 조각 선택 오류")?;
        let bytes = *PARTS.get(part_num).ok_or("대한민국 경계 내장 조각 누락")?;
        part.validated
            .get_or_init(|| self.validate_part(part, bytes))
            .as_ref()
            .map_err(|src| AppError::prefixed_ref("대한민국 경계 조각 검사 실패", src))?;
        let local = page_num
            .checked_sub(part.first)
            .filter(|&loc| loc < part.count)
            .ok_or("대한민국 경계 블록 순번 오류")?;
        let ent = 64_usize.strict_add(local.strict_mul(12));
        let (start, size, raw_size) = (
            read_count(bytes, ent)?,
            read_count(bytes, ent.strict_add(4))?,
            read_count(bytes, ent.strict_add(8))?,
        );
        let compressed = bytes
            .get(start..start.strict_add(size))
            .ok_or("대한민국 경계 블록 위치 오류")?;
        self.decode_page(page_num, tile_num, compressed, raw_size, scratch)
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
            && u64::from_le_bytes(read_chunk(bytes, 56)?) == 0)
            .ok_or("대한민국 경계 조각 버전·길이 오류")?;
        let mut exp = 64_usize.strict_add(part.count.strict_mul(12));
        for local in 0..part.count {
            let ent = 64_usize.strict_add(local.strict_mul(12));
            let (start, size, raw_size) = (
                read_count(bytes, ent)?,
                read_count(bytes, ent.strict_add(4))?,
                read_count(bytes, ent.strict_add(8))?,
            );
            (start == exp && size > 0 && (32..=0x8000).contains(&raw_size))
                .ok_or("대한민국 경계 압축 블록 인덱스 오류")?;
            exp = start
                .checked_add(size)
                .filter(|&end| end <= bytes.len())
                .ok_or("대한민국 경계 압축 블록 길이 오류")?;
        }
        (exp == bytes.len()).ok_or("대한민국 경계 조각 끝 위치 오류")?;
        Ok(())
    }
}
impl TryFrom<(&[u8], usize)> for BoundaryIndex {
    type Error = AppError;
    fn try_from((bytes, expected_parts): (&[u8], usize)) -> Result<Self> {
        (bytes.len() >= HEADER_SIZE && bytes.starts_with(b"SRGKR004"))
            .ok_or("대한민국 경계 데이터 버전 오류")?;
        let (tile_cnt, part_cnt, page_cnt) = (
            read_count(bytes, 12)?,
            read_count(bytes, 16)?,
            read_count(bytes, 20)?,
        );
        let (poly_cnt, ring_cnt, vert_cnt) = (
            read_count(bytes, 24)?,
            read_count(bytes, 28)?,
            read_count(bytes, 32)?,
        );
        let (cols, rows, scale) = (
            read_u32(bytes, 36)?,
            read_u32(bytes, 40)?,
            read_u32(bytes, 44)?,
        );
        (read_count(bytes, 8)? == HEADER_SIZE
            && (1..=0x0004_0000).contains(&tile_cnt)
            && (1..=4096).contains(&part_cnt)
            && part_cnt == expected_parts
            && (1..=1_000_000).contains(&page_cnt)
            && poly_cnt > 0
            && poly_cnt <= ring_cnt
            && ring_cnt < vert_cnt
            && vert_cnt <= 20_000_000
            && (1..=4096).contains(&cols)
            && (1..=4096).contains(&rows)
            && usize::try_from(cols.strict_mul(rows)).ok() == Some(tile_cnt)
            && scale.is_power_of_two()
            && scale <= 256
            && bytes.len()
                == HEADER_SIZE
                    .strict_add(tile_cnt.strict_mul(12))
                    .strict_add(part_cnt.strict_mul(48))
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
        let (grid_west, grid_south, divisor) =
            (read_f64(bytes, 48)?, read_f64(bytes, 56)?, f64::from(scale));
        let inv_divisor = 1.0_f64 / divisor;
        let (grid_east, grid_north) = (
            grid_west + f64::from(cols) / divisor,
            grid_south + f64::from(rows) / divisor,
        );
        (grid_west.is_finite()
            && grid_south.is_finite()
            && (grid_west * divisor).fract() == 0.0_f64
            && (grid_south * divisor).fract() == 0.0_f64
            && grid_west <= west
            && west < grid_west + inv_divisor
            && grid_south <= south
            && south < grid_south + inv_divisor
            && east <= grid_east
            && east > grid_east - inv_divisor
            && north <= grid_north
            && north > grid_north - inv_divisor)
            .ok_or("대한민국 경계 타일 범위 오류")?;
        let mut source_hash = [0_u8; 32];
        source_hash.copy_from_slice(bytes.get(96..128).ok_or("대한민국 경계 원본 식별자 오류")?);
        let (mut tile_states, mut mixed_tiles) = (Vec::with_capacity(tile_cnt), Vec::new());
        let (mut next_page, mut next_ord) = (0_usize, 0_u16);
        for tn in 0..tile_cnt {
            let off = HEADER_SIZE.strict_add(tn.strict_mul(12));
            let (st_u32, first_u32, cnt_u32) = (
                read_u32(bytes, off)?,
                read_u32(bytes, off.strict_add(4))?,
                read_u32(bytes, off.strict_add(8))?,
            );
            let cnt = cnt_u32 as usize;
            (st_u32 <= 2
                && (first_u32 as usize) == next_page
                && ((st_u32 == 2 && cnt > 0) || (st_u32 < 2 && cnt == 0)))
                .ok_or("대한민국 경계 타일 인덱스 오류")?;
            next_page = next_page
                .checked_add(cnt)
                .filter(|&np| np <= page_cnt)
                .ok_or("대한민국 경계 블록 개수 오류")?;
            tile_states.push(match st_u32 {
                0 => TILE_OUTSIDE,
                1 => TILE_INSIDE,
                _ => {
                    (next_ord < TILE_OUTSIDE).ok_or("대한민국 세부 격자 개수 오류")?;
                    let ord = next_ord;
                    next_ord = next_ord.strict_add(1);
                    mixed_tiles.push(MixedTile {
                        count: cnt_u32,
                        first: first_u32,
                    });
                    ord
                }
            });
        }
        (next_page == page_cnt).ok_or("대한민국 경계 타일 블록 개수 오류")?;
        next_page = 0;
        let mut parts = Vec::with_capacity(part_cnt);
        let part_start = HEADER_SIZE.strict_add(tile_cnt.strict_mul(12));
        for pn in 0..part_cnt {
            let off = part_start.strict_add(pn.strict_mul(48));
            let (first, count, length) = (
                read_count(bytes, off)?,
                read_count(bytes, off.strict_add(4))?,
                read_count(bytes, off.strict_add(8))?,
            );
            (first == next_page
                && (1..=page_cnt).contains(&count)
                && read_u32(bytes, off.strict_add(12))? == 0
                && length > 64_usize.strict_add(count.strict_mul(12))
                && length <= 0x0008_0000)
                .ok_or("대한민국 경계 조각 인덱스 오류")?;
            parts.push(Part {
                bytes: length,
                count,
                first,
                validated: OnceLock::new(),
            });
            next_page = next_page
                .checked_add(count)
                .ok_or("대한민국 경계 조각 개수 오류")?;
        }
        (next_page == page_cnt).ok_or("대한민국 경계 전체 블록 개수 오류")?;
        (ATLAS_DATA.len() >= 96 && ATLAS_DATA.starts_with(b"SRGKA001"))
            .ok_or("대한민국 세부 격자 데이터 버전 오류")?;
        let (atlas_mixed, state_bytes) = (read_count(ATLAS_DATA, 16)?, read_count(ATLAS_DATA, 88)?);
        (read_count(ATLAS_DATA, 8)? == 96
            && read_count(ATLAS_DATA, 12)? == 16
            && atlas_mixed > 0
            && atlas_mixed == mixed_tiles.len()
            && tile_cnt <= usize::from(u16::MAX).strict_add(1)
            && read_count(ATLAS_DATA, 20)? == tile_cnt
            && state_bytes == atlas_mixed.strict_mul(64)
            && ATLAS_DATA.len() == 96_usize.strict_add(state_bytes)
            && read_count(ATLAS_DATA, 92)? == 0
            && ATLAS_DATA.get(24..56) == Some(ATLAS_INDEX_HASH.as_slice())
            && ATLAS_DATA.get(56..88) == Some(source_hash.as_slice()))
        .ok_or("대한민국 세부 격자 인덱스 오류")?;
        let atlas_states = ATLAS_DATA.get(96..).ok_or("대한민국 세부 격자 상태 누락")?;
        atlas_states
            .as_chunks::<8>()
            .0
            .iter()
            .all(|chunk| {
                let word = u64::from_le_bytes(*chunk);
                word & (word >> 1_u32) & 0x5555_5555_5555_5555_u64 == 0
            })
            .ok_or("대한민국 세부 격자 상태 오류")?;
        (KOREA_TILE_BYTES.len() == 0x8000).ok_or("대한민국 좌표 타일 인덱스 길이 오류")?;
        let mut selected = Vec::with_capacity(KOREA_TILE_COUNT);
        for (wi, chunk) in KOREA_TILE_BYTES.as_chunks::<8>().0.iter().enumerate() {
            let mut bits = u64::from_le_bytes(*chunk);
            let [w0, w1, w2, w3, ..] = wi.to_le_bytes();
            let base = u32::from_le_bytes([w0, w1, w2, w3]).strict_mul(64);
            while bits != 0 {
                let [lo, mid, hi, _] = (base | bits.trailing_zeros()).to_le_bytes();
                selected.push([lo, mid, hi]);
                bits &= bits.wrapping_sub(1);
            }
        }
        (selected.len() == KOREA_TILE_COUNT).ok_or("대한민국 좌표 타일 개수 오류")?;
        Ok(Self {
            area_bounds: BoundaryAreaBounds::try_from(bounds)?,
            atlas_states,
            bounds,
            column_count: u16::try_from(cols)
                .ok()
                .ok_or("대한민국 경계 격자 열 범위 오류")?,
            divisor,
            fine_divisor: divisor * 16.0_f64,
            fine_inv_divisor: inv_divisor * (1.0_f64 / 16.0_f64),
            grid_south,
            grid_west,
            inv_divisor,
            mixed_tiles: mixed_tiles.into_boxed_slice(),
            page_count: page_cnt,
            parts: parts.into_boxed_slice(),
            polygon_count: poly_cnt,
            ring_count: ring_cnt,
            row_count: u16::try_from(rows)
                .ok()
                .ok_or("대한민국 경계 격자 행 범위 오류")?,
            sampler_tiles: selected.into_boxed_slice(),
            source_hash,
            tile_states: tile_states.into_boxed_slice(),
            vertex_count: vert_cnt,
        })
    }
}
impl ExactOrient2d {
    #[cold]
    #[inline(never)]
    fn det(self) -> Result<f64> {
        let [dx, dy, px, py, left, right] = self.0;
        let (mut exp, mut len) = ([0.0_f64; 4], 0_usize);
        for mut val in [dx.mul_add(py, -left), left, -dy.mul_add(px, -right), -right] {
            let (mut cnt, prev) = (0_usize, exp);
            for entry in prev.into_iter().take(len) {
                let total = val + entry;
                let vval = total - val;
                let err = (val - (total - vval)) + (entry - vval);
                if err != 0.0_f64 {
                    *exp.get_mut(cnt)
                        .ok_or("대한민국 경계 좌표 누산 범위 오류")? = err;
                    cnt = cnt.strict_add(1);
                }
                val = total;
            }
            if val != 0.0_f64 || cnt == 0 {
                *exp.get_mut(cnt)
                    .ok_or("대한민국 경계 좌표 누산 범위 오류")? = val;
                cnt = cnt.strict_add(1);
            }
            len = cnt;
        }
        Ok(exp.iter().take(len).sum())
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
        (self.supplemental_reads != 1024 || self.is_complete())
            .ok_or("대한민국 좌표 보완 난수 시도 횟수 초과")
            .map_err(Into::into)
    }
    fn try_word(&mut self, value: u64) -> Result<()> {
        let selector = low_u32_from_u64(value >> 42_i32);
        if (selector as usize) >= KOREA_TILE_COUNT.strict_mul(53) {
            return Ok(());
        }
        let &[lo, mid, hi] = self
            .index
            .sampler_tiles
            .get(selector.div_euclid(53) as usize)
            .ok_or("대한민국 좌표 타일 선택 범위 오류")?;
        let tile = u64::from(u32::from_le_bytes([lo, mid, hi, 0]));
        let vert = ((tile >> 9_i32) << 23_i32) | (((value >> 21_i32) & 0x1f_ffff) << 2_i32) | 2;
        let horiz = ((tile & 511) << 23_i32) | ((value & 0x1f_ffff) << 2_i32) | 2;
        let point = Coordinates::try_from(((vert << 32_u32) | horiz, self.index.area_bounds))?;
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
            .map_err(|src| AppError::prefixed_ref("대한민국 경계 초기화 실패", src))?;
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
fn grid_axis_index(origin: f64, divisor: f64, inv_divisor: f64, max_cell: u16, coord: f64) -> u16 {
    let scaled = ((coord - origin) * divisor)
        .floor()
        .clamp(0.0_f64, 4096.0_f64);
    let approx = low_u16_from_u64((scaled + 4_503_599_627_370_496.0_f64).to_bits() & 0x1FFF_u64)
        .min(max_cell);
    if axis_edge(origin, inv_divisor, approx.strict_add(1)) <= coord {
        approx.strict_add(1).min(max_cell)
    } else if approx > 0 && axis_edge(origin, inv_divisor, approx) > coord {
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
fn read_f64(bytes: &[u8], offset: usize) -> Result<f64> {
    Ok(f64::from_le_bytes(read_chunk(bytes, offset)?))
}
fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(read_chunk(bytes, offset)?))
}
