use crate::{diagnostic::Result, geo_uniform::orientation};
const KOREA_TILE_COUNT: usize = 79_113;
const U32_MAX_INV: f64 = 1.0 / 4_294_967_295.0;
#[derive(Default)]
pub(super) struct Coordinates {
    pub latitude: f64,
    pub longitude: f64,
}
pub(super) fn world_coordinates(value: u64) -> Coordinates {
    let [b0, b1, b2, b3, b4, b5, b6, b7] = value.to_be_bytes();
    let upper_ratio = f64::from(u32::from_be_bytes([b0, b1, b2, b3])).algebraic_mul(U32_MAX_INV);
    let lower_ratio = f64::from(u32::from_be_bytes([b4, b5, b6, b7])).algebraic_mul(U32_MAX_INV);
    Coordinates {
        latitude: 180.0_f64.mul_add(upper_ratio, -90.0),
        longitude: 360.0_f64.mul_add(lower_ratio, -180.0),
    }
}
pub(super) fn decode_tiles(bytes: &[u8]) -> Result<Box<[u32]>> {
    (bytes.len() == 32_768).ok_or("대한민국 좌표 타일 인덱스 길이 오류")?;
    let mut selected = Vec::with_capacity(KOREA_TILE_COUNT);
    for (index, byte) in bytes.iter().enumerate() {
        for bit in 0..8_usize {
            if byte & (1_u8 << bit) != 0 {
                selected.push(
                    u32::try_from(index.strict_mul(8).strict_add(bit))
                        .map_err(|_| "대한민국 타일 번호 오류")?,
                );
            }
        }
    }
    (selected.len() == KOREA_TILE_COUNT).ok_or("대한민국 좌표 타일 개수 오류")?;
    Ok(selected.into_boxed_slice())
}
pub(super) fn remap_tile(value: u64, tiles: &[u32]) -> Result<Option<u64>> {
    let selector = usize::try_from(value >> 46).map_err(|_| "대한민국 타일 선택값 오류")?;
    if selector >= KOREA_TILE_COUNT.strict_mul(2) {
        return Ok(None);
    }
    let tile = u64::from(
        *tiles
            .get(selector / 2)
            .ok_or("대한민국 좌표 타일 선택 범위 오류")?,
    );
    let vertical = ((tile >> 9) << 23) | ((value >> 23) & 0x7f_ffff);
    let horizontal = ((tile & 511) << 23) | (value & 0x7f_ffff);
    Ok(Some((vertical << 32) | horizontal))
}
pub(super) fn parity_relation<T>(
    endpoints: &[T],
    base: bool,
    latitude: f64,
    read: impl Fn(&T) -> f64,
) -> bool {
    base ^ !endpoints
        .partition_point(|endpoint| read(endpoint) <= latitude)
        .is_multiple_of(2)
}
pub(super) fn chain_relation<T>(
    vertices: &[T],
    point: &Coordinates,
    read: impl Fn(&T) -> [f64; 2],
) -> Result<(bool, bool)> {
    let mut parity = false;
    let mut boundary = false;
    for pair in vertices.windows(2) {
        let [first, last] = pair else {
            return Err("대한민국 경계 선분 오류".into());
        };
        let start = read(first);
        let end = read(last);
        let within = point.longitude >= start[0].min(end[0])
            && point.longitude <= start[0].max(end[0])
            && point.latitude >= start[1].min(end[1])
            && point.latitude <= start[1].max(end[1]);
        let crosses = (start[1] > point.latitude) != (end[1] > point.latitude);
        if within || crosses {
            let side = orientation(start, end, [point.longitude, point.latitude]);
            boundary |= within && side == 0.0_f64;
            parity ^= crosses
                && if end[1] > start[1] {
                    side > 0.0_f64
                } else {
                    side < 0.0_f64
                };
        }
    }
    Ok((parity, boundary))
}
