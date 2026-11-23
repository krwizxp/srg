use super::{BoundaryIndex, axis_edge, grid_axis_index, read_count};
use crate::{
    diagnostic::{AppError, Result},
    random_data::Coordinates,
};
pub(super) const DATA: &[u8] = include_bytes!("../../data/korea-boundary-v4/atlas-16.bin");
const HEADER_SIZE: usize = 96;
const INDEX_HASH: [u8; 32] = [
    0x8C, 0xB2, 0x0A, 0x5C, 0x0D, 0xA9, 0xC3, 0xD6, 0x22, 0x5E, 0xFB, 0x03, 0x23, 0xE2, 0x4A, 0x4F,
    0x99, 0x2E, 0xB3, 0x02, 0x9A, 0x77, 0x2D, 0xA3, 0xA7, 0x1A, 0x55, 0xD3, 0xF5, 0x03, 0xBF, 0x17,
];
const SUBDIVISION: usize = 16;
const SUBDIVISION_MAX_CELL: u16 = 15;
pub(super) struct Atlas<'data> {
    states: &'data [u8],
}
impl Atlas<'_> {
    pub(super) fn state(
        &self,
        boundary: &BoundaryIndex,
        point: &Coordinates,
        column: u16,
        row: u16,
        ordinal: u16,
    ) -> Result<u8> {
        let west = axis_edge(boundary.grid_west, boundary.inv_divisor, column);
        let south = axis_edge(boundary.grid_south, boundary.inv_divisor, row);
        let divisor = boundary.fine_divisor;
        let inv_divisor = boundary.fine_inv_divisor;
        let fine_column = usize::from(grid_axis_index(
            west,
            divisor,
            inv_divisor,
            SUBDIVISION_MAX_CELL,
            point.longitude,
        ));
        let fine_row = usize::from(grid_axis_index(
            south,
            divisor,
            inv_divisor,
            SUBDIVISION_MAX_CELL,
            point.latitude,
        ));
        (ordinal != u16::MAX).ok_or("대한민국 세부 격자 소유 범위 오류")?;
        let cell = usize::from(ordinal)
            .strict_mul(SUBDIVISION)
            .strict_add(fine_row)
            .strict_mul(SUBDIVISION)
            .strict_add(fine_column);
        let packed = self
            .states
            .get(cell >> 2_i32)
            .ok_or("대한민국 세부 격자 상태 범위 오류")?;
        Ok((packed >> ((cell & 3_usize).strict_mul(2))) & 3_u8)
    }
}
impl<'data> TryFrom<(&'data [u8], usize, usize, [u8; 32])> for Atlas<'data> {
    type Error = AppError;
    fn try_from(
        (bytes, tile_count, actual_mixed_count, source_hash): (&'data [u8], usize, usize, [u8; 32]),
    ) -> Result<Self> {
        (bytes.len() >= HEADER_SIZE && bytes.starts_with(b"SRGKA001"))
            .ok_or("대한민국 세부 격자 데이터 버전 오류")?;
        let mixed_count = read_count(bytes, 16)?;
        let state_bytes = read_count(bytes, 88)?;
        (read_count(bytes, 8)? == HEADER_SIZE
            && read_count(bytes, 12)? == SUBDIVISION
            && mixed_count > 0
            && mixed_count <= tile_count
            && tile_count <= usize::from(u16::MAX).strict_add(1)
            && read_count(bytes, 20)? == tile_count
            && state_bytes == mixed_count.strict_mul(64)
            && bytes.len() == HEADER_SIZE.strict_add(state_bytes)
            && read_count(bytes, 92)? == 0
            && bytes.get(24..56) == Some(INDEX_HASH.as_slice())
            && bytes.get(56..88) == Some(source_hash.as_slice()))
        .ok_or("대한민국 세부 격자 인덱스 오류")?;
        let states = bytes
            .get(HEADER_SIZE..)
            .ok_or("대한민국 세부 격자 상태 누락")?;
        states
            .as_chunks::<8>()
            .0
            .iter()
            .all(|chunk| {
                let word = u64::from_le_bytes(*chunk);
                word & (word >> 1_u32) & 0x5555_5555_5555_5555_u64 == 0
            })
            .ok_or("대한민국 세부 격자 상태 오류")?;
        (actual_mixed_count == mixed_count).ok_or("대한민국 세부 격자 개수 오류")?;
        Ok(Self { states })
    }
}
