use super::numeric::low_u8_from_u64;
use crate::constants::{
    GLYPHS, NIBBLE_MASK_U64, ONE_BASED_OFFSET_U8, ONE_BASED_OFFSET_U64, ONE_BASED_OFFSET_USIZE,
};
use crate::diagnostic::Result;
pub fn checked_add_one_u64(value: u64, err_msg: &'static str) -> Result<u64> {
    let Some(next_value) = value.checked_add(ONE_BASED_OFFSET_U64) else {
        return Err(err_msg.into());
    };
    Ok(next_value)
}
pub const fn checked_add_one_u8(value: u8) -> Option<u8> {
    value.checked_add(ONE_BASED_OFFSET_U8)
}
pub const fn checked_add_one_usize(value: usize) -> Option<usize> {
    value.checked_add(ONE_BASED_OFFSET_USIZE)
}
pub fn glyph_from_low_nibble(value: u64) -> Option<char> {
    GLYPHS
        .get(usize::from(low_u8_from_u64(value & NIBBLE_MASK_U64)))
        .copied()
}
pub fn galaxy_coord<const SUB: u16, const ADD: u16>(value: u16) -> Result<u16> {
    if let Some(lower_bound) = value.checked_sub(SUB) {
        return Ok(lower_bound);
    }
    let Some(coord) = value.checked_add(ADD) else {
        return Err("NMS 은하 좌표 계산 실패".into());
    };
    Ok(coord)
}
