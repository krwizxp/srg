use super::numeric::low_u8_from_u64;
use super::{
    GLYPHS, NIBBLE_MASK_U64, ONE_BASED_OFFSET_U8, ONE_BASED_OFFSET_U64, ONE_BASED_OFFSET_USIZE,
    Result,
};
pub fn checked_add_one_u64(value: u64, err_msg: &'static str) -> Result<u64> {
    value
        .checked_add(ONE_BASED_OFFSET_U64)
        .ok_or_else(|| err_msg.into())
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
pub const fn split_u64_to_u32_pair(value: u64) -> (u32, u32) {
    let [b0, b1, b2, b3, b4, b5, b6, b7] = value.to_be_bytes();
    (
        u32::from_be_bytes([b0, b1, b2, b3]),
        u32::from_be_bytes([b4, b5, b6, b7]),
    )
}
pub const fn galaxy_coord<const SUB: u16, const ADD: u16>(value: u16) -> u16 {
    let lower_bound = value.wrapping_sub(SUB);
    let upper_bound = value.wrapping_add(ADD);
    if lower_bound < upper_bound {
        lower_bound
    } else {
        upper_bound
    }
}
