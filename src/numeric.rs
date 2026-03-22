#![expect(
    clippy::as_conversions,
    reason = "these helpers intentionally extract low-order bits with truncating casts"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "these helpers intentionally extract low-order bits with truncating casts"
)]

pub const fn low_u8_from_u32(value: u32) -> u8 {
    value as u8
}
pub const fn low_u8_from_u64(value: u64) -> u8 {
    value as u8
}
pub const fn low_u8_from_u128(value: u128) -> u8 {
    value as u8
}
pub const fn low_u16_from_u64(value: u64) -> u16 {
    value as u16
}
