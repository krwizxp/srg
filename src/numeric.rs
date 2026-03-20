#[expect(
    clippy::cast_possible_truncation,
    reason = "this helper intentionally extracts the low 8 bits"
)]
pub const fn low_u8_from_u32(value: u32) -> u8 {
    value as u8
}
#[expect(
    clippy::cast_possible_truncation,
    reason = "this helper intentionally extracts the low 8 bits"
)]
pub const fn low_u8_from_u64(value: u64) -> u8 {
    value as u8
}
#[expect(
    clippy::cast_possible_truncation,
    reason = "this helper intentionally extracts the low 8 bits"
)]
pub const fn low_u8_from_u128(value: u128) -> u8 {
    value as u8
}
#[expect(
    clippy::cast_possible_truncation,
    reason = "this helper intentionally extracts the low 16 bits"
)]
pub const fn low_u16_from_u64(value: u64) -> u16 {
    value as u16
}
