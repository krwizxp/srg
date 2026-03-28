pub const fn low_u8_from_u32(value: u32) -> u8 {
    value.to_le_bytes()[0]
}
pub const fn low_u8_from_u64(value: u64) -> u8 {
    value.to_le_bytes()[0]
}
pub const fn low_u8_from_u128(value: u128) -> u8 {
    value.to_le_bytes()[0]
}
pub const fn low_u16_from_u64(value: u64) -> u16 {
    let bytes = value.to_le_bytes();
    u16::from_le_bytes([bytes[0], bytes[1]])
}
