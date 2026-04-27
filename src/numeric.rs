pub const fn low_u8_from_u32(value: u32) -> u8 {
    let [low, ..] = value.to_le_bytes();
    low
}
pub const fn low_u8_from_u64(value: u64) -> u8 {
    let [low, ..] = value.to_le_bytes();
    low
}
pub const fn low_u8_from_u128(value: u128) -> u8 {
    let [low, ..] = value.to_le_bytes();
    low
}
pub const fn low_u16_from_u64(value: u64) -> u16 {
    let [b0, b1, ..] = value.to_le_bytes();
    u16::from_le_bytes([b0, b1])
}
