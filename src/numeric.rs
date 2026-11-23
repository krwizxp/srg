macro_rules! low_u8_fn {
    ($(#[$meta:meta])* $name:ident, $ty:ty) => {
        $(#[$meta])*
        pub(super) const fn $name(value: $ty) -> u8 {
            value.to_le_bytes()[0]
        }
    };
}
low_u8_fn!(low_u8_from_u32, u32);
low_u8_fn!(low_u8_from_u64, u64);
low_u8_fn!(
    #[cfg(target_arch = "x86_64")]
    low_u8_from_u128,
    u128
);
pub(super) const fn low_u16_from_u64(value: u64) -> u16 {
    let [b0, b1, ..] = value.to_le_bytes();
    u16::from_le_bytes([b0, b1])
}
pub(super) const fn low_u32_from_u64(value: u64) -> u32 {
    let [b0, b1, b2, b3, ..] = value.to_le_bytes();
    u32::from_le_bytes([b0, b1, b2, b3])
}
pub(super) const fn split_u64_be_u32(value: u64) -> (u32, u32) {
    let [b0, b1, b2, b3, b4, b5, b6, b7] = value.to_be_bytes();
    (
        u32::from_be_bytes([b0, b1, b2, b3]),
        u32::from_be_bytes([b4, b5, b6, b7]),
    )
}
