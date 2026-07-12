use std::{
    io::{IsTerminal as TerminalDetect, stdout},
    sync::LazyLock,
};
cfg_select! {
    target_arch = "x86_64" => {
        pub(super) const BUFFERS_PER_WORKER: usize = 8;
        pub(super) const TWO_POW_32_F64: f64 = 4_294_967_296.0;
        pub(super) const U64_UNIT_SCALE: f64 = 1.0 / (TWO_POW_32_F64 * TWO_POW_32_F64);
    }
    _ => {}
}
pub(super) static IS_TERMINAL: LazyLock<bool> =
    LazyLock::new(|| TerminalDetect::is_terminal(&stdout()));
pub(super) const FILE_NAME: &str = "random_data.txt";
pub(super) const UTF8_BOM: &[u8; 3] = b"\xEF\xBB\xBF";
pub(super) const ASCII_PRINTABLE_LEN: u8 = 94;
pub(super) const ASCII_PRINTABLE_START: u8 = 33;
pub(super) const BUFFER_SIZE: usize = 1016;
pub(super) const BYTE_BITS: u8 = 64;
pub(super) const EURO_LUCKY_MODULUS: u8 = 12;
pub(super) const EURO_MAIN_MODULUS: u8 = 50;
pub(super) const INPUT_BYTE_MAX_FOR_EURO_MAIN: u8 = 249;
pub(super) const INPUT_BYTE_MAX_FOR_LOTTO: u8 = 224;
pub(super) const INPUT_BYTE_MAX_FOR_LOTTO7: u8 = 221;
pub(super) const INPUT_BYTE_MAX_FOR_LUCKY_STAR: u8 = 251;
pub(super) const INPUT_BYTE_MAX_FOR_PASSWORD: u8 = 187;
pub(super) const OUTPUT_FILE_BUFFER_CAPACITY: usize = 0x0010_0000;
pub(super) const EURO_MILLIONS_LUCKY_COUNT: usize = 2;
pub(super) const EURO_MILLIONS_MAIN_COUNT: usize = 5;
pub(super) const HANGUL_BASE_CODE_POINT: u32 = 0xAC00;
pub(super) const HANGUL_SYLLABLE_COUNT: usize = 4;
pub(super) const HANGUL_SYLLABLE_MAX: u32 = 55_859;
pub(super) const HANGUL_SYLLABLE_MODULUS: u32 = 11_172;
pub(super) const LOTTO_MODULUS: u8 = 45;
pub(super) const LOTTO7_COUNT: usize = 7;
pub(super) const LOTTO7_MODULUS: u8 = 37;
pub(super) const LOTTO_COUNT: usize = 6;
pub(super) const LOTTO_COUNT_U8: u8 = 6;
pub(super) const NIBBLE_MASK_U64: u64 = 0xF;
pub(super) const NMS_COORD_MASK: u64 = 0x0FFF;
pub(super) const NMS_GLYPH_COUNT: usize = 12;
pub(super) const NMS_PLANET_MAX_VALUE: u64 = 11;
pub(super) const U32_MAX_INV: f64 = 1.0 / 4_294_967_295.0;
pub(super) const HANGUL_SHIFTS: [u32; 4] = [48_u32, 32, 16, 0];
pub(super) const NMS_PLANET_MODULUS: u64 = 6;
pub(super) const NMS_SOLAR_SYSTEM_MAX_VALUE: u64 = 3834;
pub(super) const NMS_SOLAR_SYSTEM_MODULUS: u64 = 767;
pub(super) const PASSWORD_BYTE_LEN: usize = 8;
pub(super) const PASSWORD_BYTE_LEN_U8: u8 = 8;
const _: () = assert!(
    HANGUL_SHIFTS.len() == HANGUL_SYLLABLE_COUNT,
    "Hangul shift table must match syllable count"
);
