use std::{
    io::{IsTerminal as TerminalDetect, stdout},
    sync::LazyLock,
};
cfg_select! {
    target_arch = "x86_64" => {
        pub const BUFFERS_PER_WORKER: usize = 8;
        pub const TWO_POW_32_F64: f64 = 4_294_967_296.0;
        pub const U64_UNIT_SCALE: f64 = 1.0 / (TWO_POW_32_F64 * TWO_POW_32_F64);
    }
    _ => {}
}
pub static GLYPHS: [char; 16] = [
    '🌅', '🐦', '👫', '🦕', '🌘', '🎈', '⛵', '🕷', '🦋', '🌀', '🧊', '🐟', '⛺', '🚀', '🌳', '🔯',
];
pub static IS_TERMINAL: LazyLock<bool> = LazyLock::new(|| TerminalDetect::is_terminal(&stdout()));
pub const FILE_NAME: &str = "random_data.txt";
pub const UTF8_BOM: &[u8; 3] = b"\xEF\xBB\xBF";
pub const ASCII_PRINTABLE_LEN: u8 = 94;
pub const ASCII_PRINTABLE_START: u8 = 33;
pub const BUFFER_SIZE: usize = 1016;
pub const BYTE_BITS: u8 = 64;
pub const EURO_LUCKY_MODULUS: u8 = 12;
pub const EURO_MAIN_MODULUS: u8 = 50;
pub const INPUT_BYTE_MAX_FOR_EURO_MAIN: u8 = 249;
pub const INPUT_BYTE_MAX_FOR_LOTTO: u8 = 224;
pub const INPUT_BYTE_MAX_FOR_LOTTO7: u8 = 221;
pub const INPUT_BYTE_MAX_FOR_LUCKY_STAR: u8 = 251;
pub const INPUT_BYTE_MAX_FOR_PASSWORD: u8 = 187;
pub const OUTPUT_FILE_BUFFER_CAPACITY: usize = 0x0010_0000;
pub const EURO_MILLIONS_LUCKY_COUNT: usize = 2;
pub const EURO_MILLIONS_MAIN_COUNT: usize = 5;
pub const HANGUL_BASE_CODE_POINT: u32 = 0xAC00;
pub const HANGUL_SYLLABLE_COUNT: usize = 4;
pub const HANGUL_SYLLABLE_MAX: u32 = 55_859;
pub const HANGUL_SYLLABLE_MODULUS: u32 = 11_172;
pub const LOTTO_MODULUS: u8 = 45;
pub const LOTTO7_COUNT: usize = 7;
pub const LOTTO7_MODULUS: u8 = 37;
pub const LOTTO_COUNT: usize = 6;
pub const LOTTO_COUNT_U8: u8 = 6;
pub const NIBBLE_MASK_U64: u64 = 0xF;
pub const NMS_COORD_MASK: u64 = 0x0FFF;
pub const NMS_GLYPH_COUNT: usize = 12;
pub const NMS_GLYPH_PREFIX_COUNT: usize = 4;
pub const NMS_PLANET_MAX_VALUE: u64 = 11;
pub const U32_MAX_INV: f64 = 1.0 / 4_294_967_295.0;
pub const HANGUL_SHIFTS: [u32; 4] = [48_u32, 32, 16, 0];
pub const NMS_GLYPH_NUM_SHIFTS: [u32; 8] = [36_u32, 32, 28, 24, 20, 16, 12, 8];
pub const NMS_PLANET_MODULUS: u64 = 6;
pub const NMS_SOLAR_SYSTEM_MAX_VALUE: u64 = 3834;
pub const NMS_SOLAR_SYSTEM_MODULUS: u64 = 767;
pub const ONE_BASED_OFFSET_U8: u8 = 1;
pub const ONE_BASED_OFFSET_U64: u64 = 1;
pub const ONE_BASED_OFFSET_USIZE: usize = 1;
pub const PASSWORD_BYTE_LEN: usize = 8;
pub const PASSWORD_BYTE_LEN_U8: u8 = 8;
const _: () = assert!(
    HANGUL_SHIFTS.len() == HANGUL_SYLLABLE_COUNT,
    "Hangul shift table must match syllable count"
);
const _: () = assert!(
    NMS_GLYPH_PREFIX_COUNT <= NMS_GLYPH_COUNT,
    "NMS glyph prefix must fit in glyph buffer"
);
const _: () = assert!(
    NMS_GLYPH_NUM_SHIFTS.len() == NMS_GLYPH_COUNT - NMS_GLYPH_PREFIX_COUNT,
    "NMS suffix shift table must match suffix glyph count"
);
