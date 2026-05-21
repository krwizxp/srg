extern crate alloc;
use self::file_output::open_or_create_file;
use self::io_util::write_line_ignored;
use self::menu_app::MenuApp;
use self::tables::{BIN8_TABLE, HEX_BYTE_TABLE, HEX_UPPER};
use core::{error::Error, result::Result as StdResult};
use std::{
    io::{Error as IoError, IsTerminal as TerminalDetect, stdout},
    process::ExitCode,
    sync::{LazyLock, Mutex},
};
cfg_select! {
    target_arch = "x86_64" => {
        use self::hardware_rng::{RNG_SOURCE, RngSource};
        use self::random_data::generate_random_data;
        use self::random_output::persist_and_print_random_data;
        use std::io::{Write as IoWrite, stderr};
    }
    _ => {}
}
cfg_select! {
    target_arch = "x86_64" => {
        mod batch;
        mod hardware_rng;
        mod random_number;
        mod random_output;
    }
    _ => {}
}
mod buffmt;
mod file_output;
mod input;
mod io_util;
mod menu_app;
mod numeric;
mod output;
mod random_data;
mod random_util;
mod tables;
mod time;
#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
compile_error!("SRG currently supports only Windows, Linux, and macOS.");
cfg_select! {
    target_arch = "x86_64" => {
        const MENU: &str = concat!(
            "\n1: 사다리타기 실행, 2: 무작위 숫자 생성, 3: 데이터 생성(1회), ",
            "4: 데이터 생성(여러 회), 5: 서버 시간 확인, 6: 파일 삭제, ",
            "7: num_64/supp 수동 입력 생성, 기타: 종료\n선택해 주세요: ",
        );
    }
    _ => {
        const MENU: &str = concat!(
            "\n5: 서버 시간 확인, 6: 파일 삭제, 7: num_64/supp 수동 입력 생성, 기타(1~4 제외): 종료\n",
            "(참고: 이 플랫폼에서는 하드웨어 RNG 관련 기능이 비활성화됩니다)\n",
            "선택해 주세요: ",
        );
    }
}
static GLYPHS: [char; 16] = [
    '🌅', '🐦', '👫', '🦕', '🌘', '🎈', '⛵', '🕷', '🦋', '🌀', '🧊', '🐟', '⛺', '🚀', '🌳', '🔯',
];
static IS_TERMINAL: LazyLock<bool> = LazyLock::new(|| TerminalDetect::is_terminal(&stdout()));
const FILE_NAME: &str = "random_data.txt";
const UTF8_BOM: &[u8; 3] = b"\xEF\xBB\xBF";
const ASCII_PRINTABLE_LEN: u8 = 94;
const ASCII_PRINTABLE_START: u8 = 33;
const BUFFER_SIZE: usize = 1016;
const BYTE_BITS: u8 = 64;
const EURO_LUCKY_MODULUS: u8 = 12;
const EURO_MAIN_MODULUS: u8 = 50;
const GENERIC_INPUT_BUFFER_CAPACITY: usize = 256;
const INPUT_BYTE_MAX_FOR_EURO_MAIN: u8 = 249;
const INPUT_BYTE_MAX_FOR_LOTTO: u8 = 224;
const INPUT_BYTE_MAX_FOR_LOTTO7: u8 = 221;
const INPUT_BYTE_MAX_FOR_LUCKY_STAR: u8 = 251;
const INPUT_BYTE_MAX_FOR_PASSWORD: u8 = 187;
const OUTPUT_FILE_BUFFER_CAPACITY: usize = 0x0010_0000;
const EURO_MILLIONS_LUCKY_COUNT: usize = 2;
const EURO_MILLIONS_MAIN_COUNT: usize = 5;
const HANGUL_BASE_CODE_POINT: u32 = 0xAC00;
const HANGUL_SYLLABLE_COUNT: usize = 4;
const HANGUL_SYLLABLE_MAX: u32 = 55_859;
const HANGUL_SYLLABLE_MODULUS: u32 = 11_172;
const KST_SECS_PER_HOUR_U64: u64 = 3_600;
const KST_SECS_PER_MINUTE_U64: u64 = 60;
const KST_SECS_PER_DAY_U64: u64 = 24 * KST_SECS_PER_HOUR_U64;
const LOTTO_MODULUS: u8 = 45;
const LOTTO7_COUNT: usize = 7;
const LOTTO7_MODULUS: u8 = 37;
const LOTTO_COUNT: usize = 6;
const LOTTO_COUNT_U8: u8 = 6;
const NIBBLE_MASK_U64: u64 = 0xF;
const NMS_COORD_MASK: u64 = 0x0FFF;
const NMS_GLYPH_COUNT: usize = 12;
const NMS_GLYPH_PREFIX_COUNT: usize = 4;
const NMS_PLANET_MAX_VALUE: u64 = 11;
const U32_MAX_INV: f64 = 1.0 / 4_294_967_295.0;
const HANGUL_SHIFTS: [u32; 4] = [48_u32, 32, 16, 0];
const NMS_GLYPH_NUM_SHIFTS: [u32; 8] = [36_u32, 32, 28, 24, 20, 16, 12, 8];
const NMS_PLANET_MODULUS: u64 = 6;
const NMS_SOLAR_SYSTEM_MAX_VALUE: u64 = 3834;
const NMS_SOLAR_SYSTEM_MODULUS: u64 = 767;
const ONE_BASED_OFFSET_U8: u8 = 1;
const ONE_BASED_OFFSET_U64: u64 = 1;
const ONE_BASED_OFFSET_USIZE: usize = 1;
const PASSWORD_BYTE_LEN: usize = 8;
const PASSWORD_BYTE_LEN_U8: u8 = 8;
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
cfg_select! {
    target_arch = "x86_64" => {
        const BUFFERS_PER_WORKER: usize = 8;
        const TWO_POW_32_F64: f64 = 4_294_967_296.0;
        const U64_UNIT_SCALE: f64 = 1.0 / (TWO_POW_32_F64 * TWO_POW_32_F64);
        const BAR_WIDTH: usize = 10;
        const BAR_WIDTH_U64: u64 = 10;
        const BAR_FULL: [&str; BAR_WIDTH + 1] = [
            "[          ]",
            "[█         ]",
            "[██        ]",
            "[███       ]",
            "[████      ]",
            "[█████     ]",
            "[██████    ]",
            "[███████   ]",
            "[████████  ]",
            "[█████████ ]",
            "[██████████]",
        ];
        const INVALID_TIME: &[u8; 7] = b"--:--.-";
    }
    _ => {}
}
type Result<T> = StdResult<T, Box<dyn Error + Send + Sync>>;
cfg_select! {
    target_arch = "x86_64" => {
        type DataBuffer = Box<[u8; BUFFER_SIZE]>;
        fn reserved_string(capacity: usize, context: &str) -> Result<String> {
            let mut value = String::new();
            value
                .try_reserve(capacity)
                .map_err(|source| IoError::other(format!("{context}: {source}")))?;
            Ok(value)
        }
    }
    _ => {}
}
fn main() -> Result<ExitCode> {
    let file_mutex = Mutex::new(open_or_create_file()?);
    #[cfg(target_arch = "x86_64")]
    let num_64 = match *RNG_SOURCE {
        RngSource::RdSeed => {
            let data = generate_random_data()?;
            let num_64 = data.num_64;
            persist_and_print_random_data(&file_mutex, &data)?;
            num_64
        }
        RngSource::RdRand => {
            let mut err = stderr().lock();
            IoWrite::write_fmt(
                &mut err,
                format_args!("RDSEED를 미지원하여 RDRAND를 사용합니다.\n"),
            )?;
            let data = generate_random_data()?;
            let num_64 = data.num_64;
            persist_and_print_random_data(&file_mutex, &data)?;
            num_64
        }
        RngSource::None => {
            let mut err = stderr().lock();
            IoWrite::write_fmt(
                &mut err,
                format_args!(
                    "[경고] RDSEED/RDRAND를 지원하지 않아 하드웨어 RNG 기능(메뉴 1~4)을 비활성화합니다. 메뉴 5/7은 사용 가능합니다.\n"
                ),
            )?;
            0
        }
    };
    let input_buffer = cfg_select! {
        target_arch = "x86_64" => {
            reserved_string(GENERIC_INPUT_BUFFER_CAPACITY, "입력 버퍼 메모리 확보 실패")?
        }
        _ => {{
            let mut value = String::new();
            value
                .try_reserve(GENERIC_INPUT_BUFFER_CAPACITY)
                .map_err(|source| IoError::other(format!("입력 버퍼 메모리 확보 실패: {source}")))?;
            value
        }}
    };
    #[cfg(target_arch = "x86_64")]
    let ladder_players_storage = reserved_string(
        GENERIC_INPUT_BUFFER_CAPACITY,
        "사다리 참여자 버퍼 메모리 확보 실패",
    )?;
    #[cfg(target_arch = "x86_64")]
    let ladder_results_storage = reserved_string(
        GENERIC_INPUT_BUFFER_CAPACITY,
        "사다리 결과 버퍼 메모리 확보 실패",
    )?;
    let mut app = MenuApp {
        file_mutex,
        input_buffer,
        #[cfg(target_arch = "x86_64")]
        ladder_players_storage,
        #[cfg(target_arch = "x86_64")]
        ladder_results_storage,
        #[cfg(target_arch = "x86_64")]
        num_64,
    };
    app.run()
}
