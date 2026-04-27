extern crate alloc;
use self::{
    batch::regenerate_with_count,
    numeric::{low_u8_from_u32, low_u8_from_u64, low_u16_from_u64},
    output::{
        format_data_into_buffer, prefix_slice, write_buffer_to_file_guard, write_slice_to_console,
    },
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{_rdrand64_step, _rdseed64_step};
#[cfg(windows)]
use core::ffi::c_void;
use core::{
    char::from_u32,
    error::Error,
    fmt::{Arguments, Display},
    hint::spin_loop,
    ops::{Mul as _, Sub as _},
    result::Result as StdResult,
    time::Duration,
};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt};
#[cfg(windows)]
use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};
#[cfg(windows)]
use std::os::windows::io::AsRawHandle as _;
use std::{
    fs::{self, File},
    io::{
        BufWriter, Error as IoError, ErrorKind, IsTerminal as _, Result as IoResult, Write, stderr,
        stdin, stdout,
    },
    is_x86_feature_detected,
    path::Path,
    process::ExitCode,
    sync::{LazyLock, Mutex, MutexGuard},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
mod batch;
mod buffmt;
mod numeric;
mod output;
pub(crate) mod time;
#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
compile_error!("SRG currently supports only Windows, Linux, and macOS.");
cfg_select! {
    target_os = "linux" => {
        const OPEN_NOFOLLOW_FLAG: i32 = 0x2_0000;
    }
    target_os = "macos" => {
        const OPEN_NOFOLLOW_FLAG: i32 = 0x0100;
    }
    windows => {
        const FILE_ATTRIBUTE_REPARSE_POINT_FLAG: u32 = 0x0000_0400;
        const FILE_FLAG_OPEN_REPARSE_POINT_FLAG: u32 = 0x0020_0000;
    }
}
cfg_select! {
    target_arch = "x86_64" => {
        static RNG_SOURCE: LazyLock<RngSource> = LazyLock::new(|| {
            if is_x86_feature_detected!("rdseed") {
                RngSource::RdSeed
            } else if is_x86_feature_detected!("rdrand") {
                RngSource::RdRand
            } else {
                RngSource::None
            }
        });
        const MENU: &str = concat!(
            "\n1: 사다리타기 실행, 2: 무작위 숫자 생성, 3: 데이터 생성(1회), ",
            "4: 데이터 생성(여러 회), 5: 서버 시간 확인, 6: 파일 삭제, ",
            "7: num_64/supp 수동 입력 생성, 기타: 종료\n선택해 주세요: ",
        );
    }
    _ => {
        static RNG_SOURCE: LazyLock<RngSource> = LazyLock::new(|| RngSource::None);
        const MENU: &str = concat!(
            "\n5: 서버 시간 확인, 7: num_64/supp 수동 입력 생성, 기타(1~4, 6 제외): 종료\n",
            "(참고: 이 플랫폼에서는 하드웨어 RNG 관련 기능이 비활성화됩니다)\n",
            "선택해 주세요: ",
        );
    }
}
static GLYPHS: [char; 16] = [
    '🌅', '🐦', '👫', '🦕', '🌘', '🎈', '⛵', '🕷', '🦋', '🌀', '🧊', '🐟', '⛺', '🚀', '🌳', '🔯',
];
static IS_TERMINAL: LazyLock<bool> = LazyLock::new(|| stdout().is_terminal());
const FILE_NAME: &str = "random_data.txt";
const UTF8_BOM: &[u8; 3] = b"\xEF\xBB\xBF";
const ASCII_PRINTABLE_LEN: u8 = 94;
const ASCII_PRINTABLE_START: u8 = 33;
const BUFFER_SIZE: usize = 1016;
const BYTE_BITS: u8 = 64;
const EURO_LUCKY_MODULUS: u8 = 12;
const EURO_MAIN_MODULUS: u8 = 50;
const GENERIC_INPUT_BUFFER_CAPACITY: usize = 256;
const HARDWARE_RANDOM_RETRY_COUNT: u8 = 10;
const INPUT_BYTE_MAX_FOR_EURO_MAIN: u8 = 249;
const INPUT_BYTE_MAX_FOR_LOTTO: u8 = 224;
const INPUT_BYTE_MAX_FOR_LOTTO7: u8 = 221;
const INPUT_BYTE_MAX_FOR_LUCKY_STAR: u8 = 251;
const INPUT_BYTE_MAX_FOR_PASSWORD: u8 = 187;
const OUTPUT_FILE_BUFFER_CAPACITY: usize = 0x0010_0000;
const BUFFERS_PER_WORKER: usize = 8;
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
const TWO_POW_32_F64: f64 = 4_294_967_296.0;
const U64_UNIT_SCALE: f64 = 1.0 / (TWO_POW_32_F64 * TWO_POW_32_F64);
const BAR_WIDTH: usize = 10;
const BAR_WIDTH_U64: u64 = 10;
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
const HEX_UPPER: [u8; 16] = *b"0123456789ABCDEF";
static BIN8_TABLE: LazyLock<[[u8; 8]; 256]> = LazyLock::new(|| {
    let mut table = [[0_u8; 8]; 256];
    let mut byte = 0_u8;
    for row in &mut table {
        for (bit, slot) in row.iter_mut().enumerate() {
            let shift = 7_usize.saturating_sub(bit);
            *slot = if ((byte >> shift) & 1) == 1 {
                b'1'
            } else {
                b'0'
            };
        }
        byte = byte.wrapping_add(1);
    }
    table
});
static HEX_BYTE_TABLE: LazyLock<[[u8; 2]; 256]> = LazyLock::new(|| {
    let mut table = [[0_u8; 2]; 256];
    let mut value = 0_u8;
    for slot in &mut table {
        let hi = usize::from(value >> 4_u8);
        let lo = usize::from(value & low_u8_from_u64(NIBBLE_MASK_U64));
        let Some(&hi_byte) = HEX_UPPER.get(hi) else {
            continue;
        };
        let Some(&lo_byte) = HEX_UPPER.get(lo) else {
            continue;
        };
        *slot = [hi_byte, lo_byte];
        value = value.wrapping_add(1);
    }
    table
});
type Result<T> = StdResult<T, Box<dyn Error + Send + Sync + 'static>>;
type DataBuffer = Box<[u8; BUFFER_SIZE]>;
#[cfg(windows)]
#[repr(C)]
struct FileTime {
    low_date_time: u32,
    high_date_time: u32,
}
#[cfg(windows)]
#[repr(C)]
struct ByHandleFileInformation {
    file_attributes: u32,
    creation_time: FileTime,
    last_access_time: FileTime,
    last_write_time: FileTime,
    volume_serial_number: u32,
    file_size_high: u32,
    file_size_low: u32,
    number_of_links: u32,
    file_index_high: u32,
    file_index_low: u32,
}
#[cfg(windows)]
unsafe extern "system" {
    fn GetFileInformationByHandle(
        h_file: *mut c_void,
        file_information: *mut ByHandleFileInformation,
    ) -> i32;
}
enum RngSource {
    None,
    RdRand,
    RdSeed,
}
#[derive(Default)]
struct RandomDataSet {
    euro_lucky_next_idx: usize,
    euro_main_next_idx: usize,
    euro_millions_lucky_stars: [u8; EURO_MILLIONS_LUCKY_COUNT],
    euro_millions_main_numbers: [u8; EURO_MILLIONS_MAIN_COUNT],
    galaxy_x: u16,
    galaxy_y: u16,
    galaxy_z: u16,
    glyph_string: [char; NMS_GLYPH_COUNT],
    hangul_syllables: [char; HANGUL_SYLLABLE_COUNT],
    kor_coords: (f64, f64),
    lotto7_next_idx: usize,
    lotto7_numbers: [u8; LOTTO7_COUNT],
    lotto_next_idx: usize,
    lotto_numbers: [u8; LOTTO_COUNT],
    nms_portal_xxx: u16,
    nms_portal_yy: u8,
    nms_portal_zzz: u16,
    num_64: u64,
    numeric_password: u32,
    numeric_password_digits: u8,
    password: [u8; PASSWORD_BYTE_LEN],
    password_len: u8,
    planet_number: u8,
    seen_euro_millions_lucky: u64,
    seen_euro_millions_main: u64,
    seen_lotto: u64,
    seen_lotto7: u64,
    solar_system_index: u16,
    world_coords: (f64, f64),
}
impl RandomDataSet {
    const fn is_complete(&self) -> bool {
        self.numeric_password_digits >= LOTTO_COUNT_U8
            && self.lotto_next_idx >= LOTTO_COUNT
            && self.lotto7_next_idx >= LOTTO7_COUNT
            && self.password_len >= PASSWORD_BYTE_LEN_U8
            && self.euro_main_next_idx >= EURO_MILLIONS_MAIN_COUNT
    }
}
#[derive(Clone, Copy)]
struct RandomBitBuffer {
    bits_remaining: u8,
    value: u64,
}
#[derive(Clone, Copy)]
enum RandomNumberMode {
    Float,
    Integer,
}
#[derive(Clone, Copy)]
enum LadderEntryMode {
    Players,
    Results { expected_count: usize },
}
type SupplementalProvider<'a> = dyn FnMut(&'static str) -> Result<RandomBitBuffer> + 'a;
struct RandomDataBuilder<'a, 'b> {
    data: RandomDataSet,
    next_supp: &'a mut SupplementalProvider<'b>,
    num: u64,
    supplemental: Option<RandomBitBuffer>,
}
impl RandomDataBuilder<'_, '_> {
    fn build(mut self) -> Result<RandomDataSet> {
        self.fill_required_fields()?;
        self.fill_lucky_stars()?;
        self.fill_hangul_syllables()?;
        self.fill_coords();
        self.fill_nms_fields()?;
        Ok(self.data)
    }
    fn fill_coords(&mut self) {
        let (upper_32_bits, lower_32_bits) = split_u64_to_u32_pair(self.num);
        let upper_ratio = f64::from(upper_32_bits).mul(U32_MAX_INV);
        let lower_ratio = f64::from(lower_32_bits).mul(U32_MAX_INV);
        self.data.kor_coords = (
            5.504_167_f64.mul_add(upper_ratio, 33.112_500),
            7.263_056_f64.mul_add(lower_ratio, 124.609_722),
        );
        self.data.world_coords = (
            180.0_f64.mul_add(upper_ratio, -90.0),
            360.0_f64.mul_add(lower_ratio, -180.0),
        );
        self.data.nms_portal_yy = low_u8_from_u32(upper_32_bits);
        self.data.nms_portal_zzz = low_u16_from_u64((self.num >> 20) & NMS_COORD_MASK);
        self.data.nms_portal_xxx = low_u16_from_u64((self.num >> 8) & NMS_COORD_MASK);
        self.data.galaxy_x = galaxy_coord::<0x801, 0x7FF>(self.data.nms_portal_xxx);
        self.data.galaxy_y = galaxy_coord::<0x81, 0x7F>(u16::from(self.data.nms_portal_yy));
        self.data.galaxy_z = galaxy_coord::<0x801, 0x7FF>(self.data.nms_portal_zzz);
    }
    fn fill_hangul_syllables(&mut self) -> Result<()> {
        let mut hangul = ['\0'; HANGUL_SYLLABLE_COUNT];
        for (slot, shift) in hangul.iter_mut().zip(HANGUL_SHIFTS) {
            let mut syllable_index = u32::from(low_u16_from_u64(self.num >> shift));
            while syllable_index > HANGUL_SYLLABLE_MAX {
                if self.supplemental.is_none() {
                    self.supplemental = Some(self.next_supplemental("한글 음절 보완")?);
                }
                let Some(supp_value) = self.supplemental.as_ref().map(|supp| supp.value) else {
                    return Err("한글 음절 보완 상태 불일치".into());
                };
                let mut candidate_value = None;
                for supp_shift in HANGUL_SHIFTS {
                    let value = u32::from(low_u16_from_u64(supp_value >> supp_shift));
                    if value <= HANGUL_SYLLABLE_MAX {
                        candidate_value = Some(value);
                        break;
                    }
                }
                if let Some(candidate) = candidate_value {
                    syllable_index = candidate;
                } else {
                    self.supplemental = Some(self.next_supplemental("한글 음절 보완 재시도")?);
                }
            }
            let Some(code_point) = HANGUL_BASE_CODE_POINT
                .checked_add(syllable_index.rem_euclid(HANGUL_SYLLABLE_MODULUS))
            else {
                return Err("한글 음절 코드포인트 계산 실패".into());
            };
            *slot = from_u32(code_point).ok_or("한글 음절 변환 실패")?;
        }
        self.data.hangul_syllables = hangul;
        Ok(())
    }
    fn fill_lucky_stars(&mut self) -> Result<()> {
        let mut lucky_star_source = self
            .supplemental
            .as_ref()
            .map_or_else(|| self.num.reverse_bits(), |supp| supp.value.reverse_bits());
        'lucky_star_loop: loop {
            for byte in lucky_star_source.to_be_bytes() {
                if byte > INPUT_BYTE_MAX_FOR_LUCKY_STAR {
                    continue;
                }
                if process_lotto_numbers(
                    byte,
                    EURO_LUCKY_MODULUS,
                    &mut self.data.euro_millions_lucky_stars,
                    &mut self.data.seen_euro_millions_lucky,
                    &mut self.data.euro_lucky_next_idx,
                ) && self.data.euro_lucky_next_idx >= EURO_MILLIONS_LUCKY_COUNT
                {
                    break 'lucky_star_loop;
                }
            }
            lucky_star_source = self
                .next_supplemental("유로밀리언 럭키 스타 보완")?
                .value
                .reverse_bits();
        }
        Ok(())
    }
    fn fill_nms_fields(&mut self) -> Result<()> {
        let planet_number_base = extract_valid_bits_for_nms::<4>(
            self.num,
            &[52, 4, 0],
            NMS_PLANET_MAX_VALUE,
            "NMS 행성 번호 보완",
            &mut self.supplemental,
            self.next_supp,
        )?
        .rem_euclid(NMS_PLANET_MODULUS);
        let planet_number = checked_add_one_u64(planet_number_base, "NMS 행성 번호 계산 실패")?;
        self.data.planet_number = low_u8_from_u64(planet_number);
        let solar_system_index_base = extract_valid_bits_for_nms::<12>(
            self.num,
            &[40],
            NMS_SOLAR_SYSTEM_MAX_VALUE,
            "NMS 태양계 번호 보완",
            &mut self.supplemental,
            self.next_supp,
        )?
        .rem_euclid(NMS_SOLAR_SYSTEM_MODULUS);
        let solar_system_index =
            checked_add_one_u64(solar_system_index_base, "NMS 태양계 번호 계산 실패")?;
        self.data.solar_system_index = low_u16_from_u64(solar_system_index);
        let (prefix_glyphs, suffix_glyphs) =
            self.data.glyph_string.split_at_mut(NMS_GLYPH_PREFIX_COUNT);
        for (slot, nibble_source) in prefix_glyphs.iter_mut().zip([
            u64::from(self.data.planet_number),
            u64::from(self.data.solar_system_index >> 8_u32),
            u64::from(self.data.solar_system_index >> 4_u32),
            u64::from(self.data.solar_system_index),
        ]) {
            let Some(glyph) = glyph_from_low_nibble(nibble_source) else {
                continue;
            };
            *slot = glyph;
        }
        for (slot, shift) in suffix_glyphs.iter_mut().zip(NMS_GLYPH_NUM_SHIFTS) {
            let Some(glyph) = glyph_from_low_nibble(self.num >> shift) else {
                continue;
            };
            *slot = glyph;
        }
        Ok(())
    }
    fn fill_required_fields(&mut self) -> Result<()> {
        fill_data_fields_from_u64(self.num, &mut self.data);
        while !self.data.is_complete() {
            let new_supp = self.next_supplemental("기본 필드 보완")?;
            fill_data_fields_from_u64(new_supp.value, &mut self.data);
        }
        Ok(())
    }
    fn next_supplemental(&mut self, reason: &'static str) -> Result<RandomBitBuffer> {
        let supplemental = (self.next_supp)(reason)?;
        self.supplemental = Some(supplemental);
        Ok(supplemental)
    }
}
struct MenuApp {
    file_mutex: Mutex<BufWriter<File>>,
    input_buffer: String,
    #[cfg(target_arch = "x86_64")]
    ladder_players_storage: String,
    #[cfg(target_arch = "x86_64")]
    ladder_results_storage: String,
    #[cfg(target_arch = "x86_64")]
    num_64: u64,
}
impl MenuApp {
    fn execute_command(
        &mut self,
        command: u8,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<bool> {
        cfg_select! {
            target_arch = "x86_64" => {
                match command {
                    b'1' => self.handle_ladder_command(out, err)?,
                    b'2' => self.handle_random_number_command(out, err)?,
                    b'3' => self.handle_generate_once_command(out, err)?,
                    b'4' => self.handle_generate_many_command(out, err)?,
                    b'5' => self.handle_server_time_command(out, err)?,
                    b'6' => {
                        validate_safe_output_file_path(Path::new(FILE_NAME), true)?;
                        if let Err(remove_err) = fs::remove_file(FILE_NAME) {
                            writeln!(err, "{remove_err}")?;
                        } else {
                            writeln!(out, "파일 '{FILE_NAME}'를 삭제했습니다.")?;
                        }
                    }
                    b'7' => self.handle_manual_input_command(out, err)?,
                    _ => return Ok(false),
                }
                Ok(true)
            }
            _ => {
                match command {
                    b'1' | b'2' | b'3' | b'4' | b'6' => {
                        print_x86_64_only_feature_disabled(out)?;
                    }
                    b'5' => self.handle_server_time_command(out, err)?,
                    b'7' => self.handle_manual_input_command(out, err)?,
                    _ => return Ok(false),
                }
                Ok(true)
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    fn handle_generate_many_command(
        &mut self,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<()> {
        let mut next_num_64 = self.num_64;
        let input_buffer = &mut self.input_buffer;
        run_hw_rng_menu_command(out, |command_out| {
            let count_prompt = format_args!("\n생성할 데이터 개수를 입력해 주세요: ");
            let requested_count = loop {
                match read_line_reuse(count_prompt, input_buffer, command_out)?.parse::<u64>() {
                    Ok(0) => writeln!(err, "1 이상의 값을 입력해 주세요.")?,
                    Ok(count) => break count,
                    Err(_) => writeln!(err, "유효한 숫자를 입력해 주세요.")?,
                }
            };
            next_num_64 =
                regenerate_with_count(&self.file_mutex, requested_count, command_out, err)?;
            Ok(())
        })?;
        self.num_64 = next_num_64;
        Ok(())
    }
    #[cfg(target_arch = "x86_64")]
    fn handle_generate_once_command(
        &mut self,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<()> {
        let mut next_num_64 = self.num_64;
        run_hw_rng_menu_command(out, |command_out| {
            next_num_64 = regenerate_with_count(&self.file_mutex, 1, command_out, err)?;
            Ok(())
        })?;
        self.num_64 = next_num_64;
        Ok(())
    }
    #[cfg(target_arch = "x86_64")]
    fn handle_ladder_command(&mut self, out: &mut dyn Write, err: &mut dyn Write) -> Result<()> {
        const MAX_PLAYERS: usize = 512;
        let mut seed = self.num_64;
        let input_buffer = &mut self.input_buffer;
        let players_storage = &mut self.ladder_players_storage;
        let results_storage = &mut self.ladder_results_storage;
        run_hw_rng_menu_command(out, |command_out| {
            players_storage.clear();
            let mut players_array: [&str; MAX_PLAYERS] = [""; MAX_PLAYERS];
            let n = read_ladder_entries(
                format_args!("\n사다리타기 플레이어를 입력해 주세요 (쉼표(,)로 구분, 2~512명): "),
                input_buffer,
                (command_out, err),
                players_storage,
                &mut players_array,
                LadderEntryMode::Players,
                "플레이어 배열 인덱스 범위 초과",
            )?;
            results_storage.clear();
            let mut results_array: [&str; MAX_PLAYERS] = [""; MAX_PLAYERS];
            read_ladder_entries(
                format_args!("사다리타기 결과값을 입력해 주세요 (쉼표(,)로 구분, {n}개 필요): "),
                input_buffer,
                (command_out, err),
                results_storage,
                &mut results_array,
                LadderEntryMode::Results { expected_count: n },
                "결과 배열 인덱스 범위 초과",
            )?;
            writeln!(command_out, "사다리타기 결과:")?;
            let mut indices = [0_usize; MAX_PLAYERS];
            let indices_slice = indices
                .get_mut(..n)
                .ok_or_else(|| IoError::other("인덱스 배열 슬라이스 범위 초과"))?;
            for (index, slot) in indices_slice.iter_mut().enumerate() {
                *slot = index;
            }
            for index in (1..indices_slice.len()).rev() {
                seed ^= get_hardware_random()?;
                let next_index = checked_add_one_usize(index)
                    .ok_or_else(|| IoError::other("인덱스 상한 계산 실패"))?;
                let upper_bound = u64::try_from(next_index).map_err(|conversion_err| {
                    boxed_other_with_source("인덱스 상한 변환 실패", conversion_err)
                })?;
                let swap_index_u64 = random_bounded(upper_bound, seed)?;
                let swap_index = usize::try_from(swap_index_u64).map_err(|conversion_err| {
                    boxed_other_with_source("인덱스 변환 실패", conversion_err)
                })?;
                indices_slice.swap(index, swap_index);
            }
            let players = players_array
                .get(..n)
                .ok_or_else(|| IoError::other("플레이어 슬라이스 범위 초과"))?;
            for (player, &result_index) in players.iter().zip(indices_slice.iter()) {
                let result = results_array
                    .get(result_index)
                    .copied()
                    .ok_or_else(|| IoError::other("결과 인덱스 범위 초과"))?;
                writeln!(command_out, "{player} -> {result}")?;
            }
            Ok(())
        })
    }
    fn handle_manual_input_command(
        &mut self,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<()> {
        ensure_file_exists_and_reopen(&self.file_mutex)?;
        writeln!(out, "\nnum_64/supp 수동 입력 생성 모드")?;
        self.input_buffer.clear();
        #[cfg(target_arch = "x86_64")]
        let num_64_slot = &mut self.num_64;
        #[cfg(not(target_arch = "x86_64"))]
        let mut manual_num_64 = 0_u64;
        #[cfg(not(target_arch = "x86_64"))]
        let num_64_slot = &mut manual_num_64;
        *num_64_slot = read_u64_hex_input(
            format_args!(
                "num_64를 입력해 주세요 (최소값 예: 0 또는 0x0, 최대값 예: {max_u64} 또는 0x{max_u64:X}): ",
                max_u64 = u64::MAX
            ),
            &mut self.input_buffer,
            out,
            err,
        )?;
        let mut supp_input_count = 0_usize;
        let mut next_supp = |reason: &'static str| -> Result<RandomBitBuffer> {
            supp_input_count = checked_add_one_usize(supp_input_count)
                .ok_or_else(|| IoError::other("supp 입력 횟수 계산 실패"))?;
            let supp = read_u64_hex_input(
                format_args!(
                    concat!(
                        "supp 값 #{} 입력 ({}, 최소값 예: 0 또는 0x0, ",
                        "최대값 예: {} 또는 0x{:X}): "
                    ),
                    supp_input_count,
                    reason,
                    u64::MAX,
                    u64::MAX
                ),
                &mut self.input_buffer,
                out,
                err,
            )?;
            Ok(RandomBitBuffer {
                value: supp,
                bits_remaining: BYTE_BITS,
            })
        };
        let data = generate_random_data_from_num(*num_64_slot, &mut next_supp)?;
        persist_and_print_random_data(&self.file_mutex, &data)?;
        Ok(())
    }
    #[cfg(target_arch = "x86_64")]
    fn handle_random_number_command(
        &mut self,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<()> {
        let num_64 = self.num_64;
        let input_buffer = &mut self.input_buffer;
        run_hw_rng_menu_command(out, |command_out| {
            writeln!(command_out, "\n무작위 숫자 생성 타입 선택:")?;
            let selection = read_line_reuse(
                format_args!("1: 정수 생성, 2: 실수 생성, 기타: 취소\n선택해 주세요: "),
                input_buffer,
                command_out,
            )?;
            match selection.as_bytes() {
                b"1" => generate_random_number(
                    RandomNumberMode::Integer,
                    num_64,
                    input_buffer,
                    command_out,
                    err,
                )?,
                b"2" => {
                    generate_random_number(
                        RandomNumberMode::Float,
                        num_64,
                        input_buffer,
                        command_out,
                        err,
                    )?;
                }
                _ => writeln!(command_out, "무작위 숫자 생성을 취소합니다.")?,
            }
            Ok(())
        })
    }
    fn handle_server_time_command(
        &mut self,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<()> {
        let time_run_result = (|| -> StdResult<(), time::TimeError> {
            cfg_select! {
                windows => {
                    if !*time::CURL_AVAILABLE {
                        writeln!(
                            err,
                            "[경고] 'curl' 명령어를 찾을 수 없습니다. TCP 연결 실패 시 대체 수단이 없습니다."
                        )?;
                    }
                }
                target_os = "linux" => {
                    if !*time::XDO_TOOL_AVAILABLE {
                        writeln!(
                            err,
                            concat!(
                                "[경고] 'xdotool'이 설치되지 않았습니다. 액션 기능이 동작하지 않습니다.\n",
                                "(설치 방법: sudo apt-get install xdotool 또는 유사한 패키지 관리자 명령어)"
                            )
                        )?;
                    }
                }
            }
            let host = self.read_server_host(out, err)?;
            let target_time = self.read_target_time(out)?;
            let trigger_action = self.read_trigger_action(out, target_time)?;
            let now = Instant::now();
            let baseline_placeholder = time::TimeSample {
                response_received_inst: now,
                rtt: Duration::ZERO,
                server_time: UNIX_EPOCH,
            };
            let mut app_state = cfg_select! {
                windows => {
                    time::AppState {
                        host,
                        target_time,
                        trigger_action,
                        server_time: None,
                        baseline_rtt: None,
                        baseline_rtt_samples: [baseline_placeholder; time::NUM_SAMPLES],
                        baseline_rtt_attempts: 0,
                        baseline_rtt_valid_count: 0,
                        baseline_rtt_next_sample_at: now,
                        next_full_sync_at: now,
                        last_sample: None,
                        live_rtt: None,
                        calibration_failure_count: 0,
                        high_res_timer_guard: None,
                    }
                }
                _ => {
                    time::AppState {
                        host,
                        target_time,
                        trigger_action,
                        server_time: None,
                        baseline_rtt: None,
                        baseline_rtt_samples: [baseline_placeholder; time::NUM_SAMPLES],
                        baseline_rtt_attempts: 0,
                        baseline_rtt_valid_count: 0,
                        baseline_rtt_next_sample_at: now,
                        next_full_sync_at: now,
                        last_sample: None,
                        live_rtt: None,
                        calibration_failure_count: 0,
                    }
                }
            };
            app_state.run_loop(out, err)?;
            writeln!(out, "\n프로그램을 종료합니다.")?;
            Ok(())
        })();
        if let Err(time_err) = time_run_result
            && time_err.io_kind() != Some(ErrorKind::UnexpectedEof)
        {
            writeln!(err, "서버 시간 확인 중 오류 발생: {time_err}")?;
        }
        Ok(())
    }
    fn read_server_host(
        &mut self,
        out: &mut dyn Write,
        err_out: &mut dyn Write,
    ) -> StdResult<time::address::ParsedServer, time::TimeError> {
        let (ignored_suffix, parsed_server) = time::get_validated_input(
            "확인할 서버 주소를 입력하세요 (예: www.example.com): ",
            &mut self.input_buffer,
            out,
            |raw_input| -> StdResult<(bool, time::address::ParsedServer), &'static str> {
                if raw_input.is_empty() {
                    return Err("서버 주소를 비워둘 수 없습니다.");
                }
                let after_scheme = time::address::strip_scheme_prefix(raw_input);
                let ignored_suffix =
                    time::address::authority_end_index(after_scheme) < after_scheme.len();
                time::address::ParsedServer::try_from(raw_input)
                    .map(|parsed| (ignored_suffix, parsed))
                    .map_err(|_err| "서버 주소가 올바르지 않습니다.")
            },
        )
        .map_err(time::TimeError::from)?;
        if ignored_suffix {
            writeln!(
                err_out,
                "[안내] 서버 주소의 경로/쿼리/프래그먼트는 무시되고 호스트만 사용됩니다."
            )?;
        }
        Ok(parsed_server)
    }
    fn read_target_time(
        &mut self,
        out: &mut dyn Write,
    ) -> StdResult<Option<SystemTime>, time::TimeError> {
        const INVALID_TIME_INPUT_ERR: &str =
            "잘못된 형식, 숫자 또는 시간 범위입니다 (HH:MM:SS, 0-23:0-59:0-59).";
        time::get_validated_input(
            "액션 실행 목표 시간을 입력하세요 (예: 20:00:00 / 건너뛰려면 Enter): ",
            &mut self.input_buffer,
            out,
            |raw_input| -> StdResult<Option<SystemTime>, &'static str> {
                if raw_input.is_empty() {
                    return Ok(None);
                }
                let mut time_parts = raw_input.split(':');
                let (Some(hour_str), Some(minute_str), Some(second_str), None) = (
                    time_parts.next(),
                    time_parts.next(),
                    time_parts.next(),
                    time_parts.next(),
                ) else {
                    return Err(INVALID_TIME_INPUT_ERR);
                };
                let (Ok(hour), Ok(minute), Ok(second)) = (
                    hour_str.parse::<u32>(),
                    minute_str.parse::<u32>(),
                    second_str.parse::<u32>(),
                ) else {
                    return Err(INVALID_TIME_INPUT_ERR);
                };
                if !(hour <= 23 && minute <= 59 && second <= 59) {
                    return Err(INVALID_TIME_INPUT_ERR);
                }
                let Ok(now_local) = SystemTime::now().duration_since(UNIX_EPOCH) else {
                    return Err("시간 계산 오류: 시스템 시간이 UNIX EPOCH보다 이전입니다.");
                };
                let Some(shifted_secs) = now_local.as_secs().checked_add(time::KST_OFFSET_SECS_U64)
                else {
                    return Err("시간 계산 오류: 현재 시각 계산 실패");
                };
                let today_days = shifted_secs.div_euclid(KST_SECS_PER_DAY_U64);
                let Some(today_start_base) = today_days.checked_mul(KST_SECS_PER_DAY_U64) else {
                    return Err("시간 계산 오류: 오늘 날짜 경계 계산 실패");
                };
                let Some(today_start_secs_utc) =
                    today_start_base.checked_sub(time::KST_OFFSET_SECS_U64)
                else {
                    return Err("시간 계산 오류: 오늘 날짜 경계 계산 실패");
                };
                let Some(hour_secs) = u64::from(hour).checked_mul(KST_SECS_PER_HOUR_U64) else {
                    return Err("시간 계산 오류: 목표 시각 계산 실패");
                };
                let Some(minute_secs) = u64::from(minute).checked_mul(KST_SECS_PER_MINUTE_U64)
                else {
                    return Err("시간 계산 오류: 목표 시각 계산 실패");
                };
                let Some(hour_and_minute_secs) = hour_secs.checked_add(minute_secs) else {
                    return Err("시간 계산 오류: 목표 시각 계산 실패");
                };
                let Some(target_secs_of_day) = hour_and_minute_secs.checked_add(u64::from(second))
                else {
                    return Err("시간 계산 오류: 목표 시각 계산 실패");
                };
                let Some(target_epoch_secs) = today_start_secs_utc.checked_add(target_secs_of_day)
                else {
                    return Err("시간 계산 오류: 목표 시각 계산 실패");
                };
                let Some(current_time) = SystemTime::UNIX_EPOCH.checked_add(now_local) else {
                    return Err("시간 계산 오류: 현재 시각 계산 실패");
                };
                let Some(mut target_time) =
                    UNIX_EPOCH.checked_add(Duration::from_secs(target_epoch_secs))
                else {
                    return Err("시간 계산 오류: 목표 시각 계산 실패");
                };
                if current_time > target_time {
                    target_time = match target_time.checked_add(Duration::from_hours(24)) {
                        Some(next_day_time) => next_day_time,
                        None => return Err("시간 계산 오류: 다음날 목표 시각 계산 실패"),
                    };
                }
                Ok(Some(target_time))
            },
        )
        .map_err(time::TimeError::from)
    }
    fn read_trigger_action(
        &mut self,
        out: &mut dyn Write,
        target_time: Option<SystemTime>,
    ) -> StdResult<Option<time::TriggerAction>, time::TimeError> {
        if target_time.is_none() {
            return Ok(None);
        }
        time::get_validated_input(
            "수행할 동작을 선택하세요 (1: 마우스 왼쪽 클릭, 2: F5 입력): ",
            &mut self.input_buffer,
            out,
            |selection| -> StdResult<time::TriggerAction, &'static str> {
                match selection.as_bytes() {
                    b"1" => Ok(time::TriggerAction::LeftClick),
                    b"2" => Ok(time::TriggerAction::F5Press),
                    _ => Err("잘못된 입력입니다. 1 또는 2를 입력해주세요."),
                }
            },
        )
        .map(Some)
        .map_err(time::TimeError::from)
    }
    fn run(&mut self) -> Result<ExitCode> {
        let menu_prompt = format_args!("{MENU}");
        loop {
            let command = {
                let mut prompt_out = stdout().lock();
                match read_line_reuse(menu_prompt, &mut self.input_buffer, &mut prompt_out) {
                    Ok(command_str) => match command_str.as_bytes() {
                        b"1" | b"2" | b"3" | b"4" | b"5" | b"6" | b"7" => {
                            command_str.as_bytes().first().copied().unwrap_or_default()
                        }
                        _ => 0,
                    },
                    Err(read_err) if read_err.kind() == ErrorKind::UnexpectedEof => {
                        return Ok(ExitCode::SUCCESS);
                    }
                    Err(read_err) => return Err(read_err.into()),
                }
            };
            let mut out = stdout();
            let mut err = stderr();
            let keep_running = match self.execute_command(command, &mut out, &mut err) {
                Ok(keep_running) => keep_running,
                Err(command_err)
                    if command_err
                        .downcast_ref::<IoError>()
                        .is_some_and(|io_err| io_err.kind() == ErrorKind::UnexpectedEof) =>
                {
                    return Ok(ExitCode::SUCCESS);
                }
                Err(command_err) => return Err(command_err),
            };
            if !keep_running {
                return Ok(ExitCode::SUCCESS);
            }
        }
    }
}
fn checked_add_one_u64(value: u64, err_msg: &'static str) -> Result<u64> {
    value
        .checked_add(ONE_BASED_OFFSET_U64)
        .ok_or_else(|| err_msg.into())
}
const fn checked_add_one_u8(value: u8) -> Option<u8> {
    value.checked_add(ONE_BASED_OFFSET_U8)
}
const fn checked_add_one_usize(value: usize) -> Option<usize> {
    value.checked_add(ONE_BASED_OFFSET_USIZE)
}
fn glyph_from_low_nibble(value: u64) -> Option<char> {
    GLYPHS
        .get(usize::from(low_u8_from_u64(value & NIBBLE_MASK_U64)))
        .copied()
}
const fn split_u64_to_u32_pair(value: u64) -> (u32, u32) {
    let [b0, b1, b2, b3, b4, b5, b6, b7] = value.to_be_bytes();

    (
        u32::from_be_bytes([b0, b1, b2, b3]),
        u32::from_be_bytes([b4, b5, b6, b7]),
    )
}
const fn galaxy_coord<const SUB: u16, const ADD: u16>(value: u16) -> u16 {
    let lower_bound = value.wrapping_sub(SUB);
    let upper_bound = value.wrapping_add(ADD);
    if lower_bound < upper_bound {
        lower_bound
    } else {
        upper_bound
    }
}
pub(crate) fn write_line_ignored(output: &mut dyn Write, args: Arguments<'_>) {
    match output.write_fmt(args) {
        Ok(()) | Err(_) => {}
    }
    match output.write_all(b"\n") {
        Ok(()) | Err(_) => {}
    }
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
            writeln!(err, "RDSEED를 미지원하여 RDRAND를 사용합니다.")?;
            let data = generate_random_data()?;
            let num_64 = data.num_64;
            persist_and_print_random_data(&file_mutex, &data)?;
            num_64
        }
        RngSource::None => {
            let mut err = stderr().lock();
            writeln!(
                err,
                "[경고] RDSEED/RDRAND를 지원하지 않아 하드웨어 RNG 기능(메뉴 1~4)을 비활성화합니다. 메뉴 5/7은 사용 가능합니다."
            )?;
            0
        }
    };
    let mut app = MenuApp {
        file_mutex,
        input_buffer: String::with_capacity(GENERIC_INPUT_BUFFER_CAPACITY),
        #[cfg(target_arch = "x86_64")]
        ladder_players_storage: String::with_capacity(GENERIC_INPUT_BUFFER_CAPACITY),
        #[cfg(target_arch = "x86_64")]
        ladder_results_storage: String::with_capacity(GENERIC_INPUT_BUFFER_CAPACITY),
        #[cfg(target_arch = "x86_64")]
        num_64,
    };
    app.run()
}
#[cfg(target_arch = "x86_64")]
fn run_hw_rng_menu_command(
    out: &mut dyn Write,
    action: impl FnOnce(&mut dyn Write) -> Result<()>,
) -> Result<()> {
    if matches!(*RNG_SOURCE, RngSource::None) {
        writeln!(
            out,
            "이 기능은 RDSEED/RDRAND를 지원하는 CPU에서만 사용할 수 있습니다."
        )?;
        return Ok(());
    }
    action(out)
}
#[cfg(not(target_arch = "x86_64"))]
fn print_x86_64_only_feature_disabled(out: &mut dyn Write) -> Result<()> {
    writeln!(
        out,
        "이 기능은 x86_64 전용이라 현재 플랫폼에서는 비활성화되어 있습니다."
    )?;
    Ok(())
}
fn ensure_file_exists_and_reopen(file_mutex: &Mutex<BufWriter<File>>) -> Result<()> {
    if Path::new(FILE_NAME).try_exists()? {
        return Ok(());
    }
    *lock_mutex(file_mutex, "Mutex 잠금 실패 (파일 생성 시)")? = open_or_create_file()?;
    Ok(())
}
fn invalid_output_path_err(message: &'static str) -> Box<dyn Error + Send + Sync + 'static> {
    Box::new(IoError::other(message))
}
fn validate_safe_output_file_path(path: &Path, allow_missing: bool) -> Result<()> {
    let maybe_metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => Some(metadata),
        Err(err) if allow_missing && err.kind() == ErrorKind::NotFound => None,
        Err(err) => return Err(Box::new(err)),
    };
    let Some(metadata) = maybe_metadata else {
        return Ok(());
    };
    cfg_select! {
        windows => {
            if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT_FLAG != 0 {
                return Err(invalid_output_path_err(
                    "출력 파일은 일반 파일이어야 하며 리파스 포인트는 허용되지 않습니다.",
                ));
            }
        }
        _ => {
            if metadata.file_type().is_symlink() {
                return Err(invalid_output_path_err(
                    "출력 파일은 일반 파일이어야 하며 심볼릭 링크는 허용되지 않습니다.",
                ));
            }
        }
    }
    if !metadata.is_file() {
        return Err(invalid_output_path_err(
            "출력 경로는 일반 파일이어야 합니다.",
        ));
    }
    let has_multiple_links = cfg_select! {
        windows => {
            file_has_multiple_links(
                &File::options()
                    .read(true)
                    .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT_FLAG)
                    .open(path)?,
            )?
        }
        _ => {
            metadata.nlink() > 1
        }
    };
    if has_multiple_links {
        return Err(invalid_output_path_err(
            "출력 파일은 하드 링크가 아니어야 합니다.",
        ));
    }
    Ok(())
}
#[cfg(windows)]
fn file_has_multiple_links(file: &File) -> IoResult<bool> {
    use core::mem::MaybeUninit;
    let mut file_information = MaybeUninit::<ByHandleFileInformation>::zeroed();
    // SAFETY: `GetFileInformationByHandle` only writes to the provided output
    // struct and uses the raw OS handle borrowed from `file` for the duration
    // of this call.
    let result =
        unsafe { GetFileInformationByHandle(file.as_raw_handle(), file_information.as_mut_ptr()) };
    if result == 0_i32 {
        return Err(IoError::last_os_error());
    }
    // SAFETY: The call above succeeded, so the OS initialized the full output
    // structure.
    let initialized_file_information = unsafe { file_information.assume_init() };
    Ok(initialized_file_information.number_of_links > 1)
}
fn boxed_other_with_source(
    context_msg: &'static str,
    err: impl Display,
) -> Box<dyn Error + Send + Sync + 'static> {
    Box::new(IoError::other(format!("{context_msg}: {err}")))
}
fn open_or_create_file() -> Result<BufWriter<File>> {
    let path = Path::new(FILE_NAME);
    validate_safe_output_file_path(path, true)?;
    let mut file = cfg_select! {
        windows => {
            File::options()
                .read(true)
                .append(true)
                .create(true)
                .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT_FLAG)
                .open(path)?
        }
        _ => {
            File::options()
                .read(true)
                .append(true)
                .create(true)
                .custom_flags(OPEN_NOFOLLOW_FLAG)
                .open(path)?
        }
    };
    match file.try_lock() {
        Ok(()) => {}
        Err(fs::TryLockError::WouldBlock) => {
            return Err(invalid_output_path_err(
                "다른 srg 인스턴스가 출력 파일을 사용 중입니다.",
            ));
        }
        Err(fs::TryLockError::Error(err)) => {
            return Err(boxed_other_with_source("출력 파일 잠금 실패", err));
        }
    }
    let metadata = file.metadata()?;
    cfg_select! {
        windows => {
            if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT_FLAG != 0 {
                return Err(invalid_output_path_err(
                    "출력 파일은 일반 파일이어야 하며 리파스 포인트는 허용되지 않습니다.",
                ));
            }
        }
        _ => {}
    }
    if !metadata.is_file() {
        return Err(invalid_output_path_err(
            "출력 경로는 일반 파일이어야 합니다.",
        ));
    }
    let has_multiple_links = cfg_select! {
        windows => {
            file_has_multiple_links(&file)?
        }
        _ => {
            metadata.nlink() > 1
        }
    };
    if has_multiple_links {
        return Err(invalid_output_path_err(
            "출력 파일은 하드 링크가 아니어야 합니다.",
        ));
    }
    if metadata.len() == 0 {
        file.write_all(UTF8_BOM)?;
        file.flush()?;
    }
    Ok(BufWriter::with_capacity(OUTPUT_FILE_BUFFER_CAPACITY, file))
}
fn lock_mutex<'a, T>(mutex: &'a Mutex<T>, context_msg: &'static str) -> Result<MutexGuard<'a, T>> {
    mutex
        .lock()
        .map_err(|err| boxed_other_with_source(context_msg, err))
}
pub(crate) fn write_random_data_to_console(data: &RandomDataSet) -> Result<()> {
    let mut buffer = [0_u8; BUFFER_SIZE];
    if *IS_TERMINAL {
        let console_len = format_data_into_buffer(data, &mut buffer, true)?;
        write_slice_to_console(prefix_slice(&buffer, console_len)?)?;
    } else {
        let file_len = format_data_into_buffer(data, &mut buffer, false)?;
        write_slice_to_console(prefix_slice(&buffer, file_len)?)?;
    }
    Ok(())
}
pub(crate) fn persist_and_print_random_data(
    file_mutex: &Mutex<BufWriter<File>>,
    data: &RandomDataSet,
) -> Result<()> {
    let mut buffer = [0_u8; BUFFER_SIZE];
    let file_len = format_data_into_buffer(data, &mut buffer, false)?;
    {
        let mut file_guard = lock_mutex(file_mutex, "Mutex 잠금 실패 (단일 쓰기 시)")?;
        write_buffer_to_file_guard(&mut file_guard, prefix_slice(&buffer, file_len)?)?;
        file_guard.flush()?;
    };
    write_random_data_to_console(data)
}
fn generate_random_data_from_num(
    num: u64,
    next_supp: &mut SupplementalProvider<'_>,
) -> Result<RandomDataSet> {
    RandomDataBuilder {
        data: RandomDataSet {
            num_64: num,
            ..Default::default()
        },
        next_supp,
        num,
        supplemental: None,
    }
    .build()
}
fn generate_random_data() -> Result<RandomDataSet> {
    let num = get_hardware_random()?;
    let mut next_supp = |_reason: &'static str| {
        Ok(RandomBitBuffer {
            value: get_hardware_random()?,
            bits_remaining: BYTE_BITS,
        })
    };
    generate_random_data_from_num(num, &mut next_supp)
}
fn get_hardware_random() -> Result<u64> {
    match *RNG_SOURCE {
        RngSource::RdSeed => {
            #[cfg(target_arch = "x86_64")]
            {
                let mut value = 0_u64;
                loop {
                    // SAFETY: `RNG_SOURCE` only routes here after confirming `rdseed`
                    // support, and the intrinsic writes to the valid mutable pointer to `value`.
                    if unsafe { _rdseed64_step(&mut value) } == 1_i32 {
                        break Ok(value);
                    }
                    spin_loop();
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                Ok(0)
            }
        }
        RngSource::RdRand => {
            #[cfg(target_arch = "x86_64")]
            {
                let mut value = 0_u64;
                for _ in 0_u8..HARDWARE_RANDOM_RETRY_COUNT {
                    // SAFETY: `RNG_SOURCE` only routes here after confirming `rdrand`
                    // support, and the intrinsic writes to the valid mutable pointer to `value`.
                    if unsafe { _rdrand64_step(&mut value) } == 1_i32 {
                        return Ok(value);
                    }
                    spin_loop();
                }
                Err("RDRAND 실패".into())
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                Err("RDSEED·RDRAND 모두 미지원합니다.".into())
            }
        }
        RngSource::None => Err("RDSEED·RDRAND 모두 미지원합니다.".into()),
    }
}
fn fill_data_fields_from_u64(value: u64, data: &mut RandomDataSet) {
    for byte in value.to_be_bytes() {
        if byte > INPUT_BYTE_MAX_FOR_EURO_MAIN {
            continue;
        }
        if data.numeric_password_digits < LOTTO_COUNT_U8 && {
            let digit = u32::from(byte.rem_euclid(10));
            let Some(next_password) = data
                .numeric_password
                .checked_mul(10)
                .and_then(|current_password| current_password.checked_add(digit))
            else {
                return;
            };
            data.numeric_password = next_password;
            let Some(next_digit_count) = checked_add_one_u8(data.numeric_password_digits) else {
                return;
            };
            data.numeric_password_digits = next_digit_count;
            next_digit_count >= LOTTO_COUNT_U8 && data.is_complete()
        } {
            return;
        }
        let euro_main_added = data.euro_main_next_idx < EURO_MILLIONS_MAIN_COUNT
            && process_lotto_numbers(
                byte,
                EURO_MAIN_MODULUS,
                &mut data.euro_millions_main_numbers,
                &mut data.seen_euro_millions_main,
                &mut data.euro_main_next_idx,
            );
        if euro_main_added
            && data.euro_main_next_idx >= EURO_MILLIONS_MAIN_COUNT
            && data.is_complete()
        {
            return;
        }
        if byte > INPUT_BYTE_MAX_FOR_LOTTO {
            continue;
        }
        let lotto_added = data.lotto_next_idx < LOTTO_COUNT
            && process_lotto_numbers(
                byte,
                LOTTO_MODULUS,
                &mut data.lotto_numbers,
                &mut data.seen_lotto,
                &mut data.lotto_next_idx,
            );
        if lotto_added && data.lotto_next_idx >= LOTTO_COUNT && data.is_complete() {
            return;
        }
        if byte > INPUT_BYTE_MAX_FOR_LOTTO7 {
            continue;
        }
        let lotto7_number_added = data.lotto7_next_idx < LOTTO7_COUNT
            && process_lotto_numbers(
                byte,
                LOTTO7_MODULUS,
                &mut data.lotto7_numbers,
                &mut data.seen_lotto7,
                &mut data.lotto7_next_idx,
            );
        if lotto7_number_added && data.lotto7_next_idx >= LOTTO7_COUNT && data.is_complete() {
            return;
        }
        if byte > INPUT_BYTE_MAX_FOR_PASSWORD {
            continue;
        }
        if data.password_len < PASSWORD_BYTE_LEN_U8
            && let Some(password_byte) = byte
                .rem_euclid(ASCII_PRINTABLE_LEN)
                .checked_add(ASCII_PRINTABLE_START)
            && let Some(slot) = data.password.get_mut(usize::from(data.password_len))
        {
            *slot = password_byte;
            let Some(next_password_len) = checked_add_one_u8(data.password_len) else {
                return;
            };
            data.password_len = next_password_len;
            if next_password_len >= PASSWORD_BYTE_LEN_U8 && data.is_complete() {
                return;
            }
        }
    }
}
fn process_lotto_numbers(
    byte: u8,
    modulus: u8,
    numbers: &mut [u8],
    seen: &mut u64,
    next_idx: &mut usize,
) -> bool {
    if *next_idx >= numbers.len() {
        return false;
    }
    let Some(number) = byte.checked_rem(modulus).and_then(checked_add_one_u8) else {
        return false;
    };
    let mask = 1_u64 << number;
    if (*seen & mask) == 0 {
        let Some(slot) = numbers.get_mut(*next_idx) else {
            return false;
        };
        *slot = number;
        *seen |= mask;
        let Some(next_index) = checked_add_one_usize(*next_idx) else {
            return false;
        };
        *next_idx = next_index;
        if *next_idx == numbers.len() {
            numbers.sort_unstable();
        }
        return true;
    }
    false
}
fn extract_valid_bits_for_nms<const BITS: u8>(
    num: u64,
    shifts: &[u8],
    max_value: u64,
    reason: &'static str,
    supplemental: &mut Option<RandomBitBuffer>,
    next_supp: &mut SupplementalProvider<'_>,
) -> Result<u64> {
    let bit_count = BITS.min(64);
    let mask = match bit_count {
        0 => 0,
        64 => u64::MAX,
        _ => 1_u64
            .checked_shl(u32::from(bit_count))
            .map_or(0, |shifted| shifted.saturating_sub(1)),
    };
    for &shift in shifts {
        let extracted_value = (num >> shift) & mask;
        if extracted_value <= max_value {
            return Ok(extracted_value);
        }
    }
    loop {
        let need_new = supplemental
            .as_ref()
            .is_none_or(|supp| supp.bits_remaining < BITS);
        if need_new {
            *supplemental = Some(next_supp(reason)?);
        }
        let Some(supp) = supplemental.as_mut() else {
            return Err("보완 난수 상태 불일치".into());
        };
        let Some(shift) = supp.bits_remaining.checked_sub(BITS) else {
            return Err(format!(
                "보완 난수 비트 수가 부족합니다. (remaining={}, required={BITS}, reason={reason})",
                supp.bits_remaining
            )
            .into());
        };
        let extracted = (supp.value >> shift) & mask;
        supp.bits_remaining = shift;
        if extracted <= max_value {
            return Ok(extracted);
        }
    }
}
fn read_line_reuse<'a>(
    prompt: Arguments<'_>,
    buffer: &'a mut String,
    out: &mut dyn Write,
) -> IoResult<&'a str> {
    buffer.clear();
    out.write_fmt(prompt)?;
    out.flush()?;
    let bytes_read = stdin().read_line(buffer)?;
    if bytes_read == 0 {
        return Err(IoError::new(
            ErrorKind::UnexpectedEof,
            "표준 입력이 종료되었습니다.",
        ));
    }
    Ok(buffer.trim())
}
fn read_u64_hex_input(
    prompt: Arguments<'_>,
    input_buffer: &mut String,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<u64> {
    loop {
        let raw = read_line_reuse(prompt, input_buffer, out)?;
        match raw
            .strip_prefix("0x")
            .or_else(|| raw.strip_prefix("0X"))
            .map_or_else(|| raw.parse::<u64>(), |hex| u64::from_str_radix(hex, 16))
        {
            Ok(value) => return Ok(value),
            Err(_) => {
                writeln!(
                    err,
                    "유효한 u64 형식이 아닙니다 (최소값 예: 0 또는 0x0, 최대값 예: {max_u64} 또는 0x{max_u64:X}).",
                    max_u64 = u64::MAX
                )?;
            }
        }
    }
}
fn parse_regular_f64(raw: &str) -> Option<f64> {
    raw.parse::<f64>()
        .ok()
        .filter(|value| value.is_finite() && !value.is_subnormal())
}
fn read_ladder_entries<'a, const N: usize>(
    prompt: Arguments<'_>,
    input_buffer: &mut String,
    io: (&mut dyn Write, &mut dyn Write),
    storage: &'a mut String,
    entries: &mut [&'a str; N],
    mode: LadderEntryMode,
    index_error: &'static str,
) -> Result<usize> {
    let (out, err) = io;
    let index_err = || IoError::other(index_error);
    let count = loop {
        let line = read_line_reuse(prompt, input_buffer, out)?;
        let mut count = 0_usize;
        let mut players_overflowed = false;
        let mut trimmed_ranges = [(0_usize, 0_usize); N];
        let mut segment_start = 0_usize;
        for (segment_end, separator) in line
            .char_indices()
            .filter_map(|(idx, ch)| (ch == ',').then_some((idx, true)))
            .chain([(line.len(), false)])
        {
            count = checked_add_one_usize(count).ok_or_else(|| {
                IoError::other(match mode {
                    LadderEntryMode::Players => "플레이어 수 계산 실패",
                    LadderEntryMode::Results { .. } => "결과 수 계산 실패",
                })
            })?;
            match mode {
                LadderEntryMode::Players if count > N => {
                    writeln!(err, "플레이어 수가 최대 {N}명을 초과했습니다.")?;
                    players_overflowed = true;
                    break;
                }
                LadderEntryMode::Results { expected_count } if count > expected_count => break,
                LadderEntryMode::Players | LadderEntryMode::Results { .. } => {
                    let part = line.get(segment_start..segment_end).ok_or_else(index_err)?;
                    let leading_whitespace = part.len().saturating_sub(part.trim_start().len());
                    let trimmed = part.trim();
                    let entry_index = count.checked_sub(1).ok_or_else(index_err)?;
                    let slot = trimmed_ranges.get_mut(entry_index).ok_or_else(index_err)?;
                    let range_start = segment_start
                        .checked_add(leading_whitespace)
                        .ok_or_else(index_err)?;
                    let range_end = range_start
                        .checked_add(trimmed.len())
                        .ok_or_else(index_err)?;
                    *slot = (range_start, range_end);
                }
            }
            if separator {
                segment_start = checked_add_one_usize(segment_end).ok_or_else(index_err)?;
            }
        }
        if players_overflowed || count == 0 {
            continue;
        }
        match mode {
            LadderEntryMode::Players if count < 2 => {
                writeln!(err, "플레이어 수는 최소 2명이어야 합니다.")?;
                continue;
            }
            LadderEntryMode::Results { expected_count } if count != expected_count => {
                writeln!(
                    err,
                    "결과값의 개수({count})가 플레이어 수({expected_count})와 일치하지 않습니다.\n"
                )?;
                continue;
            }
            LadderEntryMode::Players | LadderEntryMode::Results { .. } => {}
        }
        storage.clear();
        storage.push_str(line);
        for (entry_index, (range_start, range_end)) in
            trimmed_ranges.iter().copied().enumerate().take(count)
        {
            let part = storage.get(range_start..range_end).ok_or_else(index_err)?;
            let slot = entries.get_mut(entry_index).ok_or_else(index_err)?;
            *slot = part;
        }
        break count;
    };
    Ok(count)
}
fn read_parsed_value<T, F>(
    prompt: Arguments<'_>,
    buffer: &mut String,
    out: &mut dyn Write,
    err: &mut dyn Write,
    invalid_message: &'static str,
    mut parse: F,
) -> Result<T>
where
    F: FnMut(&str) -> Option<T>,
{
    loop {
        let line = read_line_reuse(prompt, buffer, out)?;
        if let Some(value) = parse(line) {
            return Ok(value);
        }
        writeln!(err, "{invalid_message}")?;
    }
}
fn generate_random_number(
    mode: RandomNumberMode,
    seed_modifier: u64,
    input_buffer: &mut String,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<()> {
    match mode {
        RandomNumberMode::Integer => {
            const MIN_ALLOWED_VALUE: i64 = i64::MIN + 1;
            writeln!(
                out,
                "\n무작위 정수 생성기(지원 범위: {MIN_ALLOWED_VALUE} ~ {max_i64})",
                max_i64 = i64::MAX
            )?;
            let min_value = loop {
                let value = read_parsed_value(
                    format_args!("최솟값을 입력해 주세요 ({MIN_ALLOWED_VALUE} 이상): "),
                    input_buffer,
                    out,
                    err,
                    "유효한 정수 형식이 아닙니다.",
                    |line| line.parse::<i64>().ok(),
                )?;
                if value >= MIN_ALLOWED_VALUE {
                    break value;
                }
                writeln!(err, "{MIN_ALLOWED_VALUE} 이상의 값을 입력해 주세요.")?;
            };
            let max_value = loop {
                let value = read_parsed_value(
                    format_args!("최댓값을 입력해 주세요: "),
                    input_buffer,
                    out,
                    err,
                    "유효한 정수 형식이 아닙니다.",
                    |line| line.parse::<i64>().ok(),
                )?;
                if value >= min_value {
                    break value;
                }
                writeln!(err, "최댓값은 최솟값보다 크거나 같아야 합니다.")?;
            };
            let range_size = max_value
                .wrapping_sub(min_value)
                .wrapping_add(1)
                .cast_unsigned();
            let rand_offset = if range_size == 0 {
                (get_hardware_random()? ^ seed_modifier).cast_signed()
            } else {
                random_bounded(range_size, seed_modifier)?.cast_signed()
            };
            let result = min_value.wrapping_add(rand_offset);
            writeln!(
                out,
                "무작위 정수({min_value} ~ {max_value}): {result} (0x{result:X})"
            )?;
        }
        RandomNumberMode::Float => {
            writeln!(out, "\n무작위 실수 생성기")?;
            let min_value = read_parsed_value(
                format_args!("최솟값을 입력해 주세요: "),
                input_buffer,
                out,
                err,
                "유효한 정규 실수 값을 입력해야 합니다 (NaN, 무한대, 비정규 값 제외).",
                parse_regular_f64,
            )?;
            let max_value = loop {
                let value = read_parsed_value(
                    format_args!("최댓값을 입력해 주세요: "),
                    input_buffer,
                    out,
                    err,
                    "유효한 정규 실수 값을 입력해야 합니다 (NaN, 무한대, 비정규 값 제외).",
                    parse_regular_f64,
                )?;
                if value >= min_value {
                    break value;
                }
                writeln!(err, "최댓값은 최솟값보다 크거나 같아야 합니다.")?;
            };
            let random_u64 = get_hardware_random()? ^ seed_modifier;
            let (upper_32, lower_32) = split_u64_to_u32_pair(random_u64);
            let scale = f64::from(upper_32)
                .mul_add(TWO_POW_32_F64, f64::from(lower_32))
                .mul(U64_UNIT_SCALE);
            let result = if min_value.to_bits() == max_value.to_bits() {
                min_value
            } else {
                scale.mul_add(max_value.sub(min_value), min_value)
            };
            writeln!(out, "무작위 실수({min_value} ~ {max_value}): {result}")?;
        }
    }
    Ok(())
}
fn random_bounded(range_size: u64, seed_mod: u64) -> Result<u64> {
    let threshold = range_size
        .wrapping_neg()
        .checked_rem(range_size)
        .ok_or_else(|| IoError::other("난수 범위 계산 실패"))?;
    let range_size128 = u128::from(range_size);
    loop {
        let Some(product) =
            u128::from(get_hardware_random()? ^ seed_mod).checked_mul(range_size128)
        else {
            continue;
        };
        let Ok(low_bits) = u64::try_from(product & u128::from(u64::MAX)) else {
            continue;
        };
        if low_bits >= threshold {
            let Ok(high_bits) = u64::try_from(product >> 64) else {
                continue;
            };
            return Ok(high_bits);
        }
    }
}
