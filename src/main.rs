#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
compile_error!("SRG currently supports only Windows, Linux, and macOS.");
extern crate alloc;
mod batch;
mod buffmt;
mod numeric;
mod output;
pub(crate) mod time;
use self::{
    batch::regenerate_with_count,
    output::{
        format_data_into_buffer, prefix_slice, write_buffer_to_file_guard, write_slice_to_console,
    },
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{_rdrand64_step, _rdseed64_step};
use core::{
    any::Any,
    char::from_u32,
    error::Error,
    fmt::{Arguments, Display},
    hint::spin_loop,
    ops::{Mul as _, Sub as _},
    result::Result as stdResult,
    time::Duration,
};
use numeric::{low_u8_from_u32, low_u8_from_u64, low_u16_from_u64};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::unix::fs::OpenOptionsExt;
#[cfg(windows)]
use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};
use std::{
    fs::{self, File},
    io::{
        BufWriter, Error as ioErr, ErrorKind, IsTerminal as _, Result as IoRst, Write, stderr,
        stdin, stdout,
    },
    is_x86_feature_detected,
    path::Path,
    process::ExitCode,
    sync::{LazyLock, Mutex, MutexGuard},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
#[cfg(target_os = "linux")]
const OPEN_NOFOLLOW_FLAG: i32 = 0x2_0000;
#[cfg(target_os = "macos")]
const OPEN_NOFOLLOW_FLAG: i32 = 0x0100;
#[cfg(windows)]
const FILE_ATTRIBUTE_REPARSE_POINT_FLAG: u32 = 0x0000_0400;
#[cfg(windows)]
const FILE_FLAG_OPEN_REPARSE_POINT_FLAG: u32 = 0x0020_0000;
#[cfg(target_arch = "x86_64")]
static RNG_SOURCE: LazyLock<RngSource> = LazyLock::new(|| {
    if is_x86_feature_detected!("rdseed") {
        RngSource::RdSeed
    } else if is_x86_feature_detected!("rdrand") {
        RngSource::RdRand
    } else {
        RngSource::None
    }
});
#[cfg(not(target_arch = "x86_64"))]
static RNG_SOURCE: LazyLock<RngSource> = LazyLock::new(|| RngSource::None);
static GLYPHS: [char; 16] = [
    '🌅', '🐦', '👫', '🦕', '🌘', '🎈', '⛵', '🕷', '🦋', '🌀', '🧊', '🐟', '⛺', '🚀', '🌳', '🔯',
];
static IS_TERMINAL: LazyLock<bool> = LazyLock::new(|| stdout().is_terminal());
const FILE_NAME: &str = "random_data.txt";
const UTF8_BOM: &[u8; 3] = b"\xEF\xBB\xBF";
const BUFFER_SIZE: usize = 1016;
const BUFFERS_PER_WORKER: usize = 8;
const U32_MAX_INV: f64 = 1.0 / 4_294_967_295.0;
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
        let lo = usize::from(value & 0xF);
        if let (Some(hi_byte), Some(lo_byte)) = (HEX_UPPER.get(hi), HEX_UPPER.get(lo)) {
            *slot = [*hi_byte, *lo_byte];
        }
        value = value.wrapping_add(1);
    }
    table
});
#[cfg(target_arch = "x86_64")]
const MENU: &str = "\n1: 사다리타기 실행, 2: 무작위 숫자 생성, 3: 데이터 생성(1회), 4: 데이터 생성(여러 회), 5: 서버 시간 확인, 6: 파일 삭제, 7: num_64/supp 수동 입력 생성, 기타: 종료\n선택해 주세요: ";
#[cfg(not(target_arch = "x86_64"))]
const MENU: &str = "\n5: 서버 시간 확인, 7: num_64/supp 수동 입력 생성, 기타(1~4, 6 제외): 종료\n(참고: 이 플랫폼에서는 하드웨어 RNG 관련 기능이 비활성화됩니다)\n선택해 주세요: ";
type Result<T> = stdResult<T, Box<dyn Error + Send + Sync + 'static>>;
type DataBuffer = Box<[u8; BUFFER_SIZE]>;
enum RngSource {
    None,
    RdRand,
    RdSeed,
}
#[derive(Default)]
struct RandomDataSet {
    euro_lucky_next_idx: usize,
    euro_main_next_idx: usize,
    euro_millions_lucky_stars: [u8; 2],
    euro_millions_main_numbers: [u8; 5],
    galaxy_x: u16,
    galaxy_y: u16,
    galaxy_z: u16,
    glyph_string: [char; 12],
    hangul_syllables: [char; 4],
    kor_coords: (f64, f64),
    lotto7_next_idx: usize,
    lotto7_numbers: [u8; 7],
    lotto_next_idx: usize,
    lotto_numbers: [u8; 6],
    nms_portal_xxx: u16,
    nms_portal_yy: u8,
    nms_portal_zzz: u16,
    num_64: u64,
    numeric_password: u32,
    numeric_password_digits: u8,
    password: [char; 8],
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
        self.numeric_password_digits >= 6
            && self.lotto_next_idx >= 6
            && self.lotto7_next_idx >= 7
            && self.password_len >= 8
            && self.euro_main_next_idx >= 5
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
type SupplementalProvider<'provider> =
    dyn FnMut(&'static str) -> Result<RandomBitBuffer> + 'provider;
struct RandomDataBuilder<'provider, 'callback> {
    data: RandomDataSet,
    next_supp: &'provider mut SupplementalProvider<'callback>,
    num: u64,
    supplemental: Option<RandomBitBuffer>,
}
trait RandomDataBuilderExt {
    fn build(self) -> Result<RandomDataSet>;
    fn fill_coords(&mut self);
    fn fill_hangul_syllables(&mut self) -> Result<()>;
    fn fill_lucky_stars(&mut self) -> Result<()>;
    fn fill_nms_fields(&mut self) -> Result<()>;
    fn fill_required_fields(&mut self) -> Result<()>;
    fn next_supplemental(&mut self, reason: &'static str) -> Result<RandomBitBuffer>;
}
impl RandomDataBuilderExt for RandomDataBuilder<'_, '_> {
    fn build(mut self) -> Result<RandomDataSet> {
        self.fill_required_fields()?;
        self.fill_lucky_stars()?;
        self.fill_hangul_syllables()?;
        self.fill_coords();
        self.fill_nms_fields()?;
        Ok(self.data)
    }
    fn fill_coords(&mut self) {
        let upper_32_bits = (u32::from(low_u16_from_u64(self.num >> 48_u32)) << 16_u32)
            | u32::from(low_u16_from_u64(self.num >> 32_u32));
        let lower_32_bits = (u32::from(low_u16_from_u64(self.num >> 16_u32)) << 16_u32)
            | u32::from(low_u16_from_u64(self.num));
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
        self.data.nms_portal_zzz = low_u16_from_u64((self.num >> 20) & 0xFFF);
        self.data.nms_portal_xxx = low_u16_from_u64((self.num >> 8) & 0xFFF);
        self.data.galaxy_x = galaxy_coord::<0x801, 0x7FF>(self.data.nms_portal_xxx);
        self.data.galaxy_y = galaxy_coord::<0x81, 0x7F>(u16::from(self.data.nms_portal_yy));
        self.data.galaxy_z = galaxy_coord::<0x801, 0x7FF>(self.data.nms_portal_zzz);
    }
    fn fill_hangul_syllables(&mut self) -> Result<()> {
        let mut hangul = ['\0'; 4];
        for (slot, shift) in hangul.iter_mut().zip([48_u32, 32, 16, 0]) {
            let mut syllable_index = u32::from(low_u16_from_u64(self.num >> shift));
            while syllable_index > 55_859 {
                if self.supplemental.is_none() {
                    self.supplemental = Some(self.next_supplemental("한글 음절 보완")?);
                }
                let Some(supp_value) = self.supplemental.as_ref().map(|supp| supp.value) else {
                    return Err("한글 음절 보완 상태 불일치".into());
                };
                if let Some(candidate) = [48_u32, 32, 16, 0]
                    .into_iter()
                    .map(|supp_shift| u32::from(low_u16_from_u64(supp_value >> supp_shift)))
                    .find(|candidate| *candidate <= 55_859)
                {
                    syllable_index = candidate;
                } else {
                    self.supplemental = Some(self.next_supplemental("한글 음절 보완 재시도")?);
                }
            }
            let Some(code_point) = 0xAC00_u32.checked_add(syllable_index.rem_euclid(11_172)) else {
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
            for byte in u64_to_be_bytes(lucky_star_source) {
                if byte > 251 {
                    continue;
                }
                if process_lotto_numbers(
                    byte,
                    12,
                    &mut self.data.euro_millions_lucky_stars,
                    &mut self.data.seen_euro_millions_lucky,
                    &mut self.data.euro_lucky_next_idx,
                ) && self.data.euro_lucky_next_idx >= 2
                {
                    break 'lucky_star_loop;
                }
            }
            if self.data.euro_lucky_next_idx >= 2 {
                break;
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
            11,
            "NMS 행성 번호 보완",
            &mut self.supplemental,
            self.next_supp,
        )?
        .rem_euclid(6);
        let Some(planet_number) = planet_number_base.checked_add(1) else {
            return Err("NMS 행성 번호 계산 실패".into());
        };
        self.data.planet_number = low_u8_from_u64(planet_number);
        let solar_system_index_base = extract_valid_bits_for_nms::<12>(
            self.num,
            &[40],
            3834,
            "NMS 태양계 번호 보완",
            &mut self.supplemental,
            self.next_supp,
        )?
        .rem_euclid(767);
        let Some(solar_system_index) = solar_system_index_base.checked_add(1) else {
            return Err("NMS 태양계 번호 계산 실패".into());
        };
        self.data.solar_system_index = low_u16_from_u64(solar_system_index);
        for (idx, slot) in self.data.glyph_string.iter_mut().enumerate() {
            let nibble_source = match idx {
                0 => u64::from(self.data.planet_number),
                1 => u64::from(self.data.solar_system_index >> 8_u32),
                2 => u64::from(self.data.solar_system_index >> 4_u32),
                3 => u64::from(self.data.solar_system_index),
                4 => self.num >> 36_u32,
                5 => self.num >> 32_u32,
                6 => self.num >> 28_u32,
                7 => self.num >> 24_u32,
                8 => self.num >> 20_u32,
                9 => self.num >> 16_u32,
                10 => self.num >> 12_u32,
                _ => self.num >> 8_u32,
            };
            let nibble = usize::from(low_u8_from_u64(nibble_source & 0xF));
            if let Some(glyph) = GLYPHS.get(nibble).copied() {
                *slot = glyph;
            }
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
    num_64: u64,
}
impl MenuApp {
    fn execute_command(
        &mut self,
        command: &str,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<bool> {
        match command {
            #[cfg(target_arch = "x86_64")]
            "1" => self.handle_ladder_command(out, err)?,
            #[cfg(not(target_arch = "x86_64"))]
            "1" => print_x86_64_only_feature_disabled(out)?,
            #[cfg(target_arch = "x86_64")]
            "2" => self.handle_random_number_command(out, err)?,
            #[cfg(not(target_arch = "x86_64"))]
            "2" => print_x86_64_only_feature_disabled(out)?,
            #[cfg(target_arch = "x86_64")]
            "3" => self.handle_generate_once_command(out, err)?,
            #[cfg(not(target_arch = "x86_64"))]
            "3" => print_x86_64_only_feature_disabled(out)?,
            #[cfg(target_arch = "x86_64")]
            "4" => self.handle_generate_many_command(out, err)?,
            #[cfg(not(target_arch = "x86_64"))]
            "4" => print_x86_64_only_feature_disabled(out)?,
            "5" => self.handle_server_time_command(out, err)?,
            #[cfg(target_arch = "x86_64")]
            "6" => {
                if let Err(remove_err) = fs::remove_file(FILE_NAME) {
                    writeln!(err, "{remove_err}")?;
                } else {
                    writeln!(out, "파일 '{FILE_NAME}'를 삭제했습니다.")?;
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            "6" => print_x86_64_only_feature_disabled(out)?,
            "7" => self.handle_manual_input_command(out, err)?,
            _ => return Ok(false),
        }
        Ok(true)
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
        run_hw_rng_menu_command(out, |command_out| {
            let mut players_storage = String::with_capacity(256);
            let mut players_array: [&str; MAX_PLAYERS] = [""; MAX_PLAYERS];
            let n = read_ladder_entries(
                format_args!("\n사다리타기 플레이어를 입력해 주세요 (쉼표(,)로 구분, 2~512명): "),
                input_buffer,
                (command_out, err),
                &mut players_storage,
                &mut players_array,
                LadderEntryMode::Players,
                "플레이어 배열 인덱스 범위 초과",
            )?;
            let mut result_input_buffer = String::with_capacity(256);
            let mut results_storage = String::with_capacity(256);
            let mut results_array: [&str; MAX_PLAYERS] = [""; MAX_PLAYERS];
            read_ladder_entries(
                format_args!("사다리타기 결과값을 입력해 주세요 (쉼표(,)로 구분, {n}개 필요): "),
                &mut result_input_buffer,
                (command_out, err),
                &mut results_storage,
                &mut results_array,
                LadderEntryMode::Results { expected_count: n },
                "결과 배열 인덱스 범위 초과",
            )?;
            writeln!(command_out, "사다리타기 결과:")?;
            let mut indices = [0_usize; MAX_PLAYERS];
            let indices_slice = indices
                .get_mut(..n)
                .ok_or_else(|| ioErr::other("인덱스 배열 슬라이스 범위 초과"))?;
            for (index, slot) in indices_slice.iter_mut().enumerate() {
                *slot = index;
            }
            for index in (1..indices_slice.len()).rev() {
                seed ^= get_hardware_random()?;
                let next_index = index
                    .checked_add(1)
                    .ok_or_else(|| ioErr::other("인덱스 상한 계산 실패"))?;
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
                .ok_or_else(|| ioErr::other("플레이어 슬라이스 범위 초과"))?;
            for (player, &result_index) in players.iter().zip(indices_slice.iter()) {
                let result = results_array
                    .get(result_index)
                    .copied()
                    .ok_or_else(|| ioErr::other("결과 인덱스 범위 초과"))?;
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
        *num_64_slot = loop {
            let raw = read_line_reuse(
                format_args!(
                    "num_64를 입력해 주세요 (최소값 예: 0 또는 0x0, 최대값 예: {max_u64} 또는 0x{max_u64:X}): ",
                    max_u64 = u64::MAX
                ),
                &mut self.input_buffer,
                out,
            )?;
            let parsed = raw
                .strip_prefix("0x")
                .or_else(|| raw.strip_prefix("0X"))
                .map_or_else(|| raw.parse::<u64>(), |hex| u64::from_str_radix(hex, 16));
            match parsed {
                Ok(value) => break value,
                Err(_) => writeln!(
                    err,
                    "유효한 u64 형식이 아닙니다 (최소값 예: 0 또는 0x0, 최대값 예: {max_u64} 또는 0x{max_u64:X}).",
                    max_u64 = u64::MAX
                )?,
            }
        };
        let mut supp_input_count = 0_usize;
        let mut next_supp = |reason: &'static str| -> Result<RandomBitBuffer> {
            supp_input_count = supp_input_count
                .checked_add(1)
                .ok_or_else(|| ioErr::other("supp 입력 횟수 계산 실패"))?;
            let supp = loop {
                let raw = read_line_reuse(
                    format_args!(
                        "supp 값 #{supp_input_count} 입력 ({reason}, 최소값 예: 0 또는 0x0, 최대값 예: {max_u64} 또는 0x{max_u64:X}): ",
                        max_u64 = u64::MAX
                    ),
                    &mut self.input_buffer,
                    out,
                )?;
                let parsed = raw
                    .strip_prefix("0x")
                    .or_else(|| raw.strip_prefix("0X"))
                    .map_or_else(|| raw.parse::<u64>(), |hex| u64::from_str_radix(hex, 16));
                match parsed {
                    Ok(value) => break value,
                    Err(_) => writeln!(
                        err,
                        "유효한 u64 형식이 아닙니다 (최소값 예: 0 또는 0x0, 최대값 예: {max_u64} 또는 0x{max_u64:X}).",
                        max_u64 = u64::MAX
                    )?,
                }
            };
            Ok(RandomBitBuffer {
                value: supp,
                bits_remaining: 64,
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
            match read_line_reuse(
                format_args!("1: 정수 생성, 2: 실수 생성, 기타: 취소\n선택해 주세요: "),
                input_buffer,
                command_out,
            )? {
                "1" => generate_random_number(
                    RandomNumberMode::Integer,
                    num_64,
                    input_buffer,
                    command_out,
                    err,
                )?,
                "2" => {
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
        let time_run_result = (|| -> stdResult<(), time::TimeError> {
            #[cfg(target_os = "windows")]
            if !*time::CURL_AVAILABLE {
                writeln!(
                    err,
                    "[경고] 'curl' 명령어를 찾을 수 없습니다. TCP 연결 실패 시 대체 수단이 없습니다."
                )?;
            }
            #[cfg(target_os = "linux")]
            if !*time::XDO_TOOL_AVAILABLE {
                writeln!(
                    err,
                    "[경고] 'xdotool'이 설치되지 않았습니다. 액션 기능이 동작하지 않습니다.\n(설치 방법: sudo apt-get install xdotool 또는 유사한 패키지 관리자 명령어)"
                )?;
            }
            let host = self.read_server_host(out, err)?;
            let target_time = self.read_target_time(out)?;
            let trigger_action = self.read_trigger_action(out, target_time)?;
            let baseline_placeholder = time::TimeSample {
                response_received_inst: Instant::now(),
                rtt: Duration::ZERO,
                server_time: UNIX_EPOCH,
            };
            let mut app_state = time::AppState {
                host,
                target_time,
                trigger_action,
                server_time: None,
                baseline_rtt: None,
                baseline_rtt_samples: [baseline_placeholder; time::NUM_SAMPLES],
                baseline_rtt_attempts: 0,
                baseline_rtt_valid_count: 0,
                baseline_rtt_next_sample_at: Instant::now(),
                next_full_sync_at: Instant::now(),
                last_sample: None,
                live_rtt: None,
                calibration_failure_count: 0,
                #[cfg(target_os = "windows")]
                high_res_timer_guard: None,
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
    ) -> stdResult<String, time::TimeError> {
        let host = time::get_validated_input(
            "확인할 서버 주소를 입력하세요 (예: www.example.com): ",
            &mut self.input_buffer,
            out,
            |raw_input| -> stdResult<String, &'static str> {
                if raw_input.is_empty() {
                    return Err("서버 주소를 비워둘 수 없습니다.");
                }
                time::address::parse_server(raw_input)
                    .map(|_parsed| raw_input.to_owned())
                    .map_err(|_err| "서버 주소가 올바르지 않습니다.")
            },
        )
        .map_err(time::TimeError::from)?;
        let host_bytes = host.as_bytes();
        let after_scheme = if host_bytes
            .get(..8)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"https://"))
        {
            host.get(8..).unwrap_or(host.as_str())
        } else if host_bytes
            .get(..7)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"http://"))
        {
            host.get(7..).unwrap_or(host.as_str())
        } else {
            host.as_str()
        };
        if after_scheme
            .bytes()
            .any(|byte| matches!(byte, b'/' | b'?' | b'#'))
        {
            writeln!(
                err_out,
                "[안내] 서버 주소의 경로/쿼리/프래그먼트는 무시되고 호스트만 사용됩니다."
            )?;
        }
        Ok(host)
    }
    fn read_target_time(
        &mut self,
        out: &mut dyn Write,
    ) -> stdResult<Option<SystemTime>, time::TimeError> {
        time::get_validated_input(
            "액션 실행 목표 시간을 입력하세요 (예: 20:00:00 / 건너뛰려면 Enter): ",
            &mut self.input_buffer,
            out,
            |raw_input| -> stdResult<Option<SystemTime>, &'static str> {
                if raw_input.is_empty() {
                    return Ok(None);
                }
                let mut parts = raw_input.split(':');
                let (Some(hour_str), Some(minute_str), Some(second_str)) =
                    (parts.next(), parts.next(), parts.next())
                else {
                    return Err(
                        "잘못된 형식, 숫자 또는 시간 범위입니다 (HH:MM:SS, 0-23:0-59:0-59).",
                    );
                };
                if parts.next().is_some() {
                    return Err(
                        "잘못된 형식, 숫자 또는 시간 범위입니다 (HH:MM:SS, 0-23:0-59:0-59).",
                    );
                }
                let (Ok(hour), Ok(minute), Ok(second)) = (
                    hour_str.parse::<u32>(),
                    minute_str.parse::<u32>(),
                    second_str.parse::<u32>(),
                ) else {
                    return Err(
                        "잘못된 형식, 숫자 또는 시간 범위입니다 (HH:MM:SS, 0-23:0-59:0-59).",
                    );
                };
                if !(hour <= 23 && minute <= 59 && second <= 59) {
                    return Err(
                        "잘못된 형식, 숫자 또는 시간 범위입니다 (HH:MM:SS, 0-23:0-59:0-59).",
                    );
                }
                let Ok(now_local) = SystemTime::now().duration_since(UNIX_EPOCH) else {
                    return Err("시간 계산 오류: 시스템 시간이 UNIX EPOCH보다 이전입니다.");
                };
                let Some(shifted_secs) = now_local.as_secs().checked_add(time::KST_OFFSET_SECS_U64)
                else {
                    return Err("시간 계산 오류: 현재 시각 계산 실패");
                };
                let today_days = shifted_secs.div_euclid(86400);
                let Some(today_start_base) = today_days.checked_mul(86400) else {
                    return Err("시간 계산 오류: 오늘 날짜 경계 계산 실패");
                };
                let Some(today_start_secs_utc) =
                    today_start_base.checked_sub(time::KST_OFFSET_SECS_U64)
                else {
                    return Err("시간 계산 오류: 오늘 날짜 경계 계산 실패");
                };
                let Some(hour_secs) = u64::from(hour).checked_mul(3600) else {
                    return Err("시간 계산 오류: 목표 시각 계산 실패");
                };
                let Some(minute_secs) = u64::from(minute).checked_mul(60) else {
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
                let Some(mut target_time) =
                    UNIX_EPOCH.checked_add(Duration::from_secs(target_epoch_secs))
                else {
                    return Err("시간 계산 오류: 목표 시각 계산 실패");
                };
                if SystemTime::UNIX_EPOCH
                    .checked_add(now_local)
                    .is_some_and(|current_time| current_time > target_time)
                {
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
    ) -> stdResult<Option<time::TriggerAction>, time::TimeError> {
        if target_time.is_none() {
            return Ok(None);
        }
        time::get_validated_input(
            "수행할 동작을 선택하세요 (1: 마우스 왼쪽 클릭, 2: F5 입력): ",
            &mut self.input_buffer,
            out,
            |selection| -> stdResult<time::TriggerAction, &'static str> {
                match selection {
                    "1" => Ok(time::TriggerAction::LeftClick),
                    "2" => Ok(time::TriggerAction::F5Press),
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
                    Ok(cmd) => cmd.to_owned(),
                    Err(read_err) if read_err.kind() == ErrorKind::UnexpectedEof => {
                        return Ok(ExitCode::SUCCESS);
                    }
                    Err(read_err) => return Err(read_err.into()),
                }
            };
            let mut out = stdout();
            let mut err = stderr();
            let keep_running = match self.execute_command(command.as_str(), &mut out, &mut err) {
                Ok(keep_running) => keep_running,
                Err(command_err)
                    if command_err
                        .downcast_ref::<ioErr>()
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
const fn u64_to_be_bytes(value: u64) -> [u8; 8] {
    [
        low_u8_from_u64(value >> 56_u32),
        low_u8_from_u64(value >> 48_u32),
        low_u8_from_u64(value >> 40_u32),
        low_u8_from_u64(value >> 32_u32),
        low_u8_from_u64(value >> 24_u32),
        low_u8_from_u64(value >> 16_u32),
        low_u8_from_u64(value >> 8_u32),
        low_u8_from_u64(value),
    ]
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
        input_buffer: String::with_capacity(256),
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
    if !Path::new(FILE_NAME).try_exists()? {
        *lock_mutex(file_mutex, "Mutex 잠금 실패 (파일 생성 시)")? = open_or_create_file()?;
    }
    Ok(())
}
fn invalid_output_path_err(message: &'static str) -> Box<dyn Error + Send + Sync + 'static> {
    Box::new(ioErr::other(message))
}
fn boxed_other_with_source(
    context_msg: &'static str,
    err: impl Display,
) -> Box<dyn Error + Send + Sync + 'static> {
    Box::new(ioErr::other(format!("{context_msg}: {err}")))
}
fn describe_panic_payload(payload: &(dyn Any + Send + 'static)) -> String {
    match (
        payload.downcast_ref::<&str>(),
        payload.downcast_ref::<String>(),
    ) {
        (Some(message), None) => (*message).to_owned(),
        (None, Some(message)) => message.clone(),
        _ => "non-string panic payload".to_owned(),
    }
}
fn open_or_create_file() -> Result<BufWriter<File>> {
    let path = Path::new(FILE_NAME);
    let maybe_metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => Some(metadata),
        Err(err) if err.kind() == ErrorKind::NotFound => None,
        Err(err) => return Err(Box::new(err)),
    };
    if let Some(metadata) = maybe_metadata {
        #[cfg(windows)]
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT_FLAG != 0 {
            return Err(invalid_output_path_err(
                "출력 파일은 일반 파일이어야 하며 리파스 포인트는 허용되지 않습니다.",
            ));
        }
        #[cfg(not(windows))]
        if metadata.file_type().is_symlink() {
            return Err(invalid_output_path_err(
                "출력 파일은 일반 파일이어야 하며 심볼릭 링크는 허용되지 않습니다.",
            ));
        }
        if !metadata.is_file() {
            return Err(invalid_output_path_err(
                "출력 경로는 일반 파일이어야 합니다.",
            ));
        }
    }
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    let mut file = File::options()
        .read(true)
        .append(true)
        .create(true)
        .custom_flags(OPEN_NOFOLLOW_FLAG)
        .open(path)?;
    #[cfg(windows)]
    let mut file = File::options()
        .read(true)
        .append(true)
        .create(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT_FLAG)
        .open(path)?;
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    let mut file = return Err(invalid_output_path_err(
        "지원되지 않는 운영체제입니다. Windows, Linux, macOS만 지원합니다.",
    ));
    let metadata = file.metadata()?;
    #[cfg(windows)]
    if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT_FLAG != 0 {
        return Err(invalid_output_path_err(
            "출력 파일은 일반 파일이어야 하며 리파스 포인트는 허용되지 않습니다.",
        ));
    }
    if !metadata.is_file() {
        return Err(invalid_output_path_err(
            "출력 경로는 일반 파일이어야 합니다.",
        ));
    }
    if metadata.len() == 0 {
        file.write_all(UTF8_BOM)?;
        file.flush()?;
    }
    Ok(BufWriter::with_capacity(0x0010_0000, file))
}
fn lock_mutex<'mutex, T>(
    mutex: &'mutex Mutex<T>,
    context_msg: &'static str,
) -> Result<MutexGuard<'mutex, T>> {
    mutex
        .lock()
        .map_err(|err| boxed_other_with_source(context_msg, err))
}
fn persist_and_print_random_data(
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
    if *IS_TERMINAL {
        let console_len = format_data_into_buffer(data, &mut buffer, true)?;
        write_slice_to_console(prefix_slice(&buffer, console_len)?)?;
    } else {
        write_slice_to_console(prefix_slice(&buffer, file_len)?)?;
    }
    Ok(())
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
            bits_remaining: 64,
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
                    // SAFETY: `RNG_SOURCE` only routes here after confirming `rdseed` support,
                    // and the intrinsic writes to the valid mutable pointer to `value`.
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
                for _ in 0_u8..10_u8 {
                    // SAFETY: `RNG_SOURCE` only routes here after confirming `rdrand` support,
                    // and the intrinsic writes to the valid mutable pointer to `value`.
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
    for byte in u64_to_be_bytes(value) {
        if byte > 249 {
            continue;
        }
        if data.numeric_password_digits < 6 && {
            let digit = u32::from(byte.rem_euclid(10));
            let Some(next_password) = data
                .numeric_password
                .checked_mul(10)
                .and_then(|current_password| current_password.checked_add(digit))
            else {
                return;
            };
            data.numeric_password = next_password;
            let Some(next_digit_count) = data.numeric_password_digits.checked_add(1) else {
                return;
            };
            data.numeric_password_digits = next_digit_count;
            data.is_complete()
        } {
            return;
        }
        if data.euro_main_next_idx < 5
            && process_lotto_numbers(
                byte,
                50,
                &mut data.euro_millions_main_numbers,
                &mut data.seen_euro_millions_main,
                &mut data.euro_main_next_idx,
            )
            && data.is_complete()
        {
            return;
        }
        if byte > 224 {
            continue;
        }
        if data.lotto_next_idx < 6
            && process_lotto_numbers(
                byte,
                45,
                &mut data.lotto_numbers,
                &mut data.seen_lotto,
                &mut data.lotto_next_idx,
            )
            && data.is_complete()
        {
            return;
        }
        if byte > 221 {
            continue;
        }
        if data.lotto7_next_idx < 7
            && process_lotto_numbers(
                byte,
                37,
                &mut data.lotto7_numbers,
                &mut data.seen_lotto7,
                &mut data.lotto7_next_idx,
            )
            && data.is_complete()
        {
            return;
        }
        if byte > 187 {
            continue;
        }
        if data.password_len < 8
            && let Some(code_point) = u32::from(byte.rem_euclid(94)).checked_add(33)
            && let Some(ch) = from_u32(code_point)
            && let Some(slot) = data.password.get_mut(usize::from(data.password_len))
        {
            *slot = ch;
            let Some(next_password_len) = data.password_len.checked_add(1) else {
                return;
            };
            data.password_len = next_password_len;
            if data.is_complete() {
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
    let Some(number) = byte
        .checked_rem(modulus)
        .and_then(|value| value.checked_add(1))
    else {
        return false;
    };
    let mask = 1_u64 << number;
    if (*seen & mask) == 0 {
        let Some(slot) = numbers.get_mut(*next_idx) else {
            return false;
        };
        *slot = number;
        *seen |= mask;
        let Some(next_index) = (*next_idx).checked_add(1) else {
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
    let bit_count = if BITS > 64 { 64 } else { BITS };
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
fn read_line_reuse<'buffer>(
    prompt: Arguments<'_>,
    buffer: &'buffer mut String,
    out: &mut dyn Write,
) -> IoRst<&'buffer str> {
    buffer.clear();
    out.write_fmt(prompt)?;
    out.flush()?;
    let bytes_read = stdin().read_line(buffer)?;
    if bytes_read == 0 {
        return Err(ioErr::new(
            ErrorKind::UnexpectedEof,
            "표준 입력이 종료되었습니다.",
        ));
    }
    Ok(buffer.trim())
}
fn read_ladder_entries<'entry, const N: usize>(
    prompt: Arguments<'_>,
    input_buffer: &mut String,
    io: (&mut dyn Write, &mut dyn Write),
    storage: &'entry mut String,
    entries: &mut [&'entry str; N],
    mode: LadderEntryMode,
    index_error: &'static str,
) -> Result<usize> {
    let (out, err) = io;
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
            count = count.checked_add(1).ok_or_else(|| {
                ioErr::other(match mode {
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
                    let part = line
                        .get(segment_start..segment_end)
                        .ok_or_else(|| ioErr::other(index_error))?;
                    let leading_whitespace = part.len().saturating_sub(part.trim_start().len());
                    let trimmed = part.trim();
                    let entry_index = count
                        .checked_sub(1)
                        .ok_or_else(|| ioErr::other(index_error))?;
                    let slot = trimmed_ranges
                        .get_mut(entry_index)
                        .ok_or_else(|| ioErr::other(index_error))?;
                    let range_start = segment_start
                        .checked_add(leading_whitespace)
                        .ok_or_else(|| ioErr::other(index_error))?;
                    let range_end = range_start
                        .checked_add(trimmed.len())
                        .ok_or_else(|| ioErr::other(index_error))?;
                    *slot = (range_start, range_end);
                }
            }
            if separator {
                segment_start = segment_end
                    .checked_add(1)
                    .ok_or_else(|| ioErr::other(index_error))?;
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
            let part = storage
                .get(range_start..range_end)
                .ok_or_else(|| ioErr::other(index_error))?;
            let slot = entries
                .get_mut(entry_index)
                .ok_or_else(|| ioErr::other(index_error))?;
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
                |line| {
                    line.parse::<f64>()
                        .ok()
                        .filter(|value| value.is_finite() && !value.is_subnormal())
                },
            )?;
            let max_value = loop {
                let value = read_parsed_value(
                    format_args!("최댓값을 입력해 주세요: "),
                    input_buffer,
                    out,
                    err,
                    "유효한 정규 실수 값을 입력해야 합니다 (NaN, 무한대, 비정규 값 제외).",
                    |line| {
                        line.parse::<f64>()
                            .ok()
                            .filter(|value| value.is_finite() && !value.is_subnormal())
                    },
                )?;
                if value >= min_value {
                    break value;
                }
                writeln!(err, "최댓값은 최솟값보다 크거나 같아야 합니다.")?;
            };
            let random_u64 = get_hardware_random()? ^ seed_modifier;
            let upper_32 = (u32::from(low_u16_from_u64(random_u64 >> 48_u32)) << 16_u32)
                | u32::from(low_u16_from_u64(random_u64 >> 32_u32));
            let lower_32 = (u32::from(low_u16_from_u64(random_u64 >> 16_u32)) << 16_u32)
                | u32::from(low_u16_from_u64(random_u64));
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
        .ok_or_else(|| ioErr::other("난수 범위 계산 실패"))?;
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
