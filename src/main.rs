#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_rdrand64_step, _rdseed64_step};
use std::{
    char::from_u32,
    error::Error,
    fs::File,
    io::{BufWriter, Error as ioErr, IsTerminal, Result as IoRst, Write, stdin, stdout},
    is_x86_feature_detected,
    path::Path,
    process::ExitCode,
    result::Result as stdResult,
    sync::{
        LazyLock, Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
        mpsc::{Receiver, SyncSender, TryRecvError, sync_channel},
    },
    thread::{ScopedJoinHandle, available_parallelism, scope, sleep},
    time::{Duration, Instant},
};
mod numeric;
mod time;
use numeric::{low_u8_from_u32, low_u8_from_u64, low_u8_from_u128, low_u16_from_u64};
#[inline(never)]
#[cold]
fn write_zero_err() -> ioErr {
    ioErr::new(
        std::io::ErrorKind::WriteZero,
        "failed to write whole buffer",
    )
}
enum RngSource {
    RdSeed,
    RdRand,
    None,
}
#[cfg(target_arch = "x86_64")]
static RNG_SOURCE: std::sync::LazyLock<RngSource> = std::sync::LazyLock::new(|| {
    if is_x86_feature_detected!("rdseed") {
        RngSource::RdSeed
    } else if is_x86_feature_detected!("rdrand") {
        eprintln!("RDSEED를 미지원하여 RDRAND를 사용합니다.");
        RngSource::RdRand
    } else {
        RngSource::None
    }
});
#[cfg(not(target_arch = "x86_64"))]
static RNG_SOURCE: std::sync::LazyLock<RngSource> = std::sync::LazyLock::new(|| RngSource::None);
static GLYPHS: [char; 16] = [
    '🌅', '🐦', '👫', '🦕', '🌘', '🎈', '⛵', '🕷', '🦋', '🌀', '🧊', '🐟', '⛺', '🚀', '🌳', '🔯',
];
static IS_TERMINAL: LazyLock<bool> = LazyLock::new(|| stdout().is_terminal());
const fn make_two_digits_table() -> [[u8; 2]; 100] {
    let mut table = [[0u8; 2]; 100];
    let mut idx = 0usize;
    let mut value = 0u8;
    while idx < 100 {
        table[idx] = [b'0' + value / 10, b'0' + value % 10];
        idx += 1;
        value += 1;
    }
    table
}
const TWO_DIGITS: [[u8; 2]; 100] = make_two_digits_table();
type Result<T> = stdResult<T, Box<dyn Error + Send + Sync + 'static>>;
const FILE_NAME: &str = "random_data.txt";
const BUFFER_SIZE: usize = 1016;
const BUFFERS_PER_WORKER: usize = 8;
type DataBuffer = Box<[u8; BUFFER_SIZE]>;
const fn bitmask_const<const B: u8>() -> u64 {
    let mut b = B as u32;
    if b > 64 {
        b = 64;
    }
    match b {
        0 => 0,
        64 => u64::MAX,
        _ => (1u64 << b) - 1,
    }
}
const fn galaxy_coord<const SUB: u16, const ADD: u16>(value: u16) -> u16 {
    let a = value.wrapping_sub(SUB);
    let b = value.wrapping_add(ADD);
    if a < b { a } else { b }
}
const U32_MAX_INV: f64 = 1.0 / (u32::MAX as f64);
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
const DIGITS: [u8; 10] = *b"0123456789";
const HEX_UPPER: [u8; 16] = *b"0123456789ABCDEF";
const fn make_bin8_table() -> [[u8; 8]; 256] {
    let mut table = [[0u8; 8]; 256];
    let mut idx = 0usize;
    let mut byte = 0u8;
    while idx < 256 {
        let mut bit = 0usize;
        while bit < 8 {
            let shift = 7 - bit;
            table[idx][bit] = if ((byte >> shift) & 1) == 1 {
                b'1'
            } else {
                b'0'
            };
            bit += 1;
        }
        idx += 1;
        byte = byte.wrapping_add(1);
    }
    table
}
const BIN8_TABLE: [[u8; 8]; 256] = make_bin8_table();
const fn make_hex_byte_table() -> [[u8; 2]; 256] {
    let mut table = [[0u8; 2]; 256];
    let mut i = 0usize;
    while i < 256 {
        let hi = (i >> 4) & 0xF;
        let lo = i & 0xF;
        table[i] = [HEX_UPPER[hi], HEX_UPPER[lo]];
        i += 1;
    }
    table
}
const HEX_BYTE_TABLE: [[u8; 2]; 256] = make_hex_byte_table();
#[cfg(target_arch = "x86_64")]
const MENU: &str = "\n1: 사다리타기 실행, 2: 무작위 숫자 생성, 3: 데이터 생성(1회), 4: 데이터 생성(여러 회), 5: 서버 시간 확인, 6: 파일 삭제, 7: num_64/supp 수동 입력 생성, 기타: 종료\n선택해 주세요: ";
#[cfg(not(target_arch = "x86_64"))]
const MENU: &str = "\n5: 서버 시간 확인, 7: num_64/supp 수동 입력 생성, 기타(1~4, 6 제외): 종료\n(참고: 이 플랫폼에서는 하드웨어 RNG 관련 기능이 비활성화됩니다)\n선택해 주세요: ";
#[derive(Default)]
struct RandomDataSet {
    num_64: u64,
    seen_lotto: u64,
    seen_lotto7: u64,
    seen_euro_millions_main: u64,
    seen_euro_millions_lucky: u64,
    kor_coords: (f64, f64),
    world_coords: (f64, f64),
    lotto_next_idx: usize,
    lotto7_next_idx: usize,
    euro_main_next_idx: usize,
    euro_lucky_next_idx: usize,
    numeric_password: u32,
    glyph_string: [char; 12],
    password: [char; 8],
    hangul_syllables: [char; 4],
    solar_system_index: u16,
    nms_portal_zzz: u16,
    nms_portal_xxx: u16,
    galaxy_x: u16,
    galaxy_y: u16,
    galaxy_z: u16,
    lotto7_numbers: [u8; 7],
    lotto_numbers: [u8; 6],
    euro_millions_main_numbers: [u8; 5],
    euro_millions_lucky_stars: [u8; 2],
    numeric_password_digits: u8,
    password_len: u8,
    planet_number: u8,
    nms_portal_yy: u8,
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
struct RandomBitBuffer {
    value: u64,
    bits_remaining: u8,
}
impl RandomBitBuffer {
    fn new() -> Result<Self> {
        Ok(Self {
            value: get_hardware_random()?,
            bits_remaining: 64,
        })
    }
    const fn from_value(value: u64) -> Self {
        Self {
            value,
            bits_remaining: 64,
        }
    }
}
type SupplementalProvider<'a> = dyn FnMut(&'static str) -> Result<RandomBitBuffer> + 'a;
fn main() -> Result<ExitCode> {
    let file_mutex = Mutex::new(open_or_create_file()?);
    #[cfg(target_arch = "x86_64")]
    let mut num_64 = process_single_random_data(&file_mutex)?;
    #[cfg(not(target_arch = "x86_64"))]
    let mut num_64: u64 = 0;
    let mut input_buffer = String::with_capacity(256);
    let menu_prompt = format_args!("{MENU}");
    loop {
        match read_line_reuse(menu_prompt, &mut input_buffer)? {
            "1" => {
                #[cfg(target_arch = "x86_64")]
                {
                    ladder_game(num_64, &mut input_buffer)?;
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    print_x86_64_only_feature_disabled();
                }
            }
            "2" => {
                #[cfg(target_arch = "x86_64")]
                {
                    println!("\n무작위 숫자 생성 타입 선택:");
                    match read_line_reuse(
                        format_args!("1: 정수 생성, 2: 실수 생성, 기타: 취소\n선택해 주세요: "),
                        &mut input_buffer,
                    )? {
                        "1" => generate_random_integer(num_64, &mut input_buffer)?,
                        "2" => generate_random_float(num_64, &mut input_buffer)?,
                        _ => println!("무작위 숫자 생성을 취소합니다."),
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    print_x86_64_only_feature_disabled();
                }
            }
            "3" => {
                #[cfg(target_arch = "x86_64")]
                {
                    ensure_file_exists_and_reopen(&file_mutex)?;
                    num_64 = process_single_random_data(&file_mutex)?;
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    print_x86_64_only_feature_disabled();
                }
            }
            "4" => {
                #[cfg(target_arch = "x86_64")]
                {
                    num_64 = regenerate_multiple(&file_mutex, &mut input_buffer)?;
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    print_x86_64_only_feature_disabled();
                }
            }
            "5" => {
                if let Err(e) = time::run() {
                    eprintln!("서버 시간 확인 중 오류 발생: {e}");
                }
            }
            "6" => {
                #[cfg(target_arch = "x86_64")]
                {
                    match std::fs::remove_file(FILE_NAME) {
                        Ok(()) => {
                            println!("파일 '{FILE_NAME}'를 삭제했습니다.");
                        }
                        Err(e) => {
                            eprintln!("{e}");
                        }
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    print_x86_64_only_feature_disabled();
                }
            }
            "7" => {
                ensure_file_exists_and_reopen(&file_mutex)?;
                num_64 = process_manual_random_data(&file_mutex, &mut input_buffer)?;
            }
            _ => return Ok(ExitCode::SUCCESS),
        }
    }
}
#[cfg(not(target_arch = "x86_64"))]
fn print_x86_64_only_feature_disabled() {
    println!("이 기능은 x86_64 전용이라 현재 플랫폼에서는 비활성화되어 있습니다.");
}
fn ensure_file_exists_and_reopen(file_mutex: &Mutex<BufWriter<File>>) -> Result<()> {
    if !Path::new(FILE_NAME).try_exists()? {
        *lock_mutex(file_mutex, "Mutex 잠금 실패 (파일 생성 시)")? = open_or_create_file()?;
    }
    Ok(())
}
fn open_or_create_file() -> Result<BufWriter<File>> {
    Ok(BufWriter::with_capacity(
        1_048_576,
        File::options().create(true).append(true).open(FILE_NAME)?,
    ))
}
fn lock_mutex<'a, T>(mutex: &'a Mutex<T>, context_msg: &'static str) -> Result<MutexGuard<'a, T>> {
    mutex.lock().map_err(|_| Box::from(context_msg))
}
fn persist_and_print_random_data(
    file_mutex: &Mutex<BufWriter<File>>,
    data: &RandomDataSet,
) -> Result<()> {
    let mut buffer = [0u8; BUFFER_SIZE];
    let file_len = format_data_into_buffer(data, &mut buffer, false)?;
    {
        let mut file_guard = lock_mutex(file_mutex, "Mutex 잠금 실패 (단일 쓰기 시)")?;
        write_buffer_to_file_guard(&mut file_guard, &buffer[..file_len])?;
        file_guard.flush()?;
    }
    if *IS_TERMINAL {
        let console_len = format_data_into_buffer(data, &mut buffer, true)?;
        write_slice_to_console(&buffer[..console_len])?;
    } else {
        write_slice_to_console(&buffer[..file_len])?;
    }
    Ok(())
}
fn process_single_random_data(file_mutex: &Mutex<BufWriter<File>>) -> Result<u64> {
    let data = generate_random_data()?;
    let num_64 = data.num_64;
    persist_and_print_random_data(file_mutex, &data)?;
    Ok(num_64)
}
fn process_manual_random_data(
    file_mutex: &Mutex<BufWriter<File>>,
    input_buffer: &mut String,
) -> Result<u64> {
    println!("\nnum_64/supp 수동 입력 생성 모드");
    let num_64 = read_parse_u64(
        format_args!(
            "num_64를 입력해 주세요 (최소값 예: 0 또는 0x0, 최대값 예: {} 또는 0x{:X}): ",
            u64::MAX,
            u64::MAX
        ),
        input_buffer,
    )?;
    let mut supp_input_count = 0usize;
    let mut next_supp = |reason: &'static str| -> Result<RandomBitBuffer> {
        supp_input_count += 1;
        let supp = read_parse_u64(
            format_args!(
                "supp 값 #{supp_input_count} 입력 ({reason}, 최소값 예: 0 또는 0x0, 최대값 예: {} 또는 0x{:X}): ",
                u64::MAX,
                u64::MAX
            ),
            input_buffer,
        )?;
        Ok(RandomBitBuffer::from_value(supp))
    };
    let data = generate_random_data_from_num(num_64, &mut next_supp)?;
    persist_and_print_random_data(file_mutex, &data)?;
    Ok(num_64)
}
fn generate_random_data() -> Result<RandomDataSet> {
    let num = get_hardware_random()?;
    let mut next_supp = |_reason: &'static str| RandomBitBuffer::new();
    generate_random_data_from_num(num, &mut next_supp)
}
fn fill_euro_lucky_stars(
    data: &mut RandomDataSet,
    num: u64,
    supplemental: &mut Option<RandomBitBuffer>,
    next_supp: &mut SupplementalProvider<'_>,
) -> Result<()> {
    let mut lucky_star_source = supplemental
        .as_ref()
        .map_or_else(|| num.reverse_bits(), |supp| supp.value.reverse_bits());
    'lucky_star_loop: loop {
        for byte in lucky_star_source.to_be_bytes() {
            if byte > 251 {
                continue;
            }
            if process_lotto_numbers(
                byte,
                12,
                &mut data.euro_millions_lucky_stars,
                &mut data.seen_euro_millions_lucky,
                &mut data.euro_lucky_next_idx,
            ) && data.euro_lucky_next_idx >= 2
            {
                break 'lucky_star_loop;
            }
        }
        if data.euro_lucky_next_idx >= 2 {
            break;
        }
        let new_supp = next_supp("유로밀리언 럭키 스타 보완")?;
        lucky_star_source = new_supp.value.reverse_bits();
        *supplemental = Some(new_supp);
    }
    Ok(())
}
fn find_hangul_candidate(supp_value: u64) -> Option<u32> {
    let bytes = supp_value.to_be_bytes();
    for chunk in bytes.chunks_exact(2) {
        let candidate = u32::from(u16::from_be_bytes([chunk[0], chunk[1]]));
        if candidate <= 55_859 {
            return Some(candidate);
        }
    }
    None
}
fn fill_hangul_syllables(
    num: u64,
    supplemental: &mut Option<RandomBitBuffer>,
    next_supp: &mut SupplementalProvider<'_>,
) -> Result<[char; 4]> {
    let mut hangul = ['\0'; 4];
    let num_bytes = num.to_be_bytes();
    for (idx, slot) in hangul.iter_mut().enumerate() {
        let byte_idx = idx * 2;
        let mut syllable_index = u32::from(u16::from_be_bytes([
            num_bytes[byte_idx],
            num_bytes[byte_idx + 1],
        ]));
        while syllable_index > 55_859 {
            if supplemental.is_none() {
                *supplemental = Some(next_supp("한글 음절 보완")?);
            }
            let Some(supp_value) = supplemental.as_ref().map(|supp| supp.value) else {
                return Err("한글 음절 보완 상태 불일치".into());
            };
            if let Some(candidate) = find_hangul_candidate(supp_value) {
                syllable_index = candidate;
            } else {
                *supplemental = Some(next_supp("한글 음절 보완 재시도")?);
            }
        }
        *slot = from_u32(0xAC00 + (syllable_index % 11_172)).ok_or("한글 음절 변환 실패")?;
    }
    Ok(hangul)
}
fn generate_random_data_from_num(
    num: u64,
    next_supp: &mut SupplementalProvider<'_>,
) -> Result<RandomDataSet> {
    let mut data = RandomDataSet {
        num_64: num,
        ..Default::default()
    };
    fill_data_fields_from_u64(num, &mut data);
    let mut supplemental: Option<RandomBitBuffer> = None;
    while !data.is_complete() {
        let new_supp = next_supp("기본 필드 보완")?;
        fill_data_fields_from_u64(new_supp.value, &mut data);
        supplemental = Some(new_supp);
    }
    fill_euro_lucky_stars(&mut data, num, &mut supplemental, next_supp)?;
    data.hangul_syllables = fill_hangul_syllables(num, &mut supplemental, next_supp)?;
    let num_bytes = num.to_be_bytes();
    let upper_32_bits =
        u32::from_be_bytes([num_bytes[0], num_bytes[1], num_bytes[2], num_bytes[3]]);
    let lower_32_bits =
        u32::from_be_bytes([num_bytes[4], num_bytes[5], num_bytes[6], num_bytes[7]]);
    let upper_ratio = f64::from(upper_32_bits) * U32_MAX_INV;
    let lower_ratio = f64::from(lower_32_bits) * U32_MAX_INV;
    data.kor_coords = (
        5.504_167_f64.mul_add(upper_ratio, 33.112_500),
        7.263_056_f64.mul_add(lower_ratio, 124.609_722),
    );
    data.world_coords = (
        180.0f64.mul_add(upper_ratio, -90.0),
        360.0f64.mul_add(lower_ratio, -180.0),
    );
    let planet_number = extract_valid_bits_for_nms::<4>(
        num,
        &[52, 4, 0],
        11,
        "NMS 행성 번호 보완",
        &mut supplemental,
        next_supp,
    )? % 6
        + 1;
    data.planet_number = low_u8_from_u64(planet_number);
    let solar_system_index = extract_valid_bits_for_nms::<12>(
        num,
        &[40],
        3834,
        "NMS 태양계 번호 보완",
        &mut supplemental,
        next_supp,
    )? % 767
        + 1;
    data.solar_system_index = low_u16_from_u64(solar_system_index);
    data.nms_portal_yy = low_u8_from_u32(upper_32_bits);
    data.nms_portal_zzz = low_u16_from_u64((num >> 20) & 0xFFF);
    data.nms_portal_xxx = low_u16_from_u64((num >> 8) & 0xFFF);
    data.galaxy_x = galaxy_coord::<0x801, 0x7FF>(data.nms_portal_xxx);
    data.galaxy_y = galaxy_coord::<0x81, 0x7F>(u16::from(data.nms_portal_yy));
    data.galaxy_z = galaxy_coord::<0x801, 0x7FF>(data.nms_portal_zzz);
    for (idx, slot) in data.glyph_string.iter_mut().enumerate() {
        let nibble_source = match idx {
            0 => u64::from(data.planet_number),
            1 => u64::from(data.solar_system_index >> 8),
            2 => u64::from(data.solar_system_index >> 4),
            3 => u64::from(data.solar_system_index),
            _ => num >> (36usize - (idx - 4) * 4),
        };
        let nibble = usize::from(low_u8_from_u64(nibble_source & 0xF));
        *slot = GLYPHS[nibble];
    }
    Ok(data)
}
fn get_hardware_random() -> Result<u64> {
    match *RNG_SOURCE {
        RngSource::RdSeed => rdseed_impl(),
        RngSource::RdRand => rdrand_impl(),
        RngSource::None => no_hw_rng(),
    }
}
#[cfg(target_arch = "x86_64")]
fn rdseed_impl() -> Result<u64> {
    let mut v = 0u64;
    for _ in 0..1_000 {
        if unsafe { _rdseed64_step(&mut v) } == 1 {
            return Ok(v);
        }
        std::hint::spin_loop();
    }
    Err("RDSEED 실패".into())
}
#[cfg(not(target_arch = "x86_64"))]
fn rdseed_impl() -> Result<u64> {
    no_hw_rng()
}
#[cfg(target_arch = "x86_64")]
fn rdrand_impl() -> Result<u64> {
    let mut v = 0u64;
    for _ in 0..10 {
        if unsafe { _rdrand64_step(&mut v) } == 1 {
            return Ok(v);
        }
        std::hint::spin_loop();
    }
    Err("RDRAND 실패".into())
}
#[cfg(not(target_arch = "x86_64"))]
fn rdrand_impl() -> Result<u64> {
    no_hw_rng()
}
fn no_hw_rng() -> Result<u64> {
    Err("RDSEED·RDRAND 모두 미지원합니다.".into())
}
fn fill_data_fields_from_u64(v: u64, data: &mut RandomDataSet) {
    for byte in v.to_be_bytes() {
        if byte > 249 {
            continue;
        }
        if data.numeric_password_digits < 6 && {
            data.numeric_password = data.numeric_password * 10 + u32::from(byte % 10);
            data.numeric_password_digits += 1;
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
            && let Some(ch) = from_u32(u32::from(byte % 94 + 33))
            && {
                data.password[usize::from(data.password_len)] = ch;
                data.password_len += 1;
                data.is_complete()
            }
        {
            return;
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
    let number = (byte % modulus) + 1;
    let mask = 1u64 << number;
    if (*seen & mask) == 0 {
        numbers[*next_idx] = number;
        *seen |= mask;
        *next_idx += 1;
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
    let mask: u64 = bitmask_const::<BITS>();
    for &shift in shifts {
        let v = (num >> shift) & mask;
        if v <= max_value {
            return Ok(v);
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
        let shift = supp.bits_remaining - BITS;
        let extracted = (supp.value >> shift) & mask;
        supp.bits_remaining = shift;
        if extracted <= max_value {
            return Ok(extracted);
        }
    }
}
fn format_data_into_buffer(
    data: &RandomDataSet,
    buffer: &mut [u8; BUFFER_SIZE],
    use_colors: bool,
) -> Result<usize> {
    Ok(format_output(buffer.as_mut_slice(), data, use_colors)?)
}
struct BufCursor<'a> {
    buf: &'a mut [u8],
    pos: usize,
}
impl<'a> BufCursor<'a> {
    const fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    const fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
    const fn written_len(&self) -> usize {
        self.pos
    }
    fn write_bytes(&mut self, bytes: &[u8]) -> IoRst<()> {
        let len = bytes.len();
        if self.remaining() < len {
            return Err(write_zero_err());
        }
        let end = self.pos + len;
        self.buf[self.pos..end].copy_from_slice(bytes);
        self.pos = end;
        Ok(())
    }
    fn write_byte(&mut self, b: u8) -> IoRst<()> {
        if self.remaining() < 1 {
            return Err(write_zero_err());
        }
        self.buf[self.pos] = b;
        self.pos += 1;
        Ok(())
    }
}
fn cursor_write_fmt(cur: &mut BufCursor<'_>, args: std::fmt::Arguments<'_>) -> IoRst<()> {
    let start = cur.pos;
    let mut slice = &mut cur.buf[start..];
    let before = slice.len();
    let res = slice.write_fmt(args);
    let after = slice.len();
    cur.pos += before - after;
    res
}
fn buf_write_bytes(cur: &mut BufCursor<'_>, bytes: &[u8]) -> IoRst<()> {
    cur.write_bytes(bytes)
}
fn buf_write_byte(cur: &mut BufCursor<'_>, byte: u8) -> IoRst<()> {
    cur.write_byte(byte)
}
fn buf_write_chars<const N: usize>(cur: &mut BufCursor<'_>, chars: &[char; N]) -> IoRst<()> {
    let mut total = 0usize;
    let mut i = 0usize;
    while i < N {
        total += chars[i].len_utf8();
        i += 1;
    }
    if cur.remaining() < total {
        return Err(write_zero_err());
    }
    let start = cur.pos;
    let end = start + total;
    let head = &mut cur.buf[start..end];
    let mut pos = 0usize;
    let mut j = 0usize;
    while j < N {
        let written = chars[j].encode_utf8(&mut head[pos..]).len();
        pos += written;
        j += 1;
    }
    cur.pos = end;
    Ok(())
}
fn buf_write_u8_dec(cur: &mut BufCursor<'_>, n: u8) -> IoRst<()> {
    if n >= 100 {
        if cur.remaining() < 3 {
            return Err(write_zero_err());
        }
        let hundreds = usize::from(n / 100);
        let rem = usize::from(n % 100);
        let start = cur.pos;
        cur.buf[start] = DIGITS[hundreds];
        cur.buf[start + 1..start + 3].copy_from_slice(&TWO_DIGITS[rem]);
        cur.pos = start + 3;
        return Ok(());
    }
    if n >= 10 {
        if cur.remaining() < 2 {
            return Err(write_zero_err());
        }
        let start = cur.pos;
        cur.buf[start..start + 2].copy_from_slice(&TWO_DIGITS[usize::from(n)]);
        cur.pos = start + 2;
        return Ok(());
    }
    if cur.remaining() < 1 {
        return Err(write_zero_err());
    }
    cur.buf[cur.pos] = b'0' + n;
    cur.pos += 1;
    Ok(())
}
fn buf_write_u8_array_spaced<const N: usize>(cur: &mut BufCursor<'_>, nums: &[u8; N]) -> IoRst<()> {
    let mut total = N.saturating_sub(1);
    let mut i = 0usize;
    while i < N {
        let n = nums[i];
        total += if n >= 100 {
            3
        } else if n >= 10 {
            2
        } else {
            1
        };
        i += 1;
    }
    if cur.remaining() < total {
        return Err(write_zero_err());
    }
    let start = cur.pos;
    let end = start + total;
    let head = &mut cur.buf[start..end];
    let mut pos = 0usize;
    let mut j = 0usize;
    while j < N {
        if j != 0 {
            head[pos] = b' ';
            pos += 1;
        }
        let n = nums[j];
        if n >= 100 {
            let hundreds = usize::from(n / 100);
            let rem = usize::from(n % 100);
            head[pos] = DIGITS[hundreds];
            head[pos + 1..pos + 3].copy_from_slice(&TWO_DIGITS[rem]);
            pos += 3;
        } else if n >= 10 {
            head[pos..pos + 2].copy_from_slice(&TWO_DIGITS[usize::from(n)]);
            pos += 2;
        } else {
            head[pos] = b'0' + n;
            pos += 1;
        }
        j += 1;
    }
    cur.pos = end;
    Ok(())
}
fn buf_write_hash_hex24_from_bytes(cur: &mut BufCursor<'_>, b0: u8, b1: u8, b2: u8) -> IoRst<()> {
    if cur.remaining() < 7 {
        return Err(write_zero_err());
    }
    let start = cur.pos;
    let head = &mut cur.buf[start..start + 7];
    head[0] = b'#';
    head[1..3].copy_from_slice(&HEX_BYTE_TABLE[usize::from(b0)]);
    head[3..5].copy_from_slice(&HEX_BYTE_TABLE[usize::from(b1)]);
    head[5..7].copy_from_slice(&HEX_BYTE_TABLE[usize::from(b2)]);
    cur.pos = start + 7;
    Ok(())
}
fn buf_write_m_hash_hex24_from_bytes(cur: &mut BufCursor<'_>, b0: u8, b1: u8, b2: u8) -> IoRst<()> {
    if cur.remaining() < 8 {
        return Err(write_zero_err());
    }
    let start = cur.pos;
    let head = &mut cur.buf[start..start + 8];
    head[0] = b'm';
    head[1] = b'#';
    head[2..4].copy_from_slice(&HEX_BYTE_TABLE[usize::from(b0)]);
    head[4..6].copy_from_slice(&HEX_BYTE_TABLE[usize::from(b1)]);
    head[6..8].copy_from_slice(&HEX_BYTE_TABLE[usize::from(b2)]);
    cur.pos = start + 8;
    Ok(())
}
fn buf_write_bin8_line(cur: &mut BufCursor<'_>, bytes: [u8; 8]) -> IoRst<()> {
    const PREFIX: &str = "2진수: ";
    const PREFIX_LEN: usize = PREFIX.len();
    const LINE_LEN: usize = PREFIX_LEN + 8 * 8 + 7 + 1;
    if cur.remaining() < LINE_LEN {
        return Err(write_zero_err());
    }
    let start = cur.pos;
    let head = &mut cur.buf[start..start + LINE_LEN];
    head[..PREFIX_LEN].copy_from_slice(PREFIX.as_bytes());
    let mut pos = PREFIX_LEN;
    let mut i = 0usize;
    while i < 8 {
        let b = usize::from(bytes[i]);
        head[pos..pos + 8].copy_from_slice(&BIN8_TABLE[b]);
        pos += 8;
        head[pos] = if i == 7 { b'\n' } else { b' ' };
        pos += 1;
        i += 1;
    }
    cur.pos = start + LINE_LEN;
    Ok(())
}
fn buf_write_hex8_line(cur: &mut BufCursor<'_>, bytes: [u8; 8]) -> IoRst<()> {
    const PREFIX: &str = "16진수: ";
    const PREFIX_LEN: usize = PREFIX.len();
    const LINE_LEN: usize = PREFIX_LEN + 8 * 2 + 7 + 1;
    if cur.remaining() < LINE_LEN {
        return Err(write_zero_err());
    }
    let start = cur.pos;
    let head = &mut cur.buf[start..start + LINE_LEN];
    head[..PREFIX_LEN].copy_from_slice(PREFIX.as_bytes());
    let mut pos = PREFIX_LEN;
    let mut i = 0usize;
    while i < 8 {
        let b = usize::from(bytes[i]);
        head[pos..pos + 2].copy_from_slice(&HEX_BYTE_TABLE[b]);
        pos += 2;
        head[pos] = if i == 7 { b'\n' } else { b' ' };
        pos += 1;
        i += 1;
    }
    cur.pos = start + LINE_LEN;
    Ok(())
}
fn buf_write_ascii8(cur: &mut BufCursor<'_>, chars: &[char; 8]) -> IoRst<()> {
    if cur.remaining() < 8 {
        return Err(write_zero_err());
    }
    let start = cur.pos;
    let head = &mut cur.buf[start..start + 8];
    let mut i = 0usize;
    while i < 8 {
        head[i] = u8::try_from(u32::from(chars[i])).map_err(|_| {
            ioErr::new(
                std::io::ErrorKind::InvalidData,
                "password contains non-ASCII character",
            )
        })?;
        i += 1;
    }
    cur.pos = start + 8;
    Ok(())
}
fn buf_write_u32_dec(cur: &mut BufCursor<'_>, mut n: u32) -> IoRst<()> {
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    while n >= 100 {
        let rem = usize::from(low_u8_from_u32(n % 100));
        n /= 100;
        i -= 2;
        tmp[i..i + 2].copy_from_slice(&TWO_DIGITS[rem]);
    }
    if n >= 10 {
        let rem = usize::from(low_u8_from_u32(n));
        i -= 2;
        tmp[i..i + 2].copy_from_slice(&TWO_DIGITS[rem]);
    } else {
        i -= 1;
        let digit = low_u8_from_u32(n);
        tmp[i] = b'0' + digit;
    }
    buf_write_bytes(cur, &tmp[i..])
}
fn buf_write_u64_dec(cur: &mut BufCursor<'_>, mut n: u64) -> IoRst<()> {
    let mut tmp = [0u8; 20];
    let mut i = tmp.len();
    while n >= 100 {
        let rem = usize::from(low_u8_from_u64(n % 100));
        n /= 100;
        i -= 2;
        tmp[i..i + 2].copy_from_slice(&TWO_DIGITS[rem]);
    }
    if n >= 10 {
        let rem = usize::from(low_u8_from_u64(n));
        i -= 2;
        tmp[i..i + 2].copy_from_slice(&TWO_DIGITS[rem]);
    } else {
        i -= 1;
        let digit = low_u8_from_u64(n);
        tmp[i] = b'0' + digit;
    }
    buf_write_bytes(cur, &tmp[i..])
}
fn buf_write_i64_dec(cur: &mut BufCursor<'_>, n: i64) -> IoRst<()> {
    if n < 0 {
        buf_write_byte(cur, b'-')?;
        let abs = if n == i64::MIN {
            i64::MAX.cast_unsigned() + 1
        } else {
            (-n).cast_unsigned()
        };
        buf_write_u64_dec(cur, abs)
    } else {
        buf_write_u64_dec(cur, n.cast_unsigned())
    }
}
fn buf_write_u32_dec_0pad_6(cur: &mut BufCursor<'_>, n: u32) -> IoRst<()> {
    if n >= 1_000_000 {
        return buf_write_u32_dec(cur, n);
    }
    let hi = usize::from(low_u8_from_u32(n / 10_000));
    let rem = usize::from(low_u16_from_u64(u64::from(n % 10_000)));
    let mid = rem / 100;
    let lo = rem % 100;
    if cur.remaining() < 6 {
        return Err(write_zero_err());
    }
    let start = cur.pos;
    let head = &mut cur.buf[start..start + 6];
    head[0..2].copy_from_slice(&TWO_DIGITS[hi]);
    head[2..4].copy_from_slice(&TWO_DIGITS[mid]);
    head[4..6].copy_from_slice(&TWO_DIGITS[lo]);
    cur.pos = start + 6;
    Ok(())
}
fn buf_write_u64_octal(cur: &mut BufCursor<'_>, mut n: u64) -> IoRst<()> {
    if n == 0 {
        return buf_write_byte(cur, b'0');
    }
    let mut tmp = [0u8; 22];
    let mut i = tmp.len();
    while n != 0 {
        i -= 1;
        let oct_digit = low_u8_from_u64(n & 7);
        tmp[i] = b'0' + oct_digit;
        n >>= 3;
    }
    buf_write_bytes(cur, &tmp[i..])
}
fn buf_write_hex_u16_0pad4(cur: &mut BufCursor<'_>, v: u16) -> IoRst<()> {
    if cur.remaining() < 4 {
        return Err(write_zero_err());
    }
    let start = cur.pos;
    let head = &mut cur.buf[start..start + 4];
    let [upper_byte, lower_byte] = v.to_be_bytes();
    let upper = usize::from(upper_byte);
    let lower = usize::from(lower_byte);
    head[0..2].copy_from_slice(&HEX_BYTE_TABLE[upper]);
    head[2..4].copy_from_slice(&HEX_BYTE_TABLE[lower]);
    cur.pos = start + 4;
    Ok(())
}
fn buf_write_hex_u16_min3(cur: &mut BufCursor<'_>, v: u16) -> IoRst<()> {
    if v < 0x1000 {
        if cur.remaining() < 3 {
            return Err(write_zero_err());
        }
        let start = cur.pos;
        let head = &mut cur.buf[start..start + 3];
        let [hi_byte, lo_byte] = v.to_be_bytes();
        let hi = usize::from(hi_byte);
        let lo = usize::from(lo_byte);
        head[0] = HEX_UPPER[hi];
        head[1..3].copy_from_slice(&HEX_BYTE_TABLE[lo]);
        cur.pos = start + 3;
        Ok(())
    } else {
        buf_write_hex_u16_0pad4(cur, v)
    }
}
fn format_output(buf: &mut [u8], data: &RandomDataSet, use_colors: bool) -> IoRst<usize> {
    let mut cur = BufCursor::new(buf);
    let v = data.num_64;
    let bytes = v.to_be_bytes();
    let [b0, b1, b2, b3, b4, b5, _, _] = bytes;
    buf_write_bytes(&mut cur, "64비트 난수: ".as_bytes())?;
    buf_write_u64_dec(&mut cur, v)?;
    buf_write_bytes(&mut cur, " (유부호 정수: ".as_bytes())?;
    buf_write_i64_dec(&mut cur, v.cast_signed())?;
    buf_write_bytes(&mut cur, b")\n")?;
    buf_write_bin8_line(&mut cur, bytes)?;
    buf_write_bytes(&mut cur, "8진수: ".as_bytes())?;
    buf_write_u64_octal(&mut cur, v)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_hex8_line(&mut cur, bytes)?;
    buf_write_bytes(&mut cur, "Hex 코드: ".as_bytes())?;
    if use_colors {
        buf_write_bytes(&mut cur, b"\x1B[38;2;")?;
        buf_write_u8_dec(&mut cur, b0)?;
        buf_write_byte(&mut cur, b';')?;
        buf_write_u8_dec(&mut cur, b1)?;
        buf_write_byte(&mut cur, b';')?;
        buf_write_u8_dec(&mut cur, b2)?;
        buf_write_m_hash_hex24_from_bytes(&mut cur, b0, b1, b2)?;
        buf_write_bytes(&mut cur, b"\x1B[0m \x1B[38;2;")?;
        buf_write_u8_dec(&mut cur, b3)?;
        buf_write_byte(&mut cur, b';')?;
        buf_write_u8_dec(&mut cur, b4)?;
        buf_write_byte(&mut cur, b';')?;
        buf_write_u8_dec(&mut cur, b5)?;
        buf_write_m_hash_hex24_from_bytes(&mut cur, b3, b4, b5)?;
        buf_write_bytes(&mut cur, b"\x1B[0m")?;
    } else {
        buf_write_hash_hex24_from_bytes(&mut cur, b0, b1, b2)?;
        buf_write_byte(&mut cur, b' ')?;
        buf_write_hash_hex24_from_bytes(&mut cur, b3, b4, b5)?;
    }
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "바이트 배열: ".as_bytes())?;
    buf_write_u8_array_spaced(&mut cur, &bytes)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "6자리 숫자 비밀번호: ".as_bytes())?;
    buf_write_u32_dec_0pad_6(&mut cur, data.numeric_password)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "8자리 비밀번호: ".as_bytes())?;
    buf_write_ascii8(&mut cur, &data.password)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "로또 번호: ".as_bytes())?;
    buf_write_u8_array_spaced(&mut cur, &data.lotto_numbers)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "일본 로또 7 번호: ".as_bytes())?;
    buf_write_u8_array_spaced(&mut cur, &data.lotto7_numbers)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "유로밀리언 번호: ".as_bytes())?;
    buf_write_u8_array_spaced(&mut cur, &data.euro_millions_main_numbers)?;
    buf_write_bytes(&mut cur, b" + ")?;
    buf_write_u8_array_spaced(&mut cur, &data.euro_millions_lucky_stars)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "한글 음절 4글자: ".as_bytes())?;
    buf_write_chars(&mut cur, &data.hangul_syllables)?;
    buf_write_byte(&mut cur, b'\n')?;
    cursor_write_fmt(
        &mut cur,
        format_args!(
            "대한민국 위경도: {}, {}\n세계 위경도: {}, {}\n",
            data.kor_coords.0, data.kor_coords.1, data.world_coords.0, data.world_coords.1
        ),
    )?;
    buf_write_bytes(&mut cur, "NMS 은하 번호: ".as_bytes())?;
    buf_write_u32_dec(&mut cur, u32::from(u16::from(b0).wrapping_add(1)))?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "NMS 포탈 주소: ".as_bytes())?;
    buf_write_u8_dec(&mut cur, data.planet_number)?;
    buf_write_byte(&mut cur, b' ')?;
    buf_write_hex_u16_min3(&mut cur, data.solar_system_index)?;
    buf_write_byte(&mut cur, b' ')?;
    buf_write_bytes(&mut cur, &HEX_BYTE_TABLE[usize::from(data.nms_portal_yy)])?;
    buf_write_byte(&mut cur, b' ')?;
    buf_write_hex_u16_min3(&mut cur, data.nms_portal_zzz)?;
    buf_write_byte(&mut cur, b' ')?;
    buf_write_hex_u16_min3(&mut cur, data.nms_portal_xxx)?;
    buf_write_byte(&mut cur, b' ')?;
    buf_write_byte(&mut cur, b'(')?;
    buf_write_chars(&mut cur, &data.glyph_string)?;
    buf_write_bytes(&mut cur, b")\n")?;
    buf_write_bytes(&mut cur, "NMS 은하 좌표: ".as_bytes())?;
    buf_write_hex_u16_0pad4(&mut cur, data.galaxy_x)?;
    buf_write_byte(&mut cur, b':')?;
    buf_write_hex_u16_0pad4(&mut cur, data.galaxy_y)?;
    buf_write_byte(&mut cur, b':')?;
    buf_write_hex_u16_0pad4(&mut cur, data.galaxy_z)?;
    buf_write_byte(&mut cur, b':')?;
    buf_write_hex_u16_0pad4(&mut cur, data.solar_system_index)?;
    Ok(cur.written_len())
}
fn write_buffer_to_file_guard(
    file_guard: &mut MutexGuard<BufWriter<File>>,
    buffer: &[u8],
) -> stdResult<(), ioErr> {
    file_guard.write_all(buffer)?;
    file_guard.write_all(b"\n")?;
    Ok(())
}
fn write_slice_to_console(data_slice: &[u8]) -> IoRst<()> {
    let mut stdout_lock = stdout().lock();
    stdout_lock.write_all(data_slice)?;
    stdout_lock.write_all(b"\n")?;
    stdout_lock.flush()?;
    Ok(())
}
fn read_line_reuse<'a>(prompt: std::fmt::Arguments, buffer: &'a mut String) -> IoRst<&'a str> {
    buffer.clear();
    {
        let mut out = stdout().lock();
        out.write_fmt(prompt)?;
        out.flush()?;
    }
    stdin().read_line(buffer)?;
    Ok(buffer.trim())
}
fn ladder_game(num_64: u64, player_input_buffer: &mut String) -> Result<()> {
    const MAX_PLAYERS: usize = 512;
    let mut players_storage = String::with_capacity(256);
    let mut players_array: [&str; MAX_PLAYERS] = [""; MAX_PLAYERS];
    let players_prompt =
        format_args!("\n사다리타기 플레이어를 입력해 주세요 (쉼표(,)로 구분, 2~512명): ");
    let n: usize = loop {
        let line = read_line_reuse(players_prompt, player_input_buffer)?;
        let mut count = 0usize;
        for _ in line.split(',') {
            count += 1;
            if count > MAX_PLAYERS {
                eprintln!("플레이어 수가 최대 {MAX_PLAYERS}명을 초과했습니다.");
                count = 0;
                break;
            }
        }
        if count == 0 {
            continue;
        }
        if count < 2 {
            eprintln!("플레이어 수는 최소 2명이어야 합니다.");
            continue;
        }
        players_storage.clear();
        players_storage.push_str(line);
        break count;
    };
    for (i, part) in players_storage.split(',').enumerate() {
        players_array[i] = part.trim();
    }
    let mut result_input_buffer = String::with_capacity(256);
    let mut results_storage = String::with_capacity(256);
    let mut results_array: [&str; MAX_PLAYERS] = [""; MAX_PLAYERS];
    let result_prompt =
        format_args!("사다리타기 결과값을 입력해 주세요 (쉼표(,)로 구분, {n}개 필요): ");
    loop {
        let line = read_line_reuse(result_prompt, &mut result_input_buffer)?;
        let mut count = 0usize;
        for _ in line.split(',') {
            count += 1;
            if count > n {
                break;
            }
        }
        if count != n {
            eprintln!("결과값의 개수({count})가 플레이어 수({n})와 일치하지 않습니다.\n");
            continue;
        }
        results_storage.clear();
        results_storage.push_str(line);
        break;
    }
    for (i, part) in results_storage.split(',').enumerate() {
        results_array[i] = part.trim();
    }
    println!("사다리타기 결과:");
    let indices_slice = &mut [0usize; MAX_PLAYERS][..n];
    for (i, slot) in indices_slice.iter_mut().enumerate() {
        *slot = i;
    }
    let mut current_seed = num_64;
    for i in (1..n).rev() {
        current_seed ^= get_hardware_random()?;
        let upper_bound = u64::try_from(i + 1).map_err(|_| "인덱스 상한 변환 실패")?;
        let swap_index_u64 = random_bounded(upper_bound, current_seed)?;
        let swap_index = usize::try_from(swap_index_u64).map_err(|_| "인덱스 변환 실패")?;
        indices_slice.swap(i, swap_index);
    }
    for (player, &result_index) in players_array[..n].iter().zip(indices_slice.iter()) {
        println!("{player} -> {result}", result = results_array[result_index]);
    }
    Ok(())
}
fn generate_random_integer(seed_modifier: u64, input_buffer: &mut String) -> Result<()> {
    const MIN_ALLOWED_VALUE: i64 = i64::MIN + 1;
    println!(
        "\n무작위 정수 생성기(지원 범위: {MIN_ALLOWED_VALUE} ~ {})",
        i64::MAX
    );
    let min_value = loop {
        let n = read_parse_i64(
            format_args!("최솟값을 입력해 주세요 ({MIN_ALLOWED_VALUE} 이상): "),
            input_buffer,
        )?;
        if n >= MIN_ALLOWED_VALUE {
            break n;
        }
        eprintln!("{MIN_ALLOWED_VALUE} 이상의 값을 입력해 주세요.\n");
    };
    let max_value = loop {
        let n = read_parse_i64(format_args!("최댓값을 입력해 주세요: "), input_buffer)?;
        if n >= min_value {
            break n;
        }
        eprintln!("최댓값은 최솟값보다 크거나 같아야 합니다.\n");
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
    println!("무작위 정수({min_value} ~ {max_value}): {result} (0x{result:X})");
    Ok(())
}
fn read_parse_i64(prompt: std::fmt::Arguments, buffer: &mut String) -> Result<i64> {
    loop {
        match read_line_reuse(prompt, buffer)?.parse::<i64>() {
            Ok(n) => return Ok(n),
            Err(_) => eprintln!("유효한 정수 형식이 아닙니다.\n"),
        }
    }
}
fn read_parse_u64(prompt: std::fmt::Arguments, buffer: &mut String) -> Result<u64> {
    loop {
        let raw = read_line_reuse(prompt, buffer)?;
        let parsed = raw
            .strip_prefix("0x")
            .or_else(|| raw.strip_prefix("0X"))
            .map_or_else(|| raw.parse::<u64>(), |hex| u64::from_str_radix(hex, 16));
        match parsed {
            Ok(n) => return Ok(n),
            Err(_) => {
                eprintln!(
                    "유효한 u64 형식이 아닙니다 (최소값 예: 0 또는 0x0, 최대값 예: {} 또는 0x{:X}).\n",
                    u64::MAX,
                    u64::MAX
                );
            }
        }
    }
}
fn unit_f64_from_u64(random_bits: u64) -> f64 {
    let bytes = random_bits.to_be_bytes();
    let upper_32 = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let lower_32 = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    f64::from(upper_32).mul_add(TWO_POW_32_F64, f64::from(lower_32)) * U64_UNIT_SCALE
}
fn generate_random_float(seed_modifier: u64, input_buffer: &mut String) -> Result<()> {
    println!("\n무작위 실수 생성기");
    let fmin_prompt = format_args!("최솟값을 입력해 주세요: ");
    let fmax_prompt = format_args!("최댓값을 입력해 주세요: ");
    let min_value: f64 = read_parse_f64(fmin_prompt, input_buffer)?;
    let max_value: f64 = loop {
        let num = read_parse_f64(fmax_prompt, input_buffer)?;
        if num >= min_value {
            break num;
        }
        eprintln!("최댓값은 최솟값보다 크거나 같아야 합니다.\n");
    };
    let random_u64 = get_hardware_random()? ^ seed_modifier;
    let scale = unit_f64_from_u64(random_u64);
    let result = if min_value.to_bits() == max_value.to_bits() {
        min_value
    } else {
        scale.mul_add(max_value - min_value, min_value)
    };
    println!("무작위 실수({min_value} ~ {max_value}): {result}");
    Ok(())
}
fn read_parse_f64(prompt: std::fmt::Arguments, buffer: &mut String) -> Result<f64> {
    loop {
        match read_line_reuse(prompt, buffer)?.parse::<f64>() {
            Ok(n) if n.is_finite() && !n.is_subnormal() => return Ok(n),
            _ => {
                eprintln!("유효한 정규 실수 값을 입력해야 합니다 (NaN, 무한대, 비정규 값 제외).\n");
            }
        }
    }
}
fn random_bounded(s: u64, seed_mod: u64) -> Result<u64> {
    let threshold = s.wrapping_neg() % s;
    let s128 = u128::from(s);
    loop {
        let m = u128::from(get_hardware_random()? ^ seed_mod) * s128;
        let bytes = m.to_le_bytes();
        let low_bits = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        if low_bits >= threshold {
            let high_bits = u64::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                bytes[15],
            ]);
            return Ok(high_bits);
        }
    }
}
fn writer_thread_main(
    file_mutex: &Mutex<BufWriter<File>>,
    receiver: &Receiver<(DataBuffer, usize, usize)>,
    buffer_return_senders: &[SyncSender<DataBuffer>],
) -> Result<RandomDataSet> {
    let mut file_guard = lock_mutex(file_mutex, "Mutex 잠금 실패 (쓰기 스레드)")?;
    while let Ok((data_buffer, data_len, worker_idx)) = receiver.recv() {
        write_buffer_to_file_guard(&mut file_guard, &data_buffer[..data_len])?;
        let _ = buffer_return_senders[worker_idx].send(data_buffer);
        while let Ok((more_buffer, more_len, more_worker_idx)) = receiver.try_recv() {
            write_buffer_to_file_guard(&mut file_guard, &more_buffer[..more_len])?;
            let _ = buffer_return_senders[more_worker_idx].send(more_buffer);
        }
    }
    drop(file_guard);
    let final_data = generate_random_data()?;
    let mut final_buffer_file = [0u8; BUFFER_SIZE];
    let final_bytes_written_file =
        format_data_into_buffer(&final_data, &mut final_buffer_file, false)?;
    {
        let mut file_guard = lock_mutex(file_mutex, "Mutex 잠금 실패 (쓰기 스레드 종료 처리)")?;
        write_buffer_to_file_guard(
            &mut file_guard,
            &final_buffer_file[..final_bytes_written_file],
        )?;
        file_guard.flush()?;
    }
    Ok(final_data)
}
fn progress_thread_main(
    processed_ref: &AtomicU64,
    multi_thread_count: u64,
    requested_count: u64,
    start_time: &Instant,
) -> Result<()> {
    let mut elapsed_buf = [0u8; 7];
    let mut eta_buf = [0u8; 7];
    loop {
        let processed_now = processed_ref.load(Ordering::Relaxed);
        if processed_now >= multi_thread_count {
            break;
        }
        print_progress(
            processed_now,
            requested_count,
            start_time,
            &mut elapsed_buf,
            &mut eta_buf,
        )?;
        sleep(Duration::from_millis(100));
    }
    Ok(())
}
fn worker_thread_main(
    worker_idx: usize,
    return_rx: &Receiver<DataBuffer>,
    sender_clone: &SyncSender<(DataBuffer, usize, usize)>,
    processed_ref: &AtomicU64,
    failed_ref: &AtomicU64,
    base_count: u64,
    remainder: u64,
) {
    let Ok(worker_idx_u64) = u64::try_from(worker_idx) else {
        return;
    };
    let loop_count = base_count + u64::from(worker_idx_u64 < remainder);
    let mut local_pool = Vec::with_capacity(BUFFERS_PER_WORKER);
    for _ in 0..BUFFERS_PER_WORKER {
        match return_rx.try_recv() {
            Ok(buffer) => local_pool.push(buffer),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => return,
        }
    }
    for _ in 0..loop_count {
        let mut buffer = match local_pool.pop() {
            Some(buf) => buf,
            None => match return_rx.recv() {
                Ok(buf) => buf,
                Err(_) => break,
            },
        };
        let len = if let Ok(data) = generate_random_data() {
            if let Ok(len) = format_data_into_buffer(&data, buffer.as_mut(), false) {
                len
            } else {
                failed_ref.fetch_add(1, Ordering::Relaxed);
                processed_ref.fetch_add(1, Ordering::Relaxed);
                local_pool.push(buffer);
                continue;
            }
        } else {
            failed_ref.fetch_add(1, Ordering::Relaxed);
            processed_ref.fetch_add(1, Ordering::Relaxed);
            local_pool.push(buffer);
            continue;
        };
        match sender_clone.send((buffer, len, worker_idx)) {
            Ok(()) => {
                processed_ref.fetch_add(1, Ordering::Relaxed);
                while local_pool.len() < BUFFERS_PER_WORKER {
                    match return_rx.try_recv() {
                        Ok(buf) => local_pool.push(buf),
                        Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                    }
                }
            }
            Err(send_err) => {
                failed_ref.fetch_add(1, Ordering::Relaxed);
                processed_ref.fetch_add(1, Ordering::Relaxed);
                let (_returned_buffer, _, _) = send_err.0;
                break;
            }
        }
    }
}
fn read_requested_count(input_buffer: &mut String) -> Result<u64> {
    let count_prompt = format_args!("\n생성할 데이터 개수를 입력해 주세요: ");
    loop {
        match read_line_reuse(count_prompt, input_buffer)?.parse::<u64>() {
            Ok(0) => eprintln!("1 이상의 값을 입력해 주세요."),
            Ok(n) => return Ok(n),
            Err(_) => eprintln!("유효한 숫자를 입력해 주세요."),
        }
    }
}
fn print_regenerate_summary(requested_count: u64, failed_count: u64) {
    let success_count = requested_count.saturating_sub(failed_count);
    if failed_count > 0 {
        eprintln!("[경고] 생성 중 {failed_count}건의 실패가 발생했습니다.");
        println!(
            "\n총 {requested_count}개 요청 중 {success_count}개 데이터 생성 완료 ({FILE_NAME} 저장됨).\n"
        );
    } else {
        println!("\n총 {requested_count}개의 데이터 생성 완료 ({FILE_NAME} 저장됨).\n");
    }
}
fn regenerate_multiple(
    file_mutex: &Mutex<BufWriter<File>>,
    input_buffer: &mut String,
) -> Result<u64> {
    let requested_count = read_requested_count(input_buffer)?;
    if requested_count == 1 {
        ensure_file_exists_and_reopen(file_mutex)?;
        return process_single_random_data(file_mutex);
    }
    ensure_file_exists_and_reopen(file_mutex)?;
    let multi_thread_count = requested_count.saturating_sub(1);
    let max_threads = available_parallelism().map_or(4, std::num::NonZero::get);
    let calculated_thread_count = usize::try_from(multi_thread_count).map_or(max_threads, |n| {
        if n < max_threads { n } else { max_threads }
    });
    let start_time = Instant::now();
    let in_flight_buffers = calculated_thread_count.saturating_mul(BUFFERS_PER_WORKER);
    let (sender, receiver) = sync_channel::<(DataBuffer, usize, usize)>(in_flight_buffers);
    let mut buffer_return_senders = Vec::with_capacity(calculated_thread_count);
    let mut buffer_return_receivers = Vec::with_capacity(calculated_thread_count);
    for _ in 0..calculated_thread_count {
        let (tx, rx) = sync_channel::<DataBuffer>(BUFFERS_PER_WORKER);
        for _ in 0..BUFFERS_PER_WORKER {
            tx.send(Box::new([0u8; BUFFER_SIZE]))?;
        }
        buffer_return_senders.push(tx);
        buffer_return_receivers.push(rx);
    }
    let processed = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let final_data = scope(|s| -> Result<RandomDataSet> {
        let buffer_return_senders = buffer_return_senders;
        let writer_thread =
            s.spawn(move || writer_thread_main(file_mutex, &receiver, &buffer_return_senders));
        let processed_ref = &processed;
        let failed_ref = &failed;
        let progress_thread: Option<ScopedJoinHandle<Result<()>>> = if *IS_TERMINAL {
            Some(s.spawn(move || {
                progress_thread_main(
                    processed_ref,
                    multi_thread_count,
                    requested_count,
                    &start_time,
                )
            }))
        } else {
            None
        };
        let thread_count_u64 =
            u64::try_from(calculated_thread_count).map_err(|_| "스레드 수 변환 실패")?;
        let base_count = multi_thread_count / thread_count_u64;
        let remainder = multi_thread_count % thread_count_u64;
        let mut worker_handles = Vec::with_capacity(calculated_thread_count);
        for (i, return_rx) in buffer_return_receivers.into_iter().enumerate() {
            let sender_clone = sender.clone();
            worker_handles.push(s.spawn(move || {
                worker_thread_main(
                    i,
                    &return_rx,
                    &sender_clone,
                    processed_ref,
                    failed_ref,
                    base_count,
                    remainder,
                );
            }));
        }
        drop(sender);
        for handle in worker_handles {
            handle
                .join()
                .map_err(|_| ioErr::other("작업 스레드 패닉 발생"))?;
        }
        let processed_now = processed_ref.load(Ordering::Relaxed);
        if processed_now < multi_thread_count {
            let missing = multi_thread_count - processed_now;
            failed_ref.fetch_add(missing, Ordering::Relaxed);
            processed_ref.store(multi_thread_count, Ordering::Relaxed);
        }
        if let Some(handle) = progress_thread {
            join_thread(handle, "진행률 스레드 패닉 발생")?;
        }
        join_thread(writer_thread, "쓰기 스레드 패닉 발생")
    })?;
    let mut elapsed_buf = [0u8; 7];
    let mut eta_buf = [0u8; 7];
    print_progress(
        requested_count,
        requested_count,
        &start_time,
        &mut elapsed_buf,
        &mut eta_buf,
    )?;
    let failed_count = failed.load(Ordering::Relaxed);
    print_regenerate_summary(requested_count, failed_count);
    stdout().flush()?;
    let mut buffer = [0u8; BUFFER_SIZE];
    let bytes_written = format_data_into_buffer(&final_data, &mut buffer, *IS_TERMINAL)?;
    write_slice_to_console(&buffer[..bytes_written])?;
    Ok(final_data.num_64)
}
fn print_progress(
    completed: u64,
    total: u64,
    start_time: &Instant,
    elapsed_buf: &mut [u8],
    eta_buf: &mut [u8],
) -> Result<()> {
    if !*IS_TERMINAL {
        return Ok(());
    }
    let elapsed_millis = start_time.elapsed().as_millis();
    let elapsed_deci = elapsed_millis / 100;
    let eta_deci = if total == 0 || completed >= total {
        Some(0)
    } else if completed == 0 {
        None
    } else {
        Some(
            elapsed_millis.saturating_mul(u128::from(total - completed))
                / (u128::from(completed) * 100),
        )
    };
    let elapsed_len = format_time_into(Some(elapsed_deci), elapsed_buf);
    let eta_len = format_time_into(eta_deci, eta_buf);
    let bar_width_u64 = BAR_WIDTH_U64;
    let filled_u64 = if total == 0 {
        bar_width_u64
    } else {
        completed.saturating_mul(bar_width_u64) / total
    };
    let filled = usize::from(low_u8_from_u64(filled_u64.min(bar_width_u64)));
    let bar = BAR_FULL[filled];
    let percent_u64 = if total == 0 {
        100
    } else {
        completed.saturating_mul(100) / total
    };
    let percent = low_u8_from_u64(percent_u64.min(100));
    let mut line = [0u8; 128];
    let mut cur = BufCursor::new(&mut line);
    buf_write_byte(&mut cur, b'\r')?;
    buf_write_bytes(&mut cur, bar.as_bytes())?;
    buf_write_byte(&mut cur, b' ')?;
    if percent < 10 {
        buf_write_bytes(&mut cur, b"  ")?;
    } else if percent < 100 {
        buf_write_byte(&mut cur, b' ')?;
    }
    buf_write_u8_dec(&mut cur, percent)?;
    buf_write_byte(&mut cur, b'%')?;
    buf_write_bytes(&mut cur, b" (")?;
    buf_write_u64_dec(&mut cur, completed)?;
    buf_write_byte(&mut cur, b'/')?;
    buf_write_u64_dec(&mut cur, total)?;
    buf_write_bytes(&mut cur, ") | 소요: ".as_bytes())?;
    buf_write_bytes(&mut cur, &elapsed_buf[..elapsed_len])?;
    buf_write_bytes(&mut cur, b" | ETA: ")?;
    buf_write_bytes(&mut cur, &eta_buf[..eta_len])?;
    buf_write_bytes(&mut cur, b" \x1b[K")?;
    let used = cur.written_len();
    let mut out = stdout().lock();
    out.write_all(&line[..used])?;
    out.flush()?;
    Ok(())
}
fn format_time_into(deci_seconds: Option<u128>, buf: &mut [u8]) -> usize {
    let Some(deci) = deci_seconds else {
        buf[..7].copy_from_slice(INVALID_TIME);
        return 7;
    };
    let minutes = usize::from(low_u8_from_u128((deci / 600).min(99)));
    let sec_whole = usize::from(low_u8_from_u128((deci / 10) % 60));
    let tenths = usize::from(low_u8_from_u128(deci % 10));
    buf[0..2].copy_from_slice(&TWO_DIGITS[minutes]);
    buf[2] = b':';
    buf[3..5].copy_from_slice(&TWO_DIGITS[sec_whole]);
    buf[5] = b'.';
    buf[6] = DIGITS[tenths];
    7
}
fn join_thread<T>(handle: ScopedJoinHandle<'_, Result<T>>, panic_msg: &'static str) -> Result<T> {
    handle.join().map_err(|_| ioErr::other(panic_msg))?
}
