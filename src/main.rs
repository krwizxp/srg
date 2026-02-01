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
        mpsc::{TryRecvError, sync_channel},
    },
    thread::{ScopedJoinHandle, available_parallelism, scope, sleep},
    time::{Duration, Instant},
};
mod time;
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
    let mut i = 0usize;
    while i < 100 {
        let tens = (i as u8) / 10;
        let ones = (i as u8) % 10;
        table[i] = [b'0' + tens, b'0' + ones];
        i += 1;
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
const BAR_WIDTH: usize = 10;
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
    let mut i = 0usize;
    while i < 256 {
        let mut bit = 0usize;
        while bit < 8 {
            let shift = 7 - bit;
            table[i][bit] = if ((i as u8) >> shift) & 1 == 1 {
                b'1'
            } else {
                b'0'
            };
            bit += 1;
        }
        i += 1;
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
const MENU: &str = "\n1: 사다리타기 실행, 2: 무작위 숫자 생성, 3: 데이터 생성(1회), 4: 데이터 생성(여러 회), 5: 서버 시간 확인, 6: 파일 삭제, 기타: 종료\n선택해 주세요: ";
#[cfg(not(target_arch = "x86_64"))]
const MENU: &str = "\n5: 서버 시간 확인, 기타(1~4, 6 제외): 종료\n(참고: 이 플랫폼에서는 하드웨어 RNG 관련 기능이 비활성화됩니다)\n선택해 주세요: ";
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
    fn is_complete(&self) -> bool {
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
}
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
                        Ok(_) => {
                            println!("파일 '{FILE_NAME}'를 삭제했습니다.")
                        }
                        Err(e) => {
                            eprintln!("{e}")
                        }
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    print_x86_64_only_feature_disabled();
                }
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
        *lock_mutex(file_mutex, "Mutex 잠금 실패 (파일 생성 시)")? = open_or_create_file()?
    }
    Ok(())
}
fn open_or_create_file() -> Result<BufWriter<File>> {
    Ok(BufWriter::with_capacity(
        1048576,
        File::options().create(true).append(true).open(FILE_NAME)?,
    ))
}
fn lock_mutex<'a, T>(mutex: &'a Mutex<T>, context_msg: &'static str) -> Result<MutexGuard<'a, T>> {
    mutex.lock().map_err(|_| Box::from(context_msg))
}
fn process_single_random_data(file_mutex: &Mutex<BufWriter<File>>) -> Result<u64> {
    let data = generate_random_data()?;
    let num_64 = data.num_64;
    let mut buffer = [0u8; BUFFER_SIZE];
    let file_len = format_data_into_buffer(&data, &mut buffer, false)?;
    {
        let mut file_guard = lock_mutex(file_mutex, "Mutex 잠금 실패 (단일 쓰기 시)")?;
        write_buffer_to_file_guard(&mut file_guard, &buffer[..file_len])?;
        file_guard.flush()?
    }
    if *IS_TERMINAL {
        let console_len = format_data_into_buffer(&data, &mut buffer, true)?;
        write_slice_to_console(&buffer[..console_len])?
    } else {
        write_slice_to_console(&buffer[..file_len])?
    }
    Ok(num_64)
}
fn generate_random_data() -> Result<RandomDataSet> {
    let num = get_hardware_random()?;
    let mut data = RandomDataSet {
        num_64: num,
        ..Default::default()
    };
    fill_data_fields_from_u64(num, &mut data);
    let mut supplemental: Option<RandomBitBuffer> = None;
    while !data.is_complete() {
        let new_supp = RandomBitBuffer::new()?;
        fill_data_fields_from_u64(new_supp.value, &mut data);
        supplemental = Some(new_supp)
    }
    let mut lucky_star_source = if let Some(ref supp) = supplemental {
        supp.value.reverse_bits()
    } else {
        num.reverse_bits()
    };
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
        if data.euro_lucky_next_idx < 2 {
            let new_supp = RandomBitBuffer::new()?;
            lucky_star_source = new_supp.value.reverse_bits();
            supplemental = Some(new_supp)
        } else {
            break;
        }
    }
    let mut hangul = ['\0'; 4];
    for (i, slot) in hangul.iter_mut().enumerate() {
        let mut index = ((num >> (48 - 16 * i)) & 0xFFFF) as u32;
        while index > 55_859 {
            if supplemental.is_none() {
                supplemental = Some(RandomBitBuffer::new()?);
            }
            let supp_value = supplemental
                .as_ref()
                .expect("supplemental must be Some")
                .value;
            let c0 = ((supp_value >> 48) & 0xFFFF) as u32;
            if c0 <= 55_859 {
                index = c0;
                break;
            }
            let c1 = ((supp_value >> 32) & 0xFFFF) as u32;
            if c1 <= 55_859 {
                index = c1;
                break;
            }
            let c2 = ((supp_value >> 16) & 0xFFFF) as u32;
            if c2 <= 55_859 {
                index = c2;
                break;
            }
            let c3 = (supp_value & 0xFFFF) as u32;
            if c3 <= 55_859 {
                index = c3;
                break;
            }
            supplemental = Some(RandomBitBuffer::new()?);
        }
        *slot = from_u32(0xAC00 + (index % 11_172)).ok_or("한글 음절 변환 실패")?;
    }
    data.hangul_syllables = hangul;
    let upper_32_bits = num >> 32;
    let upper_ratio = (upper_32_bits as f64) * U32_MAX_INV;
    let lower_ratio = ((num & 0xFFFF_FFFF) as f64) * U32_MAX_INV;
    data.kor_coords = (
        33.112500 + 5.504167 * upper_ratio,
        124.609722 + 7.263056 * lower_ratio,
    );
    data.world_coords = (-90.0 + 180.0 * upper_ratio, -180.0 + 360.0 * lower_ratio);
    data.planet_number =
        extract_valid_bits_for_nms::<4>(num, &[52, 4, 0], 11, &mut supplemental)? as u8 % 6 + 1;
    data.solar_system_index =
        extract_valid_bits_for_nms::<12>(num, &[40], 3834, &mut supplemental)? as u16 % 767 + 1;
    data.nms_portal_yy = upper_32_bits as u8;
    data.nms_portal_zzz = ((num >> 20) & 0xFFF) as u16;
    data.nms_portal_xxx = ((num >> 8) & 0xFFF) as u16;
    data.galaxy_x = galaxy_coord::<0x801, 0x7FF>(data.nms_portal_xxx);
    data.galaxy_y = galaxy_coord::<0x81, 0x7F>(data.nms_portal_yy as u16);
    data.galaxy_z = galaxy_coord::<0x801, 0x7FF>(data.nms_portal_zzz);
    for (i, slot) in data.glyph_string.iter_mut().enumerate() {
        let nibble = (match i {
            0 => data.planet_number as u64,
            1 => (data.solar_system_index >> 8) as u64,
            2 => (data.solar_system_index >> 4) as u64,
            3 => data.solar_system_index as u64,
            _ => num >> (36 - (i as u8 - 4) * 4),
        } & 0xF) as usize;
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
    while unsafe { _rdseed64_step(&mut v) } != 1 {
        std::hint::spin_loop()
    }
    Ok(v)
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
                data.password[data.password_len as usize] = ch;
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
            numbers.sort_unstable()
        }
        return true;
    }
    false
}
fn extract_valid_bits_for_nms<const BITS: u8>(
    num: u64,
    shifts: &[u8],
    max_value: u64,
    supplemental: &mut Option<RandomBitBuffer>,
) -> Result<u64> {
    let mask: u64 = bitmask_const::<BITS>();
    for &shift in shifts {
        let v = (num >> shift) & mask;
        if v <= max_value {
            return Ok(v);
        }
    }
    loop {
        let need_new = match supplemental.as_ref() {
            Some(supp) => supp.bits_remaining < BITS,
            None => true,
        };
        if need_new {
            *supplemental = Some(RandomBitBuffer::new()?);
        }
        let supp = supplemental.as_mut().expect("supplemental must be Some");
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
    let mut slice = &mut buffer[..];
    format_output(&mut slice, data, use_colors)?;
    Ok(BUFFER_SIZE - slice.len())
}
struct BufCursor<'a> {
    buf: &'a mut [u8],
    pos: usize,
}
impl<'a> BufCursor<'a> {
    #[inline(always)]
    fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    #[inline(always)]
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
    #[inline(always)]
    fn written_len(&self) -> usize {
        self.pos
    }
    #[inline(always)]
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
    #[inline(always)]
    fn write_byte(&mut self, b: u8) -> IoRst<()> {
        if self.remaining() < 1 {
            return Err(write_zero_err());
        }
        self.buf[self.pos] = b;
        self.pos += 1;
        Ok(())
    }
}
#[inline(always)]
fn cursor_write_fmt(cur: &mut BufCursor<'_>, args: std::fmt::Arguments<'_>) -> IoRst<()> {
    let start = cur.pos;
    let mut slice = &mut cur.buf[start..];
    let before = slice.len();
    let res = slice.write_fmt(args);
    let after = slice.len();
    cur.pos += before - after;
    res
}
#[inline(always)]
fn buf_write_bytes(cur: &mut BufCursor<'_>, bytes: &[u8]) -> IoRst<()> {
    cur.write_bytes(bytes)
}
#[inline(always)]
fn buf_write_byte(cur: &mut BufCursor<'_>, byte: u8) -> IoRst<()> {
    cur.write_byte(byte)
}
#[inline(always)]
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
#[inline(always)]
fn buf_write_u8_dec(cur: &mut BufCursor<'_>, n: u8) -> IoRst<()> {
    if n >= 100 {
        if cur.remaining() < 3 {
            return Err(write_zero_err());
        }
        let hundreds = (n / 100) as usize;
        let rem = (n % 100) as usize;
        let start = cur.pos;
        cur.buf[start] = DIGITS[hundreds];
        cur.buf[start + 1..start + 3].copy_from_slice(&TWO_DIGITS[rem]);
        cur.pos = start + 3;
        Ok(())
    } else if n >= 10 {
        if cur.remaining() < 2 {
            return Err(write_zero_err());
        }
        let start = cur.pos;
        cur.buf[start..start + 2].copy_from_slice(&TWO_DIGITS[n as usize]);
        cur.pos = start + 2;
        Ok(())
    } else {
        if cur.remaining() < 1 {
            return Err(write_zero_err());
        }
        cur.buf[cur.pos] = b'0' + n;
        cur.pos += 1;
        Ok(())
    }
}
#[inline(always)]
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
            let hundreds = (n / 100) as usize;
            let rem = (n % 100) as usize;
            head[pos] = DIGITS[hundreds];
            head[pos + 1..pos + 3].copy_from_slice(&TWO_DIGITS[rem]);
            pos += 3;
        } else if n >= 10 {
            head[pos..pos + 2].copy_from_slice(&TWO_DIGITS[n as usize]);
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
#[inline(always)]
fn buf_write_hash_hex24_from_bytes(cur: &mut BufCursor<'_>, b0: u8, b1: u8, b2: u8) -> IoRst<()> {
    if cur.remaining() < 7 {
        return Err(write_zero_err());
    }
    let start = cur.pos;
    let head = &mut cur.buf[start..start + 7];
    head[0] = b'#';
    head[1..3].copy_from_slice(&HEX_BYTE_TABLE[b0 as usize]);
    head[3..5].copy_from_slice(&HEX_BYTE_TABLE[b1 as usize]);
    head[5..7].copy_from_slice(&HEX_BYTE_TABLE[b2 as usize]);
    cur.pos = start + 7;
    Ok(())
}
#[inline(always)]
fn buf_write_m_hash_hex24_from_bytes(cur: &mut BufCursor<'_>, b0: u8, b1: u8, b2: u8) -> IoRst<()> {
    if cur.remaining() < 8 {
        return Err(write_zero_err());
    }
    let start = cur.pos;
    let head = &mut cur.buf[start..start + 8];
    head[0] = b'm';
    head[1] = b'#';
    head[2..4].copy_from_slice(&HEX_BYTE_TABLE[b0 as usize]);
    head[4..6].copy_from_slice(&HEX_BYTE_TABLE[b1 as usize]);
    head[6..8].copy_from_slice(&HEX_BYTE_TABLE[b2 as usize]);
    cur.pos = start + 8;
    Ok(())
}
#[inline(always)]
fn buf_write_bin8_line(cur: &mut BufCursor<'_>, bytes: &[u8; 8]) -> IoRst<()> {
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
        let b = bytes[i] as usize;
        head[pos..pos + 8].copy_from_slice(&BIN8_TABLE[b]);
        pos += 8;
        if i != 7 {
            head[pos] = b' ';
            pos += 1;
        } else {
            head[pos] = b'\n';
            pos += 1;
        }
        i += 1;
    }
    cur.pos = start + LINE_LEN;
    Ok(())
}
#[inline(always)]
fn buf_write_hex8_line(cur: &mut BufCursor<'_>, bytes: &[u8; 8]) -> IoRst<()> {
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
        let b = bytes[i] as usize;
        head[pos..pos + 2].copy_from_slice(&HEX_BYTE_TABLE[b]);
        pos += 2;
        if i != 7 {
            head[pos] = b' ';
            pos += 1;
        } else {
            head[pos] = b'\n';
            pos += 1;
        }
        i += 1;
    }
    cur.pos = start + LINE_LEN;
    Ok(())
}
#[inline(always)]
fn buf_write_ascii8(cur: &mut BufCursor<'_>, chars: &[char; 8]) -> IoRst<()> {
    if cur.remaining() < 8 {
        return Err(write_zero_err());
    }
    let start = cur.pos;
    let head = &mut cur.buf[start..start + 8];
    head[0] = chars[0] as u8;
    head[1] = chars[1] as u8;
    head[2] = chars[2] as u8;
    head[3] = chars[3] as u8;
    head[4] = chars[4] as u8;
    head[5] = chars[5] as u8;
    head[6] = chars[6] as u8;
    head[7] = chars[7] as u8;
    cur.pos = start + 8;
    Ok(())
}
#[inline(always)]
fn buf_write_u32_dec(cur: &mut BufCursor<'_>, mut n: u32) -> IoRst<()> {
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    while n >= 100 {
        let rem = (n % 100) as usize;
        n /= 100;
        i -= 2;
        tmp[i..i + 2].copy_from_slice(&TWO_DIGITS[rem]);
    }
    if n >= 10 {
        let rem = n as usize;
        i -= 2;
        tmp[i..i + 2].copy_from_slice(&TWO_DIGITS[rem]);
    } else {
        i -= 1;
        tmp[i] = b'0' + (n as u8);
    }
    buf_write_bytes(cur, &tmp[i..])
}
#[inline(always)]
fn buf_write_u64_dec(cur: &mut BufCursor<'_>, mut n: u64) -> IoRst<()> {
    let mut tmp = [0u8; 20];
    let mut i = tmp.len();
    while n >= 100 {
        let rem = (n % 100) as usize;
        n /= 100;
        i -= 2;
        tmp[i..i + 2].copy_from_slice(&TWO_DIGITS[rem]);
    }
    if n >= 10 {
        let rem = n as usize;
        i -= 2;
        tmp[i..i + 2].copy_from_slice(&TWO_DIGITS[rem]);
    } else {
        i -= 1;
        tmp[i] = b'0' + (n as u8);
    }
    buf_write_bytes(cur, &tmp[i..])
}
#[inline(always)]
fn buf_write_i64_dec(cur: &mut BufCursor<'_>, n: i64) -> IoRst<()> {
    if n < 0 {
        buf_write_byte(cur, b'-')?;
        let abs = if n == i64::MIN {
            (i64::MAX as u64) + 1
        } else {
            (-n) as u64
        };
        buf_write_u64_dec(cur, abs)
    } else {
        buf_write_u64_dec(cur, n as u64)
    }
}
#[inline(always)]
fn buf_write_u32_dec_0pad_6(cur: &mut BufCursor<'_>, n: u32) -> IoRst<()> {
    if n >= 1_000_000 {
        return buf_write_u32_dec(cur, n);
    }
    let hi = (n / 10_000) as usize;
    let rem = (n % 10_000) as usize;
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
#[inline(always)]
fn buf_write_u64_octal(cur: &mut BufCursor<'_>, mut n: u64) -> IoRst<()> {
    if n == 0 {
        return buf_write_byte(cur, b'0');
    }
    let mut tmp = [0u8; 22];
    let mut i = tmp.len();
    while n != 0 {
        i -= 1;
        tmp[i] = b'0' + ((n & 7) as u8);
        n >>= 3;
    }
    buf_write_bytes(cur, &tmp[i..])
}
#[inline(always)]
fn buf_write_hex_u16_0pad4(cur: &mut BufCursor<'_>, v: u16) -> IoRst<()> {
    if cur.remaining() < 4 {
        return Err(write_zero_err());
    }
    let start = cur.pos;
    let head = &mut cur.buf[start..start + 4];
    head[0..2].copy_from_slice(&HEX_BYTE_TABLE[(v >> 8) as usize]);
    head[2..4].copy_from_slice(&HEX_BYTE_TABLE[(v & 0xFF) as usize]);
    cur.pos = start + 4;
    Ok(())
}
#[inline(always)]
fn buf_write_hex_u16_min3(cur: &mut BufCursor<'_>, v: u16) -> IoRst<()> {
    if v < 0x1000 {
        if cur.remaining() < 3 {
            return Err(write_zero_err());
        }
        let start = cur.pos;
        let head = &mut cur.buf[start..start + 3];
        let hi = (v >> 8) as usize;
        let lo = (v & 0xFF) as usize;
        head[0] = HEX_UPPER[hi];
        head[1..3].copy_from_slice(&HEX_BYTE_TABLE[lo]);
        cur.pos = start + 3;
        Ok(())
    } else {
        buf_write_hex_u16_0pad4(cur, v)
    }
}
fn format_output(writer: &mut &mut [u8], data: &RandomDataSet, use_colors: bool) -> IoRst<()> {
    let buf = std::mem::take(writer);
    let mut cur = BufCursor::new(buf);
    let v = data.num_64;
    let bytes = v.to_be_bytes();
    let [b0, b1, b2, b3, b4, b5, _, _] = bytes;
    buf_write_bytes(&mut cur, "64비트 난수: ".as_bytes())?;
    buf_write_u64_dec(&mut cur, v)?;
    buf_write_bytes(&mut cur, " (유부호 정수: ".as_bytes())?;
    buf_write_i64_dec(&mut cur, v as i64)?;
    buf_write_bytes(&mut cur, ")\n".as_bytes())?;
    buf_write_bin8_line(&mut cur, &bytes)?;
    buf_write_bytes(&mut cur, "8진수: ".as_bytes())?;
    buf_write_u64_octal(&mut cur, v)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_hex8_line(&mut cur, &bytes)?;
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
    buf_write_bytes(&mut cur, " + ".as_bytes())?;
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
    buf_write_u32_dec(&mut cur, (b0 as u16).wrapping_add(1) as u32)?;
    buf_write_byte(&mut cur, b'\n')?;
    buf_write_bytes(&mut cur, "NMS 포탈 주소: ".as_bytes())?;
    buf_write_u8_dec(&mut cur, data.planet_number)?;
    buf_write_byte(&mut cur, b' ')?;
    buf_write_hex_u16_min3(&mut cur, data.solar_system_index)?;
    buf_write_byte(&mut cur, b' ')?;
    buf_write_bytes(&mut cur, &HEX_BYTE_TABLE[data.nms_portal_yy as usize])?;
    buf_write_byte(&mut cur, b' ')?;
    buf_write_hex_u16_min3(&mut cur, data.nms_portal_zzz)?;
    buf_write_byte(&mut cur, b' ')?;
    buf_write_hex_u16_min3(&mut cur, data.nms_portal_xxx)?;
    buf_write_byte(&mut cur, b' ')?;
    buf_write_byte(&mut cur, b'(')?;
    buf_write_chars(&mut cur, &data.glyph_string)?;
    buf_write_bytes(&mut cur, ")\n".as_bytes())?;
    buf_write_bytes(&mut cur, "NMS 은하 좌표: ".as_bytes())?;
    buf_write_hex_u16_0pad4(&mut cur, data.galaxy_x)?;
    buf_write_byte(&mut cur, b':')?;
    buf_write_hex_u16_0pad4(&mut cur, data.galaxy_y)?;
    buf_write_byte(&mut cur, b':')?;
    buf_write_hex_u16_0pad4(&mut cur, data.galaxy_z)?;
    buf_write_byte(&mut cur, b':')?;
    buf_write_hex_u16_0pad4(&mut cur, data.solar_system_index)?;
    let BufCursor { buf, pos } = cur;
    *writer = &mut buf[pos..];
    Ok(())
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
        indices_slice.swap(i, random_bounded((i + 1) as u64, current_seed)? as usize);
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
        eprintln!("{MIN_ALLOWED_VALUE} 이상의 값을 입력해 주세요.\n")
    };
    let max_value = loop {
        let n = read_parse_i64(format_args!("최댓값을 입력해 주세요: "), input_buffer)?;
        if n >= min_value {
            break n;
        }
        eprintln!("최댓값은 최솟값보다 크거나 같아야 합니다.\n")
    };
    let range_size = max_value.wrapping_sub(min_value).wrapping_add(1) as u64;
    let rand_offset = if range_size == 0 {
        (get_hardware_random()? ^ seed_modifier) as i64
    } else {
        random_bounded(range_size, seed_modifier)? as i64
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
fn generate_random_float(seed_modifier: u64, input_buffer: &mut String) -> Result<()> {
    println!("\n무작위 실수 생성기");
    let fmin_prompt = format_args!("최솟값을 입력해 주세요: ");
    let fmax_prompt = format_args!("최댓값을 입력해 주세요: ");
    let min_value: f64 = read_parse_f64(fmin_prompt, input_buffer)?;
    let max_value: f64 = loop {
        let num = read_parse_f64(fmax_prompt, input_buffer)?;
        if num >= min_value {
            break num;
        } else {
            eprintln!("최댓값은 최솟값보다 크거나 같아야 합니다.\n")
        }
    };
    let scale = (get_hardware_random()? ^ seed_modifier) as f64 / (u64::MAX as f64);
    let result = if min_value == max_value {
        min_value
    } else {
        min_value + scale * (max_value - min_value)
    };
    println!("무작위 실수({min_value} ~ {max_value}): {result}");
    Ok(())
}
fn read_parse_f64(prompt: std::fmt::Arguments, buffer: &mut String) -> Result<f64> {
    loop {
        match read_line_reuse(prompt, buffer)?.parse::<f64>() {
            Ok(n) if n.is_finite() && !n.is_subnormal() => return Ok(n),
            _ => {
                eprintln!("유효한 정규 실수 값을 입력해야 합니다 (NaN, 무한대, 비정규 값 제외).\n")
            }
        }
    }
}
fn random_bounded(s: u64, seed_mod: u64) -> Result<u64> {
    let threshold = s.wrapping_neg() % s;
    let s128 = s as u128;
    loop {
        let m = ((get_hardware_random()? ^ seed_mod) as u128) * s128;
        if (m as u64) >= threshold {
            return Ok((m >> 64) as u64);
        }
    }
}
fn regenerate_multiple(
    file_mutex: &Mutex<BufWriter<File>>,
    input_buffer: &mut String,
) -> Result<u64> {
    let count_prompt = format_args!("\n생성할 데이터 개수를 입력해 주세요: ");
    let requested_count: u64 = loop {
        match read_line_reuse(count_prompt, input_buffer)?.parse::<u64>() {
            Ok(0) => eprintln!("1 이상의 값을 입력해 주세요."),
            Ok(n) => break n,
            Err(_) => eprintln!("유효한 숫자를 입력해 주세요."),
        }
    };
    if requested_count == 1 {
        ensure_file_exists_and_reopen(file_mutex)?;
        return process_single_random_data(file_mutex);
    }
    ensure_file_exists_and_reopen(file_mutex)?;
    let multi_thread_count = requested_count.saturating_sub(1);
    let max_threads = available_parallelism().map_or(4, |n| n.get());
    let calculated_thread_count = multi_thread_count.min(max_threads as u64) as usize;
    let start_time = Instant::now();
    let in_flight_buffers = calculated_thread_count.saturating_mul(BUFFERS_PER_WORKER);
    let (sender, receiver) = sync_channel::<(DataBuffer, usize, usize)>(in_flight_buffers);
    let mut buffer_return_txs = Vec::with_capacity(calculated_thread_count);
    let mut buffer_return_rxs = Vec::with_capacity(calculated_thread_count);
    for _ in 0..calculated_thread_count {
        let (tx, rx) = sync_channel::<DataBuffer>(BUFFERS_PER_WORKER);
        for _ in 0..BUFFERS_PER_WORKER {
            tx.send(Box::new([0u8; BUFFER_SIZE]))?;
        }
        buffer_return_txs.push(tx);
        buffer_return_rxs.push(rx);
    }
    let completed = AtomicU64::new(0);
    let final_data = scope(|s| -> Result<RandomDataSet> {
        let buffer_return_txs = buffer_return_txs;
        let writer_thread = s.spawn(move || -> Result<RandomDataSet> {
            let mut file_guard = lock_mutex(file_mutex, "Mutex 잠금 실패 (쓰기 스레드)")?;
            while let Ok((data_buffer, data_len, worker_idx)) = receiver.recv() {
                write_buffer_to_file_guard(&mut file_guard, &data_buffer[..data_len])?;
                let _ = buffer_return_txs[worker_idx].send(data_buffer);
                while let Ok((more_buffer, more_len, more_worker_idx)) = receiver.try_recv() {
                    write_buffer_to_file_guard(&mut file_guard, &more_buffer[..more_len])?;
                    let _ = buffer_return_txs[more_worker_idx].send(more_buffer);
                }
            }
            let final_data = generate_random_data()?;
            let mut final_buffer_file = [0u8; BUFFER_SIZE];
            let final_bytes_written_file =
                format_data_into_buffer(&final_data, &mut final_buffer_file, false)?;
            write_buffer_to_file_guard(
                &mut file_guard,
                &final_buffer_file[..final_bytes_written_file],
            )?;
            file_guard.flush()?;
            Ok(final_data)
        });
        let completed_ref = &completed;
        let progress_thread: Option<ScopedJoinHandle<Result<()>>> = if *IS_TERMINAL {
            Some(s.spawn(move || -> Result<()> {
                let mut elapsed_buf = [0u8; 7];
                let mut eta_buf = [0u8; 7];
                loop {
                    let completed_now = completed_ref.load(Ordering::Relaxed);
                    if completed_now >= multi_thread_count {
                        break;
                    }
                    print_progress(
                        completed_now,
                        requested_count,
                        &start_time,
                        &mut elapsed_buf,
                        &mut eta_buf,
                    )?;
                    sleep(Duration::from_millis(100))
                }
                Ok(())
            }))
        } else {
            None
        };
        let base_count = multi_thread_count / calculated_thread_count as u64;
        let remainder = multi_thread_count % calculated_thread_count as u64;
        for (i, return_rx) in buffer_return_rxs.into_iter().enumerate() {
            let sender_clone = sender.clone();
            s.spawn(move || -> Result<()> {
                let loop_count = base_count + if (i as u64) < remainder { 1 } else { 0 };
                let mut local_pool = Vec::with_capacity(BUFFERS_PER_WORKER);
                for _ in 0..BUFFERS_PER_WORKER {
                    match return_rx.try_recv() {
                        Ok(buf) => local_pool.push(buf),
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => return Ok(()),
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
                    let len = match generate_random_data() {
                        Ok(data) => match format_data_into_buffer(&data, buffer.as_mut(), false) {
                            Ok(len) => len,
                            Err(_) => {
                                local_pool.push(buffer);
                                continue;
                            }
                        },
                        Err(_) => {
                            local_pool.push(buffer);
                            continue;
                        }
                    };
                    match sender_clone.send((buffer, len, i)) {
                        Ok(()) => {
                            completed_ref.fetch_add(1, Ordering::Relaxed);
                            while local_pool.len() < BUFFERS_PER_WORKER {
                                match return_rx.try_recv() {
                                    Ok(buf) => local_pool.push(buf),
                                    Err(TryRecvError::Empty) => break,
                                    Err(TryRecvError::Disconnected) => break,
                                }
                            }
                        }
                        Err(send_err) => {
                            let (_returned_buffer, _, _) = send_err.0;
                            break;
                        }
                    }
                }
                Ok(())
            });
        }
        drop(sender);
        if let Some(handle) = progress_thread {
            join_thread(handle, "진행률 스레드 패닉 발생")?
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
    println!("\n총 {requested_count}개의 데이터 생성 완료 ({FILE_NAME} 저장됨).\n");
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
    let elapsed_secs = start_time.elapsed().as_secs_f64();
    let progress = if total == 0 {
        1.0
    } else {
        completed as f64 / total as f64
    };
    let eta_secs = match progress {
        p if p >= 1.0 => 0.0,
        p if p < 1e-9 => f64::INFINITY,
        _ => elapsed_secs * (total - completed) as f64 / completed as f64,
    };
    let elapsed_len = format_time_into(elapsed_secs, elapsed_buf)?;
    let eta_len = format_time_into(eta_secs, eta_buf)?;
    let filled = ((progress * BAR_WIDTH as f64).floor() as usize).min(BAR_WIDTH);
    let bar = BAR_FULL[filled];
    let percent = (progress * 100.0).floor() as u32;
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
    buf_write_u8_dec(&mut cur, percent as u8)?;
    buf_write_byte(&mut cur, b'%')?;
    buf_write_bytes(&mut cur, b" (")?;
    buf_write_u64_dec(&mut cur, completed)?;
    buf_write_byte(&mut cur, b'/')?;
    buf_write_u64_dec(&mut cur, total)?;
    buf_write_bytes(&mut cur, ") | 소요: ".as_bytes())?;
    buf_write_bytes(&mut cur, &elapsed_buf[..elapsed_len])?;
    buf_write_bytes(&mut cur, " | ETA: ".as_bytes())?;
    buf_write_bytes(&mut cur, &eta_buf[..eta_len])?;
    buf_write_bytes(&mut cur, b" \x1b[K")?;
    let used = cur.written_len();
    let mut out = stdout().lock();
    out.write_all(&line[..used])?;
    out.flush()?;
    Ok(())
}
fn format_time_into(seconds: f64, buf: &mut [u8]) -> IoRst<usize> {
    if !seconds.is_finite() || seconds < 0.0 {
        buf[..7].copy_from_slice(INVALID_TIME);
        return Ok(7);
    }
    let deci: u64 = (seconds * 10.0).round() as u64;
    let minutes = (deci / 600).min(99) as usize;
    let sec_whole = ((deci / 10) % 60) as usize;
    let tenths = (deci % 10) as usize;
    buf[0..2].copy_from_slice(&TWO_DIGITS[minutes]);
    buf[2] = b':';
    buf[3..5].copy_from_slice(&TWO_DIGITS[sec_whole]);
    buf[5] = b'.';
    buf[6] = DIGITS[tenths];
    Ok(7)
}
fn join_thread<T>(handle: ScopedJoinHandle<'_, Result<T>>, panic_msg: &'static str) -> Result<T> {
    handle.join().map_err(|_| ioErr::other(panic_msg))?
}
