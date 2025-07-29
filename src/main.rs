use std::{
    arch::x86_64::{_rdrand64_step, _rdseed64_step},
    array,
    char::from_u32,
    error::Error,
    fmt::{Display, Formatter, Result as fmtResult},
    fs::{File, remove_file},
    hint::spin_loop,
    io::{BufWriter, Error as ioErr, IsTerminal, Result as IoRst, Write, stdin, stdout},
    is_x86_feature_detected,
    path::Path,
    process::ExitCode,
    result::Result as stdResult,
    sync::{
        LazyLock, Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
        mpsc::{SyncSender, sync_channel},
    },
    thread::{ScopedJoinHandle, available_parallelism, scope, sleep},
    time::{Duration, Instant},
};
mod time;
enum RngSource {
    RdSeed,
    RdRand,
    None,
}
static RNG_SOURCE: LazyLock<RngSource> = LazyLock::new(|| {
    if is_x86_feature_detected!("rdseed") {
        RngSource::RdSeed
    } else if is_x86_feature_detected!("rdrand") {
        eprintln!("RDSEED를 미지원하여 RDRAND를 사용합니다.");
        RngSource::RdRand
    } else {
        RngSource::None
    }
});
static GLYPHS: [char; 16] = [
    '🌅', '🐦', '👫', '🦕', '🌘', '🎈', '⛵', '🕷', '🦋', '🌀', '🧊', '🐟', '⛺', '🚀', '🌳', '🔯',
];
static IS_TERMINAL: LazyLock<bool> = LazyLock::new(|| stdout().is_terminal());
static TWO_DIGITS: LazyLock<[[u8; 2]; 100]> =
    LazyLock::new(|| array::from_fn(|i| [b'0' + (i / 10) as u8, b'0' + (i % 10) as u8]));
type Result<T> = stdResult<T, Box<dyn Error + Send + Sync + 'static>>;
const FILE_NAME: &str = "random_data.txt";
const BUFFER_SIZE: usize = size_of_val(&[0u8; 1016]);
const fn bit(n: u8) -> u64 {
    1u64 << n
}
const fn bitmask_const<const B: u8>() -> u64 {
    if B == 0 {
        0
    } else if B >= 64 {
        !0
    } else {
        (!0u64) >> (64 - B)
    }
}
const fn galaxy_coord<const SUB: u16, const ADD: u16>(value: u16) -> u16 {
    let a = value.wrapping_sub(SUB);
    let b = value.wrapping_add(ADD);
    if a < b { a } else { b }
}
const U32_MAX_INV: f64 = 1.0 / (u32::MAX as f64);
const RGB_PREFIX: &str = "\x1B[38;2;";
const SGR_SUFFIX: &str = "m";
const COLOR_RESET: &str = "\x1B[0m";
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
const DIGITS: &[u8; 10] = b"0123456789";
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
struct HexCodeFormatter<'a> {
    data: &'a RandomDataSet,
    use_colors: bool,
}
impl Display for HexCodeFormatter<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmtResult {
        let v = self.data.num_64;
        let r = (v >> 56) as u8;
        let g = (v >> 48) as u8;
        let b = (v >> 40) as u8;
        let r2 = (v >> 32) as u8;
        let g2 = (v >> 24) as u8;
        let b2 = (v >> 16) as u8;
        let (hex1, hex2) = ((v >> 40) & 0xFF_FFFF, (v >> 16) & 0xFF_FFFF);
        if self.use_colors {
            write!(
                f,
                "{RGB_PREFIX}{r};{g};{b}{SGR_SUFFIX}#{hex1:06X}{COLOR_RESET} \
                 {RGB_PREFIX}{r2};{g2};{b2}{SGR_SUFFIX}#{hex2:06X}{COLOR_RESET}",
            )
        } else {
            write!(f, "#{hex1:06X} #{hex2:06X}")
        }
    }
}
fn main() -> Result<ExitCode> {
    let file_mutex = Mutex::new(open_or_create_file()?);
    let mut num_64 = process_single_random_data(&file_mutex)?.0;
    let mut input_buffer = String::new();
    loop {
        match read_line_reuse(
            "\n1: 사다리타기 실행, 2: 무작위 숫자 생성, 3: 데이터 생성(1회), 4: 데이터 생성(여러 회), 5: 서버 시간 확인, 6: 파일 삭제, 기타: 종료\n선택해 주세요: ",
            &mut input_buffer,
        )? {
            "1" => ladder_game(num_64, &mut input_buffer)?,
            "2" => {
                println!("\n무작위 숫자 생성 타입 선택:");
                match read_line_reuse(
                    "1: 정수 생성, 2: 실수 생성, 기타: 취소\n선택해 주세요: ",
                    &mut input_buffer,
                )? {
                    "1" => {
                        generate_random_integer(num_64, &mut input_buffer)?;
                    }
                    "2" => {
                        generate_random_float(num_64, &mut input_buffer)?;
                    }
                    _ => {
                        println!("무작위 숫자 생성을 취소합니다.");
                    }
                }
            }
            "3" => {
                ensure_file_exists_and_reopen(&file_mutex)?;
                num_64 = process_single_random_data(&file_mutex)?.0;
            }
            "4" => {
                ensure_file_exists_and_reopen(&file_mutex)?;
                num_64 = regenerate_multiple(&file_mutex, &mut input_buffer)?;
            }
            "5" => {
                if let Err(e) = time::run() {
                    eprintln!("서버 시간 확인 중 오류 발생: {e}");
                }
            }
            "6" => match remove_file(FILE_NAME) {
                Ok(_) => {
                    println!("파일 '{FILE_NAME}'를 삭제했습니다.");
                }
                Err(e) => {
                    eprintln!("{e}");
                }
            },
            _ => return Ok(ExitCode::SUCCESS),
        }
    }
}
fn ensure_file_exists_and_reopen(file_mutex: &Mutex<BufWriter<File>>) -> Result<()> {
    if !Path::new(FILE_NAME).try_exists()? {
        *lock_mutex(file_mutex, "Mutex 잠금 실패 (파일 생성 시)")? = open_or_create_file()?;
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
fn process_single_random_data(file_mutex: &Mutex<BufWriter<File>>) -> Result<(u64, RandomDataSet)> {
    let (num_64, data) = generate_random_data()?;
    let mut file_buffer = [0; BUFFER_SIZE];
    let file_len = format_data_into_buffer(&data, &mut file_buffer, false)?;
    {
        let mut file_guard = lock_mutex(file_mutex, "Mutex 잠금 실패 (단일 쓰기 시)")?;
        write_buffer_to_file_guard(&mut file_guard, &file_buffer[..file_len])?;
        file_guard.flush()?;
    }
    if *IS_TERMINAL {
        let mut console_buffer = [0; BUFFER_SIZE];
        let console_len = format_data_into_buffer(&data, &mut console_buffer, true)?;
        write_slice_to_console(&console_buffer[..console_len])?;
    } else {
        write_slice_to_console(&file_buffer[..file_len])?;
    }
    Ok((num_64, data))
}
fn generate_random_data() -> Result<(u64, RandomDataSet)> {
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
        supplemental = Some(new_supp);
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
            supplemental = Some(new_supp);
        } else {
            break;
        }
    }
    let mut hangul = ['\0'; 4];
    for (i, slot) in hangul.iter_mut().enumerate() {
        let mut index = ((num >> (48 - 16 * i)) & 0xFFFF) as u32;
        while index > 55859 {
            if supplemental.is_none() {
                supplemental = Some(RandomBitBuffer::new()?);
            }
            if let Some(supp) = supplemental.as_ref() {
                if let Some(valid_index) = (0..4)
                    .map(|i| ((supp.value >> (48 - 16 * i)) & 0xFFFF) as u32)
                    .find(|&candidate| candidate <= 55859)
                {
                    index = valid_index;
                    break;
                } else {
                    supplemental = Some(RandomBitBuffer::new()?);
                }
            }
        }
        *slot = from_u32(0xAC00 + (index % 11172)).ok_or("한글 음절 변환 실패")?;
    }
    data.hangul_syllables = hangul;
    let upper_32_bits = num >> 32;
    let upper_ratio = (upper_32_bits as f64) * U32_MAX_INV;
    let lower_ratio = ((num & 0xFFFF_FFFF) as f64) * U32_MAX_INV;
    data.kor_coords = (
        33.1123557596338 + 5.5013535429421 * upper_ratio,
        124.609717678567 + 7.263065469715 * lower_ratio,
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
    for i in 0..12 {
        data.glyph_string[i] = GLYPHS[(match i {
            0 => data.planet_number as u64,
            1 => (data.solar_system_index >> 8) as u64,
            2 => (data.solar_system_index >> 4) as u64,
            3 => data.solar_system_index as u64,
            _ => num >> (36 - (i as u8 - 4) * 4),
        } & 0xF) as usize];
    }
    Ok((num, data))
}
fn get_hardware_random() -> Result<u64> {
    match *RNG_SOURCE {
        RngSource::RdSeed => rdseed_impl(),
        RngSource::RdRand => rdrand_impl(),
        RngSource::None => no_hw_rng(),
    }
}
fn rdseed_impl() -> Result<u64> {
    let mut v: u64 = 0;
    while unsafe { _rdseed64_step(&mut v) } != 1 {
        spin_loop();
    }
    Ok(v)
}
fn rdrand_impl() -> Result<u64> {
    let mut v: u64 = 0;
    for _ in 0..10 {
        if unsafe { _rdrand64_step(&mut v) } == 1 {
            return Ok(v);
        }
        spin_loop();
    }
    Err("RDRAND 실패".into())
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
    let mask = bit(number);
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
    supplemental: &mut Option<RandomBitBuffer>,
) -> Result<u64> {
    let mask: u64 = bitmask_const::<BITS>();
    if let Some(value) = shifts
        .iter()
        .map(|&shift| (num >> shift) & mask)
        .find(|&v| v <= max_value)
    {
        return Ok(value);
    }
    loop {
        if supplemental
            .as_ref()
            .is_none_or(|supp| supp.bits_remaining < BITS)
        {
            *supplemental = Some(RandomBitBuffer::new()?);
        }
        if let Some(supp) = supplemental.as_mut() {
            let extracted = (supp.value >> (supp.bits_remaining - BITS)) & mask;
            supp.bits_remaining -= BITS;
            if extracted <= max_value {
                return Ok(extracted);
            }
        }
    }
}
fn format_data_into_buffer(
    data: &RandomDataSet,
    buffer: &mut [u8],
    use_colors: bool,
) -> Result<usize> {
    let mut slice = &mut buffer[..];
    format_output(&mut slice, data, use_colors)?;
    Ok(BUFFER_SIZE - slice.len())
}
fn format_output<W: Write>(writer: &mut W, data: &RandomDataSet, use_colors: bool) -> IoRst<()> {
    let v = data.num_64;
    let b0 = (v >> 56) as u8;
    let b1 = (v >> 48) as u8;
    let b2 = (v >> 40) as u8;
    let b3 = (v >> 32) as u8;
    let b4 = (v >> 24) as u8;
    let b5 = (v >> 16) as u8;
    let b6 = (v >> 8) as u8;
    let b7 = v as u8;
    write!(
        writer,
        concat!(
            "64비트 난수: {0} (유부호 정수: {1})\n",
            "2진수: {2:08b} {3:08b} {4:08b} {5:08b} {6:08b} {7:08b} {8:08b} {9:08b}\n",
            "8진수: {10:o}\n",
            "16진수: {11:02X} {12:02X} {13:02X} {14:02X} {15:02X} {16:02X} {17:02X} {18:02X}\n",
            "Hex 코드: {19}\n",
            "바이트 배열: {11} {12} {13} {14} {15} {16} {17} {18}\n",
            "6자리 숫자 비밀번호: {20:06}\n",
            "8자리 비밀번호: {21}{22}{23}{24}{25}{26}{27}{28}\n",
            "로또 번호: {29} {30} {31} {32} {33} {34}\n",
            "일본 로또 7 번호: {35} {36} {37} {38} {39} {40} {41}\n",
            "유로밀리언 번호: {72} {73} {74} {75} {76} + {77} {78}\n",
            "한글 음절 4글자: {42}{43}{44}{45}\n",
            "대한민국 위경도: {46}, {47}\n",
            "세계 위경도: {48}, {49}\n",
            "NMS 은하 번호: {50}\n",
            "NMS 포탈 주소: {51} {52:03X} {53:02X} {54:03X} {55:03X} ",
            "({56}{57}{58}{59}{60}{61}{62}{63}{64}{65}{66}{67})\n",
            "NMS 은하 좌표: {68:04X}:{69:04X}:{70:04X}:{71:04X}",
        ),
        data.num_64,
        data.num_64 as i64,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
        data.num_64,
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
        HexCodeFormatter { data, use_colors },
        data.numeric_password,
        data.password[0],
        data.password[1],
        data.password[2],
        data.password[3],
        data.password[4],
        data.password[5],
        data.password[6],
        data.password[7],
        data.lotto_numbers[0],
        data.lotto_numbers[1],
        data.lotto_numbers[2],
        data.lotto_numbers[3],
        data.lotto_numbers[4],
        data.lotto_numbers[5],
        data.lotto7_numbers[0],
        data.lotto7_numbers[1],
        data.lotto7_numbers[2],
        data.lotto7_numbers[3],
        data.lotto7_numbers[4],
        data.lotto7_numbers[5],
        data.lotto7_numbers[6],
        data.hangul_syllables[0],
        data.hangul_syllables[1],
        data.hangul_syllables[2],
        data.hangul_syllables[3],
        data.kor_coords.0,
        data.kor_coords.1,
        data.world_coords.0,
        data.world_coords.1,
        (b0 as u16).wrapping_add(1),
        data.planet_number,
        data.solar_system_index,
        data.nms_portal_yy,
        data.nms_portal_zzz,
        data.nms_portal_xxx,
        data.glyph_string[0],
        data.glyph_string[1],
        data.glyph_string[2],
        data.glyph_string[3],
        data.glyph_string[4],
        data.glyph_string[5],
        data.glyph_string[6],
        data.glyph_string[7],
        data.glyph_string[8],
        data.glyph_string[9],
        data.glyph_string[10],
        data.glyph_string[11],
        data.galaxy_x,
        data.galaxy_y,
        data.galaxy_z,
        data.solar_system_index,
        data.euro_millions_main_numbers[0],
        data.euro_millions_main_numbers[1],
        data.euro_millions_main_numbers[2],
        data.euro_millions_main_numbers[3],
        data.euro_millions_main_numbers[4],
        data.euro_millions_lucky_stars[0],
        data.euro_millions_lucky_stars[1]
    )?;
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
fn read_line_reuse<'a>(prompt: &str, buffer: &'a mut String) -> IoRst<&'a str> {
    buffer.clear();
    print!("{prompt}");
    stdout().flush()?;
    stdin().read_line(buffer)?;
    Ok(buffer.trim())
}
fn ladder_game(num_64: u64, player_input_buffer: &mut String) -> Result<()> {
    const MAX_PLAYERS: usize = 128;
    let n: usize;
    let mut players_array: [&str; MAX_PLAYERS] = [""; MAX_PLAYERS];
    let mut result_input_buffer = String::new();
    loop {
        let mut temp_count = 0;
        let mut validation_success = true;
        for (i, _) in parse_comma_separated(read_line_reuse(
            "\n사다리타기 플레이어를 입력해 주세요 (쉼표(,)로 구분, 2~128명): ",
            player_input_buffer,
        )?)
        .enumerate()
        {
            if i >= MAX_PLAYERS {
                eprintln!("플레이어 수가 최대 {MAX_PLAYERS}명을 초과했습니다.");
                validation_success = false;
                break;
            }
            temp_count = i + 1;
        }
        if !validation_success {
            continue;
        }
        if !(2..=MAX_PLAYERS).contains(&temp_count) {
            if temp_count < 2 {
                eprintln!("플레이어 수는 최소 2명이어야 합니다.");
            }
            continue;
        }
        n = temp_count;
        for (i, player_slice) in parse_comma_separated(player_input_buffer)
            .take(n)
            .enumerate()
        {
            players_array[i] = player_slice;
        }
        break;
    }
    let mut results_array: [&str; MAX_PLAYERS] = [""; MAX_PLAYERS];
    loop {
        let temp_count = parse_comma_separated(read_line_reuse(
            &format!("사다리타기 결과값을 입력해 주세요 (쉼표(,)로 구분, {n}개 필요): "),
            &mut result_input_buffer,
        )?)
        .count();
        if temp_count != n {
            eprintln!("결과값의 개수({temp_count})가 플레이어 수({n})와 일치하지 않습니다.\n");
            continue;
        }
        for (i, result_slice) in parse_comma_separated(&result_input_buffer)
            .take(n)
            .enumerate()
        {
            results_array[i] = result_slice;
        }
        break;
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
    for i in 0..n {
        println!(
            "{} -> {}",
            players_array[i], results_array[indices_slice[i]]
        );
    }
    Ok(())
}
fn parse_comma_separated(input: &str) -> impl Iterator<Item = &str> {
    input.split(',').map(str::trim)
}
fn generate_random_integer(seed_modifier: u64, input_buffer: &mut String) -> Result<()> {
    const MIN_ALLOWED_VALUE: i64 = i64::MIN + 1;
    println!(
        "\n무작위 정수 생성기(지원 범위: {MIN_ALLOWED_VALUE} ~ {})",
        i64::MAX
    );
    let min_value = loop {
        let n = read_parse_i64(
            &format!("최솟값을 입력해 주세요 ({MIN_ALLOWED_VALUE} 이상): "),
            input_buffer,
        )?;
        if n >= MIN_ALLOWED_VALUE {
            break n;
        }
        eprintln!("{MIN_ALLOWED_VALUE} 이상의 값을 입력해 주세요.\n");
    };
    let max_value = loop {
        let n = read_parse_i64("최댓값을 입력해 주세요: ", input_buffer)?;
        if n >= min_value {
            break n;
        }
        eprintln!("최댓값은 최솟값보다 크거나 같아야 합니다.\n");
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
fn read_parse_i64(prompt: &str, buffer: &mut String) -> Result<i64> {
    loop {
        print!("{prompt}");
        stdout().flush()?;
        match read_line_reuse("", buffer)?.parse::<i64>() {
            Ok(n) => return Ok(n),
            Err(_) => eprintln!("유효한 정수 형식이 아닙니다.\n"),
        }
    }
}
fn generate_random_float(seed_modifier: u64, input_buffer: &mut String) -> Result<()> {
    println!("\n무작위 실수 생성기");
    let min_value: f64 = read_parse_f64("최솟값을 입력해 주세요: ", input_buffer)?;
    let max_value: f64 = loop {
        let num = read_parse_f64("최댓값을 입력해 주세요: ", input_buffer)?;
        if num >= min_value {
            break num;
        } else {
            eprintln!("최댓값은 최솟값보다 크거나 같아야 합니다.\n");
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
fn read_parse_f64(prompt: &str, buffer: &mut String) -> Result<f64> {
    loop {
        print!("{prompt}");
        stdout().flush()?;
        match read_line_reuse("", buffer)?.parse::<f64>() {
            Ok(n) if n.is_finite() && !n.is_subnormal() => return Ok(n),
            _ => {
                eprintln!("유효한 정규 실수 값을 입력해야 합니다 (NaN, 무한대, 비정규 값 제외).\n")
            }
        }
    }
}
fn random_bounded(s: u64, seed_mod: u64) -> Result<u64> {
    let threshold = s.wrapping_neg() % s;
    loop {
        let m = ((get_hardware_random()? ^ seed_mod) as u128) * s as u128;
        if (m as u64) >= threshold {
            return Ok((m >> 64) as u64);
        }
    }
}
fn generate_and_send_random_data(sender: &SyncSender<([u8; BUFFER_SIZE], usize)>) -> Result<()> {
    let (_, data) = generate_random_data()?;
    let mut buffer = [0; BUFFER_SIZE];
    let len = format_data_into_buffer(&data, &mut buffer, false)?;
    sender.send((buffer, len))?;
    Ok(())
}
fn regenerate_multiple(
    file_mutex: &Mutex<BufWriter<File>>,
    input_buffer: &mut String,
) -> Result<u64> {
    let requested_count: u64 = loop {
        print!("\n생성할 데이터 개수를 입력해 주세요: ");
        stdout().flush()?;
        match read_line_reuse("", input_buffer)?.parse::<u64>() {
            Ok(0) => eprintln!("1 이상의 값을 입력해 주세요."),
            Ok(n) => break n,
            Err(_) => eprintln!("유효한 숫자를 입력해 주세요."),
        }
    };
    if requested_count == 1 {
        ensure_file_exists_and_reopen(file_mutex)?;
        return process_single_random_data(file_mutex).map(|(num, _)| num);
    }
    ensure_file_exists_and_reopen(file_mutex)?;
    let multi_thread_count = requested_count.saturating_sub(1);
    let (sender, receiver) =
        sync_channel::<([u8; BUFFER_SIZE], usize)>((multi_thread_count as usize).clamp(1, 32768));
    let start_time = Instant::now();
    let completed = AtomicU64::new(0);
    let mut last_generated_num: Option<u64> = None;
    let mut last_generated_data: Option<RandomDataSet> = None;
    scope(|s| -> Result<()> {
        let writer_thread = s.spawn(move || -> Result<(u64, RandomDataSet)> {
            let mut file_guard = lock_mutex(file_mutex, "Mutex 잠금 실패 (쓰기 스레드)")?;
            while let Ok((data_buffer, data_len)) = receiver.recv() {
                write_buffer_to_file_guard(&mut file_guard, &data_buffer[..data_len])?;
                while let Ok((more_buffer, more_len)) = receiver.try_recv() {
                    write_buffer_to_file_guard(&mut file_guard, &more_buffer[..more_len])?;
                }
            }
            let (final_num_64, final_data) = generate_random_data()?;
            let mut final_buffer_file = [0u8; BUFFER_SIZE];
            let final_bytes_written_file =
                format_data_into_buffer(&final_data, &mut final_buffer_file, false)?;
            write_buffer_to_file_guard(
                &mut file_guard,
                &final_buffer_file[..final_bytes_written_file],
            )?;
            file_guard.flush()?;
            Ok((final_num_64, final_data))
        });
        let completed_ref = &completed;
        let progress_thread: Option<ScopedJoinHandle<Result<()>>> = if *IS_TERMINAL {
            Some(s.spawn(move || -> Result<()> {
                while completed_ref.load(Ordering::Relaxed) < multi_thread_count {
                    print_progress(
                        completed_ref.load(Ordering::Relaxed),
                        requested_count,
                        &start_time,
                        &mut [0u8; 16],
                        &mut [0u8; 16],
                    )?;
                    sleep(Duration::from_millis(100));
                }
                Ok(())
            }))
        } else {
            None
        };
        let calculated_thread_count =
            (multi_thread_count as usize).min(available_parallelism().map_or(4, |n| n.get()));
        if calculated_thread_count > 0 {
            let base_count = multi_thread_count / calculated_thread_count as u64;
            let remainder = multi_thread_count % calculated_thread_count as u64;
            for i in 0..calculated_thread_count {
                let sender_clone = sender.clone();
                s.spawn(move || -> Result<()> {
                    let loop_count = base_count + if (i as u64) < remainder { 1 } else { 0 };
                    for _ in 0..loop_count {
                        if generate_and_send_random_data(&sender_clone).is_ok() {
                            completed_ref.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Ok(())
                });
            }
        }
        drop(sender);
        if let Some(handle) = progress_thread {
            join_thread(handle, "진행률 스레드 패닉 발생")?;
        }
        let (num, data) = join_thread(writer_thread, "쓰기 스레드 패닉 발생")?;
        last_generated_num = Some(num);
        last_generated_data = Some(data);
        Ok(())
    })?;
    print_progress(
        requested_count,
        requested_count,
        &start_time,
        &mut [0u8; 16],
        &mut [0u8; 16],
    )?;
    println!("\n총 {requested_count}개의 데이터 생성 완료 ({FILE_NAME} 저장됨).\n");
    stdout().flush()?;
    let final_data = last_generated_data.ok_or("최종 데이터를 가져오지 못했습니다.")?;
    if *IS_TERMINAL {
        let mut buffer = [0; BUFFER_SIZE];
        let bytes_written_console = format_data_into_buffer(&final_data, &mut buffer, true)?;
        write_slice_to_console(&buffer[..bytes_written_console])?;
    } else {
        let mut buffer = [0; BUFFER_SIZE];
        let bytes_written_file = format_data_into_buffer(&final_data, &mut buffer, false)?;
        write_slice_to_console(&buffer[..bytes_written_file])?;
    }
    Ok(last_generated_num.ok_or("최종 데이터의 num_64를 가져오지 못했습니다.")?)
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
    let mut out = stdout().lock();
    write!(
        out,
        "\r{} {:>3}% ({}/{}) | 소요: {} | ETA: {} \x1b[K",
        bar,
        (progress * 100.0).floor() as u32,
        completed,
        total,
        String::from_utf8_lossy(&elapsed_buf[..elapsed_len]),
        String::from_utf8_lossy(&eta_buf[..eta_len]),
    )?;
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
    let mut out = *b"00:00.0";
    out[0..2].copy_from_slice(&TWO_DIGITS[minutes]);
    out[3..5].copy_from_slice(&TWO_DIGITS[sec_whole]);
    out[6] = DIGITS[tenths];
    buf[..7].copy_from_slice(&out);
    Ok(7)
}
fn join_thread<T>(handle: ScopedJoinHandle<'_, Result<T>>, panic_msg: &'static str) -> Result<T> {
    let join_err = ioErr::other(panic_msg);
    handle.join().map_err(|_| join_err)?
}
