#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
compile_error!("SRG currently supports only Windows, Linux, and macOS.");
mod batch;
mod buffmt;
mod numeric;
mod output;
mod time;
use self::{
    batch::regenerate_multiple,
    output::{
        format_data_into_buffer, prefix_slice, write_buffer_to_file_guard, write_slice_to_console,
    },
};
use numeric::{low_u8_from_u32, low_u8_from_u64, low_u16_from_u64};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_rdrand64_step, _rdseed64_step};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::unix::fs::OpenOptionsExt;
#[cfg(windows)]
use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};
use std::{
    any::Any,
    char::from_u32,
    error::Error,
    fmt::{Arguments, Display},
    fs::{self, File},
    hint::spin_loop,
    io::{
        BufWriter, Error as ioErr, ErrorKind, IsTerminal as _, Result as IoRst, Write as _, stdin,
        stdout,
    },
    is_x86_feature_detected,
    path::Path,
    process::ExitCode,
    result::Result as stdResult,
    sync::{LazyLock, Mutex, MutexGuard},
};
#[cfg(target_os = "linux")]
const OPEN_NOFOLLOW_FLAG: i32 = 0x2_0000;
#[cfg(target_os = "macos")]
const OPEN_NOFOLLOW_FLAG: i32 = 0x0100;
#[cfg(windows)]
const FILE_ATTRIBUTE_REPARSE_POINT_FLAG: u32 = 0x0000_0400;
#[cfg(windows)]
const FILE_FLAG_OPEN_REPARSE_POINT_FLAG: u32 = 0x0020_0000;
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
enum RngSource {
    RdSeed,
    RdRand,
    None,
}
#[cfg(target_arch = "x86_64")]
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
#[cfg(not(target_arch = "x86_64"))]
static RNG_SOURCE: LazyLock<RngSource> = LazyLock::new(|| RngSource::None);
static GLYPHS: [char; 16] = [
    '🌅', '🐦', '👫', '🦕', '🌘', '🎈', '⛵', '🕷', '🦋', '🌀', '🧊', '🐟', '⛺', '🚀', '🌳', '🔯',
];
static IS_TERMINAL: LazyLock<bool> = LazyLock::new(|| stdout().is_terminal());
type Result<T> = stdResult<T, Box<dyn Error + Send + Sync + 'static>>;
const FILE_NAME: &str = "random_data.txt";
const UTF8_BOM: &[u8; 3] = b"\xEF\xBB\xBF";
const BUFFER_SIZE: usize = 1016;
const BUFFERS_PER_WORKER: usize = 8;
type DataBuffer = Box<[u8; BUFFER_SIZE]>;
const fn bitmask_const<const B: u8>() -> u64 {
    let b = if B > 64 { 64 } else { B };
    match b {
        0 => 0,
        64 => u64::MAX,
        _ => (1_u64 << b) - 1,
    }
}
const fn galaxy_coord<const SUB: u16, const ADD: u16>(value: u16) -> u16 {
    let a = value.wrapping_sub(SUB);
    let b = value.wrapping_add(ADD);
    if a < b { a } else { b }
}
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
fn make_bin8_table() -> [[u8; 8]; 256] {
    let mut table = [[0_u8; 8]; 256];
    let mut byte = 0_u8;
    for row in &mut table {
        for (bit, slot) in row.iter_mut().enumerate() {
            let shift = 7 - bit;
            *slot = if ((byte >> shift) & 1) == 1 {
                b'1'
            } else {
                b'0'
            };
        }
        byte = byte.wrapping_add(1);
    }
    table
}
static BIN8_TABLE: LazyLock<[[u8; 8]; 256]> = LazyLock::new(make_bin8_table);
fn make_hex_byte_table() -> [[u8; 2]; 256] {
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
}
static HEX_BYTE_TABLE: LazyLock<[[u8; 2]; 256]> = LazyLock::new(make_hex_byte_table);
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
type SupplementalProvider<'provider> =
    dyn FnMut(&'static str) -> Result<RandomBitBuffer> + 'provider;
fn is_unexpected_eof(err: &(dyn Error + 'static)) -> bool {
    err.downcast_ref::<ioErr>()
        .is_some_and(|io_err| io_err.kind() == ErrorKind::UnexpectedEof)
}
#[cfg(target_arch = "x86_64")]
fn initialize_num_64(file_mutex: &Mutex<BufWriter<File>>) -> Result<u64> {
    if hw_rng_features_available() {
        process_single_random_data(file_mutex)
    } else {
        eprintln!(
            "[경고] RDSEED/RDRAND를 지원하지 않아 하드웨어 RNG 기능(메뉴 1~4)을 비활성화합니다. 메뉴 5/7은 사용 가능합니다."
        );
        Ok(0)
    }
}
#[cfg(not(target_arch = "x86_64"))]
fn initialize_num_64(_file_mutex: &Mutex<BufWriter<File>>) -> Result<u64> {
    Ok(0)
}
fn run_menu_command(
    command: &str,
    file_mutex: &Mutex<BufWriter<File>>,
    num_64: &mut u64,
    input_buffer: &mut String,
) -> Result<bool> {
    match command {
        "1" => menu_ladder(*num_64, input_buffer)?,
        "2" => menu_random_number(*num_64, input_buffer)?,
        "3" => menu_generate_single(file_mutex, num_64)?,
        "4" => menu_generate_multiple(file_mutex, num_64, input_buffer)?,
        "5" => menu_time_sync(),
        "6" => menu_delete_file(),
        "7" => menu_manual(file_mutex, num_64, input_buffer)?,
        _ => return Ok(false),
    }
    Ok(true)
}
#[cfg(target_arch = "x86_64")]
fn menu_ladder(num_64: u64, input_buffer: &mut String) -> Result<()> {
    if !hw_rng_features_available() {
        print_hw_rng_only_feature_disabled();
        return Ok(());
    }
    ladder_game(num_64, input_buffer)
}
#[cfg(not(target_arch = "x86_64"))]
fn menu_ladder(_num_64: u64, _input_buffer: &mut String) -> Result<()> {
    print_x86_64_only_feature_disabled();
    Ok(())
}
#[cfg(target_arch = "x86_64")]
fn menu_random_number(num_64: u64, input_buffer: &mut String) -> Result<()> {
    if !hw_rng_features_available() {
        print_hw_rng_only_feature_disabled();
        return Ok(());
    }
    println!("\n무작위 숫자 생성 타입 선택:");
    match read_line_reuse(
        format_args!("1: 정수 생성, 2: 실수 생성, 기타: 취소\n선택해 주세요: "),
        input_buffer,
    )? {
        "1" => generate_random_integer(num_64, input_buffer)?,
        "2" => generate_random_float(num_64, input_buffer)?,
        _ => println!("무작위 숫자 생성을 취소합니다."),
    }
    Ok(())
}
#[cfg(not(target_arch = "x86_64"))]
fn menu_random_number(_num_64: u64, _input_buffer: &mut String) -> Result<()> {
    print_x86_64_only_feature_disabled();
    Ok(())
}
#[cfg(target_arch = "x86_64")]
fn menu_generate_single(file_mutex: &Mutex<BufWriter<File>>, num_64: &mut u64) -> Result<()> {
    if !hw_rng_features_available() {
        print_hw_rng_only_feature_disabled();
        return Ok(());
    }
    ensure_file_exists_and_reopen(file_mutex)?;
    *num_64 = process_single_random_data(file_mutex)?;
    Ok(())
}
#[cfg(not(target_arch = "x86_64"))]
fn menu_generate_single(_file_mutex: &Mutex<BufWriter<File>>, _num_64: &mut u64) -> Result<()> {
    print_x86_64_only_feature_disabled();
    Ok(())
}
#[cfg(target_arch = "x86_64")]
fn menu_generate_multiple(
    file_mutex: &Mutex<BufWriter<File>>,
    num_64: &mut u64,
    input_buffer: &mut String,
) -> Result<()> {
    if !hw_rng_features_available() {
        print_hw_rng_only_feature_disabled();
        return Ok(());
    }
    *num_64 = regenerate_multiple(file_mutex, input_buffer)?;
    Ok(())
}
#[cfg(not(target_arch = "x86_64"))]
fn menu_generate_multiple(
    _file_mutex: &Mutex<BufWriter<File>>,
    _num_64: &mut u64,
    _input_buffer: &mut String,
) -> Result<()> {
    print_x86_64_only_feature_disabled();
    Ok(())
}
fn menu_time_sync() {
    if let Err(e) = time::run()
        && e.io_kind() != Some(ErrorKind::UnexpectedEof)
    {
        eprintln!("서버 시간 확인 중 오류 발생: {e}");
    }
}
#[cfg(target_arch = "x86_64")]
fn menu_delete_file() {
    if let Err(e) = fs::remove_file(FILE_NAME) {
        eprintln!("{e}");
    } else {
        println!("파일 '{FILE_NAME}'를 삭제했습니다.");
    }
}
#[cfg(not(target_arch = "x86_64"))]
fn menu_delete_file() {
    print_x86_64_only_feature_disabled();
}
fn menu_manual(
    file_mutex: &Mutex<BufWriter<File>>,
    num_64: &mut u64,
    input_buffer: &mut String,
) -> Result<()> {
    ensure_file_exists_and_reopen(file_mutex)?;
    *num_64 = process_manual_random_data(file_mutex, input_buffer)?;
    Ok(())
}
fn main() -> Result<ExitCode> {
    let file_mutex = Mutex::new(open_or_create_file()?);
    let mut num_64 = initialize_num_64(&file_mutex)?;
    let mut input_buffer = String::with_capacity(256);
    let menu_prompt = format_args!("{MENU}");
    loop {
        let command = match read_line_reuse(menu_prompt, &mut input_buffer) {
            Ok(cmd) => cmd.to_owned(),
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                return Ok(ExitCode::SUCCESS);
            }
            Err(e) => return Err(e.into()),
        };
        let keep_running = match run_menu_command(
            command.as_str(),
            &file_mutex,
            &mut num_64,
            &mut input_buffer,
        ) {
            Ok(keep_running) => keep_running,
            Err(e) if is_unexpected_eof(e.as_ref()) => return Ok(ExitCode::SUCCESS),
            Err(e) => return Err(e),
        };
        if !keep_running {
            return Ok(ExitCode::SUCCESS);
        }
    }
}
#[cfg(not(target_arch = "x86_64"))]
fn print_x86_64_only_feature_disabled() {
    println!("이 기능은 x86_64 전용이라 현재 플랫폼에서는 비활성화되어 있습니다.");
}
#[cfg(target_arch = "x86_64")]
fn hw_rng_features_available() -> bool {
    !matches!(*RNG_SOURCE, RngSource::None)
}
#[cfg(target_arch = "x86_64")]
fn print_hw_rng_only_feature_disabled() {
    println!("이 기능은 RDSEED/RDRAND를 지원하는 CPU에서만 사용할 수 있습니다.");
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
fn validate_existing_output_path(path: &Path) -> Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(Box::new(err)),
    };
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
    Ok(())
}
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn open_output_file_without_following_links(path: &Path) -> Result<File> {
    Ok(File::options()
        .read(true)
        .append(true)
        .create(true)
        .custom_flags(OPEN_NOFOLLOW_FLAG)
        .open(path)?)
}
#[cfg(windows)]
fn open_output_file_without_following_links(path: &Path) -> Result<File> {
    Ok(File::options()
        .read(true)
        .append(true)
        .create(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT_FLAG)
        .open(path)?)
}
#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn open_output_file_without_following_links(_path: &Path) -> Result<File> {
    Err(invalid_output_path_err(
        "지원되지 않는 운영체제입니다. Windows, Linux, macOS만 지원합니다.",
    ))
}
fn validate_open_output_handle(file: &File) -> Result<()> {
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
    Ok(())
}
fn write_utf8_bom_if_empty(file: &mut File) -> Result<()> {
    if file.metadata()?.len() == 0 {
        file.write_all(UTF8_BOM)?;
        file.flush()?;
    }
    Ok(())
}
fn open_or_create_file() -> Result<BufWriter<File>> {
    let path = Path::new(FILE_NAME);
    validate_existing_output_path(path)?;
    let mut file = open_output_file_without_following_links(path)?;
    validate_open_output_handle(&file)?;
    write_utf8_bom_if_empty(&mut file)?;
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
    }
    if *IS_TERMINAL {
        let console_len = format_data_into_buffer(data, &mut buffer, true)?;
        write_slice_to_console(prefix_slice(&buffer, console_len)?)?;
    } else {
        write_slice_to_console(prefix_slice(&buffer, file_len)?)?;
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
            "num_64를 입력해 주세요 (최소값 예: 0 또는 0x0, 최대값 예: {max_u64} 또는 0x{max_u64:X}): ",
            max_u64 = u64::MAX
        ),
        input_buffer,
    )?;
    let mut supp_input_count = 0_usize;
    let mut next_supp = |reason: &'static str| -> Result<RandomBitBuffer> {
        supp_input_count += 1;
        let supp = read_parse_u64(
            format_args!(
                "supp 값 #{supp_input_count} 입력 ({reason}, 최소값 예: 0 또는 0x0, 최대값 예: {max_u64} 또는 0x{max_u64:X}): ",
                max_u64 = u64::MAX
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
        for byte in u64_to_be_bytes(lucky_star_source) {
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
    for shift in [48_u32, 32, 16, 0] {
        let candidate = u32::from(low_u16_from_u64(supp_value >> shift));
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
    for (slot, shift) in hangul.iter_mut().zip([48_u32, 32, 16, 0]) {
        let mut syllable_index = u32::from(low_u16_from_u64(num >> shift));
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
    let upper_32_bits = (u32::from(low_u16_from_u64(num >> 48_u32)) << 16_u32)
        | u32::from(low_u16_from_u64(num >> 32_u32));
    let lower_32_bits =
        (u32::from(low_u16_from_u64(num >> 16_u32)) << 16_u32) | u32::from(low_u16_from_u64(num));
    let upper_ratio = f64::from(upper_32_bits) * U32_MAX_INV;
    let lower_ratio = f64::from(lower_32_bits) * U32_MAX_INV;
    data.kor_coords = (
        5.504_167_f64.mul_add(upper_ratio, 33.112_500),
        7.263_056_f64.mul_add(lower_ratio, 124.609_722),
    );
    data.world_coords = (
        180.0_f64.mul_add(upper_ratio, -90.0),
        360.0_f64.mul_add(lower_ratio, -180.0),
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
            1 => u64::from(data.solar_system_index >> 8_u32),
            2 => u64::from(data.solar_system_index >> 4_u32),
            3 => u64::from(data.solar_system_index),
            _ => num >> (36_usize - (idx - 4) * 4),
        };
        let nibble = usize::from(low_u8_from_u64(nibble_source & 0xF));
        if let Some(glyph) = GLYPHS.get(nibble).copied() {
            *slot = glyph;
        }
    }
    Ok(data)
}
fn get_hardware_random() -> Result<u64> {
    match *RNG_SOURCE {
        RngSource::RdSeed => Ok(rdseed_impl()),
        RngSource::RdRand => rdrand_impl(),
        RngSource::None => no_hw_rng(),
    }
}
#[cfg(target_arch = "x86_64")]
fn rdseed_impl() -> u64 {
    let mut v = 0_u64;
    loop {
        // SAFETY: `RNG_SOURCE` only routes here after confirming `rdseed` support,
        // and the intrinsic writes to the valid mutable pointer to `v`.
        if unsafe { _rdseed64_step(&mut v) } == 1_i32 {
            break;
        }
        spin_loop();
    }
    v
}
#[cfg(not(target_arch = "x86_64"))]
const fn rdseed_impl() -> u64 {
    0
}
#[cfg(target_arch = "x86_64")]
fn rdrand_impl() -> Result<u64> {
    let mut v = 0_u64;
    for _ in 0_u8..10_u8 {
        // SAFETY: `RNG_SOURCE` only routes here after confirming `rdrand` support,
        // and the intrinsic writes to the valid mutable pointer to `v`.
        if unsafe { _rdrand64_step(&mut v) } == 1_i32 {
            return Ok(v);
        }
        spin_loop();
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
    for byte in u64_to_be_bytes(v) {
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
            && let Some(slot) = data.password.get_mut(usize::from(data.password_len))
        {
            *slot = ch;
            data.password_len += 1;
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
    let number = (byte % modulus) + 1;
    let mask = 1_u64 << number;
    if (*seen & mask) == 0 {
        let Some(slot) = numbers.get_mut(*next_idx) else {
            return false;
        };
        *slot = number;
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
fn read_line_reuse<'buffer>(prompt: Arguments, buffer: &'buffer mut String) -> IoRst<&'buffer str> {
    buffer.clear();
    {
        let mut out = stdout().lock();
        out.write_fmt(prompt)?;
        out.flush()?;
    }
    let bytes_read = stdin().read_line(buffer)?;
    if bytes_read == 0 {
        return Err(ioErr::new(
            ErrorKind::UnexpectedEof,
            "표준 입력이 종료되었습니다.",
        ));
    }
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
        let mut count = 0_usize;
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
        let slot = players_array
            .get_mut(i)
            .ok_or_else(|| ioErr::other("플레이어 배열 인덱스 범위 초과"))?;
        *slot = part.trim();
    }
    let mut result_input_buffer = String::with_capacity(256);
    let mut results_storage = String::with_capacity(256);
    let mut results_array: [&str; MAX_PLAYERS] = [""; MAX_PLAYERS];
    let result_prompt =
        format_args!("사다리타기 결과값을 입력해 주세요 (쉼표(,)로 구분, {n}개 필요): ");
    loop {
        let line = read_line_reuse(result_prompt, &mut result_input_buffer)?;
        let mut count = 0_usize;
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
        let slot = results_array
            .get_mut(i)
            .ok_or_else(|| ioErr::other("결과 배열 인덱스 범위 초과"))?;
        *slot = part.trim();
    }
    println!("사다리타기 결과:");
    let mut indices = [0_usize; MAX_PLAYERS];
    let indices_slice = indices
        .get_mut(..n)
        .ok_or_else(|| ioErr::other("인덱스 배열 슬라이스 범위 초과"))?;
    for (i, slot) in indices_slice.iter_mut().enumerate() {
        *slot = i;
    }
    let mut current_seed = num_64;
    for i in (1..n).rev() {
        current_seed ^= get_hardware_random()?;
        let upper_bound = u64::try_from(i + 1)
            .map_err(|err| boxed_other_with_source("인덱스 상한 변환 실패", err))?;
        let swap_index_u64 = random_bounded(upper_bound, current_seed)?;
        let swap_index = usize::try_from(swap_index_u64)
            .map_err(|err| boxed_other_with_source("인덱스 변환 실패", err))?;
        indices_slice.swap(i, swap_index);
    }
    let players = players_array
        .get(..n)
        .ok_or_else(|| ioErr::other("플레이어 슬라이스 범위 초과"))?;
    for (player, &result_index) in players.iter().zip(indices_slice.iter()) {
        let result = results_array
            .get(result_index)
            .copied()
            .ok_or_else(|| ioErr::other("결과 인덱스 범위 초과"))?;
        println!("{player} -> {result}");
    }
    Ok(())
}
fn generate_random_integer(seed_modifier: u64, input_buffer: &mut String) -> Result<()> {
    const MIN_ALLOWED_VALUE: i64 = i64::MIN + 1;
    println!(
        "\n무작위 정수 생성기(지원 범위: {MIN_ALLOWED_VALUE} ~ {max_i64})",
        max_i64 = i64::MAX
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
fn read_parse_i64(prompt: Arguments, buffer: &mut String) -> Result<i64> {
    loop {
        match read_line_reuse(prompt, buffer)?.parse::<i64>() {
            Ok(n) => return Ok(n),
            Err(_) => eprintln!("유효한 정수 형식이 아닙니다.\n"),
        }
    }
}
fn read_parse_u64(prompt: Arguments, buffer: &mut String) -> Result<u64> {
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
                    "유효한 u64 형식이 아닙니다 (최소값 예: 0 또는 0x0, 최대값 예: {max_u64} 또는 0x{max_u64:X}).\n",
                    max_u64 = u64::MAX
                );
            }
        }
    }
}
fn unit_f64_from_u64(random_bits: u64) -> f64 {
    let upper_32 = (u32::from(low_u16_from_u64(random_bits >> 48_u32)) << 16_u32)
        | u32::from(low_u16_from_u64(random_bits >> 32_u32));
    let lower_32 = (u32::from(low_u16_from_u64(random_bits >> 16_u32)) << 16_u32)
        | u32::from(low_u16_from_u64(random_bits));
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
fn read_parse_f64(prompt: Arguments, buffer: &mut String) -> Result<f64> {
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
        let Ok(low_bits) = u64::try_from(m & u128::from(u64::MAX)) else {
            continue;
        };
        if low_bits >= threshold {
            let Ok(high_bits) = u64::try_from(m >> 64) else {
                continue;
            };
            return Ok(high_bits);
        }
    }
}
