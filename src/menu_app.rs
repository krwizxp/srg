use crate::{
    FILE_NAME,
    diagnostic::Result,
    file_output::OutputFile,
    input::{get_validated_input, read_line_reuse_limited, read_u64_hex_input},
    random_data::RandomDataSet,
    time::{ParsedServer, ServerTimeSession, TargetTimeOfDay, TriggerAction},
};
#[cfg(target_arch = "x86_64")]
use crate::{
    batch::{MAX_BATCH_GENERATE_COUNT, regenerate_with_count},
    hardware_rng::{HardwareRandomSource, HardwareRng},
    input::{LadderEntryMode, parse_regular_f64, read_ladder_entries, read_parsed_value},
    ladder::write_ladder_results,
    random_number::{
        FLOAT_INPUT_ERROR, MIN_ALLOWED_INTEGER_VALUE, generate_random_float,
        generate_random_integer,
    },
};
use alloc::borrow::Cow;
use core::{error::Error, iter::successors, result::Result as CoreResult};
use std::io::{self, Write, stderr, stdout};
#[cfg(target_arch = "x86_64")]
const BATCH_COUNT_INPUT_MAX_BYTES: usize = 64;
const MENU_SELECTION_INPUT_MAX_BYTES: usize = 256;
cfg_select! {
    target_arch = "x86_64" => {
        const MENU: &str = concat!(
            "\n1: 사다리타기 실행, 2: 무작위 숫자 생성, 3: 데이터 생성(1회), ",
            "4: 데이터 생성(여러 회), 5: 서버 시간 확인, 6: 파일 초기화, ",
            "7: num_64/supp 수동 입력 변환, 기타: 종료\n선택해 주세요: ",
        );
    }
    _ => {
        const MENU: &str = concat!(
            "\n5: 서버 시간 확인, 6: 파일 초기화, 7: num_64/supp 수동 입력 변환, 기타(1~4 제외): 종료\n",
            "(참고: 이 플랫폼에서는 하드웨어 RNG 관련 기능이 비활성화됩니다)\n",
            "선택해 주세요: ",
        );
    }
}
pub(super) struct MenuApp {
    pub input_buffer: String,
    #[cfg(target_arch = "x86_64")]
    pub ladder_results_storage: String,
    #[cfg(target_arch = "x86_64")]
    pub num_64: u64,
    pub output_file: OutputFile,
    #[cfg(target_arch = "x86_64")]
    pub rng: HardwareRng,
}
impl MenuApp {
    fn execute_command(
        &mut self,
        command: u8,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<bool> {
        match command {
            b'5' => {
                self.handle_server_time_command(out, err)?;
                return Ok(true);
            }
            b'6' => {
                self.output_file.clear()?;
                writeln!(out, "파일 '{FILE_NAME}'를 초기화했습니다.")?;
                return Ok(true);
            }
            b'7' => {
                self.handle_manual_input_command(out, err)?;
                return Ok(true);
            }
            _ => {}
        }
        cfg_select! {
            target_arch = "x86_64" => {
                match command {
                    b'1' => self.handle_ladder_command(out, err)?,
                    b'2' => self.handle_random_number_command(out, err)?,
                    b'3' | b'4' => {
                        if !prepare_hw_rng_menu_command(&self.rng, out)? {
                            return Ok(true);
                        }
                        let requested_count = if command == b'3' {
                            1
                        } else {
                            let input_buffer = &mut self.input_buffer;
                            let count_prompt =
                                format_args!("\n생성할 데이터 개수를 입력해 주세요: ");
                            loop {
                                match read_line_reuse_limited(
                                    count_prompt,
                                    input_buffer,
                                    out,
                                    BATCH_COUNT_INPUT_MAX_BYTES,
                                )?
                                .parse::<usize>()
                                .ok()
                                {
                                    Some(0) => writeln!(err, "1 이상의 값을 입력해 주세요.")?,
                                    Some(count) if count > MAX_BATCH_GENERATE_COUNT => writeln!(
                                        err,
                                        "대량 생성 개수는 최대 {MAX_BATCH_GENERATE_COUNT}건까지 입력할 수 있습니다."
                                    )?,
                                    Some(count) => break count,
                                    None => writeln!(err, "유효한 숫자를 입력해 주세요.")?,
                                }
                            }
                        };
                        let completion = regenerate_with_count(
                            &mut self.output_file,
                            &self.rng,
                            requested_count,
                            command == b'4',
                            out,
                        )?;
                        self.rng.write_rdseed_fallback_notice(err)?;
                        if let Some(next_num_64) = completion {
                            self.num_64 = next_num_64;
                        }
                    }
                    _ => return Ok(false),
                }
                Ok(true)
            }
            _ => {
                match command {
                    b'1'..=b'4' => writeln!(
                        out,
                        "{}번 메뉴: 이 기능은 x86_64 전용이라 현재 플랫폼에서는 비활성화되어 있습니다.",
                        char::from(command),
                    )?,
                    _ => return Ok(false),
                }
                Ok(true)
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    fn handle_ladder_command(&mut self, out: &mut dyn Write, err: &mut dyn Write) -> Result<()> {
        if !prepare_hw_rng_menu_command(&self.rng, out)? {
            return Ok(());
        }
        let players_storage = &mut self.input_buffer;
        let results_storage = &mut self.ladder_results_storage;
        let n = read_ladder_entries(
            format_args!("\n사다리타기 플레이어를 입력해 주세요 (쉼표(,)로 구분, 2~512명): "),
            (&mut *out, &mut *err),
            players_storage,
            LadderEntryMode::Players,
        )?;
        read_ladder_entries(
            format_args!("사다리타기 결과값을 입력해 주세요 (쉼표(,)로 구분, {n}개 필요): "),
            (&mut *out, &mut *err),
            results_storage,
            LadderEntryMode::Results { expected_count: n },
        )?;
        write_ladder_results(
            players_storage.trim().split(',').map(str::trim),
            results_storage.trim().split(',').map(str::trim),
            self.num_64,
            &self.rng,
            out,
        )?;
        self.rng.write_rdseed_fallback_notice(err)
    }
    fn handle_manual_input_command(
        &mut self,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<()> {
        writeln!(out, "\nnum_64/supp 수동 입력 변환 모드")?;
        self.input_buffer.clear();
        let manual_num_64 = read_u64_hex_input(
            format_args!(
                "num_64를 입력해 주세요 (최소값 예: 0 또는 0x0, 최대값 예: {max_u64} 또는 0x{max_u64:X}): ",
                max_u64 = u64::MAX
            ),
            &mut self.input_buffer,
            out,
            err,
        )?;
        #[cfg(target_arch = "x86_64")]
        {
            self.num_64 = manual_num_64;
        }
        let mut supp_input_count = 0_usize;
        let mut next_supp = |reason: &'static str| -> Result<u64> {
            supp_input_count = supp_input_count.strict_add(1);
            read_u64_hex_input(
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
            )
        };
        let data = RandomDataSet {
            num_64: manual_num_64,
            ..Default::default()
        }
        .populate(&mut next_supp)?;
        self.output_file.persist_and_print(&data)
    }
    #[cfg(target_arch = "x86_64")]
    fn handle_random_number_command(
        &mut self,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<()> {
        if !prepare_hw_rng_menu_command(&self.rng, out)? {
            return Ok(());
        }
        let num_64 = self.num_64;
        let input_buffer = &mut self.input_buffer;
        writeln!(out, "\n무작위 숫자 생성 타입 선택:")?;
        let selection = read_line_reuse_limited(
            format_args!("1: 정수 생성, 2: 실수 생성, 기타: 취소\n선택해 주세요: "),
            input_buffer,
            out,
            MENU_SELECTION_INPUT_MAX_BYTES,
        )?;
        match selection.as_bytes() {
            b"1" => {
                writeln!(
                    out,
                    "\n무작위 정수 생성기(지원 범위: -9223372036854775807 ~ 9223372036854775807)"
                )?;
                let min_value = loop {
                    let value = read_parsed_value(
                        format_args!("최솟값을 입력해 주세요 ({MIN_ALLOWED_INTEGER_VALUE} 이상): "),
                        input_buffer,
                        out,
                        err,
                        "유효한 정수 형식이 아닙니다.",
                        |line| line.parse::<i64>().ok(),
                    )?;
                    if value >= MIN_ALLOWED_INTEGER_VALUE {
                        break value;
                    }
                    writeln!(
                        err,
                        "{MIN_ALLOWED_INTEGER_VALUE} 이상의 값을 입력해 주세요."
                    )?;
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
                generate_random_integer(min_value, max_value, num_64, out, &self.rng)?;
            }
            b"2" => {
                writeln!(out, "\n무작위 실수 생성기")?;
                let min_value = read_parsed_value(
                    format_args!("최솟값을 입력해 주세요: "),
                    input_buffer,
                    out,
                    err,
                    FLOAT_INPUT_ERROR,
                    parse_regular_f64,
                )?;
                let max_value = loop {
                    let value = read_parsed_value(
                        format_args!("최댓값을 입력해 주세요: "),
                        input_buffer,
                        out,
                        err,
                        FLOAT_INPUT_ERROR,
                        parse_regular_f64,
                    )?;
                    if value >= min_value {
                        break value;
                    }
                    writeln!(err, "최댓값은 최솟값보다 크거나 같아야 합니다.")?;
                };
                generate_random_float(min_value, max_value, num_64, out, &self.rng)?;
            }
            _ => {
                writeln!(out, "무작위 숫자 생성을 취소합니다.")?;
                return self.rng.write_rdseed_fallback_notice(err);
            }
        }
        self.rng.write_rdseed_fallback_notice(err)
    }
    fn handle_server_time_command(
        &mut self,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<()> {
        let host = get_validated_input(
            "확인할 서버 주소를 입력하세요 (스킴 생략 시 HTTPS, 평문 HTTP는 http:// 명시 / 예: www.example.com): ",
            &mut self.input_buffer,
            &mut *out,
            |raw_input| -> CoreResult<ParsedServer, Cow<'static, str>> {
                if raw_input.is_empty() {
                    return Err(Cow::Borrowed("서버 주소를 비워둘 수 없습니다."));
                }
                raw_input.parse::<ParsedServer>().map_err(|source| {
                    Cow::Owned(format!("서버 주소가 올바르지 않습니다: {source}"))
                })
            },
        )?;
        let requested_target_time = get_validated_input(
            "액션 실행 목표 시간을 입력하세요 (예: 20:00:00 / 건너뛰려면 Enter): ",
            &mut self.input_buffer,
            &mut *out,
            |raw_input| -> CoreResult<Option<TargetTimeOfDay>, &'static str> {
                if raw_input.is_empty() {
                    return Ok(None);
                }
                raw_input.parse::<TargetTimeOfDay>().map(Some)
            },
        )?;
        let scheduled_trigger = match requested_target_time {
            Some(target_time) => Some((
                target_time,
                get_validated_input(
                    "수행할 동작을 선택하세요 (1: 마우스 왼쪽 클릭, 2: F5 입력): ",
                    &mut self.input_buffer,
                    &mut *out,
                    |selection| -> CoreResult<TriggerAction, &'static str> {
                        match selection.as_bytes() {
                            b"1" => Ok(TriggerAction::LeftClick),
                            b"2" => Ok(TriggerAction::F5Press),
                            _ => Err("잘못된 입력입니다. 1 또는 2를 입력해주세요."),
                        }
                    },
                )?,
            )),
            None => None,
        };
        ServerTimeSession {
            host,
            scheduled_trigger,
            stop_after: None,
        }
        .run_loop(out, err)?;
        writeln!(out, "\n서버 시간 확인을 종료합니다.")?;
        Ok(())
    }
    pub(super) fn run(&mut self) -> Result<()> {
        let menu_prompt = format_args!("{MENU}");
        let mut out = stdout().lock();
        let mut err = stderr();
        loop {
            let command = match read_line_reuse_limited(
                menu_prompt,
                &mut self.input_buffer,
                &mut out,
                MENU_SELECTION_INPUT_MAX_BYTES,
            ) {
                Ok(command_str) if let &[command @ b'1'..=b'7'] = command_str.as_bytes() => command,
                Ok(_) => 0,
                Err(read_err) if read_err.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(read_err) => return Err(read_err.into()),
            };
            let keep_running = match self.execute_command(command, &mut out, &mut err) {
                Ok(keep_running) => keep_running,
                Err(command_err) => {
                    let root_error: &(dyn Error + 'static) = &command_err;
                    let unexpected_eof = successors(Some(root_error), |error| (*error).source())
                        .any(|error| {
                            error.downcast_ref::<io::Error>().is_some_and(|io_error| {
                                io_error.kind() == io::ErrorKind::UnexpectedEof
                            })
                        });
                    if unexpected_eof {
                        return Ok(());
                    }
                    return Err(command_err);
                }
            };
            if !keep_running {
                return Ok(());
            }
        }
    }
}
#[cfg(target_arch = "x86_64")]
fn prepare_hw_rng_menu_command(rng: &HardwareRng, out: &mut dyn Write) -> Result<bool> {
    if rng.source() != HardwareRandomSource::None {
        return Ok(true);
    }
    writeln!(
        out,
        "이 기능은 RDSEED/RDRAND를 지원하는 CPU에서만 사용할 수 있습니다."
    )?;
    Ok(false)
}
