use crate::{
    BUFFER_SIZE, FILE_NAME, IS_TERMINAL, UTF8_BOM,
    diagnostic::{Result, is_unexpected_eof},
    file_output::OutputFile,
    input::{get_validated_input, read_line_reuse_limited, read_u64_hex_input},
    output::{OutputTarget, format_data_into_buffer, prefix_slice, write_slice_to_console},
    random_data::RandomDataSet,
    time::{ParsedServer, ServerTimeSession, TargetTimeOfDay, TimeError, TriggerAction},
};
cfg_select! {
    target_arch = "x86_64" => {
        use crate::{
            batch::{MAX_BATCH_GENERATE_COUNT, regenerate_with_count},
            hardware_rng::{HardwareRandomSource, HardwareRng},
            input::{LadderEntryMode, read_ladder_entries},
            ladder::write_ladder_results,
            random_number::generate_random_number,
        };
        use crate::random_number::RandomNumberMode;
    }
    _ => {}
}
use alloc::borrow::Cow;
use core::result::Result as CoreResult;
use std::io::{self, Seek as _, SeekFrom, Write, stderr, stdout};
cfg_select! {
    target_arch = "x86_64" => {
        const BATCH_COUNT_INPUT_MAX_BYTES: usize = 64;
    }
    _ => {}
}
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
                let writer = self.output_file.writer();
                Write::flush(&mut *writer)?;
                writer.get_ref().set_len(0)?;
                writer.seek(SeekFrom::Start(0))?;
                Write::write_all(&mut *writer, UTF8_BOM)?;
                Write::flush(writer)?;
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
                        let next_num_64 = regenerate_with_count(
                            &mut self.output_file,
                            &self.rng,
                            requested_count,
                            out,
                        )?;
                        self.rng.write_rdseed_fallback_notice(err)?;
                        self.num_64 = next_num_64;
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
    cfg_select! {
        target_arch = "x86_64" => {
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
        }
        _ => {}
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
        cfg_select! {
            target_arch = "x86_64" => {
                self.num_64 = manual_num_64;
            }
            _ => {}
        }
        let mut supp_input_count = 0_usize;
        let mut next_supp = |reason: &'static str| -> Result<u64> {
            supp_input_count = supp_input_count.wrapping_add(1);
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
            Ok(supp)
        };
        let data = RandomDataSet {
            num_64: manual_num_64,
            ..Default::default()
        }
        .populate(&mut next_supp)?;
        let mut buffer = [0_u8; BUFFER_SIZE];
        let file_len = format_data_into_buffer(&data, &mut buffer, OutputTarget::File)?;
        let writer = self.output_file.writer();
        Write::write_all(&mut *writer, prefix_slice(&buffer, file_len)?)?;
        Write::flush(writer)?;
        let console_len = if *IS_TERMINAL {
            format_data_into_buffer(&data, &mut buffer, OutputTarget::Console)?
        } else {
            file_len
        };
        write_slice_to_console(prefix_slice(&buffer, console_len)?)?;
        Ok(())
    }
    cfg_select! {
        target_arch = "x86_64" => {
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
                        generate_random_number(
                            RandomNumberMode::Integer,
                            num_64,
                            input_buffer,
                            out,
                            err,
                            &self.rng,
                        )?;
                    }
                    b"2" => {
                        generate_random_number(
                            RandomNumberMode::Float,
                            num_64,
                            input_buffer,
                            out,
                            err,
                            &self.rng,
                        )?;
                    }
                    _ => writeln!(out, "무작위 숫자 생성을 취소합니다.")?,
                }
                self.rng.write_rdseed_fallback_notice(err)
            }
        }
        _ => {}
    }
    fn handle_server_time_command(
        &mut self,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<()> {
        let time_run_result = (|| -> CoreResult<(), TimeError> {
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
        })();
        match time_run_result {
            Ok(()) => {}
            Err(time_err) if is_unexpected_eof(&time_err) => {}
            Err(time_err) if time_err.is_io() => return Err(time_err.into()),
            Err(time_err) => writeln!(err, "서버 시간 확인 중 오류 발생: {time_err}")?,
        }
        Ok(())
    }
    pub(super) fn run(&mut self) -> Result<()> {
        let menu_prompt = format_args!("{MENU}");
        loop {
            let command = {
                let mut prompt_out = stdout().lock();
                match read_line_reuse_limited(
                    menu_prompt,
                    &mut self.input_buffer,
                    &mut prompt_out,
                    MENU_SELECTION_INPUT_MAX_BYTES,
                ) {
                    Ok(command_str) if let &[command @ b'1'..=b'7'] = command_str.as_bytes() => {
                        command
                    }
                    Ok(_) => 0,
                    Err(read_err) if read_err.kind() == io::ErrorKind::UnexpectedEof => {
                        return Ok(());
                    }
                    Err(read_err) => return Err(read_err.into()),
                }
            };
            let mut out = stdout();
            let mut err = stderr();
            let keep_running = match self.execute_command(command, &mut out, &mut err) {
                Ok(keep_running) => keep_running,
                Err(command_err) if is_unexpected_eof(&command_err) => {
                    return Ok(());
                }
                Err(command_err) => return Err(command_err),
            };
            if !keep_running {
                return Ok(());
            }
        }
    }
}
cfg_select! {
    target_arch = "x86_64" => {
        fn prepare_hw_rng_menu_command(rng: &HardwareRng, out: &mut dyn Write) -> Result<bool> {
            match rng.source() {
                HardwareRandomSource::None => {
                    writeln!(
                        out,
                        "이 기능은 RDSEED/RDRAND를 지원하는 CPU에서만 사용할 수 있습니다."
                    )?;
                    Ok(false)
                }
                HardwareRandomSource::RdSeed | HardwareRandomSource::RdRand => Ok(true),
            }
        }
    }
    _ => {}
}
