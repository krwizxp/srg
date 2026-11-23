use crate::{
    FILE_NAME,
    diagnostic::Result,
    file_output::OutputFile,
    input::{
        DEFAULT_INPUT_LINE_MAX_BYTES, get_validated_input, read_line_reuse_limited,
        read_u64_hex_input,
    },
    korea_boundary::BoundaryCache,
    random_data::RandomDataSet,
    time::{ParsedServer, ServerTimeSession, TargetTimeOfDay, TriggerAction},
};
#[cfg(target_arch = "x86_64")]
use crate::{
    batch::regenerate_with_count,
    hardware_rng::{HardwareRandomSource, HardwareRng},
    input::{
        BATCH_COUNT_INPUT_MAX_BYTES, parse_regular_f64, read_ladder_entries, validate_batch_count,
    },
    ladder::write_ladder_results,
    random_number::{
        FLOAT_INPUT_ERROR, MIN_ALLOWED_INTEGER_VALUE, generate_random_float,
        generate_random_integer,
    },
};
use alloc::borrow::Cow;
use core::{error::Error, fmt::Arguments, iter::successors, result::Result as CoreResult};
use std::io::{self, Write, stderr, stdout};
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
const MENU_SELECTION_INPUT_MAX_BYTES: usize = 256;
pub(super) struct MenuApp {
    pub input_buffer: String,
    #[cfg(target_arch = "x86_64")]
    pub ladder_results_storage: String,
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
        if !matches!(command, b'1'..=b'4') {
            return Ok(false);
        }
        cfg_select! {
            target_arch = "x86_64" => {
                if self.rng.source() == HardwareRandomSource::None {
                    writeln!(out, "이 기능은 RDSEED/RDRAND를 지원하는 CPU에서만 사용할 수 있습니다.")?;
                    return Ok(true);
                }
                match command {
                    b'1' => {
                        let count = read_ladder_entries(
                            format_args!("\n사다리타기 플레이어를 입력해 주세요 (쉼표(,)로 구분, 2~512명): "),
                            (&mut *out, &mut *err),
                            &mut self.input_buffer,
                            None,
                        )?;
                        read_ladder_entries(
                            format_args!("사다리타기 결과값을 입력해 주세요 (쉼표(,)로 구분, {count}개 필요): "),
                            (&mut *out, &mut *err),
                            &mut self.ladder_results_storage,
                            Some(count),
                        )?;
                        write_ladder_results(
                            self.input_buffer.split(',').map(str::trim),
                            self.ladder_results_storage.split(',').map(str::trim),
                            &self.rng,
                            out,
                        )?;
                    }
                    b'2' => self.handle_random_number_command(out, err)?,
                    b'3' | b'4' => {
                        let count = if command == b'3' {
                            1
                        } else {
                            get_validated_input(
                                format_args!("\n생성할 데이터 개수를 입력해 주세요: "),
                                &mut self.input_buffer,
                                out,
                                Some(err),
                                BATCH_COUNT_INPUT_MAX_BYTES,
                                |raw| {
                                    raw.parse::<usize>()
                                        .map_err(|_parse| Cow::Borrowed("유효한 숫자를 입력해 주세요."))
                                        .and_then(validate_batch_count)
                                },
                            )?
                        };
                        regenerate_with_count(&mut self.output_file, &self.rng, count, command == b'4', out)?;
                    }
                    _ => return Ok(false),
                }
                self.rng.write_rdseed_fallback_notice(err)?;
                Ok(true)
            }
            _ => {
                writeln!(
                    out,
                    "{}번 메뉴: 이 기능은 x86_64 전용이라 현재 플랫폼에서는 비활성화되어 있습니다.",
                    char::from(command),
                )?;
                Ok(true)
            }
        }
    }
    fn handle_manual_input_command(
        &mut self,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<()> {
        writeln!(out, "\nnum_64/supp 수동 입력 변환 모드")?;
        let manual_num_64 = read_u64_hex_input(
            format_args!(
                "num_64를 입력해 주세요 (최소값 예: 0 또는 0x0, 최대값 예: {max_u64} 또는 0x{max_u64:X}): ",
                max_u64 = u64::MAX
            ),
            &mut self.input_buffer,
            out,
            err,
        )?;
        let mut supp_count = 0_usize;
        let mut next_supp = |reason: &'static str| -> Result<u64> {
            supp_count = supp_count.strict_add(1);
            read_u64_hex_input(
                format_args!(
                    "supp 값 #{supp_count} 입력 ({reason}, 최소값 예: 0 또는 0x0, 최대값 예: {max_u64} 또는 0x{max_u64:X}): ",
                    max_u64 = u64::MAX
                ),
                &mut self.input_buffer,
                out,
                err,
            )
        };
        let mut boundary_cache = BoundaryCache::default();
        let data = RandomDataSet {
            num_64: manual_num_64,
            ..Default::default()
        }
        .populate(&mut next_supp, &mut boundary_cache)?;
        self.output_file.persist_and_print(&data)
    }
    #[cfg(target_arch = "x86_64")]
    fn handle_random_number_command(
        &mut self,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<()> {
        writeln!(out, "\n무작위 숫자 생성 타입 선택:")?;
        let sel = read_line_reuse_limited(
            format_args!("1: 정수 생성, 2: 실수 생성, 기타: 취소\n선택해 주세요: "),
            &mut self.input_buffer,
            out,
            MENU_SELECTION_INPUT_MAX_BYTES,
        )?;
        match sel.as_bytes() {
            b"1" => {
                writeln!(
                    out,
                    "\n무작위 정수 생성기(지원 범위: -9223372036854775807 ~ 9223372036854775807)"
                )?;
                let min = self.prompt_line(
                    format_args!("최솟값을 입력해 주세요 ({MIN_ALLOWED_INTEGER_VALUE} 이상): "),
                    out,
                    Some(err),
                    |line| match line.parse::<i64>().ok() {
                        Some(val) if val >= MIN_ALLOWED_INTEGER_VALUE => Ok(val),
                        Some(_) => Err(Cow::Owned(format!(
                            "{MIN_ALLOWED_INTEGER_VALUE} 이상의 값을 입력해 주세요."
                        ))),
                        None => Err(Cow::Borrowed("유효한 정수 형식이 아닙니다.")),
                    },
                )?;
                let max = self.prompt_line(
                    format_args!("최댓값을 입력해 주세요: "),
                    out,
                    Some(err),
                    |line| match line.parse::<i64>().ok() {
                        Some(val) if val >= min => Ok(val),
                        Some(_) => Err("최댓값은 최솟값보다 크거나 같아야 합니다."),
                        None => Err("유효한 정수 형식이 아닙니다."),
                    },
                )?;
                generate_random_integer(min, max, out, &self.rng)?;
            }
            b"2" => {
                writeln!(out, "\n무작위 실수 생성기")?;
                let min = self.prompt_line(
                    format_args!("최솟값을 입력해 주세요: "),
                    out,
                    Some(err),
                    |line| parse_regular_f64(line).ok_or(FLOAT_INPUT_ERROR),
                )?;
                let max = self.prompt_line(
                    format_args!("최댓값을 입력해 주세요: "),
                    out,
                    Some(err),
                    |line| match parse_regular_f64(line) {
                        Some(val) if val >= min => Ok(val),
                        Some(_) => Err("최댓값은 최솟값보다 크거나 같아야 합니다."),
                        None => Err(FLOAT_INPUT_ERROR),
                    },
                )?;
                (max - min)
                    .is_finite()
                    .ok_or("실수 범위가 너무 커서 안전하게 계산할 수 없습니다.")?;
                generate_random_float(min, max, out, &self.rng)?;
            }
            _ => writeln!(out, "무작위 숫자 생성을 취소합니다.")?,
        }
        Ok(())
    }
    fn handle_server_time_command(
        &mut self,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<()> {
        let host = self.prompt_line(
            format_args!(
                "확인할 서버 주소를 입력하세요 (스킴 생략 시 HTTPS, 평문 HTTP는 http:// 명시 / 예: www.example.com): "
            ),
            out,
            None,
            |raw| -> CoreResult<ParsedServer, Cow<'static, str>> {
                (!raw.is_empty()).ok_or("서버 주소를 비워둘 수 없습니다.")?;
                raw.parse::<ParsedServer>()
                    .map_err(|src| format!("서버 주소가 올바르지 않습니다: {src}").into())
            },
        )?;
        let target_time = self.prompt_line(
            format_args!("액션 실행 목표 시간을 입력하세요 (예: 20:00:00 / 건너뛰려면 Enter): "),
            out,
            None,
            |raw| {
                if raw.is_empty() {
                    Ok(None)
                } else {
                    raw.parse::<TargetTimeOfDay>().map(Some)
                }
            },
        )?;
        let scheduled_trigger = match target_time {
            Some(target) => Some((
                target,
                self.prompt_line(
                    format_args!("수행할 동작을 선택하세요 (1: 마우스 왼쪽 클릭, 2: F5 입력): "),
                    out,
                    None,
                    |sel| match sel.as_bytes() {
                        b"1" => Ok(TriggerAction::LeftClick),
                        b"2" => Ok(TriggerAction::F5Press),
                        _ => Err("잘못된 입력입니다. 1 또는 2를 입력해주세요."),
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
        .run_loop(out, err)
        .map_err(Into::into)
    }
    fn prompt_line<'writer, T, E>(
        &mut self,
        prompt: Arguments<'_>,
        out: &'writer mut dyn Write,
        err: Option<&'writer mut dyn Write>,
        validator: impl FnMut(&str) -> CoreResult<T, E>,
    ) -> io::Result<T>
    where
        E: AsRef<str>,
    {
        get_validated_input(
            prompt,
            &mut self.input_buffer,
            out,
            err,
            DEFAULT_INPUT_LINE_MAX_BYTES,
            validator,
        )
    }
    pub(super) fn run(&mut self) -> Result<()> {
        let mut out = stdout().lock();
        let mut err = stderr();
        loop {
            let command = match read_line_reuse_limited(
                format_args!("{MENU}"),
                &mut self.input_buffer,
                &mut out,
                MENU_SELECTION_INPUT_MAX_BYTES,
            ) {
                Ok(text) if let &[cmd @ b'1'..=b'7'] = text.as_bytes() => cmd,
                Ok(_) => 0,
                Err(read_err) if read_err.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(read_err) => return Err(read_err.into()),
            };
            match self.execute_command(command, &mut out, &mut err) {
                Ok(true) => {}
                Ok(false) => return Ok(()),
                Err(command_err) => {
                    let root: &(dyn Error + 'static) = &command_err;
                    if successors(Some(root), |error| (*error).source()).any(|error| {
                        error
                            .downcast_ref::<io::Error>()
                            .is_some_and(|io_err| io_err.kind() == io::ErrorKind::UnexpectedEof)
                    }) {
                        return Ok(());
                    }
                    return Err(command_err);
                }
            }
        }
    }
}
