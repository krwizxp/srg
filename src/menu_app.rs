use crate::{
    constants::{BUFFER_SIZE, FILE_NAME, IS_TERMINAL},
    diagnostic::{AppError, Result},
    file_output::{lock_mutex, open_or_create_file, validate_safe_output_file_path},
    input::{get_validated_input, read_line_reuse, read_u64_hex_input},
    output::{OutputTarget, format_data_into_buffer, prefix_slice},
    random_data::build_random_data,
    random_util::checked_add_one_usize,
    time,
    time::{ParsedServer, ServerTimeSession, TimeError, TriggerAction},
};
cfg_select! {
    target_arch = "x86_64" => {
        use crate::{
            batch::regenerate_with_count,
            hardware_rng::{
                HardwareRandomSource, get_hardware_random, hardware_random_source,
                take_rdseed_fallback_notice,
            },
            input::{LadderEntryMode, read_ladder_entries},
            output::write_slice_to_console,
            random_number::{generate_random_number, random_bounded},
        };
        use core::{array::from_fn, num::NonZeroU64};
        use crate::random_number::RandomNumberMode;
    }
    _ => {}
}
use alloc::borrow::Cow;
use core::{result::Result as CoreResult, time::Duration};
use std::{
    fs::{self, File},
    io::{BufWriter, ErrorKind, Write, stderr, stdout},
    path::Path,
    process::ExitCode,
    sync::Mutex,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
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
const KST_SECS_PER_HOUR_U64: u64 = 3_600;
const KST_SECS_PER_MINUTE_U64: u64 = 60;
const KST_SECS_PER_DAY_U64: u64 = 24 * KST_SECS_PER_HOUR_U64;
cfg_select! {
    target_arch = "x86_64" => {
        pub struct MenuApp {
            pub file_mutex: Mutex<BufWriter<File>>,
            pub input_buffer: String,
            pub ladder_players_storage: String,
            pub ladder_results_storage: String,
            pub num_64: u64,
        }
    }
    _ => {
        pub struct MenuApp {
            pub file_mutex: Mutex<BufWriter<File>>,
            pub input_buffer: String,
        }
    }
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
                validate_safe_output_file_path(Path::new(FILE_NAME))?;
                if let Err(remove_err) = fs::remove_file(FILE_NAME) {
                    writeln!(err, "{remove_err}")?;
                } else {
                    writeln!(out, "파일 '{FILE_NAME}'를 삭제했습니다.")?;
                }
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
                    b'3' => self.handle_generate_once_command(out, err)?,
                    b'4' => self.handle_generate_many_command(out, err)?,
                    _ => return Ok(false),
                }
                Ok(true)
            }
            _ => {
                match command {
                    b'1' | b'2' | b'3' | b'4' => {
                        writeln!(
                            out,
                            "이 기능은 x86_64 전용이라 현재 플랫폼에서는 비활성화되어 있습니다."
                        )?;
                    }
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
        if !prepare_hw_rng_menu_command(out)? {
            return Ok(());
        }
        let input_buffer = &mut self.input_buffer;
        let count_prompt = format_args!("\n생성할 데이터 개수를 입력해 주세요: ");
        let requested_count = loop {
            match read_line_reuse(count_prompt, input_buffer, out)?.parse::<u64>() {
                Ok(0) => writeln!(err, "1 이상의 값을 입력해 주세요.")?,
                Ok(count) => break count,
                Err(_) => writeln!(err, "유효한 숫자를 입력해 주세요.")?,
            }
        };
        let next_num_64 = regenerate_with_count(&self.file_mutex, requested_count, out, err)?;
        write_rdseed_fallback_notice(err)?;
        self.num_64 = next_num_64;
        Ok(())
    }
    #[cfg(target_arch = "x86_64")]
    fn handle_generate_once_command(
        &mut self,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<()> {
        if !prepare_hw_rng_menu_command(out)? {
            return Ok(());
        }
        let next_num_64 = regenerate_with_count(&self.file_mutex, 1, out, err)?;
        write_rdseed_fallback_notice(err)?;
        self.num_64 = next_num_64;
        Ok(())
    }
    #[cfg(target_arch = "x86_64")]
    fn handle_ladder_command(&mut self, out: &mut dyn Write, err: &mut dyn Write) -> Result<()> {
        const MAX_PLAYERS: usize = 512;
        if !prepare_hw_rng_menu_command(out)? {
            return Ok(());
        }
        let mut seed = self.num_64;
        let input_buffer = &mut self.input_buffer;
        let players_storage = &mut self.ladder_players_storage;
        let results_storage = &mut self.ladder_results_storage;
        players_storage.clear();
        let mut players_array: [&str; MAX_PLAYERS] = [""; MAX_PLAYERS];
        let n = read_ladder_entries(
            format_args!("\n사다리타기 플레이어를 입력해 주세요 (쉼표(,)로 구분, 2~512명): "),
            input_buffer,
            (&mut *out, &mut *err),
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
            (&mut *out, &mut *err),
            results_storage,
            &mut results_array,
            LadderEntryMode::Results { expected_count: n },
            "결과 배열 인덱스 범위 초과",
        )?;
        writeln!(out, "사다리타기 결과:")?;
        let mut indices: [usize; MAX_PLAYERS] = from_fn(|index| index);
        let indices_slice = indices
            .get_mut(..n)
            .ok_or("인덱스 배열 슬라이스 범위 초과")?;
        for index in (1..indices_slice.len()).rev() {
            seed ^= get_hardware_random()?;
            let next_index = checked_add_one_usize(index).ok_or("인덱스 상한 계산 실패")?;
            let upper_bound_raw = u64::try_from(next_index).map_err(|conversion_err| {
                AppError::context("인덱스 상한 변환 실패", conversion_err)
            })?;
            let upper_bound = NonZeroU64::new(upper_bound_raw).ok_or("인덱스 상한이 0입니다.")?;
            let swap_index_u64 = random_bounded(upper_bound, seed)?;
            let swap_index = usize::try_from(swap_index_u64)
                .map_err(|conversion_err| AppError::context("인덱스 변환 실패", conversion_err))?;
            indices_slice.swap(index, swap_index);
        }
        let players = players_array
            .get(..n)
            .ok_or("플레이어 슬라이스 범위 초과")?;
        for (player, &result_index) in players.iter().zip(indices_slice.iter()) {
            let result = results_array
                .get(result_index)
                .copied()
                .ok_or("결과 인덱스 범위 초과")?;
            writeln!(out, "{player} -> {result}")?;
        }
        write_rdseed_fallback_notice(err)
    }
    fn handle_manual_input_command(
        &mut self,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<()> {
        if !Path::new(FILE_NAME).try_exists()? {
            *lock_mutex(&self.file_mutex, "Mutex 잠금 실패 (파일 생성 시)")? =
                open_or_create_file()?;
        }
        writeln!(out, "\nnum_64/supp 수동 입력 생성 모드")?;
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
            supp_input_count =
                checked_add_one_usize(supp_input_count).ok_or("supp 입력 횟수 계산 실패")?;
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
        let data = build_random_data(manual_num_64, &mut next_supp)?;
        let mut buffer = [0_u8; BUFFER_SIZE];
        let file_len = format_data_into_buffer(&data, &mut buffer, OutputTarget::File)?;
        {
            let mut file_guard = lock_mutex(&self.file_mutex, "Mutex 잠금 실패 (단일 쓰기 시)")?;
            Write::write_all(&mut *file_guard, prefix_slice(&buffer, file_len)?)?;
            Write::flush(&mut *file_guard)?;
        }
        let console_len = if *IS_TERMINAL {
            format_data_into_buffer(&data, &mut buffer, OutputTarget::Console)?
        } else {
            file_len
        };
        let console_slice = prefix_slice(&buffer, console_len)?;
        cfg_select! {
            target_arch = "x86_64" => {
                write_slice_to_console(console_slice)?;
            }
            _ => {{
                let mut stdout_lock = stdout().lock();
                Write::write_all(&mut stdout_lock, console_slice)?;
                Write::flush(&mut stdout_lock)?;
            }}
        }
        write_rdseed_fallback_notice(err)
    }
    #[cfg(target_arch = "x86_64")]
    fn handle_random_number_command(
        &mut self,
        out: &mut dyn Write,
        err: &mut dyn Write,
    ) -> Result<()> {
        if !prepare_hw_rng_menu_command(out)? {
            return Ok(());
        }
        let num_64 = self.num_64;
        let input_buffer = &mut self.input_buffer;
        writeln!(out, "\n무작위 숫자 생성 타입 선택:")?;
        let selection = read_line_reuse(
            format_args!("1: 정수 생성, 2: 실수 생성, 기타: 취소\n선택해 주세요: "),
            input_buffer,
            out,
        )?;
        match selection.as_bytes() {
            b"1" => {
                generate_random_number(RandomNumberMode::Integer, num_64, input_buffer, out, err)?;
            }
            b"2" => {
                generate_random_number(RandomNumberMode::Float, num_64, input_buffer, out, err)?;
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
        let time_run_result = (|| -> CoreResult<(), TimeError> {
            let host = self.read_server_host(out)?;
            let target_time = self.read_target_time(out)?;
            let trigger_action = self.read_trigger_action(out, target_time)?;
            let now = Instant::now();
            ServerTimeSession {
                host,
                now,
                target_time,
                trigger_action,
            }
            .run_loop(out, err)?;
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
    fn read_server_host(&mut self, out: &mut dyn Write) -> CoreResult<ParsedServer, TimeError> {
        Ok(get_validated_input(
            "확인할 서버 주소를 입력하세요 (예: www.example.com): ",
            &mut self.input_buffer,
            out,
            |raw_input| -> CoreResult<ParsedServer, Cow<'static, str>> {
                if raw_input.is_empty() {
                    return Err(Cow::Borrowed("서버 주소를 비워둘 수 없습니다."));
                }
                raw_input.parse::<ParsedServer>().map_err(|source| {
                    Cow::Owned(format!("서버 주소가 올바르지 않습니다: {source}"))
                })
            },
        )?)
    }
    fn read_target_time(
        &mut self,
        out: &mut dyn Write,
    ) -> CoreResult<Option<SystemTime>, TimeError> {
        const INVALID_TIME_INPUT_ERR: &str =
            "잘못된 형식, 숫자 또는 시간 범위입니다 (HH:MM:SS, 0-23:0-59:0-59).";
        Ok(get_validated_input(
            "액션 실행 목표 시간을 입력하세요 (예: 20:00:00 / 건너뛰려면 Enter): ",
            &mut self.input_buffer,
            out,
            |raw_input| -> CoreResult<Option<SystemTime>, &'static str> {
                if raw_input.is_empty() {
                    return Ok(None);
                }
                let Some((hour_str, minute_second)) = raw_input.split_once(':') else {
                    return Err(INVALID_TIME_INPUT_ERR);
                };
                let Some((minute_str, second_str)) = minute_second.split_once(':') else {
                    return Err(INVALID_TIME_INPUT_ERR);
                };
                if second_str.contains(':') {
                    return Err(INVALID_TIME_INPUT_ERR);
                }
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
                    let Some(next_day_time) = target_time.checked_add(Duration::from_hours(24))
                    else {
                        return Err("시간 계산 오류: 다음날 목표 시각 계산 실패");
                    };
                    target_time = next_day_time;
                }
                Ok(Some(target_time))
            },
        )?)
    }
    fn read_trigger_action(
        &mut self,
        out: &mut dyn Write,
        target_time: Option<SystemTime>,
    ) -> CoreResult<Option<TriggerAction>, TimeError> {
        if target_time.is_none() {
            return Ok(None);
        }
        let action = get_validated_input(
            "수행할 동작을 선택하세요 (1: 마우스 왼쪽 클릭, 2: F5 입력): ",
            &mut self.input_buffer,
            out,
            |selection| -> CoreResult<TriggerAction, &'static str> {
                match selection.as_bytes() {
                    b"1" => Ok(TriggerAction::LeftClick),
                    b"2" => Ok(TriggerAction::F5Press),
                    _ => Err("잘못된 입력입니다. 1 또는 2를 입력해주세요."),
                }
            },
        )?;
        Ok(Some(action))
    }
    pub(super) fn run(&mut self) -> Result<ExitCode> {
        let menu_prompt = format_args!("{MENU}");
        loop {
            let command = {
                let mut prompt_out = stdout().lock();
                match read_line_reuse(menu_prompt, &mut self.input_buffer, &mut prompt_out) {
                    Ok(command_str) if let &[command @ b'1'..=b'7'] = command_str.as_bytes() => {
                        command
                    }
                    Ok(_) => 0,
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
                    if command_err.io_error_kind() == Some(ErrorKind::UnexpectedEof) =>
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
#[cfg(target_arch = "x86_64")]
fn prepare_hw_rng_menu_command(out: &mut dyn Write) -> Result<bool> {
    match hardware_random_source() {
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
#[cfg(target_arch = "x86_64")]
fn write_rdseed_fallback_notice(err: &mut dyn Write) -> Result<()> {
    if take_rdseed_fallback_notice() {
        writeln!(err, "RDSEED 10분 타임아웃으로 RDRAND로 전환했습니다.")?;
    }
    Ok(())
}
