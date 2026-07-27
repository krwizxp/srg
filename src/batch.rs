use crate::{
    diagnostic::{AppError, Result},
    file_output::OutputFile,
    hardware_rng::HardwareRng,
    output::{self, OutputTarget},
    random_data::generate_random_data_with_rng,
    random_output::{persist_and_print_random_data, write_random_data_to_console},
    BUFFER_SIZE, FILE_NAME, IS_TERMINAL,
};
use core::{
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::Duration,
};
use std::{
    io::{Write, stdout},
    sync::Mutex,
    thread::{self, available_parallelism, scope},
    time::Instant,
};
pub(super) const MAX_BATCH_GENERATE_COUNT: usize = 10_000_000;
const PROGRESS_UPDATE_INTERVAL: Duration = Duration::from_millis(100);
struct WorkerFailure {
    count: usize,
    first_error: AppError,
}
fn record_failure(outcome: &mut Option<WorkerFailure>, count: usize, first_error: AppError) {
    if let Some(failure) = outcome.as_mut() {
        failure.count = failure.count.strict_add(count);
    } else {
        *outcome = Some(WorkerFailure { count, first_error });
    }
}
pub(super) fn regenerate_with_count(
    output_file: &mut OutputFile,
    rng: &HardwareRng,
    requested_count: usize,
    out: &mut dyn Write,
) -> Result<u64> {
    if requested_count > MAX_BATCH_GENERATE_COUNT {
        return Err(AppError::message(format!(
            "대량 생성 개수는 최대 {MAX_BATCH_GENERATE_COUNT}건까지 입력할 수 있습니다."
        )));
    }
    let start_time = Instant::now();
    if requested_count == 0 {
        return Err("생성 개수는 1 이상이어야 합니다.".into());
    }
    if requested_count == 1 {
        let final_data = generate_random_data_with_rng(rng)?;
        persist_and_print_random_data(output_file, &final_data)?;
        return Ok(final_data.num_64);
    }
    let pending_count = requested_count.strict_sub(1);
    let thread_count = pending_count.min(available_parallelism()?.get());
    let cancelled = AtomicBool::new(false);
    let processed = AtomicUsize::new(0);
    let base_count = pending_count.div_euclid(thread_count);
    let remainder = pending_count.rem_euclid(thread_count);
    let worker_outcome = {
        let writer_lock = Mutex::new(output_file.writer());
        let coordinator = thread::current();
        scope(|scope_ctx| -> Result<Option<WorkerFailure>> {
            let mut worker_handles = Vec::new();
            worker_handles.try_reserve_exact(thread_count).map_err(|source| {
                AppError::context("작업 스레드 handle 목록 메모리 확보 실패", source)
            })?;
            for worker_idx in 0..thread_count {
                let loop_count = base_count.strict_add(usize::from(worker_idx < remainder));
                let (writer_lock_ref, cancelled_ref, processed_ref) =
                    (&writer_lock, &cancelled, &processed);
                let coordinator_ref = &coordinator;
                worker_handles.push(scope_ctx.spawn(move || {
                    let result = (|| {
                        let mut buffer = [0_u8; BUFFER_SIZE];
                        let mut outcome = None;
                        for _ in 0..loop_count {
                            if cancelled_ref.load(Ordering::Relaxed) {
                                break;
                            }
                            let data = match generate_random_data_with_rng(rng) {
                                Ok(data) => data,
                                Err(source) => {
                                    record_failure(
                                        &mut outcome,
                                        1,
                                        AppError::context("난수 생성 실패", source),
                                    );
                                    processed_ref.fetch_add(1, Ordering::Relaxed);
                                    continue;
                                }
                            };
                            let len = match output::format_data_into_buffer(
                                &data,
                                &mut buffer,
                                OutputTarget::File,
                            ) {
                                Ok(len) => len,
                                Err(source) => {
                                    record_failure(
                                        &mut outcome,
                                        1,
                                        AppError::context("난수 데이터 포맷 실패", source),
                                    );
                                    processed_ref.fetch_add(1, Ordering::Relaxed);
                                    continue;
                                }
                            };
                            let bytes = output::prefix_slice(&buffer, len)?;
                            {
                                let mut writer = writer_lock_ref.lock().map_err(|_poison| {
                                    AppError::message("output writer lock 손상")
                                })?;
                                if cancelled_ref.load(Ordering::Relaxed) {
                                    break;
                                }
                                writer.write_all(bytes).inspect_err(|_| {
                                    cancelled_ref.store(true, Ordering::Relaxed);
                                })?;
                            }
                            processed_ref.fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(outcome)
                    })()
                    .inspect_err(|_| cancelled_ref.store(true, Ordering::Relaxed));
                    coordinator_ref.unpark();
                    result
                }));
            }
            let mut progress_buffers = output::progress::ProgressBuffers::new();
            let mut progress_out = (*IS_TERMINAL).then(|| stdout().lock());
            let mut progress_error = None;
            let mut last_progress = Instant::now();
            while worker_handles.iter().any(|handle| !handle.is_finished()) {
                if cancelled.load(Ordering::Relaxed) {
                    break;
                }
                thread::park_timeout(PROGRESS_UPDATE_INTERVAL);
                if last_progress.elapsed() >= PROGRESS_UPDATE_INTERVAL
                    && let Some(progress_writer) = progress_out.as_mut()
                {
                    if let Err(error) = progress_buffers.print(
                        progress_writer,
                        processed.load(Ordering::Relaxed),
                        requested_count,
                        start_time.elapsed(),
                    ) {
                        cancelled.store(true, Ordering::Relaxed);
                        progress_error = Some(error);
                        break;
                    }
                    last_progress = Instant::now();
                }
            }
            let mut combined_outcome = None;
            let mut first_join_error = None;
            let mut first_worker_error = None;
            for handle in worker_handles {
                match handle.join() {
                    Ok(Ok(Some(failure))) => record_failure(
                        &mut combined_outcome,
                        failure.count,
                        failure.first_error,
                    ),
                    Ok(Err(error)) if first_worker_error.is_none() => {
                        first_worker_error = Some(error);
                    }
                    Err(panic_payload) if first_join_error.is_none() => {
                        let panic_detail = panic_payload
                            .downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                            .unwrap_or("non-string thread payload");
                        first_join_error = Some(AppError::message(format!(
                            "작업 스레드 패닉 발생: {panic_detail}"
                        )));
                    }
                    Ok(Ok(None) | Err(_)) | Err(_) => {}
                }
            }
            let flush_result = writer_lock
                .lock()
                .map_err(|_poison| AppError::message("output writer lock 손상"))
                .and_then(|mut writer| writer.flush().map_err(Into::into));
            if let Some(error) = first_join_error
                .or(progress_error)
                .or(first_worker_error)
            {
                return Err(error);
            }
            flush_result?;
            Ok(combined_outcome)
        })?
    };
    if let Some(WorkerFailure {
        count: failed_count,
        first_error,
    }) = worker_outcome
    {
        return Err(AppError::context(
            format!(
                "대량 생성 중 {failed_count}건이 실패했습니다. 성공한 부분 결과만 {FILE_NAME}에 기록되었습니다."
            ),
            first_error,
        ));
    }
    let final_data = generate_random_data_with_rng(rng)?;
    let mut final_buffer = [0_u8; BUFFER_SIZE];
    let final_len = output::format_data_into_buffer(
        &final_data,
        &mut final_buffer,
        OutputTarget::File,
    )?;
    let final_file = output_file.writer();
    final_file.write_all(output::prefix_slice(&final_buffer, final_len)?)?;
    final_file.flush()?;
    let mut progress_buffers = output::progress::ProgressBuffers::new();
    progress_buffers.print(
        out,
        requested_count,
        requested_count,
        start_time.elapsed(),
    )?;
    writeln!(out, "\n총 {requested_count}건 생성 완료 ({FILE_NAME} 에 추가).\n")?;
    out.flush()?;
    write_random_data_to_console(&final_data, &mut final_buffer, final_len)?;
    Ok(final_data.num_64)
}
