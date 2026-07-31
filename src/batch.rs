use crate::{
    diagnostic::{AppError, Result},
    file_output::OutputFile,
    hardware_rng::HardwareRng,
    output::{
        self, OutputTarget, format_data_into_buffer, prefix_slice, write_slice_to_console,
    },
    random_data::generate_random_data_with_rng,
    random_output::{
        persist_and_print_random_data, persist_random_data, write_random_data_to_console,
    },
    stop_input::StopInput,
    BUFFER_SIZE, FILE_NAME, IS_TERMINAL,
};
use core::{
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::Duration,
};
use std::{
    fs::File,
    io::{IsTerminal as _, Write, stdin},
    sync::Mutex,
    thread::{self, available_parallelism, scope},
    time::Instant,
};
pub(super) const MAX_BATCH_GENERATE_COUNT: usize = 10_000_000;
const PROGRESS_UPDATE_INTERVAL: Duration = Duration::from_millis(100);
const WORKER_CHUNK_CAPACITY: usize = 0x0020_0000_usize.strict_add(BUFFER_SIZE);
struct WorkerFailure {
    count: usize,
    first_error: AppError,
}
struct RecordedData {
    len: usize,
    num_64: u64,
}
#[derive(Default)]
struct WorkerChunk {
    bytes: Vec<u8>,
    last: Option<RecordedData>,
    record_count: usize,
    written_len: usize,
}
struct WorkerOutput<'writer> {
    last: Option<RecordedData>,
    writer: &'writer mut File,
}
fn record_failure(outcome: &mut Option<WorkerFailure>, count: usize, first_error: AppError) {
    if let Some(failure) = outcome.as_mut() {
        failure.count = failure.count.strict_add(count);
    } else {
        *outcome = Some(WorkerFailure { count, first_error });
    }
}
fn write_worker_chunk(
    writer_lock: &Mutex<WorkerOutput<'_>>,
    cancelled: &AtomicBool,
    processed: &AtomicUsize,
    chunk: &mut WorkerChunk,
) -> Result<()> {
    if chunk.written_len == 0 || cancelled.load(Ordering::Relaxed) {
        return Ok(());
    }
    let mut output = writer_lock
        .lock()
        .map_err(|_poison| AppError::message("output writer lock 손상"))?;
    if cancelled.load(Ordering::Relaxed) {
        return Ok(());
    }
    output
        .writer
        .write_all(
            chunk
                .bytes
                .get(..chunk.written_len)
                .ok_or_else(|| AppError::message("작업자 출력 범위 손상"))?,
        )
        .inspect_err(|_| cancelled.store(true, Ordering::Relaxed))?;
    output.last = chunk.last.take();
    drop(output);
    processed.fetch_add(chunk.record_count, Ordering::Relaxed);
    chunk.record_count = 0;
    chunk.written_len = 0;
    Ok(())
}
fn print_final_progress(
    out: &mut dyn Write,
    completed: usize,
    requested: usize,
    elapsed: Duration,
) -> Result<()> {
    if !*IS_TERMINAL {
        return Ok(());
    }
    let mut buffers = output::progress::ProgressBuffers::new();
    buffers.print(out, completed, requested, elapsed)
}
pub(super) fn regenerate_with_count(
    output_file: &mut OutputFile,
    rng: &HardwareRng,
    requested_count: usize,
    cancel_on_enter: bool,
    out: &mut dyn Write,
) -> Result<Option<u64>> {
    if requested_count > MAX_BATCH_GENERATE_COUNT {
        return Err(format!("최대 {MAX_BATCH_GENERATE_COUNT}건까지 생성할 수 있습니다.").into());
    }
    if requested_count == 0 {
        return Err(AppError::message("생성 개수는 1 이상이어야 합니다."));
    }
    if requested_count == 1 {
        let final_data = generate_random_data_with_rng(rng)?;
        persist_and_print_random_data(output_file, &final_data)?;
        return Ok(Some(final_data.num_64));
    }
    let start_time = Instant::now();
    let mut stop_input = if cancel_on_enter && stdin().is_terminal() {
        writeln!(out, "\n생성 중 Enter를 누르면 작업을 중단합니다.")?;
        out.flush()?;
        Some(StopInput::try_from(None)?)
    } else {
        None
    };
    let pending_count = requested_count.strict_sub(1);
    let thread_count = pending_count.min(available_parallelism()?.get());
    let (cancelled, processed) = (AtomicBool::new(false), AtomicUsize::new(0));
    let base_count = pending_count.div_euclid(thread_count);
    let remainder = pending_count.rem_euclid(thread_count);
    let (worker_outcome, user_cancelled, last_recorded_data) = {
        let writer_lock = Mutex::new(WorkerOutput {
            last: None,
            writer: output_file.writer(),
        });
        let coordinator = thread::current();
        scope(|scope_ctx| -> Result<(Option<WorkerFailure>, bool, Option<RecordedData>)> {
            let mut worker_handles = Vec::new();
            worker_handles
                .try_reserve_exact(thread_count)
                .map_err(|source| AppError::context("작업 스레드 목록 확보 실패", source))?;
            for worker_idx in 0..thread_count {
                let loop_count = base_count.strict_add(usize::from(worker_idx < remainder));
                let (writer, stop, done) = (&writer_lock, &cancelled, &processed);
                let wake = &coordinator;
                worker_handles.push(scope_ctx.spawn(move || {
                    let result = (|| {
                        let mut chunk = WorkerChunk::default();
                        let chunk_capacity =
                            WORKER_CHUNK_CAPACITY.min(loop_count.strict_mul(BUFFER_SIZE));
                        chunk.bytes.try_reserve_exact(chunk_capacity).map_err(|source| {
                            AppError::context("작업자 출력 버퍼 메모리 확보 실패", source)
                        })?;
                        chunk.bytes.resize(chunk_capacity, 0);
                        let mut outcome = None;
                        for _ in 0..loop_count {
                            if stop.load(Ordering::Relaxed) {
                                break;
                            }
                            if chunk.written_len > chunk.bytes.len().strict_sub(BUFFER_SIZE) {
                                write_worker_chunk(writer, stop, done, &mut chunk)?;
                                if stop.load(Ordering::Relaxed) {
                                    break;
                                }
                            }
                            let record = generate_random_data_with_rng(rng)
                                .map_err(|source| AppError::context("난수 생성 실패", source))
                                .and_then(|data| {
                                    let buffer = chunk
                                        .bytes
                                        .get_mut(chunk.written_len..)
                                        .and_then(|tail| tail.first_chunk_mut::<BUFFER_SIZE>())
                                        .ok_or_else(|| AppError::message("작업자 출력 버퍼 범위 손상"))?;
                                    let len =
                                        format_data_into_buffer(&data, buffer, OutputTarget::File)
                                            .map_err(|source| {
                                        AppError::context("난수 데이터 포맷 실패", source)
                                    })?;
                                    Ok((data.num_64, len))
                                });
                            let (num_64, len) = match record {
                                Ok(generated) => generated,
                                Err(error) => {
                                    record_failure(&mut outcome, 1, error);
                                    done.fetch_add(1, Ordering::Relaxed);
                                    continue;
                                }
                            };
                            chunk.written_len = chunk.written_len.strict_add(len);
                            chunk.record_count = chunk.record_count.strict_add(1);
                            chunk.last = Some(RecordedData { len, num_64 });
                        }
                        write_worker_chunk(writer, stop, done, &mut chunk)?;
                        Ok(outcome)
                    })()
                    .inspect_err(|_| stop.store(true, Ordering::Relaxed));
                    wake.unpark();
                    result
                }));
            }
            let mut progress_buffers = output::progress::ProgressBuffers::new();
            let mut progress_error = None;
            let mut last_progress = start_time;
            let mut user_cancelled = false;
            while !cancelled.load(Ordering::Relaxed)
                && worker_handles.iter().any(|handle| !handle.is_finished())
            {
                if let Some(input) = stop_input.as_mut() {
                    match input.poll() {
                        Ok(false) => {}
                        Ok(true) => {
                            cancelled.store(true, Ordering::Relaxed);
                            user_cancelled = true;
                            break;
                        }
                        Err(error) => {
                            cancelled.store(true, Ordering::Relaxed);
                            progress_error = Some(error.into());
                            break;
                        }
                    }
                }
                thread::park_timeout(PROGRESS_UPDATE_INTERVAL);
                if *IS_TERMINAL
                    && last_progress.elapsed() >= PROGRESS_UPDATE_INTERVAL
                {
                    if let Err(error) = progress_buffers.print(
                        out, processed.load(Ordering::Relaxed), requested_count,
                        start_time.elapsed()
                    ) {
                        cancelled.store(true, Ordering::Relaxed);
                        progress_error = Some(error);
                        break;
                    }
                    last_progress = Instant::now();
                }
            }
            let mut combined = None;
            let mut join_error = None;
            let mut worker_error = None;
            for handle in worker_handles {
                match handle.join() {
                    Ok(Ok(Some(failure))) => {
                        record_failure(&mut combined, failure.count, failure.first_error);
                    }
                    Ok(Err(error)) if worker_error.is_none() => {
                        worker_error = Some(error);
                    }
                    Err(panic_payload) if join_error.is_none() => {
                        let panic_detail = panic_payload
                            .downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                            .unwrap_or("non-string thread payload");
                        join_error = Some(AppError::message(format!(
                            "작업 스레드 패닉 발생: {panic_detail}"
                        )));
                    }
                    Ok(Ok(None) | Err(_)) | Err(_) => {}
                }
            }
            if let Some(error) = join_error.or(progress_error).or(worker_error) {
                return Err(error);
            }
            let last_recorded_data = writer_lock
                .lock()
                .map_err(|_poison| AppError::message("output writer lock 손상"))?
                .last
                .take();
            Ok((combined, user_cancelled, last_recorded_data))
        })?
    };
    if let Some(failure) = worker_outcome {
        return Err(AppError::context(
            format!(
                "대량 생성 중 {}건이 실패했습니다. 성공한 부분 결과만 {FILE_NAME}에 기록되었습니다.",
                failure.count
            ),
            failure.first_error,
        ));
    }
    if user_cancelled {
        let completed_count = processed.load(Ordering::Relaxed);
        print_final_progress(out, completed_count, requested_count, start_time.elapsed())?;
        writeln!(
            out,
            "\nEnter 입력으로 작업을 중단했습니다. 총 {completed_count}건이 {FILE_NAME}에 기록되었습니다.\n"
        )?;
        out.flush()?;
        let Some(last_record) = last_recorded_data else {
            return Ok(None);
        };
        let mut buffer = [0_u8; BUFFER_SIZE];
        output_file.read_tail_into(last_record.len, &mut buffer)?;
        write_slice_to_console(prefix_slice(&buffer, last_record.len)?)?;
        return Ok(Some(last_record.num_64));
    }
    let final_data = generate_random_data_with_rng(rng)?;
    let mut final_buffer = [0_u8; BUFFER_SIZE];
    let final_len = persist_random_data(output_file, &final_data, &mut final_buffer)?;
    print_final_progress(out, requested_count, requested_count, start_time.elapsed())?;
    writeln!(out, "\n총 {requested_count}건 생성 완료 ({FILE_NAME} 에 추가).\n")?;
    out.flush()?;
    write_random_data_to_console(&final_data, &mut final_buffer, final_len)?;
    Ok(Some(final_data.num_64))
}
