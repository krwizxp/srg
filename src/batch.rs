use crate::{
    BUFFER_SIZE, FILE_NAME, IS_TERMINAL,
    diagnostic::{AppError, Result, or_abort},
    file_output::OutputFile,
    hardware_rng::HardwareRng,
    korea_boundary::BoundaryCache,
    output::{OutputTarget, ProgressBuffers, format_data_into_buffer, write_slice_to_console},
    random_data::generate_random_data_with_rng,
    stop_input::StopInput,
};
use core::{
    mem::DropGuard,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::Duration,
};
use std::{
    fs::File,
    io::{IsTerminal as _, Write, stdin},
    panic::resume_unwind,
    process,
    sync::Mutex,
    thread::{self, available_parallelism, scope},
    time::Instant,
};
pub(super) const MAX_BATCH_GENERATE_COUNT: usize = 10_000_000;
const PROGRESS_UPDATE_INTERVAL: Duration = Duration::from_millis(100);
const WORKER_CHUNK_CAPACITY: usize = 0x0002_0000_usize.strict_add(BUFFER_SIZE);
#[derive(Default)]
struct WorkerChunk {
    bytes: Vec<u8>,
    last_len: usize,
    record_count: usize,
    written_len: usize,
}
struct WorkerFailure {
    count: usize,
    first_error: AppError,
}
struct WorkerOutput<'writer> {
    last_len: Option<usize>,
    last_record: &'writer mut [u8; BUFFER_SIZE],
    writer: &'writer mut File,
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
    cancel_on_enter: bool,
    out: &mut dyn Write,
) -> Result<()> {
    if requested_count == 1 {
        let mut cache = BoundaryCache::default();
        return output_file.persist_and_print(&generate_random_data_with_rng(rng, &mut cache)?);
    }
    let start_time = Instant::now();
    let mut stop_input = if cancel_on_enter && stdin().is_terminal() {
        writeln!(out, "\n생성 중 Enter를 누르면 작업을 중단합니다.")?;
        out.flush()?;
        Some(StopInput::try_from(None)?)
    } else {
        None
    };
    let threads = requested_count.min(available_parallelism()?.get());
    let (cancelled, processed) = (AtomicBool::new(false), AtomicUsize::new(0));
    let (base_cnt, rem) = (
        requested_count.div_euclid(threads),
        requested_count.rem_euclid(threads),
    );
    let (mut last_buf, mut progress) = ([0_u8; BUFFER_SIZE], ProgressBuffers::EMPTY);
    let writer_lock = Mutex::new(WorkerOutput {
        last_len: None,
        last_record: &mut last_buf,
        writer: output_file.writer(),
    });
    let coordinator = thread::current();
    let (worker_outcome, user_cancelled) =
        scope(|scope_ctx| -> Result<(Option<WorkerFailure>, bool)> {
            let mut handles = Vec::new();
            handles
                .try_reserve_exact(threads)
                .map_err(|src| AppError::context("작업 스레드 목록 확보 실패", src))?;
            let mut progress_error = None;
            for worker_idx in 0..threads {
                let loop_cnt = base_cnt.strict_add(usize::from(worker_idx < rem));
                let (writer, stop, done, wake) =
                    (&writer_lock, &cancelled, &processed, &coordinator);
                match thread::Builder::new().spawn_scoped(scope_ctx, move || {
                    let res = (|| {
                        let mut chunk = WorkerChunk::default();
                        let cap = WORKER_CHUNK_CAPACITY.min(loop_cnt.strict_mul(BUFFER_SIZE));
                        chunk.bytes.try_reserve_exact(cap).map_err(|src| {
                            AppError::context("작업자 출력 버퍼 메모리 확보 실패", src)
                        })?;
                        chunk.bytes.resize(cap, 0);
                        let (mut outcome, mut cache) = (None, BoundaryCache::default());
                        for _ in 0..loop_cnt {
                            if chunk.written_len > chunk.bytes.len().strict_sub(BUFFER_SIZE) {
                                write_worker_chunk(writer, stop, done, &mut chunk)?;
                            }
                            if stop.load(Ordering::Relaxed) {
                                break;
                            }
                            let data = match generate_random_data_with_rng(rng, &mut cache) {
                                Ok(data) => data,
                                Err(src) => {
                                    record_failure(
                                        &mut outcome,
                                        1,
                                        AppError::context("난수 생성 실패", src),
                                    );
                                    done.fetch_add(1, Ordering::Relaxed);
                                    continue;
                                }
                            };
                            let buf = or_abort(
                                chunk
                                    .bytes
                                    .get_mut(chunk.written_len..)
                                    .and_then(|tail| tail.first_chunk_mut::<BUFFER_SIZE>()),
                            );
                            let len = format_data_into_buffer(&data, buf, OutputTarget::File);
                            chunk.written_len = chunk.written_len.strict_add(len);
                            chunk.record_count = chunk.record_count.strict_add(1);
                            chunk.last_len = len;
                        }
                        write_worker_chunk(writer, stop, done, &mut chunk)?;
                        Ok(outcome)
                    })()
                    .inspect_err(|_| stop.store(true, Ordering::Relaxed));
                    wake.unpark();
                    res
                }) {
                    Ok(handle) => handles.push(handle),
                    Err(src) => {
                        cancelled.store(true, Ordering::Relaxed);
                        progress_error = Some(AppError::context("작업 스레드 생성 실패", src));
                        break;
                    }
                }
            }
            let (mut last_progress, mut user_cancelled) = (start_time, false);
            while !cancelled.load(Ordering::Relaxed)
                && handles.iter().any(|handle| !handle.is_finished())
            {
                if let Some(input) = stop_input.as_mut() {
                    match input.poll() {
                        Ok(false) => {}
                        Ok(true) => {
                            cancelled.store(true, Ordering::Relaxed);
                            user_cancelled = true;
                            break;
                        }
                        Err(err) => {
                            cancelled.store(true, Ordering::Relaxed);
                            progress_error = Some(err.into());
                            break;
                        }
                    }
                }
                thread::park_timeout(PROGRESS_UPDATE_INTERVAL);
                if *IS_TERMINAL && last_progress.elapsed() >= PROGRESS_UPDATE_INTERVAL {
                    if let Err(err) = progress.print(
                        out,
                        processed.load(Ordering::Relaxed),
                        requested_count,
                        start_time.elapsed(),
                    ) {
                        cancelled.store(true, Ordering::Relaxed);
                        progress_error = Some(err);
                        break;
                    }
                    last_progress = Instant::now();
                }
            }
            let rng_guard =
                DropGuard::new(rng, |active| active.set_pending_read_interruption(false));
            rng_guard.set_pending_read_interruption(cancelled.load(Ordering::Relaxed));
            let (mut combined, mut worker_panic, mut worker_error) = (None, None, None);
            for handle in handles {
                match handle.join() {
                    Ok(Ok(Some(failure))) => {
                        record_failure(&mut combined, failure.count, failure.first_error);
                    }
                    Ok(Err(err)) if worker_error.is_none() => worker_error = Some(err),
                    Err(payload) if worker_panic.is_none() => worker_panic = Some(payload),
                    Ok(Ok(None) | Err(_)) | Err(_) => {}
                }
            }
            drop(rng_guard);
            if let Some(payload) = worker_panic {
                resume_unwind(payload);
            }
            if let Some(err) = progress_error.or(worker_error) {
                return Err(err);
            }
            Ok((combined, user_cancelled))
        })?;
    let last_recorded_len = writer_lock
        .into_inner()
        .unwrap_or_else(|_| process::abort())
        .last_len;
    if let Some(failure) = worker_outcome {
        return Err(AppError::context(
            format!(
                "대량 생성 중 {}건이 실패했습니다. 성공한 부분 결과만 {FILE_NAME}에 기록되었습니다.",
                failure.count
            ),
            failure.first_error,
        ));
    }
    let completed = if user_cancelled {
        processed.load(Ordering::Relaxed)
    } else {
        requested_count
    };
    if *IS_TERMINAL {
        progress.print(out, completed, requested_count, start_time.elapsed())?;
    }
    if user_cancelled {
        writeln!(
            out,
            "\nEnter 입력으로 작업을 중단했습니다. 총 {completed}건이 {FILE_NAME}에 기록되었습니다.\n"
        )?;
    } else {
        writeln!(
            out,
            "\n총 {requested_count}건 생성 완료 ({FILE_NAME} 에 추가).\n"
        )?;
    }
    out.flush()?;
    if let Some(last_len) = last_recorded_len {
        write_slice_to_console(last_buf.split_at(last_len).0)?;
    }
    Ok(())
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
    let written = chunk.bytes.split_at(chunk.written_len).0;
    output
        .writer
        .write_all(written)
        .inspect_err(|_| cancelled.store(true, Ordering::Relaxed))?;
    let last_len = chunk.last_len;
    or_abort(output.last_record.get_mut(..last_len)).copy_from_slice(or_abort(
        written.get(chunk.written_len.strict_sub(last_len)..),
    ));
    output.last_len = Some(last_len);
    drop(output);
    processed.fetch_add(chunk.record_count, Ordering::Relaxed);
    chunk.record_count = 0;
    chunk.written_len = 0;
    Ok(())
}
