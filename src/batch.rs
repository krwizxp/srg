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
    num::NonZero,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};
use std::{
    io::{Write, stdout},
    sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError, sync_channel},
    thread::{available_parallelism, scope},
    time::Instant,
};
const BUFFERS_PER_WORKER: usize = 8;
pub(super) const MAX_BATCH_GENERATE_COUNT: usize = 10_000_000;
const PROGRESS_UPDATE_INTERVAL: Duration = Duration::from_millis(100);
type DataBuffer = Box<[u8; BUFFER_SIZE]>;
struct GeneratedChunk<'scope> {
    buffer: DataBuffer,
    len: usize,
    return_sender: &'scope SyncSender<DataBuffer>,
}
struct WorkerFailure {
    count: usize,
    first_error: AppError,
}
struct WorkerJob<'scope> {
    local_pool: Vec<DataBuffer>,
    loop_count: usize,
    processed_ref: &'scope AtomicUsize,
    return_rx: Receiver<DataBuffer>,
    return_sender: &'scope SyncSender<DataBuffer>,
    rng: &'scope HardwareRng,
    sender: SyncSender<GeneratedChunk<'scope>>,
}
impl WorkerJob<'_> {
    fn record_failure(
        processed_ref: &AtomicUsize,
        outcome: &mut Option<WorkerFailure>,
        local_pool: &mut Vec<DataBuffer>,
        buffer: DataBuffer,
        message: AppError,
    ) {
        record_failure(outcome, 1, message);
        processed_ref.fetch_add(1, Ordering::Relaxed);
        local_pool.push(buffer);
    }
    fn run(mut self) -> Option<WorkerFailure> {
        let rng = self.rng;
        let mut outcome = None;
        for item_index in 0..self.loop_count {
            let mut buffer = if let Some(buf) = self.local_pool.pop() {
                buf
            } else {
                let Ok(buf) = self.return_rx.recv() else {
                    let remaining = self.loop_count.wrapping_sub(item_index);
                    record_failure(
                        &mut outcome,
                        remaining,
                        AppError::message("writer buffer 반환 채널 연결 종료"),
                    );
                    self.processed_ref
                        .fetch_add(remaining, Ordering::Relaxed);
                    break;
                };
                buf
            };
            let data = match generate_random_data_with_rng(rng) {
                Ok(data) => data,
                Err(source) => {
                    Self::record_failure(
                        self.processed_ref,
                        &mut outcome,
                        &mut self.local_pool,
                        buffer,
                        AppError::context("난수 생성 실패", source),
                    );
                    continue;
                }
            };
            let len = match output::format_data_into_buffer(
                &data,
                buffer.as_mut(),
                OutputTarget::File,
            ) {
                Ok(len) => len,
                Err(source) => {
                    Self::record_failure(
                        self.processed_ref,
                        &mut outcome,
                        &mut self.local_pool,
                        buffer,
                        AppError::context("난수 데이터 포맷 실패", source),
                    );
                    continue;
                }
            };
            if self
                .sender
                .send(GeneratedChunk {
                    buffer,
                    len,
                    return_sender: self.return_sender,
                })
                .is_ok()
            {
                self.processed_ref.fetch_add(1, Ordering::Relaxed);
                while self.local_pool.len() < BUFFERS_PER_WORKER {
                    match self.return_rx.try_recv() {
                        Ok(buf) => self.local_pool.push(buf),
                        Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                    }
                }
            } else {
                let remaining = self.loop_count.wrapping_sub(item_index);
                record_failure(
                    &mut outcome,
                    remaining,
                    AppError::message("writer channel 전송 실패"),
                );
                self.processed_ref
                    .fetch_add(remaining, Ordering::Relaxed);
                break;
            }
        }
        outcome
    }
}
fn record_failure(outcome: &mut Option<WorkerFailure>, count: usize, first_error: AppError) {
    if let Some(failure) = outcome.as_mut() {
        failure.count = failure.count.wrapping_add(count);
    } else {
        *outcome = Some(WorkerFailure { count, first_error });
    }
}
fn try_reserved_vec<T>(capacity: usize, context: &'static str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|source| AppError::context(context, source))?;
    Ok(values)
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
    let max_threads = available_parallelism()?;
    let pending_count = requested_count.wrapping_sub(1);
    let calculated_thread_count = NonZero::<usize>::MIN
        .saturating_add(pending_count.wrapping_sub(1))
        .min(max_threads);
    let in_flight_buffers = calculated_thread_count
        .get()
        .wrapping_mul(BUFFERS_PER_WORKER);
    let mut return_senders = try_reserved_vec(
        calculated_thread_count.get(),
        "buffer sender 목록 메모리 확보 실패",
    )?;
    let mut worker_buffers = try_reserved_vec(
        calculated_thread_count.get(),
        "worker buffer 목록 메모리 확보 실패",
    )?;
    for _ in 0..calculated_thread_count.get() {
        let (tx, rx) = sync_channel(BUFFERS_PER_WORKER);
        let mut local_pool =
            try_reserved_vec(BUFFERS_PER_WORKER, "worker buffer pool 메모리 확보 실패")?;
        for _ in 0..BUFFERS_PER_WORKER {
            local_pool.push(Box::new([0_u8; BUFFER_SIZE]));
        }
        return_senders.push(tx);
        worker_buffers.push((rx, local_pool));
    }
    let (sender, receiver) = sync_channel::<GeneratedChunk<'_>>(in_flight_buffers);
    let processed = AtomicUsize::new(0);
    let processed_ref = &processed;
    let base_count = pending_count.div_euclid(calculated_thread_count.get());
    let remainder = pending_count.rem_euclid(calculated_thread_count.get());
    let worker_outcome = {
        let output_writer = output_file.writer();
        scope(|scope_ctx| -> Result<Option<WorkerFailure>> {
            let mut worker_handles = try_reserved_vec(
                calculated_thread_count.get(),
                "작업 스레드 handle 목록 메모리 확보 실패",
            )?;
            for (worker_idx, ((return_rx, local_pool), return_sender)) in worker_buffers
                .into_iter()
                .zip(&return_senders)
                .enumerate()
            {
                let loop_count = base_count.wrapping_add(usize::from(worker_idx < remainder));
                let job = WorkerJob {
                    local_pool,
                    loop_count,
                    processed_ref,
                    return_rx,
                    return_sender,
                    rng,
                    sender: sender.clone(),
                };
                worker_handles.push(scope_ctx.spawn(move || job.run()));
            }
            drop(sender);
            let mut progress_buffers = output::progress::ProgressBuffers::new();
            let mut progress_out = (*IS_TERMINAL).then(|| stdout().lock());
            let mut last_progress = Instant::now();
            let writer_result: Result<()> = loop {
                match receiver.recv_timeout(PROGRESS_UPDATE_INTERVAL) {
                    Ok(chunk) => {
                        let GeneratedChunk {
                            buffer,
                            len,
                            return_sender,
                        } = chunk;
                        let bytes = match output::prefix_slice(&buffer[..], len) {
                            Ok(bytes) => bytes,
                            Err(error) => break Err(error.into()),
                        };
                        if let Err(error) = Write::write_all(output_writer, bytes) {
                            break Err(error.into());
                        }
                        match return_sender.send(buffer) {
                            Ok(()) | Err(_) => {}
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => break output_writer.flush().map_err(Into::into),
                }
                if last_progress.elapsed() >= PROGRESS_UPDATE_INTERVAL
                    && let Some(progress_writer) = progress_out.as_mut()
                {
                    if let Err(error) = progress_buffers.print(
                        progress_writer,
                        processed_ref.load(Ordering::Relaxed),
                        requested_count,
                        start_time.elapsed(),
                    ) {
                        break Err(error);
                    }
                    last_progress = Instant::now();
                }
            };
            drop(receiver);
            let mut combined_outcome = None;
            let mut first_join_error = None;
            for handle in worker_handles {
                match handle.join() {
                    Ok(Some(failure)) => record_failure(
                        &mut combined_outcome,
                        failure.count,
                        failure.first_error,
                    ),
                    Err(panic_payload) if first_join_error.is_none() => {
                        let panic_detail = panic_payload.downcast_ref::<String>().map_or_else(
                            || {
                                panic_payload
                                    .downcast_ref::<&str>()
                                    .map_or("non-string thread payload", |message| *message)
                            },
                            String::as_str,
                        );
                        first_join_error = Some(AppError::message(format!(
                            "작업 스레드 패닉 발생: {panic_detail}"
                        )));
                    }
                    Ok(None) | Err(_) => {}
                }
            }
            let worker_result = first_join_error.map_or(Ok(combined_outcome), Err);
            let worker_outcome = worker_result?;
            writer_result?;
            Ok(worker_outcome)
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
    let mut final_buffer_file = [0_u8; BUFFER_SIZE];
    let final_bytes_written_file = output::format_data_into_buffer(
        &final_data,
        &mut final_buffer_file,
        OutputTarget::File,
    )?;
    let final_file = output_file.writer();
    Write::write_all(
        &mut *final_file,
        output::prefix_slice(&final_buffer_file, final_bytes_written_file)?,
    )?;
    final_file.flush()?;
    let mut progress_buffers = output::progress::ProgressBuffers::new();
    progress_buffers.print(
        out,
        requested_count,
        requested_count,
        start_time.elapsed(),
    )?;
    write!(
        out,
        "\n총 {requested_count}건 생성 완료 ({FILE_NAME} 에 추가).\n\n",
    )?;
    out.flush()?;
    write_random_data_to_console(
        &final_data,
        &mut final_buffer_file,
        final_bytes_written_file,
    )?;
    Ok(final_data.num_64)
}
