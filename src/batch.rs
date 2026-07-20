use crate::{
    diagnostic::{AppError, Result},
    file_output::OutputFile,
    hardware_rng::HardwareRng,
    output::{self, OutputTarget},
    random_data::generate_random_data_with_rng,
    random_data::RandomDataSet,
    random_output::{persist_and_print_random_data, write_random_data_to_console},
    BUFFER_SIZE, FILE_NAME, IS_TERMINAL,
};
use core::{
    any::Any,
    num::NonZero,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};
use std::{
    fs::File,
    io::BufWriter,
    io::{Write, stdout},
    sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel},
    thread::{ScopedJoinHandle, available_parallelism, scope, sleep},
    time::Instant,
};
const BUFFERS_PER_WORKER: usize = 8;
pub(super) const MAX_BATCH_GENERATE_COUNT: u64 = 10_000_000;
const PROGRESS_UPDATE_INTERVAL: Duration = Duration::from_millis(100);
type DataBuffer = Box<[u8; BUFFER_SIZE]>;
struct GeneratedChunk {
    buffer: DataBuffer,
    len: usize,
    worker_idx: usize,
}
#[derive(Default)]
enum WorkerOutcome {
    Failed { count: u64, first_error: AppError },
    #[default]
    Success,
}
struct WorkerPool<'scope> {
    buffer_return_receivers: Vec<Receiver<DataBuffer>>,
    pending_count: u64,
    processed_ref: &'scope AtomicU64,
    rng: &'scope HardwareRng,
    sender: SyncSender<GeneratedChunk>,
}
struct WorkerJob<'scope> {
    loop_count: u64,
    processed_ref: &'scope AtomicU64,
    return_rx: Receiver<DataBuffer>,
    rng: &'scope HardwareRng,
    sender: &'scope SyncSender<GeneratedChunk>,
    worker_idx: usize,
}
impl WorkerOutcome {
    fn merge(&mut self, other: Self) {
        match other {
            Self::Failed { count, first_error } => match *self {
                Self::Failed {
                    count: ref mut total_count,
                    ..
                } => *total_count = total_count.wrapping_add(count),
                Self::Success => *self = Self::Failed { count, first_error },
            },
            Self::Success => {}
        }
    }
    fn record_failure(&mut self, count: u64, message: AppError) {
        match *self {
            Self::Failed {
                count: ref mut total_count,
                ..
            } => *total_count = total_count.wrapping_add(count),
            Self::Success => {
                *self = Self::Failed {
                    count,
                    first_error: message,
                };
            }
        }
    }
}
impl WorkerJob<'_> {
    fn record_failure(
        &self,
        outcome: &mut WorkerOutcome,
        local_pool: &mut Vec<DataBuffer>,
        buffer: DataBuffer,
        message: AppError,
    ) {
        outcome.record_failure(1, message);
        self.processed_ref.fetch_add(1, Ordering::Relaxed);
        local_pool.push(buffer);
    }
    fn run(self) -> WorkerOutcome {
        let rng = self.rng;
        let mut outcome = WorkerOutcome::default();
        let mut local_pool = Vec::new();
        if let Err(source) = local_pool.try_reserve_exact(BUFFERS_PER_WORKER) {
            outcome.record_failure(
                self.loop_count,
                AppError::context("worker buffer pool 메모리 확보 실패", source),
            );
            self.processed_ref
                .fetch_add(self.loop_count, Ordering::Relaxed);
            return outcome;
        }
        for _ in 0..BUFFERS_PER_WORKER {
            match self.return_rx.try_recv() {
                Ok(buffer) => local_pool.push(buffer),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    outcome.record_failure(
                        self.loop_count,
                        AppError::message("writer buffer 반환 채널 연결 종료"),
                    );
                    self.processed_ref
                        .fetch_add(self.loop_count, Ordering::Relaxed);
                    return outcome;
                }
            }
        }
        for item_index in 0..self.loop_count {
            let mut buffer = if let Some(buf) = local_pool.pop() {
                buf
            } else {
                let Ok(buf) = self.return_rx.recv() else {
                    let remaining = self.loop_count.wrapping_sub(item_index);
                    outcome.record_failure(
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
                    self.record_failure(
                        &mut outcome,
                        &mut local_pool,
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
                    self.record_failure(
                        &mut outcome,
                        &mut local_pool,
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
                    worker_idx: self.worker_idx,
                })
                .is_ok()
            {
                self.processed_ref.fetch_add(1, Ordering::Relaxed);
                while local_pool.len() < BUFFERS_PER_WORKER {
                    match self.return_rx.try_recv() {
                        Ok(buf) => local_pool.push(buf),
                        Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                    }
                }
            } else {
                let remaining = self.loop_count.wrapping_sub(item_index);
                outcome.record_failure(
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
impl WorkerPool<'_> {
    fn join_all(self, thread_count: NonZero<usize>) -> Result<WorkerOutcome> {
        let thread_count_u64 = u64::from_le_bytes(thread_count.get().to_le_bytes());
        let base_count = self.pending_count.div_euclid(thread_count_u64);
        let remainder = self.pending_count.rem_euclid(thread_count_u64);
        scope(|worker_scope| -> Result<WorkerOutcome> {
            let mut worker_handles = try_reserved_vec(
                thread_count.get(),
                "작업 스레드 handle 목록 메모리 확보 실패",
            )?;
            for (worker_idx, return_rx) in self.buffer_return_receivers.into_iter().enumerate() {
                let worker_idx_u64 = u64::from_le_bytes(worker_idx.to_le_bytes());
                let loop_count =
                    base_count.wrapping_add(u64::from(worker_idx_u64 < remainder));
                let job = WorkerJob {
                    loop_count,
                    processed_ref: self.processed_ref,
                    return_rx,
                    rng: self.rng,
                    sender: &self.sender,
                    worker_idx,
                };
                worker_handles.push(worker_scope.spawn(move || job.run()));
            }
            let mut worker_outcome = WorkerOutcome::default();
            let mut first_join_error = None;
            for handle in worker_handles {
                match handle.join() {
                    Ok(outcome) => worker_outcome.merge(outcome),
                    Err(panic_payload) if first_join_error.is_none() => {
                        first_join_error = Some(panic_join_error(
                            "작업 스레드 패닉 발생",
                            panic_payload.as_ref(),
                        ));
                    }
                    Err(_) => {}
                }
            }
            if let Some(error) = first_join_error {
                return Err(error);
            }
            Ok(worker_outcome)
        })
    }
}
struct MultipleBatchRegenerator<'file, 'out, 'rng> {
    out: &'out mut dyn Write,
    output_file: &'file mut OutputFile,
    requested_count: u64,
    rng: &'rng HardwareRng,
    start_time: Instant,
}
impl MultipleBatchRegenerator<'_, '_, '_> {
    fn regenerate_multiple(&mut self) -> Result<RandomDataSet> {
        let output_writer = self.output_file.writer();
        let (requested_count, start_time) = (self.requested_count, self.start_time);
        let max_threads = available_parallelism()?;
        let pending_count = requested_count.wrapping_sub(1);
        let pending_count_usize = usize::from_le_bytes(pending_count.to_le_bytes());
        let calculated_thread_count = NonZero::<usize>::MIN
            .saturating_add(pending_count_usize.wrapping_sub(1))
            .min(max_threads);
        let in_flight_buffers = calculated_thread_count
            .get()
            .wrapping_mul(BUFFERS_PER_WORKER);
        let mut return_senders = try_reserved_vec(
            calculated_thread_count.get(),
            "buffer sender 목록 메모리 확보 실패",
        )?;
        let mut return_receivers = try_reserved_vec(
            calculated_thread_count.get(),
            "buffer receiver 목록 메모리 확보 실패",
        )?;
        for _ in 0..calculated_thread_count.get() {
            let (tx, rx) = sync_channel(BUFFERS_PER_WORKER);
            for _ in 0..BUFFERS_PER_WORKER {
                tx.send(Box::new([0_u8; BUFFER_SIZE]))
                    .map_err(|source| AppError::context("buffer channel send 실패", source))?;
            }
            return_senders.push(tx);
            return_receivers.push(rx);
        }
        let (sender, receiver) = sync_channel::<GeneratedChunk>(in_flight_buffers);
        let processed = AtomicU64::new(0);
        let processed_ref = &processed;
        let worker_outcome = scope(|scope_ctx| -> Result<WorkerOutcome> {
            let writer_thread = scope_ctx.spawn(move || {
                while let Ok(chunk) = receiver.recv() {
                    write_chunk(output_writer, &return_senders, chunk)?;
                    while let Ok(more_chunk) = receiver.try_recv() {
                        write_chunk(output_writer, &return_senders, more_chunk)?;
                    }
                }
                output_writer.flush()?;
                Ok(())
            });
            let progress_thread = (*IS_TERMINAL).then(|| {
                scope_ctx.spawn(move || {
                    let mut progress_buffers = output::progress::ProgressBuffers::new();
                    let mut out = stdout().lock();
                    loop {
                        let processed_now = processed_ref.load(Ordering::Relaxed);
                        if processed_now >= pending_count {
                            break;
                        }
                        progress_buffers.print(
                            &mut out,
                            processed_now,
                            requested_count,
                            start_time.elapsed(),
                        )?;
                        sleep(PROGRESS_UPDATE_INTERVAL);
                    }
                    Ok(())
                })
            });
            let worker_result = WorkerPool {
                buffer_return_receivers: return_receivers,
                pending_count,
                processed_ref,
                rng: &*self.rng,
                sender,
            }
            .join_all(calculated_thread_count);
            if worker_result.is_err() {
                processed_ref.store(pending_count, Ordering::Relaxed);
            }
            let progress_result = progress_thread.map_or(Ok(()), |handle| {
                join_task(handle, "진행률 스레드 패닉 발생")
            });
            let writer_result = join_task(writer_thread, "쓰기 스레드 패닉 발생");
            let worker_outcome = worker_result?;
            progress_result?;
            writer_result?;
            Ok(worker_outcome)
        })?;
        if let WorkerOutcome::Failed {
            count: failed_count,
            first_error,
        } = worker_outcome
        {
            return Err(AppError::context(
                format!(
                    "대량 생성 중 {failed_count}건이 실패했습니다. 성공한 부분 결과만 {FILE_NAME}에 기록되었습니다."
                ),
                first_error,
            ));
        }
        generate_random_data_with_rng(self.rng)
    }
    fn write_summary(&mut self, final_data: &RandomDataSet) -> Result<u64> {
        let mut final_buffer_file = [0_u8; BUFFER_SIZE];
        let final_bytes_written_file =
            output::format_data_into_buffer(final_data, &mut final_buffer_file, OutputTarget::File)?;
        let final_file = self.output_file.writer();
        Write::write_all(
            &mut *final_file,
            output::prefix_slice(&final_buffer_file, final_bytes_written_file)?,
        )?;
        final_file.flush()?;
        let mut progress_buffers = output::progress::ProgressBuffers::new();
        progress_buffers.print(
            self.out,
            self.requested_count,
            self.requested_count,
            self.start_time.elapsed(),
        )?;
        write!(
            self.out,
            "\n총 {requested_count}건 생성 완료 ({FILE_NAME} 에 추가).\n\n",
            requested_count = self.requested_count,
        )?;
        self.out.flush()?;
        write_random_data_to_console(
            final_data,
            &mut final_buffer_file,
            final_bytes_written_file,
        )?;
        Ok(final_data.num_64)
    }
}
fn try_reserved_vec<T>(capacity: usize, context: &'static str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|source| AppError::context(context, source))?;
    Ok(values)
}
fn join_task(handle: ScopedJoinHandle<'_, Result<()>>, panic_context: &str) -> Result<()> {
    match handle.join() {
        Ok(result) => result,
        Err(panic_payload) => Err(panic_join_error(panic_context, panic_payload.as_ref())),
    }
}
fn panic_join_error(context: &str, panic_payload: &(dyn Any + Send)) -> AppError {
    let panic_detail = panic_payload.downcast_ref::<String>().map_or_else(
        || {
            panic_payload
                .downcast_ref::<&str>()
                .map_or("non-string thread payload", |message| *message)
        },
        String::as_str,
    );
    AppError::message(format!("{context}: {panic_detail}"))
}
fn write_chunk(
    file: &mut BufWriter<File>,
    buffer_return_senders: &[SyncSender<DataBuffer>],
    chunk: GeneratedChunk,
) -> Result<()> {
    let GeneratedChunk {
        buffer,
        len,
        worker_idx,
    } = chunk;
    Write::write_all(file, output::prefix_slice(&buffer[..], len)?)?;
    let return_sender = buffer_return_senders
        .get(worker_idx)
        .ok_or("worker buffer 반환 채널 인덱스가 범위를 벗어났습니다")?;
    match return_sender.send(buffer) {
        Ok(()) | Err(_) => {}
    }
    Ok(())
}
pub(super) fn regenerate_with_count(
    output_file: &mut OutputFile,
    rng: &HardwareRng,
    requested_count: u64,
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
    let mut regenerator = MultipleBatchRegenerator {
        out,
        output_file,
        requested_count,
        rng,
        start_time,
    };
    let final_data = regenerator.regenerate_multiple()?;
    regenerator.write_summary(&final_data)
}
