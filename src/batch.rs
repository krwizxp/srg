use crate::{
    constants::{BUFFER_SIZE, BUFFERS_PER_WORKER, FILE_NAME, IS_TERMINAL},
    diagnostic::{AppError, Result},
    file_output::lock_mutex,
    hardware_rng::HardwareRng,
    output::{self, OutputTarget},
    random_data::generate_random_data_with_rng,
    random_data::RandomDataSet,
    random_output::{persist_and_print_random_data, write_random_data_to_console},
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
    sync::{
        Mutex, MutexGuard,
        mpsc::{Receiver, SyncSender, TryRecvError, sync_channel},
    },
    thread::{available_parallelism, scope, sleep},
    time::Instant,
};
pub(super) const MAX_BATCH_GENERATE_COUNT: u64 = 10_000_000;
const PROGRESS_UPDATE_INTERVAL: Duration = Duration::from_millis(100);
const PROGRESS_TIME_BUF_LEN: usize = 7;
type DataBuffer = Box<[u8; BUFFER_SIZE]>;
struct GeneratedChunk {
    buffer: DataBuffer,
    len: usize,
    worker_idx: usize,
}
struct ProgressReporter<'progress> {
    pending_count: u64,
    processed_ref: &'progress AtomicU64,
    requested_count: u64,
    start_time: &'progress Instant,
}
struct ProgressBuffers {
    elapsed: [u8; PROGRESS_TIME_BUF_LEN],
    eta: [u8; PROGRESS_TIME_BUF_LEN],
    line: [u8; output::PROGRESS_LINE_BUF_LEN],
}
impl ProgressBuffers {
    const fn new() -> Self {
        Self {
            elapsed: [0_u8; PROGRESS_TIME_BUF_LEN],
            eta: [0_u8; PROGRESS_TIME_BUF_LEN],
            line: [0_u8; output::PROGRESS_LINE_BUF_LEN],
        }
    }
}
impl ProgressReporter<'_> {
    fn run(self) -> Result<()> {
        let mut progress_buffers = ProgressBuffers::new();
        let mut out = stdout().lock();
        loop {
            let processed_now = self.processed_ref.load(Ordering::Relaxed);
            if processed_now >= self.pending_count {
                break;
            }
            let elapsed = self.start_time.elapsed();
            output::progress::print(
                &mut out,
                processed_now,
                &mut progress_buffers.line,
                self.requested_count,
                elapsed,
                &mut progress_buffers.elapsed,
                &mut progress_buffers.eta,
            )?;
            sleep(PROGRESS_UPDATE_INTERVAL);
        }
        Ok(())
    }
}
struct ProcessCounters<'counters> {
    failed_ref: &'counters AtomicU64,
    processed_ref: &'counters AtomicU64,
}
impl ProcessCounters<'_> {
    fn finalize(self, pending_count: u64) -> Result<()> {
        let processed_now = self.processed_ref.load(Ordering::Relaxed);
        if processed_now < pending_count {
            let missing = pending_count
                .checked_sub(processed_now)
                .ok_or("미처리 작업 수 계산 실패")?;
            self.failed_ref.fetch_add(missing, Ordering::Relaxed);
            self.processed_ref.store(pending_count, Ordering::Relaxed);
        }
        Ok(())
    }
}
struct WorkerPool<'scope> {
    buffer_return_receivers: Vec<Receiver<DataBuffer>>,
    calculated_thread_count: NonZero<usize>,
    failed_ref: &'scope AtomicU64,
    first_error_ref: &'scope Mutex<Option<AppError>>,
    pending_count: u64,
    processed_ref: &'scope AtomicU64,
    rng: &'scope HardwareRng,
    sender: SyncSender<GeneratedChunk>,
}
struct WorkerJob<'scope> {
    failed_ref: &'scope AtomicU64,
    first_error_ref: &'scope Mutex<Option<AppError>>,
    loop_count: u64,
    processed_ref: &'scope AtomicU64,
    return_rx: Receiver<DataBuffer>,
    rng: HardwareRng,
    sender: &'scope SyncSender<GeneratedChunk>,
    worker_idx: usize,
}
impl WorkerJob<'_> {
    fn record_failure(
        first_error_ref: &Mutex<Option<AppError>>,
        failed_ref: &AtomicU64,
        processed_ref: &AtomicU64,
        local_pool: &mut Vec<DataBuffer>,
        buffer: DataBuffer,
        message: AppError,
    ) {
        record_first_error(first_error_ref, message);
        failed_ref.fetch_add(1, Ordering::Relaxed);
        processed_ref.fetch_add(1, Ordering::Relaxed);
        local_pool.push(buffer);
    }
    fn run(self) {
        let mut rng = self.rng;
        let mut local_pool = Vec::new();
        if let Err(source) = local_pool.try_reserve_exact(BUFFERS_PER_WORKER) {
            record_first_error(
                self.first_error_ref,
                AppError::context("worker buffer pool 메모리 확보 실패", source),
            );
            self.failed_ref
                .fetch_add(self.loop_count, Ordering::Relaxed);
            self.processed_ref
                .fetch_add(self.loop_count, Ordering::Relaxed);
            return;
        }
        for _ in 0..BUFFERS_PER_WORKER {
            match self.return_rx.try_recv() {
                Ok(buffer) => local_pool.push(buffer),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        for _ in 0..self.loop_count {
            let mut buffer = match local_pool.pop() {
                Some(buf) => buf,
                None => match self.return_rx.recv() {
                    Ok(buf) => buf,
                    Err(_) => break,
                },
            };
            let data = match generate_random_data_with_rng(&mut rng) {
                Ok(data) => data,
                Err(source) => {
                    Self::record_failure(
                        self.first_error_ref,
                        self.failed_ref,
                        self.processed_ref,
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
                    Self::record_failure(
                        self.first_error_ref,
                        self.failed_ref,
                        self.processed_ref,
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
                record_first_error(
                    self.first_error_ref,
                    AppError::message("writer channel 전송 실패"),
                );
                self.failed_ref.fetch_add(1, Ordering::Relaxed);
                self.processed_ref.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
    }
}
impl WorkerPool<'_> {
    fn join_all(self) -> Result<()> {
        let thread_count_u64 =
            u64::try_from(self.calculated_thread_count.get()).map_err(|conversion_err| {
                AppError::context("스레드 수 변환 실패", conversion_err)
            })?;
        let base_count = self
            .pending_count
            .checked_div(thread_count_u64)
            .ok_or("작업 분배 기준 계산 실패")?;
        let remainder = self
            .pending_count
            .checked_rem(thread_count_u64)
            .ok_or("작업 분배 나머지 계산 실패")?;
        scope(|worker_scope| -> Result<()> {
            let mut worker_handles = Vec::new();
            worker_handles
                .try_reserve_exact(self.calculated_thread_count.get())
                .map_err(|source| {
                    AppError::context("작업 스레드 handle 목록 메모리 확보 실패", source)
                })?;
            for (worker_idx, return_rx) in self.buffer_return_receivers.into_iter().enumerate() {
                let worker_idx_u64 =
                    u64::try_from(worker_idx).map_err(|source| {
                        AppError::context("worker index 변환 실패", source)
                    })?;
                let loop_count = base_count
                    .checked_add(u64::from(worker_idx_u64 < remainder))
                    .ok_or("worker 작업 수 계산 실패")?;
                let job = WorkerJob {
                    failed_ref: self.failed_ref,
                    first_error_ref: self.first_error_ref,
                    loop_count,
                    processed_ref: self.processed_ref,
                    return_rx,
                    rng: self.rng.shared_source_rng(),
                    sender: &self.sender,
                    worker_idx,
                };
                worker_handles.push(worker_scope.spawn(move || job.run()));
            }
            let mut first_join_error = None;
            for handle in worker_handles {
                if let Err(panic_payload) = handle.join()
                    && first_join_error.is_none()
                {
                    first_join_error = Some(panic_join_error(
                        "작업 스레드 패닉 발생",
                        panic_payload.as_ref(),
                    ));
                }
            }
            if let Some(error) = first_join_error {
                return Err(error);
            }
            Ok(())
        })?;
        drop(self.sender);
        Ok(())
    }
}
struct WriterTask<'writer> {
    buffer_return_senders: Vec<SyncSender<DataBuffer>>,
    file_mutex: &'writer Mutex<BufWriter<File>>,
    receiver: Receiver<GeneratedChunk>,
}
impl WriterTask<'_> {
    fn run(self) -> Result<()> {
        let mut file_guard = lock_mutex(self.file_mutex, "Mutex 잠금 실패 (쓰기 스레드)")?;
        while let Ok(chunk) = self.receiver.recv() {
            Self::write_chunk_and_return_buffer(
                &mut file_guard,
                &self.buffer_return_senders,
                chunk,
            )?;
            while let Ok(more_chunk) = self.receiver.try_recv() {
                Self::write_chunk_and_return_buffer(
                    &mut file_guard,
                    &self.buffer_return_senders,
                    more_chunk,
                )?;
            }
        }
        file_guard.flush()?;
        drop(file_guard);
        Ok(())
    }
    fn write_chunk_and_return_buffer(
        file_guard: &mut MutexGuard<'_, BufWriter<File>>,
        buffer_return_senders: &[SyncSender<DataBuffer>],
        chunk: GeneratedChunk,
    ) -> Result<()> {
        let GeneratedChunk {
            buffer,
            len,
            worker_idx,
        } = chunk;
        output::write_buffer_to_file_guard(
            file_guard,
            output::prefix_slice(&buffer[..], len)?,
        )?;
        if let Some(return_sender) = buffer_return_senders.get(worker_idx) {
            match return_sender.send(buffer) {
                Ok(()) | Err(_) => {}
            }
        }
        Ok(())
    }
}
struct BatchRegenerator<'file, 'out, 'err, 'rng> {
    err: &'err mut dyn Write,
    file_mutex: &'file Mutex<BufWriter<File>>,
    out: &'out mut dyn Write,
    requested_count: u64,
    rng: &'rng mut HardwareRng,
    start_time: Instant,
}
struct BatchRunResult {
    failed_count: u64,
    final_data: RandomDataSet,
    first_error: Option<AppError>,
}
impl BatchRegenerator<'_, '_, '_, '_> {
    fn regenerate(mut self) -> Result<u64> {
        if self.requested_count == 0 {
            return Err("생성 개수는 1 이상이어야 합니다.".into());
        }
        if self.requested_count == 1 {
            return self.regenerate_single();
        }
        let run_result = self.regenerate_multiple()?;
        self.write_summary(
            &run_result.final_data,
            run_result.failed_count,
            run_result.first_error.as_ref(),
        )
    }
    fn regenerate_multiple(&mut self) -> Result<BatchRunResult> {
        let file_mutex = self.file_mutex;
        let requested_count = self.requested_count;
        let start_time = self.start_time;
        let max_threads = available_parallelism().map_or(4, NonZero::get);
        let pending_count = requested_count.saturating_sub(1);
        let calculated_thread_count_raw =
            usize::try_from(pending_count).map_or(max_threads, |count| count.min(max_threads));
        let calculated_thread_count =
            NonZero::new(calculated_thread_count_raw).ok_or("작업 스레드 수 계산 실패")?;
        let in_flight_buffers = calculated_thread_count
            .get()
            .checked_mul(BUFFERS_PER_WORKER)
            .ok_or("작업 buffer 채널 용량 계산 실패")?;
        let mut return_senders = Vec::new();
        return_senders
            .try_reserve_exact(calculated_thread_count.get())
            .map_err(|source| AppError::context("buffer sender 목록 메모리 확보 실패", source))?;
        let mut return_receivers = Vec::new();
        return_receivers
            .try_reserve_exact(calculated_thread_count.get())
            .map_err(|source| AppError::context("buffer receiver 목록 메모리 확보 실패", source))?;
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
        let failed = AtomicU64::new(0);
        let first_error = Mutex::new(None);
        let (processed_ref, failed_ref, first_error_ref) = (&processed, &failed, &first_error);
        scope(|scope_ctx| -> Result<()> {
            let writer_thread = scope_ctx.spawn(move || {
                WriterTask {
                    buffer_return_senders: return_senders,
                    file_mutex,
                    receiver,
                }
                .run()
            });
            let progress_thread = (*IS_TERMINAL).then(|| {
                scope_ctx.spawn(move || {
                    ProgressReporter {
                        pending_count,
                        processed_ref,
                        requested_count,
                        start_time: &start_time,
                    }
                    .run()
                })
            });
            let worker_result = WorkerPool {
                buffer_return_receivers: return_receivers,
                calculated_thread_count,
                failed_ref,
                first_error_ref,
                pending_count,
                processed_ref,
                rng: &*self.rng,
                sender,
            }
            .join_all();
            let finalize_result = ProcessCounters { failed_ref, processed_ref }.finalize(pending_count);
            let progress_result = progress_thread.map_or_else(
                || Ok(()),
                |handle| match handle.join() {
                    Ok(result) => result,
                    Err(panic_payload) => Err(panic_join_error(
                        "진행률 스레드 패닉 발생",
                        panic_payload.as_ref(),
                    )),
                },
            );
            let writer_result = match writer_thread.join() {
                Ok(result) => result,
                Err(panic_payload) => Err(panic_join_error(
                    "쓰기 스레드 패닉 발생",
                    panic_payload.as_ref(),
                )),
            };
            worker_result?;
            finalize_result?;
            progress_result?;
            writer_result?;
            Ok(())
        })?;
        self.rng.sync_source_from_shared();
        let final_data = generate_random_data_with_rng(self.rng)?;
        let first_error_text =
            lock_mutex(&first_error, "Mutex 잠금 실패 (batch 최초 오류)")?.take();
        Ok(BatchRunResult {
            failed_count: failed_ref.load(Ordering::Relaxed),
            final_data,
            first_error: first_error_text,
        })
    }
    fn regenerate_single(&mut self) -> Result<u64> {
        let final_data = generate_random_data_with_rng(self.rng)?;
        persist_and_print_random_data(self.file_mutex, &final_data)?;
        Ok(final_data.num_64)
    }
    fn write_summary(
        &mut self,
        final_data: &RandomDataSet,
        failed_count: u64,
        first_error: Option<&AppError>,
    ) -> Result<u64> {
        let mut final_buffer_file = [0_u8; BUFFER_SIZE];
        let final_bytes_written_file =
            output::format_data_into_buffer(final_data, &mut final_buffer_file, OutputTarget::File)?;
        let mut final_file_guard =
            lock_mutex(self.file_mutex, "Mutex 잠금 실패 (대량 생성 최종 기록)")?;
        output::write_buffer_to_file_guard(
            &mut final_file_guard,
            output::prefix_slice(&final_buffer_file, final_bytes_written_file)?,
        )?;
        final_file_guard.flush()?;
        drop(final_file_guard);
        let mut progress_buffers = ProgressBuffers::new();
        let elapsed = self.start_time.elapsed();
        output::progress::print(
            self.out,
            self.requested_count,
            &mut progress_buffers.line,
            self.requested_count,
            elapsed,
            &mut progress_buffers.elapsed,
            &mut progress_buffers.eta,
        )?;
        let success_count = self
            .requested_count
            .checked_sub(failed_count)
            .ok_or("성공 건수 계산 실패")?;
        if failed_count > 0 {
            writeln!(self.err, "[경고] 생성 중 {failed_count}건 실패했습니다.")?;
            if let Some(first_error_text) = first_error {
                writeln!(self.err, "[경고] 최초 실패 원인: {first_error_text}")?;
            }
            write!(
                self.out,
                "\n총 {requested_count}건 중 {success_count}건 생성 완료 ({FILE_NAME} 에 추가).\n\n",
                requested_count = self.requested_count,
            )?;
        } else {
            write!(
                self.out,
                "\n총 {requested_count}건 생성 완료 ({FILE_NAME} 에 추가).\n\n",
                requested_count = self.requested_count,
            )?;
        }
        self.out.flush()?;
        write_random_data_to_console(
            final_data,
            &mut final_buffer_file,
            Some(final_bytes_written_file),
        )?;
        Ok(final_data.num_64)
    }
}
fn record_first_error(
    first_error_ref: &Mutex<Option<AppError>>,
    message: AppError,
) {
    if let Ok(mut first_error) = first_error_ref.lock()
        && first_error.is_none()
    {
        *first_error = Some(message);
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
pub(super) fn regenerate_with_count(
    file_mutex: &Mutex<BufWriter<File>>,
    rng: &mut HardwareRng,
    requested_count: u64,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<u64> {
    if requested_count > MAX_BATCH_GENERATE_COUNT {
        return Err(AppError::message(format!(
            "대량 생성 개수는 최대 {MAX_BATCH_GENERATE_COUNT}건까지 입력할 수 있습니다."
        )));
    }
    BatchRegenerator {
        err,
        file_mutex,
        out,
        requested_count,
        rng,
        start_time: Instant::now(),
    }
    .regenerate()
}
