use crate::{
    BUFFER_SIZE, BUFFERS_PER_WORKER, DataBuffer, FILE_NAME, IS_TERMINAL, RandomDataSet, Result,
    boxed_other_with_source, ensure_file_exists_and_reopen, generate_random_data, lock_mutex,
    output, persist_and_print_random_data, write_random_data_to_console,
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
    io::{Error as IoError, Write, stdout},
    sync::{
        Mutex, MutexGuard,
        mpsc::{Receiver, SyncSender, TryRecvError, sync_channel},
    },
    thread::{Scope, available_parallelism, scope, sleep},
    time::Instant,
};
const PROGRESS_UPDATE_INTERVAL: Duration = Duration::from_millis(100);
const PROGRESS_TIME_BUF_LEN: usize = 7;
type GeneratedChunk = (DataBuffer, usize, usize);
struct BufferChannels {
    return_receivers: Vec<Receiver<DataBuffer>>,
    return_senders: Vec<SyncSender<DataBuffer>>,
}
struct BufferChannelsBuilder {
    thread_count: usize,
}
impl BufferChannelsBuilder {
    fn build(self) -> Result<BufferChannels> {
        let mut return_senders = Vec::with_capacity(self.thread_count);
        let mut return_receivers = Vec::with_capacity(self.thread_count);
        for _ in 0..self.thread_count {
            let (tx, rx): (SyncSender<DataBuffer>, Receiver<DataBuffer>) =
                sync_channel(BUFFERS_PER_WORKER);
            for _ in 0..BUFFERS_PER_WORKER {
                tx.send(Box::new([0_u8; BUFFER_SIZE]))?;
            }
            return_senders.push(tx);
            return_receivers.push(rx);
        }
        Ok(BufferChannels {
            return_receivers,
            return_senders,
        })
    }
}
struct ProgressReporter<'a> {
    pending_count: u64,
    processed_ref: &'a AtomicU64,
    requested_count: u64,
    start_time: &'a Instant,
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
            let elapsed_millis = elapsed_millis_since(self.start_time);
            output::print_progress(
                &mut out,
                processed_now,
                &mut progress_buffers.line,
                self.requested_count,
                elapsed_millis,
                &mut progress_buffers.elapsed,
                &mut progress_buffers.eta,
            )?;
            sleep(PROGRESS_UPDATE_INTERVAL);
        }
        Ok(())
    }
}
struct ProcessCounters<'a> {
    failed_ref: &'a AtomicU64,
    processed_ref: &'a AtomicU64,
}
impl ProcessCounters<'_> {
    fn finalize(self, multi_thread_count: u64) -> Result<()> {
        let processed_now = self.processed_ref.load(Ordering::Relaxed);
        if processed_now < multi_thread_count {
            let missing = multi_thread_count
                .checked_sub(processed_now)
                .ok_or_else(|| IoError::other("미처리 작업 수 계산 실패"))?;
            self.failed_ref.fetch_add(missing, Ordering::Relaxed);
            self.processed_ref
                .store(multi_thread_count, Ordering::Relaxed);
        }
        Ok(())
    }
}
struct WorkerPool<'a> {
    buffer_return_receivers: Vec<Receiver<DataBuffer>>,
    calculated_thread_count: usize,
    failed_ref: &'a AtomicU64,
    pending_count: u64,
    processed_ref: &'a AtomicU64,
    sender: SyncSender<GeneratedChunk>,
}
impl<'a> WorkerPool<'a> {
    fn join_all(self, scope_ctx: &'a Scope<'a, '_>) -> Result<()> {
        let thread_count_u64 =
            u64::try_from(self.calculated_thread_count).map_err(|conversion_err| {
                boxed_other_with_source("스레드 수 변환 실패", conversion_err)
            })?;
        let base_count = self
            .pending_count
            .checked_div(thread_count_u64)
            .ok_or_else(|| IoError::other("작업 분배 기준 계산 실패"))?;
        let remainder = self
            .pending_count
            .checked_rem(thread_count_u64)
            .ok_or_else(|| IoError::other("작업 분배 나머지 계산 실패"))?;
        let mut worker_handles = Vec::with_capacity(self.calculated_thread_count);
        for (worker_idx, return_rx) in self.buffer_return_receivers.into_iter().enumerate() {
            let sender_clone = self.sender.clone();
            let processed_ref = self.processed_ref;
            let failed_ref = self.failed_ref;
            worker_handles.push(scope_ctx.spawn(move || {
                let Ok(worker_idx_u64) = u64::try_from(worker_idx) else {
                    return;
                };
                let loop_count = base_count.saturating_add(u64::from(worker_idx_u64 < remainder));
                let mut local_pool = Vec::with_capacity(BUFFERS_PER_WORKER);
                for _ in 0..BUFFERS_PER_WORKER {
                    match return_rx.try_recv() {
                        Ok(buffer) => local_pool.push(buffer),
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => return,
                    }
                }
                for _ in 0..loop_count {
                    let mut buffer = match local_pool.pop() {
                        Some(buf) => buf,
                        None => match return_rx.recv() {
                            Ok(buf) => buf,
                            Err(_) => break,
                        },
                    };
                    let len = if let Ok(data) = generate_random_data()
                        && let Ok(len) =
                            output::format_data_into_buffer(&data, buffer.as_mut(), false)
                    {
                        len
                    } else {
                        failed_ref.fetch_add(1, Ordering::Relaxed);
                        processed_ref.fetch_add(1, Ordering::Relaxed);
                        local_pool.push(buffer);
                        continue;
                    };
                    match sender_clone.send((buffer, len, worker_idx)) {
                        Ok(()) => {
                            processed_ref.fetch_add(1, Ordering::Relaxed);
                            while local_pool.len() < BUFFERS_PER_WORKER {
                                match return_rx.try_recv() {
                                    Ok(buf) => local_pool.push(buf),
                                    Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                                }
                            }
                        }
                        Err(send_err) => {
                            failed_ref.fetch_add(1, Ordering::Relaxed);
                            processed_ref.fetch_add(1, Ordering::Relaxed);
                            let (_returned_buffer, _len, _worker_idx) = send_err.0;
                            break;
                        }
                    }
                }
            }));
        }
        drop(self.sender);
        for handle in worker_handles {
            handle.join().map_err(|panic_payload| {
                panic_join_error("작업 스레드 패닉 발생", panic_payload.as_ref())
            })?;
        }
        Ok(())
    }
}
struct WriterTask<'a> {
    buffer_return_senders: Vec<SyncSender<DataBuffer>>,
    file_mutex: &'a Mutex<BufWriter<File>>,
    receiver: Receiver<GeneratedChunk>,
}
impl WriterTask<'_> {
    fn run(self) -> Result<RandomDataSet> {
        let mut file_guard = lock_mutex(self.file_mutex, "Mutex 잠금 실패 (쓰기 스레드)")?;
        while let Ok((data_buffer, data_len, worker_idx)) = self.receiver.recv() {
            Self::write_chunk_and_return_buffer(
                &mut file_guard,
                &self.buffer_return_senders,
                data_buffer,
                data_len,
                worker_idx,
            )?;
            while let Ok((more_buffer, more_len, more_worker_idx)) = self.receiver.try_recv() {
                Self::write_chunk_and_return_buffer(
                    &mut file_guard,
                    &self.buffer_return_senders,
                    more_buffer,
                    more_len,
                    more_worker_idx,
                )?;
            }
        }
        drop(file_guard);
        drop(self.receiver);
        drop(self.buffer_return_senders);
        let final_data = generate_random_data()?;
        let mut final_buffer_file = [0_u8; BUFFER_SIZE];
        let final_bytes_written_file =
            output::format_data_into_buffer(&final_data, &mut final_buffer_file, false)?;
        {
            let mut final_file_guard =
                lock_mutex(self.file_mutex, "Mutex 잠금 실패 (쓰기 스레드 종료 처리)")?;
            output::write_buffer_to_file_guard(
                &mut final_file_guard,
                output::prefix_slice(&final_buffer_file, final_bytes_written_file)?,
            )?;
            final_file_guard.flush()?;
        };
        Ok(final_data)
    }
    fn write_chunk_and_return_buffer(
        file_guard: &mut MutexGuard<BufWriter<File>>,
        buffer_return_senders: &[SyncSender<DataBuffer>],
        data_buffer: DataBuffer,
        data_len: usize,
        worker_idx: usize,
    ) -> Result<()> {
        output::write_buffer_to_file_guard(
            file_guard,
            output::prefix_slice(&data_buffer[..], data_len)?,
        )?;
        if let Some(return_sender) = buffer_return_senders.get(worker_idx) {
            match return_sender.send(data_buffer) {
                Ok(()) | Err(_) => {}
            }
        }
        Ok(())
    }
}
struct BatchRegenerator<'a, 'b, 'c> {
    err: &'c mut dyn Write,
    file_mutex: &'a Mutex<BufWriter<File>>,
    out: &'b mut dyn Write,
    requested_count: u64,
    start_time: Instant,
}
impl BatchRegenerator<'_, '_, '_> {
    fn regenerate(mut self) -> Result<u64> {
        if self.requested_count == 0 {
            return Err(IoError::other("생성 개수는 1 이상이어야 합니다.").into());
        }
        if self.requested_count == 1 {
            return self.regenerate_single();
        }
        let (final_data, failed_count) = self.regenerate_multiple()?;
        self.write_summary(&final_data, failed_count)
    }
    fn regenerate_multiple(&self) -> Result<(RandomDataSet, u64)> {
        ensure_file_exists_and_reopen(self.file_mutex)?;
        let file_mutex = self.file_mutex;
        let requested_count = self.requested_count;
        let start_time = self.start_time;
        let max_threads = available_parallelism().map_or(4, NonZero::get);
        let pending_count = requested_count.saturating_sub(1);
        let calculated_thread_count =
            usize::try_from(pending_count).map_or(max_threads, |count| count.min(max_threads));
        let in_flight_buffers = calculated_thread_count.saturating_mul(BUFFERS_PER_WORKER);
        let channels = BufferChannelsBuilder {
            thread_count: calculated_thread_count,
        }
        .build()?;
        let BufferChannels {
            return_receivers,
            return_senders,
        } = channels;
        let (sender, receiver) = sync_channel::<GeneratedChunk>(in_flight_buffers);
        let processed = AtomicU64::new(0);
        let failed = AtomicU64::new(0);
        let processed_ref = &processed;
        let failed_ref = &failed;
        let final_data = scope(|scope_ctx| -> Result<RandomDataSet> {
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
            WorkerPool {
                buffer_return_receivers: return_receivers,
                calculated_thread_count,
                failed_ref,
                pending_count,
                processed_ref,
                sender,
            }
            .join_all(scope_ctx)?;
            ProcessCounters {
                failed_ref,
                processed_ref,
            }
            .finalize(pending_count)?;
            if let Some(handle) = progress_thread {
                handle.join().map_err(|panic_payload| {
                    panic_join_error("진행률 스레드 패닉 발생", panic_payload.as_ref())
                })??;
            }
            writer_thread.join().map_err(|panic_payload| {
                panic_join_error("쓰기 스레드 패닉 발생", panic_payload.as_ref())
            })?
        })?;
        Ok((final_data, failed_ref.load(Ordering::Relaxed)))
    }
    fn regenerate_single(&self) -> Result<u64> {
        ensure_file_exists_and_reopen(self.file_mutex)?;
        let final_data = generate_random_data()?;
        persist_and_print_random_data(self.file_mutex, &final_data)?;
        Ok(final_data.num_64)
    }
    fn write_summary(&mut self, final_data: &RandomDataSet, failed_count: u64) -> Result<u64> {
        let mut progress_buffers = ProgressBuffers::new();
        let elapsed_millis = elapsed_millis_since(&self.start_time);
        output::print_progress(
            self.out,
            self.requested_count,
            &mut progress_buffers.line,
            self.requested_count,
            elapsed_millis,
            &mut progress_buffers.elapsed,
            &mut progress_buffers.eta,
        )?;
        let success_count = self.requested_count.saturating_sub(failed_count);
        if failed_count > 0 {
            writeln!(self.err, "[경고] 생성 중 {failed_count}건 실패했습니다.")?;
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
        write_random_data_to_console(final_data)?;
        Ok(final_data.num_64)
    }
}
fn panic_join_error(context: &'static str, panic_payload: &(dyn Any + Send + 'static)) -> IoError {
    let panic_detail = panic_payload.downcast_ref::<&str>().map_or_else(
        || {
            panic_payload
                .downcast_ref::<String>()
                .map_or_else(|| String::from("non-string panic payload"), Clone::clone)
        },
        |message| String::from(*message),
    );
    IoError::other(format!("{context}: {panic_detail}"))
}
fn elapsed_millis_since(start_time: &Instant) -> u128 {
    Instant::now().duration_since(*start_time).as_millis()
}
pub fn regenerate_with_count(
    file_mutex: &Mutex<BufWriter<File>>,
    requested_count: u64,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<u64> {
    BatchRegenerator {
        err,
        file_mutex,
        out,
        requested_count,
        start_time: Instant::now(),
    }
    .regenerate()
}
