use crate::{
    BUFFER_SIZE, BUFFERS_PER_WORKER, DataBuffer, FILE_NAME, IS_TERMINAL, RandomDataSet, Result,
    boxed_other_with_source, describe_panic_payload, ensure_file_exists_and_reopen,
    generate_random_data, lock_mutex, output,
};
use core::{
    num::NonZero,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};
use std::{
    fs::File,
    io::BufWriter,
    io::{Error as ioErr, Write, stdout},
    sync::{
        Mutex,
        mpsc::{Receiver, SyncSender, TryRecvError, sync_channel},
    },
    thread::{Scope, ScopedJoinHandle, available_parallelism, scope, sleep},
    time::Instant,
};
type GeneratedChunk = (DataBuffer, usize, usize);
type BufferReturnChannels = (Vec<SyncSender<DataBuffer>>, Vec<Receiver<DataBuffer>>);
type ProgressHandle<'scope> = ScopedJoinHandle<'scope, Result<()>>;
type WorkerHandle<'scope> = ScopedJoinHandle<'scope, ()>;
struct BatchRegenerator<'file, 'out, 'err> {
    err: &'err mut dyn Write,
    file_mutex: &'file Mutex<BufWriter<File>>,
    out: &'out mut dyn Write,
    requested_count: u64,
    start_time: Instant,
}
trait BatchRegeneratorExt {
    fn build_buffer_return_channels(calculated_thread_count: usize)
    -> Result<BufferReturnChannels>;
    fn finalize_counts(
        processed_ref: &AtomicU64,
        failed_ref: &AtomicU64,
        multi_thread_count: u64,
    ) -> Result<()>;
    fn join_progress_thread(handle: ProgressHandle<'_>) -> Result<()>;
    fn join_worker_handles(worker_handles: Vec<WorkerHandle<'_>>) -> Result<()>;
    fn regenerate(self) -> Result<u64>;
    fn regenerate_multiple(&mut self) -> Result<(RandomDataSet, u64)>;
    fn regenerate_single(&mut self) -> Result<u64>;
    fn run_progress_thread(
        processed_ref: &AtomicU64,
        pending_count: u64,
        requested_count: u64,
        start_time: &Instant,
    ) -> Result<()>;
    fn run_worker(
        worker_idx: usize,
        loop_count: u64,
        return_rx: Receiver<DataBuffer>,
        sender_clone: SyncSender<GeneratedChunk>,
        processed_ref: &AtomicU64,
        failed_ref: &AtomicU64,
    );
    fn run_writer_thread(
        file_mutex: &Mutex<BufWriter<File>>,
        receiver: Receiver<GeneratedChunk>,
        buffer_return_senders: Vec<SyncSender<DataBuffer>>,
    ) -> Result<RandomDataSet>;
    fn spawn_workers<'scope>(
        scope_ctx: &'scope Scope<'scope, '_>,
        calculated_thread_count: usize,
        pending_count: u64,
        buffer_return_receivers: Vec<Receiver<DataBuffer>>,
        sender: SyncSender<GeneratedChunk>,
        processed_ref: &'scope AtomicU64,
        failed_ref: &'scope AtomicU64,
    ) -> Result<Vec<WorkerHandle<'scope>>>;
    fn write_summary(&mut self, final_data: &RandomDataSet, failed_count: u64) -> Result<u64>;
}
impl BatchRegeneratorExt for BatchRegenerator<'_, '_, '_> {
    fn build_buffer_return_channels(
        calculated_thread_count: usize,
    ) -> Result<BufferReturnChannels> {
        let mut buffer_return_senders = Vec::with_capacity(calculated_thread_count);
        let mut buffer_return_receivers = Vec::with_capacity(calculated_thread_count);
        for _ in 0..calculated_thread_count {
            let (tx, rx): (SyncSender<DataBuffer>, Receiver<DataBuffer>) =
                sync_channel(BUFFERS_PER_WORKER);
            for _ in 0..BUFFERS_PER_WORKER {
                tx.send(Box::new([0_u8; BUFFER_SIZE]))?;
            }
            buffer_return_senders.push(tx);
            buffer_return_receivers.push(rx);
        }
        Ok((buffer_return_senders, buffer_return_receivers))
    }
    fn finalize_counts(
        processed_ref: &AtomicU64,
        failed_ref: &AtomicU64,
        multi_thread_count: u64,
    ) -> Result<()> {
        let processed_now = processed_ref.load(Ordering::Relaxed);
        if processed_now < multi_thread_count {
            let missing = multi_thread_count
                .checked_sub(processed_now)
                .ok_or_else(|| ioErr::other("미처리 작업 수 계산 실패"))?;
            failed_ref.fetch_add(missing, Ordering::Relaxed);
            processed_ref.store(multi_thread_count, Ordering::Relaxed);
        }
        Ok(())
    }
    fn join_progress_thread(handle: ProgressHandle<'_>) -> Result<()> {
        handle.join().map_err(|panic_payload| {
            ioErr::other(format!(
                "진행률 스레드 패닉 발생: {panic_detail}",
                panic_detail = describe_panic_payload(panic_payload.as_ref()),
            ))
        })?
    }
    fn join_worker_handles(worker_handles: Vec<WorkerHandle<'_>>) -> Result<()> {
        for handle in worker_handles {
            handle.join().map_err(|panic_payload| {
                ioErr::other(format!(
                    "작업 스레드 패닉 발생: {panic_detail}",
                    panic_detail = describe_panic_payload(panic_payload.as_ref()),
                ))
            })?;
        }
        Ok(())
    }
    fn regenerate(mut self) -> Result<u64> {
        if self.requested_count == 0 {
            return Err(ioErr::other("생성 개수는 1 이상이어야 합니다.").into());
        }
        if self.requested_count == 1 {
            return self.regenerate_single();
        }
        let (final_data, failed_count) = self.regenerate_multiple()?;
        self.write_summary(&final_data, failed_count)
    }
    fn regenerate_multiple(&mut self) -> Result<(RandomDataSet, u64)> {
        ensure_file_exists_and_reopen(self.file_mutex)?;
        let file_mutex = self.file_mutex;
        let requested_count = self.requested_count;
        let start_time = self.start_time;
        let max_threads = available_parallelism().map_or(4, NonZero::get);
        let pending_count = requested_count.saturating_sub(1);
        let calculated_thread_count = usize::try_from(pending_count).map_or(max_threads, |count| {
            if count < max_threads {
                count
            } else {
                max_threads
            }
        });
        let in_flight_buffers = calculated_thread_count.saturating_mul(BUFFERS_PER_WORKER);
        let (sender, receiver) = sync_channel::<GeneratedChunk>(in_flight_buffers);
        let (buffer_return_senders, buffer_return_receivers) =
            Self::build_buffer_return_channels(calculated_thread_count)?;
        let processed = AtomicU64::new(0);
        let failed = AtomicU64::new(0);
        let processed_ref = &processed;
        let failed_ref = &failed;
        let start_time_ref = &start_time;
        let final_data = scope(|scope_ctx| -> Result<RandomDataSet> {
            let writer_thread = scope_ctx.spawn(move || {
                Self::run_writer_thread(file_mutex, receiver, buffer_return_senders)
            });
            let progress_thread = (*IS_TERMINAL).then(|| {
                scope_ctx.spawn(move || {
                    Self::run_progress_thread(
                        processed_ref,
                        pending_count,
                        requested_count,
                        start_time_ref,
                    )
                })
            });
            let worker_handles = Self::spawn_workers(
                scope_ctx,
                calculated_thread_count,
                pending_count,
                buffer_return_receivers,
                sender,
                processed_ref,
                failed_ref,
            )?;
            Self::join_worker_handles(worker_handles)?;
            Self::finalize_counts(processed_ref, failed_ref, requested_count.saturating_sub(1))?;
            if let Some(handle) = progress_thread {
                Self::join_progress_thread(handle)?;
            }
            writer_thread.join().map_err(|panic_payload| {
                ioErr::other(format!(
                    "쓰기 스레드 패닉 발생: {panic_detail}",
                    panic_detail = describe_panic_payload(panic_payload.as_ref()),
                ))
            })?
        })?;
        Ok((final_data, failed_ref.load(Ordering::Relaxed)))
    }
    fn regenerate_single(&mut self) -> Result<u64> {
        ensure_file_exists_and_reopen(self.file_mutex)?;
        let final_data = generate_random_data()?;
        let mut buffer = [0_u8; BUFFER_SIZE];
        let file_len = output::format_data_into_buffer(&final_data, &mut buffer, false)?;
        {
            let mut file_guard = lock_mutex(self.file_mutex, "Mutex 잠금 실패 (단일 쓰기 시)")?;
            output::write_buffer_to_file_guard(
                &mut file_guard,
                output::prefix_slice(&buffer, file_len)?,
            )?;
            file_guard.flush()?;
        };
        if *IS_TERMINAL {
            let console_len = output::format_data_into_buffer(&final_data, &mut buffer, true)?;
            output::write_slice_to_console(output::prefix_slice(&buffer, console_len)?)?;
        } else {
            output::write_slice_to_console(output::prefix_slice(&buffer, file_len)?)?;
        }
        Ok(final_data.num_64)
    }
    fn run_progress_thread(
        processed_ref: &AtomicU64,
        pending_count: u64,
        requested_count: u64,
        start_time: &Instant,
    ) -> Result<()> {
        let mut elapsed_buf = [0_u8; 7];
        let mut eta_buf = [0_u8; 7];
        let mut out = stdout().lock();
        loop {
            let processed_now = processed_ref.load(Ordering::Relaxed);
            if processed_now >= pending_count {
                break;
            }
            output::print_progress(
                &mut out,
                processed_now,
                requested_count,
                start_time,
                &mut elapsed_buf,
                &mut eta_buf,
            )?;
            sleep(Duration::from_millis(100));
        }
        Ok(())
    }
    fn run_worker(
        worker_idx: usize,
        loop_count: u64,
        return_rx: Receiver<DataBuffer>,
        sender_clone: SyncSender<GeneratedChunk>,
        processed_ref: &AtomicU64,
        failed_ref: &AtomicU64,
    ) {
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
                && let Ok(len) = output::format_data_into_buffer(&data, buffer.as_mut(), false)
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
                    let (_returned_buffer, _, _) = send_err.0;
                    break;
                }
            }
        }
        drop(sender_clone);
        drop(return_rx);
    }
    fn run_writer_thread(
        file_mutex: &Mutex<BufWriter<File>>,
        receiver: Receiver<GeneratedChunk>,
        buffer_return_senders: Vec<SyncSender<DataBuffer>>,
    ) -> Result<RandomDataSet> {
        let mut file_guard = lock_mutex(file_mutex, "Mutex 잠금 실패 (쓰기 스레드)")?;
        while let Ok((data_buffer, data_len, worker_idx)) = receiver.recv() {
            output::write_buffer_to_file_guard(
                &mut file_guard,
                output::prefix_slice(&data_buffer[..], data_len)?,
            )?;
            if let Some(return_sender) = buffer_return_senders.get(worker_idx) {
                match return_sender.send(data_buffer) {
                    Ok(()) | Err(_) => {}
                }
            }
            while let Ok((more_buffer, more_len, more_worker_idx)) = receiver.try_recv() {
                output::write_buffer_to_file_guard(
                    &mut file_guard,
                    output::prefix_slice(&more_buffer[..], more_len)?,
                )?;
                if let Some(return_sender) = buffer_return_senders.get(more_worker_idx) {
                    match return_sender.send(more_buffer) {
                        Ok(()) | Err(_) => {}
                    }
                }
            }
        }
        drop(file_guard);
        drop(receiver);
        drop(buffer_return_senders);
        let final_data = generate_random_data()?;
        let mut final_buffer_file = [0_u8; BUFFER_SIZE];
        let final_bytes_written_file =
            output::format_data_into_buffer(&final_data, &mut final_buffer_file, false)?;
        {
            let mut final_file_guard =
                lock_mutex(file_mutex, "Mutex 잠금 실패 (쓰기 스레드 종료 처리)")?;
            output::write_buffer_to_file_guard(
                &mut final_file_guard,
                output::prefix_slice(&final_buffer_file, final_bytes_written_file)?,
            )?;
            final_file_guard.flush()?;
        };
        Ok(final_data)
    }
    fn spawn_workers<'scope>(
        scope_ctx: &'scope Scope<'scope, '_>,
        calculated_thread_count: usize,
        pending_count: u64,
        buffer_return_receivers: Vec<Receiver<DataBuffer>>,
        sender: SyncSender<GeneratedChunk>,
        processed_ref: &'scope AtomicU64,
        failed_ref: &'scope AtomicU64,
    ) -> Result<Vec<WorkerHandle<'scope>>> {
        let thread_count_u64 =
            u64::try_from(calculated_thread_count).map_err(|conversion_err| {
                boxed_other_with_source("스레드 수 변환 실패", conversion_err)
            })?;
        let base_count = pending_count
            .checked_div(thread_count_u64)
            .ok_or_else(|| ioErr::other("작업 분배 기준 계산 실패"))?;
        let remainder = pending_count
            .checked_rem(thread_count_u64)
            .ok_or_else(|| ioErr::other("작업 분배 나머지 계산 실패"))?;
        let mut worker_handles = Vec::with_capacity(calculated_thread_count);
        for (worker_idx, return_rx) in buffer_return_receivers.into_iter().enumerate() {
            let sender_clone = sender.clone();
            worker_handles.push(scope_ctx.spawn(move || {
                let Ok(worker_idx_u64) = u64::try_from(worker_idx) else {
                    return;
                };
                let loop_count = base_count.saturating_add(u64::from(worker_idx_u64 < remainder));
                Self::run_worker(
                    worker_idx,
                    loop_count,
                    return_rx,
                    sender_clone,
                    processed_ref,
                    failed_ref,
                );
            }));
        }
        drop(sender);
        Ok(worker_handles)
    }
    fn write_summary(&mut self, final_data: &RandomDataSet, failed_count: u64) -> Result<u64> {
        let mut elapsed_buf = [0_u8; 7];
        let mut eta_buf = [0_u8; 7];
        output::print_progress(
            self.out,
            self.requested_count,
            self.requested_count,
            &self.start_time,
            &mut elapsed_buf,
            &mut eta_buf,
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
        let mut buffer = [0_u8; BUFFER_SIZE];
        let bytes_written = output::format_data_into_buffer(final_data, &mut buffer, *IS_TERMINAL)?;
        output::write_slice_to_console(output::prefix_slice(&buffer, bytes_written)?)?;
        Ok(final_data.num_64)
    }
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
