use crate::{
    BUFFER_SIZE, BUFFERS_PER_WORKER, DataBuffer, FILE_NAME, IS_TERMINAL, RandomDataSet, Result,
    boxed_other_with_source, describe_panic_payload, ensure_file_exists_and_reopen,
    generate_random_data, lock_mutex, output, read_line_reuse,
};
use std::{
    fs::File,
    io::BufWriter,
    io::{Error as ioErr, Write as _, stdout},
    num::NonZero,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{Receiver, SyncSender, TryRecvError, sync_channel},
    },
    thread::{ScopedJoinHandle, available_parallelism, scope, sleep},
    time::{Duration, Instant},
};

fn writer_thread_main(
    file_mutex: &Mutex<BufWriter<File>>,
    receiver: &Receiver<(DataBuffer, usize, usize)>,
    buffer_return_senders: &[SyncSender<DataBuffer>],
) -> Result<RandomDataSet> {
    let mut file_guard = lock_mutex(file_mutex, "Mutex 잠금 실패 (쓰기 스레드)")?;
    while let Ok((data_buffer, data_len, worker_idx)) = receiver.recv() {
        output::write_buffer_to_file_guard(
            &mut file_guard,
            output::prefix_slice(&data_buffer[..], data_len)?,
        )?;
        if let Some(sender) = buffer_return_senders.get(worker_idx) {
            let _send_result = sender.send(data_buffer);
        }
        while let Ok((more_buffer, more_len, more_worker_idx)) = receiver.try_recv() {
            output::write_buffer_to_file_guard(
                &mut file_guard,
                output::prefix_slice(&more_buffer[..], more_len)?,
            )?;
            if let Some(sender) = buffer_return_senders.get(more_worker_idx) {
                let _batched_send_result = sender.send(more_buffer);
            }
        }
    }
    drop(file_guard);
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
    }
    Ok(final_data)
}

fn progress_thread_main(
    processed_ref: &AtomicU64,
    multi_thread_count: u64,
    requested_count: u64,
    start_time: &Instant,
) -> Result<()> {
    let mut elapsed_buf = [0_u8; 7];
    let mut eta_buf = [0_u8; 7];
    loop {
        let processed_now = processed_ref.load(Ordering::Relaxed);
        if processed_now >= multi_thread_count {
            break;
        }
        output::print_progress(
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

fn worker_thread_main(
    worker_idx: usize,
    return_rx: &Receiver<DataBuffer>,
    sender_clone: &SyncSender<(DataBuffer, usize, usize)>,
    processed_ref: &AtomicU64,
    failed_ref: &AtomicU64,
    base_count: u64,
    remainder: u64,
) {
    let Ok(worker_idx_u64) = u64::try_from(worker_idx) else {
        return;
    };
    let loop_count = base_count + u64::from(worker_idx_u64 < remainder);
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
}

fn read_requested_count(input_buffer: &mut String) -> Result<u64> {
    let count_prompt = format_args!("\n생성할 데이터 개수를 입력해 주세요: ");
    loop {
        match read_line_reuse(count_prompt, input_buffer)?.parse::<u64>() {
            Ok(0) => eprintln!("1 이상의 값을 입력해 주세요."),
            Ok(n) => return Ok(n),
            Err(_) => eprintln!("유효한 숫자를 입력해 주세요."),
        }
    }
}

fn print_regenerate_summary(requested_count: u64, failed_count: u64) {
    let success_count = requested_count.saturating_sub(failed_count);
    if failed_count > 0 {
        eprintln!("[경고] 생성 중 {failed_count}건의 실패가 발생했습니다.");
        println!(
            "\n총 {requested_count}개 요청 중 {success_count}개 데이터 생성 완료 ({FILE_NAME} 저장됨).\n"
        );
    } else {
        println!("\n총 {requested_count}개의 데이터 생성 완료 ({FILE_NAME} 저장됨).\n");
    }
}

pub fn regenerate_multiple(
    file_mutex: &Mutex<BufWriter<File>>,
    input_buffer: &mut String,
) -> Result<u64> {
    let requested_count = read_requested_count(input_buffer)?;
    if requested_count == 1 {
        ensure_file_exists_and_reopen(file_mutex)?;
        return super::process_single_random_data(file_mutex);
    }
    ensure_file_exists_and_reopen(file_mutex)?;
    let multi_thread_count = requested_count.saturating_sub(1);
    let max_threads = available_parallelism().map_or(4, NonZero::get);
    let calculated_thread_count = usize::try_from(multi_thread_count).map_or(max_threads, |n| {
        if n < max_threads { n } else { max_threads }
    });
    let start_time = Instant::now();
    let in_flight_buffers = calculated_thread_count.saturating_mul(BUFFERS_PER_WORKER);
    let (sender, receiver) = sync_channel::<(DataBuffer, usize, usize)>(in_flight_buffers);
    let mut buffer_return_senders = Vec::with_capacity(calculated_thread_count);
    let mut buffer_return_receivers = Vec::with_capacity(calculated_thread_count);
    for _ in 0..calculated_thread_count {
        let (tx, rx) = sync_channel::<DataBuffer>(BUFFERS_PER_WORKER);
        for _ in 0..BUFFERS_PER_WORKER {
            tx.send(Box::new([0_u8; BUFFER_SIZE]))?;
        }
        buffer_return_senders.push(tx);
        buffer_return_receivers.push(rx);
    }
    let processed = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let final_data = scope(|s| -> Result<RandomDataSet> {
        let writer_buffer_return_senders = buffer_return_senders;
        let writer_thread = s.spawn(move || {
            writer_thread_main(file_mutex, &receiver, &writer_buffer_return_senders)
        });
        let processed_ref = &processed;
        let failed_ref = &failed;
        let progress_thread: Option<ScopedJoinHandle<Result<()>>> = (*IS_TERMINAL).then(|| {
            s.spawn(move || {
                progress_thread_main(
                    processed_ref,
                    multi_thread_count,
                    requested_count,
                    &start_time,
                )
            })
        });
        let thread_count_u64 = u64::try_from(calculated_thread_count)
            .map_err(|err| boxed_other_with_source("스레드 수 변환 실패", err))?;
        let base_count = multi_thread_count / thread_count_u64;
        let remainder = multi_thread_count % thread_count_u64;
        let mut worker_handles = Vec::with_capacity(calculated_thread_count);
        for (i, return_rx) in buffer_return_receivers.into_iter().enumerate() {
            let sender_clone = sender.clone();
            worker_handles.push(s.spawn(move || {
                worker_thread_main(
                    i,
                    &return_rx,
                    &sender_clone,
                    processed_ref,
                    failed_ref,
                    base_count,
                    remainder,
                );
            }));
        }
        drop(sender);
        for handle in worker_handles {
            handle.join().map_err(|panic_payload| {
                ioErr::other(format!(
                    "작업 스레드 패닉 발생: {panic_detail}",
                    panic_detail = describe_panic_payload(panic_payload.as_ref()),
                ))
            })?;
        }
        let processed_now = processed_ref.load(Ordering::Relaxed);
        if processed_now < multi_thread_count {
            let missing = multi_thread_count - processed_now;
            failed_ref.fetch_add(missing, Ordering::Relaxed);
            processed_ref.store(multi_thread_count, Ordering::Relaxed);
        }
        if let Some(handle) = progress_thread {
            join_thread(handle, "진행률 스레드 패닉 발생")?;
        }
        join_thread(writer_thread, "쓰기 스레드 패닉 발생")
    })?;
    let mut elapsed_buf = [0_u8; 7];
    let mut eta_buf = [0_u8; 7];
    output::print_progress(
        requested_count,
        requested_count,
        &start_time,
        &mut elapsed_buf,
        &mut eta_buf,
    )?;
    let failed_count = failed.load(Ordering::Relaxed);
    print_regenerate_summary(requested_count, failed_count);
    stdout().flush()?;
    let mut buffer = [0_u8; BUFFER_SIZE];
    let bytes_written = output::format_data_into_buffer(&final_data, &mut buffer, *IS_TERMINAL)?;
    output::write_slice_to_console(output::prefix_slice(&buffer, bytes_written)?)?;
    Ok(final_data.num_64)
}

fn join_thread<T>(handle: ScopedJoinHandle<'_, Result<T>>, panic_msg: &'static str) -> Result<T> {
    handle.join().map_err(|panic_payload| {
        ioErr::other(format!(
            "{panic_msg}: {panic_detail}",
            panic_detail = describe_panic_payload(panic_payload.as_ref()),
        ))
    })?
}
