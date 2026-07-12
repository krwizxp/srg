use super::{
    FinalCountdownSampler, FinalCountdownSamplerActive, FinalCountdownSamplerCommand,
    NetworkBuffers, NetworkContext, ParsedServer, Result, SampleWorker, SampleWorkerCommand,
    SampleWorkerResponse, TimeError, TimeSample, sample::fetch_server_time_sample,
};
use alloc::sync::Arc;
use core::{sync::atomic::Ordering, time::Duration};
use std::{
    sync::{Mutex, mpsc},
    thread,
};
struct FinalCountdownSamplerWorker {
    command_receiver: mpsc::Receiver<FinalCountdownSamplerCommand>,
    network_context: NetworkContext,
    sample_interval: Duration,
    sample_slot: Arc<Mutex<Option<Result<TimeSample>>>>,
}
impl FinalCountdownSamplerWorker {
    fn run(self) {
        let mut network_context = self.network_context;
        let mut sample_interval = self.sample_interval;
        loop {
            loop {
                match self.command_receiver.try_recv() {
                    Ok(command) => sample_interval = command.interval,
                    Err(mpsc::TryRecvError::Disconnected) => return,
                    Err(mpsc::TryRecvError::Empty) => break,
                }
            }
            let sample_result = fetch_server_time_sample(&mut network_context);
            let Ok(mut slot) = self.sample_slot.lock() else {
                return;
            };
            *slot = Some(sample_result);
            drop(slot);
            match self.command_receiver.recv_timeout(sample_interval) {
                Ok(command) => sample_interval = command.interval,
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    }
}
impl TryFrom<(Arc<ParsedServer>, Duration)> for FinalCountdownSampler {
    type Error = TimeError;
    fn try_from((host, sample_interval): (Arc<ParsedServer>, Duration)) -> Result<Self> {
        let network_buffers = NetworkBuffers::try_new()?;
        let sample_slot = Arc::new(Mutex::new(None));
        let (command_sender, command_receiver) = mpsc::channel();
        let worker_slot = Arc::clone(&sample_slot);
        let worker_name = thread_name(
            "srg-final-countdown-sampler",
            "final countdown sampler thread 이름 메모리 확보 실패",
        )?;
        let join_handle = thread::Builder::new()
            .name(worker_name)
            .spawn(move || {
                let network_context = NetworkContext::new(host, network_buffers);
                FinalCountdownSamplerWorker {
                    command_receiver,
                    network_context,
                    sample_interval,
                    sample_slot: worker_slot,
                }
                .run();
            })
            .map_err(TimeError::from)?;
        Ok(Self(Some(FinalCountdownSamplerActive {
            command_sender,
            join_handle,
            sample_interval,
            sample_slot,
        })))
    }
}
pub(super) fn spawn_sample_worker(initial_host: Arc<ParsedServer>) -> Result<SampleWorker> {
    let network_buffers = NetworkBuffers::try_new()?;
    let (command_sender, command_receiver) = mpsc::sync_channel(0);
    let (response_sender, response_receiver) = mpsc::channel();
    let worker_name = thread_name(
        "srg-sample-worker",
        "sample worker thread 이름 메모리 확보 실패",
    )?;
    let join_handle = thread::Builder::new().name(worker_name).spawn(move || {
        let mut network_context = NetworkContext::new(initial_host, network_buffers);
        loop {
            match command_receiver.recv() {
                Ok(SampleWorkerCommand::Fetch {
                    generation,
                    host: request_host,
                    kind,
                    session_active,
                }) => {
                    if !session_active.load(Ordering::Acquire) {
                        continue;
                    }
                    if !Arc::ptr_eq(&network_context.host, &request_host) {
                        network_context.reset_host(request_host);
                    }
                    let result = fetch_server_time_sample(&mut network_context);
                    if !session_active.load(Ordering::Acquire) {
                        continue;
                    }
                    if response_sender
                        .send(SampleWorkerResponse {
                            generation,
                            kind,
                            result,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
                Err(_) => return,
            }
        }
    })?;
    Ok(SampleWorker {
        command_sender,
        generation: 0,
        join_handle,
        response_receiver,
    })
}
fn thread_name(name: &'static str, context: &'static str) -> Result<String> {
    let mut out = String::new();
    out.try_reserve_exact(name.len())
        .map_err(|source| TimeError::parse_with_source(context, source))?;
    out.push_str(name);
    Ok(out)
}
