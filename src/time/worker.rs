use super::{
    FINAL_COUNTDOWN_SAMPLE_WARMUP_INTERVAL, FinalCountdownSampleSlot, FinalCountdownSampler,
    FinalCountdownSamplerCommand, FinalCountdownSamplerCommandFlow, NetworkContext, ParsedServer,
    Result, SampleWorker, SampleWorkerCommand, SampleWorkerResponse, TimeError,
    sample::fetch_server_time_sample,
};
use alloc::{string::String, sync::Arc};
use std::{
    sync::{Mutex, mpsc},
    thread,
};
struct FinalCountdownSamplerWorker {
    command_receiver: mpsc::Receiver<FinalCountdownSamplerCommand>,
    host: Arc<ParsedServer>,
    sample_slot: Arc<Mutex<FinalCountdownSampleSlot>>,
}
impl FinalCountdownSamplerWorker {
    fn publish_startup_error(&self, init_err: TimeError) {
        loop {
            match self.command_receiver.recv() {
                Ok(FinalCountdownSamplerCommand::StartPeriodic { generation, .. }) => {
                    if let Ok(mut slot) = self.sample_slot.lock() {
                        slot.generation = generation;
                        slot.latest_result = Some(Err(init_err));
                    }
                    return;
                }
                Ok(FinalCountdownSamplerCommand::Shutdown) | Err(_) => return,
                Ok(
                    FinalCountdownSamplerCommand::SetInterval { .. }
                    | FinalCountdownSamplerCommand::StopPeriodic,
                ) => {}
            }
        }
    }
    fn run(self) {
        let mut network_context = match NetworkContext::new(Arc::clone(&self.host)) {
            Ok(context) => context,
            Err(init_err) => {
                self.publish_startup_error(init_err);
                return;
            }
        };
        let mut active_generation = None;
        let mut sample_interval = FINAL_COUNTDOWN_SAMPLE_WARMUP_INTERVAL;
        loop {
            while active_generation.is_none() {
                match self.command_receiver.recv() {
                    Ok(command) => {
                        if matches!(
                            command.apply(&mut active_generation, &mut sample_interval),
                            FinalCountdownSamplerCommandFlow::Shutdown
                        ) {
                            return;
                        }
                    }
                    Err(_) => return,
                }
            }
            loop {
                match self.command_receiver.try_recv() {
                    Ok(command) => {
                        if matches!(
                            command.apply(&mut active_generation, &mut sample_interval),
                            FinalCountdownSamplerCommandFlow::Shutdown
                        ) {
                            return;
                        }
                    }
                    Err(mpsc::TryRecvError::Disconnected) => return,
                    Err(mpsc::TryRecvError::Empty) => break,
                }
            }
            let Some(generation) = active_generation else {
                continue;
            };
            let sample_result = fetch_server_time_sample(&mut network_context);
            let Ok(mut slot) = self.sample_slot.lock() else {
                return;
            };
            slot.generation = generation;
            slot.latest_result = Some(sample_result);
            drop(slot);
            match self.command_receiver.recv_timeout(sample_interval) {
                Ok(command) => {
                    if matches!(
                        command.apply(&mut active_generation, &mut sample_interval),
                        FinalCountdownSamplerCommandFlow::Shutdown
                    ) {
                        return;
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    }
}
pub(super) fn spawn_final_countdown_sampler(host: Arc<ParsedServer>) -> FinalCountdownSampler {
    let sample_slot = Arc::new(Mutex::new(FinalCountdownSampleSlot {
        generation: 0,
        latest_result: None,
    }));
    let (command_sender, command_receiver) = mpsc::channel();
    let worker_slot = Arc::clone(&sample_slot);
    let worker_name = match thread_name(
        "srg-final-countdown-sampler",
        "final countdown sampler thread 이름 메모리 확보 실패",
    ) {
        Ok(name) => name,
        Err(startup_error) => {
            return FinalCountdownSampler {
                command_sender: None,
                generation: 0,
                join_handle: None,
                sample_interval: None,
                sample_slot,
                startup_error: Some(startup_error),
                unavailable: true,
            };
        }
    };
    let spawn_result = thread::Builder::new().name(worker_name).spawn(move || {
        FinalCountdownSamplerWorker {
            command_receiver,
            host,
            sample_slot: worker_slot,
        }
        .run();
    });
    match spawn_result {
        Ok(join_handle) => FinalCountdownSampler {
            command_sender: Some(command_sender),
            generation: 0,
            join_handle: Some(join_handle),
            sample_interval: None,
            sample_slot,
            startup_error: None,
            unavailable: false,
        },
        Err(source) => FinalCountdownSampler {
            command_sender: None,
            generation: 0,
            join_handle: None,
            sample_interval: None,
            sample_slot,
            startup_error: Some(TimeError::from(source)),
            unavailable: true,
        },
    }
}
pub(super) fn spawn_sample_worker(host: Arc<ParsedServer>) -> Result<SampleWorker> {
    let (command_sender, command_receiver) = mpsc::channel();
    let (response_sender, response_receiver) = mpsc::channel();
    let worker_name = thread_name(
        "srg-sample-worker",
        "sample worker thread 이름 메모리 확보 실패",
    )?;
    let join_handle = thread::Builder::new().name(worker_name).spawn(move || {
        let mut network_context = match NetworkContext::new(host) {
            Ok(context) => context,
            Err(init_err) => {
                if let Ok(SampleWorkerCommand::Fetch { generation, kind }) = command_receiver.recv()
                {
                    let _send_result = response_sender.send(SampleWorkerResponse {
                        generation,
                        kind,
                        result: Err(init_err),
                    });
                }
                return;
            }
        };
        loop {
            match command_receiver.recv() {
                Ok(SampleWorkerCommand::Fetch { generation, kind }) => {
                    let result = fetch_server_time_sample(&mut network_context);
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
                Ok(SampleWorkerCommand::Stop) | Err(_) => return,
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
