use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use rust_evades_dqn::trainer::{train, TrainingConfig, TrainingProgress};
use tokio::sync::mpsc;

pub struct TrainingManager {
    is_running: Arc<AtomicBool>,
    stop_signal: Arc<AtomicBool>,
    progress_tx: broadcast::Sender<TrainingProgress>,
}

use tokio::sync::broadcast;

impl TrainingManager {
    pub fn new() -> (Self, broadcast::Receiver<TrainingProgress>) {
        let (tx, rx) = broadcast::channel(100);
        (
            Self {
                is_running: Arc::new(AtomicBool::new(false)),
                stop_signal: Arc::new(AtomicBool::new(false)),
                progress_tx: tx,
            },
            rx,
        )
    }

    pub fn start(&self, config: TrainingConfig, output_dir: PathBuf) {
        if self.is_running.load(Ordering::SeqCst) {
            return;
        }

        self.is_running.store(true, Ordering::SeqCst);
        self.stop_signal.store(false, Ordering::SeqCst);
        let is_running = self.is_running.clone();
        let stop_signal = self.stop_signal.clone();
        let progress_tx = self.progress_tx.clone();

        // Use spawn_blocking because trainer::train is synchronous and CPU-bound
        tokio::task::spawn_blocking(move || {
            let (tx, mut rx) = mpsc::unbounded_channel();
            
            // Bridge mpsc to broadcast for the web listeners
            let progress_tx_clone = progress_tx.clone();
            let rt = tokio::runtime::Handle::current();
            let _bridge = rt.spawn(async move {
                while let Some(progress) = rx.recv().await {
                    let _ = progress_tx_clone.send(progress);
                }
            });

            let result = train(config, &output_dir, None, Some(tx), Some(stop_signal));
            is_running.store(false, Ordering::SeqCst);
            
            if let Err(e) = result {
                tracing::error!("Training failed: {:?}", e);
            } else {
                tracing::info!("Training completed successfully");
            }
        });
    }

    pub fn stop(&self) {
        self.stop_signal.store(true, Ordering::SeqCst);
    }

    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
    }
}
