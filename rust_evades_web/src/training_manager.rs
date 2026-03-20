use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, RwLock,
    },
};

use rust_evades_dqn::{trainer::{train, TrainingConfig, TrainingProgress}, model::SavedModel};
use tokio::sync::mpsc;
use tokio::sync::broadcast;

pub struct TrainingManager {
    is_running: Arc<AtomicBool>,
    stop_signal: Arc<AtomicBool>,
    progress_tx: broadcast::Sender<TrainingProgress>,
    active_config: Arc<RwLock<Option<TrainingConfig>>>,
    active_resume_model: Arc<RwLock<Option<String>>>,
}

impl TrainingManager {
    pub fn new() -> (Self, broadcast::Receiver<TrainingProgress>) {
        // Capacity is deliberately small: training progress is now throttled to
        // at most ~2 messages/s, so a small buffer is perfectly fine.
        let (tx, rx) = broadcast::channel(32);
        (
            Self {
                is_running: Arc::new(AtomicBool::new(false)),
                stop_signal: Arc::new(AtomicBool::new(false)),
                progress_tx: tx,
                active_config: Arc::new(RwLock::new(None)),
                active_resume_model: Arc::new(RwLock::new(None)),
            },
            rx,
        )
    }

    pub fn start(&self, config: TrainingConfig, output_dir: PathBuf, resume_model: Option<SavedModel>, resume_model_path: Option<String>) {
        if self.is_running.load(Ordering::SeqCst) {
            return;
        }

        *self.active_config.write().unwrap() = Some(config.clone());
        *self.active_resume_model.write().unwrap() = resume_model_path;

        self.is_running.store(true, Ordering::SeqCst);
        self.stop_signal.store(false, Ordering::SeqCst);
        let is_running = self.is_running.clone();
        let stop_signal = self.stop_signal.clone();
        let progress_tx = self.progress_tx.clone();
        let active_config = self.active_config.clone();
        let active_resume_model = self.active_resume_model.clone();

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

            let result = train(config, &output_dir, resume_model, Some(tx), Some(stop_signal));
            is_running.store(false, Ordering::SeqCst);
            *active_config.write().unwrap() = None;
            *active_resume_model.write().unwrap() = None;
            
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

    pub fn get_active_config(&self) -> Option<(TrainingConfig, Option<String>)> {
        let config = self.active_config.read().unwrap().clone();
        let resume_model = self.active_resume_model.read().unwrap().clone();
        config.map(|c| (c, resume_model))
    }
}
