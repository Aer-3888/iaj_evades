use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use rust_evades::{config::GameConfig, game::{Action, GameState}, sensing::ObservationBuilder};
use rust_evades_dqn::{network::Network, model::EvaluationSummary};
use serde::Serialize;
use tokio::sync::broadcast;

#[derive(Clone, Serialize, Debug)]
pub struct EvaluationSeedResult {
    pub seed: u64,
    pub survival_time: f32,
    pub total_return: f32,
    pub evades: u32,
    pub timed_out: bool,
}

#[derive(Clone, Serialize, Debug)]
pub struct EvaluationProgress {
    pub current_seed_index: usize,
    pub total_seeds: usize,
    pub last_result: Option<EvaluationSeedResult>,
    pub summary: EvaluationSummary,
}

pub struct EvaluationManager {
    is_running: Arc<AtomicBool>,
    stop_signal: Arc<AtomicBool>,
    progress_tx: broadcast::Sender<EvaluationProgress>,
}

impl EvaluationManager {
    pub fn new() -> (Self, broadcast::Receiver<EvaluationProgress>) {
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

    pub fn start(
        &self,
        model: Network,
        start_seed: u64,
        num_seeds: usize,
        config: GameConfig,
    ) {
        if self.is_running.load(Ordering::SeqCst) {
            return;
        }

        self.is_running.store(true, Ordering::SeqCst);
        self.stop_signal.store(false, Ordering::SeqCst);
        
        let is_running = self.is_running.clone();
        let stop_signal = self.stop_signal.clone();
        let progress_tx = self.progress_tx.clone();

        tokio::task::spawn_blocking(move || {
            let mut results = Vec::new();
            let mut summary = EvaluationSummary::default();

            for i in 0..num_seeds {
                if stop_signal.load(Ordering::SeqCst) {
                    break;
                }

                let seed = start_seed + i as u64;
                let result = evaluate_seed(&model, seed, config.clone());
                results.push(result.clone());

                // Update summary
                let count = results.len() as f32;
                summary.average_survival_time = results.iter().map(|r| r.survival_time).sum::<f32>() / count;
                summary.average_return = results.iter().map(|r| r.total_return).sum::<f32>() / count;
                summary.average_evades = results.iter().map(|r| r.evades as f32).sum::<f32>() / count;
                summary.min_survival_time = results.iter().map(|r| r.survival_time).fold(f32::INFINITY, f32::min);
                summary.min_return = results.iter().map(|r| r.total_return).fold(f32::INFINITY, f32::min);
                summary.timeouts = results.iter().filter(|r| r.timed_out).count() as u32;

                let progress = EvaluationProgress {
                    current_seed_index: i + 1,
                    total_seeds: num_seeds,
                    last_result: Some(result),
                    summary: summary.clone(),
                };

                let _ = progress_tx.send(progress);
            }

            is_running.store(false, Ordering::SeqCst);
        });
    }

    pub fn stop(&self) {
        self.stop_signal.store(true, Ordering::SeqCst);
    }

    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
    }
}

fn evaluate_seed(network: &Network, seed: u64, config: GameConfig) -> EvaluationSeedResult {
    let mut env = GameState::new(config, Some(seed));
    let mut observation = ObservationBuilder::default();
    observation.reset(&env);
    let mut episode_return = 0.0;

    // We don't have action_repeat here in the simple web loop, 
    // but we can assume 1 or use a default.
    while !env.done {
        let state = observation.build(&env).to_vec();
        
        // Greedy action
        let q_values = network.predict(&state);
        let action_idx = q_values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
            
        let action = Action::ALL[action_idx];
        let result = env.step_fixed(action);
        episode_return += result.reward;
    }

    let report = env.episode_report();
    EvaluationSeedResult {
        seed,
        survival_time: report.elapsed_time,
        total_return: episode_return,
        evades: report.enemies_evaded,
        timed_out: report.survived_full_episode,
    }
}
