use std::{
    cmp::Ordering,
    collections::VecDeque,
    env, fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering as AtomicOrdering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::Context;
use rand::{prelude::*, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use rust_evades::{
    config::{GameConfig, MapDesign},
    game::Action,
    game::GameState,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::{
    model::{EvaluationSummary, ModelType, SavedModel},
    network::Network,
    observation::{DualRayObservationBuilder, ObservationBuilder},
};

const EVAL_CPU_FRACTION: f32 = 0.75;
const MIN_PARALLEL_EVAL_SEEDS: usize = 8;
const OPTIMIZER_CPU_FRACTION: f32 = 0.75;
const MIN_PARALLEL_TRAIN_BATCH: usize = 64;
const FOCUS_SEED_DIVISOR: usize = 4;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum SeedFocusMode {
    Original,
    #[default]
    BadSeeds,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    #[serde(default)]
    pub model_type: ModelType,
    pub episodes: usize,
    pub trainer_seed: u64,
    pub checkpoint_every: usize,
    pub seed_focus_mode: SeedFocusMode,
    pub fixed_training_seeds: Vec<u64>,
    pub random_seed_count_per_cycle: usize,
    pub hidden_sizes: Vec<usize>,
    pub replay_capacity: usize,
    pub batch_size: usize,
    pub warmup_steps: usize,
    pub train_every: usize,
    pub target_sync_interval: usize,
    pub learning_rate: f32,
    #[serde(default = "default_huber_delta")]
    pub huber_delta: f32,
    #[serde(default = "default_gradient_clip_norm")]
    pub gradient_clip_norm: f32,
    pub gamma: f32,
    pub epsilon_start: f32,
    pub epsilon_end: f32,
    pub epsilon_decay_steps: usize,
    pub action_repeat: usize,
    #[serde(default = "default_training_map_design")]
    pub map_design: MapDesign,
}

fn default_training_map_design() -> MapDesign {
    MapDesign::Open
}

fn default_huber_delta() -> f32 {
    10.0
}

fn default_gradient_clip_norm() -> f32 {
    1280.0
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::Dqn,
            episodes: 500000,
            trainer_seed: 7,
            checkpoint_every: 100,
            seed_focus_mode: SeedFocusMode::BadSeeds,
            fixed_training_seeds: default_training_seeds(2, 24),
            random_seed_count_per_cycle: 2,
            hidden_sizes: vec![128, 128],
            replay_capacity: 100_000,
            batch_size: 128,
            warmup_steps: 2_000,
            train_every: 2,
            target_sync_interval: 1_000,
            learning_rate: 0.0003,
            huber_delta: 10.0,
            gradient_clip_norm: 1280.0,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.03,
            epsilon_decay_steps: 120_000,
            action_repeat: 2,
            map_design: MapDesign::Open,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingResult {
    pub best_metrics: EvaluationSummary,
    pub completed_episodes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkMode {
    FullTraining,
    SimulatedSurvival,
}

#[derive(Clone, Copy, Debug)]
enum ExecutionMode {
    RealGame,
    SimulatedSurvival { survival_time: f32 },
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchmarkReport {
    pub mode: BenchmarkMode,
    pub output_dir: PathBuf,
    pub report_path: PathBuf,
    pub simulated_survival_seconds: f32,
    pub episodes_completed: usize,
    pub total_steps_completed: usize,
    pub best_metrics: EvaluationSummary,
    pub profile: ProfileSnapshot,
}

#[derive(Clone, Debug)]
struct ResumeState {
    network: Network,
    best_metrics: EvaluationSummary,
    completed_episodes: usize,
    total_steps: usize,
}

#[derive(Clone, Debug)]
pub struct Transition {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

#[derive(Clone, Debug)]
struct ReplayBuffer {
    entries: VecDeque<Transition>,
    capacity: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, transition: Transition) {
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(transition);
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn sample<'a>(&'a self, batch_size: usize, rng: &mut impl Rng) -> Vec<&'a Transition> {
        (0..batch_size)
            .map(|_| {
                let index = rng.gen_range(0..self.entries.len());
                &self.entries[index]
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SeedEpisodeSummary {
    total_return: f32,
    survival_time: f32,
    evades: f32,
    timed_out: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ProfileStats {
    total_wall: Duration,
    evaluation_runtime_choice: Duration,
    optimization_runtime_choice: Duration,
    environment_and_policy: Duration,
    action_selection: Duration,
    environment_step: Duration,
    observation_build: Duration,
    optimization: Duration,
    evaluation: Duration,
    serialization: Duration,
    replay_sampling: Duration,
    diagnostics: Duration,
    train_batch: Duration,
    target_sync: Duration,
    steps: usize,
    train_updates: usize,
    evaluations: usize,
    saves: usize,
}

impl ProfileStats {
    fn print(&self) {
        let total = self.total_wall.as_secs_f64().max(1.0e-9);
        println!("profiling summary:");
        println!("  total: {:.3}s (100.0%)", self.total_wall.as_secs_f64());
        println!(
            "  env+policy: {:.3}s ({:.1}%)",
            self.environment_and_policy.as_secs_f64(),
            self.environment_and_policy.as_secs_f64() * 100.0 / total
        );
        println!(
            "  optimization: {:.3}s ({:.1}%)",
            self.optimization.as_secs_f64(),
            self.optimization.as_secs_f64() * 100.0 / total
        );
        println!(
            "  action selection: {:.3}s ({:.1}%)",
            self.action_selection.as_secs_f64(),
            self.action_selection.as_secs_f64() * 100.0 / total
        );
        println!(
            "  environment step: {:.3}s ({:.1}%)",
            self.environment_step.as_secs_f64(),
            self.environment_step.as_secs_f64() * 100.0 / total
        );
        println!(
            "  observation build: {:.3}s ({:.1}%)",
            self.observation_build.as_secs_f64(),
            self.observation_build.as_secs_f64() * 100.0 / total
        );
        println!(
            "  evaluation: {:.3}s ({:.1}%)",
            self.evaluation.as_secs_f64(),
            self.evaluation.as_secs_f64() * 100.0 / total
        );
        println!(
            "  eval setup benchmark: {:.3}s ({:.1}%)",
            self.evaluation_runtime_choice.as_secs_f64(),
            self.evaluation_runtime_choice.as_secs_f64() * 100.0 / total
        );
        println!(
            "  optimizer setup benchmark: {:.3}s ({:.1}%)",
            self.optimization_runtime_choice.as_secs_f64(),
            self.optimization_runtime_choice.as_secs_f64() * 100.0 / total
        );
        println!(
            "  serialization: {:.3}s ({:.1}%)",
            self.serialization.as_secs_f64(),
            self.serialization.as_secs_f64() * 100.0 / total
        );
        println!(
            "  replay sampling: {:.3}s ({:.1}%)",
            self.replay_sampling.as_secs_f64(),
            self.replay_sampling.as_secs_f64() * 100.0 / total
        );
        println!(
            "  batch diagnostics: {:.3}s ({:.1}%)",
            self.diagnostics.as_secs_f64(),
            self.diagnostics.as_secs_f64() * 100.0 / total
        );
        println!(
            "  train batch core: {:.3}s ({:.1}%)",
            self.train_batch.as_secs_f64(),
            self.train_batch.as_secs_f64() * 100.0 / total
        );
        println!(
            "  target sync: {:.3}s ({:.1}%)",
            self.target_sync.as_secs_f64(),
            self.target_sync.as_secs_f64() * 100.0 / total
        );
        println!(
            "  steps: {}  train_updates: {}  evals: {}  saves: {}",
            self.steps, self.train_updates, self.evaluations, self.saves
        );
    }

    fn snapshot(&self) -> ProfileSnapshot {
        ProfileSnapshot {
            total_wall_seconds: self.total_wall.as_secs_f64(),
            evaluation_runtime_choice_seconds: self.evaluation_runtime_choice.as_secs_f64(),
            optimization_runtime_choice_seconds: self.optimization_runtime_choice.as_secs_f64(),
            environment_and_policy_seconds: self.environment_and_policy.as_secs_f64(),
            action_selection_seconds: self.action_selection.as_secs_f64(),
            environment_step_seconds: self.environment_step.as_secs_f64(),
            observation_build_seconds: self.observation_build.as_secs_f64(),
            optimization_seconds: self.optimization.as_secs_f64(),
            evaluation_seconds: self.evaluation.as_secs_f64(),
            serialization_seconds: self.serialization.as_secs_f64(),
            replay_sampling_seconds: self.replay_sampling.as_secs_f64(),
            diagnostics_seconds: self.diagnostics.as_secs_f64(),
            train_batch_seconds: self.train_batch.as_secs_f64(),
            target_sync_seconds: self.target_sync.as_secs_f64(),
            steps: self.steps,
            train_updates: self.train_updates,
            evaluations: self.evaluations,
            saves: self.saves,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ProfileSnapshot {
    pub total_wall_seconds: f64,
    pub evaluation_runtime_choice_seconds: f64,
    pub optimization_runtime_choice_seconds: f64,
    pub environment_and_policy_seconds: f64,
    pub action_selection_seconds: f64,
    pub environment_step_seconds: f64,
    pub observation_build_seconds: f64,
    pub optimization_seconds: f64,
    pub evaluation_seconds: f64,
    pub serialization_seconds: f64,
    pub replay_sampling_seconds: f64,
    pub diagnostics_seconds: f64,
    pub train_batch_seconds: f64,
    pub target_sync_seconds: f64,
    pub steps: usize,
    pub train_updates: usize,
    pub evaluations: usize,
    pub saves: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct StepDiagnostics {
    mean_predicted_q: f32,
    mean_target_q: f32,
    mean_abs_td_error: f32,
    terminal_fraction: f32,
}

#[derive(Clone, Debug, Default)]
struct EpisodeLoopMetrics {
    total_return: f32,
    survival_time: f32,
    evades: u32,
    losses: Vec<f32>,
    diagnostics: StepDiagnostics,
}

#[derive(Clone, Copy, Debug)]
struct SyntheticStepResult {
    reward: f32,
    done: bool,
    elapsed_time: f32,
}

#[derive(Clone, Debug)]
struct SyntheticEpisode {
    config: GameConfig,
    total_steps: usize,
    step_index: usize,
    elapsed_time: f32,
    total_return: f32,
    current_state: Vec<f32>,
}

impl SyntheticEpisode {
    fn new(
        config: GameConfig,
        seed: u64,
        model_type: ModelType,
        survival_time: f32,
        action_repeat: usize,
    ) -> Self {
        let total_steps = synthetic_total_steps(&config, survival_time, action_repeat);
        let current_state = synthetic_observation(seed, 0, Action::Idle, model_type.input_size());
        Self {
            config,
            total_steps,
            step_index: 0,
            elapsed_time: 0.0,
            total_return: 0.0,
            current_state,
        }
    }

    fn current_state(&self) -> &[f32] {
        &self.current_state
    }

    fn step(
        &mut self,
        seed: u64,
        action: Action,
        action_repeat: usize,
        input_size: usize,
    ) -> SyntheticStepResult {
        let delta = self.config.fixed_timestep * action_repeat.max(1) as f32;
        self.step_index += 1;
        self.elapsed_time = (self.step_index as f32 * delta).min(self.config.max_episode_time);

        let mut reward = self.config.survival_reward_per_second * delta;
        let done = self.step_index >= self.total_steps;
        if done {
            reward += self.config.timeout_bonus;
        }

        self.total_return += reward;
        self.current_state = synthetic_observation(seed, self.step_index, action, input_size);

        SyntheticStepResult {
            reward,
            done,
            elapsed_time: self.elapsed_time,
        }
    }
}

enum EvaluationRuntime {
    Sequential,
    Parallel { pool: ThreadPool },
}

enum OptimizationRuntime {
    Sequential,
    Parallel { pool: ThreadPool, chunk_size: usize },
}

/// Unified observation builder that handles both Dqn and Dqn2 model types.
enum ObsBuilderState {
    Dqn(ObservationBuilder),
    Dqn2(DualRayObservationBuilder),
}

impl ObsBuilderState {
    fn new(model_type: ModelType) -> Self {
        match model_type {
            ModelType::Dqn => ObsBuilderState::Dqn(ObservationBuilder::default()),
            ModelType::Dqn2 => ObsBuilderState::Dqn2(DualRayObservationBuilder::default()),
        }
    }

    fn reset(&mut self, state: &GameState) {
        match self {
            ObsBuilderState::Dqn(b) => b.reset(state),
            ObsBuilderState::Dqn2(b) => b.reset(state),
        }
    }

    fn build_vec(&mut self, state: &GameState) -> Vec<f32> {
        match self {
            ObsBuilderState::Dqn(b) => b.build(state).to_vec(),
            ObsBuilderState::Dqn2(b) => b.build(state).to_vec(),
        }
    }
}

fn env_config_for_training(map_design: MapDesign) -> GameConfig {
    let mut config = GameConfig::default();
    config.map_design = map_design;
    config
}

impl EvaluationRuntime {
    fn choose(
        network: &Network,
        seeds: &[u64],
        action_repeat: usize,
        model_type: ModelType,
        map_design: MapDesign,
        execution_mode: ExecutionMode,
    ) -> anyhow::Result<Self> {
        let available = thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1);
        let desired_threads = ((available as f32) * EVAL_CPU_FRACTION)
            .round()
            .clamp(1.0, available as f32) as usize;

        if desired_threads <= 1 || seeds.len() < MIN_PARALLEL_EVAL_SEEDS {
            println!("evaluation mode: sequential");
            return Ok(Self::Sequential);
        }

        let pool = ThreadPoolBuilder::new()
            .num_threads(desired_threads)
            .build()
            .context("failed to build evaluation thread pool")?;

        let sequential_benchmark = benchmark_evaluation(
            || {
                evaluate_seed_batch_sequential(
                    network,
                    seeds,
                    action_repeat,
                    model_type,
                    map_design,
                    execution_mode,
                )
            },
            2,
        );
        let parallel_benchmark = benchmark_evaluation(
            || {
                pool.install(|| {
                    evaluate_seed_batch_parallel(
                        network,
                        seeds,
                        action_repeat,
                        model_type,
                        map_design,
                        execution_mode,
                    )
                })
            },
            2,
        );

        if parallel_benchmark < sequential_benchmark {
            println!(
                "evaluation mode: parallel with {} threads (seq {:?}, par {:?})",
                desired_threads, sequential_benchmark, parallel_benchmark
            );
            Ok(Self::Parallel { pool })
        } else {
            println!(
                "evaluation mode: sequential (parallel {:?} was not faster than sequential {:?})",
                parallel_benchmark, sequential_benchmark
            );
            Ok(Self::Sequential)
        }
    }

    fn evaluate_batch(
        &self,
        network: &Network,
        seeds: &[u64],
        action_repeat: usize,
        model_type: ModelType,
        map_design: MapDesign,
        execution_mode: ExecutionMode,
    ) -> Vec<SeedEpisodeSummary> {
        match self {
            Self::Sequential => evaluate_seed_batch_sequential(
                network,
                seeds,
                action_repeat,
                model_type,
                map_design,
                execution_mode,
            ),
            Self::Parallel { pool } => pool.install(|| {
                evaluate_seed_batch_parallel(
                    network,
                    seeds,
                    action_repeat,
                    model_type,
                    map_design,
                    execution_mode,
                )
            }),
        }
    }
}

impl OptimizationRuntime {
    fn choose(
        network: &Network,
        target_network: &Network,
        batch_size: usize,
        gamma: f32,
        learning_rate: f32,
        huber_delta: f32,
        gradient_clip_norm: f32,
        input_size: usize,
    ) -> anyhow::Result<Self> {
        let available = thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1);
        let desired_threads = ((available as f32) * OPTIMIZER_CPU_FRACTION)
            .round()
            .clamp(1.0, available as f32) as usize;

        if desired_threads <= 1 || batch_size < MIN_PARALLEL_TRAIN_BATCH {
            println!("optimizer mode: sequential");
            return Ok(Self::Sequential);
        }

        let pool = ThreadPoolBuilder::new()
            .num_threads(desired_threads)
            .build()
            .context("failed to build optimizer thread pool")?;
        let benchmark_batch = build_optimizer_benchmark_batch(batch_size, input_size);
        let mut chunk_size_candidates = vec![
            benchmark_batch.len().div_ceil(desired_threads * 2).max(1),
            benchmark_batch.len().div_ceil(desired_threads).max(1),
            benchmark_batch.len().div_ceil(desired_threads.saturating_sub(1).max(1)).max(1),
        ];
        chunk_size_candidates.sort_unstable();
        chunk_size_candidates.dedup();

        let sequential_benchmark = benchmark_runtime(
            || {
                let mut online = network.clone();
                let _ = online.train_batch(
                    target_network,
                    &benchmark_batch,
                    gamma,
                    learning_rate,
                    huber_delta,
                    gradient_clip_norm,
                );
            },
            2,
        );
        let mut best_parallel = None;
        for chunk_size in chunk_size_candidates {
            let parallel_benchmark = benchmark_runtime(
                || {
                    let mut online = network.clone();
                    pool.install(|| {
                        let _ = online.train_batch_parallel(
                            target_network,
                            &benchmark_batch,
                            gamma,
                            learning_rate,
                            huber_delta,
                            gradient_clip_norm,
                            chunk_size,
                        );
                    });
                },
                2,
            );
            if best_parallel
                .map(|(best_duration, _)| parallel_benchmark < best_duration)
                .unwrap_or(true)
            {
                best_parallel = Some((parallel_benchmark, chunk_size));
            }
        }
        let (parallel_benchmark, chunk_size) =
            best_parallel.expect("chunk size candidates must not be empty");

        if parallel_benchmark < sequential_benchmark {
            println!(
                "optimizer mode: parallel with {} threads and chunk size {} (seq {:?}, par {:?})",
                desired_threads, chunk_size, sequential_benchmark, parallel_benchmark
            );
            Ok(Self::Parallel { pool, chunk_size })
        } else {
            println!(
                "optimizer mode: sequential (parallel {:?} was not faster than sequential {:?})",
                parallel_benchmark, sequential_benchmark
            );
            Ok(Self::Sequential)
        }
    }

    fn train_batch(
        &self,
        online: &mut Network,
        target_network: &Network,
        batch: &[Transition],
        gamma: f32,
        learning_rate: f32,
        huber_delta: f32,
        gradient_clip_norm: f32,
    ) -> f32 {
        match self {
            Self::Sequential => online.train_batch(
                target_network,
                batch,
                gamma,
                learning_rate,
                huber_delta,
                gradient_clip_norm,
            ),
            Self::Parallel { pool, chunk_size } => pool.install(|| {
                online.train_batch_parallel(
                    target_network,
                    batch,
                    gamma,
                    learning_rate,
                    huber_delta,
                    gradient_clip_norm,
                    *chunk_size,
                )
            }),
        }
    }
}

pub fn default_training_seeds(start: u64, count: usize) -> Vec<u64> {
    (start..start + count as u64).collect()
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingProgress {
    pub episode: usize,
    pub total_steps: usize,
    pub epsilon: f32,
    pub last_return: f32,
    pub last_survival: f32,
    pub last_evades: u32,
    pub avg_survival: f32,
    pub global_best_survival: f32,
    pub min_survival: f32,
    pub avg_return: f32,
    pub avg_evades: f32,
    pub min_return: f32,
    pub timeouts: u32,
    pub loss: f32,
    pub steps_per_second: f32,
    pub mean_predicted_q: f32,
    pub mean_target_q: f32,
    pub mean_abs_td_error: f32,
    pub terminal_fraction: f32,
}

fn batch_diagnostics(
    online: &Network,
    target_network: &Network,
    batch: &[Transition],
    gamma: f32,
) -> (f32, f32, f32, f32) {
    if batch.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut predicted_q_sum = 0.0;
    let mut target_q_sum = 0.0;
    let mut abs_td_error_sum = 0.0;
    let mut terminal_count = 0.0;

    for transition in batch {
        let predicted = online.predict(&transition.state);
        let next_q_values = if transition.done {
            vec![0.0; predicted.len()]
        } else {
            target_network.predict(&transition.next_state)
        };
        let next_best = next_q_values.into_iter().fold(f32::NEG_INFINITY, f32::max);
        let target = transition.reward
            + if transition.done {
                0.0
            } else {
                gamma * next_best
            };
        let prediction = predicted[transition.action];

        predicted_q_sum += prediction;
        target_q_sum += target;
        abs_td_error_sum += (prediction - target).abs();
        terminal_count += if transition.done { 1.0 } else { 0.0 };
    }

    let denom = batch.len() as f32;
    (
        predicted_q_sum / denom,
        target_q_sum / denom,
        abs_td_error_sum / denom,
        terminal_count / denom,
    )
}

struct TrainExecutionOutcome {
    result: TrainingResult,
    profile_stats: ProfileStats,
    total_steps_completed: usize,
}

fn profile_enabled_from_env() -> bool {
    env::var("DQN_PROFILE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

pub fn run_benchmark(
    config: TrainingConfig,
    output_dir: &Path,
    mode: BenchmarkMode,
    simulated_survival_seconds: f32,
) -> anyhow::Result<BenchmarkReport> {
    let execution_mode = match mode {
        BenchmarkMode::FullTraining => ExecutionMode::RealGame,
        BenchmarkMode::SimulatedSurvival => ExecutionMode::SimulatedSurvival {
            survival_time: simulated_survival_seconds,
        },
    };
    let outcome = train_internal(
        config,
        output_dir,
        None,
        None,
        None,
        execution_mode,
        true,
    )?;
    let report_path = output_dir.join("benchmark_report.json");
    let report = BenchmarkReport {
        mode,
        output_dir: output_dir.to_path_buf(),
        report_path: report_path.clone(),
        simulated_survival_seconds,
        episodes_completed: outcome.result.completed_episodes,
        total_steps_completed: outcome.total_steps_completed,
        best_metrics: outcome.result.best_metrics,
        profile: outcome.profile_stats.snapshot(),
    };
    let json = serde_json::to_string_pretty(&report).context("failed to serialize benchmark")?;
    fs::write(&report_path, json)
        .with_context(|| format!("failed to write {}", report_path.display()))?;
    Ok(report)
}

pub fn train(
    config: TrainingConfig,
    output_dir: &Path,
    resume_model: Option<SavedModel>,
    progress_tx: Option<mpsc::UnboundedSender<TrainingProgress>>,
    stop_signal: Option<Arc<AtomicBool>>,
) -> anyhow::Result<TrainingResult> {
    Ok(train_internal(
        config,
        output_dir,
        resume_model,
        progress_tx,
        stop_signal,
        ExecutionMode::RealGame,
        profile_enabled_from_env(),
    )?
    .result)
}

fn train_internal(
    config: TrainingConfig,
    output_dir: &Path,
    resume_model: Option<SavedModel>,
    progress_tx: Option<mpsc::UnboundedSender<TrainingProgress>>,
    stop_signal: Option<Arc<AtomicBool>>,
    execution_mode: ExecutionMode,
    profile_enabled: bool,
) -> anyhow::Result<TrainExecutionOutcome> {
    let total_wall_start = Instant::now();
    let mut profile_stats = ProfileStats::default();

    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let mut rng = ChaCha8Rng::seed_from_u64(config.trainer_seed);
    let resume_state = resume_model
        .map(|model| validate_resume_model(&config, model))
        .transpose()?;
    let input_size = config.model_type.input_size();
    let mut online = if let Some(resume) = &resume_state {
        resume.network.clone()
    } else {
        let mut sizes = vec![input_size];
        sizes.extend(config.hidden_sizes.iter().copied());
        sizes.push(Action::ALL.len());
        Network::new(&sizes, &mut rng)
    };
    let mut target = online.clone();
    let mut replay = ReplayBuffer::new(config.replay_capacity);

    let optimizer_runtime_start = Instant::now();
    let optimization_runtime = OptimizationRuntime::choose(
        &online,
        &target,
        config.batch_size,
        config.gamma,
        config.learning_rate,
        config.huber_delta,
        config.gradient_clip_norm,
        input_size,
    )?;
    profile_stats.optimization_runtime_choice += optimizer_runtime_start.elapsed();

    let eval_runtime_start = Instant::now();
    let evaluation_runtime = EvaluationRuntime::choose(
        &online,
        &config.fixed_training_seeds,
        config.action_repeat,
        config.model_type,
        config.map_design,
        execution_mode,
    )?;
    profile_stats.evaluation_runtime_choice += eval_runtime_start.elapsed();

    let mut best_metrics = resume_state
        .as_ref()
        .map(|resume| resume.best_metrics.clone())
        .unwrap_or_default();
    let mut total_steps = resume_state
        .as_ref()
        .map(|resume| resume.total_steps)
        .unwrap_or(0);
    let starting_episode = resume_state
        .as_ref()
        .map(|resume| resume.completed_episodes)
        .unwrap_or(0);
    let mut episode_schedule = Vec::<u64>::new();
    let mut focus_training_seeds = Vec::<u64>::new();
    let mut cycle_index = 0usize;
    let mut global_best_survival = best_metrics.average_survival_time;

    let mut last_progress_time = Instant::now();
    let mut last_progress_steps = total_steps;
    let collect_diagnostics = progress_tx.is_some();

    for episode in 0..config.episodes {
        if let Some(stop) = &stop_signal {
            if stop.load(AtomicOrdering::SeqCst) {
                println!(
                    "training stopped by signal at episode {}",
                    starting_episode + episode
                );
                break;
            }
        }

        if cycle_index >= episode_schedule.len() {
            episode_schedule = episode_seed_cycle(
                &config.fixed_training_seeds,
                &focus_training_seeds,
                config.random_seed_count_per_cycle,
                &mut rng,
            );
            cycle_index = 0;
        }
        let seed = episode_schedule[cycle_index];
        cycle_index += 1;

        let metrics = run_training_episode(
            execution_mode,
            &config,
            seed,
            &mut rng,
            &mut online,
            &mut target,
            &mut replay,
            &optimization_runtime,
            &mut total_steps,
            &mut global_best_survival,
            &mut profile_stats,
            collect_diagnostics,
        );

        let eval_start = Instant::now();
        let eval = evaluate_network_with_results(
            &online,
            &config.fixed_training_seeds,
            config.action_repeat,
            &evaluation_runtime,
            config.model_type,
            config.map_design,
            execution_mode,
        );
        focus_training_seeds = match config.seed_focus_mode {
            SeedFocusMode::Original => Vec::new(),
            SeedFocusMode::BadSeeds => {
                select_focus_training_seeds(&config.fixed_training_seeds, &eval.seed_results)
            }
        };
        profile_stats.evaluation += eval_start.elapsed();
        profile_stats.evaluations += 1;

        if is_better_metrics(&eval.summary, &best_metrics, config.seed_focus_mode) {
            best_metrics = eval.summary.clone();
            let save_start = Instant::now();
            save_model(
                output_dir.join("best_model.json"),
                SavedModel::new(
                    config.model_type,
                    config.hidden_sizes.clone(),
                    config.fixed_training_seeds.clone(),
                    config.random_seed_count_per_cycle,
                    config.action_repeat,
                    starting_episode + episode + 1,
                    total_steps,
                    best_metrics.clone(),
                    online.layers.clone(),
                ),
            )?;
            profile_stats.serialization += save_start.elapsed();
            profile_stats.saves += 1;
        }

        let mean_loss = if metrics.losses.is_empty() {
            0.0
        } else {
            metrics.losses.iter().sum::<f32>() / metrics.losses.len() as f32
        };
        println!(
            "ep {:>5}  seed {:>10}  steps {:>7}  eps {:>4.2}  return {:>7.2}  survive {:>5.2}s  evades {:>4}  eval_to {:>2}/{}  eval_min {:>5.2}s  loss {:>7.4}",
            starting_episode + episode + 1,
            seed,
            total_steps,
            epsilon_for_step(&config, total_steps),
            metrics.total_return,
            metrics.survival_time,
            metrics.evades,
            eval.summary.timeouts,
            config.fixed_training_seeds.len(),
            eval.summary.min_survival_time,
            mean_loss,
        );

        if let Some(tx) = &progress_tx {
            let now = Instant::now();
            let elapsed = now.duration_since(last_progress_time).as_secs_f32();
            let steps_done = total_steps - last_progress_steps;
            let sps = if elapsed > 0.0 {
                steps_done as f32 / elapsed
            } else {
                0.0
            };

            last_progress_time = now;
            last_progress_steps = total_steps;

            let _ = tx.send(TrainingProgress {
                episode: starting_episode + episode + 1,
                total_steps,
                epsilon: epsilon_for_step(&config, total_steps),
                last_return: metrics.total_return,
                last_survival: metrics.survival_time,
                last_evades: metrics.evades,
                avg_survival: eval.summary.average_survival_time,
                global_best_survival,
                min_survival: eval.summary.min_survival_time,
                avg_return: eval.summary.average_return,
                avg_evades: eval.summary.average_evades,
                min_return: eval.summary.min_return,
                timeouts: eval.summary.timeouts,
                loss: mean_loss,
                steps_per_second: sps,
                mean_predicted_q: metrics.diagnostics.mean_predicted_q,
                mean_target_q: metrics.diagnostics.mean_target_q,
                mean_abs_td_error: metrics.diagnostics.mean_abs_td_error,
                terminal_fraction: metrics.diagnostics.terminal_fraction,
            });
        }

        if (episode + 1) % config.checkpoint_every == 0 {
            let save_start = Instant::now();
            let saved_model = SavedModel::new(
                config.model_type,
                config.hidden_sizes.clone(),
                config.fixed_training_seeds.clone(),
                config.random_seed_count_per_cycle,
                config.action_repeat,
                starting_episode + episode + 1,
                total_steps,
                best_metrics.clone(),
                online.layers.clone(),
            );

            save_model(
                output_dir.join(format!(
                    "checkpoint_ep_{:05}.json",
                    starting_episode + episode + 1
                )),
                saved_model.clone(),
            )?;
            let _ = save_model(output_dir.join("latest_model.json"), saved_model);

            profile_stats.serialization += save_start.elapsed();
            profile_stats.saves += 1;
        }
    }

    let completed_episodes = starting_episode + config.episodes;
    let final_save_start = Instant::now();
    save_model(
        output_dir.join("final_model.json"),
        SavedModel::new(
            config.model_type,
            config.hidden_sizes.clone(),
            config.fixed_training_seeds.clone(),
            config.random_seed_count_per_cycle,
            config.action_repeat,
            completed_episodes,
            total_steps,
            best_metrics.clone(),
            online.layers.clone(),
        ),
    )?;
    profile_stats.serialization += final_save_start.elapsed();
    profile_stats.saves += 1;
    profile_stats.total_wall = total_wall_start.elapsed();

    if profile_enabled {
        profile_stats.print();
    }

    Ok(TrainExecutionOutcome {
        result: TrainingResult {
            best_metrics,
            completed_episodes,
        },
        profile_stats,
        total_steps_completed: total_steps,
    })
}

fn run_training_episode(
    execution_mode: ExecutionMode,
    config: &TrainingConfig,
    seed: u64,
    rng: &mut ChaCha8Rng,
    online: &mut Network,
    target: &mut Network,
    replay: &mut ReplayBuffer,
    optimization_runtime: &OptimizationRuntime,
    total_steps: &mut usize,
    global_best_survival: &mut f32,
    profile_stats: &mut ProfileStats,
    collect_diagnostics: bool,
) -> EpisodeLoopMetrics {
    match execution_mode {
        ExecutionMode::RealGame => run_real_training_episode(
            config,
            seed,
            rng,
            online,
            target,
            replay,
            optimization_runtime,
            total_steps,
            global_best_survival,
            profile_stats,
            collect_diagnostics,
        ),
        ExecutionMode::SimulatedSurvival { survival_time } => run_simulated_training_episode(
            config,
            seed,
            survival_time,
            rng,
            online,
            target,
            replay,
            optimization_runtime,
            total_steps,
            global_best_survival,
            profile_stats,
            collect_diagnostics,
        ),
    }
}

fn run_real_training_episode(
    config: &TrainingConfig,
    seed: u64,
    rng: &mut ChaCha8Rng,
    online: &mut Network,
    target: &mut Network,
    replay: &mut ReplayBuffer,
    optimization_runtime: &OptimizationRuntime,
    total_steps: &mut usize,
    global_best_survival: &mut f32,
    profile_stats: &mut ProfileStats,
    collect_diagnostics: bool,
) -> EpisodeLoopMetrics {
    let mut env = GameState::new(env_config_for_training(config.map_design), Some(seed));
    let mut observation = ObsBuilderState::new(config.model_type);
    observation.reset(&env);
    let mut current_state = observation.build_vec(&env);
    let mut metrics = EpisodeLoopMetrics::default();

    while !env.done {
        let env_policy_start = Instant::now();
        let state = current_state.clone();
        let action_start = Instant::now();
        let epsilon = epsilon_for_step(config, *total_steps);
        let action_index = select_action(online, &state, epsilon, rng);
        let action = Action::ALL[action_index];
        profile_stats.action_selection += action_start.elapsed();

        let env_step_start = Instant::now();
        let mut reward = 0.0;
        for _ in 0..config.action_repeat {
            if env.done {
                break;
            }
            let result = env.step_fixed(action);
            reward += result.reward;
        }
        profile_stats.environment_step += env_step_start.elapsed();

        let observation_start = Instant::now();
        let next_state = observation.build_vec(&env);
        profile_stats.observation_build += observation_start.elapsed();
        current_state = next_state.clone();
        profile_stats.environment_and_policy += env_policy_start.elapsed();

        metrics.total_return += reward;
        metrics.evades = env.enemies_evaded;
        metrics.survival_time = env.elapsed_time;
        *global_best_survival = (*global_best_survival).max(env.elapsed_time);

        process_transition(
            config,
            replay,
            state,
            action_index,
            reward,
            next_state,
            env.done,
            online,
            target,
            optimization_runtime,
            rng,
            total_steps,
            profile_stats,
            &mut metrics,
            collect_diagnostics,
        );
    }

    metrics
}

fn run_simulated_training_episode(
    config: &TrainingConfig,
    seed: u64,
    survival_time: f32,
    rng: &mut ChaCha8Rng,
    online: &mut Network,
    target: &mut Network,
    replay: &mut ReplayBuffer,
    optimization_runtime: &OptimizationRuntime,
    total_steps: &mut usize,
    global_best_survival: &mut f32,
    profile_stats: &mut ProfileStats,
    collect_diagnostics: bool,
) -> EpisodeLoopMetrics {
    let mut synthetic = SyntheticEpisode::new(
        env_config_for_training(config.map_design),
        seed,
        config.model_type,
        survival_time,
        config.action_repeat,
    );
    let mut metrics = EpisodeLoopMetrics::default();

    loop {
        let env_policy_start = Instant::now();
        let state = synthetic.current_state().to_vec();
        let action_start = Instant::now();
        let epsilon = epsilon_for_step(config, *total_steps);
        let action_index = select_action(online, &state, epsilon, rng);
        let action = Action::ALL[action_index];
        profile_stats.action_selection += action_start.elapsed();
        let env_step_start = Instant::now();
        let step = synthetic.step(seed, action, config.action_repeat, config.model_type.input_size());
        profile_stats.environment_step += env_step_start.elapsed();
        let observation_start = Instant::now();
        let next_state = synthetic.current_state().to_vec();
        profile_stats.observation_build += observation_start.elapsed();
        profile_stats.environment_and_policy += env_policy_start.elapsed();

        metrics.total_return += step.reward;
        metrics.survival_time = step.elapsed_time;
        *global_best_survival = (*global_best_survival).max(step.elapsed_time);

        process_transition(
            config,
            replay,
            state,
            action_index,
            step.reward,
            next_state,
            step.done,
            online,
            target,
            optimization_runtime,
            rng,
            total_steps,
            profile_stats,
            &mut metrics,
            collect_diagnostics,
        );

        if step.done {
            break;
        }
    }

    metrics
}

#[allow(clippy::too_many_arguments)]
fn process_transition(
    config: &TrainingConfig,
    replay: &mut ReplayBuffer,
    state: Vec<f32>,
    action_index: usize,
    reward: f32,
    next_state: Vec<f32>,
    done: bool,
    online: &mut Network,
    target: &mut Network,
    optimization_runtime: &OptimizationRuntime,
    rng: &mut ChaCha8Rng,
    total_steps: &mut usize,
    profile_stats: &mut ProfileStats,
    metrics: &mut EpisodeLoopMetrics,
    collect_diagnostics: bool,
) {
    replay.push(Transition {
        state,
        action: action_index,
        reward,
        next_state,
        done,
    });
    *total_steps += 1;
    profile_stats.steps += 1;

    if replay.len() >= config.warmup_steps && *total_steps % config.train_every == 0 {
        let optimize_start = Instant::now();
        let sample_start = Instant::now();
        let batch_refs = replay.sample(config.batch_size.min(replay.len()), rng);
        profile_stats.replay_sampling += sample_start.elapsed();
        let batch = batch_refs.into_iter().cloned().collect::<Vec<_>>();
        if collect_diagnostics {
            let diagnostics_start = Instant::now();
            let (mean_predicted_q, mean_target_q, mean_abs_td_error, terminal_fraction) =
                batch_diagnostics(online, target, &batch, config.gamma);
            profile_stats.diagnostics += diagnostics_start.elapsed();
            metrics.diagnostics = StepDiagnostics {
                mean_predicted_q,
                mean_target_q,
                mean_abs_td_error,
                terminal_fraction,
            };
        }
        let train_start = Instant::now();
        let loss = optimization_runtime.train_batch(
            online,
            target,
            &batch,
            config.gamma,
            config.learning_rate,
            config.huber_delta,
            config.gradient_clip_norm,
        );
        profile_stats.train_batch += train_start.elapsed();
        profile_stats.optimization += optimize_start.elapsed();
        profile_stats.train_updates += 1;
        metrics.losses.push(loss);
    }

    if replay.len() >= config.warmup_steps && *total_steps % config.target_sync_interval == 0 {
        let sync_start = Instant::now();
        *target = online.clone();
        let sync_elapsed = sync_start.elapsed();
        profile_stats.target_sync += sync_elapsed;
        profile_stats.optimization += sync_elapsed;
    }
}

pub fn evaluate_saved_model(model: &SavedModel, seeds: &[u64]) -> EvaluationSummary {
    let model_type = match model.model_type.as_str() {
        "dqn2" => ModelType::Dqn2,
        _ => ModelType::Dqn,
    };
    let network = Network::from_layers(model.layers.clone());
    evaluate_network_with_results(
        &network,
        seeds,
        model.action_repeat,
        &EvaluationRuntime::Sequential,
        model_type,
        MapDesign::Open,
        ExecutionMode::RealGame,
    )
    .summary
}

fn save_model(path: impl AsRef<Path>, model: SavedModel) -> anyhow::Result<()> {
    let path = path.as_ref();
    let json = serde_json::to_string_pretty(&model).context("failed to serialize model")?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}

fn validate_resume_model(
    config: &TrainingConfig,
    mut model: SavedModel,
) -> anyhow::Result<ResumeState> {
    if model.output_size != Action::ALL.len() {
        anyhow::bail!(
            "resume model output size {} does not match expected {}",
            model.output_size,
            Action::ALL.len()
        );
    }
    if model.hidden_sizes != config.hidden_sizes {
        anyhow::bail!(
            "resume model hidden sizes {:?} do not match configured {:?}",
            model.hidden_sizes,
            config.hidden_sizes
        );
    }

    let expected_input = config.model_type.input_size();

    // Allow cross-type resume by zero-expanding the first layer weights.
    if model.input_size != expected_input {
        let first = model
            .layers
            .first_mut()
            .ok_or_else(|| anyhow::anyhow!("resume model has no layers"))?;
        let old_in = first.input_size;
        let out = first.output_size;

        if expected_input > old_in {
            // Expand: insert zero-weight columns for the new inputs.
            let extra = expected_input - old_in;
            let mut new_weights = Vec::with_capacity(out * expected_input);
            for out_idx in 0..out {
                // Original columns
                let row_start = out_idx * old_in;
                new_weights.extend_from_slice(&first.weights[row_start..row_start + old_in]);
                // Zero-padded new columns
                new_weights.extend(std::iter::repeat(0.0f32).take(extra));
            }
            first.weights = new_weights;
            first.input_size = expected_input;
        } else {
            // Shrink: drop the extra input columns.
            let mut new_weights = Vec::with_capacity(out * expected_input);
            for out_idx in 0..out {
                let row_start = out_idx * old_in;
                new_weights
                    .extend_from_slice(&first.weights[row_start..row_start + expected_input]);
            }
            first.weights = new_weights;
            first.input_size = expected_input;
        }
        model.input_size = expected_input;

        println!(
            "cross-type resume: adjusted input layer {} -> {} inputs",
            old_in, expected_input
        );
    }

    Ok(ResumeState {
        network: Network::from_layers(model.layers),
        best_metrics: model.best_metrics,
        completed_episodes: model.episodes_completed,
        total_steps: model.total_steps_completed,
    })
}

fn episode_seed_cycle(
    fixed_seeds: &[u64],
    focus_seeds: &[u64],
    random_count: usize,
    rng: &mut impl Rng,
) -> Vec<u64> {
    let mut seeds = fixed_seeds.to_vec();
    seeds.extend(focus_seeds.iter().copied());
    while seeds.len() < fixed_seeds.len() + focus_seeds.len() + random_count {
        let candidate = rng.gen::<u64>();
        if !seeds.contains(&candidate) {
            seeds.push(candidate);
        }
    }
    seeds.shuffle(rng);
    seeds
}

fn build_optimizer_benchmark_batch(batch_size: usize, input_size: usize) -> Vec<Transition> {
    (0..batch_size.max(1))
        .map(|index| {
            let seed = index as f32 * 0.013;
            let mut state = vec![0.0; input_size];
            let mut next_state = vec![0.0; input_size];
            for (offset, value) in state.iter_mut().enumerate() {
                *value = ((offset as f32 * 0.031) + seed).sin();
            }
            for (offset, value) in next_state.iter_mut().enumerate() {
                *value = ((offset as f32 * 0.029) + seed).cos();
            }
            Transition {
                state,
                action: index % Action::ALL.len(),
                reward: (index % 7) as f32 * 0.1 - 0.3,
                next_state,
                done: index % 11 == 0,
            }
        })
        .collect()
}

fn benchmark_runtime(mut run_once: impl FnMut(), rounds: usize) -> Duration {
    let start = Instant::now();
    for _ in 0..rounds {
        run_once();
    }
    start.elapsed()
}

fn benchmark_evaluation(
    mut evaluate_once: impl FnMut() -> Vec<SeedEpisodeSummary>,
    rounds: usize,
) -> Duration {
    let start = Instant::now();
    for _ in 0..rounds {
        let _ = evaluate_once();
    }
    start.elapsed()
}

fn synthetic_total_steps(config: &GameConfig, survival_time: f32, action_repeat: usize) -> usize {
    let step_duration = (config.fixed_timestep * action_repeat.max(1) as f32).max(f32::EPSILON);
    (survival_time.max(step_duration) / step_duration).ceil() as usize
}

fn synthetic_observation(
    seed: u64,
    step_index: usize,
    action: Action,
    input_size: usize,
) -> Vec<f32> {
    let mut observation = vec![0.0; input_size];
    let time = step_index as f32 * 0.017 + seed as f32 * 0.000_013;
    let (dir_x, dir_y) = action.vector();
    let velocity = rust_evades::game::Vec2 { x: dir_x, y: dir_y }.normalized_or_zero();

    for (index, value) in observation.iter_mut().enumerate() {
        let phase = time + index as f32 * 0.043;
        let signal = if index % 2 == 0 {
            phase.sin()
        } else {
            phase.cos()
        };
        *value = signal.clamp(-1.0, 1.0);
    }

    if input_size >= 2 {
        observation[input_size - 2] = velocity.x;
        observation[input_size - 1] = velocity.y;
    }

    observation
}

fn epsilon_for_step(config: &TrainingConfig, step: usize) -> f32 {
    let progress = (step as f32 / config.epsilon_decay_steps.max(1) as f32).clamp(0.0, 1.0);
    config.epsilon_start + (config.epsilon_end - config.epsilon_start) * progress
}

fn select_action(network: &Network, state: &[f32], epsilon: f32, rng: &mut impl Rng) -> usize {
    if rng.gen::<f32>() < epsilon {
        rng.gen_range(0..Action::ALL.len())
    } else {
        greedy_action(network, state)
    }
}

fn greedy_action(network: &Network, state: &[f32]) -> usize {
    network
        .predict(state)
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn is_better_metrics(
    candidate: &EvaluationSummary,
    best: &EvaluationSummary,
    focus_mode: SeedFocusMode,
) -> bool {
    match focus_mode {
        SeedFocusMode::Original => {
            candidate.timeouts > best.timeouts
                || (candidate.timeouts == best.timeouts
                    && candidate.average_survival_time > best.average_survival_time)
                || (candidate.timeouts == best.timeouts
                    && (candidate.average_survival_time - best.average_survival_time).abs()
                        < 1.0e-5
                    && candidate.average_return > best.average_return)
        }
        SeedFocusMode::BadSeeds => {
            candidate.timeouts > best.timeouts
                || (candidate.timeouts == best.timeouts
                    && candidate.min_survival_time > best.min_survival_time)
                || (candidate.timeouts == best.timeouts
                    && (candidate.min_survival_time - best.min_survival_time).abs() < 1.0e-5
                    && candidate.average_survival_time > best.average_survival_time)
                || (candidate.timeouts == best.timeouts
                    && (candidate.min_survival_time - best.min_survival_time).abs() < 1.0e-5
                    && (candidate.average_survival_time - best.average_survival_time).abs()
                        < 1.0e-5
                    && candidate.min_return > best.min_return)
                || (candidate.timeouts == best.timeouts
                    && (candidate.min_survival_time - best.min_survival_time).abs() < 1.0e-5
                    && (candidate.average_survival_time - best.average_survival_time).abs()
                        < 1.0e-5
                    && (candidate.min_return - best.min_return).abs() < 1.0e-5
                    && candidate.average_return > best.average_return)
        }
    }
}

#[derive(Clone, Debug)]
struct EvaluationOutcome {
    summary: EvaluationSummary,
    seed_results: Vec<SeedEpisodeSummary>,
}

fn evaluate_network_with_results(
    network: &Network,
    seeds: &[u64],
    action_repeat: usize,
    runtime: &EvaluationRuntime,
    model_type: ModelType,
    map_design: MapDesign,
    execution_mode: ExecutionMode,
) -> EvaluationOutcome {
    let seed_results = runtime.evaluate_batch(
        network,
        seeds,
        action_repeat,
        model_type,
        map_design,
        execution_mode,
    );
    let summary = aggregate_seed_results(&seed_results);
    EvaluationOutcome {
        summary,
        seed_results,
    }
}

fn aggregate_seed_results(results: &[SeedEpisodeSummary]) -> EvaluationSummary {
    if results.is_empty() {
        return EvaluationSummary::default();
    }

    let count = results.len() as f32;
    EvaluationSummary {
        average_survival_time: results
            .iter()
            .map(|result| result.survival_time)
            .sum::<f32>()
            / count,
        average_return: results
            .iter()
            .map(|result| result.total_return)
            .sum::<f32>()
            / count,
        average_evades: results.iter().map(|result| result.evades).sum::<f32>() / count,
        min_survival_time: results
            .iter()
            .map(|result| result.survival_time)
            .fold(f32::INFINITY, f32::min),
        min_return: results
            .iter()
            .map(|result| result.total_return)
            .fold(f32::INFINITY, f32::min),
        timeouts: results.iter().filter(|result| result.timed_out).count() as u32,
    }
}

fn select_focus_training_seeds(seeds: &[u64], results: &[SeedEpisodeSummary]) -> Vec<u64> {
    if seeds.is_empty() || results.is_empty() {
        return Vec::new();
    }

    let focus_count = seeds.len().div_ceil(FOCUS_SEED_DIVISOR).min(results.len());
    let mut ranked = seeds
        .iter()
        .copied()
        .zip(results.iter().copied())
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| compare_seed_difficulty(&left.1, &right.1));
    ranked
        .into_iter()
        .take(focus_count.max(1))
        .map(|(seed, _)| seed)
        .collect()
}

fn compare_seed_difficulty(left: &SeedEpisodeSummary, right: &SeedEpisodeSummary) -> Ordering {
    left.timed_out
        .cmp(&right.timed_out)
        .then_with(|| left.survival_time.total_cmp(&right.survival_time))
        .then_with(|| left.total_return.total_cmp(&right.total_return))
        .then_with(|| left.evades.total_cmp(&right.evades))
}

fn evaluate_seed_batch_sequential(
    network: &Network,
    seeds: &[u64],
    action_repeat: usize,
    model_type: ModelType,
    map_design: MapDesign,
    execution_mode: ExecutionMode,
) -> Vec<SeedEpisodeSummary> {
    seeds
        .iter()
        .map(|&seed| {
            evaluate_seed(
                network,
                seed,
                action_repeat,
                model_type,
                map_design,
                execution_mode,
            )
        })
        .collect()
}

fn evaluate_seed_batch_parallel(
    network: &Network,
    seeds: &[u64],
    action_repeat: usize,
    model_type: ModelType,
    map_design: MapDesign,
    execution_mode: ExecutionMode,
) -> Vec<SeedEpisodeSummary> {
    seeds
        .par_iter()
        .map(|&seed| {
            evaluate_seed(
                network,
                seed,
                action_repeat,
                model_type,
                map_design,
                execution_mode,
            )
        })
        .collect()
}

fn evaluate_seed(
    network: &Network,
    seed: u64,
    action_repeat: usize,
    model_type: ModelType,
    map_design: MapDesign,
    execution_mode: ExecutionMode,
) -> SeedEpisodeSummary {
    match execution_mode {
        ExecutionMode::RealGame => {
            let mut env = GameState::new(env_config_for_training(map_design), Some(seed));
            let mut observation = ObsBuilderState::new(model_type);
            observation.reset(&env);
            let mut episode_return = 0.0;

            while !env.done {
                let state = observation.build_vec(&env);
                let action = Action::ALL[greedy_action(network, &state)];
                for _ in 0..action_repeat {
                    if env.done {
                        break;
                    }
                    let result = env.step_fixed(action);
                    episode_return += result.reward;
                }
            }

            let report = env.episode_report();
            SeedEpisodeSummary {
                total_return: episode_return,
                survival_time: report.elapsed_time,
                evades: report.enemies_evaded as f32,
                timed_out: report.survived_full_episode,
            }
        }
        ExecutionMode::SimulatedSurvival { survival_time } => {
            let config = env_config_for_training(map_design);
            let mut episode = SyntheticEpisode::new(
                config.clone(),
                seed,
                model_type,
                survival_time,
                action_repeat,
            );
            loop {
                let state = episode.current_state().to_vec();
                let action = Action::ALL[greedy_action(network, &state)];
                let step = episode.step(seed, action, action_repeat, model_type.input_size());
                if step.done {
                    let clamped_survival = survival_time.min(config.max_episode_time);
                    return SeedEpisodeSummary {
                        total_return: episode.total_return,
                        survival_time: clamped_survival,
                        evades: 0.0,
                        timed_out: true,
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_seed_schedule_matches_requested_range() {
        let seeds = default_training_seeds(2, 24);
        assert_eq!(seeds.len(), 24);
        assert_eq!(seeds[0], 2);
        assert_eq!(seeds[23], 25);
    }

    #[test]
    fn seed_cycle_keeps_fixed_and_adds_random() {
        let fixed = default_training_seeds(2, 24);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let focus = vec![fixed[0], fixed[1]];
        let seeds = episode_seed_cycle(&fixed, &focus, 2, &mut rng);
        assert_eq!(seeds.len(), 28);
        assert_eq!(seeds.iter().filter(|&&seed| seed == fixed[0]).count(), 2);
        assert_eq!(seeds.iter().filter(|&&seed| seed == fixed[1]).count(), 2);
        assert!(fixed.iter().all(|seed| seeds.contains(seed)));
    }

    #[test]
    fn timeout_heavier_metrics_rank_above_return_only() {
        let best = EvaluationSummary {
            average_survival_time: 8.0,
            average_return: 9.0,
            average_evades: 1.0,
            min_survival_time: 6.0,
            min_return: 7.0,
            timeouts: 0,
        };
        let candidate = EvaluationSummary {
            average_survival_time: 9.0,
            average_return: 8.0,
            average_evades: 1.0,
            min_survival_time: 8.5,
            min_return: 7.5,
            timeouts: 1,
        };
        assert!(is_better_metrics(
            &candidate,
            &best,
            SeedFocusMode::Original
        ));
    }

    #[test]
    fn worst_seed_survival_ranks_above_average_only() {
        let best = EvaluationSummary {
            average_survival_time: 10.0,
            average_return: 11.0,
            average_evades: 2.0,
            min_survival_time: 4.0,
            min_return: 5.0,
            timeouts: 0,
        };
        let candidate = EvaluationSummary {
            average_survival_time: 9.5,
            average_return: 10.5,
            average_evades: 2.0,
            min_survival_time: 6.0,
            min_return: 6.0,
            timeouts: 0,
        };
        assert!(is_better_metrics(
            &candidate,
            &best,
            SeedFocusMode::BadSeeds
        ));
    }

    #[test]
    fn original_mode_ignores_worst_seed_if_average_is_lower() {
        let best = EvaluationSummary {
            average_survival_time: 10.0,
            average_return: 11.0,
            average_evades: 2.0,
            min_survival_time: 4.0,
            min_return: 5.0,
            timeouts: 0,
        };
        let candidate = EvaluationSummary {
            average_survival_time: 9.5,
            average_return: 10.5,
            average_evades: 2.0,
            min_survival_time: 6.0,
            min_return: 6.0,
            timeouts: 0,
        };
        assert!(!is_better_metrics(
            &candidate,
            &best,
            SeedFocusMode::Original
        ));
    }

    #[test]
    fn focus_seed_selection_prefers_lowest_survival_runs() {
        let seeds = vec![11, 12, 13, 14];
        let results = vec![
            SeedEpisodeSummary {
                survival_time: 9.0,
                total_return: 9.0,
                timed_out: true,
                ..SeedEpisodeSummary::default()
            },
            SeedEpisodeSummary {
                survival_time: 2.0,
                total_return: 1.0,
                ..SeedEpisodeSummary::default()
            },
            SeedEpisodeSummary {
                survival_time: 3.0,
                total_return: 2.0,
                ..SeedEpisodeSummary::default()
            },
            SeedEpisodeSummary {
                survival_time: 8.0,
                total_return: 8.0,
                timed_out: true,
                ..SeedEpisodeSummary::default()
            },
        ];

        assert_eq!(select_focus_training_seeds(&seeds, &results), vec![12]);
    }

    #[test]
    fn resume_model_validation_keeps_progress_counters() {
        let config = TrainingConfig::default();
        let model = SavedModel::new(
            ModelType::Dqn,
            config.hidden_sizes.clone(),
            config.fixed_training_seeds.clone(),
            config.random_seed_count_per_cycle,
            config.action_repeat,
            42,
            1337,
            EvaluationSummary::default(),
            vec![],
        );

        let resume = validate_resume_model(&config, model).unwrap();
        assert_eq!(resume.completed_episodes, 42);
        assert_eq!(resume.total_steps, 1337);
    }

    #[test]
    fn resume_model_validation_rejects_hidden_size_mismatch() {
        let config = TrainingConfig::default();
        let model = SavedModel::new(
            ModelType::Dqn,
            vec![64, 64],
            config.fixed_training_seeds.clone(),
            config.random_seed_count_per_cycle,
            config.action_repeat,
            0,
            0,
            EvaluationSummary::default(),
            vec![],
        );

        assert!(validate_resume_model(&config, model).is_err());
    }

    #[test]
    fn resume_dqn2_model_validation_keeps_progress_counters() {
        let config = TrainingConfig {
            model_type: ModelType::Dqn2,
            ..TrainingConfig::default()
        };
        let model = SavedModel::new(
            ModelType::Dqn2,
            config.hidden_sizes.clone(),
            config.fixed_training_seeds.clone(),
            config.random_seed_count_per_cycle,
            config.action_repeat,
            10,
            500,
            EvaluationSummary::default(),
            vec![],
        );
        let resume = validate_resume_model(&config, model).unwrap();
        assert_eq!(resume.completed_episodes, 10);
        assert_eq!(resume.total_steps, 500);
    }

    #[test]
    fn training_loop_preserves_temporal_features() {
        let config = TrainingConfig::default();
        let mut env = GameState::new(GameConfig::default(), Some(42));
        let mut observation = ObsBuilderState::new(config.model_type);
        observation.reset(&env);

        // Emulate fixed loop setup
        let current_state = observation.build_vec(&env);

        // Take a step that guarantees some movement
        env.step_fixed(Action::ALL[1]);

        let state = current_state.clone();

        // Check delta of initial state is zero (vel_x and vel_y are at the end)
        assert_eq!(
            state[state.len() - 2],
            0.0,
            "initial x velocity should be 0"
        );
        assert_eq!(
            state[state.len() - 1],
            0.0,
            "initial y velocity should be 0"
        );

        // Build post-step state ONCE
        let next_state = observation.build_vec(&env);

        // Ensure post-step state HAS delta
        let n_vel_x = next_state[next_state.len() - 2];
        let n_vel_y = next_state[next_state.len() - 1];
        assert!(
            n_vel_x != 0.0 || n_vel_y != 0.0,
            "next_state should have non-zero velocity delta"
        );

        // Emulate fixed loop iteration 2: carry over next_state to state2
        let state2 = next_state.clone();

        // Verify state2 has the SAME delta as next_state (because it IS next_state)
        assert_eq!(
            state2[state2.len() - 2],
            n_vel_x,
            "state2 x vel should equal next_state x vel"
        );
        assert_eq!(
            state2[state2.len() - 1],
            n_vel_y,
            "state2 y vel should equal next_state y vel"
        );

        // Emulate BUGGY loop behavior: rebuilding state against an already-advanced builder
        let buggy_state2 = observation.build_vec(&env);
        assert_eq!(
            buggy_state2[buggy_state2.len() - 2],
            0.0,
            "buggy re-build zeroes x vel"
        );
        assert_eq!(
            buggy_state2[buggy_state2.len() - 1],
            0.0,
            "buggy re-build zeroes y vel"
        );
    }
}
