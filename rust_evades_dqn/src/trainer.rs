use std::{
    collections::VecDeque,
    env,
    fs,
    path::Path,
    thread,
    time::{Duration, Instant},
};

use anyhow::Context;
use rand::{prelude::*, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use rust_evades::{config::GameConfig, game::Action, game::GameState};

use crate::{
    model::{EvaluationSummary, SavedModel},
    network::Network,
    observation::{ObservationBuilder, INPUT_SIZE},
};

const EVAL_CPU_FRACTION: f32 = 0.75;
const MIN_PARALLEL_EVAL_SEEDS: usize = 8;

#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub episodes: usize,
    pub trainer_seed: u64,
    pub checkpoint_every: usize,
    pub fixed_training_seeds: Vec<u64>,
    pub random_seed_count_per_cycle: usize,
    pub hidden_sizes: Vec<usize>,
    pub replay_capacity: usize,
    pub batch_size: usize,
    pub warmup_steps: usize,
    pub train_every: usize,
    pub target_sync_interval: usize,
    pub learning_rate: f32,
    pub gamma: f32,
    pub epsilon_start: f32,
    pub epsilon_end: f32,
    pub epsilon_decay_steps: usize,
    pub action_repeat: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            episodes: 6000,
            trainer_seed: 7,
            checkpoint_every: 100,
            fixed_training_seeds: default_training_seeds(2, 24),
            random_seed_count_per_cycle: 2,
            hidden_sizes: vec![128, 128],
            replay_capacity: 100_000,
            batch_size: 128,
            warmup_steps: 2_000,
            train_every: 2,
            target_sync_interval: 1_000,
            learning_rate: 0.0003,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.03,
            epsilon_decay_steps: 120_000,
            action_repeat: 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingResult {
    pub best_metrics: EvaluationSummary,
    pub completed_episodes: usize,
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
struct ProfileStats {
    total_wall: Duration,
    evaluation_runtime_choice: Duration,
    environment_and_policy: Duration,
    optimization: Duration,
    evaluation: Duration,
    serialization: Duration,
    replay_sampling: Duration,
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
        println!(
            "  total: {:.3}s (100.0%)",
            self.total_wall.as_secs_f64()
        );
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
}

enum EvaluationRuntime {
    Sequential,
    Parallel { pool: ThreadPool },
}

impl EvaluationRuntime {
    fn choose(network: &Network, seeds: &[u64], action_repeat: usize) -> anyhow::Result<Self> {
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
            || evaluate_seed_batch_sequential(network, seeds, action_repeat),
            2,
        );
        let parallel_benchmark = benchmark_evaluation(
            || pool.install(|| evaluate_seed_batch_parallel(network, seeds, action_repeat)),
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
    ) -> Vec<SeedEpisodeSummary> {
        match self {
            Self::Sequential => evaluate_seed_batch_sequential(network, seeds, action_repeat),
            Self::Parallel { pool } => {
                pool.install(|| evaluate_seed_batch_parallel(network, seeds, action_repeat))
            }
        }
    }
}

pub fn default_training_seeds(start: u64, count: usize) -> Vec<u64> {
    (start..start + count as u64).collect()
}

pub fn train(
    config: TrainingConfig,
    output_dir: &Path,
    resume_model: Option<SavedModel>,
) -> anyhow::Result<TrainingResult> {
    let profile_enabled = env::var("DQN_PROFILE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);
    let total_wall_start = Instant::now();
    let mut profile_stats = ProfileStats::default();

    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let mut rng = ChaCha8Rng::seed_from_u64(config.trainer_seed);
    let resume_state = resume_model
        .map(|model| validate_resume_model(&config, model))
        .transpose()?;
    let mut online = if let Some(resume) = &resume_state {
        resume.network.clone()
    } else {
        let mut sizes = vec![INPUT_SIZE];
        sizes.extend(config.hidden_sizes.iter().copied());
        sizes.push(Action::ALL.len());
        Network::new(&sizes, &mut rng)
    };
    let mut target = online.clone();
    let mut replay = ReplayBuffer::new(config.replay_capacity);
    let eval_runtime_start = Instant::now();
    let evaluation_runtime =
        EvaluationRuntime::choose(&online, &config.fixed_training_seeds, config.action_repeat)?;
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
    let mut cycle_index = 0usize;

    for episode in 0..config.episodes {
        if cycle_index >= episode_schedule.len() {
            episode_schedule = episode_seed_cycle(
                &config.fixed_training_seeds,
                config.random_seed_count_per_cycle,
                &mut rng,
            );
            cycle_index = 0;
        }
        let seed = episode_schedule[cycle_index];
        cycle_index += 1;

        let mut env = GameState::new(GameConfig::default(), Some(seed));
        let mut observation = ObservationBuilder::default();
        observation.reset(&env);
        let mut episode_return = 0.0;
        let mut episode_evades = 0u32;
        let mut losses = Vec::new();

        while !env.done {
            let env_policy_start = Instant::now();
            let state = observation.build(&env).to_vec();
            let epsilon = epsilon_for_step(&config, total_steps);
            let action_index = select_action(&online, &state, epsilon, &mut rng);
            let action = Action::ALL[action_index];

            let mut reward = 0.0;
            for _ in 0..config.action_repeat {
                if env.done {
                    break;
                }
                let result = env.step_fixed(action);
                reward += result.reward;
            }

            let next_state = observation.build(&env).to_vec();
            profile_stats.environment_and_policy += env_policy_start.elapsed();
            let done = env.done;
            episode_return += reward;
            episode_evades = env.enemies_evaded;
            replay.push(Transition {
                state,
                action: action_index,
                reward,
                next_state,
                done,
            });
            total_steps += 1;
            profile_stats.steps += 1;

            if replay.len() >= config.warmup_steps && total_steps % config.train_every == 0 {
                let optimize_start = Instant::now();
                let sample_start = Instant::now();
                let batch_refs = replay.sample(config.batch_size.min(replay.len()), &mut rng);
                profile_stats.replay_sampling += sample_start.elapsed();
                let batch = batch_refs.into_iter().cloned().collect::<Vec<_>>();
                let train_start = Instant::now();
                let loss = online.train_batch(&target, &batch, config.gamma, config.learning_rate);
                profile_stats.train_batch += train_start.elapsed();
                profile_stats.optimization += optimize_start.elapsed();
                profile_stats.train_updates += 1;
                losses.push(loss);
            }

            if replay.len() >= config.warmup_steps && total_steps % config.target_sync_interval == 0
            {
                let sync_start = Instant::now();
                target = online.clone();
                let sync_elapsed = sync_start.elapsed();
                profile_stats.target_sync += sync_elapsed;
                profile_stats.optimization += sync_elapsed;
            }
        }

        let eval_start = Instant::now();
        let eval = evaluate_network(
            &online,
            &config.fixed_training_seeds,
            config.action_repeat,
            &evaluation_runtime,
        );
        profile_stats.evaluation += eval_start.elapsed();
        profile_stats.evaluations += 1;
        if is_better_metrics(&eval, &best_metrics) {
            best_metrics = eval.clone();
            let save_start = Instant::now();
            save_model(
                output_dir.join("best_model.json"),
                SavedModel::new(
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

        let mean_loss = if losses.is_empty() {
            0.0
        } else {
            losses.iter().sum::<f32>() / losses.len() as f32
        };
        println!(
            "ep {:>5}  seed {:>10}  steps {:>7}  eps {:>4.2}  return {:>7.2}  survive {:>5.2}s  evades {:>4}  eval_to {:>2}/{}  loss {:>7.4}",
            starting_episode + episode + 1,
            seed,
            total_steps,
            epsilon_for_step(&config, total_steps),
            episode_return,
            env.elapsed_time,
            episode_evades,
            eval.timeouts,
            config.fixed_training_seeds.len(),
            mean_loss,
        );

        if (episode + 1) % config.checkpoint_every == 0 {
            let save_start = Instant::now();
            save_model(
                output_dir.join(format!(
                    "checkpoint_ep_{:05}.json",
                    starting_episode + episode + 1
                )),
                SavedModel::new(
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
    }

    let final_save_start = Instant::now();
    save_model(
        output_dir.join("final_model.json"),
        SavedModel::new(
            config.hidden_sizes.clone(),
            config.fixed_training_seeds.clone(),
            config.random_seed_count_per_cycle,
            config.action_repeat,
            starting_episode + config.episodes,
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

    Ok(TrainingResult {
        best_metrics,
        completed_episodes: starting_episode + config.episodes,
    })
}

pub fn evaluate_saved_model(model: &SavedModel, seeds: &[u64]) -> EvaluationSummary {
    let network = Network::from_layers(model.layers.clone());
    aggregate_seed_results(&evaluate_seed_batch_sequential(
        &network,
        seeds,
        model.action_repeat,
    ))
}

fn save_model(path: impl AsRef<Path>, model: SavedModel) -> anyhow::Result<()> {
    let path = path.as_ref();
    let json = serde_json::to_string_pretty(&model).context("failed to serialize model")?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}

fn validate_resume_model(
    config: &TrainingConfig,
    model: SavedModel,
) -> anyhow::Result<ResumeState> {
    if model.model_type != "dqn" {
        anyhow::bail!("resume model type {} is not supported", model.model_type);
    }
    if model.input_size != INPUT_SIZE {
        anyhow::bail!(
            "resume model input size {} does not match expected {}",
            model.input_size,
            INPUT_SIZE
        );
    }
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

    Ok(ResumeState {
        network: Network::from_layers(model.layers),
        best_metrics: model.best_metrics,
        completed_episodes: model.episodes_completed,
        total_steps: model.total_steps_completed,
    })
}

fn episode_seed_cycle(fixed_seeds: &[u64], random_count: usize, rng: &mut impl Rng) -> Vec<u64> {
    let mut seeds = fixed_seeds.to_vec();
    while seeds.len() < fixed_seeds.len() + random_count {
        let candidate = rng.gen::<u64>();
        if !seeds.contains(&candidate) {
            seeds.push(candidate);
        }
    }
    seeds
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

fn is_better_metrics(candidate: &EvaluationSummary, best: &EvaluationSummary) -> bool {
    candidate.timeouts > best.timeouts
        || (candidate.timeouts == best.timeouts
            && candidate.average_survival_time > best.average_survival_time)
        || (candidate.timeouts == best.timeouts
            && (candidate.average_survival_time - best.average_survival_time).abs() < 1.0e-5
            && candidate.average_return > best.average_return)
}

fn evaluate_network(
    network: &Network,
    seeds: &[u64],
    action_repeat: usize,
    runtime: &EvaluationRuntime,
) -> EvaluationSummary {
    aggregate_seed_results(&runtime.evaluate_batch(network, seeds, action_repeat))
}

fn aggregate_seed_results(results: &[SeedEpisodeSummary]) -> EvaluationSummary {
    if results.is_empty() {
        return EvaluationSummary::default();
    }

    let count = results.len() as f32;
    EvaluationSummary {
        average_survival_time: results.iter().map(|result| result.survival_time).sum::<f32>()
            / count,
        average_return: results.iter().map(|result| result.total_return).sum::<f32>() / count,
        average_evades: results.iter().map(|result| result.evades).sum::<f32>() / count,
        timeouts: results.iter().filter(|result| result.timed_out).count() as u32,
    }
}

fn evaluate_seed_batch_sequential(
    network: &Network,
    seeds: &[u64],
    action_repeat: usize,
) -> Vec<SeedEpisodeSummary> {
    seeds.iter()
        .map(|&seed| evaluate_seed(network, seed, action_repeat))
        .collect()
}

fn evaluate_seed_batch_parallel(
    network: &Network,
    seeds: &[u64],
    action_repeat: usize,
) -> Vec<SeedEpisodeSummary> {
    seeds.par_iter()
        .map(|&seed| evaluate_seed(network, seed, action_repeat))
        .collect()
}

fn evaluate_seed(network: &Network, seed: u64, action_repeat: usize) -> SeedEpisodeSummary {
    let mut env = GameState::new(GameConfig::default(), Some(seed));
    let mut observation = ObservationBuilder::default();
    observation.reset(&env);
    let mut episode_return = 0.0;

    while !env.done {
        let state = observation.build(&env).to_vec();
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
        let seeds = episode_seed_cycle(&fixed, 2, &mut rng);
        assert_eq!(seeds.len(), 26);
        assert_eq!(&seeds[..24], fixed.as_slice());
    }

    #[test]
    fn timeout_heavier_metrics_rank_above_return_only() {
        let best = EvaluationSummary {
            average_survival_time: 8.0,
            average_return: 9.0,
            average_evades: 1.0,
            timeouts: 0,
        };
        let candidate = EvaluationSummary {
            average_survival_time: 9.0,
            average_return: 8.0,
            average_evades: 1.0,
            timeouts: 1,
        };
        assert!(is_better_metrics(&candidate, &best));
    }

    #[test]
    fn resume_model_validation_keeps_progress_counters() {
        let config = TrainingConfig::default();
        let model = SavedModel::new(
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
}
