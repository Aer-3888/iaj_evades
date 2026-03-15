use std::{collections::VecDeque, fs, path::Path};

use anyhow::Context;
use rand::{prelude::*, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rust_evades::{config::GameConfig, game::Action, game::GameState};

use crate::{
    model::{EvaluationSummary, SavedModel},
    network::Network,
    observation::{ObservationBuilder, INPUT_SIZE},
};

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

pub fn default_training_seeds(start: u64, count: usize) -> Vec<u64> {
    (start..start + count as u64).collect()
}

pub fn train(config: TrainingConfig, output_dir: &Path) -> anyhow::Result<TrainingResult> {
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let mut rng = ChaCha8Rng::seed_from_u64(config.trainer_seed);
    let mut sizes = vec![INPUT_SIZE];
    sizes.extend(config.hidden_sizes.iter().copied());
    sizes.push(Action::ALL.len());
    let mut online = Network::new(&sizes, &mut rng);
    let mut target = online.clone();
    let mut replay = ReplayBuffer::new(config.replay_capacity);

    let mut best_metrics = EvaluationSummary::default();
    let mut total_steps = 0usize;
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

            if replay.len() >= config.warmup_steps && total_steps % config.train_every == 0 {
                let batch_refs = replay.sample(config.batch_size.min(replay.len()), &mut rng);
                let batch = batch_refs.into_iter().cloned().collect::<Vec<_>>();
                let loss = online.train_batch(&target, &batch, config.gamma, config.learning_rate);
                losses.push(loss);
            }

            if replay.len() >= config.warmup_steps && total_steps % config.target_sync_interval == 0
            {
                target = online.clone();
            }
        }

        let eval = evaluate_network(&online, &config.fixed_training_seeds, config.action_repeat);
        if is_better_metrics(&eval, &best_metrics) {
            best_metrics = eval.clone();
            save_model(
                output_dir.join("best_model.json"),
                SavedModel::new(
                    config.hidden_sizes.clone(),
                    config.fixed_training_seeds.clone(),
                    config.random_seed_count_per_cycle,
                    config.action_repeat,
                    episode + 1,
                    best_metrics.clone(),
                    online.layers.clone(),
                ),
            )?;
        }

        let mean_loss = if losses.is_empty() {
            0.0
        } else {
            losses.iter().sum::<f32>() / losses.len() as f32
        };
        println!(
            "ep {:>5}  seed {:>10}  steps {:>7}  eps {:>4.2}  return {:>7.2}  survive {:>5.2}s  evades {:>4}  eval_to {:>2}/{}  loss {:>7.4}",
            episode + 1,
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
            save_model(
                output_dir.join(format!("checkpoint_ep_{:05}.json", episode + 1)),
                SavedModel::new(
                    config.hidden_sizes.clone(),
                    config.fixed_training_seeds.clone(),
                    config.random_seed_count_per_cycle,
                    config.action_repeat,
                    episode + 1,
                    best_metrics.clone(),
                    online.layers.clone(),
                ),
            )?;
        }
    }

    save_model(
        output_dir.join("final_model.json"),
        SavedModel::new(
            config.hidden_sizes.clone(),
            config.fixed_training_seeds.clone(),
            config.random_seed_count_per_cycle,
            config.action_repeat,
            config.episodes,
            best_metrics.clone(),
            online.layers.clone(),
        ),
    )?;

    Ok(TrainingResult {
        best_metrics,
        completed_episodes: config.episodes,
    })
}

pub fn evaluate_saved_model(model: &SavedModel, seeds: &[u64]) -> EvaluationSummary {
    let network = Network::from_layers(model.layers.clone());
    evaluate_network(&network, seeds, model.action_repeat)
}

fn save_model(path: impl AsRef<Path>, model: SavedModel) -> anyhow::Result<()> {
    let path = path.as_ref();
    let json = serde_json::to_string_pretty(&model).context("failed to serialize model")?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
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

fn evaluate_network(network: &Network, seeds: &[u64], action_repeat: usize) -> EvaluationSummary {
    let mut total_return = 0.0;
    let mut total_survival = 0.0;
    let mut total_evades = 0.0;
    let mut timeouts = 0;

    for &seed in seeds {
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
        total_return += episode_return;
        total_survival += report.elapsed_time;
        total_evades += report.enemies_evaded as f32;
        if report.survived_full_episode {
            timeouts += 1;
        }
    }

    let count = seeds.len().max(1) as f32;
    EvaluationSummary {
        average_survival_time: total_survival / count,
        average_return: total_return / count,
        average_evades: total_evades / count,
        timeouts,
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
}
