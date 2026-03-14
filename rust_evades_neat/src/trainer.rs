use std::{fs, path::Path};

use anyhow::Context;
use rand::{prelude::*, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rust_evades::{
    config::GameConfig,
    game::{Action, DoneReason, GameState},
};

use crate::{
    model::SavedModel,
    neat::{
        speciate, CompiledNetwork, EvaluationSummary, Genome, InnovationTracker, MutationConfig,
    },
    observation::{ObservationBuilder, INPUT_SIZE},
};

#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub population_size: usize,
    pub generations: usize,
    pub trainer_seed: u64,
    pub checkpoint_every: usize,
    pub evaluation_seeds: Vec<u64>,
    pub mutation: MutationConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            population_size: 256,
            generations: 1500,
            trainer_seed: 7,
            checkpoint_every: 25,
            evaluation_seeds: default_training_seeds(2, 40),
            mutation: MutationConfig::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingResult {
    pub best_genome: Genome,
    pub best_metrics: EvaluationSummary,
    pub completed_generations: usize,
}

pub fn default_training_seeds(start: u64, count: usize) -> Vec<u64> {
    (start..start + count as u64).collect()
}

pub fn train(config: TrainingConfig, output_dir: &Path) -> anyhow::Result<TrainingResult> {
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let mut rng = ChaCha8Rng::seed_from_u64(config.trainer_seed);
    let mut tracker = InnovationTracker::new(INPUT_SIZE, Action::ALL.len());
    let mut population = (0..config.population_size)
        .map(|_| tracker.initial_genome(&mut rng))
        .collect::<Vec<_>>();

    let mut best_genome = population[0].clone();
    let mut best_metrics = EvaluationSummary::default();

    for generation in 0..config.generations {
        let summaries = evaluate_population(&population, &config.evaluation_seeds);
        for (genome, summary) in population.iter_mut().zip(summaries.iter()) {
            genome.fitness = summary.fitness;
        }

        let species = speciate(&population, &config.mutation);
        let average_fitness =
            population.iter().map(|genome| genome.fitness).sum::<f32>() / population.len() as f32;
        let best_index = population
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.fitness.total_cmp(&right.1.fitness))
            .map(|(index, _)| index)
            .unwrap_or(0);

        if population[best_index].fitness >= best_metrics.fitness {
            best_genome = population[best_index].clone();
            best_metrics = summaries[best_index];
            save_model(
                output_dir.join("best_model.json"),
                SavedModel::new(
                    config.evaluation_seeds.clone(),
                    generation + 1,
                    best_metrics,
                    best_genome.clone(),
                ),
            )?;
        }

        println!(
            "gen {:>4}  species {:>3}  best {:>10.2}  avg {:>10.2}  progress {:>7.2}  wins {:>2}/{}  nodes {:>3}  conns {:>4}",
            generation + 1,
            species.len(),
            summaries[best_index].fitness,
            average_fitness,
            summaries[best_index].average_progress,
            summaries[best_index].wins,
            config.evaluation_seeds.len(),
            population[best_index].nodes.len(),
            population[best_index].connections.len(),
        );

        if (generation + 1) % config.checkpoint_every == 0 {
            save_model(
                output_dir.join(format!("checkpoint_gen_{:04}.json", generation + 1)),
                SavedModel::new(
                    config.evaluation_seeds.clone(),
                    generation + 1,
                    best_metrics,
                    best_genome.clone(),
                ),
            )?;
        }

        if generation + 1 == config.generations {
            break;
        }

        population = reproduce_population(
            &population,
            &species,
            &mut tracker,
            &config.mutation,
            &mut rng,
            config.population_size,
        );
    }

    save_model(
        output_dir.join("final_model.json"),
        SavedModel::new(
            config.evaluation_seeds.clone(),
            config.generations,
            best_metrics,
            best_genome.clone(),
        ),
    )?;

    Ok(TrainingResult {
        best_genome,
        best_metrics,
        completed_generations: config.generations,
    })
}

pub fn evaluate_saved_model(model: &SavedModel, seeds: &[u64]) -> EvaluationSummary {
    let network = model.genome.compile(INPUT_SIZE);
    evaluate_network(&network, seeds)
}

fn save_model(path: impl AsRef<Path>, model: SavedModel) -> anyhow::Result<()> {
    let path = path.as_ref();
    let json = serde_json::to_string_pretty(&model).context("failed to serialize model")?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}

fn evaluate_population(population: &[Genome], seeds: &[u64]) -> Vec<EvaluationSummary> {
    population
        .par_iter()
        .map(|genome| {
            let network = genome.compile(INPUT_SIZE);
            evaluate_network(&network, seeds)
        })
        .collect()
}

fn evaluate_network(network: &CompiledNetwork, seeds: &[u64]) -> EvaluationSummary {
    let mut total_fitness = 0.0;
    let mut total_progress = 0.0;
    let mut total_rightward_reward = 0.0;
    let mut wins = 0;

    for &seed in seeds {
        let config = GameConfig::default();
        let mut state = GameState::new(config, Some(seed));
        let mut observation = ObservationBuilder::default();
        observation.reset(&state);

        let mut rightward_reward = 0.0;
        while !state.done {
            let inputs = observation.build(&state);
            let outputs = network.activate(&inputs);
            let action = decode_action(&outputs);
            let previous_x = state.player.body.pos.x;
            state.step_fixed(action);
            let delta_x = state.player.body.pos.x - previous_x;
            if delta_x >= 0.0 {
                rightward_reward += delta_x;
            } else {
                rightward_reward += delta_x * 0.25;
            }
        }

        let report = state.episode_report();
        if report.done_reason == DoneReason::Goal {
            wins += 1;
        }

        let seed_fitness = rightward_reward * 2.5
            + report.best_progress * 4.0
            + if report.done_reason == DoneReason::Goal {
                2500.0
            } else {
                0.0
            }
            - report.elapsed_time * 2.0;
        total_fitness += seed_fitness;
        total_progress += report.best_progress;
        total_rightward_reward += rightward_reward;
    }

    let count = seeds.len().max(1) as f32;
    EvaluationSummary {
        fitness: total_fitness / count,
        average_progress: total_progress / count,
        average_rightward_reward: total_rightward_reward / count,
        wins,
    }
}

fn decode_action(outputs: &[f32]) -> Action {
    let index = outputs
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0);
    Action::ALL[index]
}

fn reproduce_population(
    population: &[Genome],
    species: &[crate::neat::Species],
    tracker: &mut InnovationTracker,
    mutation: &MutationConfig,
    rng: &mut impl Rng,
    population_size: usize,
) -> Vec<Genome> {
    let mut next_population = Vec::with_capacity(population_size);
    let total_adjusted = species
        .iter()
        .map(|species| {
            species
                .members
                .iter()
                .map(|&index| population[index].fitness / species.members.len().max(1) as f32)
                .sum::<f32>()
        })
        .sum::<f32>()
        .max(1.0);

    let mut offspring_targets = species
        .iter()
        .map(|species| {
            let adjusted = species
                .members
                .iter()
                .map(|&index| population[index].fitness / species.members.len().max(1) as f32)
                .sum::<f32>();
            ((adjusted / total_adjusted) * population_size as f32).round() as usize
        })
        .collect::<Vec<_>>();

    let mut assigned = offspring_targets.iter().sum::<usize>();
    while assigned < population_size {
        if let Some(target) = offspring_targets.choose_mut(rng) {
            *target += 1;
            assigned += 1;
        }
    }
    while assigned > population_size {
        if let Some(target) = offspring_targets
            .iter_mut()
            .find(|target| **target > mutation.elite_per_species)
        {
            *target -= 1;
            assigned -= 1;
        } else {
            break;
        }
    }

    for (species, &target_size) in species.iter().zip(offspring_targets.iter()) {
        if species.members.is_empty() || target_size == 0 {
            continue;
        }

        let mut ranked_members = species.members.clone();
        ranked_members.sort_by(|left, right| {
            population[*right]
                .fitness
                .total_cmp(&population[*left].fitness)
        });

        for &member_index in ranked_members
            .iter()
            .take(mutation.elite_per_species.min(target_size))
        {
            next_population.push(population[member_index].clone());
        }

        while next_population.len() < population_size
            && next_population
                .iter()
                .filter(|genome| {
                    species.members.iter().any(|index| {
                        genome.connections.len() == population[*index].connections.len()
                            && genome.nodes.len() == population[*index].nodes.len()
                    })
                })
                .count()
                < target_size
        {
            let parent_a =
                tournament_select(population, &ranked_members, mutation.tournament_size, rng);
            let mut child = if ranked_members.len() > 1
                && rng.gen::<f32>() < mutation.crossover_chance
                && rng.gen::<f32>() >= mutation.mutate_only_chance
            {
                let parent_b =
                    tournament_select(population, &ranked_members, mutation.tournament_size, rng);
                if parent_a.fitness >= parent_b.fitness {
                    Genome::crossover(parent_a, parent_b, rng)
                } else {
                    Genome::crossover(parent_b, parent_a, rng)
                }
            } else {
                parent_a.clone()
            };

            child.mutate(rng, tracker, mutation);
            child.fitness = 0.0;
            next_population.push(child);
        }
    }

    while next_population.len() < population_size {
        let mut clone = population
            .choose(rng)
            .cloned()
            .unwrap_or_else(|| population[0].clone());
        clone.mutate(rng, tracker, mutation);
        clone.fitness = 0.0;
        next_population.push(clone);
    }

    next_population.truncate(population_size);
    next_population
}

fn tournament_select<'a>(
    population: &'a [Genome],
    candidates: &[usize],
    tournament_size: usize,
    rng: &mut impl Rng,
) -> &'a Genome {
    let mut best_index = candidates[rng.gen_range(0..candidates.len())];
    for _ in 1..tournament_size.max(1) {
        let candidate = candidates[rng.gen_range(0..candidates.len())];
        if population[candidate].fitness > population[best_index].fitness {
            best_index = candidate;
        }
    }
    &population[best_index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_seed_schedule_matches_requested_range() {
        let seeds = default_training_seeds(2, 40);
        assert_eq!(seeds.len(), 40);
        assert_eq!(seeds[0], 2);
        assert_eq!(seeds[39], 41);
    }

    #[test]
    fn evaluation_is_deterministic_for_same_model() {
        let config = TrainingConfig {
            population_size: 4,
            generations: 1,
            trainer_seed: 3,
            checkpoint_every: 1,
            evaluation_seeds: default_training_seeds(2, 4),
            mutation: MutationConfig::default(),
        };
        let mut rng = ChaCha8Rng::seed_from_u64(config.trainer_seed);
        let mut tracker = InnovationTracker::new(INPUT_SIZE, Action::ALL.len());
        let genome = tracker.initial_genome(&mut rng);
        let model = SavedModel::new(
            config.evaluation_seeds.clone(),
            1,
            EvaluationSummary::default(),
            genome,
        );
        let left = evaluate_saved_model(&model, &config.evaluation_seeds);
        let right = evaluate_saved_model(&model, &config.evaluation_seeds);
        assert_eq!(left.fitness, right.fitness);
        assert_eq!(left.average_progress, right.average_progress);
    }
}
