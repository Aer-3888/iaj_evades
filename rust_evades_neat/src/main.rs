use std::{fs, path::PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};

use rust_evades_neat::{
    model::SavedModel,
    trainer::{default_training_seeds, evaluate_saved_model, train, TrainingConfig},
};

#[derive(Parser, Debug)]
#[command(name = "rust_evades_neat")]
#[command(about = "NEAT trainer for rust_evades using 36 raycasts, ray deltas, and x-delta input")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Train {
        #[arg(long, default_value = "training_runs/default")]
        output_dir: PathBuf,

        #[arg(long, default_value_t = 256)]
        population: usize,

        #[arg(long, default_value_t = 1500)]
        generations: usize,

        #[arg(long, default_value_t = 7)]
        trainer_seed: u64,

        #[arg(long, default_value_t = 2)]
        seed_start: u64,

        #[arg(long, default_value_t = 24)]
        seed_count: usize,

        #[arg(long, default_value_t = 4)]
        random_seed_count: usize,

        #[arg(long, default_value_t = 25)]
        checkpoint_every: usize,
    },
    Evaluate {
        #[arg(long)]
        model: PathBuf,

        #[arg(long, default_value_t = 2)]
        seed_start: u64,

        #[arg(long, default_value_t = 40)]
        seed_count: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Train {
            output_dir,
            population,
            generations,
            trainer_seed,
            seed_start,
            seed_count,
            random_seed_count,
            checkpoint_every,
        } => {
            let config = TrainingConfig {
                population_size: population,
                generations,
                trainer_seed,
                checkpoint_every,
                fixed_evaluation_seeds: default_training_seeds(seed_start, seed_count),
                random_seed_count_per_generation: random_seed_count,
                ..TrainingConfig::default()
            };

            let result = train(config, &output_dir)?;
            println!(
                "training complete after {} generations",
                result.completed_generations
            );
            println!("best fitness: {:.2}", result.best_metrics.fitness);
            println!(
                "best avg progress: {:.2}",
                result.best_metrics.average_progress
            );
            println!("best wins: {}", result.best_metrics.wins);
            println!(
                "saved final model to {}",
                output_dir.join("final_model.json").display()
            );
        }
        Command::Evaluate {
            model,
            seed_start,
            seed_count,
        } => {
            let json = fs::read_to_string(&model)
                .with_context(|| format!("failed to read {}", model.display()))?;
            let saved_model: SavedModel = serde_json::from_str(&json)
                .with_context(|| format!("failed to parse {}", model.display()))?;
            let seeds = default_training_seeds(seed_start, seed_count);
            let summary = evaluate_saved_model(&saved_model, &seeds);
            println!(
                "evaluation seeds: {}..{}",
                seed_start,
                seed_start + seed_count as u64 - 1
            );
            println!("fitness: {:.2}", summary.fitness);
            println!("avg progress: {:.2}", summary.average_progress);
            println!(
                "avg rightward reward: {:.2}",
                summary.average_rightward_reward
            );
            println!("wins: {}", summary.wins);
        }
    }

    Ok(())
}
