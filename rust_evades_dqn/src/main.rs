use std::{fs, path::PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};

use rust_evades_dqn::{
    model::SavedModel,
    trainer::{default_training_seeds, evaluate_saved_model, train, TrainingConfig},
};

#[derive(Parser, Debug)]
#[command(name = "rust_evades_dqn")]
#[command(about = "DQN trainer for rust_evades; this is the default training path")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Train {
        #[arg(long, default_value = "training_runs/dqn_default")]
        output_dir: PathBuf,

        #[arg(long, default_value_t = 6000)]
        episodes: usize,

        #[arg(long, default_value_t = 7)]
        trainer_seed: u64,

        #[arg(long, default_value_t = 2)]
        seed_start: u64,

        #[arg(long, default_value_t = 24)]
        seed_count: usize,

        #[arg(long, default_value_t = 2)]
        random_seed_count: usize,

        #[arg(long, default_value_t = 4)]
        action_repeat: usize,

        #[arg(long, default_value_t = 100)]
        checkpoint_every: usize,
    },
    Evaluate {
        #[arg(long)]
        model: PathBuf,

        #[arg(long, default_value_t = 2)]
        seed_start: u64,

        #[arg(long, default_value_t = 24)]
        seed_count: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Train {
            output_dir,
            episodes,
            trainer_seed,
            seed_start,
            seed_count,
            random_seed_count,
            action_repeat,
            checkpoint_every,
        } => {
            let config = TrainingConfig {
                episodes,
                trainer_seed,
                checkpoint_every,
                fixed_training_seeds: default_training_seeds(seed_start, seed_count),
                random_seed_count_per_cycle: random_seed_count,
                action_repeat,
                ..TrainingConfig::default()
            };
            let result = train(config, &output_dir)?;
            println!(
                "training complete after {} episodes",
                result.completed_episodes
            );
            println!(
                "best avg survival: {:.2}s",
                result.best_metrics.average_survival_time
            );
            println!("best avg return: {:.2}", result.best_metrics.average_return);
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
            let summary = evaluate_saved_model(
                &saved_model,
                &default_training_seeds(seed_start, seed_count),
            );
            println!("avg survival: {:.2}s", summary.average_survival_time);
            println!("avg return: {:.2}", summary.average_return);
            println!("avg progress: {:.2}", summary.average_progress);
            println!("wins: {}", summary.wins);
        }
    }

    Ok(())
}
