use std::{fs, path::PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};

use rust_evades_dqn::{
    model::SavedModel,
    trainer::{
        default_training_seeds, evaluate_saved_model, train, SeedFocusMode, TrainingConfig,
    },
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliSeedFocusMode {
    Original,
    BadSeeds,
}

impl From<CliSeedFocusMode> for SeedFocusMode {
    fn from(value: CliSeedFocusMode) -> Self {
        match value {
            CliSeedFocusMode::Original => SeedFocusMode::Original,
            CliSeedFocusMode::BadSeeds => SeedFocusMode::BadSeeds,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "rust_evades_dqn")]
#[command(about = "DQN trainer for the infinite-space dodge environment")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Train {
        #[arg(long, default_value = "training_runs/dqn_default")]
        output_dir: PathBuf,

        #[arg(long, help = "Resume training from an existing saved DQN model JSON")]
        resume_model: Option<PathBuf>,

        #[arg(long, default_value_t = 500000)]
        episodes: usize,

        #[arg(long, default_value_t = 7)]
        trainer_seed: u64,

        #[arg(long, default_value_t = 2)]
        seed_start: u64,

        #[arg(long, default_value_t = 24)]
        seed_count: usize,

        #[arg(long, default_value_t = 2)]
        random_seed_count: usize,

        #[arg(long, default_value_t = 2)]
        action_repeat: usize,

        #[arg(long, default_value_t = 100)]
        checkpoint_every: usize,

        #[arg(long, value_enum, default_value_t = CliSeedFocusMode::BadSeeds)]
        seed_focus_mode: CliSeedFocusMode,
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
            resume_model,
            episodes,
            trainer_seed,
            seed_start,
            seed_count,
            random_seed_count,
            action_repeat,
            checkpoint_every,
            seed_focus_mode,
        } => {
            let config = TrainingConfig {
                episodes,
                trainer_seed,
                checkpoint_every,
                seed_focus_mode: seed_focus_mode.into(),
                fixed_training_seeds: default_training_seeds(seed_start, seed_count),
                random_seed_count_per_cycle: random_seed_count,
                action_repeat,
                ..TrainingConfig::default()
            };
            let resume_model = match resume_model {
                Some(path) => {
                    let json = fs::read_to_string(&path)
                        .with_context(|| format!("failed to read {}", path.display()))?;
                    Some(
                        serde_json::from_str(&json)
                            .with_context(|| format!("failed to parse {}", path.display()))?,
                    )
                }
                None => None,
            };
            let result = train(config, &output_dir, resume_model, None, None)?;
            println!(
                "training complete after {} episodes",
                result.completed_episodes
            );
            println!(
                "best avg survival: {:.2}s",
                result.best_metrics.average_survival_time
            );
            println!(
                "best worst-seed survival: {:.2}s",
                result.best_metrics.min_survival_time
            );
            println!("best avg return: {:.2}", result.best_metrics.average_return);
            println!(
                "best worst-seed return: {:.2}",
                result.best_metrics.min_return
            );
            println!("best avg evades: {:.2}", result.best_metrics.average_evades);
            println!("best timeouts: {}", result.best_metrics.timeouts);
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
            println!("worst-seed survival: {:.2}s", summary.min_survival_time);
            println!("avg return: {:.2}", summary.average_return);
            println!("worst-seed return: {:.2}", summary.min_return);
            println!("avg evades: {:.2}", summary.average_evades);
            println!("timeouts: {}", summary.timeouts);
        }
    }

    Ok(())
}
