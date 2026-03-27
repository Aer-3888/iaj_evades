use std::{fs, path::PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use rust_evades::config::MapDesign;

use rust_evades_dqn::{
    model::{ModelType, SavedModel},
    trainer::{
        default_training_seeds, evaluate_saved_model, run_benchmark, train, BenchmarkMode,
        SeedFocusMode, TrainingConfig,
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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliModelType {
    Dqn,
    Dqn2,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliMapDesign {
    Open,
    Closed,
    Arena,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliBenchmarkMode {
    FullTraining,
    SimulatedSurvival,
}

impl From<CliMapDesign> for MapDesign {
    fn from(value: CliMapDesign) -> Self {
        match value {
            CliMapDesign::Open => MapDesign::Open,
            CliMapDesign::Closed => MapDesign::Closed,
            CliMapDesign::Arena => MapDesign::Arena,
        }
    }
}

impl From<CliModelType> for ModelType {
    fn from(value: CliModelType) -> Self {
        match value {
            CliModelType::Dqn => ModelType::Dqn,
            CliModelType::Dqn2 => ModelType::Dqn2,
        }
    }
}

impl From<CliBenchmarkMode> for BenchmarkMode {
    fn from(value: CliBenchmarkMode) -> Self {
        match value {
            CliBenchmarkMode::FullTraining => BenchmarkMode::FullTraining,
            CliBenchmarkMode::SimulatedSurvival => BenchmarkMode::SimulatedSurvival,
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

        #[arg(long, value_enum, default_value_t = CliModelType::Dqn)]
        model_type: CliModelType,

        #[arg(long, value_enum, default_value_t = CliMapDesign::Open)]
        map_design: CliMapDesign,
    },
    Evaluate {
        #[arg(long)]
        model: PathBuf,

        #[arg(long, default_value_t = 2)]
        seed_start: u64,

        #[arg(long, default_value_t = 24)]
        seed_count: usize,
    },
    Benchmark {
        #[arg(long, value_enum, default_value_t = CliBenchmarkMode::FullTraining)]
        mode: CliBenchmarkMode,

        #[arg(long)]
        output_dir: Option<PathBuf>,

        #[arg(long)]
        episodes: Option<usize>,

        #[arg(long, default_value_t = 7)]
        trainer_seed: u64,

        #[arg(long, default_value_t = 2)]
        seed_start: u64,

        #[arg(long, default_value_t = 24)]
        seed_count: usize,

        #[arg(long, default_value_t = 2)]
        random_seed_count: usize,

        #[arg(long, default_value_t = 100)]
        checkpoint_every: usize,

        #[arg(long, default_value_t = 40.0)]
        simulated_survival_seconds: f32,
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
            model_type,
            map_design,
        } => {
            let config = TrainingConfig {
                model_type: model_type.into(),
                episodes,
                trainer_seed,
                checkpoint_every,
                seed_focus_mode: seed_focus_mode.into(),
                fixed_training_seeds: default_training_seeds(seed_start, seed_count),
                random_seed_count_per_cycle: random_seed_count,
                action_repeat,
                map_design: map_design.into(),
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
        Command::Benchmark {
            mode,
            output_dir,
            episodes,
            trainer_seed,
            seed_start,
            seed_count,
            random_seed_count,
            checkpoint_every,
            simulated_survival_seconds,
        } => {
            let benchmark_mode: BenchmarkMode = mode.into();
            let output_dir = output_dir.unwrap_or_else(|| match benchmark_mode {
                BenchmarkMode::FullTraining => {
                    PathBuf::from("training_runs/benchmarks/dqn2_arena_full_training")
                }
                BenchmarkMode::SimulatedSurvival => {
                    PathBuf::from("training_runs/benchmarks/dqn2_arena_simulated_survival")
                }
            });
            let episodes = episodes.unwrap_or(match benchmark_mode {
                BenchmarkMode::FullTraining => 160,
                BenchmarkMode::SimulatedSurvival => 4,
            });
            let config = TrainingConfig {
                model_type: ModelType::Dqn2,
                episodes,
                trainer_seed,
                checkpoint_every,
                fixed_training_seeds: default_training_seeds(seed_start, seed_count),
                random_seed_count_per_cycle: random_seed_count,
                map_design: MapDesign::Arena,
                ..TrainingConfig::default()
            };
            let report = run_benchmark(
                config,
                &output_dir,
                benchmark_mode,
                simulated_survival_seconds,
            )?;
            println!("benchmark mode: {:?}", report.mode);
            println!("episodes completed: {}", report.episodes_completed);
            println!("total steps: {}", report.total_steps_completed);
            println!(
                "best avg survival: {:.2}s",
                report.best_metrics.average_survival_time
            );
            println!(
                "report written to {}",
                report.report_path.display()
            );
        }
    }

    Ok(())
}
