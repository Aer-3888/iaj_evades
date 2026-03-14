use clap::{Parser, ValueEnum};

use rust_evades::{
    config::GameConfig,
    headless::{run_headless, ControllerMode, HeadlessOptions},
    model_player::ModelController,
    render::run_window,
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliController {
    Model,
    Right,
}

impl From<CliController> for ControllerMode {
    fn from(value: CliController) -> Self {
        match value {
            CliController::Model => ControllerMode::Model,
            CliController::Right => ControllerMode::RightOnly,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "rust_evades")]
#[command(about = "Rust rewrite of Evades with graphical and headless modes")]
struct Cli {
    #[arg(long, help = "Run uncapped simulation without opening a window")]
    headless: bool,

    #[arg(long, default_value_t = 1)]
    episodes: u32,

    #[arg(long)]
    seed: Option<u64>,

    #[arg(long, help = "Path to a trained DQN or NEAT model JSON for gameplay")]
    model: Option<String>,

    #[arg(long, value_enum, default_value_t = CliController::Model)]
    controller: CliController,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let config = GameConfig::default();
    let model = cli
        .model
        .as_deref()
        .map(ModelController::load)
        .transpose()?;

    if matches!(cli.controller, CliController::Model) && model.is_none() {
        anyhow::bail!("`--model <path>` is required when using model-based control");
    }

    if cli.headless {
        let summary = run_headless(
            config,
            HeadlessOptions {
                seed: cli.seed,
                episodes: cli.episodes.max(1),
                controller: cli.controller.into(),
                model,
            },
        );

        println!("headless episodes: {}", summary.episodes);
        println!("wins: {}", summary.wins);
        println!("deaths: {}", summary.deaths);
        println!("avg progress: {:.2}", summary.average_progress);
        println!("best progress: {:.2}", summary.best_progress);
        println!("avg fitness: {:.2}", summary.average_fitness);
        println!(
            "last result: {} after {:.2}s ({:.1}%)",
            summary.last_report.done_reason.as_str(),
            summary.last_report.elapsed_time,
            summary.last_report.progress_ratio * 100.0
        );
        return Ok(());
    }

    run_window(config, cli.seed, model)
}
