use crate::{
    autopilot::AutoPilot,
    config::GameConfig,
    game::{Action, EpisodeReport, GameState},
};

#[derive(Clone, Copy, Debug)]
pub enum ControllerMode {
    Auto,
    RightOnly,
}

impl ControllerMode {
    pub fn action(self, state: &GameState, autopilot: &AutoPilot) -> Action {
        match self {
            ControllerMode::Auto => autopilot.choose_action(state),
            ControllerMode::RightOnly => Action::Right,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HeadlessOptions {
    pub seed: Option<u64>,
    pub episodes: u32,
    pub controller: ControllerMode,
}

#[derive(Clone, Debug)]
pub struct HeadlessSummary {
    pub episodes: u32,
    pub wins: u32,
    pub deaths: u32,
    pub average_progress: f32,
    pub best_progress: f32,
    pub average_fitness: f32,
    pub last_report: EpisodeReport,
}

pub fn run_headless(config: GameConfig, options: HeadlessOptions) -> HeadlessSummary {
    let mut state = GameState::new(config, options.seed);
    let autopilot = AutoPilot::default();
    let mut total_progress = 0.0;
    let mut total_fitness = 0.0;
    let mut wins = 0;
    let mut deaths = 0;
    let mut best_progress: f32 = 0.0;
    let mut last_report = state.episode_report();

    for _ in 0..options.episodes {
        state.reset(None);
        while !state.done {
            let action = options.controller.action(&state, &autopilot);
            state.step_fixed(action);
        }
        let report = state.episode_report();
        total_progress += report.best_progress;
        total_fitness += report.fitness;
        best_progress = best_progress.max(report.best_progress);
        if report.won {
            wins += 1;
        } else {
            deaths += 1;
        }
        last_report = report;
    }

    let episodes = options.episodes.max(1);
    HeadlessSummary {
        episodes,
        wins,
        deaths,
        average_progress: total_progress / episodes as f32,
        best_progress,
        average_fitness: total_fitness / episodes as f32,
        last_report,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headless_ignores_render_fps_cap() {
        let fast = run_headless(
            GameConfig::default(),
            HeadlessOptions {
                seed: Some(7),
                episodes: 1,
                controller: ControllerMode::RightOnly,
            },
        );

        let mut config = GameConfig::default();
        config.render_fps = 1;
        let slow = run_headless(
            config,
            HeadlessOptions {
                seed: Some(7),
                episodes: 1,
                controller: ControllerMode::RightOnly,
            },
        );

        assert_eq!(fast.last_report.done_reason, slow.last_report.done_reason);
        assert_eq!(fast.last_report.elapsed_time, slow.last_report.elapsed_time);
        assert_eq!(fast.best_progress, slow.best_progress);
        assert_eq!(fast.average_fitness, slow.average_fitness);
    }
}
