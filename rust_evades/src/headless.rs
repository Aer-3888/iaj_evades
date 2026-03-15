use crate::{
    config::GameConfig,
    game::{Action, EpisodeReport, GameState},
    model_player::ModelController,
};

#[derive(Clone, Copy, Debug)]
pub enum ControllerMode {
    Model,
    RightOnly,
}

impl ControllerMode {
    pub fn action(self, state: &GameState, model: &mut Option<ModelController>) -> Action {
        match self {
            ControllerMode::Model => model
                .as_mut()
                .map(|controller| controller.choose_action(state))
                .unwrap_or(Action::Idle),
            ControllerMode::RightOnly => Action::Right,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HeadlessOptions {
    pub seed: Option<u64>,
    pub episodes: u32,
    pub controller: ControllerMode,
    pub model: Option<ModelController>,
}

#[derive(Clone, Debug)]
pub struct HeadlessSummary {
    pub episodes: u32,
    pub timeouts: u32,
    pub collisions: u32,
    pub average_survival_time: f32,
    pub best_survival_time: f32,
    pub average_evades: f32,
    pub average_reward: f32,
    pub last_report: EpisodeReport,
}

pub fn run_headless(config: GameConfig, options: HeadlessOptions) -> HeadlessSummary {
    let mut state = GameState::new(config, options.seed);
    let mut model = options.model;
    let mut total_survival = 0.0;
    let mut total_evades = 0.0;
    let mut total_reward = 0.0;
    let mut timeouts = 0;
    let mut collisions = 0;
    let mut best_survival_time: f32 = 0.0;
    let mut last_report = state.episode_report();

    for _ in 0..options.episodes {
        state.reset(None);
        if let Some(controller) = &mut model {
            controller.reset(&state);
        }
        while !state.done {
            let action = options.controller.action(&state, &mut model);
            state.step_fixed(action);
        }
        let report = state.episode_report();
        total_survival += report.elapsed_time;
        total_evades += report.enemies_evaded as f32;
        total_reward += report.total_reward;
        best_survival_time = best_survival_time.max(report.elapsed_time);
        if report.survived_full_episode {
            timeouts += 1;
        } else {
            collisions += 1;
        }
        last_report = report;
    }

    let episodes = options.episodes.max(1);
    HeadlessSummary {
        episodes,
        timeouts,
        collisions,
        average_survival_time: total_survival / episodes as f32,
        best_survival_time,
        average_evades: total_evades / episodes as f32,
        average_reward: total_reward / episodes as f32,
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
                model: None,
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
                model: None,
            },
        );

        assert_eq!(fast.last_report.done_reason, slow.last_report.done_reason);
        assert_eq!(fast.last_report.elapsed_time, slow.last_report.elapsed_time);
        assert_eq!(fast.best_survival_time, slow.best_survival_time);
        assert_eq!(fast.average_reward, slow.average_reward);
    }
}
