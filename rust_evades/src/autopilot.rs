use crate::game::{simulate_enemy_body, simulate_player_position, Action, GameState};

pub struct AutoPilot {
    sample_horizons: [f32; 4],
}

impl Default for AutoPilot {
    fn default() -> Self {
        Self {
            sample_horizons: [0.10, 0.22, 0.35, 0.50],
        }
    }
}

impl AutoPilot {
    pub fn choose_action(&self, state: &GameState) -> Action {
        const CANDIDATES: [Action; 6] = [
            Action::Right,
            Action::UpRight,
            Action::DownRight,
            Action::Up,
            Action::Down,
            Action::Idle,
        ];

        let mut best_action = Action::Right;
        let mut best_score = f32::NEG_INFINITY;

        for action in CANDIDATES {
            let score = self.score_action(state, action);
            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }

        best_action
    }

    fn score_action(&self, state: &GameState, action: Action) -> f32 {
        let config = &state.config;
        let mut clearance_sum = 0.0;
        let mut worst_clearance = f32::INFINITY;

        for horizon in self.sample_horizons {
            let player_pos = simulate_player_position(&state.player, action, horizon, config);
            let mut sample_clearance = f32::INFINITY;

            for enemy in &state.enemies {
                let simulated = simulate_enemy_body(&enemy.body, horizon, config);
                let dx = simulated.pos.x - player_pos.x;
                let dy = simulated.pos.y - player_pos.y;
                let center_distance = (dx * dx + dy * dy).sqrt();
                let clearance = center_distance - (simulated.radius + state.player.body.radius);
                sample_clearance = sample_clearance.min(clearance);
            }

            clearance_sum += sample_clearance;
            worst_clearance = worst_clearance.min(sample_clearance);
        }

        let preview = simulate_player_position(&state.player, action, 0.40, config);
        let progress_bias = (preview.x - state.player.body.pos.x) * 0.22;
        let center_line = config.corridor_top + config.corridor_height() * 0.5;
        let lane_bias = -((preview.y - center_line).abs() / config.corridor_height()) * 6.0;
        let wall_bias = if preview.y < config.corridor_top + 40.0
            || preview.y > config.corridor_bottom - 40.0
        {
            -18.0
        } else {
            0.0
        };

        worst_clearance * 4.5 + clearance_sum * 0.55 + progress_bias + lane_bias + wall_bias
    }
}
