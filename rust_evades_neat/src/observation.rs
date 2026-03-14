use rust_evades::game::GameState;

pub const RAY_COUNT: usize = 36;
pub const INPUT_SIZE: usize = RAY_COUNT * 2 + 1;

#[derive(Clone, Debug)]
pub struct ObservationBuilder {
    previous_rays: [f32; RAY_COUNT],
    previous_x: f32,
    initialized: bool,
}

impl Default for ObservationBuilder {
    fn default() -> Self {
        Self {
            previous_rays: [0.0; RAY_COUNT],
            previous_x: 0.0,
            initialized: false,
        }
    }
}

impl ObservationBuilder {
    pub fn reset(&mut self, state: &GameState) {
        self.previous_rays = sample_rays(state);
        self.previous_x = state.player.body.pos.x;
        self.initialized = true;
    }

    pub fn build(&mut self, state: &GameState) -> [f32; INPUT_SIZE] {
        if !self.initialized {
            self.reset(state);
        }

        let rays = sample_rays(state);
        let mut observation = [0.0; INPUT_SIZE];
        observation[..RAY_COUNT].copy_from_slice(&rays);

        for index in 0..RAY_COUNT {
            observation[RAY_COUNT + index] = rays[index] - self.previous_rays[index];
        }

        let max_step = state.config.player_speed * state.config.fixed_timestep;
        observation[INPUT_SIZE - 1] = if max_step > 0.0 {
            ((state.player.body.pos.x - self.previous_x) / max_step).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        self.previous_rays = rays;
        self.previous_x = state.player.body.pos.x;
        observation
    }
}

pub fn sample_rays(state: &GameState) -> [f32; RAY_COUNT] {
    let mut samples = [0.0; RAY_COUNT];
    let origin_x = state.player.body.pos.x;
    let origin_y = state.player.body.pos.y;
    let max_distance = (state.config.world_width.powi(2) + state.config.corridor_height().powi(2))
        .sqrt()
        .max(1.0);

    for (index, sample) in samples.iter_mut().enumerate() {
        let angle = (index as f32) * 10.0_f32.to_radians();
        let dir_x = angle.cos();
        let dir_y = -angle.sin();

        let wall_distance = raycast_wall_distance(state, origin_x, origin_y, dir_x, dir_y);
        let enemy_distance = state
            .enemies
            .iter()
            .filter_map(|enemy| {
                raycast_circle_distance(
                    origin_x,
                    origin_y,
                    dir_x,
                    dir_y,
                    enemy.body.pos.x,
                    enemy.body.pos.y,
                    enemy.body.radius,
                )
            })
            .fold(f32::INFINITY, f32::min);

        let hit_distance = wall_distance.min(enemy_distance);
        let clearance = (hit_distance - state.player.body.radius).max(0.0);
        *sample = (clearance / max_distance).clamp(0.0, 1.0);
    }

    samples
}

fn raycast_wall_distance(
    state: &GameState,
    origin_x: f32,
    origin_y: f32,
    dir_x: f32,
    dir_y: f32,
) -> f32 {
    let mut best = f32::INFINITY;
    let epsilon = 1.0e-6;

    if dir_x.abs() > epsilon {
        if dir_x > 0.0 {
            best = best.min((state.config.world_width - origin_x) / dir_x);
        } else {
            best = best.min((0.0 - origin_x) / dir_x);
        }
    }

    if dir_y.abs() > epsilon {
        if dir_y > 0.0 {
            best = best.min((state.config.corridor_bottom - origin_y) / dir_y);
        } else {
            best = best.min((state.config.corridor_top - origin_y) / dir_y);
        }
    }

    best.max(0.0)
}

fn raycast_circle_distance(
    origin_x: f32,
    origin_y: f32,
    dir_x: f32,
    dir_y: f32,
    center_x: f32,
    center_y: f32,
    radius: f32,
) -> Option<f32> {
    let offset_x = origin_x - center_x;
    let offset_y = origin_y - center_y;
    let projection = offset_x * dir_x + offset_y * dir_y;
    let c = offset_x * offset_x + offset_y * offset_y - radius * radius;

    if c > 0.0 && projection > 0.0 {
        return None;
    }

    let discriminant = projection * projection - c;
    if discriminant < 0.0 {
        return None;
    }

    let mut distance = -projection - discriminant.sqrt();
    if distance < 0.0 {
        distance = 0.0;
    }
    Some(distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_evades::config::GameConfig;
    use rust_evades::game::GameState;

    #[test]
    fn observation_has_expected_size() {
        let state = GameState::new(GameConfig::default(), Some(2));
        let mut builder = ObservationBuilder::default();
        let observation = builder.build(&state);
        assert_eq!(observation.len(), INPUT_SIZE);
    }

    #[test]
    fn reset_zeroes_delta_channels() {
        let state = GameState::new(GameConfig::default(), Some(2));
        let mut builder = ObservationBuilder::default();
        builder.reset(&state);
        let observation = builder.build(&state);
        assert!(observation[RAY_COUNT..INPUT_SIZE]
            .iter()
            .all(|value| value.abs() < 1.0e-6));
    }
}
