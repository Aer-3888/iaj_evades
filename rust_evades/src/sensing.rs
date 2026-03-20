use crate::game::{GameState, Vec2};

pub const RAY_COUNT: usize = 36;
pub const INPUT_SIZE: usize = RAY_COUNT * 2 + 2;

#[derive(Clone, Debug)]
pub struct ObservationBuilder {
    previous_rays: [f32; RAY_COUNT],
    previous_position: Vec2,
    initialized: bool,
}

impl Default for ObservationBuilder {
    fn default() -> Self {
        Self {
            previous_rays: [1.0; RAY_COUNT],
            previous_position: Vec2::default(),
            initialized: false,
        }
    }
}

impl ObservationBuilder {
    pub fn reset(&mut self, state: &GameState) {
        self.previous_rays = sample_rays(state);
        self.previous_position = state.player.body.pos;
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
        let delta_x = state.player.body.pos.x - self.previous_position.x;
        let delta_y = state.player.body.pos.y - self.previous_position.y;
        observation[INPUT_SIZE - 2] = if max_step > 0.0 {
            (delta_x / max_step).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        observation[INPUT_SIZE - 1] = if max_step > 0.0 {
            (delta_y / max_step).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        self.previous_rays = rays;
        self.previous_position = state.player.body.pos;
        observation
    }
}

pub fn sample_rays(state: &GameState) -> [f32; RAY_COUNT] {
    let mut samples = [1.0; RAY_COUNT];
    let origin_x = state.player.body.pos.x;
    let origin_y = state.player.body.pos.y;
    let max_distance = state.config.ray_length().max(1.0);

    for (index, sample) in samples.iter_mut().enumerate() {
        let angle = (index as f32) * (360.0 / RAY_COUNT as f32).to_radians();
        let dir_x = angle.cos();
        let dir_y = -angle.sin();

        let mut min_distance = state
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

        if state.config.map_design == crate::config::MapDesign::Closed {
            // Horizontal walls (Top/Bottom)
            if let Some(d) = raycast_horizontal_line_distance(origin_y, dir_y, state.config.corridor_top) {
                min_distance = min_distance.min(d);
            }
            if let Some(d) = raycast_horizontal_line_distance(origin_y, dir_y, state.config.corridor_bottom) {
                min_distance = min_distance.min(d);
            }

            // Vertical walls (Left/Right)
            if let Some(d) = raycast_vertical_line_distance(origin_x, dir_x, 0.0) {
                min_distance = min_distance.min(d);
            }
            if let Some(d) = raycast_vertical_line_distance(origin_x, dir_x, state.config.world_width) {
                min_distance = min_distance.min(d);
            }
        }

        let clearance = (min_distance - state.player.body.radius).clamp(0.0, max_distance);
        *sample = if min_distance.is_finite() {
            (clearance / max_distance).clamp(0.0, 1.0)
        } else {
            1.0
        };
    }

    samples
}

fn raycast_horizontal_line_distance(origin_y: f32, dir_y: f32, line_y: f32) -> Option<f32> {
    if dir_y.abs() < 1e-6 {
        return None;
    }
    let t = (line_y - origin_y) / dir_y;
    if t > 0.0 {
        Some(t)
    } else {
        None
    }
}

fn raycast_vertical_line_distance(origin_x: f32, dir_x: f32, line_x: f32) -> Option<f32> {
    if dir_x.abs() < 1e-6 {
        return None;
    }
    let t = (line_x - origin_x) / dir_x;
    if t > 0.0 {
        Some(t)
    } else {
        None
    }
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
    use crate::{
        config::GameConfig,
        game::{CircleBody, Enemy, GameState},
    };

    fn approx_eq(left: f32, right: f32) {
        assert!((left - right).abs() < 1.0e-4, "{left} != {right}");
    }

    #[test]
    fn observation_has_expected_size() {
        let state = GameState::new(GameConfig::default(), Some(2));
        let mut builder = ObservationBuilder::default();
        let observation = builder.build(&state);
        assert_eq!(observation.len(), INPUT_SIZE);
    }

    #[test]
    fn ray_is_zero_when_enemy_overlaps_player() {
        let mut state = GameState::new(GameConfig::default(), Some(2));
        state.enemies.clear();
        state.enemies.push(Enemy {
            body: CircleBody {
                pos: state.player.body.pos,
                vel: Vec2::default(),
                radius: state.config.enemy_radius,
            },
            remaining_life: 1.0,
        });

        let rays = sample_rays(&state);
        approx_eq(rays[0], 0.0);
    }

    #[test]
    fn ray_is_one_when_enemy_is_outside_range() {
        let mut state = GameState::new(GameConfig::default(), Some(2));
        state.enemies.clear();
        state.enemies.push(Enemy {
            body: CircleBody {
                pos: Vec2 {
                    x: state.config.ray_length() * 4.0,
                    y: 0.0,
                },
                vel: Vec2::default(),
                radius: state.config.enemy_radius,
            },
            remaining_life: 1.0,
        });

        let rays = sample_rays(&state);
        approx_eq(rays[0], 1.0);
    }

    #[test]
    fn ray_detects_walls_in_closed_mode() {
        use crate::config::MapDesign;
        let mut state = GameState::new(GameConfig::default(), Some(2));
        state.config.map_design = MapDesign::Closed;
        state.enemies.clear();

        // Position player exactly 32 units from the top wall
        // corridor_top is 60.0. Player radius is 16.0.
        // If player.y = 92.0, distance to wall is 32.0.
        // Clearance = 32.0 - 16.0 = 16.0.
        // ray_length = 16.0 * 5.0 = 80.0.
        // Sample = 16.0 / 80.0 = 0.2.
        state.player.body.pos = Vec2 { x: 100.0, y: 92.0 };
        
        let rays = sample_rays(&state);
        
        // Ray index for 90 degrees (up) is RAY_COUNT / 4
        // RAY_COUNT = 36. 36 / 4 = 9.
        // dir_y = -sin(90) = -1.0. 
        // t = (60.0 - 92.0) / -1.0 = 32.0.
        approx_eq(rays[9], 0.2);

        // Position player 32 units from bottom wall (440.0)
        // y = 408.0. t = (440.0 - 408.0) / 1.0 = 32.0.
        // Ray index for 270 degrees (down) is 27.
        state.player.body.pos = Vec2 { x: 100.0, y: 408.0 };
        let rays = sample_rays(&state);
        approx_eq(rays[27], 0.2);
    }
}
