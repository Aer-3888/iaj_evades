use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::config::GameConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Idle,
    Up,
    Down,
    Left,
    Right,
    UpRight,
    DownRight,
    UpLeft,
    DownLeft,
}

impl Action {
    pub const ALL: [Action; 9] = [
        Action::Idle,
        Action::Up,
        Action::Down,
        Action::Left,
        Action::Right,
        Action::UpRight,
        Action::DownRight,
        Action::UpLeft,
        Action::DownLeft,
    ];

    pub fn vector(self) -> (f32, f32) {
        match self {
            Action::Idle => (0.0, 0.0),
            Action::Up => (0.0, -1.0),
            Action::Down => (0.0, 1.0),
            Action::Left => (-1.0, 0.0),
            Action::Right => (1.0, 0.0),
            Action::UpRight => (1.0, -1.0),
            Action::DownRight => (1.0, 1.0),
            Action::UpLeft => (-1.0, -1.0),
            Action::DownLeft => (-1.0, 1.0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CircleBody {
    pub pos: Vec2,
    pub vel: Vec2,
    pub radius: f32,
}

impl CircleBody {
    pub fn distance_squared_to(&self, other: &CircleBody) -> f32 {
        let dx = self.pos.x - other.pos.x;
        let dy = self.pos.y - other.pos.y;
        dx * dx + dy * dy
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Player {
    pub body: CircleBody,
}

impl Player {
    fn apply_action(&mut self, action: Action, speed: f32, dt: f32, config: &GameConfig) {
        let (dx, dy) = action.vector();
        let len_sq = dx * dx + dy * dy;
        if len_sq > 0.0 {
            let inv_len = len_sq.sqrt().recip();
            self.body.vel.x = dx * inv_len * speed;
            self.body.vel.y = dy * inv_len * speed;
        } else {
            self.body.vel.x = 0.0;
            self.body.vel.y = 0.0;
        }

        self.body.pos.x += self.body.vel.x * dt;
        self.body.pos.y += self.body.vel.y * dt;
        self.body.pos.x = self
            .body
            .pos
            .x
            .clamp(self.body.radius, config.world_width - self.body.radius);
        self.body.pos.y = self.body.pos.y.clamp(
            config.corridor_top + self.body.radius,
            config.corridor_bottom - self.body.radius,
        );
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Enemy {
    pub body: CircleBody,
}

impl Enemy {
    fn update(&mut self, dt: f32, config: &GameConfig) {
        self.body.pos.x += self.body.vel.x * dt;
        self.body.pos.y += self.body.vel.y * dt;

        if self.body.pos.x - self.body.radius <= 0.0 {
            self.body.pos.x = self.body.radius;
            self.body.vel.x *= -1.0;
        } else if self.body.pos.x + self.body.radius >= config.world_width {
            self.body.pos.x = config.world_width - self.body.radius;
            self.body.vel.x *= -1.0;
        }

        if self.body.pos.y - self.body.radius <= config.corridor_top {
            self.body.pos.y = config.corridor_top + self.body.radius;
            self.body.vel.y *= -1.0;
        } else if self.body.pos.y + self.body.radius >= config.corridor_bottom {
            self.body.pos.y = config.corridor_bottom - self.body.radius;
            self.body.vel.y *= -1.0;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoneReason {
    None,
    Collision,
    Goal,
    Timeout,
}

impl DoneReason {
    pub fn as_str(self) -> &'static str {
        match self {
            DoneReason::None => "running",
            DoneReason::Collision => "collision",
            DoneReason::Goal => "goal",
            DoneReason::Timeout => "timeout",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StepResult {
    pub reward: f32,
    pub done: bool,
    pub done_reason: DoneReason,
}

#[derive(Clone, Copy, Debug)]
pub struct EpisodeReport {
    pub elapsed_time: f32,
    pub best_progress: f32,
    pub progress_ratio: f32,
    pub fitness: f32,
    pub done_reason: DoneReason,
    pub won: bool,
}

pub struct GameState {
    pub config: GameConfig,
    pub base_seed: u64,
    pub total_deaths: u32,
    pub total_wins: u32,
    pub episode_index: u32,
    pub best_progress_ever: f32,
    pub done: bool,
    pub done_reason: DoneReason,
    pub elapsed_time: f32,
    pub best_x: f32,
    pub last_reward: f32,
    pub player: Player,
    pub enemies: Vec<Enemy>,
    rng: ChaCha8Rng,
}

impl GameState {
    pub fn new(config: GameConfig, seed: Option<u64>) -> Self {
        let base_seed = seed.unwrap_or(config.default_seed);
        let rng = ChaCha8Rng::seed_from_u64(base_seed);
        let player = Player {
            body: CircleBody {
                pos: Vec2 { x: 0.0, y: 0.0 },
                vel: Vec2 { x: 0.0, y: 0.0 },
                radius: config.player_radius,
            },
        };

        let mut state = Self {
            config,
            base_seed,
            total_deaths: 0,
            total_wins: 0,
            episode_index: 0,
            best_progress_ever: 0.0,
            done: false,
            done_reason: DoneReason::None,
            elapsed_time: 0.0,
            best_x: 0.0,
            last_reward: 0.0,
            player,
            enemies: Vec::new(),
            rng,
        };
        state.reset(None);
        state
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        let actual_seed = seed.unwrap_or(self.base_seed);
        self.rng = ChaCha8Rng::seed_from_u64(actual_seed);
        let start_y = self.config.corridor_top + self.config.corridor_height() * 0.5;
        self.player = Player {
            body: CircleBody {
                pos: Vec2 {
                    x: self.config.start_margin,
                    y: start_y,
                },
                vel: Vec2 { x: 0.0, y: 0.0 },
                radius: self.config.player_radius,
            },
        };
        self.enemies = self.spawn_enemies();
        self.done = false;
        self.done_reason = DoneReason::None;
        self.elapsed_time = 0.0;
        self.best_x = self.player.body.pos.x;
        self.last_reward = 0.0;
        self.episode_index += 1;
    }

    pub fn step(&mut self, action: Action, dt: Option<f32>) -> StepResult {
        if self.done {
            return StepResult {
                reward: 0.0,
                done: true,
                done_reason: self.done_reason,
            };
        }

        let delta = dt.unwrap_or(self.config.fixed_timestep);
        let previous_best = self.best_x;

        self.player
            .apply_action(action, self.config.player_speed, delta, &self.config);
        for enemy in &mut self.enemies {
            enemy.update(delta, &self.config);
        }

        self.elapsed_time += delta;
        self.best_x = self.best_x.max(self.player.body.pos.x);
        self.best_progress_ever = self.best_progress_ever.max(self.best_progress());

        let mut reward = (self.best_x - previous_best) * 0.02;

        if self.collided() {
            self.done = true;
            self.done_reason = DoneReason::Collision;
            self.total_deaths += 1;
            reward -= 150.0;
        } else if self.player.body.pos.x + self.player.body.radius >= self.config.goal_x() {
            self.done = true;
            self.done_reason = DoneReason::Goal;
            self.total_wins += 1;
            reward += 250.0;
        } else if self.elapsed_time >= self.config.max_episode_time {
            self.done = true;
            self.done_reason = DoneReason::Timeout;
            reward -= 50.0;
        }

        self.last_reward = reward;

        StepResult {
            reward,
            done: self.done,
            done_reason: self.done_reason,
        }
    }

    #[inline]
    pub fn step_fixed(&mut self, action: Action) -> StepResult {
        self.step(action, Some(self.config.fixed_timestep))
    }

    pub fn best_progress(&self) -> f32 {
        (self.best_x - self.config.start_margin).max(0.0)
    }

    pub fn progress_ratio(&self) -> f32 {
        let total_distance = self.config.goal_x() - self.config.start_margin;
        (self.best_progress() / total_distance).clamp(0.0, 1.0)
    }

    pub fn fitness(&self) -> f32 {
        let mut fitness = self.best_progress() - self.elapsed_time * 2.0;
        match self.done_reason {
            DoneReason::Goal => fitness += 2500.0,
            DoneReason::Collision => fitness -= 400.0,
            DoneReason::Timeout => fitness -= 150.0,
            DoneReason::None => {}
        }
        fitness
    }

    pub fn episode_report(&self) -> EpisodeReport {
        EpisodeReport {
            elapsed_time: self.elapsed_time,
            best_progress: self.best_progress(),
            progress_ratio: self.progress_ratio(),
            fitness: self.fitness(),
            done_reason: self.done_reason,
            won: self.done_reason == DoneReason::Goal,
        }
    }

    fn spawn_enemies(&mut self) -> Vec<Enemy> {
        let wall_enemy_count = self.config.wall_enemy_pairs_per_wall * 4;
        let mut enemies = Vec::with_capacity(self.config.enemy_count + wall_enemy_count);
        enemies.extend(self.spawn_wall_enemies());
        let start_safe_x = self.config.start_margin + 220.0;
        let end_safe_x = self.config.goal_x() - 180.0;

        for _ in 0..self.config.enemy_count {
            let radius = self
                .rng
                .gen_range(self.config.enemy_radius_min..=self.config.enemy_radius_max);
            let mut spawned = None;

            for _ in 0..500 {
                let x = self.rng.gen_range(start_safe_x..end_safe_x);
                let y = self.rng.gen_range(
                    self.config.corridor_top + radius..self.config.corridor_bottom - radius,
                );
                let speed = self
                    .rng
                    .gen_range(self.config.enemy_speed_min..=self.config.enemy_speed_max);
                let angle = self.rng.gen_range(0.0..std::f32::consts::TAU);
                let candidate = Enemy {
                    body: CircleBody {
                        pos: Vec2 { x, y },
                        vel: Vec2 {
                            x: angle.cos() * speed,
                            y: angle.sin() * speed,
                        },
                        radius,
                    },
                };

                if enemies.iter().all(|other: &Enemy| {
                    let dx = candidate.body.pos.x - other.body.pos.x;
                    let dy = candidate.body.pos.y - other.body.pos.y;
                    let min_gap = candidate.body.radius + other.body.radius + 28.0;
                    dx * dx + dy * dy > min_gap * min_gap
                }) {
                    spawned = Some(candidate);
                    break;
                }
            }

            let enemy = spawned.unwrap_or_else(|| Enemy {
                body: CircleBody {
                    pos: Vec2 {
                        x: self.rng.gen_range(start_safe_x..end_safe_x),
                        y: self.rng.gen_range(
                            self.config.corridor_top + radius..self.config.corridor_bottom - radius,
                        ),
                    },
                    vel: Vec2 {
                        x: if self.rng.gen_bool(0.5) { 1.0 } else { -1.0 }
                            * self.rng.gen_range(
                                self.config.enemy_speed_min..=self.config.enemy_speed_max,
                            ),
                        y: if self.rng.gen_bool(0.5) { 1.0 } else { -1.0 }
                            * self.rng.gen_range(
                                self.config.enemy_speed_min..=self.config.enemy_speed_max,
                            ),
                    },
                    radius,
                },
            });

            enemies.push(enemy);
        }

        enemies
    }

    fn spawn_wall_enemies(&self) -> Vec<Enemy> {
        let mut enemies = Vec::with_capacity(self.config.wall_enemy_pairs_per_wall * 4);
        let lane_start = self.config.start_margin + 260.0;
        let lane_end = self.config.goal_x() - 220.0;
        let spacing =
            (lane_end - lane_start) / (self.config.wall_enemy_pairs_per_wall.max(1) as f32 + 1.0);
        let top_y = self.config.corridor_top + self.config.wall_enemy_radius;
        let bottom_y = self.config.corridor_bottom - self.config.wall_enemy_radius;

        for index in 0..self.config.wall_enemy_pairs_per_wall {
            let anchor_x = lane_start + spacing * (index as f32 + 1.0);
            let offset = 48.0;

            enemies.push(Enemy {
                body: CircleBody {
                    pos: Vec2 {
                        x: anchor_x - offset,
                        y: top_y,
                    },
                    vel: Vec2 {
                        x: self.config.wall_enemy_speed,
                        y: 0.0,
                    },
                    radius: self.config.wall_enemy_radius,
                },
            });
            enemies.push(Enemy {
                body: CircleBody {
                    pos: Vec2 {
                        x: anchor_x + offset,
                        y: top_y,
                    },
                    vel: Vec2 {
                        x: -self.config.wall_enemy_speed,
                        y: 0.0,
                    },
                    radius: self.config.wall_enemy_radius,
                },
            });
            enemies.push(Enemy {
                body: CircleBody {
                    pos: Vec2 {
                        x: anchor_x - offset,
                        y: bottom_y,
                    },
                    vel: Vec2 {
                        x: self.config.wall_enemy_speed,
                        y: 0.0,
                    },
                    radius: self.config.wall_enemy_radius,
                },
            });
            enemies.push(Enemy {
                body: CircleBody {
                    pos: Vec2 {
                        x: anchor_x + offset,
                        y: bottom_y,
                    },
                    vel: Vec2 {
                        x: -self.config.wall_enemy_speed,
                        y: 0.0,
                    },
                    radius: self.config.wall_enemy_radius,
                },
            });
        }

        enemies
    }

    fn collided(&self) -> bool {
        self.enemies.iter().any(|enemy| {
            let sum = self.player.body.radius + enemy.body.radius;
            self.player.body.distance_squared_to(&enemy.body) <= sum * sum
        })
    }
}

pub fn reflect_axis(
    position: f32,
    velocity: f32,
    radius: f32,
    min: f32,
    max: f32,
    dt: f32,
) -> (f32, f32) {
    let mut pos = position + velocity * dt;
    let mut vel = velocity;
    let min_bound = min + radius;
    let max_bound = max - radius;

    if min_bound >= max_bound {
        return (position, 0.0);
    }

    while pos < min_bound || pos > max_bound {
        if pos < min_bound {
            pos = min_bound + (min_bound - pos);
            vel = -vel;
        } else if pos > max_bound {
            pos = max_bound - (pos - max_bound);
            vel = -vel;
        }
    }

    (pos, vel)
}

pub fn simulate_enemy_body(body: &CircleBody, dt: f32, config: &GameConfig) -> CircleBody {
    let (x, vx) = reflect_axis(
        body.pos.x,
        body.vel.x,
        body.radius,
        0.0,
        config.world_width,
        dt,
    );
    let (y, vy) = reflect_axis(
        body.pos.y,
        body.vel.y,
        body.radius,
        config.corridor_top,
        config.corridor_bottom,
        dt,
    );
    CircleBody {
        pos: Vec2 { x, y },
        vel: Vec2 { x: vx, y: vy },
        radius: body.radius,
    }
}

pub fn simulate_player_position(
    player: &Player,
    action: Action,
    dt: f32,
    config: &GameConfig,
) -> Vec2 {
    let (dx, dy) = action.vector();
    let direction = Vec2 { x: dx, y: dy };
    let speed = if direction.length_squared() > 0.0 {
        let inv_len = direction.length_squared().sqrt().recip();
        Vec2 {
            x: direction.x * inv_len * config.player_speed,
            y: direction.y * inv_len * config.player_speed,
        }
    } else {
        Vec2 { x: 0.0, y: 0.0 }
    };

    Vec2 {
        x: (player.body.pos.x + speed.x * dt)
            .clamp(player.body.radius, config.world_width - player.body.radius),
        y: (player.body.pos.y + speed.y * dt).clamp(
            config.corridor_top + player.body.radius,
            config.corridor_bottom - player.body.radius,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reaching_goal_finishes_episode() {
        let mut state = GameState::new(GameConfig::default(), Some(1));
        state.enemies.clear();
        state.player.body.pos.x = state.config.goal_x() - state.player.body.radius - 1.0;

        let result = state.step(Action::Right, Some(state.config.fixed_timestep));

        assert!(result.done);
        assert_eq!(result.done_reason, DoneReason::Goal);
    }

    #[test]
    fn collision_finishes_episode() {
        let mut state = GameState::new(GameConfig::default(), Some(1));
        state.enemies.clear();
        state.enemies.push(Enemy {
            body: CircleBody {
                pos: state.player.body.pos,
                vel: Vec2 { x: 0.0, y: 0.0 },
                radius: state.player.body.radius,
            },
        });

        let result = state.step(Action::Idle, Some(state.config.fixed_timestep));

        assert!(result.done);
        assert_eq!(result.done_reason, DoneReason::Collision);
    }

    #[test]
    fn reset_adds_wall_enemies() {
        let config = GameConfig::default();
        let state = GameState::new(config.clone(), Some(1));
        assert!(state.enemies.len() >= config.enemy_count + config.wall_enemy_pairs_per_wall * 4);
    }

    #[test]
    fn wall_enemies_spawn_on_top_and_bottom_edges() {
        let config = GameConfig::default();
        let state = GameState::new(config.clone(), Some(1));
        let top_y = config.corridor_top + config.wall_enemy_radius;
        let bottom_y = config.corridor_bottom - config.wall_enemy_radius;
        let wall_enemy_total = state
            .enemies
            .iter()
            .filter(|enemy| {
                enemy.body.radius == config.wall_enemy_radius
                    && (enemy.body.pos.y == top_y || enemy.body.pos.y == bottom_y)
                    && enemy.body.vel.y == 0.0
            })
            .count();
        assert_eq!(wall_enemy_total, config.wall_enemy_pairs_per_wall * 4);
    }
}
