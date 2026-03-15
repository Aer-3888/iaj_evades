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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn normalized_or_zero(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > 0.0 {
            let inv_len = len_sq.sqrt().recip();
            Self {
                x: self.x * inv_len,
                y: self.y * inv_len,
            }
        } else {
            Self::default()
        }
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
    fn apply_action(&mut self, action: Action, speed: f32, dt: f32) {
        let (dx, dy) = action.vector();
        let direction = Vec2 { x: dx, y: dy }.normalized_or_zero();
        self.body.vel = Vec2 {
            x: direction.x * speed,
            y: direction.y * speed,
        };
        self.body.pos.x += self.body.vel.x * dt;
        self.body.pos.y += self.body.vel.y * dt;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Enemy {
    pub body: CircleBody,
    pub remaining_life: f32,
}

impl Enemy {
    fn update(&mut self, dt: f32) {
        self.body.pos.x += self.body.vel.x * dt;
        self.body.pos.y += self.body.vel.y * dt;
        self.remaining_life -= dt;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoneReason {
    None,
    Collision,
    Timeout,
}

impl DoneReason {
    pub fn as_str(self) -> &'static str {
        match self {
            DoneReason::None => "running",
            DoneReason::Collision => "collision",
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
    pub enemies_evaded: u32,
    pub total_reward: f32,
    pub done_reason: DoneReason,
    pub survived_full_episode: bool,
}

pub struct GameState {
    pub config: GameConfig,
    pub base_seed: u64,
    pub total_deaths: u32,
    pub total_timeouts: u32,
    pub episode_index: u32,
    pub best_survival_ever: f32,
    pub done: bool,
    pub done_reason: DoneReason,
    pub elapsed_time: f32,
    pub last_reward: f32,
    pub episode_return: f32,
    pub player: Player,
    pub enemies: Vec<Enemy>,
    pub enemies_evaded: u32,
    rng: ChaCha8Rng,
    next_spawn_in: f32,
}

impl GameState {
    pub fn new(config: GameConfig, seed: Option<u64>) -> Self {
        let base_seed = seed.unwrap_or(config.default_seed);
        let rng = ChaCha8Rng::seed_from_u64(base_seed);
        let player = Player {
            body: CircleBody {
                pos: Vec2::default(),
                vel: Vec2::default(),
                radius: config.player_radius,
            },
        };

        let mut state = Self {
            config,
            base_seed,
            total_deaths: 0,
            total_timeouts: 0,
            episode_index: 0,
            best_survival_ever: 0.0,
            done: false,
            done_reason: DoneReason::None,
            elapsed_time: 0.0,
            last_reward: 0.0,
            episode_return: 0.0,
            player,
            enemies: Vec::new(),
            enemies_evaded: 0,
            rng,
            next_spawn_in: 0.0,
        };
        state.reset(None);
        state
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        let actual_seed = seed.unwrap_or(self.base_seed);
        self.rng = ChaCha8Rng::seed_from_u64(actual_seed);
        self.player = Player {
            body: CircleBody {
                pos: Vec2::default(),
                vel: Vec2::default(),
                radius: self.config.player_radius,
            },
        };
        self.enemies.clear();
        self.enemies_evaded = 0;
        self.done = false;
        self.done_reason = DoneReason::None;
        self.elapsed_time = 0.0;
        self.last_reward = 0.0;
        self.episode_return = 0.0;
        self.next_spawn_in = self.random_spawn_interval();
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
        self.player
            .apply_action(action, self.config.player_speed, delta);
        self.elapsed_time += delta;

        for enemy in &mut self.enemies {
            enemy.update(delta);
        }
        self.spawn_enemies(delta);

        let mut reward = self.config.survival_reward_per_second * delta;

        if self.collided() {
            self.done = true;
            self.done_reason = DoneReason::Collision;
            self.total_deaths += 1;
            reward += self.config.collision_penalty;
        } else {
            let evaded = self.collect_expired_enemies();
            self.enemies_evaded += evaded;
            reward += evaded as f32 * self.config.enemy_evade_reward;

            if self.elapsed_time >= self.config.max_episode_time {
                self.done = true;
                self.done_reason = DoneReason::Timeout;
                self.total_timeouts += 1;
                reward += self.config.timeout_bonus;
            }
        }

        self.best_survival_ever = self.best_survival_ever.max(self.elapsed_time);
        self.last_reward = reward;
        self.episode_return += reward;

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

    pub fn fitness(&self) -> f32 {
        self.episode_return
    }

    pub fn episode_report(&self) -> EpisodeReport {
        EpisodeReport {
            elapsed_time: self.elapsed_time,
            enemies_evaded: self.enemies_evaded,
            total_reward: self.episode_return,
            done_reason: self.done_reason,
            survived_full_episode: self.done_reason == DoneReason::Timeout,
        }
    }

    fn spawn_enemies(&mut self, delta: f32) {
        self.next_spawn_in -= delta;
        while self.next_spawn_in <= 0.0 {
            let enemy = self.spawn_enemy();
            self.enemies.push(enemy);
            self.next_spawn_in += self.random_spawn_interval();
        }
    }

    fn spawn_enemy(&mut self) -> Enemy {
        let angle = self.rng.gen_range(0.0..std::f32::consts::TAU);
        let direction = Vec2 {
            x: angle.cos(),
            y: angle.sin(),
        };
        let spawn_offset = Vec2 {
            x: direction.x * self.config.enemy_spawn_distance,
            y: direction.y * self.config.enemy_spawn_distance,
        };
        let spawn_pos = Vec2 {
            x: self.player.body.pos.x + spawn_offset.x,
            y: self.player.body.pos.y + spawn_offset.y,
        };

        Enemy {
            body: CircleBody {
                pos: spawn_pos,
                vel: Vec2 {
                    x: -direction.x * self.config.enemy_speed,
                    y: -direction.y * self.config.enemy_speed,
                },
                radius: self.config.enemy_radius,
            },
            remaining_life: self.config.enemy_lifetime,
        }
    }

    fn random_spawn_interval(&mut self) -> f32 {
        self.rng
            .gen_range(self.config.enemy_spawn_interval_min..=self.config.enemy_spawn_interval_max)
    }

    fn collect_expired_enemies(&mut self) -> u32 {
        let mut evaded = 0;
        self.enemies.retain(|enemy| {
            let keep = enemy.remaining_life > 0.0;
            if !keep {
                evaded += 1;
            }
            keep
        });
        evaded
    }

    fn collided(&self) -> bool {
        self.enemies.iter().any(|enemy| {
            let sum = self.player.body.radius + enemy.body.radius;
            self.player.body.distance_squared_to(&enemy.body) <= sum * sum
        })
    }
}

pub fn simulate_player_position(
    player: &Player,
    action: Action,
    dt: f32,
    config: &GameConfig,
) -> Vec2 {
    let (dx, dy) = action.vector();
    let direction = Vec2 { x: dx, y: dy }.normalized_or_zero();
    Vec2 {
        x: player.body.pos.x + direction.x * config.player_speed * dt,
        y: player.body.pos.y + direction.y * config.player_speed * dt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(left: f32, right: f32) {
        assert!((left - right).abs() < 1.0e-4, "{left} != {right}");
    }

    #[test]
    fn infinite_space_allows_negative_positions() {
        let mut state = GameState::new(GameConfig::default(), Some(1));
        for _ in 0..30 {
            let result = state.step_fixed(Action::UpLeft);
            assert!(!result.done);
        }

        assert!(state.player.body.pos.x < 0.0);
        assert!(state.player.body.pos.y < 0.0);
    }

    #[test]
    fn spawned_enemy_motion_stays_straight() {
        let mut state = GameState::new(GameConfig::default(), Some(5));
        state.next_spawn_in = 0.0;
        state.step_fixed(Action::Idle);
        assert_eq!(state.enemies.len(), 1);

        let initial = state.enemies[0];
        state.step_fixed(Action::Idle);
        let updated = state.enemies[0];

        approx_eq(updated.body.vel.x, initial.body.vel.x);
        approx_eq(updated.body.vel.y, initial.body.vel.y);
        approx_eq(
            updated.body.pos.x,
            initial.body.pos.x + initial.body.vel.x * state.config.fixed_timestep,
        );
        approx_eq(
            updated.body.pos.y,
            initial.body.pos.y + initial.body.vel.y * state.config.fixed_timestep,
        );
    }

    #[test]
    fn seeded_spawns_are_reproducible() {
        let mut left = GameState::new(GameConfig::default(), Some(9));
        let mut right = GameState::new(GameConfig::default(), Some(9));
        left.next_spawn_in = 0.0;
        right.next_spawn_in = 0.0;

        left.step_fixed(Action::Idle);
        right.step_fixed(Action::Idle);

        assert_eq!(left.enemies.len(), 1);
        assert_eq!(right.enemies.len(), 1);
        approx_eq(left.enemies[0].body.pos.x, right.enemies[0].body.pos.x);
        approx_eq(left.enemies[0].body.pos.y, right.enemies[0].body.pos.y);
        approx_eq(left.enemies[0].body.vel.x, right.enemies[0].body.vel.x);
        approx_eq(left.enemies[0].body.vel.y, right.enemies[0].body.vel.y);
    }

    #[test]
    fn collision_finishes_episode() {
        let mut state = GameState::new(GameConfig::default(), Some(1));
        state.enemies.push(Enemy {
            body: CircleBody {
                pos: state.player.body.pos,
                vel: Vec2::default(),
                radius: state.player.body.radius,
            },
            remaining_life: 1.0,
        });

        let result = state.step_fixed(Action::Idle);

        assert!(result.done);
        assert_eq!(result.done_reason, DoneReason::Collision);
    }

    #[test]
    fn timeout_finishes_episode() {
        let mut config = GameConfig::default();
        config.max_episode_time = config.fixed_timestep;
        let mut state = GameState::new(config, Some(1));

        let result = state.step_fixed(Action::Idle);

        assert!(result.done);
        assert_eq!(result.done_reason, DoneReason::Timeout);
    }

    #[test]
    fn expired_enemies_count_as_evaded() {
        let mut state = GameState::new(GameConfig::default(), Some(1));
        state.enemies.push(Enemy {
            body: CircleBody {
                pos: Vec2 { x: 128.0, y: 0.0 },
                vel: Vec2::default(),
                radius: state.config.enemy_radius,
            },
            remaining_life: 0.0,
        });

        let result = state.step_fixed(Action::Idle);

        assert!(!result.done);
        assert_eq!(state.enemies_evaded, 1);
        assert!(result.reward > state.config.fixed_timestep);
    }
}
