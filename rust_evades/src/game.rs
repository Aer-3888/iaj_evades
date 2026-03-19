use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::config::{GameConfig, MapDesign};

const RIGHTWARD_FITNESS_BONUS_PER_UNIT: f32 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Player {
    pub body: CircleBody,
}

impl Player {
    fn apply_action(&mut self, action: Action, speed: f32, dt: f32, config: &GameConfig) {
        let (dx, dy) = action.vector();
        let direction = Vec2 { x: dx, y: dy }.normalized_or_zero();
        self.body.vel = Vec2 {
            x: direction.x * speed,
            y: direction.y * speed,
        };
        self.body.pos.x += self.body.vel.x * dt;
        self.body.pos.y += self.body.vel.y * dt;

        if config.map_design == MapDesign::Closed {
            self.body.pos.x = self.body.pos.x.clamp(
                self.body.radius,
                config.world_width - self.body.radius
            );
            self.body.pos.y = self.body.pos.y.clamp(
                config.corridor_top + self.body.radius,
                config.corridor_bottom - self.body.radius
            );
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Enemy {
    pub body: CircleBody,
    pub remaining_life: f32,
}

impl Enemy {
    fn update(&mut self, dt: f32, config: &GameConfig) {
        self.body.pos.x += self.body.vel.x * dt;
        self.body.pos.y += self.body.vel.y * dt;
        
        if config.map_design == MapDesign::Open {
            self.remaining_life -= dt;
        } else {
            // Bounce logic for Closed map (from d138710)
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct StepResult {
    pub reward: f32,
    pub done: bool,
    pub done_reason: DoneReason,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EpisodeReport {
    pub elapsed_time: f32,
    pub enemies_evaded: u32,
    pub total_reward: f32,
    pub done_reason: DoneReason,
    pub survived_full_episode: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GameState {
    pub config: GameConfig,
    pub base_seed: u64,
    pub total_deaths: u32,
    pub total_timeouts: u32,
    pub total_wins: u32,
    pub episode_index: u32,
    pub best_survival_ever: f32,
    pub best_progress_ever: f32,
    pub done: bool,
    pub done_reason: DoneReason,
    pub elapsed_time: f32,
    pub last_reward: f32,
    pub episode_return: f32,
    pub player: Player,
    pub enemies: Vec<Enemy>,
    pub enemies_evaded: u32,
    #[serde(skip, default = "default_rng")]
    rng: ChaCha8Rng,
    pub next_spawn_in: f32,
    pub best_x: f32,
    pub current_level: usize,
    pub map: Vec<Vec<u8>>,
}

fn default_rng() -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(0)
}
impl GameState {
    pub fn new(config: GameConfig, seed: Option<u64>) -> Self {
        let base_seed = seed.unwrap_or(config.default_seed);
        let rng = ChaCha8Rng::seed_from_u64(base_seed);
        
        let mut state = Self {
            config,
            base_seed,
            total_deaths: 0,
            total_timeouts: 0,
            total_wins: 0,
            episode_index: 0,
            best_survival_ever: 0.0,
            done: false,
            done_reason: DoneReason::None,
            elapsed_time: 0.0,
            last_reward: 0.0,
            episode_return: 0.0,
            player: Player {
                body: CircleBody {
                    pos: Vec2::default(),
                    vel: Vec2::default(),
                    radius: 0.0,
                },
            },
            enemies: Vec::new(),
            enemies_evaded: 0,
            rng,
            next_spawn_in: 0.0,
            best_x: 0.0,
            current_level: 0,
            best_progress_ever: 0.0,
            map: Vec::new(),
        };
        state.reset(None);
        state
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        let actual_seed = seed.unwrap_or(self.base_seed);
        self.rng = ChaCha8Rng::seed_from_u64(actual_seed);
        self.current_level = 0;
        self.map = Vec::new();
        
        if self.config.map_design == MapDesign::Open {
            self.player = Player {
                body: CircleBody {
                    pos: Vec2::default(),
                    vel: Vec2::default(),
                    radius: self.config.player_radius,
                },
            };
            self.enemies.clear();
        } else {
            // Closed map reset (from d138710)
            let start_y = self.config.corridor_top + self.config.corridor_height() * 0.5;
            self.player = Player {
                body: CircleBody {
                    pos: Vec2 {
                        x: self.config.start_margin,
                        y: start_y,
                    },
                    vel: Vec2::default(),
                    radius: self.config.player_radius,
                },
            };
            self.enemies = self.spawn_initial_enemies_closed();
        }

        self.enemies_evaded = 0;
        self.done = false;
        self.done_reason = DoneReason::None;
        self.elapsed_time = 0.0;
        self.last_reward = 0.0;
        self.episode_return = 0.0;
        self.next_spawn_in = self.random_spawn_interval();
        self.best_x = self.player.body.pos.x;
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
        let prev_x = self.player.body.pos.x;
        let previous_best = self.best_x;

        self.player
            .apply_action(action, self.config.player_speed, delta, &self.config);
        
        self.elapsed_time += delta;
        self.best_x = self.best_x.max(self.player.body.pos.x);

        for enemy in &mut self.enemies {
            enemy.update(delta, &self.config);
        }

        if self.config.map_design == MapDesign::Open {
            self.spawn_enemies_open(delta);
        }

        let dx = self.player.body.pos.x - prev_x;
        let mut reward = if self.config.map_design == MapDesign::Open {
            self.config.survival_reward_per_second * delta
                + dx.max(0.0) * self.config.rightward_reward_per_unit
        } else {
            // Closed map reward (from d138710)
            (self.best_x - previous_best) * 0.02
        };

        if (self.config.map_design == MapDesign::Closed && self.touched_wall()) || self.collided() {
            self.done = true;
            self.done_reason = DoneReason::Collision;
            self.total_deaths += 1;
            reward += if self.config.map_design == MapDesign::Open {
                self.config.collision_penalty
            } else {
                -150.0 // Penalty from d138710
            };
        } else if self.config.map_design == MapDesign::Closed && self.player.body.pos.x + self.player.body.radius >= self.config.goal_x() {
            self.done = true;
            self.done_reason = DoneReason::Goal;
            self.total_wins += 1;
            reward += 250.0; // Goal bonus from d138710
        } else {
            if self.config.map_design == MapDesign::Open {
                let evaded = self.collect_expired_enemies();
                self.enemies_evaded += evaded;
                reward += evaded as f32 * self.config.enemy_evade_reward;
            }

            if self.elapsed_time >= self.config.max_episode_time {
                self.done = true;
                self.done_reason = DoneReason::Timeout;
                if self.config.map_design == MapDesign::Open {
                    self.total_timeouts += 1;
                    reward += self.config.timeout_bonus;
                } else {
                    reward -= 50.0; // Penalty from d138710
                }
            }
        }

        self.best_survival_ever = self.best_survival_ever.max(self.elapsed_time);
        self.best_progress_ever = self.best_progress_ever.max(self.best_progress());
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

    pub fn best_progress(&self) -> f32 {
        (self.best_x - self.config.start_margin).max(0.0)
    }

    pub fn fitness(&self) -> f32 {
        if self.config.map_design == MapDesign::Open {
            self.episode_return + self.player.body.pos.x.max(0.0) * RIGHTWARD_FITNESS_BONUS_PER_UNIT
        } else {
            // Closed map fitness (from d138710)
            let mut fitness = (self.best_x - self.config.start_margin).max(0.0) - self.elapsed_time * 2.0;
            match self.done_reason {
                DoneReason::Goal => fitness += 2500.0,
                DoneReason::Collision => fitness -= 400.0,
                DoneReason::Timeout => fitness -= 150.0,
                DoneReason::None => {}
            }
            fitness
        }
    }

    pub fn episode_report(&self) -> EpisodeReport {
        EpisodeReport {
            elapsed_time: self.elapsed_time,
            enemies_evaded: self.enemies_evaded,
            total_reward: self.episode_return,
            done_reason: self.done_reason,
            survived_full_episode: self.done_reason == DoneReason::Timeout || self.done_reason == DoneReason::Goal,
        }
    }

    fn spawn_enemies_open(&mut self, delta: f32) {
        self.next_spawn_in -= delta;
        while self.next_spawn_in <= 0.0 {
            let enemy = self.spawn_enemy_open();
            self.enemies.push(enemy);
            self.next_spawn_in += self.random_spawn_interval();
        }
    }

    fn spawn_enemy_open(&mut self) -> Enemy {
        let direction = self.sample_spawn_direction_open();
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

    fn spawn_initial_enemies_closed(&mut self) -> Vec<Enemy> {
        let wall_enemy_count = self.config.wall_enemy_pairs_per_wall * 4;
        let mut enemies = Vec::with_capacity(self.config.enemy_count + wall_enemy_count);
        enemies.extend(self.spawn_wall_enemies_closed());
        
        let min_x = 0.0;
        let end_safe_x = self.config.goal_x() - 180.0;
        let player_spawn_clearance = self.config.player_radius * 2.0;

        for _ in 0..self.config.enemy_count {
            let radius = self.rng.gen_range(self.config.enemy_radius_min..=self.config.enemy_radius_max);
            let mut spawned = None;

            for _ in 0..500 {
                let x = self.rng.gen_range(min_x + radius..end_safe_x);
                let y = self.rng.gen_range(self.config.corridor_top + radius..self.config.corridor_bottom - radius);
                let speed = self.rng.gen_range(self.config.enemy_speed_min..=self.config.enemy_speed_max);
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
                    remaining_life: f32::INFINITY,
                };

                let far_enough_from_player = self.player.body.distance_squared_to(&candidate.body)
                    > (candidate.body.radius + player_spawn_clearance) * (candidate.body.radius + player_spawn_clearance);
                
                let clear_of_other_enemies = enemies.iter().all(|other| {
                    let dx = candidate.body.pos.x - other.body.pos.x;
                    let dy = candidate.body.pos.y - other.body.pos.y;
                    let min_gap = candidate.body.radius + other.body.radius + 28.0;
                    dx * dx + dy * dy > min_gap * min_gap
                });

                if far_enough_from_player && clear_of_other_enemies {
                    spawned = Some(candidate);
                    break;
                }
            }

            if let Some(enemy) = spawned {
                enemies.push(enemy);
            }
        }
        enemies
    }

    fn spawn_wall_enemies_closed(&self) -> Vec<Enemy> {
        let mut enemies = Vec::with_capacity(self.config.wall_enemy_pairs_per_wall * 4);
        let lane_start = self.config.start_margin + 260.0;
        let lane_end = self.config.goal_x() - 220.0;
        let spacing = (lane_end - lane_start) / (self.config.wall_enemy_pairs_per_wall.max(1) as f32 + 1.0);
        let top_y = self.config.corridor_top + self.config.wall_enemy_radius;
        let bottom_y = self.config.corridor_bottom - self.config.wall_enemy_radius;

        for index in 0..self.config.wall_enemy_pairs_per_wall {
            let anchor_x = lane_start + spacing * (index as f32 + 1.0);
            let offset = 48.0;

            // Top pair
            enemies.push(Enemy {
                body: CircleBody {
                    pos: Vec2 { x: anchor_x - offset, y: top_y },
                    vel: Vec2 { x: self.config.wall_enemy_speed, y: 0.0 },
                    radius: self.config.wall_enemy_radius,
                },
                remaining_life: f32::INFINITY,
            });
            enemies.push(Enemy {
                body: CircleBody {
                    pos: Vec2 { x: anchor_x + offset, y: top_y },
                    vel: Vec2 { x: -self.config.wall_enemy_speed, y: 0.0 },
                    radius: self.config.wall_enemy_radius,
                },
                remaining_life: f32::INFINITY,
            });

            // Bottom pair
            enemies.push(Enemy {
                body: CircleBody {
                    pos: Vec2 { x: anchor_x - offset, y: bottom_y },
                    vel: Vec2 { x: self.config.wall_enemy_speed, y: 0.0 },
                    radius: self.config.wall_enemy_radius,
                },
                remaining_life: f32::INFINITY,
            });
            enemies.push(Enemy {
                body: CircleBody {
                    pos: Vec2 { x: anchor_x + offset, y: bottom_y },
                    vel: Vec2 { x: -self.config.wall_enemy_speed, y: 0.0 },
                    radius: self.config.wall_enemy_radius,
                },
                remaining_life: f32::INFINITY,
            });
        }
        enemies
    }

    fn random_spawn_interval(&mut self) -> f32 {
        self.rng
            .gen_range(self.config.enemy_spawn_interval_min..=self.config.enemy_spawn_interval_max)
    }

    fn sample_spawn_direction_open(&mut self) -> Vec2 {
        let candidate_count = self.config.enemy_spawn_density_sample_count.max(1);
        let phase = self.rng.gen_range(0.0..std::f32::consts::TAU);
        let probe_distance =
            self.config.enemy_spawn_distance * self.config.enemy_spawn_density_probe_distance_scale;
        let mut best_density = f32::INFINITY;
        let mut best_directions = Vec::with_capacity(candidate_count);

        for index in 0..candidate_count {
            let angle = phase + std::f32::consts::TAU * index as f32 / candidate_count as f32;
            let direction = Vec2 {
                x: angle.cos(),
                y: angle.sin(),
            };
            let probe_center = Vec2 {
                x: self.player.body.pos.x + direction.x * probe_distance,
                y: self.player.body.pos.y + direction.y * probe_distance,
            };
            let density = self.local_enemy_density(probe_center);

            if density + 1.0e-5 < best_density {
                best_density = density;
                best_directions.clear();
                best_directions.push(direction);
            } else if (density - best_density).abs() <= 1.0e-5 {
                best_directions.push(direction);
            }
        }

        best_directions
            .choose(&mut self.rng)
            .copied()
            .unwrap_or(Vec2 { x: 1.0, y: 0.0 })
    }

    fn local_enemy_density(&self, point: Vec2) -> f32 {
        let probe_radius = self
            .config
            .enemy_spawn_density_probe_radius
            .max(self.config.enemy_radius);
        let probe_radius_sq = probe_radius * probe_radius;

        self.enemies
            .iter()
            .map(|enemy| {
                let dx = point.x - enemy.body.pos.x;
                let dy = point.y - enemy.body.pos.y;
                let distance_sq = dx * dx + dy * dy;
                if distance_sq >= probe_radius_sq {
                    0.0
                } else {
                    1.0 - distance_sq / probe_radius_sq
                }
            })
            .sum()
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

    fn touched_wall(&self) -> bool {
        self.player.body.pos.x - self.player.body.radius <= 0.0
            || self.player.body.pos.x + self.player.body.radius >= self.config.world_width
            || self.player.body.pos.y - self.player.body.radius <= self.config.corridor_top
            || self.player.body.pos.y + self.player.body.radius >= self.config.corridor_bottom
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
    let mut pos = Vec2 {
        x: player.body.pos.x + direction.x * config.player_speed * dt,
        y: player.body.pos.y + direction.y * config.player_speed * dt,
    };

    if config.map_design == MapDesign::Closed {
        pos.x = pos.x.clamp(player.body.radius, config.world_width - player.body.radius);
        pos.y = pos.y.clamp(config.corridor_top + player.body.radius, config.corridor_bottom - player.body.radius);
    }
    pos
}
