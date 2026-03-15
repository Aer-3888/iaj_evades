#[derive(Clone, Copy, Debug)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub const fn to_u32(self) -> u32 {
        ((self.r as u32) << 16) | ((self.g as u32) << 8) | self.b as u32
    }
}

#[derive(Clone, Debug)]
pub struct GameConfig {
    pub screen_width: usize,
    pub screen_height: usize,
    pub player_radius: f32,
    pub enemy_radius: f32,
    pub player_speed: f32,
    pub enemy_speed: f32,
    pub enemy_spawn_distance: f32,
    pub enemy_spawn_interval_min: f32,
    pub enemy_spawn_interval_max: f32,
    pub enemy_lifetime: f32,
    pub fixed_timestep: f32,
    pub render_fps: u64,
    pub max_episode_time: f32,
    pub grid_spacing: f32,
    pub survival_reward_per_second: f32,
    pub enemy_evade_reward: f32,
    pub collision_penalty: f32,
    pub timeout_bonus: f32,
    pub default_seed: u64,
    pub background_color: Color,
    pub grid_color: Color,
    pub player_color: Color,
    pub enemy_color: Color,
    pub text_color: Color,
    pub warning_color: Color,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            screen_width: 1000,
            screen_height: 500,
            player_radius: 16.0,
            enemy_radius: 14.0,
            player_speed: 260.0,
            enemy_speed: 210.0,
            enemy_spawn_distance: 144.0,
            enemy_spawn_interval_min: 0.45,
            enemy_spawn_interval_max: 1.10,
            enemy_lifetime: 2.4,
            fixed_timestep: 1.0 / 60.0,
            render_fps: 60,
            max_episode_time: 20.0,
            grid_spacing: 64.0,
            survival_reward_per_second: 1.0,
            enemy_evade_reward: 0.35,
            collision_penalty: -6.0,
            timeout_bonus: 6.0,
            default_seed: 7,
            background_color: Color::rgb(16, 20, 28),
            grid_color: Color::rgb(42, 52, 66),
            player_color: Color::rgb(97, 218, 251),
            enemy_color: Color::rgb(255, 126, 95),
            text_color: Color::rgb(235, 240, 245),
            warning_color: Color::rgb(255, 210, 120),
        }
    }
}

impl GameConfig {
    pub fn ray_length(&self) -> f32 {
        self.player_radius * 5.0
    }
}
