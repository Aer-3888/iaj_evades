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
    pub world_width: f32,
    pub world_height: f32,
    pub corridor_top: f32,
    pub corridor_bottom: f32,
    pub goal_width: f32,
    pub start_margin: f32,
    pub player_radius: f32,
    pub enemy_radius_min: f32,
    pub enemy_radius_max: f32,
    pub player_speed: f32,
    pub enemy_speed_min: f32,
    pub enemy_speed_max: f32,
    pub enemy_count: usize,
    pub wall_enemy_radius: f32,
    pub wall_enemy_speed: f32,
    pub wall_enemy_pairs_per_wall: usize,
    pub fixed_timestep: f32,
    pub render_fps: u64,
    pub max_episode_time: f32,
    pub camera_lead: f32,
    pub default_seed: u64,
    pub background_color: Color,
    pub corridor_color: Color,
    pub corridor_line_color: Color,
    pub lane_marker_color: Color,
    pub player_color: Color,
    pub enemy_color: Color,
    pub goal_color: Color,
    pub text_color: Color,
    pub warning_color: Color,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            screen_width: 1000,
            screen_height: 500,
            world_width: 2000.0,
            world_height: 500.0,
            corridor_top: 60.0,
            corridor_bottom: 440.0,
            goal_width: 90.0,
            start_margin: 80.0,
            player_radius: 16.0,
            enemy_radius_min: 12.0,
            enemy_radius_max: 22.0,
            player_speed: 260.0,
            enemy_speed_min: 130.0,
            enemy_speed_max: 230.0,
            enemy_count: 70,
            wall_enemy_radius: 16.0,
            wall_enemy_speed: 180.0,
            wall_enemy_pairs_per_wall: 2,
            fixed_timestep: 1.0 / 60.0,
            render_fps: 60,
            max_episode_time: 60.0,
            camera_lead: 140.0,
            default_seed: 7,
            background_color: Color::rgb(16, 20, 28),
            corridor_color: Color::rgb(40, 52, 68),
            corridor_line_color: Color::rgb(90, 112, 138),
            lane_marker_color: Color::rgb(60, 76, 96),
            player_color: Color::rgb(97, 218, 251),
            enemy_color: Color::rgb(255, 126, 95),
            goal_color: Color::rgb(130, 218, 109),
            text_color: Color::rgb(235, 240, 245),
            warning_color: Color::rgb(255, 210, 120),
        }
    }
}

impl GameConfig {
    pub fn goal_x(&self) -> f32 {
        self.world_width - self.goal_width
    }

    pub fn corridor_height(&self) -> f32 {
        self.corridor_bottom - self.corridor_top
    }
}
