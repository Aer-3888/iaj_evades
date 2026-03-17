use std::time::{Duration, Instant};

use anyhow::Context;
use font8x8::{UnicodeFonts, BASIC_FONTS};
use minifb::{Key, KeyRepeat, Scale, Window, WindowOptions};
use rust_evades::{
    config::{Color, GameConfig},
    game::{DoneReason, GameState, Vec2},
};

pub struct LiveVisualizer {
    window: Window,
    buffer: Vec<u32>,
    previous: Instant,
    frame_target: Duration,
    show_hud: bool,
    followed_seed: u64,
}

pub struct VisualizerStats {
    pub episode: usize,
    pub total_steps: usize,
    pub epsilon: f32,
    pub mean_loss: f32,
    pub last_training_seed: u64,
}

pub struct VisualizerAgent<'a> {
    pub rank: usize,
    pub seed: u64,
    pub state: &'a GameState,
}

impl LiveVisualizer {
    pub fn new(config: &GameConfig) -> anyhow::Result<Self> {
        let window = Window::new(
            "Rust Evades Training",
            config.screen_width,
            config.screen_height,
            WindowOptions {
                resize: false,
                scale: Scale::X1,
                ..WindowOptions::default()
            },
        )
        .context("failed to create training visualization window")?;
        Ok(Self {
            window,
            buffer: vec![0; config.screen_width * config.screen_height],
            previous: Instant::now(),
            frame_target: Duration::from_secs_f64(1.0 / config.render_fps.max(1) as f64),
            show_hud: true,
            followed_seed: 0,
        })
    }

    pub fn is_open(&self) -> bool {
        self.window.is_open() && !self.window.is_key_down(Key::Escape)
    }

    pub fn set_followed_seed(&mut self, seed: u64) {
        self.followed_seed = seed;
    }

    pub fn render(
        &mut self,
        config: &GameConfig,
        stats: &VisualizerStats,
        agents: &[VisualizerAgent<'_>],
    ) -> anyhow::Result<bool> {
        for key in self.window.get_keys_pressed(KeyRepeat::No) {
            if key == Key::F3 {
                self.show_hud = !self.show_hud;
            }
        }

        fill(&mut self.buffer, config.background_color.to_u32());
        let camera = followed_camera(agents, self.followed_seed);
        draw_shared_world(&mut self.buffer, config, agents, camera);
        if self.show_hud {
            draw_header(&mut self.buffer, config, stats, agents, self.followed_seed);
            draw_footer(&mut self.buffer, config);
        }

        self.window
            .update_with_buffer(&self.buffer, config.screen_width, config.screen_height)
            .context("failed to present training visualization frame")?;

        let now = Instant::now();
        let elapsed = now.duration_since(self.previous);
        if elapsed < self.frame_target {
            std::thread::sleep(self.frame_target - elapsed);
        }
        self.previous = Instant::now();
        Ok(self.is_open())
    }
}

fn draw_header(
    buffer: &mut [u32],
    config: &GameConfig,
    stats: &VisualizerStats,
    agents: &[VisualizerAgent<'_>],
    followed_seed: u64,
) {
    let alive = agents.iter().filter(|agent| !agent.state.done).count();
    draw_text(
        buffer,
        config,
        16,
        12,
        &format!(
            "TRAINING LIVE  EP {}  STEPS {}  EPS {:.2}  LOSS {:.4}  LAST SEED {}  ALIVE {}/{}",
            stats.episode,
            stats.total_steps,
            stats.epsilon,
            stats.mean_loss,
            stats.last_training_seed,
            alive,
            agents.len(),
        ),
        config.text_color,
    );
    draw_text(
        buffer,
        config,
        16,
        32,
        &format!(
            "FOLLOWING SEED {}  every player shares this world view",
            followed_seed
        ),
        config.warning_color,
    );
}

fn draw_footer(buffer: &mut [u32], config: &GameConfig) {
    draw_text(
        buffer,
        config,
        16,
        config.screen_height as i32 - 18,
        "ALL PLAYERS ON ONE SCREEN  camera follows a random player each evaluation  F3 HUD  ESC close",
        config.text_color,
    );
}

fn followed_camera(agents: &[VisualizerAgent<'_>], followed_seed: u64) -> Vec2 {
    agents
        .iter()
        .find(|agent| agent.seed == followed_seed)
        .or_else(|| agents.first())
        .map(|agent| Vec2 {
            x: agent.state.player.body.pos.x,
            y: agent.state.player.body.pos.y,
        })
        .unwrap_or_default()
}

fn draw_shared_world(
    buffer: &mut [u32],
    config: &GameConfig,
    agents: &[VisualizerAgent<'_>],
    camera_center: Vec2,
) {
    let camera = Vec2 {
        x: camera_center.x - config.screen_width as f32 * 0.5,
        y: camera_center.y - config.screen_height as f32 * 0.5,
    };
    draw_grid(buffer, config, camera);

    for agent in agents {
        let enemy_color = enemy_color_for_agent(config, agent);
        for enemy in &agent.state.enemies {
            let x = (enemy.body.pos.x - camera.x) as i32;
            let y = (enemy.body.pos.y - camera.y) as i32;
            if on_screen(x, y, enemy.body.radius as i32, config) {
                draw_circle(
                    buffer,
                    config.screen_width,
                    config.screen_height,
                    x,
                    y,
                    enemy.body.radius as i32,
                    enemy_color,
                );
            }
        }
    }

    for agent in agents {
        let player_color = player_color_for_agent(config, agent);
        let x = (agent.state.player.body.pos.x - camera.x) as i32;
        let y = (agent.state.player.body.pos.y - camera.y) as i32;
        if on_screen(x, y, agent.state.player.body.radius as i32, config) {
            draw_circle(
                buffer,
                config.screen_width,
                config.screen_height,
                x,
                y,
                agent.state.player.body.radius as i32,
                player_color,
            );
            draw_label(
                buffer,
                config,
                x + 14,
                y - 12,
                &format!("{}", agent.rank),
                player_color,
            );
        }
    }
}

fn enemy_color_for_agent(config: &GameConfig, agent: &VisualizerAgent<'_>) -> u32 {
    match agent.state.done_reason {
        DoneReason::Collision => Color::rgb(180, 72, 72).to_u32(),
        DoneReason::Timeout => Color::rgb(220, 190, 96).to_u32(),
        DoneReason::None => tint_color(config.enemy_color, agent.rank),
    }
}

fn player_color_for_agent(config: &GameConfig, agent: &VisualizerAgent<'_>) -> u32 {
    if agent.state.done_reason == DoneReason::Collision {
        Color::rgb(210, 110, 110).to_u32()
    } else if agent.state.done_reason == DoneReason::Timeout {
        Color::rgb(255, 224, 140).to_u32()
    } else {
        tint_color(config.player_color, agent.rank)
    }
}

fn tint_color(base: Color, rank: usize) -> u32 {
    let shift = ((rank * 23) % 60) as i32 - 30;
    Color::rgb(
        adjust_channel(base.r, shift / 2),
        adjust_channel(base.g, shift),
        adjust_channel(base.b, -shift / 2),
    )
    .to_u32()
}

fn adjust_channel(value: u8, delta: i32) -> u8 {
    (value as i32 + delta).clamp(0, 255) as u8
}

fn on_screen(x: i32, y: i32, radius: i32, config: &GameConfig) -> bool {
    x + radius >= 0
        && x - radius < config.screen_width as i32
        && y + radius >= 0
        && y - radius < config.screen_height as i32
}

fn draw_grid(buffer: &mut [u32], config: &GameConfig, camera: Vec2) {
    let spacing = config.grid_spacing.max(8.0) as i32;
    let width = config.screen_width as i32;
    let height = config.screen_height as i32;
    let offset_x = ((camera.x.floor() as i32) % spacing + spacing) % spacing;
    let offset_y = ((camera.y.floor() as i32) % spacing + spacing) % spacing;
    let color = config.grid_color.to_u32();

    let mut x = -offset_x;
    while x < width {
        draw_rect(
            buffer,
            config.screen_width,
            config.screen_height,
            x,
            0,
            1,
            height,
            color,
        );
        x += spacing;
    }

    let mut y = -offset_y;
    while y < height {
        draw_rect(
            buffer,
            config.screen_width,
            config.screen_height,
            0,
            y,
            width,
            1,
            color,
        );
        y += spacing;
    }
}

fn fill(buffer: &mut [u32], color: u32) {
    buffer.fill(color);
}

fn draw_rect(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    color: u32,
) {
    let x0 = x.max(0) as usize;
    let y0 = y.max(0) as usize;
    let x1 = (x + w).min(width as i32).max(0) as usize;
    let y1 = (y + h).min(height as i32).max(0) as usize;

    for py in y0..y1 {
        let row = py * width;
        for px in x0..x1 {
            buffer[row + px] = color;
        }
    }
}

fn draw_circle(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    cx: i32,
    cy: i32,
    radius: i32,
    color: u32,
) {
    let min_x = (cx - radius).max(0) as usize;
    let max_x = (cx + radius).min(width as i32 - 1).max(0) as usize;
    let min_y = (cy - radius).max(0) as usize;
    let max_y = (cy + radius).min(height as i32 - 1).max(0) as usize;
    let radius_sq = radius * radius;

    for y in min_y..=max_y {
        let dy = y as i32 - cy;
        let row = y * width;
        for x in min_x..=max_x {
            let dx = x as i32 - cx;
            if dx * dx + dy * dy <= radius_sq {
                buffer[row + x] = color;
            }
        }
    }
}

fn draw_text(buffer: &mut [u32], config: &GameConfig, x: i32, y: i32, text: &str, color: Color) {
    draw_label(buffer, config, x, y, text, color.to_u32());
}

fn draw_label(buffer: &mut [u32], config: &GameConfig, x: i32, y: i32, text: &str, color: u32) {
    let mut cursor = x;
    for ch in text.chars() {
        if let Some(bitmap) = BASIC_FONTS.get(ch) {
            for (row, byte) in bitmap.iter().enumerate() {
                for col in 0..8 {
                    if (byte >> col) & 1 == 1 {
                        set_pixel(
                            buffer,
                            config.screen_width,
                            config.screen_height,
                            cursor + col,
                            y + row as i32,
                            color,
                        );
                    }
                }
            }
        }
        cursor += 8;
    }
}

fn set_pixel(buffer: &mut [u32], width: usize, height: usize, x: i32, y: i32, color: u32) {
    if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
        buffer[y as usize * width + x as usize] = color;
    }
}
