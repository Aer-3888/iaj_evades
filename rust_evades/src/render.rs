use std::time::{Duration, Instant};

use anyhow::Context;
use font8x8::{UnicodeFonts, BASIC_FONTS};
use minifb::{Key, KeyRepeat, Scale, Window, WindowOptions};

use crate::{
    config::{Color, GameConfig, MapDesign},
    game::{Action, DoneReason, GameState, Vec2},
    headless::ControllerMode,
    model_player::ModelController,
};

pub fn run_window(
    config: GameConfig,
    seed: Option<u64>,
    controller_mode: ControllerMode,
    mut model: Option<ModelController>,
) -> anyhow::Result<()> {
    if matches!(controller_mode, ControllerMode::Model) && model.is_none() {
        anyhow::bail!("`--model <path>` is required for non-headless model playback");
    }

    let mut state = GameState::new(config.clone(), seed);
    let mut window = Window::new(
        "Rust Evades",
        config.screen_width,
        config.screen_height,
        WindowOptions {
            resize: false,
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )
    .context("failed to create window")?;
    let mut buffer = vec![0; config.screen_width * config.screen_height];
    let mut model_enabled = matches!(controller_mode, ControllerMode::Model);
    if let Some(controller) = &mut model {
        controller.reset(&state);
    }
    let mut show_fps = false;
    let mut displayed_fps = 0.0f32;
    let timestep = Duration::from_secs_f32(config.fixed_timestep);
    let frame_target = Duration::from_secs_f64(1.0 / config.render_fps as f64);
    let mut previous = Instant::now();
    let mut accumulator = Duration::ZERO;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let frame = (now - previous).min(Duration::from_millis(250));
        previous = now;
        accumulator += frame;

        for key in window.get_keys_pressed(KeyRepeat::No) {
            match key {
                Key::R => {
                    state.reset(None);
                    if let Some(controller) = &mut model {
                        controller.reset(&state);
                    }
                }
                Key::B => {
                    if model.is_some() {
                        model_enabled = !model_enabled;
                    }
                }
                Key::F3 => show_fps = !show_fps,
                _ => {}
            }
        }

        let frame_secs = frame.as_secs_f32();
        if frame_secs > 0.0 {
            let instant_fps = 1.0 / frame_secs;
            displayed_fps = if displayed_fps == 0.0 {
                instant_fps
            } else {
                displayed_fps * 0.9 + instant_fps * 0.1
            };
        }

        while accumulator >= timestep {
            accumulator -= timestep;

            if state.done {
                if model_enabled
                    || matches!(
                        state.done_reason,
                        DoneReason::Collision | DoneReason::Timeout
                    )
                {
                    state.reset(None);
                    if let Some(controller) = &mut model {
                        controller.reset(&state);
                    }
                } else {
                    break;
                }
            }

            let action = if model_enabled {
                model
                    .as_mut()
                    .map(|controller| controller.choose_action(&state))
                    .unwrap_or(Action::Idle)
            } else {
                keyboard_action(&window)
            };
            state.step(action, Some(config.fixed_timestep));
        }

        let camera = camera_for_player(&state, &config);
        draw_world(
            &mut buffer,
            &state,
            &config,
            camera,
            model.is_some(),
            model_enabled,
            show_fps,
            displayed_fps,
        );
        window
            .update_with_buffer(&buffer, config.screen_width, config.screen_height)
            .context("failed to present frame")?;

        let elapsed = now.elapsed();
        if elapsed < frame_target {
            std::thread::sleep(frame_target - elapsed);
        }
    }

    Ok(())
}

fn keyboard_action(window: &Window) -> Action {
    let dx = i32::from(window.is_key_down(Key::D) || window.is_key_down(Key::Right))
        - i32::from(window.is_key_down(Key::A) || window.is_key_down(Key::Left));
    let dy = i32::from(window.is_key_down(Key::S) || window.is_key_down(Key::Down))
        - i32::from(window.is_key_down(Key::W) || window.is_key_down(Key::Up));

    match (dx, dy) {
        (0, 0) => Action::Idle,
        (0, -1) => Action::Up,
        (0, 1) => Action::Down,
        (-1, 0) => Action::Left,
        (1, 0) => Action::Right,
        (1, -1) => Action::UpRight,
        (1, 1) => Action::DownRight,
        (-1, -1) => Action::UpLeft,
        (-1, 1) => Action::DownLeft,
        _ => Action::Idle,
    }
}

fn camera_for_player(state: &GameState, config: &GameConfig) -> Vec2 {
    match config.map_design {
        MapDesign::Open | MapDesign::Arena => Vec2 {
            x: state.player.body.pos.x - config.screen_width as f32 * 0.5,
            y: state.player.body.pos.y - config.screen_height as f32 * 0.5,
        },
        MapDesign::Closed => {
            let target =
                state.player.body.pos.x - config.screen_width as f32 * 0.35 + config.camera_lead;
            Vec2 {
                x: target.clamp(0.0, config.world_width - config.screen_width as f32),
                y: 0.0,
            }
        }
    }
}

fn draw_world(
    buffer: &mut [u32],
    state: &GameState,
    config: &GameConfig,
    camera: Vec2,
    has_model: bool,
    model_enabled: bool,
    show_fps: bool,
    displayed_fps: f32,
) {
    fill(buffer, config.background_color.to_u32());

    if matches!(config.map_design, MapDesign::Open | MapDesign::Arena) {
        draw_grid(buffer, config, camera);
    } else {
        // Draw Closed map (corridor)
        draw_rect(
            buffer,
            config.screen_width,
            config.screen_height,
            0,
            config.corridor_top as i32,
            config.screen_width as i32,
            config.corridor_height() as i32,
            config.corridor_color.to_u32(),
        );
        draw_rect(
            buffer,
            config.screen_width,
            config.screen_height,
            0,
            config.corridor_top as i32,
            config.screen_width as i32,
            3,
            config.corridor_line_color.to_u32(),
        );
        draw_rect(
            buffer,
            config.screen_width,
            config.screen_height,
            0,
            config.corridor_bottom as i32 - 3,
            config.screen_width as i32,
            3,
            config.corridor_line_color.to_u32(),
        );

        let marker_spacing = 180;
        let marker_width = 70;
        let marker_height = 8;
        let marker_y =
            config.corridor_top as i32 + config.corridor_height() as i32 / 2 - marker_height / 2;
        let first_marker = ((camera.x as i32 / marker_spacing) * marker_spacing) - marker_spacing;
        let end_marker = camera.x as i32 + config.screen_width as i32 + marker_spacing;
        let mut world_x = first_marker;
        while world_x <= end_marker {
            let screen_x = world_x - camera.x as i32;
            draw_rect(
                buffer,
                config.screen_width,
                config.screen_height,
                screen_x,
                marker_y,
                marker_width,
                marker_height,
                config.lane_marker_color.to_u32(),
            );
            world_x += marker_spacing;
        }

        let goal_screen_x = (config.goal_x() - camera.x) as i32;
        draw_rect(
            buffer,
            config.screen_width,
            config.screen_height,
            goal_screen_x,
            config.corridor_top as i32,
            config.goal_width as i32,
            config.corridor_height() as i32,
            config.goal_color.to_u32(),
        );
    }

    for enemy in &state.enemies {
        let x = (enemy.body.pos.x - camera.x) as i32;
        let y = (enemy.body.pos.y - camera.y) as i32;
        if x + enemy.body.radius as i32 >= 0
            && x - (enemy.body.radius as i32) < config.screen_width as i32
            && y + enemy.body.radius as i32 >= 0
            && y - (enemy.body.radius as i32) < config.screen_height as i32
        {
            draw_circle(
                buffer,
                config.screen_width,
                config.screen_height,
                x,
                y,
                enemy.body.radius as i32,
                config.enemy_color.to_u32(),
            );
        }
    }

    draw_circle(
        buffer,
        config.screen_width,
        config.screen_height,
        (state.player.body.pos.x - camera.x) as i32,
        (state.player.body.pos.y - camera.y) as i32,
        state.player.body.radius as i32,
        config.player_color.to_u32(),
    );

    if matches!(config.map_design, MapDesign::Open | MapDesign::Arena) {
        draw_text(
            buffer,
            config,
            20,
            18,
            &format!(
                "TIME {:.2}/{:.0}",
                state.elapsed_time, state.config.max_episode_time
            ),
            config.text_color,
        );
        draw_text(
            buffer,
            config,
            20,
            46,
            &format!("EVADES {}", state.enemies_evaded),
            config.text_color,
        );
        draw_text(
            buffer,
            config,
            220,
            18,
            &format!("BEST {:.2}s", state.best_survival_ever),
            config.text_color,
        );
        draw_text(
            buffer,
            config,
            220,
            46,
            &format!("ACTIVE {}", state.enemies.len()),
            config.text_color,
        );
        draw_text(
            buffer,
            config,
            420,
            18,
            &format!("SEED {}", state.base_seed),
            config.text_color,
        );
    } else {
        let goal_total = config.goal_x() - config.start_margin;
        draw_text(
            buffer,
            config,
            20,
            18,
            &format!("PROGRESS {:.0}/{:.0}", state.best_progress(), goal_total),
            config.text_color,
        );
        draw_text(
            buffer,
            config,
            20,
            46,
            &format!("DEATHS {}", state.total_deaths),
            config.text_color,
        );
        draw_text(
            buffer,
            config,
            220,
            18,
            &format!("BEST {:.0}", state.best_survival_ever), // Reusing best_survival_ever for best_progress or just leaving it
            config.text_color,
        );
        draw_text(
            buffer,
            config,
            220,
            46,
            &format!("SEED {}", state.base_seed),
            config.text_color,
        );
    }

    draw_text(
        buffer,
        config,
        20,
        config.screen_height as i32 - 30,
        "MOVE WASD/ARROWS  RESTART R  TOGGLE MODEL B  TOGGLE FPS F3  QUIT ESC",
        config.text_color,
    );
    draw_text(
        buffer,
        config,
        config.screen_width as i32 - 180,
        18,
        if model_enabled {
            "MODE MODEL"
        } else if has_model {
            "MODE MANUAL"
        } else {
            "MODE MANUAL ONLY"
        },
        config.warning_color,
    );
    if show_fps {
        draw_text(
            buffer,
            config,
            config.screen_width as i32 - 180,
            38,
            &format!("FPS {:.1}", displayed_fps),
            config.text_color,
        );
    }

    if state.done {
        let message = match state.done_reason {
            DoneReason::Collision => "HIT - PRESS R TO TRY AGAIN",
            DoneReason::Timeout => {
                if matches!(config.map_design, MapDesign::Open | MapDesign::Arena) {
                    "SURVIVED FULL TIMER - PRESS R TO RUN AGAIN"
                } else {
                    "TIMEOUT - PRESS R TO TRY AGAIN"
                }
            }
            DoneReason::Goal => "GOAL REACHED - PRESS R TO RUN AGAIN",
            DoneReason::None => "",
        };
        if !message.is_empty() {
            draw_text_centered(buffer, config, 28, message, config.warning_color);
        }
    }
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
                            color.to_u32(),
                        );
                    }
                }
            }
        }
        cursor += 8;
    }
}

fn draw_text_centered(buffer: &mut [u32], config: &GameConfig, y: i32, text: &str, color: Color) {
    let text_width = (text.chars().count() as i32) * 8;
    let x = (config.screen_width as i32 - text_width) / 2;
    draw_text(buffer, config, x, y, text, color);
}

fn set_pixel(buffer: &mut [u32], width: usize, height: usize, x: i32, y: i32, color: u32) {
    if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
        buffer[y as usize * width + x as usize] = color;
    }
}
