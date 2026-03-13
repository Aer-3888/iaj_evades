from __future__ import annotations

import argparse
import sys

import pygame

from config import GameConfig
from entities import Action, DELTA_TO_ACTION
from game import GameEnvironment
from genetic_ai import LinearPolicyGenome, observation_size_for_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play or watch the dodge runner game.")
    parser.add_argument("--ai", type=str, default=None, help="Path to a saved genome JSON file")
    return parser


def action_from_keyboard() -> Action:
    pressed = pygame.key.get_pressed()
    dx = int(pressed[pygame.K_d] or pressed[pygame.K_RIGHT]) - int(pressed[pygame.K_a] or pressed[pygame.K_LEFT])
    dy = int(pressed[pygame.K_s] or pressed[pygame.K_DOWN]) - int(pressed[pygame.K_w] or pressed[pygame.K_UP])
    return DELTA_TO_ACTION[(dx, dy)]


def camera_x_for_player(env: GameEnvironment, config: GameConfig) -> float:
    target = env.player.x - config.screen_width * 0.35 + config.camera_lead
    return max(0.0, min(target, config.world_width - config.screen_width))


def draw_world(screen: pygame.Surface, env: GameEnvironment, config: GameConfig, camera_x: float, font: pygame.font.Font) -> None:
    screen.fill(config.background_color)

    corridor_rect = pygame.Rect(
        0,
        config.corridor_top,
        config.screen_width,
        config.corridor_height,
    )
    pygame.draw.rect(screen, config.corridor_color, corridor_rect)
    pygame.draw.line(screen, config.corridor_line_color, (0, config.corridor_top), (config.screen_width, config.corridor_top), 3)
    pygame.draw.line(screen, config.corridor_line_color, (0, config.corridor_bottom), (config.screen_width, config.corridor_bottom), 3)

    marker_spacing = 180
    marker_width = 70
    marker_height = 8
    marker_y = config.corridor_top + config.corridor_height // 2 - marker_height // 2
    first_marker = int(camera_x // marker_spacing) * marker_spacing
    for world_x in range(first_marker, int(camera_x + config.screen_width) + marker_spacing, marker_spacing):
        screen_x = int(world_x - camera_x)
        pygame.draw.rect(
            screen,
            config.lane_marker_color,
            pygame.Rect(screen_x, marker_y, marker_width, marker_height),
            border_radius=4,
        )

    goal_screen_x = int(config.goal_x - camera_x)
    pygame.draw.rect(
        screen,
        config.goal_color,
        pygame.Rect(goal_screen_x, config.corridor_top, config.goal_width, config.corridor_height),
        border_radius=8,
    )

    for enemy in env.enemies:
        screen_pos = (int(enemy.x - camera_x), int(enemy.y))
        pygame.draw.circle(screen, enemy.color, screen_pos, int(enemy.radius))

    player_pos = (int(env.player.x - camera_x), int(env.player.y))
    pygame.draw.circle(screen, env.player.color, player_pos, int(env.player.radius))

    progress_text = font.render(
        f"Progress {env.best_progress:.0f}/{config.goal_x - config.start_margin:.0f}",
        True,
        config.text_color,
    )
    deaths_text = font.render(f"Deaths {env.total_deaths}", True, config.text_color)
    best_text = font.render(f"Best {env.best_progress_ever:.0f}", True, config.text_color)
    seed_text = font.render(f"Seed {env.base_seed}", True, config.text_color)

    screen.blit(progress_text, (20, 18))
    screen.blit(deaths_text, (20, 46))
    screen.blit(best_text, (220, 18))
    screen.blit(seed_text, (220, 46))

    controls_text = font.render("Move: WASD / Arrows   Restart: R   Toggle AI: T   Quit: Esc", True, config.text_color)
    screen.blit(controls_text, (20, config.screen_height - 34))

    if env.done and env.done_reason == "goal":
        banner = font.render("Goal reached! Press R to run again.", True, config.warning_color)
        banner_rect = banner.get_rect(center=(config.screen_width // 2, 28))
        screen.blit(banner, banner_rect)


def main() -> None:
    args = build_parser().parse_args()
    pygame.init()
    config = GameConfig()
    screen = pygame.display.set_mode((config.screen_width, config.screen_height))
    pygame.display.set_caption("Genetic Dodge Runner Prototype")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 22)

    env = GameEnvironment(config=config)
    ai_genome: LinearPolicyGenome | None = None
    ai_enabled = False
    if args.ai is not None:
        ai_genome = LinearPolicyGenome.load(args.ai, expected_input_size=observation_size_for_config(config))
        ai_enabled = True

    accumulator = 0.0
    running = True

    while running:
        frame_time = clock.tick(config.render_fps) / 1000.0
        accumulator += min(frame_time, 0.25)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                elif event.key == pygame.K_t and ai_genome is not None:
                    ai_enabled = not ai_enabled

        while accumulator >= config.fixed_timestep:
            accumulator -= config.fixed_timestep

            if env.done:
                if ai_enabled or env.done_reason in {"collision", "timeout"}:
                    env.reset()
                else:
                    break

            action = ai_genome.act(env.get_observation()) if ai_enabled and ai_genome is not None else action_from_keyboard()
            env.step(action)

        camera_x = camera_x_for_player(env, config)
        draw_world(screen, env, config, camera_x, font)
        if ai_genome is not None:
            mode_text = font.render(f"Mode {'AI' if ai_enabled else 'Manual'}", True, config.warning_color)
            screen.blit(mode_text, (config.screen_width - 165, 18))
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
