from __future__ import annotations

import math
import random

from config import GameConfig
from entities import Action, Enemy, Player


class GameEnvironment:
    def __init__(self, config: GameConfig | None = None, seed: int | None = None) -> None:
        self.config = config or GameConfig()
        self.base_seed = self.config.default_seed if seed is None else seed
        self.total_deaths = 0
        self.total_wins = 0
        self.episode_index = 0
        self.best_progress_ever = 0.0
        self.done = False
        self.done_reason = ""
        self.elapsed_time = 0.0
        self.best_x = 0.0
        self.last_reward = 0.0
        self.player = Player(0.0, 0.0, self.config.player_radius, self.config.player_color)
        self.enemies: list[Enemy] = []
        self.reset()

    def reset(self, seed: int | None = None) -> list[float]:
        actual_seed = self.base_seed if seed is None else seed
        self.rng = random.Random(actual_seed)
        start_y = self.config.corridor_top + self.config.corridor_height / 2.0
        self.player = Player(
            x=float(self.config.start_margin),
            y=float(start_y),
            radius=float(self.config.player_radius),
            color=self.config.player_color,
        )
        self.enemies = self._spawn_enemies()
        self.done = False
        self.done_reason = ""
        self.elapsed_time = 0.0
        self.best_x = self.player.x
        self.last_reward = 0.0
        self.episode_index += 1
        return self.get_observation()

    def step(self, action: Action, dt: float | None = None) -> tuple[list[float], float, bool, dict[str, float | bool | str]]:
        if self.done:
            return self.get_observation(), 0.0, True, self.get_info()

        delta = self.config.fixed_timestep if dt is None else dt
        previous_best = self.best_x

        self.player.apply_action(
            action=action,
            speed=self.config.player_speed,
            dt=delta,
            min_x=0.0,
            max_x=float(self.config.world_width),
            min_y=float(self.config.corridor_top),
            max_y=float(self.config.corridor_bottom),
        )

        for enemy in self.enemies:
            enemy.update(
                dt=delta,
                min_x=0.0,
                max_x=float(self.config.world_width),
                min_y=float(self.config.corridor_top),
                max_y=float(self.config.corridor_bottom),
            )

        self.elapsed_time += delta
        self.best_x = max(self.best_x, self.player.x)
        self.best_progress_ever = max(self.best_progress_ever, self.best_progress)

        reward = (self.best_x - previous_best) * 0.02

        if self._collided():
            self.done = True
            self.done_reason = "collision"
            self.total_deaths += 1
            reward -= 150.0
        elif self.player.x + self.player.radius >= self.config.goal_x:
            self.done = True
            self.done_reason = "goal"
            self.total_wins += 1
            reward += 250.0
        elif self.elapsed_time >= self.config.max_episode_time:
            self.done = True
            self.done_reason = "timeout"
            reward -= 50.0

        self.last_reward = reward
        return self.get_observation(), reward, self.done, self.get_info()

    @property
    def best_progress(self) -> float:
        return max(0.0, self.best_x - self.config.start_margin)

    @property
    def progress_ratio(self) -> float:
        total_distance = self.config.goal_x - self.config.start_margin
        return max(0.0, min(1.0, self.best_progress / total_distance))

    def is_done(self) -> bool:
        return self.done

    def get_fitness(self) -> float:
        fitness = self.best_progress - self.elapsed_time * 2.0
        if self.done_reason == "goal":
            fitness += 2500.0
        elif self.done_reason == "collision":
            fitness -= 400.0
        elif self.done_reason == "timeout":
            fitness -= 150.0
        return fitness

    def get_observation(self) -> list[float]:
        observation = [
            self.player.x / self.config.world_width,
            self.player.y / self.config.world_height,
            self.player.vx / self.config.player_speed,
            self.player.vy / self.config.player_speed,
            max(0.0, self.config.goal_x - self.player.x) / self.config.world_width,
            self.elapsed_time / self.config.max_episode_time,
        ]

        max_enemy_speed = self.config.enemy_speed_max
        for enemy in self.enemies:
            observation.extend(
                [
                    (enemy.x - self.player.x) / self.config.world_width,
                    (enemy.y - self.player.y) / self.config.corridor_height,
                    enemy.vx / max_enemy_speed,
                    enemy.vy / max_enemy_speed,
                    enemy.radius / self.config.enemy_radius_max,
                ]
            )
        return observation

    def get_info(self) -> dict[str, float | bool | str]:
        return {
            "done": self.done,
            "done_reason": self.done_reason,
            "elapsed_time": self.elapsed_time,
            "best_progress": self.best_progress,
            "progress_ratio": self.progress_ratio,
            "fitness": self.get_fitness(),
            "deaths": float(self.total_deaths),
            "wins": float(self.total_wins),
        }

    def _spawn_enemies(self) -> list[Enemy]:
        enemies: list[Enemy] = []
        start_safe_x = self.config.start_margin + 220
        end_safe_x = self.config.goal_x - 180
        min_y = self.config.corridor_top
        max_y = self.config.corridor_bottom

        for _ in range(self.config.enemy_count):
            radius = float(self.rng.randint(self.config.enemy_radius_min, self.config.enemy_radius_max))
            enemy = None

            for _ in range(500):
                x = self.rng.uniform(start_safe_x, end_safe_x)
                y = self.rng.uniform(min_y + radius, max_y - radius)
                speed = self.rng.uniform(self.config.enemy_speed_min, self.config.enemy_speed_max)
                angle = self.rng.uniform(0.0, math.tau)
                candidate = Enemy(
                    x=x,
                    y=y,
                    radius=radius,
                    color=self.config.enemy_color,
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                )

                if all(candidate.distance_to(other) > candidate.radius + other.radius + 28 for other in enemies):
                    enemy = candidate
                    break

            if enemy is None:
                enemy = Enemy(
                    x=self.rng.uniform(start_safe_x, end_safe_x),
                    y=self.rng.uniform(min_y + radius, max_y - radius),
                    radius=radius,
                    color=self.config.enemy_color,
                    vx=self.rng.choice((-1.0, 1.0)) * self.rng.uniform(self.config.enemy_speed_min, self.config.enemy_speed_max),
                    vy=self.rng.choice((-1.0, 1.0)) * self.rng.uniform(self.config.enemy_speed_min, self.config.enemy_speed_max),
                )

            enemies.append(enemy)

        return enemies

    def _collided(self) -> bool:
        for enemy in self.enemies:
            if self.player.distance_to(enemy) <= self.player.radius + enemy.radius:
                return True
        return False
