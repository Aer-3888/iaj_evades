from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math


class Action(IntEnum):
    IDLE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    UP_RIGHT = 5
    DOWN_RIGHT = 6
    UP_LEFT = 7
    DOWN_LEFT = 8


ACTION_VECTORS: dict[Action, tuple[int, int]] = {
    Action.IDLE: (0, 0),
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
    Action.UP_RIGHT: (1, -1),
    Action.DOWN_RIGHT: (1, 1),
    Action.UP_LEFT: (-1, -1),
    Action.DOWN_LEFT: (-1, 1),
}


DELTA_TO_ACTION: dict[tuple[int, int], Action] = {
    vector: action for action, vector in ACTION_VECTORS.items()
}


@dataclass
class CircleBody:
    x: float
    y: float
    radius: float
    color: tuple[int, int, int]
    vx: float = 0.0
    vy: float = 0.0

    def distance_to(self, other: "CircleBody") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Player(CircleBody):
    def apply_action(
        self,
        action: Action,
        speed: float,
        dt: float,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> None:
        dx, dy = ACTION_VECTORS[action]
        length = math.hypot(dx, dy)
        if length > 0.0:
            self.vx = dx / length * speed
            self.vy = dy / length * speed
        else:
            self.vx = 0.0
            self.vy = 0.0

        self.x += self.vx * dt
        self.y += self.vy * dt

        self.x = max(min_x + self.radius, min(max_x - self.radius, self.x))
        self.y = max(min_y + self.radius, min(max_y - self.radius, self.y))


@dataclass
class Enemy(CircleBody):
    def update(self, dt: float, min_x: float, max_x: float, min_y: float, max_y: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt

        if self.x - self.radius <= min_x:
            self.x = min_x + self.radius
            self.vx *= -1.0
        elif self.x + self.radius >= max_x:
            self.x = max_x - self.radius
            self.vx *= -1.0

        if self.y - self.radius <= min_y:
            self.y = min_y + self.radius
            self.vy *= -1.0
        elif self.y + self.radius >= max_y:
            self.y = max_y - self.radius
            self.vy *= -1.0
