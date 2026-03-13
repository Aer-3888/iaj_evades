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
