from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class PositionComponent:
    x: float
    y: float

@dataclass
class VelocityComponent:
    vx: float
    vy: float

@dataclass
class CircleColliderComponent:
    radius: float
    color: Tuple[int, int, int]

@dataclass
class PlayerComponent:
    """Marker component for the player entity."""
    pass

@dataclass
class EnemyComponent:
    """Marker component for enemy entities."""
    pass
