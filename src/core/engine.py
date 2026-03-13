import random
import math
from typing import Optional
from config import GameConfig
from entities import Action
from src.core.components import (
    PositionComponent, VelocityComponent, CircleColliderComponent, 
    PlayerComponent, EnemyComponent
)
from src.core.systems import (
    EntityManager, MovementSystem, PlayerControlSystem, BoundarySystem, CollisionSystem
)

class CoreEngine:
    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config or GameConfig()
        self.entity_manager = EntityManager()
        self.player_id = -1
        self.enemy_ids = []
        self.elapsed_time = 0.0
        self.done = False
        self.done_reason = ""
        self.rng = random.Random()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)
        
        self.entity_manager = EntityManager()
        self.elapsed_time = 0.0
        self.done = False
        self.done_reason = ""
        
        # Create Player
        self.player_id = self.entity_manager.create_entity()
        start_y = self.config.corridor_top + self.config.corridor_height / 2.0
        self.entity_manager.add_component(self.player_id, PositionComponent(float(self.config.start_margin), float(start_y)))
        self.entity_manager.add_component(self.player_id, VelocityComponent(0.0, 0.0))
        self.entity_manager.add_component(self.player_id, CircleColliderComponent(float(self.config.player_radius), self.config.player_color))
        self.entity_manager.add_component(self.player_id, PlayerComponent())
        
        # Create Enemies
        self.enemy_ids = self._spawn_enemies()
        
    def step(self, action: Action, dt: float):
        if self.done:
            return
        
        PlayerControlSystem.apply_action(self.entity_manager, action, self.config.player_speed)
        MovementSystem.update(self.entity_manager, dt)
        BoundarySystem.update(
            self.entity_manager, 
            0.0, float(self.config.world_width), 
            float(self.config.corridor_top), float(self.config.corridor_bottom)
        )
        
        self.elapsed_time += dt
        
        player_pos = self.entity_manager.get_component(self.player_id, PositionComponent)
        player_col = self.entity_manager.get_component(self.player_id, CircleColliderComponent)
        
        if CollisionSystem.check_player_enemy_collisions(self.entity_manager):
            self.done = True
            self.done_reason = "collision"
        elif player_pos.x + player_col.radius >= self.config.goal_x:
            self.done = True
            self.done_reason = "goal"
        elif self.elapsed_time >= self.config.max_episode_time:
            self.done = True
            self.done_reason = "timeout"

    def _spawn_enemies(self) -> list[int]:
        enemy_ids = []
        start_safe_x = self.config.start_margin + 220
        end_safe_x = self.config.goal_x - 180
        min_y = self.config.corridor_top
        max_y = self.config.corridor_bottom

        for _ in range(self.config.enemy_count):
            radius = float(self.rng.randint(self.config.enemy_radius_min, self.config.enemy_radius_max))
            
            # Simplified spawning logic for brevity, keeping the spirit of the original
            x = self.rng.uniform(start_safe_x, end_safe_x)
            y = self.rng.uniform(min_y + radius, max_y - radius)
            speed = self.rng.uniform(self.config.enemy_speed_min, self.config.enemy_speed_max)
            angle = self.rng.uniform(0.0, math.tau)
            
            e_id = self.entity_manager.create_entity()
            self.entity_manager.add_component(e_id, PositionComponent(x, y))
            self.entity_manager.add_component(e_id, VelocityComponent(math.cos(angle) * speed, math.sin(angle) * speed))
            self.entity_manager.add_component(e_id, CircleColliderComponent(radius, self.config.enemy_color))
            self.entity_manager.add_component(e_id, EnemyComponent())
            enemy_ids.append(e_id)
            
        return enemy_ids
