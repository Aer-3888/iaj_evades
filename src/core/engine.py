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
        self.level = 1
        self.rng = random.Random()

    def reset(self, seed: Optional[int] = None, reset_level: bool = True):
        if seed is not None:
            self.rng.seed(seed)
        
        if reset_level:
            self.level = 1
            
        self.entity_manager = EntityManager()
        self.elapsed_time = 0.0
        self.done = False
        self.done_reason = ""
        
        # Create Player - Spawn clearly inside the start safe zone
        self.player_id = self.entity_manager.create_entity()
        start_y = self.config.corridor_top + self.config.corridor_height / 2.0
        # Position at 40 (half of start_margin 80)
        self.entity_manager.add_component(self.player_id, PositionComponent(float(self.config.start_margin / 2.0), float(start_y)))
        self.entity_manager.add_component(self.player_id, VelocityComponent(0.0, 0.0))
        self.entity_manager.add_component(self.player_id, CircleColliderComponent(float(self.config.player_radius), self.config.player_color))
        self.entity_manager.add_component(self.player_id, PlayerComponent())
        
        # Create Enemies
        self.enemy_ids = self._spawn_enemies()
        
    def step(self, action: Action, dt: float):
        if self.done:
            return
        
        # Apply speed multiplier to player speed
        PlayerControlSystem.apply_action(self.entity_manager, action, self.config.player_speed * self.config.speed_multiplier)
        MovementSystem.update(self.entity_manager, dt)
        BoundarySystem.update(
            self.entity_manager, 
            float(self.config.world_width), 
            float(self.config.corridor_top), float(self.config.corridor_bottom),
            float(self.config.start_margin), float(self.config.goal_x)
        )
        
        self.elapsed_time += dt
        
        player_pos = self.entity_manager.get_component(self.player_id, PositionComponent)
        player_col = self.entity_manager.get_component(self.player_id, CircleColliderComponent)
        
        # Only check for collisions in the "danger zone" (between safe zones)
        is_in_danger_zone = self.config.start_margin < player_pos.x < self.config.goal_x
        if is_in_danger_zone and CollisionSystem.check_player_enemy_collisions(self.entity_manager):
            self.done = True
            self.done_reason = "collision"
        
        # Seamless level transitions based on world boundaries
        if player_pos.x + player_col.radius >= self.config.world_width:
            self._transition_level(direction=1)
        elif player_pos.x - player_col.radius <= 0 and self.level > 1:
            self._transition_level(direction=-1)
            
        elif self.elapsed_time >= self.config.max_episode_time:
            self.done = True
            self.done_reason = "timeout"

    def _transition_level(self, direction: int):
        """Seamlessly transition to another level by teleporting the player."""
        self.level += direction
        
        player_pos = self.entity_manager.get_component(self.player_id, PositionComponent)
        player_vel = self.entity_manager.get_component(self.player_id, VelocityComponent)
        player_col = self.entity_manager.get_component(self.player_id, CircleColliderComponent)
        
        # Teleport to the opposite side, deep within the safe zones
        if direction == 1: # Moving to next level (right border)
            player_pos.x = float(self.config.start_margin / 2.0)
        else: # Moving back to previous level (left border)
            player_pos.x = float(self.config.world_width - self.config.goal_width / 2.0)
            
        player_vel.vx = 0.0
        player_vel.vy = 0.0
        
        # Re-generate enemies with the new level's difficulty
        for e_id in self.enemy_ids:
            for comp_type in self.entity_manager.components:
                if e_id in self.entity_manager.components[comp_type]:
                    del self.entity_manager.components[comp_type][e_id]
        
        self.enemy_ids = self._spawn_enemies()

    def _spawn_enemies(self) -> list[int]:
        enemy_ids = []
        # Enemies spawn strictly in the danger zone
        start_safe_x = self.config.start_margin + 50
        end_safe_x = self.config.goal_x - 50
        min_y = self.config.corridor_top
        max_y = self.config.corridor_bottom

        # Difficulty scaling
        current_count = self.config.enemy_count + (self.level - 1) * 2
        # Apply global speed multiplier
        speed_multiplier = (1.0 + (self.level - 1) * 0.15) * self.config.speed_multiplier

        for _ in range(current_count):
            radius = float(self.rng.randint(self.config.enemy_radius_min, self.config.enemy_radius_max))
            
            x = self.rng.uniform(start_safe_x, end_safe_x)
            y = self.rng.uniform(min_y + radius, max_y - radius)
            
            base_speed = self.rng.uniform(self.config.enemy_speed_min, self.config.enemy_speed_max)
            speed = base_speed * speed_multiplier
            angle = self.rng.uniform(0.0, math.tau)
            
            e_id = self.entity_manager.create_entity()
            self.entity_manager.add_component(e_id, PositionComponent(x, y))
            self.entity_manager.add_component(e_id, VelocityComponent(math.cos(angle) * speed, math.sin(angle) * speed))
            self.entity_manager.add_component(e_id, CircleColliderComponent(radius, self.config.enemy_color))
            self.entity_manager.add_component(e_id, EnemyComponent())
            enemy_ids.append(e_id)
            
        return enemy_ids
