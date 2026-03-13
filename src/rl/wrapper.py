from typing import List, Tuple, Dict, Any, Optional
from src.core.engine import CoreEngine
from src.core.components import PositionComponent, VelocityComponent, CircleColliderComponent
from entities import Action

class RLWrapper:
    def __init__(self, engine: CoreEngine):
        self.engine = engine
        self.config = engine.config
        self.best_x = 0.0
        self.total_deaths = 0
        self.total_wins = 0

    def reset(self, seed: Optional[int] = None) -> List[float]:
        self.engine.reset(seed)
        player_pos = self.engine.entity_manager.get_component(self.engine.player_id, PositionComponent)
        self.best_x = player_pos.x
        return self.get_observation()

    def step(self, action: Action, dt: float) -> Tuple[List[float], float, bool, Dict[str, Any]]:
        previous_best = self.best_x
        self.engine.step(action, dt)
        
        player_pos = self.engine.entity_manager.get_component(self.engine.player_id, PositionComponent)
        self.best_x = max(self.best_x, player_pos.x)
        
        reward = (self.best_x - previous_best) * 0.02
        
        if self.engine.done:
            if self.engine.done_reason == "collision":
                self.total_deaths += 1
                reward -= 150.0
            elif self.engine.done_reason == "goal":
                self.total_wins += 1
                reward += 250.0
            elif self.engine.done_reason == "timeout":
                reward -= 50.0
                
        return self.get_observation(), reward, self.engine.done, self.get_info()

    def get_observation(self) -> List[float]:
        player_id = self.engine.player_id
        em = self.engine.entity_manager
        p_pos = em.get_component(player_id, PositionComponent)
        p_vel = em.get_component(player_id, VelocityComponent)
        
        observation = [
            p_pos.x / self.config.world_width,
            p_pos.y / self.config.world_height,
            p_vel.vx / self.config.player_speed,
            p_vel.vy / self.config.player_speed,
            max(0.0, self.config.goal_x - p_pos.x) / self.config.world_width,
            self.engine.elapsed_time / self.config.max_episode_time,
        ]

        max_enemy_speed = self.config.enemy_speed_max
        for e_id in self.engine.enemy_ids:
            e_pos = em.get_component(e_id, PositionComponent)
            e_vel = em.get_component(e_id, VelocityComponent)
            e_col = em.get_component(e_id, CircleColliderComponent)
            
            observation.extend([
                (e_pos.x - p_pos.x) / self.config.world_width,
                (e_pos.y - p_pos.y) / self.config.corridor_height,
                e_vel.vx / max_enemy_speed,
                e_vel.vy / max_enemy_speed,
                e_col.radius / self.config.enemy_radius_max,
            ])
        return observation

    def get_info(self) -> Dict[str, Any]:
        total_distance = self.config.goal_x - self.config.start_margin
        best_progress = max(0.0, self.best_x - self.config.start_margin)
        progress_ratio = max(0.0, min(1.0, best_progress / total_distance))
        
        return {
            "done": self.engine.done,
            "done_reason": self.engine.done_reason,
            "elapsed_time": self.engine.elapsed_time,
            "best_progress": best_progress,
            "progress_ratio": progress_ratio,
            "deaths": float(self.total_deaths),
            "wins": float(self.total_wins),
        }
