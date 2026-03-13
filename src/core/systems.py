import math
from typing import List, Tuple
from src.core.components import PositionComponent, VelocityComponent, CircleColliderComponent, PlayerComponent, EnemyComponent
from entities import Action, ACTION_VECTORS

class EntityManager:
    def __init__(self):
        self.entities = []
        self.components = {} # type -> {entity_id: component}

    def create_entity(self) -> int:
        entity_id = len(self.entities)
        self.entities.append(entity_id)
        return entity_id

    def add_component(self, entity_id: int, component):
        comp_type = type(component)
        if comp_type not in self.components:
            self.components[comp_type] = {}
        self.components[comp_type][entity_id] = component

    def get_component(self, entity_id: int, comp_type):
        return self.components.get(comp_type, {}).get(entity_id)

    def get_entities_with(self, *comp_types):
        if not comp_types:
            return []
        
        # Start with entities that have the first component type
        entities = set(self.components.get(comp_types[0], {}).keys())
        
        # Intersect with entities that have the rest of the component types
        for comp_type in comp_types[1:]:
            entities &= set(self.components.get(comp_type, {}).keys())
            
        return list(entities)

class MovementSystem:
    @staticmethod
    def update(entity_manager: EntityManager, dt: float):
        entities = entity_manager.get_entities_with(PositionComponent, VelocityComponent)
        for entity_id in entities:
            pos = entity_manager.get_component(entity_id, PositionComponent)
            vel = entity_manager.get_component(entity_id, VelocityComponent)
            pos.x += vel.vx * dt
            pos.y += vel.vy * dt

class PlayerControlSystem:
    @staticmethod
    def apply_action(entity_manager: EntityManager, action: Action, speed: float):
        entities = entity_manager.get_entities_with(PlayerComponent, VelocityComponent)
        dx, dy = ACTION_VECTORS[action]
        length = math.hypot(dx, dy)
        
        for entity_id in entities:
            vel = entity_manager.get_component(entity_id, VelocityComponent)
            if length > 0.0:
                vel.vx = (dx / length) * speed
                vel.vy = (dy / length) * speed
            else:
                vel.vx = 0.0
                vel.vy = 0.0

class BoundarySystem:
    @staticmethod
    def update(entity_manager: EntityManager, world_width: float, corridor_top: float, corridor_bottom: float, start_margin: float, goal_x: float):
        # Handle Player boundaries (Clamping to full world for teleporting)
        players = entity_manager.get_entities_with(PlayerComponent, PositionComponent, CircleColliderComponent)
        for entity_id in players:
            pos = entity_manager.get_component(entity_id, PositionComponent)
            col = entity_manager.get_component(entity_id, CircleColliderComponent)
            
            # Allow player to reach absolute X boundaries for teleport triggers
            pos.x = max(0, min(world_width, pos.x))
            pos.y = max(corridor_top + col.radius, min(corridor_bottom - col.radius, pos.y))

        # Handle Enemy boundaries (Clamping to Danger Zone only)
        enemies = entity_manager.get_entities_with(EnemyComponent, PositionComponent, VelocityComponent, CircleColliderComponent)
        for entity_id in enemies:
            pos = entity_manager.get_component(entity_id, PositionComponent)
            vel = entity_manager.get_component(entity_id, VelocityComponent)
            col = entity_manager.get_component(entity_id, CircleColliderComponent)
            
            # Enemies stay strictly between the safe zones
            if pos.x - col.radius <= start_margin:
                pos.x = start_margin + col.radius
                vel.vx *= -1.0
            elif pos.x + col.radius >= goal_x:
                pos.x = goal_x - col.radius
                vel.vx *= -1.0

            if pos.y - col.radius <= corridor_top:
                pos.y = corridor_top + col.radius
                vel.vy *= -1.0
            elif pos.y + col.radius >= corridor_bottom:
                pos.y = corridor_bottom - col.radius
                vel.vy *= -1.0

class CollisionSystem:
    @staticmethod
    def check_player_enemy_collisions(entity_manager: EntityManager) -> bool:
        players = entity_manager.get_entities_with(PlayerComponent, PositionComponent, CircleColliderComponent)
        enemies = entity_manager.get_entities_with(EnemyComponent, PositionComponent, CircleColliderComponent)
        
        for p_id in players:
            p_pos = entity_manager.get_component(p_id, PositionComponent)
            p_col = entity_manager.get_component(p_id, CircleColliderComponent)
            
            for e_id in enemies:
                e_pos = entity_manager.get_component(e_id, PositionComponent)
                e_col = entity_manager.get_component(e_id, CircleColliderComponent)
                
                dist = math.hypot(p_pos.x - e_pos.x, p_pos.y - e_pos.y)
                if dist <= p_col.radius + e_col.radius:
                    return True
        return False
