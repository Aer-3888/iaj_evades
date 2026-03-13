import pygame
from config import GameConfig
from src.core.systems import EntityManager
from src.core.components import PositionComponent, CircleColliderComponent, PlayerComponent, EnemyComponent

class Renderer:
    def __init__(self, screen: pygame.Surface, config: GameConfig):
        self.screen = screen
        self.config = config
        self.font = pygame.font.SysFont("consolas", 22)
        self.camera_x = 0.0

    def update_camera(self, entity_manager: EntityManager, player_id: int):
        p_pos = entity_manager.get_component(player_id, PositionComponent)
        if p_pos:
            target = p_pos.x - self.config.screen_width * 0.35 + self.config.camera_lead
            self.camera_x = max(0.0, min(target, self.config.world_width - self.config.screen_width))

    def draw(self, entity_manager: EntityManager, info: dict):
        self.screen.fill(self.config.background_color)

        # Draw Corridor
        corridor_rect = pygame.Rect(0, self.config.corridor_top, self.config.screen_width, self.config.corridor_height)
        pygame.draw.rect(self.screen, self.config.corridor_color, corridor_rect)
        pygame.draw.line(self.screen, self.config.corridor_line_color, (0, self.config.corridor_top), (self.config.screen_width, self.config.corridor_top), 3)
        pygame.draw.line(self.screen, self.config.corridor_line_color, (0, self.config.corridor_bottom), (self.config.screen_width, self.config.corridor_bottom), 3)

        # Draw Lane Markers
        marker_spacing = 180
        marker_width = 70
        marker_height = 8
        marker_y = self.config.corridor_top + self.config.corridor_height // 2 - marker_height // 2
        first_marker = int(self.camera_x // marker_spacing) * marker_spacing
        for world_x in range(first_marker, int(self.camera_x + self.config.screen_width) + marker_spacing, marker_spacing):
            screen_x = int(world_x - self.camera_x)
            pygame.draw.rect(self.screen, self.config.lane_marker_color, pygame.Rect(screen_x, marker_y, marker_width, marker_height), border_radius=4)

        # Draw Goal
        goal_screen_x = int(self.config.goal_x - self.camera_x)
        pygame.draw.rect(self.screen, self.config.goal_color, pygame.Rect(goal_screen_x, self.config.corridor_top, self.config.goal_width, self.config.corridor_height), border_radius=8)

        # Draw Entities
        enemies = entity_manager.get_entities_with(EnemyComponent, PositionComponent, CircleColliderComponent)
        for e_id in enemies:
            pos = entity_manager.get_component(e_id, PositionComponent)
            col = entity_manager.get_component(e_id, CircleColliderComponent)
            pygame.draw.circle(self.screen, col.color, (int(pos.x - self.camera_x), int(pos.y)), int(col.radius))

        players = entity_manager.get_entities_with(PlayerComponent, PositionComponent, CircleColliderComponent)
        for p_id in players:
            pos = entity_manager.get_component(p_id, PositionComponent)
            col = entity_manager.get_component(p_id, CircleColliderComponent)
            pygame.draw.circle(self.screen, col.color, (int(pos.x - self.camera_x), int(pos.y)), int(col.radius))

        # Draw UI
        self._draw_ui(info)

    def _draw_ui(self, info: dict):
        progress_text = self.font.render(f"Progress {info.get('best_progress', 0):.0f}/{self.config.goal_x - self.config.start_margin:.0f}", True, self.config.text_color)
        deaths_text = self.font.render(f"Deaths {int(info.get('deaths', 0))}", True, self.config.text_color)
        
        self.screen.blit(progress_text, (20, 18))
        self.screen.blit(deaths_text, (20, 46))

        controls_text = self.font.render("Move: WASD / Arrows   Restart: R   Quit: Esc", True, self.config.text_color)
        self.screen.blit(controls_text, (20, self.config.screen_height - 34))

        if info.get('done') and info.get('done_reason') == "goal":
            banner = self.font.render("Goal reached! Press R to run again.", True, self.config.warning_color)
            banner_rect = banner.get_rect(center=(self.config.screen_width // 2, 28))
            self.screen.blit(banner, banner_rect)
