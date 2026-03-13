import pygame
from config import GameConfig
from src.core.systems import EntityManager
from src.core.components import PositionComponent, CircleColliderComponent, PlayerComponent, EnemyComponent

class Renderer:
    def __init__(self, screen: pygame.Surface, config: GameConfig):
        self.screen = screen
        self.config = config
        pygame.font.init()
        # Using a modern sans-serif font
        self.font = pygame.font.SysFont("arial,helvetica,sans-serif", 18, bold=True)
        self.label_font = pygame.font.SysFont("arial,helvetica,sans-serif", 14)
        self.camera_x = 0.0

    def update_camera(self, entity_manager: EntityManager, player_id: int):
        p_pos = entity_manager.get_component(player_id, PositionComponent)
        if p_pos:
            target = p_pos.x - self.config.screen_width * 0.35 + self.config.camera_lead
            self.camera_x = max(0.0, min(target, self.config.world_width - self.config.screen_width))

    def draw(self, entity_manager: EntityManager, info: dict):
        self.screen.fill(self.config.background_color)

        # Draw Corridor (Old visuals - RESTORED)
        corridor_rect = pygame.Rect(0, self.config.corridor_top, self.config.screen_width, self.config.corridor_height)
        pygame.draw.rect(self.screen, self.config.corridor_color, corridor_rect)
        pygame.draw.line(self.screen, self.config.corridor_line_color, (0, self.config.corridor_top), (self.config.screen_width, self.config.corridor_top), 3)
        pygame.draw.line(self.screen, self.config.corridor_line_color, (0, self.config.corridor_bottom), (self.config.screen_width, self.config.corridor_bottom), 3)

        # Draw Lane Markers (RESTORED center dashed line)
        marker_spacing = 180
        marker_width = 70
        marker_height = 8
        marker_y = self.config.corridor_top + self.config.corridor_height // 2 - marker_height // 2
        first_marker = int(self.camera_x // marker_spacing) * marker_spacing
        for world_x in range(first_marker, int(self.camera_x + self.config.screen_width) + marker_spacing, marker_spacing):
            screen_x = int(world_x - self.camera_x)
            pygame.draw.rect(
                self.screen,
                self.config.lane_marker_color,
                pygame.Rect(screen_x, marker_y, marker_width, marker_height),
                border_radius=4,
            )

        # Draw Goal (Old visuals - PRESERVED)
        goal_screen_x = int(self.config.goal_x - self.camera_x)
        pygame.draw.rect(self.screen, self.config.goal_color, pygame.Rect(goal_screen_x, self.config.corridor_top, self.config.goal_width, self.config.corridor_height), border_radius=8)

        # Draw Entities (Balls - PRESERVED)
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

        # Draw Reworked UI (PRESERVED)
        self._draw_ui(info)

    def _draw_ui(self, info: dict):
        # 1. HUD Card in top-left
        card_rect = pygame.Rect(20, 15, 200, 65)
        # Subtle semi-transparent background
        card_bg = pygame.Surface((card_rect.width, card_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(card_bg, (255, 255, 255, 30), card_bg.get_rect(), border_radius=10)
        self.screen.blit(card_bg, (card_rect.x, card_rect.y))
        
        # Border for the card
        pygame.draw.rect(self.screen, (255, 255, 255, 50), card_rect, width=1, border_radius=10)

        # Stats Content
        total_dist = self.config.goal_x - self.config.start_margin
        prog_val = info.get('best_progress', 0)
        prog_percent = min(100.0, (prog_val / total_dist) * 100)
        
        prog_label = self.label_font.render("PROGRESS", True, (160, 160, 160))
        prog_text = self.font.render(f"{prog_percent:.1f}%", True, self.config.text_color)
        self.screen.blit(prog_label, (40, 22))
        self.screen.blit(prog_text, (40, 42))

        death_label = self.label_font.render("DEATHS", True, (160, 160, 160))
        death_text = self.font.render(f"{int(info.get('deaths', 0))}", True, (255, 120, 120))
        self.screen.blit(death_label, (140, 22))
        self.screen.blit(death_text, (140, 42))

        # 2. Global Progress Bar (Bottom Edge)
        bar_width = self.config.screen_width - 100
        bar_height = 4
        bar_x = 50
        bar_y = self.config.screen_height - 25
        
        # Bar track
        pygame.draw.rect(self.screen, (255, 255, 255, 40), (bar_x, bar_y, bar_width, bar_height), border_radius=2)
        # Bar fill
        fill_width = int(bar_width * (prog_val / total_dist))
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.config.player_color, (bar_x, bar_y, fill_width, bar_height), border_radius=2)

        # 3. Controls Hint (Discreet bottom center)
        ctrl_text = self.label_font.render("WASD: Move  R: Reset  ESC: Quit", True, (100, 100, 100))
        ctrl_rect = ctrl_text.get_rect(center=(self.config.screen_width // 2, self.config.screen_height - 10))
        self.screen.blit(ctrl_text, ctrl_rect)

        # Goal reached banner
        if info.get('done') and info.get('done_reason') == "goal":
            banner = self.font.render("MISSION ACCOMPLISHED", True, self.config.goal_color)
            banner_rect = banner.get_rect(center=(self.config.screen_width // 2, 35))
            self.screen.blit(banner, banner_rect)
            
            sub_banner = self.label_font.render("PRESS R TO CONTINUE", True, (200, 200, 200))
            sub_rect = sub_banner.get_rect(center=(self.config.screen_width // 2, 55))
            self.screen.blit(sub_banner, sub_rect)
