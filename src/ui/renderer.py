import pygame
import math
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
        self.header_font = pygame.font.SysFont("arial,helvetica,sans-serif", 24, bold=True)
        self.camera_x = 0.0
        self.settings_rect = pygame.Rect(config.screen_width - 50, 15, 32, 32)

    def update_camera(self, entity_manager: EntityManager, player_id: int):
        p_pos = entity_manager.get_component(player_id, PositionComponent)
        if p_pos:
            target = p_pos.x - self.config.screen_width * 0.35 + self.config.camera_lead
            self.camera_x = max(0.0, min(target, self.config.world_width - self.config.screen_width))

    def draw(self, entity_manager: EntityManager, info: dict, settings_open: bool = False, selected_index: int = 0):
        self.screen.fill(self.config.background_color)

        # Draw Corridor (Base)
        corridor_rect = pygame.Rect(0, self.config.corridor_top, self.config.screen_width, self.config.corridor_height)
        pygame.draw.rect(self.screen, self.config.corridor_color, corridor_rect)
        
        # Draw Lane Markers
        marker_spacing = 180
        marker_width = 70
        marker_height = 8
        marker_y = self.config.corridor_top + self.config.corridor_height // 2 - marker_height // 2
        first_marker = int(self.camera_x // marker_spacing) * marker_spacing
        for world_x in range(first_marker, int(self.camera_x + self.config.screen_width) + marker_spacing, marker_spacing):
            screen_x = int(world_x - self.camera_x)
            pygame.draw.rect(self.screen, self.config.lane_marker_color, pygame.Rect(screen_x, marker_y, marker_width, marker_height), border_radius=4)

        # Draw Corridor Boundary Lines
        pygame.draw.line(self.screen, self.config.corridor_line_color, (0, self.config.corridor_top), (self.config.screen_width, self.config.corridor_top), 3)
        pygame.draw.line(self.screen, self.config.corridor_line_color, (0, self.config.corridor_bottom), (self.config.screen_width, self.config.corridor_bottom), 3)

        # Draw Safe Zones
        start_screen_x = int(0 - self.camera_x)
        if start_screen_x + self.config.start_margin > 0:
            s_width = min(self.config.start_margin, start_screen_x + self.config.start_margin)
            s_rect = pygame.Rect(max(0, start_screen_x), self.config.corridor_top, s_width, self.config.corridor_height)
            pygame.draw.rect(self.screen, self.config.goal_color, s_rect, border_radius=4)

        goal_screen_x = int(self.config.goal_x - self.camera_x)
        if goal_screen_x < self.config.screen_width:
            e_width = self.config.screen_width - max(0, goal_screen_x)
            e_rect = pygame.Rect(max(0, goal_screen_x), self.config.corridor_top, e_width, self.config.corridor_height)
            pygame.draw.rect(self.screen, self.config.goal_color, e_rect, border_radius=4)

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
        self._draw_settings_icon()
        
        if settings_open:
            self._draw_settings_menu(selected_index)

    def _draw_ui(self, info: dict):
        card_rect = pygame.Rect(20, 15, 260, 65)
        card_bg = pygame.Surface((card_rect.width, card_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(card_bg, (255, 255, 255, 30), card_bg.get_rect(), border_radius=10)
        self.screen.blit(card_bg, (card_rect.x, card_rect.y))
        pygame.draw.rect(self.screen, (255, 255, 255, 50), card_rect, width=1, border_radius=10)

        total_dist = self.config.goal_x - self.config.start_margin
        prog_percent = min(100.0, (info.get('best_progress', 0) / total_dist) * 100)
        
        level_label = self.label_font.render("LEVEL", True, (160, 160, 160))
        level_text = self.font.render(f"{info.get('level', 1)}", True, self.config.goal_color)
        self.screen.blit(level_label, (40, 22))
        self.screen.blit(level_text, (40, 42))

        prog_label = self.label_font.render("PROGRESS", True, (160, 160, 160))
        prog_text = self.font.render(f"{prog_percent:.1f}%", True, self.config.text_color)
        self.screen.blit(prog_label, (110, 22))
        self.screen.blit(prog_text, (110, 42))

        death_label = self.label_font.render("DEATHS", True, (160, 160, 160))
        death_text = self.font.render(f"{int(info.get('deaths', 0))}", True, (255, 120, 120))
        self.screen.blit(death_label, (210, 22))
        self.screen.blit(death_text, (210, 42))

        # Progress Bar
        bar_width, bar_height = self.config.screen_width - 100, 4
        bar_x, bar_y = 50, self.config.screen_height - 25
        pygame.draw.rect(self.screen, (255, 255, 255, 40), (bar_x, bar_y, bar_width, bar_height), border_radius=2)
        fill_w = int(bar_width * (info.get('best_progress', 0) / total_dist))
        if fill_w > 0:
            pygame.draw.rect(self.screen, self.config.player_color, (bar_x, bar_y, fill_w, bar_height), border_radius=2)

        ctrl_text = self.label_font.render("WASD: Move  R: Reset Game  ESC: Quit", True, (100, 100, 100))
        self.screen.blit(ctrl_text, ctrl_text.get_rect(center=(self.config.screen_width // 2, self.config.screen_height - 10)))

    def _draw_settings_icon(self):
        center = self.settings_rect.center
        radius = 12
        pygame.draw.circle(self.screen, (200, 200, 200), center, radius, width=2)
        pygame.draw.circle(self.screen, (200, 200, 200), center, 4)
        for i in range(8):
            angle = i * (math.pi / 4)
            start = (center[0] + math.cos(angle) * (radius - 2), center[1] + math.sin(angle) * (radius - 2))
            end = (center[0] + math.cos(angle) * (radius + 4), center[1] + math.sin(angle) * (radius + 4))
            pygame.draw.line(self.screen, (200, 200, 200), start, end, 3)

    def _draw_settings_menu(self, selected_index: int):
        overlay = pygame.Surface((self.config.screen_width, self.config.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        menu_rect = pygame.Rect(self.config.screen_width // 2 - 200, self.config.screen_height // 2 - 150, 400, 300)
        pygame.draw.rect(self.screen, (40, 45, 60), menu_rect, border_radius=15)
        pygame.draw.rect(self.screen, (80, 90, 110), menu_rect, width=2, border_radius=15)

        header = self.header_font.render("GAME SETTINGS", True, (255, 255, 255))
        self.screen.blit(header, (menu_rect.x + 30, menu_rect.y + 25))

        y_off = menu_rect.y + 80
        self._draw_setting_row("Speed Multiplier", f"{self.config.speed_multiplier:.1f}x", y_off, selected_index == 0)
        self._draw_setting_row("Base Enemy Count", f"{self.config.enemy_count}", y_off + 60, selected_index == 1)
        lvl_temp = getattr(self.config, '_level_temp', '?')
        self._draw_setting_row("Jump to Level", f"{lvl_temp}", y_off + 120, selected_index == 2)

        hint = self.label_font.render("UP/DOWN: Navigate  LEFT/RIGHT: Adjust", True, (150, 150, 150))
        self.screen.blit(hint, (menu_rect.x + 30, menu_rect.y + 230))
        apply_btn = self.font.render("Press ENTER to Apply  ESC to Cancel", True, self.config.goal_color)
        self.screen.blit(apply_btn, (menu_rect.x + 30, menu_rect.y + 260))

    def _draw_setting_row(self, label, value, y, is_selected):
        if is_selected:
            # Draw selection background
            sel_rect = pygame.Rect(self.config.screen_width // 2 - 180, y - 10, 360, 50)
            pygame.draw.rect(self.screen, (60, 70, 100), sel_rect, border_radius=10)
            pygame.draw.rect(self.screen, self.config.player_color, sel_rect, width=2, border_radius=10)

        lbl = self.font.render(label, True, (255, 255, 255) if is_selected else (200, 200, 200))
        val = self.font.render(value, True, (255, 255, 255))
        self.screen.blit(lbl, (self.config.screen_width // 2 - 170, y))
        
        val_rect = pygame.Rect(self.config.screen_width // 2 + 80, y - 5, 60, 30)
        pygame.draw.rect(self.screen, (80, 90, 120) if is_selected else (60, 70, 90), val_rect, border_radius=5)
        self.screen.blit(val, val.get_rect(center=val_rect.center))
        
        if is_selected:
            arr_l = self.font.render("<", True, self.config.player_color)
            arr_r = self.font.render(">", True, self.config.player_color)
            self.screen.blit(arr_l, (val_rect.left - 25, y))
            self.screen.blit(arr_r, (val_rect.right + 10, y))
