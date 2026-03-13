import pygame
import sys
from abc import ABC, abstractmethod
from config import GameConfig
from entities import Action, DELTA_TO_ACTION
from src.core.engine import CoreEngine
from src.rl.wrapper import RLWrapper
from src.ui.renderer import Renderer

class GameState(ABC):
    @abstractmethod
    def handle_events(self, events): pass
    
    @abstractmethod
    def update(self, dt): pass
    
    @abstractmethod
    def draw(self, screen): pass

class PlayingState(GameState):
    def __init__(self, config: GameConfig, renderer: Renderer):
        self.config = config
        self.renderer = renderer
        self.engine = CoreEngine(config)
        self.rl_env = RLWrapper(self.engine)
        self.rl_env.reset()
        self.accumulator = 0.0
        
        self.settings_open = False
        self.selected_row = 0 # 0: Speed, 1: Count, 2: Level

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.renderer.settings_rect.collidepoint(event.pos):
                    self.settings_open = not self.settings_open
                    if self.settings_open:
                        # Initialize temp values for level jumping
                        self.config._level_temp = self.engine.level
            
            if self.settings_open:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.selected_row = (self.selected_row - 1) % 3
                    elif event.key == pygame.K_DOWN:
                        self.selected_row = (self.selected_row + 1) % 3
                    elif event.key == pygame.K_LEFT:
                        self._adjust_setting(-1)
                    elif event.key == pygame.K_RIGHT:
                        self._adjust_setting(1)
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self._apply_settings()
                    elif event.key == pygame.K_ESCAPE:
                        self.settings_open = False
            else:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.rl_env.reset(reset_level=True)

    def _adjust_setting(self, dir):
        if self.selected_row == 0: # Speed
            self.config.speed_multiplier = round(max(0.1, min(3.0, self.config.speed_multiplier + dir * 0.1)), 1)
        elif self.selected_row == 1: # Enemy Count
            self.config.enemy_count = max(1, min(50, self.config.enemy_count + dir))
        elif self.selected_row == 2: # Level
            if hasattr(self.config, '_level_temp'):
                self.config._level_temp = max(1, min(99, self.config._level_temp + dir))

    def _apply_settings(self):
        # Jump to level if changed
        if hasattr(self.config, '_level_temp') and self.config._level_temp != self.engine.level:
            self.engine.level = self.config._level_temp
            self.rl_env.reset(reset_level=False)
        else:
            # Re-spawn enemies if count or speed changed
            self.rl_env.reset(reset_level=False)
            
        self.settings_open = False

    def _get_action(self) -> Action:
        pressed = pygame.key.get_pressed()
        dx = int(pressed[pygame.K_d] or pressed[pygame.K_RIGHT]) - int(pressed[pygame.K_a] or pressed[pygame.K_LEFT])
        dy = int(pressed[pygame.K_s] or pressed[pygame.K_DOWN]) - int(pressed[pygame.K_w] or pressed[pygame.K_UP])
        return DELTA_TO_ACTION.get((dx, dy), Action.IDLE)

    def update(self, dt):
        if self.settings_open:
            return

        self.accumulator += min(dt, 0.25)
        action = self._get_action()

        while self.accumulator >= self.config.fixed_timestep:
            self.accumulator -= self.config.fixed_timestep
            
            if self.engine.done:
                # Auto-restart from Level 1 on death/timeout
                self.rl_env.reset(reset_level=True)
            
            self.rl_env.step(action, self.config.fixed_timestep)

    def draw(self, screen):
        self.renderer.update_camera(self.engine.entity_manager, self.engine.player_id)
        # Pass selected_row to renderer
        self.renderer.draw(self.engine.entity_manager, self.rl_env.get_info(), self.settings_open, self.selected_row)

class StateMachine:
    def __init__(self, initial_state: GameState):
        self.current_state = initial_state

    def change_state(self, new_state: GameState):
        self.current_state = new_state

    def handle_events(self, events):
        self.current_state.handle_events(events)

    def update(self, dt):
        self.current_state.update(dt)

    def draw(self, screen):
        self.current_state.draw(screen)
