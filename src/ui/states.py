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

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.rl_env.reset()

    def _get_action(self) -> Action:
        pressed = pygame.key.get_pressed()
        dx = int(pressed[pygame.K_d] or pressed[pygame.K_RIGHT]) - int(pressed[pygame.K_a] or pressed[pygame.K_LEFT])
        dy = int(pressed[pygame.K_s] or pressed[pygame.K_DOWN]) - int(pressed[pygame.K_w] or pressed[pygame.K_UP])
        return DELTA_TO_ACTION.get((dx, dy), Action.IDLE)

    def update(self, dt):
        self.accumulator += min(dt, 0.25)
        action = self._get_action()

        while self.accumulator >= self.config.fixed_timestep:
            self.accumulator -= self.config.fixed_timestep
            
            if self.engine.done:
                if self.engine.done_reason in {"collision", "timeout"}:
                    self.rl_env.reset()
                else:
                    break
            
            self.rl_env.step(action, self.config.fixed_timestep)

    def draw(self, screen):
        self.renderer.update_camera(self.engine.entity_manager, self.engine.player_id)
        self.renderer.draw(self.engine.entity_manager, self.rl_env.get_info())

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
