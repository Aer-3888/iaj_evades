from __future__ import annotations
import sys
import pygame
from config import GameConfig
from src.ui.renderer import Renderer
from src.ui.states import StateMachine, PlayingState

def main() -> None:
    pygame.init()
    config = GameConfig()
    screen = pygame.display.set_mode((config.screen_width, config.screen_height))
    pygame.display.set_caption("Genetic Dodge Runner - Refactored")
    clock = pygame.time.Clock()

    renderer = Renderer(screen, config)
    playing_state = PlayingState(config, renderer)
    state_machine = StateMachine(playing_state)

    running = True
    while running:
        dt = clock.tick(config.render_fps) / 1000.0
        
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        state_machine.handle_events(events)
        state_machine.update(dt)
        state_machine.draw(screen)
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
