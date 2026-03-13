from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    screen_width: int = 1000
    screen_height: int = 500
    world_width: int = 2000
    world_height: int = 500
    corridor_top: int = 60
    corridor_bottom: int = 440
    goal_width: int = 90
    start_margin: int = 80
    player_radius: int = 16
    enemy_radius_min: int = 12
    enemy_radius_max: int = 22
    player_speed: float = 260.0
    enemy_speed_min: float = 130.0
    enemy_speed_max: float = 230.0
    enemy_count: int = 70
    fixed_timestep: float = 1.0 / 60.0
    render_fps: int = 60
    max_episode_time: float = 60.0
    camera_lead: float = 140.0
    default_seed: int = 7
    background_color: tuple[int, int, int] = (16, 20, 28)
    corridor_color: tuple[int, int, int] = (40, 52, 68)
    corridor_line_color: tuple[int, int, int] = (90, 112, 138)
    lane_marker_color: tuple[int, int, int] = (60, 76, 96)
    player_color: tuple[int, int, int] = (97, 218, 251)
    enemy_color: tuple[int, int, int] = (255, 126, 95)
    goal_color: tuple[int, int, int] = (130, 218, 109)
    text_color: tuple[int, int, int] = (235, 240, 245)
    warning_color: tuple[int, int, int] = (255, 210, 120)

    @property
    def goal_x(self) -> float:
        return float(self.world_width - self.goal_width)

    @property
    def corridor_height(self) -> int:
        return self.corridor_bottom - self.corridor_top
