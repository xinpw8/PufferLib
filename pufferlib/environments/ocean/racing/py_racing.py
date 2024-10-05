import numpy as np
import gymnasium
from .cy_racing_cy import CRacingCy

class RacingCyEnv(gymnasium.Env):
    def __init__(self, num_enemy_cars=1):
        super().__init__()

        self.num_enemy_cars = num_enemy_cars
        self.c_env = CRacingCy(num_enemy_cars)
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(5 + 2 * num_enemy_cars,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(5)  # NOOP, ACCEL, DECEL, LEFT, RIGHT

        self.render_mode = 'human'
        self.client = None

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)

        self.c_env.reset()
        state = self.c_env.get_state()
        return state, {}

    def step(self, action):
        state, reward, done, truncated, info = self.c_env.step(action)
        return state, reward, done, truncated, info

    def render(self):
        if self.client is None:
            self.client = RaylibClient()

        state = self.c_env.get_state()
        frame, action = self.client.render(state, self.num_enemy_cars)
        return frame

    def close(self):
        pass


class RaylibClient:
    def __init__(self, width=160, height=210):
        self.width = width
        self.height = height

        from raylib import rl
        rl.InitWindow(width, height, "PufferLib Racing".encode())
        rl.SetTargetFPS(60)
        self.rl = rl

        from cffi import FFI
        self.ffi = FFI()

    def _cdata_to_numpy(self):
        image = self.rl.LoadImageFromScreen()
        width, height, channels = image.width, image.height, 4
        cdata = self.ffi.buffer(image.data, width * height * channels)
        return np.frombuffer(cdata, dtype=np.uint8).reshape((height, width, channels))[:, :, :3]

    def render(self, state, num_enemy_cars):
        rl = self.rl
        action = None
        if rl.IsKeyDown(rl.KEY_UP):
            action = 1  # ACCEL
        elif rl.IsKeyDown(rl.KEY_DOWN):
            action = 2  # DECEL
        elif rl.IsKeyDown(rl.KEY_LEFT):
            action = 3  # LEFT
        elif rl.IsKeyDown(rl.KEY_RIGHT):
            action = 4  # RIGHT

        rl.BeginDrawing()
        rl.ClearBackground(rl.DARKGREEN)

        # Draw road
        road_width = 90
        road_center_x = self.width // 2
        road_left_edge = road_center_x - road_width // 2
        rl.DrawRectangle(road_left_edge, 0, road_width, self.height, rl.GRAY)

        # Draw lane dividers
        lane_width = road_width // 3
        rl.DrawLine(road_left_edge + lane_width, 0, road_left_edge + lane_width, self.height, rl.WHITE)
        rl.DrawLine(road_left_edge + 2 * lane_width, 0, road_left_edge + 2 * lane_width, self.height, rl.WHITE)

        # Draw player car
        player_x = int(state[0] * self.width)
        player_y = int(state[1] * self.height)
        rl.DrawRectangle(player_x, player_y, 16, 11, rl.BLUE)

        # Draw enemy cars
        for i in range(num_enemy_cars):
            enemy_x = int(state[5 + i * 2] * self.width)
            enemy_y = int(state[6 + i * 2] * self.height)
            rl.DrawRectangle(enemy_x, enemy_y, 16, 11, rl.RED)

        # Draw HUD
        rl.DrawRectangle(48, 161, 64, 30, rl.RED)
        rl.DrawText(f"Score: {int(state[2] * 1000)}", 56, 162, 10, rl.BLACK)
        rl.DrawText(f"Day: {int(state[3])}", 56, 179, 10, rl.BLACK)
        rl.DrawText(f"Cars: {int(state[4] * 200)}", 72, 179, 10, rl.BLACK)

        rl.EndDrawing()
        return self._cdata_to_numpy(), action

    def close(self):
        self.rl.CloseWindow()
