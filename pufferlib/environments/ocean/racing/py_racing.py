# py_racing.py
import numpy as np
import gymnasium
from .cy_racing_cy import CRacingCy
import pufferlib 
import pettingzoo


# Action definitions
ACTION_NOOP = 0
ACTION_ACCEL = 1
ACTION_DECEL = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4

# Define screen dimensions for rendering
TOTAL_SCREEN_WIDTH = 160
TOTAL_SCREEN_HEIGHT = 210
ACTION_SCREEN_X_START = 8
ACTION_SCREEN_Y_START = 0
ACTION_SCREEN_WIDTH = 152  # from x=8 to x=160
ACTION_SCREEN_HEIGHT = 155 # from y=0 to y=155

SCOREBOARD_X_START = 48
SCOREBOARD_Y_START = 161
SCOREBOARD_WIDTH = 64  # from x=48 to x=112
SCOREBOARD_HEIGHT = 30 # from y=161 to y=191

CARS_LEFT_X_START = 72
CARS_LEFT_Y_START = 179
CARS_LEFT_WIDTH = 32  # from x=72 to x=104
CARS_LEFT_HEIGHT = 9  # from y=179 to y=188

DAY_X_START = 56
DAY_Y_START = 179
DAY_WIDTH = 8    # from x=56 to x=64
DAY_HEIGHT = 9   # from y=179 to y=188
DAY_LENGTH = 300000  # Number of ticks in a day

ROAD_WIDTH = 90.0
CAR_WIDTH = 16.0
PLAYER_CAR_LENGTH = 11.0
ENEMY_CAR_LENGTH = 11.0
MAX_SPEED = 100.0
MIN_SPEED = -10.0
SPEED_INCREMENT = 5.0
MAX_Y_POSITION = ACTION_SCREEN_HEIGHT + ENEMY_CAR_LENGTH # Max Y for enemy cars
MIN_Y_POSITION = 0.0 # Min Y for enemy cars (spawn just above the screen)
MIN_DISTANCE_BETWEEN_CARS = 40.0  # Minimum Y distance between adjacent enemy cars

PASS_THRESHOLD = ACTION_SCREEN_HEIGHT  # Distance for passed cars to disappear



class RacingCyEnv():
    def __init__(self):
        super().__init__()
        self.c_env = CRacingCy()
        # Fixed observation space for 15 cars
        self.observation_space = gymnasium.spaces.Box(
            low=-1, high=MAX_Y_POSITION, shape=(37,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(5)  # NOOP, ACCEL, DECEL, LEFT, RIGHT

        self.render_mode = 'human'
        self.client = None
        self.emulated = None
        self.possible_agents = list(range(1))

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)

        self.c_env.reset()
        return self._check_observation(self.c_env.get_state()), {}

    def step(self, action):
        
        # print(f"Step called with action: {action}")
        
        state, reward, done, truncated, info = self.c_env.step(action)
        
        # print(f"State: {state}, Reward: {reward}, Done: {done}, Truncated: {truncated}, Info: {info}")
        
        return self._check_observation(state), reward, done, truncated, info

    def render(self):
        if self.client is None:
            self.client = RaylibClient()

        state = self.c_env.get_state()
        frame, action = self.client.render(state, self.num_enemy_cars)
        return frame

    def close(self):
        pass
    
    
    def _check_observation(self, obs):
        # Define valid observation ranges for all 37 elements
        low = -100 # np.array([0.0, 0.0, MIN_SPEED, 0.0, 0.0] + [-1.0, MIN_Y_POSITION] * 15)  # Player data + 15 cars
        high = 1000 # np.array([ROAD_WIDTH, MAX_Y_POSITION, MAX_SPEED, 30, 200] + [2.0, MAX_Y_POSITION] * 15)

        # print(f"Observation: {obs}")

        # Identify values outside the possible range and set those positions to -1
        out_of_range = (obs < low) | (obs > high)
        if np.any(out_of_range):
            print(f"Warning: Observation out of range: {obs}")
            obs[out_of_range] = -1

        # print(f'obs after: {obs}')

        return obs



class RaylibClient:
    def __init__(self, width=160, height=210):
        self.width = width
        self.height = height

        # Initialize Raylib once in the constructor
        from raylib import rl
        rl.InitWindow(width, height, "PufferLib Racing".encode())
        rl.SetTargetFPS(60)  # Set the target frames per second once
        self.rl = rl  # Store the raylib instance

        from cffi import FFI
        self.ffi = FFI()

        # Define colors as tuples or lists (RGBA)
        self.GREEN = (0, 255, 0, 255)
        self.BLUE = (0, 121, 241, 255)
        self.RED = (230, 41, 55, 255)
        self.GRAY = (200, 200, 200, 255)
        self.WHITE = (255, 255, 255, 255)
        self.BLACK = (0, 0, 0, 255)  # Define BLACK manually

    def render(self, state, num_enemy_cars):
        # Rendering logic (Raylib should already be initialized)
        action = None
        if self.rl.IsKeyDown(self.rl.KEY_UP):
            action = 1  # ACCEL
        elif self.rl.IsKeyDown(self.rl.KEY_DOWN):
            action = 2  # DECEL
        elif self.rl.IsKeyDown(self.rl.KEY_LEFT):
            action = 3  # LEFT
        elif self.rl.IsKeyDown(self.rl.KEY_RIGHT):
            action = 4  # RIGHT

        self.rl.BeginDrawing()
        self.rl.ClearBackground(self.GREEN)  # Use self.GREEN defined earlier

        # Draw road
        road_width = 90
        road_center_x = self.width // 2
        road_left_edge = road_center_x - road_width // 2
        self.rl.DrawRectangle(road_left_edge, 0, road_width, self.height, self.GRAY)  # Use self.GRAY

        # Draw lane dividers
        lane_width = road_width // 3
        self.rl.DrawLine(road_left_edge + lane_width, 0, road_left_edge + lane_width, self.height, self.WHITE)
        self.rl.DrawLine(road_left_edge + 2 * lane_width, 0, road_left_edge + 2 * lane_width, self.height, self.WHITE)

        # Draw player car
        player_x = int(state[0] * self.width)
        player_y = int(state[1] * self.height)
        self.rl.DrawRectangle(player_x, player_y, 16, 11, self.BLUE)

        # Draw enemy cars
        for i in range(num_enemy_cars):
            enemy_x = int(state[5 + i * 2] * self.width)
            enemy_y = int(state[6 + i * 2] * self.height)
            self.rl.DrawRectangle(enemy_x, enemy_y, 16, 11, self.RED)

        # Draw HUD
        self.rl.DrawRectangle(48, 161, 64, 30, self.RED)
        self.rl.DrawText(f"Score: {int(state[2] * 1000)}".encode(), 56, 162, 10, self.BLACK)  # Encode text
        self.rl.DrawText(f"Day: {int(state[3])}".encode(), 56, 179, 10, self.BLACK)  # Encode text
        self.rl.DrawText(f"Cars: {int(state[4] * 200)}".encode(), 72, 179, 10, self.BLACK)  # Encode text

        self.rl.EndDrawing()
        return self._cdata_to_numpy(), action

    def close(self):
        self.rl.CloseWindow()

    def _cdata_to_numpy(self):
        image = self.rl.LoadImageFromScreen()
        width, height, channels = image.width, image.height, 4
        cdata = self.ffi.buffer(image.data, width * height * channels)
        return np.frombuffer(cdata, dtype=np.uint8).reshape((height, width, channels))[:, :, :3]
