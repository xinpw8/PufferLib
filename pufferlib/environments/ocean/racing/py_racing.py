# py_racing.py
import numpy as np
import gymnasium

import pufferlib.environment
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



class RacingCyEnv(pufferlib.environment.PufferEnv):
    def __init__(self):
        super().__init__()
        self.num_agents = 1
        self.c_env = CRacingCy()
        self.step_count = 0
        self.observation_space = gymnasium.spaces.Box(
            low=-1, high=MAX_Y_POSITION, shape=(37,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(5)  # NOOP, ACCEL, DECEL, LEFT, RIGHT
        
        self.emulated = None
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.done = 0
        self.render_mode = 'human'
        self.client = None

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
            
        self.step_count = 0
        self.c_env.reset()
        return self.c_env.get_state(), {}

    def step(self, action):
        state, reward, done, truncated, info = self.c_env.step(action)
        return state, reward, done, truncated, info

    # Rename this method from render_step to render
    def render(self):
        if self.client is None:
            self.client = RaylibClient()

        # Capture the human-controlled action via rendering
        state = self.c_env.get_state()
        frame, action = self.client.render(state)  # Capture the action from the human player

        # Use the captured action to step the environment
        state, reward, done, truncated, info = self.step(action)

        return frame, state, reward, done, truncated, info

    def close(self):
        if self.client:
            self.client.close()



class RaylibClient:
    def __init__(self, width=160, height=210):
        self.width = width
        self.height = height

        # Initialize Raylib once in the constructor
        from raylib import rl
        rl.InitWindow(width, height, "PufferLib Racing".encode())
        rl.SetTargetFPS(60)  # Set the target frames per second
        self.rl = rl  # Store the raylib instance
        from cffi import FFI
        self.ffi = FFI()

        # Define colors
        self.GREEN = (0, 255, 0, 255)
        self.BLUE = (0, 121, 241, 255)
        self.RED = (230, 41, 55, 255)
        self.GRAY = (200, 200, 200, 255)
        self.WHITE = (255, 255, 255, 255)

    def render(self, state):
        # Capture player input for controlling the car
        action = ACTION_NOOP  # Default action

        if self.rl.IsKeyDown(self.rl.KEY_UP):
            action = ACTION_ACCEL
        elif self.rl.IsKeyDown(self.rl.KEY_DOWN):
            action = ACTION_DECEL
        elif self.rl.IsKeyDown(self.rl.KEY_LEFT):
            action = ACTION_LEFT
        elif self.rl.IsKeyDown(self.rl.KEY_RIGHT):
            action = ACTION_RIGHT

        self.rl.BeginDrawing()
        self.rl.ClearBackground(self.GREEN)

        # Draw road
        road_width = 90
        road_center_x = self.width // 2
        road_left_edge = road_center_x - road_width // 2
        self.rl.DrawRectangle(road_left_edge, 0, road_width, self.height, self.GRAY)

        # Draw player car
        player_x = int((state[0] / ROAD_WIDTH) * road_width + road_left_edge)
        player_y = self.height - PLAYER_CAR_LENGTH - 10  # Keep player's car near the bottom
        self.rl.DrawRectangle(player_x, int(player_y), int(CAR_WIDTH), int(PLAYER_CAR_LENGTH), self.BLUE)

        # Draw enemy cars
        for i in range(15):
            enemy_lane = state[5 + i * 2]
            enemy_y = state[6 + i * 2]

            if enemy_lane != -1 and enemy_y != -1:  # Check if the car is active
                # Calculate screen positions
                enemy_x = int(road_left_edge + (enemy_lane * (road_width / 3)))
                enemy_y_screen = int(self.height - (enemy_y - state[1]) - ENEMY_CAR_LENGTH)

                # Only render if the car is within the visible area
                if 0 <= enemy_y_screen < self.height:
                    self.rl.DrawRectangle(enemy_x, enemy_y_screen, int(CAR_WIDTH), int(ENEMY_CAR_LENGTH), self.RED)

        self.rl.EndDrawing()

        return self._cdata_to_numpy(), action


        # Return the action for human control
        return self._cdata_to_numpy(), action

    def close(self):
        self.rl.CloseWindow()

    def _cdata_to_numpy(self):
        image = self.rl.LoadImageFromScreen()
        width, height, channels = image.width, image.height, 4
        cdata = self.ffi.buffer(image.data, width * height * channels)
        return np.frombuffer(cdata, dtype=np.uint8).reshape((height, width, channels))[:, :, :3]
