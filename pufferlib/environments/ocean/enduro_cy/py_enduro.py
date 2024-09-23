import numpy as np
import gymnasium
from .cy_enduro_cy import CEnduroEnv

class EnduroCyEnv(gymnasium.Env):
    '''Enduro environment wrapped in Cython'''

    def __init__(self, frame_skip=4):
        super().__init__()

        # Initialize Cython-based environment
        self.c_env = CEnduroEnv(frame_skip)

        # Observation and action spaces
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)
        self.action_space = gymnasium.spaces.Discrete(9)

    def reset(self):
        self.c_env.reset()
        return self.c_env.get_screen(), {}

    def step(self, action):
        screen, reward, done = self.c_env.step(action)
        info = {"lives": self.c_env.get_lives(), "score": self.c_env.get_score()}
        return screen, reward, done, False, info

    def render(self):
        # Render the game screen (implement this as needed)
        return self.c_env.get_screen()

    def close(self):
        pass
