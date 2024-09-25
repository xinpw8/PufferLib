# py_enduro.py

import gym
import numpy as np
from pufferlib.environments.ocean.enduro_cy.cy_enduro_cy import CEnduroEnv

class EnduroEnv(gym.Env):  # Inherit from gym.Env
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, obs_type="rgb", frameskip=4, **kwargs):
        super(EnduroEnv, self).__init__()  # Initialize the gym.Env base class
        self.env = CEnduroEnv(frame_skip=frameskip)

        # Define action and observation space (required by gym.Env)
        self.action_space = gym.spaces.Discrete(18)  # Assuming 18 actions, modify as needed
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.reset(seed=seed)
        observation, info = self.env.reset(options=options)
        return observation, info
    
    def step(self, action):
        """Takes a step in the environment."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        truncated = False  # If no truncation logic exists
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render the environment."""
        return self.env.get_screen()

    def close(self):
        """Close the environment."""
        pass  # Nothing to do here since ALE handles closing automatically
