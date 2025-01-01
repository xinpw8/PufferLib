'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.squared.cy_squared import CySquared

NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4

EMPTY = 0
AGENT = 1
TARGET = 2

class PySquared(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode='ansi', size=11, buf=None):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(size*size,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(5)
        self.render_mode = render_mode
        self.num_agents = 1

        super().__init__(buf)
        self.size = size

    def reset(self, seed=None):
        self.observations[0, :] = EMPTY
        self.observations[0, self.size*self.size//2] = AGENT
        self.r = self.size//2
        self.c = self.size//2
        self.tick = 0
        while True:
            target_r, target_c = np.random.randint(0, self.size, 2)
            if target_r != self.r or target_c != self.c:
                self.observations[0, target_r*self.size + target_c] = TARGET
                break

        return self.observations, []

    def step(self, actions):
        atn = actions[0]
        self.terminals[0] = False
        self.rewards[0] = 0

        self.observations[0, self.r*self.size + self.c] = EMPTY

        if atn == DOWN:
            self.r += 1
        elif atn == RIGHT:
            self.c += 1
        elif atn == UP:
            self.r -= 1
        elif atn == LEFT:
            self.c -= 1

        info = []
        pos = self.r*self.size + self.c
        if (self.tick > 3*self.size
                or self.r < 0
                or self.c < 0
                or self.r >= self.size
                or self.c >= self.size):
            self.terminals[0] = True
            self.rewards[0] = -1.0
            info = {'reward': -1.0}
            self.reset()
        elif self.observations[0, pos] == TARGET:
            self.terminals[0] = True
            self.rewards[0] = 1.0
            info = {'reward': 1.0}
            self.reset()
        else:
            self.observations[0, pos] = AGENT
            self.tick += 1

        return self.observations, self.rewards, self.terminals, self.truncations, info

    def render(self):
        chars = []
        grid = self.observations.reshape(self.size, self.size)
        for row in grid:
            for val in row:
                if val == AGENT:
                    color = 94
                elif val == TARGET:
                    color = 91
                else:
                    color = 90
                chars.append(f'\033[{color}m██\033[0m')
            chars.append('\n')
        return ''.join(chars)

    def close(self):
        pass
