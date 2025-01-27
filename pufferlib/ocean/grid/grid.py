import numpy as np
import os

import gymnasium

from raylib import rl, colors

import pufferlib
from pufferlib.ocean import render
from pufferlib.ocean.grid.cy_grid import CGrid

EMPTY = 0
FOOD = 1
WALL = 2
AGENT_1 = 3
AGENT_2 = 4
AGENT_3 = 5
AGENT_4 = 6

PASS = 0
NORTH = 1
SOUTH = 2
EAST = 3
WEST = 4


class PufferGrid(pufferlib.PufferEnv):
    def __init__(self, render_mode='rgb_array', vision_range=3,
            num_envs=4096, report_interval=1024, buf=None):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(self.obs_size*self.obs_size+3,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(5)
        self.render_mode = render_mode
        self.num_agents = num_envs
        super().__init__()
        self.cenv = CGrid(self.observations, self.actions,
            self.rewards, self.dones, self.num_agents)

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
               info.append(log)

        self.tick += 1
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()

    def close(self):
        self.c_envs.close()

def test_performance(timeout=10, atn_cache=1024):
    env = Grid(num_envs=1000)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 2, (atn_cache, env.num_envs))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: %f', env.num_envs * tick / (time.time() - start))
