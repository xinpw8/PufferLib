'''
This is a cython version of the trash_pickup env.
It works but its actually slightly slower (4%) than just the normal python version.
Figured I would keep it around as reference for anyone who wants to try multi-agent with cython bindings.
'''
import time
import numpy as np
from gymnasium import spaces

import pufferlib
from pufferlib.ocean.trash_pickup.cy_trash_pickup import CyTrashPickup


class TrashPickupEnv(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=1, buf=None, grid_size=10, num_agents=3, num_trash=15, num_bins=2, max_steps=300):
        # Env Setup
        self.num_agents = num_envs * num_agents
        self.render_mode = render_mode
        self.report_interval = report_interval

        # Validate num_agents
        if not isinstance(num_agents, int) or num_agents <= 0:
            raise ValueError("num_agents must be an integer greater than 0.")
        self.num_agents = num_agents

        # Handle num_trash input
        if not isinstance(num_trash, int) or num_trash <= 0:
            raise ValueError("num_trash must be an int >= 0")
        self.num_trash = num_trash

        # Handle num_bins input
        if not isinstance(num_bins, int) or num_bins <= 0:
            raise ValueError("num_bins must be an int >= 0")
        self.num_bins = num_bins

        if not isinstance(max_steps, int) or max_steps <= 10:
            raise ValueError("max_steps must be an int >= 10")
        self.max_steps = max_steps

        # Calculate minimum required grid size
        min_grid_size = int((num_agents + self.num_trash + self.num_bins) ** 0.5) + 1
        if not isinstance(grid_size, int) or grid_size < min_grid_size:
            raise ValueError(
                f"grid_size must be an integer >= {min_grid_size}. "
                f"Received grid_size={grid_size}, with num_agents={num_agents}, num_trash={self.num_trash}, and num_bins={self.num_bins}."
            )
        self.grid_size = grid_size

        num_obs_trash = num_trash * 3  # [presence, x pos, y pos] for each trash
        num_obs_bin = num_bins * 2  # [x pos, y pos] for each bin
        num_obs_agent = num_agents * 3  # [carrying trash, x pos, y pos] for each agent
        self.num_obs = num_obs_trash + num_obs_bin + num_obs_agent;

        self.single_observation_space = spaces.Box(low=0, high=1,
            shape=(self.num_obs,), dtype=np.float32)
        self.single_action_space = spaces.Discrete(4)

        super().__init__(buf=buf)
        self.c_envs = CyTrashPickup(self.observations, self.actions, self.rewards, self.terminals, num_envs, num_agents, grid_size, num_trash, num_bins, max_steps)

    def reset(self, seed=None):
        self.c_envs.reset()
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()
        
    def close(self):
        self.c_envs.close() 
