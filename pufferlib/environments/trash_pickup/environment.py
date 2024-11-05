import functools
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from .cy_trash_pickup import CyTrashPickupEnv

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.wrappers



class TrashPickupEnv(ParallelEnv):
    def __init__(self, grid_size=10, num_agents=3, num_trash=15, num_bins=2, max_steps=300):
        self.grid_size = grid_size
        self.num_trash = num_trash
        self.num_bins = num_bins
        self.max_steps = max_steps

        # self.single_observation_space = gym.spaces.Dict({
        #     'agent_position': gym.spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.float32),
        #     'carrying_trash': gym.spaces.Discrete(2),
        #     'grid': gym.spaces.Box(low=0, high=4, shape=(grid_size, grid_size), dtype=np.float32),
        # })
        # self.single_action_space = gym.spaces.Discrete(4)

        self._num_agents = num_agents
        self.possible_agents = ["agent_" + str(i) for i in range(self._num_agents)]

        # Define the action space
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.possible_agents}

        # Observation space includes the agent's position, whether it's carrying trash, and the grid state
        self.observation_spaces = {
            agent: spaces.Dict({
                'agent_position': spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32),
                'carrying_trash': spaces.Discrete(2),  # 0 or 1
                'grid': spaces.Box(low=0, high=4, shape=(self.grid_size, self.grid_size), dtype=np.int32)
            })
            for agent in self.possible_agents
        }

       

        super().__init__()
        self.env = CyTrashPickupEnv(grid_size, num_agents, num_trash, num_bins, max_steps)

    def reset(self, seed=None):
        observations = self.env.reset()
        return observations

    def step(self, actions):
        observations, rewards, dones, infos = self.env.step(actions)
        return observations, rewards, dones, infos

    def render(self):
        # If you want to implement rendering, you can do it here
        pass

    def close(self):
        self.env = None

def env_creator():
    return functools.partial(make)

def make():
    env = TrashPickupEnv()
    # env = aec_to_parallel_wrapper(env)
    env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)