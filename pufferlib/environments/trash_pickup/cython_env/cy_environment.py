'''
This is a cython version of the trash_pickup env.
It works but its actually slightly slower (4%) than just the normal python version.
Figured I would keep it around as reference for anyone who wants to try multi-agent with cython bindings.
'''

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



class CyTrashPickupEnv(ParallelEnv):
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
        self.agents = self.possible_agents[:]

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

    def _reshape_observations(self, observations):
        """Reshape the grid component of each agent's observation."""
        for agent, obs in observations.items():
            obs['grid'] = np.array(obs['grid']).reshape(self.grid_size, self.grid_size)
        return observations

    def reset(self, seed=None):
        observations = self.env.reset()
        observations = self._reshape_observations(observations)
        return observations

    def step(self, actions):
        observations, rewards, dones, infos = self.env.step(actions)

        # Convert rewards, dones, and infos lists to dictionaries with agent IDs as keys
        agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        rewards = {agent_id: rewards[i] for i, agent_id in enumerate(agent_ids)}
        dones = {agent_id: dones[i] for i, agent_id in enumerate(agent_ids)}
        infos = {agent_id: infos[i] for i, agent_id in enumerate(agent_ids)}
    
        observations = self._reshape_observations(observations)
        
        return observations, rewards, dones, infos

    def render(self):
        # If you want to implement rendering, you can do it here
        pass

    def close(self):
        self.env = None

def env_creator():
    return functools.partial(make)

def make():
    env = CyTrashPickupEnv()
    # env = aec_to_parallel_wrapper(env)
    env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)