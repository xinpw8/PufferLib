# racing.py
import numpy as np
import gymnasium
import pufferlib
from pufferlib import PufferEnv
from pufferlib.environments.ocean.racing.cy_racing import CyEnduro

MAX_CARS = 100  # Must match value in racing.h


class MyRacing(PufferEnv):

    def __init__(self, num_envs=1, render_mode=None,
                 width=500, height=640, player_width=40, 
                 player_height=60, other_car_width=30, other_car_height=60,
                 player_speed=3.0, base_car_speed=2.0, max_player_speed=10.0,
                 min_player_speed=1.0, speed_increment=0.5, max_score=100):
        
        super().__init__()
        self.num_envs = num_envs
        self.num_agents = num_envs
        self.render_mode = render_mode

        # Environment dimensions and data
        self.width = width
        self.height = height
        self.player_width = player_width
        self.player_height = player_height
        self.other_car_width = other_car_width
        self.other_car_height = other_car_height
        self.player_speed = player_speed
        self.base_car_speed = base_car_speed
        self.max_player_speed = max_player_speed
        self.min_player_speed = min_player_speed
        self.speed_increment = speed_increment
        self.max_score = max_score

        # Observation and action spaces
        self.num_obs = 7
        self.num_act = 3

        self.observation_space = gymnasium.spaces.Box(low=0, high=1,
                                                      shape=(self.num_obs,), dtype=np.float32)
        self.single_observation_space = self.observation_space
        self.action_space = gymnasium.spaces.Discrete(self.num_act)
        self.single_action_space = self.action_space
        self.human_action = None
        self.emulated = None
        self.done = False
        
        # Internal buffer for storing state
        self.buf = self.init_buffer()
        self.terminals_uint8 = np.zeros(self.num_agents, dtype=np.uint8)
        
        self.c_envs = [
            CyEnduro(
                self.buf.observations[i], self.buf.actions[i:i+1],
                self.buf.rewards[i:i+1], self.terminals_uint8[i:i+1],
                self.buf.player_x_y[i], self.buf.other_cars_x_y[i], 
                self.buf.other_cars_active[i], self.buf.score_day[i],
                self.width, self.height, self.player_width, self.player_height,
                self.other_car_width, self.other_car_height, self.player_speed,
                self.base_car_speed, self.max_player_speed, self.min_player_speed,
                self.speed_increment, self.max_score
            )
            for i in range(self.num_envs)
        ]

    def reset(self, seed=None):
        for env in self.c_envs:
            env.reset()
        return self.buf.observations, {}

    def step(self, actions):
        self.buf.actions[:] = actions
        for env in self.c_envs:
            env.step()
        return (self.buf.observations, self.buf.rewards,
                self.buf.terminals, {}, {})

    def render(self):
        self.c_envs[0].render()

    def close(self):
        for env in self.c_envs:
            env.close()

    def init_buffer(self):
        """ Initialize buffer arrays for observations, rewards, etc."""
        return pufferlib.namespace(
            observations=np.zeros((self.num_agents, self.num_obs), dtype=np.float32),
            rewards=np.zeros(self.num_agents, dtype=np.float32),
            actions=np.zeros(self.num_agents, dtype=np.uint32),
            terminals=np.zeros(self.num_agents, dtype=np.bool),
            player_x_y=np.zeros((self.num_agents, 2), dtype=np.float32),
            other_cars_x_y=np.zeros((self.num_agents, MAX_CARS * 2), dtype=np.float32),
            other_cars_active=np.zeros((self.num_agents, MAX_CARS), dtype=np.int32),
            score_day=np.zeros((self.num_agents, 2), dtype=np.uint32),
        )
