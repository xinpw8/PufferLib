# racing.py
import numpy as np
import gymnasium
import pufferlib
from pufferlib.environments.ocean.racing.cy_racing import CyEnduro

class MyRacing(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, width=500, height=640,
                 player_width=40, player_height=60, other_car_width=30, other_car_height=60,
                 player_speed=3.0, base_car_speed=2.0, max_player_speed=10.0,
                 min_player_speed=1.0, speed_increment=0.5, max_score=100):
        super().__init__()
        
        # env
        self.num_envs = num_envs
        self.num_agents = num_envs
        self.render_mode = render_mode

        # sim hparams (px, px/tick)
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
        
        # sim data
        self.player_x_y = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.other_cars_x_y = np.zeros((self.num_envs, 200), dtype=np.float32)  # Assuming max 100 cars
        self.other_cars_active = np.zeros((self.num_envs, 100), dtype=np.int32)
        self.score_day = np.zeros((self.num_envs, 2), dtype=np.uint32)

        # spaces
        self.num_obs = 5  # player_lane, player_speed, cars_passed_ratio, day, score
        self.num_act = 4  # Left, Speed Up, Right, Slow Down
        self.observation_space = gymnasium.spaces.Box(low=0, high=1,
                                                      shape=(self.num_obs,), dtype=np.float32)
        self.single_observation_space = self.observation_space
        self.action_space = gymnasium.spaces.Discrete(self.num_act)
        self.single_action_space = self.action_space
        self.human_action = None
        
        self.emulated = None
        self.done = False
        self.buf = self.init_buffer()
        self.actions = np.zeros(self.num_agents, dtype=np.uint8)
        self.terminals_uint8 = np.zeros(self.num_agents, dtype=np.uint8)
        self.tick = 0
        self.report_interval = 128
        self.reward_sum = 0
        self.num_finished_games = 0

    def reset(self, seed=None):
        self.tick = 0
        self.c_envs = []
        for i in range(self.num_envs):
            self.c_envs.append(CyEnduro(
                self.buf.observations[i], self.actions[i:i+1], 
                self.buf.rewards[i:i+1], self.terminals_uint8[i:i+1],
                self.player_x_y[i], self.other_cars_x_y[i], self.other_cars_active[i],
                self.score_day[i], self.width, self.height, self.player_width, self.player_height,
                self.other_car_width, self.other_car_height, self.player_speed, self.base_car_speed,
                self.max_player_speed, self.min_player_speed, self.speed_increment, self.max_score
            ))
        return self.buf.observations, {}

    def step(self, actions):
        self.actions[:] = actions
        for i in range(self.num_envs):
            self.c_envs[i].step()
        
        self.buf.terminals[:] = self.terminals_uint8.astype(bool)
        
        info = {}
        self.reward_sum += self.buf.rewards.mean()
        self.num_finished_games += self.buf.terminals.sum()
        if self.tick % self.report_interval == 0:
            info.update({
                'reward': self.reward_sum / self.report_interval,
                'reward_sum': self.reward_sum,
            })
        
        self.tick += 1
        
        return (self.buf.observations, self.buf.rewards,
                self.buf.terminals, self.buf.truncations, info)

    def close(self):
        for env in self.c_envs:
            env.close()

    def init_buffer(self):
        return pufferlib.namespace(
            observations=np.zeros((self.num_agents, self.num_obs), dtype=np.float32),
            actions=np.zeros(self.num_agents, dtype=np.uint32),
            rewards=np.zeros(self.num_agents, dtype=np.float32),
            terminals=np.zeros(self.num_agents, dtype=np.uint8),
            player_x_y=np.zeros((self.num_agents, 2), dtype=np.float32),
            other_cars_x_y=np.zeros((self.num_agents, 200), dtype=np.float32),  # Assuming max 100 cars
            other_cars_active=np.zeros((self.num_agents, 100), dtype=np.int32),
            score_day=np.zeros((self.num_agents, 2), dtype=np.uint32)
        )