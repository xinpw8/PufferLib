# enduro_clone.py

import numpy as np
import gymnasium

import pufferlib
from pufferlib.environments.ocean.enduro_clone.cy_enduro_clone import CyEnduro

class MyEnduro(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=1,
                 width=160.0, height=210.0, hud_height=55.0,
                 car_width=10.0, car_height=10.0,
                 max_enemies=10,
                 crash_noop_duration=60.0, day_length=2000.0,
                 initial_cars_to_pass=5, min_speed=-1.0, max_speed=10.0,
                 buf=None):

        # Environment configuration
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval

        # Calculate observation size based on max_enemies
        obs_size = 6 + 2 * max_enemies + 3  # Total features from compute_observations
        self.num_obs = obs_size

        # Define observation and action spaces
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(self.num_obs,), dtype=np.float32)
        # noop, fire (accelerate), down (decelerate), left, right,
        # fire-left, fire-right, down-left, down-right
        self.single_action_space = gymnasium.spaces.Discrete(9)

        # Initialize parent class
        super().__init__(buf=buf)

        # Allocate arrays for observations, actions, rewards, terminals
        self.observations = np.zeros((num_envs, self.num_obs), dtype=np.float32)
        self.actions = np.zeros(num_envs, dtype=np.int32)
        self.rewards = np.zeros(num_envs, dtype=np.float32)
        self.terminals = np.zeros(num_envs, dtype=np.uint8)
        self.truncations = np.zeros(num_envs, dtype=np.uint8)

        # Initialize the Cython environment
        self.c_envs = CyEnduro(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            width, height, hud_height,
            car_width, car_height,
            max_enemies,
            crash_noop_duration, day_length,
            initial_cars_to_pass, min_speed, max_speed
        )

    def reset(self, seed=None):
        self.c_envs.reset()
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        # Assign actions to the C environment
        self.actions[:] = actions
        self.c_envs.step()
        self.tick += 1

        info = []
        # Collect logs at specified intervals
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)

        # Return observations, rewards, terminals, and info
        # print(f'from python: {self.observations, self.rewards, self.terminals, self.truncations, info}')
        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()

    def close(self):
        self.c_envs.close()

def test_performance(timeout=10, atn_cache=1024):
    num_envs = 1000
    env = MyEnduro(num_envs=num_envs)
    env.reset()
    tick = 0

    # Generate random actions for performance testing
    actions = np.random.randint(0, env.single_action_space.n, (atn_cache, num_envs))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    sps = num_envs * tick / (time.time() - start)
    print(f'SPS: {sps:,}')

if __name__ == '__main__':
    test_performance()
