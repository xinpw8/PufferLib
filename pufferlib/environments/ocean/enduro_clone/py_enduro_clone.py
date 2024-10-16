'''High-perf Enduro Clone'''

import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

sys.path.append(os.path.join(os.path.dirname(__file__), 'pufferlib', 'environments', 'ocean_build', 'enduro_clone'))


import numpy as np
import gymnasium

import pufferlib
# from .cy_enduro_clone import CyEnduro
from pufferlib.environments.ocean_build.enduro_clone.cy_enduro_clone import CyEnduro

class MyEnduro(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                 screen_width=160, screen_height=210, hud_height=55, car_width=10, car_height=10,
                 max_enemies=10, crash_noop_duration=60, day_length=2000,
                 initial_cars_to_pass=5, min_speed=-1.0, max_speed=10.0,
                 buf=None):
        
        self.single_observation_space = gymnasium.spaces.Box(low=0.0, high=1.0,
            shape=(28,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(5)  # 5 actions: noop, left, right, speed up, slow down
        self.render_mode = render_mode
        self.num_agents = num_envs

        self.human_action = None
        self.tick = 0
        
        # # Ensure the observations array is 2D
        # self.observations = np.zeros((num_envs, 37), dtype=np.float32)
        self.observations = np.zeros((num_envs, 28), dtype=np.float32)
        # self.actions = np.zeros(num_envs, dtype=np.uint8)
        # self.rewards = np.zeros(num_envs, dtype=np.float32)
        # self.terminals = np.zeros(num_envs, dtype=np.uint8)
    

        super().__init__(buf)
        self.c_envs = CyEnduro(
            self.observations.astype(np.float32, copy=False), 
            self.actions.astype(np.uint8, copy=False),
            self.rewards.astype(np.float32, copy=False),
            self.terminals.astype(np.uint8, copy=False),
            num_envs, screen_width, screen_height, hud_height, car_width,
            car_height, max_enemies, crash_noop_duration, day_length,
            initial_cars_to_pass, min_speed, max_speed)
 
    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset()
        return self.observations, []

    def step(self, actions):
        print(f'.py step actions: {actions}')
        actions = actions # np.random.randint(0, 5, size=(1024, self.num_agents), dtype=np.uint8)

        self.actions[:] = actions.astype(np.uint8)
        self.c_envs.step()

        info = []

        log = self.c_envs.log()
        if log['episode_length'] > 0:
            info.append(log)

        self.tick += 1
        print(f'.py step log: {log}')
        print(f'.py step self.tick: {self.tick}')
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()

def test_performance(timeout=10, atn_cache=16): # 1024
    num_envs = 8
    env = MyEnduro(num_envs)
    env.reset()
    tick = 0
    try:
        actions = np.random.randint(0, 5, (atn_cache, env.num_agents)).astype(np.uint8)
    except Exception as e:
        print(f"Error in step: {e}")

    

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: %f', env.num_agents * tick / (time.time() - start))

if __name__ == '__main__':
    test_performance()
