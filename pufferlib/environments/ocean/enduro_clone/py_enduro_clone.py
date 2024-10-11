'''High-perf Enduro Clone'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.environments.ocean.enduro_clone.cy_enduro_clone import CyEnduro

class MyEnduro(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                 width=160, height=210, hud_height=55, car_width=10, car_height=10,
                 max_enemies=10, crash_noop_duration=60, day_length=2000,
                 initial_cars_to_pass=5, min_speed=-1.0, max_speed=10.0,
                 frameskip=4, report_interval=128, buf=None):
        
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(8,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(5)  # 5 actions: noop, left, right, speed up, slow down
        self.render_mode = render_mode
        self.num_agents = num_envs

        self.report_interval = report_interval
        self.human_action = None
        self.tick = 0
        print('buf: ', buf)
        super().__init__(buf=buf)
        self.c_envs = CyEnduro(self.observations, self.actions, self.rewards,
            self.terminals, num_envs, width, height, hud_height, car_width,
            car_height, max_enemies, crash_noop_duration, day_length,
            initial_cars_to_pass, min_speed, max_speed, frameskip)
 
    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset()
        return self.observations, []

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

def test_performance(timeout=10, atn_cache=1024):
    env = MyEnduro(num_envs=1000)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 5, (atn_cache, env.num_envs))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: %f', env.num_envs * tick / (time.time() - start))

if __name__ == '__main__':
    test_performance()
