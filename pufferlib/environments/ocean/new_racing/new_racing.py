# new_racing.py
import numpy as np
import gymnasium
import pufferlib

from raylib import rl
from pufferlib.environments.ocean.racing.new_cy_racing import CyRacingEnv

class MyRacingEnv(pufferlib.PufferEnv):
    def __init__(self, render_mode=None, num_envs=1, report_interval=128):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1, high=160, shape=(37,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(5)  # NOOP, ACCEL, DECEL, LEFT, RIGHT
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__()
        self.c_envs = CyRacingEnv(self.observations, self.actions, self.rewards, self.terminals)

    def reset(self, seed=None):
        self.c_envs.reset()
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()
        self.tick += 1

        return (self.observations, self.rewards, self.terminals, self.truncations, [])

    def render(self):
        self.c_envs.render()

def test_performance(timeout=10, atn_cache=1024):
    env = MyRacingEnv(num_envs=1000)
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
