# enduro_clone.py

import numpy as np
import gymnasium
import pufferlib
from pufferlib.environments.ocean.puffer_enduro.cy_puffer_enduro import CyEnduro
import torch

class MyEnduro(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                 report_interval=1, buf=None):

        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.tick = 0
        self.max_enemies = 10 # max_enemies

        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval

        # max_enemies = 10
        obs_size = 6 + 2 * 10 + 3
        self.num_obs = obs_size

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(self.num_obs,), dtype=np.float32)
        # noop, fire (accelerate), down (decelerate), left, right,
        # fire-left, fire-right, down-left, down-right
        self.single_action_space = gymnasium.spaces.Discrete(9)

        super().__init__(buf=buf)

        self.c_envs = CyEnduro(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs
        )
    
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
        
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info
        )
    
    def render(self):
        self.c_envs.render()

    def close(self):
        self.c_envs.close()
        
    
    def validate_probabilities(prob_tensor):
        if torch.isnan(prob_tensor).any() or torch.isinf(prob_tensor).any() or (prob_tensor < 0).any():
            raise ValueError("Invalid probability values detected")
        return prob_tensor

        
def test_performance(timeout=10, atn_cache=1024):
    num_envs = 1000
    env = MyEnduro(num_envs=num_envs)
    env.reset()
    tick = 0

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