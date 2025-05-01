'''
High-perf Boids
Inspired by https://people.ece.cornell.edu/land/courses/ece4760/labs/s2021/Boids/Boids.html
'''

from code import interact
import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.boids import binding

class Boids(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        num_boids=1,
        max_steps=1000,
        buf=None,
        render_mode=None,
        report_interval=1,
        seed=0
    ):
        self.num_agents = num_envs
        self.num_boids = num_boids
        self.max_steps = max_steps

        # Define single observation space for one agent (boid)
        self.single_observation_space = gymnasium.spaces.Box(
            -1000.0, 1000.0, shape=(self.num_boids, 4), dtype=np.float32
        )
        
        # Keep the original action space shape that the policy expects
        self.single_action_space = gymnasium.spaces.Box(
            -3.0, 3.0, shape=(self.num_boids, 2), dtype=np.float32
        )

        self.render_mode = render_mode
        self.report_interval = report_interval

        super().__init__(buf)

        # Create C binding with flattened action buffer
        # We need to manually create a flattened action buffer to pass to C
        self.flat_actions = np.zeros((self.num_agents * self.num_boids * 2), dtype=np.float32)
        
        self.c_envs = binding.vec_init(
            self.observations,
            self.flat_actions,  # Pass the flattened buffer to C
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            num_boids,
            max_steps=max_steps
        )

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        # Clip actions to valid range
        clipped_actions = np.clip(actions, self.single_action_space.low, self.single_action_space.high)
        
        # Copy the clipped actions to our flat actions buffer for C binding
        # Flatten from [num_agents, num_boids, 2] to a 1D array for C
        self.flat_actions[:] = clipped_actions.reshape(-1)
        
        # Save the original actions for the experience buffer
        self.actions[:] = clipped_actions
        
        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            log_data = binding.vec_log(self.c_envs)
            if log_data:
                info.append(log_data)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    env = Boids(num_envs=1000)
    env.reset()
    tick = 0

    # Generate random actions with proper shape: [cache_size, num_agents, action_dim]
    actions = np.random.uniform(-3.0, 3.0, (atn_cache, env.num_agents, 2))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: {env.num_agents * tick / (time.time() - start)}')


if __name__ == '__main__':
    test_performance()