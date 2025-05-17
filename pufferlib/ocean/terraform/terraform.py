'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.terraform import binding

OBS_SIZE = 11

class Terraform(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, num_agents=8, map_size=512,
            render_mode=None, log_interval=32, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(OBS_SIZE*OBS_SIZE,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.MultiDiscrete([5, 5, 2, 2, 2])
        self.render_mode = render_mode
        self.num_agents = num_envs*num_agents
        self.log_interval = log_interval

        super().__init__(buf)
        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(
                self.observations[i*num_agents:(i+1)*num_agents],
                self.actions[i*num_agents:(i+1)*num_agents],
                self.rewards[i*num_agents:(i+1)*num_agents],
                self.terminals[i*num_agents:(i+1)*num_agents],
                self.truncations[i*num_agents:(i+1)*num_agents],
                seed,
                size=map_size,
                num_agents=num_agents,
            )
            c_envs.append(c_env)

        self.c_envs = binding.vectorize(*c_envs)
 
    def reset(self, seed=None):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1

        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        episode_returns = self.rewards[self.terminals]

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    TIME = 10
    env = Terraform(num_envs=512, num_agents=4, render_mode='human', size=11)
    actions = np.random.randint(0, 5, 2048)
    env.reset()

    import time
    steps = 0
    start = time.time()
    while time.time() - start < TIME:
        env.step(actions)
        steps += 2048

    print('Cython SPS:', steps / (time.time() - start))





