import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean.pfr_native import binding


class PfrNative(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, report_interval=128, buf=None, seed=0, **kwargs):
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.tick = 0
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(133,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(9)
        super().__init__(buf=buf)
        # C binding reads actions as float* — create float32 buffer
        self.float_actions = np.zeros_like(self.actions, dtype=np.float32)
        self.c_envs = binding.vec_init(
            self.observations, self.float_actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed)

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed or 0)
        return self.observations, []

    def step(self, actions):
        self.float_actions[:] = actions
        binding.vec_step(self.c_envs)
        info = []
        self.tick += 1
        if self.tick % self.report_interval == 0:
            info.append(binding.vec_log(self.c_envs))
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)
