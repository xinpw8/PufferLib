import numpy as np
import gymnasium

from pufferlib.environments.ocean.racing.cy_racing import CyRacing

class MyRacing:
    def __init__(self, num_envs=1, render_mode=None, width=800, height=600):
        self.num_envs = num_envs
        self.width = width
        self.height = height
        self.render_mode = render_mode

        self.c_envs = [CyRacing(width=self.width, height=self.height) for _ in range(self.num_envs)]

    def reset(self):
        for env in self.c_envs:
            env.reset()

    def step(self, actions):
        for i, env in enumerate(self.c_envs):
            env.step(actions[i])

    def render(self):
        self.c_envs[0].render()
