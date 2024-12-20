# blastar.py
import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean.blastar.cy_blastar import CyBlastar

class Blastar(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, buf=None):
        # Observation space: 6 floats (normalized positions, bullet states)
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(27,), dtype=np.float32
        )
        # Action space: 6 discrete actions (no-op, left, right, up, down, fire)
        self.single_action_space = gymnasium.spaces.Discrete(6)
        self.render_mode = render_mode
        self.num_agents = num_envs

        self.tick = 0
        self.report_interval = 1  # How often to report logs

        super().__init__(buf)

        # Initialize numpy arrays
        # Not actually necessary, but it's here
        self.observations = np.zeros((num_envs, 27), dtype=np.float32)
        self.actions = np.zeros((num_envs,), dtype=np.int32)
        self.rewards = np.zeros((num_envs,), dtype=np.float32)
        self.terminals = np.zeros((num_envs,), dtype=np.uint8)

        # Initialize C environments
        self.c_envs = CyBlastar(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            num_envs
        )

    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset()
        return self.observations.copy(), []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)

        self.tick += 1
        return (self.observations.copy(), self.rewards.copy(),
                self.terminals.copy(), self.truncations, info)

    def render(self):
        self.c_envs.render()

    def close(self):
        self.c_envs.close()

def test_performance(timeout=10, atn_cache=1024):
    env = Blastar(num_envs=1000)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 6, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print('SPS:', env.num_agents * tick / (time.time() - start))

if __name__ == '__main__':
    test_performance()
