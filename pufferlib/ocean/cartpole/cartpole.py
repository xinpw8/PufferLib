import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean.cartpole.cy_cartpole import CyCartPole

class Cartpole(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode='human', report_interval=1, continuous=False, buf=None, seed=1):
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.tick = 0
        self.is_continuous = continuous

        self.num_obs = 4
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )
        if self.is_continuous:
            self.single_action_space = gymnasium.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        else:
            self.single_action_space = gymnasium.spaces.Discrete(2)

        super().__init__(buf=buf)
        
        self.actions = np.zeros(self.num_agents, dtype=np.float32)
        self.terminals = np.zeros(self.num_agents, dtype=np.uint8)
        self.truncations = np.zeros(self.num_agents, dtype=np.uint8)

        self.c_envs = CyCartPole(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            num_envs,
            int(self.is_continuous),
        )
   
    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset()
        return self.observations, []
   
    def step(self, actions):
        if self.is_continuous:
            self.actions[:] = np.clip(actions.flatten(), -1.0, 1.0)
        else:
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
        if self.render_mode == 'human':
            self.c_envs.render()
   
    def close(self):
        self.c_envs.close()

def test_performance(timeout=10, atn_cache=8192, is_continuous=True):
    """Benchmark environment performance."""
    num_envs = 4096
    env = Cartpole(num_envs=num_envs, continuous=is_continuous)
    env.reset()
    tick = 0

    if env.is_continuous:
        actions = np.random.uniform(-1, 1, (atn_cache, num_envs, 1)).astype(np.float32)
    else:
        actions = np.random.randint(0, env.single_action_space.n, (atn_cache, num_envs)).astype(np.int8)

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
