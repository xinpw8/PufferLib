import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean.cartpole.cy_cartpole import CyCartPole

class Cartpole(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode='human', report_interval=1, frame_skip=1, buf=None):
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.frame_skip = frame_skip
        self.tick = 0
        
        # CartPole has 4 observations: cart position, cart velocity, pole angle, pole angle velocity
        obs_size = 4
        self.num_obs = obs_size

        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(2)
        
        self.observations = np.zeros((self.num_agents, self.num_obs), dtype=np.float32)
        self.actions = np.zeros((self.num_agents,), dtype=np.uint8)
        self.rewards = np.zeros((self.num_agents,), dtype=np.float32)
        self.terminals = np.zeros((self.num_agents,), dtype=np.uint8)
        self.truncations = np.zeros((self.num_agents,), dtype=np.uint8)

        super().__init__(buf=buf)

        self.c_envs = CyCartPole(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            num_envs,
            frame_skip,
            width=800,
            height=600,
            max_steps=200,
            continuous=0
        )
    
    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset()
        return self.observations, {}
    
    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()
        info = []
        if self.tick % self.report_interval == 0:
            info.append({'rewards': self.rewards})
            log = self.c_envs.log()

            if log['episode_length'] > 0:
                info.append({
                    'episode_return': log['episode_return'],
                    'episode_length': log['episode_length'],
                    'score': log['score']
                })

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

def test_performance(timeout=10, atn_cache=8192):
    num_envs = 4096
    env = Cartpole(num_envs=num_envs)
    env.reset()
    tick = 0

    actions = np.random.randint(0, env.single_action_space.n, (atn_cache, num_envs)).astype(np.float32)  # Convert to float32

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