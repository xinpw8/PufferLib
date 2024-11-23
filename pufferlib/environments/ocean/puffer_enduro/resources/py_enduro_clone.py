# py_enduro_clone.py
'''High-perf Enduro Clone'''

import numpy as np
import gymnasium

import pufferlib
from cy_enduro_clone import CyEnduro

class MyEnduro(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                 screen_width=160, screen_height=210, hud_height=55, car_width=10, car_height=10,
                 max_enemies=10, crash_noop_duration=60, day_length=2000,
                 initial_cars_to_pass=5, min_speed=-1.0, max_speed=10.0,
                 buf=None):

        # Observation space: 28 features for Enduro Clone
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(29,), dtype=np.float32)
        # Action space: 5 discrete actions for Enduro Clone
        self.single_action_space = gymnasium.spaces.Discrete(5)  # noop, left, right, speed up, slow down
        self.render_mode = render_mode
        self.num_agents = num_envs

        self.report_interval = 128
        self.human_action = None
        self.tick = 0

        super().__init__(buf)

        # Initialize observations, actions, rewards, terminals arrays
        self.observations = np.zeros((num_envs, 28), dtype=np.float32)
        self.actions = np.zeros(num_envs, dtype=np.int32)
        self.rewards = np.zeros(num_envs, dtype=np.float32)
        self.terminals = np.zeros(num_envs, dtype=np.uint8)
        self.truncateds = np.zeros(num_envs, dtype=np.uint8)

        # Initialize the Cython environment for Enduro Clone
        self.c_envs = CyEnduro(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncateds,
            num_envs, screen_width, screen_height, hud_height, car_width,
            car_height, max_enemies, crash_noop_duration, day_length,
            initial_cars_to_pass, min_speed, max_speed)

    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset()
        return self.observations, []

    def step(self, actions):
        # Assign the actions to the respective buffer
        self.actions[:] = actions
        self.c_envs.step()

        info = []
        # Gather logs every report_interval
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)

        self.tick += 1

        # Return observations, rewards, terminals, and additional info
        return (self.observations.copy(), self.rewards.copy(),
                self.terminals.copy(), self.truncateds.copy(), info)

    def render(self):
        self.c_envs.render()

    def close(self):
        self.c_envs.close()

def test_performance(timeout=10, atn_cache=1024):
    # Create the environment with 1000 parallel instances
    env = MyEnduro(num_envs=1000)
    env.reset()
    tick = 0

    # Generate random actions for performance testing
    actions = np.random.randint(0, 5, (atn_cache, env.num_agents), dtype=np.int32)

    import time
    start = time.time()

    # Run the environment step loop for the given timeout duration
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    # Calculate steps per second (SPS)
    print(f'SPS: {env.num_agents * tick / (time.time() - start)}')

if __name__ == '__main__':
    test_performance()
