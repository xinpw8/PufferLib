import numpy as np
import gymnasium

import pufferlib
# Import the new Enduro binding module
from pufferlib.ocean.enduro import binding

class Enduro(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode='human',
                 report_interval=1, seed=0, env_index=0, buf=None):
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.tick = 0
        self.max_enemies = 10

        # Observations length (from your environment logic)
        obs_size = 8 + (5 * self.max_enemies) + 9 + 1
        self.num_obs = obs_size

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(self.num_obs,), dtype=np.float32
        )
        # Example: 9 discrete actions
        self.single_action_space = gymnasium.spaces.Discrete(9)

        # Setup buffers
        self.observations = np.zeros((self.num_agents, self.num_obs), dtype=np.float32)
        self.actions      = np.zeros((self.num_agents,),             dtype=np.int32)
        self.rewards      = np.zeros((self.num_agents,),             dtype=np.float32)
        self.terminals    = np.zeros((self.num_agents,),             dtype=np.uint8)
        self.truncations  = np.zeros((self.num_agents,),             dtype=np.uint8)

        self.rewards_buffer = []
        super().__init__(buf=buf)

        # Instead of: self.c_envs = CyEnduro(...)
        # We do the vector init. The first 5 arguments are arrays:
        #   (obs, actions, rewards, terminals, truncations)
        # The 6th positional argument is num_envs.
        # Then pass the kwargs for "seed" and "index" (if you want them).
        self.c_envs = binding.init_vec(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed=seed,
            index=env_index
        )

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs)   # calls c_reset internally
        return (self.observations, [])

    def step(self, actions):
        # Put actions in the shared array
        self.actions[:] = actions
        # Step the C environment
        binding.vec_step(self.c_envs)

        # Track average reward for logging
        self.rewards_buffer.append(np.mean(self.rewards))

        info = []
        if self.tick % self.report_interval == 0:
            avg_rew = np.mean(self.rewards_buffer)
            info.append({'rewards': avg_rew})
            self.rewards_buffer = []
            # Possibly also gather the aggregated log from C:
            log_data = binding.vec_log(self.c_envs)
            # Only append if it has actual episode data:
            if log_data and log_data.get('episode_length', 0) > 0:
                info.append(log_data)

        self.tick += 1
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info
        )

    def render(self):
        # If you want to render the 0th environment
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=8192):
    """
    Example test script to measure performance (steps per second).
    """
    num_envs = 4096
    env = Enduro(num_envs=num_envs)
    env.reset()
    tick = 0

    # Generate random actions
    actions = np.random.randint(0, env.single_action_space.n, (atn_cache, num_envs))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    sps = num_envs * tick / (time.time() - start)
    print(f"SPS: {sps:,.0f}")

if __name__ == '__main__':
    test_performance()
