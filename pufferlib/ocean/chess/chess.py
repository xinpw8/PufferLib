# chess.py
import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.chess import binding

class Chess(pufferlib.PufferEnv):
    """Chess environment compatible with OpenSpiel's action space."""
    
    def __init__(self, num_envs=1, render_mode=None, log_interval=1,
                 reward_valid=0.01, reward_invalid=-0.1,
                 reward_agent_captures_enemy_piece=0.05,
                 reward_enemy_captures_agent_piece=-0.05,
                 reward_win=1.0, reward_draw=0.0, reward_loss=-1.0,
                 reward_check=0.01,
                 max_depth=200,
                 buf=None, seed=0, self_play=False):
        
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.tick = 0
        
        # observations: 21 channels of 8x8 = 8*8*21 = 1344
        self.num_obs = 8*8*21 + 4674 # legal move mask
        # actions: 4674 following openspiel encoding
        self.num_actions = 4674
        
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(self.num_obs,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(self.num_actions)
        
        self.self_play = self_play
        
        super().__init__(buf=buf)
        
        self.actions = np.zeros((num_envs,), dtype=np.int32)
        
        # initialize c environments
        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed,
            reward_valid=reward_valid, reward_invalid=reward_invalid,
            reward_agent_captures_enemy_piece=reward_agent_captures_enemy_piece,
            reward_enemy_captures_agent_piece=reward_enemy_captures_agent_piece,
            reward_win=reward_win, reward_draw=reward_draw, 
            reward_loss=reward_loss, max_depth=max_depth,
            reward_check=reward_check)
        
        # If requested, enable self-play inside the C++ envs immediately so that
        # worker processes inherit the correct behaviour without additional
        # calls from the parent process.
        if self.self_play:
            binding.vec_set_self_play(self.c_envs)
    
    def set_fen(self, env_id: int, fen: str):
        binding.vec_set_fen(self.c_envs, fen)
    
    def reset(self, *, seed=None, fen=None):
        if fen is not None:
            self.set_fen(0, fen)
            self.tick = 0
            return self.observations, []
        
        if seed is None:
            seed = 0
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []
    
    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        
        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))
        
        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)
    
    def render(self):
        import io
        import sys
        from contextlib import redirect_stdout
        
        # Capture stdout from the C++ render function
        f = io.StringIO()
        with redirect_stdout(f):
            binding.vec_render(self.c_envs, 0)
        return f.getvalue()
    
    def close(self):
        binding.vec_close(self.c_envs)


def test_performance(timeout=10, num_envs=1000):
    """Benchmark environment speed."""
    env = Chess(num_envs=num_envs)
    env.reset()
    
    # Pre-generate random actions
    action_cache = np.random.randint(0, env.single_action_space.n, 
                                    (1000, num_envs))
    
    import time
    tick = 0
    start = time.time()
    
    while time.time() - start < timeout:
        actions = action_cache[tick % len(action_cache)]
        env.step(actions)
        tick += 1
    
    sps = num_envs * tick / (time.time() - start)
    print(f'SPS: {sps:,}')


if __name__ == '__main__':
    test_performance()