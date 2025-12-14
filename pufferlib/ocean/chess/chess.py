import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.chess import binding

class Chess(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=1, buf=None, seed=0,
                 max_moves=500, reward_draw=0.0,
                 reward_invalid_piece=-0.01, reward_invalid_move=-0.01, 
                 reward_valid_piece=0.0, reward_valid_move=0.0,
                 reward_material=0.0, reward_position=0.0, reward_castling=0.0, reward_repetition=0.0,
                 render_fps=30, selfplay=1, human_play=0,
                 starting_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                 multi_fen=False,
                 enable_50_move_rule=1, enable_threefold_repetition=1):
        
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.tick = 0
        self.selfplay = selfplay
        
        factor = 2 if selfplay else 1
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(1077*factor,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(96)
        
        fen_curriculum = None
        if str(multi_fen).lower() in ('true', '1', 'yes'):
            import os
            fens_path = os.path.join(os.path.dirname(__file__), 'fens.txt')
            try:
                with open(fens_path, 'r') as f:
                    fen_curriculum = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                if num_envs == 1 or buf is None:
                    print(f"Loaded {len(fen_curriculum)} positions from pufferlib/ocean/chess/fens.txt")
            except FileNotFoundError:
                if num_envs == 1 or buf is None:
                    print("Warning: multi_fen=True but pufferlib/ocean/chess/fens.txt not found, using default starting position")
        
        super().__init__(buf)
        
        if self.selfplay:
            self.actions = np.zeros(num_envs * 2, dtype=np.int32)
        c_envs = []
        for i in range(num_envs):
            c_envs.append(binding.env_init(
                self.observations[i:(i+1)],
                self.actions[i*factor:(i+1)*factor],
                self.rewards[i:(i+1)],
                self.terminals[i:(i+1)],
                self.truncations[i:(i+1)],
                i,
                max_moves=max_moves,
                reward_draw=reward_draw,
                reward_invalid_piece=reward_invalid_piece,
                reward_invalid_move=reward_invalid_move,
                reward_valid_piece=reward_valid_piece,
                reward_valid_move=reward_valid_move,
                reward_material=reward_material,
                reward_position=reward_position,
                reward_castling=reward_castling,
                reward_repetition=reward_repetition,
                render_fps=render_fps,
                selfplay=selfplay,
                human_play=human_play,
                starting_fen=starting_fen,
                fen_curriculum=fen_curriculum,
                enable_50_move_rule=enable_50_move_rule,
                enable_threefold_repetition=enable_threefold_repetition,
                learner_color=i % 2,
                seed=seed + i
            ))
        self.c_envs = binding.vectorize(*c_envs)
    
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []
    
    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        info = [binding.vec_log(self.c_envs)] if self.tick % self.log_interval == 0 else []
        return self.observations, self.rewards, self.terminals, self.truncations, info
    
    def render(self):
        binding.vec_render(self.c_envs, 0)
    
    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 4096
    env = Chess(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 64, (CACHE, 2*N))

    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % CACHE])
        steps += 1

    print('Chess SPS:', int(env.num_agents * steps / (time.time() - start)))