import gymnasium
import numpy as np
import os

import pufferlib
from pufferlib.ocean.chess import binding

CHESS_DIR = os.path.dirname(os.path.abspath(__file__))

class Chess(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=1, buf=None, seed=0,
                 max_moves=500, reward_draw=0.0,
                 reward_invalid_piece=-0.01, reward_invalid_move=-0.01,
                 reward_repetition=0.0,
                 render_fps=30, selfplay=1, human_play=0, random_bot=0,
                 starting_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                 random_fen_pct=0,
                 fen_curric_pct=0,
                 fen_file=None,
                 enable_50_move_rule=1, enable_threefold_repetition=1):
        
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.cumulative_games = 0.0 
        self.tick = 0
        self.selfplay = selfplay
        self.random_fen_pct = random_fen_pct
        
        if fen_file and not os.path.isabs(fen_file):
            fen_file = os.path.join(CHESS_DIR, fen_file)
        self.c_curriculum = binding.shared(fen_file=fen_file)
        
        self.fen_curric_pct = fen_curric_pct
        factor = 2 if selfplay else 1
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(1129*factor,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(97)
        
        super().__init__(buf)
        
        if self.selfplay:
            self.actions = np.zeros(num_envs * 2, dtype=np.int32)
        c_envs = []
        for i in range(num_envs):
            if random_fen_pct > 0 and random_fen_pct < 100:
                use_random_fen = 1 if random.random() < random_fen_pct / 100 else 0
            elif random_fen_pct >= 100:
                use_random_fen = 1
            else:
                use_random_fen = 0
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
                reward_repetition=reward_repetition,
                render_fps=render_fps,
                selfplay=selfplay,
                human_play=human_play,
                random_bot=random_bot,
                starting_fen=starting_fen,
                random_fen=use_random_fen,
                fen_curric_pct = fen_curric_pct,
                fen_curriculum=self.c_curriculum,
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
        info = []
        if self.tick % self.log_interval == 0:
            log_dict = binding.vec_log(self.c_envs)
            if 'n' in log_dict:
                self.cumulative_games += log_dict['n']
                log_dict['games_played'] = self.cumulative_games
            info = [log_dict]
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
