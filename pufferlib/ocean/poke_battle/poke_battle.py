'''PokeBattle: Gen1 OU Pokemon battle environment for PufferLib.

Self-contained C-based Gen 1 OU singles battle simulator.
Actions 0-3: use move, Actions 4-9: switch to Pokemon at index.

Supports selfplay mode (matching xinpw8/PufferLib chess pattern):
  - Observations are concatenated [learner_obs | opponent_obs]
  - Actions buffer has 2 entries per env (learner + opponent)
  - pufferl.py selfplay code splits obs and interleaves actions
'''

import gymnasium
import numpy as np

from pufferlib.ocean.poke_battle import binding
import pufferlib


OBS_SIZE = 140


class PokeBattle(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, buf=None,
                 seed=0, selfplay=1, bot_mode=0, mcts_iterations=128,
                 mcts_depth=5, auto_reset=1):
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.selfplay = selfplay

        # Selfplay: obs is doubled (both players' views concatenated)
        # Non-selfplay: single player obs only
        factor = 2 if selfplay else 1
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(OBS_SIZE * factor,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(10)

        # In selfplay, each env appears as 1 agent to PufferLib
        # (internally handles 2 players)
        self.num_agents = num_envs

        super().__init__(buf)

        # Selfplay needs 2 actions per env (learner + opponent)
        if selfplay:
            self.actions = np.zeros(num_envs * 2, dtype=np.int32)

        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(
                self.observations[i:i+1],
                self.actions[i*factor:(i+1)*factor],
                self.rewards[i:i+1],
                self.terminals[i:i+1],
                self.truncations[i:i+1],
                seed + i,
                num_agents=factor,
                seed=seed + i,
                selfplay=selfplay,
                learner_side=i % 2,  # Alternate sides across envs
                bot_mode=bot_mode,
                mcts_iterations=mcts_iterations,
                mcts_depth=mcts_depth,
                auto_reset=auto_reset,
            )
            c_envs.append(c_env)

        self.c_env_handles = c_envs
        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=None):
        if seed is not None:
            base_seed = int(seed)
            for i, handle in enumerate(self.c_env_handles):
                # Honor caller-provided reset seed for this env backend.
                binding.env_put(handle, seed=base_seed + i, episode_count=0)
        binding.vec_reset(self.c_envs, int(seed or 0))
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def get_state(self, env_idx=0):
        '''Inspect raw simulator state for debugging/evaluation tooling.'''
        return binding.env_get(self.c_env_handles[env_idx])

    def get_states(self):
        return [binding.env_get(h) for h in self.c_env_handles]

    def render_get_action(self, env_idx=0):
        '''Render battle UI and block until human clicks a valid action.
        Returns action int (0-9), or -1 if window closed.'''
        return binding.env_render_get_action(self.c_env_handles[env_idx])

    def close(self):
        binding.vec_close(self.c_envs)


if __name__ == '__main__':
    import time
    N = 8
    CACHE = 1024

    # Test each bot mode
    bot_names = {0: 'random', 1: 'heuristic', 2: 'mcts'}
    for bot_mode in [0, 1, 2]:
        env = PokeBattle(num_envs=N, selfplay=0, bot_mode=bot_mode)
        env.reset()
        steps = 0
        actions = np.random.randint(0, 10, size=(CACHE, N))

        duration = 10 if bot_mode == 0 else 5
        start = time.time()
        i = 0
        while time.time() - start < duration:
            env.step(actions[i % CACHE])
            steps += env.num_agents
            i += 1

        elapsed = time.time() - start
        print(f'PokeBattle SPS (bot={bot_names[bot_mode]}): {int(steps / elapsed)}')
        env.close()

    # Test selfplay mode
    env = PokeBattle(num_envs=N, selfplay=1)
    env.reset()
    steps = 0
    actions_sp = np.random.randint(0, 10, size=(CACHE, N * 2))

    start = time.time()
    i = 0
    while time.time() - start < 10:
        env.step(actions_sp[i % CACHE])
        steps += env.num_agents
        i += 1

    elapsed = time.time() - start
    print(f'PokeBattle SPS (selfplay): {int(steps / elapsed)}')
    env.close()
