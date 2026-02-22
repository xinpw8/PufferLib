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

RBY_OU_RULESET = '[Gen 1] OU'
RBY_OU_STANDARD_RULES = (
    'Obtainable',
    'Desync Clause Mod',
    'Sleep Clause Mod',
    'Freeze Clause Mod',
    'Species Clause',
    'Nickname Clause',
    'OHKO Clause',
    'Evasion Moves Clause',
    'Endless Battle Clause',
    'HP Percentage Mod',
    'Cancel Mod',
)

# Species ID constants (must match SpeciesID enum in poke_battle.h)
SPECIES_NONE      = 0
SPECIES_TAUROS    = 1
SPECIES_CHANSEY   = 2
SPECIES_SNORLAX   = 3
SPECIES_ALAKAZAM  = 4
SPECIES_EXEGGUTOR = 5
SPECIES_STARMIE   = 6
SPECIES_GENGAR    = 7
SPECIES_JYNX      = 8
SPECIES_ZAPDOS    = 9
SPECIES_RHYDON    = 10
SPECIES_CLOYSTER  = 11
SPECIES_GOLEM     = 12
SPECIES_LAPRAS    = 13
SPECIES_SLOWBRO   = 14
SPECIES_JOLTEON   = 15
SPECIES_PERSIAN   = 16
SPECIES_HYPNO     = 17
SPECIES_ARTICUNO  = 18
SPECIES_DRAGONITE = 19
SPECIES_MACHAMP   = 20

SPECIES_NAMES = {
    0: 'None', 1: 'Tauros', 2: 'Chansey', 3: 'Snorlax', 4: 'Alakazam',
    5: 'Exeggutor', 6: 'Starmie', 7: 'Gengar', 8: 'Jynx', 9: 'Zapdos',
    10: 'Rhydon', 11: 'Cloyster', 12: 'Golem', 13: 'Lapras', 14: 'Slowbro',
    15: 'Jolteon', 16: 'Persian', 17: 'Hypno', 18: 'Articuno',
    19: 'Dragonite', 20: 'Machamp',
}

# Preset competitive team compositions
TEAMS = {
    'ultimate': [SPECIES_SLOWBRO, SPECIES_JYNX, SPECIES_EXEGGUTOR,
                 SPECIES_LAPRAS, SPECIES_ZAPDOS, SPECIES_TAUROS],
    'big6':     [SPECIES_TAUROS, SPECIES_CHANSEY, SPECIES_SNORLAX,
                 SPECIES_ALAKAZAM, SPECIES_EXEGGUTOR, SPECIES_STARMIE],
    'sleep_offense': [SPECIES_JYNX, SPECIES_EXEGGUTOR, SPECIES_GENGAR,
                      SPECIES_TAUROS, SPECIES_STARMIE, SPECIES_SNORLAX],
    'anti_slowbro':  [SPECIES_ZAPDOS, SPECIES_JOLTEON, SPECIES_EXEGGUTOR,
                      SPECIES_ALAKAZAM, SPECIES_STARMIE, SPECIES_GENGAR],
    'balanced': [SPECIES_TAUROS, SPECIES_STARMIE, SPECIES_ZAPDOS,
                 SPECIES_SNORLAX, SPECIES_ALAKAZAM, SPECIES_GENGAR],
    'ice_core': [SPECIES_JYNX, SPECIES_LAPRAS, SPECIES_ARTICUNO,
                 SPECIES_STARMIE, SPECIES_ALAKAZAM, SPECIES_TAUROS],
}


def team_str(team):
    '''Pretty-print a team list of species IDs.'''
    return ' / '.join(SPECIES_NAMES.get(s, f'?{s}') for s in team)


class PokeBattle(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, buf=None,
                 seed=0, selfplay=1, bot_mode=0, mcts_iterations=128,
                 mcts_depth=5, auto_reset=1, p1_team=None, p2_team=None,
                 force_accuracy=-1, force_secondary=-1, enforce_endless_clause=1):
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

        # Resolve team names to lists
        if isinstance(p1_team, str):
            p1_team = TEAMS[p1_team]
        if isinstance(p2_team, str):
            p2_team = TEAMS[p2_team]

        c_envs = []
        for i in range(num_envs):
            init_kwargs = dict(
                num_agents=factor,
                seed=seed + i,
                selfplay=selfplay,
                learner_side=i % 2,  # Alternate sides across envs
                bot_mode=bot_mode,
                mcts_iterations=mcts_iterations,
                mcts_depth=mcts_depth,
                auto_reset=auto_reset,
                force_accuracy=force_accuracy,
                force_secondary=force_secondary,
                enforce_endless_clause=enforce_endless_clause,
            )
            if p1_team is not None:
                init_kwargs['p1_team'] = list(p1_team)
            if p2_team is not None:
                init_kwargs['p2_team'] = list(p2_team)

            c_env = binding.env_init(
                self.observations[i:i+1],
                self.actions[i*factor:(i+1)*factor],
                self.rewards[i:i+1],
                self.terminals[i:i+1],
                self.truncations[i:i+1],
                seed + i,
                **init_kwargs,
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

    def put_state(self, env_idx=0, **kwargs):
        '''Write selected simulator fields exposed by env_put for tests/debugging.'''
        return binding.env_put(self.c_env_handles[env_idx], **kwargs)

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
