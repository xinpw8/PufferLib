import pufferlib.emulation
import pufferlib.postprocess

from .snake.snake import Snake
from .squared.squared import Squared
from .pong.pong import Pong
from .breakout.breakout import Breakout
from .connect4.connect4 import Connect4
from .tripletriad.tripletriad import TripleTriad
from .moba.moba import Moba
from .go.go import Go
from .rware.rware import Rware
#from .rocket_lander import rocket_lander

def make_foraging(width=1080, height=720, num_agents=4096, horizon=512,
        discretize=True, food_reward=0.1, render_mode='rgb_array'):
    from .grid import grid
    init_fn = grid.init_foraging
    reward_fn = grid.reward_foraging
    return grid.PufferGrid(width, height, num_agents,
        horizon, discretize=discretize, food_reward=food_reward,
        init_fn=init_fn, reward_fn=reward_fn,
        render_mode=render_mode)

def make_predator_prey(width=1080, height=720, num_agents=4096, horizon=512,
        discretize=True, food_reward=0.1, render_mode='rgb_array'):
    from .grid import grid
    init_fn = grid.init_predator_prey
    reward_fn = grid.reward_predator_prey
    return grid.PufferGrid(width, height, num_agents,
        horizon, discretize=discretize, food_reward=food_reward,
        init_fn=init_fn, reward_fn=reward_fn,
        render_mode=render_mode)

def make_group(width=1080, height=720, num_agents=4096, horizon=512,
        discretize=True, food_reward=0.1, render_mode='rgb_array'):
    from .grid import grid
    init_fn = grid.init_group
    reward_fn = grid.reward_group
    return grid.PufferGrid(width, height, num_agents,
        horizon, discretize=discretize, food_reward=food_reward,
        init_fn=init_fn, reward_fn=reward_fn,
        render_mode=render_mode)

def make_puffer(width=1080, height=720, num_agents=4096, horizon=512,
        discretize=True, food_reward=0.1, render_mode='rgb_array'):
    from .grid import grid
    init_fn = grid.init_puffer
    reward_fn = grid.reward_puffer
    return grid.PufferGrid(width, height, num_agents,
        horizon, discretize=discretize, food_reward=food_reward,
        init_fn=init_fn, reward_fn=reward_fn,
        render_mode=render_mode)

def make_puffergrid(render_mode='rgb_array', vision_range=3):
    assert False, 'This env is unfinished. Join our Discord and help us finish it!'
    from .grid import grid
    return grid.PufferGrid(render_mode, vision_range)

def make_continuous(discretize=False, **kwargs):
    from . import sanity
    env = sanity.Continuous(discretize=discretize)
    if not discretize:
        env = pufferlib.postprocess.ClipAction(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_squared(distance_to_target=3, num_targets=1, **kwargs):
    from . import sanity
    env = sanity.Squared(distance_to_target=distance_to_target, num_targets=num_targets, **kwargs)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, **kwargs)

def make_bandit(num_actions=10, reward_scale=1, reward_noise=1):
    from . import sanity
    env = sanity.Bandit(num_actions=num_actions, reward_scale=reward_scale,
        reward_noise=reward_noise)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_memory(mem_length=2, mem_delay=2, **kwargs):
    from . import sanity
    env = sanity.Memory(mem_length=mem_length, mem_delay=mem_delay)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_password(password_length=5, **kwargs):
    from . import sanity
    env = sanity.Password(password_length=password_length)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_performance(delay_mean=0, delay_std=0, bandwidth=1, **kwargs):
    from . import sanity
    env = sanity.Performance(delay_mean=delay_mean, delay_std=delay_std, bandwidth=bandwidth)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_performance_empiric(count_n=0, count_std=0, bandwidth=1, **kwargs):
    from . import sanity
    env = sanity.PerformanceEmpiric(count_n=count_n, count_std=count_std, bandwidth=bandwidth)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_stochastic(p=0.7, horizon=100, **kwargs):
    from . import sanity
    env = sanity.Stochastic(p=p, horizon=100)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)

def make_spaces(**kwargs):
    from . import sanity
    env = sanity.Spaces()
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, **kwargs)

def make_multiagent(**kwargs):
    from . import sanity
    env = sanity.Multiagent()
    env = pufferlib.postprocess.MultiagentEpisodeStats(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)

MAKE_FNS = {
    'breakout': Breakout,
    'moba': Moba,
    'pong': Pong,
    'snake': Snake,
    'squared': Squared,
    'connect4': Connect4,
    'tripletriad': TripleTriad,
    'go': Go,
    'rware': Rware,

    #'rocket_lander': rocket_lander.RocketLander,
    'foraging': make_foraging,
    'predator_prey': make_predator_prey,
    'group': make_group,
    'puffer': make_puffer,
    'puffer_grid': make_puffergrid,
    # 'tactical': make_tactical,
    'continuous': make_continuous,
    'bandit': make_bandit,
    'memory': make_memory,
    'password': make_password,
    'stochastic': make_stochastic,
    'multiagent': make_multiagent,
    'spaces': make_spaces,
    'performance': make_performance,
    'performance_empiric': make_performance_empiric,
}

# Alias puffer_ to all names
MAKE_FNS = {**MAKE_FNS, **{'puffer_' + k: v for k, v in MAKE_FNS.items()}}

def env_creator(name='squared'):
    if name in MAKE_FNS:
        return MAKE_FNS[name]
    else:
        raise ValueError(f'Invalid environment name: {name}')


