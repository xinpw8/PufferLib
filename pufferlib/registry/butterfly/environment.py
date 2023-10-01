from pdb import set_trace as T
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

import pufferlib.emulation
import pufferlib.registry


def env_creator(name):
    pufferlib.registry.try_import('pettingzoo.butterfly', 'butterfly')
    if name == 'cooperative_pong_v5':
        from pettingzoo.butterfly import cooperative_pong_v5 as pong
        return pong.raw_env
    elif name == 'knights_archers_zombies_v10':
        from pettingzoo.butterfly import knights_archers_zombies_v10 as kaz
        return kaz.raw_env
    else:
        raise ValueError(f'Unknown environment: {name}')
     
def make_env(name='cooperative_pong_v5'):
    env = env_creator(name)()
    env = aec_to_parallel_wrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)