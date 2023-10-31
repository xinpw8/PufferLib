from pdb import set_trace as T
import pufferlib.emulation

from pufferlib.environments import PokemonRed as env_creator


def make_env(
        headless: bool = True,
        save_video: bool = False
    ):
    '''Pokemon Red'''
    env = env_creator(headless=headless, save_video=save_video)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env,
        postprocessor_cls=pufferlib.emulation.BasicPostprocessor)