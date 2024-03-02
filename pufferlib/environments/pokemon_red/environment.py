# from pdb import set_trace as T
# import gymnasium
# import functools
# from pokegym import Environment
# import pufferlib.emulation

# def env_creator(name='pokemon_red'):
#     return functools.partial(make, name)

# def make(name, headless: bool = True, state_path=None):
#     '''Pokemon Red'''
#     env = Environment()
#     return pufferlib.emulation.GymnasiumPufferEnv(env=env,
#         postprocessor_cls=pufferlib.emulation.BasicPostprocessor)

import functools
import pufferlib.emulation
from pokegym.pokegym import Environment
from stream_wrapper import StreamWrapper

def env_creator(name="pokemon_red"):
    return functools.partial(make, name)

def make(name, **kwargs,):
    """Pokemon Red"""
    env = Environment(kwargs)
    env = StreamWrapper(env, stream_metadata={"user": "       PUFFERBOX|BET|PUFFERBOX       \nPUFFERBOX|BET|PUFFERBOX \n====BOEY====\nPUFFERBOX|BET|PUFFERBOX "})
    # Looks like the following will optionally create the object for you
    # Or use the one you pass it. I'll just construct it here.
    return pufferlib.emulation.GymnasiumPufferEnv(
        env=env, postprocessor_cls=pufferlib.emulation.BasicPostprocessor
    )