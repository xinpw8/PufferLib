import sys
from pathlib import Path
import functools

current_file_path = Path(__file__).resolve()
top_dir = None
for parent in current_file_path.parents:
    if (parent / 'pokegym').exists():
        top_dir = parent
        break
if top_dir is None:
    raise FileNotFoundError("Top directory with marker 'pokegym' not found.")
if str(top_dir) not in sys.path:
    sys.path.insert(0, str(top_dir))
    
import pufferlib.emulation
from pokegym import Environment
from stream_agent_wrapper import StreamWrapper

def env_creator(name="pokemon_red"):
    return functools.partial(make, name)

def make(name, **kwargs,):
    """Pokemon Red"""
    env = Environment(kwargs)
    env = StreamWrapper(env, stream_metadata={"user": "hileanke testingonly |BET|\n"})
    return pufferlib.emulation.GymnasiumPufferEnv(
        env=env, postprocessor_cls=pufferlib.emulation.BasicPostprocessor
    )