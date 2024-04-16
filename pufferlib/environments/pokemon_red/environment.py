import sys
from pathlib import Path
import functools
from pokegym.wrappers.async_io import AsyncWrapper

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

# def env_creator(name="pokemon_red", async_config=None):
#     # Ensure async_config is provided
#     if async_config is None:
#         raise ValueError("async_config must be provided")
#     return functools.partial(make, name, async_config=async_config)

# def make(name, async_config, **kwargs):
#     """Create an environment, wrap it with StreamWrapper and AsyncWrapper"""
#     env = Environment(**kwargs)  # Assume Environment takes keyword arguments for configuration
#     env = StreamWrapper(env, stream_metadata={"user": "checkpoings |BET|\n"})

#     # Ensure async_config is used properly
#     if 'send_queues' not in async_config or 'recv_queues' not in async_config:
#         raise ValueError("async_config must contain 'send_queues' and 'recv_queues'")

#     env = AsyncWrapper(env, async_config["send_queues"], async_config["recv_queues"])
#     return pufferlib.emulation.GymnasiumPufferEnv(
#         env=env, postprocessor_cls=pufferlib.emulation.BasicPostprocessor
#     )


def make(name, **kwargs,):
    """Pokemon Red"""
    env = Environment(kwargs)
    env = StreamWrapper(env, stream_metadata={"user": "checkpoings |BET|\n"})
    return pufferlib.emulation.GymnasiumPufferEnv(
        env=env, postprocessor_cls=pufferlib.emulation.BasicPostprocessor
    )