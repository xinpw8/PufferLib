'''High-perf many-agent snake. Inspired by snake env from https://github.com/dnbt777'''

import numpy as np
import gymnasium
import sys
import traceback

import pufferlib
from pufferlib.exceptions import APIUsageError
import pufferlib.ocean.snake.binding as binding

class Snake(pufferlib.PufferEnv):
    def __init__(self, num_envs=16, width=640, height=360,
            num_snakes=256, num_food=4096,
            vision=3, leave_corpse_on_death=True,
            reward_food=1.0, reward_corpse=0.5, reward_death=-1.0,
            survival_reward=0.01, report_interval=128, max_snake_length=1024,
            render_mode='human', buf=None, seed=0):
        
        if num_envs is not None:
            # Create lists of parameters for each environment
            # Make sure these are plain Python lists, not numpy arrays
            # Convert single values to lists if necessary
            num_snakes_list = [num_snakes] * num_envs if isinstance(num_snakes, int) else list(num_snakes)
            width_list = [width] * num_envs if isinstance(width, int) else list(width)
            height_list = [height] * num_envs if isinstance(height, int) else list(height)
            num_food_list = [num_food] * num_envs if isinstance(num_food, int) else list(num_food)
            leave_corpse_list = [leave_corpse_on_death] * num_envs if isinstance(leave_corpse_on_death, bool) else list(leave_corpse_on_death)
            
            print(f"DEBUG: num_envs={num_envs}, width_list[0]={width_list[0]}, height_list[0]={height_list[0]}", file=sys.stderr)
        else:
            # Single environment case
            num_snakes_list = [num_snakes]
            width_list = [width]
            height_list = [height]  
            num_food_list = [num_food]
            leave_corpse_list = [leave_corpse_on_death]

        if not (len(num_snakes_list) == len(width_list) == len(height_list) == len(num_food_list)):
            raise APIUsageError('num_snakes, width, height, num_food must be lists of equal length')

        # Validate width and height based on vision
        for w, h in zip(width_list, height_list):
            min_dimension = 2 * vision + 2
            if w < min_dimension or h < min_dimension:
                raise APIUsageError(f'width and height must be at least 2*vision+2 ({min_dimension}). Got width={w}, height={h}')

        max_area = max([w*h for h, w in zip(height_list, width_list)])
        self.max_snake_length = min(max_snake_length, max_area)
        self.report_interval = report_interval

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(2*vision+1, 2*vision+1), dtype=np.int8)
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.num_agents = sum(num_snakes_list)
        self.render_mode = render_mode
        self.tick = 0

        self.cell_size = int(np.ceil(1280 / max(max(width_list), max(height_list))))

        super().__init__(buf)
        
        # Wrap initialization in try-except for better error handling
        try:
            print(f"Creating C environment with num_envs={num_envs}, num_snakes={num_snakes_list}", file=sys.stderr)
            self.c_envs = binding.vec_init(
                self.observations,      # Shared buffer for observations
                self.actions,           # Shared buffer for actions
                self.rewards,           # Shared buffer for rewards
                self.terminals,         # Shared buffer for terminals
                self.truncations,       # Shared buffer for truncations
                num_envs,               # Number of parallel C env instances to create
                seed,                   # Initial seed for RNG
                width=width_list,
                height=height_list,
                num_snakes=num_snakes_list,
                num_food=num_food_list,
                vision=vision,
                max_snake_length=self.max_snake_length,
                leave_corpse_on_death=leave_corpse_list,
                reward_food=reward_food,
                reward_corpse=reward_corpse,
                reward_death=reward_death,
                survival_reward=survival_reward
            )
            print("Successfully created C environment", file=sys.stderr)
        except Exception as e:
            print(f"Error initializing C environment: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise
 
    def reset(self, seed=None):
        self.tick = 0
        # Use a default seed (0) if none is provided
        if seed is None:
            seed = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        # Remap actions for testing - shift all actions by 1
        remapped_actions = (actions + 1) % 4
        self.actions[:] = remapped_actions
        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    env = Snake()
    env.reset()
    tick = 0

    total_snakes = sum(env.num_snakes)
    actions = np.random.randint(0, 4, (atn_cache, total_snakes))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f'SPS: %f', total_snakes * tick / (time.time() - start))

if __name__ == '__main__':
    test_performance()
