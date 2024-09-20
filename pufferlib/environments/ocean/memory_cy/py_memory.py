import numpy as np
import gymnasium
from .cy_memory_cy import CMemoryCy

class MemoryCyEnv(gymnasium.Env):
    def __init__(self, mem_length=1, mem_delay=0, render_mode='ansi'):
        super().__init__()

        self.mem_length = mem_length
        self.mem_delay = mem_delay
        self.solution = np.zeros((1, 2 * mem_length + mem_delay), dtype=np.float32)
        self.submission = np.zeros((1, 2 * mem_length + mem_delay), dtype=np.float32) - 1
        self.rewards = np.zeros((1, 1), dtype=np.float32)
        self.actions = np.zeros((1, 1), dtype=np.int32)

        self.c_env = CMemoryCy(mem_length, mem_delay, self.solution, self.submission, self.rewards, self.actions)

        self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = gymnasium.spaces.Discrete(2)
        
        # Set render mode (e.g., ansi, human, etc.)
        self.render_mode = render_mode

    def reset(self, seed=None):
        seed = 0 if seed is None else seed
        self.c_env.reset(seed)
        return self.solution[0, 0], {}

    def step(self, action):
        self.actions[0, 0] = action
        self.c_env.step()

        current_tick = self.c_env.get_tick() - 1

        done = self.c_env.is_done()
        info = {'score': self.c_env.check_solution()} if done else {}

        return self.solution[0, current_tick], self.rewards[0, 0], done, False, info

    def render(self):
        if self.render_mode == 'ansi':
            def _render(val):
                if val == 1:
                    c = 94
                elif val == 0:
                    c = 91
                else:
                    c = 90
                return f'\033[{c}m██\033[0m'

            chars = []
            
            # Render solution
            for val in self.solution[0]:
                c = _render(val)
                chars.append(c)
            chars.append(' Solution\n')

            # Render submission
            for val in self.submission[0]:
                c = _render(val)
                chars.append(c)
            chars.append(' Prediction\n')

            return ''.join(chars)
        else:
            raise NotImplementedError("Only 'ansi' render mode is implemented")

    def close(self):
        pass  # This can be expanded to clean up resources if necessary


def test_performance(mem_length=1, mem_delay=0, timeout=10, atn_cache=1024):
    import time
    env = MemoryCyEnv(mem_length=mem_length, mem_delay=mem_delay)

    env.reset()

    tick = 0
    actions = np.random.randint(0, 2, (atn_cache, 1))

    start = time.time()

    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn[0])
        tick += 1

    elapsed_time = time.time() - start
    sps = tick / elapsed_time
    print(f"SPS: {sps:.2f}")

if __name__ == '__main__':
    test_performance()
