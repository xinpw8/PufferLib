import numpy as np
import gymnasium
from gymnasium import spaces
from .cy_password_cy import CPasswordCy

class PasswordCyEnv(gymnasium.Env):
    def __init__(self, password_length=5, hard_fixed_seed=42):
        self.env = CPasswordCy(password_length=password_length, hard_fixed_seed=hard_fixed_seed)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(password_length,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.render_mode = 'ansi'

    def reset(self, seed=None):
        observation = self.env.reset()
        return observation, {}

    def step(self, action):
        observation, reward, done = self.env.step(action)
        return observation, reward, done, False, {}

    def render(self):
        def _render(val):
            if val == 1:
                return f'\033[94m██\033[0m'
            elif val == 0:
                return f'\033[91m██\033[0m'
            return f'\033[90m██\033[0m'

        chars = []
        for val in self.env.get_solution():
            chars.append(_render(val))
        chars.append(' Solution\n')

        for val in self.env.get_observation():
            chars.append(_render(val))
        chars.append(' Prediction\n')

        return ''.join(chars)

def test_performance(password_length=5, timeout=10, action_cache_size=1024):
    import time
    # Initialize the environment
    env = PasswordCyEnv(password_length=password_length)
    
    # Reset the environment
    env.reset()

    tick = 0
    # Generate random actions to simulate agent behavior
    actions = np.random.randint(0, 2, size=(action_cache_size,))

    # Start the timer
    start_time = time.time()

    # Run until the timeout is reached
    while time.time() - start_time < timeout:
        # Simulate environment steps
        action = actions[tick % action_cache_size]  # Choose an action from the cache
        observation, reward, done, _, _ = env.step(action)
        
        # If the environment is done, reset it
        if done:
            env.reset()

        tick += 1

    # Calculate elapsed time and steps per second (SPS)
    elapsed_time = time.time() - start_time
    steps_per_second = tick / elapsed_time

    # Output the performance metric
    print(f"Steps per second (SPS): {steps_per_second:.2f}")

if __name__ == "__main__":
    # Run the test for the Password environment
    test_performance(password_length=5, timeout=10)
