import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pufferlib
import pufferlib.emulation
from pufferlib.emulation import GymnasiumPufferEnv

class HigherOrLower(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, n=100, max_steps=None, render_mode=None):
        super(HigherOrLower, self).__init__()
        # Define action and observation spaces
        self.n = n
        self.action_space = spaces.Discrete(n + 1)  # Guesses from 0 to n
        
        # Use a simple Box space instead of Dict for better compatibility
        # [last_guess, last_feedback]
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([n, 2], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        
        self.max_steps = max_steps if max_steps is not None else n
        self.render_mode = render_mode
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset episodic states
        self.steps = 0
        self.last_guess = 0
        self.last_feedback = 0  # 0: initial, 1: higher, 2: lower
        self.secret = self.np_random.integers(0, self.n + 1)
        
        # Return initial observation as a flat array
        observation = np.array([self.last_guess, self.last_feedback], dtype=np.float32)
        info = {}
        return observation, info
    
    def step(self, action):
        self.steps += 1
        self.last_guess = action
        
        # Check for truncation due to max steps
        truncated = self.max_steps is not None and self.steps >= self.max_steps
        
        if action == self.secret:
            reward = 1.0
            terminated = True
            self.last_feedback = 0  # Reset for clarity
            info = {"success": True, "guesses": self.steps}
        elif action < self.secret:
            reward = 0.0
            terminated = False
            self.last_feedback = 1  # higher
            info = {"success": False}
        else:  # action > self.secret
            reward = 0.0
            terminated = False
            self.last_feedback = 2  # lower
            info = {"success": False}
            
        # Return observation as a flat array
        observation = np.array([self.last_guess, self.last_feedback], dtype=np.float32)
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode != 'human':
            return

        feedback_map = {0: "initial", 1: "higher", 2: "lower"}
        print(f"Last guess: {self.last_guess}, Feedback: {feedback_map[self.last_feedback]}, Secret number: {self.secret}")

def make_env(n=100, max_steps=None, render_mode=None, **kwargs):
    """Create a HigherOrLower environment"""
    env = HigherOrLower(n=n, max_steps=max_steps, render_mode=render_mode)
    return env

def env_creator(env_name=None):
    """
    Returns a function that creates a PufferEnv-wrapped environment.
    This wrapper is required by pufferlib's structure.
    """
    def _make_env(buf=None, **kwargs):
        # Create the environment, ignoring the buf parameter that PufferLib passes
        env = HigherOrLower(**kwargs)
        # Wrap with the basic GymnasiumPufferEnv wrapper
        wrapped_env = GymnasiumPufferEnv(env)
        return wrapped_env
    
    return _make_env