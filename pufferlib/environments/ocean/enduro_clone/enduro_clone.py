import numpy as np
import gymnasium
import pufferlib
from pufferlib.environments.ocean.enduro_clone.cy_enduro_clone import CyEnduro

class MyEnduro(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
                 width=160.0, height=210.0,
                 car_width=16.0, car_height=11.0,
                 min_speed=0.0, max_speed=7.5,
                 initial_cars_to_pass=200,
                 max_enemies=10,
                 report_interval=1, buf=None):
        """
        Initialize the Enduro environment.
        """
        # Set basic attributes
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.tick = 0
        self.max_enemies = max_enemies
        

        # Environment configuration
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval

        # Calculate observation size based on max_enemies
        obs_size = 6 + 2 * max_enemies + 3  # Total features from compute_observations
        self.num_obs = obs_size

        # Define observation and action spaces
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(self.num_obs,), dtype=np.float32)
        # noop, fire (accelerate), down (decelerate), left, right,
        # fire-left, fire-right, down-left, down-right
        self.single_action_space = gymnasium.spaces.Discrete(9)

        # Initialize parent class
        super().__init__(buf=buf)
        
        # Allocate arrays for observations, actions, rewards, terminals
        self.observations = np.zeros((num_envs, self.num_obs), dtype=np.float32)
        self.actions = np.zeros(num_envs, dtype=np.int32)
        self.rewards = np.zeros(num_envs, dtype=np.float32)
        self.terminals = np.zeros(num_envs, dtype=np.uint8)
        self.truncations = np.zeros(num_envs, dtype=np.uint8)

        
        # Create environment instance
        self.c_envs = CyEnduro(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            width,
            height,
            car_width,
            car_height,
            min_speed,
            max_speed,
            initial_cars_to_pass
        )
    
    def reset(self, seed=None):
        """Reset the environment."""
        self.tick = 0
        self.c_envs.reset()
        return self.observations, []
    
    def step(self, actions):
        """Step the environment forward."""
        self.actions[:] = actions
        self.c_envs.step()
        
        # print(f'from python: obs:{self.observations}, actions:{self.actions}, rewards:{self.rewards}, terminals:{self.terminals}, truncations:{self.truncations}')
        # print(f'infos: {self.c_envs.log()}')
        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)
                
        
        self.tick += 1
        
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info
        )
    
    def close(self):
        """Clean up resources."""
        self.c_envs.close()