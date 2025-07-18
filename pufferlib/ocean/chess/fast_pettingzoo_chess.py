"""
Fast PettingZoo Chess implementation that directly uses high-performance Chess class.

This bypasses the slow AECEnv layer and uses flat Box observation spaces for maximum performance.
"""

import numpy as np
import gymnasium
from typing import Dict, List, Any, Optional, Tuple

import pufferlib
from .chess import Chess


class FastPettingZooChess(pufferlib.PufferEnv):
    """
    High-performance PettingZoo-compatible chess environment.
    
    This directly wraps the fast Chess class and presents a PettingZoo-compatible 
    interface without the overhead of AECEnv or Dict observation spaces.
    """
    
    def __init__(self, 
                 num_envs: int = 1,
                 seed: int = 0,
                 buf=None,
                 **kwargs):
        
        # Remove self_play from kwargs if present to avoid conflict
        kwargs.pop('self_play', None)
        
        # Ensure seed is an integer
        if seed is None:
            seed = 0
        seed = int(seed)
        
        # Create underlying high-performance Chess environment
        self.chess_env = Chess(num_envs=num_envs, seed=seed, self_play=True, buf=buf, **kwargs)
        
        # PufferLib requirements - must be set before calling super().__init__()
        self.num_agents = self.chess_env.num_agents
        self.single_observation_space = self.chess_env.single_observation_space
        self.single_action_space = self.chess_env.single_action_space
        
        # Initialize PufferEnv
        super().__init__(buf=buf)
        
        # PettingZoo API requirements
        self.possible_agents = [0, 1]  # WHITE=0, BLACK=1
        self.agents = []
        
        # Initialize with first reset
        self.has_reset = False
        self.current_obs = None
        self.current_rewards = None
        self.current_terminals = None
        self.current_truncations = None
        
        print(f"[FastPettingZoo Chess] Initialized with {num_envs} environments")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment.
        
        Returns:
            observations: Array of observations for all agents
            infos: List of info dicts
        """
        # Reset underlying Chess environment
        obs, info = self.chess_env.reset(seed=seed)
        
        # Copy observations to our buffers
        self.observations[:] = obs
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False
        
        # Both agents are active
        self.agents = [0, 1]
        self.has_reset = True
        
        return self.observations, info
    
    def step(self, actions):
        """
        Step the environment.
        
        Args:
            actions: Array of actions for all agents
            
        Returns:
            observations: Array of observations for all agents
            rewards: Array of rewards for all agents
            terminateds: Array of termination status for all agents
            truncateds: Array of truncation status for all agents
            infos: List of info dicts
        """
        if not self.has_reset:
            raise RuntimeError("Must call reset() before step()")
        
        # Actions are already in the correct format (array)
        self.actions[:] = actions
        
        # Step the underlying Chess environment
        obs, rewards, terminals, truncations, infos = self.chess_env.step(actions)
        
        # Copy to our buffers
        self.observations[:] = obs
        self.rewards[:] = rewards
        self.terminals[:] = terminals.astype(bool)
        self.truncations[:] = truncations.astype(bool)
        
        return self.observations, self.rewards, self.terminals, self.truncations, infos
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        return self.chess_env.render()
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'chess_env'):
            self.chess_env.close()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()