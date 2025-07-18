"""
PettingZoo-based Chess Environment for proper turn-based self-play.

This follows the OpenSpiel PettingZoo pattern where only the active player
receives observations and provides actions each step.
"""

import numpy as np
import gymnasium
from typing import Dict, List, Any, Optional, Tuple

try:
    from . import binding
except ImportError:
    import binding

class PettingZooChess:
    """
    PettingZoo-compatible chess environment for turn-based self-play.
    
    Key differences from native PufferLib multi-agent:
    - Only active player gets observations each step
    - Only active player provides actions each step
    - Uses same observation/action space for both agents
    - Proper turn-based semantics
    """
    
    def __init__(self, 
                 num_envs: int = 1,
                 seed: int = 0,
                 render_mode: str = 'rgb_array',
                 log_interval: int = 1,
                 reward_valid: float = 0.0,
                 reward_invalid_white: float = -1.0,
                 reward_invalid_black: float = -1.0,
                 reward_agent_captures_enemy_piece: float = 0.0,
                 reward_enemy_captures_agent_piece: float = 0.0,
                 reward_draw: float = 0.0,
                 reward_win_white: float = 1.0,
                 reward_win_black: float = 1.0,
                 reward_loss_white: float = -1.0,
                 reward_loss_black: float = -1.0,
                 reward_check_white: float = 0.0,
                 reward_check_black: float = 0.0,
                 reward_material_diff_white: float = 0.0,
                 reward_material_diff_black: float = 0.0,
                 debug_disable_mask: int = 0,
                 stockfish_enabled: int = 0,
                 stockfish_cmd: Optional[str] = None,
                 stockfish_elo: int = 800,
                 stockfish_search_ms: int = 10,
                 stockfish_hash_mb: int = 4,
                 full_game_logging_frequency: int = 5000000,
                 max_depth: int = 512,
                 **kwargs):
        
        self.num_envs = num_envs
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.seed_value = seed
        
        # PettingZoo API requirements
        self.possible_agents = [0, 1]  # WHITE=0, BLACK=1
        self.agents = []  # Will be set in reset()
        
        # Single observation and action space (used by both agents)
        self.num_board_obs = 8*8*21  # board planes only
        self.num_actions = 1968
        
        # Use flat Box observation space for performance (board + action_mask)
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(self.num_board_obs + self.num_actions,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(self.num_actions)
        
        # Initialize C++ environments
        # Use buffers for single environment with 2 agents
        self.num_obs = self.num_board_obs + self.num_actions  # board + action mask
        self.observations = np.zeros((2, self.num_obs), dtype=np.float32)
        self.actions = np.zeros(2, dtype=np.int32)
        self.rewards = np.zeros(2, dtype=np.float32)
        self.terminals = np.zeros(2, dtype=np.uint8)
        self.truncations = np.zeros(2, dtype=np.uint8)
        
        # Store separate observation copies for each agent
        self.current_obs = np.zeros((2, self.num_obs), dtype=np.float32)
        
        # Initialize C environments
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            reward_valid=reward_valid,
            reward_invalid_white=reward_invalid_white,
            reward_invalid_black=reward_invalid_black,
            reward_agent_captures_enemy_piece=reward_agent_captures_enemy_piece,
            reward_enemy_captures_agent_piece=reward_enemy_captures_agent_piece,
            reward_draw=reward_draw,
            reward_win_white=reward_win_white,
            reward_win_black=reward_win_black,
            reward_loss_white=reward_loss_white,
            reward_loss_black=reward_loss_black,
            reward_check_white=reward_check_white,
            reward_check_black=reward_check_black,
            reward_material_diff_white=reward_material_diff_white,
            reward_material_diff_black=reward_material_diff_black,
            debug_disable_mask=debug_disable_mask,
            stockfish_enabled=stockfish_enabled,
            stockfish_cmd=stockfish_cmd or "",
            stockfish_elo=stockfish_elo,
            stockfish_search_ms=stockfish_search_ms,
            stockfish_hash_mb=stockfish_hash_mb,
            full_game_logging_frequency=full_game_logging_frequency,
            max_depth=max_depth
        )
        
        # Enable dual-agent self-play mode
        binding.vec_set_dual_agent_self_play(self.c_envs)
        
        # Game state tracking
        self.current_player = 0  # WHITE starts
        self.game_over = False
        self.has_reset = False
        
        print(f"[PettingZoo Chess] Initialized with {num_envs} environments")
    
    def observation_space(self, agent: int) -> gymnasium.Space:
        """Return observation space for agent."""
        if agent not in self.possible_agents:
            raise ValueError(f"Agent {agent} not in possible agents {self.possible_agents}")
        return self.single_observation_space
    
    def action_space(self, agent: int) -> gymnasium.Space:
        """Return action space for agent."""
        if agent not in self.possible_agents:
            raise ValueError(f"Agent {agent} not in possible agents {self.possible_agents}")
        return self.single_action_space
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[int, np.ndarray], Dict[int, Any]]:
        """
        Reset the environment.
        
        Returns:
            observations: Dict with only the current player's observation
            infos: Dict with info for current player
        """
        if seed is not None:
            self.seed_value = seed
            np.random.seed(seed)
        
        # Reset C++ environments
        binding.vec_reset(self.c_envs, seed or self.seed_value)
        
        # WHITE always starts
        self.current_player = 0
        self.agents = [self.current_player]
        self.game_over = False
        self.has_reset = True
        
        # Store current observation state for both players (avoid copy)
        self.current_obs = self.observations
        
        # Get initial observation for WHITE (flat array)
        raw_obs = self.current_obs[self.current_player]
        
        observations = {self.current_player: raw_obs}
        infos = {self.current_player: {}}
        
        print(f"[PettingZoo Chess] Reset complete, WHITE (agent {self.current_player}) to move")
        return observations, infos
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict[int, Any]]:
        """
        Step the environment.
        
        Args:
            actions: Dict with current player's action
            
        Returns:
            observations: Dict with next player's observation (if game continues)
            rewards: Dict with rewards for both players
            terminateds: Dict with termination status
            truncateds: Dict with truncation status  
            infos: Dict with info
        """
        if not self.has_reset:
            raise RuntimeError("Must call reset() before step()")
        
        if self.game_over:
            raise RuntimeError("Game is over, must call reset()")
        
        # Validate that only current player provided an action
        if self.current_player not in actions:
            raise ValueError(f"Current player {self.current_player} must provide an action")
        
        if len(actions) != 1:
            raise ValueError(f"Expected 1 action for current player, got {len(actions)}")
        
        # Set action for current player
        action = actions[self.current_player]
        self.actions[self.current_player] = action
        
        print(f"[PettingZoo Chess] Player {self.current_player} ({'WHITE' if self.current_player == 0 else 'BLACK'}) plays action {action}")
        
        # Step C++ environment
        binding.vec_step(self.c_envs)
        
        # Get results
        rewards_dict = {0: self.rewards[0], 1: self.rewards[1]}
        terminateds_dict = {0: bool(self.terminals[0]), 1: bool(self.terminals[1])}
        truncateds_dict = {0: bool(self.truncations[0]), 1: bool(self.truncations[1])}
        infos_dict = {0: {}, 1: {}}
        
        # Check if game is over
        self.game_over = terminateds_dict[0] or terminateds_dict[1] or truncateds_dict[0] or truncateds_dict[1]
        
        if self.game_over:
            # Game over - no more active agents
            self.agents = []
            observations_dict = {}
            print(f"[PettingZoo Chess] Game over! WHITE reward: {rewards_dict[0]}, BLACK reward: {rewards_dict[1]}")
        else:
            # Switch to next player
            self.current_player = 1 - self.current_player
            self.agents = [self.current_player]
            
            # Store current observation state for both players (avoid copy)
            self.current_obs = self.observations
            
            # Get observation for next player (flat array)
            raw_obs = self.current_obs[self.current_player]
            
            observations_dict = {self.current_player: raw_obs}
            
            print(f"[PettingZoo Chess] Next player: {self.current_player} ({'WHITE' if self.current_player == 0 else 'BLACK'})")
        
        return observations_dict, rewards_dict, terminateds_dict, truncateds_dict, infos_dict
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if hasattr(binding, 'vec_render'):
            binding.vec_render(self.c_envs)
        return None
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'c_envs'):
            binding.vec_close(self.c_envs)
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()