"""
Double-buffered chess wrapper that implements the precise scheme to keep
white and black rollouts separate for proper self-play advantage estimation.

Key insight: Each episode must contain ONLY one player's data. The training
algorithm sees alternating episodes - one for white moves, one for black moves.
This prevents contamination between white and black advantage computations.
"""

import numpy as np
import gymnasium
import pufferlib
from .chess import Chess

class DoubleBufferedChess(pufferlib.PufferEnv):
    """
    Double-buffered chess environment that maintains separate episodes for white and black.
    
    Implementation:
    1. Each step() call processes ONE move and returns data for the active player
    2. Episodes alternate: WHITE episode -> BLACK episode -> WHITE episode...
    3. Each episode contains 8-16 moves from one color only
    4. No mixing of white/black data within episodes
    5. Clean advantage estimation boundaries
    
    The key trick: We modify the environment to present a single-agent view to the
    training algorithm, where each agent represents "the neural network playing as
    one specific color in this episode."
    """
    
    def __init__(self, num_envs=1, moves_per_episode=8, buf=None, seed=None, puzzle_tries_per_env=10, **kwargs):
        """
        Args:
            num_envs: Number of parallel chess games
            moves_per_episode: Number of moves per color per episode (8-16 recommended)
            buf: Buffer for shared memory (passed to base environment)
            seed: Random seed
            **kwargs: Additional arguments passed to base Chess environment
        """
        # Initialize base chess environment
        if seed is None:
            seed = 0
        
        # Extract and override conflicting parameters
        chess_kwargs = kwargs.copy()
        # Respect config settings instead of forcing
        chess_kwargs['self_play'] = kwargs.get('self_play', False)
        chess_kwargs['episode_per_color'] = False  # Base env shouldn't handle this - wrapper does
        
        # Remove DoubleBufferedChess-specific parameters that Chess doesn't understand
        chess_kwargs.pop('frozen_policy_update_frequency', None)
        # Pass through puzzle parameters
        chess_kwargs['puzzle_tries_per_env'] = puzzle_tries_per_env
        
        self.base_env = Chess(
            num_envs=num_envs, 
            buf=buf,
            seed=seed,
            **chess_kwargs
        )
        
        self.num_envs = num_envs
        self.num_agents = num_envs  # Required by vector backend
        self.agents_per_batch = num_envs  # PuffeRL expects this for LSTM
        self.moves_per_episode = moves_per_episode
        
        # Copy observation and action spaces from base environment
        self.single_observation_space = self.base_env.single_observation_space
        self.single_action_space = self.base_env.single_action_space
        
        # Initialize PufferEnv parent class
        super().__init__(buf=buf)
        
        # Expose episode_per_color attribute for training loop detection
        self.episode_per_color = kwargs.get('episode_per_color', False)
        
        # Episode tracking
        self.current_episode_color = 0  # 0: WHITE episode, 1: BLACK episode
        self.episode_move_count = 0     # Moves in current episode
        self.game_move_count = 0        # Total moves in current game
        self.games_completed = 0
        
        # Buffer for episode data
        self.episode_observations = []
        self.episode_rewards = []
        
        # Frozen policy management for self-play
        self.frozen_policy = None
        self.frozen_policy_state = None
        self.policy_update_counter = 0
        self.freeze_policy_every = 100  # Update frozen policy every N episodes
        self._pending_policy_update = None
    
    def update_frozen_policy(self, current_policy):
        """Update the frozen policy used for opponent moves in episode-per-color mode."""
        import copy
        import torch
        
        self.frozen_policy = copy.deepcopy(current_policy)
        self.frozen_policy.eval()  # Set to eval mode
        
        # Reset LSTM state for frozen policy
        self.frozen_policy_state = None
        print(f"[DoubleBufferedChess] Frozen policy updated (episode {self.policy_update_counter})")
    
    def schedule_policy_update(self, current_policy):
        """Schedule a policy update for the next episode boundary.""" 
        self._pending_policy_update = current_policy
        
    def get_opponent_action(self, observation):
        """Get an action from the frozen policy for opponent moves."""
        if self.frozen_policy is None:
            # Fallback to random legal action
            return self._get_heuristic_action(observation)
            
        import torch
        
        # Convert observation to tensor
        obs_tensor = torch.from_numpy(observation).float()
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        # Extract sparse action mask and convert to dense format for action masking
        device = obs_tensor.device
        num_legal_moves = int(observation[1472])
        action_ids = observation[1473:1473+64].astype(int)  # Get action IDs
        
        # Check observation format and extract action mask
        if observation.shape[0] != 1537:
            print(f"[DEBUG] Unexpected observation shape: {observation.shape}, expected (1537,)")
            # Try to handle old format or error gracefully
            if observation.shape[0] == 3440:
                print("[DEBUG] Using old dense format")
                legal_mask_np = observation[-1968:]
                legal_mask = torch.from_numpy(legal_mask_np).float().to(device)
                if legal_mask.dim() == 1:
                    legal_mask = legal_mask.unsqueeze(0)
            else:
                print("[ERROR] Unknown observation format! This should never happen!")
                raise ValueError(f"Unknown observation format with shape {observation.shape}")
        else:
            # Create dense mask from sparse representation
            legal_mask_np = np.zeros(1968, dtype=np.float32)
            if num_legal_moves > 0:
                valid_action_ids = action_ids[:num_legal_moves]
                # Clamp to valid range to prevent out-of-bounds
                valid_action_ids = np.clip(valid_action_ids, 0, 1967)
                legal_mask_np[valid_action_ids] = 1.0
            else:
                return self._get_heuristic_action(observation)
            
            legal_mask = torch.from_numpy(legal_mask_np).float().to(device)
            if legal_mask.dim() == 1:
                legal_mask = legal_mask.unsqueeze(0)  # Add batch dimension
            
        # Get action from frozen policy with action masking
        with torch.no_grad():
            if hasattr(self.frozen_policy, 'forward_eval'):
                # Handle LSTM policies with forward_eval method
                if self.frozen_policy_state is None:
                    # Initialize LSTM state as dictionary with zero tensors
                    batch_size = obs_tensor.shape[0]
                    hidden_size = self.frozen_policy.hidden_size
                    self.frozen_policy_state = {
                        'lstm_h': torch.zeros(batch_size, hidden_size, device=device),
                        'lstm_c': torch.zeros(batch_size, hidden_size, device=device)
                    }
                
                action_logits, values = self.frozen_policy.forward_eval(obs_tensor, self.frozen_policy_state)
                # Apply action masking - set illegal actions to -inf
                masked_logits = action_logits.masked_fill(legal_mask < 0.5, float('-inf'))
                action_probs = torch.softmax(masked_logits, dim=-1)
                
                # Safety check: if all probabilities are NaN/inf, fall back to heuristic
                if torch.isnan(action_probs).any() or torch.isinf(action_probs).any() or action_probs.sum() == 0:
                    print(f"[DEBUG] ERROR in action probs! Falling back to heuristic action")
                    return self._get_heuristic_action(observation)
                
                action = torch.multinomial(action_probs, 1).squeeze(-1)
                # Note: LSTMWrapper.forward_eval doesn't return updated state for single-step inference
            else:
                # Simple policy without LSTM
                action_logits = self.frozen_policy(obs_tensor)
                # Apply action masking - set illegal actions to -inf
                masked_logits = action_logits.masked_fill(legal_mask < 0.5, float('-inf'))
                action_probs = torch.softmax(masked_logits, dim=-1)
                
                # Safety check: if all probabilities are NaN/inf, fall back to heuristic
                if torch.isnan(action_probs).any() or torch.isinf(action_probs).any() or action_probs.sum() == 0:
                    print(f"[DEBUG] ERROR in action probs! Falling back to heuristic action")
                    return self._get_heuristic_action(observation)
                
                action = torch.multinomial(action_probs, 1).squeeze(-1)
                
        return action.cpu().numpy()
    
    def _get_heuristic_action(self, observation):
        """Get a reasonable heuristic action when no frozen policy is available."""
        # Extract sparse action mask and convert to legal actions list
        num_legal_moves = int(observation[1472])
        action_ids = observation[1473:1473+64].astype(int)  # Get action IDs
        
        if num_legal_moves > 0:
            valid_action_ids = action_ids[:num_legal_moves]
            # Clamp to valid range to prevent out-of-bounds
            legal_actions = np.clip(valid_action_ids, 0, 1967)
        else:
            legal_actions = np.array([], dtype=int)
        
        if len(legal_actions) == 0:
            return np.array([0])  # Fallback
            
        # Random legal action
        selected_action = np.random.choice(legal_actions)
        return np.array([selected_action])
        
    def reset(self, seed=None, **kwargs):
        """Reset environment and start first episode."""
        obs, info = self.base_env.reset(seed=seed, **kwargs)
        
        # Reset episode tracking
        self.current_episode_color = 0  # Start with WHITE episodes
        self.episode_move_count = 0
        self.game_move_count = 0
        
        # Clear episode buffers
        self.episode_observations = []
        self.episode_rewards = []
        self.episode_terminals = []
        self.episode_truncations = []
        
        # Check for pending policy updates
        if self._pending_policy_update is not None:
            self.update_frozen_policy(self._pending_policy_update)
            self._pending_policy_update = None
            
        return obs, info
        
    def reset(self, seed=None, **kwargs):
        """Reset environment and start first episode."""
        obs, info = self.base_env.reset(seed=seed, **kwargs)
        
        # Reset episode tracking
        self.current_episode_color = 0  # Start with WHITE episodes
        self.episode_move_count = 0
        self.game_move_count = 0
        
        # Clear episode buffers
        self.episode_observations = []
        self.episode_rewards = []
        self.episode_terminals = []
        self.episode_truncations = []
        
        # print(f"[DoubleBufferedChess] Reset - Starting WHITE episode")
        return obs, info
        
    def step(self, actions):
        """
        Double-buffered step: alternates between processing white and black moves.
        
        Returns episode data only when the current episode is complete.
        Each episode contains moves from only one color.
        """
        # If not using episode-per-color mode, just pass through
        if not self.episode_per_color:
            return self.base_env.step(actions)
        # Determine whose turn it is in the actual game
        white_to_move = (self.game_move_count % 2) == 0
        
        # Determine if this is the neural network's turn for this episode
        is_nn_turn = (white_to_move and self.current_episode_color == 0) or \
                     (not white_to_move and self.current_episode_color == 1)
        
        if is_nn_turn:
            # Neural network provides the action
            actual_actions = actions
            self.episode_move_count += 1
            player_name = "WHITE" if white_to_move else "BLACK"
            # print(f"[DoubleBufferedChess] Game move {self.game_move_count}: "
            #       f"{player_name} NN (episode move {self.episode_move_count})")
        else:
            # Opponent's turn - use frozen policy
            current_obs = self.base_env.observations[0]  # Get current observation
            opponent_action = self.get_opponent_action(current_obs)
            actual_actions = opponent_action
                
            player_name = "WHITE" if white_to_move else "BLACK"
            # print(f"[DoubleBufferedChess] Game move {self.game_move_count}: "
            #       f"{player_name} random ({len(legal_actions)} legal moves)")
        
        # Execute the move in base environment
        obs, rewards, terminals, truncations, info = self.base_env.step(actual_actions)
        self.game_move_count += 1
        
        # Only collect data when it's the neural network's turn
        if is_nn_turn:
            self.episode_observations.append(obs.copy())
            self.episode_rewards.append(rewards.copy())
            self.episode_terminals.append(terminals.copy())
            self.episode_truncations.append(truncations.copy())
        
        # Check if episode should end
        game_ended = terminals.any() or truncations.any()
        episode_full = self.episode_move_count >= self.moves_per_episode
        episode_should_end = game_ended or episode_full
        
        if episode_should_end:
            # Finalize current episode
            if len(self.episode_observations) > 0:
                # Return episode data as a batch
                episode_obs = np.array(self.episode_observations)
                episode_rewards = np.array(self.episode_rewards)
                episode_terminals = np.array(self.episode_terminals)
                episode_truncations = np.array(self.episode_truncations)
                
                # Mark last step as terminal to end episode
                episode_terminals[-1] = True
                
                # print(f"[DoubleBufferedChess] Episode complete: "
                #       f"{['WHITE', 'BLACK'][self.current_episode_color]} "
                #       f"({len(self.episode_observations)} moves)")
                
                # Reset for next episode
                if game_ended:
                    # Game ended - reset game counter
                    self.games_completed += 1
                    self.game_move_count = 0
                    # print(f"[DoubleBufferedChess] Game {self.games_completed} ended")
                
                # Switch episode color for next episode
                self.current_episode_color = 1 - self.current_episode_color
                self.episode_move_count = 0
                self.episode_observations = []
                self.episode_rewards = []
                self.episode_terminals = []
                self.episode_truncations = []
                
                # print(f"[DoubleBufferedChess] Next episode: "
                #       f"{['WHITE', 'BLACK'][self.current_episode_color]}")
                
                # Return episode data (flattened for vectorized training)
                return (episode_obs.reshape(-1, episode_obs.shape[-1]),
                        episode_rewards.flatten(),
                        episode_terminals.flatten(),
                        episode_truncations.flatten(),
                        info * len(self.episode_observations))
            else:
                # No data collected (shouldn't happen)
                # print("[WARNING] Episode ended with no data collected")
                return obs, np.zeros_like(rewards), np.array([True]), np.array([False]), info
        else:
            # Episode continues - return current step (but only if it's NN's turn)
            if is_nn_turn:
                return obs, rewards, np.array([False]), np.array([False]), info
            else:
                # Not NN's turn - return dummy data to continue
                return obs, np.zeros_like(rewards), np.array([False]), np.array([False]), info
    
    # Frozen policy delegation methods for self-play
    def update_frozen_policy(self, current_policy):
        """Delegate frozen policy updates to the underlying Chess environment."""
        return self.base_env.update_frozen_policy(current_policy)
    
    def schedule_policy_update(self, current_policy):
        """Delegate policy update scheduling to the underlying Chess environment.""" 
        return self.base_env.schedule_policy_update(current_policy)
    
    def should_update_frozen_policy(self):
        """Delegate frozen policy check to the underlying Chess environment."""
        return self.base_env.should_update_frozen_policy()
    
    def create_frozen_policy_from_template(self, template_policy):
        """Delegate frozen policy template creation to the underlying Chess environment."""
        return self.base_env.create_frozen_policy_from_template(template_policy)
    
    
    @property
    def _multiprocessing_mode(self):
        """Expose multiprocessing mode from underlying Chess environment."""
        return getattr(self.base_env, '_multiprocessing_mode', False)
    
    def notify(self):
        """Handle notifications from multiprocessing backend (e.g., policy updates)."""
        # Skip frozen policy updates in puzzle mode
        if hasattr(self.base_env, 'puzzle_mode') and self.base_env.puzzle_mode:
            return
            
        import tempfile
        import os
        import torch
        
        policy_file = os.path.join(tempfile.gettempdir(), 'puffer_chess_policy.pth')
        if os.path.exists(policy_file):
            try:
                # Load the updated policy from shared storage  
                state_dict = torch.load(policy_file, map_location='cpu', weights_only=True)
                
                # If frozen_policy doesn't exist in worker process, we need to create one
                # that matches the architecture of the saved policy  
                if self.frozen_policy is None:
                    # Use the same factory pattern as models.py
                    from pufferlib.models import policy_for
                    
                    # Create a dummy environment for policy creation (just need the observation space)
                    dummy_env = type('DummyEnv', (), {
                        'single_observation_space': self.single_observation_space,
                        'single_action_space': self.single_action_space
                    })()
                    
                    # Create policy with same architecture as training
                    policy = policy_for(dummy_env, hidden_size=256)
                    
                    # Load the state dict to get the weights
                    policy.load_state_dict(state_dict)
                    policy.eval()
                    
                    # Freeze all parameters
                    for param in policy.parameters():
                        param.requires_grad = False
                    
                    self.frozen_policy = policy
                    self.frozen_policy_state = None  # Reset LSTM state
                    self.policy_update_counter += 1
                    print(f"[DoubleBufferedChess] Environment with {self.num_envs} games created and loaded initial frozen policy #{self.policy_update_counter}")
                else:
                    # Update existing frozen policy
                    self.frozen_policy.load_state_dict(state_dict)
                    self.frozen_policy.eval()
                    self.policy_update_counter += 1
                    print(f"[DoubleBufferedChess] Environment with {self.num_envs} games loaded policy update #{self.policy_update_counter}")
                        
            except Exception as e:
                print(f"[DoubleBufferedChess] Failed to load policy update: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[DoubleBufferedChess] Policy file not found: {policy_file}")
    
    def render(self):
        return self.base_env.render()
    
    def close(self):
        if hasattr(self, 'base_env') and self.base_env is not None:
            try:
                self.base_env.close()
            except Exception as e:
                print(f"[DoubleBufferedChess] Warning: Error closing base environment: {e}")
            finally:
                self.base_env = None
    
    @property
    def emulated(self):
        return self.base_env.emulated

# Factory function for easy creation
def create_double_buffered_chess(**kwargs):
    return DoubleBufferedChess(**kwargs)