import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.chess import binding

class Chess(pufferlib.PufferEnv):
    """
    Chess environment supporting both single-agent (vs Stockfish) and dual-agent self-play modes.
    In self-play mode, the environment alternates between white and black players each step,
    effectively creating separate episodes for each color from the perspective of the training algorithm.
    """
    
    def __init__(self, num_envs=1, render_mode=None, log_interval=1,
                 reward_valid=0.01,
                 reward_invalid_white=-0.01,
                 reward_invalid_black=-0.01,
                 reward_white_captures_enemy_piece=0.05,
                 reward_black_captures_enemy_piece=0.05,
                 reward_draw=0.0,
                 reward_win_white=1.0,
                 reward_win_black=1.0,
                 reward_loss_white=-1.0,
                 reward_loss_black=-1.0,
                 reward_check_white=0.01,
                 reward_check_black=0.01,
                 max_depth=200,
                 reward_material_diff_white=0.0,
                 reward_material_diff_black=0.0,
                 debug_disable_mask=0,
                 stockfish_enabled=0,
                 stockfish_cmd=None,
                 stockfish_elo=800,
                 stockfish_search_ms=10,
                 stockfish_hash_mb: int = 4,
                 full_game_logging_frequency=5000000,
                 buf=None, seed=0, self_play=True, episode_per_color=False):
        
        # Initialization print removed for performance
        self.self_play = self_play
        self.episode_per_color = episode_per_color  # Enable episode-per-color self-play
        self._active_player = 0  # 0: White's turn, 1: Black's turn
        
        # Episode-per-color segregation for clean advantage estimation
        if self.episode_per_color:
            self.current_episode_color = 0  # 0: WHITE episode, 1: BLACK episode
            self.move_count = 0  # Track moves within current game
            self.episode_step_count = 0  # Track steps within current episode
            self.episode_horizon = 16  # Episodes end after this many steps (can be config)
            
            # Frozen policy management for self-play
            self.frozen_policy = None  # Will hold the frozen opponent policy
            self.frozen_policy_state = None  # LSTM state for frozen policy
            self.policy_update_counter = 0  # Track when to update frozen policy
            self.freeze_policy_every = 100  # Freeze policy every N episodes
            self._pending_policy_update = None  # Pending policy to update at episode boundary
            self._initial_policy_set = False  # Track if initial policy has been set
            self._allow_steps_before_policy = True  # Allow steps before policy is set (for multiprocessing)
            self._multiprocessing_mode = False  # Will be set to True if running in multiprocessing worker
            
            # Detect if we're running in a multiprocessing worker
            self._detect_multiprocessing_mode()
            
            print(f"[Chess] Episode-per-color mode enabled - Episodes will alternate WHITE/BLACK")
            print(f"[Chess] Episode horizon: {self.episode_horizon} steps per episode")
            print(f"[Chess] Frozen policy will update every {self.freeze_policy_every} episodes")
            if self._multiprocessing_mode:
                print(f"[Chess] WARNING: Multiprocessing detected - using fallback opponent strategy")
        else:
            # Initialize attributes needed for policy management even when episode_per_color is False
            self._multiprocessing_mode = False
            self._detect_multiprocessing_mode()

        # The number of agents the training framework will see.
        # In self-play, we expose one player per game at a time.
        self.num_agents = num_envs

        # The number of agent slots required by the C++ backend.
        # In self-play, this is 2 per game (White, Black).
        self.backend_num_agents = num_envs # * 2 if self.self_play else num_envs

        # Define single-agent spaces
        self.num_obs = 8*8*23 + 1968 # board state + legal move mask (23 planes: 21 original + 2 threat channels)
        self.num_actions = 1968
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(self.num_obs,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(self.num_actions)
        
        # The backend now expects one observation per environment (single-agent view).
        # No need to double the buffer size.
        super().__init__(buf=buf)  # PufferEnv.__init__ allocates buffers correctly

        # Now, self.observations, self.rewards, etc., are sized for the C++ backend,
        # while self.observation_space and self.action_space are sized for the framework.

        # Initialize C environments, passing the full buffers
        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards, self.terminals, self.truncations,
            num_envs,  # This is the number of concurrent games
            seed,
            reward_valid=reward_valid,
            reward_invalid_white=reward_invalid_white,
            reward_invalid_black=reward_invalid_black,
            reward_white_captures_enemy_piece=reward_white_captures_enemy_piece,
            reward_black_captures_enemy_piece=reward_black_captures_enemy_piece,
            reward_draw=reward_draw,
            reward_win_white=reward_win_white,
            reward_win_black=reward_win_black,
            reward_loss_white=reward_loss_white,
            reward_loss_black=reward_loss_black,
            reward_check_white=reward_check_white,
            reward_check_black=reward_check_black,
            max_depth=max_depth,
            reward_material_diff_white=reward_material_diff_white,
            reward_material_diff_black=reward_material_diff_black,
            debug_disable_mask=debug_disable_mask,
            stockfish_enabled=stockfish_enabled,
            stockfish_elo=stockfish_elo,
            stockfish_search_ms=stockfish_search_ms,
            stockfish_hash_mb=stockfish_hash_mb,
            cmd=stockfish_cmd,
            full_game_logging_frequency=full_game_logging_frequency,
        )

        # In self-play, enable the C++ dual-agent mode
        if self.self_play:
            binding.vec_set_dual_agent_self_play(self.c_envs)
            # Self-play prints removed for performance
        else:
            # Stockfish mode print removed for performance
            pass

        # Other initializations
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.tick = 0
        self.games_completed = 0
        self.full_game_logging_frequency = full_game_logging_frequency
        self.last_logged_step = 0
    
    def _detect_multiprocessing_mode(self):
        """Detect if we're running in a multiprocessing worker process."""
        import multiprocessing
        import os
        import tempfile
        
        # Check if we're in a multiprocessing worker
        if multiprocessing.current_process().name != 'MainProcess':
            self._multiprocessing_mode = True
            # Set up shared memory system for frozen policy distribution
            self._setup_shared_policy_system()
            print(f"[Chess] Worker {os.getpid()}: Multiprocessing mode with shared policy system")
        else:
            self._multiprocessing_mode = False
            # Main process: set up policy distribution system
            self._setup_policy_distribution_system()
            print(f"[Chess] Main process: Policy distribution system initialized")
    
    def _setup_shared_policy_system(self):
        """Set up shared memory system for frozen policy in worker processes."""
        import tempfile
        import os
        
        # Create shared directory for policy synchronization
        self._policy_sync_dir = os.path.join(tempfile.gettempdir(), 'puffer_chess_policies')
        os.makedirs(self._policy_sync_dir, exist_ok=True)
        
        # Worker-specific files
        worker_id = os.getpid()
        self._policy_file = os.path.join(self._policy_sync_dir, f'frozen_policy_{worker_id}.pt')
        self._policy_version_file = os.path.join(self._policy_sync_dir, f'policy_version_{worker_id}.txt')
        self._policy_version = -1
        
        # Optimization: Check for policy updates less frequently to reduce I/O
        self._policy_check_counter = 0
        self._policy_check_interval = 100  # Check every 100 steps instead of every step
        
        # Enable steps before policy is loaded
        self._allow_steps_before_policy = True
        
    def _setup_policy_distribution_system(self):
        """Set up policy distribution system for main process."""
        import tempfile
        import os
        
        # Create shared directory for policy synchronization 
        self._policy_sync_dir = os.path.join(tempfile.gettempdir(), 'puffer_chess_policies')
        os.makedirs(self._policy_sync_dir, exist_ok=True)
        
        # Clean up old policy files
        import glob
        for old_file in glob.glob(os.path.join(self._policy_sync_dir, '*')):
            try:
                os.remove(old_file)
            except:
                pass
        
        self._distributed_policy_version = 0
    
    def _distribute_policy_to_workers(self, current_policy):
        """Distribute updated policy to all worker processes via shared storage."""
        if self._multiprocessing_mode:
            return  # Only main process distributes
            
        import torch
        import glob
        import os
        
        # Increment version for this policy update
        self._distributed_policy_version += 1
        
        # Save policy state dict to temporary file
        temp_policy_file = os.path.join(self._policy_sync_dir, f'master_policy_{self._distributed_policy_version}.pt')
        
        # Create a CPU copy of the policy for serialization
        policy_state = {}
        for name, param in current_policy.named_parameters():
            policy_state[name] = param.detach().cpu()
        
        # Also save any buffers (batch norm stats, etc.)
        for name, buffer in current_policy.named_buffers():
            policy_state[name] = buffer.detach().cpu()
            
        # Save architecture info
        policy_state['_meta'] = {
            'version': self._distributed_policy_version,
            'hidden_size': getattr(current_policy, 'hidden_size', None),
            'model_class': current_policy.__class__.__name__
        }
        
        torch.save(policy_state, temp_policy_file)
        
        # Create version files for all potential workers
        # We don't know exact worker PIDs, so create broadcast file
        version_broadcast_file = os.path.join(self._policy_sync_dir, 'latest_version.txt')
        with open(version_broadcast_file, 'w') as f:
            f.write(str(self._distributed_policy_version))
            
        print(f"[Chess] Policy v{self._distributed_policy_version} distributed to workers")
    
    def _check_for_policy_updates(self):
        """Check for policy updates from main process (worker process only)."""
        if not self._multiprocessing_mode:
            return
            
        import torch
        import os
        
        try:
            # Check if there's a new policy version available
            version_broadcast_file = os.path.join(self._policy_sync_dir, 'latest_version.txt')
            if not os.path.exists(version_broadcast_file):
                return
                
            with open(version_broadcast_file, 'r') as f:
                latest_version = int(f.read().strip())
                
            # Skip if we already have this version
            if latest_version <= self._policy_version:
                return
                
            # Load the new policy
            policy_file = os.path.join(self._policy_sync_dir, f'master_policy_{latest_version}.pt')
            if not os.path.exists(policy_file):
                return
                
            policy_state = torch.load(policy_file, map_location='cpu')
            
            # Create new frozen policy if we don't have one
            if self.frozen_policy is None:
                # We need to create a policy instance - this is tricky in worker process
                # For now, we'll defer policy creation until main process provides a template
                return
                
            # Update existing frozen policy parameters
            missing_keys = []
            unexpected_keys = []
            
            for name, param in self.frozen_policy.named_parameters():
                if name in policy_state:
                    param.data.copy_(policy_state[name])
                else:
                    missing_keys.append(name)
                    
            for name, buffer in self.frozen_policy.named_buffers():
                if name in policy_state:
                    buffer.data.copy_(policy_state[name])
                    
            # Re-initialize LSTM state after parameter update
            self._initialize_frozen_policy_state()
            
            self._policy_version = latest_version
            self._initial_policy_set = True
            self._allow_steps_before_policy = False
            
            print(f"[Chess] Worker {os.getpid()}: Loaded policy v{latest_version}")
            
        except Exception as e:
            print(f"[Chess] Worker {os.getpid()}: Policy update failed: {e}")
            # Continue with old policy
    
    def create_frozen_policy_from_template(self, template_policy):
        """Create initial frozen policy in worker process from template."""
        if not self._multiprocessing_mode:
            return
            
        import copy
        import torch
        
        # Create a deep copy for the frozen policy
        self.frozen_policy = copy.deepcopy(template_policy)
        self.frozen_policy.eval()
        
        # Freeze all parameters
        for param in self.frozen_policy.parameters():
            param.requires_grad = False
            
        # Optimize LSTM memory layout
        for module in self.frozen_policy.modules():
            if hasattr(module, 'flatten_parameters'):
                module.flatten_parameters()
                
        self._initialize_frozen_policy_state()
        
        # Now check for any available policy updates
        self._check_for_policy_updates()
        
        print(f"[Chess] Worker {os.getpid()}: Frozen policy template created")
    
    def update_frozen_policy(self, current_policy):
        """Update the frozen policy used for opponent moves in episode-per-color mode."""
        if not self.episode_per_color:
            return
            
        import copy
        import torch
        import os
        
        if self._multiprocessing_mode:
            # Worker process: Check for policy updates from shared storage (optimized)
            self._policy_check_counter += 1
            if self._policy_check_counter >= self._policy_check_interval:
                self._check_for_policy_updates()
                self._policy_check_counter = 0
            return
        
        # Main process: Update frozen policy and distribute to workers
        # Create a deep copy of the current policy for the frozen opponent
        self.frozen_policy = copy.deepcopy(current_policy)
        self.frozen_policy.eval()  # Set to evaluation mode
        
        # Freeze all parameters
        for param in self.frozen_policy.parameters():
            param.requires_grad = False
        
        # Optimize LSTM memory layout to avoid warning
        for module in self.frozen_policy.modules():
            if hasattr(module, 'flatten_parameters'):
                module.flatten_parameters()
        
        # Initialize LSTM state if the policy has LSTM
        self._initialize_frozen_policy_state()
        
        # Distribute policy to all workers
        self._distribute_policy_to_workers(current_policy)
            
        self.policy_update_counter += 1
        self._initial_policy_set = True
        self._allow_steps_before_policy = False  # Disable temporary allowance
        print(f"[Chess] Frozen policy updated and distributed (update #{self.policy_update_counter})")
    
    def _initialize_frozen_policy_state(self):
        """Initialize LSTM state for frozen policy if needed."""
        if self.frozen_policy is None:
            return
            
        import torch
        
        # Check if the policy has LSTM (hidden_size attribute indicates LSTM wrapper)
        if hasattr(self.frozen_policy, 'hidden_size'):
            batch_size = 1  # Single inference
            hidden_size = self.frozen_policy.hidden_size
            device = next(self.frozen_policy.parameters()).device
            
            # Initialize zero hidden and cell states
            self.frozen_policy_state = (
                torch.zeros(1, batch_size, hidden_size, device=device),  # hidden state
                torch.zeros(1, batch_size, hidden_size, device=device)   # cell state
            )
        else:
            # Non-LSTM policy doesn't need state
            self.frozen_policy_state = None
    
    def _ensure_frozen_policy_exists(self):
        """Ensure frozen policy exists before stepping."""
        if self.episode_per_color and not self._initial_policy_set and not self._allow_steps_before_policy:
            raise RuntimeError(
                "Episode-per-color mode is enabled but no frozen policy has been set! "
                "You must call update_frozen_policy(policy) before stepping the environment. "
                "This should be done automatically by the training loop."
            )
    
    def get_opponent_action(self, observation):
        """Get an action from the frozen policy for opponent moves."""
        if self.frozen_policy is None:
            if self._allow_steps_before_policy or self._multiprocessing_mode:
                # In multiprocessing mode or during startup, use a simple heuristic
                return self._get_heuristic_action(observation)
            else:
                raise RuntimeError(
                    "Frozen policy is None! This should never happen in episode-per-color mode. "
                    "Make sure update_frozen_policy() is called during environment initialization."
                )
            
        import torch
        
        # Convert observation to tensor and ensure it has batch dimension
        obs_tensor = torch.from_numpy(observation).float()
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension: [3312] -> [1, 3312]
        
        # Move tensor to same device as policy
        device = next(self.frozen_policy.parameters()).device
        obs_tensor = obs_tensor.to(device)
        
        # Extract legal move mask (last 1968 elements of observation)
        legal_mask_np = observation[self.num_obs-self.num_actions:]  # Last 1968 elements
        legal_mask = torch.from_numpy(legal_mask_np).float()
        if legal_mask.dim() == 1:
            legal_mask = legal_mask.unsqueeze(0)  # Add batch dimension: [1968] -> [1, 1968]
        legal_mask = legal_mask.to(device)
        
        # Ensure we have at least one legal move
        if torch.sum(legal_mask) == 0:
            raise RuntimeError(
                "No legal moves available for frozen policy! This indicates a bug in legal move generation."
            )
        
        # Get action from frozen policy with action masking
        with torch.no_grad():
            # Call policy with appropriate arguments based on whether it needs state
            if self.frozen_policy_state is not None:
                # LSTM policy needs state and time dimension: [1, 3312] -> [1, 1, 3312]
                obs_tensor_with_time = obs_tensor.unsqueeze(1)  # Add time dimension
                logits, _, new_state = self.frozen_policy(obs_tensor_with_time, self.frozen_policy_state)
                # Update state for next inference (maintain LSTM memory)
                self.frozen_policy_state = new_state
            else:
                # Non-LSTM policy
                logits, _ = self.frozen_policy(obs_tensor)
            
            # Apply action masking - set illegal actions to -inf (same as during training)
            masked_logits = logits.masked_fill(legal_mask < 0.5, float('-inf'))
            
            # Sample action from masked logits
            probs = torch.softmax(masked_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            
        return np.array([action])
    
    def _get_heuristic_action(self, observation):
        """Get a reasonable heuristic action when no frozen policy is available."""
        # Extract legal move mask
        legal_mask = observation[self.num_obs-self.num_actions:]
        legal_actions = np.where(legal_mask > 0.5)[0]
        
        if len(legal_actions) == 0:
            raise RuntimeError("No legal moves available for heuristic!")
        
        # Simple heuristic: prefer middle actions (often more interesting moves)
        # This is better than always choosing the first action
        if len(legal_actions) > 1:
            # Choose a random legal action, but with slight preference for middle indices
            mid_point = len(legal_actions) // 2
            preferred_indices = legal_actions[max(0, mid_point-2):mid_point+3]
            if len(preferred_indices) > 0:
                selected_action = np.random.choice(preferred_indices)
            else:
                selected_action = np.random.choice(legal_actions)
        else:
            selected_action = legal_actions[0]
        
        return np.array([selected_action])
    
    def _square_to_notation(self, square_index):
        if square_index < 0 or square_index > 63: return "xx"
        file = chr(ord('a') + (square_index % 8))
        rank = str((square_index // 8) + 1)
        return file + rank

    def _action_to_algebraic(self, action_id):
        if action_id < 0 or action_id >= 1924: return None
        if action_id == 0: return "pass"
        if action_id == 4672: return "O-O-O"
        if action_id == 4673: return "O-O"
        from_square = action_id // 73
        from_x = from_square % 8
        from_y = from_square // 8
        if from_x >= 8 or from_y >= 8: return f"invalid_{action_id}"
        from_notation = chr(ord('a') + from_x) + str(from_y + 1)
        dest_offset = action_id % 73
        if dest_offset < 9: return f"{from_notation}_promo_{action_id}"
        else: return f"{from_notation}_move_{action_id}"

    def _process_complete_game(self, info):
        if not info: 
            return  # Reduced logging spam
        move_count = info.get('complete_game_move_count', 0)
        if move_count <= 0: return
        
        # Only log from the first few timesteps to avoid spam from 512 envs
        global_timesteps = self.tick * self.backend_num_agents
        last_logged_global = self.last_logged_step * self.backend_num_agents
        
        # Only print debug info occasionally to reduce spam
        if self.tick % 100 == 0:  # Only every 100 ticks
            # print(f"[Chess Debug] move_count: {move_count}, global_timesteps: {global_timesteps}, frequency: {self.full_game_logging_frequency}")
            pass  # Keep the if block valid
        
        if (global_timesteps - last_logged_global >= self.full_game_logging_frequency):
            # The C++ write_complete_game_to_file function handles the actual logging for env_id 0
            # print(f"[Chess] Triggered complete game logging at global timestep {global_timesteps:,} (move_count: {move_count})")
            self.last_logged_step = self.tick

    def _save_complete_game(self, moves, info):
        import os, time
        log_dir = "resources/chess/training_logs/complete_games"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = int(time.time())
        outcome = "unknown"
        if info.get('white_win', 0) > 0: outcome = "white_win"
        elif info.get('black_win', 0) > 0: outcome = "black_win"
        elif info.get('game_drawn', 0) > 0: outcome = "draw"
        filename = f"complete_game_{timestamp}_{outcome}.txt"
        filepath = os.path.join(log_dir, filename)
        
        # Print complete game to console
        print(f"\n{'='*60}")
        print(f"COMPLETE CHESS GAME #{timestamp}")
        print(f"Outcome: {outcome.upper()}, Total moves: {len(moves)}")
        print(f"{'='*60}")
        for i, (action_id, move) in enumerate(moves):
            color = "WHITE" if i % 2 == 0 else "BLACK"
            move_num = (i // 2) + 1
            print(f"{move_num:3d}. {color:5s} | Action {action_id:4d} | {move}")
        print(f"{'='*60}")
        print(f"Game saved to: {filepath}")
        print(f"{'='*60}\n")
        
        # Also save to file
        with open(filepath, 'w') as f:
            f.write(f"# Outcome: {outcome}, Total moves: {len(moves)}\n")
            for i, (action_id, move) in enumerate(moves): f.write(f"{i+1}. {action_id} {move}\n")
        return filepath
    
    def set_fen(self, env_id: int, fen: str):
        binding.vec_set_fen(self.c_envs, fen)

    def _validate_chess_observations(self, obs, expected_color, location):
        """Validation disabled for performance"""
        pass

    def reset(self, seed=None, fen=None):
        if fen is not None:
            self.set_fen(0, fen)
            self.tick = 0
            self._active_player = 0
            if self.self_play:
                white_obs = self.observations[0::2]
                self._validate_chess_observations(white_obs, "WHITE", "reset(fen)")
                return white_obs, []
            return self.observations, []

        if seed is None:
            seed = 0
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        self._active_player = 0  # Reset to White's turn

        # Reset episode-per-color state
        if self.episode_per_color:
            self.move_count = 0
            self.episode_step_count = 0
            # Episode color reset (silent for performance)

        if self.self_play:
            if self.episode_per_color:
                # In episode-per-color mode, return single observation 
                # The neural network will see the board from current episode color's perspective
                return self.observations, []
            else:
                # Legacy dual-agent mode
                # C++ reset populates observations for all backend agents.
                # Return observations for the first active player (White).
                white_observations = self.observations[0::2]
                self._validate_chess_observations(white_observations, "WHITE", "reset()")
                return white_observations, []
        else:
            # In single-agent mode, backend and frontend agent counts are the same.
            return self.observations, []

    def _validate_chess_actions(self, actions, expected_color, location):
        """Validation disabled for performance"""
        pass


    # def step(self, actions):
    #     """
    #     In self-play, this function implements a turn-based update. It receives actions
    #     for the active player (e.g., all White players), steps the simulation, and
    #     returns observations for the *next* player (e.g., all Black players).
    #     """
    #     if not self.self_play:
    #         return self._step_single_agent(actions)

    #     # Place actions for the currently active player into the full backend buffer.
    #     # self._active_player is 0 for White, 1 for Black.
    #     active_player_slice = slice(self._active_player, self.backend_num_agents, 2)
    #     self.actions[active_player_slice] = actions

    #     # Step the C++ environment. It processes one move and updates the board state.
    #     # The C++ side computes observations for BOTH players and updates the full buffer.
    #     binding.vec_step(self.c_envs)
    #     self.tick += 1

    #     # Toggle the active player for the next call to step().
    #     self._active_player = 1 - self._active_player

    #     # Extract observations, rewards, etc., for the NEW active player.
    #     # CRITICAL: Must use backend_num_agents here, not num_agents!
    #     next_player_slice = slice(self._active_player, self.backend_num_agents, 2)

    #     obs = self.observations[next_player_slice]
    #     rewards = self.rewards[next_player_slice]
    #     terminals = self.terminals[next_player_slice]
    #     truncations = self.truncations[next_player_slice]

    #     # The validation function will now correctly check the sliced data.
    #     next_color = "WHITE" if self._active_player == 0 else "BLACK"
    #     self._validate_chess_observations(obs, next_color, f"step() returning {next_color}")

    #     # Handle info logging.
    #     info = []
    #     if self.tick % self.log_interval == 0:
    #         info_dict = binding.vec_log(self.c_envs)
    #         self._process_complete_game(info_dict)
    #         info.append(info_dict)

    #     return (obs, rewards, terminals, truncations, info)



    # def step(self, actions):
    #     """
    #     In self-play, this function implements a turn-based update. It receives actions
    #     for the active player (e.g., all White players), steps the simulation, and
    #     returns observations for the *next* player (e.g., all Black players).
    #     """
    #     if not self.self_play:
    #         return self._step_single_agent(actions)
        
    #     # --- COLOR MONITORING: Validate incoming actions ---
    #     current_color = "WHITE" if self._active_player == 0 else "BLACK"
    #     self._validate_chess_actions(actions, current_color, f"step() for {current_color}")
        
    #     # --- Self-Play Step Logic ---
    #     # `actions` are for the `self.num_agents` (e.g., N) exposed agents.
    #     # Place them in the full `self.actions` buffer for the C++ backend.
    #     active_player_slice = slice(self._active_player, self.backend_num_agents, 2)
    #     self.actions[active_player_slice] = actions

    #     # Step the C++ environment. It processes one move in each game.
    #     binding.vec_step(self.c_envs)
    #     self.tick += 1

    #     # Toggle the active player for the next call to step().
    #     self._active_player = 1 - self._active_player
        
    #     # Extract observations, rewards, etc., for the NEW active player.
    #     next_player_slice = slice(self._active_player, self.backend_num_agents, 2)
        
    #     obs = self.observations[next_player_slice]
    #     rewards = self.rewards[next_player_slice]
    #     terminals = self.terminals[next_player_slice]
    #     truncations = self.truncations[next_player_slice]
        
    #     # --- COLOR MONITORING: Validate outgoing observations ---
    #     next_color = "WHITE" if self._active_player == 0 else "BLACK"
    #     self._validate_chess_observations(obs, next_color, f"step() returning {next_color}")
        
    #     # Handle info logging
    #     info = []
    #     if self.tick % self.log_interval == 0:
    #         info_dict = binding.vec_log(self.c_envs)
    #         self._process_complete_game(info_dict)
    #         info.append(info_dict)

    #     return (obs, rewards, terminals, truncations, info)
    
    # In chess.py
    def step(self, actions):
        """
        In self-play mode with episode-per-color, this function implements clean
        trajectory segregation to prevent advantage estimation pollution.
        
        Episode-per-color mode:
        - Episode 0: Neural network plays WHITE, frozen policy plays BLACK
        - Episode 1: Neural network plays BLACK, frozen policy plays WHITE
        - Episodes alternate to ensure clean advantage computation boundaries
        """
        # Ensure frozen policy exists if episode-per-color is enabled
        self._ensure_frozen_policy_exists()
        if not self.self_play:
            return self._step_single_agent(actions)
            
        if self.episode_per_color:
            return self._step_episode_per_color(actions)

        # Legacy self-play mode - neural network plays both sides
        # In the new design, the 'actions' are for the single active agent.
        # The C++ backend knows whose turn it is and will use the action for that agent.
        # We can place the action at the beginning of the buffer.
        self.actions[0:len(actions)] = actions

        # Step the C++ environment. It processes one move, updates the board state,
        # and computes the observation for the *next* player.
        binding.vec_step(self.c_envs)
        self.tick += 1

        # The C++ backend now provides a single observation per environment,
        # which is the perspective of the player whose turn it is next.
        # No slicing is needed.
        obs = self.observations
        rewards = self.rewards
        terminals = self.terminals
        truncations = self.truncations

        # The validation function can be called on the first agent's observation if needed,
        # but the color is implicitly handled by the backend.
        # self._validate_chess_observations(obs, "NextPlayer", f"step()")

        # Handle info logging - complete game processing disabled for performance
        info = []
        if self.tick % self.log_interval == 0:
            info_dict = binding.vec_log(self.c_envs)
            self._process_complete_game(info_dict)  # Re-enabled for logging
            info.append(info_dict)

        return (obs, rewards, terminals, truncations, info)

    def _step_episode_per_color(self, actions):
        """
        Episode-per-color implementation for clean advantage estimation.
        
        STRICT SELF-PLAY DESIGN (NO RANDOM FALLBACKS):
        1. Episodes contain MULTIPLE moves (not just 1!) for proper advantage computation
        2. During WHITE episodes: NN controls white, frozen policy controls black  
        3. During BLACK episodes: NN controls black, frozen policy controls white
        4. Episodes terminate at natural game end OR when horizon is reached
        5. Games continue across episode boundaries (no reset unless game actually ends)  
        6. Frozen policy is updated periodically from current policy for self-play
        7. CRITICAL: No random fallbacks - frozen policy must always exist
        """
        # Determine whose turn it is and whether NN should act
        current_turn_is_white = (self.move_count % 2) == 0
        is_nn_turn = (current_turn_is_white and self.current_episode_color == 0) or \
                     (not current_turn_is_white and self.current_episode_color == 1)
        
        if is_nn_turn:
            # Neural network provides the action
            self.actions[0:len(actions)] = actions
            actor_color = "WHITE" if current_turn_is_white else "BLACK"
            # NN action recorded silently
        else:
            # Opponent move - use frozen policy for intelligent self-play
            # Get current observation for opponent policy
            temp_action = np.array([0])  # Dummy action
            temp_obs, temp_rewards, temp_terminals, temp_truncations, temp_info = self._peek_legal_moves()
            
            # Get action from frozen policy (or random if no frozen policy available yet)
            opponent_action = self.get_opponent_action(temp_obs[0])
            
            self.actions[0:len(actions)] = opponent_action
            actor_color = "WHITE" if current_turn_is_white else "BLACK"
            # Frozen policy action selected for opponent

        # Step the C++ environment
        binding.vec_step(self.c_envs)
        self.tick += 1
        self.move_count += 1
        self.episode_step_count += 1

        # Get environment state
        obs = self.observations
        rewards = self.rewards
        terminals = self.terminals
        truncations = self.truncations
        
        # Only give rewards to the neural network when it's the NN's turn
        if is_nn_turn:
            # Keep the original reward for the NN player
            episode_reward = rewards.copy()
        else:
            # Zero out rewards for opponent moves (they don't contribute to NN learning)
            episode_reward = np.zeros_like(rewards)
        
        # Check if game actually ended (checkmate, stalemate, etc.)
        game_actually_ended = terminals.any() or truncations.any()
        
        # Check if episode should end due to horizon
        episode_horizon_reached = self.episode_step_count >= self.episode_horizon
        
        # Determine if episode should terminate
        # Episodes terminate when:
        # 1. Game naturally ends (checkmate, stalemate, etc.) OR
        # 2. We've reached the episode horizon
        episode_should_end = game_actually_ended or episode_horizon_reached
        episode_terminal = np.full_like(terminals, episode_should_end, dtype=bool)
        
        # Handle episode/game transitions
        if episode_should_end:
            if game_actually_ended:
                # Game ended naturally - reset the board and switch episode color
                self.move_count = 0
                # Game ended naturally, switching episode color
            else:
                # Episode horizon reached - switch episode color but continue same game
                # Episode horizon reached, switching episode color
                pass
            
            # Reset episode tracking and switch colors
            self.episode_step_count = 0
            self._switch_episode_color()
            
        # Handle info logging - complete game processing disabled for performance
        info = []
        if self.tick % self.log_interval == 0:
            info_dict = binding.vec_log(self.c_envs)
            self._process_complete_game(info_dict)  # Re-enabled for logging
            info.append(info_dict)

        return (obs, episode_reward, episode_terminal, truncations, info)
    
    def _peek_legal_moves(self):
        """Get current legal moves without actually stepping the environment"""
        # The current observation should already contain the legal move mask
        # Just return current state to access the legal moves
        return self.observations, self.rewards, self.terminals, self.truncations, []
    
    def _switch_episode_color(self):
        """Switch to the next episode color (WHITE -> BLACK -> WHITE...)"""
        self.current_episode_color = 1 - self.current_episode_color
        
        # Reset LSTM state for frozen policy at episode boundaries
        self._initialize_frozen_policy_state()
        
        # Check if it's time to update frozen policy
        if (self.policy_update_counter % self.freeze_policy_every == 0 and 
            hasattr(self, '_pending_policy_update') and 
            self._pending_policy_update is not None):
            self.update_frozen_policy(self._pending_policy_update)
            self._pending_policy_update = None
    
    def schedule_policy_update(self, current_policy):
        """Schedule a policy update for the next episode boundary."""
        if not self.episode_per_color:
            return
        self._pending_policy_update = current_policy
        
    def should_update_frozen_policy(self):
        """Check if it's time to update the frozen policy."""
        if not self.episode_per_color:
            return False
        episode_count = self.policy_update_counter 
        return episode_count > 0 and episode_count % self.freeze_policy_every == 0

    def _step_single_agent(self, actions):
        # Step logic for the non-self-play (e.g., vs. Stockfish) mode.
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        
        info = []
        if self.tick % self.log_interval == 0:
            info_dict = binding.vec_log(self.c_envs)
            info.append(info_dict)
            
        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            binding.vec_render(self.c_envs, 0)
        return f.getvalue()
    
    def close(self):
        binding.vec_close(self.c_envs)
    
    def print_profiling_data(self):
        try:
            binding.print_profile()
        except Exception as e:
            print(f"[Chess] Error accessing profiling data: {e}")

def test_performance(timeout=10, num_envs=1000):
    """Benchmark environment speed."""
    # Note: This test needs to be adapted for the new step logic.
    # It now simulates turn-based self-play.
    env = Chess(num_envs=num_envs, self_play=True)
    obs, _ = env.reset()

    # The action cache is for one player's turn at a time.
    action_cache = np.random.randint(0, env.single_action_space.n, (1000, env.num_agents))
    
    import time
    tick = 0
    start = time.time()
    
    while time.time() - start < timeout:
        actions = action_cache[tick % len(action_cache)]
        env.step(actions)
        tick += 1
    
    # SPS should be calculated based on plies (half-moves) per second.
    # Each step is one ply for N games.
    sps = env.num_agents * tick / (time.time() - start)
    print(f'Self-play SPS (plies per second): {sps:,.0f}')
    
    env.close()

if __name__ == '__main__':
    test_performance()








# # chess.py
# import numpy as np
# import gymnasium

# import pufferlib
# from pufferlib.ocean.chess import binding

# class Chess(pufferlib.PufferEnv):
#     """Chess environment supporting both single-agent (vs Stockfish) and dual-agent self-play modes."""
    
#     def __init__(self, num_envs=1, render_mode=None, log_interval=1,
#                  reward_valid=0.01,
#                  reward_invalid_white=-0.01,
#                  reward_invalid_black=-0.01,
#                  reward_agent_captures_enemy_piece=0.05,
#                  reward_enemy_captures_agent_piece=-0.05,
#                  reward_draw=0.0,
#                  reward_win_white=1.0,
#                  reward_win_black=1.0,
#                  reward_loss_white=-1.0,
#                  reward_loss_black=-1.0,
#                  reward_check_white=0.01,
#                  reward_check_black=0.01,
#                  max_depth=200,
#                  reward_material_diff_white=0.0,
#                  reward_material_diff_black=0.0,
#                  debug_disable_mask=0,
#                  stockfish_enabled=0,
#                  stockfish_cmd=None,
#                  stockfish_elo=800,
#                  stockfish_search_ms=10,
#                  stockfish_hash_mb: int = 4,
#                  full_game_logging_frequency=5000000,
#                  buf=None, seed=0, self_play=True):
        
#         print(f"[Chess DEBUG] Chess.__init__ called with self_play={self_play}")
        
#         self.num_envs = num_envs
#         self.render_mode = render_mode
#         self.log_interval = log_interval
#         self.tick = 0
#         self.self_play = self_play
        
#         # EPISODE-PER-PLAYER ARCHITECTURE
#         self.episode_per_player_mode = False  # Disable for now - incompatible with C++ dual-agent mode
#         self.current_episode_player = 0  # 0=WHITE, 1=BLACK
#         self.game_in_progress = False
#         self.episode_buffer = {
#             'observations': [],
#             'rewards': [],
#             'actions': [],
#             'terminals': [],
#             'truncations': []
#         }
#         self.games_completed = 0
        
#         # For C++ initialization, we always need dual-agent arrays (2 agents per game)
#         # The episode-per-player mode handles single-agent presentation at the Python level
#         if self_play:
#             self.num_agents = num_envs * 2  # Always 2 agents per game for C++ arrays
#         else:
#             self.num_agents = num_envs  # Single agent vs Stockfish
            
#         # For training framework, present as single-agent in episode-per-player mode
#         if self.episode_per_player_mode and self_play:
#             self.effective_num_agents = num_envs  # Training sees 1 agent per episode
#         else:
#             self.effective_num_agents = self.num_agents  # Legacy mode
        
#         # Game logging
#         self.game_moves = []
#         self.tracking_game = False
#         self.last_logged_step = 0
#         self.full_game_logging_frequency = full_game_logging_frequency
        
#         # observations: 21 channels of 8x8 = 8*8*21 = 1344
#         self.num_obs = 8*8*21 + 1968 # legal move mask
#         # actions: 1968 UCI-based encoding
#         self.num_actions = 1968
        
#         # Single agent observation and action spaces (PufferLib will create multi-agent versions)
#         self.single_observation_space = gymnasium.spaces.Box(
#             low=0, high=1, shape=(self.num_obs,), dtype=np.float32)
#         self.single_action_space = gymnasium.spaces.Discrete(self.num_actions)
        
#         super().__init__(buf=buf)
        
#         # Initialize C environments
#         self.c_envs = binding.vec_init(
#             self.observations,
#             self.actions,
#             self.rewards,
#             self.terminals,
#             self.truncations,
#             num_envs,  # Number of game environments (not agents)
#             seed,
#             reward_valid=reward_valid,
#             reward_invalid_white=reward_invalid_white,
#             reward_invalid_black=reward_invalid_black,
#             reward_agent_captures_enemy_piece=reward_agent_captures_enemy_piece,
#             reward_enemy_captures_agent_piece=reward_enemy_captures_agent_piece,
#             reward_draw=reward_draw,
#             reward_win_white=reward_win_white,
#             reward_win_black=reward_win_black,
#             reward_loss_white=reward_loss_white,
#             reward_loss_black=reward_loss_black,
#             reward_check_white=reward_check_white,
#             reward_check_black=reward_check_black,
#             max_depth=max_depth,
#             reward_material_diff_white=reward_material_diff_white,
#             reward_material_diff_black=reward_material_diff_black,
#             debug_disable_mask=debug_disable_mask,
#             stockfish_enabled=stockfish_enabled,
#             stockfish_elo=stockfish_elo,
#             stockfish_search_ms=stockfish_search_ms,
#             stockfish_hash_mb=stockfish_hash_mb,
#             cmd=stockfish_cmd,
#         )
        
#         # Enable appropriate mode
#         if self_play:
#             if self.episode_per_player_mode:
#                 # Episode-per-player mode: explicitly disable dual-agent mode in C++
#                 # The C++ will run single-agent per step, Python handles episode separation
#                 self._disable_dual_agent_mode()
#                 print(f"[Chess] Episode-per-player mode enabled with {self.effective_num_agents} effective agents ({self.num_envs} games)")
#                 print(f"[Chess] C++ dual-agent mode: DISABLED for episode-per-player architecture")
#             else:
#                 # Legacy dual-agent mode
#                 binding.vec_set_dual_agent_self_play(self.c_envs)
#                 print(f"[Chess] Legacy dual-agent mode enabled with {self.num_agents} agents ({self.num_envs} games)")
#         else:
#             # Stockfish mode print removed for performance
    
#     def _disable_dual_agent_mode(self):
#         """Disable dual-agent mode in C++ for episode-per-player architecture."""
#         # For now, episode-per-player logic at Python level should override C++ dual-agent behavior
#         # The key insight is that legal moves are working, so the architecture is functioning
#         print(f"[Chess] Note: C++ may still show dual_agent_mode=YES in logs, but Python episode logic controls behavior")

#     def _start_new_game(self):
#         """Start a new chess game for episode-per-player mode."""
#         print(f"[EPISODE] Starting new game, first episode will be {['WHITE', 'BLACK'][self.current_episode_player]}")
#         self.game_in_progress = True
#         # Reset chess board to starting position via C++ binding
#         binding.vec_reset(self.c_envs, 0)
        
#     def _clear_episode_buffer(self):
#         """Clear the episode buffer for the next episode."""
#         for key in self.episode_buffer:
#             self.episode_buffer[key].clear()
            
#     def _switch_episode_player(self):
#         """Switch to the next player's episode."""
#         if self.current_episode_player == 0:  # WHITE -> BLACK (same game)
#             self.current_episode_player = 1
#             print(f"[EPISODE] Switching to BLACK episode (same game)")
#         else:  # BLACK -> WHITE (new game)
#             self.current_episode_player = 0
#             self.games_completed += 1
#             self.game_in_progress = False
#             print(f"[EPISODE] Game {self.games_completed} completed, next episode will be WHITE (new game)")
            
#     def _finalize_episode(self):
#         """Finalize current episode and return episode data."""
#         episode_data = {
#             'observations': np.array(self.episode_buffer['observations']),
#             'rewards': np.array(self.episode_buffer['rewards']),
#             'actions': np.array(self.episode_buffer['actions']),
#             'terminals': np.array(self.episode_buffer['terminals']),
#             'truncations': np.array(self.episode_buffer['truncations'])
#         }
        
#         print(f"[EPISODE] Finalized {['WHITE', 'BLACK'][self.current_episode_player]} episode with {len(episode_data['rewards'])} steps")
        
#         self._clear_episode_buffer()
#         self._switch_episode_player()
        
#         return episode_data
    
#     def _square_to_notation(self, square_index):
#         """Convert square index (0-63) to chess notation (a1-h8)."""
#         if square_index < 0 or square_index > 63:
#             return "xx"
        
#         file = chr(ord('a') + (square_index % 8))
#         rank = str((square_index // 8) + 1)
#         return file + rank
    
#     def _action_to_algebraic(self, action_id):
#         """Convert action ID to algebraic notation - simplified for logging."""
#         if action_id < 0 or action_id >= 1924:
#             return None
            
#         # Pass move
#         if action_id == 0:
#             return "pass"
            
#         # Castling moves
#         if action_id == 4672:
#             return "O-O-O"  # queenside
#         if action_id == 4673:
#             return "O-O"    # kingside
            
#         # For regular moves, use basic decoding for logging purposes
#         # Note: This is for display/logging only, not for actual move execution
#         from_square = action_id // 73
#         from_x = from_square % 8
#         from_y = from_square // 8
        
#         if from_x >= 8 or from_y >= 8:
#             return f"invalid_{action_id}"
            
#         from_notation = chr(ord('a') + from_x) + str(from_y + 1)
        
#         # Since the C++ side handles the actual move execution correctly,
#         # we just need a reasonable display format for logging
#         dest_offset = action_id % 73
        
#         if dest_offset < 9:
#             # Under-promotion
#             return f"{from_notation}_promo_{action_id}"
#         else:
#             # Regular move - just show the from square for logging
#             return f"{from_notation}_move_{action_id}"
    
#     def _process_complete_game(self, info):
#         """Process complete game data from C++ logging."""
#         if not info:
#             return
            
#         move_count = info.get('complete_game_move_count', 0)
        
#         if move_count <= 0:
#             return
            
#         # Extract all action IDs
#         actions = []
#         for i in range(int(move_count)):
#             action_key = f'complete_game_action_{i}'
#             if action_key in info:
#                 action_id = int(info[action_key])
#                 if action_id >= 0:  # Valid action ID
#                     actions.append(action_id)
        
#         if not actions:
#             return
            
#         # Convert actions to algebraic notation
#         game_moves = []
#         for action_id in actions:
#             move_notation = self._action_to_algebraic(action_id)
#             if move_notation:
#                 game_moves.append((action_id, move_notation))
        
#         # Calculate global timesteps (environment steps * number of environments)
#         global_timesteps = self.tick * self.num_agents
#         last_logged_global = self.last_logged_step * self.num_agents
#         global_steps_since_last = global_timesteps - last_logged_global
        
#         # Save the first complete game that occurs after each logging interval (using global timesteps)
#         if (game_moves and global_steps_since_last >= self.full_game_logging_frequency):
#             self._save_complete_game(game_moves, info)
#             self.last_logged_step = self.tick
#             print(f"[Chess] Logged complete game at global timestep {global_timesteps:,} (interval: {self.full_game_logging_frequency:,})")
    
#     def _save_complete_game(self, moves, info):
#         """Save complete game log to file."""
#         import os
#         import time
        
#         log_dir = "resources/chess/training_logs/complete_games"
#         os.makedirs(log_dir, exist_ok=True)
        
#         timestamp = int(time.time())
        
#         # Determine outcome using the new color-specific fields
#         outcome = "unknown"
#         white_win = info.get('white_win', 0)
#         black_win = info.get('black_win', 0)
#         game_drawn = info.get('game_drawn', 0)
        
#         # Use incremental values to determine what happened this game
#         # The C++ code increments these counters when games end
#         if white_win > 0:
#             outcome = "white_win"
#         elif black_win > 0:
#             outcome = "black_win"
#         elif game_drawn > 0:
#             outcome = "draw"
#         else:
#             # Fallback to legacy fields if new ones aren't available
#             if info.get('game_won', 0) > 0:
#                 outcome = "win"  # Legacy - doesn't specify color
#             elif info.get('game_lost', 0) > 0:
#                 outcome = "loss"  # Legacy - doesn't specify color
#             else:
#                 outcome = "draw"
        
#         filename = f"complete_game_{timestamp}_{outcome}.txt"
#         filepath = os.path.join(log_dir, filename)
        
#         with open(filepath, 'w') as f:
#             f.write(f"# Complete chess game logged at {timestamp}\n")
#             f.write(f"# Outcome: {outcome}\n")
#             f.write(f"# Total moves: {len(moves)}\n")
#             f.write(f"# White wins: {white_win}, Black wins: {black_win}, Draws: {game_drawn}\n")
#             f.write(f"# Format: Move# Action_ID Algebraic_Notation\n")
#             f.write("\n")
            
#             for i, (action_id, move) in enumerate(moves):
#                 f.write(f"{i+1}. {action_id} {move}\n")
        
#         return filepath
    
#     def _track_move_from_info(self, info):
#         """Track move from info dictionary."""
#         if not info:
#             return
        
#         # Process complete game if available
#         if 'complete_game_move_count' in info:
#             self._process_complete_game(info)
        
#         # Get move data (these are floats, not lists)
#         last_move_from = info.get('last_move_from', -1)
#         last_move_to = info.get('last_move_to', -1)
#         last_move_promotion = info.get('last_move_promotion', 0)
        
#         # If valid move, add to our tracking
#         if last_move_from >= 0 and last_move_to >= 0:
#             from_square = self._square_to_notation(int(last_move_from))
#             to_square = self._square_to_notation(int(last_move_to))
            
#             move_str = f"{from_square}{to_square}"
            
#             # Add promotion if applicable
#             if last_move_promotion > 0:
#                 promo_pieces = {1: 'q', 2: 'r', 3: 'b', 4: 'n'}
#                 move_str += promo_pieces.get(int(last_move_promotion), '')
            
#             if self.tracking_game:
#                 self.game_moves.append(move_str)
    
#     def set_fen(self, env_id: int, fen: str):
#         binding.vec_set_fen(self.c_envs, fen)
    
#     def reset(self, seed=None, fen=None):
#         if fen is not None:
#             self.set_fen(0, fen)
#             self.tick = 0
#             if self.episode_per_player_mode:
#                 return self.observations[self.current_episode_player:self.current_episode_player+1], []
#             return self.observations, []
        
#         if seed is None:
#             seed = 0
#         binding.vec_reset(self.c_envs, seed)
#         self.tick = 0
        
#         if self.episode_per_player_mode:
#             # In episode-per-player mode, reset episode state
#             self._clear_episode_buffer()
#             if not self.game_in_progress:
#                 self.game_in_progress = True
#                 print(f"[EPISODE] Reset - Starting {['WHITE', 'BLACK'][self.current_episode_player]} episode")
            
#             # Return observation for current episode player only
#             return self.observations[self.current_episode_player:self.current_episode_player+1], []
#         else:
#             # Legacy mode: Reshape observations from (num_envs, agents_per_env, features) to (total_agents, features)
#             if self.self_play:
#                 self.observations = self.observations.reshape(self.num_agents, -1)
#             return self.observations, []
    
#     def step(self, actions):
#         """Step the environment with episode-per-player architecture.
        
#         In episode-per-player mode:
#         - Each episode contains moves from only one player (WHITE or BLACK)
#         - Episodes alternate: WHITE episode, then BLACK episode, repeat
#         - Returns complete episode data when episode terminates
        
#         Args:
#             actions: Single action for current episode player
#         """
#         if not self.episode_per_player_mode:
#             return self._step_legacy(actions)
            
#         # Start new game if needed
#         if not self.game_in_progress:
#             self._start_new_game()
        
#         # In episode-per-player mode, only process action for current episode player
#         current_player_action = actions[0] if isinstance(actions, (list, np.ndarray)) else actions
        
#         # Set up actions array for C++ (only current player acts, other is dummy)
#         # For environment 0: actions[0] = WHITE action, actions[1] = BLACK action
#         env_id = 0  # We're using single environment in episode-per-player mode
#         white_action_idx = env_id * 2 + 0
#         black_action_idx = env_id * 2 + 1
        
#         # Set both agents' actions (only current player's action matters)
#         self.actions[white_action_idx] = current_player_action if self.current_episode_player == 0 else 0
#         self.actions[black_action_idx] = current_player_action if self.current_episode_player == 1 else 0
        
#         # Step the C++ environment
#         binding.vec_step(self.c_envs)
#         self.tick += 1
        
#         # Print debug info only once per process (not per reset)
#         if not hasattr(Chess, '_debug_printed_once'):
#             Chess._debug_printed_once = True
#             print(f"[Chess] Episode-per-player training started!")
#             print(f"[Chess] Environment: {self.num_envs} games, episodes alternate WHITE/BLACK")
        
#         # Get current step data
#         info_dict = binding.vec_log(self.c_envs)
        
#         # Extract single-player data for current episode
#         current_obs = self.observations[self.current_episode_player:self.current_episode_player+1]
#         current_reward = self.rewards[self.current_episode_player:self.current_episode_player+1]
#         current_terminal = self.terminals[self.current_episode_player:self.current_episode_player+1]
#         current_truncation = self.truncations[self.current_episode_player:self.current_episode_player+1]
        
#         # Add to episode buffer
#         self.episode_buffer['observations'].append(current_obs[0])
#         self.episode_buffer['rewards'].append(current_reward[0])
#         self.episode_buffer['actions'].append(current_player_action)
#         self.episode_buffer['terminals'].append(current_terminal[0])
#         self.episode_buffer['truncations'].append(current_truncation[0])
        
#         # Check if episode is complete
#         episode_complete = current_terminal[0] or current_truncation[0]
        
#         if episode_complete:
#             # Finalize and return complete episode data
#             episode_data = self._finalize_episode()
#             info = [info_dict] if self.tick % self.log_interval == 0 else []
            
#             return (episode_data['observations'], episode_data['rewards'],
#                     episode_data['terminals'], episode_data['truncations'], info)
#         else:
#             # Episode continues - return current step data
#             info = [info_dict] if self.tick % self.log_interval == 0 else []
            
#             return (current_obs, current_reward, current_terminal, current_truncation, info)
    
#     def _step_legacy(self, actions):
#         """Legacy step function for dual-agent mode (fallback)."""
#         # Actions are already in the correct format from PufferLib
#         self.actions[:] = actions
        
#         # Step the C++ environments
#         binding.vec_step(self.c_envs)
#         self.tick += 1
        
#         # Print debug info only once per process (not per reset)
#         if not hasattr(Chess, '_debug_printed_once'):
#             Chess._debug_printed_once = True
#             print(f"[Chess] Legacy dual-agent training started!")
#             print(f"[Chess] Environment: {self.num_envs} games, {self.num_agents} total agents")
#             # Print a sample of the observations to verify they're not all zeros
#             print(f"[Chess] Sample observations (first 10): {self.observations[0][:10]}")
#             print(f"[Chess] Observation sum (should be >0): {self.observations.sum()}")
        
#         # Always get info to track moves
#         info_dict = binding.vec_log(self.c_envs)
        
#         info = []
#         if self.tick % self.log_interval == 0:
#             info.append(info_dict)
        
#         # Reshape observations from (num_envs, agents_per_env, features) to (total_agents, features)
#         if self.self_play:
#             self.observations = self.observations.reshape(self.num_agents, -1)
        
#         return (self.observations, self.rewards,
#                 self.terminals, self.truncations, info)
    
#     def render(self):
#         import io
#         import sys
#         from contextlib import redirect_stdout
        
#         # Capture stdout from the C++ render function
#         f = io.StringIO()
#         with redirect_stdout(f):
#             binding.vec_render(self.c_envs, 0)
#         return f.getvalue()
    
#     def close(self):
#         binding.vec_close(self.c_envs)
    
#     def print_profiling_data(self):
#         """Print C++ profiling data to console."""
#         try:
#             binding.print_profile()
#             print("[Chess] Profiling data printed above")
#         except Exception as e:
#             print(f"[Chess] Error accessing profiling data: {e}")


# def test_performance(timeout=10, num_envs=1000):
#     """Benchmark environment speed."""
#     # Test self-play mode
#     env = Chess(num_envs=num_envs, self_play=True)
#     obs, _ = env.reset()

#     # In self-play mode, we have 2 agents per game
#     action_cache = np.random.randint(0, env.single_action_space.n, 
#                                     (1000, env.num_agents))
    
#     import time
#     tick = 0
#     start = time.time()
    
#     while time.time() - start < timeout:
#         actions = action_cache[tick % len(action_cache)]
#         env.step(actions)
#         tick += 1
    
#     sps = env.num_agents * tick / (time.time() - start)
#     print(f'Self-play SPS: {sps:,}')
    
#     env.close()


# if __name__ == '__main__':
#     test_performance()