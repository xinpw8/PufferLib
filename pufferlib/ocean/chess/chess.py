# chess.py
import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.chess import binding

class Chess(pufferlib.PufferEnv):
    """Chess environment supporting both single-agent (vs Stockfish) and dual-agent self-play modes."""
    
    def __init__(self, num_envs=1, render_mode=None, log_interval=1,
                 reward_valid=0.01,
                 reward_invalid_white=-0.01,
                 reward_invalid_black=-0.01,
                 reward_agent_captures_enemy_piece=0.05,
                 reward_enemy_captures_agent_piece=-0.05,
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
                 buf=None, seed=0, self_play=True):
        
        self.num_envs = num_envs
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.tick = 0
        self.self_play = self_play
        
        # Set number of agents based on mode
        if self_play:
            self.num_agents = 2 * num_envs  # White and black agents for each game
        else:
            self.num_agents = num_envs  # Single agent vs Stockfish
        
        # Game logging
        self.game_moves = []
        self.tracking_game = False
        self.last_logged_step = 0
        self.full_game_logging_frequency = full_game_logging_frequency
        
        # observations: 21 channels of 8x8 = 8*8*21 = 1344
        self.num_obs = 8*8*21 + 1924 # legal move mask
        # actions: 1924 UCI-based encoding
        self.num_actions = 1924
        
        # Single agent observation and action spaces (PufferLib will create multi-agent versions)
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(self.num_obs,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(self.num_actions)
        
        super().__init__(buf=buf)
        
        # initialize c environments
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,  # Number of game environments (not agents)
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
            max_depth=max_depth,
            reward_material_diff_white=reward_material_diff_white,
            reward_material_diff_black=reward_material_diff_black,
            debug_disable_mask=debug_disable_mask,
            stockfish_enabled=stockfish_enabled,
            stockfish_cmd=stockfish_cmd,
            stockfish_elo=stockfish_elo,
            stockfish_search_ms=stockfish_search_ms,
            stockfish_hash_mb=stockfish_hash_mb)
        
        # Enable appropriate mode
        if self_play:
            binding.vec_set_dual_agent_self_play(self.c_envs)
            print(f"[Chess] Self-play mode enabled with {self.num_agents} agents ({self.num_agents//2} games)")
        else:
            print(f"[Chess] Single-agent mode vs Stockfish (ELO={stockfish_elo})")
    
    def _square_to_notation(self, square_index):
        """Convert square index (0-63) to chess notation (a1-h8)."""
        if square_index < 0 or square_index > 63:
            return "xx"
        
        file = chr(ord('a') + (square_index % 8))
        rank = str((square_index // 8) + 1)
        return file + rank
    
    def _action_to_algebraic(self, action_id):
        """Convert action ID to algebraic notation - simplified for logging."""
        if action_id < 0 or action_id >= 1924:
            return None
            
        # Pass move
        if action_id == 0:
            return "pass"
            
        # Castling moves
        if action_id == 4672:
            return "O-O-O"  # queenside
        if action_id == 4673:
            return "O-O"    # kingside
            
        # For regular moves, use basic decoding for logging purposes
        # Note: This is for display/logging only, not for actual move execution
        from_square = action_id // 73
        from_x = from_square % 8
        from_y = from_square // 8
        
        if from_x >= 8 or from_y >= 8:
            return f"invalid_{action_id}"
            
        from_notation = chr(ord('a') + from_x) + str(from_y + 1)
        
        # Since the C++ side handles the actual move execution correctly,
        # we just need a reasonable display format for logging
        dest_offset = action_id % 73
        
        if dest_offset < 9:
            # Under-promotion
            return f"{from_notation}_promo_{action_id}"
        else:
            # Regular move - just show the from square for logging
            return f"{from_notation}_move_{action_id}"
    
    def _process_complete_game(self, info):
        """Process complete game data from C++ logging."""
        if not info:
            return
            
        move_count = info.get('complete_game_move_count', 0)
        
        if move_count <= 0:
            return
            
        # Extract all action IDs
        actions = []
        for i in range(int(move_count)):
            action_key = f'complete_game_action_{i}'
            if action_key in info:
                action_id = int(info[action_key])
                if action_id >= 0:  # Valid action ID
                    actions.append(action_id)
        
        if not actions:
            return
            
        # Convert actions to algebraic notation
        game_moves = []
        for action_id in actions:
            move_notation = self._action_to_algebraic(action_id)
            if move_notation:
                game_moves.append((action_id, move_notation))
        
        # Calculate global timesteps (environment steps * number of environments)
        global_timesteps = self.tick * self.num_envs
        last_logged_global = self.last_logged_step * self.num_envs
        global_steps_since_last = global_timesteps - last_logged_global
        
        # Save the first complete game that occurs after each logging interval (using global timesteps)
        if (game_moves and global_steps_since_last >= self.full_game_logging_frequency):
            self._save_complete_game(game_moves, info)
            self.last_logged_step = self.tick
            print(f"[Chess] Logged complete game at global timestep {global_timesteps:,} (interval: {self.full_game_logging_frequency:,})")
    
    def _save_complete_game(self, moves, info):
        """Save complete game log to file."""
        import os
        import time
        
        log_dir = "resources/chess/training_logs/complete_games"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = int(time.time())
        
        # Determine outcome using the new color-specific fields
        outcome = "unknown"
        white_win = info.get('white_win', 0)
        black_win = info.get('black_win', 0)
        game_drawn = info.get('game_drawn', 0)
        
        # Use incremental values to determine what happened this game
        # The C++ code increments these counters when games end
        if white_win > 0:
            outcome = "white_win"
        elif black_win > 0:
            outcome = "black_win"
        elif game_drawn > 0:
            outcome = "draw"
        else:
            # Fallback to legacy fields if new ones aren't available
            if info.get('game_won', 0) > 0:
                outcome = "win"  # Legacy - doesn't specify color
            elif info.get('game_lost', 0) > 0:
                outcome = "loss"  # Legacy - doesn't specify color
            else:
                outcome = "draw"
        
        filename = f"complete_game_{timestamp}_{outcome}.txt"
        filepath = os.path.join(log_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"# Complete chess game logged at {timestamp}\n")
            f.write(f"# Outcome: {outcome}\n")
            f.write(f"# Total moves: {len(moves)}\n")
            f.write(f"# White wins: {white_win}, Black wins: {black_win}, Draws: {game_drawn}\n")
            f.write(f"# Format: Move# Action_ID Algebraic_Notation\n")
            f.write("\n")
            
            for i, (action_id, move) in enumerate(moves):
                f.write(f"{i+1}. {action_id} {move}\n")
        
        return filepath
    
    def _track_move_from_info(self, info):
        """Track move from info dictionary."""
        if not info:
            return
        
        # Process complete game if available
        if 'complete_game_move_count' in info:
            self._process_complete_game(info)
        
        # Get move data (these are floats, not lists)
        last_move_from = info.get('last_move_from', -1)
        last_move_to = info.get('last_move_to', -1)
        last_move_promotion = info.get('last_move_promotion', 0)
        
        # If valid move, add to our tracking
        if last_move_from >= 0 and last_move_to >= 0:
            from_square = self._square_to_notation(int(last_move_from))
            to_square = self._square_to_notation(int(last_move_to))
            
            move_str = f"{from_square}{to_square}"
            
            # Add promotion if applicable
            if last_move_promotion > 0:
                promo_pieces = {1: 'q', 2: 'r', 3: 'b', 4: 'n'}
                move_str += promo_pieces.get(int(last_move_promotion), '')
            
            if self.tracking_game:
                self.game_moves.append(move_str)
    
    def set_fen(self, env_id: int, fen: str):
        binding.vec_set_fen(self.c_envs, fen)
    
    def reset(self, *, seed=None, fen=None):
        if fen is not None:
            self.set_fen(0, fen)
            self.tick = 0
            return self.observations, []
        
        if seed is None:
            seed = 0
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []
    
    def step(self, actions):
        """Step the environment.
        
        Args:
            actions: In self-play mode, array of actions where actions[i*2] is white agent 
                     for game i, and actions[i*2+1] is black agent for game i.
                     In single-agent mode, array of actions for each game.
        """
        # Actions are already in the correct format from PufferLib
        self.actions[:] = actions
        print(f"actions from chess.py: {actions}")
        
        # Step the C++ environments
        binding.vec_step(self.c_envs)
        self.tick += 1
        print(f"tick from chess.py: {self.tick}")
        # Always get info to track moves
        info_dict = binding.vec_log(self.c_envs)
        print(f"info_dict from chess.py: {info_dict}")
        # Track moves if we have info
        if info_dict:
            self._track_move_from_info(info_dict)
            
            # Check for game end (for game logging)
            if self.tracking_game:
                game_won = info_dict.get('game_won', 0)
                game_lost = info_dict.get('game_lost', 0)
                game_drawn = info_dict.get('game_drawn', 0)
                
                if game_won > 0 or game_lost > 0 or game_drawn > 0:
                    # Game ended, stop tracking but don't save (complete game logging handles this)
                    self.tracking_game = False
        
        info = []
        if self.tick % self.log_interval == 0:
            info.append(info_dict)
        
        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)
    
    def render(self):
        import io
        import sys
        from contextlib import redirect_stdout
        
        # Capture stdout from the C++ render function
        f = io.StringIO()
        with redirect_stdout(f):
            binding.vec_render(self.c_envs, 0)
        return f.getvalue()
    
    def close(self):
        binding.vec_close(self.c_envs)


def test_performance(timeout=10, num_envs=1000):
    """Benchmark environment speed."""
    # Test self-play mode
    env = Chess(num_envs=num_envs, self_play=True)
    obs, _ = env.reset()

    # In self-play mode, we have 2 agents per game
    action_cache = np.random.randint(0, env.single_action_space.n, 
                                    (1000, env.num_agents))
    
    import time
    tick = 0
    start = time.time()
    
    while time.time() - start < timeout:
        actions = action_cache[tick % len(action_cache)]
        env.step(actions)
        tick += 1
    
    sps = env.num_agents * tick / (time.time() - start)
    print(f'Self-play SPS: {sps:,}')
    
    env.close()


if __name__ == '__main__':
    test_performance()