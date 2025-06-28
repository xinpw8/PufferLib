#!/usr/bin/env python3
"""Simple chess evaluation script."""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

import pufferlib.ocean.chess_old.chess as chess_env


# Chess policy architecture (copied from pufferlib/ocean/torch.py)
class ChessRecurrent(nn.Module):
    """Feed-forward encoder/decoder; temporal logic is provided by the common
    pufferlib.models.LSTMWrapper (see ChessRecurrentLSTM below)."""
    def __init__(self, env, input_size=2560, hidden_size=256, **kwargs):
        super().__init__()

        self.hidden_size = hidden_size

        # ----- Encoders -----
        self.board_encoder = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
        )
        self.move_encoder = nn.Sequential(
            nn.Linear(256 + 1536, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
        )

        self.combiner = nn.Sequential(
            nn.Linear(512, hidden_size), nn.ReLU()
        )

        # ----- Heads -----
        self.policy_head = nn.Linear(hidden_size, 256)  # 256 move logits
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        # Continuous/Multidiscrete flags for PufferLib bookkeeping
        self.is_continuous = False

        # Storage for per-step legal-move mask to apply during decode
        self._legal_mask = None

    def encode_observations(self, obs, state=None):
        """Split observation into board state and move information."""
        # obs shape: (batch, 2560) - Enhanced with piece information
        board_state = obs[:, :768]
        legal_moves = obs[:, 768:1024]
        move_encodings = obs[:, 1024:2560]  # Now 1536 elements with piece info
        # Save mask for later use in decode_actions
        self._legal_mask = legal_moves
        
        # Encode board and moves separately
        board_features = self.board_encoder(board_state)
        move_features = self.move_encoder(torch.cat([legal_moves, move_encodings], dim=-1))
        
        # Combine features
        combined = torch.cat([board_features, move_features], dim=-1)
        return self.combiner(combined)
    
    def decode_actions(self, hidden):
        """Return masked logits and value given hidden state."""
        logits = self.policy_head(hidden)
        if self._legal_mask is not None:
            logits = logits.masked_fill(self._legal_mask < 0.5, float('-inf'))
        value = self.value_head(hidden).squeeze(-1)
        return logits, value

    # Convenience forward for direct (non-wrapped) calls
    def forward(self, obs, state=None):
        hidden = self.encode_observations(obs, state)
        logits, value = self.decode_actions(hidden)
        return logits, value


# LSTM wrapper class (copied from pufferlib/ocean/torch.py)
class ChessRecurrentLSTM(nn.Module):
    def __init__(self, env, policy, input_size=256, hidden_size=256):
        super().__init__()
        self.policy = policy
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        
    def forward(self, obs, state=None):
        # Get features from policy
        features = self.policy.encode_observations(obs)
        
        # Handle LSTM state
        if state is None:
            # Initialize LSTM state
            batch_size = features.shape[0]
            h = torch.zeros(batch_size, self.hidden_size, device=features.device)
            c = torch.zeros(batch_size, self.hidden_size, device=features.device)
            state = (h, c)
        
        # Update LSTM state
        h, c = self.cell(features, state)
        
        # Get predictions from policy
        logits, value = self.policy.decode_actions(h)
        
        return logits, value, (h, c)


def evaluate_chess_simple(model_path, num_games=10):
    """Simple chess evaluation without complex config system."""
    
    print(f"🏁 Evaluating chess policy: {model_path}")
    print(f"📊 Running {num_games} games")
    print("=" * 60)
    
    # Create environment with config-matching rewards
    env = chess_env.Chess(
        num_envs=1, 
        render_mode='ansi',
        reward_win=1.0,
        reward_draw=0.0,  # Changed to neutral
        reward_loss=-1.0,
        reward_opponent_capture=-0.001,
        reward_player_capture=0.001,
        reward_move_valid=0,
        reward_move_invalid=0
    )
    
    # Open file to save detailed results
    with open('chess_eval_results.txt', 'w') as f:
        f.write(f"Chess Evaluation Results - {model_path}\n")
        f.write(f"Running {num_games} games\n")
        f.write("=" * 80 + "\n\n")
    
    # FIXED: Proper model loading and initialization with LSTM wrapper
    model = None
    use_trained_policy = False
    
    if model_path and model_path != "random":
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"✅ Loaded model from {model_path}")
            print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
            
            # Initialize base policy
            base_policy = ChessRecurrent(env, input_size=2560, hidden_size=256)
            
            # Initialize LSTM wrapper
            model = ChessRecurrentLSTM(env, base_policy, input_size=256, hidden_size=256)
            
            # Load state dict - handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Assume checkpoint is the state dict itself
                state_dict = checkpoint
            
            # Load weights
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            print(f"✅ Model loaded successfully")
            use_trained_policy = True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("🎲 Using random policy instead")
            use_trained_policy = False
    else:
        print("🎲 Using random policy")
        use_trained_policy = False
    
    # Statistics tracking
    stats = {
        'wins': 0,
        'losses': 0, 
        'draws': 0,
        'total_moves': [],
        'game_results': [],
        'total_time': 0,
        'actual_moves': 0,
        'starting_positions': 0,
        'no_legal_moves_errors': 0
    }
    
    start_time = time.time()
    
    for game_num in range(num_games):
        print(f"\n🎮 Game {game_num + 1}/{num_games}")
        
        # Reset environment
        obs, _ = env.reset()
        done = False
        moves = 0
        game_start = time.time()
        actual_moves_made = 0
        final_board = None
        
        # Initialize LSTM state for new game
        lstm_state = None
        
        # Get initial board state
        initial_board = env.render()
        
        # Debug first observation
        if game_num == 0:
            print(f"📊 Observation shape: {obs.shape}")
            print(f"📊 Observation size: {obs.size}")
            print(f"📊 Enhanced observation with piece information: 2560 features")
            print(f"📊 Layout: 768 board + 256 legal moves + 1536 move encodings (with piece info)")
            # Check legal moves location
            board_end = 768
            mask_start = 768
            mask_end = 1024
            legal_mask = obs[0][mask_start:mask_end] if obs.ndim > 1 else obs[mask_start:mask_end]
            print(f"📊 Legal moves found: {np.sum(legal_mask > 0.5)}")
            
            # Show piece information from first legal move if available
            if np.sum(legal_mask > 0.5) > 0:
                move_start = 1024
                first_move_idx = np.where(legal_mask > 0.5)[0][0]
                move_offset = move_start + first_move_idx * 6  # 6 coordinates per move
                if obs.ndim > 1:
                    move_data = obs[0][move_offset:move_offset+6]
                else:
                    move_data = obs[move_offset:move_offset+6]
                print(f"📊 First move encoding: from({move_data[0]*7:.1f},{move_data[1]*7:.1f}) to({move_data[2]*7:.1f},{move_data[3]*7:.1f}) piece_type:{move_data[4]:.0f} color:{move_data[5]:.0f}")
        
        while not done and moves < 400:  # Max 400 moves per game (MAX_PLIES in C++)
            moves += 1
            
            # FIXED: Get legal moves from correct position in observation
            # Observation layout: [768 board features][256 legal moves mask][1536 move encodings]
            if obs.ndim > 1:
                legal_mask = obs[0][768:1024]  # Legal moves are at indices 768-1023
            else:
                legal_mask = obs[768:1024]
                
            valid_actions = np.where(legal_mask > 0.5)[0]
            
            if len(valid_actions) == 0:
                print(f"⚠️  No legal moves available at move {moves}")
                stats['no_legal_moves_errors'] += 1
                # This shouldn't happen - chess always has legal moves unless game is over
                done = True
                break
            
            # FIXED: Use actual trained model for action selection with LSTM state
            if use_trained_policy and model is not None:
                with torch.no_grad():
                    # Convert observation to tensor (already batched)
                    obs_tensor = torch.FloatTensor(obs)  # Remove .unsqueeze(0)
                    
                    # Get model predictions with LSTM state
                    logits, value, lstm_state = model(obs_tensor, lstm_state)
                    
                    # Apply legal move mask
                    legal_mask_tensor = torch.FloatTensor(legal_mask).unsqueeze(0)
                    masked_logits = logits.masked_fill(legal_mask_tensor < 0.5, float('-inf'))
                    
                    # Sample action from policy
                    probs = F.softmax(masked_logits, dim=-1)
                    action_dist = torch.distributions.Categorical(probs)
                    action = action_dist.sample().item()
                    
                    # Ensure action is valid
                    if action not in valid_actions:
                        print(f"⚠️  Model selected invalid action {action}, using random")
                        action = np.random.choice(valid_actions)
            else:
                # Random policy
                action = np.random.choice(valid_actions)
            
            # Step environment
            prev_obs = obs.copy()
            obs, rewards, terminals, truncations, info = env.step([action])
            
            # Capture final board position BEFORE checking if done
            if terminals[0] or truncations[0]:
                final_board = env.render()
            
            done = terminals[0] or truncations[0]
            
            # Check if the board actually changed
            if not np.array_equal(prev_obs, obs):
                actual_moves_made += 1
        
        # Game finished - record results
        game_time = time.time() - game_start
        stats['total_time'] += game_time
        stats['total_moves'].append(moves)
        stats['actual_moves'] += actual_moves_made
        
        # Get final board state (if not already captured)
        if final_board is None:
            final_board = env.render()
        
        # Check if final position is same as starting position
        if initial_board in final_board and "8 K Q R B N P" in final_board:
            stats['starting_positions'] += 1
            print(f"🚨 WARNING: Final position might be same as starting position!")
        
        # Determine result based on last reward
        if 'rewards' not in locals():
            # No moves were made
            result = "ERROR"
            stats['losses'] += 1
        else:
            reward = rewards[0]
            if reward > 0.5:  # Win
                result = "WIN"
                stats['wins'] += 1
            elif reward < -0.5:  # Loss  
                result = "LOSS"
                stats['losses'] += 1
            else:  # Draw
                result = "DRAW"
                stats['draws'] += 1
            
        stats['game_results'].append(result)
        print(f"🏆 Result: {result} ({moves} moves, {actual_moves_made} actual, {game_time:.1f}s)")
        
        # Save detailed results to file
        with open('chess_eval_results.txt', 'a') as f:
            f.write(f"Game {game_num + 1}: {result} ({moves} moves, {actual_moves_made} actual)\n")
            f.write("Initial position:\n")
            f.write(initial_board + "\n\n")
            f.write("Final position:\n") 
            f.write(final_board + "\n")
            f.write("-" * 80 + "\n\n")
        
        # Show board for first few games or suspicious games
        if game_num < 3 or result == "ERROR":
            print("📋 Final position:")
            print(final_board if len(final_board) < 500 else final_board[:500] + "...")
    
    # Calculate and display statistics
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("📈 EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"🎯 Games Played: {num_games}")
    print(f"🏆 Wins: {stats['wins']} ({stats['wins']/num_games*100:.1f}%)")
    print(f"💀 Losses: {stats['losses']} ({stats['losses']/num_games*100:.1f}%)")  
    print(f"🤝 Draws: {stats['draws']} ({stats['draws']/num_games*100:.1f}%)")
    
    print(f"\n📊 Game Statistics:")
    if stats['total_moves']:
        print(f"   Average moves per game: {np.mean(stats['total_moves']):.1f}")
        print(f"   Shortest game: {min(stats['total_moves'])} moves")
        print(f"   Longest game: {max(stats['total_moves'])} moves")
    print(f"   Total actual moves made: {stats['actual_moves']}")
    print(f"   Average actual moves per game: {stats['actual_moves']/num_games:.1f}")
    
    print(f"\n🚨 Suspicious Results:")
    print(f"   Games ending at starting position: {stats['starting_positions']} ({stats['starting_positions']/num_games*100:.1f}%)")
    print(f"   No legal moves errors: {stats['no_legal_moves_errors']}")
    
    print(f"\n⏱️  Timing:")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Average time per game: {stats['total_time']/num_games:.1f}s")
    print(f"   Games per second: {num_games/total_time:.2f}")
    
    # Show recent game results
    print(f"\n📋 Game results: {' '.join(stats['game_results'])}")
    print(f"\n💾 Detailed results saved to: chess_eval_results.txt")
    
    print("\n" + "=" * 60)
    
    return stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_chess_eval.py <model_path_or_'random'> [num_games]")
        print("Example: python simple_chess_eval.py experiments/model.pt 20")
        print("Example: python simple_chess_eval.py random 10")
        sys.exit(1)
    
    model_path = sys.argv[1]
    num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    evaluate_chess_simple(model_path, num_games)