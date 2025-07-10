import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any

class ChessSelfPlayWrapper:
    """Wrapper that handles self-play for chess with proper reward tracking for both colors.
    
    The C++ environment in self-play mode expects actions for both white and black moves,
    but tracks which player is moving internally. This wrapper provides actions for both
    sides and tracks statistics properly.
    """
    
    def __init__(self, env, white_policy, black_policy=None, device='cuda'):
        self.env = env
        self.white_policy = white_policy
        self.black_policy = black_policy if black_policy is not None else white_policy
        self.device = device
        
        # Enable self-play on the underlying C++ environments
        from pufferlib.ocean.chess import binding
        if hasattr(env, 'c_envs'):
            binding.vec_set_self_play(env.c_envs)
        elif hasattr(env, 'driver_env') and hasattr(env.driver_env, 'c_envs'):
            binding.vec_set_self_play(env.driver_env.c_envs)
        
        # Track move counts and outcomes
        self.move_count = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.total_games = 0
        
        # Track rewards by color
        self.white_rewards = []
        self.black_rewards = []
        
        # Separate recurrent states for each policy
        self.lstm_h_white = None
        self.lstm_c_white = None
        self.lstm_h_black = None
        self.lstm_c_black = None

        if hasattr(self.white_policy, 'hidden_size'):
            h = self.white_policy.hidden_size
            n = env.num_agents
            self.lstm_h_white = torch.zeros(n, h, device=device)
            self.lstm_c_white = torch.zeros(n, h, device=device)

        if hasattr(self.black_policy, 'hidden_size'):
            h = self.black_policy.hidden_size
            n = env.num_agents
            self.lstm_h_black = torch.zeros(n, h, device=device)
            self.lstm_c_black = torch.zeros(n, h, device=device)
    
    def reset(self, seed=None) -> Tuple[np.ndarray, list]:
        obs, info = self.env.reset(seed=seed)
        self.move_count = 0
        
        # Reset LSTM states
        if self.lstm_h_white is not None:
            self.lstm_h_white.zero_()
            self.lstm_c_white.zero_()
        if self.lstm_h_black is not None:
            self.lstm_h_black.zero_()
            self.lstm_c_black.zero_()
            
        return obs, info
    
    def step(self, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        # The training loop provides actions, but we need to determine if this is for white or black
        # based on the move count (even = white, odd = black)
        is_white_move = (self.move_count % 2) == 0
        current_player = 'white' if is_white_move else 'black'
        
        # If this is a black move and we have a separate black policy, use it
        if not is_white_move and self.black_policy != self.white_policy:
            # Generate black action using black policy
            with torch.no_grad():
                obs_tensor = torch.from_numpy(self.env.observations).to(self.device)

                state_b = {}
                if self.lstm_h_black is not None:
                    state_b['lstm_h'] = self.lstm_h_black
                    state_b['lstm_c'] = self.lstm_c_black

                logits, _ = self.black_policy.forward_eval(obs_tensor, state_b)

                if isinstance(logits, torch.Tensor):
                    probs = torch.softmax(logits, dim=-1)
                    black_actions = torch.multinomial(probs, 1).squeeze(-1)
                else:
                    raise NotImplementedError("Only discrete actions supported")

                # Save updated recurrent state
                if 'lstm_h' in state_b:
                    self.lstm_h_black = state_b['lstm_h']
                    self.lstm_c_black = state_b['lstm_c']
                
                actions = black_actions.cpu().numpy()
        
        # Execute the move
        obs, rewards, dones, truncs, info = self.env.step(actions)
        
        # Track rewards by color
        if is_white_move:
            self.white_rewards.append(rewards[0])
        else:
            self.black_rewards.append(rewards[0])
        
        # Increment move count
        self.move_count += 1
        
        # Add color-specific information to info
        for i in range(len(info)):
            if isinstance(info[i], dict):
                info[i]['current_player'] = current_player
                info[i]['mover_reward'] = rewards[i]
                info[i]['move_number'] = self.move_count
                info[i]['white_move_count'] = len(self.white_rewards)
                info[i]['black_move_count'] = len(self.black_rewards)
                
                if len(self.white_rewards) > 0:
                    info[i]['white_avg_reward'] = np.mean(self.white_rewards)
                    info[i]['white_total_reward'] = np.sum(self.white_rewards)
                
                if len(self.black_rewards) > 0:
                    info[i]['black_avg_reward'] = np.mean(self.black_rewards)
                    info[i]['black_total_reward'] = np.sum(self.black_rewards)
                
                # Track game outcomes
                if dones[i]:
                    self.total_games += 1
                    
                    # Use actual game outcome information from C++ environment
                    # instead of reward thresholds which are unreliable
                    white_win = info[i].get('white_win', 0)
                    black_win = info[i].get('black_win', 0)
                    game_drawn = info[i].get('game_drawn', 0)
                    
                    # Determine outcome based on incremental counters from C++
                    if white_win > 0:
                        self.white_wins += 1
                        info[i]['game_outcome'] = 'white_win'
                    elif black_win > 0:
                        self.black_wins += 1
                        info[i]['game_outcome'] = 'black_win'
                    elif game_drawn > 0:
                        self.draws += 1
                        info[i]['game_outcome'] = 'draw'
                    else:
                        # Fallback: if no clear outcome, assume draw
                        self.draws += 1
                        info[i]['game_outcome'] = 'draw'
                    
                    # Add win rate statistics
                    if self.total_games > 0:
                        info[i]['white_win_rate'] = self.white_wins / self.total_games
                        info[i]['black_win_rate'] = self.black_wins / self.total_games
                        info[i]['draw_rate'] = self.draws / self.total_games
                        info[i]['total_games'] = self.total_games
                    
                    # Reset for next game
                    self.move_count = 0
                    self.white_rewards = []
                    self.black_rewards = []

        return obs, rewards, dones, truncs, info
    
    def __getattr__(self, name):
        """Forward other attributes to wrapped env"""
        return getattr(self.env, name)