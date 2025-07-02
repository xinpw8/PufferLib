import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any

class ChessSelfPlayWrapper:
    """Wrapper that handles self-play for chess environment"""
    
    def __init__(self, env, policy, device='cuda'):
        self.env = env
        self.policy = policy
        self.device = device
        
        # Enable self-play on the underlying C++ environments.
        from pufferlib.ocean.chess import binding

        if hasattr(env, 'c_envs'):
            # Native (non-vectorised) chess environment
            binding.vec_set_self_play(env.c_envs)
        elif hasattr(env, 'driver_env') and hasattr(env.driver_env, 'c_envs'):
            # Vectorised wrapper (Serial / Multiprocessing / Ray) – enable for the
            # exemplar driver env.  Worker processes will have the flag set when
            # their own Chess instances are constructed with self-play enabled.
            binding.vec_set_self_play(env.driver_env.c_envs)
        else:
            # Nothing to toggle; continue without hard failure so that unit tests
            # for other back-ends (e.g., mocked envs) still run.
            pass
        
        # Track game state
        self.black_turn = False
        self.stored_obs = None
        self.stored_info = None
        
        # For LSTM state
        self.lstm_h = None
        self.lstm_c = None
        if hasattr(policy, 'hidden_size'):
            h = policy.hidden_size
            n = env.num_agents
            self.lstm_h = torch.zeros(n, h, device=device)
            self.lstm_c = torch.zeros(n, h, device=device)
    
    def reset(self, seed=None) -> Tuple[np.ndarray, list]:
        obs, info = self.env.reset(seed)
        self.black_turn = False
        
        # Reset LSTM state
        if self.lstm_h is not None:
            self.lstm_h.zero_()
            self.lstm_c.zero_()
            
        return obs, info
    
    def step(self, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        if self.black_turn:
            # This is black's action being submitted
            obs, rewards, dones, truncs, info = self.env.step(actions)
            self.black_turn = False
            
            # Negate rewards for black (since env returns from white's perspective)
            rewards = -rewards
            
            return obs, rewards, dones, truncs, info
        else:
            # White's turn
            obs, rewards, dones, truncs, info = self.env.step(actions)
            
            # Check if game ended
            if np.any(dones):
                self.black_turn = False  # Reset on game end
                return obs, rewards, dones, truncs, info
            
            # Game continues - now get black's move
            self.black_turn = True
            
            # Get black's action using the policy
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).to(self.device)
                
                # Prepare state for LSTM
                state = {}
                if self.lstm_h is not None:
                    state['lstm_h'] = self.lstm_h
                    state['lstm_c'] = self.lstm_c
                
                # Get policy output
                logits, values = self.policy.forward_eval(obs_tensor, state)
                
                # Sample action
                if isinstance(logits, torch.Tensor):
                    probs = torch.softmax(logits, dim=-1)
                    black_actions = torch.multinomial(probs, 1).squeeze(-1)
                else:
                    # Handle other action space types if needed
                    raise NotImplementedError("Only discrete actions supported")
                
                # Update LSTM state if present
                if 'lstm_h' in state:
                    self.lstm_h = state['lstm_h']
                    self.lstm_c = state['lstm_c']
                
                black_actions = black_actions.cpu().numpy()
            
            # Execute black's move
            obs, black_rewards, dones, truncs, info = self.env.step(black_actions)
            
            # White to play again on next call
            self.black_turn = False
            
            # Return white's rewards (not black's)
            return obs, rewards, dones, truncs, info
    
    def __getattr__(self, name):
        """Forward other attributes to wrapped env"""
        return getattr(self.env, name)