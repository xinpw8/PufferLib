import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any

class ChessSelfPlayWrapper:
    """Wrapper that handles self-play for chess.

    The wrapper now supports *distinct* networks for the white and black
    sides so they can be trained (or frozen) independently.
    """
    
    def __init__(self, env, white_policy, black_policy=None, device='cuda'):
        self.env = env

        # Allow caller to omit black_policy – in that case we fall back to
        # the white network (behaviour identical to the previous version).
        self.white_policy = white_policy
        self.black_policy = black_policy if black_policy is not None else white_policy

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
        
        # Separate (or shared) recurrent state for each policy
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
        obs, info = self.env.reset(seed)
        self.black_turn = False
        
        # Reset both sets of LSTM states (if they exist)
        if self.lstm_h_white is not None:
            self.lstm_h_white.zero_()
            self.lstm_c_white.zero_()

        if self.lstm_h_black is not None:
            self.lstm_h_black.zero_()
            self.lstm_c_black.zero_()
            
        return obs, info
    
    def step(self, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        if self.black_turn:
            # ---------------- BLACK MOVE COMING FROM EXTERNAL ACTOR ----------------
            obs, rewards, dones, truncs, info = self.env.step(actions)
            self.black_turn = False

            # The underlying env gives reward from the POV of the mover
            # (black).  We *do not* negate it here – the caller who holds the
            # black network wants the true value.  For backward-compat we
            # also expose the white-perspective value via info.

            # Store last white-perspective reward for logging
            white_pov_rewards = -rewards
            for i in range(len(info)):
                if isinstance(info[i], dict):
                    info[i]['white_perspective_reward'] = white_pov_rewards[i]
            
            # Save for debug users
            self.last_black_rewards = rewards.copy()

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

                # ---------------- BLACK POLICY ----------------
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

                # Save updated recurrent state (if any)
                if 'lstm_h' in state_b:
                    self.lstm_h_black = state_b['lstm_h']
                    self.lstm_c_black = state_b['lstm_c']
                
                black_actions = black_actions.cpu().numpy()
            
            # Execute black's move (internal)
            obs, black_rewards, dones, truncs, info = self.env.step(black_actions)
            
            # Log black rewards through info so training loops / wandb can
            # record them.  They remain in mover's perspective (black).
            for i in range(len(info)):
                if isinstance(info[i], dict):
                    info[i]['black_reward'] = black_rewards[i]
            
            # White to play again on next call
            self.black_turn = False
            
            # Return white's rewards (not black's)
            return obs, rewards, dones, truncs, info
    
    def __getattr__(self, name):
        """Forward other attributes to wrapped env"""
        return getattr(self.env, name)