import numpy as np
import torch
import pufferlib

# First, let's verify the environment works correctly
def test_env_basic():
    """Test basic environment functionality"""
    print("=== Testing Basic Environment ===")
    from pufferlib.ocean.chess.chess import Chess
    
    env = Chess(num_envs=1)
    print(f"Observation space: {env.single_observation_space}")
    print(f"Observation shape: {env.single_observation_space.shape}")
    print(f"Action space: {env.single_action_space}")
    
    obs, info = env.reset(seed=0)
    print(f"Initial obs shape: {obs.shape}")
    print(f"Initial obs dtype: {obs.dtype}")
    print(f"Board features sum: {obs[0, :1344].sum()}")
    print(f"Legal mask sum: {obs[0, 1344:6018].sum()}")
    
    # Check if legal mask has any valid moves
    legal_moves = obs[0, 1344:6018]
    print(f"Number of legal moves: {legal_moves.sum()}")
    
    # Take a random legal action
    legal_indices = np.where(legal_moves > 0.5)[0]
    if len(legal_indices) > 0:
        action = np.array([legal_indices[0]], dtype=np.int32)
        print(f"Taking action: {action[0]}")
        
        obs, rewards, terminals, truncations, info = env.step(action)
        print(f"After step - Reward: {rewards[0]}, Terminal: {terminals[0]}")
        print(f"New legal moves: {obs[0, 1344:6018].sum()}")
    else:
        print("ERROR: No legal moves found!")
    
    env.close()

# Test the policy network
def test_policy():
    """Test the policy network"""
    print("\n=== Testing Policy Network ===")
    from pufferlib.ocean.chess.chess import Chess
    from pufferlib.ocean.torch import ChessRecurrent
    
    env = Chess(num_envs=4)
    policy = ChessRecurrent(env=env)
    
    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs)
    
    print(f"Input shape: {obs_tensor.shape}")
    
    # Test forward pass
    with torch.no_grad():
        try:
            logits, values = policy.forward(obs_tensor)
            print(f"Logits shape: {logits.shape}")
            print(f"Values shape: {values.shape}")
            print(f"Logits min/max: {logits.min().item():.2f} / {logits.max().item():.2f}")
            print(f"Any NaN in logits: {torch.isnan(logits).any()}")
            print(f"Any Inf in logits: {torch.isinf(logits).any()}")
            
            # Check if masking is working
            legal_mask = obs_tensor[:, 1344:6018]
            masked_positions = (legal_mask < 0.5)
            if masked_positions.any():
                masked_logits = logits[masked_positions[:logits.shape[0]]]
                print(f"Masked logits min/max: {masked_logits.min().item():.2e} / {masked_logits.max().item():.2e}")
        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            import traceback
            traceback.print_exc()
    
    env.close()

# Test with LSTM wrapper
def test_lstm_wrapper():
    """Test LSTM wrapper"""
    print("\n=== Testing LSTM Wrapper ===")
    from pufferlib.ocean.chess.chess import Chess
    from pufferlib.ocean.torch import ChessRecurrent
    from pufferlib.models import LSTMWrapper
    
    env = Chess(num_envs=4)
    base_policy = ChessRecurrent(env=env)
    policy = LSTMWrapper(env, base_policy, input_size=256, hidden_size=256)
    
    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs)
    
    # Initialize state
    state = {
        'lstm_h': None,
        'lstm_c': None
    }
    
    print(f"Input shape: {obs_tensor.shape}")
    
    with torch.no_grad():
        try:
            logits, values = policy.forward(obs_tensor, state)
            print(f"LSTM Logits shape: {logits.shape}")
            print(f"LSTM Values shape: {values.shape}")
            print(f"LSTM Logits min/max: {logits.min().item():.2f} / {logits.max().item():.2f}")
            print(f"Any NaN in LSTM logits: {torch.isnan(logits).any()}")
            print(f"Any Inf in LSTM logits: {torch.isinf(logits).any()}")
        except Exception as e:
            print(f"ERROR in LSTM forward pass: {e}")
            import traceback
            traceback.print_exc()
    
    env.close()

# Run all tests
if __name__ == "__main__":
    test_env_basic()
    test_policy()
    test_lstm_wrapper()  # Uncomment if you have the LSTM wrapper accessible