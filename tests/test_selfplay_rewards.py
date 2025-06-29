#!/usr/bin/env python3
"""Test script to verify self-play reward consistency"""

import torch
import numpy as np
from pufferlib.ocean.environment import make_chess_selfplay

def test_selfplay_rewards():
    """Test that rewards are correctly handled in self-play mode"""
    
    # Create self-play environment
    env, policy = make_chess_selfplay(num_envs=1, device='cpu')
    
    # Reset environment
    obs, info = env.reset()
    
    total_white_reward = 0
    total_black_reward = 0
    moves = 0
    
    # Play a few moves to test reward handling
    for step in range(10):
        # Check if game is done
        if hasattr(env, 'terminals') and env.terminals[0]:
            break
            
        # Get valid actions (just use first legal move for testing)
        legal_mask = torch.from_numpy(obs[0, 1344:6018])  # Legal move mask
        legal_actions = torch.where(legal_mask > 0.5)[0]
        
        if len(legal_actions) == 0:
            print("No legal moves available")
            break
            
        action = legal_actions[0].item()
        
        # Take step
        obs, rewards, dones, truncs, info = env.step(np.array([action]))
        
        # Track rewards by perspective
        if env.black_turn:
            total_black_reward += rewards[0]
            print(f"Step {step}: Black move, reward = {rewards[0]:.3f}")
        else:
            total_white_reward += rewards[0]
            print(f"Step {step}: White move, reward = {rewards[0]:.3f}")
            
        moves += 1
        
        if dones[0]:
            print(f"Game ended after {moves} moves")
            break
    
    print(f"\nTotal white reward: {total_white_reward:.3f}")
    print(f"Total black reward: {total_black_reward:.3f}")
    print(f"Combined reward: {total_white_reward + total_black_reward:.3f}")
    
    # In a zero-sum game, combined rewards should be close to 0
    # (except for the final outcome)
    
    env.close()

if __name__ == "__main__":
    test_selfplay_rewards() 