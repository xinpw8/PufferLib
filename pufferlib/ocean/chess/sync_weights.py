#!/usr/bin/env python3
import torch
import numpy as np
import time
import os
import shutil

def export_weights_for_cpp(model_path, output_path='chess_weights.bin'):
    """Export PyTorch model weights to binary format for C++"""
    
    # Load the model
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Extract weights in the exact order the C++ code expects
    weights = []
    
    # Board encoder layers
    weights.append(state_dict['board_encoder.0.weight'].numpy().T.flatten())  # Linear transpose for C++
    weights.append(state_dict['board_encoder.0.bias'].numpy())
    weights.append(state_dict['board_encoder.2.weight'].numpy().T.flatten())
    weights.append(state_dict['board_encoder.2.bias'].numpy())
    
    # Combiner layers  
    weights.append(state_dict['combiner.0.weight'].numpy().T.flatten())
    weights.append(state_dict['combiner.0.bias'].numpy())
    
    # Policy head
    weights.append(state_dict['policy_head.weight'].numpy().T.flatten())
    weights.append(state_dict['policy_head.bias'].numpy())
    
    # Value head
    weights.append(state_dict['value_head.0.weight'].numpy().T.flatten())
    weights.append(state_dict['value_head.0.bias'].numpy())
    weights.append(state_dict['value_head.2.weight'].numpy().T.flatten())
    weights.append(state_dict['value_head.2.bias'].numpy())
    
    # Concatenate all weights
    all_weights = np.concatenate(weights).astype(np.float32)
    
    # Write to binary file
    all_weights.tofile(output_path + '.tmp')
    shutil.move(output_path + '.tmp', output_path)  # Atomic update
    
    print(f"Exported {len(all_weights)} weights to {output_path}")

def sync_weights_loop(checkpoint_dir='experiments', interval=60):
    """Continuously sync latest model weights for self-play"""
    
    while True:
        try:
            # Find latest checkpoint
            pattern = os.path.join(checkpoint_dir, '**/model_*.pt')
            import glob
            checkpoints = glob.glob(pattern, recursive=True)
            
            if checkpoints:
                latest = max(checkpoints, key=os.path.getmtime)
                export_weights_for_cpp(latest)
                print(f"Synced weights from {latest}")
                
        except Exception as e:
            print(f"Error syncing weights: {e}")
            
        time.sleep(interval)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        # One-time export
        export_weights_for_cpp(sys.argv[1])
    else:
        # Continuous sync
        sync_weights_loop()