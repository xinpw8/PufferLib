#!/usr/bin/env python3
"""Export trained PufferLib model to ONNX for Isaac Sim"""

import torch
import torch.onnx
import numpy as np
import argparse
import os
from pathlib import Path
import json

def export_to_onnx(checkpoint_path, output_dir="onnx_export"):
    """Export PufferLib checkpoint to ONNX format for Isaac Sim
    
    Isaac Sim expects:
    - Model in ONNX format
    - Input: observations (batch_size, obs_dim)
    - Output: actions (batch_size, action_dim) or action logits
    """
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        model_state = checkpoint
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the model architecture (must match PufferLib's Policy)
    from pufferlib.models import Policy
    
    # Environment parameters for drone_pickplace
    obs_dim = 45  # 45 observations
    action_dim = 10  # 10 discrete actions
    hidden_dim = 128  # Default hidden size
    
    # Create model instance
    model = Policy(
        env=None,  # We don't need the actual env for export
        input_size=obs_dim,
        hidden_size=hidden_dim,
        output_size=action_dim
    )
    
    # Load weights
    model.load_state_dict(model_state, strict=False)
    model.eval()
    
    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, obs_dim)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, "drone_policy.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observations'],
        output_names=['action_logits'],
        dynamic_axes={
            'observations': {0: 'batch_size'},
            'action_logits': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {onnx_path}")
    
    # Create config file for Isaac Sim
    config = {
        "model_path": "drone_policy.onnx",
        "observation_space": {
            "shape": [obs_dim],
            "dtype": "float32",
            "description": {
                "drone_state": [0, 14],  # Position, velocity, quaternion, angular_vel, gripper
                "object_state": [14, 21],  # Position, velocity, status
                "target_state": [21, 25],  # Position, has_object
                "task_info": [25, 27]  # Time remaining, progress
            }
        },
        "action_space": {
            "shape": [action_dim],
            "dtype": "int32",
            "type": "discrete",
            "actions": [
                "MOVE_FORWARD",
                "MOVE_BACKWARD", 
                "MOVE_LEFT",
                "MOVE_RIGHT",
                "MOVE_UP",
                "MOVE_DOWN",
                "ROTATE_LEFT",
                "ROTATE_RIGHT",
                "GRIPPER_OPEN",
                "GRIPPER_CLOSE"
            ]
        },
        "normalization": {
            "observations": {
                "mean": [0.0] * obs_dim,
                "std": [1.0] * obs_dim
            }
        }
    }
    
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config saved to {config_path}")
    
    # Test the ONNX model
    import onnx
    import onnxruntime as ort
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")
    
    # Test inference
    ort_session = ort.InferenceSession(onnx_path)
    
    # Run test inference
    test_input = np.random.randn(1, obs_dim).astype(np.float32)
    outputs = ort_session.run(None, {'observations': test_input})
    
    print(f"Test inference successful!")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {outputs[0].shape}")
    print(f"Output (action logits): {outputs[0]}")
    
    return onnx_path, config_path


def create_isaac_sim_wrapper(output_dir="onnx_export"):
    """Create a Python wrapper for Isaac Sim integration"""
    
    wrapper_code = '''"""Isaac Sim integration for drone pick-and-place policy"""

import numpy as np
import onnxruntime as ort
import json
from pathlib import Path

class DronePickPlacePolicy:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        
        # Load config
        with open(self.model_dir / "model_config.json", 'r') as f:
            self.config = json.load(f)
        
        # Load ONNX model
        model_path = self.model_dir / self.config["model_path"]
        self.session = ort.InferenceSession(str(model_path))
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def get_action(self, observation):
        """Get action from observation
        
        Args:
            observation: numpy array of shape (obs_dim,) or (batch_size, obs_dim)
            
        Returns:
            action: discrete action index
        """
        # Handle single observation
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        # Convert to float32
        observation = observation.astype(np.float32)
        
        # Run inference
        logits = self.session.run(
            [self.output_name],
            {self.input_name: observation}
        )[0]
        
        # Get action (argmax for deterministic, or sample for stochastic)
        action = np.argmax(logits, axis=-1)
        
        return action[0] if action.shape[0] == 1 else action
    
    def get_action_name(self, action_idx):
        """Get human-readable action name"""
        return self.config["action_space"]["actions"][action_idx]
'''
    
    wrapper_path = os.path.join(output_dir, "isaac_sim_policy.py")
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_code)
    
    print(f"Isaac Sim wrapper saved to {wrapper_path}")
    return wrapper_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PufferLib model to ONNX")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--output-dir", default="onnx_export", help="Output directory")
    
    args = parser.parse_args()
    
    # Export model
    onnx_path, config_path = export_to_onnx(args.checkpoint, args.output_dir)
    
    # Create Isaac Sim wrapper
    wrapper_path = create_isaac_sim_wrapper(args.output_dir)
    
    print("\n" + "="*60)
    print("Export complete! Files created:")
    print(f"  - ONNX model: {onnx_path}")
    print(f"  - Config: {config_path}")
    print(f"  - Isaac Sim wrapper: {wrapper_path}")
    print("\nTo use in Isaac Sim:")
    print("1. Copy the onnx_export folder to your Isaac Sim project")
    print("2. Import and use the DronePickPlacePolicy class")
    print("="*60)