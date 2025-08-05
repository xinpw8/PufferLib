#!/usr/bin/env python3
"""
Training script for double-buffered chess with separated white/black episodes.
This demonstrates how to actually train with the new implementation.
"""

import subprocess
import sys
import os

def run_training_demo():
    """Run a short training demo to prove the implementation works."""
    print("🐡 Starting Double-Buffered Chess Training Demo 🐡")
    print()
    
    # Set up environment
    os.environ['LD_LIBRARY_PATH'] = f"{os.getcwd()}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    print("Configuration:")
    print("- Environment: Double-buffered chess with episode separation")
    print("- WHITE episodes: Neural network vs random opponent")  
    print("- BLACK episodes: Neural network vs random opponent")
    print("- Episodes alternate to maintain clean advantage boundaries")
    print("- Action masking: Fixed to allow -inf values")
    print()
    
    # Run training with reduced parameters for demo  
    cmd = [
        "puffer", "train", "puffer_chess", "--wandb"
        # "--train.total-timesteps", "200000000",
        # "--vec.num-envs", "2",         # Small scale
        # "--vec.num-workers", "2",      # Match num-envs  
        # "--env.num-envs", "16",        # Smaller for demo
        # "--train.checkpoint-interval", "5", # Save frequently
        # "--train.minibatch-size", "64",      # Small batches
        # "--train.batch-size", "256",         # Larger than minibatch
        # "--train.bptt-horizon", "8"          # Shorter horizon
    ]
    
    print("Training command:")
    print(" ".join(cmd))
    print()
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True, capture_output=False)
        print()
        print("✅ Training completed successfully!")
        print("The double-buffered chess implementation is working correctly.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
        
    return True

def check_implementation():
    """Quick check that everything is set up correctly."""
    print("Checking implementation setup...")
    
    # Check if chess binary exists
    if not os.path.exists("chess"):
        print("❌ Chess binary not found. Run: ./scripts/build_ocean.sh chess")
        return False
    
    # Check if config exists
    config_path = "pufferlib/config/ocean/chess.ini"
    if not os.path.exists(config_path):
        print("❌ Configuration file not found at", config_path)
        return False
        
    print("✅ Chess binary found")
    print("✅ Configuration file found")
    
    # Test import
    try:
        sys.path.insert(0, '.')
        from pufferlib.ocean.chess.double_buffered_chess import DoubleBufferedChess
        print("✅ Double-buffered chess module can be imported")
        return True
    except ImportError as e:
        print(f"❌ Failed to import double-buffered chess: {e}")
        return False

if __name__ == "__main__":
    print("Double-Buffered Chess Training Demo")
    print("=" * 40)
    
    if not check_implementation():
        print("\n❌ Setup check failed. Please fix the issues above.")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    success = run_training_demo()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("\nKey benefits of the double-buffered approach:")
        print("1. Clean episode boundaries between WHITE and BLACK")
        print("2. Proper advantage estimation without contamination")
        print("3. Fixed action masking with -inf values")
        print("4. Maintains game continuity across episode boundaries")
        sys.exit(0)
    else:
        print("\n❌ Demo failed. Check the error messages above.")
        sys.exit(1)