#!/usr/bin/env python3
"""
Config-based sweep script for chess training.
Generates separate config files for each run to avoid parameter conflicts.
"""

import subprocess
import sys
import os
import random
import time
import json
import shutil
import tempfile
import configparser
from pathlib import Path

class ChessSweep:
    def __init__(self, base_config_path=None):
        self.base_config_path = base_config_path or "pufferlib/config/ocean/double_buffered_chess.ini"
        self.sweep_dir = Path("sweep_configs")
        self.sweep_dir.mkdir(exist_ok=True)
        
        # Initialize Protein sweep algorithm
        base_config = self.load_base_config()
        if 'sweep' in base_config.sections():
            # Use the Protein sweep from config
            import pufferlib.sweep as sweep_module
            sweep_config = dict(base_config['sweep'])
            
            # Add sweep parameter definitions and convert strings to numbers
            for section in base_config.sections():
                if section.startswith('sweep.'):
                    param_dict = {}
                    for key, value in base_config[section].items():
                        # Clean up the value - remove comments 
                        clean_value = value.split('#')[0].strip()
                        
                        # For numeric values, remove underscores (like 600_000)
                        # But keep underscores in string values (like uniform_pow2)
                        if key in ['min', 'max', 'mean'] and '_' in clean_value:
                            try:
                                # Try to parse as number with underscores
                                float(clean_value.replace('_', ''))
                                clean_value = clean_value.replace('_', '')
                            except ValueError:
                                pass  # Keep underscores for non-numeric values
                        
                        # Try to convert to appropriate type
                        try:
                            if clean_value == 'auto':
                                param_dict[key] = clean_value  # Keep 'auto' as string
                            elif '.' in clean_value:
                                param_dict[key] = float(clean_value)
                            else:
                                param_dict[key] = int(clean_value)
                        except ValueError:
                            param_dict[key] = clean_value  # Keep as string
                    sweep_config[section[6:]] = param_dict
            
            self.protein = sweep_module.Protein(sweep_config)
            self.use_protein = True
            print("🧬 Using Protein sweep algorithm")
        else:
            self.protein = None
            self.use_protein = False
            print("⚠️ No sweep config found, using random sampling")
        
    def load_base_config(self):
        """Load the base configuration file"""
        config = configparser.ConfigParser()
        config.read(self.base_config_path)
        return config
        
    def generate_sweep_params(self, run_id=None):
        """Generate parameter variations using Protein or fallback to random"""
        if self.use_protein:
            # For first few runs, add some random exploration to help Protein learn
            if run_id is not None and run_id <= 3:
                # Get Protein suggestion but add some randomness
                suggestion_dict, info = self.protein.suggest(fill=None)
                
                # Add random perturbation to learning rate for first few runs
                if 'train.learning_rate' in suggestion_dict:
                    base_lr = suggestion_dict['train.learning_rate']
                    # Randomly vary learning rate by ±50%
                    import random
                    multiplier = random.uniform(0.5, 1.5)
                    suggestion_dict['train.learning_rate'] = base_lr * multiplier
                    
                # Add random perturbation to reward values
                reward_params = [k for k in suggestion_dict.keys() if 'reward_' in k]
                for param in reward_params:
                    base_val = suggestion_dict[param]
                    # Randomly vary by ±30%
                    multiplier = random.uniform(0.7, 1.3)
                    suggestion_dict[param] = base_val * multiplier
                    
                print(f"[RUN {run_id}] Added random exploration to Protein suggestions")
            else:
                # Use pure Protein suggestions after first few runs
                suggestion_dict, info = self.protein.suggest(fill=None)
            
            # Convert suggestion to our parameter format
            params = {}
            for key, value in suggestion_dict.items():
                # Convert dot notation (train.learning_rate) to underscore (train_learning_rate)
                param_key = key.replace('.', '_')
                params[param_key] = value
                
            return params
        else:
            # Fallback to known working configurations
            base_variations = [
                {
                    'env_num_envs': 512,
                    'train_batch_size': 32768,
                    'train_bptt_horizon': 16,
                    'train_minibatch_size': 4096,
                    'train_learning_rate': 0.00696969,
                    'env_reward_white_captures_enemy_piece': 0.5,
                    'env_reward_black_captures_enemy_piece': 0.5,
                },
                {
                    'env_num_envs': 256,
                    'train_batch_size': 16384,
                    'train_bptt_horizon': 16,
                    'train_minibatch_size': 2048,
                    'train_learning_rate': 0.00696969,
                    'env_reward_white_captures_enemy_piece': 0.3,
                    'env_reward_black_captures_enemy_piece': 0.3,
                },
            ]
            return random.choice(base_variations)
    
    def create_sweep_config(self, params, run_id):
        """Create a new config file with modified parameters"""
        config = self.load_base_config()
        
        # Apply parameter changes
        for param_name, value in params.items():
            section, key = param_name.split('_', 1)
            if section in config:
                config[section][key] = str(value)
                
                # Special handling for env.num_envs - also update vec.num_envs  
                if section == 'env' and key == 'num_envs':
                    # The working config has vec.num_envs = vec.num_workers = 4
                    # This suggests vec.num_envs should equal num_workers for multiprocessing
                    num_workers = int(config['vec']['num_workers'])
                    config['vec']['num_envs'] = str(num_workers)
                    
                    # Check that env.num_envs is reasonable for the number of workers
                    envs_per_worker = int(value) // num_workers
                    print(f"[RUN {run_id}] Auto-set vec.num_envs = {num_workers} (env.num_envs = {value}, {envs_per_worker} envs/worker)")
                    
            else:
                print(f"[RUN {run_id}] WARNING: Section '{section}' not found for param {param_name}")
        
        # Set shorter timesteps for sweep testing
        # Need at least batch_size timesteps to avoid division by zero
        config['train']['total_timesteps'] = '65536'
        
        # Create unique config file for this run
        config_filename = f"sweep_run_{run_id:03d}.ini"
        config_path = self.sweep_dir / config_filename
        
        with open(config_path, 'w') as f:
            config.write(f)
            
        return config_path
    
    def validate_config(self, config_path):
        """Validate that the generated config has consistent parameters"""
        config = configparser.ConfigParser()
        config.read(config_path)
        
        try:
            env_num_envs = int(config['env']['num_envs'])
            vec_num_envs = int(config['vec']['num_envs']) 
            batch_size = int(config['train']['batch_size'])
            bptt_horizon = int(config['train']['bptt_horizon'])
            minibatch_size = int(config['train']['minibatch_size'])
            
            total_agents = env_num_envs * 2  # Chess has 2 agents per env
            segments = batch_size // bptt_horizon
            
            # Check critical parameter relationships
            if total_agents > segments:
                return False, f"Total agents {total_agents} > segments {segments}"
                
            if minibatch_size > batch_size:
                return False, f"Minibatch size {minibatch_size} > batch size {batch_size}"
            
            # Check vec/env num_envs relationship - vec should be reasonable for env count
            if vec_num_envs > env_num_envs:
                return False, f"vec.num_envs ({vec_num_envs}) > env.num_envs ({env_num_envs})"
            
            if env_num_envs % vec_num_envs != 0:
                return False, f"env.num_envs ({env_num_envs}) not divisible by vec.num_envs ({vec_num_envs})"
                
            return True, f"Valid: {env_num_envs} envs, {vec_num_envs} vec_envs, {total_agents} agents, {segments} segments"
            
        except Exception as e:
            return False, f"Config validation error: {e}"
    
    def run_training_with_config(self, config_path, run_id, timeout_seconds=300, hang_timeout=120):
        """Run training with a specific config file"""
        print(f"[RUN {run_id}] Starting training with config: {config_path}")
        
        # Show key parameters from config
        config = configparser.ConfigParser()
        config.read(config_path)
        print(f"[RUN {run_id}] Key params:")
        print(f"  num_envs: {config['env']['num_envs']}")
        print(f"  batch_size: {config['train']['batch_size']}")
        print(f"  bptt_horizon: {config['train']['bptt_horizon']}")
        print(f"  learning_rate: {config['train']['learning_rate']}")
        
        # Strategy: Temporarily replace the main config file, run training, then restore
        original_config = self.base_config_path
        backup_config = f"{original_config}.backup_{run_id}"
        
        start_time = time.time()
        
        try:
            # Backup original config
            shutil.copy2(original_config, backup_config)
            
            # Replace with our sweep config
            shutil.copy2(config_path, original_config)
            
            # Build command - now puffer will use our modified config
            # Use a separate wandb project for protein sweeps to make them easily findable
            cmd = [
                "puffer", "train", "puffer_double_buffered_chess",
                "--wandb",
                "--wandb-project", "protein-chess-sweep",
                "--tag", f"run-{run_id:03d}",
            ]
            
            # Run with smart progress detection instead of just hard timeout
            log_file_path = getattr(self, 'log_file_path', 'sweep_debug.log')
            
            import subprocess
            import threading
            import queue
            
            def monitor_progress(process, log_file, hang_timeout):
                """Monitor training progress and detect hangs"""
                last_progress_time = time.time()
                progress_indicators = [
                    "Step ", "Epoch ", "SPS ", "Train()", "Evaluate()", 
                    "completed", "about to call", "Policy loss"
                ]
                
                while process.poll() is None:
                    time.sleep(1)
                    current_time = time.time()
                    
                    # Check if we've seen progress recently by reading the log
                    try:
                        with open(log_file_path, 'r') as f:
                            recent_lines = f.readlines()[-10:]  # Last 10 lines
                            recent_text = ''.join(recent_lines)
                            
                            # Check for progress indicators
                            if any(indicator in recent_text for indicator in progress_indicators):
                                last_progress_time = current_time
                                
                    except Exception:
                        pass  # Log file might not exist yet
                    
                    # Check for hang
                    time_since_progress = current_time - last_progress_time
                    if time_since_progress > hang_timeout:
                        print(f"[RUN {run_id}] 🚫 No progress for {hang_timeout}s, terminating...")
                        process.terminate()
                        time.sleep(2)
                        if process.poll() is None:
                            process.kill()
                        return "hang_detected"
                
                return "completed"
            
            # Start the training process
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\n=== RUN {run_id} START ===\n")
                log_file.write(f"Command: {' '.join(cmd)}\n")
                log_file.write(f"Config: {config_path}\n")
                log_file.write(f"Hang timeout: {hang_timeout}s, Hard timeout: {timeout_seconds}s\n")
                log_file.flush()
                
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=os.getcwd()
                )
                
                # Start progress monitoring in a separate thread
                monitor_thread = threading.Thread(
                    target=monitor_progress, 
                    args=(process, log_file, hang_timeout)
                )
                monitor_thread.daemon = True
                monitor_thread.start()
                
                # Wait for completion with hard timeout
                try:
                    result = process.wait(timeout=timeout_seconds)
                    monitor_thread.join(timeout=1)  # Give monitor thread a moment to clean up
                    
                except subprocess.TimeoutExpired:
                    print(f"[RUN {run_id}] ⏰ Hard timeout after {timeout_seconds}s")
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
                    result = -1  # Indicate timeout
            
            elapsed = time.time() - start_time
            
            if result == 0:
                print(f"[RUN {run_id}] ✅ SUCCESS in {elapsed:.1f}s")
                success_result = (True, elapsed, "", "")
            elif result == -1:
                print(f"[RUN {run_id}] ⏰ TIMEOUT in {elapsed:.1f}s")
                success_result = (False, elapsed, "", "Hard timeout exceeded")
            else:
                print(f"[RUN {run_id}] ❌ FAILED in {elapsed:.1f}s (exit code {result})")
                success_result = (False, elapsed, "", f"Process exited with code {result}")
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[RUN {run_id}] 💥 ERROR in {elapsed:.1f}s: {e}")
            success_result = (False, elapsed, "", str(e))
            
        finally:
            # Always restore the original config file
            try:
                if os.path.exists(backup_config):
                    shutil.copy2(backup_config, original_config)
                    os.remove(backup_config)
                    print(f"[RUN {run_id}] 🔄 Restored original config")
            except Exception as restore_error:
                print(f"[RUN {run_id}] ⚠️ Failed to restore config: {restore_error}")
        
        return success_result
    
    def run_sweep(self, num_runs=5, timeout_per_run=300, hang_timeout=120, log_file='sweep_debug.log'):
        """Run a parameter sweep using separate config files"""
        self.log_file_path = log_file
        print(f"🚀 Starting config-based chess parameter sweep")
        print(f"📊 Runs: {num_runs}, Hard timeout: {timeout_per_run}s, Hang timeout: {hang_timeout}s")
        print(f"📁 Config directory: {self.sweep_dir}")
        print(f"📄 Logging to: {log_file}")
        print("=" * 60)
        
        # Set up environment
        os.environ['LD_LIBRARY_PATH'] = f"{os.getcwd()}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        
        results = []
        successful_runs = 0
        
        for run_id in range(1, num_runs + 1):
            print(f"\n📋 Generating config for run {run_id}/{num_runs}")
            
            # Generate parameters and create config
            params = self.generate_sweep_params(run_id)
            config_path = self.create_sweep_config(params, run_id)
            
            # Validate the generated config
            is_valid, validation_msg = self.validate_config(config_path)
            if not is_valid:
                print(f"[RUN {run_id}] ❌ Invalid config: {validation_msg}")
                continue
            
            print(f"[RUN {run_id}] ✅ Generated valid config: {validation_msg}")
            
            # Run training
            success, elapsed, stdout, stderr = self.run_training_with_config(
                config_path, run_id, timeout_per_run, hang_timeout
            )
            
            # --- START NEW CODE ---
            # After a run completes, observe the results to make the Protein sweep intelligent
            # Note: We try to extract score even if run timed out or failed, as partial results are still useful
            if self.use_protein:
                try:
                    import re

                    # Read the entire log file to find the relevant section for this run
                    with open(self.log_file_path, 'r') as f:
                        log_content = f.read()

                    # Isolate the log text for the specific run to avoid parsing old data
                    run_start_marker = f"=== RUN {run_id} START ==="
                    next_run_start_marker = f"=== RUN {run_id + 1} START ==="
                    
                    start_index = log_content.find(run_start_marker)
                    end_index = log_content.find(next_run_start_marker, start_index)
                    
                    if start_index == -1:
                        raise ValueError(f"Could not find start marker for run {run_id} in log.")

                    run_log = log_content[start_index:end_index] if end_index != -1 else log_content[start_index:]

                    # Use a regular expression to robustly find the last episode_return
                    # This looks for "episode_return" followed by whitespace and a number
                    matches = re.findall(r"environment/episode_return\s+([\d\.\-]+)", run_log)
                    
                    if not matches:
                        raise ValueError("Could not find 'environment/episode_return' in log for this run.")

                    # The last match is the final score reported
                    score = float(matches[-1])
                    print(f"[RUN {run_id}] 🧠 Parsed final score for Protein: {score}")

                    # Feed the result back to the Protein algorithm
                    # IMPORTANT: Convert params back to dot notation for Protein.observe()
                    # The params dict uses underscore notation (train_learning_rate) but
                    # Protein expects dot notation (train.learning_rate)
                    dot_params = {}
                    for key, value in params.items():
                        # Convert underscore notation back to dot notation
                        if '_' in key:
                            section, param_name = key.split('_', 1)
                            dot_key = f"{section}.{param_name}"
                            dot_params[dot_key] = value
                        else:
                            dot_params[key] = value
                    
                    # Now pass the dot-notation parameters to observe()
                    self.protein.observe(dot_params, score, elapsed)
                    print(f"[RUN {run_id}] ✅ Successfully fed results to Protein sweep algorithm")
                    
                    # Store the score for ranking
                    protein_score = score

                except Exception as e:
                    print(f"[RUN {run_id}] ⚠️ WARNING: Could not parse score for Protein.observe(). The sweep will not learn from this run. Error: {e}")
                    protein_score = None
            else:
                protein_score = None
            # --- END NEW CODE ---
            
            results.append({
                'run_id': run_id,
                'config_path': str(config_path),
                'params': params,
                'success': success,
                'elapsed_time': elapsed,
                'stdout': stdout,
                'stderr': stderr,
                'protein_score': protein_score
            })
            
            if success:
                successful_runs += 1
            
            print(f"[RUN {run_id}] Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
            print(f"📈 Progress: {run_id}/{num_runs} runs completed, {successful_runs} successful")
            
            # Brief pause between runs
            if run_id < num_runs:
                print("😴 Waiting 3 seconds before next run...")
                time.sleep(3)
        
        # Summary
        print("\n" + "=" * 60)
        print(f"🏁 CONFIG SWEEP COMPLETE!")
        print(f"📊 Total runs: {num_runs}")
        print(f"✅ Successful: {successful_runs}")
        print(f"❌ Failed: {num_runs - successful_runs}")
        print(f"📈 Success rate: {successful_runs/num_runs*100:.1f}%")
        
        # Save results
        results_file = f"config_sweep_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {results_file}")
        print(f"📁 Config files saved in: {self.sweep_dir}")
        
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Config-based chess parameter sweep using Protein algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=10, 
                       help="Number of sweep runs to execute")
    parser.add_argument("--timeout", type=int, default=300, 
                       help="Hard timeout per run in seconds (5 minutes default)")
    parser.add_argument("--hang-timeout", type=int, default=120,
                       help="Hang detection timeout in seconds (2 minutes default)")
    parser.add_argument("--config", type=str, default=None,
                       help="Base config file path (defaults to chess config)")
    parser.add_argument("--log", type=str, default="sweep_debug.log",
                       help="Log file for detailed output")
    
    args = parser.parse_args()
    
    print(f"🐡 Chess Parameter Sweep")
    print(f"📊 Runs: {args.runs}")
    print(f"⏱️ Timeout: {args.timeout}s per run")
    print(f"📄 Log file: {args.log}")
    print()
    
    try:
        sweep = ChessSweep(base_config_path=args.config)
        results = sweep.run_sweep(
            num_runs=args.runs,
            timeout_per_run=args.timeout,
            hang_timeout=args.hang_timeout,
            log_file=args.log
        )
        
        # Show summary of best runs based on Protein scores
        runs_with_scores = [r for r in results if r['protein_score'] is not None]
        if runs_with_scores:
            print(f"\n🏆 Best runs (ranked by Protein optimization score):")
            # Sort by protein_score descending (higher is better for episode_return)
            sorted_runs = sorted(runs_with_scores, key=lambda x: x['protein_score'], reverse=True)
            for i, run in enumerate(sorted_runs[:3]):
                print(f"  {i+1}. Run {run['run_id']}: score={run['protein_score']:.3f}, time={run['elapsed_time']:.1f}s")
        elif successful:
            print(f"\n🏆 Completed runs (ranked by time, no Protein scores available):")
            sorted_runs = sorted(successful, key=lambda x: x['elapsed_time'])
            for i, run in enumerate(sorted_runs[:3]):
                print(f"  {i+1}. Run {run['run_id']}: {run['elapsed_time']:.1f}s")
                
    except KeyboardInterrupt:
        print("\n⚠️ Sweep interrupted by user")
        sys.exit(1)