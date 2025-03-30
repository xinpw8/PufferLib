'''
    Example usage:
    python run.py cartpole eval -w
    to eval the latest model file for the cartpole environment.

    eval is currently the only option for mode
    model file can be anywhere in PufferLib
    -w flag will extract weights from the latest model file,
    update the .c file with the new weights, sizes, observation size,
    and action size, compile it locally, and run the .c file.
'''

import os
import sys
import glob
import time
import torch

def find_env_name(config_dir, search_arg):
    """Search recursively in the config directory for a file containing the search argument."""
    for root, _, files in os.walk(config_dir):
        for file in files:
            print(f"Searching {config_dir} for {search_arg} in {file}")
            if search_arg in file:
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip().startswith("env_name"):
                            _, env_name = line.split("=", 1)
                            return env_name.strip()
    return None

def find_latest_model_path(base_dir, term):
    """
    Search for the newest model file matching the exact environment name.
    Uses strict prefix matching and examines only the experiments directory first.
    """
    experiments_dir = os.path.join(base_dir, "experiments")
    model_files = []
    
    print(f"Searching for model files for environment: {term}")
    
    # PHASE 1: Search only in experiments directory with strict prefix matching
    if os.path.exists(experiments_dir):
        print(f"Examining experiments directory: {experiments_dir}")
        
        # Get all subdirectories directly in the experiments directory
        experiment_dirs = [d for d in os.listdir(experiments_dir) 
                          if os.path.isdir(os.path.join(experiments_dir, d))]
        
        # Find directories that match our environment exactly (before the hash)
        matching_dirs = []
        for dir_name in experiment_dirs:
            # Extract base name (everything before the first dash)
            base_name = dir_name.split('-')[0]
            
            # Only consider EXACT matches
            if base_name == term:
                matching_dirs.append(dir_name)
                print(f"Found exact environment match: {dir_name}")
        
        # Search for model files in matching directories
        for dir_name in matching_dirs:
            dir_path = os.path.join(experiments_dir, dir_name)
            for file in os.listdir(dir_path):
                if file.startswith("model_") and file.endswith(".pt"):
                    model_files.append(os.path.join(dir_path, file))
                    print(f"  Found model file: {file}")
    
    # PHASE 2: If no models found, try fallback search
    if not model_files:
        print("No exact environment matches found in experiments directory.")
        return None
    
    # Sort by creation time and return the most recent file
    model_files.sort(key=os.path.getctime, reverse=True)
    selected_file = model_files[0]
    
    # Verification output
    print(f"Selected most recent model file: {selected_file}")
    print(f"Model creation time: {time.ctime(os.path.getctime(selected_file))}")
    print(f"Model directory: {os.path.dirname(selected_file)}")
    
    return selected_file

    if not model_files:
        # Fallback to search all files if directory matching fails
        print("No matching directories found, falling back to file search...")
        for root, _, files in os.walk(base_dir):
            # Get the parent directory name
            parent_dir = os.path.basename(root)
            parent_parts = parent_dir.split('-', 1)
            parent_name = parent_parts[0]
            parent_name_base = parent_name.replace('puffer_', '')
            
            # Apply the same exact matching logic to parent directories
            exact_match = (parent_name == term)
            exact_match_with_hash = (len(parent_parts) > 1 and parent_parts[0] == term)
            base_exact_match = (parent_name_base == term_base)
            base_exact_match_with_hash = (len(parent_parts) > 1 and parent_parts[0].replace('puffer_', '') == term_base)
            
            # Only consider exact matches
            if exact_match or exact_match_with_hash or base_exact_match or base_exact_match_with_hash:
                for file in files:
                    if file.startswith("model_") and file.endswith(".pt"):
                        model_files.append(os.path.join(root, file))
                        print(f"Found model file: {os.path.join(root, file)}")

    if not model_files:
        return None

    # Sort by creation time and return the most recent file
    model_files.sort(key=os.path.getctime, reverse=True)
    selected_file = model_files[0]
    
    # Verification output
    print(f"Selected most recent model file: {selected_file}")
    print(f"Model creation time: {time.ctime(os.path.getctime(selected_file))}")
    print(f"Model directory: {os.path.dirname(selected_file)}")
    
    return selected_file

def update_save_net_flat(model_path, env_name):
    save_net_file = "save_net_flat.py"
    if not os.path.exists(save_net_file):
        print(f"Error: {save_net_file} not found.")
        sys.exit(1)

    output_dir = os.path.join(os.getcwd(), "pufferlib", "resources", env_name.replace('puffer_', ''))
    weights_output_file = f"{env_name.replace('puffer_', '')}_weights.bin"

    with open(save_net_file, 'r') as f:
        lines = f.readlines()

    with open(save_net_file, 'w') as f:
        for line in lines:
            if line.strip().startswith("MODEL_FILE_NAME"):
                f.write(f"MODEL_FILE_NAME = '{model_path}'\n")
            elif line.strip().startswith("WEIGHTS_OUTPUT_FILE_NAME"):
                f.write(f"WEIGHTS_OUTPUT_FILE_NAME = '{weights_output_file}'\n")
            elif line.strip().startswith("OUTPUT_FILE_PATH"):
                f.write(f"OUTPUT_FILE_PATH = '{output_dir}'\n")
            else:
                f.write(line)
    print(f"Updated {save_net_file} with model and weights paths.\nModel path:{model_path}, Weights output dir:{output_dir} ")
    return output_dir

def extract_details_from_architecture_file(architecture_file):
    observation_size = 0
    action_size = 0
    num_weights = 0
    decoder_logstd = []
    is_continuous = False

    with open(architecture_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if "encoder.weight" in line:
                parts = line.split("torch.Size([")[1].split("]")[0].split(",")
                observation_size = int(parts[1].strip())

            elif "decoder_mean.weight" in line:
                parts = line.split("torch.Size([")[1].split("]")[0].split(",")
                action_size = int(parts[0].strip())
                is_continuous = True

            elif "decoder.weight" in line and "decoder_mean.weight" not in line:
                parts = line.split("torch.Size([")[1].split("]")[0].split(",")
                action_size = int(parts[0].strip())
                is_continuous = False

            elif "Num weights" in line:
                num_weights = int(line.split(":")[1].strip())

            elif line.startswith("decoder_logstd_values:"):
                decoder_logstd_str = line.split(":")[1].strip()
                decoder_logstd = [float(x) for x in decoder_logstd_str.split(",")]

    return observation_size, action_size, num_weights, decoder_logstd, is_continuous

# def extract_details_from_architecture_file(architecture_file):
#     observation_size = 0
#     action_size = 0
#     num_weights = 0

#     with open(architecture_file, 'r') as f:
#         for line in f:
#             # Extract observation size from encoder.weight
#             if "encoder.weight" in line:
#                 try:
#                     parts = line.split("torch.Size([")[1].split("]")[0].split(",")
#                     if len(parts) >= 2:
#                         observation_size = int(parts[1].strip())
#                     print(f"observation_size:{observation_size}")
#                 except (IndexError, ValueError) as e:
#                     print(f"Error parsing observation size from line: {line}. Error: {e}")

#             # Now check for either decoder_mean.weight or decoder.weight
#             elif "decoder_mean.weight" in line or "decoder.weight" in line:
#                 try:
#                     parts = line.split("torch.Size([")[1].split("]")[0].split(",")
#                     if len(parts) >= 1:
#                         action_size = int(parts[0].strip())
#                     print(f"action_size:{action_size}")
#                 except (IndexError, ValueError) as e:
#                     print(f"Error parsing action size from line: {line}. Error: {e}")
            
#             elif "Num weights" in line:
#                 try:
#                     num_weights = int(line.split(":")[1].strip())
#                     print(f"num_weights:{num_weights}")
#                 except (IndexError, ValueError) as e:
#                     print(f"Error parsing num_weights from line: {line}. Error: {e}")

#     return observation_size, action_size, num_weights

def find_c_file(top_dir, env_name):
    """Search the entire top directory for the corresponding .c file."""
    env_basename = env_name.replace('puffer_', '')
    # Preserve the full environment name including all elements after underscore
    for root, _, files in os.walk(top_dir):
        for file in files:
            if file == f"{env_basename}.c":
                return os.path.join(root, file)
    return None

def update_c_file(c_file_path, weights_file, observation_size, action_size, num_weights, decoder_logstd, is_continuous):
    print(f"\nUpdating C file: {c_file_path}")
    print(f"With weights file: {weights_file}")
    print(f"Observation size: {observation_size}, Action size: {action_size}, Num weights: {num_weights}, Continuous: {is_continuous}\n")
    
    with open(c_file_path, 'r') as f:
        lines = f.readlines()

    with open(c_file_path, 'w') as f:
        for line in lines:
            if line.strip().startswith("const char* WEIGHTS_PATH"):
                f.write(f"const char* WEIGHTS_PATH = \"{weights_file}\";\n")
            elif line.strip().startswith("#define OBSERVATIONS_SIZE"):
                f.write(f"#define OBSERVATIONS_SIZE {observation_size}\n")
            elif line.strip().startswith("#define ACTIONS_SIZE"):
                f.write(f"#define ACTIONS_SIZE {action_size}\n")
            elif line.strip().startswith("#define NUM_WEIGHTS"):
                f.write(f"#define NUM_WEIGHTS {num_weights}\n")
            elif line.strip().startswith("#define CONTINUOUS"):
                f.write(f"#define CONTINUOUS {1 if is_continuous else 0}\n")
            elif line.strip().startswith("float decoder_logstd"):
                decoder_logstd_str = ', '.join(f"{x:.7f}f" for x in decoder_logstd)
                f.write(f"float decoder_logstd[ACTIONS_SIZE] = {{{decoder_logstd_str}}};\n")
            else:
                f.write(line)
    print(f"Updated {c_file_path} with new weights and sizes.")

def cleanup_files(files_to_remove):
    """Remove temporary files created during the process."""
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed temporary file: {file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python <script_name.py> <mode> <config_arg> [-w]")
        sys.exit(1)

    mode = sys.argv[2]
    config_arg = sys.argv[1]
    extract_weights_flag = "-w" in sys.argv

    # Define directories
    top_dir = os.getcwd()  # Top-level directory
    config_dir = os.path.join(top_dir, "config")

    # Step 1: Find env_name in config files
    env_name = find_env_name(config_dir, config_arg)
    if not env_name:
        print(f"Error: No env_name found for {config_arg} in config directory.")
        sys.exit(1)

    if mode == "e" or mode == "ev" or mode == "eva" or mode == "eval":
        mode = "eval"
        # Step 2: Find the most recent model file
        model_path = find_latest_model_path(top_dir, env_name)
        if not model_path:
            print(f"Error: No model file found for environment {env_name}.")
            sys.exit(1)

        # Step 3: Update save_net_flat.py constants
        if extract_weights_flag:
            output_path = update_save_net_flat(model_path, env_name)

            # Run python save_net_flat.py
            os.system("python save_net_flat.py")
            
            if not os.path.exists(output_path):
                print(f"Error: Weights file not found after saving at {output_path}")
                sys.exit(1)
            else:
                print(f"Weights file successfully saved at {output_path}")


            weights_file = os.path.join(output_path, f"{env_name.replace('puffer_', '')}_weights.bin")
            architecture_file = f"{weights_file}_architecture.txt"

            if not os.path.exists(architecture_file):
                print(f"Error: Architecture file {architecture_file} not found.")
                sys.exit(1)

            # Step 4: Extract additional details from the architecture file
            observation_size, action_size, num_weights, decoder_logstd, is_continuous = extract_details_from_architecture_file(architecture_file)
            # Step 5: Find the .c file
            c_file_path = find_c_file(top_dir, env_name)
            if not c_file_path:
                print(f"Error: .c file for {env_name} not found.")
                sys.exit(1)

            # Step 6: Update the .c file
            update_c_file(c_file_path, weights_file, observation_size, action_size, num_weights, decoder_logstd, is_continuous)
            # Step 7: Compile and run
            env_basename = env_name.replace('puffer_', '')
            env_base_name = env_basename.split('_')[0]
            binary_path = env_base_name
            print(f"Compiling {env_basename}...")
            os.system(f"scripts/build_ocean.sh {env_basename} local")
            print(f"Running {env_basename} locally...")

            # Ensure the binary exists before running
            binary_path = os.path.join(os.getcwd(), env_basename)
            
            print(f"Expected binary path: {binary_path}")
            os.system(f"ls -l {os.getcwd()}")

            print(f"Current working directory before running save_net_flat.py: {os.getcwd()}")

            if not os.path.exists(binary_path):
                print(f"Error: Binary {env_basename} not found at {binary_path}")
                sys.exit(1)

            # Run the binary
            os.system(f"./{env_basename}")


            # # Step 8: Cleanup temporary files
            # cleanup_files([architecture_file, f"{env_basename}.bin"])
            # cleanup_files([f"{env_basename}"])
        else:
            # Default behavior: Run the command
            command = f"python demo.py --env {env_name} --mode {mode} --eval-model-path {model_path}"
            print(f"Running command: {command}")
            os.system(command)
    elif mode == "t" or mode == "tr" or mode == "tra" or mode == "trai" or mode == "train":
        mode = "train"
        # Default behavior: Run the command
        command = f"python demo.py --env {env_name} --mode {mode} --track --wandb"
        print(f"Running command: {command}")
        os.system(command)

if __name__ == "__main__":
    main()