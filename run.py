'''
    python eval.py eval env_name
    to eval the latest model file for the specified environment.
    e.g. python run.py eval blastar
    or python run.py eval blastar -w

    eval is currently the only option for mode
    model file can be anywhere in PufferLib
    -w flag will extract weights from the latest model file,
    update the .c file with the new weights and sizes, compile
    it locally, and run the .c file.
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
            if search_arg in file:
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip().startswith("env_name"):
                            _, env_name = line.split("=", 1)
                            return env_name.strip()  # Strip whitespace
    return None

def find_latest_model_path(base_dir, term):
    """Search for the newest model file matching the term."""
    search_dirs = [os.path.join(base_dir, "experiments"), base_dir]
    model_files = []

    for dir_path in search_dirs:
        if not os.path.exists(dir_path):
            continue
        for root, dirs, files in os.walk(dir_path):
            # Check for exact match of directories with term
            for d in dirs:
                if d.startswith(term) or term.replace('puffer_', '') in d:
                    model_dir = os.path.join(root, d)
                    for file in os.listdir(model_dir):
                        if file.startswith("model_"):
                            model_files.append(os.path.join(model_dir, file))

    if not model_files:
        # Fallback to search all files if directory matching fails
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.startswith("model_") and file.endswith(".pt"):
                    model_files.append(os.path.join(root, file))

    if not model_files:
        return None

    # Sort by creation time and return the most recent file
    model_files.sort(key=os.path.getctime, reverse=True)
    return model_files[0]

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
    print(f"Updated {save_net_file} with model and weights paths.")
    return output_dir

def extract_details_from_architecture_file(architecture_file):
    """Extract observation size, action size, and num_weights from the architecture file."""
    observation_size = 0
    action_size = 0
    num_weights = 0

    with open(architecture_file, 'r') as f:
        for line in f:
            if "policy.policy.encoder.weight" in line:
                observation_size = int(line.split("[")[1].split(",")[1].strip().replace("])", ""))
            elif "policy.policy.decoder.weight" in line:
                action_size = int(line.split("[")[1].split(",")[0].strip())
            elif "Num weights" in line:
                num_weights = int(line.split(":")[1].strip())

    return observation_size, action_size, num_weights

def find_c_file(top_dir, env_name):
    """Search the entire top directory for the corresponding .c file."""
    env_basename = env_name.replace('puffer_', '')
    for root, _, files in os.walk(top_dir):
        for file in files:
            if file == f"{env_basename}.c":
                return os.path.join(root, file)
    return None

def update_c_file(c_file_path, weights_file, observation_size, action_size, num_weights):
    """Update the .c file with new weights and sizes."""
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
        print("Usage: python run.py env_name mode [-w]\nenv_name: The environment name to run\nmode: eval, train, compile\n-w: Extract weights from the latest model file")
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
            observation_size, action_size, num_weights = extract_details_from_architecture_file(architecture_file)

            # Step 5: Find the .c file
            c_file_path = find_c_file(top_dir, env_name)
            if not c_file_path:
                print(f"Error: .c file for {env_name} not found.")
                sys.exit(1)

            # Step 6: Update the .c file
            update_c_file(c_file_path, weights_file, observation_size, action_size, num_weights)

            # Step 7: Compile and run
            env_basename = env_name.replace('puffer_', '')
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
        command = f"python demo.py --env {env_name} --mode {mode} --track"
        print(f"Running command: {command}")
        os.system(command)
    elif mode == "c" or mode == "co" or mode == "com" or mode == "comp" or mode == "compil" or mode == "compile":
        mode = "compile"
        # Default behavior: Run the command
        command = f"python setup.py build_ext --inplace"
        print(f"Running command: {command}")
        os.system(command)

if __name__ == "__main__":
    main()
