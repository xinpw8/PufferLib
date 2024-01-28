import shutil
import os

def copy_files(exp_name, update):
    src_dir = os.path.join('experiments', exp_name, 'sessions')
    dest_dir = os.path.join('experiments', exp_name, 'sessions')
    # print(f"Source Directory: {src_dir}")
    # print(f"Destination Directory: {dest_dir}")
    # Iterate over all session directories in the source directory
    for session_dir in os.listdir(src_dir):
        session_path = os.path.join(src_dir, session_dir)
        if os.path.isdir(session_path):
            # print(f"Processing Session Directory: {session_path}")
            # Create the destination directory if it doesn't exist
            dest_session_dir = os.path.join(dest_dir, session_dir, 'states_copy')
            os.makedirs(dest_session_dir, exist_ok=True)
            # Get the list of state files to be copied
            state_files = [f for f in os.listdir(os.path.join(session_path, 'states')) if f.endswith('.state')]
            # Iterate over the state files
            for state_file in state_files:
                file_path = os.path.join(session_path, 'states', state_file)
                # print(f"Copying File: {file_path}")
                # Extract the session ID from the file name
                session_id = state_file.split('_')[1].split('.')[0]
                # Append the update to the copied file name
                dest_file_name = f'env_{session_id}_{update}.state'
                # Copy the file to the destination directory
                dest_file_path = os.path.join(dest_session_dir, dest_file_name)
                shutil.copyfile(file_path, dest_file_path)
                # print(f"Copied: {file_path} to {dest_file_path}")
