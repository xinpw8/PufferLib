import os
from collections import defaultdict
from copy import deepcopy

# Directories and files
base_dir = "experiments"
output_file = "pokes_log.txt"
file_path = "experiments/running_experiment.txt"

# Read the exp name and strip to string
with open(file_path, "r") as pathfile:
    exp_uuid8 = pathfile.readline().strip()

# Update the sessions_path assignment
sessions_path = os.path.join(base_dir, exp_uuid8, "sessions")
output_file_path = os.path.join(base_dir, exp_uuid8, "pokes_log.txt")

# Initialize a dictionary to store Pokémon information
pokemon_summary = defaultdict(list)

# Initialize a set to store unique Pokémon names and moves
unique_pokemon = set()
unique_moves = set()

# Variable to store the session folder associated with each log file
current_session_folder = None

def analyze_pokemon_data(data):
    move_counts = {}
    highest_levels = {}
    pokemon_counts = {}

    for session_id, session_data in data.items():
        for attributes in session_data:
            if 'name' in attributes and attributes['name'] != '':
                # Unique Pokemon
                unique_pokemon.add(attributes['name'])

                # Moves and move counts
                moves = attributes['moves']
                unique_moves.update(moves)
                for move in moves:
                    move_counts[move] = move_counts.get(move, 0) + 1

                # Highest level of each Pokemon
                level = int(attributes['level'])
                pokemon_name = attributes['name']
                if pokemon_name not in highest_levels or level > highest_levels[pokemon_name]:
                    highest_levels[pokemon_name] = level

                # Count of each unique Pokemon
                pokemon_counts[pokemon_name] = pokemon_counts.get(pokemon_name, 0) + 1

    return {
        'Unique Pokemon': sorted(unique_pokemon),
        'Unique Moves': sorted(unique_moves),
        'Move Counts': {k: v for k, v in sorted(move_counts.items())},
        'Highest Levels': {k: v for k, v in sorted(highest_levels.items())},
        'Pokemon Counts': {k: v for k, v in sorted(pokemon_counts.items())}
    }

# Iterate over each session folder
for folder in os.listdir(sessions_path):
    session_path = os.path.join(sessions_path, folder)

    # Print the current session path for debugging
    # print(f"DEBUG: Current Session Path: {session_path}")

    # Store the session folder associated with the current log file
    current_session_folder = folder

    # Read the log file for each session
    log_file_path = os.path.join(session_path, "pokemon_party_log.txt")

    # Check if the log file exists
    if os.path.isfile(log_file_path):
        # Initialize current_pokemon here
        current_pokemon = {}
        current_state = None

        with open(log_file_path, 'r') as log_file:
            lines = log_file.readlines()

            # Debug print to check lines
            # print(f"DEBUG: Lines in {log_file_path}: {lines}")

            # Iterate over lines to extract Pokémon information
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue

                # Debug print to check the current line being processed
                # print(f"DEBUG: Processing line: {line.strip()}")

                # State machine for parsing
                if line.startswith("Slot:"):
                    current_state = "slot"
                    slot = line.strip("\n").split(" ")[-1]
                    current_pokemon["slot"] = slot

                elif line.startswith("Name:"):
                    current_state = "name"
                    current_pokemon["name"] = line.strip("\n").split(" ")[-1]
                    # Add the Pokémon name to the set of unique Pokémon
                    unique_pokemon.add(current_pokemon["name"])

                elif line.startswith("Level:"):
                    current_state = "level"
                    current_pokemon["level"] = line.strip("\n").split(" ")[-1]

                elif line.startswith("Moves:"):
                    current_state = "moves"
                    moves = line.strip("\n").split(":")[-1].split(", ")
                    # Strip spaces from each move
                    moves = [move.strip() for move in moves]
                    current_pokemon["moves"] = moves

                    # Update the set of unique moves
                    unique_moves.update(moves)

                    # Add the current Pokémon to the summary dictionary for the current environment
                    pokemon_summary[current_session_folder].append(deepcopy(current_pokemon))

                    # Reset current_pokemon for the next iteration
                    current_pokemon = {}
                    current_state = None



# Output the aggregated information to the file
result = analyze_pokemon_data(pokemon_summary)
caught = result['Unique Pokemon']
levels = result['Highest Levels']
moves = result['Unique Moves']
incidence = result['Move Counts']


with open(output_file_path, 'w') as output_file:

    # output_file.write(f'result ordered = \nlevels: {levels}\nincidence: {incidence}\n')
    

    output_file.write(
        "\nCaught Pokemon  Highest Level  Incidence\n")
    # output_file.write("  Moves List  Incidence\n")
    output_file.write(
        "--------------  -------------  ---------\n")
    # output_file.write(
    #     "  ----------  ---------\n")
    for pokemon in result['Highest Levels']:
        level = result['Highest Levels'][pokemon]
        count = result['Pokemon Counts'].get(pokemon, 0)
        output_file.write(f"{pokemon.ljust(16)}{str(level).ljust(15)}{str(count)}\n")

    output_file.write("\n\nMoves List      Incidence\n")
    output_file.write(
        "----------      ---------\n")
    for move, count in result['Move Counts'].items():
        output_file.write(f"{move.ljust(15)} {str(count).ljust(18)}\n")
    output_file.write("\n\n")

    # # Write the environment and Pokémon information
    # for env_folder, parties in sorted(pokemon_summary.items()):
    #     env_count = 0
    #     current_pokemon = {}  # Initialize current_pokemon here
    #     for party in parties:
    #         # Only print non-empty slots
    #         if party.get('name'):
    #             output_file.write(f"Slot: {party.get('slot', 'empty')}\n")
    #             output_file.write(f"Name: {party.get('name', 'empty')}\n")
    #             output_file.write(f"Level: {party.get('level', '0')}\n")
    #             output_file.write(f"Moves: {', '.join(party.get('moves', ['empty']))}\n\n")
    

    # Write the environment and Pokémon information
    for env_folder, parties in sorted(pokemon_summary.items()):
        # Include session_uuid4 folder name at the top
        output_file.write(f"\n========== {env_folder} ==========\n")

        env_count = 0
        current_pokemon = {}  # Initialize current_pokemon here
        for party in parties:
            # Only print non-empty slots
            if party.get('name'):
                output_file.write(f"Slot: {party.get('slot', 'empty')}\n")
                output_file.write(f"Name: {party.get('name', 'empty')}\n")
                output_file.write(f"Level: {party.get('level', '0')}\n")
                output_file.write(f"Moves: {', '.join(party.get('moves', ['empty']))}\n\n")

    # Optionally, include a line to separate different environment logs
    output_file.write("\n" + "=" * 30 + "\n")
                
    
                
                # # Print the current_pokemon and unique_pokemon
                # print(f'current_pokemon={current_pokemon}')
                # print(f'unique_pokemon={unique_pokemon}')
