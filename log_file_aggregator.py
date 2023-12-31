import os
from collections import defaultdict
import json

# Define the base directory where session folders are located
base_dir = "/home/daa/puffer0.5.2_iron/bill/pufferlib/"

# Initialize a dictionary to store Pokémon information
pokemon_summary = defaultdict(list)

# Initialize a set to store unique Pokémon names and moves
unique_pokemon = set()
unique_moves = set()

# Output file path
output_file_path = "/home/daa/puffer0.5.2_iron/bill/pufferlib/comp_log.txt"

# Initialize variables to track environment and current_pokemon
current_environment = 1
current_pokemon = {}

# Flag to indicate whether to print empty slots
print_empty_slots = False

# Variable to store the session folder associated with each log file
current_session_folder = None

def analyze_pokemon_data(data):
    unique_pokemon = set()
    unique_moves = set()
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


# Output the aggregated information to the file
with open(output_file_path, 'w') as output_file:

    # Iterate over each session folder
    for session_folder in os.listdir(base_dir):
        session_path = os.path.join(base_dir, session_folder)

        # Check if the item is a directory
        if os.path.isdir(session_path) and 'session_' in session_path and 'pokemon_party_log.txt' in os.listdir(session_path):
            # Store the session folder associated with the current log file
            current_session_folder = session_folder
            pokemon_summary[current_environment].append({'session_folder': current_session_folder})
            
            # Read the log file for each session
            log_file_path = os.path.join(session_path, "pokemon_party_log.txt")
            
            # Check if the log file exists
            if os.path.isfile(log_file_path):
                with open(log_file_path, 'r') as log_file:
                    lines = log_file.readlines()

                    # Iterate over lines to extract Pokémon information
                    for line in lines:
                        if line.startswith("Slot:"):
                            slot = line.strip("\n").split(" ")[-1]
                            if slot == "1" and current_pokemon:
                                # Increment the environment number when a new party starts
                                current_environment += 1
                                # Reset the flag to print empty slots
                                print_empty_slots = False
                            current_pokemon["slot"] = slot
                            # Reset the flag to print empty slots for the current party
                            print_empty_slots = True
                        elif line.startswith("Name:"):
                            current_pokemon["name"] = line.strip("\n").split(" ")[-1]
                            # Add the Pokémon name to the set of unique Pokémon
                            unique_pokemon.add(current_pokemon["name"])
                        elif line.startswith("Level:"):
                            current_pokemon["level"] = line.strip("\n").split(" ")[-1]
                        elif line.startswith("Moves:"):
                            moves = line.strip("\n").split(":")[-1].split(", ")
                            # Strip spaces from each move
                            moves = [move.strip() for move in moves]
                            current_pokemon["moves"] = moves
                            
                            # Update the set of unique moves
                            unique_moves.update(moves)

                            # Add the current Pokémon to the summary dictionary for the current environment
                            pokemon_summary[current_environment].append(current_pokemon.copy())

                            # Reset current_pokemon for the next iteration
                            current_pokemon = {}


    result = analyze_pokemon_data(pokemon_summary)

    # # Write the results to a file
    # output_file.write("Unique Pokemon: " + str(result['Unique Pokemon']) + "\n\n")
    # output_file.write("Unique Moves: " + str(result['Unique Moves']) + "\n\n")

    output_file.write(
        "\nUnique Pokemon       Highest Level         Number in Parties\n")
    output_file.write(
        "--------------       -------------         -------------\n")
    for pokemon in result['Highest Levels']:
        level = result['Highest Levels'][pokemon]
        count = result['Pokemon Counts'].get(pokemon, 0)
        output_file.write(f"{pokemon.ljust(24)}{str(level).ljust(24)}{str(count)}\n")

    # output_file.write("\nUnique Pokemon: Highest Level: Number Caught\n")
    # for pokemon, level in result['Highest Levels'].items():
    #     for pokemon, count in result['Pokemon Counts'].items():
    #         output_file.write(f"  {pokemon}: {level}: {count}\n")

    # output_file.write("\nPokemon Counts:\n")
    # for pokemon, count in result['Pokemon Counts'].items():
    #     output_file.write(f"  {pokemon}: {count}\n")
    
    output_file.write("\n\nUnique Moves         Instances of Move\n")
    output_file.write(
        "------------         -----------------\n")
    for move, count in result['Move Counts'].items():
        output_file.write(f"{move.ljust(20)} {str(count).ljust(20)}\n")
    # Write the unique Pokémon seen at the top of the file
    # output_file.write("Unique Pokemon Caught:\n")
    # for pokemon in sorted(unique_pokemon):
    #     output_file.write(f"{pokemon}\n")
    # output_file.write("\n")

    # Write the unique moves seen
    # output_file.write("Unique Moves List:\n")
    # for move in sorted(unique_moves):
    #     output_file.write(f"{move}\n")
    # output_file.write("\n")
    output_file.write("\n\n\n")
    
    # Write the environment and Pokémon information
    for env_number, parties in sorted(pokemon_summary.items()):
        env_count = 0
        for party in parties:
            session_folder = party.get('session_folder', '')
            if session_folder:
                env_count += 1
                # Print the session folder associated with the log file
                output_file.write(f"======\nENV_{env_count} |\n============={session_folder}==============\n")
            # Only print non-empty slots
            if party.get('name'):
                output_file.write(f"Slot: {party.get('slot', 'empty')}\n")
                output_file.write(f"Name: {party.get('name', 'empty')}\n")
                output_file.write(f"Level: {party.get('level', '0')}\n")
                output_file.write(f"Moves: {', '.join(party.get('moves', ['empty']))}\n\n")
                

    # result = analyze_pokemon_data(pokemon_summary)

    # # Write the results to a file
    # output_file_path = "pokemon_analysis_results.txt"
    # with open(output_file_path, 'w') as output_file:
    #     output_file.write("Unique Pokemon: " + str(result['Unique Pokemon']) + "\n\n")
    #     output_file.write("Unique Moves: " + str(result['Unique Moves']) + "\n\n")

    #     output_file.write("Move Counts:\n")
    #     for move, count in result['Move Counts'].items():
    #         output_file.write(f"  {move}: {count}\n")

    #     output_file.write("\nHighest Levels:\n")
    #     for pokemon, level in result['Highest Levels'].items():
    #         output_file.write(f"  {pokemon}: {level}\n")

    #     output_file.write("\nPokemon Counts:\n")
    #     for pokemon, count in result['Pokemon Counts'].items():
    #         output_file.write(f"  {pokemon}: {count}\n")

    # print(f"Results have been written to: {output_file_path}")    
