import os
import re
import argparse
from collections import Counter

# Define common chess openings in UCI notation
OPENINGS = {
    "King's Pawn Opening": ["e2e4"],
    "Queen's Pawn Opening": ["d2d4"],
    "Réti Opening": ["g1f3"],
    "English Opening": ["c2c4"],
    "Bird's Opening": ["f2f4"],
    "Sicilian Defense": ["e2e4", "c7c5"],
    "French Defense": ["e2e4", "e7e6"],
    "Caro-Kann Defense": ["e2e4", "c7c6"],
    "Queen's Gambit": ["d2d4", "d7d5", "c2c4"],
    "King's Gambit": ["e2e4", "e7e5", "f2f4"],
    "Italian Game": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
    "Ruy López": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    "Scotch Game": ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"],
    "Indian Defense": ["d2d4", "g8f6"],
    "King's Indian Defense": ["d2d4", "g8f6", "c2c4", "g7g6"],
    "Nimzo-Indian Defense": ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],
    "Queen's Indian Defense": ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"],
    "Dutch Defense": ["d2d4", "f7f5"],
    "Van't Kruijs Opening": ["e2e3"],
    "Benko's Opening": ["g2g3"],
    "Larsen's Opening": ["b2b3"],
    "Polish Opening": ["b2b4"],
    "Grob's Attack": ["g2g4"],
    "King's Fianchetto Opening": ["g2g3"],
    "Anderssen's Opening": ["a2a3"],
    "Ware Opening": ["a2a4"],
    "Sodium Attack": ["b1a3"],
    "Saragossa Opening": ["c2c3"],
    "Dunsk-Peruvian Gambit": ["h2h4"],
    "Amar Opening": ["g1h3"],
    "Other": []  # Catch-all for less common first moves
}

def parse_pgn(file_path):
    """Parses a PGN file and returns a list of moves in UCI format."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Remove headers
        content = re.sub(r'\[.*?\]\s*', '', content)
        
        # Remove move numbers, results, and newlines
        moves_str = re.sub(r'\d+\.\s*|\n|\r|1-0|0-1|1/2-1/2|\*', ' ', content).strip()
        
        # Split into individual moves, filtering out any empty strings
        moves = [move for move in moves_str.split(' ') if move]
        
        return moves
    except Exception as e:
        # print(f"Warning: Could not parse file {file_path}: {e}")
        return []

def identify_opening(moves):
    """Identifies the opening played based on the initial moves."""
    if not moves:
        return "Empty Game"

    # Sort openings by length of move sequence, descending, to match specific openings first
    sorted_openings = sorted(OPENINGS.items(), key=lambda item: len(item[1]), reverse=True)

    for name, opening_moves in sorted_openings:
        if not opening_moves:  # Skip the "Other" category for now
            continue
        if len(moves) >= len(opening_moves):
            if moves[:len(opening_moves)] == opening_moves:
                return name
    
    # If no specific match, check the first move for a general category
    first_move = moves[0]
    for name, opening_moves in sorted_openings:
        if len(opening_moves) == 1 and first_move == opening_moves[0]:
            return name
            
    return "Unknown Opening"

def analyze_openings(directory):
    """Analyzes all PGN files in a directory and reports opening frequencies."""
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    pgn_files = [f for f in os.listdir(directory) if f.endswith('.pgn')]
    if not pgn_files:
        print(f"No PGN files found in '{directory}'")
        return

    opening_counter = Counter()
    total_games = 0

    for filename in pgn_files:
        file_path = os.path.join(directory, filename)
        moves = parse_pgn(file_path)
        if moves:
            opening = identify_opening(moves)
            opening_counter[opening] += 1
            total_games += 1
        elif moves is not None: # File parsed but was empty
             opening_counter["Empty Game"] += 1
             total_games += 1


    print(f"\nAnalyzed {total_games} games from '{directory}'.")
    print("-" * 45)
    print(f"{'Opening':<25} {'Frequency':<10} {'Percentage'}")
    print("-" * 45)

    if total_games == 0:
        print("No valid games to analyze.")
        return

    sorted_openings = opening_counter.most_common()
    for opening, count in sorted_openings:
        percentage = (count / total_games) * 100
        print(f"{opening:<25} {count:<10} {percentage:.2f}%")

def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyzes chess PGN files in a directory to determine opening frequencies."
    )
    parser.add_argument(
        "directory", 
        type=str, 
        help="The path to the directory containing PGN files."
    )
    args = parser.parse_args()
    
    analyze_openings(args.directory)

if __name__ == '__main__':
    main()

