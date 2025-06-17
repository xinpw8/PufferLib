Build the native code with one script.
Validate thousands of real games quickly (mass_replay_chess.py ⇄ replay_chess).

1. scripts/build_chess.sh (bash)
• One-command builder.
Rebuilds static Abseil if it isn’t present.
Compiles three native binaries and drops them in pufferlib/:
demo_chess
replay_chess
raylib_chess (if the Raylib 5.5 bundle exists)
Run from anywhere:
chmod +x pufferlib/scripts/build_chess.sh
pufferlib/scripts/build_chess.sh        

Note: binaries appear in pufferlib top dir.

2. replay_chess (C++ → pufferlib/replay_chess)
source: pufferlib/pufferlib/ocean/chess/replay_chess.cc
• “Headless” validator used by CI tests and mass-replay.
Takes one argument: a text file that contains only whitespace-separated SAN moves.
Replays the game through the native CChess engine and exits 0 if every move was legal.
Usage: ./pufferlib/replay_chess moves.txt

3. demo_chess (console demo)
source: pufferlib/pufferlib/ocean/chess/demo_chess.cc
• Minimal interactive CLI to play against the engine.
Accepts coordinate input like e2e4, or r for a random move, q to quit.
Renders an ASCII board after every move.

4. raylib_chess (GUI demo)
source: pufferlib/pufferlib/ocean/chess/raylib_chess.cc
• Optional graphical viewer built with Raylib 5.5.
Needs the pre-built bundle in raylib-5.5_linux_amd64/ (already present in repo).
If a PGN path is supplied it auto-animates the game; otherwise it opens an interactive board (click-and-drag to move).

5. tools/extract_pgn_moves.py (Python)
• Streams through a huge PGN, strips headers/comments, returns plain move strings.
Used both for quick inspection and also by the batch tester.

6. tools/mass_replay_chess.py (Python)
• Batch harness that glues the extractor and replay_chess together.
Validates N games from any PGN with PyTest-style dot output.

python tools/mass_replay_chess.py /puffertank/release_test_pufferlib/pufferlib/resources/chess/games_database/ficsgamesdb_2024_standard_nomovetimes_443694.pgn -n 1000
....................................................................
Summary: 1000 passed / 1000 total

7. flat_chess_env.h (C++)
• Single-translation-unit “CChess” environment embedded inside pufferlib/pufferlib/ocean/chess/.
Wraps OpenSpiel’s C++ chess engine behind a tiny C interface (allocate, c_step, etc.) so PufferLib (or any RL loop) can drive it at high speed.
