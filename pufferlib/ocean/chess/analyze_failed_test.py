#!/usr/bin/env python
"""
CLI helper to step through a PGN from the FICS email archive and show where
python-chess and OpenSpiel diverge.

Usage:
    python tools/debug_chess_replay.py "<path-to-email.txt>" [--max <plies>]
"""
import argparse, io, re, sys
from pathlib import Path

import chess.pgn, pyspiel
from tests.test_chess_replay import _normalize_san  # reuse test helper


def debug_game(pgn_file: Path, max_plies: int | None = None):
    txt = pgn_file.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^1\. ", txt, flags=re.MULTILINE)
    if not m:
        sys.exit("Could not locate moves in file")

    game = chess.pgn.read_game(io.StringIO(txt[m.start() :]))
    board_py = game.board()
    state = pyspiel.load_game("chess").new_initial_state()

    for ply, move in enumerate(game.mainline_moves(), start=1):
        san = board_py.san(move)
        board_py.push(move)

        match = next(
            (
                a
                for a in state.legal_actions()
                if _normalize_san(state.action_to_string(a)) == _normalize_san(san)
            ),
            None,
        )
        if match:
            state.apply_action(match)
        else:
            print(f"\nDIVERGENCE at ply {ply}: {san}")
            print("python-chess FEN:", board_py.fen())
            print(
                "OpenSpiel legal moves:",
                [state.action_to_string(a) for a in state.legal_actions()][:40],
            )
            return

        if max_plies and ply >= max_plies:
            print(f"No divergence in first {max_plies} plies.")
            return

    print("No divergence – boards stayed in sync to the end.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("email_file", help="Path to FICS email *.txt")
    parser.add_argument("--max", type=int, help="Stop after this many plies")
    args = parser.parse_args()
    debug_game(Path(args.email_file), args.max)