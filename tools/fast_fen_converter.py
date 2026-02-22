"""Fast FEN→observation converter without python-chess Board() overhead.

Skips legal move generation (valid_pieces/valid_dests/valid_promos are left zero).
For supervised training, we only need the board state + action targets.
The action mask comes from the model learning which moves are plausible.
"""
import struct
import numpy as np
from typing import Tuple, Optional

# Observation layout
O_BOARD = 0
O_SIDE = 768
O_CASTLE = 770
O_EP = 786
O_PICK_PHASE = 851
O_SELECTED_PIECE = 853
O_VALID_PIECES = 917
O_VALID_DESTS = 981
O_VALID_PROMOS = 1045
O_SELF_CHECK = 1077
O_OPP_CHECK = 1078
O_RULE50 = 1079
O_REPETITION = 1080
O_PASS_VALID = 1081
OBS_SIZE = 1082

WHITE_OO = 1
WHITE_OOO = 2
BLACK_OO = 4
BLACK_OOO = 8

# Piece mapping: FEN char → (color, piece_type_index)
# Our planes: 0-5 = our P,N,B,R,Q,K; 6-11 = opp P,N,B,R,Q,K
PIECE_CHARS = {
    'P': (0, 0), 'N': (0, 1), 'B': (0, 2), 'R': (0, 3), 'Q': (0, 4), 'K': (0, 5),
    'p': (1, 0), 'n': (1, 1), 'b': (1, 2), 'r': (1, 3), 'q': (1, 4), 'k': (1, 5),
}

# Square name → index
FILE_MAP = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}


def fast_fen_to_obs_and_actions(fen: str, uci_move: str) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
    """Convert FEN + UCI move to (obs_phase0, obs_phase1, action0, action1).

    Returns None on parse errors.
    Much faster than the python-chess version because it skips legal move generation.
    """
    parts = fen.split()
    if len(parts) < 4:
        return None

    board_str, side, castling, ep = parts[0], parts[1], parts[2], parts[3]
    halfmove = int(parts[4]) if len(parts) > 4 else 0

    # Parse side to move
    is_white = (side == 'w')
    player = 0 if is_white else 1
    flip = player * 56

    # Parse board
    # board_pieces[sq] = (color, piece_type_index) or None
    board_pieces = [None] * 64
    sq = 56  # Start at a8 (rank 8)
    for ch in board_str:
        if ch == '/':
            sq -= 16  # Move to start of next rank below
        elif ch.isdigit():
            sq += int(ch)
        elif ch in PIECE_CHARS:
            board_pieces[sq] = PIECE_CHARS[ch]
            sq += 1

    # Build observations
    obs_p0 = np.zeros(OBS_SIZE, dtype=np.uint8)
    obs_p1 = np.zeros(OBS_SIZE, dtype=np.uint8)

    # Board planes (same for both phases)
    for sq_idx in range(64):
        if board_pieces[sq_idx] is None:
            continue
        color, pt_idx = board_pieces[sq_idx]
        is_ours = (color == 0 and is_white) or (color == 1 and not is_white)
        plane = pt_idx if is_ours else (6 + pt_idx)
        view_sq = sq_idx ^ flip
        obs_p0[O_BOARD + plane * 64 + view_sq] = 1
        obs_p1[O_BOARD + plane * 64 + view_sq] = 1

    # Side to move (always "our turn" for the training example)
    obs_p0[O_SIDE] = 1
    obs_p1[O_SIDE] = 1

    # Castling
    castle_rights = 0
    if 'K' in castling:
        castle_rights |= WHITE_OO
    if 'Q' in castling:
        castle_rights |= WHITE_OOO
    if 'k' in castling:
        castle_rights |= BLACK_OO
    if 'q' in castling:
        castle_rights |= BLACK_OOO
    if player == 1:
        flipped = 0
        if castle_rights & BLACK_OO:
            flipped |= WHITE_OO
        if castle_rights & BLACK_OOO:
            flipped |= WHITE_OOO
        if castle_rights & WHITE_OO:
            flipped |= BLACK_OO
        if castle_rights & WHITE_OOO:
            flipped |= BLACK_OOO
        castle_rights = flipped
    obs_p0[O_CASTLE + castle_rights] = 1
    obs_p1[O_CASTLE + castle_rights] = 1

    # En passant
    if ep == '-':
        obs_p0[O_EP + 64] = 1
        obs_p1[O_EP + 64] = 1
    else:
        ep_sq = FILE_MAP[ep[0]] + (int(ep[1]) - 1) * 8
        ep_view = ep_sq ^ flip
        obs_p0[O_EP + ep_view] = 1
        obs_p1[O_EP + ep_view] = 1

    # Phase
    obs_p0[O_PICK_PHASE] = 1      # Phase 0
    obs_p1[O_PICK_PHASE + 1] = 1  # Phase 1

    # Rule50
    obs_p0[O_RULE50] = min(255, (halfmove * 255) // 100)
    obs_p1[O_RULE50] = min(255, (halfmove * 255) // 100)

    # Repetition (no history available)
    obs_p0[O_REPETITION] = 255
    obs_p1[O_REPETITION] = 255

    # Parse move
    if len(uci_move) < 4:
        return None
    from_file = FILE_MAP.get(uci_move[0])
    from_rank = int(uci_move[1]) - 1
    to_file = FILE_MAP.get(uci_move[2])
    to_rank = int(uci_move[3]) - 1
    if from_file is None or to_file is None:
        return None

    from_sq = from_rank * 8 + from_file
    to_sq = to_rank * 8 + to_file

    # Actions in player's view
    action0 = from_sq ^ flip
    obs_p1[O_SELECTED_PIECE + action0] = 1  # Set selected piece for phase 1

    # Phase 1 action
    if len(uci_move) == 5:
        # Promotion
        promo_char = uci_move[4]
        promo_map = {'q': 0, 'r': 1, 'b': 2, 'n': 3}
        if promo_char not in promo_map:
            return None
        promo_idx = promo_map[promo_char]
        action1 = 64 + promo_idx * 8 + to_file
    else:
        action1 = to_sq ^ flip

    return obs_p0, obs_p1, action0, action1


def convert_bag_parallel(bag_path: str, output_dir: str,
                          chunk_size: int = 500_000,
                          max_records: int = None,
                          num_workers: int = 8):
    """Convert .bag file to numpy using multiprocessing."""
    import os
    import multiprocessing as mp
    from functools import partial

    os.makedirs(output_dir, exist_ok=True)

    # Get record count
    with open(bag_path, 'rb') as f:
        f.seek(-8, 2)
        file_size = f.tell() + 8
        index_start = struct.unpack('<Q', f.read(8))[0]
    index_size = file_size - 8 - index_start
    total_records = index_size // 8
    if max_records:
        total_records = min(total_records, max_records)

    print(f"Converting {total_records:,} records from {bag_path}")
    print(f"Using {num_workers} workers, chunk size {chunk_size:,}")

    # Split into chunks
    num_chunks = (total_records + chunk_size - 1) // chunk_size
    chunk_ranges = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_records)
        chunk_ranges.append((start, end, i))

    # Process chunks in parallel
    worker_fn = partial(_process_chunk, bag_path=bag_path, output_dir=output_dir)
    with mp.Pool(num_workers) as pool:
        results = pool.map(worker_fn, chunk_ranges)

    total_converted = sum(r[0] for r in results)
    total_skipped = sum(r[1] for r in results)
    print(f"\nDone. {total_converted:,} records converted, {total_skipped:,} skipped.")


def _process_chunk(args, bag_path: str, output_dir: str):
    """Worker function to convert a chunk of records."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    start_idx, end_idx, chunk_num = args
    count = end_idx - start_idx

    # Import BagReader here to create per-worker mmap
    from tools.fen_converter import BagReader
    reader = BagReader(bag_path)

    obs0_buf = np.zeros((count, OBS_SIZE), dtype=np.uint8)
    obs1_buf = np.zeros((count, OBS_SIZE), dtype=np.uint8)
    act_buf = np.zeros((count, 2), dtype=np.int64)

    valid = 0
    skipped = 0

    for i in range(start_idx, end_idx):
        fen, move = reader.decode_behavioral_cloning(i)
        result = fast_fen_to_obs_and_actions(fen, move)
        if result is None:
            skipped += 1
            continue
        obs_p0, obs_p1, a0, a1 = result
        obs0_buf[valid] = obs_p0
        obs1_buf[valid] = obs_p1
        act_buf[valid] = [a0, a1]
        valid += 1

    # Save
    if valid > 0:
        np.save(os.path.join(output_dir, f"obs_phase0_{chunk_num:04d}.npy"), obs0_buf[:valid])
        np.save(os.path.join(output_dir, f"obs_phase1_{chunk_num:04d}.npy"), obs1_buf[:valid])
        np.save(os.path.join(output_dir, f"actions_{chunk_num:04d}.npy"), act_buf[:valid])

    if chunk_num % 10 == 0:
        print(f"  Chunk {chunk_num}: {valid:,} converted, {skipped:,} skipped")

    return valid, skipped


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python fast_fen_converter.py <bag_path> <output_dir> [max_records] [num_workers]")
        sys.exit(1)

    bag_path = sys.argv[1]
    output_dir = sys.argv[2]
    max_records = int(sys.argv[3]) if len(sys.argv) > 3 else None
    num_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    convert_bag_parallel(bag_path, output_dir,
                          max_records=max_records,
                          num_workers=num_workers)
