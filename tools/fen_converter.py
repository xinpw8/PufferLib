"""Convert DeepMind's searchless chess .bag files to PufferLib chess observations.

Reads behavioral_cloning records (FEN, move) and produces numpy arrays of
(observation, phase0_action, phase1_action) for supervised pre-training.
"""
import struct
import numpy as np
import chess
from typing import Tuple, Optional

# Observation layout constants (from chess.h)
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

# Castling constants (from chess.h)
WHITE_OO = 1
WHITE_OOO = 2
BLACK_OO = 4
BLACK_OOO = 8

# PufferLib piece type mapping: PAWN=1..KING=6
PIECE_TYPE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def fen_to_obs(fen: str) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a FEN string to two PufferLib observations (phase 0 for each side).

    Returns obs for the side to move only (1082 bytes).
    Also returns a valid_pieces mask for action masking.
    """
    board = chess.Board(fen)
    obs = np.zeros(OBS_SIZE, dtype=np.uint8)

    # Determine perspective: the side to move is "us"
    us = board.turn  # True=WHITE, False=BLACK
    player = 0 if us else 1  # 0=white, 1=black
    flip = player * 56

    # Board planes (12 planes x 64 squares)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        pt_idx = PIECE_TYPE_MAP[piece.piece_type]
        is_ours = (piece.color == us)
        plane = pt_idx if is_ours else (6 + pt_idx)
        view_sq = sq ^ flip
        obs[O_BOARD + plane * 64 + view_sq] = 1

    # Side to move (one-hot: [our_turn, opp_turn])
    obs[O_SIDE] = 1  # Always 1 since we're constructing obs for the side to move

    # Castling rights
    castle_rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castle_rights |= WHITE_OO
    if board.has_queenside_castling_rights(chess.WHITE):
        castle_rights |= WHITE_OOO
    if board.has_kingside_castling_rights(chess.BLACK):
        castle_rights |= BLACK_OO
    if board.has_queenside_castling_rights(chess.BLACK):
        castle_rights |= BLACK_OOO

    # Flip castling for black perspective
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
    obs[O_CASTLE + castle_rights] = 1

    # En passant
    if board.ep_square is not None:
        ep_sq = board.ep_square ^ flip
        obs[O_EP + ep_sq] = 1
    else:
        obs[O_EP + 64] = 1  # No EP

    # Pick phase = 0 (selecting piece)
    obs[O_PICK_PHASE] = 1

    # Valid pieces mask (phase 0: which squares have movable pieces)
    legal_moves = list(board.legal_moves)
    for m in legal_moves:
        from_sq = m.from_square ^ flip
        obs[O_VALID_PIECES + from_sq] = 1

    # Check status
    obs[O_SELF_CHECK] = 255 if board.is_check() else 0

    # Switch perspective to check opponent check
    board.push(chess.Move.null())
    obs[O_OPP_CHECK] = 255 if board.is_check() else 0
    board.pop()

    # Rule50
    obs[O_RULE50] = min(255, (board.halfmove_clock * 255) // 100)

    # Repetition (we don't have history, so default to 255 = no repetition)
    obs[O_REPETITION] = 255

    # Pass valid = 0 (it's our turn)
    obs[O_PASS_VALID] = 0

    return obs


def fen_to_obs_phase1(fen: str, from_square: int) -> np.ndarray:
    """Create phase 1 observation (after piece selection).

    Same as phase 0 obs but with:
    - pick_phase = 1
    - selected_piece set
    - valid_dests computed for the selected piece
    - valid_promos computed
    """
    board = chess.Board(fen)
    obs = np.zeros(OBS_SIZE, dtype=np.uint8)

    us = board.turn
    player = 0 if us else 1
    flip = player * 56

    # Board planes
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        pt_idx = PIECE_TYPE_MAP[piece.piece_type]
        is_ours = (piece.color == us)
        plane = pt_idx if is_ours else (6 + pt_idx)
        view_sq = sq ^ flip
        obs[O_BOARD + plane * 64 + view_sq] = 1

    # Side to move
    obs[O_SIDE] = 1

    # Castling
    castle_rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castle_rights |= WHITE_OO
    if board.has_queenside_castling_rights(chess.WHITE):
        castle_rights |= WHITE_OOO
    if board.has_kingside_castling_rights(chess.BLACK):
        castle_rights |= BLACK_OO
    if board.has_queenside_castling_rights(chess.BLACK):
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
    obs[O_CASTLE + castle_rights] = 1

    # En passant
    if board.ep_square is not None:
        ep_sq = board.ep_square ^ flip
        obs[O_EP + ep_sq] = 1
    else:
        obs[O_EP + 64] = 1

    # Pick phase = 1
    obs[O_PICK_PHASE + 1] = 1

    # Selected piece
    view_from = from_square ^ flip
    obs[O_SELECTED_PIECE + view_from] = 1

    # Valid destinations for this piece
    for m in board.legal_moves:
        if m.from_square == from_square:
            to_sq = m.to_square ^ flip
            obs[O_VALID_DESTS + to_sq] = 1

            # Promotion moves
            if m.promotion is not None:
                promo_type_idx = {
                    chess.QUEEN: 0,
                    chess.ROOK: 1,
                    chess.BISHOP: 2,
                    chess.KNIGHT: 3,
                }[m.promotion]
                file_idx = chess.square_file(m.to_square)
                obs[O_VALID_PROMOS + promo_type_idx * 8 + file_idx] = 1

    # Check status
    obs[O_SELF_CHECK] = 255 if board.is_check() else 0
    board.push(chess.Move.null())
    obs[O_OPP_CHECK] = 255 if board.is_check() else 0
    board.pop()

    obs[O_RULE50] = min(255, (board.halfmove_clock * 255) // 100)
    obs[O_REPETITION] = 255
    obs[O_PASS_VALID] = 0

    return obs


def uci_to_actions(uci_move: str, fen: str) -> Tuple[int, int]:
    """Convert a UCI move string to (phase0_action, phase1_action).

    phase0_action: source square (0-63), in the player's view
    phase1_action: dest square (0-63) or promotion action (64-95)
    """
    board = chess.Board(fen)
    us = board.turn
    player = 0 if us else 1
    flip = player * 56

    move = chess.Move.from_uci(uci_move)
    from_sq = move.from_square ^ flip
    to_sq = move.to_square ^ flip

    phase0_action = from_sq

    if move.promotion is not None:
        promo_type_idx = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3,
        }[move.promotion]
        file_idx = chess.square_file(move.to_square)
        phase1_action = 64 + promo_type_idx * 8 + file_idx
    else:
        phase1_action = to_sq

    return phase0_action, phase1_action


class BagReader:
    """Read DeepMind's .bag file format."""

    def __init__(self, path: str):
        self.path = path
        with open(path, 'rb') as f:
            f.seek(-8, 2)
            self.file_size = f.tell() + 8
            self.index_start = struct.unpack('<Q', f.read(8))[0]

        self.index_size = self.file_size - 8 - self.index_start
        self.num_records = self.index_size // 8

        # Memory-map the file for efficient random access
        import mmap
        self._file = open(path, 'rb')
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return self.num_records

    def _get_limit(self, idx):
        """Read a single limit from the index section."""
        offset = self.index_start + idx * 8
        return struct.unpack('<Q', self._mmap[offset:offset + 8])[0]

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.num_records
        start = 0 if idx == 0 else self._get_limit(idx - 1)
        end = self._get_limit(idx)
        return self._mmap[start:end]

    def __del__(self):
        if hasattr(self, '_mmap'):
            self._mmap.close()
        if hasattr(self, '_file'):
            self._file.close()

    def decode_behavioral_cloning(self, idx) -> Tuple[str, str]:
        """Decode a behavioral cloning record into (fen, move)."""
        data = self[idx]
        # Format: varint(fen_length) + fen_bytes + move_bytes
        fen_len, pos = self._read_varint(data, 0)
        fen = data[pos:pos + fen_len].decode('ascii')
        move = data[pos + fen_len:].decode('ascii')
        return fen, move

    @staticmethod
    def _read_varint(buf, pos):
        result = 0
        shift = 0
        while pos < len(buf):
            b = buf[pos]
            result |= (b & 0x7f) << shift
            pos += 1
            if (b & 0x80) == 0:
                return result, pos
            shift += 7
        return result, pos


def convert_record(fen: str, uci_move: str) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
    """Convert a single (FEN, move) record to training data.

    Returns:
        (obs_phase0, obs_phase1, action0, action1) or None if invalid.
    """
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)

        # Validate the move is legal
        if move not in board.legal_moves:
            return None

        obs_phase0 = fen_to_obs(fen)
        action0, action1 = uci_to_actions(uci_move, fen)
        obs_phase1 = fen_to_obs_phase1(fen, move.from_square)

        return obs_phase0, obs_phase1, action0, action1
    except Exception:
        return None


def convert_bag_to_numpy(bag_path: str, output_dir: str,
                          chunk_size: int = 100_000,
                          max_records: int = None):
    """Convert a .bag file to numpy arrays for training.

    Produces files: obs_phase0_N.npy, obs_phase1_N.npy, actions_N.npy
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    reader = BagReader(bag_path)
    total = min(len(reader), max_records) if max_records else len(reader)

    obs0_buf = np.zeros((chunk_size, OBS_SIZE), dtype=np.uint8)
    obs1_buf = np.zeros((chunk_size, OBS_SIZE), dtype=np.uint8)
    act_buf = np.zeros((chunk_size, 2), dtype=np.int64)

    buf_idx = 0
    chunk_num = 0
    skipped = 0

    for i in range(total):
        if i % 10000 == 0:
            print(f"Processing {i}/{total} (skipped {skipped})...")

        fen, move = reader.decode_behavioral_cloning(i)
        result = convert_record(fen, move)

        if result is None:
            skipped += 1
            continue

        obs_p0, obs_p1, a0, a1 = result
        obs0_buf[buf_idx] = obs_p0
        obs1_buf[buf_idx] = obs_p1
        act_buf[buf_idx] = [a0, a1]
        buf_idx += 1

        if buf_idx >= chunk_size:
            _save_chunk(output_dir, chunk_num, obs0_buf, obs1_buf, act_buf, buf_idx)
            chunk_num += 1
            buf_idx = 0

    # Save remaining
    if buf_idx > 0:
        _save_chunk(output_dir, chunk_num, obs0_buf[:buf_idx], obs1_buf[:buf_idx], act_buf[:buf_idx], buf_idx)

    print(f"Done. {total - skipped} records converted, {skipped} skipped, {chunk_num + 1} chunks.")


def _save_chunk(output_dir, chunk_num, obs0, obs1, acts, count):
    import os
    np.save(os.path.join(output_dir, f"obs_phase0_{chunk_num:04d}.npy"), obs0[:count])
    np.save(os.path.join(output_dir, f"obs_phase1_{chunk_num:04d}.npy"), obs1[:count])
    np.save(os.path.join(output_dir, f"actions_{chunk_num:04d}.npy"), acts[:count])
    print(f"  Saved chunk {chunk_num}: {count} records")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python fen_converter.py <bag_path> <output_dir> [max_records]")
        sys.exit(1)

    bag_path = sys.argv[1]
    output_dir = sys.argv[2]
    max_records = int(sys.argv[3]) if len(sys.argv) > 3 else None

    convert_bag_to_numpy(bag_path, output_dir, max_records=max_records)
