import os
import shutil
import numpy as np
import chess
import chess.engine
from typing import Tuple, List, Any, Dict

# NOTE: We deliberately avoid any heavyweight imports (torch, pufferlib) here to
# keep the engine wrapper usable in minimal contexts such as unit tests.

# -----------------------------------------------------------------------------
# Helper – OpenSpiel / AlphaZero move-to-action encoding replicated in Python
# -----------------------------------------------------------------------------

_K_NUM_DEST = 73                 # queen-moves + knight destinations
_K_NUM_UNDER = 9                 # 3 promo pieces × 3 directions

_UNDER_PROMO_ORDER = [chess.KNIGHT, chess.BISHOP, chess.ROOK]  # N, B, R
_DIR_MAP = {-1: 0, 0: 1, 1: 2}  # capture left / straight / capture right


def _move_to_action(move: chess.Move, board: chess.Board) -> int:
    """Convert a python-chess move into the 0-4673 action id used by the C++ core.

    The implementation faithfully mirrors ChessBoard::move_to_action in
    pufferlib/ocean/chess/chess.h so that we can translate Stockfish moves
    without touching the C++ side.
    """
    # Pass move (not used in standard chess)
    # ------------------------------------------------------------------
    if move == chess.Move.null():
        return 0

    # Castling – rely on python-chess helper; orientation does *not* matter
    # ------------------------------------------------------------------
    if board.is_kingside_castling(move):
        return 4673  # right – short castle
    if board.is_queenside_castling(move):
        return 4672  # left  – long castle

    from_sq = move.from_square
    to_sq = move.to_square

    # Coordinates: file (x) 0=a .. 7=h, rank (y) 0=1st .. 7=8th
    fx, fy = chess.square_file(from_sq), chess.square_rank(from_sq)
    tx, ty = chess.square_file(to_sq), chess.square_rank(to_sq)

    moving_piece_color = board.piece_at(from_sq).color

    # Rotate board so mover is always WHITE (see C++ code)
    if moving_piece_color == chess.BLACK:
        fy = 7 - fy
        ty = 7 - ty

    dx = tx - fx
    dy = ty - fy

    from_base = (fx * 8 + fy) * _K_NUM_DEST

    # ------------------------------------------------------------------
    # Under-promotions (to N/B/R) – handled *before* normal queen promo
    # ------------------------------------------------------------------
    if move.promotion and move.promotion != chess.QUEEN:
        promo_index = _UNDER_PROMO_ORDER.index(move.promotion)
        dir_index = _DIR_MAP.get(np.sign(dx), 1)  # -1, 0, +1 → 0,1,2
        return from_base + promo_index * 3 + dir_index

    # ------------------------------------------------------------------
    # Queen-style and knight moves
    # ------------------------------------------------------------------
    dest_index = -1

    if dx == 0 and dy != 0:  # vertical N/S
        if dy > 0:
            dest_index = (dy - 1)  #   0–6
        else:
            dest_index = 28 + (-dy - 1)  # 28–34

    elif dy == 0 and dx != 0:  # horizontal E/W
        if dx > 0:
            dest_index = 14 + (dx - 1)  # 14–20
        else:
            dest_index = 42 + (-dx - 1)  # 42–48

    elif dx == dy != 0:  # main diagonal NE/SW
        if dx > 0:
            dest_index = 7 + (dx - 1)  # 7–13
        else:
            dest_index = 35 + (-dx - 1)  # 35–41

    elif dx == -dy != 0:  # anti-diagonal SE/NW
        if dx > 0:
            dest_index = 21 + (dx - 1)  # 21–27
        else:
            dest_index = 49 + (-dx - 1)  # 49–55

    else:
        # Knight moves – enumerate fixed offsets
        knight_offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                          (2, -1), (2, 1), (1, -2), (1, 2)]
        try:
            k_idx = knight_offsets.index((dx, dy))
        except ValueError:
            raise ValueError(f"Illegal or unsupported move {move.uci()} for mapping.")
        dest_index = 56 + k_idx  # 56–63

    if dest_index < 0:
        raise ValueError(f"Failed to map move {move.uci()} (dx={dx}, dy={dy}).")

    return from_base + _K_NUM_UNDER + dest_index


# -----------------------------------------------------------------------------
# ChessEngineOpponentWrapper – plug-in Stockfish (or any UCI engine) as Black
# -----------------------------------------------------------------------------

class ChessEngineOpponentWrapper:
    """Wrap a pufferlib Chess environment so the *white* RL agent plays vs
    Stockfish (black).  The wrapper keeps all external interfaces identical to
    the underlying VecEnv: from the agent's POV nothing changes – each `step`
    still corresponds to *one* white move, rewards are from white's perspective.
    """

    def __init__(self, env, engine_path: str = "stockfish", depth: int = 2):
        """Create a ChessEngineOpponentWrapper.

        Parameters
        ----------
        env : pufferlib VecEnv
            The white‐to‐play environment instance.
        engine_path : str, optional
            Either the path to a UCI‐compatible chess engine executable or
            the command name if the binary is available on the current $PATH.
            By default we try to launch "stockfish".
        depth : int, optional
            Search depth that will be requested from the engine for every
            black reply.
        """

        self.env = env
        self.depth = depth

        # --------------------------------------------------------------
        # 1. Resolve engine executable
        # --------------------------------------------------------------
        resolved_path = engine_path

        # If the caller did not specify an explicit path, attempt to locate
        # a suitable binary in the current $PATH.  This mirrors the behaviour
        # of the stockfish command‐line utility and avoids obscure
        # FileNotFoundError stack traces when users forget to install the
        # engine first.
        if engine_path == "stockfish":
            resolved_path = shutil.which("stockfish")

            # Look for a vendored binary inside the repository (CI friendly)
            if resolved_path is None:
                local_bin = os.path.join(os.path.dirname(__file__), "stockfish")
                if os.path.isfile(local_bin) and os.access(local_bin, os.X_OK):
                    resolved_path = local_bin

        if resolved_path is None:
            # Additional fall-back search – mirror the candidate list used on the C++ side
            # so Python and C++ locate the same bundled binary without requiring the user
            # to modify $PATH.
            search_roots = [
                os.getcwd(),                                     # current working dir
                os.path.dirname(__file__),                       # …/pufferlib/ocean/chess
                os.path.dirname(os.path.dirname(__file__)),      # …/pufferlib/ocean
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # …/pufferlib
            ]
            candidates = []
            for root in search_roots:
                candidates.extend([
                    os.path.join(root, "pufferlib/Stockfish/src/stockfish"),
                    os.path.join(root, "Stockfish/src/stockfish"),
                ])

            for cand in candidates:
                if os.path.isfile(cand) and os.access(cand, os.X_OK):
                    resolved_path = cand
                    break

        if resolved_path is None:
            raise FileNotFoundError(
                "Could not find a Stockfish engine binary. "
                "Please install Stockfish (e.g. `sudo apt install stockfish`) "
                "or pass `--engine /path/to/stockfish` when launching the script."
            )

        # --------------------------------------------------------------
        # 2. Start UCI engine via python‐chess helper
        # --------------------------------------------------------------
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(resolved_path)
        except FileNotFoundError as e:
            # Provide a more actionable error message before bubbling up
            raise FileNotFoundError(
                f"Failed to launch UCI engine at '{resolved_path}'. "
                "Ensure the file exists and is executable, or specify the "
                "correct path via the --engine CLI flag.") from e

        # Internal python-chess board – mirrors C++ board so we can translate
        # engine moves without expensive C++ ↔ Python bridging.
        self.board = chess.Board()

        # Track whose turn the wrapper expects (False=white to play, True=black)
        self.black_turn = False

    # ------------------------------------------------------------------
    # Utility: convert env action → python-chess Move (for board sync)
    # ------------------------------------------------------------------
    def _action_to_move(self, action_id: int) -> chess.Move:
        """Find the legal move on the *current* board that maps to action_id."""
        for m in self.board.legal_moves:
            if _move_to_action(m, self.board) == action_id:
                return m
        raise ValueError(f"No matching move for action {action_id} in current position.")

    # ------------------------------------------------------------------
    # VecEnv-style API
    # ------------------------------------------------------------------
    def reset(self, seed: int = 0) -> Tuple[np.ndarray, List[Any]]:
        # Sync both environments / boards
        obs, info = self.env.reset(seed=seed)
        self.board.reset()
        self.black_turn = False
        return obs, info

    def step(self, white_actions: np.ndarray):
        """Accepts WHITE move from the agent, then plays Stockfish's BLACK reply."""
        # --- 1. White move (agent) -------------------------------------
        obs, reward, done, trunc, info = self.env.step(white_actions)

        # Update python board with the agent's move so engine sees new position
        if not done.any():
            white_move = self._action_to_move(int(white_actions[0]))
            self.board.push(white_move)

            # --- 2. Black (engine) reply ----------------------------------
            result = self.engine.play(self.board, chess.engine.Limit(depth=self.depth))
            black_move = result.move
            black_action = _move_to_action(black_move, self.board)

            # Execute black move in C++ env (mover's perspective reward)
            obs, black_reward, done2, trunc2, info2 = self.env.step(np.array([black_action], dtype=np.int32))

            # Update board
            self.board.push(black_move)

            # The env returned reward from Black's POV; convert to White
            reward = reward - black_reward  # white gain minus black gain
            done |= done2
            trunc |= trunc2
            # Merge info dicts if present
            info.extend(info2)

        return obs, reward, done, trunc, info

    # --------------------------------------------------------------
    # Attribute forwarding so the wrapper behaves like the base env
    # --------------------------------------------------------------
    def __getattr__(self, name):
        return getattr(self.env, name)

    # Graceful shutdown
    def close(self):
        try:
            self.engine.quit()
        finally:
            self.env.close() 