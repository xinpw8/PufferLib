# chess.py
import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.chess import binding

class Chess(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=1,
                 reward_move_valid=0.0005, reward_move_invalid=-0.0001,
                 reward_player_capture=0.1, reward_opponent_capture=-0.01,
                 reward_win=1.0, reward_draw=-1.0, reward_loss=-1.0,
                 buf=None, seed=0, **_ignored):
        # One-hot board encoding: 12 channels per square (6 piece types × 2 colours).
        # Flattened as 64 × 12 = 768-dimensional vector in row-major order.
        # Plus 256-element legal moves mask.
        # Plus 1024-element move encodings (256 moves × 4 coordinates each).
        self.single_observation_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(64 * 12 + 256 + 256 * 4,),  # 768 + 256 + 1024 = 2048
            dtype=np.float32,
        )
        
        # Chess actions: Direct indexing into legal moves array
        # Maximum 256 possible legal moves in any chess position
        self.single_action_space = gymnasium.spaces.Discrete(256)
        
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self._prev_terminals = 0  # track completed games for debug printing

        # Silently ignore extraneous keyword args passed by generic config files
        # (e.g., reward_move_pass). They are not needed by this environment.
        self._moves = []
        self._move_history = []

        # Initialize parent PufferEnv first to set up arrays
        super().__init__(buf)
        
        # Use the standard vec_init from env_binding.h
        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed,
            reward_move_valid=reward_move_valid,
            reward_move_invalid=reward_move_invalid,
            reward_player_capture=reward_player_capture,
            reward_opponent_capture=reward_opponent_capture,
            reward_win=reward_win,
            reward_draw=reward_draw,
            reward_loss=reward_loss,
        )

        # Internal flag tracking whether the current episode has finished.
        # We expose this via a @property named ``done`` so that PufferLib's
        # Serial / Multiprocessing vectorisers can correctly reset the board
        # when an episode ends.
        self._done = False

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        self._done = False
        return self.observations, []

    def step(self, actions):
        self.tick += 1

        # Debug prints disabled

        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            log_data = binding.vec_log(self.c_envs)
            if log_data:
                info.append(log_data)

        # Update terminal count tracking
        self._prev_terminals = int(np.count_nonzero(self.terminals))

        # If any instance reached terminal, mark the env as done so that the
        # vector wrapper knows to call reset on the next cycle.
        self._done = bool(self._prev_terminals)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        # If render_mode is ANSI (text), return an ASCII board representation
        if self.render_mode in (None, 'ansi', 'auto'):
            # Extract the observation for the first env instance and reshape
            if self.observations.ndim == 1:
                raw_obs = self.observations
            else:
                # 2-D or 3-D → first env instance
                raw_obs = self.observations[0]
            board_vec = raw_obs[:768].reshape(64, 12)
            
            # Get legal moves info from observation
            legal_moves_mask = raw_obs[768:1024]
            move_encodings = raw_obs[1024:2048].reshape(256, 4)

            # Map piece codes to ASCII symbols
            symbols = {
                0: '.',
                1: 'K', 2: 'Q', 3: 'R', 4: 'B', 5: 'N', 6: 'P',
                -1: 'k', -2: 'q', -3: 'r', -4: 'b', -5: 'n', -6: 'p',
            }

            def _slice_to_code(slice12):
                idx = int(np.argmax(slice12))
                if slice12[idx] <= 0.5:
                    return 0  # empty
                if idx < 6:
                    return idx + 1  # white pieces 1-6
                return -(idx - 5)    # black pieces −1 to −6

            lines = []
            for rank in range(7, -1, -1):  # 7 → 0 for ranks 8→1
                row_syms = []
                for file in range(8):
                    sq_idx = rank * 8 + file
                    code = _slice_to_code(board_vec[sq_idx])
                    row_syms.append(symbols.get(code, '.'))
                lines.append(f"{rank + 1} " + ' '.join(row_syms))
            lines.append('  a b c d e f g h')
            
            # Show some legal moves
            lines.append("\nLegal moves (first 5):")
            files = 'abcdefgh'
            for i in range(min(5, int(np.sum(legal_moves_mask)))):
                if legal_moves_mask[i] > 0:
                    # Decode move from encoding (denormalize coordinates)
                    from_x = int(move_encodings[i, 0] * 7 + 0.5)
                    from_y = int(move_encodings[i, 1] * 7 + 0.5)
                    to_x = int(move_encodings[i, 2] * 7 + 0.5)
                    to_y = int(move_encodings[i, 3] * 7 + 0.5)
                    move_str = f"{files[from_x]}{from_y+1}{files[to_x]}{to_y+1}"
                    lines.append(f"  [{i}] {move_str}")
                    self._moves.append(move_str)
            
            # Append move history
            if self._moves:
                lines.append("\nLast moves: " + ' '.join(self._moves[-10:]))

            # If game just terminated, archive moves and reset
            if self.terminals[0]:
                lines.append("\nGame finished.")
                self._move_history.append(self._moves)
                self._moves = []

            return '\n'.join(lines)

        # Otherwise, fall back to C++ render (e.g., Raylib GUI)
        binding.vec_render(self.c_envs, 0)
        return None

    def close(self):
        binding.vec_close(self.c_envs)

    @staticmethod
    def encode_move(from_square, to_square):
        """Encode a chess move as an integer action.
        
        Args:
            from_square: Source square index (0-63)
            to_square: Destination square index (0-63)
            
        Returns:
            Integer action encoding the move
        """
        return (from_square << 6) | to_square
    
    @staticmethod
    def decode_move(action):
        """Decode an integer action into chess move squares.
        
        Args:
            action: Integer action to decode
            
        Returns:
            tuple: (from_square, to_square) both 0-63
        """
        from_square = (action >> 6) & 63
        to_square = action & 63
        return from_square, to_square
    
    @staticmethod
    def square_to_coord(square_idx):
        """Convert square index to (x, y) coordinates.
        
        Args:
            square_idx: Square index (0-63)
            
        Returns:
            tuple: (x, y) coordinates where (0,0) is bottom-left
        """
        return square_idx % 8, square_idx // 8
    
    @staticmethod
    def coord_to_square(x, y):
        """Convert (x, y) coordinates to square index.
        
        Args:
            x: File coordinate (0-7)
            y: Rank coordinate (0-7)
            
        Returns:
            Square index (0-63)
        """
        return y * 8 + x

    # ------------------------------------------------------------------
    # PufferLib expects native envs to expose a ``done`` attribute that
    # flips to True once an episode ends; the vectorised wrappers check
    # this flag to decide when to call reset().  We expose it as a read-
    # only property backed by ``self._done``.
    # ------------------------------------------------------------------
    @property
    def done(self):
        return self._done

if __name__ == '__main__':
    N = 64  # Smaller batch size for chess since games are longer

    env = Chess(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 256
    actions = np.random.randint(0, 256, (CACHE, N))  # Changed from 4096 to 256

    i = 0
    import time
    start = time.time()
    timeout = 10
    while time.time() - start < timeout:
        env.step(actions[i % CACHE])
        steps += N
        i += 1

    sps = int(steps / (time.time() - start))
    print(f'Chess SPS: {sps:,}')
    print(f'That\'s {sps//N:,} games per second with {N} parallel environments')
    
    env.close()