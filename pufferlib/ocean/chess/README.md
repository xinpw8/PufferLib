open-spiel bloat assessment:
framework: ~60%
extra features: ~30%
abseil vs stl: ~10%

Features to keep:

Zobrist Hashing: 
Open-spiel implementation: open_spiel/games/chess/chess_board.cc uses a Zobrist table (kZobristTable) to compute a 64-bit hash for board positions. The SetBoard and ApplyMove functions update this hash incrementally as pieces move using XOR operations. Hashes are stored.
Reason to keep: lightweight threefold repetition (==identical game state has occurred 3 times at any point during game) detection.

How implemented:
    A 64-bit hash is stored in the ChessBoard struct: uint64_t zobrist_hash;.

    Zobrist tables are initialized in init_zobrist_tables(void).

    The hash is computed from scratch only on initialization with compute_zobrist_hash(const ChessBoard* board).

    Most importantly, the hash is updated incrementally with XOR operations inside apply_uci_move (which uses a simplified helper that has since been expanded upon, but the principle is shown in update_zobrist_hash), which updates the hash based on the piece moved, piece captured, castling rights changes, and en-passant status.

PassthroughHash:
Open-spiel implementation: open_spiel/games/chess/chess_board.h defines a RepetitionTable as absl::flat_hash_map<uint64_t, int, Zobrist::PassthroughHash>, a struct which returns the key itself. The lookup key is therefore the Zobrist hash, making it unnecessary to apply a second (redundant) hashing function.
Reason to keep: threefold repetition is expensive to detect.

How implemented:
    The PositionHistory struct (typedef struct { uint64_t hashes[POSITION_HISTORY_SIZE]; int counts[POSITION_HISTORY_SIZE]; ... }) stores the raw uint64_t Zobrist hashes directly in an array.

    The lookup function get_position_count(ChessContext* ctx, uint64_t hash) performs a linear scan over this array, comparing the input hash directly with the stored hashes: if (history->hashes[i] == hash).

Legal-move caching:
Open-spiel implementation: open_spiel/games/chess/chess.h ChessState class's variable mutable absl::optional<std::vector<Move>> cached_legal_actions_ stores a list of all legal moves. LegalActions method checks if it's filled and populates with GenerateLegalMoves if not. Any action modifying the game state (e.g. DoApplyAction) invalidates the cache by calling cached_legal_actions_.reset(). Cached legal moves prevent regenerating the list repeatedly for a given game state.
Reason to keep: generating legal moves is computationally expensive, especially if it is necessary to query them multiple times for a given game state.

How implemented:
    The ChessContext struct contains all the necessary components: char legal_moves_buffer[256][6], int legal_moves_count, bool legal_moves_cached, and uint64_t cached_board_hash.

    The main move generation function, chess_generate_legal_moves_uci, explicitly checks the cache at the beginning: if (ctx->legal_moves_cached && ctx->cached_board_hash == current_hash) { return ctx->legal_moves_count; }.

    After making a move, apply_uci_move invalidates the cache by setting ctx->legal_moves_cached = false;.

Yield-based move generation:
Open-spiel implementation: open_spiel/games/chess/chess_board.cc uses a callback-based approach to generate moves. The primary function is GenerateLegalMoves(const std::function<void(const Move&)>& yield), which takes a function (yield) as an argument. Pseudo-legal moves are generated one at a time and passed to yield, allowing quick determination if any legal move exists, since it can return early on first legal move.
Reason to keep: avoids expensive computation of moves.

How implemented:
    A function pointer type is defined for the callback: typedef bool (*MoveYieldCallback)(ChessContext* ctx, const ChessMove* move, void* user_data);.

    The function chess_generate_legal_moves_yield(ChessContext* ctx, MoveYieldCallback yield_fn, void* user_data) takes this callback as an argument.

    Inside its loop, it calls the yield_fn and checks its return value to terminate early: if (yield_fn(ctx, &temp_moves.moves[i], user_data)) { return true; }.

    This is used in c_step to efficiently check for game-ending conditions (checkmate/stalemate) by passing first_move_callback, which returns true immediately after finding the first legal move.

Simple array board representation:
Open-spiel implementation: open_spiel/games/chess/chess_board.h represents the board as std::array<Piece, kBoardSize> board_; kBoardSize is 64. Each element holds the piece occupying that square.
Reason to keep: avoids bitboard alternative.

How implemented:
    The ChessBoard struct contains the member: Piece board[64];.

Game-state representation:
Open-spiel implementation: open_spiel/games/chess/chess.cc uses ObservationTensor method to create a flat vector representing a multi-layered tensor of shape {kObservationTensorShape[0], kObservationTensorShape[1], kObservationTensorShape[2]}, defined as {8, 8, 21}. Each 8x8 plane represents a piece type (white pawn, black pawn, white bishop, etc.), repetition count, castling rights, and whose turn it is.
Reason to keep: rich feature engineering allows effective learning.

How implemented:
    The function compute_observation_with_perspective builds this exact tensor. A breakdown of the planes it creates:

        6 planes for the current player's pieces (P, N, B, R, Q, K)

        6 planes for the opponent's pieces

        1 plane for empty squares

        1 plane for the repetition count


        1 plane for the side-to-move (always 0, as it's from the current player's perspective)

        1 plane for the 50-move rule clock

        4 planes for castling rights (WK, WQ, BK, BQ, flipped for perspective)

        1 plane for the en-passant square (flipped for perspective)

    Total: 6 + 6 + 1 + 1 + 1 + 1 + 4 + 1 = 21 planes. The function writes 21 * 8 * 8 = 1344 floats to the observation buffer.

Fixed 1924-Dimensional UCI Action Space:
PufferLib implementation: Uses UCI-based action encoding with 1924 total actions covering all possible chess moves. Each action represents a unique UCI move string (e.g., "e2e4", "e7e8q"). This is more compact than OpenSpiel's 4674 actions while maintaining full chess move coverage.
Reason to keep: avoids structured actions (e.g. piece, then destination); standard RL approach for chess.
(flat integer encoding represents all possible moves in any chess position using UCI notation.)

How implemented:
    The code includes "chess_action_mapping.h", which defines the mappings uci_to_action_id and ACTION_ID_TO_UCI.

    The constant TOTAL_CHESS_ACTIONS (which is 1924) is used throughout the code for action masking and validation (e.g., in compute_observation_with_perspective and c_step).

    The allocate function explicitly reserves space for a 1924-dimensional action space.

New Sparse Representation Action Mask
  - Board planes from indices 0-1471 (23 * 8 * 8 = 1472)
  - Sparse mask: count at index 1472, action IDs from 1473-1536 (64 max)

UPDATED SECTION - IMPORTANT!!
Regarding self-play actually working:
best way to do masking is half the agents run 1 step and half run the next step. but they can't go in the same     │
│   episode. an episode has to be all of 1 agent's moves. data from each agent must remain separate. the actual fix    │
│   is for each episode to contain 1 player's data. hack double-buffered vec so 1 call to step you get 1 player's      │
│   data and then on the next call to step() you get the other player's data.  


Legal Move Mask: (exact indices may be off if things were added/removed from obs)
the last 1968 elements of the observations array is a legal move mask, generated in the env as such: 
obs[:, 1344:3312]

models.py decode_actions() generates raw logits for all 1968 possible moves
raw_logits = self.decoder(hidden)
, 
forward_eval() extracts the legal move mask from the env,
legal_move_mask = obs[:, 1344:3312]
,
decode_actions() sets the illegal logits to -1e8 ~ -inf
logits = raw_logits.masked_fill(legal_move_mask < 0.5, -1e8)
,
pytorch.py sample_logits() samples from the masked logits
action = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, replacement=True).int()



Opening frequency of PGN-style UCI logs:
python analyze_openings.py /puffertank/release_test_pufferlib/pufferlib/resources/chess/training_logs/complete_games

chess.cpp commands:
    Usage:
    ./chess demo     - Interactive chess demo
    ./chess console  - Console chess demo
    ./chess browser  - Browse and view training games
    ./chess          - Run performance test