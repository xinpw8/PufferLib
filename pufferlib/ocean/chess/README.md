open-spiel bloat assessment:
framework: ~60%
extra features: ~30%
abseil vs stl: ~10%

Features to keep:

Zobrist Hashing: 
Open-spiel implementation: open_spiel/games/chess/chess_board.cc uses a Zobrist table (kZobristTable) to compute a 64-bit hash for board positions. The SetBoard and ApplyMove functions update this hash incrementally as pieces move using XOR operations. Hashes are stored.
Reason to keep: lightweight threefold repetition (==identical game state has occurred 3 times at any point during game) detection.

PassthroughHash:
Open-spiel implementation: open_spiel/games/chess/chess_board.h defines a RepetitionTable as absl::flat_hash_map<uint64_t, int, Zobrist::PassthroughHash>, a struct which returns the key itself. The lookup key is therefore the Zobrist hash, making it unnecessary to apply a second (redundant) hashing function.
Reason to keep: threefold repetition is expensive to detect.

Legal-move caching:
Open-spiel implementation: open_spiel/games/chess/chess.h ChessState class's variable mutable absl::optional<std::vector<Move>> cached_legal_actions_ stores a list of all legal moves. LegalActions method checks if it's filled and populates with GenerateLegalMoves if not. Any action modifying the game state (e.g. DoApplyAction) invalidates the cache by calling cached_legal_actions_.reset(). Cached legal moves prevent regenerating the list repeatedly for a given game state.
Reason to keep: generating legal moves is computationally expensive, especially if it is necessary to query them multiple times for a given game state.

Yield-based move generation:
Open-spiel implementation: open_spiel/games/chess/chess_board.cc uses a callback-based approach to generate moves. The primary function is GenerateLegalMoves(const std::function<void(const Move&)>& yield), which takes a function (yield) as an argument. Pseudo-legal moves are generated one at a time and passed to yield, allowing quick determination if any legal move exists, since it can return early on first legal move.
Reason to keep: avoids expensive computation of moves.

Simple array board representation:
Open-spiel implementation: open_spiel/games/chess/chess_board.h represents the board as std::array<Piece, kBoardSize> board_; kBoardSize is 64. Each element holds the piece occupying that square.
Reason to keep: avoids bitboard alternative.

Game-state representation:
Open-spiel implementation: open_spiel/games/chess/chess.cc uses ObservationTensor method to create a flat vector representing a multi-layered tensor of shape {kObservationTensorShape[0], kObservationTensorShape[1], kObservationTensorShape[2]}, defined as {8, 8, 21}. Each 8x8 plane represents a piece type (white pawn, black pawn, white bishop, etc.), repetition count, castling rights, and whose turn it is.
Reason to keep: rich feature engineering allows effective learning.

Fixed 4674-Dimensional Action Space:
Open-spiel implementation: open_spiel/games/chess/chess.h sets kNumDistinctActions to 4674. open_spiel/games/chess/chess.cc uses MoveToAction and ActionToMove to encode all possible moves (including all possible pawn promotions, UCI-style long algebraic notation for other moves, all from/to square combinations) into a single integer between 0 and 4673. 
Reason to keep: avoids structured actions (e.g. piece, then destination); standard RL approach for chess.
(flat integer encoding represents all possiblemoves in any chess position.)
