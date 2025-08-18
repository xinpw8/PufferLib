// // chess.h - Complete Pure C Chess Environment for PufferLib
// // Optimized for 150k+ SPS performance with single network self-play

// #ifndef CHESS_H
// #define CHESS_H

// #include <assert.h>
// #include <errno.h>
// #include <stdbool.h>
// #include <stdint.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <time.h>

// // Enable debug logging for development (set to 0 to disable)
// #ifndef DEBUG_LOG
// #define DEBUG_LOG 0
// #endif

// #if DEBUG_LOG
// #define DBG(expr) printf("%s", expr)
// #else
// #define DBG(expr)                                                              \
//   do {                                                                         \
//   } while (0)
// #endif

// // === PROFILING GLOBALS ===
// clock_t profile_total_ticks = 0;
// clock_t profile_c_step_ticks = 0;
// clock_t profile_move_gen_uci_ticks = 0;
// clock_t profile_is_legal_move_ticks = 0;
// clock_t profile_is_square_attacked_ticks = 0;
// clock_t profile_make_move_fast_ticks = 0;
// clock_t profile_unmake_move_fast_ticks = 0;
// clock_t profile_apply_uci_move_ticks = 0;
// clock_t profile_compute_obs_ticks = 0;

// #define PROFILE_START(counter) clock_t start_##counter = clock();
// #define PROFILE_STOP(counter) counter += clock() - start_##counter;

// #ifdef __cplusplus
// extern "C" {
// #endif

// // Include UCI action mapping
// #include "chess_action_mapping.h"

// // === CORE CHESS TYPES ===

// typedef enum {
//   EMPTY = 0,
//   KING = 1,
//   QUEEN = 2,
//   ROOK = 3,
//   BISHOP = 4,
//   KNIGHT = 5,
//   PAWN = 6
// } PieceType;

// typedef enum { C_WHITE = 0, C_BLACK = 1, C_NO_COLOR = 2 } PieceColor;

// typedef struct {
//   PieceColor color;
//   PieceType type;
// } Piece;

// typedef struct {
//   int8_t x, y;
// } Square;

// // === PUFFERLIB LOG STRUCTURE ===
// typedef struct Log {
//   float perf;
//   float score;
//   float episode_length;
//   float episode_return;       // combined (existing)
//   float episode_return_white; // new – white perspective total
//   float episode_return_black; // new – black perspective total
//   float reward_valid;
//   float reward_white_captures_enemy_piece;
//   float reward_black_captures_enemy_piece;
//   float reward_max_depth_termination;
//   float reward_draw;
//   // Perspective-based reward tracking
//   float reward_win_white;  // win rewards from white's perspective
//   float reward_win_black;  // win rewards from black's perspective
//   float reward_loss_white; // loss rewards from white's perspective
//   float reward_loss_black; // loss rewards from black's perspective
//   float reward_draw_white; // draw rewards from white's perspective
//   float reward_draw_black; // draw rewards from black's perspective
//   float game_drawn;
//   // New separate win/loss tracking from both perspectives
//   float white_win;  // white wins (from white's perspective)
//   float white_loss; // white losses (from white's perspective)
//   float black_win;  // black wins (from black's perspective)
//   float black_loss; // black losses (from black's perspective)
//   float stalemate;
//   float insufficient_material;
//   float threefold_repetition;
//   float fifty_move_rule;
//   float max_depth;
//   float white_checkmated; // black checkmates white
//   float black_checkmated; // white checkmates black
//   float white_moves;
//   float black_moves;
//   float valid_moves;
//   float invalid_moves_white;
//   float invalid_moves_black;
//   float reward_check_white;
//   float reward_check_black;
//   float reward_material_diff_white;
//   float reward_material_diff_black;
//   float stockfish_eval;
//   // En passant captures
//   float en_passant_white; // white captures via en passant
//   float en_passant_black; // black captures via en passant
//   // Castling moves
//   float white_castle_kingside;  // white castles kingside
//   float white_castle_queenside; // white castles queenside
//   float black_castle_kingside;  // black castles kingside
//   float black_castle_queenside; // black castles queenside
//   // Pawn promotions
//   float white_promotion_count;  // total white pawn promotions
//   float white_promotion_knight; // white promotes to knight
//   float white_promotion_bishop; // white promotes to bishop
//   float white_promotion_rook;   // white promotes to rook
//   float white_promotion_queen;  // white promotes to queen
//   float black_promotion_count;  // total black pawn promotions
//   float black_promotion_knight; // black promotes to knight
//   float black_promotion_bishop; // black promotes to bishop
//   float black_promotion_rook;   // black promotes to rook
//   float black_promotion_queen;  // black promotes to queen

//   // Game logging fields expected by binding.cpp
//   float last_move_from;
//   float last_move_to;
//   float last_move_promotion;
//   float game_step_logged;
//   float game_moves_count;

//   float complete_game_move_count;
//   // Note: complete_game_moves_serialized removed to comply with PufferLib
//   // float-only logging spec

//   // Puzzle mode statistics
//   float puzzle_solved;        // puzzles solved successfully
//   float puzzle_attempts;      // total puzzle attempts (first move of each puzzle)
//   float puzzle_correct_moves; // correct moves made in puzzles
//   float puzzle_wrong_moves;   // wrong moves made in puzzles
//   float puzzle_difficulty;    // current puzzle difficulty level
//   float puzzle_success_rate;  // success rate: solved / attempts
  
//   // New: Performance metrics
//   float puzzle_current_id;       // Current puzzle ID all envs are working on
//   float puzzle_global_attempts;  // Total attempts across all envs for current puzzle
//   float puzzle_global_successes; // Total successes across all envs for current puzzle
//   float puzzle_avg_samples_to_solve; // Average samples needed to solve puzzles
//   float puzzle_avg_time_to_solve;    // Average wall clock time to solve puzzles
  
//   // Puzzle rewards tracking
//   float reward_puzzle_solved;
//   float reward_puzzle_failed;
//   float reward_puzzle_correct_move;

//   // n field is always last per pufferlib spec
//   float n;
// } Log;

// // === CHESS BOARD STATE ===
// typedef struct {
//   Piece board[64];
//   PieceColor to_move;
//   uint8_t castle_rights; // bits: 0=WK, 1=WQ, 2=BK, 3=BQ
//   int8_t ep_square;      // en passant target square (-1 if none)
//   uint8_t halfmove_clock;
//   uint16_t fullmove_number;
//   uint64_t zobrist_hash; // Incrementally updated Zobrist hash
//   bool fen_was_set;      // Track if position was set via FEN
// } ChessBoard;

// // Position history for threefold repetition (simple hash table)
// #define POSITION_HISTORY_SIZE 512
// typedef struct {
//   uint64_t hashes[POSITION_HISTORY_SIZE];
//   int counts[POSITION_HISTORY_SIZE];
//   int size;
// } PositionHistory;

// // === CHESS CONTEXT ===
// typedef struct ChessContext {
//   ChessBoard board;
//   PositionHistory position_history;

//   // Episode tracking
//   int step_count;
//   float episode_return_white;
//   float episode_return_black;

//   // CRITICAL: Performance optimization - DUAL CACHE for both players
//   // White's legal moves cache
//   char white_legal_moves_buffer[256][6];  // White's legal moves in UCI format
//   int white_legal_moves_count;
//   int white_legal_action_ids[256];       // Pre-computed action IDs
//   bool white_moves_cached;
  
//   // Black's legal moves cache
//   char black_legal_moves_buffer[256][6];  // Black's legal moves in UCI format
//   int black_legal_moves_count;
//   int black_legal_action_ids[256];       // Pre-computed action IDs
//   bool black_moves_cached;
  
//   uint64_t cached_board_hash;             // Position hash for both caches
//   bool position_fully_cached;             // True when both players cached

//   // Game modes
//   bool dual_agent_self_play_mode;
//   bool self_play_mode;
//   bool puzzle_mode;
  
//   // Puzzle mode state
//   char puzzle_fen[128];              // Current puzzle's starting FEN
//   char puzzle_solution[10][6];       // Array of solution moves in UCI format
//   int puzzle_solution_length;       // Number of moves in solution
//   int puzzle_move_index;             // Current move in puzzle solution (0-based)
//   bool puzzle_completed;             // True when puzzle solved correctly
//   bool puzzle_failed;                // True when wrong move made
//   int puzzle_attempts_this_env;      // Track attempts for this specific environment
//   int puzzle_solved_this_env;        // Track solves for this specific environment
  
//   // New: Global puzzle training logic
//   int puzzle_tries_this_env;         // Current tries for this env on current puzzle (max 10)
//   int puzzle_max_tries_per_env;      // Maximum tries per env before reset (default 10)
//   clock_t puzzle_start_time;         // Wall clock time when puzzle started
//   int puzzle_samples_to_solve;       // Number of samples taken to solve this puzzle

//   // Complete game logging
//   char complete_game_moves[1024][6]; // Store canonical UCI moves (e.g., "e2e4")
//   int complete_game_action_count;
//   char serialized_moves[1024]; // Comma-separated action IDs for efficient
//                                // logging
  
//   // Simple game logging tracking
//   int steps_since_last_log;      // Steps in this environment since last game log
//   int game_logging_frequency;    // Log games every N steps (from config)
//   bool log_next_game;            // Flag set by Python layer to trigger logging

//   // Reward configuration (copied from CChess for performance)
//   float c_reward_valid;
//   float c_reward_invalid_white;
//   float c_reward_invalid_black;
//   float c_reward_white_captures_enemy_piece;
//   float c_reward_black_captures_enemy_piece;
//   float c_reward_max_depth_termination;
//   float c_reward_draw;
//   float c_reward_win_white;
//   float c_reward_win_black;
//   float c_reward_loss_white;
//   float c_reward_loss_black;
//   float c_reward_check_white;
//   float c_reward_check_black;
//   float c_reward_material_diff_white;
//   float c_reward_material_diff_black;
//   bool c_use_piece_value_capture_rewards;
//   float c_piece_value_reward_multiplier;
  
//   // Puzzle mode rewards
//   float c_reward_puzzle_solved;
//   float c_reward_puzzle_failed;
//   float c_reward_correct_move;

//   // ACCUMULATED REWARD COUNTERS (for add_log aggregation)
//   float accumulated_reward_valid;
//   float accumulated_reward_white_captures_enemy_piece;
//   float accumulated_reward_black_captures_enemy_piece;
//   float accumulated_reward_draw;
//   float accumulated_reward_win_white;
//   float accumulated_reward_win_black;
//   float accumulated_reward_loss_white;
//   float accumulated_reward_loss_black;
//   float accumulated_reward_draw_white;
//   float accumulated_reward_draw_black;
//   float accumulated_reward_check_white;
//   float accumulated_reward_check_black;
//   float accumulated_reward_material_diff_white;
//   float accumulated_reward_material_diff_black;
//   float accumulated_stockfish_eval;
  
//   // Puzzle reward accumulation
//   float accumulated_reward_puzzle_solved;
//   float accumulated_reward_puzzle_failed;
//   float accumulated_reward_puzzle_correct_move;
  
//   // Puzzle stats accumulation (for this episode)
//   float puzzle_attempts_this_episode;
//   float puzzle_correct_moves_this_episode;
//   float puzzle_wrong_moves_this_episode;
//   float puzzle_solved_this_episode;

//   // Accumulated statistics (for logging)
//   float c_white_moves;
//   float c_black_moves;
//   float c_valid_moves;
//   float c_invalid_moves_white;
//   float c_invalid_moves_black;

//   // Game outcome counters
//   float c_white_win;
//   float c_white_loss;
//   float c_black_win;
//   float c_black_loss;
//   float c_game_drawn;
//   float c_max_depth;

//   // Game end condition counters
//   float c_white_checkmated;
//   float c_black_checkmated;
//   float c_stalemate;
//   float c_insufficient_material;
//   float c_threefold_repetition;
//   float c_fifty_move_rule;
//   float c_en_passant_white;
//   float c_en_passant_black;
//   float c_white_castle_kingside;
//   float c_white_castle_queenside;
//   float c_black_castle_kingside;
//   float c_black_castle_queenside;
//   float c_white_promotion_count;
//   float c_white_promotion_knight;
//   float c_white_promotion_bishop;
//   float c_white_promotion_rook;
//   float c_white_promotion_queen;
//   float c_black_promotion_count;
//   float c_black_promotion_knight;
//   float c_black_promotion_bishop;
//   float c_black_promotion_rook;
//   float c_black_promotion_queen;

//   // Stockfish integration parameters (for future implementation)
//   char stockfish_cmd[256];
//   int stockfish_elo;
//   int stockfish_search_ms;
  
//   // PERFORMANCE OPTIMIZATION: Observation caching
//   float cached_observation[1472];  // 23 * 8 * 8 board planes only
//   bool observation_cached;
//   uint64_t cached_observation_hash;
//   PieceColor cached_observation_player;
// } ChessContext;

// // === PUFFERLIB ENVIRONMENT STRUCTURE ===
// typedef struct CChess {
//   Log log;
//   int env_id;  // Add back env_id field
//   float *observations;
//   int *actions;
//   float *rewards;
//   unsigned char *terminals;

//   // Configuration values from INI file
//   float reward_valid;
//   float reward_invalid_white;
//   float reward_invalid_black;
//   float reward_white_captures_enemy_piece;
//   float reward_black_captures_enemy_piece;
//   float reward_draw;
//   float reward_win_white;
//   float reward_win_black;
//   float reward_loss_white;
//   float reward_loss_black;
//   float reward_check_white;
//   float reward_check_black;
//   int max_depth;
//   float reward_material_diff_white;
//   float reward_material_diff_black;
//   float reward_max_depth_termination;
//   float reward_puzzle_solved;
//   float reward_puzzle_failed;
//   float reward_correct_move;
//   float reward_puzzle_correct_piece;
//   float reward_puzzle_closer_to_target;
//   float reward_puzzle_correct_promotion;
//   bool use_piece_value_capture_rewards;
//   float piece_value_reward_multiplier;

//   // Debug settings
//   bool debug_disable_mask;
//   bool stockfish_enabled;

//   // Chess context (pure C, no opaque pointer)
//   ChessContext context;

//   // Convenience pointer to avoid repeated dereferencing
//   ChessContext *ctx;
//   // Global puzzle coordination across all environments
//   int global_puzzle_id;              // Current puzzle ID all environments work on
//   int global_puzzle_attempts;        // Total attempts across all envs for current puzzle  
//   int global_puzzle_successes;       // Total successes across all envs for current puzzle
//   float global_puzzle_success_threshold; // Threshold to advance (default 0.9)
//   int puzzle_max_tries_per_env;      // Max tries per env before reset (default 10)
// } CChess;

// // === ADDITIONAL BINDING FUNCTIONS ===
// void enable_stockfish_black(CChess *env, const char *stockfish_cmd, int elo,
//                             int search_ms);
// void set_self_play_mode(CChess *env, bool enabled);
// void set_dual_agent_self_play_mode(CChess *env, bool enabled);
// void set_debug_disable_mask(CChess *env, bool enabled);
// void set_puzzle_mode(CChess *env, bool enabled);
// void set_puzzle_data(CChess *env, const char* fen, const char* solution_moves[], int solution_length);
// void set_puzzle_difficulty(CChess *env, int difficulty);
// void set_puzzle_training_params(CChess *env, int max_tries_per_env, float success_threshold);

// // === PUFFERLIB REQUIRED FUNCTIONS ===
// void init(CChess *env);
// void allocate(CChess *env);
// void free_allocated(CChess *env);
// void add_log(CChess *env);
// void c_reset(CChess *env);
// void c_step(CChess *env);
// void c_render(CChess *env);
// void c_close(CChess *env);

// // === MODE SETTERS FOR COMPATIBILITY ===
// void set_dual_agent_self_play_mode(CChess *env, bool enabled);
// void set_self_play_mode(CChess *env, bool enabled);
// void c_set_fen(CChess *env, const char *fen);

// // Stub implementation for notify_game_end
// void notify_game_end(int white_won, int black_won, int is_draw) {
//   // Stub implementation - could be used for logging or callbacks
// }

// // === CHESS HELPER FUNCTIONS ===

// // Print board in human-readable format
// void print_board_state(ChessBoard *board) {
//   printf("\n  a b c d e f g h\n");
//   printf("  ---------------\n");
  
//   for (int y = 7; y >= 0; y--) {
//     printf("%d|", y + 1);
//     for (int x = 0; x < 8; x++) {
//       Piece *p = &board->board[y * 8 + x];
//       char piece_char = ' ';
      
//       if (p->type != EMPTY) {
//         switch (p->type) {
//           case PAWN:   piece_char = 'P'; break;
//           case KNIGHT: piece_char = 'N'; break;
//           case BISHOP: piece_char = 'B'; break;
//           case ROOK:   piece_char = 'R'; break;
//           case QUEEN:  piece_char = 'Q'; break;
//           case KING:   piece_char = 'K'; break;
//         }
        
//         // Lowercase for black pieces
//         if (p->color == C_BLACK) {
//           piece_char = piece_char + 32; // Convert to lowercase
//         }
//       }
      
//       printf("%c ", piece_char);
//     }
//     printf("|%d\n", y + 1);
//   }
  
//   printf("  ---------------\n");
//   printf("  a b c d e f g h\n");
//   printf("Turn: %s\n\n", (board->to_move == C_WHITE) ? "WHITE" : "BLACK");
// }

// // Material calculation using standard chess piece values
// int calculate_material_value(ChessBoard *board, PieceColor color) {
//   int total = 0;
//   for (int i = 0; i < 64; i++) {
//     Piece *p = &board->board[i];
//     if (p->type != EMPTY && p->color == color) {
//       switch (p->type) {
//         case PAWN:   total += 1; break;
//         case KNIGHT: total += 3; break;
//         case BISHOP: total += 3; break;
//         case ROOK:   total += 5; break;
//         case QUEEN:  total += 9; break;
//         case KING:   total += 0; break; // King has no material value
//         default: break;
//       }
//     }
//   }
//   return total;
// }

// // Get individual piece value for capture rewards
// int get_piece_value(PieceType piece_type) {
//   switch (piece_type) {
//     case PAWN:   return 1;
//     case KNIGHT: return 3;
//     case BISHOP: return 3;
//     case ROOK:   return 5;
//     case QUEEN:  return 9;
//     case KING:   return 0; // King cannot be captured in normal play
//     default:     return 0;
//   }
// }

// // Board access
// static inline Piece *get_piece(ChessBoard *board, int x, int y) {
//   if (x < 0 || x >= 8 || y < 0 || y >= 8)
//     return NULL;
//   return &board->board[y * 8 + x];
// }

// static inline const Piece *get_piece_const(const ChessBoard *board, int x,
//                                            int y) {
//   if (x < 0 || x >= 8 || y < 0 || y >= 8)
//     return NULL;
//   return &board->board[y * 8 + x];
// }

// // Square notation conversion
// static inline Square notation_to_square(const char *notation) {
//   Square result;
//   if (!notation || strlen(notation) < 2) {
//     result.x = -1;
//     result.y = -1;
//     return result;
//   }
//   int x = notation[0] - 'a';
//   int y = notation[1] - '1';
//   if (x >= 0 && x < 8 && y >= 0 && y < 8) {
//     result.x = (int8_t)x;
//     result.y = (int8_t)y;
//   } else {
//     result.x = -1;
//     result.y = -1;
//   }
//   return result;
// }

// static inline void square_to_notation(Square sq, char *notation) {
//   if (sq.x >= 0 && sq.x < 8 && sq.y >= 0 && sq.y < 8) {
//     notation[0] = 'a' + sq.x;
//     notation[1] = '1' + sq.y;
//     notation[2] = '\0';
//   } else {
//     strcpy(notation, "--");
//   }
// }

// // Simple hash function for position history
// // Zobrist hash tables for proper position hashing
// static uint64_t zobrist_piece_square[2][7][64]; // [color][piece_type][square]
// static uint64_t zobrist_castle_rights[16];      // [castle_rights]
// static uint64_t zobrist_en_passant[64];         // [ep_square]
// static uint64_t zobrist_side_to_move;
// static bool zobrist_initialized = false;

// static void init_zobrist_tables(void) {
//   if (zobrist_initialized)
//     return;

//   // Simple PRNG for generating Zobrist values
//   uint64_t seed = 0x123456789abcdefULL;

//   for (int color = 0; color < 2; color++) {
//     for (int piece = 0; piece < 7; piece++) {
//       for (int square = 0; square < 64; square++) {
//         seed = seed * 0x9e3779b97f4a7c15ULL + 0x85ebca6b;
//         zobrist_piece_square[color][piece][square] = seed;
//       }
//     }
//   }

//   for (int i = 0; i < 16; i++) {
//     seed = seed * 0x9e3779b97f4a7c15ULL + 0x85ebca6b;
//     zobrist_castle_rights[i] = seed;
//   }

//   for (int i = 0; i < 64; i++) {
//     seed = seed * 0x9e3779b97f4a7c15ULL + 0x85ebca6b;
//     zobrist_en_passant[i] = seed;
//   }

//   seed = seed * 0x9e3779b97f4a7c15ULL + 0x85ebca6b;
//   zobrist_side_to_move = seed;

//   zobrist_initialized = true;
// }

// // Fast hash lookup - just return the incrementally maintained hash
// static inline uint64_t hash_position(const ChessBoard *board) {
//   return board->zobrist_hash;
// }

// // Compute hash from scratch (only used for initialization)
// static uint64_t compute_zobrist_hash(const ChessBoard *board) {
//   if (!zobrist_initialized)
//     init_zobrist_tables();

//   uint64_t hash = 0;

//   // Hash pieces
//   for (int i = 0; i < 64; i++) {
//     if (board->board[i].type != EMPTY) {
//       hash ^=
//           zobrist_piece_square[board->board[i].color][board->board[i].type][i];
//     }
//   }

//   // Hash side to move
//   if (board->to_move == C_BLACK) {
//     hash ^= zobrist_side_to_move;
//   }

//   // Hash castling rights
//   hash ^= zobrist_castle_rights[board->castle_rights & 15];

//   // Hash en passant square
//   if (board->ep_square >= 0 && board->ep_square < 64) {
//     hash ^= zobrist_en_passant[board->ep_square];
//   }

//   return hash;
// }

// // Incrementally update hash when making a move
// static inline void update_zobrist_hash(ChessBoard *board, int from_square,
//                                        int to_square, Piece moved_piece,
//                                        Piece captured_piece,
//                                        uint8_t old_castle_rights,
//                                        int8_t old_ep_square) {
//   if (!zobrist_initialized)
//     init_zobrist_tables();

//   // Remove old piece from from_square
//   board->zobrist_hash ^=
//       zobrist_piece_square[moved_piece.color][moved_piece.type][from_square];

//   // Add piece to to_square
//   board->zobrist_hash ^=
//       zobrist_piece_square[moved_piece.color][moved_piece.type][to_square];

//   // Remove captured piece if any
//   if (captured_piece.type != EMPTY) {
//     board->zobrist_hash ^= zobrist_piece_square[captured_piece.color]
//                                                [captured_piece.type][to_square];
//   }

//   // Update side to move
//   board->zobrist_hash ^= zobrist_side_to_move;

//   // Update castling rights
//   board->zobrist_hash ^= zobrist_castle_rights[old_castle_rights & 15];
//   board->zobrist_hash ^= zobrist_castle_rights[board->castle_rights & 15];

//   // Update en passant
//   if (old_ep_square >= 0 && old_ep_square < 64) {
//     board->zobrist_hash ^= zobrist_en_passant[old_ep_square];
//   }
//   if (board->ep_square >= 0 && board->ep_square < 64) {
//     board->zobrist_hash ^= zobrist_en_passant[board->ep_square];
//   }
// }

// // === GAME LOGGING HELPERS ===

// static void serialize_complete_game_moves(ChessContext *ctx) {
//   ctx->serialized_moves[0] = '\0';

//   if (ctx->complete_game_action_count == 0) {
//     return;
//   }

//   char temp[16];
//   for (int i = 0; i < ctx->complete_game_action_count && i < 100; i++) {
//     if (i > 0) {
//       strcat(ctx->serialized_moves, ",");
//     }
//     // Convert UCI move back to action ID for serialization compatibility
//     int action_id = uci_to_action_id(ctx->complete_game_moves[i]);
//     sprintf(temp, "%d", action_id);
//     strcat(ctx->serialized_moves, temp);
//   }
// }

// // Write complete game to file for analysis
// static void write_complete_game_to_file(ChessContext *ctx, int env_id) {
//   // Re-enabled for debugging game logging functionality
  
//   if (ctx->complete_game_action_count == 0) {
//     return;
//   }

//   // Create directory if needed
//   int mkdir_result = system("mkdir -p pufferlib/resources/chess/training_logs/complete_games");
  
//   // Determine game result and termination reason
//   char result_str[16] = "*";      // Default: incomplete/ongoing
//   char termination[64] = "depth_limit";  // Default termination reason
  
//   if (ctx->c_white_checkmated > 0) {
//     strcpy(result_str, "0-1");
//     strcpy(termination, "white_checkmated");
//   } else if (ctx->c_black_checkmated > 0) {
//     strcpy(result_str, "1-0");
//     strcpy(termination, "black_checkmated");
//   } else if (ctx->c_stalemate > 0) {
//     strcpy(result_str, "1/2-1/2");
//     strcpy(termination, "stalemate");
//   } else if (ctx->c_fifty_move_rule > 0) {
//     strcpy(result_str, "1/2-1/2");
//     strcpy(termination, "fifty_move_rule");
//   } else if (ctx->c_threefold_repetition > 0) {
//     strcpy(result_str, "1/2-1/2");
//     strcpy(termination, "threefold_repetition");
//   } else if (ctx->c_insufficient_material > 0) {
//     strcpy(result_str, "1/2-1/2");
//     strcpy(termination, "insufficient_material");
//   }
  
//   // Generate filename with timestamp, env_id, result and termination
//   time_t now = time(NULL);
//   char filename[256];
  
//   // Create safe result and termination strings for filename
//   char safe_result[16], safe_termination[64];
//   strcpy(safe_result, result_str);
//   strcpy(safe_termination, termination);
  
//   // Replace problematic characters only in result/termination parts
//   for (char *p = safe_result; *p; p++) {
//     if (*p == '/' || *p == '-') *p = '_';
//   }
//   for (char *p = safe_termination; *p; p++) {
//     if (*p == '/' || *p == '-') *p = '_';
//   }
  
//   sprintf(filename, "pufferlib/resources/chess/training_logs/complete_games/game_%d_%ld_%s_%s.pgn", 
//           env_id, now, safe_result, safe_termination);
  
//   FILE* file = fopen(filename, "w");
//   if (!file) {
//     return;
//   } else {
//   }
  
//   // Write PGN header
//   fprintf(file, "[Event \"PufferLib Training Game\"]\n");
//   fprintf(file, "[Site \"Environment %d\"]\n", env_id);
//   fprintf(file, "[Date \"%ld\"]\n", now);
//   fprintf(file, "[White \"AI-White\"]\n");
//   fprintf(file, "[Black \"AI-Black\"]\n");
//   fprintf(file, "[Result \"%s\"]\n", result_str);
//   fprintf(file, "[Termination \"%s\"]\n", termination);
//   fprintf(file, "\n");
  
//   // Write moves in algebraic notation
//   int move_number = 1;
//   for (int i = 0; i < ctx->complete_game_action_count; i++) {
//     // Use the stored canonical UCI move directly
//     const char* uci_move = ctx->complete_game_moves[i];
    
//     // Simple UCI to algebraic (basic format: from-to, e.g. e2e4)
//     if (i % 2 == 0) {
//       fprintf(file, "%d. %s ", move_number, uci_move);
//       if (i == ctx->complete_game_action_count - 1) fprintf(file, "\n");
//     } else {
//       fprintf(file, "%s ", uci_move);
//       if (i % 4 == 1) fprintf(file, "\n");
//       move_number++;
//     }
//   }
  
//   if (ctx->complete_game_action_count % 2 == 1) fprintf(file, "\n");
//   fprintf(file, "%s\n", result_str);
  
//   fclose(file);
//   printf("[Chess] Logged complete game to %s\n", filename);
// }

// // === PERSPECTIVE FLIPPING FOR SELF-PLAY ===

// static inline void flip_uci_for_black_perspective(const char *original_uci,
//                                                   char *flipped_uci) {
//   flipped_uci[0] = original_uci[0];             // file stays same
//   flipped_uci[1] = '9' - original_uci[1] + '0'; // flip rank: 1→8, 2→7, etc.
//   flipped_uci[2] = original_uci[2];             // file stays same
//   flipped_uci[3] = '9' - original_uci[3] + '0'; // flip rank

//   if (strlen(original_uci) >= 5) {
//     flipped_uci[4] = original_uci[4]; // promotion piece unchanged
//     flipped_uci[5] = '\0';
//   } else {
//     flipped_uci[4] = '\0';
//   }
// }

// // === LEGAL MOVE GENERATION (PURE C, OPTIMIZED) ===


// static bool is_square_attacked(const ChessBoard *board, Square sq,
//                                PieceColor by_color) {
//   PROFILE_START(profile_is_square_attacked_ticks)
//   // RAY-BASED ATTACK DETECTION: Start from target square and look outward for
//   // attackers

//   // Check pawn attacks (2 squares diagonally in front from attacker's
//   // perspective)
//   int pawn_direction = (by_color == C_WHITE) ? 1 : -1;
//   for (int dx = -1; dx <= 1; dx += 2) { // -1 and +1
//     int x = sq.x + dx;
//     int y =
//         sq.y - pawn_direction; // Reverse direction since we're looking backward
//     if (x >= 0 && x < 8 && y >= 0 && y < 8) {
//       const Piece *p = get_piece_const(board, x, y);
//       if (p && p->type == PAWN && p->color == by_color) {
//         PROFILE_STOP(profile_is_square_attacked_ticks)
//         return true;
//       }
//     }
//   }

//   // Check knight attacks (8 possible positions)
//   int knight_moves[][2] = {{2, 1}, {2, -1}, {-2, 1}, {-2, -1},
//                            {1, 2}, {1, -2}, {-1, 2}, {-1, -2}};
//   for (int i = 0; i < 8; i++) {
//     int x = sq.x + knight_moves[i][0];
//     int y = sq.y + knight_moves[i][1];
//     if (x >= 0 && x < 8 && y >= 0 && y < 8) {
//       const Piece *p = get_piece_const(board, x, y);
//       if (p && p->type == KNIGHT && p->color == by_color) {
//         PROFILE_STOP(profile_is_square_attacked_ticks)
//         return true;
//       }
//     }
//   }

//   // Check king attacks (8 adjacent squares)
//   for (int dx = -1; dx <= 1; dx++) {
//     for (int dy = -1; dy <= 1; dy++) {
//       if (dx == 0 && dy == 0)
//         continue;
//       int x = sq.x + dx;
//       int y = sq.y + dy;
//       if (x >= 0 && x < 8 && y >= 0 && y < 8) {
//         const Piece *p = get_piece_const(board, x, y);
//         if (p && p->type == KING && p->color == by_color) {
//           PROFILE_STOP(profile_is_square_attacked_ticks)
//           return true;
//         }
//       }
//     }
//   }

//   // Check sliding piece attacks (rook, bishop, queen) by radiating outward
//   int directions[][2] = {{0, 1}, {1, 0},  {0, -1}, {-1, 0},
//                          {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
//   for (int d = 0; d < 8; d++) {
//     int dx = directions[d][0];
//     int dy = directions[d][1];
//     bool is_diagonal = (d >= 4);

//     // Radiate outward in this direction until we hit a piece or board edge
//     for (int dist = 1; dist < 8; dist++) {
//       int x = sq.x + dx * dist;
//       int y = sq.y + dy * dist;
//       if (x < 0 || x >= 8 || y < 0 || y >= 8)
//         break;

//       const Piece *p = get_piece_const(board, x, y);
//       if (p && p->type != EMPTY) {
//         // Found a piece - check if it can attack this square
//         if (p->color == by_color) {
//           if (p->type == QUEEN) {
//             PROFILE_STOP(profile_is_square_attacked_ticks)
//             return true;
//           }
//           if (p->type == BISHOP && is_diagonal) {
//             PROFILE_STOP(profile_is_square_attacked_ticks)
//             return true;
//           }
//           if (p->type == ROOK && !is_diagonal) {
//             PROFILE_STOP(profile_is_square_attacked_ticks)
//             return true;
//           }
//         }
//         break; // Any piece blocks further pieces in this direction
//       }
//     }
//   }

//   PROFILE_STOP(profile_is_square_attacked_ticks)
//   return false;
// }

// static bool is_in_check(const ChessBoard *board, PieceColor color) {
//   // Find king
//   for (int x = 0; x < 8; x++) {
//     for (int y = 0; y < 8; y++) {
//       const Piece *piece = get_piece_const(board, x, y);
//       if (piece && piece->type == KING && piece->color == color) {
//         PieceColor opponent = (color == C_WHITE) ? C_BLACK : C_WHITE;
//         Square king_pos;
//         king_pos.x = (int8_t)x;
//         king_pos.y = (int8_t)y;
//         return is_square_attacked(board, king_pos, opponent);
//       }
//     }
//   }
//   return false; // No king found
// }

// // Optimized legal move generation (returns UCI strings directly)
// // === LEGAL MOVE GENERATION ===

// typedef struct {
//   Square from;
//   Square to;
//   PieceType promotion;
//   bool is_castling;
//   bool is_en_passant;
// } ChessMove;

// // Helper function to convert action ID to UCI notation
// static void action_to_uci(int action_id, char* uci_str) {
//   if (action_id < 0 || action_id >= TOTAL_CHESS_ACTIONS) {
//     strcpy(uci_str, "0000");
//     return;
//   }
//   strcpy(uci_str, ACTION_ID_TO_UCI[action_id]);
// }

// // Helper function to parse UCI move string into ChessMove struct
// static bool parse_uci_move(const char* uci_str, ChessMove* move) {
//   if (!uci_str || strlen(uci_str) < 4) {
//     return false;
//   }
  
//   // Parse from square
//   move->from.x = uci_str[0] - 'a';
//   move->from.y = uci_str[1] - '1';
//   move->to.x = uci_str[2] - 'a';
//   move->to.y = uci_str[3] - '1';
  
//   // Check bounds
//   if (move->from.x < 0 || move->from.x >= 8 || move->from.y < 0 || move->from.y >= 8 ||
//       move->to.x < 0 || move->to.x >= 8 || move->to.y < 0 || move->to.y >= 8) {
//     return false;
//   }
  
//   // Parse promotion if present
//   move->promotion = EMPTY;
//   if (strlen(uci_str) == 5) {
//     switch (uci_str[4]) {
//       case 'q': move->promotion = QUEEN; break;
//       case 'r': move->promotion = ROOK; break;
//       case 'b': move->promotion = BISHOP; break;
//       case 'n': move->promotion = KNIGHT; break;
//       default: return false;
//     }
//   }
  
//   // Special moves detection (will be refined based on board state)
//   move->is_castling = false;
//   move->is_en_passant = false;
  
//   return true;
// }

// typedef struct {
//   ChessMove moves[256];
//   int count;
// } LegalMoves;

// // Undo information for make/unmake move optimization
// typedef struct {
//   Piece captured_piece;
//   uint8_t old_castle_rights;
//   int8_t old_ep_square;
//   uint8_t old_halfmove_clock;
//   uint64_t old_zobrist_hash;
//   bool was_castling;
//   bool was_en_passant;
//   Square rook_from, rook_to; // For castling undo
// } UndoInfo;

// // Forward declarations
// static bool apply_uci_move(ChessContext *ctx, const char *uci_str);
// static void add_position_to_history(ChessContext *ctx, uint64_t hash);
// static int get_position_count(ChessContext *ctx, uint64_t hash);
// static bool is_threefold_repetition(ChessContext *ctx);
// static bool is_insufficient_material(ChessContext *ctx);
// static void make_move_fast(ChessBoard *board, ChessMove move, UndoInfo *undo);
// static void unmake_move_fast(ChessBoard *board, ChessMove move, UndoInfo *undo);

// static bool chess_is_legal_move(ChessContext *ctx, ChessMove move) {
//   PROFILE_START(profile_is_legal_move_ticks)
//   // MAKE/UNMAKE OPTIMIZATION: Use actual board instead of copying
//   ChessBoard *board = &ctx->board;
//   PieceColor moving_color = board->to_move;

//   // Basic validity checks
//   Piece *from_piece = get_piece(board, move.from.x, move.from.y);
//   if (!from_piece || from_piece->type == EMPTY ||
//       from_piece->color != moving_color) {
//     PROFILE_STOP(profile_is_legal_move_ticks)
//     return false;
//   }

//   // Handle castling move specially - check path before making move
//   if (move.is_castling) {
//     if (from_piece->type != KING) {
//       PROFILE_STOP(profile_is_legal_move_ticks)
//       return false;
//     }
//     if (is_in_check(board, moving_color)) {
//       PROFILE_STOP(profile_is_legal_move_ticks)
//       return false; // Can't castle out of check
//     }

//     // Verify castling path is clear and not through check
//     int rank = (moving_color == C_WHITE) ? 0 : 7;
//     if (move.to.x == 6) { // Kingside
//       for (int x = 5; x <= 6; x++) {
//         Square sq = {(int8_t)x, (int8_t)rank};
//         if (is_square_attacked(board, sq, (PieceColor)(1 - moving_color))) {
//           PROFILE_STOP(profile_is_legal_move_ticks)
//           return false;
//         }
//       }
//     } else if (move.to.x == 2) { // Queenside
//       for (int x = 2; x <= 3; x++) {
//         Square sq = {(int8_t)x, (int8_t)rank};
//         if (is_square_attacked(board, sq, (PieceColor)(1 - moving_color))) {
//           PROFILE_STOP(profile_is_legal_move_ticks)
//           return false;
//         }
//       }
//     }
//   }

//   // Make move and save undo information
//   UndoInfo undo;
//   make_move_fast(board, move, &undo);

//   // Check if our king is in check after the move
//   bool is_legal = !is_in_check(board, moving_color);

//   // Unmake the move to restore board state
//   unmake_move_fast(board, move, &undo);

//   PROFILE_STOP(profile_is_legal_move_ticks)
//   return is_legal;
// }

// static void add_legal_move(ChessContext *ctx, LegalMoves *moves,
//                            ChessMove move) {
//   if (moves->count >= 256)
//     return;

//   if (chess_is_legal_move(ctx, move)) {
//     moves->moves[moves->count] = move;
//     moves->count++;
//   }
// }

// static void generate_pseudo_legal_moves_for_piece(ChessContext *ctx,
//                                                   LegalMoves *moves,
//                                                   Square from) {
//   ChessBoard *board = &ctx->board;
//   const Piece *piece = get_piece_const(board, from.x, from.y);
//   if (!piece || piece->type == EMPTY || piece->color != board->to_move)
//     return;

//   PieceColor us = board->to_move;
//   PieceColor them = (us == C_WHITE) ? C_BLACK : C_WHITE;

//   switch (piece->type) {
//   case PAWN: {
//     int direction = (us == C_WHITE) ? 1 : -1;
//     int start_rank = (us == C_WHITE) ? 1 : 6;
//     int promote_rank = (us == C_WHITE) ? 7 : 0;

//     // --- 1. Single Push ---
//     int single_push_y = from.y + direction;
//     if (single_push_y >= 0 && single_push_y < 8) {
//       const Piece *target = get_piece_const(board, from.x, single_push_y);
//       if (target && target->type == EMPTY) {
//         Square to_sq = {from.x, (int8_t)single_push_y};
//         if (to_sq.y == promote_rank) { // Promotion on single push
//           PieceType promotions[] = {QUEEN, ROOK, BISHOP, KNIGHT};
//           for (int p = 0; p < 4; p++) {
//             add_legal_move(
//                 ctx, moves,
//                 (ChessMove){from, to_sq, promotions[p], false, false});
//           }
//         } else { // Regular single push
//           add_legal_move(ctx, moves,
//                          (ChessMove){from, to_sq, EMPTY, false, false});
//         }
//       }
//     }

//     // --- 2. Double Push ---
//     if (from.y == start_rank) {
//       int single_push_y_check = from.y + direction;
//       int double_push_y = from.y + 2 * direction;
//       // The y coordinates for double push are always in-bounds, no need for y
//       // check
//       const Piece *path_blocker =
//           get_piece_const(board, from.x, single_push_y_check);
//       const Piece *target = get_piece_const(board, from.x, double_push_y);
//       if (path_blocker && path_blocker->type == EMPTY && target &&
//           target->type == EMPTY) {
//         Square to_sq = {from.x, (int8_t)double_push_y};
//         add_legal_move(ctx, moves,
//                        (ChessMove){from, to_sq, EMPTY, false, false});
//       }
//     }

//     // --- 3. Captures & En Passant ---
//     for (int dx = -1; dx <= 1; dx += 2) {
//       int capture_x = from.x + dx;
//       int capture_y = from.y + direction;

//       // Check bounds for the destination square FIRST
//       if (capture_x >= 0 && capture_x < 8 && capture_y >= 0 && capture_y < 8) {
//         Square to_sq = {(int8_t)capture_x, (int8_t)capture_y};
//         const Piece *target = get_piece_const(board, capture_x, capture_y);

//         // Regular capture
//         if (target && target->type != EMPTY && target->color == them) {
//           if (to_sq.y == promote_rank) { // Promotion on capture
//             PieceType promotions[] = {QUEEN, ROOK, BISHOP, KNIGHT};
//             for (int p = 0; p < 4; p++) {
//               add_legal_move(
//                   ctx, moves,
//                   (ChessMove){from, to_sq, promotions[p], false, false});
//             }
//           } else { // Regular capture
//             add_legal_move(ctx, moves,
//                            (ChessMove){from, to_sq, EMPTY, false, false});
//           }
//         }
//         // En Passant capture
//         else if (board->ep_square == (capture_y * 8 + capture_x)) {
//           // An en-passant capture cannot result in a promotion.
//           add_legal_move(ctx, moves,
//                          (ChessMove){from, to_sq, EMPTY, false, true});
//         }
//       }
//     }
//     break;
//   }

//   case ROOK:
//   case BISHOP:
//   case QUEEN: {
//     // Sliding pieces
//     int directions[8][2];
//     int num_dirs;

//     if (piece->type == ROOK) {
//       int rook_dirs[][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
//       memcpy(directions, rook_dirs, sizeof(rook_dirs));
//       num_dirs = 4;
//     } else if (piece->type == BISHOP) {
//       int bishop_dirs[][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
//       memcpy(directions, bishop_dirs, sizeof(bishop_dirs));
//       num_dirs = 4;
//     } else { // QUEEN
//       int queen_dirs[][2] = {{0, 1}, {1, 0},  {0, -1}, {-1, 0},
//                              {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
//       memcpy(directions, queen_dirs, sizeof(queen_dirs));
//       num_dirs = 8;
//     }

//     for (int d = 0; d < num_dirs; d++) {
//       int dx = directions[d][0];
//       int dy = directions[d][1];

//       for (int dist = 1; dist < 8; dist++) {
//         Square to;
//         to.x = from.x + dx * dist;
//         to.y = from.y + dy * dist;
//         if (to.x < 0 || to.x >= 8 || to.y < 0 || to.y >= 8)
//           break;

//         const Piece *target = get_piece_const(board, to.x, to.y);
//         if (target && target->type != EMPTY) {
//           if (target->color == them) {
//             // Capture and stop
//             ChessMove move = {from, to, EMPTY, false, false};
//             add_legal_move(ctx, moves, move);
//           }
//           break; // Blocked
//         } else {
//           // Empty square
//           ChessMove move = {from, to, EMPTY, false, false};
//           add_legal_move(ctx, moves, move);
//         }
//       }
//     }
//     break;
//   }

//   case KNIGHT: {
//     int knight_moves[][2] = {{2, 1}, {2, -1}, {-2, 1}, {-2, -1},
//                              {1, 2}, {1, -2}, {-1, 2}, {-1, -2}};

//     for (int i = 0; i < 8; i++) {
//       Square to;
//       to.x = from.x + knight_moves[i][0];
//       to.y = from.y + knight_moves[i][1];
//       if (to.x >= 0 && to.x < 8 && to.y >= 0 && to.y < 8) {
//         const Piece *target = get_piece_const(board, to.x, to.y);
//         if (!target || target->type == EMPTY || target->color == them) {
//           ChessMove move = {from, to, EMPTY, false, false};
//           add_legal_move(ctx, moves, move);
//         }
//       }
//     }
//     break;
//   }

//   case KING: {
//     // Regular king moves
//     for (int dx = -1; dx <= 1; dx++) {
//       for (int dy = -1; dy <= 1; dy++) {
//         if (dx == 0 && dy == 0)
//           continue;

//         Square to;
//         to.x = from.x + dx;
//         to.y = from.y + dy;
//         if (to.x >= 0 && to.x < 8 && to.y >= 0 && to.y < 8) {
//           const Piece *target = get_piece_const(board, to.x, to.y);
//           if (!target || target->type == EMPTY || target->color == them) {
//             ChessMove move = {from, to, EMPTY, false, false};
//             add_legal_move(ctx, moves, move);
//           }
//         }
//       }
//     }

//     // Castling
//     if (!is_in_check(board, us)) {
//       int rank = (us == C_WHITE) ? 0 : 7;

//       if (from.x == 4 && from.y == rank) {
//         // Kingside castling
//         if ((board->castle_rights & (us == C_WHITE ? 1 : 4))) {
//           bool can_castle = true;
//           // Check squares are empty and not attacked
//           for (int x = 5; x <= 6; x++) {
//             const Piece *sq = get_piece_const(board, x, rank);
//             if (sq && sq->type != EMPTY) {
//               can_castle = false;
//               break;
//             }
//             Square check_sq = {(int8_t)x, (int8_t)rank};
//             if (is_square_attacked(board, check_sq, them)) {
//               can_castle = false;
//               break;
//             }
//           }
//           if (can_castle) {
//             Square to;
//             to.x = 6;
//             to.y = (int8_t)rank;
//             ChessMove move = {from, to, EMPTY, true, false};
//             add_legal_move(ctx, moves, move);
//           }
//         }

//         // Queenside castling
//         if ((board->castle_rights & (us == C_WHITE ? 2 : 8))) {
//           bool can_castle = true;
//           // Check squares are empty
//           for (int x = 1; x <= 3; x++) {
//             const Piece *sq = get_piece_const(board, x, rank);
//             if (sq && sq->type != EMPTY) {
//               can_castle = false;
//               break;
//             }
//           }
//           // Check squares are not attacked
//           if (can_castle) {
//             for (int x = 2; x <= 3; x++) {
//               Square check_sq = {(int8_t)x, (int8_t)rank};
//               if (is_square_attacked(board, check_sq, them)) {
//                 can_castle = false;
//                 break;
//               }
//             }
//           }
//           if (can_castle) {
//             Square to;
//             to.x = 2;
//             to.y = (int8_t)rank;
//             ChessMove move = {from, to, EMPTY, true, false};
//             add_legal_move(ctx, moves, move);
//           }
//         }
//       }
//     }
//     break;
//   }

//   case EMPTY:
//   default:
//     break;
//   }
// }

// // Yield-based move generation callback type
// typedef bool (*MoveYieldCallback)(ChessContext *ctx, const ChessMove *move,
//                                   void *user_data);

// // Callback that terminates on first legal move found
// static bool first_move_callback(ChessContext *ctx, const ChessMove *move,
//                                 void *user_data) {
//   bool *found = (bool *)user_data;
//   *found = true;
//   return true; // Terminate immediately
// }

// // Yield-based move generation - returns true if callback requested early
// // termination
// static bool chess_generate_legal_moves_yield(ChessContext *ctx,
//                                              MoveYieldCallback yield_fn,
//                                              void *user_data) {
//   // Iterate through all squares on the board
//   for (int x = 0; x < 8; x++) {
//     for (int y = 0; y < 8; y++) {
//       Square from = {(int8_t)x, (int8_t)y};
//       const Piece *piece = get_piece_const(&ctx->board, x, y);
//       if (piece && piece->type != EMPTY && piece->color == ctx->board.to_move) {
//         // Generate moves for this piece and yield each one
//         LegalMoves temp_moves;
//         temp_moves.count = 0;
//         generate_pseudo_legal_moves_for_piece(ctx, &temp_moves, from);

//         // Yield each move individually
//         for (int i = 0; i < temp_moves.count; i++) {
//           // Call yield callback - if it returns true, terminate early
//           if (yield_fn(ctx, &temp_moves.moves[i], user_data)) {
//             return true; // Early termination requested
//           }
//         }
//       }
//     }
//   }
//   return false; // Completed without early termination
// }

// static void chess_generate_legal_moves(ChessContext *ctx, LegalMoves *moves) {
//   moves->count = 0;

//   // Iterate through all squares on the board
//   for (int x = 0; x < 8; x++) {
//     for (int y = 0; y < 8; y++) {
//       Square from = {(int8_t)x, (int8_t)y};
//       const Piece *piece = get_piece_const(&ctx->board, x, y);
//       if (piece && piece->type != EMPTY && piece->color == ctx->board.to_move) {
//         generate_pseudo_legal_moves_for_piece(ctx, moves, from);
//       }
//     }
//   }
// }

// static int chess_generate_legal_moves_uci(ChessContext *ctx) {
//   PROFILE_START(profile_move_gen_uci_ticks)
  
//   uint64_t current_hash = ctx->board.zobrist_hash;
//   PieceColor current_player = ctx->board.to_move;
  
//   // Check if we already have moves for current player
//   if (current_player == C_WHITE && ctx->white_moves_cached && 
//       ctx->cached_board_hash == current_hash) {
//     PROFILE_STOP(profile_move_gen_uci_ticks)
//     return ctx->white_legal_moves_count;
//   } else if (current_player == C_BLACK && ctx->black_moves_cached && 
//              ctx->cached_board_hash == current_hash) {
//     PROFILE_STOP(profile_move_gen_uci_ticks)
//     return ctx->black_legal_moves_count;
//   }
  
//   // If position changed, clear both caches
//   if (ctx->cached_board_hash != current_hash) {
//     ctx->white_moves_cached = false;
//     ctx->black_moves_cached = false;
//     ctx->position_fully_cached = false;
//     ctx->cached_board_hash = current_hash;
//   }

//   // Generate legal moves for current player
//   LegalMoves moves;
//   chess_generate_legal_moves(ctx, &moves);

//   // Store in appropriate buffer
//   char (*buffer)[6];
//   int *count;
//   int *action_ids;
  
//   if (current_player == C_WHITE) {
//     buffer = ctx->white_legal_moves_buffer;
//     count = &ctx->white_legal_moves_count;
//     action_ids = ctx->white_legal_action_ids;
//   } else {
//     buffer = ctx->black_legal_moves_buffer;
//     count = &ctx->black_legal_moves_count;
//     action_ids = ctx->black_legal_action_ids;
//   }
  
//   *count = 0;
//   for (int i = 0; i < moves.count && i < 256; i++) {
//     ChessMove move = moves.moves[i];
//     char uci_str[6];
    
//     if (move.promotion != EMPTY) {
//       char promo_char = (move.promotion == QUEEN)    ? 'q'
//                         : (move.promotion == ROOK)   ? 'r'
//                         : (move.promotion == BISHOP) ? 'b'
//                                                      : 'n';
//       snprintf(uci_str, 6, "%c%c%c%c%c",
//                'a' + move.from.x, '1' + move.from.y, 'a' + move.to.x,
//                '1' + move.to.y, promo_char);
//     } else {
//       snprintf(uci_str, 5, "%c%c%c%c",
//                'a' + move.from.x, '1' + move.from.y, 'a' + move.to.x,
//                '1' + move.to.y);
//     }
    
//     strcpy(buffer[*count], uci_str);
    
//     // Pre-compute action IDs for both perspectives
//     if (current_player == C_WHITE) {
//       action_ids[*count] = uci_to_action_id(uci_str);
//     } else {
//       // For black, we need the flipped perspective
//       char flipped_uci[6];
//       flip_uci_for_black_perspective(uci_str, flipped_uci);
//       action_ids[*count] = uci_to_action_id(flipped_uci);
//     }
    
//     (*count)++;
//   }

//   // Mark this player's moves as cached
//   if (current_player == C_WHITE) {
//     ctx->white_moves_cached = true;
//   } else {
//     ctx->black_moves_cached = true;
//   }

//   // Check if both players are now cached
//   ctx->position_fully_cached = ctx->white_moves_cached && ctx->black_moves_cached;

//   PROFILE_STOP(profile_move_gen_uci_ticks)
//   return *count;
// }

// // NEW FUNCTION: Generate moves for both players efficiently
// static void chess_generate_all_legal_moves(ChessContext *ctx) {
//   PROFILE_START(profile_move_gen_uci_ticks)
  
//   uint64_t current_hash = ctx->board.zobrist_hash;
  
//   // If already fully cached for this position, nothing to do
//   if (ctx->position_fully_cached && ctx->cached_board_hash == current_hash) {
//     PROFILE_STOP(profile_move_gen_uci_ticks)
//     return;
//   }
  
//   // Clear caches if position changed
//   if (ctx->cached_board_hash != current_hash) {
//     ctx->white_moves_cached = false;
//     ctx->black_moves_cached = false;
//     ctx->position_fully_cached = false;
//     ctx->cached_board_hash = current_hash;
//   }
  
//   // Generate WHITE's moves if not cached
//   if (!ctx->white_moves_cached) {
//     PieceColor saved_to_move = ctx->board.to_move;
//     ctx->board.to_move = C_WHITE;
//     chess_generate_legal_moves_uci(ctx);
//     ctx->board.to_move = saved_to_move;
//   }
  
//   // Generate BLACK's moves if not cached
//   if (!ctx->black_moves_cached) {
//     PieceColor saved_to_move = ctx->board.to_move;
//     ctx->board.to_move = C_BLACK;
//     chess_generate_legal_moves_uci(ctx);
//     ctx->board.to_move = saved_to_move;
//   }
  
//   ctx->position_fully_cached = true;
//   PROFILE_STOP(profile_move_gen_uci_ticks)
// }

// // === BOARD STATE MANIPULATION ===

// static void init_board(ChessBoard *board) {
//   memset(board, 0, sizeof(ChessBoard));

//   // Set up starting position
//   const char *start_fen =
//       "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

//   // Parse FEN (simplified)
//   int x = 0, y = 7;
//   const char *p = start_fen;

//   while (*p && *p != ' ') {
//     if (*p == '/') {
//       x = 0;
//       y--;
//     } else if (*p >= '1' && *p <= '8') {
//       x += (*p - '0');
//     } else {
//       PieceColor color = (*p >= 'A' && *p <= 'Z') ? C_WHITE : C_BLACK;
//       PieceType type = EMPTY;

//       switch (*p | 32) {
//       case 'k':
//         type = KING;
//         break;
//       case 'q':
//         type = QUEEN;
//         break;
//       case 'r':
//         type = ROOK;
//         break;
//       case 'b':
//         type = BISHOP;
//         break;
//       case 'n':
//         type = KNIGHT;
//         break;
//       case 'p':
//         type = PAWN;
//         break;
//       }

//       if (type != EMPTY && x < 8) {
//         board->board[y * 8 + x] = (Piece){color, type};
//         x++;
//       }
//     }
//     p++;
//   }

//   board->to_move = C_WHITE;
//   board->castle_rights = 0xF; // KQkq
//   board->ep_square = -1;
//   board->halfmove_clock = 0;
//   board->fullmove_number = 1;

//   // Initialize Zobrist hash
//   board->zobrist_hash = compute_zobrist_hash(board);
// }

// // === OBSERVATION COMPUTATION WITH PERSPECTIVE FLIPPING (OPTIMIZED) ===

// // // Helper function to compute observation for a single agent
// // static void compute_single_agent_observation(CChess *env, ChessContext *ctx, PieceColor player, int obs_offset) {
// //   // // printf("[OBSERVE] Computing observation for player %s at offset %d\n", 
// //   //        (player == C_WHITE) ? "WHITE" : "BLACK", obs_offset);
// //   // fflush(stdout);
// //   int idx = 0;

// //   // Clear the first 13 planes (all piece planes + empty squares)
// //   memset(&env->observations[obs_offset], 0, 13 * 64 * sizeof(float));

// //   // --- SINGLE PASS OVER THE BOARD ---
// //   // Iterate through each square once to populate all piece-related planes.
// //   for (int y_white_perspective = 0; y_white_perspective < 8;
// //        y_white_perspective++) {
// //     for (int x = 0; x < 8; x++) {
// //       // Determine the actual y-coordinate based on the player's perspective
// //       int y_actual = (player == C_WHITE) ? y_white_perspective
// //                                          : (7 - y_white_perspective);
// //       int square_index_actual = y_actual * 8 + x;
// //       const Piece *p = &ctx->board.board[square_index_actual];

// //       // Determine the observation y-coordinate (always from player's perspective)
// //       int y_obs = y_white_perspective;
// //       int obs_square_idx = y_obs * 8 + x;

// //       if (p->type == EMPTY) {
// //         // Plane 12: Empty squares
// //         env->observations[obs_offset + 12 * 64 + obs_square_idx] = 1.0f;
// //       } else {
// //         int plane_offset = (p->color == player) ? 0 : 6;
// //         // Piece type is 1-6, planes are 0-5 and 6-11
// //         int piece_plane = p->type - 1;
// //         env->observations[obs_offset + (plane_offset + piece_plane) * 64 + obs_square_idx] = 1.0f;
// //       }
// //     }
// //   }

// //   // --- NON-PIECE PLANES ---
// //   // Now set idx to start after piece planes (13 planes * 64 squares each)
// //   idx = 13 * 64;

// //   // Repetition count plane (using actual position history)
// //   int reps = get_position_count(ctx, ctx->board.zobrist_hash);
// //   float rep_val = (reps >= 2) ? 1.0f : 0.0f; // Simplified: 0 for 1 rep, 1 for 2+ reps
// //   for (int i = 0; i < 64; i++) {
// //     env->observations[obs_offset + idx++] = rep_val;
// //   }

// //   // Side to move plane (always 0 from current player's perspective)
// //   for (int i = 0; i < 64; i++) {
// //     env->observations[obs_offset + idx++] = 0.0f;
// //   }

// //   // Halfmove clock plane
// //   float halfmove_val = ctx->board.halfmove_clock / 100.0f; // Normalize to 0-1 range
// //   for (int i = 0; i < 64; i++) {
// //     env->observations[obs_offset + idx++] = halfmove_val;
// //   }

// //   // Castling rights planes (4 planes, flipped for black perspective)
// //   uint8_t rights = ctx->board.castle_rights;
// //   if (player == C_BLACK) {
// //     // Flip castling rights for Black's perspective
// //     uint8_t flipped = 0;
// //     if (rights & 4) flipped |= 1; // BK -> WK
// //     if (rights & 8) flipped |= 2; // BQ -> WQ
// //     if (rights & 1) flipped |= 4; // WK -> BK
// //     if (rights & 2) flipped |= 8; // WQ -> BQ
// //     rights = flipped;
// //   }

// //   for (int i = 0; i < 4; i++) {
// //     float castle_val = (rights & (1 << i)) ? 1.0f : 0.0f;
// //     for (int j = 0; j < 64; j++) {
// //       env->observations[obs_offset + idx++] = castle_val;
// //     }
// //   }

// //   // En passant target square plane (flipped for black perspective)
// //   int8_t ep_square = ctx->board.ep_square;
// //   if (ep_square != -1 && player == C_BLACK) {
// //     int ep_x = ep_square % 8;
// //     int ep_y = ep_square / 8;
// //     ep_square = (7 - ep_y) * 8 + ep_x;
// //   }
// //   for (int i = 0; i < 64; i++) {
// //     env->observations[obs_offset + idx++] = (ep_square == i) ? 1.0f : 0.0f;
// //   }

// //   assert(idx == 1344); // 21 * 8 * 8

// //   // --- LEGAL MOVE MASK ---
// //   if (!ctx->legal_moves_cached) {
// //     chess_generate_legal_moves_uci(ctx);
// //   }
  
// //   // DEBUG: Always show legal move count
  

// //   // CRITICAL FIX: Only set legal moves for the player whose turn it is
// //   // In dual-agent mode, only the current player should have legal moves
// //   PieceColor current_player = ctx->board.to_move;
// //   bool is_player_turn = (player == current_player);

// // //   // Add diagnostic logging when clearing mask for inactive player
// // //   if (!is_player_turn) {
// // //     printf("[DIAGNOSTIC] Clearing mask for INACTIVE player %s (turn is %s) at obs_offset %d\n",
// // //            (player == C_WHITE) ? "WHITE" : "BLACK",
// // //            (current_player == C_WHITE) ? "WHITE" : "BLACK",
// // //            obs_offset);
// // //   }

// //   // Clear mask
// //   memset(&env->observations[obs_offset + idx], 0, TOTAL_CHESS_ACTIONS * sizeof(float));
  

// //   if (env->debug_disable_mask) {
// //     for (int i = 0; i < TOTAL_CHESS_ACTIONS; i++) {
// //       env->observations[obs_offset + idx + i] = 1.0f;
// //     }
// //   } else if (is_player_turn) {
// //     // Only create mask if it's this player's turn
// //     for (int i = 0; i < ctx->legal_moves_count; i++) {
// //       const char *canonical_uci = ctx->legal_moves_buffer[i];
// //       char perspective_uci[6];

// //       // The policy always sees the board as if it were white.
// //       // So for black, we must flip the canonical UCI move to match the policy's perspective.
// //       if (player == C_BLACK) {
// //         flip_uci_for_black_perspective(canonical_uci, perspective_uci);
// //       } else {
// //         strcpy(perspective_uci, canonical_uci);
// //       }

// //       int action_id = uci_to_action_id(perspective_uci);
// //       if (action_id >= 0) {
// //         int mask_idx = obs_offset + idx + action_id;
// //         env->observations[mask_idx] = 1.0f;
// //         float verify_value = env->observations[mask_idx];
// //       } else {
// //       }
// //     }
// //   }
// //   // If it's not this player's turn, the mask remains all zeros (cleared above)
// // }

// // Helper function to compute observation for a single agent
// static void compute_single_agent_observation(CChess *env, ChessContext *ctx,
//                                              PieceColor player,
//                                              int obs_offset) {
//   uint64_t current_hash = ctx->board.zobrist_hash;
  
//   // PERFORMANCE OPTIMIZATION: Check if board observation is cached
//   // DISABLED FOR DEBUGGING: Force fresh observation generation
//   if (false && ctx->observation_cached && 
//       ctx->cached_observation_hash == current_hash && 
//       ctx->cached_observation_player == player) {
//     // Use cached board observation (first 1472 floats)
//     memcpy(&env->observations[obs_offset], ctx->cached_observation, 1472 * sizeof(float));
//   } else {
//     // Compute fresh board observation
//     memset(&env->observations[obs_offset], 0, 1472 * sizeof(float));
//     int idx = 0;

//     // --- SINGLE PASS OVER THE BOARD (Correct) ---
//     for (int y_white_perspective = 0; y_white_perspective < 8;
//          y_white_perspective++) {
//       for (int x = 0; x < 8; x++) {
//         int y_actual =
//             (player == C_WHITE) ? y_white_perspective : (7 - y_white_perspective);
//         int square_index_actual = y_actual * 8 + x;
//         const Piece *p = &ctx->board.board[square_index_actual];
//         int obs_square_idx = y_white_perspective * 8 + x;

//         if (p->type == EMPTY) {
//           env->observations[obs_offset + 12 * 64 + obs_square_idx] = 1.0f;
//         } else {
//           int plane_offset = (p->color == player) ? 0 : 6;
//           int piece_plane = p->type - 1;
//           env->observations[obs_offset + (plane_offset + piece_plane) * 64 +
//                             obs_square_idx] = 1.0f;
//         }
//       }
//     }
    
//     // --- NON-PIECE PLANES ---
//     // Now set idx to start after piece planes (13 planes * 64 squares each)
//     idx = 13 * 64;

//     // Plane 13: Repetition count plane (using actual position history)
//     int reps = get_position_count(ctx, ctx->board.zobrist_hash);
//     float rep_val = (reps >= 2) ? 1.0f : 0.0f; // Simplified: 0 for 1 rep, 1 for 2+ reps
//     for (int i = 0; i < 64; i++) {
//       env->observations[obs_offset + idx++] = rep_val;
//     }

//     // Plane 14: Side to move plane (always 0 from current player's perspective)
//     for (int i = 0; i < 64; i++) {
//       env->observations[obs_offset + idx++] = 0.0f;
//     }

//     // Plane 15: Halfmove clock plane
//     float halfmove_val = ctx->board.halfmove_clock / 100.0f; // Normalize to 0-1 range
//     for (int i = 0; i < 64; i++) {
//       env->observations[obs_offset + idx++] = halfmove_val;
//     }

//     // Planes 16-19: Castling rights planes (4 planes, flipped for black perspective)
//     uint8_t rights = ctx->board.castle_rights;
//     if (player == C_BLACK) {
//       // Flip castling rights for Black's perspective
//       uint8_t flipped = 0;
//       if (rights & 4) flipped |= 1; // BK -> WK
//       if (rights & 8) flipped |= 2; // BQ -> WQ
//       if (rights & 1) flipped |= 4; // WK -> BK
//       if (rights & 2) flipped |= 8; // WQ -> BQ
//       rights = flipped;
//     }

//     for (int i = 0; i < 4; i++) {
//       float castle_val = (rights & (1 << i)) ? 1.0f : 0.0f;
//       for (int j = 0; j < 64; j++) {
//         env->observations[obs_offset + idx++] = castle_val;
//       }
//     }

//     // Plane 20: En passant target square plane (flipped for black perspective)
//     int8_t ep_square = ctx->board.ep_square;
//     if (ep_square != -1 && player == C_BLACK) {
//       int ep_x = ep_square % 8;
//       int ep_y = ep_square / 8;
//       ep_square = (7 - ep_y) * 8 + ep_x;
//     }
//     for (int i = 0; i < 64; i++) {
//       env->observations[obs_offset + idx++] = (ep_square == i) ? 1.0f : 0.0f;
//     }

//     // Plane 21: Pieces the current player can capture on next turn
//     for (int y = 0; y < 8; y++) {
//       for (int x = 0; x < 8; x++) {
//         int y_actual = (player == C_WHITE) ? y : (7 - y);
//         int square_index = y_actual * 8 + x;
//         const Piece *p = &ctx->board.board[square_index];
        
//         bool can_capture = false;
//         if (p->type != EMPTY && p->color != player) {
//           // Use optimized is_square_attacked function instead of nested loops
//           Square target_sq = {(int8_t)x, (int8_t)y_actual};
//           can_capture = is_square_attacked(&ctx->board, target_sq, player);
//         }
//         env->observations[obs_offset + idx++] = can_capture ? 1.0f : 0.0f;
//       }
//     }

//     // Plane 22: Pieces that can capture the current player's pieces on next turn
//     for (int y = 0; y < 8; y++) {
//       for (int x = 0; x < 8; x++) {
//         int y_actual = (player == C_WHITE) ? y : (7 - y);
//         int square_index = y_actual * 8 + x;
//         const Piece *p = &ctx->board.board[square_index];
        
//         bool under_threat = false;
//         if (p->type != EMPTY && p->color == player) {
//           // Check if any opponent pieces can capture this square
//           Square target_sq = {(int8_t)x, (int8_t)y_actual};
//           under_threat = is_square_attacked(&ctx->board, target_sq, (PieceColor)(1 - player));
//         }
//         env->observations[obs_offset + idx++] = under_threat ? 1.0f : 0.0f;
//       }
//     }

//     assert(idx == 1472); // 23 * 8 * 8
    
//     // Cache the computed board observation
//     memcpy(ctx->cached_observation, &env->observations[obs_offset], 1472 * sizeof(float));
//     ctx->observation_cached = true;
//     ctx->cached_observation_hash = current_hash;
//     ctx->cached_observation_player = player;
//   }
  
//   // --- SPARSE LEGAL MOVE MASK ---
//   // Format: [num_legal_moves(1)] + [legal_action_ids(MAX_LEGAL_MOVES)]
//   int sparse_mask_idx = 1472; // Start index for sparse mask
//   const int MAX_LEGAL_MOVES = 64;

//   // Ensure we have legal moves for both players
//   chess_generate_all_legal_moves(ctx);

//   PieceColor current_player_turn = ctx->board.to_move;
//   bool is_player_turn = (player == current_player_turn);

//   // Initialize sparse mask: num_legal_moves + action_ids
//   int num_legal_moves = 0;
//   float *action_ids_ptr = &env->observations[obs_offset + sparse_mask_idx + 1];
  
//   if (env->debug_disable_mask) {
//     // Debug mode: All moves are legal (first 64 actions for sparse representation)
//     env->observations[obs_offset + sparse_mask_idx] = (float)MAX_LEGAL_MOVES;
//     for (int i = 0; i < MAX_LEGAL_MOVES; i++) {
//       action_ids_ptr[i] = (float)i;
//     }
//   } else if (ctx->dual_agent_self_play_mode || is_player_turn) {
//     // Get the appropriate move buffer based on current turn
//     char (*moves_buffer)[6];
//     int moves_count;
//     int *action_ids;
    
//     if (current_player_turn == C_WHITE) {
//       moves_buffer = ctx->white_legal_moves_buffer;
//       moves_count = ctx->white_legal_moves_count;
//       action_ids = ctx->white_legal_action_ids;
//     } else {
//       moves_buffer = ctx->black_legal_moves_buffer;
//       moves_count = ctx->black_legal_moves_count;
//       action_ids = ctx->black_legal_action_ids;
//     }
    
//     // Generate sparse representation of legal moves
//     for (int i = 0; i < moves_count && num_legal_moves < MAX_LEGAL_MOVES; i++) {
//       int action_id;
      
//       // Use pre-computed action IDs when possible
//       if (player == current_player_turn) {
//         action_id = action_ids[i];
//       } else {
//         // Need to convert to other player's perspective
//         char perspective_uci[6];
//         flip_uci_for_black_perspective(moves_buffer[i], perspective_uci);
//         action_id = uci_to_action_id(perspective_uci);
//       }
      
//       if (action_id >= 0) {
//         action_ids_ptr[num_legal_moves] = (float)action_id;
//         num_legal_moves++;
//       }
//     }
//   }
  
//   // Store the count of legal moves
//   env->observations[obs_offset + sparse_mask_idx] = (float)num_legal_moves;
// }

// // Corrected version of the observation generation orchestrator.
// // This function should replace the old `compute_observation_with_perspective`.

// // COLOR MONITORING: Validates observation data integrity at generation point
// void validate_chess_observation_integrity(CChess *env, ChessContext *ctx, PieceColor player, int obs_offset) {
//   // SENTINEL 1: Check observation buffer bounds
//   // In the new single-agent-view architecture, the offset for the active player
//   // is ALWAYS 0.
//   const int expected_offset = 0; // <-- NEW, CORRECT LOGIC
//   if (obs_offset != expected_offset) {
//     printf("[MONITOR_FATAL] Chess.h observation offset mismatch!\n");
//     printf("  Expected %s at offset %d, got offset %d\n",
//            (player == C_WHITE) ? "WHITE" : "BLACK", expected_offset,
//            obs_offset);
//     printf("  This indicates an error in the calling function.\n");
//     printf("  FIX: Ensure all calls to compute_single_agent_observation pass 0 "
//            "as the offset.\n");
//     exit(1);
//   }

//   // SENTINEL 2: Validate observation content signature  
//   float *obs = &env->observations[obs_offset];
//   float board_sum = 0.0f;
//   for (int i = 0; i < 1472; i++) board_sum += obs[i];
  
//   // Sparse mask validation  
//   // Format: [num_legal_moves(1)] + [action_ids(64)]
//   float num_legal_moves = obs[1472];
//   float sparse_mask_sum = 0.0f;
  
//   // Count number of valid action IDs (should equal num_legal_moves)
//   if (num_legal_moves > 0 && num_legal_moves <= 64) {
//     for (int i = 0; i < (int)num_legal_moves; i++) {
//       int action_id = (int)obs[1473 + i];
//       if (action_id >= 0 && action_id < 1968) {
//         sparse_mask_sum += 1.0f;
//       }
//     }
//   }
  
//   // Check basic observation integrity
//   if (board_sum < 1.0f) {
//     printf("[MONITOR_FATAL] Chess.h observation content invalid!\n");
//     printf("  %s observation at offset %d: board_sum=%.3f num_legal_moves=%.0f sparse_valid_count=%.0f\n",
//            (player == C_WHITE) ? "WHITE" : "BLACK", obs_offset, board_sum, num_legal_moves, sparse_mask_sum);
//     printf("  Board sum should be >1 (pieces present).\n");
//     printf("  FIX: Check compute_single_agent_observation() is writing correct data.\n");
//     exit(1);
//   }

//   // Check for uninitialized board (real error condition)
//   if (board_sum < 10.0f) {
//     printf("[CHESS_FATAL] Board sum is %.1f - board appears uninitialized!\n", board_sum);
//     printf("  This suggests the environment reset is not working properly.\n");
//     exit(1);
//   }
  
//   // 0 legal moves is normal for terminal positions - no need to spam logs

//   // SENTINEL 3: Validate perspective correctness
//   PieceColor current_turn = ctx->board.to_move;
// //   printf("[MONITOR_OK] Chess.h: Generated %s observation (offset=%d, board_sum=%.1f, mask_sum=%.0f) on %s's turn\n",
// //          (player == C_WHITE) ? "WHITE" : "BLACK", obs_offset, board_sum, mask_sum,
// //          (current_turn == C_WHITE) ? "WHITE" : "BLACK");
// }

// // void compute_observation_with_perspective(CChess *env, ChessContext *ctx) {
// //   PROFILE_START(profile_compute_obs_ticks)
  
// //   // // printf("[COMPUTE_OBS] Called for turn=%s, fullmove=%d\n",
// //   //        (ctx->board.to_move == C_WHITE) ? "WHITE" : "BLACK", ctx->board.fullmove_number);

// //   // In a dual-agent environment, we must generate an observation for BOTH agents
// //   // every time the state changes. Each agent gets its own slice of the observation buffer.
// //   // The 'observations' pointer in the CChess struct points to the start of the memory
// //   // block for this specific game environment.

// //   if (ctx->self_play_mode && ctx->dual_agent_self_play_mode) {
// //     // This is the standard 2-player self-play mode.

// //     // Define the size of a single agent's observation to calculate offsets.
// //     const int single_obs_size = 1537; // 1472 board + sparse mask (1 + 64)

// //     // === Generate Observation for Agent 0 (White) ===
// //     // This observation will be written to the first slice of the environment's
// //     // observation buffer, starting at offset 0.
// //     int white_obs_offset = 0;
// //     compute_single_agent_observation(env, ctx, C_WHITE, white_obs_offset);
// //     validate_chess_observation_integrity(env, ctx, C_WHITE, white_obs_offset);

// //     // === Generate Observation for Agent 1 (Black) ===
// //     // This observation will be written to the second slice, starting immediately
// //     // after Agent 0's data.
// //     int black_obs_offset = single_obs_size;
// //     compute_single_agent_observation(env, ctx, C_BLACK, black_obs_offset);
// //     validate_chess_observation_integrity(env, ctx, C_BLACK, black_obs_offset);

// //   } else {
// //     // This logic handles a single-agent mode (e.g., playing vs. an engine).
// //     // In this case, there is only one agent, and it always writes to offset 0.
// //     PieceColor current_player = ctx->board.to_move;
// //     compute_single_agent_observation(env, ctx, current_player, 0);
// //   }

// //   PROFILE_STOP(profile_compute_obs_ticks)
// // }

// // In chess.h
// void compute_observation_with_perspective(CChess *env, ChessContext *ctx) {
//   PROFILE_START(profile_compute_obs_ticks)

//   // The board state reflects the current player's turn.
//   // We generate the observation for this player from their perspective.
//   PieceColor current_player = ctx->board.to_move;

//   // The obs_offset is always 0 because the framework expects one observation
//   // per env.
//   int obs_offset = 0;

//   // Compute the observation for the current player and write it to the start of
//   // the buffer.
//   compute_single_agent_observation(env, ctx, current_player, obs_offset);

//   // Optional: you could add a validation check here for the single observation
//   validate_chess_observation_integrity(env, ctx, current_player, obs_offset);

//   PROFILE_STOP(profile_compute_obs_ticks)
// }

// // === MAKE/UNMAKE MOVE FUNCTIONS FOR FAST LEGALITY CHECKING ===

// static void make_move_fast(ChessBoard *board, ChessMove move, UndoInfo *undo) {
//   PROFILE_START(profile_make_move_fast_ticks)
//   // Save current state for undo
//   undo->old_castle_rights = board->castle_rights;
//   undo->old_ep_square = board->ep_square;
//   undo->old_halfmove_clock = board->halfmove_clock;
//   undo->old_zobrist_hash = board->zobrist_hash;
//   undo->was_castling = move.is_castling;
//   undo->was_en_passant = move.is_en_passant;

//   PieceColor moving_color = board->to_move;
//   Piece *from_piece = get_piece(board, move.from.x, move.from.y);
//   Piece *to_piece = get_piece(board, move.to.x, move.to.y);

//   // Check for null pointers before dereferencing
//   if (!from_piece || !to_piece) {
//     return; // Invalid move coordinates
//   }

//   // Save captured piece
//   undo->captured_piece = *to_piece;

//   if (move.is_castling) {
//     // Handle castling
//     int rank = (moving_color == C_WHITE) ? 0 : 7;
//     bool kingside = (move.to.x == 6);

//     // Save rook positions for undo
//     undo->rook_from.x = kingside ? 7 : 0;
//     undo->rook_from.y = rank;
//     undo->rook_to.x = kingside ? 5 : 3;
//     undo->rook_to.y = rank;

//     // Move king
//     from_piece->type = EMPTY;
//     from_piece->color = C_NO_COLOR;
//     to_piece->type = KING;
//     to_piece->color = moving_color;

//     // Move rook
//     Piece *rook_from = get_piece(board, undo->rook_from.x, undo->rook_from.y);
//     Piece *rook_to = get_piece(board, undo->rook_to.x, undo->rook_to.y);
//     if (!rook_from || !rook_to) {
//       return; // Invalid rook coordinates
//     }
//     *rook_to = *rook_from;
//     rook_from->type = EMPTY;
//     rook_from->color = C_NO_COLOR;

//     // Update castling rights
//     if (moving_color == C_WHITE) {
//       board->castle_rights &= ~0x3; // Clear white castling
//     } else {
//       board->castle_rights &= ~0xC; // Clear black castling
//     }
//   } else if (move.is_en_passant) {
//     // Handle en passant
//     int captured_y = (moving_color == C_WHITE) ? move.to.y - 1 : move.to.y + 1;
//     Piece *captured_pawn = get_piece(board, move.to.x, captured_y);

//     if (!captured_pawn) {
//       return; // Invalid en passant coordinates
//     }

//     // Save the actual captured pawn position for undo
//     undo->captured_piece = *captured_pawn;

//     // Remove captured pawn
//     captured_pawn->type = EMPTY;
//     captured_pawn->color = C_NO_COLOR;

//     // Move capturing pawn
//     from_piece->type = EMPTY;
//     from_piece->color = C_NO_COLOR;
//     to_piece->type = PAWN;
//     to_piece->color = moving_color;
//   } else {
//     // Regular move
//     PieceType original_type = from_piece->type;
//     PieceType final_type =
//         (move.promotion != EMPTY) ? move.promotion : original_type;

//     // Update castling rights if king or rook moves (before clearing piece)
//     if (original_type == KING) {
//       if (moving_color == C_WHITE) {
//         board->castle_rights &= ~0x3; // Clear white castling
//       } else {
//         board->castle_rights &= ~0xC; // Clear black castling
//       }
//     } else if (original_type == ROOK) {
//       if (moving_color == C_WHITE) {
//         if (move.from.x == 0)
//           board->castle_rights &= ~0x2; // White queenside
//         if (move.from.x == 7)
//           board->castle_rights &= ~0x1; // White kingside
//       } else {
//         if (move.from.x == 0)
//           board->castle_rights &= ~0x8; // Black queenside
//         if (move.from.x == 7)
//           board->castle_rights &= ~0x4; // Black kingside
//       }
//     }

//     // Move piece
//     from_piece->type = EMPTY;
//     from_piece->color = C_NO_COLOR;
//     to_piece->type = final_type;
//     to_piece->color = moving_color;
//   }

//   // Update en passant square (need to check original piece type for regular
//   // moves)
//   board->ep_square = -1;
//   if (!move.is_castling && !move.is_en_passant) {
//     PieceType original_type = (move.promotion != EMPTY) ? PAWN : to_piece->type;
//     if (original_type == PAWN && abs(move.to.y - move.from.y) == 2) {
//       board->ep_square = move.to.x + ((move.from.y + move.to.y) / 2) * 8;
//     }
//   }

//   // Update halfmove clock (need to check original piece type)
//   PieceType moved_piece_type = EMPTY;
//   if (move.is_castling) {
//     moved_piece_type = KING;
//   } else if (move.is_en_passant) {
//     moved_piece_type = PAWN;
//   } else {
//     moved_piece_type = (move.promotion != EMPTY) ? PAWN : to_piece->type;
//   }

//   if (moved_piece_type == PAWN || undo->captured_piece.type != EMPTY) {
//     board->halfmove_clock = 0;
//   } else {
//     board->halfmove_clock++;
//   }

//   // Change side to move
//   board->to_move = (moving_color == C_WHITE) ? C_BLACK : C_WHITE;
//   PROFILE_STOP(profile_make_move_fast_ticks)
// }

// static void unmake_move_fast(ChessBoard *board, ChessMove move,
//                              UndoInfo *undo) {
//   PROFILE_START(profile_unmake_move_fast_ticks)
//   // Restore board state from undo information
//   board->castle_rights = undo->old_castle_rights;
//   board->ep_square = undo->old_ep_square;
//   board->halfmove_clock = undo->old_halfmove_clock;
//   board->zobrist_hash = undo->old_zobrist_hash;

//   // Restore side to move
//   PieceColor moving_color = (board->to_move == C_WHITE) ? C_BLACK : C_WHITE;
//   board->to_move = moving_color;

//   Piece *from_piece = get_piece(board, move.from.x, move.from.y);
//   Piece *to_piece = get_piece(board, move.to.x, move.to.y);

//   // Check for null pointers before dereferencing
//   if (!from_piece || !to_piece) {
//     PROFILE_STOP(profile_unmake_move_fast_ticks)
//     return; // Invalid move coordinates
//   }

//   if (undo->was_castling) {
//     // Undo castling
//     // Restore king
//     from_piece->type = KING;
//     from_piece->color = moving_color;
//     to_piece->type = EMPTY;
//     to_piece->color = C_NO_COLOR;

//     // Restore rook
//     Piece *rook_from = get_piece(board, undo->rook_from.x, undo->rook_from.y);
//     Piece *rook_to = get_piece(board, undo->rook_to.x, undo->rook_to.y);
//     if (!rook_from || !rook_to) {
//       PROFILE_STOP(profile_unmake_move_fast_ticks)
//       return; // Invalid rook coordinates
//     }
//     rook_from->type = ROOK;
//     rook_from->color = moving_color;
//     rook_to->type = EMPTY;
//     rook_to->color = C_NO_COLOR;
//   } else if (undo->was_en_passant) {
//     // Undo en passant
//     int captured_y = (moving_color == C_WHITE) ? move.to.y - 1 : move.to.y + 1;
//     Piece *captured_pawn = get_piece(board, move.to.x, captured_y);

//     if (!captured_pawn) {
//       PROFILE_STOP(profile_unmake_move_fast_ticks)
//       return; // Invalid en passant coordinates
//     }

//     // Restore captured pawn
//     *captured_pawn = undo->captured_piece;

//     // Restore moving pawn
//     from_piece->type = PAWN;
//     from_piece->color = moving_color;
//     to_piece->type = EMPTY;
//     to_piece->color = C_NO_COLOR;
//   } else {
//     // Undo regular move
//     // Restore original piece (handle promotion)
//     from_piece->type = (move.promotion != EMPTY) ? PAWN : to_piece->type;
//     from_piece->color = moving_color;

//     // Restore captured piece
//     *to_piece = undo->captured_piece;
//   }
//   PROFILE_STOP(profile_unmake_move_fast_ticks)
// }

// // === MOVE APPLICATION ===

// static bool apply_uci_move(ChessContext *ctx, const char *uci_str) {
//   PROFILE_START(profile_apply_uci_move_ticks)
//   if (strlen(uci_str) < 4) {
//     PROFILE_STOP(profile_apply_uci_move_ticks)
//     return false;
//   }

//   int from_x = uci_str[0] - 'a';
//   int from_y = uci_str[1] - '1';
//   int to_x = uci_str[2] - 'a';
//   int to_y = uci_str[3] - '1';

//   if (from_x < 0 || from_x >= 8 || from_y < 0 || from_y >= 8 || to_x < 0 ||
//       to_x >= 8 || to_y < 0 || to_y >= 8) {
//     PROFILE_STOP(profile_apply_uci_move_ticks)
//     return false;
//   }

//   ChessBoard *board = &ctx->board;
//   PieceColor us = board->to_move;

//   // Store old values for Zobrist updates
//   uint8_t old_castle_rights = board->castle_rights;
//   int8_t old_ep_square = board->ep_square;

//   // Get piece being moved
//   Piece *from_piece = get_piece(board, from_x, from_y);
//   if (!from_piece || from_piece->type == EMPTY || from_piece->color != us) {
//     PROFILE_STOP(profile_apply_uci_move_ticks)
//     return false;
//   }

//   Piece moving_piece = *from_piece;
//   Piece *captured_piece = get_piece(board, to_x, to_y);
//   bool is_capture = (captured_piece->type != EMPTY);
//   Piece captured_piece_copy = *captured_piece; // Store for Zobrist update

//   // XOR out old state from hash
//   board->zobrist_hash ^= zobrist_side_to_move; // Change side to move
//   if (old_ep_square >= 0) {
//     board->zobrist_hash ^= zobrist_en_passant[old_ep_square];
//   }
//   board->zobrist_hash ^= zobrist_castle_rights[old_castle_rights];

//   // Handle castling (special UCI format: king moves 2 squares)
//   if (moving_piece.type == KING && abs(to_x - from_x) == 2) {
//     int rank = (us == C_WHITE) ? 0 : 7;
//     bool kingside = (to_x > from_x);

//     // Validate castling rights and path
//     int rook_from = kingside ? 7 : 0;
//     int rook_to = kingside ? 5 : 3;

//     // Check if path is clear (king's path already checked in legal move gen)
//     for (int x = (kingside ? 5 : 1); x <= (kingside ? 6 : 3); x++) {
//       if (x != from_x && get_piece_const(board, x, rank)->type != EMPTY) {
//         PROFILE_STOP(profile_apply_uci_move_ticks)
//         return false;
//       }
//     }

//     // XOR out old piece positions
//     board->zobrist_hash ^= zobrist_piece_square[us][KING][from_y * 8 + from_x];
//     board->zobrist_hash ^= zobrist_piece_square[us][ROOK][rank * 8 + rook_from];

//     // Move king
//     from_piece->type = EMPTY;
//     from_piece->color = C_NO_COLOR;
//     get_piece(board, to_x, to_y)->type = KING;
//     get_piece(board, to_x, to_y)->color = us;

//     // Move rook
//     Piece *rook_piece = get_piece(board, rook_from, rank);
//     if (rook_piece->type != ROOK || rook_piece->color != us) {
//       PROFILE_STOP(profile_apply_uci_move_ticks)
//       return false;
//     }
//     rook_piece->type = EMPTY;
//     rook_piece->color = C_NO_COLOR;
//     get_piece(board, rook_to, rank)->type = ROOK;
//     get_piece(board, rook_to, rank)->color = us;

//     // XOR in new piece positions
//     board->zobrist_hash ^= zobrist_piece_square[us][KING][to_y * 8 + to_x];
//     board->zobrist_hash ^= zobrist_piece_square[us][ROOK][rank * 8 + rook_to];

//     // Update castling rights
//     if (us == C_WHITE) {
//       board->castle_rights &= ~0x3; // Clear white castling
//     } else {
//       board->castle_rights &= ~0xC; // Clear black castling
//     }
//   }
//   // Handle en passant capture
//   else if (moving_piece.type == PAWN && (to_y * 8 + to_x) == board->ep_square) {
//     // This is a confirmed en passant capture
//     int captured_y = (us == C_WHITE) ? to_y - 1 : to_y + 1;
//     Piece *en_passant_piece = get_piece(board, to_x, captured_y);

//     // This check is now for sanity, the ep_square check is the real guard
//     if (!en_passant_piece || en_passant_piece->type != PAWN ||
//         en_passant_piece->color == us) {
//       PROFILE_STOP(profile_apply_uci_move_ticks)
//       return false;
//     }

//     // XOR out old pieces
//     board->zobrist_hash ^= zobrist_piece_square[us][PAWN][from_y * 8 + from_x];
//     board->zobrist_hash ^=
//         zobrist_piece_square[1 - us][PAWN][captured_y * 8 + to_x];

//     // Remove the en passant captured pawn
//     en_passant_piece->type = EMPTY;
//     en_passant_piece->color = C_NO_COLOR;

//     // Move the capturing pawn
//     from_piece->type = EMPTY;
//     from_piece->color = C_NO_COLOR;
//     get_piece(board, to_x, to_y)->type = PAWN;
//     get_piece(board, to_x, to_y)->color = us;

//     // XOR in new pawn position
//     board->zobrist_hash ^= zobrist_piece_square[us][PAWN][to_y * 8 + to_x];
//   }
//   // Regular move
//   else {
//     // XOR out old piece position
//     board->zobrist_hash ^=
//         zobrist_piece_square[us][moving_piece.type][from_y * 8 + from_x];

//     // XOR out captured piece if any
//     if (is_capture) {
//       board->zobrist_hash ^=
//           zobrist_piece_square[captured_piece_copy.color]
//                               [captured_piece_copy.type][to_y * 8 + to_x];
//     }

//     // Clear source square
//     from_piece->type = EMPTY;
//     from_piece->color = C_NO_COLOR;

//     // Place piece on destination
//     get_piece(board, to_x, to_y)->type = moving_piece.type;
//     get_piece(board, to_x, to_y)->color = moving_piece.color;

//     // Handle promotion
//     PieceType final_type = moving_piece.type;
//     if (strlen(uci_str) == 5 && moving_piece.type == PAWN) {
//       char promo = uci_str[4];
//       switch (promo) {
//       case 'q':
//         final_type = QUEEN;
//         break;
//       case 'r':
//         final_type = ROOK;
//         break;
//       case 'b':
//         final_type = BISHOP;
//         break;
//       case 'n':
//         final_type = KNIGHT;
//         break;
//       default:
//         PROFILE_STOP(profile_apply_uci_move_ticks)
//         return false;
//       }
//       get_piece(board, to_x, to_y)->type = final_type;
//     }

//     // XOR in new piece position
//     board->zobrist_hash ^=
//         zobrist_piece_square[us][final_type][to_y * 8 + to_x];
//   }

//   // Update castling rights when king or rook moves
//   if (moving_piece.type == KING) {
//     if (us == C_WHITE) {
//       board->castle_rights &= ~0x3; // Clear white castling
//     } else {
//       board->castle_rights &= ~0xC; // Clear black castling
//     }
//   } else if (moving_piece.type == ROOK) {
//     if (us == C_WHITE) {
//       if (from_x == 0)
//         board->castle_rights &= ~0x2; // White queenside
//       if (from_x == 7)
//         board->castle_rights &= ~0x1; // White kingside
//     } else {
//       if (from_x == 0)
//         board->castle_rights &= ~0x8; // Black queenside
//       if (from_x == 7)
//         board->castle_rights &= ~0x4; // Black kingside
//     }
//   }

//   // Set en passant square
//   board->ep_square = -1;
//   if (moving_piece.type == PAWN && abs(to_y - from_y) == 2) {
//     board->ep_square = to_x + ((from_y + to_y) / 2) * 8;
//   }

//   // Update halfmove clock
//   if (moving_piece.type == PAWN || is_capture) {
//     board->halfmove_clock = 0;
//   } else {
//     board->halfmove_clock++;
//   }

//   // Update game state
//   board->to_move = (us == C_WHITE) ? C_BLACK : C_WHITE;
//   if (us == C_BLACK) {
//     board->fullmove_number++;
//   }

//   // XOR in new state
//   board->zobrist_hash ^= zobrist_castle_rights[board->castle_rights];
//   if (board->ep_square >= 0) {
//     board->zobrist_hash ^= zobrist_en_passant[board->ep_square];
//   }

//   // Add current position to history for threefold repetition detection
//   add_position_to_history(ctx, board->zobrist_hash);

//   // Clear caches
//   ctx->white_moves_cached = false;
//   ctx->black_moves_cached = false;
//   ctx->position_fully_cached = false;
//   ctx->observation_cached = false;
//   ctx->step_count++;
//   // Add to complete game log - store the canonical UCI move
//   if (ctx->complete_game_action_count < 1024) {
//     strcpy(ctx->complete_game_moves[ctx->complete_game_action_count], uci_str);
//     ctx->complete_game_action_count++;
//   } else {
//   }

//   PROFILE_STOP(profile_apply_uci_move_ticks)
//   return true;
// }

// // === DRAW DETECTION FUNCTIONS ===

// static void add_position_to_history(ChessContext *ctx, uint64_t hash) {
//   PositionHistory *history = &ctx->position_history;

//   // Look for existing hash
//   for (int i = 0; i < history->size; i++) {
//     if (history->hashes[i] == hash) {
//       history->counts[i]++;
//       return;
//     }
//   }

//   // Add new hash if space available
//   if (history->size < POSITION_HISTORY_SIZE) {
//     history->hashes[history->size] = hash;
//     history->counts[history->size] = 1;
//     history->size++;
//   }
//   // If history is full, we don't track this position (rare case)
// }

// static int get_position_count(ChessContext *ctx, uint64_t hash) {
//   PositionHistory *history = &ctx->position_history;
//   for (int i = 0; i < history->size; i++) {
//     if (history->hashes[i] == hash) {
//       return history->counts[i];
//     }
//   }
//   return 0;
// }

// static bool is_threefold_repetition(ChessContext *ctx) {
//   uint64_t current_hash = ctx->board.zobrist_hash;
//   return get_position_count(ctx, current_hash) >= 3;
// }

// static bool is_insufficient_material(ChessContext *ctx) {
//   ChessBoard *board = &ctx->board;

//   // Count material for both sides
//   int white_pawns = 0, black_pawns = 0;
//   int white_knights = 0, black_knights = 0;
//   int white_bishops = 0, black_bishops = 0;
//   int white_rooks = 0, black_rooks = 0;
//   int white_queens = 0, black_queens = 0;

//   for (int i = 0; i < 64; i++) {
//     const Piece *p = &board->board[i];
//     if (p->type == EMPTY)
//       continue;

//     switch (p->type) {
//     case PAWN:
//       if (p->color == C_WHITE)
//         white_pawns++;
//       else
//         black_pawns++;
//       break;
//     case KNIGHT:
//       if (p->color == C_WHITE)
//         white_knights++;
//       else
//         black_knights++;
//       break;
//     case BISHOP:
//       if (p->color == C_WHITE)
//         white_bishops++;
//       else
//         black_bishops++;
//       break;
//     case ROOK:
//       if (p->color == C_WHITE)
//         white_rooks++;
//       else
//         black_rooks++;
//       break;
//     case QUEEN:
//       if (p->color == C_WHITE)
//         white_queens++;
//       else
//         black_queens++;
//       break;
//     default:
//       break;
//     }
//   }

//   // Any pawns, rooks, or queens means sufficient material
//   if (white_pawns > 0 || black_pawns > 0 || white_rooks > 0 ||
//       black_rooks > 0 || white_queens > 0 || black_queens > 0) {
//     return false;
//   }

//   // Count total minor pieces for each side
//   int white_minor = white_knights + white_bishops;
//   int black_minor = black_knights + black_bishops;

//   // Insufficient material cases:
//   // King vs King
//   if (white_minor == 0 && black_minor == 0)
//     return true;

//   // King + minor piece vs King
//   if ((white_minor <= 1 && black_minor == 0) ||
//       (black_minor <= 1 && white_minor == 0))
//     return true;

//   // King + Bishop vs King + Bishop (same color squares) - simplified to any
//   // bishop vs bishop
//   if (white_minor == 1 && black_minor == 1 && white_bishops == 1 &&
//       black_bishops == 1)
//     return true;

//   return false;
// }

// static int global_env_counter = 0;
// static int logging_env_id = -1;  // ID of the designated logging environment
// static int first_active_env_id = -1;  // Track the first environment that actually runs games

// void init(CChess *env) {
//   memset(&env->context, 0, sizeof(ChessContext));
//   memset(&env->log, 0, sizeof(Log));

//   // Initialize puzzle logging fields with default values
//   env->log.puzzle_difficulty = 1.0f; // Default to difficulty 1
//   env->log.puzzle_success_rate = 0.0f; // Will be calculated post-aggregation
  
//   // Initialize puzzle tracking
//   env->context.puzzle_attempts_this_env = 0;
//   env->context.puzzle_solved_this_env = 0;
  
//   // Initialize new puzzle training fields
//   env->context.puzzle_tries_this_env = 0;
//   env->context.puzzle_max_tries_per_env = 10;  // Default
//   env->context.puzzle_start_time = 0;
//   env->context.puzzle_samples_to_solve = 0;
  
//   // Initialize global puzzle coordination (first env sets defaults)
//   if (global_env_counter == 1) {  // First environment
//     env->global_puzzle_id = 0;
//     env->global_puzzle_attempts = 0;
//     env->global_puzzle_successes = 0;
//     env->global_puzzle_success_threshold = 0.9f;
//     env->puzzle_max_tries_per_env = 10;
//   }

//   // Set up convenience pointer to avoid repeated dereferencing
//   env->ctx = &env->context;
//   env->env_id = global_env_counter++;  // Simple counter
  
//   // Designate the first environment as the logging environment
//   if (logging_env_id == -1) {
//     logging_env_id = env->env_id;
//   }

//   init_board(&env->context.board);
//   env->context.dual_agent_self_play_mode = true; // Default to self-play

//   // Copy reward config to context
//   env->context.c_reward_valid = env->reward_valid;
//   env->context.c_reward_invalid_white = env->reward_invalid_white;
//   env->context.c_reward_invalid_black = env->reward_invalid_black;
//   env->context.c_reward_white_captures_enemy_piece =
//       env->reward_white_captures_enemy_piece;
//   env->context.c_reward_black_captures_enemy_piece =
//       env->reward_black_captures_enemy_piece;
//   env->context.c_reward_max_depth_termination = env->reward_max_depth_termination;
//   env->context.c_use_piece_value_capture_rewards = env->use_piece_value_capture_rewards;
//   env->context.c_piece_value_reward_multiplier = env->piece_value_reward_multiplier;
//   env->context.c_reward_draw = env->reward_draw;
//   env->context.c_reward_win_white = env->reward_win_white;
//   env->context.c_reward_win_black = env->reward_win_black;
//   env->context.c_reward_loss_white = env->reward_loss_white;
//   env->context.c_reward_loss_black = env->reward_loss_black;
//   env->context.c_reward_check_white = env->reward_check_white;
//   env->context.c_reward_check_black = env->reward_check_black;
//   env->context.c_reward_material_diff_white = env->reward_material_diff_white;
//   env->context.c_reward_material_diff_black = env->reward_material_diff_black;
  
//   // Copy puzzle reward config to context
//   env->context.c_reward_puzzle_solved = env->reward_puzzle_solved;
//   env->context.c_reward_puzzle_failed = env->reward_puzzle_failed;
//   env->context.c_reward_correct_move = env->reward_correct_move;
  
//   // Initialize puzzle mode state
//   env->context.puzzle_mode = false;
//   env->context.puzzle_solution_length = 0;
//   env->context.puzzle_move_index = 0;
//   env->context.puzzle_completed = false;
//   env->context.puzzle_failed = false;
// }

// // Puzzle mode functions
// void set_puzzle_mode(CChess *env, bool enabled) {
//   env->context.puzzle_mode = enabled;
// }

// void set_puzzle_data(CChess *env, const char* fen, const char* solution_moves[], int solution_length) {
//   if (!env->context.puzzle_mode) return;
  
//   // Reset tries counter when loading new puzzle
//   env->context.puzzle_tries_this_env = 0;
  
//   // Store puzzle FEN
//   strncpy(env->context.puzzle_fen, fen, sizeof(env->context.puzzle_fen) - 1);
//   env->context.puzzle_fen[sizeof(env->context.puzzle_fen) - 1] = '\0';
  
//   // Store solution moves
//   env->context.puzzle_solution_length = (solution_length > 10) ? 10 : solution_length;
//   for (int i = 0; i < env->context.puzzle_solution_length; i++) {
//     strncpy(env->context.puzzle_solution[i], solution_moves[i], 5);
//     env->context.puzzle_solution[i][5] = '\0';
//   }
  
//   // Reset puzzle state
//   env->context.puzzle_move_index = 0;
//   env->context.puzzle_completed = false;
//   env->context.puzzle_failed = false;
  
//   // Load the FEN position
//   c_set_fen(env, fen);
  
//   // Print initial puzzle position
//   // printf("\n[PUZZLE] New puzzle loaded (ID: %d)\n", env->global_puzzle_id);
//   printf("FEN: %s\n", fen);
//   printf("Solution length: %d moves\n", solution_length);
//   printf("Solution: ");
//   for (int i = 0; i < solution_length; i++) {
//     printf("%s ", solution_moves[i]);
//   }
//   printf("\n");
  
//   // Verify this is a mate-in-1 puzzle (white to move, 1 move solution)
//   if (solution_length != 1) {
//     // Warning: Not mate-in-1 solution_length);
//   }
//   if (env->context.board.to_move != C_WHITE) {
//     // printf("[PUZZLE ERROR] Puzzle starts with BLACK to move! Only WHITE should move in puzzles.\n");
//   }
  
//   printf("Initial position:\n");
//   print_board_state(&env->context.board);
// }

// void set_puzzle_difficulty(CChess *env, int difficulty) {
//   env->log.puzzle_difficulty = (float)difficulty;
// }

// void set_puzzle_training_params(CChess *env, int max_tries_per_env, float success_threshold) {
//   env->puzzle_max_tries_per_env = max_tries_per_env;
//   env->global_puzzle_success_threshold = success_threshold;
//   env->context.puzzle_max_tries_per_env = max_tries_per_env;
// }

// void allocate(CChess *env) {
//   // Allocate RL interface arrays for PufferLib
//   // Chess has 2 players but typically trains as single agent with perspective
//   // flipping
//   const int num_players = 2;
//   const int obs_size =
//       1537; // 23*8*8 board planes + sparse action mask = 1472 + 1 + 64

//   env->observations = (float *)calloc(num_players * obs_size, sizeof(float));
//   env->actions = (int *)calloc(num_players, sizeof(int));
//   env->rewards = (float *)calloc(num_players, sizeof(float));
//   env->terminals = (unsigned char *)calloc(num_players, sizeof(unsigned char));

//   init(env);
// }

// void free_allocated(CChess *env) {
//   // Free RL interface arrays allocated by allocate()
//   free(env->observations);
//   free(env->actions);
//   free(env->rewards);
//   free(env->terminals);

//   c_close(env);
// }

// void c_reset(CChess *env) {
//   // DEBUG: Print when reset is called
//   printf("[DEBUG] c_reset called - puzzle_mode=%d, puzzle_completed=%d\n", 
//          env->context.puzzle_mode, env->context.puzzle_completed);
  
//   // Preserve puzzle difficulty and stats across resets
//   float saved_puzzle_difficulty = env->log.puzzle_difficulty;
//   int saved_puzzle_attempts = env->context.puzzle_attempts_this_env;
//   int saved_puzzle_solved = env->context.puzzle_solved_this_env;
  
//   // Only init board if no FEN was set
//   if (!env->context.board.fen_was_set) {
//     init_board(&env->context.board);
//   }
//   env->context.board.fen_was_set = false; // Reset flag after use

//   // Reset terminals and rewards for both agents
//   env->terminals[0] = 0;
//   env->terminals[1] = 0;
//   env->rewards[0] = 0.0f;
//   env->rewards[1] = 0.0f;

//   // Reset episode tracking
//   env->context.step_count = 0;
//   env->context.episode_return_white = 0.0f;
//   env->context.episode_return_black = 0.0f;
  
//   env->context.complete_game_action_count = 0;
//   env->context.serialized_moves[0] =
//       '\0'; // Initialize serialized_moves buffer to empty
//   env->context.steps_since_last_log = 0;
  
//   // Reset puzzle state for new episode
//   if (env->context.puzzle_mode) {
//     env->context.puzzle_completed = false;
//     env->context.puzzle_failed = false;
//     env->context.puzzle_move_index = 0;
//     // Reset episode-specific puzzle stats
//     env->context.puzzle_attempts_this_episode = 0.0f;
//     env->context.puzzle_correct_moves_this_episode = 0.0f;
//     env->context.puzzle_wrong_moves_this_episode = 0.0f;
//     env->context.puzzle_solved_this_episode = 0.0f;
//   }
  
//   // Don't reset game logging frequency - it's set once at init
//   // env->context.game_logging_frequency = 500000;
//   // DEBUG: Explicitly preserve the logging frequency that was set during init
//   if (env->context.game_logging_frequency == 0) {
//     // This shouldn't happen - frequency should be preserved from init
//   }

//   // Reset statistics
//   env->context.c_white_moves = 0;
//   env->context.c_black_moves = 0;
//   env->context.c_valid_moves = 0;
//   env->context.c_invalid_moves_white = 0;
//   env->context.c_invalid_moves_black = 0;

//   // Reset game outcome counters (CRITICAL BUG FIX)
//   env->context.c_white_win = 0;
//   env->context.c_black_win = 0;
//   env->context.c_white_loss = 0;
//   env->context.c_black_loss = 0;
//   env->context.c_game_drawn = 0;
//   env->context.c_max_depth = 0;
//   env->context.c_white_checkmated = 0;
//   env->context.c_black_checkmated = 0;
//   env->context.c_stalemate = 0;
//   env->context.c_insufficient_material = 0;
//   env->context.c_threefold_repetition = 0;
//   env->context.c_fifty_move_rule = 0;

//   // Reset accumulated reward counters
//   env->context.accumulated_reward_valid = 0.0f;
//   env->context.accumulated_reward_white_captures_enemy_piece = 0.0f;
//   env->context.accumulated_reward_black_captures_enemy_piece = 0.0f;
//   env->context.accumulated_reward_draw = 0.0f;
//   env->context.accumulated_reward_win_white = 0.0f;
//   env->context.accumulated_reward_win_black = 0.0f;
//   env->context.accumulated_reward_loss_white = 0.0f;
//   env->context.accumulated_reward_loss_black = 0.0f;
//   env->context.accumulated_reward_draw_white = 0.0f;
//   env->context.accumulated_reward_draw_black = 0.0f;
//   env->context.accumulated_reward_check_white = 0.0f;
//   env->context.accumulated_reward_check_black = 0.0f;
//   env->context.accumulated_reward_material_diff_white = 0.0f;
//   env->context.accumulated_reward_material_diff_black = 0.0f;
//   env->context.accumulated_stockfish_eval = 0.0f;
  
//   // Reset puzzle reward accumulation
//   env->context.accumulated_reward_puzzle_solved = 0.0f;
//   env->context.accumulated_reward_puzzle_failed = 0.0f;
//   env->context.accumulated_reward_puzzle_correct_move = 0.0f;
  
//   // Reset puzzle stats accumulation
//   env->context.puzzle_attempts_this_episode = 0.0f;
//   env->context.puzzle_correct_moves_this_episode = 0.0f;
//   env->context.puzzle_wrong_moves_this_episode = 0.0f;
//   env->context.puzzle_solved_this_episode = 0.0f;

//   // Clear caches
//   env->context.white_moves_cached = false;
//   env->context.black_moves_cached = false;
//   env->context.position_fully_cached = false;
//   env->context.cached_board_hash = 0;
//   env->context.observation_cached = false;

// //   // Clear env->log rewards accumulation
// //   env->log.reward_puzzle_solved = 0.0f;
// //   env->log.reward_puzzle_failed = 0.0f;
// //   env->log.reward_puzzle_correct_move = 0.0f;
// //   env->log.episode_return_white = 0.0f;
// //   env->log.episode_return_black = 0.0f;
// //   env->log.episode_return = 0.0f;

//   // Clear position history
//   memset(&env->context.position_history, 0, sizeof(PositionHistory));

//   // Add starting position to history for threefold repetition detection
//   add_position_to_history(&env->context, env->context.board.zobrist_hash);

//   // Compute initial observation
//   compute_observation_with_perspective(env, &env->context);
  
//   // Restore puzzle difficulty and stats after reset
//   env->log.puzzle_difficulty = saved_puzzle_difficulty;
//   env->context.puzzle_attempts_this_env = saved_puzzle_attempts;
//   env->context.puzzle_solved_this_env = saved_puzzle_solved;
// }

// // void c_step(CChess *env) {
// //   PROFILE_START(profile_c_step_ticks)
  
// //   // DEBUG: Check if we're being called after puzzle completion
// //   if (env->context.puzzle_mode && env->context.puzzle_completed) {
// //     // printf("[DEBUG] c_step called AFTER puzzle completion! This should not happen!\n");
// //     printf("[DEBUG] Current player: %s, step_count: %d\n", 
// //            (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK",
// //            env->context.step_count);
// //   }
  
// //   // Game logging counter is incremented when games end, not on every step

// // //   // In self-play mode: agent 0 = white, agent 1 = black
// // //   // Get action from the agent whose turn it is
// // //   PieceColor current_player = env->context.board.to_move;
// // //   int agent_idx = (current_player == C_WHITE) ? 0 : 1;
  
// // //   printf("[C_STEP_ENTRY] Turn: %s (agent %d), fullmove: %d, halfmove: %d\n",
// // //          (current_player == C_WHITE) ? "WHITE" : "BLACK", agent_idx,
// // //          env->context.board.fullmove_number, env->context.board.halfmove_clock);
  
// // //   // CRITICAL: In dual-agent mode, only process actions for the current player
// // //   // SAFEGUARD: Handle training loop calling wrong agent
// // //   int correct_agent_idx = agent_idx;
// // //   if (env->context.dual_agent_self_play_mode) {
// // //     // Determine which agent should actually move based on board state
// // //     int expected_agent = (current_player == C_WHITE) ? 0 : 1;
    
// // //     if (agent_idx != expected_agent) {
// // //       printf("[TRAINING_LOOP_FIX] Board says %s's turn (agent %d), but called with agent %d. Using correct agent.\n",
// // //              (current_player == C_WHITE) ? "WHITE" : "BLACK", expected_agent, agent_idx);
// // //       correct_agent_idx = expected_agent;
// // //     }
// // //   }
  
// // //   // Use the action from the correct agent (the one whose turn it actually is)
// // //   int action_idx = env->actions[correct_agent_idx];

// //   // In episode-per-color architecture, Python wrapper handles agent assignment
// //   // and episode separation. C++ always uses agent_idx = 0 for the active player
// //   // during their episode (WHITE episode or BLACK episode)
// //   int agent_idx = 0;

// //   // The action is now correctly read from env->actions[0] for BOTH White and
// //   // Black.
// //   int action_idx = env->actions[agent_idx];

// //   // Generate moves for BOTH players if not cached
// //   // This ensures we always have the right moves available
// //   chess_generate_all_legal_moves(&env->context);
  
// //   // Calculate the correct observation offset for the CURRENT player.
// //   const int single_obs_size = 1537; // 1472 board + sparse mask (1 + 64)
// //   int obs_offset = 0; // <-- NEW, CORRECT LOGIC
// //   int mask_start_idx = 1472; // Start of action mask in observation
  
// //   // Check if the chosen action corresponds to a legal move
// //   bool action_is_legal = false;
  
// //   if (action_idx >= 0 && action_idx < TOTAL_CHESS_ACTIONS) {
// //     const char* chosen_uci = ACTION_ID_TO_UCI[action_idx];
    
// //     // Get current player's moves for validation
// //     char (*moves_buffer)[6];
// //     int moves_count;
// //     if (env->context.board.to_move == C_WHITE) {
// //       moves_buffer = env->context.white_legal_moves_buffer;
// //       moves_count = env->context.white_legal_moves_count;
// //     } else {
// //       moves_buffer = env->context.black_legal_moves_buffer;
// //       moves_count = env->context.black_legal_moves_count;
// //     }
    
// //     // Check if this action corresponds to any legal move
// //     for (int i = 0; i < moves_count; i++) {
// //       const char* legal_uci = moves_buffer[i];
// //       char perspective_uci[6];
      
// //       // For BLACK, we need to flip the canonical move to match the action space
// //       if (env->context.board.to_move == C_BLACK) {
// //         flip_uci_for_black_perspective(legal_uci, perspective_uci);
// //       } else {
// //         strcpy(perspective_uci, legal_uci);
// //       }
      
// //       if (strcmp(chosen_uci, perspective_uci) == 0) {
// //         action_is_legal = true;
// //         break;
// //       }      
// //     }
// //   }


// //   if (!action_is_legal) {
// //     printf("[ERROR] No legal actions available for %s agent!\n", 
// //             (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK");
// //     printf("[DEBUG] puzzle_mode=%d, puzzle_completed=%d, puzzle_failed=%d, step_count=%d\n",
// //             env->context.puzzle_mode, env->context.puzzle_completed, 
// //             env->context.puzzle_failed, env->context.step_count);
// //     printf("[DEBUG] action_idx=%d\n", action_idx);
// //     PROFILE_STOP(profile_c_step_ticks);
// //     return;
// //   }
  

// //   // Clear all agent rewards and terminals
// // //   for (int i = 0; i < 2; i++) {
// // //     env->rewards[i] = 0.0f;
// // //     env->terminals[i] = 0;
// // //   }
// //   env->rewards[agent_idx] = 0.0f;
// //   env->terminals[agent_idx] = 0;


// //   // Debug: Print action index in puzzle mode
// //   if (env->context.puzzle_mode && env->context.puzzle_tries_this_env < 5) {
// //     // // printf("[PUZZLE DEBUG] Action index received: %d (TOTAL_CHESS_ACTIONS=%d)\n", action_idx, TOTAL_CHESS_ACTIONS);
// //   }
  
// //   // Validate action before executing
// //   if (action_idx < 0 || action_idx >= TOTAL_CHESS_ACTIONS) {
// //     printf("[ERROR] Invalid action ID: %d (max=%d)\n", action_idx, TOTAL_CHESS_ACTIONS-1);
// //     return;
// //   }
  
// //   // --- START OF NEW VALIDATION LOGIC ---
  
// //   // 1. Get the UCI string for the action chosen by the policy.
// //   // ACTION_ID_TO_UCI always represents moves in white's perspective coordinate system.
// //   const char *uci_move_white_perspective = ACTION_ID_TO_UCI[action_idx];
// //   char uci_move_canonical[6];
  
// //   if (env->context.board.to_move == C_BLACK) {
// //     // The black agent chose this action based on its flipped perspective.
// //     // The action maps to a move in white perspective coordinates, but since
// //     // the black agent sees the board flipped, this move should be interpreted
// //     // as being from black's perspective and flipped to canonical coordinates.
// //     flip_uci_for_black_perspective(uci_move_white_perspective, uci_move_canonical);
// //   } else {
// //     // For white, the white perspective move IS the canonical move.
// //     strcpy(uci_move_canonical, uci_move_white_perspective);
// //   }

// //   // 2. Use the legal moves already generated at the start of c_step (OPTIMIZATION)
// //   //    No need to regenerate - we already have the definitive list.

// //   // 3. Check if the chosen move is in the freshly generated list.
// //   bool is_action_legal = false;
  
// //   // Get the correct move buffer
// //   char (*moves_buffer)[6];
// //   int moves_count;
// //   if (env->context.board.to_move == C_WHITE) {
// //     moves_buffer = env->context.white_legal_moves_buffer;
// //     moves_count = env->context.white_legal_moves_count;
// //   } else {
// //     moves_buffer = env->context.black_legal_moves_buffer;
// //     moves_count = env->context.black_legal_moves_count;
// //   }
  
// //   for (int i = 0; i < moves_count; i++) {
// //     if (strcmp(moves_buffer[i], uci_move_canonical) == 0) {
// //       is_action_legal = true;
// //       break;
// //     }
// //   }

// //   // 4. Validate against the ground truth.
// //   if (!is_action_legal) {
// //     const char* turn_color = (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK";
// //     // printf("[ERROR] Illegal move attempted (ground truth validation): action %d (%s) - %s's turn (agent %d)\n", 
// //     //        action_idx, uci_move_canonical, turn_color, agent_idx);
    

// //     // Invalidate the move, penalize the agent, and end the step without applying the move.
// //     if (env->context.board.to_move == C_WHITE) {
// //         env->context.c_invalid_moves_white += 1;
// //         env->rewards[agent_idx] += env->context.c_reward_invalid_white;
// //     } else {
// //         env->context.c_invalid_moves_black += 1;
// //         env->rewards[agent_idx] += env->context.c_reward_invalid_black;
// //     }
// //     // Don't apply the move, just recompute observation and return.
// //     compute_observation_with_perspective(env, &env->context);
// //     PROFILE_STOP(profile_c_step_ticks);
// //     return;
// //   }
  
// //   // --- END OF NEW VALIDATION LOGIC ---

// //   // Check if this move is a capture before applying it
// //   int from_x = (uci_move_canonical[0] - 'a');
// //   int from_y = (uci_move_canonical[1] - '1');
// //   int to_x = (uci_move_canonical[2] - 'a');
// //   int to_y = (uci_move_canonical[3] - '1');
  
// //   // Get the piece at the destination to check for capture
// //   Piece *destination_piece = get_piece(&env->context.board, to_x, to_y);
// //   bool is_capture = (destination_piece && destination_piece->type != EMPTY);
  
// //   // Check for en passant capture
// //   bool is_en_passant = false;
// //   if (!is_capture) {
// //     Piece *moving_piece = get_piece(&env->context.board, from_x, from_y);
// //     if (moving_piece && moving_piece->type == PAWN && 
// //         (to_y * 8 + to_x) == env->context.board.ep_square) {
// //       is_capture = true;
// //       is_en_passant = true;
// //     }
// //   }

// // //   // Apply the move
// // //   printf("[MOVE_APPLY] Applying move %s for %s\n", uci_move_canonical, 
// // //          (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK");
// //   // Store whose turn it was before the move (the player making the move)
// //   PieceColor moving_player = env->context.board.to_move;
  
// //   bool move_applied = apply_uci_move(&env->context, uci_move_canonical);
// // //   printf("[MOVE_APPLY] Move applied successfully: %s, new turn: %s\n", 
// // //          move_applied ? "YES" : "NO",
// // //          (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK");

// //   // PUZZLE MODE - Special handling
// //   if (env->context.puzzle_mode) {
// //     // In puzzle mode, only white should be moving
// //     if (moving_player != C_WHITE) {
// //       // printf("[PUZZLE ERROR] Black tried to move in puzzle mode! This should not happen!\n");
// //       env->terminals[agent_idx] = 1;
// //       return;
// //     }
    
// //     // Print board BEFORE the move
// //     // printf("\n[PUZZLE] Board before move (puzzle move %d/%d):\n", 
// //            env->context.puzzle_move_index + 1, env->context.puzzle_solution_length);
// //     print_board_state(&env->context.board);
    
// //     // Apply the move
// //     if (!move_applied) {
// //       // Move failed to apply
// //       env->terminals[agent_idx] = 1;
// //       return;
// //     }
    
// //     // Print board AFTER the move
// //     // printf("[PUZZLE] Move played: %s by WHITE\n", uci_move_canonical);
// //     print_board_state(&env->context.board);
// //   } else {
// //     // Non-puzzle mode - normal move application
// //     if (move_applied) {
// //       // Regular game logic here
// //     }
// //   }

// //   // PUZZLE MODE - Check if move is correct
// //   if (move_applied && env->context.puzzle_mode) {
// //     if (env->context.puzzle_move_index < env->context.puzzle_solution_length) {
// //       // Track timing on first move of puzzle
// //       if (env->context.puzzle_move_index == 0) {
// //         env->context.puzzle_samples_to_solve++;
        
// //         if (env->context.puzzle_start_time == 0) {
// //           env->context.puzzle_start_time = clock();
// //         }
// //       }
      
// //       // Check if the move matches the expected solution move
// //       const char* expected_move = env->context.puzzle_solution[env->context.puzzle_move_index];
      
// //       if (strcmp(uci_move_canonical, expected_move) == 0) {
// //         // Correct move! Award reward and advance to next move
// //         // Correct move 
// //                env->context.puzzle_move_index + 1, env->context.puzzle_solution_length);
// //         env->rewards[agent_idx] += env->reward_correct_move;
// //         env->context.accumulated_reward_puzzle_correct_move += env->reward_correct_move;
// //         env->context.episode_return_white += env->reward_correct_move;
// //         env->context.puzzle_move_index++;
        
// //         // Track correct moves in context for later logging
// //         env->context.puzzle_correct_moves_this_episode += 1.0f;
        
// //         // Check if puzzle is complete
// //         if (env->context.puzzle_move_index >= env->context.puzzle_solution_length) {
// //           // Puzzle solved! Award completion reward
// //           // printf("[PUZZLE] PUZZLE SOLVED! Terminating episode.\n");
// //           env->rewards[agent_idx] += env->reward_puzzle_solved;
// //           env->context.accumulated_reward_puzzle_solved += env->reward_puzzle_solved;
// //           env->context.episode_return_white += env->reward_puzzle_solved;
// //           env->context.puzzle_completed = true;
// //           env->global_puzzle_successes++;
          
// //           // Calculate performance metrics
// //           clock_t solve_time = clock() - env->context.puzzle_start_time;
// //           env->log.puzzle_avg_time_to_solve = (float)solve_time / CLOCKS_PER_SEC;
// //           env->log.puzzle_avg_samples_to_solve = (float)env->context.puzzle_samples_to_solve;
          
// //           // Check if global threshold reached (simplified check)
// //           if (env->global_puzzle_attempts >= 20) { // Minimum sample
// //             float global_success_rate = (float)env->global_puzzle_successes / env->global_puzzle_attempts;
// //             if (global_success_rate >= env->global_puzzle_success_threshold) {
// //               // Advance puzzle globally (reset counters)
// //               env->global_puzzle_id++;
// //               env->global_puzzle_attempts = 0;
// //               env->global_puzzle_successes = 0;
// //             }
// //           }
          
// //           // Track puzzle solved in context for later logging
// //           env->context.puzzle_solved_this_env++;
// //           env->context.puzzle_solved_this_episode += 1.0f;
          
// //           // Update global logging info (these are okay to set directly as they're global state)
// //           env->log.puzzle_current_id = (float)env->global_puzzle_id;
// //           env->log.puzzle_global_attempts = (float)env->global_puzzle_attempts;
// //           env->log.puzzle_global_successes = (float)env->global_puzzle_successes;
          
// //           // Reset for next puzzle
// //           env->context.puzzle_tries_this_env = 0;
// //           env->context.puzzle_samples_to_solve = 0;
// //           env->context.puzzle_start_time = 0;
          
// //           // Terminate episode - puzzle complete
// //           env->terminals[agent_idx] = 1;
// //           add_log(env);
// //           compute_observation_with_perspective(env, &env->context);
// //           return;
// //         } else {
// //           // More moves to make - recompute observation for next move
// //           // Continue to next move
// //           compute_observation_with_perspective(env, &env->context);
// //           return;
// //         }
// //       } else {
// //         // Wrong move! Give penalty and reward shaping
// //         float total_penalty = env->reward_puzzle_failed;
        
// //         // Parse the expected and actual moves
// //         ChessMove expected_move;
// //         if (!parse_uci_move(env->context.puzzle_solution[env->context.puzzle_move_index], &expected_move)) {
// //           // Failed to parse expected move 
// //                  env->context.puzzle_solution[env->context.puzzle_move_index]);
// //         } else {
// //           // Parse the actual move made
// //           ChessMove actual_move;
// //           char uci_move[6];
// //           action_to_uci(action_idx, uci_move);
// //           if (parse_uci_move(uci_move, &actual_move)) {
// //             // Reward shaping based on move similarity
// //             float shaping_reward = 0.0f;
            
// //             // 1. Reward for moving the correct piece
// //             if (actual_move.from.x == expected_move.from.x && actual_move.from.y == expected_move.from.y) {
// //               shaping_reward += env->reward_puzzle_correct_piece;
              
// //               // 2. Additional reward based on how close we moved to the target
// //               // Calculate Manhattan distance from actual destination to expected destination
// //               int expected_row = expected_move.to.y;
// //               int expected_col = expected_move.to.x;
// //               int actual_row = actual_move.to.y;
// //               int actual_col = actual_move.to.x;
              
// //               int distance = abs(expected_row - actual_row) + abs(expected_col - actual_col);
// //               // Max distance on board is 14 (7+7), so normalize
// //               float distance_reward = env->reward_puzzle_closer_to_target * (1.0f - (float)distance / 14.0f);
// //               shaping_reward += distance_reward;
              
// //               // 3. If promotion expected and we promoted to same piece, bonus
// //               if (expected_move.promotion != EMPTY && 
// //                   actual_move.promotion == expected_move.promotion) {
// //                 shaping_reward += env->reward_puzzle_correct_promotion;
// //               }
// //             }
            
// //             total_penalty += shaping_reward;
// //           }
// //         }
        
// //         env->rewards[agent_idx] += total_penalty;
// //         env->context.accumulated_reward_puzzle_failed += total_penalty;
// //         // In puzzle mode, only white plays, so only update white's episode return
// //         env->context.episode_return_white += total_penalty;
        
// //         // Track wrong moves in context for later logging
// //         env->context.puzzle_wrong_moves_this_episode += 1.0f;

// //         env->context.puzzle_tries_this_env++;
// //         env->global_puzzle_attempts++;
// //         env->context.puzzle_attempts_this_episode += 1.0f;
        
// //         // Check if we've exceeded max tries for this puzzle
// //         if (env->context.puzzle_tries_this_env >= env->context.puzzle_max_tries_per_env) {
// //           // Terminate episode after max tries
// //           env->terminals[agent_idx] = 1;
// //           add_log(env);
// //           compute_observation_with_perspective(env, &env->context);
// //           return;
// //         }
        
// //         // Reset the puzzle to its starting position for another try
// //         // Wrong move - reset position
// //         c_set_fen(env, env->context.puzzle_fen);
// //         env->context.puzzle_move_index = 0;
// //         env->context.puzzle_failed = false;
        
// //         // Clear terminals and continue
// //         env->terminals[0] = 0;
// //         env->terminals[1] = 0;

// //         compute_observation_with_perspective(env, &env->context);
// //         return;
// //       }
// //     }
// //   } // End of puzzle mode block

// //   // IMPORTANT: Skip all regular game logic in puzzle mode
// //   if (env->context.puzzle_mode) {
// //     // This should never be reached due to returns above, but just in case
// //     // printf("[PUZZLE WARNING] Reached end of puzzle block without proper return!\n");
// //     compute_observation_with_perspective(env, &env->context);
// //     return;
// //   }

// //   // Check if the move put the opponent in check and award check reward
// //   if (move_applied) {
// //     // After the move, it's now the opponent's turn - check if they're in check
// //     PieceColor opponent = env->context.board.to_move;
// //     if (is_in_check(&env->context.board, opponent)) {
// //       // The moving player put their opponent in check - award check reward
// //       float check_reward = (moving_player == C_WHITE) ? 
// //         env->context.c_reward_check_white : env->context.c_reward_check_black;
      
// //       env->rewards[agent_idx] += check_reward;
      
// //       // Track accumulated rewards for logging
// //       if (moving_player == C_WHITE) {
// //         env->context.accumulated_reward_check_white += check_reward;
// //       } else {
// //         env->context.accumulated_reward_check_black += check_reward;
// //       }
// //     }
// //   }

// //   // Assign rewards
// //   env->rewards[agent_idx] += env->context.c_reward_valid;
// //   env->context.accumulated_reward_valid += env->context.c_reward_valid;
  
// //   // Apply material advantage rewards every step
// //   int white_material = calculate_material_value(&env->context.board, C_WHITE);
// //   int black_material = calculate_material_value(&env->context.board, C_BLACK);
// //   int material_diff = white_material - black_material;  // Positive when WHITE ahead
  
// //   // Reward based on material advantage: positive for advantage, negative for disadvantage
// //   float white_material_reward = material_diff * env->context.c_reward_material_diff_white;
// //   float black_material_reward = -material_diff * env->context.c_reward_material_diff_black;
  
// //   // Apply material advantage rewards every step
// //   env->rewards[0] += white_material_reward;  // WHITE gets + for advantage, - for disadvantage
// //   env->rewards[1] += black_material_reward;  // BLACK gets + for advantage, - for disadvantage
  
// //   // Track accumulated material rewards for logging
// //   env->context.accumulated_reward_material_diff_white += white_material_reward;
// //   env->context.accumulated_reward_material_diff_black += black_material_reward;
  
// //   // Assign capture rewards if this was a capture
// //   if (is_capture) {
// //     float capture_reward = 0.0f;
    
// //     if (env->context.c_use_piece_value_capture_rewards) {
// //       // Use piece-value-based rewards
// //       PieceType captured_piece_type = PAWN; // Default for en passant
// //       if (!is_en_passant && destination_piece) {
// //         captured_piece_type = destination_piece->type;
// //       }
// //       int piece_value = get_piece_value(captured_piece_type);
// //       capture_reward = piece_value * env->context.c_piece_value_reward_multiplier;
// //     } else {
// //       // Use fixed capture rewards
// //       if (env->context.board.to_move == C_WHITE) {
// //         capture_reward = env->context.c_reward_white_captures_enemy_piece;
// //       } else {
// //         capture_reward = env->context.c_reward_black_captures_enemy_piece;
// //       }
// //     }
    
// //     // Apply the capture reward
// //     env->rewards[agent_idx] += capture_reward;
    
// //     // Track accumulated rewards for logging
// //     if (env->context.board.to_move == C_WHITE) {
// //       env->context.accumulated_reward_white_captures_enemy_piece += capture_reward;
// //     } else {
// //       env->context.accumulated_reward_black_captures_enemy_piece += capture_reward;
// //     }
// //     // Also track en passant captures
// //     if (is_en_passant) {
// //       if (env->context.board.to_move == C_WHITE) {
// //         env->context.c_en_passant_white += 1;
// //       } else {
// //         env->context.c_en_passant_black += 1;
// //       }
// //     }
// //   }

// //   if (env->context.board.to_move == C_WHITE) {
// //     env->context.c_white_moves += 1;
// //   } else {
// //     env->context.c_black_moves += 1;
// //   }
// //   env->context.c_valid_moves += 1;

// //   // In chess, each player gets their own rewards based on their color
// //   // Agent 0 = WHITE, Agent 1 = BLACK
// //   // Only the moving player gets action-based rewards (valid move, captures, checks)
// //   // Both players always get material difference rewards every step
  
// //   // Track episode returns by color (not by agent index)
// //   env->context.episode_return_white += env->rewards[0];
// //   env->context.episode_return_black += env->rewards[1];

// //   // Check for game over conditions using already-generated legal moves (OPTIMIZATION)
// //   // Skip normal game termination logic in puzzle mode
// //   if (env->context.puzzle_mode) {
// //     // In puzzle mode, termination is handled by puzzle logic only
// //     return;
// //   }
  
// //   bool game_over = false;
// //   // Check the appropriate move count based on whose turn it is
// //   int current_move_count = (env->context.board.to_move == C_WHITE) ? 
// //                            env->context.white_legal_moves_count : 
// //                            env->context.black_legal_moves_count;
// //   bool any_legal_move_exists = (current_move_count > 0);

// //   if (!any_legal_move_exists) {
// //     game_over = true;
// //     if (is_in_check(&env->context.board, env->context.board.to_move)) {
// //       // CHECKMATE
// //       if (env->context.board.to_move == C_WHITE) { // White is checkmated (black won)
// //         float win_reward = env->context.c_reward_win_black;
// //         float loss_reward = env->context.c_reward_loss_white;
// //         // Both agents get shared reward based on game outcome
// //         env->rewards[0] += win_reward;
// //         env->rewards[1] += loss_reward;
// //         env->context.c_white_checkmated += 1;
// //         env->context.c_black_win += 1;
// //         env->context.c_white_loss += 1;
// //         // Add accumulated reward tracking for logging
// //         env->context.accumulated_reward_win_black += win_reward;
// //         env->context.accumulated_reward_loss_white += env->context.c_reward_loss_white;
// //       } else { // Black is checkmated (white won)
// //         float win_reward = env->context.c_reward_win_white;
// //         float loss_reward = env->context.c_reward_loss_black;
// //         // Both agents get shared reward based on game outcome
// //         env->rewards[0] += win_reward;
// //         env->rewards[1] += loss_reward;
// //         env->context.c_black_checkmated += 1;
// //         env->context.c_white_win += 1;
// //         env->context.c_black_loss += 1;
// //         // Add accumulated reward tracking for logging
// //         env->context.accumulated_reward_win_white += win_reward;
// //         env->context.accumulated_reward_loss_black += env->context.c_reward_loss_black;
// //       }
// //     } else {
// //       // STALEMATE
// //       env->rewards[0] += env->context.c_reward_draw;
// //       env->rewards[1] += env->context.c_reward_draw;
// //       env->context.c_stalemate += 1;
// //       env->context.c_game_drawn += 1;
// //       // Add accumulated reward tracking for logging
// //       env->context.accumulated_reward_draw += env->context.c_reward_draw;
// //     }
// //   } else if (env->context.board.halfmove_clock >= 100) {
// //     game_over = true; // FIFTY-MOVE RULE
// //     env->rewards[0] += env->context.c_reward_draw;
// //     env->rewards[1] += env->context.c_reward_draw;
// //     env->context.c_fifty_move_rule += 1;
// //     env->context.c_game_drawn += 1;
// //     // Add accumulated reward tracking for logging
// //     env->context.accumulated_reward_draw += env->context.c_reward_draw;
// //   } else if (is_threefold_repetition(&env->context)) {
// //     game_over = true; // THREEFOLD REPETITION
// //     env->rewards[0] += env->context.c_reward_draw;
// //     env->rewards[1] += env->context.c_reward_draw;
// //     env->context.c_threefold_repetition += 1;
// //     env->context.c_game_drawn += 1;
// //     // Add accumulated reward tracking for logging
// //     env->context.accumulated_reward_draw += env->context.c_reward_draw;
// //   } else if (is_insufficient_material(&env->context)) {
// //     game_over = true; // INSUFFICIENT MATERIAL
// //     env->rewards[0] += env->context.c_reward_draw;
// //     env->rewards[1] += env->context.c_reward_draw;
// //     env->context.c_insufficient_material += 1;
// //     env->context.c_game_drawn += 1;
// //     // Add accumulated reward tracking for logging
// //     env->context.accumulated_reward_draw += env->context.c_reward_draw;
// //   } else if (env->max_depth > 0 && env->context.step_count >= env->max_depth) {
// //     game_over = true; // MAX DEPTH / TRUNCATION
// //     env->rewards[0] += env->context.c_reward_max_depth_termination;
// //     env->rewards[1] += env->context.c_reward_max_depth_termination;
// //     env->context.c_max_depth += 1;
// //     env->context.c_game_drawn += 1;
// //     // Add accumulated reward tracking for logging
// //     env->context.accumulated_reward_draw += env->context.c_reward_draw;
// //   }

// //   if (game_over) {
// //     // Mark both agents as terminal
// //     env->terminals[0] = 1;
// //     env->terminals[1] = 1;
// //     env->log.complete_game_move_count =
// //         (float)env->context.complete_game_action_count;
// //     add_log(env);
    
// //     // Notify UI about game end via function call (before auto-reset clears counters)
// //     // notify_game_end(env->context.c_white_win > 0, env->context.c_black_win > 0, env->context.c_game_drawn > 0);

// //     // Check if we should log this complete game BEFORE reset
    
// //     // Designate the first environment that completes a game as the logging environment
// //     if (first_active_env_id == -1) {
// //       first_active_env_id = env->env_id;
// //       logging_env_id = env->env_id;
// //     }
    
// //     // Only log from the designated logging environment to avoid spam from 512 environments
// //     if (env->env_id == logging_env_id) {
// //       // Increment game counter for the logging environment
// //       env->context.steps_since_last_log++;
      
      
// //       // Log every N games completed by the logging environment
// //       if (env->context.game_logging_frequency > 0 && env->context.steps_since_last_log >= env->context.game_logging_frequency) {
// //         write_complete_game_to_file(&env->context, env->env_id);
// //         env->log.game_step_logged = 1.0; // Indicate a game was logged
// //         env->context.steps_since_last_log = 0; // Reset counter
// //       }
// //     } else {
// //     }
    
// //     // Debug: Always print for any env that completes a game
// //     if (env->context.complete_game_action_count > 0) {
// //     }

// //     // Save values before reset
// //     int saved_steps = env->context.steps_since_last_log;
// //     int saved_freq = env->context.game_logging_frequency;

// //     // AUTO-RESET: Manually reset the environment to start a new game
// //     c_reset(env);
    
// //     // Restore saved values after reset
// //     env->context.steps_since_last_log = saved_steps;
// //     env->context.game_logging_frequency = saved_freq;
// //   } else {
// //     // Compute new observation if the game is not over
// //     compute_observation_with_perspective(env, &env->context);
// //   }

// //   // PROFILING: Print performance statistics every 1000 steps
// //   static int profiling_step_count = 0;
// //   profiling_step_count++;
// //   if (profiling_step_count % 1000 == 0) {
// //     double total_time = (double)profile_c_step_ticks / CLOCKS_PER_SEC;
// //     double move_gen_time = (double)profile_move_gen_uci_ticks / CLOCKS_PER_SEC;
// //     double obs_time = (double)profile_compute_obs_ticks / CLOCKS_PER_SEC;
// //     double legal_move_time = (double)profile_is_legal_move_ticks / CLOCKS_PER_SEC;
// //     double square_attack_time = (double)profile_is_square_attacked_ticks / CLOCKS_PER_SEC;
// //     double apply_move_time = (double)profile_apply_uci_move_ticks / CLOCKS_PER_SEC;
    
// //     // printf("[CHESS_PROFILE] Step %d - Total: %.3fs, MoveGen: %.3fs (%.1f%%), Obs: %.3fs (%.1f%%), LegalCheck: %.3fs (%.1f%%), SquareAttack: %.3fs (%.1f%%), ApplyMove: %.3fs (%.1f%%)\n",
// //     //        profiling_step_count, total_time,
// //     //        move_gen_time, move_gen_time/total_time*100,
// //     //        obs_time, obs_time/total_time*100,
// //     //        legal_move_time, legal_move_time/total_time*100,
// //     //        square_attack_time, square_attack_time/total_time*100,
// //     //        apply_move_time, apply_move_time/total_time*100);
// //   }

// //   PROFILE_STOP(profile_c_step_ticks);
// // }


// // potential replacement c_step() (assuming it works)
// void c_step(CChess *env) {
//   // Guard against calls on an already completed puzzle episode
//   if (env->context.puzzle_mode && env->context.puzzle_completed) {
//     env->terminals[0] = 1;
//     env->terminals[1] = 1;
//     return;
//   }

//   PROFILE_START(profile_c_step_ticks)

//   // Game logging counter is incremented when games end, not on every step
//   //   // In self-play mode: agent 0 = white, agent 1 = black
//   //   // Get action from the agent whose turn it is
//   //   PieceColor current_player = env->context.board.to_move;
//   //   int agent_idx = (current_player == C_WHITE) ? 0 : 1;

//   //   printf("[C_STEP_ENTRY] Turn: %s (agent %d), fullmove: %d, halfmove:
//   //   %d\n",
//   //          (current_player == C_WHITE) ? "WHITE" : "BLACK", agent_idx,
//   //          env->context.board.fullmove_number,
//   //          env->context.board.halfmove_clock);

//   //   // CRITICAL: In dual-agent mode, only process actions for the current
//   //   player
//   //   // SAFEGUARD: Handle training loop calling wrong agent
//   //   int correct_agent_idx = agent_idx;
//   //   if (env->context.dual_agent_self_play_mode) {
//   //     // Determine which agent should actually move based on board state
//   //     int expected_agent = (current_player == C_WHITE) ? 0 : 1;

//   //     if (agent_idx != expected_agent) {
//   //       printf("[TRAINING_LOOP_FIX] Board says %s's turn (agent %d), but
//   //       called with agent %d. Using correct agent.\n",
//   //              (current_player == C_WHITE) ? "WHITE" : "BLACK",
//   //              expected_agent, agent_idx);
//   //       correct_agent_idx = expected_agent;
//   //     }
//   //   }

//   //   // Use the action from the correct agent (the one whose turn it actually
//   //   is) int action_idx = env->actions[correct_agent_idx];
//   // In episode-per-color architecture, Python wrapper handles agent assignment
//   // and episode separation. C++ always uses agent_idx = 0 for the active player
//   // during their episode (WHITE episode or BLACK episode)
//   int agent_idx = 0;
//   // The action is now correctly read from env->actions[0] for BOTH White and
//   // Black.
//   int action_idx = env->actions[agent_idx];
//   // Generate moves for BOTH players if not cached
//   // This ensures we always have the right moves available
//   chess_generate_all_legal_moves(&env->context);

//   // Calculate the correct observation offset for the CURRENT player.
//   const int single_obs_size = 1537; // 1472 board + sparse mask (1 + 64)
//   int obs_offset = 0;               // <-- NEW, CORRECT LOGIC
//   int mask_start_idx = 1472;        // Start of action mask in observation

//   // Check if the chosen action corresponds to a legal move
//   bool action_is_legal = false;

//   if (action_idx >= 0 && action_idx < TOTAL_CHESS_ACTIONS) {
//     const char *chosen_uci = ACTION_ID_TO_UCI[action_idx];

//     // Get current player's moves for validation
//     char (*moves_buffer)[6];
//     int moves_count;
//     if (env->context.board.to_move == C_WHITE) {
//       moves_buffer = env->context.white_legal_moves_buffer;
//       moves_count = env->context.white_legal_moves_count;
//     } else {
//       moves_buffer = env->context.black_legal_moves_buffer;
//       moves_count = env->context.black_legal_moves_count;
//     }

//     // Check if this action corresponds to any legal move
//     for (int i = 0; i < moves_count; i++) {
//       const char *legal_uci = moves_buffer[i];
//       char perspective_uci[6];

//       // For BLACK, we need to flip the canonical move to match the action space
//       if (env->context.board.to_move == C_BLACK) {
//         flip_uci_for_black_perspective(legal_uci, perspective_uci);
//       } else {
//         strcpy(perspective_uci, legal_uci);
//       }

//       if (strcmp(chosen_uci, perspective_uci) == 0) {
//         action_is_legal = true;
//         break;
//       }
//     }
//   }
//   if (!action_is_legal) {
//     // This can happen if the policy is random or if there are no legal moves.
//     // In puzzle mode, this signifies a failed attempt.
//     if (env->context.puzzle_mode) {
//         env->rewards[agent_idx] += env->reward_puzzle_failed;
//         env->context.accumulated_reward_puzzle_failed += env->reward_puzzle_failed;
//         env->context.puzzle_wrong_moves_this_episode += 1.0f;
//         env->context.puzzle_attempts_this_episode += 1.0f;
//         env->terminals[agent_idx] = 1; // Terminate on illegal move
//         add_log(env);
//     }
//     compute_observation_with_perspective(env, &env->context);
//     PROFILE_STOP(profile_c_step_ticks);
//     return;
//   }

//   // Clear all agent rewards and terminals
//   //   for (int i = 0; i < 2; i++) {
//   //     env->rewards[i] = 0.0f;
//   //     env->terminals[i] = 0;
//   //   }
//   env->rewards[agent_idx] = 0.0f;
//   env->terminals[agent_idx] = 0;
//   // Debug: Print action index in puzzle mode
//   if (env->context.puzzle_mode && env->context.puzzle_tries_this_env < 5) {
//     // // printf("[PUZZLE DEBUG] Action index received: %d
//     // (TOTAL_CHESS_ACTIONS=%d)\n", action_idx, TOTAL_CHESS_ACTIONS);
//   }

//   // Validate action before executing
//   if (action_idx < 0 || action_idx >= TOTAL_CHESS_ACTIONS) {
//     printf("[ERROR] Invalid action ID: %d (max=%d)\n", action_idx,
//            TOTAL_CHESS_ACTIONS - 1);
//     return;
//   }

//   // --- START OF NEW VALIDATION LOGIC ---

//   // 1. Get the UCI string for the action chosen by the policy.
//   // ACTION_ID_TO_UCI always represents moves in white's perspective coordinate
//   // system.
//   const char *uci_move_white_perspective = ACTION_ID_TO_UCI[action_idx];
//   char uci_move_canonical[6];

//   if (env->context.board.to_move == C_BLACK) {
//     // The black agent chose this action based on its flipped perspective.
//     // The action maps to a move in white perspective coordinates, but since
//     // the black agent sees the board flipped, this move should be interpreted
//     // as being from black's perspective and flipped to canonical coordinates.
//     flip_uci_for_black_perspective(uci_move_white_perspective,
//                                    uci_move_canonical);
//   } else {
//     // For white, the white perspective move IS the canonical move.
//     strcpy(uci_move_canonical, uci_move_white_perspective);
//   }
//   // 2. Use the legal moves already generated at the start of c_step
//   // (OPTIMIZATION)
//   //    No need to regenerate - we already have the definitive list.
//   // 3. Check if the chosen move is in the freshly generated list.
//   bool is_action_legal = false;

//   // Get the correct move buffer
//   char (*moves_buffer)[6];
//   int moves_count;
//   if (env->context.board.to_move == C_WHITE) {
//     moves_buffer = env->context.white_legal_moves_buffer;
//     moves_count = env->context.white_legal_moves_count;
//   } else {
//     moves_buffer = env->context.black_legal_moves_buffer;
//     moves_count = env->context.black_legal_moves_count;
//   }

//   for (int i = 0; i < moves_count; i++) {
//     if (strcmp(moves_buffer[i], uci_move_canonical) == 0) {
//       is_action_legal = true;
//       break;
//     }
//   }
//   // 4. Validate against the ground truth.
//   if (!is_action_legal) {
//     const char *turn_color =
//         (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK";
//     // printf("[ERROR] Illegal move attempted (ground truth validation): action
//     // %d (%s) - %s's turn (agent %d)\n",
//     //        action_idx, uci_move_canonical, turn_color, agent_idx);

//     // Invalidate the move, penalize the agent, and end the step without
//     // applying the move.
//     if (env->context.board.to_move == C_WHITE) {
//       env->context.c_invalid_moves_white += 1;
//       env->rewards[agent_idx] += env->context.c_reward_invalid_white;
//     } else {
//       env->context.c_invalid_moves_black += 1;
//       env->rewards[agent_idx] += env->context.c_reward_invalid_black;
//     }
    
//     // In puzzle mode, illegal move terminates the episode
//     if (env->context.puzzle_mode) {
//       env->terminals[agent_idx] = 1;
//       add_log(env);
//     }
    
//     // Don't apply the move, just recompute observation and return.
//     compute_observation_with_perspective(env, &env->context);
//     PROFILE_STOP(profile_c_step_ticks);
//     return;
//   }

//   // --- END OF NEW VALIDATION LOGIC ---
//   // Check if this move is a capture before applying it
//   int from_x = (uci_move_canonical[0] - 'a');
//   int from_y = (uci_move_canonical[1] - '1');
//   int to_x = (uci_move_canonical[2] - 'a');
//   int to_y = (uci_move_canonical[3] - '1');

//   // Get the piece at the destination to check for capture
//   Piece *destination_piece = get_piece(&env->context.board, to_x, to_y);
//   bool is_capture = (destination_piece && destination_piece->type != EMPTY);

//   // Check for en passant capture
//   bool is_en_passant = false;
//   if (!is_capture) {
//     Piece *moving_piece = get_piece(&env->context.board, from_x, from_y);
//     if (moving_piece && moving_piece->type == PAWN &&
//         (to_y * 8 + to_x) == env->context.board.ep_square) {
//       is_capture = true;
//       is_en_passant = true;
//     }
//   }
//   //   // Apply the move
//   //   printf("[MOVE_APPLY] Applying move %s for %s\n", uci_move_canonical,
//   //          (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK");
//   // Store whose turn it was before the move (the player making the move)
//   PieceColor moving_player = env->context.board.to_move;

//   bool move_applied = apply_uci_move(&env->context, uci_move_canonical);
//   //   printf("[MOVE_APPLY] Move applied successfully: %s, new turn: %s\n",
//   //          move_applied ? "YES" : "NO",
//   //          (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK");
//   // PUZZLE MODE - Special handling
//   if (env->context.puzzle_mode) {
//     // In puzzle mode, only white should be moving
//     if (moving_player != C_WHITE) {
//       // printf("[PUZZLE ERROR] Black tried to move in puzzle mode! This should "
//              "not happen!\n");
//       env->terminals[agent_idx] = 1;
//       return;
//     }

//     // Print board BEFORE the move
//     // printf("\n[PUZZLE] Board before move (puzzle move %d/%d):\n",
//            env->context.puzzle_move_index + 1,
//            env->context.puzzle_solution_length);
//     print_board_state(&env->context.board);

//     // Apply the move
//     if (!move_applied) {
//       // Move failed to apply
//       env->terminals[agent_idx] = 1;
//       return;
//     }

//     // Print board AFTER the move
//     // printf("[PUZZLE] Move played: %s by WHITE\n", uci_move_canonical);
//     print_board_state(&env->context.board);
//   } else {
//     // Non-puzzle mode - normal move application
//     if (move_applied) {
//       // Regular game logic here
//     }
//   }
//   // PUZZLE MODE - Check if move is correct
//   if (move_applied && env->context.puzzle_mode) {
//     if (env->context.puzzle_move_index < env->context.puzzle_solution_length) {
//       // Track timing on first move of puzzle
//       if (env->context.puzzle_move_index == 0) {
//         env->context.puzzle_samples_to_solve++;

//         if (env->context.puzzle_start_time == 0) {
//           env->context.puzzle_start_time = clock();
//         }
//       }

//       // Check if the move matches the expected solution move
//       const char *expected_move =
//           env->context.puzzle_solution[env->context.puzzle_move_index];

//       if (strcmp(uci_move_canonical, expected_move) == 0) {
//         // Correct move! Award reward and advance to next move
//         // Correct move
//                env->context.puzzle_move_index + 1,
//                env->context.puzzle_solution_length);
//         env->rewards[agent_idx] += env->reward_correct_move;
//         env->context.accumulated_reward_puzzle_correct_move +=
//             env->reward_correct_move;
//         env->context.episode_return_white += env->reward_correct_move;
//         env->context.puzzle_move_index++;

//         // Track correct moves in context for later logging
//         env->context.puzzle_correct_moves_this_episode += 1.0f;

//         // Check if puzzle is complete
//         if (env->context.puzzle_move_index >=
//             env->context.puzzle_solution_length) {
//           // Puzzle solved! Award completion reward
//           // printf("[PUZZLE] PUZZLE SOLVED! Terminating episode.\n");
//           env->rewards[agent_idx] += env->reward_puzzle_solved;
//           env->context.accumulated_reward_puzzle_solved +=
//               env->reward_puzzle_solved;
//           env->context.episode_return_white += env->reward_puzzle_solved;
//           env->context.puzzle_completed = true;
//           env->global_puzzle_successes++;

//           // Calculate performance metrics
//           clock_t solve_time = clock() - env->context.puzzle_start_time;
//           env->log.puzzle_avg_time_to_solve =
//               (float)solve_time / CLOCKS_PER_SEC;
//           env->log.puzzle_avg_samples_to_solve =
//               (float)env->context.puzzle_samples_to_solve;

//           // Check if global threshold reached (simplified check)
//           if (env->global_puzzle_attempts >= 20) { // Minimum sample
//             float global_success_rate = (float)env->global_puzzle_successes /
//                                         env->global_puzzle_attempts;
//             if (global_success_rate >= env->global_puzzle_success_threshold) {
//               // Advance puzzle globally (reset counters)
//               env->global_puzzle_id++;
//               env->global_puzzle_attempts = 0;
//               env->global_puzzle_successes = 0;
//             }
//           }

//           // Track puzzle solved in context for later logging
//           env->context.puzzle_solved_this_env++;
//           env->context.puzzle_solved_this_episode += 1.0f;

//           // Update global logging info (these are okay to set directly as
//           // they're global state)
//           env->log.puzzle_current_id = (float)env->global_puzzle_id;
//           env->log.puzzle_global_attempts = (float)env->global_puzzle_attempts;
//           env->log.puzzle_global_successes =
//               (float)env->global_puzzle_successes;

//           // Reset for next puzzle
//           env->context.puzzle_tries_this_env = 0;
//           env->context.puzzle_samples_to_solve = 0;
//           env->context.puzzle_start_time = 0;

//           // Terminate episode - puzzle complete
//           env->terminals[agent_idx] = 1;
//           add_log(env);
//           compute_observation_with_perspective(env, &env->context);
//           return;
//         } else {
//           // More moves to make - recompute observation for next move
//           // Continue to next move
//           compute_observation_with_perspective(env, &env->context);
//           return;
//         }
//       } else {
//         // Wrong move! Give penalty and reward shaping
//         float total_penalty = env->reward_puzzle_failed;

//         // Parse the expected and actual moves
//         ChessMove expected_move;
//         if (!parse_uci_move(
//                 env->context.puzzle_solution[env->context.puzzle_move_index],
//                 &expected_move)) {
//           // Failed to parse expected move
//                  env->context.puzzle_solution[env->context.puzzle_move_index]);
//         } else {
//           // Parse the actual move made
//           ChessMove actual_move;
//           char uci_move[6];
//           action_to_uci(action_idx, uci_move);
//           if (parse_uci_move(uci_move, &actual_move)) {
//             // Reward shaping based on move similarity
//             float shaping_reward = 0.0f;

//             // 1. Reward for moving the correct piece
//             if (actual_move.from.x == expected_move.from.x &&
//                 actual_move.from.y == expected_move.from.y) {
//               shaping_reward += env->reward_puzzle_correct_piece;

//               // 2. Additional reward based on how close we moved to the target
//               // Calculate Manhattan distance from actual destination to
//               // expected destination
//               int expected_row = expected_move.to.y;
//               int expected_col = expected_move.to.x;
//               int actual_row = actual_move.to.y;
//               int actual_col = actual_move.to.x;

//               int distance = abs(expected_row - actual_row) +
//                              abs(expected_col - actual_col);
//               // Max distance on board is 14 (7+7), so normalize
//               float distance_reward = env->reward_puzzle_closer_to_target *
//                                       (1.0f - (float)distance / 14.0f);
//               shaping_reward += distance_reward;

//               // 3. If promotion expected and we promoted to same piece, bonus
//               if (expected_move.promotion != EMPTY &&
//                   actual_move.promotion == expected_move.promotion) {
//                 shaping_reward += env->reward_puzzle_correct_promotion;
//               }
//             }

//             total_penalty += shaping_reward;
//           }
//         }

//         env->rewards[agent_idx] += total_penalty;
//         env->context.accumulated_reward_puzzle_failed += total_penalty;
//         // In puzzle mode, only white plays, so only update white's episode
//         // return
//         env->context.episode_return_white += total_penalty;

//         // Track wrong moves in context for later logging
//         env->context.puzzle_wrong_moves_this_episode += 1.0f;
//         env->context.puzzle_tries_this_env++;
//         env->global_puzzle_attempts++;
//         env->context.puzzle_attempts_this_episode += 1.0f;

//         // FIX: A wrong move terminates the episode. The Python wrapper is responsible
//         // for re-presenting the same puzzle on the next reset.
//         env->context.puzzle_failed = true;
//         env->terminals[agent_idx] = 1; // Terminate the episode
//         add_log(env);                  // Log the failure
//         compute_observation_with_perspective(env, &env->context);
//         return;
//       }
//     }
//   } // End of puzzle mode block
//   // IMPORTANT: Skip all regular game logic in puzzle mode
//   if (env->context.puzzle_mode) {
//     // This should never be reached due to returns above, but just in case
//     // Warning: reached end of puzzle block
//            "return!\n");
//     compute_observation_with_perspective(env, &env->context);
//     return;
//   }
//   // Check if the move put the opponent in check and award check reward
//   if (move_applied) {
//     // After the move, it's now the opponent's turn - check if they're in check
//     PieceColor opponent = env->context.board.to_move;
//     if (is_in_check(&env->context.board, opponent)) {
//       // The moving player put their opponent in check - award check reward
//       float check_reward = (moving_player == C_WHITE)
//                                ? env->context.c_reward_check_white
//                                : env->context.c_reward_check_black;

//       env->rewards[agent_idx] += check_reward;

//       // Track accumulated rewards for logging
//       if (moving_player == C_WHITE) {
//         env->context.accumulated_reward_check_white += check_reward;
//       } else {
//         env->context.accumulated_reward_check_black += check_reward;
//       }
//     }
//   }
//   // Assign rewards
//   env->rewards[agent_idx] += env->context.c_reward_valid;
//   env->context.accumulated_reward_valid += env->context.c_reward_valid;

//   // Apply material advantage rewards every step
//   int white_material = calculate_material_value(&env->context.board, C_WHITE);
//   int black_material = calculate_material_value(&env->context.board, C_BLACK);
//   int material_diff =
//       white_material - black_material; // Positive when WHITE ahead

//   // Reward based on material advantage: positive for advantage, negative for
//   // disadvantage
//   float white_material_reward =
//       material_diff * env->context.c_reward_material_diff_white;
//   float black_material_reward =
//       -material_diff * env->context.c_reward_material_diff_black;

//   // Apply material advantage rewards every step
//   env->rewards[0] +=
//       white_material_reward; // WHITE gets + for advantage, - for disadvantage
//   env->rewards[1] +=
//       black_material_reward; // BLACK gets + for advantage, - for disadvantage

//   // Track accumulated material rewards for logging
//   env->context.accumulated_reward_material_diff_white += white_material_reward;
//   env->context.accumulated_reward_material_diff_black += black_material_reward;

//   // Assign capture rewards if this was a capture
//   if (is_capture) {
//     float capture_reward = 0.0f;

//     if (env->context.c_use_piece_value_capture_rewards) {
//       // Use piece-value-based rewards
//       PieceType captured_piece_type = PAWN; // Default for en passant
//       if (!is_en_passant && destination_piece) {
//         captured_piece_type = destination_piece->type;
//       }
//       int piece_value = get_piece_value(captured_piece_type);
//       capture_reward =
//           piece_value * env->context.c_piece_value_reward_multiplier;
//     } else {
//       // Use fixed capture rewards
//       if (env->context.board.to_move == C_WHITE) {
//         capture_reward = env->context.c_reward_white_captures_enemy_piece;
//       } else {
//         capture_reward = env->context.c_reward_black_captures_enemy_piece;
//       }
//     }

//     // Apply the capture reward
//     env->rewards[agent_idx] += capture_reward;

//     // Track accumulated rewards for logging
//     if (env->context.board.to_move == C_WHITE) {
//       env->context.accumulated_reward_white_captures_enemy_piece +=
//           capture_reward;
//     } else {
//       env->context.accumulated_reward_black_captures_enemy_piece +=
//           capture_reward;
//     }
//     // Also track en passant captures
//     if (is_en_passant) {
//       if (env->context.board.to_move == C_WHITE) {
//         env->context.c_en_passant_white += 1;
//       } else {
//         env->context.c_en_passant_black += 1;
//       }
//     }
//   }
//   if (env->context.board.to_move == C_WHITE) {
//     env->context.c_white_moves += 1;
//   } else {
//     env->context.c_black_moves += 1;
//   }
//   env->context.c_valid_moves += 1;
//   // In chess, each player gets their own rewards based on their color
//   // Agent 0 = WHITE, Agent 1 = BLACK
//   // Only the moving player gets action-based rewards (valid move, captures,
//   // checks) Both players always get material difference rewards every step

//   // Track episode returns by color (not by agent index)
//   env->context.episode_return_white += env->rewards[0];
//   env->context.episode_return_black += env->rewards[1];
//   // Check for game over conditions using already-generated legal moves
//   // (OPTIMIZATION) Skip normal game termination logic in puzzle mode
//   if (env->context.puzzle_mode) {
//     // In puzzle mode, termination is handled by puzzle logic only
//     return;
//   }

//   bool game_over = false;
//   // Check the appropriate move count based on whose turn it is
//   int current_move_count = (env->context.board.to_move == C_WHITE)
//                                ? env->context.white_legal_moves_count
//                                : env->context.black_legal_moves_count;
//   bool any_legal_move_exists = (current_move_count > 0);
//   if (!any_legal_move_exists) {
//     game_over = true;
//     if (is_in_check(&env->context.board, env->context.board.to_move)) {
//       // CHECKMATE
//       if (env->context.board.to_move ==
//           C_WHITE) { // White is checkmated (black won)
//         float win_reward = env->context.c_reward_win_black;
//         float loss_reward = env->context.c_reward_loss_white;
//         // Both agents get shared reward based on game outcome
//         env->rewards[0] += win_reward;
//         env->rewards[1] += loss_reward;
//         env->context.c_white_checkmated += 1;
//         env->context.c_black_win += 1;
//         env->context.c_white_loss += 1;
//         // Add accumulated reward tracking for logging
//         env->context.accumulated_reward_win_black += win_reward;
//         env->context.accumulated_reward_loss_white +=
//             env->context.c_reward_loss_white;
//       } else { // Black is checkmated (white won)
//         float win_reward = env->context.c_reward_win_white;
//         float loss_reward = env->context.c_reward_loss_black;
//         // Both agents get shared reward based on game outcome
//         env->rewards[0] += win_reward;
//         env->rewards[1] += loss_reward;
//         env->context.c_black_checkmated += 1;
//         env->context.c_white_win += 1;
//         env->context.c_black_loss += 1;
//         // Add accumulated reward tracking for logging
//         env->context.accumulated_reward_win_white += win_reward;
//         env->context.accumulated_reward_loss_black +=
//             env->context.c_reward_loss_black;
//       }
//     } else {
//       // STALEMATE
//       env->rewards[0] += env->context.c_reward_draw;
//       env->rewards[1] += env->context.c_reward_draw;
//       env->context.c_stalemate += 1;
//       env->context.c_game_drawn += 1;
//       // Add accumulated reward tracking for logging
//       env->context.accumulated_reward_draw += env->context.c_reward_draw;
//     }
//   } else if (env->context.board.halfmove_clock >= 100) {
//     game_over = true; // FIFTY-MOVE RULE
//     env->rewards[0] += env->context.c_reward_draw;
//     env->rewards[1] += env->context.c_reward_draw;
//     env->context.c_fifty_move_rule += 1;
//     env->context.c_game_drawn += 1;
//     // Add accumulated reward tracking for logging
//     env->context.accumulated_reward_draw += env->context.c_reward_draw;
//   } else if (is_threefold_repetition(&env->context)) {
//     game_over = true; // THREEFOLD REPETITION
//     env->rewards[0] += env->context.c_reward_draw;
//     env->rewards[1] += env->context.c_reward_draw;
//     env->context.c_threefold_repetition += 1;
//     env->context.c_game_drawn += 1;
//     // Add accumulated reward tracking for logging
//     env->context.accumulated_reward_draw += env->context.c_reward_draw;
//   } else if (is_insufficient_material(&env->context)) {
//     game_over = true; // INSUFFICIENT MATERIAL
//     env->rewards[0] += env->context.c_reward_draw;
//     env->rewards[1] += env->context.c_reward_draw;
//     env->context.c_insufficient_material += 1;
//     env->context.c_game_drawn += 1;
//     // Add accumulated reward tracking for logging
//     env->context.accumulated_reward_draw += env->context.c_reward_draw;
//   } else if (env->max_depth > 0 && env->context.step_count >= env->max_depth) {
//     game_over = true; // MAX DEPTH / TRUNCATION
//     env->rewards[0] += env->context.c_reward_max_depth_termination;
//     env->rewards[1] += env->context.c_reward_max_depth_termination;
//     env->context.c_max_depth += 1;
//     env->context.c_game_drawn += 1;
//     // Add accumulated reward tracking for logging
//     env->context.accumulated_reward_draw += env->context.c_reward_draw;
//   }
//   if (game_over) {
//     // Mark both agents as terminal
//     env->terminals[0] = 1;
//     env->terminals[1] = 1;
//     env->log.complete_game_move_count =
//         (float)env->context.complete_game_action_count;
//     add_log(env);

//     // Notify UI about game end via function call (before auto-reset clears
//     // counters) notify_game_end(env->context.c_white_win > 0,
//     // env->context.c_black_win > 0, env->context.c_game_drawn > 0); Check if we
//     // should log this complete game BEFORE reset

//     // Designate the first environment that completes a game as the logging
//     // environment
//     if (first_active_env_id == -1) {
//       first_active_env_id = env->env_id;
//       logging_env_id = env->env_id;
//     }

//     // Only log from the designated logging environment to avoid spam from 512
//     // environments
//     if (env->env_id == logging_env_id) {
//       // Increment game counter for the logging environment
//       env->context.steps_since_last_log++;

//       // Log every N games completed by the logging environment
//       if (env->context.game_logging_frequency > 0 &&
//           env->context.steps_since_last_log >=
//               env->context.game_logging_frequency) {
//         write_complete_game_to_file(&env->context, env->env_id);
//         env->log.game_step_logged = 1.0;       // Indicate a game was logged
//         env->context.steps_since_last_log = 0; // Reset counter
//       }
//     } else {
//     }

//     // Debug: Always print for any env that completes a game
//     if (env->context.complete_game_action_count > 0) {
//     }
//     // Save values before reset
//     int saved_steps = env->context.steps_since_last_log;
//     int saved_freq = env->context.game_logging_frequency;
//     // AUTO-RESET: Manually reset the environment to start a new game
//     c_reset(env);

//     // Restore saved values after reset
//     env->context.steps_since_last_log = saved_steps;
//     env->context.game_logging_frequency = saved_freq;
//   } else {
//     // Compute new observation if the game is not over
//     compute_observation_with_perspective(env, &env->context);
//   }
//   // PROFILING: Print performance statistics every 1000 steps
//   static int profiling_step_count = 0;
//   profiling_step_count++;
//   if (profiling_step_count % 1000 == 0) {
//     double total_time = (double)profile_c_step_ticks / CLOCKS_PER_SEC;
//     double move_gen_time = (double)profile_move_gen_uci_ticks / CLOCKS_PER_SEC;
//     double obs_time = (double)profile_compute_obs_ticks / CLOCKS_PER_SEC;
//     double legal_move_time =
//         (double)profile_is_legal_move_ticks / CLOCKS_PER_SEC;
//     double square_attack_time =
//         (double)profile_is_square_attacked_ticks / CLOCKS_PER_SEC;
//     double apply_move_time =
//         (double)profile_apply_uci_move_ticks / CLOCKS_PER_SEC;

//     // printf("[CHESS_PROFILE] Step %d - Total: %.3fs, MoveGen: %.3fs (%.1f%%),
//     // Obs: %.3fs (%.1f%%), LegalCheck: %.3fs (%.1f%%), SquareAttack: %.3fs
//     // (%.1f%%), ApplyMove: %.3fs (%.1f%%)\n",
//     //        profiling_step_count, total_time,
//     //        move_gen_time, move_gen_time/total_time*100,
//     //        obs_time, obs_time/total_time*100,
//     //        legal_move_time, legal_move_time/total_time*100,
//     //        square_attack_time, square_attack_time/total_time*100,
//     //        apply_move_time, apply_move_time/total_time*100);
//   }
//   PROFILE_STOP(profile_c_step_ticks);
// }

// // === PUFFERLIB LOGGING FUNCTION ===
// void add_log(CChess *env) {

//   // Aggregate counters into log structure using = for PufferLib (CRITICAL!)
//   env->log.episode_length += (float)env->context.step_count;
//   env->log.episode_return +=
//       env->context.episode_return_white + env->context.episode_return_black;
//   env->log.episode_return_white += env->context.episode_return_white;
//   env->log.episode_return_black += env->context.episode_return_black;

//   // Reward aggregates (from accumulated counters during this game)
//   env->log.reward_valid += env->context.accumulated_reward_valid;
//   env->log.reward_white_captures_enemy_piece +=
//       env->context.accumulated_reward_white_captures_enemy_piece;
//   env->log.reward_black_captures_enemy_piece +=
//       env->context.accumulated_reward_black_captures_enemy_piece;
//   env->log.reward_draw += env->context.accumulated_reward_draw;
//   env->log.reward_win_white += env->context.accumulated_reward_win_white;
//   env->log.reward_win_black += env->context.accumulated_reward_win_black;
//   env->log.reward_loss_white += env->context.accumulated_reward_loss_white;
//   env->log.reward_loss_black += env->context.accumulated_reward_loss_black;
//   env->log.reward_draw_white += env->context.accumulated_reward_draw_white;
//   env->log.reward_draw_black += env->context.accumulated_reward_draw_black;
//   env->log.reward_check_white += env->context.accumulated_reward_check_white;
//   env->log.reward_check_black += env->context.accumulated_reward_check_black;
//   env->log.reward_material_diff_white +=
//       env->context.accumulated_reward_material_diff_white;
//   env->log.reward_material_diff_black +=
//       env->context.accumulated_reward_material_diff_black;
//   env->log.stockfish_eval += env->context.accumulated_stockfish_eval;

//   // Game outcome counters (use incremental values from current game)
//   env->log.white_win += (float)env->context.c_white_win;
//   env->log.white_loss += (float)env->context.c_white_loss;
//   env->log.black_win += (float)env->context.c_black_win;
//   env->log.black_loss += (float)env->context.c_black_loss;
//   env->log.game_drawn += (float)env->context.c_game_drawn;
//   env->log.stalemate += (float)env->context.c_stalemate;
//   env->log.insufficient_material += (float)env->context.c_insufficient_material;
//   env->log.threefold_repetition += (float)env->context.c_threefold_repetition;
//   env->log.fifty_move_rule += (float)env->context.c_fifty_move_rule;
//   env->log.max_depth += (float)env->context.c_max_depth;
//   env->log.white_checkmated += (float)env->context.c_white_checkmated;
//   env->log.black_checkmated += (float)env->context.c_black_checkmated;

//   // Move statistics
//   env->log.white_moves += (float)env->context.c_white_moves;
//   env->log.black_moves += (float)env->context.c_black_moves;
//   env->log.valid_moves += (float)env->context.c_valid_moves;
//   env->log.invalid_moves_white += (float)env->context.c_invalid_moves_white;
//   env->log.invalid_moves_black += (float)env->context.c_invalid_moves_black;

//   // Castling and special moves
//   env->log.en_passant_white += (float)env->context.c_en_passant_white;
//   env->log.en_passant_black += (float)env->context.c_en_passant_black;
//   env->log.white_castle_kingside += (float)env->context.c_white_castle_kingside;
//   env->log.white_castle_queenside +=
//       (float)env->context.c_white_castle_queenside;
//   env->log.black_castle_kingside += (float)env->context.c_black_castle_kingside;
//   env->log.black_castle_queenside +=
//       (float)env->context.c_black_castle_queenside;

//   // Promotion statistics
//   env->log.white_promotion_count += (float)env->context.c_white_promotion_count;
//   env->log.white_promotion_knight +=
//       (float)env->context.c_white_promotion_knight;
//   env->log.white_promotion_bishop +=
//       (float)env->context.c_white_promotion_bishop;
//   env->log.white_promotion_rook += (float)env->context.c_white_promotion_rook;
//   env->log.white_promotion_queen += (float)env->context.c_white_promotion_queen;
//   env->log.black_promotion_count += (float)env->context.c_black_promotion_count;
//   env->log.black_promotion_knight +=
//       (float)env->context.c_black_promotion_knight;
//   env->log.black_promotion_bishop +=
//       (float)env->context.c_black_promotion_bishop;
//   env->log.black_promotion_rook += (float)env->context.c_black_promotion_rook;
//   env->log.black_promotion_queen += (float)env->context.c_black_promotion_queen;

//   // Add puzzle rewards to log
//   env->log.reward_puzzle_solved += env->context.accumulated_reward_puzzle_solved;
//   env->log.reward_puzzle_failed += env->context.accumulated_reward_puzzle_failed;
//   env->log.reward_puzzle_correct_move += env->context.accumulated_reward_puzzle_correct_move;
  
//   // Add puzzle stats to log
//   env->log.puzzle_solved += env->context.puzzle_solved_this_episode;
//   env->log.puzzle_attempts += env->context.puzzle_attempts_this_episode;
//   env->log.puzzle_correct_moves += env->context.puzzle_correct_moves_this_episode;
//   env->log.puzzle_wrong_moves += env->context.puzzle_wrong_moves_this_episode;
  
//   // Don't calculate success rate in add_log - it gets summed incorrectly
//   // Success rate should be calculated post-aggregation

//   // Calculate performance metrics after aggregation
//   float total_games =
//       env->log.white_win + env->log.white_loss + env->log.game_drawn;
//   if (total_games > 0) {
//     env->log.perf = env->log.white_win / total_games; // White win rate
//   }
//   env->log.score =
//       env->log.white_win - env->log.white_loss; // Win-loss difference

//   // Increment n (must be last for PufferLib aggregation)
//   env->log.n += 1.0f;
  
// }

// void c_render(CChess *env) {

//   printf("\n  +---+---+---+---+---+---+---+---+\n");
//   for (int y = 7; y >= 0; y--) {
//     printf("%d |", y + 1);
//     for (int x = 0; x < 8; x++) {
//       const Piece *p = get_piece_const(&env->context.board, x, y);
//       char piece_char = ' ';

//       if (p && p->type != EMPTY) {
//         const char pieces[] = " KQRBNP";
//         piece_char = pieces[p->type];
//         if (p->color == C_BLACK) {
//           piece_char = piece_char + ('a' - 'A'); // Make lowercase
//         }
//       }

//       printf(" %c |", piece_char);
//     }
//     printf("\n  +---+---+---+---+---+---+---+---+\n");
//   }
//   printf("    a   b   c   d   e   f   g   h\n");
//   printf("\nTo move: %s\n",
//          (env->context.board.to_move == C_WHITE) ? "White" : "Black");
//   printf("Step: %d\n", env->context.step_count);
// }

// void c_close(CChess *env) {
//   // Core cleanup for chess environment
//   // Currently all major data structures use static allocation within CChess
//   // struct Future: Clean up any Stockfish process handles, pipes, or other
//   // system resources when Stockfish integration is fully implemented

//   // Clear sensitive data and reset state
//   memset(&env->context, 0, sizeof(ChessContext));
//   memset(&env->log, 0, sizeof(Log));
// }

// // === DUAL AGENT SELF-PLAY MODE SETTERS ===

// void set_dual_agent_self_play_mode(CChess *env, bool enabled) {
//   env->context.dual_agent_self_play_mode = enabled;
//   env->context.self_play_mode = enabled;  // Also set self_play_mode
// }

// void set_self_play_mode(CChess *env, bool enabled) {
//   env->context.self_play_mode = enabled;
// }

// void enable_stockfish_black(CChess *env, const char *stockfish_cmd, int elo,
//                             int search_ms) {
//   // TODO: Implement full Stockfish integration in pure C
//   // Current C++ implementation in stockfish_wrapper.h needs to be ported to C
//   // This includes:
//   // 1. Process spawning and communication via pipes
//   // 2. UCI protocol implementation
//   // 3. FEN position synchronization
//   // 4. Move request/response handling
//   // 5. ELO and time control settings
//   //
//   // For now, this is a compatibility stub
//   if (env && stockfish_cmd) {
//     env->stockfish_enabled = true;
//     // Store parameters for future implementation
//     strncpy(env->context.stockfish_cmd, stockfish_cmd,
//             sizeof(env->context.stockfish_cmd) - 1);
//     env->context.stockfish_elo = elo;
//     env->context.stockfish_search_ms = search_ms;
//   }
// }

// // FEN support (basic)
// void c_set_fen(CChess *env, const char *fen) {
//   ChessBoard *board = &env->context.board;
  
//   // Clear the board first
//   memset(board, 0, sizeof(ChessBoard));
  
//   // Parse FEN string
//   int x = 0, y = 7;  // Start from rank 8 (y=7)
//   const char *p = fen;
  
//   // Parse board position
//   while (*p && *p != ' ') {
//     if (*p == '/') {
//       x = 0;
//       y--;
//     } else if (*p >= '1' && *p <= '8') {
//       x += (*p - '0');  // Skip empty squares
//     } else {
//       // Parse piece
//       PieceColor color = (*p >= 'A' && *p <= 'Z') ? C_WHITE : C_BLACK;
//       PieceType type = EMPTY;
      
//       char piece_char = (*p >= 'A' && *p <= 'Z') ? *p : (*p - 'a' + 'A');
//       switch (piece_char) {
//         case 'P': type = PAWN; break;
//         case 'N': type = KNIGHT; break;
//         case 'B': type = BISHOP; break;
//         case 'R': type = ROOK; break;
//         case 'Q': type = QUEEN; break;
//         case 'K': type = KING; break;
//       }
      
//       if (type != EMPTY && x < 8 && y >= 0) {
//         board->board[y * 8 + x] = (Piece){color, type};
//         x++;
//       }
//     }
//     p++;
//   }
  
//   // Parse active color
//   if (*p == ' ') p++;
//   board->to_move = (*p == 'w') ? C_WHITE : C_BLACK;
  
//   // Parse castling rights
//   if (*p == ' ') p++;
//   if (*p == ' ') p++;
//   board->castle_rights = 0;
//   while (*p && *p != ' ') {
//     switch (*p) {
//       case 'K': board->castle_rights |= 0x1; break;  // White kingside
//       case 'Q': board->castle_rights |= 0x2; break;  // White queenside
//       case 'k': board->castle_rights |= 0x4; break;  // Black kingside
//       case 'q': board->castle_rights |= 0x8; break;  // Black queenside
//     }
//     p++;
//   }
  
//   // Parse en passant square
//   if (*p == ' ') p++;
//   board->ep_square = -1;
//   if (*p != '-') {
//     int ep_x = p[0] - 'a';
//     int ep_y = p[1] - '1';
//     if (ep_x >= 0 && ep_x < 8 && ep_y >= 0 && ep_y < 8) {
//       board->ep_square = ep_y * 8 + ep_x;
//     }
//   }
  
//   // Skip halfmove and fullmove (we can add these later if needed)
//   board->halfmove_clock = 0;
//   board->fullmove_number = 1;
  
//   // Compute zobrist hash for the new position
//   board->zobrist_hash = compute_zobrist_hash(board);
  
//   // Reset context state
//   env->context.white_legal_moves_count = 0;
//   env->context.black_legal_moves_count = 0;
//   env->context.white_moves_cached = false;
//   env->context.black_moves_cached = false;
//   env->context.position_fully_cached = false;
//   // Don't reset step_count in puzzle mode - it should accumulate across tries
//   if (!env->context.puzzle_mode) {
//     env->context.step_count = 0;
//   }
  
//   // Mark that FEN was set
//   board->fen_was_set = true;
  
//   // Recompute observation
//   compute_observation_with_perspective(env, &env->context);
// }

// // === PROFILING REPORT FUNCTION ===
// void c_print_profile_data() {
//   profile_total_ticks = profile_c_step_ticks;
//   if (profile_total_ticks == 0) {
//     printf("No profiling data collected yet.\n");
//     return;
//   }

//   printf("\n--- Chess Engine Profile ---\n");
//   printf("Function                               | Time (ms) | %% of Total\n");
//   printf("---------------------------------------|-----------|------------\n");

//   double total_ms = (double)profile_total_ticks * 1000.0 / CLOCKS_PER_SEC;

//   printf("c_step (Total)                         | %9.2f | %8.2f%%\n", total_ms,
//          100.0);

//   double move_gen_ms =
//       (double)profile_move_gen_uci_ticks * 1000.0 / CLOCKS_PER_SEC;
//   printf("  -> chess_generate_legal_moves_uci   | %9.2f | %8.2f%%\n",
//          move_gen_ms, (move_gen_ms / total_ms) * 100);

//   double is_legal_ms =
//       (double)profile_is_legal_move_ticks * 1000.0 / CLOCKS_PER_SEC;
//   printf("    -> chess_is_legal_move            | %9.2f | %8.2f%%\n",
//          is_legal_ms, (is_legal_ms / total_ms) * 100);

//   double is_attacked_ms =
//       (double)profile_is_square_attacked_ticks * 1000.0 / CLOCKS_PER_SEC;
//   printf("      -> is_square_attacked           | %9.2f | %8.2f%%\n",
//          is_attacked_ms, (is_attacked_ms / total_ms) * 100);

//   double make_move_ms =
//       (double)profile_make_move_fast_ticks * 1000.0 / CLOCKS_PER_SEC;
//   printf("      -> make_move_fast               | %9.2f | %8.2f%%\n",
//          make_move_ms, (make_move_ms / total_ms) * 100);

//   double unmake_move_ms =
//       (double)profile_unmake_move_fast_ticks * 1000.0 / CLOCKS_PER_SEC;
//   printf("      -> unmake_move_fast             | %9.2f | %8.2f%%\n",
//          unmake_move_ms, (unmake_move_ms / total_ms) * 100);

//   double apply_uci_ms =
//       (double)profile_apply_uci_move_ticks * 1000.0 / CLOCKS_PER_SEC;
//   printf("  -> apply_uci_move                   | %9.2f | %8.2f%%\n",
//          apply_uci_ms, (apply_uci_ms / total_ms) * 100);

//   double compute_obs_ms =
//       (double)profile_compute_obs_ticks * 1000.0 / CLOCKS_PER_SEC;
//   printf("  -> compute_observation_with_perspective | %9.2f | %8.2f%%\n",
//          compute_obs_ms, (compute_obs_ms / total_ms) * 100);

//   printf("----------------------------------------------------------\n");
// }

// #ifdef __cplusplus
// }
//  // extern "C"
// #endif

// #endif // CHESS_H

// chess.h - Complete Pure C Chess Environment for PufferLib
// Optimized for 150k+ SPS performance with single network self-play
#ifndef CHESS_H
#define CHESS_H
#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
// Enable debug logging for development (set to 0 to disable)
#ifndef DEBUG_LOG
#define DEBUG_LOG 0
#endif
#if DEBUG_LOG
// #define DBG(expr) printf("%s", expr)
#else
#define DBG(expr)                                                              \
  do {                                                                         \
  } while (0)
#endif
// === PROFILING GLOBALS ===
clock_t profile_total_ticks = 0;
clock_t profile_c_step_ticks = 0;
clock_t profile_move_gen_uci_ticks = 0;
clock_t profile_is_legal_move_ticks = 0;
clock_t profile_is_square_attacked_ticks = 0;
clock_t profile_make_move_fast_ticks = 0;
clock_t profile_unmake_move_fast_ticks = 0;
clock_t profile_apply_uci_move_ticks = 0;
clock_t profile_compute_obs_ticks = 0;
#define PROFILE_START(counter) clock_t start_##counter = clock();
#define PROFILE_STOP(counter) counter += clock() - start_##counter;
// Use extern "C" for functions that need C linkage
#ifdef __cplusplus
extern "C" {
#endif

// Include UCI action mapping
#include "chess_action_mapping.h"
// === CORE CHESS TYPES ===
typedef enum {
  EMPTY = 0,
  KING = 1,
  QUEEN = 2,
  ROOK = 3,
  BISHOP = 4,
  KNIGHT = 5,
  PAWN = 6
} PieceType;
typedef enum { C_WHITE = 0, C_BLACK = 1, C_NO_COLOR = 2 } PieceColor;
typedef struct {
  PieceColor color;
  PieceType type;
} Piece;
typedef struct {
  int8_t x, y;
} Square;
// === PUFFERLIB LOG STRUCTURE ===
typedef struct Log {
  float perf;          // Required by PufferLib
  float score;         // Use for solve percentage
  float episode_length;
  float episode_return;
  float n;             // Required by PufferLib - always last
} Log;

// === CHESS BOARD STATE ===
typedef struct {
  Piece board[64];
  PieceColor to_move;
  uint8_t castle_rights; // bits: 0=WK, 1=WQ, 2=BK, 3=BQ
  int8_t ep_square;      // en passant target square (-1 if none)
  uint8_t halfmove_clock;
  uint16_t fullmove_number;
  uint64_t zobrist_hash; // Incrementally updated Zobrist hash
  bool fen_was_set;      // Track if position was set via FEN
} ChessBoard;
// Position history for threefold repetition (simple hash table)
#define POSITION_HISTORY_SIZE 512
typedef struct {
  uint64_t hashes[POSITION_HISTORY_SIZE];
  int counts[POSITION_HISTORY_SIZE];
  int size;
} PositionHistory;
// === CHESS CONTEXT ===
typedef struct ChessContext {
  ChessBoard board;
  PositionHistory position_history;
  // Episode tracking
  int step_count;
  float episode_return_white;
  float episode_return_black;
  // CRITICAL: Performance optimization - DUAL CACHE for both players
  // White's legal moves cache
  char white_legal_moves_buffer[256][6]; // White's legal moves in UCI format
  int white_legal_moves_count;
  int white_legal_action_ids[256]; // Pre-computed action IDs
  bool white_moves_cached;

  // Black's legal moves cache
  char black_legal_moves_buffer[256][6]; // Black's legal moves in UCI format
  int black_legal_moves_count;
  int black_legal_action_ids[256]; // Pre-computed action IDs
  bool black_moves_cached;

  uint64_t cached_board_hash; // Position hash for both caches
  bool position_fully_cached; // True when both players cached
  // Game modes
  bool dual_agent_self_play_mode;
  bool self_play_mode;
  bool puzzle_mode;

  // Puzzle mode state
  char puzzle_fen[128];         // Current puzzle's starting FEN
  char puzzle_solution[10][6];  // Array of solution moves in UCI format
  int puzzle_solution_length;   // Number of moves in solution
  int puzzle_move_index;        // Current move in puzzle solution (0-based)
  bool puzzle_completed;        // True when puzzle solved correctly
  bool puzzle_failed;           // True when wrong move made
  int puzzle_attempts_this_env; // Track attempts for this specific environment
  int puzzle_solved_this_env;   // Track solves for this specific environment
  
  // Multiple puzzle support
  #define MAX_PUZZLE_SET_SIZE 20
  #define MAX_PUZZLE_MOVES 10  // Maximum moves per puzzle solution
  char puzzle_set_fens[MAX_PUZZLE_SET_SIZE][128];      // FENs for puzzle set
  char puzzle_set_solutions[MAX_PUZZLE_SET_SIZE][MAX_PUZZLE_MOVES][6]; // Solutions for each puzzle
  int puzzle_set_solution_lengths[MAX_PUZZLE_SET_SIZE];  // Solution lengths
  int puzzle_set_size;          // Number of puzzles in the set
  int current_puzzle_idx;       // Index of current puzzle in set
  
  // Cached puzzle board state for quick reset
  ChessBoard puzzle_board_cache;   // Cached board state for current puzzle
  bool puzzle_board_cached;        // Whether we have a cached board state
  
  // Cached solution action ID for BC training
  int cached_solution_action;      // Action ID of first solution move, cached when puzzle loads
  bool solution_action_cached;     // Whether we have a cached solution action

  // New: Global puzzle training logic
  int puzzle_tries_this_episode;    // Total puzzle tries during the episode
  int puzzle_solves_this_episode;   // Total puzzle solves during the episode
  int puzzle_tries_this_env;    // Current tries for this env on current puzzle
  int puzzle_max_tries_per_env; // Maximum tries per env before reset (default
                                // 10)
  clock_t puzzle_start_time;    // Wall clock time when puzzle started
  int puzzle_samples_to_solve;  // Number of samples taken to solve this puzzle
  // Complete game logging
  char complete_game_moves[1024][6]; // Store canonical UCI moves (e.g., "e2e4")
  int complete_game_action_count;
  char serialized_moves[1024]; // Comma-separated action IDs for efficient
                               // logging

  // Simple game logging tracking
  int steps_since_last_log;   // Steps in this environment since last game log
  int game_logging_frequency; // Log games every N steps (from config)
  bool log_next_game;         // Flag set by Python layer to trigger logging
  // Reward configuration (copied from CChess for performance)
  float c_reward_valid;
  float c_reward_invalid_white;
  float c_reward_invalid_black;
  float c_reward_white_captures_enemy_piece;
  float c_reward_black_captures_enemy_piece;
  float c_reward_max_depth_termination;
  float c_reward_draw;
  float c_reward_win_white;
  float c_reward_win_black;
  float c_reward_loss_white;
  float c_reward_loss_black;
  float c_reward_check_white;
  float c_reward_check_black;
  float c_reward_material_diff_white;
  float c_reward_material_diff_black;
  bool c_use_piece_value_capture_rewards;
  float c_piece_value_reward_multiplier;

  // Puzzle mode rewards
  float c_reward_puzzle_solved;
  float c_reward_puzzle_failed;
  float c_reward_correct_move;
  // ACCUMULATED REWARD COUNTERS (for add_log aggregation)
  float accumulated_reward_valid;
  float accumulated_reward_white_captures_enemy_piece;
  float accumulated_reward_black_captures_enemy_piece;
  float accumulated_reward_draw;
  float accumulated_reward_win_white;
  float accumulated_reward_win_black;
  float accumulated_reward_loss_white;
  float accumulated_reward_loss_black;
  float accumulated_reward_draw_white;
  float accumulated_reward_draw_black;
  float accumulated_reward_check_white;
  float accumulated_reward_check_black;
  float accumulated_reward_material_diff_white;
  float accumulated_reward_material_diff_black;
  float accumulated_stockfish_eval;

  // Puzzle reward accumulation
  float accumulated_reward_puzzle_solved;
  float accumulated_reward_puzzle_failed;
  float accumulated_reward_puzzle_correct_move;

  // Puzzle stats accumulation (for this episode)
  float puzzle_attempts_this_episode;
  float puzzle_correct_moves_this_episode;
  float puzzle_wrong_moves_this_episode;
  float puzzle_solve_rate_this_episode;  // Solves / tries ratio for logging
  // Accumulated statistics (for logging)
  float c_white_moves;
  float c_black_moves;
  float c_valid_moves;
  float c_invalid_moves_white;
  float c_invalid_moves_black;
  // Game outcome counters
  float c_white_win;
  float c_white_loss;
  float c_black_win;
  float c_black_loss;
  float c_game_drawn;
  float c_max_depth;
  // Game end condition counters
  float c_white_checkmated;
  float c_black_checkmated;
  float c_stalemate;
  float c_insufficient_material;
  float c_threefold_repetition;
  float c_fifty_move_rule;
  float c_en_passant_white;
  float c_en_passant_black;
  float c_white_castle_kingside;
  float c_white_castle_queenside;
  float c_black_castle_kingside;
  float c_black_castle_queenside;
  float c_white_promotion_count;
  float c_white_promotion_knight;
  float c_white_promotion_bishop;
  float c_white_promotion_rook;
  float c_white_promotion_queen;
  float c_black_promotion_count;
  float c_black_promotion_knight;
  float c_black_promotion_bishop;
  float c_black_promotion_rook;
  float c_black_promotion_queen;
  // Stockfish integration parameters (for future implementation)
  char stockfish_cmd[256];
  int stockfish_elo;
  int stockfish_search_ms;

  // PERFORMANCE OPTIMIZATION: Observation caching
  float cached_observation[1472]; // 23 * 8 * 8 board planes only
  bool observation_cached;
  uint64_t cached_observation_hash;
  PieceColor cached_observation_player;
} ChessContext;
// === PUFFERLIB ENVIRONMENT STRUCTURE ===
typedef struct CChess {
  Log log;
  int env_id; // Add back env_id field
  float *observations;
  int *actions;
  float *rewards;
  unsigned char *terminals;
  // Configuration values from INI file
  float reward_valid;
  float reward_invalid_white;
  float reward_invalid_black;
  float reward_white_captures_enemy_piece;
  float reward_black_captures_enemy_piece;
  float reward_draw;
  float reward_win_white;
  float reward_win_black;
  float reward_loss_white;
  float reward_loss_black;
  float reward_check_white;
  float reward_check_black;
  int max_depth;
  float reward_material_diff_white;
  float reward_material_diff_black;
  float reward_max_depth_termination;
  float reward_puzzle_solved;
  float reward_puzzle_failed;
  float reward_correct_move;
  float reward_puzzle_correct_piece;
  float reward_puzzle_closer_to_target;
  float reward_puzzle_correct_promotion;
  bool use_piece_value_capture_rewards;
  float piece_value_reward_multiplier;
  // Debug settings
  bool debug_disable_mask;
  bool stockfish_enabled;
  // Chess context (pure C, no opaque pointer)
  ChessContext context;
  // Convenience pointer to avoid repeated dereferencing
  ChessContext *ctx;
  // Global puzzle coordination across all environments
  int global_puzzle_id;        // Current puzzle ID all environments work on
  int global_puzzle_attempts;  // Total attempts across all envs for current
                               // puzzle
  int global_puzzle_successes; // Total successes across all envs for current
                               // puzzle
  float global_puzzle_success_threshold; // Threshold to advance (default 0.9)
  int puzzle_max_tries_per_env; // Max tries per env before reset (default 10)
  
  // Persistent puzzle tracking that survives log resets
  int total_puzzle_attempts;   // Total puzzle attempts since init
  int total_puzzle_solves;     // Total puzzles solved since init
} CChess;
// === ADDITIONAL BINDING FUNCTIONS ===
void enable_stockfish_black(CChess *env, const char *stockfish_cmd, int elo,
                            int search_ms);
void set_self_play_mode(CChess *env, bool enabled);
void set_dual_agent_self_play_mode(CChess *env, bool enabled);
void set_debug_disable_mask(CChess *env, bool enabled);
void set_puzzle_mode(CChess *env, bool enabled);
void set_puzzle_data(CChess *env, const char *fen, const char *solution_moves[],
                     int solution_length);
void set_puzzle_set(CChess *env, int num_puzzles, const char **fens, 
                    const char ***solution_moves, const int *solution_lengths);
void set_puzzle_difficulty(CChess *env, int difficulty);
void set_puzzle_training_params(CChess *env, int max_tries_per_env,
                                float success_threshold);
// === PUFFERLIB REQUIRED FUNCTIONS ===
void init(CChess *env);
void allocate(CChess *env);
void free_allocated(CChess *env);
void add_log(CChess *env);
void update_puzzle_score(CChess *env);
void c_reset(CChess *env);
void c_step(CChess *env);
void c_render(CChess *env);
void c_close(CChess *env);
// === MODE SETTERS FOR COMPATIBILITY ===
void set_dual_agent_self_play_mode(CChess *env, bool enabled);
void set_self_play_mode(CChess *env, bool enabled);
void c_set_fen(CChess *env, const char *fen);
// Stub implementation for notify_game_end
void notify_game_end(int white_won, int black_won, int is_draw) {
  // Stub implementation - could be used for logging or callbacks
}
// === CHESS HELPER FUNCTIONS ===
// Print board in human-readable format
void print_board_state(ChessBoard *board) {
  // printf("\n a b c d e f g h\n");
  // printf(" ---------------\n");

  for (int y = 7; y >= 0; y--) {
    // printf("%d|", y + 1);
    for (int x = 0; x < 8; x++) {
      Piece *p = &board->board[y * 8 + x];
      char piece_char = ' ';

      if (p->type != EMPTY) {
        switch (p->type) {
        case PAWN:
          piece_char = 'P';
          break;
        case KNIGHT:
          piece_char = 'N';
          break;
        case BISHOP:
          piece_char = 'B';
          break;
        case ROOK:
          piece_char = 'R';
          break;
        case QUEEN:
          piece_char = 'Q';
          break;
        case KING:
          piece_char = 'K';
          break;
        }

        // Lowercase for black pieces
        if (p->color == C_BLACK) {
          piece_char = piece_char + 32; // Convert to lowercase
        }
      }

      // printf("%c ", piece_char);
    }
    // printf("|%d\n", y + 1);
  }

  // printf(" ---------------\n");
  // printf(" a b c d e f g h\n");
  // printf("Turn: %s\n\n", (board->to_move == C_WHITE) ? "WHITE" : "BLACK");
}
// Material calculation using standard chess piece values
int calculate_material_value(ChessBoard *board, PieceColor color) {
  int total = 0;
  for (int i = 0; i < 64; i++) {
    Piece *p = &board->board[i];
    if (p->type != EMPTY && p->color == color) {
      switch (p->type) {
      case PAWN:
        total += 1;
        break;
      case KNIGHT:
        total += 3;
        break;
      case BISHOP:
        total += 3;
        break;
      case ROOK:
        total += 5;
        break;
      case QUEEN:
        total += 9;
        break;
      case KING:
        total += 0;
        break; // King has no material value
      default:
        break;
      }
    }
  }
  return total;
}
// Get individual piece value for capture rewards
int get_piece_value(PieceType piece_type) {
  switch (piece_type) {
  case PAWN:
    return 1;
  case KNIGHT:
    return 3;
  case BISHOP:
    return 3;
  case ROOK:
    return 5;
  case QUEEN:
    return 9;
  case KING:
    return 0; // King cannot be captured in normal play
  default:
    return 0;
  }
}
// Board access
static inline Piece *get_piece(ChessBoard *board, int x, int y) {
  if (x < 0 || x >= 8 || y < 0 || y >= 8)
    return NULL;
  return &board->board[y * 8 + x];
}
static inline const Piece *get_piece_const(const ChessBoard *board, int x,
                                           int y) {
  if (x < 0 || x >= 8 || y < 0 || y >= 8)
    return NULL;
  return &board->board[y * 8 + x];
}
// Square notation conversion
static inline Square notation_to_square(const char *notation) {
  Square result;
  if (!notation || strlen(notation) < 2) {
    result.x = -1;
    result.y = -1;
    return result;
  }
  int x = notation[0] - 'a';
  int y = notation[1] - '1';
  if (x >= 0 && x < 8 && y >= 0 && y < 8) {
    result.x = (int8_t)x;
    result.y = (int8_t)y;
  } else {
    result.x = -1;
    result.y = -1;
  }
  return result;
}
static inline void square_to_notation(Square sq, char *notation) {
  if (sq.x >= 0 && sq.x < 8 && sq.y >= 0 && sq.y < 8) {
    notation[0] = 'a' + sq.x;
    notation[1] = '1' + sq.y;
    notation[2] = '\0';
  } else {
    strcpy(notation, "--");
  }
}
// Simple hash function for position history
// Zobrist hash tables for proper position hashing
static uint64_t zobrist_piece_square[2][7][64]; // [color][piece_type][square]
static uint64_t zobrist_castle_rights[16];      // [castle_rights]
static uint64_t zobrist_en_passant[64];         // [ep_square]
static uint64_t zobrist_side_to_move;
static bool zobrist_initialized = false;
static void init_zobrist_tables(void) {
  if (zobrist_initialized)
    return;
  // Simple PRNG for generating Zobrist values
  uint64_t seed = 0x123456789abcdefULL;
  for (int color = 0; color < 2; color++) {
    for (int piece = 0; piece < 7; piece++) {
      for (int square = 0; square < 64; square++) {
        seed = seed * 0x9e3779b97f4a7c15ULL + 0x85ebca6b;
        zobrist_piece_square[color][piece][square] = seed;
      }
    }
  }
  for (int i = 0; i < 16; i++) {
    seed = seed * 0x9e3779b97f4a7c15ULL + 0x85ebca6b;
    zobrist_castle_rights[i] = seed;
  }
  for (int i = 0; i < 64; i++) {
    seed = seed * 0x9e3779b97f4a7c15ULL + 0x85ebca6b;
    zobrist_en_passant[i] = seed;
  }
  seed = seed * 0x9e3779b97f4a7c15ULL + 0x85ebca6b;
  zobrist_side_to_move = seed;
  zobrist_initialized = true;
}
// Fast hash lookup - just return the incrementally maintained hash
static inline uint64_t hash_position(const ChessBoard *board) {
  return board->zobrist_hash;
}
// Compute hash from scratch (only used for initialization)
static uint64_t compute_zobrist_hash(const ChessBoard *board) {
  if (!zobrist_initialized)
    init_zobrist_tables();
  uint64_t hash = 0;
  // Hash pieces
  for (int i = 0; i < 64; i++) {
    if (board->board[i].type != EMPTY) {
      hash ^=
          zobrist_piece_square[board->board[i].color][board->board[i].type][i];
    }
  }
  // Hash side to move
  if (board->to_move == C_BLACK) {
    hash ^= zobrist_side_to_move;
  }
  // Hash castling rights
  hash ^= zobrist_castle_rights[board->castle_rights & 15];
  // Hash en passant square
  if (board->ep_square >= 0 && board->ep_square < 64) {
    hash ^= zobrist_en_passant[board->ep_square];
  }
  return hash;
}
// Incrementally update hash when making a move
static inline void update_zobrist_hash(ChessBoard *board, int from_square,
                                       int to_square, Piece moved_piece,
                                       Piece captured_piece,
                                       uint8_t old_castle_rights,
                                       int8_t old_ep_square) {
  if (!zobrist_initialized)
    init_zobrist_tables();
  // Remove old piece from from_square
  board->zobrist_hash ^=
      zobrist_piece_square[moved_piece.color][moved_piece.type][from_square];
  // Add piece to to_square
  board->zobrist_hash ^=
      zobrist_piece_square[moved_piece.color][moved_piece.type][to_square];
  // Remove captured piece if any
  if (captured_piece.type != EMPTY) {
    board->zobrist_hash ^= zobrist_piece_square[captured_piece.color]
                                               [captured_piece.type][to_square];
  }
  // Update side to move
  board->zobrist_hash ^= zobrist_side_to_move;
  // Update castling rights
  board->zobrist_hash ^= zobrist_castle_rights[old_castle_rights & 15];
  board->zobrist_hash ^= zobrist_castle_rights[board->castle_rights & 15];
  // Update en passant
  if (old_ep_square >= 0 && old_ep_square < 64) {
    board->zobrist_hash ^= zobrist_en_passant[old_ep_square];
  }
  if (board->ep_square >= 0 && board->ep_square < 64) {
    board->zobrist_hash ^= zobrist_en_passant[board->ep_square];
  }
}
// === GAME LOGGING HELPERS ===
static void serialize_complete_game_moves(ChessContext *ctx) {
  ctx->serialized_moves[0] = '\0';
  if (ctx->complete_game_action_count == 0) {
    return;
  }
  char temp[16];
  for (int i = 0; i < ctx->complete_game_action_count && i < 100; i++) {
    if (i > 0) {
      strcat(ctx->serialized_moves, ",");
    }
    // Convert UCI move back to action ID for serialization compatibility
    int action_id = uci_to_action_id(ctx->complete_game_moves[i]);
    // sprintf(temp, "%d", action_id);
    strcat(ctx->serialized_moves, temp);
  }
}
// Write complete game to file for analysis
static void write_complete_game_to_file(ChessContext *ctx, int env_id) {
  // Re-enabled for debugging game logging functionality

  if (ctx->complete_game_action_count == 0) {
    return;
  }
  // Create directory if needed
  int mkdir_result =
      system("mkdir -p pufferlib/resources/chess/training_logs/complete_games");

  // Determine game result and termination reason
  char result_str[16] = "*";            // Default: incomplete/ongoing
  char termination[64] = "depth_limit"; // Default termination reason

  if (ctx->c_white_checkmated > 0) {
    strcpy(result_str, "0-1");
    strcpy(termination, "white_checkmated");
  } else if (ctx->c_black_checkmated > 0) {
    strcpy(result_str, "1-0");
    strcpy(termination, "black_checkmated");
  } else if (ctx->c_stalemate > 0) {
    strcpy(result_str, "1/2-1/2");
    strcpy(termination, "stalemate");
  } else if (ctx->c_fifty_move_rule > 0) {
    strcpy(result_str, "1/2-1/2");
    strcpy(termination, "fifty_move_rule");
  } else if (ctx->c_threefold_repetition > 0) {
    strcpy(result_str, "1/2-1/2");
    strcpy(termination, "threefold_repetition");
  } else if (ctx->c_insufficient_material > 0) {
    strcpy(result_str, "1/2-1/2");
    strcpy(termination, "insufficient_material");
  }

  // Generate filename with timestamp, env_id, result and termination
  time_t now = time(NULL);
  char filename[256];

  // Create safe result and termination strings for filename
  char safe_result[16], safe_termination[64];
  strcpy(safe_result, result_str);
  strcpy(safe_termination, termination);

  // Replace problematic characters only in result/termination parts
  for (char *p = safe_result; *p; p++) {
    if (*p == '/' || *p == '-')
      *p = '_';
  }
  for (char *p = safe_termination; *p; p++) {
    if (*p == '/' || *p == '-')
      *p = '_';
  }

  sprintf(filename,
          "pufferlib/resources/chess/training_logs/complete_games/"
          "game_%d_%ld_%s_%s.pgn",
          env_id, now, safe_result, safe_termination);

  FILE *file = fopen(filename, "w");
  if (!file) {
    return;
  } else {
  }

  // Write PGN header
  // fprintf(file, "[Event \"PufferLib Training Game\"]\n");
  // fprintf(file, "[Site \"Environment %d\"]\n", env_id);
  // fprintf(file, "[Date \"%ld\"]\n", now);
  // fprintf(file, "[White \"AI-White\"]\n");
  // fprintf(file, "[Black \"AI-Black\"]\n");
  // fprintf(file, "[Result \"%s\"]\n", result_str);
  // fprintf(file, "[Termination \"%s\"]\n", termination);
  // fprintf(file, "\n");

  // Write moves in algebraic notation
  int move_number = 1;
  for (int i = 0; i < ctx->complete_game_action_count; i++) {
    // Use the stored canonical UCI move directly
    const char *uci_move = ctx->complete_game_moves[i];

    // Simple UCI to algebraic (basic format: from-to, e.g. e2e4)
    if (i % 2 == 0) {
      // fprintf(file, "%d. %s ", move_number, uci_move);
      if (i == ctx->complete_game_action_count - 1) {
        // fprintf(file, "\n");
      }
    } else {
      // fprintf(file, "%s ", uci_move);
      if (i % 4 == 1) {
        // fprintf(file, "\n");
      }
      move_number++;
    }
  }

  if (ctx->complete_game_action_count % 2 == 1) {
    // fprintf(file, "\n");
  }
  // fprintf(file, "%s\n", result_str);

  fclose(file);
  // printf("[Chess] Logged complete game to %s\n", filename);
}
// === PERSPECTIVE FLIPPING FOR SELF-PLAY ===
static inline void flip_uci_for_black_perspective(const char *original_uci,
                                                  char *flipped_uci) {
  flipped_uci[0] = original_uci[0];             // file stays same
  flipped_uci[1] = '9' - original_uci[1] + '0'; // flip rank: 1→8, 2→7, etc.
  flipped_uci[2] = original_uci[2];             // file stays same
  flipped_uci[3] = '9' - original_uci[3] + '0'; // flip rank
  if (strlen(original_uci) >= 5) {
    flipped_uci[4] = original_uci[4]; // promotion piece unchanged
    flipped_uci[5] = '\0';
  } else {
    flipped_uci[4] = '\0';
  }
}
// === LEGAL MOVE GENERATION (PURE C, OPTIMIZED) ===
static bool is_square_attacked(const ChessBoard *board, Square sq,
                               PieceColor by_color) {
  PROFILE_START(profile_is_square_attacked_ticks)
  // RAY-BASED ATTACK DETECTION: Start from target square and look outward for
  // attackers
  // Check pawn attacks (2 squares diagonally in front from attacker's
  // perspective)
  int pawn_direction = (by_color == C_WHITE) ? 1 : -1;
  for (int dx = -1; dx <= 1; dx += 2) { // -1 and +1
    int x = sq.x + dx;
    int y =
        sq.y - pawn_direction; // Reverse direction since we're looking backward
    if (x >= 0 && x < 8 && y >= 0 && y < 8) {
      const Piece *p = get_piece_const(board, x, y);
      if (p && p->type == PAWN && p->color == by_color) {
        PROFILE_STOP(profile_is_square_attacked_ticks)
        return true;
      }
    }
  }
  // Check knight attacks (8 possible positions)
  int knight_moves[][2] = {{2, 1}, {2, -1}, {-2, 1}, {-2, -1},
                           {1, 2}, {1, -2}, {-1, 2}, {-1, -2}};
  for (int i = 0; i < 8; i++) {
    int x = sq.x + knight_moves[i][0];
    int y = sq.y + knight_moves[i][1];
    if (x >= 0 && x < 8 && y >= 0 && y < 8) {
      const Piece *p = get_piece_const(board, x, y);
      if (p && p->type == KNIGHT && p->color == by_color) {
        PROFILE_STOP(profile_is_square_attacked_ticks)
        return true;
      }
    }
  }
  // Check king attacks (8 adjacent squares)
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      if (dx == 0 && dy == 0)
        continue;
      int x = sq.x + dx;
      int y = sq.y + dy;
      if (x >= 0 && x < 8 && y >= 0 && y < 8) {
        const Piece *p = get_piece_const(board, x, y);
        if (p && p->type == KING && p->color == by_color) {
          PROFILE_STOP(profile_is_square_attacked_ticks)
          return true;
        }
      }
    }
  }
  // Check sliding piece attacks (rook, bishop, queen) by radiating outward
  int directions[][2] = {{0, 1}, {1, 0},  {0, -1}, {-1, 0},
                         {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
  for (int d = 0; d < 8; d++) {
    int dx = directions[d][0];
    int dy = directions[d][1];
    bool is_diagonal = (d >= 4);
    // Radiate outward in this direction until we hit a piece or board edge
    for (int dist = 1; dist < 8; dist++) {
      int x = sq.x + dx * dist;
      int y = sq.y + dy * dist;
      if (x < 0 || x >= 8 || y < 0 || y >= 8)
        break;
      const Piece *p = get_piece_const(board, x, y);
      if (p && p->type != EMPTY) {
        // Found a piece - check if it can attack this square
        if (p->color == by_color) {
          if (p->type == QUEEN) {
            PROFILE_STOP(profile_is_square_attacked_ticks)
            return true;
          }
          if (p->type == BISHOP && is_diagonal) {
            PROFILE_STOP(profile_is_square_attacked_ticks)
            return true;
          }
          if (p->type == ROOK && !is_diagonal) {
            PROFILE_STOP(profile_is_square_attacked_ticks)
            return true;
          }
        }
        break; // Any piece blocks further pieces in this direction
      }
    }
  }
  PROFILE_STOP(profile_is_square_attacked_ticks)
  return false;
}
static bool is_in_check(const ChessBoard *board, PieceColor color) {
  // Find king
  for (int x = 0; x < 8; x++) {
    for (int y = 0; y < 8; y++) {
      const Piece *piece = get_piece_const(board, x, y);
      if (piece && piece->type == KING && piece->color == color) {
        PieceColor opponent = (color == C_WHITE) ? C_BLACK : C_WHITE;
        Square king_pos;
        king_pos.x = (int8_t)x;
        king_pos.y = (int8_t)y;
        return is_square_attacked(board, king_pos, opponent);
      }
    }
  }
  return false; // No king found
}
// Optimized legal move generation (returns UCI strings directly)
// === LEGAL MOVE GENERATION ===
typedef struct {
  Square from;
  Square to;
  PieceType promotion;
  bool is_castling;
  bool is_en_passant;
} ChessMove;
// Helper function to convert action ID to UCI notation
static void action_to_uci(int action_id, char *uci_str) {
  if (action_id < 0 || action_id >= TOTAL_CHESS_ACTIONS) {
    strcpy(uci_str, "0000");
    return;
  }
  strcpy(uci_str, ACTION_ID_TO_UCI[action_id]);
}
// Helper function to parse UCI move string into ChessMove struct
static bool parse_uci_move(const char *uci_str, ChessMove *move) {
  if (!uci_str || strlen(uci_str) < 4) {
    return false;
  }

  // Parse from square
  move->from.x = uci_str[0] - 'a';
  move->from.y = uci_str[1] - '1';
  move->to.x = uci_str[2] - 'a';
  move->to.y = uci_str[3] - '1';

  // Check bounds
  if (move->from.x < 0 || move->from.x >= 8 || move->from.y < 0 ||
      move->from.y >= 8 || move->to.x < 0 || move->to.x >= 8 ||
      move->to.y < 0 || move->to.y >= 8) {
    return false;
  }

  // Parse promotion if present
  move->promotion = EMPTY;
  if (strlen(uci_str) == 5) {
    switch (uci_str[4]) {
    case 'q':
      move->promotion = QUEEN;
      break;
    case 'r':
      move->promotion = ROOK;
      break;
    case 'b':
      move->promotion = BISHOP;
      break;
    case 'n':
      move->promotion = KNIGHT;
      break;
    default:
      return false;
    }
  }

  // Special moves detection (will be refined based on board state)
  move->is_castling = false;
  move->is_en_passant = false;

  return true;
}
typedef struct {
  ChessMove moves[256];
  int count;
} LegalMoves;
// Undo information for make/unmake move optimization
typedef struct {
  Piece captured_piece;
  uint8_t old_castle_rights;
  int8_t old_ep_square;
  uint8_t old_halfmove_clock;
  uint64_t old_zobrist_hash;
  bool was_castling;
  bool was_en_passant;
  Square rook_from, rook_to; // For castling undo
} UndoInfo;
// Forward declarations
static bool apply_uci_move(ChessContext *ctx, const char *uci_str);
static void add_position_to_history(ChessContext *ctx, uint64_t hash);
static int get_position_count(ChessContext *ctx, uint64_t hash);
static bool is_threefold_repetition(ChessContext *ctx);
static bool is_insufficient_material(ChessContext *ctx);
static void make_move_fast(ChessBoard *board, ChessMove move, UndoInfo *undo);
static void unmake_move_fast(ChessBoard *board, ChessMove move, UndoInfo *undo);
static bool chess_is_legal_move(ChessContext *ctx, ChessMove move) {
  PROFILE_START(profile_is_legal_move_ticks)
  // MAKE/UNMAKE OPTIMIZATION: Use actual board instead of copying
  ChessBoard *board = &ctx->board;
  PieceColor moving_color = board->to_move;
  // Basic validity checks
  Piece *from_piece = get_piece(board, move.from.x, move.from.y);
  if (!from_piece || from_piece->type == EMPTY ||
      from_piece->color != moving_color) {
    PROFILE_STOP(profile_is_legal_move_ticks)
    return false;
  }
  // Handle castling move specially - check path before making move
  if (move.is_castling) {
    if (from_piece->type != KING) {
      PROFILE_STOP(profile_is_legal_move_ticks)
      return false;
    }
    if (is_in_check(board, moving_color)) {
      PROFILE_STOP(profile_is_legal_move_ticks)
      return false; // Can't castle out of check
    }
    // Verify castling path is clear and not through check
    int rank = (moving_color == C_WHITE) ? 0 : 7;
    if (move.to.x == 6) { // Kingside
      for (int x = 5; x <= 6; x++) {
        Square sq = {(int8_t)x, (int8_t)rank};
        if (is_square_attacked(board, sq, (PieceColor)(1 - moving_color))) {
          PROFILE_STOP(profile_is_legal_move_ticks)
          return false;
        }
      }
    } else if (move.to.x == 2) { // Queenside
      for (int x = 2; x <= 3; x++) {
        Square sq = {(int8_t)x, (int8_t)rank};
        if (is_square_attacked(board, sq, (PieceColor)(1 - moving_color))) {
          PROFILE_STOP(profile_is_legal_move_ticks)
          return false;
        }
      }
    }
  }
  // Make move and save undo information
  UndoInfo undo;
  make_move_fast(board, move, &undo);
  // Check if our king is in check after the move
  bool is_legal = !is_in_check(board, moving_color);
  // Unmake the move to restore board state
  unmake_move_fast(board, move, &undo);
  PROFILE_STOP(profile_is_legal_move_ticks)
  return is_legal;
}
static void add_legal_move(ChessContext *ctx, LegalMoves *moves,
                           ChessMove move) {
  if (moves->count >= 256)
    return;
    
  // DEBUG: Check for f7f8 move
  if (move.from.x == 5 && move.from.y == 6 && move.to.x == 5 && move.to.y == 7) {
//     // printf("[F7F8 DEBUG] Checking legality of f7f8 move\n");
    fflush(stdout);
  }
  
  if (chess_is_legal_move(ctx, move)) {
    // DEBUG: f7f8 passed legality check
    if (move.from.x == 5 && move.from.y == 6 && move.to.x == 5 && move.to.y == 7) {
//       printf("[F7F8 DEBUG] *** f7f8 PASSED legality check! Adding to moves list at index %d ***\n", moves->count);
      fflush(stdout);
    }
    
    moves->moves[moves->count] = move;
    moves->count++;
  } else {
    // DEBUG: f7f8 failed legality check
    if (move.from.x == 5 && move.from.y == 6 && move.to.x == 5 && move.to.y == 7) {
//       // printf("[F7F8 DEBUG] XXX f7f8 FAILED legality check! NOT added to moves XXX\n");
      fflush(stdout);
    }
  }
}
static void generate_pseudo_legal_moves_for_piece(ChessContext *ctx,
                                                  LegalMoves *moves,
                                                  Square from) {
  ChessBoard *board = &ctx->board;
  const Piece *piece = get_piece_const(board, from.x, from.y);
  if (!piece || piece->type == EMPTY || piece->color != board->to_move)
    return;
  PieceColor us = board->to_move;
  PieceColor them = (us == C_WHITE) ? C_BLACK : C_WHITE;
  switch (piece->type) {
  case PAWN: {
    int direction = (us == C_WHITE) ? 1 : -1;
    int start_rank = (us == C_WHITE) ? 1 : 6;
    int promote_rank = (us == C_WHITE) ? 7 : 0;
    // --- 1. Single Push ---
    int single_push_y = from.y + direction;
    if (single_push_y >= 0 && single_push_y < 8) {
      const Piece *target = get_piece_const(board, from.x, single_push_y);
      if (target && target->type == EMPTY) {
        Square to_sq = {from.x, (int8_t)single_push_y};
        if (to_sq.y == promote_rank) { // Promotion on single push
          PieceType promotions[] = {QUEEN, ROOK, BISHOP, KNIGHT};
          for (int p = 0; p < 4; p++) {
            add_legal_move(
                ctx, moves,
                (ChessMove){from, to_sq, promotions[p], false, false});
          }
        } else { // Regular single push
          add_legal_move(ctx, moves,
                         (ChessMove){from, to_sq, EMPTY, false, false});
        }
      }
    }
    // --- 2. Double Push ---
    if (from.y == start_rank) {
      int single_push_y_check = from.y + direction;
      int double_push_y = from.y + 2 * direction;
      // The y coordinates for double push are always in-bounds, no need for y
      // check
      const Piece *path_blocker =
          get_piece_const(board, from.x, single_push_y_check);
      const Piece *target = get_piece_const(board, from.x, double_push_y);
      if (path_blocker && path_blocker->type == EMPTY && target &&
          target->type == EMPTY) {
        Square to_sq = {from.x, (int8_t)double_push_y};
        add_legal_move(ctx, moves,
                       (ChessMove){from, to_sq, EMPTY, false, false});
      }
    }
    // --- 3. Captures & En Passant ---
    for (int dx = -1; dx <= 1; dx += 2) {
      int capture_x = from.x + dx;
      int capture_y = from.y + direction;
      // Check bounds for the destination square FIRST
      if (capture_x >= 0 && capture_x < 8 && capture_y >= 0 && capture_y < 8) {
        Square to_sq = {(int8_t)capture_x, (int8_t)capture_y};
        const Piece *target = get_piece_const(board, capture_x, capture_y);
        // Regular capture
        if (target && target->type != EMPTY && target->color == them) {
          if (to_sq.y == promote_rank) { // Promotion on capture
            PieceType promotions[] = {QUEEN, ROOK, BISHOP, KNIGHT};
            for (int p = 0; p < 4; p++) {
              add_legal_move(
                  ctx, moves,
                  (ChessMove){from, to_sq, promotions[p], false, false});
            }
          } else { // Regular capture
            add_legal_move(ctx, moves,
                           (ChessMove){from, to_sq, EMPTY, false, false});
          }
        }
        // En Passant capture
        else if (board->ep_square == (capture_y * 8 + capture_x)) {
          // An en-passant capture cannot result in a promotion.
          add_legal_move(ctx, moves,
                         (ChessMove){from, to_sq, EMPTY, false, true});
        }
      }
    }
    break;
  }
  case ROOK:
  case BISHOP:
  case QUEEN: {
    // Sliding pieces
    int directions[8][2];
    int num_dirs;
    if (piece->type == ROOK) {
      int rook_dirs[][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
      memcpy(directions, rook_dirs, sizeof(rook_dirs));
      num_dirs = 4;
    } else if (piece->type == BISHOP) {
      int bishop_dirs[][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
      memcpy(directions, bishop_dirs, sizeof(bishop_dirs));
      num_dirs = 4;
    } else { // QUEEN
      int queen_dirs[][2] = {{0, 1}, {1, 0},  {0, -1}, {-1, 0},
                             {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
      memcpy(directions, queen_dirs, sizeof(queen_dirs));
      num_dirs = 8;
      
      // DEBUG: Special check for Queen on f7
      if (piece->type == QUEEN && from.x == 5 && from.y == 6) {
//         // printf("[F7F8 DEBUG] Generating moves for Queen on f7 (square 53)\n");
        fflush(stdout);
      }
    }
    for (int d = 0; d < num_dirs; d++) {
      int dx = directions[d][0];
      int dy = directions[d][1];
      for (int dist = 1; dist < 8; dist++) {
        Square to;
        to.x = from.x + dx * dist;
        to.y = from.y + dy * dist;
        if (to.x < 0 || to.x >= 8 || to.y < 0 || to.y >= 8)
          break;
          
        // DEBUG: Check for f7f8 move
        if (piece->type == QUEEN && from.x == 5 && from.y == 6 && to.x == 5 && to.y == 7) {
//           printf("[F7F8 DEBUG] Found f7f8 candidate: Queen from (%d,%d) to (%d,%d)\n", from.x, from.y, to.x, to.y);
          fflush(stdout);
        }
        
        const Piece *target = get_piece_const(board, to.x, to.y);
        if (target && target->type != EMPTY) {
          if (target->color == them) {
            // Capture and stop
            ChessMove move = {from, to, EMPTY, false, false};
            
            // DEBUG: Special logging for f7f8
            if (piece->type == QUEEN && from.x == 5 && from.y == 6 && to.x == 5 && to.y == 7) {
//               printf("[F7F8 DEBUG] f7f8 is a CAPTURE (target: type=%d, color=%d). Calling add_legal_move...\n", 
//                      target->type, target->color);
//               fflush(stdout);
            }
            
            add_legal_move(ctx, moves, move);
          }
          break; // Blocked
        } else {
          // Empty square
          ChessMove move = {from, to, EMPTY, false, false};
          add_legal_move(ctx, moves, move);
        }
      }
    }
    break;
  }
  case KNIGHT: {
    int knight_moves[][2] = {{2, 1}, {2, -1}, {-2, 1}, {-2, -1},
                             {1, 2}, {1, -2}, {-1, 2}, {-1, -2}};
    for (int i = 0; i < 8; i++) {
      Square to;
      to.x = from.x + knight_moves[i][0];
      to.y = from.y + knight_moves[i][1];
      if (to.x >= 0 && to.x < 8 && to.y >= 0 && to.y < 8) {
        const Piece *target = get_piece_const(board, to.x, to.y);
        if (!target || target->type == EMPTY || target->color == them) {
          ChessMove move = {from, to, EMPTY, false, false};
          add_legal_move(ctx, moves, move);
        }
      }
    }
    break;
  }
  case KING: {
    // Regular king moves
    for (int dx = -1; dx <= 1; dx++) {
      for (int dy = -1; dy <= 1; dy++) {
        if (dx == 0 && dy == 0)
          continue;
        Square to;
        to.x = from.x + dx;
        to.y = from.y + dy;
        if (to.x >= 0 && to.x < 8 && to.y >= 0 && to.y < 8) {
          const Piece *target = get_piece_const(board, to.x, to.y);
          if (!target || target->type == EMPTY || target->color == them) {
            ChessMove move = {from, to, EMPTY, false, false};
            add_legal_move(ctx, moves, move);
          }
        }
      }
    }
    // Castling
    if (!is_in_check(board, us)) {
      int rank = (us == C_WHITE) ? 0 : 7;
      if (from.x == 4 && from.y == rank) {
        // Kingside castling
        if ((board->castle_rights & (us == C_WHITE ? 1 : 4))) {
          bool can_castle = true;
          // Check squares are empty and not attacked
          for (int x = 5; x <= 6; x++) {
            const Piece *sq = get_piece_const(board, x, rank);
            if (sq && sq->type != EMPTY) {
              can_castle = false;
              break;
            }
            Square check_sq = {(int8_t)x, (int8_t)rank};
            if (is_square_attacked(board, check_sq, them)) {
              can_castle = false;
              break;
            }
          }
          if (can_castle) {
            Square to;
            to.x = 6;
            to.y = (int8_t)rank;
            ChessMove move = {from, to, EMPTY, true, false};
            add_legal_move(ctx, moves, move);
          }
        }
        // Queenside castling
        if ((board->castle_rights & (us == C_WHITE ? 2 : 8))) {
          bool can_castle = true;
          // Check squares are empty
          for (int x = 1; x <= 3; x++) {
            const Piece *sq = get_piece_const(board, x, rank);
            if (sq && sq->type != EMPTY) {
              can_castle = false;
              break;
            }
          }
          // Check squares are not attacked
          if (can_castle) {
            for (int x = 2; x <= 3; x++) {
              Square check_sq = {(int8_t)x, (int8_t)rank};
              if (is_square_attacked(board, check_sq, them)) {
                can_castle = false;
                break;
              }
            }
          }
          if (can_castle) {
            Square to;
            to.x = 2;
            to.y = (int8_t)rank;
            ChessMove move = {from, to, EMPTY, true, false};
            add_legal_move(ctx, moves, move);
          }
        }
      }
    }
    break;
  }
  case EMPTY:
  default:
    break;
  }
}
// Yield-based move generation callback type
typedef bool (*MoveYieldCallback)(ChessContext *ctx, const ChessMove *move,
                                  void *user_data);
// Callback that terminates on first legal move found
static bool first_move_callback(ChessContext *ctx, const ChessMove *move,
                                void *user_data) {
  bool *found = (bool *)user_data;
  *found = true;
  return true; // Terminate immediately
}
// Yield-based move generation - returns true if callback requested early
// termination
static bool chess_generate_legal_moves_yield(ChessContext *ctx,
                                             MoveYieldCallback yield_fn,
                                             void *user_data) {
  // Iterate through all squares on the board
  for (int x = 0; x < 8; x++) {
    for (int y = 0; y < 8; y++) {
      Square from = {(int8_t)x, (int8_t)y};
      const Piece *piece = get_piece_const(&ctx->board, x, y);
      if (piece && piece->type != EMPTY && piece->color == ctx->board.to_move) {
        // Generate moves for this piece and yield each one
        LegalMoves temp_moves;
        temp_moves.count = 0;
        generate_pseudo_legal_moves_for_piece(ctx, &temp_moves, from);
        // Yield each move individually
        for (int i = 0; i < temp_moves.count; i++) {
          // Call yield callback - if it returns true, terminate early
          if (yield_fn(ctx, &temp_moves.moves[i], user_data)) {
            return true; // Early termination requested
          }
        }
      }
    }
  }
  return false; // Completed without early termination
}
static void chess_generate_legal_moves(ChessContext *ctx, LegalMoves *moves) {
  moves->count = 0;
  // Iterate through all squares on the board
  for (int x = 0; x < 8; x++) {
    for (int y = 0; y < 8; y++) {
      Square from = {(int8_t)x, (int8_t)y};
      const Piece *piece = get_piece_const(&ctx->board, x, y);
      if (piece && piece->type != EMPTY && piece->color == ctx->board.to_move) {
        generate_pseudo_legal_moves_for_piece(ctx, moves, from);
      }
    }
  }
}
static int chess_generate_legal_moves_uci(ChessContext *ctx) {
  PROFILE_START(profile_move_gen_uci_ticks)

  uint64_t current_hash = ctx->board.zobrist_hash;
  PieceColor current_player = ctx->board.to_move;

  // Check if we already have moves for current player
  if (current_player == C_WHITE && ctx->white_moves_cached &&
      ctx->cached_board_hash == current_hash) {
    PROFILE_STOP(profile_move_gen_uci_ticks)
    return ctx->white_legal_moves_count;
  } else if (current_player == C_BLACK && ctx->black_moves_cached &&
             ctx->cached_board_hash == current_hash) {
    PROFILE_STOP(profile_move_gen_uci_ticks)
    return ctx->black_legal_moves_count;
  }

  // If position changed, clear both caches
  if (ctx->cached_board_hash != current_hash) {
    ctx->white_moves_cached = false;
    ctx->black_moves_cached = false;
    ctx->position_fully_cached = false;
    ctx->cached_board_hash = current_hash;
  }
  // Generate legal moves for current player
  LegalMoves moves;
  chess_generate_legal_moves(ctx, &moves);
  // Store in appropriate buffer
  char (*buffer)[6];
  int *count;
  int *action_ids;

  if (current_player == C_WHITE) {
    buffer = ctx->white_legal_moves_buffer;
    count = &ctx->white_legal_moves_count;
    action_ids = ctx->white_legal_action_ids;
  } else {
    buffer = ctx->black_legal_moves_buffer;
    count = &ctx->black_legal_moves_count;
    action_ids = ctx->black_legal_action_ids;
  }

  *count = 0;
  for (int i = 0; i < moves.count && i < 256; i++) {
    ChessMove move = moves.moves[i];
    char uci_str[6];

    if (move.promotion != EMPTY) {
      char promo_char = (move.promotion == QUEEN)    ? 'q'
                        : (move.promotion == ROOK)   ? 'r'
                        : (move.promotion == BISHOP) ? 'b'
                                                     : 'n';
      snprintf(uci_str, 6, "%c%c%c%c%c", 'a' + move.from.x, '1' + move.from.y,
               'a' + move.to.x, '1' + move.to.y, promo_char);
    } else {
      snprintf(uci_str, 5, "%c%c%c%c", 'a' + move.from.x, '1' + move.from.y,
               'a' + move.to.x, '1' + move.to.y);
    }

    strcpy(buffer[*count], uci_str);

    // Pre-compute action IDs for both perspectives
    if (current_player == C_WHITE) {
      action_ids[*count] = uci_to_action_id(uci_str);
      
      // DEBUG: Print action ID for f7f8
      if (strcmp(uci_str, "f7f8") == 0) {
        // printf("[F7F8 ACTION] f7f8 mapped to action ID %d (stored at moves[%d])\n", 
        //        action_ids[*count], *count);
        // fflush(stdout);
      }
    } else {
      // For black, we need the flipped perspective
      char flipped_uci[6];
      flip_uci_for_black_perspective(uci_str, flipped_uci);
      action_ids[*count] = uci_to_action_id(flipped_uci);
    }

    (*count)++;
  }
  // Mark this player's moves as cached
  if (current_player == C_WHITE) {
    ctx->white_moves_cached = true;
  } else {
    ctx->black_moves_cached = true;
  }
  // Check if both players are now cached
  ctx->position_fully_cached =
      ctx->white_moves_cached && ctx->black_moves_cached;
  PROFILE_STOP(profile_move_gen_uci_ticks)
  return *count;
}
// NEW FUNCTION: Generate moves for both players efficiently
static void chess_generate_all_legal_moves(ChessContext *ctx) {
  PROFILE_START(profile_move_gen_uci_ticks)

  uint64_t current_hash = ctx->board.zobrist_hash;

  // If already fully cached for this position, nothing to do
  if (ctx->position_fully_cached && ctx->cached_board_hash == current_hash) {
    PROFILE_STOP(profile_move_gen_uci_ticks)
    return;
  }

  // Clear caches if position changed
  if (ctx->cached_board_hash != current_hash) {
    ctx->white_moves_cached = false;
    ctx->black_moves_cached = false;
    ctx->position_fully_cached = false;
    ctx->cached_board_hash = current_hash;
  }

  // Generate WHITE's moves if not cached
  if (!ctx->white_moves_cached) {
    PieceColor saved_to_move = ctx->board.to_move;
    ctx->board.to_move = C_WHITE;
    chess_generate_legal_moves_uci(ctx);
    ctx->board.to_move = saved_to_move;
  }

  // Generate BLACK's moves if not cached
  if (!ctx->black_moves_cached) {
    PieceColor saved_to_move = ctx->board.to_move;
    ctx->board.to_move = C_BLACK;
    chess_generate_legal_moves_uci(ctx);
    ctx->board.to_move = saved_to_move;
  }

  ctx->position_fully_cached = true;
  PROFILE_STOP(profile_move_gen_uci_ticks)
}

// === OBSERVATION COMPUTATION WITH PERSPECTIVE FLIPPING (OPTIMIZED) ===
// Helper function to compute observation for a single agent
static void compute_single_agent_observation(CChess *env, ChessContext *ctx,
                                             PieceColor player,
                                             int obs_offset) {
  uint64_t current_hash = ctx->board.zobrist_hash;
  
  // DEBUG: Print board state when f7f8 is missing
  static int debug_call_count = 0;
  debug_call_count++;
  if (debug_call_count <= 20) {
//     // printf("[OBS ENTRY %d] Turn: %s\n", debug_call_count, 
//            ctx->board.to_move == C_WHITE ? "WHITE" : "BLACK");
  }

  // PERFORMANCE OPTIMIZATION: Check if board observation is cached
  // DISABLED FOR DEBUGGING: Force fresh observation generation
  if (false && ctx->observation_cached &&
      ctx->cached_observation_hash == current_hash &&
      ctx->cached_observation_player == player) {
    // Use cached board observation (first 1472 floats)
    memcpy(&env->observations[obs_offset], ctx->cached_observation,
           1472 * sizeof(float));
  } else {
    // Compute fresh board observation
    memset(&env->observations[obs_offset], 0, 1472 * sizeof(float));
    int idx = 0;
    
    // DEBUG: Count non-empty squares
    int piece_count = 0;
    
//     // printf("[COMPUTE_OBS] Starting compute_single_agent_observation for player %s at offset %d\n",
//            player == C_WHITE ? "WHITE" : "BLACK", obs_offset);
//     // printf("[COMPUTE_OBS] env->observations pointer: %p\n", (void*)env->observations);
    
    // CRITICAL DEBUG: Test if we can write and read back
    env->observations[0] = 999.0f;
//     printf("[MEMORY TEST] Wrote 999.0 to obs[0], readback = %.1f\n", env->observations[0]);
    env->observations[0] = 0.0f;  // Reset it
    
    // --- SINGLE PASS OVER THE BOARD (Correct) ---
    for (int y_white_perspective = 0; y_white_perspective < 8;
         y_white_perspective++) {
      for (int x = 0; x < 8; x++) {
        int y_actual = (player == C_WHITE) ? y_white_perspective
                                           : (7 - y_white_perspective);
        int square_index_actual = y_actual * 8 + x;
        const Piece *p = &ctx->board.board[square_index_actual];
        int obs_square_idx = y_white_perspective * 8 + x;
        if (p->type == EMPTY) {
          env->observations[obs_offset + 12 * 64 + obs_square_idx] = 1.0f;
        } else {
          int plane_offset = (p->color == player) ? 0 : 6;
          int piece_plane = p->type - 1;
          int obs_idx = obs_offset + (plane_offset + piece_plane) * 64 + obs_square_idx;
          env->observations[obs_idx] = 1.0f;
          piece_count++;
          
          // DEBUG: Verify write immediately
          if (piece_count <= 3) {
//             // printf("[OBS VERIFY] Just wrote env->observations[%d] = 1.0, readback = %.1f\n", 
//                    obs_idx, env->observations[obs_idx]);
          }
          
          // DEBUG: Print piece encoding details
          if (piece_count <= 5) {
            const char* piece_name = "UNKNOWN";
            switch(p->type) {
              case PAWN: piece_name = "PAWN"; break;
              case KNIGHT: piece_name = "KNIGHT"; break;
              case BISHOP: piece_name = "BISHOP"; break;
              case ROOK: piece_name = "ROOK"; break;
              case QUEEN: piece_name = "QUEEN"; break;
              case KING: piece_name = "KING"; break;
              default: break;
            }
//             // printf("[OBS ENCODE] Piece %d: %s %s at square %d (x=%d,y=%d) -> obs[%d]=1.0\n",
//                    piece_count,
//                    p->color == C_WHITE ? "WHITE" : "BLACK",
//                    piece_name,
//                    square_index_actual, x, y_actual, obs_idx);
          }
          
          // DEBUG: Print Queen on f7
          if (p->type == QUEEN && square_index_actual == 53) { // f7 = 6*8+5 = 53
//             // printf("[OBS DEBUG] Found Queen on f7 (square %d), set obs[%d] = 1.0\n", 
//                    square_index_actual, 
//                    obs_offset + (plane_offset + piece_plane) * 64 + obs_square_idx);
            fflush(stdout);
          }
        }
      }
    }
    
    // DEBUG: Print piece count
    if (piece_count < 10) {
//       // printf("[OBS DEBUG] WARNING: Only %d pieces found on board!\n", piece_count);
      fflush(stdout);
    }

    // --- NON-PIECE PLANES ---
    // Now set idx to start after piece planes (13 planes * 64 squares each)
    idx = 13 * 64;
    // Plane 13: Repetition count plane (using actual position history)
    int reps = get_position_count(ctx, ctx->board.zobrist_hash);
    float rep_val =
        (reps >= 2) ? 1.0f : 0.0f; // Simplified: 0 for 1 rep, 1 for 2+ reps
    for (int i = 0; i < 64; i++) {
      env->observations[obs_offset + idx++] = rep_val;
    }
    // Plane 14: Side to move plane (always 0 from current player's perspective)
    for (int i = 0; i < 64; i++) {
      env->observations[obs_offset + idx++] = 0.0f;
    }
    // Plane 15: Halfmove clock plane
    float halfmove_val =
        ctx->board.halfmove_clock / 100.0f; // Normalize to 0-1 range
    for (int i = 0; i < 64; i++) {
      env->observations[obs_offset + idx++] = halfmove_val;
    }
    // Planes 16-19: Castling rights planes (4 planes, flipped for black
    // perspective)
    uint8_t rights = ctx->board.castle_rights;
    if (player == C_BLACK) {
      // Flip castling rights for Black's perspective
      uint8_t flipped = 0;
      if (rights & 4)
        flipped |= 1; // BK -> WK
      if (rights & 8)
        flipped |= 2; // BQ -> WQ
      if (rights & 1)
        flipped |= 4; // WK -> BK
      if (rights & 2)
        flipped |= 8; // WQ -> BQ
      rights = flipped;
    }
    for (int i = 0; i < 4; i++) {
      float castle_val = (rights & (1 << i)) ? 1.0f : 0.0f;
      for (int j = 0; j < 64; j++) {
        env->observations[obs_offset + idx++] = castle_val;
      }
    }
    // Plane 20: En passant target square plane (flipped for black perspective)
    int8_t ep_square = ctx->board.ep_square;
    if (ep_square != -1 && player == C_BLACK) {
      int ep_x = ep_square % 8;
      int ep_y = ep_square / 8;
      ep_square = (7 - ep_y) * 8 + ep_x;
    }
    for (int i = 0; i < 64; i++) {
      env->observations[obs_offset + idx++] = (ep_square == i) ? 1.0f : 0.0f;
    }
    // Plane 21: Pieces the current player can capture on next turn
    for (int y = 0; y < 8; y++) {
      for (int x = 0; x < 8; x++) {
        int y_actual = (player == C_WHITE) ? y : (7 - y);
        int square_index = y_actual * 8 + x;
        const Piece *p = &ctx->board.board[square_index];

        bool can_capture = false;
        if (p->type != EMPTY && p->color != player) {
          // Use optimized is_square_attacked function instead of nested loops
          Square target_sq = {(int8_t)x, (int8_t)y_actual};
          can_capture = is_square_attacked(&ctx->board, target_sq, player);
        }
        env->observations[obs_offset + idx++] = can_capture ? 1.0f : 0.0f;
      }
    }
    // Plane 22: Pieces that can capture the current player's pieces on next
    // turn
    for (int y = 0; y < 8; y++) {
      for (int x = 0; x < 8; x++) {
        int y_actual = (player == C_WHITE) ? y : (7 - y);
        int square_index = y_actual * 8 + x;
        const Piece *p = &ctx->board.board[square_index];

        bool under_threat = false;
        if (p->type != EMPTY && p->color == player) {
          // Check if any opponent pieces can capture this square
          Square target_sq = {(int8_t)x, (int8_t)y_actual};
          under_threat = is_square_attacked(&ctx->board, target_sq,
                                            (PieceColor)(1 - player));
        }
        env->observations[obs_offset + idx++] = under_threat ? 1.0f : 0.0f;
      }
    }
    assert(idx == 1472); // 23 * 8 * 8

    // Cache the computed board observation
    memcpy(ctx->cached_observation, &env->observations[obs_offset],
           1472 * sizeof(float));
    ctx->observation_cached = true;
    ctx->cached_observation_hash = current_hash;
    ctx->cached_observation_player = player;
  }

  // --- SPARSE LEGAL MOVE MASK ---
  // Format: [num_legal_moves(1)] + [legal_action_ids(MAX_LEGAL_MOVES)]
  int sparse_mask_idx = 1472; // Start index for sparse mask
  const int MAX_LEGAL_MOVES = 64;

  // Ensure we have legal moves for both players
  chess_generate_all_legal_moves(ctx);

  PieceColor current_player_turn = ctx->board.to_move;
  bool is_player_turn = (player == current_player_turn);

  // Initialize sparse mask: num_legal_moves + action_ids
  int num_legal_moves = 0;
  float *action_ids_ptr = &env->observations[obs_offset + sparse_mask_idx + 1];
  
  if (env->debug_disable_mask) {
    // Debug mode: All moves are legal (first 64 actions for sparse representation)
    env->observations[obs_offset + sparse_mask_idx] = (float)MAX_LEGAL_MOVES;
    for (int i = 0; i < MAX_LEGAL_MOVES; i++) {
      action_ids_ptr[i] = (float)i;
    }
  } else if (ctx->dual_agent_self_play_mode || is_player_turn) {
    // Get the appropriate move buffer based on current turn
    char (*moves_buffer)[6];
    int moves_count;
    int *action_ids;
    
    if (current_player_turn == C_WHITE) {
      moves_buffer = ctx->white_legal_moves_buffer;
      moves_count = ctx->white_legal_moves_count;
      action_ids = ctx->white_legal_action_ids;
    } else {
      moves_buffer = ctx->black_legal_moves_buffer;
      moves_count = ctx->black_legal_moves_count;
      action_ids = ctx->black_legal_action_ids;
    }
    
    // Generate sparse representation of legal moves
    for (int i = 0; i < moves_count && num_legal_moves < MAX_LEGAL_MOVES; i++) {
      int action_id;
      
      // Use pre-computed action IDs when possible
      if (player == current_player_turn) {
        action_id = action_ids[i];
      } else {
        // Need to convert to other player's perspective
        char perspective_uci[6];
        flip_uci_for_black_perspective(moves_buffer[i], perspective_uci);
        action_id = uci_to_action_id(perspective_uci);
      }
      
      if (action_id >= 0) {
        action_ids_ptr[num_legal_moves] = (float)action_id;
        num_legal_moves++;
      }
    }
  }
  
  // Store the count of legal moves
  env->observations[obs_offset + sparse_mask_idx] = (float)num_legal_moves;
  
  // DEBUG: Verify sparse mask was written correctly
  if (env->env_id == 0 && num_legal_moves > 0) {
//     // printf("[SPARSE MASK] Stored %d legal moves at index %d\n", num_legal_moves, obs_offset + sparse_mask_idx);
//     // printf("[SPARSE MASK] First 3 action IDs: %.0f, %.0f, %.0f\n",
//            env->observations[obs_offset + sparse_mask_idx + 1],
//            env->observations[obs_offset + sparse_mask_idx + 2],
//            env->observations[obs_offset + sparse_mask_idx + 3]);
  }
  
  // DEBUG: Removed orphaned FEN printing code that was causing compilation errors
  /*
        for (int file = 0; file < 8; file++) {
          Piece* p = &board->board[rank * 8 + file];
          if (p->type == EMPTY) {
            empty_count++;
          } else {
            if (empty_count > 0) {
              // printf("%d", empty_count);
              empty_count = 0;
            }
            char piece_char;
            switch (p->type) {
              case PAWN: piece_char = 'p'; break;
              case KNIGHT: piece_char = 'n'; break;
              case BISHOP: piece_char = 'b'; break;
              case ROOK: piece_char = 'r'; break;
              case QUEEN: piece_char = 'q'; break;
              case KING: piece_char = 'k'; break;
              default: piece_char = '?'; break;
            }
            if (p->color == C_WHITE) {
              piece_char = toupper(piece_char);
            }
            // printf("%c", piece_char);
          }
        }
        if (empty_count > 0) {
          // printf("%d", empty_count);
        }
        // if (rank > 0) printf("/");
      }
      // Print remaining FEN fields
      // printf(" %c", (board->to_move == C_WHITE) ? 'w' : 'b');
      
      // Castling rights
      // printf(" ");
      if (board->castle_rights == 0) {
        // printf("-");
      } else {
        // if (board->castle_rights & 0x01) printf("K");
        // if (board->castle_rights & 0x02) printf("Q");
        // if (board->castle_rights & 0x04) printf("k");
        // if (board->castle_rights & 0x08) printf("q");
      }
      
      // En passant square
      // printf(" ");
      if (board->ep_square == -1) {
        // printf("-");
      } else {
        int ep_file = board->ep_square % 8;
        int ep_rank = board->ep_square / 8;
        // printf("%c%d", 'a' + ep_file, ep_rank + 1);
      }
      
      // Halfmove clock and fullmove number
      // printf(" %d %d\n", board->halfmove_clock, board->fullmove_number);
      
      // Check key pieces for our endgame position
      // printf("[BOARD STATE] Piece at a2: ");
      Piece* piece_a2 = &board->board[1 * 8 + 0];  // row 1, col 0 = a2
      if (piece_a2->type == EMPTY) {
        // printf("EMPTY\n");
      } else {
        const char* type_str = (piece_a2->type == ROOK) ? "Rook" : 
                               (piece_a2->type == KING) ? "King" : "Other";
        const char* color_str = (piece_a2->color == C_WHITE) ? "White" : "Black";
        // printf("%s %s\n", color_str, type_str);
      }
      
      // printf("[BOARD STATE] White King at g3: ");
      Piece* wk = &board->board[2 * 8 + 6];  // row 2, col 6 = g3
      // printf("%s\n", (wk->type == KING && wk->color == C_WHITE) ? "YES" : "NO");
      
      // printf("[BOARD STATE] Black King at h1: ");
      Piece* bk = &board->board[0 * 8 + 7];  // row 0, col 7 = h1
      // printf("%s\n", (bk->type == KING && bk->color == C_BLACK) ? "YES" : "NO");
      
      // Print all legal moves to debug
      // printf("[LEGAL MOVES LIST] %d moves:\n", moves_count);
      for (int i = 0; i < moves_count && i < 10; i++) {
        int action_id = uci_to_action_id(moves_buffer[i]);
        // printf("  Move %d: %s (action %d)\n", i, moves_buffer[i], action_id);
      }
      if (moves_count > 10) {
        // printf("  ... and %d more moves\n", moves_count - 10);
      }
      
      // Check what piece is at f7 (reuse board from above)
      Piece* f7_piece = get_piece(board, 5, 6); // f=5, 7=6 (0-indexed)
      if (f7_piece && f7_piece->type != EMPTY) {
        // printf("[BOARD CHECK] Piece at f7: %s %s\n",
               f7_piece->color == C_WHITE ? "White" : "Black",
               f7_piece->type == QUEEN ? "Queen" : 
               f7_piece->type == KING ? "King" :
               f7_piece->type == ROOK ? "Rook" :
               f7_piece->type == BISHOP ? "Bishop" :
               f7_piece->type == KNIGHT ? "Knight" :
               f7_piece->type == PAWN ? "Pawn" : "Unknown");
      } else {
        // printf("[BOARD CHECK] No piece at f7!\n");
      }
      
      // Check what piece is at f8
      Piece* f8_piece = get_piece(board, 5, 7); // f=5, 8=7 (0-indexed)
      if (f8_piece && f8_piece->type != EMPTY) {
        // printf("[BOARD CHECK] Piece at f8: %s %s\n",
               f8_piece->color == C_WHITE ? "White" : "Black",
               f8_piece->type == QUEEN ? "Queen" : 
               f8_piece->type == KING ? "King" :
               f8_piece->type == ROOK ? "Rook" :
               f8_piece->type == BISHOP ? "Bishop" :
               f8_piece->type == KNIGHT ? "Knight" :
               f8_piece->type == PAWN ? "Pawn" : "Unknown");
      } else {
        // printf("[BOARD CHECK] Square f8 is empty\n");
      }
    }
    */
  }
  
  // Old bitfield code removed - using sparse mask instead
// Corrected version of the observation generation orchestrator.
// This function should replace the old `compute_observation_with_perspective`.
// COLOR MONITORING: Validates observation data integrity at generation point
void validate_chess_observation_integrity(CChess *env, ChessContext *ctx,
                                          PieceColor player, int obs_offset) {
  // SENTINEL 1: Check observation buffer bounds
  // In the new single-agent-view architecture, the offset for the active player
  // is ALWAYS 0.
  const int expected_offset = 0; // <-- NEW, CORRECT LOGIC
  if (obs_offset != expected_offset) {
    // printf("[MONITOR_FATAL] Chess.h observation offset mismatch!\n");
    // printf(" Expected %s at offset %d, got offset %d\n",
    //        (player == C_WHITE) ? "WHITE" : "BLACK", expected_offset,
    //        obs_offset);
    // printf(" This indicates an error in the calling function.\n");
    // printf(" FIX: Ensure all calls to compute_single_agent_observation pass 0 "
    //        "as the offset.\n");
    exit(1);
  }
  // SENTINEL 2: Validate observation content signature
  float *obs = &env->observations[obs_offset];
  float board_sum = 0.0f;
  for (int i = 0; i < 1472; i++)
    board_sum += obs[i];

  // Sparse mask validation
  // Format: [num_legal_moves(1)] + [action_ids(64)]
  float num_legal_moves = obs[1472];
  float sparse_mask_sum = 0.0f;
  
  // Count number of valid action IDs (should equal num_legal_moves)
  if (num_legal_moves > 0 && num_legal_moves <= 64) {
    for (int i = 0; i < (int)num_legal_moves; i++) {
      int action_id = (int)obs[1473 + i];
      if (action_id >= 0 && action_id < 1968) {
        sparse_mask_sum += 1.0f;
      }
    }
  }
  
  int total_legal_actions = (int)num_legal_moves;

  // Check basic observation integrity
  if (board_sum < 1.0f) {
    // printf("[MONITOR_FATAL] Chess.h observation content invalid!\n");
    // printf(" %s observation at offset %d: board_sum=%.3f total_legal_actions=%d\n",
    //        (player == C_WHITE) ? "WHITE" : "BLACK", obs_offset, board_sum,
    //        total_legal_actions);
    // printf(" Board sum should be >1 (pieces present).\n");
    // printf(" FIX: Check compute_single_agent_observation() is writing correct "
    //        "data.\n");
    exit(1);
  }
  // Check for uninitialized board (real error condition)
  if (board_sum < 10.0f) {
    // printf("[CHESS_FATAL] Board sum is %.1f - board appears uninitialized!\n",
    //        board_sum);
    // printf(" This suggests the environment reset is not working properly.\n");
    exit(1);
  }

  // 0 legal moves is normal for terminal positions - no need to spam logs
  // SENTINEL 3: Validate perspective correctness
  PieceColor current_turn = ctx->board.to_move;
  // printf("[MONITOR_OK] Chess.h: Generated %s observation (offset=%d,
  // board_sum=%.1f, mask_sum=%.0f) on %s's turn\n", (player == C_WHITE) ?
  // "WHITE" : "BLACK", obs_offset, board_sum, mask_sum, (current_turn ==
  // C_WHITE) ? "WHITE" : "BLACK");
}
// In chess.h
void compute_observation_with_perspective(CChess *env, ChessContext *ctx) {
  PROFILE_START(profile_compute_obs_ticks)
  // The board state reflects the current player's turn.
  // We generate the observation for this player from their perspective.
  PieceColor current_player = ctx->board.to_move;
  // The obs_offset is always 0 because the framework expects one observation
  // per env.
  int obs_offset = 0;
  // Compute the observation for the current player and write it to the start of
  // the buffer.
  compute_single_agent_observation(env, ctx, current_player, obs_offset);
  // Optional: you could add a validation check here for the single observation
  validate_chess_observation_integrity(env, ctx, current_player, obs_offset);
  
  // DEBUG: Print full observation to verify values
//   // printf("[OBS FULL] Nonzero observation values:\n");
  int nonzero_count = 0;
  for (int i = 0; i < 1537; i++) {  // Check ALL observation values including action mask
    if (env->observations[i] != 0.0f) {
      if (nonzero_count < 70) {  // Print first 70 nonzero values
        // printf("  obs[%d] = %.1f", i, env->observations[i]);
        // Identify which plane this is
        if (i < 832) {
          int plane = i / 64;
          int square = i % 64;
          // printf(" (plane %d, square %d)", plane, square);
        }
        // printf("\n");
      }
      nonzero_count++;
    }
  }
//   // printf("[OBS FULL] Total nonzero values: %d/1537\n", nonzero_count);
  
  // Check if the buffer is actually being written to
//   // printf("[OBS BUFFER] Key indices: obs[22]=%.1f obs[136]=%.1f obs[391]=%.1f obs[1472]=%.1f\n",
//          env->observations[22], env->observations[136], env->observations[391], env->observations[1472]);
//   // printf("[OBS BUFFER] First 10 raw values: ");
//   for (int i = 0; i < 10; i++) {
//     printf("%.1f ", env->observations[i]);
//   }
//   printf("\n");
  
  PROFILE_STOP(profile_compute_obs_ticks)
}
// === MAKE/UNMAKE MOVE FUNCTIONS FOR FAST LEGALITY CHECKING ===
static void make_move_fast(ChessBoard *board, ChessMove move, UndoInfo *undo) {
  PROFILE_START(profile_make_move_fast_ticks)
  // Save current state for undo
  undo->old_castle_rights = board->castle_rights;
  undo->old_ep_square = board->ep_square;
  undo->old_halfmove_clock = board->halfmove_clock;
  undo->old_zobrist_hash = board->zobrist_hash;
  undo->was_castling = move.is_castling;
  undo->was_en_passant = move.is_en_passant;
  PieceColor moving_color = board->to_move;
  Piece *from_piece = get_piece(board, move.from.x, move.from.y);
  Piece *to_piece = get_piece(board, move.to.x, move.to.y);
  // Check for null pointers before dereferencing
  if (!from_piece || !to_piece) {
    return; // Invalid move coordinates
  }
  // Save captured piece
  undo->captured_piece = *to_piece;
  if (move.is_castling) {
    // Handle castling
    int rank = (moving_color == C_WHITE) ? 0 : 7;
    bool kingside = (move.to.x == 6);
    // Save rook positions for undo
    undo->rook_from.x = kingside ? 7 : 0;
    undo->rook_from.y = rank;
    undo->rook_to.x = kingside ? 5 : 3;
    undo->rook_to.y = rank;
    // Move king
    from_piece->type = EMPTY;
    from_piece->color = C_NO_COLOR;
    to_piece->type = KING;
    to_piece->color = moving_color;
    // Move rook
    Piece *rook_from = get_piece(board, undo->rook_from.x, undo->rook_from.y);
    Piece *rook_to = get_piece(board, undo->rook_to.x, undo->rook_to.y);
    if (!rook_from || !rook_to) {
      return; // Invalid rook coordinates
    }
    *rook_to = *rook_from;
    rook_from->type = EMPTY;
    rook_from->color = C_NO_COLOR;
    // Update castling rights
    if (moving_color == C_WHITE) {
      board->castle_rights &= ~0x3; // Clear white castling
    } else {
      board->castle_rights &= ~0xC; // Clear black castling
    }
  } else if (move.is_en_passant) {
    // Handle en passant
    int captured_y = (moving_color == C_WHITE) ? move.to.y - 1 : move.to.y + 1;
    Piece *captured_pawn = get_piece(board, move.to.x, captured_y);
    if (!captured_pawn) {
      return; // Invalid en passant coordinates
    }
    // Save the actual captured pawn position for undo
    undo->captured_piece = *captured_pawn;
    // Remove captured pawn
    captured_pawn->type = EMPTY;
    captured_pawn->color = C_NO_COLOR;
    // Move capturing pawn
    from_piece->type = EMPTY;
    from_piece->color = C_NO_COLOR;
    to_piece->type = PAWN;
    to_piece->color = moving_color;
  } else {
    // Regular move
    PieceType original_type = from_piece->type;
    PieceType final_type =
        (move.promotion != EMPTY) ? move.promotion : original_type;
    // Update castling rights if king or rook moves (before clearing piece)
    if (original_type == KING) {
      if (moving_color == C_WHITE) {
        board->castle_rights &= ~0x3; // Clear white castling
      } else {
        board->castle_rights &= ~0xC; // Clear black castling
      }
    } else if (original_type == ROOK) {
      if (moving_color == C_WHITE) {
        if (move.from.x == 0)
          board->castle_rights &= ~0x2; // White queenside
        if (move.from.x == 7)
          board->castle_rights &= ~0x1; // White kingside
      } else {
        if (move.from.x == 0)
          board->castle_rights &= ~0x8; // Black queenside
        if (move.from.x == 7)
          board->castle_rights &= ~0x4; // Black kingside
      }
    }
    // Move piece
    from_piece->type = EMPTY;
    from_piece->color = C_NO_COLOR;
    to_piece->type = final_type;
    to_piece->color = moving_color;
  }
  // Update en passant square (need to check original piece type for regular
  // moves)
  board->ep_square = -1;
  if (!move.is_castling && !move.is_en_passant) {
    PieceType original_type = (move.promotion != EMPTY) ? PAWN : to_piece->type;
    if (original_type == PAWN && abs(move.to.y - move.from.y) == 2) {
      board->ep_square = move.to.x + ((move.from.y + move.to.y) / 2) * 8;
    }
  }
  // Update halfmove clock (need to check original piece type)
  PieceType moved_piece_type = EMPTY;
  if (move.is_castling) {
    moved_piece_type = KING;
  } else if (move.is_en_passant) {
    moved_piece_type = PAWN;
  } else {
    moved_piece_type = (move.promotion != EMPTY) ? PAWN : to_piece->type;
  }
  if (moved_piece_type == PAWN || undo->captured_piece.type != EMPTY) {
    board->halfmove_clock = 0;
  } else {
    board->halfmove_clock++;
  }
  // Change side to move
  board->to_move = (moving_color == C_WHITE) ? C_BLACK : C_WHITE;
  PROFILE_STOP(profile_make_move_fast_ticks)
}
static void unmake_move_fast(ChessBoard *board, ChessMove move,
                             UndoInfo *undo) {
  PROFILE_START(profile_unmake_move_fast_ticks)
  // Restore board state from undo information
  board->castle_rights = undo->old_castle_rights;
  board->ep_square = undo->old_ep_square;
  board->halfmove_clock = undo->old_halfmove_clock;
  board->zobrist_hash = undo->old_zobrist_hash;
  // Restore side to move
  PieceColor moving_color = (board->to_move == C_WHITE) ? C_BLACK : C_WHITE;
  board->to_move = moving_color;
  Piece *from_piece = get_piece(board, move.from.x, move.from.y);
  Piece *to_piece = get_piece(board, move.to.x, move.to.y);
  // Check for null pointers before dereferencing
  if (!from_piece || !to_piece) {
    PROFILE_STOP(profile_unmake_move_fast_ticks)
    return; // Invalid move coordinates
  }
  if (undo->was_castling) {
    // Undo castling
    // Restore king
    from_piece->type = KING;
    from_piece->color = moving_color;
    to_piece->type = EMPTY;
    to_piece->color = C_NO_COLOR;
    // Restore rook
    Piece *rook_from = get_piece(board, undo->rook_from.x, undo->rook_from.y);
    Piece *rook_to = get_piece(board, undo->rook_to.x, undo->rook_to.y);
    if (!rook_from || !rook_to) {
      PROFILE_STOP(profile_unmake_move_fast_ticks)
      return; // Invalid rook coordinates
    }
    rook_from->type = ROOK;
    rook_from->color = moving_color;
    rook_to->type = EMPTY;
    rook_to->color = C_NO_COLOR;
  } else if (undo->was_en_passant) {
    // Undo en passant
    int captured_y = (moving_color == C_WHITE) ? move.to.y - 1 : move.to.y + 1;
    Piece *captured_pawn = get_piece(board, move.to.x, captured_y);
    if (!captured_pawn) {
      PROFILE_STOP(profile_unmake_move_fast_ticks)
      return; // Invalid en passant coordinates
    }
    // Restore captured pawn
    *captured_pawn = undo->captured_piece;
    // Restore moving pawn
    from_piece->type = PAWN;
    from_piece->color = moving_color;
    to_piece->type = EMPTY;
    to_piece->color = C_NO_COLOR;
  } else {
    // Undo regular move
    // Restore original piece (handle promotion)
    from_piece->type = (move.promotion != EMPTY) ? PAWN : to_piece->type;
    from_piece->color = moving_color;
    // Restore captured piece
    *to_piece = undo->captured_piece;
  }
  PROFILE_STOP(profile_unmake_move_fast_ticks)
}
// === MOVE APPLICATION ===
static bool apply_uci_move(ChessContext *ctx, const char *uci_str) {
  PROFILE_START(profile_apply_uci_move_ticks)
  if (strlen(uci_str) < 4) {
    PROFILE_STOP(profile_apply_uci_move_ticks)
    return false;
  }
  int from_x = uci_str[0] - 'a';
  int from_y = uci_str[1] - '1';
  int to_x = uci_str[2] - 'a';
  int to_y = uci_str[3] - '1';
  if (from_x < 0 || from_x >= 8 || from_y < 0 || from_y >= 8 || to_x < 0 ||
      to_x >= 8 || to_y < 0 || to_y >= 8) {
    PROFILE_STOP(profile_apply_uci_move_ticks)
    return false;
  }
  ChessBoard *board = &ctx->board;
  PieceColor us = board->to_move;
  // Store old values for Zobrist updates
  uint8_t old_castle_rights = board->castle_rights;
  int8_t old_ep_square = board->ep_square;
  // Get piece being moved
  Piece *from_piece = get_piece(board, from_x, from_y);
  if (!from_piece || from_piece->type == EMPTY || from_piece->color != us) {
    PROFILE_STOP(profile_apply_uci_move_ticks)
    return false;
  }
  Piece moving_piece = *from_piece;
  Piece *captured_piece = get_piece(board, to_x, to_y);
  bool is_capture = (captured_piece->type != EMPTY);
  Piece captured_piece_copy = *captured_piece; // Store for Zobrist update
  // XOR out old state from hash
  board->zobrist_hash ^= zobrist_side_to_move; // Change side to move
  if (old_ep_square >= 0) {
    board->zobrist_hash ^= zobrist_en_passant[old_ep_square];
  }
  board->zobrist_hash ^= zobrist_castle_rights[old_castle_rights];
  // Handle castling (special UCI format: king moves 2 squares)
  if (moving_piece.type == KING && abs(to_x - from_x) == 2) {
    int rank = (us == C_WHITE) ? 0 : 7;
    bool kingside = (to_x > from_x);
    // Validate castling rights and path
    int rook_from = kingside ? 7 : 0;
    int rook_to = kingside ? 5 : 3;
    // Check if path is clear (king's path already checked in legal move gen)
    for (int x = (kingside ? 5 : 1); x <= (kingside ? 6 : 3); x++) {
      if (x != from_x && get_piece_const(board, x, rank)->type != EMPTY) {
        PROFILE_STOP(profile_apply_uci_move_ticks)
        return false;
      }
    }
    // XOR out old piece positions
    board->zobrist_hash ^= zobrist_piece_square[us][KING][from_y * 8 + from_x];
    board->zobrist_hash ^= zobrist_piece_square[us][ROOK][rank * 8 + rook_from];
    // Move king
    from_piece->type = EMPTY;
    from_piece->color = C_NO_COLOR;
    get_piece(board, to_x, to_y)->type = KING;
    get_piece(board, to_x, to_y)->color = us;
    // Move rook
    Piece *rook_piece = get_piece(board, rook_from, rank);
    if (rook_piece->type != ROOK || rook_piece->color != us) {
      PROFILE_STOP(profile_apply_uci_move_ticks)
      return false;
    }
    rook_piece->type = EMPTY;
    rook_piece->color = C_NO_COLOR;
    get_piece(board, rook_to, rank)->type = ROOK;
    get_piece(board, rook_to, rank)->color = us;
    // XOR in new piece positions
    board->zobrist_hash ^= zobrist_piece_square[us][KING][to_y * 8 + to_x];
    board->zobrist_hash ^= zobrist_piece_square[us][ROOK][rank * 8 + rook_to];
    // Update castling rights
    if (us == C_WHITE) {
      board->castle_rights &= ~0x3; // Clear white castling
    } else {
      board->castle_rights &= ~0xC; // Clear black castling
    }
  }
  // Handle en passant capture
  else if (moving_piece.type == PAWN && (to_y * 8 + to_x) == board->ep_square) {
    // This is a confirmed en passant capture
    int captured_y = (us == C_WHITE) ? to_y - 1 : to_y + 1;
    Piece *en_passant_piece = get_piece(board, to_x, captured_y);
    // This check is now for sanity, the ep_square check is the real guard
    if (!en_passant_piece || en_passant_piece->type != PAWN ||
        en_passant_piece->color == us) {
      PROFILE_STOP(profile_apply_uci_move_ticks)
      return false;
    }
    // XOR out old pieces
    board->zobrist_hash ^= zobrist_piece_square[us][PAWN][from_y * 8 + from_x];
    board->zobrist_hash ^=
        zobrist_piece_square[1 - us][PAWN][captured_y * 8 + to_x];
    // Remove the en passant captured pawn
    en_passant_piece->type = EMPTY;
    en_passant_piece->color = C_NO_COLOR;
    // Move the capturing pawn
    from_piece->type = EMPTY;
    from_piece->color = C_NO_COLOR;
    get_piece(board, to_x, to_y)->type = PAWN;
    get_piece(board, to_x, to_y)->color = us;
    // XOR in new pawn position
    board->zobrist_hash ^= zobrist_piece_square[us][PAWN][to_y * 8 + to_x];
  }
  // Regular move
  else {
    // XOR out old piece position
    board->zobrist_hash ^=
        zobrist_piece_square[us][moving_piece.type][from_y * 8 + from_x];
    // XOR out captured piece if any
    if (is_capture) {
      board->zobrist_hash ^=
          zobrist_piece_square[captured_piece_copy.color]
                              [captured_piece_copy.type][to_y * 8 + to_x];
    }
    // Clear source square
    from_piece->type = EMPTY;
    from_piece->color = C_NO_COLOR;
    // Place piece on destination
    get_piece(board, to_x, to_y)->type = moving_piece.type;
    get_piece(board, to_x, to_y)->color = moving_piece.color;
    // Handle promotion
    PieceType final_type = moving_piece.type;
    if (strlen(uci_str) == 5 && moving_piece.type == PAWN) {
      char promo = uci_str[4];
      switch (promo) {
      case 'q':
        final_type = QUEEN;
        break;
      case 'r':
        final_type = ROOK;
        break;
      case 'b':
        final_type = BISHOP;
        break;
      case 'n':
        final_type = KNIGHT;
        break;
      default:
        PROFILE_STOP(profile_apply_uci_move_ticks)
        return false;
      }
      get_piece(board, to_x, to_y)->type = final_type;
    }
    // XOR in new piece position
    board->zobrist_hash ^=
        zobrist_piece_square[us][final_type][to_y * 8 + to_x];
  }
  // Update castling rights when king or rook moves
  if (moving_piece.type == KING) {
    if (us == C_WHITE) {
      board->castle_rights &= ~0x3; // Clear white castling
    } else {
      board->castle_rights &= ~0xC; // Clear black castling
    }
  } else if (moving_piece.type == ROOK) {
    if (us == C_WHITE) {
      if (from_x == 0)
        board->castle_rights &= ~0x2; // White queenside
      if (from_x == 7)
        board->castle_rights &= ~0x1; // White kingside
    } else {
      if (from_x == 0)
        board->castle_rights &= ~0x8; // Black queenside
      if (from_x == 7)
        board->castle_rights &= ~0x4; // Black kingside
    }
  }
  // Set en passant square
  board->ep_square = -1;
  if (moving_piece.type == PAWN && abs(to_y - from_y) == 2) {
    board->ep_square = to_x + ((from_y + to_y) / 2) * 8;
  }
  // Update halfmove clock
  if (moving_piece.type == PAWN || is_capture) {
    board->halfmove_clock = 0;
  } else {
    board->halfmove_clock++;
  }
  // Update game state
  // In puzzle mode, the turn should NOT switch. The episode ends after White's
  // move.
  if (!ctx->puzzle_mode) {
    board->to_move = (us == C_WHITE) ? C_BLACK : C_WHITE;
    if (us == C_BLACK) {
      board->fullmove_number++;
    }
  }
  // XOR in new state
  board->zobrist_hash ^= zobrist_castle_rights[board->castle_rights];
  if (board->ep_square >= 0) {
    board->zobrist_hash ^= zobrist_en_passant[board->ep_square];
  }
  // Add current position to history for threefold repetition detection
  add_position_to_history(ctx, board->zobrist_hash);
  // Clear caches
  ctx->white_moves_cached = false;
  ctx->black_moves_cached = false;
  ctx->position_fully_cached = false;
  ctx->observation_cached = false;
  // Step count is now incremented in c_step() where it belongs
  // Add to complete game log - store the canonical UCI move
  if (ctx->complete_game_action_count < 1024) {
    strcpy(ctx->complete_game_moves[ctx->complete_game_action_count], uci_str);
    ctx->complete_game_action_count++;
  } else {
  }
  PROFILE_STOP(profile_apply_uci_move_ticks)
  return true;
}
// === DRAW DETECTION FUNCTIONS ===
static void add_position_to_history(ChessContext *ctx, uint64_t hash) {
  PositionHistory *history = &ctx->position_history;
  // Look for existing hash
  for (int i = 0; i < history->size; i++) {
    if (history->hashes[i] == hash) {
      history->counts[i]++;
      return;
    }
  }
  // Add new hash if space available
  if (history->size < POSITION_HISTORY_SIZE) {
    history->hashes[history->size] = hash;
    history->counts[history->size] = 1;
    history->size++;
  }
  // If history is full, we don't track this position (rare case)
}
static int get_position_count(ChessContext *ctx, uint64_t hash) {
  PositionHistory *history = &ctx->position_history;
  for (int i = 0; i < history->size; i++) {
    if (history->hashes[i] == hash) {
      return history->counts[i];
    }
  }
  return 0;
}
static bool is_threefold_repetition(ChessContext *ctx) {
  uint64_t current_hash = ctx->board.zobrist_hash;
  return get_position_count(ctx, current_hash) >= 3;
}
static bool is_insufficient_material(ChessContext *ctx) {
  ChessBoard *board = &ctx->board;
  // Count material for both sides
  int white_pawns = 0, black_pawns = 0;
  int white_knights = 0, black_knights = 0;
  int white_bishops = 0, black_bishops = 0;
  int white_rooks = 0, black_rooks = 0;
  int white_queens = 0, black_queens = 0;
  for (int i = 0; i < 64; i++) {
    const Piece *p = &board->board[i];
    if (p->type == EMPTY)
      continue;
    switch (p->type) {
    case PAWN:
      if (p->color == C_WHITE)
        white_pawns++;
      else
        black_pawns++;
      break;
    case KNIGHT:
      if (p->color == C_WHITE)
        white_knights++;
      else
        black_knights++;
      break;
    case BISHOP:
      if (p->color == C_WHITE)
        white_bishops++;
      else
        black_bishops++;
      break;
    case ROOK:
      if (p->color == C_WHITE)
        white_rooks++;
      else
        black_rooks++;
      break;
    case QUEEN:
      if (p->color == C_WHITE)
        white_queens++;
      else
        black_queens++;
      break;
    default:
      break;
    }
  }
  // Any pawns, rooks, or queens means sufficient material
  if (white_pawns > 0 || black_pawns > 0 || white_rooks > 0 ||
      black_rooks > 0 || white_queens > 0 || black_queens > 0) {
    return false;
  }
  // Count total minor pieces for each side
  int white_minor = white_knights + white_bishops;
  int black_minor = black_knights + black_bishops;
  // Insufficient material cases:
  // King vs King
  if (white_minor == 0 && black_minor == 0)
    return true;
  // King + minor piece vs King
  if ((white_minor <= 1 && black_minor == 0) ||
      (black_minor <= 1 && white_minor == 0))
    return true;
  // King + Bishop vs King + Bishop (same color squares) - simplified to any
  // bishop vs bishop
  if (white_minor == 1 && black_minor == 1 && white_bishops == 1 &&
      black_bishops == 1)
    return true;
  return false;
}
static int global_env_counter = 0;
static int logging_env_id = -1; // ID of the designated logging environment
static int first_active_env_id =
    -1; // Track the first environment that actually runs games

// Global puzzle tracking shared across ALL environments
// Use -1 as sentinel to detect first initialization
static int global_total_puzzle_attempts = -1;
static int global_total_puzzle_solves = 0;

static void init_board(ChessBoard *board) {
  memset(board, 0, sizeof(ChessBoard));
  const char *start_fen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  int x = 0, y = 7;
  const char *p = start_fen;
  while (*p && *p != ' ') {
    if (*p == '/') {
      x = 0;
      y--;
    } else if (*p >= '1' && *p <= '8') {
      x += (*p - '0');
    } else {
      PieceColor color = (*p >= 'A' && *p <= 'Z') ? C_WHITE : C_BLACK;
      PieceType type = EMPTY;
      switch (*p | 32) {
      case 'k':
        type = KING;
        break;
      case 'q':
        type = QUEEN;
        break;
      case 'r':
        type = ROOK;
        break;
      case 'b':
        type = BISHOP;
        break;
      case 'n':
        type = KNIGHT;
        break;
      case 'p':
        type = PAWN;
        break;
      }
      if (type != EMPTY && x < 8) {
        board->board[y * 8 + x] = (Piece){color, type};
        x++;
      }
    }
    p++;
  }
  board->to_move = C_WHITE;
  board->castle_rights = 0xF; // KQkq
  board->ep_square = -1;
  board->halfmove_clock = 0;
  board->fullmove_number = 1;
  board->zobrist_hash = compute_zobrist_hash(board);
}

void init(CChess *env) {
  memset(&env->context, 0, sizeof(ChessContext));
  memset(&env->log, 0, sizeof(Log));
  // Puzzle logging will use score and perf fields

  // CRITICAL: Explicitly initialize episode-specific puzzle stats to ensure they start at 0
  env->context.puzzle_attempts_this_episode = 0.0f;
  env->context.puzzle_correct_moves_this_episode = 0.0f;
  env->context.puzzle_wrong_moves_this_episode = 0.0f;
  env->context.puzzle_solve_rate_this_episode = 0.0f;

  // Initialize puzzle tracking
  env->context.puzzle_attempts_this_env = 0;
  env->context.puzzle_solved_this_env = 0;

  // Initialize new puzzle training fields
  env->context.puzzle_tries_this_episode = 0;
  env->context.puzzle_solves_this_episode = 0;
  env->context.puzzle_tries_this_env = 0;
  env->context.puzzle_max_tries_per_env = 10; // Default
  env->context.puzzle_start_time = 0;
  env->context.puzzle_samples_to_solve = 0;
  env->context.puzzle_board_cached = false;  // No cached board yet

  // Initialize global puzzle coordination (first env sets defaults)
  if (global_env_counter == 1) { // First environment
    env->global_puzzle_id = 0;
    env->global_puzzle_attempts = 0;
    env->global_puzzle_successes = 0;
    env->global_puzzle_success_threshold = 0.9f;
    env->puzzle_max_tries_per_env = 10;
    
    // Initialize global counters only once
    if (global_total_puzzle_attempts == -1) {
      global_total_puzzle_attempts = 0;
      global_total_puzzle_solves = 0;
//       // printf("[GLOBAL INIT] Initialized global puzzle counters\n");
    }
  }
  
  // Per-env counters removed - using global counters instead
  // Set up convenience pointer to avoid repeated dereferencing
  env->ctx = &env->context;
  env->env_id = global_env_counter++; // Simple counter
  
  // Debug: Environment created (removed printf for performance)

  // Designate the first environment as the logging environment
  if (logging_env_id == -1) {
    logging_env_id = env->env_id;
  }
  init_board(&env->context.board);
  env->context.dual_agent_self_play_mode = false; // Default to single-agent mode
  // Copy reward config to context
  env->context.c_reward_valid = env->reward_valid;
  env->context.c_reward_invalid_white = env->reward_invalid_white;
  env->context.c_reward_invalid_black = env->reward_invalid_black;
  env->context.c_reward_white_captures_enemy_piece =
      env->reward_white_captures_enemy_piece;
  env->context.c_reward_black_captures_enemy_piece =
      env->reward_black_captures_enemy_piece;
  env->context.c_reward_max_depth_termination =
      env->reward_max_depth_termination;
  env->context.c_use_piece_value_capture_rewards =
      env->use_piece_value_capture_rewards;
  env->context.c_piece_value_reward_multiplier =
      env->piece_value_reward_multiplier;
  env->context.c_reward_draw = env->reward_draw;
  env->context.c_reward_win_white = env->reward_win_white;
  env->context.c_reward_win_black = env->reward_win_black;
  env->context.c_reward_loss_white = env->reward_loss_white;
  env->context.c_reward_loss_black = env->reward_loss_black;
  env->context.c_reward_check_white = env->reward_check_white;
  env->context.c_reward_check_black = env->reward_check_black;
  env->context.c_reward_material_diff_white = env->reward_material_diff_white;
  env->context.c_reward_material_diff_black = env->reward_material_diff_black;

  // Copy puzzle reward config to context
  env->context.c_reward_puzzle_solved = env->reward_puzzle_solved;
  env->context.c_reward_puzzle_failed = env->reward_puzzle_failed;
  env->context.c_reward_correct_move = env->reward_correct_move;

  // Initialize puzzle mode state
  env->context.puzzle_mode = false;
  env->context.puzzle_solution_length = 0;
  env->context.puzzle_move_index = 0;
  env->context.puzzle_completed = false;
  env->context.puzzle_failed = false;
  env->context.puzzle_set_size = 0;
  env->context.current_puzzle_idx = 0;
  
  // Initialize random seed for puzzle selection
  srand(time(NULL) ^ (uintptr_t)env);
}
// Puzzle mode functions
// Parse FEN string and set up board position
bool parse_fen(const char* fen, ChessBoard* board) {
  if (!fen || !board) return false;
  
  // Clear the board
  memset(board->board, 0, sizeof(board->board));
  
  int rank = 7; // Start from rank 8 (index 7)
  int file = 0; // Start from file a (index 0)
  int i = 0;
  
  // Parse piece placement
  while (fen[i] && fen[i] != ' ') {
    if (fen[i] == '/') {
      rank--;
      file = 0;
    } else if (fen[i] >= '1' && fen[i] <= '8') {
      // Empty squares
      file += (fen[i] - '0');
    } else {
      // Piece
      int square_idx = rank * 8 + file;
      Piece* p = &board->board[square_idx];
      p->color = isupper(fen[i]) ? C_WHITE : C_BLACK;
      
      char piece_char = tolower(fen[i]);
      switch (piece_char) {
        case 'p': p->type = PAWN; break;
        case 'n': p->type = KNIGHT; break;
        case 'b': p->type = BISHOP; break;
        case 'r': p->type = ROOK; break;
        case 'q': p->type = QUEEN; break;
        case 'k': p->type = KING; break;
        default: return false;
      }
      
      file++;
    }
    i++;
  }
  
  // Skip space
  if (fen[i] == ' ') i++;
  
  // Parse side to move
  if (fen[i] == 'w') {
    board->to_move = C_WHITE;
  } else if (fen[i] == 'b') {
    board->to_move = C_BLACK;
  } else {
    return false;
  }
  i++;
  
  // Skip space
  if (fen[i] == ' ') i++;
  
  // Parse castling rights
  board->castle_rights = 0;  // Clear all castling rights
  
  if (fen[i] == '-') {
    i++;
  } else {
    while (fen[i] && fen[i] != ' ') {
      switch (fen[i]) {
        case 'K': board->castle_rights |= 0x01; break;  // White kingside
        case 'Q': board->castle_rights |= 0x02; break;  // White queenside  
        case 'k': board->castle_rights |= 0x04; break;  // Black kingside
        case 'q': board->castle_rights |= 0x08; break;  // Black queenside
      }
      i++;
    }
  }
  
  // Skip remaining fields (en passant, halfmove, fullmove) for now
  // as they're not critical for puzzle positions
  
  return true;
}

void set_puzzle_mode(CChess *env, bool enabled) {
  env->context.puzzle_mode = enabled;
}
void set_puzzle_data(CChess *env, const char *fen, const char *solution_moves[],
                     int solution_length) {
  if (!env->context.puzzle_mode)
    return;

  // // HARDCODED PUZZLE FOR FOCUSED TRAINING - Simple endgame position
  // const char* DEBUG_FEN = "8/8/8/8/8/6K1/R7/7k w - - 0 1";
  // const char* DEBUG_SOLUTION = "a2a1";  // Checkmate move
  
  // Reset tries counter when loading new puzzle
  env->context.puzzle_tries_this_env = 0;

  // Store puzzle FEN - USE THE ACTUAL FEN PASSED AS PARAMETER
  strncpy(env->context.puzzle_fen, fen, sizeof(env->context.puzzle_fen) - 1);
  env->context.puzzle_fen[sizeof(env->context.puzzle_fen) - 1] = '\0';

  // Store solution moves - USE THE ACTUAL SOLUTION PASSED AS PARAMETER
  env->context.puzzle_solution_length = solution_length;
  for (int i = 0; i < solution_length && i < 10; i++) {
    strncpy(env->context.puzzle_solution[i], solution_moves[i], 5);
    env->context.puzzle_solution[i][5] = '\0';
    // Debug: Print what's being loaded
    if (env->env_id == 0 && i == 0) {
      printf("[PUZZLE LOAD] set_puzzle_data storing solution move %d: '%s' (from param: '%s')\n", 
             i, env->context.puzzle_solution[i], solution_moves[i]);
    }
  }

  // Reset puzzle state
  env->context.puzzle_move_index = 0;
  env->context.puzzle_completed = false;
  env->context.puzzle_failed = false;

  // Load the FEN position and cache it for quick reset
  if (!parse_fen(fen, &env->context.board)) {
//     printf("[ERROR] Failed to parse puzzle FEN: %s\n", fen);
    init_board(&env->context.board);  // Fallback to starting position
  }
  
  // Cache the puzzle board state for quick reset
  env->context.puzzle_board_cache = env->context.board;  // Deep copy the board
  env->context.puzzle_board_cached = true;
  
  // Cache the solution action ID for BC training
  env->context.solution_action_cached = false;
  if (solution_length > 0) {
    const char* first_move = solution_moves[0];
    int action_id = uci_to_action_id(first_move);
    
    // If playing as black, flip the perspective
    if (env->context.board.to_move == C_BLACK) {
      char flipped_uci[6];
      flip_uci_for_black_perspective(first_move, flipped_uci);
      action_id = uci_to_action_id(flipped_uci);
    }
    
    env->context.cached_solution_action = action_id;
    env->context.solution_action_cached = true;
    
    if (env->env_id == 0) {
      printf("[PUZZLE CACHE] Cached solution action ID: %d for move: %s\n", action_id, first_move);
    }
  }
  
  // DEBUG: Verify cache is correct
//   printf("[CACHE DEBUG] Created puzzle cache with piece at a2: %s\n",
//          (env->context.puzzle_board_cache.board[8].type == ROOK && 
//           env->context.puzzle_board_cache.board[8].color == C_WHITE) ? "WHITE ROOK" : "EMPTY/OTHER");

  // Print initial puzzle position
  // Debug: Puzzle loaded (removed prints for performance)

  // Verify this is a mate-in-1 puzzle (white to move, 1 move solution)
  if (solution_length != 1) {
    // Warning: Not mate-in-1
//     // printf("[PUZZLE WARNING] Solution has %d moves, expected 1\n",
//            solution_length);
  }
  if (env->context.board.to_move != C_WHITE) {
    // Error: Black to move
//     // printf("[PUZZLE ERROR] Black to move - only white should "
//            "move in puzzles.\n");
  }

  // Debug: Initial position (removed print for performance)
}
void set_puzzle_difficulty(CChess *env, int difficulty) {
  // Puzzle difficulty tracking removed - using simplified Log struct
  (void)env;  // Suppress unused parameter warning
  (void)difficulty;  // Suppress unused parameter warning
}

void set_puzzle_set(CChess *env, int num_puzzles, const char **fens,
const char ***solution_moves, const int *solution_lengths) {
if (!env->context.puzzle_mode)
return;
// Use the actual puzzles passed as parameters
env->context.puzzle_set_size = num_puzzles;
// Store the puzzles from parameters
for (int i = 0; i < num_puzzles && i < MAX_PUZZLE_SET_SIZE; i++) {
strncpy(env->context.puzzle_set_fens[i], fens[i], 127);
env->context.puzzle_set_fens[i][127] = '\0';
// Store solution moves
env->context.puzzle_set_solution_lengths[i] = solution_lengths[i];
for (int j = 0; j < solution_lengths[i] && j < MAX_PUZZLE_MOVES; j++) {
strncpy(env->context.puzzle_set_solutions[i][j], solution_moves[i][j], 5);
env->context.puzzle_set_solutions[i][j][5] = '\0';
// Debug: Print what's being stored
if (env->env_id == 0 && i == 0 && j == 0) {
  printf("[PUZZLE STORE] Storing puzzle %d solution move %d: %s\n", i, j, solution_moves[i][j]);
}
}
}
// No initial load here - c_reset will select and load randomly
}
void set_puzzle_training_params(CChess *env, int max_tries_per_env,
float success_threshold) {
env->puzzle_max_tries_per_env = max_tries_per_env;
env->global_puzzle_success_threshold = success_threshold;
env->context.puzzle_max_tries_per_env = max_tries_per_env;
}



// void set_puzzle_set(CChess *env, int num_puzzles, const char **fens, 
//                     const char ***solution_moves, const int *solution_lengths) {
//   if (!env->context.puzzle_mode)
//     return;
    
//   // TEMPORARY HARDCODE FOR DEBUGGING - Simple endgame position
//   const char* DEBUG_FEN = "8/8/8/8/8/6K1/R7/7k w - - 0 1";
//   const char* DEBUG_SOLUTION = "a2a1";  // Checkmate move
  
// //   // // printf("[HARDCODED SET] Overriding ALL puzzles with DEBUG FEN\n");
// //   // printf("[HARDCODED SET] FEN: %s\n", DEBUG_FEN);
// //   // printf("[HARDCODED SET] Expected move: %s\n", DEBUG_SOLUTION);
  
//   // Override to just 1 puzzle
//   env->context.puzzle_set_size = 1;
  
//   // Store the hardcoded puzzle
//   strncpy(env->context.puzzle_set_fens[0], DEBUG_FEN, 127);
//   env->context.puzzle_set_fens[0][127] = '\0';
  
//   // Store solution
//   env->context.puzzle_set_solution_lengths[0] = 1;
//   strncpy(env->context.puzzle_set_solutions[0][0], DEBUG_SOLUTION, 5);
//   env->context.puzzle_set_solutions[0][0][5] = '\0';
  
//   // Initialize with first puzzle
//   env->context.current_puzzle_idx = 0;
  
//   // Disabled for training - prints for every env and clutters output
//   // // printf("[PUZZLE] Loaded set of %d puzzles\n", env->context.puzzle_set_size);
  
//   // Load the first puzzle
//   if (env->context.puzzle_set_size > 0) {
//     // Create temporary array of solution pointers
//     const char* first_solution[10];
//     for (int i = 0; i < env->context.puzzle_set_solution_lengths[0] && i < 10; i++) {
//       first_solution[i] = env->context.puzzle_set_solutions[0][i];
//     }
//     set_puzzle_data(env, env->context.puzzle_set_fens[0], 
//                     first_solution, 
//                     env->context.puzzle_set_solution_lengths[0]);
//   }
// }

// void set_puzzle_training_params(CChess *env, int max_tries_per_env,
//                                 float success_threshold) {
// //   // printf("[PUZZLE PARAMS] Setting max_tries_per_env = %d, success_threshold = %.2f\n", 
// //          max_tries_per_env, success_threshold);
//   env->puzzle_max_tries_per_env = max_tries_per_env;
//   env->global_puzzle_success_threshold = success_threshold;
//   env->context.puzzle_max_tries_per_env = max_tries_per_env;
// }



void allocate(CChess *env) {
  // Allocate RL interface arrays for PufferLib
  // Chess has 2 players but typically trains as single agent with perspective
  // flipping
  const int num_players = 2;
  const int obs_size =
      1537; // 23*8*8 board planes + sparse action mask = 1472 + 1 + 64
  env->observations = (float *)calloc(num_players * obs_size, sizeof(float));
  env->actions = (int *)calloc(num_players, sizeof(int));
  env->rewards = (float *)calloc(num_players, sizeof(float));
  env->terminals = (unsigned char *)calloc(num_players, sizeof(unsigned char));
  init(env);
}
void free_allocated(CChess *env) {
  // Free RL interface arrays allocated by allocate()
  free(env->observations);
  free(env->actions);
  free(env->rewards);
  free(env->terminals);
  c_close(env);
}



// void c_reset(CChess *env) {
//   // DEBUG: Print when reset is called
//   // Debug: c_reset called (removed printf for performance)

//   // Preserve puzzle stats across resets
//   int saved_puzzle_attempts = env->context.puzzle_attempts_this_env;
//   int saved_puzzle_solved = env->context.puzzle_solved_this_env;

//   // Only init board if no FEN was set
//   // In puzzle mode, use cached board if available
//   if (env->context.puzzle_mode && env->context.puzzle_board_cached) {
//     // Restore the cached puzzle board state
//     env->context.board = env->context.puzzle_board_cache;
//     // printf("[C_RESET] Restored cached puzzle board\n");
    
//     // Clear move caches and regenerate legal moves
//     env->context.white_moves_cached = false;
//     env->context.black_moves_cached = false;
//     env->context.position_fully_cached = false;
//     env->context.observation_cached = false;
    
//     // Generate legal moves for the current position
//     int num_moves = chess_generate_legal_moves_uci(&env->context);
//     // printf("[C_RESET] Generated %d legal moves after board restore\n", num_moves);
    
//     // CRITICAL FIX: Must recalculate observation after restoring puzzle board
//     // The observation cache must be invalidated when board state changes
//     env->context.observation_cached = false;
    
//   } else if (env->context.puzzle_mode && env->context.puzzle_set_size > 0 && strlen(env->context.puzzle_fen) > 0) {
//     // Parse the puzzle FEN if no cache available
//     if (!parse_fen(env->context.puzzle_fen, &env->context.board)) {
//       // printf("[C_RESET ERROR] Failed to parse puzzle FEN: %s\n", env->context.puzzle_fen);
//       init_board(&env->context.board);
//     }
//     env->context.board.fen_was_set = false; // Reset flag so next reset works
//   } else if (!env->context.board.fen_was_set) {
//     init_board(&env->context.board);
//   } else {
//     env->context.board.fen_was_set = false; // Reset flag after use
//   }
//   // Reset terminals and rewards for both agents
//   env->terminals[0] = 0;
//   env->terminals[1] = 0;
//   env->rewards[0] = 0.0f;
//   env->rewards[1] = 0.0f;
//   // Reset episode tracking
//   env->context.step_count = 0;
//   env->context.episode_return_white = 0.0f;
  
//   // Reset puzzle episode tracking
//   // DISABLED FOR PERFORMANCE - printf in hot path kills training speed!
//   // printf("[DEBUG RESET] Before reset: solves=%d, tries=%d\n", 
//   //        env->context.puzzle_solves_this_episode,
//   //        env->context.puzzle_tries_this_episode);
//   env->context.puzzle_tries_this_episode = 0;
//   env->context.puzzle_solves_this_episode = 0;
//   // printf("[DEBUG RESET] After reset: solves=%d, tries=%d\n", 
//   //        env->context.puzzle_solves_this_episode,
//   //        env->context.puzzle_tries_this_episode);
  
//   // DISABLED FOR PERFORMANCE - printf in hot path kills training speed!
//   // printf("[EPISODE RESET] Starting new episode (step_count reset to 0)\n");
//   env->context.episode_return_black = 0.0f;

//   env->context.complete_game_action_count = 0;
//   env->context.serialized_moves[0] =
//       '\0'; // Initialize serialized_moves buffer to empty
//   env->context.steps_since_last_log = 0;

//   // Reset puzzle state for new episode
//   if (env->context.puzzle_mode) {
//     env->context.puzzle_completed = false;
//     env->context.puzzle_failed = false;
//     env->context.puzzle_move_index = 0;
//     // Reset episode-specific puzzle stats
//     // Debug: Resetting puzzle stats (removed printf for performance)
//     env->context.puzzle_attempts_this_episode = 0.0f;
//     env->context.puzzle_correct_moves_this_episode = 0.0f;
//     env->context.puzzle_wrong_moves_this_episode = 0.0f;
//     env->context.puzzle_solve_rate_this_episode = 0.0f;
//   }

//   // Don't reset game logging frequency - it's set once at init
//   // env->context.game_logging_frequency = 500000;
//   // DEBUG: Explicitly preserve the logging frequency that was set during init
//   if (env->context.game_logging_frequency == 0) {
//     // This shouldn't happen - frequency should be preserved from init
//   }
//   // Reset statistics
//   env->context.c_white_moves = 0;
//   env->context.c_black_moves = 0;
//   env->context.c_valid_moves = 0;
//   env->context.c_invalid_moves_white = 0;
//   env->context.c_invalid_moves_black = 0;
//   // Reset game outcome counters (CRITICAL BUG FIX)
//   env->context.c_white_win = 0;
//   env->context.c_black_win = 0;
//   env->context.c_white_loss = 0;
//   env->context.c_black_loss = 0;
//   env->context.c_game_drawn = 0;
//   env->context.c_max_depth = 0;
//   // Game end condition counters
//   env->context.c_white_checkmated = 0;
//   env->context.c_black_checkmated = 0;
//   env->context.c_stalemate = 0;
//   env->context.c_insufficient_material = 0;
//   env->context.c_threefold_repetition = 0;
//   env->context.c_fifty_move_rule = 0;
//   // Reset accumulated reward counters
//   env->context.accumulated_reward_valid = 0.0f;
//   env->context.accumulated_reward_white_captures_enemy_piece = 0.0f;
//   env->context.accumulated_reward_black_captures_enemy_piece = 0.0f;
//   env->context.accumulated_reward_draw = 0.0f;
//   env->context.accumulated_reward_win_white = 0.0f;
//   env->context.accumulated_reward_win_black = 0.0f;
//   env->context.accumulated_reward_loss_white = 0.0f;
//   env->context.accumulated_reward_loss_black = 0.0f;
//   env->context.accumulated_reward_draw_white = 0.0f;
//   env->context.accumulated_reward_draw_black = 0.0f;
//   env->context.accumulated_reward_check_white = 0.0f;
//   env->context.accumulated_reward_check_black = 0.0f;
//   env->context.accumulated_reward_material_diff_white = 0.0f;
//   env->context.accumulated_reward_material_diff_black = 0.0f;
//   env->context.accumulated_stockfish_eval = 0.0f;

//   // Reset puzzle reward accumulation
//   env->context.accumulated_reward_puzzle_solved = 0.0f;
//   env->context.accumulated_reward_puzzle_failed = 0.0f;
//   env->context.accumulated_reward_puzzle_correct_move = 0.0f;

//   // Reset puzzle stats accumulation
//   // Debug: Reset stats before (removed printf for performance)
//   env->context.puzzle_attempts_this_episode = 0.0f;
//   env->context.puzzle_correct_moves_this_episode = 0.0f;
//   env->context.puzzle_wrong_moves_this_episode = 0.0f;
//   env->context.puzzle_solve_rate_this_episode = 0.0f;
//   // Debug: Reset stats after (removed printf for performance)
//   // Clear caches
//   env->context.white_moves_cached = false;
//   env->context.black_moves_cached = false;
//   env->context.position_fully_cached = false;
//   env->context.cached_board_hash = 0;
//   env->context.observation_cached = false;
//   // // Clear env->log rewards accumulation
//   // env->log.reward_puzzle_solved = 0.0f;
//   // env->log.reward_puzzle_failed = 0.0f;
//   // env->log.reward_puzzle_correct_move = 0.0f;
//   // env->log.episode_return_white = 0.0f;
//   // env->log.episode_return_black = 0.0f;
//   // env->log.episode_return = 0.0f;
//   // Clear position history
//   memset(&env->context.position_history, 0, sizeof(PositionHistory));
//   // Add starting position to history for threefold repetition detection
//   add_position_to_history(&env->context, env->context.board.zobrist_hash);
//   // CRITICAL: Generate legal moves before computing observation!
//   chess_generate_all_legal_moves(&env->context);
//   // Compute initial observation
// //   // printf("[OBS COMPUTE] Line 7525: c_reset initial observation\n");
//   compute_observation_with_perspective(env, &env->context);
  
//   // DEBUG: Check observation right after reset
// //   // printf("[RESET OBS CHECK] After reset, obs[22]=%.1f obs[136]=%.1f obs[391]=%.1f\n",
// //          env->observations[22], env->observations[136], env->observations[391]);

//   // Restore puzzle stats after reset
//   env->context.puzzle_attempts_this_env = saved_puzzle_attempts;
//   env->context.puzzle_solved_this_env = saved_puzzle_solved;
// }


void c_reset(CChess *env) {
// DEBUG: Print when reset is called
// Debug: c_reset called (removed printf for performance)
// Preserve puzzle stats across resets
int saved_puzzle_attempts = env->context.puzzle_attempts_this_env;
int saved_puzzle_solved = env->context.puzzle_solved_this_env;
// Only init board if no FEN was set
// In puzzle mode, use cached board if available
if (env->context.puzzle_mode && env->context.puzzle_board_cached) {
// Restore the cached puzzle board state
env->context.board = env->context.puzzle_board_cache;
// printf("[C_RESET] Restored cached puzzle board\n");
// Clear move caches and regenerate legal moves
env->context.white_moves_cached = false;
env->context.black_moves_cached = false;
env->context.position_fully_cached = false;
env->context.observation_cached = false;
// Generate legal moves for the current position
int num_moves = chess_generate_legal_moves_uci(&env->context);
// printf("[C_RESET] Generated %d legal moves after board restore\n", num_moves);
// CRITICAL FIX: Must recalculate observation after restoring puzzle board
// The observation cache must be invalidated when board state changes
env->context.observation_cached = false;
// In puzzle mode with set, select puzzle
if (env->context.puzzle_mode && env->context.puzzle_set_size > 0) {
// For single puzzle training (puzzle_set_size=1), always use puzzle 0
if (env->context.puzzle_set_size == 1) {
env->context.current_puzzle_idx = 0;
} else {
env->context.current_puzzle_idx = rand() % env->context.puzzle_set_size;
}
const char* selected_fen = env->context.puzzle_set_fens[env->context.current_puzzle_idx];
// Create temporary array of solution pointers
const char* solution_moves[10];
int solution_length = env->context.puzzle_set_solution_lengths[env->context.current_puzzle_idx];
for (int i = 0; i < solution_length && i < 10; i++) {
solution_moves[i] = env->context.puzzle_set_solutions[env->context.current_puzzle_idx][i];
}
// Load the selected puzzle
set_puzzle_data(env, selected_fen, solution_moves, solution_length);
}
} else if (env->context.puzzle_mode && env->context.puzzle_set_size > 0 && strlen(env->context.puzzle_fen) > 0) {
// Parse the puzzle FEN if no cache available
if (!parse_fen(env->context.puzzle_fen, &env->context.board)) {
// printf("[C_RESET ERROR] Failed to parse puzzle FEN: %s\n", env->context.puzzle_fen);
init_board(&env->context.board);
}
env->context.board.fen_was_set = false; // Reset flag so next reset works
} else if (!env->context.board.fen_was_set) {
init_board(&env->context.board);
} else {
env->context.board.fen_was_set = false; // Reset flag after use
}
// Reset terminals and rewards for both agents
env->terminals[0] = 0;
env->terminals[1] = 0;
env->rewards[0] = 0.0f;
env->rewards[1] = 0.0f;
// Reset episode tracking
env->context.step_count = 0;
env->context.episode_return_white = 0.0f;
// Reset puzzle episode tracking
// DISABLED FOR PERFORMANCE - printf in hot path kills training speed!
// printf("[DEBUG RESET] Before reset: solves=%d, tries=%d\n",
//        env->context.puzzle_solves_this_episode,
//        env->context.puzzle_tries_this_episode);
env->context.puzzle_tries_this_episode = 0;
env->context.puzzle_solves_this_episode = 0;
// printf("[DEBUG RESET] After reset: solves=%d, tries=%d\n",
//        env->context.puzzle_solves_this_episode,
//        env->context.puzzle_tries_this_episode);
// DISABLED FOR PERFORMANCE - printf in hot path kills training speed!
// printf("[EPISODE RESET] Starting new episode (step_count reset to 0)\n");
env->context.episode_return_black = 0.0f;
env->context.complete_game_action_count = 0;
env->context.serialized_moves[0] =
'\0'; // Initialize serialized_moves buffer to empty
env->context.steps_since_last_log = 0;
// Reset puzzle state for new episode
if (env->context.puzzle_mode) {
env->context.puzzle_completed = false;
env->context.puzzle_failed = false;
env->context.puzzle_move_index = 0;
// Reset episode-specific puzzle stats
// Debug: Resetting puzzle stats (removed printf for performance)
env->context.puzzle_attempts_this_episode = 0.0f;
env->context.puzzle_correct_moves_this_episode = 0.0f;
env->context.puzzle_wrong_moves_this_episode = 0.0f;
env->context.puzzle_solve_rate_this_episode = 0.0f;
}
// Don't reset game logging frequency - it's set once at init
// env->context.game_logging_frequency = 500000;
// DEBUG: Explicitly preserve the logging frequency that was set during init
if (env->context.game_logging_frequency == 0) {
// This shouldn't happen - frequency should be preserved from init
}
// Reset statistics
env->context.c_white_moves = 0;
env->context.c_black_moves = 0;
env->context.c_valid_moves = 0;
env->context.c_invalid_moves_white = 0;
env->context.c_invalid_moves_black = 0;
// Reset game outcome counters (CRITICAL BUG FIX)
env->context.c_white_win = 0;
env->context.c_black_win = 0;
env->context.c_white_loss = 0;
env->context.c_black_loss = 0;
env->context.c_game_drawn = 0;
env->context.c_max_depth = 0;
// Game end condition counters
env->context.c_white_checkmated = 0;
env->context.c_black_checkmated = 0;
env->context.c_stalemate = 0;
env->context.c_insufficient_material = 0;
env->context.c_threefold_repetition = 0;
env->context.c_fifty_move_rule = 0;
// Reset accumulated reward counters
env->context.accumulated_reward_valid = 0.0f;
env->context.accumulated_reward_white_captures_enemy_piece = 0.0f;
env->context.accumulated_reward_black_captures_enemy_piece = 0.0f;
env->context.accumulated_reward_draw = 0.0f;
env->context.accumulated_reward_win_white = 0.0f;
env->context.accumulated_reward_win_black = 0.0f;
env->context.accumulated_reward_loss_white = 0.0f;
env->context.accumulated_reward_loss_black = 0.0f;
env->context.accumulated_reward_draw_white = 0.0f;
env->context.accumulated_reward_draw_black = 0.0f;
env->context.accumulated_reward_check_white = 0.0f;
env->context.accumulated_reward_check_black = 0.0f;
env->context.accumulated_reward_material_diff_white = 0.0f;
env->context.accumulated_reward_material_diff_black = 0.0f;
env->context.accumulated_stockfish_eval = 0.0f;
// Reset puzzle reward accumulation
env->context.accumulated_reward_puzzle_solved = 0.0f;
env->context.accumulated_reward_puzzle_failed = 0.0f;
env->context.accumulated_reward_puzzle_correct_move = 0.0f;
// Reset puzzle stats accumulation
// Debug: Reset stats before (removed printf for performance)
env->context.puzzle_attempts_this_episode = 0.0f;
env->context.puzzle_correct_moves_this_episode = 0.0f;
env->context.puzzle_wrong_moves_this_episode = 0.0f;
env->context.puzzle_solve_rate_this_episode = 0.0f;
// Debug: Reset stats after (removed printf for performance)
// Clear caches
env->context.white_moves_cached = false;
env->context.black_moves_cached = false;
env->context.position_fully_cached = false;
env->context.cached_board_hash = 0;
env->context.observation_cached = false;
// // Clear env->log rewards accumulation
// env->log.reward_puzzle_solved = 0.0f;
// env->log.reward_puzzle_failed = 0.0f;
// env->log.reward_puzzle_correct_move = 0.0f;
// env->log.episode_return_white = 0.0f;
// env->log.episode_return_black = 0.0f;
// env->log.episode_return = 0.0f;
// Clear position history
memset(&env->context.position_history, 0, sizeof(PositionHistory));
// Add starting position to history for threefold repetition detection
add_position_to_history(&env->context, env->context.board.zobrist_hash);
// CRITICAL: Generate legal moves before computing observation!
chess_generate_all_legal_moves(&env->context);
// Compute initial observation
//   // printf("[OBS COMPUTE] Line 7525: c_reset initial observation\n");
compute_observation_with_perspective(env, &env->context);
// DEBUG: Check observation right after reset
//   // printf("[RESET OBS CHECK] After reset, obs[22]=%.1f obs[136]=%.1f obs[391]=%.1f\n",
//          env->observations[22], env->observations[136], env->observations[391]);
// Restore puzzle stats after reset
env->context.puzzle_attempts_this_env = saved_puzzle_attempts;
env->context.puzzle_solved_this_env = saved_puzzle_solved;
}







void c_step(CChess *env) {
  static int step_count = 0;
  static int debug_step_count = 0;  // For debug prints
  step_count++;
  
  // Debug: Log actions being processed in puzzle mode
  if (env->context.puzzle_mode && debug_step_count++ % 10 == 0) {
    printf("[C++ STEP] tick=%d, Processing action=%d for puzzle\n", step_count, env->actions[0]);
  }
  
//   printf("\n[C_STEP %d START] terminals[0]=%d, puzzle_failed=%d, puzzle_completed=%d\n", 
//          step_count, env->terminals[0], env->context.puzzle_failed, env->context.puzzle_completed);
//   printf("  Puzzle FEN: %.50s...\n", env->context.puzzle_fen ? env->context.puzzle_fen : "NULL");
  
  // Show full board position 
//   printf("  Board (8-1):\n");
  for (int y = 7; y >= 0; y--) {
    // printf("    %d: ", y+1);
    for (int x = 0; x < 8; x++) {
      Piece* p = get_piece(&env->context.board, x, y);
      if (p && p->type != EMPTY) {
        // Map piece types: KING=1, QUEEN=2, ROOK=3, BISHOP=4, KNIGHT=5, PAWN=6
        char pc = '.';
        switch(p->type) {
          case 1: pc = 'K'; break; // KING
          case 2: pc = 'Q'; break; // QUEEN
          case 3: pc = 'R'; break; // ROOK
          case 4: pc = 'B'; break; // BISHOP
          case 5: pc = 'N'; break; // KNIGHT
          case 6: pc = 'P'; break; // PAWN
        }
        if (p->color == C_BLACK) pc = pc + 32; // lowercase  
        // printf("%c", pc);
      } else {
        // printf(".");
      }
    }
    // printf("\n");
  }
//   printf("       abcdefgh\n");
  
  // CRITICAL FIX: Check if episode already terminated
  // If terminals is set, reset for next episode
  if (env->terminals && (env->terminals[0] || env->terminals[1])) {
    // printf("[C_STEP %d] Episode terminated (terminals[0]=%d), resetting for next episode\n", 
    //        step_count, env->terminals[0]);
    c_reset(env);  // Reset the environment for the next episode
    // After reset, terminals should be 0, so we can continue with the new episode
  }
  
  // Clear rewards from previous step (standard pattern in ocean envs)
  // Only clear if episode is NOT terminated
  env->rewards[0] = 0.0f;
  env->rewards[1] = 0.0f;
  
  // Only increment if not terminated
  env->context.step_count++;
  
  // Log this env's stats every 50 steps
  if (env->context.step_count % 1 == 0) {
    float solve_rate = 0.0f;
    if (env->context.puzzle_tries_this_episode > 0) {
      solve_rate = 100.0f * env->context.puzzle_solves_this_episode / 
                   env->context.puzzle_tries_this_episode;
    }

    // printf("[ENV %3d] step=%4d | solve_rate=%5.1f%% | tries=%3d | solves=%3d\n",
    //        env->env_id, env->context.step_count, solve_rate,
    //        env->context.puzzle_tries_this_episode,
    //        env->context.puzzle_solves_this_episode);
  }
  
  PROFILE_START(profile_c_step_ticks)
  // Handle puzzle reset AFTER showing failed state
  // If puzzle_failed is true, it means we showed the wrong move last step
  // Now we need to reset the puzzle for another attempt (unless we already terminated)
  if (env->context.puzzle_mode && env->context.puzzle_failed && !env->terminals[0]) {
    // Reset the puzzle to starting position for another try
    // printf("[PUZZLE] Resetting puzzle after failed attempt\n");
    c_set_fen(env, env->context.puzzle_fen);
    env->context.puzzle_move_index = 0;
    env->context.puzzle_failed = false;
    
    // Generate legal moves for reset position
    chess_generate_all_legal_moves(&env->context);
    compute_observation_with_perspective(env, &env->context);
    return; // Skip processing action this step - just show reset state
  }
  
  // Handle completed puzzles - remain inert until reset
  if (env->context.puzzle_mode && env->context.puzzle_completed) {
    env->terminals[0] = 1;
    env->terminals[1] = 1;
    return;
  }
  // Game logging counter is incremented when games end, not on every step
  // // In self-play mode: agent 0 = white, agent 1 = black
  // // Get action from the agent whose turn it is
  // PieceColor current_player = env->context.board.to_move;
  // int agent_idx = (current_player == C_WHITE) ? 0 : 1;
  // printf("[C_STEP_ENTRY] Turn: %s (agent %d), fullmove: %d, halfmove: %d\n",
  // (current_player == C_WHITE) ? "WHITE" : "BLACK", agent_idx,
  // env->context.board.fullmove_number, env->context.board.halfmove_clock);
  // // CRITICAL: In dual-agent mode, only process actions for the current
  // player
  // // SAFEGUARD: Handle training loop calling wrong agent
  // int correct_agent_idx = agent_idx;
  // if (env->context.dual_agent_self_play_mode) {
  // // Determine which agent should actually move based on board state
  // int expected_agent = (current_player == C_WHITE) ? 0 : 1;
  // if (agent_idx != expected_agent) {
  // printf("[TRAINING_LOOP_FIX] Board says %s's turn (agent %d), but called
  // with agent %d. Using correct agent.\n", (current_player == C_WHITE) ?
  // "WHITE" : "BLACK", expected_agent, agent_idx); correct_agent_idx =
  // expected_agent;
  // }
  // }
  // // Use the action from the correct agent (the one whose turn it actually
  // is) int action_idx = env->actions[correct_agent_idx]; In episode-per-color
  // architecture, Python wrapper handles agent assignment and episode
  // separation. C++ always uses agent_idx = 0 for the active player during
  // their episode (WHITE episode or BLACK episode)
  int agent_idx = 0;
  // The action is now correctly read from env->actions[0] for BOTH White and
  // Black.
  int action_idx = env->actions[agent_idx];
  
  // Debug: Print the action being processed and what it maps to
  if (env->context.puzzle_mode && action_idx >= 0 && action_idx < TOTAL_CHESS_ACTIONS) {
    const char *uci_move = ACTION_ID_TO_UCI[action_idx];
    printf("[ACTION DEBUG] Processing action=%d which maps to UCI move=%s\n", action_idx, uci_move);
    if (env->context.puzzle_solution_length > 0) {
      printf("[ACTION DEBUG] Expected solution move: %s (index %d/%d)\n", 
             env->context.puzzle_solution[env->context.puzzle_move_index],
             env->context.puzzle_move_index, env->context.puzzle_solution_length);
    }
  }
  
  // Generate moves for BOTH players if not cached
  // This ensures we always have the right moves available
  chess_generate_all_legal_moves(&env->context);
  
//   // Print who's turn it is and legal moves
//   printf("[TURN] %s to move\n", env->context.board.to_move == C_WHITE ? "WHITE" : "BLACK");
//   printf("[LEGAL MOVES] White: %.0f, Black: %.0f\n", 
//          env->context.c_white_moves, env->context.c_black_moves);
  // Calculate the correct observation offset for the CURRENT player.
  const int single_obs_size = 1537; // 1472 board + sparse mask (1 + 64)
  int obs_offset = 0;               // <-- NEW, CORRECT LOGIC
  int mask_start_idx = 1472;        // Start of action mask in observation
  // Check if the chosen action corresponds to a legal move
  bool action_is_legal = false;
  // Get the UCI string for the action chosen by the policy.
  // ACTION_ID_TO_UCI always represents moves in white's perspective coordinate
  // system.
  const char *uci_move_white_perspective = ACTION_ID_TO_UCI[action_idx];
  char uci_move_canonical[6];
  if (env->context.board.to_move == C_BLACK) {
    // The black agent chose this action based on its flipped perspective.
    // The action maps to a move in white perspective coordinates, but since
    // the black agent sees the board flipped, this move should be interpreted
    // as being from black's perspective and flipped to canonical coordinates.
    flip_uci_for_black_perspective(uci_move_white_perspective,
                                   uci_move_canonical);
  } else {
    // For white, the white perspective move IS the canonical move.
    strcpy(uci_move_canonical, uci_move_white_perspective);
  }
  // 2. Use the legal moves already generated at the start of c_step
  // (OPTIMIZATION) No need to regenerate - we already have the definitive list.
  // Get the correct move buffer
  char (*moves_buffer)[6];
  int moves_count;
  bool is_action_legal = false;
  if (env->context.board.to_move == C_WHITE) {
    moves_buffer = env->context.white_legal_moves_buffer;
    moves_count = env->context.white_legal_moves_count;
  } else {
    moves_buffer = env->context.black_legal_moves_buffer;
    moves_count = env->context.black_legal_moves_count;
  }
  
//   // Print the actual legal moves
//   printf("  Legal moves (%d): ", moves_count);
//   for (int i = 0; i < moves_count && i < 10; i++) {
//     printf("%s ", moves_buffer[i]);
//   }
//   if (moves_count > 10) printf("... (%d more)", moves_count - 10);
//   printf("\n");
  
  for (int i = 0; i < moves_count; i++) {
    if (strcmp(moves_buffer[i], uci_move_canonical) == 0) {
      is_action_legal = true;
      break;
    }
  }
  // 4. Validate against the ground truth.
  if (!is_action_legal) {
    const char *turn_color =
        (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK";
    // printf("[ILLEGAL MOVE] Action %d -> %s is not legal for %s\n", 
    //        action_idx, uci_move_canonical, turn_color);
    // printf("[ILLEGAL MOVE] Checked %d legal moves\n", moves_count);
    // printf("[ERROR] Illegal move attempted (ground truth validation): action
    // %d (%s) - %s's turn (agent %d)\n", action_idx, uci_move_canonical,
    // turn_color, agent_idx);
    // Invalidate the move, penalize the agent, and end the step without
    // applying the move.
    if (env->context.puzzle_mode) {
      env->rewards[agent_idx] += env->reward_puzzle_failed;
      env->context.accumulated_reward_puzzle_failed +=
          env->reward_puzzle_failed;
      env->context.puzzle_wrong_moves_this_episode += 1.0f;
      env->context.puzzle_attempts_this_episode += 1.0f;
      env->context.puzzle_tries_this_env++;
      env->context.puzzle_tries_this_episode++;  // Track attempt for episode
      env->global_puzzle_attempts++;
      
      // Check if we've reached the configured puzzle tries limit for the episode
      if (env->context.puzzle_tries_this_episode >= env->context.puzzle_max_tries_per_env) {
        env->terminals[0] = 1;
        env->terminals[1] = 1;
        add_log(env);
        // // printf("[OBS COMPUTE] Line 7679: Puzzle episode termination (max tries)\n");
        compute_observation_with_perspective(env, &env->context);
        return;
      }
      
      // Check if we've exceeded max tries for this puzzle
      if (env->context.puzzle_tries_this_env >=
          env->context.puzzle_max_tries_per_env) {
        // Don't terminate - continue episode for 8192 steps
        // Reset puzzle for next attempt
        env->context.puzzle_failed = false;
        env->context.puzzle_tries_this_env = 0;
        // // printf("[PUZZLE] Max tries reached, resetting puzzle. Stats: attempts=%.1f, wrong=%.1f\n",
        //        env->context.puzzle_attempts_this_episode,
        //        env->context.puzzle_wrong_moves_this_episode);
      }
      // DON'T reset yet - let agent see the penalty with current board
      // Mark for reset on NEXT step
      env->context.puzzle_failed = true;
      
      // Compute observation with CURRENT board (no move was made)
      compute_observation_with_perspective(env, &env->context);
      
    //   printf("[ILLEGAL MOVE] Showing penalty, will reset on next step\n");
      return;
    } else if (env->context.board.to_move == C_WHITE) {
      env->context.c_invalid_moves_white += 1;
      env->rewards[agent_idx] += env->context.c_reward_invalid_white;
    } else {
      env->context.c_invalid_moves_black += 1;
      env->rewards[agent_idx] += env->context.c_reward_invalid_black;
    }
    // Don't apply the move, just recompute observation and return.
    compute_observation_with_perspective(env, &env->context);
    PROFILE_STOP(profile_c_step_ticks);
    return;
  }
  // --- END OF NEW VALIDATION LOGIC ---
  // Check if this move is a capture before applying it
  int from_x = (uci_move_canonical[0] - 'a');
  int from_y = (uci_move_canonical[1] - '1');
  int to_x = (uci_move_canonical[2] - 'a');
  int to_y = (uci_move_canonical[3] - '1');
  // Get the piece at the destination to check for capture
  Piece *destination_piece = get_piece(&env->context.board, to_x, to_y);
  bool is_capture = (destination_piece && destination_piece->type != EMPTY);
  // Check for en passant capture
  bool is_en_passant = false;
  if (!is_capture) {
    Piece *moving_piece = get_piece(&env->context.board, from_x, from_y);
    if (moving_piece && moving_piece->type == PAWN &&
        (to_y * 8 + to_x) == env->context.board.ep_square) {
      is_capture = true;
      is_en_passant = true;
    }
  }
  // // Apply the move
  // printf("[MOVE_APPLY] Applying move %s for %s\n", uci_move_canonical,
  // (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK");
  // Store whose turn it was before the move (the player making the move)
  PieceColor moving_player = env->context.board.to_move;
//   printf("[MOVE] Attempting to apply: %s\n", uci_move_canonical);
  bool move_applied = apply_uci_move(&env->context, uci_move_canonical);
//   printf("[MOVE] Applied=%d, new turn: %s\n", move_applied,
//          env->context.board.to_move == C_WHITE ? "WHITE" : "BLACK");
  
  // Direct step logging in puzzle mode - log every 50 steps
  // DISABLED FOR PERFORMANCE - printf in hot path kills training speed!
  /*
  if (env->context.puzzle_mode && env->context.step_count % 50 == 0) {
//     // printf("[PUZZLE STEP] step_count=%d, wrong_moves=%.0f, solves=%d, tries=%d\n", 
           env->context.step_count,
           env->context.puzzle_wrong_moves_this_episode,
           env->context.puzzle_solves_this_episode,
           env->context.puzzle_tries_this_episode);
  }
  */
  // printf("[MOVE_APPLY] Move applied successfully: %s, new turn: %s\n",
  // move_applied ? "YES" : "NO",
  // (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK");
  
  // Episode termination is now handled where puzzle_tries_this_episode is incremented
  // This ensures episodes terminate immediately after the configured number of puzzle attempts
  
  // PUZZLE MODE - Special handling
  if (env->context.puzzle_mode) {
    // In puzzle mode, only white should be moving
    if (moving_player != C_WHITE) {
      // Error: Black move in puzzle mode
      env->terminals[agent_idx] = 1;
      return;
    }
    
    // DEBUG: Print observation BEFORE move (first few steps only)
    debug_step_count++;
    if (debug_step_count <= 5) {
    //   printf("\n[STEP %d] BEFORE MOVE:\n", debug_step_count);
    //   printf("  Move attempted: %s\n", uci_move_canonical);
    //   printf("  Observation sample (first 10): ");
      for (int i = 0; i < 10; i++) {
        // printf("%.1f ", env->observations[i]);
      }
    //   printf("\n");
    //   printf("  Mask at 1472: count=%.0f, first 5 actions: ", env->observations[1472]);
      for (int i = 0; i < 5 && i < (int)env->observations[1472]; i++) {
        int action_id = (int)env->observations[1473 + i];
        // printf("%d(%s) ", action_id, ACTION_ID_TO_UCI[action_id]);
      }
    //   printf("\n");
    }
    
    // Apply the move
    if (!move_applied) {
      // Move failed to apply
      env->terminals[agent_idx] = 1;
      return;
    }
    
    // Move was applied - board state has changed
    if (debug_step_count <= 5) {
    //   printf("  Move APPLIED successfully\n");
    }
  } else {
    // Non-puzzle mode - normal move application
    if (move_applied) {
      // Regular game logic here
    }
  }
  // PUZZLE MODE - Check if move is correct
  if (move_applied && env->context.puzzle_mode) {
    // printf("[PUZZLE CHECK] move_applied=%d, puzzle_mode=%d\n", move_applied, env->context.puzzle_mode);
    if (env->context.puzzle_move_index < env->context.puzzle_solution_length) {
    //   printf("[PUZZLE CHECK] Checking move index %d/%d\n", 
    //          env->context.puzzle_move_index, env->context.puzzle_solution_length);
      // Track timing on first move of puzzle
      if (env->context.puzzle_move_index == 0) {
        env->context.puzzle_samples_to_solve++;
        if (env->context.puzzle_start_time == 0) {
          env->context.puzzle_start_time = clock();
        }
      }
      // Check if the move matches the expected solution move
      const char *expected_move =
          env->context.puzzle_solution[env->context.puzzle_move_index];
      
      // DEBUG: Print what move we got
//       // printf("[PUZZLE DEBUG] Action %d converted to UCI: %s, Expected: %s\n", 
//              action_idx, uci_move_canonical, expected_move);
      
      // COMBINED FIX: Increment counters for EVERY attempt, BEFORE checking correctness
      env->context.puzzle_attempts_this_episode += 1.0f;
      env->global_puzzle_attempts++; // CRITICAL: Count all global attempts
      // Debug: Puzzle attempt (removed print for performance)
      
      // No progress printing
        
        // No debug stats
      
      // Log puzzle training details with rewards and state
      // // printf("[PUZZLE TRAIN] Env %d | Puzzle idx: %d | Tries: %d | FEN: %s | Expected: %s | Agent chose: %s | Action: %d | ",
      //        env->env_id, env->context.current_puzzle_idx, env->context.puzzle_tries_this_env,
      //        env->context.puzzle_fen, expected_move, uci_move_canonical, action_idx);
      
      // Log more observation values to see the actual board state
      // Check different parts of the observation space
      int num_nonzero = 0;
      for (int i = 0; i < 1472; i++) {
        if (env->observations[i] != 0.0f) num_nonzero++;
      }
      // printf("Obs[22]: %.1f | Obs[136]: %.1f | Obs[391]: %.1f | NonZero: %d/1472 | ", 
      //        env->observations[22], env->observations[136], env->observations[391], num_nonzero);
      
      int cmp_result = strcmp(uci_move_canonical, expected_move);
//       // printf("[PUZZLE DEBUG] strcmp result: %d (0 means equal)\n", cmp_result);
      
      if (cmp_result == 0) {
        // Correct move! Award reward and advance to next move
//         // printf("[PUZZLE DEBUG] CORRECT MOVE! Agent idx=%d, Adding rewards: correct=%f, solved=%f\n",
//                agent_idx, env->reward_correct_move, env->reward_puzzle_solved);
//         // printf("[PUZZLE DEBUG] Before: env->rewards[%d] = %f\n", agent_idx, env->rewards[agent_idx]);
        env->rewards[agent_idx] += env->reward_correct_move;
//         // printf("[PUZZLE DEBUG] After: env->rewards[%d] = %f\n", agent_idx, env->rewards[agent_idx]);
        env->context.accumulated_reward_puzzle_correct_move +=
            env->reward_correct_move;
        env->context.episode_return_white += env->reward_correct_move;
        env->context.puzzle_move_index++;
        // Track correct moves in context for later logging
        env->context.puzzle_correct_moves_this_episode += 1.0f;
        // Check if puzzle is complete
//         // printf("[PUZZLE DEBUG] After increment: move_index=%d, solution_length=%d\n", 
//                env->context.puzzle_move_index, env->context.puzzle_solution_length);
        if (env->context.puzzle_move_index >=
            env->context.puzzle_solution_length) {
        //   // Puzzle solved! Award completion reward
          printf("[PUZZLE SOLVED] env_id=%d, global_successes=%d, global_attempts=%d\n", 
                 env->env_id, env->global_puzzle_successes + 1, env->global_puzzle_attempts);
          env->rewards[agent_idx] += env->reward_puzzle_solved;
//           // printf("[PUZZLE DEBUG] Final reward: env->rewards[%d] = %f\n", agent_idx, env->rewards[agent_idx]);
          env->context.accumulated_reward_puzzle_solved +=
              env->reward_puzzle_solved;
          env->context.episode_return_white += env->reward_puzzle_solved;
          env->context.puzzle_completed = true;
          env->global_puzzle_successes++;
          
          // No debug logging
          
          // Calculate performance metrics
          clock_t solve_time = clock() - env->context.puzzle_start_time;
          // Time and sample tracking removed - using simplified Log struct
          (void)solve_time;  // Suppress unused variable warning
          // Check if global threshold reached (simplified check)
          if (env->global_puzzle_attempts >= 1) { // Minimum sample
            float global_success_rate = (float)env->global_puzzle_successes /
                                        env->global_puzzle_attempts;
            if (global_success_rate >= env->global_puzzle_success_threshold) {
              // Advance to next random puzzle from the set
              if (env->context.puzzle_set_size > 1) {
                // Select a random puzzle from the set
                int new_idx = rand() % env->context.puzzle_set_size;
                env->context.current_puzzle_idx = new_idx;
                
                // Create temporary array of solution pointers
                const char* new_solution[10];
                for (int i = 0; i < env->context.puzzle_set_solution_lengths[new_idx] && i < 10; i++) {
                  new_solution[i] = env->context.puzzle_set_solutions[new_idx][i];
                }
                
                // Load the new puzzle
                // // printf("[PUZZLE ROTATE] Env %d switching to puzzle %d (was %d)\n", 
                //        env->env_id, new_idx, env->context.current_puzzle_idx);
                set_puzzle_data(env, env->context.puzzle_set_fens[new_idx], 
                               new_solution, 
                               env->context.puzzle_set_solution_lengths[new_idx]);
                
                // // printf("[PUZZLE] Switching to puzzle %d from set\n", new_idx);
              }
              env->global_puzzle_id++;
              env->global_puzzle_attempts = 0;
              env->global_puzzle_successes = 0;
            }
          }
          // Track puzzle solved in context for later logging
          env->context.puzzle_solved_this_env++;
          env->context.puzzle_solves_this_episode++;  // Track solve for episode
          env->context.puzzle_tries_this_episode++;   // Track attempt for episode
          
          // Update GLOBAL persistent counters (shared across all envs)
          global_total_puzzle_solves++;
          global_total_puzzle_attempts++;
          
          // Update puzzle score for logging
          update_puzzle_score(env);
          
          // Print total reward after all components are added
          // printf("CORRECT! Total Reward: %.2f\n", env->rewards[agent_idx]);
          // fflush(stdout);
          
          // IMPORTANT: DO NOT reset board - agent needs to see the RESULT of their move
          // The agent needs to see the POST-MOVE board state with the reward signal
          // // printf("[PUZZLE SOLVED] Keeping post-move board state for observation\n");
          // env->context.board already has the move applied - keep it!
          
          // CRITICAL FIX: Terminate episode immediately after correct move
          // One attempt = one episode for proper RL training
          env->terminals[0] = 1;
          env->terminals[1] = 1;
          add_log(env);
          // // // printf("[OBS COMPUTE] Line 7903: After correct move - computing obs with POST-MOVE state\n");
//           // printf("[BEFORE OBS] rewards[0]=%f\n", env->rewards[0]);
          compute_observation_with_perspective(env, &env->context);
          
          // DEBUG: Print observation AFTER correct move
          if (debug_step_count <= 5) {
            // printf("[STEP %d] AFTER CORRECT MOVE (terminal, reward=%.1f):\n", 
            //        debug_step_count, env->rewards[0]);
            // printf("  Observation sample (first 10): ");
            for (int i = 0; i < 10; i++) {
            //   printf("%.1f ", env->observations[i]);
            }
            // printf("\n");
          }
          
          return;
          
          // Code below is now unreachable since we terminate immediately
          // Keeping it commented for reference
          /*
          // Global puzzle tracking removed - using simplified Log struct
          // Reset for next puzzle
          env->context.puzzle_tries_this_env = 0;
          env->context.puzzle_samples_to_solve = 0;
          env->context.puzzle_start_time = 0;
          */
        } else {
          // More moves to make - recompute observation for next move
          // Continue to next move
          compute_observation_with_perspective(env, &env->context);
          return;
        }
      } else {
        // Wrong move! Give penalty and reward shaping
        printf("[PUZZLE WRONG] env_id=%d, expected=%s, got action_idx=%d\n", 
               env->env_id, env->context.puzzle_solution[env->context.puzzle_move_index], action_idx);
        float total_penalty = env->reward_puzzle_failed;
        // printf("[DEBUG] env->reward_puzzle_failed = %.4f, total_penalty = %.4f\n", 
        //        env->reward_puzzle_failed, total_penalty);
        // Parse the expected and actual moves
        ChessMove expected_move;
        if (!parse_uci_move(
                env->context.puzzle_solution[env->context.puzzle_move_index],
                &expected_move)) {
          // Failed to parse expected move
          // // printf("[PUZZLE ERROR] Failed to parse expected move: %s\n",
          //        env->context.puzzle_solution[env->context.puzzle_move_index]);
        } else {
          // Parse the actual move made
          ChessMove actual_move;
          char uci_move[6];
          action_to_uci(action_idx, uci_move);
          if (parse_uci_move(uci_move, &actual_move)) {
            // Reward shaping based on move similarity
            float shaping_reward = 0.0f;
            // 1. Reward for moving the correct piece
            if (actual_move.from.x == expected_move.from.x &&
                actual_move.from.y == expected_move.from.y) {
              shaping_reward += env->reward_puzzle_correct_piece;
              // 2. Additional reward based on how close we moved to the target
              // Calculate Manhattan distance from actual destination to
              // expected destination
              int expected_row = expected_move.to.y;
              int expected_col = expected_move.to.x;
              int actual_row = actual_move.to.y;
              int actual_col = actual_move.to.x;
              int distance = abs(expected_row - actual_row) +
                             abs(expected_col - actual_col);
              // Max distance on board is 14 (7+7), so normalize
              float distance_reward = env->reward_puzzle_closer_to_target *
                                      (1.0f - (float)distance / 14.0f);
              shaping_reward += distance_reward;
              // 3. If promotion expected and we promoted to same piece, bonus
              if (expected_move.promotion != EMPTY &&
                  actual_move.promotion == expected_move.promotion) {
                shaping_reward += env->reward_puzzle_correct_promotion;
              }
            }
            total_penalty += shaping_reward;
          }
        }
        // printf("[PUZZLE WRONG] Expected: %s, Got: %s, Penalty: %.2f\n",
        //        env->context.puzzle_solution[env->context.puzzle_move_index],
        //        uci_move_canonical, total_penalty);
        env->rewards[agent_idx] += total_penalty;
        env->context.accumulated_reward_puzzle_failed += total_penalty;
        // printf("[PUZZLE WRONG] Set rewards[%d] = %.2f\n", agent_idx, env->rewards[agent_idx]);
        // In puzzle mode, only white plays, so only update white's episode
        // return
        env->context.episode_return_white += total_penalty;
        // Track wrong moves in context for later logging
        env->context.puzzle_wrong_moves_this_episode += 1.0f;
        env->context.puzzle_tries_this_env++;
        env->context.puzzle_tries_this_episode++;  // Track attempt for episode
        
        // Update GLOBAL persistent counter for failed attempt
        global_total_puzzle_attempts++;
        
        // Update puzzle score for logging
        update_puzzle_score(env);
        
        // printf("[WRONG MOVE] wrong_moves_this_episode now: %.1f, tries_this_env: %d\n",
        //        env->context.puzzle_wrong_moves_this_episode,
        //        env->context.puzzle_tries_this_env);
        // NOTE: global_puzzle_attempts already incremented above for ALL attempts

        // ALWAYS reset the board after a wrong move, even if about to terminate
        // This ensures the observation shows the correct puzzle position
        // // printf("[PUZZLE RESET] After wrong move! Resetting to FEN: %s\n", env->context.puzzle_fen);
        
        // IMPORTANT: Agent needs to see the RESULT of their wrong move
        // Mark puzzle as failed
        env->context.puzzle_move_index = 0;
        env->context.puzzle_failed = true;
        
        // ALWAYS terminate after a wrong move so agent sees the consequence
        // The episode boundary is critical for RL learning
        env->terminals[0] = 1;
        env->terminals[1] = 1;
        // printf("[PUZZLE] Wrong move - terminating episode (try %d/%d)\n",
        //        env->context.puzzle_tries_this_episode, 
        //        env->context.puzzle_max_tries_per_env);
        
        add_log(env);
        
        // Show the post-move state so agent can learn from the mistake
        compute_observation_with_perspective(env, &env->context);
        
        // printf("[C_STEP %d END] After wrong move:\n", step_count);
        // printf("  terminals[0]=%d, rewards[0]=%.2f\n", env->terminals[0], env->rewards[0]);
        // printf("  puzzle_failed=%d\n", env->context.puzzle_failed);
        // printf("  puzzle_tries_this_episode=%d/%d\n", 
        //        env->context.puzzle_tries_this_episode, 
        //        env->context.puzzle_max_tries_per_env);
        
        return;
      }
    }
  } // End of puzzle mode block
  // IMPORTANT: Skip all regular game logic in puzzle mode
  if (env->context.puzzle_mode) {
    // This should never be reached due to returns above, but just in case
    // // printf("[PUZZLE WARNING] Reached end of puzzle block without proper "
    //        "return!\n");
    // // printf("[OBS COMPUTE] Line 8081: Puzzle failsafe\n");
    compute_observation_with_perspective(env, &env->context);
    return;
  }
  // Check if the move put the opponent in check and award check reward
  if (move_applied) {
    // After the move, it's now the opponent's turn - check if they're in check
    PieceColor opponent = env->context.board.to_move;
    if (is_in_check(&env->context.board, opponent)) {
      // The moving player put their opponent in check - award check reward
      float check_reward = (moving_player == C_WHITE)
                               ? env->context.c_reward_check_white
                               : env->context.c_reward_check_black;
      env->rewards[agent_idx] += check_reward;
      // Track accumulated rewards for logging
      if (moving_player == C_WHITE) {
        env->context.accumulated_reward_check_white += check_reward;
      } else {
        env->context.accumulated_reward_check_black += check_reward;
      }
    }
  }
  // Assign rewards
  env->rewards[agent_idx] += env->context.c_reward_valid;
  env->context.accumulated_reward_valid += env->context.c_reward_valid;
  // Apply material advantage rewards every step
  int white_material = calculate_material_value(&env->context.board, C_WHITE);
  int black_material = calculate_material_value(&env->context.board, C_BLACK);
  int material_diff =
      white_material - black_material; // Positive when WHITE ahead
  // Reward based on material advantage: positive for advantage, negative for
  // disadvantage
  float white_material_reward =
      material_diff * env->context.c_reward_material_diff_white;
  float black_material_reward =
      -material_diff * env->context.c_reward_material_diff_black;
  // Apply material advantage rewards every step
  env->rewards[0] +=
      white_material_reward; // WHITE gets + for advantage, - for disadvantage
  env->rewards[1] +=
      black_material_reward; // BLACK gets + for advantage, - for disadvantage
  // Track accumulated material rewards for logging
  env->context.accumulated_reward_material_diff_white += white_material_reward;
  env->context.accumulated_reward_material_diff_black += black_material_reward;
  // Assign capture rewards if this was a capture
  if (is_capture) {
    float capture_reward = 0.0f;
    if (env->context.c_use_piece_value_capture_rewards) {
      // Use piece-value-based rewards
      PieceType captured_piece_type = PAWN; // Default for en passant
      if (!is_en_passant && destination_piece) {
        captured_piece_type = destination_piece->type;
      }
      int piece_value = get_piece_value(captured_piece_type);
      capture_reward =
          piece_value * env->context.c_piece_value_reward_multiplier;
    } else {
      // Use fixed capture rewards
      if (env->context.board.to_move == C_WHITE) {
        capture_reward = env->context.c_reward_white_captures_enemy_piece;
      } else {
        capture_reward = env->context.c_reward_black_captures_enemy_piece;
      }
    }
    // Apply the capture reward
    env->rewards[agent_idx] += capture_reward;
    // Track accumulated rewards for logging
    if (env->context.board.to_move == C_WHITE) {
      env->context.accumulated_reward_white_captures_enemy_piece +=
          capture_reward;
    } else {
      env->context.accumulated_reward_black_captures_enemy_piece +=
          capture_reward;
    }
    // Also track en passant captures
    if (is_en_passant) {
      if (env->context.board.to_move == C_WHITE) {
        env->context.c_en_passant_white += 1;
      } else {
        env->context.c_en_passant_black += 1;
      }
    }
  }
  if (env->context.board.to_move == C_WHITE) {
    env->context.c_white_moves += 1;
  } else {
    env->context.c_black_moves += 1;
  }
  env->context.c_valid_moves += 1;
  // In chess, each player gets their own rewards based on their color
  // Agent 0 = WHITE, Agent 1 = BLACK
  // Only the moving player gets action-based rewards (valid move, captures,
  // checks) Both players always get material difference rewards every step
  // Track episode returns by color (not by agent index)
  env->context.episode_return_white += env->rewards[0];
  env->context.episode_return_black += env->rewards[1];
  // Check for game over conditions using already-generated legal moves
  // (OPTIMIZATION) Skip normal game termination logic in puzzle mode
  if (env->context.puzzle_mode) {
    // In puzzle mode, termination is handled by puzzle logic only
    return;
  }
  bool game_over = false;
  // Check the appropriate move count based on whose turn it is
  int current_move_count = (env->context.board.to_move == C_WHITE)
                               ? env->context.white_legal_moves_count
                               : env->context.black_legal_moves_count;
  bool any_legal_move_exists = (current_move_count > 0);
  if (!any_legal_move_exists) {
    game_over = true;
    if (is_in_check(&env->context.board, env->context.board.to_move)) {
      // CHECKMATE
      if (env->context.board.to_move ==
          C_WHITE) { // White is checkmated (black won)
        float win_reward = env->context.c_reward_win_black;
        float loss_reward = env->context.c_reward_loss_white;
        // Both agents get shared reward based on game outcome
        env->rewards[0] += win_reward;
        env->rewards[1] += loss_reward;
        env->context.c_white_checkmated += 1;
        env->context.c_black_win += 1;
        env->context.c_white_loss += 1;
        // Add accumulated reward tracking for logging
        env->context.accumulated_reward_win_black += win_reward;
        env->context.accumulated_reward_loss_white +=
            env->context.c_reward_loss_white;
      } else { // Black is checkmated (white won)
        float win_reward = env->context.c_reward_win_white;
        float loss_reward = env->context.c_reward_loss_black;
        // Both agents get shared reward based on game outcome
        env->rewards[0] += win_reward;
        env->rewards[1] += loss_reward;
        env->context.c_black_checkmated += 1;
        env->context.c_white_win += 1;
        env->context.c_black_loss += 1;
        // Add accumulated reward tracking for logging
        env->context.accumulated_reward_win_white += win_reward;
        env->context.accumulated_reward_loss_black +=
            env->context.c_reward_loss_black;
      }
    } else {
      // STALEMATE
      env->rewards[0] += env->context.c_reward_draw;
      env->rewards[1] += env->context.c_reward_draw;
      env->context.c_stalemate += 1;
      env->context.c_game_drawn += 1;
      // Add accumulated reward tracking for logging
      env->context.accumulated_reward_draw += env->context.c_reward_draw;
    }
  } else if (env->context.board.halfmove_clock >= 100) {
    game_over = true; // FIFTY-MOVE RULE
    env->rewards[0] += env->context.c_reward_draw;
    env->rewards[1] += env->context.c_reward_draw;
    env->context.c_fifty_move_rule += 1;
    env->context.c_game_drawn += 1;
    // Add accumulated reward tracking for logging
    env->context.accumulated_reward_draw += env->context.c_reward_draw;
  } else if (is_threefold_repetition(&env->context)) {
    game_over = true; // THREEFOLD REPETITION
    env->rewards[0] += env->context.c_reward_draw;
    env->rewards[1] += env->context.c_reward_draw;
    env->context.c_threefold_repetition += 1;
    env->context.c_game_drawn += 1;
    // Add accumulated reward tracking for logging
    env->context.accumulated_reward_draw += env->context.c_reward_draw;
  } else if (is_insufficient_material(&env->context)) {
    game_over = true; // INSUFFICIENT MATERIAL
    env->rewards[0] += env->context.c_reward_draw;
    env->rewards[1] += env->context.c_reward_draw;
    env->context.c_insufficient_material += 1;
    env->context.c_game_drawn += 1;
    // Add accumulated reward tracking for logging
    env->context.accumulated_reward_draw += env->context.c_reward_draw;
  } else if (env->max_depth > 0 && env->context.step_count >= env->max_depth) {
    game_over = true; // MAX DEPTH / TRUNCATION
    env->rewards[0] += env->context.c_reward_max_depth_termination;
    env->rewards[1] += env->context.c_reward_max_depth_termination;
    env->context.c_max_depth += 1;
    env->context.c_game_drawn += 1;
    // Add accumulated reward tracking for logging
    env->context.accumulated_reward_draw += env->context.c_reward_draw;
  }
  if (game_over) {
    // Mark both agents as terminal
    env->terminals[0] = 1;
    env->terminals[1] = 1;
    // Complete game move count removed - using simplified Log struct
    add_log(env);
    // Notify UI about game end via function call (before auto-reset clears
    // counters) notify_game_end(env->context.c_white_win > 0,
    // env->context.c_black_win > 0, env->context.c_game_drawn > 0); Check if we
    // should log this complete game BEFORE reset
    // Designate the first environment that completes a game as the logging
    // environment
    if (first_active_env_id == -1) {
      first_active_env_id = env->env_id;
      logging_env_id = env->env_id;
    }
    // Only log from the designated logging environment to avoid spam from 512
    // environments
    if (env->env_id == logging_env_id) {
      // Increment game counter for the logging environment
      env->context.steps_since_last_log++;
      // Log every N games completed by the logging environment
      if (env->context.game_logging_frequency > 0 &&
          env->context.steps_since_last_log >=
              env->context.game_logging_frequency) {
        write_complete_game_to_file(&env->context, env->env_id);
        // Game step logging removed - using simplified Log struct
        env->context.steps_since_last_log = 0; // Reset counter
      }
    } else {
    }
    // Debug: Always print for any env that completes a game
    if (env->context.complete_game_action_count > 0) {
    }
    // Save values before reset
    int saved_steps = env->context.steps_since_last_log;
    int saved_freq = env->context.game_logging_frequency;
    // AUTO-RESET: Manually reset the environment to start a new game
    c_reset(env);
    // Restore saved values after reset
    env->context.steps_since_last_log = saved_steps;
    env->context.game_logging_frequency = saved_freq;
  } else {
    // Compute new observation if the game is not over
    if (env->context.puzzle_mode) {
//       // // printf("[OBS COMPUTE ERROR] Line 8308: SHOULD NOT REACH HERE IN PUZZLE MODE!\n");
//       // printf("[OBS COMPUTE ERROR] Board state - rook at a2: %s\n",
//              (env->context.board.board[8].type == ROOK && 
//               env->context.board.board[8].color == C_WHITE) ? "YES" : "NO");
    }
//     // printf("[OBS COMPUTE] Line 8308: Regular game continue\n");
    compute_observation_with_perspective(env, &env->context);
  }
  // PROFILING: Print performance statistics every 1000 steps
  static int profiling_step_count = 0;
  profiling_step_count++;
  if (profiling_step_count % 1000 == 0) {
    double total_time = (double)profile_c_step_ticks / CLOCKS_PER_SEC;
    double move_gen_time = (double)profile_move_gen_uci_ticks / CLOCKS_PER_SEC;
    double obs_time = (double)profile_compute_obs_ticks / CLOCKS_PER_SEC;
    double legal_move_time =
        (double)profile_is_legal_move_ticks / CLOCKS_PER_SEC;
    double square_attack_time =
        (double)profile_is_square_attacked_ticks / CLOCKS_PER_SEC;
    double apply_move_time =
        (double)profile_apply_uci_move_ticks / CLOCKS_PER_SEC;
    // printf("[CHESS_PROFILE] Step %d - Total: %.3fs, MoveGen: %.3fs (%.1f%%),
    // Obs: %.3fs (%.1f%%), LegalCheck: %.3fs (%.1f%%), SquareAttack: %.3fs
    // (%.1f%%), ApplyMove: %.3fs (%.1f%%)\n", profiling_step_count, total_time,
    // move_gen_time, move_gen_time/total_time*100,
    // obs_time, obs_time/total_time*100,
    // legal_move_time, legal_move_time/total_time*100,
    // square_attack_time, square_attack_time/total_time*100,
    // apply_move_time, apply_move_time/total_time*100);
  }
  PROFILE_STOP(profile_c_step_ticks);
  // // printf("[C_STEP DEBUG] c_step completed\n");
  // fflush(stdout);
  // printf("[C_STEP EXIT] Final rewards: env->rewards[0]=%f, env->rewards[1]=%f, rewards ptr=%p\n", 
  //        env->rewards[0], env->rewards[1], (void*)env->rewards);
}

// === PUFFERLIB LOGGING FUNCTION ===
void add_log(CChess *env) {
  // Only log the essential metrics as requested
  env->log.episode_length += (float)env->context.step_count;
  env->log.episode_return +=
      env->context.episode_return_white + env->context.episode_return_black;
  
  // For puzzle mode, ALWAYS set score based on global counters
  // This ensures all environments report the same score for correct averaging
  if (env->context.puzzle_mode && global_total_puzzle_attempts > 0) {
    float solve_rate = (float)global_total_puzzle_solves / (float)global_total_puzzle_attempts;
    env->log.score = solve_rate;
    env->log.perf = (float)global_total_puzzle_attempts;
  }
  
  // Increment n (must be last for PufferLib aggregation)
  env->log.n += 1.0f;
}

// Update puzzle score continuously (called from c_step)
// NOTE: This updates only the calling environment's log, but the global score
// should be propagated to ALL environments for correct averaging
void update_puzzle_score(CChess *env) {
  if (env->context.puzzle_mode && global_total_puzzle_attempts > 0) {
    float solve_rate = (float)global_total_puzzle_solves / (float)global_total_puzzle_attempts;
    env->log.score = solve_rate;
    env->log.perf = (float)global_total_puzzle_attempts;
    // Debug output only from env 0 to avoid spam
    if (env->env_id == 0) {
//       // printf("[PUZZLE SCORE] %d/%d solves, rate=%f\n",
//              global_total_puzzle_solves, global_total_puzzle_attempts, solve_rate);
    }
  }
}

// Update ALL environments' scores to the global score
// This ensures correct averaging in vec_log
void update_all_puzzle_scores(CChess *env) {
  if (env->context.puzzle_mode && global_total_puzzle_attempts > 0) {
    float solve_rate = (float)global_total_puzzle_solves / (float)global_total_puzzle_attempts;
    // This only updates the current env, but we need a different approach
    // The real fix is to make sure add_log sets the same score for all envs
    env->log.score = solve_rate;
    env->log.perf = (float)global_total_puzzle_attempts;
  }
}
void c_render(CChess *env) {
  // printf("\n +---+---+---+---+---+---+---+---+\n");
  for (int y = 7; y >= 0; y--) {
    // printf("%d |", y + 1);
    for (int x = 0; x < 8; x++) {
      const Piece *p = get_piece_const(&env->context.board, x, y);
      char piece_char = ' ';
      if (p && p->type != EMPTY) {
        const char pieces[] = " KQRBNP";
        piece_char = pieces[p->type];
        if (p->color == C_BLACK) {
          piece_char = piece_char + ('a' - 'A'); // Make lowercase
        }
      }
      // printf(" %c |", piece_char);
    }
    // printf("\n +---+---+---+---+---+---+---+---+\n");
  }
  // printf(" a b c d e f g h\n");
  // printf("\nTo move: %s\n",
  //        (env->context.board.to_move == C_WHITE) ? "White" : "Black");
  // printf("Step: %d\n", env->context.step_count);
}
void c_close(CChess *env) {
  // Core cleanup for chess environment
  // Currently all major data structures use static allocation within CChess
  // struct Future: Clean up any Stockfish process handles, pipes, or other
  // system resources when Stockfish integration is fully implemented
  // Clear sensitive data and reset state
  memset(&env->context, 0, sizeof(ChessContext));
  memset(&env->log, 0, sizeof(Log));
}
// === DUAL AGENT SELF-PLAY MODE SETTERS ===
void set_dual_agent_self_play_mode(CChess *env, bool enabled) {
  env->context.dual_agent_self_play_mode = enabled;
  env->context.self_play_mode = enabled; // Also set self_play_mode
}
void set_self_play_mode(CChess *env, bool enabled) {
  env->context.self_play_mode = enabled;
}
void enable_stockfish_black(CChess *env, const char *stockfish_cmd, int elo,
                            int search_ms) {
  // TODO: Implement full Stockfish integration in pure C
  // Current C++ implementation in stockfish_wrapper.h needs to be ported to C
  // This includes:
  // 1. Process spawning and communication via pipes
  // 2. UCI protocol protocol implementation
  // 3. FEN position synchronization
  // 4. Move request/response handling
  // 5. ELO and time control settings
  //
  // For now, this is a compatibility stub
  if (env && stockfish_cmd) {
    env->stockfish_enabled = true;
    // Store parameters for future implementation
    strncpy(env->context.stockfish_cmd, stockfish_cmd,
            sizeof(env->context.stockfish_cmd) - 1);
    env->context.stockfish_elo = elo;
    env->context.stockfish_search_ms = search_ms;
  }
}
// FEN support (basic)
void c_set_fen(CChess *env, const char *fen) {
  ChessBoard *board = &env->context.board;

  // Clear the board first
  memset(board, 0, sizeof(ChessBoard));

  // Parse FEN string
  int x = 0, y = 7; // Start from rank 8 (y=7)
  const char *p = fen;

  // Parse board position
  while (*p && *p != ' ') {
    if (*p == '/') {
      x = 0;
      y--;
    } else if (*p >= '1' && *p <= '8') {
      x += (*p - '0'); // Skip empty squares
    } else {
      // Parse piece
      PieceColor color = (*p >= 'A' && *p <= 'Z') ? C_WHITE : C_BLACK;
      PieceType type = EMPTY;

      char piece_char = (*p >= 'A' && *p <= 'Z') ? *p : (*p - 'a' + 'A');
      switch (piece_char) {
      case 'P':
        type = PAWN;
        break;
      case 'N':
        type = KNIGHT;
        break;
      case 'B':
        type = BISHOP;
        break;
      case 'R':
        type = ROOK;
        break;
      case 'Q':
        type = QUEEN;
        break;
      case 'K':
        type = KING;
        break;
      }

      if (type != EMPTY && x < 8 && y >= 0) {
        board->board[y * 8 + x] = (Piece){color, type};
        x++;
      }
    }
    p++;
  }

  // Parse active color
  if (*p == ' ')
    p++;
  board->to_move = (*p == 'w') ? C_WHITE : C_BLACK;

  // Parse castling rights
  if (*p == ' ')
    p++;
  if (*p == ' ')
    p++;
  board->castle_rights = 0;
  while (*p && *p != ' ') {
    switch (*p) {
    case 'K':
      board->castle_rights |= 0x1;
      break; // White kingside
    case 'Q':
      board->castle_rights |= 0x2;
      break; // White queenside
    case 'k':
      board->castle_rights |= 0x4;
      break; // Black kingside
    case 'q':
      board->castle_rights |= 0x8;
      break; // Black queenside
    }
    p++;
  }

  // Parse en passant square
  if (*p == ' ')
    p++;
  board->ep_square = -1;
  if (*p != '-') {
    int ep_x = p[0] - 'a';
    int ep_y = p[1] - '1';
    if (ep_x >= 0 && ep_x < 8 && ep_y >= 0 && ep_y < 8) {
      board->ep_square = ep_y * 8 + ep_x;
    }
  }

  // Skip halfmove and fullmove (we can add these later if needed)
  board->halfmove_clock = 0;
  board->fullmove_number = 1;

  // Compute zobrist hash for the new position
  board->zobrist_hash = compute_zobrist_hash(board);

  // Reset context state
  env->context.white_legal_moves_count = 0;
  env->context.black_legal_moves_count = 0;
  env->context.white_moves_cached = false;
  env->context.black_moves_cached = false;
  env->context.position_fully_cached = false;
  // Don't reset step_count in puzzle mode - it should accumulate across tries
  if (!env->context.puzzle_mode) {
    env->context.step_count = 0;
  }

  // Mark that FEN was set
  board->fen_was_set = true;

  // Recompute observation
  compute_observation_with_perspective(env, &env->context);
}


// Close extern "C" block before defining c_print_profile_data
#ifdef __cplusplus
}  // extern "C"
#endif

// Define the function outside of extern "C" block
void c_print_profile_data() {
  profile_total_ticks = profile_c_step_ticks;
  if (profile_total_ticks == 0) {
    // printf("No profiling data collected yet.\n");
    return;
  }
  // printf("\n--- Chess Engine Profile ---\n");
  // printf("Function | Time (ms) | %% of Total\n");
  // printf("---------------------------------------|-----------|------------\n");
  double total_ms = (double)profile_total_ticks * 1000.0 / CLOCKS_PER_SEC;
  // printf("c_step (Total) | %9.2f | %8.2f%%\n", total_ms, 100.0);
  double move_gen_ms =
      (double)profile_move_gen_uci_ticks * 1000.0 / CLOCKS_PER_SEC;
  // printf(" -> chess_generate_legal_moves_uci | %9.2f | %8.2f%%\n", move_gen_ms,
  //        (move_gen_ms / total_ms) * 100);
  double is_legal_ms =
      (double)profile_is_legal_move_ticks * 1000.0 / CLOCKS_PER_SEC;
  // printf(" -> chess_is_legal_move | %9.2f | %8.2f%%\n", is_legal_ms,
  //        (is_legal_ms / total_ms) * 100);
  double is_attacked_ms =
      (double)profile_is_square_attacked_ticks * 1000.0 / CLOCKS_PER_SEC;
  // printf(" -> is_square_attacked | %9.2f | %8.2f%%\n", is_attacked_ms,
  //        (is_attacked_ms / total_ms) * 100);
  double make_move_ms =
      (double)profile_make_move_fast_ticks * 1000.0 / CLOCKS_PER_SEC;
  // printf(" -> make_move_fast | %9.2f | %8.2f%%\n", make_move_ms,
  //        (make_move_ms / total_ms) * 100);
  double unmake_move_ms =
      (double)profile_unmake_move_fast_ticks * 1000.0 / CLOCKS_PER_SEC;
  // printf(" -> unmake_move_fast | %9.2f | %8.2f%%\n", unmake_move_ms,
  //        (unmake_move_ms / total_ms) * 100);
  double apply_uci_ms =
      (double)profile_apply_uci_move_ticks * 1000.0 / CLOCKS_PER_SEC;
  // printf(" -> apply_uci_move | %9.2f | %8.2f%%\n", apply_uci_ms,
  //        (apply_uci_ms / total_ms) * 100);
  double compute_obs_ms =
      (double)profile_compute_obs_ticks * 1000.0 / CLOCKS_PER_SEC;
  // printf(" -> compute_observation_with_perspective | %9.2f | %8.2f%%\n",
  //        compute_obs_ms, (compute_obs_ms / total_ms) * 100);
  // printf("----------------------------------------------------------\n");
}

#endif // CHESS_H