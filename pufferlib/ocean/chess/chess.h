// chess.h - Complete Pure C Chess Environment for PufferLib
// Optimized for 150k+ SPS performance with single network self-play

#ifndef CHESS_H
#define CHESS_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Enable debug logging for development (set to 0 to disable)
#ifndef DEBUG_LOG
#define DEBUG_LOG 0
#endif

#if DEBUG_LOG
#define DBG(expr) printf("%s", expr)
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
  float perf;
  float score;
  float episode_length;
  float episode_return;       // combined (existing)
  float episode_return_white; // new – white perspective total
  float episode_return_black; // new – black perspective total
  float reward_valid;
  float reward_white_captures_enemy_piece;
  float reward_black_captures_enemy_piece;
  float reward_max_depth_termination;
  float reward_draw;
  // Perspective-based reward tracking
  float reward_win_white;  // win rewards from white's perspective
  float reward_win_black;  // win rewards from black's perspective
  float reward_loss_white; // loss rewards from white's perspective
  float reward_loss_black; // loss rewards from black's perspective
  float reward_draw_white; // draw rewards from white's perspective
  float reward_draw_black; // draw rewards from black's perspective
  float game_drawn;
  // New separate win/loss tracking from both perspectives
  float white_win;  // white wins (from white's perspective)
  float white_loss; // white losses (from white's perspective)
  float black_win;  // black wins (from black's perspective)
  float black_loss; // black losses (from black's perspective)
  float stalemate;
  float insufficient_material;
  float threefold_repetition;
  float fifty_move_rule;
  float max_depth;
  float white_checkmated; // black checkmates white
  float black_checkmated; // white checkmates black
  float white_moves;
  float black_moves;
  float valid_moves;
  float invalid_moves_white;
  float invalid_moves_black;
  float reward_check_white;
  float reward_check_black;
  float reward_material_diff_white;
  float reward_material_diff_black;
  float stockfish_eval;
  // En passant captures
  float en_passant_white; // white captures via en passant
  float en_passant_black; // black captures via en passant
  // Castling moves
  float white_castle_kingside;  // white castles kingside
  float white_castle_queenside; // white castles queenside
  float black_castle_kingside;  // black castles kingside
  float black_castle_queenside; // black castles queenside
  // Pawn promotions
  float white_promotion_count;  // total white pawn promotions
  float white_promotion_knight; // white promotes to knight
  float white_promotion_bishop; // white promotes to bishop
  float white_promotion_rook;   // white promotes to rook
  float white_promotion_queen;  // white promotes to queen
  float black_promotion_count;  // total black pawn promotions
  float black_promotion_knight; // black promotes to knight
  float black_promotion_bishop; // black promotes to bishop
  float black_promotion_rook;   // black promotes to rook
  float black_promotion_queen;  // black promotes to queen

  // Game logging fields expected by binding.cpp
  float last_move_from;
  float last_move_to;
  float last_move_promotion;
  float game_step_logged;
  float game_moves_count;

  float complete_game_move_count;
  // Note: complete_game_moves_serialized removed to comply with PufferLib
  // float-only logging spec

  // n field is always last per pufferlib spec
  float n;
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

  // CRITICAL: Performance optimization - pre-allocated legal moves buffer
  char legal_moves_buffer[256][6];       // Up to 256 UCI strings
  int legal_moves_action_ids_white[256]; // Cached action IDs for white
                                         // perspective
  int legal_moves_action_ids_black[256]; // Cached action IDs for black
                                         // perspective
  int legal_moves_count;
  bool legal_moves_cached;
  uint64_t cached_board_hash;

  // Game modes
  bool dual_agent_self_play_mode;
  bool self_play_mode;

  // Complete game logging
  char complete_game_moves[1024][6]; // Store canonical UCI moves (e.g., "e2e4")
  int complete_game_action_count;
  char serialized_moves[1024]; // Comma-separated action IDs for efficient
                               // logging
  
  // Simple game logging tracking
  int steps_since_last_log;      // Steps in this environment since last game log
  int game_logging_frequency;    // Log games every N steps (from config)

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
  float cached_observation[1344];  // 21 * 8 * 8 board planes only
  bool observation_cached;
  uint64_t cached_observation_hash;
  PieceColor cached_observation_player;
} ChessContext;

// === PUFFERLIB ENVIRONMENT STRUCTURE ===
typedef struct CChess {
  Log log;
  int env_id;  // Add back env_id field
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
  bool use_piece_value_capture_rewards;
  float piece_value_reward_multiplier;

  // Debug settings
  bool debug_disable_mask;
  bool stockfish_enabled;

  // Chess context (pure C, no opaque pointer)
  ChessContext context;

  // Convenience pointer to avoid repeated dereferencing
  ChessContext *ctx;
} CChess;

// === ADDITIONAL BINDING FUNCTIONS ===
void enable_stockfish_black(CChess *env, const char *stockfish_cmd, int elo,
                            int search_ms);
void set_self_play_mode(CChess *env, bool enabled);
void set_dual_agent_self_play_mode(CChess *env, bool enabled);
void set_debug_disable_mask(CChess *env, bool enabled);

// === PUFFERLIB REQUIRED FUNCTIONS ===
void init(CChess *env);
void allocate(CChess *env);
void free_allocated(CChess *env);
void add_log(CChess *env);
void c_reset(CChess *env);
void c_step(CChess *env);
void c_render(CChess *env);
void c_close(CChess *env);

// === MODE SETTERS FOR COMPATIBILITY ===
void set_dual_agent_self_play_mode(CChess *env, bool enabled);
void set_self_play_mode(CChess *env, bool enabled);
void c_set_fen(CChess *env, const char *fen);

// === CHESS HELPER FUNCTIONS ===

// Material calculation using standard chess piece values
int calculate_material_value(ChessBoard *board, PieceColor color) {
  int total = 0;
  for (int i = 0; i < 64; i++) {
    Piece *p = &board->board[i];
    if (p->type != EMPTY && p->color == color) {
      switch (p->type) {
        case PAWN:   total += 1; break;
        case KNIGHT: total += 3; break;
        case BISHOP: total += 3; break;
        case ROOK:   total += 5; break;
        case QUEEN:  total += 9; break;
        case KING:   total += 0; break; // King has no material value
        default: break;
      }
    }
  }
  return total;
}

// Get individual piece value for capture rewards
int get_piece_value(PieceType piece_type) {
  switch (piece_type) {
    case PAWN:   return 1;
    case KNIGHT: return 3;
    case BISHOP: return 3;
    case ROOK:   return 5;
    case QUEEN:  return 9;
    case KING:   return 0; // King cannot be captured in normal play
    default:     return 0;
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
    sprintf(temp, "%d", action_id);
    strcat(ctx->serialized_moves, temp);
  }
}

// Write complete game to file for analysis
static void write_complete_game_to_file(ChessContext *ctx, int env_id) {
  // Re-enabled for debugging game logging functionality
  printf("[C++ DEBUG] write_complete_game_to_file called: env_id=%d, action_count=%d\n", 
         env_id, ctx->complete_game_action_count);
  
  if (ctx->complete_game_action_count == 0) {
    printf("[C++ DEBUG] No actions to log\n");
    return;
  }

  // Create directory if needed
  system("mkdir -p resources/chess/training_logs/complete_games");
  
  // Determine game result and termination reason
  char result_str[16] = "*";      // Default: incomplete/ongoing
  char termination[64] = "depth_limit";  // Default termination reason
  
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
    if (*p == '/' || *p == '-') *p = '_';
  }
  for (char *p = safe_termination; *p; p++) {
    if (*p == '/' || *p == '-') *p = '_';
  }
  
  sprintf(filename, "resources/chess/training_logs/complete_games/game_%d_%ld_%s_%s.pgn", 
          env_id, now, safe_result, safe_termination);
  
  FILE* file = fopen(filename, "w");
  if (!file) {
    printf("[Chess] Failed to open game log file: %s\n", filename);
    return;
  }
  
  // Write PGN header
  fprintf(file, "[Event \"PufferLib Training Game\"]\n");
  fprintf(file, "[Site \"Environment %d\"]\n", env_id);
  fprintf(file, "[Date \"%ld\"]\n", now);
  fprintf(file, "[White \"AI-White\"]\n");
  fprintf(file, "[Black \"AI-Black\"]\n");
  fprintf(file, "[Result \"%s\"]\n", result_str);
  fprintf(file, "[Termination \"%s\"]\n", termination);
  fprintf(file, "\n");
  
  // Write moves in algebraic notation
  int move_number = 1;
  for (int i = 0; i < ctx->complete_game_action_count; i++) {
    // Use the stored canonical UCI move directly
    const char* uci_move = ctx->complete_game_moves[i];
    
    // Simple UCI to algebraic (basic format: from-to, e.g. e2e4)
    if (i % 2 == 0) {
      fprintf(file, "%d. %s ", move_number, uci_move);
      if (i == ctx->complete_game_action_count - 1) fprintf(file, "\n");
    } else {
      fprintf(file, "%s ", uci_move);
      if (i % 4 == 1) fprintf(file, "\n");
      move_number++;
    }
  }
  
  if (ctx->complete_game_action_count % 2 == 1) fprintf(file, "\n");
  fprintf(file, "%s\n", result_str);
  
  fclose(file);
  printf("[Chess] Logged complete game to %s\n", filename);
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

  if (chess_is_legal_move(ctx, move)) {
    moves->moves[moves->count] = move;
    moves->count++;
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

        const Piece *target = get_piece_const(board, to.x, to.y);
        if (target && target->type != EMPTY) {
          if (target->color == them) {
            // Capture and stop
            ChessMove move = {from, to, EMPTY, false, false};
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
  // Check cache
  uint64_t current_hash = ctx->board.zobrist_hash;
  if (ctx->legal_moves_cached && ctx->cached_board_hash == current_hash) {
    PROFILE_STOP(profile_move_gen_uci_ticks)
    return ctx->legal_moves_count;
  }

  // Generate legal moves
  LegalMoves moves;
  chess_generate_legal_moves(ctx, &moves);

  // Convert to UCI strings
  ctx->legal_moves_count = 0;
  for (int i = 0; i < moves.count && i < 256; i++) {
    ChessMove move = moves.moves[i];
    if (move.promotion != EMPTY) {
      char promo_char = (move.promotion == QUEEN)    ? 'q'
                        : (move.promotion == ROOK)   ? 'r'
                        : (move.promotion == BISHOP) ? 'b'
                                                     : 'n';
      snprintf(ctx->legal_moves_buffer[ctx->legal_moves_count], 6, "%c%c%c%c%c",
               'a' + move.from.x, '1' + move.from.y, 'a' + move.to.x,
               '1' + move.to.y, promo_char);
    } else {
      snprintf(ctx->legal_moves_buffer[ctx->legal_moves_count], 5, "%c%c%c%c",
               'a' + move.from.x, '1' + move.from.y, 'a' + move.to.x,
               '1' + move.to.y);
    }
    ctx->legal_moves_count++;
  }

  // Cache the result
  ctx->legal_moves_cached = true;
  ctx->cached_board_hash = current_hash;

  PROFILE_STOP(profile_move_gen_uci_ticks)
  return ctx->legal_moves_count;
}

// === BOARD STATE MANIPULATION ===

static void init_board(ChessBoard *board) {
  memset(board, 0, sizeof(ChessBoard));

  // Set up starting position
  const char *start_fen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

  // Parse FEN (simplified)
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

  // Initialize Zobrist hash
  board->zobrist_hash = compute_zobrist_hash(board);
}

// === OBSERVATION COMPUTATION WITH PERSPECTIVE FLIPPING (OPTIMIZED) ===

// // Helper function to compute observation for a single agent
// static void compute_single_agent_observation(CChess *env, ChessContext *ctx, PieceColor player, int obs_offset) {
//   // printf("[OBSERVE] Computing observation for player %s at offset %d\n", 
//   //        (player == C_WHITE) ? "WHITE" : "BLACK", obs_offset);
//   // fflush(stdout);
//   int idx = 0;

//   // Clear the first 13 planes (all piece planes + empty squares)
//   memset(&env->observations[obs_offset], 0, 13 * 64 * sizeof(float));

//   // --- SINGLE PASS OVER THE BOARD ---
//   // Iterate through each square once to populate all piece-related planes.
//   for (int y_white_perspective = 0; y_white_perspective < 8;
//        y_white_perspective++) {
//     for (int x = 0; x < 8; x++) {
//       // Determine the actual y-coordinate based on the player's perspective
//       int y_actual = (player == C_WHITE) ? y_white_perspective
//                                          : (7 - y_white_perspective);
//       int square_index_actual = y_actual * 8 + x;
//       const Piece *p = &ctx->board.board[square_index_actual];

//       // Determine the observation y-coordinate (always from player's perspective)
//       int y_obs = y_white_perspective;
//       int obs_square_idx = y_obs * 8 + x;

//       if (p->type == EMPTY) {
//         // Plane 12: Empty squares
//         env->observations[obs_offset + 12 * 64 + obs_square_idx] = 1.0f;
//       } else {
//         int plane_offset = (p->color == player) ? 0 : 6;
//         // Piece type is 1-6, planes are 0-5 and 6-11
//         int piece_plane = p->type - 1;
//         env->observations[obs_offset + (plane_offset + piece_plane) * 64 + obs_square_idx] = 1.0f;
//       }
//     }
//   }

//   // --- NON-PIECE PLANES ---
//   // Now set idx to start after piece planes (13 planes * 64 squares each)
//   idx = 13 * 64;

//   // Repetition count plane (using actual position history)
//   int reps = get_position_count(ctx, ctx->board.zobrist_hash);
//   float rep_val = (reps >= 2) ? 1.0f : 0.0f; // Simplified: 0 for 1 rep, 1 for 2+ reps
//   for (int i = 0; i < 64; i++) {
//     env->observations[obs_offset + idx++] = rep_val;
//   }

//   // Side to move plane (always 0 from current player's perspective)
//   for (int i = 0; i < 64; i++) {
//     env->observations[obs_offset + idx++] = 0.0f;
//   }

//   // Halfmove clock plane
//   float halfmove_val = ctx->board.halfmove_clock / 100.0f; // Normalize to 0-1 range
//   for (int i = 0; i < 64; i++) {
//     env->observations[obs_offset + idx++] = halfmove_val;
//   }

//   // Castling rights planes (4 planes, flipped for black perspective)
//   uint8_t rights = ctx->board.castle_rights;
//   if (player == C_BLACK) {
//     // Flip castling rights for Black's perspective
//     uint8_t flipped = 0;
//     if (rights & 4) flipped |= 1; // BK -> WK
//     if (rights & 8) flipped |= 2; // BQ -> WQ
//     if (rights & 1) flipped |= 4; // WK -> BK
//     if (rights & 2) flipped |= 8; // WQ -> BQ
//     rights = flipped;
//   }

//   for (int i = 0; i < 4; i++) {
//     float castle_val = (rights & (1 << i)) ? 1.0f : 0.0f;
//     for (int j = 0; j < 64; j++) {
//       env->observations[obs_offset + idx++] = castle_val;
//     }
//   }

//   // En passant target square plane (flipped for black perspective)
//   int8_t ep_square = ctx->board.ep_square;
//   if (ep_square != -1 && player == C_BLACK) {
//     int ep_x = ep_square % 8;
//     int ep_y = ep_square / 8;
//     ep_square = (7 - ep_y) * 8 + ep_x;
//   }
//   for (int i = 0; i < 64; i++) {
//     env->observations[obs_offset + idx++] = (ep_square == i) ? 1.0f : 0.0f;
//   }

//   assert(idx == 1344); // 21 * 8 * 8

//   // --- LEGAL MOVE MASK ---
//   if (!ctx->legal_moves_cached) {
//     chess_generate_legal_moves_uci(ctx);
//   }
  
//   // DEBUG: Always show legal move count
//   printf("[DEBUG] Legal moves generated: %d for player %s (turn: %s, cached: %s)\n",
//          ctx->legal_moves_count,
//          (player == C_WHITE) ? "WHITE" : "BLACK",
//          (ctx->board.to_move == C_WHITE) ? "WHITE" : "BLACK",
//          ctx->legal_moves_cached ? "YES" : "NO");
  
//   if (ctx->legal_moves_count == 0) {
//     printf("[DEBUG] WARNING: No legal moves generated for player %s!\n", 
//            (player == C_WHITE) ? "WHITE" : "BLACK");
//     printf("[DEBUG] Board state: to_move=%s, halfmove=%d, fullmove=%d, cached=%s\n",
//            (ctx->board.to_move == C_WHITE) ? "WHITE" : "BLACK",
//            ctx->board.halfmove_clock, ctx->board.fullmove_number,
//            ctx->legal_moves_cached ? "YES" : "NO");
    
//     // Try to force regeneration
//     printf("[DEBUG] Forcing legal move regeneration...\n");
//     ctx->legal_moves_cached = false;
//     chess_generate_legal_moves_uci(ctx);
//     printf("[DEBUG] After forced regen: %d legal moves\n", ctx->legal_moves_count);
//   }

//   // CRITICAL FIX: Only set legal moves for the player whose turn it is
//   // In dual-agent mode, only the current player should have legal moves
//   PieceColor current_player = ctx->board.to_move;
//   bool is_player_turn = (player == current_player);

// //   // Add diagnostic logging when clearing mask for inactive player
// //   if (!is_player_turn) {
// //     printf("[DIAGNOSTIC] Clearing mask for INACTIVE player %s (turn is %s) at obs_offset %d\n",
// //            (player == C_WHITE) ? "WHITE" : "BLACK",
// //            (current_player == C_WHITE) ? "WHITE" : "BLACK",
// //            obs_offset);
// //   }

//   // Clear mask
//   printf("[DEBUG] Clearing action mask for player %s at offset %d (idx=%d to %d)\n", 
//          (player == C_WHITE) ? "WHITE" : "BLACK", obs_offset, obs_offset + idx, 
//          obs_offset + idx + TOTAL_CHESS_ACTIONS - 1);
//   memset(&env->observations[obs_offset + idx], 0, TOTAL_CHESS_ACTIONS * sizeof(float));
  
//   printf("[DEBUG] Action mask: player=%s, current_turn=%s, is_player_turn=%s\n",
//          (player == C_WHITE) ? "WHITE" : "BLACK",
//          (current_player == C_WHITE) ? "WHITE" : "BLACK", 
//          is_player_turn ? "YES" : "NO");

//   if (env->debug_disable_mask) {
//     for (int i = 0; i < TOTAL_CHESS_ACTIONS; i++) {
//       env->observations[obs_offset + idx + i] = 1.0f;
//     }
//   } else if (is_player_turn) {
//     // Only create mask if it's this player's turn
//     for (int i = 0; i < ctx->legal_moves_count; i++) {
//       const char *canonical_uci = ctx->legal_moves_buffer[i];
//       char perspective_uci[6];

//       // The policy always sees the board as if it were white.
//       // So for black, we must flip the canonical UCI move to match the policy's perspective.
//       if (player == C_BLACK) {
//         flip_uci_for_black_perspective(canonical_uci, perspective_uci);
//         printf("[DEBUG BLACK] Canonical: %s -> Perspective: %s\n", canonical_uci, perspective_uci);
//       } else {
//         strcpy(perspective_uci, canonical_uci);
//       }

//       int action_id = uci_to_action_id(perspective_uci);
//       if (action_id >= 0) {
//         int mask_idx = obs_offset + idx + action_id;
//         env->observations[mask_idx] = 1.0f;
//         float verify_value = env->observations[mask_idx];
//         printf("[DEBUG] Legal move %s -> action %d (masked) at idx=%d for player %s, verify=%.1f\n", 
//                perspective_uci, action_id, mask_idx,
//                (player == C_WHITE) ? "WHITE" : "BLACK", verify_value);
//       } else {
//         printf("[DEBUG] UCI move %s NOT FOUND in action mapping!\n", perspective_uci);
//       }
//     }
//   }
//   // If it's not this player's turn, the mask remains all zeros (cleared above)
// }

// Helper function to compute observation for a single agent
static void compute_single_agent_observation(CChess *env, ChessContext *ctx,
                                             PieceColor player,
                                             int obs_offset) {
//   printf("[DEBUG_OBS] compute_single_agent_observation called for %s at offset %d\n",
//          (player == C_WHITE) ? "WHITE" : "BLACK", obs_offset);
  uint64_t current_hash = ctx->board.zobrist_hash;
  
  // PERFORMANCE OPTIMIZATION: Check if board observation is cached
  // DISABLED FOR DEBUGGING: Force fresh observation generation
  if (false && ctx->observation_cached && 
      ctx->cached_observation_hash == current_hash && 
      ctx->cached_observation_player == player) {
    // Use cached board observation (first 1344 floats)
    memcpy(&env->observations[obs_offset], ctx->cached_observation, 1344 * sizeof(float));
  } else {
    // Compute fresh board observation
    memset(&env->observations[obs_offset], 0, 1344 * sizeof(float));
    int idx = 0;

    // --- SINGLE PASS OVER THE BOARD (Correct) ---
    for (int y_white_perspective = 0; y_white_perspective < 8;
         y_white_perspective++) {
      for (int x = 0; x < 8; x++) {
        int y_actual =
            (player == C_WHITE) ? y_white_perspective : (7 - y_white_perspective);
        int square_index_actual = y_actual * 8 + x;
        const Piece *p = &ctx->board.board[square_index_actual];
        int obs_square_idx = y_white_perspective * 8 + x;

        if (p->type == EMPTY) {
          env->observations[obs_offset + 12 * 64 + obs_square_idx] = 1.0f;
        } else {
          int plane_offset = (p->color == player) ? 0 : 6;
          int piece_plane = p->type - 1;
          env->observations[obs_offset + (plane_offset + piece_plane) * 64 +
                            obs_square_idx] = 1.0f;
        }
      }
    }
    
    // Cache the computed board observation
    memcpy(ctx->cached_observation, &env->observations[obs_offset], 1344 * sizeof(float));
    ctx->observation_cached = true;
    ctx->cached_observation_hash = current_hash;
    ctx->cached_observation_player = player;
  }
  
  int idx = 1344; // Start index for the legal move mask

  // --- LEGAL MOVE MASK (Corrected Logic) ---

  // *** THE FIX: Force legal moves to be regenerated for the current player ***
  // This prevents using a stale cache from the other player's turn.
  ctx->legal_moves_cached = false;
  chess_generate_legal_moves_uci(ctx);

  PieceColor current_player_turn = ctx->board.to_move;
  bool is_player_turn = (player == current_player_turn);
  
  // printf("[ACTION_MASK_DEBUG] Player=%s, current_turn=%s, is_player_turn=%s\n", 
  //        (player == C_WHITE) ? "WHITE" : "BLACK",
  //        (current_player_turn == C_WHITE) ? "WHITE" : "BLACK", 
  //        is_player_turn ? "YES" : "NO");

  // Always clear the agent's mask first
  memset(&env->observations[obs_offset + idx], 0,
         TOTAL_CHESS_ACTIONS * sizeof(float));

  if (env->debug_disable_mask) {
    // Debug mode to make all moves legal
    for (int i = 0; i < TOTAL_CHESS_ACTIONS; i++) {
      env->observations[obs_offset + idx + i] = 1.0f;
    }
  } else if (ctx->dual_agent_self_play_mode || is_player_turn) {
    // In dual-agent mode, both agents get action masks for moves from their perspective
    // In single-agent mode, only populate mask when it's the player's turn
    // printf("[ACTION_MASK_FIX] %s agent: dual_agent_mode=%s, is_player_turn=%s, will_populate_mask=YES\n",
    //        (player == C_WHITE) ? "WHITE" : "BLACK",
    //        ctx->dual_agent_self_play_mode ? "YES" : "NO",
    //        is_player_turn ? "YES" : "NO");
    // printf("[ACTION_MASK_DEBUG] Computing mask for %s at obs_offset=%d, legal_moves=%d\n", 
    //        (player == C_WHITE) ? "WHITE" : "BLACK", obs_offset, ctx->legal_moves_count);
    
    int valid_actions_set = 0;
    for (int i = 0; i < ctx->legal_moves_count; i++) {
      const char *canonical_uci = ctx->legal_moves_buffer[i];
      char perspective_uci[6];

      if (player == C_BLACK) {
        // For black, flip the canonical move to the policy's perspective
        flip_uci_for_black_perspective(canonical_uci, perspective_uci);
      } else {
        strcpy(perspective_uci, canonical_uci);
      }

      int action_id = uci_to_action_id(perspective_uci);
      if (action_id >= 0) {
        env->observations[obs_offset + idx + action_id] = 1.0f;
        valid_actions_set++;
        // if (valid_actions_set <= 3) {  // Log first 3 action IDs for debugging
        //   printf("[ACTION_ID_DEBUG] %s: %s -> action_id %d\n", 
        //          (player == C_WHITE) ? "WHITE" : "BLACK", perspective_uci, action_id);
        // }
        if (player == C_BLACK && i < 5) {  // Log first 5 for black
        //   printf("[ACTION_MASK_DEBUG] BLACK: %s -> %s -> action %d (masked at idx %d)\n", 
        //          canonical_uci, perspective_uci, action_id, obs_offset + idx + action_id);
        }
      } else if (player == C_BLACK && i < 5) {
        // printf("[ACTION_MASK_DEBUG] BLACK: %s -> %s -> NO ACTION ID FOUND\n", 
        //        canonical_uci, perspective_uci);
      }
    }
    // printf("[ACTION_MASK_DEBUG] %s: Set %d valid actions in mask\n", 
    //        (player == C_WHITE) ? "WHITE" : "BLACK", valid_actions_set);
  } else {
    // printf("[ACTION_MASK_FIX] %s agent: dual_agent_mode=%s, is_player_turn=%s, will_populate_mask=NO (mask stays empty)\n",
    //        (player == C_WHITE) ? "WHITE" : "BLACK",
    //        ctx->dual_agent_self_play_mode ? "YES" : "NO",
    //        is_player_turn ? "YES" : "NO");
  }
  // In dual-agent mode, both agents always get valid action masks
  // In single-agent mode, non-active players get empty masks (all zeros)
}

// Corrected version of the observation generation orchestrator.
// This function should replace the old `compute_observation_with_perspective`.

// COLOR MONITORING: Validates observation data integrity at generation point
void validate_chess_observation_integrity(CChess *env, ChessContext *ctx, PieceColor player, int obs_offset) {
  // SENTINEL 1: Check observation buffer bounds
  // In the new single-agent-view architecture, the offset for the active player
  // is ALWAYS 0.
  const int expected_offset = 0; // <-- NEW, CORRECT LOGIC
  if (obs_offset != expected_offset) {
    printf("[MONITOR_FATAL] Chess.h observation offset mismatch!\n");
    printf("  Expected %s at offset %d, got offset %d\n",
           (player == C_WHITE) ? "WHITE" : "BLACK", expected_offset,
           obs_offset);
    printf("  This indicates an error in the calling function.\n");
    printf("  FIX: Ensure all calls to compute_single_agent_observation pass 0 "
           "as the offset.\n");
    exit(1);
  }

  // SENTINEL 2: Validate observation content signature  
  float *obs = &env->observations[obs_offset];
  float board_sum = 0.0f;
  for (int i = 0; i < 1344; i++) board_sum += obs[i];
  
  float mask_sum = 0.0f;
  for (int i = 1344; i < 3312; i++) mask_sum += obs[i];

  if (board_sum < 1.0f || mask_sum < 1.0f) {
    printf("[MONITOR_FATAL] Chess.h observation content invalid!\n");
    printf("  %s observation at offset %d: board_sum=%.3f mask_sum=%.3f\n",
           (player == C_WHITE) ? "WHITE" : "BLACK", obs_offset, board_sum, mask_sum);
    printf("  Board sum should be >1 (pieces present), mask sum should be >1 (legal moves exist).\n");
    printf("  FIX: Check compute_single_agent_observation() is writing correct data.\n");
    exit(1);
  }

  // SENTINEL 3: Validate perspective correctness
  PieceColor current_turn = ctx->board.to_move;
//   printf("[MONITOR_OK] Chess.h: Generated %s observation (offset=%d, board_sum=%.1f, mask_sum=%.0f) on %s's turn\n",
//          (player == C_WHITE) ? "WHITE" : "BLACK", obs_offset, board_sum, mask_sum,
//          (current_turn == C_WHITE) ? "WHITE" : "BLACK");
}

// void compute_observation_with_perspective(CChess *env, ChessContext *ctx) {
//   PROFILE_START(profile_compute_obs_ticks)
  
//   // printf("[COMPUTE_OBS] Called for turn=%s, fullmove=%d\n",
//   //        (ctx->board.to_move == C_WHITE) ? "WHITE" : "BLACK", ctx->board.fullmove_number);

//   // In a dual-agent environment, we must generate an observation for BOTH agents
//   // every time the state changes. Each agent gets its own slice of the observation buffer.
//   // The 'observations' pointer in the CChess struct points to the start of the memory
//   // block for this specific game environment.

//   if (ctx->self_play_mode && ctx->dual_agent_self_play_mode) {
//     // This is the standard 2-player self-play mode.

//     // Define the size of a single agent's observation to calculate offsets.
//     const int single_obs_size = 3312; // 1344 board + 1968 mask

//     // === Generate Observation for Agent 0 (White) ===
//     // This observation will be written to the first slice of the environment's
//     // observation buffer, starting at offset 0.
//     int white_obs_offset = 0;
//     compute_single_agent_observation(env, ctx, C_WHITE, white_obs_offset);
//     validate_chess_observation_integrity(env, ctx, C_WHITE, white_obs_offset);

//     // === Generate Observation for Agent 1 (Black) ===
//     // This observation will be written to the second slice, starting immediately
//     // after Agent 0's data.
//     int black_obs_offset = single_obs_size;
//     compute_single_agent_observation(env, ctx, C_BLACK, black_obs_offset);
//     validate_chess_observation_integrity(env, ctx, C_BLACK, black_obs_offset);

//   } else {
//     // This logic handles a single-agent mode (e.g., playing vs. an engine).
//     // In this case, there is only one agent, and it always writes to offset 0.
//     PieceColor current_player = ctx->board.to_move;
//     compute_single_agent_observation(env, ctx, current_player, 0);
//   }

//   PROFILE_STOP(profile_compute_obs_ticks)
// }

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
  board->to_move = (us == C_WHITE) ? C_BLACK : C_WHITE;
  if (us == C_BLACK) {
    board->fullmove_number++;
  }

  // XOR in new state
  board->zobrist_hash ^= zobrist_castle_rights[board->castle_rights];
  if (board->ep_square >= 0) {
    board->zobrist_hash ^= zobrist_en_passant[board->ep_square];
  }

  // Add current position to history for threefold repetition detection
  add_position_to_history(ctx, board->zobrist_hash);

  // Clear caches
  ctx->legal_moves_cached = false;
  ctx->observation_cached = false;
  ctx->step_count++;
  // Add to complete game log - store the canonical UCI move
  if (ctx->complete_game_action_count < 1024) {
    strcpy(ctx->complete_game_moves[ctx->complete_game_action_count], uci_str);
    ctx->complete_game_action_count++;
    // printf("[DEBUG] Recorded move %s (total: %d)\n", uci_str, ctx->complete_game_action_count);
  } else {
    printf("[DEBUG] Failed to record move: count=%d, max=1024\n", 
           ctx->complete_game_action_count);
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

void init(CChess *env) {
  memset(&env->context, 0, sizeof(ChessContext));
  memset(&env->log, 0, sizeof(Log));

  // Set up convenience pointer to avoid repeated dereferencing
  env->ctx = &env->context;
  env->env_id = global_env_counter++;  // Simple counter

  init_board(&env->context.board);
  env->context.dual_agent_self_play_mode = true; // Default to self-play

  // Copy reward config to context
  env->context.c_reward_valid = env->reward_valid;
  env->context.c_reward_invalid_white = env->reward_invalid_white;
  env->context.c_reward_invalid_black = env->reward_invalid_black;
  env->context.c_reward_white_captures_enemy_piece =
      env->reward_white_captures_enemy_piece;
  env->context.c_reward_black_captures_enemy_piece =
      env->reward_black_captures_enemy_piece;
  env->context.c_reward_max_depth_termination = env->reward_max_depth_termination;
  env->context.c_use_piece_value_capture_rewards = env->use_piece_value_capture_rewards;
  env->context.c_piece_value_reward_multiplier = env->piece_value_reward_multiplier;
  env->context.c_reward_draw = env->reward_draw;
  env->context.c_reward_win_white = env->reward_win_white;
  env->context.c_reward_win_black = env->reward_win_black;
  env->context.c_reward_loss_white = env->reward_loss_white;
  env->context.c_reward_loss_black = env->reward_loss_black;
  env->context.c_reward_check_white = env->reward_check_white;
  env->context.c_reward_check_black = env->reward_check_black;
  env->context.c_reward_material_diff_white = env->reward_material_diff_white;
  env->context.c_reward_material_diff_black = env->reward_material_diff_black;
}

void allocate(CChess *env) {
  // Allocate RL interface arrays for PufferLib
  // Chess has 2 players but typically trains as single agent with perspective
  // flipping
  const int num_players = 2;
  const int obs_size =
      3312; // 21*8*8 board planes + 1968 action mask = 1344 + 1968

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

void c_reset(CChess *env) {
  printf("[C_RESET DEBUG] c_reset called - resetting step_count from %d to 0\n", env->context.step_count);
  init_board(&env->context.board);

  // Reset terminals and rewards for both agents
  env->terminals[0] = 0;
  env->terminals[1] = 0;
  env->rewards[0] = 0.0f;
  env->rewards[1] = 0.0f;

  // Reset episode tracking
  env->context.step_count = 0;
  env->context.episode_return_white = 0.0f;
  env->context.episode_return_black = 0.0f;
  env->context.complete_game_action_count = 0;
  env->context.serialized_moves[0] =
      '\0'; // Initialize serialized_moves buffer to empty
  env->context.steps_since_last_log = 0;
  
  // Don't reset game logging frequency - it's set once at init
  // env->context.game_logging_frequency = 500000;
  // DEBUG: Explicitly preserve the logging frequency that was set during init
  if (env->context.game_logging_frequency == 0) {
    // This shouldn't happen - frequency should be preserved from init
    printf("[C++ DEBUG] WARN: Reset detected zero frequency, but config should preserve it\n");
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

  // Clear caches
  env->context.legal_moves_cached = false;
  env->context.observation_cached = false;

  // Clear position history
  memset(&env->context.position_history, 0, sizeof(PositionHistory));

  // Add starting position to history for threefold repetition detection
  add_position_to_history(&env->context, env->context.board.zobrist_hash);

  // Compute initial observation
  compute_observation_with_perspective(env, &env->context);
}

void c_step(CChess *env) {
  PROFILE_START(profile_c_step_ticks)
  
  // Safety checks
  if (!env) {
    printf("[C_STEP DEBUG] ERROR: env is NULL\n");
    return;
  }
  
  printf("[C_STEP DEBUG] env=%p, step_count=%d, max_depth=%d\n", (void*)env, env->context.step_count, env->max_depth);
  
  // Increment step counter for game logging
  env->context.steps_since_last_log++;

//   // In self-play mode: agent 0 = white, agent 1 = black
//   // Get action from the agent whose turn it is
//   PieceColor current_player = env->context.board.to_move;
//   int agent_idx = (current_player == C_WHITE) ? 0 : 1;
  
//   printf("[C_STEP_ENTRY] Turn: %s (agent %d), fullmove: %d, halfmove: %d\n",
//          (current_player == C_WHITE) ? "WHITE" : "BLACK", agent_idx,
//          env->context.board.fullmove_number, env->context.board.halfmove_clock);
  
//   // CRITICAL: In dual-agent mode, only process actions for the current player
//   // SAFEGUARD: Handle training loop calling wrong agent
//   int correct_agent_idx = agent_idx;
//   if (env->context.dual_agent_self_play_mode) {
//     // Determine which agent should actually move based on board state
//     int expected_agent = (current_player == C_WHITE) ? 0 : 1;
    
//     if (agent_idx != expected_agent) {
//       printf("[TRAINING_LOOP_FIX] Board says %s's turn (agent %d), but called with agent %d. Using correct agent.\n",
//              (current_player == C_WHITE) ? "WHITE" : "BLACK", expected_agent, agent_idx);
//       correct_agent_idx = expected_agent;
//     }
//   }
  
//   // Use the action from the correct agent (the one whose turn it actually is)
//   int action_idx = env->actions[correct_agent_idx];

  // In a single-agent-view architecture, the Python wrapper ALWAYS places the
  // action for the current player in the first slot of the actions buffer.
  // We no longer need to determine the agent_idx based on color.
  int agent_idx = 0;

  // The action is now correctly read from env->actions[0] for BOTH White and
  // Black.
  int action_idx = env->actions[agent_idx];

  // CRITICAL FIX: Validate that the chosen action is actually in the action mask
  // Calculate the correct observation offset for the CURRENT player.
  const int single_obs_size = 3312; // 1344 board + 1968 mask
  int obs_offset = 0; // <-- NEW, CORRECT LOGIC
  int mask_start_idx = 1344; // Start of action mask in observation
  
  // Generate fresh legal moves to validate against
  chess_generate_legal_moves_uci(&env->context);
  
  // Check if the chosen action corresponds to a legal move
  bool action_is_legal = false;
  int fallback_action = -1;
  
  if (action_idx >= 0 && action_idx < TOTAL_CHESS_ACTIONS) {
    const char* chosen_uci = ACTION_ID_TO_UCI[action_idx];
    
    // Check if this action corresponds to any legal move
    for (int i = 0; i < env->context.legal_moves_count; i++) {
      const char* legal_uci = env->context.legal_moves_buffer[i];
      char perspective_uci[6];
      
      // For BLACK, we need to flip the canonical move to match the action space
      if (env->context.board.to_move == C_BLACK) {
        flip_uci_for_black_perspective(legal_uci, perspective_uci);
      } else {
        strcpy(perspective_uci, legal_uci);
      }
      
      if (strcmp(chosen_uci, perspective_uci) == 0) {
        action_is_legal = true;
        break;
      }
      
      // Store first legal action as fallback
      if (fallback_action == -1) {
        fallback_action = uci_to_action_id(perspective_uci);
      }
    }
  }

//   // NEW, CLEANER LOGGING
//   printf("[C_STEP_DEBUG] %s's turn: Chose action %d (%s) from agent 0. "
//          "legal=%s, legal_moves=%d\n",
//          (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK", action_idx,
//          ACTION_ID_TO_UCI[action_idx], action_is_legal ? "YES" : "NO",
//          env->context.legal_moves_count);

  if (!action_is_legal) {
    if (fallback_action >= 0) {
    //   printf("[ACTION_DEBUG] %s agent chose illegal action %d (%s), using fallback action %d (%s)\n", 
    //          (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK", action_idx, ACTION_ID_TO_UCI[action_idx],
    //          fallback_action, ACTION_ID_TO_UCI[fallback_action]);
      action_idx = fallback_action;
    } else {
    //   printf("[ERROR] No legal actions available for %s agent!\n", 
    //          (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK");
      PROFILE_STOP(profile_c_step_ticks)
      return;
    }
  }
  
//   printf("[ACTION_DEBUG] %s's turn (agent %d) chose action %d (%s)\n", 
//          (current_player == C_WHITE) ? "WHITE" : "BLACK", agent_idx, 
//          action_idx, ACTION_ID_TO_UCI[action_idx]);

  // Clear all agent rewards and terminals
  for (int i = 0; i < 2; i++) {
    env->rewards[i] = 0.0f;
    env->terminals[i] = 0;
  }

  // Validate action before executing (especially important for human input)
  if (action_idx < 0 || action_idx >= TOTAL_CHESS_ACTIONS) {
    // printf("[ERROR] Invalid action ID: %d\n", action_idx);
    return;
  }
  
  // --- START OF NEW VALIDATION LOGIC ---
  
  // 1. Get the UCI string for the action chosen by the policy.
  // ACTION_ID_TO_UCI always represents moves in white's perspective coordinate system.
  const char *uci_move_white_perspective = ACTION_ID_TO_UCI[action_idx];
  char uci_move_canonical[6];
  
  if (env->context.board.to_move == C_BLACK) {
    // The black agent chose this action based on its flipped perspective.
    // The action maps to a move in white perspective coordinates, but since
    // the black agent sees the board flipped, this move should be interpreted
    // as being from black's perspective and flipped to canonical coordinates.
    flip_uci_for_black_perspective(uci_move_white_perspective, uci_move_canonical);
  } else {
    // For white, the white perspective move IS the canonical move.
    strcpy(uci_move_canonical, uci_move_white_perspective);
  }

  // 2. Generate the definitive list of legal moves for the CURRENT board state.
  //    This check uses the ground-truth board, not the observation buffer.
  chess_generate_legal_moves_uci(&env->context);

  // 3. Check if the chosen move is in the freshly generated list.
  bool is_action_legal = false;
  for (int i = 0; i < env->context.legal_moves_count; i++) {
    if (strcmp(env->context.legal_moves_buffer[i], uci_move_canonical) == 0) {
      is_action_legal = true;
      break;
    }
  }

  // 4. Validate against the ground truth.
  if (!is_action_legal) {
    const char* turn_color = (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK";
    printf("[ERROR] Illegal move attempted (ground truth validation): action %d (%s) - %s's turn (agent %d)\n", 
           action_idx, uci_move_canonical, turn_color, agent_idx);
    
    // Log the actual legal moves for debugging
    printf("[DEBUG] Agent %d has %d actual legal moves:\n", agent_idx, env->context.legal_moves_count);
    for (int i=0; i < env->context.legal_moves_count && i < 10; i++) { // Print first 10
        printf("  - %s\n", env->context.legal_moves_buffer[i]);
    }

    // Invalidate the move, penalize the agent, and end the step without applying the move.
    // Note: You may want to assign a penalty here. For now, we just return.
    if (env->context.board.to_move == C_WHITE) {
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

//   // Apply the move
//   printf("[MOVE_APPLY] Applying move %s for %s\n", uci_move_canonical, 
//          (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK");
  bool move_applied = apply_uci_move(&env->context, uci_move_canonical);
//   printf("[MOVE_APPLY] Move applied successfully: %s, new turn: %s\n", 
//          move_applied ? "YES" : "NO",
//          (env->context.board.to_move == C_WHITE) ? "WHITE" : "BLACK");

  // Assign rewards
  env->rewards[agent_idx] += env->context.c_reward_valid;
  env->context.accumulated_reward_valid += env->context.c_reward_valid;
  
  // Apply material advantage rewards every step
  int white_material = calculate_material_value(&env->context.board, C_WHITE);
  int black_material = calculate_material_value(&env->context.board, C_BLACK);
  int material_diff = white_material - black_material;  // Positive when WHITE ahead
  
  // Reward based on material advantage: positive for advantage, negative for disadvantage
  float white_material_reward = material_diff * env->context.c_reward_material_diff_white;
  float black_material_reward = -material_diff * env->context.c_reward_material_diff_black;
  
  // Apply material advantage rewards every step
  env->rewards[0] += white_material_reward;  // WHITE gets + for advantage, - for disadvantage
  env->rewards[1] += black_material_reward;  // BLACK gets + for advantage, - for disadvantage
  
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
      capture_reward = piece_value * env->context.c_piece_value_reward_multiplier;
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
      env->context.accumulated_reward_white_captures_enemy_piece += capture_reward;
    } else {
      env->context.accumulated_reward_black_captures_enemy_piece += capture_reward;
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

  // Accumulate episode returns for both agents
  // Both agents get the same reward since they're on the same team (shared observation)
  float shared_reward = env->rewards[agent_idx];
  env->rewards[0] = shared_reward;
  env->rewards[1] = shared_reward;
  
  env->context.episode_return_white += env->rewards[0];
  env->context.episode_return_black += env->rewards[1];

  // Check for game over conditions
  printf("[TERMINATION DEBUG] Checking game over: step_count=%d, max_depth=%d\n", env->context.step_count, env->max_depth);
  bool game_over = false;
  bool any_legal_move_exists = false;
  chess_generate_legal_moves_yield(&env->context, first_move_callback,
                                   &any_legal_move_exists);

  if (!any_legal_move_exists) {
    game_over = true;
    if (is_in_check(&env->context.board, env->context.board.to_move)) {
      // CHECKMATE
      if (env->context.board.to_move == C_WHITE) { // White is checkmated (black won)
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
        env->context.accumulated_reward_loss_white += env->context.c_reward_loss_white;
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
        env->context.accumulated_reward_loss_black += env->context.c_reward_loss_black;
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
    printf("[TERMINATION DEBUG] MAX DEPTH REACHED: step_count=%d >= max_depth=%d\n", env->context.step_count, env->max_depth);
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
    env->log.complete_game_move_count =
        (float)env->context.complete_game_action_count;
    add_log(env);
    
    // Notify UI about game end via function call (before auto-reset clears counters)
    void notify_game_end(int white_won, int black_won, int is_draw);
    notify_game_end(env->context.c_white_win > 0, env->context.c_black_win > 0, env->context.c_game_drawn > 0);

    // Check if we should log this complete game BEFORE reset - ONLY for env_id 512 (first active env)
    if (env->env_id == 512) {
      // printf("[C++ DEBUG] Game over: env_id=%d, steps=%d, freq=%d, actions=%d\n", 
      //        env->env_id, env->context.steps_since_last_log, env->context.game_logging_frequency,
      //        env->context.complete_game_action_count);
      if (env->context.game_logging_frequency > 0 && env->context.steps_since_last_log >= env->context.game_logging_frequency) {
        printf("[C++ DEBUG] Logging game from env %d!\n", env->env_id);
        write_complete_game_to_file(&env->context, env->env_id);
        env->context.steps_since_last_log = 0; // Reset counter
      } else {
        // printf("[C++ DEBUG] Not logging: freq=%d, steps=%d\n", 
        //        env->context.game_logging_frequency, env->context.steps_since_last_log);
      }
    } else {
      // Still do the frequency check for all envs, just don't log
      if (env->context.game_logging_frequency > 0 && env->context.steps_since_last_log >= env->context.game_logging_frequency) {
        env->context.steps_since_last_log = 0; // Reset counter
      }
    }
    
    // Debug: Always print for any env that completes a game
    if (env->context.complete_game_action_count > 0) {
    //   printf("[C++ DEBUG] Game completed in env %d with %d actions (steps_since_last_log=%d)\n", 
    //          env->env_id, env->context.complete_game_action_count, env->context.steps_since_last_log);
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
    compute_observation_with_perspective(env, &env->context);
  }

  PROFILE_STOP(profile_c_step_ticks)
}


// === PUFFERLIB LOGGING FUNCTION ===
void add_log(CChess *env) {

  // Aggregate counters into log structure using = for PufferLib (CRITICAL!)
  env->log.episode_length += (float)env->context.step_count;
  env->log.episode_return +=
      env->context.episode_return_white + env->context.episode_return_black;
  env->log.episode_return_white += env->context.episode_return_white;
  env->log.episode_return_black += env->context.episode_return_black;

  // Reward aggregates (from accumulated counters during this game)
  env->log.reward_valid += env->context.accumulated_reward_valid;
  env->log.reward_white_captures_enemy_piece +=
      env->context.accumulated_reward_white_captures_enemy_piece;
  env->log.reward_black_captures_enemy_piece +=
      env->context.accumulated_reward_black_captures_enemy_piece;
  env->log.reward_draw += env->context.accumulated_reward_draw;
  env->log.reward_win_white += env->context.accumulated_reward_win_white;
  env->log.reward_win_black += env->context.accumulated_reward_win_black;
  env->log.reward_loss_white += env->context.accumulated_reward_loss_white;
  env->log.reward_loss_black += env->context.accumulated_reward_loss_black;
  env->log.reward_draw_white += env->context.accumulated_reward_draw_white;
  env->log.reward_draw_black += env->context.accumulated_reward_draw_black;
  env->log.reward_check_white += env->context.accumulated_reward_check_white;
  env->log.reward_check_black += env->context.accumulated_reward_check_black;
  env->log.reward_material_diff_white +=
      env->context.accumulated_reward_material_diff_white;
  env->log.reward_material_diff_black +=
      env->context.accumulated_reward_material_diff_black;
  env->log.stockfish_eval += env->context.accumulated_stockfish_eval;

  // Game outcome counters (use incremental values from current game)
  env->log.white_win += (float)env->context.c_white_win;
  env->log.white_loss += (float)env->context.c_white_loss;
  env->log.black_win += (float)env->context.c_black_win;
  env->log.black_loss += (float)env->context.c_black_loss;
  env->log.game_drawn += (float)env->context.c_game_drawn;
  env->log.stalemate += (float)env->context.c_stalemate;
  env->log.insufficient_material += (float)env->context.c_insufficient_material;
  env->log.threefold_repetition += (float)env->context.c_threefold_repetition;
  env->log.fifty_move_rule += (float)env->context.c_fifty_move_rule;
  env->log.max_depth += (float)env->context.c_max_depth;
  env->log.white_checkmated += (float)env->context.c_white_checkmated;
  env->log.black_checkmated += (float)env->context.c_black_checkmated;

  // Move statistics
  env->log.white_moves += (float)env->context.c_white_moves;
  env->log.black_moves += (float)env->context.c_black_moves;
  env->log.valid_moves += (float)env->context.c_valid_moves;
  env->log.invalid_moves_white += (float)env->context.c_invalid_moves_white;
  env->log.invalid_moves_black += (float)env->context.c_invalid_moves_black;

  // Castling and special moves
  env->log.en_passant_white += (float)env->context.c_en_passant_white;
  env->log.en_passant_black += (float)env->context.c_en_passant_black;
  env->log.white_castle_kingside += (float)env->context.c_white_castle_kingside;
  env->log.white_castle_queenside +=
      (float)env->context.c_white_castle_queenside;
  env->log.black_castle_kingside += (float)env->context.c_black_castle_kingside;
  env->log.black_castle_queenside +=
      (float)env->context.c_black_castle_queenside;

  // Promotion statistics
  env->log.white_promotion_count += (float)env->context.c_white_promotion_count;
  env->log.white_promotion_knight +=
      (float)env->context.c_white_promotion_knight;
  env->log.white_promotion_bishop +=
      (float)env->context.c_white_promotion_bishop;
  env->log.white_promotion_rook += (float)env->context.c_white_promotion_rook;
  env->log.white_promotion_queen += (float)env->context.c_white_promotion_queen;
  env->log.black_promotion_count += (float)env->context.c_black_promotion_count;
  env->log.black_promotion_knight +=
      (float)env->context.c_black_promotion_knight;
  env->log.black_promotion_bishop +=
      (float)env->context.c_black_promotion_bishop;
  env->log.black_promotion_rook += (float)env->context.c_black_promotion_rook;
  env->log.black_promotion_queen += (float)env->context.c_black_promotion_queen;

  // Calculate performance metrics after aggregation
  float total_games =
      env->log.white_win + env->log.white_loss + env->log.game_drawn;
  if (total_games > 0) {
    env->log.perf = env->log.white_win / total_games; // White win rate
  }
  env->log.score =
      env->log.white_win - env->log.white_loss; // Win-loss difference

  // Increment n (must be last for PufferLib aggregation)
  env->log.n += 1.0f;
}

void c_render(CChess *env) {

  printf("\n  +---+---+---+---+---+---+---+---+\n");
  for (int y = 7; y >= 0; y--) {
    printf("%d |", y + 1);
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

      printf(" %c |", piece_char);
    }
    printf("\n  +---+---+---+---+---+---+---+---+\n");
  }
  printf("    a   b   c   d   e   f   g   h\n");
  printf("\nTo move: %s\n",
         (env->context.board.to_move == C_WHITE) ? "White" : "Black");
  printf("Step: %d\n", env->context.step_count);
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
  env->context.self_play_mode = enabled;  // Also set self_play_mode
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
  // 2. UCI protocol implementation
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
  // TODO: Implement FEN parsing
  init_board(&env->context.board);
  compute_observation_with_perspective(env, &env->context);
}

// === PROFILING REPORT FUNCTION ===
void c_print_profile_data() {
  profile_total_ticks = profile_c_step_ticks;
  if (profile_total_ticks == 0) {
    printf("No profiling data collected yet.\n");
    return;
  }

  printf("\n--- Chess Engine Profile ---\n");
  printf("Function                               | Time (ms) | %% of Total\n");
  printf("---------------------------------------|-----------|------------\n");

  double total_ms = (double)profile_total_ticks * 1000.0 / CLOCKS_PER_SEC;

  printf("c_step (Total)                         | %9.2f | %8.2f%%\n", total_ms,
         100.0);

  double move_gen_ms =
      (double)profile_move_gen_uci_ticks * 1000.0 / CLOCKS_PER_SEC;
  printf("  -> chess_generate_legal_moves_uci   | %9.2f | %8.2f%%\n",
         move_gen_ms, (move_gen_ms / total_ms) * 100);

  double is_legal_ms =
      (double)profile_is_legal_move_ticks * 1000.0 / CLOCKS_PER_SEC;
  printf("    -> chess_is_legal_move            | %9.2f | %8.2f%%\n",
         is_legal_ms, (is_legal_ms / total_ms) * 100);

  double is_attacked_ms =
      (double)profile_is_square_attacked_ticks * 1000.0 / CLOCKS_PER_SEC;
  printf("      -> is_square_attacked           | %9.2f | %8.2f%%\n",
         is_attacked_ms, (is_attacked_ms / total_ms) * 100);

  double make_move_ms =
      (double)profile_make_move_fast_ticks * 1000.0 / CLOCKS_PER_SEC;
  printf("      -> make_move_fast               | %9.2f | %8.2f%%\n",
         make_move_ms, (make_move_ms / total_ms) * 100);

  double unmake_move_ms =
      (double)profile_unmake_move_fast_ticks * 1000.0 / CLOCKS_PER_SEC;
  printf("      -> unmake_move_fast             | %9.2f | %8.2f%%\n",
         unmake_move_ms, (unmake_move_ms / total_ms) * 100);

  double apply_uci_ms =
      (double)profile_apply_uci_move_ticks * 1000.0 / CLOCKS_PER_SEC;
  printf("  -> apply_uci_move                   | %9.2f | %8.2f%%\n",
         apply_uci_ms, (apply_uci_ms / total_ms) * 100);

  double compute_obs_ms =
      (double)profile_compute_obs_ticks * 1000.0 / CLOCKS_PER_SEC;
  printf("  -> compute_observation_with_perspective | %9.2f | %8.2f%%\n",
         compute_obs_ms, (compute_obs_ms / total_ms) * 100);

  printf("----------------------------------------------------------\n");
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CHESS_H
