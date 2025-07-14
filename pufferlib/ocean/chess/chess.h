// chess.h - Complete Pure C Chess Environment for PufferLib
// Optimized for 150k+ SPS performance with single network self-play

#ifndef CHESS_H
#define CHESS_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Disable debug logging for performance (set to 1 to enable)
#ifndef DEBUG_LOG
#define DEBUG_LOG 1
#endif

#if DEBUG_LOG
  #define DBG(expr) printf("%s", expr)
#else
  #define DBG(expr) do { } while (0)
#endif

// Include UCI action mapping
#include "chess_action_mapping.h"

#ifdef __cplusplus
extern "C" {
#endif

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

typedef enum {
    WHITE = 0,
    BLACK = 1,
    NO_COLOR = 2
} Color;

typedef struct {
    Color color;
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
    float episode_return;            // combined (existing)
    float episode_return_white;      // new – white perspective total
    float episode_return_black;      // new – black perspective total
    float reward_valid;
    float reward_agent_captures_enemy_piece;
    float reward_enemy_captures_agent_piece;
    float reward_draw;
    // Perspective-based reward tracking
    float reward_win_white;          // win rewards from white's perspective
    float reward_win_black;          // win rewards from black's perspective  
    float reward_loss_white;         // loss rewards from white's perspective
    float reward_loss_black;         // loss rewards from black's perspective
    float reward_draw_white;         // draw rewards from white's perspective
    float reward_draw_black;         // draw rewards from black's perspective
    float game_drawn;
    // New separate win/loss tracking from both perspectives
    float white_win;                 // white wins (from white's perspective)
    float white_loss;                // white losses (from white's perspective)
    float black_win;                 // black wins (from black's perspective)
    float black_loss;                // black losses (from black's perspective)
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
    float en_passant_white;          // white captures via en passant
    float en_passant_black;          // black captures via en passant
    // Castling moves
    float white_castle_kingside;     // white castles kingside
    float white_castle_queenside;    // white castles queenside
    float black_castle_kingside;     // black castles kingside
    float black_castle_queenside;    // black castles queenside
    // Pawn promotions
    float white_promotion_count;     // total white pawn promotions
    float white_promotion_knight;    // white promotes to knight
    float white_promotion_bishop;    // white promotes to bishop
    float white_promotion_rook;      // white promotes to rook
    float white_promotion_queen;     // white promotes to queen
    float black_promotion_count;     // total black pawn promotions
    float black_promotion_knight;    // black promotes to knight
    float black_promotion_bishop;    // black promotes to bishop
    float black_promotion_rook;      // black promotes to rook
    float black_promotion_queen;     // black promotes to queen

    // Game logging fields expected by binding.cpp
    float last_move_from;
    float last_move_to;
    float last_move_promotion;
    float game_step_logged;
    float game_moves_count;
    
    float complete_game_move_count;
    // Note: complete_game_moves_serialized removed to comply with PufferLib float-only logging spec

    // n field is always last per pufferlib spec
    float n;
} Log;

// === CHESS BOARD STATE ===
typedef struct {
    Piece board[64];
    Color to_move;
    uint8_t castle_rights;  // bits: 0=WK, 1=WQ, 2=BK, 3=BQ
    int8_t ep_square;       // en passant target square (-1 if none)
    uint8_t halfmove_clock;
    uint16_t fullmove_number;
    uint64_t zobrist_hash;  // Incrementally updated Zobrist hash
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
    char legal_moves_buffer[256][6];  // Up to 256 UCI strings
    int legal_moves_count;
    bool legal_moves_cached;
    uint64_t cached_board_hash;
    
    // Game modes
    bool dual_agent_self_play_mode;
    bool self_play_mode;
    
    // Complete game logging
    int complete_game_actions[100];
    int complete_game_action_count;
    char serialized_moves[1024];  // Comma-separated action IDs for efficient logging
    
    // Reward configuration (copied from CChess for performance)
    float c_reward_valid;
    float c_reward_invalid_white;
    float c_reward_invalid_black;
    float c_reward_agent_captures_enemy_piece;
    float c_reward_enemy_captures_agent_piece;
    float c_reward_draw;
    float c_reward_win_white;
    float c_reward_win_black;
    float c_reward_loss_white;
    float c_reward_loss_black;
    float c_reward_check_white;
    float c_reward_check_black;
    float c_reward_material_diff_white;
    float c_reward_material_diff_black;
    
    // ACCUMULATED REWARD COUNTERS (for add_log aggregation)
    float accumulated_reward_valid;
    float accumulated_reward_agent_captures_enemy_piece;
    float accumulated_reward_enemy_captures_agent_piece;
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
} ChessContext;

// === PUFFERLIB ENVIRONMENT STRUCTURE ===
typedef struct CChess {
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    
    // Configuration values from INI file
    float reward_valid;
    float reward_invalid_white;
    float reward_invalid_black;
    float reward_agent_captures_enemy_piece;
    float reward_enemy_captures_agent_piece;
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
    
    // Debug settings
    bool debug_disable_mask;
    bool stockfish_enabled;
    
    // Chess context (pure C, no opaque pointer)
    ChessContext context;
    
    // Convenience pointer to avoid repeated dereferencing
    ChessContext* ctx;
} CChess;

// === ADDITIONAL BINDING FUNCTIONS ===
void enable_stockfish_black(CChess* env, const char* stockfish_cmd, int elo, int search_ms);
void set_self_play_mode(CChess* env, bool enabled);
void set_dual_agent_self_play_mode(CChess* env, bool enabled);
void set_debug_disable_mask(CChess* env, bool enabled);

// === PUFFERLIB REQUIRED FUNCTIONS ===
void init(CChess* env);
void allocate(CChess* env);
void free_allocated(CChess* env);
void add_log(CChess* env);
void c_reset(CChess* env);
void c_step(CChess* env);
void c_render(CChess* env);
void c_close(CChess* env);

// === MODE SETTERS FOR COMPATIBILITY ===
void set_dual_agent_self_play_mode(CChess* env, bool enabled);
void set_self_play_mode(CChess* env, bool enabled);
void c_set_fen(CChess* env, const char* fen);

// === CHESS HELPER FUNCTIONS ===

// Board access
static inline Piece* get_piece(ChessBoard* board, int x, int y) {
    if (x < 0 || x >= 8 || y < 0 || y >= 8) return NULL;
    return &board->board[y * 8 + x];
}

static inline const Piece* get_piece_const(const ChessBoard* board, int x, int y) {
    if (x < 0 || x >= 8 || y < 0 || y >= 8) return NULL;
    return &board->board[y * 8 + x];
}

// Square notation conversion
static inline Square notation_to_square(const char* notation) {
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

static inline void square_to_notation(Square sq, char* notation) {
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
static uint64_t zobrist_piece_square[2][7][64];  // [color][piece_type][square]
static uint64_t zobrist_castle_rights[16];       // [castle_rights]
static uint64_t zobrist_en_passant[64];          // [ep_square]
static uint64_t zobrist_side_to_move;
static bool zobrist_initialized = false;

static void init_zobrist_tables(void) {
    if (zobrist_initialized) return;
    
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
static inline uint64_t hash_position(const ChessBoard* board) {
    return board->zobrist_hash;
}

// Compute hash from scratch (only used for initialization)
static uint64_t compute_zobrist_hash(const ChessBoard* board) {
    if (!zobrist_initialized) init_zobrist_tables();
    
    uint64_t hash = 0;
    
    // Hash pieces
    for (int i = 0; i < 64; i++) {
        if (board->board[i].type != EMPTY) {
            hash ^= zobrist_piece_square[board->board[i].color][board->board[i].type][i];
        }
    }
    
    // Hash side to move
    if (board->to_move == BLACK) {
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
static inline void update_zobrist_hash(ChessBoard* board, int from_square, int to_square, 
                                      Piece moved_piece, Piece captured_piece,
                                      uint8_t old_castle_rights, int8_t old_ep_square) {
    if (!zobrist_initialized) init_zobrist_tables();
    
    // Remove old piece from from_square
    board->zobrist_hash ^= zobrist_piece_square[moved_piece.color][moved_piece.type][from_square];
    
    // Add piece to to_square
    board->zobrist_hash ^= zobrist_piece_square[moved_piece.color][moved_piece.type][to_square];
    
    // Remove captured piece if any
    if (captured_piece.type != EMPTY) {
        board->zobrist_hash ^= zobrist_piece_square[captured_piece.color][captured_piece.type][to_square];
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

static void serialize_complete_game_moves(ChessContext* ctx) {
    ctx->serialized_moves[0] = '\0';
    
    if (ctx->complete_game_action_count == 0) {
        return;
    }
    
    char temp[16];
    for (int i = 0; i < ctx->complete_game_action_count && i < 100; i++) {
        if (i > 0) {
            strcat(ctx->serialized_moves, ",");
        }
        sprintf(temp, "%d", ctx->complete_game_actions[i]);
        strcat(ctx->serialized_moves, temp);
    }
}

// === PERSPECTIVE FLIPPING FOR SELF-PLAY ===

static inline void flip_uci_for_black_perspective(const char* original_uci, char* flipped_uci) {
    flipped_uci[0] = original_uci[0];           // file stays same
    flipped_uci[1] = '9' - original_uci[1];    // flip rank: 1→8, 2→7, etc.
    flipped_uci[2] = original_uci[2];           // file stays same
    flipped_uci[3] = '9' - original_uci[3];    // flip rank
    
    if (strlen(original_uci) >= 5) {
        flipped_uci[4] = original_uci[4];       // promotion piece unchanged
        flipped_uci[5] = '\0';
    } else {
        flipped_uci[4] = '\0';
    }
}

// === LEGAL MOVE GENERATION (PURE C, OPTIMIZED) ===

static bool is_square_attacked(const ChessBoard* board, Square sq, Color by_color) {
    // Optimized attack detection
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            const Piece* piece = get_piece_const(board, x, y);
            if (!piece || piece->type == EMPTY || piece->color != by_color) continue;
            
            int dx = sq.x - x;
            int dy = sq.y - y;
            
            switch (piece->type) {
                case PAWN: {
                    int direction = (piece->color == WHITE) ? 1 : -1;
                    if (dy == direction && abs(dx) == 1) return true;
                    break;
                }
                case KNIGHT:
                    if ((abs(dx) == 2 && abs(dy) == 1) || (abs(dx) == 1 && abs(dy) == 2)) {
                        return true;
                    }
                    break;
                case BISHOP:
                case QUEEN:
                    if (abs(dx) == abs(dy) && dx != 0) {
                        // Check diagonal path
                        int step_x = (dx > 0) ? 1 : -1;
                        int step_y = (dy > 0) ? 1 : -1;
                        bool clear = true;
                        for (int i = 1; i < abs(dx); i++) {
                            const Piece* blocker = get_piece_const(board, x + i * step_x, y + i * step_y);
                            if (blocker && blocker->type != EMPTY) {
                                clear = false;
                                break;
                            }
                        }
                        if (clear && piece->type != ROOK) return true;
                    }
                    if (piece->type == BISHOP) break;
                    // Fall through for queen
                case ROOK:
                    if ((dx == 0 || dy == 0) && (dx != 0 || dy != 0)) {
                        // Check rank/file path
                        int step_x = (dx == 0) ? 0 : ((dx > 0) ? 1 : -1);
                        int step_y = (dy == 0) ? 0 : ((dy > 0) ? 1 : -1);
                        bool clear = true;
                        int steps = (dx == 0) ? abs(dy) : abs(dx);
                        for (int i = 1; i < steps; i++) {
                            const Piece* blocker = get_piece_const(board, x + i * step_x, y + i * step_y);
                            if (blocker && blocker->type != EMPTY) {
                                clear = false;
                                break;
                            }
                        }
                        if (clear && piece->type != BISHOP) return true;
                    }
                    break;
                case KING:
                    if (abs(dx) <= 1 && abs(dy) <= 1 && (dx != 0 || dy != 0)) {
                        return true;
                    }
                    break;
                case EMPTY:
                default:
                    break;
            }
        }
    }
    return false;
}

static bool is_in_check(const ChessBoard* board, Color color) {
    // Find king
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            const Piece* piece = get_piece_const(board, x, y);
            if (piece && piece->type == KING && piece->color == color) {
                Color opponent = (color == WHITE) ? BLACK : WHITE;
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

// Forward declarations
static bool apply_uci_move(ChessContext* ctx, const char* uci_str);
static void add_position_to_history(ChessContext* ctx, uint64_t hash);
static int get_position_count(ChessContext* ctx, uint64_t hash);
static bool is_threefold_repetition(ChessContext* ctx);
static bool is_insufficient_material(ChessContext* ctx);

static bool chess_is_legal_move(ChessContext* ctx, ChessMove move) {
    // Fast legality check: make move on temporary board and check if king is in check
    ChessBoard temp_board = ctx->board;  // Copy current board
    Color moving_color = temp_board.to_move;
    
    // Apply move to temporary board (simplified version for legality checking)
    Piece* from_piece = get_piece(&temp_board, move.from.x, move.from.y);
    Piece* to_piece = get_piece(&temp_board, move.to.x, move.to.y);
    
    // Basic validity checks
    if (!from_piece || from_piece->type == EMPTY || from_piece->color != moving_color) {
        return false;
    }
    
    // Handle castling move specially
    if (move.is_castling) {
        if (from_piece->type != KING) return false;
        if (is_in_check(&temp_board, moving_color)) return false; // Can't castle out of check
        
        // Verify castling path is clear and not through check
        int rank = (moving_color == WHITE) ? 0 : 7;
        if (move.to.x == 6) { // Kingside
            for (int x = 5; x <= 6; x++) {
                Square sq = {(int8_t)x, (int8_t)rank};
                if (is_square_attacked(&temp_board, sq, (Color)(1 - moving_color))) return false;
            }
        } else if (move.to.x == 2) { // Queenside  
            for (int x = 2; x <= 3; x++) {
                Square sq = {(int8_t)x, (int8_t)rank};
                if (is_square_attacked(&temp_board, sq, (Color)(1 - moving_color))) return false;
            }
        }
    }
    
    // Make the move on temporary board
    PieceType final_type = (move.promotion != EMPTY) ? move.promotion : from_piece->type;
    
    // Handle en passant capture
    if (move.is_en_passant) {
        int captured_y = (moving_color == WHITE) ? move.to.y - 1 : move.to.y + 1;
        get_piece(&temp_board, move.to.x, captured_y)->type = EMPTY;
        get_piece(&temp_board, move.to.x, captured_y)->color = NO_COLOR;
    }
    
    // Move piece
    from_piece->type = EMPTY;
    from_piece->color = NO_COLOR;
    to_piece->type = final_type;
    to_piece->color = moving_color;
    
    // Handle castling rook move
    if (move.is_castling) {
        int rank = (moving_color == WHITE) ? 0 : 7;
        if (move.to.x == 6) { // Kingside
            get_piece(&temp_board, 7, rank)->type = EMPTY;
            get_piece(&temp_board, 7, rank)->color = NO_COLOR;
            get_piece(&temp_board, 5, rank)->type = ROOK;
            get_piece(&temp_board, 5, rank)->color = moving_color;
        } else if (move.to.x == 2) { // Queenside
            get_piece(&temp_board, 0, rank)->type = EMPTY;
            get_piece(&temp_board, 0, rank)->color = NO_COLOR;
            get_piece(&temp_board, 3, rank)->type = ROOK;
            get_piece(&temp_board, 3, rank)->color = moving_color;
        }
    }
    
    // Check if our king is in check after the move
    return !is_in_check(&temp_board, moving_color);
}

static void add_legal_move(ChessContext* ctx, LegalMoves* moves, ChessMove move) {
    if (moves->count >= 256) return;
    
    if (chess_is_legal_move(ctx, move)) {
        moves->moves[moves->count] = move;
        moves->count++;
    }
}

static void generate_pseudo_legal_moves_for_piece(ChessContext* ctx, LegalMoves* moves, Square from) {
    ChessBoard* board = &ctx->board;
    const Piece* piece = get_piece_const(board, from.x, from.y);
    if (!piece || piece->type == EMPTY || piece->color != board->to_move) return;
    
    Color us = board->to_move;
    Color them = (us == WHITE) ? BLACK : WHITE;
    
    switch (piece->type) {
        case PAWN: {
            int direction = (us == WHITE) ? 1 : -1;
            int start_rank = (us == WHITE) ? 1 : 6;
            int promote_rank = (us == WHITE) ? 7 : 0;
            
            // Forward moves
            Square to;
            to.x = from.x;
            to.y = from.y + direction;
            if (to.y >= 0 && to.y < 8) {
                const Piece* target = get_piece_const(board, to.x, to.y);
                if (target && target->type == EMPTY) {
                    if (to.y == promote_rank) {
                        // Promotions
                        PieceType promotions[] = {QUEEN, ROOK, BISHOP, KNIGHT};
                        for (int p = 0; p < 4; p++) {
                            ChessMove move = {from, to, promotions[p], false, false};
                            add_legal_move(ctx, moves, move);
                        }
                    } else {
                        ChessMove move = {from, to, EMPTY, false, false};
                        add_legal_move(ctx, moves, move);
                    }
                    
                    // Double forward from starting position
                    if (from.y == start_rank) {
                        to.y += direction;
                        target = get_piece_const(board, to.x, to.y);
                        if (target && target->type == EMPTY) {
                            ChessMove move = {from, to, EMPTY, false, false};
                            add_legal_move(ctx, moves, move);
                        }
                    }
                }
            }
            
            // Captures
            for (int dx = -1; dx <= 1; dx += 2) {
                to.x = from.x + dx;
                to.y = from.y + direction;
                if (to.x >= 0 && to.x < 8 && to.y >= 0 && to.y < 8) {
                    const Piece* target = get_piece_const(board, to.x, to.y);
                    bool can_capture = false;
                    bool is_en_passant = false;
                    
                    // Regular capture
                    if (target && target->type != EMPTY && target->color == them) {
                        can_capture = true;
                    }
                    // En passant
                    else if (board->ep_square >= 0) {
                        int ep_x = board->ep_square % 8;
                        int ep_y = board->ep_square / 8;
                        if (to.x == ep_x && to.y == ep_y) {
                            can_capture = true;
                            is_en_passant = true;
                        }
                    }
                    
                    if (can_capture) {
                        if (to.y == promote_rank) {
                            // Promotion captures
                            PieceType promotions[] = {QUEEN, ROOK, BISHOP, KNIGHT};
                            for (int p = 0; p < 4; p++) {
                                ChessMove move = {from, to, promotions[p], false, is_en_passant};
                                add_legal_move(ctx, moves, move);
                            }
                        } else {
                            ChessMove move = {from, to, EMPTY, false, is_en_passant};
                            add_legal_move(ctx, moves, move);
                        }
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
                int rook_dirs[][2] = {{0,1}, {1,0}, {0,-1}, {-1,0}};
                memcpy(directions, rook_dirs, sizeof(rook_dirs));
                num_dirs = 4;
            } else if (piece->type == BISHOP) {
                int bishop_dirs[][2] = {{1,1}, {1,-1}, {-1,1}, {-1,-1}};
                memcpy(directions, bishop_dirs, sizeof(bishop_dirs));
                num_dirs = 4;
            } else { // QUEEN
                int queen_dirs[][2] = {{0,1}, {1,0}, {0,-1}, {-1,0}, {1,1}, {1,-1}, {-1,1}, {-1,-1}};
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
                    if (to.x < 0 || to.x >= 8 || to.y < 0 || to.y >= 8) break;
                    
                    const Piece* target = get_piece_const(board, to.x, to.y);
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
            int knight_moves[][2] = {
                {2,1}, {2,-1}, {-2,1}, {-2,-1},
                {1,2}, {1,-2}, {-1,2}, {-1,-2}
            };
            
            for (int i = 0; i < 8; i++) {
                Square to;
                to.x = from.x + knight_moves[i][0];
                to.y = from.y + knight_moves[i][1];
                if (to.x >= 0 && to.x < 8 && to.y >= 0 && to.y < 8) {
                    const Piece* target = get_piece_const(board, to.x, to.y);
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
                    if (dx == 0 && dy == 0) continue;
                    
                    Square to;
                    to.x = from.x + dx;
                    to.y = from.y + dy;
                    if (to.x >= 0 && to.x < 8 && to.y >= 0 && to.y < 8) {
                        const Piece* target = get_piece_const(board, to.x, to.y);
                        if (!target || target->type == EMPTY || target->color == them) {
                            ChessMove move = {from, to, EMPTY, false, false};
                            add_legal_move(ctx, moves, move);
                        }
                    }
                }
            }
            
            // Castling
            if (!is_in_check(board, us)) {
                int rank = (us == WHITE) ? 0 : 7;
                
                if (from.x == 4 && from.y == rank) {
                    // Kingside castling
                    if ((board->castle_rights & (us == WHITE ? 1 : 4))) {
                        bool can_castle = true;
                        // Check squares are empty and not attacked
                        for (int x = 5; x <= 6; x++) {
                            const Piece* sq = get_piece_const(board, x, rank);
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
                    if ((board->castle_rights & (us == WHITE ? 2 : 8))) {
                        bool can_castle = true;
                        // Check squares are empty
                        for (int x = 1; x <= 3; x++) {
                            const Piece* sq = get_piece_const(board, x, rank);
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
typedef bool (*MoveYieldCallback)(ChessContext* ctx, const ChessMove* move, void* user_data);

// Callback that terminates on first legal move found
static bool first_move_callback(ChessContext* ctx, const ChessMove* move, void* user_data) {
    bool* found = (bool*)user_data;
    *found = true;
    return true; // Terminate immediately
}

// Yield-based move generation - returns true if callback requested early termination
static bool chess_generate_legal_moves_yield(ChessContext* ctx, MoveYieldCallback yield_fn, void* user_data) {
    // Iterate through all squares on the board
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            Square from = {(int8_t)x, (int8_t)y};
            const Piece* piece = get_piece_const(&ctx->board, x, y);
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

static void chess_generate_legal_moves(ChessContext* ctx, LegalMoves* moves) {
    moves->count = 0;
    
    // Iterate through all squares on the board
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            Square from = {(int8_t)x, (int8_t)y};
            const Piece* piece = get_piece_const(&ctx->board, x, y);
            if (piece && piece->type != EMPTY && piece->color == ctx->board.to_move) {
                generate_pseudo_legal_moves_for_piece(ctx, moves, from);
            }
        }
    }
}

static int chess_generate_legal_moves_uci(ChessContext* ctx) {
    // Check cache
    uint64_t current_hash = hash_position(&ctx->board);
    if (ctx->legal_moves_cached && ctx->cached_board_hash == current_hash) {
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
            char promo_char = (move.promotion == QUEEN) ? 'q' : 
                             (move.promotion == ROOK) ? 'r' :
                             (move.promotion == BISHOP) ? 'b' : 'n';
            snprintf(ctx->legal_moves_buffer[ctx->legal_moves_count], 6, "%c%c%c%c%c", 
                    'a' + move.from.x, '1' + move.from.y,
                    'a' + move.to.x, '1' + move.to.y, promo_char);
        } else {
            snprintf(ctx->legal_moves_buffer[ctx->legal_moves_count], 5, "%c%c%c%c", 
                    'a' + move.from.x, '1' + move.from.y,
                    'a' + move.to.x, '1' + move.to.y);
        }
        ctx->legal_moves_count++;
    }
    
    // Cache the result
    ctx->legal_moves_cached = true;
    ctx->cached_board_hash = current_hash;
    
    return ctx->legal_moves_count;
}

// === BOARD STATE MANIPULATION ===

static void init_board(ChessBoard* board) {
    memset(board, 0, sizeof(ChessBoard));
    
    // Set up starting position
    const char* start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    
    // Parse FEN (simplified)
    int x = 0, y = 7;
    const char* p = start_fen;
    
    while (*p && *p != ' ') {
        if (*p == '/') {
            x = 0;
            y--;
        } else if (*p >= '1' && *p <= '8') {
            x += (*p - '0');
        } else {
            Color color = (*p >= 'A' && *p <= 'Z') ? WHITE : BLACK;
            PieceType type = EMPTY;
            
            switch (*p | 32) {
                case 'k': type = KING; break;
                case 'q': type = QUEEN; break;
                case 'r': type = ROOK; break;
                case 'b': type = BISHOP; break;
                case 'n': type = KNIGHT; break;
                case 'p': type = PAWN; break;
            }
            
            if (type != EMPTY && x < 8) {
                board->board[y * 8 + x] = (Piece){color, type};
                x++;
            }
        }
        p++;
    }
    
    board->to_move = WHITE;
    board->castle_rights = 0xF; // KQkq
    board->ep_square = -1;
    board->halfmove_clock = 0;
    board->fullmove_number = 1;
    
    // Initialize Zobrist hash
    board->zobrist_hash = compute_zobrist_hash(board);
}

// === OBSERVATION COMPUTATION WITH PERSPECTIVE FLIPPING ===

void compute_observation_with_perspective(CChess* env, ChessContext* ctx) {
    int idx = 0;
    Color current_player = ctx->board.to_move;
    
    // CRITICAL: Always render from current player's perspective
    // Current player's pieces at bottom (ranks 1-2), opponent at top (ranks 7-8)
    
    // First 6 planes: Current player's pieces
    for (int type = 1; type <= 6; type++) { // KING=1 to PAWN=6
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                // If Black to move, flip the board vertically
                int actual_y = (current_player == WHITE) ? y : (7 - y);
                
                const Piece* p = get_piece_const(&ctx->board, x, actual_y);
                bool is_current_player_piece = (p && p->color == current_player && p->type == type);
                env->observations[idx++] = is_current_player_piece ? 1.0f : 0.0f;
            }
        }
    }
    
    // Next 6 planes: Opponent's pieces
    Color opponent = (current_player == WHITE) ? BLACK : WHITE;
    for (int type = 1; type <= 6; type++) {
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                int actual_y = (current_player == WHITE) ? y : (7 - y);
                
                const Piece* p = get_piece_const(&ctx->board, x, actual_y);
                bool is_opponent_piece = (p && p->color == opponent && p->type == type);
                env->observations[idx++] = is_opponent_piece ? 1.0f : 0.0f;
            }
        }
    }
    
    // Empty squares plane
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int actual_y = (current_player == WHITE) ? y : (7 - y);
            const Piece* p = get_piece_const(&ctx->board, x, actual_y);
            env->observations[idx++] = (!p || p->type == EMPTY) ? 1.0f : 0.0f;
        }
    }
    
    // Repetition count plane (using actual position history)
    int reps = get_position_count(ctx, ctx->board.zobrist_hash);
    float rep_val = (reps - 1) / 2.0f;
    for (int i = 0; i < 64; i++) {
        env->observations[idx++] = rep_val;
    }
    
    // Side to move plane (always 0 from current player's perspective)
    for (int i = 0; i < 64; i++) {
        env->observations[idx++] = 0.0f;
    }
    
    // Halfmove clock plane
    float halfmove_val = ctx->board.halfmove_clock / 101.0f;
    for (int i = 0; i < 64; i++) {
        env->observations[idx++] = halfmove_val;
    }
    
    // Castling rights planes (4 planes, flipped for black perspective)
    uint8_t rights = ctx->board.castle_rights;
    if (current_player == BLACK) {
        // Flip castling rights for Black's perspective
        uint8_t flipped = 0;
        if (rights & 4) flipped |= 1; // BK → WK from Black's perspective
        if (rights & 8) flipped |= 2; // BQ → WQ from Black's perspective  
        if (rights & 1) flipped |= 4; // WK → BK from Black's perspective
        if (rights & 2) flipped |= 8; // WQ → BQ from Black's perspective
        rights = flipped;
    }
    
    for (int i = 0; i < 4; i++) {
        float castle_val = (rights & (1 << i)) ? 1.0f : 0.0f;
        for (int j = 0; j < 64; j++) {
            env->observations[idx++] = castle_val;
        }
    }
    
    // En passant target square plane (flipped for black perspective)
    int8_t ep_square = ctx->board.ep_square;
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int actual_y = (current_player == WHITE) ? y : (7 - y);
            int square_idx = actual_y * 8 + x;
            env->observations[idx++] = (ep_square == square_idx) ? 1.0f : 0.0f;
        }
    }
    
    // Should be exactly 1344 floats at this point (21 * 8 * 8)
    assert(idx == 1344);
    
    // === LEGAL MOVE MASK (NEW UCI ACTION SPACE) ===
    
    // Use cached legal moves if available, otherwise generate them
    if (!ctx->legal_moves_cached) {
      chess_generate_legal_moves_uci(&env->context);
    }
    
    // Clear mask
    for (int i = 0; i < TOTAL_CHESS_ACTIONS; i++) {
        env->observations[idx + i] = 0.0f;
    }
    
    if (env->debug_disable_mask) {
        // Debug mode: allow all actions
        for (int i = 0; i < TOTAL_CHESS_ACTIONS; i++) {
            env->observations[idx + i] = 1.0f;
        }
    } else {
        // Set legal moves to 1.0
        for (int i = 0; i < ctx->legal_moves_count; i++) {
            const char* uci_move = ctx->legal_moves_buffer[i];
            
            // If Black to move, need to flip UCI coordinates for the mask
            char flipped_uci[6];
            if (current_player == BLACK) {
                flip_uci_for_black_perspective(uci_move, flipped_uci);
                uci_move = flipped_uci;
            }
            
            int action_id = uci_to_action_id(uci_move);
            if (action_id >= 0 && action_id < TOTAL_CHESS_ACTIONS) {
                env->observations[idx + action_id] = 1.0f;
            }
        }
    }
}

// === MOVE APPLICATION ===

static bool apply_uci_move(ChessContext* ctx, const char* uci_str) {
    if (strlen(uci_str) < 4) return false;
    
    int from_x = uci_str[0] - 'a';
    int from_y = uci_str[1] - '1';
    int to_x = uci_str[2] - 'a';
    int to_y = uci_str[3] - '1';
    
    if (from_x < 0 || from_x >= 8 || from_y < 0 || from_y >= 8 ||
        to_x < 0 || to_x >= 8 || to_y < 0 || to_y >= 8) {
        return false;
    }
    
    ChessBoard* board = &ctx->board;
    Color us = board->to_move;
    
    // Store old values for Zobrist updates
    uint8_t old_castle_rights = board->castle_rights;
    int8_t old_ep_square = board->ep_square;
    
    // Get piece being moved
    Piece* from_piece = get_piece(board, from_x, from_y);
    if (!from_piece || from_piece->type == EMPTY || from_piece->color != us) {
        return false;
    }
    
    Piece moving_piece = *from_piece;
    Piece* captured_piece = get_piece(board, to_x, to_y);
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
        int rank = (us == WHITE) ? 0 : 7;
        bool kingside = (to_x > from_x);
        
        // Validate castling rights and path
        int rook_from = kingside ? 7 : 0;
        int rook_to = kingside ? 5 : 3;
        
        // Check if path is clear (king's path already checked in legal move gen)
        for (int x = (kingside ? 5 : 1); x <= (kingside ? 6 : 3); x++) {
            if (x != from_x && get_piece_const(board, x, rank)->type != EMPTY) {
                return false;
            }
        }
        
        // XOR out old piece positions
        board->zobrist_hash ^= zobrist_piece_square[us][KING][from_y * 8 + from_x];
        board->zobrist_hash ^= zobrist_piece_square[us][ROOK][rank * 8 + rook_from];
        
        // Move king
        from_piece->type = EMPTY;
        from_piece->color = NO_COLOR;
        get_piece(board, to_x, to_y)->type = KING;
        get_piece(board, to_x, to_y)->color = us;
        
        // Move rook
        Piece* rook_piece = get_piece(board, rook_from, rank);
        if (rook_piece->type != ROOK || rook_piece->color != us) {
            return false;
        }
        rook_piece->type = EMPTY;
        rook_piece->color = NO_COLOR;
        get_piece(board, rook_to, rank)->type = ROOK;
        get_piece(board, rook_to, rank)->color = us;
        
        // XOR in new piece positions
        board->zobrist_hash ^= zobrist_piece_square[us][KING][to_y * 8 + to_x];
        board->zobrist_hash ^= zobrist_piece_square[us][ROOK][rank * 8 + rook_to];
        
        // Update castling rights
        if (us == WHITE) {
            board->castle_rights &= ~0x3; // Clear white castling
        } else {
            board->castle_rights &= ~0xC; // Clear black castling
        }
    }
    // Handle en passant capture
    else if (moving_piece.type == PAWN && to_x != from_x && !is_capture) {
        // This is en passant - capture the pawn behind the destination
        int captured_y = (us == WHITE) ? to_y - 1 : to_y + 1;
        Piece* en_passant_piece = get_piece(board, to_x, captured_y);
        if (en_passant_piece->type != PAWN || en_passant_piece->color == us) {
            return false;
        }
        
        // XOR out old pieces
        board->zobrist_hash ^= zobrist_piece_square[us][PAWN][from_y * 8 + from_x];
        board->zobrist_hash ^= zobrist_piece_square[1-us][PAWN][captured_y * 8 + to_x];
        
        // Remove the en passant captured pawn
        en_passant_piece->type = EMPTY;
        en_passant_piece->color = NO_COLOR;
        
        // Move the capturing pawn
        from_piece->type = EMPTY;
        from_piece->color = NO_COLOR;
        get_piece(board, to_x, to_y)->type = PAWN;
        get_piece(board, to_x, to_y)->color = us;
        
        // XOR in new pawn position
        board->zobrist_hash ^= zobrist_piece_square[us][PAWN][to_y * 8 + to_x];
    }
    // Regular move
    else {
        // XOR out old piece position
        board->zobrist_hash ^= zobrist_piece_square[us][moving_piece.type][from_y * 8 + from_x];
        
        // XOR out captured piece if any
        if (is_capture) {
            board->zobrist_hash ^= zobrist_piece_square[captured_piece_copy.color][captured_piece_copy.type][to_y * 8 + to_x];
        }
        
        // Clear source square
        from_piece->type = EMPTY;
        from_piece->color = NO_COLOR;
        
        // Place piece on destination
        get_piece(board, to_x, to_y)->type = moving_piece.type;
        get_piece(board, to_x, to_y)->color = moving_piece.color;
        
        // Handle promotion
        PieceType final_type = moving_piece.type;
        if (strlen(uci_str) == 5 && moving_piece.type == PAWN) {
            char promo = uci_str[4];
            switch (promo) {
                case 'q': final_type = QUEEN; break;
                case 'r': final_type = ROOK; break;
                case 'b': final_type = BISHOP; break;
                case 'n': final_type = KNIGHT; break;
                default: return false;
            }
            get_piece(board, to_x, to_y)->type = final_type;
        }
        
        // XOR in new piece position
        board->zobrist_hash ^= zobrist_piece_square[us][final_type][to_y * 8 + to_x];
    }
    
    // Update castling rights when king or rook moves
    if (moving_piece.type == KING) {
        if (us == WHITE) {
            board->castle_rights &= ~0x3; // Clear white castling
        } else {
            board->castle_rights &= ~0xC; // Clear black castling
        }
    } else if (moving_piece.type == ROOK) {
        if (us == WHITE) {
            if (from_x == 0) board->castle_rights &= ~0x2; // White queenside
            if (from_x == 7) board->castle_rights &= ~0x1; // White kingside
        } else {
            if (from_x == 0) board->castle_rights &= ~0x8; // Black queenside
            if (from_x == 7) board->castle_rights &= ~0x4; // Black kingside
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
    board->to_move = (us == WHITE) ? BLACK : WHITE;
    if (us == BLACK) {
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
    ctx->step_count++;
    printf("step_count from apply_uci_move: %d\n", ctx->step_count);
    // Add to complete game log
    int action_id = uci_to_action_id(uci_str);
    if (action_id >= 0 && ctx->complete_game_action_count < 100) {
        ctx->complete_game_actions[ctx->complete_game_action_count++] = action_id;
    }
    
    return true;
}

// === DRAW DETECTION FUNCTIONS ===

static void add_position_to_history(ChessContext* ctx, uint64_t hash) {
    PositionHistory* history = &ctx->position_history;
    
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

static int get_position_count(ChessContext* ctx, uint64_t hash) {
    PositionHistory* history = &ctx->position_history;
    for (int i = 0; i < history->size; i++) {
        if (history->hashes[i] == hash) {
            return history->counts[i];
        }
    }
    return 0;
}

static bool is_threefold_repetition(ChessContext* ctx) {
    uint64_t current_hash = ctx->board.zobrist_hash;
    return get_position_count(ctx, current_hash) >= 3;
}

static bool is_insufficient_material(ChessContext* ctx) {
    ChessBoard* board = &ctx->board;
    
    // Count material for both sides
    int white_pawns = 0, black_pawns = 0;
    int white_knights = 0, black_knights = 0;
    int white_bishops = 0, black_bishops = 0;
    int white_rooks = 0, black_rooks = 0;
    int white_queens = 0, black_queens = 0;
    
    for (int i = 0; i < 64; i++) {
        const Piece* p = &board->board[i];
        if (p->type == EMPTY) continue;
        
        switch (p->type) {
            case PAWN:
                if (p->color == WHITE) white_pawns++; else black_pawns++;
                break;
            case KNIGHT:
                if (p->color == WHITE) white_knights++; else black_knights++;
                break;
            case BISHOP:
                if (p->color == WHITE) white_bishops++; else black_bishops++;
                break;
            case ROOK:
                if (p->color == WHITE) white_rooks++; else black_rooks++;
                break;
            case QUEEN:
                if (p->color == WHITE) white_queens++; else black_queens++;
                break;
            default:
                break;
        }
    }
    
    // Any pawns, rooks, or queens means sufficient material
    if (white_pawns > 0 || black_pawns > 0 || white_rooks > 0 || black_rooks > 0 || 
        white_queens > 0 || black_queens > 0) {
        return false;
    }
    
    // Count total minor pieces for each side
    int white_minor = white_knights + white_bishops;
    int black_minor = black_knights + black_bishops;
    
    // Insufficient material cases:
    // King vs King
    if (white_minor == 0 && black_minor == 0) return true;
    
    // King + minor piece vs King
    if ((white_minor <= 1 && black_minor == 0) || (black_minor <= 1 && white_minor == 0)) return true;
    
    // King + Bishop vs King + Bishop (same color squares) - simplified to any bishop vs bishop
    if (white_minor == 1 && black_minor == 1 && white_bishops == 1 && black_bishops == 1) return true;
    
    return false;
}

void init(CChess* env) {
    printf("init called\n");
    memset(&env->context, 0, sizeof(ChessContext));
    memset(&env->log, 0, sizeof(Log));
    
    // Set up convenience pointer to avoid repeated dereferencing
    env->ctx = &env->context;
    
    init_board(&env->ctx->board);
    env->ctx->dual_agent_self_play_mode = true; // Default to self-play
    
    // Copy reward config to context
    env->ctx->c_reward_valid = env->reward_valid;
    env->ctx->c_reward_invalid_white = env->reward_invalid_white;
    env->ctx->c_reward_invalid_black = env->reward_invalid_black;
    env->ctx->c_reward_agent_captures_enemy_piece = env->reward_agent_captures_enemy_piece;
    env->ctx->c_reward_enemy_captures_agent_piece = env->reward_enemy_captures_agent_piece;
    env->ctx->c_reward_draw = env->reward_draw;
    env->ctx->c_reward_win_white = env->reward_win_white;
    env->ctx->c_reward_win_black = env->reward_win_black;
    env->ctx->c_reward_loss_white = env->reward_loss_white;
    env->ctx->c_reward_loss_black = env->reward_loss_black;
    env->ctx->c_reward_check_white = env->reward_check_white;
    env->ctx->c_reward_check_black = env->reward_check_black;
    env->ctx->c_reward_material_diff_white = env->reward_material_diff_white;
    env->ctx->c_reward_material_diff_black = env->reward_material_diff_black;
}

void allocate(CChess* env) {
    // Allocate RL interface arrays for PufferLib
    // Chess has 2 players but typically trains as single agent with perspective flipping
    const int num_players = 2;
    const int obs_size = 3268;  // 21*8*8 board planes + 1924 action mask = 1344 + 1924
    const int action_space = 1924;  // TOTAL_CHESS_ACTIONS
    
    env->observations = (float*)calloc(num_players * obs_size, sizeof(float));
    env->actions = (int*)calloc(num_players, sizeof(int));
    env->rewards = (float*)calloc(num_players, sizeof(float));
    env->terminals = (unsigned char*)calloc(num_players, sizeof(unsigned char));
    
    // Call core initialization after memory allocation
    init(env);
}

void free_allocated(CChess* env) {
    // Free RL interface arrays allocated by allocate()
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);

    c_close(env);
}

void c_reset(CChess* env) {
    printf("c_reset called\n");
    // Reset board to starting position
    init_board(&env->ctx->board);
    
    // Reset episode tracking
    env->ctx->step_count = 0;
    env->ctx->episode_return_white = 0.0f;
    env->ctx->episode_return_black = 0.0f;
    env->ctx->complete_game_action_count = 0;
    env->ctx->serialized_moves[0] = '\0';  // Initialize serialized_moves buffer to empty
    
    // Reset statistics
    env->ctx->c_white_moves = 0;
    env->ctx->c_black_moves = 0;
    env->ctx->c_valid_moves = 0;
    env->ctx->c_invalid_moves_white = 0;
    env->ctx->c_invalid_moves_black = 0;
    
    // Reset accumulated reward counters
    env->ctx->accumulated_reward_valid = 0.0f;
    env->ctx->accumulated_reward_agent_captures_enemy_piece = 0.0f;
    env->ctx->accumulated_reward_enemy_captures_agent_piece = 0.0f;
    env->ctx->accumulated_reward_draw = 0.0f;
    env->ctx->accumulated_reward_win_white = 0.0f;
    env->ctx->accumulated_reward_win_black = 0.0f;
    env->ctx->accumulated_reward_loss_white = 0.0f;
    env->ctx->accumulated_reward_loss_black = 0.0f;
    env->ctx->accumulated_reward_draw_white = 0.0f;
    env->ctx->accumulated_reward_draw_black = 0.0f;
    env->ctx->accumulated_reward_check_white = 0.0f;
    env->ctx->accumulated_reward_check_black = 0.0f;
    env->ctx->accumulated_reward_material_diff_white = 0.0f;
    env->ctx->accumulated_reward_material_diff_black = 0.0f;
    env->ctx->accumulated_stockfish_eval = 0.0f;
    
    // Clear caches
    env->ctx->legal_moves_cached = false;
    
    // Clear position history
    memset(&env->ctx->position_history, 0, sizeof(PositionHistory));
    
    // Add starting position to history for threefold repetition detection
    add_position_to_history(&env->context, env->ctx->board.zobrist_hash);
    
    // Compute initial observation
    compute_observation_with_perspective(env, &env->context);
}

// TODO: step_agent() steps through both white and black, so 1 call to c_step() is 1 whole
// move == 2 half moves
// Then, in c_step,    for (int i = 0; i < env->num_agents; i++) {
// step_agent(env, i);
// }
void c_step(CChess* env) {
    if (env->terminals[0] == 1 || env->terminals[1] == 1) {
        printf("terminals: %d\n", env->terminals[0]);
        fflush(stdout);
    }
    
    // CRITICAL: Generate legal moves once at start of step for current position
    chess_generate_legal_moves_uci(&env->context);

    
    // Get action from agent
    int action_idx = env->actions[0];
    // Determine whose turn it is
    Color current_player = env->ctx->board.to_move;
    
    // Clear rewards
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    
    // Convert action to UCI (from current player's perspective)
    if (action_idx < 0 || action_idx >= TOTAL_CHESS_ACTIONS) {
        // Invalid action index
        if (current_player == WHITE) {
            env->rewards[0] += env->reward_invalid_white;  // Use += not =
            env->ctx->c_invalid_moves_white += 1;
            // Accumulate for logging
            env->ctx->accumulated_reward_valid += env->reward_invalid_white; // Invalid moves count as negative valid
        } else {
            env->rewards[0] += env->reward_invalid_black;  // Use += not =
            env->ctx->c_invalid_moves_black += 1;
            // Accumulate for logging
            env->ctx->accumulated_reward_valid += env->reward_invalid_black; // Invalid moves count as negative valid
        }
        // Compute observation using already generated legal moves
        compute_observation_with_perspective(env, &env->context);
        return;
    }
    
    const char* uci_move = ACTION_ID_TO_UCI[action_idx];
    
    // If Black is moving, need to un-flip the UCI to apply to canonical board
    char canonical_uci[6];
    if (current_player == BLACK) {
        flip_uci_for_black_perspective(uci_move, canonical_uci);
        uci_move = canonical_uci;
    }
    
    // Check if move is legal using already generated moves
    bool is_legal = false;
    for (int i = 0; i < env->ctx->legal_moves_count; i++) {
        if (strcmp(env->ctx->legal_moves_buffer[i], uci_move) == 0) {
            is_legal = true;
            break;
        }
    }
    
    if (is_legal) {
        // Apply move
        apply_uci_move(&env->context, uci_move);
        
        // Give reward to current player
        if (current_player == WHITE) {
            env->rewards[0] += env->reward_valid;  // Use += not =
            env->ctx->c_white_moves += 1;
            // Accumulate for logging
            env->ctx->accumulated_reward_valid += env->reward_valid;
        } else {
            env->rewards[0] += env->reward_valid;  // Use += not =
            env->ctx->c_black_moves += 1;
            // Accumulate for logging
            env->ctx->accumulated_reward_valid += env->reward_valid;
        }
        env->ctx->c_valid_moves += 1;

        bool game_over = false;

        printf("step_count from c_step() before any_legal_move_exists: %d\n", env->ctx->step_count);
        if (env->ctx->step_count > 100) {
          game_over = true;
          // Also set the c_max_depth counter so we know this triggered
          env->ctx->c_max_depth += 1;
          printf("DEBUG: Forcing game over by max_depth at step %d\n",
                 env->ctx->step_count);
          fflush(stdout); // Force the printout immediately
        }

        // Use yield-based generation to check if any legal move exists
        bool any_legal_move_exists = false;
        chess_generate_legal_moves_yield(&env->context, first_move_callback,
                                         &any_legal_move_exists);

        if (!any_legal_move_exists) {
          game_over = true;
          if (is_in_check(&env->ctx->board, env->ctx->board.to_move)) {
            // CHECKMATE: The previous player delivered a checkmate.
            if (current_player == WHITE) { // White just moved and won
              env->rewards[0] += env->reward_win_white;
              env->rewards[0] += env->reward_loss_black;
              env->ctx->c_black_checkmated += 1;
              env->ctx->c_white_win += 1;
              env->ctx->c_black_loss += 1;
            } else { // Black just moved and won
              env->rewards[0] += env->reward_win_black;
              env->rewards[0] += env->reward_loss_white;
              env->ctx->c_white_checkmated += 1;
              env->ctx->c_black_win += 1;
              env->ctx->c_white_loss += 1;
            }
          } else {
            // STALEMATE
            game_over = true;
            env->rewards[0] += env->reward_draw;
            env->ctx->c_stalemate += 1;
            env->ctx->c_game_drawn += 1;
          }
        } else if (env->ctx->board.halfmove_clock >= 100) {
          // FIFTY-MOVE RULE
          game_over = true;
          env->rewards[0] += env->reward_draw;
          env->ctx->c_fifty_move_rule += 1;
          env->ctx->c_game_drawn += 1;
        } else if (is_threefold_repetition(&env->context)) {
          // THREEFOLD REPETITION
          game_over = true;
          env->rewards[0] += env->reward_draw;
          env->ctx->c_threefold_repetition += 1;
          env->ctx->c_game_drawn += 1;
        } else if (is_insufficient_material(&env->context)) {
          // INSUFFICIENT MATERIAL
          game_over = true;
          env->rewards[0] += env->reward_draw;
          env->ctx->c_insufficient_material += 1;
          env->ctx->c_game_drawn += 1;
        } else if (env->max_depth > 0 &&
                   env->ctx->step_count >= env->max_depth) {
          // MAX DEPTH / TRUNCATION
          game_over = true;
          env->rewards[0] += env->reward_draw; // Treat as a draw
          env->rewards[1] += env->reward_draw;
          env->ctx->c_max_depth += 1;
          env->ctx->c_game_drawn += 1;
        }

        if (game_over) {
            printf("DEBUG: Game over detected! Setting terminals and calling add_log\n");
            env->terminals[0] = 1;
            
            // Log complete game actions efficiently
            env->log.complete_game_move_count = (float)env->ctx->complete_game_action_count;
            // Note: serialize_complete_game_moves not used due to PufferLib float-only logging
            
            // CRITICAL: Call add_log() on terminal=true for PufferLib aggregation
            add_log(env);
            
            // AUTO-RESET: Start new game immediately like other PufferLib environments
            c_reset(env);
        }
        
    } else {
        // Invalid move
        if (current_player == WHITE) {
            env->rewards[0] += env->reward_invalid_white;  // Use += not =
            env->ctx->c_invalid_moves_white += 1;
            // Accumulate for logging
            env->ctx->accumulated_reward_valid += env->reward_invalid_white; // Invalid moves count as negative valid
        } else {
            env->rewards[0] += env->reward_invalid_black;  // Use += not =
            env->ctx->c_invalid_moves_black += 1;
            // Accumulate for logging
            env->ctx->accumulated_reward_valid += env->reward_invalid_black; // Invalid moves count as negative valid
        }
    }
    
    // Update episode returns
    env->ctx->episode_return_white += env->rewards[0];
    
    // Compute new observation
    compute_observation_with_perspective(env, &env->context);
}

// === PUFFERLIB LOGGING FUNCTION ===
void add_log(CChess* env) {
    printf("DEBUG add_log: step=%d, white_return=%.2f, black_return=%.2f, n=%.1f\n", 
           env->ctx->step_count, env->ctx->episode_return_white, env->ctx->episode_return_black, env->log.n);
    
    // Aggregate counters into log structure using = for PufferLib (CRITICAL!)
    env->log.episode_length = (float)env->ctx->step_count;
    env->log.episode_return = env->ctx->episode_return_white + env->ctx->episode_return_black;
    env->log.episode_return_white = env->ctx->episode_return_white;
    env->log.episode_return_black = env->ctx->episode_return_black;
    
    // Reward aggregates (from accumulated counters during this game)
    env->log.reward_valid = env->ctx->accumulated_reward_valid;
    env->log.reward_agent_captures_enemy_piece = env->ctx->accumulated_reward_agent_captures_enemy_piece;
    env->log.reward_enemy_captures_agent_piece = env->ctx->accumulated_reward_enemy_captures_agent_piece;
    env->log.reward_draw = env->ctx->accumulated_reward_draw;
    env->log.reward_win_white = env->ctx->accumulated_reward_win_white;
    env->log.reward_win_black = env->ctx->accumulated_reward_win_black;
    env->log.reward_loss_white = env->ctx->accumulated_reward_loss_white;
    env->log.reward_loss_black = env->ctx->accumulated_reward_loss_black;
    env->log.reward_draw_white = env->ctx->accumulated_reward_draw_white;
    env->log.reward_draw_black = env->ctx->accumulated_reward_draw_black;
    env->log.reward_check_white = env->ctx->accumulated_reward_check_white;
    env->log.reward_check_black = env->ctx->accumulated_reward_check_black;
    env->log.reward_material_diff_white = env->ctx->accumulated_reward_material_diff_white;
    env->log.reward_material_diff_black = env->ctx->accumulated_reward_material_diff_black;
    env->log.stockfish_eval = env->ctx->accumulated_stockfish_eval;
    
    // Game outcome counters (use incremental values from current game)
    env->log.white_win = (float)env->ctx->c_white_win;
    env->log.white_loss = (float)env->ctx->c_white_loss;
    env->log.black_win = (float)env->ctx->c_black_win;
    env->log.black_loss = (float)env->ctx->c_black_loss;
    env->log.game_drawn = (float)env->ctx->c_game_drawn;
    env->log.stalemate = (float)env->ctx->c_stalemate;
    env->log.insufficient_material = (float)env->ctx->c_insufficient_material;
    env->log.threefold_repetition = (float)env->ctx->c_threefold_repetition;
    env->log.fifty_move_rule = (float)env->ctx->c_fifty_move_rule;
    env->log.max_depth = (float)env->ctx->c_max_depth;
    env->log.white_checkmated = (float)env->ctx->c_white_checkmated;
    env->log.black_checkmated = (float)env->ctx->c_black_checkmated;
    
    // Move statistics
    env->log.white_moves = (float)env->ctx->c_white_moves;
    env->log.black_moves = (float)env->ctx->c_black_moves;
    env->log.valid_moves = (float)env->ctx->c_valid_moves;
    env->log.invalid_moves_white = (float)env->ctx->c_invalid_moves_white;
    env->log.invalid_moves_black = (float)env->ctx->c_invalid_moves_black;
    
    // Castling and special moves
    env->log.en_passant_white = (float)env->ctx->c_en_passant_white;
    env->log.en_passant_black = (float)env->ctx->c_en_passant_black;
    env->log.white_castle_kingside = (float)env->ctx->c_white_castle_kingside;
    env->log.white_castle_queenside = (float)env->ctx->c_white_castle_queenside;
    env->log.black_castle_kingside = (float)env->ctx->c_black_castle_kingside;
    env->log.black_castle_queenside = (float)env->ctx->c_black_castle_queenside;
    
    // Promotion statistics
    env->log.white_promotion_count = (float)env->ctx->c_white_promotion_count;
    env->log.white_promotion_knight = (float)env->ctx->c_white_promotion_knight;
    env->log.white_promotion_bishop = (float)env->ctx->c_white_promotion_bishop;
    env->log.white_promotion_rook = (float)env->ctx->c_white_promotion_rook;
    env->log.white_promotion_queen = (float)env->ctx->c_white_promotion_queen;
    env->log.black_promotion_count = (float)env->ctx->c_black_promotion_count;
    env->log.black_promotion_knight = (float)env->ctx->c_black_promotion_knight;
    env->log.black_promotion_bishop = (float)env->ctx->c_black_promotion_bishop;
    env->log.black_promotion_rook = (float)env->ctx->c_black_promotion_rook;
    env->log.black_promotion_queen = (float)env->ctx->c_black_promotion_queen;
    
    // Calculate performance metrics after aggregation
    float total_games = env->log.white_win + env->log.white_loss + env->log.game_drawn;
    if (total_games > 0) {
        env->log.perf = env->log.white_win / total_games;  // White win rate
    }
    env->log.score = env->log.white_win - env->log.white_loss;  // Win-loss difference
    
    // Increment n (must be last for PufferLib aggregation)
    env->log.n = 1.0f;
}

void c_render(CChess* env) {
    
    printf("\n  +---+---+---+---+---+---+---+---+\n");
    for (int y = 7; y >= 0; y--) {
        printf("%d |", y + 1);
        for (int x = 0; x < 8; x++) {
            const Piece* p = get_piece_const(&env->ctx->board, x, y);
            char piece_char = ' ';
            
            if (p && p->type != EMPTY) {
                const char pieces[] = " KQRBNP";
                piece_char = pieces[p->type];
                if (p->color == BLACK) {
                    piece_char = piece_char + ('a' - 'A'); // Make lowercase
                }
            }
            
            printf(" %c |", piece_char);
        }
        printf("\n  +---+---+---+---+---+---+---+---+\n");
    }
    printf("    a   b   c   d   e   f   g   h\n");
    printf("\nTo move: %s\n", (env->ctx->board.to_move == WHITE) ? "White" : "Black");
    printf("Step: %d\n", env->ctx->step_count);
}

void c_close(CChess* env) {
    // Core cleanup for chess environment
    // Currently all major data structures use static allocation within CChess struct
    // Future: Clean up any Stockfish process handles, pipes, or other system resources
    // when Stockfish integration is fully implemented
    
    // Clear sensitive data and reset state
    memset(&env->context, 0, sizeof(ChessContext));
    memset(&env->log, 0, sizeof(Log));
}

// === DUAL AGENT SELF-PLAY MODE SETTERS ===

void set_dual_agent_self_play_mode(CChess* env, bool enabled) {
    env->context.dual_agent_self_play_mode = enabled;
}

void set_self_play_mode(CChess* env, bool enabled) {
    env->context.self_play_mode = enabled;
}

void enable_stockfish_black(CChess* env, const char* stockfish_cmd, int elo, int search_ms) {
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
        strncpy(env->context.stockfish_cmd, stockfish_cmd, sizeof(env->context.stockfish_cmd) - 1);
        env->context.stockfish_elo = elo;
        env->context.stockfish_search_ms = search_ms;
        DBG("Stockfish integration stubbed - needs C implementation\n");
    }
}

// FEN support (basic)
void c_set_fen(CChess* env, const char* fen) {
    // TODO: Implement FEN parsing
    init_board(&env->context.board);
    compute_observation_with_perspective(env, &env->context);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CHESS_H