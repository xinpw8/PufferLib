// chess.h
#pragma once

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sstream>

#ifdef __cplusplus
extern "C" {
#endif

// pufferlib required structs
typedef struct Log {
    float perf;
    float score;
    float episode_length;
    float episode_return;
    float reward_valid;
    float reward_invalid;
    float reward_agent_captures_enemy_piece;
    float reward_enemy_captures_agent_piece;
    float reward_win;
    float reward_draw;
    float reward_loss;
    float game_won;
    float game_lost;
    float game_drawn;
    float n;

    float stalemate;
    float insufficient_material;
    float threefold_repetition;
    float white_checkmated;
    float black_checkmated;
    float fifty_move_rule;
    float max_depth;
} Log;

typedef struct CChess {
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    
    // static values from config/ocean/chess.ini
    float reward_valid;
    float reward_invalid;
    float reward_agent_captures_enemy_piece;
    float reward_enemy_captures_agent_piece;
    float reward_win;
    float reward_draw;
    float reward_loss;

    // opaque pointer to C++ context
    void* context;
} CChess;

typedef struct ChessContext ChessContext;

// pufferlib required functions
void init(CChess* env);
void allocate(CChess* env);
void free_allocated(CChess* env);
void c_reset(CChess* env);
void compute_observation(CChess* env, ChessContext* ctx);
void add_log(CChess* env, const ChessContext* ctx, bool win, bool loss, bool draw);
void c_step(CChess* env);
void c_render(CChess* env);
void c_close(CChess* env);

#ifdef __cplusplus
} // extern "C"
#endif

// c++ implementation
#ifdef __cplusplus

#include <vector>
#include <unordered_map>
#include <optional>
#include <random>
#include <algorithm>
#include <iostream>
#include <functional>
#include <cassert>

namespace chess {

// piece types matching openspiel's encoding for nn compatibility
enum PieceType : uint8_t {
    EMPTY = 0,
    KING = 1,
    QUEEN = 2,
    ROOK = 3,
    BISHOP = 4,
    KNIGHT = 5,
    PAWN = 6
};

enum Color : uint8_t {
    WHITE = 0,
    BLACK = 1,
    NO_COLOR = 2
};

struct Square {
    int8_t x, y;
    
    bool operator==(const Square& other) const {
        return x == other.x && y == other.y;
    }
    
    bool is_valid() const {
        return x >= 0 && x < 8 && y >= 0 && y < 8;
    }
    
    int index() const { return y * 8 + x; }
};

struct Piece {
    Color color;
    PieceType type;
    
    bool operator==(const Piece& other) const {
        return color == other.color && type == other.type;
    }
    
    int8_t to_obs() const {
        if (type == EMPTY) return 0;
        int sign = (color == WHITE) ? 1 : -1;
        return sign * static_cast<int8_t>(type);
    }
};

struct Move {
    Square from;
    Square to;
    Piece piece;
    PieceType promotion;
    bool is_castle_short = false;
    bool is_castle_long = false;
    
    bool operator==(const Move& other) const {
        return from == other.from && to == other.to && 
               piece == other.piece && promotion == other.promotion &&
               is_castle_short == other.is_castle_short &&
               is_castle_long == other.is_castle_long;
    }
};

// zobrist hashing for position identification
class ZobristHash {
    uint64_t pieces[2][7][64];  // [color][piece_type][square]
    uint64_t castling[4];
    uint64_t ep_file[8];
    uint64_t black_to_move;
    
public:
    ZobristHash() {
        std::mt19937_64 rng(12345);  // fixed seed
        
        for (int c = 0; c < 2; c++) {
            for (int p = 0; p < 7; p++) {
                for (int sq = 0; sq < 64; sq++) {
                    pieces[c][p][sq] = rng();
                }
            }
        }
        
        for (int i = 0; i < 4; i++) castling[i] = rng();
        for (int i = 0; i < 8; i++) ep_file[i] = rng();
        black_to_move = rng();
    }
    
    uint64_t hash_position(const Piece board[64], uint8_t castle_rights, 
                          int8_t ep_file, Color to_move) const {
        uint64_t hash = 0;
        
        for (int sq = 0; sq < 64; sq++) {
            if (board[sq].type != EMPTY) {
                hash ^= pieces[board[sq].color][board[sq].type][sq];
            }
        }
        
        for (int i = 0; i < 4; i++) {
            if (castle_rights & (1 << i)) {
                hash ^= castling[i];
            }
        }
        
        if (ep_file >= 0) {
            hash ^= this->ep_file[ep_file];
        }
        
        if (to_move == BLACK) {
            hash ^= black_to_move;
        }
        
        return hash;
    }
};

// main chess board class
class ChessBoard {
    Piece board[64];
    Color to_move = WHITE;
    uint8_t castling_rights = 0xF;  // KQkq
    int8_t ep_square = -1;  // En passant target square
    uint8_t halfmove_clock = 0;
    uint16_t fullmove_number = 1;
    
    // performance optimizations
    mutable std::optional<std::vector<Move>> cached_legal_moves;
    static ZobristHash zobrist;
    
public:
    ChessBoard() {
        reset();
    }
    
    void reset() {
        // standard starting position
        const char* start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        set_from_fen(start_fen);
        cached_legal_moves.reset();
    }
    
    void set_from_fen(const char* fen) {
        memset(board, 0, sizeof(board));
        
        int x = 0, y = 7;
        const char* p = fen;
        
        // parse board
        while (*p && *p != ' ') {
            if (*p == '/') {
                x = 0;
                y--;
            } else if (*p >= '1' && *p <= '8') {
                x += (*p - '0');
            } else {
                Color color = (*p >= 'A' && *p <= 'Z') ? WHITE : BLACK;
                PieceType type = EMPTY;
                
                switch (*p | 32) {  // Convert to lowercase
                    case 'k': type = KING; break;
                    case 'q': type = QUEEN; break;
                    case 'r': type = ROOK; break;
                    case 'b': type = BISHOP; break;
                    case 'n': type = KNIGHT; break;
                    case 'p': type = PAWN; break;
                }
                
                if (type != EMPTY) {
                    board[y * 8 + x] = {color, type};
                    x++;
                }
            }
            p++;
        }
        
        // parse side to move
        if (*p == ' ') p++;
        to_move = (*p == 'w') ? WHITE : BLACK;
        
        // parse castling rights
        castling_rights = 0;
        if (*p == ' ') p++;
        if (*p == ' ') p++;
        while (*p && *p != ' ') {
            switch (*p) {
                case 'K': castling_rights |= 0x8; break;
                case 'Q': castling_rights |= 0x4; break;
                case 'k': castling_rights |= 0x2; break;
                case 'q': castling_rights |= 0x1; break;
            }
            p++;
        }
        
        // parse en passant
        ep_square = -1;
        if (*p == ' ') p++;
        if (*p && *p != '-' && *p != ' ') {
            int file = *p - 'a';
            p++;
            int rank = *p - '1';
            ep_square = rank * 8 + file;
        }
        
        // parse halfmove clock
        if (*p == ' ') p++;
        if (*p == ' ') p++;
        halfmove_clock = 0;
        while (*p && *p != ' ') {
            halfmove_clock = halfmove_clock * 10 + (*p - '0');
            p++;
        }
        
        // parse fullmove number
        if (*p == ' ') p++;
        fullmove_number = 0;
        while (*p && *p != ' ') {
            fullmove_number = fullmove_number * 10 + (*p - '0');
            p++;
        }
    }
    
    const Piece& at(Square sq) const {
        static const Piece empty = {NO_COLOR, EMPTY};
        if (!sq.is_valid()) return empty;
        return board[sq.index()];
    }
    
    Color side_to_move() const { return to_move; }
    uint8_t get_castling_rights() const { return castling_rights; }
    int8_t get_ep_square() const { return ep_square; }
    uint8_t get_halfmove_clock() const { return halfmove_clock; }
    
    std::string fen() const {
        std::string result;
        
        // Board representation
        for (int y = 7; y >= 0; y--) {
            int empty_count = 0;
            for (int x = 0; x < 8; x++) {
                const Piece& p = board[y * 8 + x];
                if (p.type == EMPTY) {
                    empty_count++;
                } else {
                    if (empty_count > 0) {
                        result += std::to_string(empty_count);
                        empty_count = 0;
                    }
                    char piece_char;
                    switch (p.type) {
                        case KING: piece_char = 'k'; break;
                        case QUEEN: piece_char = 'q'; break;
                        case ROOK: piece_char = 'r'; break;
                        case BISHOP: piece_char = 'b'; break;
                        case KNIGHT: piece_char = 'n'; break;
                        case PAWN: piece_char = 'p'; break;
                        default: piece_char = '?'; break;
                    }
                    if (p.color == WHITE) {
                        piece_char = std::toupper(piece_char);
                    }
                    result += piece_char;
                }
            }
            if (empty_count > 0) {
                result += std::to_string(empty_count);
            }
            if (y > 0) result += '/';
        }
        
        // Side to move
        result += (to_move == WHITE) ? " w " : " b ";
        
        // Castling rights
        bool has_castling = false;
        if (castling_rights & 0x8) { result += 'K'; has_castling = true; }
        if (castling_rights & 0x4) { result += 'Q'; has_castling = true; }
        if (castling_rights & 0x2) { result += 'k'; has_castling = true; }
        if (castling_rights & 0x1) { result += 'q'; has_castling = true; }
        if (!has_castling) result += '-';
        
        // En passant
        result += ' ';
        if (ep_square >= 0) {
            int file = ep_square & 7;
            int rank = ep_square >> 3;
            result += ('a' + file);
            result += ('1' + rank);
        } else {
            result += '-';
        }
        
        // Halfmove clock and fullmove number
        result += ' ';
        result += std::to_string(halfmove_clock);
        result += ' ';
        result += std::to_string(fullmove_number);
        
        return result;
    }
    
    uint64_t hash() const {
        int8_t ep_file = (ep_square >= 0) ? (ep_square & 7) : -1;
        return zobrist.hash_position(board, castling_rights, ep_file, to_move);
    }
    
    Square find_king(Color color) const {
        for (int sq = 0; sq < 64; ++sq) {
            if (board[sq].type == KING && board[sq].color == color) {
                return {int8_t(sq & 7), int8_t(sq >> 3)};
            }
        }
        return {-1, -1};
    }
    
    // move generation with yield pattern for performance
    template<typename Fn>
    void generate_pseudo_legal_moves(Fn yield) const {
        for (int sq = 0; sq < 64; sq++) {
            const Piece& piece = board[sq];
            if (piece.type == EMPTY || piece.color != to_move) continue;
            
            Square from = {int8_t(sq & 7), int8_t(sq >> 3)};
            
            switch (piece.type) {
                case PAWN: generate_pawn_moves(from, yield); break;
                case KNIGHT: generate_knight_moves(from, yield); break;
                case BISHOP: generate_bishop_moves(from, yield); break;
                case ROOK: generate_rook_moves(from, yield); break;
                case QUEEN: generate_queen_moves(from, yield); break;
                case KING: generate_king_moves(from, yield); break;
            }
        }
    }
    
    const std::vector<Move>& legal_moves() const {
        if (!cached_legal_moves) {
            cached_legal_moves = std::vector<Move>();
            
            generate_pseudo_legal_moves([this](const Move& move) {
                ChessBoard test = *this;
                test.apply_move_unchecked(move);
                
                if (!test.is_in_check(to_move)) {
                    cached_legal_moves->push_back(move);
                }
                return true;  // continue generating
            });
        }
        return *cached_legal_moves;
    }
    
    bool apply_move(const Move& move) {
        // validate move is legal
        const auto& moves = legal_moves();
        if (std::find(moves.begin(), moves.end(), move) == moves.end()) {
            return false;
        }
        
        apply_move_unchecked(move);
        cached_legal_moves.reset();
        return true;
    }
    
    bool is_check() const {
        return is_in_check(to_move);
    }
    
    bool is_checkmate() const {
        return is_in_check(to_move) && legal_moves().empty();
    }
    
    bool is_stalemate() const {
        return !is_in_check(to_move) && legal_moves().empty();
    }
    
    bool is_insufficient_material() const {
        int white_material = 0, black_material = 0;
        int white_bishops = 0, black_bishops = 0;
        int white_knights = 0, black_knights = 0;
        bool light_bishop[2] = {false, false};
        
        for (int sq = 0; sq < 64; sq++) {
            const Piece& p = board[sq];
            if (p.type == EMPTY) continue;
            
            switch (p.type) {
                case PAWN:
                case ROOK:
                case QUEEN:
                    return false;  // sufficient material
                    
                case BISHOP:
                    if (p.color == WHITE) {
                        white_bishops++;
                        white_material += 3;
                    } else {
                        black_bishops++;
                        black_material += 3;
                    }
                    light_bishop[p.color] = ((sq & 7) + (sq >> 3)) & 1;
                    break;
                    
                case KNIGHT:
                    if (p.color == WHITE) {
                        white_knights++;
                        white_material += 3;
                    } else {
                        black_knights++;
                        black_material += 3;
                    }
                    break;
            }
        }
        
        // K vs K
        if (white_material == 0 && black_material == 0) return true;
        
        // K+B vs K or K+N vs K
        if ((white_material == 3 && black_material == 0) ||
            (white_material == 0 && black_material == 3)) return true;
        
        // K+B vs K+B with same colored bishops
        if (white_bishops == 1 && black_bishops == 1 && 
            light_bishop[WHITE] == light_bishop[BLACK]) return true;
        
        return false;
    }
    
    // convert move to openspiel's action index (0-4673)
    static int move_to_action(const Move& move) {
        if (move.is_castle_short) return 4672;
        if (move.is_castle_long) return 4673;
        
        int from_idx = move.from.index();
        int to_idx = move.to.index();
        int base_action = from_idx * 73 + to_idx;
        
        // handle promotions
        if (move.promotion != EMPTY) {
            // map promotion piece to offset: Q=0, R=1, B=2, N=3
            int promo_offset = 0;
            switch (move.promotion) {
                case QUEEN: promo_offset = 0; break;
                case ROOK: promo_offset = 1; break;
                case BISHOP: promo_offset = 2; break;
                case KNIGHT: promo_offset = 3; break;
            }
            base_action = 64 * 64 + from_idx * 4 + promo_offset;
        } else {
            // Normal moves (non-promotion)
            int dest_val = -1; // This will be the relative 'dest_idx' (0-72)

            int dx = move.to.x - move.from.x;
            int dy = move.to.y - move.from.y;

            // 1. Knight moves (dest_val 65-72)
            const int knight_moves[8][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
            for (int i = 0; i < 8; ++i) {
                if (dx == knight_moves[i][0] && dy == knight_moves[i][1]) {
                    dest_val = 65 + i;
                    break;
                }
            }

            // 2. Sliding moves (Queen-like, dest_val 9-64)
            if (dest_val == -1) {
                const int directions[8][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
                for (int i = 0; i < 8; ++i) {
                    int dir_dx = directions[i][0];
                    int dir_dy = directions[i][1];
                    
                    // Check if this move is in this direction
                    if (dir_dx == 0 && dir_dy != 0) { // Vertical movement
                        if (dx == 0 && dy != 0 && ((dy > 0) == (dir_dy > 0))) {
                            int dist = abs(dy);
                            if (dist >= 1 && dist <= 7) {
                                dest_val = 9 + (i * 7 + (dist - 1));
                                break;
                            }
                        }
                    } else if (dir_dx != 0 && dir_dy == 0) { // Horizontal movement
                        if (dy == 0 && dx != 0 && ((dx > 0) == (dir_dx > 0))) {
                            int dist = abs(dx);
                            if (dist >= 1 && dist <= 7) {
                                dest_val = 9 + (i * 7 + (dist - 1));
                                break;
                            }
                        }
                    } else if (dir_dx != 0 && dir_dy != 0) { // Diagonal movement
                        if (dx != 0 && dy != 0 && abs(dx) == abs(dy) && 
                            ((dx > 0) == (dir_dx > 0)) && ((dy > 0) == (dir_dy > 0))) {
                            int dist = abs(dx);
                            if (dist >= 1 && dist <= 7) {
                                dest_val = 9 + (i * 7 + (dist - 1));
                                break;
                            }
                        }
                    }
                }
            }

            // If dest_val is still -1, it means the move is not found in the defined action types.
            // This implies it's either an invalid move or a special pawn move that is not captured by sliding logic.
            // However, pawn single/double pushes and captures should map to these sliding moves.
            // The current action_to_move_direct handles pawn moves implicitly through the sliding/knight categories.
            
            if (dest_val == -1) {
                // This is a critical error in the action encoding/decoding.
                // It means a legal chess::Move object cannot be converted into a valid OpenSpiel action index.
                // For now, return a placeholder that will likely result in an invalid move in c_step.
                return -1; 
            }

            base_action = from_idx * 73 + dest_val;
        }
        
        return base_action;
    }
    
    // debug rendering
    void render() const {
        std::cout << "\n  a b c d e f g h\n";
        for (int y = 7; y >= 0; y--) {
            std::cout << (y + 1) << " ";
            for (int x = 0; x < 8; x++) {
                const Piece& p = board[y * 8 + x];
                char c = '.';
                if (p.type != EMPTY) {
                    const char* pieces = " KQRBNP";
                    c = pieces[p.type];
                    if (p.color == BLACK) c += 32;  // Lowercase
                }
                std::cout << c << " ";
            }
            std::cout << (y + 1) << "\n";
        }
        std::cout << "  a b c d e f g h\n";
        std::cout << (to_move == WHITE ? "White" : "Black") << " to move\n";
    }
    
private:
    void apply_move_unchecked(const Move& move) {
        // Update halfmove clock
        if (move.piece.type == PAWN || at(move.to).type != EMPTY) {
            halfmove_clock = 0;
        } else {
            halfmove_clock++;
        }
        
        // handle castling
        if (move.is_castle_short || move.is_castle_long) {
            int rank = (to_move == WHITE) ? 0 : 7;
            int king_file = 4;
            int rook_from = move.is_castle_short ? 7 : 0;
            int rook_to = move.is_castle_short ? 5 : 3;
            int king_to = move.is_castle_short ? 6 : 2;
            
            // move king
            board[rank * 8 + king_to] = board[rank * 8 + king_file];
            board[rank * 8 + king_file] = {NO_COLOR, EMPTY};
            
            // move rook
            board[rank * 8 + rook_to] = board[rank * 8 + rook_from];
            board[rank * 8 + rook_from] = {NO_COLOR, EMPTY};
        } else {
            // handle en passant capture
            if (move.piece.type == PAWN && move.to.index() == ep_square) {
                int captured_pawn = move.to.x + move.from.y * 8;
                board[captured_pawn] = {NO_COLOR, EMPTY};
            }
            
            // move piece
            board[move.to.index()] = board[move.from.index()];
            board[move.from.index()] = {NO_COLOR, EMPTY};
            
            // handle promotion
            if (move.promotion != EMPTY) {
                board[move.to.index()].type = move.promotion;
            }
            
            // update en passant square
            ep_square = -1;
            if (move.piece.type == PAWN) {
                int dy = move.to.y - move.from.y;
                if (abs(dy) == 2) {
                    ep_square = move.from.x + (move.from.y + dy/2) * 8;
                }
            }
        }
        
        // update castling rights
        if (move.piece.type == KING) {
            if (to_move == WHITE) {
                castling_rights &= ~0xC;  // Clear KQ
            } else {
                castling_rights &= ~0x3;  // Clear kq
            }
        } else if (move.piece.type == ROOK) {
            if (move.from.index() == 0) castling_rights &= ~0x4;  // Q
            if (move.from.index() == 7) castling_rights &= ~0x8;  // K
            if (move.from.index() == 56) castling_rights &= ~0x1; // q
            if (move.from.index() == 63) castling_rights &= ~0x2; // k
        }
        
        // captures affect opponent's castling rights
        if (move.to.index() == 0) castling_rights &= ~0x4;   // Q
        if (move.to.index() == 7) castling_rights &= ~0x8;   // K
        if (move.to.index() == 56) castling_rights &= ~0x1;  // q
        if (move.to.index() == 63) castling_rights &= ~0x2;  // k
        
        // switch sides
        to_move = (to_move == WHITE) ? BLACK : WHITE;
        if (to_move == WHITE) fullmove_number++;
    }
    
    bool is_in_check(Color color) const {
        // find king
        Square king_sq = {-1, -1};
        for (int sq = 0; sq < 64; sq++) {
            if (board[sq].type == KING && board[sq].color == color) {
                king_sq = {int8_t(sq & 7), int8_t(sq >> 3)};
                break;
            }
        }
        
        if (!king_sq.is_valid()) return false;
        
        // check if any opponent piece attacks the king
        Color opp = (color == WHITE) ? BLACK : WHITE;
        return is_square_attacked(king_sq, opp);
    }
    
    bool is_square_attacked(Square sq, Color by_color) const {
        // check pawn attacks
        int pawn_dir = (by_color == WHITE) ? 1 : -1;
        for (int dx = -1; dx <= 1; dx += 2) {
            Square from = {int8_t(sq.x + dx), int8_t(sq.y - pawn_dir)};
            if (from.is_valid() && at(from).type == PAWN && at(from).color == by_color) {
                return true;
            }
        }
        
        // check knight attacks
        const int knight_moves[][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
        for (auto& m : knight_moves) {
            Square from = {int8_t(sq.x + m[0]), int8_t(sq.y + m[1])};
            if (from.is_valid() && at(from).type == KNIGHT && at(from).color == by_color) {
                return true;
            }
        }
        
        // check sliding pieces
        const int directions[][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
        for (int i = 0; i < 8; i++) {
            bool is_diagonal = (i == 0 || i == 2 || i == 5 || i == 7);
            
            for (int dist = 1; dist < 8; dist++) {
                Square from = {int8_t(sq.x + directions[i][0] * dist), 
                              int8_t(sq.y + directions[i][1] * dist)};
                if (!from.is_valid()) break;
                
                const Piece& p = at(from);
                if (p.type == EMPTY) continue;
                if (p.color != by_color) break;
                
                // king (distance 1 only)
                if (dist == 1 && p.type == KING) return true;
                
                // queen
                if (p.type == QUEEN) return true;
                
                // bishop on diagonal, rook on straight
                if (is_diagonal && p.type == BISHOP) return true;
                if (!is_diagonal && p.type == ROOK) return true;
                
                break;  // piece blocks further squares
            }
        }
        
        return false;
    }
    
    // move generation helpers
    template<typename Fn>
    void generate_pawn_moves(Square from, Fn yield) const {
        const Piece& piece = at(from);
        int dir = (piece.color == WHITE) ? 1 : -1;
        int start_rank = (piece.color == WHITE) ? 1 : 6;
        int promo_rank = (piece.color == WHITE) ? 7 : 0;
        
        // forward moves
        Square to = {from.x, int8_t(from.y + dir)};
        if (to.is_valid() && at(to).type == EMPTY) {
            if (to.y == promo_rank) {
                // generate all promotions
                for (auto promo : {QUEEN, ROOK, BISHOP, KNIGHT}) {
                    if (!yield(Move{from, to, piece, promo})) return;
                }
            } else {
                if (!yield(Move{from, to, piece, EMPTY})) return;
                
                // double push from start
                if (from.y == start_rank) {
                    Square to2 = {from.x, int8_t(from.y + 2*dir)};
                    if (at(to2).type == EMPTY) {
                        if (!yield(Move{from, to2, piece, EMPTY})) return;
                    }
                }
            }
        }
        
        // captures
        for (int dx = -1; dx <= 1; dx += 2) {
            Square cap = {int8_t(from.x + dx), int8_t(from.y + dir)};
            if (!cap.is_valid()) continue;
            
            bool can_capture = (at(cap).type != EMPTY && at(cap).color != piece.color) ||
                              (cap.index() == ep_square);
            
            if (can_capture) {
                if (cap.y == promo_rank) {
                    for (auto promo : {QUEEN, ROOK, BISHOP, KNIGHT}) {
                        if (!yield(Move{from, cap, piece, promo})) return;
                    }
                } else {
                    if (!yield(Move{from, cap, piece, EMPTY})) return;
                }
            }
        }
    }
    
    template<typename Fn>
    void generate_knight_moves(Square from, Fn yield) const {
        const Piece& piece = at(from);
        const int moves[][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
        
        for (auto& m : moves) {
            Square to = {int8_t(from.x + m[0]), int8_t(from.y + m[1])};
            if (to.is_valid() && at(to).color != piece.color) {
                if (!yield(Move{from, to, piece, EMPTY})) return;
            }
        }
    }
    
    template<typename Fn>
    void generate_sliding_moves(Square from, const int dirs[][2], int num_dirs, Fn yield) const {
        const Piece& piece = at(from);
        
        for (int d = 0; d < num_dirs; d++) {
            for (int dist = 1; dist < 8; dist++) {
                Square to = {int8_t(from.x + dirs[d][0] * dist), 
                            int8_t(from.y + dirs[d][1] * dist)};
                if (!to.is_valid()) break;
                
                if (at(to).type == EMPTY) {
                    if (!yield(Move{from, to, piece, EMPTY})) return;
                } else {
                    if (at(to).color != piece.color) {
                        if (!yield(Move{from, to, piece, EMPTY})) return;
                    }
                    break;  // cannot jump over pieces
                }
            }
        }
    }
    
    template<typename Fn>
    void generate_bishop_moves(Square from, Fn yield) const {
        const int dirs[][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};
        generate_sliding_moves(from, dirs, 4, yield);
    }
    
    template<typename Fn>
    void generate_rook_moves(Square from, Fn yield) const {
        const int dirs[][2] = {{-1,0},{1,0},{0,-1},{0,1}};
        generate_sliding_moves(from, dirs, 4, yield);
    }
    
    template<typename Fn>
    void generate_queen_moves(Square from, Fn yield) const {
        const int dirs[][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
        generate_sliding_moves(from, dirs, 8, yield);
    }
    
    template<typename Fn>
    void generate_king_moves(Square from, Fn yield) const {
        const Piece& piece = at(from);
        const int dirs[][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
        
        // regular moves
        for (auto& d : dirs) {
            Square to = {int8_t(from.x + d[0]), int8_t(from.y + d[1])};
            if (to.is_valid() && at(to).color != piece.color) {
                if (!yield(Move{from, to, piece, EMPTY})) return;
            }
        }
        
        // castling
        if (!is_in_check(piece.color)) {
            int rank = (piece.color == WHITE) ? 0 : 7;
            uint8_t short_mask = (piece.color == WHITE) ? 0x8 : 0x2;
            uint8_t long_mask = (piece.color == WHITE) ? 0x4 : 0x1;
            
            // short castle
            if ((castling_rights & short_mask) &&
                at({5, int8_t(rank)}).type == EMPTY &&
                at({6, int8_t(rank)}).type == EMPTY &&
                !is_square_attacked({5, int8_t(rank)}, (piece.color == WHITE) ? BLACK : WHITE)) {
                
                Move m{from, {6, int8_t(rank)}, piece, EMPTY};
                m.is_castle_short = true;
                if (!yield(m)) return;
            }
            
            // long castle
            if ((castling_rights & long_mask) &&
                at({1, int8_t(rank)}).type == EMPTY &&
                at({2, int8_t(rank)}).type == EMPTY &&
                at({3, int8_t(rank)}).type == EMPTY &&
                !is_square_attacked({3, int8_t(rank)}, (piece.color == WHITE) ? BLACK : WHITE)) {
                
                Move m{from, {2, int8_t(rank)}, piece, EMPTY};
                m.is_castle_long = true;
                if (!yield(m)) return;
            }
        }
    }
};

// static initialization
ZobristHash ChessBoard::zobrist;
Move action_to_move_direct(int action, const ChessBoard& board) {
    // Special case: castling
    if (action == 4672 || action == 4673) {
        Square king_sq = board.find_king(board.side_to_move());
        if (action == 4672) { // Kingside
            Move m{king_sq, {6, king_sq.y}, {board.side_to_move(), KING}, EMPTY};
            m.is_castle_short = true;
            return m;
        } else { // Queenside
            Move m{king_sq, {2, king_sq.y}, {board.side_to_move(), KING}, EMPTY};
            m.is_castle_long = true;
            return m;
        }
    }
    
    // Handle promotions (actions 4096-4351)
    if (action >= 4096 && action < 4352) {
        int promo_action = action - 4096;
        int from_idx = promo_action / 4;
        int promo_type = promo_action % 4;
        
        Square from = {int8_t(from_idx % 8), int8_t(from_idx / 8)};
        const Piece& piece = board.at(from);
        
        // Determine promotion piece (Q=0, R=1, B=2, N=3)
        PieceType promotion = QUEEN;
        switch (promo_type) {
            case 0: promotion = QUEEN; break;
            case 1: promotion = ROOK; break;
            case 2: promotion = BISHOP; break;
            case 3: promotion = KNIGHT; break;
        }
        
        // For promotions, we need to figure out the destination
        // Promotions only happen when pawns reach the last rank
        int dir = (board.side_to_move() == WHITE) ? 1 : -1;
        int dest_rank = (board.side_to_move() == WHITE) ? 7 : 0;
        
        // Check straight ahead
        Square to = {from.x, int8_t(dest_rank)};
        if (board.at(to).type == EMPTY) {
            return Move{from, to, piece, promotion};
        }
        
        // Check diagonal captures
        for (int dx = -1; dx <= 1; dx += 2) {
            to = {int8_t(from.x + dx), int8_t(dest_rank)};
            if (to.is_valid() && board.at(to).type != EMPTY && board.at(to).color != piece.color) {
                return Move{from, to, piece, promotion};
            }
        }
        
        // This shouldn't happen with valid actions
        return Move{{-1, -1}, {-1, -1}, {NO_COLOR, EMPTY}, EMPTY};
    }
    
    // Regular moves (0-4095)
    int from_idx = action / 73;
    int dest_val = action % 73;
    
    Square from = {int8_t(from_idx % 8), int8_t(from_idx / 8)};
    const Piece& piece = board.at(from);
    
    // Determine destination based on dest_val
    Square to;
    
    if (dest_val >= 0 && dest_val < 9) {
        // Underpromotion moves (should not happen here, handled above)
        // This range should be empty for regular moves
        return Move{{-1, -1}, {-1, -1}, {NO_COLOR, EMPTY}, EMPTY};
    } else if (dest_val >= 9 && dest_val < 65) {
        // Sliding moves (9-64): queen-like moves in 8 directions, up to 7 squares
        int move_idx = dest_val - 9;
        int direction = move_idx / 7;
        int distance = (move_idx % 7) + 1;
        
        // Direction vectors: MUST match move_to_action directions exactly
        const int directions[8][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
        
        to = {int8_t(from.x + directions[direction][0] * distance),
              int8_t(from.y + directions[direction][1] * distance)};
    } else if (dest_val >= 65 && dest_val < 73) {
        // Knight moves (65-72)
        int knight_idx = dest_val - 65;
        // MUST match move_to_action knight_moves exactly
        const int knight_moves[8][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
        
        to = {int8_t(from.x + knight_moves[knight_idx][0]),
              int8_t(from.y + knight_moves[knight_idx][1])};
    } else {
        // Invalid dest_val
        return Move{{-1, -1}, {-1, -1}, {NO_COLOR, EMPTY}, EMPTY};
    }
    
    // Validate destination is on board
    if (!to.is_valid()) {
        return Move{{-1, -1}, {-1, -1}, {NO_COLOR, EMPTY}, EMPTY};
    }
    
    return Move{from, to, piece, EMPTY};
}

// passthrough hash for repetition detection
struct PassthroughHash {
    size_t operator()(uint64_t x) const { return x; }
};

} // namespace chess

struct ChessContext {
    chess::ChessBoard board;
    std::unordered_map<uint64_t, int, chess::PassthroughHash> position_history;
    std::mt19937 rng;
    
    // episode tracking vars
    int step_count = 0;
    float episode_return = 0.0f;
    
    // config vars from python
    float c_reward_valid = 0.0f;
    float c_reward_invalid = 0.0f;
    float c_reward_agent_captures_enemy_piece = 0.0f;
    float c_reward_enemy_captures_agent_piece = 0.0f;
    float c_reward_win = 0.0f;
    float c_reward_draw = 0.0f;
    float c_reward_loss = 0.0f;

    // env logging vars
    float c_game_won = 0.0f;
    float c_game_lost = 0.0f;
    float c_game_drawn = 0.0f;
    float c_n = 0.0f;
    float c_stalemate = 0.0f;
    float c_insufficient_material = 0.0f;
    float c_threefold_repetition = 0.0f;
    float c_fifty_move_rule = 0.0f;
    float c_max_depth = 0.0f;
    float c_white_checkmated = 0.0f;
    float c_black_checkmated = 0.0f;

    // max depth
    int max_depth = 10000;

    // Self-play support
    bool self_play_mode = false;
bool waiting_for_black_move = false;

    ChessContext(unsigned seed) : rng(seed) {}
};

extern "C" void set_self_play_mode(CChess* env, bool enabled) {
    auto* ctx = (ChessContext*)env->context;
    ctx->self_play_mode = enabled;
}

extern "C" {

void add_log(CChess* env, const ChessContext* ctx, bool win, bool loss, bool draw) {
    env->log.n += 1;
    env->log.episode_length += ctx->step_count;
    env->log.episode_return += ctx->episode_return;
    env->log.game_won += win;
    env->log.game_lost += loss;
    env->log.game_drawn += draw;
    env->log.reward_valid += ctx->c_reward_valid;
    env->log.reward_invalid += ctx->c_reward_invalid;
    env->log.reward_agent_captures_enemy_piece += ctx->c_reward_agent_captures_enemy_piece;
    env->log.reward_enemy_captures_agent_piece += ctx->c_reward_enemy_captures_agent_piece;
    env->log.reward_win += ctx->c_reward_win;
    env->log.reward_draw += ctx->c_reward_draw;
    env->log.reward_loss += ctx->c_reward_loss;
    env->log.stalemate += ctx->c_stalemate;
    env->log.insufficient_material += ctx->c_insufficient_material;
    env->log.threefold_repetition += ctx->c_threefold_repetition;
    env->log.fifty_move_rule += ctx->c_fifty_move_rule;
    env->log.max_depth += ctx->c_max_depth;
    env->log.white_checkmated += ctx->c_white_checkmated;
    env->log.black_checkmated += ctx->c_black_checkmated;
}

void init(CChess* env) {
    env->context = new ChessContext(12345);
}

void allocate(CChess* env) {
    // 8×8×21 = 1344 board features + 4674 legal move mask = 6018
    env->observations = (float*)calloc(6018, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void free_allocated(CChess* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    delete (ChessContext*)env->context;
}

void c_reset(CChess* env) {
    auto* ctx = (ChessContext*)env->context;
    
    // Reset board
    ctx->board.reset();
    ctx->position_history.clear();
    ctx->position_history[ctx->board.hash()] = 1;
    
    // Reset episode tracking
    ctx->step_count = 0;
    ctx->episode_return = 0.0f;

    // zero counters
    ctx->c_reward_valid = 0.0f;
    ctx->c_reward_invalid = 0.0f;
    ctx->c_reward_agent_captures_enemy_piece = 0.0f;
    ctx->c_reward_enemy_captures_agent_piece = 0.0f;
    ctx->c_reward_win = 0.0f;
    ctx->c_reward_draw = 0.0f;
    ctx->c_reward_loss = 0.0f;

    ctx->c_game_won = 0.0f;
    ctx->c_game_lost = 0.0f;
    ctx->c_game_drawn = 0.0f;
    ctx->c_n = 0.0f;
    ctx->c_stalemate = 0.0f;
    ctx->c_insufficient_material = 0.0f;
    ctx->c_threefold_repetition = 0.0f;
    ctx->c_fifty_move_rule = 0.0f;
    ctx->c_max_depth = 0.0f;
    ctx->c_white_checkmated = 0.0f;
    ctx->c_black_checkmated = 0.0f;
    
    // Reset self-play state - always start with white's turn
    ctx->waiting_for_black_move = false;
    
    // CRITICAL: Ensure observation is written
    compute_observation(env, ctx);
    
    // CRITICAL: Ensure outputs are initialized
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;
    

}

void compute_observation(CChess* env, ChessContext* ctx) {
    // Total size: 8*8*21 + 4674 legal move mask = 6018 floats
    int idx = 0;
    
    // 12 piece planes (6 types × 2 colors)
    for (int color = 0; color < 2; color++) {
        for (int type = 1; type <= 6; type++) { // KING=1 to PAWN=6
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    const auto& p = ctx->board.at({int8_t(x), int8_t(y)});
                    env->observations[idx++] = (p.color == color && p.type == type) ? 1.0f : 0.0f;
                }
            }
        }
    }
    
    // Empty squares plane
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            env->observations[idx++] = ctx->board.at({int8_t(x), int8_t(y)}).type == chess::EMPTY ? 1.0f : 0.0f;
        }
    }
    
    // Repetition count plane (normalized to 0-1 for 1-3 repetitions)
    uint64_t hash = ctx->board.hash();
    int reps = ctx->position_history[hash];
    float rep_val = (reps - 1) / 2.0f; // Maps 1->0, 2->0.5, 3->1
    for (int i = 0; i < 64; i++) {
        env->observations[idx++] = rep_val;
    }
    
    // Side to move plane
    float side_val = ctx->board.side_to_move() == chess::WHITE ? 0.0f : 1.0f;
    for (int i = 0; i < 64; i++) {
        env->observations[idx++] = side_val;
    }
    
    // Irreversible move counter plane (halfmove clock normalized to 0-1)
    float halfmove_val = ctx->board.get_halfmove_clock() / 101.0f;
    for (int i = 0; i < 64; i++) {
        env->observations[idx++] = halfmove_val;
    }
    
    // 4 castling rights planes
    uint8_t rights = ctx->board.get_castling_rights();
    for (int i = 0; i < 4; i++) {
        float castle_val = (rights & (1 << (3-i))) ? 1.0f : 0.0f;
        for (int j = 0; j < 64; j++) {
            env->observations[idx++] = castle_val;
        }
    }
    
    // En passant target square plane
    int8_t ep_square = ctx->board.get_ep_square();
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            env->observations[idx++] = (ep_square == y * 8 + x) ? 1.0f : 0.0f;
        }
    }
    
    // Should be exactly 1344 floats at this point
    assert(idx == 1344);
    
    // legal move mask (4674 values)
    // Initialize all to 0 (illegal)
    for (int i = 0; i < 4674; i++) {
        env->observations[idx + i] = 0.0f;
    }
    
    // Set legal moves to 1
    // Get legal moves from chess engine
    const auto& legal_moves = ctx->board.legal_moves();

    // For each legal move, set corresponding mask bit to 1
    for (const auto& move : legal_moves) {
        int action_idx = chess::ChessBoard::move_to_action(move);
        // move_to_action converts chess move to 0-4673 index
        if (action_idx >= 0 && action_idx < 4674) {
            env->observations[idx + action_idx] = 1.0f;
        }
    }
    
    // obs is now [board_features_0, ..., board_features_1343, legal_move_mask_0, ..., legal_move_mask_4673]
    // board_features is 1344 floats, legal_move_mask is 4674 floats
    idx += 4674;
    assert(idx == 6018);
}
void c_step(CChess* env) {
    auto* ctx = (ChessContext*)env->context;
    
    // Auto-reset if game is terminal
    if (env->terminals[0] == 1) {
        c_reset(env);
    }

    float reward = 0.0f;
    bool terminal = false;
    bool win = false;
    bool loss = false;
    bool draw = false;
    
    // Self-play turn tracking
    const bool handling_black = ctx->self_play_mode && ctx->waiting_for_black_move;

    ctx->step_count += 1;
    
    // Get the action
    int action_idx = *env->actions;
    
    // Decode the move
    chess::Move selected_move = chess::action_to_move_direct(action_idx, ctx->board);
    
    // Validate move
    bool valid_move = false;
    const auto& legal_moves = ctx->board.legal_moves();
    
    // Check if decoded move is in legal moves
    for (const auto& legal_move : legal_moves) {
        if (selected_move == legal_move) {
            valid_move = true;
            break;
        }
    }

    if (!valid_move) {
        reward = env->reward_invalid;
        ctx->c_reward_invalid += env->reward_invalid;
    } else {
        
        // Check for capture before move
        const auto& captured = ctx->board.at(selected_move.to);
        if (captured.type != chess::EMPTY) {
            if (handling_black) {
                reward += env->reward_enemy_captures_agent_piece;
                ctx->c_reward_enemy_captures_agent_piece += env->reward_enemy_captures_agent_piece;
            } else {
                reward += env->reward_agent_captures_enemy_piece;
                ctx->c_reward_agent_captures_enemy_piece += env->reward_agent_captures_enemy_piece;
            }
        }
        
        // Apply move
        ctx->board.apply_move(selected_move);
        if (!handling_black) {
            reward += env->reward_valid;
            ctx->c_reward_valid += env->reward_valid;
        }
        
        // Update position history
        uint64_t hash = ctx->board.hash();
        ctx->position_history[hash]++;
        
        // Check if move ended the game
        
        // 1. Threefold repetition
        if (ctx->position_history[hash] >= 3) {
            terminal = true;
            draw = true;
            ctx->c_threefold_repetition += 1;
            reward += env->reward_draw;
            ctx->c_reward_draw += env->reward_draw;
        }
        // 2. 50-move rule
        else if (ctx->board.get_halfmove_clock() >= 100) {
            terminal = true;
            draw = true;
            ctx->c_fifty_move_rule += 1;
            reward += env->reward_draw;
            ctx->c_reward_draw += env->reward_draw;
        }
        // 3. Insufficient material
        else if (ctx->board.is_insufficient_material()) {
            terminal = true;
            draw = true;
            ctx->c_insufficient_material += 1;
            reward += env->reward_draw;
            ctx->c_reward_draw += env->reward_draw;
        }
        // 4. Max depth
        else if (ctx->step_count >= ctx->max_depth) {
            terminal = true;
            draw = true;
            ctx->c_max_depth += 1;
            reward += env->reward_draw;
            ctx->c_reward_draw += env->reward_draw;
        }
        // 5. No legal moves for the opponent whose turn is next
        else if (ctx->board.legal_moves().empty()) {
            terminal = true;
            if (ctx->board.is_check()) {
                if (handling_black) {
                    // Black has no moves and is in check → white wins
                    win = true;
                    ctx->c_black_checkmated += 1;
                    reward += env->reward_win;
                    ctx->c_reward_win += env->reward_win;
                } else {
                    // White has no moves and is in check → black wins
                    loss = true;
                    ctx->c_white_checkmated += 1;
                    reward += env->reward_loss;
                    ctx->c_reward_loss += env->reward_loss;
                }
            } else {
                // Stalemate
                draw = true;
                ctx->c_stalemate += 1;
                reward += env->reward_draw;
                ctx->c_reward_draw += env->reward_draw;
            }
        }
    }
    
    // Turn management -------------------------------------------------------
    if (ctx->self_play_mode) {
        if (!terminal) {
            ctx->waiting_for_black_move = !ctx->waiting_for_black_move;
        }
        // No automatic random move in self-play
    } else {
        // Original random black move (only after white’s move and if game not over)
        if (!handling_black && !terminal) {
            const auto& black_moves = ctx->board.legal_moves();
            if (!black_moves.empty()) {
                std::uniform_int_distribution<> dist(0, black_moves.size() - 1);
                const auto& black_move = black_moves[dist(ctx->rng)];
                
                // Capture
                if (ctx->board.at(black_move.to).type != chess::EMPTY) {
                    reward += env->reward_enemy_captures_agent_piece;
                    ctx->c_reward_enemy_captures_agent_piece += env->reward_enemy_captures_agent_piece;
                }
                
                ctx->board.apply_move(black_move);
                ctx->position_history[ctx->board.hash()]++;
                
                // Repeat terminal checks for black’s move
                if (ctx->position_history[ctx->board.hash()] >= 3) {
                    terminal = true;
                    draw = true;
                    ctx->c_threefold_repetition += 1;
                    reward += env->reward_draw;
                    ctx->c_reward_draw += env->reward_draw;
                } else if (ctx->board.get_halfmove_clock() >= 100) {
                    terminal = true;
                    draw = true;
                    ctx->c_fifty_move_rule += 1;
                    reward += env->reward_draw;
                    ctx->c_reward_draw += env->reward_draw;
                } else if (ctx->board.is_insufficient_material()) {
                    terminal = true;
                    draw = true;
                    ctx->c_insufficient_material += 1;
                    reward += env->reward_draw;
                    ctx->c_reward_draw += env->reward_draw;
                } else if (ctx->step_count >= ctx->max_depth) {
                    terminal = true;
                    draw = true;
                    ctx->c_max_depth += 1;
                    reward += env->reward_draw;
                    ctx->c_reward_draw += env->reward_draw;
                } else if (ctx->board.legal_moves().empty()) {
                    terminal = true;
                    if (ctx->board.is_check()) {
                        loss = true;
                        ctx->c_white_checkmated += 1;
                        reward += env->reward_loss;
                        ctx->c_reward_loss += env->reward_loss;
                    } else {
                        draw = true;
                        ctx->c_stalemate += 1;
                        reward += env->reward_draw;
                        ctx->c_reward_draw += env->reward_draw;
                    }
                }
            }
        }
    }
    
    // Final max-depth check
    if (!terminal && ctx->step_count >= ctx->max_depth) {
        terminal = true;
        draw = true;
        ctx->c_max_depth += 1;
        reward += env->reward_draw;
        ctx->c_reward_draw += env->reward_draw;
    }

    // Always update observation to reflect current state
    compute_observation(env, ctx);
    
    // Set outputs
    env->rewards[0] = reward;
    env->terminals[0] = terminal ? 1 : 0;
    
    // Add episode return tracking
    ctx->episode_return += reward;
    
    if (terminal) {
        add_log(env, ctx, win, loss, draw);
        // envs in pufferlib MUST RESET THEMSELVES!
        // reset called at top of c_step for clarity
    }
}

void c_render(CChess* env) {
    // Capture the board representation as a string
    std::stringstream ss;
    ss << "\n  a b c d e f g h\n";
    const auto& board = ((ChessContext*)env->context)->board;
    for (int y = 7; y >= 0; y--) {
        ss << (y + 1) << " ";
        for (int x = 0; x < 8; x++) {
            const chess::Piece& p = board.at({int8_t(x), int8_t(y)});
            char c = '.';
            if (p.type != chess::EMPTY) {
                const char* pieces = " KQRBNP";
                c = pieces[p.type];
                if (p.color == chess::BLACK) c += 32;  // Lowercase
            }
            ss << c << " ";
        }
        ss << (y + 1) << "\n";
    }
    ss << "  a b c d e f g h\n";
    ss << (board.side_to_move() == chess::WHITE ? "White" : "Black") << " to move\n";
    
    // Store the result in a global buffer that Python can access
    static std::string render_buffer;
    render_buffer = ss.str();
    // Note: This is a temporary fix. A proper solution would involve modifying the Python binding
    // to return this string, but for now we'll use a global buffer
}

void c_close(CChess* env) {
    // close nothing?
}

} // extern "C"
#endif // __cplusplus