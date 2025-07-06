// chess.h
#pragma once

// Comment out the next line (or pass -DDEBUG_LOG=0 on the compiler command line)
// to turn every DBG() call into a no-op.
#ifndef DEBUG_LOG
#define DEBUG_LOG 0          // 0 = disabled, 1 = enabled
#endif

#if DEBUG_LOG
  #include <iostream>
  #define DBG(expr) do { std::cerr << expr; } while (0)
#else
  #define DBG(expr) do { } while (0)
#endif

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sstream>
#include <string>
#include <utility>
#ifdef __cplusplus
#include "stockfish_wrapper.h"
#include <mutex>
#include <memory>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// pufferlib required structs
typedef struct Log {
    float perf;
    float score;
    float episode_length;
    float episode_return;            // combined (existing)
    float episode_return_white;      // new – white perspective total
    float episode_return_black;      // new – black perspective total
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
    float invalid_moves;
    float invalid_moves_white;
    float invalid_moves_black;
    float reward_check;
    float reward_material_diff;
    float stockfish_eval;
    float n;
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
    float reward_check;
    int max_depth;
    float reward_material_diff;

    // Stockfish integration
    bool stockfish_enabled;
    
    // Debug helper: when true, compute_observation will NOT mask legal moves
    bool debug_disable_mask;

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

// Enable Stockfish as the black-side opponent. Pass binary path or NULL for default.
void enable_stockfish_black(CChess* env, const char* stockfish_cmd, int elo, int search_ms);

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
#include <mutex>

namespace chess {

static constexpr int kNumActionDestinations = 73;
static constexpr int kNumUnderPromotions    = 9;   // 3 pieces × 3 dirs

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
    
    // Convenience inequality operator – avoids repetitive !(a == b)
    bool operator!=(const Move& other) const { return !(*this == other); }
};

inline constexpr Move kPassMove{{-1,-1},{-1,-1},{NO_COLOR, EMPTY}, EMPTY};

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
    uint32_t fullmove_number = 1;  // Use 32-bit to prevent overflow
    
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
        // Initialize all members to default values first
        to_move = WHITE;
        castling_rights = 0;
        ep_square = -1;
        halfmove_clock = 0;
        fullmove_number = 1;
        
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
        p++;
        
        // parse castling rights
        castling_rights = 0;
        if (*p == ' ') p++;
        
        // Debug print
        DBG("Castling rights char: '" << *p << "'" << std::endl);
        
        // CORRECTED LOGIC
        if (*p == ' ') p++; // Skip leading space
        while (*p && *p != ' ') {
            switch (*p) {
                case 'K': castling_rights |= 0x8; break;
                case 'Q': castling_rights |= 0x4; break;
                case 'k': castling_rights |= 0x2; break;
                case 'q': castling_rights |= 0x1; break;
                case '-': break; // Handle the case of no castling rights
                default:  break; // Or handle unexpected characters
            }
            p++;
        }        
        
        // Debug print
        DBG("Final castling rights: 0x" << std::hex << (int)castling_rights << std::dec << std::endl);
        
        // parse en passant
        ep_square = -1;
        if (*p == ' ') p++;
        if (*p && *p != '-' && *p != ' ') {
            int file = *p - 'a';
            p++;
            if (*p && *p >= '1' && *p <= '8') {
                int rank = *p - '1';
                ep_square = rank * 8 + file;
            }
            p++;
        } else if (*p == '-') {
            p++; // Skip the '-'
        }
        
        // parse halfmove clock
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
            
            // Debug print
            DBG("Found piece: ");
            const char* type_names[] = {"EMPTY", "KING", "QUEEN", "ROOK", "BISHOP", "KNIGHT", "PAWN"};
            const char* color_names[] = {"WHITE", "BLACK", "NO_COLOR"};
            DBG(color_names[piece.color] << " " << type_names[piece.type]);
            DBG(" at " << char('a' + from.x) << (from.y + 1) << std::endl);
            
            switch (piece.type) {
                case PAWN: 
                    DBG("  Generating pawn moves..." << std::endl);
                    generate_pawn_moves(from, yield); 
                    break;
                case KNIGHT: 
                    DBG("  Generating knight moves..." << std::endl);
                    generate_knight_moves(from, yield); 
                    break;
                case BISHOP: 
                    DBG("  Generating bishop moves..." << std::endl);
                    generate_bishop_moves(from, yield); 
                    break;
                case ROOK: 
                    DBG("  Generating rook moves..." << std::endl);
                    generate_rook_moves(from, yield); 
                    break;
                case QUEEN: 
                    DBG("  Generating queen moves..." << std::endl);        
                    generate_queen_moves(from, yield); 
                    break;
                case KING: 
                    DBG("  Generating king moves..." << std::endl);
                    generate_king_moves(from, yield); 
                    break;
            }
        }
    }
    
    const std::vector<Move>& legal_moves() const {
        if (!cached_legal_moves) {
            cached_legal_moves = std::vector<Move>();
            
            DBG("Generating legal moves..." << std::endl);
            generate_pseudo_legal_moves([this](const Move& move) {
                DBG("  Testing move: " << int(move.from.x) << "," << int(move.from.y) << " -> " << int(move.to.x) << "," << int(move.to.y) << std::endl);
                
                ChessBoard test = *this;
                test.apply_move_unchecked(move);
                
                if (!test.is_in_check(to_move)) {
                    DBG("    Legal move found!" << std::endl);
                    cached_legal_moves->push_back(move);
                } else {
                    DBG("    Move leaves king in check, discarded" << std::endl);
                }
                return true;  // continue generating
            });
            
            // Sort deterministically by action id for reproducibility
            std::sort(cached_legal_moves->begin(), cached_legal_moves->end(),
                      [](const Move& a, const Move& b){
                          return ChessBoard::move_to_action(a) < ChessBoard::move_to_action(b);
                      });

            DBG("Total legal moves: " << cached_legal_moves->size() << std::endl);
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
        // Count pieces by type for each color
        int white_pawns = 0, black_pawns = 0;
        int white_rooks = 0, black_rooks = 0;
        int white_queens = 0, black_queens = 0;
        int white_bishops = 0, black_bishops = 0;
        int white_knights = 0, black_knights = 0;
        bool white_light_bishop = false, white_dark_bishop = false;
        bool black_light_bishop = false, black_dark_bishop = false;
        
        for (int sq = 0; sq < 64; sq++) {
            const Piece& p = board[sq];
            if (p.type == EMPTY) continue;
            
            switch (p.type) {
                case PAWN:
                    if (p.color == WHITE) white_pawns++;
                    else black_pawns++;
                    break;
                    
                case ROOK:
                    if (p.color == WHITE) white_rooks++;
                    else black_rooks++;
                    break;
                    
                case QUEEN:
                    if (p.color == WHITE) white_queens++;
                    else black_queens++;
                    break;
                    
                case BISHOP:
                    if (p.color == WHITE) {
                        white_bishops++;
                        // Check if bishop is on light or dark square
                        if (((sq & 7) + (sq >> 3)) & 1) {
                            white_light_bishop = true;
                        } else {
                            white_dark_bishop = true;
                        }
                    } else {
                        black_bishops++;
                        // Check if bishop is on light or dark square
                        if (((sq & 7) + (sq >> 3)) & 1) {
                            black_light_bishop = true;
                        } else {
                            black_dark_bishop = true;
                        }
                    }
                    break;
                    
                case KNIGHT:
                    if (p.color == WHITE) white_knights++;
                    else black_knights++;
                    break;
                    
                case KING:
                    // Kings don't affect insufficient material calculation
                    break;
            }
        }
        
        // If either side has pawns, rooks, or queens, there is sufficient material
        if (white_pawns > 0 || black_pawns > 0 || 
            white_rooks > 0 || black_rooks > 0 || 
            white_queens > 0 || black_queens > 0) {
            return false;
        }
        
        // Now we only have kings, bishops, and knights
        
        // Rule 1: K vs K
        if (white_bishops == 0 && white_knights == 0 && 
            black_bishops == 0 && black_knights == 0) {
            return true;
        }
        
        // Rule 2: K+B vs K (White has one bishop, Black has nothing)
        if (white_bishops == 1 && white_knights == 0 && 
            black_bishops == 0 && black_knights == 0) {
            return true;
        }
        
        // Rule 2: K vs K+B (Black has one bishop, White has nothing)
        if (black_bishops == 1 && black_knights == 0 && 
            white_bishops == 0 && white_knights == 0) {
            return true;
        }
        
        // Rule 3: K+N vs K (White has one knight, Black has nothing)
        if (white_knights == 1 && white_bishops == 0 && 
            black_bishops == 0 && black_knights == 0) {
            return true;
        }
        
        // Rule 3: K vs K+N (Black has one knight, White has nothing)
        if (black_knights == 1 && black_bishops == 0 && 
            white_bishops == 0 && white_knights == 0) {
            return true;
        }
        
        // Rule 4: K+B* vs K+B* (all bishops on same colored squares)
        if (white_knights == 0 && black_knights == 0 && 
            white_bishops > 0 && black_bishops > 0) {
            // All bishops must be on the same color squares
            bool all_on_light = (white_light_bishop || black_light_bishop) && 
                               !white_dark_bishop && !black_dark_bishop;
            bool all_on_dark = (white_dark_bishop || black_dark_bishop) && 
                              !white_light_bishop && !black_light_bishop;
            
            if (all_on_light || all_on_dark) {
                return true;
            }
        }
        
        // Additional insufficient material cases:
        // K+B vs K+B (same color bishops)
        if (white_bishops == 1 && white_knights == 0 && 
            black_bishops == 1 && black_knights == 0) {
            bool same_color_bishops = (white_light_bishop && black_light_bishop) || 
                                     (white_dark_bishop && black_dark_bishop);
            if (same_color_bishops) {
                return true;
            }
        }
        
        // K+N vs K+N
        if (white_knights == 1 && white_bishops == 0 && 
            black_knights == 1 && black_bishops == 0) {
            return true;
        }
        
        // K+B vs K+N (generally considered insufficient, though technically possible to mate)
        // This is debatable, but many engines consider this insufficient
        if ((white_bishops == 1 && white_knights == 0 && 
             black_knights == 1 && black_bishops == 0) ||
            (white_knights == 1 && white_bishops == 0 && 
             black_bishops == 1 && black_knights == 0)) {
            return true;
        }
        
        return false;
    }
    
    // follows openspiel encoding logic
    static int move_to_action(const Move& move) {
        // Pass move occupies dedicated slot 0, matching OpenSpiel spec
        if (move == kPassMove)         return 0;

        // Castling slots immediately follow the 64×73 block (4672 indices)
        if (move.is_castle_long)       return 4672;   // queenside (left)
        if (move.is_castle_short)      return 4673;   // kingside (right)

        // --- rotate board so mover is always White --------------------------------
        Move m = move;
        if (m.piece.color == Color::BLACK) {
            m.from.y = 7 - m.from.y;
            m.to.y   = 7 - m.to.y;
        }

        const int from_base =
            (m.from.x * 8 + m.from.y) * kNumActionDestinations;

        const int dx = m.to.x - m.from.x;   // file  difference
        const int dy = m.to.y - m.from.y;   // rank difference

        // --- under-promotions ------------------------------------------------------
        if (m.promotion != PieceType::QUEEN && m.promotion != PieceType::EMPTY) {
            // OpenSpiel ordering:
            //   Under-promotion piece order: Knight, Bishop, Rook
            static constexpr PieceType kUnder[3] = {
                KNIGHT, BISHOP, ROOK
            };
            const int promo_index = std::find(std::begin(kUnder), std::end(kUnder), m.promotion) - std::begin(kUnder);

            // Direction order: left capture (-1), straight (0), right capture (+1)
            const int dir_index =
                (dx == -1) ? 0 : (dx == 0) ? 1 : 2;

            return from_base + promo_index * 3 + dir_index;
        }

        // --- queen-style moves -----------------------------------------------------
        int dest_index = -1;
        if (dx == 0) {                               // vertical
            if (dy > 0)        dest_index =        (dy - 1);          // N 0-6
            else               dest_index = 28 + (-dy - 1);           // S 28-34
        } else if (dy == 0) {                        // horizontal
            if (dx > 0)        dest_index = 14 + (dx - 1);            // E 14-20
            else               dest_index = 42 + (-dx - 1);           // W 42-48
        } else if (dx ==  dy) {                      // main diagonal
            if (dx > 0)        dest_index =  7 + (dx - 1);            // NE 7-13
            else               dest_index = 35 + (-dx - 1);           // SW 35-41
        } else if (dx == -dy) {                      // anti-diagonal
            if (dx > 0)        dest_index = 21 + (dx - 1);            // SE 21-27
            else               dest_index = 49 + (-dx - 1);           // NW 49-55
        } else {                                     // knight
            static constexpr int kKnight[8][2] = {
                {-2,-1}, {-2, 1}, {-1,-2}, {-1, 2},
                { 2,-1}, { 2, 1}, { 1,-2}, { 1, 2}
            };
            for (int i = 0; i < 8; ++i)
                if (dx == kKnight[i][0] && dy == kKnight[i][1]) {
                    dest_index = 56 + i;             // 56-63
                    break;
                }
        }

        return from_base + kNumUnderPromotions + dest_index;
    }
        
    // debug rendering
    void render() const {
        DBG("\n  a b c d e f g h\n");
        for (int y = 7; y >= 0; y--) {
            DBG((y + 1) << " ");
            for (int x = 0; x < 8; x++) {
                const Piece& p = board[y * 8 + x];
                char c = '.';
                if (p.type != EMPTY) {
                    const char* pieces = " KQRBNP";
                    c = pieces[p.type];
                    if (p.color == BLACK) c += 32;  // Lowercase
                }
                DBG(c << " ");
            }
            DBG((y + 1) << "\n");
        }
        DBG("  a b c d e f g h\n");
        DBG((to_move == WHITE ? "White" : "Black") << " to move\n");
    }
    
    // Expose a safe way to clear the cached legal-move vector from outside
    // the class (e.g. after an illegal move that leaves the board unchanged).
    void invalidate_cache() const { cached_legal_moves.reset(); }
    
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
                Square from{ int8_t(sq.x + directions[i][0]*dist), int8_t(sq.y + directions[i][1]*dist) };
                if (!from.is_valid()) break;

                const Piece& p = at(from);
                if (p.type == EMPTY) continue;        // keep going through empties

                if (p.color == by_color) {           // only a *same-colour* piece can attack
                    if (dist == 1 && p.type == KING) return true;
                    if (p.type == QUEEN)             return true;
                    if (is_diagonal && p.type == BISHOP) return true;
                    if (!is_diagonal && p.type == ROOK)  return true;
                }
                break;                               // stop, regardless of colour
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
        // 1. Single forward move
        Square to = {from.x, int8_t(from.y + dir)};
        if (to.is_valid() && at(to).type == EMPTY) {
            if (to.y == promo_rank) {
                // generate all promotions
                for (auto promo : {QUEEN, ROOK, BISHOP, KNIGHT}) {
                    if (!yield(Move{from, to, piece, promo})) return;
                }
            } else {
                if (!yield(Move{from, to, piece, EMPTY})) return;
            }
        }
                
        // 2. Double push from start (separate check vs single forward moves)
        if (from.y == start_rank) {
            Square in_front = {from.x, int8_t(from.y + dir)};
            Square two_in_front = {from.x, int8_t(from.y + 2*dir)};
            if (at(in_front).type == EMPTY && at(two_in_front).type == EMPTY) {
                if (!yield(Move{from, two_in_front, piece, EMPTY})) return;
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
        
        DBG("    Knight at " << char('a' + from.x) << (from.y + 1) << std::endl);
        
        for (auto& m : moves) {
            Square to = {int8_t(from.x + m[0]), int8_t(from.y + m[1])};
            DBG("      Checking move to " << char('a' + to.x) << (to.y + 1));
            DBG(" - valid: " << (to.is_valid() ? "YES" : "NO"));
            
            if (to.is_valid()) {
                const Piece& target = at(to);
                DBG(", target: " << (target.type == EMPTY ? "EMPTY" : "PIECE"));
                DBG(", target color: " << (target.color == WHITE ? "WHITE" : target.color == BLACK ? "BLACK" : "NO_COLOR"));
            }
            
            DBG(std::endl);
            
            if (to.is_valid() && (at(to).type == EMPTY || at(to).color != piece.color)) {
                DBG("      YIELDING MOVE" << std::endl);
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
            if (to.is_valid() && (at(to).type == EMPTY || at(to).color != piece.color)) {
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
                !is_square_attacked({5, int8_t(rank)}, (piece.color == WHITE) ? BLACK : WHITE) &&
                !is_square_attacked({6, int8_t(rank)}, (piece.color == WHITE) ? BLACK : WHITE)) {
                
                Move m{from, {6, int8_t(rank)}, piece, EMPTY};
                m.is_castle_short = true;
                if (!yield(m)) return;
            }
            
            // long castle
            if ((castling_rights & long_mask) &&
                at({1, int8_t(rank)}).type == EMPTY &&
                at({2, int8_t(rank)}).type == EMPTY &&
                at({3, int8_t(rank)}).type == EMPTY &&
                !is_square_attacked({3, int8_t(rank)}, (piece.color == WHITE) ? BLACK : WHITE) &&
                !is_square_attacked({2, int8_t(rank)}, (piece.color == WHITE) ? BLACK : WHITE)) {
                
                Move m{from, {2, int8_t(rank)}, piece, EMPTY};
                m.is_castle_long = true;
                if (!yield(m)) return;
            }
        }
    }
};

// static initialization
ZobristHash ChessBoard::zobrist;

// Forward declaration
Move action_to_move_lookup(int action, const ChessBoard& board);

// This is the reverse of move_to_action
Move action_to_move_direct(int action, const ChessBoard& board) {
    // For now, just use the lookup method which works correctly
    return action_to_move_lookup(action, board);
}

Move action_to_move_lookup(int action, const ChessBoard& board) {
    const auto& moves = board.legal_moves();
    for (const auto& m : moves) {
        if (ChessBoard::move_to_action(m) == action) {
            return m;
        }
    }
    // If no legal move matches the action, return a sentinel invalid move
    return {{-1,-1},{-1,-1},{NO_COLOR, EMPTY}, EMPTY};
}

// passthrough hash for repetition detection
struct PassthroughHash {
    size_t operator()(uint64_t x) const { return x; }
};

} // namespace chess

// Forward declaration for per-environment Stockfish instances
class Stockfish;
namespace chess {
    struct ChessBoard;
}

// Convert UCI algebraic string to a Move object (returns kPassMove if not legal)
inline chess::Move uci_to_move(const std::string &uci, const chess::ChessBoard &board) {
    if (uci.size() < 4) return chess::kPassMove;

    int fx = uci[0] - 'a';
    int fy = uci[1] - '1';
    int tx = uci[2] - 'a';
    int ty = uci[3] - '1';

    chess::PieceType promo = chess::EMPTY;
    if (uci.size() == 5) {
        switch (uci[4]) {
            case 'q': promo = chess::QUEEN;  break;
            case 'r': promo = chess::ROOK;   break;
            case 'b': promo = chess::BISHOP; break;
            case 'n': promo = chess::KNIGHT; break;
        }
    }

    const auto &legal = board.legal_moves();
    for (const auto &m : legal) {
        if (m.from.x == fx && m.from.y == fy && m.to.x == tx && m.to.y == ty &&
            (promo == chess::EMPTY || promo == m.promotion)) {
            return m;
        }
    }
    return chess::kPassMove;
}

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
    float c_reward_check = 0.0f;
    float c_reward_material_diff = 0.0f;

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
    float c_white_moves = 0.0f;
    float c_black_moves = 0.0f;
    float c_valid_moves = 0.0f;
    float c_invalid_moves = 0.0f;
    float c_episode_return_white = 0.0f;
    float c_episode_return_black = 0.0f;
    float c_invalid_moves_white = 0.0f;
    float c_invalid_moves_black = 0.0f;
    float c_perf = 0.0f;
    float c_score = 0.0f;
    bool self_play_mode = false;
    bool waiting_for_black_move = false;
    // Stockfish* sf = nullptr;   // stockfish engine instance (plays black only)
    std::unique_ptr<Stockfish> sf; 
    float stockfish_eval = 0.0f; // last evaluation in centipawns (white perspective)
    bool stockfish_enabled = true; // enabled by default
    int max_depth = 0;  // per-episode step limit

    ChessContext(unsigned seed) : rng(seed) {}
};

// Forward declaration
bool test_action_symmetry(const chess::ChessBoard& board);

extern "C" void set_self_play_mode(CChess* env, bool enabled) {
    auto* ctx = (ChessContext*)env->context;
    ctx->self_play_mode = enabled;
}

extern "C" void c_set_fen(CChess* env, const char* fen) {
    auto* ctx = static_cast<ChessContext*>(env->context);

    ctx->board.set_from_fen(fen);          // load position
    ctx->position_history.clear();         // repetition starts fresh
    ctx->position_history[ctx->board.hash()] = 1;

    ctx->step_count = 0;                   // treat as new episode
    ctx->waiting_for_black_move = false;   // always leave white to move

    compute_observation(env, ctx);         // rebuild obs + mask
    env->terminals[0] = 0;
    env->rewards[0]   = 0.0f;
}

extern "C" void set_debug_disable_mask(CChess* env, bool enabled) {
    env->debug_disable_mask = enabled;
}

extern "C" {

void add_log(CChess* env, const ChessContext* ctx, bool win, bool loss, bool draw) {
    env->log.perf += ctx->c_perf;
    env->log.score += ctx->c_score;
    env->log.episode_length += ctx->step_count;
    env->log.episode_return += ctx->episode_return;
    env->log.episode_return_white += ctx->c_episode_return_white;
    env->log.episode_return_black += ctx->c_episode_return_black;
    env->log.reward_valid += ctx->c_reward_valid;
    env->log.reward_invalid += ctx->c_reward_invalid;
    env->log.reward_agent_captures_enemy_piece += ctx->c_reward_agent_captures_enemy_piece;
    env->log.reward_enemy_captures_agent_piece += ctx->c_reward_enemy_captures_agent_piece;
    env->log.reward_win += ctx->c_reward_win;
    env->log.reward_draw += ctx->c_reward_draw;
    env->log.reward_loss += ctx->c_reward_loss;
    env->log.game_won += win;
    env->log.game_lost += loss;
    env->log.game_drawn += draw;
    env->log.stalemate += ctx->c_stalemate;
    env->log.insufficient_material += ctx->c_insufficient_material;
    env->log.threefold_repetition += ctx->c_threefold_repetition;
    env->log.fifty_move_rule += ctx->c_fifty_move_rule;
    env->log.max_depth += ctx->c_max_depth;
    env->log.white_checkmated += ctx->c_white_checkmated; // black checkmates white
    env->log.black_checkmated += ctx->c_black_checkmated; // white checkmates black
    env->log.white_moves += ctx->c_white_moves;
    env->log.black_moves += ctx->c_black_moves;
    env->log.valid_moves += ctx->c_valid_moves;
    env->log.invalid_moves += ctx->c_invalid_moves;
    env->log.invalid_moves_white += ctx->c_invalid_moves_white;
    env->log.invalid_moves_black += ctx->c_invalid_moves_black;
    env->log.reward_check += ctx->c_reward_check;
    env->log.reward_material_diff += ctx->c_reward_material_diff;
    env->log.stockfish_eval += ctx->stockfish_eval;
    env->log.n += 1;
}

void init(CChess* env) {
    env->context = new ChessContext(12345);
    env->debug_disable_mask = false;
    auto* ctx = (ChessContext*)env->context;
    ctx->stockfish_enabled = env->stockfish_enabled;
    ctx->max_depth = env->max_depth;
    // max_depth is set from chess.ini
    // Use bundled Stockfish binary (resolved in enable_stockfish_black) rather than
    // hard-coded system path to avoid illegal-instruction crashes on CPUs lacking AVX2.
    // Note: ELO will be overridden by training config via vec_enable_stockfish_black
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

    auto* ctx = (ChessContext*)env->context;
    if (ctx) {
        ctx->sf.reset();         // unique_ptr handles destruction
        delete ctx;
    }
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
    ctx->c_reward_check = 0.0f;
    ctx->c_reward_material_diff = 0.0f;

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
    
    // Reset move counters
    ctx->c_white_moves = 0.0f;
    ctx->c_black_moves = 0.0f;
    ctx->c_valid_moves = 0.0f;
    ctx->c_invalid_moves = 0.0f;
    
    // Reset self-play state - always start with white's turn
    ctx->waiting_for_black_move = false;

    compute_observation(env, ctx);

    // CRITICAL: Ensure outputs are initialized
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;

    ctx->c_episode_return_white = 0.0f;
    ctx->c_episode_return_black = 0.0f;

    ctx->stockfish_eval = 0.0f;
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
    
    // Build legal move mask -------------------------------------------------
    // First zero mask
    for (int i = 0; i < 4674; ++i) env->observations[idx + i] = 0.0f;

    if (env->debug_disable_mask) {
        for (int i = 0; i < 4674; ++i) env->observations[idx + i] = 1.0f;
    } else {
        // Efficient build: mark only actually legal moves (O(#legal_moves))
        const auto &legal_moves = ctx->board.legal_moves();
        for (const auto &mv : legal_moves) {
            int action_id = chess::ChessBoard::move_to_action(mv);
            if (action_id >= 0 && action_id < 4674) {
                env->observations[idx + action_id] = 1.0f;
            }
        }
        // Pass move (action 0) is intentionally left disabled for chess.
    }

    // obs is now [board_features_0, ..., board_features_1343, legal_move_mask_0, ..., legal_move_mask_4673]
    // board_features is 1344 floats, legal_move_mask is 4674 floats
    idx += 4674;
    assert(idx == 6018);
}

void c_step(CChess* env) {
    auto* ctx = (ChessContext*)env->context;
        
    // ---------------------------------------------------------------------
    // EARLY TERMINAL CHECK: the side-to-move may already be in a game-ending
    // position (checkmated, stalemate, etc.) before it gets a chance to
    // select an action.  In that case we must mark the episode as terminal
    // *from the perspective of the current player* and apply the
    // appropriate draw/loss rewards.
    // ---------------------------------------------------------------------
    const auto early_legal_moves = ctx->board.legal_moves();
    if (early_legal_moves.empty()) {
        bool early_checkmate  = ctx->board.is_checkmate();
        bool early_stalemate  = ctx->board.is_stalemate();
        bool early_insuffmat  = ctx->board.is_insufficient_material();

        if (early_checkmate || early_stalemate || early_insuffmat) {
            // fprintf(stderr, "[EARLY TERMINAL] early_checkmate=%d early_stalemate=%d early_insuffmat=%d\n", 
                    // early_checkmate, early_stalemate, early_insuffmat);
            float early_reward = 0.0f;
            bool early_draw  = false;
            bool early_loss  = false;
            bool early_win   = false; // never happens here, but keep for symmetry

            if (early_checkmate) {
                // Current side has no legal moves and is in check -> LOSS
                early_loss = true;
                early_reward += env->reward_loss;
                ctx->c_reward_loss += env->reward_loss;

                // Update side-specific counters
                if (ctx->board.side_to_move() == chess::WHITE)
                    ctx->c_white_checkmated += 1;
                else
                    ctx->c_black_checkmated += 1;
            } else {
                early_draw = true;
                early_reward += env->reward_draw;
                ctx->c_reward_draw += env->reward_draw;

                if (early_stalemate)          ctx->c_stalemate              += 1;
                if (early_insuffmat)          ctx->c_insufficient_material  += 1;
            }

            // Update outputs and log, then hard-reset the environment
            env->rewards[0]   = early_reward;
            env->terminals[0] = 1;

            ctx->episode_return += early_reward;
            if (ctx->board.side_to_move() == chess::WHITE) {
                ctx->c_episode_return_white += early_reward;
            } else {
                ctx->c_episode_return_black += early_reward;
            }

            add_log(env, ctx, early_win, early_loss, early_draw);
            c_reset(env);  // envs must reset themselves!!
            return; // Finished this step early
        }
    }

    // Always update observation to reflect current state (fresh after reset if terminal)
    compute_observation(env, ctx);
    
    // Helper lambda to compute total material for a given color
    auto material_value = [&](chess::Color color) {
        int total = 0;
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                const auto& p = ctx->board.at({(int8_t)x, (int8_t)y});
                if (p.color != color) continue;
                switch (p.type) {
                    case chess::PAWN:   total += 1; break;
                    case chess::KNIGHT: total += 3; break;
                    case chess::BISHOP: total += 3; break;
                    case chess::ROOK:   total += 5; break;
                    case chess::QUEEN:  total += 9; break;
                    default: break;
                }
            }
        }
        return total;
    };

    // Compute initial material balance from the active player's perspective
    const bool initial_handling_black = ctx->self_play_mode && ctx->waiting_for_black_move;
    const chess::Color player_color_before = initial_handling_black ? chess::BLACK : chess::WHITE;
    int material_player_before = material_value(player_color_before);
    int material_opp_before    = material_value(player_color_before == chess::WHITE ? chess::BLACK : chess::WHITE);
    int material_diff_before   = material_player_before - material_opp_before;

    float reward = 0.0f;
    bool terminal = false;
    bool win = false;
    bool loss = false;
    bool draw = false;
    
    // Get the action
    int action_idx = env->actions[0];
    
    // Get legal moves first
    const auto& legal_moves = ctx->board.legal_moves();
    
    // Decode the requested move
    chess::Move selected_move = chess::action_to_move_lookup(action_idx, ctx->board);
    
    // Validate move by attempting to apply it on a copy of the board
    bool is_legal = false;
    {
        chess::ChessBoard tmp = ctx->board;
        is_legal = tmp.apply_move(selected_move);
    }

    if (is_legal) {
        // Count this ply
        ctx->step_count += 1;
        reward += env->reward_valid;
        ctx->c_reward_valid += env->reward_valid;
        bool applied_ok = ctx->board.apply_move(selected_move);
        // fprintf(stderr, "[APPLY] ok=%d  after FEN=%s\n", applied_ok, ctx->board.fen().c_str());
        ctx->board.invalidate_cache();   // already done in apply_move

        // Reward for giving check immediately after the move
        if (ctx->board.is_check()) {
            reward += env->reward_check;
            ctx->c_reward_check += env->reward_check;
        }
    } else {
        reward += env->reward_invalid;
        ctx->c_reward_invalid += env->reward_invalid;
        ctx->board.invalidate_cache();   // ensure mask stays in sync

        // Static counter for limited illegal-move logging
        static int dbg_illegal_count = 0;

        if (dbg_illegal_count < 20) {
            ++dbg_illegal_count;
            DBG("ILLEGAL[" << dbg_illegal_count << "] action=" << action_idx << "  mask=" << env->observations[1344 + action_idx] << std::endl);
            DBG("  FEN=" << ctx->board.fen() << std::endl);
            DBG("  Legal moves: " << legal_moves.size() << std::endl);
            if (!legal_moves.empty()) {
                DBG("  First legal action: " << chess::ChessBoard::move_to_action(legal_moves[0]) << std::endl);
                int cnt = std::min<size_t>(5, legal_moves.size());
                DBG("  Some legal ids: ");
                for (int i = 0; i < cnt; ++i) {
                    DBG(chess::ChessBoard::move_to_action(legal_moves[i]) << (i+1==cnt?"\n":" "));
                }
            }
        }
        // OpenSpiel "invalid" mode: leave board unchanged and let the episode continue
    }
    
    // Check for game over after player's move
    if (ctx->board.is_checkmate()) {
        // fprintf(stderr, "[TERMINAL] Checkmate detected\n");
        terminal = true;
        win = true; // Player (whose turn it was) checkmated the opponent
        ctx->c_reward_win += env->reward_win;
        // Track which side was checkmated
        if (player_color_before == chess::WHITE) {
            ctx->c_black_checkmated += 1;
        } else {
            ctx->c_white_checkmated += 1;
        }

        // Apply win reward to current step
        reward += env->reward_win;
    } else if (ctx->board.is_stalemate()) {
        // fprintf(stderr, "[TERMINAL] Stalemate detected\n");
        terminal = true;
        draw = true;
        ctx->c_stalemate += 1;
        ctx->c_reward_draw += env->reward_draw;

        // Apply draw reward
        reward += env->reward_draw;
    } else if (ctx->board.is_insufficient_material()) {
        fprintf(stderr, "[TERMINAL] Insufficient material detected\n");
        terminal = true;
        draw = true;
        ctx->c_insufficient_material += 1;
        ctx->c_reward_draw += env->reward_draw;

        reward += env->reward_draw;
    } else if (ctx->board.get_halfmove_clock() >= 100) { // 50-move rule
        // fprintf(stderr, "[TERMINAL] 50-move rule detected\n");
        terminal = true;
        draw = true;
        ctx->c_fifty_move_rule += 1;
        ctx->c_reward_draw += env->reward_draw;

        reward += env->reward_draw;
    } else if (is_legal) {
        // Only update position history for legal moves that actually changed the board
        uint64_t current_hash = ctx->board.hash();
        ctx->position_history[current_hash]++;
        if (ctx->position_history[current_hash] >= 3) { // Threefold repetition
            // fprintf(stderr, "[TERMINAL] Threefold repetition detected\n");
            terminal = true;
            draw = true;
            ctx->c_threefold_repetition += 1;
            ctx->c_reward_draw += env->reward_draw;

            reward += env->reward_draw;
        }
    }
    
    // Final max-depth check
    if (!terminal && ctx->step_count >= ctx->max_depth) {
        // fprintf(stderr, "[TERMINAL] Max depth reached: %d >= %d\n", ctx->step_count, ctx->max_depth);
        terminal = true;
        draw = true;
        ctx->c_max_depth += 1;
        ctx->c_reward_draw += env->reward_draw;

        reward += env->reward_draw;
    }

    // Material differential reward (computed at the end of both moves)
    const chess::Color player_color_after = player_color_before; // unchanged for the episode perspective
    int material_player_after = material_value(player_color_after);
    int material_opp_after    = material_value(player_color_after == chess::WHITE ? chess::BLACK : chess::WHITE);
    int material_diff_after   = material_player_after - material_opp_after;

    int delta_material = material_diff_after - material_diff_before;
    if (delta_material != 0) {
        // Scaled material differential reward (positive if agent gained material, negative otherwise)
        float scaled_mat_reward = delta_material * env->reward_material_diff;
        reward += scaled_mat_reward;
        ctx->c_reward_material_diff += scaled_mat_reward;

        // Capture-specific rewards (flat, regardless of piece value)
        if (delta_material > 0) {
            reward += env->reward_agent_captures_enemy_piece;
            ctx->c_reward_agent_captures_enemy_piece += env->reward_agent_captures_enemy_piece;
        } else if (delta_material < 0) {
            reward += env->reward_enemy_captures_agent_piece;
            ctx->c_reward_enemy_captures_agent_piece += env->reward_enemy_captures_agent_piece;
        }
    }

    // ---------------- Stockfish automatic black move -----------------
    if (!terminal && ctx->stockfish_enabled && ctx->sf && ctx->board.side_to_move() == chess::BLACK) {
        // Material before Stockfish move (white perspective)
        int mat_before_b_white = material_value(chess::WHITE);
        int mat_before_b_black = material_value(chess::BLACK);
        int diff_before_b = mat_before_b_white - mat_before_b_black;

        // Get Stockfish move and evaluation
        // CRITICAL FIX: Remove hardcoded 10ms, use configured search time
        auto sf_res = ctx->sf->bestmove_with_score(ctx->board.fen());  // Uses configured search_ms
        ctx->stockfish_eval = static_cast<float>(sf_res.second);

        // fprintf(stderr, "[Stockfish] bestmove %s (cp %d)\n", sf_res.first.c_str(), sf_res.second);

        // Helper to convert a Move into UCI
        auto move_to_uci_local = [](const chess::Move &m) -> std::string {
            if (m.from.x < 0) return "0000";
            char buf[6] = {0};
            buf[0] = 'a' + m.from.x;
            buf[1] = '1' + m.from.y;
            buf[2] = 'a' + m.to.x;
            buf[3] = '1' + m.to.y;
            if (m.promotion != chess::EMPTY) {
                switch (m.promotion) {
                    case chess::QUEEN:  buf[4] = 'q'; break;
                    case chess::ROOK:   buf[4] = 'r'; break;
                    case chess::BISHOP: buf[4] = 'b'; break;
                    case chess::KNIGHT: buf[4] = 'n'; break;
                    default: break;
                }
            }
            return std::string(buf);
        };

        // Translate UCI → Move by matching against current legal moves
        const auto &legal = ctx->board.legal_moves();
        chess::Move sf_move = chess::kPassMove;
        for (const auto &mv : legal) {
            if (move_to_uci_local(mv) == sf_res.first) {
                sf_move = mv;
                break;
            }
        }

        // ------------------------------------------------------------------
        // Recovery if the engine produced an illegal move
        // ------------------------------------------------------------------
        if (sf_move == chess::kPassMove) {
            // 1) Restart the engine and retry once
            if (ctx->sf) {
                fprintf(stderr, "[Stockfish] Move %s not found in legal moves, restarting engine\n", sf_res.first.c_str());
                ctx->sf->restart_engine();
                // CRITICAL FIX: Use configured search time, not hardcoded value
                auto sf_res_retry = ctx->sf->bestmove_with_score(ctx->board.fen());
                for (const auto &mv : legal) {
                    if (move_to_uci_local(mv) == sf_res_retry.first) {
                        sf_move = mv;
                        ctx->stockfish_eval = static_cast<float>(sf_res_retry.second);
                        break;
                    }
                }
            }
        }

        if (sf_move == chess::kPassMove && !legal.empty()) {
            // 2) Final fallback – choose a random legal move so play continues
            fprintf(stderr, "[Stockfish] Still no valid move, using random fallback\n");
            std::uniform_int_distribution<int> dist(0, static_cast<int>(legal.size()) - 1);
            sf_move = legal[dist(ctx->rng)];
        }

        // If still invalid after all recovery attempts, abort the episode
        if (sf_move == chess::kPassMove) {
            fprintf(stderr, "[Stockfish] No legal moves available, terminating episode\n");
            env->terminals[0] = 1;
            terminal = true;
            // Mark as draw rather than undefined
            draw = true;
            ctx->c_reward_draw += env->reward_draw;
            reward += env->reward_draw;
        }


        bool applied = (sf_move != chess::kPassMove) ? ctx->board.apply_move(sf_move) : false;
        if (applied) {
            ctx->step_count += 1;
            ctx->c_black_moves += 1;
            ctx->c_valid_moves += 1;
            ctx->board.invalidate_cache();


            // Reward / penalty from material change due to black move
            int mat_after_b_white = material_value(chess::WHITE);
            int mat_after_b_black = material_value(chess::BLACK);
            int diff_after_b = mat_after_b_white - mat_after_b_black;
            int delta_b = diff_after_b - diff_before_b;

            if (delta_b != 0) {
                float scaled_mat_reward_b = delta_b * env->reward_material_diff;
                reward += scaled_mat_reward_b;
                ctx->c_reward_material_diff += scaled_mat_reward_b;

                if (delta_b > 0) {
                    reward += env->reward_agent_captures_enemy_piece;
                    ctx->c_reward_agent_captures_enemy_piece += env->reward_agent_captures_enemy_piece;
                } else if (delta_b < 0) {
                    reward += env->reward_enemy_captures_agent_piece;
                    ctx->c_reward_enemy_captures_agent_piece += env->reward_enemy_captures_agent_piece;
                }
            }

            // Update position history after Stockfish move
            uint64_t sf_hash = ctx->board.hash();
            ctx->position_history[sf_hash]++;

            // Check terminal after Stockfish move (white's perspective)
            if (ctx->board.is_checkmate()) {
                // fprintf(stderr, "[TERMINAL] Checkmate after Stockfish move\n");
                terminal = true;
                loss = true;
                ctx->c_white_checkmated += 1;
                reward += env->reward_loss;
                ctx->c_reward_loss += env->reward_loss;
            } else if (ctx->board.is_stalemate() || ctx->board.is_insufficient_material() || ctx->board.get_halfmove_clock() >= 100) {
                // fprintf(stderr, "[TERMINAL] Draw condition after Stockfish move\n");
                terminal = true;
                draw = true;
                ctx->c_reward_draw += env->reward_draw;
                reward += env->reward_draw;
                if (ctx->board.is_stalemate()) ctx->c_stalemate += 1;
                if (ctx->board.is_insufficient_material()) ctx->c_insufficient_material += 1;
                if (ctx->board.get_halfmove_clock() >= 100) ctx->c_fifty_move_rule += 1;
            } else if (ctx->position_history[sf_hash] >= 3) {
                // Check for threefold repetition after Stockfish move
                // fprintf(stderr, "[TERMINAL] Threefold repetition after Stockfish move\n");
                terminal = true;
                draw = true;
                ctx->c_threefold_repetition += 1;
                ctx->c_reward_draw += env->reward_draw;
                reward += env->reward_draw;
            }
        }

        // After Stockfish move, observation should reflect new position (white to move)
        if (!terminal) {
            compute_observation(env, ctx);
        }
    }

    // Set outputs
    env->rewards[0] = reward;
    env->terminals[0] = terminal ? 1 : 0;
    
    // Add episode return tracking
    ctx->episode_return += reward;
    if (player_color_before == chess::WHITE) {
        ctx->c_episode_return_white += reward;
    } else {
        ctx->c_episode_return_black += reward;
    }
    
    // ------------------------------------------------------------------
    // Toggle waiting_for_black_move so that reward calculations on the
    // next call reflect the correct player perspective when self-play
    // is enabled.  We intentionally flip the flag *after* computing all
    // rewards that depend on the mover (player_color_before) so those
    // calculations use the pre-move value.
    // ------------------------------------------------------------------
    if (ctx->self_play_mode && !terminal) {
        ctx->waiting_for_black_move = !ctx->waiting_for_black_move;
    }

    // call reset() immediately on terminal
    if (terminal) {
        // fprintf(stderr, "[TERMINAL] Game ended: win=%d loss=%d draw=%d\n", win, loss, draw);
        add_log(env, ctx, win, loss, draw);
        c_reset(env);  // envs must reset themselves!!
    }

    // Track move counts for diagnostics
    if (player_color_before == chess::WHITE) {
        ctx->c_white_moves += 1;
    } else {
        ctx->c_black_moves += 1;
    }

    // Track validity statistics
    if (is_legal) {
        ctx->c_valid_moves += 1;
    } else {
        ctx->c_invalid_moves += 1;
        if (player_color_before == chess::WHITE) {
            ctx->c_invalid_moves_white += 1;
        } else {
            ctx->c_invalid_moves_black += 1;
        }
    }

    // just after bool is_legal = ...
    // fprintf(stderr, "[STEP] action %d legal=%d  FEN=%s\n",
            // action_idx, is_legal, ctx->board.fen().c_str());
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
                if (p.color == chess::BLACK) c += 32;
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

// Test action encoding symmetry
bool test_action_symmetry(const chess::ChessBoard& board) {
    const auto& legal_moves = board.legal_moves();
    int failures = 0;
    
    for (const auto& move : legal_moves) {
        int action = chess::ChessBoard::move_to_action(move);
        if (action < 0 || action >= 4674) {
            // printf("Invalid action %d for move\n", action);
            failures++;
            continue;
        }
        
        chess::Move decoded = chess::action_to_move_direct(action, board);
        if (!(decoded == move)) {
            // printf("Round-trip failure: action=%d\n", action);
            failures++;
        }
    }
    
    // printf("Action symmetry test: %d failures out of %d legal moves\n", 
    //        failures, (int)legal_moves.size());
    return failures == 0;
}
#endif // __cplusplus

extern "C" void enable_stockfish_black(CChess* env, const char* stockfish_cmd, int elo, int search_ms) {
    if (!env) return;
    ChessContext* ctx = (ChessContext*)env->context;
    if (!ctx) return;

    // Don't reset if we already have an instance - this was causing memory leaks
    // ctx->sf.reset(); // REMOVED - this was resetting before checking, causing leaks

    // Resolve Stockfish binary path for this environment
    const char* cmd = nullptr;

    if (stockfish_cmd && stockfish_cmd[0]) {
        cmd = stockfish_cmd;
    }

    if (!cmd) {
        const char* candidates[] = {
            "pufferlib/Stockfish/src/stockfish",
            "./pufferlib/Stockfish/src/stockfish",
            "Stockfish/src/stockfish",
            "./Stockfish/src/stockfish",
            "stockfish",
            nullptr
        };
        for (int i = 0; candidates[i]; ++i) {
            if (access(candidates[i], X_OK) == 0) { cmd = candidates[i]; break; }
        }
    }

    if (!cmd) cmd = "stockfish";

    // Create per-environment Stockfish instance
    // ctx->sf = new Stockfish(cmd, elo, search_ms);
    if (!ctx->sf) { // construct only once
        ctx->sf = std::make_unique<Stockfish>(cmd, elo, search_ms);
    }
    ctx->stockfish_enabled = ctx->sf && ctx->sf->ok();
}