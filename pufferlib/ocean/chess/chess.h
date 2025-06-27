// chess.h
#pragma once

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

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
} Log;

typedef struct CChess {
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    
    // static values from config/ocean/chess.ini
    float reward_valid_move;
    float reward_invalid_move;
    float reward_agent_captures_enemy_piece;
    float reward_enemy_captures_agent_piece;
    float reward_win;
    float reward_draw;
    float reward_loss;

    // opaque pointer to C++ context
    void* context;
} CChess;

// pufferlib required functions
void c_init(CChess* env);
void allocate(CChess* env);
void free_allocated(CChess* env);
void c_reset(CChess* env);
void compute_observation(CChess* env, ChessContext* ctx);
void add_log(Log* log, ChessContext* ctx, bool win, bool loss, bool draw);
void c_step(CChess* env);
void c_render(CChess* env);
void c_close(CChess* env);

#ifdef __cplusplus
}
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
    
    uint64_t hash() const {
        int8_t ep_file = (ep_square >= 0) ? (ep_square & 7) : -1;
        return zobrist.hash_position(board, castling_rights, ep_file, to_move);
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
    float c_reward_valid_move = 0.0f;
    float c_reward_invalid_move = 0.0f;
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
    
    ChessContext(unsigned seed) : rng(seed) {}
};

extern "C" {

void add_log(Log* log, const ChessContext* ctx, bool win, bool loss, bool draw) {
    log->n += 1;
    log->episode_length = ctx->step_count;
    log->episode_return = ctx->episode_return;
    log->game_won += win;
    log->game_lost += loss;
    log->game_drawn += draw;
    log->reward_valid += ctx->c_reward_valid_move;
    log->reward_invalid += ctx->c_reward_invalid_move;
    log->reward_agent_captures_enemy_piece += ctx->c_reward_agent_captures_enemy_piece;
    log->reward_enemy_captures_agent_piece += ctx->c_reward_enemy_captures_agent_piece;
    log->reward_win += ctx->c_reward_win;
    log->reward_draw += ctx->c_reward_draw;
    log->reward_loss += ctx->c_reward_loss;
}

void c_init(CChess* env) {
    env->context = new ChessContext(12345);
}

void allocate(CChess* env) {
    // 64 squares for observations
    env->observations = (float*)calloc(64, sizeof(float));
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
    
    // reset board
    ctx->board.reset();
    ctx->position_history.clear();
    ctx->position_history[ctx->board.hash()] = 1;
    
    // reset episode tracking
    ctx->step_count = 0;
    ctx->episode_return = 0.0f;

    // zero counters
    ctx->c_reward_valid_move = 0.0f;
    ctx->c_reward_invalid_move = 0.0f;
    ctx->c_reward_agent_captures_enemy_piece = 0.0f;
    ctx->c_reward_enemy_captures_agent_piece = 0.0f;
    ctx->c_reward_win = 0.0f;
    ctx->c_reward_draw = 0.0f;
    ctx->c_reward_loss = 0.0f;

    ctx->c_game_won = 0.0f;
    ctx->c_game_lost = 0.0f;
    ctx->c_game_drawn = 0.0f;
    ctx->c_n = 0.0f;

    
    // write initial observation
    compute_observation(env, ctx);
    
    *env->terminals = 0;
    *env->rewards = 0.0f;
}

void compute_observation(CChess* env, ChessContext* ctx) {
    for (int sq = 0; sq < 64; sq++) {
        const auto& piece = ctx->board.at({int8_t(sq & 7), int8_t(sq >> 3)});
        env->observations[sq] = piece.to_obs();
    }
}

void c_step(CChess* env) {
    auto* ctx = (ChessContext*)env->context;
    const auto& moves = ctx->board.legal_moves();
    float reward = 0.0f;

    bool terminal = false;
    bool win = false;
    bool loss = false;
    bool draw = false;

    // agent (white) to move:
    // map action to move
    int action_idx = *env->actions;
    chess::Move selected_move;
    bool valid_move = false;
    
    // find move with matching action index
    for (const auto& move : moves) {
        if (chess::ChessBoard::move_to_action(move) == action_idx) {
            selected_move = move;
            valid_move = true;
            break;
        }
    }
    
    if (valid_move) {
        // agent (white) capturing a black piece?
        const auto& captured = ctx->board.at(selected_move.to);
        if (captured.type != chess::EMPTY) {
            reward += env->reward_agent_captures_enemy_piece;
            ctx->c_reward_agent_captures_enemy_piece += env->reward_agent_captures_enemy_piece;
        }
        
        // apply agent (white) move
        ctx->board.apply_move(selected_move);
        reward += env->reward_valid_move;
        ctx->c_reward_valid_move += env->reward_valid_move;
        
        // update position history (for 3-fold repetition detection)
        uint64_t hash = ctx->board.hash();
        ctx->position_history[hash]++;
        
        // opponent (black) to move:
        const auto& opp_moves = ctx->board.legal_moves();
        if (opp_moves.empty()) {
            // black (opponent) has no legal moves, apply either checkmate (white win) or stalemate (draw)
            terminal = true;
            if (ctx->board.is_check()) {
                reward += env->reward_win;
                ctx->c_reward_win += env->reward_win;
                win = true; // checkmate (agent (white) win)
            } else {
                reward += env->reward_draw;
                ctx->c_reward_draw += env->reward_draw;
                draw = true; // stalemate (draw)
            }
        } else {
            // black (opponent) has legal moves, apply random move
            std::uniform_int_distribution<> dist(0, opp_moves.size() - 1);
            const auto& opp_move = opp_moves[dist(ctx->rng)];
            
            // opponent (black) capturing a white piece?
            if (ctx->board.at(opp_move.to).type != chess::EMPTY) {
                reward += env->reward_enemy_captures_agent_piece;
                ctx->c_reward_enemy_captures_agent_piece += env->reward_enemy_captures_agent_piece;
            }            

            // apply opponent (black) move
            ctx->board.apply_move(opp_move);
            ctx->position_history[ctx->board.hash()]++;
        }        
    } else {
        // invalid (white) move
        // penalize agent (white) for invalid move
        reward = env->reward_invalid_move;
        ctx->c_reward_invalid_move += env->reward_invalid_move;
    }

    // opponent (black) to move:
    if (!terminal) {
        const auto& white_moves = ctx->board.legal_moves();
        if (white_moves.empty()) {
            terminal = true;
            if (ctx->board.is_check()) {
                reward += env->reward_loss;
                ctx->c_reward_loss += env->reward_loss;
                loss = true;
            } else {
                reward += env->reward_draw;
                ctx->c_reward_draw += env->reward_draw;
                draw = true;
            }
        }
    }

    bool terminal = false;
    bool win = false;
    bool loss = false;
    bool draw = false;
    
    // is agent (white) checkmated or is there a stalemate?
    if (ctx->board.legal_moves().empty()) {
        terminal = true;
        if (ctx->board.is_check()) {
            reward = env->reward_loss;  // agent (white) checkmated == agent (white) loss
            ctx->c_reward_loss += env->reward_loss;
            loss = true;
        } else {
            reward = env->reward_draw;  // agent (white) stalemated == agent (white) draw
            ctx->c_reward_draw += env->reward_draw;
            draw = true;
        }
    }
    
    // is opponent (black) checkmated?
    if (!terminal && ctx->board.legal_moves().empty()) {
        terminal = true;
        reward = env->reward_win;  // opponent (black) checkmated == agent (white) win
        ctx->c_reward_win += env->reward_win;
        win = true;
    }
    
    // 3-fold repetition draw? (automatically accepted by both players upon detection)
    if (!terminal) {
        for (const auto& [pos, count] : ctx->position_history) {
            if (count >= 3) {
                terminal = true;
                reward = env->reward_draw;
                ctx->c_reward_draw += env->reward_draw;
                draw = true;
                break;
            }
        }
    }
    
    // 50-move rule draw? (automatically applied)
    if (!terminal && ctx->board.get_halfmove_clock() >= 100) {
        terminal = true;
        reward = env->reward_draw;
        ctx->c_reward_draw += env->reward_draw;
        draw = true;
    }
    
    // insufficient material draw? (automatically applied)
    if (!terminal && ctx->board.is_insufficient_material()) {
        terminal = true;
        reward = env->reward_draw;
        ctx->c_reward_draw += env->reward_draw;
        draw = true;
    }
    
    // update observations
    compute_observation(env, ctx);

    // update episode tracking
    ctx->step_count += 1;
    ctx->episode_return += reward;
    
    // on terminal, compute final win/loss/draw rewards and call add_log()
    if (terminal) {
        add_log(&env->log, ctx, win, loss, draw);
        ctx->c_n += 1.0f;
    }
    
    *env->rewards = reward;
    *env->terminals = terminal ? 1 : 0;
}

void c_render(CChess* env) {
    ((ChessContext*)env->context)->board.render();
}

void c_close(CChess* env) {
    // close nothing?
}

} // extern "C"

#endif // __cplusplus