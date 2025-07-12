// chess.h
// Comment out the next line (or pass -DDEBUG_LOG=0 on the compiler command line)
// to turn every DBG() call into a no-op.
#ifndef DEBUG_LOG
#define DEBUG_LOG 0          // 0 = disabled, 1 = enabled
#endif

#if DEBUG_LOG
  #include <iostream>
  #include <fstream>
  
  // Global debug file stream
  static std::ofstream debug_file_stream;
  static bool debug_file_initialized = false;
  
  // Initialize debug file if not already done
  static void init_debug_file() {
      if (!debug_file_initialized) {
          debug_file_stream.open("chess_debug.log", std::ios::app);
          debug_file_initialized = true;
      }
  }
  
  #define DBG(expr) do { \
      init_debug_file(); \
      std::cerr << expr; \
      if (debug_file_stream.is_open()) { \
          debug_file_stream << expr; \
          debug_file_stream.flush(); \
      } \
  } while (0)
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
    float n;
    
    // Game logging - track last move and game state
    float last_move_from;            // source square (0-63)
    float last_move_to;              // destination square (0-63)
    float last_move_promotion;       // promotion piece (0=none, 1=queen, 2=rook, 3=bishop, 4=knight)
    float game_step_logged;          // global step when game should be logged
    float game_moves_count;          // number of moves in current game
    
    // Complete game logging - store all moves as action IDs
    float complete_game_move_count;  // number of moves in complete game
    float complete_game_action_0;    // first move action ID
    float complete_game_action_1;    // second move action ID
    float complete_game_action_2;    // third move action ID
    float complete_game_action_3;    // fourth move action ID
    float complete_game_action_4;    // fifth move action ID
    float complete_game_action_5;    // sixth move action ID
    float complete_game_action_6;    // seventh move action ID
    float complete_game_action_7;    // eighth move action ID
    float complete_game_action_8;    // ninth move action ID
    float complete_game_action_9;    // tenth move action ID
    float complete_game_action_10;   // 11th move action ID
    float complete_game_action_11;   // 12th move action ID
    float complete_game_action_12;   // 13th move action ID
    float complete_game_action_13;   // 14th move action ID
    float complete_game_action_14;   // 15th move action ID
    float complete_game_action_15;   // 16th move action ID
    float complete_game_action_16;   // 17th move action ID
    float complete_game_action_17;   // 18th move action ID
    float complete_game_action_18;   // 19th move action ID
    float complete_game_action_19;   // 20th move action ID
    float complete_game_action_20;   // 21st move action ID
    float complete_game_action_21;   // 22nd move action ID
    float complete_game_action_22;   // 23rd move action ID
    float complete_game_action_23;   // 24th move action ID
    float complete_game_action_24;   // 25th move action ID
    float complete_game_action_25;   // 26th move action ID
    float complete_game_action_26;   // 27th move action ID
    float complete_game_action_27;   // 28th move action ID
    float complete_game_action_28;   // 29th move action ID
    float complete_game_action_29;   // 30th move action ID
    float complete_game_action_30;   // 31st move action ID
    float complete_game_action_31;   // 32nd move action ID
    float complete_game_action_32;   // 33rd move action ID
    float complete_game_action_33;   // 34th move action ID
    float complete_game_action_34;   // 35th move action ID
    float complete_game_action_35;   // 36th move action ID
    float complete_game_action_36;   // 37th move action ID
    float complete_game_action_37;   // 38th move action ID
    float complete_game_action_38;   // 39th move action ID
    float complete_game_action_39;   // 40th move action ID
    float complete_game_action_40;   // 41st move action ID
    float complete_game_action_41;   // 42nd move action ID
    float complete_game_action_42;   // 43rd move action ID
    float complete_game_action_43;   // 44th move action ID
    float complete_game_action_44;   // 45th move action ID
    float complete_game_action_45;   // 46th move action ID
    float complete_game_action_46;   // 47th move action ID
    float complete_game_action_47;   // 48th move action ID
    float complete_game_action_48;   // 49th move action ID
    float complete_game_action_49;   // 50th move action ID
    float complete_game_action_50;   // 51st move action ID
    float complete_game_action_51;   // 52nd move action ID
    float complete_game_action_52;   // 53rd move action ID
    float complete_game_action_53;   // 54th move action ID
    float complete_game_action_54;   // 55th move action ID
    float complete_game_action_55;   // 56th move action ID
    float complete_game_action_56;   // 57th move action ID
    float complete_game_action_57;   // 58th move action ID
    float complete_game_action_58;   // 59th move action ID
    float complete_game_action_59;   // 60th move action ID
    float complete_game_action_60;   // 61st move action ID
    float complete_game_action_61;   // 62nd move action ID
    float complete_game_action_62;   // 63rd move action ID
    float complete_game_action_63;   // 64th move action ID
    float complete_game_action_64;   // 65th move action ID
    float complete_game_action_65;   // 66th move action ID
    float complete_game_action_66;   // 67th move action ID
    float complete_game_action_67;   // 68th move action ID
    float complete_game_action_68;   // 69th move action ID
    float complete_game_action_69;   // 70th move action ID
    float complete_game_action_70;   // 71st move action ID
    float complete_game_action_71;   // 72nd move action ID
    float complete_game_action_72;   // 73rd move action ID
    float complete_game_action_73;   // 74th move action ID
    float complete_game_action_74;   // 75th move action ID
    float complete_game_action_75;   // 76th move action ID
    float complete_game_action_76;   // 77th move action ID
    float complete_game_action_77;   // 78th move action ID
    float complete_game_action_78;   // 79th move action ID
    float complete_game_action_79;   // 80th move action ID
    float complete_game_action_80;   // 81st move action ID
    float complete_game_action_81;   // 82nd move action ID
    float complete_game_action_82;   // 83rd move action ID
    float complete_game_action_83;   // 84th move action ID
    float complete_game_action_84;   // 85th move action ID
    float complete_game_action_85;   // 86th move action ID
    float complete_game_action_86;   // 87th move action ID
    float complete_game_action_87;   // 88th move action ID
    float complete_game_action_88;   // 89th move action ID
    float complete_game_action_89;   // 90th move action ID
    float complete_game_action_90;   // 91st move action ID
    float complete_game_action_91;   // 92nd move action ID
    float complete_game_action_92;   // 93rd move action ID
    float complete_game_action_93;   // 94th move action ID
    float complete_game_action_94;   // 95th move action ID
    float complete_game_action_95;   // 96th move action ID
    float complete_game_action_96;   // 97th move action ID
    float complete_game_action_97;   // 98th move action ID
    float complete_game_action_98;   // 99th move action ID
    float complete_game_action_99;   // 100th move action ID
} Log;

typedef struct CChess {
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    
    // static values from config/ocean/chess.ini
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
void compute_dual_agent_observations(CChess* env, ChessContext* ctx);
void add_log(CChess* env, const ChessContext* ctx, bool win, bool loss, bool draw);
void c_step(CChess* env);
void c_step_dual_agent(CChess* env);
void c_step_single_agent(CChess* env);
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
        // DEBUG: Track direct board resets
        static int board_reset_counter = 0;
        board_reset_counter++;
        
        DBG("[BOARD_RESET_DEBUG] ChessBoard::reset() called (reset #" << board_reset_counter << ")" << std::endl);
        DBG("[BOARD_RESET_DEBUG] Board state BEFORE board reset:" << std::endl);
        DBG("[BOARD_RESET_DEBUG]   Side to move: " << (to_move == WHITE ? "WHITE" : "BLACK") << std::endl);
        DBG("[BOARD_RESET_DEBUG]   FEN: " << to_fen() << std::endl);
        
        // standard starting position
        const char* start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        set_from_fen(start_fen);
        cached_legal_moves.reset();
        
        DBG("[BOARD_RESET_DEBUG] Board state AFTER board reset:" << std::endl);
        DBG("[BOARD_RESET_DEBUG]   Side to move: " << (to_move == WHITE ? "WHITE" : "BLACK") << std::endl);
        DBG("[BOARD_RESET_DEBUG]   FEN: " << to_fen() << std::endl);
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
    
    std::string to_fen() const {
        std::string fen;
        
        // Board position
        for (int rank = 7; rank >= 0; rank--) {
            int empty_count = 0;
            for (int file = 0; file < 8; file++) {
                const Piece& p = board[rank * 8 + file];
                if (p.type == EMPTY) {
                    empty_count++;
                } else {
                    if (empty_count > 0) {
                        fen += std::to_string(empty_count);
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
                    
                    fen += piece_char;
                }
            }
            
            if (empty_count > 0) {
                fen += std::to_string(empty_count);
            }
            
            if (rank > 0) {
                fen += '/';
            }
        }
        
        // Side to move
        fen += (to_move == WHITE) ? " w " : " b ";
        
        // Castling rights
        std::string castling;
        if (castling_rights & 0x8) castling += 'K';  // White kingside
        if (castling_rights & 0x4) castling += 'Q';  // White queenside
        if (castling_rights & 0x2) castling += 'k';  // Black kingside
        if (castling_rights & 0x1) castling += 'q';  // Black queenside
        
        if (castling.empty()) {
            fen += "-";
        } else {
            fen += castling;
        }
        
        // En passant target square
        if (ep_square >= 0) {
            char file_char = 'a' + (ep_square % 8);
            char rank_char = '1' + (ep_square / 8);
            fen += " ";
            fen += file_char;
            fen += rank_char;
        } else {
            fen += " -";
        }
        
        // Halfmove clock and fullmove number
        fen += " " + std::to_string(halfmove_clock);
        fen += " " + std::to_string(fullmove_number);
        
        return fen;
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
                case EMPTY:
                    // Empty squares don't generate moves
                    break;
            }
        }
    }
    
    const std::vector<Move>& legal_moves() const {
        if (!cached_legal_moves) {
            cached_legal_moves = std::vector<Move>();
            
            // PERFORMANCE FIX: Use pseudo-legal moves instead of full legal move validation
            // This eliminates the expensive board copying and check testing on every step
            generate_pseudo_legal_moves([this](const Move& move) {
                // For each pseudo-legal move, verify it doesn't leave the king in check
                // Create a temporary board to test the move
                ChessBoard test_board = *this;
                test_board.apply_move_unchecked(move); // Apply the move without legality check
                test_board.to_move = (to_move == WHITE) ? BLACK : WHITE; // Switch turn for check test

                const Piece& piece_to_move = at(move.from);

                if (!test_board.is_in_check(piece_to_move.color)) {
                    cached_legal_moves->push_back(move);
                }
                return true;  // continue generating
            });
            
            // Sort deterministically by action id for reproducibility
            std::sort(cached_legal_moves->begin(), cached_legal_moves->end(),
                      [](const Move& a, const Move& b){
                          return move_to_action(a) < move_to_action(b);
                      });

            DBG("[LEGAL_MOVES_DEBUG] Total legal moves: " << cached_legal_moves->size() << std::endl);
            DBG("[LEGAL_MOVES_DEBUG] Final original board state - side: " << (to_move == WHITE ? "WHITE" : "BLACK") << ", hash: " << hash() << std::endl);
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
                    
                case EMPTY:
                    // Empty squares don't affect insufficient material calculation
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
        
        // MISSING CASE: K+B+B vs K (two bishops of same color vs lone king)
        // This is insufficient material for checkmate
        if (white_bishops >= 2 && white_knights == 0 && 
            black_bishops == 0 && black_knights == 0) {
            // Check if all white bishops are on the same color squares
            bool all_same_color = (!white_light_bishop && white_dark_bishop) || 
                                 (white_light_bishop && !white_dark_bishop);
            if (all_same_color) {
                return true;
            }
        }
        
        if (black_bishops >= 2 && black_knights == 0 && 
            white_bishops == 0 && white_knights == 0) {
            // Check if all black bishops are on the same color squares  
            bool all_same_color = (!black_light_bishop && black_dark_bishop) || 
                                 (black_light_bishop && !black_dark_bishop);
            if (all_same_color) {
                return true;
            }
        }
        
        // MISSING CASE: K+N+N vs K (two knights vs lone king)
        // This is generally insufficient for checkmate
        if (white_knights >= 2 && white_bishops == 0 && 
            black_bishops == 0 && black_knights == 0) {
            return true;
        }
        
        if (black_knights >= 2 && black_bishops == 0 && 
            white_bishops == 0 && white_knights == 0) {
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
        // DEBUG: Track move application
        static int move_application_counter = 0;
        move_application_counter++;
        
        DBG("[BOARD_APPLY_DEBUG] ChessBoard::apply_move_unchecked() called (move #" << move_application_counter << ")" << std::endl);
        DBG("[BOARD_APPLY_DEBUG] BEFORE move application:" << std::endl);
        DBG("[BOARD_APPLY_DEBUG]   Side to move: " << (to_move == WHITE ? "WHITE" : "BLACK") << std::endl);
        DBG("[BOARD_APPLY_DEBUG]   Fullmove: " << fullmove_number << std::endl);
        DBG("[BOARD_APPLY_DEBUG]   Move: from " << char('a' + move.from.x) << (move.from.y + 1));
        DBG(" to " << char('a' + move.to.x) << (move.to.y + 1));
        if (move.promotion != EMPTY) {
            DBG(" promotion " << (int)move.promotion);
        }
        DBG(std::endl);
        
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
        chess::Color old_side = to_move;
        to_move = (to_move == WHITE) ? BLACK : WHITE;
        if (to_move == WHITE) fullmove_number++;
        
        DBG("[BOARD_APPLY_DEBUG] AFTER move application:" << std::endl);
        DBG("[BOARD_APPLY_DEBUG]   Side switched from " << (old_side == WHITE ? "WHITE" : "BLACK"));
        DBG(" to " << (to_move == WHITE ? "WHITE" : "BLACK") << std::endl);
        DBG("[BOARD_APPLY_DEBUG]   Fullmove: " << fullmove_number << std::endl);
        DBG("[BOARD_APPLY_DEBUG]   New FEN: " << to_fen() << std::endl);
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
            if (two_in_front.is_valid() && at(in_front).type == EMPTY && at(two_in_front).type == EMPTY) {
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
            
            // CRITICAL FIX: Add explicit boundary check to prevent off-board moves
            if (to.x < 0 || to.x >= 8 || to.y < 0 || to.y >= 8) {
                DBG(" - SKIPPING: off-board move" << std::endl);
                continue;
            }
            
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
                
                // CRITICAL FIX: Add explicit boundary check to prevent off-board moves
                if (to.x < 0 || to.x >= 8 || to.y < 0 || to.y >= 8) {
                    break;
                }
                
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
            
            // CRITICAL FIX: Add explicit boundary check to prevent off-board moves
            if (to.x < 0 || to.x >= 8 || to.y < 0 || to.y >= 8) {
                continue;
            }
            
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
    // Special cases first
    if (action == 0) return kPassMove;
    if (action == 4672) {
        // Queenside castling
        Square king_pos = board.find_king(board.side_to_move());
        if (king_pos.is_valid()) {
            int rank = (board.side_to_move() == WHITE) ? 0 : 7;
            Move m{king_pos, {2, int8_t(rank)}, {board.side_to_move(), KING}, EMPTY};
            m.is_castle_long = true;
            return m;
        }
        return {{-1,-1},{-1,-1},{NO_COLOR, EMPTY}, EMPTY};
    }
    if (action == 4673) {
        // Kingside castling
        Square king_pos = board.find_king(board.side_to_move());
        if (king_pos.is_valid()) {
            int rank = (board.side_to_move() == WHITE) ? 0 : 7;
            Move m{king_pos, {6, int8_t(rank)}, {board.side_to_move(), KING}, EMPTY};
            m.is_castle_short = true;
            return m;
        }
        return {{-1,-1},{-1,-1},{NO_COLOR, EMPTY}, EMPTY};
    }
    
    // Regular moves: decode from action ID
    if (action < 1 || action > 4671) {
        return {{-1,-1},{-1,-1},{NO_COLOR, EMPTY}, EMPTY};
    }
    
    // Decode from square and destination index
    // Action encoding: action = from_base + dest_index, where from_base = from_square * 73
    // So: from_square = action / 73, dest_index = action % 73
    int from_square = action / kNumActionDestinations;
    int dest_index = action % kNumActionDestinations;
    
    // Convert from_square to coordinates
    // FIXED: Match the encoding which uses (x * 8 + y), so x = from_square / 8, y = from_square % 8
    int from_x = from_square / 8;
    int from_y = from_square % 8;
    
    // Validate from square
    if (from_x < 0 || from_x >= 8 || from_y < 0 || from_y >= 8) {
        return {{-1,-1},{-1,-1},{NO_COLOR, EMPTY}, EMPTY};
    }
    
    Square from{int8_t(from_x), int8_t(from_y)};
    
    // Get the piece at the from square
    const Piece& piece = board.at(from);
    if (piece.type == EMPTY || piece.color != board.side_to_move()) {
        return {{-1,-1},{-1,-1},{NO_COLOR, EMPTY}, EMPTY};
    }
    
    // Create rotated coordinates for move calculation (always from white's perspective)
    int calc_from_y = from_y;
    if (board.side_to_move() == BLACK) {
        calc_from_y = 7 - from_y;
    }
    
    // Handle under-promotions (first 9 destination indices)
    if (dest_index < kNumUnderPromotions) {
        int promo_piece_idx = dest_index / 3;
        int direction = dest_index % 3;
        
        static constexpr PieceType kUnder[3] = {KNIGHT, BISHOP, ROOK};
        static constexpr int kDirs[3] = {-1, 0, 1}; // left capture, straight, right capture
        
        if (promo_piece_idx >= 3) {
            return {{-1,-1},{-1,-1},{NO_COLOR, EMPTY}, EMPTY};
        }
        
        int to_x = from_x + kDirs[direction];
        int to_y = calc_from_y + ((board.side_to_move() == WHITE) ? 1 : -1);
        
        // Convert back to actual board coordinates
        if (board.side_to_move() == BLACK) {
            to_y = 7 - to_y;
        }
        
        // Validate destination
        if (to_x < 0 || to_x >= 8 || to_y < 0 || to_y >= 8) {
            return {{-1,-1},{-1,-1},{NO_COLOR, EMPTY}, EMPTY};
        }
        
        Square to{int8_t(to_x), int8_t(to_y)};
        return {from, to, piece, kUnder[promo_piece_idx]};
    }
    
    // Handle regular moves (queen-style and knight)
    int queen_dest_index = dest_index - kNumUnderPromotions;
    
    int dx = 0, dy = 0;
    
    if (queen_dest_index >= 0 && queen_dest_index < 7) {
        // North (0-6)
        dx = 0;
        dy = queen_dest_index + 1;
    } else if (queen_dest_index >= 7 && queen_dest_index < 14) {
        // Northeast (7-13)
        int dist = queen_dest_index - 7 + 1;
        dx = dist;
        dy = dist;
    } else if (queen_dest_index >= 14 && queen_dest_index < 21) {
        // East (14-20)
        dx = queen_dest_index - 14 + 1;
        dy = 0;
    } else if (queen_dest_index >= 21 && queen_dest_index < 28) {
        // Southeast (21-27)
        int dist = queen_dest_index - 21 + 1;
        dx = dist;
        dy = -dist;
    } else if (queen_dest_index >= 28 && queen_dest_index < 35) {
        // South (28-34)
        dx = 0;
        dy = -(queen_dest_index - 28 + 1);
    } else if (queen_dest_index >= 35 && queen_dest_index < 42) {
        // Southwest (35-41)
        int dist = queen_dest_index - 35 + 1;
        dx = -dist;
        dy = -dist;
    } else if (queen_dest_index >= 42 && queen_dest_index < 49) {
        // West (42-48)
        dx = -(queen_dest_index - 42 + 1);
        dy = 0;
    } else if (queen_dest_index >= 49 && queen_dest_index < 56) {
        // Northwest (49-55)
        int dist = queen_dest_index - 49 + 1;
        dx = -dist;
        dy = dist;
    } else if (queen_dest_index >= 56 && queen_dest_index < 64) {
        // Knight moves (56-63)
        static constexpr int kKnight[8][2] = {
            {-2,-1}, {-2, 1}, {-1,-2}, {-1, 2},
            { 2,-1}, { 2, 1}, { 1,-2}, { 1, 2}
        };
        int knight_idx = queen_dest_index - 56;
        if (knight_idx >= 0 && knight_idx < 8) {
            dx = kKnight[knight_idx][0];
            dy = kKnight[knight_idx][1];
        } else {
            return {{-1,-1},{-1,-1},{NO_COLOR, EMPTY}, EMPTY};
        }
    } else {
        return {{-1,-1},{-1,-1},{NO_COLOR, EMPTY}, EMPTY};
    }
    
    // Calculate destination square
    int to_x = from_x + dx;
    int to_y = calc_from_y + dy;
    
    // Convert back to actual board coordinates
    if (board.side_to_move() == BLACK) {
        to_y = 7 - to_y;
    }
    
    // Validate destination
    if (to_x < 0 || to_x >= 8 || to_y < 0 || to_y >= 8) {
        return {{-1,-1},{-1,-1},{NO_COLOR, EMPTY}, EMPTY};
    }
    
    Square to{int8_t(to_x), int8_t(to_y)};
    
    // Check for queen promotion (pawn moving to promotion rank)
    PieceType promotion = EMPTY;
    if (piece.type == PAWN) {
        int promotion_rank = (board.side_to_move() == WHITE) ? 7 : 0;
        if (to_y == promotion_rank) {
            promotion = QUEEN; // Default to queen promotion for non-under-promotion moves
        }
    }
    
    return {from, to, piece, promotion};
}

Move action_to_move_lookup(int action, const ChessBoard& board) {
    DBG("[ACTION_DECODE_DEBUG] Converting action " << action << " to move" << std::endl);
    
    const auto& moves = board.legal_moves();
    DBG("[ACTION_DECODE_DEBUG] Board has " << moves.size() << " legal moves" << std::endl);
    
    for (const auto& m : moves) {
        int move_action = ChessBoard::move_to_action(m);
        if (move_action == action) {
            DBG("[ACTION_DECODE_DEBUG] Found matching legal move: ");
            DBG("from " << char('a' + m.from.x) << (m.from.y + 1));
            DBG(" to " << char('a' + m.to.x) << (m.to.y + 1));
            if (m.promotion != EMPTY) {
                DBG(" promotion " << (int)m.promotion);
            }
            DBG(" (action " << move_action << ")" << std::endl);
            return m;
        }
    }
    
    // If no legal move matches the action, return a sentinel invalid move
    DBG("[ACTION_DECODE_DEBUG] *** NO LEGAL MOVE FOUND FOR ACTION " << action << " ***" << std::endl);
    DBG("[ACTION_DECODE_DEBUG] Available legal actions:" << std::endl);
    for (size_t i = 0; i < std::min((size_t)10, moves.size()); i++) {
        int legal_action = ChessBoard::move_to_action(moves[i]);
        DBG("[ACTION_DECODE_DEBUG]   Action " << legal_action << ": ");
        DBG("from " << char('a' + moves[i].from.x) << (moves[i].from.y + 1));
        DBG(" to " << char('a' + moves[i].to.x) << (moves[i].to.y + 1));
        if (moves[i].promotion != EMPTY) {
            DBG(" promotion " << (int)moves[i].promotion);
        }
        DBG(std::endl);
    }
    
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
    
    // PERFORMANCE OPTIMIZATION: Cache legal moves to avoid recomputing multiple times per step
    mutable std::optional<std::vector<chess::Move>> cached_step_legal_moves;
    mutable uint64_t cached_step_board_hash = 0;
    
    // Helper function to get legal moves with caching
    const std::vector<chess::Move>& get_legal_moves_cached() const {
        uint64_t current_hash = board.hash();
        if (!cached_step_legal_moves || cached_step_board_hash != current_hash) {
            cached_step_legal_moves = board.legal_moves();
            cached_step_board_hash = current_hash;
        }
        return *cached_step_legal_moves;
    }
    
    // Clear the cache when board changes
    void invalidate_legal_moves_cache() {
        cached_step_legal_moves.reset();
        cached_step_board_hash = 0;
    }
    
    // config vars from python
    float c_reward_valid = 0.0f;
    float c_reward_invalid = 0.0f;
    float c_reward_agent_captures_enemy_piece = 0.0f;
    float c_reward_enemy_captures_agent_piece = 0.0f;
    float c_reward_draw = 0.0f;
    float c_reward_win = 0.0f;
    float c_reward_loss = 0.0f;
    float c_reward_check = 0.0f;
    float c_reward_check_white = 0.0f;
    float c_reward_check_black = 0.0f;
    float c_reward_material_diff = 0.0f;
    float c_reward_material_diff_white = 0.0f;
    float c_reward_material_diff_black = 0.0f;
    // Perspective-based reward tracking during episodes
    float c_reward_win_white = 0.0f;
    float c_reward_win_black = 0.0f;
    float c_reward_loss_white = 0.0f;
    float c_reward_loss_black = 0.0f;
    float c_reward_draw_white = 0.0f;
    float c_reward_draw_black = 0.0f;

    // env logging vars
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
    // En passant tracking
    float c_en_passant_white = 0.0f;
    float c_en_passant_black = 0.0f;
    // Castling tracking
    float c_white_castle_kingside = 0.0f;
    float c_white_castle_queenside = 0.0f;
    float c_black_castle_kingside = 0.0f;
    float c_black_castle_queenside = 0.0f;
    // Promotion tracking
    float c_white_promotion_count = 0.0f;
    float c_white_promotion_knight = 0.0f;
    float c_white_promotion_bishop = 0.0f;
    float c_white_promotion_rook = 0.0f;
    float c_white_promotion_queen = 0.0f;
    float c_black_promotion_count = 0.0f;
    float c_black_promotion_knight = 0.0f;
    float c_black_promotion_bishop = 0.0f;
    float c_black_promotion_rook = 0.0f;
    float c_black_promotion_queen = 0.0f;
    bool self_play_mode = false;
    bool dual_agent_self_play_mode = false;  // True dual agent self-play where both agents act simultaneously
    bool waiting_for_black_move = false;
    // Stockfish* sf = nullptr;   // stockfish engine instance (plays black only)
    std::unique_ptr<Stockfish> sf; 
    float stockfish_eval = 0.0f; // last evaluation in centipawns (white perspective)
    bool stockfish_enabled = true; // enabled by default
    int max_depth = 0;  // per-episode step limit
    
    // NEW: Consecutive check tracking for anti-collapse measures
    int consecutive_checks_white = 0;  // consecutive checks by white
    int consecutive_checks_black = 0;  // consecutive checks by black
    int checks_given_white = 0;        // total checks given by white this game
    int checks_given_black = 0;        // total checks given by black this game
    int moves_since_progress = 0;      // moves since last capture or pawn move
    bool last_move_was_check = false;  // whether the last move was a check
    chess::Color last_checking_color = chess::WHITE; // who gave the last check
    
    // Anti-draw measures
    float repetition_penalty_multiplier = 1.0f;  // increases with repetitions
    bool position_repeated_recently = false;     // track if position was repeated recently
    
    // Complete game tracking
    std::vector<int> complete_game_actions;  // store all action IDs for current game
    
    // GUI mode tracking to prevent duplicate step counting
    uint64_t last_processed_hash = 0;  // hash of last processed board position

    ChessContext(unsigned seed) : rng(seed) {}
};

// Forward declaration
bool test_action_symmetry(const chess::ChessBoard& board);

extern "C" void set_self_play_mode(CChess* env, bool enabled) {
    auto* ctx = (ChessContext*)env->context;
    ctx->self_play_mode = enabled;
}

extern "C" void set_dual_agent_self_play_mode(CChess* env, bool enabled) {
    auto* ctx = (ChessContext*)env->context;
    ctx->dual_agent_self_play_mode = enabled;
    // When dual agent mode is enabled, disable regular self-play mode
    if (enabled) {
        ctx->self_play_mode = false;
    }
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

// Game outcome structure for capturing results before reset
struct GameOutcome {
    bool game_ended = false;
    bool white_won = false;
    bool black_won = false;
    bool is_draw = false;
    std::string draw_reason = "";
};

// Global variable definition
GameOutcome last_game_outcome;

extern "C" {

void add_log(CChess* env, const ChessContext* ctx, bool win, bool loss, bool draw) {
    // Capture game outcome before counters get reset
    last_game_outcome.game_ended = true;
    
    // FIXED: Check max depth first - max depth games should always be draws
    if (ctx->c_max_depth > 0) {
        last_game_outcome.is_draw = true;
        last_game_outcome.draw_reason = "max depth";
    } else if (ctx->c_black_checkmated > 0) {
        last_game_outcome.white_won = true;
        last_game_outcome.draw_reason = "";
    } else if (ctx->c_white_checkmated > 0) {
        last_game_outcome.black_won = true;
        last_game_outcome.draw_reason = "";
    } else {
        last_game_outcome.is_draw = true;
        if (ctx->c_stalemate > 0) last_game_outcome.draw_reason = "stalemate";
        else if (ctx->c_insufficient_material > 0) last_game_outcome.draw_reason = "insufficient material";
        else if (ctx->c_threefold_repetition > 0) last_game_outcome.draw_reason = "threefold repetition";
        else if (ctx->c_fifty_move_rule > 0) last_game_outcome.draw_reason = "fifty move rule";
        else last_game_outcome.draw_reason = "unknown";
    }
    
    // Store complete game actions in log
    int actual_move_count = static_cast<int>(ctx->complete_game_actions.size());
    env->log.complete_game_move_count = static_cast<float>(std::min(actual_move_count, 100));
    
    // Debug output
    DBG("[add_log] Complete game has " << actual_move_count << " moves, storing " << env->log.complete_game_move_count << std::endl);
    if (actual_move_count > 0) {
        DBG("[add_log] First few actions: ");
        for (int i = 0; i < std::min(5, actual_move_count); i++) {
            DBG(ctx->complete_game_actions[i] << " ");
        }
        DBG(std::endl);
    }
    
    // Initialize all action fields to -1 (invalid)
    float* action_fields[100] = {
        &env->log.complete_game_action_0, &env->log.complete_game_action_1, &env->log.complete_game_action_2, 
        &env->log.complete_game_action_3, &env->log.complete_game_action_4, &env->log.complete_game_action_5,
        &env->log.complete_game_action_6, &env->log.complete_game_action_7, &env->log.complete_game_action_8,
        &env->log.complete_game_action_9, &env->log.complete_game_action_10, &env->log.complete_game_action_11,
        &env->log.complete_game_action_12, &env->log.complete_game_action_13, &env->log.complete_game_action_14,
        &env->log.complete_game_action_15, &env->log.complete_game_action_16, &env->log.complete_game_action_17,
        &env->log.complete_game_action_18, &env->log.complete_game_action_19, &env->log.complete_game_action_20,
        &env->log.complete_game_action_21, &env->log.complete_game_action_22, &env->log.complete_game_action_23,
        &env->log.complete_game_action_24, &env->log.complete_game_action_25, &env->log.complete_game_action_26,
        &env->log.complete_game_action_27, &env->log.complete_game_action_28, &env->log.complete_game_action_29,
        &env->log.complete_game_action_30, &env->log.complete_game_action_31, &env->log.complete_game_action_32,
        &env->log.complete_game_action_33, &env->log.complete_game_action_34, &env->log.complete_game_action_35,
        &env->log.complete_game_action_36, &env->log.complete_game_action_37, &env->log.complete_game_action_38,
        &env->log.complete_game_action_39, &env->log.complete_game_action_40, &env->log.complete_game_action_41,
        &env->log.complete_game_action_42, &env->log.complete_game_action_43, &env->log.complete_game_action_44,
        &env->log.complete_game_action_45, &env->log.complete_game_action_46, &env->log.complete_game_action_47,
        &env->log.complete_game_action_48, &env->log.complete_game_action_49, &env->log.complete_game_action_50,
        &env->log.complete_game_action_51, &env->log.complete_game_action_52, &env->log.complete_game_action_53,
        &env->log.complete_game_action_54, &env->log.complete_game_action_55, &env->log.complete_game_action_56,
        &env->log.complete_game_action_57, &env->log.complete_game_action_58, &env->log.complete_game_action_59,
        &env->log.complete_game_action_60, &env->log.complete_game_action_61, &env->log.complete_game_action_62,
        &env->log.complete_game_action_63, &env->log.complete_game_action_64, &env->log.complete_game_action_65,
        &env->log.complete_game_action_66, &env->log.complete_game_action_67, &env->log.complete_game_action_68,
        &env->log.complete_game_action_69, &env->log.complete_game_action_70, &env->log.complete_game_action_71,
        &env->log.complete_game_action_72, &env->log.complete_game_action_73, &env->log.complete_game_action_74,
        &env->log.complete_game_action_75, &env->log.complete_game_action_76, &env->log.complete_game_action_77,
        &env->log.complete_game_action_78, &env->log.complete_game_action_79, &env->log.complete_game_action_80,
        &env->log.complete_game_action_81, &env->log.complete_game_action_82, &env->log.complete_game_action_83,
        &env->log.complete_game_action_84, &env->log.complete_game_action_85, &env->log.complete_game_action_86,
        &env->log.complete_game_action_87, &env->log.complete_game_action_88, &env->log.complete_game_action_89,
        &env->log.complete_game_action_90, &env->log.complete_game_action_91, &env->log.complete_game_action_92,
        &env->log.complete_game_action_93, &env->log.complete_game_action_94, &env->log.complete_game_action_95,
        &env->log.complete_game_action_96, &env->log.complete_game_action_97, &env->log.complete_game_action_98,
        &env->log.complete_game_action_99
    };
    
    // Set all actions to -1 initially
    for (int i = 0; i < 100; i++) {
        *(action_fields[i]) = -1.0f;
    }
    
    // Copy actual actions
    for (int i = 0; i < std::min(actual_move_count, 100); i++) {
        *(action_fields[i]) = static_cast<float>(ctx->complete_game_actions[i]);
        DBG("[add_log] Stored action " << i << ": " << ctx->complete_game_actions[i] << std::endl);
    }
    
    env->log.episode_length += ctx->step_count;
    env->log.episode_return += ctx->episode_return;
    env->log.episode_return_white += ctx->c_episode_return_white;
    env->log.episode_return_black += ctx->c_episode_return_black;
    env->log.reward_valid += ctx->c_reward_valid;
    env->log.reward_agent_captures_enemy_piece += ctx->c_reward_agent_captures_enemy_piece;
    env->log.reward_enemy_captures_agent_piece += ctx->c_reward_enemy_captures_agent_piece;
    env->log.reward_draw += ctx->c_reward_draw;
    
    // Perspective-based reward tracking
    env->log.reward_win_white += ctx->c_reward_win_white;
    env->log.reward_win_black += ctx->c_reward_win_black;
    env->log.reward_loss_white += ctx->c_reward_loss_white;
    env->log.reward_loss_black += ctx->c_reward_loss_black;
    env->log.reward_draw_white += ctx->c_reward_draw_white;
    env->log.reward_draw_black += ctx->c_reward_draw_black;
    
    // FIXED: Corrected win/loss/draw tracking - max depth games are always draws
    if (ctx->c_max_depth > 0) {
        // Max depth reached - always a draw regardless of rewards
        env->log.game_drawn += 1;
    } else if (ctx->c_black_checkmated > 0) {
        // White checkmated black - white wins
        env->log.white_win += 1;
        env->log.black_loss += 1;
    } else if (ctx->c_white_checkmated > 0) {
        // Black checkmated white - black wins
        env->log.black_win += 1;
        env->log.white_loss += 1;
    } else {
        // No checkmate and no max depth - this is a draw from other causes
        env->log.game_drawn += 1;
    }
    
    // Calculate performance metrics (white win rate)
    float total_games = env->log.white_win + env->log.white_loss + env->log.game_drawn;
    if (total_games > 0) {
        env->log.perf = env->log.white_win / total_games;
    }
    
    // Calculate score (white wins minus white losses)
    env->log.score = env->log.white_win - env->log.white_loss;
    
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
    env->log.invalid_moves_white += ctx->c_invalid_moves_white;
    env->log.invalid_moves_black += ctx->c_invalid_moves_black;
    env->log.reward_check_white += ctx->c_reward_check_white;
    env->log.reward_check_black += ctx->c_reward_check_black;
    env->log.reward_material_diff_white += ctx->c_reward_material_diff_white;
    env->log.reward_material_diff_black += ctx->c_reward_material_diff_black;
    env->log.stockfish_eval += ctx->stockfish_eval;
    env->log.en_passant_white += ctx->c_en_passant_white;
    env->log.en_passant_black += ctx->c_en_passant_black;
    env->log.white_castle_kingside += ctx->c_white_castle_kingside;
    env->log.white_castle_queenside += ctx->c_white_castle_queenside;
    env->log.black_castle_kingside += ctx->c_black_castle_kingside;
    env->log.black_castle_queenside += ctx->c_black_castle_queenside;
    env->log.white_promotion_count += ctx->c_white_promotion_count;
    env->log.white_promotion_knight += ctx->c_white_promotion_knight;
    env->log.white_promotion_bishop += ctx->c_white_promotion_bishop;
    env->log.white_promotion_rook += ctx->c_white_promotion_rook;
    env->log.white_promotion_queen += ctx->c_white_promotion_queen;
    env->log.black_promotion_count += ctx->c_black_promotion_count;
    env->log.black_promotion_knight += ctx->c_black_promotion_knight;
    env->log.black_promotion_bishop += ctx->c_black_promotion_bishop;
    env->log.black_promotion_rook += ctx->c_black_promotion_rook;
    env->log.black_promotion_queen += ctx->c_black_promotion_queen;
    
    env->log.n += 1;
}

void init(CChess* env) {
    env->context = new ChessContext(12345);
    // REMOVED: env->debug_disable_mask = false;  // Don't override the value set in binding.cpp
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
    
    // DEBUG: Track reset calls with stack trace
    static int reset_call_counter = 0;
    reset_call_counter++;
    
    DBG("[RESET_DEBUG] ===== c_reset called (reset #" << reset_call_counter << ") =====" << std::endl);
    DBG("[RESET_DEBUG] Board state BEFORE reset:" << std::endl);
    DBG("[RESET_DEBUG]   Side to move: " << (ctx->board.side_to_move() == chess::WHITE ? "WHITE" : "BLACK") << std::endl);
    DBG("[RESET_DEBUG]   Board hash: " << ctx->board.hash() << std::endl);
    DBG("[RESET_DEBUG]   FEN: " << ctx->board.to_fen() << std::endl);
    
    // Reset board
    ctx->board.reset();
    ctx->position_history.clear();
    ctx->position_history[ctx->board.hash()] = 1;
    
    DBG("[RESET_DEBUG] Board state AFTER reset:" << std::endl);
    DBG("[RESET_DEBUG]   Side to move: " << (ctx->board.side_to_move() == chess::WHITE ? "WHITE" : "BLACK") << std::endl);
    DBG("[RESET_DEBUG]   Board hash: " << ctx->board.hash() << std::endl);
    DBG("[RESET_DEBUG]   FEN: " << ctx->board.to_fen() << std::endl);
    DBG("[RESET_DEBUG] ===== c_reset complete =====" << std::endl);
    
    // Reset episode tracking
    ctx->step_count = 0;
    ctx->episode_return = 0.0f;
    
    // Reset game logging
    env->log.game_moves_count = 0;

    // zero counters
    ctx->c_reward_valid = 0.0f;
    ctx->c_reward_agent_captures_enemy_piece = 0.0f;
    ctx->c_reward_enemy_captures_agent_piece = 0.0f;
    ctx->c_reward_draw = 0.0f;
    ctx->c_reward_check_white = 0.0f;
    ctx->c_reward_check_black = 0.0f;
    ctx->c_reward_material_diff_white = 0.0f;
    ctx->c_reward_material_diff_black = 0.0f;
    
    // Reset perspective-based reward tracking
    ctx->c_reward_win_white = 0.0f;
    ctx->c_reward_win_black = 0.0f;
    ctx->c_reward_loss_white = 0.0f;
    ctx->c_reward_loss_black = 0.0f;
    ctx->c_reward_draw_white = 0.0f;
    ctx->c_reward_draw_black = 0.0f;

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
    ctx->c_invalid_moves_white = 0.0f;
    ctx->c_invalid_moves_black = 0.0f;

    
    // Reset self-play state - always start with white's turn
    ctx->waiting_for_black_move = false;

    // Reset episode return tracking
    ctx->c_episode_return_white = 0.0f;
    ctx->c_episode_return_black = 0.0f;

    // Reset new tracking counters BEFORE compute_observation
    ctx->c_en_passant_white = 0.0f;
    ctx->c_en_passant_black = 0.0f;
    ctx->c_white_castle_kingside = 0.0f;
    ctx->c_white_castle_queenside = 0.0f;
    ctx->c_black_castle_kingside = 0.0f;
    ctx->c_black_castle_queenside = 0.0f;
    ctx->c_white_promotion_count = 0.0f;
    ctx->c_white_promotion_knight = 0.0f;
    ctx->c_white_promotion_bishop = 0.0f;
    ctx->c_white_promotion_rook = 0.0f;
    ctx->c_white_promotion_queen = 0.0f;
    ctx->c_black_promotion_count = 0.0f;
    ctx->c_black_promotion_knight = 0.0f;
    ctx->c_black_promotion_bishop = 0.0f;
    ctx->c_black_promotion_rook = 0.0f;
    ctx->c_black_promotion_queen = 0.0f;

    ctx->stockfish_eval = 0.0f;
    
    // Reset NEW tracking variables
    ctx->consecutive_checks_white = 0;
    ctx->consecutive_checks_black = 0;
    ctx->checks_given_white = 0;
    ctx->checks_given_black = 0;
    ctx->moves_since_progress = 0;
    ctx->last_move_was_check = false;
    ctx->last_checking_color = chess::WHITE;
    ctx->repetition_penalty_multiplier = 1.0f;
    ctx->position_repeated_recently = false;
    
    // Reset complete game tracking
    ctx->complete_game_actions.clear();
    
    // CRITICAL: Reset GUI mode tracking
    ctx->last_processed_hash = 0;
    
    // Clear legal moves cache
    ctx->invalidate_legal_moves_cache();

    compute_observation(env, ctx);

    // CRITICAL: Ensure outputs are initialized and terminal flags are cleared
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;
    
    // For dual agent mode, also reset second agent
    if (ctx->dual_agent_self_play_mode) {
        env->terminals[1] = 0;
        env->rewards[1] = 0.0f;
    }
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
        DBG("[OBSERVATION_DEBUG] Legal move masking DISABLED - all actions allowed" << std::endl);
        for (int i = 0; i < 4674; ++i) env->observations[idx + i] = 1.0f;
    } else {
        // mark only actually legal moves (O(#legal_moves))
        const auto &legal_moves = ctx->get_legal_moves_cached();
        DBG("[OBSERVATION_DEBUG] Computing legal move mask - " << legal_moves.size() << " legal moves found" << std::endl);
        
        int legal_actions_set = 0;
        for (const auto &mv : legal_moves) {
            int action_id = chess::ChessBoard::move_to_action(mv);
            if (action_id >= 0 && action_id < 4674) {
                env->observations[idx + action_id] = 1.0f;
                legal_actions_set++;
                
                // Debug first few legal actions
                if (legal_actions_set <= 5) {
                    DBG("[OBSERVATION_DEBUG] Legal action " << action_id << " (move ");
                    DBG(char('a' + mv.from.x) << (mv.from.y + 1) << " -> ");
                    DBG(char('a' + mv.to.x) << (mv.to.y + 1));
                    if (mv.promotion != chess::EMPTY) {
                        DBG(" promotion " << (int)mv.promotion);
                    }
                    DBG(")" << std::endl);
                }
            } else {
                DBG("[OBSERVATION_DEBUG] WARNING: Invalid action ID " << action_id << " for move ");
                DBG(char('a' + mv.from.x) << (mv.from.y + 1) << " -> ");
                DBG(char('a' + mv.to.x) << (mv.to.y + 1) << std::endl);
            }
        }
        
        DBG("[OBSERVATION_DEBUG] Set " << legal_actions_set << " legal actions in mask" << std::endl);
        // Pass move (action 0) is intentionally left disabled for chess.
    }

    // obs is now [board_features_0, ..., board_features_1343, legal_move_mask_0, ..., legal_move_mask_4673]
    // board_features is 1344 floats, legal_move_mask is 4674 floats
    idx += 4674;
    assert(idx == 6018);
}

void compute_dual_agent_observations(CChess* env, ChessContext* ctx) {
    // For dual-agent mode, we need to compute separate observations for each agent
    // Agent 0 (White): Always sees the board from white's perspective
    // Agent 1 (Black): Always sees the board from black's perspective
    
    chess::Color current_player = ctx->board.side_to_move();
    DBG("[DUAL_OBS_DEBUG] Computing dual agent observations - current player: " << (current_player == chess::WHITE ? "WHITE" : "BLACK") << std::endl);
    
    // Compute base observation features (same for both agents)
    int base_idx = 0;
    
    // For both agents, compute the same board features (first 1344 floats)
    for (int agent = 0; agent < 2; agent++) {
        int agent_offset = agent * 6018;  // Each agent has 6018 floats
        int idx = agent_offset;
        
        DBG("[DUAL_OBS_DEBUG] Computing observations for agent " << agent << " (" << (agent == 0 ? "WHITE" : "BLACK") << ") at offset " << agent_offset << std::endl);
        
        // 12 piece planes (6 types × 2 colors) - same for both agents
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
        
        // Empty squares plane - same for both agents
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                env->observations[idx++] = ctx->board.at({int8_t(x), int8_t(y)}).type == chess::EMPTY ? 1.0f : 0.0f;
            }
        }
        
        // Repetition count plane - same for both agents
        uint64_t hash = ctx->board.hash();
        int reps = ctx->position_history[hash];
        float rep_val = (reps - 1) / 2.0f; // Maps 1->0, 2->0.5, 3->1
        for (int i = 0; i < 64; i++) {
            env->observations[idx++] = rep_val;
        }
        
        // Side to move plane - DIFFERENT for each agent
        // Agent 0 (White): 0.0 when white to move, 1.0 when black to move
        // Agent 1 (Black): 1.0 when white to move, 0.0 when black to move
        float side_val;
        if (agent == 0) {
            // White agent perspective
            side_val = ctx->board.side_to_move() == chess::WHITE ? 0.0f : 1.0f;
        } else {
            // Black agent perspective  
            side_val = ctx->board.side_to_move() == chess::BLACK ? 0.0f : 1.0f;
        }
        for (int i = 0; i < 64; i++) {
            env->observations[idx++] = side_val;
        }
        
        // Irreversible move counter plane - same for both agents
        float halfmove_val = ctx->board.get_halfmove_clock() / 101.0f;
        for (int i = 0; i < 64; i++) {
            env->observations[idx++] = halfmove_val;
        }
        
        // 4 castling rights planes - same for both agents
        uint8_t rights = ctx->board.get_castling_rights();
        for (int i = 0; i < 4; i++) {
            float castle_val = (rights & (1 << (3-i))) ? 1.0f : 0.0f;
            for (int j = 0; j < 64; j++) {
                env->observations[idx++] = castle_val;
            }
        }
        
        // En passant target square plane - same for both agents
        int8_t ep_square = ctx->board.get_ep_square();
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                env->observations[idx++] = (ep_square == y * 8 + x) ? 1.0f : 0.0f;
            }
        }
        
        // Should be exactly 1344 floats at this point for this agent
        assert(idx == agent_offset + 1344);
        
        // Build legal move mask - DIFFERENT for each agent
        // First zero the mask
        for (int i = 0; i < 4674; ++i) env->observations[idx + i] = 0.0f;

        if (env->debug_disable_mask) {
            // Debug mode: all moves legal for both agents
            DBG("[DUAL_OBS_DEBUG] Legal move masking DISABLED for agent " << agent << " - all actions allowed" << std::endl);
            for (int i = 0; i < 4674; ++i) env->observations[idx + i] = 1.0f;
        } else {
            // Agent 0 (White): Only show legal moves when it's white's turn
            // Agent 1 (Black): Only show legal moves when it's black's turn
            chess::Color agent_color = (agent == 0) ? chess::WHITE : chess::BLACK;
            
            DBG("[DUAL_OBS_DEBUG] Agent " << agent << " (" << (agent_color == chess::WHITE ? "WHITE" : "BLACK") << ") turn check: ");
            DBG("current_player=" << (current_player == chess::WHITE ? "WHITE" : "BLACK"));
            
            if (current_player == agent_color) {
                // It's this agent's turn - show legal moves
                const auto &legal_moves = ctx->get_legal_moves_cached();
                DBG(", it's their turn - setting " << legal_moves.size() << " legal moves" << std::endl);
                
                int legal_actions_set = 0;
                for (const auto &mv : legal_moves) {
                    int action_id = chess::ChessBoard::move_to_action(mv);
                    if (action_id >= 0 && action_id < 4674) {
                        env->observations[idx + action_id] = 1.0f;
                        legal_actions_set++;
                        
                        // Debug first few legal actions
                        if (legal_actions_set <= 3) {
                            DBG("[DUAL_OBS_DEBUG] Agent " << agent << " legal action " << action_id << " (move ");
                            DBG(char('a' + mv.from.x) << (mv.from.y + 1) << " -> ");
                            DBG(char('a' + mv.to.x) << (mv.to.y + 1));
                            if (mv.promotion != chess::EMPTY) {
                                DBG(" promotion " << (int)mv.promotion);
                            }
                            DBG(")" << std::endl);
                        }
                    } else {
                        DBG("[DUAL_OBS_DEBUG] WARNING: Agent " << agent << " invalid action ID " << action_id << " for move ");
                        DBG(char('a' + mv.from.x) << (mv.from.y + 1) << " -> ");
                        DBG(char('a' + mv.to.x) << (mv.to.y + 1) << std::endl);
                    }
                }
                DBG("[DUAL_OBS_DEBUG] Agent " << agent << " set " << legal_actions_set << " legal actions in mask" << std::endl);
            } else {
                DBG(", not their turn - mask remains all zeros" << std::endl);
            }
            // If it's not this agent's turn, legal mask remains all zeros
        }
        
        // Move to next section
        idx += 4674;
        assert(idx == agent_offset + 6018);
    }
}

void c_step(CChess* env) {
    auto* ctx = (ChessContext*)env->context;
    
    // DEBUG: Track function entry and board state
    static int step_call_counter = 0;
    step_call_counter++;
    
    chess::Color entry_side_to_move = ctx->board.side_to_move();
    uint64_t entry_hash = ctx->board.hash();
    
    DBG("[C_STEP_DEBUG] ===== c_step called (call #" << step_call_counter << ") =====" << std::endl);
    DBG("[C_STEP_DEBUG] ENTRY - Side to move: " << (entry_side_to_move == chess::WHITE ? "WHITE" : "BLACK") << std::endl);
    DBG("[C_STEP_DEBUG] ENTRY - Board hash: " << entry_hash << std::endl);
    DBG("[C_STEP_DEBUG] dual_agent_self_play_mode: " << ctx->dual_agent_self_play_mode << std::endl);
    DBG("[C_STEP_DEBUG] self_play_mode: " << ctx->self_play_mode << std::endl);
    
    // REMOVED PROBLEMATIC GUARD: Don't auto-reset on terminal - let caller handle it
    // The previous guard was causing immediate resets and 1-step games
    
    // Check if we're in dual agent self-play mode
    if (ctx->dual_agent_self_play_mode) {
        DBG("[C_STEP_DEBUG] Taking dual agent path" << std::endl);
        c_step_dual_agent(env);
    } else {
        // Original single agent logic for backward compatibility
        DBG("[C_STEP_DEBUG] Taking single agent path" << std::endl);
        c_step_single_agent(env);
    }
    
    // DEBUG: Track function exit and board state
    chess::Color exit_side_to_move = ctx->board.side_to_move();
    uint64_t exit_hash = ctx->board.hash();
    
    DBG("[C_STEP_DEBUG] EXIT - Side to move: " << (exit_side_to_move == chess::WHITE ? "WHITE" : "BLACK") << std::endl);
    DBG("[C_STEP_DEBUG] EXIT - Board hash: " << exit_hash << std::endl);
    DBG("[C_STEP_DEBUG] EXIT - Side changed during c_step: " << (entry_side_to_move != exit_side_to_move ? "YES" : "NO") << std::endl);
    DBG("[C_STEP_DEBUG] ===== c_step complete (call #" << step_call_counter << ") =====" << std::endl);
}

void c_step_dual_agent(CChess* env) {
    auto* ctx = (ChessContext*)env->context;
    
    // In dual agent mode, we expect actions[0] = white_action, actions[1] = black_action
    // But we only execute the action for the player whose turn it is
    
    chess::Color current_player = ctx->board.side_to_move();
    int action_idx = (current_player == chess::WHITE) ? env->actions[0] : env->actions[1];
    
    // DEBUG: Print action and current player
    DBG("[DUAL_AGENT_DEBUG] Current player: " << (current_player == chess::WHITE ? "WHITE" : "BLACK") << std::endl);
    DBG("[DUAL_AGENT_DEBUG] Received actions - White: " << env->actions[0] << ", Black: " << env->actions[1] << std::endl);
    DBG("[DUAL_AGENT_DEBUG] Using action: " << action_idx << " for " << (current_player == chess::WHITE ? "WHITE" : "BLACK") << std::endl);
    
    // IMPORTANT: We only process the action from the current player
    // The other agent's action is ignored (not counted as invalid)
    
    // Initialize rewards for both agents
    float white_reward = 0.0f;
    float black_reward = 0.0f;
    bool terminal = false;
    
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
    
    // Compute initial material balance
    int white_material_before = material_value(chess::WHITE);
    int black_material_before = material_value(chess::BLACK);
    
    // Early terminal check
    const auto early_legal_moves = ctx->get_legal_moves_cached();
    DBG("[DUAL_AGENT_DEBUG] Legal moves available: " << early_legal_moves.size() << std::endl);
    
    if (early_legal_moves.empty()) {
        bool early_checkmate = ctx->board.is_checkmate();
        bool early_stalemate = ctx->board.is_stalemate();
        bool early_insuffmat = ctx->board.is_insufficient_material();
        
        DBG("[DUAL_AGENT_DEBUG] No legal moves - checkmate: " << early_checkmate << ", stalemate: " << early_stalemate << ", insufficient material: " << early_insuffmat << std::endl);
        
        if (early_checkmate || early_stalemate || early_insuffmat) {
            terminal = true;
            
            if (early_checkmate) {
                // Current player is checkmated - loses
                if (current_player == chess::WHITE) {
                    white_reward = env->reward_loss_white;
                    black_reward = env->reward_win_black;
                    ctx->c_reward_loss_white += env->reward_loss_white;
                    ctx->c_reward_win_black += env->reward_win_black;
                    ctx->c_white_checkmated += 1;
                } else {
                    black_reward = env->reward_loss_black;
                    white_reward = env->reward_win_white;
                    ctx->c_reward_loss_black += env->reward_loss_black;
                    ctx->c_reward_win_white += env->reward_win_white;
                    ctx->c_black_checkmated += 1;
                }
            } else {
                // Draw
                white_reward = env->reward_draw;
                black_reward = env->reward_draw;
                ctx->c_reward_draw_white += env->reward_draw;
                ctx->c_reward_draw_black += env->reward_draw;
                
                if (early_stalemate) ctx->c_stalemate += 1;
                if (early_insuffmat) ctx->c_insufficient_material += 1;
            }
            
            // Set outputs
            env->rewards[0] = white_reward;  // White agent reward
            env->rewards[1] = black_reward;  // Black agent reward
            env->terminals[0] = 1;
            env->terminals[1] = 1;
            
            // Update episode returns
            ctx->c_episode_return_white += white_reward;
            ctx->c_episode_return_black += black_reward;
            ctx->episode_return += white_reward + black_reward;
            
            // FIXED: Determine outcomes based on CHESS RULES, not rewards
            // Never use rewards to determine game outcomes!
            bool win = false;
            bool loss = false; 
            bool draw = false;
            
            // Determine outcome based on actual game state
            if (early_checkmate) {
                // Someone was checkmated - this is a win/loss
                win = true;
            } else {
                // Stalemate or insufficient material - this is a draw
                draw = true;
            }
            
            add_log(env, ctx, win, loss, draw);
            return;
        }
    }
    
    // Decode and validate the move from the CURRENT PLAYER ONLY
    chess::Move selected_move = chess::action_to_move_lookup(action_idx, ctx->board);
    
    DBG("[DUAL_AGENT_DEBUG] Action " << action_idx << " decoded to move: ");
    if (selected_move.from.x >= 0 && selected_move.from.y >= 0 && selected_move.to.x >= 0 && selected_move.to.y >= 0) {
        DBG("from " << char('a' + selected_move.from.x) << (selected_move.from.y + 1));
        DBG(" to " << char('a' + selected_move.to.x) << (selected_move.to.y + 1));
        if (selected_move.promotion != chess::EMPTY) {
            DBG(" promotion " << (int)selected_move.promotion);
        }
        DBG(std::endl);
    } else {
        DBG("INVALID COORDINATES (" << (int)selected_move.from.x << "," << (int)selected_move.from.y << " -> " << (int)selected_move.to.x << "," << (int)selected_move.to.y << ")" << std::endl);
    }
    
    bool is_legal = false;
    {
        chess::ChessBoard tmp = ctx->board;
        is_legal = tmp.apply_move(selected_move);
    }
    
    DBG("[DUAL_AGENT_DEBUG] Move legality check: " << (is_legal ? "LEGAL" : "ILLEGAL") << std::endl);
    
    // DEBUG: If move is illegal, check if action was in legal mask
    if (!is_legal) {
        DBG("[DUAL_AGENT_DEBUG] *** ILLEGAL MOVE DETECTED ***" << std::endl);
        DBG("[DUAL_AGENT_DEBUG] Action " << action_idx << " should have been masked!" << std::endl);
        
        // Check if this action was in the legal move mask
        // This requires checking the observation that was used to generate this action
        compute_dual_agent_observations(env, ctx);
        chess::Color agent_color = current_player;
        int agent_index = (agent_color == chess::WHITE) ? 0 : 1;
        int mask_offset = agent_index * 6018 + 1344; // Skip to legal mask for this agent
        
        DBG("[DUAL_AGENT_DEBUG] Legal mask for action " << action_idx << ": " << env->observations[mask_offset + action_idx] << std::endl);
        
        // Print some legal moves for comparison
        DBG("[DUAL_AGENT_DEBUG] Available legal moves:" << std::endl);
        for (size_t i = 0; i < std::min((size_t)5, early_legal_moves.size()); i++) {
            const auto& legal_move = early_legal_moves[i];
            int legal_action = chess::ChessBoard::move_to_action(legal_move);
            DBG("[DUAL_AGENT_DEBUG]   Move " << i << ": action " << legal_action);
            DBG(" from " << char('a' + legal_move.from.x) << (legal_move.from.y + 1));
            DBG(" to " << char('a' + legal_move.to.x) << (legal_move.to.y + 1));
            DBG(" (mask value: " << env->observations[mask_offset + legal_action] << ")" << std::endl);
        }
    }
    
    if (is_legal) {
        // Valid move - apply it
        ctx->step_count += 1;
        
        // DEBUG: Track board state before and after move application
        chess::Color player_before_move = ctx->board.side_to_move();
        uint64_t hash_before_move = ctx->board.hash();
        std::string fen_before_move = ctx->board.to_fen();
        
        DBG("[MOVE_APPLICATION_DEBUG] BEFORE move application:" << std::endl);
        DBG("[MOVE_APPLICATION_DEBUG]   Side to move: " << (player_before_move == chess::WHITE ? "WHITE" : "BLACK") << std::endl);
        DBG("[MOVE_APPLICATION_DEBUG]   Board hash: " << hash_before_move << std::endl);
        DBG("[MOVE_APPLICATION_DEBUG]   FEN: " << fen_before_move << std::endl);
        
        // Give valid move reward to current player
        if (current_player == chess::WHITE) {
            white_reward += env->reward_valid;
            ctx->c_reward_valid += env->reward_valid;
        } else {
            black_reward += env->reward_valid;
            ctx->c_reward_valid += env->reward_valid;
        }
        
        // Store en passant square before applying the move
        int8_t ep_square_before = ctx->board.get_ep_square();
        
        // Apply the move
        bool applied_ok = ctx->board.apply_move(selected_move);
        ctx->board.invalidate_cache();
        ctx->invalidate_legal_moves_cache();  // Clear cache after move
        
        // DEBUG: Track board state after move application
        chess::Color player_after_move = ctx->board.side_to_move();
        uint64_t hash_after_move = ctx->board.hash();
        std::string fen_after_move = ctx->board.to_fen();
        
        DBG("[MOVE_APPLICATION_DEBUG] AFTER move application:" << std::endl);
        DBG("[MOVE_APPLICATION_DEBUG]   Applied successfully: " << applied_ok << std::endl);
        DBG("[MOVE_APPLICATION_DEBUG]   Side to move: " << (player_after_move == chess::WHITE ? "WHITE" : "BLACK") << std::endl);
        DBG("[MOVE_APPLICATION_DEBUG]   Board hash: " << hash_after_move << std::endl);
        DBG("[MOVE_APPLICATION_DEBUG]   FEN: " << fen_after_move << std::endl);
        DBG("[MOVE_APPLICATION_DEBUG]   Side changed: " << (player_before_move != player_after_move ? "YES" : "NO") << std::endl);
        
        DBG("[DUAL_AGENT_DEBUG] Move applied successfully: " << applied_ok << std::endl);
        
        // Track action for complete game logging
        if (applied_ok) {
            ctx->complete_game_actions.push_back(action_idx);
        }
        
        // Track last move for logging
        if (applied_ok) {
            env->log.last_move_from = selected_move.from.index();
            env->log.last_move_to = selected_move.to.index();
            env->log.last_move_promotion = (selected_move.promotion == chess::QUEEN) ? 1 :
                                          (selected_move.promotion == chess::ROOK) ? 2 :
                                          (selected_move.promotion == chess::BISHOP) ? 3 :
                                          (selected_move.promotion == chess::KNIGHT) ? 4 : 0;
            env->log.game_moves_count += 1;
        }
        
        // Track special moves
        if (applied_ok) {
            // Castling
            if (selected_move.is_castle_short) {
                if (current_player == chess::WHITE) {
                    ctx->c_white_castle_kingside += 1;
                } else {
                    ctx->c_black_castle_kingside += 1;
                }
            } else if (selected_move.is_castle_long) {
                if (current_player == chess::WHITE) {
                    ctx->c_white_castle_queenside += 1;
                } else {
                    ctx->c_black_castle_queenside += 1;
                }
            }
            
            // En passant
            if (selected_move.piece.type == chess::PAWN && 
                selected_move.to.index() == ep_square_before && ep_square_before >= 0) {
                if (current_player == chess::WHITE) {
                    ctx->c_en_passant_white += 1;
                } else {
                    ctx->c_en_passant_black += 1;
                }
            }
            
            // Promotions
            if (selected_move.promotion != chess::EMPTY) {
                if (current_player == chess::WHITE) {
                    ctx->c_white_promotion_count += 1;
                    switch (selected_move.promotion) {
                        case chess::QUEEN:  ctx->c_white_promotion_queen += 1; break;
                        case chess::ROOK:   ctx->c_white_promotion_rook += 1; break;
                        case chess::BISHOP: ctx->c_white_promotion_bishop += 1; break;
                        case chess::KNIGHT: ctx->c_white_promotion_knight += 1; break;
                        default: break;
                    }
                } else {
                    ctx->c_black_promotion_count += 1;
                    switch (selected_move.promotion) {
                        case chess::QUEEN:  ctx->c_black_promotion_queen += 1; break;
                        case chess::ROOK:   ctx->c_black_promotion_rook += 1; break;
                        case chess::BISHOP: ctx->c_black_promotion_bishop += 1; break;
                        case chess::KNIGHT: ctx->c_black_promotion_knight += 1; break;
                        default: break;
                    }
                }
            }
        }
        
        // Check reward
        bool current_move_is_check = ctx->board.is_check();
        if (current_move_is_check) {
            float check_reward = (current_player == chess::WHITE) ? env->reward_check_white : env->reward_check_black;
            
            // Anti-exploitation: diminishing returns for consecutive checks
            if (ctx->last_move_was_check && ctx->last_checking_color == current_player) {
                if (current_player == chess::WHITE) {
                    ctx->consecutive_checks_white++;
                } else {
                    ctx->consecutive_checks_black++;
                }
                
                int consecutive_checks = (current_player == chess::WHITE) ? 
                                       ctx->consecutive_checks_white : ctx->consecutive_checks_black;
                
                if (consecutive_checks > 3) {
                    check_reward *= 0.1f * (1.0f / consecutive_checks);
                } else if (consecutive_checks > 2) {
                    check_reward *= 0.5f;
                }
            } else {
                ctx->consecutive_checks_white = (current_player == chess::WHITE) ? 1 : 0;
                ctx->consecutive_checks_black = (current_player == chess::BLACK) ? 1 : 0;
            }
            
            // Track total checks
            if (current_player == chess::WHITE) {
                ctx->checks_given_white++;
                white_reward += check_reward;
                ctx->c_reward_check_white += check_reward;
            } else {
                ctx->checks_given_black++;
                black_reward += check_reward;
                ctx->c_reward_check_black += check_reward;
            }
            
            ctx->last_move_was_check = true;
            ctx->last_checking_color = current_player;
        } else {
            ctx->consecutive_checks_white = 0;
            ctx->consecutive_checks_black = 0;
            ctx->last_move_was_check = false;
        }
        
        // Track move counts - ONLY count legal moves from current player
        if (current_player == chess::WHITE) {
            ctx->c_white_moves += 1;
        } else {
            ctx->c_black_moves += 1;
        }
        ctx->c_valid_moves += 1;
        
    } else {
        // Invalid move - penalize ONLY the current player
        // The other agent's action is ignored completely
        DBG("[DUAL_AGENT_DEBUG] Applying invalid move penalty to " << (current_player == chess::WHITE ? "WHITE" : "BLACK") << std::endl);
        
        // CRITICAL DEBUG: Log complete invalid move details
        DBG("[INVALID_MOVE_DEBUG] ===== INVALID MOVE DETECTED =====" << std::endl);
        static int invalid_move_counter = 0;
        invalid_move_counter++;
        DBG("[INVALID_MOVE_DEBUG] Invalid move #" << invalid_move_counter << std::endl);
        DBG("[INVALID_MOVE_DEBUG] Current player: " << (current_player == chess::WHITE ? "WHITE" : "BLACK") << std::endl);
        DBG("[INVALID_MOVE_DEBUG] Invalid action: " << action_idx << std::endl);
        DBG("[INVALID_MOVE_DEBUG] Board FEN: " << ctx->board.to_fen() << std::endl);
        DBG("[INVALID_MOVE_DEBUG] Board hash: " << ctx->board.hash() << std::endl);
        
        // Log all legal moves and their action IDs
        const auto& all_legal_moves = ctx->get_legal_moves_cached();
        DBG("[INVALID_MOVE_DEBUG] Legal moves available (" << all_legal_moves.size() << "):" << std::endl);
        for (size_t i = 0; i < all_legal_moves.size(); i++) {
            const auto& legal_move = all_legal_moves[i];
            int legal_action = chess::ChessBoard::move_to_action(legal_move);
            DBG("[INVALID_MOVE_DEBUG]   " << i << ": action " << legal_action);
            DBG(" from " << char('a' + legal_move.from.x) << (legal_move.from.y + 1));
            DBG(" to " << char('a' + legal_move.to.x) << (legal_move.to.y + 1));
            if (legal_move.promotion != chess::EMPTY) {
                DBG(" promotion " << (int)legal_move.promotion);
            }
            DBG(std::endl);
        }
        
        // Check if the action was properly masked
        compute_dual_agent_observations(env, ctx);
        chess::Color agent_color = current_player;
        int agent_index = (agent_color == chess::WHITE) ? 0 : 1;
        int mask_offset = agent_index * 6018 + 1344;
        float mask_value = env->observations[mask_offset + action_idx];
        
        DBG("[INVALID_MOVE_DEBUG] Action " << action_idx << " mask value: " << mask_value << std::endl);
        DBG("[INVALID_MOVE_DEBUG] Agent " << agent_index << " (" << (agent_color == chess::WHITE ? "WHITE" : "BLACK") << ") mask offset: " << mask_offset << std::endl);
        
        // Check if action is out of bounds
        if (action_idx < 0 || action_idx >= 4674) {
            DBG("[INVALID_MOVE_DEBUG] ACTION OUT OF BOUNDS! Valid range: 0-4673" << std::endl);
        }
        
        DBG("[INVALID_MOVE_DEBUG] ===== END INVALID MOVE DEBUG =====" << std::endl);
        
        if (current_player == chess::WHITE) {
            white_reward += env->reward_invalid_white;
            ctx->c_invalid_moves_white += 1;
        } else {
            black_reward += env->reward_invalid_black;
            ctx->c_invalid_moves_black += 1;
        }
        ctx->board.invalidate_cache();
        ctx->invalidate_legal_moves_cache();  // Clear cache after invalid move
    }
    
    // Check for game over conditions
    if (is_legal) {
        if (ctx->board.is_checkmate()) {
            terminal = true;
            // The player who just moved won, opponent lost
            if (current_player == chess::WHITE) {
                white_reward += env->reward_win_white;
                black_reward += env->reward_loss_black;
                ctx->c_reward_win_white += env->reward_win_white;
                ctx->c_reward_loss_black += env->reward_loss_black;
                ctx->c_black_checkmated += 1;
            } else {
                black_reward += env->reward_win_black;
                white_reward += env->reward_loss_white;
                ctx->c_reward_win_black += env->reward_win_black;
                ctx->c_reward_loss_white += env->reward_loss_white;
                ctx->c_white_checkmated += 1;
            }
        } else if (ctx->board.is_stalemate()) {
            terminal = true;
            float draw_reward = env->reward_draw;
            
            // Anti-collusion penalties
            if (ctx->step_count < 30) {
                draw_reward *= 0.3f;
            }
            if (draw_reward > 0.0f) {
                draw_reward = -0.05f;
            }
            
            white_reward += draw_reward;
            black_reward += draw_reward;
            ctx->c_reward_draw_white += draw_reward;
            ctx->c_reward_draw_black += draw_reward;
            ctx->c_stalemate += 1;
        } else if (ctx->board.is_insufficient_material()) {
            terminal = true;
            float draw_reward = env->reward_draw;
            if (draw_reward > 0.0f) {
                draw_reward = 0.0f;
            }
            
            white_reward += draw_reward;
            black_reward += draw_reward;
            ctx->c_reward_draw_white += draw_reward;
            ctx->c_reward_draw_black += draw_reward;
            ctx->c_insufficient_material += 1;
        } else if (ctx->board.get_halfmove_clock() >= 100) {
            terminal = true;
            float draw_reward = env->reward_draw;
            if (draw_reward > 0.0f) {
                draw_reward = -0.2f;
            } else {
                draw_reward *= 1.5f;
            }
            
            white_reward += draw_reward;
            black_reward += draw_reward;
            ctx->c_reward_draw_white += draw_reward;
            ctx->c_reward_draw_black += draw_reward;
            ctx->c_fifty_move_rule += 1;
        } else {
            // Update position history and check for threefold repetition
            uint64_t current_hash = ctx->board.hash();
            ctx->position_history[current_hash]++;
            
            if (ctx->position_history[current_hash] >= 3) {
                terminal = true;
                float draw_reward = env->reward_draw;
                
                // Anti-collusion penalties
                if (ctx->step_count < 20) {
                    draw_reward *= 0.1f;
                } else if (ctx->step_count < 40) {
                    draw_reward *= 0.5f;
                }
                
                if (draw_reward > 0.0f) {
                    draw_reward = -0.1f;
                }
                
                white_reward += draw_reward;
                black_reward += draw_reward;
                ctx->c_reward_draw_white += draw_reward;
                ctx->c_reward_draw_black += draw_reward;
                ctx->c_threefold_repetition += 1;
            }
        }
    }
    
    // Max depth check
    if (!terminal && ctx->step_count >= ctx->max_depth) {
        terminal = true;
        float draw_reward = env->reward_draw;
        if (draw_reward > 0.0f) {
            draw_reward = -0.3f;
        } else {
            draw_reward *= 2.0f;
        }
        
        white_reward += draw_reward;
        black_reward += draw_reward;
        ctx->c_reward_draw_white += draw_reward;
        ctx->c_reward_draw_black += draw_reward;
        ctx->c_max_depth += 1;
    }
    
    // Material differential rewards
    if (is_legal) {
        int white_material_after = material_value(chess::WHITE);
        int black_material_after = material_value(chess::BLACK);
        
        int white_material_delta = white_material_after - white_material_before;
        int black_material_delta = black_material_after - black_material_before;
        
        if (white_material_delta != 0) {
            float white_mat_reward = white_material_delta * env->reward_material_diff_white;
            white_reward += white_mat_reward;
            ctx->c_reward_material_diff_white += white_mat_reward;
            
            if (white_material_delta > 0) {
                white_reward += env->reward_agent_captures_enemy_piece;
                ctx->c_reward_agent_captures_enemy_piece += env->reward_agent_captures_enemy_piece;
            } else if (white_material_delta < 0) {
                white_reward += env->reward_enemy_captures_agent_piece;
                ctx->c_reward_enemy_captures_agent_piece += env->reward_enemy_captures_agent_piece;
            }
        }
        
        if (black_material_delta != 0) {
            float black_mat_reward = black_material_delta * env->reward_material_diff_black;
            black_reward += black_mat_reward;
            ctx->c_reward_material_diff_black += black_mat_reward;
            
            if (black_material_delta > 0) {
                black_reward += env->reward_agent_captures_enemy_piece;
                ctx->c_reward_agent_captures_enemy_piece += env->reward_agent_captures_enemy_piece;
            } else if (black_material_delta < 0) {
                black_reward += env->reward_enemy_captures_agent_piece;
                ctx->c_reward_enemy_captures_agent_piece += env->reward_enemy_captures_agent_piece;
            }
        }
    }
    
    // Set outputs
    env->rewards[0] = white_reward;  // White agent reward
    env->rewards[1] = black_reward;  // Black agent reward
    env->terminals[0] = terminal ? 1 : 0;
    env->terminals[1] = terminal ? 1 : 0;
    
    // Update episode returns
    ctx->c_episode_return_white += white_reward;
    ctx->c_episode_return_black += black_reward;
    ctx->episode_return += white_reward + black_reward;
    
    // Terminal handling
    if (terminal) {
        // FIXED: Determine outcomes based on CHESS RULES, not rewards
        // Never use rewards to determine game outcomes!
        bool win = false;
        bool loss = false; 
        bool draw = false;
        
        // Determine outcome based on actual game state counters
        if (ctx->c_max_depth > 0) {
            // Max depth reached - always a draw
            draw = true;
        } else if (ctx->c_black_checkmated > 0) {
            // White checkmated black - white wins
            win = true;
        } else if (ctx->c_white_checkmated > 0) {
            // Black checkmated white - black wins  
            win = true; // From perspective of winner
        } else {
            // Any other terminal condition is a draw
            draw = true;
        }
        
        add_log(env, ctx, win, loss, draw);
        // Don't reset here - let the training loop handle it
    }
    
    // DEBUG: Check board state consistency before computing observations
    chess::Color pre_obs_side = ctx->board.side_to_move();
    uint64_t pre_obs_hash = ctx->board.hash();
    std::string pre_obs_fen = ctx->board.to_fen();
    
    DBG("[CONSISTENCY_DEBUG] Board state BEFORE observation computation:" << std::endl);
    DBG("[CONSISTENCY_DEBUG]   Side to move: " << (pre_obs_side == chess::WHITE ? "WHITE" : "BLACK") << std::endl);
    DBG("[CONSISTENCY_DEBUG]   Board hash: " << pre_obs_hash << std::endl);
    DBG("[CONSISTENCY_DEBUG]   FEN: " << pre_obs_fen << std::endl);
    
    // Compute observations for both agents (if not terminal)
    if (!terminal) {
        // For dual agents, we need to compute observations from both perspectives
        compute_dual_agent_observations(env, ctx);
    }
    
    // DEBUG: Check board state consistency after computing observations
    chess::Color post_obs_side = ctx->board.side_to_move();
    uint64_t post_obs_hash = ctx->board.hash();
    std::string post_obs_fen = ctx->board.to_fen();
    
    DBG("[CONSISTENCY_DEBUG] Board state AFTER observation computation:" << std::endl);
    DBG("[CONSISTENCY_DEBUG]   Side to move: " << (post_obs_side == chess::WHITE ? "WHITE" : "BLACK") << std::endl);
    DBG("[CONSISTENCY_DEBUG]   Board hash: " << post_obs_hash << std::endl);
    DBG("[CONSISTENCY_DEBUG]   Board state changed during obs computation: " << (pre_obs_hash != post_obs_hash ? "YES" : "NO") << std::endl);
    
    if (pre_obs_hash != post_obs_hash) {
        DBG("[CONSISTENCY_DEBUG] *** BOARD STATE CORRUPTION DETECTED *** " << std::endl);
        DBG("[CONSISTENCY_DEBUG] FEN changed from: " << pre_obs_fen << std::endl);
        DBG("[CONSISTENCY_DEBUG] FEN changed to:   " << post_obs_fen << std::endl);
    }
}

void c_step_single_agent(CChess* env) {
    // Original single agent step logic for backward compatibility
    auto* ctx = (ChessContext*)env->context;
    
    // In single-agent mode, the human/AI agent plays white, and Stockfish (or other logic) plays black
    // We only get one action from the training loop, which is for the white player
    
    chess::Color current_player = ctx->board.side_to_move();
    int action_idx = env->actions[0];  // Single action for single agent
    
    // DEBUG: Print action and current player
    DBG("[SINGLE_AGENT_DEBUG] Current player: " << (current_player == chess::WHITE ? "WHITE" : "BLACK") << std::endl);
    DBG("[SINGLE_AGENT_DEBUG] Received action: " << action_idx << std::endl);
    
    // Initialize rewards and terminal state
    float reward = 0.0f;
    bool terminal = false;
    
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
                    case chess::EMPTY:  break;
                    default: break;
                }
            }
        }
        return total;
    };
    
    // Compute initial material balance
    int white_material_before = material_value(chess::WHITE);
    int black_material_before = material_value(chess::BLACK);
    
    // Early terminal check
    const auto early_legal_moves = ctx->get_legal_moves_cached();
    DBG("[SINGLE_AGENT_DEBUG] Legal moves available: " << early_legal_moves.size() << std::endl);
    
    if (early_legal_moves.empty()) {
        bool early_checkmate = ctx->board.is_checkmate();
        bool early_stalemate = ctx->board.is_stalemate();
        bool early_insuffmat = ctx->board.is_insufficient_material();
        
        DBG("[SINGLE_AGENT_DEBUG] No legal moves - checkmate: " << early_checkmate << ", stalemate: " << early_stalemate << ", insufficient material: " << early_insuffmat << std::endl);
        
        if (early_checkmate || early_stalemate || early_insuffmat) {
            terminal = true;
            
            if (early_checkmate) {
                // Current player is checkmated - loses
                if (current_player == chess::WHITE) {
                    ctx->c_white_checkmated += 1;
                } else {
                    ctx->c_black_checkmated += 1;
                }
            } else {
                // Draw
                reward = env->reward_draw;
                
                if (early_stalemate) ctx->c_stalemate += 1;
                if (early_insuffmat) ctx->c_insufficient_material += 1;
            }
            
            // Set outputs
            env->rewards[0] = reward;
            env->terminals[0] = 1;
            
            // Update episode return
            ctx->episode_return += reward;
            
            // FIXED: Determine outcomes based on CHESS RULES, not rewards
            // Never use rewards to determine game outcomes!
            bool win = false;
            bool loss = false; 
            bool draw = false;
            
            // Determine outcome based on actual game state
            if (early_checkmate) {
                // Someone was checkmated - this is a win/loss
                win = true;
            } else {
                // Stalemate or insufficient material - this is a draw
                draw = true;
            }
            
            add_log(env, ctx, win, loss, draw);
            return;
        }
    }
    
    // Handle the move based on whose turn it is
    if (current_player == chess::WHITE) {
        // White's turn - use the action provided by the agent/human
        chess::Move selected_move = chess::action_to_move_lookup(action_idx, ctx->board);
        
        DBG("[SINGLE_AGENT_DEBUG] Action " << action_idx << " decoded to move: ");
        if (selected_move.from.x >= 0 && selected_move.from.y >= 0 && selected_move.to.x >= 0 && selected_move.to.y >= 0) {
            DBG("from " << char('a' + selected_move.from.x) << (selected_move.from.y + 1));
            DBG(" to " << char('a' + selected_move.to.x) << (selected_move.to.y + 1));
            if (selected_move.promotion != chess::EMPTY) {
                DBG(" promotion " << (int)selected_move.promotion);
            }
            DBG(std::endl);
        } else {
            DBG("INVALID COORDINATES (" << (int)selected_move.from.x << "," << (int)selected_move.from.y << " -> " << (int)selected_move.to.x << "," << (int)selected_move.to.y << ")" << std::endl);
        }
        
        bool is_legal = false;
        {
            chess::ChessBoard tmp = ctx->board;
            is_legal = tmp.apply_move(selected_move);
        }
        
        DBG("[SINGLE_AGENT_DEBUG] Move legality check: " << (is_legal ? "LEGAL" : "ILLEGAL") << std::endl);
        
        // DEBUG: If move is illegal, check if action was in legal mask
        if (!is_legal) {
            DBG("[SINGLE_AGENT_DEBUG] *** ILLEGAL MOVE DETECTED ***" << std::endl);
            DBG("[SINGLE_AGENT_DEBUG] Action " << action_idx << " should have been masked!" << std::endl);
            
            // Check if this action was in the legal move mask
            compute_observation(env, ctx);
            int mask_offset = 1344; // Skip to legal mask
            
            DBG("[SINGLE_AGENT_DEBUG] Legal mask for action " << action_idx << ": " << env->observations[mask_offset + action_idx] << std::endl);
            
            // Print some legal moves for comparison
            DBG("[SINGLE_AGENT_DEBUG] Available legal moves:" << std::endl);
            for (size_t i = 0; i < std::min((size_t)5, early_legal_moves.size()); i++) {
                const auto& legal_move = early_legal_moves[i];
                int legal_action = chess::ChessBoard::move_to_action(legal_move);
                DBG("[SINGLE_AGENT_DEBUG]   Move " << i << ": action " << legal_action);
                DBG(" from " << char('a' + legal_move.from.x) << (legal_move.from.y + 1));
                DBG(" to " << char('a' + legal_move.to.x) << (legal_move.to.y + 1));
                DBG(" (mask value: " << env->observations[mask_offset + legal_action] << ")" << std::endl);
            }
        }
        
        if (is_legal) {
            // Valid move - apply it
            ctx->step_count += 1;
            reward += env->reward_valid;
            ctx->c_reward_valid += env->reward_valid;
            
            // Store en passant square before applying the move
            int8_t ep_square_before = ctx->board.get_ep_square();
            
            // Apply the move
            bool applied_ok = ctx->board.apply_move(selected_move);
            ctx->board.invalidate_cache();
            ctx->invalidate_legal_moves_cache();  // Clear cache after move
            
            DBG("[SINGLE_AGENT_DEBUG] Move applied successfully: " << applied_ok << std::endl);
            
            // Track action for complete game logging
            if (applied_ok) {
                ctx->complete_game_actions.push_back(action_idx);
            }
            
            // Track last move for logging
            if (applied_ok) {
                env->log.last_move_from = selected_move.from.index();
                env->log.last_move_to = selected_move.to.index();
                env->log.last_move_promotion = (selected_move.promotion == chess::QUEEN) ? 1 :
                                              (selected_move.promotion == chess::ROOK) ? 2 :
                                              (selected_move.promotion == chess::BISHOP) ? 3 :
                                              (selected_move.promotion == chess::KNIGHT) ? 4 : 0;
                env->log.game_moves_count += 1;
            }
            
            // Track special moves
            if (applied_ok) {
                // Castling
                if (selected_move.is_castle_short) {
                    ctx->c_white_castle_kingside += 1;
                } else if (selected_move.is_castle_long) {
                    ctx->c_white_castle_queenside += 1;
                }
                
                // En passant
                if (selected_move.piece.type == chess::PAWN && 
                    selected_move.to.index() == ep_square_before && ep_square_before >= 0) {
                    ctx->c_en_passant_white += 1;
                }
                
                // Promotions
                if (selected_move.promotion != chess::EMPTY) {
                    ctx->c_white_promotion_count += 1;
                    switch (selected_move.promotion) {
                        case chess::QUEEN:  ctx->c_white_promotion_queen += 1; break;
                        case chess::ROOK:   ctx->c_white_promotion_rook += 1; break;
                        case chess::BISHOP: ctx->c_white_promotion_bishop += 1; break;
                        case chess::KNIGHT: ctx->c_white_promotion_knight += 1; break;
                        default: break;
                    }
                }
            }
            
            // Check reward
            bool current_move_is_check = ctx->board.is_check();
            if (current_move_is_check) {
                reward += env->reward_check_white;
                ctx->c_reward_check_white += env->reward_check_white;
            }
            
            ctx->c_white_moves += 1;
            ctx->c_valid_moves += 1;
            
        } else {
            // Invalid move - penalize
            DBG("[SINGLE_AGENT_DEBUG] Applying invalid move penalty to WHITE" << std::endl);
            reward += env->reward_invalid_white;  // Single agent is always white
            ctx->c_invalid_moves_white += 1;
            ctx->board.invalidate_cache();
        }
        
        // Material differential rewards
        if (is_legal) {
            int white_material_after = material_value(chess::WHITE);
            int white_material_delta = white_material_after - white_material_before;
            
            if (white_material_delta != 0) {
                float mat_reward = white_material_delta * env->reward_material_diff_white;
                reward += mat_reward;
                ctx->c_reward_material_diff_white += mat_reward;
                
                if (white_material_delta > 0) {
                    reward += env->reward_agent_captures_enemy_piece;
                    ctx->c_reward_agent_captures_enemy_piece += env->reward_agent_captures_enemy_piece;
                } else if (white_material_delta < 0) {
                    reward += env->reward_enemy_captures_agent_piece;
                    ctx->c_reward_enemy_captures_agent_piece += env->reward_enemy_captures_agent_piece;
                }
            }
        }
        
            } else {
        // Black's turn - handle Stockfish or other AI
        if (ctx->stockfish_enabled && ctx->sf) {
            // Let Stockfish make the move
            std::string fen = ctx->board.to_fen();
            auto [move_uci, eval_cp] = ctx->sf->bestmove_with_score(fen);
            ctx->stockfish_eval = eval_cp;
            
            chess::Move sf_move = uci_to_move(move_uci, ctx->board);
            if (sf_move.from.is_valid() && sf_move.to.is_valid()) {
                bool applied_ok = ctx->board.apply_move(sf_move);
                if (applied_ok) {
                    ctx->step_count += 1;
                    ctx->c_black_moves += 1;
                    ctx->c_valid_moves += 1;
                    ctx->board.invalidate_cache();
                    
                    // Track action for complete game logging
                    int stockfish_action = chess::ChessBoard::move_to_action(sf_move);
                    ctx->complete_game_actions.push_back(stockfish_action);
                    
                    // Track last move for logging
                    env->log.last_move_from = sf_move.from.index();
                    env->log.last_move_to = sf_move.to.index();
                    env->log.last_move_promotion = (sf_move.promotion == chess::QUEEN) ? 1 :
                                                  (sf_move.promotion == chess::ROOK) ? 2 :
                                                  (sf_move.promotion == chess::BISHOP) ? 3 :
                                                  (sf_move.promotion == chess::KNIGHT) ? 4 : 0;
                    env->log.game_moves_count += 1;
                    
                    // Track special moves for black
                    if (sf_move.is_castle_short) {
                        ctx->c_black_castle_kingside += 1;
                    } else if (sf_move.is_castle_long) {
                        ctx->c_black_castle_queenside += 1;
                    }
                    
                    // En passant
                    if (sf_move.piece.type == chess::PAWN && 
                        sf_move.to.index() == ctx->board.get_ep_square()) {
                        ctx->c_en_passant_black += 1;
                    }
                    
                    // Promotions
                    if (sf_move.promotion != chess::EMPTY) {
                        ctx->c_black_promotion_count += 1;
                        switch (sf_move.promotion) {
                            case chess::QUEEN:  ctx->c_black_promotion_queen += 1; break;
                            case chess::ROOK:   ctx->c_black_promotion_rook += 1; break;
                            case chess::BISHOP: ctx->c_black_promotion_bishop += 1; break;
                            case chess::KNIGHT: ctx->c_black_promotion_knight += 1; break;
                            default: break;
                        }
                    }
                }
            }
        } else {
            // FIXED: When Stockfish is disabled, make random black moves
            // This is critical for "agent vs random" mode to work properly
            const auto& legal_moves = ctx->board.legal_moves();
            if (!legal_moves.empty()) {
                // Select random legal move for black
                std::uniform_int_distribution<int> dist(0, legal_moves.size() - 1);
                int move_index = dist(ctx->rng);
                const chess::Move& random_move = legal_moves[move_index];
                
                bool applied_ok = ctx->board.apply_move(random_move);
                if (applied_ok) {
                    ctx->step_count += 1;
                    ctx->c_black_moves += 1;
                    ctx->c_valid_moves += 1;
                    ctx->board.invalidate_cache();
                    
                    // Track action for complete game logging
                    int random_action = chess::ChessBoard::move_to_action(random_move);
                    ctx->complete_game_actions.push_back(random_action);
                    
                    // Track last move for logging
                    env->log.last_move_from = random_move.from.index();
                    env->log.last_move_to = random_move.to.index();
                    env->log.last_move_promotion = (random_move.promotion == chess::QUEEN) ? 1 :
                                                  (random_move.promotion == chess::ROOK) ? 2 :
                                                  (random_move.promotion == chess::BISHOP) ? 3 :
                                                  (random_move.promotion == chess::KNIGHT) ? 4 : 0;
                    env->log.game_moves_count += 1;
                    
                    // Track special moves for black
                    if (random_move.is_castle_short) {
                        ctx->c_black_castle_kingside += 1;
                    } else if (random_move.is_castle_long) {
                        ctx->c_black_castle_queenside += 1;
                    }
                    
                    // En passant
                    if (random_move.piece.type == chess::PAWN && 
                        random_move.to.index() == ctx->board.get_ep_square()) {
                        ctx->c_en_passant_black += 1;
                    }
                    
                    // Promotions
                    if (random_move.promotion != chess::EMPTY) {
                        ctx->c_black_promotion_count += 1;
                        switch (random_move.promotion) {
                            case chess::QUEEN:  ctx->c_black_promotion_queen += 1; break;
                            case chess::ROOK:   ctx->c_black_promotion_rook += 1; break;
                            case chess::BISHOP: ctx->c_black_promotion_bishop += 1; break;
                            case chess::KNIGHT: ctx->c_black_promotion_knight += 1; break;
                            default: break;
                        }
                    }
                }
            }
        }
    }
    
    // Check for game over conditions
    if (ctx->board.is_checkmate()) {
        terminal = true;
        if (ctx->board.side_to_move() == chess::WHITE) {
            // White is checkmated - black wins
            reward += env->reward_draw;  // Treat as draw for single agent
            ctx->c_white_checkmated += 1;
        } else {
            // Black is checkmated - white wins
            reward += env->reward_draw;  // Treat as draw for single agent
            ctx->c_black_checkmated += 1;
        }
    } else if (ctx->board.is_stalemate()) {
        terminal = true;
        reward += env->reward_draw;
        ctx->c_stalemate += 1;
    } else if (ctx->board.is_insufficient_material()) {
        terminal = true;
        reward += env->reward_draw;
        ctx->c_insufficient_material += 1;
    } else if (ctx->board.get_halfmove_clock() >= 100) {
        terminal = true;
        reward += env->reward_draw;
        ctx->c_fifty_move_rule += 1;
    } else {
        // Update position history and check for threefold repetition
        uint64_t current_hash = ctx->board.hash();
        ctx->position_history[current_hash]++;
        
        if (ctx->position_history[current_hash] >= 3) {
            terminal = true;
            reward += env->reward_draw;
            ctx->c_threefold_repetition += 1;
        }
    }
    
    // Max depth check
    if (!terminal && ctx->step_count >= ctx->max_depth) {
        terminal = true;
        reward += env->reward_draw;
        ctx->c_max_depth += 1;
    }
    
    // Set outputs
    env->rewards[0] = reward;
    env->terminals[0] = terminal ? 1 : 0;
    
    // Update episode return
    ctx->episode_return += reward;
    
    // Terminal handling
    if (terminal) {
        // FIXED: Determine outcomes based on CHESS RULES, not rewards
        // Never use rewards to determine game outcomes!
        bool win = false;
        bool loss = false; 
        bool draw = false;
        
        // Determine outcome based on actual game state counters
        if (ctx->c_max_depth > 0) {
            // Max depth reached - always a draw
            draw = true;
        } else if (ctx->c_black_checkmated > 0) {
            // Black was checkmated - white wins
            win = true;
        } else if (ctx->c_white_checkmated > 0) {
            // White was checkmated - black wins
            win = true; // From perspective of winner
        } else {
            // Any other terminal condition is a draw
            draw = true;
        }
        
        add_log(env, ctx, win, loss, draw);
        // Don't reset here - let the training loop handle it
    }
    
    // Compute observations (if not terminal)
    if (!terminal) {
        compute_observation(env, ctx);
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