#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <unistd.h> 
#include "raylib.h"

typedef uint64_t Bitboard;
typedef uint64_t Key;
typedef uint32_t Square;
typedef uint32_t Move;
typedef uint32_t Piece;
typedef int32_t Value;
typedef int32_t Depth;
typedef uint8_t ChessColor;

#define U64 uint64_t

enum {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE = 64
};

enum { PAWN = 1, KNIGHT, BISHOP, ROOK, QUEEN, KING };

enum {
    NO_PIECE = 0,
    W_PAWN = 1, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = 9, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING
};

enum { CHESS_WHITE = 0, CHESS_BLACK = 1 };

enum {
    NO_CASTLING = 0,
    WHITE_OO = 1, WHITE_OOO = 2,
    BLACK_OO = 4, BLACK_OOO = 8,
    WHITE_CASTLING = 3, BLACK_CASTLING = 12,
    ANY_CASTLING = 15
};


enum { NORMAL, PROMOTION, ENPASSANT, CASTLING };

enum {
    NORTH = 8, EAST = 1, SOUTH = -8, WEST = -1,
    NORTH_EAST = 9, SOUTH_EAST = -7,
    NORTH_WEST = 7, SOUTH_WEST = -9
};

enum { RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8 };
enum { FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H };


enum {
    VALUE_ZERO = 0,
    VALUE_DRAW = 0,
    VALUE_MATE = 32000,
    VALUE_INFINITE = 32001,
};



#define MOVE_NONE 0
#define MOVE_NULL 65

#define make_move(from, to) ((Move)((to) | ((from) << 6)))
#define make_promotion(from, to, pt) ((Move)((to) | ((from) << 6) | (PROMOTION << 14) | (((pt) - KNIGHT) << 12)))
#define make_enpassant(from, to) ((Move)((to) | ((from) << 6) | (ENPASSANT << 14)))
#define make_castling(from, to) ((Move)((to) | ((from) << 6) | (CASTLING << 14)))

#define from_sq(m) (((m) >> 6) & 0x3f)
#define to_sq(m) ((m) & 0x3f)
#define type_of_m(m) ((m) >> 14)
#define promotion_type(m) ((((m) >> 12) & 3) + KNIGHT)

#define make_square(f, r) ((Square)(((r) << 3) + (f)))
#define file_of(s) ((s) & 7)
#define rank_of(s) ((s) >> 3)
#define make_piece(c, pt) ((Piece)(((c) << 3) + (pt)))
#define type_of_p(p) ((p) & 7)
#define color_of(p) ((p) >> 3)
#define relative_square(c, s) ((Square)((s) ^ ((c) * 56)))
#define relative_rank(c, r) ((r) ^ ((c) * 7))

#define pieces(pos) ((pos)->byTypeBB[0])
#define pieces_p(pos, p) ((pos)->byTypeBB[p])
#define pieces_c(pos, c) ((pos)->byColorBB[c])
#define pieces_cp(pos, c, p) (pieces_p(pos, p) & pieces_c(pos, c))
#define piece_on(pos, s) ((pos)->board[s])
#define is_empty(pos, s) (piece_on(pos, s) == NO_PIECE)

#define MAX_SERVED_MOVES 64
#define MAX_GAME_PLIES 2048

#define FileABB 0x0101010101010101ULL
#define FileBBB (FileABB << 1)
#define FileCBB (FileABB << 2)
#define FileDBB (FileABB << 3)
#define FileEBB (FileABB << 4)
#define FileFBB (FileABB << 5)
#define FileGBB (FileABB << 6)
#define FileHBB (FileABB << 7)

#define Rank1BB 0xFFULL
#define Rank2BB (Rank1BB << 8)
#define Rank3BB (Rank1BB << 16)
#define Rank4BB (Rank1BB << 24)
#define Rank5BB (Rank1BB << 32)
#define Rank6BB (Rank1BB << 40)
#define Rank7BB (Rank1BB << 48)
#define Rank8BB (Rank1BB << 56)

static const char* PIECE_CHARS[] = {
    "",
    "P", "N", "B", "R", "Q", "K",
    "", "",
    "p", "n", "b", "r", "q", "k"
};

static const char* PIECE_UNICODE[] = {
    "",
    "♙", "♘", "♗", "♖", "♕", "♔",
    "", "",
    "♟", "♞", "♝", "♜", "♛", "♚"
};

static const int PIECE_VALUES_CP[7] = {0, 100, 320, 330, 500, 900, 0};

// Piece-Square Tables (numbers calculated from Stockfish)
static inline int mirror_file(int file) {
    return file < 4 ? file : (7 - file);
}

static const int PawnPST_MG[64] = {
    0,   0,   0,   0,   0,   0,   0,   0,
    -11,  7,  7, 17, 17,  7,  7, -11,
    -16, -3, 23, 23, 23, 23, -3, -16,
    -14, -7, 20, 24, 24, 20, -7, -14,
    -5, -2, -1, 12, 12, -1, -2,  -5,
    -11,-12, -2,  4,  4, -2,-12, -11,
    -2, 20,-10, -2, -2,-10, 20,  -2,
    0,   0,   0,   0,   0,   0,   0,   0
};

static const int PawnPST_EG[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,
    -3, -1,  7,  2,  2,  7, -1, -3,
    -2,  2,  6, -1, -1,  6,  2, -2,
    7, -4, -8,  2,  2, -8, -4,  7,
    13, 10, -1, -8, -8, -1, 10, 13,
    16,  6,  1, 16, 16,  1,  6, 16,
    1,-12,  6, 25, 25,  6,-12,  1,
    0,  0,  0,  0,  0,  0,  0,  0
};

static const int KnightPST_MG[64] = {
    -169,-96,-80,-79,-79,-80,-96,-169,
    -79,-39,-24, -9, -9,-24,-39, -79,
    -64,-20,  4, 19, 19,  4,-20, -64,
    -28,  5, 41, 47, 47, 41,  5, -28,
    -29, 13, 42, 52, 52, 42, 13, -29,
    -11, 28, 63, 55, 55, 63, 28, -11,
    -67,-21,  6, 37, 37,  6,-21, -67,
    -200,-80,-53,-32,-32,-53,-80,-200
};

static const int KnightPST_EG[64] = {
    -105,-74,-46,-18,-18,-46,-74,-105,
    -70,-56,-15,  6,  6,-15,-56, -70,
    -38,-33, -5, 27, 27, -5,-33, -38,
    -36,  0, 13, 34, 34, 13,  0, -36,
    -41,-20,  4, 35, 35,  4,-20, -41,
    -51,-38,-17, 19, 19,-17,-38, -51,
    -64,-45,-37, 16, 16,-37,-45, -64,
    -98,-89,-53,-16,-16,-53,-89, -98
};

static const int BishopPST_MG[64] = {
    -49, -7,-10,-34,-34,-10, -7,-49,
    -24,  9, 15,  1,  1, 15,  9,-24,
    -9, 22, -3, 12, 12, -3, 22, -9,
    4,  9, 18, 40, 40, 18,  9,  4,
    -8, 27, 13, 30, 30, 13, 27, -8,
    -17, 14, -6,  6,  6, -6, 14,-17,
    -19,-13,  7,-11,-11,  7,-13,-19,
    -47, -7,-17,-29,-29,-17, -7,-47
};

static const int BishopPST_EG[64] = {
    -58,-31,-37,-19,-19,-37,-31,-58,
    -34, -9,-14,  4,  4,-14, -9,-34,
    -23,  0, -3, 16, 16, -3,  0,-23,
    -26, -3, -5, 16, 16, -5, -3,-26,
    -26, -4, -7, 14, 14, -7, -4,-26,
    -24, -2,  0, 13, 13,  0, -2,-24,
    -34,-10,-12,  6,  6,-12,-10,-34,
    -55,-32,-36,-17,-17,-36,-32,-55
};

static const int RookPST_MG[64] = {
    -24,-15, -8,  0,  0, -8,-15,-24,
    -18, -5, -1,  1,  1, -1, -5,-18,
    -19,-10,  1,  0,  0,  1,-10,-19,
    -21, -7, -4, -4, -4, -4, -7,-21,
    -21,-12, -1,  4,  4, -1,-12,-21,
    -23,-10,  1,  6,  6,  1,-10,-23,
    -11,  8,  9, 12, 12,  9,  8,-11,
    -25,-18,-11,  2,  2,-11,-18,-25
};

static const int RookPST_EG[64] = {
    0,  3,  0,  3,  3,  0,  3,  0,
    -7, -5, -5, -1, -1, -5, -5, -7,
    6, -7,  3,  3,  3,  3, -7,  6,
    0,  4, -2,  1,  1, -2,  4,  0,
    -7,  5, -5, -7, -7, -5,  5, -7,
    3,  2, -1,  3,  3, -1,  2,  3,
    -1,  7, 11, -1, -1, 11,  7, -1,
    6,  4,  6,  2,  2,  6,  4,  6
};

static const int QueenPST_MG[64] = {
    3, -5, -5,  4,  4, -5, -5,  3,
    -3,  5,  8, 12, 12,  8,  5, -3,
    -3,  6, 13,  7,  7, 13,  6, -3,
    4,  5,  9,  8,  8,  9,  5,  4,
    0, 14, 12,  5,  5, 12, 14,  0,
    -4, 10,  6,  8,  8,  6, 10, -4,
    -5,  6, 10,  8,  8, 10,  6, -5,
    -2, -2,  1, -2, -2,  1, -2, -2
};

static const int QueenPST_EG[64] = {
    -69,-57,-47,-26,-26,-47,-57,-69,
    -55,-31,-22, -4, -4,-22,-31,-55,
    -39,-18, -9,  3,  3, -9,-18,-39,
    -23, -3, 13, 24, 24, 13, -3,-23,
    -29, -6,  9, 21, 21,  9, -6,-29,
    -38,-18,-12,  1,  1,-12,-18,-38,
    -50,-27,-24, -8, -8,-24,-27,-50,
    -75,-52,-43,-36,-36,-43,-52,-75
};

static const int KingPST_MG[64] = {
    272,325,273,190,190,273,325,272,
    277,305,241,183,183,241,305,277,
    198,253,168,120,120,168,253,198,
    169,191,136,108,108,136,191,169,
    145,176,112, 69, 69,112,176,145,
    122,159, 85, 36, 36, 85,159,122,
    87,120, 64, 25, 25, 64,120, 87,
    64, 87, 49,  0,  0, 49, 87, 64
};

static const int KingPST_EG[64] = {
    0, 41, 80, 93, 93, 80, 41,  0,
    57, 98,138,131,131,138, 98, 57,
    86,138,165,173,173,165,138, 86,
    103,152,168,169,169,168,152,103,
    98,166,197,194,194,197,166, 98,
    87,164,174,189,189,174,164, 87,
    40, 99,128,141,141,128, 99, 40,
    5, 60, 75, 75, 75, 75, 60,  5
};

static uint64_t prng_state = 1070372;
static inline uint64_t prng_rand(void) {
    prng_state ^= prng_state >> 12;
    prng_state ^= prng_state << 25;
    prng_state ^= prng_state >> 27;
    return prng_state * 2685821657736338717ULL;
}

extern Bitboard SquareBB[65];
extern Bitboard FileBB[8];
extern Bitboard RankBB[8];
extern Bitboard PawnAttacks[2][64];
extern Bitboard KnightAttacks[64];
extern Bitboard KingAttacks[64];
extern Bitboard BetweenBB[64][64];
extern Bitboard LineBB[64][64];

extern Bitboard rook_file_attacks[8][256];
extern Bitboard rook_rank_attacks[8][256];

typedef struct {
    Key psq[16][64];
    Key enpassant[8];
    Key castling[16];
    Key side;
} Zobrist;

extern Zobrist zob;

typedef struct {
    Bitboard byTypeBB[7];    // [0]=all, [1-6]=PAWN,KNIGHT,BISHOP,ROOK,QUEEN,KING
    Bitboard byColorBB[2];
    uint8_t board[64];
    uint8_t pieceCount[16];
    ChessColor sideToMove;
    uint8_t castlingRights;
    uint8_t epSquare;
    uint8_t rule50;
    uint16_t gamePly;
    Key key;
    int16_t materialScore;
    int16_t psqtScore;
    int16_t cachedEval;
    uint8_t evalValid;
} Position;

typedef struct {
    Move move;
    int16_t value;
} ExtMove;

typedef struct {
    ExtMove moves[256];
    int count;
} MoveList;

enum {
    O_BOARD = 0,
    O_SIDE = 768,
    O_CASTLE = 770,
    O_EP = 786,
    O_PICK_PHASE = 851,
    O_SELECTED_PIECE = 853,
    O_VALID_PIECES = 917,
    O_VALID_DESTS = 981,
    O_VALID_PROMOS = 1045,
    OBS_SIZE = 1077
};


typedef struct {
    float perf;
    float score;
    float draw_rate;
    float timeout_rate;
    float chess_moves;          // Average chess moves per game
    float episode_length;       // Average episode length in ticks (2-phase system)
    float episode_return;
    float invalid_action_rate;
    float game_length_score;    
    float n;
} Log;

typedef struct {
    Texture2D pieces;
    int cell_size;
    Font piece_font;
    int use_unicode_pieces;
} Client;

typedef struct {
    Piece captured;
    uint8_t castlingRights;
    uint8_t epSquare;
    uint8_t rule50;
    int16_t materialScore;
    int16_t psqtScore;
    Key key;
    uint8_t pliesFromNull;
} UndoInfo;

typedef struct {
    Log log;
    Client* client;
    uint8_t* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    
    Position pos;
    MoveList legal_moves;
    ChessColor legal_moves_side;
    Key legal_moves_key;
    int game_result;
    int tick;
    int chess_moves;
    int max_moves;
    float reward_draw;
    int render_fps;
    int selfplay;
    int human_play;
    
    char starting_fen[128];
    char** fen_curriculum;
    int num_fens;
    
    UndoInfo undo_stack[MAX_GAME_PLIES];
    int undo_stack_ptr;
    
    int invalid_actions_this_episode;
    
    int pick_phase[2];
    Square selected_square[2];
    MoveList valid_destinations[2];
    float reward_invalid_piece;
    float reward_invalid_move;
    float reward_valid_piece;
    float reward_valid_move;
    float reward_material;
    float reward_position;  // PST-based positional reward
    float reward_castling;
    float reward_repetition;
    
    int enable_50_move_rule;
    int enable_threefold_repetition;
    
    int learner_color; // 0 for White, 1 for Black
    int human_color;
    float white_score;
    float black_score;
    float learner_wins;
    float learner_losses; 
    float learner_draws;
    char last_result[32];
} Chess;


static inline Bitboard sq_bb(Square s) {
    return SquareBB[s];
}

static inline int popcount(Bitboard b) {
    return __builtin_popcountll(b);
}

static inline Square lsb(Bitboard b) {
    assert(b);
    return __builtin_ctzll(b);
}

static inline Square pop_lsb(Bitboard* b) {
    Square s = lsb(*b);
    *b &= *b - 1;
    return s;
}

static inline bool more_than_one(Bitboard b) {
    return b & (b - 1);
}

static inline Bitboard shift_bb(int Direction, Bitboard b) {
    return Direction == NORTH ? b << 8
         : Direction == SOUTH ? b >> 8
         : Direction == EAST ? (b & ~FileHBB) << 1
         : Direction == WEST ? (b & ~FileABB) >> 1
         : Direction == NORTH_EAST ? (b & ~FileHBB) << 9
         : Direction == SOUTH_EAST ? (b & ~FileHBB) >> 7
         : Direction == NORTH_WEST ? (b & ~FileABB) << 7
         : Direction == SOUTH_WEST ? (b & ~FileABB) >> 9
         : 0;
}

static inline Bitboard pawn_attacks_bb(ChessColor c, Square s) {
    return PawnAttacks[c][s];
}

static inline Bitboard knight_attacks_bb(Square s) {
    return KnightAttacks[s];
}

static inline Bitboard king_attacks_bb(Square s) {
    return KingAttacks[s];
}


static inline Bitboard rook_attacks_bb(Square s, Bitboard occupied) {
    Bitboard attacks = 0;
    int r = rank_of(s), f = file_of(s);
    for (int rr = r + 1; rr < 8; rr++) {
        Square sq = make_square(f, rr);
        attacks |= sq_bb(sq);
        if (occupied & sq_bb(sq)) break;
    }
    for (int rr = r - 1; rr >= 0; rr--) {
        Square sq = make_square(f, rr);
        attacks |= sq_bb(sq);
        if (occupied & sq_bb(sq)) break;
    }
    for (int ff = f + 1; ff < 8; ff++) {
        Square sq = make_square(ff, r);
        attacks |= sq_bb(sq);
        if (occupied & sq_bb(sq)) break;
    }
    for (int ff = f - 1; ff >= 0; ff--) {
        Square sq = make_square(ff, r);
        attacks |= sq_bb(sq);
        if (occupied & sq_bb(sq)) break;
    }
    return attacks;
}

static inline Bitboard bishop_attacks_bb(Square s, Bitboard occupied) {
    Bitboard attacks = 0;
    int r = rank_of(s), f = file_of(s);
    for (int rr = r + 1, ff = f + 1; rr < 8 && ff < 8; rr++, ff++) {
        Square sq = make_square(ff, rr);
        attacks |= sq_bb(sq);
        if (occupied & sq_bb(sq)) break;
    }
    for (int rr = r - 1, ff = f + 1; rr >= 0 && ff < 8; rr--, ff++) {
        Square sq = make_square(ff, rr);
        attacks |= sq_bb(sq);
        if (occupied & sq_bb(sq)) break;
    }
    for (int rr = r - 1, ff = f - 1; rr >= 0 && ff >= 0; rr--, ff--) {
        Square sq = make_square(ff, rr);
        attacks |= sq_bb(sq);
        if (occupied & sq_bb(sq)) break;
    }
    for (int rr = r + 1, ff = f - 1; rr < 8 && ff >= 0; rr++, ff--) {
        Square sq = make_square(ff, rr);
        attacks |= sq_bb(sq);
        if (occupied & sq_bb(sq)) break;
    }
    return attacks;
}

static inline Bitboard queen_attacks_bb(Square s, Bitboard occupied) {
    return rook_attacks_bb(s, occupied) | bishop_attacks_bb(s, occupied);
}


Bitboard SquareBB[65];
Bitboard FileBB[8];
Bitboard RankBB[8];
Bitboard PawnAttacks[2][64];
Bitboard KnightAttacks[64];
Bitboard KingAttacks[64];
Bitboard BetweenBB[64][64];
Bitboard LineBB[64][64];
Zobrist zob;

static bool bitboards_initialized = false;

void init_bitboards(void) {
    if (bitboards_initialized) return;
    
    for (int c = 0; c < 2; c++) {
        for (int pt = PAWN; pt <= KING; pt++) {
            for (int s = 0; s < 64; s++) {
                zob.psq[make_piece(c, pt)][s] = prng_rand();
            }
        }
    }
    for (int f = 0; f < 8; f++) {
        zob.enpassant[f] = prng_rand();
    }
    for (int cr = 0; cr < 16; cr++) {
        zob.castling[cr] = prng_rand();
    }
    zob.side = prng_rand();
    
    for (int i = 0; i < 64; i++) {
        SquareBB[i] = 1ULL << i;
    }
    SquareBB[64] = 0;
    
    FileBB[0] = FileABB; FileBB[1] = FileBBB; FileBB[2] = FileCBB; FileBB[3] = FileDBB;
    FileBB[4] = FileEBB; FileBB[5] = FileFBB; FileBB[6] = FileGBB; FileBB[7] = FileHBB;
    
    RankBB[0] = Rank1BB; RankBB[1] = Rank2BB; RankBB[2] = Rank3BB; RankBB[3] = Rank4BB;
    RankBB[4] = Rank5BB; RankBB[5] = Rank6BB; RankBB[6] = Rank7BB; RankBB[7] = Rank8BB;
    
    for (int s = 0; s < 64; s++) {
        Bitboard bb = sq_bb(s);
        PawnAttacks[CHESS_WHITE][s] = shift_bb(NORTH_WEST, bb) | shift_bb(NORTH_EAST, bb);
        PawnAttacks[CHESS_BLACK][s] = shift_bb(SOUTH_WEST, bb) | shift_bb(SOUTH_EAST, bb);
    }
    
    int knight_dirs[] = {-17, -15, -10, -6, 6, 10, 15, 17};
    for (int s = 0; s < 64; s++) {
        Bitboard attack = 0;
        int file = file_of(s);
        int rank = rank_of(s);
        
        for (int i = 0; i < 8; i++) {
            int to = s + knight_dirs[i];
            if (to >= 0 && to < 64) {
                int to_file = file_of(to);
                int to_rank = rank_of(to);
                if (abs(to_file - file) <= 2 && abs(to_rank - rank) <= 2) {
                    attack |= sq_bb(to);
                }
            }
        }
        KnightAttacks[s] = attack;
    }
    
    int king_dirs[] = {-9, -8, -7, -1, 1, 7, 8, 9};
    for (int s = 0; s < 64; s++) {
        Bitboard attack = 0;
        int file = file_of(s);
        
        for (int i = 0; i < 8; i++) {
            int to = s + king_dirs[i];
            if (to >= 0 && to < 64) {
                int to_file = file_of(to);
                if (abs(to_file - file) <= 1) {
                    attack |= sq_bb(to);
                }
            }
        }
        KingAttacks[s] = attack;
    }
    
    for (int s1 = 0; s1 < 64; s1++) {
        for (int s2 = 0; s2 < 64; s2++) {
            BetweenBB[s1][s2] = 0;
            LineBB[s1][s2] = 0;
            
            if (s1 == s2) continue;
            
            int f1 = file_of(s1), r1 = rank_of(s1);
            int f2 = file_of(s2), r2 = rank_of(s2);
            int df = f2 - f1, dr = r2 - r1;
            
            if (df == 0 || dr == 0 || abs(df) == abs(dr)) {
                int step_f = df == 0 ? 0 : (df > 0 ? 1 : -1);
                int step_r = dr == 0 ? 0 : (dr > 0 ? 1 : -1);
                
                LineBB[s1][s2] = sq_bb(s1) | sq_bb(s2);
                
                int f = f1 + step_f;
                int r = r1 + step_r;
                
                while (f != f2 || r != r2) {
                    Square sq = make_square(f, r);
                    BetweenBB[s1][s2] |= sq_bb(sq);
                    LineBB[s1][s2] |= sq_bb(sq);
                    f += step_f;
                    r += step_r;
                }
            }
        }
    }
    
    bitboards_initialized = true;
}

static inline int get_pst_value(Piece pc, Square sq);

void pos_set_startpos(Position* pos) {
    memset(pos, 0, sizeof(Position));
    
    const char* fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    pos_set(pos, fen);
}

void pos_set(Position* pos, const char* fen) {
    memset(pos, 0, sizeof(Position));
    
    int rank = 7, file = 0;
    const char* ptr = fen;
    
    while (*ptr && *ptr != ' ') {
        char c = *ptr++;
        
        if (c == '/') {
            rank--;
            file = 0;
        } else if (c >= '1' && c <= '8') {
            file += c - '0';
        } else {
            Square sq = make_square(file, rank);
            Piece pc = NO_PIECE;
            int pt = 0, color = 0;
            
            switch (c) {
                case 'P': pc = W_PAWN; pt = PAWN; color = CHESS_WHITE; break;
                case 'N': pc = W_KNIGHT; pt = KNIGHT; color = CHESS_WHITE; break;
                case 'B': pc = W_BISHOP; pt = BISHOP; color = CHESS_WHITE; break;
                case 'R': pc = W_ROOK; pt = ROOK; color = CHESS_WHITE; break;
                case 'Q': pc = W_QUEEN; pt = QUEEN; color = CHESS_WHITE; break;
                case 'K': pc = W_KING; pt = KING; color = CHESS_WHITE; break;
                case 'p': pc = B_PAWN; pt = PAWN; color = CHESS_BLACK; break;
                case 'n': pc = B_KNIGHT; pt = KNIGHT; color = CHESS_BLACK; break;
                case 'b': pc = B_BISHOP; pt = BISHOP; color = CHESS_BLACK; break;
                case 'r': pc = B_ROOK; pt = ROOK; color = CHESS_BLACK; break;
                case 'q': pc = B_QUEEN; pt = QUEEN; color = CHESS_BLACK; break;
                case 'k': pc = B_KING; pt = KING; color = CHESS_BLACK; break;
            }
            
            if (pc != NO_PIECE) {
                pos->board[sq] = pc;
                pos->byTypeBB[pt] |= sq_bb(sq);
                pos->byColorBB[color] |= sq_bb(sq);
                pos->byTypeBB[0] |= sq_bb(sq);
                pos->pieceCount[pc]++;
            }
            file++;
        }
    }
    
    if (*ptr == ' ') ptr++;
    
    pos->sideToMove = (*ptr == 'w') ? CHESS_WHITE : CHESS_BLACK;
    ptr += 2;
    
    pos->castlingRights = NO_CASTLING;
    while (*ptr && *ptr != ' ') {
        if (*ptr == 'K') pos->castlingRights |= WHITE_OO;
        else if (*ptr == 'Q') pos->castlingRights |= WHITE_OOO;
        else if (*ptr == 'k') pos->castlingRights |= BLACK_OO;
        else if (*ptr == 'q') pos->castlingRights |= BLACK_OOO;
        ptr++;
    }
    

    if (*ptr == ' ') ptr++;
    
    pos->epSquare = SQ_NONE;
    if (*ptr != '-') {
        int ep_file = ptr[0] - 'a';
        int ep_rank = ptr[1] - '1';
        pos->epSquare = make_square(ep_file, ep_rank);
    }
    
    pos->materialScore = 0;
    pos->psqtScore = 0;
    
    for (Square sq = SQ_A1; sq <= SQ_H8; sq++) {
        Piece pc = pos->board[sq];
        if (pc == NO_PIECE) continue;
        
        int pt = type_of_p(pc);
        ChessColor c = color_of(pc);
        int sign = (c == CHESS_WHITE) ? 1 : -1;
        
        pos->materialScore += sign * PIECE_VALUES_CP[pt];
        
        pos->psqtScore += sign * get_pst_value(pc, sq);
    }
    
    pos->key = 0;
    for (Square sq = SQ_A1; sq <= SQ_H8; sq++) {
        Piece pc = pos->board[sq];
        if (pc != NO_PIECE) {
            pos->key ^= zob.psq[pc][sq];
        }
    }
    if (pos->sideToMove == CHESS_BLACK) {
        pos->key ^= zob.side;
    }
    if (pos->castlingRights) {
        pos->key ^= zob.castling[pos->castlingRights];
    }
    if (pos->epSquare != SQ_NONE) {
        pos->key ^= zob.enpassant[file_of(pos->epSquare)];
    }
}


static void add_move(MoveList* ml, Move m) {
    ml->moves[ml->count].move = m;
    ml->moves[ml->count].value = 0;
    ml->count++;
}

static void generate_pawn_moves(Position* pos, MoveList* ml, ChessColor us) {
    ChessColor them = !us;
    int up = (us == CHESS_WHITE) ? NORTH : SOUTH;
    Bitboard rank7 = (us == CHESS_WHITE) ? Rank7BB : Rank2BB;
    Bitboard rank3 = (us == CHESS_WHITE) ? Rank3BB : Rank6BB;
    
    Bitboard pawns = pieces_cp(pos, us, PAWN);
    Bitboard pawnsOn7 = pawns & rank7;
    Bitboard pawnsNotOn7 = pawns & ~rank7;
    
    Bitboard enemies = pieces_c(pos, them);
    Bitboard empty = ~pieces(pos);
    
    Bitboard b1 = shift_bb(up, pawnsNotOn7) & empty;
    Bitboard b2 = shift_bb(up, b1 & rank3) & empty;
    
    while (b1) {
        Square to = pop_lsb(&b1);
        add_move(ml, make_move(to - up, to));
    }
    
    while (b2) {
        Square to = pop_lsb(&b2);
        add_move(ml, make_move(to - up - up, to));
    }
    
    if (pawnsOn7) {
        Bitboard b3 = shift_bb(up, pawnsOn7) & empty;
        while (b3) {
            Square to = pop_lsb(&b3);
            Square from = to - up;
            add_move(ml, make_promotion(from, to, QUEEN));
            add_move(ml, make_promotion(from, to, ROOK));
            add_move(ml, make_promotion(from, to, BISHOP));
            add_move(ml, make_promotion(from, to, KNIGHT));
        }
    }
    
    Bitboard b4 = shift_bb(up + WEST, pawnsNotOn7) & enemies;
    Bitboard b5 = shift_bb(up + EAST, pawnsNotOn7) & enemies;
    
    while (b4) {
        Square to = pop_lsb(&b4);
        add_move(ml, make_move(to - up - WEST, to));
    }
    
    while (b5) {
        Square to = pop_lsb(&b5);
        add_move(ml, make_move(to - up - EAST, to));
    }
    
    if (pawnsOn7) {
        Bitboard b6 = shift_bb(up + WEST, pawnsOn7) & enemies;
        Bitboard b7 = shift_bb(up + EAST, pawnsOn7) & enemies;
        
        while (b6) {
            Square to = pop_lsb(&b6);
            Square from = to - up - WEST;
            add_move(ml, make_promotion(from, to, QUEEN));
            add_move(ml, make_promotion(from, to, ROOK));
            add_move(ml, make_promotion(from, to, BISHOP));
            add_move(ml, make_promotion(from, to, KNIGHT));
        }
        
        while (b7) {
            Square to = pop_lsb(&b7);
            Square from = to - up - EAST;
            add_move(ml, make_promotion(from, to, QUEEN));
            add_move(ml, make_promotion(from, to, ROOK));
            add_move(ml, make_promotion(from, to, BISHOP));
            add_move(ml, make_promotion(from, to, KNIGHT));
        }
    }
    
    if (pos->epSquare != SQ_NONE) {
        Bitboard ep_pawns = pawnsNotOn7 & pawn_attacks_bb(them, pos->epSquare);
        while (ep_pawns) {
            Square from = pop_lsb(&ep_pawns);
            add_move(ml, make_enpassant(from, pos->epSquare));
        }
    }
}

static void generate_piece_moves(Position* pos, MoveList* ml, int pt, ChessColor us) {
    Bitboard pieces_bb = pieces_cp(pos, us, pt);
    Bitboard target = ~pieces_c(pos, us);
    Bitboard occupied = pieces(pos);
    
    while (pieces_bb) {
        Square from = pop_lsb(&pieces_bb);
        Bitboard attacks = 0;
        
        switch (pt) {
            case KNIGHT:
                attacks = knight_attacks_bb(from);
                break;
            case BISHOP:
                attacks = bishop_attacks_bb(from, occupied);
                break;
            case ROOK:
                attacks = rook_attacks_bb(from, occupied);
                break;
            case QUEEN:
                attacks = queen_attacks_bb(from, occupied);
                break;
            case KING:
                attacks = king_attacks_bb(from);
                break;
        }
        
        attacks &= target;
        
        while (attacks) {
            Square to = pop_lsb(&attacks);
            add_move(ml, make_move(from, to));
        }
    }
}

static bool is_square_attacked(Position* pos, Square sq, ChessColor by_color) {
    Bitboard occupied = pieces(pos);
    
    if (pawn_attacks_bb(!by_color, sq) & pieces_cp(pos, by_color, PAWN))
        return true;
    
    if (knight_attacks_bb(sq) & pieces_cp(pos, by_color, KNIGHT))
        return true;
    
    if (bishop_attacks_bb(sq, occupied) & (pieces_cp(pos, by_color, BISHOP) | pieces_cp(pos, by_color, QUEEN)))
        return true;
    
    if (rook_attacks_bb(sq, occupied) & (pieces_cp(pos, by_color, ROOK) | pieces_cp(pos, by_color, QUEEN)))
        return true;
    
    if (king_attacks_bb(sq) & pieces_cp(pos, by_color, KING))
        return true;
    
    return false;
}

 static void generate_castling(Position* pos, MoveList* ml, ChessColor us) {
    Bitboard occupied = pieces(pos);
    
    if (us == CHESS_WHITE) {
        if (pos->castlingRights & WHITE_OO) {
            if (!(occupied & (sq_bb(SQ_F1) | sq_bb(SQ_G1)))) {
                add_move(ml, make_castling(SQ_E1, SQ_G1));
            }
        }
        if (pos->castlingRights & WHITE_OOO) {
            if (!(occupied & (sq_bb(SQ_D1) | sq_bb(SQ_C1) | sq_bb(SQ_B1)))) {
                add_move(ml, make_castling(SQ_E1, SQ_C1));
            }
        }
    } else {
        if (pos->castlingRights & BLACK_OO) {
            if (!(occupied & (sq_bb(SQ_F8) | sq_bb(SQ_G8)))) {
                add_move(ml, make_castling(SQ_E8, SQ_G8));
            }
        }
        if (pos->castlingRights & BLACK_OOO) {
            if (!(occupied & (sq_bb(SQ_D8) | sq_bb(SQ_C8) | sq_bb(SQ_B8)))) {
                add_move(ml, make_castling(SQ_E8, SQ_C8));
            }
        }
    }
}

static Bitboard attackers_to_sq(Position* pos, Square sq, Bitboard occupied) {
    return (pawn_attacks_bb(CHESS_WHITE, sq) & pieces_cp(pos, CHESS_BLACK, PAWN))
         | (pawn_attacks_bb(CHESS_BLACK, sq) & pieces_cp(pos, CHESS_WHITE, PAWN))
         | (knight_attacks_bb(sq) & pieces_p(pos, KNIGHT))
         | (king_attacks_bb(sq) & pieces_p(pos, KING))
         | (bishop_attacks_bb(sq, occupied) & (pieces_p(pos, BISHOP) | pieces_p(pos, QUEEN)))
         | (rook_attacks_bb(sq, occupied) & (pieces_p(pos, ROOK) | pieces_p(pos, QUEEN)));
}

static inline Bitboard all_pawn_attacks(Bitboard pawns, ChessColor c) {
    if (c == CHESS_WHITE) {
        return ((pawns << 7) & ~FileHBB) | ((pawns << 9) & ~FileABB);
    } else {
        return ((pawns >> 7) & ~FileABB) | ((pawns >> 9) & ~FileHBB);
    }
}

bool is_check(Position* pos, ChessColor c) {
    Bitboard king_bb = pieces_cp(pos, c, KING);
    if (!king_bb) return false;
    Square king_sq = lsb(king_bb);
    return (attackers_to_sq(pos, king_sq, pieces(pos)) & pieces_c(pos, !c)) != 0;
}

static inline bool is_legal_move(Position* pos, Move m) {
    ChessColor us = pos->sideToMove;
    ChessColor them = (ChessColor)!us;
    int mt = type_of_m(m);
    if (mt == CASTLING) {
        if (is_check(pos, us)) return false;
        Square from = from_sq(m), to = to_sq(m);
        Square mid = (from + to) / 2;
        if (is_square_attacked(pos, mid, them) || is_square_attacked(pos, to, them)) return false;
        return true;
    }
    if (mt == ENPASSANT) {
        Bitboard king_bb = pieces_cp(pos, us, KING);
        if (!king_bb) return false;
        Square ksq = lsb(king_bb);
        Square from = from_sq(m), to = to_sq(m);
        Square capsq = (us == CHESS_WHITE) ? (to - 8) : (to + 8);
        Bitboard occ = pieces(pos) ^ sq_bb(from) ^ sq_bb(capsq) ^ sq_bb(to);
        return (attackers_to_sq(pos, ksq, occ) & pieces_c(pos, them)) == 0;
    }
    UndoInfo u[1]; int p = 0;
    do_move(pos, m, u, &p);
    bool ok = !is_check(pos, us);
    undo_move(pos, m, u, &p);
    return ok;
}

static inline void generate_pseudo_legal(Position* pos, MoveList* ml, ChessColor us) {
    ml->count = 0;
    generate_pawn_moves(pos, ml, us);
    generate_piece_moves(pos, ml, KNIGHT, us);
    generate_piece_moves(pos, ml, BISHOP, us);
    generate_piece_moves(pos, ml, ROOK, us);
    generate_piece_moves(pos, ml, QUEEN, us);
    generate_piece_moves(pos, ml, KING, us);
    generate_castling(pos, ml, us);
}

void generate_legal(Position* pos, MoveList* ml, UndoInfo* undo_stack, int* undo_stack_ptr) {
    MoveList pseudo;
    generate_pseudo_legal(pos, &pseudo, pos->sideToMove);
    ml->count = 0;
    for (int i = 0; i < pseudo.count; i++) {
        if (is_legal_move(pos, pseudo.moves[i].move))
            ml->moves[ml->count++] = pseudo.moves[i];
    }
}


static inline int get_material_value(Piece pc) {
    return PIECE_VALUES_CP[type_of_p(pc)];
}

static inline int get_pst_value_phase(Piece pc, Square sq, int phase) {
    int pt = type_of_p(pc);
    ChessColor c = color_of(pc);
    Square s = (c == CHESS_WHITE) ? sq : (sq ^ 56);
    
    int mg, eg;
    switch (pt) {
        case PAWN:
            mg = PawnPST_MG[s];
            eg = PawnPST_EG[s];
            break;
        case KNIGHT:
            mg = KnightPST_MG[s];
            eg = KnightPST_EG[s];
            break;
        case BISHOP:
            mg = BishopPST_MG[s];
            eg = BishopPST_EG[s];
            break;
        case ROOK:
            mg = RookPST_MG[s];
            eg = RookPST_EG[s];
            break;
        case QUEEN:
            mg = QueenPST_MG[s];
            eg = QueenPST_EG[s];
            break;
        case KING:
            mg = KingPST_MG[s];
            eg = KingPST_EG[s];
            break;
        default:
            return 0;
    }
    
    return (mg * phase + eg * (24 - phase)) / 24;
}

static inline int get_pst_value(Piece pc, Square sq) {
    return get_pst_value_phase(pc, sq, 12);
}

void do_move(Position* pos, Move m, UndoInfo* undo_stack, int* undo_stack_ptr) {
    if (m == MOVE_NULL) {
        undo_stack[*undo_stack_ptr].captured = NO_PIECE;
        undo_stack[*undo_stack_ptr].castlingRights = pos->castlingRights;
        undo_stack[*undo_stack_ptr].epSquare = pos->epSquare;
        undo_stack[*undo_stack_ptr].rule50 = pos->rule50;
        undo_stack[*undo_stack_ptr].materialScore = pos->materialScore;
        undo_stack[*undo_stack_ptr].psqtScore = pos->psqtScore;
        undo_stack[*undo_stack_ptr].key = pos->key;
        undo_stack[*undo_stack_ptr].pliesFromNull = 0;
        (*undo_stack_ptr)++;
        
        if (pos->epSquare != SQ_NONE) {
            pos->key ^= zob.enpassant[file_of(pos->epSquare)];
            pos->epSquare = SQ_NONE;
        }
        pos->sideToMove = !pos->sideToMove;
        pos->key ^= zob.side;
        return;
    }
    
    Square from = from_sq(m);
    Square to = to_sq(m);
    int move_type = type_of_m(m);
    Piece pc = piece_on(pos, from);
    Piece captured = piece_on(pos, to);
    int pt = type_of_p(pc);
    ChessColor us = pos->sideToMove;
    ChessColor them = !us;
    
    undo_stack[*undo_stack_ptr].captured = captured;
    undo_stack[*undo_stack_ptr].castlingRights = pos->castlingRights;
    undo_stack[*undo_stack_ptr].epSquare = pos->epSquare;
    undo_stack[*undo_stack_ptr].rule50 = pos->rule50;
    undo_stack[*undo_stack_ptr].materialScore = pos->materialScore;
    undo_stack[*undo_stack_ptr].psqtScore = pos->psqtScore;
    undo_stack[*undo_stack_ptr].key = pos->key;
    undo_stack[*undo_stack_ptr].pliesFromNull = (*undo_stack_ptr > 0) ? undo_stack[*undo_stack_ptr - 1].pliesFromNull + 1 : 0;
    (*undo_stack_ptr)++;
    
    int sign = (us == CHESS_WHITE) ? 1 : -1;
    int mat_delta = 0, pst_delta = 0;
    
    pst_delta -= sign * get_pst_value(pc, from);
    
    if (captured != NO_PIECE) {
        int cap_sign = (color_of(captured) == CHESS_WHITE) ? 1 : -1;
        mat_delta -= cap_sign * get_material_value(captured);
        pst_delta -= cap_sign * get_pst_value(captured, to);
    }
    
    if (pt == PAWN || captured != NO_PIECE) {
        pos->rule50 = 0;
        undo_stack[*undo_stack_ptr - 1].pliesFromNull = 0;
    }
    else {
        pos->rule50++;
    }
    
    if (pos->epSquare != SQ_NONE) {
        pos->key ^= zob.enpassant[file_of(pos->epSquare)];
    }
    pos->epSquare = SQ_NONE;
    
    if (move_type == CASTLING) {
        pos->key ^= zob.psq[pc][from];
        
        pos->board[from] = NO_PIECE;
        pos->board[to] = pc;
        pos->byTypeBB[pt] ^= sq_bb(from) ^ sq_bb(to);
        pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
        pos->byTypeBB[0] ^= sq_bb(from) ^ sq_bb(to);
        pst_delta += sign * get_pst_value(pc, to);
        pos->key ^= zob.psq[pc][to];
        
        Square rook_from, rook_to;
        if (to > from) { 
            rook_from = from + 3;
            rook_to = from + 1;
        } else {
            rook_from = from - 4;
            rook_to = from - 1;
        }
        
        Piece rook = piece_on(pos, rook_from);
        pos->key ^= zob.psq[rook][rook_from];
        pos->board[rook_from] = NO_PIECE;
        pos->board[rook_to] = rook;
        pos->byTypeBB[ROOK] ^= sq_bb(rook_from) ^ sq_bb(rook_to);
        pos->byColorBB[us] ^= sq_bb(rook_from) ^ sq_bb(rook_to);
        pos->byTypeBB[0] ^= sq_bb(rook_from) ^ sq_bb(rook_to);
        pst_delta -= sign * get_pst_value(rook, rook_from);
        pst_delta += sign * get_pst_value(rook, rook_to);
        pos->key ^= zob.psq[rook][rook_to]; 
        
    } else if (move_type == ENPASSANT) {
        pos->key ^= zob.psq[pc][from];
        
        pos->board[from] = NO_PIECE;
        pos->board[to] = pc;
        pos->byTypeBB[pt] ^= sq_bb(from) ^ sq_bb(to);
        pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
        pos->byTypeBB[0] ^= sq_bb(from) ^ sq_bb(to);
        pst_delta += sign * get_pst_value(pc, to); 
        pos->key ^= zob.psq[pc][to];
        
        Square cap_sq = to + (us == CHESS_WHITE ? SOUTH : NORTH);
        Piece cap_pawn = piece_on(pos, cap_sq);
        pos->key ^= zob.psq[cap_pawn][cap_sq];
        pos->board[cap_sq] = NO_PIECE;
        pos->byTypeBB[PAWN] ^= sq_bb(cap_sq);
        pos->byColorBB[them] ^= sq_bb(cap_sq);
        pos->byTypeBB[0] ^= sq_bb(cap_sq);
        pos->pieceCount[cap_pawn]--;
        int cap_sign = (color_of(cap_pawn) == CHESS_WHITE) ? 1 : -1;
        mat_delta -= cap_sign * get_material_value(cap_pawn);
        pst_delta -= cap_sign * get_pst_value(cap_pawn, cap_sq);
        
    } else {
        pos->key ^= zob.psq[pc][from];
        
        if (captured != NO_PIECE) {
            pos->key ^= zob.psq[captured][to];
            int cap_pt = type_of_p(captured);
            pos->byTypeBB[cap_pt] ^= sq_bb(to);
            pos->byColorBB[them] ^= sq_bb(to);
            pos->byTypeBB[0] ^= sq_bb(to);
            pos->pieceCount[captured]--;
        }
        
		pos->board[from] = NO_PIECE;
		pos->board[to] = pc;
		pos->byTypeBB[pt] ^= sq_bb(from) ^ sq_bb(to);
		pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
		pos->byTypeBB[0] ^= sq_bb(from) ^ sq_bb(to);
        pst_delta += sign * get_pst_value(pc, to);
        pos->key ^= zob.psq[pc][to];
        
        if (move_type == PROMOTION) {
            int promo_pt = promotion_type(m);
            Piece promo_pc = make_piece(us, promo_pt);
            pos->key ^= zob.psq[pc][to];
            pos->board[to] = promo_pc;
            pos->byTypeBB[pt] ^= sq_bb(to);
            pos->byTypeBB[promo_pt] ^= sq_bb(to);
            pos->pieceCount[pc]--;
            pos->pieceCount[promo_pc]++;
            pos->key ^= zob.psq[promo_pc][to];
            mat_delta -= sign * get_material_value(pc);
            mat_delta += sign * get_material_value(promo_pc);
            pst_delta -= sign * get_pst_value(pc, to);
            pst_delta += sign * get_pst_value(promo_pc, to);
        }
        
        if (pt == PAWN) {
            int diff = to - from;
            if (diff == 16 || diff == -16) {
                Square ep_sq = (from + to) / 2;
                if (pawn_attacks_bb(us, ep_sq) & pieces_cp(pos, them, PAWN)) {
                    pos->epSquare = ep_sq;
                    pos->key ^= zob.enpassant[file_of(ep_sq)];
                }
            }
        }
    }
    
    uint8_t old_castling = pos->castlingRights;
    if (pt == KING) {
        pos->castlingRights &= us == CHESS_WHITE ? ~WHITE_CASTLING : ~BLACK_CASTLING;
    }
    if (from == SQ_A1 || to == SQ_A1) pos->castlingRights &= ~WHITE_OOO;
    if (from == SQ_H1 || to == SQ_H1) pos->castlingRights &= ~WHITE_OO;
    if (from == SQ_A8 || to == SQ_A8) pos->castlingRights &= ~BLACK_OOO;
    if (from == SQ_H8 || to == SQ_H8) pos->castlingRights &= ~BLACK_OO;
    
    if (old_castling != pos->castlingRights) {
        pos->key ^= zob.castling[old_castling];
        pos->key ^= zob.castling[pos->castlingRights];
    }
    
    pos->materialScore += mat_delta;
    pos->psqtScore += pst_delta;
    
    pos->sideToMove = them;
    pos->key ^= zob.side;
    pos->gamePly++;
}

void undo_move(Position* pos, Move m, UndoInfo* undo_stack, int* undo_stack_ptr) {
    (*undo_stack_ptr)--;
    UndoInfo* undo = &undo_stack[*undo_stack_ptr];
    
    if (m == MOVE_NULL) {
        pos->castlingRights = undo->castlingRights;
        pos->epSquare = undo->epSquare;
        pos->rule50 = undo->rule50;
        pos->materialScore = undo->materialScore;
        pos->psqtScore = undo->psqtScore;
        pos->key = undo->key;
        pos->sideToMove = !pos->sideToMove;
        return;
    }
    
    Square from = from_sq(m);
    Square to = to_sq(m);
    int move_type = type_of_m(m);
    ChessColor us = !pos->sideToMove;
    ChessColor them = pos->sideToMove;
    
    Piece pc = piece_on(pos, to);
    int pt = type_of_p(pc);
    
    pos->castlingRights = undo->castlingRights;
    pos->epSquare = undo->epSquare;
    pos->rule50 = undo->rule50;
    pos->materialScore = undo->materialScore;
    pos->psqtScore = undo->psqtScore;
    pos->key = undo->key;
    pos->sideToMove = us;
    pos->gamePly--;
    
    if (move_type == CASTLING) {
        pos->board[to] = NO_PIECE;
        pos->board[from] = pc;
        pos->byTypeBB[pt] ^= sq_bb(from) ^ sq_bb(to);
        pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
        pos->byTypeBB[0] ^= sq_bb(from) ^ sq_bb(to);
        
        Square rook_from, rook_to;
        if (to > from) {
            rook_from = from + 3;
            rook_to = from + 1;
        } else {
            rook_from = from - 4;
            rook_to = from - 1;
        }
        
        Piece rook = piece_on(pos, rook_to);
        pos->board[rook_to] = NO_PIECE;
        pos->board[rook_from] = rook;
        pos->byTypeBB[ROOK] ^= sq_bb(rook_from) ^ sq_bb(rook_to);
        pos->byColorBB[us] ^= sq_bb(rook_from) ^ sq_bb(rook_to);
        pos->byTypeBB[0] ^= sq_bb(rook_from) ^ sq_bb(rook_to);
        
    } else if (move_type == ENPASSANT) {
        pos->board[to] = NO_PIECE;
        pos->board[from] = pc;
        pos->byTypeBB[pt] ^= sq_bb(from) ^ sq_bb(to);
        pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
        pos->byTypeBB[0] ^= sq_bb(from) ^ sq_bb(to);

        
        Square cap_sq = to + (us == CHESS_WHITE ? SOUTH : NORTH);
        Piece cap_pawn = make_piece(them, PAWN);
        pos->board[cap_sq] = cap_pawn;
        pos->byTypeBB[PAWN] ^= sq_bb(cap_sq);
        pos->byColorBB[them] ^= sq_bb(cap_sq);
        pos->byTypeBB[0] ^= sq_bb(cap_sq);
        pos->pieceCount[cap_pawn]++;
        
    } else {
        if (move_type == PROMOTION) {
            int promo_pt = promotion_type(m);
            Piece promo_pc = make_piece(us, promo_pt);
            pc = make_piece(us, PAWN);
            pt = PAWN;
            pos->board[to] = NO_PIECE;
            pos->byTypeBB[promo_pt] ^= sq_bb(to);
            pos->byTypeBB[pt] ^= sq_bb(to);
            pos->pieceCount[promo_pc]--;
            pos->pieceCount[pc]++;
        }
        
        pos->board[to] = undo->captured;
        pos->board[from] = pc;
        pos->byTypeBB[pt] ^= sq_bb(from) ^ sq_bb(to);
        pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
        
        if (undo->captured != NO_PIECE) {
            int cap_pt = type_of_p(undo->captured);
            pos->byTypeBB[cap_pt] ^= sq_bb(to);
            pos->byColorBB[them] ^= sq_bb(to);
            pos->byTypeBB[0] ^= sq_bb(from);
            pos->pieceCount[undo->captured]++;
        } else {
            pos->byTypeBB[0] ^= sq_bb(from) ^ sq_bb(to);
        }
    }
}

static inline bool is_insufficient_material(const Position* pos) {
    if (pieces_p(pos, PAWN) | pieces_p(pos, ROOK) | pieces_p(pos, QUEEN))
        return false;

    int wN = popcount(pieces_cp(pos, CHESS_WHITE, KNIGHT));
    int bN = popcount(pieces_cp(pos, CHESS_BLACK, KNIGHT));
    int wB = popcount(pieces_cp(pos, CHESS_WHITE, BISHOP));
    int bB = popcount(pieces_cp(pos, CHESS_BLACK, BISHOP));
    int totalMinors = wN + bN + wB + bB;

    if (totalMinors == 0)
        return true;

    if (totalMinors == 1)
        return true;

    if (totalMinors == 2) {
        if ((wN == 2 && wB == 0 && bN == 0 && bB == 0) || (bN == 2 && bB == 0 && wN == 0 && wB == 0))
            return true;
        if ((wN + wB) == 1 && (bN + bB) == 1)
            return true;
    }

    return false;
}

bool is_draw_with_history(Position* pos, UndoInfo* undo_stack, int undo_stack_ptr) {
    if (pos->rule50 >= 100)
        return true;
    
    if (is_insufficient_material(pos))
        return true;

    // Backward scan for repetitions
    int e = (undo_stack_ptr > 0) ? undo_stack[undo_stack_ptr - 1].pliesFromNull - 4 : -1;
    if (e >= 0) {
        int repetitions = 0;
        for (int i = 4; i <= e + 4; i += 2) {
            int idx = undo_stack_ptr - 1 - i;
            if (idx >= 0 && undo_stack[idx].key == pos->key) {
                repetitions++;
                if (repetitions >= 2) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

int game_result_with_legal_count(Position* pos, int legal_count, UndoInfo* undo_stack, int undo_stack_ptr, 
                                  int enable_50_move_rule, int enable_threefold_repetition) {
    if (legal_count == 0) {
        if (is_check(pos, pos->sideToMove)) {
            return pos->sideToMove == CHESS_WHITE ? 1 : 2;
        } else {
            return 3;
        }
    }
    
    if (is_insufficient_material(pos)) {
        return 3;
    }
    
    if (enable_50_move_rule && pos->rule50 >= 100) {
        return 3;
    }
    
    if (enable_threefold_repetition && undo_stack_ptr > 0) {
        uint8_t plies = undo_stack[undo_stack_ptr - 1].pliesFromNull;
        
        if (plies >= 4) {
            int repetitions = 0;
            // Search backward: only check positions with same side to move (step by 2 plies)
            for (int i = 4; i <= plies; i += 2) {
                int idx = undo_stack_ptr - 1 - i;
                if (idx >= 0 && undo_stack[idx].key == pos->key) {
                    repetitions++;
                    if (repetitions >= 2) {
                        return 3;
                    }
                }
            }
        }
    }
    
    return 0;
}

uint64_t perft(Position* pos, int depth) {
    if (depth == 0)
        return 1ULL;
    
    UndoInfo local_undo[512];
    int local_ptr = 0;
    
    MoveList ml;
    generate_legal(pos, &ml, local_undo, &local_ptr);
    
    uint64_t nodes = 0;
    for (int i = 0; i < ml.count; i++) {
        Move m = ml.moves[i].move;
        do_move(pos, m, local_undo, &local_ptr);
        nodes += perft(pos, depth - 1);
        undo_move(pos, m, local_undo, &local_ptr);
    }
    
    return nodes;
}


void populate_observations(Chess* env) {
    uint8_t* obs = env->observations;
    Position* pos = &env->pos;
    
    int num_players = env->selfplay ? 2 : 1;
    for (int player_iter = 0; player_iter < num_players; player_iter++) {
        int player = env->selfplay ? player_iter : env->learner_color;
        int buffer_idx;
        if (env->selfplay) {
            buffer_idx = (env->learner_color == CHESS_WHITE) ? player_iter : (1 - player_iter);
        } else {
            buffer_idx = 0;
        }
        
        uint8_t* player_obs = obs + (buffer_idx * OBS_SIZE);
        uint8_t* board_planes = player_obs + O_BOARD;
        memset(board_planes, 0, 12 * 64);
        
        ChessColor us = (ChessColor)player;  // 0=White, 1=Black
        
        Bitboard occupied = pos->byTypeBB[0];
        while (occupied) {
            Square sq = pop_lsb(&occupied);
        Piece p = pos->board[sq];
        
        int plane;
            int obs_sq;
            
            if (player == 1) {
                obs_sq = sq ^ 56;
                int piece_color = color_of(p);
                int piece_type = type_of_p(p);
                if (piece_color == CHESS_BLACK) {
                    plane = piece_type - 1;
                } else {
                    plane = 6 + (piece_type - 1);
                }
            } else {
                obs_sq = sq;
        if (p >= B_PAWN) {
            plane = 6 + (p - B_PAWN);
        } else {
            plane = p - 1;
        }
            }
            board_planes[plane * 64 + obs_sq] = 1;
        }
        
        uint8_t* side_onehot = player_obs + O_SIDE;
        side_onehot[0] = 0; side_onehot[1] = 0;
        side_onehot[(pos->sideToMove == us) ? 0 : 1] = 1;
        
        uint8_t* castle_onehot = player_obs + O_CASTLE;
        memset(castle_onehot, 0, 16);
        uint8_t castle_rights = pos->castlingRights;
        if (player == 1) {
            uint8_t flipped = 0;
            if (castle_rights & BLACK_OO) flipped |= WHITE_OO;
            if (castle_rights & BLACK_OOO) flipped |= WHITE_OOO;
            if (castle_rights & WHITE_OO) flipped |= BLACK_OO;
            if (castle_rights & WHITE_OOO) flipped |= BLACK_OOO;
            castle_rights = flipped;
        }
        castle_onehot[castle_rights] = 1;

        uint8_t* ep_onehot = player_obs + O_EP;
        memset(ep_onehot, 0, 65);
    if (pos->epSquare < 64) {
            int ep_sq = (player == 1) ? (pos->epSquare ^ 56) : pos->epSquare;
            ep_onehot[ep_sq] = 1;
    } else {
        ep_onehot[64] = 1;
    }
    
        uint8_t* valid_pieces = player_obs + O_VALID_PIECES;
        uint8_t* valid_dests = player_obs + O_VALID_DESTS;
        memset(valid_pieces, 0, 64);
        memset(valid_dests, 0, 64);
        
        int player_idx = (int)us;
        
        if (pos->sideToMove != us) {
            env->pick_phase[player_idx] = 0;
            env->selected_square[player_idx] = SQ_NONE;
            env->valid_destinations[player_idx].count = 0;
            
            MoveList next_player_legal;
            Position temp_pos = *pos;
            temp_pos.sideToMove = !temp_pos.sideToMove;
            temp_pos.key ^= zob.side;
            UndoInfo local_undo[64];
            int local_undo_ptr = 0;
            generate_legal(&temp_pos, &next_player_legal, local_undo, &local_undo_ptr);
            
            if (next_player_legal.count > 0) {
                memset(valid_pieces, 0, 64);
                for (int i = 0; i < next_player_legal.count; i++) {
                    Square from = from_sq(next_player_legal.moves[i].move);
                    Square obs_sq = (us == CHESS_BLACK) ? (from ^ 56) : from;
                    valid_pieces[obs_sq] = 1;
                }
            } else {
                memset(valid_pieces, 1, 64);
            }
            memset(valid_dests, 0, 64);
        } else {
            if (env->pick_phase[player_idx] == 0) {
                if (env->legal_moves.count > 0) {
                    for (int i = 0; i < env->legal_moves.count; i++) {
                        Square from = from_sq(env->legal_moves.moves[i].move);
                        int view_from = (player == 1) ? (from ^ 56) : from;
                        valid_pieces[view_from] = 1;
                    }
                } else {
                    memset(valid_pieces, 1, 64);
                }
            } else {
                if (env->valid_destinations[player_idx].count > 0) {
                    for (int i = 0; i < env->valid_destinations[player_idx].count; i++) {
                        Square to = to_sq(env->valid_destinations[player_idx].moves[i].move);
                        int view_to = (player == 1) ? (to ^ 56) : to;
                        valid_dests[view_to] = 1;
                    }
                }
            }
        }
        
        uint8_t* phase_onehot = player_obs + O_PICK_PHASE;
        phase_onehot[0] = 0; phase_onehot[1] = 0;
        phase_onehot[env->pick_phase[player_idx]] = 1;
        
        uint8_t* selected_piece_plane = player_obs + O_SELECTED_PIECE;
        memset(selected_piece_plane, 0, 64);
        if (env->pick_phase[player_idx] == 1 && env->selected_square[player_idx] != SQ_NONE) {
            int view_selected = (player == 1) ? (env->selected_square[player_idx] ^ 56) : env->selected_square[player_idx];
            selected_piece_plane[view_selected] = 1;
        }
        
        // Mask for valid promotion pieces (actions 64-95: 4 rows by 8 files)
        uint8_t* valid_promos = player_obs + O_VALID_PROMOS;
        memset(valid_promos, 0, 32);
        
        if (env->pick_phase[player_idx] == 1 && env->valid_destinations[player_idx].count > 0) {
            for (int i = 0; i < env->valid_destinations[player_idx].count; i++) {
                Move m = env->valid_destinations[player_idx].moves[i].move;
                if (type_of_m(m) == PROMOTION) {
                    int type_idx = QUEEN - promotion_type(m);
                    int file_idx = file_of(to_sq(m));
                    valid_promos[type_idx * 8 + file_idx] = 1;
                }
            }
        }
    }
}

void generate_random_fen(char* fen_out) {
    char board[64];
    memset(board, '.', 64);
    
    int wk_sq, bk_sq;
    do {
        wk_sq = rand() % 64;
        bk_sq = rand() % 64;
        int wk_rank = wk_sq / 8, wk_file = wk_sq % 8;
        int bk_rank = bk_sq / 8, bk_file = bk_sq % 8;
        int rank_diff = abs(wk_rank - bk_rank);
        int file_diff = abs(wk_file - bk_file);
        if (wk_sq != bk_sq && (rank_diff > 1 || file_diff > 1)) break;
    } while (1);
    
    board[wk_sq] = 'K';
    board[bk_sq] = 'k';
    
    const char* white_pieces = "QRRNNBBPP";
    const char* black_pieces = "qrrnnbbpp";
    int num_white = rand() % 16;
    int num_black = rand() % 16;
    
    for (int i = 0; i < num_white; i++) {
        int sq, rank;
        char piece;
        do {
            sq = rand() % 64;
            rank = sq / 8;
            piece = white_pieces[rand() % 9];
        } while (board[sq] != '.' || (piece == 'P' && (rank == 0 || rank == 7)));
        board[sq] = piece;
    }
    
    for (int i = 0; i < num_black; i++) {
        int sq, rank;
        char piece;
        do {
            sq = rand() % 64;
            rank = sq / 8;
            piece = black_pieces[rand() % 9];
        } while (board[sq] != '.' || (piece == 'p' && (rank == 0 || rank == 7)));
        board[sq] = piece;
    }
    
    char* ptr = fen_out;
    for (int rank = 7; rank >= 0; rank--) {
        int empty = 0;
        for (int file = 0; file < 8; file++) {
            char piece = board[rank * 8 + file];
            if (piece == '.') {
                empty++;
            } else {
                if (empty > 0) {
                    *ptr++ = '0' + empty;
                    empty = 0;
                }
                *ptr++ = piece;
            }
        }
        if (empty > 0) *ptr++ = '0' + empty;
        if (rank > 0) *ptr++ = '/';
    }
    strcpy(ptr, " w - - 0 1");
}

void c_reset(Chess* env) {
    env->tick = 0;
    env->chess_moves = 0;
    env->game_result = 0;
    env->undo_stack_ptr = 0;
    env->invalid_actions_this_episode = 0;
    
    env->pick_phase[0] = 0;
    env->pick_phase[1] = 0;
    env->selected_square[0] = SQ_NONE;
    env->selected_square[1] = SQ_NONE;
    env->valid_destinations[0].count = 0;
    env->valid_destinations[1].count = 0;
    env->learner_color = 1 - env->learner_color;
    
    // Select starting position or curriculum
    if (env->num_fens > 0) {
        int fen_idx = rand() % env->num_fens;
        pos_set(&env->pos, env->fen_curriculum[fen_idx]);
    } else if (strcmp(env->starting_fen, "random") == 0) {
        char random_fen[128];
        generate_random_fen(random_fen);
        pos_set(&env->pos, random_fen);
    } else {
        pos_set(&env->pos, env->starting_fen);
    }
    
    generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
    env->legal_moves_side = env->pos.sideToMove;
    env->legal_moves_key = env->pos.key;
    populate_observations(env);
}

bool process_player_action(Chess* env, int action, ChessColor player) {
    if (env->pos.sideToMove != player) {
        return false;
    }
    
    if (action < 0) action = 0;
    if (action >= 96) action = 95;
    
    // Actions 64-95 are promotion selections (4 rows by 8 files)
    bool is_promotion_selection = (action >= 64 && action <= 95);
    Square picked_sq = SQ_NONE;
    
    if (!is_promotion_selection) {
        picked_sq = (Square)action;
        if (player == CHESS_BLACK) {
            picked_sq = action ^ 56;
        }
    }
    
    int pidx = (int)player;
    
    if (env->legal_moves_side != env->pos.sideToMove || 
        env->legal_moves.count == 0 || 
        env->legal_moves_key != env->pos.key) {
        generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
        env->legal_moves_side = env->pos.sideToMove;
        env->legal_moves_key = env->pos.key;
    }
    
    if (env->legal_moves.count == 0) {
        return false;
    }
    
    if (env->pick_phase[pidx] == 0) {
        if (picked_sq >= 64) {
            if (player == env->learner_color) {
                env->rewards[0] += env->reward_invalid_piece;
                env->invalid_actions_this_episode++;
            }
            return false;
        }

        Piece pc = piece_on(&env->pos, picked_sq);
        
        if (pc != NO_PIECE && color_of(pc) == player) {
            env->valid_destinations[pidx].count = 0;
            for (int i = 0; i < env->legal_moves.count; i++) {
                Move m = env->legal_moves.moves[i].move;
                if (from_sq(m) == picked_sq) {
                    env->valid_destinations[pidx].moves[env->valid_destinations[pidx].count++] = env->legal_moves.moves[i];
                }
            }
            
            if (env->valid_destinations[pidx].count > 0) {
                env->selected_square[pidx] = picked_sq;
                env->pick_phase[pidx] = 1;
                if (player == env->learner_color) env->rewards[0] += env->reward_valid_piece;
            } else {
                if (player == env->learner_color) {
                    env->rewards[0] += env->reward_invalid_piece;
                    env->invalid_actions_this_episode++;
                }
            }
        } else {
            if (player == env->learner_color) {
                env->rewards[0] += env->reward_invalid_piece;
                env->invalid_actions_this_episode++;
            }
        }
        return false;
    }
    
    Move chosen_move = MOVE_NONE;
    Square selected_sq = env->selected_square[pidx];
    
    // Handle promotion piece selection (actions 64-95)
    if (is_promotion_selection) {
        int promo_idx = action - 64;
        int promo_row = promo_idx / 8; // 0-3
        int promo_file = promo_idx % 8; // 0-7
        int desired_promo = QUEEN - promo_row;
        
        for (int i = 0; i < env->valid_destinations[pidx].count; i++) {
            Move m = env->valid_destinations[pidx].moves[i].move;
            if ((int)type_of_m(m) == PROMOTION && 
                promotion_type(m) == desired_promo &&
                file_of(to_sq(m)) == promo_file) {
                
                chosen_move = m;
                break;
            }
        }
    } else {
        for (int i = 0; i < env->valid_destinations[pidx].count; i++) {
            if (to_sq(env->valid_destinations[pidx].moves[i].move) == picked_sq) {
                chosen_move = env->valid_destinations[pidx].moves[i].move;
                break;
            }
        }
    }
    
    if (chosen_move == MOVE_NONE && selected_sq != SQ_NONE) {
        for (int i = 0; i < env->legal_moves.count; i++) {
            Move m = env->legal_moves.moves[i].move;
            if (from_sq(m) == selected_sq && to_sq(m) == picked_sq) {
                chosen_move = m;
                env->valid_destinations[pidx].count = 0;
                for (int j = 0; j < env->legal_moves.count; j++) {
                    if (from_sq(env->legal_moves.moves[j].move) == selected_sq) {
                        env->valid_destinations[pidx].moves[env->valid_destinations[pidx].count++] = env->legal_moves.moves[j];
                    }
                }
                break;
            }
        }
    }
    
    if (chosen_move == MOVE_NONE) {
        if (player == env->learner_color) {
            env->rewards[0] += env->reward_invalid_move;
            env->invalid_actions_this_episode++;
        }
        env->pick_phase[pidx] = 0;
        env->selected_square[pidx] = SQ_NONE;
        env->valid_destinations[pidx].count = 0;
        return false;
    }
    
    if (player == env->learner_color) env->rewards[0] += env->reward_valid_move;
    env->chess_moves++;
    env->pick_phase[pidx] = 0;
    env->selected_square[pidx] = SQ_NONE;
    env->valid_destinations[pidx].count = 0;

    
    if (env->reward_castling != 0.0f && player == env->learner_color && (int)type_of_m(chosen_move) == CASTLING) {
        env->rewards[0] += env->reward_castling;
    }
    
    do_move(&env->pos, chosen_move, env->undo_stack, &env->undo_stack_ptr);
    
    if (env->undo_stack_ptr > 0 && env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull > 99) {
        env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull = 99;
    }
    
    if (env->reward_repetition != 0.0f && player == env->learner_color && env->undo_stack_ptr > 0) {
        uint8_t plies = env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull;
        if (plies >= 4) {
            Key current_key = env->pos.key;
            for (int i = 4; i <= plies; i += 2) {
                int idx = env->undo_stack_ptr - 1 - i;
                if (idx >= 0 && env->undo_stack[idx].key == current_key) {
                    env->rewards[0] += env->reward_repetition;
                    break;
                }
            }
        }
    }
    
    return true;
}

void clip_rewards(Chess* env) {
    if (env->rewards[0] > 1.0f) {
        env->rewards[0] = 1.0f;
    }
    if (env->rewards[0] < -1.0f) {
        env->rewards[0] = -1.0f;
    }
}

void human_play(Chess* env) {
    if (!env->human_play) {
        return;
    }
    
    if (env->selfplay && env->human_play) {
        fprintf(stderr, "FATAL: selfplay=1 AND human_play=1 is invalid configuration\n");
        exit(1);
    }
    

    if (env->human_color == -1) {
        env->actions[0] = -1;
        return;
    }
    
    ChessColor current_side = env->pos.sideToMove;
    if (current_side == env->human_color) {
        env->actions[0] = -1;
    }
}

void c_step(Chess* env) {
    if (!env->selfplay && !env->human_play) {
        fprintf(stderr, "FATAL: selfplay=0 AND human_play=0 is invalid configuration\n");
        exit(1);
    }
    
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->tick++;
    
    int action;
    if (env->human_play) {
        human_play(env);
    }
    if (env->selfplay) {
        ChessColor current_side = env->pos.sideToMove;
        action = (current_side == env->learner_color) ? env->actions[0] : env->actions[1];
    } else {
        action = env->actions[0];
    }
    
    if (action == -1) {
        if (env->legal_moves_side != env->pos.sideToMove || env->legal_moves_key != env->pos.key) {
            generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
            env->legal_moves_side = env->pos.sideToMove;
            env->legal_moves_key = env->pos.key;
        }
        populate_observations(env);
        return;
    }

    bool use_dense_rewards = (env->reward_material != 0.0f || env->reward_position != 0.0f);
    int16_t mat_before = 0, pst_before = 0;
    if (use_dense_rewards) {
        mat_before = env->pos.materialScore;
        pst_before = env->pos.psqtScore;
    }
    
    ChessColor mover = env->pos.sideToMove;
    
    bool move_completed = process_player_action(env, action, mover);
    if (move_completed && use_dense_rewards) {
        int16_t mat_after = env->pos.materialScore;
        int16_t pst_after = env->pos.psqtScore;
        int mat_delta = mat_after - mat_before;
        int pst_delta = pst_after - pst_before;
        float mat_reward = (float)mat_delta / 100.0f * env->reward_material;
        float pos_reward = (float)pst_delta / 50.0f * env->reward_position;
        if (env->learner_color == CHESS_BLACK) {
            mat_reward = -mat_reward;
            pos_reward = -pos_reward;
        }
        
        env->rewards[0] += mat_reward + pos_reward;
        clip_rewards(env);
    }
    
    if (move_completed) {
        if (env->legal_moves_side != env->pos.sideToMove || env->legal_moves_key != env->pos.key) {
            generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
            env->legal_moves_side = env->pos.sideToMove;
            env->legal_moves_key = env->pos.key;
        }
        populate_observations(env);
    }
    
    if (env->chess_moves >= env->max_moves || env->undo_stack_ptr >= MAX_GAME_PLIES - 2) {
        env->terminals[0] = 1;
        env->rewards[0] = env->reward_draw;
        env->log.perf = (env->log.perf * env->log.n + 0.5f) / (env->log.n + 1.0f);
        env->log.draw_rate += 1.0f;
        env->log.timeout_rate += 1.0f;
        env->log.chess_moves += env->chess_moves;
        env->log.episode_length += env->tick;
        float invalid_rate = (env->tick > 0) ? ((float)env->invalid_actions_this_episode / (float)env->tick) : 0.0f;
        env->log.invalid_action_rate += invalid_rate;
        float length_score = fminf(1.0f, (float)env->chess_moves / 40.0f);
        env->log.game_length_score = (env->log.game_length_score * env->log.n + length_score) / (env->log.n + 1.0f);
        float avg_draw_rate = (env->log.n > 0) ? (env->log.draw_rate / env->log.n) : 0.0f;
        env->log.score = env->log.perf + 0.2f * env->log.game_length_score - 0.1f * avg_draw_rate;
        
        env->log.n += 1.0f;
        c_reset(env);
        return;
    }
    
    if (env->legal_moves_side != env->pos.sideToMove || env->legal_moves_key != env->pos.key) {
        generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
        env->legal_moves_side = env->pos.sideToMove;
        env->legal_moves_key = env->pos.key;
    }
    env->game_result = game_result_with_legal_count(&env->pos, env->legal_moves.count, env->undo_stack, env->undo_stack_ptr,
                                                     env->enable_50_move_rule, env->enable_threefold_repetition);
    
    if (env->game_result != 0) {
        env->terminals[0] = 1;
        float win_value = 0.0f;
        if (env->game_result == 3) {
            env->rewards[0] = env->reward_draw;
            win_value = 0.5f;
            env->log.draw_rate += 1.0f;
            
            env->white_score += 0.5f;
            env->black_score += 0.5f;
            env->learner_draws += 1.0f;
            strcpy(env->last_result, "Draw");
        } else if (env->game_result == 1) {
            if (env->learner_color == CHESS_WHITE) {
                env->rewards[0] = -1.0f;
                win_value = 0.0f;
            } else {
                env->rewards[0] = 1.0f;
                win_value = 1.0f;
            }
            env->black_score += 1.0f;
            if (env->learner_color == CHESS_BLACK) {
                env->learner_wins += 1.0f;
            } else {
                env->learner_losses += 1.0f;
            }
            strcpy(env->last_result, "Black Wins");
            env->log.draw_rate += 0.0f;
        } else if (env->game_result == 2) {
            if (env->learner_color == CHESS_WHITE) {
                env->rewards[0] = 1.0f;
                win_value = 1.0f;
            } else {
                env->rewards[0] = -1.0f;
                win_value = 0.0f;
            }
            env->white_score += 1.0f;
            if (env->learner_color == CHESS_WHITE) {
                env->learner_wins += 1.0f;
            } else {
                env->learner_losses += 1.0f;
            }
            strcpy(env->last_result, "White Wins");
            env->log.draw_rate += 0.0f;
        }
        
        env->log.perf = (env->log.perf * env->log.n + win_value) / (env->log.n + 1.0f);
        env->log.timeout_rate += 0.0f;
        env->log.chess_moves += env->chess_moves;
        env->log.episode_length += env->tick;
        float invalid_rate = (env->tick > 0) ? ((float)env->invalid_actions_this_episode / (float)env->tick) : 0.0f;
        env->log.invalid_action_rate += invalid_rate;
        
        float length_score = fminf(1.0f, (float)env->chess_moves / 40.0f);
        env->log.game_length_score = (env->log.game_length_score * env->log.n + length_score) / (env->log.n + 1.0f);
        
        float avg_draw_rate = (env->log.n > 0) ? (env->log.draw_rate / env->log.n) : 0.0f;
        env->log.score = env->log.perf + 0.2f * env->log.game_length_score - 0.1f * avg_draw_rate;
        
        env->log.n += 1.0f;
        c_reset(env);
        return;
    }
    
    populate_observations(env);
}

static Font load_piece_font(int cell_size, int* loaded) {
    const char* candidates[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Apple Symbols.ttf",
        "C:\\Windows\\Fonts\\seguisym.ttf"
    };

    int codepoints[] = {0x2654, 0x2655, 0x2656, 0x2657, 0x2658, 0x2659, 0x265A, 0x265B, 0x265C, 0x265D, 0x265E, 0x265F};
    Font font = {0};
    size_t candidate_count = sizeof(candidates) / sizeof(candidates[0]);
    size_t codepoint_count = sizeof(codepoints) / sizeof(codepoints[0]);

    for (size_t i = 0; i < candidate_count; i++) {
        if (!FileExists(candidates[i])) {
            continue;
        }
        font = LoadFontEx(candidates[i], cell_size, codepoints, (int)codepoint_count);
        if (font.texture.id != 0) {
            if (loaded) {
                *loaded = 1;
            }
            SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);
            return font;
        }
    }

    if (loaded) {
        *loaded = 0;
    }
    return GetFontDefault();
}

static void draw_piece(Chess* env, Piece pc, int file, int rank, int cell_size) {
    if (pc == NO_PIECE) {
        return;
    }
    
    Color pc_color = color_of(pc) == CHESS_WHITE 
        ? (Color){255, 255, 255, 255}
        : (Color){0, 0, 0, 255};

    int draw_x = file * cell_size;
    int draw_y = (7 - rank) * cell_size;

    if (env->client && env->client->use_unicode_pieces) {
        float icon_size = cell_size * 0.85f;
        Vector2 pos = {
            draw_x + (cell_size - icon_size) / 2.0f,
            draw_y + (cell_size - icon_size) / 2.0f - cell_size * 0.05f
        };
        DrawTextEx(env->client->piece_font, PIECE_UNICODE[pc], pos, icon_size, 0, pc_color);
    } else {
        int x = draw_x + cell_size / 4;
        int y = draw_y + cell_size / 8;
        DrawText(PIECE_CHARS[pc], x, y, cell_size / 2, pc_color);
    }
}

// GUI is a tad scuffed, but it works.
void c_render(Chess* env) {
    const int cell_size = 64;
    const int board_size = 8 * cell_size;
    
    if (env->client == NULL) {
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        InitWindow(board_size, board_size + 80, "PufferLib Chess - AI vs Opponent");
        SetTargetFPS(env->render_fps > 0 ? env->render_fps : 30);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->cell_size = cell_size;
        int font_loaded = 0;
        env->client->piece_font = load_piece_font(cell_size, &font_loaded);
        env->client->use_unicode_pieces = font_loaded;
        
        env->white_score = 0.0f;
        env->black_score = 0.0f;
        env->learner_wins = 0.0f;
        env->learner_losses = 0.0f;
        env->learner_draws = 0.0f;
        strcpy(env->last_result, "Game starting...");
    }
    
    if (IsKeyDown(KEY_ESCAPE) || WindowShouldClose()) {
        CloseWindow();
        exit(0);
    }
    

    if (env->human_play && env->human_color == -1) {
        BeginDrawing();
        ClearBackground((Color){40, 40, 40, 255});
        
        DrawText("Choose Your Color", 256 - 100, 200, 24, WHITE);
        
        int button_width = 120;
        int button_height = 40;
        int center_x = 256;
        int white_y = 280;
        int black_y = 340;
        
        Rectangle white_button = {center_x - button_width/2, white_y, button_width, button_height};
        Rectangle black_button = {center_x - button_width/2, black_y, button_width, button_height};
        
        DrawRectangleRec(white_button, LIGHTGRAY);
        DrawRectangleLinesEx(white_button, 2, BLACK);
        DrawText("Play White", center_x - 45, white_y + 12, 18, BLACK);
        
        DrawRectangleRec(black_button, GRAY);
        DrawRectangleLinesEx(black_button, 2, BLACK);
        DrawText("Play Black", center_x - 45, black_y + 12, 18, WHITE);
        
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            Vector2 mousePos = GetMousePosition();
            if (CheckCollisionPointRec(mousePos, white_button)) {
                env->human_color = CHESS_WHITE;
                env->learner_color = CHESS_BLACK;
            } else if (CheckCollisionPointRec(mousePos, black_button)) {
                env->human_color = CHESS_BLACK;
                env->learner_color = CHESS_WHITE;
            }
        }
        
        EndDrawing();
        return;
    }
    
    // Speed controls
    static int paused = 0;
    static int frame_delay = 12;
    
    static int selected_sq = -1;
    if (env->human_play && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        Vector2 mp = GetMousePosition();
        int file = (int)(mp.x) / cell_size;
        int rank = 7 - ((int)(mp.y) / cell_size);
        if (file >= 0 && file < 8 && rank >= 0 && rank < 8) {
            int clicked_sq = (int)make_square(file, rank);
            if (selected_sq == -1) {

                if (env->pos.sideToMove == env->human_color) {
                    Piece pc = piece_on(&env->pos, (Square)clicked_sq);
                    if (pc != NO_PIECE && color_of(pc) == env->human_color) {
                        bool has_from = false;
                        for (int i = 0; i < env->legal_moves.count; i++) {
                            if ((int)from_sq(env->legal_moves.moves[i].move) == clicked_sq) { has_from = true; break; }
                        }
                        if (has_from) selected_sq = clicked_sq;
                    }
                }
            } else {
                Move chosen = MOVE_NONE;
                for (int i = 0; i < env->legal_moves.count; i++) {
                    Move m = env->legal_moves.moves[i].move;
                    if ((int)from_sq(m) == selected_sq && (int)to_sq(m) == clicked_sq) { chosen = m; break; }
                }
                if (chosen != MOVE_NONE) {
                    do_move(&env->pos, chosen, env->undo_stack, &env->undo_stack_ptr);
                    env->tick++;
                    env->chess_moves++;
                    generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
                    env->legal_moves_side = env->pos.sideToMove;
                    env->legal_moves_key = env->pos.key;
                    env->actions[0] = -1;
                    populate_observations(env);
                }
                selected_sq = -1;
            }
        }
    }

    BeginDrawing();
    ClearBackground((Color){40, 40, 40, 255});
    
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            Color square_color = ((rank + file) % 2 == 1) 
                ? (Color){240, 217, 181, 255}
                : (Color){181, 136, 99, 255};
            
            int draw_x = file * cell_size;
            int draw_y = (7 - rank) * cell_size;
            DrawRectangle(draw_x, draw_y, cell_size, cell_size, square_color);

            if (selected_sq != -1) {
                int sel_f = file_of((Square)selected_sq);
                int sel_r = rank_of((Square)selected_sq);
                if (sel_f == file && sel_r == rank) {
                    DrawRectangleLines(draw_x, draw_y, cell_size, cell_size, (Color){255, 215, 0, 255});
                }
                for (int i = 0; i < env->legal_moves.count; i++) {
                    Move m = env->legal_moves.moves[i].move;
                    if ((int)from_sq(m) == selected_sq) {
                        Square to = to_sq(m);
                        int tf = file_of(to);
                        int tr = rank_of(to);
                        if (tf == file && tr == rank) {
                            DrawRectangleLines(draw_x+2, draw_y+2, cell_size-4, cell_size-4, (Color){0, 200, 0, 255});
                        }
                    }
                }
            }
        }
    }
    
    for (Square sq = SQ_A1; sq <= SQ_H8; sq++) {
        Piece pc = piece_on(&env->pos, sq);
        if (pc != NO_PIECE) {
            int file = file_of(sq);
            int rank = rank_of(sq);
            draw_piece(env, pc, file, rank, cell_size);
        }
    }
    
    const int scoreboard_y = board_size + 10;
    char score_text[128];
    snprintf(score_text, sizeof(score_text), "White: %.1f  Black: %.1f", 
             env->white_score, env->black_score);
    DrawText(score_text, 10, scoreboard_y, 20, WHITE);
    
    char learner_text[128];
    snprintf(learner_text, sizeof(learner_text), "Learner: %.0f-%.0f-%.0f (W-L-D)", 
             env->learner_wins, env->learner_losses, env->learner_draws);
    DrawText(learner_text, 10, scoreboard_y + 55, 16, GREEN);
    
    if (env->last_result[0] != '\0') {
        Color result_color = YELLOW;
        if (strstr(env->last_result, "White")) result_color = (Color){240, 217, 181, 255};
        else if (strstr(env->last_result, "Black")) result_color = (Color){100, 100, 100, 255};
        
        DrawText(env->last_result, 10, scoreboard_y + 25, 18, result_color);
    }
    
    char move_text[64];
    snprintf(move_text, sizeof(move_text), "Move: %d", env->chess_moves);
    DrawText(move_text, board_size - 100, scoreboard_y, 18, LIGHTGRAY);
    
    const char* learner_str = (env->learner_color == CHESS_WHITE) ? "Learner: White" : "Learner: Black";
    DrawText(learner_str, board_size - 120, scoreboard_y + 25, 16, LIGHTGRAY);
    
    const int btn_width = 36;
    const int btn_height = 24;
    const int btn_y = scoreboard_y + 45;
    const int btn_start_x = board_size / 2 - 70;
    
    Rectangle minus_btn = {btn_start_x, btn_y, btn_width, btn_height};
    DrawRectangleRec(minus_btn, DARKGRAY);
    DrawRectangleLinesEx(minus_btn, 2, LIGHTGRAY);
    DrawText("-", btn_start_x + 14, btn_y + 4, 20, WHITE);
    
    Rectangle pause_btn = {btn_start_x + btn_width + 5, btn_y, btn_width + 10, btn_height};
    DrawRectangleRec(pause_btn, paused ? MAROON : DARKGREEN);
    DrawRectangleLinesEx(pause_btn, 2, LIGHTGRAY);
    DrawText(paused ? ">" : "||", btn_start_x + btn_width + 14, btn_y + 4, 18, WHITE);
    
    Rectangle plus_btn = {btn_start_x + 2*btn_width + 20, btn_y, btn_width, btn_height};
    DrawRectangleRec(plus_btn, DARKGRAY);
    DrawRectangleLinesEx(plus_btn, 2, LIGHTGRAY);
    DrawText("+", btn_start_x + 2*btn_width + 32, btn_y + 4, 20, WHITE);
    
    // Speed indicator
    char speed_text[32];
    int speed_val = frame_delay > 0 ? 60 / frame_delay : 60;
    snprintf(speed_text, sizeof(speed_text), "%dx", speed_val > 0 ? speed_val : 1);
    DrawText(speed_text, btn_start_x + 3*btn_width + 30, btn_y + 4, 18, paused ? RED : LIGHTGRAY);
    
    EndDrawing();
    
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        Vector2 mouse = GetMousePosition();
        if (CheckCollisionPointRec(mouse, minus_btn)) {
            frame_delay = frame_delay < 60 ? frame_delay + 4 : 60;
        }
        if (CheckCollisionPointRec(mouse, pause_btn)) {
            paused = !paused;
        }
        if (CheckCollisionPointRec(mouse, plus_btn)) {
            frame_delay = frame_delay > 4 ? frame_delay - 4 : 1;
        }
    }
    
    if (IsKeyPressed(KEY_SPACE)) paused = !paused;
    if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) frame_delay = frame_delay > 4 ? frame_delay - 4 : 1;
    if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) frame_delay = frame_delay < 60 ? frame_delay + 4 : 60;
    
    while (paused) {
        BeginDrawing();
        ClearBackground(DARKGRAY);
        
        for (int r = 0; r < 8; r++) {
            for (int f = 0; f < 8; f++) {
                Color sq_color = ((r + f) % 2 == 0) ? (Color){181, 136, 99, 255} : (Color){240, 217, 181, 255};
                DrawRectangle(f * cell_size, (7 - r) * cell_size, cell_size, cell_size, sq_color);
                Piece pc = piece_on(&env->pos, (Square)make_square(f, r));
                if (pc != NO_PIECE) {
                    draw_piece(env, pc, f, r, cell_size);
                }
            }
        }
        
        DrawRectangle(0, board_size, board_size, 80, (Color){40, 40, 40, 255});
        DrawText("PAUSED", board_size / 2 - 50, scoreboard_y + 10, 24, RED);
        
        DrawRectangleRec(minus_btn, DARKGRAY);
        DrawRectangleLinesEx(minus_btn, 2, LIGHTGRAY);
        DrawText("-", btn_start_x + 14, btn_y + 4, 20, WHITE);
        
        DrawRectangleRec(pause_btn, MAROON);
        DrawRectangleLinesEx(pause_btn, 2, LIGHTGRAY);
        DrawText(">", btn_start_x + btn_width + 18, btn_y + 4, 18, WHITE);
        
        DrawRectangleRec(plus_btn, DARKGRAY);
        DrawRectangleLinesEx(plus_btn, 2, LIGHTGRAY);
        DrawText("+", btn_start_x + 2*btn_width + 32, btn_y + 4, 20, WHITE);
        
        snprintf(speed_text, sizeof(speed_text), "%dx", speed_val > 0 ? speed_val : 1);
        DrawText(speed_text, btn_start_x + 3*btn_width + 30, btn_y + 4, 18, RED);
        
        EndDrawing();
        
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            Vector2 mouse = GetMousePosition();
            if (CheckCollisionPointRec(mouse, pause_btn)) {
                paused = 0;
                break;
            }
            if (CheckCollisionPointRec(mouse, minus_btn)) {
                frame_delay = frame_delay < 60 ? frame_delay + 4 : 60;
                speed_val = 60 / frame_delay;
            }
            if (CheckCollisionPointRec(mouse, plus_btn)) {
                frame_delay = frame_delay > 4 ? frame_delay - 4 : 1;
                speed_val = 60 / frame_delay;
            }
        }
        if (IsKeyPressed(KEY_SPACE)) {
            paused = 0;
            break;
        }
        if (IsKeyDown(KEY_ESCAPE) || WindowShouldClose()) {
            CloseWindow();
            exit(0);
        }
        usleep(16000); 
    }

    if (frame_delay > 1 && !(env->human_play && env->human_color != -1 && env->pos.sideToMove == env->human_color)) {
        usleep(frame_delay * 16000);
    }
}

void c_close(Chess* env) {
    if (env->client != NULL) {
        if (env->client->use_unicode_pieces && env->client->piece_font.texture.id != 0) {
            UnloadFont(env->client->piece_font);
        }
        if (IsWindowReady()) {
            CloseWindow();
        }
        free(env->client);
        env->client = NULL;
    }
    if (env->fen_curriculum != NULL) {
        for (int i = 0; i < env->num_fens; i++) {
            free(env->fen_curriculum[i]);
        }
        free(env->fen_curriculum);
        env->fen_curriculum = NULL;
        env->num_fens = 0;
    }
}
