#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <unistd.h> 
#include "raylib.h"

typedef uint64_t Bitboard;
typedef uint64_t Key;
typedef uint32_t Square;
typedef uint32_t Move;
typedef uint32_t Piece;
typedef uint8_t ChessColor;

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
    WHITE_CASTLING = 3, BLACK_CASTLING = 12
};


enum { NORMAL, PROMOTION, ENPASSANT, CASTLING };

enum {
    NORTH = 8, EAST = 1, SOUTH = -8, WEST = -1,
    NORTH_EAST = 9, SOUTH_EAST = -7,
    NORTH_WEST = 7, SOUTH_WEST = -9
};

// #define MAX_TOKENS 64
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

#define pieces(pos) ((pos)->byTypeBB[0])
#define pieces_p(pos, p) ((pos)->byTypeBB[p])
#define pieces_c(pos, c) ((pos)->byColorBB[c])
#define pieces_cp(pos, c, p) (pieces_p(pos, p) & pieces_c(pos, c))
#define piece_on(pos, s) ((pos)->board[s])
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

static const char* PIECE_FILLED[] = {
    "",
    "♟", "♞", "♝", "♜", "♛", "♚",
    "", "",
    "♟", "♞", "♝", "♜", "♛", "♚"
};


static uint64_t prng_state = 1070372;
static inline uint64_t prng_rand(void) {
    prng_state ^= prng_state >> 12;
    prng_state ^= prng_state << 25;
    prng_state ^= prng_state >> 27;
    return prng_state * 2685821657736338717ULL;
}

extern Bitboard SquareBB[65];
extern Bitboard PawnAttacks[2][64];
extern Bitboard KnightAttacks[64];
extern Bitboard KingAttacks[64];
extern Bitboard BetweenBB[64][64];
extern Bitboard LineBB[64][64];

static Bitboard BishopMasks[64];
static uint64_t BishopMagics[64];
static int BishopShifts[64];
static Bitboard BishopTable[64 * 512];
static Bitboard* BishopAttacks[64];
static const uint64_t BISHOP_MAGICS[64] = {
    9368648609924554880ULL, 9009475591934976ULL,     4504776450605056ULL,
    1130334595844096ULL,    1725202480235520ULL,     288516396277699584ULL,
    613618303369805920ULL,  10168455467108368ULL,    9046920051966080ULL,
    36031066926022914ULL,   1152925941509587232ULL,  9301886096196101ULL,
    290536121828773904ULL,  5260205533369993472ULL,  7512287909098426400ULL,
    153141218749450240ULL,  9241386469758076456ULL,  5352528174448640064ULL,
    2310346668982272096ULL, 1154049638051909890ULL,  282645627930625ULL,
    2306405976892514304ULL, 11534281888680707074ULL, 72339630111982113ULL,
    8149474640617539202ULL, 2459884588819024896ULL,  11675583734899409218ULL,
    1196543596102144ULL,    5774635144585216ULL,     145242600416216065ULL,
    2522607328671633440ULL, 145278609400071184ULL,   5101802674455216ULL,
    650979603259904ULL,     9511646410653040801ULL,  1153493285013424640ULL,
    18016048314974752ULL,   4688397299729694976ULL,  9226754220791842050ULL,
    4611969694574863363ULL, 145532532652773378ULL,   5265289125480634376ULL,
    288239448330604544ULL,  2395019802642432ULL,     14555704381721968898ULL,
    2324459974457168384ULL, 23652833739932677ULL,    282583111844497ULL,
    4629880776036450560ULL, 5188716322066279440ULL,  146367151686549765ULL,
    1153170821083299856ULL, 2315697107408912522ULL,  2342448293961403408ULL,
    2309255902098161920ULL, 469501395595331584ULL,   4615626809856761874ULL,
    576601773662552642ULL,  621501155230386208ULL,   13835058055890469376ULL,
    3748138521932726784ULL, 9223517207018883457ULL,  9237736128969216257ULL,
    1127068154855556ULL,
};

static Bitboard RookMasks[64];
static uint64_t RookMagics[64];
static int RookShifts[64];
static Bitboard RookTable[64 * 4096];  
static Bitboard* RookAttacks[64];
static const uint64_t ROOK_MAGICS[64] = {
    612498416294952992ULL,  2377936612260610304ULL,  36037730568766080ULL,
    72075188908654856ULL,   144119655536003584ULL,   5836666216720237568ULL,
    9403535813175676288ULL, 1765412295174865024ULL,  3476919663777054752ULL,
    288300746238222339ULL,  9288811671472386ULL,     146648600474026240ULL,
    3799946587537536ULL,    704237264700928ULL,      10133167915730964ULL,
    2305983769267405952ULL, 9223634270415749248ULL,  10344480540467205ULL,
    9376496898355021824ULL, 2323998695235782656ULL,  9241527722809755650ULL,
    189159985010188292ULL,  2310421375767019786ULL,  4647717014536733827ULL,
    5585659813035147264ULL, 1442911135872321664ULL,  140814801969667ULL,
    1188959108457300100ULL, 288815318485696640ULL,   758869733499076736ULL,
    234750139167147013ULL,  2305924931420225604ULL,  9403727128727390345ULL,
    9223970239903959360ULL, 309094713112139074ULL,   38290492990967808ULL,
    3461016597114651648ULL, 181289678366835712ULL,   4927518981226496513ULL,
    1155212901905072225ULL, 36099167912755202ULL,    9024792514543648ULL,
    4611826894462124048ULL, 291045264466247688ULL,   83880127713378308ULL,
    1688867174481936ULL,    563516973121544ULL,      9227888831703941123ULL,
    703691741225216ULL,     45203259517829248ULL,    693563138976596032ULL,
    4038638777286134272ULL, 865817582546978176ULL,   13835621555058516608ULL,
    11541041685463296ULL,   288511853443695360ULL,   283749161902275ULL,
    176489098445378ULL,     2306124759338845321ULL,  720584805193941061ULL,
    4977040710267061250ULL, 10097633331715778562ULL, 325666550235288577ULL,
    1100057149646ULL,
};

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
    Key key;
} Position;

typedef struct {
    Move move;
} ExtMove;

typedef struct {
    ExtMove moves[256];
    int count;
} MoveList;

// Relational NNUE tokens
/*enum {
    O_TOKEN_COUNT = 0,                    
    O_TOKEN_DATA  = 2,                  
    O_SIDE = 130,                      
    O_CASTLE = 132,                   
    O_EP = 148,                      
    O_PICK_PHASE = 213,             
    O_SELECTED_PIECE = 215,        
    O_VALID_PIECES = 279,         
    O_VALID_DESTS = 343,         
    O_VALID_PROMOS = 407,       

    O_SELF_CHECK = 439,
    O_OPP_CHECK = 440,
    O_RULE50 = 441,
    O_REPETITION = 442,
    O_PASS_VALID = 443,

    OBS_SIZE = 444
};*/

/*enum {
    O_BOARD = 0,
    O_SIDE = 768,
    O_CASTLE = 770,
    O_EP = 786,
    O_PICK_PHASE = 851,
    O_SELECTED_PIECE = 853,
    O_VALID_PIECES = 917,
    O_VALID_DESTS = 981,
    O_VALID_PROMOS = 1045,
    O_SELF_CHECK = 1077,
    O_OPP_CHECK = 1078,
    O_RULE50 = 1079,
    O_REPETITION = 1080,
    O_PASS_VALID = 1081,
    O_CONTROL_US = 1082,
    O_CONTROL_THEM = 1146,
    OBS_SIZE = 1210
};*/

// NNUE: relational token encoding
// static inline uint16_t make_rel_token(int king_sq, int piece_type, int piece_sq) {
//     return (uint16_t)(king_sq * 12 * 64 + piece_type * 64 + piece_sq);
// }

#define SQ_FEATURES 17
enum {
    O_SQUARES = 0,
    O_VALID_PROMOS = 1088,
    O_SIDE = 1120,
    O_CASTLE = 1121,
    O_EP = 1122,
    O_PICK_PHASE = 1123,
    O_SELF_CHECK = 1124,
    O_OPP_CHECK = 1125,
    O_RULE50 = 1126,
    O_REPETITION = 1127,
    O_PASS_VALID = 1128,
    OBS_SIZE = 1129
};

#define PASS_ACTION 96
#define NUM_ACTIONS 97
enum {
    CHESS_MODE_SELFPLAY = 0,
    CHESS_MODE_HUMAN = 1,
    CHESS_MODE_RANDOM_BOT = 2
};

typedef struct {
    float perf;
    float score;
    float draw_rate;
    float timeout_rate;
    float chess_moves;
    float episode_length;
    float episode_return;
    float invalid_action_rate;
    float n;
} Log;

typedef struct {
    int cell_size;
    Font piece_font;
    int use_unicode_pieces;
} Client;

typedef struct {
    Piece captured;
    uint8_t castlingRights;
    uint8_t epSquare;
    uint8_t rule50;
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
    int legal_dirty;
    int game_result;
    int tick;
    int chess_moves;
    int max_moves;
    float reward_draw;
    float episode_reward;
    int render_fps;
    int selfplay;
    int human_play;
    int random_bot;
    int mode;
    
    char starting_fen[128];
    char** fen_curriculum;
    float fen_curric_pct;
    int num_fens;
    int random_fen;
    
    UndoInfo undo_stack[MAX_GAME_PLIES];
    int undo_stack_ptr;
    
    int invalid_actions_this_episode;
    
    int pick_phase[2];
    Square selected_square[2];
    MoveList valid_destinations[2];
    float reward_invalid_piece;
    float reward_invalid_move;
    float reward_repetition;
    
    int enable_50_move_rule;
    int enable_threefold_repetition;
    
    int learner_color;
    int human_color;
    float white_score;
    float black_score;
    float learner_wins;
    float learner_losses; 
    float learner_draws;
    char last_result[32];
    
    Move pgn_moves[MAX_GAME_PLIES];
    int pgn_move_count;
    int show_game_end_popup;
    
    int log_pgn;
    int log_pgn_choice_made;
    char pgn_filename[128];
    int pgn_game_number;
    
    int white_captured[6];
    int black_captured[6];
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
    occupied &= RookMasks[s];
    return RookAttacks[s][(occupied * RookMagics[s]) >> RookShifts[s]];
}

static inline Bitboard bishop_attacks_bb(Square s, Bitboard occupied) {
    occupied &= BishopMasks[s];
    return BishopAttacks[s][(occupied * BishopMagics[s]) >> BishopShifts[s]];
}

static inline Bitboard queen_attacks_bb(Square s, Bitboard occupied) {
    return rook_attacks_bb(s, occupied) | bishop_attacks_bb(s, occupied);
}


Bitboard SquareBB[65];
Bitboard PawnAttacks[2][64];
Bitboard KnightAttacks[64];
Bitboard KingAttacks[64];
Bitboard BetweenBB[64][64];
Bitboard LineBB[64][64];
Zobrist zob;

static bool bitboards_initialized = false;

static Bitboard index_to_occupancy(int index, Bitboard mask) {
    Bitboard occ = 0;
    int bits = popcount(mask);
    for (int i = 0; i < bits; i++) {
        Square sq = lsb(mask);
        mask &= mask - 1;
        if (index & (1 << i)) occ |= sq_bb(sq);
    }
    return occ;
}
static Bitboard compute_bishop_mask(Square s) {
    Bitboard mask = 0;
    int r = rank_of(s), f = file_of(s);
    for (int rr = r + 1, ff = f + 1; rr < 7 && ff < 7; rr++, ff++) mask |= sq_bb(make_square(ff, rr));
    for (int rr = r - 1, ff = f + 1; rr > 0 && ff < 7; rr--, ff++) mask |= sq_bb(make_square(ff, rr));
    for (int rr = r - 1, ff = f - 1; rr > 0 && ff > 0; rr--, ff--) mask |= sq_bb(make_square(ff, rr));
    for (int rr = r + 1, ff = f - 1; rr < 7 && ff > 0; rr++, ff--) mask |= sq_bb(make_square(ff, rr));
    return mask;
}

static void init_bishop_magics(void) {
    Bitboard* table_ptr = BishopTable;
    
    for (Square sq = 0; sq < 64; sq++) {
        BishopMasks[sq] = compute_bishop_mask(sq);
        BishopMagics[sq] = BISHOP_MAGICS[sq];
        
        int bits = popcount(BishopMasks[sq]);
        BishopShifts[sq] = 64 - bits;
        BishopAttacks[sq] = table_ptr;
        
        int num_entries = 1 << bits;
        memset(table_ptr, 0, num_entries * sizeof(Bitboard));
        
        for (int i = 0; i < num_entries; i++) {
            Bitboard occ = index_to_occupancy(i, BishopMasks[sq]);
            
            Bitboard attacks = 0;
            int r = rank_of(sq), f = file_of(sq);
            for (int rr = r + 1, ff = f + 1; rr < 8 && ff < 8; rr++, ff++) {
                Square tsq = make_square(ff, rr);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            for (int rr = r - 1, ff = f + 1; rr >= 0 && ff < 8; rr--, ff++) {
                Square tsq = make_square(ff, rr);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            for (int rr = r - 1, ff = f - 1; rr >= 0 && ff >= 0; rr--, ff--) {
                Square tsq = make_square(ff, rr);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            for (int rr = r + 1, ff = f - 1; rr < 8 && ff >= 0; rr++, ff--) {
                Square tsq = make_square(ff, rr);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            
            uint64_t idx = (occ * BishopMagics[sq]) >> BishopShifts[sq];
            table_ptr[idx] = attacks;
        }
        table_ptr += num_entries;
    }
}

static Bitboard compute_rook_mask(Square s) {
    Bitboard mask = 0;
    int r = rank_of(s), f = file_of(s);
    for (int rr = r + 1; rr < 7; rr++) mask |= sq_bb(make_square(f, rr));
    for (int rr = r - 1; rr > 0; rr--) mask |= sq_bb(make_square(f, rr));
    for (int ff = f + 1; ff < 7; ff++) mask |= sq_bb(make_square(ff, r));
    for (int ff = f - 1; ff > 0; ff--) mask |= sq_bb(make_square(ff, r));
    return mask;
}

static void init_rook_magics(void) {
    Bitboard* table_ptr = RookTable;
    
    for (Square sq = 0; sq < 64; sq++) {
        RookMasks[sq] = compute_rook_mask(sq);
        RookMagics[sq] = ROOK_MAGICS[sq];
        
        int bits = popcount(RookMasks[sq]);
        RookShifts[sq] = 64 - bits;
        RookAttacks[sq] = table_ptr;
        
        int num_entries = 1 << bits;
        memset(table_ptr, 0, num_entries * sizeof(Bitboard));
        
        for (int i = 0; i < num_entries; i++) {
            Bitboard occ = index_to_occupancy(i, RookMasks[sq]);
            
            Bitboard attacks = 0;
            int r = rank_of(sq), f = file_of(sq);
            for (int rr = r + 1; rr < 8; rr++) {
                Square tsq = make_square(f, rr);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            for (int rr = r - 1; rr >= 0; rr--) {
                Square tsq = make_square(f, rr);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            for (int ff = f + 1; ff < 8; ff++) {
                Square tsq = make_square(ff, r);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            for (int ff = f - 1; ff >= 0; ff--) {
                Square tsq = make_square(ff, r);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            
            uint64_t idx = (occ * RookMagics[sq]) >> RookShifts[sq];
            table_ptr[idx] = attacks;
        }
        table_ptr += num_entries;
    }
}

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
                
                // BetweenBB: squares strictly between s1 and s2
                int f = f1 + step_f;
                int r = r1 + step_r;
                while (f != f2 || r != r2) {
                    Square sq = make_square(f, r);
                    BetweenBB[s1][s2] |= sq_bb(sq);
                    f += step_f;
                    r += step_r;
                }

                f = f1;
                r = r1;
                while (f - step_f >= 0 && f - step_f < 8 && r - step_r >= 0 && r - step_r < 8) {
                    f -= step_f;
                    r -= step_r;
                }
                while (f >= 0 && f < 8 && r >= 0 && r < 8) {
                    LineBB[s1][s2] |= sq_bb(make_square(f, r));
                    f += step_f;
                    r += step_r;
                }
            }
        }
    }
    init_bishop_magics(); 
    init_rook_magics();
    bitboards_initialized = true;
}

static void pos_set(Position* pos, const char* fen) {
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

static void do_move(Position* pos, Move m, UndoInfo* undo_stack, int* undo_stack_ptr) {
    if (m == MOVE_NULL) {
        undo_stack[*undo_stack_ptr].captured = NO_PIECE;
        undo_stack[*undo_stack_ptr].castlingRights = pos->castlingRights;
        undo_stack[*undo_stack_ptr].epSquare = pos->epSquare;
        undo_stack[*undo_stack_ptr].rule50 = pos->rule50;
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
    undo_stack[*undo_stack_ptr].key = pos->key;
    undo_stack[*undo_stack_ptr].pliesFromNull = (*undo_stack_ptr > 0) ? undo_stack[*undo_stack_ptr - 1].pliesFromNull + 1 : 0;
    (*undo_stack_ptr)++;
    
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
    
    switch (move_type) {
        case CASTLING: {
            pos->key ^= zob.psq[pc][from];
            
            pos->board[from] = NO_PIECE;
            pos->board[to] = pc;
            pos->byTypeBB[pt] ^= sq_bb(from) ^ sq_bb(to);
            pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
            pos->byTypeBB[0] ^= sq_bb(from) ^ sq_bb(to);
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
            pos->key ^= zob.psq[rook][rook_to];
            break;
        }
        case ENPASSANT: {
            pos->key ^= zob.psq[pc][from];
            
            pos->board[from] = NO_PIECE;
            pos->board[to] = pc;
            pos->byTypeBB[pt] ^= sq_bb(from) ^ sq_bb(to);
            pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
            pos->byTypeBB[0] ^= sq_bb(from) ^ sq_bb(to);
            pos->key ^= zob.psq[pc][to];
            
            Square cap_sq = to + (us == CHESS_WHITE ? SOUTH : NORTH);
            Piece cap_pawn = piece_on(pos, cap_sq);
            pos->key ^= zob.psq[cap_pawn][cap_sq];
            pos->board[cap_sq] = NO_PIECE;
            pos->byTypeBB[PAWN] ^= sq_bb(cap_sq);
            pos->byColorBB[them] ^= sq_bb(cap_sq);
            pos->byTypeBB[0] ^= sq_bb(cap_sq);
            pos->pieceCount[cap_pawn]--;
            break;
        }
        case NORMAL:
        case PROMOTION: {
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
            break;
        }
        default:
            break;
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
    
    pos->sideToMove = them;
    pos->key ^= zob.side;
}

static void undo_move(Position* pos, Move m, UndoInfo* undo_stack, int* undo_stack_ptr) {
    (*undo_stack_ptr)--;
    UndoInfo* undo = &undo_stack[*undo_stack_ptr];
    
    if (m == MOVE_NULL) {
        pos->castlingRights = undo->castlingRights;
        pos->epSquare = undo->epSquare;
        pos->rule50 = undo->rule50;
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
    pos->key = undo->key;
    pos->sideToMove = us;
    
    switch (move_type) {
        case CASTLING: {
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
            break;
        }
        case ENPASSANT: {
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
            break;
        }
        case NORMAL:
        case PROMOTION: {
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
            break;
        }
        default:
            break;
    }
}

static void add_move(MoveList* ml, Move m) {
    ml->moves[ml->count].move = m;
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
    return (pawn_attacks_bb(CHESS_WHITE, sq) & pieces_cp(pos, CHESS_BLACK, PAWN) & occupied)
         | (pawn_attacks_bb(CHESS_BLACK, sq) & pieces_cp(pos, CHESS_WHITE, PAWN) & occupied)
         | (knight_attacks_bb(sq) & pieces_p(pos, KNIGHT) & occupied)
         | (king_attacks_bb(sq) & pieces_p(pos, KING) & occupied)
         | (bishop_attacks_bb(sq, occupied) & (pieces_p(pos, BISHOP) | pieces_p(pos, QUEEN)))
         | (rook_attacks_bb(sq, occupied) & (pieces_p(pos, ROOK) | pieces_p(pos, QUEEN)));
}

static bool is_check(Position* pos, ChessColor c) {
    Bitboard king_bb = pieces_cp(pos, c, KING);
    if (!king_bb) return false;
    Square king_sq = lsb(king_bb);
    return (attackers_to_sq(pos, king_sq, pieces(pos)) & pieces_c(pos, !c)) != 0;
}

static Bitboard compute_pinned(Position* pos, ChessColor c) {
    Bitboard pinned = 0;
    Bitboard our_pieces = pieces_c(pos, c);
    Bitboard king_bb = pieces_cp(pos, c, KING);
    if (!king_bb) return 0;
    
    Square ksq = lsb(king_bb);
    ChessColor them = !c;
    Bitboard occupied = pieces(pos);
    
    Bitboard diag_pinners = (pieces_cp(pos, them, BISHOP) | pieces_cp(pos, them, QUEEN)) 
                          & bishop_attacks_bb(ksq, 0);
    
    while (diag_pinners) {
        Square pinner_sq = pop_lsb(&diag_pinners);
        Bitboard between = BetweenBB[ksq][pinner_sq] & occupied;
        if (popcount(between) == 1) {
            pinned |= between & our_pieces;
        }
    }
    
    Bitboard rook_pinners = (pieces_cp(pos, them, ROOK) | pieces_cp(pos, them, QUEEN)) 
                          & rook_attacks_bb(ksq, 0);
    
    while (rook_pinners) {
        Square pinner_sq = pop_lsb(&rook_pinners);
        Bitboard between = BetweenBB[ksq][pinner_sq] & occupied;
        if (popcount(between) == 1) {
            pinned |= between & our_pieces;
        }
    }
    
    return pinned;
}

static inline bool is_legal_move_fast(Position* pos, Move m, Bitboard pinned, Square ksq, ChessColor us) {
    Square from = from_sq(m);
    Square to = to_sq(m);
    int mt = type_of_m(m);
    
    if (from == ksq) {
        if (mt == CASTLING) {
            ChessColor them = !us;
            if (is_check(pos, us)) return false;
            Square mid = (from + to) / 2;
            Bitboard occ = pieces(pos) ^ sq_bb(from);
            if (attackers_to_sq(pos, mid, occ) & pieces_c(pos, them)) return false;
            if (attackers_to_sq(pos, to, occ) & pieces_c(pos, them)) return false;
            return true;
        }
        Bitboard occ = pieces(pos) ^ sq_bb(from);
        return !(attackers_to_sq(pos, to, occ) & pieces_c(pos, !us));
    }
    
    if (mt == ENPASSANT) {
        Bitboard occ = pieces(pos) ^ sq_bb(from) ^ sq_bb(to);
        Square capsq = to + (us == CHESS_WHITE ? -8 : 8);
        occ ^= sq_bb(capsq);
        return !(attackers_to_sq(pos, ksq, occ) & pieces_c(pos, !us));
    }
    
    if (!(pinned & sq_bb(from))) {
        return true;
    }
    
    return LineBB[ksq][from] & sq_bb(to);
}

static inline bool is_legal_move(Position* pos, Move m) {
    ChessColor us = pos->sideToMove;
    ChessColor them = (ChessColor)!us;
    int mt = type_of_m(m);
    if (mt == CASTLING) {
        if (is_check(pos, us)) return false;
        Square from = from_sq(m), to = to_sq(m);
        Square mid = (from + to) / 2;
        Bitboard occ = pieces(pos);
        if ((attackers_to_sq(pos, mid, occ) & pieces_c(pos, them))
         || (attackers_to_sq(pos, to, occ) & pieces_c(pos, them))) return false;
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

static void generate_legal(Position* pos, MoveList* ml, UndoInfo* undo_stack, int* undo_stack_ptr) {
    generate_pseudo_legal(pos, ml, pos->sideToMove);
    ChessColor us = pos->sideToMove;
    Bitboard king_bb = pieces_cp(pos, us, KING);
    Square ksq = king_bb ? lsb(king_bb) : SQ_NONE;
    Bitboard pinned = compute_pinned(pos, us);
    bool in_check = is_check(pos, us);
    
    int write = 0;
    for (int i = 0; i < ml->count; i++) {
        Move m = ml->moves[i].move;
        bool legal = in_check
            ? is_legal_move(pos, m)
            : is_legal_move_fast(pos, m, pinned, ksq, us);
        if (legal) {
            ml->moves[write++] = ml->moves[i];
        }
    }
    ml->count = write;
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

static inline Bitboard all_pawn_attacks(Bitboard pawns, ChessColor c) {
    if (c == CHESS_WHITE) {
        return ((pawns << 7) & ~FileHBB) | ((pawns << 9) & ~FileABB);
    } else {
        return ((pawns >> 7) & ~FileABB) | ((pawns >> 9) & ~FileHBB);
    }
}

static inline Bitboard control_map_for_side(Position* pos, ChessColor side) {
    Bitboard occupied = pieces(pos);
    Bitboard attacks = 0;

    Bitboard bb = pieces_cp(pos, side, PAWN);
    attacks |= all_pawn_attacks(bb, side);

    bb = pieces_cp(pos, side, KNIGHT);
    while (bb) {
        Square sq = pop_lsb(&bb);
        attacks |= knight_attacks_bb(sq);
    }

    bb = pieces_cp(pos, side, KING);
    if (bb) {
        attacks |= king_attacks_bb(lsb(bb));
    }

    bb = pieces_cp(pos, side, BISHOP);
    while (bb) {
        Square sq = pop_lsb(&bb);
        attacks |= bishop_attacks_bb(sq, occupied);
    }

    bb = pieces_cp(pos, side, ROOK);
    while (bb) {
        Square sq = pop_lsb(&bb);
        attacks |= rook_attacks_bb(sq, occupied);
    }

    bb = pieces_cp(pos, side, QUEEN);
    while (bb) {
        Square sq = pop_lsb(&bb);
        attacks |= queen_attacks_bb(sq, occupied);
    }

    return attacks;
}

static void populate_observations(Chess* env) {
    uint8_t* obs = env->observations;
    Position* pos = &env->pos;
    Bitboard white_control = control_map_for_side(pos, CHESS_WHITE);
    Bitboard black_control = control_map_for_side(pos, CHESS_BLACK);
    ChessColor side_to_move = pos->sideToMove;

    uint8_t rep_val = 255;
    if (env->undo_stack_ptr >= 4) {
        uint8_t plies = env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull;
        if (plies >= 4) {
            int repetitions = 0;
            for (int i = 4; i <= plies; i += 2) {
                int idx = env->undo_stack_ptr - i;
                if (idx >= 0 && env->undo_stack[idx].key == pos->key) {
                    repetitions++;
                    if (repetitions >= 2) {
                        break;
                    }
                }
            }
            if (repetitions >= 2) {
                rep_val = 0;
            } else if (repetitions == 1) {
                rep_val = 128;
            }
        }
    }

    int num_players = env->mode == CHESS_MODE_SELFPLAY ? 2 : 1;
    for (int player_iter = 0; player_iter < num_players; player_iter++) {
        int player = env->mode == CHESS_MODE_SELFPLAY ? player_iter : env->learner_color;
        int buffer_idx = env->mode == CHESS_MODE_SELFPLAY
            ? ((env->learner_color == CHESS_WHITE) ? player_iter : (1 - player_iter))
            : 0;

        uint8_t* player_obs = obs + (buffer_idx * OBS_SIZE);
        memset(player_obs, 0, OBS_SIZE);

        ChessColor us = (ChessColor)player;
        ChessColor them = (ChessColor)!us;
        int player_idx = (int)us;
        int flip = player * 56;

        Bitboard valid_from_bb = 0;
        Bitboard valid_to_bb = 0;
        if (side_to_move == us) {
            int mark_destinations = env->pick_phase[player_idx] == 1;
            MoveList* marks = mark_destinations ? &env->valid_destinations[player_idx] : &env->legal_moves;
            for (int i = 0; i < marks->count; i++) {
                Move m = marks->moves[i].move;
                if (mark_destinations) {
                    valid_to_bb |= sq_bb(to_sq(m));
                } else {
                    valid_from_bb |= sq_bb(from_sq(m));
                }
            }
        }

        Bitboard us_control = (us == CHESS_WHITE) ? white_control : black_control;
        Bitboard them_control = (us == CHESS_WHITE) ? black_control : white_control;

        Bitboard selected_bb = 0;
        if (env->pick_phase[player_idx] == 1 && env->selected_square[player_idx] != SQ_NONE)
            selected_bb = sq_bb(env->selected_square[player_idx]);

        uint8_t* sq_out = player_obs + O_SQUARES;
        for (int sq = 0; sq < 64; sq++) {
            int view_sq = sq ^ flip;
            uint8_t* feat = sq_out + view_sq * SQ_FEATURES;
            Piece pc = pos->board[sq];
            if (pc != NO_PIECE) {
                int pt = type_of_p(pc);
                ChessColor c = color_of(pc);
                int channel = (c == us) ? (pt - 1) : (6 + pt - 1);
                feat[channel] = 1;
            }
            Bitboard bb = sq_bb(sq);
            feat[12] = (selected_bb & bb) ? 1 : 0;
            feat[13] = (valid_from_bb & bb) ? 1 : 0;
            feat[14] = (valid_to_bb & bb) ? 1 : 0;
            feat[15] = (us_control & bb) ? 1 : 0;
            feat[16] = (them_control & bb) ? 1 : 0;
        }

        uint8_t* valid_promos = player_obs + O_VALID_PROMOS;
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

        uint8_t castle_rights = pos->castlingRights;
        if (player == 1) {
            uint8_t flipped = 0;
            if (castle_rights & BLACK_OO) flipped |= WHITE_OO;
            if (castle_rights & BLACK_OOO) flipped |= WHITE_OOO;
            if (castle_rights & WHITE_OO) flipped |= BLACK_OO;
            if (castle_rights & WHITE_OOO) flipped |= BLACK_OOO;
            castle_rights = flipped;
        }

        player_obs[O_SIDE] = (pos->sideToMove == us) ? 0 : 1;
        player_obs[O_CASTLE] = castle_rights;
        player_obs[O_EP] = (pos->epSquare < 64)
            ? (uint8_t)((player == 1) ? (pos->epSquare ^ 56) : pos->epSquare)
            : 64;
        player_obs[O_PICK_PHASE] = (uint8_t)env->pick_phase[player_idx];
        player_obs[O_SELF_CHECK] = is_check(pos, us) ? 255 : 0;
        player_obs[O_OPP_CHECK] = is_check(pos, them) ? 255 : 0;
        player_obs[O_RULE50] = (uint8_t)((pos->rule50 * 255) / 100);
        player_obs[O_PASS_VALID] = (side_to_move != us) ? 255 : 0;
        player_obs[O_REPETITION] = rep_val;
    }
}

static int move_to_san(Position* pos, Move m, char* buf, UndoInfo* undo_stack, int* undo_stack_ptr) {
    const char files[] = "abcdefgh";
    const char ranks[] = "12345678";
    const char piece_chars[] = ".PNBRQK";
    char* ptr = buf;
    
    Square from = from_sq(m);
    Square to = to_sq(m);
    int move_type = type_of_m(m);
    Piece pc = piece_on(pos, from);
    int pt = type_of_p(pc);
    ChessColor us = pos->sideToMove;
    
    if (move_type == CASTLING) {
        if (to > from) {
            strcpy(ptr, "O-O");
            ptr += 3;
        } else {
            strcpy(ptr, "O-O-O");
            ptr += 5;
        }
    } else {
        if (pt != PAWN) {
            *ptr++ = piece_chars[pt];
            
            Bitboard same_pieces = pieces_cp(pos, us, pt) & ~sq_bb(from);
            Bitboard attackers = 0;
            
            if (pt == KNIGHT) {
                attackers = knight_attacks_bb(to) & same_pieces;
            } else if (pt == BISHOP) {
                attackers = bishop_attacks_bb(to, pieces(pos)) & same_pieces;
            } else if (pt == ROOK) {
                attackers = rook_attacks_bb(to, pieces(pos)) & same_pieces;
            } else if (pt == QUEEN) {
                attackers = (bishop_attacks_bb(to, pieces(pos)) | rook_attacks_bb(to, pieces(pos))) & same_pieces;
            } else if (pt == KING) {
                attackers = king_attacks_bb(to) & same_pieces;
            }
            
            Bitboard legal_attackers = 0;
            while (attackers) {
                Square attacker_sq = pop_lsb(&attackers);
                Move test_move = make_move(attacker_sq, to);
                if (is_legal_move(pos, test_move)) {
                    legal_attackers |= sq_bb(attacker_sq);
                }
            }
            
            if (legal_attackers) {
                int same_file = 0, same_rank = 0;
                Bitboard temp = legal_attackers;
                while (temp) {
                    Square s = pop_lsb(&temp);
                    if (file_of(s) == file_of(from)) same_file++;
                    if (rank_of(s) == rank_of(from)) same_rank++;
                }
                
                if (same_file == 0) {
                    *ptr++ = files[file_of(from)];
                } else if (same_rank == 0) {
                    *ptr++ = ranks[rank_of(from)];
                } else {
                    *ptr++ = files[file_of(from)];
                    *ptr++ = ranks[rank_of(from)];
                }
            }
        }
        
        Piece captured = piece_on(pos, to);
        bool is_capture = (captured != NO_PIECE) || (move_type == ENPASSANT);
        
        if (is_capture) {
            if (pt == PAWN) {
                *ptr++ = files[file_of(from)];
            }
            *ptr++ = 'x';
        }
        
        *ptr++ = files[file_of(to)];
        *ptr++ = ranks[rank_of(to)];
        
        if (move_type == PROMOTION) {
            *ptr++ = '=';
            const char promo_pieces[] = "..NBRQ";
            *ptr++ = promo_pieces[promotion_type(m)];
        }
    }
    
    do_move(pos, m, undo_stack, undo_stack_ptr);
    
    ChessColor them = pos->sideToMove;
    if (is_check(pos, them)) {
        MoveList ml;
        generate_legal(pos, &ml, undo_stack, undo_stack_ptr);
        if (ml.count == 0) {
            *ptr++ = '#';
        } else {
            *ptr++ = '+';
        }
    }
    
    undo_move(pos, m, undo_stack, undo_stack_ptr);
    
    *ptr = '\0';
    return ptr - buf;
}

static void export_pgn_append(Chess* env, const char* filename, int append) {
    FILE* f = fopen(filename, append ? "a" : "w");
    if (!f) return;
    
    if (env->mode == CHESS_MODE_HUMAN) {
        fprintf(f, "[Event \"Human vs AI\"]\n");
        fprintf(f, "[White \"%s\"]\n", env->human_color == CHESS_WHITE ? "Human" : "AI");
        fprintf(f, "[Black \"%s\"]\n", env->human_color == CHESS_BLACK ? "Human" : "AI");
    } else {
        fprintf(f, "[Event \"Selfplay Eval Game %d\"]\n", env->pgn_game_number);
        fprintf(f, "[White \"%s\"]\n", env->learner_color == CHESS_BLACK ? "Learner" : "Opponent");
        fprintf(f, "[Black \"%s\"]\n", env->learner_color == CHESS_BLACK ? "Opponent" : "Learner");
    }
    fprintf(f, "[Site \"PufferLib\"]\n");
    fprintf(f, "[Result \"%s\"]\n\n", env->last_result);
    
    Position replay_pos;
    pos_set(&replay_pos, env->starting_fen);
    
    UndoInfo replay_undo[MAX_GAME_PLIES];
    int replay_undo_ptr = 0;
    
    char san_buf[16];
    
    for (int i = 0; i < env->pgn_move_count; i++) {
        if (i % 2 == 0) {
            fprintf(f, "%d. ", i/2 + 1);
        }
        
        Move m = env->pgn_moves[i];
        move_to_san(&replay_pos, m, san_buf, replay_undo, &replay_undo_ptr);
        fprintf(f, "%s ", san_buf);
        
        do_move(&replay_pos, m, replay_undo, &replay_undo_ptr);
        
        if ((i + 1) % 8 == 0) fprintf(f, "\n");
    }
    
    if (strcmp(env->last_result, "White Wins") == 0) {
        fprintf(f, "1-0");
    } else if (strcmp(env->last_result, "Black Wins") == 0) {
        fprintf(f, "0-1");
    } else {
        fprintf(f, "1/2-1/2");
    }
    
    fprintf(f, "\n\n");
    fclose(f);
}

static void generate_random_fen(char* fen_out) {
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

static inline int apply_move_to_env(Chess* env, Move chosen, int* is_timeout) {
    env->chess_moves++;
    
    if ((env->mode == CHESS_MODE_HUMAN || env->log_pgn) && env->pgn_move_count < MAX_GAME_PLIES) {
        env->pgn_moves[env->pgn_move_count++] = chosen;
    }
    
    ChessColor side_before = env->pos.sideToMove;
    do_move(&env->pos, chosen, env->undo_stack, &env->undo_stack_ptr);
    env->legal_dirty = 1;
    {
        int side = (int)env->pos.sideToMove;
        env->pick_phase[side] = 0;
        env->selected_square[side] = SQ_NONE;
        env->valid_destinations[side].count = 0;
    }
    
    if (env->undo_stack_ptr > 0) {
        Piece cap = env->undo_stack[env->undo_stack_ptr - 1].captured;
        if (cap != NO_PIECE) {
            int pt = type_of_p(cap) - 1;
            if (pt >= 0 && pt < 6) {
                if (color_of(cap) == CHESS_WHITE) env->white_captured[pt]++;
                else env->black_captured[pt]++;
            }
        } else if ((int)type_of_m(chosen) == ENPASSANT) {
            Piece cap_pawn = (side_before == CHESS_WHITE) ? B_PAWN : W_PAWN;
            int pt = type_of_p(cap_pawn) - 1;
            if (pt >= 0 && pt < 6) {
                if (color_of(cap_pawn) == CHESS_WHITE) env->white_captured[pt]++;
                else env->black_captured[pt]++;
            }
        }
        if (env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull > 99) {
            env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull = 99;
        }
    }
    
    generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
    env->legal_dirty = 0;

    int game_result = 0;
    *is_timeout = 0;
    if (env->chess_moves >= env->max_moves || env->undo_stack_ptr >= MAX_GAME_PLIES - 2) {
        *is_timeout = 1;
        game_result = 3;
    } else if (env->legal_moves.count == 0) {
        if (is_check(&env->pos, env->pos.sideToMove)) {
            game_result = env->pos.sideToMove == CHESS_WHITE ? 1 : 2;
        } else {
            game_result = 3;
        }
    } else if (is_insufficient_material(&env->pos)) {
        game_result = 3;
    } else if (env->enable_50_move_rule && env->pos.rule50 >= 100) {
        game_result = 3;
    } else if (env->enable_threefold_repetition && env->undo_stack_ptr >= 4) {
        uint8_t plies = env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull;
        if (plies >= 4) {
            int repetitions = 0;
            for (int i = 4; i <= plies; i += 2) {
                int idx = env->undo_stack_ptr - i;
                if (idx >= 0 && env->undo_stack[idx].key == env->pos.key) {
                    repetitions++;
                    if (repetitions >= 2) {
                        game_result = 3;
                        break;
                    }
                }
            }
        }
    }

    populate_observations(env);
    return game_result;
}

void c_reset(Chess* env) {
    env->tick = 0;
    env->chess_moves = 0;
    env->game_result = 0;
    env->undo_stack_ptr = 0;
    env->invalid_actions_this_episode = 0;
    env->episode_reward = 0.0f;
    env->pgn_move_count = 0;
    env->show_game_end_popup = 0;
    env->pick_phase[0] = 0;
    env->pick_phase[1] = 0;
    env->selected_square[0] = SQ_NONE;
    env->selected_square[1] = SQ_NONE;
    env->valid_destinations[0].count = 0;
    env->valid_destinations[1].count = 0;
    
    memset(env->white_captured, 0, sizeof(env->white_captured));
    memset(env->black_captured, 0, sizeof(env->black_captured));
    
    if (env->mode == CHESS_MODE_HUMAN) {
        env->human_color = -1;
    } else {
        env->learner_color = 1 - env->learner_color;
    }
    
    if (env->fen_curriculum != NULL && env->num_fens > 0) {
        float randvalue = (float)rand() / (float)(RAND_MAX);
        if(env->fen_curric_pct >= randvalue){
            int idx = rand() % env->num_fens;
            pos_set(&env->pos, env->fen_curriculum[idx]);
        }
        else {
            pos_set(&env->pos, env->starting_fen);
        }

    } else if (env->random_fen) {
        char fen_buf[128];
        generate_random_fen(fen_buf);
        pos_set(&env->pos, fen_buf);
    } else {
        pos_set(&env->pos, env->starting_fen);
    }
    
    env->legal_dirty = 1;
    generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
    env->legal_dirty = 0;
    populate_observations(env);

}

void c_step(Chess* env) {
    if (env->mode == CHESS_MODE_HUMAN && env->human_color == -1) {
        return;
    }
    
    if (env->mode == CHESS_MODE_SELFPLAY && !env->log_pgn_choice_made) {
        if (env->client != NULL) {
            return;
        }
        env->log_pgn = 0;
        env->log_pgn_choice_made = 1;
    }

    if (env->legal_dirty) {
        generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
        env->legal_dirty = 0;
    }
    
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->tick++;
    int move_completed = 0;
    ChessColor mover = env->pos.sideToMove;
    int mover_idx = (int)mover;
    int game_result = 0;
    int is_timeout = 0;

    if (env->mode == CHESS_MODE_RANDOM_BOT && env->pos.sideToMove != env->learner_color) {
        if (env->legal_moves.count > 0) {
            int idx = rand() % env->legal_moves.count;
            int opp = (int)!env->learner_color;
            env->pick_phase[opp] = 0;
            env->selected_square[opp] = SQ_NONE;
            env->valid_destinations[opp].count = 0;
            mover = env->pos.sideToMove;
            mover_idx = (int)mover;
            game_result = apply_move_to_env(env, env->legal_moves.moves[idx].move, &is_timeout);
            move_completed = 1;
        }
    } else {
        int action = env->actions[0];
        if (env->mode == CHESS_MODE_SELFPLAY) {
            ChessColor current_side = env->pos.sideToMove;
            action = (current_side == env->learner_color) ? env->actions[0] : env->actions[1];
        } else if (env->mode == CHESS_MODE_HUMAN && env->pos.sideToMove == env->human_color) {
            action = -1;
            env->actions[0] = -1;
        }

        if (action == -1) {
            populate_observations(env);
            return;
        }

        if (action == PASS_ACTION) {
            populate_observations(env);
            return;
        }

        mover = env->pos.sideToMove;
        mover_idx = (int)mover;

        if (action < 0 || action >= PASS_ACTION) {
            if (mover == env->learner_color) {
                env->rewards[0] += (env->pick_phase[mover_idx] == 0)
                    ? env->reward_invalid_piece : env->reward_invalid_move;
                env->invalid_actions_this_episode++;
            }
            if (env->pick_phase[mover_idx] == 1) {
                env->pick_phase[mover_idx] = 0;
                env->selected_square[mover_idx] = SQ_NONE;
                env->valid_destinations[mover_idx].count = 0;
            }
        } else {
            bool is_promo = (action >= 64 && action < 96);

            if (env->legal_moves.count == 0) {
                env->pick_phase[mover_idx] = 0;
                env->selected_square[mover_idx] = SQ_NONE;
                env->valid_destinations[mover_idx].count = 0;
            } else if (env->pick_phase[mover_idx] == 0) {
                env->pick_phase[mover_idx] = 0;
                env->selected_square[mover_idx] = SQ_NONE;
                env->valid_destinations[mover_idx].count = 0;

                bool valid_pick = !is_promo;
                Square picked_sq = SQ_NONE;
                if (valid_pick) {
                    picked_sq = (mover == CHESS_BLACK) ? (Square)(action ^ 56) : (Square)action;
                    Piece pc = piece_on(&env->pos, picked_sq);
                    valid_pick = (pc != NO_PIECE && color_of(pc) == mover);
                }

                if (valid_pick) {
                    MoveList* dests = &env->valid_destinations[mover_idx];
                    dests->count = 0;
                    for (int i = 0; i < env->legal_moves.count; i++) {
                        if (from_sq(env->legal_moves.moves[i].move) == picked_sq) {
                            dests->moves[dests->count++] = env->legal_moves.moves[i];
                        }
                    }

                    if (dests->count > 0) {
                        env->selected_square[mover_idx] = picked_sq;
                        env->pick_phase[mover_idx] = 1;
                    } else {
                        valid_pick = false;
                        env->pick_phase[mover_idx] = 0;
                        env->selected_square[mover_idx] = SQ_NONE;
                        env->valid_destinations[mover_idx].count = 0;
                    }
                }

                if (!valid_pick && mover == env->learner_color) {
                    env->rewards[0] += env->reward_invalid_piece;
                    env->invalid_actions_this_episode++;
                }
            } else {
                if (env->selected_square[mover_idx] == SQ_NONE || env->valid_destinations[mover_idx].count == 0) {
                    fprintf(stderr, "c_step: pick_phase=1 but selected_square=%u, valid_destinations.count=%d (mover=%d)\n",
                            env->selected_square[mover_idx], env->valid_destinations[mover_idx].count, mover_idx);
                    exit(1);
                }

                Square target_sq = SQ_NONE;
                Move chosen_move = MOVE_NONE;
                int desired_promo = -1;
                int desired_file = -1;

                if (is_promo) {
                    int promo_row = (action - 64) / 8;
                    desired_file = (action - 64) % 8;
                    desired_promo = QUEEN - promo_row;
                } else {
                    target_sq = (mover == CHESS_BLACK) ? (Square)(action ^ 56) : (Square)action;
                }

                for (int i = 0; i < env->valid_destinations[mover_idx].count; i++) {
                    Move m = env->valid_destinations[mover_idx].moves[i].move;
                    if (!is_promo) {
                        if ((int)to_sq(m) == (int)target_sq) {
                            chosen_move = m;
                            break;
                        }
                    } else {
                        if ((int)type_of_m(m) == PROMOTION
                                && (int)promotion_type(m) == desired_promo
                                && (int)file_of(to_sq(m)) == desired_file) {
                            chosen_move = m;
                            break;
                        }
                    }
                }

                if (chosen_move == MOVE_NONE) {
                    if (mover == env->learner_color) {
                        env->rewards[0] += env->reward_invalid_move;
                        env->invalid_actions_this_episode++;
                    }
                    env->pick_phase[mover_idx] = 0;
                    env->selected_square[mover_idx] = SQ_NONE;
                    env->valid_destinations[mover_idx].count = 0;
                } else {
                    game_result = apply_move_to_env(env, chosen_move, &is_timeout);
                    if (env->reward_repetition != 0.0f
                            && mover == env->learner_color
                            && env->undo_stack_ptr >= 4) {
                        uint8_t plies = env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull;
                        if (plies >= 4) {
                            Key current_key = env->pos.key;
                            for (int i = 4; i <= plies; i += 2) {
                                int idx = env->undo_stack_ptr - i;
                                if (idx >= 0 && env->undo_stack[idx].key == current_key) {
                                    env->rewards[0] += env->reward_repetition;
                                    break;
                                }
                            }
                        }
                    }
                    move_completed = 1;
                }
            }
        }
    }

    if (!move_completed) {
        if (env->chess_moves >= env->max_moves || env->undo_stack_ptr >= MAX_GAME_PLIES - 2) {
            game_result = 3;
            is_timeout = 1;
        } else {
            if (env->legal_moves.count == 0) {
                if (is_check(&env->pos, env->pos.sideToMove)) {
                    game_result = env->pos.sideToMove == CHESS_WHITE ? 1 : 2;
                } else {
                    game_result = 3;
                }
            } else if (is_insufficient_material(&env->pos)) {
                game_result = 3;
            } else if (env->enable_50_move_rule && env->pos.rule50 >= 100) {
                game_result = 3;
            } else if (env->enable_threefold_repetition && env->undo_stack_ptr >= 4) {
                uint8_t plies = env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull;
                if (plies >= 4) {
                    int repetitions = 0;
                    for (int i = 4; i <= plies; i += 2) {
                        int idx = env->undo_stack_ptr - i;
                        if (idx >= 0 && env->undo_stack[idx].key == env->pos.key) {
                            repetitions++;
                            if (repetitions >= 2) {
                                game_result = 3;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    if (game_result != 0) {
        env->terminals[0] = 1;
        env->game_result = game_result;
        float win_value = 0.0f;

        switch (game_result) {
            case 3:
                env->rewards[0] = env->reward_draw;
                win_value = 0.5f;
                env->log.draw_rate += 1.0f;
                if (is_timeout) {
                    env->log.timeout_rate += 1.0f;
                }
                env->white_score += 0.5f;
                env->black_score += 0.5f;
                env->learner_draws += 1.0f;
                strcpy(env->last_result, "Draw");
                break;
            case 1:
                env->black_score += 1.0f;
                if (env->learner_color == CHESS_WHITE) {
                    env->rewards[0] = -1.0f;
                    env->learner_losses += 1.0f;
                } else {
                    env->rewards[0] = 1.0f;
                    win_value = 1.0f;
                    env->learner_wins += 1.0f;
                }
                strcpy(env->last_result, "Black Wins");
                break;
            case 2:
                env->white_score += 1.0f;
                if (env->learner_color == CHESS_WHITE) {
                    env->rewards[0] = 1.0f;
                    win_value = 1.0f;
                    env->learner_wins += 1.0f;
                } else {
                    env->rewards[0] = -1.0f;
                    env->learner_losses += 1.0f;
                }
                strcpy(env->last_result, "White Wins");
                break;
            default:
                break;
        }

        env->episode_reward += env->rewards[0];
        env->log.episode_return += env->episode_reward;
        env->log.perf += win_value;
        env->log.score += win_value;
        env->log.chess_moves += env->chess_moves;
        env->log.episode_length += env->tick;
        env->log.invalid_action_rate += (env->tick > 0)
            ? ((float)env->invalid_actions_this_episode / (float)env->tick) : 0.0f;

        env->log.n += 1.0f;

        if (env->mode == CHESS_MODE_HUMAN) {
            env->show_game_end_popup = 1;
        } else {
            if (env->log_pgn && env->pgn_filename[0] != '\0') {
                env->pgn_game_number++;
                export_pgn_append(env, env->pgn_filename, 1);
            }
            c_reset(env);
        }
    } else {
        env->episode_reward += env->rewards[0];
    }

    if (!move_completed) {
        populate_observations(env);
    }
}

static Font load_piece_font(int cell_size, int* loaded) {
    const char* candidates[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Apple Symbols.ttf",
        "C:\\Windows\\Fonts\\seguisym.ttf"
    };

    int codepoints[] = {0x2654, 0x2655, 0x2656, 0x2657, 0x2658, 0x2659, 0x265A, 0x265B, 0x265C, 0x265D, 0x265E, 0x265F};
    Font font = (Font){0};
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
    
    Color outline = (color_of(pc) == CHESS_WHITE) 
        ? (Color){0, 0, 0, 220} 
        : (Color){255, 255, 255, 180};

    int draw_x = file * cell_size;
    int draw_y = (7 - rank) * cell_size;

    if (env->client && env->client->use_unicode_pieces) {
        float icon_size = cell_size * 0.85f;
        Vector2 pos = (Vector2){
            draw_x + (cell_size - icon_size) / 2.0f,
            draw_y + (cell_size - icon_size) / 2.0f - cell_size * 0.05f
        };
        const char* str = PIECE_FILLED[pc];
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx != 0 || dy != 0) {
                    Vector2 opos = (Vector2){pos.x + dx, pos.y + dy};
                    DrawTextEx(env->client->piece_font, str, opos, icon_size, 0, outline);
                }
            }
        }
        DrawTextEx(env->client->piece_font, str, pos, icon_size, 0, pc_color);
    } else {
        int x = draw_x + cell_size / 4;
        int y = draw_y + cell_size / 8;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx != 0 || dy != 0) {
                    DrawText(PIECE_CHARS[pc], x + dx, y + dy, cell_size / 2, outline);
                }
            }
        }
        DrawText(PIECE_CHARS[pc], x, y, cell_size / 2, pc_color);
    }
}

static void init_chess_client(Chess* env, int cell_size) {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    int board_size = 8 * cell_size;
    InitWindow(board_size, board_size + 80, "PufferLib Chess - AI vs Opponent");
    SetTargetFPS(env->render_fps > 0 ? env->render_fps : 30);
    env->client = (Client*)calloc(1, sizeof(Client));
    env->client->cell_size = cell_size;
    int font_loaded = 0;
    env->client->piece_font = load_piece_font(cell_size, &font_loaded);
    env->client->use_unicode_pieces = font_loaded;
    if (env->mode == CHESS_MODE_SELFPLAY) env->log_pgn_choice_made = 0;
}

void c_render(Chess* env) {
    const int cell_size = 64;
    const int board_size = 8 * cell_size;
    const int scoreboard_y = board_size + 10;
    static int paused = 0;
    static int frame_delay = 12;
    static int selected_sq = -1;
    
    if (env->client == NULL) {
        init_chess_client(env, cell_size);
    }
    
    if (IsKeyDown(KEY_ESCAPE) || WindowShouldClose()) { CloseWindow(); exit(0); }
    
    int flip_board = (env->mode == CHESS_MODE_HUMAN && env->human_color == CHESS_BLACK) ? 1 : 0;
    Vector2 mouse = GetMousePosition();
    int clicked = IsMouseButtonPressed(MOUSE_LEFT_BUTTON);
    
    if (IsKeyPressed(KEY_SPACE)) paused = !paused;
    if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) frame_delay = frame_delay > 4 ? frame_delay - 4 : 1;
    if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) frame_delay = frame_delay < 60 ? frame_delay + 4 : 60;
    
    if (!paused && env->mode == CHESS_MODE_HUMAN && env->human_color != -1 && !env->show_game_end_popup && clicked) {
        int file = (int)(mouse.x) / cell_size;
        int rank = 7 - ((int)(mouse.y) / cell_size);
        if (flip_board) { file = 7 - file; rank = 7 - rank; }
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
                    int is_timeout = 0;
                    apply_move_to_env(env, chosen, &is_timeout);
                    env->actions[0] = -1;
                }
                selected_sq = -1;
            }
        }
    }
    
    BeginDrawing();
    ClearBackground((Color){40, 40, 40, 255});
    
    if (env->mode == CHESS_MODE_HUMAN && env->show_game_end_popup) {
        int pw = 300, ph = 200;
        int px = (board_size - pw) / 2, py = (board_size - ph) / 2;
        DrawRectangle(px, py, pw, ph, (Color){60, 60, 60, 255});
        DrawRectangleLines(px, py, pw, ph, WHITE);
        DrawText("Game Over!", px + 70, py + 20, 24, WHITE);
        DrawText(env->last_result, px + 80, py + 55, 18, YELLOW);
        
        Rectangle save_btn = {px + 20, py + 110, 120, 35};
        Rectangle new_btn  = {px + 160, py + 110, 120, 35};
        DrawRectangleRec(save_btn, DARKGREEN);
        DrawRectangleLinesEx(save_btn, 2, WHITE);
        DrawText("Save PGN", px + 35, py + 120, 16, WHITE);
        DrawRectangleRec(new_btn, DARKBLUE);
        DrawRectangleLinesEx(new_btn, 2, WHITE);
        DrawText("New Game", px + 175, py + 120, 16, WHITE);
        
        if (clicked) {
            if (CheckCollisionPointRec(mouse, save_btn)) {
                char filename[64];
                snprintf(filename, sizeof(filename), "game_%d.pgn", (int)time(NULL));
                export_pgn_append(env, filename, 0);
                printf("Saved PGN to %s\n", filename);
            } else if (CheckCollisionPointRec(mouse, new_btn)) {
                c_reset(env);
            }
        }
    } else if (env->mode == CHESS_MODE_SELFPLAY && !env->log_pgn_choice_made) {
        int cx = board_size / 2;
        DrawText("Log PGN Files?", cx - 80, 180, 24, WHITE);
        DrawText("Games will be appended to a timestamped file", cx - 160, 220, 14, LIGHTGRAY);
        
        Rectangle yes_btn = {cx - 70, 270, 140, 40};
        Rectangle no_btn  = {cx - 70, 330, 140, 40};
        DrawRectangleRec(yes_btn, DARKGREEN);
        DrawRectangleLinesEx(yes_btn, 2, WHITE);
        DrawText("Yes, Log PGN", cx - 55, 282, 16, WHITE);
        DrawRectangleRec(no_btn, MAROON);
        DrawRectangleLinesEx(no_btn, 2, WHITE);
        DrawText("No Logging", cx - 45, 342, 16, WHITE);
        
        if (clicked) {
            if (CheckCollisionPointRec(mouse, yes_btn)) {
                env->log_pgn = 1;
                env->log_pgn_choice_made = 1;
                env->pgn_game_number = 0;
                snprintf(env->pgn_filename, sizeof(env->pgn_filename), "run_%d_pgns.pgn", (int)time(NULL));
                printf("PGN logging enabled: %s\n", env->pgn_filename);
            } else if (CheckCollisionPointRec(mouse, no_btn)) {
                env->log_pgn = 0;
                env->log_pgn_choice_made = 1;
                printf("PGN logging disabled\n");
            }
        }
    } else if (env->mode == CHESS_MODE_HUMAN && env->human_color == -1) {
        int cx = board_size / 2;
        DrawText("Choose Your Color", cx - 100, 200, 24, WHITE);
        
        Rectangle white_btn = {cx - 60, 280, 120, 40};
        Rectangle black_btn = {cx - 60, 340, 120, 40};
        DrawRectangleRec(white_btn, LIGHTGRAY);
        DrawRectangleLinesEx(white_btn, 2, BLACK);
        DrawText("Play White", cx - 45, 292, 18, BLACK);
        DrawRectangleRec(black_btn, GRAY);
        DrawRectangleLinesEx(black_btn, 2, BLACK);
        DrawText("Play Black", cx - 45, 352, 18, WHITE);
        
        if (clicked) {
            if (CheckCollisionPointRec(mouse, white_btn)) {
                env->human_color = CHESS_WHITE;
                env->learner_color = CHESS_BLACK;
            } else if (CheckCollisionPointRec(mouse, black_btn)) {
                env->human_color = CHESS_BLACK;
                env->learner_color = CHESS_WHITE;
            }
        }
    } else {
        Bitboard selected_destinations = 0;
        if (selected_sq != -1) {
            for (int i = 0; i < env->legal_moves.count; i++) {
                Move m = env->legal_moves.moves[i].move;
                if ((int)from_sq(m) == selected_sq) {
                    selected_destinations |= sq_bb(to_sq(m));
                }
            }
        }
        int selected_file = -1;
        int selected_rank = -1;
        if (selected_sq != -1) {
            selected_file = file_of((Square)selected_sq);
            selected_rank = rank_of((Square)selected_sq);
        }
        for (int rank = 0; rank < 8; rank++) {
            for (int file = 0; file < 8; file++) {
                Color sq_color = ((rank + file) % 2 == 1) ? (Color){240, 217, 181, 255} : (Color){181, 136, 99, 255};
                int draw_file = flip_board ? (7 - file) : file;
                int draw_rank = flip_board ? (7 - rank) : rank;
                int draw_x = draw_file * cell_size;
                int draw_y = (7 - draw_rank) * cell_size;
                DrawRectangle(draw_x, draw_y, cell_size, cell_size, sq_color);

                if (selected_sq != -1 && selected_file == file && selected_rank == rank) {
                    DrawRectangleLines(draw_x, draw_y, cell_size, cell_size, (Color){255, 215, 0, 255});
                }
                if (selected_sq != -1 && (selected_destinations & sq_bb(make_square(file, rank)))) {
                    DrawRectangleLines(draw_x + 2, draw_y + 2, cell_size - 4, cell_size - 4, (Color){0, 200, 0, 255});
                }
            }
        }
        for (int pt = PAWN; pt <= KING; pt++) {
            Bitboard bb = pieces_p(&env->pos, pt);
            while (bb) {
                Square sq = pop_lsb(&bb);
                Piece pc = piece_on(&env->pos, sq);
                int f = file_of(sq), r = rank_of(sq);
                int draw_f = flip_board ? (7 - f) : f;
                int draw_r = flip_board ? (7 - r) : r;
                draw_piece(env, pc, draw_f, draw_r, cell_size);
            }
        }
        
        char buf[128];
        snprintf(buf, sizeof(buf), "White: %.1f  Black: %.1f", env->white_score, env->black_score);
        DrawText(buf, 10, scoreboard_y, 20, WHITE);
        
        snprintf(buf, sizeof(buf), "Learner: %.0f-%.0f-%.0f (W-L-D)", env->learner_wins, env->learner_losses, env->learner_draws);
        DrawText(buf, 10, scoreboard_y + 30, 16, GREEN);
        
        // Captured pieces
        int cap_y = scoreboard_y + 75;
        char wcap[128] = "White captured: ";
        char bcap[128] = "Black captured: ";
        int wl = strlen(wcap), bl = strlen(bcap);
        const char* pc_chars = "PNBRQK";
        for (int pt = 0; pt < 6; pt++) {
            for (int i = 0; i < env->white_captured[pt]; i++) wcap[wl++] = pc_chars[pt];
            for (int i = 0; i < env->black_captured[pt]; i++) bcap[bl++] = pc_chars[pt];
        }
        wcap[wl] = '\0'; bcap[bl] = '\0';
        DrawText(wcap, 10, cap_y, 14, (Color){240, 217, 181, 255});
        DrawText(bcap, 10, cap_y + 18, 14, (Color){100, 100, 100, 255});
        
        if (env->last_result[0] != '\0') {
            Color rc = YELLOW;
            if (strstr(env->last_result, "White")) rc = (Color){240, 217, 181, 255};
            else if (strstr(env->last_result, "Black")) rc = (Color){100, 100, 100, 255};
            DrawText(env->last_result, 10, cap_y + 40, 18, rc);
        }
        
        snprintf(buf, sizeof(buf), "Move: %d", env->chess_moves);
        DrawText(buf, board_size - 100, scoreboard_y, 18, LIGHTGRAY);
        
        if (env->mode != CHESS_MODE_HUMAN) {
            DrawText(env->learner_color == CHESS_WHITE ? "Learner: White" : "Learner: Black",
                     board_size - 120, scoreboard_y + 25, 16, LIGHTGRAY);
        }
        
        int btn_w = 36;
        int btn_h = 24;
        int btn_y = scoreboard_y + 45;
        int btn_x = env->mode == CHESS_MODE_HUMAN ? board_size / 2 - 100 : board_size / 2 - 70;
        Rectangle minus_btn = {btn_x, btn_y, btn_w, btn_h};
        Rectangle pause_btn = {btn_x + btn_w + 5, btn_y, btn_w + 10, btn_h};
        Rectangle plus_btn = {btn_x + 2 * btn_w + 20, btn_y, btn_w, btn_h};
        DrawRectangleRec(minus_btn, DARKGRAY);
        DrawRectangleLinesEx(minus_btn, 2, LIGHTGRAY);
        DrawText("-", btn_x + 14, btn_y + 4, 20, WHITE);
        DrawRectangleRec(pause_btn, paused ? MAROON : DARKGREEN);
        DrawRectangleLinesEx(pause_btn, 2, LIGHTGRAY);
        DrawText(paused ? ">" : "||", btn_x + btn_w + 14, btn_y + 4, 18, WHITE);
        DrawRectangleRec(plus_btn, DARKGRAY);
        DrawRectangleLinesEx(plus_btn, 2, LIGHTGRAY);
        DrawText("+", btn_x + 2 * btn_w + 32, btn_y + 4, 20, WHITE);
        int speed_val = frame_delay > 0 ? 60 / frame_delay : 60;
        char speed_buf[32];
        snprintf(speed_buf, sizeof(speed_buf), "%dx", speed_val > 0 ? speed_val : 1);
        DrawText(speed_buf, btn_x + 3 * btn_w + 30, btn_y + 4, 18, paused ? RED : LIGHTGRAY);
        
        Rectangle restart_btn = {0, 0, 0, 0};
        if (env->mode == CHESS_MODE_HUMAN) {
            restart_btn = (Rectangle){board_size - 60, minus_btn.y, 55, minus_btn.height};
            DrawRectangleRec(restart_btn, MAROON);
            DrawRectangleLinesEx(restart_btn, 2, LIGHTGRAY);
            DrawText("Exit", board_size - 53, minus_btn.y + 4, 16, WHITE);
        }
        
        if (paused) {
            DrawRectangle(0, 0, board_size, board_size, (Color){0, 0, 0, 120});
            DrawText("PAUSED", board_size / 2 - 60, board_size / 2 - 15, 30, RED);
        }
        
        if (clicked) {
            if (CheckCollisionPointRec(mouse, minus_btn)) frame_delay = frame_delay < 60 ? frame_delay + 4 : 60;
            if (CheckCollisionPointRec(mouse, pause_btn)) paused = !paused;
            if (CheckCollisionPointRec(mouse, plus_btn))  frame_delay = frame_delay > 4 ? frame_delay - 4 : 1;
            if (env->mode == CHESS_MODE_HUMAN && CheckCollisionPointRec(mouse, restart_btn)) c_reset(env);
        }
    }
    
    EndDrawing();
    
    // Pause: block thread so c_step doesn't advance. Must call BeginDrawing/EndDrawing
    // each iteration so raylib pumps input events and the window stays responsive.
    while (paused) {
        BeginDrawing();
        ClearBackground((Color){40, 40, 40, 255});
        for (int rank = 0; rank < 8; rank++) {
            for (int file = 0; file < 8; file++) {
                Color sq_color = ((rank + file) % 2 == 1) ? (Color){240, 217, 181, 255} : (Color){181, 136, 99, 255};
                int draw_file = flip_board ? (7 - file) : file;
                int draw_rank = flip_board ? (7 - rank) : rank;
                int draw_x = draw_file * cell_size;
                int draw_y = (7 - draw_rank) * cell_size;
                DrawRectangle(draw_x, draw_y, cell_size, cell_size, sq_color);
            }
        }
        for (int pt = PAWN; pt <= KING; pt++) {
            Bitboard bb = pieces_p(&env->pos, pt);
            while (bb) {
                Square sq = pop_lsb(&bb);
                Piece pc = piece_on(&env->pos, sq);
                int f = file_of(sq), r = rank_of(sq);
                int draw_f = flip_board ? (7 - f) : f;
                int draw_r = flip_board ? (7 - r) : r;
                draw_piece(env, pc, draw_f, draw_r, cell_size);
            }
        }
        
        DrawRectangle(0, 0, board_size, board_size, (Color){0, 0, 0, 120});
        DrawText("PAUSED", board_size / 2 - 60, board_size / 2 - 15, 30, RED);
        
        int p_bw = 36;
        int p_bh = 24;
        int p_by = scoreboard_y + 45;
        int p_bx = env->mode == CHESS_MODE_HUMAN ? board_size / 2 - 100 : board_size / 2 - 70;
        Rectangle p_minus = {p_bx, p_by, p_bw, p_bh};
        Rectangle p_pause = {p_bx + p_bw + 5, p_by, p_bw + 10, p_bh};
        Rectangle p_plus = {p_bx + 2 * p_bw + 20, p_by, p_bw, p_bh};
        DrawRectangleRec(p_minus, DARKGRAY);
        DrawRectangleLinesEx(p_minus, 2, LIGHTGRAY);
        DrawText("-", p_bx + 14, p_by + 4, 20, WHITE);
        DrawRectangleRec(p_pause, MAROON);
        DrawRectangleLinesEx(p_pause, 2, LIGHTGRAY);
        DrawText(">", p_bx + p_bw + 14, p_by + 4, 18, WHITE);
        DrawRectangleRec(p_plus, DARKGRAY);
        DrawRectangleLinesEx(p_plus, 2, LIGHTGRAY);
        DrawText("+", p_bx + 2 * p_bw + 32, p_by + 4, 20, WHITE);
        int p_speed_val = frame_delay > 0 ? 60 / frame_delay : 60;
        char p_speed_buf[32];
        snprintf(p_speed_buf, sizeof(p_speed_buf), "%dx", p_speed_val > 0 ? p_speed_val : 1);
        DrawText(p_speed_buf, p_bx + 3 * p_bw + 30, p_by + 4, 18, RED);
        
        EndDrawing();
        
        if (IsKeyDown(KEY_ESCAPE) || WindowShouldClose()) { CloseWindow(); exit(0); }
        if (IsKeyPressed(KEY_SPACE)) { paused = 0; break; }
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            Vector2 pm = GetMousePosition();
            if (CheckCollisionPointRec(pm, p_pause)) { paused = 0; break; }
            if (CheckCollisionPointRec(pm, p_minus)) frame_delay = frame_delay < 60 ? frame_delay + 4 : 60;
            if (CheckCollisionPointRec(pm, p_plus))  frame_delay = frame_delay > 4 ? frame_delay - 4 : 1;
        }
    }
    
    if (frame_delay > 1 && !(env->mode == CHESS_MODE_HUMAN && env->human_color != -1 && env->pos.sideToMove == env->human_color)) {
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
    env->fen_curriculum = NULL;
    env->num_fens = 0;
}