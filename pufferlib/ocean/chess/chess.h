#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <spawn.h>
#include "raylib.h"
#include "fathom/tbprobe.h"

// Curriculum annealing globals (shared across all envs)
static int _g_sf_random_pct = -1;          // Integer used in game logic
static float _g_sf_random_pct_f = -1.0f;   // Float for smooth annealing
static float _g_ema_wr = 0.0f;             // EMA of combined learner win rate
static int _g_annealing_games = 0;         // Total games (for warmup)

#define EMA_ALPHA 0.001f                    // ~1000 game effective window
#define ANNEAL_WR_THRESHOLD 0.15f           // Start annealing above this
#define ANNEAL_RATE 0.0001f                 // Per-game decrease per unit excess WR

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


#define MAX_TOKENS 64
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

static const char* PIECE_FILLED[] = {
    "",
    "♟", "♞", "♝", "♜", "♛", "♚",
    "", "",
    "♟", "♞", "♝", "♜", "♛", "♚"
};

static const int PIECE_VALUES_CP[7] = {0, 100, 320, 330, 500, 900, 0};

static const int PHASE_VALUES[7] = {0, 0, 1, 1, 2, 4, 0};
#define TOTAL_PHASE 24

static const int SUNFISH_PIECE_VALUES_CP[7] = {0, 136, 782, 830, 1289, 2529, 32000};

static const int SunfishPawnPST[64] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    15, 31, 20, 14, 23, 11, 37, 24,
    -1, -3, 15, 26, 1, 10, -7, -9,
    8, -1, -5, 13, 24, 11, -10, 3,
    -9, -18, 8, 32, 43, 25, -4, -16,
    -9, -13, -40, 22, 26, -40, 1, -22,
    2, 0, 15, 3, 11, 22, 11, -1,
    0, 0, 0, 0, 0, 0, 0, 0
};

static const int SunfishKnightPST[64] = {
    -200, -80, -53, -32, -32, -53, -80, -200,
    -67, -21, 6, 37, 37, 6, -21, -67,
    -11, 28, 63, 55, 55, 63, 28, -11,
    -29, 13, 42, 52, 52, 42, 13, -29,
    -28, 5, 41, 47, 47, 41, 5, -28,
    -64, -20, 4, 19, 19, 4, -20, -64,
    -79, -39, -24, -9, -9, -24, -39, -79,
    -169, -96, -80, -79, -79, -80, -96, -169
};

static const int SunfishBishopPST[64] = {
    -48, -3, -12, -25, -25, -12, -3, -48,
    -21, -19, 10, -6, -6, 10, -19, -21,
    -17, 4, -1, 8, 8, -1, 4, -17,
    -7, 30, 23, 28, 28, 23, 30, -7,
    1, 8, 26, 37, 37, 26, 8, 1,
    -8, 24, -3, 15, 15, -3, 24, -8,
    -18, 7, 14, 3, 3, 14, 7, -18,
    -44, -4, -11, -28, -28, -11, -4, -44
};

static const int SunfishRookPST[64] = {
    -22, -24, -6, 4, 4, -6, -24, -22,
    -8, 6, 10, 12, 12, 10, 6, -8,
    -24, -4, 4, 10, 10, 4, -4, -24,
    -24, -12, -1, 6, 6, -1, -12, -24,
    -13, -5, -4, -6, -6, -4, -5, -13,
    -21, -7, 3, -1, -1, 3, -7, -21,
    -18, -10, -5, 9, 9, -5, -10, -18,
    -24, -13, -7, 2, 2, -7, -13, -24
};

static const int SunfishQueenPST[64] = {
    -2, -2, 1, -2, -2, 1, -2, -2,
    -5, 6, 10, 8, 8, 10, 6, -5,
    -4, 10, 6, 8, 8, 6, 10, -4,
    0, 14, 12, 5, 5, 12, 14, 0,
    4, 5, 9, 8, 8, 9, 5, 4,
    -3, 6, 13, 7, 7, 13, 6, -3,
    -3, 5, 8, 12, 12, 8, 5, -3,
    3, -5, -5, 4, 4, -5, -5, 3
};

static const int SunfishKingPST[64] = {
    6, 8, 4, 0, 0, 4, 8, 6,
    8, 12, 6, 2, 2, 6, 12, 8,
    12, 15, 8, 3, 3, 8, 15, 12,
    14, 17, 11, 6, 6, 11, 17, 15,
    16, 19, 13, 10, 10, 13, 19, 16,
    19, 25, 16, 12, 12, 16, 25, 19,
    27, 30, 24, 18, 18, 24, 30, 27,
    27, 32, 27, 19, 19, 27, 32, 27
};

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
static Bitboard RookTable[64 * 4096];  // Rooks need more entries (up to 12 bits)
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
    uint16_t gamePly;
    Key key;
    int16_t materialScore;
    int16_t psqtScore;
    int16_t psqtScore_mg;
    int16_t psqtScore_eg;
    int8_t gamePhase;
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

/*enum {
    // Relational NNUE tokens
    O_TOKEN_COUNT = 0,                    
    O_TOKEN_DATA  = 2,                  
    // Meta data
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
    O_SELF_CHECK = 1077,
    O_OPP_CHECK = 1078,
    O_RULE50 = 1079,
    O_REPETITION = 1080,
    O_PASS_VALID = 1081,
    OBS_SIZE = 1082
};

#define PASS_ACTION 96
#define NUM_ACTIONS 97

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
    float material_score;
    float positional_score;
    float n;
    float white_winrate;
    float black_winrate;
    float white_lossrate;
    float black_lossrate;
    float draw_by_stalemate;
    float draw_by_insufficient;
    float draw_by_50move;
    float draw_by_repetition;
    float opponent_winrate;
    float stockfish_random_pct;
    float stockfish_query_pct;
    float ema_winrate;
    float tutor_piece_match;
    float tutor_move_match;
    float tutor_total;
    float syzygy_probes;
    float syzygy_wins;
    float syzygy_draws;
    float syzygy_reward_total;
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
    int16_t psqtScore_mg;
    int16_t psqtScore_eg;
    int8_t gamePhase;
    Key key;
    uint8_t pliesFromNull;
} UndoInfo;

typedef struct {
    Log log;
    Client* client;
    uint8_t* observations;
    double* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    
    Position pos;
    MoveList legal_moves;
    ChessColor legal_moves_side;
    Key legal_moves_key;
    int game_result;
    int tick;
    int chess_moves;
    int max_moves;
    float reward_draw;
    float episode_reward;  // Accumulate episode reward (like g2048)
    int render_fps;
    int selfplay;
    int human_play;
    int random_bot;
    int stockfish_bot;
    int stockfish_limit_strength;
    int stockfish_elo;
    int stockfish_movetime_ms;
    int stockfish_depth;
    int stockfish_random_pct;
    int stockfish_query_pct;
    FILE* stockfish_in;
    FILE* stockfish_out;
    int stockfish_pid;
    int stockfish_ready;
    
    char starting_fen[128];
    char** fen_curriculum;
    float fen_curric_pct;
    int num_fens;
    int random_fen;

    char** fen_curriculum_dm;   // DeepMind FEN curriculum
    int num_fens_dm;            // Count of DeepMind FENs
    float deepmind_fen_pct;     // Fraction of curriculum resets that use DeepMind FENs
    
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
    float reward_check;
    float reward_mate;
    float reward_syzygy;
    int syzygy_wdl_prev;  // previous WDL probe result for delta reward
    
    int last_see_value;
    Move last_move;
    
    int enable_50_move_rule;
    int enable_threefold_repetition;
    
    int learner_color; // 0 for White, 1 for Black
    int opp_in_check;
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
    int reset_requested;
    
    int log_pgn;
    int log_pgn_choice_made;
    char pgn_filename[128];
    int pgn_game_number;
    
    int white_captured[6];
    int black_captured[6];
    
    int debug_mode;

    // Move tutor (expert-guided training from pre-computed Stockfish data)
    uint16_t* tutor_moves_dm;      // Pointer to global packed-move array
    uint16_t tutor_target;          // Packed target for current episode (0 = none)
    int tutor_phase;                // 0=piece, 1=dest, 2=done
    float reward_tutor_piece;       // Bonus for matching expert's source square
    float reward_tutor_move;        // Bonus for matching expert's destination
    float reward_tutor_wrong;       // Penalty for wrong move (optional, default 0)
    int tutor_only_mode;            // If 1, episode ends after first move attempt

#ifdef CHESS_DEBUG_BUILD
    int debug_paused;
    int debug_history_idx;
    int debug_history_count;
    int debug_view_player;
    int debug_selected_plane;
    #define DEBUG_HISTORY_SIZE 100
    uint8_t debug_obs_history[DEBUG_HISTORY_SIZE][OBS_SIZE * 2];
    Position debug_pos_history[DEBUG_HISTORY_SIZE];
    int debug_pick_phase_history[DEBUG_HISTORY_SIZE][2];
    Square debug_selected_sq_history[DEBUG_HISTORY_SIZE][2];
    int debug_actions_history[DEBUG_HISTORY_SIZE][2];
    Move debug_last_move_history[DEBUG_HISTORY_SIZE];
    int debug_chess_moves_history[DEBUG_HISTORY_SIZE];
    float debug_rewards_history[DEBUG_HISTORY_SIZE];
#endif
} Chess;

void pos_set(Position* pos, const char* fen);
void do_move(Position* pos, Move m, UndoInfo* undo_stack, int* undo_stack_ptr);
void undo_move(Position* pos, Move m, UndoInfo* undo_stack, int* undo_stack_ptr);
void export_pgn_append(Chess* env, const char* filename, int append);

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


static inline Bitboard rook_attacks_slow(Square s, Bitboard occupied) {
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

static inline Bitboard rook_attacks_bb(Square s, Bitboard occupied) {
    occupied &= RookMasks[s];
    return RookAttacks[s][(occupied * RookMagics[s]) >> RookShifts[s]];
}

static inline Bitboard bishop_attacks_bb(Square s, Bitboard occupied) {
    occupied &= BishopMasks[s];
    return BishopAttacks[s][(occupied * BishopMagics[s]) >> BishopShifts[s]];
}

static inline Bitboard bishop_attacks_slow(Square s, Bitboard occupied) {
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
static bool syzygy_initialized = false;

static void init_syzygy(const char* path) {
    if (syzygy_initialized || path == NULL || path[0] == 0) return;
    if (tb_init(path)) {
        fprintf(stderr, "Syzygy tablebases loaded from %s (TB_LARGEST=%u)\n", path, TB_LARGEST);
        syzygy_initialized = true;
    } else {
        fprintf(stderr, "WARNING: Failed to load Syzygy tablebases from %s\n", path);
    }
}

// Probe Syzygy WDL for current position. Returns TB_WIN/TB_DRAW/TB_LOSS or -1 on failure.
static int probe_syzygy_wdl(Position* pos) {
    if (!syzygy_initialized) return -1;
    // Fast checks first: most positions have >5 pieces or castling rights
    int piece_count = popcount(pos->byTypeBB[0]);
    if (piece_count > (int)TB_LARGEST) return -1;  // too many pieces
    if (pos->castlingRights != NO_CASTLING) return -1;  // no castling in tablebases
    // Fathom expects consistent bitboards and exactly one king per side.
    Bitboard white = pos->byColorBB[CHESS_WHITE];
    Bitboard black = pos->byColorBB[CHESS_BLACK];
    Bitboard kings = pos->byTypeBB[KING];
    Bitboard occ = white | black;
    Bitboard occ_types = 0ULL;
    for (int pt = PAWN; pt <= KING; pt++) {
        occ_types |= pos->byTypeBB[pt];
    }
    if ((white & black) != 0ULL
        || popcount(white & kings) != 1
        || popcount(black & kings) != 1
        || pos->byTypeBB[0] != occ
        || occ_types != occ) {
        static int syzygy_invalid_warns = 0;
        if (syzygy_invalid_warns < 8) {
            fprintf(stderr,
                "WARNING: Skipping Syzygy probe on invalid position "
                "(white=%llx black=%llx kings=%llx all=%llx)\n",
                (unsigned long long)white,
                (unsigned long long)black,
                (unsigned long long)kings,
                (unsigned long long)pos->byTypeBB[0]);
            syzygy_invalid_warns++;
        }
        return -1;
    }
    unsigned ep = (pos->epSquare == SQ_NONE) ? 0 : pos->epSquare;
    unsigned result = tb_probe_wdl(
        pos->byColorBB[CHESS_WHITE],
        pos->byColorBB[CHESS_BLACK],
        pos->byTypeBB[KING],
        pos->byTypeBB[QUEEN],
        pos->byTypeBB[ROOK],
        pos->byTypeBB[BISHOP],
        pos->byTypeBB[KNIGHT],
        pos->byTypeBB[PAWN],
        pos->rule50,
        pos->castlingRights,
        ep,
        pos->sideToMove == CHESS_WHITE
    );
    if (result == TB_RESULT_FAILED) return -1;
    return (int)result;
}

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
    int errors = 0;
    
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
            Bitboard attacks = bishop_attacks_slow(sq, occ);
            uint64_t idx = (occ * BishopMagics[sq]) >> BishopShifts[sq];
            
            if (table_ptr[idx] != 0 && table_ptr[idx] != attacks) {
                printf("BAD MAGIC sq=%d idx=%lu\n", sq, idx);
                errors++;
            }
            table_ptr[idx] = attacks;
        }
        table_ptr += num_entries;
    }
    
    if (errors) {
        printf("Bishop magic init FAILED with %d errors\n", errors);
        exit(1);
    }
}
static void test_bishop_magics(void) {
    printf("Testing bishop magics...\n");
    int errors = 0;
    
    // Test all squares with random occupancies
    for (Square sq = 0; sq < 64; sq++) {
        for (int test = 0; test < 1000; test++) {
            Bitboard occ = prng_rand() & prng_rand(); // Random sparse occupancy
            Bitboard fast = bishop_attacks_bb(sq, occ);
            Bitboard slow = bishop_attacks_slow(sq, occ);
            
            if (fast != slow) {
                printf("MISMATCH sq=%d occ=%lu fast=%lu slow=%lu\n", sq, occ, fast, slow);
                errors++;
            }
        }
    }
    
    if (errors == 0) {
        printf("Bishop magics: ALL TESTS PASSED\n");
    } else {
        printf("Bishop magics: %d FAILURES\n", errors);
        exit(1);
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
    int errors = 0;
    
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
            Bitboard attacks = rook_attacks_slow(sq, occ);
            uint64_t idx = (occ * RookMagics[sq]) >> RookShifts[sq];
            
            if (table_ptr[idx] != 0 && table_ptr[idx] != attacks) {
                printf("BAD ROOK MAGIC sq=%d idx=%lu\n", sq, idx);
                errors++;
            }
            table_ptr[idx] = attacks;
        }
        table_ptr += num_entries;
    }
    
    if (errors) {
        printf("Rook magic init FAILED with %d errors\n", errors);
        exit(1);
    }
}

static void test_rook_magics(void) {
    printf("Testing rook magics...\n");
    int errors = 0;
    
    for (Square sq = 0; sq < 64; sq++) {
        for (int test = 0; test < 1000; test++) {
            Bitboard occ = prng_rand() & prng_rand();
            Bitboard fast = rook_attacks_bb(sq, occ);
            Bitboard slow = rook_attacks_slow(sq, occ);
            
            if (fast != slow) {
                printf("ROOK MISMATCH sq=%d\n", sq);
                errors++;
            }
        }
    }
    
    printf(errors ? "Rook magics: %d FAILURES\n" : "Rook magics: ALL TESTS PASSED\n", errors);
    if (errors) exit(1);
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
    //test_bishop_magics();
    init_rook_magics();
    //test_rook_magics();
    bitboards_initialized = true;
}


static inline int get_pst_value(Piece pc, Square sq);
static inline int get_pst_mg(Piece pc, Square sq);
static inline int get_pst_eg(Piece pc, Square sq);
static inline int blend_pst(int mg, int eg, int phase);

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
    pos->psqtScore_mg = 0;
    pos->psqtScore_eg = 0;
    pos->gamePhase = 0;
    
    for (Square sq = SQ_A1; sq <= SQ_H8; sq++) {
        Piece pc = pos->board[sq];
        if (pc == NO_PIECE) continue;
        
        int pt = type_of_p(pc);
        ChessColor c = color_of(pc);
        int sign = (c == CHESS_WHITE) ? 1 : -1;
        
        pos->materialScore += sign * PIECE_VALUES_CP[pt];
        pos->psqtScore_mg += sign * get_pst_mg(pc, sq);
        pos->psqtScore_eg += sign * get_pst_eg(pc, sq);
        pos->gamePhase += PHASE_VALUES[pt];
    }
    
    if (pos->gamePhase > TOTAL_PHASE) pos->gamePhase = TOTAL_PHASE;
    pos->psqtScore = blend_pst(pos->psqtScore_mg, pos->psqtScore_eg, pos->gamePhase);
    
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
    
    // King captures are illegal in chess; never generate moves to the king square.
    Bitboard enemies = pieces_c(pos, them) & ~pieces_cp(pos, them, KING);
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
    ChessColor them = !us;
    Bitboard pieces_bb = pieces_cp(pos, us, pt);
    // Exclude opponent king square from all targets; checkmate is handled by no-legal-move logic.
    Bitboard target = ~pieces_c(pos, us) & ~pieces_cp(pos, them, KING);
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
    return (pawn_attacks_bb(CHESS_WHITE, sq) & pieces_cp(pos, CHESS_BLACK, PAWN) & occupied)
         | (pawn_attacks_bb(CHESS_BLACK, sq) & pieces_cp(pos, CHESS_WHITE, PAWN) & occupied)
         | (knight_attacks_bb(sq) & pieces_p(pos, KNIGHT) & occupied)
         | (king_attacks_bb(sq) & pieces_p(pos, KING) & occupied)
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

static Bitboard compute_pinned(Position* pos, ChessColor c) {
    Bitboard pinned = 0;
    Bitboard our_pieces = pieces_c(pos, c);
    Bitboard king_bb = pieces_cp(pos, c, KING);
    if (!king_bb) return 0;
    
    Square ksq = lsb(king_bb);
    ChessColor them = !c;
    Bitboard occupied = pieces(pos);
    
    // Potential diagonal pinners
    Bitboard diag_pinners = (pieces_cp(pos, them, BISHOP) | pieces_cp(pos, them, QUEEN)) 
                          & bishop_attacks_bb(ksq, 0);  // Empty board attacks
    
    while (diag_pinners) {
        Square pinner_sq = pop_lsb(&diag_pinners);
        Bitboard between = BetweenBB[ksq][pinner_sq] & occupied;
        if (popcount(between) == 1) {
            pinned |= between & our_pieces;
        }
    }
    
    // Potential rook/file pinners
    Bitboard rook_pinners = (pieces_cp(pos, them, ROOK) | pieces_cp(pos, them, QUEEN)) 
                          & rook_attacks_bb(ksq, 0);  // Empty board attacks
    
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
    
    // King moves always need full check
    if (from == ksq) {
        if (mt == CASTLING) {
            // Castling checks path squares
            ChessColor them = !us;
            if (is_check(pos, us)) return false;
            Square mid = (from + to) / 2;
            Bitboard occ = pieces(pos) ^ sq_bb(from);
            if (attackers_to_sq(pos, mid, occ) & pieces_c(pos, them)) return false;
            if (attackers_to_sq(pos, to, occ) & pieces_c(pos, them)) return false;
            return true;
        }
        // Regular king move - check destination isn't attacked
        Bitboard occ = pieces(pos) ^ sq_bb(from);
        return !(attackers_to_sq(pos, to, occ) & pieces_c(pos, !us));
    }
    
    // En passant is tricky - always do full check
    if (mt == ENPASSANT) {
        Bitboard occ = pieces(pos) ^ sq_bb(from) ^ sq_bb(to);
        Square capsq = to + (us == CHESS_WHITE ? -8 : 8);
        occ ^= sq_bb(capsq);
        return !(attackers_to_sq(pos, ksq, occ) & pieces_c(pos, !us));
    }
    
    // If piece is not pinned, move is legal
    if (!(pinned & sq_bb(from))) {
        return true;
    }
    
    // Pinned piece can only move along the pin ray
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
    ChessColor us = pos->sideToMove;
    Bitboard king_bb = pieces_cp(pos, us, KING);
    Square ksq = king_bb ? lsb(king_bb) : SQ_NONE;
    Bitboard pinned = compute_pinned(pos, us);
    bool in_check = is_check(pos, us);
    
    ml->count = 0;
    for (int i = 0; i < pseudo.count; i++) {
        Move m = pseudo.moves[i].move;
        
        // If in check, must do full verification
        if (in_check) {
            if (is_legal_move(pos, m)) {
                ml->moves[ml->count++] = pseudo.moves[i];
            }
        } else {
            // Not in check - use fast path
            if (is_legal_move_fast(pos, m, pinned, ksq, us)) {
                ml->moves[ml->count++] = pseudo.moves[i];
            }
        }
    }
}


static inline int get_material_value(Piece pc) {
    return PIECE_VALUES_CP[type_of_p(pc)];
}

static inline Piece least_valuable_attacker(Position* pos, Bitboard attackers, ChessColor side, Square* out_sq) {
    for (int pt = PAWN; pt <= KING; pt++) {
        Bitboard bb = attackers & pieces_cp(pos, side, pt);
        if (bb) {
            *out_sq = lsb(bb);
            return make_piece(side, pt);
        }
    }
    return NO_PIECE;
}

static int see_capture(Position* pos, Move m) {
    Square from = from_sq(m);
    Square to = to_sq(m);
    Piece attacker = piece_on(pos, from);
    Piece captured = piece_on(pos, to);
    
    if (captured == NO_PIECE) return 0;
    
    int gain[32];
    int depth = 0;
    ChessColor side = color_of(attacker);
    
    gain[0] = get_material_value(captured);
    
    Bitboard occupied = pieces(pos) ^ sq_bb(from);
    Bitboard attackers = attackers_to_sq(pos, to, occupied);
    
    Piece moving = attacker;
    side = !side;
    
    while (1) {
        depth++;
        gain[depth] = get_material_value(moving) - gain[depth - 1];
        
        if (gain[depth] < 0 && gain[depth - 1] < 0) break;
        
        attackers &= occupied;
        Bitboard side_attackers = attackers & pieces_c(pos, side);
        if (!side_attackers) break;
        
        Square attacker_sq;
        moving = least_valuable_attacker(pos, side_attackers, side, &attacker_sq);
        if (moving == NO_PIECE) break;
        
        occupied ^= sq_bb(attacker_sq);
        
        if (type_of_p(moving) == PAWN || type_of_p(moving) == BISHOP || type_of_p(moving) == QUEEN) {
            attackers |= bishop_attacks_bb(to, occupied) & (pieces_p(pos, BISHOP) | pieces_p(pos, QUEEN));
        }
        if (type_of_p(moving) == ROOK || type_of_p(moving) == QUEEN) {
            attackers |= rook_attacks_bb(to, occupied) & (pieces_p(pos, ROOK) | pieces_p(pos, QUEEN));
        }
        
        side = !side;
        if (depth >= 31) break;
    }
    
    while (--depth > 0) {
        gain[depth - 1] = -(-gain[depth - 1] > gain[depth] ? -gain[depth - 1] : gain[depth]);
    }
    
    return gain[0];
}

static int see_square(Position* pos, Square sq, Piece defender, Bitboard occupied) {
    ChessColor defender_color = color_of(defender);
    ChessColor attacker_color = !defender_color;
    
    Bitboard attackers = attackers_to_sq(pos, sq, occupied);
    Bitboard side_attackers = attackers & pieces_c(pos, attacker_color);
    
    if (!side_attackers) return 0;
    
    int gain[32];
    int depth = 0;
    
    gain[0] = -get_material_value(defender);
    
    Square attacker_sq;
    Piece moving = least_valuable_attacker(pos, side_attackers, attacker_color, &attacker_sq);
    if (moving == NO_PIECE) return 0;
    
    occupied ^= sq_bb(attacker_sq);
    ChessColor side = defender_color;
    
    while (1) {
        depth++;
        gain[depth] = get_material_value(moving) - gain[depth - 1];
        
        if (gain[depth] < 0 && gain[depth - 1] < 0) break;
        
        attackers &= occupied;
        Bitboard current_side_attackers = attackers & pieces_c(pos, side);
        if (!current_side_attackers) break;
        
        moving = least_valuable_attacker(pos, current_side_attackers, side, &attacker_sq);
        if (moving == NO_PIECE) break;
        
        occupied ^= sq_bb(attacker_sq);
        
        if (type_of_p(moving) == PAWN || type_of_p(moving) == BISHOP || type_of_p(moving) == QUEEN) {
            attackers |= bishop_attacks_bb(sq, occupied) & (pieces_p(pos, BISHOP) | pieces_p(pos, QUEEN));
        }
        if (type_of_p(moving) == ROOK || type_of_p(moving) == QUEEN) {
            attackers |= rook_attacks_bb(sq, occupied) & (pieces_p(pos, ROOK) | pieces_p(pos, QUEEN));
        }
        
        side = !side;
        if (depth >= 31) break;
    }
    
    while (--depth > 0) {
        gain[depth - 1] = -(-gain[depth - 1] > gain[depth] ? -gain[depth - 1] : gain[depth]);
    }
    
    return gain[0];
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

static inline int get_pst_mg(Piece pc, Square sq) {
    int pt = type_of_p(pc);
    ChessColor c = color_of(pc);
    Square s = (c == CHESS_WHITE) ? sq : (sq ^ 56);
    
    switch (pt) {
        case PAWN:   return PawnPST_MG[s];
        case KNIGHT: return KnightPST_MG[s];
        case BISHOP: return BishopPST_MG[s];
        case ROOK:   return RookPST_MG[s];
        case QUEEN:  return QueenPST_MG[s];
        case KING:   return KingPST_MG[s];
        default:     return 0;
    }
}

static inline int get_pst_eg(Piece pc, Square sq) {
    int pt = type_of_p(pc);
    ChessColor c = color_of(pc);
    Square s = (c == CHESS_WHITE) ? sq : (sq ^ 56);
    
    switch (pt) {
        case PAWN:   return PawnPST_EG[s];
        case KNIGHT: return KnightPST_EG[s];
        case BISHOP: return BishopPST_EG[s];
        case ROOK:   return RookPST_EG[s];
        case QUEEN:  return QueenPST_EG[s];
        case KING:   return KingPST_EG[s];
        default:     return 0;
    }
}

static inline int blend_pst(int mg, int eg, int phase) {
    if (phase > TOTAL_PHASE) phase = TOTAL_PHASE;
    if (phase < 0) phase = 0;
    return (mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE;
}

static inline int get_sunfish_pst_value(Piece pc, Square sq) {
    int pt = type_of_p(pc);
    ChessColor c = color_of(pc);
    int rank = rank_of(sq);
    int file = file_of(sq);
    
    const int* pst_table = NULL;
    int sunfish_idx;
    
    switch (pt) {
        case PAWN:   pst_table = SunfishPawnPST; break;
        case KNIGHT: pst_table = SunfishKnightPST; break;
        case BISHOP: pst_table = SunfishBishopPST; break;
        case ROOK:   pst_table = SunfishRookPST; break;
        case QUEEN:  pst_table = SunfishQueenPST; break;
        case KING:   pst_table = SunfishKingPST; break;
        default: return 0;
    }
    
    if (c == CHESS_WHITE) {
        sunfish_idx = (7 - rank) * 8 + file;
    } else {
        int flipped_rank = 7 - rank;
        int flipped_file = 7 - file;
        sunfish_idx = flipped_rank * 8 + flipped_file;
    }
    
    int pst_val = pst_table[sunfish_idx];
    return (c == CHESS_WHITE) ? pst_val : -pst_val;
}

static inline int32_t evaluate_sunfish(Position* pos) {
    int32_t eval = 0;
    
    for (Square sq = SQ_A1; sq <= SQ_H8; sq++) {
        Piece pc = pos->board[sq];
        if (pc == NO_PIECE) continue;
        
        int pt = type_of_p(pc);
        ChessColor c = color_of(pc);
        int sign = (c == CHESS_WHITE) ? 1 : -1;
        
        eval += sign * SUNFISH_PIECE_VALUES_CP[pt];
        eval += get_sunfish_pst_value(pc, sq);
    }
    
    return eval;
}

void do_move(Position* pos, Move m, UndoInfo* undo_stack, int* undo_stack_ptr) {
    if (m == MOVE_NULL) {
        undo_stack[*undo_stack_ptr].captured = NO_PIECE;
        undo_stack[*undo_stack_ptr].castlingRights = pos->castlingRights;
        undo_stack[*undo_stack_ptr].epSquare = pos->epSquare;
        undo_stack[*undo_stack_ptr].rule50 = pos->rule50;
        undo_stack[*undo_stack_ptr].materialScore = pos->materialScore;
        undo_stack[*undo_stack_ptr].psqtScore = pos->psqtScore;
        undo_stack[*undo_stack_ptr].psqtScore_mg = pos->psqtScore_mg;
        undo_stack[*undo_stack_ptr].psqtScore_eg = pos->psqtScore_eg;
        undo_stack[*undo_stack_ptr].gamePhase = pos->gamePhase;
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
    undo_stack[*undo_stack_ptr].psqtScore_mg = pos->psqtScore_mg;
    undo_stack[*undo_stack_ptr].psqtScore_eg = pos->psqtScore_eg;
    undo_stack[*undo_stack_ptr].gamePhase = pos->gamePhase;
    undo_stack[*undo_stack_ptr].key = pos->key;
    undo_stack[*undo_stack_ptr].pliesFromNull = (*undo_stack_ptr > 0) ? undo_stack[*undo_stack_ptr - 1].pliesFromNull + 1 : 0;
    (*undo_stack_ptr)++;
    
    int sign = (us == CHESS_WHITE) ? 1 : -1;
    int mat_delta = 0, pst_delta = 0;
    int mg_delta = 0, eg_delta = 0;
    int phase_delta = 0;
    
    pst_delta -= sign * get_pst_value(pc, from);
    mg_delta -= sign * get_pst_mg(pc, from);
    eg_delta -= sign * get_pst_eg(pc, from);
    
    if (captured != NO_PIECE) {
        int cap_sign = (color_of(captured) == CHESS_WHITE) ? 1 : -1;
        int cap_pt = type_of_p(captured);
        mat_delta -= cap_sign * get_material_value(captured);
        pst_delta -= cap_sign * get_pst_value(captured, to);
        mg_delta -= cap_sign * get_pst_mg(captured, to);
        eg_delta -= cap_sign * get_pst_eg(captured, to);
        phase_delta -= PHASE_VALUES[cap_pt];
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
        mg_delta += sign * get_pst_mg(pc, to);
        eg_delta += sign * get_pst_eg(pc, to);
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
        mg_delta -= sign * get_pst_mg(rook, rook_from);
        mg_delta += sign * get_pst_mg(rook, rook_to);
        eg_delta -= sign * get_pst_eg(rook, rook_from);
        eg_delta += sign * get_pst_eg(rook, rook_to);
        pos->key ^= zob.psq[rook][rook_to]; 
        
    } else if (move_type == ENPASSANT) {
        pos->key ^= zob.psq[pc][from];
        
        pos->board[from] = NO_PIECE;
        pos->board[to] = pc;
        pos->byTypeBB[pt] ^= sq_bb(from) ^ sq_bb(to);
        pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
        pos->byTypeBB[0] ^= sq_bb(from) ^ sq_bb(to);
        pst_delta += sign * get_pst_value(pc, to);
        mg_delta += sign * get_pst_mg(pc, to);
        eg_delta += sign * get_pst_eg(pc, to);
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
        mg_delta -= cap_sign * get_pst_mg(cap_pawn, cap_sq);
        eg_delta -= cap_sign * get_pst_eg(cap_pawn, cap_sq);
        
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
        mg_delta += sign * get_pst_mg(pc, to);
        eg_delta += sign * get_pst_eg(pc, to);
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
            mg_delta -= sign * get_pst_mg(pc, to);
            mg_delta += sign * get_pst_mg(promo_pc, to);
            eg_delta -= sign * get_pst_eg(pc, to);
            eg_delta += sign * get_pst_eg(promo_pc, to);
            phase_delta += PHASE_VALUES[promo_pt];
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
    pos->psqtScore_mg += mg_delta;
    pos->psqtScore_eg += eg_delta;
    pos->gamePhase += phase_delta;
    if (pos->gamePhase > TOTAL_PHASE) pos->gamePhase = TOTAL_PHASE;
    if (pos->gamePhase < 0) pos->gamePhase = 0;
    pos->psqtScore = blend_pst(pos->psqtScore_mg, pos->psqtScore_eg, pos->gamePhase);
    
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
        pos->psqtScore_mg = undo->psqtScore_mg;
        pos->psqtScore_eg = undo->psqtScore_eg;
        pos->gamePhase = undo->gamePhase;
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
    pos->psqtScore_mg = undo->psqtScore_mg;
    pos->psqtScore_eg = undo->psqtScore_eg;
    pos->gamePhase = undo->gamePhase;
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

    if (undo_stack_ptr >= 4) {
        int plies = undo_stack[undo_stack_ptr - 1].pliesFromNull;
        if (plies >= 4) {
            int repetitions = 0;
            for (int i = 4; i <= plies; i += 2) {
                int idx = undo_stack_ptr - i;
                if (idx >= 0 && undo_stack[idx].key == pos->key) {
                    repetitions++;
                    if (repetitions >= 2) {
                        return true;
                    }
                }
            }
        }
    }
    
    return false;
}

// Game result codes:
// 0 = game continues
// 1 = Black wins (White checkmated)
// 2 = White wins (Black checkmated)
// 3 = Draw by stalemate
// 4 = Draw by insufficient material
// 5 = Draw by 50-move rule
// 6 = Draw by threefold repetition
int game_result_with_legal_count(Position* pos, int legal_count, UndoInfo* undo_stack, int undo_stack_ptr,
                                  int enable_50_move_rule, int enable_threefold_repetition) {
    if (legal_count == 0) {
        if (is_check(pos, pos->sideToMove)) {
            return pos->sideToMove == CHESS_WHITE ? 1 : 2;
        } else {
            return 3;  // Stalemate
        }
    }

    if (is_insufficient_material(pos)) {
        return 4;
    }

    if (enable_50_move_rule && pos->rule50 >= 100) {
        return 5;
    }

    if (enable_threefold_repetition && undo_stack_ptr >= 4) {
        uint8_t plies = undo_stack[undo_stack_ptr - 1].pliesFromNull;

        if (plies >= 4) {
            int repetitions = 0;
            for (int i = 4; i <= plies; i += 2) {
                int idx = undo_stack_ptr - i;
                if (idx >= 0 && undo_stack[idx].key == pos->key) {
                    repetitions++;
                    if (repetitions >= 2) {
                        return 6;
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
static inline uint16_t make_rel_token(int king_sq, int piece_type, int piece_sq) {
    return (uint16_t)(
        king_sq * 12 * 64 +
        piece_type * 64 +
        piece_sq
    );
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
        memset(player_obs, 0, OBS_SIZE);
        uint8_t* board_planes = player_obs + O_BOARD;
        
        /*uint16_t* token_count_ptr = (uint16_t*)(player_obs + O_TOKEN_COUNT);
        uint16_t* token_buffer = (uint16_t*)(player_obs + O_TOKEN_DATA);
        int token_count = 0;
        ChessColor us = (ChessColor)player;  // 0=White, 1=Black
        ChessColor them = (ChessColor)!us;

        Bitboard occupied = pos->byTypeBB[0];
        int flip = player * 56;
        Square white_king_sq =
            lsb(pieces_cp(pos, CHESS_WHITE, KING));

        Square black_king_sq =
            lsb(pieces_cp(pos, CHESS_BLACK, KING));

        int wking = white_king_sq ^ flip;
        int bking = black_king_sq ^ flip;
        for (int color = CHESS_WHITE; color <= CHESS_BLACK; color++) {
            for (int pt = PAWN; pt <= KING; pt++) {
                Bitboard bb = pieces_cp(pos, color, pt);
                int piece_type = (color == CHESS_WHITE)
                    ? (pt - 1)
                    : (6 + (pt - 1));
                while (bb) {
                    Square sq = pop_lsb(&bb);
                    int psq = sq ^ flip;
                    if (token_count < MAX_TOKENS - 1) {
                        token_buffer[token_count++] =
                            make_rel_token(wking, piece_type, psq);
                        token_buffer[token_count++] =
                            make_rel_token(bking, piece_type, psq);
                    }
                }
            }
        }

        *token_count_ptr = (uint16_t)token_count;
        */
        ChessColor us = (ChessColor)player;  // 0=White, 1=Black
        ChessColor them = (ChessColor)!us;

        int flip = player * 56;


        // our pieces
        for (int pt = PAWN; pt <= KING; pt++) {
            Bitboard bb = pieces_cp(pos, player, pt);
            int plane = pt - 1;  // 0-5
            while (bb) {
                Square sq = pop_lsb(&bb);
                board_planes[plane * 64 + (sq ^ flip)] = 1;
            }
        }
        
        // Their pieces (planes 6-11)
        for (int pt = PAWN; pt <= KING; pt++) {
            Bitboard bb = pieces_cp(pos, them, pt);
            int plane = 6 + (pt - 1);  // 6-11
            while (bb) {
                Square sq = pop_lsb(&bb);
                board_planes[plane * 64 + (sq ^ flip)] = 1;
            }
        }
        /*
        // capture planes
        uint8_t* our_captures_plane = player_obs + O_BOARD + 12*64;  // Example offset, adjust to your plane order
        uint8_t* opp_threats_plane = player_obs + O_BOARD + 13*64;

        ChessColor side_to_move = pos->sideToMove;
        memset(our_captures_plane, 0, 64);
        if (side_to_move == us && env->legal_moves.count > 0) {
            for (int i = 0; i < env->legal_moves.count; i++) {
                Move m = env->legal_moves.moves[i].move;
                Piece captured = piece_on(&env->pos, to_sq(m));
                if (captured != NO_PIECE || type_of_m(m) == ENPASSANT) {  // True capture or EP
                    Square to = to_sq(m);
                    int view_to = (player == 1) ? (to ^ 56) : to;
                    our_captures_plane[view_to] = 1;  // Or 255 for full
                }
            }
        }

        memset(opp_threats_plane, 0, 64);
        Bitboard our_pieces_bb = pieces_c(&env->pos, us);
        Bitboard occupied_us = pieces(&env->pos);

        Bitboard opp_attacks = 0;

        // Pawns
        Bitboard opp_pawns = pieces_cp(&env->pos, them, PAWN);
        opp_attacks |= all_pawn_attacks(opp_pawns, them); 

        // Knights
        Bitboard opp_knights = pieces_cp(&env->pos, them, KNIGHT);
        while (opp_knights) {
            Square s = pop_lsb(&opp_knights);
            opp_attacks |= knight_attacks_bb(s);
        }

        // King
        Square opp_king = lsb(pieces_cp(&env->pos, them, KING));
        if (opp_king != SQ_NONE) opp_attacks |= king_attacks_bb(opp_king);

        Bitboard opp_bishops = pieces_cp(&env->pos, them, BISHOP);
        while (opp_bishops) {
            Square s = pop_lsb(&opp_bishops);
            opp_attacks |= bishop_attacks_bb(s, occupied_us);
        }
        Bitboard opp_rooks = pieces_cp(&env->pos, them, ROOK);
        while (opp_rooks) {
            Square s = pop_lsb(&opp_rooks);
            opp_attacks |= rook_attacks_bb(s, occupied_us);
        }
        Bitboard opp_queens = pieces_cp(&env->pos, them, QUEEN);
        while (opp_queens) {
            Square s = pop_lsb(&opp_queens);
            opp_attacks |= queen_attacks_bb(s, occupied_us);
        }

        // Now intersect with our pieces
        Bitboard threatened = opp_attacks & our_pieces_bb;
        while (threatened) {
            Square sq = pop_lsb(&threatened);
            int view_sq = (player == 1) ? (sq ^ 56) : sq;
            opp_threats_plane[view_sq] = 1;  // Or 255
        }
        */
        
        ChessColor side_to_move = pos->sideToMove;
        
        uint8_t* side_onehot = player_obs + O_SIDE;
        side_onehot[(pos->sideToMove == us) ? 0 : 1] = 1;
        
        uint8_t* castle_onehot = player_obs + O_CASTLE;
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
        if (pos->epSquare < 64) {
                int ep_sq = (player == 1) ? (pos->epSquare ^ 56) : pos->epSquare;
                ep_onehot[ep_sq] = 1;
        } else {
            ep_onehot[64] = 1;
        }
    
        uint8_t* valid_pieces = player_obs + O_VALID_PIECES;
        uint8_t* valid_dests = player_obs + O_VALID_DESTS;
        
        int player_idx = (int)us;
        
        if (side_to_move == us) {
            if (env->pick_phase[player_idx] == 0) {
                if (env->legal_moves.count > 0) {
                    for (int i = 0; i < env->legal_moves.count; i++) {
                        Square from = from_sq(env->legal_moves.moves[i].move);
                        int view_from = (player == 1) ? (from ^ 56) : from;
                        valid_pieces[view_from] = 1;
                    }
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
        player_obs[O_PASS_VALID] = (side_to_move != us) ? 255 : 0;
        
        uint8_t* phase_onehot = player_obs + O_PICK_PHASE;
        phase_onehot[env->pick_phase[player_idx]] = 1;
        
        uint8_t* selected_piece_plane = player_obs + O_SELECTED_PIECE;
        if (env->pick_phase[player_idx] == 1 && env->selected_square[player_idx] != SQ_NONE) {
            int view_selected = (player == 1) ? (env->selected_square[player_idx] ^ 56) : env->selected_square[player_idx];
            selected_piece_plane[view_selected] = 1;
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
        
        player_obs[O_SELF_CHECK] = is_check(pos, us) ? 255 : 0;
        player_obs[O_OPP_CHECK] = is_check(pos, them) ? 255 : 0;
        
        player_obs[O_RULE50] = (uint8_t)((pos->rule50 * 255) / 100);
        
        uint8_t rep_val = 255;
        if (env->undo_stack_ptr >= 4) {
            uint8_t plies = env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull;
            if (plies >= 4) {
                int repetitions = 0;
                for (int i = 4; i <= plies; i += 2) {
                    int idx = env->undo_stack_ptr - i;
                    if (idx >= 0 && env->undo_stack[idx].key == pos->key) {
                        repetitions++;
                    }
                }
                if (repetitions >= 2) {
                    rep_val = 0;
                } else if (repetitions == 1) {
                    rep_val = 128;
                }
            }
        }
        player_obs[O_REPETITION] = rep_val;
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
    env->episode_reward = 0.0f;
    env->pgn_move_count = 0;
    env->show_game_end_popup = 0;
    env->opp_in_check = 0;
    
    env->pick_phase[0] = 0;
    env->pick_phase[1] = 0;
    env->selected_square[0] = SQ_NONE;
    env->selected_square[1] = SQ_NONE;
    env->valid_destinations[0].count = 0;
    env->valid_destinations[1].count = 0;
    
    memset(env->white_captured, 0, sizeof(env->white_captured));
    memset(env->black_captured, 0, sizeof(env->black_captured));
    
    if (env->human_play) {
        env->human_color = -1;
    } else {
        env->learner_color = 1 - env->learner_color;
    }

    env->tutor_target = 0;
    env->syzygy_wdl_prev = -99;
    env->tutor_phase = 0;

    if (env->fen_curriculum != NULL && env->num_fens > 0) {
        float randvalue = (float)rand() / (float)(RAND_MAX);
        if(env->fen_curric_pct >= randvalue){
            // Pick which curriculum: DeepMind or original
            float dm_roll = (float)rand() / (float)(RAND_MAX);
            if (env->fen_curriculum_dm != NULL && env->num_fens_dm > 0
                && dm_roll < env->deepmind_fen_pct) {
                int idx = rand() % env->num_fens_dm;
                pos_set(&env->pos, env->fen_curriculum_dm[idx]);

                // Load tutor target if available
                if (env->tutor_moves_dm != NULL && env->tutor_moves_dm[idx] != 0) {
                    uint16_t packed = env->tutor_moves_dm[idx];
                    // Force learner to play whichever side the FEN says moves next
                    env->learner_color = (int)env->pos.sideToMove;

                    uint16_t from_abs = packed & 0x3F;
                    uint16_t to_abs = (packed >> 6) & 0x3F;
                    uint16_t promo = (packed >> 12) & 0xF;

                    // Convert absolute squares to learner perspective
                    if (env->learner_color == CHESS_BLACK) {
                        from_abs = from_abs ^ 56;
                        to_abs = to_abs ^ 56;
                    }
                    env->tutor_target = from_abs | (to_abs << 6) | (promo << 12);
                }
            } else {
                int idx = rand() % env->num_fens;
                pos_set(&env->pos, env->fen_curriculum[idx]);
            }
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
    
    generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
    env->legal_moves_side = env->pos.sideToMove;
    env->legal_moves_key = env->pos.key;

    // Validate tutor target against legal moves
    if (env->tutor_target != 0) {
        uint16_t t_from = env->tutor_target & 0x3F;
        uint16_t t_to = (env->tutor_target >> 6) & 0x3F;
        // Convert back from learner perspective to absolute squares
        uint16_t abs_from = (env->learner_color == CHESS_BLACK) ? (t_from ^ 56) : t_from;
        uint16_t abs_to = (env->learner_color == CHESS_BLACK) ? (t_to ^ 56) : t_to;
        int found = 0;
        for (int i = 0; i < env->legal_moves.count; i++) {
            Move m = env->legal_moves.moves[i].move;
            if (from_sq(m) == abs_from && to_sq(m) == abs_to) {
                found = 1;
                break;
            }
        }
        if (!found) {
            env->tutor_target = 0;  // Invalid target, clear it
        }
    }

    populate_observations(env);

}

bool process_player_action(Chess* env, int action, ChessColor player) {
    if (env->pos.sideToMove != player) {
        return false;
    }
    
    if (action < 0) action = 0;
    if (action >= 96) action = 95;
    
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
                // Tutor: reward for matching expert's source square
                if (player == env->learner_color && env->tutor_target != 0 && env->tutor_phase == 0) {
                    uint16_t tutor_from = env->tutor_target & 0x3F;
                    if ((uint16_t)action == tutor_from) {
                        env->rewards[0] += env->reward_tutor_piece;
                        env->log.tutor_piece_match += 1.0f;
                    }
                    env->tutor_phase = 1;
                    env->log.tutor_total += 1.0f;
                }
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
        // Tutor: failed move attempt ends tutor phase
        if (player == env->learner_color && env->tutor_target != 0 && env->tutor_phase == 1) {
            env->tutor_phase = 2;
        }
        env->pick_phase[pidx] = 0;
        env->selected_square[pidx] = SQ_NONE;
        env->valid_destinations[pidx].count = 0;
        return false;
    }

    if (player == env->learner_color) env->rewards[0] += env->reward_valid_move;

    // Tutor: reward for matching expert's destination
    if (player == env->learner_color && env->tutor_target != 0 && env->tutor_phase == 1) {
        uint16_t tutor_to = (env->tutor_target >> 6) & 0x3F;
        uint16_t tutor_promo = (env->tutor_target >> 12) & 0xF;
        int match = 0;
        if (is_promotion_selection) {
            // For promotions, compare destination file and promo type
            int promo_idx = action - 64;
            int promo_file = promo_idx % 8;
            int promo_row = promo_idx / 8;
            int desired_promo = QUEEN - promo_row;
            // tutor_to is in learner perspective; get file
            int tutor_file = tutor_to & 7;
            if (promo_file == tutor_file && tutor_promo != 0 && desired_promo == (int)tutor_promo) {
                match = 1;
            }
        } else {
            if ((uint16_t)action == tutor_to) {
                match = 1;
            }
        }
        if (match) {
            env->rewards[0] += env->reward_tutor_move;
            env->log.tutor_move_match += 1.0f;
        } else if (env->reward_tutor_wrong != 0.0f) {
            env->rewards[0] += env->reward_tutor_wrong;
        }
        env->tutor_phase = 2;
    }

    env->chess_moves++;
    env->pick_phase[pidx] = 0;
    env->selected_square[pidx] = SQ_NONE;
    env->valid_destinations[pidx].count = 0;

    
    if (env->reward_castling != 0.0f && player == env->learner_color && (int)type_of_m(chosen_move) == CASTLING) {
        env->rewards[0] += env->reward_castling;
    }
    
    if ((env->human_play || env->log_pgn || env->stockfish_bot) && env->pgn_move_count < MAX_GAME_PLIES) {
        env->pgn_moves[env->pgn_move_count++] = chosen_move;
    }
    
    env->last_move = chosen_move;
    Piece captured = piece_on(&env->pos, to_sq(chosen_move));
    if (env->reward_material != 0.0f) {
        if ((int)type_of_m(chosen_move) == PROMOTION) {
            Square to = to_sq(chosen_move);
            ChessColor them = !env->pos.sideToMove;
            if (captured != NO_PIECE) {
                env->last_see_value = see_capture(&env->pos, chosen_move);
            } else if (is_square_attacked(&env->pos, to, them)) {
                env->last_see_value = -1;
            } else {
                env->last_see_value = 0;
            }
        } else if (captured != NO_PIECE) {
            env->last_see_value = see_capture(&env->pos, chosen_move);
        } else {
            Square to = to_sq(chosen_move);
            Square from = from_sq(chosen_move);
            ChessColor them = !env->pos.sideToMove;
            if (is_square_attacked(&env->pos, to, them)) {
                Piece moving_piece = piece_on(&env->pos, from);
                Bitboard occupied = (pieces(&env->pos) ^ sq_bb(from)) | sq_bb(to);
                env->last_see_value = see_square(&env->pos, to, moving_piece, occupied);
            } else {
                env->last_see_value = 0;
            }
        }
    } else {
        env->last_see_value = 0;
    }
    
    ChessColor side_before_move = env->pos.sideToMove;
    do_move(&env->pos, chosen_move, env->undo_stack, &env->undo_stack_ptr);
    
    if (env->undo_stack_ptr > 0) {
        Piece cap = env->undo_stack[env->undo_stack_ptr - 1].captured;
        if (cap != NO_PIECE) {
            int pt = type_of_p(cap) - 1;
            if (pt >= 0 && pt < 6) {
                if (color_of(cap) == CHESS_WHITE) {
                    env->white_captured[pt]++;
                } else {
                    env->black_captured[pt]++;
                }
            }
        } else if ((int)type_of_m(chosen_move) == ENPASSANT) {
            Piece cap_pawn = (side_before_move == CHESS_WHITE) ? B_PAWN : W_PAWN;
            int pt = type_of_p(cap_pawn) - 1;
            if (pt >= 0 && pt < 6) {
                if (color_of(cap_pawn) == CHESS_WHITE) {
                    env->white_captured[pt]++;
                } else {
                    env->black_captured[pt]++;
                }
            }
        }
    }
    
    if (env->undo_stack_ptr > 0 && env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull > 99) {
        env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull = 99;
    }
    
    if (env->reward_repetition != 0.0f && player == env->learner_color && env->undo_stack_ptr >= 4) {
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
    
    return true;
}

void clip_rewards(Chess* env) {
    if (env->rewards[0] > 0.9f) {
        env->rewards[0] = 0.9f;
    }
    if (env->rewards[0] < -0.9f) {
        env->rewards[0] = -0.9f;
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
        env->actions[0] = -1.0;
        return;
    }

    ChessColor current_side = env->pos.sideToMove;
    if (current_side == env->human_color) {
        env->actions[0] = -1.0;
    }
}

static inline char piece_to_fen_char(Piece p) {
    switch (p) {
        case W_PAWN: return 'P';
        case W_KNIGHT: return 'N';
        case W_BISHOP: return 'B';
        case W_ROOK: return 'R';
        case W_QUEEN: return 'Q';
        case W_KING: return 'K';
        case B_PAWN: return 'p';
        case B_KNIGHT: return 'n';
        case B_BISHOP: return 'b';
        case B_ROOK: return 'r';
        case B_QUEEN: return 'q';
        case B_KING: return 'k';
        default: return 0;
    }
}

static int position_to_fen(const Position* pos, char* out, size_t out_size) {
    if (!pos || !out || out_size == 0) {
        return 0;
    }

    size_t off = 0;
#define APPEND_FMT(...) do { \
    int n = snprintf(out + off, out_size - off, __VA_ARGS__); \
    if (n < 0 || (size_t)n >= out_size - off) { \
        out[out_size - 1] = '\0'; \
        return 0; \
    } \
    off += (size_t)n; \
} while (0)

    for (int rank = 7; rank >= 0; rank--) {
        int empty = 0;
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file;
            Piece p = (Piece)pos->board[sq];
            if (p == NO_PIECE) {
                empty++;
                continue;
            }
            if (empty > 0) {
                APPEND_FMT("%d", empty);
                empty = 0;
            }
            char pc = piece_to_fen_char(p);
            if (pc == 0) {
                return 0;
            }
            APPEND_FMT("%c", pc);
        }
        if (empty > 0) {
            APPEND_FMT("%d", empty);
        }
        if (rank > 0) {
            APPEND_FMT("/");
        }
    }

    APPEND_FMT(" %c ", pos->sideToMove == CHESS_WHITE ? 'w' : 'b');

    if (pos->castlingRights == NO_CASTLING) {
        APPEND_FMT("- ");
    } else {
        if (pos->castlingRights & WHITE_OO) APPEND_FMT("K");
        if (pos->castlingRights & WHITE_OOO) APPEND_FMT("Q");
        if (pos->castlingRights & BLACK_OO) APPEND_FMT("k");
        if (pos->castlingRights & BLACK_OOO) APPEND_FMT("q");
        APPEND_FMT(" ");
    }

    if (pos->epSquare == SQ_NONE) {
        APPEND_FMT("- ");
    } else {
        int file = file_of(pos->epSquare);
        int rank = rank_of(pos->epSquare);
        APPEND_FMT("%c%c ", (char)('a' + file), (char)('1' + rank));
    }

    int fullmove = (int)(pos->gamePly / 2) + 1;
    APPEND_FMT("%u %d", (unsigned)pos->rule50, fullmove);

#undef APPEND_FMT
    return 1;
}

static int parse_uci_square(const char* sq_str, Square* out_sq) {
    if (!sq_str || !out_sq) {
        return 0;
    }
    char file = sq_str[0];
    char rank = sq_str[1];
    if (file < 'a' || file > 'h' || rank < '1' || rank > '8') {
        return 0;
    }
    int f = file - 'a';
    int r = rank - '1';
    *out_sq = (Square)(r * 8 + f);
    return 1;
}

static int promotion_char_to_piece_type(char c) {
    switch (c) {
        case 'q': case 'Q': return QUEEN;
        case 'r': case 'R': return ROOK;
        case 'b': case 'B': return BISHOP;
        case 'n': case 'N': return KNIGHT;
        default: return 0;
    }
}

static int parse_bestmove_line(const char* line, Square* from, Square* to, int* promo_type) {
    if (!line || !from || !to || !promo_type) {
        return 0;
    }
    char move_str[32];
    if (sscanf(line, "bestmove %31s", move_str) != 1) {
        return 0;
    }
    if (strcmp(move_str, "(none)") == 0 || strlen(move_str) < 4) {
        return 0;
    }
    if (!parse_uci_square(&move_str[0], from) || !parse_uci_square(&move_str[2], to)) {
        return 0;
    }
    *promo_type = 0;
    if (strlen(move_str) >= 5) {
        *promo_type = promotion_char_to_piece_type(move_str[4]);
        if (*promo_type == 0) {
            return 0;
        }
    }
    return 1;
}

static void stockfish_stop(Chess* env);

static int stockfish_start(Chess* env) {
    if (!env || !env->stockfish_bot) {
        return 0;
    }
    if (env->stockfish_ready && env->stockfish_in != NULL && env->stockfish_out != NULL) {
        return 1;
    }

    const char* path = getenv("PUFFER_STOCKFISH_PATH");
    if (path == NULL || path[0] == '\0') {
        path = "/usr/games/stockfish";
        if (access(path, X_OK) != 0) {
            path = "stockfish";
        }
    }

    int to_child[2];
    int from_child[2];
    if (pipe(to_child) != 0 || pipe(from_child) != 0) {
        fprintf(stderr, "WARNING: Failed to create pipes for Stockfish (%s)\n", strerror(errno));
        env->stockfish_bot = 0;
        return 0;
    }

    // Use posix_spawn instead of fork to avoid CUDA/fork interaction crashes
    posix_spawn_file_actions_t file_actions;
    posix_spawn_file_actions_init(&file_actions);
    posix_spawn_file_actions_adddup2(&file_actions, to_child[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&file_actions, from_child[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&file_actions, from_child[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&file_actions, to_child[0]);
    posix_spawn_file_actions_addclose(&file_actions, to_child[1]);
    posix_spawn_file_actions_addclose(&file_actions, from_child[0]);
    posix_spawn_file_actions_addclose(&file_actions, from_child[1]);

    extern char **environ;
    char* const argv[] = {(char*)path, NULL};
    pid_t pid;
    int spawn_rc = posix_spawnp(&pid, path, &file_actions, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&file_actions);

    if (spawn_rc != 0) {
        fprintf(stderr, "WARNING: Failed to spawn Stockfish process (%s)\n", strerror(spawn_rc));
        close(to_child[0]); close(to_child[1]);
        close(from_child[0]); close(from_child[1]);
        env->stockfish_bot = 0;
        return 0;
    }

    close(to_child[0]);
    close(from_child[1]);

    FILE* sf_in = fdopen(to_child[1], "w");
    FILE* sf_out = fdopen(from_child[0], "r");
    if (sf_in == NULL || sf_out == NULL) {
        fprintf(stderr, "WARNING: Failed to open Stockfish streams\n");
        if (sf_in) fclose(sf_in);
        if (sf_out) fclose(sf_out);
        kill(pid, SIGTERM);
        waitpid(pid, NULL, 0);
        env->stockfish_bot = 0;
        return 0;
    }

    if (setvbuf(sf_in, NULL, _IOLBF, 0) != 0 || setvbuf(sf_out, NULL, _IOLBF, 0) != 0) {
        fprintf(stderr, "WARNING: Failed to set Stockfish stream buffering\n");
    }

    env->stockfish_in = sf_in;
    env->stockfish_out = sf_out;
    env->stockfish_pid = (int)pid;
    env->stockfish_ready = 0;

    char line[512];
    fprintf(sf_in, "uci\n");
    fflush(sf_in);

    int got_uciok = 0;
    for (int i = 0; i < 1024; i++) {
        if (!fgets(line, sizeof(line), sf_out)) {
            break;
        }
        if (strncmp(line, "uciok", 5) == 0) {
            got_uciok = 1;
            break;
        }
    }

    if (!got_uciok) {
        fprintf(stderr, "WARNING: Failed to start Stockfish at '%s'\n", path);
        stockfish_stop(env);
        env->stockfish_bot = 0;
        return 0;
    }

    fprintf(sf_in, "setoption name Threads value 1\n");
    fprintf(sf_in, "setoption name Hash value 1\n");
    if (env->stockfish_limit_strength) {
        fprintf(sf_in, "setoption name UCI_LimitStrength value true\n");
        fprintf(sf_in, "setoption name UCI_Elo value %d\n", env->stockfish_elo);
    }
    fprintf(sf_in, "isready\n");
    fflush(sf_in);

    int got_readyok = 0;
    for (int i = 0; i < 1024; i++) {
        if (!fgets(line, sizeof(line), sf_out)) {
            break;
        }
        if (strncmp(line, "readyok", 7) == 0) {
            got_readyok = 1;
            break;
        }
    }
    if (!got_readyok) {
        fprintf(stderr, "WARNING: Stockfish did not reply with readyok\n");
        stockfish_stop(env);
        env->stockfish_bot = 0;
        return 0;
    }

    env->stockfish_ready = 1;
    return 1;
}

static void stockfish_stop(Chess* env) {
    if (!env) {
        return;
    }
    if (env->stockfish_in != NULL) {
        fprintf(env->stockfish_in, "quit\n");
        fflush(env->stockfish_in);
        fclose(env->stockfish_in);
        env->stockfish_in = NULL;
    }
    if (env->stockfish_out != NULL) {
        fclose(env->stockfish_out);
        env->stockfish_out = NULL;
    }
    if (env->stockfish_pid > 0) {
        waitpid((pid_t)env->stockfish_pid, NULL, 0);
        env->stockfish_pid = -1;
    }
    env->stockfish_ready = 0;
}

// Convert a Move to UCI string (e.g. "e2e4", "e7e8q" for promotion)
static void move_to_uci(Move m, char* buf) {
    const char files[] = "abcdefgh";
    const char ranks[] = "12345678";
    Square from = from_sq(m);
    Square to = to_sq(m);
    buf[0] = files[file_of(from)];
    buf[1] = ranks[rank_of(from)];
    buf[2] = files[file_of(to)];
    buf[3] = ranks[rank_of(to)];
    buf[4] = '\0';
    if ((int)type_of_m(m) == PROMOTION) {
        const char promos[] = " pnbrqk";
        buf[4] = promos[promotion_type(m)];
        buf[5] = '\0';
    }
}

static int stockfish_select_move(Chess* env, Move* out_move) {
    if (!env || !out_move) {
        return 0;
    }
    if (!stockfish_start(env)) {
        return 0;
    }

    if (env->legal_moves_side != env->pos.sideToMove || env->legal_moves_key != env->pos.key) {
        generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
        env->legal_moves_side = env->pos.sideToMove;
        env->legal_moves_key = env->pos.key;
    }
    if (env->legal_moves.count == 0) {
        return 0;
    }

    // Send full game history so Stockfish can detect repetitions.
    // Without history, Stockfish sees each position as fresh and will
    // blindly repeat the same "best" move, letting the agent exploit
    // threefold repetition for free draws.
    if (env->pgn_move_count > 0) {
        fprintf(env->stockfish_in, "position fen %s moves", env->starting_fen);
        char uci[8];
        for (int i = 0; i < env->pgn_move_count; i++) {
            move_to_uci(env->pgn_moves[i], uci);
            fprintf(env->stockfish_in, " %s", uci);
        }
        fprintf(env->stockfish_in, "\n");
    } else {
        fprintf(env->stockfish_in, "position fen %s\n", env->starting_fen);
    }
    if (env->stockfish_depth > 0) {
        fprintf(env->stockfish_in, "go depth %d\n", env->stockfish_depth);
    } else {
        int movetime = env->stockfish_movetime_ms > 0 ? env->stockfish_movetime_ms : 30;
        fprintf(env->stockfish_in, "go movetime %d\n", movetime);
    }
    fflush(env->stockfish_in);

    char line[512];
    Square from = SQ_NONE, to = SQ_NONE;
    int promo_type = 0;
    int got_move = 0;
    for (int i = 0; i < 4096; i++) {
        if (!fgets(line, sizeof(line), env->stockfish_out)) {
            break;
        }
        if (strncmp(line, "bestmove", 8) == 0) {
            got_move = parse_bestmove_line(line, &from, &to, &promo_type);
            break;
        }
    }
    if (!got_move) {
        return 0;
    }

    for (int i = 0; i < env->legal_moves.count; i++) {
        Move m = env->legal_moves.moves[i].move;
        if (from_sq(m) != from || to_sq(m) != to) {
            continue;
        }
        if (promo_type != 0) {
            if ((int)type_of_m(m) == PROMOTION && promotion_type(m) == promo_type) {
                *out_move = m;
                return 1;
            }
            continue;
        }
        if ((int)type_of_m(m) == PROMOTION) {
            continue;
        }
        *out_move = m;
        return 1;
    }

    return 0;
}

static void execute_opponent_move(Chess* env, ChessColor opp_color, Move chosen) {
    int pidx = (int)opp_color;

    env->chess_moves++;
    env->pick_phase[pidx] = 0;
    env->selected_square[pidx] = SQ_NONE;
    env->valid_destinations[pidx].count = 0;

    if ((env->log_pgn || env->stockfish_bot) && env->pgn_move_count < MAX_GAME_PLIES) {
        env->pgn_moves[env->pgn_move_count++] = chosen;
    }

    env->last_move = chosen;

    ChessColor side_before = env->pos.sideToMove;
    do_move(&env->pos, chosen, env->undo_stack, &env->undo_stack_ptr);

    if (env->undo_stack_ptr > 0) {
        Piece cap = env->undo_stack[env->undo_stack_ptr - 1].captured;
        if (cap != NO_PIECE) {
            int pt = type_of_p(cap) - 1;
            if (pt >= 0 && pt < 6) {
                if (color_of(cap) == CHESS_WHITE) {
                    env->white_captured[pt]++;
                } else {
                    env->black_captured[pt]++;
                }
            }
        } else if ((int)type_of_m(chosen) == ENPASSANT) {
            Piece cap_pawn = (side_before == CHESS_WHITE) ? B_PAWN : W_PAWN;
            int pt = type_of_p(cap_pawn) - 1;
            if (pt >= 0 && pt < 6) {
                if (color_of(cap_pawn) == CHESS_WHITE) {
                    env->white_captured[pt]++;
                } else {
                    env->black_captured[pt]++;
                }
            }
        }

        if (env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull > 99) {
            env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull = 99;
        }
    }

    generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
    env->legal_moves_side = env->pos.sideToMove;
    env->legal_moves_key = env->pos.key;
}

void random_bot_move(Chess* env) {
    if (!env->random_bot) {
        return;
    }
    
    ChessColor opp_color = !env->learner_color;
    
    if (env->pos.sideToMove != opp_color) {
        return;
    }
    
    // Ensure legal moves are up to date
    if (env->legal_moves_side != env->pos.sideToMove || env->legal_moves_key != env->pos.key) {
        generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
        env->legal_moves_side = env->pos.sideToMove;
        env->legal_moves_key = env->pos.key;
    }
    
    if (env->legal_moves.count == 0) {
        return;
    }

    int idx = rand() % env->legal_moves.count;
    Move chosen = env->legal_moves.moves[idx].move;
    execute_opponent_move(env, opp_color, chosen);
}

// Built-in 1-ply eval using the position's incrementally-maintained
// materialScore + psqtScore.  Replaces Stockfish pipe I/O for training
// (~1000x faster per move, no process overhead).
static int builtin_select_move(Chess* env, Move* out_move) {
    if (!out_move) return 0;
    MoveList* ml = &env->legal_moves;
    if (ml->count == 0) return 0;

    ChessColor us = env->pos.sideToMove;
    int sign = (us == CHESS_WHITE) ? 1 : -1;

    UndoInfo local_undo[2];
    int local_ptr = 0;

    int best_score = -999999;
    int best_idx = 0;

    for (int i = 0; i < ml->count; i++) {
        Move m = ml->moves[i].move;
        local_ptr = 0;
        do_move(&env->pos, m, local_undo, &local_ptr);
        int score = sign * (env->pos.materialScore + env->pos.psqtScore);
        undo_move(&env->pos, m, local_undo, &local_ptr);
        // Noise band ±150 cp  ≈ ELO 1200-1400 play
        score += (rand() % 301) - 150;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    *out_move = ml->moves[best_idx].move;
    return 1;
}

void stockfish_bot_move(Chess* env) {
    if (!env->stockfish_bot) {
        return;
    }

    ChessColor opp_color = !env->learner_color;
    if (env->pos.sideToMove != opp_color) {
        return;
    }

    if (env->legal_moves_side != env->pos.sideToMove || env->legal_moves_key != env->pos.key) {
        generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
        env->legal_moves_side = env->pos.sideToMove;
        env->legal_moves_key = env->pos.key;
    }
    if (env->legal_moves.count == 0) {
        return;
    }

    Move chosen = MOVE_NONE;
    int query_pct = env->stockfish_query_pct;
    if (query_pct < 0) query_pct = 0;
    if (query_pct > 100) query_pct = 100;
    int random_pct = (_g_sf_random_pct >= 0) ? _g_sf_random_pct : env->stockfish_random_pct;
    int do_query = (query_pct >= 100) ? 1 : ((query_pct > 0) && ((rand() % 100) < query_pct));
    int use_random = !do_query || ((random_pct > 0) && ((rand() % 100) < random_pct));
    if (use_random || !builtin_select_move(env, &chosen)) {
        int idx = rand() % env->legal_moves.count;
        chosen = env->legal_moves.moves[idx].move;
    }
    execute_opponent_move(env, opp_color, chosen);
}


void end_game(Chess* env){
    env->terminals[0] = 1.0f;
    float win_value = 0.0f;
    int is_draw_result = (env->game_result >= 3);

    if (is_draw_result) {
        // All draw types (3=stalemate, 4=insufficient, 5=50-move, 6=repetition)
        env->rewards[0] = env->reward_draw;
        win_value = 0.5f;
        env->log.draw_rate += 1.0f;

        // Track specific draw type
        if (env->game_result == 3) env->log.draw_by_stalemate += 1.0f;
        else if (env->game_result == 4) env->log.draw_by_insufficient += 1.0f;
        else if (env->game_result == 5) env->log.draw_by_50move += 1.0f;
        else if (env->game_result == 6) env->log.draw_by_repetition += 1.0f;

        env->white_score += 0.5f;
        env->black_score += 0.5f;
        env->learner_draws += 1.0f;
        strcpy(env->last_result, "Draw");
    } else if (env->game_result == 1) {
        // Black wins (White checkmated)
        if (env->learner_color == CHESS_WHITE) {
            env->rewards[0] = -1.0f;
            win_value = 0.0f;
            env->log.white_lossrate += 1.0f;
            env->log.opponent_winrate += 1.0f;
        } else {
            env->rewards[0] = 1.0f + env->reward_mate;
            win_value = 1.0f;
        }
        env->black_score += 1.0f;
        if (env->learner_color == CHESS_BLACK) {
            env->learner_wins += 1.0f;
            env->log.black_winrate += 1.0f;
        } else {
            env->learner_losses += 1.0f;
        }
        strcpy(env->last_result, "Black Wins");
    } else if (env->game_result == 2) {
        // White wins (Black checkmated)
        if (env->learner_color == CHESS_WHITE) {
            env->rewards[0] = 1.0f + env->reward_mate;
            win_value = 1.0f;
            env->log.white_winrate += 1.0f;
        } else {
            env->rewards[0] = -1.0f;
            win_value = 0.0f;
            env->log.black_lossrate += 1.0f;
            env->log.opponent_winrate += 1.0f;
        }
        env->white_score += 1.0f;
        if (env->learner_color == CHESS_WHITE) {
            env->learner_wins += 1.0f;
        } else {
            env->learner_losses += 1.0f;
        }
        strcpy(env->last_result, "White Wins");
    }

    // Accumulate final reward and log episode return
    env->episode_reward += env->rewards[0];
    env->log.episode_return += env->episode_reward;

    env->log.perf += win_value;
    env->log.timeout_rate += 0.0f;
    env->log.chess_moves += env->chess_moves;
    env->log.episode_length += env->tick;
    float invalid_rate = (env->tick > 0) ? ((float)env->invalid_actions_this_episode / (float)env->tick) : 0.0f;
    env->log.invalid_action_rate += invalid_rate;

    float length_score = fminf(1.0f, (float)env->chess_moves / 40.0f);
    env->log.game_length_score += length_score;

    float is_draw = is_draw_result ? 1.0f : 0.0f;
    env->log.score += win_value + 0.2f * length_score - 0.1f * is_draw;
    
    float mat = (float)env->pos.materialScore / 100.0f;
    float pst = (float)env->pos.psqtScore / 100.0f;
    if (env->learner_color == CHESS_BLACK) { mat = -mat; pst = -pst; }
    env->log.material_score += mat;
    env->log.positional_score += pst;
    
    env->log.n += 1.0f;

    // Log current random_pct and query_pct
    env->log.stockfish_random_pct += _g_sf_random_pct_f;
    env->log.stockfish_query_pct += (float)env->stockfish_query_pct;

    // Smooth EMA curriculum annealing
    if (env->stockfish_bot && _g_sf_random_pct_f > 0) {
        float learner_won = 0.0f;
        if (env->game_result == 2 && env->learner_color == CHESS_WHITE) learner_won = 1.0f;
        if (env->game_result == 1 && env->learner_color == CHESS_BLACK) learner_won = 1.0f;

        // Update EMA (warmup: simple average for first 1000 games, then EMA)
        _g_annealing_games++;
        if (_g_annealing_games <= 1000) {
            _g_ema_wr += (learner_won - _g_ema_wr) / (float)_g_annealing_games;
        } else {
            _g_ema_wr = (1.0f - EMA_ALPHA) * _g_ema_wr + EMA_ALPHA * learner_won;
        }

        // Smooth annealing: decrease proportional to excess WR above threshold
        if (_g_ema_wr > ANNEAL_WR_THRESHOLD) {
            float excess = _g_ema_wr - ANNEAL_WR_THRESHOLD;
            float old_pct = _g_sf_random_pct_f;
            _g_sf_random_pct_f -= ANNEAL_RATE * excess;
            if (_g_sf_random_pct_f < 0.0f) _g_sf_random_pct_f = 0.0f;
            _g_sf_random_pct = (int)roundf(_g_sf_random_pct_f);

            // Log at whole-number boundaries
            if ((int)roundf(old_pct) != _g_sf_random_pct) {
                printf("ANNEAL: ema_wr=%.4f random_pct=%.1f->%d%% (games=%d)\n",
                       _g_ema_wr, old_pct, _g_sf_random_pct, _g_annealing_games);
            }
        }
    }

    // Log EMA for wandb tracking
    env->log.ema_winrate += _g_ema_wr;

    if (env->human_play) {
        env->show_game_end_popup = 1;
    } else {
        if (env->log_pgn && env->pgn_filename[0] != '\0') {
            env->pgn_game_number++;
            export_pgn_append(env, env->pgn_filename, 1);
        }
        c_reset(env);
    }
    return;
 
}
void c_step(Chess* env) {
    if (!env->selfplay && !env->human_play && !env->random_bot && !env->stockfish_bot) {
        fprintf(stderr, "FATAL: selfplay=0 AND human_play=0 and random_bot=0 and stockfish_bot=0 is invalid configuration\n");
        exit(1);
    }
    
    if (env->human_play && env->human_color == -1) {
        return;
    }
    
    if (!env->human_play && env->selfplay && !env->log_pgn_choice_made) {
        if (env->debug_mode) {
            env->log_pgn = 0;
            env->log_pgn_choice_made = 1;
        } else {
            return;
        }
    }
    
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    env->tick++;

    if ((env->random_bot || env->stockfish_bot) && env->pos.sideToMove != env->learner_color) {
        if (env->stockfish_bot) {
            stockfish_bot_move(env);
        } else {
            random_bot_move(env);
        }
        
        env->game_result = game_result_with_legal_count(&env->pos, env->legal_moves.count, 
            env->undo_stack, env->undo_stack_ptr,
            env->enable_50_move_rule, env->enable_threefold_repetition);
        
        if (env->game_result != 0) {
            end_game(env);
        }
        
        populate_observations(env);
        if (env->pos.sideToMove != env->learner_color) {
            return;
        }
    }
    
    int action;
    if (env->human_play) {
        human_play(env);
    }
    if (env->selfplay) {
        ChessColor current_side = env->pos.sideToMove;
        action = (current_side == env->learner_color) ? (int)env->actions[0] : (int)env->actions[1];
    } else {
        action = (int)env->actions[0];
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
    
    if (action == PASS_ACTION) {
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
        
        float mat_reward = 0.0f;
        float pos_reward = 0.0f;

        float raw_mat = (float)mat_delta / 900.0f * env->reward_material;
        float raw_pos = (float)pst_delta / 1000.0f * env->reward_position;

        if (env->learner_color == CHESS_BLACK) {
            raw_mat = -raw_mat;
            raw_pos = -raw_pos;
        }

        if (mat_delta != 0) {
            raw_pos = 0.0f;
        }

        if (mover == env->learner_color) {
            if (env->reward_material != 0.0f) {
                if (raw_mat > 0) {
                    if (env->last_see_value >= 0) {
                        mat_reward = raw_mat;
                    } else {
                        mat_reward = 0.0f;
                    }
                } else {
                    mat_reward = raw_mat;
                }
            }
            pos_reward = raw_pos;
        } else {
            mat_reward = raw_mat;
            pos_reward = raw_pos;
        }
        
        env->rewards[0] += mat_reward + pos_reward;
    }
    
    if (move_completed && mover == env->learner_color && env->reward_material != 0.0f) {
        if (env->last_see_value < 0) {
            float hanging_penalty = (float)env->last_see_value / 900.0f * env->reward_material;
            env->rewards[0] += hanging_penalty;
        }
    }
    if (move_completed && mover == env->learner_color && env->opp_in_check == 0 && is_check(&env->pos, !env->learner_color)){
        env->opp_in_check = 1;
        env->rewards[0] += env->reward_check;
    }
    if (move_completed && mover == !env->learner_color){
        env->opp_in_check = 0;
    }
    
    // Syzygy tablebase reward: reward for improving WDL status
    if (move_completed && mover == env->learner_color && env->reward_syzygy != 0.0f) {
        int wdl = probe_syzygy_wdl(&env->pos);
        if (wdl >= 0) {
            env->log.syzygy_probes += 1.0f;
            // Convert WDL to score from learner perspective
            // TB result is from side-to-move perspective, but we just moved,
            // so side-to-move is now the opponent. Flip the result.
            int learner_wdl;
            if (wdl == TB_WIN) learner_wdl = -2;       // opponent wins = we lose
            else if (wdl == TB_CURSED_WIN) learner_wdl = -1;
            else if (wdl == TB_DRAW) learner_wdl = 0;
            else if (wdl == TB_BLESSED_LOSS) learner_wdl = 1;
            else if (wdl == TB_LOSS) learner_wdl = 2;  // opponent loses = we win
            else learner_wdl = 0;

            if (learner_wdl > 0) env->log.syzygy_wins += 1.0f;
            else if (learner_wdl == 0) env->log.syzygy_draws += 1.0f;

            // Delta reward: improvement from previous probe
            if (env->syzygy_wdl_prev != -99) {
                int delta = learner_wdl - env->syzygy_wdl_prev;
                float syzygy_reward = (float)delta * env->reward_syzygy;
                env->rewards[0] += syzygy_reward;
                env->log.syzygy_reward_total += syzygy_reward;
            }
            env->syzygy_wdl_prev = learner_wdl;
        }
    }
    clip_rewards(env);
    if (move_completed) {
        if (env->legal_moves_side != env->pos.sideToMove || env->legal_moves_key != env->pos.key) {
            generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
            env->legal_moves_side = env->pos.sideToMove;
            env->legal_moves_key = env->pos.key;
        }
    }

    // Tutor-only mode: end episode after first move attempt
    if (move_completed && env->tutor_only_mode && env->tutor_phase == 2) {
        env->terminals[0] = 1.0f;
        env->episode_reward += env->rewards[0];
        env->log.episode_return += env->episode_reward;
        env->log.perf += 0.5f;
        env->log.chess_moves += env->chess_moves;
        env->log.episode_length += env->tick;
        float invalid_rate = (env->tick > 0) ? ((float)env->invalid_actions_this_episode / (float)env->tick) : 0.0f;
        env->log.invalid_action_rate += invalid_rate;
        env->log.n += 1.0f;
        env->log.stockfish_random_pct += _g_sf_random_pct_f;
        env->log.stockfish_query_pct += (float)env->stockfish_query_pct;
        env->log.ema_winrate += _g_ema_wr;
        c_reset(env);
        return;
    }

    if (env->chess_moves >= env->max_moves || env->undo_stack_ptr >= MAX_GAME_PLIES - 2) {
        env->terminals[0] = 1.0f;
        env->rewards[0] = env->reward_draw;
        // Accumulate final reward and log episode return
        env->episode_reward += env->rewards[0];
        env->log.episode_return += env->episode_reward;
        
        env->log.perf += 0.5f;
        env->log.draw_rate += 1.0f;
        env->log.timeout_rate += 1.0f;
        env->log.chess_moves += env->chess_moves;
        env->log.episode_length += env->tick;
        float invalid_rate = (env->tick > 0) ? ((float)env->invalid_actions_this_episode / (float)env->tick) : 0.0f;
        env->log.invalid_action_rate += invalid_rate;
        float length_score = fminf(1.0f, (float)env->chess_moves / 40.0f);
        env->log.game_length_score += length_score;
        env->log.score += 0.5f + 0.2f * length_score - 0.1f;
        float mat = (float)env->pos.materialScore / 100.0f;
        float pst = (float)env->pos.psqtScore / 100.0f;
        if (env->learner_color == CHESS_BLACK) { mat = -mat; pst = -pst; }
        env->log.material_score += mat;
        env->log.positional_score += pst;
        
        env->log.n += 1.0f;
        env->log.stockfish_random_pct += (float)_g_sf_random_pct;
        env->log.stockfish_query_pct += (float)env->stockfish_query_pct;
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
        end_game(env);
    } else {
        // Accumulate intermediate rewards (end_game handles its own accumulation)
        env->episode_reward += env->rewards[0];
    }
    
    populate_observations(env);
}

int move_to_san(Position* pos, Move m, char* buf) {
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
    
    // Castling
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
    
    UndoInfo undo[1];
    int undo_ptr = 0;
    do_move(pos, m, undo, &undo_ptr);
    
    ChessColor them = pos->sideToMove;
    if (is_check(pos, them)) {
        MoveList ml;
        generate_legal(pos, &ml, undo, &undo_ptr);
        if (ml.count == 0) {
            *ptr++ = '#';
        } else {
            *ptr++ = '+';
        }
    }
    
    undo_move(pos, m, undo, &undo_ptr);
    
    *ptr = '\0';
    return ptr - buf;
}

void export_pgn_append(Chess* env, const char* filename, int append) {
    FILE* f = fopen(filename, append ? "a" : "w");
    if (!f) return;
    
    if (env->human_play) {
        fprintf(f, "[Event \"Human vs AI\"]\n");
        fprintf(f, "[White \"%s\"]\n", env->human_color == CHESS_WHITE ? "Human" : "AI");
        fprintf(f, "[Black \"%s\"]\n", env->human_color == CHESS_BLACK ? "Human" : "AI");
    } else {
        fprintf(f, "[Event \"Selfplay Eval Game %d\"]\n", env->pgn_game_number);
        fprintf(f, "[White \"%s\"]\n", env->learner_color == CHESS_WHITE ? "Learner" : "Opponent");
        fprintf(f, "[Black \"%s\"]\n", env->learner_color == CHESS_BLACK ? "Learner" : "Opponent");
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
        move_to_san(&replay_pos, m, san_buf);
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

void export_pgn(Chess* env, const char* filename) {
    export_pgn_append(env, filename, 0);
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
        
        if (!env->human_play && env->selfplay) {
            env->log_pgn_choice_made = 0;
        }
    }
    
    if (IsKeyDown(KEY_ESCAPE) || WindowShouldClose()) {
        CloseWindow();
        exit(0);
    }
    
    if (env->human_play && env->show_game_end_popup) {
        BeginDrawing();
        ClearBackground((Color){40, 40, 40, 255});
        
        int popup_width = 300;
        int popup_height = 200;
        int popup_x = (board_size - popup_width) / 2;
        int popup_y = (board_size - popup_height) / 2;
        
        DrawRectangle(popup_x, popup_y, popup_width, popup_height, (Color){60, 60, 60, 255});
        DrawRectangleLines(popup_x, popup_y, popup_width, popup_height, WHITE);
        
        DrawText("Game Over!", popup_x + 70, popup_y + 20, 24, WHITE);
        DrawText(env->last_result, popup_x + 80, popup_y + 55, 18, YELLOW);
        
        int btn_width = 120;
        int btn_height = 35;
        int btn_y = popup_y + 110;
        
        Rectangle save_btn = {popup_x + 20, btn_y, btn_width, btn_height};
        Rectangle new_game_btn = {popup_x + 160, btn_y, btn_width, btn_height};
        
        DrawRectangleRec(save_btn, DARKGREEN);
        DrawRectangleLinesEx(save_btn, 2, WHITE);
        DrawText("Save PGN", popup_x + 35, btn_y + 10, 16, WHITE);
        
        DrawRectangleRec(new_game_btn, DARKBLUE);
        DrawRectangleLinesEx(new_game_btn, 2, WHITE);
        DrawText("New Game", popup_x + 175, btn_y + 10, 16, WHITE);
        
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            Vector2 mouse = GetMousePosition();
            if (CheckCollisionPointRec(mouse, save_btn)) {
                char filename[64];
                snprintf(filename, sizeof(filename), "game_%d.pgn", (int)time(NULL));
                export_pgn(env, filename);
                printf("Saved PGN to %s\n", filename);
            } else if (CheckCollisionPointRec(mouse, new_game_btn)) {
                c_reset(env);
            }
        }
        
        EndDrawing();
        return;
    }

    if (!env->human_play && env->selfplay && !env->log_pgn_choice_made) {
        BeginDrawing();
        ClearBackground((Color){40, 40, 40, 255});
        
        DrawText("Log PGN Files?", 256 - 80, 180, 24, WHITE);
        DrawText("Games will be appended to a timestamped file", 256 - 160, 220, 14, LIGHTGRAY);
        
        int button_width = 140;
        int button_height = 40;
        int center_x = 256;
        int yes_y = 270;
        int no_y = 330;
        
        Rectangle yes_button = {center_x - button_width/2, yes_y, button_width, button_height};
        Rectangle no_button = {center_x - button_width/2, no_y, button_width, button_height};
        
        DrawRectangleRec(yes_button, DARKGREEN);
        DrawRectangleLinesEx(yes_button, 2, WHITE);
        DrawText("Yes, Log PGN", center_x - 55, yes_y + 12, 16, WHITE);
        
        DrawRectangleRec(no_button, MAROON);
        DrawRectangleLinesEx(no_button, 2, WHITE);
        DrawText("No Logging", center_x - 45, no_y + 12, 16, WHITE);
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            Vector2 mousePos = GetMousePosition();
            if (CheckCollisionPointRec(mousePos, yes_button)) {
                env->log_pgn = 1;
                env->log_pgn_choice_made = 1;
                env->pgn_game_number = 0;
                snprintf(env->pgn_filename, sizeof(env->pgn_filename), "run_%d_pgns.pgn", (int)time(NULL));
                printf("PGN logging enabled: %s\n", env->pgn_filename);
            } else if (CheckCollisionPointRec(mousePos, no_button)) {
                env->log_pgn = 0;
                env->log_pgn_choice_made = 1;
                printf("PGN logging disabled\n");
            }
        }
        
        EndDrawing();
        return;
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
    int flip_board = (env->human_play && env->human_color == CHESS_BLACK) ? 1 : 0;
    if (env->human_play && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        Vector2 mp = GetMousePosition();
        int file = (int)(mp.x) / cell_size;
        int rank = 7 - ((int)(mp.y) / cell_size);
        if (flip_board) {
            file = 7 - file;
            rank = 7 - rank;
        }
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
                    if (env->pgn_move_count < MAX_GAME_PLIES) {
                        env->pgn_moves[env->pgn_move_count++] = chosen;
                    }
                    do_move(&env->pos, chosen, env->undo_stack, &env->undo_stack_ptr);
                    env->tick++;
                    env->chess_moves++;
                    generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
                    env->legal_moves_side = env->pos.sideToMove;
                    env->legal_moves_key = env->pos.key;
                    env->actions[0] = -1.0;
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
            
            int display_file = flip_board ? (7 - file) : file;
            int display_rank = flip_board ? (7 - rank) : rank;
            int draw_x = display_file * cell_size;
            int draw_y = (7 - display_rank) * cell_size;
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
            int display_file = flip_board ? (7 - file) : file;
            int display_rank = flip_board ? (7 - rank) : rank;
            draw_piece(env, pc, display_file, display_rank, cell_size);
        }
    }
    
    const int scoreboard_y = board_size + 10;
    char score_text[128];
    snprintf(score_text, sizeof(score_text), "White: %.1f  Black: %.1f", 
             env->white_score, env->black_score);
    DrawText(score_text, 10, scoreboard_y, 20, WHITE);
    
    int32_t engine_eval = evaluate_sunfish(&env->pos);
    char eval_text[64];
    float eval_pawns = engine_eval / 100.0f;
    snprintf(eval_text, sizeof(eval_text), "Engine eval: %.2f", eval_pawns);
    Color eval_color = (engine_eval > 0) ? (Color){240, 217, 181, 255} : 
                      (engine_eval < 0) ? (Color){100, 100, 100, 255} : WHITE;
    DrawText(eval_text, 10, scoreboard_y + 30, 16, eval_color);
    
    char learner_text[128];
    snprintf(learner_text, sizeof(learner_text), "Learner: %.0f-%.0f-%.0f (W-L-D)", 
             env->learner_wins, env->learner_losses, env->learner_draws);
    DrawText(learner_text, 10, scoreboard_y + 55, 16, GREEN);
    
    int captured_y = scoreboard_y + 75;
    char white_captured_text[128] = "White captured: ";
    char black_captured_text[128] = "Black captured: ";
    int white_len = strlen(white_captured_text);
    int black_len = strlen(black_captured_text);
    
    const char* piece_chars = "PNBRQK";
    for (int pt = 0; pt < 6; pt++) {
        for (int i = 0; i < env->white_captured[pt]; i++) {
            white_captured_text[white_len++] = piece_chars[pt];
        }
        for (int i = 0; i < env->black_captured[pt]; i++) {
            black_captured_text[black_len++] = piece_chars[pt];
        }
    }
    white_captured_text[white_len] = '\0';
    black_captured_text[black_len] = '\0';
    
    DrawText(white_captured_text, 10, captured_y, 14, (Color){240, 217, 181, 255});
    DrawText(black_captured_text, 10, captured_y + 18, 14, (Color){100, 100, 100, 255});
    
    if (env->last_result[0] != '\0') {
        Color result_color = YELLOW;
        if (strstr(env->last_result, "White")) result_color = (Color){240, 217, 181, 255};
        else if (strstr(env->last_result, "Black")) result_color = (Color){100, 100, 100, 255};
        
        DrawText(env->last_result, 10, captured_y + 40, 18, result_color);
    }
    
    char move_text[64];
    snprintf(move_text, sizeof(move_text), "Move: %d", env->chess_moves);
    DrawText(move_text, board_size - 100, scoreboard_y, 18, LIGHTGRAY);
    
    if (!env->human_play) {
        const char* learner_str = (env->learner_color == CHESS_WHITE) ? "Learner: White" : "Learner: Black";
        DrawText(learner_str, board_size - 120, scoreboard_y + 25, 16, LIGHTGRAY);
    }
    
    const int btn_width = 36;
    const int btn_height = 24;
    const int btn_y = scoreboard_y + 45;
    const int btn_start_x = env->human_play ? board_size / 2 - 100 : board_size / 2 - 70;
    
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
    
    Rectangle restart_btn = {0, 0, 0, 0};
    if (env->human_play) {
        restart_btn = (Rectangle){board_size - 60, btn_y, 55, btn_height};
        DrawRectangleRec(restart_btn, MAROON);
        DrawRectangleLinesEx(restart_btn, 2, LIGHTGRAY);
        DrawText("Exit", board_size - 53, btn_y + 4, 16, WHITE);
    }
    
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
        if (env->human_play && CheckCollisionPointRec(mouse, restart_btn)) {
            c_reset(env);
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
    stockfish_stop(env);
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
