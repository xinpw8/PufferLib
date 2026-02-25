#ifndef TBCONFIG_H
#define TBCONFIG_H

#include <stdint.h>

/* Forward declarations from chess.h - no circular include */
typedef uint64_t Bitboard;
extern Bitboard PawnAttacks[2][64];
extern Bitboard KnightAttacks[64];
extern Bitboard KingAttacks[64];

/* These need the full chess.h inline functions, so we use Fathom's builtins instead */
/* #define TB_ROOK_ATTACKS(sq, occ)    rook_attacks_bb(sq, occ) */
/* #define TB_BISHOP_ATTACKS(sq, occ)  bishop_attacks_bb(sq, occ) */

#define TB_CUSTOM_POP_COUNT(x) __builtin_popcountll(x)
#define TB_CUSTOM_LSB(x) __builtin_ctzll(x)

/* Use chess.h's precomputed lookup tables for non-sliding pieces */
#define TB_KING_ATTACKS(sq)         KingAttacks[sq]
#define TB_KNIGHT_ATTACKS(sq)       KnightAttacks[sq]
#define TB_PAWN_ATTACKS(sq, color)  PawnAttacks[color][sq]

/* Let Fathom compute sliding attacks internally (avoids needing magic bitboard inlines) */

/* Scoring constants */
#define TB_VALUE_PAWN     100
#define TB_VALUE_MATE     32000
#define TB_VALUE_INFINITE 32767
#define TB_VALUE_DRAW     0
#define TB_MAX_MATE_PLY   255

#endif
