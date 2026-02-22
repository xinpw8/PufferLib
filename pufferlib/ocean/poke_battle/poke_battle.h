// poke_battle.h - Self-contained Gen 1 OU Pokemon Battle Simulator for PufferLib
// Implements core Gen 1 battle mechanics: type chart, damage formula, status,
// stat stages, critical hits, switching, and competitive OU team generation.

#ifndef POKE_BATTLE_H
#define POKE_BATTLE_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

// Forward declare Client for render.h
typedef struct Client Client;

// ============================================================================
// Constants
// ============================================================================

#define NUM_TYPES       15
#define NUM_SPECIES     149
#define NUM_MOVES       45
#define NUM_POKEMON     6
#define NUM_MOVE_SLOTS  4
#define MAX_STAT_STAGE  6
#define NUM_ACTIONS     10   // 0-3 = moves, 4-9 = switch to pokemon 0-5
#define OBS_SIZE        140
#define MAX_TURNS       500
#define ENDLESS_BATTLE_STALE_TURNS 32

// Reward shaping weights (per-step intermediate rewards)
// Rewards are normalized by max team HP (~2400 for 6 pokemon)
#define REWARD_DAMAGE_DEALT   0.05f   // per HP of damage dealt to opponent (normalized)
#define REWARD_DAMAGE_TAKEN  -0.05f   // per HP of damage taken (normalized)
#define REWARD_KO_OPPONENT    0.1f    // bonus for KOing an opponent pokemon
#define REWARD_KO_SELF       -0.1f    // penalty for own pokemon being KOed
#define REWARD_WIN            1.0f    // terminal win
#define REWARD_LOSS          -1.0f    // terminal loss

// Bot mode constants
#define BOT_RANDOM      0
#define BOT_HEURISTIC   1   // 1-ply minimax with evaluation
#define BOT_MCTS        2   // Monte Carlo rollouts with evaluation

// Default MCTS configuration
#define MCTS_DEFAULT_ITERATIONS  128
#define MCTS_DEFAULT_DEPTH       5

// Adaptive team-builder configuration
#define TEAM_BUILDER_DISABLED 0
#define TEAM_BUILDER_ADAPTIVE 1
#define TEAM_BUILDER_RECENT_WINDOW 64
#define TEAM_BUILDER_TARGET_WINRATE 0.99f
#define TEAM_BUILDER_EXPLORE_MIN_PCT 8
#define TEAM_BUILDER_EXPLORE_MAX_PCT 45
#define TEAM_BUILDER_UCB_C 0.35f

// ============================================================================
// Enums
// ============================================================================

typedef enum {
    TYPE_NORMAL = 0,
    TYPE_FIRE,
    TYPE_WATER,
    TYPE_ELECTRIC,
    TYPE_GRASS,
    TYPE_ICE,
    TYPE_FIGHTING,
    TYPE_POISON,
    TYPE_GROUND,
    TYPE_FLYING,
    TYPE_PSYCHIC,
    TYPE_BUG,
    TYPE_ROCK,
    TYPE_GHOST,
    TYPE_DRAGON,
    TYPE_NONE = 15
} Type;

// Physical types in Gen 1: Normal, Fighting, Poison, Ground, Flying, Bug, Rock, Ghost
// Special types in Gen 1: Fire, Water, Electric, Grass, Ice, Psychic, Dragon
static const int TYPE_IS_PHYSICAL[NUM_TYPES] = {
    1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0
};

typedef enum {
    STATUS_NONE = 0,
    STATUS_SLEEP,
    STATUS_FREEZE,
    STATUS_BURN,
    STATUS_POISON,
    STATUS_TOXIC,
    STATUS_PARALYSIS
} StatusCondition;

typedef enum {
    EFFECT_NONE = 0,
    EFFECT_PARALYZE_CHANCE,  // chance to paralyze
    EFFECT_BURN_CHANCE,      // chance to burn
    EFFECT_FREEZE_CHANCE,    // chance to freeze
    EFFECT_LOWER_SPECIAL,    // chance to lower target special 1 stage
    EFFECT_LOWER_SPEED,      // chance to lower target speed 1 stage
    EFFECT_LOWER_DEFENSE,    // chance to lower target defense 1 stage
    EFFECT_LOWER_ACCURACY,   // chance to lower target accuracy 1 stage
    EFFECT_RECOIL,           // recoil damage
    EFFECT_SELF_DESTRUCT,    // user faints, halves target defense in calc
    EFFECT_HIGH_CRIT,        // increased crit rate
    EFFECT_FIXED_DAMAGE,     // fixed damage = user's level (100)
    EFFECT_SLEEP,            // puts target to sleep
    EFFECT_RECOVER,          // heals 50% max HP
    EFFECT_REST,             // heals fully + sleep 2 turns
    EFFECT_BOOST_ATK_2,      // +2 attack stages
    EFFECT_BOOST_SPC_2,      // +2 special stages
    EFFECT_BOOST_SPE_2,      // +2 speed stages
    EFFECT_SUBSTITUTE,       // create substitute (25% HP)
    EFFECT_TOXIC,            // badly poison
    EFFECT_REFLECT,          // set reflect
    EFFECT_LIGHT_SCREEN,     // set light screen
    EFFECT_CONFUSE,          // confuse target
    EFFECT_TRAPPING,         // trap target 2-5 turns
    EFFECT_HYPER_BEAM,       // must recharge next turn if doesn't KO
    EFFECT_DRAIN,            // heal 50% of damage dealt
    EFFECT_MULTI_HIT,        // hit 2-5 times
    EFFECT_DOUBLE_HIT,       // hit exactly twice
    EFFECT_THUNDER_WAVE,     // paralyze (status move, not a chance)
    EFFECT_LEECH_SEED,       // seed target
    EFFECT_BOOST_ATK_1,      // +1 attack
    EFFECT_BOOST_DEF_1,      // +1 defense
    EFFECT_FLINCH_CHANCE,    // chance to flinch (not very relevant in Gen 1)
    EFFECT_RAISE_SPECIAL_CHANCE, // chance to raise user's special
    EFFECT_TRANSFORM,        // copy target's species/types/moves
} MoveEffect;

typedef enum {
    MOVE_NONE = 0,
    MOVE_BODY_SLAM,
    MOVE_HYPER_BEAM,
    MOVE_EARTHQUAKE,
    MOVE_BLIZZARD,
    MOVE_PSYCHIC,
    MOVE_THUNDERBOLT,
    MOVE_ICE_BEAM,
    MOVE_SURF,
    MOVE_FIRE_BLAST,
    MOVE_ROCK_SLIDE,
    MOVE_DRILL_PECK,
    MOVE_NIGHT_SHADE,
    MOVE_SEISMIC_TOSS,
    MOVE_DOUBLE_EDGE,
    MOVE_EXPLOSION,
    MOVE_SELF_DESTRUCT,
    MOVE_SLASH,
    MOVE_BUBBLE_BEAM,
    MOVE_THUNDER_WAVE,
    MOVE_SLEEP_POWDER,
    MOVE_STUN_SPORE,
    MOVE_LOVELY_KISS,
    MOVE_HYPNOSIS,
    MOVE_SING,
    MOVE_RECOVER,
    MOVE_SOFT_BOILED,
    MOVE_REST,
    MOVE_SWORDS_DANCE,
    MOVE_AMNESIA,
    MOVE_AGILITY,
    MOVE_SUBSTITUTE,
    MOVE_TOXIC,
    MOVE_REFLECT,
    MOVE_LIGHT_SCREEN,
    MOVE_CONFUSE_RAY,
    MOVE_MEGA_DRAIN,
    MOVE_THUNDERSHOCK,
    MOVE_PIN_MISSILE,
    MOVE_LEECH_SEED,
    MOVE_TACKLE,
    MOVE_STRING_SHOT,
    MOVE_HARDEN,
    MOVE_SPLASH,
    MOVE_TRANSFORM,
} MoveID;

typedef enum {
    SPECIES_NONE = 0,
    SPECIES_TAUROS,
    SPECIES_CHANSEY,
    SPECIES_SNORLAX,
    SPECIES_ALAKAZAM,
    SPECIES_EXEGGUTOR,
    SPECIES_STARMIE,
    SPECIES_GENGAR,
    SPECIES_JYNX,
    SPECIES_ZAPDOS,
    SPECIES_RHYDON,
    SPECIES_CLOYSTER,
    SPECIES_GOLEM,
    SPECIES_LAPRAS,
    SPECIES_SLOWBRO,
    SPECIES_JOLTEON,
    SPECIES_PERSIAN,
    SPECIES_HYPNO,
    SPECIES_ARTICUNO,
    SPECIES_DRAGONITE,
    SPECIES_MACHAMP,
    SPECIES_BULBASAUR,
    SPECIES_IVYSAUR,
    SPECIES_VENUSAUR,
    SPECIES_CHARMANDER,
    SPECIES_CHARMELEON,
    SPECIES_CHARIZARD,
    SPECIES_SQUIRTLE,
    SPECIES_WARTORTLE,
    SPECIES_BLASTOISE,
    SPECIES_CATERPIE,
    SPECIES_METAPOD,
    SPECIES_BUTTERFREE,
    SPECIES_WEEDLE,
    SPECIES_KAKUNA,
    SPECIES_BEEDRILL,
    SPECIES_PIDGEY,
    SPECIES_PIDGEOTTO,
    SPECIES_PIDGEOT,
    SPECIES_RATTATA,
    SPECIES_RATICATE,
    SPECIES_SPEAROW,
    SPECIES_FEAROW,
    SPECIES_EKANS,
    SPECIES_ARBOK,
    SPECIES_PIKACHU,
    SPECIES_RAICHU,
    SPECIES_SANDSHREW,
    SPECIES_SANDSLASH,
    SPECIES_NIDORAN_F,
    SPECIES_NIDORINA,
    SPECIES_NIDOQUEEN,
    SPECIES_NIDORAN_M,
    SPECIES_NIDORINO,
    SPECIES_NIDOKING,
    SPECIES_CLEFAIRY,
    SPECIES_CLEFABLE,
    SPECIES_VULPIX,
    SPECIES_NINETALES,
    SPECIES_JIGGLYPUFF,
    SPECIES_WIGGLYTUFF,
    SPECIES_ZUBAT,
    SPECIES_GOLBAT,
    SPECIES_ODDISH,
    SPECIES_GLOOM,
    SPECIES_VILEPLUME,
    SPECIES_PARAS,
    SPECIES_PARASECT,
    SPECIES_VENONAT,
    SPECIES_VENOMOTH,
    SPECIES_DIGLETT,
    SPECIES_DUGTRIO,
    SPECIES_MEOWTH,
    SPECIES_PSYDUCK,
    SPECIES_GOLDUCK,
    SPECIES_MANKEY,
    SPECIES_PRIMEAPE,
    SPECIES_GROWLITHE,
    SPECIES_ARCANINE,
    SPECIES_POLIWAG,
    SPECIES_POLIWHIRL,
    SPECIES_POLIWRATH,
    SPECIES_ABRA,
    SPECIES_KADABRA,
    SPECIES_MACHOP,
    SPECIES_MACHOKE,
    SPECIES_BELLSPROUT,
    SPECIES_WEEPINBELL,
    SPECIES_VICTREEBEL,
    SPECIES_TENTACOOL,
    SPECIES_TENTACRUEL,
    SPECIES_GEODUDE,
    SPECIES_GRAVELER,
    SPECIES_PONYTA,
    SPECIES_RAPIDASH,
    SPECIES_SLOWPOKE,
    SPECIES_MAGNEMITE,
    SPECIES_MAGNETON,
    SPECIES_FARFETCH_D,
    SPECIES_DODUO,
    SPECIES_DODRIO,
    SPECIES_SEEL,
    SPECIES_DEWGONG,
    SPECIES_GRIMER,
    SPECIES_MUK,
    SPECIES_SHELLDER,
    SPECIES_GASTLY,
    SPECIES_HAUNTER,
    SPECIES_ONIX,
    SPECIES_DROWZEE,
    SPECIES_KRABBY,
    SPECIES_KINGLER,
    SPECIES_VOLTORB,
    SPECIES_ELECTRODE,
    SPECIES_EXEGGCUTE,
    SPECIES_CUBONE,
    SPECIES_MAROWAK,
    SPECIES_HITMONLEE,
    SPECIES_HITMONCHAN,
    SPECIES_LICKITUNG,
    SPECIES_KOFFING,
    SPECIES_WEEZING,
    SPECIES_RHYHORN,
    SPECIES_TANGELA,
    SPECIES_KANGASKHAN,
    SPECIES_HORSEA,
    SPECIES_SEADRA,
    SPECIES_GOLDEEN,
    SPECIES_SEAKING,
    SPECIES_STARYU,
    SPECIES_MR_MIME,
    SPECIES_SCYTHER,
    SPECIES_ELECTABUZZ,
    SPECIES_MAGMAR,
    SPECIES_PINSIR,
    SPECIES_MAGIKARP,
    SPECIES_GYARADOS,
    SPECIES_DITTO,
    SPECIES_EEVEE,
    SPECIES_VAPOREON,
    SPECIES_FLAREON,
    SPECIES_PORYGON,
    SPECIES_OMANYTE,
    SPECIES_OMASTAR,
    SPECIES_KABUTO,
    SPECIES_KABUTOPS,
    SPECIES_AERODACTYL,
    SPECIES_MOLTRES,
    SPECIES_DRATINI,
    SPECIES_DRAGONAIR,
} SpeciesID;

// ============================================================================
// Gen 1 Type Effectiveness Chart
// Stored as multiplier * 2: 0 = immune, 1 = 0.5x, 2 = 1x, 3 = 2x
// type_chart[attacking_type][defending_type]
// ============================================================================

static const unsigned char TYPE_CHART[NUM_TYPES][NUM_TYPES] = {
    //                  NOR FIR WAT ELE GRA ICE FIG POI GRO FLY PSY BUG ROC GHO DRA
    /* NORMAL   */ {     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  0,  2 },
    /* FIRE     */ {     2,  1,  1,  2,  3,  3,  2,  2,  2,  2,  2,  3,  1,  2,  1 },
    /* WATER    */ {     2,  3,  1,  2,  1,  2,  2,  2,  3,  2,  2,  2,  3,  2,  1 },
    /* ELECTRIC */ {     2,  2,  3,  1,  1,  2,  2,  2,  0,  3,  2,  2,  2,  2,  1 },
    /* GRASS    */ {     2,  1,  3,  2,  1,  2,  2,  1,  3,  1,  2,  1,  3,  2,  1 },
    /* ICE      */ {     2,  2,  1,  2,  3,  1,  2,  2,  3,  3,  2,  2,  2,  2,  3 },
    /* FIGHTING */ {     3,  2,  2,  2,  2,  3,  2,  1,  2,  1,  1,  1,  3,  0,  2 },
    /* POISON   */ {     2,  2,  2,  2,  3,  2,  2,  1,  1,  2,  2,  3,  1,  1,  2 },
    /* GROUND   */ {     2,  3,  2,  3,  1,  2,  2,  3,  2,  0,  2,  1,  3,  2,  2 },
    /* FLYING   */ {     2,  2,  2,  1,  3,  2,  3,  2,  2,  2,  2,  3,  1,  2,  2 },
    /* PSYCHIC  */ {     2,  2,  2,  2,  2,  2,  3,  3,  2,  2,  1,  2,  2,  2,  2 },
    /* BUG      */ {     2,  1,  2,  2,  3,  2,  1,  3,  2,  1,  3,  2,  2,  1,  2 },
    /* ROCK     */ {     2,  3,  2,  2,  2,  3,  1,  2,  1,  3,  2,  3,  2,  2,  2 },
    /* GHOST    */ {     0,  2,  2,  2,  2,  2,  2,  2,  2,  2,  0,  2,  2,  3,  2 },
    /* DRAGON   */ {     2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3 },
};

// ============================================================================
// Move Data
// ============================================================================

typedef struct {
    const char* name;
    Type type;
    int power;        // 0 for status moves
    int accuracy;     // 0-100, 0 means always hits
    int max_pp;
    MoveEffect effect;
    int effect_chance; // percentage chance of secondary effect (0-100)
} MoveData;

static const MoveData MOVE_DATA[NUM_MOVES + 1] = {
    // [0] MOVE_NONE - placeholder
    { "None",          TYPE_NORMAL,   0,   0,  0, EFFECT_NONE,             0 },
    // [1] Body Slam - Normal, 85 power, 100 acc, 30% para
    { "Body Slam",     TYPE_NORMAL,  85, 100, 24, EFFECT_PARALYZE_CHANCE, 30 },
    // [2] Hyper Beam - Normal, 150 power, 90 acc, recharge
    { "Hyper Beam",    TYPE_NORMAL, 150,  90,  8, EFFECT_HYPER_BEAM,       0 },
    // [3] Earthquake - Ground, 100 power, 100 acc
    { "Earthquake",    TYPE_GROUND, 100, 100, 16, EFFECT_NONE,             0 },
    // [4] Blizzard - Ice, 120 power, 90 acc, 10% freeze (Gen1 cart)
    { "Blizzard",      TYPE_ICE,    120,  90,  8, EFFECT_FREEZE_CHANCE,   10 },
    // [5] Psychic - Psychic, 90 power, 100 acc, 33% lower special
    { "Psychic",       TYPE_PSYCHIC, 90, 100, 16, EFFECT_LOWER_SPECIAL,   33 },
    // [6] Thunderbolt - Electric, 95 power, 100 acc, 10% para
    { "Thunderbolt",   TYPE_ELECTRIC,95, 100, 24, EFFECT_PARALYZE_CHANCE, 10 },
    // [7] Ice Beam - Ice, 95 power, 100 acc, 10% freeze
    { "Ice Beam",      TYPE_ICE,     95, 100, 16, EFFECT_FREEZE_CHANCE,   10 },
    // [8] Surf - Water, 95 power, 100 acc
    { "Surf",          TYPE_WATER,   95, 100, 24, EFFECT_NONE,             0 },
    // [9] Fire Blast - Fire, 120 power, 85 acc, 30% burn
    { "Fire Blast",    TYPE_FIRE,   120,  85,  8, EFFECT_BURN_CHANCE,     30 },
    // [10] Rock Slide - Rock, 75 power, 90 acc
    { "Rock Slide",    TYPE_ROCK,    75,  90, 16, EFFECT_NONE,             0 },
    // [11] Drill Peck - Flying, 80 power, 100 acc
    { "Drill Peck",    TYPE_FLYING,  80, 100, 32, EFFECT_NONE,             0 },
    // [12] Night Shade - Ghost, fixed 100 damage, 100 acc
    { "Night Shade",   TYPE_GHOST,    0, 100, 24, EFFECT_FIXED_DAMAGE,     0 },
    // [13] Seismic Toss - Fighting, fixed 100 damage, 100 acc
    { "Seismic Toss",  TYPE_FIGHTING, 0, 100, 32, EFFECT_FIXED_DAMAGE,     0 },
    // [14] Double-Edge - Normal, 100 power, 100 acc, 25% recoil
    { "Double-Edge",   TYPE_NORMAL, 100, 100, 24, EFFECT_RECOIL,          25 },
    // [15] Explosion - Normal, 170 power, 100 acc, user faints
    { "Explosion",     TYPE_NORMAL, 170, 100,  8, EFFECT_SELF_DESTRUCT,    0 },
    // [16] Self-Destruct - Normal, 130 power, 100 acc, user faints
    { "Self-Destruct", TYPE_NORMAL, 130, 100,  8, EFFECT_SELF_DESTRUCT,    0 },
    // [17] Slash - Normal, 70 power, 100 acc, high crit
    { "Slash",         TYPE_NORMAL,  70, 100, 32, EFFECT_HIGH_CRIT,        0 },
    // [18] Bubble Beam - Water, 65 power, 100 acc, 33% lower speed
    { "Bubble Beam",   TYPE_WATER,   65, 100, 32, EFFECT_LOWER_SPEED,     33 },
    // [19] Thunder Wave - Electric, status, 100 acc, paralyze
    { "Thunder Wave",  TYPE_ELECTRIC, 0, 100, 32, EFFECT_THUNDER_WAVE,     0 },
    // [20] Sleep Powder - Grass, status, 75 acc, sleep
    { "Sleep Powder",  TYPE_GRASS,    0,  75, 24, EFFECT_SLEEP,            0 },
    // [21] Stun Spore - Grass, status, 75 acc, paralyze
    { "Stun Spore",    TYPE_GRASS,    0,  75, 48, EFFECT_THUNDER_WAVE,     0 },
    // [22] Lovely Kiss - Normal, status, 75 acc, sleep
    { "Lovely Kiss",   TYPE_NORMAL,   0,  75, 16, EFFECT_SLEEP,            0 },
    // [23] Hypnosis - Psychic, status, 60 acc, sleep
    { "Hypnosis",      TYPE_PSYCHIC,  0,  60, 32, EFFECT_SLEEP,            0 },
    // [24] Sing - Normal, status, 55 acc, sleep
    { "Sing",          TYPE_NORMAL,   0,  55, 24, EFFECT_SLEEP,            0 },
    // [25] Recover - Normal, status, heals 50% HP
    { "Recover",       TYPE_NORMAL,   0,   0, 32, EFFECT_RECOVER,          0 },
    // [26] Soft-Boiled - Normal, status, heals 50% HP
    { "Soft-Boiled",   TYPE_NORMAL,   0,   0, 16, EFFECT_RECOVER,          0 },
    // [27] Rest - Psychic, status, full heal + sleep
    { "Rest",          TYPE_PSYCHIC,  0,   0, 16, EFFECT_REST,             0 },
    // [28] Swords Dance - Normal, status, +2 attack
    { "Swords Dance",  TYPE_NORMAL,   0,   0, 48, EFFECT_BOOST_ATK_2,      0 },
    // [29] Amnesia - Psychic, status, +2 special
    { "Amnesia",       TYPE_PSYCHIC,  0,   0, 32, EFFECT_BOOST_SPC_2,      0 },
    // [30] Agility - Psychic, status, +2 speed
    { "Agility",       TYPE_PSYCHIC,  0,   0, 48, EFFECT_BOOST_SPE_2,      0 },
    // [31] Substitute - Normal, status, 25% HP
    { "Substitute",    TYPE_NORMAL,   0,   0, 16, EFFECT_SUBSTITUTE,       0 },
    // [32] Toxic - Poison, status, 85 acc, badly poison
    { "Toxic",         TYPE_POISON,   0,  85, 16, EFFECT_TOXIC,            0 },
    // [33] Reflect - Psychic, status
    { "Reflect",       TYPE_PSYCHIC,  0,   0, 32, EFFECT_REFLECT,          0 },
    // [34] Light Screen - Psychic, status
    { "Light Screen",  TYPE_PSYCHIC,  0,   0, 48, EFFECT_LIGHT_SCREEN,     0 },
    // [35] Confuse Ray - Ghost, status, 100 acc
    { "Confuse Ray",   TYPE_GHOST,    0, 100, 16, EFFECT_CONFUSE,          0 },
    // [36] Mega Drain - Grass, 40 power, 100 acc, drain
    { "Mega Drain",    TYPE_GRASS,   40, 100, 16, EFFECT_DRAIN,            0 },
    // [37] Thunder - Electric, 120 power, 70 acc, 10% para
    { "Thunder",       TYPE_ELECTRIC,120,  70,  8, EFFECT_PARALYZE_CHANCE, 10 },
    // [38] Pin Missile - Bug, 14 power, 85 acc, 2-5 hits
    { "Pin Missile",   TYPE_BUG,     14,  85, 32, EFFECT_MULTI_HIT,        0 },
    // [39] Leech Seed - Grass, status, 90 acc
    { "Leech Seed",    TYPE_GRASS,    0,  90, 16, EFFECT_LEECH_SEED,       0 },
    // [40] Tackle - Normal, 35 power, 95 acc
    { "Tackle",        TYPE_NORMAL,  35,  95, 56, EFFECT_NONE,             0 },
    // [41] String Shot - Bug, status, 95 acc, lowers speed
    { "String Shot",   TYPE_BUG,      0,  95, 64, EFFECT_LOWER_SPEED,     100 },
    // [42] Harden - Normal, status, +1 defense
    { "Harden",        TYPE_NORMAL,   0,   0, 48, EFFECT_BOOST_DEF_1,      0 },
    // [43] Splash - Normal, status, no effect
    { "Splash",        TYPE_NORMAL,   0,   0, 64, EFFECT_NONE,             0 },
    // [44] Transform - Normal, status, copy target species/types/moves
    { "Transform",     TYPE_NORMAL,   0,   0, 16, EFFECT_TRANSFORM,        0 },
};

// ============================================================================
// Pokemon Base Stats & Movesets
// Gen 1 stats: HP, Attack, Defense, Special, Speed (5 stats)
// At level 100 with max DVs (15) and max Stat Exp:
//   HP = 2 * base_hp + 203
//   Stat = 2 * base_stat + 98
// ============================================================================

typedef struct {
    const char* name;
    SpeciesID id;
    Type type1;
    Type type2;     // TYPE_NONE if mono-type
    int base_hp;    // raw base stat
    int base_atk;
    int base_def;
    int base_spc;
    int base_spe;
    MoveID moveset[NUM_MOVE_SLOTS]; // Default competitive moveset
} SpeciesData;

static const SpeciesData SPECIES_DATA[NUM_SPECIES + 1] = {
    { "None",       SPECIES_NONE,      TYPE_NORMAL, TYPE_NONE,   0,  0,  0,  0,  0,
      { MOVE_NONE, MOVE_NONE, MOVE_NONE, MOVE_NONE } },
    { "Tauros", SPECIES_TAUROS, TYPE_NORMAL, TYPE_NONE,  75,100, 95, 70,110,
      { MOVE_BODY_SLAM, MOVE_HYPER_BEAM, MOVE_EARTHQUAKE, MOVE_BLIZZARD } },
    { "Chansey", SPECIES_CHANSEY, TYPE_NORMAL, TYPE_NONE, 250,  5,  5,105, 50,
      { MOVE_THUNDER_WAVE, MOVE_ICE_BEAM, MOVE_SOFT_BOILED, MOVE_SEISMIC_TOSS } },
    { "Snorlax", SPECIES_SNORLAX, TYPE_NORMAL, TYPE_NONE, 160,110, 65, 65, 30,
      { MOVE_BODY_SLAM, MOVE_EARTHQUAKE, MOVE_SELF_DESTRUCT, MOVE_REST } },
    { "Alakazam", SPECIES_ALAKAZAM, TYPE_PSYCHIC, TYPE_NONE,  55, 50, 45,135,120,
      { MOVE_PSYCHIC, MOVE_THUNDER_WAVE, MOVE_SEISMIC_TOSS, MOVE_RECOVER } },
    { "Exeggutor", SPECIES_EXEGGUTOR, TYPE_GRASS, TYPE_PSYCHIC,  95, 95, 85,125, 55,
      { MOVE_PSYCHIC, MOVE_SLEEP_POWDER, MOVE_STUN_SPORE, MOVE_EXPLOSION } },
    { "Starmie", SPECIES_STARMIE, TYPE_WATER, TYPE_PSYCHIC,  60, 75, 85,100,115,
      { MOVE_PSYCHIC, MOVE_THUNDER_WAVE, MOVE_THUNDERBOLT, MOVE_RECOVER } },
    { "Gengar", SPECIES_GENGAR, TYPE_GHOST, TYPE_POISON,  60, 65, 60,130,110,
      { MOVE_HYPNOSIS, MOVE_THUNDERBOLT, MOVE_NIGHT_SHADE, MOVE_EXPLOSION } },
    { "Jynx", SPECIES_JYNX, TYPE_ICE, TYPE_PSYCHIC,  65, 50, 35, 95, 95,
      { MOVE_BLIZZARD, MOVE_PSYCHIC, MOVE_LOVELY_KISS, MOVE_REST } },
    { "Zapdos", SPECIES_ZAPDOS, TYPE_ELECTRIC, TYPE_FLYING,  90, 90, 85,125,100,
      { MOVE_THUNDERBOLT, MOVE_DRILL_PECK, MOVE_THUNDER_WAVE, MOVE_AGILITY } },
    { "Rhydon", SPECIES_RHYDON, TYPE_GROUND, TYPE_ROCK, 105,130,120, 45, 40,
      { MOVE_EARTHQUAKE, MOVE_ROCK_SLIDE, MOVE_SUBSTITUTE, MOVE_BODY_SLAM } },
    { "Cloyster", SPECIES_CLOYSTER, TYPE_WATER, TYPE_ICE,  50, 95,180, 85, 70,
      { MOVE_BLIZZARD, MOVE_HYPER_BEAM, MOVE_EXPLOSION, MOVE_SURF } },
    { "Golem", SPECIES_GOLEM, TYPE_ROCK, TYPE_GROUND,  80,110,130, 55, 45,
      { MOVE_EARTHQUAKE, MOVE_ROCK_SLIDE, MOVE_EXPLOSION, MOVE_BODY_SLAM } },
    { "Lapras", SPECIES_LAPRAS, TYPE_WATER, TYPE_ICE, 130, 85, 80, 95, 60,
      { MOVE_BLIZZARD, MOVE_THUNDERBOLT, MOVE_BODY_SLAM, MOVE_SING } },
    { "Slowbro", SPECIES_SLOWBRO, TYPE_WATER, TYPE_PSYCHIC,  95, 75,110, 80, 30,
      { MOVE_PSYCHIC, MOVE_THUNDER_WAVE, MOVE_AMNESIA, MOVE_REST } },
    { "Jolteon", SPECIES_JOLTEON, TYPE_ELECTRIC, TYPE_NONE,  65, 65, 60,110,130,
      { MOVE_THUNDERBOLT, MOVE_THUNDER_WAVE, MOVE_PIN_MISSILE, MOVE_DOUBLE_EDGE } },
    { "Persian", SPECIES_PERSIAN, TYPE_NORMAL, TYPE_NONE,  65, 70, 60, 65,115,
      { MOVE_SLASH, MOVE_HYPER_BEAM, MOVE_BUBBLE_BEAM, MOVE_THUNDERBOLT } },
    { "Hypno", SPECIES_HYPNO, TYPE_PSYCHIC, TYPE_NONE,  85, 73, 70,115, 67,
      { MOVE_PSYCHIC, MOVE_HYPNOSIS, MOVE_THUNDER_WAVE, MOVE_SEISMIC_TOSS } },
    { "Articuno", SPECIES_ARTICUNO, TYPE_ICE, TYPE_FLYING,  90, 85,100,125, 85,
      { MOVE_BLIZZARD, MOVE_AGILITY, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE } },
    { "Dragonite", SPECIES_DRAGONITE, TYPE_DRAGON, TYPE_FLYING,  91,134, 95,100, 80,
      { MOVE_HYPER_BEAM, MOVE_BLIZZARD, MOVE_THUNDERBOLT, MOVE_AGILITY } },
    { "Machamp", SPECIES_MACHAMP, TYPE_FIGHTING, TYPE_NONE,  90,130, 80, 65, 55,
      { MOVE_BODY_SLAM, MOVE_EARTHQUAKE, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE } },
    { "Bulbasaur", SPECIES_BULBASAUR, TYPE_GRASS, TYPE_POISON,  45, 49, 49, 65, 45,
      { MOVE_SLEEP_POWDER, MOVE_MEGA_DRAIN, MOVE_DOUBLE_EDGE, MOVE_LEECH_SEED } },
    { "Ivysaur", SPECIES_IVYSAUR, TYPE_GRASS, TYPE_POISON,  60, 62, 63, 80, 60,
      { MOVE_SLEEP_POWDER, MOVE_MEGA_DRAIN, MOVE_DOUBLE_EDGE, MOVE_LEECH_SEED } },
    { "Venusaur", SPECIES_VENUSAUR, TYPE_GRASS, TYPE_POISON,  80, 82, 83,100, 80,
      { MOVE_SLEEP_POWDER, MOVE_HYPER_BEAM, MOVE_MEGA_DRAIN, MOVE_DOUBLE_EDGE } },
    { "Charmander", SPECIES_CHARMANDER, TYPE_FIRE, TYPE_NONE,  39, 52, 43, 50, 65,
      { MOVE_REST, MOVE_FIRE_BLAST, MOVE_SEISMIC_TOSS, MOVE_DOUBLE_EDGE } },
    { "Charmeleon", SPECIES_CHARMELEON, TYPE_FIRE, TYPE_NONE,  58, 64, 58, 65, 80,
      { MOVE_REST, MOVE_FIRE_BLAST, MOVE_SEISMIC_TOSS, MOVE_DOUBLE_EDGE } },
    { "Charizard", SPECIES_CHARIZARD, TYPE_FIRE, TYPE_FLYING,  78, 84, 78, 85,100,
      { MOVE_REST, MOVE_FIRE_BLAST, MOVE_HYPER_BEAM, MOVE_EARTHQUAKE } },
    { "Squirtle", SPECIES_SQUIRTLE, TYPE_WATER, TYPE_NONE,  44, 48, 65, 50, 43,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_BLIZZARD } },
    { "Wartortle", SPECIES_WARTORTLE, TYPE_WATER, TYPE_NONE,  59, 63, 80, 65, 58,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_BLIZZARD } },
    { "Blastoise", SPECIES_BLASTOISE, TYPE_WATER, TYPE_NONE,  79, 83,100, 85, 78,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_HYPER_BEAM } },
    { "Caterpie", SPECIES_CATERPIE, TYPE_BUG, TYPE_NONE,  45, 30, 35, 20, 45,
      { MOVE_NONE, MOVE_NONE, MOVE_TACKLE, MOVE_STRING_SHOT } },
    { "Metapod", SPECIES_METAPOD, TYPE_BUG, TYPE_NONE,  50, 20, 55, 25, 30,
      { MOVE_NONE, MOVE_TACKLE, MOVE_HARDEN, MOVE_STRING_SHOT } },
    { "Butterfree", SPECIES_BUTTERFREE, TYPE_BUG, TYPE_FLYING,  60, 45, 50, 80, 70,
      { MOVE_SLEEP_POWDER, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE, MOVE_PSYCHIC } },
    { "Weedle", SPECIES_WEEDLE, TYPE_BUG, TYPE_POISON,  40, 35, 30, 20, 50,
      { MOVE_NONE, MOVE_NONE, MOVE_NONE, MOVE_STRING_SHOT } },
    { "Kakuna", SPECIES_KAKUNA, TYPE_BUG, TYPE_POISON,  45, 25, 50, 25, 35,
      { MOVE_NONE, MOVE_NONE, MOVE_NONE, MOVE_STRING_SHOT } },
    { "Beedrill", SPECIES_BEEDRILL, TYPE_BUG, TYPE_POISON,  65, 80, 40, 45, 75,
      { MOVE_REST, MOVE_HYPER_BEAM, MOVE_PIN_MISSILE, MOVE_DOUBLE_EDGE } },
    { "Pidgey", SPECIES_PIDGEY, TYPE_NORMAL, TYPE_FLYING,  40, 45, 40, 35, 56,
      { MOVE_REST, MOVE_DOUBLE_EDGE, MOVE_SUBSTITUTE, MOVE_AGILITY } },
    { "Pidgeotto", SPECIES_PIDGEOTTO, TYPE_NORMAL, TYPE_FLYING,  63, 60, 55, 50, 71,
      { MOVE_REST, MOVE_DOUBLE_EDGE, MOVE_SUBSTITUTE, MOVE_AGILITY } },
    { "Pidgeot", SPECIES_PIDGEOT, TYPE_NORMAL, TYPE_FLYING,  83, 80, 75, 70, 91,
      { MOVE_REST, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE, MOVE_AGILITY } },
    { "Rattata", SPECIES_RATTATA, TYPE_NORMAL, TYPE_NONE,  30, 56, 35, 25, 72,
      { MOVE_REST, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM, MOVE_TACKLE } },
    { "Raticate", SPECIES_RATICATE, TYPE_NORMAL, TYPE_NONE,  55, 81, 60, 50, 97,
      { MOVE_REST, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM } },
    { "Spearow", SPECIES_SPEAROW, TYPE_NORMAL, TYPE_FLYING,  40, 60, 30, 31, 70,
      { MOVE_REST, MOVE_DOUBLE_EDGE, MOVE_DRILL_PECK, MOVE_AGILITY } },
    { "Fearow", SPECIES_FEAROW, TYPE_NORMAL, TYPE_FLYING,  65, 90, 65, 61,100,
      { MOVE_REST, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE, MOVE_DRILL_PECK } },
    { "Ekans", SPECIES_EKANS, TYPE_POISON, TYPE_NONE,  35, 60, 44, 40, 55,
      { MOVE_REST, MOVE_EARTHQUAKE, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM } },
    { "Arbok", SPECIES_ARBOK, TYPE_POISON, TYPE_NONE,  60, 85, 69, 65, 80,
      { MOVE_REST, MOVE_HYPER_BEAM, MOVE_EARTHQUAKE, MOVE_DOUBLE_EDGE } },
    { "Pikachu", SPECIES_PIKACHU, TYPE_ELECTRIC, TYPE_NONE,  35, 55, 30, 50, 90,
      { MOVE_THUNDER_WAVE, MOVE_THUNDERSHOCK, MOVE_THUNDERBOLT, MOVE_SEISMIC_TOSS } },
    { "Raichu", SPECIES_RAICHU, TYPE_ELECTRIC, TYPE_NONE,  60, 90, 55, 90,100,
      { MOVE_THUNDER_WAVE, MOVE_THUNDERSHOCK, MOVE_THUNDERBOLT, MOVE_HYPER_BEAM } },
    { "Sandshrew", SPECIES_SANDSHREW, TYPE_GROUND, TYPE_NONE,  50, 75, 85, 30, 40,
      { MOVE_REST, MOVE_EARTHQUAKE, MOVE_DOUBLE_EDGE, MOVE_SEISMIC_TOSS } },
    { "Sandslash", SPECIES_SANDSLASH, TYPE_GROUND, TYPE_NONE,  75,100,110, 55, 65,
      { MOVE_REST, MOVE_EARTHQUAKE, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE } },
    { "Nidoran-F", SPECIES_NIDORAN_F, TYPE_POISON, TYPE_NONE,  55, 47, 52, 40, 41,
      { MOVE_REST, MOVE_BLIZZARD, MOVE_THUNDERSHOCK, MOVE_DOUBLE_EDGE } },
    { "Nidorina", SPECIES_NIDORINA, TYPE_POISON, TYPE_NONE,  70, 62, 67, 55, 56,
      { MOVE_REST, MOVE_BLIZZARD, MOVE_THUNDERSHOCK, MOVE_DOUBLE_EDGE } },
    { "Nidoqueen", SPECIES_NIDOQUEEN, TYPE_POISON, TYPE_GROUND,  90, 82, 87, 75, 76,
      { MOVE_REST, MOVE_EARTHQUAKE, MOVE_HYPER_BEAM, MOVE_BLIZZARD } },
    { "Nidoran-M", SPECIES_NIDORAN_M, TYPE_POISON, TYPE_NONE,  46, 57, 40, 40, 50,
      { MOVE_REST, MOVE_BLIZZARD, MOVE_THUNDERSHOCK, MOVE_DOUBLE_EDGE } },
    { "Nidorino", SPECIES_NIDORINO, TYPE_POISON, TYPE_NONE,  61, 72, 57, 55, 65,
      { MOVE_REST, MOVE_BLIZZARD, MOVE_THUNDERSHOCK, MOVE_DOUBLE_EDGE } },
    { "Nidoking", SPECIES_NIDOKING, TYPE_POISON, TYPE_GROUND,  81, 92, 77, 75, 85,
      { MOVE_REST, MOVE_EARTHQUAKE, MOVE_HYPER_BEAM, MOVE_BLIZZARD } },
    { "Clefairy", SPECIES_CLEFAIRY, TYPE_NORMAL, TYPE_NONE,  70, 45, 48, 60, 35,
      { MOVE_SING, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM, MOVE_BLIZZARD } },
    { "Clefable", SPECIES_CLEFABLE, TYPE_NORMAL, TYPE_NONE,  95, 70, 73, 85, 60,
      { MOVE_SING, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM } },
    { "Vulpix", SPECIES_VULPIX, TYPE_FIRE, TYPE_NONE,  38, 41, 40, 65, 65,
      { MOVE_REST, MOVE_FIRE_BLAST, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM } },
    { "Ninetales", SPECIES_NINETALES, TYPE_FIRE, TYPE_NONE,  73, 76, 75,100,100,
      { MOVE_REST, MOVE_FIRE_BLAST, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE } },
    { "Jigglypuff", SPECIES_JIGGLYPUFF, TYPE_NORMAL, TYPE_NONE, 115, 45, 20, 25, 20,
      { MOVE_SING, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM, MOVE_BLIZZARD } },
    { "Wigglytuff", SPECIES_WIGGLYTUFF, TYPE_NORMAL, TYPE_NONE, 140, 70, 45, 50, 45,
      { MOVE_SING, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM } },
    { "Zubat", SPECIES_ZUBAT, TYPE_POISON, TYPE_FLYING,  40, 45, 35, 40, 55,
      { MOVE_REST, MOVE_DOUBLE_EDGE, MOVE_MEGA_DRAIN, MOVE_CONFUSE_RAY } },
    { "Golbat", SPECIES_GOLBAT, TYPE_POISON, TYPE_FLYING,  75, 80, 70, 75, 90,
      { MOVE_REST, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE, MOVE_MEGA_DRAIN } },
    { "Oddish", SPECIES_ODDISH, TYPE_GRASS, TYPE_POISON,  45, 50, 55, 75, 30,
      { MOVE_SLEEP_POWDER, MOVE_MEGA_DRAIN, MOVE_DOUBLE_EDGE, MOVE_SWORDS_DANCE } },
    { "Gloom", SPECIES_GLOOM, TYPE_GRASS, TYPE_POISON,  60, 65, 70, 85, 40,
      { MOVE_SLEEP_POWDER, MOVE_MEGA_DRAIN, MOVE_DOUBLE_EDGE, MOVE_SWORDS_DANCE } },
    { "Vileplume", SPECIES_VILEPLUME, TYPE_GRASS, TYPE_POISON,  75, 80, 85,100, 50,
      { MOVE_SLEEP_POWDER, MOVE_HYPER_BEAM, MOVE_MEGA_DRAIN, MOVE_DOUBLE_EDGE } },
    { "Paras", SPECIES_PARAS, TYPE_BUG, TYPE_GRASS,  35, 70, 55, 55, 25,
      { MOVE_REST, MOVE_MEGA_DRAIN, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM } },
    { "Parasect", SPECIES_PARASECT, TYPE_BUG, TYPE_GRASS,  60, 95, 80, 80, 30,
      { MOVE_REST, MOVE_HYPER_BEAM, MOVE_MEGA_DRAIN, MOVE_DOUBLE_EDGE } },
    { "Venonat", SPECIES_VENONAT, TYPE_BUG, TYPE_POISON,  60, 55, 50, 40, 45,
      { MOVE_SLEEP_POWDER, MOVE_DOUBLE_EDGE, MOVE_PSYCHIC, MOVE_MEGA_DRAIN } },
    { "Venomoth", SPECIES_VENOMOTH, TYPE_BUG, TYPE_POISON,  70, 65, 60, 90, 90,
      { MOVE_SLEEP_POWDER, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE, MOVE_PSYCHIC } },
    { "Diglett", SPECIES_DIGLETT, TYPE_GROUND, TYPE_NONE,  10, 55, 25, 45, 95,
      { MOVE_REST, MOVE_EARTHQUAKE, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM } },
    { "Dugtrio", SPECIES_DUGTRIO, TYPE_GROUND, TYPE_NONE,  35, 80, 50, 70,120,
      { MOVE_REST, MOVE_EARTHQUAKE, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE } },
    { "Meowth", SPECIES_MEOWTH, TYPE_NORMAL, TYPE_NONE,  40, 45, 35, 40, 90,
      { MOVE_THUNDERBOLT, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM, MOVE_SLASH } },
    { "Psyduck", SPECIES_PSYDUCK, TYPE_WATER, TYPE_NONE,  50, 52, 48, 50, 55,
      { MOVE_BODY_SLAM, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_BLIZZARD } },
    { "Golduck", SPECIES_GOLDUCK, TYPE_WATER, TYPE_NONE,  80, 82, 78, 80, 85,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_HYPER_BEAM } },
    { "Mankey", SPECIES_MANKEY, TYPE_FIGHTING, TYPE_NONE,  40, 80, 35, 35, 70,
      { MOVE_REST, MOVE_SEISMIC_TOSS, MOVE_THUNDERSHOCK, MOVE_DOUBLE_EDGE } },
    { "Primeape", SPECIES_PRIMEAPE, TYPE_FIGHTING, TYPE_NONE,  65,105, 60, 60, 95,
      { MOVE_REST, MOVE_SEISMIC_TOSS, MOVE_HYPER_BEAM, MOVE_THUNDERSHOCK } },
    { "Growlithe", SPECIES_GROWLITHE, TYPE_FIRE, TYPE_NONE,  55, 70, 45, 50, 60,
      { MOVE_REST, MOVE_FIRE_BLAST, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM } },
    { "Arcanine", SPECIES_ARCANINE, TYPE_FIRE, TYPE_NONE,  90,110, 80, 80, 95,
      { MOVE_REST, MOVE_FIRE_BLAST, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE } },
    { "Poliwag", SPECIES_POLIWAG, TYPE_WATER, TYPE_NONE,  40, 50, 40, 40, 90,
      { MOVE_HYPNOSIS, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_BLIZZARD } },
    { "Poliwhirl", SPECIES_POLIWHIRL, TYPE_WATER, TYPE_NONE,  65, 65, 65, 50, 90,
      { MOVE_HYPNOSIS, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_BLIZZARD } },
    { "Poliwrath", SPECIES_POLIWRATH, TYPE_WATER, TYPE_FIGHTING,  90, 85, 95, 70, 70,
      { MOVE_HYPNOSIS, MOVE_SEISMIC_TOSS, MOVE_SURF, MOVE_BUBBLE_BEAM } },
    { "Abra", SPECIES_ABRA, TYPE_PSYCHIC, TYPE_NONE,  25, 20, 15,105, 90,
      { MOVE_THUNDER_WAVE, MOVE_PSYCHIC, MOVE_SEISMIC_TOSS, MOVE_DOUBLE_EDGE } },
    { "Kadabra", SPECIES_KADABRA, TYPE_PSYCHIC, TYPE_NONE,  40, 35, 30,120,105,
      { MOVE_THUNDER_WAVE, MOVE_PSYCHIC, MOVE_SEISMIC_TOSS, MOVE_DOUBLE_EDGE } },
    { "Machop", SPECIES_MACHOP, TYPE_FIGHTING, TYPE_NONE,  70, 80, 50, 35, 35,
      { MOVE_REST, MOVE_SEISMIC_TOSS, MOVE_FIRE_BLAST, MOVE_EARTHQUAKE } },
    { "Machoke", SPECIES_MACHOKE, TYPE_FIGHTING, TYPE_NONE,  80,100, 70, 50, 45,
      { MOVE_REST, MOVE_SEISMIC_TOSS, MOVE_FIRE_BLAST, MOVE_EARTHQUAKE } },
    { "Bellsprout", SPECIES_BELLSPROUT, TYPE_GRASS, TYPE_POISON,  50, 75, 35, 70, 40,
      { MOVE_SLEEP_POWDER, MOVE_MEGA_DRAIN, MOVE_DOUBLE_EDGE, MOVE_SWORDS_DANCE } },
    { "Weepinbell", SPECIES_WEEPINBELL, TYPE_GRASS, TYPE_POISON,  65, 90, 50, 85, 55,
      { MOVE_SLEEP_POWDER, MOVE_MEGA_DRAIN, MOVE_DOUBLE_EDGE, MOVE_SWORDS_DANCE } },
    { "Victreebel", SPECIES_VICTREEBEL, TYPE_GRASS, TYPE_POISON,  80,105, 65,100, 70,
      { MOVE_SLEEP_POWDER, MOVE_HYPER_BEAM, MOVE_MEGA_DRAIN, MOVE_DOUBLE_EDGE } },
    { "Tentacool", SPECIES_TENTACOOL, TYPE_WATER, TYPE_POISON,  40, 40, 35,100, 70,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_BLIZZARD } },
    { "Tentacruel", SPECIES_TENTACRUEL, TYPE_WATER, TYPE_POISON,  80, 70, 65,120,100,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_HYPER_BEAM } },
    { "Geodude", SPECIES_GEODUDE, TYPE_ROCK, TYPE_GROUND,  40, 80,100, 30, 20,
      { MOVE_REST, MOVE_EARTHQUAKE, MOVE_ROCK_SLIDE, MOVE_EXPLOSION } },
    { "Graveler", SPECIES_GRAVELER, TYPE_ROCK, TYPE_GROUND,  55, 95,115, 45, 35,
      { MOVE_REST, MOVE_EARTHQUAKE, MOVE_ROCK_SLIDE, MOVE_EXPLOSION } },
    { "Ponyta", SPECIES_PONYTA, TYPE_FIRE, TYPE_NONE,  50, 85, 55, 65, 90,
      { MOVE_REST, MOVE_FIRE_BLAST, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM } },
    { "Rapidash", SPECIES_RAPIDASH, TYPE_FIRE, TYPE_NONE,  65,100, 70, 80,105,
      { MOVE_REST, MOVE_FIRE_BLAST, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE } },
    { "Slowpoke", SPECIES_SLOWPOKE, TYPE_WATER, TYPE_PSYCHIC,  90, 65, 65, 40, 15,
      { MOVE_THUNDER_WAVE, MOVE_SURF, MOVE_PSYCHIC, MOVE_BUBBLE_BEAM } },
    { "Magnemite", SPECIES_MAGNEMITE, TYPE_ELECTRIC, TYPE_NONE,  25, 35, 70, 95, 45,
      { MOVE_THUNDER_WAVE, MOVE_THUNDERSHOCK, MOVE_THUNDERBOLT, MOVE_DOUBLE_EDGE } },
    { "Magneton", SPECIES_MAGNETON, TYPE_ELECTRIC, TYPE_NONE,  50, 60, 95,120, 70,
      { MOVE_THUNDER_WAVE, MOVE_THUNDERSHOCK, MOVE_THUNDERBOLT, MOVE_HYPER_BEAM } },
    { "Farfetch'd", SPECIES_FARFETCH_D, TYPE_NORMAL, TYPE_FLYING,  52, 65, 55, 58, 60,
      { MOVE_REST, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM, MOVE_SLASH } },
    { "Doduo", SPECIES_DODUO, TYPE_NORMAL, TYPE_FLYING,  35, 85, 45, 35, 75,
      { MOVE_REST, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM, MOVE_DRILL_PECK } },
    { "Dodrio", SPECIES_DODRIO, TYPE_NORMAL, TYPE_FLYING,  60,110, 70, 60,100,
      { MOVE_REST, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM } },
    { "Seel", SPECIES_SEEL, TYPE_WATER, TYPE_NONE,  65, 45, 55, 70, 45,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_BLIZZARD } },
    { "Dewgong", SPECIES_DEWGONG, TYPE_WATER, TYPE_ICE,  90, 70, 80, 95, 70,
      { MOVE_REST, MOVE_BLIZZARD, MOVE_SURF, MOVE_ICE_BEAM } },
    { "Grimer", SPECIES_GRIMER, TYPE_POISON, TYPE_NONE,  80, 80, 50, 40, 25,
      { MOVE_REST, MOVE_EXPLOSION, MOVE_SELF_DESTRUCT, MOVE_FIRE_BLAST } },
    { "Muk", SPECIES_MUK, TYPE_POISON, TYPE_NONE, 105,105, 75, 65, 50,
      { MOVE_REST, MOVE_EXPLOSION, MOVE_HYPER_BEAM, MOVE_SELF_DESTRUCT } },
    { "Shellder", SPECIES_SHELLDER, TYPE_WATER, TYPE_NONE,  30, 65,100, 45, 40,
      { MOVE_REST, MOVE_SURF, MOVE_EXPLOSION, MOVE_BUBBLE_BEAM } },
    { "Gastly", SPECIES_GASTLY, TYPE_GHOST, TYPE_POISON,  30, 35, 30,100, 80,
      { MOVE_HYPNOSIS, MOVE_NIGHT_SHADE, MOVE_EXPLOSION, MOVE_SELF_DESTRUCT } },
    { "Haunter", SPECIES_HAUNTER, TYPE_GHOST, TYPE_POISON,  45, 50, 45,115, 95,
      { MOVE_HYPNOSIS, MOVE_NIGHT_SHADE, MOVE_EXPLOSION, MOVE_SELF_DESTRUCT } },
    { "Onix", SPECIES_ONIX, TYPE_ROCK, TYPE_GROUND,  35, 45,160, 30, 70,
      { MOVE_REST, MOVE_EARTHQUAKE, MOVE_ROCK_SLIDE, MOVE_EXPLOSION } },
    { "Drowzee", SPECIES_DROWZEE, TYPE_PSYCHIC, TYPE_NONE,  60, 48, 45, 90, 42,
      { MOVE_HYPNOSIS, MOVE_PSYCHIC, MOVE_SEISMIC_TOSS, MOVE_DOUBLE_EDGE } },
    { "Krabby", SPECIES_KRABBY, TYPE_WATER, TYPE_NONE,  30,105, 90, 25, 50,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_BLIZZARD } },
    { "Kingler", SPECIES_KINGLER, TYPE_WATER, TYPE_NONE,  55,130,115, 50, 75,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_HYPER_BEAM } },
    { "Voltorb", SPECIES_VOLTORB, TYPE_ELECTRIC, TYPE_NONE,  40, 30, 50, 55,100,
      { MOVE_THUNDER_WAVE, MOVE_THUNDERSHOCK, MOVE_THUNDERBOLT, MOVE_EXPLOSION } },
    { "Electrode", SPECIES_ELECTRODE, TYPE_ELECTRIC, TYPE_NONE,  60, 50, 70, 80,140,
      { MOVE_THUNDER_WAVE, MOVE_THUNDERSHOCK, MOVE_THUNDERBOLT, MOVE_EXPLOSION } },
    { "Exeggcute", SPECIES_EXEGGCUTE, TYPE_GRASS, TYPE_PSYCHIC,  60, 40, 80, 60, 40,
      { MOVE_SLEEP_POWDER, MOVE_PSYCHIC, MOVE_EXPLOSION, MOVE_DOUBLE_EDGE } },
    { "Cubone", SPECIES_CUBONE, TYPE_GROUND, TYPE_NONE,  50, 50, 95, 40, 35,
      { MOVE_REST, MOVE_EARTHQUAKE, MOVE_BLIZZARD, MOVE_FIRE_BLAST } },
    { "Marowak", SPECIES_MAROWAK, TYPE_GROUND, TYPE_NONE,  60, 80,110, 50, 45,
      { MOVE_REST, MOVE_EARTHQUAKE, MOVE_HYPER_BEAM, MOVE_BLIZZARD } },
    { "Hitmonlee", SPECIES_HITMONLEE, TYPE_FIGHTING, TYPE_NONE,  50,120, 53, 35, 87,
      { MOVE_REST, MOVE_SEISMIC_TOSS, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM } },
    { "Hitmonchan", SPECIES_HITMONCHAN, TYPE_FIGHTING, TYPE_NONE,  50,105, 79, 35, 76,
      { MOVE_REST, MOVE_SEISMIC_TOSS, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM } },
    { "Lickitung", SPECIES_LICKITUNG, TYPE_NORMAL, TYPE_NONE,  90, 55, 75, 60, 30,
      { MOVE_REST, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM } },
    { "Koffing", SPECIES_KOFFING, TYPE_POISON, TYPE_NONE,  40, 65, 95, 60, 35,
      { MOVE_REST, MOVE_EXPLOSION, MOVE_SELF_DESTRUCT, MOVE_FIRE_BLAST } },
    { "Weezing", SPECIES_WEEZING, TYPE_POISON, TYPE_NONE,  65, 90,120, 85, 60,
      { MOVE_REST, MOVE_EXPLOSION, MOVE_HYPER_BEAM, MOVE_SELF_DESTRUCT } },
    { "Rhyhorn", SPECIES_RHYHORN, TYPE_GROUND, TYPE_ROCK,  80, 85, 95, 30, 25,
      { MOVE_REST, MOVE_EARTHQUAKE, MOVE_ROCK_SLIDE, MOVE_BODY_SLAM } },
    { "Tangela", SPECIES_TANGELA, TYPE_GRASS, TYPE_NONE,  65, 55,115,100, 60,
      { MOVE_SLEEP_POWDER, MOVE_HYPER_BEAM, MOVE_MEGA_DRAIN, MOVE_DOUBLE_EDGE } },
    { "Kangaskhan", SPECIES_KANGASKHAN, TYPE_NORMAL, TYPE_NONE, 105, 95, 80, 40, 90,
      { MOVE_REST, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM } },
    { "Horsea", SPECIES_HORSEA, TYPE_WATER, TYPE_NONE,  30, 40, 70, 70, 60,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_BLIZZARD } },
    { "Seadra", SPECIES_SEADRA, TYPE_WATER, TYPE_NONE,  55, 65, 95, 95, 85,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_HYPER_BEAM } },
    { "Goldeen", SPECIES_GOLDEEN, TYPE_WATER, TYPE_NONE,  45, 67, 60, 50, 63,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_BLIZZARD } },
    { "Seaking", SPECIES_SEAKING, TYPE_WATER, TYPE_NONE,  80, 92, 65, 80, 68,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_HYPER_BEAM } },
    { "Staryu", SPECIES_STARYU, TYPE_WATER, TYPE_NONE,  30, 45, 55, 70, 85,
      { MOVE_THUNDER_WAVE, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_BLIZZARD } },
    { "Mr. Mime", SPECIES_MR_MIME, TYPE_PSYCHIC, TYPE_NONE,  40, 45, 65,100, 90,
      { MOVE_BODY_SLAM, MOVE_PSYCHIC, MOVE_HYPER_BEAM, MOVE_THUNDERSHOCK } },
    { "Scyther", SPECIES_SCYTHER, TYPE_BUG, TYPE_FLYING,  70,110, 80, 55,105,
      { MOVE_REST, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE, MOVE_SLASH } },
    { "Electabuzz", SPECIES_ELECTABUZZ, TYPE_ELECTRIC, TYPE_NONE,  65, 83, 57, 85,105,
      { MOVE_THUNDER_WAVE, MOVE_THUNDERSHOCK, MOVE_THUNDERBOLT, MOVE_HYPER_BEAM } },
    { "Magmar", SPECIES_MAGMAR, TYPE_FIRE, TYPE_NONE,  65, 95, 57, 85, 93,
      { MOVE_REST, MOVE_FIRE_BLAST, MOVE_HYPER_BEAM, MOVE_SEISMIC_TOSS } },
    { "Pinsir", SPECIES_PINSIR, TYPE_BUG, TYPE_NONE,  65,125,100, 55, 85,
      { MOVE_REST, MOVE_HYPER_BEAM, MOVE_SEISMIC_TOSS, MOVE_DOUBLE_EDGE } },
    { "Magikarp", SPECIES_MAGIKARP, TYPE_WATER, TYPE_NONE,  20, 10, 55, 20, 80,
      { MOVE_NONE, MOVE_NONE, MOVE_SPLASH, MOVE_TACKLE } },
    { "Gyarados", SPECIES_GYARADOS, TYPE_WATER, TYPE_FLYING,  95,125, 79,100, 81,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_HYPER_BEAM } },
    { "Ditto", SPECIES_DITTO, TYPE_NORMAL, TYPE_NONE,  48, 48, 48, 48, 48,
      { MOVE_NONE, MOVE_NONE, MOVE_NONE, MOVE_TRANSFORM } },
    { "Eevee", SPECIES_EEVEE, TYPE_NORMAL, TYPE_NONE,  55, 55, 50, 65, 55,
      { MOVE_REST, MOVE_DOUBLE_EDGE, MOVE_BODY_SLAM, MOVE_TACKLE } },
    { "Vaporeon", SPECIES_VAPOREON, TYPE_WATER, TYPE_NONE, 130, 65, 60,110, 65,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_HYPER_BEAM } },
    { "Flareon", SPECIES_FLAREON, TYPE_FIRE, TYPE_NONE,  65,130, 60,110, 65,
      { MOVE_REST, MOVE_FIRE_BLAST, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE } },
    { "Porygon", SPECIES_PORYGON, TYPE_NORMAL, TYPE_NONE,  65, 60, 70, 75, 40,
      { MOVE_THUNDER_WAVE, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE, MOVE_TACKLE } },
    { "Omanyte", SPECIES_OMANYTE, TYPE_ROCK, TYPE_WATER,  35, 40,100, 90, 35,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_BLIZZARD } },
    { "Omastar", SPECIES_OMASTAR, TYPE_ROCK, TYPE_WATER,  70, 60,125,115, 55,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_HYPER_BEAM } },
    { "Kabuto", SPECIES_KABUTO, TYPE_ROCK, TYPE_WATER,  30, 80, 90, 45, 55,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_BLIZZARD } },
    { "Kabutops", SPECIES_KABUTOPS, TYPE_ROCK, TYPE_WATER,  60,115,105, 70, 80,
      { MOVE_REST, MOVE_SURF, MOVE_BUBBLE_BEAM, MOVE_HYPER_BEAM } },
    { "Aerodactyl", SPECIES_AERODACTYL, TYPE_ROCK, TYPE_FLYING,  80,105, 65, 60,130,
      { MOVE_REST, MOVE_HYPER_BEAM, MOVE_FIRE_BLAST, MOVE_DOUBLE_EDGE } },
    { "Moltres", SPECIES_MOLTRES, TYPE_FIRE, TYPE_FLYING,  90,100, 90,125, 90,
      { MOVE_REST, MOVE_FIRE_BLAST, MOVE_HYPER_BEAM, MOVE_DOUBLE_EDGE } },
    { "Dratini", SPECIES_DRATINI, TYPE_DRAGON, TYPE_NONE,  41, 64, 45, 50, 50,
      { MOVE_THUNDER_WAVE, MOVE_HYPER_BEAM, MOVE_BLIZZARD, MOVE_FIRE_BLAST } },
    { "Dragonair", SPECIES_DRAGONAIR, TYPE_DRAGON, TYPE_NONE,  61, 84, 65, 70, 70,
      { MOVE_THUNDER_WAVE, MOVE_HYPER_BEAM, MOVE_BLIZZARD, MOVE_FIRE_BLAST } },
};

// ============================================================================
// OU Team Pool
// Index arrays for building random OU teams
// ============================================================================

// Tier weights for team building (higher = more likely to be picked)
// Core 6 OU Pokemon that appear on most teams
static const SpeciesID OU_CORE[] = {
    SPECIES_TAUROS, SPECIES_CHANSEY, SPECIES_SNORLAX,
    SPECIES_ALAKAZAM, SPECIES_EXEGGUTOR, SPECIES_STARMIE,
};
#define OU_CORE_SIZE 6

// Good OU Pokemon that fill remaining slots
static const SpeciesID OU_GOOD[] = {
    SPECIES_GENGAR, SPECIES_JYNX, SPECIES_ZAPDOS, SPECIES_RHYDON,
    SPECIES_CLOYSTER, SPECIES_GOLEM, SPECIES_LAPRAS, SPECIES_SLOWBRO,
    SPECIES_JOLTEON, SPECIES_PERSIAN, SPECIES_HYPNO, SPECIES_ARTICUNO,
    SPECIES_DRAGONITE, SPECIES_MACHAMP,
};
#define OU_GOOD_SIZE 14

// All Gen 1 OU-legal species modeled by this simulator (non-Uber).
// This pool is used for comprehensive sampling and adaptive team search.
static const SpeciesID OU_LEGAL[] = {
    SPECIES_TAUROS, SPECIES_CHANSEY, SPECIES_SNORLAX, SPECIES_ALAKAZAM, SPECIES_EXEGGUTOR,
    SPECIES_STARMIE,SPECIES_GENGAR, SPECIES_JYNX, SPECIES_ZAPDOS, SPECIES_RHYDON, SPECIES_CLOYSTER,
    SPECIES_GOLEM,SPECIES_LAPRAS, SPECIES_SLOWBRO, SPECIES_JOLTEON, SPECIES_PERSIAN, SPECIES_HYPNO,
    SPECIES_ARTICUNO,SPECIES_DRAGONITE, SPECIES_MACHAMP, SPECIES_BULBASAUR, SPECIES_IVYSAUR,
    SPECIES_VENUSAUR,SPECIES_CHARMANDER, SPECIES_CHARMELEON, SPECIES_CHARIZARD, SPECIES_SQUIRTLE,
    SPECIES_WARTORTLE,SPECIES_BLASTOISE, SPECIES_CATERPIE, SPECIES_METAPOD, SPECIES_BUTTERFREE,
    SPECIES_WEEDLE,SPECIES_KAKUNA, SPECIES_BEEDRILL, SPECIES_PIDGEY, SPECIES_PIDGEOTTO,
    SPECIES_PIDGEOT,SPECIES_RATTATA, SPECIES_RATICATE, SPECIES_SPEAROW, SPECIES_FEAROW,
    SPECIES_EKANS,SPECIES_ARBOK, SPECIES_PIKACHU, SPECIES_RAICHU, SPECIES_SANDSHREW,
    SPECIES_SANDSLASH,SPECIES_NIDORAN_F, SPECIES_NIDORINA, SPECIES_NIDOQUEEN, SPECIES_NIDORAN_M,
    SPECIES_NIDORINO,SPECIES_NIDOKING, SPECIES_CLEFAIRY, SPECIES_CLEFABLE, SPECIES_VULPIX,
    SPECIES_NINETALES,SPECIES_JIGGLYPUFF, SPECIES_WIGGLYTUFF, SPECIES_ZUBAT, SPECIES_GOLBAT,
    SPECIES_ODDISH,SPECIES_GLOOM, SPECIES_VILEPLUME, SPECIES_PARAS, SPECIES_PARASECT,
    SPECIES_VENONAT,SPECIES_VENOMOTH, SPECIES_DIGLETT, SPECIES_DUGTRIO, SPECIES_MEOWTH,
    SPECIES_PSYDUCK,SPECIES_GOLDUCK, SPECIES_MANKEY, SPECIES_PRIMEAPE, SPECIES_GROWLITHE,
    SPECIES_ARCANINE,SPECIES_POLIWAG, SPECIES_POLIWHIRL, SPECIES_POLIWRATH, SPECIES_ABRA,
    SPECIES_KADABRA,SPECIES_MACHOP, SPECIES_MACHOKE, SPECIES_BELLSPROUT, SPECIES_WEEPINBELL,
    SPECIES_VICTREEBEL,SPECIES_TENTACOOL, SPECIES_TENTACRUEL, SPECIES_GEODUDE, SPECIES_GRAVELER,
    SPECIES_PONYTA,SPECIES_RAPIDASH, SPECIES_SLOWPOKE, SPECIES_MAGNEMITE, SPECIES_MAGNETON,
    SPECIES_FARFETCH_D,SPECIES_DODUO, SPECIES_DODRIO, SPECIES_SEEL, SPECIES_DEWGONG,
    SPECIES_GRIMER,SPECIES_MUK, SPECIES_SHELLDER, SPECIES_GASTLY, SPECIES_HAUNTER, SPECIES_ONIX,
    SPECIES_DROWZEE,SPECIES_KRABBY, SPECIES_KINGLER, SPECIES_VOLTORB, SPECIES_ELECTRODE,
    SPECIES_EXEGGCUTE,SPECIES_CUBONE, SPECIES_MAROWAK, SPECIES_HITMONLEE, SPECIES_HITMONCHAN,
    SPECIES_LICKITUNG,SPECIES_KOFFING, SPECIES_WEEZING, SPECIES_RHYHORN, SPECIES_TANGELA,
    SPECIES_KANGASKHAN,SPECIES_HORSEA, SPECIES_SEADRA, SPECIES_GOLDEEN, SPECIES_SEAKING,
    SPECIES_STARYU,SPECIES_MR_MIME, SPECIES_SCYTHER, SPECIES_ELECTABUZZ, SPECIES_MAGMAR,
    SPECIES_PINSIR,SPECIES_MAGIKARP, SPECIES_GYARADOS, SPECIES_DITTO, SPECIES_EEVEE,
    SPECIES_VAPOREON,SPECIES_FLAREON, SPECIES_PORYGON, SPECIES_OMANYTE, SPECIES_OMASTAR,
    SPECIES_KABUTO,SPECIES_KABUTOPS, SPECIES_AERODACTYL, SPECIES_MOLTRES, SPECIES_DRATINI,
    SPECIES_DRAGONAIR,
};
#define OU_LEGAL_SIZE 149

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    MoveID id;
    int pp;
    int max_pp;
} MoveSlot;

typedef struct {
    SpeciesID species;
    Type type1;
    Type type2;
    int max_hp;
    int hp;
    int base_atk;
    int base_def;
    int base_spc;
    int base_spe;
    MoveSlot moves[NUM_MOVE_SLOTS];
    StatusCondition status;
    int sleep_turns;      // turns remaining asleep
    int toxic_counter;    // toxic damage counter (increases each turn)
    int is_alive;
    // Clause source tracking: side index (0/1) that inflicted sleep/freeze, or -1
    int sleep_source_side;
    int freeze_source_side;
} Pokemon;

typedef struct {
    Pokemon team[NUM_POKEMON];
    int active_idx;       // index of currently active pokemon
    int alive_count;

    // Active pokemon volatile status (reset on switch)
    int atk_stage;        // -6 to +6
    int def_stage;
    int spc_stage;
    int spe_stage;
    int accuracy_stage;
    int evasion_stage;

    int is_confused;
    int confusion_turns;
    int is_seeded;        // leech seed
    int substitute_hp;
    int has_reflect;
    int has_light_screen;
    int is_recharging;    // hyper beam recharge
    int is_trapped;       // wrap/bind
    int trap_turns;
} Player;

typedef struct {
    Player players[2];
    int turn;
    int mode;             // 0 = normal, 1 = p1 must switch, 2 = p2 must switch, 3 = both
} Battle;

// PufferLib required Log struct (all floats)
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float p1_wins;
    float p2_wins;
    float draws;
    float n;
    // Per-species tracking: wins and games when species was on learner's team
    float species_wins[NUM_SPECIES + 1];  // indexed by SpeciesID (1-20)
    float species_games[NUM_SPECIES + 1]; // indexed by SpeciesID (1-20)
    // Team-builder diagnostics
    float team_builder_recent_winrate;
    float team_builder_pool_coverage;
} Log;

// Battle event types for the render log
enum {
    EVT_NONE = 0,
    EVT_MOVE_USED,       // player used move (data1=player_idx, data2=move_id, data3=species)
    EVT_MOVE_MISSED,     // move missed (data1=player_idx, data2=move_id, data3=species)
    EVT_IMMUNE,          // type immunity (data1=player_idx, data2=move_id, data3=def_species)
    EVT_DAMAGE,          // damage dealt (data1=player_idx, data2=amount, data3=species)
    EVT_CRITICAL,        // critical hit (data1=player_idx)
    EVT_FAINT,           // pokemon fainted (data1=player_idx, data3=species)
    EVT_SWITCH,          // switched pokemon (data1=player_idx, data3=new_species)
    EVT_STATUS,          // status inflicted (data1=player_idx, data2=status, data3=species)
    EVT_STAT_CHANGE,     // stat stage changed (data1=player_idx, data2=stat<<4|direction, data3=species)
    EVT_HEAL,            // healed HP (data1=player_idx, data2=amount, data3=species)
    EVT_SLEEP,           // couldn't move: asleep (data1=player_idx, data3=species)
    EVT_FROZEN,          // couldn't move: frozen (data1=player_idx, data3=species)
    EVT_PARALYZED,       // fully paralyzed (data1=player_idx, data3=species)
    EVT_CONFUSED_HIT,    // hit self in confusion (data1=player_idx, data3=species)
    EVT_RECHARGING,      // recharging (data1=player_idx, data3=species)
    EVT_SUBSTITUTE,      // substitute took damage (data1=player_idx, data2=amount)
    EVT_SUB_BROKE,       // substitute broke (data1=player_idx)
    EVT_SUPER_EFFECTIVE, // super effective (data1=player_idx)
    EVT_NOT_EFFECTIVE,   // not very effective (data1=player_idx)
    EVT_WAKE_UP,         // woke up (data1=player_idx, data3=species)
};

#define MAX_EVENTS 64

typedef struct {
    int type;
    int data1;      // typically player_idx (0 or 1)
    int data2;      // context-dependent
    int data3;      // typically species ID
} BattleEvent;

// PufferLib environment struct
typedef struct {
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;

    Battle battle;
    int num_agents;       // 1 = vs random bot, 2 = both RL-controlled
    int selfplay;         // 1 = selfplay mode (obs concatenated, 1 reward)
    int learner_side;     // 0 or 1: which player the learner controls
    int bot_mode;         // 0 = random, 1 = heuristic (1-ply), 2 = MCTS
    int mcts_iterations;  // number of rollouts for MCTS bot
    int mcts_depth;       // rollout depth for MCTS bot
    int auto_reset;       // 1 = reset immediately on done, 0 = wait for caller reset
    int tick;
    int last_p1_action;   // validated action from previous step
    int last_p2_action;   // validated action from previous step
    int last_result;      // previous step result: 1 p1 win, -1 p2 win, 0 draw/ongoing
    float p1_episode_return;
    float p2_episode_return;
    unsigned long long seed;
    unsigned long long rng_state;     // per-env xorshift64 state (for OpenMP thread safety)
    unsigned long long episode_count;
    Client* client;             // raylib render client (lazy init, NULL in headless)
    int mouse_action;           // -1=none, 0-9=clicked action, -2=restart
    int stale_turns;            // endless-battle stall counter
    unsigned long long last_progress_signature;
    int enforce_endless_clause; // 1=enabled, 0=disabled
    // Determinism hooks for parity tests
    // -1 = normal behavior, 0 = force false, 1 = force true
    int force_accuracy;
    int force_secondary;
    int team_builder_mode;  // 0=legacy random generator, 1=adaptive team search

    // Fixed team support: SPECIES_NONE in [0] = use random generation
    SpeciesID p1_fixed_team[NUM_POKEMON];
    SpeciesID p2_fixed_team[NUM_POKEMON];

    // Adaptive team search stats (learner and adversary species-level outcomes)
    float learner_species_wins[NUM_SPECIES + 1];
    float learner_species_games[NUM_SPECIES + 1];
    float opp_species_wins[NUM_SPECIES + 1];
    float opp_species_games[NUM_SPECIES + 1];
    float team_builder_recent_results[TEAM_BUILDER_RECENT_WINDOW];
    int team_builder_recent_count;
    int team_builder_recent_idx;
    float team_builder_recent_sum;
    int team_builder_coverage_cursor;
    SpeciesID team_builder_coverage_order[OU_LEGAL_SIZE];

    // Event buffer for battle log
    BattleEvent events[MAX_EVENTS];
    int event_count;
} PokeBattle;

// Global pointer for event logging from nested functions
static __thread PokeBattle* g_event_env = NULL;

static void evt_push(int type, int d1, int d2, int d3) {
    if (!g_event_env || g_event_env->event_count >= MAX_EVENTS) return;
    BattleEvent* e = &g_event_env->events[g_event_env->event_count++];
    e->type = type;
    e->data1 = d1;
    e->data2 = d2;
    e->data3 = d3;
}

// ============================================================================
// Random Number Generator (xorshift64)
// ============================================================================

static __thread unsigned long long pb_rng_state = 12345;

static inline unsigned long long pb_rng_next(void) {
    pb_rng_state ^= pb_rng_state << 13;
    pb_rng_state ^= pb_rng_state >> 7;
    pb_rng_state ^= pb_rng_state << 17;
    return pb_rng_state;
}

static inline int pb_rand_int(int max) {
    if (max <= 0) return 0;
    return (int)(pb_rng_next() % (unsigned long long)max);
}

static inline int pb_rand_range(int min, int max) {
    return min + pb_rand_int(max - min);
}

static inline float pb_rand_float01(void) {
    // Keep RNG deterministic across runs while allowing weighted sampling.
    return (float)((double)pb_rng_next() / (double)UINT64_MAX);
}

// Returns 1 with probability percent/100
static inline int pb_rand_chance(int percent) {
    return pb_rand_int(100) < percent;
}

static inline int forced_coin_flip(int override_value, int fallback_percent) {
    if (override_value == 0) return 0;
    if (override_value == 1) return 1;
    return pb_rand_chance(fallback_percent);
}

// ============================================================================
// Stat Calculation
// ============================================================================

static inline int calc_hp(int base) {
    return 2 * base + 203;
}

static inline int calc_stat(int base) {
    return 2 * base + 98;
}

// Stat stage multiplier table (for stages -6 to +6)
// Numerator / Denominator
static const int STAGE_NUMER[] = { 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8 };
static const int STAGE_DENOM[] = { 8, 7, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2 };

static inline int apply_stage(int base_stat, int stage) {
    int idx = stage + 6; // convert -6..+6 to 0..12
    if (idx < 0) idx = 0;
    if (idx > 12) idx = 12;
    return base_stat * STAGE_NUMER[idx] / STAGE_DENOM[idx];
}

// Accuracy/evasion stage multiplier
static const int ACC_NUMER[] = { 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9 };
static const int ACC_DENOM[] = { 9, 8, 7, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3 };

// ============================================================================
// Type Effectiveness
// ============================================================================

static int get_type_effectiveness(Type atk_type, Type def_type1, Type def_type2) {
    // Returns multiplier * 4 (to avoid floats):
    // 0 = immune, 1 = 0.25x, 2 = 0.5x, 4 = 1x, 8 = 2x, 16 = 4x
    int eff1 = TYPE_CHART[atk_type][def_type1]; // 0, 1, 2, or 3
    int mult = 1;

    // Convert: 0->0, 1->1 (0.5x), 2->2 (1x), 3->4 (2x)
    if (eff1 == 0) return 0;
    if (eff1 == 1) mult = 1;
    else if (eff1 == 2) mult = 2;
    else if (eff1 == 3) mult = 4;

    if (def_type2 != TYPE_NONE && def_type2 != def_type1) {
        int eff2 = TYPE_CHART[atk_type][def_type2];
        if (eff2 == 0) return 0;
        if (eff2 == 1) mult = mult * 1; // mult * 0.5 -> divide by 2 later
        else if (eff2 == 2) mult = mult * 2;
        else if (eff2 == 3) mult = mult * 4;

        // At this point mult encoding:
        // Single type: 1=0.5x, 2=1x, 4=2x
        // Dual type:   1=0.25x, 2=0.5x, 4=1x, 8=2x, 16=4x
        // We need to adjust: for dual type, the second application multiplied by 1,2,4
        // So we need to divide by 2 to normalize
        // Actually, let me redo this more carefully.
    }

    return mult;
}

// Simpler approach: returns float multiplier
static float type_effectiveness_float(Type atk_type, Type def_type1, Type def_type2) {
    float eff = 1.0f;

    int e1 = TYPE_CHART[atk_type][def_type1];
    if (e1 == 0) return 0.0f;
    if (e1 == 1) eff *= 0.5f;
    else if (e1 == 3) eff *= 2.0f;

    if (def_type2 != TYPE_NONE && def_type2 != def_type1) {
        int e2 = TYPE_CHART[atk_type][def_type2];
        if (e2 == 0) return 0.0f;
        if (e2 == 1) eff *= 0.5f;
        else if (e2 == 3) eff *= 2.0f;
    }

    return eff;
}

// ============================================================================
// Damage Calculation (Gen 1 formula)
// ============================================================================

static int calculate_damage(Pokemon* attacker, Pokemon* defender,
                           Player* atk_player, Player* def_player,
                           MoveSlot* move_slot, int is_critical) {
    const MoveData* mdata = &MOVE_DATA[move_slot->id];

    // Fixed damage moves
    if (mdata->effect == EFFECT_FIXED_DAMAGE) {
        return 100; // level 100
    }

    if (mdata->power == 0) return 0; // status move

    int power = mdata->power;
    int is_physical = TYPE_IS_PHYSICAL[mdata->type];

    // Get attack and defense stats
    int atk_stat, def_stat;
    if (is_physical) {
        atk_stat = calc_stat(attacker->base_atk);
        def_stat = calc_stat(defender->base_def);
    } else {
        atk_stat = calc_stat(attacker->base_spc);
        def_stat = calc_stat(defender->base_spc);
    }

    // Apply stat stages (not on critical hits)
    // Gen 1 Showdown: crits bypass stat stages AND burn's attack halving
    if (!is_critical) {
        if (is_physical) {
            atk_stat = apply_stage(atk_stat, atk_player->atk_stage);
            def_stat = apply_stage(def_stat, def_player->def_stage);
        } else {
            atk_stat = apply_stage(atk_stat, atk_player->spc_stage);
            def_stat = apply_stage(def_stat, def_player->spc_stage);
        }
        // Burn halves attack (physical only) - inside !is_critical per Showdown
        if (attacker->status == STATUS_BURN && is_physical) {
            atk_stat /= 2;
        }
    }

    // Reflect halves physical damage, Light Screen halves special damage
    if (is_physical && def_player->has_reflect && !is_critical) {
        def_stat *= 2;
    }
    if (!is_physical && def_player->has_light_screen && !is_critical) {
        def_stat *= 2;
    }

    // Explosion/Self-Destruct: halve defender's defense in the calculation
    if (mdata->effect == EFFECT_SELF_DESTRUCT) {
        def_stat /= 2;
        if (def_stat < 1) def_stat = 1;
    }

    // Ensure minimums
    if (atk_stat < 1) atk_stat = 1;
    if (def_stat < 1) def_stat = 1;

    // Gen 1 damage formula:
    // damage = ((2 * level * crit / 5 + 2) * power * A / D) / 50 + 2
    int level = 100;
    int crit_mult = is_critical ? 2 : 1;

    int damage = ((2 * level * crit_mult / 5 + 2) * power * atk_stat / def_stat) / 50 + 2;

    // STAB (Same Type Attack Bonus): 1.5x
    if (mdata->type == attacker->type1 || mdata->type == attacker->type2) {
        damage = damage * 3 / 2;
    }

    // Type effectiveness (Showdown Gen 1: per-type integer multiply with floor)
    int e1 = TYPE_CHART[mdata->type][defender->type1];
    if (e1 == 0) return 0;
    if (e1 == 3) { damage = damage * 20 / 10; }
    if (e1 == 1) { damage = damage * 5 / 10; }
    if (defender->type2 != TYPE_NONE && defender->type2 != defender->type1) {
        int e2 = TYPE_CHART[mdata->type][defender->type2];
        if (e2 == 0) return 0;
        if (e2 == 3) { damage = damage * 20 / 10; }
        if (e2 == 1) { damage = damage * 5 / 10; }
    }

    // Random factor: 217-255 / 255 (in Gen 1)
    int rand_factor = pb_rand_range(217, 256);
    damage = damage * rand_factor / 255;

    if (damage < 1) damage = 1;
    return damage;
}

// ============================================================================
// Critical Hit Check
// ============================================================================

static int check_critical(Pokemon* attacker, MoveSlot* move_slot) {
    const MoveData* mdata = &MOVE_DATA[move_slot->id];
    int base_speed = attacker->base_spe;

    // Gen 1: crit threshold = base_speed / 2 (or base_speed * 4 / 2 for high crit)
    int threshold;
    if (mdata->effect == EFFECT_HIGH_CRIT) {
        threshold = base_speed * 4; // effectively 8x normal rate
        if (threshold > 255) threshold = 255;
    } else {
        threshold = base_speed / 2;
        if (threshold > 255) threshold = 255;
    }

    // Gen 1 crit check: random 0-255, crit if < threshold
    int roll = pb_rand_int(256);
    return roll < threshold;
}

// ============================================================================
// Accuracy Check
// ============================================================================

static int check_accuracy(Player* atk_player, Player* def_player, MoveSlot* move_slot) {
    const MoveData* mdata = &MOVE_DATA[move_slot->id];

    // Moves with 0 accuracy always hit (self-targeting moves like Recover, Swords Dance)
    if (mdata->accuracy == 0) return 1;
    if (g_event_env && g_event_env->force_accuracy >= 0) {
        return g_event_env->force_accuracy ? 1 : 0;
    }

    int acc = mdata->accuracy;

    // Apply accuracy/evasion stages
    int acc_idx = atk_player->accuracy_stage + 6;
    int eva_idx = def_player->evasion_stage + 6;
    if (acc_idx < 0) acc_idx = 0;
    if (acc_idx > 12) acc_idx = 12;
    if (eva_idx < 0) eva_idx = 0;
    if (eva_idx > 12) eva_idx = 12;

    acc = acc * ACC_NUMER[acc_idx] / ACC_DENOM[acc_idx];
    acc = acc * ACC_DENOM[eva_idx] / ACC_NUMER[eva_idx];

    if (acc > 255) acc = 255;

    // Gen 1: 1/256 miss glitch - even 100% accuracy becomes 255/256
    int scaled = acc * 255 / 100;
    if (scaled > 255) scaled = 255;
    return pb_rand_int(256) < scaled;
}

// ============================================================================
// Pokemon Initialization
// ============================================================================

static void init_pokemon(Pokemon* poke, SpeciesID species) {
    if (species == SPECIES_NONE || species > NUM_SPECIES) {
        memset(poke, 0, sizeof(Pokemon));
        return;
    }

    const SpeciesData* sdata = &SPECIES_DATA[species];
    poke->species = species;
    poke->type1 = sdata->type1;
    poke->type2 = sdata->type2;
    poke->base_atk = sdata->base_atk;
    poke->base_def = sdata->base_def;
    poke->base_spc = sdata->base_spc;
    poke->base_spe = sdata->base_spe;
    poke->max_hp = calc_hp(sdata->base_hp);
    poke->hp = poke->max_hp;
    poke->status = STATUS_NONE;
    poke->sleep_turns = 0;
    poke->toxic_counter = 0;
    poke->is_alive = 1;
    poke->sleep_source_side = -1;
    poke->freeze_source_side = -1;

    for (int i = 0; i < NUM_MOVE_SLOTS; i++) {
        MoveID mid = sdata->moveset[i];
        poke->moves[i].id = mid;
        poke->moves[i].pp = MOVE_DATA[mid].max_pp;
        poke->moves[i].max_pp = MOVE_DATA[mid].max_pp;
    }
}

// ============================================================================
// Player Initialization
// ============================================================================

static void reset_volatile(Player* p) {
    p->atk_stage = 0;
    p->def_stage = 0;
    p->spc_stage = 0;
    p->spe_stage = 0;
    p->accuracy_stage = 0;
    p->evasion_stage = 0;
    p->is_confused = 0;
    p->confusion_turns = 0;
    p->is_seeded = 0;
    p->substitute_hp = 0;
    // Showdown Gen 1: Reflect/Light Screen are volatile, cleared on switch
    p->has_reflect = 0;
    p->has_light_screen = 0;
    p->is_recharging = 0;
    p->is_trapped = 0;
    p->trap_turns = 0;
}

static void init_player(Player* p, const SpeciesID team_species[NUM_POKEMON]) {
    memset(p, 0, sizeof(Player));
    p->alive_count = 0;

    for (int i = 0; i < NUM_POKEMON; i++) {
        init_pokemon(&p->team[i], team_species[i]);
        if (team_species[i] != SPECIES_NONE) {
            p->alive_count++;
        }
    }

    p->active_idx = 0;
    reset_volatile(p);
}

// ============================================================================
// Team Generation
// ============================================================================

static int species_in_list(const SpeciesID* list, int n, SpeciesID sp) {
    for (int i = 0; i < n; i++) {
        if (list[i] == sp) return 1;
    }
    return 0;
}

static float species_base_weight(SpeciesID sp) {
    // Prior team quality bias while keeping the full legal pool available.
    if (species_in_list(OU_CORE, OU_CORE_SIZE, sp)) return 3.0f;
    if (sp == SPECIES_MACHAMP) return 1.0f;
    if (species_in_list(OU_GOOD, OU_GOOD_SIZE, sp)) return 1.6f;
    return 1.0f;
}

static SpeciesID sample_species_weighted(const int used[NUM_SPECIES + 1],
                                         const float species_weights[NUM_SPECIES + 1]) {
    float total = 0.0f;
    for (int i = 0; i < OU_LEGAL_SIZE; i++) {
        SpeciesID sp = OU_LEGAL[i];
        if (!used[sp]) {
            float w = species_weights ? species_weights[sp] : 1.0f;
            if (w < 0.0001f) w = 0.0001f;
            total += w;
        }
    }

    if (total <= 0.0f) return SPECIES_NONE;

    float r = pb_rand_float01() * total;
    float acc = 0.0f;
    for (int i = 0; i < OU_LEGAL_SIZE; i++) {
        SpeciesID sp = OU_LEGAL[i];
        if (used[sp]) continue;
        float w = species_weights ? species_weights[sp] : 1.0f;
        if (w < 0.0001f) w = 0.0001f;
        acc += w;
        if (r <= acc) return sp;
    }

    // Numerical fallback.
    for (int i = 0; i < OU_LEGAL_SIZE; i++) {
        SpeciesID sp = OU_LEGAL[i];
        if (!used[sp]) return sp;
    }
    return SPECIES_NONE;
}

static void generate_ou_team(SpeciesID team[NUM_POKEMON]) {
    // Comprehensive sampling from all modeled Gen1 OU-legal species.
    int used[NUM_SPECIES + 1];
    float weights[NUM_SPECIES + 1];
    memset(used, 0, sizeof(used));
    memset(weights, 0, sizeof(weights));

    for (int i = 0; i < OU_LEGAL_SIZE; i++) {
        SpeciesID sp = OU_LEGAL[i];
        weights[sp] = species_base_weight(sp);
    }

    for (int i = 0; i < NUM_POKEMON; i++) {
        SpeciesID sp = sample_species_weighted(used, weights);
        if (sp == SPECIES_NONE) {
            for (int s = 1; s <= NUM_SPECIES; s++) {
                if (!used[s]) {
                    sp = (SpeciesID)s;
                    break;
                }
            }
        }
        used[sp] = 1;
        team[i] = sp;
    }
}

static void shuffle_team_builder_coverage(PokeBattle* env) {
    for (int i = 0; i < OU_LEGAL_SIZE; i++) {
        env->team_builder_coverage_order[i] = OU_LEGAL[i];
    }
    // Fisher-Yates shuffle
    for (int i = OU_LEGAL_SIZE - 1; i > 0; i--) {
        int j = pb_rand_int(i + 1);
        SpeciesID tmp = env->team_builder_coverage_order[i];
        env->team_builder_coverage_order[i] = env->team_builder_coverage_order[j];
        env->team_builder_coverage_order[j] = tmp;
    }
    env->team_builder_coverage_cursor = 0;
}

static void team_builder_reset_state(PokeBattle* env) {
    memset(env->learner_species_wins, 0, sizeof(env->learner_species_wins));
    memset(env->learner_species_games, 0, sizeof(env->learner_species_games));
    memset(env->opp_species_wins, 0, sizeof(env->opp_species_wins));
    memset(env->opp_species_games, 0, sizeof(env->opp_species_games));
    memset(env->team_builder_recent_results, 0, sizeof(env->team_builder_recent_results));
    env->team_builder_recent_count = 0;
    env->team_builder_recent_idx = 0;
    env->team_builder_recent_sum = 0.0f;
    shuffle_team_builder_coverage(env);
}

static float team_builder_recent_winrate(const PokeBattle* env) {
    if (env->team_builder_recent_count <= 0) return 0.5f;
    return env->team_builder_recent_sum / (float)env->team_builder_recent_count;
}

static int team_builder_explore_pct(const PokeBattle* env) {
    float wr = team_builder_recent_winrate(env);
    if (wr >= TEAM_BUILDER_TARGET_WINRATE) return TEAM_BUILDER_EXPLORE_MIN_PCT;
    if (wr <= 0.5f) return TEAM_BUILDER_EXPLORE_MAX_PCT;
    float denom = TEAM_BUILDER_TARGET_WINRATE - 0.5f;
    if (denom < 0.0001f) denom = 0.0001f;
    float t = (TEAM_BUILDER_TARGET_WINRATE - wr) / denom;
    int span = TEAM_BUILDER_EXPLORE_MAX_PCT - TEAM_BUILDER_EXPLORE_MIN_PCT;
    int pct = TEAM_BUILDER_EXPLORE_MIN_PCT + (int)roundf(t * (float)span);
    if (pct < TEAM_BUILDER_EXPLORE_MIN_PCT) pct = TEAM_BUILDER_EXPLORE_MIN_PCT;
    if (pct > TEAM_BUILDER_EXPLORE_MAX_PCT) pct = TEAM_BUILDER_EXPLORE_MAX_PCT;
    return pct;
}

static int team_builder_unique_seen(const float species_games[NUM_SPECIES + 1]) {
    int count = 0;
    for (int i = 0; i < OU_LEGAL_SIZE; i++) {
        SpeciesID sp = OU_LEGAL[i];
        if (species_games[sp] > 0.0f) count++;
    }
    return count;
}

static float team_builder_species_score(const float wins[NUM_SPECIES + 1],
                                        const float games[NUM_SPECIES + 1],
                                        SpeciesID sp,
                                        float total_games) {
    float prior = 1.0f;
    float mean = (wins[sp] + prior) / (games[sp] + 2.0f * prior);
    float bonus = TEAM_BUILDER_UCB_C * sqrtf(logf(total_games + 2.0f) / (games[sp] + 1.0f));
    return mean + bonus + 0.03f * species_base_weight(sp);
}

static SpeciesID team_builder_pick_species(PokeBattle* env,
                                           const int used[NUM_SPECIES + 1],
                                           int for_learner,
                                           int explore_pct) {
    if (pb_rand_chance(explore_pct)) {
        SpeciesID sample = sample_species_weighted(used, NULL);
        if (sample != SPECIES_NONE) return sample;
    }

    const float* wins = for_learner ? env->learner_species_wins : env->opp_species_wins;
    const float* games = for_learner ? env->learner_species_games : env->opp_species_games;
    float total_games = 0.0f;
    for (int i = 0; i < OU_LEGAL_SIZE; i++) {
        total_games += games[OU_LEGAL[i]];
    }

    SpeciesID best = SPECIES_NONE;
    float best_score = -1.0e30f;
    for (int i = 0; i < OU_LEGAL_SIZE; i++) {
        SpeciesID sp = OU_LEGAL[i];
        if (used[sp]) continue;
        float score = team_builder_species_score(wins, games, sp, total_games);
        if (score > best_score || (fabsf(score - best_score) < 1e-6f && pb_rand_int(2) == 0)) {
            best_score = score;
            best = sp;
        }
    }

    if (best != SPECIES_NONE) return best;
    return sample_species_weighted(used, NULL);
}

static SpeciesID team_builder_next_anchor(PokeBattle* env) {
    if (env->team_builder_coverage_cursor >= OU_LEGAL_SIZE) {
        shuffle_team_builder_coverage(env);
    }
    SpeciesID sp = env->team_builder_coverage_order[env->team_builder_coverage_cursor];
    env->team_builder_coverage_cursor++;
    return sp;
}

static void generate_adaptive_team(PokeBattle* env, SpeciesID team[NUM_POKEMON], int for_learner) {
    int used[NUM_SPECIES + 1];
    memset(used, 0, sizeof(used));
    int count = 0;

    if (for_learner) {
        // Force full pool coverage so every legal species receives training signal.
        SpeciesID anchor = team_builder_next_anchor(env);
        team[count++] = anchor;
        used[anchor] = 1;
    }

    int explore_pct = team_builder_explore_pct(env);
    if (!for_learner) {
        // Adversary is slightly more exploitative to keep pressure high.
        explore_pct /= 2;
        if (explore_pct < TEAM_BUILDER_EXPLORE_MIN_PCT) {
            explore_pct = TEAM_BUILDER_EXPLORE_MIN_PCT;
        }
    }

    while (count < NUM_POKEMON) {
        SpeciesID sp = team_builder_pick_species(env, used, for_learner, explore_pct);
        if (sp == SPECIES_NONE) {
            for (int i = 0; i < OU_LEGAL_SIZE; i++) {
                SpeciesID fallback = OU_LEGAL[i];
                if (!used[fallback]) {
                    sp = fallback;
                    break;
                }
            }
        }
        if (sp == SPECIES_NONE) break;
        used[sp] = 1;
        team[count++] = sp;
    }

    while (count < NUM_POKEMON) {
        team[count++] = OU_LEGAL[pb_rand_int(OU_LEGAL_SIZE)];
    }
}

static void team_builder_record_result(PokeBattle* env, float outcome) {
    if (env->team_builder_recent_count < TEAM_BUILDER_RECENT_WINDOW) {
        env->team_builder_recent_results[env->team_builder_recent_idx] = outcome;
        env->team_builder_recent_sum += outcome;
        env->team_builder_recent_idx = (env->team_builder_recent_idx + 1) % TEAM_BUILDER_RECENT_WINDOW;
        env->team_builder_recent_count++;
        return;
    }

    float old = env->team_builder_recent_results[env->team_builder_recent_idx];
    env->team_builder_recent_results[env->team_builder_recent_idx] = outcome;
    env->team_builder_recent_sum += outcome - old;
    env->team_builder_recent_idx = (env->team_builder_recent_idx + 1) % TEAM_BUILDER_RECENT_WINDOW;
}

// ============================================================================
// Switch Pokemon
// ============================================================================

static int can_switch(Player* p, int target_idx) {
    if (target_idx < 0 || target_idx >= NUM_POKEMON) return 0;
    if (target_idx == p->active_idx) return 0;
    if (!p->team[target_idx].is_alive) return 0;
    return 1;
}

static void do_switch(Player* p, int target_idx) {
    if (!can_switch(p, target_idx)) return;

    // Gen 1: toxic counter resets when switching out
    Pokemon* old_active = &p->team[p->active_idx];
    if (old_active->status == STATUS_TOXIC) {
        old_active->toxic_counter = 0;
    }

    // Clear volatile status (stat stages, confusion, etc.)
    reset_volatile(p);

    // Switch active
    p->active_idx = target_idx;
}

// ============================================================================
// Get Active Pokemon Helpers
// ============================================================================

static inline Pokemon* active_pokemon(Player* p) {
    return &p->team[p->active_idx];
}

static inline int get_effective_speed(Player* p) {
    Pokemon* poke = active_pokemon(p);
    int spe = calc_stat(poke->base_spe);
    spe = apply_stage(spe, p->spe_stage);

    // Paralysis quarters speed
    if (poke->status == STATUS_PARALYSIS) {
        spe /= 4;
    }

    if (spe < 1) spe = 1;
    return spe;
}

// ============================================================================
// Action Validation
// ============================================================================

static int can_use_move(Player* p, int move_idx) {
    if (move_idx < 0 || move_idx >= NUM_MOVE_SLOTS) return 0;
    Pokemon* poke = active_pokemon(p);
    if (poke->moves[move_idx].id == MOVE_NONE) return 0;
    if (poke->moves[move_idx].pp <= 0) return 0;
    return 1;
}

static void get_action_mask(Player* p, int mode, int player_idx, int mask[NUM_ACTIONS]) {
    memset(mask, 0, NUM_ACTIONS * sizeof(int));

    if (mode == 0) {
        // Normal turn: can use moves or switch
        // Recharging: can only wait (all actions valid but will skip turn)
        if (p->is_recharging) {
            // Must wait - set action 0 as valid (will be treated as recharge)
            mask[0] = 1;
            return;
        }

        // Check moves
        int any_move = 0;
        for (int i = 0; i < NUM_MOVE_SLOTS; i++) {
            if (can_use_move(p, i)) {
                mask[i] = 1;
                any_move = 1;
            }
        }

        // Check switches (can't switch if trapped)
        if (!p->is_trapped) {
            for (int i = 0; i < NUM_POKEMON; i++) {
                if (can_switch(p, i)) {
                    mask[4 + i] = 1;
                }
            }
        }

        // If no moves available (all PP depleted), must switch or struggle
        if (!any_move) {
            // Struggle - allow action 0 (will be treated as struggle)
            mask[0] = 1;
        }
    } else {
        // Force switch mode: must switch to a living pokemon
        int needs_switch = 0;
        if (player_idx == 0 && (mode == 1 || mode == 3)) needs_switch = 1;
        if (player_idx == 1 && (mode == 2 || mode == 3)) needs_switch = 1;

        if (needs_switch) {
            for (int i = 0; i < NUM_POKEMON; i++) {
                if (can_switch(p, i)) {
                    mask[4 + i] = 1;
                }
            }
        } else {
            // This player doesn't need to act in this mode, pass
            mask[0] = 1;
        }
    }
}

static int side_has_opponent_inflicted_status(Player* side, StatusCondition status, int source_side) {
    if (!side || source_side < 0) return 0;
    for (int i = 0; i < NUM_POKEMON; i++) {
        Pokemon* mon = &side->team[i];
        if (mon->hp <= 0 || mon->status != status) continue;
        if (status == STATUS_SLEEP && mon->sleep_source_side == source_side) return 1;
        if (status == STATUS_FREEZE && mon->freeze_source_side == source_side) return 1;
    }
    return 0;
}

// ============================================================================
// Apply Status Effect
// ============================================================================

static int try_inflict_status(Player* target_side, Pokemon* target,
                             StatusCondition status, int source_side) {
    if (target->status != STATUS_NONE) return 0; // already has status

    if (status == STATUS_SLEEP &&
        side_has_opponent_inflicted_status(target_side, STATUS_SLEEP, source_side)) {
        return 0;
    }
    if (status == STATUS_FREEZE &&
        side_has_opponent_inflicted_status(target_side, STATUS_FREEZE, source_side)) {
        return 0;
    }

    // Type immunities
    if (status == STATUS_PARALYSIS) {
        // Electric types can't be paralyzed in Gen 1... actually they can.
        // In Gen 1, there are no type-based status immunities except:
        // - Fire types can't be burned (actually not in Gen 1!)
        // Actually in Gen 1, there ARE NO type-based immunities to status.
        // The only immunity is via type chart (e.g., Electric ground type is immune to Thunder Wave
        // because Thunder Wave is Electric type and can't hit Ground types).
    }

    if (status == STATUS_BURN) {
        // Fire types can't be burned... wait, in Gen 1 they CAN be burned.
        // Actually, Gen 1 has no type-based status immunities.
        // The immunity only comes from the type chart for damaging moves.
    }

    if (status == STATUS_FREEZE) {
        // Ice types can't be frozen
        if (target->type1 == TYPE_ICE || target->type2 == TYPE_ICE) return 0;
    }

    if (status == STATUS_POISON || status == STATUS_TOXIC) {
        // Poison types can't be poisoned
        if (target->type1 == TYPE_POISON || target->type2 == TYPE_POISON) return 0;
    }

    target->status = status;
    if (status == STATUS_SLEEP) {
        target->sleep_turns = pb_rand_range(1, 8); // Gen 1: 1-7 turns
        target->sleep_source_side = source_side;
    } else {
        target->sleep_source_side = -1;
        target->sleep_turns = 0;
    }
    if (status == STATUS_FREEZE) {
        target->freeze_source_side = source_side;
    } else {
        target->freeze_source_side = -1;
    }
    if (status == STATUS_TOXIC) {
        target->toxic_counter = 1;
    }
    return 1;
}

// ============================================================================
// Apply Move Effect
// ============================================================================

static void apply_move_effect(Player* atk_player, Player* def_player,
                             MoveSlot* move_slot, int damage_dealt, int atk_pidx) {
    const MoveData* mdata = &MOVE_DATA[move_slot->id];
    Pokemon* attacker = active_pokemon(atk_player);
    Pokemon* defender = active_pokemon(def_player);
    int def_pidx = 1 - atk_pidx;
    int secondary_override = g_event_env ? g_event_env->force_secondary : -1;

    switch (mdata->effect) {
    case EFFECT_PARALYZE_CHANCE:
        // Showdown Gen 1: secondary status blocked if move type matches target type
        if (forced_coin_flip(secondary_override, mdata->effect_chance)) {
            if (defender->type1 != mdata->type && defender->type2 != mdata->type) {
                if (try_inflict_status(def_player, defender, STATUS_PARALYSIS, atk_pidx))
                    evt_push(EVT_STATUS, def_pidx, STATUS_PARALYSIS, defender->species);
            }
        }
        break;

    case EFFECT_BURN_CHANCE:
        if (forced_coin_flip(secondary_override, mdata->effect_chance)) {
            if (defender->type1 != mdata->type && defender->type2 != mdata->type) {
                if (try_inflict_status(def_player, defender, STATUS_BURN, atk_pidx))
                    evt_push(EVT_STATUS, def_pidx, STATUS_BURN, defender->species);
            }
        }
        break;

    case EFFECT_FREEZE_CHANCE:
        if (forced_coin_flip(secondary_override, mdata->effect_chance)) {
            if (defender->type1 != mdata->type && defender->type2 != mdata->type) {
                if (try_inflict_status(def_player, defender, STATUS_FREEZE, atk_pidx))
                    evt_push(EVT_STATUS, def_pidx, STATUS_FREEZE, defender->species);
            }
        }
        break;

    case EFFECT_LOWER_SPECIAL:
        if (forced_coin_flip(secondary_override, mdata->effect_chance)) {
            if (def_player->spc_stage > -MAX_STAT_STAGE) {
                def_player->spc_stage--;
                evt_push(EVT_STAT_CHANGE, def_pidx, (2 << 4) | 0, defender->species);
            }
        }
        break;

    case EFFECT_LOWER_SPEED:
        if (forced_coin_flip(secondary_override, mdata->effect_chance)) {
            if (def_player->spe_stage > -MAX_STAT_STAGE) {
                def_player->spe_stage--;
                evt_push(EVT_STAT_CHANGE, def_pidx, (3 << 4) | 0, defender->species);
            }
        }
        break;

    case EFFECT_LOWER_DEFENSE:
        if (forced_coin_flip(secondary_override, mdata->effect_chance)) {
            if (def_player->def_stage > -MAX_STAT_STAGE) {
                def_player->def_stage--;
                evt_push(EVT_STAT_CHANGE, def_pidx, (1 << 4) | 0, defender->species);
            }
        }
        break;

    case EFFECT_RECOIL:
        if (damage_dealt > 0) {
            int recoil = damage_dealt * mdata->effect_chance / 100;
            if (recoil < 1) recoil = 1;
            attacker->hp -= recoil;
            evt_push(EVT_DAMAGE, atk_pidx, recoil, attacker->species);
            if (attacker->hp <= 0) {
                attacker->hp = 0;
                attacker->is_alive = 0;
                atk_player->alive_count--;
                evt_push(EVT_FAINT, atk_pidx, 0, attacker->species);
            }
        }
        break;

    case EFFECT_SELF_DESTRUCT:
        // User faints (defense halving is handled in damage calc)
        attacker->hp = 0;
        attacker->is_alive = 0;
        atk_player->alive_count--;
        evt_push(EVT_FAINT, atk_pidx, 0, attacker->species);
        break;

    case EFFECT_HYPER_BEAM:
        // Must recharge next turn (unless target fainted - Gen 1 quirk)
        if (defender->is_alive) {
            atk_player->is_recharging = 1;
        }
        break;

    case EFFECT_DRAIN:
        if (damage_dealt > 0) {
            int heal = damage_dealt / 2;
            if (heal < 1) heal = 1;
            attacker->hp += heal;
            if (attacker->hp > attacker->max_hp) {
                attacker->hp = attacker->max_hp;
            }
            evt_push(EVT_HEAL, atk_pidx, heal, attacker->species);
        }
        break;

    case EFFECT_THUNDER_WAVE: {
        // Check type immunity (Thunder Wave is Electric type)
        Type tw_type = mdata->type;
        float eff = type_effectiveness_float(tw_type, defender->type1, defender->type2);
        if (eff > 0.0f) {
            if (try_inflict_status(def_player, defender, STATUS_PARALYSIS, atk_pidx))
                evt_push(EVT_STATUS, def_pidx, STATUS_PARALYSIS, defender->species);
        } else {
            evt_push(EVT_IMMUNE, atk_pidx, move_slot->id, defender->species);
        }
        break;
    }

    case EFFECT_SLEEP:
        if (try_inflict_status(def_player, defender, STATUS_SLEEP, atk_pidx))
            evt_push(EVT_STATUS, def_pidx, STATUS_SLEEP, defender->species);
        break;

    case EFFECT_RECOVER: {
        int prev_hp = attacker->hp;
        attacker->hp += attacker->max_hp / 2;
        if (attacker->hp > attacker->max_hp) {
            attacker->hp = attacker->max_hp;
        }
        int healed = attacker->hp - prev_hp;
        if (healed > 0) evt_push(EVT_HEAL, atk_pidx, healed, attacker->species);
        break;
    }

    case EFFECT_REST:
        if (attacker->hp < attacker->max_hp) {
            int prev_hp = attacker->hp;
            attacker->hp = attacker->max_hp;
            attacker->status = STATUS_SLEEP;
            attacker->sleep_turns = 2; // Rest always sleeps exactly 2 turns
            attacker->sleep_source_side = atk_pidx;
            attacker->freeze_source_side = -1;
            evt_push(EVT_HEAL, atk_pidx, attacker->hp - prev_hp, attacker->species);
            evt_push(EVT_STATUS, atk_pidx, STATUS_SLEEP, attacker->species);
        }
        break;

    case EFFECT_BOOST_ATK_2:
        if (atk_player->atk_stage < MAX_STAT_STAGE) {
            atk_player->atk_stage += 2;
            if (atk_player->atk_stage > MAX_STAT_STAGE) atk_player->atk_stage = MAX_STAT_STAGE;
            evt_push(EVT_STAT_CHANGE, atk_pidx, (0 << 4) | 1, attacker->species);
        }
        break;

    case EFFECT_BOOST_SPC_2:
        if (atk_player->spc_stage < MAX_STAT_STAGE) {
            atk_player->spc_stage += 2;
            if (atk_player->spc_stage > MAX_STAT_STAGE) atk_player->spc_stage = MAX_STAT_STAGE;
            evt_push(EVT_STAT_CHANGE, atk_pidx, (2 << 4) | 1, attacker->species);
        }
        break;

    case EFFECT_BOOST_SPE_2:
        if (atk_player->spe_stage < MAX_STAT_STAGE) {
            atk_player->spe_stage += 2;
            if (atk_player->spe_stage > MAX_STAT_STAGE) atk_player->spe_stage = MAX_STAT_STAGE;
            evt_push(EVT_STAT_CHANGE, atk_pidx, (3 << 4) | 1, attacker->species);
        }
        break;

    case EFFECT_SUBSTITUTE:
        if (atk_player->substitute_hp == 0) {
            int sub_cost = attacker->max_hp / 4;
            if (attacker->hp > sub_cost) {
                attacker->hp -= sub_cost;
                atk_player->substitute_hp = sub_cost;
            }
        }
        break;

    case EFFECT_TOXIC: {
        Type t_type = mdata->type;
        float eff = type_effectiveness_float(t_type, defender->type1, defender->type2);
        if (eff > 0.0f) {
            if (try_inflict_status(def_player, defender, STATUS_TOXIC, atk_pidx))
                evt_push(EVT_STATUS, def_pidx, STATUS_TOXIC, defender->species);
        } else {
            evt_push(EVT_IMMUNE, atk_pidx, move_slot->id, defender->species);
        }
        break;
    }

    case EFFECT_REFLECT:
        atk_player->has_reflect = 1;
        break;

    case EFFECT_LIGHT_SCREEN:
        atk_player->has_light_screen = 1;
        break;

    case EFFECT_CONFUSE:
        if (!def_player->is_confused) {
            def_player->is_confused = 1;
            def_player->confusion_turns = pb_rand_range(1, 5); // 1-4 turns
        }
        break;

    case EFFECT_LEECH_SEED:
        if (defender->type1 != TYPE_GRASS && defender->type2 != TYPE_GRASS) {
            def_player->is_seeded = 1;
        }
        break;

    case EFFECT_TRANSFORM:
        // Simplified Transform: copy species identity, typing, offensive/defensive
        // base stats, and current moveset with Gen 1 Transform PP behavior (5 PP each).
        // HP remains unchanged by design.
        attacker->species = defender->species;
        attacker->type1 = defender->type1;
        attacker->type2 = defender->type2;
        attacker->base_atk = defender->base_atk;
        attacker->base_def = defender->base_def;
        attacker->base_spc = defender->base_spc;
        attacker->base_spe = defender->base_spe;
        for (int i = 0; i < NUM_MOVE_SLOTS; i++) {
            MoveID copied = defender->moves[i].id;
            attacker->moves[i].id = copied;
            attacker->moves[i].max_pp = 5;
            attacker->moves[i].pp = 5;
        }
        // Copy current combat stages for closer Showdown behavior.
        atk_player->atk_stage = def_player->atk_stage;
        atk_player->def_stage = def_player->def_stage;
        atk_player->spc_stage = def_player->spc_stage;
        atk_player->spe_stage = def_player->spe_stage;
        break;

    case EFFECT_TRAPPING:
        if (damage_dealt > 0 && defender->is_alive) {
            def_player->is_trapped = 1;
            def_player->trap_turns = pb_rand_range(2, 6); // 2-5 turns
        }
        break;

    case EFFECT_MULTI_HIT: {
        // Already dealt one hit's worth of damage. Deal 1-4 more.
        int hits = pb_rand_range(1, 5); // 1-4 more hits (total 2-5)
        for (int h = 0; h < hits; h++) {
            if (!defender->is_alive) break;
            // Recalculate damage for each hit
            int is_crit = check_critical(attacker, move_slot);
            int dmg = calculate_damage(attacker, defender, atk_player, def_player,
                                      move_slot, is_crit);
            if (def_player->substitute_hp > 0) {
                def_player->substitute_hp -= dmg;
                if (def_player->substitute_hp <= 0) {
                    def_player->substitute_hp = 0;
                }
            } else {
                defender->hp -= dmg;
                if (defender->hp <= 0) {
                    defender->hp = 0;
                    defender->is_alive = 0;
                    def_player->alive_count--;
                }
            }
        }
        break;
    }

    default:
        break;
    }
}

// ============================================================================
// Execute Move
// ============================================================================

static int execute_move(Player* atk_player, Player* def_player, int move_idx, int atk_pidx) {
    Pokemon* attacker = active_pokemon(atk_player);
    Pokemon* defender = active_pokemon(def_player);
    int def_pidx = 1 - atk_pidx;

    if (!attacker->is_alive) return 0;

    // Check if recharging (Hyper Beam)
    if (atk_player->is_recharging) {
        evt_push(EVT_RECHARGING, atk_pidx, 0, attacker->species);
        atk_player->is_recharging = 0;
        return 0;
    }

    // Check sleep
    if (attacker->status == STATUS_SLEEP) {
        attacker->sleep_turns--;
        if (attacker->sleep_turns <= 0) {
            attacker->status = STATUS_NONE;
            attacker->sleep_turns = 0;
            attacker->sleep_source_side = -1;
            evt_push(EVT_WAKE_UP, atk_pidx, 0, attacker->species);
            return 0;
        }
        evt_push(EVT_SLEEP, atk_pidx, attacker->sleep_turns, attacker->species);
        return 0;
    }

    // Check freeze (Gen 1 Showdown: frozen permanently, thawed only by opponent's Fire move)
    if (attacker->status == STATUS_FREEZE) {
        evt_push(EVT_FROZEN, atk_pidx, 0, attacker->species);
        return 0;
    }

    // Check paralysis (25% chance of full paralysis)
    if (attacker->status == STATUS_PARALYSIS) {
        if (pb_rand_int(256) < 63) {
            evt_push(EVT_PARALYZED, atk_pidx, 0, attacker->species);
            return 0;
        }
    }

    // Check confusion
    if (atk_player->is_confused) {
        atk_player->confusion_turns--;
        if (atk_player->confusion_turns <= 0) {
            atk_player->is_confused = 0;
        } else if (pb_rand_chance(50)) {
            evt_push(EVT_CONFUSED_HIT, atk_pidx, 0, attacker->species);
            int conf_atk = apply_stage(calc_stat(attacker->base_atk), atk_player->atk_stage);
            int conf_def = apply_stage(calc_stat(attacker->base_def), atk_player->def_stage);
            int self_dmg = ((2 * 100 / 5 + 2) * 40 * conf_atk / conf_def) / 50 + 2;
            attacker->hp -= self_dmg;
            if (attacker->hp <= 0) {
                attacker->hp = 0;
                attacker->is_alive = 0;
                atk_player->alive_count--;
                evt_push(EVT_FAINT, atk_pidx, 0, attacker->species);
            }
            return 0;
        }
    }

    // Determine the move to use
    MoveSlot* move_slot;
    int is_struggle = 0;

    if (move_idx < 0 || move_idx >= NUM_MOVE_SLOTS ||
        attacker->moves[move_idx].id == MOVE_NONE ||
        attacker->moves[move_idx].pp <= 0) {
        // Struggle
        is_struggle = 1;
        // Create a temporary struggle move
        static MoveSlot struggle = { MOVE_NONE, 1, 1 };
        move_slot = &struggle;
    } else {
        move_slot = &attacker->moves[move_idx];
        move_slot->pp--;
    }

    if (is_struggle) {
        // Struggle: 50 power Normal type, typeless, 1/2 recoil
        // Just deal some damage
        int dmg = ((2 * 100 / 5 + 2) * 50 * calc_stat(attacker->base_atk) /
                  calc_stat(defender->base_def)) / 50 + 2;
        int rand_factor = pb_rand_range(217, 256);
        dmg = dmg * rand_factor / 255;
        if (dmg < 1) dmg = 1;

        if (def_player->substitute_hp > 0) {
            def_player->substitute_hp -= dmg;
            if (def_player->substitute_hp <= 0) def_player->substitute_hp = 0;
        } else {
            defender->hp -= dmg;
            if (defender->hp <= 0) {
                defender->hp = 0;
                defender->is_alive = 0;
                def_player->alive_count--;
            }
        }

        // Recoil
        int recoil = dmg / 2;
        if (recoil < 1) recoil = 1;
        attacker->hp -= recoil;
        if (attacker->hp <= 0) {
            attacker->hp = 0;
            attacker->is_alive = 0;
            atk_player->alive_count--;
        }
        return 1;
    }

    const MoveData* mdata = &MOVE_DATA[move_slot->id];

    evt_push(EVT_MOVE_USED, atk_pidx, move_slot->id, attacker->species);

    // Accuracy check
    if (mdata->accuracy > 0 && !check_accuracy(atk_player, def_player, move_slot)) {
        evt_push(EVT_MOVE_MISSED, atk_pidx, move_slot->id, attacker->species);
        return 0;
    }

    // For status moves (power == 0), just apply effect
    if (mdata->power == 0 && mdata->effect != EFFECT_FIXED_DAMAGE) {
        apply_move_effect(atk_player, def_player, move_slot, 0, atk_pidx);
        return 1;
    }

    // Damaging move
    int is_critical = check_critical(attacker, move_slot);
    int damage = calculate_damage(attacker, defender, atk_player, def_player,
                                 move_slot, is_critical);

    if (damage == 0) {
        evt_push(EVT_IMMUNE, atk_pidx, move_slot->id, defender->species);
        return 1;
    }

    if (is_critical) {
        evt_push(EVT_CRITICAL, atk_pidx, 0, 0);
    }

    // Type effectiveness hints
    float eff = type_effectiveness_float(mdata->type, defender->type1, defender->type2);
    if (eff > 1.5f) evt_push(EVT_SUPER_EFFECTIVE, atk_pidx, 0, 0);
    else if (eff > 0.0f && eff < 0.9f) evt_push(EVT_NOT_EFFECTIVE, atk_pidx, 0, 0);

    // Apply damage to substitute or directly
    if (def_player->substitute_hp > 0) {
        evt_push(EVT_SUBSTITUTE, def_pidx, damage, 0);
        def_player->substitute_hp -= damage;
        if (def_player->substitute_hp <= 0) {
            def_player->substitute_hp = 0;
            evt_push(EVT_SUB_BROKE, def_pidx, 0, 0);
        }
    } else {
        int prev_hp = defender->hp;
        defender->hp -= damage;
        if (defender->hp <= 0) {
            defender->hp = 0;
            defender->is_alive = 0;
            def_player->alive_count--;
        }
        evt_push(EVT_DAMAGE, def_pidx, prev_hp - defender->hp, defender->species);
        if (!defender->is_alive) {
            evt_push(EVT_FAINT, def_pidx, 0, defender->species);
        }

        // Gen 1 Showdown: Fire-type damaging move thaws frozen target
        if (defender->status == STATUS_FREEZE && mdata->type == TYPE_FIRE && damage > 0) {
            defender->status = STATUS_NONE;
            defender->freeze_source_side = -1;
        }

        // Apply secondary effect
        apply_move_effect(atk_player, def_player, move_slot, damage, atk_pidx);
    }

    return 1;
}

// ============================================================================
// End-of-turn Effects
// ============================================================================

static void apply_end_of_turn(Player* p, Player* opponent, int pidx) {
    Pokemon* poke = active_pokemon(p);
    if (!poke->is_alive) return;
    int opp_pidx = 1 - pidx;

    // Burn damage: 1/16 max HP
    if (poke->status == STATUS_BURN) {
        int burn_dmg = poke->max_hp / 16;
        if (burn_dmg < 1) burn_dmg = 1;
        poke->hp -= burn_dmg;
        evt_push(EVT_DAMAGE, pidx, burn_dmg, poke->species);
        if (poke->hp <= 0) {
            poke->hp = 0;
            poke->is_alive = 0;
            p->alive_count--;
            evt_push(EVT_FAINT, pidx, 0, poke->species);
            return;
        }
    }

    // Poison damage: 1/16 max HP
    if (poke->status == STATUS_POISON) {
        int poison_dmg = poke->max_hp / 16;
        if (poison_dmg < 1) poison_dmg = 1;
        poke->hp -= poison_dmg;
        evt_push(EVT_DAMAGE, pidx, poison_dmg, poke->species);
        if (poke->hp <= 0) {
            poke->hp = 0;
            poke->is_alive = 0;
            p->alive_count--;
            evt_push(EVT_FAINT, pidx, 0, poke->species);
            return;
        }
    }

    // Toxic damage: increases each turn
    if (poke->status == STATUS_TOXIC) {
        int toxic_dmg = poke->max_hp * poke->toxic_counter / 16;
        if (toxic_dmg < 1) toxic_dmg = 1;
        poke->hp -= toxic_dmg;
        poke->toxic_counter++;
        evt_push(EVT_DAMAGE, pidx, toxic_dmg, poke->species);
        if (poke->hp <= 0) {
            poke->hp = 0;
            poke->is_alive = 0;
            p->alive_count--;
            evt_push(EVT_FAINT, pidx, 0, poke->species);
            return;
        }
    }

    // Leech Seed: drain 1/16 max HP, give to opponent
    if (p->is_seeded && poke->is_alive) {
        Pokemon* opp_poke = active_pokemon(opponent);
        if (opp_poke->is_alive) {
            int seed_dmg = poke->max_hp / 16;
            if (seed_dmg < 1) seed_dmg = 1;
            poke->hp -= seed_dmg;
            opp_poke->hp += seed_dmg;
            if (opp_poke->hp > opp_poke->max_hp) {
                opp_poke->hp = opp_poke->max_hp;
            }
            evt_push(EVT_DAMAGE, pidx, seed_dmg, poke->species);
            evt_push(EVT_HEAL, opp_pidx, seed_dmg, opp_poke->species);
            if (poke->hp <= 0) {
                poke->hp = 0;
                poke->is_alive = 0;
                p->alive_count--;
                evt_push(EVT_FAINT, pidx, 0, poke->species);
                return;
            }
        }
    }

    // Trapping moves: deal damage each turn
    if (p->is_trapped) {
        p->trap_turns--;
        if (p->trap_turns <= 0) {
            p->is_trapped = 0;
        }
    }
}

// ============================================================================
// Observation Packing
// ============================================================================

static void pack_player_obs(float* obs, Player* my, Player* opp,
                           int turn, int mode, int player_idx) {
    memset(obs, 0, OBS_SIZE * sizeof(float));
    int idx = 0;

    Pokemon* my_active = active_pokemon(my);
    Pokemon* opp_active = active_pokemon(opp);

    // ========== My Active Pokemon (23 features: idx 0-22) ==========
    obs[idx++] = (my_active->max_hp > 0) ?
                 (float)my_active->hp / (float)my_active->max_hp : 0.0f;    // 0
    obs[idx++] = (float)my_active->species / (float)NUM_SPECIES;             // 1
    obs[idx++] = (float)my_active->type1 / (float)NUM_TYPES;                 // 2
    obs[idx++] = (my_active->type2 != TYPE_NONE) ?
                 (float)my_active->type2 / (float)NUM_TYPES : 0.0f;          // 3
    obs[idx++] = (float)calc_stat(my_active->base_atk) / 500.0f;            // 4
    obs[idx++] = (float)calc_stat(my_active->base_def) / 500.0f;            // 5
    obs[idx++] = (float)calc_stat(my_active->base_spc) / 500.0f;            // 6
    obs[idx++] = (float)calc_stat(my_active->base_spe) / 500.0f;            // 7
    obs[idx++] = (float)(my->atk_stage + 6) / 12.0f;                        // 8
    obs[idx++] = (float)(my->def_stage + 6) / 12.0f;                        // 9
    obs[idx++] = (float)(my->spc_stage + 6) / 12.0f;                        // 10
    obs[idx++] = (float)(my->spe_stage + 6) / 12.0f;                        // 11
    // Status one-hot (6 flags)
    obs[idx++] = (my_active->status == STATUS_SLEEP) ? 1.0f : 0.0f;         // 12
    obs[idx++] = (my_active->status == STATUS_FREEZE) ? 1.0f : 0.0f;        // 13
    obs[idx++] = (my_active->status == STATUS_BURN) ? 1.0f : 0.0f;          // 14
    obs[idx++] = (my_active->status == STATUS_POISON) ? 1.0f : 0.0f;        // 15
    obs[idx++] = (my_active->status == STATUS_TOXIC) ? 1.0f : 0.0f;         // 16
    obs[idx++] = (my_active->status == STATUS_PARALYSIS) ? 1.0f : 0.0f;     // 17
    // Volatile status
    obs[idx++] = (float)my->is_confused;                                     // 18
    obs[idx++] = (float)my->is_seeded;                                       // 19
    obs[idx++] = (my->substitute_hp > 0) ? 1.0f : 0.0f;                     // 20
    obs[idx++] = (float)my->is_recharging;                                   // 21
    obs[idx++] = (float)my->is_trapped;                                      // 22

    // ========== My Side Conditions (2 features: idx 23-24) ==========
    obs[idx++] = (float)my->has_reflect;                                     // 23
    obs[idx++] = (float)my->has_light_screen;                                // 24

    // ========== My Moves (4 x 7 = 28 features: idx 25-52) ==========
    // Per move: type, power, accuracy, pp_frac, is_physical, has_stab, type_eff
    for (int i = 0; i < NUM_MOVE_SLOTS; i++) {
        MoveSlot* ms = &my_active->moves[i];
        if (ms->id != MOVE_NONE) {
            const MoveData* md = &MOVE_DATA[ms->id];
            obs[idx++] = (float)md->type / (float)NUM_TYPES;
            obs[idx++] = (float)md->power / 250.0f;
            obs[idx++] = (float)md->accuracy / 100.0f;
            obs[idx++] = (ms->max_pp > 0) ?
                         (float)ms->pp / (float)ms->max_pp : 0.0f;
            obs[idx++] = (md->power > 0) ?
                         (TYPE_IS_PHYSICAL[md->type] ? 1.0f : 0.0f) : 0.5f;
            // STAB indicator
            obs[idx++] = (md->type == my_active->type1 ||
                         md->type == my_active->type2) ? 1.0f : 0.0f;
            // Type effectiveness vs opponent active (0..4 mapped to 0..1)
            float eff = type_effectiveness_float(md->type,
                            opp_active->type1, opp_active->type2);
            obs[idx++] = eff / 4.0f;
        } else {
            idx += 7;
        }
    }

    // ========== My Team (6 x 4 = 24 features: idx 53-76) ==========
    for (int i = 0; i < NUM_POKEMON; i++) {
        Pokemon* p = &my->team[i];
        if (p->species != SPECIES_NONE && p->max_hp > 0) {
            obs[idx++] = (float)p->hp / (float)p->max_hp;
            obs[idx++] = (float)p->status / 6.0f;
            obs[idx++] = (float)p->type1 / (float)NUM_TYPES;
            obs[idx++] = (float)p->is_alive;
        } else {
            idx += 4;
        }
    }

    // ========== Opponent Active Pokemon (23 features: idx 77-99) ==========
    obs[idx++] = (opp_active->max_hp > 0) ?
                 (float)opp_active->hp / (float)opp_active->max_hp : 0.0f;  // 77
    obs[idx++] = (float)opp_active->species / (float)NUM_SPECIES;            // 78
    obs[idx++] = (float)opp_active->type1 / (float)NUM_TYPES;                // 79
    obs[idx++] = (opp_active->type2 != TYPE_NONE) ?
                 (float)opp_active->type2 / (float)NUM_TYPES : 0.0f;         // 80
    obs[idx++] = (float)calc_stat(opp_active->base_atk) / 500.0f;           // 81
    obs[idx++] = (float)calc_stat(opp_active->base_def) / 500.0f;           // 82
    obs[idx++] = (float)calc_stat(opp_active->base_spc) / 500.0f;           // 83
    obs[idx++] = (float)calc_stat(opp_active->base_spe) / 500.0f;           // 84
    obs[idx++] = (float)(opp->atk_stage + 6) / 12.0f;                       // 85
    obs[idx++] = (float)(opp->def_stage + 6) / 12.0f;                       // 86
    obs[idx++] = (float)(opp->spc_stage + 6) / 12.0f;                       // 87
    obs[idx++] = (float)(opp->spe_stage + 6) / 12.0f;                       // 88
    // Status one-hot (6 flags)
    obs[idx++] = (opp_active->status == STATUS_SLEEP) ? 1.0f : 0.0f;        // 89
    obs[idx++] = (opp_active->status == STATUS_FREEZE) ? 1.0f : 0.0f;       // 90
    obs[idx++] = (opp_active->status == STATUS_BURN) ? 1.0f : 0.0f;         // 91
    obs[idx++] = (opp_active->status == STATUS_POISON) ? 1.0f : 0.0f;       // 92
    obs[idx++] = (opp_active->status == STATUS_TOXIC) ? 1.0f : 0.0f;        // 93
    obs[idx++] = (opp_active->status == STATUS_PARALYSIS) ? 1.0f : 0.0f;    // 94
    // Volatile status
    obs[idx++] = (float)opp->is_confused;                                    // 95
    obs[idx++] = (float)opp->is_seeded;                                      // 96
    obs[idx++] = (opp->substitute_hp > 0) ? 1.0f : 0.0f;                    // 97
    obs[idx++] = (float)opp->is_recharging;                                  // 98
    obs[idx++] = (float)opp->is_trapped;                                     // 99

    // ========== Opponent Side Conditions (2 features: idx 100-101) ==========
    obs[idx++] = (float)opp->has_reflect;                                    // 100
    obs[idx++] = (float)opp->has_light_screen;                               // 101

    // ========== Opponent Team (6 x 4 = 24 features: idx 102-125) ==========
    for (int i = 0; i < NUM_POKEMON; i++) {
        Pokemon* p = &opp->team[i];
        if (p->species != SPECIES_NONE && p->max_hp > 0) {
            obs[idx++] = (float)p->hp / (float)p->max_hp;
            obs[idx++] = (float)p->status / 6.0f;
            obs[idx++] = (float)p->type1 / (float)NUM_TYPES;
            obs[idx++] = (float)p->is_alive;
        } else {
            idx += 4;
        }
    }

    // ========== Battle Info (4 features: idx 126-129) ==========
    obs[idx++] = (float)turn / (float)MAX_TURNS;                             // 126
    obs[idx++] = (float)mode / 3.0f;                                         // 127
    obs[idx++] = (float)my->alive_count / 6.0f;                              // 128
    obs[idx++] = (float)opp->alive_count / 6.0f;                             // 129

    // ========== Action Mask (10 features: idx 130-139) ==========
    int mask[NUM_ACTIONS];
    get_action_mask(my, mode, player_idx, mask);
    for (int i = 0; i < NUM_ACTIONS; i++) {
        obs[idx++] = (float)mask[i];
    }

    // Total packed: 23 + 2 + 28 + 24 + 23 + 2 + 24 + 4 + 10 = 140
}

// ============================================================================
// Battle Turn Resolution
// ============================================================================

static void resolve_turn(Battle* battle, int p1_action, int p2_action) {
    Player* p1 = &battle->players[0];
    Player* p2 = &battle->players[1];

    // Handle force switch mode
    if (battle->mode != 0) {
        if ((battle->mode == 1 || battle->mode == 3) && p1_action >= 4 && p1_action <= 9) {
            int switch_target = p1_action - 4;
            if (can_switch(p1, switch_target)) {
                do_switch(p1, switch_target);
                evt_push(EVT_SWITCH, 0, 0, p1->team[p1->active_idx].species);
            }
        }
        if ((battle->mode == 2 || battle->mode == 3) && p2_action >= 4 && p2_action <= 9) {
            int switch_target = p2_action - 4;
            if (can_switch(p2, switch_target)) {
                do_switch(p2, switch_target);
                evt_push(EVT_SWITCH, 1, 0, p2->team[p2->active_idx].species);
            }
        }
        battle->mode = 0;
        return;
    }

    // Normal turn
    battle->turn++;

    // Determine if players are switching
    int p1_switching = (p1_action >= 4 && p1_action <= 9);
    int p2_switching = (p2_action >= 4 && p2_action <= 9);

    // Switches happen first (before moves)
    if (p1_switching) {
        int target = p1_action - 4;
        if (can_switch(p1, target)) {
            do_switch(p1, target);
            evt_push(EVT_SWITCH, 0, 0, p1->team[p1->active_idx].species);
        }
    }
    if (p2_switching) {
        int target = p2_action - 4;
        if (can_switch(p2, target)) {
            do_switch(p2, target);
            evt_push(EVT_SWITCH, 1, 0, p2->team[p2->active_idx].species);
        }
    }

    // If both used moves, determine speed priority
    if (!p1_switching && !p2_switching) {
        int p1_speed = get_effective_speed(p1);
        int p2_speed = get_effective_speed(p2);

        // Speed tie: random
        int p1_first;
        if (p1_speed == p2_speed) {
            p1_first = pb_rand_chance(50);
        } else {
            p1_first = (p1_speed > p2_speed);
        }

        int p1_move = p1_action; // 0-3
        int p2_move = p2_action; // 0-3
        if (p1_move >= NUM_MOVE_SLOTS) p1_move = 0;
        if (p2_move >= NUM_MOVE_SLOTS) p2_move = 0;

        if (p1_first) {
            execute_move(p1, p2, p1_move, 0);
            if (active_pokemon(p2)->is_alive) {
                execute_move(p2, p1, p2_move, 1);
            }
        } else {
            execute_move(p2, p1, p2_move, 1);
            if (active_pokemon(p1)->is_alive) {
                execute_move(p1, p2, p1_move, 0);
            }
        }
    } else if (!p1_switching) {
        // P1 uses move (P2 already switched)
        int p1_move = p1_action;
        if (p1_move >= NUM_MOVE_SLOTS) p1_move = 0;
        execute_move(p1, p2, p1_move, 0);
    } else if (!p2_switching) {
        // P2 uses move (P1 already switched)
        int p2_move = p2_action;
        if (p2_move >= NUM_MOVE_SLOTS) p2_move = 0;
        execute_move(p2, p1, p2_move, 1);
    }
    // Both switched: nothing more to do

    // End of turn effects
    apply_end_of_turn(p1, p2, 0);
    apply_end_of_turn(p2, p1, 1);

    // Check for fainted Pokemon that need replacement
    int p1_fainted = !active_pokemon(p1)->is_alive && p1->alive_count > 0;
    int p2_fainted = !active_pokemon(p2)->is_alive && p2->alive_count > 0;

    if (p1_fainted && p2_fainted) {
        battle->mode = 3;
    } else if (p1_fainted) {
        battle->mode = 1;
    } else if (p2_fainted) {
        battle->mode = 2;
    }
}

// ============================================================================
// Reward Shaping Helpers
// ============================================================================

static int total_team_hp(Player* p) {
    int total = 0;
    for (int i = 0; i < NUM_POKEMON; i++) {
        total += p->team[i].hp;
    }
    return total;
}

static int total_team_max_hp(Player* p) {
    int total = 0;
    for (int i = 0; i < NUM_POKEMON; i++) {
        total += p->team[i].max_hp;
    }
    return total;
}

static inline unsigned long long signature_mix(unsigned long long h, unsigned long long v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

// Coarse progress signature for Endless Battle Clause detection.
// We intentionally ignore active slot, stat stages, and other volatile flags so
// switch loops and no-impact turns still count as stale progress.
static unsigned long long battle_progress_signature(const Battle* battle) {
    unsigned long long h = 0x14650fb0739d0383ULL;
    h = signature_mix(h, (unsigned long long)battle->mode);

    for (int side = 0; side < 2; side++) {
        const Player* p = &battle->players[side];
        h = signature_mix(h, (unsigned long long)p->alive_count);

        for (int i = 0; i < NUM_POKEMON; i++) {
            const Pokemon* mon = &p->team[i];
            h = signature_mix(h, (unsigned long long)mon->species);
            h = signature_mix(h, (unsigned long long)mon->hp);
            h = signature_mix(h, (unsigned long long)mon->status);
            h = signature_mix(h, (unsigned long long)mon->sleep_turns);
            h = signature_mix(h, (unsigned long long)mon->toxic_counter);
            h = signature_mix(h, (unsigned long long)mon->is_alive);
        }
    }
    return h;
}

// ============================================================================
// Check Win Condition
// ============================================================================

static int check_winner(Battle* battle) {
    // Returns: 0 = ongoing, 1 = P1 wins, -1 = P2 wins
    if (battle->players[0].alive_count <= 0) return -1;
    if (battle->players[1].alive_count <= 0) return 1;
    return 0;
}

// ============================================================================
// Random Legal Action (for bot)
// ============================================================================

static int random_legal_action(Player* p, int mode, int player_idx) {
    int mask[NUM_ACTIONS];
    get_action_mask(p, mode, player_idx, mask);

    int legal[NUM_ACTIONS];
    int n_legal = 0;
    for (int i = 0; i < NUM_ACTIONS; i++) {
        if (mask[i]) {
            legal[n_legal++] = i;
        }
    }
    if (n_legal == 0) return 0; // fallback
    return legal[pb_rand_int(n_legal)];
}

// ============================================================================
// Position Evaluation (ported from poke-engine Gen1 evaluate.rs)
// Heuristic evaluation of a battle position from a given player's perspective.
// ============================================================================

// Evaluation weights
#define EVAL_ALIVE          30.0f
#define EVAL_HP            100.0f
#define EVAL_ATK_BOOST      30.0f
#define EVAL_DEF_BOOST      15.0f
#define EVAL_SPC_BOOST      30.0f
#define EVAL_SPE_BOOST      30.0f
#define EVAL_FROZEN        -40.0f
#define EVAL_ASLEEP        -25.0f
#define EVAL_PARALYZED     -25.0f
#define EVAL_TOXIC         -30.0f
#define EVAL_POISONED      -10.0f
#define EVAL_BURNED        -25.0f
#define EVAL_LEECH_SEED    -30.0f
#define EVAL_SUBSTITUTE     40.0f
#define EVAL_CONFUSION     -20.0f
#define EVAL_REFLECT        20.0f
#define EVAL_LIGHT_SCREEN   20.0f

static float eval_boost_multiplier(int stage) {
    static const float mults[] = {
        -3.3f, -3.15f, -3.0f, -2.5f, -2.0f, -1.0f, 0.0f,
         1.0f,  2.0f,   2.5f,  3.0f,  3.15f, 3.3f
    };
    int idx = stage + 6;
    if (idx < 0) idx = 0;
    if (idx > 12) idx = 12;
    return mults[idx];
}

static float eval_pokemon_burn(Pokemon* p) {
    // Burn penalty scales with how many physical moves the pokemon has.
    // Special attackers are penalized less.
    float mult = 0.0f;
    for (int i = 0; i < NUM_MOVE_SLOTS; i++) {
        if (p->moves[i].id == MOVE_NONE) continue;
        const MoveData* md = &MOVE_DATA[p->moves[i].id];
        if (md->power > 0 && TYPE_IS_PHYSICAL[md->type]) {
            mult += 1.0f;
        }
    }
    if (p->base_spc > p->base_atk) {
        mult /= 2.0f;
    }
    return mult * EVAL_BURNED;
}

static float eval_single_pokemon(Pokemon* p) {
    float score = 0.0f;
    score += EVAL_HP * ((float)p->hp / (float)p->max_hp);

    switch (p->status) {
        case STATUS_BURN:      score += eval_pokemon_burn(p); break;
        case STATUS_FREEZE:    score += EVAL_FROZEN; break;
        case STATUS_SLEEP:     score += EVAL_ASLEEP; break;
        case STATUS_PARALYSIS: score += EVAL_PARALYZED; break;
        case STATUS_TOXIC:     score += EVAL_TOXIC; break;
        case STATUS_POISON:    score += EVAL_POISONED; break;
        default: break;
    }

    if (score < 0.0f) score = 0.0f;
    score += EVAL_ALIVE;
    return score;
}

// Evaluate battle position from perspective of player `side` (0 or 1).
// Positive = good for `side`, negative = bad.
static float evaluate_position(Battle* battle, int side) {
    float score = 0.0f;
    Player* me = &battle->players[side];
    Player* opp = &battle->players[1 - side];

    // Score my team
    for (int i = 0; i < NUM_POKEMON; i++) {
        if (me->team[i].is_alive) {
            score += eval_single_pokemon(&me->team[i]);
            if (i == me->active_idx) {
                if (me->is_seeded)        score += EVAL_LEECH_SEED;
                if (me->substitute_hp > 0) score += EVAL_SUBSTITUTE;
                if (me->is_confused)      score += EVAL_CONFUSION;
                if (me->has_reflect)      score += EVAL_REFLECT;
                if (me->has_light_screen) score += EVAL_LIGHT_SCREEN;
                score += eval_boost_multiplier(me->atk_stage) * EVAL_ATK_BOOST;
                score += eval_boost_multiplier(me->def_stage) * EVAL_DEF_BOOST;
                score += eval_boost_multiplier(me->spc_stage) * EVAL_SPC_BOOST;
                score += eval_boost_multiplier(me->spe_stage) * EVAL_SPE_BOOST;
            }
        }
    }

    // Subtract opponent's team score
    for (int i = 0; i < NUM_POKEMON; i++) {
        if (opp->team[i].is_alive) {
            score -= eval_single_pokemon(&opp->team[i]);
            if (i == opp->active_idx) {
                if (opp->is_seeded)        score -= EVAL_LEECH_SEED;
                if (opp->substitute_hp > 0) score -= EVAL_SUBSTITUTE;
                if (opp->is_confused)      score -= EVAL_CONFUSION;
                if (opp->has_reflect)      score -= EVAL_REFLECT;
                if (opp->has_light_screen) score -= EVAL_LIGHT_SCREEN;
                score -= eval_boost_multiplier(opp->atk_stage) * EVAL_ATK_BOOST;
                score -= eval_boost_multiplier(opp->def_stage) * EVAL_DEF_BOOST;
                score -= eval_boost_multiplier(opp->spc_stage) * EVAL_SPC_BOOST;
                score -= eval_boost_multiplier(opp->spe_stage) * EVAL_SPE_BOOST;
            }
        }
    }

    return score;
}

// ============================================================================
// Heuristic Bot (1-ply minimax: best worst-case across opponent actions)
// ============================================================================

static int heuristic_action(Battle* battle, int player_idx, int mode) {
    // Suppress event logging during heuristic simulations
    PokeBattle* saved_event_env = g_event_env;
    g_event_env = NULL;

    int mask[NUM_ACTIONS];
    get_action_mask(&battle->players[player_idx], mode, player_idx, mask);

    int opp_idx = 1 - player_idx;
    int opp_mask[NUM_ACTIONS];
    get_action_mask(&battle->players[opp_idx], mode, opp_idx, opp_mask);

    int legal[NUM_ACTIONS], n_legal = 0;
    for (int i = 0; i < NUM_ACTIONS; i++) {
        if (mask[i]) legal[n_legal++] = i;
    }
    if (n_legal <= 1) {
        g_event_env = saved_event_env;
        return n_legal == 0 ? 0 : legal[0];
    }

    int opp_legal[NUM_ACTIONS], n_opp_legal = 0;
    for (int i = 0; i < NUM_ACTIONS; i++) {
        if (opp_mask[i]) opp_legal[n_opp_legal++] = i;
    }
    if (n_opp_legal == 0) { opp_legal[0] = 0; n_opp_legal = 1; }

    unsigned long long saved_rng = pb_rng_state;
    float best_score = -1e9f;
    int best_action = legal[0];

    for (int i = 0; i < n_legal; i++) {
        float worst_case = 1e9f;

        for (int j = 0; j < n_opp_legal; j++) {
            Battle sim;
            memcpy(&sim, battle, sizeof(Battle));

            int p1_act = (player_idx == 0) ? legal[i] : opp_legal[j];
            int p2_act = (player_idx == 0) ? opp_legal[j] : legal[i];

            // Use deterministic but unique RNG per simulation
            pb_rng_state = saved_rng ^ ((unsigned long long)(i * NUM_ACTIONS + j + 1) * 6364136223846793005ULL);
            resolve_turn(&sim, p1_act, p2_act);

            int result = check_winner(&sim);
            float eval;
            if (result != 0) {
                eval = (result == (player_idx == 0 ? 1 : -1)) ? 10000.0f : -10000.0f;
            } else {
                eval = evaluate_position(&sim, player_idx);
            }

            if (eval < worst_case) worst_case = eval;
        }

        if (worst_case > best_score) {
            best_score = worst_case;
            best_action = legal[i];
        }
    }

    pb_rng_state = saved_rng;
    g_event_env = saved_event_env;
    return best_action;
}

// ============================================================================
// MCTS Bot (Monte Carlo rollouts with static evaluation at leaves)
// Inspired by poke-engine's approach: no random playouts, uses heuristic eval.
// ============================================================================

static int mcts_action(Battle* battle, int player_idx, int mode,
                       int iterations, int depth) {
    // Suppress event logging during MCTS simulations
    PokeBattle* saved_event_env = g_event_env;
    g_event_env = NULL;

    int mask[NUM_ACTIONS];
    get_action_mask(&battle->players[player_idx], mode, player_idx, mask);

    int legal[NUM_ACTIONS], n_legal = 0;
    for (int i = 0; i < NUM_ACTIONS; i++) {
        if (mask[i]) legal[n_legal++] = i;
    }
    if (n_legal <= 1) return n_legal == 0 ? 0 : legal[0];

    unsigned long long saved_rng = pb_rng_state;

    float scores[NUM_ACTIONS];
    int counts[NUM_ACTIONS];
    memset(scores, 0, sizeof(scores));
    memset(counts, 0, sizeof(counts));

    int iters_per_action = iterations / n_legal;
    if (iters_per_action < 1) iters_per_action = 1;

    for (int i = 0; i < n_legal; i++) {
        int action = legal[i];

        for (int iter = 0; iter < iters_per_action; iter++) {
            Battle sim;
            memcpy(&sim, battle, sizeof(Battle));

            // Unique RNG seed per simulation
            pb_rng_state = saved_rng ^ ((unsigned long long)(i * iters_per_action + iter + 1) * 6364136223846793005ULL);

            // Get random opponent action for this rollout
            int opp_idx = 1 - player_idx;
            int opp_action = random_legal_action(&sim.players[opp_idx], mode, opp_idx);

            int p1_act = (player_idx == 0) ? action : opp_action;
            int p2_act = (player_idx == 0) ? opp_action : action;

            resolve_turn(&sim, p1_act, p2_act);

            // Continue with random rollouts for 'depth' turns
            for (int d = 0; d < depth && check_winner(&sim) == 0; d++) {
                int sim_mode = sim.mode;
                int r1 = random_legal_action(&sim.players[0], sim_mode, 0);
                int r2 = random_legal_action(&sim.players[1], sim_mode, 1);
                resolve_turn(&sim, r1, r2);
            }

            int result = check_winner(&sim);
            float eval;
            if (result != 0) {
                eval = (result == (player_idx == 0 ? 1 : -1)) ? 10000.0f : -10000.0f;
            } else {
                eval = evaluate_position(&sim, player_idx);
            }

            scores[i] += eval;
            counts[i]++;
        }
    }

    pb_rng_state = saved_rng;

    // Pick action with best average score
    float best_avg = -1e9f;
    int best_action = legal[0];
    for (int i = 0; i < n_legal; i++) {
        if (counts[i] > 0) {
            float avg = scores[i] / (float)counts[i];
            if (avg > best_avg) {
                best_avg = avg;
                best_action = legal[i];
            }
        }
    }

    g_event_env = saved_event_env;
    return best_action;
}

// Select bot action based on bot_mode setting
static int bot_action(PokeBattle* env, int player_idx) {
    int mode = env->battle.mode;
    switch (env->bot_mode) {
        case BOT_HEURISTIC:
            return heuristic_action(&env->battle, player_idx, mode);
        case BOT_MCTS:
            return mcts_action(&env->battle, player_idx, mode,
                              env->mcts_iterations, env->mcts_depth);
        default:
            return random_legal_action(&env->battle.players[player_idx],
                                      mode, player_idx);
    }
}

// ============================================================================
// PufferLib Interface: init, reset, step, render, close
// ============================================================================

#ifndef NDEBUG
static int move_present_in_pool(const char* name) {
    for (int i = 1; i <= NUM_MOVES; i++) {
        if (strcmp(MOVE_DATA[i].name, name) == 0) return 1;
    }
    return 0;
}

static int species_moveset_contains(SpeciesID species, MoveID move) {
    if (species <= SPECIES_NONE || species > NUM_SPECIES) return 0;
    for (int i = 0; i < NUM_MOVE_SLOTS; i++) {
        if (SPECIES_DATA[species].moveset[i] == move) return 1;
    }
    return 0;
}
#endif

static void validate_battle_rules(void) {
    // === Type Chart Assertions ===
    // Psychic deals normal damage to Ghost (NOT immune) — Showdown Gen 1
    assert(TYPE_CHART[TYPE_PSYCHIC][TYPE_GHOST] == 2);
    // Ghost is immune to Psychic
    assert(TYPE_CHART[TYPE_GHOST][TYPE_PSYCHIC] == 0);
    // Ghost immune to Normal
    assert(TYPE_CHART[TYPE_NORMAL][TYPE_GHOST] == 0);
    // Normal immune to Ghost
    assert(TYPE_CHART[TYPE_GHOST][TYPE_NORMAL] == 0);
    // Bug and Poison mutually SE (Gen 1)
    assert(TYPE_CHART[TYPE_BUG][TYPE_POISON] == 3);
    assert(TYPE_CHART[TYPE_POISON][TYPE_BUG] == 3);
    // Ice neutral to Fire (Gen 1)
    assert(TYPE_CHART[TYPE_ICE][TYPE_FIRE] == 2);
    // Psychic SE against Fighting and Poison
    assert(TYPE_CHART[TYPE_PSYCHIC][TYPE_FIGHTING] == 3);
    assert(TYPE_CHART[TYPE_PSYCHIC][TYPE_POISON] == 3);

    // === Stat Formula Assertions ===
    assert(calc_hp(75) == 353);    // Tauros HP: 2*75+203
    assert(calc_hp(250) == 703);   // Chansey HP: 2*250+203
    assert(calc_stat(135) == 368); // Alakazam Spc: 2*135+98

    // === Move Data Assertions ===
    assert(MOVE_DATA[MOVE_BODY_SLAM].type == TYPE_NORMAL);
    assert(MOVE_DATA[MOVE_BODY_SLAM].power == 85);
    assert(MOVE_DATA[MOVE_BODY_SLAM].effect == EFFECT_PARALYZE_CHANCE);
    assert(MOVE_DATA[MOVE_BODY_SLAM].effect_chance == 30);
    assert(MOVE_DATA[MOVE_PSYCHIC].type == TYPE_PSYCHIC);
    assert(MOVE_DATA[MOVE_PSYCHIC].power == 90);
    assert(MOVE_DATA[MOVE_BLIZZARD].power == 120);
    assert(MOVE_DATA[MOVE_BLIZZARD].effect_chance == 10);
    assert(MOVE_DATA[MOVE_THUNDER_WAVE].type == TYPE_ELECTRIC);
    assert(MOVE_DATA[MOVE_THUNDER_WAVE].power == 0);
    assert(MOVE_DATA[MOVE_EXPLOSION].effect == EFFECT_SELF_DESTRUCT);

    // === Species Data Assertions ===
    assert(SPECIES_DATA[SPECIES_TAUROS].type1 == TYPE_NORMAL);
    assert(SPECIES_DATA[SPECIES_TAUROS].base_hp == 75);
    assert(SPECIES_DATA[SPECIES_TAUROS].base_spe == 110);
    assert(SPECIES_DATA[SPECIES_GENGAR].type1 == TYPE_GHOST);
    assert(SPECIES_DATA[SPECIES_GENGAR].type2 == TYPE_POISON);
    assert(SPECIES_DATA[SPECIES_STARMIE].type1 == TYPE_WATER);
    assert(SPECIES_DATA[SPECIES_STARMIE].type2 == TYPE_PSYCHIC);

    // === Physical/Special Split Assertions (Gen 1) ===
    assert(TYPE_IS_PHYSICAL[TYPE_NORMAL] == 1);
    assert(TYPE_IS_PHYSICAL[TYPE_FIRE] == 0);
    assert(TYPE_IS_PHYSICAL[TYPE_FIGHTING] == 1);
    assert(TYPE_IS_PHYSICAL[TYPE_PSYCHIC] == 0);
    assert(TYPE_IS_PHYSICAL[TYPE_GHOST] == 1);

    // === Stat Stage Table Assertions ===
    assert(STAGE_NUMER[6] == 2 && STAGE_DENOM[6] == 2);   // Stage 0 = 1x
    assert(STAGE_NUMER[12] == 8 && STAGE_DENOM[12] == 2);  // Stage +6 = 4x
    assert(STAGE_NUMER[0] == 2 && STAGE_DENOM[0] == 8);    // Stage -6 = 0.25x

    // === Showdown [Gen 1] OU Standard Rule Alignment ===
    // Ruleset reference:
    // data/mods/gen1/rulesets.ts -> Standard
    // banlist includes Dig/Fly + OHKO + Evasion moves via clauses.
    assert(!move_present_in_pool("Dig"));
    assert(!move_present_in_pool("Fly"));
    assert(!move_present_in_pool("Fissure"));
    assert(!move_present_in_pool("Guillotine"));
    assert(!move_present_in_pool("Horn Drill"));
    assert(!move_present_in_pool("Double Team"));
    assert(!move_present_in_pool("Minimize"));

    // === Team Pool Assertions ===
    assert(OU_LEGAL_SIZE == NUM_SPECIES);
    int seen[NUM_SPECIES + 1];
    memset(seen, 0, sizeof(seen));
    for (int i = 0; i < OU_LEGAL_SIZE; i++) {
        SpeciesID sp = OU_LEGAL[i];
        assert(sp > SPECIES_NONE && sp <= NUM_SPECIES);
        assert(!seen[sp]);
        seen[sp] = 1;
    }
    for (int s = 1; s <= NUM_SPECIES; s++) {
        assert(seen[s] == 1);
    }

    // Every hardcoded species set must avoid duplicate active moves.
    for (int s = 1; s <= NUM_SPECIES; s++) {
        int seen_move[NUM_MOVES + 1];
        memset(seen_move, 0, sizeof(seen_move));
        for (int i = 0; i < NUM_MOVE_SLOTS; i++) {
            MoveID mid = SPECIES_DATA[s].moveset[i];
            if (mid == MOVE_NONE) continue;
            assert(mid > MOVE_NONE && mid <= NUM_MOVES);
            assert(!seen_move[mid]);
            seen_move[mid] = 1;
        }
    }

    // Guard known Gen 1 OU no-tradeback legality regressions.
    assert(!species_moveset_contains(SPECIES_WEEDLE, MOVE_TACKLE));
    assert(!species_moveset_contains(SPECIES_PIDGEY, MOVE_TACKLE));
    assert(!species_moveset_contains(SPECIES_PIDGEOTTO, MOVE_TACKLE));
    assert(!species_moveset_contains(SPECIES_PIDGEOT, MOVE_TACKLE));
    assert(!species_moveset_contains(SPECIES_VULPIX, MOVE_HYPNOSIS));
    assert(!species_moveset_contains(SPECIES_MEOWTH, MOVE_HYPNOSIS));
    assert(!species_moveset_contains(SPECIES_PSYDUCK, MOVE_HYPNOSIS));
    assert(!species_moveset_contains(SPECIES_PONYTA, MOVE_HYPNOSIS));
    assert(!species_moveset_contains(SPECIES_EXEGGCUTE, MOVE_MEGA_DRAIN));
    assert(!species_moveset_contains(SPECIES_RHYHORN, MOVE_BLIZZARD));
    assert(!species_moveset_contains(SPECIES_MR_MIME, MOVE_HYPNOSIS));
}

void init(PokeBattle* env) {
    validate_battle_rules();
    memset(&env->battle, 0, sizeof(Battle));
    env->tick = 0;
    env->last_p1_action = 0;
    env->last_p2_action = 0;
    env->last_result = 0;
    env->p1_episode_return = 0.0f;
    env->p2_episode_return = 0.0f;
    env->episode_count = 0;
    env->client = NULL;
    env->mouse_action = -1;
    env->stale_turns = 0;
    env->last_progress_signature = 0;
    env->enforce_endless_clause = 1;
    env->force_accuracy = -1;
    env->force_secondary = -1;
    env->team_builder_mode = TEAM_BUILDER_ADAPTIVE;
    memset(env->p1_fixed_team, 0, sizeof(env->p1_fixed_team));
    memset(env->p2_fixed_team, 0, sizeof(env->p2_fixed_team));
    if (env->auto_reset != 0 && env->auto_reset != 1) env->auto_reset = 1;
    if (env->mcts_iterations <= 0) env->mcts_iterations = MCTS_DEFAULT_ITERATIONS;
    if (env->mcts_depth <= 0) env->mcts_depth = MCTS_DEFAULT_DEPTH;
    pb_rng_state = env->seed;
    env->rng_state = pb_rng_state;
    team_builder_reset_state(env);
}

static void pack_observations(PokeBattle* env) {
    int turn = env->battle.turn;
    int mode = env->battle.mode;

    if (env->selfplay) {
        // Selfplay: pack learner's obs first, then opponent's obs
        // pufferl.py expects: obs[:obs_size] = learner, obs[obs_size:] = opponent
        int learner = env->learner_side;
        int opponent = 1 - learner;
        pack_player_obs(env->observations, &env->battle.players[learner],
                       &env->battle.players[opponent], turn, mode, learner);
        pack_player_obs(&env->observations[OBS_SIZE], &env->battle.players[opponent],
                       &env->battle.players[learner], turn, mode, opponent);
    } else if (env->num_agents == 2) {
        // Legacy 2-agent mode: separate obs rows per agent
        pack_player_obs(env->observations, &env->battle.players[0],
                       &env->battle.players[1], turn, mode, 0);
        pack_player_obs(&env->observations[OBS_SIZE], &env->battle.players[1],
                       &env->battle.players[0], turn, mode, 1);
    } else {
        // Single agent vs random bot
        pack_player_obs(env->observations, &env->battle.players[0],
                       &env->battle.players[1], turn, mode, 0);
    }
}

void c_reset(PokeBattle* env) {
    env->tick = 0;
    env->last_p1_action = 0;
    env->last_p2_action = 0;
    env->last_result = 0;
    env->p1_episode_return = 0.0f;
    env->p2_episode_return = 0.0f;
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;
    env->episode_count++;
    env->stale_turns = 0;

    // Seed RNG
    pb_rng_state = env->seed + env->episode_count * 1000003ULL;

    // Generate teams (fixed team > adaptive team-builder > random comprehensive OU sampling)
    SpeciesID team1[NUM_POKEMON], team2[NUM_POKEMON];
    int learner_idx = env->selfplay ? env->learner_side : 0;
    int adaptive = (env->team_builder_mode == TEAM_BUILDER_ADAPTIVE);

    if (env->p1_fixed_team[0] != SPECIES_NONE) {
        memcpy(team1, env->p1_fixed_team, sizeof(team1));
    } else if (adaptive) {
        generate_adaptive_team(env, team1, learner_idx == 0);
    } else {
        generate_ou_team(team1);
    }

    if (env->p2_fixed_team[0] != SPECIES_NONE) {
        memcpy(team2, env->p2_fixed_team, sizeof(team2));
    } else if (adaptive) {
        generate_adaptive_team(env, team2, learner_idx == 1);
    } else {
        generate_ou_team(team2);
    }

    // Initialize players
    init_player(&env->battle.players[0], team1);
    init_player(&env->battle.players[1], team2);
    env->battle.turn = 0;
    env->battle.mode = 0;
    env->last_progress_signature = battle_progress_signature(&env->battle);

    pack_observations(env);
    env->rng_state = pb_rng_state;
}

void c_step(PokeBattle* env) {
    // Load per-env RNG state into thread-local for OpenMP safety
    pb_rng_state = env->rng_state;
    // Set up event logging
    g_event_env = env;
    env->event_count = 0;

    // Zero rewards and terminals
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    if (env->num_agents == 2 && !env->selfplay) {
        env->rewards[1] = 0.0f;
        env->terminals[1] = 0;
    }

    env->tick++;

    // Get actions - map from action buffer to p1/p2
    int p1_action, p2_action;

    if (env->selfplay) {
        // Selfplay: actions[0] = learner, actions[1] = opponent
        // Map to p1/p2 based on learner_side
        if (env->learner_side == 0) {
            p1_action = env->actions[0];
            p2_action = env->actions[1];
        } else {
            p1_action = env->actions[1];
            p2_action = env->actions[0];
        }
    } else if (env->num_agents == 2) {
        // Legacy 2-agent: actions[0] = p1, actions[1] = p2
        p1_action = env->actions[0];
        p2_action = env->actions[1];
    } else {
        // Single agent vs bot opponent
        p1_action = env->actions[0];
        p2_action = bot_action(env, 1);
    }

    // Validate P1 action (safety fallback - policy should mask properly)
    {
        int mask[NUM_ACTIONS];
        get_action_mask(&env->battle.players[0], env->battle.mode, 0, mask);
        if (p1_action < 0 || p1_action >= NUM_ACTIONS || !mask[p1_action]) {
            p1_action = random_legal_action(&env->battle.players[0],
                                           env->battle.mode, 0);
        }
    }

    // Validate P2 action
    if (env->num_agents == 2 || env->selfplay) {
        int mask[NUM_ACTIONS];
        get_action_mask(&env->battle.players[1], env->battle.mode, 1, mask);
        if (p2_action < 0 || p2_action >= NUM_ACTIONS || !mask[p2_action]) {
            p2_action = random_legal_action(&env->battle.players[1],
                                           env->battle.mode, 1);
        }
    }
    env->last_p1_action = p1_action;
    env->last_p2_action = p2_action;

    // Snapshot state before resolve for reward shaping
    int pre_p1_hp = total_team_hp(&env->battle.players[0]);
    int pre_p2_hp = total_team_hp(&env->battle.players[1]);
    int pre_p1_alive = env->battle.players[0].alive_count;
    int pre_p2_alive = env->battle.players[1].alive_count;
    float p1_max_hp = (float)total_team_max_hp(&env->battle.players[0]);
    float p2_max_hp = (float)total_team_max_hp(&env->battle.players[1]);
    if (p1_max_hp < 1.0f) p1_max_hp = 1.0f;
    if (p2_max_hp < 1.0f) p2_max_hp = 1.0f;

    unsigned long long pre_progress_signature = battle_progress_signature(&env->battle);

    // Resolve the turn
    resolve_turn(&env->battle, p1_action, p2_action);

    // Auto-resolve opponent-only forced switches in single-agent mode.
    // When P2's Pokemon faints, the bot picks a replacement immediately
    // so the human player never has to "pass" through a dead turn.
    if (!env->selfplay && env->num_agents == 1) {
        while (env->battle.mode == 2) {
            int bot_switch = bot_action(env, 1);
            resolve_turn(&env->battle, 0, bot_switch);
        }
    }

    // Compute reward shaping from HP/KO deltas
    int post_p1_hp = total_team_hp(&env->battle.players[0]);
    int post_p2_hp = total_team_hp(&env->battle.players[1]);
    int post_p1_alive = env->battle.players[0].alive_count;
    int post_p2_alive = env->battle.players[1].alive_count;

    // Damage dealt/taken (normalized by max HP)
    int p1_damage_taken = pre_p1_hp - post_p1_hp;  // positive = lost HP
    int p2_damage_taken = pre_p2_hp - post_p2_hp;  // positive = lost HP
    int p1_kos_scored = pre_p2_alive - post_p2_alive;  // how many opp fainted
    int p1_kos_lost = pre_p1_alive - post_p1_alive;    // how many own fainted

    // P1's shaping reward: good to deal damage and KO, bad to take damage and lose mons
    float p1_shaping = 0.0f;
    p1_shaping += REWARD_DAMAGE_DEALT * (float)p2_damage_taken / p2_max_hp;
    p1_shaping += REWARD_DAMAGE_TAKEN * (float)p1_damage_taken / p1_max_hp;
    p1_shaping += REWARD_KO_OPPONENT * (float)p1_kos_scored;
    p1_shaping += REWARD_KO_SELF * (float)p1_kos_lost;

    // Check for game end
    int result = check_winner(&env->battle);
    env->last_result = result;
    unsigned long long post_progress_signature = battle_progress_signature(&env->battle);
    if (result == 0 && env->battle.mode == 0 && post_progress_signature == pre_progress_signature) {
        env->stale_turns++;
    } else {
        env->stale_turns = 0;
    }
    env->last_progress_signature = post_progress_signature;

    int endless_draw = env->enforce_endless_clause &&
        (env->stale_turns >= ENDLESS_BATTLE_STALE_TURNS);
    if (endless_draw) {
        result = 0;
        env->last_result = 0;
    }
    int done = (result != 0) || endless_draw || (env->tick >= MAX_TURNS);

    // Assign rewards: shaping + terminal
    if (env->selfplay) {
        float learner_shaping = (env->learner_side == 0) ? p1_shaping : -p1_shaping;
        env->rewards[0] = learner_shaping;
        if (result == 1) {
            env->rewards[0] += (env->learner_side == 0) ? REWARD_WIN : REWARD_LOSS;
        } else if (result == -1) {
            env->rewards[0] += (env->learner_side == 0) ? REWARD_LOSS : REWARD_WIN;
        }
        env->p1_episode_return += env->rewards[0];
    } else {
        float p2_shaping = -p1_shaping;  // zero-sum
        env->rewards[0] = p1_shaping;
        if (result == 1) {
            env->rewards[0] += REWARD_WIN;
        } else if (result == -1) {
            env->rewards[0] += REWARD_LOSS;
        }
        env->p1_episode_return += env->rewards[0];
        if (env->num_agents == 2) {
            env->rewards[1] = p2_shaping;
            if (result == 1) {
                env->rewards[1] += REWARD_LOSS;
            } else if (result == -1) {
                env->rewards[1] += REWARD_WIN;
            }
            env->p2_episode_return += env->rewards[1];
        }
    }

    if (done) {
        env->terminals[0] = 1;
        if (env->num_agents == 2 && !env->selfplay) {
            env->terminals[1] = 1;
        }

        // Log metrics
        if (env->selfplay) {
            // Accumulate episode stats until vec_log flushes and averages them.
            env->log.episode_return += env->rewards[0]; // learner's result
            env->log.score += env->rewards[0];
            int learner_won = (result == 1 && env->learner_side == 0) ||
                             (result == -1 && env->learner_side == 1);
            int learner_lost = (result == 1 && env->learner_side == 1) ||
                              (result == -1 && env->learner_side == 0);
            env->log.perf += learner_won ? 1.0f : (learner_lost ? 0.0f : 0.5f);
            env->log.p1_wins += learner_won ? 1.0f : 0.0f;
            env->log.p2_wins += learner_lost ? 1.0f : 0.0f;
            env->log.draws += (result == 0) ? 1.0f : 0.0f;
        } else {
            env->log.episode_return += env->p1_episode_return;
            env->log.score += env->p1_episode_return;
            if (result == 1) {
                env->log.perf += 1.0f;
                env->log.p1_wins += 1.0f;
            } else if (result == -1) {
                env->log.perf += 0.0f;
                env->log.p2_wins += 1.0f;
            } else {
                env->log.perf += 0.5f;
                env->log.draws += 1.0f;
            }
        }
        env->log.episode_length += (float)env->tick;
        env->log.n += 1.0f;

        // Per-species tracking + adaptive team-builder updates.
        {
            int learner_idx = env->selfplay ? env->learner_side : 0;
            int opponent_idx = 1 - learner_idx;
            int learner_won = (result == 1 && learner_idx == 0) || (result == -1 && learner_idx == 1);
            int learner_lost = (result == -1 && learner_idx == 0) || (result == 1 && learner_idx == 1);
            float learner_outcome = learner_won ? 1.0f : (learner_lost ? 0.0f : 0.5f);

            Player* learner = &env->battle.players[learner_idx];
            Player* opponent = &env->battle.players[opponent_idx];

            for (int s = 0; s < NUM_POKEMON; s++) {
                SpeciesID sp = learner->team[s].species;
                if (sp > 0 && sp <= NUM_SPECIES) {
                    env->log.species_games[sp] += 1.0f;
                    env->learner_species_games[sp] += 1.0f;
                    if (learner_won) {
                        env->log.species_wins[sp] += 1.0f;
                        env->learner_species_wins[sp] += 1.0f;
                    }
                }
            }

            for (int s = 0; s < NUM_POKEMON; s++) {
                SpeciesID sp = opponent->team[s].species;
                if (sp > 0 && sp <= NUM_SPECIES) {
                    env->opp_species_games[sp] += 1.0f;
                    if (learner_lost) {
                        env->opp_species_wins[sp] += 1.0f;
                    }
                }
            }

            team_builder_record_result(env, learner_outcome);
            env->log.team_builder_recent_winrate += team_builder_recent_winrate(env);
            env->log.team_builder_pool_coverage +=
                (float)team_builder_unique_seen(env->learner_species_games) / (float)OU_LEGAL_SIZE;
        }

        if (env->auto_reset) {
            c_reset(env);
        } else {
            pack_observations(env);
        }
    } else {
        pack_observations(env);
    }

    // Save thread-local RNG state back to per-env struct
    env->rng_state = pb_rng_state;
}

// c_render is defined in render.h (provides full Showdown battle UI)
#include "render.h"

void c_close(PokeBattle* env) {
    if (env->client) {
        close_client(env->client);
        env->client = NULL;
    }
}

#endif // POKE_BATTLE_H
