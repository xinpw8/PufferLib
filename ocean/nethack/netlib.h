// Static data for the NetHack env: layout and glyph constants, the action
// space and verb table, engine options, and the telemetry structs.
// Included by nethack.h after nletypes.h.
#pragma once

// object-type -> armor slot (ARM_SUIT=0..ARM_SHIRT=6, -1 = not armor), indexed
// by otyp = glyph - NH_GLYPH_OBJ_OFF; generated from the engine's objects[]
// (gen_obj_armcat, NetHack 3.6.6). Device copy inlined in src/nethack.cu.
#define NH_NUM_OBJECTS 453
#define NH_GLYPH_OBJ_OFF 1906
static const signed char nh_obj_armcat[NH_NUM_OBJECTS] = {
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,2,2,2,2,2,2,2,2,2,
  2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,5,5,5,
  5,5,5,5,5,5,5,5,5,1,1,1,1,1,1,1,3,3,3,3,
  4,4,4,4,4,4,4,4,4,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
};

#define NH_ROWS 21
#define NH_COLS 79
#define NH_GRID (NH_ROWS * NH_COLS)

// encoder views (GPU-side): egocentric crop + 5x5 patches over the grid
#define NETHACK_CROP       9
#define NETHACK_CROP_GRID  (NETHACK_CROP * NETHACK_CROP)
#define NETHACK_PAD_GLYPH  5976

// corpse glyphs: [GLYPH_BODY_OFF, +NUMMONS), display.h
#define NETHACK_GLYPH_BODY_OFF 1144
#define NETHACK_NUMMONS        381

// obs layout: glyphs | blstats | extra | inventory | item state | message
#define NETHACK_NUM_OCLASSES 18   // MAXOCLASSES; inv_oclasses pads with 18
#define NETHACK_OFF_GLYPHS  0
#define NETHACK_OFF_BLSTATS (NH_GRID * 2)
#define NETHACK_OFF_EXTRA   (NETHACK_OFF_BLSTATS + NLE_BLSTATS_SIZE * 4)
#define NETHACK_EXTRA_INTS  (2 + NETHACK_NUM_OCLASSES)
// inventory: 55 slot glyphs (slot heads index these), then 8 gated int8
// state fields per slot [buc, spe, quan, ero1, ero2, flags, typeknown, rsvd]
#define NETHACK_INV_SLOTS   NLE_INVENTORY_SIZE
#define NETHACK_OFF_INV     (NETHACK_OFF_EXTRA + NETHACK_EXTRA_INTS * 4)
#define NETHACK_OFF_INVST   (NETHACK_OFF_INV + NETHACK_INV_SLOTS * 2)
// raw topline chars, null-padded; must match NH_MSG_LEN in src/nethack.cu
#define NETHACK_OFF_MSG     (NETHACK_OFF_INVST + NETHACK_INV_SLOTS * NLE_INV_STATE_FIELDS)
#define NETHACK_MSG_LEN     128
#define NETHACK_OBS_SIZE    (NETHACK_OFF_MSG + NETHACK_MSG_LEN)
#define NETHACK_INTERNAL_KILLER_MNUM 9 // killer monster index + 1 (0 = not a monster), death only
#define NETHACK_INTERNAL_KILLER_MLEV 10 // killer monster level, death only

#define NETHACK_MAX_EPISODE_STEPS 10000
#define NETHACK_AUTODISMISS_MAX   64   // cap on prompt-dismiss keystrokes per step
#define NETHACK_MAX_DEPTH         64   // scout bitmaps tracked per episode

// Stats.areas bits; logged as reach_* proportions
#define NETHACK_AREA_MINES      1u   // Gnomish Mines (dnum 2)
#define NETHACK_AREA_MINETOWN   2u   // Mines level 3+ (Minetown band)
#define NETHACK_AREA_DEEP_MINES 4u   // Mines level 5+ (past Minetown)
#define NETHACK_AREA_MAIN_D5    8u   // Dungeons of Doom depth 5+ (Oracle route)
#define NETHACK_AREA_SOKOBAN    16u  // Sokoban (dnum 4)

// nle_obs.misc[] prompt-state flags
enum { NETHACK_MISC_YN = 0, NETHACK_MISC_GETLIN = 1, NETHACK_MISC_XWAIT = 2 };

// action space: verb head (22) + 12 item-slot heads (55) + direction head (8)
#define NETHACK_NUM_ACTIONS 22
#define NETHACK_NUM_DIRS    8
static const int NETHACK_DIR_KEYS[NETHACK_NUM_DIRS] =
    {'k','j','h','l','y','u','b','n'};   // N S W E NW NE SW SE
static const int NETHACK_DIR_DX[NETHACK_NUM_DIRS] = { 0, 0,-1, 1,-1, 1,-1, 1};
static const int NETHACK_DIR_DY[NETHACK_NUM_DIRS] = {-1, 1, 0, 0,-1,-1, 1, 1};
// cmap wall glyphs S_vwall..S_trwall; S_stone excluded (= "unexplored")
#define NETHACK_WALL_GLYPH_LO 2360
#define NETHACK_WALL_GLYPH_HI 2370
// hunger states (hack.h): SATIATED 0, NOT_HUNGRY 1, HUNGRY 2, WEAK 3, FAINTING 4
#define NETHACK_HUNGER_WEAK 3
// major-trouble condition bits (botl.h BL_MASK_): STONE|SLIME|STRNGL|FOODPOIS|TERMILL
#define NETHACK_COND_MAJOR 0x1Fu
#define NETHACK_COND_BAD   0x3FFu  // all afflictions STONE..HALLU; excludes LEV/FLY/RIDE
// stair/ladder cmap glyphs: GLYPH_CMAP_OFF(2359) + S_upstair(23)..S_dnladder(26)
#define NETHACK_GLYPH_UPSTAIR  2382
#define NETHACK_GLYPH_DNSTAIR  2383
#define NETHACK_GLYPH_UPLADDER 2384
#define NETHACK_GLYPH_DNLADDER 2385
// object glyphs [GLYPH_OBJ_OFF, GLYPH_CMAP_OFF); underfoot objects win over terrain
#define NETHACK_GLYPH_OBJ_LO 1906
#define NETHACK_GLYPH_OBJ_HI 2359

enum {
    NETHACK_ACT_MOVE     = 0,
    NETHACK_ACT_RUN      = 1,
    NETHACK_ACT_DOWN     = 2,
    NETHACK_ACT_UP       = 3,
    NETHACK_ACT_KICK     = 4,
    NETHACK_ACT_SEARCH   = 5,
    NETHACK_ACT_ELBERETH = 6,
    NETHACK_ACT_WEAR     = 7,
    NETHACK_ACT_EAT      = 8,
    NETHACK_ACT_QUAFF    = 9,
    NETHACK_ACT_PRAY     = 10,
    NETHACK_ACT_THROW    = 11,
    NETHACK_ACT_ZAP      = 12,
    NETHACK_ACT_SEARCH20 = 13,   // count-prefixed search: ~20 turns of rest
    NETHACK_ACT_PICKUP   = 14,   // no slot: grab the pile underfoot (beyond narrow autopickup)
    NETHACK_ACT_TAKEOFF  = 15,
    NETHACK_ACT_PUTON    = 16,
    NETHACK_ACT_REMOVE   = 17,
    NETHACK_ACT_WIELD    = 18,
    NETHACK_ACT_APPLY    = 19,
    NETHACK_ACT_READ     = 20,
    NETHACK_ACT_DROP     = 21,
};

// !status_updates skips the status renderer + recalc_mapseen (~25% of engine)
#define NETHACK_DEFAULT_OPTIONS \
    "name:Agent-val-dwa-law-fem," \
    "autopickup,color,disclose:+i +a +v +g +c +o," \
    "mention_walls,nobones,nocmdassist,nolegacy,nosparkle," \
    "pickup_burden:unencumbered,pickup_types:$[%!)/," \
    "runmode:teleport,showexp,showscore,time," \
    "!status_updates"

// slot-head legality per verb (masking + decode; execution is nethack_execute)
enum { WORN_ANY = 0, WORN_ONLY = 1, UNWORN_ONLY = 2 };

typedef struct {
    signed char head;       // -1 direct verb, 0..11 = item slot head
    unsigned int item_classes;
    unsigned char wornreq;
} Verb;

static const Verb NETHACK_VERBS[NETHACK_NUM_ACTIONS] = {
    {-1},
    {-1},
    {-1},
    {-1},
    {-1},
    {-1},
    {-1},
    {0,  1u<<3, UNWORN_ONLY},
    {1,  1u<<7, WORN_ANY},
    {2,  1u<<8, WORN_ANY},
    {-1},
    {3,  1u<<2, WORN_ANY},
    {4,  1u<<11, WORN_ANY},
    {-1},
    {-1},
    {5,  1u<<3, WORN_ONLY},
    {6,  (1u<<4)|(1u<<5), UNWORN_ONLY},
    {7,  (1u<<4)|(1u<<5), WORN_ONLY},
    {8,  1u<<2, WORN_ANY},
    {9,  1u<<6, WORN_ANY},
    {10, (1u<<9)|(1u<<10), WORN_ANY},
    {11, 0x3FFFFu, UNWORN_ONLY},
};

// wandb key per verb success counter (NULL = not logged)
static const char* NETHACK_VERB_STAT[NETHACK_NUM_ACTIONS] = {
    NULL, NULL, NULL, NULL, NULL,
    "searches", "engraves", "wears", "eats", "quaffs",
    "prayers", "throws", "zaps", "search20", "pickups",
    "takeoffs", "putons", "removes", "wields", "applies",
    "reads", "drops",
};

typedef struct Log {
    float perf;
    float verb_uses[NETHACK_NUM_ACTIONS];   // success counters, keys = NETHACK_VERB_STAT
    float score;
    float episode_return;
    float episode_length;
    float valid_moves;        // steps that advanced NetHack's turn counter
    float illegal_actions;    // steps that hit a sub-prompt we ESC'd
    float new_tiles;
    float max_depth;          // deepest level reached (depth under-reports at death)
    float enhances;           // #enhance presses (skill advancement claims)
    float floor_eats;         // eats that accepted a floor "eat it?" offer
    float prayers_low_hp;     // prayers at <=25% max HP (looser than real trouble)
    float prayers_starving;   // prayers at hunger >= Weak: TROUBLE_STARVING, prayer feeds you
    // rest/retreat/burden diagnostics (log-only)
    float burdened_frac;      // steps with encumbrance > Unencumbered
    float damage_taken;
    float ac;                 // mean armor class over the episode (lower = better)
    float min_ac;             // best (lowest) AC reached this episode
    float armor_swaps;        // atomic WEAR swaps (auto-takeoff + wear in one step)
    float heal_hp;            // HP restored by heal actions (quaff/pray) this episode
    float cures;              // bad conditions cleared this episode
    float game_time;          // NetHack turns survived
    float max_xp_level;
    // episode end reason, one-hot (game_end_types in hack.h)
    float death_combat;
    float death_starved;
    float death_smited;       // god's wrath (NLE_HOW_WRATH), not a monster kill
    float death_other;
    // combat-death anatomy (0 for non-combat episodes; ~95% are combat)
    float death_mon_level;    // killer's monster level (vs max_xp_level = the mismatch)
    float death_adj_monsters; // hostile monsters adjacent on the last obs before death
    float death_maxhp;        // max HP at death (progression measure)
    float truncated;          // hit NETHACK_MAX_EPISODE_STEPS
    // 0/1 per episode; the logged mean is the proportion
    float reach_mines;
    float reach_minetown;
    float reach_deep_mines;
    float reach_main_d5;
    float reach_sokoban;
    float n;
} Log;

// per-episode stats; cleared with one memset per reset
typedef struct Stats {
    long verb_uses[NETHACK_NUM_ACTIONS];
    long valid_moves;
    long illegal_actions;
    long new_tiles;
    long enhances;
    long armor_swaps;         // atomic WEAR that auto-took-off an occupant
    long burdened_steps;
    long heal_hp;             // HP restored by heal actions (quaff/pray)
    long cures;               // bad conditions cleared
    int  min_ac;              // best (lowest) AC reached this episode
    long last_maxhp;
    int  last_adj;       // hostile monsters adjacent, last obs
    long prayers_low_hp;
    long prayers_starving;
    long floor_eats;
    long damage;
    long ac_sum;            // sum of AC over living steps; mean = ac_sum/length
    int max_depth;
    int max_xp;
    unsigned areas;         // NETHACK_AREA_* bits
    float ret;
    int length;
    // per-level first-visit bitmaps; branch levels sharing a depth share one
    unsigned char visited[NETHACK_MAX_DEPTH][(NH_GRID + 7) / 8];
} Stats;
