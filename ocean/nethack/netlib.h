// Static data for the NetHack env: layout and glyph constants, the action
// space and verb table, engine options, and the telemetry structs.
// Included by nethack.h after nletypes.h.
#pragma once

// object-type -> armor slot (ARM_SUIT=0..ARM_SHIRT=6, -1 = not armor), indexed
// by otyp = glyph - NH_GLYPH_OBJ_OFF; generated from the engine's objects[]
// (gen_obj_armcat, NetHack 3.6.6). Device copy inlined in ocean/nethack/nethack.cu.
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

// object-type -> base cost in zorkmids, same otyp indexing as nh_obj_armcat
// (gen_obj_cost, NetHack 3.6.6). Telemetry only: glyphs are post-shuffle,
// so for shuffled classes this is the appearance's cost, not the item's.
static const short nh_obj_cost[NH_NUM_OBJECTS] = {
  0,2,2,2,5,4,2,2,5,20,3,3,3,3,40,3,5,4,4,4,
  40,4,6,4,4,2,100,8,40,10,10,10,10,15,75,10,10,15,50,80,
  500,300,10,6,5,6,10,10,7,5,50,5,5,7,7,8,5,10,5,3,
  3,5,4,4,4,60,60,60,60,20,40,8,10,20,1,80,1,8,10,50,
  50,50,1200,1200,900,900,900,1200,900,900,900,700,700,500,500,500,700,500,500,500,
  600,820,400,80,90,240,240,75,75,45,15,100,80,5,10,3,2,2,60,40,
  50,50,50,50,40,50,60,60,50,3,7,7,7,10,10,50,8,50,50,50,
  8,16,12,50,50,50,8,8,30,30,100,150,150,150,150,100,200,200,100,100,
  200,100,150,300,100,150,200,150,150,200,200,200,300,300,300,150,150,100,150,150,
  150,150,150,150,150,150,150,0,30000,8,16,42,2,100,100,100,10,20,10,10,
  20,12,10,50,200,10,60,80,20,50,150,20,75,30,30,20,80,50,180,60,
  10,10,12,36,15,50,50,50,50,50,50,15,25,25,50,50,100,5000,5000,15,
  5,9,5,5,105,1,6,6,6,6,6,6,7,9,7,10,9,7,7,7,
  17,15,10,10,7,15,45,35,45,25,20,5,300,100,100,150,300,200,200,100,
  150,50,100,100,300,200,150,150,150,100,200,200,50,50,50,250,250,100,80,100,
  100,100,80,60,200,200,300,50,100,100,100,20,100,200,100,200,300,300,300,100,
  100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,60,
  500,200,400,400,100,700,100,100,100,100,100,200,200,200,200,200,200,200,300,300,
  300,300,300,300,400,300,400,400,400,300,500,300,600,600,600,600,700,100,100,300,
  0,20,10000,100,150,150,200,500,100,150,150,150,150,150,200,200,200,150,150,150,
  150,150,175,175,175,500,175,150,150,150,1,4500,4000,3500,3250,3000,2500,2500,2000,1500,
  1500,1000,900,850,800,700,700,600,500,400,200,200,300,0,0,0,0,0,0,0,
  0,0,60,1,45,1,0,0,0,10,0,0,0,
};

#define NETHACK_PATH_MAX     128    // engine records up to 128 hero tiles/step
#define NETHACK_GLYPH_ALTAR  2386   // GLYPH_CMAP_OFF + S_altar

// corpse glyphs: [GLYPH_BODY_OFF, +NUMMONS), display.h
#define NETHACK_GLYPH_BODY_OFF 1144
#define NETHACK_NUMMONS        381

// obs layout: glyphs | blstats | extra | inventory | item state | message
#define NETHACK_NUM_OCLASSES 18   // MAXOCLASSES; inv_oclasses pads with 18
#define NETHACK_OFF_GLYPHS  0
#define NETHACK_OFF_BLSTATS (NH_GRID * 2)
#define NETHACK_OFF_EXTRA   (NETHACK_OFF_BLSTATS + NLE_BLSTATS_SIZE * 4)
// [0] engraving, [1] prev_action, [2..] per-class inv counts,
// then in-shop bit and affordability percent
#define NETHACK_EXTRA_INTS  (2 + NETHACK_NUM_OCLASSES + 2)
#define NETHACK_EXTRA_SHOP  (2 + NETHACK_NUM_OCLASSES)
// inventory: 55 slot glyphs (slot heads index these), then 8 gated int8
// state fields per slot [buc, spe, quan, ero1, ero2, flags, typeknown, rsvd]
#define NETHACK_INV_SLOTS   NLE_INVENTORY_SIZE
#define NETHACK_OFF_INV     (NETHACK_OFF_EXTRA + NETHACK_EXTRA_INTS * 4)
#define NETHACK_OFF_INVST   (NETHACK_OFF_INV + NETHACK_INV_SLOTS * 2)
// raw topline chars, null-padded; must match NH_MSG_LEN in ocean/nethack/nethack.cu
#define NETHACK_OFF_MSG     (NETHACK_OFF_INVST + NETHACK_INV_SLOTS * NLE_INV_STATE_FIELDS)
#define NETHACK_MSG_LEN     128
#define NETHACK_OBS_SIZE    (NETHACK_OFF_MSG + NETHACK_MSG_LEN)
#define NETHACK_INTERNAL_KILLER_MNUM 9 // killer monster index + 1 (0 = not a monster), death only
#define NETHACK_INTERNAL_KILLER_MLEV 10 // killer monster level, death only

#define NETHACK_MAX_EPISODE_STEPS 10000
#define NETHACK_AUTODISMISS_MAX   64   // cap on prompt-dismiss keystrokes per step
#define NETHACK_MAX_DEPTH         64   // scout bitmaps: max distinct (dnum, dlevel) floors per episode

// Stats.areas bits; logged as reach_* proportions
#define NETHACK_AREA_MINES      1u   // Gnomish Mines (dnum 2)
#define NETHACK_AREA_MINETOWN   2u   // Mines level 3+ (Minetown band)
#define NETHACK_AREA_DEEP_MINES 4u   // Mines level 5+ (past Minetown)
#define NETHACK_AREA_MAIN_D5    8u   // Dungeons of Doom depth 5+ (Oracle route)
#define NETHACK_AREA_SOKOBAN    16u  // Sokoban (dnum 4)

// nle_obs.misc[] prompt-state flags
enum { NETHACK_MISC_YN = 0, NETHACK_MISC_GETLIN = 1, NETHACK_MISC_XWAIT = 2 };

// action space: verb head (22) + 12 item-slot heads (55) + 6 per-verb
// direction heads (8 each: MOVE RUN KICK THROW ZAP APPLY)
#define NETHACK_NUM_ACTIONS 23
#define NETHACK_NUM_DIRS    8
#define NETHACK_DIR_HEADS   6
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
    NETHACK_ACT_ALTAR_ID = 22,   // bulk BUC-identify on an altar
};

// dir-head index (0..NETHACK_DIR_HEADS-1) for verbs that take a direction
static inline int nethack_dir_head(int verb) {
    switch (verb) {
        case NETHACK_ACT_MOVE:  return 0;
        case NETHACK_ACT_RUN:   return 1;
        case NETHACK_ACT_KICK:  return 2;
        case NETHACK_ACT_THROW: return 3;
        case NETHACK_ACT_ZAP:   return 4;
        case NETHACK_ACT_APPLY: return 5;
    }
    return -1;
}

// !status_updates skips the status renderer + recalc_mapseen (~25% of engine)
#define NETHACK_DEFAULT_OPTIONS \
    "name:Agent-mon-hum-neu-mal," \
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
    {-1}   /* MOVE */,
    {-1}   /* RUN */,
    {-1}   /* DOWN */,
    {-1}   /* UP */,
    {-1}   /* KICK */,
    {-1}   /* SEARCH */,
    {-1}   /* ELBERETH */,
    {0,  1u<<3, UNWORN_ONLY}   /* WEAR */,
    {1,  1u<<7, WORN_ANY}   /* EAT */,
    {2,  1u<<8, WORN_ANY}   /* QUAFF */,
    {-1}   /* PRAY */,
    {3,  1u<<2, WORN_ANY}   /* THROW */,
    {4,  1u<<11, WORN_ANY}   /* ZAP */,
    {-1}   /* SEARCH20 */,
    {-1}   /* PICKUP */,
    {5,  1u<<3, WORN_ONLY}   /* TAKEOFF */,
    {6,  (1u<<4)|(1u<<5), UNWORN_ONLY}   /* PUTON */,
    {7,  (1u<<4)|(1u<<5), WORN_ONLY}   /* REMOVE */,
    {8,  1u<<2, WORN_ANY}   /* WIELD */,
    {9,  1u<<6, WORN_ANY}   /* APPLY */,
    {10, (1u<<9)|(1u<<10), WORN_ANY}   /* READ */,
    {11, 0x3FFFFu, UNWORN_ONLY}   /* DROP */,
    {-1}   /* ALTAR_ID */
};

// wandb key per verb success counter (NULL = not logged)
static const char* NETHACK_VERB_STAT[NETHACK_NUM_ACTIONS] = {
    NULL   /* MOVE */,
    "runs"   /* RUN */,
    NULL   /* DOWN */,
    NULL   /* UP */,
    NULL   /* KICK */,
    "searches"   /* SEARCH */,
    "engraves"   /* ELBERETH */,
    "wears"   /* WEAR */,
    "eats"   /* EAT */,
    "quaffs"   /* QUAFF */,
    "prayers"   /* PRAY */,
    "throws"   /* THROW */,
    "zaps"   /* ZAP */,
    "search20"   /* SEARCH20 */,
    "pickups"   /* PICKUP */,
    "takeoffs"   /* TAKEOFF */,
    "putons"   /* PUTON */,
    "removes"   /* REMOVE */,
    "wields"   /* WIELD */,
    "applies"   /* APPLY */,
    "reads"   /* READ */,
    "drops"   /* DROP */,
    "altar_ids"   /* ALTAR_ID */
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
    float floors;             // unique (dnum, dlevel) floors visited
    float enhances;           // #enhance presses (skill advancement claims)
    float floor_eats;         // eats that accepted a floor "eat it?" offer
    float sells;              // shop sale offers accepted (deliberate drop in a shop)
    float buys;               // shop pickups paid for
    float sale_gold;          // gold received from those sales
    float drop_value;         // base-cost value of everything dropped this episode
    float trouble_frac;       // fraction of steps in prayer-fixable trouble
    float prayers_fed;
    float altar_steps;
    float wear_blind;
    float cursed_worn_frac;
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
    float r_raw[6], r_clip[6], r_death;   // reward decomposition ledgers
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
    long sells;
    long buys;
    long sale_gold;
    long drop_value;
    long trouble_steps;       // steps at Weak+ hunger or hp <= maxhp/7
    long prayers_fed;         // prayers that improved the hunger state
    long altar_steps;
    long wear_blind;          // WEAR of an item whose BUC was unknown
    long cursed_worn_steps;   // steps wearing a known-cursed item
    int  last_hunger;
    long damage;
    long ac_sum;            // sum of AC over living steps; mean = ac_sum/length
    int max_depth;
    int floors;               // unique (dnum, dlevel) count
    unsigned long long floors_bits[16];   // dnum 0..15, bit dlevel-1
    int max_xp;
    unsigned areas;         // NETHACK_AREA_* bits
    float ret;
    // per-term reward ledgers: raw sum, and clip-attributed (each step's
    // terms scaled by clamp(sum)/sum, mirroring the trainer's [-1,1] clamp)
    float r_raw[6], r_clip[6];   // exp gold descent floor xp scout
    float r_death;
    int length;
    // per-level first-visit bitmaps; branch levels sharing a depth share one
    unsigned char visited[NETHACK_MAX_DEPTH][(NH_GRID + 7) / 8];
    unsigned short visited_key[NETHACK_MAX_DEPTH];   // dnum << 8 | dlevel per slot
    int n_visited_floors;
} Stats;
