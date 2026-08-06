#line 76 "encounter_colosseum.h"

#define COLO_ARENA_MIN_X    0
#define COLO_ARENA_MAX_X    33
#define COLO_ARENA_MIN_Y    0
#define COLO_ARENA_MAX_Y    33
#define COLO_ARENA_WIDTH    (COLO_ARENA_MAX_X - COLO_ARENA_MIN_X + 1)
#define COLO_ARENA_HEIGHT   (COLO_ARENA_MAX_Y - COLO_ARENA_MIN_Y + 1)

#define COLO_PLAYER_START_X 7
#define COLO_PLAYER_START_Y 18

#define COLO_MAX_TICKS      9000
#define COLO_SPEEDRUN_MAX_TICKS 4000
#define COLO_BEGINNER_MAX_TICKS 6000
#define COLO_NUM_WAVES      12
#define COLO_WAVE_BOSS      (COLO_NUM_WAVES - 1)

#define COLO_WAVE_SPAWN_DELAY_TICKS 5

#define COLO_BOSS_SPAWN_DELAY_TICKS 11

static inline int col_wave_entry_delay_ticks(int target_wave) {
    return target_wave == COLO_WAVE_BOSS
        ? COLO_BOSS_SPAWN_DELAY_TICKS : COLO_WAVE_SPAWN_DELAY_TICKS;
}
#define COLO_WAVE_READY_TICKS 2
#define COLO_WAVE_ATTACK_GATE_TICKS 4
#define COLO_REINFORCEMENT_TICKS 66

#define COLO_FARM_CAP_WAVES 4
#define COLO_NUM_ACTION_HEADS 20

#define COLO_MAX_NPCS 24
#define COLO_OBS_NPCS COLO_MAX_NPCS

typedef enum {
    COLO_FORECAST_BEHAVIOR_STANDARD = 0,
    COLO_FORECAST_BEHAVIOR_WARBAND,
    COLO_FORECAST_BEHAVIOR_MANTICORE,
    COLO_FORECAST_BEHAVIOR_SOL,
} ColoForecastBehaviorKind;

typedef struct {
    int active;
    ColoNpcType type;
    ColoForecastBehaviorKind forecast_behavior;
    int x;
    int y;
    int size;
    int hp;
    int attack_range;
    int attack_speed;
    int can_move;
    AttackStyle default_style;
    int attack_timer;
    int stun_timer;
    int frozen_ticks;
    int death_ticks;
    int cycle_step;
    AttackStyle orb_style[3];
    AttackStyle fixed_orb_style[3];
    int formation_dir;
    int javelin_attack_count;
    int sol_attack_delay;
    int sol_immobile_ticks;
} ColoForecastNpcLocal;

typedef struct { uint8_t lo, hi; } ColoWallSpan;
#define COLO_WALL_SPANS_PER_ROW 3
static const ColoWallSpan COLO_WALL_SPANS[COLO_ARENA_HEIGHT][COLO_WALL_SPANS_PER_ROW] = {
    [0]  = { {0,13}, {15,19}, {21,34} },
    [1]  = { {0,9},  {25,34} },
    [2]  = { {0,7},  {27,34} },
    [3]  = { {0,5},  {29,34} },
    [4]  = { {0,5},  {29,34} },
    [5]  = { {0,3},  {31,34} },
    [6]  = { {0,3},  {31,34} },
    [7]  = { {0,2},  {32,34} },
    [8]  = { {0,2},  {32,34} },
    [9]  = { {0,1},  {33,34} },
    [10] = { {0,1},  {33,34} },
    [11] = { {0,1},  {32,34} },
    [12] = { {0,1},  {31,34} },
    [13] = { {31,34} },
    [14] = { {31,34} },
    [15] = { {0,1},  {31,34} },
    [16] = { {0,1},  {31,34} },
    [17] = { {0,1},  {31,34} },
    [18] = { {0,1},  {31,34} },
    [19] = { {31,34} },
    [20] = { {31,34} },
    [21] = { {0,1},  {31,34} },
    [22] = { {0,1},  {32,34} },
    [23] = { {0,1},  {33,34} },
    [24] = { {0,1},  {33,34} },
    [25] = { {0,2},  {32,34} },
    [26] = { {0,2},  {32,34} },
    [27] = { {0,3},  {31,34} },
    [28] = { {0,3},  {31,34} },
    [29] = { {0,5},  {29,34} },
    [30] = { {0,6},  {29,34} },
    [31] = { {0,7},  {27,34} },
    [32] = { {0,9},  {25,34} },
    [33] = { {0,13}, {15,19}, {21,34} },
};

#define COLO_NUM_PILLARS  4
#define COLO_PILLAR_SIZE  3
static const int COLO_PILLARS[COLO_NUM_PILLARS][2] = {
    {8, 8}, {23, 8}, {8, 23}, {23, 23},
};

static uint8_t COLO_STATIC_BLOCKED[COLO_ARENA_WIDTH][COLO_ARENA_HEIGHT];

#define COLO_ARENA_TILE_COUNT (COLO_ARENA_WIDTH * COLO_ARENA_HEIGHT)
#define COLO_STATIC_LOS_TABLE_BITS (COLO_ARENA_TILE_COUNT * COLO_ARENA_TILE_COUNT)
#define COLO_STATIC_LOS_TABLE_WORDS ((COLO_STATIC_LOS_TABLE_BITS + 63) / 64)
#define COLO_STATIC_FOOTPRINT_MAX_SIZE 5
#define COLO_STATIC_FOOTPRINT_PAD (COLO_STATIC_FOOTPRINT_MAX_SIZE - 1)
#define COLO_STATIC_FOOTPRINT_MIN_X (COLO_ARENA_MIN_X - COLO_STATIC_FOOTPRINT_PAD)
#define COLO_STATIC_FOOTPRINT_MIN_Y (COLO_ARENA_MIN_Y - COLO_STATIC_FOOTPRINT_PAD)
#define COLO_STATIC_FOOTPRINT_MAX_X (COLO_ARENA_MAX_X + 1)
#define COLO_STATIC_FOOTPRINT_MAX_Y (COLO_ARENA_MAX_Y + 1)
#define COLO_STATIC_FOOTPRINT_WIDTH \
    (COLO_STATIC_FOOTPRINT_MAX_X - COLO_STATIC_FOOTPRINT_MIN_X + 1)
#define COLO_STATIC_FOOTPRINT_HEIGHT \
    (COLO_STATIC_FOOTPRINT_MAX_Y - COLO_STATIC_FOOTPRINT_MIN_Y + 1)
static uint64_t COLO_STATIC_LOS_BITS[COLO_STATIC_LOS_TABLE_WORDS];
static uint8_t COLO_STATIC_FOOTPRINT_BLOCKED[COLO_STATIC_FOOTPRINT_MAX_SIZE]
    [COLO_STATIC_FOOTPRINT_WIDTH][COLO_STATIC_FOOTPRINT_HEIGHT];
static int COLO_STATIC_LOOKUP_TABLES_BUILT = 0;
static int COLO_STATIC_ARENA_BUILT = 0;
static int COLO_NPC_STATS_BUILT = 0;

static void col_build_static_lookup_tables_once(void);

/* Derived purely from compile-time tables, so build it once. Rebuilding per reset
   zeroed a grid that every other env thread paths against, which handed them an
   arena with no walls for the width of the rebuild. puf_init runs serially, so the
   single build lands before any worker thread exists. */
static void col_build_static_arena(void) {
    if (COLO_STATIC_ARENA_BUILT) return;
    COLO_STATIC_ARENA_BUILT = 1;
    memset(COLO_STATIC_BLOCKED, 0, sizeof(COLO_STATIC_BLOCKED));
    for (int y = 0; y < COLO_ARENA_HEIGHT; y++) {
        for (int sp = 0; sp < COLO_WALL_SPANS_PER_ROW; sp++) {
            ColoWallSpan span = COLO_WALL_SPANS[y][sp];
            for (int x = span.lo; x < span.hi; x++) COLO_STATIC_BLOCKED[x][y] = 1;
        }
    }
    for (int p = 0; p < COLO_NUM_PILLARS; p++)
        for (int dx = 0; dx < COLO_PILLAR_SIZE; dx++)
            for (int dy = 0; dy < COLO_PILLAR_SIZE; dy++)
                COLO_STATIC_BLOCKED[COLO_PILLARS[p][0] + dx][COLO_PILLARS[p][1] + dy] = 1;
    col_build_static_lookup_tables_once();
}

static inline int col_static_blocked(int x, int y) {
    if (x < COLO_ARENA_MIN_X || x > COLO_ARENA_MAX_X ||
        y < COLO_ARENA_MIN_Y || y > COLO_ARENA_MAX_Y) return 1;
    return COLO_STATIC_BLOCKED[x][y];
}

static inline int col_static_los_endpoint_in_bounds(int x, int y) {
    return x >= COLO_ARENA_MIN_X && x <= COLO_ARENA_MAX_X &&
           y >= COLO_ARENA_MIN_Y && y <= COLO_ARENA_MAX_Y;
}

static inline uint32_t col_static_los_index(int x0, int y0, int x1, int y1) {
    uint32_t a = (uint32_t)((x0 - COLO_ARENA_MIN_X) * COLO_ARENA_HEIGHT +
        (y0 - COLO_ARENA_MIN_Y));
    uint32_t b = (uint32_t)((x1 - COLO_ARENA_MIN_X) * COLO_ARENA_HEIGHT +
        (y1 - COLO_ARENA_MIN_Y));
    return a * COLO_ARENA_TILE_COUNT + b;
}

static inline void col_static_los_set(uint32_t idx) {
    COLO_STATIC_LOS_BITS[idx >> 6] |= 1ULL << (idx & 63);
}

static inline int col_static_los_get(int x0, int y0, int x1, int y1) {
    uint32_t idx = col_static_los_index(x0, y0, x1, y1);
    return (int)((COLO_STATIC_LOS_BITS[idx >> 6] >> (idx & 63)) & 1ULL);
}

static inline int col_static_footprint_anchor_in_table(int x, int y) {
    return x >= COLO_STATIC_FOOTPRINT_MIN_X && x <= COLO_STATIC_FOOTPRINT_MAX_X &&
           y >= COLO_STATIC_FOOTPRINT_MIN_Y && y <= COLO_STATIC_FOOTPRINT_MAX_Y;
}

static int col_static_footprint_blocked_reference_slowpath(int x, int y, int size) {
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            if (col_static_blocked(x + dx, y + dy)) return 1;
        }
    }
    return 0;
}

static inline int col_static_footprint_blocked_lookup(int x, int y, int size) {
    if (size < 1 || size > COLO_STATIC_FOOTPRINT_MAX_SIZE)
        return col_static_footprint_blocked_reference_slowpath(x, y, size);
    if (!COLO_STATIC_LOOKUP_TABLES_BUILT) col_build_static_arena();
    if (!col_static_footprint_anchor_in_table(x, y)) return 1;
    return COLO_STATIC_FOOTPRINT_BLOCKED[size - 1]
        [x - COLO_STATIC_FOOTPRINT_MIN_X]
        [y - COLO_STATIC_FOOTPRINT_MIN_Y];
}

static int col_los_tile_blocked_reference(void* ctx, int x, int y) {
    (void)ctx;
    return col_static_blocked(x, y);
}

static int col_tiles_have_los_reference_slowpath(int x0, int y0, int x1, int y1) {
    OsrsLosQuery query = osrs_los_tile(col_los_tile_blocked_reference, NULL);
    return osrs_los_tile_ray_clear(&query, x0, y0, x1, y1);
}

static void col_build_static_lookup_tables_once(void) {
    if (COLO_STATIC_LOOKUP_TABLES_BUILT) return;
    for (int x0 = COLO_ARENA_MIN_X; x0 <= COLO_ARENA_MAX_X; x0++) {
        for (int y0 = COLO_ARENA_MIN_Y; y0 <= COLO_ARENA_MAX_Y; y0++) {
            for (int x1 = COLO_ARENA_MIN_X; x1 <= COLO_ARENA_MAX_X; x1++) {
                for (int y1 = COLO_ARENA_MIN_Y; y1 <= COLO_ARENA_MAX_Y; y1++) {
                    if (col_tiles_have_los_reference_slowpath(x0, y0, x1, y1))
                        col_static_los_set(col_static_los_index(x0, y0, x1, y1));
                }
            }
        }
    }
    for (int size = 1; size <= COLO_STATIC_FOOTPRINT_MAX_SIZE; size++) {
        for (int x = COLO_STATIC_FOOTPRINT_MIN_X; x <= COLO_STATIC_FOOTPRINT_MAX_X; x++) {
            for (int y = COLO_STATIC_FOOTPRINT_MIN_Y; y <= COLO_STATIC_FOOTPRINT_MAX_Y; y++) {
                COLO_STATIC_FOOTPRINT_BLOCKED[size - 1]
                    [x - COLO_STATIC_FOOTPRINT_MIN_X]
                    [y - COLO_STATIC_FOOTPRINT_MIN_Y] =
                    (uint8_t)col_static_footprint_blocked_reference_slowpath(x, y, size);
            }
        }
    }
    COLO_STATIC_LOOKUP_TABLES_BUILT = 1;
}

#define COLO_NUM_SPAWN_ANCHORS 13
#define COLO_SPAWN_ZONE_SIZE   3
static const int COLO_SPAWN_ANCHORS[COLO_NUM_SPAWN_ANCHORS][2] = {
    {3, 14}, {9, 16}, {3, 19}, {13, 19}, {16, 20}, {19, 19}, {17, 24},
    {13, 13}, {19, 13}, {16, 9}, {24, 17}, {28, 19}, {28, 14},
};

#define COLO_SPAWN_EXCLUSION_CHEB 4

#define COLO_WARBAND_BOX_MIN_X 12
#define COLO_WARBAND_BOX_MIN_Y 15
#define COLO_WARBAND_BOX_MAX_X 18
#define COLO_WARBAND_BOX_MAX_Y 21

#define COLO_WARBAND_CYCLE_TICKS 6

#define COLO_WARBAND_TILES_PER_TICK 2

typedef enum {
    COLO_WARBAND_FORM_NORTH = 0,
    COLO_WARBAND_FORM_EAST,
    COLO_WARBAND_FORM_WEST,
    COLO_WARBAND_FORM_SOUTH,
} ColoWarbandFormation;

static const int COLO_WARBAND_FORM_OFFSET[4][2] = {
    [COLO_WARBAND_FORM_NORTH] = { 0, 1 },
    [COLO_WARBAND_FORM_EAST]  = { 1, 0 },
    [COLO_WARBAND_FORM_WEST]  = { -1, 0 },
    [COLO_WARBAND_FORM_SOUTH] = { 0, -1 },
};

#define COLO_GATE_MIN_X            15
#define COLO_GATE_MAX_X            18
#define COLO_GATE_LINE_NORTH_MIN_Y 16
#define COLO_GATE_NORTH_SPAWN_ROW  32
#define COLO_GATE_SOUTH_SPAWN_ROW  1

#define COLO_SOL_SPAWN_X 16
#define COLO_SOL_SPAWN_Y 19

#define COLO_BOSS_ARENA_MIN_X 9
#define COLO_BOSS_ARENA_MIN_Y 9
#define COLO_BOSS_ARENA_MAX_X 24
#define COLO_BOSS_ARENA_MAX_Y 24
#define COLO_BOSS_ARENA_SIZE  (COLO_BOSS_ARENA_MAX_X - COLO_BOSS_ARENA_MIN_X + 1)

#define COLO_BOSS_PLAYER_START_X 16
#define COLO_BOSS_PLAYER_START_Y 10

static const int COLO_NPC_DEF_IDS[COLO_NUM_NPC_TYPES] = {
    [COLO_FREMENNIK_BERSERKER] = 12816,
    [COLO_FREMENNIK_ARCHER]    = 12814,
    [COLO_FREMENNIK_SEER]      = 12815,
    [COLO_SERPENT_SHAMAN]      = 12811,
    [COLO_JAGUAR_WARRIOR]      = 12810,
    [COLO_JAVELIN_COLOSSUS]    = 12817,
    [COLO_SHOCKWAVE_COLOSSUS]  = 12819,
    [COLO_MINOTAUR]            = 12812,
    [COLO_MANTICORE]           = 12818,
    [COLO_SOL_HEREDIT]         = 12821,
    [COLO_HEALING_TOTEM]       = 12825,
    [COLO_BEE_SWARM]           = 12823,
};

#define COLO_NPC_DEF_ID_MINOTAUR_ROUTEFIND 12813

#define COLO_NPC_DEF_ID_THRALL 10880

#define COLO_NPC_SLOT_THRALL    COLO_MAX_NPCS
#define COLO_THRALL_RENDER_INSTANCE_ID 0xC01055EEu

#define COLO_NPC_DEF_ID_SOLARFLARE 12826
#define COLO_SOLARFLARE_IDLE_ANIM_ID 10817
#define COLO_NPC_SLOT_SOLARFLARE    (COLO_MAX_NPCS + 1)
#define COLO_SOLARFLARE_RENDER_INSTANCE_ID 0xC0105F1Au

#define COLO_NPC_DEF_ID_BEAM_CRYSTAL 12824
#define COLO_BEAM_CRYSTAL_IDLE_ANIM_ID 10799
#define COLO_NPC_SLOT_BEAM_CRYSTAL  (COLO_MAX_NPCS + 1 + COLO_NUM_PILLARS)
#define COLO_BEAM_CRYSTAL_RENDER_INSTANCE_ID 0xC0105EC5u

static int col_type_is_hazard_entity(ColoNpcType type) {
    return type == COLO_HEALING_TOTEM || type == COLO_BEE_SWARM;
}

static const char* colo_npc_type_name(int type) {
    switch (type) {
        case COLO_FREMENNIK_BERSERKER: return "Fremennik Berserker";
        case COLO_FREMENNIK_ARCHER:    return "Fremennik Archer";
        case COLO_FREMENNIK_SEER:      return "Fremennik Seer";
        case COLO_SERPENT_SHAMAN:      return "Serpent Shaman";
        case COLO_JAGUAR_WARRIOR:      return "Jaguar Warrior";
        case COLO_JAVELIN_COLOSSUS:    return "Javelin Colossus";
        case COLO_SHOCKWAVE_COLOSSUS:  return "Shockwave Colossus";
        case COLO_MINOTAUR:            return "Minotaur";
        case COLO_MANTICORE:           return "Manticore";
        case COLO_SOL_HEREDIT:         return "Sol Heredit";
        case COLO_HEALING_TOTEM:       return "Healing totem";
        case COLO_BEE_SWARM:           return "Bee swarm";
        default: return "unknown";
    }
}

typedef struct {
    int hp;
    int attack_speed;
    int attack_range;
    int size;
    AttackStyle default_style;
    MeleeStyle melee_style;
    int can_melee;

    int att_level, str_level, def_level, range_level, magic_level;
    int melee_att_bonus, range_att_bonus, magic_att_bonus;
    int melee_str_bonus, ranged_str_bonus;
    int stab_def, slash_def, crush_def, magic_def_bonus, ranged_def_bonus;

    int max_hit;
    AttackStyle player_style_that_max_hits;
    int can_move;
} ColoNpcStats;

typedef struct {
    int attack_range;
    AttackStyle default_style;
    MeleeStyle melee_style;
    int can_melee;
    AttackStyle player_style_that_max_hits;
    int can_move;
} ColoNpcOverlay;

static const MonsterStats COLO_NPC_BASE[COLO_NUM_NPC_TYPES] = {
    [COLO_FREMENNIK_BERSERKER] = { .npc_id=12816, .name="Fremennik berserker",
        .hp=48, .att_level=110,.str_level=110,.def_level=80,.magic_level=110,.range_level=110,
        .attack_speed=6,.size=1,.max_hit=29,
        .melee_att_bonus=150,.melee_str_bonus=90,.magic_att_bonus=0,.magic_str_bonus=0,.range_att_bonus=0,.ranged_str_bonus=0,
        .stab_def=50,.slash_def=50,.crush_def=50,.magic_def=0,.ranged_def=75 },
    [COLO_FREMENNIK_ARCHER] = { .npc_id=12814, .name="Fremennik archer",
        .hp=50, .att_level=110,.str_level=110,.def_level=80,.magic_level=110,.range_level=110,
        .attack_speed=6,.size=1,.max_hit=14,
        .melee_att_bonus=150,.melee_str_bonus=0,.magic_att_bonus=0,.magic_str_bonus=0,.range_att_bonus=150,.ranged_str_bonus=10,
        .stab_def=0,.slash_def=0,.crush_def=0,.magic_def=75,.ranged_def=50 },
    [COLO_FREMENNIK_SEER] = { .npc_id=12815, .name="Fremennik seer",
        .hp=50, .att_level=110,.str_level=110,.def_level=80,.magic_level=110,.range_level=110,
        .attack_speed=6,.size=1,.max_hit=12,
        .melee_att_bonus=0,.melee_str_bonus=0,.magic_att_bonus=150,.magic_str_bonus=0,.range_att_bonus=0,.ranged_str_bonus=0,
        .stab_def=50,.slash_def=50,.crush_def=50,.magic_def=30,.ranged_def=0 },
    [COLO_SERPENT_SHAMAN] = { .npc_id=12811, .name="Serpent shaman",
        .hp=125, .att_level=100,.str_level=90,.def_level=90,.magic_level=220,.range_level=160,
        .attack_speed=5,.size=1,.max_hit=28,
        .melee_att_bonus=0,.melee_str_bonus=0,.magic_att_bonus=50,.magic_str_bonus=15,.range_att_bonus=0,.ranged_str_bonus=0,
        .stab_def=30,.slash_def=30,.crush_def=30,.magic_def=15,.ranged_def=50 },
    [COLO_JAGUAR_WARRIOR] = { .npc_id=12810, .name="Jaguar warrior",
        .hp=125, .att_level=200,.str_level=330,.def_level=125,.magic_level=100,.range_level=160,
        .attack_speed=5,.size=2,.max_hit=47,
        .melee_att_bonus=0,.melee_str_bonus=25,.magic_att_bonus=0,.magic_str_bonus=0,.range_att_bonus=0,.ranged_str_bonus=0,
        .stab_def=30,.slash_def=30,.crush_def=30,.magic_def=15,.ranged_def=45 },
    [COLO_JAVELIN_COLOSSUS] = { .npc_id=12817, .name="Javelin Colossus",
        .hp=225, .att_level=200,.str_level=300,.def_level=190,.magic_level=225,.range_level=360,
        .attack_speed=5,.size=3,.max_hit=48,
        .melee_att_bonus=0,.melee_str_bonus=0,.magic_att_bonus=0,.magic_str_bonus=0,.range_att_bonus=25,.ranged_str_bonus=20,
        .stab_def=15,.slash_def=15,.crush_def=15,.magic_def=20,.ranged_def=30 },
    [COLO_SHOCKWAVE_COLOSSUS] = { .npc_id=12819, .name="Shockwave Colossus",
        .hp=125, .att_level=120,.str_level=190,.def_level=150,.magic_level=350,.range_level=220,
        .attack_speed=5,.size=3,.max_hit=56,
        .melee_att_bonus=0,.melee_str_bonus=0,.magic_att_bonus=55,.magic_str_bonus=35,.range_att_bonus=0,.ranged_str_bonus=0,
        .stab_def=15,.slash_def=15,.crush_def=15,.magic_def=5,.ranged_def=35 },
    [COLO_MINOTAUR] = { .npc_id=12812, .name="Minotaur",
        .hp=225, .att_level=300,.str_level=360,.def_level=190,.magic_level=250,.range_level=120,
        .attack_speed=5,.size=3,.max_hit=74,
        .melee_att_bonus=15,.melee_str_bonus=64,.magic_att_bonus=0,.magic_str_bonus=0,.range_att_bonus=0,.ranged_str_bonus=0,
        .stab_def=0,.slash_def=0,.crush_def=0,.magic_def=0,.ranged_def=12 },
    [COLO_MANTICORE] = { .npc_id=12818, .name="Manticore",
        .hp=250, .att_level=300,.str_level=300,.def_level=250,.magic_level=300,.range_level=350,
        .attack_speed=10,.size=3,.max_hit=36,
        .melee_att_bonus=0,.melee_str_bonus=0,.magic_att_bonus=0,.magic_str_bonus=0,.range_att_bonus=0,.ranged_str_bonus=0,
        .stab_def=0,.slash_def=0,.crush_def=0,.magic_def=10,.ranged_def=25 },
    [COLO_SOL_HEREDIT] = { .npc_id=12821, .name="Sol Heredit",
        .hp=1500, .att_level=350,.str_level=400,.def_level=200,.magic_level=300,.range_level=350,
        .attack_speed=5,.size=5,.max_hit=44,
        .melee_att_bonus=250,.melee_str_bonus=5,.magic_att_bonus=0,.magic_str_bonus=0,.range_att_bonus=150,.ranged_str_bonus=5,
        .stab_def=65,.slash_def=5,.crush_def=30,.magic_def=750,.ranged_def=825 },

    [COLO_HEALING_TOTEM] = { .npc_id=12825, .name="Healing totem",
        .hp=1, .att_level=1,.str_level=1,.def_level=1,.magic_level=1,.range_level=1,
        .attack_speed=0,.size=1,.max_hit=0 },
    [COLO_BEE_SWARM] = { .npc_id=12823, .name="Bee swarm",
        .hp=1, .att_level=1,.str_level=1,.def_level=1,.magic_level=1,.range_level=1,
        .attack_speed=0,.size=2,.max_hit=0 },
};

#define COLO_MANTICORE_MAX_HIT_RANGED 36
#define COLO_MANTICORE_MAX_HIT_MAGIC  31
#define COLO_MANTICORE_MAX_HIT_MELEE  31

static const ColoNpcOverlay COLO_NPC_OVERLAY[COLO_NUM_NPC_TYPES] = {
    [COLO_FREMENNIK_BERSERKER] = { .attack_range=1,  .default_style=ATTACK_STYLE_MELEE,  .melee_style=MELEE_STYLE_STAB,  .can_melee=1, .player_style_that_max_hits=ATTACK_STYLE_MAGIC,  .can_move=1 },
    [COLO_FREMENNIK_ARCHER]    = { .attack_range=1,  .default_style=ATTACK_STYLE_RANGED, .melee_style=MELEE_STYLE_STAB,  .can_melee=0, .player_style_that_max_hits=ATTACK_STYLE_MELEE,  .can_move=1 },
    [COLO_FREMENNIK_SEER]      = { .attack_range=1,  .default_style=ATTACK_STYLE_MAGIC,  .melee_style=MELEE_STYLE_STAB,  .can_melee=0, .player_style_that_max_hits=ATTACK_STYLE_RANGED, .can_move=1 },
    [COLO_SERPENT_SHAMAN]      = { .attack_range=10, .default_style=ATTACK_STYLE_MAGIC,  .melee_style=MELEE_STYLE_STAB,  .can_melee=0, .player_style_that_max_hits=ATTACK_STYLE_NONE,  .can_move=1 },
    [COLO_JAGUAR_WARRIOR]      = { .attack_range=1,  .default_style=ATTACK_STYLE_MELEE,  .melee_style=MELEE_STYLE_SLASH, .can_melee=1, .player_style_that_max_hits=ATTACK_STYLE_NONE,  .can_move=1 },
    [COLO_JAVELIN_COLOSSUS]    = { .attack_range=15, .default_style=ATTACK_STYLE_RANGED, .melee_style=MELEE_STYLE_STAB,  .can_melee=0, .player_style_that_max_hits=ATTACK_STYLE_NONE,  .can_move=1 },
    [COLO_SHOCKWAVE_COLOSSUS]  = { .attack_range=15, .default_style=ATTACK_STYLE_MAGIC,  .melee_style=MELEE_STYLE_STAB,  .can_melee=0, .player_style_that_max_hits=ATTACK_STYLE_NONE,  .can_move=1 },
    [COLO_MINOTAUR]            = { .attack_range=1,  .default_style=ATTACK_STYLE_MELEE,  .melee_style=MELEE_STYLE_CRUSH, .can_melee=1, .player_style_that_max_hits=ATTACK_STYLE_NONE,  .can_move=1 },
    [COLO_MANTICORE]           = { .attack_range=15, .default_style=ATTACK_STYLE_RANGED, .melee_style=MELEE_STYLE_STAB,  .can_melee=1, .player_style_that_max_hits=ATTACK_STYLE_NONE,  .can_move=1 },
    [COLO_SOL_HEREDIT]         = { .attack_range=1,  .default_style=ATTACK_STYLE_MELEE,  .melee_style=MELEE_STYLE_SLASH, .can_melee=1, .player_style_that_max_hits=ATTACK_STYLE_NONE,  .can_move=1 },
    [COLO_HEALING_TOTEM]       = { .attack_range=0,  .default_style=ATTACK_STYLE_NONE,   .melee_style=MELEE_STYLE_STAB,  .can_melee=0, .player_style_that_max_hits=ATTACK_STYLE_NONE,  .can_move=0 },
    [COLO_BEE_SWARM]           = { .attack_range=0,  .default_style=ATTACK_STYLE_NONE,   .melee_style=MELEE_STYLE_STAB,  .can_melee=0, .player_style_that_max_hits=ATTACK_STYLE_NONE,  .can_move=0 },
};

static ColoNpcStats COLO_NPC_STATS[COLO_NUM_NPC_TYPES];

static void col_build_npc_stats(void) {
    col_build_static_arena();
    if (COLO_NPC_STATS_BUILT) return;
    COLO_NPC_STATS_BUILT = 1;
    for (int i = 0; i < COLO_NUM_NPC_TYPES; i++) {
        const MonsterStats* m = &COLO_NPC_BASE[i];
        const ColoNpcOverlay* o = &COLO_NPC_OVERLAY[i];
        ColoNpcStats* s = &COLO_NPC_STATS[i];

        s->hp              = m->hp;
        s->attack_speed    = m->attack_speed;
        s->size            = m->size;
        s->att_level       = m->att_level;
        s->str_level       = m->str_level;
        s->def_level       = m->def_level;
        s->range_level     = m->range_level;
        s->magic_level     = m->magic_level;
        s->melee_att_bonus = m->melee_att_bonus;
        s->range_att_bonus = m->range_att_bonus;
        s->magic_att_bonus = m->magic_att_bonus;
        s->melee_str_bonus = m->melee_str_bonus;
        s->ranged_str_bonus = m->ranged_str_bonus;
        s->stab_def        = m->stab_def;
        s->slash_def       = m->slash_def;
        s->crush_def       = m->crush_def;
        s->magic_def_bonus = m->magic_def;
        s->ranged_def_bonus = m->ranged_def;
        s->max_hit         = m->max_hit;

        s->attack_range    = o->attack_range;
        s->default_style   = o->default_style;
        s->melee_style     = o->melee_style;
        s->can_melee       = o->can_melee;
        s->player_style_that_max_hits = o->player_style_that_max_hits;
        s->can_move        = o->can_move;
    }
}

#define COLO_MAX_NPCS_PER_WAVE 10
#define COLO_MAX_REINFORCE_PER_WAVE 2

typedef struct {
    uint8_t types[COLO_MAX_NPCS_PER_WAVE];
    int count;
    uint8_t reinforce[COLO_MAX_REINFORCE_PER_WAVE];
    int reinforce_count;
} ColoWaveDef;

static const ColoWaveDef COLO_WAVES[COLO_NUM_WAVES] = {
    #define BZ COLO_FREMENNIK_BERSERKER
    #define AR COLO_FREMENNIK_ARCHER
    #define SE COLO_FREMENNIK_SEER
    #define SH COLO_SERPENT_SHAMAN
    #define JG COLO_JAGUAR_WARRIOR
    #define JV COLO_JAVELIN_COLOSSUS
    #define SW COLO_SHOCKWAVE_COLOSSUS
    #define MT COLO_MINOTAUR
    #define MC COLO_MANTICORE
    #define SOL COLO_SOL_HEREDIT
    #define W(start, reinf) { \
        .types = { COLO_WAVE_UNPACK start }, \
        .count = sizeof((uint8_t[]){ COLO_WAVE_UNPACK start }), \
        .reinforce = { COLO_WAVE_UNPACK reinf }, \
        .reinforce_count = sizeof((uint8_t[]){ COLO_WAVE_UNPACK reinf }) }
    #define COLO_WAVE_UNPACK(...) __VA_ARGS__

    [0]  = W((BZ, AR, SE, SH),              (JG)),
    [1]  = W((BZ, AR, SE, SH, JV),          (JG)),
    [2]  = W((BZ, AR, SE, SH, JV, JV),      (JG)),
    [3]  = W((BZ, AR, SE, SH, MC),          (JG, SH)),
    [4]  = W((BZ, AR, SE, SH, JV, MC),      (JG, SH)),
    [5]  = W((BZ, AR, SE, SH, JV, JV, MC),  (JG, SH)),
    [6]  = W((BZ, AR, SE, JV, MC, SW),      (MT)),
    [7]  = W((BZ, AR, SE, JV, JV, MC, SW),  (MT)),
    [8]  = W((BZ, AR, SE, JV, MC, MC),      (MT)),
    [9]  = W((BZ, AR, SE, JV, JV, MC, MC),  (MT, SH)),
    [10] = W((BZ, AR, SE, JV, MC, MC, SW),  (MT, SH)),
    [11] = W((SOL),                         ()),

    #undef BZ
    #undef AR
    #undef SE
    #undef SH
    #undef JG
    #undef JV
    #undef SW
    #undef MT
    #undef MC
    #undef SOL
    #undef W
    #undef COLO_WAVE_UNPACK
};

#define COLO_NUM_MODIFIERS 16
#define COLO_MODIFIER_DRAFT_OPTIONS 3

#define COLO_RFDD_FIRST_DRAFT_WAVE 6

#define COLO_DRAFT_UPGRADE_WEIGHT 2

static const uint8_t COLO_MODIFIER_MAX_TIER[COLO_NUM_REAL_MODIFIERS] = {
    [COLO_MOD_BEES]        = 3,
    [COLO_MOD_BLASPHEMY]   = 3,
    [COLO_MOD_DOOM]        = 3,
    [COLO_MOD_DYNAMIC_DUO] = 1,
    [COLO_MOD_FRAILTY]     = 3,
    [COLO_MOD_MANTIMAYHEM] = 3,
    [COLO_MOD_MYOPIA]      = 3,
    [COLO_MOD_REENTRY]     = 3,
    [COLO_MOD_RED_FLAG]    = 1,
    [COLO_MOD_RELENTLESS]  = 3,
    [COLO_MOD_SOLARFLARE]  = 3,
    [COLO_MOD_QUARTET]     = 1,
    [COLO_MOD_TOTEMIC]     = 1,
    [COLO_MOD_VOLATILITY]  = 3,
};

static const uint8_t COLO_MODIFIER_PRE_BOSS_ONLY[COLO_NUM_REAL_MODIFIERS] = {
    0, 0, 0, 1,
    0, 1, 0, 1,
    1, 0, 0, 0,
    0, 0,
};

typedef struct {
    uint32_t active_mask;
    uint8_t tier[COLO_NUM_MODIFIERS];
    int draft_options[COLO_MODIFIER_DRAFT_OPTIONS];
    int draft_pending;
    int draft_gates_spawn;
    int draft_free_movement;

} ColoModifierState;

typedef enum {
    COLO_HAZARD_NONE = 0,
    COLO_HAZARD_ALIVE,
    COLO_HAZARD_RESPAWNING,
} ColoHazardPhase;

#define COLO_MAX_BEE_SWARMS 3
#define COLO_BEE_MOVE_INTERVAL 12

#define COLO_BEE_MIN_DAMAGE 15
#define COLO_BEE_MAX_DAMAGE 20
#define COLO_BEE_RESPAWN_TICKS 50

typedef struct {
    ColoHazardPhase phase;
    int npc_slot;
    int respawn_timer;
    int move_timer;
} ColoBeeSwarm;

#define COLO_MAX_TOTEMS COLO_MAX_NPCS
#define COLO_TOTEM_HEAL_PCT 30
#define COLO_TOTEM_HEAL_INTERVAL 7
#define COLO_TOTEM_SPAWN_HEAL_DELAY 5
#define COLO_TOTEM_PROJECTILE_TICKS 2
#define COLO_TOTEM_RESPAWN_TICKS 200
#define COLO_TOTEM_SOL_HEAL 75
#define COLO_TOTEM_SOL_EXTRA_INTERVAL 200
#define COLO_TOTEM_OBS_COUNT_NORM 4

typedef struct {
    ColoHazardPhase phase;
    int owner_slot;
    int npc_slot;
    int respawn_timer;
    int heal_timer;
    int projectile_timer;
} ColoTotem;

#define COLO_SOLARFLARE_RING_STEPS       16
#define COLO_SOLARFLARE_MOVE_INTERVAL    2
#define COLO_SOLARFLARE_MOVE_INTERVAL_T3 1
#define COLO_SOLARFLARE_CORNER_PAUSE     7
#define COLO_SOLARFLARE_CORNER_PAUSE_T3  2
#define COLO_SOLARFLARE_MAX_HIT_T1 12
#define COLO_SOLARFLARE_MAX_HIT_T2 18
#define COLO_SOLARFLARE_MAX_HIT_T3 24

typedef struct {
    int active;
    int step;
    int move_timer;
    int pause_timer;
} ColoSolarflareOrb;

static const int COLO_DOOM_CAP[4] = { 0, 15, 10, 5 };

#define COLO_VOLATILITY_MAX_HIT 25

typedef enum {
    COLO_POOL_VOLATILITY = 1,
    COLO_POOL_REENTRY = 2,
    COLO_POOL_REENTRY_VOLATILITY = 3,
} ColoMoltenPoolKind;

typedef enum {
    COLO_POOL_TEMPORARY = 0,
    COLO_POOL_PERMANENT,
} ColoMoltenPoolLifetime;

static const int COLO_RELENTLESS_DEF_BYPASS_PCT[4] = { 0, 33, 66, 100 };
static const int COLO_RELENTLESS_MAX_HIT_BONUS[4]  = { 0, 1, 3, 6 };

static const int COLO_BLASPHEMY_DRAIN_PCT[4] = { 0, 20, 40, 60 };

static const int COLO_FRAILTY_HP_CUT_PCT[4] = { 0, 10, 20, 40 };

static const int COLO_MYOPIA_RANGE_CUT[4] = { 0, 2, 4, 6 };

#define COLO_VENOM_START 6
#define COLO_VENOM_STEP 2
#define COLO_VENOM_CAP 20
#define COLO_VENOM_INTERVAL 30

#define COLO_POISON_BEE_CONTACT_SEVERITY 5
#define COLO_POISON_INTERVAL 30

typedef enum {
    COLO_SOL_AOE_NONE = 0,
    COLO_SOL_AOE_SPEAR1,
    COLO_SOL_AOE_SPEAR2,
    COLO_SOL_AOE_SHIELD1,
    COLO_SOL_AOE_SHIELD2,
} ColoSolAoeAttack;

typedef enum {
    COLO_SOL_DAMAGE_SPEAR_1 = 0,
    COLO_SOL_DAMAGE_SPEAR_2,
    COLO_SOL_DAMAGE_SHIELD_1,
    COLO_SOL_DAMAGE_SHIELD_2,
    COLO_SOL_DAMAGE_TRIPLE_PARRY,
    COLO_SOL_DAMAGE_GRAPPLE,
    COLO_SOL_DAMAGE_CRYSTAL_LASER,
    COLO_SOL_DAMAGE_MOLTEN_SAND,
    COLO_NUM_SOL_DAMAGE_SOURCES,
} ColoSolDamageSource;

typedef enum {
    COLO_JAVELIN_DAMAGE_BASIC_RANGED = 0,
    COLO_JAVELIN_DAMAGE_SKYFALL,
    COLO_JAVELIN_DAMAGE_REENTRY_POOL,
    COLO_JAVELIN_DAMAGE_REENTRY_VOLATILITY_POOL,
    COLO_NUM_JAVELIN_DAMAGE_SOURCES,
} ColoJavelinDamageSource;

typedef enum {
    COLO_DAMAGE_NPC_ATTACK = 0,
    COLO_DAMAGE_JAVELIN_BASIC_RANGED,
    COLO_DAMAGE_MANTICORE_VENOM,
    COLO_DAMAGE_BEE_POISON,
    COLO_DAMAGE_BEE_CONTACT,
    COLO_DAMAGE_JAVELIN_SKYFALL,
    COLO_DAMAGE_REENTRY_POOL,
    COLO_DAMAGE_VOLATILITY_EXPLOSION,
    COLO_DAMAGE_VOLATILITY_POOL,
    COLO_DAMAGE_REENTRY_VOLATILITY_POOL,
    COLO_DAMAGE_SOLARFLARE,
    COLO_DAMAGE_SELF,
    COLO_DAMAGE_SOL_SPEAR_1,
    COLO_DAMAGE_SOL_SPEAR_2,
    COLO_DAMAGE_SOL_SHIELD_1,
    COLO_DAMAGE_SOL_SHIELD_2,
    COLO_DAMAGE_SOL_TRIPLE_PARRY,
    COLO_DAMAGE_SOL_GRAPPLE,
    COLO_DAMAGE_SOL_CRYSTAL_LASER,
    COLO_DAMAGE_SOL_MOLTEN_SAND,
    COLO_NUM_DAMAGE_SOURCES,
} ColoDamageSource;

typedef enum {
    COLO_SOL_ATTACK_NONE = 0,
    COLO_SOL_ATTACK_SPEAR,
    COLO_SOL_ATTACK_SHIELD,
    COLO_SOL_ATTACK_TRIPLE,
    COLO_SOL_ATTACK_GRAPPLE,
} ColoSolAttackKind;

typedef enum {
    COLO_GRAPPLE_SLOT_BODY = 0,
    COLO_GRAPPLE_SLOT_BACK,
    COLO_GRAPPLE_SLOT_HANDS,
    COLO_GRAPPLE_SLOT_LEGS,
    COLO_GRAPPLE_SLOT_FEET,
    COLO_NUM_GRAPPLE_SLOTS
} ColoGrappleSlot;

typedef enum {
    COLO_GRAPPLE_OUTCOME_NONE = 0,
    COLO_GRAPPLE_OUTCOME_DEFENDED,
    COLO_GRAPPLE_OUTCOME_PERFECT,
    COLO_GRAPPLE_OUTCOME_FAILED,
} ColoGrappleOutcome;

#define COLO_SOL_HAZARD_TILES_MAX 32
#define COLO_SOL_MAX_CRYSTALS 4
#define COLO_SOL_BEAM_MAX 32
#define COLO_SOL_PARRY_HITS 3
#define COLO_SOL_PRAYER_HISTORY 8
#define COLO_SOL_NUM_PHASES 6
#define COLO_SOL_HP_MAX 1500

typedef enum {
    COLO_SOL_EDGE_NORTH,
    COLO_SOL_EDGE_EAST,
    COLO_SOL_EDGE_SOUTH,
    COLO_SOL_EDGE_WEST,
} ColoSolCrystalEdge;

typedef struct {
    int active;
    int edge;
    int x, y;
    int dir;
    int move_timer;
    int firing_freeze;
} ColoSolCrystal;

typedef struct {
    int active;
    int x, y;
    int ticks_to_pool;
} ColoSolBeam;

typedef struct {
    int started;
    int phase;
    int boss_idx;

    int attack_delay;
    int immobile_ticks;
    int special_cooldown;
    int force_spear;
    int last_attack_kind;
    int last_variant;

    ColoSolAoeAttack aoe_attack;
    int aoe_age;
    int aoe_x, aoe_y;
    int aoe_dir_x, aoe_dir_y;

    int parry_land_in[COLO_SOL_PARRY_HITS];
    int parry_damage[COLO_SOL_PARRY_HITS];
    int parry_hits_left;
    uint8_t overhead_history[COLO_SOL_PRAYER_HISTORY];

    int grapple_active;
    int grapple_body_slot;
    int grapple_timer;
    int grapple_resolved;
    int grapple_outcome;
    int grapple_outcome_tick;
    int next_attack_guaranteed_max;
    int guaranteed_max_ticks;

    ColoSolCrystal crystals[COLO_SOL_MAX_CRYSTALS];
    int crystal_count;
    int laser_cooldown;

    ColoSolBeam beams[COLO_SOL_BEAM_MAX];
    int hazard_tile_x[COLO_SOL_HAZARD_TILES_MAX];
    int hazard_tile_y[COLO_SOL_HAZARD_TILES_MAX];
    int hazard_tile_count;
    int enrage_pool_timer;

    int boss_arena_min_x, boss_arena_min_y;
    int boss_arena_max_x, boss_arena_max_y;
} SolHereditState;

#define COLO_SOL_SPEAR_DELAY        7
#define COLO_SOL_SPEAR_DELAY_FAST   6
#define COLO_SOL_SHIELD_DELAY       6
#define COLO_SOL_SHIELD_DELAY_FAST  5
#define COLO_SOL_TRIPLE_DELAY       12
#define COLO_SOL_TRIPLE_DELAY_MID   11
#define COLO_SOL_GRAPPLE_DELAY      7
#define COLO_SOL_SPECIAL_COOLDOWN   2

#define COLO_SOL_SPEAR_FREEZE       6
#define COLO_SOL_SHIELD_FREEZE      4
#define COLO_SOL_GRAPPLE_FREEZE     5
#define COLO_SOL_TRANSITION_FREEZE  5

#define COLO_SOL_SLAM_MIN_HIT 20
#define COLO_SOL_SLAM_RAND    25

#define COLO_SOL_AOE_DAMAGE_AGE 2
#define COLO_SOL_AOE_LIFETIME   3

#define COLO_SOL_SPEAR_LINE_LEN 7
#define COLO_SOL_SHIELD1_RING 4
#define COLO_SOL_SHIELD2_RING 5

#define COLO_SOL_GRAPPLE_WINDOW    4
#define COLO_SOL_PERFECT_MAX_TICKS 5

#define COLO_SOL_TP_HIT_GAP        3
#define COLO_SOL_TP_HIT3_GAP_LOW   4

#define COLO_SOL_CRYSTAL_MOVE_TICKS_SPAWN 4
#define COLO_SOL_CRYSTAL_MOVE_TICKS       2
#define COLO_SOL_CRYSTAL_COOLDOWN_MIN     25
#define COLO_SOL_CRYSTAL_COOLDOWN_RAND    10
#define COLO_SOL_CRYSTAL_COOLDOWN_ENRAGE  12
#define COLO_SOL_LASER_FREEZE             9
#define COLO_SOL_LASER_BEAM_SHOW_MAX      6
#define COLO_SOL_LASER_BEAM_SHOW_MIN      2
#define COLO_SOL_LASER_DAMAGE_FREEZE      3
#define COLO_SOL_LASER_MIN_HIT            60
#define COLO_SOL_LASER_RAND               16

#define COLO_SOL_BEAM_COUNT         6
#define COLO_SOL_BEAM_TO_POOL_TICKS 3
#define COLO_SOL_BEAM_SPREAD        4
#define COLO_SOL_ENRAGE_OPEN_POOLS  5
#define COLO_SOL_ENRAGE_POOL_SPREAD 4
#define COLO_SOL_MOLTEN_INTERVAL    3

#define COLO_MOLTEN_SAND_MIN_HIT 6
#define COLO_MOLTEN_SAND_RAND    3

static int col_type_is_warbander(ColoNpcType type) {
    return type == COLO_FREMENNIK_BERSERKER || type == COLO_FREMENNIK_ARCHER ||
           type == COLO_FREMENNIK_SEER;
}

typedef struct {
    int formation_dir;
} ColoWarbandState;

typedef struct {
    int cycle_step;
    AttackStyle orb_style[3];
    AttackStyle fixed_orb_style[3];
} ColoManticoreState;

typedef struct {
    int attack_count;
    int skyfall_pending;
    int skyfall_tile_x, skyfall_tile_y;
    int skyfall_timer;
    int skyfall_damage;
} ColoJavelinState;

typedef struct {
    ColoNpcType type;
    int x, y;
    int hp, max_hp;
    int size;
    uint32_t render_id;
    int attack_timer;
    AttackStyle attack_style;
    int active;
    int target_x, target_y;
    int stun_timer;
    int frozen_ticks;
    int def_drained;
    int min_hp_seen;

    union {
        ColoManticoreState manticore;
        ColoJavelinState javelin;
        ColoWarbandState warband;
    } type_state;

    EncounterPendingHitQueue pending_hits;

    int death_ticks;
    int spawned_as_reinforcement;

    int attacked_this_tick;
    AttackStyle attack_style_this_tick;
    int moved_this_tick;
    int moved_last_tick;

    int hit_landed_this_tick;
    int hit_damage;
    int hit_was_successful_this_tick;
} ColoNPC;

static inline ColoManticoreState* colo_npc_manticore(ColoNPC* npc) {
    assert(npc->type == COLO_MANTICORE);
    return &npc->type_state.manticore;
}

static inline ColoJavelinState* colo_npc_javelin(ColoNPC* npc) {
    assert(npc->type == COLO_JAVELIN_COLOSSUS);
    return &npc->type_state.javelin;
}

static inline ColoWarbandState* colo_npc_warband(ColoNPC* npc) {
    assert(col_type_is_warbander(npc->type));
    return &npc->type_state.warband;
}

static inline int col_warband_window_offset(ColoNpcType type) {
    switch (type) {
        case COLO_FREMENNIK_BERSERKER: return 1;
        case COLO_FREMENNIK_SEER:      return 2;
        case COLO_FREMENNIK_ARCHER:    return 3;
        default:
            assert(0 && "warband window offset queried for a non-warbander");
            return 0;
    }
}

#define COLO_JAVELIN_SKYFALL_DELAY 6
#define COLO_JAVELIN_SKYFALL_MAX_HIT 40
#define COLO_JAVELIN_SKYFALL_ANIM_ID 10893
#define COLO_MANTICORE_STAGGER_TICKS 5

#define COLO_MINOTAUR_HEAL_RANGE 7

static void colo_npc_init_type_state(ColoNPC* npc) {
    switch (npc->type) {
        case COLO_FREMENNIK_BERSERKER:
            npc->type_state.warband = (ColoWarbandState){ .formation_dir = COLO_WARBAND_FORM_NORTH };
            break;
        case COLO_FREMENNIK_SEER:
            npc->type_state.warband = (ColoWarbandState){ .formation_dir = COLO_WARBAND_FORM_EAST };
            break;
        case COLO_FREMENNIK_ARCHER:
            npc->type_state.warband = (ColoWarbandState){ .formation_dir = COLO_WARBAND_FORM_WEST };
            break;
        case COLO_MANTICORE:
            npc->type_state.manticore = (ColoManticoreState){
                .cycle_step = -1,
                .orb_style = { ATTACK_STYLE_NONE, ATTACK_STYLE_NONE, ATTACK_STYLE_NONE },
                .fixed_orb_style = { ATTACK_STYLE_NONE, ATTACK_STYLE_NONE, ATTACK_STYLE_NONE },
            };
            break;
        case COLO_JAVELIN_COLOSSUS:
            npc->type_state.javelin = (ColoJavelinState){0};
            break;
        default:
            break;
    }
}

typedef enum {
    COLO_GEAR_MELEE = 0,
    COLO_GEAR_RANGED,
    COLO_GEAR_MAGIC,
    COLO_NUM_WEAPON_SETS
} ColoWeaponSet;

typedef struct {
    float dpt;
    uint8_t setup[NUM_GEAR_SLOTS];
} ColoBestGear;

#define COLO_BEST_GEAR_CACHE_WAYS 12

/* Feeds encounter_init_maxed_player_combat_stats and, through it, every prayer-restore
 * feature the item observation table bakes. Never rewritten during an episode. */
#define COLO_PLAYER_PRAYER_LEVEL 99

/* Replaces the 11 per-slot item stat rows. Those were a second encoding of combat maths
 * the player block already carries: encounter_compute_loadout_stats sums the equipped set
 * into EncounterLoadoutStats and the prefix emits attack_bonus, strength_bonus, max_hit,
 * attack_speed, attack_range, all five defence bonuses, eff_level and spell_base_damage.
 * Equipment EFFECTS were the one part with no aggregate in the observation. */
#define COLO_EQUIPPED_AGG_OBS_FEATURES 10

#define COLO_INVENTORY_OBS_CACHE_WAYS 8
#define COLO_INVENTORY_OBS_CACHE_FLOATS \
    (OSRS_INVENTORY_SIZE * OSRS_INVENTORY_CELL_OBS_FEATURES_CODED + \
     COLO_EQUIPPED_AGG_OBS_FEATURES)

typedef struct {
    ColoBestGear best_gear[COLO_BEST_GEAR_CACHE_WAYS][COLO_NUM_WEAPON_SETS][COLO_NUM_NPC_TYPES];
    uint64_t best_gear_sig[COLO_BEST_GEAR_CACHE_WAYS];
    int best_gear_valid[COLO_BEST_GEAR_CACHE_WAYS];
    int best_gear_next;
    float inventory[COLO_INVENTORY_OBS_CACHE_WAYS][COLO_INVENTORY_OBS_CACHE_FLOATS];
    uint64_t inventory_sig[COLO_INVENTORY_OBS_CACHE_WAYS];
    int inventory_valid[COLO_INVENTORY_OBS_CACHE_WAYS];
    int inventory_next;
} ColoObsMemoCaches;

typedef enum {
    COLO_LOADOUT_PROFILE_SPEEDRUN = 0,
    COLO_LOADOUT_PROFILE_BEGINNER,
    COLO_NUM_LOADOUT_PROFILES
} ColoLoadoutProfile;

typedef enum {
    COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY = 0,
    COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY,
    COLO_LOADOUT_PROFILE_MODE_MIXED,
} ColoLoadoutProfileMode;

typedef enum {
    COLO_RESTORE_SUPER_RESTORE = 0,
    COLO_RESTORE_SANFEW,
} ColoRestoreKind;

static const uint8_t COLO_BEGINNER_MELEE_LOADOUT[NUM_GEAR_SLOTS] = {
    [GEAR_SLOT_HEAD]   = ITEM_SERPENTINE_HELM,
    [GEAR_SLOT_CAPE]   = ITEM_INFERNAL_CAPE,
    [GEAR_SLOT_NECK]   = ITEM_AMULET_OF_BLOOD_FURY,
    [GEAR_SLOT_AMMO]   = ITEM_GOD_BLESSING,
    [GEAR_SLOT_WEAPON] = ITEM_OSMUMTENS_FANG,
    [GEAR_SLOT_SHIELD] = ITEM_DRAGON_DEFENDER,
    [GEAR_SLOT_BODY]   = ITEM_FIGHTER_TORSO,
    [GEAR_SLOT_LEGS]   = ITEM_CRYSTAL_LEGS,
    [GEAR_SLOT_HANDS]  = ITEM_BARROWS_GLOVES,
    [GEAR_SLOT_FEET]   = ITEM_ECHO_BOOTS,
    [GEAR_SLOT_RING]   = ITEM_BERSERKER_RING_I,
};

static const uint8_t COLO_BEGINNER_RANGED_LOADOUT[NUM_GEAR_SLOTS] = {
    [GEAR_SLOT_HEAD]   = ITEM_CRYSTAL_HELM,
    [GEAR_SLOT_CAPE]   = ITEM_INFERNAL_CAPE,
    [GEAR_SLOT_NECK]   = ITEM_NECKLACE_OF_ANGUISH,
    [GEAR_SLOT_AMMO]   = ITEM_GOD_BLESSING,
    [GEAR_SLOT_WEAPON] = ITEM_BOW_OF_FAERDHINEN,
    [GEAR_SLOT_SHIELD] = ITEM_NONE,
    [GEAR_SLOT_BODY]   = ITEM_CRYSTAL_BODY,
    [GEAR_SLOT_LEGS]   = ITEM_CRYSTAL_LEGS,
    [GEAR_SLOT_HANDS]  = ITEM_BARROWS_GLOVES,
    [GEAR_SLOT_FEET]   = ITEM_ECHO_BOOTS,
    [GEAR_SLOT_RING]   = ITEM_BERSERKER_RING_I,
};

static const uint8_t COLO_BEGINNER_MAGIC_LOADOUT[NUM_GEAR_SLOTS] = {
    [GEAR_SLOT_HEAD]   = ITEM_SERPENTINE_HELM,
    [GEAR_SLOT_CAPE]   = ITEM_INFERNAL_CAPE,
    [GEAR_SLOT_NECK]   = ITEM_AMULET_OF_BLOOD_FURY,
    [GEAR_SLOT_AMMO]   = ITEM_GOD_BLESSING,
    [GEAR_SLOT_WEAPON] = ITEM_TRIDENT_OF_SWAMP,
    [GEAR_SLOT_SHIELD] = ITEM_DRAGON_DEFENDER,
    [GEAR_SLOT_BODY]   = ITEM_FIGHTER_TORSO,
    [GEAR_SLOT_LEGS]   = ITEM_CRYSTAL_LEGS,
    [GEAR_SLOT_HANDS]  = ITEM_BARROWS_GLOVES,
    [GEAR_SLOT_FEET]   = ITEM_ECHO_BOOTS,
    [GEAR_SLOT_RING]   = ITEM_BERSERKER_RING_I,
};

static const uint8_t COLO_SPEEDRUN_MELEE_LOADOUT[NUM_GEAR_SLOTS] = {
    [GEAR_SLOT_HEAD]   = ITEM_TORVA_FULL_HELM,
    [GEAR_SLOT_CAPE]   = ITEM_INFERNAL_CAPE,
    [GEAR_SLOT_NECK]   = ITEM_AMULET_OF_RANCOUR,
    [GEAR_SLOT_AMMO]   = ITEM_DRAGON_ARROWS,
    [GEAR_SLOT_WEAPON] = ITEM_SCYTHE_OF_VITUR,
    [GEAR_SLOT_SHIELD] = ITEM_NONE,
    [GEAR_SLOT_BODY]   = ITEM_OATHPLATE_CHEST,
    [GEAR_SLOT_LEGS]   = ITEM_OATHPLATE_LEGS,
    [GEAR_SLOT_HANDS]  = ITEM_FEROCIOUS_GLOVES,
    [GEAR_SLOT_FEET]   = ITEM_AVERNIC_TREADS,
    [GEAR_SLOT_RING]   = ITEM_ULTOR_RING,
};

static const uint8_t COLO_SPEEDRUN_RANGED_LOADOUT[NUM_GEAR_SLOTS] = {
    [GEAR_SLOT_HEAD]   = ITEM_MASORI_MASK_F,
    [GEAR_SLOT_CAPE]   = ITEM_DIZANAS_QUIVER,
    [GEAR_SLOT_NECK]   = ITEM_NECKLACE_OF_ANGUISH,
    [GEAR_SLOT_AMMO]   = ITEM_DRAGON_ARROWS,
    [GEAR_SLOT_WEAPON] = ITEM_TWISTED_BOW,
    [GEAR_SLOT_SHIELD] = ITEM_NONE,
    [GEAR_SLOT_BODY]   = ITEM_MASORI_BODY_F,
    [GEAR_SLOT_LEGS]   = ITEM_MASORI_CHAPS_F,
    [GEAR_SLOT_HANDS]  = ITEM_ZARYTE_VAMBRACES,
    [GEAR_SLOT_FEET]   = ITEM_AVERNIC_TREADS,
    [GEAR_SLOT_RING]   = ITEM_ULTOR_RING,
};

static const uint8_t COLO_SPEEDRUN_MAGIC_LOADOUT[NUM_GEAR_SLOTS] = {
    [GEAR_SLOT_HEAD]   = ITEM_TORVA_FULL_HELM,
    [GEAR_SLOT_CAPE]   = ITEM_INFERNAL_CAPE,
    [GEAR_SLOT_NECK]   = ITEM_OCCULT_NECKLACE,
    [GEAR_SLOT_AMMO]   = ITEM_DRAGON_ARROWS,
    [GEAR_SLOT_WEAPON] = ITEM_TUMEKENS_SHADOW,
    [GEAR_SLOT_SHIELD] = ITEM_NONE,
    [GEAR_SLOT_BODY]   = ITEM_OATHPLATE_CHEST,
    [GEAR_SLOT_LEGS]   = ITEM_OATHPLATE_LEGS,
    [GEAR_SLOT_HANDS]  = ITEM_CONFLICTION_GAUNTLETS,
    [GEAR_SLOT_FEET]   = ITEM_AVERNIC_TREADS,
    [GEAR_SLOT_RING]   = ITEM_ULTOR_RING,
};

static const uint8_t* const COLO_SPEEDRUN_LOADOUTS[COLO_NUM_WEAPON_SETS] = {
    COLO_SPEEDRUN_MELEE_LOADOUT,
    COLO_SPEEDRUN_RANGED_LOADOUT,
    COLO_SPEEDRUN_MAGIC_LOADOUT,
};

static const uint8_t* const COLO_BEGINNER_LOADOUTS[COLO_NUM_WEAPON_SETS] = {
    COLO_BEGINNER_MELEE_LOADOUT,
    COLO_BEGINNER_RANGED_LOADOUT,
    COLO_BEGINNER_MAGIC_LOADOUT,
};

static const int COLO_MAGIC_SPELL_BASE_DAMAGE[COLO_NUM_LOADOUT_PROFILES] = {
    [COLO_LOADOUT_PROFILE_SPEEDRUN] = 34,
    [COLO_LOADOUT_PROFILE_BEGINNER] = 31,
};

static const uint8_t* const* const COLO_LOADOUT_PROFILES[COLO_NUM_LOADOUT_PROFILES] = {
    [COLO_LOADOUT_PROFILE_SPEEDRUN] = COLO_SPEEDRUN_LOADOUTS,
    [COLO_LOADOUT_PROFILE_BEGINNER] = COLO_BEGINNER_LOADOUTS,
};

typedef struct {
    int brew_doses;
    int restore_doses;
    int combat_doses;
    int ranged_doses;
    int surge_doses;
    ColoRestoreKind restore_kind;
} ColoSupplyLoadout;

static const ColoSupplyLoadout COLO_SUPPLY_LOADOUTS[COLO_NUM_LOADOUT_PROFILES] = {
    [COLO_LOADOUT_PROFILE_SPEEDRUN] = {
        .brew_doses = 4, .restore_doses = 28, .combat_doses = 4,
        .ranged_doses = 4, .surge_doses = 4,
        .restore_kind = COLO_RESTORE_SANFEW,
    },
    [COLO_LOADOUT_PROFILE_BEGINNER] = {
        .brew_doses = 24, .restore_doses = 32, .combat_doses = 8,
        .ranged_doses = 8, .surge_doses = 0,
        .restore_kind = COLO_RESTORE_SUPER_RESTORE,
    },
};

static const uint8_t* const* col_loadouts_for_profile(int profile) {
    if (profile < 0 || profile >= COLO_NUM_LOADOUT_PROFILES) {
        fprintf(stderr, "colosseum: invalid loadout profile %d\n", profile);
        abort();
    }
    return COLO_LOADOUT_PROFILES[profile];
}

#define COLO_INVENTORY_DISPLAY_SLOTS 28
#if COLO_INVENTORY_DISPLAY_SLOTS != OSRS_INVENTORY_SIZE
#error "Colosseum inventory size must match shared OSRS inventory size"
#endif

typedef OsrsInventoryCell ColoInvCell;

static const int COLO_GAMEPLAY_INVENTORY_LOADOUT[COLO_NUM_LOADOUT_PROFILES][COLO_INVENTORY_DISPLAY_SLOTS] = {
    [COLO_LOADOUT_PROFILE_SPEEDRUN] = {
        20997, 27238, 27235, 28947,
        26235, 27241, 19547, 13652,
        27275, 31106, 27610, 24780,

        30875, 12002, 12006, 21003,
        23685, 23733, 6685, 4417,

        10925, 10925, 10925, 10925,
        10925, 10925, 10925, 27281,
    },
    [COLO_LOADOUT_PROFILE_BEGINNER] = {
        27610, 26219, 25865, 19547,
        11806, 13652, 23975, 23971,
        6685, 6685, 3024, 3024,
        6685, 6685, 3024, 3024,
        6685, 6685, 3024, 3024,
        12695, 2444, 3024, 3024,
        23685, 23733, 12006, 27281,

    },
};

typedef void (*ColRawOsrsIdVisitor)(int raw_osrs_id, void* ctx);

static const int COLO_NON_CONSUMABLE_DISPLAY_RAW_OSRS_IDS[] = {
    27610,
    12006,
    27281,
};

static int col_raw_osrs_id_seen(const int* seen, int seen_count, int raw_osrs_id) {
    for (int i = 0; i < seen_count; i++) {
        if (seen[i] == raw_osrs_id) return 1;
    }
    return 0;
}

static void col_visit_raw_osrs_id_once(
    int raw_osrs_id,
    int* seen,
    int* seen_count,
    int seen_capacity,
    ColRawOsrsIdVisitor visit,
    void* ctx
) {
    if (raw_osrs_id <= 0 || col_raw_osrs_id_seen(seen, *seen_count, raw_osrs_id)) {
        return;
    }
    if (*seen_count >= seen_capacity) {
        fprintf(stderr, "colosseum sprite id visitor capacity exceeded\n");
        abort();
    }
    seen[(*seen_count)++] = raw_osrs_id;
    visit(raw_osrs_id, ctx);
}

static void col_for_each_reachable_consumable_dose_raw_osrs_id(
    ColRawOsrsIdVisitor visit,
    void* ctx
) {
    if (!visit) {
        fprintf(stderr, "colosseum consumable dose visitor is null\n");
        abort();
    }

    enum {
        REGISTRY_COUNT = (int)(
            sizeof(OSRS_CONSUMABLE_CLICK_REGISTRY) /
            sizeof(OSRS_CONSUMABLE_CLICK_REGISTRY[0])
        )
    };
    int seen[REGISTRY_COUNT];
    int seen_count = 0;

    for (int p = 0; p < COLO_NUM_LOADOUT_PROFILES; p++) {
        for (int slot = 0; slot < COLO_INVENTORY_DISPLAY_SLOTS; slot++) {
            uint16_t raw_osrs_id =
                (uint16_t)COLO_GAMEPLAY_INVENTORY_LOADOUT[p][slot];
            OsrsConsumableClick root =
                osrs_consumable_click_lookup_raw_osrs_id(raw_osrs_id);
            if (root.click_action != OSRS_CLICK_DRINK || root.dose_count == 0) {
                continue;
            }

            uint16_t current_raw_osrs_id = raw_osrs_id;
            for (int step = 0; current_raw_osrs_id != 0; step++) {
                if (step >= REGISTRY_COUNT) {
                    fprintf(stderr, "colosseum consumable dose chain cycles at raw id %u\n",
                        current_raw_osrs_id);
                    abort();
                }
                OsrsConsumableClick current =
                    osrs_consumable_click_lookup_raw_osrs_id(current_raw_osrs_id);
                if (current.click_action != OSRS_CLICK_DRINK ||
                        current.dose_count == 0 ||
                        current.consumable_kind != root.consumable_kind) {
                    fprintf(stderr, "colosseum consumable dose chain broke at raw id %u\n",
                        current_raw_osrs_id);
                    abort();
                }
                col_visit_raw_osrs_id_once(
                    current_raw_osrs_id,
                    seen,
                    &seen_count,
                    REGISTRY_COUNT,
                    visit,
                    ctx);
                current_raw_osrs_id =
                    osrs_consumable_raw_osrs_id_after_drink(current_raw_osrs_id);
            }
        }
    }
}

static void col_for_each_display_inventory_sprite_raw_osrs_id(
    ColRawOsrsIdVisitor visit,
    void* ctx
) {
    if (!visit) {
        fprintf(stderr, "colosseum display sprite visitor is null\n");
        abort();
    }

    enum {
        DISPLAY_CAPACITY =
            (int)(sizeof(COLO_NON_CONSUMABLE_DISPLAY_RAW_OSRS_IDS) /
                  sizeof(COLO_NON_CONSUMABLE_DISPLAY_RAW_OSRS_IDS[0])) +
            (int)(sizeof(OSRS_CONSUMABLE_CLICK_REGISTRY) /
                  sizeof(OSRS_CONSUMABLE_CLICK_REGISTRY[0]))
    };
    int seen[DISPLAY_CAPACITY];
    int seen_count = 0;

    for (int i = 0; i < (int)(sizeof(COLO_NON_CONSUMABLE_DISPLAY_RAW_OSRS_IDS) /
            sizeof(COLO_NON_CONSUMABLE_DISPLAY_RAW_OSRS_IDS[0])); i++) {
        col_visit_raw_osrs_id_once(
            COLO_NON_CONSUMABLE_DISPLAY_RAW_OSRS_IDS[i],
            seen,
            &seen_count,
            DISPLAY_CAPACITY,
            visit,
            ctx);
    }
    col_for_each_reachable_consumable_dose_raw_osrs_id(visit, ctx);
}

#define COLO_THRALL_MAX_HIT  3
#define COLO_THRALL_TICK     4
#define COLO_THRALL_RECAST_CD 17
static const int COLO_THRALL_LIFETIME_TICKS[COLO_NUM_LOADOUT_PROFILES] = {
    [COLO_LOADOUT_PROFILE_SPEEDRUN] = 198,
    [COLO_LOADOUT_PROFILE_BEGINNER] = 99,
};

#define COLO_DEATH_CHARGE_WINDOW   100
#define COLO_DEATH_CHARGE_CD       100
#define COLO_DEATH_CHARGE_SPEC_GAIN 15

static ColoLoadoutProfile col_sample_loadout_profile(
    int mode, float beginner_fraction, uint32_t* rng_state
) {
    if (mode == COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY)
        return COLO_LOADOUT_PROFILE_SPEEDRUN;
    if (mode == COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY)
        return COLO_LOADOUT_PROFILE_BEGINNER;
    if (mode != COLO_LOADOUT_PROFILE_MODE_MIXED) {
        fprintf(stderr, "colosseum: invalid loadout_profile_mode %d\n", mode);
        abort();
    }
    if (!(beginner_fraction >= 0.0f && beginner_fraction <= 1.0f)) {
        fprintf(stderr, "colosseum: beginner_loadout_fraction must be in [0,1], got %.6f\n",
                beginner_fraction);
        abort();
    }
    return encounter_rand_float(rng_state) < beginner_fraction
        ? COLO_LOADOUT_PROFILE_BEGINNER
        : COLO_LOADOUT_PROFILE_SPEEDRUN;
}

#define COLO_SURGE_RESTORE_AMOUNT 25
#define COLO_SURGE_COOLDOWN_TICKS 500

typedef struct {
    int start_wave;
    float damage_reward_coeff;
    float boss_damage_reward_coeff;
    float wave_clear_bonus;
    float boss_phase_bonus;
    float win_bonus;



    int farm_safe_damage_cap;

    int farm_cap_waves;
    float death_penalty_coeff;
    float timeout_penalty;
    int curriculum_agent;
    int loadout_profile_mode;
    float beginner_loadout_fraction;
    int mask_inventory_heads;
    int late_start_state_mode;

    float late_start_supply_fraction_per_wave;

    int bis_gear_oracle_mode;

    int episode_max_ticks_override;

    float prayer_switch_fail_prob;

    float player_damage_received_scale;

    float damage_scale_anneal_start;

    int damage_scale_anneal_ticks;

} ColConfig;

typedef enum {
    COLO_NPC_RENDER_ATTACK_NORMAL = 0,
    COLO_NPC_RENDER_ATTACK_JAVELIN_SKYFALL,
} ColoNpcRenderAttackAnim;

typedef struct {
    ColConfig config;
    const CollisionMap* collision_map;
    int world_offset_x;
    int world_offset_y;
    const HumanCommand* human_commands;
    int human_command_count;
    int human_command_mode;
    ColoNpcRenderAttackAnim npc_render_attack_anim[COLO_MAX_NPCS];
    int manticore_charge_started[COLO_MAX_NPCS];
    int player_render_hit_count;
    int player_render_hit_damage[ENCOUNTER_RENDER_HITS_MAX];
    int npc_render_hit_count[COLO_MAX_NPCS];
    int npc_render_hit_damage[COLO_MAX_NPCS][ENCOUNTER_RENDER_HITS_MAX];

    int thrall_attacked_this_tick;
    int player_spec_fired_kind;
    int player_venator_chain_count;
    int player_venator_chain_slots[OSRS_VENATOR_MAX_CHAIN_HITS];
} ColosseumContext;

/* Clears only what was written. col_render_hit_events_add aborts past
 * ENCOUNTER_RENDER_HITS_MAX and fills strictly below count, and a row whose count is zero
 * is already zero from the previous clear, so the old blanket memset of
 * COLO_MAX_NPCS x ENCOUNTER_RENDER_HITS_MAX ints re-zeroed memory that was already zero.
 * That was 3488 B per tick per env and 15% of the whole env step. */
static inline void col_context_clear_render_events(ColosseumContext* ctx) {
    memset(ctx->npc_render_attack_anim, 0, sizeof(ctx->npc_render_attack_anim));
    memset(ctx->manticore_charge_started, 0, sizeof(ctx->manticore_charge_started));
    for (int h = 0; h < ctx->player_render_hit_count; h++)
        ctx->player_render_hit_damage[h] = 0;
    ctx->player_render_hit_count = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        int n = ctx->npc_render_hit_count[i];
        if (n <= 0) continue;
        for (int h = 0; h < n; h++) ctx->npc_render_hit_damage[i][h] = 0;
        ctx->npc_render_hit_count[i] = 0;
    }
    ctx->thrall_attacked_this_tick = 0;
    ctx->player_spec_fired_kind = 0;
    ctx->player_venator_chain_count = 0;
    for (int i = 0; i < OSRS_VENATOR_MAX_CHAIN_HITS; i++)
        ctx->player_venator_chain_slots[i] = -1;
}

static inline void col_context_record_venator_chain(
    ColosseumContext* ctx,
    const OsrsVenatorChain* chain
) {
    if (!chain) abort();
    osrs_venator_validate_chain_length(chain->length);
    ctx->player_venator_chain_count = (int)chain->length;
    for (int i = 0; i < OSRS_VENATOR_MAX_CHAIN_HITS; i++)
        ctx->player_venator_chain_slots[i] = -1;
    for (int i = 0; i < (int)chain->length; i++)
        ctx->player_venator_chain_slots[i] = chain->hits[i].slot;
}

typedef struct {
    float damage_dealt;
    float damage_boss;
    float damage_received;
    int prayer_correct;
    int attacks_fired;
    int wave_completed;
    int player_moved;
    int player_attacked;

    float fresh_damage_dealt;
    float fresh_damage_boss;

    float fresh_damage_reinforcement;

    int styles_faced_mask;
    float offpray_damage_this_tick;
    float avoidable_threat_by_style[4];

    float landed_offpray_damage;
    float landed_prayed_damage;
    float landed_unprayable_damage;
    float landed_self_damage;
} ColoTickScratch;

typedef struct {
    float episode_return;
    int episode_length;
    int wave_reached;
    int waves_cleared;
    int win;
    int died;
    int timed_out;
    float total_damage_dealt;
    float total_damage_received;
    float total_boss_damage;
    int boss_phase_reached;
    int total_npc_kills;
    int total_prayer_correct;
    int total_npc_attacks;
    int modifiers_picked;
    int active_tier_sum;

    float pray_faced_by_type[COLO_NUM_NPC_TYPES];
    float pray_correct_by_type[COLO_NUM_NPC_TYPES];
    float offpray_damage_by_type[COLO_NUM_NPC_TYPES];
    float total_damage_by_type[COLO_NUM_NPC_TYPES];

    float death_by_type[COLO_NUM_NPC_TYPES];
    float npc_attack_death_by_type[COLO_NUM_NPC_TYPES];
    float death_fatal_damage;

    float offpray_damage_conflict;
    float offpray_damage_solo;
    float death_on_conflict_tick;

    float death_dmg_unprayable;
    float death_dmg_offpray;
    float death_dmg_prayed;
    float death_dmg_self;
    float death_heal_remaining;
    float farm_damage;
    float reward_steps;
    float reward_clamped_steps;
    float reward_clamp_loss;
    float reward_raw_peak;
    float clamp_loss_wave_clear;
    float clamp_loss_win;
    float rew_damage;
    float rew_boss_phase;
    float rew_wave_clear;
    float rew_win;
    float rew_death;
    float rew_timeout;
    float avoid_total;
    float avoid_achieved;
    float avoid_missed;
    float avoid_impossible;
    float dmg_unprayable;
    float inv_memo_hits;
    float inv_memo_misses;
    float npc_blocked_calls;
    float npc_blocked_tiles;
    float npc_stamp_tiles;
    float typeless_damage_by_type[COLO_NUM_NPC_TYPES];
    float sol_damage_by_source[COLO_NUM_SOL_DAMAGE_SOURCES];
    float javelin_damage_by_source[COLO_NUM_JAVELIN_DAMAGE_SOURCES];
    float death_by_source[COLO_NUM_DAMAGE_SOURCES];
    float doom_death_by_source[COLO_NUM_DAMAGE_SOURCES];
    float outcome_score;
    float max_wave_depth;
} ColosseumLog;

typedef struct {
    Player player;

    ColoTickScratch tick_scratch;

    ColoNPC npcs[COLO_MAX_NPCS];
    uint32_t next_npc_render_id;
    int current_obs_slots[COLO_OBS_NPCS];

    int wave;
    int wave_spawn_target;
    int tick;
    int wave_spawn_delay;
    int wave_ready_delay;

    int wave_attack_delay;
    int episode_max_ticks;
    int reinforcement_timer;
    int warband_cycle_anchor;

    AttackStyle wave_manticore_orb_style[3];
    int wave_manticore_pattern_rolled;
    int current_wave_total_killable;

    int current_wave_hp_pool;
    float current_wave_fresh_damage;
    int episode_over;
    int time_limit_truncated;
    ColoOutcome winner;

    ColoModifierState modifiers;
    SolHereditState sol;

    int doom_stacks;
    ColoBeeSwarm bees[COLO_MAX_BEE_SWARMS];
    ColoTotem totems[COLO_MAX_TOTEMS];
    int sol_totem_spawn_timer;
    ColoSolarflareOrb solarflare;
    int player_venom;
    int player_venom_timer;
    int player_poison;
    int player_poison_timer;

    int molten_x[COLO_SOL_HAZARD_TILES_MAX];
    int molten_y[COLO_SOL_HAZARD_TILES_MAX];
    int molten_kind[COLO_SOL_HAZARD_TILES_MAX];
    int molten_lifetime[COLO_SOL_HAZARD_TILES_MAX];
    int molten_burn_timer[COLO_SOL_HAZARD_TILES_MAX];
    int molten_count;

    float reward;
    float episode_return;
    int last_hit_by_type;
    int last_damage_source;
    int last_doom_damage_source;
    int rewarded_boss_phase;
    int min_sol_hp_seen;

    OsrsInteraction interaction;
    int player_last_interaction_target_slot;
    int player_last_interaction_age;
    ColoWeaponSet weapon_set;
    EncounterLoadoutStats loadout_stats_live;
    OsrsEquipmentEffectProfile effects_live;
    int loadout_stats_dirty;

    ColoLoadoutProfile active_loadout_profile;
    ColoSupplyLoadout full_supplies;

    int stat_drift_timer;
    int divine_combat_timer;
    int divine_ranged_timer;
    int surge_doses;
    int surge_cooldown;
    OsrsInventoryCell inventory_cells[OSRS_INVENTORY_SIZE];

    int thrall_active;
    int thrall_target_slot;
    int thrall_lifetime_left;
    int thrall_attack_timer;
    int thrall_recast_cd;

    int death_charge_window_left;
    int death_charge_cd;

    int player_attack_npc_idx;
    int player_attack_dmg;
    AttackStyle player_attack_style_id;
    EncounterProjectileTiming player_attack_timing;

    EncounterPendingHitQueue player_pending_hits;

    int player_dest_x, player_dest_y;

    uint8_t npc_collision_flags[COLO_ARENA_WIDTH][COLO_ARENA_HEIGHT];
    /* The player occupies one tile, so a full arena grid recorded one cell and was
     * memset in full twice per tick. Cache the cell instead. -1 means off-grid. */
    int player_grid_x, player_grid_y;

    int start_wave;
    uint32_t rng_state;

    float active_player_damage_received_scale;
    int profiling_suppressed;

    /* Last two members, and both are skipped by probe_colo_sim_invariant. The probe hashes
     * this struct's raw bytes, so a member that grows mid-struct repads everything after it
     * and moves all twelve digests with no behaviour change -- which is exactly what adding
     * reward telemetry to the log did. At the tail, neither can shift a hashed byte. */
    ColosseumLog log;
    ColoObsMemoCaches obs_memos;
} ColosseumState;

typedef enum {
    COLO_DMG_OFFPRAY = 0,
    COLO_DMG_UNPRAYABLE = 1,
} ColoDamageChannel;

static inline int col_damage_player_from(
    ColosseumState* s, int dmg, int npc_type, ColoDamageChannel channel);
static void col_apply_bis_gear_oracle(ColosseumState* s);

static void col_build_live_inventory_display(
    const ColosseumState* s,
    int out[COLO_INVENTORY_DISPLAY_SLOTS]
) {
    for (int i = 0; i < COLO_INVENTORY_DISPLAY_SLOTS; i++)
        out[i] = s->inventory_cells[i].raw_osrs_id;
}
