#line 39 "encounter_inferno.h"
#define INF_ARENA_MIN_X    11
#define INF_ARENA_MAX_X    39
#define INF_ARENA_MIN_Y    14
#define INF_ARENA_MAX_Y    43
#define INF_ARENA_WIDTH    (INF_ARENA_MAX_X - INF_ARENA_MIN_X + 1)
#define INF_ARENA_HEIGHT   (INF_ARENA_MAX_Y - INF_ARENA_MIN_Y + 1)
#define INF_TOPOLOGY_MIN_X 4
#define INF_TOPOLOGY_MAX_X 43
#define INF_TOPOLOGY_MIN_Y 5
#define INF_TOPOLOGY_MAX_Y 55
#define INF_TOPOLOGY_WIDTH \
    (INF_TOPOLOGY_MAX_X - INF_TOPOLOGY_MIN_X + 1)
#define INF_TOPOLOGY_HEIGHT \
    (INF_TOPOLOGY_MAX_Y - INF_TOPOLOGY_MIN_Y + 1)

#define INF_PLAYER_START_X 25
#define INF_PLAYER_START_Y 16
#define INF_ZUK_PLAYER_START_X 25
#define INF_ZUK_PLAYER_START_Y 42
#define INF_ZUK_X 22
#define INF_ZUK_Y 49
#define INF_ZUK_SHIELD_X 23
#define INF_ZUK_SHIELD_Y 44

#define INF_NUM_PILLARS   3
#define INF_PILLAR_SIZE   3
#define INF_PILLAR_HP     255

static const int INF_PILLAR_POS[INF_NUM_PILLARS][2] = {
    { 21, 20 },
    { 11, 34 },
    { 28, 36 },
};

#define INF_NUM_SPAWN_POS 9
static const int INF_SPAWN_POS[INF_NUM_SPAWN_POS][2] = {
    {12, 38}, {33, 38}, {14, 32}, {34, 31}, {27, 26},
    {16, 20}, {34, 18}, {12, 15}, {26, 15},
};

#define INF_NIBBLER_SPAWN_X 20
#define INF_NIBBLER_SPAWN_Y 31

#define INF_MAX_TICKS     18000
#define INF_NUM_WAVES     69
#define INF_WAVE_ZUK      (INF_NUM_WAVES - 1)
#define INF_START_WAVE_ZUK_JAD      (INF_WAVE_ZUK + 1)
#define INF_MAX_PUBLIC_START_WAVE   (INF_NUM_WAVES + 2)
#define INF_NUM_ACTION_HEADS OSRS_BASE_NUM_ACTION_HEADS

typedef enum {
    INF_NPC_NIBBLER = 0,
    INF_NPC_BAT,
    INF_NPC_BLOB,
    INF_NPC_BLOB_MELEE,
    INF_NPC_BLOB_RANGE,
    INF_NPC_BLOB_MAGE,
    INF_NPC_MELEER,
    INF_NPC_RANGER,
    INF_NPC_MAGER,
    INF_NPC_JAD,
    INF_NPC_ZUK,
    INF_NPC_HEALER_JAD,
    INF_NPC_HEALER_ZUK,
    INF_NPC_ZUK_SHIELD,
    INF_NUM_NPC_TYPES
} InfNPCType;

typedef enum {
    INF_OUTCOME_PLAYER_WON = 0,
    INF_OUTCOME_PLAYER_DIED = 1,
} InfOutcome;

static const int INF_NPC_DEF_IDS[INF_NUM_NPC_TYPES] = {
    [INF_NPC_NIBBLER]    = 7691,
    [INF_NPC_BAT]        = 7692,
    [INF_NPC_BLOB]       = 7693,
    [INF_NPC_BLOB_MELEE] = 7696,
    [INF_NPC_BLOB_RANGE] = 7695,
    [INF_NPC_BLOB_MAGE]  = 7694,
    [INF_NPC_MELEER]     = 7697,
    [INF_NPC_RANGER]     = 7698,
    [INF_NPC_MAGER]      = 7699,
    [INF_NPC_JAD]        = 7700,
    [INF_NPC_ZUK]        = 7706,
    [INF_NPC_HEALER_JAD] = 7701,
    [INF_NPC_HEALER_ZUK] = 7708,
    [INF_NPC_ZUK_SHIELD] = 7707,
};

static const char* inf_npc_type_name(int type) {
    switch (type) {
        case INF_NPC_NIBBLER: return "Jal-Nib";
        case INF_NPC_BAT: return "Jal-MejRah";
        case INF_NPC_BLOB: return "Jal-Ak";
        case INF_NPC_BLOB_MELEE: return "Jal-AkRek-Ket";
        case INF_NPC_BLOB_RANGE: return "Jal-AkRek-Xil";
        case INF_NPC_BLOB_MAGE: return "Jal-AkRek-Mej";
        case INF_NPC_MELEER: return "Jal-ImKot";
        case INF_NPC_RANGER: return "Jal-Xil";
        case INF_NPC_MAGER: return "Jal-Zek";
        case INF_NPC_JAD: return "JalTok-Jad";
        case INF_NPC_ZUK: return "TzKal-Zuk";
        case INF_NPC_HEALER_JAD: return "Yt-HurKot";
        case INF_NPC_HEALER_ZUK: return "Jal-MejJak";
        case INF_NPC_ZUK_SHIELD: return "Ancestral Glyph";
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

    int melee_att_bonus;
    int range_att_bonus;
    int magic_att_bonus;

    int melee_str_bonus;
    int ranged_str_bonus;
    int magic_base_dmg;
    int magic_dmg_pct;

    int magic_def_bonus;
    int ranged_def_bonus;

    int max_hit_cap;

    int stun_on_spawn;
    int can_move;
} InfNPCStats;

typedef struct {
    int attack_range;
    AttackStyle default_style;
    MeleeStyle melee_style;
    int can_melee;
    int magic_base_dmg;
    int magic_dmg_pct;
    int max_hit_cap;
    int stun_on_spawn;
    int can_move;
    int attack_decision_ticks;
} InfNPCOverlay;

static const MonsterIndex INF_NPC_TO_MON[INF_NUM_NPC_TYPES] = {
    [INF_NPC_NIBBLER]    = MON_JAL_NIB,
    [INF_NPC_BAT]        = MON_JAL_MEJRAH,
    [INF_NPC_BLOB]       = MON_JAL_AK,
    [INF_NPC_BLOB_MELEE] = MON_JAL_AKREK_KET,
    [INF_NPC_BLOB_RANGE] = MON_JAL_AKREK_XIL,
    [INF_NPC_BLOB_MAGE]  = MON_JAL_AKREK_MEJ,
    [INF_NPC_MELEER]     = MON_JAL_IMKOT,
    [INF_NPC_RANGER]     = MON_JAL_XIL,
    [INF_NPC_MAGER]      = MON_JAL_ZEK,
    [INF_NPC_JAD]        = MON_JALTOK_JAD,
    [INF_NPC_ZUK]        = MON_TZKAL_ZUK,
    [INF_NPC_HEALER_JAD] = MON_YT_HURKOT,
    [INF_NPC_HEALER_ZUK] = MON_JAL_MEJJAK,
    [INF_NPC_ZUK_SHIELD] = MON_ZUK_SHIELD,
};

static const InfNPCOverlay INF_NPC_OVERLAY[INF_NUM_NPC_TYPES] = {
    [INF_NPC_NIBBLER]    = { 1,  ATTACK_STYLE_MELEE,  MELEE_STYLE_CRUSH, 0,   0,   0, 0, 1, 1, 0 },
    [INF_NPC_BAT]        = { 4,  ATTACK_STYLE_RANGED, MELEE_STYLE_STAB,  0,   0,   0, 0, 0, 1, 0 },
    [INF_NPC_BLOB]       = { 15, ATTACK_STYLE_MAGIC,  MELEE_STYLE_CRUSH, 1,  29, 100, 0, 0, 1, 3 },
    [INF_NPC_BLOB_MELEE] = { 1,  ATTACK_STYLE_MELEE,  MELEE_STYLE_CRUSH, 0,   0,   0, 0, 0, 1, 0 },
    [INF_NPC_BLOB_RANGE] = { 15, ATTACK_STYLE_RANGED, MELEE_STYLE_STAB,  0,   0,   0, 0, 0, 1, 0 },
    [INF_NPC_BLOB_MAGE]  = { 15, ATTACK_STYLE_MAGIC,  MELEE_STYLE_STAB,  0,  18, 100, 0, 0, 1, 0 },
    [INF_NPC_MELEER]     = { 1,  ATTACK_STYLE_MELEE,  MELEE_STYLE_SLASH, 0,   0,   0, 0, 0, 1, 0 },
    [INF_NPC_RANGER]     = { 15, ATTACK_STYLE_RANGED, MELEE_STYLE_CRUSH, 1,   0,   0, 0, 0, 1, 0 },
    [INF_NPC_MAGER]      = { 15, ATTACK_STYLE_MAGIC,  MELEE_STYLE_STAB,  1,  70, 100, 0, 0, 1, 0 },
    [INF_NPC_JAD]        = { 50, ATTACK_STYLE_RANGED, MELEE_STYLE_STAB,  1, 113, 100, 113, 0, 1, 0 },
    [INF_NPC_ZUK]        = { 99, ATTACK_STYLE_MAGIC,  MELEE_STYLE_STAB,  0, 148, 100, 0, 8, 0, 0 },
    [INF_NPC_HEALER_JAD] = { 1,  ATTACK_STYLE_MELEE,  MELEE_STYLE_CRUSH, 0,   0,   0, 0, 1, 1, 0 },
    [INF_NPC_HEALER_ZUK] = { 99, ATTACK_STYLE_MAGIC,  MELEE_STYLE_STAB,  0,  10, 100, 0, 1, 0, 0 },
    [INF_NPC_ZUK_SHIELD] = { 0,  ATTACK_STYLE_NONE,   MELEE_STYLE_STAB,  0,   0,   0, 0, 1, 0, 0 },
};

static InfNPCStats INF_NPC_STATS[INF_NUM_NPC_TYPES];

static void inf_build_npc_stats(void) {
    for (int i = 0; i < INF_NUM_NPC_TYPES; i++) {
        const MonsterStats* m = &MONSTER_DATABASE[INF_NPC_TO_MON[i]];
        const InfNPCOverlay* o = &INF_NPC_OVERLAY[i];
        InfNPCStats* s = &INF_NPC_STATS[i];

        s->hp              = m->hp;
        s->attack_speed    = o->attack_decision_ticks > 0
            ? o->attack_decision_ticks
            : m->attack_speed;
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
        s->magic_def_bonus = m->magic_def;
        s->ranged_def_bonus = m->ranged_def;

        s->attack_range    = o->attack_range;
        s->default_style   = o->default_style;
        s->melee_style     = o->melee_style;
        s->can_melee       = o->can_melee;
        s->magic_base_dmg  = o->magic_base_dmg;
        s->magic_dmg_pct   = o->magic_dmg_pct;
        s->max_hit_cap     = o->max_hit_cap;
        s->stun_on_spawn   = o->stun_on_spawn;
        s->can_move        = o->can_move;
    }
}

#define INF_MAX_NPCS_PER_WAVE 9

typedef struct {
    uint8_t types[INF_MAX_NPCS_PER_WAVE];
    int count;
} InfWaveDef;

static const InfWaveDef INF_WAVES[INF_NUM_WAVES] = {
    #define N INF_NPC_NIBBLER
    #define B INF_NPC_BAT
    #define BL INF_NPC_BLOB
    #define M INF_NPC_MELEER
    #define R INF_NPC_RANGER
    #define MA INF_NPC_MAGER
    #define J INF_NPC_JAD
    #define Z INF_NPC_ZUK
    #define W(...) { .types = { __VA_ARGS__ }, .count = sizeof((uint8_t[]){__VA_ARGS__}) }

    [0]  = W(N,N,N, B),
    [1]  = W(N,N,N, B,B),
    [2]  = W(N,N,N, N,N,N),
    [3]  = W(N,N,N, BL),
    [4]  = W(N,N,N, B,BL),
    [5]  = W(N,N,N, B,B,BL),
    [6]  = W(N,N,N, BL,BL),
    [7]  = W(N,N,N, N,N,N),

    [8]  = W(N,N,N, M),
    [9]  = W(N,N,N, B,M),
    [10] = W(N,N,N, B,B,M),
    [11] = W(N,N,N, BL,M),
    [12] = W(N,N,N, B,BL,M),
    [13] = W(N,N,N, B,B,BL,M),
    [14] = W(N,N,N, BL,BL,M),
    [15] = W(N,N,N, M,M),
    [16] = W(N,N,N, N,N,N),

    [17] = W(N,N,N, R),
    [18] = W(N,N,N, B,R),
    [19] = W(N,N,N, B,B,R),
    [20] = W(N,N,N, BL,R),
    [21] = W(N,N,N, B,BL,R),
    [22] = W(N,N,N, B,B,BL,R),
    [23] = W(N,N,N, BL,BL,R),
    [24] = W(N,N,N, M,R),
    [25] = W(N,N,N, B,M,R),
    [26] = W(N,N,N, B,B,M,R),
    [27] = W(N,N,N, BL,M,R),
    [28] = W(N,N,N, B,BL,M,R),
    [29] = W(N,N,N, B,B,BL,M,R),
    [30] = W(N,N,N, BL,BL,M,R),
    [31] = W(N,N,N, M,M,R),
    [32] = W(N,N,N, R,R),
    [33] = W(N,N,N, N,N,N),

    [34] = W(N,N,N, MA),
    [35] = W(N,N,N, B,MA),
    [36] = W(N,N,N, B,B,MA),
    [37] = W(N,N,N, BL,MA),
    [38] = W(N,N,N, B,BL,MA),
    [39] = W(N,N,N, B,B,BL,MA),
    [40] = W(N,N,N, BL,BL,MA),
    [41] = W(N,N,N, M,MA),
    [42] = W(N,N,N, B,M,MA),
    [43] = W(N,N,N, B,B,M,MA),
    [44] = W(N,N,N, BL,M,MA),
    [45] = W(N,N,N, B,BL,M,MA),
    [46] = W(N,N,N, B,B,BL,M,MA),
    [47] = W(N,N,N, BL,BL,M,MA),
    [48] = W(N,N,N, M,M,MA),
    [49] = W(N,N,N, R,MA),
    [50] = W(N,N,N, B,R,MA),
    [51] = W(N,N,N, B,B,R,MA),
    [52] = W(N,N,N, BL,R,MA),
    [53] = W(N,N,N, B,BL,R,MA),
    [54] = W(N,N,N, B,B,BL,R,MA),
    [55] = W(N,N,N, BL,BL,R,MA),
    [56] = W(N,N,N, M,R,MA),
    [57] = W(N,N,N, B,M,R,MA),
    [58] = W(N,N,N, B,B,M,R,MA),
    [59] = W(N,N,N, BL,M,R,MA),
    [60] = W(N,N,N, B,BL,M,R,MA),
    [61] = W(N,N,N, B,B,BL,M,R,MA),
    [62] = W(N,N,N, BL,BL,M,R,MA),
    [63] = W(N,N,N, M,M,R,MA),
    [64] = W(N,N,N, R,R,MA),
    [65] = W(N,N,N, MA,MA),

    [66] = W(J),
    [67] = W(J,J,J),
    [68] = W(Z),

    #undef N
    #undef B
    #undef BL
    #undef M
    #undef R
    #undef MA
    #undef J
    #undef Z
    #undef W
};

#define INF_MAX_NPCS      37
#define INF_OBS_NPCS      14
#define INF_START_READY_TICKS 6

#define INF_MAX_DEAD_MOBS 16

typedef struct {
    InfNPCType type;
    int x, y;
    int hp, max_hp;
} InfDeadMob;

#define INF_MAX_PENDING_SPARKS 32

typedef struct {
    int active;
    int src_x, src_y;
    int x, y;
    int damage;
    int ticks_remaining;
    int visual_emitted;
} InfPendingSpark;

#define INF_MAX_NPC_TARGET_HITS 16

typedef struct {
    int active;
    int target_idx;
    int damage;
    int ticks_remaining;
} InfNpcTargetHit;

typedef struct {
    AttackStyle attack_style;
    int healer_spawned;
} InfJadState;

typedef struct {
    int owner_idx;
    int target;
    int timer;
} InfHealerState;

typedef struct {
    InfNPCType type;
    int x, y;
    int hp, max_hp;
    int size;
    uint32_t render_id;
    int attack_timer;
    AttackStyle attack_style;
    int active;
    int target_x, target_y;
    int stun_timer;

    int no_los_ticks;
    int dig_freeze_timer;
    int dig_attack_delay;

    int blob_scanned_prayer;
    int had_los_last_tick;

    int resurrect_cooldown;
    int resurrection_count;

    int frozen_ticks;

    union {
        InfJadState jad;
        InfHealerState healer;
    } type_state;

    EncounterPendingHitQueue pending_hits;

    int death_ticks;

    int aggro_target;

    int attacked_this_tick;
    AttackStyle attack_style_this_tick;
    int attack_visual_target;
    int resurrecting_this_tick;
    int resurrection_visual_target;
    int moved_this_tick;
    int hit_landed_this_tick;
    int hit_damage;
    int hit_was_successful_this_tick;
    int hit_spell_type;
    int spawn_blob_splits_on_removal;
} InfNPC;

static inline int inf_npc_is_healer(InfNPCType type) {
    return type == INF_NPC_HEALER_JAD || type == INF_NPC_HEALER_ZUK;
}

static inline InfHealerState* inf_npc_healer(InfNPC* npc) {
    assert(inf_npc_is_healer(npc->type));
    return &npc->type_state.healer;
}

static inline InfJadState* inf_npc_jad(InfNPC* npc) {
    assert(npc->type == INF_NPC_JAD);
    return &npc->type_state.jad;
}

static inline const InfJadState* inf_npc_jad_const(const InfNPC* npc) {
    assert(npc->type == INF_NPC_JAD);
    return &npc->type_state.jad;
}

static void inf_npc_init_type_state(InfNPC* npc) {
    switch (npc->type) {
        case INF_NPC_JAD:
            npc->type_state.jad = (InfJadState){
                .attack_style = ATTACK_STYLE_NONE, .healer_spawned = 0 };
            break;
        case INF_NPC_HEALER_JAD:
        case INF_NPC_HEALER_ZUK:
            npc->type_state.healer = (InfHealerState){
                .owner_idx = -1, .target = -1, .timer = 0 };
            break;
        default:
            break;
    }
}

typedef struct {
    int x, y;
    int hp;
    int active;
} InfPillar;

typedef struct {
    int shield_idx;
    int shield_dir;
    int shield_freeze;

    int set_timer;
    int set_interval;
    int enraged;

    int healer_spawned;
    int jad_spawned;

    int timer_paused;
    int has_paused;
} InfZukState;

typedef enum {
    INF_GEAR_MAGE = 0,
    INF_GEAR_LONG_RANGE,
    INF_GEAR_BP,
    INF_NUM_WEAPON_SETS
} InfWeaponSet;

typedef enum {
    INF_LOADOUT_PROFILE_MAX = 0,
    INF_LOADOUT_PROFILE_BUDGET,
    INF_NUM_LOADOUT_PROFILES
} InfLoadoutProfile;

typedef enum {
    INF_LOADOUT_PROFILE_MODE_MAX_ONLY = 0,
    INF_LOADOUT_PROFILE_MODE_BUDGET_ONLY,
    INF_LOADOUT_PROFILE_MODE_MIXED_MAX_BUDGET,
} InfLoadoutProfileMode;

static const uint8_t INF_MAX_MAGE_LOADOUT[NUM_GEAR_SLOTS] = {
    [GEAR_SLOT_HEAD]   = ITEM_MASORI_MASK_F,
    [GEAR_SLOT_CAPE]   = ITEM_DIZANAS_QUIVER,
    [GEAR_SLOT_NECK]   = ITEM_OCCULT_NECKLACE,
    [GEAR_SLOT_AMMO]   = ITEM_DRAGON_ARROWS,
    [GEAR_SLOT_WEAPON] = ITEM_KODAI_WAND,
    [GEAR_SLOT_SHIELD] = ITEM_ELYSIAN_SPIRIT_SHIELD,
    [GEAR_SLOT_BODY]   = ITEM_VIRTUS_ROBE_TOP,
    [GEAR_SLOT_LEGS]   = ITEM_VIRTUS_ROBE_BOTTOM,
    [GEAR_SLOT_HANDS]  = ITEM_CONFLICTION_GAUNTLETS,
    [GEAR_SLOT_FEET]   = ITEM_AVERNIC_TREADS,
    [GEAR_SLOT_RING]   = ITEM_VENATOR_RING,
};

static const uint8_t INF_MAX_RANGE_LONG_LOADOUT[NUM_GEAR_SLOTS] = {
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
    [GEAR_SLOT_RING]   = ITEM_VENATOR_RING,
};

static const uint8_t INF_MAX_RANGE_FAST_LOADOUT[NUM_GEAR_SLOTS] = {
    [GEAR_SLOT_HEAD]   = ITEM_MASORI_MASK_F,
    [GEAR_SLOT_CAPE]   = ITEM_DIZANAS_QUIVER,
    [GEAR_SLOT_NECK]   = ITEM_NECKLACE_OF_ANGUISH,
    [GEAR_SLOT_AMMO]   = ITEM_DRAGON_DART,
    [GEAR_SLOT_WEAPON] = ITEM_TOXIC_BLOWPIPE,
    [GEAR_SLOT_SHIELD] = ITEM_NONE,
    [GEAR_SLOT_BODY]   = ITEM_MASORI_BODY_F,
    [GEAR_SLOT_LEGS]   = ITEM_MASORI_CHAPS_F,
    [GEAR_SLOT_HANDS]  = ITEM_ZARYTE_VAMBRACES,
    [GEAR_SLOT_FEET]   = ITEM_AVERNIC_TREADS,
    [GEAR_SLOT_RING]   = ITEM_VENATOR_RING,
};

static const uint8_t INF_BUDGET_MAGE_LOADOUT[NUM_GEAR_SLOTS] = {
    [GEAR_SLOT_HEAD]   = ITEM_CRYSTAL_HELM,
    [GEAR_SLOT_CAPE]   = ITEM_DIZANAS_QUIVER,
    [GEAR_SLOT_NECK]   = ITEM_OCCULT_NECKLACE,
    [GEAR_SLOT_AMMO]   = ITEM_GOD_BLESSING,
    [GEAR_SLOT_WEAPON] = ITEM_DRAGON_HUNTER_WAND,
    [GEAR_SLOT_SHIELD] = ITEM_CRYSTAL_SHIELD,
    [GEAR_SLOT_BODY]   = ITEM_AHRIMS_ROBETOP,
    [GEAR_SLOT_LEGS]   = ITEM_AHRIMS_ROBESKIRT,
    [GEAR_SLOT_HANDS]  = ITEM_CONFLICTION_GAUNTLETS,
    [GEAR_SLOT_FEET]   = ITEM_ECHO_BOOTS,
    [GEAR_SLOT_RING]   = ITEM_VENATOR_RING,
};

static const uint8_t INF_BUDGET_RANGE_LONG_LOADOUT[NUM_GEAR_SLOTS] = {
    [GEAR_SLOT_HEAD]   = ITEM_CRYSTAL_HELM,
    [GEAR_SLOT_CAPE]   = ITEM_DIZANAS_QUIVER,
    [GEAR_SLOT_NECK]   = ITEM_NECKLACE_OF_ANGUISH,
    [GEAR_SLOT_AMMO]   = ITEM_GOD_BLESSING,
    [GEAR_SLOT_WEAPON] = ITEM_BOW_OF_FAERDHINEN,
    [GEAR_SLOT_SHIELD] = ITEM_NONE,
    [GEAR_SLOT_BODY]   = ITEM_CRYSTAL_BODY,
    [GEAR_SLOT_LEGS]   = ITEM_CRYSTAL_LEGS,
    [GEAR_SLOT_HANDS]  = ITEM_BARROWS_GLOVES,
    [GEAR_SLOT_FEET]   = ITEM_ECHO_BOOTS,
    [GEAR_SLOT_RING]   = ITEM_VENATOR_RING,
};

static const uint8_t INF_BUDGET_RANGE_FAST_LOADOUT[NUM_GEAR_SLOTS] = {
    [GEAR_SLOT_HEAD]   = ITEM_CRYSTAL_HELM,
    [GEAR_SLOT_CAPE]   = ITEM_DIZANAS_QUIVER,
    [GEAR_SLOT_NECK]   = ITEM_NECKLACE_OF_ANGUISH,
    [GEAR_SLOT_AMMO]   = ITEM_DRAGON_DART,
    [GEAR_SLOT_WEAPON] = ITEM_TOXIC_BLOWPIPE,
    [GEAR_SLOT_SHIELD] = ITEM_NONE,
    [GEAR_SLOT_BODY]   = ITEM_CRYSTAL_BODY,
    [GEAR_SLOT_LEGS]   = ITEM_CRYSTAL_LEGS,
    [GEAR_SLOT_HANDS]  = ITEM_BARROWS_GLOVES,
    [GEAR_SLOT_FEET]   = ITEM_ECHO_BOOTS,
    [GEAR_SLOT_RING]   = ITEM_VENATOR_RING,
};

static const uint8_t* const INF_MAX_LOADOUTS[INF_NUM_WEAPON_SETS] = {
    INF_MAX_MAGE_LOADOUT,
    INF_MAX_RANGE_LONG_LOADOUT,
    INF_MAX_RANGE_FAST_LOADOUT,
};

static const uint8_t* const INF_BUDGET_LOADOUTS[INF_NUM_WEAPON_SETS] = {
    INF_BUDGET_MAGE_LOADOUT,
    INF_BUDGET_RANGE_LONG_LOADOUT,
    INF_BUDGET_RANGE_FAST_LOADOUT,
};

static const uint8_t* const* const INF_LOADOUT_PROFILES[INF_NUM_LOADOUT_PROFILES] = {
    [INF_LOADOUT_PROFILE_MAX] = INF_MAX_LOADOUTS,
    [INF_LOADOUT_PROFILE_BUDGET] = INF_BUDGET_LOADOUTS,
};

static const uint8_t* const* inf_loadouts_for_profile(int profile) {
    if (profile < 0 || profile >= INF_NUM_LOADOUT_PROFILES) {
        fprintf(stderr, "inferno: invalid active_loadout_profile %d\n", profile);
        abort();
    }
    return INF_LOADOUT_PROFILES[profile];
}

enum {
    INF_ZUK_HEALER_REWARD_MODE_BASELINE = 0,
    INF_ZUK_HEALER_REWARD_MODE_TAGS_FIRST = 1,
};

enum {
    INF_JOSEPH_REWARD_MODE_OFF = 0,
    INF_JOSEPH_REWARD_MODE_ON = 1,
};

enum {
    INF_CURRICULUM_SUPPLY_MODE_OFF = 0,
    INF_CURRICULUM_SUPPLY_MODE_ALL = 1,
    INF_CURRICULUM_SUPPLY_MODE_ZUK = 2,
    INF_CURRICULUM_SUPPLY_MODE_PRE_ZUK = 3,
};

typedef enum {
    INF_IDLE_PHASE_SET = 0,
    INF_IDLE_PHASE_JAD = 1,
    INF_IDLE_PHASE_ZUK_PRE_JAD = 2,
    INF_IDLE_PHASE_ZUK_JAD = 3,
    INF_IDLE_PHASE_ZUK_HEALERS = 4,
    INF_IDLE_PHASE_ZUK_POST_HEALERS = 5,
} InfIdleDiagnosticPhase;

typedef struct {
    int brew_doses;
    int restore_doses;
    int bastion_doses;
    int stamina_doses;
} InfSupplyDoses;

typedef struct {
    float brew_fraction;
    float restore_fraction;
    float bastion_fraction;
    float stamina_fraction;
} InfSupplyFractions;

typedef struct {
    int public_wave;
    InfSupplyFractions fractions;
} InfSupplyProfileAnchor;

typedef struct {
    int start_wave;
    InfLoadoutProfileMode loadout_profile_mode;
    float budget_loadout_fraction;
    float damage_reward_coeff;
    float offensive_prayer_reward_coeff;
    float shield_penalty_coeff;
    float tag_reward_coeff;
    float late_start_supply_profile_scale;
    int curriculum_agent;
    int curriculum_supply_jitter_mode;
    float curriculum_supply_shared_jitter;
    float curriculum_supply_brew_jitter;
    float curriculum_supply_restore_jitter;
    int curriculum_no_brew_mode;
    float curriculum_no_brew_frac;
    float supply_milestone_brew_reward_coeff;
    float supply_milestone_restore_reward_coeff;
    float death_penalty_coeff;
    int terminal_penalty_enabled;
    float phase_900_bonus;
    float phase_600_bonus;
    float phase_300_bonus;
    float shield_penalty_episode_cap;

    float jad_damage_reward_coeff;
    float zuk_healer_damage_reward_coeff;
    float set_damage_reward_coeff;
    float jad_kill_bonus;
    float zuk_healer_kill_bonus;
    float set_kill_bonus;
    float post_healer_zuk_damage_coeff;
    float post_healer_set_damage_reward_coeff;
    float post_healer_set_kill_bonus;
    float post_healer_set_alive_tick_penalty_coeff;
    float post_healer_set_alive_penalty_cap;
    float zuk_healer_phase_hp_delta_coeff;
    float zuk_untagged_healer_tick_penalty_coeff;
    float zuk_untagged_healer_target_bonus_coeff;
    float zuk_safe_untagged_healer_target_bonus_coeff;
    float zuk_untagged_healer_nonmagic_attack_bonus_coeff;
    float zuk_healer_mage_attack_penalty_coeff;
    int zuk_safe_untagged_healer_target_mask;
    int zuk_force_safe_untagged_healer_target_mask;
    int zuk_healer_reward_mode;
    float post_jad_zuk_multiplier;
    float jad_alive_zuk_multiplier;
    float shield_tag_reward_coeff;
    int joseph_reward_mode;
} InfConfig;

typedef struct {
    InfConfig config;
    const CollisionMap* collision_map;
    const EncounterArenaTopology* route_topology;
    OsrsActorRouteCache player_route_cache;
    int8_t npc_player_los_frame[INF_MAX_NPCS];
    int world_offset_x;
    int world_offset_y;
    Log* log;
    const HumanCommand* human_commands;
    int human_command_count;
    int human_command_mode;
} InfernoContext;

typedef struct {
    float damage_dealt;
    float damage_resurrected;
    float damage_zuk;
    float damage_zuk_healers;
    float damage_jad_healers;
    float damage_jad;
    float damage_set;
    float shield_damage;
    float spark_damage;
    float damage_received;
    float hp_restored;
    float hp_restored_jad;
    float hp_restored_zuk;
    float offensive_prayer_correct_damage_roll;
    int offensive_prayer_correct;
    int kill_jad;
    int kill_zuk_healer;
    int kill_set;
    int healer_tags;
    int zuk_healer_tags;
    int shield_tags;
    int zuk_untagged_healer_targets;
    int zuk_safe_untagged_healer_targets;
    int zuk_untagged_healer_nonmagic_attacks;
    int zuk_healer_mage_attack_fires;
    int prayer_correct;
    int off_prayer_hits;
    int styles_fired;
    int attacks_fired;
    int wave_completed;
    int pillar_lost;
    int player_moved;
    int player_attacked;
    int behind_shield;
    int brewed;
    int blood_heal;
} InfTickScratch;
typedef struct {
    uint16_t content_code;
    uint8_t click_action;
    uint8_t consumable_kind;
} InfInventoryClickMemo;

static_assert(sizeof(InfInventoryClickMemo) == 4, "Inferno inventory click memo size");

typedef struct {
    Player player;

    InfTickScratch tick_scratch;

    InfNPC npcs[INF_MAX_NPCS];
    uint32_t next_npc_render_id;
    int current_obs_slots[INF_OBS_NPCS];
    uint8_t current_obs_slots_valid;
    InfPillar pillars[INF_NUM_PILLARS];
    InfZukState zuk;

    InfDeadMob dead_mobs[INF_MAX_DEAD_MOBS];
    int dead_mob_count;

    uint8_t reserved_topology_pillar_storage[
        sizeof(int) * (4 * INF_NUM_PILLARS + 1)];

    int wave;
    int wave_spawn_target;
    int tick;
    int wave_spawn_delay;
    int wave_ready_delay;
    int episode_over;
    InfOutcome winner;

    float reward;
    float episode_return;
    float min_zuk_hp_seen;
    int tick_at_le_300;
    int tick_at_le_240;
    int tick_at_le_150;
    int tick_at_zuk_healer_spawn;
    int tick_at_first_zuk_healer_tag;
    int tick_at_all_zuk_healers_tagged;
    int tick_at_all_zuk_healers_dead;
    int tick_at_first_zuk_hit_after_all_healers_dead;
    float damage_after_300;
    float damage_after_240;
    float damage_after_150;
    float damage_after_all_zuk_healers_dead;
    float post_healer_set_alive_penalty_total;
    float post_healer_set_damage_reward_total;
    float post_healer_set_kill_bonus_total;
    float post_healer_set_pressure_total;
    float zuk_hp_at_all_zuk_healers_dead;
    float hp_restored_after_240;
    float spark_damage_after_240;
    float zuk_hp_max_after_healer_spawn;
    int player_moved_last_tick;

    float total_damage_dealt;
    float total_zuk_healer_damage;
    float total_damage_received;
    float total_hp_restored;
    int total_zuk_healer_tags;
    int total_zuk_healer_kills;
    int total_zuk_untagged_healer_targets;
    int total_zuk_safe_untagged_healer_targets;
    int total_zuk_unsafe_untagged_healer_targets;
    int total_zuk_untagged_healer_target_rewards;
    int total_zuk_safe_untagged_healer_target_rewards;
    int ticks_without_action;
    int total_prayer_correct;
    int total_npc_attacks;
    int total_unavoidable_off;
    int total_offensive_prayer_attacks;
    int total_offensive_prayer_correct;
    int offensive_prayer_attacks_by_style[4];
    int offensive_prayer_correct_by_style[4];
    int total_ranger_mager_same_tick_attacks;
    int total_step_out_ranger_mager_same_tick_attacks;
    int prayer_correct_by_type[INF_NUM_NPC_TYPES];
    int attacks_by_type[INF_NUM_NPC_TYPES];
    float dmg_from_type[INF_NUM_NPC_TYPES];
    int last_hit_by_type;
    int killed_by_type[INF_NUM_NPC_TYPES];
    int total_idle_ticks;
    int total_attack_ready_no_attack_ticks;
    int total_target_available_no_attack_ticks;
    int total_safe_attack_opportunity_missed_ticks;
    int total_progressless_ticks;
    int attack_ready_no_attack_ticks_by_phase[OSRS_INFERNO_IDLE_PHASE_COUNT];
    int target_available_no_attack_ticks_by_phase[OSRS_INFERNO_IDLE_PHASE_COUNT];
    int safe_attack_opportunity_missed_ticks_by_phase[OSRS_INFERNO_IDLE_PHASE_COUNT];
    int progressless_ticks_by_phase[OSRS_INFERNO_IDLE_PHASE_COUNT];
    int total_brews_used;
    int total_blood_healed;
    int total_npc_kills;
    int total_gear_switches;

    int behind_shield_ticks;
    int total_zuk_ticks;
    int offshield_ticks_after_240;
    int offshield_ticks_after_all_zuk_healers_dead;

    OsrsInteraction interaction;
    int player_last_interaction_target_slot;
    int player_last_interaction_age;

    InfWeaponSet weapon_set;
    InfLoadoutProfile active_loadout_profile;
    EncounterLoadoutStats loadout_stats[INF_NUM_WEAPON_SETS];
    InfInventoryClickMemo inventory_click_memos[OSRS_INVENTORY_SIZE];
    EncounterLoadoutStats live_stats;
    EncounterLoadoutStats human_loadout_stats;
    int stamina_active_ticks;

    int player_attack_npc_idx;
    int player_attack_dmg;
    AttackStyle player_attack_style_id;
    EncounterProjectileTiming player_attack_timing;

    EncounterPendingHitQueue player_pending_hits;
    InfPendingSpark pending_sparks[INF_MAX_PENDING_SPARKS];
    InfNpcTargetHit npc_target_hits[INF_MAX_NPC_TARGET_HITS];

    int nibbler_target_pillar;

    int spawn_order[INF_NUM_SPAWN_POS];

    int player_dest_x, player_dest_y;

    uint8_t reserved_topology_npc_storage[INF_MAX_NPCS];
    uint8_t npc_collision_flags[INF_ARENA_WIDTH][INF_ARENA_HEIGHT];
    uint8_t reserved_topology_player_footprint_storage[
        INF_ARENA_WIDTH * INF_ARENA_HEIGHT];

    int start_wave;
    uint32_t rng_state;
    uint32_t supply_milestone_rewarded_mask;
    uint8_t phase_900_fired;
    uint8_t phase_600_fired;
    uint8_t phase_300_fired;
    float shield_penalty_episode_total;

    float pending_jad_kill_bonus;
    float pending_zuk_healer_kill_bonus;
    float pending_set_kill_bonus;
    uint32_t rewarded_zuk_healer_target_mask;
    uint8_t jad_killed_this_episode;

    int tick_at_first_zuk_healer_target;
    int tick_at_first_zuk_healer_attack;
    int total_zuk_healer_target_ticks;
    int total_zuk_healer_attack_fires;
    int total_zuk_healer_cannot_attack_ticks;
    int total_zuk_healer_cooldown_ticks;
    int total_zuk_healer_out_of_range_ticks;
    int total_zuk_healer_attackable_ticks;
    int total_action_mask_checks;
    int total_shield_tags;
    float total_hp_restored_jad;
    float total_hp_restored_zuk;
} InfernoState;
static_assert(sizeof(InfernoState) == 27012, "InfernoState serialized layout");
