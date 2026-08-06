#ifndef ENCOUNTER_ZULRAH_H
#define ENCOUNTER_ZULRAH_H

#include "../osrs_encounter.h"
#include "../osrs_encounter_player.h"
#include "../osrs_encounter_visual_events.h"
#include "../osrs_interaction.h"
#include "../osrs_inventory_actions.h"
#include "../osrs_types.h"
#include "../osrs_items.h"
#include "../osrs_combat.h"
#include "../osrs_combat_visuals.h"
#include "../osrs_special_attacks.h"
#include "../osrs_pvp_gear.h"
#include "../osrs_consumables.h"
#include "../osrs_player_consumables.h"
#include "../osrs_damage.h"
#include "../osrs_collision.h"
#include "../osrs_monsters_generated.h"
#include "../data/npc_models.h"
#include <stdlib.h>
#include <string.h>

#define ZUL_ARENA_SIZE    28
#define ZUL_NPC_SIZE      5

#define ZUL_PLATFORM_MIN  5
#define ZUL_PLATFORM_MAX  22

#define ZUL_POS_NORTH   0
#define ZUL_POS_SOUTH   1
#define ZUL_POS_EAST    2
#define ZUL_POS_WEST    3
#define ZUL_NUM_POSITIONS 4

static const int ZUL_POSITIONS[ZUL_NUM_POSITIONS][2] = {
    { 10, 12 },
    { 10,  1 },
    { 20, 10 },
    {  0, 10 },
};

#define ZUL_PLAYER_START_X  11
#define ZUL_PLAYER_START_Y  7

#define ZUL_MELEE_STARE_TICKS 3
#define ZUL_MELEE_INTERVAL    6
#define ZUL_MELEE_STUN_TICKS  5

#define ZUL_DAMAGE_CAP  50
#define ZUL_DAMAGE_CAP_MIN 45

#define ZUL_SURFACE_TICKS_INITIAL 3
#define ZUL_SURFACE_TICKS         2
#define ZUL_DIVE_PHASE_TICKS      3
#define ZUL_DIVE_ANIM_TICKS       2
#define ZUL_RANGED_ANIM_TICKS     6
#define ZUL_MAGIC_ANIM_TICKS      2
#define ZUL_TAIL_ANIM_TICKS       7
#define ZUL_DEATH_ANIM_TICKS      5

#define ZUL_MAX_CLOUDS     7
#define ZUL_MAX_SNAKELINGS 4
#define ZUL_CLOUD_SIZE     3
#define ZUL_CLOUD_DURATION 30
#define ZUL_CLOUD_DAMAGE_MIN 1
#define ZUL_CLOUD_DAMAGE_MAX 5

#define ZUL_SNAKELING_HP       1
#define ZUL_SNAKELING_SPEED    3
#define ZUL_SNAKELING_LIFESPAN 67

#define ZUL_VENOM_INTERVAL  30
#define ZUL_VENOM_START     6
#define ZUL_VENOM_MAX       20

#define ZUL_SPAWN_INTERVAL  3
#define ZUL_CLOUD_FLIGHT_1  3
#define ZUL_CLOUD_FLIGHT_2  4
#define ZUL_REWARD_WIN_DEFAULT 1.0f
#define ZUL_REWARD_LOSS_PENALTY_DEFAULT 0.0f
#define ZUL_REWARD_DAMAGE_DEALT_DEFAULT 0.02f
#define ZUL_REWARD_CORRECT_STYLE_DEFAULT 0.05f
#define ZUL_REWARD_DAMAGE_RECEIVED_PENALTY_DEFAULT 0.01f
#define ZUL_REWARD_CLOUD_OCCUPANCY_PENALTY_DEFAULT 0.08f
#define ZUL_SCORE_SPEED_BONUS_DEFAULT 0.3f
#define ZUL_TRIP_RESPAWN_DELAY_TICKS 12
#define ZUL_MAX_ATTACK_EVENTS 8
#define ZUL_MAX_CLOUD_EVENTS 4

#define ZUL_ANTIVENOM_DURATION   300
#define ZUL_ANTIVENOM_DOSES      4

#define ZUL_THRALL_MAX_HIT       3
#define ZUL_THRALL_SPEED         4
#define ZUL_THRALL_DURATION      99
#define ZUL_THRALL_COOLDOWN      17

#define ZUL_PLAYER_HP         99
#define ZUL_PLAYER_PRAYER     77
#define ZUL_PLAYER_FOOD       10
#define ZUL_PLAYER_KARAMBWAN  4
#define ZUL_PLAYER_RESTORE_DOSES 8
#define ZUL_MAX_TICKS         600

#define ZUL_NUM_SCALAR_OBS    118
#define ZUL_NUM_OBS           (ZUL_NUM_SCALAR_OBS + \
    OSRS_INVENTORY_SIZE * OSRS_INVENTORY_CELL_OBS_FEATURES + \
    NUM_GEAR_SLOTS * OSRS_EQUIPPED_SELF_OBS_FEATURES)
#define ZUL_NUM_ACTION_HEADS  (2 + NUM_GEAR_SLOTS + 2 + 2)

#define ZUL_OBS_NPC_SLOTS (1 + ZUL_MAX_SNAKELINGS)

#define ZUL_MOVE_DIM      ENCOUNTER_MOVE_ACTIONS
#define ZUL_PRIMARY_DIM   (ZUL_MOVE_DIM + ZUL_OBS_NPC_SLOTS)
#define ZUL_PRAYER_DIM    ENCOUNTER_OVERHEAD_DIM_PVE
#define ZUL_OFFENSIVE_DIM ENCOUNTER_OFFENSIVE_DIM
#define ZUL_INV_CLICK_HEADS (NUM_GEAR_SLOTS + 2)
#define ZUL_INV_CLICK_DIM (OSRS_INVENTORY_SIZE + 1)
#define ZUL_SPEC_DIM      3

#define ZUL_ACTION_MASK_SIZE (ZUL_PRIMARY_DIM + ZUL_PRAYER_DIM + \
    ZUL_INV_CLICK_HEADS * ZUL_INV_CLICK_DIM + ZUL_SPEC_DIM + ZUL_OFFENSIVE_DIM)

#define ZUL_HEAD_PRIMARY    0
#define ZUL_HEAD_PRAYER     1
#define ZUL_HEAD_EQUIP_BASE 2
#define ZUL_HEAD_EQUIP_SLOT(slot) (ZUL_HEAD_EQUIP_BASE + (slot))
#define ZUL_HEAD_EAT        (ZUL_HEAD_EQUIP_BASE + NUM_GEAR_SLOTS)
#define ZUL_HEAD_DRINK      (ZUL_HEAD_EAT + 1)
#define ZUL_HEAD_SPEC       (ZUL_HEAD_DRINK + 1)
#define ZUL_HEAD_OFFENSIVE  (ZUL_HEAD_SPEC + 1)

#define ZUL_PRIMARY_ATTACK_BASE ZUL_MOVE_DIM
#define ZUL_MOVE_STAY 0

typedef enum {
    ZUL_FORM_GREEN = 0,
    ZUL_FORM_RED,
    ZUL_FORM_BLUE,
} ZulrahForm;

static const int ZUL_FORM_MONSTER_IDX[] = {
    [ZUL_FORM_GREEN] = MON_ZULRAH_GREEN,
    [ZUL_FORM_RED]   = MON_ZULRAH_RED,
    [ZUL_FORM_BLUE]  = MON_ZULRAH_BLUE,
};

typedef enum {
    ZA_END = 0,
    ZA_RANGED,
    ZA_MAGIC_RANGED,
    ZA_MELEE,
    ZA_JAD_RM,
    ZA_JAD_MR,
    ZA_CLOUDS,
    ZA_SNAKELINGS,
    ZA_SNAKECLOUD_ALT,
    ZA_CLOUDSNAKE_ALT,
} ZulActionType;

typedef struct {
    uint8_t type;
    uint8_t count;
} ZulAction;

#define ZUL_MAX_PHASE_ACTIONS 6

typedef struct {
    uint8_t position;
    uint8_t form;
    uint8_t stand;
    uint8_t stall;
    uint8_t phase_ticks;
    ZulAction actions[ZUL_MAX_PHASE_ACTIONS];
} ZulRotationPhase;

#define ZUL_MAX_ROT_PHASES 13
#define ZUL_NUM_ROTATIONS  4

#define ZUL_STAND_SOUTHWEST       0
#define ZUL_STAND_WEST            1
#define ZUL_STAND_CENTER          2
#define ZUL_STAND_NORTHEAST_TOP   3
#define ZUL_STAND_NORTHEAST_BOT   4
#define ZUL_STAND_NORTHWEST_TOP   5
#define ZUL_STAND_NORTHWEST_BOT   6
#define ZUL_STAND_EAST_PILLAR_S   7
#define ZUL_STAND_EAST_PILLAR     8
#define ZUL_STAND_EAST_PILLAR_N   9
#define ZUL_STAND_EAST_PILLAR_N2 10
#define ZUL_STAND_WEST_PILLAR_S  11
#define ZUL_STAND_WEST_PILLAR    12
#define ZUL_STAND_WEST_PILLAR_N  13
#define ZUL_STAND_WEST_PILLAR_N2 14
#define ZUL_NUM_STAND_LOCATIONS  15
#define ZUL_STAND_NONE           255

static const int ZUL_STAND_COORDS[ZUL_NUM_STAND_LOCATIONS][2] = {
    {  8,  8 },
    {  6, 14 },
    { 13,  9 },
    { 18, 16 },
    { 17, 16 },
    {  6, 15 },
    {  8, 16 },
    { 16,  9 },
    { 16, 10 },
    { 16, 12 },
    { 16, 13 },
    {  8,  9 },
    {  8, 10 },
    {  8, 12 },
    {  8, 13 },
};

#define ZA(t,c) { (uint8_t)(t), (uint8_t)(c) }
#define ZE { 0, 0 }
#define _N ZUL_STAND_NONE

static const ZulRotationPhase ZUL_ROT1[11] = {
     { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_CLOUDS,4), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_NORTHEAST_TOP, _N, 21, { ZA(ZA_MELEE,2), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_BLUE,  ZUL_STAND_EAST_PILLAR_N, ZUL_STAND_EAST_PILLAR_S, 18, { ZA(ZA_MAGIC_RANGED,4), ZE } },
     { ZUL_POS_SOUTH,  ZUL_FORM_GREEN, ZUL_STAND_WEST_PILLAR_N, ZUL_STAND_WEST_PILLAR_N2, 39, { ZA(ZA_RANGED,5), ZA(ZA_SNAKELINGS,2), ZA(ZA_CLOUDS,2), ZA(ZA_SNAKELINGS,2), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_WEST_PILLAR_N, _N, 22, { ZA(ZA_MELEE,2), ZE } },
     { ZUL_POS_WEST,   ZUL_FORM_BLUE,  ZUL_STAND_WEST_PILLAR_S, ZUL_STAND_EAST_PILLAR_S, 20, { ZA(ZA_MAGIC_RANGED,5), ZE } },
     { ZUL_POS_SOUTH,  ZUL_FORM_GREEN, ZUL_STAND_EAST_PILLAR, _N, 28, { ZA(ZA_CLOUDS,3), ZA(ZA_SNAKELINGS,4), ZE } },
     { ZUL_POS_SOUTH,  ZUL_FORM_BLUE,  ZUL_STAND_EAST_PILLAR, ZUL_STAND_EAST_PILLAR_N2, 36, { ZA(ZA_MAGIC_RANGED,5), ZA(ZA_SNAKECLOUD_ALT,5), ZE } },
     { ZUL_POS_WEST,   ZUL_FORM_GREEN, ZUL_STAND_WEST_PILLAR_S, ZUL_STAND_EAST_PILLAR_S, 48, { ZA(ZA_JAD_RM,10), ZA(ZA_CLOUDS,4), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_NORTHEAST_TOP, _N, 21, { ZA(ZA_MELEE,2), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_RANGED,5), ZA(ZA_CLOUDS,4), ZE } },
};

static const ZulRotationPhase ZUL_ROT2[11] = {
     { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_CLOUDS,4), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_NORTHEAST_TOP, _N, 21, { ZA(ZA_MELEE,2), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_BLUE,  ZUL_STAND_EAST_PILLAR_N, ZUL_STAND_EAST_PILLAR_S, 18, { ZA(ZA_MAGIC_RANGED,4), ZE } },
     { ZUL_POS_WEST,   ZUL_FORM_GREEN, ZUL_STAND_WEST_PILLAR_S, _N, 28, { ZA(ZA_CLOUDS,3), ZA(ZA_SNAKELINGS,4), ZE } },
     { ZUL_POS_SOUTH,  ZUL_FORM_BLUE,  ZUL_STAND_WEST_PILLAR_N, ZUL_STAND_WEST_PILLAR_N2, 39, { ZA(ZA_MAGIC_RANGED,5), ZA(ZA_SNAKELINGS,2), ZA(ZA_CLOUDS,2), ZA(ZA_SNAKELINGS,2), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_WEST_PILLAR_N, _N, 21, { ZA(ZA_MELEE,2), ZE } },
     { ZUL_POS_EAST,   ZUL_FORM_GREEN, ZUL_STAND_CENTER, ZUL_STAND_WEST_PILLAR_S, 20, { ZA(ZA_RANGED,5), ZE } },
     { ZUL_POS_SOUTH,  ZUL_FORM_BLUE,  ZUL_STAND_WEST_PILLAR_S, ZUL_STAND_WEST_PILLAR_N2, 36, { ZA(ZA_MAGIC_RANGED,5), ZA(ZA_SNAKECLOUD_ALT,5), ZE } },
     { ZUL_POS_WEST,   ZUL_FORM_GREEN, ZUL_STAND_WEST_PILLAR_S, ZUL_STAND_EAST_PILLAR_S, 48, { ZA(ZA_JAD_RM,10), ZA(ZA_CLOUDS,4), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_NORTHEAST_TOP, _N, 21, { ZA(ZA_MELEE,2), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_RANGED,5), ZA(ZA_CLOUDS,4), ZE } },
};

static const ZulRotationPhase ZUL_ROT3[12] = {
     { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_CLOUDS,4), ZE } },
     { ZUL_POS_EAST,   ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 30, { ZA(ZA_RANGED,5), ZA(ZA_SNAKELINGS,3), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_WEST, _N, 40, { ZA(ZA_CLOUDSNAKE_ALT,6), ZA(ZA_MELEE,2), ZE } },
     { ZUL_POS_WEST,   ZUL_FORM_BLUE,  ZUL_STAND_WEST, ZUL_STAND_EAST_PILLAR_S, 20, { ZA(ZA_MAGIC_RANGED,5), ZE } },
     { ZUL_POS_SOUTH,  ZUL_FORM_GREEN, ZUL_STAND_EAST_PILLAR_S, ZUL_STAND_EAST_PILLAR_N2, 20, { ZA(ZA_RANGED,5), ZE } },
     { ZUL_POS_EAST,   ZUL_FORM_BLUE,  ZUL_STAND_EAST_PILLAR_S, ZUL_STAND_WEST_PILLAR_S, 20, { ZA(ZA_MAGIC_RANGED,5), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_WEST_PILLAR_N, _N, 25, { ZA(ZA_CLOUDS,3), ZA(ZA_SNAKELINGS,3), ZE } },
     { ZUL_POS_WEST,   ZUL_FORM_GREEN, ZUL_STAND_WEST_PILLAR_N, _N, 20, { ZA(ZA_RANGED,5), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_BLUE,  ZUL_STAND_EAST_PILLAR_N, ZUL_STAND_EAST_PILLAR_S, 36, { ZA(ZA_MAGIC_RANGED,5), ZA(ZA_CLOUDS,2), ZA(ZA_SNAKELINGS,3), ZE } },
     { ZUL_POS_EAST,   ZUL_FORM_GREEN, ZUL_STAND_EAST_PILLAR_N, _N, 35, { ZA(ZA_JAD_MR,10), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_BLUE,  ZUL_STAND_NORTHEAST_TOP, _N, 18, { ZA(ZA_SNAKELINGS,4), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_RANGED,5), ZA(ZA_CLOUDS,4), ZE } },
};

static const ZulRotationPhase ZUL_ROT4[13] = {
     { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_CLOUDS,4), ZE } },
     { ZUL_POS_EAST,   ZUL_FORM_BLUE,  ZUL_STAND_NORTHEAST_TOP, _N, 36, { ZA(ZA_SNAKELINGS,4), ZA(ZA_MAGIC_RANGED,6), ZE } },
     { ZUL_POS_SOUTH,  ZUL_FORM_GREEN, ZUL_STAND_WEST_PILLAR_N, ZUL_STAND_WEST_PILLAR_N2, 24, { ZA(ZA_RANGED,4), ZA(ZA_CLOUDS,2), ZE } },
     { ZUL_POS_WEST,   ZUL_FORM_BLUE,  ZUL_STAND_WEST_PILLAR_N, _N, 30, { ZA(ZA_SNAKELINGS,4), ZA(ZA_MAGIC_RANGED,4), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_RED,   ZUL_STAND_EAST_PILLAR_N, _N, 28, { ZA(ZA_MELEE,2), ZA(ZA_CLOUDS,2), ZE } },
     { ZUL_POS_EAST,   ZUL_FORM_GREEN, ZUL_STAND_EAST_PILLAR, _N, 17, { ZA(ZA_RANGED,4), ZE } },
     { ZUL_POS_SOUTH,  ZUL_FORM_GREEN, ZUL_STAND_EAST_PILLAR, _N, 34, { ZA(ZA_SNAKELINGS,6), ZA(ZA_CLOUDS,3), ZE } },
     { ZUL_POS_WEST,   ZUL_FORM_BLUE,  ZUL_STAND_WEST_PILLAR_S, _N, 33, { ZA(ZA_MAGIC_RANGED,5), ZA(ZA_SNAKELINGS,4), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_EAST_PILLAR_N, ZUL_STAND_EAST_PILLAR_S, 20, { ZA(ZA_RANGED,4), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_BLUE,  ZUL_STAND_EAST_PILLAR_N, ZUL_STAND_EAST_PILLAR_S, 27, { ZA(ZA_MAGIC_RANGED,4), ZA(ZA_CLOUDS,3), ZE } },
     { ZUL_POS_EAST,   ZUL_FORM_GREEN, ZUL_STAND_EAST_PILLAR_N, _N, 29, { ZA(ZA_JAD_MR,8), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_BLUE,  ZUL_STAND_NORTHEAST_TOP, _N, 18, { ZA(ZA_SNAKELINGS,4), ZE } },
     { ZUL_POS_NORTH, ZUL_FORM_GREEN, ZUL_STAND_NORTHEAST_TOP, _N, 28, { ZA(ZA_RANGED,5), ZA(ZA_CLOUDS,4), ZE } },
};

#undef _N

static const ZulRotationPhase* const ZUL_ROTATIONS[ZUL_NUM_ROTATIONS] = {
    ZUL_ROT1, ZUL_ROT2, ZUL_ROT3, ZUL_ROT4,
};
static const int ZUL_ROT_LENGTHS[ZUL_NUM_ROTATIONS] = { 11, 11, 12, 13 };

#undef ZA
#undef ZE

#define ZUL_ACTION_DIMS_INIT { \
    ZUL_PRIMARY_DIM, \
    ZUL_PRAYER_DIM, \
    ZUL_INV_CLICK_DIM, ZUL_INV_CLICK_DIM, ZUL_INV_CLICK_DIM, ZUL_INV_CLICK_DIM, \
    ZUL_INV_CLICK_DIM, ZUL_INV_CLICK_DIM, ZUL_INV_CLICK_DIM, ZUL_INV_CLICK_DIM, \
    ZUL_INV_CLICK_DIM, ZUL_INV_CLICK_DIM, ZUL_INV_CLICK_DIM, \
    ZUL_INV_CLICK_DIM, \
    ZUL_INV_CLICK_DIM, \
    ZUL_SPEC_DIM, \
    ZUL_OFFENSIVE_DIM \
}
static const int ZUL_ACTION_HEAD_DIMS[ZUL_NUM_ACTION_HEADS] = ZUL_ACTION_DIMS_INIT;
static_assert(ZUL_HEAD_OFFENSIVE == ZUL_NUM_ACTION_HEADS - 1,
    "OFFENSIVE must be the last zulrah action head");

#define ZUL_NUM_GEAR_TIERS 3
#define ZUL_GEAR_TIER_FIXED 0
#define ZUL_GEAR_TIER_UNIFORM 1
#define ZUL_GEAR_TIER_WEIGHTED 2

static const uint8_t ZUL_MAGE_LOADOUT[ZUL_NUM_GEAR_TIERS][NUM_GEAR_SLOTS] = {
    { ITEM_MYSTIC_HAT, ITEM_GOD_CAPE, ITEM_GLORY, ITEM_AMETHYST_ARROW,
      ITEM_TRIDENT_OF_SWAMP, ITEM_BOOK_OF_DARKNESS, ITEM_MYSTIC_TOP, ITEM_MYSTIC_BOTTOM,
      ITEM_BARROWS_GLOVES, ITEM_MYSTIC_BOOTS, ITEM_RING_OF_RECOIL },
    { ITEM_AHRIMS_HOOD, ITEM_GOD_CAPE, ITEM_OCCULT_NECKLACE, ITEM_GOD_BLESSING,
      ITEM_SANGUINESTI_STAFF, ITEM_MAGES_BOOK, ITEM_AHRIMS_ROBETOP, ITEM_AHRIMS_ROBESKIRT,
      ITEM_TORMENTED_BRACELET, ITEM_INFINITY_BOOTS, ITEM_RING_OF_RECOIL },
    { ITEM_ANCESTRAL_HAT, ITEM_IMBUED_SARA_CAPE, ITEM_OCCULT_NECKLACE, ITEM_DRAGON_ARROWS,
      ITEM_EYE_OF_AYAK, ITEM_ELIDINIS_WARD_F, ITEM_ANCESTRAL_TOP, ITEM_ANCESTRAL_BOTTOM,
      ITEM_CONFLICTION_GAUNTLETS, ITEM_AVERNIC_TREADS, ITEM_RING_OF_SUFFERING_RI },
};

static const uint8_t ZUL_RANGE_LOADOUT[ZUL_NUM_GEAR_TIERS][NUM_GEAR_SLOTS] = {
    { ITEM_BLESSED_COIF, ITEM_AVAS_ACCUMULATOR, ITEM_GLORY, ITEM_AMETHYST_ARROW,
      ITEM_MAGIC_SHORTBOW_I, ITEM_NONE, ITEM_BLACK_DHIDE_BODY, ITEM_BLACK_DHIDE_CHAPS,
      ITEM_BARROWS_GLOVES, ITEM_MYSTIC_BOOTS, ITEM_RING_OF_RECOIL },
    { ITEM_CRYSTAL_HELM, ITEM_AVAS_ASSEMBLER, ITEM_NECKLACE_OF_ANGUISH, ITEM_GOD_BLESSING,
      ITEM_BOW_OF_FAERDHINEN, ITEM_NONE, ITEM_CRYSTAL_BODY, ITEM_CRYSTAL_LEGS,
      ITEM_BARROWS_GLOVES, ITEM_BLESSED_DHIDE_BOOTS, ITEM_RING_OF_RECOIL },
    { ITEM_MASORI_MASK_F, ITEM_DIZANAS_QUIVER, ITEM_NECKLACE_OF_ANGUISH, ITEM_DRAGON_ARROWS,
      ITEM_TWISTED_BOW, ITEM_NONE, ITEM_MASORI_BODY_F, ITEM_MASORI_CHAPS_F,
      ITEM_ZARYTE_VAMBRACES, ITEM_AVERNIC_TREADS, ITEM_RING_OF_SUFFERING_RI },
};

static void zul_populate_player_inventory(Player* p, int gear_tier) {
    const uint8_t* loadouts[] = {
        ZUL_MAGE_LOADOUT[gear_tier],
        ZUL_RANGE_LOADOUT[gear_tier],
    };
    encounter_populate_inventory(p, loadouts, 2, NULL);
}

#define ZUL_NUM_SNAKELING_POSITIONS 5
static const int ZUL_SNAKELING_POSITIONS[ZUL_NUM_SNAKELING_POSITIONS][2] = {
    { 7, 14 }, { 7, 10 }, { 12, 8 }, { 17, 10 }, { 17, 16 },
};

static int zul_cloud_overlaps_safe_center(int x, int y, int sx, int sy) {
    return x <= sx + 1 && x + ZUL_CLOUD_SIZE - 1 >= sx - 1 &&
           y <= sy + 1 && y + ZUL_CLOUD_SIZE - 1 >= sy - 1;
}

static int zul_cloud_overlaps_safe_area(int x, int y, int stand_id, int stall_id) {
    if (stand_id < ZUL_NUM_STAND_LOCATIONS) {
        int sx = ZUL_STAND_COORDS[stand_id][0];
        int sy = ZUL_STAND_COORDS[stand_id][1];
        if (zul_cloud_overlaps_safe_center(x, y, sx, sy)) return 1;
    }
    if (stall_id < ZUL_NUM_STAND_LOCATIONS) {
        int sx = ZUL_STAND_COORDS[stall_id][0];
        int sy = ZUL_STAND_COORDS[stall_id][1];
        if (zul_cloud_overlaps_safe_center(x, y, sx, sy)) return 1;
    }
    return 0;
}

typedef struct {
    int x, y;
    int active;
    int ticks_remaining;
} ZulrahCloud;

#define ZUL_MAX_PENDING_CLOUDS 16
typedef struct {
    int x, y;
    int delay;
} ZulrahPendingCloud;

typedef struct {
    Player entity;
    uint32_t npc_instance_id;
    int active;
    int attack_timer;
    int is_magic;
    int lifespan;
} ZulrahSnakeling;

typedef struct {
    float win;
    float loss_penalty;
    float damage_dealt;
    float correct_style;
    float damage_received_penalty;
    float cloud_occupancy_penalty;
} ZulrahRewardConfig;

typedef enum {
    ZUL_EPISODE_SINGLE_KILL = 0,
    ZUL_EPISODE_TRIP = 1,
} ZulrahEpisodeMode;

typedef enum {
    ZUL_OUTCOME_PLAYER_WON = 0,
    ZUL_OUTCOME_PLAYER_DIED = 1,
} ZulOutcome;

typedef struct {
    Player player;
    Player zulrah;
    uint32_t zulrah_npc_instance_id;
    uint32_t next_npc_instance_id;

    int rotation_index;
    int phase_index;

    int action_index;
    int action_progress;
    int action_timer;
    int jad_is_magic_next;

    ZulrahForm current_form;
    int zulrah_visible;
    int zulrah_attacking;

    int melee_target_x, melee_target_y;
    int melee_pending;
    int melee_stare_timer;
    OverheadPrayer melee_prayer_at_calc;

    EncounterPendingHitQueue player_pending_hits;
    EncounterPendingHitQueue zulrah_pending_hits;

    int phase_timer;
    int surface_timer;
    int is_diving;
    int zulrah_anim_until_tick;
    int zulrah_anim_event_tick;
    int zulrah_death_ticks;

    int player_stunned_ticks;

    ZulrahCloud clouds[ZUL_MAX_CLOUDS];
    ZulrahPendingCloud pending_clouds[ZUL_MAX_PENDING_CLOUDS];
    ZulrahSnakeling snakelings[ZUL_MAX_SNAKELINGS];

    OsrsInventoryCell inventory_cells[OSRS_INVENTORY_SIZE];
    OsrsInteraction interaction;
    int player_dest_x, player_dest_y;
    int player_dest_explicit;
    int player_moved_this_tick;
    int player_chased_target_this_tick;

    int venom_counter;
    int venom_timer;
    int antivenom_timer;

    int gear_tier;
    int gear_tier_fixed;
    int gear_tier_mode;
    int episode_mode;
    float gear_tier_weights[ZUL_NUM_GEAR_TIERS];
    ZulrahRewardConfig reward_config;

    EncounterLoadoutStats live_stats;
    int live_stats_dirty;
    int human_command_mode;
    const HumanCommand* human_commands;
    int human_command_count;

    int magic_def_drain;

    int thrall_active;
    int thrall_attack_timer;
    int thrall_duration_remaining;
    int thrall_cooldown;

    void* collision_map;
    int world_offset_x;
    int world_offset_y;

    int tick;
    int episode_over;
    ZulOutcome winner;
    int kills_this_episode;
    int boss_killed_this_tick;
    int player_lost_this_tick;
    int kill_start_tick;
    int respawn_timer;
    float score_speed_bonus_sum;
    uint32_t rng_state;

    float reward;
    float episode_return;
    float damage_dealt_this_tick;
    float damage_received_this_tick;
    int cloud_occupancy_this_tick;
    float total_damage_dealt;
    float total_damage_received;
    int total_cloud_occupancy_ticks;
    int total_active_cloud_ticks;
    int total_pending_cloud_ticks;
    float total_cloud_damage_received;

    int player_attacked_this_tick;
    int player_attack_dmg;
    int player_attack_style_id;
    int player_attack_is_special;
    EncounterProjectileTiming player_attack_timing;

    struct {
        int src_x, src_y, dst_x, dst_y;
        int style;
        int damage;
    } attack_events[ZUL_MAX_ATTACK_EVENTS];
    int attack_event_count;

    struct {
        int src_x, src_y, dst_x, dst_y;
        int flight_ticks;
    } cloud_events[ZUL_MAX_CLOUD_EVENTS];
    int cloud_event_count;

    Log log;
} ZulrahState;

static void zul_set_npc_anim_event(ZulrahState* s, int anim_id, int duration_ticks) {
    osrs_npc_primary_anim_event_set(
        &s->zulrah,
        &s->zulrah_anim_event_tick,
        &s->zulrah_anim_until_tick,
        s->tick,
        anim_id,
        duration_ticks);
}

static int zul_should_emit_npc_anim_event(const ZulrahState* s) {
    return osrs_npc_primary_anim_event_should_emit(
        &s->zulrah, s->zulrah_anim_event_tick, s->tick);
}

static int zul_attack_anim_for_style(ZulrahState* s, int style) {
    if (style == 2) {
        int player_center_x = s->player.x;
        int zulrah_center_x = s->zulrah.x + ZUL_NPC_SIZE / 2;
        return player_center_x < zulrah_center_x
            ? ZULRAH_ANIM_TAIL_LEFT
            : ZULRAH_ANIM_TAIL_RIGHT;
    }
    if (style == 1) return ZULRAH_ANIM_ATTACK_MAGIC;
    return ZULRAH_ANIM_ATTACK;
}

static int zul_attack_anim_ticks_for_style(int style) {
    if (style == 2) return ZUL_TAIL_ANIM_TICKS;
    if (style == 1) return ZUL_MAGIC_ANIM_TICKS;
    return ZUL_RANGED_ANIM_TICKS;
}

static inline EncounterProjectileTiming zul_player_projectile_timing(
    int style, uint8_t weapon, int is_special, int distance
) {
    EncounterProjectileDelayKind kind = encounter_projectile_delay_kind_for_style((AttackStyle)style);
    EncounterProjectileDelayOptions options = {0};
    if (style == ATTACK_STYLE_MAGIC && weapon == ITEM_EYE_OF_AYAK) {
        kind = ENCOUNTER_PROJECTILE_DELAY_EYE_OF_AYAK;
    }
    if (style == ATTACK_STYLE_RANGED) {
        if (weapon == ITEM_TOXIC_BLOWPIPE) {
            kind = ENCOUNTER_PROJECTILE_DELAY_THROWN;
            options.visual_delay_ticks = 1;
            if (is_special) {
                options.reduce_delay = -1;
                options.visual_hit_early_ticks = 1;
            }
        } else if (weapon == ITEM_TWISTED_BOW) {
            options.visual_delay_ticks = 1;
        }
    }
    return encounter_projectile_timing(distance, 1, kind, options);
}

static void zul_record_player_attack_visual(
    ZulrahState* s, int style, int damage, int is_special
) {
    int distance = encounter_projectile_distance(
        s->player.x, s->player.y, 1,
        s->zulrah.x, s->zulrah.y, ZUL_NPC_SIZE,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    s->player_attacked_this_tick = 1;
    s->player_attack_dmg = damage;
    s->player_attack_style_id = style;
    s->player_attack_is_special = is_special;
    s->player_attack_timing = zul_player_projectile_timing(
        style, s->player.equipped[GEAR_SLOT_WEAPON], is_special, distance);
}

static uint32_t zul_next_npc_instance_id(ZulrahState* s) {
    s->next_npc_instance_id++;
    if (s->next_npc_instance_id == 0) {
        s->next_npc_instance_id = 1;
    }
    return s->next_npc_instance_id;
}

static void zul_update_npc_anim_lifetime(ZulrahState* s) {
    if (!s->zulrah_visible) return;
    if (s->tick < s->zulrah_anim_until_tick) return;
    if (s->is_diving || s->surface_timer > 0) return;
    osrs_npc_primary_anim_event_expire(
        &s->zulrah, s->zulrah_anim_until_tick, s->tick);
    if (s->zulrah.npc_anim_id < 0)
        s->zulrah_anim_event_tick = -1;
}

static const EncounterLoadoutStats* zul_live_stats(ZulrahState* s);

static int zul_lookup_player_attack_target(
    void* ctx,
    int target_slot,
    OsrsAttackTarget* out
) {
    ZulrahState* s = (ZulrahState*)ctx;
    const EncounterLoadoutStats* ls = zul_live_stats(s);
    if (target_slot == 0) {
        if (!s->zulrah_visible || s->is_diving) return 0;
        *out = (OsrsAttackTarget){
            .slot = 0,
            .x = s->zulrah.x,
            .y = s->zulrah.y,
            .size = ZUL_NPC_SIZE,
            .attack_range = ls->attack_range,
        };
        return 1;
    }
    int snakeling_idx = target_slot - 1;
    if (snakeling_idx < 0 || snakeling_idx >= ZUL_MAX_SNAKELINGS) return 0;
    const ZulrahSnakeling* sn = &s->snakelings[snakeling_idx];
    if (!sn->active) return 0;
    *out = (OsrsAttackTarget){
        .slot = target_slot,
        .x = sn->entity.x,
        .y = sn->entity.y,
        .size = 1,
        .attack_range = ls->attack_range,
    };
    return 1;
}

static int zul_sample_gear_tier(ZulrahState* s) {
    if (s->gear_tier_mode == ZUL_GEAR_TIER_FIXED) {
        return s->gear_tier_fixed;
    }

    if (s->gear_tier_mode == ZUL_GEAR_TIER_UNIFORM) {
        return encounter_rand_int(&s->rng_state, ZUL_NUM_GEAR_TIERS);
    }

    float total = 0.0f;
    for (int i = 0; i < ZUL_NUM_GEAR_TIERS; i++) {
        if (s->gear_tier_weights[i] < 0.0f) {
            fprintf(stderr, "zulrah gear_tier_weight_%d must be >= 0, got %.6f\n",
                i, s->gear_tier_weights[i]);
            abort();
        }
        total += s->gear_tier_weights[i];
    }

    if (total <= 0.0f) {
        fprintf(stderr, "zulrah weighted gear tier mode requires positive total weight\n");
        abort();
    }

    float threshold = encounter_rand_float(&s->rng_state) * total;
    for (int i = 0; i < ZUL_NUM_GEAR_TIERS; i++) {
        threshold -= s->gear_tier_weights[i];
        if (threshold <= 0.0f) return i;
    }
    return ZUL_NUM_GEAR_TIERS - 1;
}

static inline int zul_on_platform_bounds(int x, int y) {
    return x >= ZUL_PLATFORM_MIN && x <= ZUL_PLATFORM_MAX &&
           y >= ZUL_PLATFORM_MIN && y <= ZUL_PLATFORM_MAX;
}

static inline int zul_on_platform(ZulrahState* s, int x, int y) {
    if (!s->collision_map) return zul_on_platform_bounds(x, y);
    int wx = x + s->world_offset_x;
    int wy = y + s->world_offset_y;
    return collision_tile_walkable((const CollisionMap*)s->collision_map, 0, wx, wy);
}

#define zul_pathfind(s, sx, sy, dx, dy) \
    encounter_pathfind((const CollisionMap*)(s)->collision_map, \
        (s)->world_offset_x, (s)->world_offset_y, (sx), (sy), (dx), (dy), NULL, NULL)

static int zul_tile_walkable(void* ctx, int x, int y) {
    return zul_on_platform((ZulrahState*)ctx, x, y);
}

static inline int zul_player_in_cloud(int cx, int cy, int px, int py) {
    return px >= cx && px < cx + ZUL_CLOUD_SIZE &&
           py >= cy && py < cy + ZUL_CLOUD_SIZE;
}

static int zul_active_cloud_count(const ZulrahState* s) {
    int count = 0;
    for (int i = 0; i < ZUL_MAX_CLOUDS; i++)
        count += s->clouds[i].active ? 1 : 0;
    return count;
}

static int zul_pending_cloud_count(const ZulrahState* s) {
    int count = 0;
    for (int i = 0; i < ZUL_MAX_PENDING_CLOUDS; i++)
        count += s->pending_clouds[i].delay > 0 ? 1 : 0;
    return count;
}

static int zul_rect_signed_delta_1d(int p, int lo, int size) {
    int hi = lo + size - 1;
    if (p < lo) return lo - p;
    if (p > hi) return hi - p;
    return 0;
}

static void zul_cloud_escape_delta(int cx, int cy, int px, int py, int* dx, int* dy) {
    int west = cx - 1 - px;
    int east = cx + ZUL_CLOUD_SIZE - px;
    int south = cy - 1 - py;
    int north = cy + ZUL_CLOUD_SIZE - py;
    int best_dx = west;
    int best_dy = 0;
    int best_abs = abs_int(west);
    if (abs_int(east) < best_abs) {
        best_dx = east;
        best_dy = 0;
        best_abs = abs_int(east);
    }
    if (abs_int(south) < best_abs) {
        best_dx = 0;
        best_dy = south;
        best_abs = abs_int(south);
    }
    if (abs_int(north) < best_abs) {
        best_dx = 0;
        best_dy = north;
    }
    *dx = best_dx;
    *dy = best_dy;
}

static int zul_nearest_active_cloud_features(
    const ZulrahState* s,
    int* signed_dx,
    int* signed_dy,
    int* escape_dx,
    int* escape_dy
) {
    int found = 0;
    int best_dist = 0;
    *signed_dx = 0;
    *signed_dy = 0;
    *escape_dx = 0;
    *escape_dy = 0;
    for (int i = 0; i < ZUL_MAX_CLOUDS; i++) {
        if (!s->clouds[i].active) continue;
        int dx = zul_rect_signed_delta_1d(s->player.x, s->clouds[i].x, ZUL_CLOUD_SIZE);
        int dy = zul_rect_signed_delta_1d(s->player.y, s->clouds[i].y, ZUL_CLOUD_SIZE);
        int dist = max_int(abs_int(dx), abs_int(dy));
        int inside = zul_player_in_cloud(
            s->clouds[i].x, s->clouds[i].y, s->player.x, s->player.y);
        if (!found || inside || dist < best_dist) {
            found = 1;
            best_dist = dist;
            *signed_dx = dx;
            *signed_dy = dy;
            if (inside)
                zul_cloud_escape_delta(
                    s->clouds[i].x, s->clouds[i].y,
                    s->player.x, s->player.y, escape_dx, escape_dy);
            else {
                *escape_dx = 0;
                *escape_dy = 0;
            }
            if (inside) break;
        }
    }
    return found;
}

static int zul_nearest_pending_cloud_features(
    const ZulrahState* s,
    int* dx,
    int* dy,
    int* delay
) {
    int found = 0;
    int best_dist = 0;
    *dx = 0;
    *dy = 0;
    *delay = 0;
    for (int i = 0; i < ZUL_MAX_PENDING_CLOUDS; i++) {
        if (s->pending_clouds[i].delay <= 0) continue;
        int pdx = s->pending_clouds[i].x - s->player.x;
        int pdy = s->pending_clouds[i].y - s->player.y;
        int dist = max_int(abs_int(pdx), abs_int(pdy));
        if (!found || dist < best_dist) {
            found = 1;
            best_dist = dist;
            *dx = pdx;
            *dy = pdy;
            *delay = s->pending_clouds[i].delay;
        }
    }
    return found;
}

static int zul_active_cloud_overlaps_safe(const ZulrahState* s, int safe_id) {
    if (safe_id >= ZUL_NUM_STAND_LOCATIONS) return 0;
    for (int i = 0; i < ZUL_MAX_CLOUDS; i++) {
        if (!s->clouds[i].active) continue;
        if (zul_cloud_overlaps_safe_area(
                s->clouds[i].x, s->clouds[i].y, safe_id, ZUL_STAND_NONE))
            return 1;
    }
    return 0;
}

static int zul_pending_cloud_overlaps_safe(const ZulrahState* s, int safe_id) {
    if (safe_id >= ZUL_NUM_STAND_LOCATIONS) return 0;
    for (int i = 0; i < ZUL_MAX_PENDING_CLOUDS; i++) {
        if (s->pending_clouds[i].delay <= 0) continue;
        if (zul_cloud_overlaps_safe_area(
                s->pending_clouds[i].x, s->pending_clouds[i].y,
                safe_id, ZUL_STAND_NONE))
            return 1;
    }
    return 0;
}

static int zul_tile_in_active_cloud(const ZulrahState* s, int x, int y) {
    for (int i = 0; i < ZUL_MAX_CLOUDS; i++) {
        if (!s->clouds[i].active) continue;
        if (zul_player_in_cloud(s->clouds[i].x, s->clouds[i].y, x, y))
            return 1;
    }
    return 0;
}

static int zul_tile_in_pending_cloud_by_delay(
    const ZulrahState* s,
    int x,
    int y,
    int max_delay
) {
    for (int i = 0; i < ZUL_MAX_PENDING_CLOUDS; i++) {
        if (s->pending_clouds[i].delay <= 0 ||
                s->pending_clouds[i].delay > max_delay)
            continue;
        if (zul_player_in_cloud(s->pending_clouds[i].x, s->pending_clouds[i].y, x, y))
            return 1;
    }
    return 0;
}

static int zul_move_action_cloud_unsafe(const ZulrahState* s, int action) {
    if (action < 0 || action >= ZUL_MOVE_DIM) {
        fprintf(stderr, "zulrah move action out of range for cloud obs: %d\n", action);
        abort();
    }
    int x = s->player.x + ENCOUNTER_MOVE_TARGET_DX[action];
    int y = s->player.y + ENCOUNTER_MOVE_TARGET_DY[action];
    return zul_tile_in_active_cloud(s, x, y) ||
        zul_tile_in_pending_cloud_by_delay(s, x, y, 1);
}

static int zul_form_npc_id(ZulrahForm f) {
    return MONSTER_DATABASE[ZUL_FORM_MONSTER_IDX[f]].npc_id;
}

static inline int zul_cap_damage(ZulrahState* s, int damage) {
    if (damage > ZUL_DAMAGE_CAP) {
        return ZUL_DAMAGE_CAP_MIN + encounter_rand_int(&s->rng_state, ZUL_DAMAGE_CAP - ZUL_DAMAGE_CAP_MIN + 1);
    }
    return damage;
}

static void zul_apply_recoil(ZulrahState* s, int damage, AttackStyle style,
                             Player* attacker) {
    osrs_ensure_player_equipment(&s->player);
    DamageResult damage_result = osrs_apply_passive_damage_pipeline(
        damage,
        style,
        s->player.prayer,
         0,
         0,
         0,
        &s->player.equipment_effect_profile,
        &s->player.item_effect_state,
        &s->rng_state
    );
    if (damage_result.recoil_damage > 0) {
        int recoil = damage_result.recoil_damage;
        if (s->player.equipment_effect_profile.recoil_source == OSRS_RECOIL_SOURCE_RING_OF_RECOIL &&
            recoil > s->player.item_effect_state.recoil_charges) {
            recoil = s->player.item_effect_state.recoil_charges;
        }
        encounter_damage_player(attacker, recoil, NULL);
        osrs_consume_recoil_charges(&s->player, recoil);
    }
}

static void zul_apply_player_damage(ZulrahState* s, int damage, AttackStyle style,
                                    Player* attacker) {
    if (damage <= 0) return;
    encounter_damage_player(&s->player, damage, &s->damage_received_this_tick);
    s->total_damage_received += damage;
    s->player.hit_style = style;

    if (attacker) zul_apply_recoil(s, damage, style, attacker);
}

/** Landing-side accounting for one player->Zulrah hit; returns 1 when it landed
    this tick. Shared by the instant (delay 0) and queued paths. */
static int zul_land_zulrah_hit(ZulrahState* s, EncounterPendingHit* ph) {
    int landed = 0;
    int hit_damage = 0;
    float dealt = 0.0f;
    if (!encounter_resolve_npc_pending_hit(
            ph, &s->zulrah.current_hitpoints, &landed, &hit_damage,
            NULL, NULL, &dealt))
        return 0;
    s->damage_dealt_this_tick += dealt;
    s->total_damage_dealt += dealt;
    s->zulrah.hit_landed_this_tick = 1;
    s->zulrah.hit_damage = hit_damage;
    s->zulrah.hit_was_successful = hit_damage > 0;
    return 1;
}

static void zul_queue_zulrah_hit(ZulrahState* s, int damage, AttackStyle style,
                                 int is_special) {
    int distance = encounter_projectile_distance(
        s->player.x, s->player.y, 1,
        s->zulrah.x, s->zulrah.y, ZUL_NPC_SIZE,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    int delay = zul_player_projectile_timing(
        style, s->player.equipped[GEAR_SLOT_WEAPON], is_special, distance)
        .damage_delay_ticks;

    EncounterPendingHit hit = {
        .active = 1,
        .ticks_remaining = (int8_t)delay,
        .attack_style = (int8_t)style,
        .check_prayer = 0,
        .spell_type = ENCOUNTER_SPELL_NONE,
        .source_npc_slot = -1,
        .damage = (int16_t)damage,
    };
    if (delay <= 0) {
        zul_land_zulrah_hit(s, &hit);
        return;
    }
    encounter_pending_hit_queue_push(
        &s->zulrah_pending_hits, hit, "zulrah-npc", s->tick, -1, 0);
}

static void zul_resolve_zulrah_pending_hits(ZulrahState* s) {
    for (int i = 0; i < s->zulrah_pending_hits.count; i++) {
        EncounterPendingHit* ph = &s->zulrah_pending_hits.hits[i];
        zul_land_zulrah_hit(s, ph);
        if (!ph->active) {
            encounter_pending_hit_queue_remove(&s->zulrah_pending_hits, i, "zulrah-npc");
            i--;
        }
    }
}

static void zul_try_envenom(ZulrahState* s) {
    if (s->venom_counter > 0) return;
    if (s->antivenom_timer > 0) return;
    if (encounter_rand_int(&s->rng_state, 4) != 0) return;
    s->venom_counter = 1;
    s->venom_timer = ZUL_VENOM_INTERVAL;
}

static int zul_player_def_roll(ZulrahState* s, int attack_style) {
    const EncounterLoadoutStats* ls = zul_live_stats(s);
    int roll = encounter_player_def_roll_from_loadout(
        99, 99,
        ls->def_stab, ls->def_slash, ls->def_crush, ls->def_magic, ls->def_ranged,
        attack_style, 2);
    return roll > 0 ? roll : 0;
}

static void zul_record_attack(ZulrahState* s, int src_x, int src_y,
                               int dst_x, int dst_y, int style, int damage) {
    int anim_id = zul_attack_anim_for_style(s, style);
    zul_set_npc_anim_event(s, anim_id, zul_attack_anim_ticks_for_style(style));
    if (s->attack_event_count >= ZUL_MAX_ATTACK_EVENTS) {
        fprintf(stderr, "zulrah attack event capacity exceeded: %d\n",
            ZUL_MAX_ATTACK_EVENTS);
        abort();
    }
    int i = s->attack_event_count++;
    s->attack_events[i].src_x = src_x;
    s->attack_events[i].src_y = src_y;
    s->attack_events[i].dst_x = dst_x;
    s->attack_events[i].dst_y = dst_y;
    s->attack_events[i].style = style;
    s->attack_events[i].damage = damage;
}

/** Queues one Zulrah->player hit. Protect-prayer and damage freeze on THIS tick
    (the calculation tick); only the application waits out the hit delay. */
static int zul_queue_player_hit(ZulrahState* s, int raw_damage, AttackStyle style,
                                int accuracy_hit) {
    int distance = encounter_projectile_distance(
        s->zulrah.x, s->zulrah.y, ZUL_NPC_SIZE,
        s->player.x, s->player.y, 1,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    int delay = encounter_projectile_base_hit_delay(
        distance, 0, encounter_projectile_delay_kind_for_style(style));

    int prayed = 0;
    EncounterPendingHit hit = encounter_pending_hit_resolved_at_throw(
        raw_damage, delay, style, s->player.prayer,
        s->zulrah.npc_def_id, -1, accuracy_hit, &prayed);
    encounter_pending_hit_queue_push(
        &s->player_pending_hits, hit, "zulrah-player", s->tick, -1, style);
    return prayed ? 0 : raw_damage;
}

static void zul_player_hit_landed(
    void* user, const EncounterPendingHit* hit, int damage_after_prayer,
    int damage_applied, int prayer_was_correct, int prayer_was_checked
) {
    (void)damage_applied;
    (void)prayer_was_correct;
    (void)prayer_was_checked;
    ZulrahState* s = (ZulrahState*)user;
    if (damage_after_prayer <= 0) return;
    s->total_damage_received += damage_after_prayer;
    s->player.hit_style = (AttackStyle)hit->attack_style;
    zul_apply_recoil(s, damage_after_prayer, (AttackStyle)hit->attack_style,
                     &s->zulrah);
}

static void zul_resolve_player_pending_hits(ZulrahState* s) {
    encounter_resolve_player_pending_hits_observed(
        &s->player_pending_hits, &s->player, s->player.prayer,
        &s->damage_received_this_tick, NULL, NULL,
        zul_player_hit_landed, s);
}

static void zul_attack_ranged(ZulrahState* s) {
    const MonsterStats* m = &MONSTER_DATABASE[MON_ZULRAH_GREEN];
    int npc_att_roll = osrs_npc_attack_roll(m->range_level, m->range_att_bonus);
    int def_roll = zul_player_def_roll(s, ATTACK_STYLE_RANGED);
    int did_hit = 0;
    int raw = encounter_npc_roll_attack_ex(
        npc_att_roll, def_roll, m->max_hit, 0, &s->rng_state, &did_hit);
    int dmg = zul_queue_player_hit(s, raw, ATTACK_STYLE_RANGED, did_hit);
    if (did_hit) zul_try_envenom(s);
    zul_record_attack(s, s->zulrah.x, s->zulrah.y,
                      s->player.x, s->player.y, 0, dmg);
}

static void zul_attack_magic(ZulrahState* s) {
    const MonsterStats* m = &MONSTER_DATABASE[MON_ZULRAH_BLUE];
    int npc_att_roll = osrs_npc_attack_roll(m->magic_level, m->magic_att_bonus);
    int def_roll = zul_player_def_roll(s, ATTACK_STYLE_MAGIC);
    int did_hit = 0;
    int raw = encounter_npc_roll_attack_ex(
        npc_att_roll, def_roll, m->max_hit, 0, &s->rng_state, &did_hit);
    int dmg = zul_queue_player_hit(s, raw, ATTACK_STYLE_MAGIC, did_hit);
    zul_try_envenom(s);
    zul_record_attack(s, s->zulrah.x, s->zulrah.y,
                      s->player.x, s->player.y, 1, dmg);
}

static void zul_attack_magic_ranged(ZulrahState* s) {
    if (encounter_rand_int(&s->rng_state, 4) < 3) {
        zul_attack_magic(s);
    } else {
        zul_attack_ranged(s);
    }
}

static int zul_on_pillar_safespot(int px, int py) {
    if (py != 11) return 0;
    return (px == 17 || px == 7);
}

static void zul_melee_start(ZulrahState* s) {
    s->melee_target_x = s->player.x;
    s->melee_target_y = s->player.y;
    s->melee_pending = 1;
    s->melee_stare_timer = ZUL_MELEE_STARE_TICKS;
    s->melee_prayer_at_calc = s->player.prayer;
}

static void zul_melee_hit(ZulrahState* s) {
    s->melee_pending = 0;
    int dmg = 0;
    if (s->player.x == s->melee_target_x && s->player.y == s->melee_target_y
        && !zul_on_pillar_safespot(s->player.x, s->player.y)) {
        if (!encounter_prayer_correct_for_style(s->melee_prayer_at_calc, ATTACK_STYLE_MELEE)) {
            dmg = 20 + encounter_rand_int(&s->rng_state, 11);
            zul_apply_player_damage(s, dmg, ATTACK_STYLE_MELEE, &s->zulrah);
            s->player_stunned_ticks = ZUL_MELEE_STUN_TICKS;
        }
    }
    zul_record_attack(s, s->zulrah.x, s->zulrah.y,
                      s->melee_target_x, s->melee_target_y, 2, dmg);
}

static void zul_attack_jad(ZulrahState* s) {
    if (s->jad_is_magic_next) {
        zul_attack_magic(s);
    } else {
        zul_attack_ranged(s);
    }
    s->jad_is_magic_next = !s->jad_is_magic_next;
}

static inline void zul_form_def_bonuses(ZulrahForm form, int* def_magic, int* def_ranged) {
    const MonsterStats* m = &MONSTER_DATABASE[ZUL_FORM_MONSTER_IDX[form]];
    *def_magic = m->magic_def;
    *def_ranged = m->ranged_def;
}

static AttackStyle zul_player_equipped_attack_style(const ZulrahState* s) {
    AttackStyle style = (AttackStyle)get_item_attack_style(s->player.equipped[GEAR_SLOT_WEAPON]);
    if (style == ATTACK_STYLE_MAGIC ||
        style == ATTACK_STYLE_RANGED ||
        style == ATTACK_STYLE_MELEE) {
        return style;
    }
    return ATTACK_STYLE_RANGED;
}

static void zul_mark_live_stats_dirty(ZulrahState* s) {
    s->live_stats_dirty = 1;
}



static const EncounterLoadoutStats* zul_live_stats(ZulrahState* s) {
    if (s->live_stats_dirty) {
        AttackStyle style = zul_player_equipped_attack_style(s);
        FightStyle fight_style = s->human_command_mode
            ? s->player.fight_style
            : (style == ATTACK_STYLE_RANGED ? FIGHT_STYLE_RAPID : FIGHT_STYLE_ACCURATE);
        int spell_base_damage = (style == ATTACK_STYLE_MAGIC) ? 30 : 0;
        encounter_compute_player_equipped_stats(
            &s->player, style, fight_style, spell_base_damage, &s->live_stats);
        s->live_stats_dirty = 0;
    }
    return &s->live_stats;
}

static int zul_player_can_attack_zulrah(
    ZulrahState* s,
    const EncounterLoadoutStats* loadout_stats
) {
    return encounter_player_can_attack(
        s->player.x, s->player.y,
        s->zulrah.x, s->zulrah.y, ZUL_NPC_SIZE,
        loadout_stats->attack_range, osrs_los_open_query());
}

static int zul_zulrah_def_roll(ZulrahState* s, int is_mage) {
    int def_magic = 0, def_ranged = 0;
    zul_form_def_bonuses(s->current_form, &def_magic, &def_ranged);
    if (is_mage) {
        def_magic -= s->magic_def_drain;
        if (def_magic < -64) def_magic = -64;
    }
    int def_bonus = is_mage ? def_magic : def_ranged;
    int def_roll = (MONSTER_DATABASE[ZUL_FORM_MONSTER_IDX[s->current_form]].def_level + 8) * (def_bonus + 64);
    if (def_roll < 0) def_roll = 0;
    return def_roll;
}

static int zul_player_attack_hits(
    ZulrahState* s, int is_mage, const OsrsPreparedAttackEffects* attack_effects
) {
    int att_roll = attack_effects->attack_roll;
    int def_roll = zul_zulrah_def_roll(s, is_mage);

    return attack_effects->use_double_accuracy
        ? encounter_roll_hit_chance_double(&s->rng_state, att_roll, def_roll)
        : encounter_roll_hit_chance(&s->rng_state, att_roll, def_roll);
}

static void zul_player_attack(ZulrahState* s) {
    if (!s->zulrah_visible || s->is_diving) return;
    if (s->player.attack_timer > 0) return;
    if (s->player_stunned_ticks > 0) return;

    const EncounterLoadoutStats* ls = zul_live_stats(s);
    if (!zul_player_can_attack_zulrah(s, ls)) return;

    AttackStyle style = zul_player_equipped_attack_style(s);
    int is_mage = (style == ATTACK_STYLE_MAGIC);
    const MonsterStats* monster = &MONSTER_DATABASE[ZUL_FORM_MONSTER_IDX[s->current_form]];
    OsrsMagicAttackKind magic_kind = is_mage ? OSRS_MAGIC_ATTACK_POWERED_STAFF : OSRS_MAGIC_ATTACK_NONE;
    OsrsPreparedAttackEffects attack_effects = osrs_prepare_attack_effects(
        &s->player.equipment_effect_profile,
        &s->player.item_effect_state,
        s->player.equipped[GEAR_SLOT_WEAPON],
        style,
        magic_kind,
        (OsrsTargetRef){ .kind = OSRS_TARGET_NPC, .id = 0 },
        1,
        osrs_player_att_roll(ls->eff_level, ls->attack_bonus),
        ls->max_hit,
        osrs_target_effect_context_magic(
            monster->magic_level,
            monster->magic_att_bonus),
        s->player.current_hitpoints,
        s->player.base_hitpoints
    );
    s->player.attack_timer = ls->attack_speed;

    int dmg = 0;
    int hit = zul_player_attack_hits(s, is_mage, &attack_effects);
    if (hit) {
        dmg = encounter_rand_int(&s->rng_state, attack_effects.max_hit + 1);
        dmg = zul_cap_damage(s, dmg);
    }
    zul_queue_zulrah_hit(s, dmg, style, 0);
    {
        OsrsPostAttackEffects post_effects = osrs_finalize_attack_effects(
            &s->player.equipment_effect_profile,
            &s->player.item_effect_state,
            s->player.equipped[GEAR_SLOT_WEAPON],
            style,
            magic_kind,
            (OsrsTargetRef){ .kind = OSRS_TARGET_NPC, .id = 0 },
            1,
            attack_effects.use_double_accuracy,
            hit,
            dmg,
            &s->rng_state
        );
        if (post_effects.heal_amount > 0) {
            s->player.current_hitpoints += post_effects.heal_amount;
            if (s->player.current_hitpoints > s->player.base_hitpoints)
                s->player.current_hitpoints = s->player.base_hitpoints;
        }
    }
    s->player.just_attacked = 1;
    s->player.last_attack_style = style;
    s->player.attack_style_this_tick = style;
    zul_record_player_attack_visual(
        s, s->player.attack_style_this_tick, dmg, 0);
}

static void zul_player_spec(ZulrahState* s) {
    if (!s->zulrah_visible || s->is_diving) return;
    if (s->player.attack_timer > 0) return;
    if (s->player_stunned_ticks > 0) return;

    AttackStyle style = zul_player_equipped_attack_style(s);
    int is_mage = (style == ATTACK_STYLE_MAGIC);
    const EncounterLoadoutStats* ls = zul_live_stats(s);
    if (!zul_player_can_attack_zulrah(s, ls)) return;

    int weapon = s->player.equipped[GEAR_SLOT_WEAPON];

    int cost = osrs_spec_cost(weapon);
    if (cost == 0) return;
    if (s->player.special_energy < cost) return;

    const MonsterStats* m = &MONSTER_DATABASE[ZUL_FORM_MONSTER_IDX[s->current_form]];
    int def_roll = zul_zulrah_def_roll(s, is_mage);

    int att_roll = osrs_player_att_roll(ls->eff_level, ls->attack_bonus);
    SpecResult sr = osrs_resolve_spec(weapon, att_roll, ls->max_hit,
                                       def_roll, m->def_level, &s->rng_state);

    s->player.special_energy -= sr.spec_cost;
    s->player.just_attacked = 1;
    s->player.used_special_this_tick = 1;
    s->player.last_attack_style = style;
    s->player.attack_style_this_tick = s->player.last_attack_style;
    s->player.attack_timer = sr.attack_speed_override ? sr.attack_speed_override : ls->attack_speed;
    zul_record_player_attack_visual(s, s->player.attack_style_this_tick, 0, 1);

    int total_dmg = 0;
    for (int i = 0; i < sr.num_hits; i++) {
        int dmg = zul_cap_damage(s, sr.damage[i]);
        zul_queue_zulrah_hit(s, dmg, style, 1);
        total_dmg += dmg;
    }

    if (sr.heal > 0) {
        s->player.current_hitpoints += sr.heal;
        if (s->player.current_hitpoints > s->player.base_hitpoints)
            s->player.current_hitpoints = s->player.base_hitpoints;
    }

    s->magic_def_drain += sr.magic_def_drain;

    s->player_attack_dmg = total_dmg;
}

static void zul_pick_snakeling_pos(ZulrahState* s, int* ox, int* oy) {
    int order[ZUL_NUM_SNAKELING_POSITIONS];
    for (int i = 0; i < ZUL_NUM_SNAKELING_POSITIONS; i++) order[i] = i;
    encounter_shuffle(order, ZUL_NUM_SNAKELING_POSITIONS, &s->rng_state);
    for (int i = 0; i < ZUL_NUM_SNAKELING_POSITIONS; i++) {
        int px = ZUL_SNAKELING_POSITIONS[order[i]][0];
        int py = ZUL_SNAKELING_POSITIONS[order[i]][1];
        if (zul_on_platform(s, px, py) &&
            !(px == s->player.x && py == s->player.y)) {
            *ox = px; *oy = py; return;
        }
    }
    *ox = s->player.x;
    *oy = s->player.y;
}

static void zul_spawn_snakeling(ZulrahState* s) {
    for (int i = 0; i < ZUL_MAX_SNAKELINGS; i++) {
        if (s->snakelings[i].active) continue;
        ZulrahSnakeling* sn = &s->snakelings[i];
        memset(sn, 0, sizeof(ZulrahSnakeling));
        sn->active = 1;
        sn->npc_instance_id = zul_next_npc_instance_id(s);
        sn->entity.entity_type = ENTITY_NPC;
        sn->entity.npc_size = 1;
        sn->entity.npc_visible = 1;
        sn->is_magic = encounter_rand_int(&s->rng_state, 2);
        sn->entity.npc_def_id = sn->is_magic
            ? MONSTER_DATABASE[MON_ZULRAH_SNAKELING_MAGIC].npc_id
            : MONSTER_DATABASE[MON_ZULRAH_SNAKELING_MELEE].npc_id;
        sn->entity.npc_anim_id = -1;
        zul_pick_snakeling_pos(s, &sn->entity.x, &sn->entity.y);
        sn->entity.current_hitpoints = ZUL_SNAKELING_HP;
        sn->entity.base_hitpoints = ZUL_SNAKELING_HP;
        sn->attack_timer = ZUL_SNAKELING_SPEED;
        sn->lifespan = ZUL_SNAKELING_LIFESPAN;

        if (s->attack_event_count < 8) {
            int ei = s->attack_event_count++;
            s->attack_events[ei].src_x = s->zulrah.x;
            s->attack_events[ei].src_y = s->zulrah.y;
            s->attack_events[ei].dst_x = sn->entity.x;
            s->attack_events[ei].dst_y = sn->entity.y;
            s->attack_events[ei].style = 4;
            s->attack_events[ei].damage = 0;
        }
        return;
    }
}

static void zul_snakeling_tick(ZulrahState* s) {
    for (int i = 0; i < ZUL_MAX_SNAKELINGS; i++) {
        ZulrahSnakeling* sn = &s->snakelings[i];
        if (!sn->active) continue;
        sn->entity.npc_anim_id = -1;

        sn->lifespan--;
        if (sn->lifespan <= 0) { sn->active = 0; continue; }

        int adx = abs_int(sn->entity.x - s->player.x);
        int ady = abs_int(sn->entity.y - s->player.y);
        int in_range = (adx <= 1 && ady <= 1);
        if (!in_range) {
            PathResult pr = zul_pathfind(s, sn->entity.x, sn->entity.y,
                                          s->player.x, s->player.y);
            if (pr.found && (pr.next_dx != 0 || pr.next_dy != 0)) {
                int nx = sn->entity.x + pr.next_dx;
                int ny = sn->entity.y + pr.next_dy;
                if (zul_on_platform(s, nx, ny)) {
                    sn->entity.x = nx; sn->entity.y = ny;
                }
            }
        }

        if (sn->attack_timer > 0) { sn->attack_timer--; continue; }
        adx = abs_int(sn->entity.x - s->player.x);
        ady = abs_int(sn->entity.y - s->player.y);
        if (adx > 1 || ady > 1) continue;

        sn->attack_timer = ZUL_SNAKELING_SPEED;
        sn->entity.npc_anim_id = sn->is_magic ? SNAKELING_ANIM_MAGIC : SNAKELING_ANIM_MELEE;
        AttackStyle sn_style = sn->is_magic ? ATTACK_STYLE_MAGIC : ATTACK_STYLE_MELEE;
        if (encounter_prayer_correct_for_style(s->player.prayer, sn_style)) {
            continue;
        }
        int sn_max = sn->is_magic ? MONSTER_DATABASE[MON_ZULRAH_SNAKELING_MAGIC].max_hit
                                   : MONSTER_DATABASE[MON_ZULRAH_SNAKELING_MELEE].max_hit;
        int dmg = encounter_rand_int(&s->rng_state, sn_max + 1);
        AttackStyle st = sn->is_magic ? ATTACK_STYLE_MAGIC : ATTACK_STYLE_MELEE;
        zul_apply_player_damage(s, dmg, st, &sn->entity);

        if (sn->entity.current_hitpoints <= 0) {
            sn->entity.current_hitpoints = 0;
            sn->active = 0;
        }
    }
}

static const ZulRotationPhase* zul_current_phase(ZulrahState* s) {
    return &ZUL_ROTATIONS[s->rotation_index][s->phase_index];
}

static int zul_cloud_fits(ZulrahState* s, int x, int y) {
    for (int dx = 0; dx < ZUL_CLOUD_SIZE; dx++) {
        for (int dy = 0; dy < ZUL_CLOUD_SIZE; dy++) {
            if (!zul_on_platform(s, x + dx, y + dy)) return 0;
        }
    }
    return 1;
}

static int zul_pick_cloud_pos(ZulrahState* s, int stand, int stall, int* ox, int* oy) {
    int attempts = 0;
    while (attempts++ < 100) {
        int x = ZUL_PLATFORM_MIN + encounter_rand_int(&s->rng_state, ZUL_PLATFORM_MAX - ZUL_PLATFORM_MIN + 1);
        int y = ZUL_PLATFORM_MIN + encounter_rand_int(&s->rng_state, ZUL_PLATFORM_MAX - ZUL_PLATFORM_MIN + 1);

        if (!zul_cloud_fits(s, x, y)) continue;
        if (zul_cloud_overlaps_safe_area(x, y, stand, stall)) continue;

        int overlap = 0;
        for (int j = 0; j < ZUL_MAX_CLOUDS && !overlap; j++) {
            if (s->clouds[j].active &&
                abs(s->clouds[j].x - x) < ZUL_CLOUD_SIZE &&
                abs(s->clouds[j].y - y) < ZUL_CLOUD_SIZE)
                overlap = 1;
        }
        for (int j = 0; j < ZUL_MAX_PENDING_CLOUDS && !overlap; j++) {
            if (s->pending_clouds[j].delay > 0 &&
                abs(s->pending_clouds[j].x - x) < ZUL_CLOUD_SIZE &&
                abs(s->pending_clouds[j].y - y) < ZUL_CLOUD_SIZE)
                overlap = 1;
        }
        if (overlap) continue;

        *ox = x; *oy = y;
        return 1;
    }
    return 0;
}

static void zul_queue_pending_cloud(ZulrahState* s, int x, int y, int delay) {
    for (int i = 0; i < ZUL_MAX_PENDING_CLOUDS; i++) {
        if (s->pending_clouds[i].delay <= 0) {
            s->pending_clouds[i].x = x;
            s->pending_clouds[i].y = y;
            s->pending_clouds[i].delay = delay;
            return;
        }
    }
}

static void zul_activate_cloud(ZulrahState* s, int x, int y) {
    for (int i = 0; i < ZUL_MAX_CLOUDS; i++) {
        if (!s->clouds[i].active) {
            s->clouds[i].x = x;
            s->clouds[i].y = y;
            s->clouds[i].active = 1;
            s->clouds[i].ticks_remaining = ZUL_CLOUD_DURATION;
            return;
        }
    }
}

static void zul_emit_cloud_event(ZulrahState* s, int dst_x, int dst_y, int flight_ticks) {
    if (s->cloud_event_count >= ZUL_MAX_CLOUD_EVENTS) {
        fprintf(stderr, "zulrah cloud event capacity exceeded: %d\n",
            ZUL_MAX_CLOUD_EVENTS);
        abort();
    }
    int i = s->cloud_event_count++;
    s->cloud_events[i].src_x = s->zulrah.x;
    s->cloud_events[i].src_y = s->zulrah.y;
    s->cloud_events[i].dst_x = dst_x;
    s->cloud_events[i].dst_y = dst_y;
    s->cloud_events[i].flight_ticks = flight_ticks;
}

static void zul_spawn_cloud(ZulrahState* s) {
    const ZulRotationPhase* phase = zul_current_phase(s);
    int stand = phase->stand;
    int stall = phase->stall;
    int x, y;
    int spawned = 0;
    if (zul_pick_cloud_pos(s, stand, stall, &x, &y)) {
        zul_queue_pending_cloud(s, x, y, ZUL_CLOUD_FLIGHT_1);
        zul_emit_cloud_event(s, x, y, ZUL_CLOUD_FLIGHT_1);
        spawned = 1;
    }
    if (zul_pick_cloud_pos(s, stand, stall, &x, &y)) {
        zul_queue_pending_cloud(s, x, y, ZUL_CLOUD_FLIGHT_2);
        zul_emit_cloud_event(s, x, y, ZUL_CLOUD_FLIGHT_2);
        spawned = 1;
    }
    if (spawned)
        zul_set_npc_anim_event(s, ZULRAH_ANIM_ATTACK, ZUL_RANGED_ANIM_TICKS);
}

static void zul_pending_cloud_tick(ZulrahState* s) {
    for (int i = 0; i < ZUL_MAX_PENDING_CLOUDS; i++) {
        if (s->pending_clouds[i].delay <= 0) continue;
        s->pending_clouds[i].delay--;
        if (s->pending_clouds[i].delay <= 0) {
            zul_activate_cloud(s, s->pending_clouds[i].x, s->pending_clouds[i].y);
        }
    }
}

static void zul_cloud_tick(ZulrahState* s) {
    int active_count = 0;
    int occupied = 0;
    s->total_pending_cloud_ticks += zul_pending_cloud_count(s);
    for (int i = 0; i < ZUL_MAX_CLOUDS; i++) {
        if (!s->clouds[i].active) continue;
        active_count++;
        s->clouds[i].ticks_remaining--;
        if (s->clouds[i].ticks_remaining <= 0) { s->clouds[i].active = 0; continue; }

        if (zul_player_in_cloud(s->clouds[i].x, s->clouds[i].y,
                                s->player.x, s->player.y)) {
            occupied = 1;
            int dmg = ZUL_CLOUD_DAMAGE_MIN +
                      encounter_rand_int(&s->rng_state, ZUL_CLOUD_DAMAGE_MAX - ZUL_CLOUD_DAMAGE_MIN + 1);
            zul_apply_player_damage(s, dmg, ATTACK_STYLE_MAGIC, NULL);
            s->total_cloud_damage_received += dmg;
        }
    }
    s->total_active_cloud_ticks += active_count;
    s->total_cloud_occupancy_ticks += occupied;
    s->cloud_occupancy_this_tick = occupied;
}

static void zul_venom_tick(ZulrahState* s) {
    if (s->antivenom_timer > 0) s->antivenom_timer--;

    if (s->venom_counter == 0) return;
    if (s->antivenom_timer > 0) return;
    if (s->venom_timer > 0) { s->venom_timer--; return; }
    int dmg = ZUL_VENOM_START + 2 * (s->venom_counter - 1);
    if (dmg > ZUL_VENOM_MAX) dmg = ZUL_VENOM_MAX;
    zul_apply_player_damage(s, dmg, ATTACK_STYLE_MAGIC, NULL);
    s->venom_counter++;
    s->venom_timer = ZUL_VENOM_INTERVAL;
}

static void zul_thrall_tick(ZulrahState* s) {
    if (!s->thrall_active) {
        if (s->thrall_cooldown > 0) { s->thrall_cooldown--; return; }
        s->thrall_active = 1;
        s->thrall_duration_remaining = ZUL_THRALL_DURATION;
        s->thrall_attack_timer = 1;
        return;
    }

    s->thrall_duration_remaining--;
    if (s->thrall_duration_remaining <= 0) {
        s->thrall_active = 0;
        s->thrall_cooldown = ZUL_THRALL_COOLDOWN;
        return;
    }

    if (s->thrall_attack_timer > 0) { s->thrall_attack_timer--; return; }
    s->thrall_attack_timer = ZUL_THRALL_SPEED;

    if (!s->zulrah_visible || s->is_diving) return;

    int dmg = encounter_rand_int(&s->rng_state, ZUL_THRALL_MAX_HIT + 1);
    dmg = zul_cap_damage(s, dmg);
    encounter_damage_player(&s->zulrah, dmg, &s->damage_dealt_this_tick);
    s->total_damage_dealt += dmg;
}

static void zul_fire_action(ZulrahState* s, ZulActionType type) {
    switch (type) {
        case ZA_RANGED:        zul_attack_ranged(s); break;
        case ZA_MAGIC_RANGED:  zul_attack_magic_ranged(s); break;
        case ZA_MELEE:         zul_melee_start(s); break;
        case ZA_JAD_RM:
        case ZA_JAD_MR:        zul_attack_jad(s); break;
        case ZA_CLOUDS:        zul_spawn_cloud(s); break;
        case ZA_SNAKELINGS:    zul_spawn_snakeling(s); break;
        case ZA_SNAKECLOUD_ALT:
            if (s->action_progress % 2 == 0) zul_spawn_snakeling(s);
            else zul_spawn_cloud(s);
            break;
        case ZA_CLOUDSNAKE_ALT:
            if (s->action_progress % 2 == 0) zul_spawn_cloud(s);
            else zul_spawn_snakeling(s);
            break;
        case ZA_END: break;
    }
}

static int zul_action_interval(ZulActionType type) {
    switch (type) {
        case ZA_RANGED:
        case ZA_MAGIC_RANGED:
        case ZA_JAD_RM:
        case ZA_JAD_MR:        return MONSTER_DATABASE[MON_ZULRAH_GREEN].attack_speed;
        case ZA_MELEE:         return ZUL_MELEE_INTERVAL;
        case ZA_CLOUDS:
        case ZA_SNAKELINGS:
        case ZA_SNAKECLOUD_ALT:
        case ZA_CLOUDSNAKE_ALT: return ZUL_SPAWN_INTERVAL;
        default:               return 1;
    }
}

static int zul_action_is_attack(ZulActionType type) {
    return type == ZA_RANGED || type == ZA_MAGIC_RANGED || type == ZA_MELEE ||
           type == ZA_JAD_RM || type == ZA_JAD_MR;
}

static int zul_phase_action_ticks(const ZulRotationPhase* phase) {
    int total = 0;
    for (int i = 0; i < ZUL_MAX_PHASE_ACTIONS; i++) {
        if (phase->actions[i].type == ZA_END) break;
        total += phase->actions[i].count * zul_action_interval((ZulActionType)phase->actions[i].type);
    }
    return total;
}

static void zul_enter_phase(ZulrahState* s) {
    const ZulRotationPhase* phase = zul_current_phase(s);
    s->current_form = (ZulrahForm)phase->form;
    s->zulrah.npc_def_id = zul_form_npc_id(s->current_form);
    s->zulrah.x = ZUL_POSITIONS[phase->position][0];
    s->zulrah.y = ZUL_POSITIONS[phase->position][1];
    s->zulrah_visible = 1;
    s->zulrah.npc_visible = 1;
    s->is_diving = 0;

    int is_initial = (s->phase_index == 0 && s->tick <= 1);
    int surface_ticks = is_initial ? ZUL_SURFACE_TICKS_INITIAL : ZUL_SURFACE_TICKS;
    zul_set_npc_anim_event(
        s,
        is_initial ? ZULRAH_ANIM_SURFACE : ZULRAH_ANIM_RISE,
        surface_ticks);
    s->surface_timer = surface_ticks;

    s->phase_timer = phase->phase_ticks;

    int action_ticks = zul_phase_action_ticks(phase);
    int available = phase->phase_ticks - surface_ticks - ZUL_DIVE_PHASE_TICKS - action_ticks;
    int initial_delay = (available > 1) ? available : 1;

    s->action_index = 0;
    s->action_progress = 0;
    s->action_timer = initial_delay;

    ZulActionType first_type = (ZulActionType)phase->actions[0].type;
    if (first_type == ZA_JAD_RM) s->jad_is_magic_next = 0;
    else if (first_type == ZA_JAD_MR) s->jad_is_magic_next = 1;

    s->zulrah_attacking = 0;
}

static void zul_enter_dive(ZulrahState* s) {
    s->is_diving = 1;
    s->zulrah_attacking = 0;
    zul_set_npc_anim_event(s, ZULRAH_ANIM_DIVE, ZUL_DIVE_ANIM_TICKS);
}

static void zul_next_phase(ZulrahState* s) {
    int rot_len = ZUL_ROT_LENGTHS[s->rotation_index];
    s->phase_index++;

    if (s->phase_index >= rot_len) {
        s->rotation_index = encounter_rand_int(&s->rng_state, ZUL_NUM_ROTATIONS);
        s->phase_index = 1;
    }

    zul_enter_phase(s);
}

static void zul_phase_tick(ZulrahState* s) {
    if (!s->zulrah_visible) return;

    if (s->phase_timer > 0) s->phase_timer--;

    if (s->phase_timer <= 0) {
        s->zulrah_visible = 0;
        s->zulrah.npc_visible = 0;
        zul_next_phase(s);
        return;
    }

    if (s->phase_timer <= ZUL_DIVE_PHASE_TICKS && !s->is_diving) {
        zul_enter_dive(s);
    }
    if (s->is_diving) return;

    if (s->surface_timer > 0) {
        s->surface_timer--;
        return;
    }

    const ZulRotationPhase* phase = zul_current_phase(s);
    const ZulAction* act = &phase->actions[s->action_index];

    if (act->type == ZA_END) {
        s->zulrah_attacking = 0;
        return;
    }

    s->action_timer--;
    if (s->action_timer > 0) return;

    zul_fire_action(s, (ZulActionType)act->type);
    s->action_progress++;

    if (s->action_progress >= act->count) {
        s->action_index++;
        s->action_progress = 0;

        const ZulAction* next = &phase->actions[s->action_index];
        if (next->type == ZA_END) {
            s->zulrah_attacking = 0;
            return;
        }

        s->zulrah_attacking = zul_action_is_attack((ZulActionType)next->type);

        if (next->type == ZA_JAD_RM) s->jad_is_magic_next = 0;
        else if (next->type == ZA_JAD_MR) s->jad_is_magic_next = 1;

        s->action_timer = zul_action_interval((ZulActionType)next->type);
    } else {
        s->action_timer = zul_action_interval((ZulActionType)act->type);
    }
}

static void zul_apply_eat_cell(ZulrahState* s, int cell_idx, OsrsConsumableKind kind) {
    FoodType type;
    switch (kind) {
        case OSRS_CONSUMABLE_SHARK_FOOD: type = FOOD_SHARK; break;
        case OSRS_CONSUMABLE_KARAMBWAN: type = FOOD_KARAMBWAN; break;
        default:
            fprintf(stderr, "zulrah eat: unsupported consumable kind %d\n", (int)kind);
            abort();
    }
    OsrsPlayerEatResult r = osrs_player_eat_food_effects(&s->player, type);
    if (r.consumed) osrs_inventory_cell_consume_eat(&s->inventory_cells[cell_idx]);
}

static void zul_apply_drink_one_dose_effect(void* ctx, OsrsConsumableKind kind) {
    ZulrahState* s = (ZulrahState*)ctx;
    switch (kind) {
        case OSRS_CONSUMABLE_PRAYER_RESTORE:
            encounter_add_prayer_restore(
                &s->player, osrs_prayer_potion_restore_amount(s->player.base_prayer));
            encounter_cap_prayer_restore(&s->player);
            return;
        case OSRS_CONSUMABLE_ANTIVENOM_PLUS:
            s->venom_counter = 0;
            s->venom_timer = 0;
            s->antivenom_timer = ZUL_ANTIVENOM_DURATION;
            return;
        default:
            break;
    }
    fprintf(stderr, "zulrah drink: unsupported consumable kind %d\n", (int)kind);
    abort();
}

static int zul_drink_has_effect(const ZulrahState* s, OsrsConsumableKind kind) {
    switch (kind) {
        case OSRS_CONSUMABLE_PRAYER_RESTORE:
            return s->player.current_prayer < s->player.base_prayer;
        case OSRS_CONSUMABLE_ANTIVENOM_PLUS:
            return s->antivenom_timer == 0;
        default:
            break;
    }
    fprintf(stderr, "zulrah drink mask: unsupported consumable kind %d\n", (int)kind);
    abort();
}

static void zul_apply_drink_cell(
    ZulrahState* s, int cell_idx, OsrsInventoryClickResolution resolution
) {
    (void)osrs_inventory_cell_consume_drink_one_dose(
        &s->inventory_cells[cell_idx], resolution, &s->player.potion_timer,
        zul_apply_drink_one_dose_effect, s);
}

static void zul_player_attack_snakeling(ZulrahState* s, int snakeling_idx) {
    if (snakeling_idx < 0 || snakeling_idx >= ZUL_MAX_SNAKELINGS) return;
    ZulrahSnakeling* sn = &s->snakelings[snakeling_idx];
    if (!sn->active) return;

    const EncounterLoadoutStats* ls = zul_live_stats(s);
    if (!encounter_player_can_attack(
            s->player.x, s->player.y,
            sn->entity.x, sn->entity.y, 1,
            ls->attack_range, osrs_los_open_query())) return;

    AttackStyle style = zul_player_equipped_attack_style(s);
    s->player.attack_timer = ls->attack_speed;
    s->player.just_attacked = 1;
    s->player.last_attack_style = style;

    const MonsterStats* m = &MONSTER_DATABASE[sn->is_magic
        ? MON_ZULRAH_SNAKELING_MAGIC : MON_ZULRAH_SNAKELING_MELEE];
    int def_bonus = style == ATTACK_STYLE_MAGIC ? m->magic_def : m->ranged_def;
    int def_roll = (m->def_level + 8) * (def_bonus + 64);
    if (def_roll < 0) def_roll = 0;
    int att_roll = osrs_player_att_roll(ls->eff_level, ls->attack_bonus);
    if (!encounter_roll_hit_chance(&s->rng_state, att_roll, def_roll)) return;

    int dmg = encounter_rand_int(&s->rng_state, ls->max_hit + 1);
    encounter_damage_player(&sn->entity, dmg, &s->damage_dealt_this_tick);
    if (sn->entity.current_hitpoints <= 0) sn->active = 0;
}

static void zul_process_prayer(ZulrahState* s, int overhead_action, int offensive_action) {
    if (encounter_apply_overhead_action(&s->player.prayer, overhead_action)) {
        s->player.prayer_just_activated = 1;
    }
    OffensivePrayer prev_offensive = s->player.offensive_prayer;
    if (encounter_apply_offensive_action(&s->player.offensive_prayer, offensive_action)) {
        s->player.offensive_prayer_just_activated = 1;
    }
    if (s->player.offensive_prayer != prev_offensive)
        zul_mark_live_stats_dirty(s);
}




static FightStyle zul_default_fight_style_for_style(AttackStyle style) {
    if (style == ATTACK_STYLE_MAGIC) return FIGHT_STYLE_ACCURATE;
    if (style == ATTACK_STYLE_RANGED) return FIGHT_STYLE_RAPID;
    return FIGHT_STYLE_ACCURATE;
}


static void zul_apply_human_player_commands(ZulrahState* s) {
    int did_change_stats = 0;
    for (int i = 0; i < s->human_command_count; i++) {
        const HumanCommand* cmd = &s->human_commands[i];
        if (cmd->kind == HUMAN_COMMAND_EQUIP_INVENTORY_ITEM) {
            if (cmd->gear_slot >= 0 && cmd->gear_slot < NUM_GEAR_SLOTS &&
                cmd->item_db_idx >= 0 && cmd->item_db_idx < NUM_ITEMS) {
                int changed = slot_equip_item(&s->player, cmd->gear_slot, (uint8_t)cmd->item_db_idx);
                if (changed) {
                    did_change_stats = 1;
                    if (cmd->gear_slot == GEAR_SLOT_WEAPON) {
                        AttackStyle style = zul_player_equipped_attack_style(s);
                        s->player.fight_style = zul_default_fight_style_for_style(style);
                    }
                }
            }
        } else if (cmd->kind == HUMAN_COMMAND_FIGHT_STYLE) {
            if (cmd->fight_style >= FIGHT_STYLE_ACCURATE &&
                cmd->fight_style <= FIGHT_STYLE_DEFENSIVE_AUTOCAST) {
                s->player.fight_style = (FightStyle)cmd->fight_style;
                did_change_stats = 1;
            }
        }
    }
    if (did_change_stats)
        zul_mark_live_stats_dirty(s);
}

static void zul_write_obs(EncounterState* state, EncounterContext* context, float* obs) {
    (void)context;
    ZulrahState* s = (ZulrahState*)state;
    memset(obs, 0, ZUL_NUM_OBS * sizeof(float));
    int i = 0;

    obs[i++] = (float)s->player.current_hitpoints / s->player.base_hitpoints;
    obs[i++] = (float)s->player.current_prayer / s->player.base_prayer;
    obs[i++] = (float)s->player.x / ZUL_ARENA_SIZE;
    obs[i++] = (float)s->player.y / ZUL_ARENA_SIZE;
    obs[i++] = (float)s->player.attack_timer / 5.0f;
    obs[i++] = (float)s->player.food_timer / 3.0f;
    obs[i++] = (float)s->player.potion_timer / 3.0f;
    obs[i++] = (s->player.prayer == PRAYER_PROTECT_MAGIC) ? 1.0f : 0.0f;
    obs[i++] = (s->player.prayer == PRAYER_PROTECT_RANGED) ? 1.0f : 0.0f;
    obs[i++] = (s->player.prayer == PRAYER_PROTECT_MELEE) ? 1.0f : 0.0f;
    obs[i++] = (s->player.offensive_prayer == OFFENSIVE_PRAYER_PIETY) ? 1.0f : 0.0f;
    obs[i++] = (s->player.offensive_prayer == OFFENSIVE_PRAYER_RIGOUR) ? 1.0f : 0.0f;
    obs[i++] = (s->player.offensive_prayer == OFFENSIVE_PRAYER_AUGURY) ? 1.0f : 0.0f;
    obs[i++] = (float)s->player_stunned_ticks / ZUL_MELEE_STUN_TICKS;

    obs[i++] = (float)s->zulrah.current_hitpoints / MONSTER_DATABASE[MON_ZULRAH_GREEN].hp;
    obs[i++] = (float)(s->zulrah.x - s->player.x) / ZUL_ARENA_SIZE;
    obs[i++] = (float)(s->zulrah.y - s->player.y) / ZUL_ARENA_SIZE;
    obs[i++] = (s->current_form == ZUL_FORM_GREEN) ? 1.0f : 0.0f;
    obs[i++] = (s->current_form == ZUL_FORM_RED) ? 1.0f : 0.0f;
    obs[i++] = (s->current_form == ZUL_FORM_BLUE) ? 1.0f : 0.0f;
    obs[i++] = s->zulrah_visible ? 1.0f : 0.0f;
    obs[i++] = s->is_diving ? 1.0f : 0.0f;
    obs[i++] = s->zulrah_attacking ? 1.0f : 0.0f;
    obs[i++] = (float)s->action_timer / MONSTER_DATABASE[MON_ZULRAH_GREEN].attack_speed;
    obs[i++] = (float)encounter_dist_to_npc(s->player.x, s->player.y, s->zulrah.x, s->zulrah.y, ZUL_NPC_SIZE) / ZUL_ARENA_SIZE;
    obs[i++] = (float)s->rotation_index / (ZUL_NUM_ROTATIONS - 1);
    obs[i++] = (float)s->phase_index / 12.0f;
    obs[i++] = (s->melee_pending) ? 1.0f : 0.0f;

    obs[i++] = (s->venom_counter > 0) ? 1.0f : 0.0f;
    obs[i++] = (float)s->venom_timer / ZUL_VENOM_INTERVAL;

    for (int c = 0; c < ZUL_MAX_CLOUDS; c++) {
        obs[i++] = s->clouds[c].active ? (float)(s->clouds[c].x - s->player.x) / ZUL_ARENA_SIZE : 0.0f;
        obs[i++] = s->clouds[c].active ? (float)(s->clouds[c].y - s->player.y) / ZUL_ARENA_SIZE : 0.0f;
        obs[i++] = s->clouds[c].active ? 1.0f : 0.0f;
    }

    for (int n = 0; n < ZUL_MAX_SNAKELINGS; n++) {
        ZulrahSnakeling* sn = &s->snakelings[n];
        obs[i++] = sn->active ? (float)(sn->entity.x - s->player.x) / ZUL_ARENA_SIZE : 0.0f;
        obs[i++] = sn->active ? (float)(sn->entity.y - s->player.y) / ZUL_ARENA_SIZE : 0.0f;
        obs[i++] = sn->active ? 1.0f : 0.0f;
        obs[i++] = sn->active ? (float)chebyshev_distance(
            s->player.x, s->player.y, sn->entity.x, sn->entity.y) / ZUL_ARENA_SIZE : 0.0f;
    }

    obs[i++] = (float)s->tick / ZUL_MAX_TICKS;
    obs[i++] = s->damage_dealt_this_tick / 50.0f;
    obs[i++] = s->damage_received_this_tick / 50.0f;
    obs[i++] = s->total_damage_dealt / MONSTER_DATABASE[MON_ZULRAH_GREEN].hp;

    obs[i++] = (float)s->player.special_energy / 100.0f;
    obs[i++] = (s->antivenom_timer > 0) ? 1.0f : 0.0f;
    obs[i++] = (float)s->antivenom_timer / ZUL_ANTIVENOM_DURATION;
    obs[i++] = (float)s->gear_tier / (ZUL_NUM_GEAR_TIERS - 1);

    const ZulRotationPhase* phase = zul_current_phase(s);
    if (phase->stand < ZUL_NUM_STAND_LOCATIONS) {
        obs[i++] = (float)(ZUL_STAND_COORDS[phase->stand][0] - s->player.x) / ZUL_ARENA_SIZE;
        obs[i++] = (float)(ZUL_STAND_COORDS[phase->stand][1] - s->player.y) / ZUL_ARENA_SIZE;
    } else {
        obs[i++] = 0.0f; obs[i++] = 0.0f;
    }
    if (phase->stall < ZUL_NUM_STAND_LOCATIONS) {
        obs[i++] = (float)(ZUL_STAND_COORDS[phase->stall][0] - s->player.x) / ZUL_ARENA_SIZE;
        obs[i++] = (float)(ZUL_STAND_COORDS[phase->stall][1] - s->player.y) / ZUL_ARENA_SIZE;
    } else {
        obs[i++] = 0.0f; obs[i++] = 0.0f;
    }

    int active_dx = 0;
    int active_dy = 0;
    int escape_dx = 0;
    int escape_dy = 0;
    zul_nearest_active_cloud_features(
        s, &active_dx, &active_dy, &escape_dx, &escape_dy);
    int pending_dx = 0;
    int pending_dy = 0;
    int pending_delay = 0;
    zul_nearest_pending_cloud_features(
        s, &pending_dx, &pending_dy, &pending_delay);
    obs[i++] = (escape_dx != 0 || escape_dy != 0) ? 1.0f : 0.0f;
    obs[i++] = (float)active_dx / (float)ZUL_ARENA_SIZE;
    obs[i++] = (float)active_dy / (float)ZUL_ARENA_SIZE;
    obs[i++] = (float)escape_dx / (float)ZUL_ARENA_SIZE;
    obs[i++] = (float)escape_dy / (float)ZUL_ARENA_SIZE;
    obs[i++] = (float)zul_active_cloud_count(s) / (float)ZUL_MAX_CLOUDS;
    obs[i++] = (float)zul_pending_cloud_count(s) / (float)ZUL_MAX_PENDING_CLOUDS;
    obs[i++] = (float)pending_dx / (float)ZUL_ARENA_SIZE;
    obs[i++] = (float)pending_dy / (float)ZUL_ARENA_SIZE;
    obs[i++] = (float)pending_delay / (float)ZUL_CLOUD_FLIGHT_2;
    obs[i++] = zul_active_cloud_overlaps_safe(s, phase->stand) ? 1.0f : 0.0f;
    obs[i++] = zul_active_cloud_overlaps_safe(s, phase->stall) ? 1.0f : 0.0f;
    obs[i++] = zul_pending_cloud_overlaps_safe(s, phase->stand) ? 1.0f : 0.0f;
    obs[i++] = zul_pending_cloud_overlaps_safe(s, phase->stall) ? 1.0f : 0.0f;
    for (int m = 0; m < ZUL_MOVE_DIM; m++)
        obs[i++] = zul_move_action_cloud_unsafe(s, m) ? 1.0f : 0.0f;

    static const float ZUL_ZERO_POST_USE_DELTAS[6] = {0};
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
        osrs_write_inventory_cell_affordance_features(
            &obs[i],
            s->inventory_cells[cell].item_idx,
            s->inventory_cells[cell].raw_osrs_id,
            s->inventory_cells[cell].dose,
            osrs_inventory_cell_holds_equipped_item(&s->player, s->inventory_cells, cell),
            ZUL_ZERO_POST_USE_DELTAS,
            s->player.base_hitpoints,
            s->player.base_prayer,
            s->player.base_attack);
        i += OSRS_INVENTORY_CELL_OBS_FEATURES;
    }
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        osrs_write_equipped_self_features(&obs[i], s->player.equipped[slot]);
        i += OSRS_EQUIPPED_SELF_OBS_FEATURES;
    }

    if (i != ZUL_NUM_OBS) {
        fprintf(stderr, "zulrah obs size mismatch: wrote %d expected %d\n", i, ZUL_NUM_OBS);
        abort();
    }
}

static void zul_write_mask(EncounterState* state, EncounterContext* context, float* mask) {
    (void)context;
    ZulrahState* s = (ZulrahState*)state;
    for (int i = 0; i < ZUL_ACTION_MASK_SIZE; i++) mask[i] = 1.0f;
    int off = 0;

    for (int m = 0; m < ZUL_MOVE_DIM; m++) {
        if (m > 0) {
            if (s->player_stunned_ticks > 0) { mask[off] = 0.0f; }
            else {
                int nx = s->player.x + ENCOUNTER_MOVE_TARGET_DX[m];
                int ny = s->player.y + ENCOUNTER_MOVE_TARGET_DY[m];
                if (!zul_on_platform(s, nx, ny)) mask[off] = 0.0f;
            }
        }
        off++;
    }
    mask[off] = s->zulrah_visible && !s->is_diving ? 1.0f : 0.0f;
    off++;
    for (int n = 0; n < ZUL_MAX_SNAKELINGS; n++) {
        mask[off] = s->snakelings[n].active ? 1.0f : 0.0f;
        off++;
    }
    for (int p = 0; p < ZUL_PRAYER_DIM; p++) {
        if (p == ENCOUNTER_OVERHEAD_OFF && s->player.prayer == PRAYER_NONE)
            mask[off] = 0.0f;
        if (p >= ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE && s->player.current_prayer <= 0)
            mask[off] = 0.0f;
        off++;
    }
    int cell_equip_slot[OSRS_INVENTORY_SIZE];
    int cell_can_eat[OSRS_INVENTORY_SIZE];
    int cell_can_drink[OSRS_INVENTORY_SIZE];
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
        cell_equip_slot[cell] = -1;
        cell_can_eat[cell] = 0;
        cell_can_drink[cell] = 0;
        OsrsInventoryClickResolution r = osrs_inventory_cell_click_interpret(
            &s->inventory_cells[cell], OSRS_CLICK_TICK_FIRST);
        if (r.click_action == OSRS_CLICK_EQUIP) {
            if (osrs_can_equip_from_cell(&s->player, s->inventory_cells, cell))
                cell_equip_slot[cell] =
                    osrs_item_gear_slot(s->inventory_cells[cell].item_idx);
        } else if (r.click_action == OSRS_CLICK_EAT) {
            cell_can_eat[cell] =
                osrs_can_eat_consumable_kind(&s->player, r.consumable_kind);
        } else if (r.click_action == OSRS_CLICK_DRINK) {
            cell_can_drink[cell] = s->inventory_cells[cell].dose > 0 &&
                s->player.potion_timer == 0 &&
                zul_drink_has_effect(s, r.consumable_kind);
        }
    }
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        mask[off] = 1.0f;
        off++;
        for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
            mask[off] = cell_equip_slot[cell] == slot ? 1.0f : 0.0f;
            off++;
        }
    }
    mask[off] = 1.0f;
    off++;
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
        mask[off] = cell_can_eat[cell] ? 1.0f : 0.0f;
        off++;
    }
    mask[off] = 1.0f;
    off++;
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
        mask[off] = cell_can_drink[cell] ? 1.0f : 0.0f;
        off++;
    }
    {
        int weapon = s->player.equipped[GEAR_SLOT_WEAPON];
        int weapon_spec_cost = osrs_spec_cost(weapon);
        mask[off] = 1.0f;
        off++;
        mask[off] = weapon_spec_cost > 0 &&
            (s->player.special_energy >= weapon_spec_cost || s->player.spec_armed)
            ? 1.0f : 0.0f;
        off++;
        mask[off] = s->player.spec_armed ? 1.0f : 0.0f;
        off++;
    }
    for (int o = 0; o < ZUL_OFFENSIVE_DIM; o++) {
        if (o == ENCOUNTER_OFFENSIVE_OFF &&
                s->player.offensive_prayer == OFFENSIVE_PRAYER_NONE)
            mask[off] = 0.0f;
        if (o >= ENCOUNTER_OFFENSIVE_SET_REFRESH_PIETY && s->player.current_prayer <= 0)
            mask[off] = 0.0f;
        off++;
    }
    if (off != ZUL_ACTION_MASK_SIZE) {
        fprintf(stderr, "zulrah mask size mismatch: wrote %d expected %d\n",
            off, ZUL_ACTION_MASK_SIZE);
        abort();
    }
}

static float zul_compute_reward(ZulrahState* s) {
    const ZulrahRewardConfig* cfg = &s->reward_config;
    float r = 0.0f;
    if (s->boss_killed_this_tick) {
        r += cfg->win;
    }
    if (s->player_lost_this_tick) {
        r -= cfg->loss_penalty;
    } else {
        if (s->damage_dealt_this_tick > 0.0f) {
            float norm_dmg = s->damage_dealt_this_tick / 50.0f;
            r += cfg->damage_dealt * norm_dmg;
            int correct = (s->current_form == ZUL_FORM_BLUE &&
                           s->player.attack_style_this_tick == ATTACK_STYLE_RANGED) ||
                          ((s->current_form == ZUL_FORM_GREEN ||
                            s->current_form == ZUL_FORM_RED) &&
                           s->player.attack_style_this_tick == ATTACK_STYLE_MAGIC);
            if (correct) r += cfg->correct_style * norm_dmg;
        }

        if (s->damage_received_this_tick > 0.0f)
            r -= cfg->damage_received_penalty * (s->damage_received_this_tick / 50.0f);
    }

    if (s->cloud_occupancy_this_tick > 0)
        r -= cfg->cloud_occupancy_penalty;

    return r;
}

static ZulrahRewardConfig zul_default_reward_config(void) {
    return (ZulrahRewardConfig){
        .win = ZUL_REWARD_WIN_DEFAULT,
        .loss_penalty = ZUL_REWARD_LOSS_PENALTY_DEFAULT,
        .damage_dealt = ZUL_REWARD_DAMAGE_DEALT_DEFAULT,
        .correct_style = ZUL_REWARD_CORRECT_STYLE_DEFAULT,
        .damage_received_penalty = ZUL_REWARD_DAMAGE_RECEIVED_PENALTY_DEFAULT,
        .cloud_occupancy_penalty = ZUL_REWARD_CLOUD_OCCUPANCY_PENALTY_DEFAULT,
    };
}

static EncounterState* zul_create(void) {
    ZulrahState* s = (ZulrahState*)calloc(1, sizeof(ZulrahState));
    if (!s) abort();
    s->reward_config = zul_default_reward_config();
    return (EncounterState*)s;
}

static void zul_destroy(EncounterState* state) { free(state); }

static float zul_score_speed_bonus_for_duration(int duration) {
    if (duration < 1) duration = 1;
    if (duration > ZUL_MAX_TICKS) duration = ZUL_MAX_TICKS;
    return (1.0f - (float)duration / (float)ZUL_MAX_TICKS) *
        ZUL_SCORE_SPEED_BONUS_DEFAULT;
}

static float zul_current_kill_progress(const ZulrahState* s) {
    int max_hp = MONSTER_DATABASE[MON_ZULRAH_GREEN].hp;
    if (max_hp <= 0 || s->zulrah.current_hitpoints <= 0)
        return 0.0f;
    return 1.0f - (float)s->zulrah.current_hitpoints / (float)max_hp;
}

typedef struct {
    float win;
    float score;
} ZulEpisodeOutcome;

static ZulEpisodeOutcome zul_episode_outcome(const ZulrahState* s) {
    float kills = (float)s->kills_this_episode;
    float win = (s->episode_mode == ZUL_EPISODE_TRIP)
        ? kills
        : ((s->winner == ZUL_OUTCOME_PLAYER_WON) ? 1.0f : 0.0f);
    float partial = (s->episode_mode == ZUL_EPISODE_TRIP)
        ? zul_current_kill_progress(s)
        : 0.0f;
    float speed_bonus = (s->episode_mode == ZUL_EPISODE_TRIP)
        ? s->score_speed_bonus_sum
        : (win > 0.0f
            ? zul_score_speed_bonus_for_duration(s->tick)
            : 0.0f);
    return (ZulEpisodeOutcome){
        .win = win,
        .score = win + partial + speed_bonus,
    };
}

static void zul_clear_active_kill(ZulrahState* s) {
    memset(s->clouds, 0, sizeof(s->clouds));
    memset(s->pending_clouds, 0, sizeof(s->pending_clouds));
    memset(s->snakelings, 0, sizeof(s->snakelings));
    s->attack_event_count = 0;
    s->cloud_event_count = 0;
    s->melee_pending = 0;
    s->melee_stare_timer = 0;
    encounter_pending_hit_queue_clear(&s->player_pending_hits);
    encounter_pending_hit_queue_clear(&s->zulrah_pending_hits);
    s->phase_timer = 0;
    s->surface_timer = 0;
    s->is_diving = 0;
    s->zulrah_anim_until_tick = 0;
    s->zulrah_anim_event_tick = -1;
    s->zulrah_death_ticks = 0;
    s->zulrah_attacking = 0;
    s->action_index = 0;
    s->action_progress = 0;
    s->action_timer = 0;
    s->zulrah_visible = 0;
    s->zulrah.current_hitpoints = 0;
    s->magic_def_drain = 0;
    osrs_interaction_init(&s->interaction);
}

static void zul_start_active_kill(ZulrahState* s) {
    zul_clear_active_kill(s);
    s->zulrah.entity_type = ENTITY_NPC;
    s->zulrah.npc_def_id = zul_form_npc_id(ZUL_FORM_GREEN);
    s->zulrah.npc_size = ZUL_NPC_SIZE;
    s->zulrah.npc_anim_id = -1;
    s->zulrah.base_hitpoints = MONSTER_DATABASE[MON_ZULRAH_GREEN].hp;
    s->zulrah.current_hitpoints = MONSTER_DATABASE[MON_ZULRAH_GREEN].hp;
    s->zulrah_npc_instance_id = zul_next_npc_instance_id(s);
    s->rotation_index = encounter_rand_int(&s->rng_state, ZUL_NUM_ROTATIONS);
    s->phase_index = 0;
    s->kill_start_tick = s->tick;
    s->respawn_timer = 0;
    zul_enter_phase(s);
}

static void zul_record_boss_kill(ZulrahState* s) {
    s->boss_killed_this_tick = 1;
    s->kills_this_episode++;
    int duration = s->tick - s->kill_start_tick;
    s->score_speed_bonus_sum += zul_score_speed_bonus_for_duration(duration);
    if (!osrs_npc_death_linger_start(
            s->zulrah.current_hitpoints,
            s->zulrah_visible,
            &s->zulrah_death_ticks,
            ZUL_DEATH_ANIM_TICKS)) {
        fprintf(stderr, "zulrah death linger did not start for boss kill\n");
        abort();
    }
    s->zulrah_visible = 1;
    s->zulrah.npc_visible = 1;
    s->zulrah_attacking = 0;
    s->is_diving = 0;
    s->surface_timer = 0;
    s->phase_timer = 0;
    s->action_timer = 0;
    s->melee_pending = 0;
    s->melee_stare_timer = 0;
    zul_set_npc_anim_event(s, ZULRAH_ANIM_DEATH, ZUL_DEATH_ANIM_TICKS);
}

static void zul_finish_boss_death(ZulrahState* s) {
    if (s->episode_mode == ZUL_EPISODE_TRIP) {
        zul_clear_active_kill(s);
        s->respawn_timer = ZUL_TRIP_RESPAWN_DELAY_TICKS;
        return;
    }
    s->episode_over = 1;
    s->winner = ZUL_OUTCOME_PLAYER_WON;
}

static void zul_record_player_loss(ZulrahState* s) {
    s->player_lost_this_tick = 1;
    s->episode_over = 1;
    s->winner = ZUL_OUTCOME_PLAYER_DIED;
}

static void zul_record_episode_timeout(ZulrahState* s) {
    if (s->episode_mode == ZUL_EPISODE_TRIP && s->kills_this_episode > 0) {
        s->episode_over = 1;
        s->winner = ZUL_OUTCOME_PLAYER_WON;
    } else {
        zul_record_player_loss(s);
    }
}

static void zul_seed_inventory_cells(ZulrahState* s) {
    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++)
        s->inventory_cells[i] = osrs_inventory_cell_empty();
    int cell = 0;
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        uint8_t item = ZUL_RANGE_LOADOUT[s->gear_tier][slot];
        if (item == ITEM_NONE || item == s->player.equipped[slot]) continue;
        s->inventory_cells[cell++] = osrs_inventory_cell_from_item(item);
    }
    static const struct { uint16_t raw_osrs_id; int count; } ZUL_TRIP_CONSUMABLES[] = {
        {385, ZUL_PLAYER_FOOD},
        {3144, ZUL_PLAYER_KARAMBWAN},
        {2434, ZUL_PLAYER_RESTORE_DOSES / 4},
        {12913, ZUL_ANTIVENOM_DOSES / 4},
    };
    for (size_t k = 0; k < sizeof(ZUL_TRIP_CONSUMABLES) / sizeof(*ZUL_TRIP_CONSUMABLES); k++) {
        for (int n = 0; n < ZUL_TRIP_CONSUMABLES[k].count; n++) {
            if (cell >= OSRS_INVENTORY_SIZE) {
                fprintf(stderr, "zulrah trip inventory overflows %d cells\n", OSRS_INVENTORY_SIZE);
                abort();
            }
            s->inventory_cells[cell++] = osrs_inventory_cell_from_raw_osrs_id(
                ZUL_TRIP_CONSUMABLES[k].raw_osrs_id);
        }
    }
}

static void zul_reset(EncounterState* state, EncounterContext* context, uint32_t seed) {
    (void)context;
    ZulrahState* s = (ZulrahState*)state;
    Log saved_log = s->log;
    void* saved_cmap = s->collision_map;
    int saved_wx = s->world_offset_x;
    int saved_wy = s->world_offset_y;
    int saved_tier = s->gear_tier;
    int saved_fixed_tier = s->gear_tier_fixed;
    int saved_tier_mode = s->gear_tier_mode;
    int saved_episode_mode = s->episode_mode;
    float saved_tier_weights[ZUL_NUM_GEAR_TIERS];
    memcpy(saved_tier_weights, s->gear_tier_weights, sizeof(saved_tier_weights));
    ZulrahRewardConfig saved_reward_config = s->reward_config;
    uint32_t saved_rng = s->rng_state;
    memset(s, 0, sizeof(ZulrahState));
    s->log = saved_log;
    s->collision_map = saved_cmap;
    s->world_offset_x = saved_wx;
    s->world_offset_y = saved_wy;
    s->gear_tier = saved_tier;
    s->gear_tier_fixed = saved_fixed_tier;
    s->gear_tier_mode = saved_tier_mode;
    s->episode_mode = saved_episode_mode;
    memcpy(s->gear_tier_weights, saved_tier_weights, sizeof(saved_tier_weights));
    s->reward_config = saved_reward_config;
    s->rng_state = encounter_resolve_seed(saved_rng, seed);
    s->gear_tier = zul_sample_gear_tier(s);

    s->player.entity_type = ENTITY_PLAYER;
    memset(s->player.equipped, ITEM_NONE, NUM_GEAR_SLOTS);
    encounter_init_maxed_player_combat_stats(&s->player, ZUL_PLAYER_PRAYER);
    s->player.x = ZUL_PLAYER_START_X;
    s->player.y = ZUL_PLAYER_START_Y;
    s->player.special_energy = 100;
    osrs_item_effect_state_init(&s->player.item_effect_state);
    if (s->gear_tier == 2) {
        s->player.saturated_heart_count = 1;
        encounter_apply_saturated_heart_boost(&s->player);
    }
    if (s->gear_tier >= 1) {
        s->thrall_active = 1;
        s->thrall_duration_remaining = ZUL_THRALL_DURATION;
        s->thrall_attack_timer = ZUL_THRALL_SPEED;
    }
    osrs_interaction_init(&s->interaction);
    s->player.spec_armed = 0;
    encounter_apply_loadout(&s->player, ZUL_MAGE_LOADOUT[s->gear_tier], GEAR_MAGE);
    zul_populate_player_inventory(&s->player, s->gear_tier);
    zul_seed_inventory_cells(s);
    s->player.offensive_prayer =
        (s->gear_tier >= 1) ? OFFENSIVE_PRAYER_AUGURY : OFFENSIVE_PRAYER_NONE;
    zul_mark_live_stats_dirty(s);
    zul_start_active_kill(s);
}

static void zul_step_tick(ZulrahState* s, const int* actions) {
    s->reward = 0.0f;
    s->damage_dealt_this_tick = 0.0f;
    s->damage_received_this_tick = 0.0f;
    s->cloud_occupancy_this_tick = 0;
    s->boss_killed_this_tick = 0;
    s->player_lost_this_tick = 0;
    s->player.just_attacked = 0;
    s->player.hit_landed_this_tick = 0;
    s->player.attack_style_this_tick = ATTACK_STYLE_NONE;
    s->player.used_special_this_tick = 0;
    s->player.ate_food_this_tick = 0;
    s->player.ate_karambwan_this_tick = 0;
    s->zulrah.hit_landed_this_tick = 0;
    s->player_attacked_this_tick = 0;
    s->player_attack_dmg = 0;
    s->player_attack_style_id = ATTACK_STYLE_NONE;
    s->player_attack_is_special = 0;
    s->player_attack_timing = (EncounterProjectileTiming){0};
    s->player_moved_this_tick = 0;
    s->player_chased_target_this_tick = 0;
    s->attack_event_count = 0;
    s->cloud_event_count = 0;
    s->tick++;

    if (s->zulrah_death_ticks > 0) {
        if (osrs_npc_death_linger_tick(&s->zulrah_death_ticks)) {
            s->zulrah_visible = 0;
            s->zulrah.npc_visible = 0;
            zul_finish_boss_death(s);
        }
        return;
    }

    zul_update_npc_anim_lifetime(s);

    if (s->player.attack_timer > 0) s->player.attack_timer--;
    if (s->player.food_timer > 0) s->player.food_timer--;
    if (s->player.karambwan_timer > 0) s->player.karambwan_timer--;
    if (s->player.potion_timer > 0) s->player.potion_timer--;
    if (s->player_stunned_ticks > 0) s->player_stunned_ticks--;
    int stats_changed = encounter_tick_saturated_heart(&s->player);
    if (s->tick > 0 && s->tick % 60 == 0)
        stats_changed |= encounter_decay_player_combat_stats_toward_base(&s->player);
    if (stats_changed)
        zul_mark_live_stats_dirty(s);

    zul_resolve_zulrah_pending_hits(s);
    zul_resolve_player_pending_hits(s);

    if (s->melee_pending) {
        s->melee_stare_timer--;
        if (s->melee_stare_timer <= 0) zul_melee_hit(s);
    }

    if (s->zulrah_visible && s->zulrah.current_hitpoints <= 0) {
        zul_record_boss_kill(s);
        return;
    }
    if (s->player.current_hitpoints <= 0) {
        zul_record_player_loss(s);
        return;
    }

    zul_process_prayer(s, actions[ZUL_HEAD_PRAYER], actions[ZUL_HEAD_OFFENSIVE]);

    if (s->human_command_mode)
        zul_apply_human_player_commands(s);

    int spec_act = actions[ZUL_HEAD_SPEC];
    int spec_cost = osrs_spec_cost(s->player.equipped[GEAR_SLOT_WEAPON]);
    if (spec_act == 1 && spec_cost > 0 && s->player.special_energy >= spec_cost) {
        s->player.spec_armed = 1;
    } else if (spec_act == 2) {
        s->player.spec_armed = 0;
    }

    {
        OsrsInventoryClickActions clicks;
        for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++)
            clicks.equip_by_slot[slot] = actions[ZUL_HEAD_EQUIP_SLOT(slot)];
        clicks.eat = actions[ZUL_HEAD_EAT];
        clicks.drink = actions[ZUL_HEAD_DRINK];
        OsrsInventoryTickIntent intent =
            osrs_resolve_inventory_tick_intent(&s->player, s->inventory_cells, &clicks);
        if (osrs_inventory_tick_intent_has_effect(&intent))
            osrs_interaction_check_interrupt(&s->interaction, OSRS_IACT_EQUIP);
        OsrsInventoryApplyStep click_step;
        while (osrs_inventory_intent_next(&intent, &click_step)) {
            switch (click_step.kind) {
                case OSRS_INVENTORY_APPLY_EQUIP:
                    if (osrs_equip_from_cell(
                            &s->player, s->inventory_cells, click_step.cell_idx) >= 0)
                        zul_mark_live_stats_dirty(s);
                    break;
                case OSRS_INVENTORY_APPLY_EAT:
                    zul_apply_eat_cell(
                        s, click_step.cell_idx, click_step.resolution.consumable_kind);
                    break;
                case OSRS_INVENTORY_APPLY_DRINK:
                    zul_apply_drink_cell(s, click_step.cell_idx, click_step.resolution);
                    break;
            }
        }
    }

    int primary = actions[ZUL_HEAD_PRIMARY];
    int has_new_target = 0;
    int new_target_slot = 0;
    if (primary >= ZUL_PRIMARY_ATTACK_BASE && primary < ZUL_PRIMARY_DIM) {
        has_new_target = 1;
        new_target_slot = primary - ZUL_PRIMARY_ATTACK_BASE;
    }

    OsrsPlayerCommand command = { .kind = OSRS_PLAYER_CMD_NONE };
    if (has_new_target) {
        command.kind = OSRS_PLAYER_CMD_TARGET;
        command.target_slot = new_target_slot;
        s->player_dest_explicit = 0;
        s->player_dest_x = -1;
        s->player_dest_y = -1;
    } else if (s->player_dest_explicit) {
        s->player_dest_explicit = 0;
        command.kind = OSRS_PLAYER_CMD_MOVE;
        command.move_kind = OSRS_PLAYER_MOVE_DESTINATION;
    } else if (primary > 0 && primary < ZUL_MOVE_DIM) {
        s->player_dest_x = s->player.x + ENCOUNTER_MOVE_TARGET_DX[primary];
        s->player_dest_y = s->player.y + ENCOUNTER_MOVE_TARGET_DY[primary];
        command.kind = OSRS_PLAYER_CMD_MOVE;
        command.move_kind = OSRS_PLAYER_MOVE_DESTINATION;
    } else {
        s->player_dest_x = -1;
        s->player_dest_y = -1;
    }

    OsrsPlayerStepInput step_input = {
        .player = &s->player,
        .interaction = &s->interaction,
        .target_lookup = zul_lookup_player_attack_target,
        .target_ctx = s,
        .command = command,
        .dest_x = &s->player_dest_x,
        .dest_y = &s->player_dest_y,
        .blocked_ticks = s->player_stunned_ticks,
        .arena = {
            .collision_map = (const CollisionMap*)s->collision_map,
            .world_offset_x = s->world_offset_x,
            .world_offset_y = s->world_offset_y,
            .is_walkable = zul_tile_walkable,
            .walkable_ctx = s,
            .los_query = osrs_los_open_query(),
            .arena_base_x = 0,
            .arena_base_y = 0,
            .arena_w = ZUL_ARENA_SIZE,
            .arena_h = ZUL_ARENA_SIZE,
        },
    };
    OsrsPlayerStepResult step_result = osrs_encounter_player_step(&step_input);
    s->player_moved_this_tick = step_result.moved;
    s->player_chased_target_this_tick = step_result.chased_target;

    if (s->respawn_timer > 0) {
        s->respawn_timer--;
        if (s->respawn_timer == 0)
            zul_start_active_kill(s);
    }

    if (osrs_interaction_active(&s->interaction) &&
        s->player.attack_timer == 0 && s->player_stunned_ticks == 0) {
        int target = s->interaction.target_slot;
        if (target == 0) {
            if (s->zulrah_visible && !s->is_diving) {
                if (s->player.spec_armed && s->player.special_energy >=
                        osrs_spec_cost(s->player.equipped[GEAR_SLOT_WEAPON])) {
                    zul_player_spec(s);
                    osrs_spec_disarm(&s->player.spec_armed);
                } else {
                    zul_player_attack(s);
                }
            }
        } else {
            zul_player_attack_snakeling(s, target - 1);
        }
    }

    if (s->zulrah_visible && s->zulrah.current_hitpoints <= 0) {
        zul_record_boss_kill(s);
        return;
    }

    zul_pending_cloud_tick(s);
    zul_cloud_tick(s);
    if (s->player.current_hitpoints <= 0) {
        zul_record_player_loss(s);
        return;
    }

    zul_phase_tick(s);

    zul_snakeling_tick(s);

    zul_thrall_tick(s);

    zul_venom_tick(s);

    OffensivePrayer prev_off_drain = s->player.offensive_prayer;
    encounter_drain_all_prayers(
        &s->player, encounter_player_prayer_bonus(&s->player));
    if (s->player.offensive_prayer != prev_off_drain)
        zul_mark_live_stats_dirty(s);

    if (s->player.current_hitpoints <= 0) {
        zul_record_player_loss(s);
        return;
    }
    if (s->tick >= ZUL_MAX_TICKS) {
        zul_record_episode_timeout(s);
        return;
    }
}

static void zul_step(EncounterState* state, EncounterContext* context, const int* actions) {
    (void)context;
    ZulrahState* s = (ZulrahState*)state;
    if (s->episode_over) return;
    zul_step_tick(s, actions);
    s->reward = zul_compute_reward(s);
    s->episode_return += s->reward;
}

static int zul_first_cell_with_kind(
    const ZulrahState* s, OsrsClickAction click, OsrsConsumableKind kind
) {
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
        OsrsInventoryClickResolution r = osrs_inventory_cell_click_interpret(
            &s->inventory_cells[cell], OSRS_CLICK_TICK_FIRST);
        if (r.click_action == click && r.consumable_kind == kind) return cell;
    }
    return -1;
}

static int zul_cell_with_item(const ZulrahState* s, uint8_t item_idx) {
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++)
        if (s->inventory_cells[cell].item_idx == item_idx) return cell;
    return -1;
}

static void zul_heuristic_gear_swap(
    const ZulrahState* s, int* actions, const uint8_t* loadout
) {
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        uint8_t want = loadout[slot];
        if (want == ITEM_NONE || s->player.equipped[slot] == want) continue;
        int cell = zul_cell_with_item(s, want);
        if (cell >= 0) actions[ZUL_HEAD_EQUIP_SLOT(slot)] = cell + 1;
    }
}

static void zul_heuristic_actions(ZulrahState* s, int* actions) {
    for (int i = 0; i < ZUL_NUM_ACTION_HEADS; i++) actions[i] = 0;

    int hp = s->player.current_hitpoints;

    if (s->zulrah_visible && !s->is_diving) {
        switch (s->current_form) {
            case ZUL_FORM_GREEN:
                if (s->player.prayer != PRAYER_PROTECT_RANGED)
                    actions[ZUL_HEAD_PRAYER] = ENCOUNTER_OVERHEAD_SET_REFRESH_RANGED;
                break;
            case ZUL_FORM_BLUE:
                if (s->player.prayer != PRAYER_PROTECT_MAGIC)
                    actions[ZUL_HEAD_PRAYER] = ENCOUNTER_OVERHEAD_SET_REFRESH_MAGIC;
                break;
            case ZUL_FORM_RED:
                if (s->player.prayer != PRAYER_PROTECT_MELEE)
                    actions[ZUL_HEAD_PRAYER] = ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE;
                break;
        }
        OffensivePrayer target_off = OFFENSIVE_PRAYER_NONE;
        if (s->current_form == ZUL_FORM_BLUE) target_off = OFFENSIVE_PRAYER_AUGURY;
        else if (s->current_form == ZUL_FORM_GREEN) target_off = OFFENSIVE_PRAYER_RIGOUR;
        else if (s->current_form == ZUL_FORM_RED) target_off = OFFENSIVE_PRAYER_PIETY;
        if (target_off != OFFENSIVE_PRAYER_NONE && s->player.offensive_prayer != target_off) {
            if (target_off == OFFENSIVE_PRAYER_AUGURY)
                actions[ZUL_HEAD_OFFENSIVE] = ENCOUNTER_OFFENSIVE_SET_REFRESH_AUGURY;
            else if (target_off == OFFENSIVE_PRAYER_RIGOUR)
                actions[ZUL_HEAD_OFFENSIVE] = ENCOUNTER_OFFENSIVE_SET_REFRESH_RIGOUR;
            else if (target_off == OFFENSIVE_PRAYER_PIETY)
                actions[ZUL_HEAD_OFFENSIVE] = ENCOUNTER_OFFENSIVE_SET_REFRESH_PIETY;
        }
    }

    if (s->player.potion_timer <= 0 && s->antivenom_timer <= 5) {
        int cell = zul_first_cell_with_kind(
            s, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_ANTIVENOM_PLUS);
        if (cell >= 0) {
            actions[ZUL_HEAD_DRINK] = cell + 1;
            return;
        }
    }

    if (hp < 60 && s->player.food_timer <= 0 &&
        hp <= s->player.base_hitpoints - osrs_food_heal_amount(FOOD_SHARK)) {
        int cell = zul_first_cell_with_kind(
            s, OSRS_CLICK_EAT, OSRS_CONSUMABLE_SHARK_FOOD);
        if (cell >= 0) actions[ZUL_HEAD_EAT] = cell + 1;
    }
    else if (hp < 40 && s->player.karambwan_timer <= 0 &&
             hp <= s->player.base_hitpoints - osrs_food_heal_amount(FOOD_KARAMBWAN)) {
        int cell = zul_first_cell_with_kind(
            s, OSRS_CLICK_EAT, OSRS_CONSUMABLE_KARAMBWAN);
        if (cell >= 0) actions[ZUL_HEAD_EAT] = cell + 1;
    }

    if (s->player.current_prayer < 30 && s->player.potion_timer <= 0 &&
        s->player.current_prayer < s->player.base_prayer) {
        int cell = zul_first_cell_with_kind(
            s, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_PRAYER_RESTORE);
        if (cell >= 0) actions[ZUL_HEAD_DRINK] = cell + 1;
    }

    {
        const ZulRotationPhase* phase = zul_current_phase(s);
        int stand = phase->stand;
        if (stand < ZUL_NUM_STAND_LOCATIONS) {
            int tx = ZUL_STAND_COORDS[stand][0];
            int ty = ZUL_STAND_COORDS[stand][1];
            if (tx != s->player.x || ty != s->player.y) {
                s->player_dest_x = tx;
                s->player_dest_y = ty;
                s->player_dest_explicit = 1;
            }
        }
    }

    if (s->zulrah_visible && !s->is_diving) {
        AttackStyle want_style = s->current_form == ZUL_FORM_BLUE
            ? ATTACK_STYLE_RANGED : ATTACK_STYLE_MAGIC;
        const uint8_t* want_loadout = s->current_form == ZUL_FORM_BLUE
            ? ZUL_RANGE_LOADOUT[s->gear_tier] : ZUL_MAGE_LOADOUT[s->gear_tier];
        if (zul_player_equipped_attack_style(s) != want_style) {
            zul_heuristic_gear_swap(s, actions, want_loadout);
        } else {
            actions[ZUL_HEAD_PRIMARY] = ZUL_PRIMARY_ATTACK_BASE;
            if (want_style == ATTACK_STYLE_RANGED) {
                int spec_cost = osrs_spec_cost(s->player.equipped[GEAR_SLOT_WEAPON]);
                if (spec_cost > 0 && s->player.special_energy >= spec_cost &&
                        !s->player.spec_armed)
                    actions[ZUL_HEAD_SPEC] = 1;
            }
        }
    }
}

static float zul_get_reward(EncounterState* state, EncounterContext* context) {
    (void)context;
    return ((ZulrahState*)state)->reward;
}
static int zul_is_terminal(EncounterState* state, EncounterContext* context) {
    (void)context;
    return ((ZulrahState*)state)->episode_over;
}

static int zul_get_entity_count(EncounterState* state, EncounterContext* context) {
    (void)context;
    ZulrahState* s = (ZulrahState*)state;
    int n = 2;
    for (int i = 0; i < ZUL_MAX_SNAKELINGS; i++)
        if (s->snakelings[i].active) n++;
    return n;
}
static void* zul_get_entity(EncounterState* state, EncounterContext* context, int index) {
    (void)context;
    ZulrahState* s = (ZulrahState*)state;
    if (index == 0) return &s->player;
    if (index == 1) return &s->zulrah;
    int si = 0;
    for (int i = 0; i < ZUL_MAX_SNAKELINGS; i++) {
        if (s->snakelings[i].active) {
            if (si + 2 == index) return &s->snakelings[i].entity;
            si++;
        }
    }
    fprintf(stderr, "zulrah get_entity invalid index %d\n", index);
    abort();
}

static void zul_fill_render_entities(
    EncounterState* state,
    EncounterContext* context,
    RenderEntity* out,
    int max_entities,
    int* count
) {
    (void)context;
    ZulrahState* s = (ZulrahState*)state;
    int n = 0;
    if (n < max_entities) osrs_render_entity_from_player_entity(&s->player, &out[n++]);
    if (n < max_entities) {
        osrs_render_entity_from_npc_player(
            &s->zulrah, &out[n], 0, s->zulrah_npc_instance_id);
        out[n].dest_x = s->zulrah.x + ZUL_NPC_SIZE / 2;
        out[n].dest_y = s->zulrah.y + ZUL_NPC_SIZE / 2;
        if (s->surface_timer > 0)
            out[n].render_movement_kind = RENDER_MOVEMENT_TELEPORT;
        if (s->is_diving && s->tick >= s->zulrah_anim_until_tick)
            out[n].npc_visible = 0;
        osrs_render_entity_suppress_pose_anims(
            &out[n], ZULRAH_ANIM_IDLE, ZULRAH_ANIM_IDLE);
        if (!zul_should_emit_npc_anim_event(s)) {
            out[n].npc_anim_id = -1;
        }
        n++;
    }
    for (int i = 0; i < ZUL_MAX_SNAKELINGS && n < max_entities; i++) {
        if (s->snakelings[i].active) {
            osrs_render_entity_from_npc_player(
                &s->snakelings[i].entity,
                &out[n],
                i + 1,
                s->snakelings[i].npc_instance_id);
            osrs_render_entity_suppress_pose_anims(
                &out[n], SNAKELING_ANIM_IDLE, SNAKELING_ANIM_WALK);
            n++;
            int adx = abs(s->snakelings[i].entity.x - s->player.x);
            int ady = abs(s->snakelings[i].entity.y - s->player.y);
            if (adx <= 1 && ady <= 1)
                out[n - 1].attack_target_entity_idx = 0;
        }
    }
    if ((s->player_attacked_this_tick || s->player_chased_target_this_tick ||
                (osrs_interaction_active(&s->interaction) && !s->player_moved_this_tick)) &&
            s->zulrah_visible && !s->is_diving) {
        encounter_resolve_attack_target(out, n, 0);
    }
    int attack_anim_active = s->zulrah.npc_anim_id >= 0 &&
        s->zulrah.npc_anim_id != ZULRAH_ANIM_SURFACE &&
        s->zulrah.npc_anim_id != ZULRAH_ANIM_RISE &&
        s->zulrah.npc_anim_id != ZULRAH_ANIM_DIVE &&
        s->zulrah.npc_anim_id != ZULRAH_ANIM_DEATH &&
        s->tick < s->zulrah_anim_until_tick;
    if ((s->attack_event_count > 0 || s->cloud_event_count > 0 ||
            s->melee_pending || attack_anim_active) &&
        s->zulrah_visible && !s->is_diving && n > 1)
        out[1].attack_target_entity_idx = 0;
    *count = n;
}

static void zul_put_int(EncounterState* state, EncounterContext* context, const char* key, int value) {
    (void)context;
    ZulrahState* s = (ZulrahState*)state;
    if (strcmp(key, "seed") == 0) s->rng_state = (uint32_t)value;
    else if (strcmp(key, "world_offset_x") == 0) s->world_offset_x = value;
    else if (strcmp(key, "world_offset_y") == 0) s->world_offset_y = value;
    else if (strcmp(key, "gear_tier") == 0) {
        s->gear_tier_fixed = encounter_require_int_range_config(
            "zulrah", key, value, 0, ZUL_NUM_GEAR_TIERS - 1);
        s->gear_tier = s->gear_tier_fixed;
    }
    else if (strcmp(key, "gear_tier_mode") == 0) {
        s->gear_tier_mode = encounter_require_int_range_config(
            "zulrah", key, value, ZUL_GEAR_TIER_FIXED, ZUL_GEAR_TIER_WEIGHTED);
    }
    else if (strcmp(key, "episode_mode") == 0) {
        s->episode_mode = encounter_require_int_range_config(
            "zulrah", key, value, ZUL_EPISODE_SINGLE_KILL, ZUL_EPISODE_TRIP);
    }
    else if (strcmp(key, "player_dest_x") == 0) {
        s->player_dest_x = value;
        if (value >= 0) s->player_dest_explicit = 1;
    }
    else if (strcmp(key, "player_dest_y") == 0) {
        s->player_dest_y = value;
        if (value >= 0) s->player_dest_explicit = 1;
    }
    else if (strcmp(key, "human_command_mode") == 0)
        s->human_command_mode = encounter_require_binary_config("zulrah", key, value);
    else encounter_abort_unknown_config("zulrah", "int", key);
}
static float zul_require_nonnegative_float_config(const char* k, float v) {
    if (v < 0.0f) {
        fprintf(stderr, "zulrah config %s must be >= 0, got %.6f\n", k, v);
        abort();
    }
    return v;
}
static void zul_put_float(EncounterState* st, EncounterContext* context, const char* k, float v) {
    (void)context;
    ZulrahState* s = (ZulrahState*)st;
    if (strncmp(k, "gear_tier_weight_", 17) == 0) {
        int idx = k[17] - '0';
        if (k[18] != '\0' || idx < 0 || idx >= ZUL_NUM_GEAR_TIERS) {
            encounter_abort_unknown_config("zulrah", "float", k);
        }
        s->gear_tier_weights[idx] = zul_require_nonnegative_float_config(k, v);
    }
    else if (strcmp(k, "reward_win") == 0) {
        s->reward_config.win = zul_require_nonnegative_float_config(k, v);
    }
    else if (strcmp(k, "reward_loss_penalty") == 0) {
        s->reward_config.loss_penalty = zul_require_nonnegative_float_config(k, v);
    }
    else if (strcmp(k, "reward_damage_dealt") == 0) {
        s->reward_config.damage_dealt = zul_require_nonnegative_float_config(k, v);
    }
    else if (strcmp(k, "reward_correct_style") == 0) {
        s->reward_config.correct_style = zul_require_nonnegative_float_config(k, v);
    }
    else if (strcmp(k, "reward_damage_received_penalty") == 0) {
        s->reward_config.damage_received_penalty =
            zul_require_nonnegative_float_config(k, v);
    }
    else if (strcmp(k, "reward_cloud_occupancy_penalty") == 0) {
        s->reward_config.cloud_occupancy_penalty =
            zul_require_nonnegative_float_config(k, v);
    }
    else encounter_abort_unknown_config("zulrah", "float", k);
}
static void zul_put_ptr(EncounterState* st, EncounterContext* context, const char* k, void* v) {
    (void)context;
    ZulrahState* s = (ZulrahState*)st;
    if (strcmp(k, "collision_map") == 0) s->collision_map = v;
    else encounter_abort_unknown_config("zulrah", "ptr", k);
}

static void* zul_get_log(EncounterState* state, EncounterContext* context) {
    (void)context;
    ZulrahState* s = (ZulrahState*)state;
    if (s->episode_over) {
        s->log.episode_return += s->episode_return;
        s->log.episode_length += (float)s->tick;
        float kills = (float)s->kills_this_episode;
        ZulEpisodeOutcome outcome = zul_episode_outcome(s);
        s->log.wins += outcome.win;
        s->log.zulrah_kills += kills;
        s->log.damage_dealt += s->total_damage_dealt;
        s->log.damage_received += s->total_damage_received;
        s->log.cloud_occupancy_ticks += (float)s->total_cloud_occupancy_ticks;
        s->log.cloud_occupancy_frac += s->tick > 0
            ? (float)s->total_cloud_occupancy_ticks / (float)s->tick
            : 0.0f;
        s->log.cloud_damage_received += s->total_cloud_damage_received;
        s->log.active_cloud_count_ticks += (float)s->total_active_cloud_ticks;
        s->log.pending_cloud_count_ticks += (float)s->total_pending_cloud_ticks;
        int tier = s->gear_tier;
        if (tier < 0 || tier >= ZUL_NUM_GEAR_TIERS) {
            fprintf(stderr, "zulrah invalid sampled gear tier %d\n", tier);
            abort();
        }
        s->log.zulrah_tier_n[tier] += 1.0f;
        s->log.zulrah_tier_wins[tier] += outcome.win;
        s->log.zulrah_tier_score_sum[tier] += outcome.score;
        s->log.zulrah_tier_damage_received[tier] += s->total_damage_received;
        s->log.zulrah_tier_episode_length[tier] += (float)s->tick;
        s->log.zulrah_tier_cloud_occupancy_ticks[tier] +=
            (float)s->total_cloud_occupancy_ticks;
        s->log.zulrah_tier_cloud_damage_received[tier] +=
            s->total_cloud_damage_received;
        s->log.n += 1.0f;
    }
    return &s->log;
}
static int zul_get_tick(EncounterState* state, EncounterContext* context) {
    (void)context;
    return ((ZulrahState*)state)->tick;
}

static void zul_emit_player_projectile_profile(
    ZulrahState* s,
    EncounterOverlay* ov,
    const OsrsCombatProjectileProfile* profile,
    int style,
    int damage,
    int sequence_index,
    int sequence_count
) {
    if (!profile || profile->projectile_model_id <= 0) {
        uint8_t weapon = s->player.equipped[GEAR_SLOT_WEAPON];
        int item_id = weapon < NUM_ITEMS ? ITEM_DATABASE[weapon].item_id : -1;
        fprintf(stderr, "zulrah: missing player projectile model for item %d style %d\n",
            item_id, style);
        abort();
    }

    int target_size = ZUL_NPC_SIZE;
    int fallback_start_h = 64;
    int fallback_end_h = (int)(target_size * 0.5f * 128);
    int p_duration = s->player_attack_timing.visual_duration_ticks * 30;
    int p_start_delay = s->player_attack_timing.visual_start_delay_ticks * 30;
    int visual_damage = sequence_index == sequence_count - 1 ? damage : 0;

    OsrsCombatProjectileEmitSpec emit_spec = {
        .src_x = s->player.x,
        .src_y = s->player.y,
        .dst_x = s->zulrah.x,
        .dst_y = s->zulrah.y,
        .src_size = 1,
        .dst_size = target_size,
        .target_npc_slot = 0,
        .attack_style = (AttackStyle)style,
        .damage = visual_damage,
        .duration_ticks = p_duration,
        .start_delay = p_start_delay,
        .fallback_start_h = fallback_start_h,
        .fallback_end_h = fallback_end_h,
        .curve = 16,
        .splash_gfx_id = GFX_SPLASH,
    };
    (void)osrs_emit_combat_projectile_profile_player_to_npc(ov, profile, &emit_spec);
}

static void zul_emit_player_attack_projectiles(ZulrahState* s, EncounterOverlay* ov) {
    if (!s->player_attacked_this_tick ||
            s->player_attack_style_id == ATTACK_STYLE_MELEE) {
        return;
    }

    uint8_t weapon = s->player.equipped[GEAR_SLOT_WEAPON];
    int style = s->player_attack_style_id;
    if (style == ATTACK_STYLE_MAGIC) {
        const OsrsCombatProjectileProfile* profile = NULL;
        if (s->player_attack_is_special && weapon < NUM_ITEMS) {
            const OsrsCombatVisualRow* effect =
                osrs_combat_visual_find_special_projectile_item_id(
                    ITEM_DATABASE[weapon].item_id, ATTACK_STYLE_MAGIC);
            if (effect) profile = &effect->projectile;
        }
        if (!profile) profile = osrs_combat_visual_magic_projectile_profile(weapon);
        zul_emit_player_projectile_profile(
            s, ov, profile, style, s->player_attack_dmg, 0, 1);
        return;
    }

    const OsrsCombatProjectileProfile* base_profile =
        osrs_combat_visual_ranged_projectile_profile(
            weapon, OSRS_COMBAT_PROJECTILE_NONE);
    const OsrsCombatVisualRow* effect = NULL;
    if (s->player_attack_is_special && weapon < NUM_ITEMS) {
        effect = osrs_combat_visual_find_special_projectile_item_id(
            ITEM_DATABASE[weapon].item_id, ATTACK_STYLE_RANGED);
    }
    OsrsCombatProjectileSequencePart parts[OSRS_COMBAT_PROJECTILE_SEQUENCE_MAX];
    int part_count = osrs_combat_visual_build_projectile_sequence(
        base_profile, effect, parts, OSRS_COMBAT_PROJECTILE_SEQUENCE_MAX);
    if (part_count <= 0) {
        int item_id = weapon < NUM_ITEMS ? ITEM_DATABASE[weapon].item_id : -1;
        fprintf(stderr, "zulrah: missing ranged projectile sequence for item %d\n",
            item_id);
        abort();
    }
    for (int i = 0; i < part_count; i++) {
        zul_emit_player_projectile_profile(
            s, ov, &parts[i].projectile, style, s->player_attack_dmg,
            parts[i].sequence_index, parts[i].sequence_count);
    }
}

static void zul_render_post_tick(EncounterState* state, EncounterContext* context, EncounterOverlay* ov) {
    (void)context;
    ZulrahState* s = (ZulrahState*)state;

    ov->tile_shadow_count = 0;
    ov->hazard_count = 0;
    for (int i = 0; i < ZUL_MAX_CLOUDS && ov->hazard_count < ENCOUNTER_MAX_OVERLAY_TILES; i++) {
        if (!s->clouds[i].active) continue;
        ov->hazards[ov->hazard_count].x = s->clouds[i].x;
        ov->hazards[ov->hazard_count].y = s->clouds[i].y;
        ov->hazards[ov->hazard_count].active = 1;
        ov->hazard_count++;
    }

    ov->boss_x = s->zulrah.x;
    ov->boss_y = s->zulrah.y;
    ov->boss_visible = s->zulrah_visible;
    ov->boss_form = (int)s->current_form;
    ov->boss_size = ZUL_NPC_SIZE;

    ov->add_count = 0;

    ov->melee_target_active = s->melee_pending;
    ov->melee_target_x = s->melee_target_x;
    ov->melee_target_y = s->melee_target_y;

    ov->projectile_count = 0;
    for (int i = 0; i < s->attack_event_count; i++) {
        if (s->attack_events[i].style == 4) {
            OsrsProjectileEventSpec spawn_spec = {
                .src_x = s->attack_events[i].src_x,
                .src_y = s->attack_events[i].src_y,
                .dst_x = s->attack_events[i].dst_x,
                .dst_y = s->attack_events[i].dst_y,
                .style = 4,
                .damage = 0,
                .duration_ticks = 40,
                .start_h = 100,
                .end_h = 0,
                .curve = 12,
                .arc_height = 0.0f,
                .src_size = ZUL_NPC_SIZE,
                .dst_size = 1,
                .model_id = GFX_SNAKELING_SPAWN_MODEL,
                .anim_id = GFX_SNAKELING_SPAWN_ANIM,
            };
            int pi = osrs_emit_projectile_with_spec(ov, &spawn_spec, 0);
            encounter_set_projectile_source_npc_slot(ov, pi, 0);
        } else if (s->attack_events[i].style == 2) {
            continue;
        } else {
            uint32_t zul_proj_model = (s->attack_events[i].style == 0)
                ? GFX_RANGED_PROJ_MODEL : GFX_MAGIC_PROJ_MODEL;
            int zul_proj_anim = (s->attack_events[i].style == 0)
                ? GFX_RANGED_PROJ_ANIM : GFX_MAGIC_PROJ_ANIM;
            OsrsProjectileEventSpec attack_spec = {
                .src_x = s->attack_events[i].src_x,
                .src_y = s->attack_events[i].src_y,
                .dst_x = s->attack_events[i].dst_x,
                .dst_y = s->attack_events[i].dst_y,
                .style = s->attack_events[i].style,
                .damage = s->attack_events[i].damage,
                .duration_ticks = 35,
                .start_h = 480,
                .end_h = 64,
                .curve = 16,
                .arc_height = 0.0f,
                .src_size = ZUL_NPC_SIZE,
                .dst_size = 1,
                .model_id = zul_proj_model,
                .anim_id = zul_proj_anim,
            };
            (void)osrs_emit_projectile_npc_to_player(ov, &attack_spec, 0);
        }
    }
    for (int i = 0; i < s->cloud_event_count; i++) {
        OsrsProjectileEventSpec cloud_spec = {
            .src_x = s->cloud_events[i].src_x,
            .src_y = s->cloud_events[i].src_y,
            .dst_x = s->cloud_events[i].dst_x,
            .dst_y = s->cloud_events[i].dst_y,
            .style = 3,
            .damage = 0,
            .duration_ticks = s->cloud_events[i].flight_ticks * 30,
            .start_h = 200,
            .end_h = 0,
            .curve = 10,
            .arc_height = 3.0f,
            .src_size = ZUL_NPC_SIZE,
            .dst_size = 1,
            .model_id = GFX_CLOUD_PROJ_MODEL,
            .anim_id = GFX_CLOUD_PROJ_ANIM,
        };
        int pi = osrs_emit_projectile_with_spec(ov, &cloud_spec, 0);
        encounter_set_projectile_source_npc_slot(ov, pi, 0);
    }
    zul_emit_player_attack_projectiles(s, ov);
}
static int zul_get_winner(EncounterState* state, EncounterContext* context) {
    (void)context;
    return ((ZulrahState*)state)->winner;
}

static void zul_translate_human_commands(HumanInput* hi, int* actions, ZulrahState* s) {
    for (int h = 0; h < ZUL_NUM_ACTION_HEADS; h++) actions[h] = 0;

    int path_command_seen = 0;
    for (int i = 0; i < hi->commands.count; i++) {
        const HumanCommand* cmd = &hi->commands.items[i];
        switch (cmd->kind) {
            case HUMAN_COMMAND_WALK:
                path_command_seen = 1;
                s->player_dest_x = cmd->world_x;
                s->player_dest_y = cmd->world_y;
                s->player_dest_explicit = 1;
                actions[ZUL_HEAD_PRIMARY] = 0;
                osrs_interaction_clear(&s->interaction);
                break;
            case HUMAN_COMMAND_ATTACK_NPC:
            case HUMAN_COMMAND_SPELL_TARGET:
                path_command_seen = 1;
                actions[ZUL_HEAD_PRIMARY] = ZUL_PRIMARY_ATTACK_BASE;
                s->player_dest_x = -1;
                s->player_dest_y = -1;
                s->player_dest_explicit = 0;
                break;
            case HUMAN_COMMAND_OVERHEAD_PRAYER:
                actions[ZUL_HEAD_PRAYER] = cmd->overhead_prayer;
                break;
            case HUMAN_COMMAND_OFFENSIVE_PRAYER:
                actions[ZUL_HEAD_OFFENSIVE] = cmd->offensive_prayer;
                break;
            case HUMAN_COMMAND_EAT: {
                int cell = zul_first_cell_with_kind(s, OSRS_CLICK_EAT,
                    cmd->food == 1 ? OSRS_CONSUMABLE_KARAMBWAN : OSRS_CONSUMABLE_SHARK_FOOD);
                if (cell >= 0) actions[ZUL_HEAD_EAT] = cell + 1;
                break;
            }
            case HUMAN_COMMAND_DRINK: {
                OsrsConsumableKind kind = OSRS_CONSUMABLE_NONE;
                if (cmd->potion == POTION_RESTORE || cmd->potion == POTION_PRAYER_POT)
                    kind = OSRS_CONSUMABLE_PRAYER_RESTORE;
                else if (cmd->potion == POTION_ANTIVENOM)
                    kind = OSRS_CONSUMABLE_ANTIVENOM_PLUS;
                if (kind != OSRS_CONSUMABLE_NONE) {
                    int cell = zul_first_cell_with_kind(s, OSRS_CLICK_DRINK, kind);
                    if (cell >= 0) actions[ZUL_HEAD_DRINK] = cell + 1;
                }
                break;
            }
            case HUMAN_COMMAND_SPEC_TOGGLE:
                actions[ZUL_HEAD_SPEC] = s->player.spec_armed ? 2 : 1;
                break;
            case HUMAN_COMMAND_EQUIP_INVENTORY_ITEM:
            case HUMAN_COMMAND_FIGHT_STYLE:
            case HUMAN_COMMAND_SET_AUTOCAST:
            case HUMAN_COMMAND_ITEM_ON_ITEM:
            case HUMAN_COMMAND_ITEM_ON_WIDGET:
            case HUMAN_COMMAND_SPELL_ON_WIDGET:
            case HUMAN_COMMAND_INVENTORY_PRIMARY_CLICK:
            case HUMAN_COMMAND_NONE:
                break;
        }
    }

    if (!path_command_seen && hi->pending_move_x >= 0 && hi->pending_move_y >= 0) {
        s->player_dest_x = hi->pending_move_x;
        s->player_dest_y = hi->pending_move_y;
        s->player_dest_explicit = 1;
        actions[ZUL_HEAD_PRIMARY] = 0;
        osrs_interaction_clear(&s->interaction);
    }
}

static void zul_step_human_commands(EncounterState* state, EncounterContext* context, HumanInput* hi) {
    ZulrahState* s = (ZulrahState*)state;
    int actions[ZUL_NUM_ACTION_HEADS];
    s->human_command_mode = 1;
    s->human_commands = hi->commands.items;
    s->human_command_count = hi->commands.count;
    zul_mark_live_stats_dirty(s);
    zul_translate_human_commands(hi, actions, s);
    zul_step(state, context, actions);
    s->human_commands = NULL;
    s->human_command_count = 0;
    human_input_clear_pending(hi);
}

typedef struct {
    int unused;
} ZulrahContext;

static void zul_init_context(EncounterContext* context) {
    (void)context;
}

static void zul_destroy_context(EncounterContext* context) {
    (void)context;
}

static void zul_init_state_ctx(EncounterState* state, EncounterContext* context) {
    (void)context;
    memset(state, 0, sizeof(ZulrahState));
    ((ZulrahState*)state)->reward_config = zul_default_reward_config();
}

static const EncounterDef ENCOUNTER_ZULRAH = {
    .name = "zulrah",
    .obs_size = ZUL_NUM_OBS,
    .num_action_heads = ZUL_NUM_ACTION_HEADS,
    .action_head_dims = ZUL_ACTION_HEAD_DIMS,
    .mask_size = ZUL_ACTION_MASK_SIZE,
    .state_size = sizeof(ZulrahState),
    .context_size = sizeof(ZulrahContext),
    .init_context = zul_init_context,
    .destroy_context = zul_destroy_context,
    .init_state = zul_init_state_ctx,
    .create = zul_create,
    .destroy = zul_destroy,
    .reset = zul_reset,
    .step = zul_step,
    .step_human_commands = zul_step_human_commands,
    .write_obs = zul_write_obs,
    .write_mask = zul_write_mask,
    .get_reward = zul_get_reward,
    .is_terminal = zul_is_terminal,
    .get_entity_count = zul_get_entity_count,
    .get_entity = zul_get_entity,
    .fill_render_entities = zul_fill_render_entities,
    .put_int = zul_put_int,
    .put_float = zul_put_float,
    .put_ptr = zul_put_ptr,
    .arena_base_x = 0,
    .arena_base_y = 0,
    .arena_width = ZUL_ARENA_SIZE,
    .arena_height = ZUL_ARENA_SIZE,
    .head_move = ZUL_HEAD_PRIMARY,
    .head_prayer = ZUL_HEAD_PRAYER,
    .head_target = -1,

    .render_post_tick = zul_render_post_tick,
    .get_log = zul_get_log,
    .get_tick = zul_get_tick,
    .get_winner = zul_get_winner,
};

__attribute__((constructor))
static void zul_register(void) {
    encounter_register(&ENCOUNTER_ZULRAH);
}

#endif
