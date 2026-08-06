#ifndef OSRS_INVENTORY_CLICKS_H
#define OSRS_INVENTORY_CLICKS_H

#include "osrs_inventory.h"
#include "osrs_items.h"
#include "osrs_consumables.h"
#include "osrs_special_attacks.h"

typedef enum {
    OSRS_CLICK_NONE = 0,
    OSRS_CLICK_EQUIP = 1,
    OSRS_CLICK_EAT = 2,
    OSRS_CLICK_DRINK = 3,
} OsrsClickAction;

typedef enum {
    OSRS_CONSUMABLE_NONE = 0,
    OSRS_CONSUMABLE_BREW = 1,
    OSRS_CONSUMABLE_SUPER_RESTORE = 2,
    OSRS_CONSUMABLE_SANFEW = 3,
    OSRS_CONSUMABLE_SUPER_COMBAT = 4,
    OSRS_CONSUMABLE_DIVINE_COMBAT = 5,
    OSRS_CONSUMABLE_RANGING = 6,
    OSRS_CONSUMABLE_DIVINE_RANGING = 7,
    OSRS_CONSUMABLE_SURGE = 8,
    OSRS_CONSUMABLE_GUTHIX_REST = 9,
    OSRS_CONSUMABLE_SATURATED_HEART = 10,
    OSRS_CONSUMABLE_ANTIVENOM_PLUS = 11,
    OSRS_CONSUMABLE_SHARK_FOOD = 12,
    OSRS_CONSUMABLE_KARAMBWAN = 13,
    OSRS_CONSUMABLE_PRAYER_RESTORE = 14,
    OSRS_CONSUMABLE_BASTION = 15,
    OSRS_CONSUMABLE_STAMINA = 16,
} OsrsConsumableKind;

typedef enum {
    OSRS_CLICK_TICK_FIRST = 0,
    OSRS_CLICK_TICK_DUPLICATE = 1,
} OsrsClickTickMultiplicity;

typedef struct {
    uint16_t raw_osrs_id;
    OsrsClickAction click_action;
    OsrsConsumableKind consumable_kind;
    uint8_t dose_count;
} OsrsConsumableClick;

typedef struct {
    OsrsClickAction click_action;
    OsrsConsumableKind consumable_kind;
    uint8_t dose_count;
    uint16_t raw_osrs_id_after_drink;
} OsrsInventoryClickResolution;

typedef struct {
    int consumed;
    OsrsConsumableKind consumable_kind;
    uint8_t dose_count_before;
    uint8_t dose_count_after;
    uint16_t raw_osrs_id_before;
    uint16_t raw_osrs_id_after_drink;
} OsrsInventoryDrinkConsumeResult;

typedef void (*OsrsInventoryDrinkOneDoseEffectFn)(
    void* ctx,
    OsrsConsumableKind kind
);

#define OSRS_INVENTORY_CELL_OBS_FEATURES 28
#define OSRS_INVENTORY_CELL_OBS_SHARED 5
#define OSRS_INVENTORY_CELL_OBS_KIND_UNION 10
#define OSRS_INVENTORY_CELL_OBS_FEATURES_COMPACT \
    (OSRS_INVENTORY_CELL_OBS_SHARED + OSRS_INVENTORY_CELL_OBS_KIND_UNION)

/* Slot names for the compact record. The coded observation ships three of these and an
   item-table gather rebuilds the rest, so both sides must agree on where they sit. */
#define OSRS_INVENTORY_CELL_COMPACT_PRESENT   0
#define OSRS_INVENTORY_CELL_COMPACT_EQUIPPED  1
#define OSRS_INVENTORY_CELL_COMPACT_DOSE      2
#define OSRS_INVENTORY_CELL_COMPACT_IS_ARMOR  3
#define OSRS_INVENTORY_CELL_COMPACT_IS_WEAPON 4
/* Union slot: effect_class4[2] on a gear cell, hp_heal on a consumable cell. */
#define OSRS_INVENTORY_CELL_COMPACT_HP_HEAL   (OSRS_INVENTORY_CELL_OBS_SHARED + 5)

/* The coded observation: everything a cell carries that the item alone does not determine.
   hp_heal is here because it divides by base_hitpoints, which the Frailty modifier rewrites
   mid-episode; dose and the other ten features are pure item facts and live in the table. */
#define OSRS_INVENTORY_CELL_OBS_CODE     0
#define OSRS_INVENTORY_CELL_OBS_EQUIPPED 1
#define OSRS_INVENTORY_CELL_OBS_HP_HEAL  2
#define OSRS_INVENTORY_CELL_OBS_FEATURES_CODED 3
#define OSRS_EQUIPPED_SELF_OBS_FEATURES 18

static const OsrsConsumableClick OSRS_CONSUMABLE_CLICK_REGISTRY[] = {
    {6685, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_BREW, 4},
    {6687, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_BREW, 3},
    {6689, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_BREW, 2},
    {6691, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_BREW, 1},
    {3024, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SUPER_RESTORE, 4},
    {3026, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SUPER_RESTORE, 3},
    {3028, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SUPER_RESTORE, 2},
    {3030, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SUPER_RESTORE, 1},
    {10925, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SANFEW, 4},
    {10927, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SANFEW, 3},
    {10929, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SANFEW, 2},
    {10931, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SANFEW, 1},
    {12695, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SUPER_COMBAT, 4},
    {12697, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SUPER_COMBAT, 3},
    {12699, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SUPER_COMBAT, 2},
    {12701, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SUPER_COMBAT, 1},
    {23685, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_DIVINE_COMBAT, 4},
    {23688, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_DIVINE_COMBAT, 3},
    {23691, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_DIVINE_COMBAT, 2},
    {23694, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_DIVINE_COMBAT, 1},
    {2444, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_RANGING, 4},
    {169, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_RANGING, 3},
    {171, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_RANGING, 2},
    {173, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_RANGING, 1},
    {23733, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_DIVINE_RANGING, 4},
    {23736, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_DIVINE_RANGING, 3},
    {23739, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_DIVINE_RANGING, 2},
    {23742, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_DIVINE_RANGING, 1},
    {30875, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SURGE, 4},
    {30878, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SURGE, 3},
    {30881, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SURGE, 2},
    {30884, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SURGE, 1},
    {4417, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_GUTHIX_REST, 4},
    {4419, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_GUTHIX_REST, 3},
    {4421, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_GUTHIX_REST, 2},
    {4423, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_GUTHIX_REST, 1},
    {27641, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_SATURATED_HEART, 1},
    {12913, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_ANTIVENOM_PLUS, 4},
    {12915, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_ANTIVENOM_PLUS, 3},
    {12917, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_ANTIVENOM_PLUS, 2},
    {12919, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_ANTIVENOM_PLUS, 1},
    {2434, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_PRAYER_RESTORE, 4},
    {139, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_PRAYER_RESTORE, 3},
    {141, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_PRAYER_RESTORE, 2},
    {143, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_PRAYER_RESTORE, 1},
    {22461, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_BASTION, 4},
    {22464, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_BASTION, 3},
    {22467, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_BASTION, 2},
    {22470, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_BASTION, 1},
    {12625, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_STAMINA, 4},
    {12627, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_STAMINA, 3},
    {12629, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_STAMINA, 2},
    {12631, OSRS_CLICK_DRINK, OSRS_CONSUMABLE_STAMINA, 1},
    {385, OSRS_CLICK_EAT, OSRS_CONSUMABLE_SHARK_FOOD, 0},
    {3144, OSRS_CLICK_EAT, OSRS_CONSUMABLE_KARAMBWAN, 0},
};

static inline OsrsConsumableClick osrs_consumable_click_lookup_raw_osrs_id(
    uint16_t raw_osrs_id
);

static inline void osrs_inventory_clicks_trap(void) {
#if defined(__clang__) || defined(__GNUC__)
    __builtin_trap();
#else
    *(volatile int*)0 = 0;
#endif
}

static inline float osrs_clamp_unit(float v) {
    if (v < -1.0f) return -1.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

typedef enum {
    COL_CKIND6_NONE = 0,
    COL_CKIND6_BREW,
    COL_CKIND6_RESTORE,
    COL_CKIND6_COMBAT_BOOST,
    COL_CKIND6_RANGED_BOOST,
    COL_CKIND6_SPECIAL,
    COL_CKIND6_FOOD,
} OsrsConsumableKind6;

static inline OsrsConsumableKind6 col_consumable_kind6(OsrsConsumableKind k) {
    switch (k) {
        case OSRS_CONSUMABLE_BREW:            return COL_CKIND6_BREW;
        case OSRS_CONSUMABLE_SUPER_RESTORE:
        case OSRS_CONSUMABLE_SANFEW:
        case OSRS_CONSUMABLE_PRAYER_RESTORE:  return COL_CKIND6_RESTORE;
        case OSRS_CONSUMABLE_SUPER_COMBAT:
        case OSRS_CONSUMABLE_DIVINE_COMBAT:   return COL_CKIND6_COMBAT_BOOST;
        case OSRS_CONSUMABLE_RANGING:
        case OSRS_CONSUMABLE_DIVINE_RANGING:
        case OSRS_CONSUMABLE_BASTION:         return COL_CKIND6_RANGED_BOOST;
        case OSRS_CONSUMABLE_SURGE:
        case OSRS_CONSUMABLE_GUTHIX_REST:
        case OSRS_CONSUMABLE_SATURATED_HEART:
        case OSRS_CONSUMABLE_ANTIVENOM_PLUS:
        case OSRS_CONSUMABLE_STAMINA:         return COL_CKIND6_SPECIAL;
        case OSRS_CONSUMABLE_SHARK_FOOD:
        case OSRS_CONSUMABLE_KARAMBWAN:       return COL_CKIND6_FOOD;
        case OSRS_CONSUMABLE_NONE:            return COL_CKIND6_NONE;
    }
    osrs_inventory_clicks_trap();
    return COL_CKIND6_NONE;
}

static inline int osrs_consumable_hp_heal_amount(OsrsConsumableKind k, int base_hp) {
    switch (k) {
        case OSRS_CONSUMABLE_BREW:       return osrs_brew_heal_amount(base_hp);
        case OSRS_CONSUMABLE_SHARK_FOOD: return osrs_food_heal_amount(FOOD_SHARK);
        case OSRS_CONSUMABLE_KARAMBWAN:  return osrs_food_heal_amount(FOOD_KARAMBWAN);
        default:                         return 0;
    }
}

static inline int osrs_consumable_prayer_restore_amount(OsrsConsumableKind k, int base_prayer) {
    switch (k) {
        case OSRS_CONSUMABLE_SUPER_RESTORE: return osrs_super_restore_amount(base_prayer);
        case OSRS_CONSUMABLE_PRAYER_RESTORE: return osrs_prayer_potion_restore_amount(base_prayer);
        case OSRS_CONSUMABLE_SANFEW:        return osrs_sanfew_restore_amount(base_prayer);
        default:                            return 0;
    }
}

static inline int osrs_consumable_offensive_boost_amount(OsrsConsumableKind k, int base_level) {
    switch (k) {
        case OSRS_CONSUMABLE_SUPER_COMBAT:
        case OSRS_CONSUMABLE_DIVINE_COMBAT:   return osrs_super_combat_boost_amount(base_level);
        case OSRS_CONSUMABLE_RANGING:
        case OSRS_CONSUMABLE_DIVINE_RANGING:
        case OSRS_CONSUMABLE_BASTION:         return osrs_ranging_boost_amount(base_level);
        case OSRS_CONSUMABLE_SATURATED_HEART: return osrs_saturated_heart_magic_boost(base_level);
        default:                              return 0;
    }
}

static inline void osrs_item_effect_class4(uint32_t effect_mask, float out[4]) {
    uint32_t lifesteal = OSRS_ITEM_EFFECT_BLOOD_FURY | OSRS_ITEM_EFFECT_SANG_HEAL;
    uint32_t damage_amp = OSRS_ITEM_EFFECT_TWISTED_BOW | OSRS_ITEM_EFFECT_FANG |
        OSRS_ITEM_EFFECT_TUMEKENS_SHADOW | OSRS_ITEM_EFFECT_DHAROK_PIECE |
        OSRS_ITEM_EFFECT_DRAGON_HUNTER_WAND | OSRS_ITEM_EFFECT_VENATOR_BOUNCE;
    uint32_t defensive = OSRS_ITEM_EFFECT_ELYSIAN | OSRS_ITEM_EFFECT_CRYSTAL_ARMOUR |
        OSRS_ITEM_EFFECT_RECOIL_RING | OSRS_ITEM_EFFECT_VENOM_IMMUNE |
        OSRS_ITEM_EFFECT_ECHO_BOOTS | OSRS_ITEM_EFFECT_CONFLICTION |
        OSRS_ITEM_EFFECT_VIRTUS_PIECE;
    uint32_t util = OSRS_ITEM_EFFECT_LIGHTBEARER;
    out[0] = (effect_mask & lifesteal)  ? 1.0f : 0.0f;
    out[1] = (effect_mask & damage_amp) ? 1.0f : 0.0f;
    out[2] = (effect_mask & defensive)  ? 1.0f : 0.0f;
    out[3] = (effect_mask & util)       ? 1.0f : 0.0f;
}

static inline uint8_t osrs_item_index_for_raw_osrs_id(uint16_t raw_osrs_id) {
    for (int i = 0; i < NUM_ITEMS; i++) {
        if (ITEM_DATABASE[i].item_id == raw_osrs_id) return (uint8_t)i;
    }
    return ITEM_NONE;
}

typedef struct {
    float present;
    float is_equipped;
    float dose;
    float is_armor;
    float is_weapon;
    float style3[3];
    float post_use_deltas[6];
    float effect_class4[4];
    float attack_speed;
    float attack_range;
    float spec_cost;
    float kind5[5];
    float hp_heal;
    float prayer_restore;
    float offensive_boost;
} OsrsInventoryCellAffordance;

static inline OsrsInventoryCellAffordance osrs_inventory_cell_affordance(
    uint8_t item_idx,
    uint16_t raw_osrs_id,
    uint8_t dose,
    int is_equipped,
    const float post_use_deltas[6],
    int base_hitpoints,
    int base_prayer,
    int base_level
) {
    OsrsConsumableClick consumable =
        osrs_consumable_click_lookup_raw_osrs_id(raw_osrs_id);
    int present = raw_osrs_id != 0 || item_idx != ITEM_NONE;
    int is_gear = item_idx != ITEM_NONE;
    int style = is_gear ? get_item_attack_style(item_idx) : 0;
    uint32_t effect_mask =
        is_gear ? ITEM_DATABASE[item_idx].effect_mask : OSRS_ITEM_EFFECT_NONE;
    int is_weapon = is_gear && ITEM_DATABASE[item_idx].slot == SLOT_WEAPON;
    OsrsConsumableKind6 k6 = col_consumable_kind6(consumable.consumable_kind);
    OsrsConsumableKind ck = consumable.consumable_kind;
    int hp_heal = osrs_consumable_hp_heal_amount(ck, base_hitpoints);
    int pray_restore = osrs_consumable_prayer_restore_amount(ck, base_prayer);
    int off_boost = osrs_consumable_offensive_boost_amount(ck, base_level);

    OsrsInventoryCellAffordance a;
    a.present = present ? 1.0f : 0.0f;
    a.is_equipped = is_equipped ? 1.0f : 0.0f;
    a.dose = dose > 0 ? osrs_clamp_unit((float)dose / 4.0f) : 0.0f;
    a.is_armor = (is_gear && !is_weapon) ? 1.0f : 0.0f;
    a.is_weapon = is_weapon ? 1.0f : 0.0f;
    a.style3[0] = style == 1 ? 1.0f : 0.0f;
    a.style3[1] = style == 2 ? 1.0f : 0.0f;
    a.style3[2] = style == 3 ? 1.0f : 0.0f;
    for (int i = 0; i < 6; i++)
        a.post_use_deltas[i] =
            post_use_deltas ? osrs_clamp_unit(post_use_deltas[i]) : 0.0f;
    /* Normalised energy cost, 0 when the weapon has no special. Doubles as the "this item
     * has a special attack" flag while also telling the agent whether it can afford one. */
    a.spec_cost = is_weapon
        ? osrs_clamp_unit((float)osrs_spec_cost(item_idx) / 100.0f)
        : 0.0f;
    osrs_item_effect_class4(effect_mask, a.effect_class4);
    a.attack_speed = is_weapon
        ? osrs_clamp_unit((float)ITEM_DATABASE[item_idx].attack_speed / STAT_NORM_SPEED)
        : 0.0f;
    a.attack_range = is_weapon
        ? osrs_clamp_unit((float)ITEM_DATABASE[item_idx].attack_range / STAT_NORM_RANGE)
        : 0.0f;
    a.kind5[0] = k6 == COL_CKIND6_BREW ? 1.0f : 0.0f;
    a.kind5[1] = k6 == COL_CKIND6_RESTORE ? 1.0f : 0.0f;
    a.kind5[2] = k6 == COL_CKIND6_COMBAT_BOOST ? 1.0f : 0.0f;
    a.kind5[3] = k6 == COL_CKIND6_RANGED_BOOST ? 1.0f : 0.0f;
    a.kind5[4] = k6 == COL_CKIND6_SPECIAL ? 1.0f : 0.0f;
    a.hp_heal = base_hitpoints > 0
        ? osrs_clamp_unit((float)hp_heal / (float)base_hitpoints) : 0.0f;
    a.prayer_restore = base_prayer > 0
        ? osrs_clamp_unit((float)pray_restore / (float)base_prayer) : 0.0f;
    a.offensive_boost = osrs_clamp_unit((float)off_boost / STAT_NORM_STRENGTH);
    return a;
}

static inline void osrs_write_inventory_cell_affordance_features(
    float* out,
    uint8_t item_idx,
    uint16_t raw_osrs_id,
    uint8_t dose,
    int is_equipped,
    const float post_use_deltas[6],
    int base_hitpoints,
    int base_prayer,
    int base_level
) {
    OsrsInventoryCellAffordance a = osrs_inventory_cell_affordance(
        item_idx, raw_osrs_id, dose, is_equipped, post_use_deltas,
        base_hitpoints, base_prayer, base_level);

    out[0] = a.present;
    out[1] = a.is_equipped;
    out[2] = a.dose;
    out[3] = a.style3[0];
    out[4] = a.style3[1];
    out[5] = a.style3[2];
    for (int i = 0; i < 6; i++) out[6 + i] = a.post_use_deltas[i];
    out[12] = a.is_armor;
    out[13] = a.is_weapon;
    for (int i = 0; i < 5; i++) out[14 + i] = a.kind5[i];
    out[19] = a.hp_heal;
    out[20] = a.prayer_restore;
    out[21] = a.offensive_boost;
    for (int i = 0; i < 4; i++) out[22 + i] = a.effect_class4[i];
    out[26] = a.attack_speed;
    out[27] = a.attack_range;
}

static inline void osrs_write_inventory_cell_affordance_features_compact(
    float* out,
    uint8_t item_idx,
    uint16_t raw_osrs_id,
    uint8_t dose,
    int is_equipped,
    int base_hitpoints,
    int base_prayer,
    int base_level
) {
    OsrsInventoryCellAffordance a = osrs_inventory_cell_affordance(
        item_idx, raw_osrs_id, dose, is_equipped, NULL,
        base_hitpoints, base_prayer, base_level);

    out[OSRS_INVENTORY_CELL_COMPACT_PRESENT] = a.present;
    out[OSRS_INVENTORY_CELL_COMPACT_EQUIPPED] = a.is_equipped;
    out[OSRS_INVENTORY_CELL_COMPACT_DOSE] = a.dose;
    out[OSRS_INVENTORY_CELL_COMPACT_IS_ARMOR] = a.is_armor;
    out[OSRS_INVENTORY_CELL_COMPACT_IS_WEAPON] = a.is_weapon;

    float* u = out + OSRS_INVENTORY_CELL_OBS_SHARED;
    for (int i = 0; i < OSRS_INVENTORY_CELL_OBS_KIND_UNION; i++) u[i] = 0.0f;
    if (a.is_armor != 0.0f || a.is_weapon != 0.0f) {
        u[0] = a.style3[0];
        u[1] = a.style3[1];
        u[2] = a.style3[2];
        for (int i = 0; i < 4; i++) u[3 + i] = a.effect_class4[i];
        u[7] = a.attack_speed;
        u[8] = a.attack_range;
        u[9] = a.spec_cost;
    } else {
        for (int i = 0; i < 5; i++) u[i] = a.kind5[i];
        u[5] = a.hp_heal;
        u[6] = a.prayer_restore;
        u[7] = a.offensive_boost;
    }
}

static inline void osrs_write_equipped_self_features(float* out, uint8_t item_idx) {
    for (int i = 0; i < OSRS_EQUIPPED_SELF_OBS_FEATURES; i++) out[i] = 0.0f;
    if (item_idx == ITEM_NONE) return;
    if (item_idx >= NUM_ITEMS) osrs_inventory_clicks_trap();

    const Item* item = &ITEM_DATABASE[item_idx];
    int style = get_item_attack_style(item_idx);
    out[0] = 1.0f;
    out[1] = style == 1 ? 1.0f : 0.0f;
    out[2] = style == 2 ? 1.0f : 0.0f;
    out[3] = style == 3 ? 1.0f : 0.0f;
    out[4] = osrs_clamp_unit((float)item->attack_slash / STAT_NORM_ATTACK);
    out[5] = osrs_clamp_unit((float)item->melee_strength / STAT_NORM_STRENGTH);
    out[6] = osrs_clamp_unit((float)item->attack_ranged / STAT_NORM_ATTACK);
    out[7] = osrs_clamp_unit((float)item->ranged_strength / STAT_NORM_STRENGTH);
    out[8] = osrs_clamp_unit(((float)item->attack_magic / STAT_NORM_ATTACK) +
        ((float)item->magic_damage / STAT_NORM_MAGIC_DMG));
    out[9] = osrs_clamp_unit((float)(item->defence_stab + item->defence_slash +
        item->defence_crush + item->defence_magic + item->defence_ranged) /
        (5.0f * STAT_NORM_DEFENCE));
    out[10] = item->effect_mask != OSRS_ITEM_EFFECT_NONE ? 1.0f : 0.0f;
    out[11] = item->slot == SLOT_WEAPON ? 1.0f : 0.0f;

    float eff4[4];
    osrs_item_effect_class4(item->effect_mask, eff4);
    out[12] = eff4[0];
    out[13] = eff4[1];
    out[14] = eff4[2];
    out[15] = eff4[3];
    out[16] = item->slot == SLOT_WEAPON
        ? osrs_clamp_unit((float)item->attack_speed / STAT_NORM_SPEED) : 0.0f;
    out[17] = item->slot == SLOT_WEAPON
        ? osrs_clamp_unit((float)item->attack_range / STAT_NORM_RANGE) : 0.0f;
}

static inline OsrsClickAction osrs_item_click_action(uint8_t item_idx) {
    if (item_idx == ITEM_NONE) return OSRS_CLICK_NONE;
    if (item_idx >= NUM_ITEMS) osrs_inventory_clicks_trap();

    switch (ITEM_DATABASE[item_idx].slot) {
        case SLOT_HEAD:
        case SLOT_CAPE:
        case SLOT_NECK:
        case SLOT_WEAPON:
        case SLOT_BODY:
        case SLOT_SHIELD:
        case SLOT_LEGS:
        case SLOT_HANDS:
        case SLOT_FEET:
        case SLOT_RING:
        case SLOT_AMMO:
            return OSRS_CLICK_EQUIP;
        default:
            osrs_inventory_clicks_trap();
            return OSRS_CLICK_NONE;
    }
}

static inline OsrsConsumableKind osrs_item_click_consumable_kind(uint8_t item_idx) {
    if (item_idx == ITEM_NONE) return OSRS_CONSUMABLE_NONE;
    if (item_idx >= NUM_ITEMS) osrs_inventory_clicks_trap();
    return OSRS_CONSUMABLE_NONE;
}

static inline OsrsConsumableClick osrs_consumable_click_lookup_raw_osrs_id(
    uint16_t raw_osrs_id
) {
    int count = (int)(
        sizeof(OSRS_CONSUMABLE_CLICK_REGISTRY) /
        sizeof(OSRS_CONSUMABLE_CLICK_REGISTRY[0])
    );

    for (int i = 0; i < count; i++) {
        if (OSRS_CONSUMABLE_CLICK_REGISTRY[i].raw_osrs_id == raw_osrs_id) {
            return OSRS_CONSUMABLE_CLICK_REGISTRY[i];
        }
    }

    return (OsrsConsumableClick){
        .raw_osrs_id = raw_osrs_id,
        .click_action = OSRS_CLICK_NONE,
        .consumable_kind = OSRS_CONSUMABLE_NONE,
        .dose_count = 0,
    };
}

static inline uint8_t osrs_consumable_dose_count_after_drink(uint8_t dose_count) {
    switch (dose_count) {
        case 1: return 0;
        case 2: return 1;
        case 3: return 2;
        case 4: return 3;
        default:
            osrs_inventory_clicks_trap();
            return 0;
    }
}

static inline uint16_t osrs_consumable_raw_osrs_id_after_drink(uint16_t raw_osrs_id) {
    OsrsConsumableClick before =
        osrs_consumable_click_lookup_raw_osrs_id(raw_osrs_id);
    if (before.click_action != OSRS_CLICK_DRINK || before.dose_count == 0) {
        osrs_inventory_clicks_trap();
    }

    uint8_t after_dose =
        osrs_consumable_dose_count_after_drink(before.dose_count);
    if (after_dose == 0) return 0;

    int count = (int)(
        sizeof(OSRS_CONSUMABLE_CLICK_REGISTRY) /
        sizeof(OSRS_CONSUMABLE_CLICK_REGISTRY[0])
    );

    for (int i = 0; i < count; i++) {
        OsrsConsumableClick candidate = OSRS_CONSUMABLE_CLICK_REGISTRY[i];
        if (candidate.consumable_kind == before.consumable_kind &&
            candidate.dose_count == after_dose) {
            return candidate.raw_osrs_id;
        }
    }

    osrs_inventory_clicks_trap();
    return 0;
}

static inline OsrsInventoryClickResolution osrs_inventory_click_interpret(
    uint8_t item_idx,
    uint16_t raw_osrs_id,
    OsrsClickTickMultiplicity tick_multiplicity
) {
    OsrsInventoryClickResolution none_resolution = {
        .click_action = OSRS_CLICK_NONE,
        .consumable_kind = OSRS_CONSUMABLE_NONE,
        .dose_count = 0,
        .raw_osrs_id_after_drink = 0,
    };
    switch (tick_multiplicity) {
        case OSRS_CLICK_TICK_DUPLICATE:
            return none_resolution;
        case OSRS_CLICK_TICK_FIRST:
            break;
        default:
            osrs_inventory_clicks_trap();
    }

    if (item_idx != ITEM_NONE) {
        OsrsClickAction action = osrs_item_click_action(item_idx);
        if (raw_osrs_id != 0 && raw_osrs_id != ITEM_DATABASE[item_idx].item_id) {
            osrs_inventory_clicks_trap();
        }
        return (OsrsInventoryClickResolution){
            .click_action = action,
            .consumable_kind = OSRS_CONSUMABLE_NONE,
            .dose_count = 0,
            .raw_osrs_id_after_drink = 0,
        };
    }

    if (raw_osrs_id == 0) {
        return none_resolution;
    }

    OsrsConsumableClick consumable =
        osrs_consumable_click_lookup_raw_osrs_id(raw_osrs_id);
    uint16_t after_drink = 0;
    if (consumable.click_action == OSRS_CLICK_DRINK &&
        consumable.dose_count > 0) {
        after_drink = osrs_consumable_raw_osrs_id_after_drink(raw_osrs_id);
    }

    return (OsrsInventoryClickResolution){
        .click_action = consumable.click_action,
        .consumable_kind = consumable.consumable_kind,
        .dose_count = consumable.dose_count,
        .raw_osrs_id_after_drink = after_drink,
    };
}

static inline OsrsInventoryClickResolution osrs_inventory_cell_click_interpret(
    const OsrsInventoryCell* cell,
    OsrsClickTickMultiplicity tick_multiplicity
) {
    return osrs_inventory_click_interpret(
        cell->item_idx,
        cell->raw_osrs_id,
        tick_multiplicity);
}

static inline OsrsInventoryClickResolution osrs_inventory_snapshot_click_interpret(
    const OsrsInventorySlotSnapshot* snapshot,
    int slot,
    OsrsClickTickMultiplicity tick_multiplicity
) {
    if (!osrs_inventory_slot_valid(slot)) {
        fprintf(stderr, "inventory click: invalid slot %d\n", slot);
        abort();
    }
    return osrs_inventory_cell_click_interpret(
        &snapshot->cells[slot],
        tick_multiplicity);
}

static inline OsrsInventoryCell osrs_inventory_cell_from_raw_osrs_id(
    uint16_t raw_osrs_id
) {
    if (raw_osrs_id == 0) return osrs_inventory_cell_empty();

    uint8_t item_idx = osrs_item_index_for_raw_osrs_id(raw_osrs_id);
    if (item_idx != ITEM_NONE) return osrs_inventory_cell_from_item(item_idx);

    OsrsConsumableClick consumable =
        osrs_consumable_click_lookup_raw_osrs_id(raw_osrs_id);
    return (OsrsInventoryCell){
        .item_idx = ITEM_NONE,
        .raw_osrs_id = raw_osrs_id,
        .dose = consumable.dose_count,
    };
}

static inline void osrs_inventory_cell_decrement_drink(
    OsrsInventoryCell* cell,
    OsrsInventoryClickResolution resolution
) {
    if (resolution.click_action != OSRS_CLICK_DRINK ||
            resolution.dose_count == 0 ||
            cell->dose != resolution.dose_count) {
        fprintf(stderr, "inventory drink decrement: invalid cell raw=%u dose=%u action=%d resolved_dose=%u\n",
            cell->raw_osrs_id, cell->dose, (int)resolution.click_action,
            resolution.dose_count);
        abort();
    }

    if (resolution.raw_osrs_id_after_drink == 0) {
        *cell = osrs_inventory_cell_empty();
        return;
    }

    cell->item_idx = ITEM_NONE;
    cell->raw_osrs_id = resolution.raw_osrs_id_after_drink;
    cell->dose = osrs_consumable_dose_count_after_drink(resolution.dose_count);
}

static inline OsrsInventoryDrinkConsumeResult osrs_inventory_cell_consume_drink_one_dose(
    OsrsInventoryCell* cell,
    OsrsInventoryClickResolution resolution,
    int* potion_timer,
    OsrsInventoryDrinkOneDoseEffectFn apply_one_dose,
    void* ctx
) {
    if (cell == NULL || potion_timer == NULL || apply_one_dose == NULL) {
        fprintf(stderr, "inventory drink consume: null argument\n");
        abort();
    }
    if (resolution.click_action != OSRS_CLICK_DRINK ||
            resolution.dose_count == 0 ||
            cell->dose != resolution.dose_count) {
        fprintf(stderr, "inventory drink consume: invalid cell raw=%u dose=%u action=%d resolved_dose=%u\n",
            cell->raw_osrs_id, cell->dose, (int)resolution.click_action,
            resolution.dose_count);
        abort();
    }

    uint8_t dose_after =
        osrs_consumable_dose_count_after_drink(resolution.dose_count);
    OsrsInventoryDrinkConsumeResult result = {
        .consumed = 0,
        .consumable_kind = resolution.consumable_kind,
        .dose_count_before = resolution.dose_count,
        .dose_count_after = dose_after,
        .raw_osrs_id_before = cell->raw_osrs_id,
        .raw_osrs_id_after_drink = resolution.raw_osrs_id_after_drink,
    };
    if (*potion_timer > 0) return result;

    osrs_inventory_cell_decrement_drink(cell, resolution);
    *potion_timer = 3;
    apply_one_dose(ctx, resolution.consumable_kind);
    result.consumed = 1;
    return result;
}

static inline void osrs_inventory_cell_consume_eat(OsrsInventoryCell* cell) {
    *cell = osrs_inventory_cell_empty();
}

/* One dense code per distinguishable cell content. Gear keys on the item index; consumables
   key on their click-registry entry, because a potion's raw id is what carries its kind and
   its dose; every remaining present-but-inert raw id collapses onto a single code, since no
   derived feature can tell two of them apart.

   The code rides the observation divided by OSRS_INVENTORY_CELL_OBS_CODE_SCALE. The encoder's
   global linear layer reads the WHOLE observation, so 28 raw codes near 200 sitting beside
   894 features in [0,1] outweigh everything else by four orders of magnitude in the input
   variance -- measured, and it cost episode_return 7.8 -> 0.6 with entropy collapsing 7.3 ->
   1.9 over 30M steps. Scaling by a power of two only shifts the exponent, so a code that is
   exact in bf16 stays exact and the gather's multiply-back is lossless; that is also why the
   count must stay at or under the scale. */
#define OSRS_CONSUMABLE_CLICK_REGISTRY_COUNT \
    ((int)(sizeof(OSRS_CONSUMABLE_CLICK_REGISTRY) / \
           sizeof(OSRS_CONSUMABLE_CLICK_REGISTRY[0])))
#define OSRS_INVENTORY_CELL_OBS_CODE_EMPTY 0
#define OSRS_INVENTORY_CELL_OBS_CODE_GEAR_BASE 1
#define OSRS_INVENTORY_CELL_OBS_CODE_CONSUMABLE_BASE \
    (OSRS_INVENTORY_CELL_OBS_CODE_GEAR_BASE + NUM_ITEMS)
#define OSRS_INVENTORY_CELL_OBS_CODE_INERT \
    (OSRS_INVENTORY_CELL_OBS_CODE_CONSUMABLE_BASE + OSRS_CONSUMABLE_CLICK_REGISTRY_COUNT)
#define OSRS_INVENTORY_CELL_OBS_CODE_COUNT (OSRS_INVENTORY_CELL_OBS_CODE_INERT + 1)
#define OSRS_INVENTORY_CELL_OBS_CODE_SCALE 256

static inline float osrs_inventory_cell_obs_code_encode(int code) {
    return (float)code / (float)OSRS_INVENTORY_CELL_OBS_CODE_SCALE;
}

static inline int osrs_inventory_cell_obs_code_decode(float observed) {
    return (int)lrintf(observed * (float)OSRS_INVENTORY_CELL_OBS_CODE_SCALE);
}

/* Stands in for every present raw id that is neither gear nor a registered consumable. */
#define OSRS_INVENTORY_CELL_INERT_RAW_OSRS_ID 0xFFFFu

static inline int osrs_inventory_cell_obs_code(uint8_t item_idx, uint16_t raw_osrs_id) {
    if (item_idx != ITEM_NONE) {
        if (item_idx >= NUM_ITEMS) osrs_inventory_clicks_trap();
        return OSRS_INVENTORY_CELL_OBS_CODE_GEAR_BASE + item_idx;
    }
    if (raw_osrs_id == 0) return OSRS_INVENTORY_CELL_OBS_CODE_EMPTY;
    for (int i = 0; i < OSRS_CONSUMABLE_CLICK_REGISTRY_COUNT; i++) {
        if (OSRS_CONSUMABLE_CLICK_REGISTRY[i].raw_osrs_id == raw_osrs_id) {
            return OSRS_INVENTORY_CELL_OBS_CODE_CONSUMABLE_BASE + i;
        }
    }
    return OSRS_INVENTORY_CELL_OBS_CODE_INERT;
}

static inline OsrsInventoryCell osrs_inventory_cell_for_obs_code(int code) {
    if (code < 0 || code >= OSRS_INVENTORY_CELL_OBS_CODE_COUNT) {
        osrs_inventory_clicks_trap();
    }
    if (code == OSRS_INVENTORY_CELL_OBS_CODE_EMPTY) return osrs_inventory_cell_empty();
    if (code < OSRS_INVENTORY_CELL_OBS_CODE_CONSUMABLE_BASE) {
        return osrs_inventory_cell_from_item(
            (uint8_t)(code - OSRS_INVENTORY_CELL_OBS_CODE_GEAR_BASE));
    }
    if (code < OSRS_INVENTORY_CELL_OBS_CODE_INERT) {
        return osrs_inventory_cell_from_raw_osrs_id(
            OSRS_CONSUMABLE_CLICK_REGISTRY[
                code - OSRS_INVENTORY_CELL_OBS_CODE_CONSUMABLE_BASE].raw_osrs_id);
    }
    return (OsrsInventoryCell){
        .item_idx = ITEM_NONE,
        .raw_osrs_id = OSRS_INVENTORY_CELL_INERT_RAW_OSRS_ID,
        .dose = 0,
    };
}

/* The item-table row: the compact record with the two slots the observation supplies zeroed,
   so a gather can add the observed values in without branching on the cell's kind. On a gear
   cell the union slot holds effect_class4[2], which is an item fact and stays. */
static inline void osrs_write_inventory_cell_obs_table_row(
    float* out,
    int code,
    int base_hitpoints,
    int base_prayer,
    int base_level
) {
    OsrsInventoryCell cell = osrs_inventory_cell_for_obs_code(code);
    osrs_write_inventory_cell_affordance_features_compact(
        out, cell.item_idx, cell.raw_osrs_id, cell.dose, 0,
        base_hitpoints, base_prayer, base_level);

    out[OSRS_INVENTORY_CELL_COMPACT_EQUIPPED] = 0.0f;
    if (out[OSRS_INVENTORY_CELL_COMPACT_IS_ARMOR] == 0.0f &&
            out[OSRS_INVENTORY_CELL_COMPACT_IS_WEAPON] == 0.0f) {
        out[OSRS_INVENTORY_CELL_COMPACT_HP_HEAL] = 0.0f;
    }
}

/* The other half of the same record. Projecting both sides off one compact write is what
   makes them complementary: a change to the affordance reaches the table and the
   observation together instead of drifting between them. */
static inline void osrs_write_inventory_cell_obs_code_features(
    float* out,
    uint8_t item_idx,
    uint16_t raw_osrs_id,
    uint8_t dose,
    int is_equipped,
    int base_hitpoints,
    int base_prayer,
    int base_level
) {
    float compact[OSRS_INVENTORY_CELL_OBS_FEATURES_COMPACT];
    osrs_write_inventory_cell_affordance_features_compact(
        compact, item_idx, raw_osrs_id, dose, is_equipped,
        base_hitpoints, base_prayer, base_level);

    int is_gear = compact[OSRS_INVENTORY_CELL_COMPACT_IS_ARMOR] != 0.0f ||
        compact[OSRS_INVENTORY_CELL_COMPACT_IS_WEAPON] != 0.0f;

    out[OSRS_INVENTORY_CELL_OBS_CODE] = osrs_inventory_cell_obs_code_encode(
        osrs_inventory_cell_obs_code(item_idx, raw_osrs_id));
    out[OSRS_INVENTORY_CELL_OBS_EQUIPPED] =
        compact[OSRS_INVENTORY_CELL_COMPACT_EQUIPPED];
    out[OSRS_INVENTORY_CELL_OBS_HP_HEAL] =
        is_gear ? 0.0f : compact[OSRS_INVENTORY_CELL_COMPACT_HP_HEAL];
}

/* Inverse of the split above, and the exact arithmetic colo_ent_gather_inv performs. */
static inline void osrs_expand_inventory_cell_obs_code_features(
    float* out,
    const float* coded,
    const float* table_row
) {
    for (int f = 0; f < OSRS_INVENTORY_CELL_OBS_FEATURES_COMPACT; f++) {
        out[f] = table_row[f];
    }
    out[OSRS_INVENTORY_CELL_COMPACT_EQUIPPED] +=
        coded[OSRS_INVENTORY_CELL_OBS_EQUIPPED];
    out[OSRS_INVENTORY_CELL_COMPACT_HP_HEAL] +=
        coded[OSRS_INVENTORY_CELL_OBS_HP_HEAL];
}

#endif
