#ifndef OSRS_INVENTORY_CLICKS_H
#define OSRS_INVENTORY_CLICKS_H

#include "osrs_inventory.h"
#include "osrs_items.h"
#include "osrs_consumables.h"

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
        case OSRS_CONSUMABLE_SANFEW:          return COL_CKIND6_RESTORE;
        case OSRS_CONSUMABLE_SUPER_COMBAT:
        case OSRS_CONSUMABLE_DIVINE_COMBAT:   return COL_CKIND6_COMBAT_BOOST;
        case OSRS_CONSUMABLE_RANGING:
        case OSRS_CONSUMABLE_DIVINE_RANGING:  return COL_CKIND6_RANGED_BOOST;
        case OSRS_CONSUMABLE_SURGE:
        case OSRS_CONSUMABLE_GUTHIX_REST:
        case OSRS_CONSUMABLE_SATURATED_HEART:
        case OSRS_CONSUMABLE_ANTIVENOM_PLUS:  return COL_CKIND6_SPECIAL;
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
        case OSRS_CONSUMABLE_SANFEW:        return osrs_sanfew_restore_amount(base_prayer);
        default:                            return 0;
    }
}

static inline int osrs_consumable_offensive_boost_amount(OsrsConsumableKind k, int base_level) {
    switch (k) {
        case OSRS_CONSUMABLE_SUPER_COMBAT:
        case OSRS_CONSUMABLE_DIVINE_COMBAT:   return osrs_super_combat_boost_amount(base_level);
        case OSRS_CONSUMABLE_RANGING:
        case OSRS_CONSUMABLE_DIVINE_RANGING:  return osrs_ranging_boost_amount(base_level);
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
    OsrsConsumableClick consumable =
        osrs_consumable_click_lookup_raw_osrs_id(raw_osrs_id);
    int present = raw_osrs_id != 0 || item_idx != ITEM_NONE;
    int is_gear = item_idx != ITEM_NONE;
    int style = is_gear ? get_item_attack_style(item_idx) : 0;
    uint32_t effect_mask = is_gear ? ITEM_DATABASE[item_idx].effect_mask : OSRS_ITEM_EFFECT_NONE;

    out[0] = present ? 1.0f : 0.0f;
    out[1] = is_equipped ? 1.0f : 0.0f;
    out[2] = dose > 0 ? osrs_clamp_unit((float)dose / 4.0f) : 0.0f;
    out[3] = style == 1 ? 1.0f : 0.0f;
    out[4] = style == 2 ? 1.0f : 0.0f;
    out[5] = style == 3 ? 1.0f : 0.0f;
    for (int i = 0; i < 6; i++) out[6 + i] = osrs_clamp_unit(post_use_deltas[i]);

    int is_weapon = is_gear && ITEM_DATABASE[item_idx].slot == SLOT_WEAPON;
    out[12] = (is_gear && !is_weapon) ? 1.0f : 0.0f;
    out[13] = is_weapon ? 1.0f : 0.0f;
    OsrsConsumableKind6 k6 = col_consumable_kind6(consumable.consumable_kind);
    out[14] = k6 == COL_CKIND6_BREW ? 1.0f : 0.0f;
    out[15] = k6 == COL_CKIND6_RESTORE ? 1.0f : 0.0f;
    out[16] = k6 == COL_CKIND6_COMBAT_BOOST ? 1.0f : 0.0f;
    out[17] = k6 == COL_CKIND6_RANGED_BOOST ? 1.0f : 0.0f;
    out[18] = k6 == COL_CKIND6_SPECIAL ? 1.0f : 0.0f;
    OsrsConsumableKind ck = consumable.consumable_kind;
    int hp_heal = osrs_consumable_hp_heal_amount(ck, base_hitpoints);
    int pray_restore = osrs_consumable_prayer_restore_amount(ck, base_prayer);
    int off_boost = osrs_consumable_offensive_boost_amount(ck, base_level);
    out[19] = base_hitpoints > 0 ? osrs_clamp_unit((float)hp_heal / (float)base_hitpoints) : 0.0f;
    out[20] = base_prayer > 0 ? osrs_clamp_unit((float)pray_restore / (float)base_prayer) : 0.0f;
    out[21] = osrs_clamp_unit((float)off_boost / STAT_NORM_STRENGTH);
    float eff4[4];
    osrs_item_effect_class4(effect_mask, eff4);
    out[22] = eff4[0];
    out[23] = eff4[1];
    out[24] = eff4[2];
    out[25] = eff4[3];
    out[26] = is_weapon
        ? osrs_clamp_unit((float)ITEM_DATABASE[item_idx].attack_speed / STAT_NORM_SPEED) : 0.0f;
    out[27] = is_weapon
        ? osrs_clamp_unit((float)ITEM_DATABASE[item_idx].attack_range / STAT_NORM_RANGE) : 0.0f;
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

#endif
