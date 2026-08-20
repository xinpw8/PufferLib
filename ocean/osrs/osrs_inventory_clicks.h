#ifndef OSRS_INVENTORY_CLICKS_H
#define OSRS_INVENTORY_CLICKS_H

#include "osrs_item_effects.h"
#include "osrs_inventory.h"
#include "osrs_items.h"
#include "osrs_consumables.h"
#include "osrs_special_attacks.h"


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
#define OSRS_INVENTORY_CELL_OBS_SHARED 4
#define OSRS_INVENTORY_CELL_OBS_KIND_UNION 10
#define OSRS_INVENTORY_CELL_OBS_FEATURES_COMPACT \
    (OSRS_INVENTORY_CELL_OBS_SHARED + OSRS_INVENTORY_CELL_OBS_KIND_UNION)

#define OSRS_INVENTORY_CELL_COMPACT_PRESENT   0
#define OSRS_INVENTORY_CELL_COMPACT_DOSE      1
#define OSRS_INVENTORY_CELL_COMPACT_IS_ARMOR  2
#define OSRS_INVENTORY_CELL_COMPACT_IS_WEAPON 3
/* Gear effect, consumable healing. */
#define OSRS_INVENTORY_CELL_COMPACT_HP_HEAL   (OSRS_INVENTORY_CELL_OBS_SHARED + 5)

#define OSRS_INVENTORY_CELL_OBS_CODE 0

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
        case OSRS_CONSUMABLE_COUNT:           break;
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

static inline OsrsInventoryCellAffordance osrs_item_content_affordance(
    const OsrsItemContentMetadata* metadata,
    int is_equipped,
    const float post_use_deltas[6],
    int base_hitpoints,
    int base_prayer,
    int base_level
) {
    int present = metadata->raw_osrs_id != 0;
    int is_gear = metadata->item != NULL;
    int is_weapon = is_gear && metadata->item->slot == SLOT_WEAPON;
    OsrsConsumableKind consumable_kind =
        (OsrsConsumableKind)metadata->consumable_kind;
    OsrsConsumableKind6 kind6 = col_consumable_kind6(consumable_kind);
    int hp_heal =
        osrs_consumable_hp_heal_amount(consumable_kind, base_hitpoints);
    int prayer_restore =
        osrs_consumable_prayer_restore_amount(consumable_kind, base_prayer);
    int offensive_boost =
        osrs_consumable_offensive_boost_amount(consumable_kind, base_level);
    uint32_t effect_mask =
        is_gear ? metadata->item->effect_mask : OSRS_ITEM_EFFECT_NONE;

    OsrsInventoryCellAffordance affordance;
    affordance.present = present ? 1.0f : 0.0f;
    affordance.is_equipped = is_equipped ? 1.0f : 0.0f;
    affordance.dose = metadata->dose_count > 0
        ? osrs_clamp_unit((float)metadata->dose_count / 4.0f)
        : 0.0f;
    affordance.is_armor = (is_gear && !is_weapon) ? 1.0f : 0.0f;
    affordance.is_weapon = is_weapon ? 1.0f : 0.0f;
    affordance.style3[0] = metadata->attack_style == 1 ? 1.0f : 0.0f;
    affordance.style3[1] = metadata->attack_style == 2 ? 1.0f : 0.0f;
    affordance.style3[2] = metadata->attack_style == 3 ? 1.0f : 0.0f;
    for (int index = 0; index < 6; index++) {
        affordance.post_use_deltas[index] = post_use_deltas
            ? osrs_clamp_unit(post_use_deltas[index])
            : 0.0f;
    }
    affordance.spec_cost = is_weapon
        ? osrs_clamp_unit(
            (float)osrs_spec_cost(metadata->item_idx) / 100.0f)
        : 0.0f;
    osrs_item_effect_class4(effect_mask, affordance.effect_class4);
    affordance.attack_speed = is_weapon
        ? osrs_clamp_unit(
            (float)metadata->item->attack_speed / STAT_NORM_SPEED)
        : 0.0f;
    affordance.attack_range = is_weapon
        ? osrs_clamp_unit(
            (float)metadata->item->attack_range / STAT_NORM_RANGE)
        : 0.0f;
    affordance.kind5[0] = kind6 == COL_CKIND6_BREW ? 1.0f : 0.0f;
    affordance.kind5[1] = kind6 == COL_CKIND6_RESTORE ? 1.0f : 0.0f;
    affordance.kind5[2] = kind6 == COL_CKIND6_COMBAT_BOOST ? 1.0f : 0.0f;
    affordance.kind5[3] = kind6 == COL_CKIND6_RANGED_BOOST ? 1.0f : 0.0f;
    affordance.kind5[4] = kind6 == COL_CKIND6_SPECIAL ? 1.0f : 0.0f;
    affordance.hp_heal = base_hitpoints > 0
        ? osrs_clamp_unit((float)hp_heal / (float)base_hitpoints)
        : 0.0f;
    affordance.prayer_restore = base_prayer > 0
        ? osrs_clamp_unit((float)prayer_restore / (float)base_prayer)
        : 0.0f;
    affordance.offensive_boost =
        osrs_clamp_unit((float)offensive_boost / STAT_NORM_STRENGTH);
    return affordance;
}

static inline void osrs_write_inventory_cell_affordance_features(
    float* out,
    const OsrsInventoryCell* cell,
    int is_equipped,
    const float post_use_deltas[6],
    int base_hitpoints,
    int base_prayer,
    int base_level
) {
    OsrsInventoryCellAffordance affordance = osrs_item_content_affordance(
        osrs_inventory_cell_metadata(cell), is_equipped, post_use_deltas,
        base_hitpoints, base_prayer, base_level);

    out[0] = affordance.present;
    out[1] = affordance.is_equipped;
    out[2] = affordance.dose;
    out[3] = affordance.style3[0];
    out[4] = affordance.style3[1];
    out[5] = affordance.style3[2];
    for (int index = 0; index < 6; index++) {
        out[6 + index] = affordance.post_use_deltas[index];
    }
    out[12] = affordance.is_armor;
    out[13] = affordance.is_weapon;
    for (int index = 0; index < 5; index++) {
        out[14 + index] = affordance.kind5[index];
    }
    out[19] = affordance.hp_heal;
    out[20] = affordance.prayer_restore;
    out[21] = affordance.offensive_boost;
    for (int index = 0; index < 4; index++) {
        out[22 + index] = affordance.effect_class4[index];
    }
    out[26] = affordance.attack_speed;
    out[27] = affordance.attack_range;
}

static inline void osrs_write_item_content_affordance_features_compact(
    float* out,
    const OsrsItemContentMetadata* metadata,
    int is_equipped,
    int base_hitpoints,
    int base_prayer,
    int base_level
) {
    OsrsInventoryCellAffordance affordance = osrs_item_content_affordance(
        metadata, is_equipped, NULL,
        base_hitpoints, base_prayer, base_level);

    out[OSRS_INVENTORY_CELL_COMPACT_PRESENT] = affordance.present;
    out[OSRS_INVENTORY_CELL_COMPACT_DOSE] = affordance.dose;
    out[OSRS_INVENTORY_CELL_COMPACT_IS_ARMOR] = affordance.is_armor;
    out[OSRS_INVENTORY_CELL_COMPACT_IS_WEAPON] = affordance.is_weapon;

    float* content = out + OSRS_INVENTORY_CELL_OBS_SHARED;
    for (int index = 0; index < OSRS_INVENTORY_CELL_OBS_KIND_UNION; index++) {
        content[index] = 0.0f;
    }
    if (affordance.is_armor != 0.0f || affordance.is_weapon != 0.0f) {
        content[0] = affordance.style3[0];
        content[1] = affordance.style3[1];
        content[2] = affordance.style3[2];
        for (int index = 0; index < 4; index++) {
            content[3 + index] = affordance.effect_class4[index];
        }
        content[7] = affordance.attack_speed;
        content[8] = affordance.attack_range;
        content[9] = affordance.spec_cost;
    } else {
        for (int index = 0; index < 5; index++) {
            content[index] = affordance.kind5[index];
        }
        content[5] = affordance.hp_heal;
        content[6] = affordance.prayer_restore;
        content[7] = affordance.offensive_boost;
    }
}

static inline void osrs_write_inventory_cell_affordance_features_compact(
    float* out,
    const OsrsInventoryCell* cell,
    int is_equipped,
    int base_hitpoints,
    int base_prayer,
    int base_level
) {
    osrs_write_item_content_affordance_features_compact(
        out, osrs_inventory_cell_metadata(cell), is_equipped,
        base_hitpoints, base_prayer, base_level);
}


static inline OsrsClickAction osrs_item_click_action(uint8_t item_idx) {
    if (item_idx == ITEM_NONE) return OSRS_CLICK_NONE;
    const OsrsItemContentMetadata* metadata = osrs_item_content_metadata(
        osrs_inventory_content_code_from_item(item_idx));
    return (OsrsClickAction)metadata->click_action;
}

static inline OsrsConsumableKind osrs_item_click_consumable_kind(uint8_t item_idx) {
    if (item_idx == ITEM_NONE) return OSRS_CONSUMABLE_NONE;
    const OsrsItemContentMetadata* metadata = osrs_item_content_metadata(
        osrs_inventory_content_code_from_item(item_idx));
    return (OsrsConsumableKind)metadata->consumable_kind;
}

static inline OsrsConsumableClick osrs_consumable_click_lookup_raw_osrs_id(
    uint16_t raw_osrs_id
) {
    const OsrsItemContentMetadata* metadata = osrs_item_content_metadata(
        osrs_inventory_content_code_from_raw_osrs_id(raw_osrs_id));
    return (OsrsConsumableClick){
        .raw_osrs_id = metadata->raw_osrs_id,
        .click_action = (OsrsClickAction)metadata->click_action,
        .consumable_kind = (OsrsConsumableKind)metadata->consumable_kind,
        .dose_count = metadata->dose_count,
    };
}

static inline uint8_t osrs_consumable_dose_count_after_drink(uint8_t dose_count) {
    if (dose_count < 1 || dose_count > 4) osrs_inventory_clicks_trap();
    return (uint8_t)(dose_count - 1);
}

static inline uint16_t osrs_consumable_raw_osrs_id_after_drink(
    uint16_t raw_osrs_id
) {
    const OsrsItemContentMetadata* metadata = osrs_item_content_metadata(
        osrs_inventory_content_code_from_raw_osrs_id(raw_osrs_id));
    if (metadata->click_action != OSRS_CLICK_DRINK ||
            metadata->dose_count == 0) {
        osrs_inventory_clicks_trap();
    }
    if (metadata->next_content_code == 0) return 0;
    return osrs_item_content_metadata(
        metadata->next_content_code)->raw_osrs_id;
}

static inline OsrsInventoryClickResolution osrs_inventory_cell_click_classify(
    const OsrsInventoryCell* cell
) {
    const OsrsItemContentMetadata* metadata =
        osrs_inventory_cell_metadata(cell);
    return (OsrsInventoryClickResolution){
        .click_action = (OsrsClickAction)metadata->click_action,
        .consumable_kind = (OsrsConsumableKind)metadata->consumable_kind,
        .dose_count = metadata->dose_count,
    };
}

static inline OsrsInventoryClickResolution osrs_inventory_cell_click_interpret(
    const OsrsInventoryCell* cell,
    OsrsClickTickMultiplicity tick_multiplicity
) {
    switch (tick_multiplicity) {
        case OSRS_CLICK_TICK_DUPLICATE:
            return (OsrsInventoryClickResolution){
                .click_action = OSRS_CLICK_NONE,
            };
        case OSRS_CLICK_TICK_FIRST:
            break;
        default:
            osrs_inventory_clicks_trap();
    }

    const OsrsItemContentMetadata* metadata =
        osrs_inventory_cell_metadata(cell);
    OsrsInventoryClickResolution resolution =
        osrs_inventory_cell_click_classify(cell);
    if (resolution.click_action == OSRS_CLICK_DRINK) {
        resolution.raw_osrs_id_after_drink =
            metadata->next_content_code == 0
                ? 0
                : osrs_item_content_metadata(
                    metadata->next_content_code)->raw_osrs_id;
    }
    return resolution;
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
        &snapshot->cells[slot], tick_multiplicity);
}

static inline void osrs_inventory_cell_decrement_drink(
    OsrsInventoryCell* cell,
    OsrsInventoryClickResolution resolution
) {
    const OsrsItemContentMetadata* metadata =
        osrs_inventory_cell_metadata(cell);
    if (resolution.click_action != OSRS_CLICK_DRINK ||
            resolution.dose_count == 0 ||
            metadata->dose_count != resolution.dose_count ||
            metadata->next_content_code == cell->content_code) {
        fprintf(stderr,
            "inventory drink decrement: invalid content=%u dose=%u action=%d resolved_dose=%u\n",
            cell->content_code, metadata->dose_count,
            (int)resolution.click_action, resolution.dose_count);
        abort();
    }
    *cell = osrs_inventory_cell_from_content_code(
        metadata->next_content_code);
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
    const OsrsItemContentMetadata* metadata =
        osrs_inventory_cell_metadata(cell);
    if (resolution.click_action != OSRS_CLICK_DRINK ||
            resolution.dose_count == 0 ||
            metadata->dose_count != resolution.dose_count) {
        fprintf(stderr,
            "inventory drink consume: invalid content=%u dose=%u action=%d resolved_dose=%u\n",
            cell->content_code, metadata->dose_count,
            (int)resolution.click_action, resolution.dose_count);
        abort();
    }

    uint8_t dose_after =
        osrs_consumable_dose_count_after_drink(resolution.dose_count);
    OsrsInventoryDrinkConsumeResult result = {
        .consumed = 0,
        .consumable_kind = resolution.consumable_kind,
        .dose_count_before = resolution.dose_count,
        .dose_count_after = dose_after,
        .raw_osrs_id_before = metadata->raw_osrs_id,
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

static inline float osrs_inventory_cell_obs_code_encode(uint16_t content_code) {
    (void)osrs_item_content_metadata(content_code);
    return (float)content_code / (float)OSRS_ITEM_OBS_CODE_SCALE;
}

static inline uint16_t osrs_inventory_cell_obs_code_decode(float observed) {
    int content_code =
        (int)lrintf(observed * (float)OSRS_ITEM_OBS_CODE_SCALE);
    if (content_code < 0 || content_code >= OSRS_ITEM_CONTENT_COUNT) {
        fprintf(stderr, "inventory observation: invalid content code %d\n",
            content_code);
        abort();
    }
    return (uint16_t)content_code;
}

static inline void osrs_write_inventory_cell_obs_code(
    float* out,
    const OsrsInventoryCell* cell
) {
    out[OSRS_INVENTORY_CELL_OBS_CODE] =
        osrs_inventory_cell_obs_code_encode(cell->content_code);
}

#endif
