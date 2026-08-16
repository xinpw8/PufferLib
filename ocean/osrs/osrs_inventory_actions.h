#ifndef OSRS_INVENTORY_ACTIONS_H
#define OSRS_INVENTORY_ACTIONS_H

#include "osrs_types.h"
#include "osrs_items.h"
#include "osrs_item_effects.h"
#include "osrs_inventory_clicks.h"

static inline int osrs_first_empty_inventory_cell(
    const OsrsInventoryCell* cells,
    int except_cell
) {
    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++) {
        if (i == except_cell) continue;
        if (osrs_inventory_cell_is_empty(&cells[i])) return i;
    }
    return -1;
}

static inline int osrs_can_equip_metadata(
    const Player* p,
    const OsrsItemContentMetadata* metadata,
    int inventory_has_empty_cell
) {
    uint8_t item_idx = metadata->item_idx;
    int gear_slot = metadata->gear_slot;
    if (item_idx == ITEM_NONE || gear_slot < 0) return 0;
    if (p->equipped[gear_slot] == item_idx) return 0;
    if (gear_slot == GEAR_SLOT_WEAPON && item_is_two_handed(item_idx) &&
            p->equipped[GEAR_SLOT_SHIELD] != ITEM_NONE &&
            p->equipped[GEAR_SLOT_WEAPON] != ITEM_NONE &&
            !inventory_has_empty_cell) return 0;
    if (gear_slot == GEAR_SLOT_SHIELD &&
            item_is_two_handed(p->equipped[GEAR_SLOT_WEAPON]) &&
            !inventory_has_empty_cell) return 0;
    return 1;
}

static inline int osrs_can_equip_from_cell(
    const Player* p,
    const OsrsInventoryCell* cells,
    int cell_idx
) {
    const OsrsItemContentMetadata* metadata =
        osrs_inventory_cell_metadata(&cells[cell_idx]);
    uint8_t item_idx = metadata->item_idx;
    if (item_idx == ITEM_NONE || metadata->gear_slot < 0) return 0;

    int gear_slot = metadata->gear_slot;
    if (p->equipped[gear_slot] == item_idx) return 0;
    if (gear_slot == GEAR_SLOT_WEAPON && item_is_two_handed(item_idx) &&
            p->equipped[GEAR_SLOT_SHIELD] != ITEM_NONE &&
            p->equipped[GEAR_SLOT_WEAPON] != ITEM_NONE &&
            osrs_first_empty_inventory_cell(cells, cell_idx) < 0) return 0;
    if (gear_slot == GEAR_SLOT_SHIELD &&
            item_is_two_handed(p->equipped[GEAR_SLOT_WEAPON]) &&
            osrs_first_empty_inventory_cell(cells, cell_idx) < 0) return 0;
    return 1;
}

/** Equips the cell's item, displacing worn gear (2h weapon/shield rules
 *  included) back into the inventory. Returns the gear slot equipped into,
 *  or -1 if the click was not equippable. */
static inline int osrs_equip_from_cell(
    Player* p,
    OsrsInventoryCell* cells,
    int cell_idx
) {
    if (!osrs_can_equip_from_cell(p, cells, cell_idx)) return -1;

    OsrsInventoryCell* cell = &cells[cell_idx];
    const OsrsItemContentMetadata* metadata =
        osrs_inventory_cell_metadata(cell);
    uint8_t item_idx = metadata->item_idx;
    int gear_slot = metadata->gear_slot;
    if (gear_slot < 0) {
        fprintf(stderr, "inventory equip: item %u has no gear slot\n", item_idx);
        abort();
    }
    uint8_t displaced = p->equipped[gear_slot];

    if (gear_slot == GEAR_SLOT_WEAPON && item_is_two_handed(item_idx)) {
        uint8_t shield = p->equipped[GEAR_SLOT_SHIELD];
        if (shield != ITEM_NONE) {
            if (displaced == ITEM_NONE) {
                displaced = shield;
            } else {
                int empty = osrs_first_empty_inventory_cell(cells, cell_idx);
                if (empty < 0) return -1;
                cells[empty] = osrs_inventory_cell_from_item(shield);
            }
            p->equipped[GEAR_SLOT_SHIELD] = ITEM_NONE;
        }
    } else if (gear_slot == GEAR_SLOT_SHIELD &&
            item_is_two_handed(p->equipped[GEAR_SLOT_WEAPON])) {
        displaced = p->equipped[GEAR_SLOT_WEAPON];
        p->equipped[GEAR_SLOT_WEAPON] = ITEM_NONE;
        p->spec_armed = 0;
    }

    p->equipped[gear_slot] = item_idx;
    *cell = osrs_inventory_cell_from_item(displaced);
    if (gear_slot == GEAR_SLOT_WEAPON) p->spec_armed = 0;
    osrs_refresh_player_equipment(p);
    return gear_slot;
}

static inline int osrs_can_eat_consumable_kind(
    const Player* p,
    OsrsConsumableKind kind
) {
    switch (kind) {
        case OSRS_CONSUMABLE_SHARK_FOOD:
            return p->food_timer == 0 &&
                p->current_hitpoints < p->base_hitpoints;
        case OSRS_CONSUMABLE_KARAMBWAN:
            return p->karambwan_timer == 0 &&
                p->current_hitpoints < p->base_hitpoints;
        case OSRS_CONSUMABLE_BREW:
        case OSRS_CONSUMABLE_SUPER_RESTORE:
        case OSRS_CONSUMABLE_SANFEW:
        case OSRS_CONSUMABLE_SUPER_COMBAT:
        case OSRS_CONSUMABLE_DIVINE_COMBAT:
        case OSRS_CONSUMABLE_RANGING:
        case OSRS_CONSUMABLE_DIVINE_RANGING:
        case OSRS_CONSUMABLE_SURGE:
        case OSRS_CONSUMABLE_GUTHIX_REST:
        case OSRS_CONSUMABLE_SATURATED_HEART:
        case OSRS_CONSUMABLE_ANTIVENOM_PLUS:
        case OSRS_CONSUMABLE_PRAYER_RESTORE:
        case OSRS_CONSUMABLE_BASTION:
        case OSRS_CONSUMABLE_STAMINA:
        case OSRS_CONSUMABLE_NONE:
            return 0;
        case OSRS_CONSUMABLE_COUNT:
            break;
    }
    abort();
}

/** One tick's worth of inventory-click actions: per-gear-slot equip heads
 *  plus eat and drink heads, each 0 = no-op or 1..28 = inventory cell + 1. */
typedef struct {
    int equip_by_slot[NUM_GEAR_SLOTS];
    int eat;
    int drink;
} OsrsInventoryClickActions;

static_assert(NUM_GEAR_SLOTS <= 32, "inventory intent gear mask too narrow");

typedef struct {
    uint32_t equip_slot_mask;
    int equip_cell_by_slot[NUM_GEAR_SLOTS];
    int eat_cell;
    OsrsInventoryClickResolution eat_resolution;
    int drink_cell;
    OsrsInventoryClickResolution drink_resolution;
} OsrsInventoryTickIntent;

static inline OsrsInventoryTickIntent osrs_inventory_tick_intent_empty(void) {
    OsrsInventoryTickIntent intent = {
        .eat_cell = -1,
        .drink_cell = -1,
    };
    return intent;
}

static inline OsrsInventoryTickIntent osrs_resolve_inventory_tick_intent(
    const Player* p,
    const OsrsInventoryCell* cells,
    const OsrsInventoryClickActions* clicks
) {
    OsrsInventoryTickIntent intent = osrs_inventory_tick_intent_empty();
    int inventory_has_empty_cell = -1;

    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        int action = clicks->equip_by_slot[slot];
        if (action <= 0 || action > OSRS_INVENTORY_SIZE) continue;
        int cell_idx = action - 1;
        const OsrsItemContentMetadata* metadata =
            osrs_inventory_cell_metadata(&cells[cell_idx]);
        int needs_empty_cell =
            (metadata->gear_slot == GEAR_SLOT_WEAPON &&
                item_is_two_handed(metadata->item_idx) &&
                p->equipped[GEAR_SLOT_SHIELD] != ITEM_NONE &&
                p->equipped[GEAR_SLOT_WEAPON] != ITEM_NONE) ||
            (metadata->gear_slot == GEAR_SLOT_SHIELD &&
                item_is_two_handed(p->equipped[GEAR_SLOT_WEAPON]));
        if (needs_empty_cell && inventory_has_empty_cell < 0) {
            inventory_has_empty_cell =
                osrs_first_empty_inventory_cell(cells, -1) >= 0;
        }
        if (!osrs_can_equip_metadata(
                p, metadata,
                !needs_empty_cell || inventory_has_empty_cell)) continue;
        int gear_slot = metadata->gear_slot;
        if (gear_slot < 0 || gear_slot >= NUM_GEAR_SLOTS) {
            fprintf(stderr, "inventory equip intent: invalid gear slot %d\n", gear_slot);
            abort();
        }
        if (gear_slot != slot) continue;
        intent.equip_cell_by_slot[slot] = cell_idx;
        intent.equip_slot_mask |= UINT32_C(1) << slot;
    }

    if (clicks->eat > 0 && clicks->eat <= OSRS_INVENTORY_SIZE) {
        int cell_idx = clicks->eat - 1;
        OsrsInventoryClickResolution resolution =
            osrs_inventory_cell_click_interpret(&cells[cell_idx], OSRS_CLICK_TICK_FIRST);
        if (resolution.click_action == OSRS_CLICK_EAT &&
                osrs_can_eat_consumable_kind(p, resolution.consumable_kind)) {
            intent.eat_cell = cell_idx;
            intent.eat_resolution = resolution;
        }
    }

    if (clicks->drink > 0 && clicks->drink <= OSRS_INVENTORY_SIZE) {
        int cell_idx = clicks->drink - 1;
        OsrsInventoryClickResolution resolution =
            osrs_inventory_cell_click_interpret(&cells[cell_idx], OSRS_CLICK_TICK_FIRST);
        if (resolution.click_action == OSRS_CLICK_DRINK &&
                osrs_inventory_cell_dose_count(&cells[cell_idx]) > 0 &&
                p->potion_timer == 0) {
            intent.drink_cell = cell_idx;
            intent.drink_resolution = resolution;
        }
    }

    return intent;
}

static inline int osrs_inventory_tick_intent_has_effect(
    const OsrsInventoryTickIntent* intent
) {
    return intent->equip_slot_mask != 0 ||
        intent->eat_cell >= 0 ||
        intent->drink_cell >= 0;
}

typedef enum {
    OSRS_INVENTORY_APPLY_EQUIP = 0,
    OSRS_INVENTORY_APPLY_EAT = 1,
    OSRS_INVENTORY_APPLY_DRINK = 2,
} OsrsInventoryApplyKind;

typedef struct {
    OsrsInventoryApplyKind kind;
    int gear_slot;
    int cell_idx;
    OsrsInventoryClickResolution resolution;
} OsrsInventoryApplyStep;

/** Pops the intent's next click in OSRS application order (equip slots
 *  ascending, then eat, then drink). Returns 0 when the intent is drained. */
static inline int osrs_inventory_intent_next(
    OsrsInventoryTickIntent* intent,
    OsrsInventoryApplyStep* out
) {
    if (intent->equip_slot_mask != 0) {
        int slot = __builtin_ctz(intent->equip_slot_mask);
        intent->equip_slot_mask &= intent->equip_slot_mask - 1;
        out->kind = OSRS_INVENTORY_APPLY_EQUIP;
        out->gear_slot = slot;
        out->cell_idx = intent->equip_cell_by_slot[slot];
        memset(&out->resolution, 0, sizeof(out->resolution));
        return 1;
    }
    if (intent->eat_cell >= 0) {
        out->kind = OSRS_INVENTORY_APPLY_EAT;
        out->gear_slot = -1;
        out->cell_idx = intent->eat_cell;
        out->resolution = intent->eat_resolution;
        intent->eat_cell = -1;
        return 1;
    }
    if (intent->drink_cell >= 0) {
        out->kind = OSRS_INVENTORY_APPLY_DRINK;
        out->gear_slot = -1;
        out->cell_idx = intent->drink_cell;
        out->resolution = intent->drink_resolution;
        intent->drink_cell = -1;
        return 1;
    }
    return 0;
}

#endif
