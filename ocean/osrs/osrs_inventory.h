#ifndef OSRS_INVENTORY_H
#define OSRS_INVENTORY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osrs_types.h"
#include "osrs_items.h"

#define OSRS_INVENTORY_SIZE 28

typedef struct {
    uint8_t item_idx;
    uint8_t binary_padding0;
    uint16_t raw_osrs_id;
    uint8_t dose;
    uint8_t binary_padding1;
} OsrsInventoryCell;

typedef struct {
    OsrsInventoryCell cells[OSRS_INVENTORY_SIZE];
} OsrsInventorySlotSnapshot;

static inline int osrs_inventory_slot_valid(int slot) {
    return slot >= 0 && slot < OSRS_INVENTORY_SIZE;
}

static inline OsrsInventoryCell osrs_inventory_cell_empty(void) {
    return (OsrsInventoryCell){
        .item_idx = ITEM_NONE,
        .raw_osrs_id = 0,
        .dose = 0,
    };
}

static inline int osrs_inventory_cell_is_empty(const OsrsInventoryCell* cell) {
    return cell->item_idx == ITEM_NONE && cell->raw_osrs_id == 0;
}

static inline OsrsInventoryCell osrs_inventory_cell_from_item(uint8_t item_idx) {
    if (item_idx == ITEM_NONE) return osrs_inventory_cell_empty();
    if (item_idx >= NUM_ITEMS) {
        fprintf(stderr, "inventory cell: invalid item index %u\n", item_idx);
        abort();
    }
    return (OsrsInventoryCell){
        .item_idx = item_idx,
        .raw_osrs_id = ITEM_DATABASE[item_idx].item_id,
        .dose = 0,
    };
}

static inline void osrs_inventory_swap_cells(
    OsrsInventoryCell cells[OSRS_INVENTORY_SIZE],
    int source_slot,
    int target_slot
) {
    if (!osrs_inventory_slot_valid(source_slot) ||
            !osrs_inventory_slot_valid(target_slot)) {
        fprintf(stderr, "inventory swap: invalid slots %d -> %d\n",
            source_slot, target_slot);
        abort();
    }
    if (source_slot == target_slot) return;
    OsrsInventoryCell tmp = cells[target_slot];
    cells[target_slot] = cells[source_slot];
    cells[source_slot] = tmp;
}

static inline OsrsInventorySlotSnapshot osrs_inventory_slot_snapshot(
    const OsrsInventoryCell cells[OSRS_INVENTORY_SIZE]
) {
    OsrsInventorySlotSnapshot snapshot;
    memcpy(snapshot.cells, cells, sizeof(snapshot.cells));
    return snapshot;
}

static inline int osrs_item_gear_slot(uint8_t item_idx) {
    if (item_idx >= NUM_ITEMS) return -1;
    switch (ITEM_DATABASE[item_idx].slot) {
        case SLOT_HEAD:   return GEAR_SLOT_HEAD;
        case SLOT_CAPE:   return GEAR_SLOT_CAPE;
        case SLOT_NECK:   return GEAR_SLOT_NECK;
        case SLOT_WEAPON: return GEAR_SLOT_WEAPON;
        case SLOT_BODY:   return GEAR_SLOT_BODY;
        case SLOT_SHIELD: return GEAR_SLOT_SHIELD;
        case SLOT_LEGS:   return GEAR_SLOT_LEGS;
        case SLOT_HANDS:  return GEAR_SLOT_HANDS;
        case SLOT_FEET:   return GEAR_SLOT_FEET;
        case SLOT_RING:   return GEAR_SLOT_RING;
        case SLOT_AMMO:   return GEAR_SLOT_AMMO;
        default: return -1;
    }
}

#endif
