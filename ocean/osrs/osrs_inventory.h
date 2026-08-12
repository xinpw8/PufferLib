#ifndef OSRS_INVENTORY_H
#define OSRS_INVENTORY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osrs_types.h"
#include "osrs_items.h"

#define OSRS_INVENTORY_SIZE 28

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
    OSRS_CONSUMABLE_COUNT = 17,
} OsrsConsumableKind;

#include "osrs_item_obs_generated.h"

typedef struct {
    const Item* item;
    uint16_t raw_osrs_id;
    uint16_t next_content_code;
    uint8_t item_idx;
    int8_t gear_slot;
    uint8_t click_action;
    uint8_t consumable_kind;
    uint8_t dose_count;
    uint8_t attack_style;
    float observation_row[OSRS_ITEM_OBS_TABLE_COLS];
} OsrsItemContentMetadata;

#define OSRS_ITEM_CONTENT_METADATA_ROW( \
    CODE, ITEM_POINTER, ITEM_IDX, RAW_OSRS_ID, GEAR_SLOT, CLICK_ACTION, \
    CONSUMABLE_KIND, DOSE_COUNT, NEXT_CONTENT_CODE, ATTACK_STYLE, ...) \
    [CODE] = { \
        .item = ITEM_POINTER, \
        .raw_osrs_id = RAW_OSRS_ID, \
        .next_content_code = NEXT_CONTENT_CODE, \
        .item_idx = ITEM_IDX, \
        .gear_slot = GEAR_SLOT, \
        .click_action = CLICK_ACTION, \
        .consumable_kind = CONSUMABLE_KIND, \
        .dose_count = DOSE_COUNT, \
        .attack_style = ATTACK_STYLE, \
        .observation_row = {__VA_ARGS__}, \
    },
static const OsrsItemContentMetadata
    OSRS_ITEM_CONTENT_METADATA[OSRS_ITEM_CONTENT_COUNT] = {
    OSRS_ITEM_CONTENT_ROWS(OSRS_ITEM_CONTENT_METADATA_ROW)
};
#undef OSRS_ITEM_CONTENT_METADATA_ROW

#define OSRS_RAW_CONTENT_CODE_ROW( \
    CODE, ITEM_POINTER, ITEM_IDX, RAW_OSRS_ID, GEAR_SLOT, CLICK_ACTION, \
    CONSUMABLE_KIND, DOSE_COUNT, NEXT_CONTENT_CODE, ATTACK_STYLE, ...) \
    [RAW_OSRS_ID] = (uint16_t)((CODE) + 1),
static const uint16_t OSRS_CONTENT_CODE_BY_RAW_OSRS_ID[UINT16_MAX + 1] = {
    OSRS_ITEM_CONTENT_ROWS(OSRS_RAW_CONTENT_CODE_ROW)
};
#undef OSRS_RAW_CONTENT_CODE_ROW

#define OSRS_CONSUMABLE_CONTENT_CODE_ROW(KIND, DOSE, CODE) \
    [KIND][DOSE] = (uint16_t)((CODE) + 1),
static const uint16_t
    OSRS_CONTENT_CODE_BY_CONSUMABLE_KIND_AND_DOSE[OSRS_CONSUMABLE_COUNT][5] = {
    OSRS_CONSUMABLE_CONTENT_ROWS(OSRS_CONSUMABLE_CONTENT_CODE_ROW)
};
#undef OSRS_CONSUMABLE_CONTENT_CODE_ROW

typedef struct {
    uint16_t content_code;
} OsrsInventoryCell;

typedef struct {
    OsrsInventoryCell cells[OSRS_INVENTORY_SIZE];
} OsrsInventorySlotSnapshot;

_Static_assert(
    sizeof(OSRS_ITEM_CONTENT_METADATA) /
        sizeof(OSRS_ITEM_CONTENT_METADATA[0]) == OSRS_ITEM_CONTENT_COUNT,
    "generated item metadata row count must match its content-code domain");
_Static_assert(OSRS_ITEM_CONTENT_COUNT <= UINT16_MAX,
    "item content code must fit uint16_t");
_Static_assert(sizeof(OsrsInventoryCell) == sizeof(uint16_t),
    "inventory cells carry only canonical content identity");

static inline const OsrsItemContentMetadata* osrs_item_content_metadata(
    uint16_t content_code
) {
    if (content_code >= OSRS_ITEM_CONTENT_COUNT) {
        fprintf(stderr, "inventory content: invalid content code %u\n", content_code);
        abort();
    }
    return &OSRS_ITEM_CONTENT_METADATA[content_code];
}

static inline uint16_t osrs_inventory_content_code_from_item(uint8_t item_idx) {
    if (item_idx >= NUM_ITEMS) {
        fprintf(stderr, "inventory content: invalid item index %u\n", item_idx);
        abort();
    }
    return (uint16_t)(1 + item_idx);
}

static inline uint16_t osrs_inventory_content_code_from_raw_osrs_id(
    uint16_t raw_osrs_id
) {
    uint16_t encoded = OSRS_CONTENT_CODE_BY_RAW_OSRS_ID[raw_osrs_id];
    if (encoded == 0) {
        fprintf(stderr, "inventory content: unrepresentable raw OSRS id %u\n",
            raw_osrs_id);
        abort();
    }
    return (uint16_t)(encoded - 1);
}

static inline uint16_t osrs_inventory_content_code_from_consumable(
    OsrsConsumableKind kind,
    uint8_t dose_count
) {
    if (kind <= OSRS_CONSUMABLE_NONE || kind >= OSRS_CONSUMABLE_COUNT ||
            dose_count > 4) {
        fprintf(stderr, "inventory content: invalid consumable kind %d dose %u\n",
            (int)kind, dose_count);
        abort();
    }
    uint16_t encoded =
        OSRS_CONTENT_CODE_BY_CONSUMABLE_KIND_AND_DOSE[kind][dose_count];
    if (encoded == 0) {
        fprintf(stderr, "inventory content: unrepresentable consumable kind %d dose %u\n",
            (int)kind, dose_count);
        abort();
    }
    return (uint16_t)(encoded - 1);
}

static inline int osrs_inventory_slot_valid(int slot) {
    return slot >= 0 && slot < OSRS_INVENTORY_SIZE;
}

static inline OsrsInventoryCell osrs_inventory_cell_from_content_code(
    uint16_t content_code
) {
    (void)osrs_item_content_metadata(content_code);
    return (OsrsInventoryCell){.content_code = content_code};
}

static inline OsrsInventoryCell osrs_inventory_cell_empty(void) {
    return osrs_inventory_cell_from_content_code(0);
}

static inline int osrs_inventory_cell_is_empty(const OsrsInventoryCell* cell) {
    return cell->content_code == 0;
}

static inline OsrsInventoryCell osrs_inventory_cell_from_item(uint8_t item_idx) {
    if (item_idx == ITEM_NONE) return osrs_inventory_cell_empty();
    return osrs_inventory_cell_from_content_code(
        osrs_inventory_content_code_from_item(item_idx));
}

static inline OsrsInventoryCell osrs_inventory_cell_from_raw_osrs_id(
    uint16_t raw_osrs_id
) {
    return osrs_inventory_cell_from_content_code(
        osrs_inventory_content_code_from_raw_osrs_id(raw_osrs_id));
}

static inline const OsrsItemContentMetadata* osrs_inventory_cell_metadata(
    const OsrsInventoryCell* cell
) {
    return osrs_item_content_metadata(cell->content_code);
}

static inline uint8_t osrs_inventory_cell_item_index(
    const OsrsInventoryCell* cell
) {
    return osrs_inventory_cell_metadata(cell)->item_idx;
}

static inline uint16_t osrs_inventory_cell_raw_osrs_id(
    const OsrsInventoryCell* cell
) {
    return osrs_inventory_cell_metadata(cell)->raw_osrs_id;
}

static inline uint8_t osrs_inventory_cell_dose_count(
    const OsrsInventoryCell* cell
) {
    return osrs_inventory_cell_metadata(cell)->dose_count;
}

static inline int osrs_item_gear_slot(uint8_t item_idx) {
    if (item_idx == ITEM_NONE) return -1;
    return osrs_item_content_metadata(
        osrs_inventory_content_code_from_item(item_idx))->gear_slot;
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
    OsrsInventoryCell temporary = cells[target_slot];
    cells[target_slot] = cells[source_slot];
    cells[source_slot] = temporary;
}

static inline OsrsInventorySlotSnapshot osrs_inventory_slot_snapshot(
    const OsrsInventoryCell cells[OSRS_INVENTORY_SIZE]
) {
    OsrsInventorySlotSnapshot snapshot;
    memcpy(snapshot.cells, cells, sizeof(snapshot.cells));
    return snapshot;
}

#endif
