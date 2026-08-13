/* Emits the shared OSRS item content metadata and observation table.
 *
 *   cc -std=c11 -O2 -I. -o /tmp/gen_osrs_item_obs \
 *      ocean/osrs/tools/gen_osrs_item_obs_table.c -lm
 *   /tmp/gen_osrs_item_obs ocean/osrs/osrs_item_obs_generated.h \
 *                           ocean/osrs/osrs_item_obs_table.inc
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ocean/osrs/osrs_inventory_clicks.h"

#define GEN_BASE_HITPOINTS 99
#define GEN_BASE_PRAYER 99
#define GEN_BASE_RANGED 99
#define GEN_INERT_RAW_OSRS_ID 27281

typedef struct {
    uint16_t raw_osrs_id;
    OsrsClickAction click_action;
    OsrsConsumableKind consumable_kind;
    uint8_t dose_count;
} GenConsumable;

static const GenConsumable GEN_CONSUMABLES[] = {
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

#define GEN_CONSUMABLE_COUNT \
    ((int)(sizeof(GEN_CONSUMABLES) / sizeof(GEN_CONSUMABLES[0])))
#define GEN_GEAR_BASE 1
#define GEN_CONSUMABLE_BASE (GEN_GEAR_BASE + NUM_ITEMS)
#define GEN_INERT_CODE (GEN_CONSUMABLE_BASE + GEN_CONSUMABLE_COUNT)
#define GEN_CONTENT_COUNT (GEN_INERT_CODE + 1)

static FILE* open_or_die(const char* path) {
    FILE* file = fopen(path, "w");
    if (!file) {
        fprintf(stderr, "OSRS item metadata: cannot write %s\n", path);
        exit(1);
    }
    return file;
}

static void emit_float(FILE* out, float value) {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%.9g", (double)value);
    if (!strpbrk(buffer, ".eEnN")) {
        strncat(buffer, ".0", sizeof(buffer) - strlen(buffer) - 1);
    }
    fprintf(out, "%sf", buffer);
}

static int gear_slot_for_item_slot(uint8_t slot) {
    switch (slot) {
        case SLOT_HEAD: return GEAR_SLOT_HEAD;
        case SLOT_CAPE: return GEAR_SLOT_CAPE;
        case SLOT_NECK: return GEAR_SLOT_NECK;
        case SLOT_WEAPON: return GEAR_SLOT_WEAPON;
        case SLOT_BODY: return GEAR_SLOT_BODY;
        case SLOT_SHIELD: return GEAR_SLOT_SHIELD;
        case SLOT_LEGS: return GEAR_SLOT_LEGS;
        case SLOT_HANDS: return GEAR_SLOT_HANDS;
        case SLOT_FEET: return GEAR_SLOT_FEET;
        case SLOT_RING: return GEAR_SLOT_RING;
        case SLOT_AMMO: return GEAR_SLOT_AMMO;
        default:
            fprintf(stderr, "OSRS item metadata: unsupported item slot %u\n", slot);
            exit(1);
    }
}

static int content_code_for_raw_osrs_id(uint16_t raw_osrs_id) {
    if (raw_osrs_id == 0) return 0;
    for (int item_idx = 0; item_idx < NUM_ITEMS; item_idx++) {
        if (ITEM_DATABASE[item_idx].item_id == raw_osrs_id) {
            return GEN_GEAR_BASE + item_idx;
        }
    }
    for (int index = 0; index < GEN_CONSUMABLE_COUNT; index++) {
        if (GEN_CONSUMABLES[index].raw_osrs_id == raw_osrs_id) {
            return GEN_CONSUMABLE_BASE + index;
        }
    }
    if (raw_osrs_id == GEN_INERT_RAW_OSRS_ID) return GEN_INERT_CODE;
    fprintf(stderr, "OSRS item metadata: unrepresentable raw OSRS id %u\n", raw_osrs_id);
    exit(1);
}

static int next_content_code(const GenConsumable* consumable) {
    if (consumable->click_action != OSRS_CLICK_DRINK) return 0;
    if (consumable->dose_count == 1) return 0;
    for (int index = 0; index < GEN_CONSUMABLE_COUNT; index++) {
        const GenConsumable* candidate = &GEN_CONSUMABLES[index];
        if (candidate->consumable_kind == consumable->consumable_kind &&
                candidate->dose_count + 1 == consumable->dose_count) {
            return GEN_CONSUMABLE_BASE + index;
        }
    }
    fprintf(stderr, "OSRS item metadata: incomplete dose chain for raw OSRS id %u\n",
        consumable->raw_osrs_id);
    exit(1);
}

static void build_row(
    float* row,
    uint8_t item_idx,
    uint16_t raw_osrs_id,
    OsrsConsumableKind consumable_kind,
    uint8_t dose_count
) {
    OsrsItemContentMetadata metadata = {
        .item = item_idx == ITEM_NONE ? NULL : &ITEM_DATABASE[item_idx],
        .raw_osrs_id = raw_osrs_id,
        .item_idx = item_idx,
        .consumable_kind = (uint8_t)consumable_kind,
        .dose_count = dose_count,
        .attack_style =
            item_idx == ITEM_NONE ? 0 : get_item_attack_style(item_idx),
    };
    osrs_write_item_content_affordance_features_compact(
        row, &metadata, 0,
        GEN_BASE_HITPOINTS, GEN_BASE_PRAYER, GEN_BASE_RANGED);
}

static void emit_content_row(
    FILE* out,
    int code,
    const char* item_pointer,
    uint8_t item_idx,
    uint16_t raw_osrs_id,
    int gear_slot,
    OsrsClickAction click_action,
    OsrsConsumableKind consumable_kind,
    uint8_t dose_count,
    int next_code,
    int attack_style,
    const float* observation_row,
    int final_row
) {
    fprintf(out, "    X(%d, %s, %u, %u, %d, %d, %d, %u, %d, %d",
        code, item_pointer, item_idx, raw_osrs_id, gear_slot,
        (int)click_action, (int)consumable_kind, dose_count, next_code,
        attack_style);
    for (int feature = 0;
            feature < OSRS_INVENTORY_CELL_OBS_FEATURES_COMPACT;
            feature++) {
        fprintf(out, ", ");
        emit_float(out, observation_row[feature]);
    }
    fprintf(out, ")%s\n", final_row ? "" : " \\");
}

static void write_content_rows(FILE* out) {
    fprintf(out, "#define OSRS_ITEM_CONTENT_ROWS(X) \\\n");
    float row[OSRS_INVENTORY_CELL_OBS_FEATURES_COMPACT];
    build_row(row, ITEM_NONE, 0, OSRS_CONSUMABLE_NONE, 0);
    emit_content_row(out, 0, "NULL", ITEM_NONE, 0, -1,
        OSRS_CLICK_NONE, OSRS_CONSUMABLE_NONE, 0, 0, 0, row, 0);

    for (int item_idx = 0; item_idx < NUM_ITEMS; item_idx++) {
        char item_pointer[64];
        snprintf(item_pointer, sizeof(item_pointer),
            "&ITEM_DATABASE[%d]", item_idx);
        const Item* item = &ITEM_DATABASE[item_idx];
        build_row(row, (uint8_t)item_idx, item->item_id,
            OSRS_CONSUMABLE_NONE, 0);
        emit_content_row(out, GEN_GEAR_BASE + item_idx, item_pointer,
            (uint8_t)item_idx, item->item_id,
            gear_slot_for_item_slot(item->slot), OSRS_CLICK_EQUIP,
            OSRS_CONSUMABLE_NONE, 0, 0, get_item_attack_style(item_idx),
            row, 0);
    }

    for (int index = 0; index < GEN_CONSUMABLE_COUNT; index++) {
        const GenConsumable* consumable = &GEN_CONSUMABLES[index];
        build_row(row, ITEM_NONE, consumable->raw_osrs_id,
            consumable->consumable_kind, consumable->dose_count);
        emit_content_row(out, GEN_CONSUMABLE_BASE + index, "NULL", ITEM_NONE,
            consumable->raw_osrs_id, -1, consumable->click_action,
            consumable->consumable_kind, consumable->dose_count,
            next_content_code(consumable), 0, row, 0);
    }

    build_row(row, ITEM_NONE, GEN_INERT_RAW_OSRS_ID,
        OSRS_CONSUMABLE_NONE, 0);
    emit_content_row(out, GEN_INERT_CODE, "NULL", ITEM_NONE,
        GEN_INERT_RAW_OSRS_ID, -1, OSRS_CLICK_NONE, OSRS_CONSUMABLE_NONE,
        0, 0, 0, row, 1);
}

static void write_consumable_rows(FILE* out) {
    fprintf(out, "\n#define OSRS_CONSUMABLE_CONTENT_ROWS(X) \\\n");
    for (int index = 0; index < GEN_CONSUMABLE_COUNT; index++) {
        const GenConsumable* consumable = &GEN_CONSUMABLES[index];
        fprintf(out, "    X(%d, %d, %u)%s\n",
            (int)consumable->consumable_kind, consumable->dose_count,
            GEN_CONSUMABLE_BASE + index,
            index + 1 == GEN_CONSUMABLE_COUNT ? "" : " \\");
    }
}

static void write_header(const char* path) {
    FILE* out = open_or_die(path);
    fprintf(out,
        "/* Generated by ocean/osrs/tools/gen_osrs_item_obs_table.c. Do not edit. */\n"
        "#ifndef OSRS_ITEM_OBS_GENERATED_H\n"
        "#define OSRS_ITEM_OBS_GENERATED_H\n\n"
        "#define OSRS_ITEM_CONTENT_COUNT %d\n"
        "#define OSRS_ITEM_OBS_TABLE_ROWS %d\n"
        "#define OSRS_ITEM_OBS_TABLE_COLS %d\n"
        "#define OSRS_ITEM_OBS_TABLE_BASE_HITPOINTS %d\n"
        "#define OSRS_ITEM_OBS_TABLE_BASE_PRAYER %d\n"
        "#define OSRS_ITEM_OBS_TABLE_BASE_RANGED %d\n\n"
        "#define OSRS_ITEM_OBS_CODE_SCALE %d\n\n",
        GEN_CONTENT_COUNT,
        GEN_CONTENT_COUNT,
        OSRS_INVENTORY_CELL_OBS_FEATURES_COMPACT,
        GEN_BASE_HITPOINTS,
        GEN_BASE_PRAYER,
        GEN_BASE_RANGED,
        OSRS_ITEM_OBS_CODE_SCALE);
    write_content_rows(out);
    write_consumable_rows(out);
    fprintf(out, "\n#endif\n");
    fclose(out);
}

static void write_table(const char* path) {
    FILE* out = open_or_die(path);
    fprintf(out,
        "/* Generated by ocean/osrs/tools/gen_osrs_item_obs_table.c. Do not edit.\n"
        " * Expands the observation fields from the canonical content rows. */\n"
        "#define OSRS_ITEM_CONTENT_OBS_ROW(code, item_pointer, item_idx, raw_osrs_id, gear_slot, click_action, consumable_kind, dose_count, next_content_code, attack_style, ...) {__VA_ARGS__},\n"
        "OSRS_ITEM_CONTENT_ROWS(OSRS_ITEM_CONTENT_OBS_ROW)\n"
        "#undef OSRS_ITEM_CONTENT_OBS_ROW\n");
    fclose(out);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <header.h> <table.inc>\n", argv[0]);
        return 1;
    }
    if (GEN_CONTENT_COUNT > OSRS_ITEM_OBS_CODE_SCALE) {
        fprintf(stderr,
            "OSRS item metadata: %d codes exceeds observation scale %d\n",
            GEN_CONTENT_COUNT, OSRS_ITEM_OBS_CODE_SCALE);
        return 1;
    }
    for (int code = 0; code < GEN_CONTENT_COUNT; code++) {
        int decoded = osrs_inventory_cell_obs_code_decode(
            osrs_inventory_cell_obs_code_encode(code));
        if (decoded != code) {
            fprintf(stderr, "OSRS item metadata: code %d decodes as %d\n",
                code, decoded);
            return 1;
        }
    }
    for (int item_idx = 0; item_idx < NUM_ITEMS; item_idx++) {
        if (content_code_for_raw_osrs_id(ITEM_DATABASE[item_idx].item_id) !=
                GEN_GEAR_BASE + item_idx) {
            fprintf(stderr, "OSRS item metadata: item %d raw id is ambiguous\n",
                item_idx);
            return 1;
        }
    }
    write_header(argv[1]);
    write_table(argv[2]);
    return 0;
}
