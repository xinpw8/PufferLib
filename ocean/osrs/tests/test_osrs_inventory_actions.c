#include <stdio.h>
#include <string.h>

#include "../osrs_inventory_actions.h"
#include "osrs_test_check.h"

static uint32_t test_rng = 0x1234567u;

static uint32_t next_rand(void) {
    test_rng ^= test_rng << 13;
    test_rng ^= test_rng >> 17;
    test_rng ^= test_rng << 5;
    return test_rng;
}

static int rand_below(int n) {
    return (int)(next_rand() % (uint32_t)n);
}

static const uint8_t TEST_ITEM_POOL[] = {
    ITEM_MYSTIC_HAT, ITEM_GOD_CAPE, ITEM_GLORY, ITEM_AMETHYST_ARROW,
    ITEM_TRIDENT_OF_SWAMP, ITEM_BOOK_OF_DARKNESS, ITEM_MYSTIC_TOP,
    ITEM_MYSTIC_BOTTOM, ITEM_BARROWS_GLOVES, ITEM_MYSTIC_BOOTS,
    ITEM_RING_OF_RECOIL, ITEM_TWISTED_BOW, ITEM_BOW_OF_FAERDHINEN,
    ITEM_MASORI_MASK_F, ITEM_MASORI_BODY_F, ITEM_ELIDINIS_WARD_F,
};
#define TEST_ITEM_POOL_SIZE ((int)(sizeof(TEST_ITEM_POOL) / sizeof(*TEST_ITEM_POOL)))

static void count_items(
    const Player* p, const OsrsInventoryCell* cells, int counts[NUM_ITEMS]
) {
    memset(counts, 0, NUM_ITEMS * sizeof(int));
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++)
        if (p->equipped[slot] != ITEM_NONE) counts[p->equipped[slot]]++;
    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++) {
        uint8_t item_idx = osrs_inventory_cell_item_index(&cells[i]);
        if (item_idx != ITEM_NONE) counts[item_idx]++;
    }
}

static void run_equip_storm(int trial) {
    Player p;
    memset(&p, 0, sizeof(p));
    memset(p.equipped, ITEM_NONE, NUM_GEAR_SLOTS);
    p.base_hitpoints = 99;
    p.current_hitpoints = 50;
    p.base_prayer = 99;

    OsrsInventoryCell cells[OSRS_INVENTORY_SIZE];
    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++)
        cells[i] = osrs_inventory_cell_empty();

    int num_seeded = 4 + rand_below(OSRS_INVENTORY_SIZE - 6);
    for (int i = 0; i < num_seeded; i++)
        cells[i] = osrs_inventory_cell_from_item(
            TEST_ITEM_POOL[rand_below(TEST_ITEM_POOL_SIZE)]);
    osrs_refresh_player_equipment(&p);

    int before[NUM_ITEMS];
    int after[NUM_ITEMS];

    for (int step = 0; step < 200; step++) {
        count_items(&p, cells, before);

        OsrsInventoryClickActions clicks;
        for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++)
            clicks.equip_by_slot[slot] = rand_below(OSRS_INVENTORY_SIZE + 1);
        clicks.eat = rand_below(OSRS_INVENTORY_SIZE + 1);
        clicks.drink = rand_below(OSRS_INVENTORY_SIZE + 1);

        OsrsInventoryTickIntent intent =
            osrs_resolve_inventory_tick_intent(&p, cells, &clicks);
        OsrsInventoryApplyStep apply_step;
        while (osrs_inventory_intent_next(&intent, &apply_step)) {
            CHECK("gear-only storm must only yield equip steps",
                apply_step.kind == OSRS_INVENTORY_APPLY_EQUIP);
            int slot = osrs_equip_from_cell(&p, cells, apply_step.cell_idx);
            if (slot >= 0) {
                CHECK("equip must leave an item in the slot",
                    p.equipped[slot] != ITEM_NONE);
                CHECK("equipped item must match its gear slot",
                    osrs_item_gear_slot(p.equipped[slot]) == slot);
            }
        }

        count_items(&p, cells, after);
        for (int item = 0; item < NUM_ITEMS; item++) {
            if (before[item] == after[item]) continue;
            fprintf(stderr, "trial %d step %d: item %d count %d -> %d\n",
                trial, step, item, before[item], after[item]);
            CHECK("equip storm must conserve the item multiset", 0);
            return;
        }

        uint8_t weapon = p.equipped[GEAR_SLOT_WEAPON];
        if (weapon != ITEM_NONE && item_is_two_handed(weapon)) {
            CHECK("two-handed weapon must never coexist with a shield",
                p.equipped[GEAR_SLOT_SHIELD] == ITEM_NONE);
        }
    }
}

static void run_drink_dose_check(void) {
    Player p;
    memset(&p, 0, sizeof(p));
    memset(p.equipped, ITEM_NONE, NUM_GEAR_SLOTS);
    p.base_hitpoints = 99;
    p.current_hitpoints = 99;

    OsrsInventoryCell cells[OSRS_INVENTORY_SIZE];
    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++)
        cells[i] = osrs_inventory_cell_empty();
    cells[0] = osrs_inventory_cell_from_raw_osrs_id(2434);
    ASSERT_INT_EQ("prayer potion seeds with 4 doses",
        osrs_inventory_cell_dose_count(&cells[0]), 4);

    for (int expected_dose = 4; expected_dose >= 1; expected_dose--) {
        p.potion_timer = 0;
        OsrsInventoryClickActions clicks;
        memset(&clicks, 0, sizeof(clicks));
        clicks.drink = 1;
        OsrsInventoryTickIntent intent =
            osrs_resolve_inventory_tick_intent(&p, cells, &clicks);
        OsrsInventoryApplyStep apply_step;
        int steps = 0;
        while (osrs_inventory_intent_next(&intent, &apply_step)) {
            ASSERT_INT_EQ("drink click must yield a drink step",
                apply_step.kind, OSRS_INVENTORY_APPLY_DRINK);
            ASSERT_INT_EQ("resolution must carry the current dose count",
                apply_step.resolution.dose_count, expected_dose);
            osrs_inventory_cell_decrement_drink(&cells[0], apply_step.resolution);
            steps++;
        }
        ASSERT_INT_EQ("one drink step per tick", steps, 1);
        ASSERT_INT_EQ("dose decrements by one",
            osrs_inventory_cell_dose_count(&cells[0]), expected_dose - 1);
    }
    CHECK("cell empties after the last dose",
        osrs_inventory_cell_is_empty(&cells[0]));

    p.potion_timer = 3;
    OsrsInventoryClickActions clicks;
    memset(&clicks, 0, sizeof(clicks));
    clicks.drink = 1;
    OsrsInventoryTickIntent intent =
        osrs_resolve_inventory_tick_intent(&p, cells, &clicks);
    CHECK("empty cell resolves to no drink intent",
        !osrs_inventory_tick_intent_has_effect(&intent));
}

static void test_weapon_ranges(void) {
    const Item* dragon_dart = get_item(ITEM_DRAGON_DART);
    ASSERT_INT_EQ("dragon dart weapon range", dragon_dart->attack_range, 3);

    uint8_t unarmed_loadout[NUM_GEAR_SLOTS];
    memset(unarmed_loadout, ITEM_NONE, sizeof(unarmed_loadout));
    EquipmentBonuses unarmed = {0};
    osrs_sum_equipment_bonuses(unarmed_loadout, &unarmed);
    ASSERT_INT_EQ("unarmed attack speed", unarmed.attack_speed, 4);
    ASSERT_INT_EQ("unarmed attack range", unarmed.attack_range, 1);
}
static void test_content_code_round_trips(void) {
    ASSERT_INT_EQ("inventory cell stores only one content code",
        (int)sizeof(OsrsInventoryCell), (int)sizeof(uint16_t));
    for (uint16_t code = 0; code < OSRS_ITEM_CONTENT_COUNT; code++) {
        const OsrsItemContentMetadata* metadata =
            osrs_item_content_metadata(code);
        OsrsInventoryCell cell = osrs_inventory_cell_from_content_code(code);
        ASSERT_INT_EQ("cell preserves content code", cell.content_code, code);
        ASSERT_INT_EQ("content code preserves raw OSRS id",
            osrs_inventory_cell_raw_osrs_id(&cell), metadata->raw_osrs_id);
        ASSERT_INT_EQ("content code preserves item index",
            osrs_inventory_cell_item_index(&cell), metadata->item_idx);
        ASSERT_INT_EQ("raw OSRS id round trips to content code",
            osrs_inventory_content_code_from_raw_osrs_id(metadata->raw_osrs_id),
            code);
        if (metadata->item_idx != ITEM_NONE) {
            ASSERT_INT_EQ("item index round trips to content code",
                osrs_inventory_content_code_from_item(metadata->item_idx), code);
        }
    }
}


int main(void) {
    for (int trial = 0; trial < 500; trial++)
        run_equip_storm(trial);
    run_drink_dose_check();
    test_weapon_ranges();
    test_content_code_round_trips();
    return osrs_test_summary();
}
