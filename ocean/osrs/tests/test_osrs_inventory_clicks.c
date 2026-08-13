#include <stdio.h>
#include <math.h>

#include "ocean/osrs/osrs_inventory_clicks.h"

#define CHECK(label, cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", label); \
        return 1; \
    } \
} while (0)

static int test_item_index_classification(void) {
    CHECK("venator bow item index equips",
          osrs_item_click_action(ITEM_VENATOR_BOW) == OSRS_CLICK_EQUIP);
    CHECK("empty item index noops",
          osrs_item_click_action(ITEM_NONE) == OSRS_CLICK_NONE);
    CHECK("gear item has no consumable tag",
          osrs_item_click_consumable_kind(ITEM_VENATOR_BOW) ==
              OSRS_CONSUMABLE_NONE);
    return 0;
}

static int test_raw_consumable_classification(void) {
    OsrsConsumableClick brew =
        osrs_consumable_click_lookup_raw_osrs_id(6685);
    CHECK("sara brew drinks", brew.click_action == OSRS_CLICK_DRINK);
    CHECK("sara brew kind", brew.consumable_kind == OSRS_CONSUMABLE_BREW);
    CHECK("sara brew four-dose count", brew.dose_count == 4);

    OsrsConsumableClick shark =
        osrs_consumable_click_lookup_raw_osrs_id(385);
    CHECK("shark eats", shark.click_action == OSRS_CLICK_EAT);
    CHECK("shark food kind", shark.consumable_kind == OSRS_CONSUMABLE_SHARK_FOOD);
    CHECK("shark has no dose", shark.dose_count == 0);

    OsrsConsumableClick unknown =
        osrs_consumable_click_lookup_raw_osrs_id(9999);
    CHECK("unknown raw id noops", unknown.click_action == OSRS_CLICK_NONE);
    CHECK("unknown raw id has no kind",
          unknown.consumable_kind == OSRS_CONSUMABLE_NONE);
    CHECK("unknown raw id has no dose", unknown.dose_count == 0);
    return 0;
}

static int test_sara_brew_dose_variants(void) {
    OsrsConsumableClick brew4 =
        osrs_consumable_click_lookup_raw_osrs_id(6685);
    OsrsConsumableClick brew3 =
        osrs_consumable_click_lookup_raw_osrs_id(6687);
    OsrsConsumableClick brew2 =
        osrs_consumable_click_lookup_raw_osrs_id(6689);
    OsrsConsumableClick brew1 =
        osrs_consumable_click_lookup_raw_osrs_id(6691);

    CHECK("6685 is four-dose", brew4.dose_count == 4);
    CHECK("6687 is three-dose", brew3.dose_count == 3);
    CHECK("6689 is two-dose", brew2.dose_count == 2);
    CHECK("6691 is one-dose", brew1.dose_count == 1);
    return 0;
}

static int test_dose_after_drink(void) {
    CHECK("four-dose count decrements to three",
          osrs_consumable_dose_count_after_drink(4) == 3);
    CHECK("one-dose count decrements to empty",
          osrs_consumable_dose_count_after_drink(1) == 0);
    CHECK("6685 decrements to 6687",
          osrs_consumable_raw_osrs_id_after_drink(6685) == 6687);
    CHECK("6687 decrements to 6689",
          osrs_consumable_raw_osrs_id_after_drink(6687) == 6689);
    CHECK("6691 decrements to empty",
          osrs_consumable_raw_osrs_id_after_drink(6691) == 0);
    CHECK("guthix rest uses odd id family",
          osrs_consumable_raw_osrs_id_after_drink(4417) == 4419);
    return 0;
}

static int test_pure_click_interpreter(void) {
    OsrsInventoryClickResolution gear = osrs_inventory_click_interpret(
        ITEM_VENATOR_BOW,
        27610,
        OSRS_CLICK_TICK_FIRST
    );
    CHECK("gear click resolves equip", gear.click_action == OSRS_CLICK_EQUIP);
    CHECK("gear click has no consumable kind",
          gear.consumable_kind == OSRS_CONSUMABLE_NONE);

    OsrsInventoryClickResolution brew = osrs_inventory_click_interpret(
        ITEM_NONE,
        6685,
        OSRS_CLICK_TICK_FIRST
    );
    CHECK("brew click resolves drink", brew.click_action == OSRS_CLICK_DRINK);
    CHECK("brew interpreter kind", brew.consumable_kind == OSRS_CONSUMABLE_BREW);
    CHECK("brew interpreter dose", brew.dose_count == 4);
    CHECK("brew interpreter next raw id", brew.raw_osrs_id_after_drink == 6687);

    OsrsInventoryClickResolution shark = osrs_inventory_click_interpret(
        ITEM_NONE,
        385,
        OSRS_CLICK_TICK_FIRST
    );
    CHECK("shark click resolves eat", shark.click_action == OSRS_CLICK_EAT);
    CHECK("shark interpreter kind",
          shark.consumable_kind == OSRS_CONSUMABLE_SHARK_FOOD);

    OsrsInventoryClickResolution duplicate = osrs_inventory_click_interpret(
        ITEM_NONE,
        6685,
        OSRS_CLICK_TICK_DUPLICATE
    );
    CHECK("duplicate click noops", duplicate.click_action == OSRS_CLICK_NONE);
    return 0;
}

static int test_cell_click_attributes_to_slot_item(void) {
    OsrsInventoryCell cells[OSRS_INVENTORY_SIZE];
    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++)
        cells[i] = osrs_inventory_cell_empty();
    cells[3] = osrs_inventory_cell_from_raw_osrs_id(3024);
    cells[4] = osrs_inventory_cell_from_item(ITEM_TWISTED_BOW);

    OsrsInventorySlotSnapshot snapshot = osrs_inventory_slot_snapshot(cells);
    OsrsInventoryClickResolution restore =
        osrs_inventory_snapshot_click_interpret(
            &snapshot, 3, OSRS_CLICK_TICK_FIRST);
    OsrsInventoryClickResolution bow =
        osrs_inventory_snapshot_click_interpret(
            &snapshot, 4, OSRS_CLICK_TICK_FIRST);

    CHECK("slot 3 resolves restore, not adjacent bow",
          restore.click_action == OSRS_CLICK_DRINK &&
          restore.consumable_kind == OSRS_CONSUMABLE_SUPER_RESTORE);
    CHECK("slot 4 resolves bow equip, not adjacent restore",
          bow.click_action == OSRS_CLICK_EQUIP);
    return 0;
}

static int test_cell_drink_decrements_one_dose(void) {
    OsrsInventoryCell restore = osrs_inventory_cell_from_raw_osrs_id(3024);
    OsrsInventoryClickResolution resolution =
        osrs_inventory_cell_click_interpret(&restore, OSRS_CLICK_TICK_FIRST);
    osrs_inventory_cell_decrement_drink(&restore, resolution);

    CHECK("restore one drink leaves three doses", restore.dose == 3);
    CHECK("restore one drink updates raw id to three-dose",
          restore.raw_osrs_id == 3026);
    return 0;
}

typedef struct {
    int calls;
    OsrsConsumableKind kind;
} TestDrinkEffect;

static void test_record_drink_effect(void* ctx, OsrsConsumableKind kind) {
    TestDrinkEffect* effect = (TestDrinkEffect*)ctx;
    effect->calls++;
    effect->kind = kind;
}

static int test_shared_drink_consume_owns_timer_and_one_dose(void) {
    OsrsInventoryCell restore = osrs_inventory_cell_from_raw_osrs_id(3024);
    OsrsInventoryClickResolution resolution =
        osrs_inventory_cell_click_interpret(&restore, OSRS_CLICK_TICK_FIRST);
    int potion_timer = 0;
    TestDrinkEffect effect = {0};
    OsrsInventoryDrinkConsumeResult first =
        osrs_inventory_cell_consume_drink_one_dose(
            &restore,
            resolution,
            &potion_timer,
            test_record_drink_effect,
            &effect);

    CHECK("shared consume accepts first drink", first.consumed == 1);
    CHECK("shared consume decrements one dose",
          restore.dose == 3 && restore.raw_osrs_id == 3026);
    CHECK("shared consume arms potion timer", potion_timer == 3);
    CHECK("shared consume applies one effect",
          effect.calls == 1 && effect.kind == OSRS_CONSUMABLE_SUPER_RESTORE);

    resolution =
        osrs_inventory_cell_click_interpret(&restore, OSRS_CLICK_TICK_FIRST);
    OsrsInventoryDrinkConsumeResult gated =
        osrs_inventory_cell_consume_drink_one_dose(
            &restore,
            resolution,
            &potion_timer,
            test_record_drink_effect,
            &effect);
    CHECK("shared consume blocks live timer", gated.consumed == 0);
    CHECK("timer-gated shared consume leaves cell intact",
          restore.dose == 3 && restore.raw_osrs_id == 3026);
    CHECK("timer-gated shared consume skips effect", effect.calls == 1);
    return 0;
}

static int test_cell_rearrange_swaps_two_slots(void) {
    OsrsInventoryCell cells[OSRS_INVENTORY_SIZE];
    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++)
        cells[i] = osrs_inventory_cell_empty();
    cells[1] = osrs_inventory_cell_from_item(ITEM_TWISTED_BOW);
    cells[9] = osrs_inventory_cell_from_raw_osrs_id(6685);

    osrs_inventory_swap_cells(cells, 1, 9);

    CHECK("swap moves bow to target slot",
          cells[9].item_idx == ITEM_TWISTED_BOW);
    CHECK("swap moves brew to source slot",
          cells[1].raw_osrs_id == 6685 && cells[1].dose == 4);
    return 0;
}

static int cell_in_unit_range(const float* out) {
    for (int i = 0; i < OSRS_INVENTORY_CELL_OBS_FEATURES; i++) {
        if (out[i] < -1.0f || out[i] > 1.0f) return 0;
    }
    return 1;
}

static int test_enriched_feature_counts(void) {
    CHECK("cell obs features is 28", OSRS_INVENTORY_CELL_OBS_FEATURES == 28);
    CHECK("equipped obs features is 18", OSRS_EQUIPPED_SELF_OBS_FEATURES == 18);
    return 0;
}

static int test_brew_cell_semantics(void) {
    float out[OSRS_INVENTORY_CELL_OBS_FEATURES];
    float zero_deltas[6] = {0};

    osrs_write_inventory_cell_affordance_features(
        out, ITEM_NONE, 6685, 4, 0, zero_deltas, 99, 99, 99);
    CHECK("brew is not armor", out[12] == 0.0f);
    CHECK("brew is not weapon", out[13] == 0.0f);
    CHECK("brew kind is brew", out[14] == 1.0f);

    CHECK("brew hp-heal norm ~= 16/99",
          fabsf(out[19] - (16.0f / 99.0f)) < 1e-4f);
    CHECK("brew has no prayer restore", out[20] == 0.0f);
    CHECK("brew weapon speed/range are zero", out[26] == 0.0f && out[27] == 0.0f);
    CHECK("brew cell within [-1,1]", cell_in_unit_range(out));
    return 0;
}

static int test_weapon_cell_semantics(void) {
    float out[OSRS_INVENTORY_CELL_OBS_FEATURES];
    float zero_deltas[6] = {0};

    uint16_t fang_raw = ITEM_DATABASE[ITEM_OSMUMTENS_FANG].item_id;
    osrs_write_inventory_cell_affordance_features(
        out, ITEM_OSMUMTENS_FANG, fang_raw, 0, 0, zero_deltas, 99, 99, 99);
    CHECK("fang is weapon", out[13] == 1.0f);
    CHECK("fang is not armor", out[12] == 0.0f);
    CHECK("fang is not consumable", out[14] == 0.0f && out[15] == 0.0f &&
          out[16] == 0.0f && out[17] == 0.0f && out[18] == 0.0f);
    CHECK("fang weapon speed norm > 0", out[26] > 0.0f);
    CHECK("fang weapon range norm > 0", out[27] > 0.0f);
    CHECK("fang effect class is damage amp", out[23] == 1.0f);
    CHECK("fang is not lifesteal/defensive/util",
          out[22] == 0.0f && out[24] == 0.0f && out[25] == 0.0f);
    CHECK("fang cell within [-1,1]", cell_in_unit_range(out));
    return 0;
}

static int test_effect_class4_decoder(void) {
    float eff4[4];
    osrs_item_effect_class4(OSRS_ITEM_EFFECT_BLOOD_FURY, eff4);
    CHECK("blood fury is lifesteal only",
          eff4[0] == 1.0f && eff4[1] == 0.0f && eff4[2] == 0.0f && eff4[3] == 0.0f);
    osrs_item_effect_class4(OSRS_ITEM_EFFECT_TWISTED_BOW, eff4);
    CHECK("twisted bow is damage amp only",
          eff4[0] == 0.0f && eff4[1] == 1.0f && eff4[2] == 0.0f && eff4[3] == 0.0f);
    osrs_item_effect_class4(OSRS_ITEM_EFFECT_VENATOR_BOUNCE, eff4);
    CHECK("venator bounce is damage amp only",
          eff4[0] == 0.0f && eff4[1] == 1.0f && eff4[2] == 0.0f && eff4[3] == 0.0f);
    osrs_item_effect_class4(OSRS_ITEM_EFFECT_LIGHTBEARER, eff4);
    CHECK("lightbearer is util only",
          eff4[0] == 0.0f && eff4[1] == 0.0f && eff4[2] == 0.0f && eff4[3] == 1.0f);

    float eq[OSRS_EQUIPPED_SELF_OBS_FEATURES];
    osrs_write_equipped_self_features(eq, ITEM_AMULET_OF_BLOOD_FURY);
    CHECK("equipped blood fury lifesteal", eq[12] == 1.0f);
    CHECK("equipped blood fury no other class",
          eq[13] == 0.0f && eq[14] == 0.0f && eq[15] == 0.0f);
    return 0;
}

static int test_kind6_totality(void) {
    CHECK("none -> none", col_consumable_kind6(OSRS_CONSUMABLE_NONE) == COL_CKIND6_NONE);
    CHECK("brew -> brew", col_consumable_kind6(OSRS_CONSUMABLE_BREW) == COL_CKIND6_BREW);
    CHECK("super restore -> restore",
          col_consumable_kind6(OSRS_CONSUMABLE_SUPER_RESTORE) == COL_CKIND6_RESTORE);
    CHECK("sanfew -> restore",
          col_consumable_kind6(OSRS_CONSUMABLE_SANFEW) == COL_CKIND6_RESTORE);
    CHECK("super combat -> combat boost",
          col_consumable_kind6(OSRS_CONSUMABLE_SUPER_COMBAT) == COL_CKIND6_COMBAT_BOOST);
    CHECK("divine combat -> combat boost",
          col_consumable_kind6(OSRS_CONSUMABLE_DIVINE_COMBAT) == COL_CKIND6_COMBAT_BOOST);
    CHECK("ranging -> ranged boost",
          col_consumable_kind6(OSRS_CONSUMABLE_RANGING) == COL_CKIND6_RANGED_BOOST);
    CHECK("divine ranging -> ranged boost",
          col_consumable_kind6(OSRS_CONSUMABLE_DIVINE_RANGING) == COL_CKIND6_RANGED_BOOST);
    CHECK("surge -> special",
          col_consumable_kind6(OSRS_CONSUMABLE_SURGE) == COL_CKIND6_SPECIAL);
    CHECK("guthix rest -> special",
          col_consumable_kind6(OSRS_CONSUMABLE_GUTHIX_REST) == COL_CKIND6_SPECIAL);
    CHECK("saturated heart -> special",
          col_consumable_kind6(OSRS_CONSUMABLE_SATURATED_HEART) == COL_CKIND6_SPECIAL);
    CHECK("anti-venom+ -> special",
          col_consumable_kind6(OSRS_CONSUMABLE_ANTIVENOM_PLUS) == COL_CKIND6_SPECIAL);
    CHECK("shark -> food",
          col_consumable_kind6(OSRS_CONSUMABLE_SHARK_FOOD) == COL_CKIND6_FOOD);
    CHECK("karambwan -> food",
          col_consumable_kind6(OSRS_CONSUMABLE_KARAMBWAN) == COL_CKIND6_FOOD);
    return 0;
}

static int test_empty_cell_clamp(void) {
    float out[OSRS_INVENTORY_CELL_OBS_FEATURES];
    float zero_deltas[6] = {0};
    osrs_write_inventory_cell_affordance_features(
        out, ITEM_NONE, 0, 0, 0, zero_deltas, 99, 99, 99);
    CHECK("empty cell not present", out[0] == 0.0f);
    CHECK("empty cell within [-1,1]", cell_in_unit_range(out));
    return 0;
}

int main(void) {
    if (test_item_index_classification()) return 1;
    if (test_raw_consumable_classification()) return 1;
    if (test_sara_brew_dose_variants()) return 1;
    if (test_dose_after_drink()) return 1;
    if (test_pure_click_interpreter()) return 1;
    if (test_cell_click_attributes_to_slot_item()) return 1;
    if (test_cell_drink_decrements_one_dose()) return 1;
    if (test_shared_drink_consume_owns_timer_and_one_dose()) return 1;
    if (test_cell_rearrange_swaps_two_slots()) return 1;
    if (test_enriched_feature_counts()) return 1;
    if (test_brew_cell_semantics()) return 1;
    if (test_weapon_cell_semantics()) return 1;
    if (test_effect_class4_decoder()) return 1;
    if (test_kind6_totality()) return 1;
    if (test_empty_cell_clamp()) return 1;

    printf("test_osrs_inventory_clicks: OK\n");
    return 0;
}
