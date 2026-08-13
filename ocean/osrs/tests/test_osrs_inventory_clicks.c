#include <stdio.h>
#include <math.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>


#include "ocean/osrs/osrs_policy.h"



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

    return 0;
}
static int operation_aborts(void (*operation)(void)) {
    fflush(NULL);
    pid_t pid = fork();
    if (pid == 0) {
        operation();
        _exit(0);
    }
    int status = 0;
    if (pid < 0 || waitpid(pid, &status, 0) != pid) return 0;
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

static void lookup_unknown_raw_id(void) {
    (void)osrs_inventory_cell_from_raw_osrs_id(9999);
}

static void lookup_invalid_content_code(void) {
    (void)osrs_item_content_metadata(OSRS_ITEM_CONTENT_COUNT);
}

static void construct_invalid_content_code(void) {
    (void)osrs_inventory_cell_from_content_code(UINT16_MAX);
}
static void lookup_invalid_item_index(void) {
    (void)osrs_inventory_content_code_from_item(NUM_ITEMS);
}

static void lookup_invalid_consumable_kind_none(void) {
    (void)osrs_inventory_content_code_from_consumable(
        OSRS_CONSUMABLE_NONE, 4);
}

static void lookup_invalid_consumable_kind_count(void) {
    (void)osrs_inventory_content_code_from_consumable(
        OSRS_CONSUMABLE_COUNT, 4);
}

static void lookup_invalid_consumable_zero_dose(void) {
    (void)osrs_inventory_content_code_from_consumable(
        OSRS_CONSUMABLE_BREW, 0);
}

static void lookup_invalid_consumable_high_dose(void) {
    (void)osrs_inventory_content_code_from_consumable(
        OSRS_CONSUMABLE_BREW, 5);
}

static void lookup_invalid_gear_slot_item(void) {
    (void)osrs_item_gear_slot(NUM_ITEMS);
}


static int test_invalid_content_aborts(void) {
    CHECK("unknown raw OSRS id aborts", operation_aborts(lookup_unknown_raw_id));
    CHECK("invalid metadata code aborts", operation_aborts(lookup_invalid_content_code));
    CHECK("invalid cell code aborts", operation_aborts(construct_invalid_content_code));
    CHECK("invalid item index aborts",
        operation_aborts(lookup_invalid_item_index));
    CHECK("none consumable kind aborts",
        operation_aborts(lookup_invalid_consumable_kind_none));
    CHECK("sentinel consumable kind aborts",
        operation_aborts(lookup_invalid_consumable_kind_count));
    CHECK("zero consumable dose aborts",
        operation_aborts(lookup_invalid_consumable_zero_dose));
    CHECK("high consumable dose aborts",
        operation_aborts(lookup_invalid_consumable_high_dose));
    CHECK("invalid gear-slot item aborts",
        operation_aborts(lookup_invalid_gear_slot_item));
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
    OsrsInventoryCell gear_cell =
        osrs_inventory_cell_from_item(ITEM_VENATOR_BOW);
    OsrsInventoryClickResolution gear =
        osrs_inventory_cell_click_interpret(
            &gear_cell, OSRS_CLICK_TICK_FIRST);
    CHECK("gear click resolves equip", gear.click_action == OSRS_CLICK_EQUIP);
    CHECK("gear click has no consumable kind",
          gear.consumable_kind == OSRS_CONSUMABLE_NONE);

    OsrsInventoryCell brew_cell =
        osrs_inventory_cell_from_raw_osrs_id(6685);
    OsrsInventoryClickResolution brew =
        osrs_inventory_cell_click_interpret(
            &brew_cell, OSRS_CLICK_TICK_FIRST);
    CHECK("brew click resolves drink", brew.click_action == OSRS_CLICK_DRINK);
    CHECK("brew interpreter kind", brew.consumable_kind == OSRS_CONSUMABLE_BREW);
    CHECK("brew interpreter dose", brew.dose_count == 4);
    CHECK("brew interpreter next raw id", brew.raw_osrs_id_after_drink == 6687);

    OsrsInventoryCell shark_cell =
        osrs_inventory_cell_from_raw_osrs_id(385);
    OsrsInventoryClickResolution shark =
        osrs_inventory_cell_click_interpret(
            &shark_cell, OSRS_CLICK_TICK_FIRST);
    CHECK("shark click resolves eat", shark.click_action == OSRS_CLICK_EAT);
    CHECK("shark interpreter kind",
          shark.consumable_kind == OSRS_CONSUMABLE_SHARK_FOOD);

    OsrsInventoryClickResolution duplicate =
        osrs_inventory_cell_click_interpret(
            &brew_cell, OSRS_CLICK_TICK_DUPLICATE);
    CHECK("duplicate click noops", duplicate.click_action == OSRS_CLICK_NONE);
    return 0;
}

static int test_cell_click_classification(void) {
    OsrsInventoryCell brew = osrs_inventory_cell_from_raw_osrs_id(6685);
    OsrsInventoryClickResolution resolution =
        osrs_inventory_cell_click_classify(&brew);
    CHECK("brew classification resolves drink",
          resolution.click_action == OSRS_CLICK_DRINK);
    CHECK("brew classification preserves kind",
          resolution.consumable_kind == OSRS_CONSUMABLE_BREW);
    CHECK("brew classification preserves dose", resolution.dose_count == 4);
    CHECK("classification skips post-drink mutation",
          resolution.raw_osrs_id_after_drink == 0);
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

    CHECK("restore one drink leaves three doses",
        osrs_inventory_cell_dose_count(&restore) == 3);
    CHECK("restore one drink updates raw id to three-dose",
        osrs_inventory_cell_raw_osrs_id(&restore) == 3026);
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
        osrs_inventory_cell_dose_count(&restore) == 3 &&
        osrs_inventory_cell_raw_osrs_id(&restore) == 3026);
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
        osrs_inventory_cell_dose_count(&restore) == 3 &&
        osrs_inventory_cell_raw_osrs_id(&restore) == 3026);
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
        osrs_inventory_cell_item_index(&cells[9]) == ITEM_TWISTED_BOW);
    CHECK("swap moves brew to source slot",
        osrs_inventory_cell_raw_osrs_id(&cells[1]) == 6685 &&
        osrs_inventory_cell_dose_count(&cells[1]) == 4);
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
    return 0;
}

static int test_shared_policy_contract(void) {
    CHECK("shared self observation width is 52",
        OSRS_SHARED_SELF_OBS_SIZE == 52);
    CHECK("shared inventory cells carry one canonical content code",
        OSRS_SHARED_INVENTORY_CELL_OBS_FEATURES == 1);
    CHECK("shared inventory observation covers 28 cells",
        OSRS_SHARED_INVENTORY_OBS_SIZE == 28);
    CHECK("shared equipment observation covers eleven worn slots",
        OSRS_SHARED_EQUIPPED_OBS_SIZE == NUM_GEAR_SLOTS);
    CHECK("shared equipment effects retain their aggregate",
        OSRS_SHARED_EFFECT_OBS_SIZE == 10);
    CHECK("shared observation prefix is 101",
        OSRS_SHARED_OBS_SIZE == 101);
    CHECK("shared base action contract has 18 heads",
        OSRS_BASE_NUM_ACTION_HEADS == 18);
    CHECK("shared action heads have stable semantic order",
        OSRS_HEAD_PRIMARY == 0 &&
        OSRS_HEAD_OVERHEAD == 1 &&
        OSRS_HEAD_EQUIP_BASE == 2 &&
        OSRS_HEAD_EAT == 13 &&
        OSRS_HEAD_DRINK == 14 &&
        OSRS_HEAD_SPELL == 15 &&
        OSRS_HEAD_SPECIAL == 16 &&
        OSRS_HEAD_OFFENSIVE == 17);
    return 0;
}

static int test_shared_action_layout(void) {
    const int target_slots = 14;
    CHECK("primary combines movement and targets",
        OSRS_PRIMARY_DIM(target_slots) == 39);
    CHECK("overhead follows primary",
        osrs_base_action_head_mask_offset(target_slots, OSRS_HEAD_OVERHEAD) == 39);
    CHECK("equipment heads follow overhead",
        osrs_base_action_head_mask_offset(target_slots, OSRS_HEAD_EQUIP_BASE) == 46);
    CHECK("eat follows eleven equipment heads",
        osrs_base_action_head_mask_offset(target_slots, OSRS_HEAD_EAT) ==
            46 + NUM_GEAR_SLOTS * OSRS_INVENTORY_CLICK_DIM);
    CHECK("spell follows eat and drink",
        osrs_base_action_head_mask_offset(target_slots, OSRS_HEAD_SPELL) ==
            46 + (NUM_GEAR_SLOTS + 2) * OSRS_INVENTORY_CLICK_DIM);
    CHECK("base mask size matches final head boundary",
        OSRS_BASE_ACTION_MASK_SIZE(target_slots) ==
            osrs_base_action_head_mask_offset(target_slots, OSRS_HEAD_OFFENSIVE) +
                OSRS_OFFENSIVE_DIM);
    return 0;
}

static int test_shared_observation_layout(void) {
    Player player = {0};
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++)
        player.equipped[slot] = ITEM_NONE;
    player.base_hitpoints = 99;
    player.current_hitpoints = 66;
    player.base_prayer = 77;
    player.current_prayer = 55;
    player.x = 12;
    player.y = 23;
    player.equipped[GEAR_SLOT_WEAPON] = ITEM_WHIP;
    player.inventory_cells[7] =
        osrs_inventory_cell_from_item(ITEM_TWISTED_BOW);
    player.equipment_effect_profile.effect_mask =
        OSRS_ITEM_EFFECT_TWISTED_BOW;
    player.equipment_effect_profile.shield_item = ITEM_NONE;

    CHECK("real equip transition succeeds",
        osrs_equip_from_cell(&player, player.inventory_cells, 7) ==
            GEAR_SLOT_WEAPON);
    CHECK("equipped item leaves the inventory",
        osrs_inventory_cell_item_index(&player.inventory_cells[7]) == ITEM_WHIP);
    CHECK("clicked item occupies the worn weapon slot",
        player.equipped[GEAR_SLOT_WEAPON] == ITEM_TWISTED_BOW);

    float obs[OSRS_SHARED_OBS_SIZE] = {0};
    OsrsSharedObservationInput input = {
        .player = &player,
        .interaction = &player.interaction,
        .arena_min_x = 10,
        .arena_max_x = 20,
        .arena_min_y = 20,
        .arena_max_y = 30,
        .attack_style = ATTACK_STYLE_RANGED,
    };
    CHECK("shared writer returns exact width",
        osrs_write_shared_observations(obs, &input) == OSRS_SHARED_OBS_SIZE);
    CHECK("shared self prefix begins with hitpoints",
        fabsf(obs[0] - 66.0f / 99.0f) < 1e-6f);
    int cell_offset = OSRS_SHARED_OBS_INVENTORY_START + 7;
    CHECK("inventory observation contains the displaced item",
        obs[cell_offset] == osrs_inventory_cell_obs_code_encode(
            player.inventory_cells[7].content_code));
    int weapon_offset =
        OSRS_SHARED_OBS_EQUIPPED_START + GEAR_SLOT_WEAPON;
    CHECK("equipment observation contains the worn item",
        obs[weapon_offset] == osrs_inventory_cell_obs_code_encode(
            osrs_inventory_content_code_from_item(ITEM_TWISTED_BOW)));
    CHECK("shared equipment effects follow worn item codes",
        obs[OSRS_SHARED_OBS_EFFECT_START + 1] == 1.0f);
    return 0;
}

static int test_player_owns_canonical_inventory(void) {
    Player player = {0};
    OsrsInventoryCell* cells = osrs_player_inventory_cells(&player);
    cells[7] = osrs_inventory_cell_from_item(ITEM_TWISTED_BOW);
    CHECK("player owns exactly 28 canonical inventory cells",
        sizeof(player.inventory_cells) / sizeof(player.inventory_cells[0]) ==
            OSRS_INVENTORY_SIZE);
    CHECK("canonical inventory accessor returns player storage",
        &player.inventory_cells[7] == &cells[7]);
    CHECK("canonical inventory stores content identity",
        osrs_inventory_cell_item_index(&player.inventory_cells[7]) ==
            ITEM_TWISTED_BOW);
    return 0;
}

static int test_brew_cell_semantics(void) {
    float out[OSRS_INVENTORY_CELL_OBS_FEATURES];
    float zero_deltas[6] = {0};
    OsrsInventoryCell brew = osrs_inventory_cell_from_raw_osrs_id(6685);

    osrs_write_inventory_cell_affordance_features(
        out, &brew, 0, zero_deltas, 99, 99, 99);
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
    OsrsInventoryCell fang =
        osrs_inventory_cell_from_item(ITEM_OSMUMTENS_FANG);

    osrs_write_inventory_cell_affordance_features(
        out, &fang, 0, zero_deltas, 99, 99, 99);
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
    OsrsInventoryCell empty = osrs_inventory_cell_empty();
    osrs_write_inventory_cell_affordance_features(
        out, &empty, 0, zero_deltas, 99, 99, 99);
    CHECK("empty cell not present", out[0] == 0.0f);
    CHECK("empty cell within [-1,1]", cell_in_unit_range(out));
    return 0;
}
static int test_exhaustive_content_metadata(void) {
    for (uint16_t code = 0; code < OSRS_ITEM_CONTENT_COUNT; code++) {
        const OsrsItemContentMetadata* metadata =
            osrs_item_content_metadata(code);
        OsrsInventoryCell cell = osrs_inventory_cell_from_content_code(code);
        OsrsInventoryClickResolution classified =
            osrs_inventory_cell_click_classify(&cell);
        CHECK("metadata click action matches classification",
            classified.click_action == metadata->click_action);
        CHECK("metadata consumable kind matches classification",
            classified.consumable_kind == metadata->consumable_kind);
        CHECK("metadata dose matches classification",
            classified.dose_count == metadata->dose_count);

        OsrsInventoryClickResolution interpreted =
            osrs_inventory_cell_click_interpret(&cell, OSRS_CLICK_TICK_FIRST);
        uint16_t expected_next_raw = metadata->next_content_code == 0
            ? 0
            : osrs_item_content_metadata(
                metadata->next_content_code)->raw_osrs_id;
        CHECK("metadata next dose matches click interpretation",
            interpreted.raw_osrs_id_after_drink == expected_next_raw);
        if (metadata->click_action == OSRS_CLICK_DRINK) {
            if (metadata->dose_count == 1) {
                CHECK("one-dose drink transitions to empty",
                    metadata->next_content_code == 0);
            } else {
                const OsrsItemContentMetadata* next =
                    osrs_item_content_metadata(metadata->next_content_code);
                CHECK("drink transition preserves consumable kind",
                    next->consumable_kind == metadata->consumable_kind);
                CHECK("drink transition decrements exactly one dose",
                    next->dose_count + 1 == metadata->dose_count);
            }
        } else {
            CHECK("non-drink content has no dose transition",
                metadata->next_content_code == 0);
        }

        float expected[OSRS_INVENTORY_CELL_OBS_FEATURES_COMPACT];
        osrs_write_item_content_affordance_features_compact(
            expected,
            metadata,
            0,
            OSRS_ITEM_OBS_TABLE_BASE_HITPOINTS,
            OSRS_ITEM_OBS_TABLE_BASE_PRAYER,
            OSRS_ITEM_OBS_TABLE_BASE_RANGED);
        for (int feature = 0;
                feature < OSRS_INVENTORY_CELL_OBS_FEATURES_COMPACT;
                feature++) {
            CHECK("generated observation row remains exact",
                metadata->observation_row[feature] == expected[feature]);
        }
    }
    return 0;
}


int main(void) {
    if (test_item_index_classification()) return 1;
    if (test_raw_consumable_classification()) return 1;
    if (test_invalid_content_aborts()) return 1;
    if (test_sara_brew_dose_variants()) return 1;
    if (test_dose_after_drink()) return 1;
    if (test_pure_click_interpreter()) return 1;
    if (test_cell_click_classification()) return 1;
    if (test_cell_click_attributes_to_slot_item()) return 1;
    if (test_cell_drink_decrements_one_dose()) return 1;
    if (test_shared_drink_consume_owns_timer_and_one_dose()) return 1;
    if (test_cell_rearrange_swaps_two_slots()) return 1;
    if (test_enriched_feature_counts()) return 1;
    if (test_shared_policy_contract()) return 1;
    if (test_player_owns_canonical_inventory()) return 1;
    if (test_shared_action_layout()) return 1;
    if (test_shared_observation_layout()) return 1;
    if (test_brew_cell_semantics()) return 1;
    if (test_weapon_cell_semantics()) return 1;
    if (test_effect_class4_decoder()) return 1;
    if (test_kind6_totality()) return 1;
    if (test_empty_cell_clamp()) return 1;
    if (test_exhaustive_content_metadata()) return 1;

    printf("test_osrs_inventory_clicks: OK (%d content codes, 9 abort boundaries)\n",
        OSRS_ITEM_CONTENT_COUNT);
    return 0;
}
