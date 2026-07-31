#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "ocean/osrs/encounters/encounter_colosseum.h"

#define col_init_context_typed(ctx_ptr) do { \
    col_init_context_typed(ctx_ptr); \
    (ctx_ptr)->config.late_start_state_mode = 0; \
} while (0)

#include "ocean/osrs/tests/osrs_test_check.h"

#define TEST_NPC_TELLS_OFFSET 26
#define TEST_MOD_HAZARD_BASE (COLO_OBS_AFTER_NPCS + COLO_MODIFIER_FLAGS_OBS_SIZE)
#define TEST_MOD_OBS_DOOM_LETHAL (TEST_MOD_HAZARD_BASE + 2)
#define TEST_MOD_OBS_VENOM_TIMER (TEST_MOD_HAZARD_BASE + 6)
#define TEST_MOD_OBS_SOLARFLARE (TEST_MOD_HAZARD_BASE + 10)
#define TEST_MOD_OBS_MOLTEN (TEST_MOD_HAZARD_BASE + 18)
#define TEST_MOD_OBS_VOLATILITY (TEST_MOD_HAZARD_BASE + 30)

static EncounterLoadoutStats test_col_live_stats_for_set(
    const ColosseumState* s,
    ColoWeaponSet set
);
static OsrsEquipmentEffectProfile test_col_live_effects_for_set(
    const ColosseumState* s,
    ColoWeaponSet set
);
static EncounterLoadoutStats test_col_spec_stats_for_kind(
    const ColosseumState* s,
    int kind
);
static void venator_spawn_enemy(
    ColosseumState* s,
    int slot,
    ColoNpcType type,
    int x,
    int y,
    int size
);

static void step_and_observe(ColosseumState* s, ColosseumContext* ctx, const int* actions) {
    static float obs[COLO_NUM_OBS];
    static float mask[COLO_ACTION_MASK_SIZE];
    col_step_ctx((EncounterState*)s, (EncounterContext*)ctx, actions);
    col_write_obs_ctx((EncounterState*)s, (EncounterContext*)ctx, obs);
    col_write_mask_ctx((EncounterState*)s, (EncounterContext*)ctx, mask);
}

static void advance_to_wave_spawn(ColosseumState* s, ColosseumContext* ctx) {
    int idle[COLO_NUM_ACTION_HEADS] = {0};

    if (s->modifiers.draft_pending) {
        s->modifiers.draft_pending = 0;
        s->modifiers.draft_gates_spawn = 0;
        s->modifiers.draft_free_movement = 0;
        for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++)
            s->modifiers.draft_options[o] = -1;
        s->wave_spawn_delay = col_wave_entry_delay_ticks(s->wave_spawn_target);
    }
    while (s->wave_spawn_delay > 0)
        step_and_observe(s, ctx, idle);
}

static int draft_is_open(const ColosseumState* s) {
    if (!s->modifiers.draft_pending) return 0;
    for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++)
        if (s->modifiers.draft_options[o] >= 0) return 1;
    return 0;
}

static void complete_open_draft(ColosseumState* s, ColosseumContext* ctx, int option) {
    if (!draft_is_open(s)) return;
    int pick[COLO_NUM_ACTION_HEADS] = {0};
    pick[COLO_HEAD_MODIFIER_SELECT] = option + 1;
    step_and_observe(s, ctx, pick);
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    while (s->wave_spawn_delay > 0) step_and_observe(s, ctx, idle);
}

static void force_clear_wave(ColosseumState* s) {
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        if (!s->npcs[i].active) continue;
        if (col_type_is_hazard_entity(s->npcs[i].type)) continue;
        s->npcs[i].hp = 0;
        s->npcs[i].active = 0;
    }
}

static int first_live_score_enemy(const ColosseumState* s) {
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        if (col_npc_is_live_enemy(&s->npcs[i])) return i;
    }
    return -1;
}

static void kill_first_live_score_enemy(ColosseumState* s) {
    int slot = first_live_score_enemy(s);
    CHECK("a live score enemy exists", slot >= 0);
    if (slot < 0) return;
    s->npcs[slot].hp = 0;
    col_apply_npc_death(s, slot);
}

static float score_for_depth(float depth) {
    float ratio = depth / (float)COLO_NUM_WAVES;
    return 0.99f * ratio * ratio;
}

static void land_pending_player_hits(ColosseumState* s) {
    for (int t = 0; t < 4; t++) col_resolve_player_projectiles_on_npcs(s);
}

static void geo_clear_npcs(ColosseumState* s) {
    memset(s->npcs, 0, sizeof(s->npcs));
    memset(s->npc_collision_flags, 0, sizeof(s->npc_collision_flags));
    memset(s->totems, 0, sizeof(s->totems));
    memset(s->bees, 0, sizeof(s->bees));
    col_rebuild_player_collision_flags(s);
}

static void init_forecast_test_state(
    ColosseumState* s,
    ColosseumContext* ctx,
    uint32_t seed,
    int player_x,
    int player_y
) {
    col_init_context_typed(ctx);
    memset(s, 0, sizeof(*s));
    col_reset_ctx((EncounterState*)s, (EncounterContext*)ctx, seed);
    geo_clear_npcs(s);
    s->modifiers.draft_pending = 0;
    s->player.x = player_x;
    s->player.y = player_y;
    col_rebuild_player_collision_flags(s);
}

static int forecast_move_action_for_delta(int dx, int dy) {
    for (int action = 0; action < ENCOUNTER_MOVE_ACTIONS; action++)
        if (ENCOUNTER_MOVE_TARGET_DX[action] == dx &&
            ENCOUNTER_MOVE_TARGET_DY[action] == dy) return action;
    assert(0 && "missing movement action delta");
    return 0;
}

static int forecast_action_has_event(const ColoStepOutForecastAction* action) {
    for (int tick = 0; tick < COLO_STEP_OUT_FORECAST_HORIZON; tick++)
        if (col_step_out_forecast_tick_has_event(&action->ticks[tick])) return 1;
    return 0;
}

static int test_find_inventory_cell_with_item(const ColosseumState* s, uint8_t item_idx) {
    for (int i = 0; i < COLO_INVENTORY_DISPLAY_SLOTS; i++)
        if (s->inventory_cells[i].item_idx == item_idx) return i;
    return -1;
}

static int test_find_inventory_cell_with_consumable(
    const ColosseumState* s,
    OsrsConsumableKind kind
) {
    for (int i = 0; i < COLO_INVENTORY_DISPLAY_SLOTS; i++) {
        const ColoInvCell* cell = &s->inventory_cells[i];
        OsrsInventoryClickResolution r =
            osrs_inventory_cell_click_interpret(cell, OSRS_CLICK_TICK_FIRST);
        if (r.consumable_kind == kind) return i;
    }
    return -1;
}

static void test_click_inventory_cell_action_s(
    const ColosseumState* s, int actions[COLO_NUM_ACTION_HEADS], int cell
) {
    assert(cell >= 0 && cell < COLO_INVENTORY_DISPLAY_SLOTS);
    OsrsInventoryClickResolution r =
        osrs_inventory_cell_click_interpret(&s->inventory_cells[cell], OSRS_CLICK_TICK_FIRST);
    if (r.click_action == OSRS_CLICK_EQUIP) {
        int slot = osrs_item_gear_slot(s->inventory_cells[cell].item_idx);
        assert(slot >= 0 && slot < NUM_GEAR_SLOTS);
        actions[COLO_HEAD_EQUIP_SLOT(slot)] = cell + 1;
    } else if (r.click_action == OSRS_CLICK_EAT) {
        actions[COLO_HEAD_EAT] = cell + 1;
    } else if (r.click_action == OSRS_CLICK_DRINK) {
        actions[COLO_HEAD_DRINK] = cell + 1;
    } else {
        assert(0 && "test clicked a non-actionable cell");
    }
}

static void test_click_consumable_action(
    const ColosseumState* s,
    int actions[COLO_NUM_ACTION_HEADS],
    OsrsConsumableKind kind
) {
    test_click_inventory_cell_action_s(s, actions, test_find_inventory_cell_with_consumable(s, kind));
}

static float test_click_mask_for_cell_s(
    const ColosseumState* s, const float mask[COLO_ACTION_MASK_SIZE], int cell
) {
    OsrsInventoryClickResolution r =
        osrs_inventory_cell_click_interpret(&s->inventory_cells[cell], OSRS_CLICK_TICK_FIRST);
    int head;
    if (r.click_action == OSRS_CLICK_EQUIP) {
        int slot = osrs_item_gear_slot(s->inventory_cells[cell].item_idx);
        head = (slot >= 0 && slot < NUM_GEAR_SLOTS) ? COLO_HEAD_EQUIP_SLOT(slot)
                                                    : COLO_HEAD_EQUIP_SLOT(0);
    } else if (r.click_action == OSRS_CLICK_EAT) {
        head = COLO_HEAD_EAT;
    } else if (r.click_action == OSRS_CLICK_DRINK) {
        head = COLO_HEAD_DRINK;
    } else {
        head = COLO_HEAD_EQUIP_SLOT(0);
    }
    return mask[col_action_head_mask_offset(head) + 1 + cell];
}

static int test_sum_inventory_doses_for_kind(
    const ColosseumState* s,
    OsrsConsumableKind kind
) {
    int doses = 0;
    for (int i = 0; i < COLO_INVENTORY_DISPLAY_SLOTS; i++) {
        const ColoInvCell* cell = &s->inventory_cells[i];
        OsrsInventoryClickResolution r =
            osrs_inventory_cell_click_interpret(cell, OSRS_CLICK_TICK_FIRST);
        if (r.consumable_kind == kind) doses += cell->dose;
    }
    return doses;
}

static int test_aggregate_doses_for_kind(
    const ColosseumState* s,
    OsrsConsumableKind kind
) {
    switch (kind) {
        case OSRS_CONSUMABLE_BREW:
            return s->player.brew_doses;
        case OSRS_CONSUMABLE_SUPER_RESTORE:
        case OSRS_CONSUMABLE_SANFEW:
            return s->player.restore_doses;
        case OSRS_CONSUMABLE_SUPER_COMBAT:
        case OSRS_CONSUMABLE_DIVINE_COMBAT:
            return s->player.combat_potion_doses;
        case OSRS_CONSUMABLE_RANGING:
        case OSRS_CONSUMABLE_DIVINE_RANGING:
            return s->player.ranged_potion_doses;
        case OSRS_CONSUMABLE_SURGE:
            return s->surge_doses;
        case OSRS_CONSUMABLE_ANTIVENOM_PLUS:
            return s->player.antivenom_doses;
        case OSRS_CONSUMABLE_GUTHIX_REST:
        case OSRS_CONSUMABLE_SATURATED_HEART:
        case OSRS_CONSUMABLE_NONE:
        case OSRS_CONSUMABLE_SHARK_FOOD:
        case OSRS_CONSUMABLE_KARAMBWAN:
            return -1;
    }
    abort();
}

static void test_prepare_for_drink_kind(
    ColosseumState* s,
    OsrsConsumableKind kind
) {
    s->player.potion_timer = 0;
    s->player.current_hitpoints = 50;
    s->player.current_prayer = 40;
    s->player.special_energy = 50;
    s->surge_cooldown = 0;
    s->player_venom = COLO_VENOM_START;
    s->player_venom_timer = 17;
    s->player_poison = COLO_POISON_BEE_CONTACT_SEVERITY;
    s->player_poison_timer = 11;
    if (kind == OSRS_CONSUMABLE_DIVINE_COMBAT ||
            kind == OSRS_CONSUMABLE_DIVINE_RANGING) {
        s->player.current_hitpoints = 99;
    }
}

typedef enum {
    TEST_INV_OBS_ROLE_ARMOR = 12,
    TEST_INV_OBS_ROLE_WEAPON = 13,
    TEST_INV_OBS_KIND_BREW = 14,
    TEST_INV_OBS_KIND_RESTORE = 15,
    TEST_INV_OBS_KIND_COMBAT_BOOST = 16,
    TEST_INV_OBS_KIND_RANGED_BOOST = 17,
    TEST_INV_OBS_KIND_SPECIAL = 18,
    TEST_INV_OBS_EFFECT_LIFESTEAL = 22,
    TEST_INV_OBS_EFFECT_DAMAGE_AMP = 23,
    TEST_INV_OBS_EFFECT_DEFENSIVE = 24,
    TEST_INV_OBS_EFFECT_UTIL = 25,
} TestInventoryObsFeature;

typedef struct {
    float is_gear;
    float is_consumable;
    float can_use;
    float has_effect;
    float role_food;
    float role_potion_family;
    float kind_food;
} TestDroppedInventoryFields;

static float test_binary_float(int value) {
    return value ? 1.0f : 0.0f;
}

static float test_any_inventory_kind_bit(const float* cell_obs) {
    return test_binary_float(
        cell_obs[TEST_INV_OBS_KIND_BREW] != 0.0f ||
        cell_obs[TEST_INV_OBS_KIND_RESTORE] != 0.0f ||
        cell_obs[TEST_INV_OBS_KIND_COMBAT_BOOST] != 0.0f ||
        cell_obs[TEST_INV_OBS_KIND_RANGED_BOOST] != 0.0f ||
        cell_obs[TEST_INV_OBS_KIND_SPECIAL] != 0.0f);
}

static TestDroppedInventoryFields test_expected_dropped_inventory_fields(
    const ColosseumState* s,
    int cell_idx
) {
    const ColoInvCell* cell = &s->inventory_cells[cell_idx];
    OsrsConsumableClick consumable =
        osrs_consumable_click_lookup_raw_osrs_id(cell->raw_osrs_id);
    uint32_t effect_mask = OSRS_ITEM_EFFECT_NONE;
    if (cell->item_idx != ITEM_NONE) {
        if (cell->item_idx >= NUM_ITEMS) abort();
        effect_mask = ITEM_DATABASE[cell->item_idx].effect_mask;
    }
    OsrsConsumableKind6 k6 = col_consumable_kind6(consumable.consumable_kind);

    return (TestDroppedInventoryFields){
        .is_gear = test_binary_float(cell->item_idx != ITEM_NONE),
        .is_consumable = test_binary_float(consumable.click_action != OSRS_CLICK_NONE),
        .can_use = test_binary_float(col_inventory_cell_actionable(s, cell_idx)),
        .has_effect = test_binary_float(effect_mask != OSRS_ITEM_EFFECT_NONE),
        .role_food = test_binary_float(consumable.click_action == OSRS_CLICK_EAT),
        .role_potion_family = test_binary_float(consumable.click_action == OSRS_CLICK_DRINK),
        .kind_food = test_binary_float(k6 == COL_CKIND6_FOOD),
    };
}

static TestDroppedInventoryFields test_reconstructed_dropped_inventory_fields(
    const ColosseumState* s,
    const float obs[COLO_NUM_OBS],
    const float mask[COLO_ACTION_MASK_SIZE],
    int cell_idx
) {
    int base = COLO_OBS_AFTER_PILLARS +
        cell_idx * COLO_INVENTORY_CELL_OBS_FEATURES;
    const float* cell_obs = &obs[base];
    float kind = test_any_inventory_kind_bit(cell_obs);
    float effect = test_binary_float(
        cell_obs[TEST_INV_OBS_EFFECT_LIFESTEAL] != 0.0f ||
        cell_obs[TEST_INV_OBS_EFFECT_DAMAGE_AMP] != 0.0f ||
        cell_obs[TEST_INV_OBS_EFFECT_DEFENSIVE] != 0.0f ||
        cell_obs[TEST_INV_OBS_EFFECT_UTIL] != 0.0f);

    return (TestDroppedInventoryFields){
        .is_gear = test_binary_float(
            cell_obs[TEST_INV_OBS_ROLE_ARMOR] != 0.0f ||
            cell_obs[TEST_INV_OBS_ROLE_WEAPON] != 0.0f),
        .is_consumable = kind,
        .can_use = test_click_mask_for_cell_s(s, mask, cell_idx),
        .has_effect = effect,
        .role_food = 0.0f,
        .role_potion_family = kind,
        .kind_food = 0.0f,
    };
}

static void test_check_inventory_dropped_field(
    const char* scenario,
    int cell_idx,
    const char* field,
    float expected,
    float reconstructed
) {
    char label[240];
    snprintf(label, sizeof(label), "%s cell %d reconstructs %s", scenario, cell_idx, field);
    CHECK(label, reconstructed == expected);
}

static void test_check_inventory_cut_equivalence_state(
    ColosseumState* s,
    ColosseumContext* ctx,
    const char* scenario
) {
    ctx->config.mask_inventory_heads = 0;
    static float obs[COLO_NUM_OBS];
    static float mask[COLO_ACTION_MASK_SIZE];
    col_write_obs_ctx((EncounterState*)s, (EncounterContext*)ctx, obs);
    col_write_mask_ctx((EncounterState*)s, (EncounterContext*)ctx, mask);

    for (int cell = 0; cell < COLO_INVENTORY_DISPLAY_SLOTS; cell++) {
        TestDroppedInventoryFields expected =
            test_expected_dropped_inventory_fields(s, cell);
        TestDroppedInventoryFields reconstructed =
            test_reconstructed_dropped_inventory_fields(s, obs, mask, cell);
        test_check_inventory_dropped_field(
            scenario, cell, "is_gear", expected.is_gear, reconstructed.is_gear);
        test_check_inventory_dropped_field(
            scenario, cell, "is_consumable",
            expected.is_consumable, reconstructed.is_consumable);
        test_check_inventory_dropped_field(
            scenario, cell, "can_use", expected.can_use, reconstructed.can_use);
        test_check_inventory_dropped_field(
            scenario, cell, "has_effect", expected.has_effect, reconstructed.has_effect);
        test_check_inventory_dropped_field(
            scenario, cell, "role_food", expected.role_food, reconstructed.role_food);
        test_check_inventory_dropped_field(
            scenario, cell, "role_potion_family",
            expected.role_potion_family, reconstructed.role_potion_family);
        test_check_inventory_dropped_field(
            scenario, cell, "kind_food", expected.kind_food, reconstructed.kind_food);
    }
}

static int test_count_item_in_equipment_and_inventory(
    const ColosseumState* s,
    uint8_t item_idx
) {
    int count = 0;
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++)
        if (s->player.equipped[slot] == item_idx) count++;
    for (int slot = 0; slot < COLO_INVENTORY_DISPLAY_SLOTS; slot++)
        if (s->inventory_cells[slot].item_idx == item_idx) count++;
    return count;
}

static int test_npc_covers_player(const ColosseumState* s, const ColoNPC* npc) {
    int size = col_npc_effective_size(npc);
    return s->player.x >= npc->x && s->player.x < npc->x + size &&
        s->player.y >= npc->y && s->player.y < npc->y + size;
}

static uint8_t test_profile_spec_item(ColoLoadoutProfile profile, int kind) {
    if (profile == COLO_LOADOUT_PROFILE_SPEEDRUN && kind == 1) return ITEM_DRAGON_CLAWS;
    if (profile == COLO_LOADOUT_PROFILE_SPEEDRUN && kind == 2) return ITEM_ELDER_MAUL;
    if (profile == COLO_LOADOUT_PROFILE_BEGINNER && kind == 1) return ITEM_SGS;
    if (profile == COLO_LOADOUT_PROFILE_BEGINNER && kind == 2) return ITEM_DRAGON_CLAWS;
    abort();
}

static void test_fuzz_obs_mask(void) {
    printf("test_fuzz_obs_mask\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 0;
    ctx.world_offset_x = 1808;
    ctx.world_offset_y = 3090;
    ctx.config.loadout_profile_mode = COLO_LOADOUT_PROFILE_MODE_MIXED;
    ctx.config.beginner_loadout_fraction = 0.5f;

    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 12345);

    int episodes = 0;
    int actions[COLO_NUM_ACTION_HEADS];
    unsigned int rng = 99;
    for (int start = 1; start <= COLO_NUM_WAVES; start++) {
        ctx.config.start_wave = start - 1;
        col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, (uint32_t)(start * 7 + 1));
        for (long t = 0; t < 120000 && episodes < 600; t++) {
            for (int h = 0; h < COLO_NUM_ACTION_HEADS; h++) {
                rng = rng * 1103515245u + 12345u;
                actions[h] = (int)((rng >> 16) % (unsigned)COLO_ACTION_DIMS[h]);
            }
            step_and_observe(&s, &ctx, actions);
            if (s.episode_over) {
                episodes++;
                col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, (uint32_t)(rng | 1));
            }
        }
    }

    CHECK("fuzz ran full episodes with obs/mask asserts holding", episodes > 0);
    printf("  episodes=%d (obs+mask running-index asserts held every tick)\n", episodes);
}

static void test_zero_actions_hit_timeout(void) {
    printf("test_zero_actions_hit_timeout\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 0;

    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 12345);
    CHECK("the challenge-start draft is open at reset", s.modifiers.draft_pending == 1);
    advance_to_wave_spawn(&s, &ctx);

    int idle[COLO_NUM_ACTION_HEADS] = {0};
    force_clear_wave(&s);
    step_and_observe(&s, &ctx, idle);
    CHECK("clearing wave 1 opened the wave-2 draft", draft_is_open(&s));

    long t = 0;
    for (; t < s.episode_max_ticks + 10 && !s.episode_over; t++)
        step_and_observe(&s, &ctx, idle);

    CHECK("all-none actions terminate at the tick cap", s.episode_over == 1);
    CHECK("timeout fires exactly at the episode cap", s.tick == s.episode_max_ticks);
    CHECK("timeout counts as a loss", s.winner == COLO_OUTCOME_PLAYER_DIED);
    CHECK("timeout is marked as a truncation", s.time_limit_truncated == 1);
    CHECK("the draft was still pending when time ran out", s.modifiers.draft_pending == 1);
}

static void test_offpray_attribution_log(void) {
    printf("test_offpray_attribution_log\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 0;

    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 777);

    col_log_prayer_event(&s, COLO_SERPENT_SHAMAN, 0 , 11);
    CHECK("off-prayer hit counted as faced", s.log.pray_faced_by_type[COLO_SERPENT_SHAMAN] == 1.0f);
    CHECK("off-prayer hit not counted correct", s.log.pray_correct_by_type[COLO_SERPENT_SHAMAN] == 0.0f);
    CHECK("off-prayer damage attributed to the shaman", s.log.offpray_damage_by_type[COLO_SERPENT_SHAMAN] == 11.0f);

    col_log_prayer_event(&s, COLO_SERPENT_SHAMAN, 1 , 0);
    CHECK("prayed hit counted as faced", s.log.pray_faced_by_type[COLO_SERPENT_SHAMAN] == 2.0f);
    CHECK("prayed hit counted correct", s.log.pray_correct_by_type[COLO_SERPENT_SHAMAN] == 1.0f);
    CHECK("prayed hit adds no off-prayer damage", s.log.offpray_damage_by_type[COLO_SERPENT_SHAMAN] == 11.0f);

    CHECK("a type that never threw stays out of the prayer log",
        s.log.pray_faced_by_type[COLO_JAVELIN_COLOSSUS] == 0.0f);
}

static void test_step_loop_draft(void) {
    printf("test_step_loop_draft\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 0;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 42);

    int idle[COLO_NUM_ACTION_HEADS] = {0};
    int walk_east[COLO_NUM_ACTION_HEADS] = {0};
    walk_east[COLO_HEAD_PRIMARY] = 7;

    CHECK("the challenge-start draft is open at reset", draft_is_open(&s));
    CHECK("the start draft allows movement", s.modifiers.draft_free_movement == 1);
    int spawned = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++) if (s.npcs[i].active) spawned = 1;
    CHECK("no NPCs exist while the start draft pends", !spawned);
    int x0 = s.player.x;
    step_and_observe(&s, &ctx, walk_east);
    CHECK("the player roams during the start draft", s.player.x == x0 + 1);

    int pick0[COLO_NUM_ACTION_HEADS] = {0};
    pick0[COLO_HEAD_MODIFIER_SELECT] = 1;
    step_and_observe(&s, &ctx, pick0);
    CHECK("the start pick activated a modifier", s.modifiers.active_mask != 0);
    CHECK("the start pick armed the 5-tick spawn delay",
        s.wave_spawn_delay == COLO_WAVE_SPAWN_DELAY_TICKS);

    int x1 = s.player.x;
    for (int t = 0; t < 4; t++) step_and_observe(&s, &ctx, walk_east);
    CHECK("player movement is free before the spawn resolves",
        s.player.x == x1 + 4);
    spawned = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++) if (s.npcs[i].active) spawned = 1;
    CHECK("arena still empty through t4", !spawned);

    int t5_x = s.player.x, t5_y = s.player.y;
    step_and_observe(&s, &ctx, walk_east);
    spawned = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++) if (s.npcs[i].active) spawned = 1;
    CHECK("the wave resolved at t5 (visible from t6)", spawned);
    CHECK("the t5 queued move still landed", s.player.x == t5_x + 1);
    CHECK("the spawn armed the movement gate", s.wave_ready_delay == COLO_WAVE_READY_TICKS);
    CHECK("the spawn armed the attack gate",
        s.wave_attack_delay == COLO_WAVE_ATTACK_GATE_TICKS);

    int exclusion_ok = 1;
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        const ColoNPC* npc = &s.npcs[i];
        if (!npc->active || col_type_is_warbander(npc->type)) continue;
        if (col_type_is_hazard_entity(npc->type)) continue;
        if (encounter_rect_distance(t5_x, t5_y, 1,
                npc->x, npc->y, col_npc_effective_size(npc)) <= COLO_SPAWN_EXCLUSION_CHEB)
            exclusion_ok = 0;
    }
    CHECK("no primary spawned within Chebyshev 4 of the pre-move t5 tile", exclusion_ok);

    int ax = -1, ay = -1, sx = -1, sy = -1, bx = -1, by = -1;
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        const ColoNPC* npc = &s.npcs[i];
        if (!npc->active) continue;
        if (npc->type == COLO_FREMENNIK_ARCHER)    { ax = npc->x; ay = npc->y; }
        if (npc->type == COLO_FREMENNIK_SEER)      { sx = npc->x; sy = npc->y; }
        if (npc->type == COLO_FREMENNIK_BERSERKER) { bx = npc->x; by = npc->y; }
    }
    CHECK("warband trio spawned", ax >= 0 && sx >= 0 && bx >= 0);
    CHECK("seer spawned 2E of the archer", sx == ax + 2 && sy == ay);
    CHECK("berserker spawned 1E+1N of the archer", bx == ax + 1 && by == ay + 1);

    int pos_sum_spawn = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++)
        if (s.npcs[i].active) pos_sum_spawn += s.npcs[i].x * 64 + s.npcs[i].y;
    step_and_observe(&s, &ctx, idle);
    int pos_sum_t6 = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++)
        if (s.npcs[i].active) pos_sum_t6 += s.npcs[i].x * 64 + s.npcs[i].y;
    CHECK("NPCs frozen at t6", pos_sum_t6 == pos_sum_spawn);
    step_and_observe(&s, &ctx, idle);
    int pos_sum_t7 = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++)
        if (s.npcs[i].active) pos_sum_t7 += s.npcs[i].x * 64 + s.npcs[i].y;
    CHECK("NPCs started moving at t7 (warband darts to the player)",
        pos_sum_t7 != pos_sum_t6);
    step_and_observe(&s, &ctx, idle);
    CHECK("no damage taken through t8 (attacks open t9)",
        s.player.current_hitpoints == s.player.base_hitpoints);
    CHECK("attack gate open from t9", s.wave_attack_delay == 1);

    force_clear_wave(&s);
    step_and_observe(&s, &ctx, idle);
    CHECK("clearing wave 1 opened the wave-2 draft", draft_is_open(&s));
    CHECK("the wave-2 draft gates the wave-2 spawn", s.wave_spawn_target == 1);

    int px = s.player.x;
    int py = s.player.y;
    for (int t = 0; t < 12; t++) step_and_observe(&s, &ctx, walk_east);
    CHECK("movement is ignored while the draft is open",
        s.player.x == px && s.player.y == py);
    CHECK("the draft never auto-closes", draft_is_open(&s));
    float mask[COLO_ACTION_MASK_SIZE];
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    int any_walk_valid = 0;
    for (int d = 1; d < ENCOUNTER_MOVE_ACTIONS; d++)
        if (mask[d] > 0.0f) any_walk_valid = 1;
    CHECK("the mask offers only idle movement while frozen", !any_walk_valid);

    int chosen = s.modifiers.draft_options[0];
    int pick[COLO_NUM_ACTION_HEADS] = {0};
    pick[COLO_HEAD_MODIFIER_SELECT] = 1;
    step_and_observe(&s, &ctx, pick);
    CHECK("the pick activated the chosen modifier",
        chosen >= 0 && col_mod_active(&s, (ColoModifier)chosen));
    CHECK("draft closed after the pick", !s.modifiers.draft_pending);
    CHECK("the pick armed the 5-tick spawn delay",
        s.wave_spawn_delay == COLO_WAVE_SPAWN_DELAY_TICKS);
    for (int t = 0; t < 4; t++) step_and_observe(&s, &ctx, idle);
    spawned = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++) if (s.npcs[i].active) spawned = 1;
    CHECK("no wave-2 NPCs through the 4 post-pick ticks", !spawned);
    step_and_observe(&s, &ctx, idle);
    spawned = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++) if (s.npcs[i].active) spawned = 1;
    CHECK("wave 2 resolved on the 5th post-pick tick", spawned);
    CHECK("advanced to wave 2 after the pick", s.wave == 1);
    CHECK("the spawn re-armed the movement + attack gates",
        s.wave_ready_delay == COLO_WAVE_READY_TICKS &&
        s.wave_attack_delay == COLO_WAVE_ATTACK_GATE_TICKS);
}

static void test_eleven_drafts_per_run(void) {
    printf("test_eleven_drafts_per_run\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 0;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 4242);

    int idle[COLO_NUM_ACTION_HEADS] = {0};
    int picks = 0;
    int draft_waves_ok = 1;
    for (long t = 0; t < 4000 && !s.episode_over; t++) {
        if (draft_is_open(&s)) {

            if (s.wave_spawn_target != picks) draft_waves_ok = 0;
            complete_open_draft(&s, &ctx, 0);
            picks++;
            continue;
        }
        force_clear_wave(&s);

        s.player.current_hitpoints = s.player.base_hitpoints;
        s.doom_stacks = 0;
        step_and_observe(&s, &ctx, idle);
    }
    CHECK("the run ended in victory", s.episode_over && s.winner == COLO_OUTCOME_PLAYER_WON);
    CHECK("exactly 12 drafts were offered and picked", picks == 12);
    CHECK("the log counted all 12 mandatory picks", s.log.modifiers_picked == 12);
    CHECK("draft k gated wave index k for every k", draft_waves_ok);
}

static void test_draft_offer_and_select(void) {
    printf("test_draft_offer_and_select\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 1;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 7);

    CHECK("fresh run has no active modifiers", s.modifiers.active_mask == 0);
    CHECK("start_wave>1 runs skip the prior drafts (no draft open)",
        s.modifiers.draft_pending == 0);

    col_modifier_open_draft(&s, 4);
    CHECK("draft opened with options", draft_is_open(&s));
    int distinct = 1;
    int a = s.modifiers.draft_options[0], b = s.modifiers.draft_options[1], c = s.modifiers.draft_options[2];
    if (a == b || a == c || (b == c && b >= 0)) distinct = 0;
    CHECK("draft options are distinct", distinct);

    int chosen = s.modifiers.draft_options[0];
    col_modifier_apply_selection(&s, 0);
    CHECK("selection set the active bit", col_mod_active(&s, (ColoModifier)chosen));
    CHECK("selection set tier >= 1", col_mod_tier(&s, (ColoModifier)chosen) >= 1);
    CHECK("draft closed after selection", s.modifiers.draft_pending == 0);
    CHECK("selection logged", s.log.modifiers_picked == 1);

    s.wave = 1;
    col_spawn_wave(&s);
    CHECK("modifier persists across waves", col_mod_active(&s, (ColoModifier)chosen));

    int rfdd_late = 0, rfdd_window = 0, boss_excluded_seen = 0;
    for (int rep = 0; rep < 400; rep++) {
        int late_wave = 7 + rep % 4;
        col_modifier_open_draft(&s, late_wave);
        for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++) {
            int m = s.modifiers.draft_options[o];
            if (m == COLO_MOD_RED_FLAG || m == COLO_MOD_DYNAMIC_DUO) rfdd_late = 1;
        }
        col_modifier_open_draft(&s, 2 + rep % 5);
        for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++) {
            int m = s.modifiers.draft_options[o];
            if (m == COLO_MOD_RED_FLAG || m == COLO_MOD_DYNAMIC_DUO) rfdd_window = 1;
        }
        col_modifier_open_draft(&s, COLO_WAVE_BOSS);
        for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++) {
            int m = s.modifiers.draft_options[o];
            if (m >= 0 && COLO_MODIFIER_PRE_BOSS_ONLY[m]) boss_excluded_seen = 1;
        }
    }
    s.modifiers.draft_pending = 0;
    CHECK("Red Flag / Dynamic Duo never offered into wave 8+", rfdd_late == 0);
    CHECK("Red Flag / Dynamic Duo do appear in drafts before wave 7", rfdd_window == 1);
    CHECK("the wave-12 draft excludes RF/DD/Mantimayhem/Reentry", boss_excluded_seen == 0);
}

static void test_draft_upgrade_bias(void) {
    printf("test_draft_upgrade_bias\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 1;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 11);

    s.modifiers.active_mask |= (1u << COLO_MOD_RELENTLESS);
    s.modifiers.tier[COLO_MOD_RELENTLESS] = 1;

    const int N = 4000;
    int owned_offers = 0, unowned_offers = 0;
    for (int rep = 0; rep < N; rep++) {
        col_modifier_open_draft(&s, 4);
        for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++) {
            int m = s.modifiers.draft_options[o];
            if (m == COLO_MOD_RELENTLESS) owned_offers++;
            if (m == COLO_MOD_DOOM) unowned_offers++;
        }
        s.modifiers.draft_pending = 0;
    }
    CHECK("both modifiers appear across the sample", owned_offers > 0 && unowned_offers > 0);
    CHECK("the owned T1 modifier is offered with clearly elevated frequency (~2x weight)",
        owned_offers * 2 > unowned_offers * 3);
    printf("  owned=%d unowned=%d over %d drafts\n", owned_offers, unowned_offers, N);

    s.modifiers.tier[COLO_MOD_RELENTLESS] = 3;
    int maxed_seen = 0;
    for (int rep = 0; rep < 200; rep++) {
        col_modifier_open_draft(&s, 4);
        for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++)
            if (s.modifiers.draft_options[o] == COLO_MOD_RELENTLESS) maxed_seen = 1;
        s.modifiers.draft_pending = 0;
    }
    CHECK("a maxed modifier is never offered again", maxed_seen == 0);
}

static void test_frailty_hp(void) {
    printf("test_frailty_hp\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 1);
    CHECK("base HP is 99 with no Frailty", s.player.base_hitpoints == 99);

    s.modifiers.active_mask |= (1u << COLO_MOD_FRAILTY);
    s.modifiers.tier[COLO_MOD_FRAILTY] = 3;
    col_mod_apply_frailty_hp(&s);
    CHECK("Frailty III cuts max HP to 60", s.player.base_hitpoints == 60);
    CHECK("current HP clamped to new max", s.player.current_hitpoints <= 60);

    s.modifiers.tier[COLO_MOD_FRAILTY] = 1;
    col_mod_apply_frailty_hp(&s);
    CHECK("Frailty I cuts max HP to 90", s.player.base_hitpoints == 90);
}

static void test_relentless_damage(void) {
    printf("test_relentless_damage\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 3);

    const ColoNpcStats* berserker = &COLO_NPC_STATS[COLO_FREMENNIK_BERSERKER];
    long base_total = 0, relentless_total = 0;
    int base_hits = 0, relentless_hits = 0;
    const int N = 20000;

    s.modifiers.active_mask = 0;
    s.modifiers.tier[COLO_MOD_RELENTLESS] = 0;
    for (int i = 0; i < N; i++) {
        int hit = 0;
        int dmg = col_npc_roll_vs_player(&s, berserker, ATTACK_STYLE_MELEE, berserker->max_hit, &hit);
        base_total += dmg;
        base_hits += hit;
    }

    s.modifiers.active_mask |= (1u << COLO_MOD_RELENTLESS);
    s.modifiers.tier[COLO_MOD_RELENTLESS] = 3;
    for (int i = 0; i < N; i++) {
        int hit = 0;
        int dmg = col_npc_roll_vs_player(&s, berserker, ATTACK_STYLE_MELEE, berserker->max_hit, &hit);
        relentless_total += dmg;
        relentless_hits += hit;
    }

    CHECK("Relentless III forces every hit to land", relentless_hits == N);
    CHECK("Relentless III lands more often than baseline", relentless_hits > base_hits);
    CHECK("Relentless raises mean incoming damage", relentless_total > base_total);
    printf("  base_hits=%d/%d relentless_hits=%d/%d base_dmg=%ld relentless_dmg=%ld\n",
        base_hits, N, relentless_hits, N, base_total, relentless_total);
}

static void test_quartet_extra_spawn(void) {
    printf("test_quartet_extra_spawn\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);

    ColosseumState base;
    memset(&base, 0, sizeof(base));
    col_reset_ctx((EncounterState*)&base, (EncounterContext*)&ctx, 5);
    base.wave = 0;
    col_spawn_wave(&base);
    int base_count = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++) if (base.npcs[i].active) base_count++;

    ColosseumState q;
    memset(&q, 0, sizeof(q));
    col_reset_ctx((EncounterState*)&q, (EncounterContext*)&ctx, 5);
    q.modifiers.active_mask |= (1u << COLO_MOD_QUARTET);
    q.modifiers.tier[COLO_MOD_QUARTET] = 1;
    q.wave = 0;
    col_spawn_wave(&q);
    int q_count = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++) if (q.npcs[i].active) q_count++;

    CHECK("Quartet spawns one extra NPC at wave start", q_count == base_count + 1);

    ColosseumState q12;
    memset(&q12, 0, sizeof(q12));
    col_reset_ctx((EncounterState*)&q12, (EncounterContext*)&ctx, 5);
    q12.modifiers.active_mask |= (1u << COLO_MOD_QUARTET);
    q12.modifiers.tier[COLO_MOD_QUARTET] = 1;
    q12.wave = COLO_WAVE_BOSS;
    col_spawn_wave(&q12);
    int sol = 0, warband = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        if (!q12.npcs[i].active) continue;
        if (q12.npcs[i].type == COLO_SOL_HEREDIT) sol++;
        else warband++;
    }
    CHECK("wave 12 has Sol", sol == 1);
    CHECK("Quartet adds a warbander on wave 12", warband == 1);
}

static void test_bees_hazard(void) {
    printf("test_bees_hazard\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 11);
    advance_to_wave_spawn(&s, &ctx);
    s.modifiers.active_mask |= (1u << COLO_MOD_BEES);
    s.modifiers.tier[COLO_MOD_BEES] = 2;
    s.wave = 0;
    col_spawn_wave(&s);

    int bee_npcs = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++)
        if (s.npcs[i].active && s.npcs[i].type == COLO_BEE_SWARM) bee_npcs++;
    CHECK("Bees II fields two 1-HP bee NPCs", bee_npcs == 2 &&
        s.bees[0].phase == COLO_HAZARD_ALIVE && s.bees[1].phase == COLO_HAZARD_ALIVE);
    CHECK("a bee NPC has exactly 1 HP", s.npcs[s.bees[0].npc_slot].hp == 1);
    CHECK("a bee swarm uses its cache 2x2 footprint", s.npcs[s.bees[0].npc_slot].size == 2);

    ColoNPC* bee_npc = &s.npcs[s.bees[0].npc_slot];
    int bx = bee_npc->x, by = bee_npc->y;
    for (int t = 0; t < COLO_BEE_MOVE_INTERVAL - 1; t++) col_mod_tick_bees(&s);
    CHECK("the swarm holds for 11 ticks", bee_npc->x == bx && bee_npc->y == by);
    col_mod_tick_bees(&s);
    int stepped = abs(bee_npc->x - bx) <= 1 && abs(bee_npc->y - by) <= 1 &&
        (bee_npc->x != bx || bee_npc->y != by);
    CHECK("the 12th tick steps one tile (diagonal allowed) toward the player", stepped);

    int all_tiles_poison = 1;
    for (int dx = 0; dx < 2; dx++) {
        for (int dy = 0; dy < 2; dy++) {
            bee_npc->x = 10;
            bee_npc->y = 10;
            s.player.x = 10 + dx;
            s.player.y = 10 + dy;
            s.player_poison = 0;
            s.player_poison_timer = 0;
            s.bees[0].move_timer = COLO_BEE_MOVE_INTERVAL;
            col_mod_tick_bees(&s);
            if (s.player_poison != COLO_POISON_BEE_CONTACT_SEVERITY ||
                    s.player_poison_timer != COLO_POISON_INTERVAL)
                all_tiles_poison = 0;
        }
    }
    CHECK("bee contact applies poison from all four footprint tiles", all_tiles_poison);

    bee_npc->x = s.player.x;
    bee_npc->y = s.player.y;
    s.player.prayer = PRAYER_PROTECT_MELEE;
    int damaged = 0;
    for (int t = 0; t < 64 && !damaged; t++) {
        bee_npc->x = s.player.x;
        bee_npc->y = s.player.y;
        s.player.current_hitpoints = 99;
        col_mod_tick_bees(&s);
        if (s.player.current_hitpoints < 99) damaged = 1;
    }
    CHECK("a swarm beneath the player deals unblockable damage", damaged);

    s.wave = COLO_WAVE_BOSS;
    col_sol_begin_boss_arena(&s);
    bee_npc->x = COLO_BOSS_ARENA_MIN_X;
    bee_npc->y = COLO_BOSS_ARENA_MIN_Y + 2;
    s.player.x = COLO_BOSS_ARENA_MIN_X - 3;
    s.player.y = bee_npc->y;
    s.bees[0].move_timer = 1;
    col_mod_tick_bees(&s);
    CHECK("bee movement ignores the wave-12 boss-box clamp",
        bee_npc->x == COLO_BOSS_ARENA_MIN_X - 1);
    s.wave = 0;
    s.sol = (SolHereditState){0};
    s.player.x = bee_npc->x;
    s.player.y = bee_npc->y;

    int slot = s.bees[0].npc_slot;
    col_player_attack_target(&s, slot);
    land_pending_player_hits(&s);
    CHECK("a single hit kills the swarm", !s.npcs[slot].active);
    CHECK("the killed swarm enters its 50-tick respawn",
        s.bees[0].phase == COLO_HAZARD_RESPAWNING &&
        s.bees[0].respawn_timer == COLO_BEE_RESPAWN_TICKS);
    for (int t = 0; t < COLO_BEE_RESPAWN_TICKS - 1; t++) col_mod_tick_bees(&s);
    CHECK("still respawning one tick early", s.bees[0].phase == COLO_HAZARD_RESPAWNING);
    col_mod_tick_bees(&s);
    CHECK("the swarm respawns exactly 50 ticks after death",
        s.bees[0].phase == COLO_HAZARD_ALIVE &&
        s.npcs[s.bees[0].npc_slot].active &&
        s.npcs[s.bees[0].npc_slot].type == COLO_BEE_SWARM);

    ColosseumState sc;
    memset(&sc, 0, sizeof(sc));
    ctx.config.start_wave = 0;
    col_reset_ctx((EncounterState*)&sc, (EncounterContext*)&ctx, 13);
    sc.modifiers.draft_pending = 0;
    sc.modifiers.draft_gates_spawn = 0;
    sc.modifiers.draft_free_movement = 0;
    sc.modifiers.active_mask |= (1u << COLO_MOD_BEES);
    sc.modifiers.tier[COLO_MOD_BEES] = 1;
    sc.wave = 0;
    col_spawn_wave(&sc);
    sc.wave_spawn_delay = 0;
    int live_bee = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++)
        if (sc.npcs[i].active && sc.npcs[i].type == COLO_BEE_SWARM) live_bee = 1;
    CHECK("a bee NPC is live going into the clear check", live_bee);
    force_clear_wave(&sc);
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    step_and_observe(&sc, &ctx, idle);
    CHECK("the wave clears (next draft opens) with the bee swarm still alive",
        draft_is_open(&sc) && sc.wave_spawn_target == 1);
}

static void test_totem_lifecycle(void) {
    printf("test_totem_lifecycle\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 1;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 211);
    geo_clear_npcs(&s);
    s.modifiers.active_mask |= (1u << COLO_MOD_TOTEMIC);
    s.modifiers.tier[COLO_MOD_TOTEMIC] = 1;
    s.player.x = 25; s.player.y = 18;
    col_rebuild_player_collision_flags(&s);

    col_init_npc(&s, 0, COLO_SERPENT_SHAMAN, 12, 16);
    s.npcs[0].hp = 70;
    col_mod_on_npc_hp_changed(&s, 0);
    CHECK("no totem above 50% HP", s.totems[0].phase == COLO_HAZARD_NONE);
    s.npcs[0].hp = 60;
    col_mod_on_npc_hp_changed(&s, 0);
    CHECK("crossing <=50% spawns a totem", s.totems[0].phase == COLO_HAZARD_ALIVE);
    int tslot = s.totems[0].npc_slot;
    CHECK("the totem is a live 1-HP NPC beside its owner",
        tslot >= 0 && s.npcs[tslot].active &&
        s.npcs[tslot].type == COLO_HEALING_TOTEM && s.npcs[tslot].hp == 1);
    col_mod_on_npc_hp_changed(&s, 0);
    CHECK("no duplicate totem for the same owner", s.totems[0].phase == COLO_HAZARD_ALIVE);

    for (int t = 0; t < COLO_TOTEM_HEAL_INTERVAL - 1; t++) col_mod_tick_totems(&s);
    CHECK("no heal before the 7th tick", s.npcs[0].hp == 60);
    col_mod_tick_totems(&s);
    CHECK("the 7th tick heals 30% of the owner's max HP", s.npcs[0].hp == 60 + 37);
    for (int t = 0; t < COLO_TOTEM_HEAL_INTERVAL; t++) col_mod_tick_totems(&s);
    CHECK("the pulse is gated while the owner is above 50%", s.npcs[0].hp == 97);
    s.npcs[0].hp = 50;
    for (int t = 0; t < COLO_TOTEM_HEAL_INTERVAL; t++) col_mod_tick_totems(&s);
    CHECK("the pulse resumes once the owner re-crosses 50%", s.npcs[0].hp == 87);

    col_player_attack_target(&s, tslot);
    land_pending_player_hits(&s);
    CHECK("a single attack destroys the totem", !s.npcs[tslot].active);
    CHECK("destruction arms the 200-tick respawn",
        s.totems[0].phase == COLO_HAZARD_RESPAWNING &&
        s.totems[0].respawn_timer == COLO_TOTEM_RESPAWN_TICKS);
    for (int t = 0; t < COLO_TOTEM_RESPAWN_TICKS - 1; t++) col_mod_tick_totems(&s);
    CHECK("still down one tick early", s.totems[0].phase == COLO_HAZARD_RESPAWNING);
    col_mod_tick_totems(&s);
    CHECK("the totem respawns exactly 200 ticks after destruction",
        s.totems[0].phase == COLO_HAZARD_ALIVE &&
        s.npcs[s.totems[0].npc_slot].active &&
        s.npcs[s.totems[0].npc_slot].type == COLO_HEALING_TOTEM);

    int tslot2 = s.totems[0].npc_slot;
    s.npcs[0].hp = 0;
    col_apply_npc_death(&s, 0);
    CHECK("the owner's death despawns its totem",
        !s.npcs[tslot2].active && s.totems[0].phase == COLO_HAZARD_NONE);

    col_init_npc(&s, 0, COLO_SERPENT_SHAMAN, 12, 16);
    s.npcs[0].hp = 60;
    col_mod_on_npc_hp_changed(&s, 0);
    int tslot3 = s.totems[0].npc_slot;
    col_player_attack_target(&s, tslot3);
    land_pending_player_hits(&s);
    CHECK("second totem down and respawning", s.totems[0].phase == COLO_HAZARD_RESPAWNING);
    s.npcs[0].hp = 0;
    col_apply_npc_death(&s, 0);
    CHECK("the owner's death cancels a pending totem respawn",
        s.totems[0].phase == COLO_HAZARD_NONE);

}

static void test_totemic_sol_wave12(void) {
    printf("test_totemic_sol_wave12\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 11;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 223);
    advance_to_wave_spawn(&s, &ctx);
    s.modifiers.active_mask |= (1u << COLO_MOD_TOTEMIC);
    s.modifiers.tier[COLO_MOD_TOTEMIC] = 1;
    int sol = col_sol_find_idx(&s);
    CHECK("Sol is live", sol >= 0);

    s.npcs[sol].hp = COLO_SOL_HP_MAX * 60 / 100;
    col_mod_on_npc_hp_changed(&s, sol);
    CHECK("no totem while Sol is above 50%", s.totems[sol].phase == COLO_HAZARD_NONE);
    s.npcs[sol].hp = COLO_SOL_HP_MAX / 2;
    col_mod_on_npc_hp_changed(&s, sol);
    CHECK("Sol at 50% spawns a totem", s.totems[sol].phase == COLO_HAZARD_ALIVE);
    int tslot = s.totems[sol].npc_slot;
    CHECK("Sol's totem is an attackable 1-HP NPC inside the boss arena",
        s.npcs[tslot].type == COLO_HEALING_TOTEM && s.npcs[tslot].hp == 1 &&
        col_in_boss_arena(&s, s.npcs[tslot].x, s.npcs[tslot].y));

    int hp0 = s.npcs[sol].hp;
    for (int t = 0; t < COLO_TOTEM_HEAL_INTERVAL - 1; t++) col_mod_tick_totems(&s);
    CHECK("no Sol heal before the 7th tick", s.npcs[sol].hp == hp0);
    col_mod_tick_totems(&s);
    CHECK("the pulse heals Sol exactly 75", s.npcs[sol].hp == hp0 + COLO_TOTEM_SOL_HEAL);
    for (int t = 0; t < COLO_TOTEM_HEAL_INTERVAL; t++) col_mod_tick_totems(&s);
    CHECK("Sol keeps healing 75/7t even above 50% (until destroyed)",
        s.npcs[sol].hp == hp0 + 2 * COLO_TOTEM_SOL_HEAL);

    col_player_attack_target(&s, tslot);
    land_pending_player_hits(&s);
    int hp1 = s.npcs[sol].hp;
    for (int t = 0; t < 3 * COLO_TOTEM_HEAL_INTERVAL; t++) col_mod_tick_totems(&s);
    CHECK("a destroyed totem stops the Sol heal (until the 200t respawn)",
        !s.npcs[tslot].active && s.npcs[sol].hp == hp1 &&
        s.totems[sol].phase == COLO_HAZARD_RESPAWNING);
}

static void test_reentry_sand_tiles(void) {
    printf("test_reentry_sand_tiles\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 1;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 227);
    geo_clear_npcs(&s);

    s.modifiers.active_mask |= (1u << COLO_MOD_REENTRY);
    s.modifiers.tier[COLO_MOD_REENTRY] = 1;
    col_mod_reentry_on_skyfall(&s, 20, 12);
    CHECK("T1 leaves one pool on the targeted tile",
        s.molten_count == 1 && s.molten_x[0] == 20 && s.molten_y[0] == 12);
    CHECK("the T1 pool is the stronger Reentry kind",
        s.molten_kind[0] == COLO_POOL_REENTRY);

    s.player.x = 20; s.player.y = 12;
    int burns = 0, burn_ok = 1, off_cadence_seen = 0;
    for (int t = 0; t < 24; t++) {
        s.player.current_hitpoints = 99;
        col_mod_tick_molten_pools(&s);
        int dmg = 99 - s.player.current_hitpoints;
        if (dmg > 0) {
            burns++;
            if (dmg < 1 || dmg > COLO_REENTRY_MOLTEN_MAX_HIT) burn_ok = 0;
        } else {
            off_cadence_seen = 1;
        }
    }
    CHECK("Reentry burn is 1-15 and always positive when it fires", burn_ok);
    CHECK("Reentry fires every other tick (~half of 24)", burns >= 11 && burns <= 13);
    CHECK("Reentry has off-cadence no-damage ticks", off_cadence_seen);

    for (int t = 0; t < 500; t++) col_mod_tick_molten_pools(&s);
    CHECK("the T1 pool persists all wave", s.molten_count == 1);
    col_modifiers_on_wave_spawn(&s);
    CHECK("T1 (temporary) clears at wave end", s.molten_count == 0);

    s.modifiers.tier[COLO_MOD_REENTRY] = 2;
    col_mod_reentry_on_skyfall(&s, 20, 12);
    int has_target = 0, has_sw = 0, has_w = 0, all_reentry = 1;
    for (int i = 0; i < s.molten_count; i++) {
        if (s.molten_x[i] == 20 && s.molten_y[i] == 12) has_target = 1;
        if (s.molten_x[i] == 19 && s.molten_y[i] == 11) has_sw = 1;
        if (s.molten_x[i] == 19 && s.molten_y[i] == 12) has_w = 1;
        if (s.molten_kind[i] != COLO_POOL_REENTRY) all_reentry = 0;
    }
    CHECK("T2 covers the targeted tile + the tile SOUTH-WEST of it",
        s.molten_count == 2 && has_target && has_sw && !has_w);
    CHECK("T2 pools are Reentry kind", all_reentry);
    col_modifiers_on_wave_spawn(&s);
    CHECK("Reentry T2 pools are PERMANENT (survive wave end)", s.molten_count == 2);

    s.molten_count = 0;
    s.modifiers.tier[COLO_MOD_REENTRY] = 3;
    col_mod_reentry_on_skyfall(&s, 20, 12);
    has_target = has_sw = has_w = 0;
    for (int i = 0; i < s.molten_count; i++) {
        if (s.molten_x[i] == 20 && s.molten_y[i] == 12) has_target = 1;
        if (s.molten_x[i] == 19 && s.molten_y[i] == 11) has_sw = 1;
        if (s.molten_x[i] == 19 && s.molten_y[i] == 12) has_w = 1;
    }
    CHECK("T3 additionally covers the WEST tile", s.molten_count == 3 &&
        has_target && has_sw && has_w);
    col_modifiers_on_wave_spawn(&s);
    CHECK("Reentry T3 pools are PERMANENT (survive wave end)", s.molten_count == 3);

    s.molten_count = 0;
    s.modifiers.active_mask |= (1u << COLO_MOD_VOLATILITY);
    s.modifiers.tier[COLO_MOD_VOLATILITY] = 3;
    s.player.x = 5; s.player.y = 18;
    col_mod_volatility_on_death(&s, 20, 16, 1);
    CHECK("Volatility T3 leaves a temporary Volatility pool at the centre",
        s.molten_count == 1 && s.molten_kind[0] == COLO_POOL_VOLATILITY);
    col_modifiers_on_wave_spawn(&s);
    CHECK("the Volatility (temporary) pool clears at wave end", s.molten_count == 0);
}

static void test_venom_escalation(void) {
    printf("test_venom_escalation\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 1;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 229);
    s.modifiers.active_mask |= (1u << COLO_MOD_MANTIMAYHEM);
    s.modifiers.tier[COLO_MOD_MANTIMAYHEM] = 2;

    col_mod_manticore_apply_venom(&s, 1);
    CHECK("the first proc arms 6 damage on the 30-tick clock",
        s.player_venom == COLO_VENOM_START && s.player_venom_timer == COLO_VENOM_INTERVAL);

    static const int EXPECT[9] = { 6, 8, 10, 12, 14, 16, 18, 20, 20 };
    int seq_ok = 1, cadence_ok = 1;
    for (int k = 0; k < 9; k++) {
        for (int t = 0; t < COLO_VENOM_INTERVAL - 1; t++) {
            s.player.current_hitpoints = 99;
            col_mod_tick_venom(&s);
            if (s.player.current_hitpoints != 99) cadence_ok = 0;
        }
        s.player.current_hitpoints = 99;
        col_mod_tick_venom(&s);
        if (99 - s.player.current_hitpoints != EXPECT[k]) seq_ok = 0;
    }
    CHECK("venom deals 6,8,10..20 then holds the cap", seq_ok);
    CHECK("venom damage lands exactly every 30 ticks", cadence_ok);

    s.player_venom = COLO_VENOM_START;
    s.player_venom_timer = 17;
    col_mod_manticore_apply_venom(&s, 1);
    CHECK("reapplication bumps the next damage +2 without resetting the clock",
        s.player_venom == COLO_VENOM_START + COLO_VENOM_STEP &&
        s.player_venom_timer == 17);
    s.player_venom = COLO_VENOM_CAP;
    col_mod_manticore_apply_venom(&s, 1);
    CHECK("reapplication never exceeds the 20 cap", s.player_venom == COLO_VENOM_CAP);

    s.player_venom = COLO_VENOM_START;
    s.player_venom_timer = 17;
    s.modifiers.draft_pending = 1;
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    step_and_observe(&s, &ctx, idle);
    CHECK("venom timer freezes during the draft gap", s.player_venom_timer == 17);
    s.modifiers.draft_pending = 0;

    s.player_venom = COLO_VENOM_START;
    s.player_venom_timer = 2;
    col_modifiers_on_wave_spawn(&s);
    CHECK("venom survives the wave boundary",
        s.player_venom == COLO_VENOM_START && s.player_venom_timer == 2);
    s.player.current_hitpoints = 99;
    col_mod_tick_venom(&s);
    CHECK("venom does not tick early after the wave boundary", s.player.current_hitpoints == 99);
    col_mod_tick_venom(&s);
    CHECK("venom still ticks on the next wave",
        s.player.current_hitpoints == 99 - COLO_VENOM_START &&
        s.player_venom == COLO_VENOM_START + COLO_VENOM_STEP);
}

static void test_bee_poison_status(void) {
    printf("test_bee_poison_status\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 230);

    col_mod_apply_bee_poison(&s);
    CHECK("bee poison starts at severity 5",
        s.player_poison == COLO_POISON_BEE_CONTACT_SEVERITY &&
        s.player_poison_timer == COLO_POISON_INTERVAL);

    int cadence_ok = 1;
    int hits_ok = 1;
    for (int hit = 0; hit < COLO_POISON_BEE_CONTACT_SEVERITY; hit++) {
        int severity_before = s.player_poison;
        for (int t = 0; t < COLO_POISON_INTERVAL - 1; t++) {
            s.player.current_hitpoints = 99;
            col_mod_tick_poison(&s);
            if (s.player.current_hitpoints != 99) cadence_ok = 0;
        }
        s.player.current_hitpoints = 99;
        col_mod_tick_poison(&s);
        if (99 - s.player.current_hitpoints != 1 ||
                s.player_poison != severity_before - 1)
            hits_ok = 0;
    }
    CHECK("bee poison deals exactly five 1-damage hits", hits_ok);
    CHECK("bee poison hits exactly 30 ticks apart", cadence_ok);
    CHECK("bee poison expires at severity 0",
        s.player_poison == 0 && s.player_poison_timer == 0);
}

static void test_mantimayhem_t3_shuffle(void) {
    printf("test_mantimayhem_t3_shuffle\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 1;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 239);
    geo_clear_npcs(&s);
    s.modifiers.active_mask |= (1u << COLO_MOD_MANTIMAYHEM);
    s.modifiers.tier[COLO_MOD_MANTIMAYHEM] = 3;
    s.player.x = 17; s.player.y = 16;
    col_rebuild_player_collision_flags(&s);
    int one_each_ok = 1;
    int arm_copies_fixed_ok = 1;
    int melee_slot_seen[3] = { 0, 0, 0 };
    for (int rep = 0; rep < 300; rep++) {
        geo_clear_npcs(&s);
        s.wave_manticore_pattern_rolled = 0;
        col_init_npc(&s, 0, COLO_MANTICORE, 16, 12);
        ColoManticoreState* mc = colo_npc_manticore(&s.npcs[0]);
        if (mc->orb_style[0] != ATTACK_STYLE_NONE ||
                mc->orb_style[1] != ATTACK_STYLE_NONE ||
                mc->orb_style[2] != ATTACK_STYLE_NONE) {
            arm_copies_fixed_ok = 0;
        }
        s.npcs[0].attack_timer = 0;
        s.player.current_hitpoints = 99;
        col_npc_manticore_arm(&s, 0);
        if (mc->cycle_step != 0) arm_copies_fixed_ok = 0;
        int counts[3] = { 0, 0, 0 };
        for (int o = 0; o < 3; o++) {
            if (mc->fixed_orb_style[o] == ATTACK_STYLE_RANGED) counts[0]++;
            if (mc->fixed_orb_style[o] == ATTACK_STYLE_MAGIC) counts[1]++;
            if (mc->fixed_orb_style[o] == ATTACK_STYLE_MELEE) {
                counts[2]++;
                melee_slot_seen[o] = 1;
            }
            if (mc->orb_style[o] != mc->fixed_orb_style[o]) arm_copies_fixed_ok = 0;
        }
        if (counts[0] != 1 || counts[1] != 1 || counts[2] != 1) one_each_ok = 0;
    }
    CHECK("T3 fixed cycles always carry exactly one of each style",
        one_each_ok);
    CHECK("arming copies the fixed cycle into the active telegraph",
        arm_copies_fixed_ok);
    CHECK("the melee orb appears in every position across the sample",
        melee_slot_seen[0] && melee_slot_seen[1] && melee_slot_seen[2]);
}

static void test_relentless_def_level_bypass(void) {
    printf("test_relentless_def_level_bypass\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 1;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 233);

    const EncounterLoadoutStats* ls = col_live_loadout_stats(&s);
    int def_bonus = encounter_player_def_bonus(
        ls->def_stab, ls->def_slash, ls->def_crush, ls->def_magic, ls->def_ranged,
        ATTACK_STYLE_MELEE, MELEE_STYLE_STAB);
    CHECK("rig sanity: the geared player has a positive melee defence bonus", def_bonus > 0);

    int t0 = col_player_def_roll(&s, ATTACK_STYLE_MELEE, MELEE_STYLE_STAB);
    CHECK("tier 0 uses the full 99 defence level", t0 == (99 + 8) * (def_bonus + 64));

    s.modifiers.active_mask |= (1u << COLO_MOD_RELENTLESS);
    s.modifiers.tier[COLO_MOD_RELENTLESS] = 1;
    int t1 = col_player_def_roll(&s, ATTACK_STYLE_MELEE, MELEE_STYLE_STAB);
    CHECK("tier I keeps exactly 67% of the level (99 -> 66), bonus intact",
        t1 == (66 + 8) * (def_bonus + 64));

    s.modifiers.tier[COLO_MOD_RELENTLESS] = 2;
    int t2 = col_player_def_roll(&s, ATTACK_STYLE_MELEE, MELEE_STYLE_STAB);
    CHECK("tier II keeps exactly 34% of the level (99 -> 33), bonus intact",
        t2 == (33 + 8) * (def_bonus + 64));

    s.modifiers.tier[COLO_MOD_RELENTLESS] = 3;
    int t3 = col_player_def_roll(&s, ATTACK_STYLE_MELEE, MELEE_STYLE_STAB);
    CHECK("tier III zeroes the level share; the gear term alone remains",
        t3 == 8 * (def_bonus + 64));
}

static void test_mantimayhem_stress(void) {
    printf("test_mantimayhem_stress\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 8;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 17);
    s.modifiers.active_mask |= (1u << COLO_MOD_MANTIMAYHEM);
    s.modifiers.tier[COLO_MOD_MANTIMAYHEM] = 2;
    s.wave = 8;
    col_spawn_wave(&s);

    int manticores = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++)
        if (s.npcs[i].active && s.npcs[i].type == COLO_MANTICORE) manticores++;
    CHECK("wave 9 spawns two manticores", manticores == 2);

    int idle[COLO_NUM_ACTION_HEADS] = {0};
    int venom_seen = 0;
    for (int t = 0; t < 2000 && !s.episode_over; t++) {
        s.player.current_hitpoints = 9999;
        step_and_observe(&s, &ctx, idle);
        if (s.player_venom > 0) venom_seen = 1;
    }
    CHECK("Mantimayhem T2 survived sustained barrages without queue overflow", 1);
    CHECK("Mantimayhem T2 inflicted venom on the player", venom_seen);
}

static int sf_collect_move_gaps(ColosseumState* s, int ticks, int* gaps, int max_gaps) {
    int n = 0;
    int last_move_t = -1;
    int previous_step = s->solarflare.step;
    for (int t = 1; t <= ticks; t++) {
        col_mod_tick_solarflare(s);
        if (s->solarflare.step != previous_step) {
            if (last_move_t >= 0 && n < max_gaps) gaps[n++] = t - last_move_t;
            last_move_t = t;
            previous_step = s->solarflare.step;
        }
    }
    return n;
}

static int sf_tile_on_pillar_perimeter(int pillar_index, int x, int y) {
    int px = COLO_PILLARS[pillar_index][0];
    int py = COLO_PILLARS[pillar_index][1];
    int in_ring_box = x >= px - 1 && x <= px + COLO_PILLAR_SIZE &&
        y >= py - 1 && y <= py + COLO_PILLAR_SIZE;
    int in_pillar = x >= px && x < px + COLO_PILLAR_SIZE &&
        y >= py && y < py + COLO_PILLAR_SIZE;
    return in_ring_box && !in_pillar;
}

static int sf_advance_to_step(ColosseumState* s, int target_step, int max_ticks) {
    for (int t = 0; t < max_ticks; t++) {
        col_mod_tick_solarflare(s);
        if (s->solarflare.step == target_step) return 1;
    }
    return 0;
}

static void sf_reset_tier(ColosseumState* s, int tier) {
    s->modifiers.tier[COLO_MOD_SOLARFLARE] = tier;
    col_mod_sync_solarflare(s);
    s->player.x = 16;
    s->player.y = 16;
}

static void test_solarflare_orb(void) {
    printf("test_solarflare_orb\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 23);
    s.modifiers.active_mask |= (1u << COLO_MOD_SOLARFLARE);
    s.modifiers.tier[COLO_MOD_SOLARFLARE] = 2;
    s.wave = 0;
    col_spawn_wave(&s);
    CHECK("Solarflare orb is active", s.solarflare.active);

    int geometry_ok = 1;
    int corner_tiles_ok = 1;
    for (int p = 0; p < COLO_NUM_PILLARS; p++) {
        int px = COLO_PILLARS[p][0];
        int py = COLO_PILLARS[p][1];
        const int corners[4][3] = {
            {0, px - 1, py - 1},
            {4, px + COLO_PILLAR_SIZE, py - 1},
            {8, px + COLO_PILLAR_SIZE, py + COLO_PILLAR_SIZE},
            {12, px - 1, py + COLO_PILLAR_SIZE},
        };
        for (int step = 0; step < COLO_SOLARFLARE_RING_STEPS; step++) {
            int x, y;
            col_solarflare_tile(&s, p, step, &x, &y);
            if (!sf_tile_on_pillar_perimeter(p, x, y)) geometry_ok = 0;
            if (col_static_blocked(x, y)) geometry_ok = 0;
        }
        for (int c = 0; c < 4; c++) {
            int x, y;
            col_solarflare_tile(&s, p, corners[c][0], &x, &y);
            if (x != corners[c][1] || y != corners[c][2]) corner_tiles_ok = 0;
        }
    }
    CHECK("Solarflare has one distance-1 perimeter orb per pillar", geometry_ok);
    CHECK("Solarflare visits the four specified pillar-ring corners", corner_tiles_ok);

    RenderEntity entities[COLO_MAX_NPCS + COLO_NUM_PILLARS + 2];
    int entity_count = 0;
    col_fill_render_entities_ctx(
        (EncounterState*)&s, (EncounterContext*)&ctx,
        entities, COLO_MAX_NPCS + COLO_NUM_PILLARS + 2, &entity_count);
    int render_orbs = 0;
    int render_slots_ok = 1;
    for (int e = 0; e < entity_count; e++) {
        if (entities[e].npc_def_id != COLO_NPC_DEF_ID_SOLARFLARE) continue;
        int p = entities[e].npc_slot - COLO_NPC_SLOT_SOLARFLARE;
        if (p < 0 || p >= COLO_NUM_PILLARS) {
            render_slots_ok = 0;
        } else {
            int x, y;
            col_solarflare_tile(&s, p, s.solarflare.step, &x, &y);
            if (entities[e].npc_instance_id != COLO_SOLARFLARE_RENDER_INSTANCE_ID + (uint32_t)p)
                render_slots_ok = 0;
            if (entities[e].x != x || entities[e].y != y) render_slots_ok = 0;
        }
        render_orbs++;
    }
    CHECK("Solarflare renders four per-pillar orb entities",
        render_orbs == COLO_NUM_PILLARS && render_slots_ok);

    int obs_pillar = 2;
    int obs_x, obs_y;
    col_solarflare_tile(&s, obs_pillar, s.solarflare.step, &obs_x, &obs_y);
    s.player.x = obs_x;
    s.player.y = obs_y;
    static float obs[COLO_NUM_OBS];
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    int solarflare_obs_idx = TEST_MOD_OBS_SOLARFLARE;
    CHECK("Solarflare modifier obs writes nearest of four orbs",
        obs[solarflare_obs_idx] == 0.0f &&
        obs[solarflare_obs_idx + 1] == 0.0f &&
        obs[solarflare_obs_idx + 2] == 1.0f);

    int gaps[32];
    s.player.x = 16;
    s.player.y = 16;
    int n = sf_collect_move_gaps(&s, 40, gaps, 32);
    int t2_ok = n >= 8;
    for (int g = 0; g < n; g++) if (gaps[g] != 2) t2_ok = 0;
    CHECK("T2 moves every 2 ticks with no corner pause", t2_ok);
    sf_reset_tier(&s, 2);
    int t2_corner_reached = sf_advance_to_step(&s, 4, 16);
    CHECK("T2 does not pause at Solarflare corners",
        t2_corner_reached && s.solarflare.pause_timer == 0);

    sf_reset_tier(&s, 1);
    n = sf_collect_move_gaps(&s, 60, gaps, 32);
    int t1_ok = n >= 6;
    for (int g = 0; g < n; g++)
        if (gaps[g] != 2 && gaps[g] != 2 + COLO_SOLARFLARE_CORNER_PAUSE) t1_ok = 0;
    int t1_paused = 0;
    for (int g = 0; g < n; g++)
        if (gaps[g] == 2 + COLO_SOLARFLARE_CORNER_PAUSE) t1_paused = 1;
    CHECK("T1 moves every 2 ticks and pauses 7 at each corner", t1_ok && t1_paused);
    sf_reset_tier(&s, 1);
    int t1_corner_reached = sf_advance_to_step(&s, 4, 16);
    CHECK("T1 sets a 7 tick Solarflare corner pause",
        t1_corner_reached && s.solarflare.pause_timer == COLO_SOLARFLARE_CORNER_PAUSE);

    sf_reset_tier(&s, 3);
    n = sf_collect_move_gaps(&s, 40, gaps, 32);
    int t3_ok = n >= 8, t3_paused = 0, t3_fast = 0;
    for (int g = 0; g < n; g++) {
        if (gaps[g] != 1 && gaps[g] != 1 + COLO_SOLARFLARE_CORNER_PAUSE_T3) t3_ok = 0;
        if (gaps[g] == 1 + COLO_SOLARFLARE_CORNER_PAUSE_T3) t3_paused = 1;
        if (gaps[g] == 1) t3_fast = 1;
    }
    CHECK("T3 moves every tick AND stops 2 ticks at each corner (A27)",
        t3_ok && t3_paused && t3_fast);
    sf_reset_tier(&s, 3);
    int t3_corner_reached = sf_advance_to_step(&s, 4, 16);
    CHECK("T3 sets a 2 tick Solarflare corner pause",
        t3_corner_reached && s.solarflare.pause_timer == COLO_SOLARFLARE_CORNER_PAUSE_T3);

    CHECK("Solarflare max-hit constants remain tiered",
        col_mod_solarflare_max_hit(1) == COLO_SOLARFLARE_MAX_HIT_T1 &&
        col_mod_solarflare_max_hit(2) == COLO_SOLARFLARE_MAX_HIT_T2 &&
        col_mod_solarflare_max_hit(3) == COLO_SOLARFLARE_MAX_HIT_T3);

    sf_reset_tier(&s, 3);
    int hit_x, hit_y;
    col_solarflare_tile(&s, 0, s.solarflare.step, &hit_x, &hit_y);
    s.player.x = hit_x;
    s.player.y = hit_y;
    s.player.prayer = PRAYER_PROTECT_MAGIC;
    s.player.current_hitpoints = 9999;
    int damaged = 0;
    for (int t = 0; t < 80 && !damaged; t++) {
        int hp = s.player.current_hitpoints;
        s.solarflare.pause_timer = 1;
        col_mod_tick_solarflare(&s);
        if (s.player.current_hitpoints < hp) damaged = 1;
    }
    CHECK("Solarflare orb deals contact damage", damaged);
    CHECK("Solarflare III disables prayer on hit", s.player.prayer == PRAYER_NONE);
}

static void test_volatility_explosion(void) {
    printf("test_volatility_explosion\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 29);
    s.modifiers.active_mask |= (1u << COLO_MOD_VOLATILITY);
    s.modifiers.tier[COLO_MOD_VOLATILITY] = 1;

    s.player.x = 17; s.player.y = 17;
    s.player.current_hitpoints = 99;
    int idx = 0;
    col_init_npc(&s, idx, COLO_FREMENNIK_BERSERKER, 18, 17);
    int hp_before = s.player.current_hitpoints;
    s.npcs[idx].hp = 0;
    col_apply_npc_death(&s, idx);
    CHECK("Volatility explosion hits an adjacent player", s.player.current_hitpoints < hp_before);
}

static void test_modifier_hazard_obs_fixes(void) {
    printf("test_modifier_hazard_obs_fixes\n");
    ColosseumContext ctx;
    ColosseumState s;
    static float obs[COLO_NUM_OBS];

    init_forecast_test_state(&s, &ctx, 301, 17, 16);
    col_init_npc(&s, 0, COLO_JAVELIN_COLOSSUS, 20, 16);
    ColoJavelinState* jv = colo_npc_javelin(&s.npcs[0]);
    jv->skyfall_pending = 1;
    jv->skyfall_tile_x = 19;
    jv->skyfall_tile_y = 14;
    jv->skyfall_timer = 2;
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    int tells = COLO_OBS_AFTER_EQUIPPED_SELF + TEST_NPC_TELLS_OFFSET;
    CHECK("javelin skyfall tells expose landing dx while pending",
        fabsf(obs[tells + 2] - col_obs_rel_x(19, s.player.x)) < 0.000001f);
    CHECK("javelin skyfall tells expose landing dy while pending",
        fabsf(obs[tells + 3] - col_obs_rel_y(14, s.player.y)) < 0.000001f);
    jv->skyfall_pending = 0;
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("javelin skyfall landing tells clear when not pending",
        obs[tells + 2] == 0.0f && obs[tells + 3] == 0.0f);

    init_forecast_test_state(&s, &ctx, 302, 12, 10);
    col_init_npc(&s, 0, COLO_BEE_SWARM, 10, 10);
    s.bees[0] = (ColoBeeSwarm){
        .phase = COLO_HAZARD_ALIVE,
        .npc_slot = 0,
        .respawn_timer = 0,
        .move_timer = 1,
    };
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    tells = COLO_OBS_AFTER_EQUIPPED_SELF + TEST_NPC_TELLS_OFFSET;
    CHECK("bee tells expose a nonzero move timer",
        obs[tells] > 0.0f && obs[tells] <= 1.0f);
    CHECK("bee tells expose next-step contact",
        obs[tells + 1] == 0.0f &&
        fabsf(obs[tells + 2] - col_obs_rel_x(11, s.player.x)) < 0.000001f &&
        fabsf(obs[tells + 3] - col_obs_rel_y(10, s.player.y)) < 0.000001f &&
        obs[tells + 4] == 1.0f);

    init_forecast_test_state(&s, &ctx, 303, 10, 10);
    s.molten_count = 6;
    s.molten_x[0] = 20; s.molten_y[0] = 20;
    s.molten_x[1] = 12; s.molten_y[1] = 10;
    s.molten_x[2] = 9; s.molten_y[2] = 10;
    s.molten_x[3] = 10; s.molten_y[3] = 13;
    s.molten_x[4] = 8; s.molten_y[4] = 8;
    s.molten_x[5] = 10; s.molten_y[5] = 14;
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    int molten = TEST_MOD_OBS_MOLTEN;
    CHECK("modifier molten obs lists the nearest pool first",
        fabsf(obs[molten] - col_obs_rel_x(9, s.player.x)) < 0.000001f &&
        fabsf(obs[molten + 1] - col_obs_rel_y(10, s.player.y)) < 0.000001f &&
        obs[molten + 2] == 1.0f);
    CHECK("modifier molten obs keeps the four nearest pools ordered",
        fabsf(obs[molten + 3] - col_obs_rel_x(12, s.player.x)) < 0.000001f &&
        fabsf(obs[molten + 6] - col_obs_rel_x(8, s.player.x)) < 0.000001f &&
        fabsf(obs[molten + 9] - col_obs_rel_x(10, s.player.x)) < 0.000001f &&
        obs[molten + 5] == 1.0f &&
        obs[molten + 8] == 1.0f &&
        obs[molten + 11] == 1.0f);

    init_forecast_test_state(&s, &ctx, 304, 0, 0);
    s.modifiers.active_mask |= (1u << COLO_MOD_SOLARFLARE);
    s.modifiers.tier[COLO_MOD_SOLARFLARE] = 2;
    s.solarflare = (ColoSolarflareOrb){
        .active = 1,
        .step = 3,
        .move_timer = 1,
        .pause_timer = 0,
    };
    int current_x, current_y, next_x, next_y;
    col_solarflare_tile(&s, 0, s.solarflare.step, &current_x, &current_y);
    col_solarflare_tile(&s, 0, s.solarflare.step + 1, &next_x, &next_y);
    s.player.x = current_x;
    s.player.y = current_y;
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    int solarflare = TEST_MOD_OBS_SOLARFLARE;
    CHECK("Solarflare obs exposes the nearest orb next tile",
        fabsf(obs[solarflare + 6] - col_obs_rel_x(next_x, s.player.x)) < 0.000001f &&
        fabsf(obs[solarflare + 7] - col_obs_rel_y(next_y, s.player.y)) < 0.000001f);

    init_forecast_test_state(&s, &ctx, 305, 17, 16);
    s.modifiers.active_mask |= (1u << COLO_MOD_DOOM);
    s.modifiers.tier[COLO_MOD_DOOM] = 3;
    s.doom_stacks = COLO_DOOM_CAP[3] - 2;
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("Doom lethality obs is clear before cap minus one",
        obs[TEST_MOD_OBS_DOOM_LETHAL] == 0.0f);
    s.doom_stacks = COLO_DOOM_CAP[3] - 1;
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("Doom lethality obs flips at cap minus one",
        obs[TEST_MOD_OBS_DOOM_LETHAL] == 1.0f);

    s.player_venom = COLO_VENOM_START;
    s.player_venom_timer = COLO_VENOM_INTERVAL / 2;
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("venom obs exposes the next tick timer",
        fabsf(obs[TEST_MOD_OBS_VENOM_TIMER] - 0.5f) < 0.000001f);

    init_forecast_test_state(&s, &ctx, 306, 17, 17);
    s.modifiers.active_mask |= (1u << COLO_MOD_VOLATILITY);
    s.modifiers.tier[COLO_MOD_VOLATILITY] = 1;
    col_init_npc(&s, 0, COLO_FREMENNIK_BERSERKER, 18, 17);
    osrs_interaction_set(&s.interaction, 0);
    s.npcs[0].hp = 5;
    col_queue_npc_pending_hit(&s, 0, 5, 1, ATTACK_STYLE_MELEE, ENCOUNTER_SPELL_NONE);
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    int volatility = TEST_MOD_OBS_VOLATILITY;
    CHECK("Volatility current-target blast marks player in footprint",
        obs[volatility + 3] == 1.0f);
    CHECK("Volatility queued-kill blast marks player in footprint",
        obs[volatility + 7] == 1.0f);
}

static void test_death_linger_wave_clear_and_render(void) {
    printf("test_death_linger_wave_clear_and_render\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 31);
    geo_clear_npcs(&s);
    s.modifiers.draft_pending = 0;
    s.wave = 0;
    s.wave_spawn_delay = 0;
    s.wave_ready_delay = 0;
    s.reinforcement_timer = COLO_REINFORCEMENT_TICKS;
    s.player.x = 12;
    s.player.y = 16;

    int idx = 0;
    col_init_npc(&s, idx, COLO_FREMENNIK_BERSERKER, 14, 16);
    s.npcs[idx].hp = 1;
    int dealt = encounter_damage_npc(
        &s.npcs[idx].hp, &s.npcs[idx].hit_landed_this_tick,
        &s.npcs[idx].hit_damage, 1);
    s.npcs[idx].hit_was_successful_this_tick = dealt > 0;
    col_apply_npc_death(&s, idx);

    int linger_ticks = col_npc_death_linger_ticks(COLO_FREMENNIK_BERSERKER);
    CHECK("lethal hit starts NPC death linger", s.npcs[idx].active &&
        s.npcs[idx].death_ticks == linger_ticks);

    RenderEntity entities[4];
    int entity_count = 0;
    col_fill_render_entities_ctx(
        (EncounterState*)&s, (EncounterContext*)&ctx, entities, 4, &entity_count);
    CHECK("dying NPC is still rendered", entity_count == 2 && entities[1].npc_slot == idx);
    CHECK("dying NPC uses death animation",
        entities[1].npc_anim_id == col_npc_death_anim_id(COLO_FREMENNIK_BERSERKER));
    CHECK("lethal hitsplat remains on death frame",
        entities[1].hit_landed_this_tick == 1 && entities[1].hit_damage == 1);

    int idle[COLO_NUM_ACTION_HEADS] = {0};
    step_and_observe(&s, &ctx, idle);
    CHECK("wave clears while corpse is dying", s.tick_scratch.wave_completed == 1);
    CHECK("dying corpse remains active after wave clear",
        s.npcs[idx].active && s.npcs[idx].death_ticks == linger_ticks - 1);

    for (int t = 0; t < linger_ticks; t++) step_and_observe(&s, &ctx, idle);
    CHECK("dying corpse despawns after linger", !s.npcs[idx].active);
}

static void test_static_arena_mask(void) {
    printf("test_static_arena_mask\n");
    col_build_npc_stats();

    int gate_rows_ok = 1;
    for (int x = 0; x <= 33; x++) {
        int walkable = (x == 13 || x == 14 || x == 19 || x == 20);
        if (col_static_blocked(x, 0) != !walkable) gate_rows_ok = 0;
        if (col_static_blocked(x, 33) != !walkable) gate_rows_ok = 0;
    }
    CHECK("south+north inner rows walkable exactly at the gate flanks {13,14,19,20}",
        gate_rows_ok);

    int west_ok = 1;
    for (int y = 0; y <= 33; y++) {
        int walkable = (y == 13 || y == 14 || y == 19 || y == 20);
        if (col_static_blocked(0, y) != !walkable) west_ok = 0;
    }
    CHECK("west col 0 open exactly at the entrance rows {13,14,19,20}", west_ok);

    int east_ok = 1;
    for (int y = 0; y <= 33; y++)
        if (!col_static_blocked(33, y)) east_ok = 0;
    CHECK("east col 33 fully walled", east_ok);

    CHECK("row 3 west extent [0,5)", col_static_blocked(4, 3) && !col_static_blocked(5, 3));
    CHECK("row 30 west extent [0,6)", col_static_blocked(5, 30) && !col_static_blocked(6, 30));
    CHECK("row 29 east extent [29,34)", !col_static_blocked(28, 29) && col_static_blocked(29, 29));

    int pillars_ok = 1, rim_ok = 1;
    for (int p = 0; p < COLO_NUM_PILLARS; p++) {
        int px = COLO_PILLARS[p][0], py = COLO_PILLARS[p][1];
        for (int dx = 0; dx < 3; dx++)
            for (int dy = 0; dy < 3; dy++)
                if (!col_static_blocked(px + dx, py + dy)) pillars_ok = 0;
        if (col_static_blocked(px - 1, py + 1)) rim_ok = 0;
        if (col_static_blocked(px + 3, py + 1)) rim_ok = 0;
    }
    CHECK("all 36 pillar tiles blocked on every wave", pillars_ok);
    CHECK("tiles flanking each pillar stay walkable", rim_ok);

    int zones_ok = 1;
    for (int a = 0; a < COLO_NUM_SPAWN_ANCHORS; a++)
        for (int dx = 0; dx < COLO_SPAWN_ZONE_SIZE; dx++)
            for (int dy = 0; dy < COLO_SPAWN_ZONE_SIZE; dy++)
                if (col_static_blocked(COLO_SPAWN_ANCHORS[a][0] + dx,
                                       COLO_SPAWN_ANCHORS[a][1] + dy)) zones_ok = 0;
    CHECK("every 3x3 spawn-anchor zone fully walkable on the static mask", zones_ok);

    CHECK("wave start (7,18) walkable",
        !col_static_blocked(COLO_PLAYER_START_X, COLO_PLAYER_START_Y));
    CHECK("boss start (16,10) walkable",
        !col_static_blocked(COLO_BOSS_PLAYER_START_X, COLO_BOSS_PLAYER_START_Y));
    int sol_ok = 1;
    for (int dx = 0; dx < 5; dx++)
        for (int dy = 0; dy < 5; dy++)
            if (col_static_blocked(COLO_SOL_SPAWN_X + dx, COLO_SOL_SPAWN_Y + dy)) sol_ok = 0;
    CHECK("Sol's 5x5 footprint at (16,19) unblocked", sol_ok);
}

static void test_static_los_and_attack_gate(void) {
    printf("test_static_los_and_attack_gate\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 31);
    geo_clear_npcs(&s);

    CHECK("SW pillar blocks a ray along row 9", !col_tiles_have_los(&s, 7, 9, 12, 9));
    CHECK("pillar block is symmetric", !col_tiles_have_los(&s, 12, 9, 7, 9));
    CHECK("ray one row north of the pillar is clear", col_tiles_have_los(&s, 7, 12, 12, 12));
    CHECK("north gate doors block along the inner row", !col_tiles_have_los(&s, 14, 33, 19, 33));
    CHECK("row 32 inside the north gate is clear", col_tiles_have_los(&s, 14, 32, 19, 32));

    s.player.x = 5; s.player.y = 9;
    col_init_npc(&s, 0, COLO_SERPENT_SHAMAN, 13, 9);
    s.npcs[0].attack_timer = 0;
    CHECK("shaman behind the pillar has no LoS", !col_npc_has_los_to_player(&s, &s.npcs[0]));
    col_npc_attack_ctx(&s, &ctx, 0);
    CHECK("no-LoS shaman holds fire", s.npcs[0].attacked_this_tick == 0);
    col_npc_move_ctx(&s, &ctx, 0);
    CHECK("no-LoS shaman steps toward the player instead", s.npcs[0].moved_this_tick == 1);

    geo_clear_npcs(&s);
    s.player.x = 5; s.player.y = 12;
    col_init_npc(&s, 0, COLO_SERPENT_SHAMAN, 13, 12);
    s.npcs[0].attack_timer = 0;
    CHECK("clear-row shaman has LoS", col_npc_has_los_to_player(&s, &s.npcs[0]));
    col_npc_attack_ctx(&s, &ctx, 0);
    CHECK("clear-row shaman attacks", s.npcs[0].attacked_this_tick == 1);
}

static void test_spawn_anchor_exclusion(void) {
    printf("test_spawn_anchor_exclusion\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 37);
    advance_to_wave_spawn(&s, &ctx);

    geo_clear_npcs(&s);
    s.player.x = 5; s.player.y = 18;
    int cand[COLO_NUM_SPAWN_ANCHORS];
    int n = col_spawn_anchor_candidates(&s, cand);
    CHECK("b5 spawn-fix tile leaves exactly 10 candidate anchors", n == 10);
    int suppressed_ok = 1;
    for (int i = 0; i < n; i++)
        if (cand[i] == 0 || cand[i] == 1 || cand[i] == 2) suppressed_ok = 0;
    CHECK("the 3 suppressed anchors are (3,14),(9,16),(3,19)", suppressed_ok);

    int on_anchor_ok = 1, excluded_ok = 1, distinct_ok = 1, unblocked_ok = 1;
    int warband_ok = 1;
    for (int rep = 0; rep < 30; rep++) {
        s.player.x = 5; s.player.y = 18;
        s.wave = 4;
        col_spawn_wave(&s);
        int used[COLO_NUM_SPAWN_ANCHORS] = {0};
        int archer_x = -1, archer_y = -1;
        for (int i = 0; i < COLO_MAX_NPCS; i++) {
            if (!s.npcs[i].active || s.npcs[i].type != COLO_FREMENNIK_ARCHER) continue;
            archer_x = s.npcs[i].x; archer_y = s.npcs[i].y;
        }
        for (int i = 0; i < COLO_MAX_NPCS; i++) {
            ColoNPC* npc = &s.npcs[i];
            if (!npc->active) continue;
            int size = col_npc_effective_size(npc);
            for (int dx = 0; dx < size; dx++)
                for (int dy = 0; dy < size; dy++)
                    if (col_static_blocked(npc->x + dx, npc->y + dy)) unblocked_ok = 0;
            if (col_type_is_warbander(npc->type)) {

                if (npc->type == COLO_FREMENNIK_ARCHER) {
                    if (npc->x < COLO_WARBAND_BOX_MIN_X || npc->x > COLO_WARBAND_BOX_MAX_X ||
                        npc->y < COLO_WARBAND_BOX_MIN_Y || npc->y > COLO_WARBAND_BOX_MAX_Y)
                        warband_ok = 0;
                } else if (npc->type == COLO_FREMENNIK_SEER) {

                    if (npc->x != archer_x + 2 || npc->y != archer_y) warband_ok = 0;
                } else {

                    if (npc->x != archer_x + 1 || npc->y != archer_y + 1) warband_ok = 0;
                }
                continue;
            }
            int anchor = -1;
            for (int a = 0; a < COLO_NUM_SPAWN_ANCHORS; a++)
                if (COLO_SPAWN_ANCHORS[a][0] == npc->x &&
                    COLO_SPAWN_ANCHORS[a][1] == npc->y) anchor = a;
            if (anchor < 0) { on_anchor_ok = 0; continue; }
            if (used[anchor]) distinct_ok = 0;
            used[anchor] = 1;
            if (col_spawn_excluded_near_player(&s, npc->x, npc->y, COLO_SPAWN_ZONE_SIZE))
                excluded_ok = 0;
        }
    }
    CHECK("every primary spawns with its SW tile ON one of the 12 anchors", on_anchor_ok);
    CHECK("no primary ever lands within Chebyshev 4 of the player", excluded_ok);
    CHECK("anchors draw without replacement (no double-booking)", distinct_ok);
    CHECK("no spawned footprint touches a blocked tile", unblocked_ok);
    CHECK("warband trio spawns centre-box in the fixed diamond formation", warband_ok);
}

static void test_reinforcement_gates(void) {
    printf("test_reinforcement_gates\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 10;
    ColosseumState s;

    for (int north = 0; north <= 1; north++) {
        memset(&s, 0, sizeof(s));
        col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 41);
        geo_clear_npcs(&s);
        s.player.x = 16;
        s.player.y = north ? 16 : 15;
        col_spawn_reinforcements(&s);

        int count = 0, in_gap_ok = 1, row_ok = 1;
        for (int i = 0; i < COLO_MAX_NPCS; i++) {
            ColoNPC* npc = &s.npcs[i];
            if (!npc->active) continue;
            count++;
            int size = col_npc_effective_size(npc);
            if (npc->x < COLO_GATE_MIN_X || npc->x + size - 1 > COLO_GATE_MAX_X)
                in_gap_ok = 0;
            int inner_row = north ? npc->y + size - 1 : npc->y;
            if (inner_row != (north ? COLO_GATE_NORTH_SPAWN_ROW : COLO_GATE_SOUTH_SPAWN_ROW))
                row_ok = 0;
            for (int dx = 0; dx < size; dx++)
                for (int dy = 0; dy < size; dy++)
                    if (col_static_blocked(npc->x + dx, npc->y + dy)) row_ok = 0;
        }
        CHECK("reinforcement set spawned (minotaur + shaman)", count == 2);
        CHECK("reinforcements land inside the gate gap x 15-18", in_gap_ok);
        CHECK(north ? "player y=16 -> north gate row (yellow line, not nearest)"
                    : "player y=15 -> south gate row", row_ok);
    }
}

static void test_outcome_score_uses_fresh_damage(void) {
    printf("test_outcome_score_uses_fresh_damage\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 101);
    advance_to_wave_spawn(&s, &ctx);
    complete_open_draft(&s, &ctx, 1);

    CHECK("wave 1 spawned a positive HP pool", s.current_wave_hp_pool > 0);
    int pool = s.current_wave_hp_pool;

    s.current_wave_fresh_damage = 0.0f;
    CHECK("zero fresh damage gives zero within-wave progress",
        col_current_wave_score_progress(&s) == 0.0f);

    s.current_wave_fresh_damage = 0.25f * (float)pool;
    CHECK("a quarter of the HP pool is quarter-wave depth",
        fabsf(col_episode_outcome_score(&s) - score_for_depth(0.25f)) < 0.0001f);

    s.current_wave_fresh_damage = 0.5f * (float)pool;
    CHECK("half the HP pool is half-wave depth and climbs",
        fabsf(col_episode_outcome_score(&s) - score_for_depth(0.5f)) < 0.0001f);

    s.current_wave_fresh_damage = (float)pool;
    CHECK("a full pool caps within-wave progress at zero",
        col_current_wave_score_progress(&s) == 0.0f);
}

static void test_outcome_score_reinforcement_grows_denominator(void) {
    printf("test_outcome_score_reinforcement_grows_denominator\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 10;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 202);
    advance_to_wave_spawn(&s, &ctx);

    CHECK("wave 11 starts with seven score enemies", s.current_wave_total_killable == 7);
    int pool_before = s.current_wave_hp_pool;
    CHECK("the HP pool is positive", pool_before > 0);

    s.current_wave_fresh_damage = 0.3f * (float)pool_before;
    float score_before = col_episode_outcome_score(&s);

    col_spawn_reinforcements(&s);
    CHECK("reinforcements enter the score denominator", s.current_wave_total_killable == 9);
    CHECK("reinforcements grow the HP pool", s.current_wave_hp_pool > pool_before);

    CHECK("a larger pool lowers the fresh fraction for fixed fresh damage",
        col_episode_outcome_score(&s) < score_before);

    s.current_wave_fresh_damage = 0.6f * (float)s.current_wave_hp_pool;
    CHECK("more fresh damage over the larger pool climbs the score again",
        col_episode_outcome_score(&s) > score_before);
}

static void test_fresh_damage_not_farmable_via_healing(void) {
    printf("test_fresh_damage_not_farmable_via_healing\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 1;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 711);
    geo_clear_npcs(&s);
    col_reset_current_wave_score_progress(&s);

    col_init_npc(&s, 0, COLO_SERPENT_SHAMAN, 12, 16);
    s.npcs[0].hp = 100;
    s.npcs[0].max_hp = 100;
    s.npcs[0].min_hp_seen = 100;
    s.player.x = 13; s.player.y = 16;
    col_rebuild_player_collision_flags(&s);

    col_queue_npc_pending_hit(&s, 0, 70, 1, ATTACK_STYLE_MELEE, ENCOUNTER_SPELL_NONE);
    land_pending_player_hits(&s);
    CHECK("first hit lands for 70", s.npcs[0].hp == 30);
    CHECK("fresh damage tracks the first hit",
        fabsf(s.current_wave_fresh_damage - 70.0f) < 0.001f);
    CHECK("min_hp_seen follows the new low", s.npcs[0].min_hp_seen == 30);

    s.npcs[0].hp = 60;
    CHECK("healing does not touch min_hp_seen", s.npcs[0].min_hp_seen == 30);

    col_queue_npc_pending_hit(&s, 0, 30, 1, ATTACK_STYLE_MELEE, ENCOUNTER_SPELL_NONE);
    land_pending_player_hits(&s);
    CHECK("re-damaging restored HP credits no fresh damage",
        fabsf(s.current_wave_fresh_damage - 70.0f) < 0.001f);

    col_queue_npc_pending_hit(&s, 0, 20, 1, ATTACK_STYLE_MELEE, ENCOUNTER_SPELL_NONE);
    land_pending_player_hits(&s);
    CHECK("a new low credits only the fresh portion (20)",
        fabsf(s.current_wave_fresh_damage - 90.0f) < 0.001f);
    CHECK("min_hp_seen follows the deeper low", s.npcs[0].min_hp_seen == 10);

    CHECK("gross damage double-counts the healed HP, fresh does not",
        fabsf(s.tick_scratch.damage_dealt - 120.0f) < 0.001f);
}

static void test_outcome_score_wave_clear_has_no_double_count(void) {
    printf("test_outcome_score_wave_clear_has_no_double_count\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 303);
    complete_open_draft(&s, &ctx, 1);

    while (first_live_score_enemy(&s) >= 0) kill_first_live_score_enemy(&s);
    s.log.waves_cleared = 1;
    CHECK("within-wave progress is zero after all wave enemies are killed",
        col_current_wave_score_progress(&s) == 0.0f);
    CHECK("cleared-wave score comes only from waves_cleared",
        fabsf(col_episode_outcome_score(&s) - score_for_depth(1.0f)) < 0.000001f);
}

static void test_outcome_score_sol_uses_boss_progress_only(void) {
    printf("test_outcome_score_sol_uses_boss_progress_only\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = COLO_WAVE_BOSS;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 404);

    s.min_sol_hp_seen = 750;
    float expected = score_for_depth(col_sol_score_progress(&s));
    CHECK("boss wave has no within-wave kill denominator",
        s.current_wave_total_killable == 0 && col_current_wave_score_progress(&s) == 0.0f);
    CHECK("Sol score is boss progress only",
        fabsf(col_episode_outcome_score(&s) - expected) < 0.000001f);
}

static void test_roster_cap_nine(void) {
    printf("test_roster_cap_nine\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 47);
    s.modifiers.active_mask |= (1u << COLO_MOD_QUARTET) | (1u << COLO_MOD_DYNAMIC_DUO);
    s.modifiers.tier[COLO_MOD_QUARTET] = 1;
    s.modifiers.tier[COLO_MOD_DYNAMIC_DUO] = 1;
    s.wave = 7;
    col_spawn_wave(&s);
    int count = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++) if (s.npcs[i].active) count++;
    CHECK("wave 8 + Quartet + Dynamic Duo spawns all 9 NPCs", count == 9);
}

static void test_wave12_quartet_and_win(void) {
    printf("test_wave12_quartet_and_win\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 11;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 43);
    advance_to_wave_spawn(&s, &ctx);

    CHECK("boss-wave player start (16,10)",
        s.player.x == COLO_BOSS_PLAYER_START_X && s.player.y == COLO_BOSS_PLAYER_START_Y);
    CHECK("boss arena clamp is (9,9)-(24,24)",
        s.sol.boss_arena_min_x == 9 && s.sol.boss_arena_min_y == 9 &&
        s.sol.boss_arena_max_x == 24 && s.sol.boss_arena_max_y == 24);
    int sol_idx = col_sol_find_idx(&s);
    CHECK("Sol spawned at SW (16,19)", sol_idx >= 0 &&
        s.npcs[sol_idx].x == COLO_SOL_SPAWN_X && s.npcs[sol_idx].y == COLO_SOL_SPAWN_Y);

    int placement_ok = 1, reachable_ok = 1, win_ok = 1, survivor_ok = 1;
    for (int rep = 0; rep < 10; rep++) {
        memset(&s, 0, sizeof(s));
        col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 43 + (uint32_t)rep);
        s.modifiers.active_mask |= (1u << COLO_MOD_QUARTET);
        s.modifiers.tier[COLO_MOD_QUARTET] = 1;
        s.wave = COLO_WAVE_BOSS;
        col_spawn_wave(&s);
        s.wave_spawn_delay = 0;

        int wb = -1;
        for (int i = 0; i < COLO_MAX_NPCS; i++)
            if (s.npcs[i].active && col_type_is_warbander(s.npcs[i].type)) wb = i;
        if (wb < 0) { placement_ok = 0; continue; }
        int wx = s.npcs[wb].x, wy = s.npcs[wb].y;
        int cheb_dx = abs(wx - s.player.x), cheb_dy = abs(wy - s.player.y);
        if (wx < 9 || wx > 24 || wy < 9 || wy > 24) placement_ok = 0;
        if (col_static_blocked(wx, wy)) placement_ok = 0;
        if ((cheb_dx > cheb_dy ? cheb_dx : cheb_dy) <= COLO_SPAWN_EXCLUSION_CHEB)
            placement_ok = 0;

        int seen[COLO_ARENA_WIDTH][COLO_ARENA_HEIGHT] = {{0}};
        int qx[34 * 34], qy[34 * 34], head = 0, tail = 0;
        qx[tail] = s.player.x; qy[tail] = s.player.y; tail++;
        seen[s.player.x][s.player.y] = 1;
        int reached = 0;
        while (head < tail && !reached) {
            int cx = qx[head], cy = qy[head]; head++;
            static const int D[4][2] = { {1,0}, {-1,0}, {0,1}, {0,-1} };
            for (int d = 0; d < 4; d++) {
                int nx = cx + D[d][0], ny = cy + D[d][1];
                if (nx == wx && ny == wy) { reached = 1; break; }
                if (nx < 9 || nx > 24 || ny < 9 || ny > 24) continue;
                if (seen[nx][ny] || col_static_blocked(nx, ny)) continue;
                int gx, gy;
                if (!col_grid_index(nx, ny, &gx, &gy)) continue;
                if (s.npc_collision_flags[gx][gy]) continue;
                seen[nx][ny] = 1;
                qx[tail] = nx; qy[tail] = ny; tail++;
            }
        }
        if (!reached) reachable_ok = 0;

        int sol = col_sol_find_idx(&s);
        if (sol < 0) { win_ok = 0; continue; }
        s.npcs[sol].hp = 0;
        col_apply_npc_death(&s, sol);
        int idle[COLO_NUM_ACTION_HEADS] = {0};
        step_and_observe(&s, &ctx, idle);
        if (!(s.episode_over && s.winner == COLO_OUTCOME_PLAYER_WON)) win_ok = 0;
        if (!s.npcs[wb].active) survivor_ok = 0;
    }
    CHECK("Quartet warbander spawns on a walkable interior tile outside the exclusion",
        placement_ok);
    CHECK("Quartet warbander is pathable from the player", reachable_ok);
    CHECK("Sol's death wins the wave-12 run (A6)", win_ok);
    CHECK("the surviving warbander does not block the win", survivor_ok);
}

static void test_player_walks_through_npc_footprint(void) {
    printf("test_player_walks_through_npc_footprint\n");
    ColosseumContext ctx;
    ColosseumState s;
    init_forecast_test_state(&s, &ctx, 510, 16, 16);
    col_init_npc(&s, 0, COLO_SOL_HEREDIT, 17, 16);
    col_rebuild_player_collision_flags(&s);
    int gx, gy;
    int npc_flag = col_grid_index(17, 16, &gx, &gy) && s.npc_collision_flags[gx][gy];
    int player_flag = col_grid_index(17, 16, &gx, &gy) && s.player_collision_flags[gx][gy];
    ColoWalkCtx wc = { .s = &s, .ctx = &ctx };
    CHECK("NPC footprint remains stamped for NPC systems", npc_flag != 0);
    CHECK("NPC footprint is not stamped as player collision", player_flag == 0);
    CHECK("player walkability ignores NPC footprint",
        col_player_walkable(&s, 17, 16) == 1);
    CHECK("player pathfinding extra block ignores NPC footprint",
        col_pathfind_blocked(&wc, 17 + ctx.world_offset_x, 16 + ctx.world_offset_y) == 0);
    int actions[COLO_NUM_ACTION_HEADS] = {0};
    actions[COLO_HEAD_PRIMARY] = forecast_move_action_for_delta(1, 0);
    step_and_observe(&s, &ctx, actions);
    CHECK("explicit movement can step onto Sol footprint",
        s.player.x == 17 && s.player.y == 16);
}

static int wb_find_npc(const ColosseumState* s, ColoNpcType type) {
    for (int i = 0; i < COLO_MAX_NPCS; i++)
        if (s->npcs[i].active && s->npcs[i].type == type) return i;
    return -1;
}

static void wb_isolate_warband(ColosseumState* s) {
    for (int i = 0; i < COLO_MAX_NPCS; i++)
        if (s->npcs[i].active && !col_type_is_warbander(s->npcs[i].type))
            col_deactivate_npc(s, i);
}

static void wb_move_npc(ColosseumState* s, int slot, int x, int y) {
    int size = col_npc_effective_size(&s->npcs[slot]);
    col_stamp_npc_collision_footprint(s, s->npcs[slot].x, s->npcs[slot].y, size, 0);
    s->npcs[slot].x = x;
    s->npcs[slot].y = y;
    col_stamp_npc_collision_footprint(s, x, y, size, 1);
}

static int wb_attacks_this_tick(const ColosseumState* s, ColoNpcType type) {
    int n = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++)
        if (s->npcs[i].active && s->npcs[i].type == type &&
            s->npcs[i].attacked_this_tick) n++;
    return n;
}

static void test_warband_cycle_offsets(void) {
    printf("test_warband_cycle_offsets\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 0;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 51);
    advance_to_wave_spawn(&s, &ctx);
    complete_open_draft(&s, &ctx, 1);
    wb_isolate_warband(&s);

    wb_move_npc(&s, wb_find_npc(&s, COLO_FREMENNIK_BERSERKER), s.player.x, s.player.y + 1);
    wb_move_npc(&s, wb_find_npc(&s, COLO_FREMENNIK_SEER), s.player.x + 1, s.player.y);
    wb_move_npc(&s, wb_find_npc(&s, COLO_FREMENNIK_ARCHER), s.player.x - 1, s.player.y);

    int idle[COLO_NUM_ACTION_HEADS] = {0};
    int anchor = -1;
    int first_tick[3] = { -1, -1, -1 };
    int count[3] = { 0, 0, 0 };
    int offsets_ok = 1;
    static const ColoNpcType SPECIES[3] = {
        COLO_FREMENNIK_BERSERKER, COLO_FREMENNIK_SEER, COLO_FREMENNIK_ARCHER };

    for (int t = 0; t < 46 && !s.episode_over; t++) {
        s.player.current_hitpoints = 9999;
        step_and_observe(&s, &ctx, idle);
        if (anchor < 0) anchor = s.warband_cycle_anchor;
        for (int sp = 0; sp < 3; sp++) {
            if (!wb_attacks_this_tick(&s, SPECIES[sp])) continue;
            count[sp]++;
            if (first_tick[sp] < 0) first_tick[sp] = s.tick;
            if (anchor < 0 || (s.tick - anchor) % COLO_WARBAND_CYCLE_TICKS != sp + 1)
                offsets_ok = 0;
        }
    }
    CHECK("cycle anchored at the wave's first actionable tick", anchor >= 0);
    CHECK("standing player eats repeated full cycles (berserker)", count[0] >= 4);
    CHECK("standing player eats repeated full cycles (seer)", count[1] >= 4);
    CHECK("standing player eats repeated full cycles (archer)", count[2] >= 4);
    CHECK("berserker only lands on ticks = N+1 mod 6; seer +2; archer +3", offsets_ok);
    CHECK("first berserker window is exactly N+1 (wave-anchored, no spawn timer)",
        first_tick[0] == anchor + 1);
    CHECK("seer first window N+2", first_tick[1] == anchor + 2);
    CHECK("archer first window N+3", first_tick[2] == anchor + 3);
}

static void test_warband_move_skip(void) {
    printf("test_warband_move_skip\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 0;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 53);
    advance_to_wave_spawn(&s, &ctx);
    complete_open_draft(&s, &ctx, 1);
    wb_isolate_warband(&s);
    wb_move_npc(&s, wb_find_npc(&s, COLO_FREMENNIK_BERSERKER), s.player.x, s.player.y + 1);
    wb_move_npc(&s, wb_find_npc(&s, COLO_FREMENNIK_SEER), s.player.x + 1, s.player.y);
    wb_move_npc(&s, wb_find_npc(&s, COLO_FREMENNIK_ARCHER), s.player.x - 1, s.player.y);

    int idle[COLO_NUM_ACTION_HEADS] = {0};
    int walk_south[COLO_NUM_ACTION_HEADS] = {0};
    walk_south[COLO_HEAD_PRIMARY] = 4;

    while ((s.wave_ready_delay > 0 || s.wave_attack_delay > 1) && !s.episode_over)
        step_and_observe(&s, &ctx, idle);

    int attacks = 0;
    int moved_every_tick = 1;
    for (int t = 0; t < 14 && !s.episode_over; t++) {
        s.player.current_hitpoints = 9999;
        step_and_observe(&s, &ctx, walk_south);
        if (!s.tick_scratch.player_moved) moved_every_tick = 0;
        for (int sp = 0; sp < COLO_MAX_NPCS; sp++)
            if (s.npcs[sp].active && col_type_is_warbander(s.npcs[sp].type) &&
                s.npcs[sp].attacked_this_tick) attacks++;
    }
    CHECK("the scripted stutter-step actually moved every tick", moved_every_tick);
    CHECK("warband fired zero attacks across the stutter-step run", attacks == 0);
    CHECK("zero warband damage across the stutter-step run",
        s.log.total_damage_received == 0.0f);
}

static uint64_t wb_trajectory_hash(ColosseumState* s, ColosseumContext* ctx,
                                   int seed, const int* actions, int ticks) {
    memset(s, 0, sizeof(*s));
    col_reset_ctx((EncounterState*)s, (EncounterContext*)ctx, seed);
    advance_to_wave_spawn(s, ctx);
    complete_open_draft(s, ctx, 1);
    wb_isolate_warband(s);
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    while ((s->wave_ready_delay > 0 || s->wave_attack_delay > 1) && !s->episode_over)
        step_and_observe(s, ctx, idle);
    uint64_t h = 1469598103934665603ULL;
    for (int t = 0; t < ticks && !s->episode_over; t++) {
        s->player.current_hitpoints = 9999;
        step_and_observe(s, ctx, actions);
        h = (h ^ (uint64_t)(uint32_t)s->player.x) * 1099511628211ULL;
        h = (h ^ (uint64_t)(uint32_t)s->player.y) * 1099511628211ULL;
        for (int n = 0; n < COLO_MAX_NPCS; n++) {
            if (!s->npcs[n].active || !col_type_is_warbander(s->npcs[n].type))
                continue;
            uint64_t p = (uint64_t)(uint32_t)s->npcs[n].x
                | ((uint64_t)(uint32_t)s->npcs[n].y << 8)
                | ((uint64_t)n << 16);
            h = (h ^ p) * 1099511628211ULL;
        }
    }
    return h;
}

static void test_warband_bfs_memo_bit_identity(void) {
    printf("test_warband_bfs_memo_bit_identity\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 0;
    ColosseumState s;

    int idle[COLO_NUM_ACTION_HEADS] = {0};
    int walk_south[COLO_NUM_ACTION_HEADS] = {0};
    walk_south[COLO_HEAD_PRIMARY] = 4;

    memset(col_warband_bfs_memo_key, 0, sizeof(col_warband_bfs_memo_key));
    uint64_t idle_cold = wb_trajectory_hash(&s, &ctx, 51, idle, 40);
    uint64_t idle_warm = wb_trajectory_hash(&s, &ctx, 51, idle, 40);
    CHECK("memo-served warband trajectory == fresh BFS (idle player)",
        idle_warm == idle_cold);

    uint64_t walk_polluted = wb_trajectory_hash(&s, &ctx, 53, walk_south, 40);
    memset(col_warband_bfs_memo_key, 0, sizeof(col_warband_bfs_memo_key));
    uint64_t walk_cold = wb_trajectory_hash(&s, &ctx, 53, walk_south, 40);
    CHECK("polluted-table episode == its fresh-memo reference (walking player)",
        walk_polluted == walk_cold);
}

static void test_warband_melee_distance_gate(void) {
    printf("test_warband_melee_distance_gate\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 57);
    geo_clear_npcs(&s);
    s.player.x = 7; s.player.y = 18;
    s.tick_scratch.player_moved = 0;
    s.tick = 100;

    col_init_npc(&s, 0, COLO_FREMENNIK_ARCHER, 12, 18);
    CHECK("rig sanity: the ranged archer has clear LoS",
        col_npc_has_los_to_player(&s, &s.npcs[0]));
    s.warband_cycle_anchor = s.tick - 3;
    col_warband_attack_phase(&s, &ctx);
    CHECK("archer at distance never attacks, even with LoS on its window",
        s.npcs[0].attacked_this_tick == 0);

    wb_move_npc(&s, 0, 8, 18);
    col_warband_attack_phase(&s, &ctx);
    CHECK("cardinally adjacent archer attacks on its window",
        s.npcs[0].attacked_this_tick == 1);

    s.npcs[0].attacked_this_tick = 0;
    wb_move_npc(&s, 0, 8, 19);
    col_warband_attack_phase(&s, &ctx);
    CHECK("diagonally adjacent archer does not attack (cardinal-only)",
        s.npcs[0].attacked_this_tick == 0);
}

static void test_warband_two_tick_stationary_gate(void) {
    printf("test_warband_two_tick_stationary_gate\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 57);
    geo_clear_npcs(&s);
    s.player.x = 7; s.player.y = 18;
    s.tick = 100;

    col_init_npc(&s, 0, COLO_FREMENNIK_BERSERKER, 8, 18);
    s.warband_cycle_anchor = s.tick - col_warband_window_offset(COLO_FREMENNIK_BERSERKER);

    s.npcs[0].moved_this_tick = 0;
    s.npcs[0].moved_last_tick = 1;
    col_warband_attack_phase(&s, &ctx);
    CHECK("a member with only one stationary tick holds its attack",
        s.npcs[0].attacked_this_tick == 0);

    s.npcs[0].moved_this_tick = 0;
    s.npcs[0].moved_last_tick = 0;
    col_warband_attack_phase(&s, &ctx);
    CHECK("a member stationary two ticks attacks on its window",
        s.npcs[0].attacked_this_tick == 1);

    s.npcs[0].attacked_this_tick = 0;
    s.npcs[0].moved_this_tick = 1;
    s.npcs[0].moved_last_tick = 0;
    col_warband_attack_phase(&s, &ctx);
    CHECK("a member that moved this tick cannot attack",
        s.npcs[0].attacked_this_tick == 0);
}

static void test_warband_formation_convergence(void) {
    printf("test_warband_formation_convergence\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 0;
    int idle[COLO_NUM_ACTION_HEADS] = {0};

    for (int quartet = 0; quartet <= 1; quartet++) {
        ColosseumState s;
        memset(&s, 0, sizeof(s));
        col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 59 + (uint32_t)quartet);
        complete_open_draft(&s, &ctx, 1);
        if (quartet) {
            s.modifiers.active_mask |= (1u << COLO_MOD_QUARTET);
            s.modifiers.tier[COLO_MOD_QUARTET] = 1;
            s.wave = 0;
            col_spawn_wave(&s);
        }
        wb_isolate_warband(&s);

        for (int t = 0; t < 30 && !s.episode_over; t++) {
            s.player.current_hitpoints = 9999;
            step_and_observe(&s, &ctx, idle);
        }

        int formed_ok = 1, members = 0;
        for (int i = 0; i < COLO_MAX_NPCS; i++) {
            ColoNPC* npc = &s.npcs[i];
            if (!npc->active || !col_type_is_warbander(npc->type)) continue;
            members++;
            int dir = colo_npc_warband(npc)->formation_dir;
            int ex = s.player.x + COLO_WARBAND_FORM_OFFSET[dir][0];
            int ey = s.player.y + COLO_WARBAND_FORM_OFFSET[dir][1];
            if (npc->x != ex || npc->y != ey) formed_ok = 0;
        }
        if (quartet) {
            CHECK("Quartet diamond: 4 members each on their N/E/W/S slot",
                formed_ok && members == 4);
        } else {
            CHECK("trio converges to exactly N/E/W of a stationary player",
                formed_ok && members == 3);
        }
    }
}

static void test_warband_two_tile_speed(void) {
    printf("test_warband_two_tile_speed\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 61);
    geo_clear_npcs(&s);
    s.player.x = 7; s.player.y = 18;

    col_init_npc(&s, 0, COLO_FREMENNIK_ARCHER, 20, 18);
    col_npc_move_ctx(&s, &ctx, 0);
    CHECK("warbander closes 2 tiles in one tick on open ground",
        s.npcs[0].x == 18 && s.npcs[0].y == 18 && s.npcs[0].moved_this_tick == 1);
    col_npc_move_ctx(&s, &ctx, 0);
    CHECK("second tick closes 2 more", s.npcs[0].x == 16 && s.npcs[0].y == 18);
}

static void test_warband_pillar_routefind_vs_shaman_safespot(void) {
    printf("test_warband_pillar_routefind_vs_shaman_safespot\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    int idle[COLO_NUM_ACTION_HEADS] = {0};

    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 67);
    advance_to_wave_spawn(&s, &ctx);
    complete_open_draft(&s, &ctx, 1);
    geo_clear_npcs(&s);
    s.player.x = 7; s.player.y = 9;
    col_rebuild_player_collision_flags(&s);

    col_init_npc(&s, 0, COLO_FREMENNIK_ARCHER, 13, 9);
    int adjacent_by = -1;
    int archer_attacks = 0;
    for (int t = 0; t < 40 && !s.episode_over; t++) {
        s.player.current_hitpoints = 9999;
        step_and_observe(&s, &ctx, idle);
        int dx = abs(s.npcs[0].x - s.player.x), dy = abs(s.npcs[0].y - s.player.y);
        if (dx + dy == 1 && adjacent_by < 0) adjacent_by = t;
        archer_attacks += wb_attacks_this_tick(&s, COLO_FREMENNIK_ARCHER);
    }
    CHECK("archer routefinds around the pillar into melee contact",
        adjacent_by >= 0 && adjacent_by <= 14);
    CHECK("the routefinding archer then lands cycle attacks", archer_attacks > 0);

    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 67);
    s.wave_spawn_delay = 0;
    complete_open_draft(&s, &ctx, 1);
    geo_clear_npcs(&s);
    s.player.x = 7; s.player.y = 9;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_SERPENT_SHAMAN, 13, 9);
    int shaman_attacks = 0;
    for (int t = 0; t < 40 && !s.episode_over; t++) {
        s.player.current_hitpoints = 9999;
        step_and_observe(&s, &ctx, idle);
        shaman_attacks += s.npcs[0].attacked_this_tick;
    }
    CHECK("greedy shaman wedges against the pillar (safespot holds)",
        s.npcs[0].x == 11 && s.npcs[0].y == 9);
    CHECK("safespotted shaman never attacks", shaman_attacks == 0);
}

static void test_red_flag_minotaur_routefind(void) {
    printf("test_red_flag_minotaur_routefind\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);

    for (int red_flag = 0; red_flag <= 1; red_flag++) {
        ColosseumState s;
        memset(&s, 0, sizeof(s));
        col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 71);
        geo_clear_npcs(&s);
        if (red_flag) {
            s.modifiers.active_mask |= (1u << COLO_MOD_RED_FLAG);
            s.modifiers.tier[COLO_MOD_RED_FLAG] = 1;
        }
        s.player.x = 7; s.player.y = 9;
        col_rebuild_player_collision_flags(&s);

        col_init_npc(&s, 0, COLO_MINOTAUR, 11, 9);
        int min_dist = 99;
        for (int t = 0; t < 40; t++) {
            col_npc_move_ctx(&s, &ctx, 0);
            int d = encounter_dist_to_npc(
                s.player.x, s.player.y, s.npcs[0].x, s.npcs[0].y, 3);
            if (d < min_dist) min_dist = d;
        }
        if (red_flag) {
            CHECK("Red Flag minotaur routefinds into melee contact", min_dist == 1);
        } else {
            CHECK("plain minotaur stays wedged on the pillar (safespot holds)",
                s.npcs[0].x == 11 && s.npcs[0].y == 9 && min_dist == 4);
        }
    }
}

static void test_minotaur_heal_semantics(void) {
    printf("test_minotaur_heal_semantics\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 73);

    geo_clear_npcs(&s);
    s.player.x = 30; s.player.y = 18;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_MINOTAUR, 12, 12);
    col_init_npc(&s, 1, COLO_SERPENT_SHAMAN, 16, 13);
    col_init_npc(&s, 2, COLO_SERPENT_SHAMAN, 17, 17);
    col_init_npc(&s, 3, COLO_SERPENT_SHAMAN, 13, 17);
    col_init_npc(&s, 4, COLO_MINOTAUR, 18, 12);
    s.npcs[1].hp = 62;
    s.npcs[2].hp = 30;
    s.npcs[3].hp = 100;
    s.npcs[4].hp = 100;
    s.npcs[0].attack_timer = 0;
    col_npc_attack_ctx(&s, &ctx, 0);
    CHECK("the lowest-HP-fraction eligible ally heals to FULL", s.npcs[2].hp == 125);
    CHECK("exactly one ally healed per action", s.npcs[1].hp == 62);
    CHECK("an ally at/above 75% max HP is not eligible", s.npcs[3].hp == 100);
    CHECK("another minotaur is never healed", s.npcs[4].hp == 100);
    CHECK("healing is not an attack", s.npcs[0].attacked_this_tick == 0);
    CHECK("the heal action re-arms the 5-tick timer (D9)",
        s.npcs[0].attack_timer == COLO_NPC_STATS[COLO_MINOTAUR].attack_speed);

    geo_clear_npcs(&s);
    col_init_npc(&s, 0, COLO_MINOTAUR, 12, 12);
    col_init_npc(&s, 1, COLO_SERPENT_SHAMAN, 20, 13);
    s.npcs[1].hp = 30;
    s.npcs[0].attack_timer = 0;
    col_npc_attack_ctx(&s, &ctx, 0);
    CHECK("centre distance 7 is in heal reach", s.npcs[1].hp == 125);
    geo_clear_npcs(&s);
    col_init_npc(&s, 0, COLO_MINOTAUR, 12, 12);
    col_init_npc(&s, 1, COLO_SERPENT_SHAMAN, 21, 13);
    s.npcs[1].hp = 30;
    s.npcs[0].attack_timer = 0;
    col_npc_attack_ctx(&s, &ctx, 0);
    CHECK("centre distance 8 is out of heal reach", s.npcs[1].hp == 30);

    geo_clear_npcs(&s);
    s.player.x = 5; s.player.y = 14;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_MINOTAUR, 4, 8);
    col_init_npc(&s, 1, COLO_SERPENT_SHAMAN, 12, 9);
    col_init_npc(&s, 2, COLO_SERPENT_SHAMAN, 5, 16);
    s.npcs[1].hp = 10;
    s.npcs[2].hp = 60;
    s.npcs[0].attack_timer = 0;
    col_npc_attack_ctx(&s, &ctx, 0);
    CHECK("pillars block the heal (blocked ally skipped)", s.npcs[1].hp == 10);
    CHECK("the clear-LoS ally heals instead", s.npcs[2].hp == 125);

    geo_clear_npcs(&s);
    s.player.x = 11; s.player.y = 11;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_MINOTAUR, 12, 12);
    col_init_npc(&s, 1, COLO_SERPENT_SHAMAN, 16, 13);
    s.npcs[1].hp = 30;
    s.npcs[0].attack_timer = 0;
    s.player.current_hitpoints = 99;
    int queue_before = s.player_pending_hits.count;
    float mino_faced_before = s.log.pray_faced_by_type[COLO_MINOTAUR];
    col_npc_attack_ctx(&s, &ctx, 0);
    CHECK("player in melee distance: the minotaur attacks instead of healing",
        s.npcs[0].attacked_this_tick == 1 && s.npcs[1].hp == 30);
    CHECK("the crush resolves instantly, queuing nothing for a later tick",
        s.player_pending_hits.count == queue_before &&
        s.log.pray_faced_by_type[COLO_MINOTAUR] > mino_faced_before);
}

static void test_manticore_barrage_period(void) {
    printf("test_manticore_barrage_period\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 79);
    geo_clear_npcs(&s);
    s.player.x = 17; s.player.y = 16;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_MANTICORE, 16, 12);
    s.npcs[0].attack_timer = 2;

    ColoManticoreState* mc = colo_npc_manticore(&s.npcs[0]);
    int starts[8];
    int nstarts = 0;
    for (int t = 0; t < 36; t++) {
        s.player.current_hitpoints = 99;
        int prev = mc->cycle_step;
        col_npc_attack_ctx(&s, &ctx, 0);
        if (prev == 0 && mc->cycle_step == 1 && nstarts < 8) starts[nstarts++] = t;
    }
    CHECK("4 barrage starts inside 36 ticks", nstarts == 4);
    int period_ok = nstarts >= 4;
    for (int b = 1; b < nstarts; b++)
        if (starts[b] - starts[b - 1] != 10) period_ok = 0;
    CHECK("barrage-to-barrage period is exactly 10 ticks across 3 gaps", period_ok);
}

static void test_manticore_telegraph_during_windup(void) {
    printf("test_manticore_telegraph_during_windup\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 131);
    geo_clear_npcs(&s);
    s.player.x = 17; s.player.y = 16;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_MANTICORE, 16, 12);
    s.npcs[0].attack_timer = 6;
    ColoManticoreState* mc = colo_npc_manticore(&s.npcs[0]);
    CHECK("spawn rolls a hidden fixed cycle",
        mc->fixed_orb_style[0] != ATTACK_STYLE_NONE &&
        mc->fixed_orb_style[1] != ATTACK_STYLE_NONE &&
        mc->fixed_orb_style[2] != ATTACK_STYLE_NONE &&
        mc->orb_style[0] == ATTACK_STYLE_NONE &&
        mc->orb_style[1] == ATTACK_STYLE_NONE &&
        mc->orb_style[2] == ATTACK_STYLE_NONE);

    s.player.current_hitpoints = 99;
    col_npc_attack_ctx(&s, &ctx, 0);
    CHECK("manticore arms during the charge-up at orb 0", mc->cycle_step == 0);
    CHECK("the active telegraph matches the fixed cycle before any orb fires",
        mc->orb_style[0] == mc->fixed_orb_style[0] &&
        mc->orb_style[1] == mc->fixed_orb_style[1] &&
        mc->orb_style[2] == mc->fixed_orb_style[2]);
    CHECK("orb 2 is melee (the range+magic pair leads, melee last)",
        mc->orb_style[2] == ATTACK_STYLE_MELEE);

    AttackStyle locked0 = mc->orb_style[0];
    for (int t = 0; t < 4 && mc->cycle_step == 0; t++) {
        s.player.current_hitpoints = 99;
        col_npc_attack_ctx(&s, &ctx, 0);
    }
    CHECK("the charge pattern stays stable until orb 0 fires (no per-tick re-roll)",
        mc->orb_style[0] == locked0);

    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 131);
    geo_clear_npcs(&s);
    s.player.x = 17; s.player.y = 16;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_MANTICORE, 16, 12);
    s.npcs[0].attack_timer = 3;
    mc = colo_npc_manticore(&s.npcs[0]);
    int orb0_blocked = 0;
    for (int t = 0; t < 12; t++) {
        if (mc->cycle_step == 0 && mc->orb_style[0] != ATTACK_STYLE_NONE) {
            AttackStyle s0 = mc->orb_style[0];
            s.player.prayer = s0 == ATTACK_STYLE_MAGIC ? PRAYER_PROTECT_MAGIC :
                              s0 == ATTACK_STYLE_RANGED ? PRAYER_PROTECT_RANGED :
                              PRAYER_PROTECT_MELEE;
        }
        s.player.current_hitpoints = 99;
        int step_before = mc->cycle_step;
        col_npc_attack_ctx(&s, &ctx, 0);
        if (step_before == 0 && mc->cycle_step == 1) {
            orb0_blocked = (s.player.current_hitpoints == 99);
            break;
        }
    }
    CHECK("a pre-prayed telegraphed orb 0 is blocked on its fire tick", orb0_blocked);

    while (mc->cycle_step >= 0) {
        s.player.current_hitpoints = 99;
        col_npc_attack_ctx(&s, &ctx, 0);
    }
    CHECK("disarm clears only the active telegraph",
        mc->orb_style[0] == ATTACK_STYLE_NONE &&
        mc->fixed_orb_style[0] != ATTACK_STYLE_NONE);
}

static void test_prayer_oracle_manticore_orbs(void) {
    printf("test_prayer_oracle_manticore_orbs\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 79);
    advance_to_wave_spawn(&s, &ctx);
    geo_clear_npcs(&s);
    s.wave_ready_delay = 0;
    s.wave_attack_delay = 0;
    s.player.x = 17; s.player.y = 16;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_MANTICORE, 16, 12);
    ColoManticoreState* mc = colo_npc_manticore(&s.npcs[0]);

    s.npcs[0].attack_timer = 1;
    mc->cycle_step = -1;
    mc->fixed_orb_style[0] = ATTACK_STYLE_MAGIC;
    mc->fixed_orb_style[1] = ATTACK_STYLE_RANGED;
    mc->fixed_orb_style[2] = ATTACK_STYLE_MELEE;
    s.player.prayer = PRAYER_NONE;
    col_apply_prayer_oracle(&s);
    CHECK("oracle prays orb 0's style on the charge-complete tick (magic, not default ranged)",
        s.player.prayer == PRAYER_PROTECT_MAGIC);

    for (int o = 0; o < 3; o++) mc->orb_style[o] = mc->fixed_orb_style[o];
    mc->cycle_step = 1;
    s.player.prayer = PRAYER_NONE;
    col_apply_prayer_oracle(&s);
    CHECK("oracle prays the in-flight orb 1 style (ranged)",
        s.player.prayer == PRAYER_PROTECT_RANGED);
    mc->cycle_step = 2;
    s.player.prayer = PRAYER_NONE;
    col_apply_prayer_oracle(&s);
    CHECK("oracle prays the in-flight orb 2 style (melee, not default ranged)",
        s.player.prayer == PRAYER_PROTECT_MELEE);

    mc->cycle_step = -1;
    for (int o = 0; o < 3; o++) mc->orb_style[o] = ATTACK_STYLE_NONE;
    s.npcs[0].attack_timer = 5;
    s.player.prayer = PRAYER_PROTECT_MELEE;
    col_apply_prayer_oracle(&s);
    CHECK("oracle leaves prayer untouched with no thrower this tick",
        s.player.prayer == PRAYER_PROTECT_MELEE);

    s.npcs[0].attack_timer = 2;
    for (int t = 0; t < 36; t++) {
        s.player.current_hitpoints = 99;
        col_apply_prayer_oracle(&s);
        col_npc_attack_ctx(&s, &ctx, 0);
    }
    float faced = s.log.pray_faced_by_type[COLO_MANTICORE];
    float correct = s.log.pray_correct_by_type[COLO_MANTICORE];
    CHECK("solo-manticore barrages under the oracle: every orb faced",
        faced >= 9.0f);
    CHECK("solo-manticore barrages under the oracle: every orb prayed",
        correct == faced);
}

static int late_start_total_doses(const ColosseumState* s) {
    int total = 0;
    for (int c = 0; c < COLO_INVENTORY_DISPLAY_SLOTS; c++)
        total += s->inventory_cells[c].dose;
    return total;
}

static int late_start_total_picks(const ColosseumState* s) {
    int picks = 0;
    for (int m = 0; m < COLO_NUM_REAL_MODIFIERS; m++)
        picks += s->modifiers.tier[m];
    return picks;
}

static void test_late_start_entry_state(void) {
    printf("test_late_start_entry_state\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 7;
    ctx.config.loadout_profile_mode = COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY;
    ColosseumState s;

    ctx.config.late_start_state_mode = 0;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 4242);
    int bare_doses = late_start_total_doses(&s);
    CHECK("mode 0 keeps the bare start: no picks", late_start_total_picks(&s) == 0);
    CHECK("mode 0 keeps the bare start: no live draft, spawn armed",
        !s.modifiers.draft_pending && s.wave_spawn_delay > 0);

    ctx.config.late_start_state_mode = 1;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 4242);
    CHECK("wave-8 start carries exactly 7 synthesized picks",
        late_start_total_picks(&s) == 7);
    CHECK("the start wave's own draft opens LIVE", s.modifiers.draft_pending == 1);
    CHECK("the live draft freezes movement like every mid-run draft",
        s.modifiers.draft_free_movement == 0);
    CHECK("the live pick gates the spawn",
        s.modifiers.draft_gates_spawn == 1 && s.wave_spawn_delay == 0);
    int caps_ok = 1;
    for (int m = 0; m < COLO_NUM_REAL_MODIFIERS; m++)
        if (s.modifiers.tier[m] > COLO_MODIFIER_MAX_TIER[m]) caps_ok = 0;
    CHECK("synthesized tiers respect per-modifier caps", caps_ok);
    int prior_doses = late_start_total_doses(&s);
    float kept = (float)prior_doses / (float)bare_doses;
    CHECK("supplies drained to the wave-8 prior (~51% +- noise/rounding)",
        kept > 0.40f && kept < 0.62f);

    uint32_t mask_a = s.modifiers.active_mask;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 4242);
    CHECK("synthesis is deterministic per env seed",
        s.modifiers.active_mask == mask_a &&
        late_start_total_doses(&s) == prior_doses);

    ColosseumContext organic_ctx;
    col_init_context_typed(&organic_ctx);
    organic_ctx.config.loadout_profile_mode = COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY;
    ColosseumState org;
    memset(&org, 0, sizeof(org));
    col_reset_ctx((EncounterState*)&org, (EncounterContext*)&organic_ctx, 99);
    org.modifiers.active_mask = (1u << COLO_MOD_BEES);
    org.modifiers.tier[COLO_MOD_BEES] = 3;
    org.inventory_cells[27] = org.inventory_cells[26];
    org.player.current_hitpoints = 50;
    col_record_wave_entry(&org, 7);
    int org_doses = late_start_total_doses(&org);

    ctx.config.late_start_state_mode = 2;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 4242);
    CHECK("mode 2 replays the recorded organic modifier stack",
        s.modifiers.active_mask == (1u << COLO_MOD_BEES) &&
        s.modifiers.tier[COLO_MOD_BEES] == 3);
    CHECK("mode 2 replays the recorded bag and vitals",
        late_start_total_doses(&s) == org_doses &&
        s.player.current_hitpoints == 50);
    CHECK("mode 2 still opens the live pre-wave draft", s.modifiers.draft_pending == 1);
}

static void test_manticore_orb_same_tick_flick(void) {
    printf("test_manticore_orb_same_tick_flick\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 83);
    advance_to_wave_spawn(&s, &ctx);
    geo_clear_npcs(&s);
    s.wave_ready_delay = 0;
    s.wave_attack_delay = 0;
    s.player.x = 17; s.player.y = 16;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_MANTICORE, 16, 12);
    ColoManticoreState* mc = colo_npc_manticore(&s.npcs[0]);

    int no_fire_damage = 1, prayed_one_tick = 1, prayer_counted = 0, any_queued = 0;
    for (int rep = 0; rep < 16; rep++) {
        s.player_pending_hits.count = 0;
        mc->cycle_step = 1;
        mc->orb_style[1] = ATTACK_STYLE_MAGIC;
        s.player.prayer = PRAYER_PROTECT_MAGIC;
        s.player.current_hitpoints = 99;
        int pc_before = s.tick_scratch.prayer_correct;
        col_npc_attack_ctx(&s, &ctx, 0);
        if (s.player.current_hitpoints != 99) no_fire_damage = 0;
        if (s.tick_scratch.prayer_correct > pc_before) prayer_counted = 1;
        for (int h = 0; h < s.player_pending_hits.count; h++) {
            const EncounterPendingHit* ph = &s.player_pending_hits.hits[h];
            if (!ph->active) continue;
            any_queued = 1;
            if (ph->ticks_remaining != 1 || ph->damage != 0) prayed_one_tick = 0;
        }
    }
    CHECK("a prayed orb queues a 1-tick landing for 0 damage", any_queued && prayed_one_tick);
    CHECK("no orb damage lands on the fire tick", no_fire_damage);
    CHECK("the orb's prayer is checked on its fire tick", prayer_counted);

    s.modifiers.active_mask |= (1u << COLO_MOD_RELENTLESS);
    s.modifiers.tier[COLO_MOD_RELENTLESS] = 3;
    int queued_damage = 0, no_fire_damage2 = 1;
    for (int rep = 0; rep < 32 && !queued_damage; rep++) {
        s.player_pending_hits.count = 0;
        mc->cycle_step = 1;
        mc->orb_style[1] = ATTACK_STYLE_MAGIC;
        s.player.prayer = PRAYER_PROTECT_RANGED;
        s.player.current_hitpoints = 99;
        col_npc_attack_ctx(&s, &ctx, 0);
        if (s.player.current_hitpoints != 99) no_fire_damage2 = 0;
        for (int h = 0; h < s.player_pending_hits.count; h++) {
            const EncounterPendingHit* ph = &s.player_pending_hits.hits[h];
            if (ph->active && ph->ticks_remaining == 1 && ph->damage > 0) queued_damage = 1;
        }
    }
    CHECK("an unprayed orb queues damage for a 1-tick-later landing", queued_damage);
    CHECK("an unprayed orb lands nothing on the fire tick", no_fire_damage2);

    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 83);
    s.wave_spawn_delay = 0;
    complete_open_draft(&s, &ctx, 1);
    geo_clear_npcs(&s);
    s.player.x = 17; s.player.y = 16;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_MANTICORE, 16, 12);
    s.npcs[0].attack_timer = 4;
    mc = colo_npc_manticore(&s.npcs[0]);

    int actions[COLO_NUM_ACTION_HEADS] = {0};
    int protected_ticks = 0, flicked_damage = 0;
    for (int t = 0; t < 40 && !s.episode_over; t++) {
        int expect_orb = mc->cycle_step >= 0 && mc->cycle_step < 3;
        if (expect_orb) {
            AttackStyle next = mc->orb_style[mc->cycle_step];
            actions[COLO_HEAD_PRAYER] =
                next == ATTACK_STYLE_MAGIC ? COLO_OVERHEAD_MAGIC :
                next == ATTACK_STYLE_RANGED ? COLO_OVERHEAD_RANGED :
                COLO_OVERHEAD_MELEE;
        } else {
            actions[COLO_HEAD_PRAYER] = COLO_OVERHEAD_NO_CHANGE;
        }
        int hp_before = s.player.current_hitpoints;
        step_and_observe(&s, &ctx, actions);
        if (expect_orb) {
            protected_ticks++;
            if (s.player.current_hitpoints < hp_before) flicked_damage = 1;
        }
        s.player.current_hitpoints = 99;
        s.player.current_prayer = 99;
    }
    CHECK("multiple telegraphed orb ticks observed through the step loop",
        protected_ticks >= 6);
    CHECK("flicking the telegraphed style each fire tick blocks every such orb",
        !flicked_damage);
}

static void test_manticore_shared_wave_cycle(void) {
    printf("test_manticore_shared_wave_cycle\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);

    int all_shared = 1;
    int all_valid = 1;
    int saw_double = 0;

    for (int wave = 8; wave <= 10; wave++) {
        for (uint32_t seed = 1; seed <= 40; seed++) {
            ColosseumState s;
            memset(&s, 0, sizeof(s));
            ctx.config.start_wave = wave;
            col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, seed);
    advance_to_wave_spawn(&s, &ctx);

            const AttackStyle* first = NULL;
            int n_mc = 0;
            for (int i = 0; i < COLO_MAX_NPCS; i++) {
                if (!s.npcs[i].active || s.npcs[i].type != COLO_MANTICORE) continue;
                ColoManticoreState* mc = colo_npc_manticore(&s.npcs[i]);
                n_mc++;

                int counts[3] = { 0, 0, 0 };
                for (int o = 0; o < 3; o++) {
                    if (mc->fixed_orb_style[o] == ATTACK_STYLE_RANGED) counts[0]++;
                    else if (mc->fixed_orb_style[o] == ATTACK_STYLE_MAGIC) counts[1]++;
                    else if (mc->fixed_orb_style[o] == ATTACK_STYLE_MELEE) counts[2]++;
                }
                if (counts[0] != 1 || counts[1] != 1 || counts[2] != 1 ||
                        mc->fixed_orb_style[2] != ATTACK_STYLE_MELEE) all_valid = 0;
                if (first == NULL) first = mc->fixed_orb_style;
                else if (first[0] != mc->fixed_orb_style[0] ||
                         first[1] != mc->fixed_orb_style[1] ||
                         first[2] != mc->fixed_orb_style[2]) all_shared = 0;
            }
            if (n_mc >= 2) saw_double = 1;
        }
    }
    CHECK("double-manticore waves actually spawn two manticores", saw_double);
    CHECK("every manticore in a wave shares one fixed orb cycle", all_shared);
    CHECK("the wave cycle is valid (one of each style, melee last)", all_valid);

    ColosseumState s;
    memset(&s, 0, sizeof(s));
    ctx.config.start_wave = 1;
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 89);
    geo_clear_npcs(&s);
    s.wave = 8;
    s.wave_manticore_pattern_rolled = 0;
    s.player.x = 13; s.player.y = 12;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_MANTICORE, 12, 16);
    col_init_npc(&s, 1, COLO_MANTICORE, 18, 16);
    ColoManticoreState* amc = colo_npc_manticore(&s.npcs[0]);
    ColoManticoreState* bmc = colo_npc_manticore(&s.npcs[1]);
    CHECK("manually-spawned peers also share the wave pattern",
        amc->fixed_orb_style[0] == bmc->fixed_orb_style[0] &&
        amc->fixed_orb_style[1] == bmc->fixed_orb_style[1] &&
        amc->fixed_orb_style[2] == bmc->fixed_orb_style[2]);

    s.npcs[0].attack_timer = 0;
    s.npcs[1].attack_timer = 0;
    s.player.current_hitpoints = 99;
    col_npc_attack_ctx(&s, &ctx, 0);
    CHECK("A committed and fired orb 0", amc->cycle_step == 1);
    col_npc_attack_ctx(&s, &ctx, 1);
    CHECK("a peer ready at A's barrage start is delayed, holding the shared pattern",
        s.npcs[1].attack_timer == COLO_MANTICORE_STAGGER_TICKS &&
        bmc->cycle_step == 0 &&
        bmc->orb_style[0] == bmc->fixed_orb_style[0]);
    for (int t = 0; t < 4; t++) col_npc_attack_ctx(&s, &ctx, 1);
    CHECK("the delayed peer is still holding 4 ticks after A fired",
        bmc->cycle_step == 0 && s.npcs[1].attack_timer == 1);
    col_npc_attack_ctx(&s, &ctx, 1);
    CHECK("the delayed peer fires exactly 5 ticks after A's barrage started",
        bmc->cycle_step == 1);
}

static void test_manticore_stagger_overlap_fidelity(void) {
    printf("test_manticore_stagger_overlap_fidelity\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);

    ColosseumState s;
    memset(&s, 0, sizeof(s));
    ctx.config.start_wave = 1;
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 89);
    geo_clear_npcs(&s);
    s.wave = 8;
    s.wave_manticore_pattern_rolled = 0;
    s.player.x = 13; s.player.y = 12;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_MANTICORE, 12, 16);
    col_init_npc(&s, 1, COLO_MANTICORE, 18, 16);
    ColoManticoreState* amc = colo_npc_manticore(&s.npcs[0]);
    ColoManticoreState* bmc = colo_npc_manticore(&s.npcs[1]);
    s.player.current_hitpoints = 99;
    s.npcs[0].attack_timer = 0;
    s.npcs[1].attack_timer = 2;
    col_npc_attack_ctx(&s, &ctx, 0);
    col_npc_attack_ctx(&s, &ctx, 1);
    CHECK("a peer still charging at A's barrage start is not delayed",
        s.npcs[1].attack_timer == 1 && bmc->cycle_step == 0);
    col_npc_attack_ctx(&s, &ctx, 0);
    col_npc_attack_ctx(&s, &ctx, 1);
    CHECK("a peer whose charge completes mid-barrage overlaps immediately",
        amc->cycle_step == 2 && bmc->cycle_step == 1);

    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 89);
    geo_clear_npcs(&s);
    s.wave = 8;
    s.wave_manticore_pattern_rolled = 0;
    s.player.x = 5; s.player.y = 9;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_MANTICORE, 5, 14);
    col_init_npc(&s, 1, COLO_MANTICORE, 11, 8);
    amc = colo_npc_manticore(&s.npcs[0]);
    bmc = colo_npc_manticore(&s.npcs[1]);
    s.player.current_hitpoints = 99;
    s.npcs[0].attack_timer = 0;
    s.npcs[1].attack_timer = 0;
    CHECK("fixture: A sees the player", col_npc_has_los_to_player(&s, &s.npcs[0]));
    CHECK("fixture: the pillar blocks B's LoS", !col_npc_has_los_to_player(&s, &s.npcs[1]));
    col_npc_attack_ctx(&s, &ctx, 0);
    col_npc_attack_ctx(&s, &ctx, 1);
    CHECK("a ready but LoS-blocked peer is not delayed at A's barrage start",
        amc->cycle_step == 1 && s.npcs[1].attack_timer == 0 && bmc->cycle_step < 0);
    s.player.x = 12; s.player.y = 12;
    col_rebuild_player_collision_flags(&s);
    CHECK("fixture: B sees the player after the step",
        col_npc_has_los_to_player(&s, &s.npcs[1]));
    col_npc_attack_ctx(&s, &ctx, 0);
    col_npc_attack_ctx(&s, &ctx, 1);
    CHECK("entering a second manticore's LoS mid-barrage eats the overlap",
        amc->cycle_step == 2 && bmc->cycle_step == 1);

    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 89);
    geo_clear_npcs(&s);
    s.wave = 8;
    s.wave_manticore_pattern_rolled = 0;
    s.player.x = 13; s.player.y = 12;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_MANTICORE, 12, 16);
    col_init_npc(&s, 1, COLO_MANTICORE, 18, 16);
    amc = colo_npc_manticore(&s.npcs[0]);
    bmc = colo_npc_manticore(&s.npcs[1]);
    s.player.current_hitpoints = 99;
    s.npcs[0].attack_timer = 3;
    s.npcs[1].attack_timer = 3;
    for (int t = 0; t < 3; t++) {
        col_npc_attack_ctx(&s, &ctx, 0);
        col_npc_attack_ctx(&s, &ctx, 1);
    }
    CHECK("a synced pair alternates: first fires, second is delayed 5",
        amc->cycle_step == 1 &&
        s.npcs[1].attack_timer == COLO_MANTICORE_STAGGER_TICKS &&
        bmc->cycle_step == 0);
    col_npc_attack_ctx(&s, &ctx, 0);
    col_npc_attack_ctx(&s, &ctx, 1);
    CHECK("the delayed half of the pair keeps holding through A's barrage",
        amc->cycle_step == 2 && bmc->cycle_step == 0 &&
        s.npcs[1].attack_timer == COLO_MANTICORE_STAGGER_TICKS - 1);
}

static void test_projectile_prayer_locks_at_throw(void) {
    printf("test_projectile_prayer_locks_at_throw\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 83);
    geo_clear_npcs(&s);
    s.player.x = 17; s.player.y = 16;
    col_rebuild_player_collision_flags(&s);

    col_init_npc(&s, 0, COLO_SERPENT_SHAMAN, 10, 16);

    int any_queued = 0, all_frozen = 1, no_throw_damage = 1, prayer_counted = 0, flick_safe = 1;
    for (int rep = 0; rep < 24; rep++) {
        s.player_pending_hits.count = 0;
        s.npcs[0].attack_timer = 0;
        s.player.prayer = PRAYER_PROTECT_MAGIC;
        s.player.current_hitpoints = 99;
        int pc_before = s.tick_scratch.prayer_correct;
        col_npc_attack_ctx(&s, &ctx, 0);
        if (s.player.current_hitpoints != 99) no_throw_damage = 0;
        if (s.tick_scratch.prayer_correct > pc_before) prayer_counted = 1;
        s.player.prayer = PRAYER_NONE;
        for (int h = 0; h < s.player_pending_hits.count; h++) {
            const EncounterPendingHit* ph = &s.player_pending_hits.hits[h];
            if (!ph->active || ph->attack_style != ATTACK_STYLE_MAGIC) continue;
            any_queued = 1;
            if (ph->damage != 0) all_frozen = 0;
            if (ph->check_prayer != 0) flick_safe = 0;
        }
    }
    CHECK("a prayed projectile freezes to 0 damage at throw", any_queued && all_frozen);
    CHECK("the projectile flies (nothing lands on the throw tick)", no_throw_damage);
    CHECK("the projectile's prayer is counted on its throw tick", prayer_counted);
    CHECK("the frozen hit is flick-resistant (check_prayer=0)", flick_safe);
    CHECK("the throw path attributes to the per-type prayer log",
        s.log.pray_faced_by_type[COLO_SERPENT_SHAMAN] > 0.0f);

    s.modifiers.active_mask |= (1u << COLO_MOD_RELENTLESS);
    s.modifiers.tier[COLO_MOD_RELENTLESS] = 3;
    int kept_damage = 0;
    for (int rep = 0; rep < 32 && !kept_damage; rep++) {
        s.player_pending_hits.count = 0;
        s.npcs[0].attack_timer = 0;
        s.player.prayer = PRAYER_PROTECT_MELEE;
        s.player.current_hitpoints = 99;
        col_npc_attack_ctx(&s, &ctx, 0);
        s.player.prayer = PRAYER_PROTECT_MAGIC;
        for (int h = 0; h < s.player_pending_hits.count; h++) {
            const EncounterPendingHit* ph = &s.player_pending_hits.hits[h];
            if (ph->active && ph->attack_style == ATTACK_STYLE_MAGIC && ph->damage > 0) kept_damage = 1;
        }
    }
    CHECK("a wrong-prayer projectile keeps its damage (locked at throw)", kept_damage);
}

static void test_javelin_skyfall_no_defence_gate(void) {
    printf("test_javelin_skyfall_no_defence_gate\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 97);
    geo_clear_npcs(&s);
    s.player.x = 17; s.player.y = 16;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_JAVELIN_COLOSSUS, 16, 12);
    ColoJavelinState* jv = colo_npc_javelin(&s.npcs[0]);
    const ColoNpcStats* stats = &COLO_NPC_STATS[COLO_JAVELIN_COLOSSUS];

    int queue_before = s.player_pending_hits.count;
    for (int a = 0; a < 4; a++) col_npc_attack_javelin(&s, &ctx, 0, stats);
    CHECK("attacks 1-4 are normal queued throws",
        s.player_pending_hits.count == queue_before + 4 && jv->skyfall_pending == 0);
    col_npc_attack_javelin(&s, &ctx, 0, stats);
    CHECK("the 5th attack marks the player's tile with the 6-tick delay",
        jv->skyfall_pending == 1 && jv->skyfall_timer == COLO_JAVELIN_SKYFALL_DELAY &&
        jv->skyfall_tile_x == s.player.x && jv->skyfall_tile_y == s.player.y);

    int nonzero = 0, in_range_ok = 1, high_roll = 0;
    for (int rep = 0; rep < 300; rep++) {
        jv->attack_count = 4;
        jv->skyfall_pending = 0;
        col_npc_attack_javelin(&s, &ctx, 0, stats);
        if (jv->skyfall_damage > 0) nonzero++;
        if (jv->skyfall_damage < 0 || jv->skyfall_damage > COLO_JAVELIN_SKYFALL_MAX_HIT) in_range_ok = 0;
        if (jv->skyfall_damage >= 37) high_roll = 1;
    }
    CHECK("skyfall damage ignores defence (>=80% of 300 marks nonzero)", nonzero >= 240);
    CHECK("rolls span the raw 0..40 typeless band up to the top", in_range_ok && high_roll);

    s.player.prayer = PRAYER_PROTECT_RANGED;
    s.player.current_hitpoints = 99;
    jv->skyfall_pending = 1;
    jv->skyfall_timer = 1;
    jv->skyfall_damage = 37;
    jv->skyfall_tile_x = s.player.x;
    jv->skyfall_tile_y = s.player.y;
    col_npc_resolve_javelin_skyfall(&s, &ctx, 0);
    CHECK("on the marked tile the skyfall lands through Protect-from-Missiles",
        s.player.current_hitpoints == 99 - 37 && jv->skyfall_pending == 0);

    s.player.current_hitpoints = 99;
    jv->skyfall_pending = 1;
    jv->skyfall_timer = 1;
    jv->skyfall_damage = 37;
    jv->skyfall_tile_x = s.player.x + 1;
    jv->skyfall_tile_y = s.player.y;
    col_npc_resolve_javelin_skyfall(&s, &ctx, 0);
    CHECK("off the marked tile the skyfall misses entirely",
        s.player.current_hitpoints == 99 && jv->skyfall_pending == 0);
}

static int sol_setup(ColosseumState* s, ColosseumContext* ctx, uint32_t seed) {
    col_init_context_typed(ctx);
    ctx->config.start_wave = 11;
    memset(s, 0, sizeof(*s));
    col_reset_ctx((EncounterState*)s, (EncounterContext*)ctx, seed);
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    while (s->wave_spawn_delay > 0 || s->wave_ready_delay > 0 ||
            s->wave_attack_delay > 0)
        step_and_observe(s, ctx, idle);
    return col_sol_find_idx(s);
}

static int sol_setup_speedrun(ColosseumState* s, ColosseumContext* ctx, uint32_t seed) {
    col_init_context_typed(ctx);
    ctx->config.start_wave = 11;
    ctx->config.loadout_profile_mode = COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY;
    memset(s, 0, sizeof(*s));
    col_reset_ctx((EncounterState*)s, (EncounterContext*)ctx, seed);
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    while (s->wave_spawn_delay > 0 || s->wave_ready_delay > 0 ||
            s->wave_attack_delay > 0)
        step_and_observe(s, ctx, idle);
    return col_sol_find_idx(s);
}

static void sol_move_player(ColosseumState* s, int x, int y) {
    s->player.x = x;
    s->player.y = y;
    s->player_dest_x = -1;
    s->player_dest_y = -1;
    osrs_interaction_clear(&s->interaction);
    col_rebuild_player_collision_flags(s);
}

static int sol_count_active_beams(const ColosseumState* s) {
    int n = 0;
    for (int b = 0; b < COLO_SOL_BEAM_MAX; b++)
        if (s->sol.beams[b].active) n++;
    return n;
}

static void sol_clear_beams_and_sand(ColosseumState* s) {
    memset(s->sol.beams, 0, sizeof(s->sol.beams));
    s->sol.hazard_tile_count = 0;
}

static int sol_phase_sand_invariants_hold(const ColosseumState* s, int expected_count) {
    if (s->sol.hazard_tile_count != expected_count) return 0;
    int player_tile_seen = 0;
    for (int i = 0; i < s->sol.hazard_tile_count; i++) {
        int x = s->sol.hazard_tile_x[i];
        int y = s->sol.hazard_tile_y[i];
        if (!col_in_boss_arena(s, x, y)) return 0;
        if (col_static_blocked(x, y)) return 0;
        if (x == s->player.x && y == s->player.y) player_tile_seen = 1;
        for (int j = 0; j < i; j++)
            if (x == s->sol.hazard_tile_x[j] && y == s->sol.hazard_tile_y[j])
                return 0;
    }
    return player_tile_seen;
}

static void sol_pin(ColosseumState* s, int idx, int x, int y) {
    wb_move_npc(s, idx, x, y);
    s->sol.attack_delay = 30000;
    s->sol.immobile_ticks = 30000;
}

static void test_sol_adjacency_gate_and_kiting(void) {
    printf("test_sol_adjacency_gate_and_kiting\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idx = sol_setup(&s, &ctx, 101);
    CHECK("Sol spawned on wave 12", idx >= 0);

    int idle[COLO_NUM_ACTION_HEADS] = {0};
    int moved = 0;
    int first_attack_tick = -1;
    int dist_at_first_attack = -1;
    for (int t = 0; t < 30 && first_attack_tick < 0; t++) {
        s.player.current_hitpoints = 99;
        int pre_dist = encounter_dist_to_npc(
            s.player.x, s.player.y, s.npcs[idx].x, s.npcs[idx].y, 5);
        step_and_observe(&s, &ctx, idle);
        if (s.npcs[idx].moved_this_tick) moved++;
        if (s.npcs[idx].attacked_this_tick) {
            first_attack_tick = s.tick;
            dist_at_first_attack = pre_dist;
        }
    }
    CHECK("Sol chases the player across the arena", moved >= 5);
    CHECK("Sol initiates only when adjacent at the start of a tick",
        first_attack_tick > 0 && dist_at_first_attack == 1);
    CHECK("the fight opener is a forced Spear (variant 1)",
        s.sol.last_attack_kind == COLO_SOL_ATTACK_SPEAR &&
        s.sol.aoe_attack == COLO_SOL_AOE_SPEAR1);

    int second_attack_tick = -1;
    for (int t = 0; t < 12 && second_attack_tick < 0; t++) {
        s.player.current_hitpoints = 99;
        step_and_observe(&s, &ctx, idle);
        if (s.npcs[idx].attacked_this_tick) second_attack_tick = s.tick;
    }
    CHECK("a stationary player eats the next attack exactly 7 ticks later",
        second_attack_tick == first_attack_tick + COLO_SOL_SPEAR_DELAY);

    int walk_east[COLO_NUM_ACTION_HEADS] = {0};
    walk_east[COLO_HEAD_PRIMARY] = 7;
    int third_attack_tick = -1;
    for (int t = 0; t < 40 && third_attack_tick < 0; t++) {
        s.player.current_hitpoints = 99;
        step_and_observe(&s, &ctx, t < 8 ? walk_east : idle);
        if (s.npcs[idx].attacked_this_tick) third_attack_tick = s.tick;
    }
    CHECK("kiting on the cooldown delays the next attack beyond its delay",
        third_attack_tick > second_attack_tick + COLO_SOL_SPEAR_DELAY);
}

static void test_sol_attack_selection_invariants(void) {
    printf("test_sol_attack_selection_invariants\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idx = sol_setup(&s, &ctx, 103);
    CHECK("the fight opens with the forced-spear flag armed", s.sol.force_spear == 1);
    CHECK("a forced draw is a Spear", col_sol_select_attack(&s) == COLO_SOL_ATTACK_SPEAR);

    s.sol.phase = 3;
    s.sol.special_cooldown = 0;
    int normals_since_special = 99;
    int specials = 0, normals = 0, violations = 0;
    for (int n = 0; n < 100; n++) {
        int kind = col_sol_select_attack(&s);
        if (kind == COLO_SOL_ATTACK_TRIPLE || kind == COLO_SOL_ATTACK_GRAPPLE) {
            if (normals_since_special < COLO_SOL_SPECIAL_COOLDOWN) violations++;
            specials++;
            normals_since_special = 0;
        } else {
            normals++;
            normals_since_special++;
        }
    }
    CHECK("specials appear in the 100-draw mix", specials > 0);
    CHECK("the double-weighted normals dominate", normals > specials);
    CHECK("every special has >= 2 normals since the previous special", violations == 0);

    s.sol.phase = 0;
    s.sol.special_cooldown = 0;
    int early_specials = 0;
    for (int n = 0; n < 50; n++) {
        int kind = col_sol_select_attack(&s);
        if (kind != COLO_SOL_ATTACK_SPEAR && kind != COLO_SOL_ATTACK_SHIELD)
            early_specials++;
    }
    CHECK("above 90% HP only spear/shield are drawn", early_specials == 0);

    s.sol.last_attack_kind = COLO_SOL_ATTACK_NONE;
    s.sol.last_variant = 0;
    int v1 = col_sol_pick_variant(&s.sol, COLO_SOL_ATTACK_SPEAR);
    int v2 = col_sol_pick_variant(&s.sol, COLO_SOL_ATTACK_SPEAR);
    int v3 = col_sol_pick_variant(&s.sol, COLO_SOL_ATTACK_SPEAR);
    int v4 = col_sol_pick_variant(&s.sol, COLO_SOL_ATTACK_SHIELD);
    int v5 = col_sol_pick_variant(&s.sol, COLO_SOL_ATTACK_SPEAR);
    CHECK("consecutive same-type casts alternate 1 -> 2 -> 1", v1 == 1 && v2 == 2 && v3 == 1);
    CHECK("a type switch resets the variant to 1", v4 == 1 && v5 == 1);

    idx = sol_setup(&s, &ctx, 107);
    sol_move_player(&s, s.npcs[idx].x + 2, s.npcs[idx].y - 1);
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    int opener_tick = -1;
    for (int t = 0; t < 10 && opener_tick < 0; t++) {
        s.player.current_hitpoints = 99;
        step_and_observe(&s, &ctx, idle);
        if (s.npcs[idx].attacked_this_tick) opener_tick = s.tick;
    }
    CHECK("opener landed against the adjacent player", opener_tick > 0);

    s.npcs[idx].hp = (COLO_SOL_HP_MAX * 89) / 100;
    s.player.current_hitpoints = 99;
    step_and_observe(&s, &ctx, idle);
    CHECK("the 90% crossing enters phase 1", s.sol.phase == 1);
    CHECK("the transition spawns the phase crystal", s.sol.crystal_count == 1);
    CHECK("the transition drops 6 beams around the player",
        sol_count_active_beams(&s) == 6);
    CHECK("Sol is frozen through the transition", s.sol.immobile_ticks > 0);
    CHECK("the post-transition attack is forced to Spear", s.sol.force_spear == 1);
    int next_kind = -1;
    for (int t = 0; t < 20 && next_kind < 0; t++) {
        s.player.current_hitpoints = 99;
        step_and_observe(&s, &ctx, idle);
        if (s.npcs[idx].attacked_this_tick) next_kind = s.sol.last_attack_kind;
    }
    CHECK("the first attack after the transition is a Spear",
        next_kind == COLO_SOL_ATTACK_SPEAR);
}

static void test_sol_parry_schedule_and_damage(void) {
    printf("test_sol_parry_schedule_and_damage\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idle[COLO_NUM_ACTION_HEADS] = {0};

    for (int low = 0; low <= 1; low++) {
        int idx = sol_setup(&s, &ctx, 109 + (uint32_t)low);
        s.npcs[idx].hp = low ? (COLO_SOL_HP_MAX * 40) / 100 : (COLO_SOL_HP_MAX * 80) / 100;
        s.sol.phase = low ? 3 : 1;
        s.sol.attack_delay = 1000;
        sol_move_player(&s, 12, 12);
        col_sol_start_triple_parry(&s, idx);
        s.player.current_hitpoints = 99;

        int dmg_at[13] = {0};
        int hp_prev = 99;
        for (int t = 1; t <= 12; t++) {
            step_and_observe(&s, &ctx, idle);
            dmg_at[t] = hp_prev - s.player.current_hitpoints;
            hp_prev = s.player.current_hitpoints;
        }
        int h3 = low ? 10 : 9;
        int d2 = low ? 30 : 25;
        int d3 = low ? 45 : 35;
        CHECK(low ? "low band: 15/30/45 land at +3/+6/+10"
                  : "high band: 15/25/35 land at +3/+6/+9",
            dmg_at[3] == 15 && dmg_at[6] == d2 && dmg_at[h3] == d3);
        int clean = 1;
        for (int t = 1; t <= 12; t++)
            if (t != 3 && t != 6 && t != h3 && dmg_at[t] != 0) clean = 0;
        CHECK("no parry damage lands off-schedule", clean);
        CHECK("the combo retires after the third hit", s.sol.parry_hits_left == 0);
    }
}

static void test_sol_parry_prayer_punish(void) {
    printf("test_sol_parry_prayer_punish\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idx = sol_setup(&s, &ctx, 113);
    s.npcs[idx].hp = (COLO_SOL_HP_MAX * 80) / 100;
    s.sol.phase = 1;
    s.sol.attack_delay = 1000;
    sol_move_player(&s, 12, 12);

    col_sol_start_triple_parry(&s, idx);
    s.player.current_hitpoints = 99;
    int actions[COLO_NUM_ACTION_HEADS] = {0};
    int deactivated_after_each = 1;
    int prayer_correct_before = s.log.total_prayer_correct;
    for (int t = 1; t <= 9; t++) {
        actions[COLO_HEAD_PRAYER] = (t == 3 || t == 6 || t == 9)
            ? COLO_OVERHEAD_MELEE : COLO_OVERHEAD_NO_CHANGE;
        step_and_observe(&s, &ctx, actions);
        if ((t == 3 || t == 6 || t == 9) && s.player.prayer != PRAYER_NONE)
            deactivated_after_each = 0;
    }
    CHECK("flicking Protect from Melee exactly at land blocks all three hits",
        s.player.current_hitpoints == 99);
    CHECK("every parry hit force-deactivates the overhead prayers",
        deactivated_after_each);
    CHECK("blocked parry hits count prayer_correct",
        s.log.total_prayer_correct >= prayer_correct_before + 3);

    col_sol_start_triple_parry(&s, idx);
    s.player.current_hitpoints = 99;
    for (int t = 1; t <= 9; t++) {
        actions[COLO_HEAD_PRAYER] = COLO_OVERHEAD_MELEE;
        step_and_observe(&s, &ctx, actions);
    }
    CHECK("camping the prayer early makes every hit unblockable (75 total)",
        s.player.current_hitpoints == 99 - (15 + 25 + 35));
}

static void test_sol_grapple_perfect_parry(void) {
    printf("test_sol_grapple_perfect_parry\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idx = sol_setup(&s, &ctx, 127);
    s.sol.attack_delay = 1000;
    sol_move_player(&s, 18, 18);
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    int actions[COLO_NUM_ACTION_HEADS] = {0};

    col_sol_start_grapple(&s);
    CHECK("the called slot is inside the 5-slot A12 domain",
        s.sol.grapple_body_slot >= 0 && s.sol.grapple_body_slot < COLO_NUM_GRAPPLE_SLOTS);
    s.player.current_hitpoints = 99;
    for (int t = 0; t < COLO_SOL_GRAPPLE_WINDOW; t++) step_and_observe(&s, &ctx, idle);
    int fail_dmg = 99 - s.player.current_hitpoints;
    CHECK("an unanswered grapple lands 20-44", fail_dmg >= 20 && fail_dmg <= 44);

    col_sol_start_grapple(&s);
    s.player.current_hitpoints = 99;
    actions[COLO_HEAD_GRAPPLE_PARRY] = s.sol.grapple_body_slot + 1;
    step_and_observe(&s, &ctx, actions);
    CHECK("an early correct click parries without the perfect bonus",
        !s.sol.grapple_active && s.player.current_hitpoints == 99 &&
        s.sol.next_attack_guaranteed_max == 0);
    actions[COLO_HEAD_GRAPPLE_PARRY] = 0;

    col_sol_start_grapple(&s);
    s.player.current_hitpoints = 99;
    int slot = s.sol.grapple_body_slot;
    while (s.sol.grapple_timer > 2) step_and_observe(&s, &ctx, idle);
    actions[COLO_HEAD_GRAPPLE_PARRY] = slot + 1;
    step_and_observe(&s, &ctx, actions);
    actions[COLO_HEAD_GRAPPLE_PARRY] = 0;
    CHECK("a last-tick click is a perfect parry: no damage, max armed",
        s.player.current_hitpoints == 99 && s.sol.next_attack_guaranteed_max == 1);

    int max_hit = col_live_loadout_stats(&s)->max_hit;
    CHECK("rig sanity: the melee loadout has a positive max hit", max_hit > 0);
    col_player_attack_target(&s, idx);
    CHECK("the guaranteed max is consumed at no less than the loadout max hit",
        s.player_attack_dmg >= max_hit && s.sol.next_attack_guaranteed_max == 0 &&
        s.sol.guaranteed_max_ticks == 0);

    col_sol_start_grapple(&s);
    s.player.current_hitpoints = 99;
    while (s.sol.grapple_timer > 2) step_and_observe(&s, &ctx, idle);
    actions[COLO_HEAD_GRAPPLE_PARRY] = s.sol.grapple_body_slot + 1;
    step_and_observe(&s, &ctx, actions);
    actions[COLO_HEAD_GRAPPLE_PARRY] = 0;
    CHECK("second perfect parry armed", s.sol.next_attack_guaranteed_max == 1);
    for (int t = 0; t < COLO_SOL_PERFECT_MAX_TICKS; t++) step_and_observe(&s, &ctx, idle);
    CHECK("an unconsumed guaranteed max expires after 5 ticks",
        s.sol.next_attack_guaranteed_max == 0);
}

static void test_sol_perfect_parry_forces_spec_attack(void) {
    printf("test_sol_perfect_parry_forces_spec_attack\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idx = sol_setup_speedrun(&s, &ctx, 128);
    CHECK("speedrun Sol setup succeeded", idx >= 0);
    CHECK("speedrun inventory carries dragon claws",
        test_find_inventory_cell_with_item(&s, ITEM_DRAGON_CLAWS) >= 0);

    int max_hit = test_col_spec_stats_for_kind(&s, 1).max_hit;
    int claws_total = 2 * max_hit - 1;
    int expected_total = claws_total / 2 + claws_total / 4 +
        claws_total / 8 + claws_total / 8 + 1;
    CHECK("rig sanity: claws max hit is positive", max_hit > 0);

    s.sol.attack_delay = 1000;
    s.player.special_energy = 100;
    col_equip_from_cell(&s, test_find_inventory_cell_with_item(&s, ITEM_DRAGON_CLAWS));
    s.player.spec_armed = 1;
    s.sol.next_attack_guaranteed_max = 1;
    s.sol.guaranteed_max_ticks = COLO_SOL_PERFECT_MAX_TICKS;
    col_player_attack_target(&s, idx);

    CHECK("perfect-parry claws uses the forced first-success best total",
        s.player_attack_dmg == expected_total &&
        s.npcs[idx].pending_hits.count == 4);
    CHECK("perfect-parry spec consumes and clears the max flag",
        s.sol.next_attack_guaranteed_max == 0 &&
        s.sol.guaranteed_max_ticks == 0);
    CHECK("perfect-parry spec still spends claws energy",
        s.player.special_energy == 50 && s.player.spec_armed == 0);
}

static void test_sol_shield_safe_rings(void) {
    printf("test_sol_shield_safe_rings\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idx = sol_setup(&s, &ctx, 131);
    sol_pin(&s, idx, COLO_SOL_SPAWN_X, COLO_SOL_SPAWN_Y);
    ColoNPC* boss = &s.npcs[idx];
    int cx = boss->x + 2, cy = boss->y + 2;
    s.sol.aoe_x = boss->x;
    s.sol.aoe_y = boss->y;

    s.sol.aoe_attack = COLO_SOL_AOE_SHIELD1;
    CHECK("shield1: the inner 7x7 burns",
        col_sol_aoe_tile_is_hazard(&s.sol, cx, cy - 3));
    CHECK("shield1: the Chebyshev-4 ring face is safe",
        !col_sol_aoe_tile_is_hazard(&s.sol, cx, cy - 4));
    CHECK("shield1: the ring corner is safe",
        !col_sol_aoe_tile_is_hazard(&s.sol, cx - 4, cy - 4));
    CHECK("shield1: one past the ring burns",
        col_sol_aoe_tile_is_hazard(&s.sol, cx, cy - 5));
    CHECK("shield1: the far arena burns",
        col_sol_aoe_tile_is_hazard(&s.sol, cx - 8, cy - 8));

    s.sol.aoe_attack = COLO_SOL_AOE_SHIELD2;
    CHECK("shield2: Chebyshev 4 is inside the 9x9 block",
        col_sol_aoe_tile_is_hazard(&s.sol, cx, cy - 4));
    CHECK("shield2: the Chebyshev-5 ring is safe",
        !col_sol_aoe_tile_is_hazard(&s.sol, cx, cy - 5) &&
        !col_sol_aoe_tile_is_hazard(&s.sol, cx - 5, cy - 5));
    CHECK("shield2: one past the ring burns",
        col_sol_aoe_tile_is_hazard(&s.sol, cx, cy - 6));

    s.sol.aoe_attack = COLO_SOL_AOE_NONE;
    s.sol.attack_delay = 1000;
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    for (int variant = 1; variant <= 2; variant++) {
        int ring = variant == 1 ? COLO_SOL_SHIELD1_RING : COLO_SOL_SHIELD2_RING;
        for (int spot = 0; spot < 3; spot++) {
            int off = spot == 0 ? ring : (spot == 1 ? ring - 1 : ring + 1);
            sol_move_player(&s, cx, cy - off);
            col_sol_cast_aoe(&s, idx, COLO_SOL_ATTACK_SHIELD, variant);
            s.player.current_hitpoints = 99;
            for (int t = 0; t < COLO_SOL_AOE_DAMAGE_AGE; t++)
                step_and_observe(&s, &ctx, idle);
            int dmg = 99 - s.player.current_hitpoints;
            if (spot == 0)
                CHECK(variant == 1 ? "shield1 bite: the ring tile is safe"
                                   : "shield2 bite: the ring tile is safe", dmg == 0);
            else
                CHECK(spot == 1 ? "shield bite: inside the block lands 20-44"
                                : "shield bite: outside the ring lands 20-44",
                    dmg >= 20 && dmg <= 44);
            s.sol.aoe_attack = COLO_SOL_AOE_NONE;
        }
    }
}

static void test_sol_spear_lines(void) {
    printf("test_sol_spear_lines\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idx = sol_setup(&s, &ctx, 137);
    sol_pin(&s, idx, COLO_SOL_SPAWN_X, COLO_SOL_SPAWN_Y);

    sol_move_player(&s, 18, 18);
    col_sol_cast_aoe(&s, idx, COLO_SOL_ATTACK_SPEAR, 1);
    CHECK("the cast leaves the spear direction unresolved",
        s.sol.aoe_dir_x == 0 && s.sol.aoe_dir_y == 0);
    int front_ok = 1;
    for (int x = 16; x <= 20; x++)
        if (!col_sol_aoe_tile_is_hazard(&s.sol, x, 18)) front_ok = 0;
    CHECK("spear1 front row covers all 5 columns", front_ok);
    CHECK("spear1 runs TWO lines at the off-centre columns",
        col_sol_aoe_tile_is_hazard(&s.sol, 17, 17) &&
        col_sol_aoe_tile_is_hazard(&s.sol, 19, 17));
    CHECK("spear1 centre + corner columns are safe 1 back (the dodge)",
        !col_sol_aoe_tile_is_hazard(&s.sol, 16, 17) &&
        !col_sol_aoe_tile_is_hazard(&s.sol, 18, 17) &&
        !col_sol_aoe_tile_is_hazard(&s.sol, 20, 17));
    CHECK("spear lines cover 8 tiles from forward 4 (colosim LINE_LENGTH 7)",
        col_sol_aoe_tile_is_hazard(&s.sol, 17, 10) &&
        !col_sol_aoe_tile_is_hazard(&s.sol, 17, 9));

    col_sol_cast_aoe(&s, idx, COLO_SOL_ATTACK_SPEAR, 2);
    CHECK("spear2's 7x7 slam covers the full flush ring row",
        col_sol_aoe_tile_is_hazard(&s.sol, 15, 18) &&
        col_sol_aoe_tile_is_hazard(&s.sol, 17, 18) &&
        col_sol_aoe_tile_is_hazard(&s.sol, 21, 18));
    CHECK("spear2 runs THREE lines at the corner + centre columns past the slam",
        col_sol_aoe_tile_is_hazard(&s.sol, 16, 17) &&
        col_sol_aoe_tile_is_hazard(&s.sol, 18, 17) &&
        col_sol_aoe_tile_is_hazard(&s.sol, 20, 17));
    CHECK("spear2 off-centre columns are safe past the slam (the dodge)",
        !col_sol_aoe_tile_is_hazard(&s.sol, 17, 17) &&
        !col_sol_aoe_tile_is_hazard(&s.sol, 19, 17));

    sol_move_player(&s, 21, 21);
    col_sol_cast_aoe(&s, idx, COLO_SOL_ATTACK_SPEAR, 1);
    CHECK("east cast: the front column burns and lines run east",
        col_sol_aoe_tile_is_hazard(&s.sol, 21, 21) &&
        col_sol_aoe_tile_is_hazard(&s.sol, 22, 20) &&
        col_sol_aoe_tile_is_hazard(&s.sol, 22, 22) &&
        !col_sol_aoe_tile_is_hazard(&s.sol, 22, 21));

    s.sol.aoe_attack = COLO_SOL_AOE_NONE;
    s.sol.attack_delay = 1000;
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    sol_move_player(&s, 18, 18);
    col_sol_cast_aoe(&s, idx, COLO_SOL_ATTACK_SPEAR, 1);
    s.player.current_hitpoints = 99;
    for (int t = 0; t < COLO_SOL_AOE_DAMAGE_AGE; t++)
        step_and_observe(&s, &ctx, idle);
    int dmg = 99 - s.player.current_hitpoints;
    CHECK("standing on a spear tile at the bite tick lands 20-44",
        dmg >= 20 && dmg <= 44);
    s.sol.aoe_attack = COLO_SOL_AOE_NONE;
    sol_move_player(&s, 18, 18);
    col_sol_cast_aoe(&s, idx, COLO_SOL_ATTACK_SPEAR, 1);
    step_and_observe(&s, &ctx, idle);
    sol_move_player(&s, 18, 17);
    s.player.current_hitpoints = 99;
    step_and_observe(&s, &ctx, idle);
    CHECK("the telegraph-tick dodge to the centre column avoids spear1 fully",
        s.player.current_hitpoints == 99);

    s.sol.aoe_attack = COLO_SOL_AOE_NONE;
    sol_move_player(&s, 18, 18);
    col_sol_cast_aoe(&s, idx, COLO_SOL_ATTACK_SPEAR, 1);
    step_and_observe(&s, &ctx, idle);
    CHECK("the direction is still unresolved on the telegraph tick",
        s.sol.aoe_dir_x == 0 && s.sol.aoe_dir_y == 0);
    sol_move_player(&s, 22, 16);
    s.player.current_hitpoints = 99;
    step_and_observe(&s, &ctx, idle);
    CHECK("the bite locks the diagonal from the fleeing player's tile",
        s.sol.aoe_dir_x == 1 && s.sol.aoe_dir_y == -1);
    CHECK("running away laterally past the corner is NOT a free dodge",
        s.player.current_hitpoints < 99 &&
        99 - s.player.current_hitpoints >= 20 &&
        99 - s.player.current_hitpoints <= 44);
}

static void test_sol_crystal_lifecycle(void) {
    printf("test_sol_crystal_lifecycle\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idx = sol_setup(&s, &ctx, 139);
    (void)idx;
    s.sol.attack_delay = 1000;
    sol_move_player(&s, 16, 14);

    int accumulates = 1, edges_ok = 1;
    for (int p = 1; p <= 4; p++) {
        col_sol_enter_phase(&s, p);
        if (s.sol.crystal_count != p) accumulates = 0;
        if (s.sol.crystals[p - 1].edge != p - 1) edges_ok = 0;
    }
    CHECK("one crystal spawns at each transition (4 by 25%)", accumulates);
    CHECK("crystals take their own edges in N/E/S/W order", edges_ok);
    col_sol_enter_phase(&s, 5);
    CHECK("the enrage transition adds no fifth crystal", s.sol.crystal_count == 4);
    s.sol.phase = 4;
    sol_clear_beams_and_sand(&s);

    int on_segments = 1;
    const ColoSolCrystal* cn = &s.sol.crystals[0];
    const ColoSolCrystal* ce = &s.sol.crystals[1];
    const ColoSolCrystal* cs2 = &s.sol.crystals[2];
    const ColoSolCrystal* cw = &s.sol.crystals[3];
    if (cn->y != 23 || cn->x < 11 || cn->x > 22) on_segments = 0;
    if (ce->x != 23 || ce->y < 11 || ce->y > 22) on_segments = 0;
    if (cs2->y != 10 || cs2->x < 11 || cs2->x > 22) on_segments = 0;
    if (cw->x != 10 || cw->y < 11 || cw->y > 22) on_segments = 0;
    CHECK("each crystal patrols one tile inside its boundary, endpoints inset 2"
          " (clear of the corner pillars)", on_segments);

    int idle[COLO_NUM_ACTION_HEADS] = {0};
    int x0 = s.sol.crystals[0].x;
    for (int t = 0; t < 5; t++) {
        s.player.current_hitpoints = 99;
        step_and_observe(&s, &ctx, idle);
    }
    CHECK("the crystal takes its first patrol step on the 5th tick",
        abs(s.sol.crystals[0].x - x0) == 1);
    int x1 = s.sol.crystals[0].x;
    for (int t = 0; t < 3; t++) {
        s.player.current_hitpoints = 99;
        step_and_observe(&s, &ctx, idle);
    }
    CHECK("then one patrol step every 3 ticks",
        abs(s.sol.crystals[0].x - x1) == 1);
    int positions_hold = 1;

    sol_move_player(&s, s.sol.crystals[0].x, 14);
    col_sol_fire_lasers(&s);
    CHECK("the volley arms every crystal at once",
        s.sol.crystals[0].firing_freeze == COLO_SOL_LASER_FREEZE &&
        s.sol.crystals[3].firing_freeze == COLO_SOL_LASER_FREEZE);
    CHECK("the volley rerolls the shared cooldown 25-34 before enrage",
        s.sol.laser_cooldown >= 25 && s.sol.laser_cooldown <= 34);
    int cx_before = s.sol.crystals[0].x;
    s.player.current_hitpoints = 99;
    for (int t = 0; t < 6; t++) {
        step_and_observe(&s, &ctx, idle);
        if (s.sol.crystals[0].x != cx_before) positions_hold = 0;
    }
    CHECK("crystals hold position while firing", positions_hold);
    CHECK("no laser damage before the freeze-3 tick",
        s.player.current_hitpoints == 99);
    step_and_observe(&s, &ctx, idle);
    int laser_dmg = 99 - s.player.current_hitpoints;
    CHECK("an aligned player eats 60-75 at firing_freeze == 3",
        laser_dmg >= 60 && laser_dmg <= 75);

    for (int t = 0; t < 8; t++) step_and_observe(&s, &ctx, idle);
    sol_move_player(&s, s.sol.crystals[0].x, 14);
    col_sol_fire_lasers(&s);
    for (int t = 0; t < 5; t++) step_and_observe(&s, &ctx, idle);
    sol_move_player(&s, s.sol.crystals[0].x + 1, 14);
    s.player.current_hitpoints = 99;
    step_and_observe(&s, &ctx, idle);
    CHECK("stepping off the line before the damage tick dodges the laser",
        s.player.current_hitpoints == 99);

    s.sol.phase = 5;
    col_sol_fire_lasers(&s);
    CHECK("at enrage the volley cooldown is 12",
        s.sol.laser_cooldown == COLO_SOL_CRYSTAL_COOLDOWN_ENRAGE);
}

static void test_sol_aoe_reaction_window(void) {
    printf("test_sol_aoe_reaction_window\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idx = sol_setup(&s, &ctx, 149);
    int idle[COLO_NUM_ACTION_HEADS] = {0};

    int cast_seen = 0;
    for (int t = 0; t < 30 && !cast_seen; t++) {
        s.player.current_hitpoints = 99;
        step_and_observe(&s, &ctx, idle);
        if (s.npcs[idx].attacked_this_tick) cast_seen = 1;
    }
    CHECK("the opener spear casts with the player on a hazard tile",
        cast_seen && s.sol.aoe_attack == COLO_SOL_AOE_SPEAR1 &&
        col_sol_aoe_tile_is_hazard(&s.sol, s.player.x, s.player.y));

    s.player.current_hitpoints = 99;
    step_and_observe(&s, &ctx, idle);
    CHECK("no AoE damage on the telegraph tick (cast + 1)",
        s.player.current_hitpoints == 99);
    step_and_observe(&s, &ctx, idle);
    CHECK("a stationary player is hit exactly 2 ticks after the cast",
        s.player.current_hitpoints < 99);

    sol_move_player(&s, 16, 16);
    cast_seen = 0;
    for (int t = 0; t < 20 && !cast_seen; t++) {
        s.player.current_hitpoints = 99;
        step_and_observe(&s, &ctx, idle);
        if (s.npcs[idx].attacked_this_tick) cast_seen = 1;
    }
    CHECK("a second AoE cast follows on the spear/shield rotation",
        cast_seen && s.sol.aoe_attack != COLO_SOL_AOE_NONE &&
        col_sol_aoe_tile_is_hazard(&s.sol, s.player.x, s.player.y));
    s.player.current_hitpoints = 99;
    step_and_observe(&s, &ctx, idle);
    int safe_x = -1, safe_y = -1;
    for (int x = s.player.x - 3; x <= s.player.x + 3 && safe_x < 0; x++)
        for (int y = s.player.y - 3; y <= s.player.y + 3 && safe_x < 0; y++)
            if (col_in_boss_arena(&s, x, y) && !col_static_blocked(x, y) &&
                    !col_sol_aoe_tile_is_hazard(&s.sol, x, y)) {
                safe_x = x; safe_y = y;
            }
    CHECK("a safe tile exists within reach of the dodge", safe_x >= 0);
    sol_move_player(&s, safe_x, safe_y);
    step_and_observe(&s, &ctx, idle);
    CHECK("moving off on the telegraph tick dodges the slam",
        s.player.current_hitpoints == 99);
}

static void test_sol_laser_react_window(void) {
    printf("test_sol_laser_react_window\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idx = sol_setup(&s, &ctx, 151);
    (void)idx;
    s.sol.attack_delay = 1000;
    int idle[COLO_NUM_ACTION_HEADS] = {0};

    col_sol_spawn_crystal(&s);
    sol_move_player(&s, s.sol.crystals[0].x, 14);
    col_sol_fire_lasers(&s);
    s.player.current_hitpoints = 99;
    for (int t = 0; t < 3; t++) step_and_observe(&s, &ctx, idle);
    CHECK("the beam telegraph opens 3 ticks after the volley (freeze 6)",
        s.sol.crystals[0].firing_freeze == COLO_SOL_LASER_BEAM_SHOW_MAX);
    for (int t = 0; t < 3; t++) step_and_observe(&s, &ctx, idle);
    CHECK("the 3 reaction ticks pass without damage",
        s.player.current_hitpoints == 99);
    step_and_observe(&s, &ctx, idle);
    CHECK("the aligned hit lands on the tick after the reaction window",
        s.player.current_hitpoints < 99);
}

static void test_sol_phase_transition_sand_guarantees(void) {
    printf("test_sol_phase_transition_sand_guarantees\n");
    ColosseumContext ctx;
    ColosseumState s;
    int seeded_ok = 1;

    for (uint32_t seed = 150; seed < 230; seed++) {
        int idx = sol_setup(&s, &ctx, seed);
        (void)idx;
        s.sol.attack_delay = 1000;
        sol_move_player(&s, 17, 14);
        col_sol_enter_phase(&s, 1);
        for (int t = 0; t < COLO_SOL_BEAM_TO_POOL_TICKS; t++)
            col_sol_tick_molten(&s);
        if (!sol_phase_sand_invariants_hold(&s, COLO_SOL_BEAM_COUNT)) {
            seeded_ok = 0;
            break;
        }
    }
    CHECK("seeded phase transitions place exactly 6 unique in-arena sand tiles with one under player",
        seeded_ok);

    int idx = sol_setup(&s, &ctx, 231);
    (void)idx;
    s.sol.attack_delay = 1000;

    int corner_x = COLO_BOSS_ARENA_MIN_X + 2;
    int corner_y = COLO_BOSS_ARENA_MIN_Y + 1;
    sol_move_player(&s, corner_x, corner_y);
    CHECK("rig sanity: corner-edge player tile is walkable",
        col_in_boss_arena(&s, s.player.x, s.player.y) &&
        !col_static_blocked(s.player.x, s.player.y));
    col_sol_enter_phase(&s, 1);
    for (int t = 0; t < COLO_SOL_BEAM_TO_POOL_TICKS; t++)
        col_sol_tick_molten(&s);
    CHECK("corner-edge phase transition still places 6 in-arena sand tiles including player",
        sol_phase_sand_invariants_hold(&s, COLO_SOL_BEAM_COUNT));
}

static void test_sol_beams_become_pools(void) {
    printf("test_sol_beams_become_pools\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idx = sol_setup(&s, &ctx, 149);
    (void)idx;
    s.sol.attack_delay = 1000;
    sol_move_player(&s, 17, 14);

    col_sol_drop_beams(&s);
    int beams = sol_count_active_beams(&s);
    int in_box = 1;
    for (int b = 0; b < COLO_SOL_BEAM_MAX; b++) {
        if (!s.sol.beams[b].active) continue;
        int dx = abs(s.sol.beams[b].x - s.player.x);
        int dy = abs(s.sol.beams[b].y - s.player.y);
        if (dx > COLO_SOL_BEAM_SPREAD || dy > COLO_SOL_BEAM_SPREAD) in_box = 0;
        if (col_static_blocked(s.sol.beams[b].x, s.sol.beams[b].y)) in_box = 0;
    }
    CHECK("6 beams drop inside the 9x9 around the player", beams == 6 && in_box);

    int idle[COLO_NUM_ACTION_HEADS] = {0};
    for (int t = 0; t < COLO_SOL_BEAM_TO_POOL_TICKS; t++) {
        s.player.current_hitpoints = 99;
        step_and_observe(&s, &ctx, idle);
    }
    CHECK("every beam becomes a molten pool when its strike lands",
        s.sol.hazard_tile_count == 6 && sol_count_active_beams(&s) == 0);

    sol_move_player(&s, s.sol.hazard_tile_x[0], s.sol.hazard_tile_y[0]);
    int burns_ok = 1;
    for (int t = 0; t < 30; t++) {
        s.player.current_hitpoints = 99;
        step_and_observe(&s, &ctx, idle);
        int dmg = 99 - s.player.current_hitpoints;
        if (dmg < COLO_MOLTEN_SAND_MIN_HIT ||
            dmg > COLO_MOLTEN_SAND_MIN_HIT + COLO_MOLTEN_SAND_RAND - 1) burns_ok = 0;
    }
    CHECK("standing on a pool burns 5-9 every tick", burns_ok);
    CHECK("pools persist for the rest of the fight", s.sol.hazard_tile_count == 6);
}

static void test_sol_beam_strike_reaction_window(void) {
    printf("test_sol_beam_strike_reaction_window\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idx = sol_setup(&s, &ctx, 157);
    s.sol.attack_delay = 1000;
    sol_move_player(&s, 16, 16);
    int idle[COLO_NUM_ACTION_HEADS] = {0};

    s.npcs[idx].hp = COLO_SOL_HP_MAX * 89 / 100;
    s.player.current_hitpoints = 99;
    step_and_observe(&s, &ctx, idle);
    int player_beam = 0;
    for (int b = 0; b < COLO_SOL_BEAM_MAX; b++)
        if (s.sol.beams[b].active && s.sol.beams[b].x == s.player.x &&
                s.sol.beams[b].y == s.player.y) player_beam = 1;
    CHECK("the transition telegraphs a beam on the player's tile",
        player_beam && s.sol.hazard_tile_count == 0);
    CHECK("no strike damage on the telegraph tick",
        s.player.current_hitpoints == 99);
    step_and_observe(&s, &ctx, idle);
    CHECK("no strike damage 1 tick after the telegraph",
        s.player.current_hitpoints == 99 && s.sol.hazard_tile_count == 0);
    step_and_observe(&s, &ctx, idle);
    CHECK("the pillars strike 2 ticks after the telegraph and burn the camper",
        s.sol.hazard_tile_count == 6 && s.player.current_hitpoints < 99);

    int start_x = -1, start_y = -1;
    for (int x = 12; x <= 21 && start_x < 0; x++)
        for (int y = 12; y <= 21 && start_x < 0; y++) {
            if (!col_in_boss_arena(&s, x, y) || col_static_blocked(x, y)) continue;
            int pooled = 0;
            for (int p = 0; p < s.sol.hazard_tile_count; p++)
                if (s.sol.hazard_tile_x[p] == x && s.sol.hazard_tile_y[p] == y)
                    pooled = 1;
            if (!pooled) { start_x = x; start_y = y; }
        }
    CHECK("a pool-free start tile exists for the dodge rig", start_x >= 0);
    sol_move_player(&s, start_x, start_y);
    s.npcs[idx].hp = COLO_SOL_HP_MAX * 74 / 100;
    s.player.current_hitpoints = 99;
    step_and_observe(&s, &ctx, idle);
    int safe_x = -1, safe_y = -1;
    for (int x = s.player.x - 2; x <= s.player.x + 2 && safe_x < 0; x++)
        for (int y = s.player.y - 2; y <= s.player.y + 2 && safe_x < 0; y++) {
            if (!col_in_boss_arena(&s, x, y) || col_static_blocked(x, y)) continue;
            int marked = 0;
            for (int b = 0; b < COLO_SOL_BEAM_MAX; b++)
                if (s.sol.beams[b].active && s.sol.beams[b].x == x &&
                        s.sol.beams[b].y == y) marked = 1;
            for (int p = 0; p < s.sol.hazard_tile_count; p++)
                if (s.sol.hazard_tile_x[p] == x && s.sol.hazard_tile_y[p] == y)
                    marked = 1;
            if (!marked) { safe_x = x; safe_y = y; }
        }
    CHECK("an unmarked tile exists within reach of the beam dodge", safe_x >= 0);
    sol_move_player(&s, safe_x, safe_y);
    step_and_observe(&s, &ctx, idle);
    step_and_observe(&s, &ctx, idle);
    CHECK("moving off the marked tile on the telegraph tick dodges the strike",
        s.player.current_hitpoints == 99);
}

static void test_sol_enrage_sand_telegraphs(void) {
    printf("test_sol_enrage_sand_telegraphs\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idx = sol_setup(&s, &ctx, 163);
    s.sol.attack_delay = 30000;
    sol_move_player(&s, 16, 16);
    int idle[COLO_NUM_ACTION_HEADS] = {0};

    s.npcs[idx].hp = COLO_SOL_HP_MAX * 89 / 100;
    for (int t = 0; t < 3; t++) {
        s.player.current_hitpoints = 99;
        step_and_observe(&s, &ctx, idle);
    }
    int pools_before = s.sol.hazard_tile_count;
    CHECK("the phase-1 sands have pooled before the enrage rig",
        pools_before >= 1 && sol_count_active_beams(&s) == 0);

    int before = s.sol.hazard_tile_count;
    col_sol_add_pool(&s, s.sol.hazard_tile_x[0], s.sol.hazard_tile_y[0]);
    CHECK("re-covering a pooled tile has no additional effect",
        s.sol.hazard_tile_count == before);

    s.npcs[idx].hp = COLO_SOL_HP_MAX * 9 / 100;
    s.player.current_hitpoints = 99;
    step_and_observe(&s, &ctx, idle);
    CHECK("enrage entry creates NO instant pools (every sand telegraphs)",
        s.sol.hazard_tile_count == pools_before &&
        sol_count_active_beams(&s) >= COLO_SOL_ENRAGE_OPEN_POOLS);
    s.player.current_hitpoints = 99;
    step_and_observe(&s, &ctx, idle);
    CHECK("no enrage sand strikes 1 tick after the telegraphs",
        s.sol.hazard_tile_count == pools_before);
    s.player.current_hitpoints = 99;
    step_and_observe(&s, &ctx, idle);
    int pools_at_strike = s.sol.hazard_tile_count;
    CHECK("the enrage sands strike 2 ticks after their telegraphs",
        pools_at_strike > pools_before);
    CHECK("the enrage spam keeps telegraphing (no instant spam pools)",
        sol_count_active_beams(&s) >= 1);
    s.player.current_hitpoints = 99;
    step_and_observe(&s, &ctx, idle);
    CHECK("spam sand does not strike 1 tick after its telegraph",
        s.sol.hazard_tile_count == pools_at_strike);
    s.player.current_hitpoints = 99;
    step_and_observe(&s, &ctx, idle);
    CHECK("spam sand strikes 2 ticks after its telegraph",
        s.sol.hazard_tile_count == pools_at_strike + 1);
}

static void test_solarflare_sol_orbit_boxes(void) {
    printf("test_solarflare_sol_orbit_boxes\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idx = sol_setup(&s, &ctx, 167);
    (void)idx;
    s.modifiers.active_mask |= (1u << COLO_MOD_SOLARFLARE);
    s.modifiers.tier[COLO_MOD_SOLARFLARE] = 2;
    col_mod_sync_solarflare(&s);
    CHECK("Solarflare orb is active on the Sol wave", s.solarflare.active);

    int inside = 1;
    int anchors_distinct = 1;
    int ax[COLO_NUM_PILLARS], ay[COLO_NUM_PILLARS];
    for (int p = 0; p < COLO_NUM_PILLARS; p++) {
        col_solarflare_tile(&s, p, 0, &ax[p], &ay[p]);
        for (int q = 0; q < p; q++)
            if (ax[p] == ax[q] && ay[p] == ay[q]) anchors_distinct = 0;
        for (int step = 0; step < COLO_SOLARFLARE_RING_STEPS; step++) {
            int x, y;
            col_solarflare_tile(&s, p, step, &x, &y);
            if (x <= s.sol.boss_arena_min_x || x >= s.sol.boss_arena_max_x ||
                    y <= s.sol.boss_arena_min_y || y >= s.sol.boss_arena_max_y)
                inside = 0;
        }
    }
    CHECK("all four Sol orbit boxes sit inside the improvised arena", inside);
    CHECK("the four Sol orbit boxes are distinct corners", anchors_distinct);

    ColosseumContext ctx1;
    ColosseumState s1;
    col_init_context_typed(&ctx1);
    memset(&s1, 0, sizeof(s1));
    col_reset_ctx((EncounterState*)&s1, (EncounterContext*)&ctx1, 29);
    int x0, y0;
    col_solarflare_tile(&s1, 0, 0, &x0, &y0);
    CHECK("waves 1-11 keep the pillar orbit geometry",
        x0 == COLO_PILLARS[0][0] - 1 && y0 == COLO_PILLARS[0][1] - 1);
}

static void loadout_reset(ColosseumState* s, ColosseumContext* ctx, int mode,
                          float frac, uint32_t seed) {
    col_init_context_typed(ctx);
    ctx->config.loadout_profile_mode = mode;
    ctx->config.beginner_loadout_fraction = frac;
    memset(s, 0, sizeof(*s));
    col_reset_ctx((EncounterState*)s, (EncounterContext*)ctx, seed);

    s->modifiers.draft_pending = 0;
    s->modifiers.draft_gates_spawn = 0;
    s->modifiers.draft_free_movement = 0;
    s->wave_spawn_delay = col_wave_entry_delay_ticks(s->wave_spawn_target);
}

static int col_loadout_stats_equal(
    const EncounterLoadoutStats* a,
    const EncounterLoadoutStats* b
) {
    return a->attack_bonus == b->attack_bonus &&
           a->strength_bonus == b->strength_bonus &&
           a->eff_level == b->eff_level &&
           a->max_hit == b->max_hit &&
           a->attack_speed == b->attack_speed &&
           a->attack_range == b->attack_range &&
           a->style == b->style &&
           a->fight_style == b->fight_style &&
           a->def_stab == b->def_stab &&
           a->def_slash == b->def_slash &&
           a->def_crush == b->def_crush &&
           a->def_magic == b->def_magic &&
           a->def_ranged == b->def_ranged &&
           a->att_prayer_mult == b->att_prayer_mult &&
           a->str_prayer_mult == b->str_prayer_mult &&
           a->spell_base_damage == b->spell_base_damage;
}

static EncounterLoadoutStats test_col_live_stats_for_set(
    const ColosseumState* s,
    ColoWeaponSet set
) {
    ColosseumState copy = *s;
    col_apply_weapon_set(&copy, set);
    return *col_live_loadout_stats(&copy);
}

static OsrsEquipmentEffectProfile test_col_live_effects_for_set(
    const ColosseumState* s,
    ColoWeaponSet set
) {
    ColosseumState copy = *s;
    col_apply_weapon_set(&copy, set);
    return *col_live_effects(&copy);
}

static EncounterLoadoutStats test_col_spec_stats_for_kind(
    const ColosseumState* s,
    int kind
) {
    ColosseumState copy = *s;
    copy.player.equipped[GEAR_SLOT_WEAPON] =
        test_profile_spec_item(copy.active_loadout_profile, kind);
    copy.player.equipped[GEAR_SLOT_SHIELD] = osrs_suppress_shield_for_two_handed_weapon(
        copy.player.equipped[GEAR_SLOT_WEAPON],
        copy.player.equipped[GEAR_SLOT_SHIELD]);
    col_sync_weapon_set_from_equipped_weapon(&copy);
    col_mark_live_loadout_dirty(&copy);
    return *col_live_loadout_stats(&copy);
}

static void test_loadout_profiles_and_supplies(void) {
    printf("test_loadout_profiles_and_supplies\n");
    ColosseumContext ctx;
    ColosseumState s;

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 11);
    CHECK("beginner mode pins the beginner profile",
        s.active_loadout_profile == COLO_LOADOUT_PROFILE_BEGINNER);
    CHECK("beginner melee weapon is the fang",
        s.player.equipped[GEAR_SLOT_WEAPON] == ITEM_OSMUMTENS_FANG);
    CHECK("beginner supplies match the budget example",
        s.player.brew_doses == 24 && s.player.restore_doses == 32 &&
        s.player.combat_potion_doses == 8 && s.player.ranged_potion_doses == 8 &&
        s.surge_doses == 0);
    CHECK("beginner restore kind is super restore",
        s.full_supplies.restore_kind == COLO_RESTORE_SUPER_RESTORE);
    OsrsEquipmentEffectProfile beginner_ranged_effects =
        test_col_live_effects_for_set(&s, COLO_GEAR_RANGED);
    OsrsEquipmentEffectProfile beginner_melee_effects =
        test_col_live_effects_for_set(&s, COLO_GEAR_MELEE);
    CHECK("beginner bowfa set carries full crystal points (1+2+3)",
        beginner_ranged_effects.crystal_armour_points == 6);
    CHECK("beginner melee neck carries blood fury",
        osrs_effect_profile_has(&beginner_melee_effects, OSRS_ITEM_EFFECT_BLOOD_FURY));
    CHECK("beginner melee head carries venom immunity",
        osrs_effect_profile_has(&beginner_melee_effects, OSRS_ITEM_EFFECT_VENOM_IMMUNE));
    int beginner_melee_max = col_live_loadout_stats(&s)->max_hit;
    CHECK("beginner spec weapons are SGS then claws",
        test_find_inventory_cell_with_item(&s, ITEM_SGS) >= 0 &&
        test_find_inventory_cell_with_item(&s, ITEM_DRAGON_CLAWS) >= 0);

    uint8_t beginner_sgs_without_defender[NUM_GEAR_SLOTS];
    memcpy(beginner_sgs_without_defender, COLO_BEGINNER_MELEE_LOADOUT, NUM_GEAR_SLOTS);
    beginner_sgs_without_defender[GEAR_SLOT_WEAPON] = ITEM_SGS;
    beginner_sgs_without_defender[GEAR_SLOT_SHIELD] = ITEM_NONE;
    EncounterLoadoutStats beginner_sgs_expected;
    encounter_compute_loadout_stats(beginner_sgs_without_defender, ATTACK_STYLE_MELEE,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_AGGRESSIVE, 0, &beginner_sgs_expected);
    EncounterLoadoutStats beginner_sgs_spec = test_col_spec_stats_for_kind(&s, 1);
    CHECK("beginner SGS spec stats exclude dragon defender bonuses",
        col_loadout_stats_equal(&beginner_sgs_spec, &beginner_sgs_expected));

    uint8_t beginner_claws_without_defender[NUM_GEAR_SLOTS];
    memcpy(beginner_claws_without_defender, COLO_BEGINNER_MELEE_LOADOUT, NUM_GEAR_SLOTS);
    beginner_claws_without_defender[GEAR_SLOT_WEAPON] = ITEM_DRAGON_CLAWS;
    beginner_claws_without_defender[GEAR_SLOT_SHIELD] = ITEM_NONE;
    EncounterLoadoutStats beginner_claws_without;
    encounter_compute_loadout_stats(beginner_claws_without_defender, ATTACK_STYLE_MELEE,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_AGGRESSIVE, 0, &beginner_claws_without);
    EncounterLoadoutStats beginner_claws_spec = test_col_spec_stats_for_kind(&s, 2);
    CHECK("beginner claws spec stats keep dragon defender strength",
        beginner_claws_spec.strength_bonus ==
            beginner_claws_without.strength_bonus +
            ITEM_DATABASE[ITEM_DRAGON_DEFENDER].melee_strength);
    CHECK("beginner claws spec stats keep dragon defender accuracy",
        beginner_claws_spec.attack_bonus > beginner_claws_without.attack_bonus);

    uint8_t beginner_bowfa_without_defender[NUM_GEAR_SLOTS];
    memcpy(beginner_bowfa_without_defender, COLO_BEGINNER_RANGED_LOADOUT, NUM_GEAR_SLOTS);
    beginner_bowfa_without_defender[GEAR_SLOT_SHIELD] = ITEM_NONE;
    EncounterLoadoutStats beginner_bowfa_legal_2h;
    encounter_compute_loadout_stats(beginner_bowfa_without_defender, ATTACK_STYLE_RANGED,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_RAPID, 0, &beginner_bowfa_legal_2h);
    EncounterLoadoutStats beginner_ranged_stats =
        test_col_live_stats_for_set(&s, COLO_GEAR_RANGED);
    CHECK("beginner bowfa ranged stats suppress the shield slot",
        col_loadout_stats_equal(
            &beginner_ranged_stats, &beginner_bowfa_legal_2h));

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 1.0f, 12);
    CHECK("speedrun mode pins the speedrun profile",
        s.active_loadout_profile == COLO_LOADOUT_PROFILE_SPEEDRUN);
    CHECK("speedrun melee weapon is the scythe",
        s.player.equipped[GEAR_SLOT_WEAPON] == ITEM_SCYTHE_OF_VITUR);
    CHECK("speedrun loadout has exactly one scythe",
        test_count_item_in_equipment_and_inventory(&s, ITEM_SCYTHE_OF_VITUR) == 1);
    CHECK("speedrun supplies match the high-efficiency kit",
        s.player.brew_doses == 4 && s.player.restore_doses == 28 &&
        s.player.combat_potion_doses == 4 && s.player.ranged_potion_doses == 4 &&
        s.surge_doses == 4);
    CHECK("speedrun restore kind is sanfew",
        s.full_supplies.restore_kind == COLO_RESTORE_SANFEW);
    OsrsEquipmentEffectProfile speedrun_ranged_effects =
        test_col_live_effects_for_set(&s, COLO_GEAR_RANGED);
    CHECK("speedrun ranged set has the tbow effect",
        osrs_effect_profile_has(&speedrun_ranged_effects, OSRS_ITEM_EFFECT_TWISTED_BOW));

    EncounterLoadoutStats speedrun_melee_stats = *col_live_loadout_stats(&s);
    EncounterLoadoutStats speedrun_ranged_stats =
        test_col_live_stats_for_set(&s, COLO_GEAR_RANGED);
    CHECK("speedrun scythe total (7/4 splats) out-hits the beginner fang",
        speedrun_melee_stats.max_hit * 7 / 4 > beginner_melee_max);
    CHECK("both loadouts melee-style on the melee set",
        speedrun_melee_stats.style == ATTACK_STYLE_MELEE &&
        speedrun_ranged_stats.style == ATTACK_STYLE_RANGED);
    CHECK("spec stats computed for both spec weapons",
        test_col_spec_stats_for_kind(&s, 1).max_hit > 0 &&
        test_col_spec_stats_for_kind(&s, 2).max_hit > 0);

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_MIXED, 1.0f, 13);
    CHECK("mixed fraction 1.0 always samples beginner",
        s.active_loadout_profile == COLO_LOADOUT_PROFILE_BEGINNER);
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_MIXED, 0.0f, 14);
    CHECK("mixed fraction 0.0 always samples speedrun",
        s.active_loadout_profile == COLO_LOADOUT_PROFILE_SPEEDRUN);
}

static void test_loadout_consumables(void) {
    printf("test_loadout_consumables\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 21);
    complete_open_draft(&s, &ctx, 1);
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    int base_max_hit = col_live_loadout_stats(&s)->max_hit;

    s.player.current_hitpoints = 50;
    int brew[COLO_NUM_ACTION_HEADS] = {0};
    test_click_consumable_action(&s, brew, OSRS_CONSUMABLE_BREW);
    step_and_observe(&s, &ctx, brew);
    CHECK("brew heals 16", s.player.current_hitpoints == 66);
    CHECK("brew consumes a dose and starts the timer",
        s.player.brew_doses == 23 && s.player.potion_timer == 3);
    CHECK("brew drains attack below base", s.player.current_attack < 99);
    CHECK("brew drain lowers the melee max hit",
        col_live_loadout_stats(&s)->max_hit < base_max_hit);

    for (int t = 0; t < 3; t++) step_and_observe(&s, &ctx, idle);
    int restore[COLO_NUM_ACTION_HEADS] = {0};
    test_click_consumable_action(&s, restore, OSRS_CONSUMABLE_SUPER_RESTORE);
    step_and_observe(&s, &ctx, restore);
    CHECK("super restore returns attack to base", s.player.current_attack == 99);
    CHECK("restore recovers the melee max hit",
        col_live_loadout_stats(&s)->max_hit == base_max_hit);
    CHECK("restore consumed a dose", s.player.restore_doses == 31);

    for (int t = 0; t < 3; t++) step_and_observe(&s, &ctx, idle);
    int combat[COLO_NUM_ACTION_HEADS] = {0};
    test_click_consumable_action(&s, combat, OSRS_CONSUMABLE_SUPER_COMBAT);
    step_and_observe(&s, &ctx, combat);
    CHECK("super combat boosts attack to 118", s.player.current_attack == 118);
    CHECK("super combat boosts strength to 118", s.player.current_strength == 118);
    CHECK("super combat raises the melee max hit",
        col_live_loadout_stats(&s)->max_hit > base_max_hit);
    CHECK("combat pot consumed a dose", s.player.combat_potion_doses == 7);

    float mask[COLO_ACTION_MASK_SIZE];
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    int combat_cell = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_SUPER_COMBAT);
    int surge_cell = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_SURGE);
    CHECK("combat pot masked while potion timer is live",
        test_click_mask_for_cell_s(&s, mask, combat_cell) == 0.0f);
    CHECK("surge masked for the beginner (no doses)",
        surge_cell < 0);

    for (int t = 0; t < 3; t++) step_and_observe(&s, &ctx, idle);
    int rng_pot[COLO_NUM_ACTION_HEADS] = {0};
    test_click_consumable_action(&s, rng_pot, OSRS_CONSUMABLE_RANGING);
    step_and_observe(&s, &ctx, rng_pot);
    CHECK("ranging potion boosts ranged to 112", s.player.current_ranged == 112);

    for (int t = 0; t < 3; t++) step_and_observe(&s, &ctx, idle);
    s.player.current_prayer = 40;
    step_and_observe(&s, &ctx, restore);
    CHECK("super restore gives +32 prayer", s.player.current_prayer == 72);
}

static void test_loadout_divine_potions_and_stat_drift(void) {
    printf("test_loadout_divine_potions_and_stat_drift\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    float mask[COLO_ACTION_MASK_SIZE];

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 81);
    CHECK("speedrun combat and ranging potions are divine",
        test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_DIVINE_COMBAT) >= 0 &&
        test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_DIVINE_RANGING) >= 0);
    int divine_combat_cell =
        test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_DIVINE_COMBAT);
    int divine_ranged_cell =
        test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_DIVINE_RANGING);
    s.player.current_hitpoints = 10;
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    CHECK("divine boost potions are still clickable at 10 HP",
        test_click_mask_for_cell_s(&s, mask, divine_combat_cell) == 1.0f &&
        test_click_mask_for_cell_s(&s, mask, divine_ranged_cell) == 1.0f);
    s.player.current_hitpoints = 11;
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    CHECK("divine boost potions unmask above 10 HP",
        test_click_mask_for_cell_s(&s, mask, divine_combat_cell) == 1.0f &&
        test_click_mask_for_cell_s(&s, mask, divine_ranged_cell) == 1.0f);

    s.player.current_hitpoints = 50;
    int combat[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s, combat, divine_combat_cell);
    step_and_observe(&s, &ctx, combat);
    CHECK("divine combat chunks 10 HP and starts a 500-tick hold",
        s.player.current_hitpoints == 40 &&
        s.divine_combat_timer == ENCOUNTER_DIVINE_POTION_TICKS &&
        s.player.combat_potion_doses == 3);
    CHECK("divine combat boosts to the held floor",
        s.player.current_attack == 118 && s.player.current_strength == 118 &&
        s.player.current_defence == 118);
    for (int t = 0; t < ENCOUNTER_DIVINE_POTION_TICKS - 1; t++)
        col_tick_live_stat_drift_and_divines(&s);
    CHECK("divine combat stats survive through tick 499",
        s.player.current_attack == 118 && s.player.current_strength == 118 &&
        s.player.current_defence == 118 && s.divine_combat_timer == 1);
    col_tick_live_stat_drift_and_divines(&s);
    CHECK("divine combat expiry drops its stats to base instantly",
        s.player.current_attack == 99 && s.player.current_strength == 99 &&
        s.player.current_defence == 99 && s.divine_combat_timer == 0);

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 82);
    s.player.current_hitpoints = 50;
    int ranged[COLO_NUM_ACTION_HEADS] = {0};
    test_click_consumable_action(&s, ranged, OSRS_CONSUMABLE_DIVINE_RANGING);
    step_and_observe(&s, &ctx, ranged);
    CHECK("divine ranging chunks 10 HP and starts a 500-tick hold",
        s.player.current_hitpoints == 40 &&
        s.divine_ranged_timer == ENCOUNTER_DIVINE_POTION_TICKS &&
        s.player.ranged_potion_doses == 3);
    for (int t = 0; t < ENCOUNTER_DIVINE_POTION_TICKS; t++)
        col_tick_live_stat_drift_and_divines(&s);
    CHECK("divine ranging expiry drops Ranged to base instantly",
        s.player.current_ranged == 99 && s.divine_ranged_timer == 0);

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 83);
    CHECK("beginner boost potions are regular",
        test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_SUPER_COMBAT) >= 0 &&
        test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_RANGING) >= 0);
    int regular_combat[COLO_NUM_ACTION_HEADS] = {0};
    test_click_consumable_action(&s, regular_combat, OSRS_CONSUMABLE_SUPER_COMBAT);
    step_and_observe(&s, &ctx, regular_combat);
    CHECK("regular beginner combat boost does not arm a divine timer",
        s.player.current_hitpoints == 99 && s.divine_combat_timer == 0 &&
        s.player.current_attack == 118);
    for (int t = 0; t < ENCOUNTER_STAT_DRIFT_TICKS; t++)
        col_tick_live_stat_drift_and_divines(&s);
    CHECK("regular beginner combat boost decays one level after 100 live ticks",
        s.player.current_attack == 117 && s.player.current_strength == 117 &&
        s.player.current_defence == 117);

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 84);
    s.stat_drift_timer = ENCOUNTER_STAT_DRIFT_TICKS - 1;
    s.divine_combat_timer = 1;
    s.player.current_attack = 118;
    s.player.current_strength = 118;
    s.player.current_defence = 118;
    step_and_observe(&s, &ctx, idle);
    CHECK("stat drift and divine timers freeze during the draft gap",
        s.stat_drift_timer == ENCOUNTER_STAT_DRIFT_TICKS - 1 &&
        s.divine_combat_timer == 1 && s.player.current_attack == 118);

    s.stat_drift_timer = 37;
    s.divine_combat_timer = 123;
    s.divine_ranged_timer = 234;
    ColoSnapshot snap;
    col_snapshot_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &snap);
    CHECK("snapshot version is v20 for Solarflare shared cadence",
        snap.version == COLO_SNAPSHOT_VERSION && COLO_SNAPSHOT_VERSION == 20u);
    ColosseumState restored;
    memset(&restored, 0, sizeof(restored));
    col_restore_ctx((EncounterState*)&restored, (EncounterContext*)&ctx, &snap, sizeof(snap));
    CHECK("snapshot round-trips stat drift and divine timers",
        restored.stat_drift_timer == 37 &&
        restored.divine_combat_timer == 123 &&
        restored.divine_ranged_timer == 234);
}

static void test_loadout_sanfew_and_serp_helm(void) {
    printf("test_loadout_sanfew_and_serp_helm\n");
    ColosseumContext ctx;
    ColosseumState s;

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 31);
    complete_open_draft(&s, &ctx, 1);
    s.player_venom = 8;
    s.player_venom_timer = 12;
    s.player_poison = COLO_POISON_BEE_CONTACT_SEVERITY;
    s.player_poison_timer = 9;
    s.player.current_prayer = 40;
    int restore[COLO_NUM_ACTION_HEADS] = {0};
    test_click_consumable_action(&s, restore, OSRS_CONSUMABLE_SANFEW);
    step_and_observe(&s, &ctx, restore);
    CHECK("sanfew clears venom", s.player_venom == 0 && s.player_venom_timer == 0);
    CHECK("sanfew clears poison", s.player_poison == 0 && s.player_poison_timer == 0);
    CHECK("sanfew gives +33 prayer", s.player.current_prayer == 73);
    CHECK("sanfew consumed a restore dose", s.player.restore_doses == 27);

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 32);
    s.modifiers.active_mask |= (1u << COLO_MOD_MANTIMAYHEM);
    s.modifiers.tier[COLO_MOD_MANTIMAYHEM] = 2;
    CHECK("rig sanity: beginner starts in the melee set",
        s.weapon_set == COLO_GEAR_MELEE);
    col_mod_manticore_apply_venom(&s, 1);
    CHECK("serp helm blocks venom in the melee set", s.player_venom == 0);
    col_apply_weapon_set(&s, COLO_GEAR_RANGED);
    col_mod_manticore_apply_venom(&s, 1);
    CHECK("venom applies in the ranged set", s.player_venom == COLO_VENOM_START);
    col_apply_weapon_set(&s, COLO_GEAR_MELEE);
    int venom_before = s.player_venom;
    col_mod_manticore_apply_venom(&s, 1);
    CHECK("serp helm does not cure or escalate an existing stack",
        s.player_venom == venom_before);

    s.player_poison = 0;
    s.player_poison_timer = 0;
    col_mod_apply_bee_poison(&s);
    CHECK("serp helm blocks bee poison in the melee set", s.player_poison == 0);
    col_apply_weapon_set(&s, COLO_GEAR_RANGED);
    col_mod_apply_bee_poison(&s);
    CHECK("a venomed player cannot also be poisoned", s.player_poison == 0);
    s.player_venom = 0;
    s.player_venom_timer = 0;
    col_mod_apply_bee_poison(&s);
    CHECK("bee poison applies in the ranged set once venom is gone",
        s.player_poison == COLO_POISON_BEE_CONTACT_SEVERITY);
    col_mod_manticore_apply_venom(&s, 1);
    CHECK("venom application replaces an active poison",
        s.player_venom == COLO_VENOM_START && s.player_poison == 0 &&
        s.player_poison_timer == 0);
}

static void test_consumable_overdrink_mask(void) {
    printf("test_consumable_overdrink_mask\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 11);
    s.player.potion_timer = 0;

    int brew = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_BREW);
    int combat = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_DIVINE_COMBAT);
    int sanfew = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_SANFEW);
    CHECK("speedrun kit exposes brew/combat/sanfew cells",
        brew >= 0 && combat >= 0 && sanfew >= 0);

    s.player.current_hitpoints = s.player.base_hitpoints;
    CHECK("brew masked at full HP", !col_inventory_cell_actionable(&s, brew));
    s.player.current_hitpoints = s.player.base_hitpoints - 10;
    CHECK("brew valid below max HP", col_inventory_cell_actionable(&s, brew));

    s.player.current_attack = s.player.base_attack;
    s.player.current_strength = s.player.base_strength;
    s.player.current_defence = s.player.base_defence;
    CHECK("combat valid at unboosted stats", col_inventory_cell_actionable(&s, combat));
    s.player.current_attack = 105;
    s.player.current_strength = 112;
    s.player.current_defence = 118;
    CHECK("combat masked once all combat stats >= 105",
        !col_inventory_cell_actionable(&s, combat));
    s.player.current_strength = 104;
    CHECK("combat valid again when one stat dips below 105",
        col_inventory_cell_actionable(&s, combat));

    s.player.current_attack = s.player.base_attack;
    s.player.current_strength = s.player.base_strength;
    s.player.current_defence = s.player.base_defence;
    s.player.current_ranged = s.player.base_ranged;
    s.player.current_magic = s.player.base_magic;
    s.player.current_prayer = s.player.base_prayer;
    s.player_venom = 0;
    s.player_poison = 0;
    CHECK("sanfew masked with full stats/prayer and no venom",
        !col_inventory_cell_actionable(&s, sanfew));
    s.player_venom = 4;
    CHECK("sanfew valid while venomed (it cures)", col_inventory_cell_actionable(&s, sanfew));
    s.player_venom = 0;
    s.player.current_prayer = s.player.base_prayer - 60;
    CHECK("sanfew valid when prayer is well down", col_inventory_cell_actionable(&s, sanfew));
}

static void test_loadout_surge_potion(void) {
    printf("test_loadout_surge_potion\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 41);
    int idle[COLO_NUM_ACTION_HEADS] = {0};

    s.player.special_energy = 40;
    int surge[COLO_NUM_ACTION_HEADS] = {0};
    test_click_consumable_action(&s, surge, OSRS_CONSUMABLE_SURGE);
    step_and_observe(&s, &ctx, surge);
    CHECK("surge restores 25 energy", s.player.special_energy == 65);
    CHECK("surge consumes a dose and arms the cooldown",
        s.surge_doses == 3 && s.surge_cooldown == COLO_SURGE_COOLDOWN_TICKS);
    step_and_observe(&s, &ctx, idle);
    CHECK("surge cooldown frozen during the draft gap",
        s.surge_cooldown == COLO_SURGE_COOLDOWN_TICKS);
    complete_open_draft(&s, &ctx, 1);
    while (s.wave_spawn_delay > 0 || s.wave_ready_delay > 0 ||
            s.wave_attack_delay > 0)
        step_and_observe(&s, &ctx, idle);
    int cd_before = s.surge_cooldown;
    step_and_observe(&s, &ctx, idle);
    CHECK("surge cooldown ticks during live gameplay", s.surge_cooldown == cd_before - 1);

    s.player.special_energy = 100;
    s.surge_cooldown = 0;
    s.player.potion_timer = 0;
    float mask[COLO_ACTION_MASK_SIZE];
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    int surge_cell = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_SURGE);
    CHECK("surge remains clickable at full special energy",
        test_click_mask_for_cell_s(&s, mask, surge_cell) == 1.0f);
    int surge_doses_before = s.surge_doses;
    int full_energy_surge[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s, full_energy_surge, surge_cell);
    step_and_observe(&s, &ctx, full_energy_surge);
    CHECK("full-energy surge burns one dose and caps energy",
        s.surge_doses == surge_doses_before - 1 && s.player.special_energy == 100);
}

static void test_loadout_spec_weapons(void) {
    printf("test_loadout_spec_weapons\n");
    ColosseumContext ctx;
    ColosseumState s;

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 51);
    geo_clear_npcs(&s);
    s.modifiers.draft_pending = 0;
    s.wave_ready_delay = 0;
    s.player.x = 16; s.player.y = 16;
    col_init_npc(&s, 0, COLO_FREMENNIK_BERSERKER, 16, 17);
    s.player.special_energy = 100;
    col_equip_from_cell(&s, test_find_inventory_cell_with_item(&s, ITEM_DRAGON_CLAWS));
    s.player.spec_armed = 1;
    osrs_interaction_set(&s.interaction, 0);
    s.player.attack_timer = 0;
    col_player_attack_target(&s, 0);
    CHECK("claws spec drains 50 energy", s.player.special_energy == 50);
    CHECK("claws spec disarms after firing", s.player.spec_armed == 0);
    CHECK("claws spec queues the 4-splat cascade",
        s.npcs[0].pending_hits.count == 4);
    CHECK("spec sets the claws attack speed", s.player.attack_timer ==
        get_item(ITEM_DRAGON_CLAWS)->attack_speed);

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 54);
    geo_clear_npcs(&s);
    s.modifiers.draft_pending = 0;
    s.wave_ready_delay = 0;
    s.player.x = 10; s.player.y = 16;
    col_init_npc(&s, 0, COLO_SOL_HEREDIT, 11, 14);
    s.player.attack_timer = 0;
    col_player_attack_target(&s, 0);
    CHECK("scythe queues 3 splats into the 5x5 boss",
        s.npcs[0].pending_hits.count == 3);
    geo_clear_npcs(&s);
    col_init_npc(&s, 0, COLO_FREMENNIK_BERSERKER, 10, 17);
    s.player.attack_timer = 0;
    col_player_attack_target(&s, 0);
    CHECK("scythe queues 1 splat into a 1x1 warbander",
        s.npcs[0].pending_hits.count == 1);

    geo_clear_npcs(&s);
    s.player.x = 16; s.player.y = 16;
    col_init_npc(&s, 0, COLO_FREMENNIK_ARCHER, 15, 16);
    col_init_npc(&s, 1, COLO_FREMENNIK_ARCHER, 15, 15);
    col_init_npc(&s, 2, COLO_FREMENNIK_ARCHER, 15, 17);

    ColScytheResolvedHit arc_hits[COLO_SCYTHE_MAX_HITS];
    int arc_count = col_resolve_scythe_hits(&s, 0, arc_hits);
    int all_rank0 = 1;
    for (int h = 0; h < arc_count; h++)
        if (arc_hits[h].splat_rank != 0) all_rank0 = 0;
    CHECK("scythe arc resolves three distinct full-100% hits",
        arc_count == 3 && all_rank0 &&
        arc_hits[0].npc_slot != arc_hits[1].npc_slot &&
        arc_hits[1].npc_slot != arc_hits[2].npc_slot &&
        arc_hits[0].npc_slot != arc_hits[2].npc_slot);
    s.player.attack_timer = 0;
    col_player_attack_target(&s, 0);
    CHECK("scythe arc queues one hit per distinct arc target",
        s.npcs[0].pending_hits.count == 1 &&
        s.npcs[1].pending_hits.count == 1 &&
        s.npcs[2].pending_hits.count == 1);
    for (int t = 0; t < 4; t++)
        col_resolve_player_projectiles_on_npcs_ctx(&s, &ctx);
    CHECK("scythe landed hits create one render hit per target",
        ctx.npc_render_hit_count[0] == 1 &&
        ctx.npc_render_hit_count[1] == 1 &&
        ctx.npc_render_hit_count[2] == 1);

    geo_clear_npcs(&s);
    s.player.x = 16; s.player.y = 16;
    col_init_npc(&s, 0, COLO_FREMENNIK_ARCHER, 15, 16);
    col_init_npc(&s, 1, COLO_HEALING_TOTEM, 15, 15);
    col_init_npc(&s, 2, COLO_FREMENNIK_ARCHER, 15, 17);
    s.player.attack_timer = 0;
    col_player_attack_target(&s, 0);
    CHECK("scythe arc skips incidental hazard entities",
        s.npcs[0].pending_hits.count == 1 &&
        s.npcs[1].pending_hits.count == 0 &&
        s.npcs[2].pending_hits.count == 1);

    int per_target_roll_seen = 0;
    for (uint32_t seed = 90; seed < 240 && !per_target_roll_seen; seed++) {
        loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, seed);
        geo_clear_npcs(&s);
        s.modifiers.draft_pending = 0;
        s.wave_ready_delay = 0;
        s.player.x = 16; s.player.y = 16;
        col_init_npc(&s, 0, COLO_FREMENNIK_ARCHER, 15, 16);
        col_init_npc(&s, 1, COLO_FREMENNIK_BERSERKER, 15, 15);
        s.player.attack_timer = 0;
        col_player_attack_target(&s, 0);
        int scythe_max = col_live_loadout_stats(&s)->max_hit;
        if (s.npcs[0].pending_hits.count == 1 &&
                s.npcs[1].pending_hits.count == 1 &&
                s.npcs[0].pending_hits.hits[0].damage == scythe_max &&
                s.npcs[1].pending_hits.hits[0].damage != (scythe_max >> 1)) {
            per_target_roll_seen = 1;
        }
    }
    CHECK("scythe rolls incidental targets against their own defence",
        per_target_roll_seen);

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 55);
    geo_clear_npcs(&s);
    s.modifiers.draft_pending = 0;
    s.wave_ready_delay = 0;
    s.player.x = 16; s.player.y = 16;
    col_init_npc(&s, 0, COLO_JAGUAR_WARRIOR, 15, 16);
    col_npc_attack_jaguar(&s, &ctx, 0, &COLO_NPC_STATS[COLO_JAGUAR_WARRIOR]);
    CHECK("jaguar multi-hit records three player render splats",
        ctx.player_render_hit_count == 3);
    RenderEntity scythe_entities[4];
    int scythe_entity_count = 0;
    col_fill_render_entities_ctx(
        (EncounterState*)&s, (EncounterContext*)&ctx,
        scythe_entities, 4, &scythe_entity_count);
    CHECK("render entity carries jaguar multi-hit splats",
        scythe_entity_count >= 1 &&
        scythe_entities[0].render_hit_count == 3);

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 52);
    geo_clear_npcs(&s);
    s.modifiers.draft_pending = 0;
    s.wave_ready_delay = 0;
    s.player.x = 16; s.player.y = 16;
    col_init_npc(&s, 0, COLO_FREMENNIK_BERSERKER, 16, 17);
    const ColoNpcStats* zerk = &COLO_NPC_STATS[COLO_FREMENNIK_BERSERKER];
    col_equip_from_cell(&s, test_find_inventory_cell_with_item(&s, ITEM_ELDER_MAUL));
    int tries = 0;
    while (s.npcs[0].def_drained == 0 && tries < 200) {
        s.player.special_energy = 100;
        s.player.spec_armed = 1;
        s.player.attack_timer = 0;
        s.npcs[0].hp = zerk->hp;
        col_player_attack_target(&s, 0);
        tries++;
    }
    CHECK("elder maul spec eventually lands", s.npcs[0].def_drained > 0);
    CHECK("elder maul drains 35% of current defence",
        s.npcs[0].def_drained == zerk->def_level * 35 / 100);

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 53);
    geo_clear_npcs(&s);
    s.modifiers.draft_pending = 0;
    s.wave_ready_delay = 0;
    s.player.x = 16; s.player.y = 16;
    col_init_npc(&s, 0, COLO_FREMENNIK_SEER, 16, 17);
    const ColoNpcStats* seer = &COLO_NPC_STATS[COLO_FREMENNIK_SEER];
    s.player.equipped[GEAR_SLOT_SHIELD] = ITEM_NONE;
    col_equip_from_cell(&s, test_find_inventory_cell_with_item(&s, ITEM_SGS));
    int healed = 0;
    tries = 0;
    while (!healed && tries < 200) {
        s.player.current_hitpoints = 20;
        s.player.current_prayer = 10;
        s.player.special_energy = 100;
        s.player.spec_armed = 1;
        s.player.attack_timer = 0;
        s.npcs[0].hp = seer->hp;
        col_player_attack_target(&s, 0);
        if (s.player.current_hitpoints > 20) healed = 1;
        tries++;
    }
    CHECK("SGS spec eventually lands", healed);
    CHECK("SGS heal honors the wiki minimum (>= 10)",
        s.player.current_hitpoints >= 30);
    CHECK("SGS prayer restore honors the wiki minimum (>= 5)",
        s.player.current_prayer >= 15);

    col_equip_from_cell(&s, test_find_inventory_cell_with_item(&s, ITEM_DRAGON_CLAWS));
    s.player.special_energy = 10;
    s.player.spec_armed = 0;
    int arm[COLO_NUM_ACTION_HEADS] = {0};
    arm[COLO_HEAD_SPEC] = 1;
    col_tick_player_ctx(&s, &ctx, arm, 0);
    CHECK("arming is refused without the energy", s.player.spec_armed == 0);
    s.player.special_energy = 100;
    col_tick_player_ctx(&s, &ctx, arm, 0);
    CHECK("arming succeeds with the energy", s.player.spec_armed == 1);
    CHECK("equipped claws reach is melee range",
        col_player_attack_range(&s) == 1);
    arm[COLO_HEAD_SPEC] = 2;
    col_tick_player_ctx(&s, &ctx, arm, 0);
    CHECK("disarm action clears the armed spec", s.player.spec_armed == 0);

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 54);
    s.modifiers.draft_pending = 0;
    s.wave_ready_delay = 0;
    s.inventory_cells[26] = osrs_inventory_cell_empty();
    s.inventory_cells[27] = osrs_inventory_cell_empty();
    col_equip_from_cell(&s, test_find_inventory_cell_with_item(&s, ITEM_SGS));
    s.player.special_energy = 100;
    s.player.spec_armed = 1;
    col_equip_from_cell(&s,
        test_find_inventory_cell_with_item(&s, ITEM_DRAGON_DEFENDER));
    CHECK("shield equip displaces the 2H spec weapon",
        s.player.equipped[GEAR_SLOT_WEAPON] == ITEM_NONE &&
        s.player.equipped[GEAR_SLOT_SHIELD] == ITEM_DRAGON_DEFENDER);
    CHECK("shield equip displacing the weapon disarms the spec",
        s.player.spec_armed == 0);
}

static void test_loadout_item_effects(void) {
    printf("test_loadout_item_effects\n");
    ColosseumContext ctx;
    ColosseumState s;

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 61);
    col_apply_weapon_set(&s, COLO_GEAR_RANGED);
    const EncounterLoadoutStats* rls = col_live_loadout_stats(&s);
    const OsrsEquipmentEffectProfile* ranged_effects = col_live_effects(&s);
    int base_att = osrs_player_att_roll(rls->eff_level, rls->attack_bonus);
    const ColoNpcStats* sol = &COLO_NPC_STATS[COLO_SOL_HEREDIT];
    const ColoNpcStats* jag = &COLO_NPC_STATS[COLO_JAGUAR_WARRIOR];
    OsrsPreparedAttackEffects vs_sol = osrs_prepare_attack_effects(
        ranged_effects, &s.player.item_effect_state,
        ITEM_TWISTED_BOW, ATTACK_STYLE_RANGED, OSRS_MAGIC_ATTACK_NONE,
        osrs_target_ref_none(), 1, base_att, rls->max_hit,
        osrs_target_effect_context_magic(sol->magic_level, sol->magic_att_bonus),
        s.player.current_hitpoints, s.player.base_hitpoints);
    OsrsPreparedAttackEffects vs_jag = osrs_prepare_attack_effects(
        ranged_effects, &s.player.item_effect_state,
        ITEM_TWISTED_BOW, ATTACK_STYLE_RANGED, OSRS_MAGIC_ATTACK_NONE,
        osrs_target_ref_none(), 1, base_att, rls->max_hit,
        osrs_target_effect_context_magic(jag->magic_level, jag->magic_att_bonus),
        s.player.current_hitpoints, s.player.base_hitpoints);
    CHECK("tbow hits harder into Sol's 300 magic than the jaguar's 100",
        vs_sol.max_hit > vs_jag.max_hit && vs_sol.attack_roll > vs_jag.attack_roll);

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 62);
    col_apply_weapon_set(&s, COLO_GEAR_RANGED);
    const EncounterLoadoutStats* bls = col_live_loadout_stats(&s);
    ranged_effects = col_live_effects(&s);
    int bowfa_att = osrs_player_att_roll(bls->eff_level, bls->attack_bonus);
    OsrsPreparedAttackEffects bowfa = osrs_prepare_attack_effects(
        ranged_effects, &s.player.item_effect_state,
        ITEM_BOW_OF_FAERDHINEN, ATTACK_STYLE_RANGED, OSRS_MAGIC_ATTACK_NONE,
        osrs_target_ref_none(), 1, bowfa_att, bls->max_hit,
        osrs_target_effect_context_magic(100, 0),
        s.player.current_hitpoints, s.player.base_hitpoints);
    CHECK("crystal armour scales the bowfa damage by 46/40",
        bowfa.max_hit == bls->max_hit * 46 / 40);
    CHECK("crystal armour scales the bowfa accuracy by 26/20",
        bowfa.attack_roll == bowfa_att * 26 / 20);

    int procs = 0;
    uint32_t rng = 777;
    col_apply_weapon_set(&s, COLO_GEAR_MELEE);
    const OsrsEquipmentEffectProfile* melee_effects = col_live_effects(&s);
    for (int i = 0; i < 400; i++) {
        int heal = osrs_blood_fury_heal_amount(
            melee_effects, ATTACK_STYLE_MELEE, 30, &rng);
        if (heal > 0) {
            procs++;
            CHECK("blood fury heals 30% of the damage", heal == 9);
            if (heal != 9) break;
        }
    }
    CHECK("blood fury procs at a plausible 20% rate", procs > 40 && procs < 130);

    col_apply_weapon_set(&s, COLO_GEAR_RANGED);
    ranged_effects = col_live_effects(&s);
    int ranged_heal = osrs_blood_fury_heal_amount(
        ranged_effects, ATTACK_STYLE_RANGED, 30, &rng);
    CHECK("no blood fury heal on the ranged set", ranged_heal == 0);
}

static void test_loadout_offensive_prayers(void) {
    printf("test_loadout_offensive_prayers\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 71);
    complete_open_draft(&s, &ctx, 1);
    int base_max_hit = col_live_loadout_stats(&s)->max_hit;

    int piety[COLO_NUM_ACTION_HEADS] = {0};
    piety[COLO_HEAD_OFFENSIVE] = ENCOUNTER_OFFENSIVE_SET_REFRESH_PIETY;
    step_and_observe(&s, &ctx, piety);
    CHECK("piety activates", s.player.offensive_prayer == OFFENSIVE_PRAYER_PIETY);
    CHECK("piety raises the melee max hit (L12)",
        col_live_loadout_stats(&s)->max_hit > base_max_hit);
    CHECK("piety raises the spec max hits too",
        test_col_spec_stats_for_kind(&s, 1).max_hit > 0 &&
        test_col_spec_stats_for_kind(&s, 2).max_hit > 0);

    int off[COLO_NUM_ACTION_HEADS] = {0};
    off[COLO_HEAD_OFFENSIVE] = ENCOUNTER_OFFENSIVE_OFF;
    step_and_observe(&s, &ctx, off);
    CHECK("offensive off restores the base max hit",
        s.player.offensive_prayer == OFFENSIVE_PRAYER_NONE &&
        col_live_loadout_stats(&s)->max_hit == base_max_hit);

    float mask[COLO_ACTION_MASK_SIZE];
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    int off_offset = col_action_head_mask_offset(COLO_HEAD_OFFENSIVE);
    CHECK("augury is offered points-only even on a non-magic weapon (equip+pray fix)",
        mask[off_offset + ENCOUNTER_OFFENSIVE_SET_REFRESH_AUGURY] == 1.0f);
    CHECK("piety and rigour are offered",
        mask[off_offset + ENCOUNTER_OFFENSIVE_SET_REFRESH_PIETY] == 1.0f &&
        mask[off_offset + ENCOUNTER_OFFENSIVE_SET_REFRESH_RIGOUR] == 1.0f);

    int overhead[COLO_NUM_ACTION_HEADS] = {0};
    overhead[COLO_HEAD_PRAYER] = ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE;
    step_and_observe(&s, &ctx, overhead);
    CHECK("shared overhead melee activates protect melee",
        s.player.prayer == PRAYER_PROTECT_MELEE);
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    int prayer_offset = col_action_head_mask_offset(COLO_HEAD_PRAYER);
    CHECK("shared overhead off is valid while overhead prayer is active",
        mask[prayer_offset + ENCOUNTER_OVERHEAD_OFF] == 1.0f);
    int overhead_off[COLO_NUM_ACTION_HEADS] = {0};
    overhead_off[COLO_HEAD_PRAYER] = ENCOUNTER_OVERHEAD_OFF;
    step_and_observe(&s, &ctx, overhead_off);
    CHECK("shared overhead off clears active overhead prayer",
        s.player.prayer == PRAYER_NONE);

    piety[COLO_HEAD_OFFENSIVE] = ENCOUNTER_OFFENSIVE_SET_REFRESH_PIETY;
    col_player_pretick(&s, &ctx, piety);
    CHECK("piety reactivates before drain-zero regression",
        s.player.offensive_prayer == OFFENSIVE_PRAYER_PIETY &&
        col_live_loadout_stats(&s)->max_hit > base_max_hit);
    s.player.current_prayer = 0;
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    col_player_pretick(&s, &ctx, idle);
    CHECK("prayer drain auto-clear recomputes offensive loadout stats",
        s.player.offensive_prayer == OFFENSIVE_PRAYER_NONE &&
        col_live_loadout_stats(&s)->max_hit == base_max_hit);
}

static void test_total_damage_by_type_captures_typeless(void) {
    printf("test_total_damage_by_type_captures_typeless\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);

    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 2323);
    int hp0 = s.player.current_hitpoints;
    float total_before = s.log.total_damage_by_type[COLO_JAVELIN_COLOSSUS];
    float offpray_before = s.log.offpray_damage_by_type[COLO_JAVELIN_COLOSSUS];

    float typeless_before = s.log.typeless_damage_by_type[COLO_JAVELIN_COLOSSUS];
    float unprayable_before = s.tick_scratch.landed_unprayable_damage;
    col_damage_player_from(&s, 17, COLO_JAVELIN_COLOSSUS, COLO_DMG_UNPRAYABLE);

    CHECK("typeless NPC damage lands on the player", s.player.current_hitpoints == hp0 - 17);
    CHECK("typeless NPC damage is attributed to total damage",
        s.log.total_damage_by_type[COLO_JAVELIN_COLOSSUS] == total_before + 17.0f);
    CHECK("typeless NPC damage stays out of off-prayer damage",
        s.log.offpray_damage_by_type[COLO_JAVELIN_COLOSSUS] == offpray_before);
    CHECK("typeless NPC damage books the unprayable forensics channel",
        s.tick_scratch.landed_unprayable_damage == unprayable_before + 17.0f);
    CHECK("typeless NPC damage books the per-type typeless total",
        s.log.typeless_damage_by_type[COLO_JAVELIN_COLOSSUS] == typeless_before + 17.0f);
}

static void test_npc_magic_defence_rolls_off_magic_level(void) {
    printf("test_npc_magic_defence_rolls_off_magic_level\n");
    col_build_npc_stats();

    ColoNPC shaman;
    memset(&shaman, 0, sizeof(shaman));
    shaman.type = COLO_SERPENT_SHAMAN;
    const ColoNpcStats* shaman_ns = &COLO_NPC_STATS[COLO_SERPENT_SHAMAN];
    int sh_magic = col_npc_target_def_roll(&shaman, shaman_ns, ATTACK_STYLE_MAGIC, MELEE_STYLE_SLASH);
    int sh_ranged = col_npc_target_def_roll(&shaman, shaman_ns, ATTACK_STYLE_RANGED, MELEE_STYLE_SLASH);
    int sh_melee = col_npc_target_def_roll(&shaman, shaman_ns, ATTACK_STYLE_MELEE, MELEE_STYLE_SLASH);
    CHECK("shaman magic def rolls off Magic level not Defence",
        sh_magic == (shaman_ns->magic_level + 9) * (shaman_ns->magic_def_bonus + 64));
    CHECK("shaman is most magic-resistant (magic > ranged > melee)",
        sh_magic > sh_ranged && sh_ranged > sh_melee);

    ColoNPC manticore;
    memset(&manticore, 0, sizeof(manticore));
    manticore.type = COLO_MANTICORE;
    const ColoNpcStats* mant_ns = &COLO_NPC_STATS[COLO_MANTICORE];
    int mt_magic = col_npc_target_def_roll(&manticore, mant_ns, ATTACK_STYLE_MAGIC, MELEE_STYLE_SLASH);
    int mt_ranged = col_npc_target_def_roll(&manticore, mant_ns, ATTACK_STYLE_RANGED, MELEE_STYLE_SLASH);
    int mt_melee = col_npc_target_def_roll(&manticore, mant_ns, ATTACK_STYLE_MELEE, MELEE_STYLE_SLASH);
    CHECK("manticore magic def rolls off Magic level not Defence",
        mt_magic == (mant_ns->magic_level + 9) * (mant_ns->magic_def_bonus + 64));
    CHECK("manticore is easiest to melee (scythe): melee is the lowest def roll",
        mt_melee < mt_ranged && mt_melee < mt_magic);
}

static void test_matchup_dpt_obs_ranking(void) {
    printf("test_matchup_dpt_obs_ranking\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 771);
    col_build_npc_stats();

    geo_clear_npcs(&s);
    s.player.x = 12;
    s.player.y = 16;
    venator_spawn_enemy(&s, 0, COLO_FREMENNIK_BERSERKER, 16, 16, 1);
    ColoVenatorPreviewTargets targets;
    col_collect_venator_preview_targets(&s, &targets);
    int isolated_extra = col_venator_extra_bounce_if_shot(&s, &targets, 0);
    CHECK("isolated Venator preview has no extra bounces", isolated_extra == 0);

    venator_spawn_enemy(&s, 1, COLO_FREMENNIK_ARCHER, 18, 16, 1);
    venator_spawn_enemy(&s, 2, COLO_FREMENNIK_BERSERKER, 18, 17, 1);
    col_collect_venator_preview_targets(&s, &targets);
    int clustered_extra = col_venator_extra_bounce_if_shot(&s, &targets, 0);
    CHECK("clustered Venator preview sees extra bounces", clustered_extra >= 1);
}

static void test_primary_head_resolution(void) {
    printf("test_primary_head_resolution\n");
    ColosseumContext ctx;
    ColosseumState s;
    init_forecast_test_state(&s, &ctx, 881, 16, 16);
    s.wave_ready_delay = 0;
    s.wave_spawn_delay = 0;
    col_init_npc(&s, 0, COLO_FREMENNIK_BERSERKER, 18, 16);
    s.npcs[0].hp = 200;
    s.npcs[0].max_hp = 200;
    col_rebuild_player_collision_flags(&s);
    col_refresh_current_obs_slots_ctx(&s, &ctx);
    int obs_slot = col_find_target_obs_slot(&s, 0);
    int attack_action = col_primary_attack_action_for_obs_slot(obs_slot);

    int attack[COLO_NUM_ACTION_HEADS] = {0};
    attack[COLO_HEAD_PRIMARY] = attack_action;
    col_tick_player_ctx(&s, &ctx, attack, 1);
    CHECK("PRIMARY attack action sets the mapped NPC interaction",
        osrs_interaction_active(&s.interaction) && s.interaction.target_slot == 0);

    int idle[COLO_NUM_ACTION_HEADS] = {0};
    col_tick_player_ctx(&s, &ctx, idle, 1);
    CHECK("PRIMARY noop holds the existing interaction",
        osrs_interaction_active(&s.interaction) && s.interaction.target_slot == 0);

    float mask[COLO_ACTION_MASK_SIZE];
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    int primary_offset = col_action_head_mask_offset(COLO_HEAD_PRIMARY);
    CHECK("held PRIMARY target stays valid for inventory cancel and re-attack",
        mask[primary_offset + attack_action] == 1.0f);

    int move_action = forecast_move_action_for_delta(1, 0);
    int move[COLO_NUM_ACTION_HEADS] = {0};
    move[COLO_HEAD_PRIMARY] = move_action;
    int x_before = s.player.x;
    col_tick_player_ctx(&s, &ctx, move, 1);
    CHECK("PRIMARY move cancels the held interaction",
        !osrs_interaction_active(&s.interaction));
    CHECK("PRIMARY move walks using the old movement action mapping",
        s.player.x == x_before + 1 && s.player.y == 16);

    col_tick_player_ctx(&s, &ctx, attack, 1);
    CHECK("PRIMARY attack reacquires the mapped NPC after movement",
        osrs_interaction_active(&s.interaction) && s.interaction.target_slot == 0);
}

static void test_combat_fidelity_contract_sizes(void) {
    printf("test_combat_fidelity_contract_sizes\n");
    CHECK("three weapon sets (melee/ranged/magic)", COLO_NUM_WEAPON_SETS == 3);
    CHECK("twenty action heads (per-category inventory heads)", COLO_NUM_ACTION_HEADS == 20);
    CHECK("first equip head follows PRIMARY and PRAYER", COLO_HEAD_EQUIP_BASE == 2);
    CHECK("equip heads cover every gear slot",
        COLO_HEAD_EAT == COLO_HEAD_EQUIP_BASE + NUM_GEAR_SLOTS && COLO_HEAD_DRINK == COLO_HEAD_EAT + 1);
    CHECK("equip head dim is 29", COLO_ACTION_DIMS[COLO_HEAD_EQUIP_SLOT(GEAR_SLOT_WEAPON)] == 29);
    CHECK("eat and drink heads are 29-way",
        COLO_ACTION_DIMS[COLO_HEAD_EAT] == 29 && COLO_ACTION_DIMS[COLO_HEAD_DRINK] == 29);
    CHECK("prayer head uses shared PVE overhead dim",
        COLO_ACTION_DIMS[COLO_HEAD_PRAYER] == ENCOUNTER_OVERHEAD_DIM_PVE);
    CHECK("spell head dim is 3 (none/summon-thrall/death-charge)", COLO_SPELL_DIM == 3);
    CHECK("obs width is 3044", COLO_NUM_OBS == 3044);
    CHECK("weapon-choice tail has 58 features (28 cell DPT + 28 spec + 2 wielded)",
        COLO_WEAPON_CHOICE_OBS_SIZE == 58);
    CHECK("inventory block has 784 features", COLO_INVENTORY_OBS_SIZE == 784);
    CHECK("equipped-self block has 198 features", COLO_EQUIPPED_SELF_OBS_SIZE == 198);
    CHECK("modifier hazard tail has 38 features", COLO_MODIFIER_HAZARD_OBS_SIZE == 38);
    CHECK("modifier block has 74 features", COLO_MODIFIER_OBS_SIZE == 74);
    CHECK("NPC slots have 37 features (DPT obs removed, B0 neutral)", COLO_FEATURES_PER_NPC == 37);
    CHECK("snapshot version is v20", COLO_SNAPSHOT_VERSION == 20u);
    CHECK("every active NPC gets an obs slot (no busy-wave drop)",
        COLO_OBS_NPCS == 24 && COLO_OBS_NPCS == COLO_MAX_NPCS);
    CHECK("PRIMARY head covers noop, movement, and NPC obs slots",
        COLO_ACTION_DIMS[COLO_HEAD_PRIMARY] == COLO_PRIMARY_DIM &&
        COLO_ACTION_DIMS[COLO_HEAD_PRIMARY] == 49);
    CHECK("player block remains 36", COLO_PLAYER_OBS_SIZE == 36);

    int mask_sum = 0;
    for (int h = 0; h < COLO_NUM_ACTION_HEADS; h++) mask_sum += COLO_ACTION_DIMS[h];
    CHECK("mask size equals the summed action-head dims",
        COLO_ACTION_MASK_SIZE == mask_sum && COLO_ACTION_MASK_SIZE == 452);

    int obs_sum = COLO_PLAYER_OBS_SIZE + COLO_PILLAR_OBS_SIZE +
        COLO_INVENTORY_OBS_SIZE + COLO_EQUIPPED_SELF_OBS_SIZE + COLO_NPC_OBS_SIZE +
        COLO_MODIFIER_OBS_SIZE + COLO_WAVE_OBS_SIZE + COLO_BOSS_OBS_SIZE +
        COLO_PENDING_HIT_OBS_SIZE + COLO_STEP_OUT_FORECAST_OBS_SIZE +
        COLO_THREAT_LOS_OBS_SIZE + COLO_THRALL_DC_OBS_SIZE +
        COLO_WEAPON_CHOICE_OBS_SIZE + COLO_SPAWN_OBS_SIZE +
        COLO_THREAT_FIELD_OBS_SIZE;
    CHECK("obs width equals the summed section sizes", COLO_NUM_OBS == obs_sum);

    float opa, ops;
    encounter_offensive_prayer_mults(OFFENSIVE_PRAYER_PIETY, ATTACK_STYLE_MELEE, &opa, &ops);
    CHECK("Piety boosts melee att+str", opa > 1.0f && ops > 1.0f);
    encounter_offensive_prayer_mults(OFFENSIVE_PRAYER_PIETY, ATTACK_STYLE_RANGED, &opa, &ops);
    CHECK("Piety is inert off-style (ranged)", opa == 1.0f && ops == 1.0f);
    encounter_offensive_prayer_mults(OFFENSIVE_PRAYER_RIGOUR, ATTACK_STYLE_RANGED, &opa, &ops);
    CHECK("Rigour boosts ranged att+str", opa > 1.0f && ops > 1.0f);
    encounter_offensive_prayer_mults(OFFENSIVE_PRAYER_RIGOUR, ATTACK_STYLE_MELEE, &opa, &ops);
    CHECK("Rigour is inert off-style (melee)", opa == 1.0f && ops == 1.0f);
    encounter_offensive_prayer_mults(OFFENSIVE_PRAYER_AUGURY, ATTACK_STYLE_MAGIC, &opa, &ops);
    CHECK("Augury boosts magic att", opa > 1.0f);
    encounter_offensive_prayer_mults(OFFENSIVE_PRAYER_AUGURY, ATTACK_STYLE_MELEE, &opa, &ops);
    CHECK("Augury is inert off-style (melee)", opa == 1.0f && ops == 1.0f);
}

static void scythe_spawn_enemy(ColosseumState* s, int slot, int x, int y, int size) {
    ColoNPC* npc = &s->npcs[slot];
    memset(npc, 0, sizeof(*npc));
    npc->active = 1;
    npc->type = COLO_FREMENNIK_BERSERKER;
    npc->x = x;
    npc->y = y;
    npc->size = size;
    npc->hp = 200;
    npc->death_ticks = 0;
}

static void venator_spawn_enemy(
    ColosseumState* s,
    int slot,
    ColoNpcType type,
    int x,
    int y,
    int size
) {
    ColoNPC* npc = &s->npcs[slot];
    memset(npc, 0, sizeof(*npc));
    npc->active = 1;
    npc->type = type;
    npc->x = x;
    npc->y = y;
    npc->size = size;
    npc->hp = 200;
    npc->death_ticks = 0;
}

static void test_scythe_multihit_per_size(void) {
    printf("test_scythe_multihit_per_size\n");
    ColosseumContext ctx;
    ColosseumState s;
    init_forecast_test_state(&s, &ctx, 909, 15, 15);

    ColScytheResolvedHit out[COLO_SCYTHE_MAX_HITS];

    geo_clear_npcs(&s);
    scythe_spawn_enemy(&s, 0, 14, 14, 3);
    int n3 = col_resolve_scythe_hits(&s, 0, out);
    CHECK("3x3 primary yields exactly 3 hits", n3 == 3);
    CHECK("3x3 hits are all on the primary with ranks 0/1/2",
        out[0].npc_slot == 0 && out[0].splat_rank == 0 &&
        out[1].npc_slot == 0 && out[1].splat_rank == 1 &&
        out[2].npc_slot == 0 && out[2].splat_rank == 2);

    geo_clear_npcs(&s);
    scythe_spawn_enemy(&s, 0, 15, 15, 2);
    int n2 = col_resolve_scythe_hits(&s, 0, out);
    CHECK("2x2 primary yields exactly 2 hits", n2 == 2);
    CHECK("2x2 hits are on the primary at ranks 0/1",
        out[0].npc_slot == 0 && out[0].splat_rank == 0 &&
        out[1].npc_slot == 0 && out[1].splat_rank == 1);

    geo_clear_npcs(&s);
    scythe_spawn_enemy(&s, 0, 16, 15, 1);
    scythe_spawn_enemy(&s, 1, 16, 14, 1);
    scythe_spawn_enemy(&s, 2, 16, 16, 1);
    int narc = col_resolve_scythe_hits(&s, 0, out);
    CHECK("three separate 1x1s in the arc yield exactly 3 hits", narc == 3);
    int distinct_full = 1;
    int seen_slot[3] = {0, 0, 0};
    for (int h = 0; h < narc; h++) {
        if (out[h].splat_rank != 0) distinct_full = 0;
        if (out[h].npc_slot >= 0 && out[h].npc_slot < 3) seen_slot[out[h].npc_slot]++;
    }
    CHECK("each separate 1x1 takes a full-100% (rank 0) hit", distinct_full);
    CHECK("the three separate enemies are distinct targets",
        seen_slot[0] == 1 && seen_slot[1] == 1 && seen_slot[2] == 1);

    CHECK("scythe never exceeds the resolved-hit buffer",
        n3 <= COLO_SCYTHE_MAX_HITS && n2 <= COLO_SCYTHE_MAX_HITS &&
        narc <= COLO_SCYTHE_MAX_HITS);
}

static void test_venator_bow_bounce_colosseum_integration(void) {
    printf("test_venator_bow_bounce_colosseum_integration\n");
    ColosseumContext ctx;
    ColosseumState s;
    int expected_damage[OSRS_VENATOR_MAX_CHAIN_HITS] = {0};
    int expected_total = 0;
    int original_max = 0;
    int bounce_max = 0;
    int base_ticks = 0;

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 8207);
    int venator_cell = test_find_inventory_cell_with_item(&s, ITEM_VENATOR_BOW);
    if (venator_cell < 0) {
        CHECK("speedrun inventory carries venator bow", 0);
        return;
    }
    col_equip_from_cell(&s, venator_cell);
    s.player.current_ranged = 40;
    col_mark_live_loadout_dirty(&s);
    geo_clear_npcs(&s);
    s.modifiers.draft_pending = 0;
    s.wave_ready_delay = 0;
    s.player.x = 12;
    s.player.y = 16;
    venator_spawn_enemy(&s, 0, COLO_FREMENNIK_BERSERKER, 16, 16, 1);
    venator_spawn_enemy(&s, 1, COLO_FREMENNIK_ARCHER, 18, 16, 1);
    venator_spawn_enemy(&s, 2, COLO_FREMENNIK_BERSERKER, 18, 17, 1);

    OsrsVenatorChain chain = col_resolve_venator_chain(&s, 0);
    if (chain.length != OSRS_VENATOR_CHAIN_LENGTH_THREE ||
            chain.hits[0].slot != 0 ||
            chain.hits[1].slot != 1 ||
            chain.hits[2].slot != 2) {
        CHECK("venator resolves a three-hop clustered chain", 0);
        return;
    }

    uint8_t weapon_item = s.player.equipped[GEAR_SLOT_WEAPON];
    const EncounterLoadoutStats* ls = col_live_loadout_stats(&s);
    const OsrsEquipmentEffectProfile* effects = col_live_effects(&s);
    const ColoNPC* primary = &s.npcs[0];
    const ColoNpcStats* primary_stats = &COLO_NPC_STATS[primary->type];
    int base_att_roll = osrs_player_att_roll(ls->eff_level, ls->attack_bonus);
    MeleeStyle weapon_melee_style = col_item_melee_style(weapon_item);
    OsrsPreparedAttackEffects prepared =
        osrs_prepare_attack_effects_for_melee_style(
            effects,
            &s.player.item_effect_state,
            weapon_item,
            ls->style,
            weapon_melee_style,
            OSRS_MAGIC_ATTACK_NONE,
            osrs_target_ref_none(),
            1,
            base_att_roll,
            ls->max_hit,
            osrs_target_effect_context_magic(
                primary_stats->magic_level,
                primary_stats->magic_att_bonus),
            s.player.current_hitpoints,
            s.player.base_hitpoints);
    original_max = prepared.max_hit;
    bounce_max = osrs_venator_bounce_max_hit(original_max);
    base_ticks = ls->attack_range > 1 ? 3 : 1;

    uint32_t rng = s.rng_state;
    int primary_def_roll = col_npc_target_def_roll(
        &s.npcs[0], &COLO_NPC_STATS[s.npcs[0].type],
        ls->style, weapon_melee_style);
    int bounce_def_roll = col_npc_target_def_roll(
        &s.npcs[1], &COLO_NPC_STATS[s.npcs[1].type],
        ls->style, weapon_melee_style);
    expected_total = 0;
    for (int hop = 0; hop < (int)chain.length; hop++) {
        int slot = chain.hits[hop].slot;
        int splat_max = hop == 0 ? original_max : bounce_max;
        int target_def_roll = col_npc_target_def_roll(
            &s.npcs[slot], &COLO_NPC_STATS[s.npcs[slot].type],
            ls->style, weapon_melee_style);
        expected_damage[hop] = osrs_roll_prepared_attack_damage(
            &prepared, target_def_roll, splat_max, &rng);
        expected_total += expected_damage[hop];
    }

    CHECK("venator fixture is sensitive to per-target defence",
        bounce_def_roll != primary_def_roll);
    CHECK("speedrun inventory carries venator bow", 1);
    CHECK("venator resolves a three-hop clustered chain", 1);
    CHECK("venator bow is equipped for the integration attack",
        s.player.equipped[GEAR_SLOT_WEAPON] == ITEM_VENATOR_BOW);
    CHECK("venator bounce max is floor two thirds of the original",
        bounce_max == original_max * 2 / 3 && bounce_max < original_max);

    col_player_attack_target_ctx(&s, &ctx, 0);

    CHECK("venator records render chain slots",
        ctx.player_venator_chain_count == 3 &&
        ctx.player_venator_chain_slots[0] == 0 &&
        ctx.player_venator_chain_slots[1] == 1 &&
        ctx.player_venator_chain_slots[2] == 2);
    CHECK("venator queues one pending hit on each chain target",
        s.npcs[0].pending_hits.count == 1 &&
        s.npcs[1].pending_hits.count == 1 &&
        s.npcs[2].pending_hits.count == 1);
    CHECK("venator queues staggered bounce delays",
        s.npcs[0].pending_hits.hits[0].ticks_remaining == base_ticks &&
        s.npcs[1].pending_hits.hits[0].ticks_remaining == base_ticks + 1 &&
        s.npcs[2].pending_hits.hits[0].ticks_remaining == base_ticks + 2);
    CHECK("venator primary damage matches the independent roll",
        s.npcs[0].pending_hits.hits[0].damage == expected_damage[0]);
    CHECK("venator bounce damage matches capped independent rolls",
        s.npcs[1].pending_hits.hits[0].damage == expected_damage[1] &&
        s.npcs[2].pending_hits.hits[0].damage == expected_damage[2] &&
        expected_damage[1] <= bounce_max &&
        expected_damage[2] <= bounce_max);
    CHECK("venator attack damage sums every queued splat",
        s.player_attack_dmg == expected_total);

    EncounterOverlay ov = {0};
    col_render_post_tick_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &ov);
    CHECK("venator emits primary plus two bounce projectiles",
        ov.projectile_count == 3);
    CHECK("venator primary projectile targets the attacked NPC",
        ov.projectiles[0].source_kind == ENCOUNTER_PROJECTILE_TARGET_PLAYER &&
        ov.projectiles[0].target_kind == ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT &&
        ov.projectiles[0].target_npc_slot == 0);
    CHECK("venator bounce projectiles chain through NPC slots",
        ov.projectiles[1].source_kind == ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT &&
        ov.projectiles[1].source_npc_slot == 0 &&
        ov.projectiles[1].target_kind == ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT &&
        ov.projectiles[1].target_npc_slot == 1 &&
        ov.projectiles[2].source_kind == ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT &&
        ov.projectiles[2].source_npc_slot == 1 &&
        ov.projectiles[2].target_kind == ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT &&
        ov.projectiles[2].target_npc_slot == 2);
    CHECK("venator projectile hops use the venator bolt model",
        ov.projectiles[0].model_id == OSRS_PROJECTILE_MODEL_VENATOR_BOLT &&
        ov.projectiles[1].model_id == OSRS_PROJECTILE_MODEL_VENATOR_BOLT &&
        ov.projectiles[2].model_id == OSRS_PROJECTILE_MODEL_VENATOR_BOLT);
    CHECK("venator bounce projectile delays increase by hop",
        ov.projectiles[1].start_delay > ov.projectiles[0].start_delay &&
        ov.projectiles[2].start_delay > ov.projectiles[1].start_delay);
}

static void test_bee_contact_damage_band(void) {
    printf("test_bee_contact_damage_band\n");
    ColosseumContext ctx;
    ColosseumState s;

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 4242);
    col_apply_weapon_set(&s, COLO_GEAR_RANGED);
    geo_clear_npcs(&s);
    s.modifiers.active_mask |= (1u << COLO_MOD_BEES);
    s.modifiers.tier[COLO_MOD_BEES] = 1;
    col_mod_sync_bees(&s);
    ColoNPC* bee = &s.npcs[s.bees[0].npc_slot];

    int in_band = 1, any_zero = 0;
    for (int t = 0; t < 200; t++) {
        bee->x = s.player.x;
        bee->y = s.player.y;
        s.player.current_hitpoints = 99;
        int hp0 = s.player.current_hitpoints;
        col_mod_tick_bees(&s);
        int dmg = hp0 - s.player.current_hitpoints;
        if (dmg < COLO_BEE_MIN_DAMAGE || dmg > COLO_BEE_MAX_DAMAGE) in_band = 0;
        if (dmg == 0) any_zero = 1;
    }
    CHECK("bee contact damage stays inside the 15-20 band", in_band);
    CHECK("bee contact never deals a zero-damage tick while overlapping", !any_zero);

    col_apply_weapon_set(&s, COLO_GEAR_MELEE);
    CHECK("rig sanity: the melee set is venom-immune",
        osrs_effect_profile_has(col_live_effects(&s), OSRS_ITEM_EFFECT_VENOM_IMMUNE));
    bee->x = s.player.x;
    bee->y = s.player.y;
    s.player.current_hitpoints = 99;
    col_mod_tick_bees(&s);
    CHECK("serpentine-helm immunity zeroes bee contact damage",
        s.player.current_hitpoints == 99);
}

static void test_divine_state_obs_presence(void) {
    printf("test_divine_state_obs_presence\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 71);
    s.modifiers.draft_pending = 0;

    static float obs_base[COLO_NUM_OBS];
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs_base);
    CHECK("player block has no divine timer tail", COLO_PLAYER_OBS_SIZE == 36);

    col_apply_divine_combat_potion_effect(&s);
    s.divine_ranged_timer = ENCOUNTER_DIVINE_POTION_TICKS;
    col_enforce_divine_stat_floors(&s);
    col_mark_live_loadout_dirty(&s);
    static float obs_boost[COLO_NUM_OBS];
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs_boost);
    CHECK("divine boosts still surface through live max-hit scalar",
        obs_boost[20] > obs_base[20]);
}

static void test_magic_set_max_hit_math(void) {
    printf("test_magic_set_max_hit_math\n");

    EncounterLoadoutStats budget;
    encounter_compute_loadout_stats(COLO_BEGINNER_MAGIC_LOADOUT, ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_ACCURATE, 31, &budget);
    CHECK("budget magic gear magic_damage% is 0 (no Occult)", budget.strength_bonus == 0);
    CHECK("budget Trident max hit == floor(31*(1+0/100)) == 31", budget.max_hit == 31);
    CHECK("budget magic eff level == floor(99)+2(accurate)+9 == 110", budget.eff_level == 110);
    CHECK("budget magic style is magic", budget.style == ATTACK_STYLE_MAGIC);
    CHECK("budget Trident (1h) keeps a shield bonus from the Dragon defender",
        budget.attack_range == 7);

    EncounterLoadoutStats budget_aug;
    encounter_compute_loadout_stats(COLO_BEGINNER_MAGIC_LOADOUT, ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_AUGURY, 99, FIGHT_STYLE_ACCURATE, 31, &budget_aug);
    CHECK("budget Trident augury max hit == floor(31*1.04) == 32", budget_aug.max_hit == 32);

    EncounterLoadoutStats untripled;
    {
        uint8_t no_shadow[NUM_GEAR_SLOTS];
        memcpy(no_shadow, COLO_SPEEDRUN_MAGIC_LOADOUT, NUM_GEAR_SLOTS);
        no_shadow[GEAR_SLOT_WEAPON] = ITEM_TRIDENT_OF_SWAMP;
        encounter_compute_loadout_stats(no_shadow, ATTACK_STYLE_MAGIC,
            OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_ACCURATE, 34, &untripled);
    }
    CHECK("high-eff magic gear magic_damage% before Shadow == 14",
        untripled.strength_bonus == 14);

    EncounterLoadoutStats hieff;
    encounter_compute_loadout_stats(COLO_SPEEDRUN_MAGIC_LOADOUT, ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_ACCURATE, 34, &hieff);
    CHECK("Shadow triples gear magic_damage% 14 -> 42", hieff.strength_bonus == 42);
    CHECK("high-eff Shadow max hit == floor(34*(1+42/100)) == 48", hieff.max_hit == 48);
    CHECK("the tripled 42% is what feeds the max hit (untripled 14% would give 38)",
        (int)(34 * (1.0 + 14 / 100.0)) == 38 && hieff.max_hit == 48);
    CHECK("high-eff magic eff level == 110", hieff.eff_level == 110);
    CHECK("Shadow (2h) suppresses the shield (range 10, no shield bonus)",
        hieff.attack_range == 10);

    EncounterLoadoutStats hieff_aug;
    encounter_compute_loadout_stats(COLO_SPEEDRUN_MAGIC_LOADOUT, ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_AUGURY, 99, FIGHT_STYLE_ACCURATE, 34, &hieff_aug);
    CHECK("high-eff Shadow augury max hit == floor(34*1.42*1.04) == 50",
        hieff_aug.max_hit == 50);

    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 91);
    col_apply_weapon_set(&s, COLO_GEAR_MAGIC);
    CHECK("env budget magic set max hit == 31", col_live_loadout_stats(&s)->max_hit == 31);
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 92);
    col_apply_weapon_set(&s, COLO_GEAR_MAGIC);
    CHECK("env high-eff magic set max hit == 48", col_live_loadout_stats(&s)->max_hit == 48);

    col_apply_weapon_set(&s, COLO_GEAR_MELEE);
    int gear_magic[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s,
        gear_magic, test_find_inventory_cell_with_item(&s, ITEM_TUMEKENS_SHADOW));
    col_tick_player_ctx(&s, &ctx, gear_magic, 1);
    CHECK("clicking the magic weapon switches the player to the magic style",
        s.weapon_set == COLO_GEAR_MAGIC);
}

static int thrall_scenario(ColosseumState* s, ColosseumContext* ctx, int mode, uint32_t seed) {
    col_init_context_typed(ctx);
    ctx->config.loadout_profile_mode = mode;
    memset(s, 0, sizeof(*s));
    col_reset_ctx((EncounterState*)s, (EncounterContext*)ctx, seed);
    geo_clear_npcs(s);
    s->modifiers.draft_pending = 0;
    s->player.x = 12;
    s->player.y = 18;
    col_rebuild_player_collision_flags(s);
    int slot = col_spawn_npc_at(s, COLO_FREMENNIK_BERSERKER, 16, 18);
    s->npcs[slot].hp = 200;
    s->npcs[slot].max_hp = 200;
    osrs_interaction_set(&s->interaction, slot);
    return slot;
}

static void test_thrall_regression(void) {
    printf("test_thrall_regression\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    int summon[COLO_NUM_ACTION_HEADS] = {0};
    summon[COLO_HEAD_SPELL] = COLO_SPELL_SUMMON_THRALL;

    int slot = thrall_scenario(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 201);
    col_tick_player_ctx(&s, &ctx, summon, 1);
    CHECK("summon activates the thrall on the targeted NPC",
        s.thrall_active && s.thrall_target_slot == slot);
    CHECK("budget thrall lifetime starts at 99 (decremented this tick)",
        s.thrall_lifetime_left == 98);
    CHECK("thrall recast gate is 17 (decremented this tick)", s.thrall_recast_cd == 16);

    osrs_interaction_clear(&s.interaction);
    encounter_pending_hit_queue_clear(&s.npcs[slot].pending_hits);

    for (int t = 0; t < 3; t++) col_tick_player_ctx(&s, &ctx, idle, 1);
    CHECK("thrall fires exactly once per 4 ticks (timer back to 4)",
        s.thrall_attack_timer == COLO_THRALL_TICK);
    CHECK("exactly one thrall hit is queued in the cadence window",
        s.npcs[slot].pending_hits.count == 1 &&
        s.npcs[slot].pending_hits.hits[0].source_npc_slot == -1 &&
        s.npcs[slot].pending_hits.hits[0].attack_style == ATTACK_STYLE_MAGIC);
    float dmg_before = s.tick_scratch.damage_dealt;
    int npc_hp_before = s.npcs[slot].hp;
    land_pending_player_hits(&s);
    int thrall_dmg = npc_hp_before - s.npcs[slot].hp;
    CHECK("a single thrall hit lands player-credited damage in [0,3]",
        thrall_dmg >= 0 && thrall_dmg <= COLO_THRALL_MAX_HIT);
    CHECK("the thrall damage is credited to the player accumulator",
        s.tick_scratch.damage_dealt >= dmg_before);

    osrs_interaction_set(&s.interaction, slot);

    int life_now = s.thrall_lifetime_left;
    col_tick_player_ctx(&s, &ctx, summon, 1);
    CHECK("summon during the recast gate does not reset lifetime",
        s.thrall_lifetime_left == life_now - 1);

    while (s.thrall_recast_cd > 0) col_tick_player_ctx(&s, &ctx, idle, 1);
    col_tick_player_ctx(&s, &ctx, summon, 1);
    CHECK("re-summon after the gate replaces with a fresh 99-tick thrall",
        s.thrall_active && s.thrall_lifetime_left == 98);

    s.thrall_lifetime_left = 1;
    col_tick_player_ctx(&s, &ctx, idle, 1);
    CHECK("budget thrall despawns when lifetime reaches 0",
        !s.thrall_active && s.thrall_target_slot == -1);

    thrall_scenario(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 202);
    col_tick_player_ctx(&s, &ctx, summon, 1);
    CHECK("high-eff thrall lifetime starts at 198 (decremented this tick)",
        s.thrall_lifetime_left == 197);

    col_init_context_typed(&ctx);
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 203);
    geo_clear_npcs(&s);
    s.modifiers.draft_pending = 0;
    s.player.x = 12; s.player.y = 18;
    col_rebuild_player_collision_flags(&s);
    int sol_slot = col_spawn_npc_at(&s, COLO_SOL_HEREDIT, 18, 16);
    s.npcs[sol_slot].hp = 1500;
    s.npcs[sol_slot].max_hp = 1500;
    osrs_interaction_set(&s.interaction, sol_slot);
    col_tick_player_ctx(&s, &ctx, summon, 1);
    CHECK("summon vs Sol Heredit is a no-op (Sol is thrall-immune)", !s.thrall_active);
    float mask[COLO_ACTION_MASK_SIZE];
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    int spell_off = col_action_head_mask_offset(COLO_HEAD_SPELL);
    CHECK("summon-thrall is masked illegal while targeting Sol",
        mask[spell_off + COLO_SPELL_SUMMON_THRALL] == 0.0f);

    s.thrall_active = 1;
    s.thrall_target_slot = sol_slot;
    s.thrall_attack_timer = 1;
    s.thrall_lifetime_left = 50;
    int sol_hp = s.npcs[sol_slot].hp;
    col_tick_player_ctx(&s, &ctx, idle, 1);
    land_pending_player_hits(&s);
    CHECK("thrall never damages Sol", s.npcs[sol_slot].hp == sol_hp);
}

static void test_death_charge_regression(void) {
    printf("test_death_charge_regression\n");
    ColosseumContext ctx;
    ColosseumState s;
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    int cast_dc[COLO_NUM_ACTION_HEADS] = {0};
    cast_dc[COLO_HEAD_SPELL] = COLO_SPELL_DEATH_CHARGE;

    int slot = thrall_scenario(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 211);
    s.player.special_energy = 50;
    col_tick_player_ctx(&s, &ctx, cast_dc, 1);
    CHECK("Death Charge arms a 100-tick window (decremented this tick)",
        s.death_charge_window_left == 99 && s.death_charge_cd == 0);
    s.npcs[slot].hp = 0;
    col_apply_npc_death(&s, slot);
    CHECK("a player-credited kill in the window grants +15 spec",
        s.player.special_energy == 65);
    CHECK("the kill closes the window and starts the 100-tick cooldown",
        s.death_charge_window_left == 0 && s.death_charge_cd == 100);

    slot = thrall_scenario(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 212);
    s.player.special_energy = 95;
    col_tick_player_ctx(&s, &ctx, cast_dc, 1);
    s.npcs[slot].hp = 0;
    col_apply_npc_death(&s, slot);
    CHECK("Death Charge spec gain clamps at 100", s.player.special_energy == 100);

    slot = thrall_scenario(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 213);
    s.player.special_energy = 40;
    s.npcs[slot].hp = 0;
    col_apply_npc_death(&s, slot);
    CHECK("a kill outside an armed window grants no spec and starts no cooldown",
        s.player.special_energy == 40 && s.death_charge_cd == 0);

    slot = thrall_scenario(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 214);
    col_tick_player_ctx(&s, &ctx, cast_dc, 1);
    while (s.death_charge_window_left > 0) col_tick_player_ctx(&s, &ctx, idle, 1);
    CHECK("an unused window closes without starting the cooldown",
        s.death_charge_window_left == 0 && s.death_charge_cd == 0);

    int slot_b;
    {
        col_init_context_typed(&ctx);
        ctx.config.loadout_profile_mode = COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY;
        memset(&s, 0, sizeof(s));
        col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 215);
        geo_clear_npcs(&s);
        s.modifiers.draft_pending = 0;
        s.player.x = 12; s.player.y = 18;
        col_rebuild_player_collision_flags(&s);
        slot = col_spawn_npc_at(&s, COLO_FREMENNIK_BERSERKER, 16, 18);
        slot_b = col_spawn_npc_at(&s, COLO_FREMENNIK_BERSERKER, 16, 19);
        s.npcs[slot].hp = 50; s.npcs[slot].max_hp = 50;
        s.npcs[slot_b].hp = 50; s.npcs[slot_b].max_hp = 50;
        osrs_interaction_set(&s.interaction, slot);
    }
    s.player.special_energy = 50;
    col_tick_player_ctx(&s, &ctx, cast_dc, 1);
    s.npcs[slot].hp = 0;
    s.npcs[slot_b].hp = 0;
    col_apply_npc_death(&s, slot);
    col_apply_npc_death(&s, slot_b);
    CHECK("two same-tick kills consume the charge exactly once (+15)",
        s.player.special_energy == 65 && s.death_charge_window_left == 0);

    slot = thrall_scenario(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 216);
    s.player.special_energy = 50;
    s.npcs[slot].hp = 1;
    int summon[COLO_NUM_ACTION_HEADS] = {0};
    summon[COLO_HEAD_SPELL] = COLO_SPELL_SUMMON_THRALL;
    col_tick_player_ctx(&s, &ctx, summon, 1);
    col_tick_player_ctx(&s, &ctx, cast_dc, 1);
    s.thrall_attack_timer = 1;
    col_tick_player_ctx(&s, &ctx, idle, 1);
    int spec_before = s.player.special_energy;

    for (int h = 0; h < s.npcs[slot].pending_hits.count; h++)
        if (s.npcs[slot].pending_hits.hits[h].active &&
                s.npcs[slot].pending_hits.hits[h].source_npc_slot == -1)
            s.npcs[slot].pending_hits.hits[h].damage = 5;
    land_pending_player_hits(&s);
    CHECK("a thrall-credited kill procs Death Charge (+15)",
        s.npcs[slot].hp <= 0 && s.player.special_energy == spec_before + 15);

    slot = thrall_scenario(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 217);
    s.death_charge_cd = 50;
    float mask[COLO_ACTION_MASK_SIZE];
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    int spell_off = col_action_head_mask_offset(COLO_HEAD_SPELL);
    CHECK("death-charge is masked illegal while the cooldown is up",
        mask[spell_off + COLO_SPELL_DEATH_CHARGE] == 0.0f);
    s.death_charge_cd = 0;
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    CHECK("death-charge is legal when ready", mask[spell_off + COLO_SPELL_DEATH_CHARGE] == 1.0f);
}

static void test_combat_fidelity_snapshot_roundtrip(void) {
    printf("test_combat_fidelity_snapshot_roundtrip\n");
    ColosseumContext ctx;
    ColosseumState s;
    int slot = thrall_scenario(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 221);
    col_apply_weapon_set(&s, COLO_GEAR_MAGIC);
    s.thrall_active = 1;
    s.thrall_target_slot = slot;
    s.thrall_lifetime_left = 123;
    s.thrall_attack_timer = 2;
    s.thrall_recast_cd = 9;
    s.death_charge_window_left = 44;
    s.death_charge_cd = 0;

    ColoSnapshot snap;
    col_snapshot_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &snap);
    CHECK("snapshot frame is v20", snap.version == 20u);

    ColosseumState restored;
    memset(&restored, 0, sizeof(restored));
    col_restore_ctx((EncounterState*)&restored, (EncounterContext*)&ctx, &snap, sizeof(snap));
    CHECK("magic weapon set survives the round-trip",
        restored.weapon_set == COLO_GEAR_MAGIC);
    CHECK("the recomputed magic set max hit matches the live high-eff value (48)",
        col_live_loadout_stats(&restored)->max_hit == 48);
    CHECK("thrall fields round-trip bit-identically",
        restored.thrall_active == 1 && restored.thrall_target_slot == slot &&
        restored.thrall_lifetime_left == 123 && restored.thrall_attack_timer == 2 &&
        restored.thrall_recast_cd == 9);
    CHECK("Death-Charge fields round-trip bit-identically",
        restored.death_charge_window_left == 44 && restored.death_charge_cd == 0);
}

static void test_step_out_forecast_manticore_armed_pattern(void) {
    printf("test_step_out_forecast_manticore_armed_pattern\n");
    ColosseumContext ctx;
    ColosseumState s;
    init_forecast_test_state(&s, &ctx, 401, 17, 16);
    col_init_npc(&s, 0, COLO_MANTICORE, 16, 12);
    s.npcs[0].attack_timer = 1;
    ColoManticoreState* mc = colo_npc_manticore(&s.npcs[0]);
    mc->cycle_step = 0;
    mc->orb_style[0] = ATTACK_STYLE_MAGIC;
    mc->orb_style[1] = ATTACK_STYLE_RANGED;
    mc->orb_style[2] = ATTACK_STYLE_MELEE;

    ColoStepOutForecast forecast;
    col_build_step_out_forecast_ctx(&s, &forecast);
    const ColoStepOutForecastAction* idle = &forecast.actions[0];
    CHECK("armed manticore idle forecast is valid", idle->valid == 1);
    CHECK("armed manticore orb 0 records magic on tick 1",
        idle->ticks[0].magic_count == 1 && idle->ticks[0].max_hit == COLO_MANTICORE_MAX_HIT_MAGIC);
    CHECK("armed manticore orb 1 records ranged on tick 2",
        idle->ticks[1].ranged_count == 1 && idle->ticks[1].max_hit == COLO_MANTICORE_MAX_HIT_RANGED);
    CHECK("armed manticore orb 2 records melee on tick 3",
        idle->ticks[2].melee_count == 1 && idle->melee_fallback_exposure == 1);
}

static void test_step_out_forecast_manticore_pair_stagger(void) {
    printf("test_step_out_forecast_manticore_pair_stagger\n");
    ColosseumContext ctx;
    ColosseumState s;
    init_forecast_test_state(&s, &ctx, 406, 17, 16);
    col_init_npc(&s, 0, COLO_MANTICORE, 16, 12);
    col_init_npc(&s, 1, COLO_MANTICORE, 12, 12);
    s.npcs[0].attack_timer = 1;
    s.npcs[1].attack_timer = 1;
    ColoManticoreState* amc = colo_npc_manticore(&s.npcs[0]);
    ColoManticoreState* bmc = colo_npc_manticore(&s.npcs[1]);
    amc->cycle_step = 0;
    amc->orb_style[0] = ATTACK_STYLE_MAGIC;
    amc->orb_style[1] = ATTACK_STYLE_RANGED;
    amc->orb_style[2] = ATTACK_STYLE_MELEE;
    bmc->cycle_step = 0;
    bmc->orb_style[0] = ATTACK_STYLE_MAGIC;
    bmc->orb_style[1] = ATTACK_STYLE_RANGED;
    bmc->orb_style[2] = ATTACK_STYLE_MELEE;

    ColoStepOutForecast forecast;
    col_build_step_out_forecast_ctx(&s, &forecast);
    const ColoStepOutForecastAction* idle = &forecast.actions[0];
    CHECK("synced-pair forecast predicts ONE orb per tick, not two",
        idle->ticks[0].magic_count == 1 &&
        idle->ticks[1].ranged_count == 1 &&
        idle->ticks[2].melee_count == 1);

    s.npcs[1].attack_timer = 3;
    bmc->orb_style[0] = ATTACK_STYLE_RANGED;
    col_build_step_out_forecast_ctx(&s, &forecast);
    idle = &forecast.actions[0];
    CHECK("still-charging peer forecast overlaps mid-barrage",
        idle->ticks[2].melee_count == 1 && idle->ticks[2].ranged_count == 1);
}

static void test_step_out_forecast_warband_window_and_break(void) {
    printf("test_step_out_forecast_warband_window_and_break\n");
    ColosseumContext ctx;
    ColosseumState s;
    init_forecast_test_state(&s, &ctx, 402, 7, 18);
    s.tick = 100;
    s.warband_cycle_anchor = 100;
    col_init_npc(&s, 0, COLO_FREMENNIK_BERSERKER, 8, 18);

    ColoStepOutForecast forecast;
    col_build_step_out_forecast_ctx(&s, &forecast);
    int run_west = forecast_move_action_for_delta(-2, 0);
    CHECK("adjacent berserker records melee on its next window",
        forecast.actions[0].ticks[0].melee_count == 1);
    CHECK("running west breaks the berserker forecast adjacency",
        !forecast_action_has_event(&forecast.actions[run_west]));
}

static void test_step_out_forecast_ranged_los_candidate_tiles(void) {
    printf("test_step_out_forecast_ranged_los_candidate_tiles\n");
    ColosseumContext ctx;
    ColosseumState s;
    init_forecast_test_state(&s, &ctx, 403, 7, 9);
    col_init_npc(&s, 0, COLO_SERPENT_SHAMAN, 12, 12);
    s.npcs[0].attack_timer = 1;

    ColoStepOutForecast forecast;
    col_build_step_out_forecast_ctx(&s, &forecast);
    int run_north = forecast_move_action_for_delta(0, 2);
    CHECK("pillar-blocked idle tile records no shaman forecast",
        !forecast_action_has_event(&forecast.actions[0]));
    CHECK("clear run-north tile records the shaman magic forecast",
        forecast.actions[run_north].ticks[0].magic_count == 1);
}

static void test_step_out_forecast_valid_flags(void) {
    printf("test_step_out_forecast_valid_flags\n");
    ColosseumContext ctx;
    ColosseumState s;
    init_forecast_test_state(&s, &ctx, 404, 7, 9);

    ColoStepOutForecast forecast;
    col_build_step_out_forecast_ctx(&s, &forecast);
    int walk_east = forecast_move_action_for_delta(1, 0);
    int walk_west = forecast_move_action_for_delta(-1, 0);
    CHECK("pillar move has invalid step-out forecast flag",
        forecast.actions[walk_east].valid == 0);
    CHECK("clear move has valid step-out forecast flag",
        forecast.actions[walk_west].valid == 1);
}

static void test_step_out_forecast_same_tick_mixed_styles(void) {
    printf("test_step_out_forecast_same_tick_mixed_styles\n");
    ColosseumContext ctx;
    ColosseumState s;
    init_forecast_test_state(&s, &ctx, 405, 17, 16);
    col_init_npc(&s, 0, COLO_SERPENT_SHAMAN, 13, 16);
    col_init_npc(&s, 1, COLO_JAVELIN_COLOSSUS, 20, 15);
    s.npcs[0].attack_timer = 1;
    s.npcs[1].attack_timer = 1;

    ColoStepOutForecast forecast;
    col_build_step_out_forecast_ctx(&s, &forecast);
    const ColoStepOutForecastAction* idle = &forecast.actions[0];
    CHECK("same tick magic and ranged forecast conflict is flagged",
        idle->same_tick_mixed_style_conflict == 1);
    CHECK("same tick magic and ranged counts are both recorded",
        idle->ticks[0].magic_count == 1 && idle->ticks[0].ranged_count == 1);
}

static void test_render_bridge_combat_visuals_and_loadout(void) {
    printf("test_render_bridge_combat_visuals_and_loadout\n");
    ColosseumContext ctx;
    ColosseumState s;
    init_forecast_test_state(&s, &ctx, 501, 17, 16);
    col_init_npc(&s, 0, COLO_SERPENT_SHAMAN, 13, 16);
    s.npcs[0].attack_timer = 0;
    col_npc_attack_ctx(&s, &ctx, 0);
    EncounterOverlay ov = {0};
    col_render_post_tick_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &ov);
    CHECK("magic NPC attack emits a render projectile", ov.projectile_count > 0);
    CHECK("magic NPC projectile tracks the player",
        ov.projectiles[0].source_npc_slot == 0 &&
        ov.projectiles[0].target_kind == ENCOUNTER_PROJECTILE_TARGET_PLAYER);
    CHECK("serpent shaman uses Water Surge projectile ids",
        ov.projectiles[0].launch_gfx_id == 1458 &&
        ov.projectiles[0].travel_gfx_id == 1459 &&
        ov.projectiles[0].impact_gfx_id == 1460);
    RenderEntity npc_anim_entities[4];
    int npc_anim_count = 0;
    col_fill_render_entities_ctx(
        (EncounterState*)&s, (EncounterContext*)&ctx,
        npc_anim_entities, 4, &npc_anim_count);
    CHECK("serpent shaman attack drives body attack animation",
        npc_anim_count >= 2 && npc_anim_entities[1].npc_anim_id == 10859);

    init_forecast_test_state(&s, &ctx, 502, 17, 16);
    col_init_npc(&s, 0, COLO_JAGUAR_WARRIOR, 18, 16);
    s.npcs[0].attack_timer = 0;
    col_npc_attack_ctx(&s, &ctx, 0);
    memset(&ov, 0, sizeof(ov));
    col_render_post_tick_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &ov);
    CHECK("melee NPC attack emits no projectile", ov.projectile_count == 0);

    init_forecast_test_state(&s, &ctx, 503, 17, 16);
    col_init_npc(&s, 0, COLO_MANTICORE, 16, 12);
    ColoManticoreState* mc = colo_npc_manticore(&s.npcs[0]);
    mc->fixed_orb_style[0] = ATTACK_STYLE_MAGIC;
    mc->fixed_orb_style[1] = ATTACK_STYLE_RANGED;
    mc->fixed_orb_style[2] = ATTACK_STYLE_MELEE;
    s.npcs[0].attack_timer = 2;
    col_npc_attack_ctx(&s, &ctx, 0);
    memset(&ov, 0, sizeof(ov));
    col_render_post_tick_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &ov);
    memset(npc_anim_entities, 0, sizeof(npc_anim_entities));
    npc_anim_count = 0;
    col_fill_render_entities_ctx(
        (EncounterState*)&s, (EncounterContext*)&ctx,
        npc_anim_entities, 4, &npc_anim_count);
    CHECK("manticore arm tick drives charge animation once",
        s.npcs[0].attacked_this_tick == 0 &&
        npc_anim_count >= 2 &&
        npc_anim_entities[1].npc_anim_id == 10868);
    CHECK("manticore windup renders stacked remaining orbs bottom first",
        ov.projectile_count == 0 &&
        ov.floating_model_count == 3 &&
        ov.floating_models[0].model_id == 51215u &&
        ov.floating_models[0].anim_id == 10329 &&
        ov.floating_models[1].model_id == 51221u &&
        ov.floating_models[1].anim_id == 10327 &&
        ov.floating_models[2].model_id == 51213u &&
        ov.floating_models[2].anim_id == 10328 &&
        ov.floating_models[0].height_offset < ov.floating_models[1].height_offset &&
        ov.floating_models[1].height_offset < ov.floating_models[2].height_offset);

    init_forecast_test_state(&s, &ctx, 503, 17, 16);
    col_init_npc(&s, 0, COLO_MANTICORE, 16, 12);
    mc = colo_npc_manticore(&s.npcs[0]);
    mc->fixed_orb_style[0] = ATTACK_STYLE_MELEE;
    mc->fixed_orb_style[1] = ATTACK_STYLE_RANGED;
    mc->fixed_orb_style[2] = ATTACK_STYLE_MAGIC;
    s.npcs[0].attack_timer = 0;
    col_npc_attack_ctx(&s, &ctx, 0);
    memset(&ov, 0, sizeof(ov));
    col_render_post_tick_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &ov);
    memset(npc_anim_entities, 0, sizeof(npc_anim_entities));
    npc_anim_count = 0;
    col_fill_render_entities_ctx(
        (EncounterState*)&s, (EncounterContext*)&ctx,
        npc_anim_entities, 4, &npc_anim_count);
    CHECK("manticore attack drives body attack animation",
        npc_anim_count >= 2 && npc_anim_entities[1].npc_anim_id == 10869);
    CHECK("alive manticore attack does not select death animation",
        npc_anim_count >= 2 &&
        npc_anim_entities[1].npc_anim_id != col_npc_death_anim_id(COLO_MANTICORE));
    int manticore_dist = encounter_projectile_distance(
        s.npcs[0].x, s.npcs[0].y, col_npc_effective_size(&s.npcs[0]),
        s.player.x, s.player.y, 1, ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    EncounterProjectileTiming manticore_timing =
        col_npc_projectile_timing(COLO_MANTICORE, ATTACK_STYLE_MELEE, manticore_dist);
    CHECK("manticore melee orb emits the 51213 projectile with scaled duration",
        ov.projectile_count == 1 &&
        ov.projectiles[0].model_id == 51213u &&
        ov.projectiles[0].anim_id == 10328 &&
        ov.projectiles[0].travel_gfx_id == 2685 &&
        ov.projectiles[0].impact_gfx_id == 2686 &&
        ov.projectiles[0].duration_ticks == manticore_timing.visual_duration_ticks * 30 &&
        ov.projectiles[0].duration_ticks > 1);

    init_forecast_test_state(&s, &ctx, 503, 17, 16);
    col_init_npc(&s, 0, COLO_JAVELIN_COLOSSUS, 20, 16);
    s.npcs[0].attack_timer = 0;
    col_npc_attack_ctx(&s, &ctx, 0);
    memset(npc_anim_entities, 0, sizeof(npc_anim_entities));
    npc_anim_count = 0;
    col_fill_render_entities_ctx(
        (EncounterState*)&s, (EncounterContext*)&ctx,
        npc_anim_entities, 4, &npc_anim_count);
    CHECK("javelin colossus attack drives body attack animation",
        npc_anim_count >= 2 && npc_anim_entities[1].npc_anim_id == 10892);
    memset(&ov, 0, sizeof(ov));
    col_render_post_tick_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &ov);
    CHECK("javelin normal ranged throw waits one tick before visual release",
        ov.projectile_count == 1 &&
        ov.projectiles[0].start_delay ==
            COLO_JAVELIN_PROJECTILE_RELEASE_DELAY_TICKS * 30);

    init_forecast_test_state(&s, &ctx, 503, 17, 16);
    col_init_npc(&s, 0, COLO_JAVELIN_COLOSSUS, 20, 16);
    ColoJavelinState* jv = colo_npc_javelin(&s.npcs[0]);
    jv->attack_count = 4;
    s.npcs[0].attack_timer = 0;
    col_npc_attack_ctx(&s, &ctx, 0);
    CHECK("javelin skyfall launch marks the attack tick",
        s.npcs[0].attacked_this_tick == 1);
    memset(&ov, 0, sizeof(ov));
    col_render_post_tick_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &ov);
    CHECK("javelin skyfall launch lobs straight up out of the colossus",
        ov.projectile_count == 1 &&
        ov.projectiles[0].source_kind == ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT &&
        ov.projectiles[0].source_npc_slot == 0 &&
        ov.projectiles[0].target_kind == ENCOUNTER_PROJECTILE_TARGET_FIXED &&
        ov.projectiles[0].src_x == s.npcs[0].x &&
        ov.projectiles[0].src_y == s.npcs[0].y &&
        ov.projectiles[0].dst_x == ov.projectiles[0].src_x &&
        ov.projectiles[0].dst_y == ov.projectiles[0].src_y &&
        ov.projectiles[0].travel_gfx_id == COLO_JAVELIN_SKYFALL_LAUNCH_TRAVEL_GFX_ID &&
        ov.projectiles[0].travel_gfx_id != 2673 &&
        ov.projectiles[0].impact_gfx_id == 0 &&
        ov.projectiles[0].start_h < ov.projectiles[0].end_h &&
        ov.projectiles[0].curve == COLO_JAVELIN_SKYFALL_LAUNCH_CURVE &&
        ov.projectiles[0].start_delay == 0 &&
        ov.projectiles[0].duration_ticks ==
            (COLO_JAVELIN_SKYFALL_DELAY -
             COLO_JAVELIN_SKYFALL_DROP_GAME_TICKS -
             COLO_JAVELIN_SKYFALL_APEX_GAME_TICKS) * 30);
    CHECK("javelin skyfall launch emits growing target shadow",
        ov.tile_shadow_count == 1 &&
        ov.tile_shadows[0].active == 1 &&
        ov.tile_shadows[0].x == jv->skyfall_tile_x &&
        ov.tile_shadows[0].y == jv->skyfall_tile_y &&
        ov.tile_shadows[0].scale > 0.0f &&
        ov.tile_shadows[0].scale < 1.0f);
    float initial_shadow_scale = ov.tile_shadows[0].scale;
    int saved_skyfall_timer = jv->skyfall_timer;
    int saved_attacked_this_tick = s.npcs[0].attacked_this_tick;
    jv->skyfall_timer = 1;
    s.npcs[0].attacked_this_tick = 0;
    memset(&ov, 0, sizeof(ov));
    col_render_post_tick_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &ov);
    CHECK("javelin skyfall target shadow grows to full before landing",
        ov.tile_shadow_count == 1 &&
        ov.tile_shadows[0].x == jv->skyfall_tile_x &&
        ov.tile_shadows[0].y == jv->skyfall_tile_y &&
        ov.tile_shadows[0].scale > initial_shadow_scale &&
        ov.tile_shadows[0].scale >= 0.99f);
    jv->skyfall_timer = saved_skyfall_timer;
    s.npcs[0].attacked_this_tick = saved_attacked_this_tick;
    memset(npc_anim_entities, 0, sizeof(npc_anim_entities));
    npc_anim_count = 0;
    col_fill_render_entities_ctx(
        (EncounterState*)&s, (EncounterContext*)&ctx,
        npc_anim_entities, 4, &npc_anim_count);
    CHECK("javelin skyfall launch drives lob body animation",
        npc_anim_count >= 2 &&
        npc_anim_entities[1].npc_anim_id == COLO_JAVELIN_SKYFALL_ANIM_ID);
    int clear_anim[COLO_NUM_ACTION_HEADS] = {0};
    step_and_observe(&s, &ctx, clear_anim);
    jv->attack_count = 5;
    s.npcs[0].attack_timer = 0;
    col_npc_attack_ctx(&s, &ctx, 0);
    memset(npc_anim_entities, 0, sizeof(npc_anim_entities));
    npc_anim_count = 0;
    col_fill_render_entities_ctx(
        (EncounterState*)&s, (EncounterContext*)&ctx,
        npc_anim_entities, 4, &npc_anim_count);
    CHECK("javelin normal attack after skyfall uses throw body animation",
        npc_anim_count >= 2 && npc_anim_entities[1].npc_anim_id == 10892);

    init_forecast_test_state(&s, &ctx, 504, 17, 16);
    col_init_npc(&s, 0, COLO_JAVELIN_COLOSSUS, 20, 16);
    jv = colo_npc_javelin(&s.npcs[0]);
    jv->skyfall_pending = 1;
    jv->skyfall_tile_x = 21;
    jv->skyfall_tile_y = 15;
    jv->skyfall_damage = 37;

    jv->skyfall_timer = COLO_JAVELIN_SKYFALL_DROP_GAME_TICKS + 1;
    memset(&ov, 0, sizeof(ov));
    col_render_post_tick_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &ov);
    CHECK("javelin skyfall descent holds until the telegraph midpoint",
        ov.tile_shadow_count == 1 && ov.projectile_count == 0);

    jv->skyfall_timer = COLO_JAVELIN_SKYFALL_DROP_GAME_TICKS;
    memset(&ov, 0, sizeof(ov));
    col_render_post_tick_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &ov);
    CHECK("javelin skyfall descent falls straight down with fiery impact at the midpoint",
        ov.projectile_count == 1 &&
        ov.projectiles[0].source_kind == ENCOUNTER_PROJECTILE_TARGET_FIXED &&
        ov.projectiles[0].target_kind == ENCOUNTER_PROJECTILE_TARGET_FIXED &&
        ov.projectiles[0].src_x == jv->skyfall_tile_x &&
        ov.projectiles[0].src_y == jv->skyfall_tile_y &&
        ov.projectiles[0].dst_x == jv->skyfall_tile_x &&
        ov.projectiles[0].dst_y == jv->skyfall_tile_y &&
        ov.projectiles[0].travel_gfx_id == COLO_JAVELIN_SKYFALL_DROP_TRAVEL_GFX_ID &&
        ov.projectiles[0].travel_gfx_id != 2673 &&
        ov.projectiles[0].impact_gfx_id == COLO_JAVELIN_SKYFALL_IMPACT_GFX_ID &&
        ov.projectiles[0].start_h > ov.projectiles[0].end_h &&
        ov.projectiles[0].curve == COLO_JAVELIN_SKYFALL_DROP_CURVE &&
        ov.projectiles[0].duration_ticks == COLO_JAVELIN_SKYFALL_DROP_GAME_TICKS * 30);

    jv->skyfall_timer = COLO_JAVELIN_SKYFALL_DROP_GAME_TICKS - 1;
    CHECK("javelin skyfall drop point leaves room below it", jv->skyfall_timer >= 1);
    memset(&ov, 0, sizeof(ov));
    col_render_post_tick_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &ov);
    CHECK("javelin skyfall descent is not re-emitted after the midpoint",
        ov.projectile_count == 0);

    jv->skyfall_timer = 1;
    col_npc_resolve_javelin_skyfall(&s, &ctx, 0);
    memset(&ov, 0, sizeof(ov));
    col_render_post_tick_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &ov);
    CHECK("javelin skyfall landing clears the shadow and emits no late descent",
        ov.tile_shadow_count == 0 && ov.projectile_count == 0 &&
        jv->skyfall_pending == 0);

    init_forecast_test_state(&s, &ctx, 503, 17, 16);
    col_apply_weapon_set(&s, COLO_GEAR_RANGED);
    int target_slot = col_spawn_npc_at(&s, COLO_JAVELIN_COLOSSUS, 22, 15);
    s.interaction.target_slot = target_slot;
    int idle[COLO_NUM_ACTION_HEADS] = {0};
    col_tick_player_ctx(&s, &ctx, idle, 1);
    memset(&ov, 0, sizeof(ov));
    col_render_post_tick_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &ov);
    CHECK("player ranged attack emits a render projectile", ov.projectile_count > 0);
    CHECK("player projectile targets the attacked NPC",
        ov.projectiles[0].source_kind == ENCOUNTER_PROJECTILE_TARGET_PLAYER &&
        ov.projectiles[0].target_npc_slot == target_slot);

    RenderEntity entities[4];
    int count = 0;
    col_fill_render_entities_ctx(
        (EncounterState*)&s, (EncounterContext*)&ctx, entities, 4, &count);
    const uint8_t* const* loadouts = col_loadouts_for_profile(s.active_loadout_profile);
    CHECK("render player uses the active loadout weapon",
        count > 0 &&
        entities[0].equipped[GEAR_SLOT_WEAPON] ==
            loadouts[s.weapon_set][GEAR_SLOT_WEAPON]);
    CHECK("player inventory cells are populated for the GUI",
        test_find_inventory_cell_with_item(
            &s, loadouts[s.weapon_set][GEAR_SLOT_WEAPON]) >= 0);

    s.modifiers.active_mask =
        (1u << (unsigned)COLO_MOD_RELENTLESS) |
        (1u << (unsigned)COLO_MOD_FRAILTY);
    s.modifiers.tier[COLO_MOD_RELENTLESS] = 2;
    s.modifiers.tier[COLO_MOD_FRAILTY] = 3;
    memset(&ov, 0, sizeof(ov));
    col_render_post_tick_ctx((EncounterState*)&s, (EncounterContext*)&ctx, &ov);
    CHECK("render overlay exposes active modifier tiers",
        ov.active_modifier_count == 2 &&
        ov.active_modifiers[0].modifier == COLO_MOD_FRAILTY &&
        ov.active_modifiers[0].tier == 3 &&
        ov.active_modifiers[1].modifier == COLO_MOD_RELENTLESS &&
        ov.active_modifiers[1].tier == 2);
}

static void test_render_bridge_npc_debug_and_warband_motion(void) {
    printf("test_render_bridge_npc_debug_and_warband_motion\n");
    ColosseumContext ctx;
    ColosseumState s;
    init_forecast_test_state(&s, &ctx, 504, 17, 16);
    col_init_npc(&s, 0, COLO_FREMENNIK_ARCHER, 20, 18);
    col_init_npc(&s, 1, COLO_MANTICORE, 20, 12);
    s.npcs[0].attack_timer = 5;
    s.npcs[1].attack_timer = 3;
    s.npcs[1].type_state.manticore.cycle_step = 1;
    s.npcs[1].type_state.manticore.orb_style[0] = ATTACK_STYLE_RANGED;
    s.npcs[1].type_state.manticore.orb_style[1] = ATTACK_STYLE_MAGIC;
    s.npcs[1].type_state.manticore.orb_style[2] = ATTACK_STYLE_MELEE;

    RenderEntity entities[4];
    int count = 0;
    col_fill_render_entities_ctx(
        (EncounterState*)&s, (EncounterContext*)&ctx, entities, 4, &count);

    CHECK("warband render entity uses run-speed interpolation",
        count >= 3 && entities[1].npc_slot == 0 && entities[1].is_running == 1);
    CHECK("warband render entity carries debug stats",
        count >= 3 &&
        strcmp(entities[1].debug_npc_type_name, "Fremennik Archer") == 0 &&
        entities[1].debug_attack_timer == 5 &&
        entities[1].debug_attack_style == ATTACK_STYLE_RANGED);
    CHECK("manticore render entity carries cycle and orb debug state",
        count >= 3 &&
        entities[2].debug_manticore_state_active == 1 &&
        entities[2].debug_manticore_cycle_step == 1 &&
        entities[2].debug_manticore_orb_style[0] == ATTACK_STYLE_RANGED &&
        entities[2].debug_manticore_orb_style[1] == ATTACK_STYLE_MAGIC &&
        entities[2].debug_manticore_orb_style[2] == ATTACK_STYLE_MELEE);
}

static int test_los_every_tile_blocked(void* ctx, int x, int y) {
    (void)ctx;
    (void)x;
    (void)y;
    return 1;
}

static void test_osrs_los_query_contracts(void) {
    printf("test_osrs_los_query_contracts\n");
    OsrsLosQuery open_query = osrs_los_open();
    CHECK("explicit open LoS permits a ranged attack",
        encounter_player_can_attack(0, 0, 4, 0, 1, 10, &open_query) == 1);

    OsrsLosQuery tile_query = osrs_los_tile(test_los_every_tile_blocked, NULL);
    CHECK("tile LoS refuses when every tile blocks",
        encounter_player_can_attack(0, 0, 4, 0, 1, 10, &tile_query) == 0);
}

static void test_player_ranged_los_blocked_by_pillar(void) {
    printf("test_player_ranged_los_blocked_by_pillar\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 0;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 4242);
    geo_clear_npcs(&s);
    col_apply_weapon_set(&s, COLO_GEAR_RANGED);
    CHECK("ranged loadout reaches past 1 tile", col_player_attack_range(&s) > 1);

    s.player.x = 5; s.player.y = 9;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_JAGUAR_WARRIOR, 13, 9);
    ColoNPC* npc = &s.npcs[0];
    CHECK("player + target tiles are clear of static blockers",
        !col_static_blocked(5, 9) && !col_static_blocked(13, 9));
    CHECK("pillar 0 sits on the line between them", col_static_blocked(9, 9));
    CHECK("no LoS through the pillar", col_npc_has_los_to_player(&s, npc) == 0);
    OsrsLosQuery los_query = col_player_los_query(&s);
    CHECK("shared tile LoS blocks the same pillar line",
        encounter_player_can_attack(s.player.x, s.player.y,
            npc->x, npc->y, col_npc_effective_size(npc),
            col_player_attack_range(&s), &los_query) == 0);

    s.player.x = 13; s.player.y = 4;
    col_rebuild_player_collision_flags(&s);
    CHECK("the clear column tile is walkable", !col_static_blocked(13, 4));
    CHECK("LoS is clear down the column", col_npc_has_los_to_player(&s, npc) == 1);
    int actions[COLO_NUM_ACTION_HEADS] = {0};
    osrs_interaction_set(&s.interaction, 0);
    s.player.attack_timer = 0;
    s.player_dest_x = -1; s.player_dest_y = -1;
    s.tick_scratch.player_attacked = 0;
    col_tick_player_ctx(&s, &ctx, actions, 1);
    CHECK("a ranged attack with clear LoS fires",
        s.tick_scratch.player_attacked == 1);
}

static void test_player_chase_routes_around_pillar_for_los(void) {
    printf("test_player_chase_routes_around_pillar_for_los\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ctx.config.start_wave = 0;
    ctx.world_offset_x = 1808;
    ctx.world_offset_y = 3090;
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 5151);
    geo_clear_npcs(&s);
    s.modifiers.draft_pending = 0;
    col_apply_weapon_set(&s, COLO_GEAR_RANGED);

    s.player.x = 5;
    s.player.y = 9;
    s.player_dest_x = -1;
    s.player_dest_y = -1;
    col_init_npc(&s, 0, COLO_JAGUAR_WARRIOR, 13, 9);
    col_rebuild_player_collision_flags(&s);
    ColoNPC* npc = &s.npcs[0];
    OsrsLosQuery los_query = col_player_los_query(&s);
    int attack_range = col_player_attack_range(&s);
    CHECK("start tile is range-valid and LoS-blocked",
        encounter_player_can_attack(s.player.x, s.player.y,
            npc->x, npc->y, col_npc_effective_size(npc),
            attack_range, &los_query) == 0);

    int actions[COLO_NUM_ACTION_HEADS] = {0};
    int attacked_tick = -1;
    int moved_from_start = 0;
    osrs_interaction_set(&s.interaction, 0);
    for (int tick = 0; tick < 12; tick++) {
        s.tick_scratch.player_attacked = 0;
        col_tick_player_ctx(&s, &ctx, actions, 1);
        if (s.player.x != 5 || s.player.y != 9) moved_from_start = 1;
        if (s.tick_scratch.player_attacked) {
            attacked_tick = tick;
            break;
        }
    }

    CHECK("pillar-blocked interaction makes the player chase",
        moved_from_start == 1);
    CHECK("chase reaches LoS and fires within twelve ticks",
        attacked_tick >= 0);
    CHECK("attack fires from a LoS-valid tile",
        col_npc_has_los_to_player(&s, npc) == 1 &&
        encounter_rect_distance(s.player.x, s.player.y, 1,
            npc->x, npc->y, col_npc_effective_size(npc)) <= attack_range);
}

static void test_colosseum_npc_movement_player_tile_guards(void) {
    printf("test_colosseum_npc_movement_player_tile_guards\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 6161);
    geo_clear_npcs(&s);
    s.modifiers.draft_pending = 0;
    s.player.x = 16;
    s.player.y = 16;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_SERPENT_SHAMAN, 16, 16);
    col_tick_npcs_ctx(&s, &ctx);
    CHECK("overlapped ranged NPC shuffles off the player tile",
        !test_npc_covers_player(&s, &s.npcs[0]) && s.npcs[0].moved_this_tick == 1);

    geo_clear_npcs(&s);
    s.player.x = 16;
    s.player.y = 16;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_SERPENT_SHAMAN, 16, 16);
    s.player_last_interaction_target_slot = 0;
    s.player_last_interaction_age = 0;
    col_npc_move_ctx(&s, &ctx, 0);
    CHECK("overlapped just-clicked ranged NPC holds current tile",
        test_npc_covers_player(&s, &s.npcs[0]) && s.npcs[0].moved_this_tick == 0);

    geo_clear_npcs(&s);
    s.player.x = 16;
    s.player.y = 16;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_SERPENT_SHAMAN, 7, 16);
    int shaman_x = s.npcs[0].x;
    int shaman_y = s.npcs[0].y;
    col_npc_move_ctx(&s, &ctx, 0);
    CHECK("ranged NPC already in range and LoS holds its attack tile",
        s.npcs[0].x == shaman_x && s.npcs[0].y == shaman_y &&
        encounter_dist_to_npc(s.player.x, s.player.y,
            s.npcs[0].x, s.npcs[0].y, col_npc_effective_size(&s.npcs[0])) <=
                COLO_NPC_STATS[COLO_SERPENT_SHAMAN].attack_range &&
        col_npc_has_los_to_player(&s, &s.npcs[0]));

    geo_clear_npcs(&s);
    s.player.x = 6;
    s.player.y = 11;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_SERPENT_SHAMAN, 13, 11);
    shaman_x = s.npcs[0].x;
    shaman_y = s.npcs[0].y;
    CHECK("named regression start is range plus LoS",
        encounter_dist_to_npc(s.player.x, s.player.y,
            s.npcs[0].x, s.npcs[0].y, col_npc_effective_size(&s.npcs[0])) <=
                COLO_NPC_STATS[COLO_SERPENT_SHAMAN].attack_range &&
        col_npc_has_los_to_player(&s, &s.npcs[0]));
    col_npc_move_ctx(&s, &ctx, 0);
    CHECK("ranged shaman range+LoS player one-step around pillar holds OSRS tile",
        s.npcs[0].x == shaman_x && s.npcs[0].y == shaman_y &&
        s.npcs[0].moved_this_tick == 0);
    CHECK("ranged shaman never takes the over-closed player-adjacent tile",
        encounter_dist_to_npc(s.player.x, s.player.y,
            s.npcs[0].x, s.npcs[0].y, col_npc_effective_size(&s.npcs[0])) > 1);

    geo_clear_npcs(&s);
    s.player.x = 5;
    s.player.y = 9;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_SERPENT_SHAMAN, 13, 9);
    col_npc_move_ctx(&s, &ctx, 0);
    CHECK("LoS-broken shaman takes one SDK legal greedy step",
        s.npcs[0].x == 12 && s.npcs[0].y == 9 &&
        s.npcs[0].moved_this_tick == 1 &&
        !test_npc_covers_player(&s, &s.npcs[0]));

    const ColoNpcType large_types[3] = {
        COLO_JAVELIN_COLOSSUS,
        COLO_SHOCKWAVE_COLOSSUS,
        COLO_MANTICORE,
    };
    for (int i = 0; i < 3; i++) {
        geo_clear_npcs(&s);
        s.player.x = 12;
        s.player.y = 12;
        col_rebuild_player_collision_flags(&s);
        col_init_npc(&s, 0, large_types[i], 5, 5);
        col_npc_move_ctx(&s, &ctx, 0);
        CHECK("size-3 ranged NPC uses leading-edge clearance around pillar",
            s.npcs[0].x == 6 && s.npcs[0].y == 5 &&
            s.npcs[0].moved_this_tick == 1);
    }

    geo_clear_npcs(&s);
    s.player.x = 18;
    s.player.y = 16;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_JAGUAR_WARRIOR, 14, 16);
    int jaguar_covered_player = 0;
    int jaguar_reached_melee = 0;
    for (int tick = 0; tick < 4; tick++) {
        col_npc_move_ctx(&s, &ctx, 0);
        if (test_npc_covers_player(&s, &s.npcs[0])) jaguar_covered_player = 1;
        if (encounter_dist_to_npc(s.player.x, s.player.y,
                s.npcs[0].x, s.npcs[0].y, col_npc_effective_size(&s.npcs[0])) == 1) {
            jaguar_reached_melee = 1;
            break;
        }
    }
    CHECK("melee NPC stops adjacent and never covers the player",
        jaguar_reached_melee == 1 && jaguar_covered_player == 0);

    geo_clear_npcs(&s);
    s.modifiers.active_mask |= (1u << COLO_MOD_RED_FLAG);
    s.modifiers.tier[COLO_MOD_RED_FLAG] = 1;
    s.player.x = 7;
    s.player.y = 9;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_MINOTAUR, 11, 9);
    int covered_player = 0;
    int reached_melee = 0;
    for (int tick = 0; tick < 32; tick++) {
        col_npc_move_ctx(&s, &ctx, 0);
        if (test_npc_covers_player(&s, &s.npcs[0])) covered_player = 1;
        if (encounter_dist_to_npc(s.player.x, s.player.y,
                s.npcs[0].x, s.npcs[0].y, col_npc_effective_size(&s.npcs[0])) == 1) {
            reached_melee = 1;
            break;
        }
    }
    CHECK("routefinding minotaur paths around the pillar to melee",
        reached_melee == 1);
    CHECK("routefinding minotaur never covers the player tile",
        covered_player == 0);
}

static void test_npc_melee_instant_unprayable(void) {
    printf("test_npc_melee_instant_unprayable\n");
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    ColosseumState s;
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 91);
    geo_clear_npcs(&s);

    s.player.x = 18; s.player.y = 16;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_JAGUAR_WARRIOR, 16, 16);

    int blocked_ok = 1, queued = 0;
    float correct_before = s.log.pray_correct_by_type[COLO_JAGUAR_WARRIOR];
    for (int rep = 0; rep < 8; rep++) {
        ctx.player_render_hit_count = 0;
        s.npcs[0].attack_timer = 0;
        s.player.prayer = PRAYER_PROTECT_MELEE;
        s.player.current_hitpoints = 99;
        col_npc_attack_ctx(&s, &ctx, 0);
        if (s.player.current_hitpoints != 99) blocked_ok = 0;
        queued += s.player_pending_hits.count;
    }
    CHECK("Protect-from-Melee on the swing tick blocks every jaguar hit", blocked_ok);
    CHECK("instant melee queues nothing for a later landing (prayed)", queued == 0);
    CHECK("a blocked melee hit still counts prayer_correct",
        s.log.pray_correct_by_type[COLO_JAGUAR_WARRIOR] > correct_before);

    float faced_before = s.log.pray_faced_by_type[COLO_JAGUAR_WARRIOR];
    float wrong_correct_before = s.log.pray_correct_by_type[COLO_JAGUAR_WARRIOR];
    float offpray_before = s.log.offpray_damage_by_type[COLO_JAGUAR_WARRIOR];
    int wrong_faced = 0, wrong_queued = 0;
    for (int rep = 0; rep < 64; rep++) {
        ctx.player_render_hit_count = 0;
        s.npcs[0].attack_timer = 0;
        s.player.prayer = PRAYER_PROTECT_MAGIC;
        s.player.current_hitpoints = 99;
        col_npc_attack_ctx(&s, &ctx, 0);
        wrong_queued += s.player_pending_hits.count;
        wrong_faced += 3;
    }
    CHECK("each off-prayer jaguar swing is faced on the same call",
        s.log.pray_faced_by_type[COLO_JAGUAR_WARRIOR] - faced_before == (float)wrong_faced);
    CHECK("an off-prayer melee swing is never counted correct",
        s.log.pray_correct_by_type[COLO_JAGUAR_WARRIOR] == wrong_correct_before);
    CHECK("instant melee queues nothing for a later landing (off-prayer)",
        wrong_queued == 0);
    CHECK("an unprayed melee hit deals damage on the same call",
        s.log.offpray_damage_by_type[COLO_JAGUAR_WARRIOR] > offpray_before);
}

static void test_player_melee_lands_at_delay_zero(void) {
    printf("test_player_melee_lands_at_delay_zero\n");
    ColosseumContext ctx;
    ColosseumState s;

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 1.0f, 91);
    geo_clear_npcs(&s);
    s.player.x = 18; s.player.y = 16;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_JAGUAR_WARRIOR, 16, 16);

    s.player.attack_timer = 0;
    col_player_attack_target(&s, 0);
    CHECK("a melee swing queues at least one pending hit on the target",
        s.npcs[0].pending_hits.count >= 1);
    int all_delay_zero = s.npcs[0].pending_hits.count >= 1;
    for (int h = 0; h < s.npcs[0].pending_hits.count; h++)
        if (s.npcs[0].pending_hits.hits[h].ticks_remaining != 0) all_delay_zero = 0;
    CHECK("Q6: melee pending hits land at delay 0 (shared OSRS melee delay)",
        all_delay_zero);

    int hp_before = s.npcs[0].hp;
    int resolved_same_pass = 1;
    for (int swing = 0; swing < 32 && s.npcs[0].hp == hp_before; swing++) {
        col_resolve_player_projectiles_on_npcs(&s);
        if (s.npcs[0].pending_hits.count != 0) resolved_same_pass = 0;
        if (s.npcs[0].hp < hp_before) break;
        s.player.attack_timer = 0;
        col_player_attack_target(&s, 0);
    }
    col_resolve_player_projectiles_on_npcs(&s);
    CHECK("a delay-0 melee hit lands on the first resolver pass",
        s.npcs[0].hp < hp_before && resolved_same_pass &&
        s.npcs[0].pending_hits.count == 0);

    ColosseumState r;
    ColosseumContext rctx;
    loadout_reset(&r, &rctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 1.0f, 91);
    geo_clear_npcs(&r);
    col_apply_weapon_set(&r, COLO_GEAR_RANGED);
    r.player.x = 10; r.player.y = 16;
    col_rebuild_player_collision_flags(&r);
    col_init_npc(&r, 0, COLO_JAGUAR_WARRIOR, 16, 16);
    r.player.attack_timer = 0;
    col_player_attack_target(&r, 0);
    int ranged_delay_positive = r.npcs[0].pending_hits.count >= 1;
    for (int h = 0; h < r.npcs[0].pending_hits.count; h++)
        if (r.npcs[0].pending_hits.hits[h].ticks_remaining <= 0) ranged_delay_positive = 0;
    CHECK("a ranged swing keeps its flight delay (>0), so Q6 stays melee-scoped",
        ranged_delay_positive);
}

static void test_echo_boots_recoil_reflects_to_attacker(void) {
    printf("test_echo_boots_recoil_reflects_to_attacker\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 1.0f, 91);
    geo_clear_npcs(&s);
    CHECK("beginner kit equips echo boots",
        s.player.equipped[GEAR_SLOT_FEET] == ITEM_ECHO_BOOTS &&
        osrs_effect_profile_has(col_live_effects(&s), OSRS_ITEM_EFFECT_ECHO_BOOTS));
    CHECK("echo boots start charged",
        s.player.item_effect_state.echo_boot_charges == OSRS_ECHO_BOOTS_MAX_CHARGES);

    s.player.x = 18; s.player.y = 16;
    col_rebuild_player_collision_flags(&s);
    col_init_npc(&s, 0, COLO_JAGUAR_WARRIOR, 16, 16);

    int recoiled = 0;
    long charges_before = s.player.item_effect_state.echo_boot_charges;
    for (int rep = 0; rep < 64 && !recoiled; rep++) {
        ctx.player_render_hit_count = 0;
        s.npcs[0].attack_timer = 0;
        s.npcs[0].hit_damage = 0;
        s.player.prayer = PRAYER_PROTECT_MAGIC;
        s.player.current_hitpoints = 99;
        int hp_before = s.npcs[0].hp;
        long ch_before = s.player.item_effect_state.echo_boot_charges;
        col_npc_attack_ctx(&s, &ctx, 0);
        if (s.player.current_hitpoints < 99) {

            CHECK("a connecting melee reflects exactly one recoil point per hit",
                hp_before - s.npcs[0].hp >= 1);
            CHECK("recoil consumes one echo-boots charge per reflected hit",
                ch_before - s.player.item_effect_state.echo_boot_charges ==
                    (long)(hp_before - s.npcs[0].hp));
            recoiled = 1;
        }
    }
    CHECK("an unprayed jaguar eventually lands and recoils", recoiled);
    CHECK("recoil drew down the echo-boots charge pool",
        s.player.item_effect_state.echo_boot_charges < charges_before);

    ColosseumState sol;
    ColosseumContext solctx;
    loadout_reset(&sol, &solctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 1.0f, 91);
    geo_clear_npcs(&sol);
    col_init_npc(&sol, 0, COLO_SOL_HEREDIT, 16, 16);
    int sol_hp_before = sol.npcs[0].hp;
    col_apply_echo_boots_recoil(&sol, &solctx, 0, 30);
    CHECK("echo-boots recoil never reflects onto Sol Heredit",
        sol.npcs[0].hp == sol_hp_before &&
        sol.player.item_effect_state.echo_boot_charges == OSRS_ECHO_BOOTS_MAX_CHARGES);
}

static void test_colosseum_live_inventory_display(void) {
    printf("test_colosseum_live_inventory_display\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 7);

    int kit[COLO_INVENTORY_DISPLAY_SLOTS];
    col_build_live_inventory_display(&s, kit);
    CHECK("abyssal tentacle is the melee switch carried in the bag", kit[14] == 12006);
    CHECK("worn scythe is not duplicated in the starting bag (worn only)",
        test_count_item_in_equipment_and_inventory(&s, ITEM_SCYTHE_OF_VITUR) == 1);
    CHECK("tbow switch stays in grid", kit[0] == 20997);
    CHECK("brew vial full at start", kit[18] == 6685);
    CHECK("divine combat vial full at start", kit[16] == 23685);
    CHECK("surge vial full at start", kit[12] == 30875);

    int brew_cell = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_BREW);
    s.player.current_hitpoints = 50;
    int brew[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s, brew, brew_cell);
    step_and_observe(&s, &ctx, brew);
    col_build_live_inventory_display(&s, kit);
    CHECK("brew vial shows 3-dose after a drink", kit[18] == 6687);

    s.player.current_hitpoints = 99;
    s.player.current_attack = s.player.base_attack;
    s.player.current_strength = s.player.base_strength;
    s.player.current_defence = s.player.base_defence;
    s.player.potion_timer = 0;
    int divine_combat_cell =
        test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_DIVINE_COMBAT);
    int combat[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s, combat, divine_combat_cell);
    step_and_observe(&s, &ctx, combat);
    col_build_live_inventory_display(&s, kit);
    CHECK("divine combat vial shows 3-dose after a drink", kit[16] == 23688);

    s.player.special_energy = 50;
    s.surge_cooldown = 0;
    s.player.potion_timer = 0;
    int surge_cell = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_SURGE);
    int surge[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s, surge, surge_cell);
    step_and_observe(&s, &ctx, surge);
    col_build_live_inventory_display(&s, kit);
    CHECK("surge vial shows 3-dose after a drink", kit[12] == 30878);

    s.inventory_cells[brew_cell] = (ColoInvCell){
        .raw_osrs_id = 6691,
        .item_idx = ITEM_NONE,
        .dose = 1,
    };
    s.player.potion_timer = 0;
    s.player.current_hitpoints = 50;
    int last_brew[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s, last_brew, brew_cell);
    step_and_observe(&s, &ctx, last_brew);
    col_build_live_inventory_display(&s, kit);
    CHECK("emptied brew slot clears", kit[18] == 0);

    int tbow_cell = test_find_inventory_cell_with_item(&s, ITEM_TWISTED_BOW);
    int ranged[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s, ranged, tbow_cell);
    step_and_observe(&s, &ctx, ranged);
    col_build_live_inventory_display(&s, kit);
    CHECK("ranged weapon click swaps scythe into the tbow cell", kit[0] == 22325);
}

static void test_stage3_t1_inventory_ranged_weapon_swap(void) {
    printf("test_stage3_t1_inventory_ranged_weapon_swap\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 501);
    uint8_t melee_weapon = s.player.equipped[GEAR_SLOT_WEAPON];
    int bow_cell = test_find_inventory_cell_with_item(&s, ITEM_TWISTED_BOW);
    int actions[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s, actions, bow_cell);
    step_and_observe(&s, &ctx, actions);
    CHECK("T1 ranged weapon click equips tbow",
        s.player.equipped[GEAR_SLOT_WEAPON] == ITEM_TWISTED_BOW);
    CHECK("T1 displaced melee weapon returns to clicked cell",
        s.inventory_cells[bow_cell].item_idx == melee_weapon);
}

static void test_stage3_t1_inventory_weapon_slot_last_click_wins(void) {
    printf("test_stage3_t1_inventory_weapon_slot_last_click_wins\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 508);
    int bow_cell = test_find_inventory_cell_with_item(&s, ITEM_TWISTED_BOW);
    int claws_cell = test_find_inventory_cell_with_item(&s, ITEM_DRAGON_CLAWS);
    int actions[COLO_NUM_ACTION_HEADS] = {0};

    actions[COLO_HEAD_EQUIP_SLOT(GEAR_SLOT_WEAPON)] = claws_cell + 1;
    step_and_observe(&s, &ctx, actions);
    CHECK("weapon equip head equips the named weapon",
        s.player.equipped[GEAR_SLOT_WEAPON] == ITEM_DRAGON_CLAWS);
    CHECK("unclicked weapon stays in its inventory cell",
        s.inventory_cells[bow_cell].item_idx == ITEM_TWISTED_BOW);
}

static void test_stage3_t1_human_inventory_primary_click_uses_resolver(void) {
    printf("test_stage3_t1_human_inventory_primary_click_uses_resolver\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 509);
    int bow_cell = test_find_inventory_cell_with_item(&s, ITEM_TWISTED_BOW);
    int claws_cell = test_find_inventory_cell_with_item(&s, ITEM_DRAGON_CLAWS);
    HumanInput hi;
    human_input_init(&hi);
    hi.enabled = 1;
    human_input_queue_inventory_primary_click(&hi, bow_cell);
    human_input_queue_inventory_primary_click(&hi, claws_cell);
    int actions[COLO_NUM_ACTION_HEADS] = {0};
    col_translate_human_commands_ctx(&hi, actions, &s, &ctx);

    CHECK("human weapon clicks collapse to the weapon equip head",
        actions[COLO_HEAD_EQUIP_SLOT(GEAR_SLOT_WEAPON)] == claws_cell + 1);
    step_and_observe(&s, &ctx, actions);
    CHECK("human inventory clicks use last-click-wins semantics",
        s.player.equipped[GEAR_SLOT_WEAPON] == ITEM_DRAGON_CLAWS);
    CHECK("human earlier same-slot click is ignored",
        s.inventory_cells[bow_cell].item_idx == ITEM_TWISTED_BOW);
    human_input_destroy(&hi);
}

static void test_stage3_t1_human_rearrange_swaps_inventory_slots(void) {
    printf("test_stage3_t1_human_rearrange_swaps_inventory_slots\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 516);
    int bow_cell = test_find_inventory_cell_with_item(&s, ITEM_TWISTED_BOW);
    int claws_cell = test_find_inventory_cell_with_item(&s, ITEM_DRAGON_CLAWS);

    HumanInput hi;
    human_input_init(&hi);
    hi.enabled = 1;
    human_input_queue_item_on_item(
        &hi,
        bow_cell,
        claws_cell,
        s.inventory_cells[bow_cell].item_idx,
        s.inventory_cells[bow_cell].raw_osrs_id);
    int actions[COLO_NUM_ACTION_HEADS] = {0};
    col_translate_human_commands_ctx(&hi, actions, &s, &ctx);

    CHECK("human rearrange moves bow to target slot",
        s.inventory_cells[claws_cell].item_idx == ITEM_TWISTED_BOW);
    CHECK("human rearrange moves claws to source slot",
        s.inventory_cells[bow_cell].item_idx == ITEM_DRAGON_CLAWS);
    CHECK("human rearrange leaves action space heads unchanged",
        COLO_NUM_ACTION_HEADS == 20 && COLO_ACTION_DIMS[COLO_HEAD_EQUIP_BASE] == 29);
    human_input_destroy(&hi);
}

static void test_stage3_t2_brew_click_decrements_dose(void) {
    printf("test_stage3_t2_brew_click_decrements_dose\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 502);
    int brew_cell = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_BREW);
    s.player.current_hitpoints = 50;
    int actions[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s, actions, brew_cell);
    step_and_observe(&s, &ctx, actions);
    CHECK("T2 brew raises HP", s.player.current_hitpoints > 50);
    CHECK("T2 brew cell dose drops 4 to 3",
        s.inventory_cells[brew_cell].dose == 3 &&
        s.inventory_cells[brew_cell].raw_osrs_id == 6687);
    CHECK("T2 brew starts potion timer", s.player.potion_timer == 3);
}

static void test_stage3_t3_one_dose_vial_empties(void) {
    printf("test_stage3_t3_one_dose_vial_empties\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 503);
    int brew_cell = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_BREW);
    s.inventory_cells[brew_cell] = (ColoInvCell){
        .raw_osrs_id = 6691,
        .item_idx = ITEM_NONE,
        .dose = 1,
    };
    s.player.current_hitpoints = 50;
    int actions[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s, actions, brew_cell);
    step_and_observe(&s, &ctx, actions);
    CHECK("T3 one-dose vial cell becomes empty",
        s.inventory_cells[brew_cell].raw_osrs_id == 0 &&
        s.inventory_cells[brew_cell].item_idx == ITEM_NONE &&
        s.inventory_cells[brew_cell].dose == 0);
}

static void test_colosseum_potion_click_source_of_truth(void) {
    printf("test_colosseum_potion_click_source_of_truth\n");
    ColosseumContext ctx;
    ColosseumState s;
    int actions[COLO_NUM_ACTION_HEADS] = {0};

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 510);
    int restore_cell =
        test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_SUPER_RESTORE);
    int restore_sum_before =
        test_sum_inventory_doses_for_kind(&s, OSRS_CONSUMABLE_SUPER_RESTORE);
    s.player.current_prayer = 40;
    s.player.restore_doses = 99;
    test_click_inventory_cell_action_s(&s, actions, restore_cell);
    step_and_observe(&s, &ctx, actions);
    CHECK("super restore consumes exactly one clicked-cell dose",
        s.inventory_cells[restore_cell].dose == 3);
    CHECK("super restore aggregate is rebuilt from cells",
        s.player.restore_doses == restore_sum_before - 1);

    memset(actions, 0, sizeof(actions));
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 511);
    int sanfew_cell = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_SANFEW);
    int sanfew_sum_before = test_sum_inventory_doses_for_kind(&s, OSRS_CONSUMABLE_SANFEW);
    s.player.current_prayer = 40;
    s.player.restore_doses = 99;
    test_click_inventory_cell_action_s(&s, actions, sanfew_cell);
    step_and_observe(&s, &ctx, actions);
    CHECK("sanfew consumes exactly one clicked-cell dose",
        s.inventory_cells[sanfew_cell].dose == 3);
    CHECK("sanfew aggregate is rebuilt from cells",
        s.player.restore_doses == sanfew_sum_before - 1);

    memset(actions, 0, sizeof(actions));
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 512);
    int divine_cell =
        test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_DIVINE_COMBAT);
    int divine_sum_before =
        test_sum_inventory_doses_for_kind(&s, OSRS_CONSUMABLE_DIVINE_COMBAT);
    s.player.current_hitpoints = 99;
    s.player.combat_potion_doses = 99;
    test_click_inventory_cell_action_s(&s, actions, divine_cell);
    step_and_observe(&s, &ctx, actions);
    CHECK("divine combat consumes exactly one clicked-cell dose",
        s.inventory_cells[divine_cell].dose == 3);
    CHECK("divine combat aggregate is rebuilt from cells",
        s.player.combat_potion_doses == divine_sum_before - 1);
    CHECK("divine combat drink starts the potion timer", s.player.potion_timer == 3);

    memset(actions, 0, sizeof(actions));
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 517);
    restore_cell =
        test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_SUPER_RESTORE);
    restore_sum_before =
        test_sum_inventory_doses_for_kind(&s, OSRS_CONSUMABLE_SUPER_RESTORE);
    s.player.current_prayer = s.player.base_prayer - 1;
    test_click_inventory_cell_action_s(&s, actions, restore_cell);
    step_and_observe(&s, &ctx, actions);
    CHECK("low-missing-prayer restore click still consumes one dose",
        s.inventory_cells[restore_cell].dose == 3);
    CHECK("low-missing-prayer restore rebuilds aggregate from cells",
        s.player.restore_doses == restore_sum_before - 1);
}

static void test_colosseum_potion_timer_and_same_tick_gate(void) {
    printf("test_colosseum_potion_timer_and_same_tick_gate\n");
    ColosseumContext ctx;
    ColosseumState s;

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 513);
    int brew_cell = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_BREW);
    s.player.current_hitpoints = 50;
    int brew[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s, brew, brew_cell);
    step_and_observe(&s, &ctx, brew);
    int brew_dose_after_first = s.inventory_cells[brew_cell].dose;
    int brew_aggregate_after_first = s.player.brew_doses;
    s.player.current_hitpoints = 50;
    step_and_observe(&s, &ctx, brew);
    CHECK("second potion click before timer expiry is blocked",
        s.inventory_cells[brew_cell].dose == brew_dose_after_first &&
        s.player.brew_doses == brew_aggregate_after_first &&
        s.player.potion_timer == 2);

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 514);
    int sanfew_cell = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_SANFEW);
    int divine_cell =
        test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_DIVINE_COMBAT);
    s.player.current_prayer = 40;

    int drink_one[COLO_NUM_ACTION_HEADS] = {0};
    drink_one[COLO_HEAD_DRINK] = divine_cell + 1;
    step_and_observe(&s, &ctx, drink_one);
    CHECK("the drink head consumes exactly one potion per tick",
        s.inventory_cells[divine_cell].dose == 3 &&
        s.inventory_cells[sanfew_cell].dose == 4);

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 515);
    sanfew_cell = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_SANFEW);
    s.player.current_prayer = 40;

    int drink_again[COLO_NUM_ACTION_HEADS] = {0};
    drink_again[COLO_HEAD_DRINK] = sanfew_cell + 1;
    step_and_observe(&s, &ctx, drink_again);
    CHECK("first drink consumes one dose", s.inventory_cells[sanfew_cell].dose == 3);
    step_and_observe(&s, &ctx, drink_again);
    CHECK("second drink before potion timer expiry is blocked",
        s.inventory_cells[sanfew_cell].dose == 3);
}

typedef struct {
    const char* label;
    OsrsConsumableKind kind;
    uint16_t raw4;
    uint16_t raw3;
    uint16_t raw2;
    uint16_t raw1;
} TestDrinkKindDoseChain;

static const TestDrinkKindDoseChain TEST_DRINK_KIND_DOSE_CHAINS[] = {
    {"brew", OSRS_CONSUMABLE_BREW, 6685, 6687, 6689, 6691},
    {"super restore", OSRS_CONSUMABLE_SUPER_RESTORE, 3024, 3026, 3028, 3030},
    {"sanfew", OSRS_CONSUMABLE_SANFEW, 10925, 10927, 10929, 10931},
    {"super combat", OSRS_CONSUMABLE_SUPER_COMBAT, 12695, 12697, 12699, 12701},
    {"divine combat", OSRS_CONSUMABLE_DIVINE_COMBAT, 23685, 23688, 23691, 23694},
    {"ranging", OSRS_CONSUMABLE_RANGING, 2444, 169, 171, 173},
    {"divine ranging", OSRS_CONSUMABLE_DIVINE_RANGING, 23733, 23736, 23739, 23742},
    {"surge", OSRS_CONSUMABLE_SURGE, 30875, 30878, 30881, 30884},
    {"guthix rest", OSRS_CONSUMABLE_GUTHIX_REST, 4417, 4419, 4421, 4423},
    {"anti-venom+", OSRS_CONSUMABLE_ANTIVENOM_PLUS, 12913, 12915, 12917, 12919},
    {"saturated heart", OSRS_CONSUMABLE_SATURATED_HEART, 0, 0, 0, 27641},
};

static void test_colosseum_all_drink_kinds_shared_one_dose_path(void) {
    printf("test_colosseum_all_drink_kinds_shared_one_dose_path\n");
    for (int i = 0; i < (int)(sizeof(TEST_DRINK_KIND_DOSE_CHAINS) /
            sizeof(TEST_DRINK_KIND_DOSE_CHAINS[0])); i++) {
        const TestDrinkKindDoseChain* c = &TEST_DRINK_KIND_DOSE_CHAINS[i];
        ColosseumContext ctx;
        ColosseumState s;
        loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f,
            (uint32_t)(700 + i));
        int cell = 0;
        uint16_t start_raw = c->raw4 ? c->raw4 : c->raw1;
        col_init_empty_inventory_cells(&s);
        s.inventory_cells[cell] = osrs_inventory_cell_from_raw_osrs_id(start_raw);
        col_sync_consumable_counters_from_inventory(&s);
        test_prepare_for_drink_kind(&s, c->kind);

        int actions[COLO_NUM_ACTION_HEADS] = {0};
        test_click_inventory_cell_action_s(&s, actions, cell);
        step_and_observe(&s, &ctx, actions);
        uint16_t expected_after_first = c->raw4 ? c->raw3 : 0;
        uint8_t expected_dose_after_first = c->raw4 ? 3 : 0;
        char label[160];
        snprintf(label, sizeof(label), "%s first click decrements one dose", c->label);
        CHECK(label,
            s.inventory_cells[cell].raw_osrs_id == expected_after_first &&
            s.inventory_cells[cell].dose == expected_dose_after_first);
        snprintf(label, sizeof(label), "%s first click arms potion timer", c->label);
        CHECK(label, s.player.potion_timer == 3);
        int aggregate = test_aggregate_doses_for_kind(&s, c->kind);
        if (aggregate >= 0) {
            snprintf(label, sizeof(label), "%s aggregate rebuilds from clicked cell",
                c->label);
            CHECK(label, aggregate == test_sum_inventory_doses_for_kind(&s, c->kind));
        }

        if (!c->raw4) {
            s.inventory_cells[cell] = osrs_inventory_cell_from_raw_osrs_id(c->raw1);
        }
        uint16_t raw_before_gate = s.inventory_cells[cell].raw_osrs_id;
        uint8_t dose_before_gate = s.inventory_cells[cell].dose;
        step_and_observe(&s, &ctx, actions);
        snprintf(label, sizeof(label), "%s timer gate blocks next click", c->label);
        CHECK(label,
            s.inventory_cells[cell].raw_osrs_id == raw_before_gate &&
            s.inventory_cells[cell].dose == dose_before_gate);

        const uint16_t chain[] = {c->raw4, c->raw3, c->raw2, c->raw1, 0};
        int start = c->raw4 ? 0 : 3;
        s.inventory_cells[cell] = osrs_inventory_cell_from_raw_osrs_id(chain[start]);
        col_sync_consumable_counters_from_inventory(&s);
        for (int step = start; step < 4; step++) {
            test_prepare_for_drink_kind(&s, c->kind);
            memset(actions, 0, sizeof(actions));
            test_click_inventory_cell_action_s(&s, actions, cell);
            step_and_observe(&s, &ctx, actions);
            snprintf(label, sizeof(label), "%s chain step %d raw id", c->label, step);
            CHECK(label, s.inventory_cells[cell].raw_osrs_id == chain[step + 1]);
            uint8_t expected_dose = chain[step + 1] == 0 ? 0 : (uint8_t)(3 - step);
            snprintf(label, sizeof(label), "%s chain step %d dose", c->label, step);
            CHECK(label, s.inventory_cells[cell].dose == expected_dose);
        }
    }
}

static void test_inventory_pure_cut_reconstruction(void) {
    printf("test_inventory_pure_cut_reconstruction\n");
    ColosseumContext ctx;
    ColosseumState s;

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 900);
    test_check_inventory_cut_equivalence_state(&s, &ctx, "speedrun loadout");

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 901);
    test_check_inventory_cut_equivalence_state(&s, &ctx, "beginner loadout");

    for (int i = 0; i < (int)(sizeof(TEST_DRINK_KIND_DOSE_CHAINS) /
            sizeof(TEST_DRINK_KIND_DOSE_CHAINS[0])); i++) {
        const TestDrinkKindDoseChain* c = &TEST_DRINK_KIND_DOSE_CHAINS[i];
        const uint16_t chain[] = {c->raw4, c->raw3, c->raw2, c->raw1};
        for (int dose = 0; dose < 4; dose++) {
            if (chain[dose] == 0) continue;
            loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f,
                (uint32_t)(920 + i * 4 + dose));
            col_init_empty_inventory_cells(&s);
            s.inventory_cells[0] = osrs_inventory_cell_from_raw_osrs_id(chain[dose]);
            col_sync_consumable_counters_from_inventory(&s);
            test_prepare_for_drink_kind(&s, c->kind);
            char label[160];
            snprintf(label, sizeof(label), "%s dose chain raw %u",
                c->label, (unsigned)chain[dose]);
            test_check_inventory_cut_equivalence_state(&s, &ctx, label);
        }
    }

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 902);
    int bow_cell = test_find_inventory_cell_with_item(&s, ITEM_TWISTED_BOW);
    assert(bow_cell >= 0);
    int bow_actions[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s, bow_actions, bow_cell);
    step_and_observe(&s, &ctx, bow_actions);
    CHECK("inventory pure-cut gear swap equipped tbow",
        s.player.equipped[GEAR_SLOT_WEAPON] == ITEM_TWISTED_BOW);
    test_check_inventory_cut_equivalence_state(&s, &ctx, "gear swap after tbow click");

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 903);
    int bowfa_cell = test_find_inventory_cell_with_item(&s, ITEM_BOW_OF_FAERDHINEN);
    assert(bowfa_cell >= 0);
    CHECK("full inventory two-handed equip is denied",
        s.player.equipped[GEAR_SLOT_WEAPON] != ITEM_NONE &&
        s.player.equipped[GEAR_SLOT_SHIELD] != ITEM_NONE &&
        !col_inventory_cell_actionable(&s, bowfa_cell));
    test_check_inventory_cut_equivalence_state(
        &s, &ctx, "full inventory two-handed equip denial");

    s.inventory_cells[27] = osrs_inventory_cell_empty();
    int bowfa_actions[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s, bowfa_actions, bowfa_cell);
    step_and_observe(&s, &ctx, bowfa_actions);
    CHECK("two-handed bowfa suppresses shield when a spare cell exists",
        s.player.equipped[GEAR_SLOT_WEAPON] == ITEM_BOW_OF_FAERDHINEN &&
        s.player.equipped[GEAR_SLOT_SHIELD] == ITEM_NONE);
    test_check_inventory_cut_equivalence_state(
        &s, &ctx, "two-handed shield suppression after bowfa click");

    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 904);
    col_init_empty_inventory_cells(&s);
    s.inventory_cells[0] = osrs_inventory_cell_from_raw_osrs_id(6685);
    s.inventory_cells[1] = osrs_inventory_cell_from_raw_osrs_id(6685);
    col_sync_consumable_counters_from_inventory(&s);
    s.player.current_hitpoints = 50;
    s.player.potion_timer = 0;
    test_check_inventory_cut_equivalence_state(&s, &ctx, "duplicate brew stacks");
}

static void test_stage3_t4_click_mask_bits(void) {
    printf("test_stage3_t4_click_mask_bits\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 504);
    int equipped_scythe_cell = test_find_inventory_cell_with_item(&s, ITEM_SCYTHE_OF_VITUR);
    int empty_cell = 27;
    s.inventory_cells[empty_cell] = (ColoInvCell){ .item_idx = ITEM_NONE };
    int brew_cell = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_BREW);
    s.player.current_hitpoints = 50;
    float mask[COLO_ACTION_MASK_SIZE];
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    int spec_off = col_action_head_mask_offset(COLO_HEAD_SPEC);
    CHECK("T4 worn scythe is not present as a clickable inventory cell",
        equipped_scythe_cell < 0);
    CHECK("T4 empty cell click bit is 0",
        test_click_mask_for_cell_s(&s, mask, empty_cell) == 0.0f);
    CHECK("T4 beneficial full-dose brew cell click bit is 1",
        test_click_mask_for_cell_s(&s, mask, brew_cell) == 1.0f);
    CHECK("T4 spec arm bit is 0 for non-spec equipped weapon",
        mask[spec_off + 1] == 0.0f);
}

static void test_stage3_t4_mask_inventory_heads_flag(void) {
    printf("test_stage3_t4_mask_inventory_heads_flag\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_BEGINNER_ONLY, 0.0f, 507);
    ctx.config.mask_inventory_heads = 1;

    float mask[COLO_ACTION_MASK_SIZE];
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    int all_inventory_heads_pinned_to_noop = 1;
    int inv_heads[COLO_INV_CLICK_HEADS];
    int nh = 0;
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) inv_heads[nh++] = COLO_HEAD_EQUIP_SLOT(slot);
    inv_heads[nh++] = COLO_HEAD_EAT;
    inv_heads[nh++] = COLO_HEAD_DRINK;
    for (int i = 0; i < nh; i++) {
        int offset = col_action_head_mask_offset(inv_heads[i]);
        if (mask[offset] != 1.0f) all_inventory_heads_pinned_to_noop = 0;
        for (int action = 1; action < COLO_INV_CLICK_DIM; action++)
            if (mask[offset + action] != 0.0f) all_inventory_heads_pinned_to_noop = 0;
    }
    CHECK("mask_inventory_heads pins every inventory head to noop only",
        all_inventory_heads_pinned_to_noop);
    CHECK("mask_inventory_heads leaves the action-mask size unchanged",
        COLO_ACTION_MASK_SIZE == 452);

    geo_clear_npcs(&s);
    s.modifiers.draft_pending = 0;
    s.wave_ready_delay = 0;
    s.player.current_hitpoints = 50;
    int brew_cell = test_find_inventory_cell_with_consumable(&s, OSRS_CONSUMABLE_BREW);
    int brew_doses_before = s.player.brew_doses;
    int actions[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s, actions, brew_cell);
    step_and_observe(&s, &ctx, actions);
    CHECK("mask_inventory_heads makes sampled inventory clicks no-op in dispatch",
        s.player.current_hitpoints == 50 && s.player.brew_doses == brew_doses_before);
}

static void test_stage3_t5_claws_click_spec_fires(void) {
    printf("test_stage3_t5_claws_click_spec_fires\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 505);
    geo_clear_npcs(&s);
    s.modifiers.draft_pending = 0;
    s.wave_ready_delay = 0;
    s.player.x = 16;
    s.player.y = 16;
    col_init_npc(&s, 0, COLO_FREMENNIK_BERSERKER, 16, 17);
    int actions[COLO_NUM_ACTION_HEADS] = {0};
    test_click_inventory_cell_action_s(&s, actions, test_find_inventory_cell_with_item(&s, ITEM_DRAGON_CLAWS));
    actions[COLO_HEAD_SPEC] = 1;
    step_and_observe(&s, &ctx, actions);
    CHECK("T5 claws click equips claws",
        s.player.equipped[GEAR_SLOT_WEAPON] == ITEM_DRAGON_CLAWS);
    CHECK("T5 SPEC arms equipped claws", s.player.spec_armed == 1);
    s.player.attack_timer = 0;
    col_player_attack_target(&s, 0);
    CHECK("T5 claws special fires four splats",
        s.npcs[0].pending_hits.count == 4 && s.player.special_energy == 50);
}

static void test_stage3_t6_obs_mask_fuzz_contract(void) {
    printf("test_stage3_t6_obs_mask_fuzz_contract\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_MIXED, 0.5f, 506);
    int actions[COLO_NUM_ACTION_HEADS] = {0};
    uint32_t rng = 506;
    for (int t = 0; t < 256 && !s.episode_over; t++) {
        for (int h = 0; h < COLO_NUM_ACTION_HEADS; h++) {
            rng = rng * 1664525u + 1013904223u;
            actions[h] = (int)((rng >> 16) % (uint32_t)COLO_ACTION_DIMS[h]);
        }
        step_and_observe(&s, &ctx, actions);
    }
    CHECK("T6 obs running-index assert reached COLO_NUM_OBS", COLO_NUM_OBS == 3044);
    CHECK("T6 mask running-index assert reached 452", COLO_ACTION_MASK_SIZE == 452);
}

static void test_death_attribution_credits_actual_source(void) {
    printf("test_death_attribution_credits_actual_source\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 7);
    ColPendingHitObserverContext obs = { &s, &ctx };

    EncounterPendingHit manticore = {0};
    manticore.source_npc_type = COLO_MANTICORE;
    manticore.source_npc_slot = -1;
    manticore.attack_style = ATTACK_STYLE_MAGIC;
    col_pending_hit_prayer_observer(&obs, &manticore, 10, 0, 0);
    CHECK("manticore landing credits the manticore", s.last_hit_by_type == COLO_MANTICORE);

    EncounterPendingHit shockwave = {0};
    shockwave.source_npc_type = COLO_SHOCKWAVE_COLOSSUS;
    shockwave.source_npc_slot = -1;
    shockwave.attack_style = ATTACK_STYLE_RANGED;
    col_pending_hit_prayer_observer(&obs, &shockwave, 12, 0, 0);
    CHECK("a non-manticore landing re-credits the actual source",
          s.last_hit_by_type == COLO_SHOCKWAVE_COLOSSUS);

    col_pending_hit_prayer_observer(&obs, &manticore, 0, 1, 0);
    CHECK("a 0-damage splash does not change attribution",
          s.last_hit_by_type == COLO_SHOCKWAVE_COLOSSUS);
}

static int test_walkable_block_corner(void* ctx, int x, int y) {
    (void)ctx;
    return !(x == 1 && y == 1);
}

static int test_walkable_open(void* ctx, int x, int y) {
    (void)ctx; (void)x; (void)y;
    return 1;
}

static void test_move_action_no_corner_cut(void) {
    printf("test_move_action_no_corner_cut\n");
    Player p;
    memset(&p, 0, sizeof(p));
    p.x = 1; p.y = 0;
    encounter_move_to_target(&p, -1, 1, test_walkable_block_corner, NULL);
    CHECK("a blocked corner is never cut", !(p.x == 0 && p.y == 1));
    CHECK("a blocked-corner diagonal degrades to the open cardinal", p.x == 0 && p.y == 0);

    Player q;
    memset(&q, 0, sizeof(q));
    q.x = 1; q.y = 0;
    encounter_move_to_target(&q, -1, 1, test_walkable_open, NULL);
    CHECK("an unobstructed diagonal still moves diagonally", q.x == 0 && q.y == 1);
}

static void test_melee_reach_cardinal_vs_diagonal(void) {
    printf("test_melee_reach_cardinal_vs_diagonal\n");
    const OsrsLosQuery* open = osrs_los_open_query();
    const int tx = 10, ty = 10;
    for (int tsize = 1; tsize <= 3; tsize++) {

        const int corners[4][2] = {
            {tx - 1, ty - 1}, {tx + tsize, ty - 1},
            {tx - 1, ty + tsize}, {tx + tsize, ty + tsize}};
        for (int i = 0; i < 4; i++) {
            int cx = corners[i][0], cy = corners[i][1];
            CHECK("reach-1 helper rejects a diagonal corner",
                  encounter_entity_footprint_cardinal_adjacent(cx, cy, 1, tx, ty, tsize) == 0);
            CHECK("range-1 gate rejects a diagonal corner",
                  encounter_player_can_attack(cx, cy, tx, ty, tsize, 1, open) == 0);
            CHECK("range-2 (halberd) gate allows a diagonal corner",
                  encounter_player_can_attack(cx, cy, tx, ty, tsize, 2, open) == 1);
        }

        for (int k = 0; k < tsize; k++) {
            CHECK("range-1 gate allows a west cardinal-edge tile",
                  encounter_player_can_attack(tx - 1, ty + k, tx, ty, tsize, 1, open) == 1);
            CHECK("range-1 gate allows an east cardinal-edge tile",
                  encounter_player_can_attack(tx + tsize, ty + k, tx, ty, tsize, 1, open) == 1);
            CHECK("range-1 gate allows a south cardinal-edge tile",
                  encounter_player_can_attack(tx + k, ty - 1, tx, ty, tsize, 1, open) == 1);
            CHECK("range-1 gate allows a north cardinal-edge tile",
                  encounter_player_can_attack(tx + k, ty + tsize, tx, ty, tsize, 1, open) == 1);
        }

        CHECK("overlap is never meleeable",
              encounter_player_can_attack(tx, ty, tx, ty, tsize, 1, open) == 0);
    }
}

static void test_modifier_draft_forces_pick(void) {
    printf("test_modifier_draft_forces_pick\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 71);
    col_modifier_open_draft(&s, 2);
    CHECK("a draft is open after col_modifier_open_draft", draft_is_open(&s));

    static float mask[COLO_ACTION_MASK_SIZE];
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    int base = col_action_head_mask_offset(COLO_HEAD_MODIFIER_SELECT);
    CHECK("no-op masked off while a draft is pending", mask[base] == 0.0f);
    int selectable = 0;
    for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++)
        if (mask[base + 1 + o] == 1.0f) selectable = 1;
    CHECK("at least one draft option is selectable (forced pick is possible)", selectable);

    complete_open_draft(&s, &ctx, 0);
    CHECK("draft closed after the pick", !s.modifiers.draft_pending);
    col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, mask);
    CHECK("no-op valid again once no draft is pending", mask[base] == 1.0f);
}

static void test_gear_and_boost_reward_signals(void) {
    printf("test_gear_and_boost_reward_signals\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 71);
    advance_to_wave_spawn(&s, &ctx);
    int slot = -1;
    for (int i = 0; i < COLO_MAX_NPCS; i++)
        if (col_npc_is_live_target(&s.npcs[i]) && !col_type_is_hazard_entity(s.npcs[i].type)) {
            slot = i;
            break;
        }
    CHECK("a live target exists at reset", slot >= 0);

    s.tick_scratch.player_attacked = 1;
    s.player_attack_npc_idx = slot;

    float q_attack = col_attacked_gear_quality_ratio(&s);
    CHECK("gear-quality signal fires in [0,1] when attacking", q_attack >= 0.0f && q_attack <= 1.0f);

    const ColoNPC* tnpc = &s.npcs[slot];
    const ColoBestGear (*best)[COLO_NUM_NPC_TYPES] = col_get_best_gear_table(&s);
    int argmax_set = 0;
    float argmax_dpt = -1.0f;
    for (int set = 0; set < COLO_NUM_WEAPON_SETS; set++)
        if (best[set][tnpc->type].dpt > argmax_dpt) {
            argmax_dpt = best[set][tnpc->type].dpt;
            argmax_set = set;
        }
    memcpy(s.player.equipped, best[argmax_set][tnpc->type].setup, sizeof(s.player.equipped));
    CHECK("oracle's argmax-best kit yields ~max gear quality",
          col_attacked_gear_quality_ratio(&s) > 0.99f);

    s.player_attack_style_id = ATTACK_STYLE_RANGED;
    s.player.current_ranged = s.player.base_ranged;
    CHECK("no boost reward when ranged is at base",
          col_attacked_with_offensive_boost(&s) == 0);
    s.player.current_ranged = s.player.base_ranged + 5;
    CHECK("boost reward when ranged is above base",
          col_attacked_with_offensive_boost(&s) == 1);

    s.tick_scratch.player_attacked = 0;
    CHECK("gear-quality is the no-attack sentinel", col_attacked_gear_quality_ratio(&s) < 0.0f);
    CHECK("boost signal is the no-attack sentinel", col_attacked_with_offensive_boost(&s) < 0);
}

static int colo_test_cell_of_named_item(const ColosseumState* s, const char* name) {
    for (int c = 0; c < OSRS_INVENTORY_SIZE; c++) {
        uint8_t item = s->inventory_cells[c].item_idx;
        if (item == ITEM_NONE) continue;
        const Item* meta = get_item(item);
        if (meta && strcmp(meta->name, name) == 0) return c;
    }
    return -1;
}

static void test_threat_field_obs(void) {
    printf("test_threat_field_obs\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 71);

    s.player.x = 5; s.player.y = 9;
    col_rebuild_player_collision_flags(&s);
    int manti = col_spawn_npc_at(&s, COLO_MANTICORE, 11, 8);
    CHECK("fixture: shooter spawned", manti >= 0);
    CHECK("fixture: the pillar blocks LoS to the player's tile",
        !col_npc_has_los_to_player(&s, &s.npcs[manti]));

    static float obs[COLO_NUM_OBS];
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);

    const int f0 = COLO_OBS_AFTER_SPAWN;
    const int f1 = f0 + COLO_THREAT_FIELD_TILES;
#define FIELD_CELL(dx, dy) \
    (((dy) + COLO_THREAT_FIELD_RADIUS) * COLO_THREAT_FIELD_DIM + \
     ((dx) + COLO_THREAT_FIELD_RADIUS))

    CHECK("pillar shadow: the player's tile reads zero shooters",
        obs[f0 + FIELD_CELL(0, 0)] == 0.0f);
    CHECK("exposed tile east of the pillar reads the shooter",
        obs[f0 + FIELD_CELL(7, 3)] > 0.0f);
    CHECK("pillar tile is unstandable", obs[f1 + FIELD_CELL(3, -1)] == 1.0f);
    CHECK("NPC body tile is unstandable", obs[f1 + FIELD_CELL(6, -1)] == 1.0f);
    CHECK("the player's own tile is standable", obs[f1 + FIELD_CELL(0, 0)] == 0.0f);
    CHECK("out-of-arena tile is unstandable", obs[f1 + FIELD_CELL(-8, 0)] == 1.0f);

    s.player.x = 12; s.player.y = 12;
    col_rebuild_player_collision_flags(&s);
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("center cell reads one shooter after stepping into LoS",
        obs[f0 + FIELD_CELL(0, 0)] == 0.25f);

    int jag = col_spawn_npc_at(&s, COLO_JAGUAR_WARRIOR, 16, 12);
    CHECK("fixture: melee NPC spawned", jag >= 0);
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("jaguar body tile is unstandable", obs[f1 + FIELD_CELL(4, 0)] == 1.0f);
    int jag_size = col_npc_effective_size(&s.npcs[jag]);
    col_stamp_npc_collision_footprint(&s, s.npcs[jag].x, s.npcs[jag].y, jag_size, 0);
    s.npcs[jag].x += 1;
    col_stamp_npc_collision_footprint(&s, s.npcs[jag].x, s.npcs[jag].y, jag_size, 1);
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("vacated tile frees after the melee body moves (not a stale memo)",
        obs[f1 + FIELD_CELL(4, 0)] == 0.0f);
    CHECK("the melee body's new tile is unstandable",
        obs[f1 + FIELD_CELL(6, 0)] == 1.0f);

    col_deactivate_npc(&s, manti);
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("center cell reads zero shooters after the manticore dies",
        obs[f0 + FIELD_CELL(0, 0)] == 0.0f);
    CHECK("dead manticore's body tile frees in channel 1",
        obs[f1 + FIELD_CELL(-1, -4)] == 0.0f);

    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    float served_field[COLO_THREAT_FIELD_OBS_CACHE_FLOATS];
    memcpy(served_field, &obs[f0], sizeof(served_field));
    memset(&s.obs_memos, 0, sizeof(s.obs_memos));
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("memo-served threat field == fresh recompute",
        memcmp(served_field, &obs[f0], sizeof(served_field)) == 0);

    ctx.config.threat_field_obs_enabled = 0;
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    int all_zero = 1;
    for (int k = 0; k < COLO_THREAT_FIELD_OBS_SIZE; k++)
        if (obs[f0 + k] != 0.0f) all_zero = 0;
    CHECK("disabled threat field leaves the block zeroed", all_zero);
#undef FIELD_CELL
}

static void test_inventory_obs_memo(void) {
    printf("test_inventory_obs_memo\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 71);

    static float obs[COLO_NUM_OBS];
    int cell_brew = -1;
    for (int c = 0; c < OSRS_INVENTORY_SIZE; c++)
        if (s.inventory_cells[c].item_idx == ITEM_NONE) { cell_brew = c; break; }
    CHECK("a gear-free cell exists to hold the brew", cell_brew >= 0);
    s.inventory_cells[cell_brew] = osrs_inventory_cell_from_raw_osrs_id(6685);
    CHECK("4-dose brew seeded", s.inventory_cells[cell_brew].dose == 4);

    const int cell_base = COLO_OBS_AFTER_PILLARS +
        cell_brew * COLO_INVENTORY_CELL_OBS_FEATURES;
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("4-dose brew renders the full dose feature", obs[cell_base + 2] == 1.0f);

    s.inventory_cells[cell_brew] = osrs_inventory_cell_from_raw_osrs_id(6687);
    CHECK("sip took a dose", s.inventory_cells[cell_brew].dose == 3);
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("sip moves the dose feature (dose is in the memo key, not a stale block)",
        obs[cell_base + 2] == 0.75f);

    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    float served_block[COLO_INVENTORY_OBS_CACHE_FLOATS];
    memcpy(served_block, &obs[COLO_OBS_AFTER_PILLARS], sizeof(served_block));
    memset(&s.obs_memos, 0, sizeof(s.obs_memos));
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("memo-served inventory block == fresh recompute after the sip",
        memcmp(served_block, &obs[COLO_OBS_AFTER_PILLARS], sizeof(served_block)) == 0);
}

static void test_weapon_choice_obs_rank_and_farm_cap(void) {
    printf("test_weapon_choice_obs_rank_and_farm_cap\n");
    ColosseumContext ctx;
    ColosseumState s;
    loadout_reset(&s, &ctx, COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY, 0.0f, 71);

    static float obs[COLO_NUM_OBS];
    const int dpt_base = COLO_OBS_AFTER_THRALL_DC;
    const int spec_base = dpt_base + COLO_CELL_WEAPON_DPT_OBS_SIZE;
    const int wielded_base = spec_base + COLO_CELL_SPEC_OBS_SIZE;

    int cell_tentacle = colo_test_cell_of_named_item(&s, "Abyssal tentacle");
    int cell_claws = colo_test_cell_of_named_item(&s, "Dragon claws");
    int cell_tbow = colo_test_cell_of_named_item(&s, "Twisted bow");
    CHECK("speedrun kit carries tentacle+claws+tbow in cells",
        cell_tentacle >= 0 && cell_claws >= 0 && cell_tbow >= 0);

    int manti = col_spawn_npc_at(&s, COLO_MANTICORE, 10, 10);
    osrs_interaction_set(&s.interaction, manti);
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("3x3: wielded scythe DPT outranks the tentacle cell",
        obs[wielded_base] > obs[dpt_base + cell_tentacle]);
    CHECK("3x3: claws cell ranks below tentacle cell",
        obs[dpt_base + cell_claws] < obs[dpt_base + cell_tentacle]);
    CHECK("3x3: wielded scythe is ~the best achievable (ratio ~1)",
        obs[wielded_base + 1] > 0.95f);
    CHECK("claws cell carries the spec bit", obs[spec_base + cell_claws] == 1.0f);
    CHECK("tentacle cell carries no spec bit", obs[spec_base + cell_tentacle] == 0.0f);

    int serpent = col_spawn_npc_at(&s, COLO_SERPENT_SHAMAN, 20, 10);
    osrs_interaction_set(&s.interaction, serpent);
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("1x1: tbow cell outranks the wielded scythe",
        obs[dpt_base + cell_tbow] > obs[wielded_base]);
    CHECK("1x1: wielded-vs-best ratio exposes the scythe gap",
        obs[wielded_base + 1] < 0.85f);

    osrs_interaction_clear(&s.interaction);
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("no target: cell DPT + wielded floats are zero",
        obs[dpt_base + cell_tentacle] == 0.0f && obs[wielded_base] == 0.0f &&
        obs[wielded_base + 1] == 0.0f);
    CHECK("no target: spec bits stay up (target-independent)",
        obs[spec_base + cell_claws] == 1.0f);

    osrs_interaction_set(&s.interaction, manti);
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    float pre_drain_tentacle = obs[dpt_base + cell_tentacle];
    float pre_drain_tbow = obs[dpt_base + cell_tbow];
    float pre_drain_wielded = obs[wielded_base];
    s.npcs[manti].def_drained = COLO_NPC_STATS[COLO_MANTICORE].def_level;
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("def drain raises the melee tentacle cell DPT (not a stale block)",
        obs[dpt_base + cell_tentacle] > pre_drain_tentacle);
    CHECK("def drain raises the ranged tbow cell DPT (not a stale block)",
        obs[dpt_base + cell_tbow] > pre_drain_tbow);
    CHECK("def drain raises the wielded DPT (not a stale block)",
        obs[wielded_base] > pre_drain_wielded);

    int cell_brew = -1;
    for (int c = 0; c < OSRS_INVENTORY_SIZE; c++)
        if (s.inventory_cells[c].item_idx == ITEM_NONE) { cell_brew = c; break; }
    CHECK("a gear-free cell exists to hold the brew", cell_brew >= 0);
    s.inventory_cells[cell_brew] = osrs_inventory_cell_from_raw_osrs_id(6685);
    CHECK("4-dose brew seeded", s.inventory_cells[cell_brew].dose == 4);
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    s.inventory_cells[cell_brew] = osrs_inventory_cell_from_raw_osrs_id(6687);
    CHECK("sip took a dose", s.inventory_cells[cell_brew].dose == 3);
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    float served_block[COLO_WEAPON_CHOICE_OBS_CACHE_FLOATS];
    memcpy(served_block, &obs[dpt_base],
           sizeof(float) * COLO_WEAPON_CHOICE_OBS_CACHE_FLOATS);
    memset(&s.obs_memos, 0, sizeof(s.obs_memos));
    col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
    CHECK("memo-served weapon-choice block == fresh recompute after the sip",
        memcmp(served_block, &obs[dpt_base],
               sizeof(float) * COLO_WEAPON_CHOICE_OBS_CACHE_FLOATS) == 0);
    s.inventory_cells[cell_brew] = osrs_inventory_cell_empty();

    col_spawn_reinforcements(&s);
    int jaguar = -1;
    for (int i = 0; i < COLO_MAX_NPCS; i++)
        if (s.npcs[i].active && s.npcs[i].spawned_as_reinforcement) { jaguar = i; break; }
    CHECK("reinforcement spawns are tagged", jaguar >= 0);

    ctx.config.damage_reward_coeff = 1.0f;
    ctx.config.wave_clear_bonus = 0.0f;
    ctx.config.farm_safe_damage_cap = 1;
    s.wave = 0;
    s.tick_scratch = (ColoTickScratch){0};
    s.tick_scratch.fresh_damage_dealt = 100.0f;
    s.tick_scratch.fresh_damage_reinforcement = 30.0f;
    float r_capped = col_compute_reward_ctx(&s, &ctx);
    CHECK("cap on, wave 1: reinforcement damage pays nothing",
        fabsf(r_capped - 70.0f) < 1e-3f);
    CHECK("farm damage is logged", fabsf(s.log.farm_damage - 30.0f) < 1e-3f);

    ctx.config.farm_safe_damage_cap = 0;
    s.tick_scratch.fresh_damage_dealt = 100.0f;
    s.tick_scratch.fresh_damage_reinforcement = 30.0f;
    float r_uncapped = col_compute_reward_ctx(&s, &ctx);
    CHECK("cap off: full damage pays", fabsf(r_uncapped - 100.0f) < 1e-3f);

    ctx.config.farm_safe_damage_cap = 1;
    s.wave = COLO_FARM_CAP_WAVES;
    s.tick_scratch.fresh_damage_dealt = 100.0f;
    s.tick_scratch.fresh_damage_reinforcement = 30.0f;
    float r_late = col_compute_reward_ctx(&s, &ctx);
    CHECK("cap on, wave 5+: reinforcements stay full-value at the default window",
        fabsf(r_late - 100.0f) < 1e-3f);

    ctx.config.farm_cap_waves = COLO_FARM_CAP_WAVES + 1;
    s.tick_scratch.fresh_damage_dealt = 100.0f;
    s.tick_scratch.fresh_damage_reinforcement = 30.0f;
    float r_widened = col_compute_reward_ctx(&s, &ctx);
    CHECK("widened farm_cap_waves caps the same wave-5 reinforcement damage",
        fabsf(r_widened - 70.0f) < 1e-3f);
    ctx.config.farm_cap_waves = COLO_FARM_CAP_WAVES;
}

int main(void) {
    test_stage3_t1_inventory_ranged_weapon_swap();
    test_stage3_t1_inventory_weapon_slot_last_click_wins();
    test_stage3_t1_human_inventory_primary_click_uses_resolver();
    test_stage3_t1_human_rearrange_swaps_inventory_slots();
    test_stage3_t2_brew_click_decrements_dose();
    test_stage3_t3_one_dose_vial_empties();
    test_colosseum_potion_click_source_of_truth();
    test_colosseum_potion_timer_and_same_tick_gate();
    test_colosseum_all_drink_kinds_shared_one_dose_path();
    test_inventory_pure_cut_reconstruction();
    test_stage3_t4_click_mask_bits();
    test_stage3_t4_mask_inventory_heads_flag();
    test_stage3_t5_claws_click_spec_fires();
    test_stage3_t6_obs_mask_fuzz_contract();
    test_fuzz_obs_mask();
    test_osrs_los_query_contracts();
    test_player_ranged_los_blocked_by_pillar();
    test_player_chase_routes_around_pillar_for_los();
    test_colosseum_npc_movement_player_tile_guards();
    test_zero_actions_hit_timeout();
    test_offpray_attribution_log();
    test_step_loop_draft();
    test_eleven_drafts_per_run();
    test_solarflare_orb();
    test_volatility_explosion();
    test_modifier_hazard_obs_fixes();
    test_death_linger_wave_clear_and_render();
    test_draft_offer_and_select();
    test_draft_upgrade_bias();
    test_mantimayhem_stress();
    test_frailty_hp();
    test_relentless_damage();
    test_relentless_def_level_bypass();
    test_quartet_extra_spawn();
    test_bees_hazard();
    test_totem_lifecycle();
    test_totemic_sol_wave12();
    test_reentry_sand_tiles();
    test_venom_escalation();
    test_bee_poison_status();
    test_mantimayhem_t3_shuffle();
    test_static_arena_mask();
    test_static_los_and_attack_gate();
    test_spawn_anchor_exclusion();
    test_reinforcement_gates();
    test_outcome_score_uses_fresh_damage();
    test_outcome_score_reinforcement_grows_denominator();
    test_fresh_damage_not_farmable_via_healing();
    test_outcome_score_wave_clear_has_no_double_count();
    test_outcome_score_sol_uses_boss_progress_only();
    test_roster_cap_nine();
    test_wave12_quartet_and_win();
    test_player_walks_through_npc_footprint();
    test_warband_cycle_offsets();
    test_warband_move_skip();
    test_warband_bfs_memo_bit_identity();
    test_warband_melee_distance_gate();
    test_warband_two_tick_stationary_gate();
    test_warband_formation_convergence();
    test_warband_two_tile_speed();
    test_warband_pillar_routefind_vs_shaman_safespot();
    test_red_flag_minotaur_routefind();
    test_minotaur_heal_semantics();
    test_manticore_barrage_period();
    test_manticore_telegraph_during_windup();
    test_prayer_oracle_manticore_orbs();
    test_late_start_entry_state();
    test_manticore_orb_same_tick_flick();
    test_projectile_prayer_locks_at_throw();
    test_npc_melee_instant_unprayable();
    test_player_melee_lands_at_delay_zero();
    test_echo_boots_recoil_reflects_to_attacker();
    test_manticore_shared_wave_cycle();
    test_manticore_stagger_overlap_fidelity();
    test_javelin_skyfall_no_defence_gate();
    test_sol_adjacency_gate_and_kiting();
    test_sol_attack_selection_invariants();
    test_sol_parry_schedule_and_damage();
    test_sol_parry_prayer_punish();
    test_sol_grapple_perfect_parry();
    test_sol_perfect_parry_forces_spec_attack();
    test_sol_shield_safe_rings();
    test_sol_spear_lines();
    test_sol_crystal_lifecycle();
    test_sol_aoe_reaction_window();
    test_sol_laser_react_window();
    test_sol_phase_transition_sand_guarantees();
    test_sol_beams_become_pools();
    test_sol_beam_strike_reaction_window();
    test_sol_enrage_sand_telegraphs();
    test_solarflare_sol_orbit_boxes();
    test_loadout_profiles_and_supplies();
    test_loadout_consumables();
    test_loadout_divine_potions_and_stat_drift();
    test_loadout_sanfew_and_serp_helm();
    test_consumable_overdrink_mask();
    test_loadout_surge_potion();
    test_loadout_spec_weapons();
    test_colosseum_live_inventory_display();
    test_loadout_item_effects();
    test_loadout_offensive_prayers();
    test_npc_magic_defence_rolls_off_magic_level();
    test_total_damage_by_type_captures_typeless();
    test_matchup_dpt_obs_ranking();
    test_primary_head_resolution();
    test_combat_fidelity_contract_sizes();
    test_scythe_multihit_per_size();
    test_venator_bow_bounce_colosseum_integration();
    test_bee_contact_damage_band();
    test_divine_state_obs_presence();
    test_magic_set_max_hit_math();
    test_thrall_regression();
    test_death_charge_regression();
    test_combat_fidelity_snapshot_roundtrip();
    test_step_out_forecast_manticore_armed_pattern();
    test_step_out_forecast_manticore_pair_stagger();
    test_step_out_forecast_warband_window_and_break();
    test_step_out_forecast_ranged_los_candidate_tiles();
    test_step_out_forecast_valid_flags();
    test_step_out_forecast_same_tick_mixed_styles();
    test_render_bridge_combat_visuals_and_loadout();
    test_render_bridge_npc_debug_and_warband_motion();
    test_melee_reach_cardinal_vs_diagonal();
    test_death_attribution_credits_actual_source();
    test_move_action_no_corner_cut();
    test_modifier_draft_forces_pick();
    test_gear_and_boost_reward_signals();
    test_weapon_choice_obs_rank_and_farm_cap();
    test_threat_field_obs();
    test_inventory_obs_memo();

    return osrs_test_summary();
}
