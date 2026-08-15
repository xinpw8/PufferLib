#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "ocean/osrs/osrs_policy.h"
#include "ocean/osrs/encounters/encounter_inferno.h"
static void inf_init_unfinalized_context(InfernoContext* ctx) {
    inf_init_context_typed(ctx);
}
#define inf_init_context_typed(ctx_ptr) do { \
    inf_init_context_typed(ctx_ptr); \
    inf_finalize_route_topology((ctx_ptr)); \
} while (0)
#include "ocean/osrs/osrs_anim.h"
#include "ocean/osrs/osrs_effects.h"
#include "ocean/osrs/osrs_projectile_orientation.h"
#include "ocean/osrs/osrs_render_motion.h"
#include <math.h>

#include "ocean/osrs/tests/osrs_test_check.h"


static void assert_child_aborts(const char* label, void (*fn)(void)) {
    fflush(NULL);
    pid_t pid = fork();
    if (pid == 0) {
        fn();
        _exit(0);
    }

    int status = 0;
    waitpid(pid, &status, 0);
    tests_run++;
    if (WIFSIGNALED(status) || (WIFEXITED(status) && WEXITSTATUS(status) != 0)) {
        tests_passed++;
    } else {
        tests_failed++;
        printf("  FAIL: %s - child returned successfully\n", label);
    }
}


#define ASSERT_STR_EQ(label, actual, expected) do { \
    tests_run++; \
    if (strcmp((actual), (expected)) == 0) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  FAIL: %s - got \"%s\", expected \"%s\"\n", \
            (label), (actual), (expected)); \
    } \
} while (0)

static InfernoContext test_context;

static void reset_test_context(void) {
    InfernoContext* ctx = &test_context;
    ctx->config = inf_default_config();
    ctx->collision_map = NULL;
    ctx->world_offset_x = 0;
    ctx->world_offset_y = 0;
    ctx->human_commands = NULL;
    ctx->human_command_count = 0;
    ctx->human_command_mode = 0;
    inf_reset_npc_player_los_frame(ctx);
}

static InfernoState make_test_state(int player_x, int player_y) {
    reset_test_context();
    InfernoState state;
    memset(&state, 0, sizeof(state));
    state.player.x = player_x;
    state.player.y = player_y;
    state.player_last_interaction_target_slot = -1;
    state.player_last_interaction_age = 1;
    return state;
}

static InfConfig* test_config(void) {
    return &test_context.config;
}

static float test_supply_milestone_surplus_reward(
    InfernoState* state,
    int public_wave
) {
    return inf_supply_milestone_surplus_reward(
        state, &test_context, public_wave);
}

static InfNPC make_test_npc(InfNPCType type, int x, int y, int size) {
    InfNPC npc;
    memset(&npc, 0, sizeof(npc));
    npc.type = type;
    npc.x = x;
    npc.y = y;
    npc.size = size;
    npc.aggro_target = -1;
    npc.attack_visual_target = -1;
    npc.resurrection_visual_target = -1;
    npc.attack_style = INF_NPC_STATS[type].default_style;
    npc.blob_scanned_prayer = -1;
    inf_npc_init_type_state(&npc);
    return npc;
}

static InfNPCStats make_test_stats(int default_style) {
    InfNPCStats stats;
    memset(&stats, 0, sizeof(stats));
    stats.default_style = default_style;
    stats.can_melee = 1;
    return stats;
}

static HumanInput make_human_input(void) {
    HumanInput input;
    memset(&input, 0, sizeof(input));
    input.pending_move_x = -1;
    input.pending_move_y = -1;
    input.pending_prayer = -1;
    input.pending_offensive_prayer = -1;
    input.pending_target_idx = -1;
    return input;
}

static void init_spell_cast_test_state(InfernoState* state, InfNPCType target_type) {
    *state = make_test_state(10, 10);
    state->rng_state = 1;
    state->weapon_set = INF_GEAR_MAGE;
    state->player.entity_type = ENTITY_PLAYER;
    state->player.base_hitpoints = 99;
    state->player.current_hitpoints = 99;
    state->player.base_attack = 99;
    state->player.base_strength = 99;
    state->player.base_defence = 99;
    state->player.base_ranged = 99;
    state->player.base_magic = 99;
    state->player.current_attack = 99;
    state->player.current_strength = 99;
    state->player.current_defence = 99;
    state->player.current_ranged = 99;
    state->player.current_magic = 99;
    state->player.autocast_enabled = 1;
    state->player.autocast_defensive = 0;
    state->player.autocast_spell = ENCOUNTER_SPELL_BLOOD;
    state->player_dest_x = -1;
    state->player_dest_y = -1;
    osrs_interaction_init(&state->interaction);
    encounter_apply_loadout(&state->player, INF_MAX_MAGE_LOADOUT, GEAR_MAGE);
    inf_refresh_live_stats(state);
    encounter_compute_loadout_stats(INF_MAX_MAGE_LOADOUT, ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_AUTOCAST, 30,
        &state->loadout_stats[INF_GEAR_MAGE]);
    encounter_compute_loadout_stats(INF_MAX_RANGE_LONG_LOADOUT, ATTACK_STYLE_RANGED,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_RAPID, 0,
        &state->loadout_stats[INF_GEAR_LONG_RANGE]);
    encounter_compute_loadout_stats(INF_MAX_RANGE_FAST_LOADOUT, ATTACK_STYLE_RANGED,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_RAPID, 0,
        &state->loadout_stats[INF_GEAR_BP]);

    state->npcs[0] = make_test_npc(
        target_type, 16, 10, INF_NPC_STATS[target_type].size);
    state->npcs[0].active = 1;
    state->npcs[0].hp = state->npcs[0].max_hp = INF_NPC_STATS[target_type].hp;
    inf_refresh_current_obs_slots_ctx(state, &test_context);
}

static int inf_action_target_for_npc(InfernoState* state, int npc_slot) {
    inf_refresh_current_obs_slots_ctx(state, &test_context);
    int target_slot = inf_find_target_obs_slot(state, npc_slot);
    ASSERT_INT_EQ("target NPC has observation slot", target_slot >= 0, 1);
    if (target_slot < 0) return 0;
    return inf_primary_attack_action_for_obs_slot(target_slot);
}

static int inferno_action_head_mask_offset(int head) {
    int offset = 0;
    for (int h = 0; h < head; h++)
        offset += INF_ACTION_DIMS[h];
    return offset;
}

static void fire_player_action_at_slot_zero(
    InfernoState* state,
    int spell_action
) {
    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_PRIMARY] = inf_action_target_for_npc(state, 0);
    actions[INF_HEAD_SPELL] = spell_action;
    state->player.attack_timer = 0;
    encounter_pending_hit_queue_clear(&state->npcs[0].pending_hits);
    inf_tick_player_ctx(state, &test_context, actions, 1);
}

static int inferno_pending_hit_obs_start(void);
static int inferno_pillar_obs_start(int pillar_idx);
static int inferno_obs_slot_dig_index(int slot_idx);
static int inferno_obs_slot_start(int slot_idx);
static void init_zuk_timing_state(InfernoState* state);


static void init_jad_timing_test_state(InfernoState* state, int player_x, int player_y, int jad_x, int jad_y) {
    if (player_x == 10 && player_y == 10) {
        int distance = jad_x - player_x;
        player_x = 20;
        player_y = 20;
        if (distance > 10) {
            jad_x = player_x;
            jad_y = player_y + distance;
        } else {
            jad_x = player_x + distance;
            jad_y = player_y;
        }
    }
    reset_test_context();
    memset(state, 0, sizeof(*state));
    state->rng_state = 12345;
    state->wave = 66;
    state->player.entity_type = ENTITY_PLAYER;
    state->player.x = player_x;
    state->player.y = player_y;
    state->player.base_hitpoints = 99;
    state->player.current_hitpoints = 99;
    state->player.base_prayer = 99;
    state->player.current_prayer = 99;
    state->player.base_attack = 99;
    state->player.base_strength = 99;
    state->player.base_defence = 99;
    state->player.base_ranged = 99;
    state->player.base_magic = 99;
    state->player.current_attack = 99;
    state->player.current_strength = 99;
    state->player.current_defence = 99;
    state->player.current_ranged = 99;
    state->player.current_magic = 99;
    state->player.prayer = PRAYER_NONE;
    state->weapon_set = INF_GEAR_MAGE;
    state->player_last_interaction_target_slot = -1;
    state->player_last_interaction_age = 1;
    state->player_dest_x = -1;
    state->player_dest_y = -1;
    osrs_interaction_init(&state->interaction);
    encounter_compute_loadout_stats(INF_MAX_MAGE_LOADOUT, ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_AUTOCAST, 30,
        &state->loadout_stats[INF_GEAR_MAGE]);

    state->npcs[0] = make_test_npc(
        INF_NPC_JAD, jad_x, jad_y, INF_NPC_STATS[INF_NPC_JAD].size);
    state->npcs[0].active = 1;
    state->npcs[0].attack_timer = 0;
    inf_npc_jad(&state->npcs[0])->attack_style = ATTACK_STYLE_MAGIC;
    state->npcs[0].attack_style = ATTACK_STYLE_RANGED;
}

static void step_inferno_with_prayer(InfernoState* state, int prayer_action) {
    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_PRAYER] = prayer_action;
    inf_step_ctx((EncounterState*)state, (EncounterContext*)&test_context, actions);
}

static void step_inferno_noop(InfernoState* state) {
    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    inf_step_ctx((EncounterState*)state, (EncounterContext*)&test_context, actions);
}

static int find_active_npc_type(const InfernoState* state, InfNPCType type) {
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (state->npcs[i].active && state->npcs[i].type == type)
            return i;
    }
    return -1;
}

static int count_active_npcs(const InfernoState* state) {
    int count = 0;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (state->npcs[i].active) count++;
    }
    return count;
}

static int count_active_npc_type(const InfernoState* state, InfNPCType type) {
    int count = 0;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (state->npcs[i].active && state->npcs[i].type == type)
            count++;
    }
    return count;
}

static int find_active_npc_type_at(
    const InfernoState* state, InfNPCType type, int x, int y
) {
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (state->npcs[i].active && state->npcs[i].type == type &&
                state->npcs[i].x == x && state->npcs[i].y == y)
            return i;
    }
    return -1;
}

static int force_mager_resurrect(InfernoState* s, int idx) {
    for (uint32_t seed = 1; seed < 100000; seed++) {
        InfernoState probe = *s;
        probe.rng_state = seed;
        if (inf_mager_resurrect_ctx(&probe, &test_context, idx)) {
            s->rng_state = seed;
            return inf_mager_resurrect_ctx(s, &test_context, idx);
        }
    }
    return 0;
}

static int force_mager_attack_resurrection(InfernoState* s, int idx) {
    for (uint32_t seed = 1; seed < 100000; seed++) {
        InfernoState probe = *s;
        probe.rng_state = seed;
        inf_npc_attack_ctx(&probe, &test_context, idx);
        if (probe.dead_mob_count == 0 &&
                count_active_npc_type(&probe, INF_NPC_RANGER) > 0) {
            s->rng_state = seed;
            inf_npc_attack_ctx(s, &test_context, idx);
            return 1;
        }
    }
    return 0;
}

static int distance_to_player(const InfernoState* state, const InfNPC* npc) {
    return encounter_dist_to_npc(
        state->player.x, state->player.y, npc->x, npc->y, npc->size);
}

static int test_profiled_supply_count(int full_doses, float profile_fraction, float scale) {
    float effective_fraction = 1.0f - scale * (1.0f - profile_fraction);
    int doses = (int)((float)full_doses * effective_fraction + 0.5f);
    if (doses < 0) doses = 0;
    if (doses > full_doses) doses = full_doses;
    return doses;
}

static void reset_inferno_at_public_wave(EncounterState* raw_state,
                                         int public_wave,
                                         float supply_profile_scale) {
    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "start_wave", public_wave);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "shield_penalty_coeff", 0.01f);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "tag_reward_coeff", 0.25f);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "late_start_supply_profile_scale", supply_profile_scale);
    inf_reset_ctx(raw_state, (EncounterContext*)&test_context, 123);
}

static void assert_supply_doses(const char* label,
                                const Player* player,
                                InfSupplyDoses expected) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s brew doses", label);
    ASSERT_INT_EQ(buf, player->brew_doses, expected.brew_doses);
    snprintf(buf, sizeof(buf), "%s restore doses", label);
    ASSERT_INT_EQ(buf, player->restore_doses, expected.restore_doses);
    snprintf(buf, sizeof(buf), "%s bastion doses", label);
    ASSERT_INT_EQ(buf, player->bastion_doses, expected.bastion_doses);
    snprintf(buf, sizeof(buf), "%s stamina doses", label);
    ASSERT_INT_EQ(buf, player->stamina_doses, expected.stamina_doses);
}

static int test_occupied_inventory_cells(const InfernoState* s) {
    int occupied = 0;
    for (int c = 0; c < OSRS_INVENTORY_SIZE; c++) {
        if (!osrs_inventory_cell_is_empty(&s->player.inventory_cells[c]))
            occupied++;
    }
    return occupied;
}

static int test_cell_holding_item(const InfernoState* s, uint8_t item) {
    for (int c = 0; c < OSRS_INVENTORY_SIZE; c++) {
        if (osrs_inventory_cell_item_index(&s->player.inventory_cells[c]) == item)
            return c;
    }
    return -1;
}

static int test_cell_doses_of_kind(const InfernoState* s, OsrsConsumableKind kind) {
    int doses = 0;
    for (int c = 0; c < OSRS_INVENTORY_SIZE; c++) {
        const OsrsInventoryCell* cell = &s->player.inventory_cells[c];
        if (osrs_inventory_cell_raw_osrs_id(cell) == 0) continue;
        if (osrs_consumable_click_lookup_raw_osrs_id(
                osrs_inventory_cell_raw_osrs_id(cell)).consumable_kind == kind)
            doses += osrs_inventory_cell_dose_count(cell);
    }
    return doses;
}

static OsrsConsumableKind test_drink_click_kind(const InfernoState* s, int action) {
    if (action <= 0) return OSRS_CONSUMABLE_NONE;
    return osrs_consumable_click_lookup_raw_osrs_id(
        osrs_inventory_cell_raw_osrs_id(&s->player.inventory_cells[action - 1])).consumable_kind;
}

static void test_final_wave_reward_applies_healer_tags_and_heal_cost(void) {
    printf("--- final-wave reward applies healer tags and heal cost ---\n");

    InfernoState healing_state = make_test_state(24, 24);
    InfernoState damage_state = make_test_state(24, 24);

    inf_put_float_ctx((EncounterState*)&healing_state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&healing_state, (EncounterContext*)&test_context, "shield_penalty_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&healing_state, (EncounterContext*)&test_context, "tag_reward_coeff", 0.25f);
    healing_state.wave = INF_NUM_WAVES - 1;
    healing_state.tick_scratch.damage_dealt = 50.0f;
    healing_state.tick_scratch.hp_restored = 10.0f;
    healing_state.tick_scratch.shield_damage = 7.0f;
    healing_state.tick_scratch.healer_tags = 2;
    healing_state.npcs[0] = make_test_npc(INF_NPC_HEALER_ZUK, 26, 24, 1);
    healing_state.npcs[0].active = 1;
    healing_state.npcs[0].aggro_target = 1;
    healing_state.npcs[1] = make_test_npc(INF_NPC_ZUK, 28, 24, 5);
    healing_state.npcs[1].active = 1;
    healing_state.npcs[1].hp = 1150;
    healing_state.min_zuk_hp_seen = 1200.0f;

    damage_state = healing_state;
    damage_state.wave = 0;
    damage_state.npcs[0].aggro_target = -1;
    damage_state.tick_scratch.healer_tags = 0;

    ASSERT_FLOAT_NEAR("active healer reward includes tags, heal cost, and shield penalty",
        inf_compute_reward_ctx(&healing_state, &test_context), 0.33f, 0.0001f);
    ASSERT_FLOAT_NEAR("active healer reward updates zuk low watermark",
        healing_state.min_zuk_hp_seen, 1150.0f, 0.0001f);
    ASSERT_FLOAT_NEAR("non-final-wave reward still uses damage path",
        inf_compute_reward_ctx(&damage_state, &test_context), 0.33f, 0.0001f);
}

static void test_final_wave_reward_uses_zuk_low_watermark_progress(void) {
    printf("--- final-wave reward uses zuk low-watermark progress ---\n");

    InfernoState state = make_test_state(24, 24);

    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "shield_penalty_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "tag_reward_coeff", 0.25f);
    state.wave = INF_NUM_WAVES - 1;
    state.min_zuk_hp_seen = 1200.0f;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 1150;
    state.npcs[0].max_hp = 1200;
    state.npcs[1] = make_test_npc(INF_NPC_JAD, 24, 32, 5);
    state.npcs[1].active = 1;

    state.tick_scratch.damage_dealt = 250.0f;
    state.tick_scratch.hp_restored = 100.0f;
    state.tick_scratch.shield_damage = 7.0f;
    ASSERT_FLOAT_NEAR("first zuk low watermark pays progress minus shield penalty",
        inf_compute_reward_ctx(&state, &test_context), 0.43f, 0.0001f);
    ASSERT_FLOAT_NEAR("first zuk low watermark updates state",
        state.min_zuk_hp_seen, 1150.0f, 0.0001f);

    state.tick_scratch.damage_dealt = 400.0f;
    state.tick_scratch.hp_restored = 0.0f;
    state.tick_scratch.shield_damage = 0.0f;
    ASSERT_FLOAT_NEAR("repeated hits at same zuk hp give zero reward",
        inf_compute_reward_ctx(&state, &test_context), 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR("same-hp hits keep low watermark",
        state.min_zuk_hp_seen, 1150.0f, 0.0001f);

    state.tick_scratch.damage_dealt = 600.0f;
    state.npcs[0].hp = 1180;
    ASSERT_FLOAT_NEAR("healed zuk above low watermark gives zero reward",
        inf_compute_reward_ctx(&state, &test_context), 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR("healed zuk does not revoke low watermark",
        state.min_zuk_hp_seen, 1150.0f, 0.0001f);

    state.tick_scratch.damage_dealt = 900.0f;
    ASSERT_FLOAT_NEAR("non-zuk damage without new low watermark gives zero reward",
        inf_compute_reward_ctx(&state, &test_context), 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR("non-zuk damage leaves low watermark unchanged",
        state.min_zuk_hp_seen, 1150.0f, 0.0001f);

    state.npcs[0].hp = 1140;
    state.tick_scratch.damage_dealt = 50.0f;
    ASSERT_FLOAT_NEAR("new lower zuk hp pays only incremental progress",
        inf_compute_reward_ctx(&state, &test_context), 0.10f, 0.0001f);
    ASSERT_FLOAT_NEAR("new lower zuk hp refreshes low watermark",
        state.min_zuk_hp_seen, 1140.0f, 0.0001f);
}

static void test_final_wave_reward_blocks_zuk_damage_while_healers_heal(void) {
    printf("--- final-wave reward blocks zuk damage while healers heal ---\n");

    InfernoState state = make_test_state(24, 24);

    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "shield_penalty_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "tag_reward_coeff", 0.25f);
    state.wave = INF_NUM_WAVES - 1;
    state.min_zuk_hp_seen = 240.0f;
    state.zuk.healer_spawned = 1;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 220;
    state.npcs[0].max_hp = 1200;
    state.npcs[1] = make_test_npc(INF_NPC_HEALER_ZUK, 20, 34, 1);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp =
        INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[1].aggro_target = 0;

    state.min_zuk_hp_seen = 245.0f;
    state.npcs[0].hp = 235;
    ASSERT_FLOAT_NEAR("zuk threshold-crossing hit pays once after healer spawn",
        inf_compute_reward_ctx(&state, &test_context), 0.10f, 0.0001f);
    ASSERT_FLOAT_NEAR("threshold-crossing hit updates low watermark",
        state.min_zuk_hp_seen, 235.0f, 0.0001f);

    state.min_zuk_hp_seen = 240.0f;
    state.npcs[0].hp = 220;

    ASSERT_FLOAT_NEAR("zuk progress pays nothing while a healer heals zuk",
        inf_compute_reward_ctx(&state, &test_context), 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR("low watermark still tracks observed zuk hp",
        state.min_zuk_hp_seen, 220.0f, 0.0001f);

    state.min_zuk_hp_seen = 240.0f;
    state.npcs[1].aggro_target = -1;

    ASSERT_FLOAT_NEAR("zuk progress resumes after healers are tagged",
        inf_compute_reward_ctx(&state, &test_context), 0.20f, 0.0001f);
}

static void test_final_wave_reward_pays_zuk_healer_damage(void) {
    printf("--- final-wave reward pays zuk healer damage ---\n");

    InfernoState state = make_test_state(24, 24);

    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "shield_penalty_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "tag_reward_coeff", 0.25f);
    state.wave = INF_NUM_WAVES - 1;
    state.min_zuk_hp_seen = 240.0f;
    state.zuk.healer_spawned = 1;
    state.tick_scratch.damage_dealt = 31.0f;
    state.tick_scratch.damage_zuk_healers = 31.0f;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 240;
    state.npcs[0].max_hp = 1200;
    state.npcs[1] = make_test_npc(INF_NPC_HEALER_ZUK, 20, 34, 1);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp =
        INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[1].aggro_target = -1;

    ASSERT_FLOAT_NEAR("zuk healer damage uses base damage reward",
        inf_compute_reward_ctx(&state, &test_context), 0.31f, 0.0001f);
}

static void test_post_healer_zuk_damage_reward_is_after_clear_only(void) {
    printf("--- post-healer Zuk damage reward is after clear only ---\n");

    InfernoState state = make_test_state(24, 24);

    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "post_healer_zuk_damage_coeff", 0.005f);
    state.wave = INF_NUM_WAVES - 1;
    state.min_zuk_hp_seen = 180.0f;
    state.zuk.healer_spawned = 1;
    state.tick_at_all_zuk_healers_dead = -1;
    state.tick_scratch.damage_dealt = 50.0f;
    state.tick_scratch.damage_zuk = 50.0f;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 220;
    state.npcs[0].max_hp = 1200;

    ASSERT_FLOAT_NEAR("post-healer coeff does not pay before healer clear",
        inf_compute_reward_ctx(&state, &test_context), 0.0f, 0.0001f);

    state.tick_at_all_zuk_healers_dead = 321;
    state.min_zuk_hp_seen = 180.0f;
    ASSERT_FLOAT_NEAR("post-healer coeff pays healed-back Zuk damage after clear",
        inf_compute_reward_ctx(&state, &test_context), 0.25f, 0.0001f);
    ASSERT_FLOAT_NEAR("post-healer damage reward does not revoke low watermark",
        state.min_zuk_hp_seen, 180.0f, 0.0001f);
}

static void test_zuk_healer_phase_hp_delta_default_preserves_low_watermark(void) {
    printf("--- Zuk healer phase HP delta reward defaults off ---\n");

    InfernoState state = make_test_state(24, 24);

    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    state.wave = INF_NUM_WAVES - 1;
    state.zuk.healer_spawned = 1;
    state.min_zuk_hp_seen = 180.0f;
    state.tick_scratch.damage_zuk = 50.0f;
    state.tick_scratch.damage_dealt = 50.0f;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 220;
    state.npcs[0].max_hp = 1200;

    ASSERT_FLOAT_NEAR("default coefficient preserves old no-pay healed-back Zuk hit",
        inf_compute_reward_ctx(&state, &test_context), 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR("default coefficient preserves low watermark",
        state.min_zuk_hp_seen, 180.0f, 0.0001f);
}

static void test_zuk_healer_phase_hp_delta_pays_healed_back_zuk_damage(void) {
    printf("--- Zuk healer phase HP delta pays healed-back Zuk damage ---\n");

    InfernoState state = make_test_state(24, 24);

    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "zuk_healer_phase_hp_delta_coeff", 0.005f);
    state.wave = INF_NUM_WAVES - 1;
    state.zuk.healer_spawned = 1;
    state.min_zuk_hp_seen = 180.0f;
    state.tick_scratch.damage_zuk = 50.0f;
    state.tick_scratch.damage_dealt = 50.0f;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 220;
    state.npcs[0].max_hp = 1200;

    ASSERT_FLOAT_NEAR("healer-phase HP delta pays current Zuk damage",
        inf_compute_reward_ctx(&state, &test_context), 0.25f, 0.0001f);
    ASSERT_FLOAT_NEAR("healed-back Zuk hit does not change low watermark",
        state.min_zuk_hp_seen, 180.0f, 0.0001f);
}

static void test_zuk_healer_phase_hp_delta_avoids_double_pay_below_low_watermark(void) {
    printf("--- Zuk healer phase HP delta avoids double pay below low watermark ---\n");

    InfernoState state = make_test_state(24, 24);

    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "zuk_healer_phase_hp_delta_coeff", 0.005f);
    state.wave = INF_NUM_WAVES - 1;
    state.zuk.healer_spawned = 1;
    state.min_zuk_hp_seen = 245.0f;
    state.tick_scratch.damage_zuk = 10.0f;
    state.tick_scratch.damage_dealt = 10.0f;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 235;
    state.npcs[0].max_hp = 1200;

    ASSERT_FLOAT_NEAR("post-spawn low-watermark damage is paid once through HP delta",
        inf_compute_reward_ctx(&state, &test_context), 0.05f, 0.0001f);
    ASSERT_FLOAT_NEAR("low watermark still tracks metrics",
        state.min_zuk_hp_seen, 235.0f, 0.0001f);
}

static void test_zuk_healer_phase_hp_delta_penalizes_zuk_healing_once(void) {
    printf("--- Zuk healer phase HP delta penalizes Zuk healing once ---\n");

    InfernoState state = make_test_state(24, 24);

    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "zuk_healer_phase_hp_delta_coeff", 0.005f);
    state.wave = INF_NUM_WAVES - 1;
    state.zuk.healer_spawned = 1;
    state.min_zuk_hp_seen = 180.0f;
    state.tick_scratch.hp_restored = 40.0f;
    state.tick_scratch.hp_restored_zuk = 40.0f;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 220;
    state.npcs[0].max_hp = 1200;
    state.npcs[1] = make_test_npc(INF_NPC_HEALER_ZUK, 20, 34, 1);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp =
        INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[1].aggro_target = 0;

    ASSERT_FLOAT_NEAR("Zuk heal penalty is not double-counted",
        inf_compute_reward_ctx(&state, &test_context), -0.20f, 0.0001f);
}

static void test_zuk_healer_phase_hp_delta_pays_net_same_tick_delta(void) {
    printf("--- Zuk healer phase HP delta pays net same-tick delta ---\n");

    InfernoState state = make_test_state(24, 24);

    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "zuk_healer_phase_hp_delta_coeff", 0.005f);
    state.wave = INF_NUM_WAVES - 1;
    state.zuk.healer_spawned = 1;
    state.min_zuk_hp_seen = 180.0f;
    state.tick_scratch.damage_zuk = 50.0f;
    state.tick_scratch.damage_dealt = 50.0f;
    state.tick_scratch.hp_restored = 20.0f;
    state.tick_scratch.hp_restored_zuk = 20.0f;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 220;
    state.npcs[0].max_hp = 1200;
    state.npcs[1] = make_test_npc(INF_NPC_HEALER_ZUK, 20, 34, 1);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp =
        INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[1].aggro_target = 0;

    ASSERT_FLOAT_NEAR("same tick damage and healing use net Zuk HP delta",
        inf_compute_reward_ctx(&state, &test_context), 0.15f, 0.0001f);
}

static void test_zuk_healer_phase_hp_delta_keeps_non_zuk_heal_cost(void) {
    printf("--- Zuk healer phase HP delta keeps non-Zuk heal cost ---\n");

    InfernoState state = make_test_state(24, 24);

    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "zuk_healer_phase_hp_delta_coeff", 0.005f);
    state.wave = INF_NUM_WAVES - 1;
    state.zuk.healer_spawned = 1;
    state.min_zuk_hp_seen = 180.0f;
    state.tick_scratch.hp_restored = 30.0f;
    state.tick_scratch.hp_restored_zuk = 0.0f;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 220;
    state.npcs[0].max_hp = 1200;
    state.npcs[1] = make_test_npc(INF_NPC_JAD, 24, 32, 5);
    state.npcs[1].active = 1;
    state.npcs[1].hp = 200;
    state.npcs[1].max_hp = INF_NPC_STATS[INF_NPC_JAD].hp;
    state.npcs[2] = make_test_npc(INF_NPC_HEALER_JAD, 20, 34, 1);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp =
        INF_NPC_STATS[INF_NPC_HEALER_JAD].hp;
    state.npcs[2].aggro_target = 1;

    ASSERT_FLOAT_NEAR("non-Zuk healing still uses generic heal cost",
        inf_compute_reward_ctx(&state, &test_context), -0.30f, 0.0001f);
}

static void init_post_healer_set_reward_state(InfernoState* state) {
    *state = make_test_state(24, 24);
    state->wave = INF_NUM_WAVES - 1;
    state->tick = 200;
    state->tick_at_all_zuk_healers_dead = 100;
    state->min_zuk_hp_seen = 500.0f;
    state->npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state->npcs[0].active = 1;
    state->npcs[0].hp = 500;
    state->npcs[0].max_hp = 1200;
}

static void test_post_healer_set_damage_reward_defaults_off(void) {
    printf("--- post-healer set damage reward defaults off ---\n");

    InfernoState state;
    init_post_healer_set_reward_state(&state);
    state.tick_scratch.damage_set = 50.0f;

    ASSERT_FLOAT_NEAR("default post-healer set damage reward is off",
        inf_compute_reward_ctx(&state, &test_context), 0.0f, 0.0001f);
}

static void test_post_healer_set_damage_reward_pays_after_healer_clear(void) {
    printf("--- post-healer set damage reward pays after healer clear ---\n");

    InfernoState state;
    init_post_healer_set_reward_state(&state);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "post_healer_set_damage_reward_coeff", 0.002f);
    state.tick_scratch.damage_set = 50.0f;

    ASSERT_FLOAT_NEAR("post-healer set damage is rewarded",
        inf_compute_reward_ctx(&state, &test_context), 0.10f, 0.0001f);
}

static void test_post_healer_set_kill_bonus_uses_existing_emitter(void) {
    printf("--- post-healer set kill bonus uses existing emitter ---\n");

    InfernoState state;
    init_post_healer_set_reward_state(&state);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "post_healer_set_kill_bonus", 0.09f);
    state.tick_scratch.kill_set = 1;

    ASSERT_FLOAT_NEAR("post-healer set kill bonus emits through set channel",
        inf_compute_reward_ctx(&state, &test_context), 0.03f, 0.0001f);
    ASSERT_FLOAT_NEAR("remaining post-healer set kill bonus is pending",
        state.pending_set_kill_bonus, 0.06f, 0.0001f);
}

static void test_post_healer_set_alive_penalty_caps_per_episode(void) {
    printf("--- post-healer set alive penalty caps per episode ---\n");

    InfernoState state;
    init_post_healer_set_reward_state(&state);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "post_healer_set_alive_tick_penalty_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "post_healer_set_alive_penalty_cap", 0.03f);
    state.post_healer_set_alive_penalty_total = 0.025f;
    state.npcs[1] = make_test_npc(INF_NPC_RANGER, 29, 36, 1);
    state.npcs[1].active = 1;
    state.npcs[1].hp = 46;
    state.npcs[2] = make_test_npc(INF_NPC_MAGER, 20, 36, 1);
    state.npcs[2].active = 1;
    state.npcs[2].hp = 70;

    ASSERT_FLOAT_NEAR("post-healer set alive penalty respects cap",
        inf_compute_reward_ctx(&state, &test_context), -0.005f, 0.0001f);
    ASSERT_FLOAT_NEAR("post-healer set alive penalty total reaches cap",
        state.post_healer_set_alive_penalty_total, 0.03f, 0.0001f);
}

static void test_zuk_untagged_healer_tick_penalty_defaults_off(void) {
    printf("--- Zuk untagged healer tick penalty defaults off ---\n");

    InfernoState state = make_test_state(24, 24);

    state.wave = INF_NUM_WAVES - 1;
    state.zuk.healer_spawned = 1;
    state.min_zuk_hp_seen = 180.0f;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 220;
    state.npcs[0].max_hp = 1200;
    state.npcs[1] = make_test_npc(INF_NPC_HEALER_ZUK, 20, 34, 1);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp =
        INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[1].aggro_target = 0;

    ASSERT_FLOAT_NEAR("default untagged healer pressure is off",
        inf_compute_reward_ctx(&state, &test_context), 0.0f, 0.0001f);
}

static void test_zuk_untagged_healer_tick_penalty_counts_only_untagged_zuk_healers(void) {
    printf("--- Zuk untagged healer tick penalty counts only untagged Zuk healers ---\n");

    InfernoState state = make_test_state(24, 24);

    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "zuk_untagged_healer_tick_penalty_coeff", 0.01f);
    state.wave = INF_NUM_WAVES - 1;
    state.zuk.healer_spawned = 1;
    state.min_zuk_hp_seen = 180.0f;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 220;
    state.npcs[0].max_hp = 1200;
    state.npcs[1] = make_test_npc(INF_NPC_HEALER_ZUK, 20, 34, 1);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp =
        INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[1].aggro_target = 0;
    state.npcs[2] = make_test_npc(INF_NPC_HEALER_ZUK, 21, 34, 1);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp =
        INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[2].aggro_target = -1;
    state.npcs[3] = make_test_npc(INF_NPC_JAD, 24, 32, 5);
    state.npcs[3].active = 1;
    state.npcs[3].hp = 200;
    state.npcs[3].max_hp = INF_NPC_STATS[INF_NPC_JAD].hp;
    state.npcs[4] = make_test_npc(INF_NPC_HEALER_JAD, 20, 35, 1);
    state.npcs[4].active = 1;
    state.npcs[4].hp = state.npcs[4].max_hp =
        INF_NPC_STATS[INF_NPC_HEALER_JAD].hp;
    state.npcs[4].aggro_target = 3;

    ASSERT_FLOAT_NEAR("only the untagged Zuk healer creates pressure",
        inf_compute_reward_ctx(&state, &test_context), -0.01f, 0.0001f);
}

static void test_zuk_untagged_healer_target_bonus_defaults_off(void) {
    printf("--- Zuk untagged healer target bonus defaults off ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    state.zuk.healer_spawned = 1;
    state.min_zuk_hp_seen = 180.0f;
    state.npcs[2] = make_test_npc(
        INF_NPC_HEALER_ZUK, 20, 48, INF_NPC_STATS[INF_NPC_HEALER_ZUK].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[2].aggro_target = 0;

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_PRIMARY] = inf_action_target_for_npc(&state, 2);
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("default target bonus pays no per-tick target reward",
        state.tick_scratch.zuk_untagged_healer_targets, 0);
    ASSERT_INT_EQ("default target bonus still records target attempts",
        state.total_zuk_untagged_healer_targets, 1);
    ASSERT_FLOAT_NEAR("default target bonus pays nothing",
        inf_compute_reward_ctx(&state, &test_context), 0.0f, 0.0001f);
}

static void test_zuk_untagged_healer_target_bonus_rewards_distinct_healers(void) {
    printf("--- Zuk untagged healer target bonus rewards distinct healers ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    state.zuk.healer_spawned = 1;
    state.min_zuk_hp_seen = 180.0f;
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "zuk_untagged_healer_target_bonus_coeff", 0.07f);

    state.npcs[2] = make_test_npc(
        INF_NPC_HEALER_ZUK, 20, 48, INF_NPC_STATS[INF_NPC_HEALER_ZUK].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[2].aggro_target = 0;
    state.npcs[3] = make_test_npc(
        INF_NPC_HEALER_ZUK, 21, 48, INF_NPC_STATS[INF_NPC_HEALER_ZUK].size);
    state.npcs[3].active = 1;
    state.npcs[3].hp = state.npcs[3].max_hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[3].aggro_target = 0;

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);
    int first_healer_slot = inf_find_target_obs_slot(&state, 2);
    int second_healer_slot = inf_find_target_obs_slot(&state, 3);
    ASSERT_INT_EQ("first healer has an observation slot",
        first_healer_slot >= 0, 1);
    ASSERT_INT_EQ("second healer has an observation slot",
        second_healer_slot >= 0, 1);
    ASSERT_INT_EQ("first healer slot maps to npc 2",
        state.current_obs_slots[first_healer_slot], 2);
    ASSERT_INT_EQ("second healer slot maps to npc 3",
        state.current_obs_slots[second_healer_slot], 3);

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_PRIMARY] = inf_primary_attack_action_for_obs_slot(first_healer_slot);
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("first untagged healer target rewarded",
        state.tick_scratch.zuk_untagged_healer_targets, 1);
    ASSERT_INT_EQ("first untagged healer target reward count",
        state.total_zuk_untagged_healer_target_rewards, 1);
    ASSERT_FLOAT_NEAR("first untagged healer target reward",
        inf_compute_reward_ctx(&state, &test_context), 0.07f, 0.0001f);

    state.tick_scratch.zuk_untagged_healer_targets = 0;
    actions[INF_HEAD_PRIMARY] = inf_primary_attack_action_for_obs_slot(first_healer_slot);
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("repeat target does not reward twice",
        state.tick_scratch.zuk_untagged_healer_targets, 0);

    actions[INF_HEAD_PRIMARY] = inf_primary_attack_action_for_obs_slot(second_healer_slot);
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("second distinct untagged healer target rewarded",
        state.tick_scratch.zuk_untagged_healer_targets, 1);
    ASSERT_INT_EQ("second distinct untagged healer target reward count",
        state.total_zuk_untagged_healer_target_rewards, 2);
}

static void test_zuk_safe_untagged_healer_target_bonus_records_safe_subset(void) {
    printf("--- Zuk safe untagged healer target bonus records safe subset ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    state.zuk.healer_spawned = 1;
    state.min_zuk_hp_seen = 180.0f;
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "zuk_safe_untagged_healer_target_bonus_coeff", 0.11f);

    state.npcs[2] = make_test_npc(
        INF_NPC_HEALER_ZUK, 20, 48, INF_NPC_STATS[INF_NPC_HEALER_ZUK].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[2].aggro_target = 0;

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_PRIMARY] = inf_action_target_for_npc(&state, 2);
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("safe untagged healer target attempt",
        state.total_zuk_untagged_healer_targets, 1);
    ASSERT_INT_EQ("safe untagged healer target subset count",
        state.total_zuk_safe_untagged_healer_targets, 1);
    ASSERT_INT_EQ("unsafe untagged healer target subset count",
        state.total_zuk_unsafe_untagged_healer_targets, 0);
    ASSERT_INT_EQ("safe untagged healer target reward count",
        state.total_zuk_safe_untagged_healer_target_rewards, 1);
    ASSERT_INT_EQ("safe untagged healer per-tick reward event",
        state.tick_scratch.zuk_safe_untagged_healer_targets, 1);
    ASSERT_FLOAT_NEAR("safe untagged healer target reward",
        inf_compute_reward_ctx(&state, &test_context), 0.11f, 0.0001f);
}

static void test_zuk_untagged_healer_target_bonus_excludes_tagged_healers(void) {
    printf("--- Zuk untagged healer target bonus excludes tagged healers ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    state.zuk.healer_spawned = 1;
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "zuk_untagged_healer_target_bonus_coeff", 0.07f);

    state.npcs[2] = make_test_npc(
        INF_NPC_HEALER_ZUK, 20, 48, INF_NPC_STATS[INF_NPC_HEALER_ZUK].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[2].aggro_target = -1;

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_PRIMARY] = inf_action_target_for_npc(&state, 2);
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("already tagged healer target gets no bonus",
        state.tick_scratch.zuk_untagged_healer_targets, 0);
}

static void test_zuk_healer_tags_first_reward_mode_blocks_pre_tag_damage(void) {
    printf("--- Zuk healer tags-first reward mode blocks pre-tag damage ---\n");

    InfernoState state = make_test_state(24, 24);
    state.wave = INF_NUM_WAVES - 1;
    state.zuk.healer_spawned = 1;
    test_config()->zuk_healer_reward_mode = 1;
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "tag_reward_coeff", 0.25f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "zuk_healer_damage_reward_coeff", 0.02f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "zuk_healer_kill_bonus", 0.30f);
    state.min_zuk_hp_seen = 300.0f;
    state.total_zuk_healer_tags = 2;
    state.tick_scratch.healer_tags = 1;
    state.tick_scratch.zuk_healer_tags = 1;
    state.tick_scratch.damage_zuk = 80.0f;
    state.tick_scratch.damage_zuk_healers = 10.0f;
    state.tick_scratch.kill_zuk_healer = 1;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 220;
    state.npcs[0].max_hp = 1200;

    ASSERT_FLOAT_NEAR("only healer tag reward is paid before all tags",
        inf_compute_reward_ctx(&state, &test_context), 0.25f, 0.0001f);
    ASSERT_FLOAT_NEAR("pre-tag kill bonus is not delayed",
        state.pending_zuk_healer_kill_bonus, 0.0f, 0.0001f);
    ASSERT_FLOAT_NEAR("low watermark still tracks for metrics",
        state.min_zuk_hp_seen, 220.0f, 0.0001f);
}

static void test_zuk_healer_tags_first_reward_mode_resumes_after_all_tags(void) {
    printf("--- Zuk healer tags-first reward mode resumes after all tags ---\n");

    InfernoState state = make_test_state(24, 24);
    state.wave = INF_NUM_WAVES - 1;
    state.zuk.healer_spawned = 1;
    test_config()->zuk_healer_reward_mode = 1;
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "zuk_healer_damage_reward_coeff", 0.02f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "zuk_healer_kill_bonus", 0.30f);
    state.min_zuk_hp_seen = 300.0f;
    state.total_zuk_healer_tags = 4;
    state.tick_scratch.damage_zuk = 80.0f;
    state.tick_scratch.damage_zuk_healers = 10.0f;
    state.tick_scratch.kill_zuk_healer = 1;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 220;
    state.npcs[0].max_hp = 1200;

    ASSERT_FLOAT_NEAR("damage and kill rewards resume after all tags",
        inf_compute_reward_ctx(&state, &test_context), 1.15f, 0.0001f);
}

static void test_joseph_reward_mode_pays_tags_while_healers_heal(void) {
    printf("--- Joseph reward mode pays tags while healers heal ---\n");

    InfernoState state = make_test_state(24, 24);
    state.wave = INF_NUM_WAVES - 1;
    test_config()->joseph_reward_mode = 1;
    state.min_zuk_hp_seen = 300.0f;
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "tag_reward_coeff", 0.50f);
    state.tick_scratch.damage_dealt = 70.0f;
    state.tick_scratch.damage_zuk = 70.0f;
    state.tick_scratch.healer_tags = 1;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 220;
    state.npcs[0].max_hp = 1200;
    state.npcs[1] = make_test_npc(INF_NPC_HEALER_ZUK, 20, 48, 1);
    state.npcs[1].active = 1;
    state.npcs[1].hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[1].aggro_target = 0;

    ASSERT_FLOAT_NEAR("only tag reward is paid while a Zuk healer heals",
        inf_compute_reward_ctx(&state, &test_context), 0.50f, 0.0001f);
    ASSERT_FLOAT_NEAR("Joseph mode still tracks Zuk low watermark",
        state.min_zuk_hp_seen, 220.0f, 0.0001f);
}

static void test_zuk_healer_attack_shape_reward_applies_in_joseph_mode(void) {
    printf("--- Zuk healer attack shape reward applies in Joseph mode ---\n");

    InfernoState state = make_test_state(24, 24);
    state.wave = INF_NUM_WAVES - 1;
    test_config()->joseph_reward_mode = 1;
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "tag_reward_coeff", 0.50f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "zuk_untagged_healer_nonmagic_attack_bonus_coeff", 0.07f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "zuk_healer_mage_attack_penalty_coeff", 0.04f);
    state.tick_scratch.healer_tags = 1;
    state.tick_scratch.zuk_untagged_healer_nonmagic_attacks = 2;
    state.tick_scratch.zuk_healer_mage_attack_fires = 1;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 220;
    state.npcs[0].max_hp = 1200;
    state.npcs[1] = make_test_npc(INF_NPC_HEALER_ZUK, 20, 48, 1);
    state.npcs[1].active = 1;
    state.npcs[1].hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[1].aggro_target = 0;

    ASSERT_FLOAT_NEAR("tag reward is shaped by healer attack style",
        inf_compute_reward_ctx(&state, &test_context), 0.60f, 0.0001f);
}

static void test_offensive_prayer_reward_shapes_normal_and_joseph_mode(void) {
    printf("--- offensive prayer reward shapes normal and Joseph mode ---\n");

    InfernoState normal = make_test_state(24, 24);
    inf_put_float_ctx((EncounterState*)&normal, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&normal, (EncounterContext*)&test_context, "offensive_prayer_reward_coeff", 0.25f);
    normal.tick_scratch.damage_dealt = 40.0f;
    normal.tick_scratch.offensive_prayer_correct_damage_roll = 40.0f;

    ASSERT_FLOAT_NEAR("normal reward multiplies correct offensive prayer damage",
        inf_compute_reward_ctx(&normal, &test_context), 0.50f, 0.0001f);

    InfernoState wrong = make_test_state(24, 24);
    inf_put_float_ctx((EncounterState*)&wrong, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&wrong, (EncounterContext*)&test_context, "offensive_prayer_reward_coeff", 0.25f);
    wrong.tick_scratch.damage_dealt = 40.0f;

    ASSERT_FLOAT_NEAR("wrong offensive prayer receives base damage reward only",
        inf_compute_reward_ctx(&wrong, &test_context), 0.40f, 0.0001f);

    InfernoState zero = make_test_state(24, 24);
    inf_put_float_ctx((EncounterState*)&zero, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&zero, (EncounterContext*)&test_context, "offensive_prayer_reward_coeff", 0.25f);
    zero.tick_scratch.offensive_prayer_correct = 1;

    ASSERT_FLOAT_NEAR("correct offensive prayer without damage receives no shape",
        inf_compute_reward_ctx(&zero, &test_context), 0.0f, 0.0001f);

    InfernoState joseph = make_test_state(24, 24);
    test_config()->joseph_reward_mode = 1;
    inf_put_float_ctx((EncounterState*)&joseph, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&joseph, (EncounterContext*)&test_context, "offensive_prayer_reward_coeff", 0.25f);
    joseph.tick_scratch.damage_dealt = 40.0f;
    joseph.tick_scratch.offensive_prayer_correct_damage_roll = 40.0f;

    ASSERT_FLOAT_NEAR("Joseph reward multiplies correct offensive prayer damage",
        inf_compute_reward_ctx(&joseph, &test_context), 0.50f, 0.0001f);
}

static void init_ranged_offensive_prayer_test_state(InfernoState* state) {
    init_spell_cast_test_state(state, INF_NPC_NIBBLER);
    state->weapon_set = INF_GEAR_BP;
    state->player.autocast_enabled = 0;
    state->npcs[0].x = 13;
    state->npcs[0].y = 10;
    encounter_apply_loadout(&state->player, INF_MAX_RANGE_FAST_LOADOUT, GEAR_RANGED);
    inf_refresh_live_stats(state);
    encounter_compute_loadout_stats(INF_MAX_RANGE_FAST_LOADOUT, ATTACK_STYLE_RANGED,
        state->player.offensive_prayer, 99, FIGHT_STYLE_RAPID, 0,
        &state->loadout_stats[INF_GEAR_BP]);
    inf_refresh_current_obs_slots_ctx(state, &test_context);
}

static void test_offensive_prayer_attack_events_count_real_attacks(void) {
    printf("--- offensive prayer attack events count real attacks ---\n");

    InfernoState ranged = make_test_state(10, 10);
    init_ranged_offensive_prayer_test_state(&ranged);
    ranged.player.offensive_prayer = OFFENSIVE_PRAYER_RIGOUR;
    fire_player_action_at_slot_zero(&ranged, 0);

    ASSERT_INT_EQ("ranged attack fires", ranged.tick_scratch.player_attacked, 1);
    ASSERT_INT_EQ("ranged attack counted", ranged.total_offensive_prayer_attacks, 1);
    ASSERT_INT_EQ("ranged Rigour counted correct", ranged.total_offensive_prayer_correct, 1);
    ASSERT_INT_EQ("ranged style total counted",
        ranged.offensive_prayer_attacks_by_style[ATTACK_STYLE_RANGED], 1);
    ASSERT_INT_EQ("ranged style correct counted",
        ranged.offensive_prayer_correct_by_style[ATTACK_STYLE_RANGED], 1);

    InfernoState wrong_ranged = make_test_state(10, 10);
    init_ranged_offensive_prayer_test_state(&wrong_ranged);
    wrong_ranged.player.offensive_prayer = OFFENSIVE_PRAYER_PIETY;
    fire_player_action_at_slot_zero(&wrong_ranged, 0);

    ASSERT_INT_EQ("wrong ranged attack counted",
        wrong_ranged.total_offensive_prayer_attacks, 1);
    ASSERT_INT_EQ("ranged with Piety counted wrong",
        wrong_ranged.total_offensive_prayer_correct, 0);

    InfernoState magic = make_test_state(10, 10);
    init_spell_cast_test_state(&magic, INF_NPC_NIBBLER);
    magic.player.offensive_prayer = OFFENSIVE_PRAYER_AUGURY;
    fire_player_action_at_slot_zero(&magic, 1);

    ASSERT_INT_EQ("magic attack counted", magic.total_offensive_prayer_attacks, 1);
    ASSERT_INT_EQ("magic Augury counted correct", magic.total_offensive_prayer_correct, 1);
    ASSERT_INT_EQ("magic style total counted",
        magic.offensive_prayer_attacks_by_style[ATTACK_STYLE_MAGIC], 1);
    ASSERT_INT_EQ("magic style correct counted",
        magic.offensive_prayer_correct_by_style[ATTACK_STYLE_MAGIC], 1);

    InfernoState wrong_magic = make_test_state(10, 10);
    init_spell_cast_test_state(&wrong_magic, INF_NPC_NIBBLER);
    wrong_magic.player.offensive_prayer = OFFENSIVE_PRAYER_RIGOUR;
    fire_player_action_at_slot_zero(&wrong_magic, 2);

    ASSERT_INT_EQ("magic with Rigour counted wrong",
        wrong_magic.total_offensive_prayer_correct, 0);
}

static void test_offensive_prayer_barrage_aoe_counts_once(void) {
    printf("--- offensive prayer barrage AoE counts once ---\n");

    InfernoState state = make_test_state(10, 10);
    init_spell_cast_test_state(&state, INF_NPC_NIBBLER);
    state.player.offensive_prayer = OFFENSIVE_PRAYER_AUGURY;
    state.npcs[1] = make_test_npc(INF_NPC_NIBBLER, 17, 10, INF_NPC_STATS[INF_NPC_NIBBLER].size);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp = INF_NPC_STATS[INF_NPC_NIBBLER].hp;
    inf_refresh_current_obs_slots_ctx(&state, &test_context);

    fire_player_action_at_slot_zero(&state, 1);

    ASSERT_INT_EQ("barrage attack fires", state.tick_scratch.player_attacked, 1);
    ASSERT_INT_EQ("barrage counts one offensive prayer event",
        state.total_offensive_prayer_attacks, 1);
    ASSERT_INT_EQ("barrage with Augury counts correct",
        state.total_offensive_prayer_correct, 1);
}

static void test_offensive_prayer_no_attack_no_event(void) {
    printf("--- offensive prayer no attack no event ---\n");

    InfernoState state = make_test_state(10, 10);
    init_spell_cast_test_state(&state, INF_NPC_NIBBLER);
    state.player.offensive_prayer = OFFENSIVE_PRAYER_AUGURY;
    state.player.attack_timer = 3;

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_PRIMARY] = inf_action_target_for_npc(&state, 0);
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("cooldown prevents attack", state.tick_scratch.player_attacked, 0);
    ASSERT_INT_EQ("cooldown produces no offensive prayer event",
        state.total_offensive_prayer_attacks, 0);
    ASSERT_INT_EQ("cooldown produces no correct event",
        state.tick_scratch.offensive_prayer_correct, 0);
}

static void test_offensive_prayer_melee_maps_to_piety(void) {
    printf("--- offensive prayer melee maps to Piety ---\n");

    InfernoState state = make_test_state(10, 10);
    state.player.offensive_prayer = OFFENSIVE_PRAYER_PIETY;
    inf_record_offensive_prayer_attack(&state, ATTACK_STYLE_MELEE, 7.0f);

    ASSERT_INT_EQ("melee requires Piety",
        encounter_offensive_prayer_for_style(ATTACK_STYLE_MELEE),
        OFFENSIVE_PRAYER_PIETY);
    ASSERT_INT_EQ("melee Piety counted correct",
        state.total_offensive_prayer_correct, 1);
    ASSERT_INT_EQ("melee style counted",
        state.offensive_prayer_attacks_by_style[ATTACK_STYLE_MELEE], 1);
    ASSERT_FLOAT_NEAR("melee correct prayer records damage roll",
        state.tick_scratch.offensive_prayer_correct_damage_roll, 7.0f, 1e-6f);
}

static void test_player_reward_damage_uses_xp_drop_tick(void) {
    printf("--- player reward damage uses XP-drop tick ---\n");

    InfernoState state = make_test_state(10, 10);
    state.npcs[0] = make_test_npc(
        INF_NPC_RANGER, 16, 10, INF_NPC_STATS[INF_NPC_RANGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 15;
    state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_RANGER].hp;

    float reward_damage = inf_record_player_reward_damage(&state, 0, 50);

    ASSERT_FLOAT_NEAR("reward damage caps to current hp",
        reward_damage, 15.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("damage dealt stat records on fire tick",
        state.tick_scratch.damage_dealt, 15.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("set damage stat records on fire tick",
        state.tick_scratch.damage_set, 15.0f, 1e-6f);
    ASSERT_INT_EQ("reward damage does not apply hp before hitsplat",
        state.npcs[0].hp, 15);
}

static void test_idle_diagnostics_count_missed_attack_opportunities(void) {
    printf("--- idle diagnostics count missed attack opportunities ---\n");

    InfernoState state = make_test_state(10, 10);
    state.weapon_set = INF_GEAR_BP;
    encounter_apply_loadout(&state.player, INF_MAX_RANGE_FAST_LOADOUT, GEAR_RANGED);
    inf_refresh_live_stats(&state);
    state.npcs[0] = make_test_npc(
        INF_NPC_RANGER, 14, 10, INF_NPC_STATS[INF_NPC_RANGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = INF_NPC_STATS[INF_NPC_RANGER].hp;
    state.npcs[0].attack_timer = 5;


    inf_record_idle_diagnostics(
        &state, 1, 1, INF_IDLE_PHASE_SET, 1, 1, 1);

    ASSERT_INT_EQ("attack ready no attack total",
        state.total_attack_ready_no_attack_ticks, 1);
    ASSERT_INT_EQ("target available no attack total",
        state.total_target_available_no_attack_ticks, 1);
    ASSERT_INT_EQ("safe opportunity missed total",
        state.total_safe_attack_opportunity_missed_ticks, 1);
    ASSERT_INT_EQ("progressless total",
        state.total_progressless_ticks, 1);
    ASSERT_INT_EQ("set phase attack ready counter",
        state.attack_ready_no_attack_ticks_by_phase[INF_IDLE_PHASE_SET], 1);
    ASSERT_INT_EQ("set phase safe opportunity counter",
        state.safe_attack_opportunity_missed_ticks_by_phase[INF_IDLE_PHASE_SET], 1);
    ASSERT_INT_EQ("set phase progressless counter",
        state.progressless_ticks_by_phase[INF_IDLE_PHASE_SET], 1);
}

static void test_idle_diagnostics_phase_split(void) {
    printf("--- idle diagnostics phase split ---\n");

    InfernoState set = make_test_state(10, 10);
    set.wave = 20;
    ASSERT_INT_EQ("ordinary waves use set phase",
        inf_idle_diagnostic_phase_from_summary(
            &set, inf_idle_diagnostic_summary(&set)), INF_IDLE_PHASE_SET);

    InfernoState jad = make_test_state(10, 10);
    jad.wave = 66;
    jad.npcs[0] = make_test_npc(
        INF_NPC_JAD, 14, 10, INF_NPC_STATS[INF_NPC_JAD].size);
    jad.npcs[0].active = 1;
    jad.npcs[0].hp = INF_NPC_STATS[INF_NPC_JAD].hp;
    ASSERT_INT_EQ("non-final live jad uses jad phase",
        inf_idle_diagnostic_phase_from_summary(
            &jad, inf_idle_diagnostic_summary(&jad)), INF_IDLE_PHASE_JAD);

    InfernoState zuk = make_test_state(25, 42);
    zuk.wave = INF_WAVE_ZUK;
    zuk.tick_at_all_zuk_healers_dead = -1;
    ASSERT_INT_EQ("final wave before jad uses zuk pre-jad phase",
        inf_idle_diagnostic_phase_from_summary(
            &zuk, inf_idle_diagnostic_summary(&zuk)), INF_IDLE_PHASE_ZUK_PRE_JAD);

    zuk.npcs[0] = make_test_npc(
        INF_NPC_JAD, 24, 44, INF_NPC_STATS[INF_NPC_JAD].size);
    zuk.npcs[0].active = 1;
    zuk.npcs[0].hp = INF_NPC_STATS[INF_NPC_JAD].hp;
    ASSERT_INT_EQ("final wave live jad uses zuk jad phase",
        inf_idle_diagnostic_phase_from_summary(
            &zuk, inf_idle_diagnostic_summary(&zuk)), INF_IDLE_PHASE_ZUK_JAD);

    zuk.npcs[1] = make_test_npc(
        INF_NPC_HEALER_ZUK, 22, 44, INF_NPC_STATS[INF_NPC_HEALER_ZUK].size);
    zuk.npcs[1].active = 1;
    zuk.npcs[1].hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    ASSERT_INT_EQ("live zuk healer uses zuk healer phase",
        inf_idle_diagnostic_phase_from_summary(
            &zuk, inf_idle_diagnostic_summary(&zuk)), INF_IDLE_PHASE_ZUK_HEALERS);

    zuk.npcs[0].active = 0;
    zuk.npcs[1].active = 0;
    zuk.tick_at_all_zuk_healers_dead = 500;
    ASSERT_INT_EQ("after healers dead uses post-healer phase",
        inf_idle_diagnostic_phase_from_summary(
            &zuk, inf_idle_diagnostic_summary(&zuk)), INF_IDLE_PHASE_ZUK_POST_HEALERS);
}

static void test_joseph_reward_mode_damps_healed_zuk_damage(void) {
    printf("--- Joseph reward mode damps healed Zuk damage ---\n");

    InfernoState state = make_test_state(24, 24);
    state.wave = INF_NUM_WAVES - 1;
    test_config()->joseph_reward_mode = 1;
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    state.tick_scratch.damage_dealt = 100.0f;
    state.tick_scratch.damage_zuk = 100.0f;
    state.total_hp_restored_zuk = 1200.0f;
    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 22, 50, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 200;
    state.npcs[0].max_hp = 1200;

    ASSERT_FLOAT_NEAR("Zuk damage is downweighted by prior healing",
        inf_compute_reward_ctx(&state, &test_context), 0.20f, 0.0001f);
}

static void test_jad_damage_reward_pauses_while_jad_healers_heal(void) {
    printf("--- jad damage reward pauses while jad healers heal ---\n");

    InfernoState state = make_test_state(24, 24);

    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "shield_penalty_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "tag_reward_coeff", 0.25f);
    state.wave = 66;
    state.tick_scratch.damage_dealt = 40.0f;
    state.tick_scratch.damage_jad = 40.0f;
    state.npcs[0] = make_test_npc(INF_NPC_JAD, 24, 32, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 200;
    state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_JAD].hp;
    state.npcs[1] = make_test_npc(INF_NPC_HEALER_JAD, 20, 34, 1);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp =
        INF_NPC_STATS[INF_NPC_HEALER_JAD].hp;
    inf_npc_healer(&state.npcs[1])->owner_idx = 0;
    state.npcs[1].aggro_target = 0;

    ASSERT_FLOAT_NEAR("jad damage pays nothing while a healer heals jad",
        inf_compute_reward_ctx(&state, &test_context), 0.0f, 0.0001f);

    state.npcs[1].aggro_target = -1;

    ASSERT_FLOAT_NEAR("jad damage resumes after the healer is tagged",
        inf_compute_reward_ctx(&state, &test_context), 0.40f, 0.0001f);
}

static void test_jad_healer_damage_never_gets_damage_reward(void) {
    printf("--- jad healer damage never gets damage reward ---\n");

    InfernoState state = make_test_state(24, 24);

    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "damage_reward_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "shield_penalty_coeff", 0.01f);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "tag_reward_coeff", 0.25f);
    state.wave = 66;
    state.tick_scratch.damage_dealt = 40.0f;
    state.tick_scratch.damage_jad_healers = 40.0f;
    state.npcs[0] = make_test_npc(INF_NPC_HEALER_JAD, 20, 34, 1);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp =
        INF_NPC_STATS[INF_NPC_HEALER_JAD].hp;
    state.npcs[0].aggro_target = -1;

    ASSERT_FLOAT_NEAR("jad healer damage is not rewarded",
        inf_compute_reward_ctx(&state, &test_context), 0.0f, 0.0001f);
}

static void test_shield_tag_reward_excludes_zuk(void) {
    printf("--- shield tag reward excludes zuk ---\n");

    InfernoState state = make_test_state(24, 24);

    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "shield_tag_reward_coeff", 0.20f);
    state.npcs[0] = make_test_npc(INF_NPC_ZUK_SHIELD, 23, 44, 1);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 100;

    state.npcs[1] = make_test_npc(INF_NPC_MAGER, 20, 44, 1);
    state.npcs[1].active = 1;
    state.npcs[1].hp = 100;
    state.npcs[1].aggro_target = 0;

    state.npcs[2] = make_test_npc(INF_NPC_RANGER, 22, 44, 1);
    state.npcs[2].active = 1;
    state.npcs[2].hp = 100;
    state.npcs[2].aggro_target = 0;

    state.npcs[3] = make_test_npc(INF_NPC_JAD, 24, 44, 1);
    state.npcs[3].active = 1;
    state.npcs[3].hp = 100;
    state.npcs[3].aggro_target = 0;

    state.npcs[4] = make_test_npc(INF_NPC_ZUK, 26, 44, 5);
    state.npcs[4].active = 1;
    state.npcs[4].hp = 1000;
    state.npcs[4].aggro_target = 0;

    state.npcs[5] = make_test_npc(INF_NPC_HEALER_ZUK, 28, 44, 1);
    state.npcs[5].active = 1;
    state.npcs[5].hp = 100;
    state.npcs[5].aggro_target = 0;

    ASSERT_INT_EQ("mager can be tagged off shield",
        inf_is_shield_taggable_slot(&state, 1), 1);
    ASSERT_INT_EQ("ranger can be tagged off shield",
        inf_is_shield_taggable_slot(&state, 2), 1);
    ASSERT_INT_EQ("jad can be tagged off shield",
        inf_is_shield_taggable_slot(&state, 3), 1);
    ASSERT_INT_EQ("zuk cannot be tagged off shield",
        inf_is_shield_taggable_slot(&state, 4), 0);
    ASSERT_INT_EQ("zuk healer cannot be tagged off shield",
        inf_is_shield_taggable_slot(&state, 5), 0);
    ASSERT_INT_EQ("shield cannot be tagged off itself",
        inf_is_shield_taggable_slot(&state, 0), 0);

    state.tick_scratch.shield_tags = 3;
    ASSERT_FLOAT_NEAR("shield tag reward pays per valid shield tag",
        inf_compute_reward_ctx(&state, &test_context), 0.60f, 0.0001f);
}

static void test_inferno_reset_supplies_match_current_inventory(void) {
    printf("--- inferno reset supplies match current inventory ---\n");

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;
    InfSupplyDoses full = inf_full_starting_supplies();

    reset_inferno_at_public_wave(raw_state, 1, 1.0f);

    assert_supply_doses("wave 1", &state->player, full);
    ASSERT_INT_EQ("brew cells match brew counter",
        test_cell_doses_of_kind(state, OSRS_CONSUMABLE_BREW),
        state->player.brew_doses);
    ASSERT_INT_EQ("restore cells match restore counter",
        test_cell_doses_of_kind(state, OSRS_CONSUMABLE_SUPER_RESTORE),
        state->player.restore_doses);
    ASSERT_INT_EQ("bastion cells match bastion counter",
        test_cell_doses_of_kind(state, OSRS_CONSUMABLE_BASTION),
        state->player.bastion_doses);
    ASSERT_INT_EQ("stamina cells match stamina counter",
        test_cell_doses_of_kind(state, OSRS_CONSUMABLE_STAMINA),
        state->player.stamina_doses);

    inf_destroy(raw_state);
}

static int test_inventory_potion_vials(int doses) {
    return (doses + 3) / 4;
}

static int test_player_slot_inventory_contains(
    const Player* p,
    int gear_slot,
    uint8_t item
) {
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
        const OsrsItemContentMetadata* metadata =
            osrs_inventory_cell_metadata(&p->inventory_cells[cell]);
        if (metadata->gear_slot == gear_slot && metadata->item_idx == item)
            return 1;
    }
    return p->equipped[gear_slot] == item;
}

static void test_inferno_reset_inventory_leaves_one_empty_slot(void) {
    printf("--- inferno reset inventory leaves one empty slot ---\n");

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;

    reset_inferno_at_public_wave(raw_state, 1, 1.0f);

    ASSERT_INT_EQ("full kit bastion doses", state->player.bastion_doses, 16);
    ASSERT_INT_EQ("full kit bastion vials",
        test_inventory_potion_vials(state->player.bastion_doses), 4);
    ASSERT_INT_EQ("full kit occupied inventory cells",
        test_occupied_inventory_cells(state), 27);

    inf_destroy(raw_state);
}

static void test_inferno_max_profile_reset_uses_existing_gear(void) {
    printf("--- inferno max profile reset uses existing gear ---\n");

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;

    reset_inferno_at_public_wave(raw_state, 1, 1.0f);

    ASSERT_INT_EQ("default loadout profile mode",
        test_config()->loadout_profile_mode, INF_LOADOUT_PROFILE_MODE_MAX_ONLY);
    ASSERT_INT_EQ("default active profile",
        state->active_loadout_profile, INF_LOADOUT_PROFILE_MAX);
    ASSERT_INT_EQ("max mage weapon",
        state->player.equipped[GEAR_SLOT_WEAPON], ITEM_KODAI_WAND);
    ASSERT_INT_EQ("max mage shield",
        state->player.equipped[GEAR_SLOT_SHIELD], ITEM_ELYSIAN_SPIRIT_SHIELD);
    ASSERT_INT_EQ("max long-range weapon in inventory",
        test_player_slot_inventory_contains(
            &state->player, GEAR_SLOT_WEAPON, ITEM_TWISTED_BOW), 1);
    ASSERT_INT_EQ("max fast-range weapon in inventory",
        test_player_slot_inventory_contains(
            &state->player, GEAR_SLOT_WEAPON, ITEM_TOXIC_BLOWPIPE), 1);

    inf_destroy(raw_state);
}

static void test_inferno_budget_profile_reset_uses_budget_gear(void) {
    printf("--- inferno budget profile reset uses budget gear ---\n");

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;

    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "loadout_profile_mode", INF_LOADOUT_PROFILE_MODE_BUDGET_ONLY);
    reset_inferno_at_public_wave(raw_state, 1, 1.0f);

    ASSERT_INT_EQ("budget active profile",
        state->active_loadout_profile, INF_LOADOUT_PROFILE_BUDGET);
    ASSERT_INT_EQ("budget mage head",
        state->player.equipped[GEAR_SLOT_HEAD], ITEM_CRYSTAL_HELM);
    ASSERT_INT_EQ("budget mage cape",
        state->player.equipped[GEAR_SLOT_CAPE], ITEM_DIZANAS_QUIVER);
    ASSERT_INT_EQ("budget mage neck",
        state->player.equipped[GEAR_SLOT_NECK], ITEM_OCCULT_NECKLACE);
    ASSERT_INT_EQ("budget mage ammo",
        state->player.equipped[GEAR_SLOT_AMMO], ITEM_GOD_BLESSING);
    ASSERT_INT_EQ("budget mage weapon",
        state->player.equipped[GEAR_SLOT_WEAPON], ITEM_DRAGON_HUNTER_WAND);
    ASSERT_INT_EQ("budget mage shield",
        state->player.equipped[GEAR_SLOT_SHIELD], ITEM_CRYSTAL_SHIELD);
    ASSERT_INT_EQ("budget mage body",
        state->player.equipped[GEAR_SLOT_BODY], ITEM_AHRIMS_ROBETOP);
    ASSERT_INT_EQ("budget mage legs",
        state->player.equipped[GEAR_SLOT_LEGS], ITEM_AHRIMS_ROBESKIRT);
    ASSERT_INT_EQ("budget mage hands",
        state->player.equipped[GEAR_SLOT_HANDS], ITEM_CONFLICTION_GAUNTLETS);
    ASSERT_INT_EQ("budget mage feet",
        state->player.equipped[GEAR_SLOT_FEET], ITEM_ECHO_BOOTS);
    ASSERT_INT_EQ("budget mage ring",
        state->player.equipped[GEAR_SLOT_RING], ITEM_VENATOR_RING);
    ASSERT_INT_EQ("budget bowfa available",
        test_player_slot_inventory_contains(
            &state->player, GEAR_SLOT_WEAPON, ITEM_BOW_OF_FAERDHINEN), 1);
    ASSERT_INT_EQ("budget blowpipe available",
        test_player_slot_inventory_contains(
            &state->player, GEAR_SLOT_WEAPON, ITEM_TOXIC_BLOWPIPE), 1);
    ASSERT_INT_EQ("budget range body available",
        test_player_slot_inventory_contains(
            &state->player, GEAR_SLOT_BODY, ITEM_CRYSTAL_BODY), 1);
    ASSERT_INT_EQ("budget range legs available",
        test_player_slot_inventory_contains(
            &state->player, GEAR_SLOT_LEGS, ITEM_CRYSTAL_LEGS), 1);
    ASSERT_INT_EQ("budget inventory leaves one empty cell",
        test_occupied_inventory_cells(state), 27);

    inf_destroy(raw_state);
}

static void test_inferno_mixed_profile_sampling_respects_fraction(void) {
    printf("--- inferno mixed profile sampling respects fraction ---\n");

    EncounterState* raw_a = inf_create();
    EncounterState* raw_b = inf_create();
    EncounterState* raw_c = inf_create();
    InfernoState* a = (InfernoState*)raw_a;
    InfernoState* b = (InfernoState*)raw_b;
    InfernoState* c = (InfernoState*)raw_c;

    inf_put_int_ctx(raw_a, (EncounterContext*)&test_context, "loadout_profile_mode", INF_LOADOUT_PROFILE_MODE_MIXED_MAX_BUDGET);
    inf_put_float_ctx(raw_a, (EncounterContext*)&test_context, "budget_loadout_fraction", 0.0f);
    inf_reset_ctx(raw_a, (EncounterContext*)&test_context, 456u);
    ASSERT_INT_EQ("zero fraction samples max",
        a->active_loadout_profile, INF_LOADOUT_PROFILE_MAX);

    inf_put_int_ctx(raw_b, (EncounterContext*)&test_context, "loadout_profile_mode", INF_LOADOUT_PROFILE_MODE_MIXED_MAX_BUDGET);
    inf_put_float_ctx(raw_b, (EncounterContext*)&test_context, "budget_loadout_fraction", 1.0f);
    inf_reset_ctx(raw_b, (EncounterContext*)&test_context, 456u);
    ASSERT_INT_EQ("one fraction samples budget",
        b->active_loadout_profile, INF_LOADOUT_PROFILE_BUDGET);

    inf_put_int_ctx(raw_c, (EncounterContext*)&test_context, "loadout_profile_mode", INF_LOADOUT_PROFILE_MODE_MIXED_MAX_BUDGET);
    inf_put_float_ctx(raw_c, (EncounterContext*)&test_context, "budget_loadout_fraction", 1.0f);
    inf_reset_ctx(raw_c, (EncounterContext*)&test_context, 456u);
    ASSERT_INT_EQ("same seed and fraction are deterministic",
        c->active_loadout_profile, b->active_loadout_profile);

    inf_destroy(raw_a);
    inf_destroy(raw_b);
    inf_destroy(raw_c);
}

static void test_inferno_equip_actions_move_cells_and_sync_weapon_set(void) {
    printf("--- inferno equip actions move cells and sync weapon set ---\n");

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;
    int actions[INF_NUM_ACTION_HEADS];

    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "loadout_profile_mode", INF_LOADOUT_PROFILE_MODE_BUDGET_ONLY);
    inf_reset_ctx(raw_state, (EncounterContext*)&test_context, 789u);

    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_EQUIP_SLOT(GEAR_SLOT_WEAPON)] =
        test_cell_holding_item(state, ITEM_BOW_OF_FAERDHINEN) + 1;
    actions[INF_HEAD_EQUIP_SLOT(GEAR_SLOT_BODY)] =
        test_cell_holding_item(state, ITEM_CRYSTAL_BODY) + 1;
    inf_tick_player_ctx(state, &test_context, actions, 1);
    ASSERT_INT_EQ("equipping bowfa syncs long-range weapon set",
        state->weapon_set, INF_GEAR_LONG_RANGE);
    ASSERT_INT_EQ("long range budget weapon",
        state->player.equipped[GEAR_SLOT_WEAPON], ITEM_BOW_OF_FAERDHINEN);
    ASSERT_INT_EQ("two-handed bow displaces budget shield",
        state->player.equipped[GEAR_SLOT_SHIELD], ITEM_NONE);
    ASSERT_INT_EQ("body head equips crystal body in same tick",
        state->player.equipped[GEAR_SLOT_BODY], ITEM_CRYSTAL_BODY);
    ASSERT_INT_EQ("equipped bowfa left its cell",
        test_cell_holding_item(state, ITEM_BOW_OF_FAERDHINEN), -1);
    ASSERT_INT_EQ("displaced wand returns to a cell",
        test_cell_holding_item(state, ITEM_DRAGON_HUNTER_WAND) >= 0, 1);
    ASSERT_INT_EQ("displaced shield returns to a cell",
        test_cell_holding_item(state, ITEM_CRYSTAL_SHIELD) >= 0, 1);
    ASSERT_INT_EQ("displaced robe top returns to a cell",
        test_cell_holding_item(state, ITEM_AHRIMS_ROBETOP) >= 0, 1);

    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_EQUIP_SLOT(GEAR_SLOT_WEAPON)] =
        test_cell_holding_item(state, ITEM_TOXIC_BLOWPIPE) + 1;
    inf_tick_player_ctx(state, &test_context, actions, 1);
    ASSERT_INT_EQ("equipping blowpipe syncs fast-range weapon set",
        state->weapon_set, INF_GEAR_BP);
    ASSERT_INT_EQ("fast range budget weapon",
        state->player.equipped[GEAR_SLOT_WEAPON], ITEM_TOXIC_BLOWPIPE);
    ASSERT_INT_EQ("displaced bowfa returns to a cell",
        test_cell_holding_item(state, ITEM_BOW_OF_FAERDHINEN) >= 0, 1);

    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_EQUIP_SLOT(GEAR_SLOT_WEAPON)] =
        test_cell_holding_item(state, ITEM_DRAGON_HUNTER_WAND) + 1;
    inf_tick_player_ctx(state, &test_context, actions, 1);
    ASSERT_INT_EQ("equipping wand syncs mage weapon set",
        state->weapon_set, INF_GEAR_MAGE);
    ASSERT_INT_EQ("mage budget weapon",
        state->player.equipped[GEAR_SLOT_WEAPON], ITEM_DRAGON_HUNTER_WAND);

    inf_destroy(raw_state);
}

static void test_inferno_gear_switch_cancels_entity_interaction(void) {
    printf("--- inferno gear switch cancels entity interaction ---\n");

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;
    int actions[INF_NUM_ACTION_HEADS];

    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "loadout_profile_mode", INF_LOADOUT_PROFILE_MODE_BUDGET_ONLY);
    inf_reset_ctx(raw_state, (EncounterContext*)&test_context, 789u);

    int npc_slot = -1;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (state->npcs[i].active) { npc_slot = i; break; }
    }
    ASSERT_INT_EQ("reset spawns an npc to target", npc_slot >= 0, 1);

    int body_cell = test_cell_holding_item(state, ITEM_CRYSTAL_BODY);
    ASSERT_INT_EQ("budget reset carries a crystal body", body_cell >= 0, 1);

    osrs_interaction_set(&state->interaction, npc_slot);
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_EQUIP_SLOT(GEAR_SLOT_WEAPON)] = body_cell + 1;
    inf_tick_player_ctx(state, &test_context, actions, 1);
    ASSERT_INT_EQ("a click that equips nothing keeps interaction",
        osrs_interaction_active(&state->interaction), 1);

    osrs_interaction_set(&state->interaction, npc_slot);
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_EQUIP_SLOT(GEAR_SLOT_WEAPON)] =
        test_cell_holding_item(state, ITEM_BOW_OF_FAERDHINEN) + 1;
    inf_tick_player_ctx(state, &test_context, actions, 1);
    ASSERT_INT_EQ("gear switch equips the clicked weapon",
        state->player.equipped[GEAR_SLOT_WEAPON], ITEM_BOW_OF_FAERDHINEN);
    ASSERT_INT_EQ("gear switch clears interaction",
        osrs_interaction_active(&state->interaction), 0);

    inf_destroy(raw_state);
}

static void test_inferno_reset_preserves_reward_config(void) {
    printf("--- inferno reset preserves reward config ---\n");

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;

    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "supply_milestone_brew_reward_coeff", 0.001f);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "supply_milestone_restore_reward_coeff", 0.002f);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "offensive_prayer_reward_coeff", 0.009f);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "post_healer_zuk_damage_coeff", 0.003f);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "zuk_healer_phase_hp_delta_coeff", 0.004f);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "zuk_untagged_healer_tick_penalty_coeff", 0.005f);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "zuk_untagged_healer_target_bonus_coeff", 0.006f);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "zuk_untagged_healer_nonmagic_attack_bonus_coeff", 0.007f);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "zuk_healer_mage_attack_penalty_coeff", 0.008f);
    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "zuk_safe_untagged_healer_target_mask", 1);
    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "zuk_force_safe_untagged_healer_target_mask", 1);
    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "zuk_healer_reward_mode", 1);
    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "joseph_reward_mode", 1);
    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "terminal_penalty_enabled", 1);
    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "loadout_profile_mode", INF_LOADOUT_PROFILE_MODE_BUDGET_ONLY);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "budget_loadout_fraction", 1.0f);
    inf_reset_ctx(raw_state, (EncounterContext*)&test_context, 123u);

    ASSERT_FLOAT_NEAR("supply milestone brew reward coefficient",
        test_config()->supply_milestone_brew_reward_coeff, 0.001f, 1e-6f);
    ASSERT_FLOAT_NEAR("supply milestone restore reward coefficient",
        test_config()->supply_milestone_restore_reward_coeff, 0.002f, 1e-6f);
    ASSERT_FLOAT_NEAR("offensive prayer reward coefficient",
        test_config()->offensive_prayer_reward_coeff, 0.009f, 1e-6f);
    ASSERT_FLOAT_NEAR("post-healer Zuk damage coefficient",
        test_config()->post_healer_zuk_damage_coeff, 0.003f, 1e-6f);
    ASSERT_FLOAT_NEAR("Zuk healer-phase HP delta coefficient",
        test_config()->zuk_healer_phase_hp_delta_coeff, 0.004f, 1e-6f);
    ASSERT_FLOAT_NEAR("Zuk untagged healer tick penalty coefficient",
        test_config()->zuk_untagged_healer_tick_penalty_coeff, 0.005f, 1e-6f);
    ASSERT_FLOAT_NEAR("Zuk untagged healer target bonus coefficient",
        test_config()->zuk_untagged_healer_target_bonus_coeff, 0.006f, 1e-6f);
    ASSERT_FLOAT_NEAR("Zuk untagged healer non-magic attack bonus coefficient",
        test_config()->zuk_untagged_healer_nonmagic_attack_bonus_coeff, 0.007f, 1e-6f);
    ASSERT_FLOAT_NEAR("Zuk healer mage attack penalty coefficient",
        test_config()->zuk_healer_mage_attack_penalty_coeff, 0.008f, 1e-6f);
    ASSERT_INT_EQ("safe untagged healer target mask",
        test_config()->zuk_safe_untagged_healer_target_mask, 1);
    ASSERT_INT_EQ("force safe untagged healer target mask",
        test_config()->zuk_force_safe_untagged_healer_target_mask, 1);
    ASSERT_INT_EQ("Zuk healer reward mode", test_config()->zuk_healer_reward_mode, 1);
    ASSERT_INT_EQ("Joseph reward mode", test_config()->joseph_reward_mode, 1);
    ASSERT_INT_EQ("terminal penalty enabled", test_config()->terminal_penalty_enabled, 1);
    ASSERT_INT_EQ("loadout profile mode preserved",
        test_config()->loadout_profile_mode, INF_LOADOUT_PROFILE_MODE_BUDGET_ONLY);
    ASSERT_FLOAT_NEAR("budget loadout fraction preserved",
        test_config()->budget_loadout_fraction, 1.0f, 1e-6f);
    ASSERT_INT_EQ("budget-only reset selects budget profile",
        state->active_loadout_profile, INF_LOADOUT_PROFILE_BUDGET);

    inf_destroy(raw_state);
}

static void test_supply_milestone_reward_defaults_off(void) {
    printf("--- supply milestone reward defaults off ---\n");

    InfernoState state = make_test_state(24, 24);
    test_config()->late_start_supply_profile_scale = 1.0f;
    state.player.brew_doses = 24;
    state.player.restore_doses = 36;

    ASSERT_FLOAT_NEAR("default supply milestone reward",
        test_supply_milestone_surplus_reward(&state, 64), 0.0f, 0.0001f);
    ASSERT_INT_EQ("default supply milestone does not consume anchor",
        (int)state.supply_milestone_rewarded_mask, 0);
}

static void test_supply_milestone_reward_pays_surplus_at_anchor_once(void) {
    printf("--- supply milestone reward pays surplus at anchor once ---\n");

    InfernoState state = make_test_state(24, 24);
    test_config()->late_start_supply_profile_scale = 1.0f;
    test_config()->supply_milestone_brew_reward_coeff = 0.24f;
    test_config()->supply_milestone_restore_reward_coeff = 0.20f;
    state.player.brew_doses = 18;
    state.player.restore_doses = 27;

    ASSERT_FLOAT_NEAR("wave 64 supply surplus reward",
        test_supply_milestone_surplus_reward(&state, 64), 0.09f, 0.0001f);
    ASSERT_FLOAT_NEAR("wave 64 supply surplus pays once",
        test_supply_milestone_surplus_reward(&state, 64), 0.0f, 0.0001f);
}

static void test_supply_milestone_reward_never_penalizes_shortage(void) {
    printf("--- supply milestone reward never penalizes shortage ---\n");

    InfernoState state = make_test_state(24, 24);
    test_config()->late_start_supply_profile_scale = 1.0f;
    test_config()->supply_milestone_brew_reward_coeff = 0.24f;
    test_config()->supply_milestone_restore_reward_coeff = 0.20f;

    ASSERT_FLOAT_NEAR("shortage reward is zero",
        test_supply_milestone_surplus_reward(&state, 64), 0.0f, 0.0001f);

    InfernoState non_anchor = make_test_state(24, 24);
    test_config()->late_start_supply_profile_scale = 1.0f;
    test_config()->supply_milestone_brew_reward_coeff = 0.24f;
    non_anchor.player.brew_doses = 24;
    non_anchor.player.restore_doses = 36;
    ASSERT_FLOAT_NEAR("non-anchor reward is zero",
        test_supply_milestone_surplus_reward(&non_anchor, 63), 0.0f, 0.0001f);
}

static void test_late_start_supply_profile_anchor_waves(void) {
    printf("--- inferno late-start supply profile anchor waves ---\n");

    struct {
        int public_wave;
        float brew_fraction;
        float restore_fraction;
        float bastion_fraction;
        float stamina_fraction;
    } anchors[] = {
        { 20, 1.0000f, 0.9500f, 1.0000f, 1.0000f },
        { 40, 0.9167f, 0.8750f, 1.0000f, 1.0000f },
        { 61, 0.8333f, 0.7500f, 1.0000f, 1.0000f },
        { 64, 0.5833f, 0.5000f, 0.7500f, 1.0000f },
        { 68, 0.5833f, 0.4250f, 0.6250f, 1.0000f },
        { 69, 0.5000f, 0.3000f, 0.3750f, 1.0000f },
        { 70, 0.4375f, 0.2250f, 0.3750f, 1.0000f },
        { 71, 0.3750f, 0.1500f, 0.3750f, 1.0000f },
    };

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;
    InfSupplyDoses full = inf_full_starting_supplies();

    for (int i = 0; i < (int)(sizeof(anchors) / sizeof(anchors[0])); i++) {
        reset_inferno_at_public_wave(raw_state, anchors[i].public_wave, 1.0f);
        InfSupplyDoses expected = {
            .brew_doses = test_profiled_supply_count(full.brew_doses,
                anchors[i].brew_fraction, 1.0f),
            .restore_doses = test_profiled_supply_count(full.restore_doses,
                anchors[i].restore_fraction, 1.0f),
            .bastion_doses = test_profiled_supply_count(full.bastion_doses,
                anchors[i].bastion_fraction, 1.0f),
            .stamina_doses = test_profiled_supply_count(full.stamina_doses,
                anchors[i].stamina_fraction, 1.0f),
        };
        char label[64];
        snprintf(label, sizeof(label), "wave %d", anchors[i].public_wave);
        assert_supply_doses(label, &state->player, expected);
    }

    inf_destroy(raw_state);
}

static void test_late_start_supply_profile_interpolation_and_scale(void) {
    printf("--- inferno late-start supply profile interpolation and scale ---\n");

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;
    InfSupplyDoses full = inf_full_starting_supplies();
    float t = 1.0f / 3.0f;
    float brew_fraction = 0.8333f + (0.5833f - 0.8333f) * t;
    float restore_fraction = 0.7500f + (0.5000f - 0.7500f) * t;
    float bastion_fraction = 1.0000f + (0.7500f - 1.0000f) * t;

    reset_inferno_at_public_wave(raw_state, 62, 1.0f);
    InfSupplyDoses interpolated = {
        .brew_doses = test_profiled_supply_count(full.brew_doses, brew_fraction, 1.0f),
        .restore_doses = test_profiled_supply_count(full.restore_doses, restore_fraction, 1.0f),
        .bastion_doses = test_profiled_supply_count(full.bastion_doses, bastion_fraction, 1.0f),
        .stamina_doses = full.stamina_doses,
    };
    assert_supply_doses("wave 62", &state->player, interpolated);

    reset_inferno_at_public_wave(raw_state, 69, 0.0f);
    assert_supply_doses("wave 69 scale 0", &state->player, full);

    reset_inferno_at_public_wave(raw_state, 69, 0.5f);
    InfSupplyDoses half_scale = {
        .brew_doses = test_profiled_supply_count(full.brew_doses, 0.5000f, 0.5f),
        .restore_doses = test_profiled_supply_count(full.restore_doses, 0.3000f, 0.5f),
        .bastion_doses = test_profiled_supply_count(full.bastion_doses, 0.3750f, 0.5f),
        .stamina_doses = test_profiled_supply_count(full.stamina_doses, 1.0000f, 0.5f),
    };
    assert_supply_doses("wave 69 scale 0.5", &state->player, half_scale);

    inf_destroy(raw_state);
}

static void test_curriculum_supply_no_brew_is_curriculum_only(void) {
    printf("--- curriculum no-brew starts are curriculum-only ---\n");

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;

    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_no_brew_mode", INF_CURRICULUM_SUPPLY_MODE_ALL);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_no_brew_frac", 1.0f);
    reset_inferno_at_public_wave(raw_state, 71, 1.0f);
    ASSERT_INT_EQ("normal start ignores curriculum no-brew",
        state->player.brew_doses, 9);

    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_agent", 1);
    reset_inferno_at_public_wave(raw_state, 71, 1.0f);
    ASSERT_INT_EQ("curriculum start applies no-brew",
        state->player.brew_doses, 0);
    ASSERT_INT_EQ("curriculum no-brew leaves restores alone",
        state->player.restore_doses, 5);

    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_agent", 0);
    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_no_brew_mode", INF_CURRICULUM_SUPPLY_MODE_OFF);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_no_brew_frac", 0.0f);
    inf_destroy(raw_state);
}

static void test_curriculum_supply_modes_gate_zuk_and_pre_zuk(void) {
    printf("--- curriculum supply modes gate Zuk and pre-Zuk starts ---\n");

    ASSERT_INT_EQ("off mode does not apply",
        inf_curriculum_supply_mode_applies(INF_CURRICULUM_SUPPLY_MODE_OFF, 69), 0);
    ASSERT_INT_EQ("all mode applies to Zuk",
        inf_curriculum_supply_mode_applies(INF_CURRICULUM_SUPPLY_MODE_ALL, 69), 1);
    ASSERT_INT_EQ("Zuk mode applies to wave 69",
        inf_curriculum_supply_mode_applies(INF_CURRICULUM_SUPPLY_MODE_ZUK, 69), 1);
    ASSERT_INT_EQ("Zuk mode applies to wave 71",
        inf_curriculum_supply_mode_applies(INF_CURRICULUM_SUPPLY_MODE_ZUK, 71), 1);
    ASSERT_INT_EQ("Zuk mode skips wave 54",
        inf_curriculum_supply_mode_applies(INF_CURRICULUM_SUPPLY_MODE_ZUK, 54), 0);
    ASSERT_INT_EQ("pre-Zuk mode applies to wave 54",
        inf_curriculum_supply_mode_applies(INF_CURRICULUM_SUPPLY_MODE_PRE_ZUK, 54), 1);
    ASSERT_INT_EQ("pre-Zuk mode skips wave 69",
        inf_curriculum_supply_mode_applies(INF_CURRICULUM_SUPPLY_MODE_PRE_ZUK, 69), 0);
}

static void test_curriculum_supply_jitter_clamps_to_inventory_bounds(void) {
    printf("--- curriculum supply jitter clamps to inventory bounds ---\n");

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;

    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_agent", 1);
    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_supply_jitter_mode", INF_CURRICULUM_SUPPLY_MODE_ALL);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_supply_shared_jitter", 1.0f);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_supply_brew_jitter", 1.0f);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_supply_restore_jitter", 1.0f);
    reset_inferno_at_public_wave(raw_state, 71, 1.0f);

    ASSERT_INT_EQ("jitter keeps brew nonnegative",
        state->player.brew_doses >= 0, 1);
    ASSERT_INT_EQ("jitter keeps brew within full supplies",
        state->player.brew_doses <= 24, 1);
    ASSERT_INT_EQ("jitter keeps restore nonnegative",
        state->player.restore_doses >= 0, 1);
    ASSERT_INT_EQ("jitter keeps restore within full supplies",
        state->player.restore_doses <= 36, 1);

    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_agent", 0);
    inf_put_int_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_supply_jitter_mode", INF_CURRICULUM_SUPPLY_MODE_OFF);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_supply_shared_jitter", 0.0f);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_supply_brew_jitter", 0.0f);
    inf_put_float_ctx(raw_state, (EncounterContext*)&test_context, "curriculum_supply_restore_jitter", 0.0f);
    inf_destroy(raw_state);
}

static void test_late_start_supply_observations(void) {
    printf("--- inferno late-start supply observations ---\n");

    EncounterState* raw_state = inf_create();
    InfernoState* state = (InfernoState*)raw_state;
    float obs[INF_NUM_OBS];

    reset_inferno_at_public_wave(raw_state, 69, 1.0f);
    inf_write_obs_ctx(raw_state, (EncounterContext*)&test_context, obs);

    int observed_brew_doses = 0;
    int observed_restore_doses = 0;
    int observed_bastion_doses = 0;
    int observed_stamina_doses = 0;
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
        int offset = OSRS_SHARED_OBS_INVENTORY_START +
            cell * OSRS_SHARED_INVENTORY_CELL_OBS_FEATURES;
        uint16_t content_code =
            osrs_inventory_cell_obs_code_decode(obs[offset]);
        const OsrsItemContentMetadata* metadata =
            osrs_item_content_metadata(content_code);
        if (metadata->consumable_kind == OSRS_CONSUMABLE_BREW) {
            observed_brew_doses += metadata->dose_count;
        } else if (metadata->consumable_kind == OSRS_CONSUMABLE_SUPER_RESTORE) {
            observed_restore_doses += metadata->dose_count;
        } else if (metadata->consumable_kind == OSRS_CONSUMABLE_BASTION) {
            observed_bastion_doses += metadata->dose_count;
        } else if (metadata->consumable_kind == OSRS_CONSUMABLE_STAMINA) {
            observed_stamina_doses += metadata->dose_count;
        }
    }
    ASSERT_INT_EQ("shared inventory exposes brew doses",
        observed_brew_doses, state->player.brew_doses);
    ASSERT_INT_EQ("shared inventory exposes restore doses",
        observed_restore_doses, state->player.restore_doses);
    ASSERT_INT_EQ("shared inventory exposes bastion doses",
        observed_bastion_doses, state->player.bastion_doses);
    ASSERT_INT_EQ("shared inventory exposes stamina doses",
        observed_stamina_doses, state->player.stamina_doses);

    inf_destroy(raw_state);
}

static void test_tagged_jad_healer_melee_geometry(void) {
    printf("--- tagged jad healer melee geometry ---\n");

    InfernoState diagonal_state = make_test_state(20, 20);
    InfernoState cardinal_state = make_test_state(20, 20);
    InfernoState meleer_diagonal_state = make_test_state(20, 20);

    diagonal_state.player.current_defence = 99;
    diagonal_state.player.current_magic = 99;
    diagonal_state.player.prayer = PRAYER_NONE;
    diagonal_state.weapon_set = INF_GEAR_MAGE;

    cardinal_state.player.current_defence = 99;
    cardinal_state.player.current_magic = 99;
    cardinal_state.player.prayer = PRAYER_NONE;
    cardinal_state.weapon_set = INF_GEAR_MAGE;

    meleer_diagonal_state.player.current_defence = 99;
    meleer_diagonal_state.player.current_magic = 99;
    meleer_diagonal_state.player.prayer = PRAYER_NONE;
    meleer_diagonal_state.weapon_set = INF_GEAR_MAGE;

    diagonal_state.npcs[0] =
        make_test_npc(INF_NPC_HEALER_JAD, 21, 21, 1);
    diagonal_state.npcs[0].active = 1;
    diagonal_state.npcs[0].aggro_target = -1;

    cardinal_state.npcs[0] =
        make_test_npc(INF_NPC_HEALER_JAD, 21, 20, 1);
    cardinal_state.npcs[0].active = 1;
    cardinal_state.npcs[0].aggro_target = -1;

    meleer_diagonal_state.npcs[0] =
        make_test_npc(INF_NPC_MELEER, 21, 21, 1);
    meleer_diagonal_state.npcs[0].active = 1;
    meleer_diagonal_state.npcs[0].aggro_target = -1;

    inf_npc_attack_ctx(&diagonal_state, &test_context, 0);
    inf_npc_attack_ctx(&cardinal_state, &test_context, 0);
    inf_npc_attack_ctx(&meleer_diagonal_state, &test_context, 0);

    ASSERT_INT_EQ("diagonal healer does not attack",
        diagonal_state.npcs[0].attacked_this_tick, 0);
    ASSERT_INT_EQ("diagonal healer keeps attack style none",
        diagonal_state.npcs[0].attack_style_this_tick, ATTACK_STYLE_NONE);
    ASSERT_INT_EQ("cardinal healer attacks",
        cardinal_state.npcs[0].attacked_this_tick, 1);
    ASSERT_INT_EQ("cardinal healer uses melee",
        cardinal_state.npcs[0].attack_style_this_tick, ATTACK_STYLE_MELEE);
    ASSERT_INT_EQ("diagonal pure meleer does not attack",
        meleer_diagonal_state.npcs[0].attacked_this_tick, 0);
    ASSERT_INT_EQ("diagonal pure meleer keeps attack style none",
        meleer_diagonal_state.npcs[0].attack_style_this_tick,
        ATTACK_STYLE_NONE);
}

static void test_overlap_shuffle_hold_after_recent_target_click(void) {
    printf("--- overlap shuffle held after recent target click ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player_last_interaction_target_slot = 0;
    state.player_last_interaction_age = 0;

    state.npcs[0] = make_test_npc(
        INF_NPC_RANGER, 20, 20, INF_NPC_STATS[INF_NPC_RANGER].size);
    state.npcs[0].active = 1;

    inf_npc_move_ctx(&state, &test_context, 0);

    ASSERT_INT_EQ("held overlap keeps x", state.npcs[0].x, 20);
    ASSERT_INT_EQ("held overlap keeps y", state.npcs[0].y, 20);
    ASSERT_INT_EQ("held overlap does not mark moved", state.npcs[0].moved_this_tick, 0);
}

static void test_overlap_shuffle_respects_npc_occupancy(void) {
    printf("--- overlap shuffle respects npc occupancy ---\n");

    const uint32_t west_shuffle_seed = 12345;

    InfernoState clear_state = make_test_state(20, 20);
    clear_state.rng_state = west_shuffle_seed;
    clear_state.npcs[0] = make_test_npc(INF_NPC_HEALER_JAD, 20, 20, 1);
    clear_state.npcs[0].active = 1;
    clear_state.npcs[1] = make_test_npc(INF_NPC_HEALER_JAD, 21, 20, 1);
    clear_state.npcs[1].active = 1;
    clear_state.npcs[2] = make_test_npc(INF_NPC_HEALER_JAD, 20, 21, 1);
    clear_state.npcs[2].active = 1;
    clear_state.npcs[3] = make_test_npc(INF_NPC_HEALER_JAD, 20, 19, 1);
    clear_state.npcs[3].active = 1;
    inf_rebuild_npc_collision_flags(&clear_state);

    inf_npc_move_ctx(&clear_state, &test_context, 0);

    ASSERT_INT_EQ("clear sampled overlap shuffle moves west x", clear_state.npcs[0].x, 19);
    ASSERT_INT_EQ("clear sampled overlap shuffle moves west y", clear_state.npcs[0].y, 20);
    ASSERT_INT_EQ("clear sampled overlap shuffle marks moved", clear_state.npcs[0].moved_this_tick, 1);

    InfernoState blocked_state = make_test_state(20, 20);
    blocked_state.rng_state = west_shuffle_seed;
    blocked_state.npcs[0] = make_test_npc(INF_NPC_HEALER_JAD, 20, 20, 1);
    blocked_state.npcs[0].active = 1;
    blocked_state.npcs[1] = make_test_npc(INF_NPC_HEALER_JAD, 21, 20, 1);
    blocked_state.npcs[1].active = 1;
    blocked_state.npcs[2] = make_test_npc(INF_NPC_HEALER_JAD, 19, 20, 1);
    blocked_state.npcs[2].active = 1;
    blocked_state.npcs[3] = make_test_npc(INF_NPC_HEALER_JAD, 20, 21, 1);
    blocked_state.npcs[3].active = 1;
    inf_rebuild_npc_collision_flags(&blocked_state);

    inf_npc_move_ctx(&blocked_state, &test_context, 0);

    ASSERT_INT_EQ("blocked sampled overlap shuffle does not fallback x", blocked_state.npcs[0].x, 20);
    ASSERT_INT_EQ("blocked sampled overlap shuffle does not fallback y", blocked_state.npcs[0].y, 20);
    ASSERT_INT_EQ("blocked sampled overlap shuffle does not mark moved",
                  blocked_state.npcs[0].moved_this_tick, 0);
}

static void test_large_npc_overlap_shuffle_can_partially_unclip(void) {
    printf("--- large npc overlap shuffle can partially unclip ---\n");

    InfernoState state = make_test_state(21, 21);
    state.rng_state = 12345;
    state.npcs[0] = make_test_npc(
        INF_NPC_MAGER, 20, 20, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[0].active = 1;

    inf_npc_move_ctx(&state, &test_context, 0);

    int dx = abs(state.npcs[0].x - 20);
    int dy = abs(state.npcs[0].y - 20);
    ASSERT_INT_EQ("large npc takes one shuffle step", dx + dy, 1);
    ASSERT_INT_EQ("large npc marks moved", state.npcs[0].moved_this_tick, 1);
}


static void test_tagged_jad_healer_stops_at_melee_contact(void) {
    printf("--- tagged jad healer stops at melee contact ---\n");

    InfernoState state = make_test_state(20, 20);
    state.npcs[0] = make_test_npc(INF_NPC_HEALER_JAD, 19, 20, 1);
    state.npcs[0].active = 1;
    state.npcs[0].aggro_target = -1;

    inf_npc_move_ctx(&state, &test_context, 0);

    ASSERT_INT_EQ("healer keeps melee contact x", state.npcs[0].x, 19);
    ASSERT_INT_EQ("healer keeps melee contact y", state.npcs[0].y, 20);
    ASSERT_INT_EQ("healer does not mark moved", state.npcs[0].moved_this_tick, 0);
}

static void test_tagged_jad_healers_queue_behind_front_healer(void) {
    printf("--- tagged jad healers queue behind front healer ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player.current_defence = 99;
    state.player.current_magic = 99;
    state.player.prayer = PRAYER_PROTECT_MAGIC;
    state.weapon_set = INF_GEAR_MAGE;

    for (int i = 0; i < 5; i++) {
        state.npcs[i] = make_test_npc(INF_NPC_HEALER_JAD, 19 - i, 20, 1);
        state.npcs[i].active = 1;
        state.npcs[i].aggro_target = -1;
        state.npcs[i].attack_timer = 0;
    }

    inf_tick_npcs_ctx(&state, &test_context);

    int attacks = 0;
    int on_player = 0;
    for (int i = 0; i < 5; i++) {
        if (state.npcs[i].attacked_this_tick) attacks++;
        if (state.npcs[i].x == state.player.x && state.npcs[i].y == state.player.y)
            on_player++;
    }

    ASSERT_INT_EQ("only front healer attacks", attacks, 1);
    ASSERT_INT_EQ("no healer steps onto player", on_player, 0);
    ASSERT_INT_EQ("front healer remains first in queue", state.npcs[0].x, 19);
    ASSERT_INT_EQ("second healer remains blocked behind front", state.npcs[1].x, 18);
}

static void test_meleer_dig_can_stack_without_losing_collision_flag(void) {
    printf("--- meleer dig can stack without losing collision flag ---\n");

    InfernoState state = make_test_state(20, 20);
    state.npcs[0] = make_test_npc(
        INF_NPC_MELEER, 5, 5, INF_NPC_STATS[INF_NPC_MELEER].size);
    state.npcs[0].active = 1;
    state.npcs[0].dig_freeze_timer = 1;
    int dig_x = state.player.x - state.npcs[0].size + 1;
    int dig_y = state.player.y - state.npcs[0].size + 1;
    state.npcs[1] = make_test_npc(
        INF_NPC_RANGER, dig_x, dig_y, INF_NPC_STATS[INF_NPC_RANGER].size);
    state.npcs[1].active = 1;

    inf_meleer_dig_check_ctx(&state, &test_context, 0);

    ASSERT_INT_EQ("dig lands on first candidate x", state.npcs[0].x, dig_x);
    ASSERT_INT_EQ("dig lands on first candidate y", state.npcs[0].y, dig_y);
    ASSERT_INT_EQ("stacked landing keeps both NPCs at x",
        state.npcs[1].x, state.npcs[0].x);
    ASSERT_INT_EQ("stacked landing keeps both NPCs at y",
        state.npcs[1].y, state.npcs[0].y);
}

static void test_jad_healer_spawn_offsets_match_wave_67_reference(void) {
    printf("--- jad healer spawn offsets match wave 67 reference ---\n");

    InfernoState state = make_test_state(18, 32);
    state.rng_state = 12345;
    state.wave = 66;
    state.npcs[0] = make_test_npc(INF_NPC_JAD, 23, 30, INF_NPC_STATS[INF_NPC_JAD].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 100;
    state.npcs[0].max_hp = 300;

    inf_jad_check_healers_ctx(&state, &test_context, 0);

    int healers = 0;
    for (int i = 1; i < INF_MAX_NPCS; i++) {
        if (!state.npcs[i].active || state.npcs[i].type != INF_NPC_HEALER_JAD) continue;
        healers++;
        int dx = state.npcs[i].x - state.npcs[0].x;
        int dy = state.npcs[i].y - state.npcs[0].y;
        ASSERT_INT_EQ("wave 67 healer owner", inf_npc_healer(&state.npcs[i])->owner_idx, 0);
        ASSERT_INT_EQ("wave 67 healer aggro", state.npcs[i].aggro_target, 0);
        ASSERT_INT_EQ("wave 67 healer x min", dx >= -5, 1);
        ASSERT_INT_EQ("wave 67 healer x max", dx <= 5, 1);
        ASSERT_INT_EQ("wave 67 healer y min", dy >= -4, 1);
        ASSERT_INT_EQ("wave 67 healer y max", dy <= 10, 1);
        ASSERT_INT_EQ("wave 67 healer outside jad footprint",
            encounter_entity_footprints_overlap(
                state.npcs[i].x, state.npcs[i].y, 1,
                state.npcs[0].x, state.npcs[0].y, state.npcs[0].size),
            0);
    }
    ASSERT_INT_EQ("wave 67 healer count", healers, 5);
}

static void test_jad_healer_spawn_offsets_match_zuk_reference(void) {
    printf("--- jad healer spawn offsets match zuk reference ---\n");

    InfernoState state = make_test_state(INF_ZUK_PLAYER_START_X, INF_ZUK_PLAYER_START_Y);
    state.rng_state = 67890;
    state.wave = 68;
    state.npcs[0] = make_test_npc(INF_NPC_JAD, 24, 32, INF_NPC_STATS[INF_NPC_JAD].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 100;
    state.npcs[0].max_hp = 300;

    inf_jad_check_healers_ctx(&state, &test_context, 0);

    int healers = 0;
    for (int i = 1; i < INF_MAX_NPCS; i++) {
        if (!state.npcs[i].active || state.npcs[i].type != INF_NPC_HEALER_JAD) continue;
        healers++;
        int dx = state.npcs[i].x - state.npcs[0].x;
        int dy = state.npcs[i].y - state.npcs[0].y;
        ASSERT_INT_EQ("zuk healer x min", dx >= 0, 1);
        ASSERT_INT_EQ("zuk healer x max", dx <= 5, 1);
        ASSERT_INT_EQ("zuk healer y min", dy >= 5, 1);
        ASSERT_INT_EQ("zuk healer y max", dy <= 8, 1);
        ASSERT_INT_EQ("zuk healer outside jad footprint",
            encounter_entity_footprints_overlap(
                state.npcs[i].x, state.npcs[i].y, 1,
                state.npcs[0].x, state.npcs[0].y, state.npcs[0].size),
            0);
    }
    ASSERT_INT_EQ("zuk healer count", healers, 3);
}

static void test_npc_terrain_blocks_full_footprint_lava_shelf(void) {
    printf("--- npc terrain blocks full footprint lava shelf ---\n");

    InfernoState state = make_test_state(INF_ZUK_PLAYER_START_X, INF_ZUK_PLAYER_START_Y);
    state.wave = 68;

    ASSERT_INT_EQ("jad footprint y39 fits player arena",
        inf_npc_environment_blocked_ctx(
            &state, &test_context,
            24, 39, INF_NPC_STATS[INF_NPC_JAD].size), 0);
    ASSERT_INT_EQ("jad footprint y40 enters lava shelf",
        inf_npc_environment_blocked_ctx(
            &state, &test_context,
            24, 40, INF_NPC_STATS[INF_NPC_JAD].size), 1);
    ASSERT_INT_EQ("jad movement y40 is blocked",
        inf_npc_environment_blocked_ctx(
            &state, &test_context,
            24, 40, INF_NPC_STATS[INF_NPC_JAD].size), 1);
}

static void test_zuk_jad_healer_spawn_falls_back_to_passable_arena_tiles(void) {
    printf("--- zuk jad healer spawn falls back to passable arena tiles ---\n");

    InfernoState state = make_test_state(INF_ZUK_PLAYER_START_X, INF_ZUK_PLAYER_START_Y);
    state.rng_state = 96969;
    state.wave = 68;
    state.npcs[0] = make_test_npc(INF_NPC_JAD, 24, 39, INF_NPC_STATS[INF_NPC_JAD].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 100;
    state.npcs[0].max_hp = 300;

    inf_jad_check_healers_ctx(&state, &test_context, 0);

    int healers = 0;
    for (int i = 1; i < INF_MAX_NPCS; i++) {
        if (!state.npcs[i].active || state.npcs[i].type != INF_NPC_HEALER_JAD) continue;
        healers++;
        int dx = state.npcs[i].x - state.npcs[0].x;
        int dy = state.npcs[i].y - state.npcs[0].y;
        ASSERT_INT_EQ("fallback healer x min", dx >= -5, 1);
        ASSERT_INT_EQ("fallback healer x max", dx <= 5, 1);
        ASSERT_INT_EQ("fallback healer y min", dy >= -4, 1);
        ASSERT_INT_EQ("fallback healer y max", dy <= 10, 1);
        ASSERT_INT_EQ("fallback healer stays in arena", state.npcs[i].y <= INF_ARENA_MAX_Y, 1);
        ASSERT_INT_EQ("fallback healer terrain valid",
            inf_npc_environment_blocked_ctx(
                &state, &test_context,
                state.npcs[i].x, state.npcs[i].y, 1), 0);
        ASSERT_INT_EQ("fallback healer outside jad footprint",
            encounter_entity_footprints_overlap(
                state.npcs[i].x, state.npcs[i].y, 1,
                state.npcs[0].x, state.npcs[0].y, state.npcs[0].size),
            0);
    }
    ASSERT_INT_EQ("fallback healer count", healers, 3);
}

static void test_meleer_dig_landing_order(void) {
    printf("--- meleer dig landing order ---\n");

    InfernoState state = make_test_state(20, 20);
    state.npcs[0] = make_test_npc(
        INF_NPC_MELEER, 5, 5, INF_NPC_STATS[INF_NPC_MELEER].size);
    state.npcs[0].active = 1;
    state.npcs[0].dig_freeze_timer = 1;

    state.pillars[0].active = 1;
    state.pillars[0].x = 17;
    state.pillars[0].y = 17;

    inf_meleer_dig_check_ctx(&state, &test_context, 0);

    ASSERT_INT_EQ("blocked first landing candidate falls through to player tile x", state.npcs[0].x, 20);
    ASSERT_INT_EQ("blocked first landing candidate falls through to player tile y", state.npcs[0].y, 20);
    ASSERT_INT_EQ("dig freeze consumed", state.npcs[0].dig_freeze_timer, 0);
    ASSERT_INT_EQ("post-dig stun applied", state.npcs[0].stun_timer, 2);
    ASSERT_INT_EQ("post-dig attack delay applied", state.npcs[0].dig_attack_delay, 6);
}


static void test_melee_fallback_geometry(void) {
    printf("--- inferno melee fallback geometry ---\n");

    InfernoState diagonal_state = make_test_state(5, 5);
    InfernoState cardinal_state = make_test_state(5, 5);
    InfernoState distant_state = make_test_state(5, 5);

    InfNPC ranger_diagonal = make_test_npc(INF_NPC_RANGER, 6, 6, 1);
    InfNPC mager_diagonal = make_test_npc(INF_NPC_MAGER, 6, 6, 1);
    InfNPC blob_diagonal = make_test_npc(INF_NPC_BLOB, 6, 6, 1);
    InfNPC blob_cardinal = make_test_npc(INF_NPC_BLOB, 6, 5, 1);
    InfNPC jad_diagonal = make_test_npc(INF_NPC_JAD, 6, 6, 1);
    InfNPC jad_cardinal = make_test_npc(INF_NPC_JAD, 6, 5, 1);
    InfNPC blob_distant = make_test_npc(INF_NPC_BLOB, 7, 5, 1);

    InfNPCStats ranged_stats = make_test_stats(ATTACK_STYLE_RANGED);
    InfNPCStats magic_stats = make_test_stats(ATTACK_STYLE_MAGIC);

    ASSERT_INT_EQ(
        "ranger diagonal melee fallback",
        inf_melee_fallback_possible_at_tile((&diagonal_state)->player.x, (&diagonal_state)->player.y, &ranger_diagonal, &ranged_stats, ATTACK_STYLE_RANGED, distance_to_player(&diagonal_state, &ranger_diagonal)),
        1);
    ASSERT_INT_EQ(
        "mager diagonal melee fallback",
        inf_melee_fallback_possible_at_tile((&diagonal_state)->player.x, (&diagonal_state)->player.y, &mager_diagonal, &magic_stats, ATTACK_STYLE_MAGIC, distance_to_player(&diagonal_state, &mager_diagonal)),
        1);
    ASSERT_INT_EQ(
        "blob diagonal melee fallback blocked",
        inf_melee_fallback_possible_at_tile((&diagonal_state)->player.x, (&diagonal_state)->player.y, &blob_diagonal, &magic_stats, ATTACK_STYLE_MAGIC, distance_to_player(&diagonal_state, &blob_diagonal)),
        0);
    ASSERT_INT_EQ(
        "blob cardinal melee fallback",
        inf_melee_fallback_possible_at_tile((&cardinal_state)->player.x, (&cardinal_state)->player.y, &blob_cardinal, &magic_stats, ATTACK_STYLE_MAGIC, distance_to_player(&cardinal_state, &blob_cardinal)),
        1);
    ASSERT_INT_EQ(
        "jad diagonal melee fallback blocked",
        inf_melee_fallback_possible_at_tile((&diagonal_state)->player.x, (&diagonal_state)->player.y, &jad_diagonal, &ranged_stats, ATTACK_STYLE_RANGED, distance_to_player(&diagonal_state, &jad_diagonal)),
        0);
    ASSERT_INT_EQ(
        "jad cardinal melee fallback",
        inf_melee_fallback_possible_at_tile((&cardinal_state)->player.x, (&cardinal_state)->player.y, &jad_cardinal, &ranged_stats, ATTACK_STYLE_RANGED, distance_to_player(&cardinal_state, &jad_cardinal)),
        1);
    ASSERT_INT_EQ(
        "fallback blocked outside melee distance",
        inf_melee_fallback_possible_at_tile((&distant_state)->player.x, (&distant_state)->player.y, &blob_distant, &magic_stats, ATTACK_STYLE_MAGIC, distance_to_player(&distant_state, &blob_distant)),
        0);
    ASSERT_INT_EQ(
        "fallback blocked when planned style already melee",
        inf_melee_fallback_possible_at_tile((&cardinal_state)->player.x, (&cardinal_state)->player.y, &blob_cardinal, &magic_stats, ATTACK_STYLE_MELEE, distance_to_player(&cardinal_state, &blob_cardinal)),
        0);
}

static void test_style_choice_sampling(void) {
    printf("--- inferno style choice sampling ---\n");

    uint32_t rng_state = 12345;
    int saw_melee = 0;
    int saw_ranged = 0;

    for (int i = 0; i < 128; i++) {
        int style = inf_choose_attack_style_for_tick(
            &rng_state, INF_STYLE_MASK_MELEE | INF_STYLE_MASK_RANGED);
        if (style == ATTACK_STYLE_MELEE) saw_melee = 1;
        if (style == ATTACK_STYLE_RANGED) saw_ranged = 1;
    }

    ASSERT_INT_EQ("50-50 branch can emit melee", saw_melee, 1);
    ASSERT_INT_EQ("50-50 branch can emit primary style", saw_ranged, 1);
    ASSERT_INT_EQ(
        "single-style mask stays deterministic",
        inf_choose_attack_style_for_tick(&rng_state, INF_STYLE_MASK_MAGIC),
        ATTACK_STYLE_MAGIC);
}

static void test_dead_mob_store_eligibility(void) {
    printf("--- inferno dead mob resurrection eligibility ---\n");

    ASSERT_INT_EQ("bat resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_BAT), 1);
    ASSERT_INT_EQ("blob parent resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_BLOB), 1);
    ASSERT_INT_EQ("meleer resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_MELEER), 1);
    ASSERT_INT_EQ("ranger resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_RANGER), 1);
    ASSERT_INT_EQ("mager resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_MAGER), 1);

    ASSERT_INT_EQ("nibbler not resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_NIBBLER), 0);
    ASSERT_INT_EQ("blob melee split not resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_BLOB_MELEE), 0);
    ASSERT_INT_EQ("blob range split not resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_BLOB_RANGE), 0);
    ASSERT_INT_EQ("blob mage split not resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_BLOB_MAGE), 0);
    ASSERT_INT_EQ("jad not resurrectable", inf_dead_mob_is_resurrectable(INF_NPC_JAD), 0);
}

static void test_resurrected_mob_does_not_reenter_dead_store(void) {
    printf("--- resurrected mob does not reenter dead store ---\n");

    InfernoState state = make_test_state(25, 16);
    state.wave = 35;

    state.npcs[0] = make_test_npc(INF_NPC_MAGER, 20, 20, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;

    state.dead_mobs[0].type = INF_NPC_RANGER;
    state.dead_mobs[0].x = 18;
    state.dead_mobs[0].y = 18;
    state.dead_mobs[0].hp = INF_NPC_STATS[INF_NPC_RANGER].hp / 2;
    state.dead_mobs[0].max_hp = INF_NPC_STATS[INF_NPC_RANGER].hp;
    state.dead_mob_count = 1;

    ASSERT_INT_EQ("resurrection succeeds", force_mager_resurrect(&state, 0), 1);
    ASSERT_INT_EQ("dead store consumed", state.dead_mob_count, 0);

    int resurrected_slot = -1;
    for (int i = 1; i < INF_MAX_NPCS; i++) {
        if (state.npcs[i].active && state.npcs[i].type == INF_NPC_RANGER) {
            resurrected_slot = i;
            break;
        }
    }
    ASSERT_INT_EQ("resurrected ranger spawned", resurrected_slot >= 0, 1);
    ASSERT_INT_EQ("respawned ranger marked resurrected",
        state.npcs[resurrected_slot].resurrection_count, 1);

    inf_store_dead_mob(&state, &state.npcs[resurrected_slot]);
    ASSERT_INT_EQ("resurrected ranger not re-added", state.dead_mob_count, 0);
}

static void test_blob_split_waits_for_death_removal(void) {
    printf("--- blob split waits for death removal ---\n");

    InfernoState state = make_test_state(25, 16);
    state.wave = 30;
    state.npcs[0] = make_test_npc(
        INF_NPC_BLOB, 10, 10, INF_NPC_STATS[INF_NPC_BLOB].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 0;
    state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_BLOB].hp;

    inf_apply_npc_death(&state, 0);

    ASSERT_INT_EQ("dead blob remains during death linger", state.npcs[0].active, 1);
    ASSERT_INT_EQ("dead blob death linger starts", state.npcs[0].death_ticks, INF_NPC_DEATH_LINGER_TICKS);
    ASSERT_INT_EQ("blob parent enters mager resurrection store", state.dead_mob_count, 1);
    ASSERT_INT_EQ("blob parent store type", state.dead_mobs[0].type, INF_NPC_BLOB);
    ASSERT_INT_EQ("blob split melee not spawned before removal",
        count_active_npc_type(&state, INF_NPC_BLOB_MELEE), 0);
    ASSERT_INT_EQ("blob split range not spawned before removal",
        count_active_npc_type(&state, INF_NPC_BLOB_RANGE), 0);
    ASSERT_INT_EQ("blob split mage not spawned before removal",
        count_active_npc_type(&state, INF_NPC_BLOB_MAGE), 0);

    for (int t = 0; t < INF_NPC_DEATH_LINGER_TICKS; t++)
        inf_tick_npcs_ctx(&state, &test_context);

    ASSERT_INT_EQ("blob parent removed after death linger", state.npcs[0].active, 0);
    ASSERT_INT_EQ("blob split melee spawned after removal",
        find_active_npc_type_at(&state, INF_NPC_BLOB_MELEE, 10, 10) >= 0, 1);
    ASSERT_INT_EQ("blob split range spawned after removal",
        find_active_npc_type_at(&state, INF_NPC_BLOB_RANGE, 11, 11) >= 0, 1);
    ASSERT_INT_EQ("blob split mage spawned after removal",
        find_active_npc_type_at(&state, INF_NPC_BLOB_MAGE, 12, 12) >= 0, 1);
    ASSERT_INT_EQ("blob split melee not resurrectable",
        state.dead_mob_count, 1);

    int range_idx = find_active_npc_type_at(&state, INF_NPC_BLOB_RANGE, 11, 11);
    int melee_idx = find_active_npc_type_at(&state, INF_NPC_BLOB_MELEE, 10, 10);
    int mage_idx = find_active_npc_type_at(&state, INF_NPC_BLOB_MAGE, 12, 12);
    ASSERT_INT_EQ("split range cooldown", state.npcs[range_idx].attack_timer, 4);
    ASSERT_INT_EQ("split melee cooldown", state.npcs[melee_idx].attack_timer, 4);
    ASSERT_INT_EQ("split mage cooldown", state.npcs[mage_idx].attack_timer, 4);
}

static void test_mager_resurrection_render_event_is_not_magic_projectile(void) {
    printf("--- mager resurrection render event is not magic projectile ---\n");

    InfernoState state = make_test_state(25, 16);
    state.wave = 35;
    state.player.current_defence = 99;
    state.player.current_magic = 99;
    state.weapon_set = INF_GEAR_MAGE;
    state.npcs[0] = make_test_npc(
        INF_NPC_MAGER, 30, 20, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;
    state.npcs[0].attack_timer = 0;
    state.dead_mobs[0].type = INF_NPC_RANGER;
    state.dead_mobs[0].x = 18;
    state.dead_mobs[0].y = 18;
    state.dead_mobs[0].hp = INF_NPC_STATS[INF_NPC_RANGER].hp / 2;
    state.dead_mobs[0].max_hp = INF_NPC_STATS[INF_NPC_RANGER].hp;
    state.dead_mob_count = 1;

    ASSERT_INT_EQ("mager attack converted into resurrection",
        force_mager_attack_resurrection(&state, 0), 1);
    ASSERT_INT_EQ("resurrection is marked for render",
        state.npcs[0].resurrecting_this_tick, 1);
    ASSERT_INT_EQ("resurrection does not expose a magic attack style",
        state.npcs[0].attack_style_this_tick, ATTACK_STYLE_NONE);
    ASSERT_INT_EQ("resurrected ranger attack delay",
        state.npcs[state.npcs[0].resurrection_visual_target].attack_timer,
        INF_NPC_STATS[INF_NPC_RANGER].attack_speed);

    RenderEntity entities[4];
    int count = 0;
    inf_fill_render_entities_ctx((EncounterState*)&state, (EncounterContext*)&test_context, entities, 4, &count);
    ASSERT_INT_EQ("resurrection render has mager and resurrected mob", count >= 3, 1);
    ASSERT_INT_EQ("mager uses resurrection animation",
        entities[1].npc_anim_id, INF_GEN_ANIM_MAGER_RESURRECT);
    ASSERT_INT_EQ("mager faces resurrected mob",
        entities[1].dest_x,
        state.npcs[state.npcs[0].resurrection_visual_target].x +
            state.npcs[state.npcs[0].resurrection_visual_target].size / 2);

    EncounterOverlay ov;
    memset(&ov, 0, sizeof(ov));
    inf_render_post_tick_ctx((EncounterState*)&state, (EncounterContext*)&test_context, &ov);
    ASSERT_INT_EQ("resurrection does not emit magic projectile", ov.projectile_count, 0);
}

static void test_double_mager_wave_resurrection_limit(void) {
    printf("--- double mager wave respects once-only resurrection ---\n");

    InfernoState state = make_test_state(25, 16);
    state.wave = 65;

    state.npcs[0] = make_test_npc(INF_NPC_MAGER, 18, 18, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;

    state.dead_mobs[0].type = INF_NPC_MAGER;
    state.dead_mobs[0].x = 22;
    state.dead_mobs[0].y = 22;
    state.dead_mobs[0].hp = INF_NPC_STATS[INF_NPC_MAGER].hp / 2;
    state.dead_mobs[0].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;
    state.dead_mob_count = 1;

    ASSERT_INT_EQ("first mage resurrection succeeds", force_mager_resurrect(&state, 0), 1);

    int resurrected_slot = -1;
    for (int i = 1; i < INF_MAX_NPCS; i++) {
        if (state.npcs[i].active && state.npcs[i].type == INF_NPC_MAGER) {
            resurrected_slot = i;
            break;
        }
    }
    ASSERT_INT_EQ("second mager spawned", resurrected_slot >= 0, 1);
    ASSERT_INT_EQ("respawned mager marked resurrected",
        state.npcs[resurrected_slot].resurrection_count, 1);

    inf_store_dead_mob(&state, &state.npcs[0]);
    ASSERT_INT_EQ("original mage can still enter dead store once", state.dead_mob_count, 1);

    ASSERT_INT_EQ("resurrected mage can resurrect the original once",
        force_mager_resurrect(&state, resurrected_slot), 1);

    int original_respawn_slot = -1;
    for (int i = 1; i < INF_MAX_NPCS; i++) {
        if (i == resurrected_slot) continue;
        if (state.npcs[i].active && state.npcs[i].type == INF_NPC_MAGER &&
            state.npcs[i].resurrection_count == 1) {
            original_respawn_slot = i;
            break;
        }
    }
    ASSERT_INT_EQ("original mage respawned once", original_respawn_slot >= 0, 1);

    inf_store_dead_mob(&state, &state.npcs[resurrected_slot]);
    ASSERT_INT_EQ("already-resurrected mage stays out of store", state.dead_mob_count, 0);
    inf_store_dead_mob(&state, &state.npcs[original_respawn_slot]);
    ASSERT_INT_EQ("re-resurrected original mage stays out of store", state.dead_mob_count, 0);
}

static void test_pending_hit_obs_timer_prefers_prayer_window(void) {
    printf("--- pending hit obs timer prefers prayer window ---\n");

    EncounterPendingHit jad_hit = {0};
    jad_hit.check_prayer = 1;
    jad_hit.prayer_check_delay = 3;
    jad_hit.ticks_remaining = 4;

    EncounterPendingHit normal_hit = {0};
    normal_hit.check_prayer = 0;
    normal_hit.prayer_check_delay = 0;
    normal_hit.ticks_remaining = 2;

    ASSERT_INT_EQ("jad timer uses prayer window", inf_pending_hit_obs_timer(&jad_hit), 3);
    ASSERT_INT_EQ("normal timer uses travel time", inf_pending_hit_obs_timer(&normal_hit), 2);
}

static void test_blob_attacks_player_on_six_tick_cadence(void) {
    printf("--- blob attacks the player on a 6-tick cadence ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player.current_defence = 99;
    state.player.current_magic = 99;
    state.player.prayer = PRAYER_NONE;
    state.weapon_set = INF_GEAR_MAGE;

    state.npcs[0] = make_test_npc(
        INF_NPC_BLOB, 30, 20, INF_NPC_STATS[INF_NPC_BLOB].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = INF_NPC_STATS[INF_NPC_BLOB].hp;

    int prev_scanned = state.npcs[0].blob_scanned_prayer;
    int last_fire = -1, gap_a = -1, gap_b = -1;
    for (int tick = 0; tick < 40; tick++) {
        inf_npc_attack_ctx(&state, &test_context, 0);
        int cur_scanned = state.npcs[0].blob_scanned_prayer;
        if (prev_scanned >= 0 && cur_scanned < 0) {
            if (last_fire >= 0) {
                if (gap_a < 0) gap_a = tick - last_fire;
                else if (gap_b < 0) gap_b = tick - last_fire;
            }
            last_fire = tick;
        }
        prev_scanned = cur_scanned;
    }

    ASSERT_INT_EQ("blob fire-to-fire cadence is 6 ticks", gap_a, 6);
    ASSERT_INT_EQ("blob cadence stays 6 across cycles", gap_b, 6);
}

static void test_jad_has_no_pre_fire_style_preview(void) {
    printf("--- jad has no pre-fire style preview ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player.current_defence = 99;
    state.player.current_magic = 99;
    state.player.prayer = PRAYER_NONE;
    state.weapon_set = INF_GEAR_MAGE;
    state.wave = 66;

    state.npcs[0] = make_test_npc(
        INF_NPC_JAD, 30, 20, INF_NPC_STATS[INF_NPC_JAD].size);
    state.npcs[0].active = 1;
    state.npcs[0].attack_timer = 2;

    inf_npc_attack_ctx(&state, &test_context, 0);

    ASSERT_INT_EQ("jad timer decrements without preview", state.npcs[0].attack_timer, 1);
    ASSERT_INT_EQ("jad style stays hidden before fire", inf_npc_jad(&state.npcs[0])->attack_style, ATTACK_STYLE_NONE);

}

static void test_jad_fire_tick_exposes_three_tick_prayer_deadline(void) {
    printf("--- jad fire tick exposes three tick prayer deadline ---\n");

    InfernoState state;
    init_jad_timing_test_state(&state, 10, 10, 16, 10);

    step_inferno_with_prayer(&state, 0);

    ASSERT_INT_EQ("jad attack queued one pending hit", state.player_pending_hits.count, 1);
    ASSERT_INT_EQ("jad style resets after firing", inf_npc_jad(&state.npcs[0])->attack_style, ATTACK_STYLE_NONE);
    ASSERT_INT_EQ("jad pending hit shows three tick prayer delay after fire", state.player_pending_hits.hits[0].prayer_check_delay, 3);
    ASSERT_INT_EQ("jad close-range hit lands four ticks after fire", state.player_pending_hits.hits[0].ticks_remaining, 4);

    float obs[INF_NUM_OBS];
    memset(obs, 0, sizeof(obs));
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);
    int pending_start = inferno_pending_hit_obs_start();
    ASSERT_FLOAT_NEAR("pending hit obs style is magic",
        obs[pending_start], (float)ATTACK_STYLE_MAGIC / 4.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("pending hit obs timer uses prayer window",
        obs[pending_start + 1], 0.3f, 1e-6f);
    ASSERT_FLOAT_NEAR("pending hit pre-check damage exposes max threat",
        obs[pending_start + 2], 113.0f / 150.0f, 1e-6f);
}

static void test_jad_prayer_on_third_tick_blocks(void) {
    printf("--- jad prayer on third tick blocks ---\n");

    InfernoState state;
    init_jad_timing_test_state(&state, 10, 10, 16, 10);

    step_inferno_with_prayer(&state, 0);
    step_inferno_with_prayer(&state, 0);
    step_inferno_with_prayer(&state, 0);
    step_inferno_with_prayer(&state, ENCOUNTER_OVERHEAD_SET_REFRESH_MAGIC);

    ASSERT_INT_EQ("jad prayer check consumed pending protection", state.player_pending_hits.hits[0].check_prayer, 0);
    ASSERT_INT_EQ("jad protected damage is frozen at zero", state.player_pending_hits.hits[0].damage, 0);
    ASSERT_INT_EQ("jad prayer check counted correct prayer", state.tick_scratch.prayer_correct, 1);

    step_inferno_with_prayer(&state, 0);
    ASSERT_INT_EQ("jad protected hit removed after landing", state.player_pending_hits.count, 0);
    ASSERT_INT_EQ("jad protected hit leaves player hp unchanged", state.player.current_hitpoints, 99);
}

static void test_jad_prayer_first_on_fourth_tick_does_not_block(void) {
    printf("--- jad prayer first on fourth tick does not block ---\n");

    int saw_late_damage = 0;
    for (uint32_t seed = 1; seed < 10000 && !saw_late_damage; seed++) {
        InfernoState state;
        init_jad_timing_test_state(&state, 10, 10, 16, 10);
        state.rng_state = seed;

        step_inferno_with_prayer(&state, 0);
        step_inferno_with_prayer(&state, 0);
        step_inferno_with_prayer(&state, 0);
        step_inferno_with_prayer(&state, 0);
        ASSERT_INT_EQ("late-prayer test reaches checked pending hit", state.player_pending_hits.hits[0].check_prayer, 0);

        step_inferno_with_prayer(&state, ENCOUNTER_OVERHEAD_SET_REFRESH_MAGIC);
        if (state.tick_scratch.damage_received > 0.0f) {
            saw_late_damage = 1;
            ASSERT_INT_EQ("late prayer did not block queued jad damage", state.player.current_hitpoints < 99, 1);
        }
    }
    ASSERT_INT_EQ("found a seed where late jad prayer takes damage", saw_late_damage, 1);
}

static void test_jad_long_distance_damage_uses_delayed_projectile_landing(void) {
    printf("--- jad long distance damage uses delayed projectile landing ---\n");

    int saw_expected_landing = 0;
    for (uint32_t seed = 1; seed < 10000 && !saw_expected_landing; seed++) {
        InfernoState state;
        init_jad_timing_test_state(&state, 10, 10, 36, 10);
        state.rng_state = seed;

        int dist = encounter_projectile_distance(
            state.npcs[0].x, state.npcs[0].y, state.npcs[0].size,
            state.player.x, state.player.y, 1,
            ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
        EncounterProjectileTiming timing =
            inf_npc_projectile_timing(INF_NPC_JAD, ATTACK_STYLE_MAGIC, dist);
        int expected_landing_after_fire =
            INF_JAD_PROJECTILE_DELAY + timing.damage_delay_ticks;

        step_inferno_with_prayer(&state, 0);
        for (int t = 1; t < expected_landing_after_fire; t++) {
            step_inferno_with_prayer(&state, 0);
            ASSERT_FLOAT_NEAR("jad long-distance hit has not landed early", state.tick_scratch.damage_received, 0.0f, 1e-6f);
        }
        step_inferno_with_prayer(&state, 0);
        if (state.tick_scratch.damage_received > 0.0f) {
            saw_expected_landing = 1;
        }
    }
    ASSERT_INT_EQ("found a seed where long-distance jad damage lands on expected tick", saw_expected_landing, 1);
}

static void test_triple_jad_pending_threats_fit_obs_layout(void) {
    printf("--- triple jad pending threats fit obs layout ---\n");

    InfernoState state;
    init_jad_timing_test_state(&state, 25, 30, 18, 33);
    state.wave = 67;
    state.npcs[1] = make_test_npc(INF_NPC_JAD, 28, 33, INF_NPC_STATS[INF_NPC_JAD].size);
    state.npcs[1].active = 1;
    state.npcs[1].attack_timer = 0;
    inf_npc_jad(&state.npcs[1])->attack_style = ATTACK_STYLE_RANGED;
    state.npcs[2] = make_test_npc(INF_NPC_JAD, 23, 22, INF_NPC_STATS[INF_NPC_JAD].size);
    state.npcs[2].active = 1;
    state.npcs[2].attack_timer = 0;
    inf_npc_jad(&state.npcs[2])->attack_style = ATTACK_STYLE_MAGIC;

    step_inferno_with_prayer(&state, 0);

    ASSERT_INT_EQ("triple jad queues three pending threats", state.player_pending_hits.count, 3);
    for (int h = 0; h < state.player_pending_hits.count; h++) {
        ASSERT_INT_EQ("each jad threat keeps three tick prayer deadline", state.player_pending_hits.hits[h].prayer_check_delay, 3);
    }

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);
    ASSERT_INT_EQ("inferno obs uses shared-prefix layout", INF_NUM_OBS, 530);
}

static void test_inferno_action_and_compact_obs_shape(void) {
    printf("--- inferno action and compact obs shape ---\n");

    ASSERT_INT_EQ("equip heads span every gear slot",
        INF_HEAD_EAT - OSRS_HEAD_EQUIP_BASE, NUM_GEAR_SLOTS);
    ASSERT_INT_EQ("equip head clicks cover every cell",
        INF_ACTION_DIMS[INF_HEAD_EQUIP_SLOT(GEAR_SLOT_WEAPON)],
        OSRS_INVENTORY_SIZE + 1);
    ASSERT_INT_EQ("eat head clicks cover every cell",
        INF_ACTION_DIMS[INF_HEAD_EAT], OSRS_INVENTORY_SIZE + 1);
    ASSERT_INT_EQ("drink head clicks cover every cell",
        INF_ACTION_DIMS[INF_HEAD_DRINK], OSRS_INVENTORY_SIZE + 1);
    ASSERT_INT_EQ("prayer action head uses shared overhead actions",
        INF_ACTION_DIMS[INF_HEAD_PRAYER], OSRS_OVERHEAD_DIM);
    ASSERT_INT_EQ("action mask spans the shared action heads",
        INF_ACTION_MASK_SIZE, 436);
    ASSERT_INT_EQ("shared player observation width",
        INF_OBS_AFTER_SHARED, OSRS_SHARED_OBS_SIZE);
    ASSERT_INT_EQ("compact pillar observation width", INF_PILLAR_OBS_SIZE, 9);
    ASSERT_INT_EQ("compact NPC observation width",
        INF_TOTAL_NPC_OBS_SIZE, INF_OBS_NPCS * INF_NPC_SLOT_FEATURES);
    ASSERT_INT_EQ("compact spark observation width",
        INF_PENDING_SPARK_OBS_SIZE, 128);
    ASSERT_INT_EQ("inferno observation width", INF_NUM_OBS, 530);
}

static void test_inferno_obs_wave_phase_code(void) {
    printf("--- inferno obs wave phase code ---\n");

    int waves[6] = {1, 18, 35, 50, 67, 69};
    for (int phase = 0; phase < 6; phase++) {
        InfernoState state = make_test_state(20, 20);
        state.wave = waves[phase] - 1;
        state.player.current_hitpoints = 99;
        state.player.base_hitpoints = 99;
        state.player.base_prayer = 99;
        state.player.current_prayer = 99;

        float obs[INF_NUM_OBS];
        inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);
        ASSERT_FLOAT_NEAR("wave phase compact code",
            obs[INF_OBS_WAVE_PHASE], (float)(phase + 1) / 8.0f, 1e-6f);
    }

    InfernoState triple_jad = make_test_state(20, 20);
    triple_jad.wave = 67;
    triple_jad.player.current_hitpoints = 99;
    triple_jad.player.base_hitpoints = 99;
    triple_jad.player.base_prayer = 99;
    triple_jad.player.current_prayer = 99;
    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&triple_jad, (EncounterContext*)&test_context, obs);
    ASSERT_FLOAT_NEAR("wave 68 stays in Jad phase",
        obs[INF_OBS_WAVE_PHASE], 5.0f / 8.0f, 1e-6f);
}

static void test_inferno_obs_exposes_compact_pillars(void) {
    printf("--- inferno obs exposes compact pillars ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player.current_hitpoints = 99;
    state.player.base_hitpoints = 99;
    state.player.base_prayer = 99;
    state.player.current_prayer = 99;
    state.pillars[0] = (InfPillar){
        .x = INF_PILLAR_POS[0][0],
        .y = INF_PILLAR_POS[0][1],
        .hp = INF_PILLAR_HP,
        .active = 1,
    };
    state.pillars[1] = (InfPillar){
        .x = INF_PILLAR_POS[1][0],
        .y = INF_PILLAR_POS[1][1],
        .hp = 0,
        .active = 0,
    };

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    int active_start = inferno_pillar_obs_start(0);
    ASSERT_FLOAT_NEAR("active pillar hp",
        obs[active_start], 1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("active pillar relative x",
        obs[active_start + 1],
        (float)(INF_PILLAR_POS[0][0] - state.player.x) /
            (float)INF_ARENA_WIDTH,
        1e-6f);
    ASSERT_FLOAT_NEAR("active pillar relative y",
        obs[active_start + 2],
        (float)(INF_PILLAR_POS[0][1] - state.player.y) /
            (float)INF_ARENA_HEIGHT,
        1e-6f);

    int inactive_start = inferno_pillar_obs_start(1);
    ASSERT_FLOAT_NEAR("inactive pillar hp",
        obs[inactive_start], 0.0f, 1e-6f);
}

static void test_inferno_obs_exposes_meleer_dig_state(void) {
    printf("--- inferno obs exposes meleer dig state ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player.current_hitpoints = 99;
    state.player.base_hitpoints = 99;
    state.player.base_prayer = 99;
    state.player.current_prayer = 99;
    state.npcs[0] = make_test_npc(
        INF_NPC_MELEER, 24, 20, INF_NPC_STATS[INF_NPC_MELEER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_MELEER].hp;
    state.npcs[0].no_los_ticks = 25;
    state.npcs[0].dig_freeze_timer = 3;
    state.npcs[0].dig_attack_delay = 6;

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    int meleer_slot = inf_find_target_obs_slot(&state, 0);
    int dig_start = inferno_obs_slot_dig_index(meleer_slot);
    ASSERT_INT_EQ("meleer occupies first meleer slot",
        state.current_obs_slots[meleer_slot], 0);
    ASSERT_FLOAT_NEAR("meleer no-los dig progress",
        obs[dig_start], 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR("meleer emerge timer",
        obs[dig_start + 1], 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR("meleer post-dig attack delay",
        obs[dig_start + 2], 1.0f, 1e-6f);

}

static void test_jad_special_wave_spawn_cadence_matches_reference(void) {
    printf("--- jad special wave spawn cadence matches reference ---\n");

    InfernoState single = make_test_state(0, 0);
    single.wave = 66;
    inf_spawn_wave(&single);

    int single_jad = -1;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (single.npcs[i].active && single.npcs[i].type == INF_NPC_JAD) {
            single_jad = i;
            break;
        }
    }
    ASSERT_INT_EQ("wave 67 spawns one jad", single_jad >= 0, 1);
    ASSERT_INT_EQ("wave 67 jad stun", single.npcs[single_jad].stun_timer, 1);
    ASSERT_INT_EQ("wave 67 jad attack speed timer", single.npcs[single_jad].attack_timer, 8);

    InfernoState triple = make_test_state(0, 0);
    triple.wave = 67;
    triple.rng_state = 12345;
    inf_spawn_wave(&triple);

    int num_jads = 0;
    int stun_sum = 0;
    int stun_product = 1;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (!triple.npcs[i].active || triple.npcs[i].type != INF_NPC_JAD)
            continue;
        num_jads++;
        stun_sum += triple.npcs[i].stun_timer;
        stun_product *= triple.npcs[i].stun_timer;
        ASSERT_INT_EQ("wave 68 jad attack timer includes stun offset",
            triple.npcs[i].attack_timer, 9 + triple.npcs[i].stun_timer);
    }
    ASSERT_INT_EQ("wave 68 spawns three jads", num_jads, 3);
    ASSERT_INT_EQ("wave 68 shuffled stun sum", stun_sum, 12);
    ASSERT_INT_EQ("wave 68 shuffled stun product", stun_product, 28);
}

static void test_triple_jad_first_attacks_are_staggered(void) {
    printf("--- triple jad first attacks are staggered ---\n");

    InfernoState state;
    memset(&state, 0, sizeof(state));
    test_config()->start_wave = 67;
    test_config()->late_start_supply_profile_scale = 1.0f;
    inf_reset_ctx((EncounterState*)&state, (EncounterContext*)&test_context, 12345);
    state.wave_ready_delay = 0;

    int jad_slots[3] = { -1, -1, -1 };
    int first_attack_ticks[3] = { 0, 0, 0 };
    int num_jads = 0;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (!state.npcs[i].active || state.npcs[i].type != INF_NPC_JAD)
            continue;
        if (num_jads < 3) jad_slots[num_jads] = i;
        num_jads++;
    }
    ASSERT_INT_EQ("wave 68 first-attack test has three jads", num_jads, 3);

    for (int tick = 1; tick <= 32; tick++) {
        step_inferno_noop(&state);
        for (int j = 0; j < 3; j++) {
            int slot = jad_slots[j];
            if (slot < 0) continue;
            if (first_attack_ticks[j] == 0 && state.npcs[slot].attacked_this_tick)
                first_attack_ticks[j] = tick;
        }
    }

    for (int a = 0; a < 2; a++) {
        for (int b = a + 1; b < 3; b++) {
            if (first_attack_ticks[b] < first_attack_ticks[a]) {
                int tmp = first_attack_ticks[a];
                first_attack_ticks[a] = first_attack_ticks[b];
                first_attack_ticks[b] = tmp;
            }
        }
    }

    ASSERT_INT_EQ("wave 68 first jad attacks", first_attack_ticks[0] > 0, 1);
    ASSERT_INT_EQ("wave 68 second jad attacks", first_attack_ticks[1] > 0, 1);
    ASSERT_INT_EQ("wave 68 third jad attacks", first_attack_ticks[2] > 0, 1);
    ASSERT_INT_EQ("wave 68 first attack gap 1", first_attack_ticks[1] - first_attack_ticks[0], 3);
    ASSERT_INT_EQ("wave 68 first attack gap 2", first_attack_ticks[2] - first_attack_ticks[1], 3);
}

static void test_jad_melee_stays_instant_and_untelegraphed(void) {
    printf("--- jad melee stays instant and untelegraphed ---\n");

    InfernoState preview_state = make_test_state(5, 5);
    preview_state.player.current_defence = 99;
    preview_state.player.current_magic = 99;
    preview_state.player.prayer = PRAYER_NONE;
    preview_state.weapon_set = INF_GEAR_MAGE;
    preview_state.wave = 66;

    preview_state.npcs[0] = make_test_npc(
        INF_NPC_JAD, 6, 5, INF_NPC_STATS[INF_NPC_JAD].size);
    preview_state.npcs[0].active = 1;
    preview_state.npcs[0].attack_timer = 1;
    inf_npc_jad(&preview_state.npcs[0])->attack_style = ATTACK_STYLE_RANGED;

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&preview_state, (EncounterContext*)&test_context, obs);

    int preview_slot = inf_find_target_obs_slot(&preview_state, 0);
    ASSERT_FLOAT_NEAR("Jad dynamic style stays ranged in its dense record",
        obs[inferno_obs_slot_start(preview_slot) + 5],
        (float)ATTACK_STYLE_RANGED / 4.0f, 1e-6f);

    int saw_melee = 0;
    for (uint32_t seed = 0; seed < 256; seed++) {
        InfernoState attack_state = make_test_state(5, 5);
        attack_state.rng_state = seed;
        attack_state.player.current_defence = 99;
        attack_state.player.current_magic = 99;
        attack_state.player.prayer = PRAYER_NONE;
        attack_state.weapon_set = INF_GEAR_MAGE;
        attack_state.wave = 66;

        attack_state.npcs[0] = make_test_npc(
            INF_NPC_JAD, 6, 5, INF_NPC_STATS[INF_NPC_JAD].size);
        attack_state.npcs[0].active = 1;
        attack_state.npcs[0].attack_timer = 0;
        inf_npc_jad(&attack_state.npcs[0])->attack_style = ATTACK_STYLE_RANGED;

        inf_npc_attack_ctx(&attack_state, &test_context, 0);

        if (attack_state.npcs[0].attack_style_this_tick == ATTACK_STYLE_MELEE) {
            saw_melee = 1;
            ASSERT_INT_EQ(
                "jad melee fallback does not queue a pending hit",
                attack_state.player_pending_hits.count, 0);
            break;
        }
    }

    ASSERT_INT_EQ("jad can still choose melee instantly at fire time", saw_melee, 1);
}

static int inferno_obs_slot_start(int slot_idx) {
    return INF_OBS_AFTER_PILLARS +
        slot_idx * INF_NPC_SLOT_FEATURES;
}

static int inferno_pillar_obs_start(int pillar_idx) {
    return INF_OBS_AFTER_ENCOUNTER + pillar_idx * INF_PILLAR_FEATURES;
}

static int inferno_target_mask_slot_offset(int slot_idx) {
    return inf_primary_attack_action_for_obs_slot(slot_idx);
}

static int inferno_target_mask_none_offset(void) {
    return 0;
}


static int inferno_pending_hit_obs_start(void) {
    return INF_OBS_AFTER_NPCS;
}

static int inferno_spark_obs_start(void) {
    return INF_OBS_AFTER_PENDING_HITS;
}

static int inferno_obs_slot_hp_index(int slot_idx) {
    return inferno_obs_slot_start(slot_idx) + 1;
}

static int inferno_obs_slot_npc_los_index(int slot_idx) {
    return inferno_obs_slot_start(slot_idx) + 6;
}

static int inferno_obs_slot_frozen_index(int slot_idx) {
    return inferno_obs_slot_start(slot_idx) + 7;
}

static int inferno_obs_slot_target_category_start(int slot_idx) {
    return inferno_obs_slot_start(slot_idx) + 8;
}


static int inferno_obs_slot_dig_index(int slot_idx) {
    return inferno_obs_slot_start(slot_idx) + 10;
}

static void init_threat_obs_state(
    InfernoState* state,
    int player_x,
    int player_y
) {
    inf_build_npc_stats();
    if (player_x == 10 && player_y == 10) {
        player_x += 10;
        player_y += 10;
    }
    *state = make_test_state(player_x, player_y);
    state->player.entity_type = ENTITY_PLAYER;
    state->player.base_hitpoints = 99;
    state->player.current_hitpoints = 99;
    state->player.base_attack = 99;
    state->player.base_strength = 99;
    state->player.base_defence = 99;
    state->player.base_ranged = 99;
    state->player.base_magic = 99;
    state->player.current_attack = 99;
    state->player.current_strength = 99;
    state->player.current_defence = 99;
    state->player.current_ranged = 99;
    state->player.current_magic = 99;
    state->weapon_set = INF_GEAR_BP;
    osrs_interaction_init(&state->interaction);
    encounter_compute_loadout_stats(
        INF_MAX_MAGE_LOADOUT, ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_AUTOCAST, 30,
        &state->loadout_stats[INF_GEAR_MAGE]);
    encounter_compute_loadout_stats(
        INF_MAX_RANGE_LONG_LOADOUT, ATTACK_STYLE_RANGED,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_RAPID, 0,
        &state->loadout_stats[INF_GEAR_LONG_RANGE]);
    encounter_compute_loadout_stats(
        INF_MAX_RANGE_FAST_LOADOUT, ATTACK_STYLE_RANGED,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_RAPID, 0,
        &state->loadout_stats[INF_GEAR_BP]);
}

static void add_threat_obs_npc(
    InfernoState* state,
    int slot,
    InfNPCType type,
    int x,
    int y
) {
    if (state->player.x == 20 && state->player.y == 20 &&
            x < 20 && y < 20) {
        x += 10;
        y += 10;
    }
    state->npcs[slot] =
        make_test_npc(type, x, y, INF_NPC_STATS[type].size);
    state->npcs[slot].active = 1;
    state->npcs[slot].hp =
        state->npcs[slot].max_hp = INF_NPC_STATS[type].hp;
    state->npcs[slot].attack_timer = 1;
}

static void test_npc_threat_obs_exposes_frozen_meleer_pressure(void) {
    printf("--- npc threat obs exposes frozen meleer pressure ---\n");

    InfernoState state;
    init_threat_obs_state(&state, 10, 10);
    add_threat_obs_npc(&state, 0, INF_NPC_MELEER, 11, 10);
    state.npcs[0].frozen_ticks = 8;
    inf_refresh_current_obs_slots_ctx(&state, &test_context);

    InfNpcPlayerThreat threat = inf_npc_player_threat_ctx(&state, &test_context, &state.npcs[0]);
    ASSERT_INT_EQ("frozen adjacent meleer can attack if ready",
        threat.can_attack_if_ready, 1);
    ASSERT_INT_EQ("frozen adjacent meleer can attack this tick",
        threat.can_attack_this_tick, 1);

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);
    int obs_slot = inf_find_target_obs_slot(&state, 0);
    ASSERT_INT_EQ("frozen meleer has obs slot", obs_slot >= 0, 1);
    ASSERT_FLOAT_NEAR("frozen ticks obs",
        obs[inferno_obs_slot_frozen_index(obs_slot)],
        8.0f / (float)BARRAGE_FREEZE_TICKS, 1e-6f);
}

static void test_npc_threat_obs_respects_overlap_range_and_stun(void) {
    printf("--- npc threat obs respects overlap range and stun ---\n");

    InfernoState under;
    init_threat_obs_state(&under, 10, 10);
    add_threat_obs_npc(&under, 0, INF_NPC_MELEER, 9, 9);
    under.npcs[0].frozen_ticks = 8;
    InfNpcPlayerThreat under_threat = inf_npc_player_threat_ctx(&under, &test_context, &under.npcs[0]);
    ASSERT_INT_EQ("standing under frozen meleer is not attackable",
        under_threat.can_attack_if_ready, 0);

    InfernoState diagonal;
    init_threat_obs_state(&diagonal, 10, 10);
    add_threat_obs_npc(&diagonal, 0, INF_NPC_MELEER, 11, 11);
    diagonal.npcs[0].frozen_ticks = 8;
    InfNpcPlayerThreat diagonal_threat =
        inf_npc_player_threat_ctx(&diagonal, &test_context, &diagonal.npcs[0]);
    ASSERT_INT_EQ("frozen meleer diagonal corner contact is not attackable",
        diagonal_threat.can_attack_if_ready, 0);

    InfernoState far;
    init_threat_obs_state(&far, 10, 10);
    add_threat_obs_npc(&far, 0, INF_NPC_MELEER, 13, 10);
    far.npcs[0].frozen_ticks = 8;
    InfNpcPlayerThreat far_threat = inf_npc_player_threat_ctx(&far, &test_context, &far.npcs[0]);
    ASSERT_INT_EQ("frozen meleer outside melee distance is not attackable",
        far_threat.can_attack_if_ready, 0);

    InfernoState stunned;
    init_threat_obs_state(&stunned, 10, 10);
    add_threat_obs_npc(&stunned, 0, INF_NPC_MELEER, 11, 10);
    stunned.npcs[0].stun_timer = 2;
    InfNpcPlayerThreat stunned_threat =
        inf_npc_player_threat_ctx(&stunned, &test_context, &stunned.npcs[0]);
    ASSERT_INT_EQ("stunned adjacent meleer would threaten if ready",
        stunned_threat.can_attack_if_ready, 1);
    ASSERT_INT_EQ("stunned adjacent meleer cannot attack this tick",
        stunned_threat.can_attack_this_tick, 0);
}


static void test_npc_threat_obs_keeps_ranger_mager_diagonal_melee(void) {
    printf("--- npc threat obs keeps ranger and mager diagonal melee ---\n");

    InfernoState ranger_state;
    init_threat_obs_state(&ranger_state, 10, 10);
    add_threat_obs_npc(&ranger_state, 0, INF_NPC_RANGER, 11, 11);
    InfNpcPlayerThreat ranger_threat =
        inf_npc_player_threat_ctx(&ranger_state, &test_context, &ranger_state.npcs[0]);
    ASSERT_INT_EQ("diagonal ranger can attack player",
        ranger_threat.can_attack_if_ready, 1);
    ASSERT_INT_EQ("diagonal ranger threat includes melee fallback",
        (ranger_threat.style_mask & INF_STYLE_MASK_MELEE) != 0, 1);

    InfernoState mager_state;
    init_threat_obs_state(&mager_state, 10, 10);
    add_threat_obs_npc(&mager_state, 0, INF_NPC_MAGER, 11, 11);
    InfNpcPlayerThreat mager_threat =
        inf_npc_player_threat_ctx(&mager_state, &test_context, &mager_state.npcs[0]);
    ASSERT_INT_EQ("diagonal mager can attack player",
        mager_threat.can_attack_if_ready, 1);
    ASSERT_INT_EQ("diagonal mager threat includes melee fallback",
        (mager_threat.style_mask & INF_STYLE_MASK_MELEE) != 0, 1);
}


static void init_step_out_forecast_stack_state(
    InfernoState* state,
    int player_x,
    int player_y
) {
    reset_test_context();
    inf_build_npc_stats();
    memset(state, 0, sizeof(*state));
    state->rng_state = 20260515u;
    state->wave = 59;
    state->player.entity_type = ENTITY_PLAYER;
    state->player.x = player_x;
    state->player.y = player_y;
    state->player.base_hitpoints = 99;
    state->player.current_hitpoints = 99;
    state->player.current_defence = 99;
    state->player.current_magic = 99;
    state->player.current_prayer = 99;
    state->player_last_interaction_target_slot = -1;
    state->player_last_interaction_age = 1;
    state->player_dest_x = -1;
    state->player_dest_y = -1;
    state->weapon_set = INF_GEAR_LONG_RANGE;
    osrs_interaction_init(&state->interaction);
    for (int p = 0; p < INF_NUM_PILLARS; p++) {
        state->pillars[p].x = INF_PILLAR_POS[p][0];
        state->pillars[p].y = INF_PILLAR_POS[p][1];
        state->pillars[p].hp = INF_PILLAR_HP;
        state->pillars[p].active = 1;
    }
}

static void add_step_out_forecast_npc(
    InfernoState* state, int slot, InfNPCType type, int x, int y, int timer
) {
    inf_init_npc(state, slot, type, x, y);
    state->npcs[slot].attack_timer = timer;
    state->npcs[slot].stun_timer = 0;
    state->npcs[slot].frozen_ticks = 0;
}


static void assert_step_out_ranger_then_mager(
    const char* label,
    const InfStepOutForecastAction* action,
    int expected_x,
    int expected_y
) {
    char msg[128];
    snprintf(msg, sizeof(msg), "%s action is valid", label);
    ASSERT_INT_EQ(msg, action->valid, 1);
    snprintf(msg, sizeof(msg), "%s landing x", label);
    ASSERT_INT_EQ(msg, action->land_x, expected_x);
    snprintf(msg, sizeof(msg), "%s landing y", label);
    ASSERT_INT_EQ(msg, action->land_y, expected_y);
    snprintf(msg, sizeof(msg), "%s ranger fires first", label);
    ASSERT_INT_EQ(msg, action->ticks[0].ranger_count, 1);
    snprintf(msg, sizeof(msg), "%s mager waits first tick", label);
    ASSERT_INT_EQ(msg, action->ticks[0].mager_count, 0);
    snprintf(msg, sizeof(msg), "%s mager fires second", label);
    ASSERT_INT_EQ(msg, action->ticks[1].mager_count, 1);
    snprintf(msg, sizeof(msg), "%s ranger does not double-fire", label);
    ASSERT_INT_EQ(msg, action->ticks[1].ranger_count, 0);
    snprintf(msg, sizeof(msg), "%s exposes off-tick opportunity", label);
    ASSERT_INT_EQ(msg, action->ranger_mager_offtick_opportunity, 1);
    snprintf(msg, sizeof(msg), "%s avoids same-tick conflict", label);
    ASSERT_INT_EQ(msg, action->same_tick_mixed_style_conflict, 0);
}

static void test_step_out_forecast_matches_movement_head_destinations(void) {
    printf("--- step-out forecast matches movement head destinations ---\n");

    InfernoState state;
    init_step_out_forecast_stack_state(&state, 29, 39);

    InfStepOutForecast forecast;
    inf_build_step_out_forecast_ctx(&state, &test_context, &forecast);

    for (int action = 0; action < ENCOUNTER_MOVE_ACTIONS; action++) {
        Player moved = state.player;
        if (action > 0) {
            InfWalkCtx walk_ctx = { &state, &test_context };
            encounter_move_to_target(
                &moved,
                ENCOUNTER_MOVE_TARGET_DX[action],
                ENCOUNTER_MOVE_TARGET_DY[action],
                inf_tile_walkable,
                &walk_ctx);
        }

        ASSERT_INT_EQ("forecast movement landing x",
            forecast.actions[action].land_x, moved.x);
        ASSERT_INT_EQ("forecast movement landing y",
            forecast.actions[action].land_y, moved.y);
    }
}

static void assert_inferno_npc_sw_origin_step(
    const char* label,
    int pillar_idx,
    InfNPCType type,
    int player_x,
    int player_y,
    int npc_x,
    int npc_y,
    int expected_x,
    int expected_y
) {
    InfernoState state;
    init_step_out_forecast_stack_state(&state, player_x, player_y);
    for (int p = 0; p < INF_NUM_PILLARS; p++) {
        state.pillars[p].active = p == pillar_idx;
        state.pillars[p].hp = p == pillar_idx ? INF_PILLAR_HP : 0;
    }
    add_step_out_forecast_npc(&state, 0, type, npc_x, npc_y, 0);
    ASSERT_INT_EQ("starting NPC has no LOS",
        inf_npc_has_los_direct_ctx(
            &state, &test_context, 0), 0);

    inf_npc_move_ctx(&state, &test_context, 0);

    char msg[128];
    snprintf(msg, sizeof(msg), "%s x", label);
    ASSERT_INT_EQ(msg, state.npcs[0].x, expected_x);
    snprintf(msg, sizeof(msg), "%s y", label);
    ASSERT_INT_EQ(msg, state.npcs[0].y, expected_y);
}

static void test_inferno_npc_travel_uses_sw_origin_around_all_pillars(void) {
    printf("--- inferno NPC travel uses SW origin around all pillars ---\n");

    assert_inferno_npc_sw_origin_step(
        "south pillar mager",
        0, INF_NPC_MAGER,
        15, 14,
        12, 30,
        13, 29);
    assert_inferno_npc_sw_origin_step(
        "west pillar ranger",
        1, INF_NPC_RANGER,
        12, 28,
        11, 37,
        12, 37);
    assert_inferno_npc_sw_origin_step(
        "north pillar ranger",
        2, INF_NPC_RANGER,
        18, 40,
        16, 22,
        17, 23);
}

static void assert_inferno_jal_npc_uses_edge_clearance(
    const char* label,
    InfNPCType type,
    int player_x,
    int player_y,
    int npc_x,
    int npc_y,
    int expected_x,
    int expected_y
) {
    InfernoState state;
    init_step_out_forecast_stack_state(&state, player_x, player_y);
    for (int p = 0; p < INF_NUM_PILLARS; p++) {
        state.pillars[p].active = p == 0;
        state.pillars[p].hp = p == 0 ? INF_PILLAR_HP : 0;
    }
    add_step_out_forecast_npc(&state, 0, type, npc_x, npc_y, 0);
    ASSERT_INT_EQ("starting Jal NPC has no LOS",
        inf_npc_has_los_ctx(&state, &test_context, 0), 0);

    inf_npc_move_ctx(&state, &test_context, 0);

    char msg[128];
    snprintf(msg, sizeof(msg), "%s x", label);
    ASSERT_INT_EQ(msg, state.npcs[0].x, expected_x);
    snprintf(msg, sizeof(msg), "%s y", label);
    ASSERT_INT_EQ(msg, state.npcs[0].y, expected_y);
}

static void test_inferno_jal_npcs_use_edge_clearance_at_pillars(void) {
    printf("--- inferno Jal NPCs use edge clearance at pillars ---\n");

    assert_inferno_jal_npc_uses_edge_clearance(
        "JalXil south pillar corner",
        INF_NPC_RANGER,
        21, 16,
        20, 20,
        20, 19);
    assert_inferno_jal_npc_uses_edge_clearance(
        "JalZek south pillar corner",
        INF_NPC_MAGER,
        21, 20,
        17, 16,
        18, 16);
}

static void test_step_out_forecast_north_pillar_ranger_mager_order(void) {
    printf("--- step-out forecast north pillar ranger/mager order ---\n");

    InfernoState state;
    init_step_out_forecast_stack_state(&state, 29, 39);
    add_step_out_forecast_npc(&state, 0, INF_NPC_RANGER, 24, 31, 0);
    add_step_out_forecast_npc(&state, 1, INF_NPC_MAGER, 29, 30, 0);

    InfStepOutForecast forecast;
    inf_build_step_out_forecast_ctx(&state, &test_context, &forecast);

    const InfStepOutForecastAction* idle = &forecast.actions[0];
    ASSERT_INT_EQ("idle remains safe from ranged tick one",
        idle->ticks[0].ranged_count, 0);
    ASSERT_INT_EQ("idle remains safe from magic tick one",
        idle->ticks[0].magic_count, 0);

    const InfStepOutForecastAction* run_west = &forecast.actions[11];
    assert_step_out_ranger_then_mager("north pillar run west", run_west, 27, 39);
}

static void test_step_out_forecast_south_pillar_ranger_mager_order(void) {
    printf("--- step-out forecast south pillar ranger/mager order ---\n");

    InfernoState state;
    init_step_out_forecast_stack_state(&state, 22, 17);
    add_step_out_forecast_npc(&state, 0, INF_NPC_RANGER, 17, 25, 0);
    add_step_out_forecast_npc(&state, 1, INF_NPC_MAGER, 22, 26, 0);

    InfStepOutForecast forecast;
    inf_build_step_out_forecast_ctx(&state, &test_context, &forecast);

    const InfStepOutForecastAction* run_west = &forecast.actions[11];
    assert_step_out_ranger_then_mager("south pillar run west", run_west, 20, 17);
}

static void test_step_out_forecast_west_pillar_ranger_mager_order(void) {
    printf("--- step-out forecast west pillar ranger/mager order ---\n");

    InfernoState state;
    init_step_out_forecast_stack_state(&state, 11, 29);
    add_step_out_forecast_npc(&state, 0, INF_NPC_RANGER, 8, 40, 0);
    add_step_out_forecast_npc(&state, 1, INF_NPC_MAGER, 16, 42, 0);

    InfStepOutForecast forecast;
    inf_build_step_out_forecast_ctx(&state, &test_context, &forecast);

    const InfStepOutForecastAction* walk_north = &forecast.actions[4];
    assert_step_out_ranger_then_mager("west pillar walk north", walk_north, 11, 28);
}

static void test_step_out_forecast_inactive_pillar_does_not_create_cover(void) {
    printf("--- step-out forecast inactive pillar does not create cover ---\n");

    InfernoState state;
    init_step_out_forecast_stack_state(&state, 29, 39);
    state.pillars[2].active = 0;
    state.pillars[2].hp = 0;
    add_step_out_forecast_npc(
        &state, 0, INF_NPC_RANGER, 24, 31, 0);
    add_step_out_forecast_npc(
        &state, 1, INF_NPC_MAGER, 29, 30, 0);

    InfStepOutForecast forecast;
    inf_build_step_out_forecast_ctx(&state, &test_context, &forecast);

    const InfStepOutForecastAction* idle = &forecast.actions[0];
    ASSERT_INT_EQ("inactive north pillar exposes ranger immediately",
        idle->ticks[0].ranger_count, 1);
    ASSERT_INT_EQ("inactive north pillar exposes mager immediately",
        idle->ticks[0].mager_count, 1);
    ASSERT_INT_EQ("inactive north pillar shows same-tick conflict",
        idle->same_tick_mixed_style_conflict, 1);
    ASSERT_INT_EQ("inactive north pillar has no off-tick cover",
        idle->ranger_mager_offtick_opportunity, 0);
}

static void test_step_out_same_tick_ranger_mager_event_logs(void) {
    printf("--- step-out same-tick ranger/mager event logs ---\n");

    InfernoState state;
    init_step_out_forecast_stack_state(&state, 14, 35);
    add_step_out_forecast_npc(
        &state, 0, INF_NPC_RANGER, 5, 39, 0);
    add_step_out_forecast_npc(
        &state, 1, INF_NPC_MAGER, 4, 34, 0);

    int actions[INF_NUM_ACTION_HEADS] = {0};
    actions[INF_HEAD_PRIMARY] = 13;
    inf_step_ctx((EncounterState*)&state, (EncounterContext*)&test_context, actions);

    ASSERT_INT_EQ("step-out tick moved the player",
        state.tick_scratch.player_moved, 1);
    ASSERT_INT_EQ(
        "movement tick does not count attacks before NPCs see new tile",
        state.total_step_out_ranger_mager_same_tick_attacks, 0);

    int noop[INF_NUM_ACTION_HEADS] = {0};
    inf_step_ctx((EncounterState*)&state, (EncounterContext*)&test_context, noop);

    ASSERT_INT_EQ("same-tick ranger/mager event counted",
        state.total_ranger_mager_same_tick_attacks, 1);
    ASSERT_INT_EQ("step-out same-tick ranger/mager event counted",
        state.total_step_out_ranger_mager_same_tick_attacks, 1);
}

static void test_direct_start_waves_spawn_without_empty_gap(void) {
    printf("--- direct start waves spawn without empty gap ---\n");

    EncounterState* raw = inf_create();
    reset_inferno_at_public_wave(raw, 20, 1.0f);
    InfernoState* regular = (InfernoState*)raw;

    ASSERT_INT_EQ("late regular start has no empty wave delay",
        regular->wave_spawn_delay, 0);
    ASSERT_INT_EQ("late regular start spawns mobs immediately",
        count_active_npcs(regular) > 0, 1);

    reset_inferno_at_public_wave(raw, 69, 1.0f);
    InfernoState* zuk = (InfernoState*)raw;

    ASSERT_INT_EQ("zuk start has no empty wave delay", zuk->wave_spawn_delay, 0);
    ASSERT_INT_EQ("zuk spawned immediately", find_active_npc_type(zuk, INF_NPC_ZUK) >= 0, 1);
    ASSERT_INT_EQ("zuk shield spawned immediately",
        find_active_npc_type(zuk, INF_NPC_ZUK_SHIELD) >= 0, 1);
    ASSERT_INT_EQ("zuk start player x", zuk->player.x, INF_ZUK_PLAYER_START_X);
    ASSERT_INT_EQ("zuk start player y", zuk->player.y, INF_ZUK_PLAYER_START_Y);

    inf_destroy(raw);
}

static void test_joseph_start_wave_70_seeds_zuk_jad_checkpoint(void) {
    printf("--- Joseph start wave 70 seeds Zuk Jad checkpoint ---\n");

    EncounterState* raw = inf_create();
    reset_inferno_at_public_wave(raw, 70, 1.0f);
    InfernoState* state = (InfernoState*)raw;
    int zuk_idx = find_active_npc_type(state, INF_NPC_ZUK);
    int jad_idx = find_active_npc_type(state, INF_NPC_JAD);
    int shield_idx = find_active_npc_type(state, INF_NPC_ZUK_SHIELD);

    ASSERT_INT_EQ("runtime wave is Zuk", state->wave, INF_NUM_WAVES - 1);
    ASSERT_INT_EQ("Zuk exists", zuk_idx >= 0, 1);
    ASSERT_INT_EQ("Zuk starts under Jad threshold", state->npcs[zuk_idx].hp, 479);
    ASSERT_INT_EQ("Jad checkpoint spawned Jad", jad_idx >= 0, 1);
    ASSERT_INT_EQ("Jad targets shield", state->npcs[jad_idx].aggro_target, shield_idx);
    ASSERT_INT_EQ("Zuk healers not spawned yet", state->zuk.healer_spawned, 0);

    inf_destroy(raw);
}

static void test_joseph_start_wave_71_seeds_zuk_healer_checkpoint(void) {
    printf("--- Joseph start wave 71 seeds Zuk healer checkpoint ---\n");

    EncounterState* raw = inf_create();
    reset_inferno_at_public_wave(raw, 71, 1.0f);
    InfernoState* state = (InfernoState*)raw;
    int zuk_idx = find_active_npc_type(state, INF_NPC_ZUK);

    ASSERT_INT_EQ("runtime wave is Zuk", state->wave, INF_NUM_WAVES - 1);
    ASSERT_INT_EQ("Zuk exists", zuk_idx >= 0, 1);
    ASSERT_INT_EQ("Zuk starts under healer threshold", state->npcs[zuk_idx].hp, 239);
    ASSERT_INT_EQ("Zuk healer checkpoint marks spawned", state->zuk.healer_spawned, 1);
    ASSERT_INT_EQ("Zuk healer count", count_active_npc_type(state, INF_NPC_HEALER_ZUK), 4);
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (state->npcs[i].active && state->npcs[i].type == INF_NPC_HEALER_ZUK) {
            ASSERT_INT_EQ("Zuk healer targets Zuk", state->npcs[i].aggro_target, zuk_idx);
        }
    }

    inf_destroy(raw);
}

static void test_zuk_ready_countdown_holds_npcs_then_releases(void) {
    printf("--- zuk ready countdown holds npcs then releases ---\n");

    EncounterState* raw = inf_create();
    reset_inferno_at_public_wave(raw, 69, 1.0f);
    InfernoState* state = (InfernoState*)raw;
    int zuk_idx = find_active_npc_type(state, INF_NPC_ZUK);

    ASSERT_INT_EQ("zuk exists during ready countdown", zuk_idx >= 0, 1);
    ASSERT_INT_EQ("zuk attack timer starts at reference delay",
        state->npcs[zuk_idx].attack_timer, 14);

    for (int i = 0; i < 5; i++)
        step_inferno_noop(state);
    ASSERT_INT_EQ("ready countdown does not tick zuk early",
        state->npcs[zuk_idx].attack_timer, 14);

    step_inferno_noop(state);
    ASSERT_INT_EQ("zuk attack timer starts once ready countdown clears",
        state->npcs[zuk_idx].attack_timer, 13);

    inf_destroy(raw);
}

static void init_zuk_timing_state(InfernoState* state) {
    reset_test_context();
    memset(state, 0, sizeof(*state));
    state->rng_state = 7;
    state->wave = 68;
    state->player.entity_type = ENTITY_PLAYER;
    state->player.x = INF_ZUK_PLAYER_START_X;
    state->player.y = INF_ZUK_PLAYER_START_Y;
    state->player.base_hitpoints = 99;
    state->player.current_hitpoints = 99;
    state->player.base_prayer = 99;
    state->player.current_prayer = 99;
    state->player.base_defence = 99;
    state->player.current_defence = 99;
    state->player.base_magic = 99;
    state->player.current_magic = 99;
    state->player.base_ranged = 99;
    state->player.current_ranged = 99;
    state->player_dest_x = -1;
    state->player_dest_y = -1;
    state->player_last_interaction_target_slot = -1;
    state->player_last_interaction_age = 1;
    state->tick_at_first_zuk_healer_target = -1;
    state->tick_at_first_zuk_healer_attack = -1;
    state->weapon_set = INF_GEAR_LONG_RANGE;
    osrs_interaction_init(&state->interaction);
    encounter_apply_loadout(&state->player, INF_MAX_RANGE_LONG_LOADOUT, GEAR_RANGED);
    inf_refresh_live_stats(state);
    encounter_compute_loadout_stats(INF_MAX_RANGE_LONG_LOADOUT, ATTACK_STYLE_RANGED,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_RAPID, 0,
        &state->loadout_stats[INF_GEAR_LONG_RANGE]);

    state->npcs[0] = make_test_npc(
        INF_NPC_ZUK, INF_ZUK_X, INF_ZUK_Y, INF_NPC_STATS[INF_NPC_ZUK].size);
    state->npcs[0].active = 1;
    state->npcs[0].hp = state->npcs[0].max_hp = INF_NPC_STATS[INF_NPC_ZUK].hp;
    state->npcs[0].attack_timer = 14;
    state->npcs[0].stun_timer = 8;

    state->npcs[1] = make_test_npc(
        INF_NPC_ZUK_SHIELD, INF_ZUK_SHIELD_X, INF_ZUK_SHIELD_Y,
        INF_NPC_STATS[INF_NPC_ZUK_SHIELD].size);
    state->npcs[1].active = 1;
    state->npcs[1].hp = state->npcs[1].max_hp = INF_NPC_STATS[INF_NPC_ZUK_SHIELD].hp;

    state->zuk.shield_idx = 1;
    state->zuk.shield_dir = 1;
    state->zuk.set_timer = 72;
    state->zuk.set_interval = 350;
}

static void equip_zuk_timing_state_blowpipe(InfernoState* state) {
    state->weapon_set = INF_GEAR_BP;
    encounter_apply_loadout(&state->player, INF_MAX_RANGE_FAST_LOADOUT, GEAR_RANGED);
    inf_refresh_live_stats(state);
    encounter_compute_loadout_stats(INF_MAX_RANGE_FAST_LOADOUT, ATTACK_STYLE_RANGED,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_RAPID, 0,
        &state->loadout_stats[INF_GEAR_BP]);
}

static void test_zuk_shield_does_not_set_collision_flags(void) {
    printf("--- zuk shield does not set collision flags ---\n");

    ASSERT_INT_EQ("zuk shield follows reference CollisionType.NONE",
        inf_npc_sets_collision_flag(INF_NPC_ZUK_SHIELD), 0);
}

static void test_zuk_obs_exposes_attack_timer_summary(void) {
    printf("--- zuk obs exposes attack timer summary ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    state.npcs[0].attack_timer = 3;

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    ASSERT_FLOAT_NEAR("zuk attack timer uses compact player field",
        obs[INF_OBS_ZUK_ATTACK_TIMER], 0.3f, 1e-6f);
}

static void test_zuk_obs_exposes_pending_sparks(void) {
    printf("--- zuk obs exposes exact pending spark landings ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    state.pending_sparks[0] = (InfPendingSpark){
        .active = 1, .src_x = state.player.x + 5, .src_y = state.player.y,
        .x = state.player.x + 4, .y = state.player.y,
        .damage = 3, .ticks_remaining = 4,
    };
    state.pending_sparks[1] = (InfPendingSpark){
        .active = 1, .src_x = state.player.x + 5, .src_y = state.player.y,
        .x = state.player.x - 1, .y = state.player.y + 2,
        .damage = 7, .ticks_remaining = 2,
    };
    state.pending_sparks[2] = (InfPendingSpark){
        .active = 1, .src_x = state.player.x - 3, .src_y = state.player.y + 1,
        .x = state.player.x - 2, .y = state.player.y + 1,
        .damage = 5, .ticks_remaining = 3,
    };
    state.pending_sparks[3] = (InfPendingSpark){
        .active = 1, .src_x = state.player.x + 1, .src_y = state.player.y + 1,
        .x = state.player.x + 1, .y = state.player.y + 1,
        .damage = 6, .ticks_remaining = 4,
    };
    state.pending_sparks[4] = (InfPendingSpark){
        .active = 1, .src_x = state.player.x + 2, .src_y = state.player.y + 2,
        .x = state.player.x + 2, .y = state.player.y + 2,
        .damage = 8, .ticks_remaining = 5,
    };
    state.pending_sparks[5] = (InfPendingSpark){
        .active = 1, .src_x = state.player.x + 3, .src_y = state.player.y + 3,
        .x = state.player.x + 3, .y = state.player.y + 3,
        .damage = 9, .ticks_remaining = 6,
    };

    int spark_start = inferno_spark_obs_start();
    int spark_features = INF_FEATURES_PER_SPARK;
    int spark_slots = INF_SPARK_OBS_SLOTS;
    ASSERT_INT_EQ("inferno obs has full spark section",
        INF_NUM_OBS >= spark_start + spark_features * spark_slots, 1);
    if (INF_NUM_OBS < spark_start + spark_features * spark_slots)
        return;

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    ASSERT_INT_EQ("spark obs keeps all pending slots", spark_slots, INF_MAX_PENDING_SPARKS);
    ASSERT_INT_EQ("spark obs carries compact landing record", spark_features, 4);
    ASSERT_FLOAT_NEAR("first spark landing x",
        obs[spark_start], -1.0f / (float)INF_ARENA_WIDTH, 1e-6f);
    ASSERT_FLOAT_NEAR("first spark landing y",
        obs[spark_start + 1], 2.0f / (float)INF_ARENA_HEIGHT, 1e-6f);
    ASSERT_FLOAT_NEAR("first spark timer",
        obs[spark_start + 2], 0.2f, 1e-6f);
    ASSERT_FLOAT_NEAR("first spark damage",
        obs[spark_start + 3], 0.7f, 1e-6f);
    ASSERT_FLOAT_NEAR("second spark landing x",
        obs[spark_start + spark_features],
        -2.0f / (float)INF_ARENA_WIDTH, 1e-6f);
    ASSERT_FLOAT_NEAR("third spark sorts same-tick nearest landing first",
        obs[spark_start + 2 * spark_features],
        1.0f / (float)INF_ARENA_WIDTH, 1e-6f);
}

static void assert_human_blowpipe_zuk_chase_endpoint(
    int start_x, int start_y, int expected_x, int expected_y,
    const char* label
) {
    InfernoState endpoint_state;
    init_zuk_timing_state(&endpoint_state);
    equip_zuk_timing_state_blowpipe(&endpoint_state);
    endpoint_state.player.x = start_x;
    endpoint_state.player.y = start_y;
    endpoint_state.player.attack_timer = 3;
    endpoint_state.npcs[0].stun_timer = 64;
    endpoint_state.npcs[0].attack_timer = 64;

    ASSERT_INT_EQ("Zuk blowpipe endpoint starts out of range",
        inf_player_can_attack_npc_from_current_tile_ctx(
            &endpoint_state, &test_context, 0), 0);

    HumanInput hi = make_human_input();
    human_input_queue_attack_npc(&hi, 0);
    inf_step_human_commands_ctx((EncounterState*)&endpoint_state, (EncounterContext*)&test_context, &hi);
    human_input_destroy(&hi);

    for (int i = 0; i < 16; i++) {
        HumanInput empty = make_human_input();
        inf_step_human_commands_ctx((EncounterState*)&endpoint_state, (EncounterContext*)&test_context, &empty);
        human_input_destroy(&empty);
    }

    ASSERT_INT_EQ(label, endpoint_state.player.x, expected_x);
    ASSERT_INT_EQ("Zuk blowpipe endpoint y matches InfernoTrainer",
        endpoint_state.player.y, expected_y);
    ASSERT_INT_EQ("Zuk blowpipe endpoint remains out of range",
        inf_player_can_attack_npc_from_current_tile_ctx(
            &endpoint_state, &test_context, 0), 0);
    ASSERT_INT_EQ("Zuk blowpipe endpoint does not fire",
        endpoint_state.npcs[0].pending_hits.hits[0].active, 0);
    ASSERT_INT_EQ("Zuk blowpipe endpoint keeps interaction active",
        osrs_interaction_active(&endpoint_state.interaction), 1);
}

static void test_human_blowpipe_click_chases_zuk_out_of_range(void) {
    printf("--- human blowpipe click chases Zuk out of range ---\n");

    assert_human_blowpipe_zuk_chase_endpoint(
        11, 42, 22, INF_ARENA_MAX_Y,
        "left-edge Zuk blowpipe endpoint x matches InfernoTrainer");

    assert_human_blowpipe_zuk_chase_endpoint(
        39, 42, 28, INF_ARENA_MAX_Y,
        "right-edge Zuk blowpipe endpoint x matches InfernoTrainer");

    InfernoState edge_state;
    init_zuk_timing_state(&edge_state);
    equip_zuk_timing_state_blowpipe(&edge_state);
    edge_state.player.x = 25;
    edge_state.player.y = 42;
    edge_state.player.attack_timer = 3;
    edge_state.npcs[0].stun_timer = 64;
    edge_state.npcs[0].attack_timer = 64;

    ASSERT_INT_EQ("north-row Zuk click starts out of blowpipe range",
        inf_player_can_attack_npc_from_current_tile_ctx(
            &edge_state, &test_context, 0), 0);

    HumanInput edge_hi = make_human_input();
    human_input_queue_attack_npc(&edge_hi, 0);
    inf_step_human_commands_ctx((EncounterState*)&edge_state, (EncounterContext*)&test_context, &edge_hi);
    human_input_destroy(&edge_hi);

    ASSERT_INT_EQ("north-row Zuk click follows reference seek x", edge_state.player.x, 24);
    ASSERT_INT_EQ("north-row Zuk click walks to max north row",
        edge_state.player.y, INF_ARENA_MAX_Y);
    ASSERT_INT_EQ("north-row Zuk click remains outside blowpipe range",
        inf_player_can_attack_npc_from_current_tile_ctx(
            &edge_state, &test_context, 0), 0);
    ASSERT_INT_EQ("north-row Zuk cooldown prevents immediate hit",
        edge_state.npcs[0].pending_hits.hits[0].active, 0);
    ASSERT_INT_EQ("north-row Zuk interaction remains active",
        osrs_interaction_active(&edge_state.interaction), 1);

    InfernoState state;
    init_zuk_timing_state(&state);
    equip_zuk_timing_state_blowpipe(&state);
    state.player.x = 25;
    state.player.y = 37;
    state.player.attack_timer = 3;
    state.npcs[0].stun_timer = 64;
    state.npcs[0].attack_timer = 64;

    ASSERT_INT_EQ("human Zuk click starts out of blowpipe range",
        inf_player_can_attack_npc_from_current_tile_ctx(
            &state, &test_context, 0), 0);

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    HumanInput hi = make_human_input();
    human_input_queue_attack_npc(&hi, 0);
    inf_step_human_commands_ctx((EncounterState*)&state, (EncounterContext*)&test_context, &hi);
    human_input_destroy(&hi);

    ASSERT_INT_EQ("human Zuk click keeps interaction active",
        osrs_interaction_active(&state.interaction), 1);
    ASSERT_INT_EQ("human Zuk click targets Zuk", state.interaction.target_slot, 0);
    ASSERT_INT_EQ("human Zuk click keeps player x", state.player.x, 25);
    ASSERT_INT_EQ("human Zuk click walks north toward range", state.player.y > 37, 1);
    ASSERT_INT_EQ("human Zuk click stays inside arena", state.player.y <= INF_ARENA_MAX_Y, 1);
    ASSERT_INT_EQ("cooldown prevents immediate Zuk hit",
        state.npcs[0].pending_hits.hits[0].active, 0);
    ASSERT_INT_EQ("attack timer decrements after chase", state.player.attack_timer, 2);

    for (int i = 0; i < 8; i++) {
        HumanInput empty = make_human_input();
        inf_step_human_commands_ctx((EncounterState*)&state, (EncounterContext*)&test_context, &empty);
        human_input_destroy(&empty);
    }

    ASSERT_INT_EQ("human Zuk click ends at max north row", state.player.y, INF_ARENA_MAX_Y);
    ASSERT_INT_EQ("human Zuk click never reaches blowpipe attack tile",
        inf_player_can_attack_npc_from_current_tile_ctx(
            &state, &test_context, 0), 0);
    ASSERT_INT_EQ("human Zuk click does not fire unreachable blowpipe hit",
        state.npcs[0].pending_hits.hits[0].active, 0);
    ASSERT_INT_EQ("human Zuk click keeps interaction after chase",
        osrs_interaction_active(&state.interaction), 1);
}

static void test_zuk_healer_blowpipe_target_chases_out_of_range(void) {
    printf("--- Zuk healer blowpipe target chases out of range ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    equip_zuk_timing_state_blowpipe(&state);
    state.tick = 432;
    state.player.x = 25;
    state.player.y = 42;
    state.player.attack_timer = 3;

    state.zuk.healer_spawned = 1;
    state.npcs[2] = make_test_npc(
        INF_NPC_HEALER_ZUK, 20, 48, INF_NPC_STATS[INF_NPC_HEALER_ZUK].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[2].aggro_target = 0;

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);
    int healer_slot = inf_find_target_obs_slot(&state, 2);
    ASSERT_INT_EQ("healer appears in target obs", healer_slot >= 0, 1);

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_PRIMARY] = inf_primary_attack_action_for_obs_slot(healer_slot);
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("healer target keeps interaction active",
        osrs_interaction_active(&state.interaction), 1);
    ASSERT_INT_EQ("healer target selects healer", state.interaction.target_slot, 2);
    ASSERT_INT_EQ("healer target follows shortest seek x", state.player.x, 25);
    ASSERT_INT_EQ("healer target follows shortest seek y", state.player.y, 43);
    ASSERT_INT_EQ("healer target reaches blowpipe range",
        inf_player_can_attack_npc_from_current_tile_ctx(
            &state, &test_context, 2), 1);
    ASSERT_INT_EQ("cooldown prevents immediate healer hit",
        state.npcs[2].pending_hits.hits[0].active, 0);
    ASSERT_INT_EQ("attack timer decrements after healer chase",
        state.player.attack_timer, 2);
    ASSERT_INT_EQ("healer target is counted as cooldown after chase",
        state.total_zuk_healer_cooldown_ticks, 1);
    ASSERT_INT_EQ("healer target is not counted as range blocked after chase",
        state.total_zuk_healer_out_of_range_ticks, 0);

    int healer_hit_seen = 0;
    for (int i = 0; i < 4 && !healer_hit_seen; i++) {
        memset(actions, 0, sizeof(actions));
        inf_tick_player_ctx(&state, &test_context, actions, 1);
        healer_hit_seen = state.npcs[2].pending_hits.hits[0].active;
    }

    ASSERT_INT_EQ("healer target eventually fires after chase and cooldown",
        healer_hit_seen, 1);
    ASSERT_INT_EQ("healer target remains attackable after chase",
        inf_player_can_attack_npc_from_current_tile_ctx(
            &state, &test_context, 2), 1);
}

static void test_render_facing_prefers_attack_target_while_chasing(void) {
    printf("--- render facing prefers attack target while chasing ---\n");

    RenderEntity entity;
    memset(&entity, 0, sizeof(entity));
    entity.current_hitpoints = 99;
    entity.attack_target_entity_idx = 2;
    entity.attack_style_this_tick = ATTACK_STYLE_NONE;

    ASSERT_INT_EQ("attack target facing wins over movement",
        render_entity_select_facing_mode(&entity, 1),
        RENDER_ENTITY_FACE_ATTACK_TARGET);

    entity.attack_target_entity_idx = -1;
    ASSERT_INT_EQ("untargeted movement faces movement direction",
        render_entity_select_facing_mode(&entity, 1),
        RENDER_ENTITY_FACE_MOVEMENT);
}

static void test_render_identity_survives_npc_death_compaction(void) {
    printf("--- render identity survives NPC death compaction ---\n");

    RenderEntity previous[4];
    memset(previous, 0, sizeof(previous));
    previous[0].entity_type = ENTITY_PLAYER;
    previous[1].entity_type = ENTITY_NPC;
    previous[1].npc_slot = 0;
    previous[1].npc_def_id = INF_NPC_DEF_IDS[INF_NPC_MAGER];
    previous[2].entity_type = ENTITY_NPC;
    previous[2].npc_slot = 1;
    previous[2].npc_def_id = INF_NPC_DEF_IDS[INF_NPC_RANGER];
    previous[3].entity_type = ENTITY_NPC;
    previous[3].npc_slot = 2;
    previous[3].npc_def_id = INF_NPC_DEF_IDS[INF_NPC_MELEER];

    int used[4] = {0};
    RenderEntity current = previous[2];
    int previous_idx = render_entity_find_previous_identity_index(
        previous, 4, used, &current);

    ASSERT_INT_EQ("ranger keeps old visual slot after earlier NPC dies",
        previous_idx, 2);
    used[previous_idx] = 1;
    current = previous[3];
    previous_idx = render_entity_find_previous_identity_index(
        previous, 4, used, &current);
    ASSERT_INT_EQ("meleer keeps old visual slot after earlier NPC dies",
        previous_idx, 3);

    current.npc_def_id = INF_NPC_DEF_IDS[INF_NPC_RANGER];
    previous_idx = render_entity_find_previous_identity_index(
        previous, 4, used, &current);
    ASSERT_INT_EQ("same slot with different NPC type is not the same identity",
        previous_idx, -1);

    used[0] = 0;
    used[1] = 0;
    used[2] = 0;
    used[3] = 0;
    previous[1].npc_instance_id = 41;
    current = previous[1];
    current.npc_instance_id = 42;
    previous_idx = render_entity_find_previous_identity_index(
        previous, 4, used, &current);
    ASSERT_INT_EQ("same slot and type with new spawn id resets visual state",
        previous_idx, -1);

    current.npc_instance_id = 41;
    previous_idx = render_entity_find_previous_identity_index(
        previous, 4, used, &current);
    ASSERT_INT_EQ("same spawn id keeps visual state",
        previous_idx, 1);
}

static void test_render_identity_matches_two_players_across_tick(void) {
    printf("--- render identity matches two players across tick ---\n");

    RenderEntity previous[2];
    memset(previous, 0, sizeof(previous));
    previous[0].entity_type = ENTITY_PLAYER;
    previous[1].entity_type = ENTITY_PLAYER;

    int used[2] = {0, 0};
    RenderEntity current = previous[0];
    int idx0 = render_entity_find_previous_identity_index(previous, 2, used, &current);
    ASSERT_INT_EQ("first PvP player matches previous[0]", idx0, 0);

    used[idx0] = 1;
    current = previous[1];
    int idx1 = render_entity_find_previous_identity_index(previous, 2, used, &current);
    ASSERT_INT_EQ("second PvP player matches previous[1]", idx1, 1);
}

static void test_render_identity_two_players_claim_unique_slots(void) {
    printf("--- render identity two players claim unique slots ---\n");

    RenderEntity previous[2];
    memset(previous, 0, sizeof(previous));
    previous[0].entity_type = ENTITY_PLAYER;
    previous[1].entity_type = ENTITY_PLAYER;

    int used[2] = {0, 0};
    RenderEntity current;
    memset(&current, 0, sizeof(current));
    current.entity_type = ENTITY_PLAYER;

    int idx0 = render_entity_find_previous_identity_index(previous, 2, used, &current);
    if (idx0 >= 0) used[idx0] = 1;
    int idx1 = render_entity_find_previous_identity_index(previous, 2, used, &current);

    ASSERT_INT_EQ("first player gets a previous slot", (idx0 >= 0), 1);
    ASSERT_INT_EQ("second player gets a previous slot", (idx1 >= 0), 1);
    ASSERT_INT_EQ("two concurrent players claim distinct previous slots",
        (idx0 != idx1), 1);
}

static void test_render_identity_single_player_unchanged(void) {
    printf("--- render identity single player unchanged ---\n");

    RenderEntity previous[1];
    memset(previous, 0, sizeof(previous));
    previous[0].entity_type = ENTITY_PLAYER;

    int used[1] = {0};
    RenderEntity current = previous[0];
    int idx = render_entity_find_previous_identity_index(previous, 1, used, &current);
    ASSERT_INT_EQ("single-player encounter still matches previous[0]", idx, 0);
}

static void test_render_motion_speed_ladder_matches_deob(void) {
    printf("--- render motion speed ladder matches deob ---\n");

    int stall_debt = 0;
    ASSERT_INT_EQ("depth one walks at 4",
        osrs_render_speed_one_client_tick(1, 0, &stall_debt), 4);
    ASSERT_INT_EQ("depth two walks at 4",
        osrs_render_speed_one_client_tick(2, 0, &stall_debt), 4);
    ASSERT_INT_EQ("depth three catches up at 6",
        osrs_render_speed_one_client_tick(3, 0, &stall_debt), 6);
    ASSERT_INT_EQ("depth four catches up at 8",
        osrs_render_speed_one_client_tick(4, 0, &stall_debt), 8);

    ASSERT_INT_EQ("run doubles base to 8",
        osrs_render_speed_one_client_tick(1, 1, &stall_debt), 8);
    ASSERT_INT_EQ("run doubles depth-three catch-up to 12",
        osrs_render_speed_one_client_tick(3, 1, &stall_debt), 12);
    ASSERT_INT_EQ("run doubles depth-four catch-up to 16",
        osrs_render_speed_one_client_tick(4, 1, &stall_debt), 16);

    stall_debt = 3;
    ASSERT_INT_EQ("stall debt with queue depth repays at 8",
        osrs_render_speed_one_client_tick(2, 0, &stall_debt), 8);
    ASSERT_INT_EQ("stall debt repays one tick", stall_debt, 2);

    stall_debt = 3;
    ASSERT_INT_EQ("depth one does not spend stall debt",
        osrs_render_speed_one_client_tick(1, 0, &stall_debt), 4);
    ASSERT_INT_EQ("depth one keeps stall debt", stall_debt, 3);

    ASSERT_INT_EQ("speed 8 selects run pose",
        osrs_render_speed_uses_run_pose(8.0f), 1);
    ASSERT_INT_EQ("speed 6 stays on walk pose",
        osrs_render_speed_uses_run_pose(6.0f), 0);
    ASSERT_INT_EQ("speed 4 stays on walk pose",
        osrs_render_speed_uses_run_pose(4.0f), 0);
}

static void test_render_motion_lone_step_takes_32_client_ticks(void) {
    printf("--- render motion lone step takes 32 client ticks ---\n");

    OsrsRenderWaypointQueue q;
    osrs_render_waypoint_queue_clear(&q);
    float sub_x = 64.0f, sub_y = 64.0f;
    osrs_render_waypoint_push(&q, 64.0f + 128.0f, 64.0f, 0);

    int stall_debt = 0;
    int ticks = 0;
    while (q.length > 0 && ticks < 100) {
        int speed;
        float ddx, ddy;
        osrs_render_waypoint_advance_one_client_tick(
            &q, &sub_x, &sub_y, &stall_debt, &speed, &ddx, &ddy);
        ticks++;
    }
    ASSERT_INT_EQ("isolated 1-tile step takes 32 client ticks (640ms > 600ms tick)",
        ticks, 32);
    ASSERT_FLOAT_NEAR("arrived at the waypoint", sub_x, 192.0f, 1e-6f);
}

static void render_motion_continuity_case(
    const char* label, float tiles_per_tick, int running
) {
    OsrsRenderWaypointQueue q;
    osrs_render_waypoint_queue_clear(&q);
    float sub_x = 64.0f, sub_y = 64.0f;
    float true_x = 64.0f;
    int stall_debt = 0;
    int started = 0, pauses = 0, max_depth = 0;

    for (int tick = 0; tick < 40; tick++) {
        true_x += tiles_per_tick * OSRS_RENDER_SUB_UNITS_PER_TILE;
        osrs_render_waypoint_push(&q, true_x, 64.0f, running);
        for (int ct = 0; ct < (int)OSRS_RENDER_CLIENT_TICKS_PER_GAME_TICK; ct++) {
            int speed;
            float ddx, ddy;
            int moving = osrs_render_waypoint_advance_one_client_tick(
                &q, &sub_x, &sub_y, &stall_debt, &speed, &ddx, &ddy);
            if (started && !moving) pauses++;
            if (moving) started = 1;
            if (q.length > max_depth) max_depth = q.length;
        }
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "%s never pauses mid-walk", label);
    ASSERT_INT_EQ(msg, pauses, 0);
    snprintf(msg, sizeof(msg), "%s queue depth stays bounded", label);
    ASSERT_INT_EQ(msg, max_depth <= 3, 1);
    snprintf(msg, sizeof(msg), "%s visual trails within three tiles", label);
    ASSERT_INT_EQ(msg,
        (true_x - sub_x) <= 3.0f * OSRS_RENDER_SUB_UNITS_PER_TILE, 1);
}

static void test_render_motion_continuous_movement_never_pauses(void) {
    printf("--- render motion continuous movement never pauses ---\n");

    render_motion_continuity_case("continuous walk", 1.0f, 0);
    render_motion_continuity_case("continuous run", 2.0f, 1);
}

static void test_render_motion_waypoint_pop_snap_and_overflow(void) {
    printf("--- render motion waypoint pop, axis snap, queue overflow ---\n");

    OsrsRenderWaypointQueue q;
    osrs_render_waypoint_queue_clear(&q);
    float sub_x = 64.0f, sub_y = 64.0f;
    osrs_render_waypoint_push(&q, 192.0f, 64.0f, 0);
    osrs_render_waypoint_push(&q, 320.0f, 64.0f, 0);
    int stall_debt = 0;
    for (int ct = 0; ct < 32; ct++) {
        int speed;
        float ddx, ddy;
        osrs_render_waypoint_advance_one_client_tick(
            &q, &sub_x, &sub_y, &stall_debt, &speed, &ddx, &ddy);
    }
    ASSERT_FLOAT_NEAR("arrival tick clamps at the popped waypoint",
        sub_x, 192.0f, 1e-6f);
    ASSERT_INT_EQ("first waypoint popped, second still queued", q.length, 1);

    osrs_render_waypoint_queue_clear(&q);
    sub_x = 0.0f;
    sub_y = 0.0f;
    osrs_render_waypoint_push(&q, 300.0f, 40.0f, 0);
    {
        int speed;
        float ddx, ddy;
        osrs_render_waypoint_advance_one_client_tick(
            &q, &sub_x, &sub_y, &stall_debt, &speed, &ddx, &ddy);
    }
    ASSERT_FLOAT_NEAR("beyond-2-tile gap snaps x to the waypoint",
        sub_x, 300.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("beyond-2-tile gap snaps y to the waypoint too",
        sub_y, 40.0f, 1e-6f);
    ASSERT_INT_EQ("snapped waypoint pops the same cycle", q.length, 0);

    osrs_render_waypoint_queue_clear(&q);
    for (int i = 1; i <= 11; i++)
        osrs_render_waypoint_push(&q, (float)(i * 128), 0.0f, 0);
    ASSERT_INT_EQ("queue caps at 10 waypoints",
        q.length, OSRS_RENDER_WAYPOINT_QUEUE_DEPTH);
    ASSERT_FLOAT_NEAR("overflow drops the oldest waypoint",
        q.x[q.length - 1], 256.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("newest waypoint at the queue head",
        q.x[0], 11.0f * 128.0f, 1e-6f);
}

static void test_render_motion_seed_classification_uses_explicit_teleport(void) {
    printf("--- render motion seed classification uses explicit teleport ---\n");

    ASSERT_INT_EQ("persistent normal entity does not seed from distance",
        osrs_render_should_seed_visual_position(
            1, 0, 0, RENDER_MOVEMENT_NORMAL),
        0);
    ASSERT_INT_EQ("explicit teleport seeds visual position",
        osrs_render_should_seed_visual_position(
            1, 0, 0, RENDER_MOVEMENT_TELEPORT),
        1);
    ASSERT_INT_EQ("new identity seeds visual position",
        osrs_render_should_seed_visual_position(
            1, 1, 0, RENDER_MOVEMENT_NORMAL),
        1);
    ASSERT_INT_EQ("invisible to visible appearance seeds visual position",
        osrs_render_should_seed_visual_position(
            1, 0, 1, RENDER_MOVEMENT_NORMAL),
        1);
}


static void test_entity_model_ground_lift_keeps_floor_planes_above_terrain(void) {
    printf("--- entity model ground lift keeps floor planes above terrain ---\n");

    float ground = 2.0f;
    ASSERT_FLOAT_NEAR("model ground is lifted above terrain",
        osrs_render_entity_model_ground(ground),
        ground + OSRS_RENDER_ENTITY_GROUND_LIFT,
        1e-6f);
    ASSERT_INT_EQ("model ground lift is positive",
        OSRS_RENDER_ENTITY_GROUND_LIFT > 0.0f, 1);
}

static void test_spotanim_lookup_prefers_recolored_model_alias(void) {
    printf("--- spotanim lookup prefers recolored model alias ---\n");

    OsrsModel models[2];
    memset(models, 0, sizeof(models));
    models[0].model_id = 3136;
    models[1].model_id = OSRS_SPOTANIM_RECOLOR_MODEL_BASE | 1384u;

    ModelCache secondary_cache;
    memset(&secondary_cache, 0, sizeof(secondary_cache));
    secondary_cache.models = models;
    secondary_cache.count = 2;

    OsrsSpotAnimDef meta;
    memset(&meta, 0, sizeof(meta));
    meta.id = 1384;
    meta.model_id = 3136;

    OsrsModel* found = effect_find_model(&meta, NULL, &secondary_cache, NULL);
    ASSERT_INT_EQ("blob magic spotanim resolves recolored model",
        found ? (int)found->model_id : -1,
        (int)(OSRS_SPOTANIM_RECOLOR_MODEL_BASE | 1384u));
}

static void test_inferno_npc_spawn_id_changes_on_slot_reuse(void) {
    printf("--- inferno npc spawn id changes on slot reuse ---\n");

    InfernoState state = make_test_state(20, 20);
    inf_build_npc_stats();
    inf_init_npc(&state, 3, INF_NPC_BLOB_RANGE, 18, 20);
    uint32_t first_render_id = state.npcs[3].render_id;
    inf_deactivate_npc(&state, 3);
    inf_init_npc(&state, 3, INF_NPC_BLOB_RANGE, 19, 21);
    uint32_t second_render_id = state.npcs[3].render_id;

    ASSERT_INT_EQ("first render id is nonzero", first_render_id != 0, 1);
    ASSERT_INT_EQ("slot reuse gets a new render id",
        second_render_id != first_render_id, 1);

    RenderEntity entities[4];
    int count = 0;
    inf_fill_render_entities_ctx((EncounterState*)&state, (EncounterContext*)&test_context, entities, 4, &count);
    ASSERT_INT_EQ("reused NPC appears in render list", count, 2);
    ASSERT_INT_EQ("render entity carries spawn id",
        entities[1].npc_instance_id, second_render_id);
}

static void test_anim_rest_pose_resets_working_vertices(void) {
    printf("--- animation rest pose resets working vertices ---\n");

    uint8_t skins[2] = {0, 0};
    int16_t base[6] = {10, -20, 30, 40, -50, 60};
    AnimModelState* state = anim_model_state_create(skins, 2);
    for (int i = 0; i < 6; i++)
        state->verts[i] = (int16_t)(999 - i);

    anim_apply_rest_pose(state, base);

    for (int i = 0; i < 6; i++) {
        char label[64];
        snprintf(label, sizeof(label), "rest vertex %d", i);
        ASSERT_INT_EQ(label, state->verts[i], base[i]);
    }

    anim_model_state_free(state);
}

static void test_zuk_healer_target_action_tags_on_landed_hit(void) {
    printf("--- zuk healer target action tags on landed hit ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    state.tick = 321;
    state.player.x = 20;
    state.player.y = 46;
    encounter_apply_loadout(&state.player, INF_MAX_RANGE_LONG_LOADOUT, GEAR_RANGED);
    inf_refresh_live_stats(&state);

    state.npcs[2] = make_test_npc(
        INF_NPC_HEALER_ZUK, 20, 48, INF_NPC_STATS[INF_NPC_HEALER_ZUK].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[2].aggro_target = 0;

    float obs[INF_NUM_OBS];
    float mask[INF_ACTION_MASK_SIZE];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);
    inf_write_mask_ctx((EncounterState*)&state, (EncounterContext*)&test_context, mask);

    int healer_slot = inf_find_target_obs_slot(&state, 2);
    ASSERT_INT_EQ("zuk healer occupies first healer slot",
        state.current_obs_slots[healer_slot], 2);
    ASSERT_FLOAT_NEAR("zuk healer target mask is valid",
        mask[inferno_target_mask_slot_offset(healer_slot)], 1.0f, 1e-6f);

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_PRIMARY] = inf_primary_attack_action_for_obs_slot(healer_slot);
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("target action selects zuk healer",
        state.interaction.target_slot, 2);
    ASSERT_INT_EQ("target action records first healer target tick",
        state.tick_at_first_zuk_healer_target, 321);
    ASSERT_INT_EQ("target action records healer target tick count",
        state.total_zuk_healer_target_ticks, 1);
    ASSERT_INT_EQ("player attack queues healer hit",
        state.npcs[2].pending_hits.hits[0].active, 1);
    ASSERT_INT_EQ("player attack records first healer attack tick",
        state.tick_at_first_zuk_healer_attack, 321);
    ASSERT_INT_EQ("player attack records healer attack fire count",
        state.total_zuk_healer_attack_fires, 1);
    ASSERT_INT_EQ("non-magic attack at untagged healer is counted",
        state.tick_scratch.zuk_untagged_healer_nonmagic_attacks, 1);
    ASSERT_INT_EQ("non-magic attack does not count mage healer fire",
        state.tick_scratch.zuk_healer_mage_attack_fires, 0);
    ASSERT_INT_EQ("attackable healer target tick counted",
        state.total_zuk_healer_attackable_ticks, 1);
    ASSERT_INT_EQ("healer target was not blocked by cooldown",
        state.total_zuk_healer_cooldown_ticks, 0);
    ASSERT_INT_EQ("healer target was not blocked by range",
        state.total_zuk_healer_out_of_range_ticks, 0);

    state.npcs[2].pending_hits.hits[0].damage = 0;
    state.npcs[2].pending_hits.hits[0].ticks_remaining = 1;
    inf_resolve_player_projectiles_on_npcs(&state);

    ASSERT_INT_EQ("landed zero-damage hit tags zuk healer",
        state.npcs[2].aggro_target, -1);
    ASSERT_INT_EQ("landed zero-damage hit increments tag count",
        state.tick_scratch.healer_tags, 1);
}

static void test_zuk_healer_mage_attack_counts_penalty_event(void) {
    printf("--- Zuk healer mage attack counts penalty event ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    state.tick = 654;
    state.player.x = 20;
    state.player.y = 46;
    state.weapon_set = INF_GEAR_MAGE;
    state.player.autocast_enabled = 1;
    state.player.autocast_spell = ENCOUNTER_SPELL_ICE;
    encounter_apply_loadout(&state.player, INF_MAX_MAGE_LOADOUT, GEAR_MAGE);
    inf_refresh_live_stats(&state);
    encounter_compute_loadout_stats(INF_MAX_MAGE_LOADOUT, ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_AUTOCAST, 30,
        &state.loadout_stats[INF_GEAR_MAGE]);

    state.npcs[2] = make_test_npc(
        INF_NPC_HEALER_ZUK, 20, 48, INF_NPC_STATS[INF_NPC_HEALER_ZUK].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[2].aggro_target = 0;

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_PRIMARY] = inf_action_target_for_npc(&state, 2);
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("mage healer attack fires once",
        state.tick_scratch.zuk_healer_mage_attack_fires, 1);
    ASSERT_INT_EQ("mage healer attack gets no non-magic attempt count",
        state.tick_scratch.zuk_untagged_healer_nonmagic_attacks, 0);
    ASSERT_INT_EQ("mage attack still records total healer fire count",
        state.total_zuk_healer_attack_fires, 1);
}

static void test_zuk_safe_healer_target_mask_requires_fire_window(void) {
    printf("--- safe Zuk healer target mask requires fire window ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    test_config()->zuk_safe_untagged_healer_target_mask = 1;
    encounter_apply_loadout(&state.player, INF_MAX_RANGE_LONG_LOADOUT, GEAR_RANGED);
    inf_refresh_live_stats(&state);

    state.npcs[2] = make_test_npc(
        INF_NPC_HEALER_ZUK, 20, 48, INF_NPC_STATS[INF_NPC_HEALER_ZUK].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[2].aggro_target = 0;

    float obs[INF_NUM_OBS];
    float mask[INF_ACTION_MASK_SIZE];

    state.player.x = 20;
    state.player.y = 46;
    state.player.attack_timer = 0;
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);
    inf_write_mask_ctx((EncounterState*)&state, (EncounterContext*)&test_context, mask);
    int healer_slot = inf_find_target_obs_slot(&state, 2);
    ASSERT_FLOAT_NEAR("unsafe healer target masked while off shield",
        mask[inferno_target_mask_slot_offset(healer_slot)], 0.0f, 1e-6f);

    state.player.x = 24;
    state.player.y = 46;
    state.player.attack_timer = 0;
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);
    inf_write_mask_ctx((EncounterState*)&state, (EncounterContext*)&test_context, mask);
    ASSERT_FLOAT_NEAR("safe fire-ready healer target remains valid",
        mask[inferno_target_mask_slot_offset(healer_slot)], 1.0f, 1e-6f);

    state.player.attack_timer = 2;
    inf_write_mask_ctx((EncounterState*)&state, (EncounterContext*)&test_context, mask);
    ASSERT_FLOAT_NEAR("cooldown healer target is masked",
        mask[inferno_target_mask_slot_offset(healer_slot)], 0.0f, 1e-6f);

    state.npcs[2].aggro_target = -1;
    inf_write_mask_ctx((EncounterState*)&state, (EncounterContext*)&test_context, mask);
    ASSERT_FLOAT_NEAR("tagged healer target remains valid for killing",
        mask[inferno_target_mask_slot_offset(healer_slot)], 1.0f, 1e-6f);
}

static void test_zuk_safe_healer_target_mask_clears_unsafe_target(void) {
    printf("--- safe Zuk healer target mask clears unsafe target ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    test_config()->zuk_safe_untagged_healer_target_mask = 1;
    state.player.x = 24;
    state.player.y = 46;
    state.player.attack_timer = 3;
    encounter_apply_loadout(&state.player, INF_MAX_RANGE_LONG_LOADOUT, GEAR_RANGED);
    inf_refresh_live_stats(&state);

    state.npcs[2] = make_test_npc(
        INF_NPC_HEALER_ZUK, 20, 48, INF_NPC_STATS[INF_NPC_HEALER_ZUK].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[2].aggro_target = 0;
    osrs_interaction_set(&state.interaction, 2);

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("unsafe active healer target cleared",
        osrs_interaction_active(&state.interaction), 0);
    ASSERT_INT_EQ("cooldown target tick is not counted after clear",
        state.total_zuk_healer_cooldown_ticks, 0);
}

static void test_zuk_force_safe_healer_target_mask_blocks_idle_when_safe(void) {
    printf("--- force safe Zuk healer target mask blocks idle when safe ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    test_config()->zuk_force_safe_untagged_healer_target_mask = 1;
    encounter_apply_loadout(&state.player, INF_MAX_RANGE_LONG_LOADOUT, GEAR_RANGED);
    inf_refresh_live_stats(&state);
    state.player.x = 24;
    state.player.y = 46;
    state.player.attack_timer = 0;

    state.npcs[2] = make_test_npc(
        INF_NPC_HEALER_ZUK, 20, 48, INF_NPC_STATS[INF_NPC_HEALER_ZUK].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[2].aggro_target = 0;
    state.npcs[3] = make_test_npc(
        INF_NPC_MAGER, 28, 48, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[3].active = 1;
    state.npcs[3].hp = state.npcs[3].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;

    float obs[INF_NUM_OBS];
    float mask[INF_ACTION_MASK_SIZE];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);
    inf_write_mask_ctx((EncounterState*)&state, (EncounterContext*)&test_context, mask);
    int healer_slot = inf_find_target_obs_slot(&state, 2);
    int mager_slot = inf_find_target_obs_slot(&state, 3);

    ASSERT_INT_EQ("healer appears in target obs", healer_slot >= 0, 1);
    ASSERT_INT_EQ("mager appears in target obs", mager_slot >= 0, 1);
    ASSERT_FLOAT_NEAR("idle target action masked while safe healer exists",
        mask[inferno_target_mask_none_offset()], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("safe untagged healer target remains valid",
        mask[inferno_target_mask_slot_offset(healer_slot)], 1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("non-healer target masked while safe healer exists",
        mask[inferno_target_mask_slot_offset(mager_slot)], 0.0f, 1e-6f);

    state.player.attack_timer = 2;
    inf_write_mask_ctx((EncounterState*)&state, (EncounterContext*)&test_context, mask);
    ASSERT_FLOAT_NEAR("idle target action returns when no safe fire window exists",
        mask[inferno_target_mask_none_offset()], 1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("unsafe untagged healer remains masked during cooldown",
        mask[inferno_target_mask_slot_offset(healer_slot)], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("non-healer target returns when no safe healer exists",
        mask[inferno_target_mask_slot_offset(mager_slot)], 1.0f, 1e-6f);
}

static void test_zuk_force_safe_healer_target_mask_clears_stale_target(void) {
    printf("--- force safe Zuk healer target mask clears stale target ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    test_config()->zuk_force_safe_untagged_healer_target_mask = 1;
    encounter_apply_loadout(&state.player, INF_MAX_RANGE_LONG_LOADOUT, GEAR_RANGED);
    inf_refresh_live_stats(&state);
    state.player.x = 24;
    state.player.y = 46;
    state.player.attack_timer = 0;

    state.npcs[2] = make_test_npc(
        INF_NPC_HEALER_ZUK, 20, 48, INF_NPC_STATS[INF_NPC_HEALER_ZUK].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[2].aggro_target = 0;
    state.npcs[3] = make_test_npc(
        INF_NPC_MAGER, 28, 48, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[3].active = 1;
    state.npcs[3].hp = state.npcs[3].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;
    osrs_interaction_set(&state.interaction, 3);

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("stale non-healer target cleared",
        osrs_interaction_active(&state.interaction), 0);
    ASSERT_INT_EQ("stale target did not count as healer target tick",
        state.total_zuk_healer_target_ticks, 0);
}

static void test_zuk_spark_render_matches_pending_spark_state(void) {
    printf("--- zuk spark render matches pending spark state ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    state.pending_sparks[0] = (InfPendingSpark){
        .active = 1, .src_x = 16, .src_y = 48,
        .x = state.player.x, .y = state.player.y,
        .damage = 9, .ticks_remaining = 4,
    };

    EncounterOverlay ov;
    memset(&ov, 0, sizeof(ov));
    inf_render_post_tick_ctx((EncounterState*)&state, (EncounterContext*)&test_context, &ov);

    ASSERT_INT_EQ("spark render emits one projectile", ov.projectile_count, 1);
    ASSERT_INT_EQ("spark source x", ov.projectiles[0].src_x, 16);
    ASSERT_INT_EQ("spark source y", ov.projectiles[0].src_y, 48);
    ASSERT_INT_EQ("spark target x", ov.projectiles[0].dst_x, state.player.x);
    ASSERT_INT_EQ("spark target y", ov.projectiles[0].dst_y, state.player.y);
    ASSERT_INT_EQ("spark visual duration", ov.projectiles[0].duration_ticks, 4 * 30);
    ASSERT_INT_EQ("spark projectile model", ov.projectiles[0].model_id, INF_GFX_660_MODEL);
    ASSERT_INT_EQ("spark projectile animation", ov.projectiles[0].anim_id, INF_GFX_660_ANIM);
    ASSERT_INT_EQ("spark impact spotanim", ov.projectiles[0].impact_gfx_id, 659);
    ASSERT_INT_EQ("spark render marks visual emitted",
        state.pending_sparks[0].visual_emitted, 1);
}

static void test_zuk_attack_delay_counts_down_while_stunned(void) {
    printf("--- zuk attack delay counts down while stunned ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);

    inf_npc_attack_ctx(&state, &test_context, 0);

    ASSERT_INT_EQ("zuk stun decremented", state.npcs[0].stun_timer, 7);
    ASSERT_INT_EQ("zuk attack delay decremented during stun",
        state.npcs[0].attack_timer, 13);
    ASSERT_INT_EQ("zuk does not attack while stunned",
        state.npcs[0].attacked_this_tick, 0);
}

static void test_zuk_set_timer_spawns_on_decrement_to_zero(void) {
    printf("--- zuk set timer spawns on decrement to zero ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    state.zuk.set_timer = 1;

    inf_zuk_tick(&state);

    ASSERT_INT_EQ("mager spawned when set timer reaches zero",
        find_active_npc_type(&state, INF_NPC_MAGER) >= 0, 1);
    ASSERT_INT_EQ("ranger spawned when set timer reaches zero",
        find_active_npc_type(&state, INF_NPC_RANGER) >= 0, 1);
    ASSERT_INT_EQ("set timer resets to interval", state.zuk.set_timer, 350);
}

static void test_zuk_hp_threshold_pause_happens_before_set_tick(void) {
    printf("--- zuk hp threshold pause happens before set tick ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    state.npcs[0].hp = 601;
    state.npcs[0].attack_timer = 100;
    state.npcs[0].stun_timer = 0;
    state.zuk.set_timer = 10;
    state.npcs[0].pending_hits.hits[0].active = 1;
    state.npcs[0].pending_hits.hits[0].damage = 2;
    state.npcs[0].pending_hits.hits[0].ticks_remaining = 1;
    state.npcs[0].pending_hits.hits[0].attack_style = ATTACK_STYLE_RANGED;
    state.npcs[0].pending_hits.count = 1;

    step_inferno_noop(&state);

    ASSERT_INT_EQ("zuk damage landed", state.npcs[0].hp, 599);
    ASSERT_INT_EQ("zuk set timer paused on same tick", state.zuk.timer_paused, 1);
    ASSERT_INT_EQ("zuk set timer did not tick after pause", state.zuk.set_timer, 10);
}

static void test_set_attack_to_shield_is_projectile_delayed(void) {
    printf("--- set attack to shield is projectile delayed ---\n");

    int found_immediate_damage = 0;
    for (uint32_t seed = 1; seed < 200; seed++) {
        InfernoState state;
        init_zuk_timing_state(&state);
        state.rng_state = seed;
        state.npcs[2] = make_test_npc(
            INF_NPC_MAGER, 20, 36, INF_NPC_STATS[INF_NPC_MAGER].size);
        state.npcs[2].active = 1;
        state.npcs[2].hp = state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;
        state.npcs[2].attack_timer = 0;
        state.npcs[2].aggro_target = 1;

        inf_npc_attack_ctx(&state, &test_context, 2);
        if (state.npcs[1].hp < state.npcs[1].max_hp ||
            state.tick_scratch.shield_damage > 0.0f) {
            found_immediate_damage = 1;
            break;
        }
    }

    ASSERT_INT_EQ("set attack does not damage shield on fire tick",
        found_immediate_damage, 0);
}

static void test_npc_target_projectile_delays_match_reference(void) {
    printf("--- npc target projectile delays match reference ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);

    InfNPC mager = make_test_npc(
        INF_NPC_MAGER, 20, 36, INF_NPC_STATS[INF_NPC_MAGER].size);
    InfNPC ranger = make_test_npc(
        INF_NPC_RANGER, 29, 36, INF_NPC_STATS[INF_NPC_RANGER].size);
    InfNPC jad = make_test_npc(
        INF_NPC_JAD, 24, 32, INF_NPC_STATS[INF_NPC_JAD].size);
    InfNPC* shield = &state.npcs[1];

    int mager_dist = encounter_projectile_distance(
        mager.x, mager.y, mager.size, shield->x, shield->y, shield->size,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    int ranger_dist = encounter_projectile_distance(
        ranger.x, ranger.y, ranger.size, shield->x, shield->y, shield->size,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    int jad_dist = encounter_projectile_distance(
        jad.x, jad.y, jad.size, shield->x, shield->y, shield->size,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);

    ASSERT_INT_EQ("mager shield hit delay",
        inf_npc_target_hit_delay(&mager, shield, ATTACK_STYLE_MAGIC),
        inf_npc_projectile_timing(
            INF_NPC_MAGER, ATTACK_STYLE_MAGIC, mager_dist).damage_delay_ticks);
    ASSERT_INT_EQ("ranger shield hit delay includes reduceDelay -2",
        inf_npc_target_hit_delay(&ranger, shield, ATTACK_STYLE_RANGED),
        inf_npc_projectile_timing(
            INF_NPC_RANGER, ATTACK_STYLE_RANGED, ranger_dist).damage_delay_ticks);
    ASSERT_INT_EQ("jad magic shield hit delay uses jad path",
        inf_npc_target_hit_delay(&jad, shield, ATTACK_STYLE_MAGIC),
        INF_JAD_PROJECTILE_DELAY + inf_npc_projectile_timing(
            INF_NPC_JAD, ATTACK_STYLE_MAGIC, jad_dist).damage_delay_ticks);
    ASSERT_INT_EQ("jad ranged shield hit delay uses jad path",
        inf_npc_target_hit_delay(&jad, shield, ATTACK_STYLE_RANGED),
        INF_JAD_PROJECTILE_DELAY + inf_npc_projectile_timing(
            INF_NPC_JAD, ATTACK_STYLE_RANGED, jad_dist).damage_delay_ticks);
}

static void test_npc_player_projectile_delays_use_reference_options(void) {
    printf("--- npc player projectile delays use reference options ---\n");

    InfernoState state = make_test_state(20, 20);
    InfNPC* ranger = &state.npcs[0];
    *ranger = make_test_npc(
        INF_NPC_RANGER, 26, 20, INF_NPC_STATS[INF_NPC_RANGER].size);
    ranger->active = 1;
    ranger->attack_timer = 0;
    ranger->attack_style = ATTACK_STYLE_RANGED;
    ranger->had_los_last_tick = 1;

    int dist = encounter_projectile_distance(
        ranger->x, ranger->y, ranger->size, state.player.x, state.player.y, 1,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    EncounterProjectileTiming timing =
        inf_npc_projectile_timing(INF_NPC_RANGER, ATTACK_STYLE_RANGED, dist);

    inf_npc_attack_ctx(&state, &test_context, 0);

    ASSERT_INT_EQ("ranger queued one pending hit", state.player_pending_hits.count, 1);

    /* Was `- 1`. The queue is resolved before NPCs throw and lands on
       --ticks_remaining <= 0, so a hit queued with the raw delay D lands exactly D
       ticks after the throw tick, which is what the section-8 table states. The old
       expectation pinned every inferno NPC->player hit one tick early. */
    ASSERT_INT_EQ("ranger pending hit carries the raw projectile delay",
        state.player_pending_hits.hits[0].ticks_remaining, timing.damage_delay_ticks);
}
static void test_npc_hit_lands_on_the_reference_tick(void) {
    printf("--- npc hit lands on the reference tick ---\n");

    const struct { const char* label; InfNPCType type; AttackStyle style; int x; int y; }
    cases[] = {
        { "ranger d=5", INF_NPC_RANGER, ATTACK_STYLE_RANGED, 29, 24 },
        { "ranger d=8", INF_NPC_RANGER, ATTACK_STYLE_RANGED, 32, 24 },
        { "mager d=5",  INF_NPC_MAGER,  ATTACK_STYLE_MAGIC,  29, 24 },
        { "mager d=9",  INF_NPC_MAGER,  ATTACK_STYLE_MAGIC,  33, 24 },
    };

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        InfernoState* s = (InfernoState*)inf_create();
        inf_reset_ctx((EncounterState*)s, (EncounterContext*)&test_context, 20260728u);
        inf_lab_apply_command_ctx(s, &test_context,
            &(InfernoLabCommand){ .kind = INF_LAB_COMMAND_CLEAR_NPCS });
        inf_lab_apply_command_ctx(s, &test_context, &(InfernoLabCommand){
            .kind = INF_LAB_COMMAND_SET_PLAYER,
            .as.tile = { .x = 24, .y = 24 },
        });
        inf_lab_apply_command_ctx(s, &test_context, &(InfernoLabCommand){
            .kind = INF_LAB_COMMAND_SPAWN_NPC,
            .as.spawn_npc = {
                .slot = 0, .type = cases[c].type, .x = cases[c].x, .y = cases[c].y,
                .hp = { .kind = ENCOUNTER_LAB_OPTIONAL_INT_UNSET },
                .timer = { .kind = ENCOUNTER_LAB_OPTIONAL_INT_SET, .value = 0 },
            },
        });

        InfNPC* npc = &s->npcs[0];
        npc->attack_style = cases[c].style;
        npc->aggro_target = -1;
        npc->had_los_last_tick = 1;
        s->player.prayer = PRAYER_NONE;
        s->wave_spawn_delay = 0;
        s->wave_ready_delay = 0;

        int dist = encounter_projectile_distance(
            npc->x, npc->y, npc->size, s->player.x, s->player.y, 1,
            ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
        int expected_delay =
            inf_npc_projectile_timing(cases[c].type, cases[c].style, dist)
                .damage_delay_ticks;

        int actions[INF_NUM_ACTION_HEADS] = {0};
        int throw_tick = -1;
        int land_tick = -1;

        for (int t = 1; t <= 20 && land_tick < 0; t++) {
            int hp_before = s->player.current_hitpoints;
            npc->x = cases[c].x;
            npc->y = cases[c].y;
            inf_step_ctx((EncounterState*)s, (EncounterContext*)&test_context, actions);

            if (throw_tick < 0 && npc->attacked_this_tick) {
                throw_tick = t;
                npc->attack_timer = 10000;
                for (int i = 0; i < s->player_pending_hits.count; i++) {
                    if (!s->player_pending_hits.hits[i].active) continue;
                    s->player_pending_hits.hits[i].damage = 7;
                    s->player_pending_hits.hits[i].hit_success = 1;
                }
            } else if (throw_tick >= 0 && s->player.current_hitpoints != hp_before) {
                land_tick = t;
            }
        }

        ASSERT_INT_EQ(cases[c].label, land_tick - throw_tick, expected_delay);
        inf_destroy((EncounterState*)s);
    }
}

static void test_player_projectile_timing_uses_reference_options(void) {
    printf("--- player projectile timing uses reference options ---\n");

    int closest = encounter_projectile_distance(
        16, 11, 1, 12, 10, 3,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    int sw_tile = encounter_projectile_distance(
        16, 11, 1, 12, 10, 3,
        ENCOUNTER_PROJECTILE_DISTANCE_TARGET_SW_TILE);
    ASSERT_INT_EQ("barrage distance uses target SW tile", sw_tile > closest, 1);

    EncounterProjectileTiming barrage =
        inf_player_projectile_timing(ATTACK_STYLE_MAGIC, ITEM_KODAI_WAND, 0, sw_tile);
    ASSERT_INT_EQ("barrage damage delay uses SW distance",
        barrage.damage_delay_ticks, encounter_magic_hit_delay(sw_tile, 1));

    EncounterProjectileTiming blowpipe_spec = inf_player_projectile_timing(
        ATTACK_STYLE_RANGED, ITEM_TOXIC_BLOWPIPE, 1, 12);
    ASSERT_INT_EQ("blowpipe spec adds one damage tick",
        blowpipe_spec.damage_delay_ticks, encounter_blowpipe_hit_delay(12, 1) + 1);
    ASSERT_INT_EQ("blowpipe spec visual delay", blowpipe_spec.visual_start_delay_ticks, 1);
    ASSERT_INT_EQ("blowpipe spec visual duration",
        blowpipe_spec.visual_duration_ticks, encounter_blowpipe_hit_delay(12, 1) - 1);

    EncounterProjectileTiming tbow =
        inf_player_projectile_timing(ATTACK_STYLE_RANGED, ITEM_TWISTED_BOW, 0, 12);
    ASSERT_INT_EQ("tbow damage delay",
        tbow.damage_delay_ticks, encounter_ranged_hit_delay(12, 1));
    ASSERT_INT_EQ("tbow visual delay", tbow.visual_start_delay_ticks, 1);
    ASSERT_INT_EQ("tbow visual duration",
        tbow.visual_duration_ticks, encounter_ranged_hit_delay(12, 1) - 1);
}

static void init_phantom_barrage_test_state(
    InfernoState* state,
    int death_ticks,
    int player_attack_timer
) {
    init_zuk_timing_state(state);
    memset(state->npcs, 0, sizeof(state->npcs));
    state->zuk.shield_idx = -1;
    state->wave = 60;
    state->player.x = 20;
    state->player.y = 40;
    state->player.attack_timer = player_attack_timer;
    state->weapon_set = INF_GEAR_MAGE;
    state->player.autocast_enabled = 1;
    state->player.autocast_spell = ENCOUNTER_SPELL_ICE;
    encounter_apply_loadout(&state->player, INF_MAX_MAGE_LOADOUT, GEAR_MAGE);
    inf_refresh_live_stats(state);
    encounter_compute_loadout_stats(INF_MAX_MAGE_LOADOUT, ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_AUTOCAST, 30,
        &state->loadout_stats[INF_GEAR_MAGE]);

    state->npcs[0] = make_test_npc(
        INF_NPC_NIBBLER, 28, 40, INF_NPC_STATS[INF_NPC_NIBBLER].size);
    state->npcs[0].active = 1;
    state->npcs[0].hp = 0;
    state->npcs[0].max_hp = INF_NPC_STATS[INF_NPC_NIBBLER].hp;
    state->npcs[0].death_ticks = death_ticks;

    state->npcs[1] = make_test_npc(
        INF_NPC_NIBBLER, 28, 41, INF_NPC_STATS[INF_NPC_NIBBLER].size);
    state->npcs[1].active = 1;
    state->npcs[1].hp = state->npcs[1].max_hp = INF_NPC_STATS[INF_NPC_NIBBLER].hp;
}

static void test_phantom_barrage_target_is_masked_until_cast_window(void) {
    printf("--- phantom barrage target is masked until cast window ---\n");

    InfernoState state;
    float mask[INF_ACTION_MASK_SIZE];

    init_phantom_barrage_test_state(&state, 2, 1);
    inf_write_mask_ctx((EncounterState*)&state, (EncounterContext*)&test_context, mask);
    int target_slot = inf_find_target_obs_slot(&state, 0);
    ASSERT_INT_EQ("dying target appears before cast window", target_slot >= 0, 1);
    ASSERT_FLOAT_NEAR("next tick phantom target is valid",
        mask[inferno_target_mask_slot_offset(target_slot)], 1.0f, 1e-6f);

    init_phantom_barrage_test_state(&state, 2, 2);
    inf_write_mask_ctx((EncounterState*)&state, (EncounterContext*)&test_context, mask);
    target_slot = inf_find_target_obs_slot(&state, 0);
    ASSERT_INT_EQ("cooldown dying target still appears", target_slot >= 0, 1);
    ASSERT_FLOAT_NEAR("cooldown phantom target is masked",
        mask[inferno_target_mask_slot_offset(target_slot)], 0.0f, 1e-6f);
}

static void test_phantom_barrage_hits_aoe_on_first_cast_window(void) {
    printf("--- manual phantom barrage hits AoE on first cast window ---\n");

    int found_aoe_hit = 0;
    for (uint32_t seed = 1; seed < 200 && !found_aoe_hit; seed++) {
        InfernoState state;
        init_phantom_barrage_test_state(&state, 1, 1);
        state.rng_state = seed;
        inf_refresh_current_obs_slots_ctx(&state, &test_context);
        int target_slot = inf_find_target_obs_slot(&state, 0);
        ASSERT_INT_EQ("dying target appears in cast window", target_slot >= 0, 1);
        if (target_slot < 0) return;

        int actions[INF_NUM_ACTION_HEADS];
        memset(actions, 0, sizeof(actions));
        actions[INF_HEAD_PRIMARY] = inf_primary_attack_action_for_obs_slot(target_slot);
        actions[INF_HEAD_SPELL] = 2;
        inf_tick_player_ctx(&state, &test_context, actions, 1);

        ASSERT_INT_EQ("phantom primary does not receive stale pending hit",
            state.npcs[0].pending_hits.hits[0].active, 0);
        if (state.tick_scratch.player_attacked &&
                state.npcs[1].pending_hits.hits[0].active &&
                state.npcs[1].pending_hits.hits[0].attack_style == ATTACK_STYLE_MAGIC) {
            found_aoe_hit = 1;
        }
    }

    ASSERT_INT_EQ("phantom barrage can queue adjacent AoE hit", found_aoe_hit, 1);
}

static void test_ranged_attack_cannot_fire_on_dying_target(void) {
    printf("--- ranged attack cannot fire on dying target ---\n");

    InfernoState state;
    init_spell_cast_test_state(&state, INF_NPC_NIBBLER);
    state.weapon_set = INF_GEAR_LONG_RANGE;
    encounter_apply_loadout(&state.player, INF_MAX_RANGE_LONG_LOADOUT, GEAR_RANGED);
    inf_refresh_live_stats(&state);
    state.player.autocast_enabled = 0;
    state.player.attack_timer = 0;
    state.npcs[0].hp = 0;
    state.npcs[0].death_ticks = 1;
    osrs_interaction_set(&state.interaction, 0);

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("ranged attack does not fire on dying target",
        state.tick_scratch.player_attacked, 0);
    ASSERT_INT_EQ("ranged attack does not queue dying target pending hit",
        state.npcs[0].pending_hits.hits[0].active, 0);
    ASSERT_INT_EQ("ranged attack does not start cooldown",
        state.player.attack_timer, 0);
}

static void test_autocast_barrage_cannot_fire_on_dying_target(void) {
    printf("--- autocast barrage cannot fire on dying target ---\n");

    InfernoState state;
    init_spell_cast_test_state(&state, INF_NPC_NIBBLER);
    state.player.attack_timer = 0;
    state.npcs[0].hp = 0;
    state.npcs[0].death_ticks = 1;
    osrs_interaction_set(&state.interaction, 0);

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("autocast does not fire on dying target",
        state.tick_scratch.player_attacked, 0);
    ASSERT_INT_EQ("autocast does not queue dying target pending hit",
        state.npcs[0].pending_hits.hits[0].active, 0);
    ASSERT_INT_EQ("autocast does not start cooldown",
        state.player.attack_timer, 0);
}

static void test_manual_blood_barrage_can_heal_from_dying_primary(void) {
    printf("--- manual blood barrage can heal from dying primary ---\n");

    int found_heal = 0;
    for (uint32_t seed = 1; seed < 400 && !found_heal; seed++) {
        InfernoState state;
        init_spell_cast_test_state(&state, INF_NPC_NIBBLER);
        state.rng_state = seed;
        state.player.current_hitpoints = 80;
        state.player.attack_timer = 0;
        state.npcs[0].hp = 0;
        state.npcs[0].death_ticks = 1;
        osrs_interaction_set(&state.interaction, 0);

        int actions[INF_NUM_ACTION_HEADS];
        memset(actions, 0, sizeof(actions));
        actions[INF_HEAD_SPELL] = 1;
        inf_tick_player_ctx(&state, &test_context, actions, 1);

        ASSERT_INT_EQ("manual blood barrage fires on dying target",
            state.tick_scratch.player_attacked, 1);
        ASSERT_INT_EQ("manual blood barrage does not queue dying target pending hit",
            state.npcs[0].pending_hits.hits[0].active, 0);
        if (state.tick_scratch.blood_heal > 0 &&
                state.player.current_hitpoints > 80) {
            found_heal = 1;
        }
    }

    ASSERT_INT_EQ("manual blood barrage can heal from dying primary",
        found_heal, 1);
}

static void test_phantom_barrage_close_barrage_timing_cannot_recast(void) {
    printf("--- phantom barrage close barrage timing cannot recast ---\n");

    InfernoState state;
    init_phantom_barrage_test_state(&state, 1, 2);
    inf_refresh_current_obs_slots_ctx(&state, &test_context);
    int target_slot = inf_find_target_obs_slot(&state, 0);
    ASSERT_INT_EQ("dying target appears during cooldown", target_slot >= 0, 1);
    if (target_slot < 0) return;

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_PRIMARY] = inf_primary_attack_action_for_obs_slot(target_slot);
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("cooldown prevents phantom barrage fire",
        state.tick_scratch.player_attacked, 0);
    ASSERT_INT_EQ("cooldown prevents AoE pending hit",
        state.npcs[1].pending_hits.hits[0].active, 0);
    ASSERT_INT_EQ("attack timer only decrements",
        state.player.attack_timer, 1);
}

static void test_phantom_barrage_does_not_displace_live_obs_slots(void) {
    printf("--- phantom barrage does not displace live obs slots ---\n");

    InfernoState state;
    init_phantom_barrage_test_state(&state, 1, 1);
    for (int i = 1; i <= 6; i++) {
        state.npcs[i] = make_test_npc(
            INF_NPC_NIBBLER, 28 + i, 41, INF_NPC_STATS[INF_NPC_NIBBLER].size);
        state.npcs[i].active = 1;
        state.npcs[i].hp = state.npcs[i].max_hp = INF_NPC_STATS[INF_NPC_NIBBLER].hp;
    }

    inf_refresh_current_obs_slots_ctx(&state, &test_context);

    for (int npc_idx = 1; npc_idx <= 6; npc_idx++) {
        ASSERT_INT_EQ("live nibbler fills capped obs slot",
            inf_find_target_obs_slot(&state, npc_idx) >= 0, 1);
    }
    ASSERT_INT_EQ("dying phantom target does not displace live cap",
        inf_find_target_obs_slot(&state, 0), -1);
}


static void init_confliction_barrage_test_state(
    InfernoState* state,
    InfNPCType target_type
) {
    *state = make_test_state(10, 10);
    state->rng_state = 1;
    state->weapon_set = INF_GEAR_MAGE;
    state->player.entity_type = ENTITY_PLAYER;
    state->player.base_hitpoints = 99;
    state->player.current_hitpoints = 99;
    state->player.base_attack = 99;
    state->player.base_strength = 99;
    state->player.base_defence = 99;
    state->player.base_ranged = 99;
    state->player.base_magic = 99;
    state->player.current_attack = 99;
    state->player.current_strength = 99;
    state->player.current_defence = 99;
    state->player.current_ranged = 99;
    state->player.current_magic = 99;
    state->player.autocast_enabled = 1;
    state->player.autocast_spell = ENCOUNTER_SPELL_BLOOD;
    state->player_dest_x = -1;
    state->player_dest_y = -1;
    osrs_interaction_init(&state->interaction);
    encounter_apply_loadout(&state->player, INF_MAX_MAGE_LOADOUT, GEAR_MAGE);
    inf_refresh_live_stats(state);
    encounter_compute_loadout_stats(INF_MAX_MAGE_LOADOUT, ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_AUTOCAST, 30,
        &state->loadout_stats[INF_GEAR_MAGE]);

    state->npcs[0] = make_test_npc(
        target_type, 16, 10, INF_NPC_STATS[target_type].size);
    state->npcs[0].active = 1;
    state->npcs[0].hp = state->npcs[0].max_hp = INF_NPC_STATS[target_type].hp;
    osrs_interaction_set(&state->interaction, 0);
}

static int inferno_fire_blood_barrage_at_slot_zero(
    InfernoState* state,
    uint32_t seed
) {
    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    state->rng_state = seed;
    state->player.attack_timer = 0;
    encounter_pending_hit_queue_clear(&state->npcs[0].pending_hits);
    osrs_interaction_set(&state->interaction, 0);
    inf_tick_player_ctx(state, &test_context, actions, 1);
    return state->npcs[0].pending_hits.hits[0].hit_success;
}

static void test_default_autocast_casts_blood_barrage(void) {
    printf("--- default autocast casts blood barrage ---\n");

    InfernoState state;
    init_confliction_barrage_test_state(&state, INF_NPC_RANGER);

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    state.rng_state = 1;
    state.player.attack_timer = 0;
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("pending hit records blood barrage",
        state.npcs[0].pending_hits.hits[0].spell_type, ENCOUNTER_SPELL_BLOOD);
    ASSERT_INT_EQ("player render spell records blood barrage",
        state.player.magic_type_this_tick, ENCOUNTER_SPELL_BLOOD);
}

static void test_ice_barrage_success_freezes_target_and_records_spell(void) {
    printf("--- ice barrage success freezes target and records spell ---\n");

    int found = 0;
    for (uint32_t seed = 1; seed < 256 && !found; seed++) {
        InfernoState state;
        init_confliction_barrage_test_state(&state, INF_NPC_RANGER);
        state.player.autocast_spell = ENCOUNTER_SPELL_ICE;

        int actions[INF_NUM_ACTION_HEADS];
        memset(actions, 0, sizeof(actions));
        state.rng_state = seed;
        state.player.attack_timer = 0;
        inf_tick_player_ctx(&state, &test_context, actions, 1);

        if (state.npcs[0].pending_hits.hits[0].hit_success) {
            found = 1;
            ASSERT_INT_EQ("ice pending hit records ice barrage",
                state.npcs[0].pending_hits.hits[0].spell_type, ENCOUNTER_SPELL_ICE);
            ASSERT_INT_EQ("ice barrage freezes on successful accuracy",
                state.npcs[0].frozen_ticks, BARRAGE_FREEZE_TICKS);
            ASSERT_INT_EQ("player render spell records ice barrage",
                state.player.magic_type_this_tick, ENCOUNTER_SPELL_ICE);
        }
    }
    ASSERT_INT_EQ("found deterministic successful ice barrage seed", found, 1);
}

static void test_inferno_barrage_primes_confliction_and_reuses_double_accuracy(void) {
    printf("--- inferno barrage primes Confliction and reuses double accuracy ---\n");

    InfernoState state;
    uint32_t miss_seed = 0;
    for (uint32_t seed = 1; seed < 10000 && miss_seed == 0; seed++) {
        init_confliction_barrage_test_state(&state, INF_NPC_RANGER);
        int hit = inferno_fire_blood_barrage_at_slot_zero(&state, seed);
        if (state.npcs[0].pending_hits.hits[0].active && !hit)
            miss_seed = seed;
    }
    ASSERT_INT_EQ("deterministic miss seed found", miss_seed > 0, 1);
    if (miss_seed == 0) return;

    init_confliction_barrage_test_state(&state, INF_NPC_RANGER);
    ASSERT_INT_EQ("first barrage splashes",
        inferno_fire_blood_barrage_at_slot_zero(&state, miss_seed), 0);
    ASSERT_INT_EQ("splash primes Confliction",
        state.player.item_effect_state.confliction_is_primed, 1);
    ASSERT_INT_EQ("prime target kind",
        state.player.item_effect_state.confliction_target.kind, OSRS_TARGET_NPC);
    ASSERT_INT_EQ("prime target slot",
        state.player.item_effect_state.confliction_target.id, 0);

    const EncounterLoadoutStats* ls = &state.loadout_stats[INF_GEAR_MAGE];
    OsrsPreparedAttackEffects prepared = osrs_prepare_attack_effects(
        &state.player.equipment_effect_profile,
        &state.player.item_effect_state,
        state.player.equipped[GEAR_SLOT_WEAPON],
        ATTACK_STYLE_MAGIC,
        OSRS_MAGIC_ATTACK_ANCIENT_BLOOD,
        (OsrsTargetRef){ .kind = OSRS_TARGET_NPC, .id = 0 },
        1,
        ls->eff_level * (ls->attack_bonus + 64),
        ls->max_hit,
        osrs_target_effect_context_none(),
        state.player.current_hitpoints,
        state.player.base_hitpoints
    );
    ASSERT_INT_EQ("same target prepares double accuracy",
        prepared.use_double_accuracy, 1);

    inferno_fire_blood_barrage_at_slot_zero(&state, miss_seed);
    ASSERT_INT_EQ("double accuracy attempt clears Confliction",
        state.player.item_effect_state.confliction_is_primed, 0);
}

static void test_barrage_accuracy_regression_against_ranger_and_mager(void) {
    printf("--- barrage accuracy regression against ranger and mager ---\n");

    InfNPCType targets[2] = { INF_NPC_RANGER, INF_NPC_MAGER };
    for (int i = 0; i < 2; i++) {
        int hits = 0;
        for (uint32_t seed = 1; seed <= 64; seed++) {
            InfernoState state;
            init_confliction_barrage_test_state(&state, targets[i]);
            hits += inferno_fire_blood_barrage_at_slot_zero(&state, seed);
        }
        ASSERT_INT_EQ("repeated barrages are not all splashes", hits > 0, 1);
        ASSERT_INT_EQ("repeated barrages are not guaranteed hits", hits < 64, 1);
    }
}

static void init_barrage_pending_queue_edge_state(InfernoState* state) {
    init_confliction_barrage_test_state(state, INF_NPC_RANGER);
    state->player.x = 24;
    state->player.y = 14;
    state->npcs[0].x = 12;
    state->npcs[0].y = 14;
    state->npcs[0].hp = state->npcs[0].max_hp = INF_NPC_STATS[INF_NPC_RANGER].hp;
    encounter_pending_hit_queue_clear(&state->npcs[0].pending_hits);
    osrs_interaction_set(&state->interaction, 0);
}

static void tick_barrage_pending_queue_edge_state(InfernoState* state) {
    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_SPELL] = 1;
    state->tick_scratch.player_attacked = 0;
    state->npcs[0].hit_landed_this_tick = 0;
    state->npcs[0].hit_damage = 0;
    state->npcs[0].hit_was_successful_this_tick = 0;
    inf_resolve_player_projectiles_on_npcs(state);
    inf_tick_player_ctx(state, &test_context, actions, 1);
    state->tick++;
}

static void test_barrage_pending_queue_handles_slow_hit_delay(void) {
    printf("--- barrage pending queue handles slow hit delay ---\n");

    InfernoState state;
    init_barrage_pending_queue_edge_state(&state);

    int closest = encounter_projectile_distance(
        state.player.x, state.player.y, 1,
        state.npcs[0].x, state.npcs[0].y, state.npcs[0].size,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    int sw_tile = encounter_projectile_distance(
        state.player.x, state.player.y, 1,
        state.npcs[0].x, state.npcs[0].y, state.npcs[0].size,
        ENCOUNTER_PROJECTILE_DISTANCE_TARGET_SW_TILE);
    ASSERT_INT_EQ("ranger is attackable at closest-tile range ten", closest, 10);
    ASSERT_INT_EQ("barrage timing still sees target SW distance twelve", sw_tile, 12);
    ASSERT_INT_EQ("player can cast at closest-tile range ten",
        inf_player_can_attack_npc_from_current_tile_ctx(
            &state, &test_context, 0), 1);

    tick_barrage_pending_queue_edge_state(&state);
    ASSERT_INT_EQ("first cast queues one hit",
        state.npcs[0].pending_hits.count, 1);
    ASSERT_INT_EQ("first queued hit uses slow barrage travel",
        state.npcs[0].pending_hits.hits[0].ticks_remaining,
        encounter_magic_hit_delay(sw_tile, 1));

    for (int i = 0; i < 5; i++)
        tick_barrage_pending_queue_edge_state(&state);

    ASSERT_INT_EQ("second cast queues behind the first in-flight hit",
        state.npcs[0].pending_hits.count, 2);
    ASSERT_INT_EQ("oldest hit is one tick from landing",
        state.npcs[0].pending_hits.hits[0].ticks_remaining, 1);

    tick_barrage_pending_queue_edge_state(&state);
    ASSERT_INT_EQ("oldest hit lands instead of being overwritten",
        state.npcs[0].hit_landed_this_tick, 1);
    ASSERT_INT_EQ("one queued hit remains after first land",
        state.npcs[0].pending_hits.count, 1);
    ASSERT_INT_EQ("ranger took damage from queued hit",
        state.npcs[0].hp < state.npcs[0].max_hp, 1);
}

static void test_barrage_aoe_queues_hits_on_multiple_npcs(void) {
    printf("--- barrage AoE queues hits on multiple NPCs ---\n");

    InfernoState state;
    init_confliction_barrage_test_state(&state, INF_NPC_RANGER);
    state.npcs[1] = make_test_npc(
        INF_NPC_RANGER, 17, 10, INF_NPC_STATS[INF_NPC_RANGER].size);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp = INF_NPC_STATS[INF_NPC_RANGER].hp;

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    state.player.attack_timer = 0;
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("primary ranger has queued barrage hit",
        state.npcs[0].pending_hits.count, 1);
    ASSERT_INT_EQ("secondary ranger has queued barrage hit",
        state.npcs[1].pending_hits.count, 1);
    ASSERT_INT_EQ("primary queued hit is magic",
        state.npcs[0].pending_hits.hits[0].attack_style, ATTACK_STYLE_MAGIC);
    ASSERT_INT_EQ("secondary queued hit is magic",
        state.npcs[1].pending_hits.hits[0].attack_style, ATTACK_STYLE_MAGIC);
    ASSERT_INT_EQ("secondary queued hit preserves spell type",
        state.npcs[1].pending_hits.hits[0].spell_type, ENCOUNTER_SPELL_BLOOD);
}

static void test_repeated_edge_barrages_kill_ranger(void) {
    printf("--- repeated edge barrages kill ranger ---\n");

    InfernoState state;
    init_barrage_pending_queue_edge_state(&state);

    for (int tick = 0; tick < 240 && state.npcs[0].hp > 0; tick++)
        tick_barrage_pending_queue_edge_state(&state);

    ASSERT_INT_EQ("edge barrage loop kills the ranger",
        state.npcs[0].hp <= 0, 1);
}

static void test_npc_pending_queue_lands_multiple_hits_in_order(void) {
    printf("--- npc pending queue lands multiple hits in order ---\n");

    InfernoState state = make_test_state(10, 10);
    state.npcs[0] = make_test_npc(
        INF_NPC_RANGER, 16, 10, INF_NPC_STATS[INF_NPC_RANGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_RANGER].hp;
    inf_queue_npc_pending_hit(
        &state, 0, 1, 7, ATTACK_STYLE_MAGIC, ENCOUNTER_SPELL_BLOOD, 1);
    inf_queue_npc_pending_hit(
        &state, 0, 2, 11, ATTACK_STYLE_MAGIC, ENCOUNTER_SPELL_BLOOD, 1);

    inf_resolve_player_projectiles_on_npcs(&state);
    ASSERT_INT_EQ("first queued hit lands",
        state.npcs[0].hit_damage, 7);
    ASSERT_INT_EQ("second hit remains queued",
        state.npcs[0].pending_hits.count, 1);
    state.npcs[0].hit_landed_this_tick = 0;
    state.npcs[0].hit_damage = 0;

    inf_resolve_player_projectiles_on_npcs(&state);
    ASSERT_INT_EQ("second queued hit lands",
        state.npcs[0].hit_damage, 11);
    ASSERT_INT_EQ("queue is empty after both hits",
        state.npcs[0].pending_hits.count, 0);
}

static void test_npc_death_clears_pending_hits(void) {
    printf("--- npc death clears pending hits ---\n");

    InfernoState state = make_test_state(10, 10);
    state.npcs[0] = make_test_npc(
        INF_NPC_RANGER, 16, 10, INF_NPC_STATS[INF_NPC_RANGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 5;
    state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_RANGER].hp;
    inf_queue_npc_pending_hit(
        &state, 0, 1, 7, ATTACK_STYLE_MAGIC, ENCOUNTER_SPELL_BLOOD, 1);
    inf_queue_npc_pending_hit(
        &state, 0, 4, 11, ATTACK_STYLE_MAGIC, ENCOUNTER_SPELL_BLOOD, 1);

    inf_resolve_player_projectiles_on_npcs(&state);
    ASSERT_INT_EQ("lethal hit starts death linger",
        state.npcs[0].death_ticks > 0, 1);
    ASSERT_INT_EQ("death clears remaining pending hits",
        state.npcs[0].pending_hits.count, 0);
}

static void test_lab_dump_reports_npc_pending_hit_queue(void) {
    printf("--- lab dump reports npc pending hit queue ---\n");

    InfernoState state = make_test_state(10, 10);
    state.npcs[0] = make_test_npc(
        INF_NPC_RANGER, 16, 10, INF_NPC_STATS[INF_NPC_RANGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_RANGER].hp;
    inf_queue_npc_pending_hit(
        &state, 0, 3, 17, ATTACK_STYLE_MAGIC, ENCOUNTER_SPELL_BLOOD, 1);

    char* dump = inf_lab_alloc_json_ctx(&state, &test_context);
    ASSERT_INT_EQ("lab dump includes pending count",
        strstr(dump, "\"pending_count\":1") != NULL, 1);
    ASSERT_INT_EQ("lab dump includes pending timer",
        strstr(dump, "\"pending_earliest_ticks\":3") != NULL, 1);
    ASSERT_INT_EQ("lab dump includes pending damage",
        strstr(dump, "\"pending_damage\":17") != NULL, 1);
    free(dump);
}

static void test_explicit_spell_cast_does_not_persist(void) {
    printf("--- explicit spell cast does not persist ---\n");

    InfernoState state;
    init_spell_cast_test_state(&state, INF_NPC_RANGER);
    state.player.autocast_spell = ENCOUNTER_SPELL_BLOOD;

    fire_player_action_at_slot_zero(&state, 2);
    ASSERT_INT_EQ("manual ice cast records ice pending hit",
        state.npcs[0].pending_hits.hits[0].spell_type, ENCOUNTER_SPELL_ICE);
    ASSERT_INT_EQ("manual ice render records ice",
        state.player.magic_type_this_tick, ENCOUNTER_SPELL_ICE);

    state.player.attack_timer = 0;
    encounter_pending_hit_queue_clear(&state.npcs[0].pending_hits);
    fire_player_action_at_slot_zero(&state, 0);
    ASSERT_INT_EQ("later normal attack falls back to blood autocast",
        state.npcs[0].pending_hits.hits[0].spell_type, ENCOUNTER_SPELL_BLOOD);
}

static void test_spell_without_target_does_not_affect_later_attack(void) {
    printf("--- spell without target does not affect later attack ---\n");

    InfernoState state;
    init_spell_cast_test_state(&state, INF_NPC_RANGER);
    state.player.autocast_spell = ENCOUNTER_SPELL_BLOOD;

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_SPELL] = 2;
    inf_tick_player_ctx(&state, &test_context, actions, 1);
    ASSERT_INT_EQ("spell without target does not fire",
        state.tick_scratch.player_attacked, 0);

    fire_player_action_at_slot_zero(&state, 0);
    ASSERT_INT_EQ("next normal attack uses autocast blood",
        state.npcs[0].pending_hits.hits[0].spell_type, ENCOUNTER_SPELL_BLOOD);
}

static void test_target_without_spell_uses_autocast(void) {
    printf("--- target without spell uses autocast ---\n");

    InfernoState state;
    init_spell_cast_test_state(&state, INF_NPC_RANGER);
    state.player.autocast_spell = ENCOUNTER_SPELL_ICE;

    fire_player_action_at_slot_zero(&state, 0);
    ASSERT_INT_EQ("no-spell target attack uses ice autocast",
        state.npcs[0].pending_hits.hits[0].spell_type, ENCOUNTER_SPELL_ICE);
}

static void test_manual_spell_overrides_autocast(void) {
    printf("--- manual spell overrides autocast ---\n");

    InfernoState state;
    init_spell_cast_test_state(&state, INF_NPC_RANGER);
    state.player.autocast_spell = ENCOUNTER_SPELL_BLOOD;

    fire_player_action_at_slot_zero(&state, 2);
    ASSERT_INT_EQ("manual ice overrides blood autocast",
        state.npcs[0].pending_hits.hits[0].spell_type, ENCOUNTER_SPELL_ICE);

    init_spell_cast_test_state(&state, INF_NPC_RANGER);
    state.player.autocast_spell = ENCOUNTER_SPELL_ICE;
    fire_player_action_at_slot_zero(&state, 1);
    ASSERT_INT_EQ("manual blood overrides ice autocast",
        state.npcs[0].pending_hits.hits[0].spell_type, ENCOUNTER_SPELL_BLOOD);
}

static void test_blood_barrage_at_full_hp_is_valid_and_heals_zero(void) {
    printf("--- blood barrage at full HP is valid and heals zero ---\n");

    InfernoState state;
    init_spell_cast_test_state(&state, INF_NPC_NIBBLER);
    float mask[INF_ACTION_MASK_SIZE];
    inf_write_mask_ctx((EncounterState*)&state, (EncounterContext*)&test_context, mask);
    ASSERT_FLOAT_NEAR("blood barrage action valid at full HP",
        mask[inferno_action_head_mask_offset(INF_HEAD_SPELL) + 1], 1.0f, 1e-6f);

    fire_player_action_at_slot_zero(&state, 1);
    state.npcs[0].pending_hits.hits[0].damage = 12;
    state.npcs[0].pending_hits.hits[0].ticks_remaining = 1;
    state.player.current_hitpoints = state.player.base_hitpoints;
    inf_resolve_player_projectiles_on_npcs(&state);

    ASSERT_INT_EQ("full HP blood barrage heals zero",
        state.tick_scratch.blood_heal, 0);
    ASSERT_INT_EQ("HP stays capped",
        state.player.current_hitpoints, state.player.base_hitpoints);
}

static void test_manual_spell_in_range_gear_uses_range_gear_magic_stats(void) {
    printf("--- manual spell in range gear uses range gear magic stats ---\n");

    InfernoState state;
    init_spell_cast_test_state(&state, INF_NPC_RANGER);
    state.weapon_set = INF_GEAR_BP;
    encounter_apply_loadout(&state.player, INF_MAX_RANGE_FAST_LOADOUT, GEAR_RANGED);
    inf_refresh_live_stats(&state);
    state.player.autocast_spell = ENCOUNTER_SPELL_BLOOD;

    InfPlayerAttack attack;
    int resolved = inf_resolve_player_attack_ctx(&state, &test_context, ENCOUNTER_SPELL_ICE, &attack);
    EncounterLoadoutStats expected;
    encounter_compute_player_equipped_stats(
        &state.player, ATTACK_STYLE_MAGIC, FIGHT_STYLE_AUTOCAST, 30, &expected);

    ASSERT_INT_EQ("manual spell resolves as attack",
        resolved, 1);
    ASSERT_INT_EQ("manual spell is magic",
        attack.stats.style, ATTACK_STYLE_MAGIC);
    ASSERT_INT_EQ("manual spell uses current gear magic attack bonus",
        attack.stats.attack_bonus, expected.attack_bonus);
    ASSERT_INT_EQ("manual spell uses barrage speed",
        attack.stats.attack_speed, 5);
    ASSERT_INT_EQ("manual spell uses barrage range",
        attack.stats.attack_range, 10);
}

static void test_phantom_barrage_allows_explicit_spell_from_range_gear(void) {
    printf("--- phantom barrage allows explicit spell from range gear ---\n");

    InfernoState state;
    init_phantom_barrage_test_state(&state, 1, 1);
    state.weapon_set = INF_GEAR_BP;
    encounter_apply_loadout(&state.player, INF_MAX_RANGE_FAST_LOADOUT, GEAR_RANGED);
    inf_refresh_live_stats(&state);
    state.player.autocast_enabled = 1;
    state.player.autocast_spell = ENCOUNTER_SPELL_BLOOD;
    inf_refresh_current_obs_slots_ctx(&state, &test_context);

    int target_slot = inf_find_target_obs_slot(&state, 0);
    ASSERT_INT_EQ("dying target appears in obs slots", target_slot >= 0, 1);
    if (target_slot < 0) return;

    float mask[INF_ACTION_MASK_SIZE];
    inf_write_mask_ctx((EncounterState*)&state, (EncounterContext*)&test_context, mask);
    ASSERT_FLOAT_NEAR("explicit spell can target phantom from range gear",
        mask[inferno_target_mask_slot_offset(target_slot)], 1.0f, 1e-6f);

    int actions[INF_NUM_ACTION_HEADS];
    memset(actions, 0, sizeof(actions));
    actions[INF_HEAD_PRIMARY] = inf_primary_attack_action_for_obs_slot(target_slot);
    actions[INF_HEAD_SPELL] = 2;
    inf_tick_player_ctx(&state, &test_context, actions, 1);

    ASSERT_INT_EQ("explicit phantom barrage fires from range gear",
        state.tick_scratch.player_attacked, 1);
    ASSERT_INT_EQ("explicit phantom barrage uses magic style",
        state.player_attack_style_id, ATTACK_STYLE_MAGIC);
}

static void test_zuk_obs_tracks_shield_and_mager_aggro(void) {
    printf("--- zuk obs tracks shield hp/death and mager aggro ---\n");

    InfernoState state = make_test_state(INF_ZUK_PLAYER_START_X, INF_ZUK_PLAYER_START_Y);
    state.wave = 68;
    state.player.current_defence = 99;
    state.player.current_magic = 99;
    state.player.current_ranged = 99;
    state.player.current_hitpoints = 99;
    state.player.base_hitpoints = 99;
    state.player.base_prayer = 99;
    state.player.current_prayer = 99;
    state.player.prayer = PRAYER_NONE;
    state.weapon_set = INF_GEAR_LONG_RANGE;

    state.npcs[0] = make_test_npc(
        INF_NPC_MAGER, 20, 36, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;
    state.npcs[0].attack_timer = 4;

    state.npcs[1] = make_test_npc(
        INF_NPC_ZUK, 22, 14, INF_NPC_STATS[INF_NPC_ZUK].size);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp = INF_NPC_STATS[INF_NPC_ZUK].hp;

    state.npcs[2] = make_test_npc(
        INF_NPC_ZUK_SHIELD, 23, 44, INF_NPC_STATS[INF_NPC_ZUK_SHIELD].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = 300;
    state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_ZUK_SHIELD].hp;

    state.zuk.shield_idx = 2;
    state.zuk.shield_dir = -1;
    state.zuk.shield_freeze = 3;
    state.npcs[0].aggro_target = 2;

    float obs[INF_NUM_OBS];
    float mask[INF_ACTION_MASK_SIZE];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);
    inf_write_mask_ctx((EncounterState*)&state, (EncounterContext*)&test_context, mask);

    int mager_slot = inf_find_target_obs_slot(&state, 0);
    int shield_slot = inf_find_target_obs_slot(&state, 2);
    int shield_hp = inferno_obs_slot_hp_index(shield_slot);
    int mager_target_category = inferno_obs_slot_target_category_start(mager_slot);

    ASSERT_INT_EQ("first mager occupies mager slot 0", state.current_obs_slots[mager_slot], 0);
    ASSERT_INT_EQ("shield occupies dedicated shield slot", state.current_obs_slots[shield_slot], 2);
    ASSERT_FLOAT_NEAR("mager target shield uses compact category code",
        obs[mager_target_category],
        (float)INF_TARGET_CATEGORY_SHIELD / 8.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("shield direction visible while alive", obs[INF_OBS_ZUK_SHIELD_DIR], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("shield freeze visible while alive", obs[INF_OBS_ZUK_SHIELD_FREEZE], 0.6f, 1e-6f);
    ASSERT_FLOAT_NEAR("mager target mask is valid", mask[inferno_target_mask_slot_offset(mager_slot)], 1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("shield target mask stays invalid", mask[inferno_target_mask_slot_offset(shield_slot)], 0.0f, 1e-6f);

    state.npcs[2].active = 0;
    state.zuk.shield_idx = -1;
    state.npcs[0].aggro_target = -1;
    inf_invalidate_current_obs_slots(&state);

    memset(obs, 0, sizeof(obs));
    memset(mask, 0, sizeof(mask));
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);
    inf_write_mask_ctx((EncounterState*)&state, (EncounterContext*)&test_context, mask);

    ASSERT_INT_EQ("dead shield drops out of shield slot", state.current_obs_slots[shield_slot], -1);
    ASSERT_FLOAT_NEAR("dead shield slot hp zeros out", obs[shield_hp], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("dead shield zeroes stale direction", obs[INF_OBS_ZUK_SHIELD_DIR], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("dead shield zeroes stale freeze", obs[INF_OBS_ZUK_SHIELD_FREEZE], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("mager target player uses compact category code",
        obs[mager_target_category],
        (float)INF_TARGET_CATEGORY_PLAYER / 8.0f, 1e-6f);
}

static void test_zuk_healer_obs_exposes_target_category(void) {
    printf("--- zuk healer obs exposes target category ---\n");

    InfernoState state = make_test_state(INF_ZUK_PLAYER_START_X, INF_ZUK_PLAYER_START_Y);
    state.wave = 68;
    state.player.current_hitpoints = 99;
    state.player.base_hitpoints = 99;
    state.player.base_prayer = 99;
    state.player.current_prayer = 99;
    state.player.current_defence = 99;
    state.player.current_magic = 99;
    state.player.current_ranged = 99;
    state.weapon_set = INF_GEAR_LONG_RANGE;
    state.zuk.healer_spawned = 1;

    state.npcs[0] = make_test_npc(
        INF_NPC_ZUK, 22, 14, INF_NPC_STATS[INF_NPC_ZUK].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_ZUK].hp;

    state.npcs[1] = make_test_npc(
        INF_NPC_HEALER_ZUK, 20, 42, INF_NPC_STATS[INF_NPC_HEALER_ZUK].size);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp = INF_NPC_STATS[INF_NPC_HEALER_ZUK].hp;
    state.npcs[1].attack_timer = 4;
    state.npcs[1].aggro_target = 0;

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    int healer_slot = inf_find_target_obs_slot(&state, 1);
    int healer_target_category = inferno_obs_slot_target_category_start(healer_slot);
    ASSERT_INT_EQ("Zuk healer occupies first Zuk healer obs slot",
        state.current_obs_slots[healer_slot], 1);
    ASSERT_FLOAT_NEAR("untagged Zuk healer target uses compact Zuk code",
        obs[healer_target_category],
        (float)INF_TARGET_CATEGORY_ZUK / 8.0f, 1e-6f);

    state.npcs[1].aggro_target = -1;
    memset(obs, 0, sizeof(obs));
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    ASSERT_FLOAT_NEAR("tagged Zuk healer target uses compact player code",
        obs[healer_target_category],
        (float)INF_TARGET_CATEGORY_PLAYER / 8.0f, 1e-6f);
}

static void test_inferno_obs_target_categories_cover_boss_helpers(void) {
    printf("--- inferno obs target categories cover boss helpers ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player.current_hitpoints = 99;
    state.player.base_hitpoints = 99;
    state.player.base_prayer = 99;
    state.player.current_prayer = 99;
    state.player.current_defence = 99;
    state.player.current_magic = 99;
    state.player.current_ranged = 99;

    state.npcs[0] = make_test_npc(
        INF_NPC_JAD, 24, 20, INF_NPC_STATS[INF_NPC_JAD].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_JAD].hp;

    state.npcs[1] = make_test_npc(
        INF_NPC_HEALER_JAD, 25, 20, INF_NPC_STATS[INF_NPC_HEALER_JAD].size);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp = INF_NPC_STATS[INF_NPC_HEALER_JAD].hp;
    state.npcs[1].aggro_target = 0;

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    int healer_slot = inf_find_target_obs_slot(&state, 1);
    int target_category = inferno_obs_slot_target_category_start(healer_slot);
    ASSERT_INT_EQ("Jad healer occupies helper slot",
        state.current_obs_slots[healer_slot], 1);
    ASSERT_FLOAT_NEAR("Jad healer targeting Jad uses compact other-NPC code",
        obs[target_category],
        (float)INF_TARGET_CATEGORY_OTHER_NPC / 8.0f, 1e-6f);

    state.npcs[1].aggro_target = -1;
    memset(obs, 0, sizeof(obs));
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);
    ASSERT_FLOAT_NEAR("tagged Jad healer uses compact player code",
        obs[target_category],
        (float)INF_TARGET_CATEGORY_PLAYER / 8.0f, 1e-6f);
}

static void test_zuk_set_obs_los_uses_current_target(void) {
    printf("--- zuk set obs los uses current target ---\n");

    InfernoState state = make_test_state(11, 14);
    state.wave = 68;
    state.player.current_hitpoints = 99;
    state.player.base_hitpoints = 99;
    state.player.base_prayer = 99;
    state.player.current_prayer = 99;
    state.player.current_defence = 99;
    state.player.current_magic = 99;
    state.player.current_ranged = 99;
    state.weapon_set = INF_GEAR_LONG_RANGE;

    state.npcs[0] = make_test_npc(
        INF_NPC_MAGER, 20, 36, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;
    state.npcs[0].attack_timer = 4;
    state.npcs[0].aggro_target = 1;

    state.npcs[1] = make_test_npc(
        INF_NPC_ZUK_SHIELD, 23, 44, INF_NPC_STATS[INF_NPC_ZUK_SHIELD].size);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp = INF_NPC_STATS[INF_NPC_ZUK_SHIELD].hp;
    state.zuk.shield_idx = 1;

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    int mager_slot = inf_find_target_obs_slot(&state, 0);
    int mager_target_category = inferno_obs_slot_target_category_start(mager_slot);
    ASSERT_INT_EQ("mager occupies a dense obs slot",
        state.current_obs_slots[mager_slot], 0);
    ASSERT_FLOAT_NEAR("mager los follows shield target",
        obs[inferno_obs_slot_npc_los_index(mager_slot)], 1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("mager target shield uses compact category code",
        obs[mager_target_category],
        (float)INF_TARGET_CATEGORY_SHIELD / 8.0f, 1e-6f);
}

static void test_zuk_set_threat_ignores_shield_target(void) {
    printf("--- zuk set threat ignores shield target ---\n");

    InfernoState state = make_test_state(20, 34);
    state.wave = 68;
    state.player.current_hitpoints = 99;
    state.player.base_hitpoints = 99;
    state.player.base_prayer = 99;
    state.player.current_prayer = 99;
    state.player.current_defence = 99;
    state.player.current_magic = 99;
    state.player.current_ranged = 99;
    state.weapon_set = INF_GEAR_LONG_RANGE;

    state.npcs[0] = make_test_npc(
        INF_NPC_MAGER, 20, 36, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;
    state.npcs[0].attack_timer = 1;
    state.npcs[0].attack_style = ATTACK_STYLE_MAGIC;
    state.npcs[0].aggro_target = 1;

    state.npcs[1] = make_test_npc(
        INF_NPC_ZUK_SHIELD, 23, 44, INF_NPC_STATS[INF_NPC_ZUK_SHIELD].size);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp = INF_NPC_STATS[INF_NPC_ZUK_SHIELD].hp;
    state.zuk.shield_idx = 1;

    InfNpcPlayerThreat threat =
        inf_npc_player_threat_ctx(&state, &test_context, &state.npcs[0]);
    ASSERT_INT_EQ("shield-targeted mager does not threaten player",
        threat.can_attack_if_ready, 0);
    ASSERT_INT_EQ("shield-targeted mager has no player style mask",
        threat.style_mask, 0);
}

static void child_inf_put_bad_start_wave(void) {
    InfernoState state = make_test_state(0, 0);
    inf_put_int_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "start_wave", 0);
}

static void child_inf_put_unknown_int(void) {
    InfernoState state = make_test_state(0, 0);
    inf_put_int_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "bogus_key", 1);
}

static void child_inf_put_removed_win_bonus(void) {
    InfernoState state = make_test_state(0, 0);
    inf_put_float_ctx((EncounterState*)&state, (EncounterContext*)&test_context, "win_bonus_coeff", 1.0f);
}

static void child_encounter_emit_projectile_overflow(void) {
    EncounterOverlay ov;
    memset(&ov, 0, sizeof(ov));
    ov.projectile_count = ENCOUNTER_MAX_OVERLAY_PROJECTILES;
    encounter_emit_projectile(
        &ov, 0, 0, 1, 1, 0, 0, 30, 0, 0, 0, 0.0f, 0, 1, 1, 0, 0);
}

static void child_inf_pending_spark_overflow(void) {
    InfernoState state = make_test_state(20, 20);
    for (int i = 0; i < INF_MAX_PENDING_SPARKS; i++)
        state.pending_sparks[i].active = 1;
    inf_queue_pending_spark(&state, 0, 0, 20, 20, 1, 1);
}

static void child_inf_restore_previous_inventory_snapshot(void) {
    EncounterState* raw = inf_create();
    inf_reset_ctx(raw, (EncounterContext*)&test_context, 123u);
    size_t snap_size = inf_snapshot_size_ctx(raw, (EncounterContext*)&test_context);
    InfSnapshot* snap = (InfSnapshot*)malloc(snap_size);
    inf_snapshot_ctx(raw, (EncounterContext*)&test_context, snap);
    snap->version = INF_SNAPSHOT_VERSION - 1u;
    inf_restore_ctx(raw, (EncounterContext*)&test_context, snap, snap_size);
    free(snap);
    inf_destroy(raw);
}

static void child_inf_reset_before_topology_finalize(void) {
    InfernoContext ctx;
    InfernoState state;
    inf_init_unfinalized_context(&ctx);
    inf_init_state_typed(&state, &ctx);
    inf_reset_ctx((EncounterState*)&state, (EncounterContext*)&ctx, 1u);
}

static void child_inf_step_before_topology_finalize(void) {
    InfernoContext ctx;
    InfernoState state;
    int actions[INF_NUM_ACTION_HEADS] = {0};
    inf_init_unfinalized_context(&ctx);
    inf_init_state_typed(&state, &ctx);
    inf_step_ctx((EncounterState*)&state, (EncounterContext*)&ctx, actions);
}

static void child_inf_query_before_topology_finalize(void) {
    InfernoContext ctx;
    InfernoState state;
    inf_init_unfinalized_context(&ctx);
    inf_init_state_typed(&state, &ctx);
    (void)inf_footprint_blocked_ctx(&state, &ctx, 20, 20, 1);
}

static void child_inf_restore_wrong_config_snapshot(void) {
    InfernoState state_a;
    InfernoState state_b;
    InfernoContext ctx_a;
    InfernoContext ctx_b;

    inf_init_context_typed(&ctx_a);
    inf_init_context_typed(&ctx_b);
    ctx_a.config.damage_reward_coeff = 0.01f;
    ctx_b.config.damage_reward_coeff = 0.02f;
    inf_init_state_typed(&state_a, &ctx_a);
    inf_init_state_typed(&state_b, &ctx_b);
    inf_reset_ctx((EncounterState*)&state_a, (EncounterContext*)&ctx_a, 123u);
    inf_reset_ctx((EncounterState*)&state_b, (EncounterContext*)&ctx_b, 123u);

    InfSnapshot snap;
    inf_snapshot_ctx((EncounterState*)&state_a, (EncounterContext*)&ctx_a, &snap);
    inf_restore_ctx(
        (EncounterState*)&state_b,
        (EncounterContext*)&ctx_b,
        &snap,
        sizeof(snap));
}

static void test_fail_fast_boundaries(void) {
    printf("--- fail fast boundaries ---\n");

    assert_child_aborts("invalid inferno start wave aborts", child_inf_put_bad_start_wave);
    assert_child_aborts("unknown inferno int config aborts", child_inf_put_unknown_int);
    assert_child_aborts("removed win bonus config aborts", child_inf_put_removed_win_bonus);
    assert_child_aborts("overlay projectile overflow aborts", child_encounter_emit_projectile_overflow);
    assert_child_aborts("inferno pending spark overflow aborts", child_inf_pending_spark_overflow);
    assert_child_aborts(
        "inferno reset before topology finalize aborts",
        child_inf_reset_before_topology_finalize);
    assert_child_aborts(
        "inferno step before topology finalize aborts",
        child_inf_step_before_topology_finalize);
    assert_child_aborts(
        "inferno query before topology finalize aborts",
        child_inf_query_before_topology_finalize);
    assert_child_aborts(
        "inferno previous inventory snapshot restore aborts",
        child_inf_restore_previous_inventory_snapshot);
    assert_child_aborts(
        "inferno wrong-config snapshot restore aborts",
        child_inf_restore_wrong_config_snapshot);
}

static void test_human_target_and_potion_translation(void) {
    printf("--- inferno human target and potion translation ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player.base_hitpoints = 99;
    state.player.current_hitpoints = 80;
    state.player.base_prayer = 99;
    state.player.current_prayer = 60;
    state.player.current_attack = 99;
    state.player.current_strength = 99;
    state.player.current_defence = 99;
    state.player.current_ranged = 99;
    state.player.current_magic = 99;

    state.npcs[0] = make_test_npc(
        INF_NPC_MAGER, 24, 24, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;

    state.npcs[1] = make_test_npc(
        INF_NPC_MAGER, 26, 24, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;

    state.npcs[2] = make_test_npc(
        INF_NPC_MAGER, 28, 24, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp = INF_NPC_STATS[INF_NPC_MAGER].hp;

    state.npcs[3] = make_test_npc(
        INF_NPC_ZUK_SHIELD, 23, 44, INF_NPC_STATS[INF_NPC_ZUK_SHIELD].size);
    state.npcs[3].active = 1;
    state.npcs[3].hp = state.npcs[3].max_hp = INF_NPC_STATS[INF_NPC_ZUK_SHIELD].hp;

    {
        float obs[INF_NUM_OBS];
        inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);
    }

    ASSERT_INT_EQ("first visible mager is targetable",
        inf_is_human_targetable_npc_slot_ctx((EncounterState*)&state, (EncounterContext*)&test_context, 0), 1);
    ASSERT_INT_EQ("second visible mager is targetable",
        inf_is_human_targetable_npc_slot_ctx((EncounterState*)&state, (EncounterContext*)&test_context, 1), 1);
    ASSERT_INT_EQ("third capped-out mager is not targetable",
        inf_is_human_targetable_npc_slot_ctx((EncounterState*)&state, (EncounterContext*)&test_context, 2), 0);
    ASSERT_INT_EQ("shield is never targetable",
        inf_is_human_targetable_npc_slot_ctx((EncounterState*)&state, (EncounterContext*)&test_context, 3), 0);

    {
        HumanInput hi;
        int actions[INF_NUM_ACTION_HEADS];

        hi = make_human_input();
        hi.pending_target_idx = 0;
        inf_translate_human_input_ctx(&hi, actions, (EncounterState*)&state, (EncounterContext*)&test_context);
        ASSERT_INT_EQ("visible mager click maps into primary target range",
            actions[INF_HEAD_PRIMARY],
            inf_primary_attack_action_for_obs_slot(0));

        hi = make_human_input();
        hi.pending_target_idx = 2;
        inf_translate_human_input_ctx(&hi, actions, (EncounterState*)&state, (EncounterContext*)&test_context);
        ASSERT_INT_EQ("capped-out mager click is rejected",
            actions[INF_HEAD_PRIMARY], 0);

        hi = make_human_input();
        hi.pending_move_x = state.player.x + 1;
        hi.pending_move_y = state.player.y;
        hi.pending_target_idx = 2;
        inf_translate_human_input_ctx(&hi, actions, (EncounterState*)&state, (EncounterContext*)&test_context);
        ASSERT_INT_EQ("untargetable click preserves queued east move",
            actions[INF_HEAD_PRIMARY], 3);

        hi = make_human_input();
        hi.pending_target_idx = 3;
        inf_translate_human_input_ctx(&hi, actions, (EncounterState*)&state, (EncounterContext*)&test_context);
        ASSERT_INT_EQ("shield click is rejected",
            actions[INF_HEAD_PRIMARY], 0);

        state.player.brew_doses = 8;
        state.player.restore_doses = 8;
        state.player.bastion_doses = 4;
        state.player.stamina_doses = 4;
        inf_seed_inventory_cells(&state);

        hi = make_human_input();
        hi.pending_potion = POTION_BREW;
        inf_translate_human_input_ctx(&hi, actions, (EncounterState*)&state, (EncounterContext*)&test_context);
        ASSERT_INT_EQ("brew maps to a brew drink click",
            test_drink_click_kind(&state, actions[INF_HEAD_DRINK]),
            OSRS_CONSUMABLE_BREW);
        ASSERT_INT_EQ("brew does not touch eat head", actions[INF_HEAD_EAT], 0);

        hi = make_human_input();
        hi.pending_potion = POTION_RESTORE;
        inf_translate_human_input_ctx(&hi, actions, (EncounterState*)&state, (EncounterContext*)&test_context);
        ASSERT_INT_EQ("restore maps to a restore drink click",
            test_drink_click_kind(&state, actions[INF_HEAD_DRINK]),
            OSRS_CONSUMABLE_SUPER_RESTORE);

        hi = make_human_input();
        hi.pending_potion = POTION_BASTION;
        inf_translate_human_input_ctx(&hi, actions, (EncounterState*)&state, (EncounterContext*)&test_context);
        ASSERT_INT_EQ("bastion maps to a bastion drink click",
            test_drink_click_kind(&state, actions[INF_HEAD_DRINK]),
            OSRS_CONSUMABLE_BASTION);

        hi = make_human_input();
        hi.pending_potion = POTION_STAMINA;
        inf_translate_human_input_ctx(&hi, actions, (EncounterState*)&state, (EncounterContext*)&test_context);
        ASSERT_INT_EQ("stamina maps to a stamina drink click",
            test_drink_click_kind(&state, actions[INF_HEAD_DRINK]),
            OSRS_CONSUMABLE_STAMINA);

        hi = make_human_input();
        hi.pending_potion = POTION_PRAYER_POT;
        inf_translate_human_input_ctx(&hi, actions, (EncounterState*)&state, (EncounterContext*)&test_context);
        ASSERT_INT_EQ("prayer pot no longer aliases to restore",
            actions[INF_HEAD_DRINK], 0);
    }
}

static void test_human_targeting_refreshes_stale_obs_slots(void) {
    printf("--- inferno human targeting refreshes stale obs slots ---\n");

    InfernoState state = make_test_state(20, 20);
    for (int i = 0; i < INF_OBS_NPCS; i++) {
        state.current_obs_slots[i] = -1;
    }
    inf_invalidate_current_obs_slots(&state);

    state.npcs[5] = make_test_npc(
        INF_NPC_RANGER, 24, 24, INF_NPC_STATS[INF_NPC_RANGER].size);
    state.npcs[5].active = 1;
    state.npcs[5].hp = state.npcs[5].max_hp = INF_NPC_STATS[INF_NPC_RANGER].hp;

    ASSERT_INT_EQ("stale targetability slots start invalid",
        state.current_obs_slots_valid, 0);
    ASSERT_INT_EQ("live ranger is targetable without prior obs write",
        inf_is_human_targetable_npc_slot_ctx((EncounterState*)&state, (EncounterContext*)&test_context, 5), 1);
    ASSERT_INT_EQ("targetability refresh validates obs slots",
        state.current_obs_slots_valid, 1);

    inf_refresh_current_obs_slots_ctx(&state, &test_context);
    int ranger_slot = inf_find_target_obs_slot(&state, 5);
    ASSERT_INT_EQ("live ranger has a dense observation slot",
        ranger_slot >= 0, 1);

    {
        HumanInput hi = make_human_input();
        int actions[INF_NUM_ACTION_HEADS];
        hi.pending_target_idx = 5;
        state.current_obs_slots[ranger_slot] = -1;
        inf_invalidate_current_obs_slots(&state);
        ASSERT_INT_EQ("pending target slots start invalid",
            state.current_obs_slots_valid, 0);
        inf_translate_human_input_ctx(&hi, actions, (EncounterState*)&state, (EncounterContext*)&test_context);
        ASSERT_INT_EQ("pending target refresh validates obs slots",
            state.current_obs_slots_valid, 1);
        ASSERT_INT_EQ("pending human target refreshes obs slot",
            actions[INF_HEAD_PRIMARY],
            inf_primary_attack_action_for_obs_slot(ranger_slot));
    }

    {
        HumanInput hi;
        int actions[INF_NUM_ACTION_HEADS];
        human_input_init(&hi);
        human_input_queue_attack_npc(&hi, 5);
        state.current_obs_slots[ranger_slot] = -1;
        inf_invalidate_current_obs_slots(&state);
        ASSERT_INT_EQ("queued target slots start invalid",
            state.current_obs_slots_valid, 0);
        inf_translate_human_commands_ctx(&hi, actions, &state, &test_context);
        ASSERT_INT_EQ("queued target refresh validates obs slots",
            state.current_obs_slots_valid, 1);
        ASSERT_INT_EQ("queued human target refreshes obs slot",
            actions[INF_HEAD_PRIMARY],
            inf_primary_attack_action_for_obs_slot(ranger_slot));
        human_input_destroy(&hi);
    }
}

static void test_human_spell_selection_is_client_local_until_target_click(void) {
    printf("--- human spell selection is client local until target click ---\n");

    InfernoState state;
    init_spell_cast_test_state(&state, INF_NPC_RANGER);
    HumanInput hi;
    human_input_init(&hi);
    hi.cursor_mode = CURSOR_SPELL_TARGET;
    hi.selected_spell = OSRS_SPELL_BLOOD_BARRAGE;

    int actions[INF_NUM_ACTION_HEADS];
    inf_translate_human_commands_ctx(&hi, actions, &state, &test_context);
    ASSERT_INT_EQ("client-only selection queues no command",
        hi.commands.count, 0);
    ASSERT_INT_EQ("client-only selection sends no spell action",
        actions[INF_HEAD_SPELL], 0);

    human_input_queue_spell_target(&hi, OSRS_SPELL_BLOOD_BARRAGE, 0);
    inf_translate_human_commands_ctx(&hi, actions, &state, &test_context);
    ASSERT_INT_EQ("spell target command kind",
        hi.commands.items[0].kind, HUMAN_COMMAND_SPELL_TARGET);
    ASSERT_INT_EQ("spell target command carries blood",
        hi.commands.items[0].spell, OSRS_SPELL_BLOOD_BARRAGE);
    ASSERT_INT_EQ("spell target command maps spell action",
        actions[INF_HEAD_SPELL], OSRS_SPELL_BLOOD_BARRAGE);

    human_input_destroy(&hi);
}

static void test_human_walk_command_sends_no_selected_spell_cast(void) {
    printf("--- human walk command sends no selected spell cast ---\n");

    InfernoState state;
    init_spell_cast_test_state(&state, INF_NPC_RANGER);
    HumanInput hi;
    human_input_init(&hi);
    hi.cursor_mode = CURSOR_SPELL_TARGET;
    hi.selected_spell = OSRS_SPELL_ICE_BARRAGE;
    human_input_queue_walk(&hi, 20, 20);

    int actions[INF_NUM_ACTION_HEADS];
    inf_translate_human_commands_ctx(&hi, actions, &state, &test_context);
    ASSERT_INT_EQ("walk command sends no spell action",
        actions[INF_HEAD_SPELL], 0);
    ASSERT_INT_EQ("walk command sends no primary action",
        actions[INF_HEAD_PRIMARY], 0);

    human_input_destroy(&hi);
}

static void test_human_autocast_selection_persists_across_weapon_switches(void) {
    printf("--- human autocast selection persists across weapon switches ---\n");

    InfernoState state;
    init_spell_cast_test_state(&state, INF_NPC_RANGER);
    test_context.human_command_mode = 1;
    state.player.autocast_spell = ENCOUNTER_SPELL_BLOOD;

    HumanInput hi;
    human_input_init(&hi);
    human_input_queue_set_autocast(&hi, ENCOUNTER_SPELL_ICE, 1);
    human_input_queue_equip_inventory_item(&hi, 0, ITEM_TWISTED_BOW, GEAR_SLOT_WEAPON);
    human_input_queue_equip_inventory_item(&hi, 0, ITEM_KODAI_WAND, GEAR_SLOT_WEAPON);
    test_context.human_commands = hi.commands.items;
    test_context.human_command_count = hi.commands.count;
    inf_apply_human_player_commands_ctx(&state, &test_context);

    ASSERT_INT_EQ("autocast spell persists after weapon switches",
        state.player.autocast_spell, ENCOUNTER_SPELL_ICE);
    ASSERT_INT_EQ("autocast stays enabled",
        state.player.autocast_enabled, 1);
    ASSERT_INT_EQ("defensive autocast persists",
        state.player.autocast_defensive, 1);
    ASSERT_INT_EQ("kodai returns to defensive autocast stance",
        state.player.fight_style, FIGHT_STYLE_DEFENSIVE_AUTOCAST);

    human_input_destroy(&hi);
}

static void test_autocast_is_inactive_with_non_autocast_weapon(void) {
    printf("--- autocast is inactive with non-autocast weapon ---\n");

    InfernoState state;
    init_spell_cast_test_state(&state, INF_NPC_RANGER);
    state.weapon_set = INF_GEAR_BP;
    encounter_apply_loadout(&state.player, INF_MAX_RANGE_FAST_LOADOUT, GEAR_RANGED);
    inf_refresh_live_stats(&state);
    state.player.autocast_enabled = 1;
    state.player.autocast_spell = ENCOUNTER_SPELL_ICE;

    InfPlayerAttack attack;
    int resolved = inf_resolve_player_attack_ctx(&state, &test_context, ENCOUNTER_SPELL_NONE, &attack);
    ASSERT_INT_EQ("normal attack still resolves",
        resolved, 1);
    ASSERT_INT_EQ("non-autocast weapon ignores remembered autocast",
        attack.stats.style, ATTACK_STYLE_RANGED);
    ASSERT_INT_EQ("remembered autocast spell remains stored",
        state.player.autocast_spell, ENCOUNTER_SPELL_ICE);
}

static void test_echo_boots_recoil_reflects_to_attacker_only(void) {
    printf("--- echo boots recoil reflects to the attacking NPC only ---\n");

    InfernoState state = make_test_state(20, 20);
    memset(state.player.equipped, ITEM_NONE, sizeof(state.player.equipped));
    osrs_item_effect_state_init(&state.player.item_effect_state);
    state.player.equipped[GEAR_SLOT_FEET] = ITEM_ECHO_BOOTS;
    osrs_refresh_player_equipment(&state.player);

    state.npcs[0] = make_test_npc(INF_NPC_BAT, 21, 20, INF_NPC_STATS[INF_NPC_BAT].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = 10;
    state.npcs[1] = make_test_npc(INF_NPC_BAT, 23, 20, INF_NPC_STATS[INF_NPC_BAT].size);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp = 10;
    state.npcs[2] = make_test_npc(INF_NPC_ZUK, 20, 20, INF_NPC_STATS[INF_NPC_ZUK].size);
    state.npcs[2].active = 1;
    state.npcs[2].hp = state.npcs[2].max_hp = 1200;

    inf_apply_echo_boots_recoil(&state, 1, 0);
    ASSERT_INT_EQ("zero damage does not consume echo charge",
        state.player.item_effect_state.echo_boot_charges, OSRS_ECHO_BOOTS_MAX_CHARGES);
    ASSERT_INT_EQ("zero damage does not recoil the attacker",
        state.npcs[1].hp, 10);

    inf_apply_echo_boots_recoil(&state, 1, 7);
    ASSERT_INT_EQ("positive damage consumes one echo charge",
        state.player.item_effect_state.echo_boot_charges, OSRS_ECHO_BOOTS_MAX_CHARGES - 1);
    ASSERT_INT_EQ("the attacker takes echo recoil regardless of distance",
        state.npcs[1].hp, 9);
    ASSERT_INT_EQ("an adjacent bystander is not hit",
        state.npcs[0].hp, 10);
    ASSERT_FLOAT_NEAR("echo recoil records one damage",
        state.tick_scratch.damage_dealt, 1.0f, 1e-6f);

    inf_apply_echo_boots_recoil(&state, 2, 7);
    ASSERT_INT_EQ("Zuk attacker avoids echo recoil",
        state.npcs[2].hp, 1200);
}

static void test_redemption_action_maps_without_smite(void) {
    printf("--- redemption action maps without smite ---\n");

    InfernoState state = make_test_state(20, 20);
    InfernoContext* ctx = &test_context;
    state.player.base_prayer = 99;
    state.player.current_prayer = 99;

    int actions[INF_NUM_ACTION_HEADS] = {0};
    actions[INF_HEAD_PRAYER] =
        ENCOUNTER_OVERHEAD_SET_REFRESH_REDEMPTION;
    inf_player_pretick(&state, ctx, actions);

    ASSERT_INT_EQ("inferno uses the shared overhead actions",
        INF_ACTION_DIMS[INF_HEAD_PRAYER], OSRS_OVERHEAD_DIM);
    ASSERT_INT_EQ("shared redemption action activates redemption",
        state.player.prayer, PRAYER_REDEMPTION);
    ASSERT_INT_EQ("redemption action does not activate smite",
        state.player.prayer == PRAYER_SMITE, 0);
}

static void test_redemption_zero_hit_landing_heals_and_drains(void) {
    printf("--- redemption zero-hit landing heals and drains ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player.base_hitpoints = 99;
    state.player.current_hitpoints = 7;
    state.player.base_prayer = 99;
    state.player.current_prayer = 12;
    state.player.prayer = PRAYER_REDEMPTION;
    state.player.offensive_prayer = OFFENSIVE_PRAYER_RIGOUR;

    inf_damage_player(&state, 0);

    ASSERT_INT_EQ("zero hit procs redemption at low HP",
        state.player.current_hitpoints, 31);
    ASSERT_INT_EQ("redemption drains prayer points",
        state.player.current_prayer, 0);
    ASSERT_INT_EQ("redemption clears overhead",
        state.player.prayer, PRAYER_NONE);
    ASSERT_INT_EQ("redemption clears offensive prayer",
        state.player.offensive_prayer, OFFENSIVE_PRAYER_NONE);
    ASSERT_INT_EQ("zero hit still shows a hitsplat",
        state.player.hit_landed_this_tick, 1);
    ASSERT_INT_EQ("zero hit remains zero damage",
        state.player.hit_damage, 0);
}

static void test_redemption_does_not_prevent_lethal_damage(void) {
    printf("--- redemption does not prevent lethal damage ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player.base_hitpoints = 99;
    state.player.current_hitpoints = 7;
    state.player.base_prayer = 99;
    state.player.current_prayer = 12;
    state.player.prayer = PRAYER_REDEMPTION;

    inf_damage_player(&state, 8);

    ASSERT_INT_EQ("lethal damage still kills through redemption",
        state.player.current_hitpoints, 0);
    ASSERT_INT_EQ("lethal damage does not drain redemption",
        state.player.current_prayer, 12);
}

static void test_redemption_procs_on_locked_zero_projectile_landing(void) {
    printf("--- redemption procs on locked zero projectile landing ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player.base_hitpoints = 99;
    state.player.current_hitpoints = 7;
    state.player.base_prayer = 99;
    state.player.current_prayer = 12;
    state.player.prayer = PRAYER_REDEMPTION;
    state.player_pending_hits.count = 1;
    state.player_pending_hits.hits[0] = (EncounterPendingHit){
        .active = 1,
        .damage = 0,
        .ticks_remaining = 1,
        .attack_style = ATTACK_STYLE_MAGIC,
        .check_prayer = 0,
        .prayer_check_delay = 0,
        .source_npc_type = INF_NPC_HEALER_ZUK,
        .hit_success = 1,
    };

    inf_resolve_player_pending_hits(&state);

    ASSERT_INT_EQ("locked zero projectile lands",
        state.player_pending_hits.count, 0);
    ASSERT_INT_EQ("redemption heals on landing after protection was locked",
        state.player.current_hitpoints, 31);
    ASSERT_INT_EQ("redemption drains prayer on landing",
        state.player.current_prayer, 0);
}

static void test_human_autocast_works_with_dragon_hunter_wand(void) {
    printf("--- human autocast works with dragon hunter wand ---\n");

    InfernoState state;
    init_spell_cast_test_state(&state, INF_NPC_RANGER);
    test_context.human_command_mode = 1;
    state.player.equipped[GEAR_SLOT_WEAPON] = ITEM_DRAGON_HUNTER_WAND;
    state.player.equipped[GEAR_SLOT_SHIELD] = ITEM_CRYSTAL_SHIELD;
    state.player.autocast_enabled = 1;
    state.player.autocast_defensive = 1;
    state.player.autocast_spell = ENCOUNTER_SPELL_ICE;
    state.player.fight_style = FIGHT_STYLE_DEFENSIVE_AUTOCAST;
    osrs_refresh_player_equipment(&state.player);

    InfPlayerAttack attack;
    int resolved = inf_resolve_player_attack_ctx(&state, &test_context, ENCOUNTER_SPELL_NONE, &attack);

    ASSERT_INT_EQ("dragon hunter wand autocast resolves",
        resolved, 1);
    ASSERT_INT_EQ("dragon hunter wand attack stays magic",
        attack.stats.style, ATTACK_STYLE_MAGIC);
    ASSERT_INT_EQ("dragon hunter wand casts remembered ice",
        attack.spell, ENCOUNTER_SPELL_ICE);
    ASSERT_INT_EQ("dragon hunter wand barrage flag",
        attack.is_barrage, 1);
    ASSERT_INT_EQ("dragon hunter wand autocast range",
        attack.stats.attack_range, 10);
}

static void test_inferno_snapshot_restore_round_trip(void) {
    printf("--- inferno snapshot/restore round trip ---\n");

    EncounterState* raw = inf_create();
    InfernoState* state = (InfernoState*)raw;
    inf_reset_ctx(raw, (EncounterContext*)&test_context, 31415u);

    int actions_a[INF_NUM_ACTION_HEADS] = {0};
    actions_a[INF_HEAD_PRIMARY] = 1;
    actions_a[INF_HEAD_PRIMARY] = inf_primary_attack_action_for_obs_slot(1 - 1);
    actions_a[INF_HEAD_PRAYER] = ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE;

    int actions_b[INF_NUM_ACTION_HEADS] = {0};
    actions_b[INF_HEAD_PRIMARY] = 5;
    actions_b[INF_HEAD_PRIMARY] = inf_primary_attack_action_for_obs_slot(2 - 1);
    actions_b[INF_HEAD_PRAYER] = ENCOUNTER_OVERHEAD_SET_REFRESH_RANGED;

    const int N1 = 12;
    for (int i = 0; i < N1; i++) inf_step_ctx(raw, (EncounterContext*)&test_context, actions_a);

    size_t snap_size = inf_snapshot_size_ctx(raw, (EncounterContext*)&test_context);
    ASSERT_INT_EQ("snapshot size matches sizeof(InfSnapshot)",
        (int)snap_size, (int)sizeof(InfSnapshot));
    InfSnapshot* snap_A = (InfSnapshot*)malloc(snap_size);
    inf_snapshot_ctx(raw, (EncounterContext*)&test_context, snap_A);
    ASSERT_INT_EQ("snapshot magic stamped",
        (int)snap_A->magic, (int)INF_SNAPSHOT_MAGIC);
    ASSERT_INT_EQ("snapshot version stamped",
        (int)snap_A->version, 22);

    const int N2 = 18;
    for (int i = 0; i < N2; i++) inf_step_ctx(raw, (EncounterContext*)&test_context, actions_b);

    InfSnapshot* snap_B = (InfSnapshot*)malloc(snap_size);
    inf_snapshot_ctx(raw, (EncounterContext*)&test_context, snap_B);

    inf_restore_ctx(raw, (EncounterContext*)&test_context, snap_A, snap_size);
    ASSERT_INT_EQ("tick reset to N1 after restore", state->tick, N1);
    for (int i = 0; i < N2; i++) inf_step_ctx(raw, (EncounterContext*)&test_context, actions_b);

    InfSnapshot* snap_B_prime = (InfSnapshot*)malloc(snap_size);
    inf_snapshot_ctx(raw, (EncounterContext*)&test_context, snap_B_prime);

    int diff = memcmp(&snap_B->state, &snap_B_prime->state, sizeof(InfernoState));
    ASSERT_INT_EQ("memcmp(state at N1+N2, state after restore+replay) == 0", diff, 0);

    InfernoState* a = &snap_B->state;
    InfernoState* b = &snap_B_prime->state;
    ASSERT_INT_EQ("tick", a->tick, b->tick);
    ASSERT_INT_EQ("wave", a->wave, b->wave);
    ASSERT_INT_EQ("episode_over", a->episode_over, b->episode_over);
    ASSERT_INT_EQ("winner", a->winner, b->winner);
    ASSERT_INT_EQ("rng_state", (int)a->rng_state, (int)b->rng_state);
    ASSERT_INT_EQ("player x", a->player.x, b->player.x);
    ASSERT_INT_EQ("player y", a->player.y, b->player.y);
    ASSERT_INT_EQ("player hp", a->player.current_hitpoints, b->player.current_hitpoints);
    ASSERT_INT_EQ("player prayer", a->player.current_prayer, b->player.current_prayer);

    free(snap_A);
    free(snap_B);
    free(snap_B_prime);
    inf_destroy(raw);
}

static void test_inferno_snapshot_preserves_loadout_profile(void) {
    printf("--- inferno snapshot preserves loadout profile ---\n");

    EncounterState* raw = inf_create();
    InfernoState* state = (InfernoState*)raw;

    inf_put_int_ctx(raw, (EncounterContext*)&test_context, "loadout_profile_mode", INF_LOADOUT_PROFILE_MODE_BUDGET_ONLY);
    inf_put_float_ctx(raw, (EncounterContext*)&test_context, "budget_loadout_fraction", 1.0f);
    inf_reset_ctx(raw, (EncounterContext*)&test_context, 314u);

    size_t snap_size = inf_snapshot_size_ctx(raw, (EncounterContext*)&test_context);
    InfSnapshot* snap = (InfSnapshot*)malloc(snap_size);
    inf_snapshot_ctx(raw, (EncounterContext*)&test_context, snap);

    state->active_loadout_profile = INF_LOADOUT_PROFILE_MAX;

    inf_restore_ctx(raw, (EncounterContext*)&test_context, snap, snap_size);

    ASSERT_INT_EQ("restored active budget profile",
        state->active_loadout_profile, INF_LOADOUT_PROFILE_BUDGET);
    ASSERT_INT_EQ("restore keeps live loadout profile mode",
        test_config()->loadout_profile_mode, INF_LOADOUT_PROFILE_MODE_BUDGET_ONLY);
    ASSERT_FLOAT_NEAR("restore keeps live budget loadout fraction",
        test_config()->budget_loadout_fraction, 1.0f, 1e-6f);

    free(snap);
    inf_destroy(raw);
}

static void test_inferno_restore_builds_npc_stats_before_late_spawn(void) {
    printf("--- inferno restore builds NPC stats before late spawn ---\n");

    EncounterState* raw_a = inf_create();
    reset_inferno_at_public_wave(raw_a, 69, 1.0f);

    InfSnapshot snap;
    inf_snapshot_ctx(raw_a, (EncounterContext*)&test_context, &snap);
    inf_destroy(raw_a);

    EncounterState* raw_b = inf_create();
    inf_put_int_ctx(raw_b, (EncounterContext*)&test_context, "start_wave", 69);
    memset(INF_NPC_STATS, 0, sizeof(INF_NPC_STATS));
    inf_restore_ctx(raw_b, (EncounterContext*)&test_context, &snap, sizeof(snap));

    InfernoState* state = (InfernoState*)raw_b;
    int zuk_idx = find_active_npc_type(state, INF_NPC_ZUK);
    ASSERT_INT_EQ("Zuk exists after restore", zuk_idx >= 0, 1);
    state->wave_ready_delay = 0;
    state->zuk.set_timer = 1;
    state->npcs[zuk_idx].attack_timer = 999;

    int actions[INF_NUM_ACTION_HEADS] = {0};
    inf_step_ctx(raw_b, (EncounterContext*)&test_context, actions);

    int mager_idx = find_active_npc_type(state, INF_NPC_MAGER);
    int ranger_idx = find_active_npc_type(state, INF_NPC_RANGER);
    ASSERT_INT_EQ("restored env spawned mager", mager_idx >= 0, 1);
    ASSERT_INT_EQ("restored env spawned ranger", ranger_idx >= 0, 1);
    ASSERT_INT_EQ("spawned mager has real HP",
        state->npcs[mager_idx].hp, INF_NPC_STATS[INF_NPC_MAGER].hp);
    ASSERT_INT_EQ("spawned ranger has real HP",
        state->npcs[ranger_idx].hp, INF_NPC_STATS[INF_NPC_RANGER].hp);
    ASSERT_INT_EQ("spawned mager has real max HP",
        state->npcs[mager_idx].max_hp, INF_NPC_STATS[INF_NPC_MAGER].hp);
    ASSERT_INT_EQ("spawned ranger has real max HP",
        state->npcs[ranger_idx].max_hp, INF_NPC_STATS[INF_NPC_RANGER].hp);
    ASSERT_INT_EQ("spawned mager did not attack on spawn tick",
        state->npcs[mager_idx].attacked_this_tick, 0);
    ASSERT_INT_EQ("spawned ranger did not attack on spawn tick",
        state->npcs[ranger_idx].attacked_this_tick, 0);

    inf_destroy(raw_b);
}

static void test_inferno_snapshot_preserves_external_pointers(void) {
    printf("--- inferno snapshot preserves external pointers across restore ---\n");

    InfernoState state_a;
    InfernoState state_b;
    InfernoContext ctx_a;
    InfernoContext ctx_b;
    Log log_a = {0};
    Log log_b = {0};
    int dummy_a = 1, dummy_b = 2;

    inf_init_context_typed(&ctx_a);
    inf_init_context_typed(&ctx_b);
    ctx_a.collision_map = (const CollisionMap*)&dummy_a;
    ctx_b.collision_map = (const CollisionMap*)&dummy_b;
    ctx_a.world_offset_x = 100;
    ctx_a.world_offset_y = 200;
    ctx_b.world_offset_x = 300;
    ctx_b.world_offset_y = 400;
    ctx_a.log = &log_a;
    ctx_b.log = &log_b;
    ctx_a.config.start_wave = 12;
    ctx_b.config = ctx_a.config;
    ctx_b.human_command_mode = 1;

    inf_reset_ctx((EncounterState*)&state_a, (EncounterContext*)&ctx_a, 7u);
    inf_reset_ctx((EncounterState*)&state_b, (EncounterContext*)&ctx_b, 7u);

    size_t snap_size = inf_snapshot_size_ctx(
        (EncounterState*)&state_a, (EncounterContext*)&ctx_a);
    InfSnapshot* snap = (InfSnapshot*)malloc(snap_size);
    inf_snapshot_ctx((EncounterState*)&state_a, (EncounterContext*)&ctx_a, snap);

    inf_restore_ctx((EncounterState*)&state_b, (EncounterContext*)&ctx_b, snap, snap_size);
    ASSERT_INT_EQ("env B keeps its own collision_map after restore",
        (int)(ctx_b.collision_map == (const CollisionMap*)&dummy_b), 1);
    ASSERT_INT_EQ("env A snapshot did not leak its collision_map into B",
        (int)(ctx_b.collision_map != (const CollisionMap*)&dummy_a), 1);
    ASSERT_INT_EQ("env B keeps world offset x after restore", ctx_b.world_offset_x, 300);
    ASSERT_INT_EQ("env B keeps world offset y after restore", ctx_b.world_offset_y, 400);
    ASSERT_INT_EQ("env B keeps live log pointer after restore",
        (int)(ctx_b.log == &log_b), 1);
    ASSERT_INT_EQ("env B keeps live start_wave after restore",
        ctx_b.config.start_wave, 12);
    ASSERT_INT_EQ("env B keeps live human command mode after restore",
        ctx_b.human_command_mode, 1);

    free(snap);
}

static void test_inferno_state_assignment_copy_replays_trajectory(void) {
    printf("--- inferno state assignment copy replays trajectory ---\n");

    InfernoContext ctx_a;
    InfernoContext ctx_b;
    InfernoState state_a;
    InfernoState state_b;
    inf_init_context_typed(&ctx_a);
    inf_init_context_typed(&ctx_b);
    inf_init_state_typed(&state_a, &ctx_a);
    inf_init_state_typed(&state_b, &ctx_b);
    inf_reset_ctx((EncounterState*)&state_a, (EncounterContext*)&ctx_a, 987u);

    int prefix[INF_NUM_ACTION_HEADS] = {0};
    prefix[INF_HEAD_PRIMARY] = 1;
    prefix[INF_HEAD_PRIMARY] = inf_primary_attack_action_for_obs_slot(1 - 1);
    prefix[INF_HEAD_EQUIP_SLOT(GEAR_SLOT_WEAPON)] =
        test_cell_holding_item(&state_a, ITEM_TWISTED_BOW) + 1;

    for (int i = 0; i < 9; i++)
        inf_step_ctx((EncounterState*)&state_a, (EncounterContext*)&ctx_a, prefix);

    state_b = state_a;
    inf_refresh_after_state_load(&state_b, &ctx_b);

    int suffix[INF_NUM_ACTION_HEADS] = {0};
    suffix[INF_HEAD_PRIMARY] = 5;
    suffix[INF_HEAD_PRIMARY] = inf_primary_attack_action_for_obs_slot(2 - 1);
    suffix[INF_HEAD_PRAYER] = ENCOUNTER_OVERHEAD_SET_REFRESH_RANGED;
    suffix[INF_HEAD_OFFENSIVE] = ENCOUNTER_OFFENSIVE_SET_REFRESH_RIGOUR;

    for (int i = 0; i < 17; i++) {
        inf_step_ctx((EncounterState*)&state_a, (EncounterContext*)&ctx_a, suffix);
        inf_step_ctx((EncounterState*)&state_b, (EncounterContext*)&ctx_b, suffix);
    }

    inf_refresh_after_state_load(&state_a, &ctx_a);
    inf_refresh_after_state_load(&state_b, &ctx_b);
    ASSERT_INT_EQ("assignment copy replay state memcmp",
        memcmp(&state_a, &state_b, sizeof(InfernoState)), 0);
}

static void test_inferno_refresh_after_state_load_rebuilds_derived_state(void) {
    printf("--- inferno refresh after state load rebuilds derived state ---\n");

    InfernoContext ctx;
    InfernoState state;
    inf_init_context_typed(&ctx);
    inf_init_state_typed(&state, &ctx);
    inf_reset_ctx((EncounterState*)&state, (EncounterContext*)&ctx, 42u);

    memset(state.current_obs_slots, -1, sizeof(state.current_obs_slots));
    state.loadout_stats[INF_GEAR_LONG_RANGE].max_hit = -1;

    inf_refresh_after_state_load(&state, &ctx);

    int visible_count = 0;
    for (int i = 0; i < INF_OBS_NPCS; i++) {
        if (state.current_obs_slots[i] >= 0)
            visible_count++;
    }

    ASSERT_INT_EQ("refresh repopulates visible obs slots", visible_count > 0, 1);
    ASSERT_INT_EQ("refresh recomputes long-range max hit",
        state.loadout_stats[INF_GEAR_LONG_RANGE].max_hit > 0, 1);
}

static void test_inferno_healer_transition_stats_track_episode_progress(void) {
    printf("--- inferno healer transition stats track episode progress ---\n");

    InfernoState state = make_test_state(INF_ZUK_PLAYER_START_X, INF_ZUK_PLAYER_START_Y);
    state.tick = 120;
    state.tick_at_le_240 = -1;
    state.tick_at_zuk_healer_spawn = -1;
    state.tick_at_first_zuk_healer_tag = -1;
    state.tick_at_all_zuk_healers_tagged = -1;
    state.tick_at_all_zuk_healers_dead = -1;
    state.zuk.healer_spawned = 1;

    state.npcs[0] = make_test_npc(INF_NPC_ZUK, 20, 52, 5);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 239;
    state.npcs[0].max_hp = 1200;

    for (int i = 1; i <= 4; i++) {
        state.npcs[i] = make_test_npc(INF_NPC_HEALER_ZUK, 15 + i, 48, 1);
        state.npcs[i].active = 1;
        state.npcs[i].hp = 100;
        state.npcs[i].max_hp = 100;
    }

    state.tick_scratch.zuk_healer_tags = 1;
    state.tick_scratch.hp_restored = 21.0f;
    state.tick_scratch.spark_damage = 7.0f;
    inf_update_healer_transition_stats(&state);

    ASSERT_INT_EQ("healer spawn tick recorded",
        state.tick_at_zuk_healer_spawn, 120);
    ASSERT_INT_EQ("healer stats infer 240 threshold tick",
        state.tick_at_le_240, 120);
    ASSERT_INT_EQ("first healer tag tick recorded",
        state.tick_at_first_zuk_healer_tag, 120);
    ASSERT_INT_EQ("one healer tag accumulated",
        state.total_zuk_healer_tags, 1);
    ASSERT_FLOAT_NEAR("post-240 restored hp accumulated",
        state.hp_restored_after_240, 21.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("post-240 spark damage accumulated",
        state.spark_damage_after_240, 7.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("post-spawn max Zuk HP initialized",
        state.zuk_hp_max_after_healer_spawn, 239.0f, 1e-6f);

    state.tick = 121;
    state.npcs[0].hp = 420;
    state.tick_scratch.zuk_healer_tags = 3;
    state.tick_scratch.kill_zuk_healer = 2;
    state.tick_scratch.hp_restored = 13.0f;
    state.tick_scratch.spark_damage = 5.0f;
    inf_update_healer_transition_stats(&state);

    ASSERT_INT_EQ("all healer tag tick recorded",
        state.tick_at_all_zuk_healers_tagged, 121);
    ASSERT_INT_EQ("four healer tags accumulated",
        state.total_zuk_healer_tags, 4);
    ASSERT_INT_EQ("two healer kills accumulated",
        state.total_zuk_healer_kills, 2);
    ASSERT_FLOAT_NEAR("post-spawn max Zuk HP tracks healer restore",
        state.zuk_hp_max_after_healer_spawn, 420.0f, 1e-6f);

    state.tick = 122;
    state.tick_scratch.zuk_healer_tags = 0;
    state.tick_scratch.kill_zuk_healer = 2;
    state.tick_scratch.hp_restored = 0.0f;
    state.tick_scratch.spark_damage = 0.0f;
    for (int i = 1; i <= 4; i++)
        state.npcs[i].active = 0;
    inf_update_healer_transition_stats(&state);

    ASSERT_INT_EQ("all healer dead tick recorded",
        state.tick_at_all_zuk_healers_dead, 122);
    ASSERT_INT_EQ("four healer kills accumulated",
        state.total_zuk_healer_kills, 4);
}

static void test_inferno_human_equip_does_not_snap_loadout(void) {
    printf("--- inferno human equip does not snap full loadout ---\n");

    EncounterState* raw = inf_create();
    InfernoState* state = (InfernoState*)raw;
    inf_reset_ctx(raw, (EncounterContext*)&test_context, 123);

    HumanInput input;
    human_input_init(&input);
    input.enabled = 1;

    uint8_t old_body = state->player.equipped[GEAR_SLOT_BODY];
    human_input_queue_equip_inventory_item(
        &input, 0, ITEM_TOXIC_BLOWPIPE, GEAR_SLOT_WEAPON);

    inf_step_human_commands_ctx(raw, (EncounterContext*)&test_context, &input);

    ASSERT_INT_EQ("weapon changed to clicked blowpipe",
        state->player.equipped[GEAR_SLOT_WEAPON], ITEM_TOXIC_BLOWPIPE);
    ASSERT_INT_EQ("body slot did not snap to ranged preset",
        state->player.equipped[GEAR_SLOT_BODY], old_body);
    ASSERT_INT_EQ("2h weapon clears shield",
        state->player.equipped[GEAR_SLOT_SHIELD], ITEM_NONE);
    ASSERT_INT_EQ("queued command drained", input.commands.count, 0);

    human_input_destroy(&input);
    inf_destroy(raw);
}

static void test_jad_render_uses_style_specific_attack_animation(void) {
    printf("--- jad render uses style-specific attack animation ---\n");

    InfernoState magic_state;
    init_jad_timing_test_state(&magic_state, 10, 10, 16, 10);
    magic_state.npcs[0].attacked_this_tick = 1;
    magic_state.npcs[0].attack_style_this_tick = ATTACK_STYLE_MAGIC;

    RenderEntity magic_entities[4];
    int magic_count = 0;
    inf_fill_render_entities_ctx((EncounterState*)&magic_state, (EncounterContext*)&test_context, magic_entities, 4, &magic_count);

    InfernoState range_state;
    init_jad_timing_test_state(&range_state, 10, 10, 16, 10);
    range_state.npcs[0].attacked_this_tick = 1;
    range_state.npcs[0].attack_style_this_tick = ATTACK_STYLE_RANGED;

    RenderEntity range_entities[4];
    int range_count = 0;
    inf_fill_render_entities_ctx((EncounterState*)&range_state, (EncounterContext*)&test_context, range_entities, 4, &range_count);

    ASSERT_INT_EQ("jad magic render entity count", magic_count, 2);
    ASSERT_INT_EQ("jad ranged render entity count", range_count, 2);
    ASSERT_INT_EQ("jad magic attack animation", magic_entities[1].npc_anim_id, 7592);
    ASSERT_INT_EQ("jad ranged attack animation", range_entities[1].npc_anim_id, 7593);
}

static void test_inferno_render_uses_npc_death_animation(void) {
    printf("--- inferno render uses NPC death animation ---\n");

    InfernoState state;
    init_phantom_barrage_test_state(&state, 2, 1);

    RenderEntity entities[4];
    int count = 0;
    inf_fill_render_entities_ctx((EncounterState*)&state, (EncounterContext*)&test_context, entities, 4, &count);

    ASSERT_INT_EQ("dying NPC still renders", count >= 2, 1);
    ASSERT_INT_EQ("nibbler death animation",
        entities[1].npc_anim_id, INF_GEN_ANIM_NIBBLER_DEATH);
}

static void test_jad_magic_render_emits_three_offset_projectiles(void) {
    printf("--- jad magic render emits three offset projectiles ---\n");

    InfernoState state;
    init_jad_timing_test_state(&state, 10, 10, 16, 10);
    state.npcs[0].attacked_this_tick = 1;
    state.npcs[0].attack_style_this_tick = ATTACK_STYLE_MAGIC;

    EncounterOverlay ov;
    memset(&ov, 0, sizeof(ov));
    inf_render_post_tick_ctx((EncounterState*)&state, (EncounterContext*)&test_context, &ov);

    ASSERT_INT_EQ("jad magic emits three projectile models", ov.projectile_count, 3);
    ASSERT_INT_EQ("jad magic front model", ov.projectiles[0].model_id, INF_GFX_448_MODEL);
    ASSERT_INT_EQ("jad magic middle model", ov.projectiles[1].model_id, INF_GFX_449_MODEL);
    ASSERT_INT_EQ("jad magic rear model", ov.projectiles[2].model_id, INF_GFX_450_MODEL);
    ASSERT_INT_EQ("jad magic front anim", ov.projectiles[0].anim_id, INF_GFX_448_ANIM);
    ASSERT_INT_EQ("jad magic middle anim", ov.projectiles[1].anim_id, INF_GFX_449_ANIM);
    ASSERT_INT_EQ("jad magic rear anim", ov.projectiles[2].anim_id, INF_GFX_450_ANIM);
    ASSERT_INT_EQ("jad magic visible duration is two ticks close range", ov.projectiles[0].duration_ticks, 2 * 30);
    ASSERT_INT_EQ("jad magic start delay is three ticks", ov.projectiles[0].start_delay, 3 * 30);
    ASSERT_FLOAT_NEAR("jad magic arc height", ov.projectiles[0].arc_height, 1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("jad magic front offset", ov.projectiles[0].offset_y, 1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("jad magic middle offset", ov.projectiles[1].offset_y, 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR("jad magic rear offset", ov.projectiles[2].offset_y, 0.0f, 1e-6f);
}

static void test_jad_ranged_render_uses_target_anchored_two_tick_visual(void) {
    printf("--- jad ranged render uses target anchored two tick visual ---\n");

    InfernoState state;
    init_jad_timing_test_state(&state, 10, 10, 16, 10);
    state.npcs[0].attacked_this_tick = 1;
    state.npcs[0].attack_style_this_tick = ATTACK_STYLE_RANGED;

    EncounterOverlay ov;
    memset(&ov, 0, sizeof(ov));
    inf_render_post_tick_ctx((EncounterState*)&state, (EncounterContext*)&test_context, &ov);

    ASSERT_INT_EQ("jad ranged emits one projectile", ov.projectile_count, 1);
    ASSERT_INT_EQ("jad ranged model", ov.projectiles[0].model_id, INF_GFX_451_MODEL);
    ASSERT_INT_EQ("jad ranged anim", ov.projectiles[0].anim_id, INF_GFX_451_ANIM);
    ASSERT_INT_EQ("jad ranged target-anchored motion",
        ov.projectiles[0].motion_mode, ENCOUNTER_PROJECTILE_MOTION_TARGET_ANCHORED);
    ASSERT_INT_EQ("jad ranged start height is player target height", ov.projectiles[0].start_h, 64);
    ASSERT_INT_EQ("jad ranged end height is player target height", ov.projectiles[0].end_h, 64);
    ASSERT_INT_EQ("jad ranged visible duration is two ticks close range", ov.projectiles[0].duration_ticks, 2 * 30);
    ASSERT_INT_EQ("jad ranged start delay is three ticks", ov.projectiles[0].start_delay, 3 * 30);
}

static void test_jad_projectile_long_distance_visual_duration_uses_reference_formula(void) {
    printf("--- jad long-distance projectile visual duration uses reference formula ---\n");

    InfernoState range_state;
    init_jad_timing_test_state(&range_state, 10, 10, 36, 10);
    range_state.npcs[0].attacked_this_tick = 1;
    range_state.npcs[0].attack_style_this_tick = ATTACK_STYLE_RANGED;

    EncounterOverlay range_ov;
    memset(&range_ov, 0, sizeof(range_ov));
    inf_render_post_tick_ctx((EncounterState*)&range_state, (EncounterContext*)&test_context, &range_ov);

    int range_dist = encounter_projectile_distance(
        range_state.npcs[0].x, range_state.npcs[0].y, range_state.npcs[0].size,
        range_state.player.x, range_state.player.y, 1,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    EncounterProjectileTiming range_timing =
        inf_npc_projectile_timing(INF_NPC_JAD, ATTACK_STYLE_RANGED, range_dist);
    ASSERT_INT_EQ("jad ranged long-distance duration",
        range_ov.projectiles[0].duration_ticks,
        range_timing.visual_duration_ticks * 30);

    InfernoState magic_state;
    init_jad_timing_test_state(&magic_state, 10, 10, 36, 10);
    magic_state.npcs[0].attacked_this_tick = 1;
    magic_state.npcs[0].attack_style_this_tick = ATTACK_STYLE_MAGIC;

    EncounterOverlay magic_ov;
    memset(&magic_ov, 0, sizeof(magic_ov));
    inf_render_post_tick_ctx((EncounterState*)&magic_state, (EncounterContext*)&test_context, &magic_ov);

    int magic_dist = encounter_projectile_distance(
        magic_state.npcs[0].x, magic_state.npcs[0].y, magic_state.npcs[0].size,
        magic_state.player.x, magic_state.player.y, 1,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    EncounterProjectileTiming magic_timing =
        inf_npc_projectile_timing(INF_NPC_JAD, ATTACK_STYLE_MAGIC, magic_dist);
    ASSERT_INT_EQ("jad magic long-distance duration",
        magic_ov.projectiles[0].duration_ticks,
        magic_timing.visual_duration_ticks * 30);
}

static void test_inferno_npc_projectile_render_uses_reference_visual_timing(void) {
    printf("--- inferno npc projectile render uses reference visual timing ---\n");

    InfernoState mager_state = make_test_state(10, 10);
    mager_state.npcs[0] = make_test_npc(
        INF_NPC_MAGER, 16, 10, INF_NPC_STATS[INF_NPC_MAGER].size);
    mager_state.npcs[0].active = 1;
    mager_state.npcs[0].attacked_this_tick = 1;
    mager_state.npcs[0].attack_style_this_tick = ATTACK_STYLE_MAGIC;

    EncounterOverlay mager_ov;
    memset(&mager_ov, 0, sizeof(mager_ov));
    inf_render_post_tick_ctx((EncounterState*)&mager_state, (EncounterContext*)&test_context, &mager_ov);

    int mager_dist = encounter_projectile_distance(
        mager_state.npcs[0].x, mager_state.npcs[0].y, mager_state.npcs[0].size,
        mager_state.player.x, mager_state.player.y, 1,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    EncounterProjectileTiming mager_timing =
        inf_npc_projectile_timing(INF_NPC_MAGER, ATTACK_STYLE_MAGIC, mager_dist);

    ASSERT_INT_EQ("mager projectile count", mager_ov.projectile_count, 1);
    ASSERT_INT_EQ("mager projectile model",
        mager_ov.projectiles[0].model_id, INF_GFX_1376_MODEL);
    ASSERT_INT_EQ("mager projectile animation",
        mager_ov.projectiles[0].anim_id, INF_GFX_1376_ANIM);
    ASSERT_INT_EQ("mager impact spotanim",
        mager_ov.projectiles[0].impact_gfx_id, 0);
    ASSERT_INT_EQ("mager projectile tracks player", mager_ov.projectiles[0].tracks_target, 1);
    ASSERT_INT_EQ("mager projectile target kind",
        mager_ov.projectiles[0].target_kind, ENCOUNTER_PROJECTILE_TARGET_PLAYER);
    ASSERT_INT_EQ("mager projectile source kind",
        mager_ov.projectiles[0].source_kind, ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT);
    ASSERT_INT_EQ("mager projectile source slot",
        mager_ov.projectiles[0].source_npc_slot, 0);
    ASSERT_INT_EQ("mager visual start delay",
        mager_ov.projectiles[0].start_delay, mager_timing.visual_start_delay_ticks * 30);
    ASSERT_INT_EQ("mager visual duration",
        mager_ov.projectiles[0].duration_ticks, mager_timing.visual_duration_ticks * 30);

    InfernoState ranger_state = make_test_state(10, 10);
    ranger_state.npcs[0] = make_test_npc(
        INF_NPC_RANGER, 16, 10, INF_NPC_STATS[INF_NPC_RANGER].size);
    ranger_state.npcs[0].active = 1;
    ranger_state.npcs[0].attacked_this_tick = 1;
    ranger_state.npcs[0].attack_style_this_tick = ATTACK_STYLE_RANGED;

    EncounterOverlay ranger_ov;
    memset(&ranger_ov, 0, sizeof(ranger_ov));
    inf_render_post_tick_ctx((EncounterState*)&ranger_state, (EncounterContext*)&test_context, &ranger_ov);

    int ranger_dist = encounter_projectile_distance(
        ranger_state.npcs[0].x, ranger_state.npcs[0].y, ranger_state.npcs[0].size,
        ranger_state.player.x, ranger_state.player.y, 1,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    EncounterProjectileTiming ranger_timing =
        inf_npc_projectile_timing(INF_NPC_RANGER, ATTACK_STYLE_RANGED, ranger_dist);

    ASSERT_INT_EQ("ranger projectile count", ranger_ov.projectile_count, 1);
    ASSERT_INT_EQ("ranger projectile model",
        ranger_ov.projectiles[0].model_id, INF_GFX_1377_MODEL);
    ASSERT_INT_EQ("ranger impact spotanim",
        ranger_ov.projectiles[0].impact_gfx_id, 0);
    ASSERT_INT_EQ("ranger projectile tracks player", ranger_ov.projectiles[0].tracks_target, 1);
    ASSERT_INT_EQ("ranger projectile target kind",
        ranger_ov.projectiles[0].target_kind, ENCOUNTER_PROJECTILE_TARGET_PLAYER);
    ASSERT_INT_EQ("ranger projectile source kind",
        ranger_ov.projectiles[0].source_kind, ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT);
    ASSERT_INT_EQ("ranger projectile source slot",
        ranger_ov.projectiles[0].source_npc_slot, 0);
    ASSERT_INT_EQ("ranger visual start delay",
        ranger_ov.projectiles[0].start_delay, ranger_timing.visual_start_delay_ticks * 30);
    ASSERT_INT_EQ("ranger visual duration",
        ranger_ov.projectiles[0].duration_ticks, ranger_timing.visual_duration_ticks * 30);

    InfernoState blob_state = make_test_state(10, 10);
    blob_state.npcs[3] = make_test_npc(
        INF_NPC_BLOB, 16, 10, INF_NPC_STATS[INF_NPC_BLOB].size);
    blob_state.npcs[3].active = 1;
    blob_state.npcs[3].attacked_this_tick = 1;
    blob_state.npcs[3].attack_style_this_tick = ATTACK_STYLE_RANGED;

    EncounterOverlay blob_ov;
    memset(&blob_ov, 0, sizeof(blob_ov));
    inf_render_post_tick_ctx((EncounterState*)&blob_state, (EncounterContext*)&test_context, &blob_ov);

    ASSERT_INT_EQ("blob projectile count", blob_ov.projectile_count, 1);
    ASSERT_INT_EQ("blob ranged projectile model",
        blob_ov.projectiles[0].model_id, INF_GFX_1383_MODEL);
    ASSERT_INT_EQ("blob ranged projectile animation",
        blob_ov.projectiles[0].anim_id, INF_GFX_1383_ANIM);
    ASSERT_INT_EQ("blob ranged impact spotanim",
        blob_ov.projectiles[0].impact_gfx_id, 0);
    ASSERT_INT_EQ("blob projectile source kind",
        blob_ov.projectiles[0].source_kind, ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT);
    ASSERT_INT_EQ("blob projectile source slot", blob_ov.projectiles[0].source_npc_slot, 3);
    ASSERT_INT_EQ("blob projectile target kind",
        blob_ov.projectiles[0].target_kind, ENCOUNTER_PROJECTILE_TARGET_PLAYER);

    InfernoState blob_magic_state = make_test_state(10, 10);
    blob_magic_state.npcs[3] = make_test_npc(
        INF_NPC_BLOB, 16, 10, INF_NPC_STATS[INF_NPC_BLOB].size);
    blob_magic_state.npcs[3].active = 1;
    blob_magic_state.npcs[3].attacked_this_tick = 1;
    blob_magic_state.npcs[3].attack_style_this_tick = ATTACK_STYLE_MAGIC;

    EncounterOverlay blob_magic_ov;
    memset(&blob_magic_ov, 0, sizeof(blob_magic_ov));
    inf_render_post_tick_ctx((EncounterState*)&blob_magic_state, (EncounterContext*)&test_context, &blob_magic_ov);

    ASSERT_INT_EQ("blob magic projectile count", blob_magic_ov.projectile_count, 1);
    ASSERT_INT_EQ("blob magic projectile model",
        blob_magic_ov.projectiles[0].model_id, INF_GFX_1384_MODEL);
    ASSERT_INT_EQ("blob magic projectile has no placeholder animation",
        blob_magic_ov.projectiles[0].anim_id, OSRS_COMBAT_PROJECTILE_MISSING);
    ASSERT_INT_EQ("blob magic impact spotanim",
        blob_magic_ov.projectiles[0].impact_gfx_id, 0);

    InfernoState blob_split_range_state = make_test_state(10, 10);
    blob_split_range_state.npcs[3] = make_test_npc(
        INF_NPC_BLOB_RANGE, 16, 10, INF_NPC_STATS[INF_NPC_BLOB_RANGE].size);
    blob_split_range_state.npcs[3].active = 1;
    blob_split_range_state.npcs[3].attacked_this_tick = 1;
    blob_split_range_state.npcs[3].attack_style_this_tick = ATTACK_STYLE_RANGED;

    EncounterOverlay blob_split_range_ov;
    memset(&blob_split_range_ov, 0, sizeof(blob_split_range_ov));
    inf_render_post_tick_ctx((EncounterState*)&blob_split_range_state, (EncounterContext*)&test_context, &blob_split_range_ov);

    ASSERT_INT_EQ("blob split ranged projectile count",
        blob_split_range_ov.projectile_count, 1);
    ASSERT_INT_EQ("blob split ranged projectile model",
        blob_split_range_ov.projectiles[0].model_id, INF_GFX_1379_MODEL);
    ASSERT_INT_EQ("blob split ranged projectile animation",
        blob_split_range_ov.projectiles[0].anim_id, INF_GFX_1379_ANIM);

    InfernoState blob_split_magic_state = make_test_state(10, 10);
    blob_split_magic_state.npcs[3] = make_test_npc(
        INF_NPC_BLOB_MAGE, 16, 10, INF_NPC_STATS[INF_NPC_BLOB_MAGE].size);
    blob_split_magic_state.npcs[3].active = 1;
    blob_split_magic_state.npcs[3].attacked_this_tick = 1;
    blob_split_magic_state.npcs[3].attack_style_this_tick = ATTACK_STYLE_MAGIC;

    EncounterOverlay blob_split_magic_ov;
    memset(&blob_split_magic_ov, 0, sizeof(blob_split_magic_ov));
    inf_render_post_tick_ctx((EncounterState*)&blob_split_magic_state, (EncounterContext*)&test_context, &blob_split_magic_ov);

    ASSERT_INT_EQ("blob split magic projectile count",
        blob_split_magic_ov.projectile_count, 1);
    ASSERT_INT_EQ("blob split magic projectile model",
        blob_split_magic_ov.projectiles[0].model_id, INF_GFX_1381_MODEL);
    ASSERT_INT_EQ("blob split magic projectile animation",
        blob_split_magic_ov.projectiles[0].anim_id, INF_GFX_1381_ANIM);
}

static void test_inferno_npc_projectile_render_tracks_target_npc_slot(void) {
    printf("--- inferno npc projectile render tracks target npc slot ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    state.npcs[2] = make_test_npc(
        INF_NPC_MAGER, 20, 36, INF_NPC_STATS[INF_NPC_MAGER].size);
    state.npcs[2].active = 1;
    state.npcs[2].attacked_this_tick = 1;
    state.npcs[2].attack_style_this_tick = ATTACK_STYLE_MAGIC;
    state.npcs[2].attack_visual_target = 1;

    EncounterOverlay ov;
    memset(&ov, 0, sizeof(ov));
    inf_render_post_tick_ctx((EncounterState*)&state, (EncounterContext*)&test_context, &ov);

    ASSERT_INT_EQ("shield-target projectile count", ov.projectile_count, 1);
    ASSERT_INT_EQ("shield-target projectile tracks target", ov.projectiles[0].tracks_target, 1);
    ASSERT_INT_EQ("shield-target projectile target kind",
        ov.projectiles[0].target_kind, ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT);
    ASSERT_INT_EQ("shield-target projectile target slot", ov.projectiles[0].target_npc_slot, 1);
    ASSERT_INT_EQ("shield-target projectile source kind",
        ov.projectiles[0].source_kind, ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT);
    ASSERT_INT_EQ("shield-target projectile source slot", ov.projectiles[0].source_npc_slot, 2);
}

static void test_inferno_zuk_projectile_render_uses_combat_visual_rows(void) {
    printf("--- inferno zuk projectile render uses combat visual rows ---\n");

    InfernoState state;
    init_zuk_timing_state(&state);
    state.npcs[0].attacked_this_tick = 1;
    state.npcs[0].attack_style_this_tick = ATTACK_STYLE_NONE;

    EncounterOverlay ov;
    memset(&ov, 0, sizeof(ov));
    inf_render_post_tick_ctx((EncounterState*)&state, (EncounterContext*)&test_context, &ov);

    ASSERT_INT_EQ("zuk projectile count", ov.projectile_count, 1);
    ASSERT_INT_EQ("zuk projectile model", ov.projectiles[0].model_id, INF_GFX_1375_MODEL);
    ASSERT_INT_EQ("zuk projectile animation", ov.projectiles[0].anim_id, INF_GFX_1375_ANIM);
    ASSERT_INT_EQ("zuk projectile source kind",
        ov.projectiles[0].source_kind, ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT);
    ASSERT_INT_EQ("zuk projectile source slot", ov.projectiles[0].source_npc_slot, 0);

    InfernoState healer_state;
    init_zuk_timing_state(&healer_state);
    healer_state.npcs[2] = make_test_npc(
        INF_NPC_HEALER_ZUK, 28, 49, INF_NPC_STATS[INF_NPC_HEALER_ZUK].size);
    healer_state.npcs[2].active = 1;
    healer_state.npcs[2].attacked_this_tick = 1;
    healer_state.npcs[2].attack_style_this_tick = ATTACK_STYLE_MAGIC;
    healer_state.npcs[2].attack_visual_target = 0;

    EncounterOverlay healer_ov;
    memset(&healer_ov, 0, sizeof(healer_ov));
    inf_render_post_tick_ctx((EncounterState*)&healer_state, (EncounterContext*)&test_context, &healer_ov);

    ASSERT_INT_EQ("healer projectile count", healer_ov.projectile_count, 1);
    ASSERT_INT_EQ("healer projectile model",
        healer_ov.projectiles[0].model_id, INF_GFX_660_MODEL);
    ASSERT_INT_EQ("healer projectile animation",
        healer_ov.projectiles[0].anim_id, INF_GFX_660_ANIM);
    ASSERT_INT_EQ("healer impact spotanim", healer_ov.projectiles[0].impact_gfx_id, 659);
    ASSERT_INT_EQ("healer projectile source kind",
        healer_ov.projectiles[0].source_kind, ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT);
    ASSERT_INT_EQ("healer projectile source slot", healer_ov.projectiles[0].source_npc_slot, 2);
}

static void test_player_projectile_render_uses_stored_reference_timing(void) {
    printf("--- player projectile render uses stored reference timing ---\n");

    InfernoState blowpipe_state = make_test_state(10, 10);
    blowpipe_state.player.equipped[GEAR_SLOT_WEAPON] = ITEM_TOXIC_BLOWPIPE;
    blowpipe_state.npcs[0] = make_test_npc(
        INF_NPC_JAD, 18, 10, INF_NPC_STATS[INF_NPC_JAD].size);
    blowpipe_state.npcs[0].active = 1;
    blowpipe_state.tick_scratch.player_attacked = 1;
    blowpipe_state.player_attack_npc_idx = 0;
    blowpipe_state.player_attack_style_id = ATTACK_STYLE_RANGED;
    blowpipe_state.player_attack_dmg = 7;
    blowpipe_state.player.used_special_this_tick = 1;

    int blowpipe_dist = encounter_projectile_distance(
        blowpipe_state.player.x, blowpipe_state.player.y, 1,
        blowpipe_state.npcs[0].x, blowpipe_state.npcs[0].y, blowpipe_state.npcs[0].size,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    blowpipe_state.player_attack_timing = inf_player_projectile_timing(
        ATTACK_STYLE_RANGED, ITEM_TOXIC_BLOWPIPE, 1, blowpipe_dist);

    EncounterOverlay blowpipe_ov;
    memset(&blowpipe_ov, 0, sizeof(blowpipe_ov));
    inf_render_post_tick_ctx((EncounterState*)&blowpipe_state, (EncounterContext*)&test_context, &blowpipe_ov);

    ASSERT_INT_EQ("blowpipe spec projectile count", blowpipe_ov.projectile_count, 1);
    ASSERT_INT_EQ("blowpipe projectile tracks target", blowpipe_ov.projectiles[0].tracks_target, 1);
    ASSERT_INT_EQ("blowpipe projectile target kind",
        blowpipe_ov.projectiles[0].target_kind, ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT);
    ASSERT_INT_EQ("blowpipe projectile target slot", blowpipe_ov.projectiles[0].target_npc_slot, 0);
    ASSERT_INT_EQ("blowpipe projectile model",
        blowpipe_ov.projectiles[0].model_id, OSRS_PROJECTILE_MODEL_DRAGON_DART);
    ASSERT_INT_EQ("blowpipe projectile animation",
        blowpipe_ov.projectiles[0].anim_id, OSRS_PROJECTILE_ANIM_DRAGON_DART);
    ASSERT_INT_EQ("blowpipe launch spotanim",
        blowpipe_ov.projectiles[0].launch_gfx_id, GFX_BLOWPIPE_SPEC);
    ASSERT_INT_EQ("blowpipe impact spotanim",
        blowpipe_ov.projectiles[0].impact_gfx_id, 0);
    ASSERT_INT_EQ("blowpipe projectile start height",
        blowpipe_ov.projectiles[0].start_h, 163);
    ASSERT_INT_EQ("blowpipe projectile end height",
        blowpipe_ov.projectiles[0].end_h, 146);
    ASSERT_INT_EQ("blowpipe spec visual start delay",
        blowpipe_ov.projectiles[0].start_delay,
        blowpipe_state.player_attack_timing.visual_start_delay_ticks * 30);
    ASSERT_INT_EQ("blowpipe spec visual duration",
        blowpipe_ov.projectiles[0].duration_ticks,
        blowpipe_state.player_attack_timing.visual_duration_ticks * 30);

    InfernoState tbow_state = make_test_state(10, 10);
    tbow_state.player.equipped[GEAR_SLOT_WEAPON] = ITEM_TWISTED_BOW;
    tbow_state.npcs[0] = make_test_npc(
        INF_NPC_JAD, 18, 10, INF_NPC_STATS[INF_NPC_JAD].size);
    tbow_state.npcs[0].active = 1;
    tbow_state.tick_scratch.player_attacked = 1;
    tbow_state.player_attack_npc_idx = 0;
    tbow_state.player_attack_style_id = ATTACK_STYLE_RANGED;
    tbow_state.player_attack_dmg = 7;

    int tbow_dist = encounter_projectile_distance(
        tbow_state.player.x, tbow_state.player.y, 1,
        tbow_state.npcs[0].x, tbow_state.npcs[0].y, tbow_state.npcs[0].size,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    tbow_state.player_attack_timing = inf_player_projectile_timing(
        ATTACK_STYLE_RANGED, ITEM_TWISTED_BOW, 0, tbow_dist);

    EncounterOverlay tbow_ov;
    memset(&tbow_ov, 0, sizeof(tbow_ov));
    inf_render_post_tick_ctx((EncounterState*)&tbow_state, (EncounterContext*)&test_context, &tbow_ov);

    ASSERT_INT_EQ("tbow projectile count", tbow_ov.projectile_count, 1);
    ASSERT_INT_EQ("tbow projectile tracks target", tbow_ov.projectiles[0].tracks_target, 1);
    ASSERT_INT_EQ("tbow projectile target kind",
        tbow_ov.projectiles[0].target_kind, ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT);
    ASSERT_INT_EQ("tbow projectile target slot", tbow_ov.projectiles[0].target_npc_slot, 0);
    ASSERT_INT_EQ("tbow projectile model",
        tbow_ov.projectiles[0].model_id, OSRS_PROJECTILE_MODEL_DRAGON_ARROW);
    ASSERT_INT_EQ("tbow projectile animation",
        tbow_ov.projectiles[0].anim_id, OSRS_PROJECTILE_ANIM_DRAGON_ARROW);
    ASSERT_INT_EQ("tbow launch spotanim",
        tbow_ov.projectiles[0].launch_gfx_id, GFX_DRAGON_ARROW_LAUNCH);
    ASSERT_INT_EQ("tbow impact spotanim",
        tbow_ov.projectiles[0].impact_gfx_id, 0);
    ASSERT_INT_EQ("tbow visual start delay",
        tbow_ov.projectiles[0].start_delay,
        tbow_state.player_attack_timing.visual_start_delay_ticks * 30);
    ASSERT_INT_EQ("tbow visual duration",
        tbow_ov.projectiles[0].duration_ticks,
        tbow_state.player_attack_timing.visual_duration_ticks * 30);

    InfernoState bowfa_state = make_test_state(10, 10);
    bowfa_state.player.equipped[GEAR_SLOT_WEAPON] = ITEM_BOW_OF_FAERDHINEN;
    bowfa_state.npcs[0] = make_test_npc(
        INF_NPC_JAD, 18, 10, INF_NPC_STATS[INF_NPC_JAD].size);
    bowfa_state.npcs[0].active = 1;
    bowfa_state.tick_scratch.player_attacked = 1;
    bowfa_state.player_attack_npc_idx = 0;
    bowfa_state.player_attack_style_id = ATTACK_STYLE_RANGED;
    bowfa_state.player_attack_dmg = 7;

    int bowfa_dist = encounter_projectile_distance(
        bowfa_state.player.x, bowfa_state.player.y, 1,
        bowfa_state.npcs[0].x, bowfa_state.npcs[0].y, bowfa_state.npcs[0].size,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    bowfa_state.player_attack_timing = inf_player_projectile_timing(
        ATTACK_STYLE_RANGED, ITEM_BOW_OF_FAERDHINEN, 0, bowfa_dist);

    EncounterOverlay bowfa_ov;
    memset(&bowfa_ov, 0, sizeof(bowfa_ov));
    inf_render_post_tick_ctx((EncounterState*)&bowfa_state, (EncounterContext*)&test_context, &bowfa_ov);

    ASSERT_INT_EQ("bowfa projectile count", bowfa_ov.projectile_count, 1);
    ASSERT_INT_EQ("bowfa projectile model",
        bowfa_ov.projectiles[0].model_id, OSRS_PROJECTILE_MODEL_ARROW);
    ASSERT_INT_EQ("bowfa launch spotanim",
        bowfa_ov.projectiles[0].launch_gfx_id, GFX_RUNE_ARROW_LAUNCH);
    ASSERT_INT_EQ("bowfa impact spotanim",
        bowfa_ov.projectiles[0].impact_gfx_id, 0);
}

static void test_projectile_anchor_effect_subtile_round_trips_entity_center(void) {
    printf("--- projectile anchor effect subtile round-trips entity center ---\n");

    int player_sub_x = 10 * 128 + 64;
    int player_sub_y = 17 * 128 + 64;
    float anchor_x = osrs_projectile_anchor_coord_from_subtile(player_sub_x);
    float anchor_y = osrs_projectile_anchor_coord_from_subtile(player_sub_y);

    ASSERT_FLOAT_NEAR("projectile anchor x is tile origin",
        anchor_x, 10.0f, 0.0001f);
    ASSERT_FLOAT_NEAR("projectile anchor y is tile origin",
        anchor_y, 17.0f, 0.0001f);
    ASSERT_FLOAT_NEAR("projectile effect x returns to entity center",
        osrs_projectile_subtile_from_anchor_coord(anchor_x),
        (float)player_sub_x, 0.0001f);
    ASSERT_FLOAT_NEAR("projectile effect y returns to entity center",
        osrs_projectile_subtile_from_anchor_coord(anchor_y),
        (float)player_sub_y, 0.0001f);
}

static void test_magic_splash_landing_keeps_spell_visual_context(void) {
    printf("--- magic splash landing keeps spell visual context ---\n");

    ASSERT_INT_EQ("ice barrage visual id matches encounter spell id",
        OSRS_COMBAT_VISUAL_SPELL_ICE_BARRAGE, ENCOUNTER_SPELL_ICE);
    ASSERT_INT_EQ("blood barrage visual id matches encounter spell id",
        OSRS_COMBAT_VISUAL_SPELL_BLOOD_BARRAGE, ENCOUNTER_SPELL_BLOOD);

    InfernoState state = make_test_state(10, 10);
    state.npcs[0] = make_test_npc(
        INF_NPC_RANGER, 16, 10, INF_NPC_STATS[INF_NPC_RANGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_RANGER].hp;
    state.npcs[0].pending_hits.hits[0].active = 1;
    state.npcs[0].pending_hits.hits[0].ticks_remaining = 1;
    state.npcs[0].pending_hits.hits[0].damage = 0;
    state.npcs[0].pending_hits.hits[0].attack_style = ATTACK_STYLE_MAGIC;
    state.npcs[0].pending_hits.hits[0].spell_type = ENCOUNTER_SPELL_BLOOD;
    state.npcs[0].pending_hits.hits[0].hit_success = 0;
    state.npcs[0].pending_hits.count = 1;

    inf_resolve_player_projectiles_on_npcs(&state);

    RenderEntity entities[4];
    int count = 0;
    inf_fill_render_entities_ctx((EncounterState*)&state, (EncounterContext*)&test_context, entities, 4, &count);

    ASSERT_INT_EQ("splashing ranger still emits landed visual event",
        entities[1].hit_landed_this_tick, 1);
    ASSERT_INT_EQ("splashing ranger records failed accuracy",
        entities[1].hit_was_successful, 0);
    ASSERT_INT_EQ("splashing ranger records blood barrage spell",
        entities[1].hit_spell_type, ENCOUNTER_SPELL_BLOOD);
}

static void test_npc_overkill_hit_caps_splat_hp_and_damage_stats(void) {
    printf("--- npc overkill hit caps splat hp and damage stats ---\n");

    InfernoState state = make_test_state(10, 10);
    state.npcs[0] = make_test_npc(
        INF_NPC_RANGER, 16, 10, INF_NPC_STATS[INF_NPC_RANGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 15;
    state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_RANGER].hp;
    state.npcs[0].pending_hits.hits[0].active = 1;
    state.npcs[0].pending_hits.hits[0].ticks_remaining = 1;
    state.npcs[0].pending_hits.hits[0].damage = 50;
    state.npcs[0].pending_hits.hits[0].attack_style = ATTACK_STYLE_RANGED;
    state.npcs[0].pending_hits.hits[0].hit_success = 1;
    state.npcs[0].pending_hits.count = 1;

    inf_resolve_player_projectiles_on_npcs(&state);

    RenderEntity entities[4];
    int count = 0;
    inf_fill_render_entities_ctx((EncounterState*)&state, (EncounterContext*)&test_context, entities, 4, &count);

    ASSERT_INT_EQ("ranger hp clamps at zero", state.npcs[0].hp, 0);
    ASSERT_INT_EQ("ranger hit splat caps to remaining hp",
        state.npcs[0].hit_damage, 15);
    ASSERT_INT_EQ("render entity hit splat caps to remaining hp",
        entities[1].hit_damage, 15);
    ASSERT_FLOAT_NEAR("landing does not double count XP-drop damage",
        state.tick_scratch.damage_dealt, 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("landing does not double count set damage",
        state.tick_scratch.damage_set, 0.0f, 1e-6f);
    ASSERT_INT_EQ("overkill still counts the kill", state.tick_scratch.kill_set, 1);
}

static void test_blood_barrage_overkill_heals_from_capped_damage(void) {
    printf("--- blood barrage overkill heals from capped damage ---\n");

    InfernoState state = make_test_state(10, 10);
    state.player.base_hitpoints = 99;
    state.player.current_hitpoints = 80;
    state.npcs[0] = make_test_npc(
        INF_NPC_RANGER, 16, 10, INF_NPC_STATS[INF_NPC_RANGER].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = 8;
    state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_RANGER].hp;
    state.npcs[0].pending_hits.hits[0].active = 1;
    state.npcs[0].pending_hits.hits[0].ticks_remaining = 1;
    state.npcs[0].pending_hits.hits[0].damage = 40;
    state.npcs[0].pending_hits.hits[0].attack_style = ATTACK_STYLE_MAGIC;
    state.npcs[0].pending_hits.hits[0].spell_type = ENCOUNTER_SPELL_BLOOD;
    state.npcs[0].pending_hits.hits[0].hit_success = 1;
    state.npcs[0].pending_hits.count = 1;

    inf_resolve_player_projectiles_on_npcs(&state);

    ASSERT_INT_EQ("blood barrage hit splat caps to remaining hp",
        state.npcs[0].hit_damage, 8);
    ASSERT_FLOAT_NEAR("blood barrage landing does not double count damage stat",
        state.tick_scratch.damage_dealt, 0.0f, 1e-6f);
    ASSERT_INT_EQ("blood barrage heal uses capped damage",
        state.tick_scratch.blood_heal, 2);
    ASSERT_INT_EQ("player receives capped blood heal",
        state.player.current_hitpoints, 82);
}

static void test_elysian_proc_propagates_to_player_render_entity(void) {
    printf("--- elysian proc propagates to player render entity ---\n");

    InfernoState state = make_test_state(10, 10);
    state.player.entity_type = ENTITY_PLAYER;
    state.player.base_hitpoints = 99;
    state.player.current_hitpoints = 99;
    state.player.equipped[GEAR_SLOT_SHIELD] = ITEM_ELYSIAN_SPIRIT_SHIELD;
    state.player_pending_hits.count = 1;
    state.player_pending_hits.hits[0].active = 1;
    state.player_pending_hits.hits[0].ticks_remaining = 1;
    state.player_pending_hits.hits[0].damage = 12;
    state.player_pending_hits.hits[0].attack_style = ATTACK_STYLE_RANGED;
    state.player_pending_hits.hits[0].source_npc_type = INF_NPC_RANGER;
    state.player_pending_hits.hits[0].elysian_reduced = 1;

    inf_resolve_player_pending_hits(&state);

    RenderEntity entities[2];
    int count = 0;
    inf_fill_render_entities_ctx((EncounterState*)&state, (EncounterContext*)&test_context, entities, 2, &count);

    ASSERT_INT_EQ("player render entity exists", count >= 1, 1);
    ASSERT_INT_EQ("elysian proc reaches player state",
        state.player.elysian_proc_this_tick, 1);
    ASSERT_INT_EQ("elysian proc reaches render entity",
        entities[0].elysian_proc_this_tick, 1);
}

static void test_delayed_player_hit_records_landing_source(void) {
    printf("--- delayed player hit records landing source ---\n");

    InfernoState state = make_test_state(10, 10);
    state.player.base_hitpoints = 99;
    state.player.current_hitpoints = 10;
    state.last_hit_by_type = INF_NPC_MAGER;
    state.player_pending_hits.count = 1;
    state.player_pending_hits.hits[0] = (EncounterPendingHit){
        .active = 1,
        .damage = 20,
        .ticks_remaining = 1,
        .attack_style = ATTACK_STYLE_NONE,
        .check_prayer = 0,
        .source_npc_type = INF_NPC_ZUK,
    };

    inf_resolve_player_pending_hits(&state);

    ASSERT_INT_EQ("zuk hit killed player", state.player.current_hitpoints, 0);
    ASSERT_INT_EQ("last hit source updated on landing",
        state.last_hit_by_type, INF_NPC_ZUK);
}

static void test_terminal_reward_uses_fixed_win_reward(void) {
    printf("--- terminal reward uses fixed win reward ---\n");

    InfernoState state = make_test_state(10, 10);
    state.episode_over = 1;
    state.winner = INF_OUTCOME_PLAYER_WON;
    ASSERT_FLOAT_NEAR("terminal win reward is fixed",
        inf_compute_reward_ctx(&state, &test_context), 1.0f, 1e-6f);

    state.winner = INF_OUTCOME_PLAYER_DIED;
    test_config()->death_penalty_coeff = 0.25f;
    ASSERT_FLOAT_NEAR("terminal loss uses configured death penalty",
        inf_compute_reward_ctx(&state, &test_context), -0.25f, 1e-6f);
}

static void test_final_wave_completion_emits_terminal_reward(void) {
    printf("--- final wave completion emits terminal reward ---\n");

    InfernoState state = make_test_state(10, 10);
    state.player.base_hitpoints = 99;
    state.player.current_hitpoints = 99;
    state.wave = INF_NUM_WAVES - 1;

    step_inferno_noop(&state);

    ASSERT_INT_EQ("final wave completion ends episode", state.episode_over, 1);
    ASSERT_INT_EQ("final wave completion marks win", state.winner, INF_OUTCOME_PLAYER_WON);
    ASSERT_INT_EQ("final wave completion marks wave clear",
        state.tick_scratch.wave_completed, 1);
    ASSERT_FLOAT_NEAR("final wave completion emits clipped terminal reward",
        state.reward, 1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("episode return includes terminal reward",
        state.episode_return, 1.0f, 1e-6f);
}

static void test_lethal_pending_hit_banks_damage_stats_before_terminal(void) {
    printf("--- lethal pending hit banks damage stats before terminal ---\n");

    InfernoState state = make_test_state(10, 10);
    state.player.base_hitpoints = 99;
    state.player.current_hitpoints = 10;
    state.player_pending_hits.count = 1;
    state.player_pending_hits.hits[0] = (EncounterPendingHit){
        .active = 1,
        .damage = 20,
        .ticks_remaining = 1,
        .attack_style = ATTACK_STYLE_MAGIC,
        .check_prayer = 0,
        .source_npc_type = INF_NPC_ZUK,
    };

    step_inferno_noop(&state);

    ASSERT_INT_EQ("lethal hit ends episode", state.episode_over, 1);
    ASSERT_INT_EQ("lethal hit marks loss", state.winner, INF_OUTCOME_PLAYER_DIED);
    ASSERT_INT_EQ("lethal source counted", state.killed_by_type[INF_NPC_ZUK], 1);
    ASSERT_FLOAT_NEAR("lethal damage is banked",
        state.total_damage_received, 20.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("lethal tick emits zero reward", state.reward, 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("episode return matches emitted terminal reward",
        state.episode_return, 0.0f, 1e-6f);
}

static void test_terminal_penalty_applies_to_death_when_enabled(void) {
    printf("--- terminal penalty applies to death when enabled ---\n");

    InfernoState state = make_test_state(10, 10);
    test_config()->terminal_penalty_enabled = 1;
    state.player.base_hitpoints = 99;
    state.player.current_hitpoints = 10;
    state.player_pending_hits.count = 1;
    state.player_pending_hits.hits[0] = (EncounterPendingHit){
        .active = 1,
        .damage = 20,
        .ticks_remaining = 1,
        .attack_style = ATTACK_STYLE_MAGIC,
        .check_prayer = 0,
        .source_npc_type = INF_NPC_ZUK,
    };

    step_inferno_noop(&state);

    ASSERT_INT_EQ("lethal hit ends episode", state.episode_over, 1);
    ASSERT_INT_EQ("lethal hit marks loss", state.winner, INF_OUTCOME_PLAYER_DIED);
    ASSERT_FLOAT_NEAR("lethal tick emits terminal penalty",
        state.reward, -1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("episode return includes terminal penalty",
        state.episode_return, -1.0f, 1e-6f);
}

static void test_timeout_reward_matches_episode_return(void) {
    printf("--- timeout reward matches episode return ---\n");

    InfernoState state = make_test_state(10, 10);
    state.player.base_hitpoints = 99;
    state.player.current_hitpoints = 99;
    state.tick = INF_MAX_TICKS - 1;

    step_inferno_noop(&state);

    ASSERT_INT_EQ("timeout ends episode", state.episode_over, 1);
    ASSERT_INT_EQ("timeout marks loss", state.winner, INF_OUTCOME_PLAYER_DIED);
    ASSERT_FLOAT_NEAR("timeout emits zero reward", state.reward, 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("episode return matches timeout reward",
        state.episode_return, 0.0f, 1e-6f);
}

static void test_terminal_penalty_applies_to_timeout_when_enabled(void) {
    printf("--- terminal penalty applies to timeout when enabled ---\n");

    InfernoState state = make_test_state(10, 10);
    test_config()->terminal_penalty_enabled = 1;
    state.player.base_hitpoints = 99;
    state.player.current_hitpoints = 99;
    state.tick = INF_MAX_TICKS - 1;

    step_inferno_noop(&state);

    ASSERT_INT_EQ("timeout ends episode", state.episode_over, 1);
    ASSERT_INT_EQ("timeout marks loss", state.winner, INF_OUTCOME_PLAYER_DIED);
    ASSERT_FLOAT_NEAR("timeout emits terminal penalty", state.reward, -1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("episode return includes terminal penalty",
        state.episode_return, -1.0f, 1e-6f);
}

static void test_inferno_render_overlay_reports_death_source(void) {
    printf("--- inferno render overlay reports death source ---\n");

    InfernoState state = make_test_state(10, 10);
    state.episode_over = 1;
    state.winner = INF_OUTCOME_PLAYER_DIED;
    state.last_hit_by_type = INF_NPC_ZUK;

    EncounterOverlay ov;
    memset(&ov, 0, sizeof(ov));
    inf_render_post_tick_ctx((EncounterState*)&state, (EncounterContext*)&test_context, &ov);

    ASSERT_INT_EQ("death banner active", ov.status_text_active, 1);
    ASSERT_STR_EQ("death banner text", ov.status_text, "Killed by TzKal-Zuk");
}


static void test_inferno_reset_uses_osrs_run_energy_units(void) {
    printf("--- inferno reset uses osrs run energy units ---\n");

    InfernoContext ctx;
    InfernoState state;
    inf_init_context_typed(&ctx);
    inf_init_state_typed(&state, &ctx);
    inf_reset_ctx((EncounterState*)&state, (EncounterContext*)&ctx, 20260519u);

    ASSERT_INT_EQ("inferno starts at full run energy",
        state.player.run_energy, OSRS_RUN_ENERGY_FULL);
    ASSERT_INT_EQ("inferno run energy renders as percent",
        osrs_run_energy_percent(state.player.run_energy), 100);
}

static void activate_dense_target_test_npc(
    InfernoState* state,
    int npc_idx,
    InfNPCType type
) {
    state->npcs[npc_idx] = make_test_npc(
        type, 10 + npc_idx, 20, INF_NPC_STATS[type].size);
    state->npcs[npc_idx].active = 1;
    state->npcs[npc_idx].hp = state->npcs[npc_idx].max_hp =
        INF_NPC_STATS[type].hp;
}

static void test_dense_target_contract_dimensions(void) {
    printf("--- dense target contract dimensions ---\n");

    ASSERT_INT_EQ("primary head includes movement and every target",
        INF_ACTION_DIMS[INF_HEAD_PRIMARY], OSRS_PRIMARY_DIM(INF_OBS_NPCS));
    ASSERT_INT_EQ("shared primary action mask width",
        INF_ACTION_MASK_SIZE, 436);
}

static void test_dense_target_slots_follow_type_priority_without_holes(void) {
    printf("--- dense target slots follow type priority without holes ---\n");

    InfernoState state = make_test_state(20, 20);
    activate_dense_target_test_npc(&state, 0, INF_NPC_HEALER_ZUK);
    activate_dense_target_test_npc(&state, 1, INF_NPC_BAT);
    activate_dense_target_test_npc(&state, 2, INF_NPC_MAGER);
    activate_dense_target_test_npc(&state, 3, INF_NPC_BLOB_MELEE);
    activate_dense_target_test_npc(&state, 4, INF_NPC_RANGER);
    activate_dense_target_test_npc(&state, 5, INF_NPC_NIBBLER);
    activate_dense_target_test_npc(&state, 6, INF_NPC_MELEER);
    activate_dense_target_test_npc(&state, 7, INF_NPC_BLOB);
    activate_dense_target_test_npc(&state, 8, INF_NPC_BLOB_MAGE);
    activate_dense_target_test_npc(&state, 9, INF_NPC_BLOB_RANGE);
    activate_dense_target_test_npc(&state, 10, INF_NPC_JAD);
    activate_dense_target_test_npc(&state, 11, INF_NPC_ZUK);
    activate_dense_target_test_npc(&state, 12, INF_NPC_ZUK_SHIELD);
    activate_dense_target_test_npc(&state, 13, INF_NPC_HEALER_JAD);

    static const int expected_npc_indices[14] = {
        2, 4, 6, 7, 1, 8, 9, 3, 5, 10, 11, 12, 13, 0,
    };

    inf_refresh_current_obs_slots_ctx(&state, &test_context);

    for (int slot = 0; slot < 14; slot++) {
        ASSERT_INT_EQ("dense slot follows type priority",
            state.current_obs_slots[slot], expected_npc_indices[slot]);
    }
    for (int slot = 14; slot < INF_OBS_NPCS; slot++) {
        ASSERT_INT_EQ("unused dense target slot is empty",
            state.current_obs_slots[slot], -1);
    }
}

static void test_regular_waves_select_every_live_npc(void) {
    printf("--- regular waves select every live NPC ---\n");

    for (int wave = 0; wave < INF_WAVE_ZUK; wave++) {
        InfernoState state = make_test_state(20, 20);
        state.wave = wave;
        inf_spawn_wave(&state);
        inf_refresh_current_obs_slots_ctx(&state, &test_context);

        for (int npc_idx = 0; npc_idx < INF_MAX_NPCS; npc_idx++) {
            if (!state.npcs[npc_idx].active ||
                    state.npcs[npc_idx].death_ticks != 0 ||
                    state.npcs[npc_idx].hp <= 0)
                continue;
            ASSERT_INT_EQ("regular-wave live NPC is selected",
                inf_find_target_obs_slot(&state, npc_idx) >= 0, 1);
        }
    }
}

static void test_maximal_zuk_concurrency_selects_all_live_candidates(void) {
    printf("--- maximal Zuk concurrency selects all live candidates ---\n");

    static const InfNPCType types[14] = {
        INF_NPC_ZUK,
        INF_NPC_ZUK_SHIELD,
        INF_NPC_JAD,
        INF_NPC_HEALER_JAD,
        INF_NPC_HEALER_JAD,
        INF_NPC_HEALER_JAD,
        INF_NPC_MAGER,
        INF_NPC_MAGER,
        INF_NPC_RANGER,
        INF_NPC_RANGER,
        INF_NPC_HEALER_ZUK,
        INF_NPC_HEALER_ZUK,
        INF_NPC_HEALER_ZUK,
        INF_NPC_HEALER_ZUK,
    };

    InfernoState state = make_test_state(20, 20);
    for (int npc_idx = 0; npc_idx < 14; npc_idx++)
        activate_dense_target_test_npc(&state, npc_idx, types[npc_idx]);

    inf_refresh_current_obs_slots_ctx(&state, &test_context);

    int selected = 0;
    for (int slot = 0; slot < INF_OBS_NPCS; slot++)
        selected += state.current_obs_slots[slot] >= 0;
    ASSERT_INT_EQ("maximal Zuk concurrency fills all dense slots", selected, 14);

    for (int npc_idx = 0; npc_idx < 14; npc_idx++) {
        ASSERT_INT_EQ("maximal Zuk live candidate is selected",
            inf_find_target_obs_slot(&state, npc_idx) >= 0, 1);
    }
}

static void refresh_fifteen_eligible_target_candidates(void) {
    static const InfNPCType types[15] = {
        INF_NPC_MAGER, INF_NPC_MAGER,
        INF_NPC_RANGER, INF_NPC_RANGER,
        INF_NPC_MELEER, INF_NPC_MELEER,
        INF_NPC_BLOB, INF_NPC_BLOB,
        INF_NPC_BAT, INF_NPC_BAT,
        INF_NPC_BLOB_MAGE, INF_NPC_BLOB_MAGE,
        INF_NPC_BLOB_RANGE, INF_NPC_BLOB_RANGE,
        INF_NPC_BLOB_MELEE,
    };

    InfernoState state = make_test_state(20, 20);
    for (int npc_idx = 0; npc_idx < 15; npc_idx++)
        activate_dense_target_test_npc(&state, npc_idx, types[npc_idx]);
    inf_refresh_current_obs_slots_ctx(&state, &test_context);
}

static void test_dense_target_overflow_aborts_instead_of_truncating(void) {
    printf("--- dense target overflow aborts instead of truncating ---\n");

    assert_child_aborts("fifteenth eligible target aborts refresh",
        refresh_fifteen_eligible_target_candidates);
}

static void test_compact_observation_layout_contract(void) {
    printf("--- compact observation layout contract ---\n");

    ASSERT_INT_EQ("shared prefix width", INF_OBS_AFTER_SHARED, 101);
    ASSERT_INT_EQ("inferno encounter width", INF_ENCOUNTER_OBS_SIZE, 14);
    ASSERT_INT_EQ("compact pillar width", INF_PILLAR_OBS_SIZE, 9);
    ASSERT_INT_EQ("compact NPC stride", INF_NPC_SLOT_FEATURES, 13);
    ASSERT_INT_EQ("compact pending hit stride", INF_FEATURES_PER_HIT, 3);
    ASSERT_INT_EQ("compact spark stride", INF_FEATURES_PER_SPARK, 4);
    ASSERT_INT_EQ("inventory carries one canonical code per cell",
        OSRS_SHARED_INVENTORY_OBS_SIZE, 28);
    ASSERT_INT_EQ("equipment carries one canonical code per worn slot",
        OSRS_SHARED_EQUIPPED_OBS_SIZE, NUM_GEAR_SLOTS);

    ASSERT_INT_EQ("shared prefix end", INF_OBS_AFTER_SHARED, 101);
    ASSERT_INT_EQ("inferno encounter end", INF_OBS_AFTER_ENCOUNTER, 115);
    ASSERT_INT_EQ("compact pillar end", INF_OBS_AFTER_PILLARS, 124);
    ASSERT_INT_EQ("compact NPC end", INF_OBS_AFTER_NPCS, 306);
    ASSERT_INT_EQ("compact pending hit end", INF_OBS_AFTER_PENDING_HITS, 402);
    ASSERT_INT_EQ("compact spark end", INF_OBS_AFTER_SPARKS, 530);
    ASSERT_INT_EQ("inferno observation width", INF_NUM_OBS, 530);
}

static void test_compact_player_and_pillar_observation_semantics(void) {
    printf("--- compact player and pillar observation semantics ---\n");

    InfernoState state = make_test_state(20, 20);
    state.wave = INF_WAVE_ZUK;
    state.weapon_set = INF_GEAR_BP;
    state.player.current_hitpoints = 99;
    state.player.base_hitpoints = 99;
    state.player.base_prayer = 99;
    state.player.current_prayer = 99;
    state.player.current_defence = 99;
    state.player.current_ranged = 99;
    state.player.current_magic = 99;
    state.player.prayer = PRAYER_PROTECT_MAGIC;
    state.player.offensive_prayer = OFFENSIVE_PRAYER_AUGURY;
    state.zuk.enraged = 1;
    state.pillars[0] = (InfPillar){
        .x = 24,
        .y = 26,
        .hp = INF_PILLAR_HP / 2,
        .active = 1,
    };

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    ASSERT_FLOAT_NEAR("shared overhead prayer is one-hot",
        obs[8], 1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("shared offensive prayer is one-hot",
        obs[13], 1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("wave phase uses compact code",
        obs[INF_OBS_WAVE_PHASE],
        (float)(inf_wave_phase_index(state.wave) + 1) / 8.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("Zuk enraged state follows the shared prefix",
        obs[INF_OBS_ZUK_PHASE_START + 4], 1.0f, 1e-6f);

    ASSERT_FLOAT_NEAR("compact pillar hp",
        obs[INF_OBS_AFTER_ENCOUNTER],
        (float)state.pillars[0].hp / (float)INF_PILLAR_HP, 1e-6f);
    ASSERT_FLOAT_NEAR("compact pillar relative x",
        obs[INF_OBS_AFTER_ENCOUNTER + 1],
        4.0f / (float)INF_ARENA_WIDTH, 1e-6f);
    ASSERT_FLOAT_NEAR("compact pillar relative y",
        obs[INF_OBS_AFTER_ENCOUNTER + 2],
        6.0f / (float)INF_ARENA_HEIGHT, 1e-6f);
}

static void test_compact_npc_observation_semantics(void) {
    printf("--- compact NPC observation semantics ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player.current_hitpoints = 99;
    state.player.base_hitpoints = 99;
    state.player.base_prayer = 99;
    state.player.current_prayer = 99;
    state.player.current_defence = 99;
    state.player.current_ranged = 99;
    state.player.current_magic = 99;

    state.npcs[0] = make_test_npc(
        INF_NPC_BLOB, 24, 20, INF_NPC_STATS[INF_NPC_BLOB].size);
    state.npcs[0].active = 1;
    state.npcs[0].hp = INF_NPC_STATS[INF_NPC_BLOB].hp / 2;
    state.npcs[0].max_hp = INF_NPC_STATS[INF_NPC_BLOB].hp;
    state.npcs[0].attack_timer = 5;
    state.npcs[0].attack_style = ATTACK_STYLE_MAGIC;
    state.npcs[0].blob_scanned_prayer = PRAYER_PROTECT_RANGED;
    state.npcs[0].frozen_ticks = BARRAGE_FREEZE_TICKS / 2;
    osrs_interaction_set(&state.interaction, 0);

    state.npcs[1] = make_test_npc(
        INF_NPC_MELEER, 22, 20, INF_NPC_STATS[INF_NPC_MELEER].size);
    state.npcs[1].active = 1;
    state.npcs[1].hp = state.npcs[1].max_hp = INF_NPC_STATS[INF_NPC_MELEER].hp;
    state.npcs[1].attack_timer = 3;
    state.npcs[1].no_los_ticks = 25;
    state.npcs[1].dig_freeze_timer = 3;

    float obs[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);

    int blob_slot = inf_find_target_obs_slot(&state, 0);
    int blob_start = INF_OBS_AFTER_PILLARS + blob_slot * INF_NPC_SLOT_FEATURES;
    InfNpcPlayerThreat blob_threat = inf_npc_player_threat_ctx(&state, &test_context, &state.npcs[0]);
    ASSERT_FLOAT_NEAR("compact NPC type code",
        obs[blob_start], (float)(INF_NPC_BLOB + 1) / 16.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("compact NPC hp", obs[blob_start + 1], 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR("compact NPC relative x",
        obs[blob_start + 2], 4.0f / (float)INF_ARENA_WIDTH, 1e-6f);
    ASSERT_FLOAT_NEAR("compact NPC relative y", obs[blob_start + 3], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("compact NPC attack timer",
        obs[blob_start + 4], 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR("compact NPC attack style",
        obs[blob_start + 5], (float)ATTACK_STYLE_MAGIC / 4.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("compact NPC LOS",
        obs[blob_start + 6],
        (float)inf_npc_has_los_ctx(&state, &test_context, 0),
        1e-6f);
    ASSERT_FLOAT_NEAR("compact NPC frozen timer",
        obs[blob_start + 7], 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR("compact NPC target category",
        obs[blob_start + 8], (float)INF_TARGET_CATEGORY_PLAYER / 8.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("compact NPC targeted bit",
        obs[blob_start + 9], 1.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("Blob type state 0 is zero",
        obs[blob_start + 10], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("Blob type state 1 is zero",
        obs[blob_start + 11], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("Blob type state 2 is zero",
        obs[blob_start + 12], 0.0f, 1e-6f);

    int meleer_slot = inf_find_target_obs_slot(&state, 1);
    int meleer_start = INF_OBS_AFTER_PILLARS + meleer_slot * INF_NPC_SLOT_FEATURES;
    ASSERT_FLOAT_NEAR("meleer compact no-LOS progress",
        obs[meleer_start + 10], 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR("meleer compact dig-freeze state",
        obs[meleer_start + 11], 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR("meleer compact dig-delay state is clear",
        obs[meleer_start + 12], 0.0f, 1e-6f);

    state.npcs[1].dig_freeze_timer = 0;
    state.npcs[1].dig_attack_delay = 3;
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs);
    ASSERT_FLOAT_NEAR("meleer compact dig-freeze state clears",
        obs[meleer_start + 11], 0.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("meleer compact dig-delay state",
        obs[meleer_start + 12], 0.5f, 1e-6f);
}

static void test_compact_transient_inventory_equipment_semantics(void) {
    printf("--- compact transient inventory equipment semantics ---\n");

    InfernoState state = make_test_state(20, 20);
    state.player.current_hitpoints = 99;
    state.player.base_hitpoints = 99;
    state.player.base_prayer = 99;
    state.player.current_prayer = 99;
    state.player.current_defence = 99;
    state.player.current_ranged = 99;
    state.player.current_magic = 99;
    state.player_pending_hits.count = 1;
    state.player_pending_hits.hits[0] = (EncounterPendingHit){
        .active = 1,
        .attack_style = ATTACK_STYLE_MAGIC,
        .ticks_remaining = 4,
        .damage = 75,
    };
    state.pending_sparks[0] = (InfPendingSpark){
        .active = 1,
        .src_x = 11,
        .src_y = 12,
        .x = 24,
        .y = 26,
        .ticks_remaining = 5,
        .damage = 8,
    };
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++)
        state.player.inventory_cells[cell] = osrs_inventory_cell_empty();
    state.player.inventory_cells[0] =
        osrs_inventory_cell_from_item(ITEM_OSMUMTENS_FANG);
    state.player.inventory_cells[1] =
        osrs_inventory_cell_from_raw_osrs_id(6685);
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++)
        state.player.equipped[slot] = ITEM_NONE;
    state.player.equipped[GEAR_SLOT_WEAPON] = ITEM_TWISTED_BOW;
    state.player.equipped[GEAR_SLOT_SHIELD] = ITEM_ELYSIAN_SPIRIT_SHIELD;
    state.player.equipment_effect_profile = (OsrsEquipmentEffectProfile){
        .effect_mask = OSRS_ITEM_EFFECT_BLOOD_FURY |
            OSRS_ITEM_EFFECT_LIGHTBEARER,
        .virtus_piece_count = 2,
        .dharok_piece_count = 3,
        .crystal_armour_points = 4,
        .recoil_source = OSRS_RECOIL_SOURCE_RING_OF_RECOIL,
        .spec_regen_mode = OSRS_SPEC_REGEN_MODE_LIGHTBEARER,
        .shield_item = ITEM_ELYSIAN_SPIRIT_SHIELD,
    };

    float obs_off[INF_NUM_OBS];
    inf_write_obs_ctx((EncounterState*)&state, (EncounterContext*)&test_context, obs_off);
    ASSERT_FLOAT_NEAR("compact pending hit style",
        obs_off[INF_OBS_AFTER_NPCS],
        (float)ATTACK_STYLE_MAGIC / 4.0f, 1e-6f);
    ASSERT_FLOAT_NEAR("compact pending hit timer",
        obs_off[INF_OBS_AFTER_NPCS + 1], 0.4f, 1e-6f);
    ASSERT_FLOAT_NEAR("compact pending hit damage",
        obs_off[INF_OBS_AFTER_NPCS + 2], 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR("compact spark target relative x",
        obs_off[INF_OBS_AFTER_PENDING_HITS],
        4.0f / (float)INF_ARENA_WIDTH, 1e-6f);
    ASSERT_FLOAT_NEAR("compact spark target relative y",
        obs_off[INF_OBS_AFTER_PENDING_HITS + 1],
        6.0f / (float)INF_ARENA_HEIGHT, 1e-6f);
    ASSERT_FLOAT_NEAR("compact spark timer",
        obs_off[INF_OBS_AFTER_PENDING_HITS + 2], 0.5f, 1e-6f);
    ASSERT_FLOAT_NEAR("compact spark damage",
        obs_off[INF_OBS_AFTER_PENDING_HITS + 3], 0.8f, 1e-6f);

    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
        const OsrsInventoryCell* inventory_cell =
            &state.player.inventory_cells[cell];
        int inventory_offset = OSRS_SHARED_OBS_INVENTORY_START +
            cell * OSRS_SHARED_INVENTORY_CELL_OBS_FEATURES;
        ASSERT_FLOAT_NEAR("shared inventory cell code",
            obs_off[inventory_offset],
            osrs_inventory_cell_obs_code_encode(
                inventory_cell->content_code),
            1e-6f);
    }

    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        uint8_t item = state.player.equipped[slot];
        uint16_t content_code = item == ITEM_NONE
            ? 0 : osrs_inventory_content_code_from_item(item);
        ASSERT_FLOAT_NEAR("shared worn equipment code",
            obs_off[OSRS_SHARED_OBS_EQUIPPED_START + slot],
            osrs_inventory_cell_obs_code_encode(content_code), 1e-6f);
    }

    float expected_equipment[OSRS_EQUIPMENT_EFFECT_AGGREGATE_FEATURES];
    osrs_write_equipment_effect_aggregate(
        expected_equipment, &state.player.equipment_effect_profile);
    for (int feature = 0;
            feature < OSRS_EQUIPMENT_EFFECT_AGGREGATE_FEATURES;
            feature++) {
        ASSERT_FLOAT_NEAR("shared equipment effect aggregate",
            obs_off[OSRS_SHARED_OBS_EFFECT_START + feature],
            expected_equipment[feature], 1e-6f);
    }

}

static int reference_inferno_pillar_footprint_blocked(
    const InfernoState* state,
    int x,
    int y,
    int size
) {
    for (int pillar_idx = 0; pillar_idx < INF_NUM_PILLARS; pillar_idx++) {
        const InfPillar* pillar = &state->pillars[pillar_idx];
        if (!pillar->active) continue;
        if (los_aabb_overlap(
                x, y, size,
                pillar->x, pillar->y, INF_PILLAR_SIZE))
            return 1;
    }
    return 0;
}

static int reference_inferno_footprint_blocked(
    const InfernoState* state,
    int x,
    int y,
    int size
) {
    if (x < INF_ARENA_MIN_X || y < INF_ARENA_MIN_Y ||
            x + size - 1 > INF_ARENA_MAX_X ||
            y + size - 1 > INF_ARENA_MAX_Y)
        return 1;
    return reference_inferno_pillar_footprint_blocked(
        state, x, y, size);
}

static int reference_inferno_los_clear(
    const InfernoState* state,
    int actor_x,
    int actor_y,
    int actor_size,
    int target_x,
    int target_y,
    int target_size,
    int attack_range
) {

    LOSBlocker blockers[INF_NUM_PILLARS];
    int blocker_count = 0;
    for (int pillar_idx = 0; pillar_idx < INF_NUM_PILLARS; pillar_idx++) {
        const InfPillar* pillar = &state->pillars[pillar_idx];
        if (!pillar->active) continue;
        blockers[blocker_count++] = (LOSBlocker){
            .x = pillar->x,
            .y = pillar->y,
            .size = INF_PILLAR_SIZE,
            .los_mask = LOS_FULL_MASK,
        };
    }
    return entity_has_line_of_sight(
        blockers,
        blocker_count,
        actor_x,
        actor_y,
        actor_size,
        target_x,
        target_y,
        target_size,
        attack_range);
}

static void set_inferno_pillar_phase(
    InfernoState* state,
    int phase
) {
    for (int pillar_idx = 0; pillar_idx < INF_NUM_PILLARS; pillar_idx++) {
        state->pillars[pillar_idx].x = INF_PILLAR_POS[pillar_idx][0];
        state->pillars[pillar_idx].y = INF_PILLAR_POS[pillar_idx][1];
        state->pillars[pillar_idx].active =
            (phase & (1 << pillar_idx)) != 0;
        state->pillars[pillar_idx].hp =
            state->pillars[pillar_idx].active ? INF_PILLAR_HP : 0;
    }
}

static void test_inferno_topology_geometry_parity(void) {
    printf("--- inferno topology geometry parity ---\n");

    InfernoState state = make_test_state(20, 20);
    const InfernoContext* ctx = &test_context;
    int footprint_checks = 0;
    int los_checks = 0;
    const int target_sizes[] = {1, INF_PILLAR_SIZE, 5};
    const int attack_ranges[] = {1, 4, 10, 0};

    for (int phase = 0; phase < (1 << INF_NUM_PILLARS); phase++) {
        set_inferno_pillar_phase(&state, phase);
        for (int size = 1;
                size <= ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE;
                size++) {
            for (int x = INF_ARENA_MIN_X - size;
                    x <= INF_ARENA_MAX_X + 1;
                    x++) {
                for (int y = INF_ARENA_MIN_Y - size;
                        y <= INF_ARENA_MAX_Y + 1;
                        y++) {
                    int expected = reference_inferno_footprint_blocked(
                        &state, x, y, size);
                    int actual = inf_footprint_blocked_ctx(
                        &state, ctx, x, y, size);
                    int topology_actual =
                        encounter_arena_topology_footprint_blocked(
                            inf_route_topology_for_state(ctx, &state),
                            x,
                            y,
                            size);
                    if (expected != actual ||
                            expected != topology_actual) {
                        printf(
                            "  FAIL: footprint phase=%d anchor=(%d,%d) "
                            "size=%d expected=%d actual=%d topology=%d\n",
                            phase, x, y, size, expected, actual,
                            topology_actual);
                        tests_failed++;
                        tests_run++;
                        return;
                    }
                    footprint_checks++;
                }
            }
        }

        for (int actor_size = 1;
                actor_size <= ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE;
                actor_size++) {
            for (int actor_x = INF_ARENA_MIN_X;
                    actor_x + actor_size - 1 <= INF_ARENA_MAX_X;
                    actor_x += 3) {
                for (int actor_y = INF_ARENA_MIN_Y;
                        actor_y + actor_size - 1 <= INF_ARENA_MAX_Y;
                        actor_y += 3) {
                    for (size_t target_size_idx = 0;
                            target_size_idx <
                                sizeof(target_sizes) /
                                sizeof(target_sizes[0]);
                            target_size_idx++) {
                        int target_size = target_sizes[target_size_idx];
                        for (int target_x = INF_ARENA_MIN_X;
                                target_x + target_size - 1 <=
                                    INF_ARENA_MAX_X;
                                target_x += 4) {
                            for (int target_y = INF_ARENA_MIN_Y;
                                    target_y + target_size - 1 <=
                                        INF_ARENA_MAX_Y;
                                    target_y += 4) {
                                for (size_t range_idx = 0;
                                        range_idx <
                                            sizeof(attack_ranges) /
                                            sizeof(attack_ranges[0]);
                                        range_idx++) {
                                    int attack_range =
                                        attack_ranges[range_idx];
                                    int expected =
                                        reference_inferno_los_clear(
                                            &state,
                                            actor_x,
                                            actor_y,
                                            actor_size,
                                            target_x,
                                            target_y,
                                            target_size,
                                            attack_range);
                                    int actual = inf_los_clear_ctx(
                                        &state,
                                        ctx,
                                        actor_x,
                                        actor_y,
                                        actor_size,
                                        target_x,
                                        target_y,
                                        target_size,
                                        attack_range);
                                    if (expected != actual) {
                                        printf(
                                            "  FAIL: LOS phase=%d "
                                            "actor=(%d,%d,%d) "
                                            "target=(%d,%d,%d) range=%d "
                                            "expected=%d actual=%d\n",
                                            phase,
                                            actor_x,
                                            actor_y,
                                            actor_size,
                                            target_x,
                                            target_y,
                                            target_size,
                                            attack_range,
                                            expected,
                                            actual);
                                        tests_failed++;
                                        tests_run++;
                                        return;
                                    }
                                    los_checks++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    ASSERT_INT_EQ(
        "all topology footprint parity cases checked",
        footprint_checks > 0,
        1);
    ASSERT_INT_EQ(
        "all topology direct, area, and large-footprint LOS cases checked",
        los_checks > 0,
        1);
}

static void test_pillar_removal_resets_same_tick_los_frame(void) {
    printf("--- inferno pillar removal resets same-tick LOS frame ---\n");

    InfernoContext* ctx = &test_context;
    InfernoState state = make_test_state(30, 40);
    set_inferno_pillar_phase(&state, 1);
    state.pillars[0].hp = 1;
    state.npcs[0] = make_test_npc(
        INF_NPC_NIBBLER,
        state.pillars[0].x,
        state.pillars[0].y,
        1);
    state.npcs[0].active = 1;
    state.npcs[0].attack_timer = 0;

    uint32_t seed = 1;
    for (;;) {
        uint32_t probe = seed;
        if (encounter_rand_int(&probe, 5) > 0) break;
        seed++;
    }
    state.rng_state = seed;
    memset(ctx->npc_player_los_frame, 1, sizeof(ctx->npc_player_los_frame));

    inf_npc_attack_ctx(&state, ctx, 0);

    ASSERT_INT_EQ("nibbler removes one-hp pillar", state.pillars[0].active, 0);
    int stale_entries = 0;
    for (int npc_idx = 0; npc_idx < INF_MAX_NPCS; npc_idx++)
        stale_entries += ctx->npc_player_los_frame[npc_idx] != -1;
    ASSERT_INT_EQ(
        "pillar removal clears every same-tick NPC LOS sample",
        stale_entries,
        0);
}

static void test_inferno_topology_observation_mask_identity(void) {
    printf("--- inferno topology observation and mask identity ---\n");

    InfernoState state = make_test_state(20, 20);
    InfernoContext* ctx = &test_context;
    static const InfNPCType types[] = {
        INF_NPC_BAT,
        INF_NPC_BLOB,
        INF_NPC_MELEER,
        INF_NPC_RANGER,
        INF_NPC_MAGER,
        INF_NPC_JAD,
        INF_NPC_ZUK,
    };
    static const int positions[][2] = {
        {20, 16},
        {17, 23},
        {24, 24},
        {36, 34},
        {31, 33},
        {14, 38},
        {12, 45},
    };
    float observation[INF_NUM_OBS];
    float mask[INF_ACTION_MASK_SIZE];

    for (int phase = 0; phase < (1 << INF_NUM_PILLARS); phase++) {
        memset(state.npcs, 0, sizeof(state.npcs));
        set_inferno_pillar_phase(&state, phase);
        for (size_t npc_idx = 0;
                npc_idx < sizeof(types) / sizeof(types[0]);
                npc_idx++) {
            state.npcs[npc_idx] = make_test_npc(
                types[npc_idx],
                positions[npc_idx][0],
                positions[npc_idx][1],
                INF_NPC_STATS[types[npc_idx]].size);
            state.npcs[npc_idx].active = 1;
            state.npcs[npc_idx].aggro_target = -1;
        }
        inf_refresh_current_obs_slots_ctx(&state, ctx);
        inf_write_obs_ctx(
            (EncounterState*)&state,
            (EncounterContext*)ctx,
            observation);
        inf_write_mask_ctx(
            (EncounterState*)&state,
            (EncounterContext*)ctx,
            mask);

        for (int slot_idx = 0; slot_idx < INF_OBS_NPCS; slot_idx++) {
            int npc_idx = state.current_obs_slots[slot_idx];
            if (npc_idx < 0) continue;
            const InfNPC* npc = &state.npcs[npc_idx];
            int expected = reference_inferno_los_clear(
                &state,
                npc->x,
                npc->y,
                npc->size,
                state.player.x,
                state.player.y,
                1,
                INF_NPC_STATS[npc->type].attack_range);
            int obs_offset =
                INF_OBS_AFTER_PILLARS +
                slot_idx * INF_NPC_SLOT_FEATURES + 6;
            ASSERT_FLOAT_NEAR(
                "NPC observation LOS bit matches independent reference",
                observation[obs_offset],
                (float)expected,
                0.0f);
        }

        int movement_offset = 0;
        for (int head = 0; head < INF_HEAD_PRIMARY; head++)
            movement_offset += INF_ACTION_DIMS[head];
        for (int action = 0; action < ENCOUNTER_MOVE_ACTIONS; action++) {
            int x = state.player.x + ENCOUNTER_MOVE_TARGET_DX[action];
            int y = state.player.y + ENCOUNTER_MOVE_TARGET_DY[action];
            int expected =
                !reference_inferno_footprint_blocked(&state, x, y, 1);
            ASSERT_FLOAT_NEAR(
                "movement mask bit matches independent reference",
                mask[movement_offset + action],
                (float)expected,
                0.0f);
        }
    }
}


static void test_observation_overwrites_dirty_buffer(void) {
    printf("test_observation_overwrites_dirty_buffer\n");
    EncounterState* raw_state = inf_create();
    for (int public_wave = 62; public_wave <= 69; public_wave += 7) {
        reset_inferno_at_public_wave(raw_state, public_wave, 1.0f);
        float clean[INF_NUM_OBS] = {0};
        float dirty[INF_NUM_OBS];
        memset(dirty, 0x7f, sizeof(dirty));
        inf_write_obs_ctx(raw_state, (EncounterContext*)&test_context, clean);
        inf_write_obs_ctx(raw_state, (EncounterContext*)&test_context, dirty);
        ASSERT_INT_EQ(
            "Inferno observation overwrites every output",
            memcmp(clean, dirty, sizeof(clean)),
            0);
    }
    inf_destroy(raw_state);
}

int main(void) {
    inf_build_npc_stats();
    inf_init_context_typed(&test_context);
    test_inferno_topology_geometry_parity();
    test_pillar_removal_resets_same_tick_los_frame();
    test_inferno_topology_observation_mask_identity();
    test_compact_observation_layout_contract();
    test_compact_player_and_pillar_observation_semantics();
    test_compact_npc_observation_semantics();
    test_observation_overwrites_dirty_buffer();
    test_compact_transient_inventory_equipment_semantics();
    test_dense_target_contract_dimensions();
    test_dense_target_slots_follow_type_priority_without_holes();
    test_regular_waves_select_every_live_npc();
    test_maximal_zuk_concurrency_selects_all_live_candidates();
    test_dense_target_overflow_aborts_instead_of_truncating();

    test_melee_fallback_geometry();
    test_style_choice_sampling();
    test_tagged_jad_healer_melee_geometry();
    test_overlap_shuffle_hold_after_recent_target_click();
    test_overlap_shuffle_respects_npc_occupancy();
    test_large_npc_overlap_shuffle_can_partially_unclip();
    test_tagged_jad_healer_stops_at_melee_contact();
    test_tagged_jad_healers_queue_behind_front_healer();
    test_meleer_dig_can_stack_without_losing_collision_flag();
    test_jad_healer_spawn_offsets_match_wave_67_reference();
    test_jad_healer_spawn_offsets_match_zuk_reference();
    test_npc_terrain_blocks_full_footprint_lava_shelf();
    test_zuk_jad_healer_spawn_falls_back_to_passable_arena_tiles();
    test_meleer_dig_landing_order();
    test_final_wave_reward_applies_healer_tags_and_heal_cost();
    test_final_wave_reward_uses_zuk_low_watermark_progress();
    test_final_wave_reward_blocks_zuk_damage_while_healers_heal();
    test_final_wave_reward_pays_zuk_healer_damage();
    test_post_healer_zuk_damage_reward_is_after_clear_only();
    test_zuk_healer_phase_hp_delta_default_preserves_low_watermark();
    test_zuk_healer_phase_hp_delta_pays_healed_back_zuk_damage();
    test_zuk_healer_phase_hp_delta_avoids_double_pay_below_low_watermark();
    test_zuk_healer_phase_hp_delta_penalizes_zuk_healing_once();
    test_zuk_healer_phase_hp_delta_pays_net_same_tick_delta();
    test_zuk_healer_phase_hp_delta_keeps_non_zuk_heal_cost();
    test_post_healer_set_damage_reward_defaults_off();
    test_post_healer_set_damage_reward_pays_after_healer_clear();
    test_post_healer_set_kill_bonus_uses_existing_emitter();
    test_post_healer_set_alive_penalty_caps_per_episode();
    test_zuk_untagged_healer_tick_penalty_defaults_off();
    test_zuk_untagged_healer_tick_penalty_counts_only_untagged_zuk_healers();
    test_zuk_untagged_healer_target_bonus_defaults_off();
    test_zuk_untagged_healer_target_bonus_rewards_distinct_healers();
    test_zuk_safe_untagged_healer_target_bonus_records_safe_subset();
    test_zuk_untagged_healer_target_bonus_excludes_tagged_healers();
    test_zuk_healer_tags_first_reward_mode_blocks_pre_tag_damage();
    test_zuk_healer_tags_first_reward_mode_resumes_after_all_tags();
    test_joseph_reward_mode_pays_tags_while_healers_heal();
    test_zuk_healer_attack_shape_reward_applies_in_joseph_mode();
    test_offensive_prayer_reward_shapes_normal_and_joseph_mode();
    test_offensive_prayer_attack_events_count_real_attacks();
    test_offensive_prayer_barrage_aoe_counts_once();
    test_offensive_prayer_no_attack_no_event();
    test_offensive_prayer_melee_maps_to_piety();
    test_player_reward_damage_uses_xp_drop_tick();
    test_idle_diagnostics_count_missed_attack_opportunities();
    test_idle_diagnostics_phase_split();
    test_joseph_reward_mode_damps_healed_zuk_damage();
    test_jad_damage_reward_pauses_while_jad_healers_heal();
    test_jad_healer_damage_never_gets_damage_reward();
    test_shield_tag_reward_excludes_zuk();
    test_inferno_reset_supplies_match_current_inventory();
    test_inferno_reset_inventory_leaves_one_empty_slot();
    test_inferno_max_profile_reset_uses_existing_gear();
    test_inferno_budget_profile_reset_uses_budget_gear();
    test_inferno_mixed_profile_sampling_respects_fraction();
    test_inferno_equip_actions_move_cells_and_sync_weapon_set();
    test_inferno_gear_switch_cancels_entity_interaction();
    test_inferno_reset_preserves_reward_config();
    test_supply_milestone_reward_defaults_off();
    test_supply_milestone_reward_pays_surplus_at_anchor_once();
    test_supply_milestone_reward_never_penalizes_shortage();
    test_late_start_supply_profile_anchor_waves();
    test_late_start_supply_profile_interpolation_and_scale();
    test_curriculum_supply_no_brew_is_curriculum_only();
    test_curriculum_supply_modes_gate_zuk_and_pre_zuk();
    test_curriculum_supply_jitter_clamps_to_inventory_bounds();
    test_late_start_supply_observations();
    test_dead_mob_store_eligibility();
    test_resurrected_mob_does_not_reenter_dead_store();
    test_blob_split_waits_for_death_removal();
    test_mager_resurrection_render_event_is_not_magic_projectile();
    test_double_mager_wave_resurrection_limit();
    test_pending_hit_obs_timer_prefers_prayer_window();
    test_blob_attacks_player_on_six_tick_cadence();
    test_jad_has_no_pre_fire_style_preview();
    test_jad_fire_tick_exposes_three_tick_prayer_deadline();
    test_jad_prayer_on_third_tick_blocks();
    test_jad_prayer_first_on_fourth_tick_does_not_block();
    test_jad_long_distance_damage_uses_delayed_projectile_landing();
    test_triple_jad_pending_threats_fit_obs_layout();
    test_inferno_action_and_compact_obs_shape();
    test_inferno_obs_wave_phase_code();
    test_inferno_obs_exposes_compact_pillars();
    test_inferno_obs_exposes_meleer_dig_state();
    test_npc_threat_obs_exposes_frozen_meleer_pressure();
    test_npc_threat_obs_respects_overlap_range_and_stun();
    test_npc_threat_obs_keeps_ranger_mager_diagonal_melee();
    test_jad_special_wave_spawn_cadence_matches_reference();
    test_triple_jad_first_attacks_are_staggered();
    test_jad_melee_stays_instant_and_untelegraphed();
    test_step_out_forecast_matches_movement_head_destinations();
    test_inferno_npc_travel_uses_sw_origin_around_all_pillars();
    test_inferno_jal_npcs_use_edge_clearance_at_pillars();
    test_step_out_forecast_north_pillar_ranger_mager_order();
    test_step_out_forecast_south_pillar_ranger_mager_order();
    test_step_out_forecast_west_pillar_ranger_mager_order();
    test_step_out_forecast_inactive_pillar_does_not_create_cover();
    test_step_out_same_tick_ranger_mager_event_logs();
    test_direct_start_waves_spawn_without_empty_gap();
    test_joseph_start_wave_70_seeds_zuk_jad_checkpoint();
    test_joseph_start_wave_71_seeds_zuk_healer_checkpoint();
    test_zuk_ready_countdown_holds_npcs_then_releases();
    test_zuk_shield_does_not_set_collision_flags();
    test_zuk_attack_delay_counts_down_while_stunned();
    test_zuk_set_timer_spawns_on_decrement_to_zero();
    test_zuk_hp_threshold_pause_happens_before_set_tick();
    test_set_attack_to_shield_is_projectile_delayed();
    test_npc_target_projectile_delays_match_reference();
    test_npc_player_projectile_delays_use_reference_options();
    test_npc_hit_lands_on_the_reference_tick();
    test_player_projectile_timing_uses_reference_options();
    test_phantom_barrage_target_is_masked_until_cast_window();
    test_phantom_barrage_hits_aoe_on_first_cast_window();
    test_ranged_attack_cannot_fire_on_dying_target();
    test_autocast_barrage_cannot_fire_on_dying_target();
    test_manual_blood_barrage_can_heal_from_dying_primary();
    test_phantom_barrage_close_barrage_timing_cannot_recast();
    test_phantom_barrage_does_not_displace_live_obs_slots();
    test_default_autocast_casts_blood_barrage();
    test_ice_barrage_success_freezes_target_and_records_spell();
    test_inferno_barrage_primes_confliction_and_reuses_double_accuracy();
    test_barrage_accuracy_regression_against_ranger_and_mager();
    test_barrage_pending_queue_handles_slow_hit_delay();
    test_barrage_aoe_queues_hits_on_multiple_npcs();
    test_repeated_edge_barrages_kill_ranger();
    test_npc_pending_queue_lands_multiple_hits_in_order();
    test_npc_death_clears_pending_hits();
    test_lab_dump_reports_npc_pending_hit_queue();
    test_explicit_spell_cast_does_not_persist();
    test_spell_without_target_does_not_affect_later_attack();
    test_target_without_spell_uses_autocast();
    test_manual_spell_overrides_autocast();
    test_blood_barrage_at_full_hp_is_valid_and_heals_zero();
    test_manual_spell_in_range_gear_uses_range_gear_magic_stats();
    test_phantom_barrage_allows_explicit_spell_from_range_gear();
    test_zuk_obs_exposes_attack_timer_summary();
    test_zuk_obs_exposes_pending_sparks();
    test_human_blowpipe_click_chases_zuk_out_of_range();
    test_zuk_healer_blowpipe_target_chases_out_of_range();
    test_render_facing_prefers_attack_target_while_chasing();
    test_render_identity_survives_npc_death_compaction();
    test_render_identity_matches_two_players_across_tick();
    test_render_identity_two_players_claim_unique_slots();
    test_render_identity_single_player_unchanged();
    test_render_motion_speed_ladder_matches_deob();
    test_render_motion_lone_step_takes_32_client_ticks();
    test_render_motion_continuous_movement_never_pauses();
    test_render_motion_waypoint_pop_snap_and_overflow();
    test_render_motion_seed_classification_uses_explicit_teleport();
    test_entity_model_ground_lift_keeps_floor_planes_above_terrain();
    test_spotanim_lookup_prefers_recolored_model_alias();
    test_inferno_npc_spawn_id_changes_on_slot_reuse();
    test_anim_rest_pose_resets_working_vertices();
    test_zuk_healer_target_action_tags_on_landed_hit();
    test_zuk_healer_mage_attack_counts_penalty_event();
    test_zuk_safe_healer_target_mask_requires_fire_window();
    test_zuk_safe_healer_target_mask_clears_unsafe_target();
    test_zuk_force_safe_healer_target_mask_blocks_idle_when_safe();
    test_zuk_force_safe_healer_target_mask_clears_stale_target();
    test_zuk_spark_render_matches_pending_spark_state();
    test_zuk_obs_tracks_shield_and_mager_aggro();
    test_zuk_healer_obs_exposes_target_category();
    test_inferno_obs_target_categories_cover_boss_helpers();
    test_zuk_set_obs_los_uses_current_target();
    test_zuk_set_threat_ignores_shield_target();
    test_fail_fast_boundaries();
    test_human_target_and_potion_translation();
    test_human_targeting_refreshes_stale_obs_slots();
    test_human_spell_selection_is_client_local_until_target_click();
    test_human_walk_command_sends_no_selected_spell_cast();
    test_human_autocast_selection_persists_across_weapon_switches();
    test_autocast_is_inactive_with_non_autocast_weapon();
    test_echo_boots_recoil_reflects_to_attacker_only();
    test_redemption_action_maps_without_smite();
    test_redemption_zero_hit_landing_heals_and_drains();
    test_redemption_does_not_prevent_lethal_damage();
    test_redemption_procs_on_locked_zero_projectile_landing();
    test_human_autocast_works_with_dragon_hunter_wand();
    test_inferno_snapshot_restore_round_trip();
    test_inferno_snapshot_preserves_loadout_profile();
    test_inferno_restore_builds_npc_stats_before_late_spawn();
    test_inferno_snapshot_preserves_external_pointers();
    test_inferno_state_assignment_copy_replays_trajectory();
    test_inferno_refresh_after_state_load_rebuilds_derived_state();
    test_inferno_healer_transition_stats_track_episode_progress();
    test_inferno_human_equip_does_not_snap_loadout();
    test_jad_render_uses_style_specific_attack_animation();
    test_inferno_render_uses_npc_death_animation();
    test_jad_magic_render_emits_three_offset_projectiles();
    test_jad_ranged_render_uses_target_anchored_two_tick_visual();
    test_jad_projectile_long_distance_visual_duration_uses_reference_formula();
    test_inferno_npc_projectile_render_uses_reference_visual_timing();
    test_inferno_npc_projectile_render_tracks_target_npc_slot();
    test_inferno_zuk_projectile_render_uses_combat_visual_rows();
    test_player_projectile_render_uses_stored_reference_timing();
    test_projectile_anchor_effect_subtile_round_trips_entity_center();
    test_magic_splash_landing_keeps_spell_visual_context();
    test_npc_overkill_hit_caps_splat_hp_and_damage_stats();
    test_blood_barrage_overkill_heals_from_capped_damage();
    test_elysian_proc_propagates_to_player_render_entity();
    test_delayed_player_hit_records_landing_source();
    test_terminal_reward_uses_fixed_win_reward();
    test_final_wave_completion_emits_terminal_reward();
    test_lethal_pending_hit_banks_damage_stats_before_terminal();
    test_terminal_penalty_applies_to_death_when_enabled();
    test_timeout_reward_matches_episode_return();
    test_terminal_penalty_applies_to_timeout_when_enabled();
    test_inferno_render_overlay_reports_death_source();
    test_inferno_reset_uses_osrs_run_energy_units();

    return osrs_test_summary();
}
