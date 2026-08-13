#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ocean/osrs/encounters/encounter_inferno.h"

#include "ocean/osrs/tests/osrs_test_check.h"

#define ASSERT_CONTAINS(label, haystack, needle) do { \
    tests_run++; \
    if (strstr((haystack), (needle)) != NULL) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        printf("  FAIL: %s - missing %s\n", (label), (needle)); \
    } \
} while (0)

static InfernoState* make_lab_state(void) {
    InfernoState* state = (InfernoState*)inf_create();
    inf_put_float((EncounterState*)state, "late_start_supply_profile_scale", 1.0f);
    inf_reset((EncounterState*)state, 20260515u);
    inf_lab_apply_command(state, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_CLEAR_NPCS,
    });
    return state;
}

static void test_lab_typed_commands_mutate_state(void) {
    printf("--- inferno lab typed commands mutate state ---\n");

    InfernoState* state = make_lab_state();

    inf_lab_apply_command(state, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_SET_PLAYER,
        .as.tile = { .x = 29, .y = 39 },
    });
    ASSERT_INT_EQ("player x", state->player.x, 29);
    ASSERT_INT_EQ("player y", state->player.y, 39);

    inf_lab_apply_command(state, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_SPAWN_NPC,
        .as.spawn_npc = {
            .slot = 0,
            .type = INF_NPC_RANGER,
            .x = 24,
            .y = 31,
            .hp = { .kind = INF_LAB_OPTIONAL_INT_UNSET },
            .timer = { .kind = INF_LAB_OPTIONAL_INT_SET, .value = 3 },
        },
    });
    ASSERT_INT_EQ("ranger active", state->npcs[0].active, 1);
    ASSERT_INT_EQ("ranger type", state->npcs[0].type, INF_NPC_RANGER);
    ASSERT_INT_EQ("ranger timer", state->npcs[0].attack_timer, 3);

    inf_lab_apply_command(state, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_MOVE_NPC,
        .as.move_npc = { .slot = 0, .x = 20, .y = 32 },
    });
    ASSERT_INT_EQ("ranger moved x", state->npcs[0].x, 20);
    ASSERT_INT_EQ("ranger moved y", state->npcs[0].y, 32);

    inf_lab_apply_command(state, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_SET_NPC_HP,
        .as.npc_hp = { .slot = 0, .hp = 7 },
    });
    inf_lab_apply_command(state, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_SET_NPC_TIMER,
        .as.npc_timer = { .slot = 0, .timer = 0 },
    });
    ASSERT_INT_EQ("ranger hp", state->npcs[0].hp, 7);
    ASSERT_INT_EQ("ranger timer zero", state->npcs[0].attack_timer, 0);

    inf_lab_apply_command(state, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_SET_PILLAR,
        .as.pillar = {
            .pillar_idx = 2,
            .state = INF_LAB_PILLAR_REMOVED,
            .hp = { .kind = INF_LAB_OPTIONAL_INT_SET, .value = 0 },
        },
    });
    ASSERT_INT_EQ("north pillar inactive", state->pillars[2].active, 0);
    ASSERT_INT_EQ("north pillar removed from LOS blockers", state->los_blocker_count, 2);

    inf_destroy((EncounterState*)state);
}

static void add_script_line(InfernoState* state, const char* line) {
    InfLabLineResult result = inf_lab_apply_script_line(state, line);
    ASSERT_INT_EQ("script line does not dump", result, INF_LAB_LINE_NONE);
}

static void setup_north_pillar_stack_from_script(InfernoState* state) {
    add_script_line(state, "clear_npcs");
    add_script_line(state, "player x=29 y=39");
    add_script_line(state, "pillar idx=0 active=1 hp=255");
    add_script_line(state, "pillar idx=1 active=1 hp=255");
    add_script_line(state, "pillar idx=2 active=1 hp=255");
    add_script_line(state, "npc slot=0 type=ranger x=24 y=31 hp=full timer=0");
    add_script_line(state, "npc slot=1 type=mager x=29 y=30 hp=full timer=0");
}

static void test_lab_script_reaches_exact_forecast(void) {
    printf("--- inferno lab script reaches exact forecast ---\n");

    InfernoState* state = make_lab_state();
    setup_north_pillar_stack_from_script(state);

    InfLabLineResult result = inf_lab_apply_script_line(state, "forecast");
    ASSERT_INT_EQ("forecast line result", result, INF_LAB_LINE_FORECAST);

    InfStepOutForecast forecast;
    inf_build_step_out_forecast(state, &forecast);
    const InfStepOutForecastAction* run_west = &forecast.actions[11];
    ASSERT_INT_EQ("run west valid", run_west->valid, 1);
    ASSERT_INT_EQ("run west land x", run_west->land_x, 27);
    ASSERT_INT_EQ("run west land y", run_west->land_y, 39);
    ASSERT_INT_EQ("ranger fires first", run_west->ticks[0].ranger_count, 1);
    ASSERT_INT_EQ("mager fires second", run_west->ticks[1].mager_count, 1);
    ASSERT_INT_EQ("offtick opportunity", run_west->ranger_mager_offtick_opportunity, 1);
    ASSERT_INT_EQ("no same tick conflict", run_west->same_tick_mixed_style_conflict, 0);

    inf_destroy((EncounterState*)state);
}

static void test_lab_json_contains_state_and_forecast(void) {
    printf("--- inferno lab json contains state and forecast ---\n");

    InfernoState* state = make_lab_state();
    setup_north_pillar_stack_from_script(state);

    char* json = inf_lab_alloc_json(state);
    ASSERT_CONTAINS("json has player", json, "\"player\":{\"x\":29,\"y\":39");
    ASSERT_CONTAINS("json has ranger", json, "\"type\":\"ranger\"");
    ASSERT_CONTAINS("json has mager", json, "\"type\":\"mager\"");
    ASSERT_CONTAINS("json has forecast", json, "\"forecast\"");
    ASSERT_CONTAINS("json has rich ticks", json, "\"ticks\":[");
    ASSERT_CONTAINS("json has off-tick affordance", json, "\"ranger_mager_offtick\":1");
    free(json);

    char* dump = NULL;
    InfLabLineResult result = inf_lab_apply_script_line_alloc_json(
        state, "dump", &dump);
    ASSERT_INT_EQ("dump line result", result, INF_LAB_LINE_DUMP);
    ASSERT_CONTAINS("dump has action array", dump, "\"actions\":[");
    free(dump);

    inf_destroy((EncounterState*)state);
}

static void test_lab_spawn_wave_and_delete(void) {
    printf("--- inferno lab spawn wave and delete ---\n");

    InfernoState* state = make_lab_state();

    inf_lab_apply_command(state, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_SPAWN_WAVE,
        .as.wave = { .wave = 60 },
    });
    ASSERT_INT_EQ("wave 60 internal index", state->wave, 59);

    int active_count = 0;
    for (int i = 0; i < INF_MAX_NPCS; i++)
        if (state->npcs[i].active) active_count++;
    ASSERT_INT_EQ("wave 60 active NPC count", active_count, 7);

    inf_lab_apply_command(state, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_DELETE_NPC,
        .as.npc_slot = { .slot = 0 },
    });
    ASSERT_INT_EQ("slot deleted", state->npcs[0].active, 0);

    inf_destroy((EncounterState*)state);
}

static void test_lab_snapshot_restore_round_trip(void) {
    printf("--- inferno lab snapshot restore round trip ---\n");

    InfernoState* state = make_lab_state();
    inf_lab_apply_command(state, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_SET_PLAYER,
        .as.tile = { .x = 29, .y = 39 },
    });
    inf_lab_apply_command(state, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_SPAWN_NPC,
        .as.spawn_npc = {
            .slot = 3,
            .type = INF_NPC_MAGER,
            .x = 27,
            .y = 32,
            .hp = { .kind = INF_LAB_OPTIONAL_INT_SET, .value = 99 },
            .timer = { .kind = INF_LAB_OPTIONAL_INT_SET, .value = 2 },
        },
    });
    inf_lab_apply_command(state, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_SET_PILLAR,
        .as.pillar = {
            .pillar_idx = 1,
            .state = INF_LAB_PILLAR_REMOVED,
            .hp = { .kind = INF_LAB_OPTIONAL_INT_SET, .value = 0 },
        },
    });
    state->wave = 11;
    state->tick = 321;
    state->rng_state = 0x1234abcd;
    state->player.brew_doses = 7;
    state->player.restore_doses = 11;
    state->player_pending_hits.count = 2;

    size_t snapshot_size = ENCOUNTER_INFERNO.snapshot_size(
        (EncounterState*)state, (EncounterContext*)inf_legacy_context());
    ASSERT_INT_EQ("snapshot size", (int)snapshot_size, (int)sizeof(InfSnapshot));
    InfSnapshot* snapshot = (InfSnapshot*)malloc(snapshot_size);
    ENCOUNTER_INFERNO.snapshot(
        (EncounterState*)state,
        (EncounterContext*)inf_legacy_context(),
        snapshot);

    inf_lab_apply_command(state, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_SET_PLAYER,
        .as.tile = { .x = 18, .y = 18 },
    });
    inf_lab_apply_command(state, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_CLEAR_NPCS,
    });
    inf_lab_apply_command(state, &(InfernoLabCommand){
        .kind = INF_LAB_COMMAND_SET_PILLAR,
        .as.pillar = {
            .pillar_idx = 1,
            .state = INF_LAB_PILLAR_ACTIVE,
            .hp = { .kind = INF_LAB_OPTIONAL_INT_SET, .value = INF_PILLAR_HP },
        },
    });
    state->wave = 60;
    state->tick = 999;
    state->rng_state = 7;
    state->player.brew_doses = 0;
    state->player.restore_doses = 0;
    encounter_pending_hit_queue_clear(&state->player_pending_hits);

    ENCOUNTER_INFERNO.restore(
        (EncounterState*)state,
        (EncounterContext*)inf_legacy_context(),
        snapshot,
        snapshot_size);

    ASSERT_INT_EQ("restored player x", state->player.x, 29);
    ASSERT_INT_EQ("restored player y", state->player.y, 39);
    ASSERT_INT_EQ("restored mager active", state->npcs[3].active, 1);
    ASSERT_INT_EQ("restored mager type", state->npcs[3].type, INF_NPC_MAGER);
    ASSERT_INT_EQ("restored mager hp", state->npcs[3].hp, 99);
    ASSERT_INT_EQ("restored mager timer", state->npcs[3].attack_timer, 2);
    ASSERT_INT_EQ("restored west pillar inactive", state->pillars[1].active, 0);
    ASSERT_INT_EQ("restored LOS blockers", state->los_blocker_count, 2);
    ASSERT_INT_EQ("restored wave", state->wave, 11);
    ASSERT_INT_EQ("restored tick", state->tick, 321);
    ASSERT_INT_EQ("restored rng", (int)state->rng_state, (int)0x1234abcd);
    ASSERT_INT_EQ("restored brews", state->player.brew_doses, 7);
    ASSERT_INT_EQ("restored restores", state->player.restore_doses, 11);
    ASSERT_INT_EQ("restored pending hits", state->player_pending_hits.count, 2);

    free(snapshot);
    inf_destroy((EncounterState*)state);
}

int main(void) {
    test_lab_typed_commands_mutate_state();
    test_lab_script_reaches_exact_forecast();
    test_lab_json_contains_state_and_forecast();
    test_lab_spawn_wave_and_delete();
    test_lab_snapshot_restore_round_trip();

    return osrs_test_summary();
}
