#include "g1_combat_tick.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t checks;
static size_t apply_calls;

#define CHECK(value) do { \
    checks++; \
    if (!(value)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #value); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static const RekG1ImpactEvent LEFT_KICK_APEX = {
    .impact_time_seconds = 0.1f,
    .lead_time_seconds = 0.1f,
    .release_time_seconds = 0.2f,
    .limb = REK_G1_AIM_LIMB_LEFT_LOWER_BODY,
};

static RekG1HitContact contact_at(
        float time_seconds,
        RekG1BodyZone zone,
        int32_t move_id) {
    const RekG1HitContact contact = {
        .strike_intent = {
            .impact_events = &LEFT_KICK_APEX,
            .impact_event_count = 1u,
            .clip_cursor_frames = 5.0f,
            .clip_fps = 50.0f,
            .move_id = move_id,
            .action_playing = 1u,
            .layer_active = 1u,
            .layer_loop = 0u,
        },
        .striker_body_position_world = {0.0f, 0.0f, 1.0f},
        .target_body_position_world = {1.0f, 0.0f, 1.0f},
        .striker_body_linear_velocity_world = {2.5f, 0.0f, 0.0f},
        .target_body_linear_velocity_world = {0.0f, 0.0f, 0.0f},
        .relative_speed_mps = 2.5f,
        .time_seconds = time_seconds,
        .striker_part = REK_G1_BODY_PART_FOOT,
        .striker_side = REK_G1_HAND_LEFT,
        .target_zone = zone,
        .striker_fighter = 0u,
        .target_fighter = 1u,
        .striker_body_slot = 2u,
        .is_enter = 1u,
        .round_active = 1u,
        .striker_upright = 1u,
        .target_upright = 1u,
        .target_standing = 1u,
    };
    return contact;
}

static RekG1CombatSubstepInput neutral_substep(
        const RekG1CombatArenaState* state,
        float delta_seconds) {
    RekG1CombatSubstepInput input;
    memset(&input, 0, sizeof(input));
    input.delta_seconds = delta_seconds;
    input.time_remaining_seconds = state->fight.time_remaining_seconds
        > delta_seconds
        ? state->fight.time_remaining_seconds - delta_seconds
        : 0.0f;
    input.fall_phase[0] = REK_G1_FALL_UPRIGHT;
    input.fall_phase[1] = REK_G1_FALL_UPRIGHT;
    return input;
}

static void apply(
        RekG1CombatArenaState* state,
        const RekG1HitDetectorConfig* config,
        const RekG1CombatSubstepInput* input,
        RekG1CombatSubstepResult* output) {
    const RekG1CombatTickStatus status = rek_g1_combat_arena_substep(
        state, config, input, output);
    apply_calls++;
    if (status != REK_G1_COMBAT_TICK_OK) {
        fprintf(stderr, "combat substep %zu status: %s\n",
            apply_calls, rek_g1_combat_tick_status_string(status));
    }
    CHECK(status == REK_G1_COMBAT_TICK_OK);
    *state = output->next_state;
}

static void test_init_and_score_once(void) {
    RekG1CombatArenaState state;
    memset(&state, 0xA5, sizeof(state));
    CHECK(rek_g1_combat_arena_init(&state) == REK_G1_COMBAT_TICK_OK);
    CHECK(state.initialized == 1u);
    CHECK(state.fight.phase == REK_G1_FIGHT_ROUND_ACTIVE);
    CHECK(state.fight.current_round_number == 1u);
    CHECK(state.fight.time_remaining_seconds == 120.0f);

    const RekG1HitDetectorConfig config =
        rek_g1_current_build_hit_detector_config();
    RekG1HitContact contact = contact_at(
        0.002f, REK_G1_BODY_ZONE_TORSO, 1);
    RekG1CombatSubstepInput input = neutral_substep(&state, 0.002f);
    input.contacts = &contact;
    input.contact_count = 1u;
    RekG1CombatSubstepResult result;
    apply(&state, &config, &input, &result);
    CHECK(result.attributed_contact_count == 1u);
    CHECK(result.scored_contact_count == 1u);
    CHECK(result.score_delta[0] == 2 && result.score_delta[1] == 0);
    CHECK(state.fight.clean_hits[0] == 2);

    contact.time_seconds = 0.004f;
    input = neutral_substep(&state, 0.002f);
    input.contacts = &contact;
    input.contact_count = 1u;
    apply(&state, &config, &input, &result);
    CHECK(result.attributed_contact_count == 1u);
    CHECK(result.scored_contact_count == 0u);
    CHECK(result.score_delta[0] == 0 && result.score_delta[1] == 0);
    CHECK(state.fight.clean_hits[0] == 2);
}

static void test_attribution_fall_and_no_recovery_reset(void) {
    RekG1CombatArenaState state;
    CHECK(rek_g1_combat_arena_init(&state) == REK_G1_COMBAT_TICK_OK);
    const RekG1HitDetectorConfig config =
        rek_g1_current_build_hit_detector_config();
    RekG1HitContact contact = contact_at(
        0.002f, REK_G1_BODY_ZONE_LEFT_WRIST, 10);
    RekG1CombatSubstepInput input = neutral_substep(&state, 0.002f);
    input.contacts = &contact;
    input.contact_count = 1u;
    input.fall_phase[1] = REK_G1_FALL_FALLING;
    input.fall_events[1] = REK_G1_FALL_EVENT_FALLING_STARTED;
    RekG1CombatSubstepResult result;
    apply(&state, &config, &input, &result);
    CHECK(result.attributed_contact_count == 1u);
    CHECK(result.scored_contact_count == 0u);
    CHECK(result.score_delta[0] == 0);
    CHECK(state.fight.last_struck_valid[1] == 1u);
    CHECK(state.fight.fall_classification[1]
        == REK_G1_FALL_KNOCKDOWN);

    input = neutral_substep(&state, 0.002f);
    input.fall_phase[1] = REK_G1_FALL_FALLEN;
    input.fall_events[1] = REK_G1_FALL_EVENT_BECAME_FALLEN;
    apply(&state, &config, &input, &result);
    CHECK(state.fight.falls[1] == 1u);
    CHECK(state.fight.count_active[1] == 1u);
    CHECK(state.fight.count_elapsed_seconds == 0.0f);
    CHECK(state.fight.count_duration_seconds == 3.0f);
    CHECK((result.referee_calls & REK_G1_REFEREE_KNOCKDOWN) != 0u);

    input = neutral_substep(&state, 3.0f);
    input.fall_phase[1] = REK_G1_FALL_FALLEN;
    apply(&state, &config, &input, &result);
    CHECK(result.score_delta[0] == 5 && result.score_delta[1] == 0);
    CHECK((result.referee_calls & REK_G1_REFEREE_KNOCKOUT) != 0u);
    CHECK((result.signals & REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN)
        != 0u);
    CHECK(state.fight.phase == REK_G1_FIGHT_ROUND_ACTIVE);
    CHECK(state.fight.clean_hits[0] == 5);

    RekG1CombatArenaState reset;
    CHECK(rek_g1_combat_arena_apply_spawn_reset(&state, &reset)
        == REK_G1_COMBAT_TICK_OK);
    CHECK(!reset.fight.last_struck_valid[0]
        && !reset.fight.last_struck_valid[1]);
    CHECK(reset.hit_detector.scored_move_seen[0] == 0u
        && reset.hit_detector.cooldown_seen[0][2] == 0u);
    CHECK(reset.fight.clean_hits[0] == 5);
}

static void test_double_fall_and_deadline_order(void) {
    RekG1CombatArenaState state;
    CHECK(rek_g1_combat_arena_init(&state) == REK_G1_COMBAT_TICK_OK);
    const RekG1HitDetectorConfig config =
        rek_g1_current_build_hit_detector_config();
    RekG1CombatSubstepInput input = neutral_substep(&state, 0.002f);
    input.fall_phase[0] = REK_G1_FALL_FALLEN;
    input.fall_phase[1] = REK_G1_FALL_FALLEN;
    input.fall_events[0] = REK_G1_FALL_EVENT_FALLING_STARTED
        | REK_G1_FALL_EVENT_BECAME_FALLEN;
    input.fall_events[1] = REK_G1_FALL_EVENT_FALLING_STARTED
        | REK_G1_FALL_EVENT_BECAME_FALLEN;
    RekG1CombatSubstepResult result;
    apply(&state, &config, &input, &result);
    CHECK(state.fight.count_active[0] && state.fight.count_active[1]);
    CHECK(state.fight.count_duration_seconds == 3.0f);
    CHECK(state.fight.count_elapsed_seconds == 0.0f);
    CHECK((result.referee_calls & REK_G1_REFEREE_DOUBLE_KNOCKDOWN)
        != 0u);

    CHECK(rek_g1_combat_arena_init(&state) == REK_G1_COMBAT_TICK_OK);
    input = neutral_substep(&state, 0.02f);
    input.time_remaining_seconds = 0.0f;
    input.fall_phase[0] = REK_G1_FALL_FALLEN;
    input.fall_events[0] = REK_G1_FALL_EVENT_FALLING_STARTED
        | REK_G1_FALL_EVENT_BECAME_FALLEN;
    apply(&state, &config, &input, &result);
    CHECK(state.fight.phase == REK_G1_FIGHT_ROUND_ACTIVE);
    CHECK(state.fight.count_active[0]);
    CHECK(state.fight.count_elapsed_seconds == 0.0f);
    CHECK((result.signals & REK_G1_FIGHT_SIGNAL_ROUND_ENDED) == 0u);
}

static void test_transactional_rejection(void) {
    RekG1CombatArenaState state;
    CHECK(rek_g1_combat_arena_init(&state) == REK_G1_COMBAT_TICK_OK);
    const RekG1HitDetectorConfig config =
        rek_g1_current_build_hit_detector_config();
    const RekG1CombatArenaState before = state;
    RekG1CombatSubstepInput input = neutral_substep(&state, 0.002f);
    input.fighter_can_get_up[0] = 2u;
    RekG1CombatSubstepResult output;
    memset(&output, 0x5A, sizeof(output));
    const RekG1CombatSubstepResult output_before = output;
    CHECK(rek_g1_combat_arena_substep(
        &state, &config, &input, &output)
        == REK_G1_COMBAT_TICK_INPUT_INVALID);
    CHECK(memcmp(&state, &before, sizeof(state)) == 0);
    CHECK(memcmp(&output, &output_before, sizeof(output)) == 0);

    input = neutral_substep(&state, 0.002f);
    RekG1HitContact invalid_contact = contact_at(
        0.002f, REK_G1_BODY_ZONE_TORSO, 2);
    invalid_contact.relative_speed_mps = NAN;
    input.contacts = &invalid_contact;
    input.contact_count = 1u;
    CHECK(rek_g1_combat_arena_substep(
        &state, &config, &input, &output)
        == REK_G1_COMBAT_TICK_HIT_REJECTED);
    CHECK(memcmp(&state, &before, sizeof(state)) == 0);
    CHECK(memcmp(&output, &output_before, sizeof(output)) == 0);
}

static void test_inactive_round_attribution_only(void) {
    RekG1CombatArenaState state;
    CHECK(rek_g1_combat_arena_init(&state) == REK_G1_COMBAT_TICK_OK);
    const RekG1HitDetectorConfig config =
        rek_g1_current_build_hit_detector_config();
    RekG1CombatSubstepInput input = neutral_substep(&state, 0.002f);
    input.time_remaining_seconds = 0.0f;
    RekG1CombatSubstepResult result;
    apply(&state, &config, &input, &result);
    CHECK(state.fight.phase == REK_G1_FIGHT_BETWEEN_ROUNDS);

    RekG1HitContact contact = contact_at(
        0.002f, REK_G1_BODY_ZONE_TORSO, 20);
    contact.round_active = 0u;
    input = neutral_substep(&state, 0.002f);
    input.time_remaining_seconds = state.fight.time_remaining_seconds;
    input.contacts = &contact;
    input.contact_count = 1u;
    apply(&state, &config, &input, &result);
    CHECK(result.attributed_contact_count == 1u);
    CHECK(result.scored_contact_count == 0u);
    CHECK(result.score_delta[0] == 0 && result.score_delta[1] == 0);
    CHECK(state.fight.last_struck_valid[1] == 1u);
    CHECK(state.fight.clean_hits[0] == 0);
    CHECK(state.fight.phase == REK_G1_FIGHT_BETWEEN_ROUNDS);

    input = neutral_substep(&state, 0.002f);
    input.time_remaining_seconds = state.fight.time_remaining_seconds;
    input.fall_events[0] = REK_G1_FALL_EVENT_FALLING_STARTED;
    RekG1CombatSubstepResult sentinel;
    memset(&sentinel, 0x5a, sizeof(sentinel));
    result = sentinel;
    CHECK(rek_g1_combat_arena_substep(
        &state, &config, &input, &result)
        == REK_G1_COMBAT_TICK_INPUT_INVALID);
    CHECK(memcmp(&result, &sentinel, sizeof(result)) == 0);
}

int main(void) {
    test_init_and_score_once();
    test_attribution_fall_and_no_recovery_reset();
    test_double_fall_and_deadline_order();
    test_inactive_round_attribution_only();
    test_transactional_rejection();
    printf("g1 combat tick: %zu checks passed\n", checks);
    return 0;
}
