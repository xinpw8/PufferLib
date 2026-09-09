#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "puffer_action_adapter.h"

static int assertions;

static void require(int condition, const char* name) {
    assertions += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

static uint8_t code(uint8_t held) {
    uint8_t result = 255;
    require(rek_g1_semantic_encode_held(held, &result) ==
        REK_G1_SEMANTIC_OK, "fixture_held_code");
    return result;
}

int main(void) {
    const RekG1InputTiming timing = {
        .elapsed_seconds = 0.02f,
        .yaw_ramp_seconds = 0.02f,
    };
    const uint8_t locomotion_masks[] = {
        REK_G1_HELD_FORWARD,
        0,
        REK_G1_HELD_STRAFE_RIGHT | REK_G1_HELD_YAW_LEFT,
        REK_G1_HELD_BACKWARD,
        REK_G1_HELD_STRAFE_LEFT,
        REK_G1_HELD_STRAFE_RIGHT,
        REK_G1_HELD_YAW_LEFT,
        REK_G1_HELD_YAW_RIGHT,
        REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_LEFT,
        REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_RIGHT,
        REK_G1_HELD_BACKWARD | REK_G1_HELD_YAW_LEFT,
        REK_G1_HELD_BACKWARD | REK_G1_HELD_YAW_RIGHT,
        REK_G1_HELD_STRAFE_LEFT | REK_G1_HELD_YAW_LEFT,
        REK_G1_HELD_STRAFE_LEFT | REK_G1_HELD_YAW_RIGHT,
        REK_G1_HELD_STRAFE_RIGHT | REK_G1_HELD_YAW_RIGHT,
    };
    const uint16_t kick_move_indices[] = {6, 7, 8, 9};
    const uint32_t kick_duration_ticks[] = {3, 4, 5, 6};
    const uint8_t kick_held_masks[] = {
        REK_G1_HELD_YAW_RIGHT,
        0,
        REK_G1_HELD_YAW_LEFT,
        REK_G1_HELD_YAW_RIGHT,
    };
    enum {
        LOCOMOTION_COUNT = sizeof(locomotion_masks) / sizeof(locomotion_masks[0]),
        CATEGORY_COUNT = 1 + LOCOMOTION_COUNT + REK_G1_REQUIRED_KICK_COUNT,
    };
    RekG1PufferCategory categories[CATEGORY_COUNT] = {0};
    categories[0].kind = REK_G1_PUFFER_CONTINUE;
    for (uint32_t index = 0; index < LOCOMOTION_COUNT; index++) {
        categories[1 + index] = (RekG1PufferCategory){
            .kind = REK_G1_PUFFER_START,
            .command = {
                .kind = REK_G1_SEMANTIC_LOCOMOTION,
                .held_code = code(locomotion_masks[index]),
                .duration_ticks = index == 1 ? 1 : 2,
                .kick_registry_index = REK_G1_SEMANTIC_KICK_NONE,
            },
        };
    }
    const uint32_t kick_category = 1 + LOCOMOTION_COUNT;
    for (uint16_t kick = 0; kick < REK_G1_REQUIRED_KICK_COUNT; kick++) {
        categories[kick_category + kick] = (RekG1PufferCategory){
            .kind = REK_G1_PUFFER_START,
            .command = {
                .kind = REK_G1_SEMANTIC_KICK,
                .held_code = code(kick_held_masks[kick]),
                .duration_ticks = kick_duration_ticks[kick],
                .kick_registry_index = kick,
            },
        };
    }
    RekG1PufferActionTable table = {
        .categories = categories,
        .count = CATEGORY_COUNT,
        .kick_move_indices = kick_move_indices,
        .kick_duration_ticks = kick_duration_ticks,
        .kick_registry_count = REK_G1_REQUIRED_KICK_COUNT,
    };
    require(rek_g1_puffer_validate_table(&table) == REK_G1_PUFFER_OK,
        "generated_table_valid");

    RekG1PufferAdapter adapter;
    require(rek_g1_puffer_init(&adapter, &table) == REK_G1_PUFFER_OK,
        "generated_table_bound_once");
    RekG1SemanticScheduler before_bad_timing = adapter.scheduler;
    RekG1PufferStep bad_timing = rek_g1_puffer_step(
        &adapter,
        1.0f,
        (RekG1InputTiming){
            .elapsed_seconds = 0.0f,
            .yaw_ramp_seconds = 0.5f,
        },
        1,
        0);
    require(bad_timing.status == REK_G1_PUFFER_INPUT_TIMING_INVALID,
        "puffer_step_rejects_missing_elapsed_time");
    require(adapter.scheduler.active == before_bad_timing.active &&
            adapter.scheduler.remaining_ticks ==
                before_bad_timing.remaining_ticks,
        "invalid_timing_does_not_start_category");
    uint8_t mask[CATEGORY_COUNT] = {0};
    require(rek_g1_puffer_write_mask(
            &adapter, 1, 0, mask, sizeof(mask)) ==
        REK_G1_PUFFER_OK, "idle_mask_written");
    require(!mask[0], "idle_mask_hides_continue");
    for (uint32_t category = 1; category < table.count; category++) {
        require(mask[category], "idle_mask_exposes_all_validated_starts");
    }

    RekG1PufferStep step = rek_g1_puffer_step(
        &adapter, 1.0f, timing, 1, 0);
    require(step.status == REK_G1_PUFFER_OK, "forward_start_dispatched");
    require(step.semantic.input.forward == 1 &&
        step.semantic.remaining_ticks == 1, "forward_first_tick");
    require(rek_g1_puffer_write_mask(
            &adapter, 1, 0, mask, sizeof(mask)) ==
        REK_G1_PUFFER_OK, "active_mask_written");
    require(mask[0], "active_segment_exposes_continue");
    for (uint32_t category = 1; category < table.count; category++) {
        require(!mask[category], "active_segment_hides_every_start");
    }

    RekG1SemanticScheduler before_invalid = adapter.scheduler;
    step = rek_g1_puffer_step(
        &adapter, (float)kick_category, timing, 1, 0);
    require(step.status == REK_G1_PUFFER_ACTION_MASKED,
        "new_start_masked_during_segment");
    require(adapter.scheduler.remaining_ticks == before_invalid.remaining_ticks,
        "masked_action_does_not_mutate_scheduler");
    step = rek_g1_puffer_step(&adapter, 0.0f, timing, 1, 0);
    require(step.status == REK_G1_PUFFER_OK && step.semantic.segment_complete,
        "continue_finishes_forward_segment");

    require(rek_g1_puffer_write_mask(
            &adapter, 1, 0, mask, sizeof(mask)) ==
        REK_G1_PUFFER_OK, "post_hold_mask_written");
    for (uint16_t kick = 0; kick < REK_G1_REQUIRED_KICK_COUNT; kick++) {
        require(!mask[kick_category + kick], "held_translation_masks_kick_start");
    }
    RekG1SemanticScheduler held_before_masked_kick = adapter.scheduler;
    step = rek_g1_puffer_step(
        &adapter, (float)kick_category, timing, 1, 0);
    require(step.status == REK_G1_PUFFER_ACTION_MASKED,
        "translation_held_kick_is_rejected_before_scheduler");
    require(adapter.scheduler.input_state.held ==
            held_before_masked_kick.input_state.held &&
            adapter.scheduler.active == held_before_masked_kick.active,
        "masked_translation_kick_creates_no_latent_edge");

    step = rek_g1_puffer_step(&adapter, 2.0f, timing, 1, 0);
    require(step.status == REK_G1_PUFFER_OK &&
        step.semantic.input.released_edges == REK_G1_HELD_FORWARD,
        "neutral_segment_releases_translation");
    require(rek_g1_puffer_write_mask(
            &adapter, 0, 0, mask, sizeof(mask)) ==
        REK_G1_PUFFER_OK, "unsettled_mask_written");
    require(!mask[kick_category], "unsettled_translation_masks_kick_start");
    RekG1SemanticScheduler unsettled_before = adapter.scheduler;
    step = rek_g1_puffer_step(
        &adapter, (float)kick_category, timing, 0, 0);
    require(step.status == REK_G1_PUFFER_ACTION_MASKED &&
            adapter.scheduler.active == unsettled_before.active,
        "unsettled_masked_kick_creates_no_latent_edge");
    require(rek_g1_puffer_write_mask(
            &adapter, 1, 1, mask, sizeof(mask)) ==
        REK_G1_PUFFER_OK, "busy_mask_written");
    require(!mask[kick_category], "busy_action_masks_kick_start");
    step = rek_g1_puffer_step(
        &adapter, (float)kick_category, timing, 1, 1);
    require(step.status == REK_G1_PUFFER_ACTION_MASKED &&
            !adapter.scheduler.active,
        "busy_masked_kick_creates_no_latent_edge");

    step = rek_g1_puffer_step(
        &adapter, (float)kick_category, timing, 1, 0);
    require(step.status == REK_G1_PUFFER_OK && step.semantic.kick_start_edge,
        "settled_kick_starts");
    require(step.semantic.input.yaw == 0 && step.semantic.kick_active,
        "kick_preempts_yaw_on_start");
    for (int tick = 0; tick < 2; tick++) {
        step = rek_g1_puffer_step(&adapter, 0.0f, timing, 1, 1);
        require(step.status == REK_G1_PUFFER_OK && step.semantic.kick_active,
            "continue_advances_accepted_kick");
        require(step.semantic.input.yaw == 0 &&
            step.semantic.input.yaw_suppressed_for_attack,
            "kick_preempts_yaw_for_every_duration_tick");
    }
    require(step.semantic.segment_complete, "kick_exact_duration_complete");

    step = rek_g1_puffer_step(&adapter, 1.5f, timing, 1, 0);
    require(step.status == REK_G1_PUFFER_ACTION_NOT_INTEGRAL,
        "fractional_category_rejected");
    step = rek_g1_puffer_step(&adapter, NAN, timing, 1, 0);
    require(step.status == REK_G1_PUFFER_ACTION_NOT_FINITE,
        "nan_category_rejected");
    step = rek_g1_puffer_step(
        &adapter, (float)table.count, timing, 1, 0);
    require(step.status == REK_G1_PUFFER_ACTION_OUT_OF_RANGE,
        "out_of_range_category_rejected");

    RekG1PufferCategory duplicate_categories[] = {
        categories[0], categories[1], categories[1],
    };
    RekG1PufferActionTable duplicate_table = {
        .categories = duplicate_categories,
        .count = 3,
        .kick_move_indices = kick_move_indices,
        .kick_duration_ticks = kick_duration_ticks,
        .kick_registry_count = REK_G1_REQUIRED_KICK_COUNT,
    };
    require(rek_g1_puffer_validate_table(&duplicate_table) ==
        REK_G1_PUFFER_TABLE_DUPLICATE, "duplicate_category_rejected");

    RekG1PufferActionTable precision_unsafe_table = table;
    precision_unsafe_table.count = REK_G1_PUFFER_EXACT_CATEGORY_LIMIT + 1u;
    require(rek_g1_puffer_validate_table(&precision_unsafe_table) ==
        REK_G1_PUFFER_TABLE_COUNT_INVALID,
        "bf16_unsafe_category_count_rejected");

    RekG1PufferCategory wrong_duration_categories[CATEGORY_COUNT];
    for (uint32_t index = 0; index < CATEGORY_COUNT; index++) {
        wrong_duration_categories[index] = categories[index];
    }
    wrong_duration_categories[kick_category].command.duration_ticks += 1;
    RekG1PufferActionTable wrong_duration_table = table;
    wrong_duration_table.categories = wrong_duration_categories;
    require(rek_g1_puffer_validate_table(&wrong_duration_table) ==
        REK_G1_PUFFER_TABLE_KICK_DURATION_INVALID,
        "kick_duration_must_match_configured_compositor_registry");

    uint16_t wrong_move_indices[] = {6, 7, 8, 10};
    RekG1PufferActionTable wrong_move_table = table;
    wrong_move_table.kick_move_indices = wrong_move_indices;
    require(rek_g1_puffer_validate_table(&wrong_move_table) ==
        REK_G1_PUFFER_TABLE_KICK_REGISTRY_INVALID,
        "kick_registry_must_bind_moves_6_through_9");

    RekG1PufferCategory missing_held_categories[CATEGORY_COUNT];
    for (uint32_t index = 0; index < CATEGORY_COUNT; index++) {
        missing_held_categories[index] = categories[index];
    }
    missing_held_categories[4].command.held_code = code(
        REK_G1_HELD_FORWARD | REK_G1_HELD_STRAFE_LEFT);
    RekG1PufferActionTable missing_held_table = table;
    missing_held_table.categories = missing_held_categories;
    require(rek_g1_puffer_validate_table(&missing_held_table) ==
        REK_G1_PUFFER_TABLE_HELD_COVERAGE_INVALID,
        "required_cardinal_hold_cannot_be_omitted");

    rek_g1_puffer_reset(&adapter);
    step = rek_g1_puffer_step(&adapter, 3.0f, timing, 1, 0);
    require(step.status == REK_G1_PUFFER_OK &&
        step.semantic.input.strafe == -1 && step.semantic.input.yaw == 1,
        "d_hold_and_q_overlap_dispatch");
    step = rek_g1_puffer_step(&adapter, 0.0f, timing, 1, 0);
    require(step.status == REK_G1_PUFFER_OK && step.semantic.segment_complete,
        "d_q_hold_finishes_at_exact_duration");
    step = rek_g1_puffer_step(&adapter, 3.0f, timing, 1, 0);
    require(step.status == REK_G1_PUFFER_OK &&
            step.semantic.input.strafe == -1 && step.semantic.input.yaw == 1 &&
            step.semantic.input.pressed_edges == 0 &&
            step.semantic.input.released_edges == 0,
        "same_mask_segment_relatch_continues_hold_without_false_edges");

    printf("PASS g1_puffer_action_adapter assertions=%d categories=%u\n",
        assertions, table.count);
    return 0;
}
