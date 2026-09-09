#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    const uint16_t move_indices[REK_G1_REQUIRED_DISCRETE_MOVE_COUNT] = {
        6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16,
    };
    const uint32_t move_duration_ticks[
            REK_G1_REQUIRED_DISCRETE_MOVE_COUNT] = {
        3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 6,
    };
    enum {
        LOCOMOTION_COUNT = sizeof(locomotion_masks) / sizeof(locomotion_masks[0]),
        CATEGORY_COUNT = 1 + LOCOMOTION_COUNT +
            REK_G1_REQUIRED_DISCRETE_MOVE_COUNT,
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
                .move_registry_index = REK_G1_SEMANTIC_MOVE_NONE,
            },
        };
    }
    const uint32_t move_category = 1 + LOCOMOTION_COUNT;
    for (uint16_t move = 0;
            move < REK_G1_REQUIRED_DISCRETE_MOVE_COUNT; move++) {
        categories[move_category + move] = (RekG1PufferCategory){
            .kind = REK_G1_PUFFER_START,
            .command = {
                .kind = REK_G1_SEMANTIC_DISCRETE_MOVE,
                .held_code = code(0u),
                .duration_ticks = move_duration_ticks[move],
                .move_registry_index = move,
            },
        };
    }
    RekG1PufferActionTable table = {
        .categories = categories,
        .count = CATEGORY_COUNT,
        .move_indices = move_indices,
        .move_duration_ticks = move_duration_ticks,
        .move_registry_count = REK_G1_REQUIRED_DISCRETE_MOVE_COUNT,
    };
    require(rek_g1_puffer_validate_table(&table) == REK_G1_PUFFER_OK,
        "generated_table_valid");
    require(REK_G1_PUFFER_TABLE_KICK_REGISTRY_INVALID
            == REK_G1_PUFFER_TABLE_MOVE_REGISTRY_INVALID
            && REK_G1_PUFFER_TABLE_KICK_DURATION_INVALID
                == REK_G1_PUFFER_TABLE_MOVE_DURATION_INVALID,
        "legacy_table_status_aliases_preserved");

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
        &adapter, (float)move_category, timing, 1, 0);
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
    for (uint16_t move = 0;
            move < REK_G1_REQUIRED_DISCRETE_MOVE_COUNT; move++) {
        require(!mask[move_category + move],
            "held_translation_masks_move_start");
    }
    RekG1SemanticScheduler held_before_masked_move = adapter.scheduler;
    step = rek_g1_puffer_step(
        &adapter, (float)move_category, timing, 1, 0);
    require(step.status == REK_G1_PUFFER_ACTION_MASKED,
        "translation_held_move_is_rejected_before_scheduler");
    require(adapter.scheduler.input_state.held ==
            held_before_masked_move.input_state.held &&
            adapter.scheduler.active == held_before_masked_move.active,
        "masked_translation_move_creates_no_latent_edge");

    step = rek_g1_puffer_step(&adapter, 2.0f, timing, 1, 0);
    require(step.status == REK_G1_PUFFER_OK &&
        step.semantic.input.released_edges == REK_G1_HELD_FORWARD,
        "neutral_segment_releases_translation");
    require(rek_g1_puffer_write_mask(
            &adapter, 0, 0, mask, sizeof(mask)) ==
        REK_G1_PUFFER_OK, "unsettled_mask_written");
    require(!mask[move_category], "unsettled_translation_masks_move_start");
    RekG1SemanticScheduler unsettled_before = adapter.scheduler;
    step = rek_g1_puffer_step(
        &adapter, (float)move_category, timing, 0, 0);
    require(step.status == REK_G1_PUFFER_ACTION_MASKED &&
            adapter.scheduler.active == unsettled_before.active,
        "unsettled_masked_move_creates_no_latent_edge");
    require(rek_g1_puffer_write_mask(
            &adapter, 1, 1, mask, sizeof(mask)) ==
        REK_G1_PUFFER_OK, "busy_mask_written");
    require(!mask[move_category], "busy_action_masks_move_start");
    step = rek_g1_puffer_step(
        &adapter, (float)move_category, timing, 1, 1);
    require(step.status == REK_G1_PUFFER_ACTION_MASKED &&
            !adapter.scheduler.active,
        "busy_masked_move_creates_no_latent_edge");

    step = rek_g1_puffer_step(
        &adapter, (float)move_category, timing, 1, 0);
    require(step.status == REK_G1_PUFFER_OK && step.semantic.move_start_edge,
        "settled_move_starts");
    require(step.semantic.input.yaw == 0 && step.semantic.move_active,
        "move_preempts_yaw_on_start");
    for (int tick = 0; tick < 2; tick++) {
        step = rek_g1_puffer_step(&adapter, 0.0f, timing, 1, 1);
        require(step.status == REK_G1_PUFFER_OK && step.semantic.move_active,
            "continue_advances_accepted_move");
        require(step.semantic.input.yaw == 0 &&
            !step.semantic.input.yaw_suppressed_for_attack,
            "neutral_move_keeps_effective_yaw_zero");
    }
    require(step.semantic.segment_complete, "move_exact_duration_complete");

    const RekG1InputTiming ramp_timing = {
        .elapsed_seconds = 0.02f,
        .yaw_ramp_seconds = 0.10f,
    };
    for (uint16_t move = 0;
            move < REK_G1_REQUIRED_DISCRETE_MOVE_COUNT; move++) {
        rek_g1_puffer_reset(&adapter);
        const uint8_t desired_held = (move & 1u) ?
            REK_G1_HELD_YAW_RIGHT : REK_G1_HELD_YAW_LEFT;
        const uint32_t yaw_category = (move & 1u) ? 8u : 7u;
        const float expected_sign = (move & 1u) ? -1.0f : 1.0f;

        step = rek_g1_puffer_step(
            &adapter, (float)yaw_category, ramp_timing, 1, 0);
        require(step.status == REK_G1_PUFFER_OK &&
                step.semantic.input.held == desired_held &&
                fabsf(step.semantic.input.yaw - expected_sign * 0.2f) <
                    1.0e-6f,
            "yaw_ramp_starts_before_move");
        step = rek_g1_puffer_step(&adapter, 0.0f, ramp_timing, 1, 0);
        require(step.status == REK_G1_PUFFER_OK &&
                step.semantic.segment_complete &&
                fabsf(step.semantic.input.yaw - expected_sign * 0.4f) <
                    1.0e-6f,
            "yaw_ramp_advances_before_move");

        step = rek_g1_puffer_step(
            &adapter, (float)(move_category + move), ramp_timing, 1, 0);
        require(step.status == REK_G1_PUFFER_OK &&
                step.semantic.move_start_edge &&
                step.semantic.input.held == desired_held &&
                step.semantic.input.desired_yaw == (int8_t)expected_sign &&
                step.semantic.input.yaw == 0.0f &&
                step.semantic.input.yaw_suppressed_for_attack &&
                fabsf(step.semantic.input.yaw_ramp - 0.6f) < 1.0e-6f,
            "move_preserves_desired_yaw_and_suppresses_effective_yaw");

        float prior_ramp = step.semantic.input.yaw_ramp;
        while (!step.semantic.segment_complete) {
            step = rek_g1_puffer_step(&adapter, 0.0f, ramp_timing, 1, 1);
            require(step.status == REK_G1_PUFFER_OK &&
                    step.semantic.input.held == desired_held &&
                    step.semantic.input.desired_yaw == (int8_t)expected_sign &&
                    step.semantic.input.yaw == 0.0f &&
                    step.semantic.input.yaw_suppressed_for_attack &&
                    step.semantic.input.yaw_ramp + 1.0e-6f >= prior_ramp,
                "move_retains_yaw_state_and_advances_ramp");
            prior_ramp = step.semantic.input.yaw_ramp;
        }

        step = rek_g1_puffer_step(
            &adapter, (float)yaw_category, ramp_timing, 1, 0);
        require(step.status == REK_G1_PUFFER_OK &&
                step.semantic.input.held == desired_held &&
                step.semantic.input.pressed_edges == 0 &&
                step.semantic.input.released_edges == 0 &&
                fabsf(step.semantic.input.yaw - expected_sign) < 1.0e-6f,
            "held_yaw_resumes_after_move_without_false_edge");
    }

    rek_g1_puffer_reset(&adapter);
    step = rek_g1_puffer_step(&adapter, 7.0f, ramp_timing, 1, 0);
    require(step.status == REK_G1_PUFFER_OK &&
            step.semantic.input.held == REK_G1_HELD_YAW_LEFT,
        "mid_kick_fixture_q_start");
    step = rek_g1_puffer_step(&adapter, 0.0f, ramp_timing, 1, 0);
    require(step.status == REK_G1_PUFFER_OK && step.semantic.segment_complete &&
            fabsf(step.semantic.input.yaw_ramp - 0.4f) < 1.0e-6f,
        "mid_kick_fixture_q_ramped");
    step = rek_g1_puffer_step(
        &adapter,
        (float)(move_category +
            REK_G1_REQUIRED_DISCRETE_MOVE_COUNT - 1u),
        ramp_timing,
        1,
        0);
    require(step.status == REK_G1_PUFFER_OK &&
            step.semantic.move_start_edge &&
            step.semantic.move_registry_index ==
                REK_G1_REQUIRED_DISCRETE_MOVE_COUNT - 1u &&
            step.semantic.remaining_ticks == 5u &&
            step.semantic.input.held == REK_G1_HELD_YAW_LEFT &&
            fabsf(step.semantic.input.yaw_ramp - 0.6f) < 1.0e-6f,
        "mid_move_fixture_started_without_releasing_q");
    require(rek_g1_puffer_write_mask(
            &adapter, 0, 1, mask, sizeof(mask)) == REK_G1_PUFFER_OK,
        "active_move_input_update_mask_written");
    for (uint32_t category = 0; category < table.count; category++) {
        const int expected = category == 0u || category == 2u ||
            category == 7u || category == 8u;
        require(mask[category] == expected,
            "active_move_mask_exposes_only_continue_neutral_q_e");
    }

    RekG1SemanticScheduler before_masked_active_move = adapter.scheduler;
    step = rek_g1_puffer_step(&adapter, 1.0f, ramp_timing, 0, 1);
    require(step.status == REK_G1_PUFFER_ACTION_MASKED,
        "active_move_masks_translation_selection");
    require(memcmp(
            &adapter.scheduler,
            &before_masked_active_move,
            sizeof(adapter.scheduler)) == 0,
        "masked_translation_preserves_full_active_move_scheduler");
    step = rek_g1_puffer_step(
        &adapter, (float)move_category, ramp_timing, 0, 1);
    require(step.status == REK_G1_PUFFER_ACTION_MASKED,
        "active_move_masks_second_move_selection");
    require(memcmp(
            &adapter.scheduler,
            &before_masked_active_move,
            sizeof(adapter.scheduler)) == 0,
        "masked_second_move_preserves_full_active_move_scheduler");

    step = rek_g1_puffer_step(&adapter, 0.0f, ramp_timing, 0, 1);
    require(step.status == REK_G1_PUFFER_OK &&
            !step.semantic.command_started && !step.semantic.move_start_edge &&
            step.semantic.move_registry_index ==
                REK_G1_REQUIRED_DISCRETE_MOVE_COUNT - 1u &&
            step.semantic.remaining_ticks == 4u &&
            step.semantic.input.pressed_edges == 0u &&
            step.semantic.input.released_edges == 0u &&
            fabsf(step.semantic.input.yaw_ramp - 0.8f) < 1.0e-6f &&
            step.semantic.input.yaw == 0.0f,
        "active_move_continue_retains_q_without_restart");
    step = rek_g1_puffer_step(&adapter, 2.0f, ramp_timing, 0, 1);
    require(step.status == REK_G1_PUFFER_OK &&
            step.semantic.remaining_ticks == 3u &&
            step.semantic.input.held == 0u &&
            step.semantic.input.released_edges == REK_G1_HELD_YAW_LEFT &&
            step.semantic.input.pressed_edges == 0u &&
            step.semantic.input.yaw_ramp == 0.0f &&
            step.semantic.input.yaw == 0.0f,
        "active_move_neutral_releases_q_without_restart");
    step = rek_g1_puffer_step(&adapter, 8.0f, ramp_timing, 0, 1);
    require(step.status == REK_G1_PUFFER_OK &&
            step.semantic.remaining_ticks == 2u &&
            step.semantic.input.held == REK_G1_HELD_YAW_RIGHT &&
            step.semantic.input.pressed_edges == REK_G1_HELD_YAW_RIGHT &&
            step.semantic.input.released_edges == 0u &&
            step.semantic.input.desired_yaw == -1 &&
            fabsf(step.semantic.input.yaw_ramp - 0.2f) < 1.0e-6f &&
            step.semantic.input.yaw == 0.0f,
        "active_move_e_press_starts_suppressed_ramp");
    step = rek_g1_puffer_step(&adapter, 7.0f, ramp_timing, 0, 1);
    require(step.status == REK_G1_PUFFER_OK &&
            step.semantic.remaining_ticks == 1u &&
            step.semantic.input.held == REK_G1_HELD_YAW_LEFT &&
            step.semantic.input.pressed_edges == REK_G1_HELD_YAW_LEFT &&
            step.semantic.input.released_edges == REK_G1_HELD_YAW_RIGHT &&
            step.semantic.input.desired_yaw == 1 &&
            fabsf(step.semantic.input.yaw_ramp - 0.2f) < 1.0e-6f &&
            step.semantic.input.yaw == 0.0f,
        "active_move_e_to_q_reversal_resets_suppressed_ramp");
    step = rek_g1_puffer_step(&adapter, 0.0f, ramp_timing, 0, 1);
    require(step.status == REK_G1_PUFFER_OK &&
            step.semantic.segment_complete &&
            step.semantic.remaining_ticks == 0u &&
            step.semantic.input.pressed_edges == 0u &&
            step.semantic.input.released_edges == 0u &&
            fabsf(step.semantic.input.yaw_ramp - 0.4f) < 1.0e-6f &&
            step.semantic.input.yaw == 0.0f,
        "active_move_final_continue_retains_reversed_q");
    step = rek_g1_puffer_step(&adapter, 7.0f, ramp_timing, 1, 0);
    require(step.status == REK_G1_PUFFER_OK &&
            step.semantic.input.pressed_edges == 0u &&
            step.semantic.input.released_edges == 0u &&
            fabsf(step.semantic.input.yaw - 0.6f) < 1.0e-6f,
        "reversed_q_resumes_after_move_without_false_edge");

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
        .move_indices = move_indices,
        .move_duration_ticks = move_duration_ticks,
        .move_registry_count = REK_G1_REQUIRED_DISCRETE_MOVE_COUNT,
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
    wrong_duration_categories[move_category].command.duration_ticks += 1;
    RekG1PufferActionTable wrong_duration_table = table;
    wrong_duration_table.categories = wrong_duration_categories;
    require(rek_g1_puffer_validate_table(&wrong_duration_table) ==
        REK_G1_PUFFER_TABLE_MOVE_DURATION_INVALID,
        "move_duration_must_match_configured_compositor_registry");

    RekG1PufferCategory nonneutral_move_categories[CATEGORY_COUNT];
    for (uint32_t index = 0; index < CATEGORY_COUNT; index++) {
        nonneutral_move_categories[index] = categories[index];
    }
    nonneutral_move_categories[move_category].command.held_code =
        code(REK_G1_HELD_YAW_LEFT);
    RekG1PufferActionTable nonneutral_move_table = table;
    nonneutral_move_table.categories = nonneutral_move_categories;
    require(rek_g1_puffer_validate_table(&nonneutral_move_table) ==
        REK_G1_PUFFER_TABLE_START_INVALID,
        "move_template_must_be_neutral_before_dynamic_yaw_inheritance");

    uint16_t wrong_move_indices[REK_G1_REQUIRED_DISCRETE_MOVE_COUNT];
    memcpy(wrong_move_indices, move_indices, sizeof(wrong_move_indices));
    wrong_move_indices[3] = 10u;
    RekG1PufferActionTable wrong_move_table = table;
    wrong_move_table.move_indices = wrong_move_indices;
    require(rek_g1_puffer_validate_table(&wrong_move_table) ==
        REK_G1_PUFFER_TABLE_MOVE_REGISTRY_INVALID,
        "move_registry_order_is_exact");

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
