#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "semantic_binding.h"

enum {
    FAKE_OUTPUT_PROBES = 29,
    MASK_GUARD_LEFT = 0xa5,
    MASK_GUARD_RIGHT = 0x5a,
};

#define ACTION_GUARD_LEFT 123456.0f
#define ACTION_GUARD_RIGHT -654321.0f

typedef struct FakeRuntime {
    int reset_calls;
    int advance_calls;
    int fail_reset;
    int fail_advance;
    RekG1RuntimeFacts facts;
    RekG1SemanticTick last_semantic;
    float semantic_output_probes[FAKE_OUTPUT_PROBES];
} FakeRuntime;

static int assertions;

static void require(int condition, const char* name) {
    assertions += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

static uint8_t held_code(uint8_t held) {
    uint8_t code = UINT8_MAX;
    require(rek_g1_semantic_encode_held(held, &code) ==
        REK_G1_SEMANTIC_OK, "fixture_held_code");
    return code;
}

static int fake_reset(void* context, RekG1RuntimeFacts* facts_out) {
    FakeRuntime* runtime = (FakeRuntime*)context;
    runtime->reset_calls += 1;
    if (runtime->fail_reset) return 1;
    memset(&runtime->last_semantic, 0, sizeof(runtime->last_semantic));
    for (int probe = 0; probe < FAKE_OUTPUT_PROBES; probe++) {
        runtime->semantic_output_probes[probe] = -1000.0f - (float)probe;
    }
    *facts_out = runtime->facts;
    return 0;
}

static int fake_advance(
        void* context,
        const RekG1SemanticTick* semantic,
        RekG1RuntimeFacts* next_facts_out) {
    FakeRuntime* runtime = (FakeRuntime*)context;
    runtime->advance_calls += 1;
    if (runtime->fail_advance) return 1;
    runtime->last_semantic = *semantic;

    // This is test instrumentation, not motor behavior. Its only purpose is to
    // prove that the runtime sees semantic fields and never the raw category.
    float marker = 100.0f * (float)semantic->input.forward +
        10.0f * (float)semantic->input.strafe +
        semantic->input.yaw;
    if (semantic->kick_start_edge) marker += 1000.0f;
    for (int probe = 0; probe < FAKE_OUTPUT_PROBES; probe++) {
        runtime->semantic_output_probes[probe] = marker + (float)probe / 100.0f;
    }
    *next_facts_out = runtime->facts;
    return 0;
}

static void build_table(
        RekG1PufferCategory categories[REK_G1_PUFFER_MIN_CATEGORIES],
        RekG1PufferActionTable* table) {
    static const uint8_t locomotion_masks[] = {
        0,
        REK_G1_HELD_FORWARD,
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
        REK_G1_HELD_STRAFE_RIGHT | REK_G1_HELD_YAW_LEFT,
        REK_G1_HELD_STRAFE_RIGHT | REK_G1_HELD_YAW_RIGHT,
    };
    static const uint16_t kick_move_indices[] = {6, 7, 8, 9};
    static const uint32_t kick_duration_ticks[] = {2, 3, 4, 5};

    memset(
        categories,
        0,
        REK_G1_PUFFER_MIN_CATEGORIES * sizeof(*categories));
    categories[0].kind = REK_G1_PUFFER_CONTINUE;
    for (uint32_t index = 0;
            index < sizeof(locomotion_masks) / sizeof(locomotion_masks[0]);
            index++) {
        categories[1 + index] = (RekG1PufferCategory){
            .kind = REK_G1_PUFFER_START,
            .command = {
                .kind = REK_G1_SEMANTIC_LOCOMOTION,
                .held_code = held_code(locomotion_masks[index]),
                .duration_ticks = index == 0 ? 1 : 2,
                .kick_registry_index = REK_G1_SEMANTIC_KICK_NONE,
            },
        };
    }
    for (uint16_t kick = 0; kick < REK_G1_REQUIRED_KICK_COUNT; kick++) {
        categories[16 + kick] = (RekG1PufferCategory){
            .kind = REK_G1_PUFFER_START,
            .command = {
                .kind = REK_G1_SEMANTIC_KICK,
                .held_code = held_code(0),
                .duration_ticks = kick_duration_ticks[kick],
                .kick_registry_index = kick,
            },
        };
    }
    *table = (RekG1PufferActionTable){
        .categories = categories,
        .count = REK_G1_PUFFER_MIN_CATEGORIES,
        .kick_move_indices = kick_move_indices,
        .kick_duration_ticks = kick_duration_ticks,
        .kick_registry_count = REK_G1_REQUIRED_KICK_COUNT,
    };
}

static void require_idle_mask(const uint8_t* mask) {
    require(mask[0] == 0, "idle_continue_masked");
    for (int category = 1;
            category < REK_G1_PUFFER_MIN_CATEGORIES;
            category++) {
        require(mask[category] == 1, "idle_start_exposed");
    }
}

static void require_zero_mask(const uint8_t* mask) {
    for (int category = 0;
            category < REK_G1_PUFFER_MIN_CATEGORIES;
            category++) {
        require(mask[category] == 0, "invalid_binding_masks_every_category");
    }
}

static void require_mask_guards(const uint8_t* storage) {
    require(storage[0] == MASK_GUARD_LEFT, "mask_left_guard_intact");
    require(storage[REK_G1_PUFFER_MIN_CATEGORIES + 1] == MASK_GUARD_RIGHT,
        "mask_right_guard_intact");
}

// Models the eventual c_step boundary: one and only one Puffer action float is
// passed to the semantic binding. The surrounding values verify that this
// buffer-facing entry point does not overwrite adjacent action storage.
static RekG1BindingStep step_category(
        RekG1SemanticBinding* binding,
        float category) {
    float action_storage[] = {
        ACTION_GUARD_LEFT,
        category,
        ACTION_GUARD_RIGHT,
    };
    RekG1BindingStep result = rek_g1_binding_step_actions(
        binding,
        &action_storage[1],
        REK_G1_PUFFER_ACTION_HEADS);
    require(action_storage[0] == ACTION_GUARD_LEFT &&
            action_storage[2] == ACTION_GUARD_RIGHT,
        "one_float_action_buffer_guards_intact");
    return result;
}

int main(void) {
    RekG1PufferCategory categories[REK_G1_PUFFER_MIN_CATEGORIES];
    RekG1PufferActionTable table;
    build_table(categories, &table);
    require(rek_g1_puffer_validate_table(&table) == REK_G1_PUFFER_OK,
        "fixture_table_valid");
    require(REK_G1_PUFFER_ACTION_HEADS == 1,
        "binding_has_one_categorical_head");

    RekG1RuntimeOps runtime_ops = {
        .reset = fake_reset,
        .advance = fake_advance,
    };
    FakeRuntime runtime = {
        .facts = {
            .timing = {
                .elapsed_seconds = 0.02f,
                .yaw_ramp_seconds = 0.5f,
            },
            .translation_transition_settled = 1,
        },
    };
    uint8_t mask_storage[REK_G1_PUFFER_MIN_CATEGORIES + 2];
    memset(mask_storage, 0xcc, sizeof(mask_storage));
    mask_storage[0] = MASK_GUARD_LEFT;
    mask_storage[REK_G1_PUFFER_MIN_CATEGORIES + 1] = MASK_GUARD_RIGHT;
    uint8_t* mask = &mask_storage[1];
    RekG1SemanticBinding binding;
    require(rek_g1_binding_init(
            &binding,
            &table,
            &runtime_ops,
            &runtime,
            mask,
            REK_G1_PUFFER_MIN_CATEGORIES) == REK_G1_BINDING_OK,
        "binding_initializes");
    require(binding.initialized && !binding.ready,
        "binding_requires_successful_reset");
    require_zero_mask(mask);
    require_mask_guards(mask_storage);

    RekG1BindingStep step = step_category(&binding, 2.0f);
    require(step.status == REK_G1_BINDING_NOT_READY && !step.puffer_valid,
        "step_before_reset_fails_closed");
    require(runtime.advance_calls == 0,
        "step_before_reset_never_reaches_runtime");
    float invalid_head_count_actions[] = {2.0f, 3.0f};
    step = rek_g1_binding_step_actions(
        &binding, invalid_head_count_actions, 2);
    require(step.status == REK_G1_BINDING_ACTION_ABI_INVALID &&
            !step.puffer_valid,
        "action_head_count_must_be_exactly_one");
    require(runtime.advance_calls == 0,
        "invalid_action_abi_never_reaches_runtime");

    require(rek_g1_binding_reset(&binding) == REK_G1_BINDING_OK,
        "binding_resets");
    require(binding.ready, "binding_ready_after_reset");
    require(runtime.reset_calls == 1, "runtime_reset_once");
    require_idle_mask(mask);
    require_mask_guards(mask_storage);

    // Category 2 is W in this fixture. The one-element Puffer action buffer is
    // guarded on both sides. The only value delivered to the fake runtime is
    // the resulting semantic tick.
    step = step_category(&binding, 2.0f);
    require(step.status == REK_G1_BINDING_OK,
        "forward_category_dispatched");
    require(step.puffer_valid, "forward_adapter_result_valid");
    require(runtime.advance_calls == 1, "runtime_advanced_once");
    require(runtime.last_semantic.input.forward == 1 &&
            runtime.last_semantic.input.held == REK_G1_HELD_FORWARD,
        "runtime_received_forward_semantics");
    require(runtime.semantic_output_probes[0] == 100.0f,
        "fake_output_probe_derives_from_semantics");
    require(mask[0] == 1, "active_continue_exposed");
    for (int category = 1;
            category < REK_G1_PUFFER_MIN_CATEGORIES;
            category++) {
        require(mask[category] == 0, "active_start_masked");
    }

    float probes_before_rejected[FAKE_OUTPUT_PROBES];
    memcpy(
        probes_before_rejected,
        runtime.semantic_output_probes,
        sizeof(probes_before_rejected));
    int calls_before_rejected = runtime.advance_calls;
    step = step_category(&binding, 16.0f);
    require(step.status == REK_G1_BINDING_ACTION_REJECTED &&
            step.puffer_valid &&
            step.puffer.status == REK_G1_PUFFER_ACTION_MASKED,
        "masked_start_rejected");
    require(runtime.advance_calls == calls_before_rejected,
        "masked_start_never_reaches_runtime");
    require(memcmp(
            probes_before_rejected,
            runtime.semantic_output_probes,
            sizeof(probes_before_rejected)) == 0,
        "masked_category_cannot_touch_output_probes");

    step = step_category(&binding, 0.0f);
    require(step.status == REK_G1_BINDING_OK &&
            step.puffer.semantic.segment_complete,
        "continue_completes_forward_segment");
    require(mask[16] == 0,
        "prior_translation_masks_kick_after_segment");

    runtime.facts.recovery_active = 1;
    step = step_category(&binding, 1.0f);
    require(step.status == REK_G1_BINDING_OK &&
            step.puffer.semantic.input.released_edges ==
                REK_G1_HELD_FORWARD,
        "neutral_category_releases_translation");
    for (int category = 1; category < 16; category++) {
        require(mask[category] == 1,
            "recovery_keeps_locomotion_categories_available");
    }
    for (int category = 16;
            category < REK_G1_PUFFER_MIN_CATEGORIES;
            category++) {
        require(mask[category] == 0,
            "recovery_masks_kick_categories");
    }
    calls_before_rejected = runtime.advance_calls;
    step = step_category(&binding, 16.0f);
    require(step.status == REK_G1_BINDING_ACTION_REJECTED &&
            runtime.advance_calls == calls_before_rejected,
        "recovery_masked_kick_never_reaches_runtime");

    runtime.facts.recovery_active = 0;
    step = step_category(&binding, 1.0f);
    require(step.status == REK_G1_BINDING_OK,
        "runtime_recovery_release_observed");
    require_idle_mask(mask);

    runtime.facts.input_reset = 1;
    step = step_category(&binding, 2.0f);
    require(step.status == REK_G1_BINDING_OK
            && runtime.last_semantic.input.held == REK_G1_HELD_FORWARD,
        "input_reset_transition_reaches_runtime");
    require(!binding.adapter.scheduler.active,
        "input_reset_clears_semantic_scheduler");
    require_idle_mask(mask);
    runtime.facts.input_reset = 0;

    step = step_category(&binding, 6.0f);
    require(step.status == REK_G1_BINDING_OK &&
            runtime.last_semantic.input.held == REK_G1_HELD_YAW_LEFT &&
            fabsf(runtime.last_semantic.input.yaw - 0.04f) < 1.0e-6f,
        "pre_kick_q_ramp_starts");
    step = step_category(&binding, 0.0f);
    require(step.status == REK_G1_BINDING_OK &&
            step.puffer.semantic.segment_complete &&
            fabsf(runtime.last_semantic.input.yaw - 0.08f) < 1.0e-6f,
        "pre_kick_q_ramp_advances");

    step = step_category(&binding, 19.0f);
    require(step.status == REK_G1_BINDING_OK,
        "kick_category_dispatched");
    require(runtime.last_semantic.kick_start_edge &&
            runtime.last_semantic.kick_registry_index == 3u &&
            runtime.last_semantic.input.held == REK_G1_HELD_YAW_LEFT &&
            runtime.last_semantic.input.desired_yaw == 1 &&
            fabsf(runtime.last_semantic.input.yaw_ramp - 0.12f) < 1.0e-6f &&
            runtime.last_semantic.input.yaw == 0.0f,
        "runtime_received_preempted_kick_semantics");
    require(runtime.semantic_output_probes[0] == 1000.0f,
        "kick_output_probe_uses_semantic_edge");
    for (int category = 0;
            category < REK_G1_PUFFER_MIN_CATEGORIES;
            category++) {
        const int expected = category == 0 || category == 1 ||
            category == 6 || category == 7;
        require(mask[category] == expected,
            "kick_exposes_only_continue_neutral_q_e");
    }

    step = step_category(&binding, 0.0f);
    require(step.status == REK_G1_BINDING_OK &&
            !runtime.last_semantic.kick_start_edge &&
            runtime.last_semantic.kick_registry_index == 3u &&
            runtime.last_semantic.remaining_ticks == 3u &&
            runtime.last_semantic.input.pressed_edges == 0u &&
            runtime.last_semantic.input.released_edges == 0u &&
            fabsf(runtime.last_semantic.input.yaw_ramp - 0.16f) < 1.0e-6f &&
            runtime.last_semantic.input.yaw == 0.0f,
        "binding_continue_retains_suppressed_q");
    step = step_category(&binding, 1.0f);
    require(step.status == REK_G1_BINDING_OK &&
            runtime.last_semantic.kick_registry_index == 3u &&
            runtime.last_semantic.remaining_ticks == 2u &&
            runtime.last_semantic.input.held == 0u &&
            runtime.last_semantic.input.released_edges ==
                REK_G1_HELD_YAW_LEFT &&
            runtime.last_semantic.input.yaw_ramp == 0.0f,
        "binding_neutral_updates_active_kick");
    step = step_category(&binding, 7.0f);
    require(step.status == REK_G1_BINDING_OK &&
            runtime.last_semantic.kick_registry_index == 3u &&
            runtime.last_semantic.remaining_ticks == 1u &&
            runtime.last_semantic.input.held == REK_G1_HELD_YAW_RIGHT &&
            runtime.last_semantic.input.pressed_edges ==
                REK_G1_HELD_YAW_RIGHT &&
            runtime.last_semantic.input.yaw == 0.0f,
        "binding_e_updates_active_kick");
    step = step_category(&binding, 6.0f);
    require(step.status == REK_G1_BINDING_OK &&
            step.puffer.semantic.segment_complete &&
            runtime.last_semantic.kick_registry_index == 3u &&
            runtime.last_semantic.input.held == REK_G1_HELD_YAW_LEFT &&
            runtime.last_semantic.input.pressed_edges ==
                REK_G1_HELD_YAW_LEFT &&
            runtime.last_semantic.input.released_edges ==
                REK_G1_HELD_YAW_RIGHT &&
            fabsf(runtime.last_semantic.input.yaw_ramp - 0.04f) < 1.0e-6f &&
            runtime.last_semantic.input.yaw == 0.0f,
        "binding_q_reversal_completes_same_kick");
    step = step_category(&binding, 6.0f);
    require(step.status == REK_G1_BINDING_OK &&
            runtime.last_semantic.input.pressed_edges == 0u &&
            runtime.last_semantic.input.released_edges == 0u &&
            fabsf(runtime.last_semantic.input.yaw - 0.08f) < 1.0e-6f,
        "binding_q_resumes_after_kick_without_false_edge");

    int calls_before_invalid = runtime.advance_calls;
    float probes_before_invalid[FAKE_OUTPUT_PROBES];
    memcpy(
        probes_before_invalid,
        runtime.semantic_output_probes,
        sizeof(probes_before_invalid));
    const float invalid_categories[] = {
        NAN,
        1.5f,
        (float)REK_G1_PUFFER_MIN_CATEGORIES,
    };
    const RekG1PufferStatus invalid_statuses[] = {
        REK_G1_PUFFER_ACTION_NOT_FINITE,
        REK_G1_PUFFER_ACTION_NOT_INTEGRAL,
        REK_G1_PUFFER_ACTION_OUT_OF_RANGE,
    };
    for (int index = 0; index < 3; index++) {
        step = step_category(&binding, invalid_categories[index]);
        require(step.status == REK_G1_BINDING_ACTION_REJECTED &&
                step.puffer_valid &&
                step.puffer.status == invalid_statuses[index],
            "invalid_category_rejected");
        require(runtime.advance_calls == calls_before_invalid,
            "invalid_category_never_reaches_runtime");
        require(memcmp(
                probes_before_invalid,
                runtime.semantic_output_probes,
                sizeof(probes_before_invalid)) == 0,
            "invalid_category_cannot_touch_output_probes");
    }

    RekG1PufferAdapter first_adapter_snapshot = binding.adapter;
    RekG1RuntimeFacts first_facts_snapshot = binding.facts;
    uint8_t first_mask_snapshot[REK_G1_PUFFER_MIN_CATEGORIES];
    memcpy(first_mask_snapshot, mask, sizeof(first_mask_snapshot));
    int first_calls_snapshot = runtime.advance_calls;

    FakeRuntime second_runtime = runtime;
    second_runtime.reset_calls = 0;
    second_runtime.advance_calls = 0;
    uint8_t second_mask[REK_G1_PUFFER_MIN_CATEGORIES] = {0};
    RekG1SemanticBinding second_binding;
    require(rek_g1_binding_init(
            &second_binding,
            &table,
            &runtime_ops,
            &second_runtime,
            second_mask,
            sizeof(second_mask)) == REK_G1_BINDING_OK,
        "second_binding_initializes");
    require(rek_g1_binding_reset(&second_binding) == REK_G1_BINDING_OK,
        "second_binding_resets");
    require(step_category(&second_binding, 4.0f).status ==
            REK_G1_BINDING_OK,
        "second_binding_strafe_starts");
    require(second_runtime.last_semantic.input.strafe == 1,
        "second_runtime_receives_own_semantics");
    require(memcmp(
            &binding.adapter,
            &first_adapter_snapshot,
            sizeof(first_adapter_snapshot)) == 0,
        "first_adapter_state_isolated");
    require(memcmp(
            &binding.facts,
            &first_facts_snapshot,
            sizeof(first_facts_snapshot)) == 0,
        "first_runtime_facts_isolated");
    require(memcmp(mask, first_mask_snapshot, sizeof(first_mask_snapshot)) == 0,
        "first_action_mask_isolated");
    require(runtime.advance_calls == first_calls_snapshot,
        "first_runtime_callback_count_isolated");

    uint8_t short_mask[REK_G1_PUFFER_MIN_CATEGORIES - 1] = {0};
    RekG1SemanticBinding rejected_binding;
    require(rek_g1_binding_init(
            &rejected_binding,
            &table,
            &runtime_ops,
            &runtime,
            short_mask,
            sizeof(short_mask)) == REK_G1_BINDING_ACTION_MASK_INVALID,
        "mask_width_must_equal_category_count");

    for (int invalid_fact = 0; invalid_fact < 5; invalid_fact++) {
        FakeRuntime bad_facts_runtime = runtime;
        bad_facts_runtime.reset_calls = 0;
        bad_facts_runtime.facts = (RekG1RuntimeFacts){
            .timing = {
                .elapsed_seconds = 0.02f,
                .yaw_ramp_seconds = 0.5f,
            },
            .translation_transition_settled = 1,
        };
        if (invalid_fact == 0) {
            bad_facts_runtime.facts.timing.elapsed_seconds = 0.0f;
        } else if (invalid_fact == 1) {
            bad_facts_runtime.facts.translation_transition_settled = 2;
        } else if (invalid_fact == 2) {
            bad_facts_runtime.facts.action_busy = 2;
        } else if (invalid_fact == 3) {
            bad_facts_runtime.facts.recovery_active = 2;
        } else {
            bad_facts_runtime.facts.input_reset = 2;
        }
        RekG1SemanticBinding bad_facts_binding;
        uint8_t bad_facts_mask[REK_G1_PUFFER_MIN_CATEGORIES];
        memset(bad_facts_mask, 0xff, sizeof(bad_facts_mask));
        require(rek_g1_binding_init(
                &bad_facts_binding,
                &table,
                &runtime_ops,
                &bad_facts_runtime,
                bad_facts_mask,
                sizeof(bad_facts_mask)) == REK_G1_BINDING_OK,
            "bad_facts_binding_initializes");
        require(rek_g1_binding_reset(&bad_facts_binding) ==
                REK_G1_BINDING_RUNTIME_FACTS_INVALID,
            "invalid_runtime_facts_fail_closed");
        require(!bad_facts_binding.ready,
            "invalid_runtime_facts_leave_binding_not_ready");
        require_zero_mask(bad_facts_mask);
    }

    FakeRuntime failure_runtime = {
        .facts = {
            .timing = {
                .elapsed_seconds = 0.02f,
                .yaw_ramp_seconds = 0.5f,
            },
            .translation_transition_settled = 1,
        },
    };
    RekG1RuntimeOps copied_runtime_ops = runtime_ops;
    RekG1SemanticBinding failure_binding;
    uint8_t failure_mask[REK_G1_PUFFER_MIN_CATEGORIES];
    memset(failure_mask, 0xff, sizeof(failure_mask));
    require(rek_g1_binding_init(
            &failure_binding,
            &table,
            &copied_runtime_ops,
            &failure_runtime,
            failure_mask,
            sizeof(failure_mask)) == REK_G1_BINDING_OK,
        "failure_binding_initializes");
    copied_runtime_ops = (RekG1RuntimeOps){0};
    require(rek_g1_binding_reset(&failure_binding) == REK_G1_BINDING_OK,
        "binding_owns_runtime_callback_copy");
    require(step_category(&failure_binding, 2.0f).status ==
            REK_G1_BINDING_OK,
        "failure_fixture_segment_starts");

    failure_runtime.fail_reset = 1;
    require(rek_g1_binding_reset(&failure_binding) ==
            REK_G1_BINDING_RUNTIME_RESET_FAILED,
        "runtime_reset_failure_reported");
    require(!failure_binding.ready &&
            !failure_binding.adapter.scheduler.active,
        "reset_failure_invalidates_episode_state");
    require_zero_mask(failure_mask);
    int failure_calls = failure_runtime.advance_calls;
    step = step_category(&failure_binding, 2.0f);
    require(step.status == REK_G1_BINDING_NOT_READY && !step.puffer_valid,
        "step_after_reset_failure_is_rejected");
    require(failure_runtime.advance_calls == failure_calls,
        "step_after_reset_failure_never_reaches_runtime");

    failure_runtime.fail_reset = 0;
    require(rek_g1_binding_reset(&failure_binding) == REK_G1_BINDING_OK,
        "successful_reset_recovers_binding");
    failure_runtime.fail_advance = 1;
    float probes_before_advance_failure[FAKE_OUTPUT_PROBES];
    memcpy(
        probes_before_advance_failure,
        failure_runtime.semantic_output_probes,
        sizeof(probes_before_advance_failure));
    step = step_category(&failure_binding, 2.0f);
    require(step.status == REK_G1_BINDING_RUNTIME_ADVANCE_FAILED &&
            step.puffer_valid,
        "runtime_advance_failure_reported");
    require(!failure_binding.ready &&
            !failure_binding.adapter.scheduler.active,
        "advance_failure_invalidates_episode_state");
    require_zero_mask(failure_mask);
    require(memcmp(
            probes_before_advance_failure,
            failure_runtime.semantic_output_probes,
            sizeof(probes_before_advance_failure)) == 0,
        "failed_runtime_does_not_write_output_probes");
    failure_calls = failure_runtime.advance_calls;
    require(step_category(&failure_binding, 2.0f).status ==
            REK_G1_BINDING_NOT_READY,
        "advance_failure_requires_reset");
    require(failure_runtime.advance_calls == failure_calls,
        "invalidated_binding_never_retries_runtime");

    failure_runtime.fail_advance = 0;
    require(rek_g1_binding_reset(&failure_binding) == REK_G1_BINDING_OK,
        "reset_after_advance_failure_succeeds");
    failure_runtime.facts.action_busy = 2;
    step = step_category(&failure_binding, 2.0f);
    require(step.status == REK_G1_BINDING_RUNTIME_FACTS_INVALID &&
            step.puffer_valid,
        "invalid_post_advance_facts_reported");
    require(!failure_binding.ready, "invalid_post_advance_facts_invalidate");
    require_zero_mask(failure_mask);
    failure_calls = failure_runtime.advance_calls;
    require(step_category(&failure_binding, 2.0f).status ==
            REK_G1_BINDING_NOT_READY,
        "invalid_post_advance_facts_require_reset");
    require(failure_runtime.advance_calls == failure_calls,
        "invalid_post_advance_facts_prevent_runtime_retry");

    failure_runtime.facts.action_busy = 0;
    require(rek_g1_binding_reset(&failure_binding) == REK_G1_BINDING_OK,
        "reset_before_mask_failure_succeeds");
    uint32_t saved_table_count = table.count;
    table.count -= 1;
    step = step_category(&failure_binding, 2.0f);
    require(step.status == REK_G1_BINDING_ACTION_MASK_INVALID &&
            step.puffer_valid,
        "post_dispatch_mask_failure_reported");
    require(!failure_binding.ready, "mask_failure_invalidates_episode");
    require_zero_mask(failure_mask);
    table.count = saved_table_count;
    require_mask_guards(mask_storage);

    printf(
        "PASS g1_semantic_binding assertions=%d categories=%d "
        "semantic_output_probes=%d raw_category_callback_fields=0\n",
        assertions,
        REK_G1_PUFFER_MIN_CATEGORIES,
        FAKE_OUTPUT_PROBES);
    return 0;
}
