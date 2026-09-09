#include "native_puffer_vector.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEST_ENVS = 3,
    TEST_OBS_FLOATS = 4,
    TEST_CATEGORIES = 20,
};

typedef struct FakeRuntime {
    int reset_calls;
    int advance_calls;
    int close_calls;
    int fail_reset;
    int fail_advance;
    int emit_invalid_facts;
    int emit_invalid_reward;
    int emit_invalid_terminal;
    int emit_terminal_pair;
    uint32_t emit_input_reset_mask;
    RekG1SemanticTick captured[TEST_ENVS];
} FakeRuntime;

static int assertions;

static void require(int condition, const char* name) {
    assertions += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

static RekG1RuntimeFacts good_facts(void) {
    return (RekG1RuntimeFacts){
        .timing = {
            .elapsed_seconds = 0.02f,
            .yaw_ramp_seconds = 0.10f,
        },
        .translation_transition_settled = 1,
        .action_busy = 0,
        .recovery_active = 0,
    };
}

static int fake_reset(
        void* context,
        const RekG1NativeMotionRouteTable* motion_routes,
        const RekG1PufferActionTable* action_table,
        size_t environment_count,
        RekG1RuntimeFacts* facts_out,
        void* observations,
        size_t observation_stride_bytes,
        float* rewards,
        float* terminals,
        char* error,
        size_t error_capacity) {
    FakeRuntime* runtime = (FakeRuntime*)context;
    runtime->reset_calls += 1;
    require(rek_g1_native_validate_static_motion_routes(motion_routes),
        "reset_receives_pinned_motion_routes");
    require(action_table != NULL && action_table->count == TEST_CATEGORIES,
        "reset_receives_action_registry");
    require(environment_count == TEST_ENVS, "reset_receives_complete_batch");
    if (runtime->fail_reset) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity, "injected reset failure");
        }
        return 0;
    }
    for (size_t index = 0; index < environment_count; index++) {
        facts_out[index] = good_facts();
        if (runtime->emit_invalid_facts && index == 1) {
            facts_out[index].timing.elapsed_seconds = NAN;
        }
        float* row = (float*)((unsigned char*)observations
            + index * observation_stride_bytes);
        row[0] = (float)index;
        row[1] = 0.0f;
        row[2] = 0.0f;
        row[3] = 0.0f;
        rewards[index] = 0.0f;
        terminals[index] = 0.0f;
        if (runtime->emit_invalid_reward && index == 1u) {
            rewards[index] = NAN;
        }
        if (runtime->emit_invalid_terminal && index == 1u) {
            terminals[index] = -1.0f;
        }
    }
    return 1;
}

static int fake_advance(
        void* context,
        const RekG1NativeMotionRouteTable* motion_routes,
        const RekG1PufferActionTable* action_table,
        const RekG1SemanticTick* semantics,
        size_t environment_count,
        RekG1RuntimeFacts* next_facts_out,
        void* observations,
        size_t observation_stride_bytes,
        float* rewards,
        float* terminals,
        char* error,
        size_t error_capacity) {
    FakeRuntime* runtime = (FakeRuntime*)context;
    runtime->advance_calls += 1;
    require(rek_g1_native_validate_static_motion_routes(motion_routes),
        "advance_receives_pinned_motion_routes");
    require(action_table != NULL && action_table->count == TEST_CATEGORIES,
        "advance_receives_action_registry");
    require(environment_count == TEST_ENVS, "advance_receives_complete_batch");
    memcpy(runtime->captured, semantics, sizeof(runtime->captured));
    if (runtime->fail_advance) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity, "injected advance failure");
        }
        return 0;
    }
    for (size_t index = 0; index < environment_count; index++) {
        next_facts_out[index] = good_facts();
        next_facts_out[index].input_reset =
            (runtime->emit_input_reset_mask & (1u << index)) != 0u;
        if (semantics[index].kind == REK_G1_SEMANTIC_KICK) {
            require(semantics[index].kick_registry_index <
                    action_table->kick_registry_count,
                "kick_registry_index_in_callback_range");
            uint16_t move_index = action_table->kick_move_indices[
                semantics[index].kick_registry_index];
            require(rek_g1_native_kick_route(
                    motion_routes, move_index) != NULL,
                "kick_registry_resolves_exact_static_route");
        }
        if (runtime->emit_invalid_facts && index == 2) {
            next_facts_out[index].action_busy = 2;
        }
        float* row = (float*)((unsigned char*)observations
            + index * observation_stride_bytes);
        row[0] = (float)semantics[index].input.held;
        row[1] = (float)semantics[index].input.forward;
        row[2] = (float)semantics[index].input.strafe;
        row[3] = semantics[index].input.yaw;
        rewards[index] = (float)semantics[index].kind;
        terminals[index] = 0.0f;
        if (runtime->emit_invalid_reward && index == 1u) {
            rewards[index] = INFINITY;
        }
        if (runtime->emit_invalid_terminal && index == 1u) {
            terminals[index] = 2.0f;
        }
    }
    if (runtime->emit_terminal_pair) {
        terminals[0] = 1.0f;
        terminals[1] = 1.0f;
        runtime->emit_terminal_pair = 0;
    }
    runtime->emit_input_reset_mask = 0u;
    return 1;
}

static void fake_close(void* context) {
    FakeRuntime* runtime = (FakeRuntime*)context;
    runtime->close_calls += 1;
}

static RekG1NativeBatchOps fake_ops(void) {
    return (RekG1NativeBatchOps){
        .runtime_facts_abi_version = REK_G1_RUNTIME_FACTS_ABI_VERSION,
        .runtime_facts_size = REK_G1_RUNTIME_FACTS_SIZE,
        .reset = fake_reset,
        .advance = fake_advance,
        .close = fake_close,
    };
}

static uint8_t held_code(uint8_t held) {
    uint8_t code = 255;
    require(
        rek_g1_semantic_encode_held(held, &code) == REK_G1_SEMANTIC_OK,
        "test_held_code_valid");
    return code;
}

static RekG1PufferActionTable make_table(
        RekG1PufferCategory* categories,
        uint16_t* kick_indices,
        uint32_t* kick_durations) {
    static const uint8_t required_held[] = {
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
    memset(categories, 0, TEST_CATEGORIES * sizeof(*categories));
    categories[0].kind = REK_G1_PUFFER_CONTINUE;
    for (size_t index = 0; index < sizeof(required_held); index++) {
        categories[index + 1] = (RekG1PufferCategory){
            .kind = REK_G1_PUFFER_START,
            .command = {
                .kind = REK_G1_SEMANTIC_LOCOMOTION,
                .held_code = held_code(required_held[index]),
                .duration_ticks = 2,
                .kick_registry_index = REK_G1_SEMANTIC_KICK_NONE,
            },
        };
    }
    for (uint16_t kick = 0; kick < REK_G1_REQUIRED_KICK_COUNT; kick++) {
        kick_indices[kick] = (uint16_t)(6 + kick);
        kick_durations[kick] = 2;
        categories[16 + kick] = (RekG1PufferCategory){
            .kind = REK_G1_PUFFER_START,
            .command = {
                .kind = REK_G1_SEMANTIC_KICK,
                .held_code = held_code(0),
                .duration_ticks = 2,
                .kick_registry_index = kick,
            },
        };
    }
    return (RekG1PufferActionTable){
        .categories = categories,
        .count = TEST_CATEGORIES,
        .kick_move_indices = kick_indices,
        .kick_duration_ticks = kick_durations,
        .kick_registry_count = REK_G1_REQUIRED_KICK_COUNT,
    };
}

static RekG1NativePufferIO make_io(
        float* actions,
        float (*observations)[TEST_OBS_FLOATS],
        float* rewards,
        float* terminals,
        uint8_t (*masks)[TEST_CATEGORIES]) {
    return (RekG1NativePufferIO){
        .actions = actions,
        .action_rows = TEST_ENVS,
        .action_heads = REK_G1_PUFFER_ACTION_HEADS,
        .observations = observations,
        .observation_rows = TEST_ENVS,
        .observation_stride_bytes = sizeof(observations[0]),
        .rewards = rewards,
        .reward_rows = TEST_ENVS,
        .terminals = terminals,
        .terminal_rows = TEST_ENVS,
        .action_masks = &masks[0][0],
        .action_mask_rows = TEST_ENVS,
        .action_mask_stride_bytes = sizeof(masks[0]),
    };
}

static int mask_count(const uint8_t* mask) {
    int count = 0;
    for (size_t index = 0; index < TEST_CATEGORIES; index++) {
        count += mask[index] != 0;
    }
    return count;
}

static void test_static_motion_route_identity(void) {
    const RekG1NativeMotionRouteTable* table =
        rek_g1_native_static_motion_routes();
    require(rek_g1_native_validate_static_motion_routes(table),
        "static_route_table_valid");
    require(strcmp(
        table->source_probe_sha256,
        REK_G1_STATIC_ROUTE_PROBE_SHA256) == 0,
        "static_route_source_probe_pinned");
    const RekG1NativeMotionRoute* backward = rek_g1_native_route_by_id(
        table, REK_G1_NATIVE_BACKWARD);
    require(backward != NULL, "backward_route_present");
    require(backward->mocap_clip_config_path_id == 2721,
        "backward_mocap_identity");
    require(backward->npz_path_id == 370,
        "backward_reuses_walking_asset");
    require(backward->playback_speed == -1.0f,
        "backward_reverse_playback");
    require(backward->loop == 1 && backward->asset_fps == 50.0f
            && backward->blend_in_seconds == 1.0f
            && backward->blend_out_seconds == 1.0f,
        "backward_full_clip_config");
    const RekG1NativeMotionRoute* strafe_right = rek_g1_native_route_by_id(
        table, REK_G1_NATIVE_STRAFE_RIGHT);
    require(strafe_right != NULL, "strafe_right_route_present");
    require(strafe_right->mocap_clip_config_path_id == 2716,
        "strafe_right_mocap_identity");
    require(strafe_right->npz_path_id == 388
            && strafe_right->playback_speed == -1.0f,
        "strafe_right_reverse_left_asset");
    require(strafe_right->blend_in_seconds == 0.09399999678134918f
            && strafe_right->blend_out_seconds == 0.29499998688697815f
            && strafe_right->yaw_blend == 0.0f,
        "strafe_right_full_clip_config");
    const RekG1NativeMotionRoute* turn_left = rek_g1_native_route_by_id(
        table, REK_G1_NATIVE_TURN_LEFT);
    require(turn_left != NULL, "turn_left_route_present");
    require(turn_left->mocap_clip_config_path_id == 2719,
        "turn_left_mocap_identity");
    require(turn_left->npz_path_id == 375
            && turn_left->playback_speed == -1.0f,
        "turn_left_reverse_right_turn_asset");
    require(turn_left->blend_in_seconds == 0.06499999761581421f
            && turn_left->blend_out_seconds == 0.07100000232458115f
            && turn_left->yaw_blend == 1.0f,
        "turn_left_full_clip_config");
    static const uint32_t expected_frames[REK_G1_REQUIRED_KICK_COUNT] = {
        158, 146, 159, 140,
    };
    for (uint16_t offset = 0; offset < REK_G1_REQUIRED_KICK_COUNT; offset++) {
        uint16_t move_index = (uint16_t)(6 + offset);
        const RekG1NativeMotionRoute* kick = rek_g1_native_kick_route(
            table, move_index);
        require(kick != NULL, "kick_route_present");
        require(kick->runtime_move_index == move_index,
            "kick_route_move_identity");
        require(kick->asset_frames == expected_frames[offset],
            "kick_asset_frame_count_pinned");
    }

    RekG1NativeMotionRoute modified[REK_G1_STATIC_ROUTE_COUNT];
    memcpy(modified, table->routes, sizeof(modified));
    modified[REK_G1_NATIVE_FORWARD].playback_speed = -1.0f;
    RekG1NativeMotionRouteTable wrong = *table;
    wrong.routes = modified;
    require(!rek_g1_native_validate_static_motion_routes(&wrong),
        "modified_route_table_rejected");
    wrong = *table;
    wrong.source_probe_sha256 =
        "0000000000000000000000000000000000000000000000000000000000000000";
    require(!rek_g1_native_validate_static_motion_routes(&wrong),
        "modified_route_source_rejected");
}

static void test_runtime_facts_abi_gate(void) {
    RekG1PufferCategory categories[TEST_CATEGORIES];
    uint16_t kick_indices[REK_G1_REQUIRED_KICK_COUNT];
    uint32_t kick_durations[REK_G1_REQUIRED_KICK_COUNT];
    RekG1PufferActionTable table = make_table(
        categories, kick_indices, kick_durations);
    FakeRuntime runtime = {0};
    RekG1NativePufferVector vector = {0};

    RekG1NativeBatchOps ops = fake_ops();
    ops.runtime_facts_abi_version = 0;
    require(rek_g1_native_puffer_open(
        &vector, TEST_ENVS, &table, rek_g1_native_static_motion_routes(),
        ops, &runtime) == REK_G1_NATIVE_PUFFER_RUNTIME_ABI_MISMATCH,
        "legacy_zero_version_rejected");
    require(!vector.initialized && runtime.reset_calls == 0,
        "version_mismatch_does_not_open_or_call_runtime");

    ops = fake_ops();
    ops.runtime_facts_size = REK_G1_RUNTIME_FACTS_SIZE - 1u;
    require(rek_g1_native_puffer_open(
        &vector, TEST_ENVS, &table, rek_g1_native_static_motion_routes(),
        ops, &runtime) == REK_G1_NATIVE_PUFFER_RUNTIME_ABI_MISMATCH,
        "facts_size_mismatch_rejected");
    require(!vector.initialized && runtime.reset_calls == 0,
        "size_mismatch_does_not_open_or_call_runtime");
}

static void test_batch_boundary_and_held_state(void) {
    RekG1PufferCategory categories[TEST_CATEGORIES];
    uint16_t kick_indices[REK_G1_REQUIRED_KICK_COUNT];
    uint32_t kick_durations[REK_G1_REQUIRED_KICK_COUNT];
    RekG1PufferActionTable table = make_table(
        categories, kick_indices, kick_durations);
    FakeRuntime runtime = {0};
    RekG1NativePufferVector vector = {0};
    RekG1NativeBatchOps ops = fake_ops();
    require(rek_g1_native_puffer_open(
        &vector, TEST_ENVS, &table, rek_g1_native_static_motion_routes(),
        ops, &runtime) == REK_G1_NATIVE_PUFFER_OK,
        "vector_open");
    RekG1PufferAdapter* const adapters_before_reopen = vector.adapters;
    require(rek_g1_native_puffer_open(
        &vector, TEST_ENVS, &table, rek_g1_native_static_motion_routes(),
        ops, &runtime) == REK_G1_NATIVE_PUFFER_NOT_READY,
        "reopen_of_live_vector_rejected");
    require(vector.adapters == adapters_before_reopen && vector.initialized,
        "rejected_reopen_preserves_live_vector");
    require(runtime.close_calls == 0,
        "rejected_reopen_does_not_close_runtime");

    float actions[TEST_ENVS] = {0};
    float observations[TEST_ENVS][TEST_OBS_FLOATS] = {{0}};
    float rewards[TEST_ENVS] = {0};
    float terminals[TEST_ENVS] = {0};
    uint8_t masks[TEST_ENVS][TEST_CATEGORIES] = {{0}};
    RekG1NativePufferIO io = make_io(
        actions, observations, rewards, terminals, masks);
    char error[128] = {0};
    require(rek_g1_native_puffer_reset(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK,
        "vector_reset");
    require(runtime.reset_calls == 1, "one_batch_reset_call");
    require(mask_count(masks[0]) == TEST_CATEGORIES - 1,
        "reset_masks_all_start_categories");

    actions[0] = 2.0f;
    actions[1] = 4.0f;
    actions[2] = 6.0f;
    require(rek_g1_native_puffer_step(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK,
        "start_locomotion_batch");
    require(runtime.advance_calls == 1, "one_advance_for_three_environments");
    require(runtime.captured[0].input.held == REK_G1_HELD_FORWARD,
        "forward_held_reaches_batch_runtime");
    require(runtime.captured[1].input.held == REK_G1_HELD_STRAFE_LEFT,
        "strafe_held_reaches_batch_runtime");
    require(runtime.captured[2].input.held == REK_G1_HELD_YAW_LEFT,
        "yaw_held_reaches_batch_runtime");
    require(observations[0][1] == 1.0f, "forward_observation_from_backend");
    require(observations[1][2] == 1.0f, "strafe_observation_from_backend");
    require(fabsf(observations[2][3] - 0.2f) < 1e-6f,
        "yaw_ramp_first_tick_preserved");
    for (size_t index = 0; index < TEST_ENVS; index++) {
        require(mask_count(masks[index]) == 1 && masks[index][0] == 1,
            "active_segment_masks_only_continue");
    }

    memset(actions, 0, sizeof(actions));
    require(rek_g1_native_puffer_step(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK,
        "continue_locomotion_batch");
    require(runtime.advance_calls == 2, "one_advance_per_puffer_step");
    require(runtime.captured[0].input.held == REK_G1_HELD_FORWARD,
        "forward_hold_repeated_on_continue");
    require(runtime.captured[1].input.held == REK_G1_HELD_STRAFE_LEFT,
        "strafe_hold_repeated_on_continue");
    require(fabsf(runtime.captured[2].input.yaw - 0.4f) < 1e-6f,
        "yaw_ramp_advances_on_continue");
    require(masks[0][16] == 0,
        "kick_remains_masked_while_translation_state_is_held");
    require(masks[2][16] == 1,
        "kick_legal_after_yaw_only_segment");

    require(rek_g1_native_puffer_reset(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK,
        "reset_before_kick_batch");
    actions[0] = 16.0f;
    actions[1] = 17.0f;
    actions[2] = 18.0f;
    require(rek_g1_native_puffer_step(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK,
        "start_kick_batch");
    require(runtime.advance_calls == 3,
        "kick_batch_is_one_additional_advance");
    for (uint16_t index = 0; index < TEST_ENVS; index++) {
        require(runtime.captured[index].kind == REK_G1_SEMANTIC_KICK,
            "kick_kind_reaches_batch_runtime");
        require(runtime.captured[index].kick_registry_index == index,
            "kick_registry_order_reaches_batch_runtime");
        require(runtime.captured[index].kick_start_edge,
            "kick_start_edge_reaches_batch_runtime");
    }

    rek_g1_native_puffer_close(&vector);
    require(runtime.close_calls == 1, "runtime_closed_once");
}

static void test_terminal_resets_only_terminal_adapters(void) {
    RekG1PufferCategory categories[TEST_CATEGORIES];
    uint16_t kick_indices[REK_G1_REQUIRED_KICK_COUNT];
    uint32_t kick_durations[REK_G1_REQUIRED_KICK_COUNT];
    RekG1PufferActionTable table = make_table(
        categories, kick_indices, kick_durations);
    FakeRuntime runtime = {0};
    RekG1NativePufferVector vector = {0};
    RekG1NativeBatchOps ops = fake_ops();
    require(rek_g1_native_puffer_open(
        &vector, TEST_ENVS, &table, rek_g1_native_static_motion_routes(),
        ops, &runtime) == REK_G1_NATIVE_PUFFER_OK,
        "terminal_vector_open");

    float actions[TEST_ENVS] = {2.0f, 4.0f, 6.0f};
    float observations[TEST_ENVS][TEST_OBS_FLOATS] = {{0}};
    float rewards[TEST_ENVS] = {0};
    float terminals[TEST_ENVS] = {0};
    uint8_t masks[TEST_ENVS][TEST_CATEGORIES] = {{0}};
    RekG1NativePufferIO io = make_io(
        actions, observations, rewards, terminals, masks);
    char error[128] = {0};
    require(rek_g1_native_puffer_reset(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK,
        "terminal_vector_reset");

    runtime.emit_terminal_pair = 1;
    require(rek_g1_native_puffer_step(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK,
        "terminal_pair_step_succeeds");
    require(terminals[0] == 1.0f && terminals[1] == 1.0f
            && terminals[2] == 0.0f,
        "terminal_pair_preserved");
    require(rewards[0] == (float)REK_G1_SEMANTIC_LOCOMOTION
            && rewards[1] == (float)REK_G1_SEMANTIC_LOCOMOTION,
        "terminal_rewards_preserved");
    require(!vector.adapters[0].scheduler.active
            && !vector.adapters[1].scheduler.active,
        "terminal_pair_adapters_reset");
    require(vector.adapters[2].scheduler.active,
        "nonterminal_adapter_remains_active");
    require(mask_count(masks[0]) == TEST_CATEGORIES - 1
            && mask_count(masks[1]) == TEST_CATEGORIES - 1,
        "terminal_pair_masks_fresh_episode");
    require(mask_count(masks[2]) == 1 && masks[2][0] == 1u,
        "nonterminal_mask_remains_continue_only");

    actions[0] = 3.0f;
    actions[1] = 5.0f;
    actions[2] = 0.0f;
    require(rek_g1_native_puffer_step(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK,
        "post_terminal_new_actions_succeed");
    require(runtime.captured[0].input.held == REK_G1_HELD_BACKWARD,
        "terminal_row_zero_starts_fresh_action");
    require(runtime.captured[1].input.held == REK_G1_HELD_STRAFE_RIGHT,
        "terminal_row_one_starts_fresh_action");
    require(runtime.captured[2].input.held == REK_G1_HELD_YAW_LEFT,
        "nonterminal_row_continues_prior_action");
    require(fabsf(runtime.captured[2].input.yaw - 0.4f) < 1e-6f,
        "nonterminal_row_yaw_state_is_uninterrupted");
    require(terminals[0] == 0.0f && terminals[1] == 0.0f
            && terminals[2] == 0.0f,
        "terminal_pair_emitted_exactly_once");

    rek_g1_native_puffer_close(&vector);
    require(runtime.close_calls == 1, "terminal_runtime_closed_once");
}

static void test_input_reset_resets_only_signaled_adapter(void) {
    RekG1PufferCategory categories[TEST_CATEGORIES];
    uint16_t kick_indices[REK_G1_REQUIRED_KICK_COUNT];
    uint32_t kick_durations[REK_G1_REQUIRED_KICK_COUNT];
    RekG1PufferActionTable table = make_table(
        categories, kick_indices, kick_durations);
    FakeRuntime runtime = {0};
    RekG1NativePufferVector vector = {0};
    RekG1NativeBatchOps ops = fake_ops();
    require(rek_g1_native_puffer_open(
        &vector, TEST_ENVS, &table, rek_g1_native_static_motion_routes(),
        ops, &runtime) == REK_G1_NATIVE_PUFFER_OK,
        "input_reset_vector_open");

    float actions[TEST_ENVS] = {2.0f, 4.0f, 6.0f};
    float observations[TEST_ENVS][TEST_OBS_FLOATS] = {{0}};
    float rewards[TEST_ENVS] = {0};
    float terminals[TEST_ENVS] = {0};
    uint8_t masks[TEST_ENVS][TEST_CATEGORIES] = {{0}};
    RekG1NativePufferIO io = make_io(
        actions, observations, rewards, terminals, masks);
    char error[128] = {0};
    require(rek_g1_native_puffer_reset(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK,
        "input_reset_vector_reset");

    runtime.emit_input_reset_mask = 1u << 1;
    require(rek_g1_native_puffer_step(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK,
        "input_reset_step_succeeds");
    require(terminals[0] == 0.0f && terminals[1] == 0.0f
            && terminals[2] == 0.0f,
        "input_reset_does_not_fabricate_terminal");
    require(vector.adapters[0].scheduler.active
            && !vector.adapters[1].scheduler.active
            && vector.adapters[2].scheduler.active,
        "input_reset_is_row_local");
    require(mask_count(masks[0]) == 1 && masks[0][0] == 1u
            && mask_count(masks[1]) == TEST_CATEGORIES - 1
            && mask_count(masks[2]) == 1 && masks[2][0] == 1u,
        "input_reset_writes_fresh_row_mask_only");

    actions[0] = 0.0f;
    actions[1] = 3.0f;
    actions[2] = 0.0f;
    require(rek_g1_native_puffer_step(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK,
        "post_input_reset_new_action_succeeds");
    require(runtime.captured[1].input.held == REK_G1_HELD_BACKWARD,
        "input_reset_row_starts_fresh_action");
    require(runtime.captured[0].input.held == REK_G1_HELD_FORWARD
            && runtime.captured[2].input.held == REK_G1_HELD_YAW_LEFT,
        "input_reset_does_not_change_other_held_rows");

    rek_g1_native_puffer_close(&vector);
    require(runtime.close_calls == 1, "input_reset_runtime_closed_once");
}

static void test_fail_closed_and_reset_recovery(void) {
    RekG1PufferCategory categories[TEST_CATEGORIES];
    uint16_t kick_indices[REK_G1_REQUIRED_KICK_COUNT];
    uint32_t kick_durations[REK_G1_REQUIRED_KICK_COUNT];
    RekG1PufferActionTable table = make_table(
        categories, kick_indices, kick_durations);
    FakeRuntime runtime = {0};
    RekG1NativePufferVector vector = {0};
    RekG1NativeBatchOps ops = fake_ops();
    require(rek_g1_native_puffer_open(
        &vector, TEST_ENVS, &table, rek_g1_native_static_motion_routes(),
        ops, &runtime) == REK_G1_NATIVE_PUFFER_OK,
        "failure_vector_open");

    float actions[TEST_ENVS] = {1.0f, 1.5f, 1.0f};
    float observations[TEST_ENVS][TEST_OBS_FLOATS] = {{0}};
    float rewards[TEST_ENVS] = {0};
    float terminals[TEST_ENVS] = {0};
    uint8_t masks[TEST_ENVS][TEST_CATEGORIES] = {{0}};
    RekG1NativePufferIO io = make_io(
        actions, observations, rewards, terminals, masks);
    char error[128] = {0};
    require(rek_g1_native_puffer_reset(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK,
        "failure_vector_reset");
    require(rek_g1_native_puffer_step(
        &vector, io, error, sizeof(error)) ==
            REK_G1_NATIVE_PUFFER_ACTION_REJECTED,
        "fractional_action_rejected");
    require(runtime.advance_calls == 0,
        "invalid_row_prevents_partial_batch_advance");
    require(vector.failed && !vector.ready, "invalid_action_poisoned_vector");
    for (size_t index = 0; index < TEST_ENVS; index++) {
        require(mask_count(masks[index]) == 0, "failure_clears_all_masks");
    }
    require(rek_g1_native_puffer_step(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_NOT_READY,
        "poisoned_vector_cannot_retry");

    memset(actions, 0, sizeof(actions));
    require(rek_g1_native_puffer_reset(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK,
        "full_reset_recovers_poisoned_vector");
    actions[0] = 1.0f;
    actions[1] = 1.0f;
    actions[2] = 1.0f;
    runtime.fail_advance = 1;
    require(rek_g1_native_puffer_step(
        &vector, io, error, sizeof(error)) ==
            REK_G1_NATIVE_PUFFER_RUNTIME_ADVANCE_FAILED,
        "batch_backend_failure_reported");
    require(runtime.advance_calls == 1, "failed_backend_called_once");
    require(strcmp(error, "injected advance failure") == 0,
        "backend_error_preserved");
    for (size_t index = 0; index < TEST_ENVS; index++) {
        require(mask_count(masks[index]) == 0,
            "backend_failure_clears_all_masks");
    }

    runtime.fail_advance = 0;
    runtime.emit_invalid_facts = 1;
    require(rek_g1_native_puffer_reset(
        &vector, io, error, sizeof(error)) ==
            REK_G1_NATIVE_PUFFER_RUNTIME_FACTS_INVALID,
        "invalid_reset_facts_rejected");
    require(vector.failed && !vector.ready,
        "invalid_reset_facts_poison_vector");
    rek_g1_native_puffer_close(&vector);
}

static void test_io_shape_is_exact(void) {
    RekG1PufferCategory categories[TEST_CATEGORIES];
    uint16_t kick_indices[REK_G1_REQUIRED_KICK_COUNT];
    uint32_t kick_durations[REK_G1_REQUIRED_KICK_COUNT];
    RekG1PufferActionTable table = make_table(
        categories, kick_indices, kick_durations);
    FakeRuntime runtime = {0};
    RekG1NativePufferVector vector = {0};
    RekG1NativeBatchOps ops = fake_ops();
    require(rek_g1_native_puffer_open(
        &vector, TEST_ENVS, &table, rek_g1_native_static_motion_routes(),
        ops, &runtime) == REK_G1_NATIVE_PUFFER_OK,
        "shape_vector_open");
    float actions[TEST_ENVS] = {0};
    float observations[TEST_ENVS][TEST_OBS_FLOATS] = {{0}};
    float rewards[TEST_ENVS] = {0};
    float terminals[TEST_ENVS] = {0};
    uint8_t masks[TEST_ENVS][TEST_CATEGORIES] = {{0}};
    RekG1NativePufferIO io = make_io(
        actions, observations, rewards, terminals, masks);
    io.observation_rows -= 1;
    require(rek_g1_native_puffer_reset(
        &vector, io, NULL, 0) == REK_G1_NATIVE_PUFFER_SIZE_INVALID,
        "reset_rejects_observation_row_mismatch");
    require(runtime.reset_calls == 0, "shape_error_does_not_call_backend");
    io.observation_rows += 1;
    io.reward_rows -= 1;
    require(rek_g1_native_puffer_reset(
        &vector, io, NULL, 0) == REK_G1_NATIVE_PUFFER_SIZE_INVALID,
        "reset_rejects_reward_row_mismatch");
    require(runtime.reset_calls == 0,
        "reward_shape_error_does_not_call_backend");
    rek_g1_native_puffer_close(&vector);
}

static void test_runtime_output_validation(void) {
    RekG1PufferCategory categories[TEST_CATEGORIES];
    uint16_t kick_indices[REK_G1_REQUIRED_KICK_COUNT];
    uint32_t kick_durations[REK_G1_REQUIRED_KICK_COUNT];
    RekG1PufferActionTable table = make_table(
        categories, kick_indices, kick_durations);
    FakeRuntime runtime = {0};
    RekG1NativePufferVector vector = {0};
    RekG1NativeBatchOps ops = fake_ops();
    require(rek_g1_native_puffer_open(
        &vector, TEST_ENVS, &table, rek_g1_native_static_motion_routes(),
        ops, &runtime) == REK_G1_NATIVE_PUFFER_OK,
        "output_validation_vector_open");

    float actions[TEST_ENVS] = {1.0f, 1.0f, 1.0f};
    float observations[TEST_ENVS][TEST_OBS_FLOATS] = {{0}};
    float rewards[TEST_ENVS] = {0};
    float terminals[TEST_ENVS] = {0};
    uint8_t masks[TEST_ENVS][TEST_CATEGORIES] = {{0}};
    RekG1NativePufferIO io = make_io(
        actions, observations, rewards, terminals, masks);
    require(rek_g1_native_puffer_reset(&vector, io, NULL, 0)
            == REK_G1_NATIVE_PUFFER_OK,
        "output_validation_initial_reset");

    runtime.emit_invalid_reward = 1;
    require(rek_g1_native_puffer_step(&vector, io, NULL, 0)
            == REK_G1_NATIVE_PUFFER_RUNTIME_OUTPUT_INVALID,
        "nonfinite_reward_rejected_before_publication");
    require(vector.failed && !vector.ready,
        "invalid_reward_poisoned_vector");

    runtime.emit_invalid_reward = 0;
    require(rek_g1_native_puffer_reset(&vector, io, NULL, 0)
            == REK_G1_NATIVE_PUFFER_OK,
        "reset_after_invalid_reward");
    runtime.emit_invalid_terminal = 1;
    require(rek_g1_native_puffer_step(&vector, io, NULL, 0)
            == REK_G1_NATIVE_PUFFER_RUNTIME_OUTPUT_INVALID,
        "nonbinary_terminal_rejected_before_adapter_reset");
    require(vector.failed && !vector.ready,
        "invalid_terminal_poisoned_vector");

    runtime.emit_invalid_terminal = 0;
    runtime.emit_invalid_reward = 1;
    require(rek_g1_native_puffer_reset(&vector, io, NULL, 0)
            == REK_G1_NATIVE_PUFFER_RUNTIME_OUTPUT_INVALID,
        "nonfinite_reset_reward_rejected");
    rek_g1_native_puffer_close(&vector);
}

int main(void) {
    test_static_motion_route_identity();
    test_runtime_facts_abi_gate();
    test_batch_boundary_and_held_state();
    test_terminal_resets_only_terminal_adapters();
    test_input_reset_resets_only_signaled_adapter();
    test_fail_closed_and_reset_recovery();
    test_io_shape_is_exact();
    test_runtime_output_validation();
    printf("native puffer vector tests passed: %d assertions\n", assertions);
    return 0;
}
