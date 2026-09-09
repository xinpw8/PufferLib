#include "semantic_duel_runtime.h"
#include "g1_semantic_action_table.h"
#include "sonic_motion_composer_libm_candidate.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct TestClipStorage {
    int32_t npz_path_id;
    size_t frames;
    float* dof;
    float* root_position;
    float* root_wxyz;
    float* root_xyzw;
} TestClipStorage;

typedef struct TestProbeContext {
    int fail_forgiveness;
    size_t matcher_calls;
} TestProbeContext;

enum {
    TEST_UNIQUE_CLIP_COUNT = 21,
};

static int checks = 0;

#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        fprintf(stderr, "check failed at %s:%d: %s\n", \
            __FILE__, __LINE__, #condition); \
        return 0; \
    } \
} while (0)

static float* read_f32(const char* path, size_t count) {
    if (path == NULL || count == 0 || count > SIZE_MAX / sizeof(float)) {
        return NULL;
    }
    FILE* stream = fopen(path, "rb");
    if (stream == NULL) return NULL;
    float* values = (float*)malloc(count * sizeof(float));
    if (values == NULL) {
        fclose(stream);
        return NULL;
    }
    const size_t read_count = fread(values, sizeof(float), count, stream);
    const int trailing = fgetc(stream);
    const int close_status = fclose(stream);
    if (read_count != count || trailing != EOF || close_status != 0) {
        free(values);
        return NULL;
    }
    return values;
}

static int make_path(
        char* output,
        size_t capacity,
        const char* asset_dir,
        int32_t npz_path_id,
        const char* suffix) {
    if (output == NULL || capacity == 0 || asset_dir == NULL || suffix == NULL) {
        return 0;
    }
    const int length = snprintf(
        output, capacity, "%s/motion_%d_%s.f32le",
        asset_dir, npz_path_id, suffix);
    return length >= 0 && (size_t)length < capacity;
}

static TestClipStorage* storage_by_npz(
        TestClipStorage* storage,
        size_t count,
        int32_t npz_path_id) {
    for (size_t index = 0; index < count; index++) {
        if (storage[index].npz_path_id == npz_path_id) return &storage[index];
    }
    return NULL;
}

static void free_storage(TestClipStorage* storage, size_t count) {
    if (storage == NULL) return;
    for (size_t index = 0; index < count; index++) {
        free(storage[index].root_xyzw);
        free(storage[index].root_wxyz);
        free(storage[index].root_position);
        free(storage[index].dof);
    }
}

static int load_assets(
        const char* asset_dir,
        const RekG1NativeMotionRouteTable* routes,
        TestClipStorage storage[TEST_UNIQUE_CLIP_COUNT],
        RekG1SemanticDuelRouteAsset assets[REK_G1_STATIC_ROUTE_COUNT]) {
    char path[1024];
    size_t storage_count = 0u;
    for (size_t route_index = 0u; route_index < routes->count; route_index++) {
        const RekG1NativeMotionRoute* route = &routes->routes[route_index];
        if (storage_by_npz(storage, storage_count, route->npz_path_id) != NULL) {
            continue;
        }
        if (storage_count >= TEST_UNIQUE_CLIP_COUNT) return 0;
        TestClipStorage* clip = &storage[storage_count++];
        clip->npz_path_id = route->npz_path_id;
        clip->frames = route->asset_frames;
        if (!make_path(path, sizeof(path), asset_dir, clip->npz_path_id,
                "dof_position")) return 0;
        clip->dof = read_f32(
            path, clip->frames * GEAR_SONIC_ACTION_DIM);
        if (!make_path(path, sizeof(path), asset_dir, clip->npz_path_id,
                "root_position")) return 0;
        clip->root_position = read_f32(path, clip->frames * 3u);
        if (!make_path(path, sizeof(path), asset_dir, clip->npz_path_id,
                "root_rotation_wxyz")) return 0;
        clip->root_wxyz = read_f32(path, clip->frames * 4u);
        if (!make_path(path, sizeof(path), asset_dir, clip->npz_path_id,
                "root_rotation_xyzw")) return 0;
        clip->root_xyzw = read_f32(path, clip->frames * 4u);
        if (clip->dof == NULL || clip->root_position == NULL
                || clip->root_wxyz == NULL || clip->root_xyzw == NULL) {
            return 0;
        }
    }
    if (storage_count != TEST_UNIQUE_CLIP_COUNT) return 0;
    for (size_t index = 0; index < routes->count; index++) {
        const RekG1NativeMotionRoute* route = &routes->routes[index];
        TestClipStorage* clip = storage_by_npz(
            storage, storage_count, route->npz_path_id);
        if (clip == NULL || clip->frames != route->asset_frames) return 0;
        assets[index] = (RekG1SemanticDuelRouteAsset){
            .route_id = route->id,
            .clip = {
                .dof_position_mujoco = clip->dof,
                .root_quaternion_wxyz = clip->root_wxyz,
                .dof_position_count = clip->frames * GEAR_SONIC_ACTION_DIM,
                .root_quaternion_count = clip->frames * 4u,
                .frame_count = clip->frames,
                .fps = route->asset_fps,
            },
            /* Test fixture only. This value is not a REK timing measurement. */
            .configured_compositor_duration_ticks =
                route->kind == REK_G1_NATIVE_ROUTE_DISCRETE_MOVE
                    ? (uint32_t)ceilf(
                        (float)(clip->frames - 1u) /
                        fabsf(route->playback_speed))
                    : 0u,
        };
    }
    return 1;
}

static int test_loop_matcher(
        void* context,
        const SonicMotionComposerNativeLayer* target,
        const SonicMotionComposerNativeLayer* outgoing,
        float* matched_cursor) {
    TestProbeContext* probe = (TestProbeContext*)context;
    if (probe == NULL || target == NULL || outgoing == NULL
            || matched_cursor == NULL || !target->active || !outgoing->active) {
        return 0;
    }
    probe->matcher_calls++;
    *matched_cursor = target->per_tick < 0.0f
        ? (float)target->end_frame : (float)target->start_frame;
    return 1;
}

static int test_forgiveness(
        void* context,
        const GearSonicNativeDuelVector* duel,
        size_t robot_row,
        float* delta_radians_out) {
    TestProbeContext* probe = (TestProbeContext*)context;
    if (probe == NULL || duel == NULL || delta_radians_out == NULL
            || robot_row >= duel->robot_count
            || probe->fail_forgiveness == (int)robot_row + 1) {
        return 0;
    }
    *delta_radians_out = 0.0f;
    return 1;
}

static int make_action_table(
        RekG1PufferCategory categories[REK_G1_SEMANTIC_ACTION_COUNT],
        uint16_t move_indices[REK_G1_REQUIRED_DISCRETE_MOVE_COUNT],
        uint32_t move_durations[REK_G1_REQUIRED_DISCRETE_MOVE_COUNT],
        const RekG1SemanticDuelRouteAsset assets[REK_G1_STATIC_ROUTE_COUNT],
        RekG1PufferActionTable* table) {
    static const uint8_t held_masks[15] = {
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
    static const uint16_t registry_order[
            REK_G1_REQUIRED_DISCRETE_MOVE_COUNT] = {
        6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16,
    };
    memset(
        categories, 0, REK_G1_SEMANTIC_ACTION_COUNT * sizeof(*categories));
    categories[0].kind = REK_G1_PUFFER_CONTINUE;
    for (size_t index = 0; index < 15; index++) {
        uint8_t held_code = 0;
        if (rek_g1_semantic_encode_held(
                held_masks[index], &held_code) != REK_G1_SEMANTIC_OK) {
            return 0;
        }
        categories[index + 1u] = (RekG1PufferCategory){
            .kind = REK_G1_PUFFER_START,
            .command = {
                .kind = REK_G1_SEMANTIC_LOCOMOTION,
                .held_code = held_code,
                .duration_ticks = 3u,
                .move_registry_index = REK_G1_SEMANTIC_MOVE_NONE,
            },
        };
    }
    for (size_t index = 0;
            index < REK_G1_REQUIRED_DISCRETE_MOVE_COUNT; index++) {
        const uint16_t move_index = registry_order[index];
        const RekG1NativeMotionRoute* route =
            rek_g1_native_discrete_move_route(
                rek_g1_native_static_motion_routes(), move_index);
        if (route == NULL) return 0;
        move_indices[index] = move_index;
        move_durations[index] =
            assets[(size_t)route->id].configured_compositor_duration_ticks;
        categories[16u + index] = (RekG1PufferCategory){
            .kind = REK_G1_PUFFER_START,
            .command = {
                .kind = REK_G1_SEMANTIC_DISCRETE_MOVE,
                .held_code = 0u,
                .duration_ticks = move_durations[index],
                .move_registry_index = (uint16_t)index,
            },
        };
    }
    *table = (RekG1PufferActionTable){
        .categories = categories,
        .count = REK_G1_SEMANTIC_ACTION_COUNT,
        .move_indices = move_indices,
        .move_duration_ticks = move_durations,
        .move_registry_count = REK_G1_REQUIRED_DISCRETE_MOVE_COUNT,
    };
    return rek_g1_puffer_validate_table(table) == REK_G1_PUFFER_OK;
}

static RekG1SemanticTick locomotion_tick(uint8_t held) {
    RekG1HeldInputState state = {0};
    return (RekG1SemanticTick){
        .status = REK_G1_SEMANTIC_OK,
        .input = rek_g1_apply_input_frame(
            &state,
            (RekG1InputFrame){.held = held},
            (RekG1InputTiming){.elapsed_seconds = 0.02f,
                               .yaw_ramp_seconds = 0.5f},
            1,
            0),
        .kind = REK_G1_SEMANTIC_LOCOMOTION,
        .move_registry_index = REK_G1_SEMANTIC_MOVE_NONE,
        .command_started = 1u,
    };
}

static RekG1SemanticTick move_tick(
        uint16_t registry_index,
        int first,
        int complete,
        uint32_t remaining_ticks) {
    return (RekG1SemanticTick){
        .status = REK_G1_SEMANTIC_OK,
        .input = {
            .status = REK_G1_INPUT_ACCEPTED,
            .attack_gate = first
                ? REK_G1_ATTACK_ACCEPTED_PREEMPT_YAW
                : REK_G1_ATTACK_NOT_REQUESTED,
        },
        .kind = REK_G1_SEMANTIC_DISCRETE_MOVE,
        .move_registry_index = registry_index,
        .remaining_ticks = remaining_ticks,
        .command_started = (uint8_t)first,
        .segment_complete = (uint8_t)complete,
        .move_start_edge = (uint8_t)first,
        .move_active = 1u,
    };
}

static int all_finite_observation(const RekG1SemanticDuelObservation* value) {
    const float* fields = (const float*)value;
    const size_t count = sizeof(*value) / sizeof(float);
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(fields[index])) return 0;
    }
    return 1;
}

static float max_abs_float_difference(
        const float* left,
        const float* right,
        size_t count) {
    float maximum = 0.0f;
    for (size_t index = 0; index < count; index++) {
        const float difference = fabsf(left[index] - right[index]);
        if (difference > maximum) maximum = difference;
    }
    return maximum;
}

static double max_abs_double_difference(
        const double* left,
        const double* right,
        size_t count) {
    double maximum = 0.0;
    for (size_t index = 0; index < count; index++) {
        const double difference = fabs(left[index] - right[index]);
        if (difference > maximum) maximum = difference;
    }
    return maximum;
}

static void quaternion_multiply(
        const double left[4],
        const double right[4],
        double output[4]) {
    output[0] = left[0] * right[0] - left[1] * right[1]
        - left[2] * right[2] - left[3] * right[3];
    output[1] = left[0] * right[1] + left[1] * right[0]
        + left[2] * right[3] - left[3] * right[2];
    output[2] = left[0] * right[2] - left[1] * right[3]
        + left[2] * right[0] + left[3] * right[1];
    output[3] = left[0] * right[3] + left[1] * right[2]
        - left[2] * right[1] + left[3] * right[0];
}

static int run_test(
        const char* asset_dir,
        const char* encoder,
        const char* decoder,
        const char* multi_encoder,
        const char* multi_decoder) {
    CHECK(REK_G1_SEMANTIC_DUEL_SCHEMA_VERSION == 4u);
    CHECK(REK_G1_SEMANTIC_DUEL_FALL_OBSERVATION_FLOATS == 15);
    CHECK(REK_G1_SEMANTIC_DUEL_ENTITY_OBSERVATION_FLOATS == 86);
    CHECK(REK_G1_SEMANTIC_DUEL_FIGHT_OBSERVATION_FLOATS == 39);
    CHECK(REK_G1_SEMANTIC_DUEL_OBSERVATION_FLOATS == 223);
    CHECK(sizeof(RekG1SemanticDuelObservation)
        == REK_G1_SEMANTIC_DUEL_OBSERVATION_FLOATS * sizeof(float));
    const RekG1NativeMotionRouteTable* routes =
        rek_g1_native_static_motion_routes();
    CHECK(rek_g1_native_validate_static_motion_routes(routes));
    CHECK(rek_g1_validate_strike_catalog(
        rek_g1_current_build_strike_catalog()));

    TestClipStorage storage[TEST_UNIQUE_CLIP_COUNT] = {0};
    RekG1SemanticDuelRouteAsset assets[REK_G1_STATIC_ROUTE_COUNT] = {0};
    CHECK(load_assets(asset_dir, routes, storage, assets));
    TestClipStorage* idle = storage_by_npz(
        storage, TEST_UNIQUE_CLIP_COUNT, 377);
    CHECK(idle != NULL);

    char model_path[1024];
    const int model_length = snprintf(
        model_path, sizeof(model_path), "%s/model.two_fighter_arena.xml",
        asset_dir);
    CHECK(model_length > 0 && (size_t)model_length < sizeof(model_path));
    GearSonicNativeMotion fixed_idle = {
        .dof_position_mujoco = idle->dof,
        .root_position_m = idle->root_position,
        .root_rotation_xyzw = idle->root_xyzw,
        .frames = idle->frames,
        .loop = 1,
    };
    char error[1024] = {0};
    GearSonicNativeDuelVector duel = {0};
    CHECK(gear_sonic_native_duel_open(
        &duel, model_path, encoder, decoder, fixed_idle,
        1u, 1, error, sizeof(error)));

    uint32_t mirror_source[GEAR_SONIC_ACTION_DIM];
    uint8_t mirror_negate[GEAR_SONIC_ACTION_DIM] = {0};
    for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
        mirror_source[index] = (uint32_t)index;
    }
    TestProbeContext probe = {0};
    RekG1SemanticDuelConfig config = {
        .input_timing = {.elapsed_seconds = 0.02f,
                         .yaw_ramp_seconds = 0.5f},
        .command = {
            .locomotion_speed_scale = 1.0f,
            .command_yaw_rate_scale = 1.0f,
            .heading_yaw_rate_scale = 1.0f,
            .controller_rate_hz = 50u,
        },
        .locomotion = {
            .settle_linear_speed = 0.03f,
            .settle_yaw_rate = 0.03f,
            .stop_brake_rate = 2.0f,
            .transition_settle = 1u,
        },
        .composer_backends = {
            .quaternion_slerp =
                sonic_motion_composer_libm_candidate_quaternion_slerp,
            .atan2_f = sonic_motion_composer_libm_candidate_atan2_f,
            .sin_cos_f = sonic_motion_composer_libm_candidate_sin_cos_f,
            .loop_entry_matcher = test_loop_matcher,
            .context = &probe,
        },
        .mirror_table = {
            .source_indices = mirror_source,
            .negate = mirror_negate,
            .source_index_count = GEAR_SONIC_ACTION_DIM,
            .negate_count = GEAR_SONIC_ACTION_DIM,
        },
        .forgiveness_delta = test_forgiveness,
        .forgiveness_context = &probe,
    };
    RekG1SemanticDuelRuntime runtime = {0};

    /* Reject every corrupt public reset-state encoding at the open gate. */
    {
        RekG1SemanticDuelRuntime rejected = {0};

        duel.dampened_rows[0] = 2u;
        CHECK(rek_g1_semantic_duel_open(
            &rejected, &duel, routes, assets, REK_G1_STATIC_ROUTE_COUNT,
            &config, error, sizeof(error)) == REK_G1_SEMANTIC_DUEL_INVALID_DUEL);
        duel.dampened_rows[0] = 0u;

        duel.resetting_rows[0] = 2u;
        CHECK(rek_g1_semantic_duel_open(
            &rejected, &duel, routes, assets, REK_G1_STATIC_ROUTE_COUNT,
            &config, error, sizeof(error)) == REK_G1_SEMANTIC_DUEL_INVALID_DUEL);
        duel.resetting_rows[0] = 0u;

        duel.reset_pending_arenas[0] = 2u;
        CHECK(rek_g1_semantic_duel_open(
            &rejected, &duel, routes, assets, REK_G1_STATIC_ROUTE_COUNT,
            &config, error, sizeof(error)) == REK_G1_SEMANTIC_DUEL_INVALID_DUEL);
        duel.reset_pending_arenas[0] = 0u;

        duel.reset_completed_in_step_arenas[0] = 2u;
        CHECK(rek_g1_semantic_duel_open(
            &rejected, &duel, routes, assets, REK_G1_STATIC_ROUTE_COUNT,
            &config, error, sizeof(error)) == REK_G1_SEMANTIC_DUEL_INVALID_DUEL);
        duel.reset_completed_in_step_arenas[0] = 0u;

        duel.reset_pending_arenas[0] = 1u;
        duel.resetting_rows[0] = 1u;
        duel.reset_complete_not_before_time[0] = 0.002;
        CHECK(rek_g1_semantic_duel_open(
            &rejected, &duel, routes, assets, REK_G1_STATIC_ROUTE_COUNT,
            &config, error, sizeof(error)) == REK_G1_SEMANTIC_DUEL_INVALID_DUEL);
        duel.resetting_rows[1] = 1u;

        duel.reset_completed_in_step_arenas[0] = 1u;
        CHECK(rek_g1_semantic_duel_open(
            &rejected, &duel, routes, assets, REK_G1_STATIC_ROUTE_COUNT,
            &config, error, sizeof(error)) == REK_G1_SEMANTIC_DUEL_INVALID_DUEL);
        duel.reset_completed_in_step_arenas[0] = 0u;

        duel.reset_complete_not_before_time[0] = NAN;
        CHECK(rek_g1_semantic_duel_open(
            &rejected, &duel, routes, assets, REK_G1_STATIC_ROUTE_COUNT,
            &config, error, sizeof(error)) == REK_G1_SEMANTIC_DUEL_INVALID_DUEL);
        duel.reset_complete_not_before_time[0] = -0.002;
        CHECK(rek_g1_semantic_duel_open(
            &rejected, &duel, routes, assets, REK_G1_STATIC_ROUTE_COUNT,
            &config, error, sizeof(error)) == REK_G1_SEMANTIC_DUEL_INVALID_DUEL);

        duel.reset_pending_arenas[0] = 0u;
        duel.resetting_rows[0] = 0u;
        duel.resetting_rows[1] = 0u;
        duel.reset_complete_not_before_time[0] = 0.002;
        CHECK(rek_g1_semantic_duel_open(
            &rejected, &duel, routes, assets, REK_G1_STATIC_ROUTE_COUNT,
            &config, error, sizeof(error)) == REK_G1_SEMANTIC_DUEL_INVALID_DUEL);
        duel.reset_complete_not_before_time[0] = 0.0;

        duel.resetting_rows[0] = 1u;
        CHECK(rek_g1_semantic_duel_open(
            &rejected, &duel, routes, assets, REK_G1_STATIC_ROUTE_COUNT,
            &config, error, sizeof(error)) == REK_G1_SEMANTIC_DUEL_INVALID_DUEL);
        duel.resetting_rows[0] = 0u;

        /* A completed marker is valid until the following outer step. */
        duel.reset_completed_in_step_arenas[0] = 1u;
        CHECK(rek_g1_semantic_duel_open(
            &rejected, &duel, routes, assets, REK_G1_STATIC_ROUTE_COUNT,
            &config, error, sizeof(error)) == REK_G1_SEMANTIC_DUEL_OK);
        rek_g1_semantic_duel_close(&rejected);
        duel.reset_completed_in_step_arenas[0] = 0u;
    }

    CHECK(rek_g1_semantic_duel_open(
        &runtime, &duel, routes, assets, REK_G1_STATIC_ROUTE_COUNT,
        &config, error, sizeof(error)) == REK_G1_SEMANTIC_DUEL_OK);
    RekG1SemanticDuelEntityObservation* const observations_before_reopen =
        runtime.entity_observations;
    CHECK(rek_g1_semantic_duel_open(
        &runtime, &duel, routes, assets, REK_G1_STATIC_ROUTE_COUNT,
        &config, error, sizeof(error)) == REK_G1_SEMANTIC_DUEL_NOT_READY);
    CHECK(runtime.initialized && !runtime.failed);
    CHECK(runtime.entity_observations == observations_before_reopen);

    RekG1PufferCategory categories[REK_G1_SEMANTIC_ACTION_COUNT];
    uint16_t move_indices[REK_G1_REQUIRED_DISCRETE_MOVE_COUNT];
    uint32_t move_durations[REK_G1_REQUIRED_DISCRETE_MOVE_COUNT];
    RekG1PufferActionTable table = {0};
    CHECK(make_action_table(
        categories, move_indices, move_durations, assets, &table));

    RekG1RuntimeFacts facts[2] = {0};
    RekG1SemanticDuelObservation observations[2] = {0};
    float rewards[2] = {-1.0f, -1.0f};
    float terminals[2] = {-1.0f, -1.0f};
    CHECK(rek_g1_semantic_duel_reset_batch(
        &runtime, routes, &table, 2u, facts,
        observations, sizeof(observations[0]), rewards, terminals,
        error, sizeof(error)));
    CHECK(runtime.ready && !runtime.failed);
    CHECK(facts[0].translation_transition_settled
        && facts[1].translation_transition_settled);
    CHECK(rewards[0] == 0.0f && rewards[1] == 0.0f);
    CHECK(terminals[0] == 0.0f && terminals[1] == 0.0f);
    CHECK(all_finite_observation(&observations[0]));
    CHECK(all_finite_observation(&observations[1]));
    CHECK(runtime.fall_adapter.ready == 1u);
    CHECK(runtime.fall_states[0].phase == REK_G1_FALL_UPRIGHT);
    CHECK(runtime.fall_states[1].phase == REK_G1_FALL_UPRIGHT);
    CHECK(runtime.fall_states[0].reset_grace_remaining_seconds == 0.0f);
    CHECK(runtime.fall_states[1].reset_grace_remaining_seconds == 0.0f);
    CHECK(runtime.fall_events[0] == REK_G1_FALL_EVENT_NONE);
    CHECK(runtime.fall_events[1] == REK_G1_FALL_EVENT_NONE);
    CHECK(observations[0].self.fall.tracking_active == 1.0f);
    CHECK(observations[0].opponent.fall.tracking_active == 1.0f);
    CHECK(fabsf(observations[0].self.fall.tilt_degrees) < 1e-4f);
    CHECK(fabsf(observations[0].opponent.fall.tilt_degrees) < 1e-4f);
    CHECK(fabsf(observations[0].self.fall.pelvis_height_ratio - 1.0f)
        < 1e-6f);
    CHECK(fabsf(observations[0].opponent.fall.pelvis_height_ratio - 1.0f)
        < 1e-6f);
    CHECK(observations[0].self.fall.build_pinned_can_get_up == 0.0f);
    CHECK(observations[0].self.fall.phase
        == (float)REK_G1_FALL_UPRIGHT);
    CHECK(observations[0].self.fall.fallen_timer_seconds
        == REK_G1_FALL_CONFIG_F84F1874.fallen_reset_timeout_seconds);
    CHECK(runtime.hit_adapter.ready == 1u);
    CHECK(runtime.combat_states[0].initialized == 1u);
    CHECK(runtime.combat_states[0].fight.phase
        == REK_G1_FIGHT_ROUND_ACTIVE);
    CHECK(observations[0].fight.phase
        == (float)REK_G1_FIGHT_ROUND_ACTIVE);
    CHECK(observations[0].fight.current_round_number == 1.0f);
    CHECK(observations[0].fight.time_remaining_seconds == 120.0f);
    CHECK(observations[0].fight.self_fighter_index == 0.0f);
    CHECK(observations[1].fight.self_fighter_index == 1.0f);
    CHECK(memcmp(
        &observations[0].self,
        &observations[1].opponent,
        sizeof(observations[0].self)) == 0);
    CHECK(memcmp(
        &observations[0].opponent,
        &observations[1].self,
        sizeof(observations[0].opponent)) == 0);

    /* Every registry entry must start its exact pinned native route. */
    for (uint16_t registry_index = 0u;
            registry_index < table.move_registry_count; registry_index++) {
        const uint16_t runtime_move_index =
            table.move_indices[registry_index];
        const RekG1NativeMotionRoute* expected_route =
            rek_g1_native_discrete_move_route(routes, runtime_move_index);
        CHECK(expected_route != NULL);
        const RekG1SemanticTick move_semantics[2] = {
            move_tick(
                registry_index,
                1,
                0,
                move_durations[registry_index] - 1u),
            locomotion_tick(0u),
        };
        CHECK(rek_g1_semantic_duel_advance_batch(
            &runtime, routes, &table, move_semantics, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
        CHECK(runtime.active_route_ids[0] == expected_route->id);
        CHECK(runtime.composers[0].action_playing);
        CHECK(runtime.composers[0].current_layer.clip.frame_count
            == expected_route->asset_frames);
        CHECK(rek_g1_semantic_duel_reset_batch(
            &runtime, routes, &table, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
    }

    /*
     * The attack gate is translation-specific. A held Q or E turn remains
         * preemptible by a discrete move without an intervening neutral tick.
         * A combined
     * W+Q command still carries translation and must keep the gate closed.
     */
    {
        const RekG1SemanticTick yaw_semantics[2] = {
            locomotion_tick(REK_G1_HELD_YAW_LEFT),
            locomotion_tick(REK_G1_HELD_YAW_RIGHT),
        };
        CHECK(rek_g1_semantic_duel_advance_batch(
            &runtime, routes, &table, yaw_semantics, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
        CHECK(runtime.locomotion_states[0].locomotion_active
            && runtime.locomotion_states[1].locomotion_active);
        CHECK(runtime.locomotion_states[0].current_route_id
            == REK_G1_NATIVE_TURN_LEFT);
        CHECK(runtime.locomotion_states[1].current_route_id
            == REK_G1_NATIVE_TURN_RIGHT);
        CHECK(facts[0].translation_transition_settled
            && facts[1].translation_transition_settled);

        const RekG1SemanticTick move_from_yaw[2] = {
            move_tick(3u, 1, 0, move_durations[3] - 1u),
            move_tick(3u, 1, 0, move_durations[3] - 1u),
        };
        CHECK(rek_g1_semantic_duel_advance_batch(
            &runtime, routes, &table, move_from_yaw, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
        CHECK(runtime.active_route_ids[0]
            == REK_G1_NATIVE_KICK_MOVE_9_RIGHT_KNEE);
        CHECK(runtime.active_route_ids[1]
            == REK_G1_NATIVE_KICK_MOVE_9_RIGHT_KNEE);
        CHECK(runtime.composers[0].action_playing
            && runtime.composers[1].action_playing);

        CHECK(rek_g1_semantic_duel_reset_batch(
            &runtime, routes, &table, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
        const RekG1SemanticTick translation_yaw_semantics[2] = {
            locomotion_tick(REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_LEFT),
            locomotion_tick(REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_RIGHT),
        };
        CHECK(rek_g1_semantic_duel_advance_batch(
            &runtime, routes, &table, translation_yaw_semantics, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
        CHECK(runtime.locomotion_states[0].current_route_id
            == REK_G1_NATIVE_FORWARD);
        CHECK(runtime.locomotion_states[1].current_route_id
            == REK_G1_NATIVE_FORWARD);
        CHECK(!facts[0].translation_transition_settled
            && !facts[1].translation_transition_settled);
        CHECK(rek_g1_semantic_duel_reset_batch(
            &runtime, routes, &table, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
    }

    /* Corruption after open rejects before policy or physics can advance. */
    {
        const uint64_t guarded_policy_tick = duel.policy_ticks[0];
        const double guarded_time = duel.data[0]->time;
        const RekG1SemanticTick neutral_semantics[2] = {
            locomotion_tick(0u),
            locomotion_tick(0u),
        };
        duel.reset_complete_not_before_time[0] = 0.002;
        CHECK(!rek_g1_semantic_duel_advance_batch(
            &runtime, routes, &table, neutral_semantics, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
        CHECK(runtime.failed && !runtime.ready);
        CHECK(duel.policy_ticks[0] == guarded_policy_tick
            && duel.policy_ticks[1] == guarded_policy_tick);
        CHECK(duel.data[0]->time == guarded_time);
        duel.reset_complete_not_before_time[0] = 0.0;
        CHECK(rek_g1_semantic_duel_reset_batch(
            &runtime, routes, &table, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
    }

    /*
     * Trace the complete command-to-physics path from identical resets. A
     * semantic route flag alone is insufficient: sustained forward input
     * must change the composed reference, controller output, and MuJoCo state
     * relative to an equal-length neutral baseline.
     */
    {
        const size_t trace_ticks = 64u;
        const size_t reference_count = runtime.robot_count
            * REK_G1_SEMANTIC_DUEL_REFERENCE_ROWS * GEAR_SONIC_ACTION_DIM;
        const size_t encoder_count = runtime.robot_count
            * GEAR_SONIC_ENCODER_INPUT_WIDTH;
        const size_t token_count = runtime.robot_count
            * GEAR_SONIC_ENCODER_OUTPUT_WIDTH;
        const size_t action_count = runtime.robot_count * GEAR_SONIC_ACTION_DIM;
        const size_t qpos_count = (size_t)duel.model->nq;
        float* neutral_reference = malloc(reference_count * sizeof(float));
        float* neutral_encoder = malloc(encoder_count * sizeof(float));
        float* neutral_tokens = malloc(token_count * sizeof(float));
        float* neutral_actions = malloc(action_count * sizeof(float));
        float* neutral_targets = malloc(action_count * sizeof(float));
        double* neutral_qpos = malloc(qpos_count * sizeof(double));
        CHECK(neutral_reference != NULL && neutral_encoder != NULL
            && neutral_tokens != NULL && neutral_actions != NULL
            && neutral_targets != NULL && neutral_qpos != NULL);

        const RekG1SemanticTick neutral_semantics[2] = {
            locomotion_tick(0u),
            locomotion_tick(0u),
        };
        for (size_t tick = 0; tick < trace_ticks; tick++) {
            CHECK(rek_g1_semantic_duel_advance_batch(
                &runtime, routes, &table, neutral_semantics, 2u, facts,
                observations, sizeof(observations[0]), rewards, terminals,
                error, sizeof(error)));
        }
        memcpy(neutral_reference, runtime.reference_dof_position,
            reference_count * sizeof(float));
        memcpy(neutral_encoder, duel.controller.encoder_observations,
            encoder_count * sizeof(float));
        memcpy(neutral_tokens, duel.controller.tokens,
            token_count * sizeof(float));
        memcpy(neutral_actions, duel.controller.actions_policy,
            action_count * sizeof(float));
        memcpy(neutral_targets, duel.controller.targets_mujoco,
            action_count * sizeof(float));
        memcpy(neutral_qpos, duel.data[0]->qpos,
            qpos_count * sizeof(double));

        CHECK(rek_g1_semantic_duel_reset_batch(
            &runtime, routes, &table, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
        const RekG1SemanticTick forward_semantics[2] = {
            locomotion_tick(REK_G1_HELD_FORWARD),
            locomotion_tick(REK_G1_HELD_FORWARD),
        };
        for (size_t tick = 0; tick < trace_ticks; tick++) {
            CHECK(rek_g1_semantic_duel_advance_batch(
                &runtime, routes, &table, forward_semantics, 2u, facts,
                observations, sizeof(observations[0]), rewards, terminals,
                error, sizeof(error)));
        }

        const float reference_delta = max_abs_float_difference(
            neutral_reference, runtime.reference_dof_position,
            reference_count);
        const float encoder_delta = max_abs_float_difference(
            neutral_encoder, duel.controller.encoder_observations,
            encoder_count);
        const float token_delta = max_abs_float_difference(
            neutral_tokens, duel.controller.tokens, token_count);
        const float action_delta = max_abs_float_difference(
            neutral_actions, duel.controller.actions_policy, action_count);
        const float target_delta = max_abs_float_difference(
            neutral_targets, duel.controller.targets_mujoco, action_count);
        const double qpos_delta = max_abs_double_difference(
            neutral_qpos, duel.data[0]->qpos, qpos_count);
        printf(
            "neutral_forward_trace ticks=%zu reference_linf=%.9g "
            "encoder_linf=%.9g token_linf=%.9g action_linf=%.9g "
            "target_linf=%.9g qpos_linf=%.17g\n",
            trace_ticks, reference_delta, encoder_delta, token_delta,
            action_delta, target_delta, qpos_delta);

        free(neutral_qpos);
        free(neutral_targets);
        free(neutral_actions);
        free(neutral_tokens);
        free(neutral_encoder);
        free(neutral_reference);
        CHECK(reference_delta > 0.0f);
        CHECK(encoder_delta > 0.0f);
        CHECK(token_delta > 0.0f);
        CHECK(action_delta > 0.0f);
        CHECK(target_delta > 0.0f);
        CHECK(qpos_delta > 0.0);
        CHECK(rek_g1_semantic_duel_reset_batch(
            &runtime, routes, &table, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
    }

    /* Invalid row 1 rejects before the shared physics call and live publish. */
    RekG1SemanticTick invalid_semantics[2] = {
        locomotion_tick(REK_G1_HELD_FORWARD),
        locomotion_tick(REK_G1_HELD_BACKWARD),
    };
    invalid_semantics[1].status = REK_G1_SEMANTIC_INVALID_KIND;
    const uint64_t tick_before = duel.policy_ticks[0];
    const float cursor_before = runtime.composers[0].current_layer.cursor;
    CHECK(!rek_g1_semantic_duel_advance_batch(
        &runtime, routes, &table, invalid_semantics, 2u, facts,
        observations, sizeof(observations[0]), rewards, terminals,
        error, sizeof(error)));
    CHECK(duel.policy_ticks[0] == tick_before);
    CHECK(duel.policy_ticks[1] == tick_before);
    CHECK(runtime.composers[0].current_layer.cursor == cursor_before);

    CHECK(rek_g1_semantic_duel_reset_batch(
        &runtime, routes, &table, 2u, facts,
        observations, sizeof(observations[0]), rewards, terminals,
        error, sizeof(error)));

    RekG1SemanticTick valid_semantics[2] = {
        locomotion_tick(REK_G1_HELD_FORWARD),
        locomotion_tick(REK_G1_HELD_BACKWARD),
    };

    /*
     * Put the player in a measured, qualifying side-fall pose. The adapter
     * selects a physical pose before the runtime tick; the post-physics sample
     * must drive the state machine to FALLEN and publish recovery_active.
     */
    const GearSonicDuelFighterMap* player_map =
        &duel.fighters[GEAR_SONIC_DUEL_PLAYER];
    double reset_root_qpos[7];
    memcpy(
        reset_root_qpos,
        duel.data[0]->qpos + player_map->root_qpos_address,
        sizeof(reset_root_qpos));
    const double half_sqrt = sqrt(0.5);
    const double world_roll_90[4] = {half_sqrt, half_sqrt, 0.0, 0.0};
    double fallen_quaternion[4];
    quaternion_multiply(
        world_roll_90, reset_root_qpos + 3, fallen_quaternion);
    int qualifying_pose_found = 0;
    double qualifying_fallen_height = 0.0;
    for (int centimetres = 45; centimetres >= 10; centimetres--) {
        duel.data[0]->qpos[player_map->root_qpos_address + 2]
            = (double)centimetres / 100.0;
        memcpy(
            duel.data[0]->qpos + player_map->root_qpos_address + 3,
            fallen_quaternion,
            sizeof(fallen_quaternion));
        mju_zero(duel.data[0]->qvel, duel.model->nv);
        mj_forward(duel.model, duel.data[0]);
        RekG1FallMujocoMeasurement candidate = {0};
        CHECK(rek_g1_fall_mujoco_sample(
            &runtime.fall_adapter,
            0u,
            REK_G1_SEMANTIC_DUEL_CONTROL_DELTA_SECONDS,
            REK_G1_SEMANTIC_DUEL_PROVISIONAL_CAN_GET_UP,
            &candidate,
            error,
            sizeof(error)) == REK_G1_FALL_MUJOCO_OK);
        const int no_foot_contact =
            !candidate.fall_sample.has_foot_body_contact
            && candidate.fall_sample.distinct_nonfoot_body_contact_count
                >= 1u;
        const int enough_contacts =
            candidate.fall_sample.distinct_nonfoot_body_contact_count
                >= REK_G1_FALL_CONFIG_F84F1874.fallen_contact_points;
        if (candidate.fall_sample.tilt_degrees
                    > REK_G1_FALL_CONFIG_F84F1874.fallen_tilt_degrees
                && candidate.fall_sample.pelvis_height_ratio
                    < REK_G1_FALL_CONFIG_F84F1874.fallen_height_ratio
                && (no_foot_contact || enough_contacts)) {
            qualifying_pose_found = 1;
            qualifying_fallen_height =
                duel.data[0]->qpos[player_map->root_qpos_address + 2];
            break;
        }
    }
    CHECK(qualifying_pose_found);
    const RekG1SemanticTick neutral_semantics[2] = {
        locomotion_tick(0u),
        locomotion_tick(0u),
    };
    size_t measured_fall_ticks = 0u;
    do {
        CHECK(rek_g1_semantic_duel_advance_batch(
            &runtime, routes, &table, neutral_semantics, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
        measured_fall_ticks++;
    } while (runtime.fall_states[0].phase != REK_G1_FALL_FALLEN
        && measured_fall_ticks < 200u);
    printf(
        "measured_fall_path ticks=%zu phase=%d events=%u tilt=%.9g ratio=%.9g "
        "foot=%u nonfoot=%u hold=%.9g\n",
        measured_fall_ticks,
        (int)runtime.fall_states[0].phase,
        runtime.fall_events[0],
        runtime.fall_measurements[0].fall_sample.tilt_degrees,
        runtime.fall_measurements[0].fall_sample.pelvis_height_ratio,
        (unsigned int)runtime.fall_measurements[0]
            .fall_sample.has_foot_body_contact,
        runtime.fall_measurements[0]
            .fall_sample.distinct_nonfoot_body_contact_count,
        runtime.fall_states[0].fallen_hold_seconds);
    CHECK(runtime.fall_states[0].phase == REK_G1_FALL_FALLEN);
    CHECK((runtime.fall_events[0] & REK_G1_FALL_EVENT_BECAME_FALLEN) != 0u);
    CHECK(facts[0].recovery_active == 1u);
    CHECK(facts[0].translation_transition_settled == 0u);
    CHECK(observations[0].self.fall.phase == (float)REK_G1_FALL_FALLEN);
    CHECK(observations[0].self.fall.events
        == (float)REK_G1_FALL_EVENT_BECAME_FALLEN);
    CHECK(observations[1].opponent.fall.phase
        == (float)REK_G1_FALL_FALLEN);

    /*
     * BECAME_FALLEN suspends only the measured row. Its policy counter,
     * history, reference composition, locomotion, heading, forgiveness, and
     * motion cursor must stay bitwise fixed until the arena reset boundary. A
     * row-selective failing forgiveness fixture proves the callback is not
     * consulted for row 0 while the opponent continues normally.
     */
    CHECK(duel.dampened_rows[0] == 1u);
    CHECK(duel.dampened_rows[1] == 0u);
    CHECK(duel.resetting_rows[0] == 0u && duel.resetting_rows[1] == 0u);
    const SonicMotionComposerNative fallen_composer = runtime.composers[0];
    const RekG1NativeLocomotionState fallen_locomotion =
        runtime.locomotion_states[0];
    const RekG1NativeVelocityCommand fallen_effective_velocity =
        runtime.effective_velocity[0];
    const RekG1NativeRouteId fallen_active_route = runtime.active_route_ids[0];
    const float fallen_forgiveness = runtime.forgiveness_deltas[0];
    double fallen_heading[4];
    memcpy(fallen_heading, duel.heading_delta_wxyz, sizeof(fallen_heading));
    float fallen_lpf[GEAR_SONIC_ACTION_DIM];
    memcpy(
        fallen_lpf,
        duel.command_lpf_state_mujoco,
        sizeof(fallen_lpf));
    const uint8_t fallen_lpf_initialized = duel.command_lpf_initialized[0];
    const size_t fallen_reference_frame = duel.reference_frames[0];
    float fallen_history_quaternion[GEAR_SONIC_HISTORY_FRAMES * 4u];
    float fallen_history_angular[GEAR_SONIC_HISTORY_FRAMES * 3u];
    float fallen_history_joint_position[
        GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM];
    float fallen_history_joint_velocity[
        GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM];
    float fallen_history_last_action[
        GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM];
    memcpy(
        fallen_history_quaternion,
        duel.controller.history_base_quaternion_wxyz,
        sizeof(fallen_history_quaternion));
    memcpy(
        fallen_history_angular,
        duel.controller.history_base_angular_velocity,
        sizeof(fallen_history_angular));
    memcpy(
        fallen_history_joint_position,
        duel.controller.history_joint_position_policy,
        sizeof(fallen_history_joint_position));
    memcpy(
        fallen_history_joint_velocity,
        duel.controller.history_joint_velocity_policy,
        sizeof(fallen_history_joint_velocity));
    memcpy(
        fallen_history_last_action,
        duel.controller.history_last_action_policy,
        sizeof(fallen_history_last_action));
    const uint8_t fallen_history_count = duel.controller.history_count[0];
    const uint8_t fallen_history_head = duel.controller.history_head[0];
    const uint64_t fallen_policy_tick = duel.policy_ticks[0];
    const uint64_t opponent_policy_tick = duel.policy_ticks[1];
    const uint64_t fallen_motion_tick = duel.motion_ticks[0];
    const uint64_t opponent_motion_tick = duel.motion_ticks[1];
    const RekG1SemanticTick suspended_probe_semantics[2] = {
        locomotion_tick(REK_G1_HELD_FORWARD),
        locomotion_tick(0u),
    };
    probe.fail_forgiveness = 1;
    CHECK(rek_g1_semantic_duel_advance_batch(
        &runtime, routes, &table, suspended_probe_semantics, 2u, facts,
        observations, sizeof(observations[0]), rewards, terminals,
        error, sizeof(error)));
    probe.fail_forgiveness = 0;
    CHECK(duel.dampened_rows[0] == 1u);
    CHECK(duel.dampened_rows[1] == 0u);
    CHECK(duel.policy_ticks[0] == fallen_policy_tick);
    CHECK(duel.policy_ticks[1] == opponent_policy_tick + 1u);
    CHECK(duel.motion_ticks[0] == fallen_motion_tick);
    CHECK(duel.motion_ticks[1] == opponent_motion_tick + 1u);
    CHECK(duel.reference_frames[0] == fallen_reference_frame);
    CHECK(memcmp(
        &runtime.composers[0],
        &fallen_composer,
        sizeof(fallen_composer)) == 0);
    CHECK(memcmp(
        &runtime.locomotion_states[0],
        &fallen_locomotion,
        sizeof(fallen_locomotion)) == 0);
    CHECK(memcmp(
        &runtime.effective_velocity[0],
        &fallen_effective_velocity,
        sizeof(fallen_effective_velocity)) == 0);
    CHECK(runtime.active_route_ids[0] == fallen_active_route);
    CHECK(runtime.forgiveness_deltas[0] == fallen_forgiveness);
    CHECK(memcmp(
        duel.heading_delta_wxyz,
        fallen_heading,
        sizeof(fallen_heading)) == 0);
    CHECK(memcmp(
        duel.command_lpf_state_mujoco,
        fallen_lpf,
        sizeof(fallen_lpf)) == 0);
    CHECK(duel.command_lpf_initialized[0] == fallen_lpf_initialized);
    CHECK(memcmp(
        duel.controller.history_base_quaternion_wxyz,
        fallen_history_quaternion,
        sizeof(fallen_history_quaternion)) == 0);
    CHECK(memcmp(
        duel.controller.history_base_angular_velocity,
        fallen_history_angular,
        sizeof(fallen_history_angular)) == 0);
    CHECK(memcmp(
        duel.controller.history_joint_position_policy,
        fallen_history_joint_position,
        sizeof(fallen_history_joint_position)) == 0);
    CHECK(memcmp(
        duel.controller.history_joint_velocity_policy,
        fallen_history_joint_velocity,
        sizeof(fallen_history_joint_velocity)) == 0);
    CHECK(memcmp(
        duel.controller.history_last_action_policy,
        fallen_history_last_action,
        sizeof(fallen_history_last_action)) == 0);
    CHECK(duel.controller.history_count[0] == fallen_history_count);
    CHECK(duel.controller.history_head[0] == fallen_history_head);
    CHECK(facts[0].recovery_active == 1u);
    CHECK(facts[0].translation_transition_settled == 0u);
    CHECK(facts[0].input_reset == 0u && facts[1].input_reset == 0u);
    CHECK(runtime.arena_reset_events[0] == 0u);

    size_t no_recovery_count_ticks = 1u;
    do {
        const int count_step_ok = rek_g1_semantic_duel_advance_batch(
            &runtime, routes, &table, neutral_semantics, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error));
        if (!count_step_ok) {
            fprintf(
                stderr,
                "count step failed: status=%d hit_status=%d duel_failed=%d "
                "error=%s\n",
                (int)runtime.last_status,
                (int)runtime.hit_adapter.last_status,
                duel.failed,
                error);
        }
        CHECK(count_step_ok);
        no_recovery_count_ticks++;
    } while (!runtime.arena_reset_events[0]
        && no_recovery_count_ticks < 200u);
    CHECK(runtime.arena_reset_events[0] == 1u);
    CHECK(no_recovery_count_ticks <= 151u);
    CHECK(runtime.arena_input_reset_events[0] == 1u);
    CHECK(duel.reset_pending_arenas[0] == 0u);
    CHECK(duel.reset_completed_in_step_arenas[0] == 1u);
    CHECK(duel.dampened_rows[0] == 0u && duel.dampened_rows[1] == 0u);
    CHECK(duel.resetting_rows[0] == 0u && duel.resetting_rows[1] == 0u);
    CHECK(runtime.combat_states[0].fight.falls[0] == 1u);
    CHECK(runtime.combat_states[0].fight.clean_hits[1] == 5);
    CHECK(runtime.score_deltas[0] == 0
        && runtime.score_deltas[1] == 5);
    CHECK(rewards[0] == -5.0f && rewards[1] == 5.0f);
    CHECK(terminals[0] == 0.0f && terminals[1] == 0.0f);
    CHECK((runtime.fight_signals[0]
        & REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN) != 0u);
    CHECK(runtime.fall_states[0].phase == REK_G1_FALL_UPRIGHT);
    CHECK(runtime.fall_states[1].phase == REK_G1_FALL_UPRIGHT);
    CHECK(facts[0].recovery_active == 0u
        && facts[1].recovery_active == 0u);
    CHECK(facts[0].input_reset == 1u && facts[1].input_reset == 1u);
    CHECK(observations[0].fight.tick_opponent_score_delta == 5.0f);
    CHECK(observations[1].fight.tick_self_score_delta == 5.0f);
    CHECK(rek_g1_semantic_duel_reset_batch(
        &runtime, routes, &table, 2u, facts,
        observations, sizeof(observations[0]), rewards, terminals,
        error, sizeof(error)));

    /*
     * Force count expiry on fixed substep 9. This is the only placement in a
     * ten-substep controller tick that cannot complete the deferred reset in
     * the same outer call. It therefore exposes both observable phases:
     * immediate root teleport and score publication on the request tick, then
     * joint/controller/composer reset with paired input_reset at the next 2 ms
     * boundary.
     */
    {
        const GearSonicDuelFighterMap* opponent_map =
            &duel.fighters[GEAR_SONIC_DUEL_OPPONENT];
        double spawn_root_qpos[2][7];
        memcpy(
            spawn_root_qpos[0],
            duel.data[0]->qpos + player_map->root_qpos_address,
            sizeof(spawn_root_qpos[0]));
        memcpy(
            spawn_root_qpos[1],
            duel.data[0]->qpos + opponent_map->root_qpos_address,
            sizeof(spawn_root_qpos[1]));
        const SonicMotionComposerNative fresh_player_composer =
            runtime.composers[0];
        const RekG1NativeLocomotionState fresh_player_locomotion =
            runtime.locomotion_states[0];
        const RekG1NativeVelocityCommand fresh_player_velocity =
            runtime.effective_velocity[0];

        const RekG1SemanticTick prime_composer_semantics[2] = {
            locomotion_tick(REK_G1_HELD_FORWARD),
            locomotion_tick(0u),
        };
        CHECK(rek_g1_semantic_duel_advance_batch(
            &runtime, routes, &table, prime_composer_semantics, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
        CHECK(runtime.active_route_ids[0] == REK_G1_NATIVE_FORWARD);
        CHECK(memcmp(
            &runtime.composers[0],
            &fresh_player_composer,
            sizeof(fresh_player_composer)) != 0);

        memcpy(
            duel.data[0]->qpos + player_map->root_qpos_address,
            spawn_root_qpos[0],
            sizeof(spawn_root_qpos[0]));
        duel.data[0]->qpos[player_map->root_qpos_address + 2] =
            qualifying_fallen_height;
        memcpy(
            duel.data[0]->qpos + player_map->root_qpos_address + 3,
            fallen_quaternion,
            sizeof(fallen_quaternion));
        duel.data[0]->qpos[player_map->qpos_addresses[0]] = 0.37;
        mju_zero(duel.data[0]->qvel, duel.model->nv);
        mj_forward(duel.model, duel.data[0]);
        runtime.fall_states[0] = (RekG1FallState){
            .phase = REK_G1_FALL_FALLEN,
            .fallen_timer_seconds =
                REK_G1_FALL_CONFIG_F84F1874.fallen_reset_timeout_seconds,
        };
        CHECK(gear_sonic_native_duel_set_row_dampened(
            &duel, 0u, 1, error, sizeof(error)));

        RekG1FightState* forced_count =
            &runtime.combat_states[0].fight;
        forced_count->falls[0] += 1u;
        forced_count->count_active[0] = 1u;
        forced_count->count_active[1] = 0u;
        forced_count->count_is_slip[0] = 0u;
        forced_count->count_is_slip[1] = 0u;
        forced_count->count_duration_seconds =
            REK_G1_FIGHT_CONFIG_F84F1874.no_recovery_count_seconds;
        forced_count->count_elapsed_seconds = 2.98f;
        float predicted_count_elapsed = forced_count->count_elapsed_seconds;
        for (size_t substep = 0u; substep < 9u; substep++) {
            predicted_count_elapsed +=
                REK_G1_SEMANTIC_DUEL_PHYSICS_DELTA_SECONDS;
            CHECK(predicted_count_elapsed
                < forced_count->count_duration_seconds);
        }
        predicted_count_elapsed +=
            REK_G1_SEMANTIC_DUEL_PHYSICS_DELTA_SECONDS;
        CHECK(predicted_count_elapsed
            >= forced_count->count_duration_seconds);

        const SonicMotionComposerNative pending_player_composer =
            runtime.composers[0];
        const uint64_t player_policy_before_pending = duel.policy_ticks[0];
        const uint64_t opponent_policy_before_pending = duel.policy_ticks[1];
        const uint64_t player_motion_before_pending = duel.motion_ticks[0];
        const uint64_t opponent_motion_before_pending = duel.motion_ticks[1];
        const double time_before_pending = duel.data[0]->time;
        CHECK(rek_g1_semantic_duel_advance_batch(
            &runtime, routes, &table, neutral_semantics, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));

        CHECK(fabs(
            duel.data[0]->time
                - (time_before_pending
                    + REK_G1_SEMANTIC_DUEL_CONTROL_DELTA_SECONDS))
            < 1e-8);
        CHECK(duel.reset_pending_arenas[0] == 1u);
        CHECK(duel.reset_completed_in_step_arenas[0] == 0u);
        CHECK(duel.resetting_rows[0] == 1u
            && duel.resetting_rows[1] == 1u);
        CHECK(duel.dampened_rows[0] == 1u);
        CHECK(duel.dampened_rows[1] == 0u);
        CHECK(fabs(
            duel.reset_complete_not_before_time[0]
                - (duel.data[0]->time
                    + REK_G1_SEMANTIC_DUEL_PHYSICS_DELTA_SECONDS))
            < 1e-8);
        CHECK(duel.policy_ticks[0] == player_policy_before_pending);
        CHECK(duel.policy_ticks[1] == opponent_policy_before_pending);
        CHECK(duel.motion_ticks[0] == player_motion_before_pending);
        CHECK(duel.motion_ticks[1] == opponent_motion_before_pending);
        CHECK(memcmp(
            &runtime.composers[0],
            &pending_player_composer,
            sizeof(pending_player_composer)) == 0);
        CHECK(max_abs_double_difference(
            duel.data[0]->qpos + player_map->root_qpos_address,
            spawn_root_qpos[0],
            7u) == 0.0);
        CHECK(max_abs_double_difference(
            duel.data[0]->qpos + opponent_map->root_qpos_address,
            spawn_root_qpos[1],
            7u) == 0.0);
        CHECK(fabs(
            duel.data[0]->qpos[player_map->qpos_addresses[0]]) > 1e-6);
        CHECK(runtime.arena_reset_events[0] == 1u);
        CHECK(runtime.arena_input_reset_events[0] == 0u);
        CHECK(runtime.score_deltas[0] == 0
            && runtime.score_deltas[1]
                == REK_G1_FIGHT_CONFIG_F84F1874.ko_points);
        CHECK(rewards[0]
            == -(float)REK_G1_FIGHT_CONFIG_F84F1874.ko_points);
        CHECK(rewards[1]
            == (float)REK_G1_FIGHT_CONFIG_F84F1874.ko_points);
        CHECK(terminals[0] == 0.0f && terminals[1] == 0.0f);
        CHECK((runtime.fight_signals[0]
            & REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN) != 0u);
        CHECK(facts[0].recovery_active == 1u
            && facts[1].recovery_active == 1u);
        CHECK(facts[0].input_reset == 0u && facts[1].input_reset == 0u);
        CHECK(observations[0].fight.tick_opponent_score_delta
            == (float)REK_G1_FIGHT_CONFIG_F84F1874.ko_points);
        CHECK(observations[1].fight.tick_self_score_delta
            == (float)REK_G1_FIGHT_CONFIG_F84F1874.ko_points);
        CHECK(runtime.fall_states[0].phase == REK_G1_FALL_UPRIGHT);
        CHECK(runtime.fall_states[1].phase == REK_G1_FALL_UPRIGHT);
        CHECK(runtime.fall_states[0].reset_grace_remaining_seconds
            == REK_G1_FALL_CONFIG_F84F1874.fight_spawn_reset_grace_seconds);

        const RekG1SemanticTick ignored_during_reset_semantics[2] = {
            locomotion_tick(REK_G1_HELD_FORWARD),
            locomotion_tick(REK_G1_HELD_BACKWARD),
        };
        probe.fail_forgiveness = 1;
        CHECK(rek_g1_semantic_duel_advance_batch(
            &runtime, routes, &table, ignored_during_reset_semantics, 2u,
            facts, observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
        probe.fail_forgiveness = 0;
        CHECK(duel.reset_pending_arenas[0] == 0u);
        CHECK(duel.reset_completed_in_step_arenas[0] == 1u);
        CHECK(duel.resetting_rows[0] == 0u
            && duel.resetting_rows[1] == 0u);
        CHECK(duel.dampened_rows[0] == 0u
            && duel.dampened_rows[1] == 0u);
        CHECK(duel.reset_complete_not_before_time[0] == 0.0);
        CHECK(duel.policy_ticks[0] == 0u && duel.policy_ticks[1] == 0u);
        CHECK(duel.motion_ticks[0] == 0u && duel.motion_ticks[1] == 0u);
        CHECK(runtime.arena_reset_events[0] == 1u);
        CHECK(runtime.arena_input_reset_events[0] == 1u);
        CHECK(facts[0].input_reset == 1u && facts[1].input_reset == 1u);
        CHECK(facts[0].recovery_active == 0u
            && facts[1].recovery_active == 0u);
        CHECK(runtime.score_deltas[0] == 0
            && runtime.score_deltas[1] == 0);
        CHECK(rewards[0] == 0.0f && rewards[1] == 0.0f);
        CHECK(terminals[0] == 0.0f && terminals[1] == 0.0f);
        CHECK(runtime.active_route_ids[0] == REK_G1_NATIVE_IDLE);
        CHECK(memcmp(
            &runtime.composers[0],
            &fresh_player_composer,
            sizeof(fresh_player_composer)) == 0);
        CHECK(memcmp(
            &runtime.locomotion_states[0],
            &fresh_player_locomotion,
            sizeof(fresh_player_locomotion)) == 0);
        CHECK(memcmp(
            &runtime.effective_velocity[0],
            &fresh_player_velocity,
            sizeof(fresh_player_velocity)) == 0);
        CHECK(runtime.forgiveness_deltas[0] == 0.0f
            && runtime.forgiveness_deltas[1] == 0.0f);
        CHECK(runtime.fall_states[0].reset_grace_remaining_seconds <
            REK_G1_FALL_CONFIG_F84F1874.fight_spawn_reset_grace_seconds);
        CHECK(runtime.fall_states[0].reset_grace_remaining_seconds > 1.97f);

        CHECK(rek_g1_semantic_duel_advance_batch(
            &runtime, routes, &table, prime_composer_semantics, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
        CHECK(duel.reset_completed_in_step_arenas[0] == 0u);
        CHECK(duel.policy_ticks[0] == 1u && duel.policy_ticks[1] == 1u);
        CHECK(duel.motion_ticks[0] == 1u && duel.motion_ticks[1] == 1u);
        CHECK(facts[0].input_reset == 0u && facts[1].input_reset == 0u);
        CHECK(runtime.active_route_ids[0] == REK_G1_NATIVE_FORWARD);
    }

    CHECK(rek_g1_semantic_duel_reset_batch(
        &runtime, routes, &table, 2u, facts,
        observations, sizeof(observations[0]), rewards, terminals,
        error, sizeof(error)));

    /*
     * A recovered round-end signal terminates both Puffer rows in the arena on
     * the same native step.  The reward remains the explicitly labelled
     * score-delta training contract rather than an inferred REK reward.
     */
    runtime.combat_states[0].fight.time_remaining_seconds = 0.001f;
    CHECK(rek_g1_semantic_duel_advance_batch(
        &runtime, routes, &table, neutral_semantics, 2u, facts,
        observations, sizeof(observations[0]), rewards, terminals,
        error, sizeof(error)));
    CHECK((runtime.fight_signals[0]
        & REK_G1_FIGHT_SIGNAL_ROUND_ENDED) != 0u);
    CHECK(runtime.arena_terminals[0] == 1u);
    CHECK(runtime.pending_episode_resets[0] == 1u);
    CHECK(terminals[0] == 1.0f && terminals[1] == 1.0f);
    CHECK(rewards[0] == -rewards[1]);
    CHECK(observations[0].fight.phase
        == (float)REK_G1_FIGHT_BETWEEN_ROUNDS);
    CHECK(observations[1].fight.phase
        == (float)REK_G1_FIGHT_BETWEEN_ROUNDS);
    CHECK(facts[0].translation_transition_settled == 1u
        && facts[1].translation_transition_settled == 1u);
    CHECK(facts[0].action_busy == 0u && facts[1].action_busy == 0u);
    CHECK(facts[0].recovery_active == 0u
        && facts[1].recovery_active == 0u);
    const RekG1SemanticDuelObservation terminal_observation = observations[0];
    CHECK(rek_g1_semantic_duel_advance_batch(
        &runtime, routes, &table, neutral_semantics, 2u, facts,
        observations, sizeof(observations[0]), rewards, terminals,
        error, sizeof(error)));
    CHECK(runtime.pending_episode_resets[0] == 0u);
    CHECK(runtime.arena_terminals[0] == 0u);
    CHECK(terminals[0] == 0.0f && terminals[1] == 0.0f);
    CHECK(runtime.combat_states[0].fight.phase
        == REK_G1_FIGHT_ROUND_ACTIVE);
    CHECK(observations[0].fight.phase
        == (float)REK_G1_FIGHT_ROUND_ACTIVE);
    CHECK(terminal_observation.fight.phase
        == (float)REK_G1_FIGHT_BETWEEN_ROUNDS);
    CHECK(rek_g1_semantic_duel_reset_batch(
        &runtime, routes, &table, 2u, facts,
        observations, sizeof(observations[0]), rewards, terminals,
        error, sizeof(error)));

    probe.fail_forgiveness = 1;
    CHECK(!rek_g1_semantic_duel_advance_batch(
        &runtime, routes, &table, valid_semantics, 2u, facts,
        observations, sizeof(observations[0]), rewards, terminals,
        error, sizeof(error)));
    CHECK(runtime.last_status
        == REK_G1_SEMANTIC_DUEL_FORGIVENESS_UNAVAILABLE);
    CHECK(duel.policy_ticks[0] == 0u && duel.policy_ticks[1] == 0u);
    probe.fail_forgiveness = 0;
    CHECK(rek_g1_semantic_duel_reset_batch(
        &runtime, routes, &table, 2u, facts,
        observations, sizeof(observations[0]), rewards, terminals,
        error, sizeof(error)));

    /*
     * ExecuteMove has no completion callback. The terminal non-loop layer
     * stays active and busy until the next zero-command stop update plays
     * idle. The supplied duration must align with composer completion.
     */
    const uint32_t move_ticks = move_durations[0];
    CHECK(move_ticks == 157u);
    for (uint32_t tick = 0; tick < move_ticks; tick++) {
        RekG1SemanticTick move_semantics[2] = {
            move_tick(
                0u,
                tick == 0u,
                tick + 1u == move_ticks,
                move_ticks - tick - 1u),
            locomotion_tick(0u),
        };
        CHECK(rek_g1_semantic_duel_advance_batch(
            &runtime, routes, &table, move_semantics, 2u, facts,
            observations, sizeof(observations[0]), rewards, terminals,
            error, sizeof(error)));
    }
    CHECK(!runtime.composers[0].action_playing);
    CHECK(runtime.composers[0].current_layer.active);
    CHECK(!runtime.composers[0].current_layer.config.loop);
    CHECK(facts[0].action_busy);
    CHECK(runtime.active_route_ids[0]
        == REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE);
    RekG1SemanticTick post_move_idle[2] = {
        locomotion_tick(0u),
        locomotion_tick(0u),
    };
    CHECK(rek_g1_semantic_duel_advance_batch(
        &runtime, routes, &table, post_move_idle, 2u, facts,
        observations, sizeof(observations[0]), rewards, terminals,
        error, sizeof(error)));
    CHECK(runtime.active_route_ids[0] == REK_G1_NATIVE_IDLE);
    CHECK(!facts[0].action_busy);
    CHECK(rek_g1_semantic_duel_reset_batch(
        &runtime, routes, &table, 2u, facts,
        observations, sizeof(observations[0]), rewards, terminals,
        error, sizeof(error)));

    /* The native Puffer bridge drives both physical rows in one duel tick. */
    RekG1NativePufferVector vector = {0};
    CHECK(rek_g1_native_puffer_open(
        &vector, 2u, &table, routes,
        rek_g1_semantic_duel_batch_ops(), &runtime)
        == REK_G1_NATIVE_PUFFER_OK);
    float actions[2] = {0.0f, 0.0f};
    uint8_t masks[2][REK_G1_SEMANTIC_ACTION_COUNT] = {{0}};
    RekG1NativePufferIO io = {
        .actions = actions,
        .action_rows = 2u,
        .action_heads = 1u,
        .observations = observations,
        .observation_rows = 2u,
        .observation_stride_bytes = sizeof(observations[0]),
        .rewards = rewards,
        .reward_rows = 2u,
        .terminals = terminals,
        .terminal_rows = 2u,
        .action_masks = &masks[0][0],
        .action_mask_rows = 2u,
        .action_mask_stride_bytes = REK_G1_SEMANTIC_ACTION_COUNT,
    };
    CHECK(rek_g1_native_puffer_reset(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK);
    /* Category 2 is forward. Category 3 is backward. */
    actions[0] = 2.0f;
    actions[1] = 3.0f;
    CHECK(rek_g1_native_puffer_step(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK);
    CHECK(duel.policy_ticks[0] == 1u && duel.policy_ticks[1] == 1u);
    CHECK(runtime.active_route_ids[0] == REK_G1_NATIVE_FORWARD);
    CHECK(runtime.active_route_ids[1] == REK_G1_NATIVE_BACKWARD);
    CHECK(runtime.composers[0].current_layer.cursor != 0.0f);
    CHECK(runtime.composers[1].current_layer.cursor != 0.0f);
    CHECK(probe.matcher_calls >= 2u);
    CHECK(observations[0].self.fall.tilt_degrees
        == runtime.fall_measurements[0].fall_sample.tilt_degrees);
    CHECK(observations[0].self.fall.pelvis_height_ratio
        == runtime.fall_measurements[0].fall_sample.pelvis_height_ratio);
    CHECK(observations[0].self.fall.phase
        == (float)runtime.fall_states[0].phase);
    CHECK(observations[0].self.fall.events
        == (float)runtime.fall_events[0]);
    CHECK(observations[1].self.fall.tilt_degrees
        == runtime.fall_measurements[1].fall_sample.tilt_degrees);
    CHECK(observations[1].self.fall.phase
        == (float)runtime.fall_states[1].phase);
    for (size_t category = 16u;
            category < REK_G1_SEMANTIC_ACTION_COUNT; category++) {
        CHECK(masks[0][category] == 0u);
        CHECK(masks[1][category] == 0u);
    }
    CHECK(all_finite_observation(&observations[0]));
    CHECK(all_finite_observation(&observations[1]));
    CHECK(rewards[0] == 0.0f && rewards[1] == 0.0f);
    CHECK(terminals[0] == 0.0f && terminals[1] == 0.0f);

    actions[0] = 0.0f;
    actions[1] = 0.0f;
    CHECK(rek_g1_native_puffer_step(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK);
    CHECK(rek_g1_native_puffer_step(
        &vector, io, error, sizeof(error)) == REK_G1_NATIVE_PUFFER_OK);
    CHECK(duel.policy_ticks[0] == 3u && duel.policy_ticks[1] == 3u);
    CHECK(masks[0][1] == 1u && masks[1][1] == 1u);
    for (size_t category = 16u;
            category < REK_G1_SEMANTIC_ACTION_COUNT; category++) {
        CHECK(masks[0][category] == 0u);
        CHECK(masks[1][category] == 0u);
    }

    rek_g1_native_puffer_close(&vector);
    CHECK(!runtime.initialized);

    /*
     * A terminal is arena-local. Its final observation is returned once, and
     * only that arena is reinitialized before the following outer step.
     */
    if (multi_encoder != NULL && multi_decoder != NULL) {
        GearSonicNativeDuelVector multi_duel = {0};
        CHECK(gear_sonic_native_duel_open(
            &multi_duel, model_path, multi_encoder, multi_decoder, fixed_idle,
            4u, 1, error, sizeof(error)));
        RekG1SemanticDuelRuntime multi_runtime = {0};
        CHECK(rek_g1_semantic_duel_open(
            &multi_runtime, &multi_duel, routes, assets,
            REK_G1_STATIC_ROUTE_COUNT, &config, error, sizeof(error))
            == REK_G1_SEMANTIC_DUEL_OK);

        RekG1RuntimeFacts multi_facts[8] = {0};
        RekG1SemanticDuelObservation multi_observations[8] = {0};
        float multi_rewards[8] = {0};
        float multi_terminals[8] = {0};
        const RekG1SemanticTick multi_semantics[8] = {
            locomotion_tick(0u),
            locomotion_tick(0u),
            locomotion_tick(0u),
            locomotion_tick(0u),
            locomotion_tick(0u),
            locomotion_tick(0u),
            locomotion_tick(0u),
            locomotion_tick(0u),
        };
        CHECK(rek_g1_semantic_duel_reset_batch(
            &multi_runtime, routes, &table, 8u, multi_facts,
            multi_observations, sizeof(multi_observations[0]),
            multi_rewards, multi_terminals, error, sizeof(error)));
        multi_runtime.combat_states[0].fight.time_remaining_seconds = 0.001f;
        multi_runtime.combat_states[1].fight.clean_hits[0] = 7;

        CHECK(rek_g1_semantic_duel_advance_batch(
            &multi_runtime, routes, &table, multi_semantics, 8u, multi_facts,
            multi_observations, sizeof(multi_observations[0]),
            multi_rewards, multi_terminals, error, sizeof(error)));
        CHECK(multi_terminals[0] == 1.0f
            && multi_terminals[1] == 1.0f);
        for (size_t row = 2u; row < 8u; row++) {
            CHECK(multi_terminals[row] == 0.0f);
        }
        CHECK(multi_rewards[0] == -multi_rewards[1]);
        CHECK(multi_runtime.pending_episode_resets[0] == 1u);
        for (size_t arena = 1u; arena < 4u; arena++) {
            CHECK(multi_runtime.pending_episode_resets[arena] == 0u);
        }
        CHECK(multi_observations[0].fight.phase
            == (float)REK_G1_FIGHT_BETWEEN_ROUNDS);
        CHECK(multi_observations[2].fight.self_clean_hits == 7.0f);
        for (size_t row = 0u; row < 8u; row++) {
            CHECK(multi_duel.policy_ticks[row] == 1u);
        }
        const RekG1SemanticDuelObservation final_observation =
            multi_observations[0];

        CHECK(rek_g1_semantic_duel_advance_batch(
            &multi_runtime, routes, &table, multi_semantics, 8u, multi_facts,
            multi_observations, sizeof(multi_observations[0]),
            multi_rewards, multi_terminals, error, sizeof(error)));
        for (size_t row = 0u; row < 8u; row++) {
            CHECK(multi_terminals[row] == 0.0f);
        }
        for (size_t arena = 0u; arena < 4u; arena++) {
            CHECK(multi_runtime.pending_episode_resets[arena] == 0u);
        }
        CHECK(final_observation.fight.phase
            == (float)REK_G1_FIGHT_BETWEEN_ROUNDS);
        CHECK(multi_runtime.combat_states[0].fight.phase
            == REK_G1_FIGHT_ROUND_ACTIVE);
        CHECK(multi_runtime.combat_states[1].fight.phase
            == REK_G1_FIGHT_ROUND_ACTIVE);
        CHECK(multi_runtime.combat_states[1].fight.clean_hits[0] == 7);
        CHECK(multi_observations[2].fight.self_clean_hits == 7.0f);
        CHECK(multi_duel.policy_ticks[0] == 1u
            && multi_duel.policy_ticks[1] == 1u);
        for (size_t row = 2u; row < 8u; row++) {
            CHECK(multi_duel.policy_ticks[row] == 2u);
        }
        CHECK(multi_duel.motion_ticks[0] == 1u
            && multi_duel.motion_ticks[1] == 1u);
        for (size_t row = 2u; row < 8u; row++) {
            CHECK(multi_duel.motion_ticks[row] == 2u);
        }
        CHECK(fabsf(
            multi_runtime.combat_states[0].fight.time_remaining_seconds
                - 119.98f) < 1e-3f);
        CHECK(fabsf(
            multi_runtime.combat_states[1].fight.time_remaining_seconds
                - 119.96f) < 1e-3f);

        /* Deferred referee reset is also arena-local across a batch. */
        CHECK(rek_g1_semantic_duel_reset_batch(
            &multi_runtime, routes, &table, 8u, multi_facts,
            multi_observations, sizeof(multi_observations[0]),
            multi_rewards, multi_terminals, error, sizeof(error)));
        multi_runtime.combat_states[1].fight.clean_hits[0] = 7;
        double multi_spawn_root[2][7];
        memcpy(
            multi_spawn_root[0],
            multi_duel.data[0]->qpos + player_map->root_qpos_address,
            sizeof(multi_spawn_root[0]));
        memcpy(
            multi_spawn_root[1],
            multi_duel.data[0]->qpos +
                multi_duel.fighters[GEAR_SONIC_DUEL_OPPONENT]
                    .root_qpos_address,
            sizeof(multi_spawn_root[1]));
        memcpy(
            multi_duel.data[0]->qpos + player_map->root_qpos_address,
            multi_spawn_root[0],
            sizeof(multi_spawn_root[0]));
        multi_duel.data[0]->qpos[player_map->root_qpos_address + 2] =
            qualifying_fallen_height;
        memcpy(
            multi_duel.data[0]->qpos + player_map->root_qpos_address + 3,
            fallen_quaternion,
            sizeof(fallen_quaternion));
        multi_duel.data[0]->qpos[player_map->qpos_addresses[0]] = 0.37;
        mju_zero(multi_duel.data[0]->qvel, multi_duel.model->nv);
        mj_forward(multi_duel.model, multi_duel.data[0]);
        multi_runtime.fall_states[0] = (RekG1FallState){
            .phase = REK_G1_FALL_FALLEN,
            .fallen_timer_seconds =
                REK_G1_FALL_CONFIG_F84F1874.fallen_reset_timeout_seconds,
        };
        CHECK(gear_sonic_native_duel_set_row_dampened(
            &multi_duel, 0u, 1, error, sizeof(error)));
        RekG1FightState* multi_forced_count =
            &multi_runtime.combat_states[0].fight;
        multi_forced_count->falls[0] += 1u;
        multi_forced_count->count_active[0] = 1u;
        multi_forced_count->count_active[1] = 0u;
        multi_forced_count->count_is_slip[0] = 0u;
        multi_forced_count->count_is_slip[1] = 0u;
        multi_forced_count->count_duration_seconds =
            REK_G1_FIGHT_CONFIG_F84F1874.no_recovery_count_seconds;
        multi_forced_count->count_elapsed_seconds = 2.98f;
        const double isolated_time_before = multi_duel.data[1]->time;

        CHECK(rek_g1_semantic_duel_advance_batch(
            &multi_runtime, routes, &table, multi_semantics, 8u, multi_facts,
            multi_observations, sizeof(multi_observations[0]),
            multi_rewards, multi_terminals, error, sizeof(error)));
        CHECK(multi_duel.reset_pending_arenas[0] == 1u);
        CHECK(multi_duel.reset_completed_in_step_arenas[0] == 0u);
        CHECK(multi_runtime.arena_reset_events[0] == 1u);
        CHECK(multi_runtime.arena_input_reset_events[0] == 0u);
        for (size_t arena = 1u; arena < 4u; arena++) {
            CHECK(multi_duel.reset_pending_arenas[arena] == 0u);
            CHECK(multi_duel.reset_completed_in_step_arenas[arena] == 0u);
            CHECK(multi_runtime.arena_reset_events[arena] == 0u);
            CHECK(multi_runtime.arena_input_reset_events[arena] == 0u);
        }
        CHECK(max_abs_double_difference(
            multi_duel.data[0]->qpos + player_map->root_qpos_address,
            multi_spawn_root[0],
            7u) == 0.0);
        CHECK(max_abs_double_difference(
            multi_duel.data[0]->qpos +
                multi_duel.fighters[GEAR_SONIC_DUEL_OPPONENT]
                    .root_qpos_address,
            multi_spawn_root[1],
            7u) == 0.0);
        CHECK(multi_rewards[0]
            == -(float)REK_G1_FIGHT_CONFIG_F84F1874.ko_points);
        CHECK(multi_rewards[1]
            == (float)REK_G1_FIGHT_CONFIG_F84F1874.ko_points);
        for (size_t row = 2u; row < 8u; row++) {
            CHECK(multi_rewards[row] == 0.0f);
            CHECK(multi_terminals[row] == 0.0f);
            CHECK(multi_facts[row].input_reset == 0u);
            CHECK(multi_duel.policy_ticks[row] == 1u);
            CHECK(multi_duel.motion_ticks[row] == 1u);
        }
        CHECK(multi_runtime.combat_states[1].fight.clean_hits[0] == 7);
        CHECK(fabs(
            multi_duel.data[1]->time
                - (isolated_time_before
                    + REK_G1_SEMANTIC_DUEL_CONTROL_DELTA_SECONDS))
            < 1e-8);

        CHECK(rek_g1_semantic_duel_advance_batch(
            &multi_runtime, routes, &table, multi_semantics, 8u, multi_facts,
            multi_observations, sizeof(multi_observations[0]),
            multi_rewards, multi_terminals, error, sizeof(error)));
        CHECK(multi_duel.reset_pending_arenas[0] == 0u);
        CHECK(multi_duel.reset_completed_in_step_arenas[0] == 1u);
        CHECK(multi_duel.policy_ticks[0] == 0u
            && multi_duel.policy_ticks[1] == 0u);
        CHECK(multi_duel.motion_ticks[0] == 0u
            && multi_duel.motion_ticks[1] == 0u);
        CHECK(multi_runtime.arena_input_reset_events[0] == 1u);
        CHECK(multi_facts[0].input_reset == 1u
            && multi_facts[1].input_reset == 1u);
        for (size_t arena = 1u; arena < 4u; arena++) {
            CHECK(multi_duel.reset_pending_arenas[arena] == 0u);
            CHECK(multi_duel.reset_completed_in_step_arenas[arena] == 0u);
            CHECK(multi_runtime.arena_reset_events[arena] == 0u);
            CHECK(multi_runtime.arena_input_reset_events[arena] == 0u);
        }
        for (size_t row = 2u; row < 8u; row++) {
            CHECK(multi_facts[row].input_reset == 0u);
            CHECK(multi_duel.policy_ticks[row] == 2u);
            CHECK(multi_duel.motion_ticks[row] == 2u);
        }
        CHECK(multi_runtime.combat_states[1].fight.clean_hits[0] == 7);
        CHECK(fabs(
            multi_duel.data[1]->time
                - (isolated_time_before
                    + 2.0 * REK_G1_SEMANTIC_DUEL_CONTROL_DELTA_SECONDS))
            < 1e-8);

        rek_g1_semantic_duel_close(&multi_runtime);
        gear_sonic_native_duel_close(&multi_duel);
    }
    gear_sonic_native_duel_close(&duel);
    free_storage(storage, TEST_UNIQUE_CLIP_COUNT);
    return 1;
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 6) {
        fprintf(stderr,
            "usage: %s ASSET_DIR ENCODER_ONNX DECODER_ONNX "
            "[MULTI_ENCODER_ONNX MULTI_DECODER_ONNX]\n",
            argv[0]);
        return 2;
    }
    if (!run_test(
            argv[1], argv[2], argv[3],
            argc == 6 ? argv[4] : NULL,
            argc == 6 ? argv[5] : NULL)) {
        fprintf(stderr, "semantic duel runtime test failed after %d checks\n",
            checks);
        return 1;
    }
    printf("semantic duel runtime: %d checks passed\n", checks);
    printf("classification: executable public-family controller candidate\n");
    printf("parity_claim: false\n");
    printf("reward_support: zero_sum_score_delta_training_contract\n");
    printf("terminal_support: recovered_round_end_event\n");
    printf("test_matcher: explicit_authored_entry_fixture_not_REK_matcher\n");
    printf("test_forgiveness: explicit_zero_fixture_not_runtime_measurement\n");
    printf("fall_detection: build_pinned_mujoco_root_floor_contact_measurement\n");
    printf("suspension_execution: policy_and_motion_ticks_frozen\n");
    printf("recovery_execution: candidate_can_get_up_false_count_then_two_phase_spawn_reset\n");
    return 0;
}
