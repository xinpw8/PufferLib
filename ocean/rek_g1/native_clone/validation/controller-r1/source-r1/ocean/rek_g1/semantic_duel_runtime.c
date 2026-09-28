#include "semantic_duel_runtime.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(
    sizeof(RekG1SemanticDuelFallObservation)
        == REK_G1_SEMANTIC_DUEL_FALL_OBSERVATION_FLOATS * sizeof(float),
    "fall observation must be packed binary32 fields");
_Static_assert(
    sizeof(RekG1SemanticDuelEntityObservation)
        == REK_G1_SEMANTIC_DUEL_ENTITY_OBSERVATION_FLOATS * sizeof(float),
    "entity observation must be packed binary32 fields");
_Static_assert(
    sizeof(RekG1SemanticDuelFightObservation)
        == REK_G1_SEMANTIC_DUEL_FIGHT_OBSERVATION_FLOATS * sizeof(float),
    "fight observation must be packed binary32 fields");
_Static_assert(
    sizeof(RekG1SemanticDuelObservation)
        == REK_G1_SEMANTIC_DUEL_OBSERVATION_FLOATS * sizeof(float),
    "duel observation must be packed binary32 fields");

static const SonicMotionComposerNativeReferenceTiming REFERENCE_TIMING = {
    .current_offsets = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45},
    .next_offsets = {1, 6, 11, 16, 21, 26, 31, 36, 41, 46},
};

static void set_error(
        char* error,
        size_t error_capacity,
        const char* operation,
        const char* detail) {
    if (error == NULL || error_capacity == 0) return;
    if (operation == NULL) operation = "semantic duel runtime";
    if (detail == NULL) detail = "unknown failure";
    (void)snprintf(error, error_capacity, "%s: %s", operation, detail);
}

static int checked_product(size_t left, size_t right, size_t* output) {
    if (output == NULL || (right != 0 && left > SIZE_MAX / right)) return 0;
    *output = left * right;
    return 1;
}

static int finite_f32(float value) {
    return isfinite(value);
}

/* SonicMotionComposer.IsBusy is currentLayer.active && !currentLayer.loop. */
static int composer_is_busy(const SonicMotionComposerNative* composer) {
    return composer != NULL
        && composer->current_layer.active
        && !composer->current_layer.config.loop;
}

static float f32_add(float left, float right) {
    volatile float result = left + right;
    return result;
}

static float f32_sub(float left, float right) {
    volatile float result = left - right;
    return result;
}

static float f32_mul(float left, float right) {
    volatile float result = left * right;
    return result;
}

static RekG1SemanticDuelStatus latch_failure(
        RekG1SemanticDuelRuntime* runtime,
        RekG1SemanticDuelStatus status,
        char* error,
        size_t error_capacity,
        const char* operation,
        const char* detail) {
    if (runtime != NULL) {
        runtime->failed = 1u;
        runtime->ready = 0u;
        runtime->last_status = status;
    }
    set_error(error, error_capacity, operation, detail);
    return status;
}

const char* rek_g1_semantic_duel_status_string(
        RekG1SemanticDuelStatus status) {
    switch (status) {
        case REK_G1_SEMANTIC_DUEL_OK:
            return "ok";
        case REK_G1_SEMANTIC_DUEL_NULL_ARGUMENT:
            return "null argument";
        case REK_G1_SEMANTIC_DUEL_INVALID_DUEL:
            return "invalid shared-contact duel";
        case REK_G1_SEMANTIC_DUEL_INVALID_CONFIG:
            return "invalid or incomplete runtime config";
        case REK_G1_SEMANTIC_DUEL_INVALID_ROUTE_ASSET:
            return "invalid route asset";
        case REK_G1_SEMANTIC_DUEL_INVALID_FIXED_IDLE:
            return "duel fixed motion does not equal semantic idle";
        case REK_G1_SEMANTIC_DUEL_ALLOCATION_FAILED:
            return "allocation failed";
        case REK_G1_SEMANTIC_DUEL_NOT_READY:
            return "runtime not ready";
        case REK_G1_SEMANTIC_DUEL_RECOVERY_REQUIRED:
            return "recovery required but recovery references are unavailable";
        case REK_G1_SEMANTIC_DUEL_FORGIVENESS_UNAVAILABLE:
            return "heading forgiveness delta unavailable";
        case REK_G1_SEMANTIC_DUEL_COMPOSER_FAILED:
            return "motion composer failed";
        case REK_G1_SEMANTIC_DUEL_LOCOMOTION_FAILED:
            return "locomotion state update failed";
        case REK_G1_SEMANTIC_DUEL_PHYSICS_FAILED:
            return "shared-contact physics step failed";
        case REK_G1_SEMANTIC_DUEL_OBSERVATION_FAILED:
            return "factual observation failed";
        case REK_G1_SEMANTIC_DUEL_PROTOCOL_INVALID:
            return "semantic protocol invalid";
        case REK_G1_SEMANTIC_DUEL_FALL_MEASUREMENT_FAILED:
            return "fall measurement failed";
        case REK_G1_SEMANTIC_DUEL_HIT_MEASUREMENT_FAILED:
            return "hit measurement failed";
        case REK_G1_SEMANTIC_DUEL_COMBAT_FAILED:
            return "combat state transition failed";
        case REK_G1_SEMANTIC_DUEL_ARENA_RESET_FAILED:
            return "arena spawn reset failed";
        default:
            return "unknown semantic duel status";
    }
}

static const RekG1SemanticDuelRouteAsset* asset_by_id(
        const RekG1SemanticDuelRuntime* runtime,
        RekG1NativeRouteId route_id) {
    if (runtime == NULL || runtime->route_assets == NULL) return NULL;
    for (size_t index = 0; index < runtime->route_asset_count; index++) {
        if (runtime->route_assets[index].route_id == route_id) {
            return &runtime->route_assets[index];
        }
    }
    return NULL;
}

static int route_config_from_contract(
        const RekG1NativeMotionRoute* route,
        SonicMotionComposerNativeConfig* output) {
    if (route == NULL || output == NULL) return 0;
    *output = (SonicMotionComposerNativeConfig){
        .mirror = route->mirror,
        .loop = route->loop,
        .playback_speed = route->playback_speed,
        .start_frame = route->start_frame,
        .end_frame = route->end_frame,
        .blend_in_seconds = route->blend_in_seconds,
        .blend_out_seconds = route->blend_out_seconds,
        .yaw_blend = route->yaw_blend,
    };
    return 1;
}

static int mirror_table_valid(
        const SonicMotionComposerNativeMirrorTable* table) {
    if (table == NULL || table->source_indices == NULL
            || table->negate == NULL
            || table->source_index_count != GEAR_SONIC_ACTION_DIM
            || table->negate_count != GEAR_SONIC_ACTION_DIM) {
        return 0;
    }
    for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
        if (table->source_indices[index] >= GEAR_SONIC_ACTION_DIM
                || table->negate[index] > 1u) {
            return 0;
        }
    }
    return 1;
}

static int command_config_valid(const RekG1SemanticDuelConfig* config) {
    if (config == NULL
            || rek_g1_validate_input_timing(config->input_timing)
                != REK_G1_INPUT_ACCEPTED
            || config->command.controller_rate_hz
                != REK_G1_SEMANTIC_DUEL_CONTROLLER_RATE_HZ
            || !finite_f32(config->command.locomotion_speed_scale)
            || !finite_f32(config->command.command_yaw_rate_scale)
            || !finite_f32(config->command.heading_yaw_rate_scale)
            || !finite_f32(config->locomotion.settle_linear_speed)
            || !finite_f32(config->locomotion.settle_yaw_rate)
            || !finite_f32(config->locomotion.stop_brake_rate)
            || config->locomotion.settle_linear_speed < 0.0f
            || config->locomotion.settle_yaw_rate < 0.0f
            || config->locomotion.stop_brake_rate < 0.0f
            || config->locomotion.transition_settle > 1u
            || config->input_timing.elapsed_seconds
                != REK_G1_SEMANTIC_DUEL_CONTROL_DELTA_SECONDS
            || config->input_timing.yaw_ramp_seconds
                != REK_G1_SEMANTIC_DUEL_YAW_RAMP_SECONDS
            || config->command.locomotion_speed_scale != 1.0f
            || config->command.command_yaw_rate_scale != 1.0f
            || config->command.heading_yaw_rate_scale != 1.0f
            || config->locomotion.settle_linear_speed
                != REK_G1_SEMANTIC_DUEL_SETTLE_LINEAR_SPEED
            || config->locomotion.settle_yaw_rate
                != REK_G1_SEMANTIC_DUEL_SETTLE_YAW_RATE
            || config->locomotion.stop_brake_rate
                != REK_G1_SEMANTIC_DUEL_STOP_BRAKE_RATE
            || config->locomotion.transition_settle != 1u
            || config->composer_backends.quaternion_slerp == NULL
            || config->composer_backends.atan2_f == NULL
            || config->composer_backends.sin_cos_f == NULL
            || config->composer_backends.loop_entry_matcher == NULL
            || !mirror_table_valid(&config->mirror_table)
            || config->forgiveness_delta == NULL) {
        return 0;
    }
    return 1;
}

static int duel_valid(const GearSonicNativeDuelVector* duel) {
    if (duel == NULL || duel->model == NULL || duel->data == NULL
            || duel->arena_count == 0
            || duel->arena_count > SIZE_MAX / GEAR_SONIC_DUEL_FIGHTERS
            || duel->robot_count
                != duel->arena_count * GEAR_SONIC_DUEL_FIGHTERS
            || duel->controller.batch_size != duel->robot_count
            || !duel->controller.initialized || duel->controller.failed
            || duel->ort.batch_size != duel->robot_count
            || duel->heading_delta_wxyz == NULL
            || duel->dampened_rows == NULL
            || duel->resetting_rows == NULL
            || duel->reset_pending_arenas == NULL
            || duel->reset_completed_in_step_arenas == NULL
            || duel->reset_complete_not_before_time == NULL
            || !duel->spawn_prefixes_verified || duel->failed) {
        return 0;
    }
    for (size_t arena = 0; arena < duel->arena_count; arena++) {
        if (duel->data[arena] == NULL) return 0;
    }
    for (size_t row = 0; row < duel->robot_count; row++) {
        if (duel->dampened_rows[row] > 1u
                || duel->resetting_rows[row] > 1u) {
            return 0;
        }
    }
    for (size_t arena = 0; arena < duel->arena_count; arena++) {
        const uint8_t pending = duel->reset_pending_arenas[arena];
        const uint8_t completed =
            duel->reset_completed_in_step_arenas[arena];
        const double deadline =
            duel->reset_complete_not_before_time[arena];
        const size_t first_row = arena * GEAR_SONIC_DUEL_FIGHTERS;
        if (pending > 1u || completed > 1u) return 0;
        if (pending) {
            if (completed || !duel->resetting_rows[first_row]
                    || !duel->resetting_rows[first_row + 1u]
                    || !isfinite(deadline) || deadline < 0.0) {
                return 0;
            }
        } else if (duel->resetting_rows[first_row]
                || duel->resetting_rows[first_row + 1u]
                || deadline != 0.0) {
            return 0;
        }
    }
    return 1;
}

static int clip_shape_counts(
        const SonicMotionComposerNativeClip* clip,
        size_t* dof_count,
        size_t* root_count) {
    return clip != NULL && clip->frame_count > 0
        && checked_product(
            clip->frame_count,
            GEAR_SONIC_ACTION_DIM,
            dof_count)
        && checked_product(clip->frame_count, 4u, root_count);
}

static int route_asset_valid(
        const RekG1NativeMotionRoute* route,
        const RekG1SemanticDuelRouteAsset* asset,
        const SonicMotionComposerNativeConfig* config) {
    size_t dof_count = 0;
    size_t root_count = 0;
    if (route == NULL || asset == NULL || config == NULL
            || asset->route_id != route->id
            || !clip_shape_counts(&asset->clip, &dof_count, &root_count)
            || asset->clip.dof_position_mujoco == NULL
            || asset->clip.root_quaternion_wxyz == NULL
            || asset->clip.dof_position_count != dof_count
            || asset->clip.root_quaternion_count != root_count
            || asset->clip.frame_count != route->asset_frames
            || asset->clip.fps != route->asset_fps
            || (route->kind == REK_G1_NATIVE_ROUTE_DISCRETE_MOVE
                ? asset->configured_compositor_duration_ticks == 0u
                : asset->configured_compositor_duration_ticks != 0u)) {
        return 0;
    }
    for (size_t index = 0; index < dof_count; index++) {
        if (!finite_f32(asset->clip.dof_position_mujoco[index])) return 0;
    }
    for (size_t frame = 0; frame < asset->clip.frame_count; frame++) {
        double norm_squared = 0.0;
        for (size_t axis = 0; axis < 4; axis++) {
            const float value = asset->clip.root_quaternion_wxyz[frame * 4 + axis];
            if (!finite_f32(value)) return 0;
            norm_squared += (double)value * (double)value;
        }
        if (!isfinite(norm_squared)
                || fabs(sqrt(norm_squared) - 1.0) > 1e-4) {
            return 0;
        }
    }
    SonicMotionComposerNativeLayer layer = {0};
    return sonic_motion_composer_native_install_layer(
        &layer,
        &asset->clip,
        config,
        REK_G1_SEMANTIC_DUEL_CONTROLLER_RATE_HZ)
        == SONIC_MOTION_COMPOSER_NATIVE_OK;
}

static int fixed_idle_matches(
        const GearSonicNativeDuelVector* duel,
        const RekG1SemanticDuelRouteAsset* idle) {
    if (duel == NULL || idle == NULL
            || duel->fixed_motion.dof_position_mujoco == NULL
            || duel->fixed_motion.root_rotation_xyzw == NULL
            || duel->fixed_motion.root_position_m == NULL
            || duel->fixed_motion.frames != idle->clip.frame_count
            || duel->fixed_motion.loop != 1) {
        return 0;
    }
    size_t dof_count = 0;
    size_t root_count = 0;
    if (!clip_shape_counts(&idle->clip, &dof_count, &root_count)) return 0;
    for (size_t index = 0; index < dof_count; index++) {
        if (duel->fixed_motion.dof_position_mujoco[index]
                != idle->clip.dof_position_mujoco[index]) {
            return 0;
        }
    }
    for (size_t frame = 0; frame < idle->clip.frame_count; frame++) {
        const float* xyzw = duel->fixed_motion.root_rotation_xyzw + frame * 4;
        const float* wxyz = idle->clip.root_quaternion_wxyz + frame * 4;
        if (xyzw[0] != wxyz[1] || xyzw[1] != wxyz[2]
                || xyzw[2] != wxyz[3] || xyzw[3] != wxyz[0]) {
            return 0;
        }
    }
    return 1;
}

static void* checked_calloc(size_t count, size_t element_size) {
    size_t bytes = 0;
    if (!checked_product(count, element_size, &bytes) || bytes == 0) {
        return NULL;
    }
    return calloc(1, bytes);
}

static int allocate_runtime(RekG1SemanticDuelRuntime* runtime) {
    size_t reference_rows = 0;
    size_t dof_values = 0;
    size_t root_values = 0;
    if (!checked_product(
            runtime->robot_count,
            REK_G1_SEMANTIC_DUEL_REFERENCE_ROWS,
            &reference_rows)
            || !checked_product(
                reference_rows, GEAR_SONIC_ACTION_DIM, &dof_values)
            || !checked_product(reference_rows, 4u, &root_values)) {
        return 0;
    }
    runtime->composers = checked_calloc(
        runtime->robot_count, sizeof(*runtime->composers));
    runtime->scratch_composers = checked_calloc(
        runtime->robot_count, sizeof(*runtime->scratch_composers));
    runtime->locomotion_states = checked_calloc(
        runtime->robot_count, sizeof(*runtime->locomotion_states));
    runtime->scratch_locomotion_states = checked_calloc(
        runtime->robot_count, sizeof(*runtime->scratch_locomotion_states));
    runtime->effective_velocity = checked_calloc(
        runtime->robot_count, sizeof(*runtime->effective_velocity));
    runtime->scratch_effective_velocity = checked_calloc(
        runtime->robot_count, sizeof(*runtime->scratch_effective_velocity));
    runtime->active_route_ids = checked_calloc(
        runtime->robot_count, sizeof(*runtime->active_route_ids));
    runtime->scratch_active_route_ids = checked_calloc(
        runtime->robot_count, sizeof(*runtime->scratch_active_route_ids));
    runtime->velocity_samples = checked_calloc(
        runtime->robot_count, sizeof(*runtime->velocity_samples));
    runtime->entity_observations = checked_calloc(
        runtime->robot_count, sizeof(*runtime->entity_observations));
    runtime->combat_states = checked_calloc(
        runtime->duel->arena_count, sizeof(*runtime->combat_states));
    runtime->scratch_combat_states = checked_calloc(
        runtime->duel->arena_count,
        sizeof(*runtime->scratch_combat_states));
    runtime->fall_states = checked_calloc(
        runtime->robot_count, sizeof(*runtime->fall_states));
    runtime->scratch_fall_states = checked_calloc(
        runtime->robot_count, sizeof(*runtime->scratch_fall_states));
    runtime->fall_measurements = checked_calloc(
        runtime->robot_count, sizeof(*runtime->fall_measurements));
    runtime->scratch_fall_measurements = checked_calloc(
        runtime->robot_count, sizeof(*runtime->scratch_fall_measurements));
    runtime->fall_events = checked_calloc(
        runtime->robot_count, sizeof(*runtime->fall_events));
    runtime->scratch_fall_events = checked_calloc(
        runtime->robot_count, sizeof(*runtime->scratch_fall_events));
    runtime->substep_fall_events = checked_calloc(
        runtime->robot_count, sizeof(*runtime->substep_fall_events));
    runtime->fight_signals = checked_calloc(
        runtime->duel->arena_count, sizeof(*runtime->fight_signals));
    runtime->referee_calls = checked_calloc(
        runtime->duel->arena_count, sizeof(*runtime->referee_calls));
    runtime->score_deltas = checked_calloc(
        runtime->robot_count, sizeof(*runtime->score_deltas));
    runtime->attributed_contact_counts = checked_calloc(
        runtime->duel->arena_count,
        sizeof(*runtime->attributed_contact_counts));
    runtime->scored_contact_counts = checked_calloc(
        runtime->duel->arena_count,
        sizeof(*runtime->scored_contact_counts));
    runtime->arena_reset_events = checked_calloc(
        runtime->duel->arena_count,
        sizeof(*runtime->arena_reset_events));
    runtime->arena_input_reset_events = checked_calloc(
        runtime->duel->arena_count,
        sizeof(*runtime->arena_input_reset_events));
    runtime->arena_terminals = checked_calloc(
        runtime->duel->arena_count,
        sizeof(*runtime->arena_terminals));
    runtime->pending_episode_resets = checked_calloc(
        runtime->duel->arena_count,
        sizeof(*runtime->pending_episode_resets));
    runtime->forgiveness_deltas = checked_calloc(
        runtime->robot_count, sizeof(*runtime->forgiveness_deltas));
    runtime->scratch_heading_delta_wxyz = checked_calloc(
        runtime->robot_count * 4u,
        sizeof(*runtime->scratch_heading_delta_wxyz));
    runtime->reference_dof_position = checked_calloc(
        dof_values, sizeof(*runtime->reference_dof_position));
    runtime->reference_dof_next_position = checked_calloc(
        dof_values, sizeof(*runtime->reference_dof_next_position));
    runtime->reference_root_rotation_xyzw = checked_calloc(
        root_values, sizeof(*runtime->reference_root_rotation_xyzw));
    return runtime->composers != NULL
        && runtime->scratch_composers != NULL
        && runtime->locomotion_states != NULL
        && runtime->scratch_locomotion_states != NULL
        && runtime->effective_velocity != NULL
        && runtime->scratch_effective_velocity != NULL
        && runtime->active_route_ids != NULL
        && runtime->scratch_active_route_ids != NULL
        && runtime->velocity_samples != NULL
        && runtime->entity_observations != NULL
        && runtime->combat_states != NULL
        && runtime->scratch_combat_states != NULL
        && runtime->fall_states != NULL
        && runtime->scratch_fall_states != NULL
        && runtime->fall_measurements != NULL
        && runtime->scratch_fall_measurements != NULL
        && runtime->fall_events != NULL
        && runtime->scratch_fall_events != NULL
        && runtime->substep_fall_events != NULL
        && runtime->fight_signals != NULL
        && runtime->referee_calls != NULL
        && runtime->score_deltas != NULL
        && runtime->attributed_contact_counts != NULL
        && runtime->scored_contact_counts != NULL
        && runtime->arena_reset_events != NULL
        && runtime->arena_input_reset_events != NULL
        && runtime->arena_terminals != NULL
        && runtime->pending_episode_resets != NULL
        && runtime->forgiveness_deltas != NULL
        && runtime->scratch_heading_delta_wxyz != NULL
        && runtime->reference_dof_position != NULL
        && runtime->reference_dof_next_position != NULL
        && runtime->reference_root_rotation_xyzw != NULL;
}

RekG1SemanticDuelStatus rek_g1_semantic_duel_open(
        RekG1SemanticDuelRuntime* runtime,
        GearSonicNativeDuelVector* duel,
        const RekG1NativeMotionRouteTable* motion_routes,
        const RekG1SemanticDuelRouteAsset* route_assets,
        size_t route_asset_count,
        const RekG1SemanticDuelConfig* config,
        char* error,
        size_t error_capacity) {
    if (runtime == NULL || duel == NULL || motion_routes == NULL
            || route_assets == NULL || config == NULL) {
        set_error(error, error_capacity, "open semantic duel", "null argument");
        return REK_G1_SEMANTIC_DUEL_NULL_ARGUMENT;
    }
    if (runtime->initialized) {
        set_error(
            error, error_capacity, "open semantic duel",
            "runtime is already initialized");
        return REK_G1_SEMANTIC_DUEL_NOT_READY;
    }
    memset(runtime, 0, sizeof(*runtime));
    if (!duel_valid(duel)) {
        return latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_INVALID_DUEL,
            error, error_capacity, "open semantic duel", "invalid duel state");
    }
    if (!rek_g1_native_validate_static_motion_routes(motion_routes)
            || route_asset_count != REK_G1_STATIC_ROUTE_COUNT) {
        return latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_INVALID_ROUTE_ASSET,
            error, error_capacity, "open semantic duel", "route table mismatch");
    }
    if (!rek_g1_validate_strike_catalog(
            rek_g1_current_build_strike_catalog())) {
        return latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_COMBAT_FAILED,
            error, error_capacity,
            "open semantic duel", "strike catalog mismatch");
    }
    if (!command_config_valid(config)) {
        return latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_INVALID_CONFIG,
            error, error_capacity, "open semantic duel", "config is incomplete");
    }

    runtime->duel = duel;
    runtime->motion_routes = motion_routes;
    runtime->route_assets = route_assets;
    runtime->route_asset_count = route_asset_count;
    runtime->robot_count = duel->robot_count;
    runtime->config = *config;
    for (size_t index = 0; index < motion_routes->count; index++) {
        const RekG1NativeMotionRoute* route = &motion_routes->routes[index];
        const RekG1SemanticDuelRouteAsset* asset = asset_by_id(runtime, route->id);
        if ((size_t)route->id >= REK_G1_STATIC_ROUTE_COUNT
                || !route_config_from_contract(
                    route, &runtime->route_configs[(size_t)route->id])
                || !route_asset_valid(
                    route,
                    asset,
                    &runtime->route_configs[(size_t)route->id])) {
            return latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_INVALID_ROUTE_ASSET,
                error, error_capacity, "open semantic duel", "route asset mismatch");
        }
    }
    const RekG1SemanticDuelRouteAsset* idle = asset_by_id(
        runtime, REK_G1_NATIVE_IDLE);
    if (!fixed_idle_matches(duel, idle)) {
        return latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_INVALID_FIXED_IDLE,
            error, error_capacity, "open semantic duel", "fixed idle mismatch");
    }
    if (!allocate_runtime(runtime)) {
        rek_g1_semantic_duel_close(runtime);
        set_error(error, error_capacity, "open semantic duel", "allocation failed");
        return REK_G1_SEMANTIC_DUEL_ALLOCATION_FAILED;
    }
    const RekG1FallMujocoStatus fall_adapter_status =
        rek_g1_fall_mujoco_open(
            &runtime->fall_adapter,
            duel,
            error,
            error_capacity);
    if (fall_adapter_status != REK_G1_FALL_MUJOCO_OK) {
        rek_g1_semantic_duel_close(runtime);
        set_error(
            error,
            error_capacity,
            "open semantic duel fall measurement",
            rek_g1_fall_mujoco_status_string(fall_adapter_status));
        runtime->failed = 1u;
        runtime->last_status =
            REK_G1_SEMANTIC_DUEL_FALL_MEASUREMENT_FAILED;
        return runtime->last_status;
    }
    runtime->fall_adapter_open = 1u;
    const RekG1HitMujocoStatus hit_adapter_status =
        rek_g1_hit_mujoco_open(
            &runtime->hit_adapter,
            duel,
            error,
            error_capacity);
    if (hit_adapter_status != REK_G1_HIT_MUJOCO_OK) {
        rek_g1_semantic_duel_close(runtime);
        set_error(
            error,
            error_capacity,
            "open semantic duel hit measurement",
            rek_g1_hit_mujoco_status_string(hit_adapter_status));
        runtime->failed = 1u;
        runtime->last_status =
            REK_G1_SEMANTIC_DUEL_HIT_MEASUREMENT_FAILED;
        return runtime->last_status;
    }
    runtime->hit_adapter_open = 1u;
    runtime->hit_candidate_capacity =
        runtime->hit_adapter.candidate_scratch_capacity;
    runtime->hit_candidates = checked_calloc(
        runtime->hit_candidate_capacity,
        sizeof(*runtime->hit_candidates));
    runtime->hit_contacts = checked_calloc(
        runtime->hit_candidate_capacity,
        sizeof(*runtime->hit_contacts));
    if (runtime->hit_candidates == NULL || runtime->hit_contacts == NULL) {
        rek_g1_semantic_duel_close(runtime);
        set_error(
            error, error_capacity,
            "open semantic duel hit measurement", "allocation failed");
        runtime->failed = 1u;
        runtime->last_status = REK_G1_SEMANTIC_DUEL_ALLOCATION_FAILED;
        return runtime->last_status;
    }
    runtime->initialized = 1u;
    runtime->last_status = REK_G1_SEMANTIC_DUEL_OK;
    runtime->failed = 0u;
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return REK_G1_SEMANTIC_DUEL_OK;
}

static int action_table_matches_assets(
        const RekG1SemanticDuelRuntime* runtime,
        const RekG1PufferActionTable* action_table) {
    if (runtime == NULL || action_table == NULL
            || rek_g1_puffer_validate_table(action_table) != REK_G1_PUFFER_OK) {
        return 0;
    }
    for (uint16_t index = 0;
            index < action_table->move_registry_count; index++) {
        const RekG1NativeMotionRoute* route =
            rek_g1_native_discrete_move_route(
                runtime->motion_routes, action_table->move_indices[index]);
        const RekG1SemanticDuelRouteAsset* asset = route == NULL
            ? NULL : asset_by_id(runtime, route->id);
        if (asset == NULL || asset->configured_compositor_duration_ticks
                != action_table->move_duration_ticks[index]) {
            return 0;
        }
    }
    return 1;
}

static int callback_arguments_valid(
        const RekG1SemanticDuelRuntime* runtime,
        const RekG1NativeMotionRouteTable* routes,
        const RekG1PufferActionTable* action_table,
        size_t environment_count,
        const RekG1RuntimeFacts* facts,
        const void* observations,
        size_t observation_stride_bytes,
        const float* rewards,
        const float* terminals) {
    return runtime != NULL && runtime->initialized
        && routes == runtime->motion_routes
        && action_table_matches_assets(runtime, action_table)
        && environment_count == runtime->robot_count
        && facts != NULL && observations != NULL
        && observation_stride_bytes >= sizeof(RekG1SemanticDuelObservation)
        && rewards != NULL && terminals != NULL;
}

static int row_arena_fighter(
        const RekG1SemanticDuelRuntime* runtime,
        size_t row,
        size_t* arena,
        size_t* fighter) {
    if (runtime == NULL || row >= runtime->robot_count
            || arena == NULL || fighter == NULL) return 0;
    *arena = row / GEAR_SONIC_DUEL_FIGHTERS;
    *fighter = row % GEAR_SONIC_DUEL_FIGHTERS;
    return *arena < runtime->duel->arena_count;
}

static int double_to_float(double value, float* output) {
    if (output == NULL || !isfinite(value)) return 0;
    const float converted = (float)value;
    if (!isfinite(converted)) return 0;
    *output = converted;
    return 1;
}

static int gather_velocity_sample(
        RekG1SemanticDuelRuntime* runtime,
        size_t row,
        RekG1NativeBaseVelocitySample* output) {
    size_t arena = 0;
    size_t fighter = 0;
    if (output == NULL || !row_arena_fighter(
            runtime, row, &arena, &fighter)) return 0;
    const GearSonicDuelFighterMap* map = &runtime->duel->fighters[fighter];
    mjData* data = runtime->duel->data[arena];
    double velocity[6] = {0};
    mj_objectVelocity(
        runtime->duel->model,
        data,
        mjOBJ_BODY,
        map->root_body_id,
        velocity,
        1);
    RekG1NativeBaseVelocitySample local = {.available = 1u};
    if (!double_to_float(velocity[0], &local.angular_velocity_local.forward)
            || !double_to_float(velocity[1], &local.angular_velocity_local.strafe)
            || !double_to_float(velocity[2], &local.angular_velocity_local.yaw)
            || !double_to_float(velocity[3], &local.linear_velocity_local.forward)
            || !double_to_float(velocity[4], &local.linear_velocity_local.strafe)
            || !double_to_float(velocity[5], &local.linear_velocity_local.yaw)) {
        return 0;
    }
    *output = local;
    return 1;
}

static int gather_entity_observation(
        RekG1SemanticDuelRuntime* runtime,
        size_t row,
        RekG1SemanticDuelEntityObservation* output) {
    size_t arena = 0;
    size_t fighter = 0;
    if (output == NULL || !row_arena_fighter(
            runtime, row, &arena, &fighter)) return 0;
    const GearSonicDuelFighterMap* map = &runtime->duel->fighters[fighter];
    mjData* data = runtime->duel->data[arena];
    double velocity[6] = {0};
    RekG1SemanticDuelEntityObservation local = {0};
    mj_objectVelocity(
        runtime->duel->model,
        data,
        mjOBJ_BODY,
        map->root_body_id,
        velocity,
        1);
    for (size_t axis = 0; axis < 3; axis++) {
        if (!double_to_float(
                data->qpos[map->root_qpos_address + (int)axis],
                &local.root_position_world[axis])
                || !double_to_float(
                    velocity[3 + axis], &local.linear_velocity_local[axis])
                || !double_to_float(
                    velocity[axis], &local.angular_velocity_local[axis])) {
            return 0;
        }
    }
    for (size_t axis = 0; axis < 4; axis++) {
        if (!double_to_float(
                data->qpos[map->root_qpos_address + 3 + (int)axis],
                &local.root_quaternion_wxyz[axis])) {
            return 0;
        }
    }
    for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
        if (!double_to_float(
                data->qpos[map->qpos_addresses[index]],
                &local.joint_position_mujoco[index])
                || !double_to_float(
                    data->qvel[map->qvel_addresses[index]],
                    &local.joint_velocity_mujoco[index])) {
            return 0;
        }
    }
    *output = local;
    return 1;
}

static int fall_state_allows_semantic_motion(
        RekG1SemanticDuelRuntime* runtime,
        size_t row,
        char* error,
        size_t error_capacity) {
    if (runtime == NULL || row >= runtime->robot_count) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_FALL_MEASUREMENT_FAILED,
            error, error_capacity,
            "check measured fall state", "row is unavailable");
        return 0;
    }
    const RekG1FallPhase phase = runtime->fall_states[row].phase;
    if (phase != REK_G1_FALL_UPRIGHT
            && phase != REK_G1_FALL_FALLING
            && phase != REK_G1_FALL_FALLEN) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_FALL_MEASUREMENT_FAILED,
            error, error_capacity,
            "check measured fall state", "fall phase is invalid");
        return 0;
    }
    return 1;
}

static int row_policy_suspended(
        const RekG1SemanticDuelRuntime* runtime, size_t row) {
    return runtime != NULL && runtime->duel != NULL
        && row < runtime->robot_count
        && (runtime->duel->dampened_rows[row]
            || runtime->duel->resetting_rows[row]);
}

static int initialize_fall_measurements(
        RekG1SemanticDuelRuntime* runtime,
        char* error,
        size_t error_capacity) {
    if (rek_g1_fall_mujoco_calibrate_reset(
            &runtime->fall_adapter,
            error,
            error_capacity) != REK_G1_FALL_MUJOCO_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_FALL_MEASUREMENT_FAILED,
            error, error_capacity,
            "calibrate measured fall state",
            rek_g1_fall_mujoco_status_string(
                runtime->fall_adapter.last_status));
        return 0;
    }
    for (size_t row = 0u; row < runtime->robot_count; row++) {
        if (rek_g1_fall_state_init(
                &REK_G1_FALL_CONFIG_F84F1874,
                &runtime->fall_states[row]) != REK_G1_FALL_OK) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_FALL_MEASUREMENT_FAILED,
                error, error_capacity,
                "initialize measured fall state", "state config rejected");
            return 0;
        }
        if (rek_g1_fall_mujoco_sample(
                &runtime->fall_adapter,
                row,
                REK_G1_SEMANTIC_DUEL_CONTROL_DELTA_SECONDS,
                REK_G1_SEMANTIC_DUEL_PROVISIONAL_CAN_GET_UP,
                &runtime->fall_measurements[row],
                error,
                error_capacity) != REK_G1_FALL_MUJOCO_OK) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_FALL_MEASUREMENT_FAILED,
                error, error_capacity,
                "sample reset fall state",
                rek_g1_fall_mujoco_status_string(
                    runtime->fall_adapter.last_status));
            return 0;
        }
        runtime->fall_events[row] = REK_G1_FALL_EVENT_NONE;
    }
    return 1;
}

static int stage_fall_measurement(
        RekG1SemanticDuelRuntime* runtime,
        size_t row,
        float fixed_delta_seconds,
        char* error,
        size_t error_capacity) {
    RekG1FallMujocoMeasurement measurement = {0};
    if (rek_g1_fall_mujoco_sample(
            &runtime->fall_adapter,
            row,
            fixed_delta_seconds,
            REK_G1_SEMANTIC_DUEL_PROVISIONAL_CAN_GET_UP,
            &measurement,
            error,
            error_capacity) != REK_G1_FALL_MUJOCO_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_FALL_MEASUREMENT_FAILED,
            error, error_capacity,
            "sample post-step fall state",
            rek_g1_fall_mujoco_status_string(
                runtime->fall_adapter.last_status));
        return 0;
    }
    RekG1FallStepResult result = {0};
    if (rek_g1_fall_state_step(
            &REK_G1_FALL_CONFIG_F84F1874,
            &runtime->scratch_fall_states[row],
            &measurement.fall_sample,
            &result) != REK_G1_FALL_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_FALL_MEASUREMENT_FAILED,
            error, error_capacity,
            "advance measured fall state", "state transition rejected");
        return 0;
    }
    runtime->scratch_fall_states[row] = result.next_state;
    runtime->scratch_fall_measurements[row] = measurement;
    runtime->substep_fall_events[row] = result.events;
    runtime->scratch_fall_events[row] |= result.events;
    return 1;
}

static int populate_fall_observations(
        RekG1SemanticDuelRuntime* runtime,
        const RekG1FallMujocoMeasurement* measurements,
        const RekG1FallState* states,
        const uint32_t* events) {
    if (runtime == NULL || measurements == NULL || states == NULL
            || events == NULL) {
        return 0;
    }
    const uint32_t valid_events = REK_G1_FALL_EVENT_FALLING_STARTED
        | REK_G1_FALL_EVENT_FALLING_CLEARED
        | REK_G1_FALL_EVENT_BECAME_FALLEN
        | REK_G1_FALL_EVENT_RESET_TIMEOUT_DUE;
    for (size_t row = 0u; row < runtime->robot_count; row++) {
        const RekG1FallMujocoMeasurement* measurement = &measurements[row];
        const RekG1FallSample* sample = &measurement->fall_sample;
        const RekG1FallState* state = &states[row];
        if (sample->tracking_active > 1u
                || sample->both_feet_off_floor > 1u
                || sample->has_foot_body_contact > 1u
                || sample->can_get_up > 1u
                || measurement->left_foot_body_contact > 1u
                || measurement->right_foot_body_contact > 1u
                || sample->has_foot_body_contact
                    != (uint8_t)(measurement->left_foot_body_contact
                        || measurement->right_foot_body_contact)
                || sample->both_feet_off_floor
                    == sample->has_foot_body_contact
                || (events[row] & ~valid_events) != 0u
                || (state->phase != REK_G1_FALL_UPRIGHT
                    && state->phase != REK_G1_FALL_FALLING
                    && state->phase != REK_G1_FALL_FALLEN)
                || !finite_f32(sample->tilt_degrees)
                || !finite_f32(sample->pelvis_height_ratio)
                || !finite_f32(state->fallen_hold_seconds)
                || !finite_f32(state->fallen_elapsed_seconds)
                || !finite_f32(state->fallen_timer_seconds)
                || !finite_f32(state->reset_grace_remaining_seconds)) {
            return 0;
        }
        const float nonfoot =
            (float)sample->distinct_nonfoot_body_contact_count;
        if ((uint32_t)nonfoot
                != sample->distinct_nonfoot_body_contact_count) {
            return 0;
        }
        runtime->entity_observations[row].fall =
            (RekG1SemanticDuelFallObservation){
                .tracking_active = (float)sample->tracking_active,
                .tilt_degrees = sample->tilt_degrees,
                .pelvis_height_ratio = sample->pelvis_height_ratio,
                .both_feet_off_floor =
                    (float)sample->both_feet_off_floor,
                .left_foot_body_contact =
                    (float)measurement->left_foot_body_contact,
                .right_foot_body_contact =
                    (float)measurement->right_foot_body_contact,
                .distinct_nonfoot_body_contact_count = nonfoot,
                .build_pinned_can_get_up = (float)sample->can_get_up,
                .phase = (float)state->phase,
                .fallen_hold_seconds = state->fallen_hold_seconds,
                .fallen_elapsed_seconds = state->fallen_elapsed_seconds,
                .fallen_timer_seconds = state->fallen_timer_seconds,
                .reset_grace_remaining_seconds =
                    state->reset_grace_remaining_seconds,
                .recovery_armed = (float)state->recovery_armed,
                .events = (float)events[row],
            };
    }
    return 1;
}

static int u32_to_observation(uint32_t value, float* output) {
    if (output == NULL || value > UINT32_C(16777216)) return 0;
    *output = (float)value;
    return 1;
}

static int i32_to_observation(int32_t value, float* output) {
    if (output == NULL || value < -16777216 || value > 16777216) return 0;
    *output = (float)value;
    return 1;
}

static int write_fight_observation(
        const RekG1SemanticDuelRuntime* runtime,
        const RekG1CombatArenaState* combat_states,
        const uint32_t* fall_events,
        size_t row,
        RekG1SemanticDuelFightObservation* output) {
    size_t arena = 0u;
    size_t fighter = 0u;
    if (runtime == NULL || combat_states == NULL || fall_events == NULL
            || output == NULL
            || !row_arena_fighter(runtime, row, &arena, &fighter)) {
        return 0;
    }
    const size_t opponent = 1u - fighter;
    const size_t arena_base = arena * GEAR_SONIC_DUEL_FIGHTERS;
    const RekG1FightState* fight = &combat_states[arena].fight;
    RekG1SemanticDuelFightObservation value = {
        .self_fighter_index = (float)fighter,
        .phase = (float)fight->phase,
        .current_round_is_redo = (float)fight->current_round_is_redo,
        .round_duration_seconds = fight->round_duration_seconds,
        .time_remaining_seconds = fight->time_remaining_seconds,
        .self_last_struck_valid =
            (float)fight->last_struck_valid[fighter],
        .opponent_last_struck_valid =
            (float)fight->last_struck_valid[opponent],
        .self_last_struck_age_seconds =
            fight->last_struck_age_seconds[fighter],
        .opponent_last_struck_age_seconds =
            fight->last_struck_age_seconds[opponent],
        .self_last_struck_speed = fight->last_struck_speed[fighter],
        .opponent_last_struck_speed = fight->last_struck_speed[opponent],
        .self_fall_classification =
            (float)fight->fall_classification[fighter],
        .opponent_fall_classification =
            (float)fight->fall_classification[opponent],
        .self_count_active = (float)fight->count_active[fighter],
        .opponent_count_active = (float)fight->count_active[opponent],
        .self_count_is_slip = (float)fight->count_is_slip[fighter],
        .opponent_count_is_slip = (float)fight->count_is_slip[opponent],
        .count_elapsed_seconds = fight->count_elapsed_seconds,
        .count_duration_seconds = fight->count_duration_seconds,
        .round_result = (float)fight->round_result,
        .knockout_occurred = (float)fight->knockout_occurred,
        .fight_result = (float)fight->fight_result,
    };
    if (!u32_to_observation(
            fight->current_round_number, &value.current_round_number)
            || !i32_to_observation(
                fight->clean_hits[fighter], &value.self_clean_hits)
            || !i32_to_observation(
                fight->clean_hits[opponent], &value.opponent_clean_hits)
            || !u32_to_observation(
                fight->falls[fighter], &value.self_falls)
            || !u32_to_observation(
                fight->falls[opponent], &value.opponent_falls)
            || !u32_to_observation(
                fight->rounds_won[fighter], &value.self_rounds_won)
            || !u32_to_observation(
                fight->rounds_won[opponent], &value.opponent_rounds_won)
            || !i32_to_observation(
                fight->round_winner_index, &value.round_winner_index)
            || !i32_to_observation(
                fight->fight_winner_index, &value.fight_winner_index)
            || !u32_to_observation(
                runtime->fight_signals[arena], &value.tick_signals)
            || !u32_to_observation(
                runtime->referee_calls[arena], &value.tick_referee_calls)
            || !i32_to_observation(
                runtime->score_deltas[arena_base + fighter],
                &value.tick_self_score_delta)
            || !i32_to_observation(
                runtime->score_deltas[arena_base + opponent],
                &value.tick_opponent_score_delta)
            || !u32_to_observation(
                fall_events[arena_base + fighter],
                &value.tick_self_fall_events)
            || !u32_to_observation(
                fall_events[arena_base + opponent],
                &value.tick_opponent_fall_events)
            || !u32_to_observation(
                runtime->attributed_contact_counts[arena],
                &value.tick_attributed_contact_count)
            || !u32_to_observation(
                runtime->scored_contact_counts[arena],
                &value.tick_scored_contact_count)) {
        return 0;
    }
    *output = value;
    return 1;
}

static int write_observations(
        RekG1SemanticDuelRuntime* runtime,
        const SonicMotionComposerNative* composers,
        const RekG1NativeLocomotionState* locomotion,
        const RekG1NativeVelocityCommand* velocity,
        const RekG1NativeRouteId* active_routes,
        const double* heading,
        const RekG1CombatArenaState* combat_states,
        const uint32_t* fall_events,
        void* observations,
        size_t observation_stride_bytes) {
    for (size_t row = 0; row < runtime->robot_count; row++) {
        const size_t arena_base =
            (row / GEAR_SONIC_DUEL_FIGHTERS) * GEAR_SONIC_DUEL_FIGHTERS;
        const size_t opponent = arena_base
            + (row % GEAR_SONIC_DUEL_FIGHTERS == GEAR_SONIC_DUEL_PLAYER
                ? GEAR_SONIC_DUEL_OPPONENT : GEAR_SONIC_DUEL_PLAYER);
        RekG1SemanticDuelObservation value = {
            .self = runtime->entity_observations[row],
            .opponent = runtime->entity_observations[opponent],
            .effective_forward = velocity[row].forward,
            .effective_strafe = velocity[row].strafe,
            .effective_yaw = velocity[row].yaw,
            .active_route_id = (float)active_routes[row],
            .locomotion_active = (float)locomotion[row].locomotion_active,
            .transition_settling = (float)locomotion[row].transition_settling,
            .action_playing = composers[row].action_playing ? 1.0f : 0.0f,
            .composer_busy = composer_is_busy(&composers[row]) ? 1.0f : 0.0f,
        };
        for (size_t axis = 0; axis < 4; axis++) {
            if (!double_to_float(
                    heading[row * 4 + axis],
                    &value.self_heading_delta_wxyz[axis])) {
                return 0;
            }
        }
        if (!write_fight_observation(
                runtime, combat_states, fall_events, row, &value.fight)) {
            return 0;
        }
        memcpy(
            (uint8_t*)observations + row * observation_stride_bytes,
            &value,
            sizeof(value));
    }
    return 1;
}

static int gather_all_entities(RekG1SemanticDuelRuntime* runtime) {
    for (size_t row = 0; row < runtime->robot_count; row++) {
        if (!gather_entity_observation(
                runtime, row, &runtime->entity_observations[row])) {
            return 0;
        }
    }
    return 1;
}

static void clear_tick_combat_outputs(RekG1SemanticDuelRuntime* runtime) {
    memset(
        runtime->scratch_fall_events,
        0,
        runtime->robot_count * sizeof(*runtime->scratch_fall_events));
    memset(
        runtime->substep_fall_events,
        0,
        runtime->robot_count * sizeof(*runtime->substep_fall_events));
    memset(
        runtime->fight_signals,
        0,
        runtime->duel->arena_count * sizeof(*runtime->fight_signals));
    memset(
        runtime->referee_calls,
        0,
        runtime->duel->arena_count * sizeof(*runtime->referee_calls));
    memset(
        runtime->score_deltas,
        0,
        runtime->robot_count * sizeof(*runtime->score_deltas));
    memset(
        runtime->attributed_contact_counts,
        0,
        runtime->duel->arena_count
            * sizeof(*runtime->attributed_contact_counts));
    memset(
        runtime->scored_contact_counts,
        0,
        runtime->duel->arena_count
            * sizeof(*runtime->scored_contact_counts));
    memset(
        runtime->arena_reset_events,
        0,
        runtime->duel->arena_count * sizeof(*runtime->arena_reset_events));
    memset(
        runtime->arena_input_reset_events,
        0,
        runtime->duel->arena_count
            * sizeof(*runtime->arena_input_reset_events));
    memset(
        runtime->arena_terminals,
        0,
        runtime->duel->arena_count * sizeof(*runtime->arena_terminals));
}

static int initialize_combat_states(
        RekG1SemanticDuelRuntime* runtime,
        char* error,
        size_t error_capacity) {
    if (rek_g1_hit_mujoco_reset(
            &runtime->hit_adapter,
            error,
            error_capacity) != REK_G1_HIT_MUJOCO_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_HIT_MEASUREMENT_FAILED,
            error, error_capacity,
            "reset semantic duel hit measurement",
            rek_g1_hit_mujoco_status_string(
                runtime->hit_adapter.last_status));
        return 0;
    }
    for (size_t arena = 0u; arena < runtime->duel->arena_count; arena++) {
        if (rek_g1_combat_arena_init(&runtime->combat_states[arena])
                != REK_G1_COMBAT_TICK_OK) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_COMBAT_FAILED,
                error, error_capacity,
                "reset semantic duel combat", "fight initialization failed");
            return 0;
        }
    }
    memset(
        runtime->pending_episode_resets,
        0,
        runtime->duel->arena_count
            * sizeof(*runtime->pending_episode_resets));
    clear_tick_combat_outputs(runtime);
    return 1;
}

static RekG1RuntimeFacts episode_start_facts(
        const RekG1SemanticDuelRuntime* runtime) {
    return (RekG1RuntimeFacts){
        .timing = runtime->config.input_timing,
        .translation_transition_settled = 1u,
        .action_busy = 0u,
        .recovery_active = 0u,
    };
}

int rek_g1_semantic_duel_reset_batch(
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
    RekG1SemanticDuelRuntime* runtime = (RekG1SemanticDuelRuntime*)context;
    if (!callback_arguments_valid(
            runtime, motion_routes, action_table, environment_count,
            facts_out, observations, observation_stride_bytes,
            rewards, terminals)) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_PROTOCOL_INVALID,
            error, error_capacity, "reset semantic duel", "invalid callback arguments");
        return 0;
    }
    runtime->failed = 0u;
    runtime->ready = 0u;
    if (!gear_sonic_native_duel_reset(
            runtime->duel, error, error_capacity)) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_PHYSICS_FAILED,
            error, error_capacity, "reset semantic duel", "duel reset failed");
        return 0;
    }
    if (!initialize_fall_measurements(runtime, error, error_capacity)) {
        return 0;
    }
    if (!initialize_combat_states(runtime, error, error_capacity)) {
        return 0;
    }
    const RekG1SemanticDuelRouteAsset* idle = asset_by_id(
        runtime, REK_G1_NATIVE_IDLE);
    const SonicMotionComposerNativeConfig* idle_config =
        &runtime->route_configs[REK_G1_NATIVE_IDLE];
    for (size_t row = 0; row < runtime->robot_count; row++) {
        SonicMotionComposerNativeStatus status =
            sonic_motion_composer_native_init(
                &runtime->composers[row],
                REK_G1_SEMANTIC_DUEL_CONTROLLER_RATE_HZ,
                &runtime->config.composer_backends);
        if (status == SONIC_MOTION_COMPOSER_NATIVE_OK) {
            status = sonic_motion_composer_native_play_action_immediate(
                &runtime->composers[row], &idle->clip, idle_config);
        }
        if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_COMPOSER_FAILED,
                error, error_capacity,
                "reset semantic duel",
                sonic_motion_composer_native_status_string(status));
            return 0;
        }
        runtime->locomotion_states[row] = (RekG1NativeLocomotionState){0};
        runtime->effective_velocity[row] = (RekG1NativeVelocityCommand){0};
        runtime->active_route_ids[row] = REK_G1_NATIVE_IDLE;
    }
    if (!gather_all_entities(runtime)
            || !populate_fall_observations(
                runtime,
                runtime->fall_measurements,
                runtime->fall_states,
                runtime->fall_events)
            || !write_observations(
                runtime,
                runtime->composers,
                runtime->locomotion_states,
                runtime->effective_velocity,
                runtime->active_route_ids,
                runtime->duel->heading_delta_wxyz,
                runtime->combat_states,
                runtime->fall_events,
                observations,
                observation_stride_bytes)) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_OBSERVATION_FAILED,
            error, error_capacity, "reset semantic duel", "state gather failed");
        return 0;
    }
    for (size_t row = 0; row < runtime->robot_count; row++) {
        facts_out[row] = episode_start_facts(runtime);
        rewards[row] = 0.0f;
        terminals[row] = 0.0f;
    }
    runtime->ready = 1u;
    runtime->failed = 0u;
    runtime->last_status = REK_G1_SEMANTIC_DUEL_OK;
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

static int semantic_valid(const RekG1SemanticTick* semantic) {
    if (semantic == NULL || semantic->status != REK_G1_SEMANTIC_OK
            || semantic->input.status != REK_G1_INPUT_ACCEPTED
            || semantic->kind > REK_G1_SEMANTIC_DISCRETE_MOVE
            || (semantic->input.held & ~REK_G1_HELD_VALID_MASK) != 0u
            || rek_g1_has_opposite_translation(semantic->input.held)
            || rek_g1_has_opposite_yaw(semantic->input.held)
            || semantic->input.forward < -1 || semantic->input.forward > 1
            || semantic->input.strafe < -1 || semantic->input.strafe > 1
            || semantic->input.desired_yaw < -1
            || semantic->input.desired_yaw > 1
            || !finite_f32(semantic->input.yaw)
            || !finite_f32(semantic->input.yaw_ramp)
            || semantic->input.yaw_ramp < 0.0f
            || semantic->input.yaw_ramp > 1.0f
            || semantic->command_started > 1u
            || semantic->segment_complete > 1u
            || semantic->move_start_edge > 1u
            || semantic->move_active > 1u
            || semantic->move_blocked > 1u) {
        return 0;
    }
    if (semantic->kind == REK_G1_SEMANTIC_LOCOMOTION) {
        return semantic->move_registry_index == REK_G1_SEMANTIC_MOVE_NONE
            && !semantic->move_start_edge && !semantic->move_active
            && !semantic->move_blocked
            && semantic->input.attack_gate == REK_G1_ATTACK_NOT_REQUESTED;
    }
    return semantic->move_registry_index != REK_G1_SEMANTIC_MOVE_NONE
        && semantic->move_active && !semantic->move_blocked
        && (semantic->move_start_edge
            ? (semantic->command_started
                && semantic->input.attack_gate
                    == REK_G1_ATTACK_ACCEPTED_PREEMPT_YAW)
            : semantic->input.attack_gate == REK_G1_ATTACK_NOT_REQUESTED)
        && semantic->input.forward == 0
        && semantic->input.strafe == 0
        && semantic->input.yaw == 0.0f;
}

static int transition_gate_open(
        RekG1SemanticDuelRuntime* runtime,
        size_t row,
        const RekG1NativeLocomotionState* state,
        uint8_t* gate_open) {
    if (runtime == NULL || state == NULL || gate_open == NULL) return 0;

    /*
     * This fact gates attacks on translation, not on yaw-only locomotion.
     * Classify every retained route and its matching velocity state so corrupt
     * or internally inconsistent state cannot accidentally open the gate.
     */
    const RekG1NativeMotionRoute* current_route =
        rek_g1_native_route_by_id(
            runtime->motion_routes, state->current_route_id);
    const RekG1NativeMotionRoute* transition_route =
        rek_g1_native_route_by_id(
            runtime->motion_routes, state->transition_from_route_id);
    const RekG1NativeMotionRoute* momentum_route =
        rek_g1_native_route_by_id(
            runtime->motion_routes, state->momentum_route_id);
    if (current_route == NULL || transition_route == NULL
            || momentum_route == NULL
            || current_route->kind > REK_G1_NATIVE_ROUTE_TURN
            || transition_route->kind > REK_G1_NATIVE_ROUTE_TURN
            || momentum_route->kind > REK_G1_NATIVE_ROUTE_TURN
            || state->locomotion_active > 1u
            || state->transition_settling > 1u
            || state->stop_braking > 1u
            || state->has_momentum > 1u
            || !isfinite(state->last_driven_command.forward)
            || !isfinite(state->last_driven_command.strafe)
            || !isfinite(state->last_driven_command.yaw)
            || !isfinite(state->stop_brake_command.forward)
            || !isfinite(state->stop_brake_command.strafe)
            || !isfinite(state->stop_brake_command.yaw)
            || (state->locomotion_active && state->transition_settling)
            || (state->locomotion_active && state->has_momentum)
            || (state->transition_settling && state->has_momentum)
            || (state->transition_settling && state->stop_braking)
            || (state->locomotion_active
                && current_route->kind != REK_G1_NATIVE_ROUTE_TRANSLATION
                && current_route->kind != REK_G1_NATIVE_ROUTE_TURN)
            || (state->transition_settling
                && transition_route->kind != REK_G1_NATIVE_ROUTE_TRANSLATION
                && transition_route->kind != REK_G1_NATIVE_ROUTE_TURN)
            || (state->has_momentum
                && momentum_route->kind != REK_G1_NATIVE_ROUTE_TRANSLATION
                && momentum_route->kind != REK_G1_NATIVE_ROUTE_TURN)
            || (state->stop_braking
                && current_route->kind != REK_G1_NATIVE_ROUTE_TRANSLATION
                && current_route->kind != REK_G1_NATIVE_ROUTE_TURN)) {
        return 0;
    }

    const uint8_t last_driven_has_translation =
        fabsf(state->last_driven_command.forward)
                >= REK_G1_NATIVE_COMMAND_EPSILON
            || fabsf(state->last_driven_command.strafe)
                >= REK_G1_NATIVE_COMMAND_EPSILON;
    const uint8_t stop_brake_has_translation =
        fabsf(state->stop_brake_command.forward)
                >= REK_G1_NATIVE_COMMAND_EPSILON
            || fabsf(state->stop_brake_command.strafe)
                >= REK_G1_NATIVE_COMMAND_EPSILON;

    if ((state->locomotion_active
            && (current_route->kind == REK_G1_NATIVE_ROUTE_TRANSLATION
                || last_driven_has_translation))
            || (state->transition_settling
                && (transition_route->kind
                        == REK_G1_NATIVE_ROUTE_TRANSLATION
                    || last_driven_has_translation))
            || (state->stop_braking
                && (current_route->kind == REK_G1_NATIVE_ROUTE_TRANSLATION
                    || stop_brake_has_translation))) {
        *gate_open = 0u;
        return 1;
    }
    if (!state->has_momentum) {
        *gate_open = 1u;
        return 1;
    }
    if (momentum_route->kind == REK_G1_NATIVE_ROUTE_TURN) {
        *gate_open = 1u;
        return 1;
    }
    RekG1NativeBaseVelocitySample sample = {0};
    uint8_t settled = 0u;
    if (!gather_velocity_sample(runtime, row, &sample)
            || rek_g1_native_transition_settled(
                state->momentum_route_id,
                &runtime->config.locomotion,
                &sample,
                &settled) != REK_G1_NATIVE_COMMAND_OK) {
        return 0;
    }
    *gate_open = settled;
    return 1;
}

static int apply_route(
        RekG1SemanticDuelRuntime* runtime,
        SonicMotionComposerNative* composer,
        RekG1NativeRouteId route_id,
        char* error,
        size_t error_capacity) {
    const RekG1SemanticDuelRouteAsset* asset = asset_by_id(runtime, route_id);
    if (asset == NULL || (size_t)route_id >= REK_G1_STATIC_ROUTE_COUNT) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_INVALID_ROUTE_ASSET,
            error, error_capacity, "compose semantic row", "route is unavailable");
        return 0;
    }
    const SonicMotionComposerNativeStatus status =
        sonic_motion_composer_native_play_action(
            composer,
            &asset->clip,
            &runtime->route_configs[(size_t)route_id]);
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_COMPOSER_FAILED,
            error, error_capacity,
            "compose semantic row",
            sonic_motion_composer_native_status_string(status));
        return 0;
    }
    return 1;
}

static int compose_row(
        RekG1SemanticDuelRuntime* runtime,
        const RekG1PufferActionTable* action_table,
        const RekG1SemanticTick* semantic,
        size_t row,
        char* error,
        size_t error_capacity) {
    SonicMotionComposerNative* composer = &runtime->scratch_composers[row];
    RekG1NativeLocomotionState* locomotion =
        &runtime->scratch_locomotion_states[row];
    const RekG1NativeVelocityCommand input_command = {
        .forward = (float)semantic->input.forward,
        .strafe = (float)semantic->input.strafe,
        .yaw = semantic->input.yaw,
    };

    if (semantic->kind == REK_G1_SEMANTIC_DISCRETE_MOVE
            && !semantic->move_start_edge
            && !composer->action_playing) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_PROTOCOL_INVALID,
            error, error_capacity,
            "compose semantic row",
            "configured discrete-move segment outlived composer action");
        return 0;
    }

    if (semantic->kind == REK_G1_SEMANTIC_DISCRETE_MOVE
            && semantic->move_start_edge) {
        if (semantic->move_registry_index >= action_table->move_registry_count
                || composer->action_playing) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_PROTOCOL_INVALID,
                error, error_capacity,
                "compose semantic row", "invalid discrete-move start edge");
            return 0;
        }
        const uint16_t move_index = action_table->move_indices[
            semantic->move_registry_index];
        const RekG1NativeMotionRoute* route =
            rek_g1_native_discrete_move_route(
                runtime->motion_routes, move_index);
        if (route == NULL || !apply_route(
                runtime, composer, route->id, error, error_capacity)) {
            return 0;
        }
        /* ExecuteMove clears only locomotionActive on the composer branch. */
        locomotion->locomotion_active = 0u;
        runtime->scratch_active_route_ids[row] = route->id;
        runtime->scratch_effective_velocity[row] = input_command;
    } else {
        RekG1NativeRouteSelection selection = {0};
        if (rek_g1_native_select_locomotion_route(
                input_command, &selection) != REK_G1_NATIVE_COMMAND_OK
                || asset_by_id(runtime, selection.route_id) == NULL) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_LOCOMOTION_FAILED,
                error, error_capacity,
                "compose semantic row", "route selection failed");
            return 0;
        }
        const RekG1NativeLocomotionStepInput step_input = {
            .command = input_command,
            .base_velocity = runtime->velocity_samples[row],
            .delta_seconds = runtime->config.input_timing.elapsed_seconds,
            .restrict_yaw = 1u,
            .composer_action_playing = composer->action_playing ? 1u : 0u,
            .composer_busy = composer_is_busy(composer) ? 1u : 0u,
            .selected_route_playable = 1u,
        };
        RekG1NativeLocomotionStepResult step = {0};
        const RekG1NativeCommandStatus locomotion_status =
            rek_g1_native_locomotion_step(
                locomotion,
                &runtime->config.locomotion,
                &step_input,
                &step);
        if (locomotion_status != REK_G1_NATIVE_COMMAND_OK) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_LOCOMOTION_FAILED,
                error, error_capacity,
                "compose semantic row", "locomotion transition failed");
            return 0;
        }
        if (step.event != REK_G1_NATIVE_LOCOMOTION_EVENT_NONE) {
            if (!apply_route(
                    runtime,
                    composer,
                    step.event_route_id,
                    error,
                    error_capacity)) {
                return 0;
            }
            runtime->scratch_active_route_ids[row] = step.event_route_id;
        }
        *locomotion = step.next_state;
        runtime->scratch_effective_velocity[row] = step.effective_velocity;
    }

    RekG1NativePlaybackUpdate playback = {0};
    if (rek_g1_native_playback_update(
            runtime->scratch_effective_velocity[row],
            &runtime->config.command,
            &playback) != REK_G1_NATIVE_COMMAND_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_LOCOMOTION_FAILED,
            error, error_capacity,
            "compose semantic row", "playback update failed");
        return 0;
    }
    if (playback.apply) {
        const SonicMotionComposerNativeStatus composer_status =
            sonic_motion_composer_native_set_locomotion_speed(
                composer, playback.scale);
        if (composer_status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_COMPOSER_FAILED,
                error, error_capacity,
                "compose semantic row",
                sonic_motion_composer_native_status_string(composer_status));
            return 0;
        }
    }
    const size_t dof_offset = row
        * REK_G1_SEMANTIC_DUEL_REFERENCE_ROWS * GEAR_SONIC_ACTION_DIM;
    const size_t root_offset = row
        * REK_G1_SEMANTIC_DUEL_REFERENCE_ROWS * 4u;
    SonicMotionComposerNativeReferenceOutput output = {
        .dof_position_mujoco = runtime->reference_dof_position + dof_offset,
        .dof_next_position_mujoco =
            runtime->reference_dof_next_position + dof_offset,
        .root_rotation_xyzw =
            runtime->reference_root_rotation_xyzw + root_offset,
        .dof_position_capacity =
            REK_G1_SEMANTIC_DUEL_REFERENCE_ROWS * GEAR_SONIC_ACTION_DIM,
        .dof_next_position_capacity =
            REK_G1_SEMANTIC_DUEL_REFERENCE_ROWS * GEAR_SONIC_ACTION_DIM,
        .root_rotation_capacity =
            REK_G1_SEMANTIC_DUEL_REFERENCE_ROWS * 4u,
    };
    const SonicMotionComposerNativeStatus reference_status =
        sonic_motion_composer_native_build_reference_rows(
            composer,
            &REFERENCE_TIMING,
            &runtime->config.mirror_table,
            &output);
    if (reference_status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_COMPOSER_FAILED,
            error, error_capacity,
            "compose semantic row",
            sonic_motion_composer_native_status_string(reference_status));
        return 0;
    }
    return 1;
}

static int stage_heading_update(
        RekG1SemanticDuelRuntime* runtime,
        size_t row,
        char* error,
        size_t error_capacity) {
    SonicMotionComposerNative* composer = &runtime->scratch_composers[row];
    SonicMotionComposerNativeAdvanceResult advance = {0};
    SonicMotionComposerNativeStatus composer_status =
        sonic_motion_composer_native_advance(composer, &advance);
    if (composer_status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_COMPOSER_FAILED,
            error, error_capacity,
            "advance semantic composer",
            sonic_motion_composer_native_status_string(composer_status));
        return 0;
    }
    float clip_delta = 0.0f;
    float ownership = 0.0f;
    composer_status = sonic_motion_composer_native_consume_heading_delta(
        composer, &clip_delta);
    if (composer_status == SONIC_MOTION_COMPOSER_NATIVE_OK) {
        composer_status = sonic_motion_composer_native_heading_clip_ownership(
            composer, &ownership);
    }
    if (composer_status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_COMPOSER_FAILED,
            error, error_capacity,
            "advance semantic heading",
            sonic_motion_composer_native_status_string(composer_status));
        return 0;
    }
    RekG1NativeHeadingUpdate heading = {0};
    if (rek_g1_native_heading_update(
            runtime->scratch_effective_velocity[row],
            &runtime->config.command,
            ownership,
            clip_delta,
            runtime->forgiveness_deltas[row],
            &heading) != REK_G1_NATIVE_COMMAND_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_LOCOMOTION_FAILED,
            error, error_capacity,
            "advance semantic heading", "scalar heading update failed");
        return 0;
    }
    const float half = f32_mul(heading.total_delta_radians, 0.5f);
    float sine = 0.0f;
    float cosine = 0.0f;
    if (!runtime->config.composer_backends.sin_cos_f(
            runtime->config.composer_backends.context,
            half,
            &sine,
            &cosine)
            || !finite_f32(sine) || !finite_f32(cosine)) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_COMPOSER_FAILED,
            error, error_capacity,
            "advance semantic heading", "sin/cos backend failed");
        return 0;
    }
    float old[4];
    for (size_t axis = 0; axis < 4; axis++) {
        if (!double_to_float(
                runtime->duel->heading_delta_wxyz[row * 4 + axis],
                &old[axis])) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_LOCOMOTION_FAILED,
                error, error_capacity,
                "advance semantic heading", "existing heading is non-finite");
            return 0;
        }
    }
    /* [cos, 0, 0, sin] left-multiplied by the existing WXYZ heading. */
    const float next[4] = {
        f32_sub(f32_mul(cosine, old[0]), f32_mul(sine, old[3])),
        f32_sub(f32_mul(cosine, old[1]), f32_mul(sine, old[2])),
        f32_add(f32_mul(cosine, old[2]), f32_mul(sine, old[1])),
        f32_add(f32_mul(cosine, old[3]), f32_mul(sine, old[0])),
    };
    for (size_t axis = 0; axis < 4; axis++) {
        if (!finite_f32(next[axis])) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_LOCOMOTION_FAILED,
                error, error_capacity,
                "advance semantic heading", "new heading is non-finite");
            return 0;
        }
        runtime->scratch_heading_delta_wxyz[row * 4 + axis]
            = (double)next[axis];
    }
    return 1;
}

static int strike_intent_for_row(
        RekG1SemanticDuelRuntime* runtime,
        size_t row,
        RekG1StrikeIntent* output,
        char* error,
        size_t error_capacity) {
    if (runtime == NULL || output == NULL || row >= runtime->robot_count) {
        return 0;
    }
    memset(output, 0, sizeof(*output));
    const RekG1NativeMotionRoute* route = rek_g1_native_route_by_id(
        runtime->motion_routes,
        runtime->scratch_active_route_ids[row]);
    const SonicMotionComposerNative* composer =
        &runtime->scratch_composers[row];
    if (route == NULL) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_COMBAT_FAILED,
            error, error_capacity,
            "measure strike intent", "active route is unavailable");
        return 0;
    }
    if (route->kind != REK_G1_NATIVE_ROUTE_DISCRETE_MOVE
            || !composer->action_playing) {
        return 1;
    }
    const RekG1StrikeComposerSnapshot snapshot = {
        .active_route_id = route->id,
        .clip_cursor_frames = composer->current_layer.cursor,
        .clip_fps = composer->current_layer.clip.fps,
        .action_move_id = composer->action_move_id,
        .action_playing = composer->action_playing ? 1u : 0u,
        .current_layer_has_clip =
            composer->current_layer.has_clip ? 1u : 0u,
        .current_layer_has_config =
            composer->current_layer.has_config ? 1u : 0u,
        .current_layer_active =
            composer->current_layer.active ? 1u : 0u,
        .current_layer_loop =
            composer->current_layer.config.loop ? 1u : 0u,
    };
    if (!rek_g1_strike_intent_from_snapshot(
            rek_g1_current_build_strike_catalog(), &snapshot, output)) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_COMBAT_FAILED,
            error, error_capacity,
            "measure strike intent",
            "active discrete-move snapshot is incomplete");
        return 0;
    }
    return 1;
}

static int initialize_scratch_composer_row(
        RekG1SemanticDuelRuntime* runtime,
        size_t row,
        char* error,
        size_t error_capacity) {
    const RekG1SemanticDuelRouteAsset* idle = asset_by_id(
        runtime, REK_G1_NATIVE_IDLE);
    if (idle == NULL || row >= runtime->robot_count) return 0;
    SonicMotionComposerNativeStatus status =
        sonic_motion_composer_native_init(
            &runtime->scratch_composers[row],
            REK_G1_SEMANTIC_DUEL_CONTROLLER_RATE_HZ,
            &runtime->config.composer_backends);
    if (status == SONIC_MOTION_COMPOSER_NATIVE_OK) {
        status = sonic_motion_composer_native_play_action_immediate(
            &runtime->scratch_composers[row],
            &idle->clip,
            &runtime->route_configs[REK_G1_NATIVE_IDLE]);
    }
    if (status != SONIC_MOTION_COMPOSER_NATIVE_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_COMPOSER_FAILED,
            error, error_capacity,
            "reset semantic arena composer",
            sonic_motion_composer_native_status_string(status));
        return 0;
    }
    runtime->scratch_locomotion_states[row] =
        (RekG1NativeLocomotionState){0};
    runtime->scratch_effective_velocity[row] =
        (RekG1NativeVelocityCommand){0};
    runtime->scratch_active_route_ids[row] = REK_G1_NATIVE_IDLE;
    return 1;
}

static int stage_pending_episode_reset(
        RekG1SemanticDuelRuntime* runtime,
        size_t arena,
        char* error,
        size_t error_capacity) {
    if (runtime == NULL || arena >= runtime->duel->arena_count
            || runtime->pending_episode_resets == NULL) {
        return 0;
    }
    if (!runtime->pending_episode_resets[arena]) return 1;

    RekG1CombatArenaState combat = {0};
    if (rek_g1_combat_arena_init(&combat) != REK_G1_COMBAT_TICK_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_COMBAT_FAILED,
            error, error_capacity,
            "reset pending episode combat", "fight initialization failed");
        return 0;
    }
    for (size_t fighter = 0u; fighter < GEAR_SONIC_DUEL_FIGHTERS;
            fighter++) {
        const size_t row = arena * GEAR_SONIC_DUEL_FIGHTERS + fighter;
        if (!initialize_scratch_composer_row(
                runtime, row, error, error_capacity)) {
            return 0;
        }
        if (rek_g1_fall_state_init(
                &REK_G1_FALL_CONFIG_F84F1874,
                &runtime->scratch_fall_states[row]) != REK_G1_FALL_OK) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_FALL_MEASUREMENT_FAILED,
                error, error_capacity,
                "reset pending episode fall state",
                "state initialization failed");
            return 0;
        }
        runtime->scratch_fall_measurements[row] =
            (RekG1FallMujocoMeasurement){0};
    }
    if (rek_g1_hit_mujoco_clear_arena_contacts(
            &runtime->hit_adapter,
            arena,
            error,
            error_capacity) != REK_G1_HIT_MUJOCO_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_HIT_MEASUREMENT_FAILED,
            error, error_capacity,
            "reset pending episode contacts",
            rek_g1_hit_mujoco_status_string(
                runtime->hit_adapter.last_status));
        return 0;
    }
    if (!gear_sonic_native_duel_reset_arena_immediate(
            runtime->duel, arena, error, error_capacity)) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_ARENA_RESET_FAILED,
            error, error_capacity,
            "reset pending episode arena", "immediate arena reset failed");
        return 0;
    }

    for (size_t fighter = 0u; fighter < GEAR_SONIC_DUEL_FIGHTERS;
            fighter++) {
        const size_t row = arena * GEAR_SONIC_DUEL_FIGHTERS + fighter;
        for (size_t axis = 0u; axis < 4u; axis++) {
            runtime->scratch_heading_delta_wxyz[row * 4u + axis] =
                runtime->duel->heading_delta_wxyz[row * 4u + axis];
        }
    }
    runtime->scratch_combat_states[arena] = combat;
    runtime->pending_episode_resets[arena] = 0u;
    return 1;
}

static int stage_semantic_spawn_reset_begin(
        RekG1SemanticDuelRuntime* runtime,
        size_t arena,
        char* error,
        size_t error_capacity) {
    if (runtime == NULL || arena >= runtime->duel->arena_count) return 0;
    RekG1CombatArenaState combat = {0};
    if (rek_g1_combat_arena_apply_spawn_reset(
            &runtime->scratch_combat_states[arena], &combat)
            != REK_G1_COMBAT_TICK_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_COMBAT_FAILED,
            error, error_capacity,
            "reset semantic arena combat", "spawn reset was rejected");
        return 0;
    }
    for (size_t fighter = 0u; fighter < GEAR_SONIC_DUEL_FIGHTERS;
            fighter++) {
        const size_t row = arena * GEAR_SONIC_DUEL_FIGHTERS + fighter;
        RekG1FallState fall_reset = {0};
        if (rek_g1_fall_state_apply_fight_spawn_reset(
                &REK_G1_FALL_CONFIG_F84F1874,
                &runtime->scratch_fall_states[row],
                &fall_reset) != REK_G1_FALL_OK) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_FALL_MEASUREMENT_FAILED,
                error, error_capacity,
                "reset semantic arena fall state",
                "fight spawn reset was rejected");
            return 0;
        }
        runtime->scratch_fall_states[row] = fall_reset;
    }
    if (rek_g1_hit_mujoco_clear_arena_contacts(
            &runtime->hit_adapter,
            arena,
            error,
            error_capacity) != REK_G1_HIT_MUJOCO_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_HIT_MEASUREMENT_FAILED,
            error, error_capacity,
            "reset semantic arena contacts",
            rek_g1_hit_mujoco_status_string(
                runtime->hit_adapter.last_status));
        return 0;
    }
    if (!gear_sonic_native_duel_begin_arena_reset(
            runtime->duel, arena, error, error_capacity)) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_ARENA_RESET_FAILED,
            error, error_capacity,
            "begin semantic arena reset", "deferred arena reset failed");
        return 0;
    }
    runtime->scratch_combat_states[arena] = combat;
    runtime->arena_reset_events[arena] = 1u;
    return 1;
}

static int complete_semantic_spawn_reset(
        RekG1SemanticDuelRuntime* runtime,
        size_t arena,
        char* error,
        size_t error_capacity) {
    if (runtime == NULL || arena >= runtime->duel->arena_count) return 0;
    if (!gear_sonic_native_duel_complete_arena_reset(
            runtime->duel, arena, error, error_capacity)) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_ARENA_RESET_FAILED,
            error, error_capacity,
            "complete semantic arena reset", "deferred arena reset failed");
        return 0;
    }
    for (size_t fighter = 0u; fighter < GEAR_SONIC_DUEL_FIGHTERS;
            fighter++) {
        const size_t row = arena * GEAR_SONIC_DUEL_FIGHTERS + fighter;
        if (!initialize_scratch_composer_row(
                runtime, row, error, error_capacity)) {
            return 0;
        }
        runtime->forgiveness_deltas[row] = 0.0f;
        for (size_t axis = 0u; axis < 4u; axis++) {
            runtime->scratch_heading_delta_wxyz[row * 4u + axis] =
                runtime->duel->heading_delta_wxyz[row * 4u + axis];
        }
    }
    runtime->arena_reset_events[arena] = 1u;
    runtime->arena_input_reset_events[arena] = 1u;
    return 1;
}

static int add_tick_score_delta(
        RekG1SemanticDuelRuntime* runtime,
        size_t row,
        int32_t delta) {
    if (runtime == NULL || row >= runtime->robot_count || delta < 0
            || runtime->score_deltas[row] > INT32_MAX - delta) {
        return 0;
    }
    runtime->score_deltas[row] += delta;
    return 1;
}

static int add_tick_contact_count(uint32_t* target, uint32_t delta) {
    if (target == NULL || *target > UINT32_MAX - delta) return 0;
    *target += delta;
    return 1;
}

static int accumulate_combat_result(
        RekG1SemanticDuelRuntime* runtime,
        size_t arena,
        const RekG1CombatSubstepResult* result) {
    if (runtime == NULL || result == NULL
            || arena >= runtime->duel->arena_count) {
        return 0;
    }
    const size_t row = arena * GEAR_SONIC_DUEL_FIGHTERS;
    if (!add_tick_score_delta(runtime, row, result->score_delta[0])
            || !add_tick_score_delta(
                runtime, row + 1u, result->score_delta[1])
            || !add_tick_contact_count(
                &runtime->attributed_contact_counts[arena],
                result->attributed_contact_count)
            || !add_tick_contact_count(
                &runtime->scored_contact_counts[arena],
                result->scored_contact_count)) {
        return 0;
    }
    runtime->fight_signals[arena] |= result->signals;
    runtime->referee_calls[arena] |= result->referee_calls;
    if ((result->signals & REK_G1_FIGHT_SIGNAL_ROUND_ENDED) != 0u) {
        runtime->arena_terminals[arena] = 1u;
        runtime->pending_episode_resets[arena] = 1u;
    }
    return 1;
}

static int sample_final_fall_measurement(
        RekG1SemanticDuelRuntime* runtime,
        size_t row,
        char* error,
        size_t error_capacity) {
    RekG1FallMujocoMeasurement measurement = {0};
    if (rek_g1_fall_mujoco_sample(
            &runtime->fall_adapter,
            row,
            REK_G1_SEMANTIC_DUEL_PHYSICS_DELTA_SECONDS,
            REK_G1_SEMANTIC_DUEL_PROVISIONAL_CAN_GET_UP,
            &measurement,
            error,
            error_capacity) != REK_G1_FALL_MUJOCO_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_FALL_MEASUREMENT_FAILED,
            error, error_capacity,
            "sample final fall state",
            rek_g1_fall_mujoco_status_string(
                runtime->fall_adapter.last_status));
        return 0;
    }
    runtime->scratch_fall_measurements[row] = measurement;
    return 1;
}

static int semantic_post_step_directive(
        void* context,
        const GearSonicNativeDuelPostStepObservation* observation,
        GearSonicNativeDuelPostStepDirective* directive) {
    RekG1SemanticDuelRuntime* runtime =
        (RekG1SemanticDuelRuntime*)context;
    if (runtime == NULL || observation == NULL || directive == NULL
            || observation->arena_index >= runtime->duel->arena_count) {
        return 0;
    }
    *directive = GEAR_SONIC_DUEL_POST_STEP_CONTINUE;
    const size_t arena = observation->arena_index;
    if (runtime->duel->reset_pending_arenas[arena]) {
        size_t discarded_contact_count = 0u;
        if (rek_g1_hit_mujoco_scan_substep(
                &runtime->hit_adapter,
                observation,
                runtime->hit_candidates,
                runtime->hit_candidate_capacity,
                &discarded_contact_count,
                NULL,
                0u) != REK_G1_HIT_MUJOCO_OK
                || rek_g1_hit_mujoco_clear_arena_contacts(
                    &runtime->hit_adapter,
                    arena,
                    NULL,
                    0u) != REK_G1_HIT_MUJOCO_OK) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_HIT_MEASUREMENT_FAILED,
                NULL, 0u, "discard reset-boundary contacts",
                "hit substep sequence failed");
            return 0;
        }
        const double deadline =
            runtime->duel->reset_complete_not_before_time[arena];
        if (!isfinite(observation->data->time) || !isfinite(deadline)) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_ARENA_RESET_FAILED,
                NULL, 0u, "complete semantic arena reset",
                "reset boundary time is not finite");
            return 0;
        }
        if ((double)observation->data->time < deadline) {
            return 1;
        }
        if (!complete_semantic_spawn_reset(runtime, arena, NULL, 0u)) {
            return 0;
        }
        return 1;
    }
    size_t candidate_count = 0u;
    if (rek_g1_hit_mujoco_scan_substep(
            &runtime->hit_adapter,
            observation,
            runtime->hit_candidates,
            runtime->hit_candidate_capacity,
            &candidate_count,
            NULL,
            0u) != REK_G1_HIT_MUJOCO_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_HIT_MEASUREMENT_FAILED,
            NULL, 0u, "scan substep hit contacts",
            rek_g1_hit_mujoco_status_string(
                runtime->hit_adapter.last_status));
        return 0;
    }
    const size_t arena_base = arena * GEAR_SONIC_DUEL_FIGHTERS;
    const RekG1FallPhase pre_substep_fall_phase[GEAR_SONIC_DUEL_FIGHTERS] = {
        runtime->scratch_fall_states[arena_base].phase,
        runtime->scratch_fall_states[arena_base + 1u].phase,
    };
    for (size_t fighter = 0u; fighter < GEAR_SONIC_DUEL_FIGHTERS;
            fighter++) {
        if (!stage_fall_measurement(
                runtime,
                arena_base + fighter,
                REK_G1_SEMANTIC_DUEL_PHYSICS_DELTA_SECONDS,
                NULL,
                0u)) {
            return 0;
        }
        const size_t row = arena_base + fighter;
        if ((runtime->substep_fall_events[row]
                & REK_G1_FALL_EVENT_BECAME_FALLEN) != 0u
                && !gear_sonic_native_duel_set_row_dampened(
                    runtime->duel, row, 1, NULL, 0u)) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_ARENA_RESET_FAILED,
                NULL, 0u, "dampen fallen semantic row",
                "native dampening transition failed");
            return 0;
        }
    }
    if (runtime->arena_terminals[arena]) return 1;

    float time_seconds = 0.0f;
    if (!double_to_float(observation->data->time, &time_seconds)) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_HIT_MEASUREMENT_FAILED,
            NULL, 0u, "scan substep hit contacts",
            "MuJoCo time is not finite binary32");
        return 0;
    }
    const RekG1FightState* fight =
        &runtime->scratch_combat_states[arena].fight;
    const uint8_t round_active =
        fight->phase == REK_G1_FIGHT_ROUND_ACTIVE ? 1u : 0u;
    for (size_t index = 0u; index < candidate_count; index++) {
        const RekG1HitMujocoCandidate* candidate =
            &runtime->hit_candidates[index];
        const size_t striker_row = arena_base
            + (size_t)candidate->striker_fighter;
        RekG1StrikeIntent intent = {0};
        if (!strike_intent_for_row(
                runtime, striker_row, &intent, NULL, 0u)) {
            return 0;
        }
        const RekG1HitMujocoCallerFacts facts = {
            .strike_intent = intent,
            .time_seconds = time_seconds,
            .round_active = round_active,
            .fighter_upright = {
                pre_substep_fall_phase[0]
                    != REK_G1_FALL_FALLEN,
                pre_substep_fall_phase[1]
                    != REK_G1_FALL_FALLEN,
            },
            .fighter_standing = {
                pre_substep_fall_phase[0]
                    != REK_G1_FALL_FALLEN,
                pre_substep_fall_phase[1]
                    != REK_G1_FALL_FALLEN,
            },
        };
        if (rek_g1_hit_mujoco_candidate_to_contact(
                candidate,
                &facts,
                &runtime->hit_contacts[index],
                NULL,
                0u) != REK_G1_HIT_MUJOCO_OK) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_HIT_MEASUREMENT_FAILED,
                NULL, 0u, "assemble substep hit contact",
                "candidate conversion failed");
            return 0;
        }
    }

    float next_time_remaining = f32_sub(
        fight->time_remaining_seconds,
        REK_G1_SEMANTIC_DUEL_PHYSICS_DELTA_SECONDS);
    if (next_time_remaining < 0.0f) next_time_remaining = 0.0f;
    const RekG1CombatSubstepInput combat_input = {
        .delta_seconds = REK_G1_SEMANTIC_DUEL_PHYSICS_DELTA_SECONDS,
        .time_remaining_seconds = next_time_remaining,
        .contacts = runtime->hit_contacts,
        .contact_count = candidate_count,
        .fall_phase = {
            runtime->scratch_fall_states[arena_base].phase,
            runtime->scratch_fall_states[arena_base + 1u].phase,
        },
        .fall_events = {
            runtime->substep_fall_events[arena_base],
            runtime->substep_fall_events[arena_base + 1u],
        },
        .fighter_is_recovering = {0u, 0u},
        .fighter_can_get_up = {
            REK_G1_SEMANTIC_DUEL_PROVISIONAL_CAN_GET_UP,
            REK_G1_SEMANTIC_DUEL_PROVISIONAL_CAN_GET_UP,
        },
        .force_slip_estop = {0u, 0u},
    };
    RekG1CombatSubstepResult combat_result = {0};
    const RekG1CombatTickStatus combat_status =
        rek_g1_combat_arena_substep(
            &runtime->scratch_combat_states[arena],
            &(RekG1HitDetectorConfig){
                .speed_threshold_mps = 1.75f,
                .knockdown_strike_approach_mps = 2.0f,
                .per_body_cooldown_seconds = 0.30000001192092896f,
                .apex_min_ramp = 0.20000000298023224f,
            },
            &combat_input,
            &combat_result);
    if (combat_status != REK_G1_COMBAT_TICK_OK) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_COMBAT_FAILED,
            NULL, 0u, "advance substep combat",
            rek_g1_combat_tick_status_string(combat_status));
        return 0;
    }
    runtime->scratch_combat_states[arena] = combat_result.next_state;
    if (!accumulate_combat_result(runtime, arena, &combat_result)) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_COMBAT_FAILED,
            NULL, 0u, "accumulate substep combat", "counter overflow");
        return 0;
    }
    if ((combat_result.signals
            & REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN) != 0u) {
        if (!stage_semantic_spawn_reset_begin(
                runtime, arena, NULL, 0u)) {
            return 0;
        }
    }
    return 1;
}

int rek_g1_semantic_duel_advance_batch(
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
    RekG1SemanticDuelRuntime* runtime = (RekG1SemanticDuelRuntime*)context;
    if (!callback_arguments_valid(
            runtime, motion_routes, action_table, environment_count,
            next_facts_out, observations, observation_stride_bytes,
            rewards, terminals)
            || semantics == NULL || !runtime->ready || runtime->failed
            || !duel_valid(runtime->duel)) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_NOT_READY,
            error, error_capacity,
            "advance semantic duel", "runtime or callback arguments invalid");
        return 0;
    }

    memcpy(
        runtime->scratch_composers,
        runtime->composers,
        runtime->robot_count * sizeof(*runtime->composers));
    memcpy(
        runtime->scratch_locomotion_states,
        runtime->locomotion_states,
        runtime->robot_count * sizeof(*runtime->locomotion_states));
    memcpy(
        runtime->scratch_effective_velocity,
        runtime->effective_velocity,
        runtime->robot_count * sizeof(*runtime->effective_velocity));
    memcpy(
        runtime->scratch_active_route_ids,
        runtime->active_route_ids,
        runtime->robot_count * sizeof(*runtime->active_route_ids));
    memcpy(
        runtime->scratch_heading_delta_wxyz,
        runtime->duel->heading_delta_wxyz,
        runtime->robot_count * 4u * sizeof(double));
    memcpy(
        runtime->scratch_fall_states,
        runtime->fall_states,
        runtime->robot_count * sizeof(*runtime->fall_states));
    memcpy(
        runtime->scratch_fall_measurements,
        runtime->fall_measurements,
        runtime->robot_count * sizeof(*runtime->fall_measurements));
    memcpy(
        runtime->scratch_combat_states,
        runtime->combat_states,
        runtime->duel->arena_count * sizeof(*runtime->combat_states));
    clear_tick_combat_outputs(runtime);

    for (size_t row = 0; row < runtime->robot_count; row++) {
        if (!semantic_valid(&semantics[row])
                || (semantics[row].kind == REK_G1_SEMANTIC_DISCRETE_MOVE
                    && semantics[row].move_registry_index
                        >= action_table->move_registry_count)) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_PROTOCOL_INVALID,
                error, error_capacity,
                "advance semantic duel", "invalid semantic row");
            return 0;
        }
    }
    for (size_t arena = 0u; arena < runtime->duel->arena_count; arena++) {
        if (!stage_pending_episode_reset(
                runtime, arena, error, error_capacity)) {
            return 0;
        }
    }

    for (size_t row = 0; row < runtime->robot_count; row++) {
        if (!fall_state_allows_semantic_motion(
                runtime, row, error, error_capacity)) {
            return 0;
        }
        if (!gather_velocity_sample(
                runtime, row, &runtime->velocity_samples[row])) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_OBSERVATION_FAILED,
                error, error_capacity,
                "advance semantic duel", "base velocity unavailable");
            return 0;
        }
        if (row_policy_suspended(runtime, row)) {
            continue;
        }
        float forgiveness = 0.0f;
        if (!runtime->config.forgiveness_delta(
                runtime->config.forgiveness_context,
                runtime->duel,
                row,
                &forgiveness)
                || !finite_f32(forgiveness)) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_FORGIVENESS_UNAVAILABLE,
                error, error_capacity,
                "advance semantic duel", "forgiveness callback failed");
            return 0;
        }
        runtime->forgiveness_deltas[row] = forgiveness;
        if (!compose_row(
                runtime,
                action_table,
                &semantics[row],
                row,
                error,
                error_capacity)) {
            return 0;
        }
    }

    const GearSonicNativeReferenceInput references = {
        .dof_position_mujoco = runtime->reference_dof_position,
        .dof_next_position_mujoco = runtime->reference_dof_next_position,
        .root_rotation_xyzw = runtime->reference_root_rotation_xyzw,
    };
    if (!gear_sonic_native_duel_step_references_with_post_step_directive(
            runtime->duel,
            &references,
            semantic_post_step_directive,
            runtime,
            error,
            error_capacity)) {
        if (runtime->failed
                && runtime->last_status != REK_G1_SEMANTIC_DUEL_OK) {
            set_error(
                error,
                error_capacity,
                "advance semantic duel post-step",
                rek_g1_semantic_duel_status_string(runtime->last_status));
            return 0;
        }
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_PHYSICS_FAILED,
            error, error_capacity,
            "advance semantic duel", "shared-contact step failed");
        return 0;
    }

    for (size_t row = 0; row < runtime->robot_count; row++) {
        const size_t arena = row / GEAR_SONIC_DUEL_FIGHTERS;
        const int suspended = row_policy_suspended(runtime, row);
        if (runtime->arena_reset_events[arena]) {
            for (size_t axis = 0u; axis < 4u; axis++) {
                runtime->scratch_heading_delta_wxyz[row * 4u + axis] =
                    runtime->duel->heading_delta_wxyz[row * 4u + axis];
            }
            if (!sample_final_fall_measurement(
                    runtime, row, error, error_capacity)) {
                return 0;
            }
        } else if (!suspended && !stage_heading_update(
                runtime, row, error, error_capacity)) {
            return 0;
        }
        if (!runtime->arena_reset_events[arena]
                && !suspended
                && semantics[row].kind == REK_G1_SEMANTIC_DISCRETE_MOVE
                && ((semantics[row].segment_complete
                        && runtime->scratch_composers[row].action_playing)
                    || (!semantics[row].segment_complete
                        && !runtime->scratch_composers[row].action_playing))) {
            (void)latch_failure(
                runtime, REK_G1_SEMANTIC_DUEL_PROTOCOL_INVALID,
                error, error_capacity,
                "advance semantic duel",
                "configured discrete-move duration does not match composer completion");
            return 0;
        }
    }
    if (!gather_all_entities(runtime)
            || !populate_fall_observations(
                runtime,
                runtime->scratch_fall_measurements,
                runtime->scratch_fall_states,
                runtime->scratch_fall_events)) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_OBSERVATION_FAILED,
            error, error_capacity,
            "advance semantic duel", "post-step state gather failed");
        return 0;
    }
    if (!write_observations(
            runtime,
            runtime->scratch_composers,
            runtime->scratch_locomotion_states,
            runtime->scratch_effective_velocity,
            runtime->scratch_active_route_ids,
            runtime->scratch_heading_delta_wxyz,
            runtime->scratch_combat_states,
            runtime->scratch_fall_events,
            observations,
            observation_stride_bytes)) {
        (void)latch_failure(
            runtime, REK_G1_SEMANTIC_DUEL_OBSERVATION_FAILED,
            error, error_capacity,
            "advance semantic duel", "observation serialization failed");
        return 0;
    }

    memcpy(
        runtime->composers,
        runtime->scratch_composers,
        runtime->robot_count * sizeof(*runtime->composers));
    memcpy(
        runtime->locomotion_states,
        runtime->scratch_locomotion_states,
        runtime->robot_count * sizeof(*runtime->locomotion_states));
    memcpy(
        runtime->effective_velocity,
        runtime->scratch_effective_velocity,
        runtime->robot_count * sizeof(*runtime->effective_velocity));
    memcpy(
        runtime->active_route_ids,
        runtime->scratch_active_route_ids,
        runtime->robot_count * sizeof(*runtime->active_route_ids));
    memcpy(
        runtime->duel->heading_delta_wxyz,
        runtime->scratch_heading_delta_wxyz,
        runtime->robot_count * 4u * sizeof(double));
    memcpy(
        runtime->fall_states,
        runtime->scratch_fall_states,
        runtime->robot_count * sizeof(*runtime->fall_states));
    memcpy(
        runtime->fall_measurements,
        runtime->scratch_fall_measurements,
        runtime->robot_count * sizeof(*runtime->fall_measurements));
    memcpy(
        runtime->fall_events,
        runtime->scratch_fall_events,
        runtime->robot_count * sizeof(*runtime->fall_events));
    memcpy(
        runtime->combat_states,
        runtime->scratch_combat_states,
        runtime->duel->arena_count * sizeof(*runtime->combat_states));

    for (size_t row = 0; row < runtime->robot_count; row++) {
        const size_t arena = row / GEAR_SONIC_DUEL_FIGHTERS;
        if (runtime->arena_terminals[arena]) {
            next_facts_out[row] = episode_start_facts(runtime);
        } else {
            uint8_t gate_open = 0u;
            if (!transition_gate_open(
                    runtime,
                    row,
                    &runtime->locomotion_states[row],
                    &gate_open)) {
                (void)latch_failure(
                    runtime, REK_G1_SEMANTIC_DUEL_OBSERVATION_FAILED,
                    error, error_capacity,
                    "advance semantic duel",
                    "transition measurement unavailable");
                return 0;
            }
            const uint8_t recovery_active =
                runtime->fall_states[row].phase == REK_G1_FALL_FALLEN
                    || row_policy_suspended(runtime, row)
                    ? 1u : 0u;
            next_facts_out[row] = (RekG1RuntimeFacts){
                .timing = runtime->config.input_timing,
                .translation_transition_settled = recovery_active
                    ? 0u : gate_open,
                .action_busy = composer_is_busy(&runtime->composers[row])
                    ? 1u : 0u,
                .recovery_active = recovery_active,
                .input_reset = runtime->arena_input_reset_events[arena],
            };
        }
        const size_t fighter = row % GEAR_SONIC_DUEL_FIGHTERS;
        const size_t opponent_row = arena * GEAR_SONIC_DUEL_FIGHTERS
            + (1u - fighter);
        rewards[row] = (float)runtime->score_deltas[row]
            - (float)runtime->score_deltas[opponent_row];
        terminals[row] = runtime->arena_terminals[arena] ? 1.0f : 0.0f;
    }
    runtime->last_status = REK_G1_SEMANTIC_DUEL_OK;
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return 1;
}

RekG1NativeBatchOps rek_g1_semantic_duel_batch_ops(void) {
    return (RekG1NativeBatchOps){
        .runtime_facts_abi_version = REK_G1_RUNTIME_FACTS_ABI_VERSION,
        .runtime_facts_size = REK_G1_RUNTIME_FACTS_SIZE,
        .reset = rek_g1_semantic_duel_reset_batch,
        .advance = rek_g1_semantic_duel_advance_batch,
        .close = rek_g1_semantic_duel_close,
    };
}

void rek_g1_semantic_duel_close(void* context) {
    RekG1SemanticDuelRuntime* runtime = (RekG1SemanticDuelRuntime*)context;
    if (runtime == NULL) return;
    if (runtime->hit_adapter_open || runtime->hit_adapter.duel != NULL) {
        rek_g1_hit_mujoco_close(&runtime->hit_adapter);
    }
    if (runtime->fall_adapter_open || runtime->fall_adapter.duel != NULL) {
        rek_g1_fall_mujoco_close(&runtime->fall_adapter);
    }
    free(runtime->pending_episode_resets);
    free(runtime->arena_terminals);
    free(runtime->arena_input_reset_events);
    free(runtime->arena_reset_events);
    free(runtime->scored_contact_counts);
    free(runtime->attributed_contact_counts);
    free(runtime->score_deltas);
    free(runtime->referee_calls);
    free(runtime->fight_signals);
    free(runtime->substep_fall_events);
    free(runtime->hit_contacts);
    free(runtime->hit_candidates);
    free(runtime->scratch_combat_states);
    free(runtime->combat_states);
    free(runtime->reference_root_rotation_xyzw);
    free(runtime->reference_dof_next_position);
    free(runtime->reference_dof_position);
    free(runtime->scratch_heading_delta_wxyz);
    free(runtime->forgiveness_deltas);
    free(runtime->scratch_fall_events);
    free(runtime->fall_events);
    free(runtime->scratch_fall_measurements);
    free(runtime->fall_measurements);
    free(runtime->scratch_fall_states);
    free(runtime->fall_states);
    free(runtime->entity_observations);
    free(runtime->velocity_samples);
    free(runtime->scratch_active_route_ids);
    free(runtime->active_route_ids);
    free(runtime->scratch_effective_velocity);
    free(runtime->effective_velocity);
    free(runtime->scratch_locomotion_states);
    free(runtime->locomotion_states);
    free(runtime->scratch_composers);
    free(runtime->composers);
    memset(runtime, 0, sizeof(*runtime));
}
