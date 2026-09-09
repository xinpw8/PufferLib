#include "gear_sonic_native_duel.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct RowSnapshot {
    double base_quaternion_wxyz[4];
    double base_angular_velocity_local[3];
    double joint_position_mujoco[GEAR_SONIC_ACTION_DIM];
    double joint_velocity_mujoco[GEAR_SONIC_ACTION_DIM];
    double heading_delta_wxyz[4];
    size_t reference_frame;
    uint64_t policy_tick;
    float command_lpf_state_mujoco[GEAR_SONIC_ACTION_DIM];
    uint8_t command_lpf_initialized;
    float encoder_observations[GEAR_SONIC_ENCODER_INPUT_WIDTH];
    float tokens[GEAR_SONIC_ENCODER_OUTPUT_WIDTH];
    float decoder_observations[GEAR_SONIC_DECODER_INPUT_WIDTH];
    float actions_policy[GEAR_SONIC_ACTION_DIM];
    float clipped_actions_policy[GEAR_SONIC_ACTION_DIM];
    float targets_mujoco[GEAR_SONIC_ACTION_DIM];
    float history_base_quaternion_wxyz[GEAR_SONIC_HISTORY_FRAMES * 4];
    float history_base_angular_velocity[GEAR_SONIC_HISTORY_FRAMES * 3];
    float history_joint_position_policy[
        GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM];
    float history_joint_velocity_policy[
        GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM];
    float history_last_action_policy[
        GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM];
    uint8_t history_count;
    uint8_t history_head;
} RowSnapshot;

typedef struct DataSnapshot {
    mjData data;
    void* buffer;
    void* arena;
    mjtSize nbuffer;
    mjtSize narena;
} DataSnapshot;

static size_t checks = 0;

#define CHECK(expression) do { \
    checks += 1; \
    if (!(expression)) { \
        fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #expression); \
        goto fail; \
    } \
} while (0)

static float* read_exact_floats(const char* path, size_t count) {
    FILE* stream = fopen(path, "rb");
    if (stream == NULL || count > SIZE_MAX / sizeof(float)) {
        if (stream != NULL) fclose(stream);
        return NULL;
    }
    float* values = (float*)malloc(count * sizeof(float));
    if (values == NULL) {
        fclose(stream);
        return NULL;
    }
    int ok = fread(values, sizeof(float), count, stream) == count;
    if (ok) ok = fgetc(stream) == EOF && !ferror(stream);
    if (fclose(stream) != 0) ok = 0;
    if (!ok) {
        free(values);
        return NULL;
    }
    return values;
}

static void fill_floats(float* values, size_t count, float seed) {
    for (size_t index = 0; index < count; index++) {
        values[index] = seed + (float)index * 0.000125f;
    }
}

static void fill_doubles(double* values, size_t count, double seed) {
    for (size_t index = 0; index < count; index++) {
        values[index] = seed + (double)index * 0.000125;
    }
}

static void seed_row(GearSonicNativeDuelVector* vector, size_t row) {
    const float seed = (float)(row + 1u) * 10.0f;
    fill_doubles(vector->base_quaternion_wxyz + row * 4, 4, seed + 0.1);
    fill_doubles(
        vector->base_angular_velocity_local + row * 3, 3, seed + 0.2);
    fill_doubles(
        vector->joint_position_mujoco + row * GEAR_SONIC_ACTION_DIM,
        GEAR_SONIC_ACTION_DIM,
        seed + 0.3);
    fill_doubles(
        vector->joint_velocity_mujoco + row * GEAR_SONIC_ACTION_DIM,
        GEAR_SONIC_ACTION_DIM,
        seed + 0.4);
    fill_doubles(vector->heading_delta_wxyz + row * 4, 4, seed + 0.5);
    vector->reference_frames[row] = row + 6u;
    vector->policy_ticks[row] = UINT64_C(0x1020304050607080) + row;
    fill_floats(
        vector->command_lpf_state_mujoco + row * GEAR_SONIC_ACTION_DIM,
        GEAR_SONIC_ACTION_DIM,
        seed + 0.6f);
    vector->command_lpf_initialized[row] = 1;

    GearSonicNativeBatch* batch = &vector->controller;
    fill_floats(
        batch->encoder_observations + row * GEAR_SONIC_ENCODER_INPUT_WIDTH,
        GEAR_SONIC_ENCODER_INPUT_WIDTH,
        seed + 1.0f);
    fill_floats(
        batch->tokens + row * GEAR_SONIC_ENCODER_OUTPUT_WIDTH,
        GEAR_SONIC_ENCODER_OUTPUT_WIDTH,
        seed + 2.0f);
    fill_floats(
        batch->decoder_observations + row * GEAR_SONIC_DECODER_INPUT_WIDTH,
        GEAR_SONIC_DECODER_INPUT_WIDTH,
        seed + 3.0f);
    fill_floats(
        batch->actions_policy + row * GEAR_SONIC_ACTION_DIM,
        GEAR_SONIC_ACTION_DIM,
        seed + 4.0f);
    fill_floats(
        batch->clipped_actions_policy + row * GEAR_SONIC_ACTION_DIM,
        GEAR_SONIC_ACTION_DIM,
        seed + 5.0f);
    fill_floats(
        batch->targets_mujoco + row * GEAR_SONIC_ACTION_DIM,
        GEAR_SONIC_ACTION_DIM,
        seed + 6.0f);
    const size_t history_row = row * GEAR_SONIC_HISTORY_FRAMES;
    fill_floats(
        batch->history_base_quaternion_wxyz + history_row * 4,
        GEAR_SONIC_HISTORY_FRAMES * 4,
        seed + 7.0f);
    fill_floats(
        batch->history_base_angular_velocity + history_row * 3,
        GEAR_SONIC_HISTORY_FRAMES * 3,
        seed + 8.0f);
    fill_floats(
        batch->history_joint_position_policy
            + history_row * GEAR_SONIC_ACTION_DIM,
        GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM,
        seed + 9.0f);
    fill_floats(
        batch->history_joint_velocity_policy
            + history_row * GEAR_SONIC_ACTION_DIM,
        GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM,
        seed + 10.0f);
    fill_floats(
        batch->history_last_action_policy
            + history_row * GEAR_SONIC_ACTION_DIM,
        GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM,
        seed + 11.0f);
    batch->history_count[row] = 9;
    batch->history_head[row] = 7;
}

static void capture_row(
        const GearSonicNativeDuelVector* vector,
        size_t row,
        RowSnapshot* snapshot) {
    memset(snapshot, 0, sizeof(*snapshot));
    memcpy(
        snapshot->base_quaternion_wxyz,
        vector->base_quaternion_wxyz + row * 4,
        sizeof(snapshot->base_quaternion_wxyz));
    memcpy(
        snapshot->base_angular_velocity_local,
        vector->base_angular_velocity_local + row * 3,
        sizeof(snapshot->base_angular_velocity_local));
    memcpy(
        snapshot->joint_position_mujoco,
        vector->joint_position_mujoco + row * GEAR_SONIC_ACTION_DIM,
        sizeof(snapshot->joint_position_mujoco));
    memcpy(
        snapshot->joint_velocity_mujoco,
        vector->joint_velocity_mujoco + row * GEAR_SONIC_ACTION_DIM,
        sizeof(snapshot->joint_velocity_mujoco));
    memcpy(
        snapshot->heading_delta_wxyz,
        vector->heading_delta_wxyz + row * 4,
        sizeof(snapshot->heading_delta_wxyz));
    snapshot->reference_frame = vector->reference_frames[row];
    snapshot->policy_tick = vector->policy_ticks[row];
    memcpy(
        snapshot->command_lpf_state_mujoco,
        vector->command_lpf_state_mujoco + row * GEAR_SONIC_ACTION_DIM,
        sizeof(snapshot->command_lpf_state_mujoco));
    snapshot->command_lpf_initialized = vector->command_lpf_initialized[row];

    const GearSonicNativeBatch* batch = &vector->controller;
    memcpy(
        snapshot->encoder_observations,
        batch->encoder_observations + row * GEAR_SONIC_ENCODER_INPUT_WIDTH,
        sizeof(snapshot->encoder_observations));
    memcpy(
        snapshot->tokens,
        batch->tokens + row * GEAR_SONIC_ENCODER_OUTPUT_WIDTH,
        sizeof(snapshot->tokens));
    memcpy(
        snapshot->decoder_observations,
        batch->decoder_observations + row * GEAR_SONIC_DECODER_INPUT_WIDTH,
        sizeof(snapshot->decoder_observations));
    memcpy(
        snapshot->actions_policy,
        batch->actions_policy + row * GEAR_SONIC_ACTION_DIM,
        sizeof(snapshot->actions_policy));
    memcpy(
        snapshot->clipped_actions_policy,
        batch->clipped_actions_policy + row * GEAR_SONIC_ACTION_DIM,
        sizeof(snapshot->clipped_actions_policy));
    memcpy(
        snapshot->targets_mujoco,
        batch->targets_mujoco + row * GEAR_SONIC_ACTION_DIM,
        sizeof(snapshot->targets_mujoco));
    const size_t history_row = row * GEAR_SONIC_HISTORY_FRAMES;
    memcpy(
        snapshot->history_base_quaternion_wxyz,
        batch->history_base_quaternion_wxyz + history_row * 4,
        sizeof(snapshot->history_base_quaternion_wxyz));
    memcpy(
        snapshot->history_base_angular_velocity,
        batch->history_base_angular_velocity + history_row * 3,
        sizeof(snapshot->history_base_angular_velocity));
    memcpy(
        snapshot->history_joint_position_policy,
        batch->history_joint_position_policy
            + history_row * GEAR_SONIC_ACTION_DIM,
        sizeof(snapshot->history_joint_position_policy));
    memcpy(
        snapshot->history_joint_velocity_policy,
        batch->history_joint_velocity_policy
            + history_row * GEAR_SONIC_ACTION_DIM,
        sizeof(snapshot->history_joint_velocity_policy));
    memcpy(
        snapshot->history_last_action_policy,
        batch->history_last_action_policy
            + history_row * GEAR_SONIC_ACTION_DIM,
        sizeof(snapshot->history_last_action_policy));
    snapshot->history_count = batch->history_count[row];
    snapshot->history_head = batch->history_head[row];
}

static int capture_data(const mjData* data, DataSnapshot* snapshot) {
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->data = *data;
    snapshot->nbuffer = data->nbuffer;
    snapshot->narena = data->narena;
    if (data->nbuffer < 0 || data->narena < 0
            || (uintmax_t)data->nbuffer > SIZE_MAX
            || (uintmax_t)data->narena > SIZE_MAX) {
        return 0;
    }
    if (data->nbuffer > 0) {
        snapshot->buffer = malloc((size_t)data->nbuffer);
        if (snapshot->buffer == NULL) return 0;
        memcpy(snapshot->buffer, data->buffer, (size_t)data->nbuffer);
    }
    if (data->narena > 0) {
        snapshot->arena = malloc((size_t)data->narena);
        if (snapshot->arena == NULL) {
            free(snapshot->buffer);
            snapshot->buffer = NULL;
            return 0;
        }
        memcpy(snapshot->arena, data->arena, (size_t)data->narena);
    }
    return 1;
}

static int data_matches(const mjData* data, const DataSnapshot* snapshot) {
    return data->nbuffer == snapshot->nbuffer
        && data->narena == snapshot->narena
        && memcmp(data, &snapshot->data, sizeof(*data)) == 0
        && (data->nbuffer == 0
            || memcmp(data->buffer, snapshot->buffer, (size_t)data->nbuffer) == 0)
        && (data->narena == 0
            || memcmp(data->arena, snapshot->arena, (size_t)data->narena) == 0);
}

static int all_zero(const void* bytes, size_t count) {
    const uint8_t* values = (const uint8_t*)bytes;
    for (size_t index = 0; index < count; index++) {
        if (values[index] != 0) return 0;
    }
    return 1;
}

typedef enum DirectiveProbeMode {
    DIRECTIVE_PROBE_RESET = 0,
    DIRECTIVE_PROBE_INVALID = 1,
    DIRECTIVE_PROBE_REJECT = 2,
} DirectiveProbeMode;

typedef struct DirectiveProbe {
    GearSonicNativeDuelVector* vector;
    const double* canonical_qpos;
    size_t arena_count;
    size_t calls;
    size_t reset_arena;
    uint32_t reset_substep;
    DirectiveProbeMode mode;
    mjtNum requested_reset_time;
    int saw_reset_before_later_callback;
    int invalid_observation;
} DirectiveProbe;

static int apply_post_step_directive(
        void* opaque,
        const GearSonicNativeDuelPostStepObservation* observation,
        GearSonicNativeDuelPostStepDirective* directive) {
    DirectiveProbe* probe = (DirectiveProbe*)opaque;
    if (probe == NULL || observation == NULL || directive == NULL
            || probe->vector == NULL || probe->arena_count == 0) {
        return 0;
    }
    const uint32_t expected_substep
        = (uint32_t)(probe->calls / probe->arena_count);
    const size_t expected_arena = probe->calls % probe->arena_count;
    const double expected_time = 0.002 * (double)(expected_substep + 1u);
    if (observation->vector != probe->vector
            || observation->model != probe->vector->model
            || observation->data != probe->vector->data[expected_arena]
            || observation->arena_index != expected_arena
            || observation->physics_substep_index != expected_substep
            || observation->physics_dt_seconds != 0.002
            || fabs(observation->data->time - expected_time) > 1e-12) {
        probe->invalid_observation = 1;
        return 0;
    }
    if (probe->mode == DIRECTIVE_PROBE_RESET
            && expected_substep == probe->reset_substep
            && expected_arena == probe->reset_arena + 1u) {
        const mjData* reset_data = probe->vector->data[probe->reset_arena];
        int reset_state_matches = probe->canonical_qpos != NULL
            && memcmp(
                &reset_data->time,
                &probe->requested_reset_time,
                sizeof(reset_data->time)) == 0
            && memcmp(
                reset_data->qpos,
                probe->canonical_qpos,
                GEAR_SONIC_DUEL_QPOS_DIM * sizeof(double)) == 0
            && all_zero(
                reset_data->qvel,
                GEAR_SONIC_DUEL_QVEL_DIM * sizeof(double))
            && all_zero(
                reset_data->ctrl,
                GEAR_SONIC_DUEL_CONTROL_DIM * sizeof(double));
        for (size_t fighter = 0;
                fighter < GEAR_SONIC_DUEL_FIGHTERS && reset_state_matches;
                fighter++) {
            const size_t row
                = probe->reset_arena * GEAR_SONIC_DUEL_FIGHTERS + fighter;
            const size_t history_row = row * GEAR_SONIC_HISTORY_FRAMES;
            reset_state_matches = probe->vector->policy_ticks[row] == 0
                && probe->vector->reference_frames[row] == 0
                && probe->vector->command_lpf_initialized[row] == 0
                && all_zero(
                    probe->vector->command_lpf_state_mujoco
                        + row * GEAR_SONIC_ACTION_DIM,
                    GEAR_SONIC_ACTION_DIM * sizeof(float))
                && probe->vector->controller.history_count[row] == 0
                && probe->vector->controller.history_head[row] == 0
                && all_zero(
                    probe->vector->controller.clipped_actions_policy
                        + row * GEAR_SONIC_ACTION_DIM,
                    GEAR_SONIC_ACTION_DIM * sizeof(float))
                && all_zero(
                    probe->vector->controller.history_base_quaternion_wxyz
                        + history_row * 4,
                    GEAR_SONIC_HISTORY_FRAMES * 4 * sizeof(float))
                && all_zero(
                    probe->vector->controller.history_base_angular_velocity
                        + history_row * 3,
                    GEAR_SONIC_HISTORY_FRAMES * 3 * sizeof(float))
                && all_zero(
                    probe->vector->controller.history_joint_position_policy
                        + history_row * GEAR_SONIC_ACTION_DIM,
                    GEAR_SONIC_HISTORY_FRAMES
                        * GEAR_SONIC_ACTION_DIM * sizeof(float))
                && all_zero(
                    probe->vector->controller.history_joint_velocity_policy
                        + history_row * GEAR_SONIC_ACTION_DIM,
                    GEAR_SONIC_HISTORY_FRAMES
                        * GEAR_SONIC_ACTION_DIM * sizeof(float))
                && all_zero(
                    probe->vector->controller.history_last_action_policy
                        + history_row * GEAR_SONIC_ACTION_DIM,
                    GEAR_SONIC_HISTORY_FRAMES
                        * GEAR_SONIC_ACTION_DIM * sizeof(float));
        }
        if (!reset_state_matches) {
            probe->invalid_observation = 1;
            return 0;
        }
        probe->saw_reset_before_later_callback = 1;
    }
    probe->calls += 1;
    if (probe->mode == DIRECTIVE_PROBE_INVALID) {
        *directive = (GearSonicNativeDuelPostStepDirective)99;
        return 1;
    }
    *directive = GEAR_SONIC_DUEL_POST_STEP_CONTINUE;
    if (probe->mode == DIRECTIVE_PROBE_REJECT
            && expected_arena == probe->reset_arena
            && expected_substep == probe->reset_substep) {
        return 0;
    }
    if (probe->mode == DIRECTIVE_PROBE_RESET
            && expected_arena == probe->reset_arena
            && expected_substep == probe->reset_substep) {
        probe->requested_reset_time = observation->data->time;
        *directive = GEAR_SONIC_DUEL_POST_STEP_RESET_ARENA_IMMEDIATE;
    }
    return 1;
}

int main(int argc, char** argv) {
    int result = 1;
    float* dof = NULL;
    float* position = NULL;
    float* rotation = NULL;
    float* reference_position = NULL;
    float* reference_next_position = NULL;
    float* reference_rotation = NULL;
    double canonical_qpos[GEAR_SONIC_DUEL_QPOS_DIM];
    double canonical_heading[GEAR_SONIC_DUEL_FIGHTERS][4];
    RowSnapshot untouched_rows[GEAR_SONIC_DUEL_FIGHTERS];
    DataSnapshot untouched_data = {0};
    GearSonicNativeDuelVector vector = {0};
    char error[1024] = {0};
    if (argc != 8) {
        fprintf(
            stderr,
            "usage: %s MODEL ENCODER DECODER FRAMES DOF ROOT_POSITION ROOT_ROTATION\n",
            argv[0]);
        return 64;
    }
    char* end = NULL;
    const unsigned long parsed_frames = strtoul(argv[4], &end, 10);
    if (end == argv[4] || *end != '\0' || parsed_frames == 0
            || parsed_frames > SIZE_MAX / GEAR_SONIC_ACTION_DIM) {
        return 64;
    }
    const size_t frames = (size_t)parsed_frames;
    dof = read_exact_floats(argv[5], frames * GEAR_SONIC_ACTION_DIM);
    position = read_exact_floats(argv[6], frames * 3);
    rotation = read_exact_floats(argv[7], frames * 4);
    CHECK(dof != NULL && position != NULL && rotation != NULL);
    const GearSonicNativeMotion motion = {
        .dof_position_mujoco = dof,
        .root_position_m = position,
        .root_rotation_xyzw = rotation,
        .frames = frames,
        .loop = 1,
    };
    CHECK(gear_sonic_native_duel_open(
        &vector,
        argv[1],
        argv[2],
        argv[3],
        motion,
        2,
        1,
        error,
        sizeof(error)));
    CHECK(vector.arena_count == 2 && vector.robot_count == 4);
    memcpy(canonical_qpos, vector.data[0]->qpos, sizeof(canonical_qpos));
    for (size_t fighter = 0; fighter < GEAR_SONIC_DUEL_FIGHTERS; fighter++) {
        memcpy(
            canonical_heading[fighter],
            vector.heading_delta_wxyz + fighter * 4,
            sizeof(canonical_heading[fighter]));
    }

    for (size_t row = 0; row < vector.robot_count; row++) seed_row(&vector, row);
    for (size_t arena = 0; arena < vector.arena_count; arena++) {
        mjData* data = vector.data[arena];
        data->time = arena == 0 ? 7.125 : 19.75;
        data->qpos[vector.fighters[0].qpos_addresses[0]]
            += arena == 0 ? 0.03125 : -0.046875;
        for (int index = 0; index < vector.model->nv; index++) {
            data->qvel[index] = (double)(arena + 1u) * 0.01 + (double)index * 0.001;
        }
        for (int index = 0; index < vector.model->nu; index++) {
            data->ctrl[index] = (double)(arena + 1u) * 0.1 + (double)index * 0.002;
        }
        mj_forward(vector.model, data);
    }
    const mjtNum preserved_time = vector.data[0]->time;
    for (size_t fighter = 0; fighter < GEAR_SONIC_DUEL_FIGHTERS; fighter++) {
        capture_row(&vector, 2u + fighter, &untouched_rows[fighter]);
    }
    CHECK(capture_data(vector.data[1], &untouched_data));

    CHECK(gear_sonic_native_duel_reset_arena_immediate(
        &vector, 0, error, sizeof(error)));
    CHECK(!vector.failed && !vector.controller.failed);
    CHECK(memcmp(&vector.data[0]->time, &preserved_time, sizeof(preserved_time)) == 0);
    CHECK(memcmp(vector.data[0]->qpos, canonical_qpos, sizeof(canonical_qpos)) == 0);
    CHECK(all_zero(
        vector.data[0]->qvel,
        GEAR_SONIC_DUEL_QVEL_DIM * sizeof(vector.data[0]->qvel[0])));
    CHECK(all_zero(
        vector.data[0]->ctrl,
        GEAR_SONIC_DUEL_CONTROL_DIM * sizeof(vector.data[0]->ctrl[0])));
    for (size_t fighter = 0; fighter < GEAR_SONIC_DUEL_FIGHTERS; fighter++) {
        const size_t row = fighter;
        const size_t history_row = row * GEAR_SONIC_HISTORY_FRAMES;
        CHECK(vector.policy_ticks[row] == 0);
        CHECK(vector.reference_frames[row] == 0);
        CHECK(vector.command_lpf_initialized[row] == 0);
        CHECK(all_zero(
            vector.command_lpf_state_mujoco + row * GEAR_SONIC_ACTION_DIM,
            GEAR_SONIC_ACTION_DIM * sizeof(float)));
        CHECK(memcmp(
            vector.heading_delta_wxyz + row * 4,
            canonical_heading[fighter],
            sizeof(canonical_heading[fighter])) == 0);
        CHECK(vector.controller.history_count[row] == 0);
        CHECK(vector.controller.history_head[row] == 0);
        CHECK(all_zero(
            vector.controller.clipped_actions_policy
                + row * GEAR_SONIC_ACTION_DIM,
            GEAR_SONIC_ACTION_DIM * sizeof(float)));
        CHECK(all_zero(
            vector.controller.history_base_quaternion_wxyz + history_row * 4,
            GEAR_SONIC_HISTORY_FRAMES * 4 * sizeof(float)));
        CHECK(all_zero(
            vector.controller.history_base_angular_velocity + history_row * 3,
            GEAR_SONIC_HISTORY_FRAMES * 3 * sizeof(float)));
        CHECK(all_zero(
            vector.controller.history_joint_position_policy
                + history_row * GEAR_SONIC_ACTION_DIM,
            GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM * sizeof(float)));
        CHECK(all_zero(
            vector.controller.history_joint_velocity_policy
                + history_row * GEAR_SONIC_ACTION_DIM,
            GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM * sizeof(float)));
        CHECK(all_zero(
            vector.controller.history_last_action_policy
                + history_row * GEAR_SONIC_ACTION_DIM,
            GEAR_SONIC_HISTORY_FRAMES * GEAR_SONIC_ACTION_DIM * sizeof(float)));
    }
    CHECK(data_matches(vector.data[1], &untouched_data));
    for (size_t fighter = 0; fighter < GEAR_SONIC_DUEL_FIGHTERS; fighter++) {
        RowSnapshot after;
        capture_row(&vector, 2u + fighter, &after);
        CHECK(memcmp(&after, &untouched_rows[fighter], sizeof(after)) == 0);
    }

    free(untouched_data.arena);
    free(untouched_data.buffer);
    memset(&untouched_data, 0, sizeof(untouched_data));
    const size_t reference_rows
        = vector.robot_count * GEAR_SONIC_HISTORY_FRAMES;
    const size_t reference_position_count
        = reference_rows * GEAR_SONIC_ACTION_DIM;
    reference_position = (float*)malloc(
        reference_position_count * sizeof(float));
    reference_next_position = (float*)malloc(
        reference_position_count * sizeof(float));
    reference_rotation = (float*)malloc(reference_rows * 4 * sizeof(float));
    CHECK(reference_position != NULL
        && reference_next_position != NULL
        && reference_rotation != NULL);
    for (size_t row = 0; row < vector.robot_count; row++) {
        for (size_t future = 0; future < GEAR_SONIC_HISTORY_FRAMES; future++) {
            const size_t frame = future % frames;
            const size_t next = (frame + 1u) % frames;
            const size_t reference_row = row * GEAR_SONIC_HISTORY_FRAMES + future;
            memcpy(
                reference_position + reference_row * GEAR_SONIC_ACTION_DIM,
                dof + frame * GEAR_SONIC_ACTION_DIM,
                GEAR_SONIC_ACTION_DIM * sizeof(float));
            memcpy(
                reference_next_position + reference_row * GEAR_SONIC_ACTION_DIM,
                dof + next * GEAR_SONIC_ACTION_DIM,
                GEAR_SONIC_ACTION_DIM * sizeof(float));
            memcpy(
                reference_rotation + reference_row * 4,
                rotation + frame * 4,
                4 * sizeof(float));
        }
    }
    const GearSonicNativeReferenceInput references = {
        .dof_position_mujoco = reference_position,
        .dof_next_position_mujoco = reference_next_position,
        .root_rotation_xyzw = reference_rotation,
    };

    CHECK(gear_sonic_native_duel_reset(&vector, error, sizeof(error)));
    CHECK(gear_sonic_native_duel_step_references(
        &vector, &references, error, sizeof(error)));
    CHECK(capture_data(vector.data[1], &untouched_data));
    for (size_t fighter = 0; fighter < GEAR_SONIC_DUEL_FIGHTERS; fighter++) {
        capture_row(&vector, 2u + fighter, &untouched_rows[fighter]);
    }

    CHECK(gear_sonic_native_duel_reset(&vector, error, sizeof(error)));
    DirectiveProbe reset_probe = {
        .vector = &vector,
        .canonical_qpos = canonical_qpos,
        .arena_count = vector.arena_count,
        .reset_arena = 0,
        .reset_substep = 2,
        .mode = DIRECTIVE_PROBE_RESET,
    };
    CHECK(gear_sonic_native_duel_step_references_with_post_step_directive(
        &vector,
        &references,
        apply_post_step_directive,
        &reset_probe,
        error,
        sizeof(error)));
    CHECK(!reset_probe.invalid_observation);
    CHECK(reset_probe.saw_reset_before_later_callback);
    CHECK(reset_probe.calls == vector.arena_count * 10u);
    CHECK(fabs(vector.data[0]->time - 0.02) <= 1e-12
        && fabs(vector.data[1]->time - 0.02) <= 1e-12);
    CHECK(vector.reset_completed_in_step_arenas[0] == 1u);
    CHECK(vector.reset_completed_in_step_arenas[1] == 0u);
    CHECK(vector.policy_ticks[0] == 0 && vector.policy_ticks[1] == 0);
    CHECK(data_matches(vector.data[1], &untouched_data));
    for (size_t fighter = 0; fighter < GEAR_SONIC_DUEL_FIGHTERS; fighter++) {
        RowSnapshot after;
        capture_row(&vector, 2u + fighter, &after);
        CHECK(memcmp(&after, &untouched_rows[fighter], sizeof(after)) == 0);
    }

    CHECK(gear_sonic_native_duel_reset(&vector, error, sizeof(error)));
    DirectiveProbe invalid_probe = {
        .vector = &vector,
        .arena_count = vector.arena_count,
        .mode = DIRECTIVE_PROBE_INVALID,
    };
    CHECK(!gear_sonic_native_duel_step_references_with_post_step_directive(
        &vector,
        &references,
        apply_post_step_directive,
        &invalid_probe,
        error,
        sizeof(error)));
    CHECK(vector.failed && invalid_probe.calls == 1);
    CHECK(strstr(error, "invalid post-step directive") != NULL);
    CHECK(fabs(vector.data[0]->time - 0.002) <= 1e-12
        && fabs(vector.data[1]->time - 0.002) <= 1e-12);
    CHECK(vector.policy_ticks[0] == 0 && vector.policy_ticks[1] == 0);

    CHECK(gear_sonic_native_duel_reset(&vector, error, sizeof(error)));
    DirectiveProbe rejection_probe = {
        .vector = &vector,
        .arena_count = vector.arena_count,
        .reset_arena = 1,
        .reset_substep = 1,
        .mode = DIRECTIVE_PROBE_REJECT,
    };
    CHECK(!gear_sonic_native_duel_step_references_with_post_step_directive(
        &vector,
        &references,
        apply_post_step_directive,
        &rejection_probe,
        error,
        sizeof(error)));
    CHECK(vector.failed && rejection_probe.calls == 4);
    CHECK(strstr(error, "post-step callback rejected arena 1 substep 1") != NULL);
    CHECK(fabs(vector.data[0]->time - 0.004) <= 1e-12
        && fabs(vector.data[1]->time - 0.004) <= 1e-12);
    CHECK(vector.policy_ticks[0] == 0 && vector.policy_ticks[1] == 0);
    CHECK(gear_sonic_native_duel_reset(&vector, error, sizeof(error)));

    printf(
        "{\"checks\":%zu,\"arena_count\":2,"
        "\"target_time_bitwise_preserved\":true,"
        "\"target_spawn_and_joint_qpos_restored\":true,"
        "\"target_qvel_and_ctrl_zeroed\":true,"
        "\"target_two_controller_rows_reset\":true,"
        "\"other_mjdata_struct_buffer_arena_bitwise_unchanged\":true,"
        "\"other_two_rows_bitwise_unchanged\":true,"
        "\"poststep_directive_order_exact\":true,"
        "\"reset_completed_before_later_callback\":true,"
        "\"subsequent_substeps_continued\":true,"
        "\"ordinary_other_arena_result_bitwise_unchanged\":true,"
        "\"invalid_directive_failed_closed\":true,"
        "\"callback_rejection_failed_closed\":true,"
        "\"delayed_rek_round_reset_claim\":false,"
        "\"rek_parity_claim\":false}\n",
        checks);
    result = 0;

fail:
    if (result != 0 && error[0] != '\0') fprintf(stderr, "%s\n", error);
    free(untouched_data.arena);
    free(untouched_data.buffer);
    gear_sonic_native_duel_close(&vector);
    free(reference_rotation);
    free(reference_next_position);
    free(reference_position);
    free(rotation);
    free(position);
    free(dof);
    return result;
}
