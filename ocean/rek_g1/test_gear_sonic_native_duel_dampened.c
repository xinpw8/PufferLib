#include "gear_sonic_native_duel.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ControllerRowSnapshot {
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
} ControllerRowSnapshot;

typedef struct ControlProbe {
    const GearSonicNativeDuelVector* vector;
    double expected[GEAR_SONIC_ACTION_DIM];
    size_t calls;
    int checked;
    int invalid;
} ControlProbe;

typedef struct DeferredResetProbe {
    GearSonicNativeDuelVector* vector;
    ControllerRowSnapshot frozen_controller[GEAR_SONIC_DUEL_FIGHTERS];
    float frozen_lpf[GEAR_SONIC_DUEL_FIGHTERS][GEAR_SONIC_ACTION_DIM];
    uint8_t frozen_lpf_initialized[GEAR_SONIC_DUEL_FIGHTERS];
    double canonical_root_qpos[GEAR_SONIC_DUEL_FIGHTERS][7];
    size_t calls;
    int completed;
    int invalid;
    char error[1024];
} DeferredResetProbe;

static size_t checks = 0;

static int all_zero(const void* memory, size_t bytes);

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

static void capture_controller_row(
        const GearSonicNativeDuelVector* vector,
        size_t row,
        ControllerRowSnapshot* snapshot) {
    memset(snapshot, 0, sizeof(*snapshot));
    const GearSonicNativeBatch* controller = &vector->controller;
    const size_t action_offset = row * GEAR_SONIC_ACTION_DIM;
    const size_t history_offset = row * GEAR_SONIC_HISTORY_FRAMES;
    memcpy(
        snapshot->actions_policy,
        controller->actions_policy + action_offset,
        sizeof(snapshot->actions_policy));
    memcpy(
        snapshot->clipped_actions_policy,
        controller->clipped_actions_policy + action_offset,
        sizeof(snapshot->clipped_actions_policy));
    memcpy(
        snapshot->targets_mujoco,
        controller->targets_mujoco + action_offset,
        sizeof(snapshot->targets_mujoco));
    memcpy(
        snapshot->history_base_quaternion_wxyz,
        controller->history_base_quaternion_wxyz + history_offset * 4,
        sizeof(snapshot->history_base_quaternion_wxyz));
    memcpy(
        snapshot->history_base_angular_velocity,
        controller->history_base_angular_velocity + history_offset * 3,
        sizeof(snapshot->history_base_angular_velocity));
    memcpy(
        snapshot->history_joint_position_policy,
        controller->history_joint_position_policy
            + history_offset * GEAR_SONIC_ACTION_DIM,
        sizeof(snapshot->history_joint_position_policy));
    memcpy(
        snapshot->history_joint_velocity_policy,
        controller->history_joint_velocity_policy
            + history_offset * GEAR_SONIC_ACTION_DIM,
        sizeof(snapshot->history_joint_velocity_policy));
    memcpy(
        snapshot->history_last_action_policy,
        controller->history_last_action_policy
            + history_offset * GEAR_SONIC_ACTION_DIM,
        sizeof(snapshot->history_last_action_policy));
    snapshot->history_count = controller->history_count[row];
    snapshot->history_head = controller->history_head[row];
}

static int observe_controls(
        void* opaque,
        const GearSonicNativeDuelPostStepObservation* observation) {
    ControlProbe* probe = (ControlProbe*)opaque;
    if (probe == NULL || observation == NULL || probe->vector == NULL) return 0;
    probe->calls += 1;
    if (observation->arena_index != 0 || observation->physics_substep_index != 0) {
        return 1;
    }
    const GearSonicDuelFighterMap* map
        = &probe->vector->fighters[GEAR_SONIC_DUEL_PLAYER];
    for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
        const double observed = observation->data->ctrl[map->actuator_ids[index]];
        const double expected = probe->expected[index];
        const double tolerance = 1e-12 * fmax(1.0, fabs(expected));
        if (!isfinite(observed) || fabs(observed - expected) > tolerance) {
            probe->invalid = 1;
            return 0;
        }
    }
    probe->checked = 1;
    return 1;
}

static int observe_deferred_reset(
        void* opaque,
        const GearSonicNativeDuelPostStepObservation* observation) {
    DeferredResetProbe* probe = (DeferredResetProbe*)opaque;
    if (probe == NULL || observation == NULL || probe->vector == NULL) return 0;
    probe->calls += 1;
    if (observation->arena_index != 0 || observation->physics_substep_index != 0) {
        return 1;
    }
    for (size_t fighter = 0; fighter < GEAR_SONIC_DUEL_FIGHTERS; fighter++) {
        ControllerRowSnapshot observed;
        capture_controller_row(probe->vector, fighter, &observed);
        if (memcmp(
                &observed,
                &probe->frozen_controller[fighter],
                sizeof(observed)) != 0
                || memcmp(
                    probe->vector->command_lpf_state_mujoco
                        + fighter * GEAR_SONIC_ACTION_DIM,
                    probe->frozen_lpf[fighter],
                    sizeof(probe->frozen_lpf[fighter])) != 0
                || probe->vector->command_lpf_initialized[fighter]
                    != probe->frozen_lpf_initialized[fighter]) {
            probe->invalid = 1;
            return 0;
        }
    }

    double root_qvel[GEAR_SONIC_DUEL_FIGHTERS][6];
    for (size_t fighter = 0; fighter < GEAR_SONIC_DUEL_FIGHTERS; fighter++) {
        const GearSonicDuelFighterMap* map = &probe->vector->fighters[fighter];
        memcpy(
            root_qvel[fighter],
            observation->data->qvel + map->root_qvel_address,
            sizeof(root_qvel[fighter]));
    }
    const double completion_time = observation->data->time;
    if (!gear_sonic_native_duel_complete_arena_reset(
            probe->vector, 0, probe->error, sizeof(probe->error))) {
        probe->invalid = 1;
        return 0;
    }
    if (observation->data->time != completion_time
            || probe->vector->reset_pending_arenas[0]
            || probe->vector->reset_complete_not_before_time[0] != 0.0) {
        probe->invalid = 1;
        return 0;
    }
    for (size_t fighter = 0; fighter < GEAR_SONIC_DUEL_FIGHTERS; fighter++) {
        const size_t row = fighter;
        const GearSonicDuelFighterMap* map = &probe->vector->fighters[fighter];
        if (memcmp(
                observation->data->qpos + map->root_qpos_address,
                probe->canonical_root_qpos[fighter],
                sizeof(probe->canonical_root_qpos[fighter])) != 0
                || memcmp(
                    observation->data->qvel + map->root_qvel_address,
                    root_qvel[fighter],
                    sizeof(root_qvel[fighter])) != 0
                || probe->vector->dampened_rows[row]
                || probe->vector->resetting_rows[row]
                || probe->vector->policy_ticks[row] != 0
                || probe->vector->motion_ticks[row] != 0
                || probe->vector->reference_frames[row] != 0
                || probe->vector->command_lpf_initialized[row]
                || !all_zero(
                    probe->vector->command_lpf_state_mujoco
                        + row * GEAR_SONIC_ACTION_DIM,
                    GEAR_SONIC_ACTION_DIM * sizeof(float))) {
            probe->invalid = 1;
            return 0;
        }
        for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
            if (observation->data->qpos[map->qpos_addresses[index]] != 0.0
                    || observation->data->qvel[map->qvel_addresses[index]] != 0.0) {
                probe->invalid = 1;
                return 0;
            }
        }
        ControllerRowSnapshot reset_controller;
        capture_controller_row(probe->vector, row, &reset_controller);
        if (!all_zero(&reset_controller, sizeof(reset_controller))) {
            probe->invalid = 1;
            return 0;
        }
    }
    probe->completed = 1;
    return 1;
}

static int all_zero(const void* memory, size_t bytes) {
    const unsigned char* values = (const unsigned char*)memory;
    for (size_t index = 0; index < bytes; index++) {
        if (values[index] != 0) return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
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
    float* dof = read_exact_floats(
        argv[5], frames * GEAR_SONIC_ACTION_DIM);
    float* position = read_exact_floats(argv[6], frames * 3);
    float* rotation = read_exact_floats(argv[7], frames * 4);
    GearSonicNativeDuelVector vector = {0};
    mjtNum* gain_copy = NULL;
    mjtNum* bias_copy = NULL;
    mjtNum* force_copy = NULL;
    int result = 1;
    char error[1024] = {0};
    CHECK(dof != NULL && position != NULL && rotation != NULL);

    GearSonicNativeMotion motion = {
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
        4,
        1,
        error,
        sizeof(error)));
    CHECK(gear_sonic_native_duel_step_fixed(&vector, error, sizeof(error)));

    const size_t gain_bytes
        = (size_t)vector.model->nu * mjNGAIN * sizeof(mjtNum);
    const size_t bias_bytes
        = (size_t)vector.model->nu * mjNBIAS * sizeof(mjtNum);
    const size_t force_bytes
        = (size_t)vector.model->nu * 2 * sizeof(mjtNum);
    gain_copy = (mjtNum*)malloc(gain_bytes);
    bias_copy = (mjtNum*)malloc(bias_bytes);
    force_copy = (mjtNum*)malloc(force_bytes);
    CHECK(gain_copy != NULL && bias_copy != NULL && force_copy != NULL);
    memcpy(gain_copy, vector.model->actuator_gainprm, gain_bytes);
    memcpy(bias_copy, vector.model->actuator_biasprm, bias_bytes);
    memcpy(force_copy, vector.model->actuator_forcerange, force_bytes);

    const size_t dampened_row = 0;
    const size_t active_row = 2;
    const size_t dampened_offset = dampened_row * GEAR_SONIC_ACTION_DIM;
    const GearSonicDuelFighterMap* dampened_map
        = &vector.fighters[GEAR_SONIC_DUEL_PLAYER];
    ControllerRowSnapshot frozen_controller;
    capture_controller_row(&vector, dampened_row, &frozen_controller);
    float frozen_lpf[GEAR_SONIC_ACTION_DIM];
    memcpy(
        frozen_lpf,
        vector.command_lpf_state_mujoco + dampened_offset,
        sizeof(frozen_lpf));
    const uint8_t frozen_lpf_initialized
        = vector.command_lpf_initialized[dampened_row];
    const size_t frozen_reference_frame = vector.reference_frames[dampened_row];
    const uint64_t frozen_motion_tick = vector.motion_ticks[dampened_row];
    const uint64_t frozen_policy_tick = vector.policy_ticks[dampened_row];
    const uint64_t active_motion_tick = vector.motion_ticks[active_row];
    const uint64_t active_policy_tick = vector.policy_ticks[active_row];
    const double initial_time = vector.data[0]->time;
    float expected_target[GEAR_SONIC_ACTION_DIM];
    for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
        expected_target[index]
            = (float)vector.data[0]->ctrl[dampened_map->actuator_ids[index]];
    }

    CHECK(gear_sonic_native_duel_set_row_dampened(
        &vector, dampened_row, 1, error, sizeof(error)));
    CHECK(vector.dampened_rows[dampened_row] == 1);
    uint32_t retention_bits = 0;
    memcpy(
        &retention_bits,
        &(float){GEAR_SONIC_DUEL_DAMPEN_RETENTION},
        sizeof(retention_bits));
    CHECK(retention_bits == UINT32_C(0x3DCCCCCD));
    CHECK(memcmp(
        vector.dampened_control_target_mujoco + dampened_offset,
        expected_target,
        sizeof(expected_target)) == 0);
    for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
        const int actuator_id = dampened_map->actuator_ids[index];
        const float expected_kp
            = (float)gain_copy[actuator_id * mjNGAIN]
                * GEAR_SONIC_DUEL_DAMPEN_RETENTION;
        const float expected_kd
            = (float)(-bias_copy[actuator_id * mjNBIAS + 2])
                * GEAR_SONIC_DUEL_DAMPEN_RETENTION;
        const float expected_force
            = (float)force_copy[2 * actuator_id + 1]
                * GEAR_SONIC_DUEL_DAMPEN_RETENTION;
        CHECK(memcmp(
            vector.dampened_kp_mujoco + dampened_offset + index,
            &expected_kp,
            sizeof(float)) == 0);
        CHECK(memcmp(
            vector.dampened_kd_mujoco + dampened_offset + index,
            &expected_kd,
            sizeof(float)) == 0);
        CHECK(memcmp(
            vector.dampened_force_limit_mujoco + dampened_offset + index,
            &expected_force,
            sizeof(float)) == 0);
    }
    CHECK(memcmp(gain_copy, vector.model->actuator_gainprm, gain_bytes) == 0);
    CHECK(memcmp(bias_copy, vector.model->actuator_biasprm, bias_bytes) == 0);
    CHECK(memcmp(force_copy, vector.model->actuator_forcerange, force_bytes) == 0);

    ControlProbe probe = {.vector = &vector};
    for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
        const int actuator_id = dampened_map->actuator_ids[index];
        const int qpos_address = dampened_map->qpos_addresses[index];
        const int qvel_address = dampened_map->qvel_addresses[index];
        const double kp
            = (double)vector.dampened_kp_mujoco[dampened_offset + index];
        const double kd
            = (double)vector.dampened_kd_mujoco[dampened_offset + index];
        const double limit = (double)
            vector.dampened_force_limit_mujoco[dampened_offset + index];
        const double target = (double)
            vector.dampened_control_target_mujoco[dampened_offset + index];
        const double joint_position = vector.data[0]->qpos[qpos_address];
        const double joint_velocity = vector.data[0]->qvel[qvel_address];
        double force = kp * (target - joint_position) - kd * joint_velocity;
        if (force < -limit) force = -limit;
        if (force > limit) force = limit;
        const mjtNum* gain
            = vector.model->actuator_gainprm + actuator_id * mjNGAIN;
        const mjtNum* bias
            = vector.model->actuator_biasprm + actuator_id * mjNBIAS;
        const double base_bias = bias[0]
            + bias[1] * joint_position
            + bias[2] * joint_velocity;
        probe.expected[index] = (force - base_bias) / gain[0];
    }
    CHECK(gear_sonic_native_duel_step_fixed_with_post_step_observer(
        &vector, observe_controls, &probe, error, sizeof(error)));
    CHECK(probe.checked && !probe.invalid && probe.calls == 40);
    CHECK(gear_sonic_native_duel_step_fixed(&vector, error, sizeof(error)));
    CHECK(gear_sonic_native_duel_step_fixed(&vector, error, sizeof(error)));

    ControllerRowSnapshot observed_controller;
    capture_controller_row(&vector, dampened_row, &observed_controller);
    CHECK(memcmp(
        &observed_controller,
        &frozen_controller,
        sizeof(frozen_controller)) == 0);
    CHECK(memcmp(
        vector.command_lpf_state_mujoco + dampened_offset,
        frozen_lpf,
        sizeof(frozen_lpf)) == 0);
    CHECK(vector.command_lpf_initialized[dampened_row] == frozen_lpf_initialized);
    CHECK(vector.reference_frames[dampened_row] == frozen_reference_frame);
    CHECK(vector.motion_ticks[dampened_row] == frozen_motion_tick);
    CHECK(vector.policy_ticks[dampened_row] == frozen_policy_tick);
    CHECK(vector.motion_ticks[active_row] == active_motion_tick + 3);
    CHECK(vector.policy_ticks[active_row] == active_policy_tick + 3);
    CHECK(fabs(vector.data[0]->time - (initial_time + 0.06)) <= 1e-12);
    CHECK(memcmp(gain_copy, vector.model->actuator_gainprm, gain_bytes) == 0);
    CHECK(memcmp(bias_copy, vector.model->actuator_biasprm, bias_bytes) == 0);
    CHECK(memcmp(force_copy, vector.model->actuator_forcerange, force_bytes) == 0);

    CHECK(gear_sonic_native_duel_set_row_dampened(
        &vector, dampened_row, 0, error, sizeof(error)));
    CHECK(!vector.dampened_rows[dampened_row]);
    CHECK(all_zero(
        vector.dampened_kp_mujoco + dampened_offset,
        GEAR_SONIC_ACTION_DIM * sizeof(float)));
    CHECK(gear_sonic_native_duel_step_fixed(&vector, error, sizeof(error)));
    CHECK(vector.motion_ticks[dampened_row] == frozen_motion_tick + 1);
    CHECK(vector.reference_frames[dampened_row]
        == (size_t)(frozen_motion_tick % frames));
    CHECK(vector.policy_ticks[dampened_row] == frozen_policy_tick + 1);

    CHECK(gear_sonic_native_duel_set_row_dampened(
        &vector, 0, 1, error, sizeof(error)));
    CHECK(gear_sonic_native_duel_set_row_dampened(
        &vector, 2, 1, error, sizeof(error)));
    CHECK(gear_sonic_native_duel_reset_arena_immediate(
        &vector, 0, error, sizeof(error)));
    CHECK(!vector.dampened_rows[0] && !vector.dampened_rows[1]);
    CHECK(vector.dampened_rows[2] && !vector.dampened_rows[3]);
    CHECK(vector.motion_ticks[0] == 0 && vector.motion_ticks[1] == 0);
    CHECK(gear_sonic_native_duel_reset(&vector, error, sizeof(error)));
    CHECK(all_zero(
        vector.dampened_rows,
        vector.robot_count * sizeof(uint8_t)));
    CHECK(memcmp(gain_copy, vector.model->actuator_gainprm, gain_bytes) == 0);
    CHECK(memcmp(bias_copy, vector.model->actuator_biasprm, bias_bytes) == 0);
    CHECK(memcmp(force_copy, vector.model->actuator_forcerange, force_bytes) == 0);

    DeferredResetProbe reset_probe = {.vector = &vector};
    for (size_t fighter = 0; fighter < GEAR_SONIC_DUEL_FIGHTERS; fighter++) {
        const GearSonicDuelFighterMap* map = &vector.fighters[fighter];
        memcpy(
            reset_probe.canonical_root_qpos[fighter],
            vector.data[0]->qpos + map->root_qpos_address,
            sizeof(reset_probe.canonical_root_qpos[fighter]));
    }
    CHECK(gear_sonic_native_duel_step_fixed(&vector, error, sizeof(error)));
    mjData* reset_data = vector.data[0];
    for (size_t fighter = 0; fighter < GEAR_SONIC_DUEL_FIGHTERS; fighter++) {
        const GearSonicDuelFighterMap* map = &vector.fighters[fighter];
        reset_data->qpos[map->root_qpos_address] += 0.25 + (double)fighter * 0.1;
        reset_data->qpos[map->root_qpos_address + 1] += 0.125;
        for (size_t axis = 0; axis < 6; axis++) {
            reset_data->qvel[map->root_qvel_address + (int)axis]
                = 0.5 + (double)fighter + (double)axis * 0.03125;
        }
        for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
            const int joint_id = map->joint_ids[index];
            const double lower = vector.model->jnt_range[2 * joint_id];
            const double upper = vector.model->jnt_range[2 * joint_id + 1];
            reset_data->qpos[map->qpos_addresses[index]]
                = lower + (upper - lower) * 0.45;
            reset_data->qvel[map->qvel_addresses[index]]
                = 0.0625 + (double)index * 0.001;
        }
    }
    mj_forward(vector.model, reset_data);
    double preserved_joint_qpos[
        GEAR_SONIC_DUEL_FIGHTERS][GEAR_SONIC_ACTION_DIM];
    double preserved_qvel[GEAR_SONIC_DUEL_QVEL_DIM];
    memcpy(preserved_qvel, reset_data->qvel, sizeof(preserved_qvel));
    for (size_t fighter = 0; fighter < GEAR_SONIC_DUEL_FIGHTERS; fighter++) {
        const GearSonicDuelFighterMap* map = &vector.fighters[fighter];
        for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
            preserved_joint_qpos[fighter][index]
                = reset_data->qpos[map->qpos_addresses[index]];
        }
        capture_controller_row(
            &vector, fighter, &reset_probe.frozen_controller[fighter]);
        memcpy(
            reset_probe.frozen_lpf[fighter],
            vector.command_lpf_state_mujoco
                + fighter * GEAR_SONIC_ACTION_DIM,
            sizeof(reset_probe.frozen_lpf[fighter]));
        reset_probe.frozen_lpf_initialized[fighter]
            = vector.command_lpf_initialized[fighter];
    }
    CHECK(gear_sonic_native_duel_set_row_dampened(
        &vector, 0, 1, error, sizeof(error)));
    const double reset_begin_time = reset_data->time;
    CHECK(gear_sonic_native_duel_begin_arena_reset(
        &vector, 0, error, sizeof(error)));
    CHECK(vector.reset_pending_arenas[0]);
    CHECK(vector.reset_complete_not_before_time[0] == reset_begin_time + 0.002);
    CHECK(vector.resetting_rows[0] && vector.resetting_rows[1]);
    CHECK(vector.dampened_rows[0] && !vector.dampened_rows[1]);
    CHECK(reset_data->time == reset_begin_time);
    for (size_t fighter = 0; fighter < GEAR_SONIC_DUEL_FIGHTERS; fighter++) {
        const GearSonicDuelFighterMap* map = &vector.fighters[fighter];
        CHECK(memcmp(
            reset_data->qpos + map->root_qpos_address,
            reset_probe.canonical_root_qpos[fighter],
            sizeof(reset_probe.canonical_root_qpos[fighter])) == 0);
        for (size_t index = 0; index < GEAR_SONIC_ACTION_DIM; index++) {
            CHECK(reset_data->qpos[map->qpos_addresses[index]]
                == preserved_joint_qpos[fighter][index]);
        }
    }
    CHECK(memcmp(reset_data->qvel, preserved_qvel, sizeof(preserved_qvel)) == 0);
    CHECK(!gear_sonic_native_duel_complete_arena_reset(
        &vector, 0, error, sizeof(error)));
    CHECK(!vector.failed && strstr(error, "next 2 ms") != NULL);
    CHECK(gear_sonic_native_duel_step_fixed_with_post_step_observer(
        &vector, observe_deferred_reset, &reset_probe, error, sizeof(error)));
    CHECK(reset_probe.completed && !reset_probe.invalid && reset_probe.calls == 40);
    CHECK(vector.reset_completed_in_step_arenas[0]);
    CHECK(vector.policy_ticks[0] == 0 && vector.policy_ticks[1] == 0);
    CHECK(vector.motion_ticks[0] == 0 && vector.motion_ticks[1] == 0);
    CHECK(memcmp(gain_copy, vector.model->actuator_gainprm, gain_bytes) == 0);
    CHECK(memcmp(bias_copy, vector.model->actuator_biasprm, bias_bytes) == 0);
    CHECK(memcmp(force_copy, vector.model->actuator_forcerange, force_bytes) == 0);

    CHECK(gear_sonic_native_duel_step_fixed(
        &vector, error, sizeof(error)));
    CHECK(!vector.reset_completed_in_step_arenas[0]);
    CHECK(vector.policy_ticks[0] == 1 && vector.policy_ticks[1] == 1);
    CHECK(vector.motion_ticks[0] == 1 && vector.motion_ticks[1] == 1);

    CHECK(gear_sonic_native_duel_begin_arena_reset(
        &vector, 1, error, sizeof(error)));
    CHECK(!gear_sonic_native_duel_begin_arena_reset(
        &vector, 1, error, sizeof(error)));
    CHECK(vector.failed && strstr(error, "already pending") != NULL);

    printf(
        "{\"checks\":%zu,\"arena_count\":4,"
        "\"float32_retention_bits\":\"0x3DCCCCCD\","
        "\"row_local_dampening\":true,"
        "\"suspended_policy_tick_frozen\":true,"
        "\"motion_reference_frozen\":true,"
        "\"controller_and_lpf_frozen\":true,"
        "\"physics_time_advanced\":true,"
        "\"retained_pd_control_verified\":true,"
        "\"shared_model_bitwise_unchanged\":true,"
        "\"arena_reset_isolated\":true,"
        "\"deferred_reset_root_first\":true,"
        "\"deferred_reset_joint_state_preserved_at_begin\":true,"
        "\"deferred_reset_completed_on_next_2ms_boundary\":true,"
        "\"deferred_reset_controller_state_cleared\":true,"
        "\"deferred_reset_root_qvel_preserved\":true,"
        "\"duplicate_reset_begin_rejected\":true,"
        "\"current_steam_gain_authority\":false,"
        "\"same_tick_ordering_claim\":false,"
        "\"rek_trajectory_parity_claim\":false}\n",
        checks);
    result = 0;

fail:
    if (result != 0 && error[0] != '\0') fprintf(stderr, "%s\n", error);
    free(force_copy);
    free(bias_copy);
    free(gain_copy);
    gear_sonic_native_duel_close(&vector);
    free(rotation);
    free(position);
    free(dof);
    return result;
}
