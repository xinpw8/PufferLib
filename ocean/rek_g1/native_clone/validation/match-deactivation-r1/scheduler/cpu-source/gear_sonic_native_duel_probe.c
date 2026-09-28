#define _POSIX_C_SOURCE 200809L

#include "gear_sonic_native_duel.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define OUTPUT_COUNT 4

typedef struct NewOutput {
    const char* path;
    FILE* stream;
    int owned;
} NewOutput;

static int checked_product(size_t left, size_t right, size_t* result) {
    if (left != 0 && right > SIZE_MAX / left) return 0;
    *result = left * right;
    return 1;
}

static int host_is_little_endian(void) {
    const uint16_t value = 1;
    return *((const uint8_t*)&value) == 1;
}

static int parse_positive_size(const char* text, size_t* result) {
    char* end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || value == 0
            || value > SIZE_MAX) {
        return 0;
    }
    *result = (size_t)value;
    return 1;
}

static int parse_positive_int(const char* text, int* result) {
    size_t value = 0;
    if (!parse_positive_size(text, &value) || value > INT32_MAX) return 0;
    *result = (int)value;
    return 1;
}

static float* read_exact_floats(const char* path, size_t count) {
    if (count > SIZE_MAX / sizeof(float)) return NULL;
    FILE* stream = fopen(path, "rb");
    if (stream == NULL) return NULL;
    float* values = (float*)malloc(count * sizeof(float));
    if (values == NULL) {
        fclose(stream);
        return NULL;
    }
    int ok = fread(values, sizeof(float), count, stream) == count
        && fgetc(stream) == EOF
        && !ferror(stream);
    if (fclose(stream) != 0) ok = 0;
    if (!ok) {
        free(values);
        return NULL;
    }
    return values;
}

static void rollback_outputs(NewOutput outputs[OUTPUT_COUNT]) {
    for (size_t index = 0; index < OUTPUT_COUNT; index++) {
        if (outputs[index].stream != NULL) {
            fclose(outputs[index].stream);
            outputs[index].stream = NULL;
        }
    }
    for (size_t index = 0; index < OUTPUT_COUNT; index++) {
        if (outputs[index].owned) {
            remove(outputs[index].path);
            outputs[index].owned = 0;
        }
    }
}

static int open_outputs(NewOutput outputs[OUTPUT_COUNT]) {
    for (size_t left = 0; left < OUTPUT_COUNT; left++) {
        for (size_t right = left + 1; right < OUTPUT_COUNT; right++) {
            if (strcmp(outputs[left].path, outputs[right].path) == 0) return 0;
        }
    }
    for (size_t index = 0; index < OUTPUT_COUNT; index++) {
        outputs[index].stream = fopen(outputs[index].path, "wbx");
        if (outputs[index].stream == NULL) {
            rollback_outputs(outputs);
            return 0;
        }
        outputs[index].owned = 1;
    }
    return 1;
}

static int write_output(
        FILE* stream, const void* values, size_t count, size_t width) {
    if (stream == NULL || values == NULL
            || (count != 0 && width > SIZE_MAX / count)) {
        return 0;
    }
    return fwrite(values, width, count, stream) == count;
}

static int finish_outputs(NewOutput outputs[OUTPUT_COUNT]) {
    int ok = 1;
    for (size_t index = 0; index < OUTPUT_COUNT; index++) {
        if (outputs[index].stream == NULL || fclose(outputs[index].stream) != 0) {
            ok = 0;
        }
        outputs[index].stream = NULL;
    }
    if (!ok) {
        rollback_outputs(outputs);
        return 0;
    }
    for (size_t index = 0; index < OUTPUT_COUNT; index++) {
        outputs[index].owned = 0;
    }
    return 1;
}

static double elapsed_seconds(struct timespec start, struct timespec stop) {
    return (double)(stop.tv_sec - start.tv_sec)
        + (double)(stop.tv_nsec - start.tv_nsec) * 1e-9;
}

static void free_motion(
        float* motion_position,
        float* root_position,
        float* root_rotation) {
    free(root_rotation);
    free(root_position);
    free(motion_position);
}

int main(int argc, char** argv) {
    if (argc != 16) {
        fprintf(
            stderr,
            "usage: %s MODEL ENCODER DECODER ARENAS WORKERS STEPS FRAMES "
            "clamp|loop DOF ROOT_POSITION ROOT_ROTATION QPOS_OUT QVEL_OUT "
            "ACTION_OUT TARGET_OUT\n",
            argv[0]);
        return 64;
    }
    if (!host_is_little_endian()) {
        fprintf(stderr, "native raw-file probe requires a little-endian host\n");
        return 69;
    }
    size_t arena_count = 0;
    size_t robot_count = 0;
    size_t steps = 0;
    size_t frames = 0;
    int workers = 0;
    if (!parse_positive_size(argv[4], &arena_count)
            || !checked_product(
                arena_count, GEAR_SONIC_DUEL_FIGHTERS, &robot_count)
            || !parse_positive_int(argv[5], &workers)
            || !parse_positive_size(argv[6], &steps)
            || !parse_positive_size(argv[7], &frames)) {
        return 64;
    }
    const int loop = strcmp(argv[8], "loop") == 0;
    if (!loop && strcmp(argv[8], "clamp") != 0) return 64;
    size_t motion_action_count = 0;
    size_t motion_position_count = 0;
    size_t motion_rotation_count = 0;
    size_t qpos_count = 0;
    size_t qvel_count = 0;
    size_t action_count = 0;
    if (!checked_product(frames, GEAR_SONIC_ACTION_DIM, &motion_action_count)
            || !checked_product(frames, 3, &motion_position_count)
            || !checked_product(frames, 4, &motion_rotation_count)
            || !checked_product(
                arena_count, GEAR_SONIC_DUEL_QPOS_DIM, &qpos_count)
            || !checked_product(
                arena_count, GEAR_SONIC_DUEL_QVEL_DIM, &qvel_count)
            || !checked_product(robot_count, GEAR_SONIC_ACTION_DIM, &action_count)
            || qpos_count > SIZE_MAX / sizeof(double)
            || qvel_count > SIZE_MAX / sizeof(double)
            || action_count > SIZE_MAX / sizeof(float)) {
        return 64;
    }
    float* motion_position = read_exact_floats(argv[9], motion_action_count);
    float* root_position = read_exact_floats(argv[10], motion_position_count);
    float* root_rotation = read_exact_floats(argv[11], motion_rotation_count);
    if (motion_position == NULL || root_position == NULL || root_rotation == NULL) {
        fprintf(stderr, "failed to read exact motion arrays\n");
        free_motion(motion_position, root_position, root_rotation);
        return 66;
    }
    GearSonicNativeMotion motion = {
        .dof_position_mujoco = motion_position,
        .root_position_m = root_position,
        .root_rotation_xyzw = root_rotation,
        .frames = frames,
        .loop = loop,
    };
    GearSonicNativeDuelVector vector = {0};
    char error[1024] = {0};
    if (!gear_sonic_native_duel_open(
            &vector,
            argv[1],
            argv[2],
            argv[3],
            motion,
            arena_count,
            workers,
            error,
            sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        free_motion(motion_position, root_position, root_rotation);
        return 1;
    }
    struct timespec start = {0};
    struct timespec stop = {0};
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        fprintf(stderr, "failed to start monotonic timer\n");
        gear_sonic_native_duel_close(&vector);
        free_motion(motion_position, root_position, root_rotation);
        return 70;
    }
    for (size_t step = 0; step < steps; step++) {
        if (!gear_sonic_native_duel_step_fixed(&vector, error, sizeof(error))) {
            fprintf(stderr, "%s\n", error);
            gear_sonic_native_duel_close(&vector);
            free_motion(motion_position, root_position, root_rotation);
            return 1;
        }
    }
    if (clock_gettime(CLOCK_MONOTONIC, &stop) != 0) {
        fprintf(stderr, "failed to stop monotonic timer\n");
        gear_sonic_native_duel_close(&vector);
        free_motion(motion_position, root_position, root_rotation);
        return 70;
    }
    const double seconds = elapsed_seconds(start, stop);
    if (!isfinite(seconds) || seconds <= 0.0) {
        fprintf(stderr, "monotonic timer produced an invalid duration\n");
        gear_sonic_native_duel_close(&vector);
        free_motion(motion_position, root_position, root_rotation);
        return 70;
    }
    double* qpos = (double*)malloc(qpos_count * sizeof(double));
    double* qvel = (double*)malloc(qvel_count * sizeof(double));
    if (qpos == NULL || qvel == NULL) {
        fprintf(stderr, "state output allocation failed\n");
        free(qvel);
        free(qpos);
        gear_sonic_native_duel_close(&vector);
        free_motion(motion_position, root_position, root_rotation);
        return 70;
    }
    double root_z_min = vector.data[0]->qpos[
        vector.fighters[GEAR_SONIC_DUEL_PLAYER].root_qpos_address + 2];
    double root_z_max = root_z_min;
    int contact_min = vector.data[0]->ncon;
    int contact_max = contact_min;
    for (size_t arena_index = 0; arena_index < arena_count; arena_index++) {
        memcpy(
            qpos + arena_index * GEAR_SONIC_DUEL_QPOS_DIM,
            vector.data[arena_index]->qpos,
            GEAR_SONIC_DUEL_QPOS_DIM * sizeof(double));
        memcpy(
            qvel + arena_index * GEAR_SONIC_DUEL_QVEL_DIM,
            vector.data[arena_index]->qvel,
            GEAR_SONIC_DUEL_QVEL_DIM * sizeof(double));
        if (vector.data[arena_index]->ncon < contact_min) {
            contact_min = vector.data[arena_index]->ncon;
        }
        if (vector.data[arena_index]->ncon > contact_max) {
            contact_max = vector.data[arena_index]->ncon;
        }
        for (size_t fighter_index = 0;
                fighter_index < GEAR_SONIC_DUEL_FIGHTERS;
                fighter_index++) {
            const double root_z = vector.data[arena_index]->qpos[
                vector.fighters[fighter_index].root_qpos_address + 2];
            if (root_z < root_z_min) root_z_min = root_z;
            if (root_z > root_z_max) root_z_max = root_z;
        }
    }
    NewOutput outputs[OUTPUT_COUNT] = {
        {.path = argv[12]},
        {.path = argv[13]},
        {.path = argv[14]},
        {.path = argv[15]},
    };
    const int write_ok = open_outputs(outputs)
        && write_output(outputs[0].stream, qpos, qpos_count, sizeof(double))
        && write_output(outputs[1].stream, qvel, qvel_count, sizeof(double))
        && write_output(
            outputs[2].stream,
            vector.controller.actions_policy,
            action_count,
            sizeof(float))
        && write_output(
            outputs[3].stream,
            vector.controller.targets_mujoco,
            action_count,
            sizeof(float))
        && finish_outputs(outputs);
    if (!write_ok) {
        rollback_outputs(outputs);
        fprintf(stderr, "failed to write rollout outputs\n");
        free(qvel);
        free(qpos);
        gear_sonic_native_duel_close(&vector);
        free_motion(motion_position, root_position, root_rotation);
        return 74;
    }
    printf(
        "{\"arena_count\":%zu,\"physical_robot_count\":%zu,"
        "\"controller_batch_size\":%zu,\"mjdata_count\":%zu,"
        "\"policy_steps\":%zu,\"physics_workers\":%d,"
        "\"seconds\":%.9g,\"arena_steps_per_second\":%.9g,"
        "\"physical_robot_samples_per_second\":%.9g,"
        "\"physics_steps_per_policy_step\":10,"
        "\"qpos_dim_per_arena\":72,\"qvel_dim_per_arena\":70,"
        "\"control_dim_per_arena\":58,\"spawn_prefixes_verified\":true,"
        "\"shared_contact_physics_per_arena\":true,"
        "\"final_contact_count_min\":%d,\"final_contact_count_max\":%d,"
        "\"root_z_min_m\":%.17g,\"root_z_max_m\":%.17g,"
        "\"motion_routing\":\"fixed_shared_reference\","
        "\"semantic_routing\":\"fail_closed_unavailable\","
        "\"classification\":\"public_family_candidate\","
        "\"rek_parity_claim\":false}\n",
        arena_count,
        robot_count,
        vector.controller.batch_size,
        arena_count,
        steps,
        workers,
        seconds,
        ((double)arena_count * (double)steps) / seconds,
        ((double)robot_count * (double)steps) / seconds,
        contact_min,
        contact_max,
        root_z_min,
        root_z_max);
    free(qvel);
    free(qpos);
    gear_sonic_native_duel_close(&vector);
    free_motion(motion_position, root_position, root_rotation);
    return 0;
}
