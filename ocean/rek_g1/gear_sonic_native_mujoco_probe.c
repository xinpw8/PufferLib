#define _POSIX_C_SOURCE 200809L

#include "gear_sonic_native_mujoco.h"

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

int main(int argc, char** argv) {
    if (argc != 16) {
        fprintf(
            stderr,
            "usage: %s MODEL ENCODER DECODER BATCH WORKERS STEPS FRAMES "
            "clamp|loop DOF ROOT_POSITION ROOT_ROTATION QPOS_OUT QVEL_OUT "
            "ACTION_OUT TARGET_OUT\n",
            argv[0]);
        return 64;
    }
    if (!host_is_little_endian()) {
        fprintf(stderr, "native raw-file probe requires a little-endian host\n");
        return 69;
    }
    size_t batch_size = 0;
    size_t steps = 0;
    size_t frames = 0;
    int workers = 0;
    if (!parse_positive_size(argv[4], &batch_size)
            || !parse_positive_int(argv[5], &workers)
            || !parse_positive_size(argv[6], &steps)
            || !parse_positive_size(argv[7], &frames)) {
        return 64;
    }
    int loop = strcmp(argv[8], "loop") == 0;
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
            || !checked_product(batch_size, 36, &qpos_count)
            || !checked_product(batch_size, 35, &qvel_count)
            || !checked_product(batch_size, GEAR_SONIC_ACTION_DIM, &action_count)
            || qpos_count > SIZE_MAX / sizeof(double)
            || qvel_count > SIZE_MAX / sizeof(double)
            || action_count > SIZE_MAX / sizeof(float)) {
        return 64;
    }
    float* motion_position = read_exact_floats(
        argv[9], motion_action_count);
    float* root_position = read_exact_floats(argv[10], motion_position_count);
    float* root_rotation = read_exact_floats(argv[11], motion_rotation_count);
    if (motion_position == NULL || root_position == NULL || root_rotation == NULL) {
        fprintf(stderr, "failed to read exact motion arrays\n");
        free(root_rotation);
        free(root_position);
        free(motion_position);
        return 66;
    }
    GearSonicNativeMotion motion = {
        .dof_position_mujoco = motion_position,
        .root_position_m = root_position,
        .root_rotation_xyzw = root_rotation,
        .frames = frames,
        .loop = loop,
    };
    GearSonicNativeMujocoVector vector = {0};
    char error[1024] = {0};
    if (!gear_sonic_native_mujoco_open(
            &vector,
            argv[1],
            argv[2],
            argv[3],
            motion,
            batch_size,
            workers,
            error,
            sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        free(root_rotation);
        free(root_position);
        free(motion_position);
        return 1;
    }
    struct timespec start = {0};
    struct timespec stop = {0};
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        fprintf(stderr, "failed to start monotonic timer\n");
        gear_sonic_native_mujoco_close(&vector);
        free(root_rotation);
        free(root_position);
        free(motion_position);
        return 70;
    }
    for (size_t step = 0; step < steps; step++) {
        if (!gear_sonic_native_mujoco_step(&vector, error, sizeof(error))) {
            fprintf(stderr, "%s\n", error);
            gear_sonic_native_mujoco_close(&vector);
            free(root_rotation);
            free(root_position);
            free(motion_position);
            return 1;
        }
    }
    if (clock_gettime(CLOCK_MONOTONIC, &stop) != 0) {
        fprintf(stderr, "failed to stop monotonic timer\n");
        gear_sonic_native_mujoco_close(&vector);
        free(root_rotation);
        free(root_position);
        free(motion_position);
        return 70;
    }
    double seconds = elapsed_seconds(start, stop);
    if (!isfinite(seconds) || seconds <= 0.0) {
        fprintf(stderr, "monotonic timer produced an invalid duration\n");
        gear_sonic_native_mujoco_close(&vector);
        free(root_rotation);
        free(root_position);
        free(motion_position);
        return 70;
    }
    double* qpos = (double*)malloc(qpos_count * sizeof(double));
    double* qvel = (double*)malloc(qvel_count * sizeof(double));
    if (qpos == NULL || qvel == NULL) {
        fprintf(stderr, "state output allocation failed\n");
        free(qvel);
        free(qpos);
        gear_sonic_native_mujoco_close(&vector);
        free(root_rotation);
        free(root_position);
        free(motion_position);
        return 70;
    }
    double root_z_min = vector.data[0]->qpos[vector.root_qpos_address + 2];
    double root_z_max = root_z_min;
    for (size_t env_index = 0; env_index < batch_size; env_index++) {
        memcpy(qpos + env_index * 36, vector.data[env_index]->qpos, 36 * sizeof(double));
        memcpy(qvel + env_index * 35, vector.data[env_index]->qvel, 35 * sizeof(double));
        double root_z = vector.data[env_index]->qpos[vector.root_qpos_address + 2];
        if (root_z < root_z_min) root_z_min = root_z;
        if (root_z > root_z_max) root_z_max = root_z;
    }
    NewOutput outputs[OUTPUT_COUNT] = {
        {.path = argv[12]},
        {.path = argv[13]},
        {.path = argv[14]},
        {.path = argv[15]},
    };
    int write_ok = open_outputs(outputs)
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
        gear_sonic_native_mujoco_close(&vector);
        free(root_rotation);
        free(root_position);
        free(motion_position);
        return 74;
    }
    printf(
        "{\"batch_size\":%zu,\"policy_steps\":%zu,\"physics_workers\":%d,"
        "\"seconds\":%.9g,\"policy_samples_per_second\":%.9g,"
        "\"physics_steps_per_policy_step\":10,\"root_z_min_m\":%.17g,"
        "\"root_z_max_m\":%.17g,\"classification\":"
        "\"public_family_candidate\",\"rek_parity_claim\":false}\n",
        batch_size,
        steps,
        workers,
        seconds,
        ((double)batch_size * (double)steps) / seconds,
        root_z_min,
        root_z_max);
    free(qvel);
    free(qpos);
    gear_sonic_native_mujoco_close(&vector);
    free(root_rotation);
    free(root_position);
    free(motion_position);
    return 0;
}
