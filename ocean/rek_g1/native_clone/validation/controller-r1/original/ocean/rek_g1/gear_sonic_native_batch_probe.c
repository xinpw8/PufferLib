#define _POSIX_C_SOURCE 200809L

#include "gear_sonic_native_batch.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static float probe_float(size_t row, size_t column, size_t salt) {
    size_t value = (row * 131u + column * 17u + salt * 29u) % 2003u;
    return ((float)value - 1001.0f) / 317.0f;
}

static double probe_double(size_t row, size_t column, size_t salt) {
    size_t value = (row * 137u + column * 19u + salt * 31u) % 2011u;
    return ((double)value - 1005.0) / 911.0;
}

static int parse_positive(const char* text, size_t* result) {
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

static int write_new(const char* path, const float* values, size_t count) {
    FILE* stream = fopen(path, "wbx");
    if (stream == NULL) return 0;
    int ok = fwrite(values, sizeof(float), count, stream) == count;
    if (fclose(stream) != 0) ok = 0;
    return ok;
}

static double elapsed_seconds(struct timespec start, struct timespec stop) {
    return (double)(stop.tv_sec - start.tv_sec)
        + (double)(stop.tv_nsec - start.tv_nsec) * 1e-9;
}

int main(int argc, char** argv) {
    if (argc != 10) {
        fprintf(
            stderr,
            "usage: %s ENCODER DECODER BATCH ITERATIONS ENCODER_OUT TOKEN_OUT "
            "DECODER_OUT ACTION_OUT TARGET_OUT\n",
            argv[0]);
        return 64;
    }
    size_t batch_size = 0;
    size_t iterations = 0;
    if (!parse_positive(argv[3], &batch_size)
            || !parse_positive(argv[4], &iterations)
            || batch_size > SIZE_MAX / GEAR_SONIC_ACTION_DIM) {
        return 64;
    }
    const size_t frames = 67;
    size_t action_count = batch_size * GEAR_SONIC_ACTION_DIM;
    float* motion_position = calloc(frames * GEAR_SONIC_ACTION_DIM, sizeof(float));
    float* motion_rotation = calloc(frames * 4, sizeof(float));
    double* base_quaternion = calloc(batch_size * 4, sizeof(double));
    double* base_angular_velocity = calloc(batch_size * 3, sizeof(double));
    double* joint_position = calloc(action_count, sizeof(double));
    double* joint_velocity = calloc(action_count, sizeof(double));
    double* heading_delta = calloc(batch_size * 4, sizeof(double));
    size_t* reference_frames = calloc(batch_size, sizeof(size_t));
    if (motion_position == NULL || motion_rotation == NULL
            || base_quaternion == NULL || base_angular_velocity == NULL
            || joint_position == NULL || joint_velocity == NULL
            || heading_delta == NULL || reference_frames == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 70;
    }
    for (size_t frame = 0; frame < frames; frame++) {
        for (size_t joint = 0; joint < GEAR_SONIC_ACTION_DIM; joint++) {
            motion_position[frame * GEAR_SONIC_ACTION_DIM + joint]
                = probe_float(frame, joint, 5);
        }
        motion_rotation[frame * 4 + 3] = 1.0f;
    }
    for (size_t row = 0; row < batch_size; row++) {
        base_quaternion[row * 4] = 1.0;
        heading_delta[row * 4] = 1.0;
    }

    GearSonicOrtBatch ort = {0};
    GearSonicNativeBatch native = {0};
    char error[1024] = {0};
    if (!gear_sonic_ort_open(
            &ort, argv[1], argv[2], batch_size, error, sizeof(error))
            || !gear_sonic_native_batch_open(
                &native, batch_size, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        gear_sonic_native_batch_close(&native);
        gear_sonic_ort_close(&ort);
        return 1;
    }
    GearSonicNativeMotion motion = {
        .dof_position_mujoco = motion_position,
        .root_rotation_xyzw = motion_rotation,
        .frames = frames,
        .loop = 1,
    };
    GearSonicNativeStateInput state = {
        .base_quaternion_wxyz = base_quaternion,
        .base_angular_velocity_local = base_angular_velocity,
        .joint_position_mujoco = joint_position,
        .joint_velocity_mujoco = joint_velocity,
        .heading_delta_wxyz = heading_delta,
        .reference_frames = reference_frames,
    };

    struct timespec start = {0};
    struct timespec stop = {0};
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (size_t iteration = 0; iteration < iterations; iteration++) {
        for (size_t row = 0; row < batch_size; row++) {
            reference_frames[row] = (iteration + row * 3) % frames;
            for (size_t axis = 0; axis < 3; axis++) {
                base_angular_velocity[row * 3 + axis]
                    = probe_double(row + iteration, axis, 7);
            }
            for (size_t joint = 0; joint < GEAR_SONIC_ACTION_DIM; joint++) {
                joint_position[row * GEAR_SONIC_ACTION_DIM + joint]
                    = probe_double(row + iteration, joint, 11);
                joint_velocity[row * GEAR_SONIC_ACTION_DIM + joint]
                    = probe_double(row + iteration, joint, 13);
            }
        }
        if (!gear_sonic_native_batch_step(
                &native, &ort, &motion, &state, error, sizeof(error))) {
            fprintf(stderr, "%s\n", error);
            gear_sonic_native_batch_close(&native);
            gear_sonic_ort_close(&ort);
            return 1;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &stop);
    double seconds = elapsed_seconds(start, stop);
    size_t encoder_count = batch_size * GEAR_SONIC_ENCODER_INPUT_WIDTH;
    size_t token_count = batch_size * GEAR_SONIC_ENCODER_OUTPUT_WIDTH;
    size_t decoder_count = batch_size * GEAR_SONIC_DECODER_INPUT_WIDTH;
    int write_ok = write_new(argv[5], native.encoder_observations, encoder_count)
        && write_new(argv[6], native.tokens, token_count)
        && write_new(argv[7], native.decoder_observations, decoder_count)
        && write_new(argv[8], native.actions_policy, action_count)
        && write_new(argv[9], native.targets_mujoco, action_count);
    if (!write_ok) {
        fprintf(stderr, "failed to write native batch probe outputs\n");
        gear_sonic_native_batch_close(&native);
        gear_sonic_ort_close(&ort);
        return 74;
    }
    printf(
        "{\"batch_size\":%zu,\"iterations\":%zu,\"seconds\":%.9g,"
        "\"controller_samples_per_second\":%.9g,\"history_count\":%u}\n",
        batch_size,
        iterations,
        seconds,
        (double)(batch_size * iterations) / seconds,
        (unsigned int)native.history_count[0]);
    gear_sonic_native_batch_close(&native);
    gear_sonic_ort_close(&ort);
    free(reference_frames);
    free(heading_delta);
    free(joint_velocity);
    free(joint_position);
    free(base_angular_velocity);
    free(base_quaternion);
    free(motion_rotation);
    free(motion_position);
    return 0;
}
