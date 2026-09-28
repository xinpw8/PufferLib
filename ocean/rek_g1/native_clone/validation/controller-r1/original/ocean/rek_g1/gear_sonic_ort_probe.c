#include "gear_sonic_ort.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int checked_product(size_t left, size_t right, size_t* result) {
    if (left != 0 && right > SIZE_MAX / left) return 0;
    *result = left * right;
    return 1;
}

static int write_new(
    const char* path,
    const float* values,
    size_t count,
    int* created
) {
    *created = 0;
    FILE* stream = fopen(path, "wbx");
    if (stream == NULL) return 0;
    *created = 1;
    int ok = fwrite(values, sizeof(float), count, stream) == count;
    if (fclose(stream) != 0) ok = 0;
    return ok;
}

static float probe_value(size_t row, size_t column, size_t salt) {
    size_t value = (row * 131u + column * 17u + salt * 29u) % 2003u;
    return ((float)value - 1001.0f) / 317.0f;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(
            stderr,
            "usage: %s ENCODER DECODER BATCH TOKENS_OUT ACTIONS_OUT\n",
            argv[0]
        );
        return 64;
    }
    char* end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(argv[3], &end, 10);
    if (errno == ERANGE || end == argv[3] || *end != '\0' || parsed == 0
            || parsed > SIZE_MAX) {
        return 64;
    }
    size_t batch_size = (size_t)parsed;

    size_t encoder_count = 0;
    size_t token_count = 0;
    size_t decoder_count = 0;
    size_t action_count = 0;
    if (!checked_product(
            batch_size, GEAR_SONIC_ENCODER_INPUT_WIDTH, &encoder_count)
            || !checked_product(
                batch_size, GEAR_SONIC_ENCODER_OUTPUT_WIDTH, &token_count)
            || !checked_product(
                batch_size, GEAR_SONIC_DECODER_INPUT_WIDTH, &decoder_count)
            || !checked_product(
                batch_size, GEAR_SONIC_DECODER_OUTPUT_WIDTH, &action_count)
            || encoder_count > SIZE_MAX / sizeof(float)
            || token_count > SIZE_MAX / sizeof(float)
            || decoder_count > SIZE_MAX / sizeof(float)
            || action_count > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "tensor size overflow\n");
        return 64;
    }
    float* encoder_input = calloc(encoder_count, sizeof(float));
    float* tokens = calloc(token_count, sizeof(float));
    float* decoder_input = calloc(decoder_count, sizeof(float));
    float* actions = calloc(action_count, sizeof(float));
    GearSonicOrtBatch batch = {0};
    int exit_code = 0;
    if (encoder_input == NULL || tokens == NULL
            || decoder_input == NULL || actions == NULL) {
        fprintf(stderr, "allocation failed\n");
        exit_code = 70;
        goto cleanup;
    }

    for (size_t row = 0; row < batch_size; row++) {
        encoder_input[row * GEAR_SONIC_ENCODER_INPUT_WIDTH] = 1.0f;
        for (size_t column = 4; column < 584; column++) {
            encoder_input[row * GEAR_SONIC_ENCODER_INPUT_WIDTH + column]
                = probe_value(row, column, 1);
        }
        for (size_t column = 601; column < 661; column++) {
            encoder_input[row * GEAR_SONIC_ENCODER_INPUT_WIDTH + column]
                = probe_value(row, column, 2);
        }
        for (size_t column = 0; column < GEAR_SONIC_DECODER_INPUT_WIDTH; column++) {
            decoder_input[row * GEAR_SONIC_DECODER_INPUT_WIDTH + column]
                = probe_value(row, column, 3);
        }
    }

    char error[1024] = {0};
    int ok = gear_sonic_ort_open(
            &batch, argv[1], argv[2], batch_size, error, sizeof(error))
        && gear_sonic_ort_encode(
            &batch, encoder_input, tokens, error, sizeof(error))
        && gear_sonic_ort_decode(
            &batch, decoder_input, actions, error, sizeof(error));
    if (!ok) {
        fprintf(stderr, "%s\n", error);
        exit_code = 1;
        goto cleanup;
    }
    int tokens_created = 0;
    int actions_created = 0;
    ok = write_new(argv[4], tokens, token_count, &tokens_created)
        && write_new(argv[5], actions, action_count, &actions_created);
    if (!ok) {
        fprintf(stderr, "failed to write probe output\n");
        if (actions_created && remove(argv[5]) != 0) {
            fprintf(stderr, "failed to remove incomplete action output\n");
        }
        if (tokens_created && remove(argv[4]) != 0) {
            fprintf(stderr, "failed to remove incomplete token output\n");
        }
        exit_code = 74;
        goto cleanup;
    }
    printf(
        "{\"batch_size\":%zu,\"encoder_input_width\":%d,"
        "\"token_width\":%d,\"decoder_input_width\":%d,"
        "\"action_width\":%d,\"onnxruntime\":\"%s\"}\n",
        batch_size,
        GEAR_SONIC_ENCODER_INPUT_WIDTH,
        GEAR_SONIC_ENCODER_OUTPUT_WIDTH,
        GEAR_SONIC_DECODER_INPUT_WIDTH,
        GEAR_SONIC_DECODER_OUTPUT_WIDTH,
        GEAR_SONIC_ORT_VERSION
    );
cleanup:
    gear_sonic_ort_close(&batch);
    free(actions);
    free(decoder_input);
    free(tokens);
    free(encoder_input);
    return exit_code;
}
