#pragma once

#include <stddef.h>

#include <onnxruntime_c_api.h>

#define GEAR_SONIC_ORT_VERSION "1.22.1"
#define GEAR_SONIC_ENCODER_INPUT_WIDTH 1762
#define GEAR_SONIC_ENCODER_OUTPUT_WIDTH 64
#define GEAR_SONIC_DECODER_INPUT_WIDTH 994
#define GEAR_SONIC_DECODER_OUTPUT_WIDTH 29

typedef struct GearSonicOrtBatch {
    const OrtApi* api;
    OrtEnv* environment;
    OrtSessionOptions* session_options;
    OrtSession* encoder;
    OrtSession* decoder;
    OrtMemoryInfo* cpu_memory;
    size_t batch_size;
} GearSonicOrtBatch;

/*
 * The batch must be zero-initialized before its first open. Reopening a live
 * batch is rejected without mutation; close it before opening it again.
 */
int gear_sonic_ort_open(
    GearSonicOrtBatch* batch,
    const char* encoder_path,
    const char* decoder_path,
    size_t batch_size,
    char* error,
    size_t error_capacity
);

/*
 * Construct both sessions from caller-owned model bytes. ONNX Runtime consumes
 * the arrays during this call; they may be released after the call returns.
 * The same zero-initialization and close-before-reopen contract applies.
 */
int gear_sonic_ort_open_from_memory(
    GearSonicOrtBatch* batch,
    const void* encoder_data,
    size_t encoder_byte_count,
    const void* decoder_data,
    size_t decoder_byte_count,
    size_t batch_size,
    char* error,
    size_t error_capacity
);

int gear_sonic_ort_encode(
    GearSonicOrtBatch* batch,
    float* observations,
    float* tokens,
    char* error,
    size_t error_capacity
);

int gear_sonic_ort_decode(
    GearSonicOrtBatch* batch,
    float* observations,
    float* actions,
    char* error,
    size_t error_capacity
);

void gear_sonic_ort_close(GearSonicOrtBatch* batch);
