#pragma once

#include <stddef.h>
#include <stdint.h>

#include "native_motion_routes.h"
#include "semantic_binding.h"

// Vector-level Puffer-to-runtime boundary for the G1 candidate. One instance
// represents one Puffer vector buffer. It owns independent semantic scheduler
// state for every environment, but it does not own or interpret policy assets.
// The caller must pass a zero-initialized vector and close it before reopening.
// The supplied runtime callbacks must consume every semantic row in one batch.
// After a successful advance, each row with a nonzero terminal or input-reset
// completion event has its action adapter reset before the next action mask is
// written. The runtime owns the corresponding physical and episode state.

typedef struct RekG1NativePufferIO {
    const float* actions;
    size_t action_rows;
    size_t action_heads;
    void* observations;
    size_t observation_rows;
    size_t observation_stride_bytes;
    float* rewards;
    size_t reward_rows;
    float* terminals;
    size_t terminal_rows;
    uint8_t* action_masks;
    size_t action_mask_rows;
    size_t action_mask_stride_bytes;
} RekG1NativePufferIO;

typedef int (*RekG1NativeResetBatchFn)(
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
    size_t error_capacity);

typedef int (*RekG1NativeAdvanceBatchFn)(
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
    size_t error_capacity);

typedef void (*RekG1NativeCloseBatchFn)(void* context);

typedef struct RekG1NativeBatchOps {
    uint32_t runtime_facts_abi_version;
    size_t runtime_facts_size;
    RekG1NativeResetBatchFn reset;
    RekG1NativeAdvanceBatchFn advance;
    RekG1NativeCloseBatchFn close;
} RekG1NativeBatchOps;

typedef enum RekG1NativePufferStatus {
    REK_G1_NATIVE_PUFFER_OK = 0,
    REK_G1_NATIVE_PUFFER_NULL_ARGUMENT = 1,
    REK_G1_NATIVE_PUFFER_SIZE_INVALID = 2,
    REK_G1_NATIVE_PUFFER_ACTION_TABLE_INVALID = 3,
    REK_G1_NATIVE_PUFFER_ALLOCATION_FAILED = 4,
    REK_G1_NATIVE_PUFFER_NOT_READY = 5,
    REK_G1_NATIVE_PUFFER_RUNTIME_RESET_FAILED = 6,
    REK_G1_NATIVE_PUFFER_RUNTIME_ADVANCE_FAILED = 7,
    REK_G1_NATIVE_PUFFER_RUNTIME_FACTS_INVALID = 8,
    REK_G1_NATIVE_PUFFER_ACTION_REJECTED = 9,
    REK_G1_NATIVE_PUFFER_ACTION_MASK_FAILED = 10,
    REK_G1_NATIVE_PUFFER_MOTION_ROUTES_INVALID = 11,
    REK_G1_NATIVE_PUFFER_RUNTIME_ABI_MISMATCH = 12,
    REK_G1_NATIVE_PUFFER_RUNTIME_OUTPUT_INVALID = 13,
} RekG1NativePufferStatus;

typedef struct RekG1NativePufferVector {
    size_t environment_count;
    const RekG1PufferActionTable* action_table;
    const RekG1NativeMotionRouteTable* motion_routes;
    RekG1PufferAdapter* adapters;
    RekG1RuntimeFacts* facts;
    RekG1RuntimeFacts* next_facts;
    RekG1SemanticTick* semantics;
    RekG1NativeBatchOps runtime_ops;
    void* runtime_context;
    uint8_t initialized;
    uint8_t ready;
    uint8_t failed;
} RekG1NativePufferVector;

RekG1NativePufferStatus rek_g1_native_puffer_open(
    RekG1NativePufferVector* vector,
    size_t environment_count,
    const RekG1PufferActionTable* action_table,
    const RekG1NativeMotionRouteTable* motion_routes,
    RekG1NativeBatchOps runtime_ops,
    void* runtime_context);

RekG1NativePufferStatus rek_g1_native_puffer_reset(
    RekG1NativePufferVector* vector,
    RekG1NativePufferIO io,
    char* error,
    size_t error_capacity);

RekG1NativePufferStatus rek_g1_native_puffer_step(
    RekG1NativePufferVector* vector,
    RekG1NativePufferIO io,
    char* error,
    size_t error_capacity);

void rek_g1_native_puffer_close(RekG1NativePufferVector* vector);
