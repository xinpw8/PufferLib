#include "native_puffer_vector.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int io_valid(
        const RekG1NativePufferVector* vector,
        RekG1NativePufferIO io,
        int require_actions) {
    if (vector == NULL || vector->action_table == NULL
            || io.observations == NULL || io.rewards == NULL
            || io.terminals == NULL || io.action_masks == NULL
            || io.observation_stride_bytes == 0
            || io.action_mask_stride_bytes < vector->action_table->count
            || io.observation_rows != vector->environment_count
            || io.reward_rows != vector->environment_count
            || io.terminal_rows != vector->environment_count
            || io.action_mask_rows != vector->environment_count) {
        return 0;
    }
    if (!require_actions) return 1;
    return io.actions != NULL
        && io.action_rows == vector->environment_count
        && io.action_heads == REK_G1_PUFFER_ACTION_HEADS;
}

static void clear_masks(
        const RekG1NativePufferVector* vector,
        RekG1NativePufferIO io) {
    if (vector == NULL || vector->action_table == NULL
            || io.action_masks == NULL
            || io.action_mask_stride_bytes < vector->action_table->count
            || io.action_mask_rows < vector->environment_count) {
        return;
    }
    for (size_t index = 0; index < vector->environment_count; index++) {
        memset(
            io.action_masks + index * io.action_mask_stride_bytes,
            0,
            vector->action_table->count);
    }
}

static void invalidate(
        RekG1NativePufferVector* vector,
        RekG1NativePufferIO io) {
    if (vector == NULL) return;
    vector->ready = 0;
    vector->failed = 1;
    if (vector->adapters != NULL) {
        for (size_t index = 0; index < vector->environment_count; index++) {
            rek_g1_puffer_reset(&vector->adapters[index]);
        }
    }
    if (vector->facts != NULL) {
        memset(
            vector->facts,
            0,
            vector->environment_count * sizeof(RekG1RuntimeFacts));
    }
    if (vector->next_facts != NULL) {
        memset(
            vector->next_facts,
            0,
            vector->environment_count * sizeof(RekG1RuntimeFacts));
    }
    if (vector->semantics != NULL) {
        memset(
            vector->semantics,
            0,
            vector->environment_count * sizeof(RekG1SemanticTick));
    }
    clear_masks(vector, io);
}

static RekG1NativePufferStatus write_masks(
        RekG1NativePufferVector* vector,
        RekG1NativePufferIO io) {
    for (size_t index = 0; index < vector->environment_count; index++) {
        RekG1RuntimeFacts facts = vector->facts[index];
        uint8_t* mask = io.action_masks
            + index * io.action_mask_stride_bytes;
        RekG1PufferStatus status = rek_g1_puffer_write_mask(
            &vector->adapters[index],
            facts.translation_transition_settled,
            rek_g1_binding_action_busy(facts),
            mask,
            vector->action_table->count);
        if (status != REK_G1_PUFFER_OK) {
            clear_masks(vector, io);
            return REK_G1_NATIVE_PUFFER_ACTION_MASK_FAILED;
        }
    }
    return REK_G1_NATIVE_PUFFER_OK;
}

static int runtime_outputs_valid(
        const RekG1NativePufferVector* vector,
        RekG1NativePufferIO io) {
    if (vector == NULL) return 0;
    for (size_t index = 0; index < vector->environment_count; index++) {
        if (!isfinite(io.rewards[index])
                || (io.terminals[index] != 0.0f
                    && io.terminals[index] != 1.0f)) {
            return 0;
        }
    }
    return 1;
}

RekG1NativePufferStatus rek_g1_native_puffer_open(
        RekG1NativePufferVector* vector,
        size_t environment_count,
        const RekG1PufferActionTable* action_table,
        const RekG1NativeMotionRouteTable* motion_routes,
        RekG1NativeBatchOps runtime_ops,
        void* runtime_context) {
    if (vector != NULL && vector->initialized) {
        return REK_G1_NATIVE_PUFFER_NOT_READY;
    }
    if (vector == NULL || action_table == NULL || motion_routes == NULL
            || runtime_context == NULL
            || runtime_ops.reset == NULL || runtime_ops.advance == NULL) {
        return REK_G1_NATIVE_PUFFER_NULL_ARGUMENT;
    }
    if (runtime_ops.runtime_facts_abi_version
                != REK_G1_RUNTIME_FACTS_ABI_VERSION
            || runtime_ops.runtime_facts_size != REK_G1_RUNTIME_FACTS_SIZE) {
        return REK_G1_NATIVE_PUFFER_RUNTIME_ABI_MISMATCH;
    }
    if (environment_count == 0
            || environment_count > SIZE_MAX / sizeof(RekG1PufferAdapter)
            || environment_count > SIZE_MAX / sizeof(RekG1RuntimeFacts)
            || environment_count > SIZE_MAX / sizeof(RekG1SemanticTick)) {
        return REK_G1_NATIVE_PUFFER_SIZE_INVALID;
    }
    if (rek_g1_puffer_validate_table(action_table) != REK_G1_PUFFER_OK) {
        return REK_G1_NATIVE_PUFFER_ACTION_TABLE_INVALID;
    }
    if (!rek_g1_native_validate_static_motion_routes(motion_routes)) {
        return REK_G1_NATIVE_PUFFER_MOTION_ROUTES_INVALID;
    }

    *vector = (RekG1NativePufferVector){0};
    vector->environment_count = environment_count;
    vector->action_table = action_table;
    vector->motion_routes = motion_routes;
    vector->runtime_ops = runtime_ops;
    vector->runtime_context = runtime_context;
    vector->adapters = (RekG1PufferAdapter*)calloc(
        environment_count, sizeof(RekG1PufferAdapter));
    vector->facts = (RekG1RuntimeFacts*)calloc(
        environment_count, sizeof(RekG1RuntimeFacts));
    vector->next_facts = (RekG1RuntimeFacts*)calloc(
        environment_count, sizeof(RekG1RuntimeFacts));
    vector->semantics = (RekG1SemanticTick*)calloc(
        environment_count, sizeof(RekG1SemanticTick));
    if (vector->adapters == NULL || vector->facts == NULL
            || vector->next_facts == NULL || vector->semantics == NULL) {
        rek_g1_native_puffer_close(vector);
        return REK_G1_NATIVE_PUFFER_ALLOCATION_FAILED;
    }
    for (size_t index = 0; index < environment_count; index++) {
        if (rek_g1_puffer_init(
                &vector->adapters[index], action_table) != REK_G1_PUFFER_OK) {
            rek_g1_native_puffer_close(vector);
            return REK_G1_NATIVE_PUFFER_ACTION_TABLE_INVALID;
        }
    }
    vector->initialized = 1;
    return REK_G1_NATIVE_PUFFER_OK;
}

RekG1NativePufferStatus rek_g1_native_puffer_reset(
        RekG1NativePufferVector* vector,
        RekG1NativePufferIO io,
        char* error,
        size_t error_capacity) {
    if (vector == NULL || !vector->initialized) {
        return REK_G1_NATIVE_PUFFER_NOT_READY;
    }
    if (!io_valid(vector, io, 0)) {
        return REK_G1_NATIVE_PUFFER_SIZE_INVALID;
    }
    invalidate(vector, io);
    memset(io.rewards, 0, vector->environment_count * sizeof(float));
    memset(io.terminals, 0, vector->environment_count * sizeof(float));
    for (size_t index = 0; index < vector->environment_count; index++) {
        rek_g1_puffer_reset(&vector->adapters[index]);
    }
    if (!vector->runtime_ops.reset(
            vector->runtime_context,
            vector->motion_routes,
            vector->action_table,
            vector->environment_count,
            vector->facts,
            io.observations,
            io.observation_stride_bytes,
            io.rewards,
            io.terminals,
            error,
            error_capacity)) {
        invalidate(vector, io);
        return REK_G1_NATIVE_PUFFER_RUNTIME_RESET_FAILED;
    }
    for (size_t index = 0; index < vector->environment_count; index++) {
        if (!rek_g1_binding_facts_valid(vector->facts[index])) {
            invalidate(vector, io);
            return REK_G1_NATIVE_PUFFER_RUNTIME_FACTS_INVALID;
        }
    }
    if (!runtime_outputs_valid(vector, io)) {
        invalidate(vector, io);
        return REK_G1_NATIVE_PUFFER_RUNTIME_OUTPUT_INVALID;
    }
    RekG1NativePufferStatus status = write_masks(vector, io);
    if (status != REK_G1_NATIVE_PUFFER_OK) {
        invalidate(vector, io);
        return status;
    }
    vector->failed = 0;
    vector->ready = 1;
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return REK_G1_NATIVE_PUFFER_OK;
}

RekG1NativePufferStatus rek_g1_native_puffer_step(
        RekG1NativePufferVector* vector,
        RekG1NativePufferIO io,
        char* error,
        size_t error_capacity) {
    if (vector == NULL || !vector->initialized || !vector->ready
            || vector->failed) {
        return REK_G1_NATIVE_PUFFER_NOT_READY;
    }
    if (!io_valid(vector, io, 1)) {
        invalidate(vector, io);
        return REK_G1_NATIVE_PUFFER_SIZE_INVALID;
    }
    memset(io.rewards, 0, vector->environment_count * sizeof(float));
    memset(io.terminals, 0, vector->environment_count * sizeof(float));
    for (size_t index = 0; index < vector->environment_count; index++) {
        if (!rek_g1_binding_facts_valid(vector->facts[index])) {
            invalidate(vector, io);
            return REK_G1_NATIVE_PUFFER_RUNTIME_FACTS_INVALID;
        }
        RekG1PufferStep step = rek_g1_puffer_step(
            &vector->adapters[index],
            io.actions[index * io.action_heads],
            vector->facts[index].timing,
            vector->facts[index].translation_transition_settled,
            rek_g1_binding_action_busy(vector->facts[index]));
        if (step.status != REK_G1_PUFFER_OK) {
            invalidate(vector, io);
            return REK_G1_NATIVE_PUFFER_ACTION_REJECTED;
        }
        vector->semantics[index] = step.semantic;
    }

    memset(
        vector->next_facts,
        0,
        vector->environment_count * sizeof(RekG1RuntimeFacts));
    if (!vector->runtime_ops.advance(
            vector->runtime_context,
            vector->motion_routes,
            vector->action_table,
            vector->semantics,
            vector->environment_count,
            vector->next_facts,
            io.observations,
            io.observation_stride_bytes,
            io.rewards,
            io.terminals,
            error,
            error_capacity)) {
        invalidate(vector, io);
        return REK_G1_NATIVE_PUFFER_RUNTIME_ADVANCE_FAILED;
    }
    for (size_t index = 0; index < vector->environment_count; index++) {
        if (!rek_g1_binding_facts_valid(vector->next_facts[index])) {
            invalidate(vector, io);
            return REK_G1_NATIVE_PUFFER_RUNTIME_FACTS_INVALID;
        }
    }
    if (!runtime_outputs_valid(vector, io)) {
        invalidate(vector, io);
        return REK_G1_NATIVE_PUFFER_RUNTIME_OUTPUT_INVALID;
    }
    for (size_t index = 0; index < vector->environment_count; index++) {
        if (io.terminals[index] != 0.0f
                || vector->next_facts[index].input_reset) {
            rek_g1_puffer_reset(&vector->adapters[index]);
        }
    }
    memcpy(
        vector->facts,
        vector->next_facts,
        vector->environment_count * sizeof(RekG1RuntimeFacts));
    RekG1NativePufferStatus status = write_masks(vector, io);
    if (status != REK_G1_NATIVE_PUFFER_OK) {
        invalidate(vector, io);
        return status;
    }
    if (error != NULL && error_capacity > 0) error[0] = '\0';
    return REK_G1_NATIVE_PUFFER_OK;
}

void rek_g1_native_puffer_close(RekG1NativePufferVector* vector) {
    if (vector == NULL) return;
    if (vector->initialized && vector->runtime_ops.close != NULL
            && vector->runtime_context != NULL) {
        vector->runtime_ops.close(vector->runtime_context);
    }
    free(vector->semantics);
    free(vector->next_facts);
    free(vector->facts);
    free(vector->adapters);
    memset(vector, 0, sizeof(*vector));
}
