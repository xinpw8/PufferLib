#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "puffer_action_adapter.h"

// G1-only bridge between Puffer's one-float categorical action ABI and the
// semantic controller contract. The runtime callback receives no raw Puffer
// action value and has no actuator-buffer parameter. A production runtime owns
// all motor-target and physics behavior behind that callback.

enum {
    REK_G1_PUFFER_ACTION_HEADS = 1,
    // continue + 15 required held masks + four required kick registry entries
    REK_G1_PUFFER_MIN_CATEGORIES = 20,
};

typedef struct RekG1RuntimeFacts {
    RekG1InputTiming timing;
    uint8_t translation_transition_settled;
    uint8_t action_busy;
    uint8_t recovery_active;
    /* One-transition RobotInputController reset-completion event. */
    uint8_t input_reset;
} RekG1RuntimeFacts;

/* Explicit callback ABI gate. A zero-initialized legacy ops table must fail. */
enum {
    REK_G1_RUNTIME_FACTS_ABI_VERSION = 1,
};
#define REK_G1_RUNTIME_FACTS_SIZE ((size_t)sizeof(RekG1RuntimeFacts))

typedef int (*RekG1RuntimeResetFn)(
    void* context,
    RekG1RuntimeFacts* facts_out);

typedef int (*RekG1RuntimeAdvanceFn)(
    void* context,
    const RekG1SemanticTick* semantic,
    RekG1RuntimeFacts* next_facts_out);

typedef struct RekG1RuntimeOps {
    RekG1RuntimeResetFn reset;
    RekG1RuntimeAdvanceFn advance;
} RekG1RuntimeOps;

typedef enum RekG1BindingStatus {
    REK_G1_BINDING_OK = 0,
    REK_G1_BINDING_NULL_ARGUMENT = 1,
    REK_G1_BINDING_ACTION_TABLE_INVALID = 2,
    REK_G1_BINDING_ACTION_MASK_INVALID = 3,
    REK_G1_BINDING_RUNTIME_RESET_FAILED = 4,
    REK_G1_BINDING_RUNTIME_FACTS_INVALID = 5,
    REK_G1_BINDING_ACTION_REJECTED = 6,
    REK_G1_BINDING_RUNTIME_ADVANCE_FAILED = 7,
    REK_G1_BINDING_ACTION_MASK_WRITE_FAILED = 8,
    REK_G1_BINDING_NOT_READY = 9,
    REK_G1_BINDING_ACTION_ABI_INVALID = 10,
} RekG1BindingStatus;

typedef struct RekG1SemanticBinding {
    RekG1PufferAdapter adapter;
    RekG1RuntimeOps runtime_ops;
    // The generated table, runtime context, and mask storage must outlive this
    // binding. Mutable context and mask storage must be exclusive to one env.
    void* runtime_context;
    uint8_t* action_mask;
    size_t action_mask_bytes;
    RekG1RuntimeFacts facts;
    uint8_t initialized;
    uint8_t ready;
} RekG1SemanticBinding;

typedef struct RekG1BindingStep {
    RekG1BindingStatus status;
    // False when a binding-level check rejected the call before the Puffer
    // adapter ran. Only inspect puffer when this flag is true.
    uint8_t puffer_valid;
    RekG1PufferStep puffer;
} RekG1BindingStep;

static inline int rek_g1_binding_facts_valid(RekG1RuntimeFacts facts) {
    return rek_g1_validate_input_timing(facts.timing) ==
            REK_G1_INPUT_ACCEPTED &&
        facts.translation_transition_settled <= 1u &&
        facts.action_busy <= 1u &&
        facts.recovery_active <= 1u &&
        facts.input_reset <= 1u;
}

static inline int rek_g1_binding_action_busy(RekG1RuntimeFacts facts) {
    return facts.action_busy || facts.recovery_active;
}

// Failure is terminal for the current episode. The runtime callback may have
// partially advanced external state, so retrying the same or next category is
// unsafe. Invalidation removes every legal action until reset succeeds.
static inline void rek_g1_binding_invalidate(RekG1SemanticBinding* binding) {
    if (binding == 0) return;
    binding->ready = 0;
    binding->facts = (RekG1RuntimeFacts){0};
    if (binding->initialized && binding->adapter.table != 0) {
        rek_g1_puffer_reset(&binding->adapter);
    }
    if (binding->action_mask != 0 && binding->action_mask_bytes > 0) {
        memset(binding->action_mask, 0, binding->action_mask_bytes);
    }
}

static inline RekG1BindingStatus rek_g1_binding_write_mask(
        RekG1SemanticBinding* binding) {
    if (binding == 0 || binding->action_mask == 0 ||
            binding->adapter.table == 0 ||
            binding->action_mask_bytes != binding->adapter.table->count) {
        return REK_G1_BINDING_ACTION_MASK_INVALID;
    }
    RekG1PufferStatus status = rek_g1_puffer_write_mask(
        &binding->adapter,
        binding->facts.translation_transition_settled,
        rek_g1_binding_action_busy(binding->facts),
        binding->action_mask,
        binding->action_mask_bytes);
    return status == REK_G1_PUFFER_OK ?
        REK_G1_BINDING_OK : REK_G1_BINDING_ACTION_MASK_WRITE_FAILED;
}

static inline RekG1BindingStatus rek_g1_binding_init(
        RekG1SemanticBinding* binding,
        const RekG1PufferActionTable* table,
        const RekG1RuntimeOps* runtime_ops,
        void* runtime_context,
        uint8_t* action_mask,
        size_t action_mask_bytes) {
    if (binding == 0 || table == 0 || runtime_ops == 0 ||
            runtime_ops->reset == 0 || runtime_ops->advance == 0) {
        return REK_G1_BINDING_NULL_ARGUMENT;
    }
    if (action_mask == 0 || action_mask_bytes != table->count) {
        return REK_G1_BINDING_ACTION_MASK_INVALID;
    }

    RekG1PufferAdapter adapter;
    if (rek_g1_puffer_init(&adapter, table) != REK_G1_PUFFER_OK) {
        return REK_G1_BINDING_ACTION_TABLE_INVALID;
    }

    *binding = (RekG1SemanticBinding){0};
    binding->adapter = adapter;
    binding->runtime_ops = *runtime_ops;
    binding->runtime_context = runtime_context;
    binding->action_mask = action_mask;
    binding->action_mask_bytes = action_mask_bytes;
    binding->initialized = 1;
    memset(binding->action_mask, 0, binding->action_mask_bytes);
    return REK_G1_BINDING_OK;
}

static inline RekG1BindingStatus rek_g1_binding_reset(
        RekG1SemanticBinding* binding) {
    if (binding == 0) {
        return REK_G1_BINDING_NULL_ARGUMENT;
    }
    if (!binding->initialized) return REK_G1_BINDING_NOT_READY;
    rek_g1_binding_invalidate(binding);
    if (binding->runtime_ops.reset == 0) {
        return REK_G1_BINDING_NULL_ARGUMENT;
    }

    RekG1RuntimeFacts facts = {0};
    if (binding->runtime_ops.reset(
            binding->runtime_context, &facts) != 0) {
        return REK_G1_BINDING_RUNTIME_RESET_FAILED;
    }
    if (!rek_g1_binding_facts_valid(facts)) {
        return REK_G1_BINDING_RUNTIME_FACTS_INVALID;
    }

    rek_g1_puffer_reset(&binding->adapter);
    binding->facts = facts;
    RekG1BindingStatus status = rek_g1_binding_write_mask(binding);
    if (status != REK_G1_BINDING_OK) {
        rek_g1_binding_invalidate(binding);
        return status;
    }
    binding->ready = 1;
    return REK_G1_BINDING_OK;
}

static inline RekG1BindingStep rek_g1_binding_step(
        RekG1SemanticBinding* binding,
        float category) {
    RekG1BindingStep result = {0};
    if (binding == 0) {
        result.status = REK_G1_BINDING_NULL_ARGUMENT;
        return result;
    }
    if (!binding->initialized || !binding->ready) {
        result.status = REK_G1_BINDING_NOT_READY;
        return result;
    }
    if (binding->runtime_ops.advance == 0) {
        rek_g1_binding_invalidate(binding);
        result.status = REK_G1_BINDING_NULL_ARGUMENT;
        return result;
    }
    if (!rek_g1_binding_facts_valid(binding->facts)) {
        rek_g1_binding_invalidate(binding);
        result.status = REK_G1_BINDING_RUNTIME_FACTS_INVALID;
        return result;
    }

    result.puffer = rek_g1_puffer_step(
        &binding->adapter,
        category,
        binding->facts.timing,
        binding->facts.translation_transition_settled,
        rek_g1_binding_action_busy(binding->facts));
    result.puffer_valid = 1;
    if (result.puffer.status != REK_G1_PUFFER_OK) {
        result.status = REK_G1_BINDING_ACTION_REJECTED;
        return result;
    }

    RekG1RuntimeFacts next_facts = {0};
    if (binding->runtime_ops.advance(
            binding->runtime_context,
            &result.puffer.semantic,
            &next_facts) != 0) {
        rek_g1_binding_invalidate(binding);
        result.status = REK_G1_BINDING_RUNTIME_ADVANCE_FAILED;
        return result;
    }
    if (!rek_g1_binding_facts_valid(next_facts)) {
        rek_g1_binding_invalidate(binding);
        result.status = REK_G1_BINDING_RUNTIME_FACTS_INVALID;
        return result;
    }

    if (next_facts.input_reset) {
        rek_g1_puffer_reset(&binding->adapter);
    }
    binding->facts = next_facts;
    result.status = rek_g1_binding_write_mask(binding);
    if (result.status != REK_G1_BINDING_OK) {
        rek_g1_binding_invalidate(binding);
    }
    return result;
}

// Binding-facing form of the Puffer ABI. The caller must pass exactly the one
// categorical float head declared by REK_G1_PUFFER_ACTION_HEADS.
static inline RekG1BindingStep rek_g1_binding_step_actions(
        RekG1SemanticBinding* binding,
        const float* actions,
        size_t action_heads) {
    RekG1BindingStep result = {0};
    if (actions == 0) {
        result.status = REK_G1_BINDING_NULL_ARGUMENT;
        return result;
    }
    if (action_heads != REK_G1_PUFFER_ACTION_HEADS) {
        result.status = REK_G1_BINDING_ACTION_ABI_INVALID;
        return result;
    }
    return rek_g1_binding_step(binding, actions[0]);
}
