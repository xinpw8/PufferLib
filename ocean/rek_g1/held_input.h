#pragma once

#include "g1_cuda_qualifiers.h"

#include <math.h>
#include <stdint.h>

// Unitree G1 state-based input boundary. This contract only represents input
// state and gating that has direct static or observed support. It does not
// synthesize motor targets and is not a parity claim.

enum {
    REK_G1_HELD_FORWARD = 1u << 0,      // W
    REK_G1_HELD_BACKWARD = 1u << 1,     // S
    REK_G1_HELD_STRAFE_LEFT = 1u << 2,  // A
    REK_G1_HELD_STRAFE_RIGHT = 1u << 3, // D
    REK_G1_HELD_YAW_LEFT = 1u << 4,     // Q
    REK_G1_HELD_YAW_RIGHT = 1u << 5,    // E
    REK_G1_HELD_VALID_MASK = (1u << 6) - 1u,
    REK_G1_HELD_TRANSLATION_MASK =
        REK_G1_HELD_FORWARD |
        REK_G1_HELD_BACKWARD |
        REK_G1_HELD_STRAFE_LEFT |
        REK_G1_HELD_STRAFE_RIGHT,
    REK_G1_HELD_YAW_MASK = REK_G1_HELD_YAW_LEFT | REK_G1_HELD_YAW_RIGHT,
};

typedef enum RekG1InputStatus {
    REK_G1_INPUT_ACCEPTED = 0,
    REK_G1_INPUT_REJECTED_UNKNOWN_KEY = 1,
    REK_G1_INPUT_REJECTED_OPPOSITE_TRANSLATION = 2,
    REK_G1_INPUT_REJECTED_OPPOSITE_YAW = 3,
    REK_G1_INPUT_REJECTED_INVALID_ELAPSED_SECONDS = 4,
    REK_G1_INPUT_REJECTED_INVALID_YAW_RAMP_SECONDS = 5,
} RekG1InputStatus;

typedef enum RekG1AttackGate {
    REK_G1_ATTACK_NOT_REQUESTED = 0,
    REK_G1_ATTACK_BLOCKED_TRANSLATION_HELD = 1,
    REK_G1_ATTACK_BLOCKED_TRANSLATION_SETTLING = 2,
    REK_G1_ATTACK_BLOCKED_ACTION_BUSY = 3,
    REK_G1_ATTACK_ACCEPTED_PREEMPT_YAW = 4,
} RekG1AttackGate;

typedef struct RekG1HeldInputState {
    uint8_t held;
    // Ramp magnitude and sign are retained independently of effective output.
    // This lets an accepted kick suppress yaw without discarding the desired
    // held state or the ramp progress accumulated during the action.
    float yaw_ramp;
    int8_t yaw_sign;
} RekG1HeldInputState;

typedef struct RekG1InputTiming {
    // Both values are required on every controller tick. The adapter has no
    // assumed tick rate and no fallback keyboard ramp duration.
    float elapsed_seconds;
    float yaw_ramp_seconds;
} RekG1InputTiming;

typedef struct RekG1InputFrame {
    // Full desired held state for this control tick. Keeping a bit set across
    // ticks is a hold. Clearing it is a release.
    uint8_t held;
    // A one-tick attack request edge. What REK does with an edge rejected by
    // a locomotion gate is not yet measured, so this contract never queues it.
    uint8_t attack_edge;
} RekG1InputFrame;

typedef struct RekG1InputDecision {
    RekG1InputStatus status;
    RekG1AttackGate attack_gate;
    uint8_t held;
    uint8_t pressed_edges;
    uint8_t released_edges;
    int8_t forward;
    int8_t strafe;
    int8_t desired_yaw;
    float yaw;
    float yaw_ramp;
    uint8_t yaw_suppressed_for_attack;
    uint8_t blocked_attack_retention_unknown;
} RekG1InputDecision;

static REK_G1_FN inline int8_t rek_g1_desired_yaw(uint8_t held) {
    return (held & REK_G1_HELD_YAW_LEFT) ? 1 :
        (held & REK_G1_HELD_YAW_RIGHT) ? -1 : 0;
}

static REK_G1_FN inline RekG1InputStatus rek_g1_validate_input_timing(
        RekG1InputTiming timing) {
    if (!isfinite(timing.elapsed_seconds) || timing.elapsed_seconds <= 0.0f) {
        return REK_G1_INPUT_REJECTED_INVALID_ELAPSED_SECONDS;
    }
    if (!isfinite(timing.yaw_ramp_seconds) ||
            timing.yaw_ramp_seconds <= 0.0f) {
        return REK_G1_INPUT_REJECTED_INVALID_YAW_RAMP_SECONDS;
    }
    return REK_G1_INPUT_ACCEPTED;
}

static REK_G1_FN inline void rek_g1_fill_effective_axes(
        RekG1InputDecision* result,
        const RekG1HeldInputState* state) {
    result->held = state->held;
    result->forward = (state->held & REK_G1_HELD_FORWARD) ? 1 :
        (state->held & REK_G1_HELD_BACKWARD) ? -1 : 0;
    result->strafe = (state->held & REK_G1_HELD_STRAFE_LEFT) ? 1 :
        (state->held & REK_G1_HELD_STRAFE_RIGHT) ? -1 : 0;
    result->desired_yaw = rek_g1_desired_yaw(state->held);
    result->yaw_ramp = state->yaw_ramp;
    result->yaw = state->yaw_ramp * (float)state->yaw_sign;
}

static REK_G1_FN inline void rek_g1_advance_yaw_ramp(
        RekG1HeldInputState* state,
        RekG1InputTiming timing) {
    int8_t desired_yaw = rek_g1_desired_yaw(state->held);
    if (desired_yaw == 0) {
        state->yaw_sign = 0;
        state->yaw_ramp = 0.0f;
        return;
    }

    if (desired_yaw != state->yaw_sign) {
        // Pinned RampKeyboardYaw semantics reset before accumulating the new
        // sign during the same update.
        state->yaw_sign = desired_yaw;
        state->yaw_ramp = 0.0f;
    }
    float next_ramp = state->yaw_ramp +
        timing.elapsed_seconds / timing.yaw_ramp_seconds;
    state->yaw_ramp = next_ramp >= 1.0f ? 1.0f : next_ramp;
}

static REK_G1_FN inline int rek_g1_has_opposite_translation(uint8_t held) {
    return ((held & REK_G1_HELD_FORWARD) &&
            (held & REK_G1_HELD_BACKWARD)) ||
        ((held & REK_G1_HELD_STRAFE_LEFT) &&
         (held & REK_G1_HELD_STRAFE_RIGHT));
}

static REK_G1_FN inline int rek_g1_has_opposite_yaw(uint8_t held) {
    return (held & REK_G1_HELD_YAW_LEFT) &&
        (held & REK_G1_HELD_YAW_RIGHT);
}

static REK_G1_FN inline RekG1InputDecision rek_g1_apply_input_frame(
        RekG1HeldInputState* state,
        RekG1InputFrame frame,
        RekG1InputTiming timing,
        int translation_transition_settled,
        int action_busy) {
    RekG1InputDecision result = REK_G1_ZERO_INIT;
    result.status = REK_G1_INPUT_ACCEPTED;
    result.attack_gate = REK_G1_ATTACK_NOT_REQUESTED;

    if ((frame.held & ~REK_G1_HELD_VALID_MASK) != 0) {
        result.status = REK_G1_INPUT_REJECTED_UNKNOWN_KEY;
        rek_g1_fill_effective_axes(&result, state);
        return result;
    }
    if (rek_g1_has_opposite_translation(frame.held)) {
        result.status = REK_G1_INPUT_REJECTED_OPPOSITE_TRANSLATION;
        rek_g1_fill_effective_axes(&result, state);
        return result;
    }
    if (rek_g1_has_opposite_yaw(frame.held)) {
        result.status = REK_G1_INPUT_REJECTED_OPPOSITE_YAW;
        rek_g1_fill_effective_axes(&result, state);
        return result;
    }
    result.status = rek_g1_validate_input_timing(timing);
    if (result.status != REK_G1_INPUT_ACCEPTED) {
        rek_g1_fill_effective_axes(&result, state);
        return result;
    }

    uint8_t previous = state->held;
    state->held = frame.held;
    result.pressed_edges = frame.held & (uint8_t)~previous;
    result.released_edges = previous & (uint8_t)~frame.held;
    rek_g1_advance_yaw_ramp(state, timing);
    rek_g1_fill_effective_axes(&result, state);
    if (!frame.attack_edge) return result;

    if ((frame.held & REK_G1_HELD_TRANSLATION_MASK) != 0) {
        result.attack_gate = REK_G1_ATTACK_BLOCKED_TRANSLATION_HELD;
        result.blocked_attack_retention_unknown = 1;
        return result;
    }
    if (!translation_transition_settled) {
        result.attack_gate = REK_G1_ATTACK_BLOCKED_TRANSLATION_SETTLING;
        result.blocked_attack_retention_unknown = 1;
        return result;
    }
    if (action_busy) {
        result.attack_gate = REK_G1_ATTACK_BLOCKED_ACTION_BUSY;
        result.blocked_attack_retention_unknown = 1;
        return result;
    }

    result.attack_gate = REK_G1_ATTACK_ACCEPTED_PREEMPT_YAW;
    result.yaw_suppressed_for_attack = result.desired_yaw != 0;
    result.yaw = 0.0f;
    return result;
}
