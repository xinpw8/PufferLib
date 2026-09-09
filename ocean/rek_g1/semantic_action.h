#pragma once

#include <stdint.h>

#include "held_input.h"

// Pure scheduling protocol for the eventual G1 SonicPolicyRunner adapter.
// It does not select clips, execute ONNX, or synthesize motor targets.

enum {
    REK_G1_SEMANTIC_HELD_CODES = 27,
};

#define REK_G1_SEMANTIC_KICK_NONE UINT16_MAX

typedef enum RekG1SemanticKind {
    REK_G1_SEMANTIC_LOCOMOTION = 0,
    REK_G1_SEMANTIC_KICK = 1,
} RekG1SemanticKind;

typedef enum RekG1SemanticStatus {
    REK_G1_SEMANTIC_OK = 0,
    REK_G1_SEMANTIC_NO_ACTIVE_SEGMENT = 1,
    REK_G1_SEMANTIC_SEGMENT_ALREADY_ACTIVE = 2,
    REK_G1_SEMANTIC_ZERO_DURATION = 3,
    REK_G1_SEMANTIC_INVALID_HELD_CODE = 4,
    REK_G1_SEMANTIC_INVALID_KIND = 5,
    REK_G1_SEMANTIC_LOCOMOTION_HAS_KICK = 6,
    REK_G1_SEMANTIC_KICK_INDEX_OUT_OF_RANGE = 7,
    REK_G1_SEMANTIC_KICK_HAS_TRANSLATION = 8,
    REK_G1_SEMANTIC_INVALID_INPUT_TIMING = 9,
} RekG1SemanticStatus;

typedef struct RekG1SemanticCommand {
    RekG1SemanticKind kind;
    // A bijective code for the syntactically valid held masks. The generated
    // runtime table decides which of these masks receive Puffer categories.
    uint8_t held_code;
    // Supplied by a validated generated action table. The protocol has no
    // fallback duration and does not derive one from clip metadata.
    uint32_t duration_ticks;
    // REK_G1_SEMANTIC_KICK_NONE for locomotion. For a kick, this indexes a
    // validated runtime kick registry whose count is supplied at load time.
    uint16_t kick_registry_index;
} RekG1SemanticCommand;

typedef struct RekG1SemanticScheduler {
    RekG1HeldInputState input_state;
    RekG1SemanticCommand command;
    uint32_t remaining_ticks;
    uint8_t active;
    uint8_t first_tick;
    uint8_t kick_accepted;
} RekG1SemanticScheduler;

typedef struct RekG1SemanticTick {
    RekG1SemanticStatus status;
    RekG1InputDecision input;
    RekG1SemanticKind kind;
    uint16_t kick_registry_index;
    uint32_t remaining_ticks;
    uint8_t command_started;
    uint8_t segment_complete;
    uint8_t kick_start_edge;
    uint8_t kick_active;
    uint8_t kick_blocked;
} RekG1SemanticTick;

static inline RekG1SemanticStatus rek_g1_semantic_encode_held(
        uint8_t held,
        uint8_t* code_out) {
    if (code_out == 0 || (held & ~REK_G1_HELD_VALID_MASK) != 0 ||
            rek_g1_has_opposite_translation(held) ||
            rek_g1_has_opposite_yaw(held)) {
        return REK_G1_SEMANTIC_INVALID_HELD_CODE;
    }

    uint8_t forward_axis = (held & REK_G1_HELD_FORWARD) ? 1u :
        (held & REK_G1_HELD_BACKWARD) ? 2u : 0u;
    uint8_t strafe_axis = (held & REK_G1_HELD_STRAFE_LEFT) ? 1u :
        (held & REK_G1_HELD_STRAFE_RIGHT) ? 2u : 0u;
    uint8_t yaw_axis = (held & REK_G1_HELD_YAW_LEFT) ? 1u :
        (held & REK_G1_HELD_YAW_RIGHT) ? 2u : 0u;
    *code_out = (uint8_t)(
        forward_axis + 3u * strafe_axis + 9u * yaw_axis);
    return REK_G1_SEMANTIC_OK;
}

static inline RekG1SemanticStatus rek_g1_semantic_decode_held(
        uint8_t code,
        uint8_t* held_out) {
    if (held_out == 0 || code >= REK_G1_SEMANTIC_HELD_CODES) {
        return REK_G1_SEMANTIC_INVALID_HELD_CODE;
    }

    uint8_t forward_axis = code % 3u;
    uint8_t strafe_axis = (code / 3u) % 3u;
    uint8_t yaw_axis = (code / 9u) % 3u;
    uint8_t held = 0;
    if (forward_axis == 1u) held |= REK_G1_HELD_FORWARD;
    if (forward_axis == 2u) held |= REK_G1_HELD_BACKWARD;
    if (strafe_axis == 1u) held |= REK_G1_HELD_STRAFE_LEFT;
    if (strafe_axis == 2u) held |= REK_G1_HELD_STRAFE_RIGHT;
    if (yaw_axis == 1u) held |= REK_G1_HELD_YAW_LEFT;
    if (yaw_axis == 2u) held |= REK_G1_HELD_YAW_RIGHT;
    *held_out = held;
    return REK_G1_SEMANTIC_OK;
}

static inline void rek_g1_semantic_reset(RekG1SemanticScheduler* scheduler) {
    *scheduler = (RekG1SemanticScheduler){0};
}

static inline RekG1SemanticStatus rek_g1_semantic_start(
        RekG1SemanticScheduler* scheduler,
        RekG1SemanticCommand command,
        uint16_t kick_registry_count) {
    if (scheduler->active) return REK_G1_SEMANTIC_SEGMENT_ALREADY_ACTIVE;
    if (command.kind != REK_G1_SEMANTIC_LOCOMOTION &&
            command.kind != REK_G1_SEMANTIC_KICK) {
        return REK_G1_SEMANTIC_INVALID_KIND;
    }
    if (command.duration_ticks == 0) return REK_G1_SEMANTIC_ZERO_DURATION;

    uint8_t held = 0;
    RekG1SemanticStatus status = rek_g1_semantic_decode_held(
        command.held_code,
        &held);
    if (status != REK_G1_SEMANTIC_OK) return status;

    if (command.kind == REK_G1_SEMANTIC_LOCOMOTION) {
        if (command.kick_registry_index != REK_G1_SEMANTIC_KICK_NONE) {
            return REK_G1_SEMANTIC_LOCOMOTION_HAS_KICK;
        }
    } else {
        if (command.kick_registry_index >= kick_registry_count) {
            return REK_G1_SEMANTIC_KICK_INDEX_OUT_OF_RANGE;
        }
        if ((held & REK_G1_HELD_TRANSLATION_MASK) != 0) {
            return REK_G1_SEMANTIC_KICK_HAS_TRANSLATION;
        }
    }

    scheduler->command = command;
    scheduler->remaining_ticks = command.duration_ticks;
    scheduler->active = 1;
    scheduler->first_tick = 1;
    scheduler->kick_accepted = 0;
    return REK_G1_SEMANTIC_OK;
}

static inline RekG1SemanticTick rek_g1_semantic_tick(
        RekG1SemanticScheduler* scheduler,
        RekG1InputTiming timing,
        int translation_transition_settled,
        int action_busy) {
    RekG1SemanticTick result = {0};
    result.status = REK_G1_SEMANTIC_NO_ACTIVE_SEGMENT;
    result.kick_registry_index = REK_G1_SEMANTIC_KICK_NONE;
    if (!scheduler->active) return result;
    if (rek_g1_validate_input_timing(timing) != REK_G1_INPUT_ACCEPTED) {
        result.status = REK_G1_SEMANTIC_INVALID_INPUT_TIMING;
        return result;
    }

    uint8_t held = 0;
    RekG1SemanticStatus decode_status = rek_g1_semantic_decode_held(
        scheduler->command.held_code,
        &held);
    if (decode_status != REK_G1_SEMANTIC_OK) {
        result.status = decode_status;
        scheduler->active = 0;
        scheduler->remaining_ticks = 0;
        result.segment_complete = 1;
        return result;
    }

    result.status = REK_G1_SEMANTIC_OK;
    result.kind = scheduler->command.kind;
    result.kick_registry_index = scheduler->command.kick_registry_index;
    result.command_started = scheduler->first_tick;
    uint8_t attack_edge = scheduler->first_tick &&
        scheduler->command.kind == REK_G1_SEMANTIC_KICK;
    result.input = rek_g1_apply_input_frame(
        &scheduler->input_state,
        (RekG1InputFrame){.held = held, .attack_edge = attack_edge},
        timing,
        translation_transition_settled,
        action_busy);

    if (attack_edge) {
        scheduler->first_tick = 0;
        if (result.input.attack_gate == REK_G1_ATTACK_ACCEPTED_PREEMPT_YAW) {
            scheduler->kick_accepted = 1;
            result.kick_start_edge = 1;
        } else {
            // A blocked edge is never retried or queued. It completes this
            // proposed kick segment immediately and leaves the caller to
            // choose a new command on the next control tick.
            result.kick_blocked = 1;
            result.segment_complete = 1;
            scheduler->active = 0;
            scheduler->remaining_ticks = 0;
            return result;
        }
    } else {
        scheduler->first_tick = 0;
    }

    if (scheduler->command.kind == REK_G1_SEMANTIC_KICK &&
            scheduler->kick_accepted) {
        result.kick_active = 1;
        result.input.yaw_suppressed_for_attack = result.input.yaw != 0 ||
            (held & REK_G1_HELD_YAW_MASK) != 0;
        result.input.yaw = 0.0f;
    }

    scheduler->remaining_ticks -= 1;
    result.remaining_ticks = scheduler->remaining_ticks;
    if (scheduler->remaining_ticks == 0) {
        result.segment_complete = 1;
        scheduler->active = 0;
    }
    return result;
}
