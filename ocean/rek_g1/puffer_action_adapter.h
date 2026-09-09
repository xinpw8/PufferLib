#pragma once

#include <stddef.h>
#include <stdint.h>

#include "semantic_action.h"

// Puffer's categorical action buffer is float-valued even for discrete heads.
// This adapter validates and dispatches one generated semantic category. It
// does not run the Sonic model or physics.

// The origin/5.0 rollout path stores categorical IDs in BF16 by default before
// casting them to the environment's float action buffer. BF16 represents every
// integer only through 256, so indices 0..256 and at most 257 categories are
// transport-safe. A verified float32 action transport may override this at
// compile time, up to float32's contiguous integer range.
#define REK_G1_PUFFER_FLOAT32_EXACT_CATEGORY_LIMIT 16777217u
#ifndef REK_G1_PUFFER_EXACT_CATEGORY_LIMIT
#define REK_G1_PUFFER_EXACT_CATEGORY_LIMIT 257u
#endif
#if REK_G1_PUFFER_EXACT_CATEGORY_LIMIT > REK_G1_PUFFER_FLOAT32_EXACT_CATEGORY_LIMIT
#error "Puffer category count cannot exceed float32's exact contiguous ID range"
#endif

typedef enum RekG1PufferCategoryKind {
    REK_G1_PUFFER_CONTINUE = 0,
    REK_G1_PUFFER_START = 1,
} RekG1PufferCategoryKind;

typedef struct RekG1PufferCategory {
    RekG1PufferCategoryKind kind;
    // Adapter-table kick templates identify route and duration and must use a
    // neutral held_code. This normalization does not restrict direct semantic
    // commands, which may carry yaw. At adapter dispatch, held_code is replaced
    // with the current desired Q/E state. During the kick, validated yaw-only
    // locomotion categories may update it. Retaining and advancing the yaw ramp
    // is provisional candidate behavior, not recovered REK parity.
    RekG1SemanticCommand command;
} RekG1PufferCategory;

typedef struct RekG1PufferActionTable {
    const RekG1PufferCategory* categories;
    uint32_t count;
    // Registry identity and duration are generated only after runtime capture.
    // Every kick category must match the duration at its registry index.
    const uint16_t* kick_move_indices;
    const uint32_t* kick_duration_ticks;
    uint16_t kick_registry_count;
} RekG1PufferActionTable;

enum {
    REK_G1_REQUIRED_KICK_COUNT = 4,
};

typedef enum RekG1PufferStatus {
    REK_G1_PUFFER_OK = 0,
    REK_G1_PUFFER_TABLE_NULL = 1,
    REK_G1_PUFFER_TABLE_COUNT_INVALID = 2,
    REK_G1_PUFFER_TABLE_CONTINUE_INVALID = 3,
    REK_G1_PUFFER_TABLE_START_INVALID = 4,
    REK_G1_PUFFER_TABLE_DUPLICATE = 5,
    REK_G1_PUFFER_ACTION_NOT_FINITE = 6,
    REK_G1_PUFFER_ACTION_NOT_INTEGRAL = 7,
    REK_G1_PUFFER_ACTION_OUT_OF_RANGE = 8,
    REK_G1_PUFFER_ACTION_MASKED = 9,
    REK_G1_PUFFER_PROTOCOL_ERROR = 10,
    REK_G1_PUFFER_TABLE_HELD_COVERAGE_INVALID = 11,
    REK_G1_PUFFER_TABLE_KICK_REGISTRY_INVALID = 12,
    REK_G1_PUFFER_TABLE_KICK_DURATION_INVALID = 13,
    REK_G1_PUFFER_INPUT_TIMING_INVALID = 14,
} RekG1PufferStatus;

typedef struct RekG1PufferAdapter {
    RekG1SemanticScheduler scheduler;
    const RekG1PufferActionTable* table;
} RekG1PufferAdapter;

typedef struct RekG1PufferStep {
    RekG1PufferStatus status;
    uint32_t category;
    RekG1SemanticTick semantic;
} RekG1PufferStep;

RekG1PufferStatus rek_g1_puffer_validate_table(
    const RekG1PufferActionTable* table);

RekG1PufferStatus rek_g1_puffer_init(
    RekG1PufferAdapter* adapter,
    const RekG1PufferActionTable* table);

void rek_g1_puffer_reset(RekG1PufferAdapter* adapter);

int rek_g1_puffer_category_legal(
    const RekG1PufferAdapter* adapter,
    uint32_t category,
    int translation_transition_settled,
    int action_busy);

RekG1PufferStatus rek_g1_puffer_write_mask(
    const RekG1PufferAdapter* adapter,
    int translation_transition_settled,
    int action_busy,
    uint8_t* mask,
    size_t mask_bytes);

RekG1PufferStep rek_g1_puffer_step(
    RekG1PufferAdapter* adapter,
    float action,
    RekG1InputTiming timing,
    int translation_transition_settled,
    int action_busy);
