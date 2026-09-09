#pragma once

#include <stdint.h>

#include "puffer_action_adapter.h"

enum {
    REK_G1_SEMANTIC_ACTION_COUNT = 33,
};

typedef struct RekG1SemanticActionTableStorage {
    RekG1PufferCategory categories[REK_G1_SEMANTIC_ACTION_COUNT];
    uint16_t move_indices[REK_G1_REQUIRED_DISCRETE_MOVE_COUNT];
    uint32_t move_duration_ticks[REK_G1_REQUIRED_DISCRETE_MOVE_COUNT];
    RekG1PufferActionTable table;
} RekG1SemanticActionTableStorage;

/*
 * Locomotion segment duration is an explicit agent-interface choice. Move
 * durations are configured compositor traversal lengths and must match the
 * loaded route assets. They are not measured physical completion times.
 * The configured duration array is indexed by runtime move index 0..16.
 */
RekG1PufferStatus rek_g1_semantic_action_table_init(
    RekG1SemanticActionTableStorage* storage,
    uint32_t locomotion_segment_ticks,
    const uint32_t configured_compositor_move_duration_ticks[
        REK_G1_REQUIRED_DISCRETE_MOVE_COUNT]);
