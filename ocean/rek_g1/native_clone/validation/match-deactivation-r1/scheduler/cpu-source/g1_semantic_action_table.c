#include "g1_semantic_action_table.h"

#include <string.h>

static REK_G1_CONSTANT const uint8_t HELD_MASKS[15] = {
    0,
    REK_G1_HELD_FORWARD,
    REK_G1_HELD_BACKWARD,
    REK_G1_HELD_STRAFE_LEFT,
    REK_G1_HELD_STRAFE_RIGHT,
    REK_G1_HELD_YAW_LEFT,
    REK_G1_HELD_YAW_RIGHT,
    REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_LEFT,
    REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_RIGHT,
    REK_G1_HELD_BACKWARD | REK_G1_HELD_YAW_LEFT,
    REK_G1_HELD_BACKWARD | REK_G1_HELD_YAW_RIGHT,
    REK_G1_HELD_STRAFE_LEFT | REK_G1_HELD_YAW_LEFT,
    REK_G1_HELD_STRAFE_LEFT | REK_G1_HELD_YAW_RIGHT,
    REK_G1_HELD_STRAFE_RIGHT | REK_G1_HELD_YAW_LEFT,
    REK_G1_HELD_STRAFE_RIGHT | REK_G1_HELD_YAW_RIGHT,
};

static REK_G1_CONSTANT const uint16_t MOVE_REGISTRY_ORDER[
        REK_G1_REQUIRED_DISCRETE_MOVE_COUNT] = {
    6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16,
};

REK_G1_FN RekG1PufferStatus rek_g1_semantic_action_table_init(
        RekG1SemanticActionTableStorage* storage,
        uint32_t locomotion_segment_ticks,
        const uint32_t configured_compositor_move_duration_ticks[
            REK_G1_REQUIRED_DISCRETE_MOVE_COUNT]) {
    if (storage == NULL
            || configured_compositor_move_duration_ticks == NULL) {
        return REK_G1_PUFFER_TABLE_NULL;
    }
    if (locomotion_segment_ticks == 0u) {
        return REK_G1_PUFFER_TABLE_START_INVALID;
    }
    memset(storage, 0, sizeof(*storage));
    storage->categories[0].kind = REK_G1_PUFFER_CONTINUE;
    for (size_t index = 0; index < 15u; index++) {
        uint8_t held_code = 0u;
        if (rek_g1_semantic_encode_held(HELD_MASKS[index], &held_code)
                != REK_G1_SEMANTIC_OK) {
            return REK_G1_PUFFER_TABLE_START_INVALID;
        }
        storage->categories[index + 1u] = (RekG1PufferCategory){
            .kind = REK_G1_PUFFER_START,
            .command = {
                .kind = REK_G1_SEMANTIC_LOCOMOTION,
                .held_code = held_code,
                .duration_ticks = locomotion_segment_ticks,
                .move_registry_index = REK_G1_SEMANTIC_MOVE_NONE,
            },
        };
    }
    for (uint16_t index = 0;
            index < REK_G1_REQUIRED_DISCRETE_MOVE_COUNT; index++) {
        const uint16_t move_index = MOVE_REGISTRY_ORDER[index];
        const uint32_t duration =
            configured_compositor_move_duration_ticks[move_index];
        if (duration == 0u) {
            return REK_G1_PUFFER_TABLE_MOVE_DURATION_INVALID;
        }
        uint8_t neutral_code = 0u;
        if (rek_g1_semantic_encode_held(0u, &neutral_code)
                != REK_G1_SEMANTIC_OK) {
            return REK_G1_PUFFER_TABLE_START_INVALID;
        }
        storage->move_indices[index] = move_index;
        storage->move_duration_ticks[index] = duration;
        storage->categories[16u + index] = (RekG1PufferCategory){
            .kind = REK_G1_PUFFER_START,
            .command = {
                .kind = REK_G1_SEMANTIC_DISCRETE_MOVE,
                .held_code = neutral_code,
                .duration_ticks = duration,
                .move_registry_index = index,
            },
        };
    }
    storage->table = (RekG1PufferActionTable){
        .categories = storage->categories,
        .count = REK_G1_SEMANTIC_ACTION_COUNT,
        .move_indices = storage->move_indices,
        .move_duration_ticks = storage->move_duration_ticks,
        .move_registry_count = REK_G1_REQUIRED_DISCRETE_MOVE_COUNT,
    };
    return rek_g1_puffer_validate_table(&storage->table);
}
