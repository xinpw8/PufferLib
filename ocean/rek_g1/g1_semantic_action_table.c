#include "g1_semantic_action_table.h"

#include <string.h>

static const uint8_t HELD_MASKS[15] = {
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

RekG1PufferStatus rek_g1_semantic_action_table_init(
        RekG1SemanticActionTableStorage* storage,
        uint32_t locomotion_segment_ticks,
        const uint32_t configured_compositor_kick_duration_ticks[
            REK_G1_REQUIRED_KICK_COUNT]) {
    if (storage == NULL
            || configured_compositor_kick_duration_ticks == NULL) {
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
                .kick_registry_index = REK_G1_SEMANTIC_KICK_NONE,
            },
        };
    }
    for (uint16_t index = 0; index < REK_G1_REQUIRED_KICK_COUNT; index++) {
        if (configured_compositor_kick_duration_ticks[index] == 0u) {
            return REK_G1_PUFFER_TABLE_KICK_DURATION_INVALID;
        }
        uint8_t neutral_code = 0u;
        if (rek_g1_semantic_encode_held(0u, &neutral_code)
                != REK_G1_SEMANTIC_OK) {
            return REK_G1_PUFFER_TABLE_START_INVALID;
        }
        storage->kick_move_indices[index] = (uint16_t)(6u + index);
        storage->kick_duration_ticks[index] =
            configured_compositor_kick_duration_ticks[index];
        storage->categories[16u + index] = (RekG1PufferCategory){
            .kind = REK_G1_PUFFER_START,
            .command = {
                .kind = REK_G1_SEMANTIC_KICK,
                .held_code = neutral_code,
                .duration_ticks =
                    configured_compositor_kick_duration_ticks[index],
                .kick_registry_index = index,
            },
        };
    }
    storage->table = (RekG1PufferActionTable){
        .categories = storage->categories,
        .count = REK_G1_SEMANTIC_ACTION_COUNT,
        .kick_move_indices = storage->kick_move_indices,
        .kick_duration_ticks = storage->kick_duration_ticks,
        .kick_registry_count = REK_G1_REQUIRED_KICK_COUNT,
    };
    return rek_g1_puffer_validate_table(&storage->table);
}
