#include "g1_semantic_action_table.h"

#include <stdio.h>
#include <stdlib.h>

static int assertions;

static void require(int condition, const char* name) {
    assertions += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

int main(void) {
    const uint32_t durations[REK_G1_REQUIRED_DISCRETE_MOVE_COUNT] = {
        35u, 27u, 31u, 45u, 32u, 45u, 157u, 145u, 158u,
        139u, 134u, 138u, 73u, 75u, 68u, 71u, 103u,
    };
    const uint16_t registry_order[REK_G1_REQUIRED_DISCRETE_MOVE_COUNT] = {
        6u, 7u, 8u, 9u, 0u, 1u, 2u, 3u, 4u,
        5u, 10u, 11u, 12u, 13u, 14u, 15u, 16u,
    };
    RekG1SemanticActionTableStorage storage;
    require(rek_g1_semantic_action_table_init(&storage, 5u, durations)
        == REK_G1_PUFFER_OK, "valid_table");
    require(storage.table.count == REK_G1_SEMANTIC_ACTION_COUNT,
        "category_count");
    require(storage.categories[0].kind == REK_G1_PUFFER_CONTINUE,
        "continue_category");
    for (uint32_t category = 1u; category <= 15u; category++) {
        require(storage.categories[category].command.kind
            == REK_G1_SEMANTIC_LOCOMOTION, "locomotion_kind");
        require(storage.categories[category].command.duration_ticks == 5u,
            "locomotion_duration");
    }
    uint8_t held = 0u;
    require(rek_g1_semantic_decode_held(
        storage.categories[2].command.held_code, &held) == REK_G1_SEMANTIC_OK
        && held == REK_G1_HELD_FORWARD, "forward_mapping");
    require(rek_g1_semantic_decode_held(
        storage.categories[15].command.held_code, &held) == REK_G1_SEMANTIC_OK
        && held == (REK_G1_HELD_STRAFE_RIGHT | REK_G1_HELD_YAW_RIGHT),
        "combined_mapping");
    for (uint16_t move = 0u;
            move < REK_G1_REQUIRED_DISCRETE_MOVE_COUNT; move++) {
        const RekG1PufferCategory* category = &storage.categories[16u + move];
        const uint16_t runtime_move_index = registry_order[move];
        require(category->command.kind == REK_G1_SEMANTIC_DISCRETE_MOVE,
            "discrete_move_kind");
        require(category->command.move_registry_index == move,
            "move_registry_index");
        require(category->command.duration_ticks ==
                durations[runtime_move_index],
            "move_duration");
        require(storage.move_indices[move] == runtime_move_index,
            "runtime_move_index");
    }
    require(storage.move_indices[0] == 6u && storage.move_indices[3] == 9u,
        "legacy_categories_16_through_19_preserved");
    require(rek_g1_semantic_action_table_init(&storage, 0u, durations)
        == REK_G1_PUFFER_TABLE_START_INVALID, "zero_locomotion_rejected");
    uint32_t missing[REK_G1_REQUIRED_DISCRETE_MOVE_COUNT];
    for (size_t index = 0u;
            index < REK_G1_REQUIRED_DISCRETE_MOVE_COUNT; index++) {
        missing[index] = durations[index];
    }
    missing[12] = 0u;
    require(rek_g1_semantic_action_table_init(&storage, 1u, missing)
        == REK_G1_PUFFER_TABLE_MOVE_DURATION_INVALID,
        "zero_move_rejected");
    printf("G1 semantic action table passed: assertions=%d\n", assertions);
    return 0;
}
