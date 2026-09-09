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
    const uint32_t durations[REK_G1_REQUIRED_KICK_COUNT] = {
        157u, 145u, 158u, 139u,
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
    for (uint16_t kick = 0u; kick < REK_G1_REQUIRED_KICK_COUNT; kick++) {
        const RekG1PufferCategory* category = &storage.categories[16u + kick];
        require(category->command.kind == REK_G1_SEMANTIC_KICK,
            "kick_kind");
        require(category->command.kick_registry_index == kick,
            "kick_registry_index");
        require(category->command.duration_ticks == durations[kick],
            "kick_duration");
        require(storage.kick_move_indices[kick] == (uint16_t)(6u + kick),
            "kick_move_index");
    }
    require(rek_g1_semantic_action_table_init(&storage, 0u, durations)
        == REK_G1_PUFFER_TABLE_START_INVALID, "zero_locomotion_rejected");
    uint32_t missing[REK_G1_REQUIRED_KICK_COUNT] = {
        durations[0], durations[1], 0u, durations[3],
    };
    require(rek_g1_semantic_action_table_init(&storage, 1u, missing)
        == REK_G1_PUFFER_TABLE_KICK_DURATION_INVALID,
        "zero_kick_rejected");
    printf("G1 semantic action table passed: assertions=%d\n", assertions);
    return 0;
}
