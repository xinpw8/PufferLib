#include <stdio.h>
#include <stdlib.h>

#include "semantic_action.h"

static int assertions;

static void require(int condition, const char* name) {
    assertions += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

static uint8_t held_code(uint8_t held) {
    uint8_t code = 255;
    require(
        rek_g1_semantic_encode_held(held, &code) == REK_G1_SEMANTIC_OK,
        "test_mask_encodes");
    return code;
}

int main(void) {
    const RekG1InputTiming timing = {
        .elapsed_seconds = 0.02f,
        .yaw_ramp_seconds = 0.02f,
    };
    int valid_mask_count = 0;
    uint8_t seen[REK_G1_SEMANTIC_HELD_CODES] = {0};
    for (int mask = 0; mask <= REK_G1_HELD_VALID_MASK; mask++) {
        uint8_t code = 255;
        RekG1SemanticStatus status = rek_g1_semantic_encode_held(
            (uint8_t)mask,
            &code);
        int valid = !rek_g1_has_opposite_translation((uint8_t)mask) &&
            !rek_g1_has_opposite_yaw((uint8_t)mask);
        require((status == REK_G1_SEMANTIC_OK) == valid,
            "held_mask_validity_exact");
        if (!valid) continue;
        valid_mask_count += 1;
        require(code < REK_G1_SEMANTIC_HELD_CODES, "held_code_in_range");
        require(!seen[code], "held_code_bijective");
        seen[code] = 1;
        uint8_t decoded = 255;
        require(
            rek_g1_semantic_decode_held(code, &decoded) ==
                REK_G1_SEMANTIC_OK,
            "held_code_decodes");
        require(decoded == mask, "held_code_round_trip");
    }
    require(valid_mask_count == REK_G1_SEMANTIC_HELD_CODES,
        "exactly_27_syntactically_valid_masks");

    uint8_t ignored = 0;
    require(
        rek_g1_semantic_encode_held(1u << 6, &ignored) ==
            REK_G1_SEMANTIC_INVALID_HELD_CODE,
        "unknown_key_rejected");
    require(
        rek_g1_semantic_decode_held(REK_G1_SEMANTIC_HELD_CODES, &ignored) ==
            REK_G1_SEMANTIC_INVALID_HELD_CODE,
        "out_of_range_code_rejected");

    RekG1SemanticScheduler scheduler;
    rek_g1_semantic_reset(&scheduler);
    RekG1SemanticCommand command = {
        .kind = REK_G1_SEMANTIC_LOCOMOTION,
        .held_code = held_code(
            REK_G1_HELD_FORWARD |
            REK_G1_HELD_STRAFE_LEFT |
            REK_G1_HELD_YAW_LEFT),
        .duration_ticks = 3,
        .move_registry_index = REK_G1_SEMANTIC_MOVE_NONE,
    };
    require(rek_g1_semantic_start(&scheduler, command, 17) ==
        REK_G1_SEMANTIC_OK, "locomotion_segment_starts");
    require(rek_g1_semantic_start(&scheduler, command, 17) ==
        REK_G1_SEMANTIC_SEGMENT_ALREADY_ACTIVE,
        "active_segment_cannot_be_replaced");
    for (int tick = 0; tick < 3; tick++) {
        RekG1SemanticTick result = rek_g1_semantic_tick(
            &scheduler, timing, 1, 0);
        require(result.status == REK_G1_SEMANTIC_OK,
            "locomotion_tick_accepted");
        require(result.input.held ==
            (REK_G1_HELD_FORWARD |
             REK_G1_HELD_STRAFE_LEFT |
             REK_G1_HELD_YAW_LEFT),
            "full_held_mask_repeated");
        require(result.input.forward == 1 && result.input.strafe == 1 &&
            result.input.yaw == 1, "translation_and_yaw_overlap");
        require(result.command_started == (tick == 0),
            "command_start_edge_once");
        require(result.remaining_ticks == (uint32_t)(2 - tick),
            "duration_counts_control_ticks");
        require(result.segment_complete == (tick == 2),
            "locomotion_completes_exactly");
    }
    require(rek_g1_semantic_tick(&scheduler, timing, 1, 0).status ==
        REK_G1_SEMANTIC_NO_ACTIVE_SEGMENT, "continue_requires_segment");

    command = (RekG1SemanticCommand){
        .kind = REK_G1_SEMANTIC_DISCRETE_MOVE,
        .held_code = held_code(REK_G1_HELD_YAW_RIGHT),
        .duration_ticks = 4,
        .move_registry_index = 2,
    };
    require(rek_g1_semantic_start(&scheduler, command, 17) ==
        REK_G1_SEMANTIC_OK, "move_segment_starts");
    for (int tick = 0; tick < 4; tick++) {
        RekG1SemanticTick result = rek_g1_semantic_tick(
            &scheduler, timing, 1, tick > 0);
        require(result.status == REK_G1_SEMANTIC_OK, "move_tick_accepted");
        require(result.move_start_edge == (tick == 0),
            "move_edge_dispatched_once");
        require(result.move_active, "move_active_for_full_duration");
        require(result.input.held == REK_G1_HELD_YAW_RIGHT,
            "desired_yaw_hold_retained");
        require(result.input.yaw == 0 &&
            result.input.yaw_suppressed_for_attack,
            "yaw_preempted_for_full_move_duration");
        require(result.segment_complete == (tick == 3),
            "move_completes_exactly");
    }

    command = (RekG1SemanticCommand){
        .kind = REK_G1_SEMANTIC_LOCOMOTION,
        .held_code = held_code(REK_G1_HELD_FORWARD),
        .duration_ticks = 1,
        .move_registry_index = REK_G1_SEMANTIC_MOVE_NONE,
    };
    require(rek_g1_semantic_start(&scheduler, command, 17) ==
        REK_G1_SEMANTIC_OK, "forward_segment_starts");
    require(rek_g1_semantic_tick(&scheduler, timing, 1, 0).segment_complete,
        "forward_segment_completes");

    command = (RekG1SemanticCommand){
        .kind = REK_G1_SEMANTIC_DISCRETE_MOVE,
        .held_code = held_code(0),
        .duration_ticks = 5,
        .move_registry_index = 0,
    };
    require(rek_g1_semantic_start(&scheduler, command, 17) ==
        REK_G1_SEMANTIC_OK, "post_translation_move_proposed");
    RekG1SemanticTick blocked = rek_g1_semantic_tick(
        &scheduler, timing, 0, 0);
    require(blocked.input.attack_gate ==
        REK_G1_ATTACK_BLOCKED_TRANSLATION_SETTLING,
        "translation_settle_gate_applied");
    require(blocked.input.released_edges == REK_G1_HELD_FORWARD,
        "translation_release_visible_on_move_tick");
    require(blocked.move_blocked && blocked.segment_complete,
        "blocked_move_segment_terminates");
    require(rek_g1_semantic_tick(&scheduler, timing, 1, 0).status ==
        REK_G1_SEMANTIC_NO_ACTIVE_SEGMENT,
        "blocked_move_is_not_queued_or_retried");

    require(rek_g1_semantic_start(&scheduler, command, 17) ==
        REK_G1_SEMANTIC_OK, "busy_gate_move_proposed");
    blocked = rek_g1_semantic_tick(&scheduler, timing, 1, 1);
    require(blocked.input.attack_gate == REK_G1_ATTACK_BLOCKED_ACTION_BUSY,
        "action_busy_gate_applied");
    require(blocked.move_blocked && blocked.segment_complete,
        "busy_blocked_move_not_queued");

    command.duration_ticks = 0;
    require(rek_g1_semantic_start(&scheduler, command, 17) ==
        REK_G1_SEMANTIC_ZERO_DURATION, "zero_duration_rejected");
    command.duration_ticks = 1;
    command.move_registry_index = 17;
    require(rek_g1_semantic_start(&scheduler, command, 17) ==
        REK_G1_SEMANTIC_MOVE_INDEX_OUT_OF_RANGE,
        "unvalidated_move_index_rejected");
    command.move_registry_index = 0;
    command.held_code = held_code(REK_G1_HELD_STRAFE_RIGHT);
    require(rek_g1_semantic_start(&scheduler, command, 17) ==
        REK_G1_SEMANTIC_MOVE_HAS_TRANSLATION,
        "move_category_cannot_encode_translation");

    command = (RekG1SemanticCommand){
        .kind = REK_G1_SEMANTIC_LOCOMOTION,
        .held_code = held_code(0),
        .duration_ticks = 1,
        .move_registry_index = 0,
    };
    require(rek_g1_semantic_start(&scheduler, command, 17) ==
        REK_G1_SEMANTIC_LOCOMOTION_HAS_MOVE,
        "locomotion_category_cannot_encode_move");
    require(REK_G1_SEMANTIC_KICK == REK_G1_SEMANTIC_DISCRETE_MOVE,
        "legacy_kind_alias_preserved");
    require(REK_G1_SEMANTIC_KICK_NONE == REK_G1_SEMANTIC_MOVE_NONE,
        "legacy_none_alias_preserved");
    require(REK_G1_SEMANTIC_LOCOMOTION_HAS_KICK
            == REK_G1_SEMANTIC_LOCOMOTION_HAS_MOVE
            && REK_G1_SEMANTIC_KICK_INDEX_OUT_OF_RANGE
                == REK_G1_SEMANTIC_MOVE_INDEX_OUT_OF_RANGE
            && REK_G1_SEMANTIC_KICK_HAS_TRANSLATION
                == REK_G1_SEMANTIC_MOVE_HAS_TRANSLATION,
        "legacy_status_aliases_preserved");

    printf(
        "PASS g1_semantic_action_protocol assertions=%d held_codes=%d\n",
        assertions,
        REK_G1_SEMANTIC_HELD_CODES);
    return 0;
}
