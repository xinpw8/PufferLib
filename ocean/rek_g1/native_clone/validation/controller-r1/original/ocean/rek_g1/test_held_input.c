#include <stdio.h>
#include <stdlib.h>

#include "held_input.h"

static int cases;

static void require(int condition, const char* name) {
    cases += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

int main(void) {
    RekG1HeldInputState state = {0};
    const RekG1InputTiming timing = {
        .elapsed_seconds = 0.02f,
        .yaw_ramp_seconds = 0.02f,
    };

    RekG1InputDecision decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){.held = REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_LEFT},
        timing,
        1,
        0);
    require(decision.status == REK_G1_INPUT_ACCEPTED, "forward_yaw_accepted");
    require(decision.forward == 1 && decision.strafe == 0 && decision.yaw == 1,
        "translation_yaw_overlap");
    require(decision.pressed_edges ==
        (REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_LEFT), "initial_edges");

    decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){.held = REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_LEFT},
        timing,
        1,
        0);
    require(decision.pressed_edges == 0 && decision.released_edges == 0,
        "held_state_persists_without_new_edge");

    for (int tick = 0; tick < 64; tick++) {
        decision = rek_g1_apply_input_frame(
            &state,
            (RekG1InputFrame){
                .held = REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_LEFT,
            },
            timing,
            1,
            0);
        require(decision.status == REK_G1_INPUT_ACCEPTED,
            "long_hold_frame_accepted");
        require(decision.forward == 1 && decision.yaw == 1,
            "long_hold_frame_remains_active");
        require(decision.pressed_edges == 0 && decision.released_edges == 0,
            "long_hold_frame_has_no_false_edges");
    }

    decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){
            .held = REK_G1_HELD_FORWARD | REK_G1_HELD_YAW_LEFT,
            .attack_edge = 1,
        },
        timing,
        1,
        0);
    require(decision.attack_gate == REK_G1_ATTACK_BLOCKED_TRANSLATION_HELD,
        "translation_hold_blocks_attack");
    require(decision.yaw == 1 && !decision.yaw_suppressed_for_attack,
        "blocked_attack_does_not_preempt_yaw");
    require(decision.blocked_attack_retention_unknown,
        "blocked_attack_queue_semantics_explicitly_unknown");

    decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){.held = REK_G1_HELD_YAW_LEFT, .attack_edge = 1},
        timing,
        0,
        0);
    require(decision.attack_gate == REK_G1_ATTACK_BLOCKED_TRANSLATION_SETTLING,
        "released_translation_must_settle");
    require(decision.released_edges == REK_G1_HELD_FORWARD,
        "translation_release_edge");

    decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){.held = REK_G1_HELD_YAW_LEFT, .attack_edge = 1},
        timing,
        1,
        0);
    require(decision.attack_gate == REK_G1_ATTACK_ACCEPTED_PREEMPT_YAW,
        "settled_attack_accepted");
    require(decision.yaw == 0 && decision.yaw_suppressed_for_attack,
        "accepted_attack_preempts_yaw");
    require(state.held == REK_G1_HELD_YAW_LEFT,
        "yaw_hold_state_not_invented_after_attack");

    decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){.held = REK_G1_HELD_YAW_LEFT, .attack_edge = 1},
        timing,
        1,
        1);
    require(decision.attack_gate == REK_G1_ATTACK_BLOCKED_ACTION_BUSY,
        "active_action_blocks_new_attack");

    uint8_t prior = state.held;
    decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){.held = REK_G1_HELD_FORWARD | REK_G1_HELD_BACKWARD},
        timing,
        1,
        0);
    require(decision.status == REK_G1_INPUT_REJECTED_OPPOSITE_TRANSLATION,
        "opposite_translation_unmeasured_rejected");
    require(state.held == prior, "rejected_frame_does_not_mutate_state");

    decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){.held = 1u << 7},
        timing,
        1,
        0);
    require(decision.status == REK_G1_INPUT_REJECTED_UNKNOWN_KEY,
        "unknown_key_rejected");
    require(decision.held == REK_G1_HELD_YAW_LEFT &&
            decision.forward == 0 && decision.strafe == 0 && decision.yaw == 1,
        "unknown_f_like_bit_preserves_effective_prior_axes");

    state.held = 0;
    const uint8_t translation_bits[] = {
        REK_G1_HELD_FORWARD,
        REK_G1_HELD_BACKWARD,
        REK_G1_HELD_STRAFE_LEFT,
        REK_G1_HELD_STRAFE_RIGHT,
    };
    const int8_t expected_forward[] = {1, -1, 0, 0};
    const int8_t expected_strafe[] = {0, 0, 1, -1};
    for (int item = 0; item < 4; item++) {
        state.held = 0;
        decision = rek_g1_apply_input_frame(
            &state,
            (RekG1InputFrame){
                .held = translation_bits[item] | REK_G1_HELD_YAW_RIGHT,
            },
            timing,
            1,
            0);
        require(decision.forward == expected_forward[item]
                && decision.strafe == expected_strafe[item]
                && decision.yaw == -1,
            "each_translation_axis_overlaps_yaw");

        for (int tick = 0; tick < 16; tick++) {
            decision = rek_g1_apply_input_frame(
                &state,
                (RekG1InputFrame){
                    .held = translation_bits[item] | REK_G1_HELD_YAW_RIGHT,
                },
                timing,
                1,
                0);
            require(decision.forward == expected_forward[item] &&
                    decision.strafe == expected_strafe[item] &&
                    decision.yaw == -1,
                "each_translation_axis_sustains_with_yaw");
            require(decision.pressed_edges == 0 && decision.released_edges == 0,
                "sustained_cardinal_and_yaw_has_no_false_edges");
        }

        decision = rek_g1_apply_input_frame(
            &state,
            (RekG1InputFrame){
                .held = translation_bits[item] | REK_G1_HELD_YAW_RIGHT,
                .attack_edge = 1,
            },
            timing,
            1,
            0);
        require(decision.attack_gate == REK_G1_ATTACK_BLOCKED_TRANSLATION_HELD,
            "each_translation_axis_blocks_attack");
    }

    state.held = REK_G1_HELD_YAW_LEFT;
    decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){
            .held = REK_G1_HELD_YAW_LEFT | REK_G1_HELD_YAW_RIGHT,
        },
        timing,
        1,
        0);
    require(decision.status == REK_G1_INPUT_REJECTED_OPPOSITE_YAW,
        "opposite_yaw_unmeasured_rejected");
    require(state.held == REK_G1_HELD_YAW_LEFT,
        "rejected_opposite_yaw_does_not_mutate_state");

    decision = rek_g1_apply_input_frame(
        &state,
        (RekG1InputFrame){.held = 0},
        timing,
        1,
        0);
    require(decision.held == 0 && state.held == 0,
        "explicit_zero_mask_releases_all_inputs");
    require(decision.released_edges == REK_G1_HELD_YAW_LEFT,
        "release_edge_reports_prior_yaw_hold");

    printf("PASS g1_held_input_contract assertions=%d long_hold_ticks=64\n", cases);
    return 0;
}
