#include "g1_fight_state.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int assertions;

static void require(int condition, const char* name) {
    assertions += 1;
    if (!condition) {
        fprintf(stderr, "FAIL %s\n", name);
        exit(1);
    }
}

static uint32_t float_bits(float value) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static RekG1FightState initialized_state(void) {
    RekG1FightState state;
    require(rek_g1_fight_state_init(&state) == REK_G1_FIGHT_OK,
        "initialized_state");
    return state;
}

static RekG1FightState prepared_state(void) {
    RekG1FightState state = initialized_state();
    RekG1FightStepResult result;
    require(rek_g1_fight_prepare_first_round(&state, &result)
        == REK_G1_FIGHT_OK, "prepared_state");
    return result.next_state;
}

static RekG1FightState active_state(void) {
    RekG1FightState state = prepared_state();
    RekG1FightStepResult result;
    require(rek_g1_fight_activate_round(&state, &result)
        == REK_G1_FIGHT_OK, "active_state");
    return result.next_state;
}

static RekG1StrikeEvent strike_event(void) {
    RekG1StrikeEvent event = {
        .attacker_fighter_index = 0u,
        .victim_fighter_index = 1u,
        .hand_side = REK_G1_HAND_LEFT,
        .striker_part = REK_G1_BODY_PART_HAND,
        .zone = REK_G1_BODY_ZONE_HEAD,
        .relative_speed = 1.75f,
        .attribution_passed = 1u,
        .score_accepted = 1u,
    };
    return event;
}

static RekG1FightAdvanceInput advance_input(
    float delta_seconds,
    float time_remaining_seconds
) {
    RekG1FightAdvanceInput input;
    memset(&input, 0, sizeof(input));
    input.delta_seconds = delta_seconds;
    input.time_remaining_seconds = time_remaining_seconds;
    return input;
}

static uint32_t record_strike(
    RekG1FightState* state,
    const RekG1StrikeEvent* event,
    RekG1FightStepResult* result_out
) {
    RekG1FightStepResult result;
    require(rek_g1_fight_record_strike(state, event, &result)
        == REK_G1_FIGHT_OK, "record_strike helper");
    *state = result.next_state;
    if (result_out != NULL) {
        *result_out = result;
    }
    return result.signals;
}

static uint32_t falling(
    RekG1FightState* state,
    uint32_t fighter,
    uint8_t estop,
    RekG1FightStepResult* result_out
) {
    RekG1FightStepResult result;
    require(rek_g1_fight_on_falling(state, fighter, estop, &result)
        == REK_G1_FIGHT_OK, "falling helper");
    *state = result.next_state;
    if (result_out != NULL) {
        *result_out = result;
    }
    return result.signals;
}

static uint32_t fallen(
    RekG1FightState* state,
    uint32_t fighter,
    uint8_t can0,
    uint8_t can1,
    RekG1FightStepResult* result_out
) {
    const uint8_t can_get_up[2] = {can0, can1};
    RekG1FightStepResult result;
    require(rek_g1_fight_on_fallen(
        state, fighter, can_get_up, &result) == REK_G1_FIGHT_OK,
        "fallen helper");
    *state = result.next_state;
    if (result_out != NULL) {
        *result_out = result;
    }
    return result.signals;
}

static uint32_t advance_active(
    RekG1FightState* state,
    const RekG1FightAdvanceInput* input,
    RekG1FightStepResult* result_out
) {
    RekG1FightStepResult result;
    require(rek_g1_fight_advance_active(state, input, &result)
        == REK_G1_FIGHT_OK, "advance_active helper");
    *state = result.next_state;
    if (result_out != NULL) {
        *result_out = result;
    }
    return result.signals;
}

static uint32_t advance_transition(
    RekG1FightState* state,
    float delta_seconds,
    RekG1FightStepResult* result_out
) {
    RekG1FightStepResult result;
    require(rek_g1_fight_advance_transition(
        state, delta_seconds, &result) == REK_G1_FIGHT_OK,
        "advance_transition helper");
    *state = result.next_state;
    if (result_out != NULL) {
        *result_out = result;
    }
    return result.signals;
}

static void activate(RekG1FightState* state) {
    RekG1FightStepResult result;
    require(rek_g1_fight_activate_round(state, &result)
        == REK_G1_FIGHT_OK, "activate helper");
    require(result.signals == REK_G1_FIGHT_SIGNAL_ROUND_STARTED,
        "activate helper signal");
    *state = result.next_state;
}

static void test_exact_config(void) {
    const RekG1FightConfig* config = &REK_G1_FIGHT_CONFIG_F84F1874;
    require(strcmp(REK_G1_FIGHT_BUILD_FINGERPRINT,
        "f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659")
        == 0, "config fingerprint");
    require(float_bits(config->normal_round_seconds) == 0x42f00000u,
        "normal round bits");
    require(float_bits(config->redo_round_seconds) == 0x41f00000u,
        "redo round bits");
    require(float_bits(config->between_round_seconds) == 0x40a00000u,
        "between round bits");
    require(float_bits(config->fight_over_seconds) == 0x40a00000u,
        "fight over bits");
    require(float_bits(config->hit_speed_threshold) == 0x3fe00000u,
        "hit speed bits");
    require(float_bits(config->knockdown_weak_speed) == 0x3fe00000u,
        "weak speed bits");
    require(float_bits(config->knockdown_strong_speed) == 0x40c00000u,
        "strong speed bits");
    require(float_bits(config->knockdown_window_min_seconds) == 0x3fc00000u,
        "window min bits");
    require(float_bits(config->knockdown_window_max_seconds) == 0x40a00000u,
        "window max bits");
    require(float_bits(config->ko_count_seconds) == 0x41200000u,
        "ko count bits");
    require(float_bits(config->double_knockdown_count_seconds)
        == 0x41a00000u, "double count bits");
    require(float_bits(config->no_recovery_count_seconds) == 0x40400000u,
        "no recovery bits");
    require(config->hand_hit_points == 1, "hand points");
    require(config->kick_hit_points == 2, "kick points");
    require(config->slip_points_to_opponent == 3, "slip points");
    require(config->ko_points == 5, "ko points");
    require(config->regular_round_limit == 3u, "regular round limit");
    require(config->rounds_to_win == 2u, "rounds to win");
}

static void test_initial_and_countdown_flow(void) {
    RekG1FightState state;
    require(rek_g1_fight_state_init(NULL) == REK_G1_FIGHT_NULL_ARGUMENT,
        "init null");
    require(rek_g1_fight_state_init(&state) == REK_G1_FIGHT_OK,
        "init ok");
    require(state.phase == REK_G1_FIGHT_IDLE, "init idle");
    require(state.current_round_number == 0u, "init round zero");
    require(state.round_winner_index == -1, "init round winner");
    require(state.fight_winner_index == -1, "init fight winner");

    RekG1FightStepResult result;
    require(rek_g1_fight_prepare_first_round(&state, &result)
        == REK_G1_FIGHT_OK, "prepare first");
    require(result.signals == (REK_G1_FIGHT_SIGNAL_ROUND_PREPARED
        | REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN),
        "prepare signals");
    state = result.next_state;
    require(state.phase == REK_G1_FIGHT_ROUND_COUNTDOWN,
        "prepare countdown");
    require(state.current_round_number == 1u, "prepare round one");
    require(!state.current_round_is_redo, "prepare regular");
    require(state.round_duration_seconds == 120.0f, "prepare duration");
    require(state.time_remaining_seconds == 120.0f, "prepare time");

    require(rek_g1_fight_prepare_first_round(&state, &result)
        == REK_G1_FIGHT_WRONG_PHASE, "prepare twice rejected");

    require(rek_g1_fight_on_reset_due(&state, 0u, &result)
        == REK_G1_FIGHT_OK, "countdown reset due");
    require(result.signals == REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN,
        "countdown reset signal");

    RekG1FightState unchanged = state;
    require(falling(&state, 0u, 0u, &result) == 0u,
        "countdown falling ignored");
    require(memcmp(&state, &unchanged, sizeof(state)) == 0,
        "countdown falling unchanged");
    require(fallen(&state, 0u, 0u, 0u, &result) == 0u,
        "countdown fallen ignored");
    require(memcmp(&state, &unchanged, sizeof(state)) == 0,
        "countdown fallen unchanged");

    require(advance_transition(&state, 2.5f, &result) == 0u,
        "countdown advance no activation");
    require(state.phase == REK_G1_FIGHT_ROUND_COUNTDOWN,
        "countdown remains pending");
    activate(&state);
    require(state.phase == REK_G1_FIGHT_ROUND_ACTIVE, "round active");
}

static void test_strike_scoring_and_validation(void) {
    RekG1FightState state = active_state();
    RekG1FightStepResult result;
    RekG1StrikeEvent event = strike_event();
    event.attribution_passed = 0u;
    require(record_strike(&state, &event, &result)
        == REK_G1_FIGHT_SIGNAL_SCORE_CHANGED, "hand score signal");
    require(result.score_delta[0] == 1 && result.score_delta[1] == 0,
        "hand score delta");
    require(state.clean_hits[0] == 1, "hand score state");
    require(!state.last_struck_valid[1], "score can omit attribution");

    event.striker_part = REK_G1_BODY_PART_FOOT;
    event.hand_side = REK_G1_HAND_RIGHT;
    require(record_strike(&state, &event, &result)
        == REK_G1_FIGHT_SIGNAL_SCORE_CHANGED, "foot score signal");
    require(result.score_delta[0] == 2, "foot score delta");

    event.striker_part = REK_G1_BODY_PART_SHIN;
    event.zone = REK_G1_BODY_ZONE_RIGHT_HIP;
    record_strike(&state, &event, &result);
    require(result.score_delta[0] == 2, "shin score delta");
    require(state.clean_hits[0] == 5, "all score sum");

    event = strike_event();
    event.attribution_passed = 1u;
    event.score_accepted = 0u;
    event.zone = REK_G1_BODY_ZONE_UNKNOWN;
    event.relative_speed = 2.25f;
    require(record_strike(&state, &event, &result) == 0u,
        "attribution only no score signal");
    require(result.score_delta[0] == 0, "attribution no score delta");
    require(state.clean_hits[0] == 5, "attribution no score state");
    require(state.last_struck_valid[1], "attribution stored");
    require(state.last_struck_speed[1] == 2.25f,
        "attribution speed stored");

    RekG1FightState base = active_state();
    event = strike_event();
    event.relative_speed = nextafterf(1.75f, 0.0f);
    require(rek_g1_fight_record_strike(&base, &event, &result)
        == REK_G1_FIGHT_INPUT_INVALID, "strike below threshold rejected");
    event = strike_event();
    event.zone = REK_G1_BODY_ZONE_LEFT_SHOULDER;
    require(rek_g1_fight_record_strike(&base, &event, &result)
        == REK_G1_FIGHT_INPUT_INVALID, "nonscoring zone rejected");
    event = strike_event();
    event.striker_part = REK_G1_BODY_PART_FOREARM;
    require(rek_g1_fight_record_strike(&base, &event, &result)
        == REK_G1_FIGHT_INPUT_INVALID, "nonstriker part rejected");
    event = strike_event();
    event.victim_fighter_index = 0u;
    require(rek_g1_fight_record_strike(&base, &event, &result)
        == REK_G1_FIGHT_INPUT_INVALID, "same fighter rejected");
    event = strike_event();
    event.attribution_passed = 0u;
    event.score_accepted = 0u;
    require(rek_g1_fight_record_strike(&base, &event, &result)
        == REK_G1_FIGHT_INPUT_INVALID, "empty strike rejected");
    event = strike_event();
    event.relative_speed = NAN;
    require(rek_g1_fight_record_strike(&base, &event, &result)
        == REK_G1_FIGHT_NON_FINITE, "nan strike rejected");
    event = strike_event();
    base.clean_hits[0] = INT32_MAX;
    require(rek_g1_fight_record_strike(&base, &event, &result)
        == REK_G1_FIGHT_OVERFLOW, "score overflow rejected");
}

static RekG1FightState state_with_attribution(
    float speed,
    float age_seconds
) {
    RekG1FightState state = active_state();
    RekG1StrikeEvent event = strike_event();
    event.relative_speed = speed;
    event.score_accepted = 0u;
    record_strike(&state, &event, NULL);
    if (age_seconds > 0.0f) {
        RekG1FightAdvanceInput input = advance_input(
            age_seconds, state.time_remaining_seconds - age_seconds);
        advance_active(&state, &input, NULL);
    }
    return state;
}

static void test_fall_classification_boundaries(void) {
    RekG1FightStepResult result;
    RekG1FightState state = active_state();
    require(falling(&state, 1u, 0u, &result)
        == REK_G1_FIGHT_SIGNAL_FALL_CLASSIFIED,
        "no strike fall classified");
    require(result.fall_classification == REK_G1_FALL_SLIP,
        "no strike is slip");

    state = state_with_attribution(1.75f, 1.5f);
    falling(&state, 1u, 0u, &result);
    require(result.fall_classification == REK_G1_FALL_KNOCKDOWN,
        "weak exact window knockdown");

    state = state_with_attribution(1.75f, 0.0f);
    state.last_struck_age_seconds[1] = nextafterf(1.5f, INFINITY);
    falling(&state, 1u, 0u, &result);
    require(result.fall_classification == REK_G1_FALL_SLIP,
        "weak above window slip");

    state = state_with_attribution(6.0f, 5.0f);
    falling(&state, 1u, 0u, &result);
    require(result.fall_classification == REK_G1_FALL_KNOCKDOWN,
        "strong exact window knockdown");

    state = state_with_attribution(8.0f, 0.0f);
    state.last_struck_age_seconds[1] = nextafterf(5.0f, INFINITY);
    falling(&state, 1u, 0u, &result);
    require(result.fall_classification == REK_G1_FALL_SLIP,
        "above strong clamps window");

    state = state_with_attribution(6.0f, 0.0f);
    falling(&state, 1u, 1u, &result);
    require(result.fall_classification == REK_G1_FALL_SLIP,
        "estop forces slip");
    require(state.fall_forced_by_estop[1], "estop latch");
    RekG1FightState latched = state;
    require(falling(&state, 1u, 0u, &result) == 0u,
        "classification latches once");
    require(memcmp(&state, &latched, sizeof(state)) == 0,
        "latched classification unchanged");
}

static void test_count_start_and_restart(void) {
    RekG1FightState state = active_state();
    RekG1FightStepResult result;
    require(fallen(&state, 0u, 1u, 0u, &result)
        == REK_G1_FIGHT_SIGNAL_COUNT_STARTED,
        "direct fallen starts count");
    require(state.falls[0] == 1u, "fallen increments fall");
    require(state.count_is_slip[0], "direct fallen defaults slip");
    require(state.count_duration_seconds == 10.0f,
        "can get up count ten");
    require(result.referee_calls == REK_G1_REFEREE_SLIP,
        "direct fall slip call");

    state = active_state();
    falling(&state, 0u, 1u, NULL);
    fallen(&state, 0u, 0u, 0u, &result);
    require(state.count_duration_seconds == 3.0f,
        "no recovery count three");
    require(result.referee_calls == REK_G1_REFEREE_SLIP_ESTOP,
        "estop slip call");

    state = state_with_attribution(3.0f, 0.0f);
    falling(&state, 1u, 0u, NULL);
    fallen(&state, 1u, 1u, 1u, &result);
    require(!state.count_is_slip[1], "knockdown count classification");
    require(result.referee_calls == REK_G1_REFEREE_KNOCKDOWN,
        "knockdown referee call");

    state = active_state();
    fallen(&state, 0u, 0u, 0u, NULL);
    RekG1FightAdvanceInput input = advance_input(2.0f, 118.0f);
    input.fighter_is_fallen[0] = 1u;
    advance_active(&state, &input, NULL);
    require(state.count_elapsed_seconds == 2.0f,
        "first count elapsed");
    require(fallen(&state, 1u, 0u, 1u, &result)
        == REK_G1_FIGHT_SIGNAL_COUNT_RESTARTED,
        "second fall restarts count");
    require(state.count_elapsed_seconds == 0.0f,
        "double count restarts elapsed");
    require(state.count_duration_seconds == 20.0f,
        "either recovery double twenty");
    require((result.referee_calls & REK_G1_REFEREE_DOUBLE_KNOCKDOWN) != 0u,
        "double knockdown call");

    state = active_state();
    fallen(&state, 0u, 0u, 0u, NULL);
    fallen(&state, 1u, 0u, 0u, &result);
    require(state.count_duration_seconds == 3.0f,
        "neither recovery double three");
    const uint8_t bad_flags[2] = {0u, 2u};
    RekG1FightState bad_flag_state = active_state();
    require(rek_g1_fight_on_fallen(&bad_flag_state, 0u,
        bad_flags, &result) == REK_G1_FIGHT_INPUT_INVALID,
        "bad CanGetUp rejected");
}

static void test_recovery_flow(void) {
    RekG1FightState state = active_state();
    RekG1FightStepResult result;
    fallen(&state, 0u, 1u, 0u, NULL);
    RekG1FightAdvanceInput input = advance_input(0.5f, 119.5f);
    input.fighter_is_fallen[0] = 0u;
    input.fighter_is_recovering[0] = 0u;
    advance_active(&state, &input, &result);
    require((result.signals & REK_G1_FIGHT_SIGNAL_COUNT_CLEARED) != 0u,
        "slip recovery clears");
    require((result.signals & REK_G1_FIGHT_SIGNAL_SCORE_CHANGED) != 0u,
        "slip recovery scores");
    require(result.score_delta[1] == 3, "slip recovery three");
    require(result.referee_calls == REK_G1_REFEREE_BEAT_COUNT,
        "slip recovery beat count");
    require(!state.count_active[0], "slip count inactive");

    state = state_with_attribution(3.0f, 0.0f);
    falling(&state, 1u, 0u, NULL);
    fallen(&state, 1u, 1u, 1u, NULL);
    input = advance_input(0.5f, 119.5f);
    advance_active(&state, &input, &result);
    require(result.score_delta[0] == 0 && result.score_delta[1] == 0,
        "knockdown recovery no points");
    require(result.referee_calls == REK_G1_REFEREE_BEAT_COUNT,
        "knockdown recovery beat count");

    state = active_state();
    fallen(&state, 0u, 0u, 0u, NULL);
    input = advance_input(1.0f, 119.0f);
    input.fighter_is_recovering[0] = 1u;
    advance_active(&state, &input, &result);
    require(state.count_active[0], "recovering keeps count active");
    require(state.count_elapsed_seconds == 1.0f,
        "recovering count advances");

    state = active_state();
    fallen(&state, 0u, 0u, 0u, NULL);
    input = advance_input(3.0f, 117.0f);
    input.fighter_is_fallen[0] = 0u;
    advance_active(&state, &input, &result);
    require(result.score_delta[1] == 3,
        "recovery wins deadline boundary");
    require((result.referee_calls & REK_G1_REFEREE_KNOCKOUT) == 0u,
        "recovery boundary no knockout");

    state = active_state();
    fallen(&state, 0u, 1u, 0u, NULL);
    input = advance_input(1.0f, 0.0f);
    advance_active(&state, &input, &result);
    require((result.signals & REK_G1_FIGHT_SIGNAL_ROUND_ENDED) != 0u,
        "recovery at zero ends round");
    require(state.round_result == REK_G1_ROUND_WON_BY_POINTS,
        "slip points decide zero-time round");
    require(state.round_winner_index == 1, "slip zero-time winner");
}

static RekG1FightState single_count_state(uint8_t can_get_up_at_fall) {
    RekG1FightState state = active_state();
    falling(&state, 0u, 0u, NULL);
    fallen(&state, 0u, can_get_up_at_fall, 0u, NULL);
    return state;
}

static RekG1FightState double_count_state(
    uint8_t can0,
    uint8_t can1
) {
    RekG1FightState state = active_state();
    fallen(&state, 0u, can0, can1, NULL);
    fallen(&state, 1u, can0, can1, NULL);
    return state;
}

static void test_single_count_expiry(void) {
    RekG1FightStepResult result;
    RekG1FightState state = single_count_state(1u);
    RekG1FightAdvanceInput input = advance_input(10.0f, 110.0f);
    input.fighter_is_fallen[0] = 1u;
    input.fighter_can_get_up[0] = 1u;
    advance_active(&state, &input, &result);
    require(result.score_delta[1] == 5, "single ko five points");
    require(result.rounds_won_delta[1] == 1, "single ko round win");
    require(result.referee_calls == REK_G1_REFEREE_KNOCKOUT,
        "single ko call");
    require((result.signals & REK_G1_FIGHT_SIGNAL_ROUND_ENDED) != 0u,
        "single ko round terminal");
    require((result.signals & REK_G1_FIGHT_SIGNAL_FIGHT_ENDED) == 0u,
        "first ko not fight terminal");
    require(state.round_result == REK_G1_ROUND_WON_BY_KO,
        "single ko result");
    require(state.round_winner_index == 1, "single ko winner");
    require(state.phase == REK_G1_FIGHT_BETWEEN_ROUNDS,
        "single ko between rounds");

    state = single_count_state(1u);
    input = advance_input(10.0f, 110.0f);
    input.fighter_is_fallen[0] = 1u;
    input.fighter_can_get_up[0] = 0u;
    advance_active(&state, &input, &result);
    require(result.score_delta[1] == 5, "refreshed no recovery five");
    require((result.signals & REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN)
        != 0u, "refreshed no recovery resets");
    require((result.signals & REK_G1_FIGHT_SIGNAL_ROUND_ENDED) == 0u,
        "refreshed no recovery continues");
    require(state.phase == REK_G1_FIGHT_ROUND_ACTIVE,
        "no recovery remains active");

    state = single_count_state(0u);
    input = advance_input(3.0f, 117.0f);
    input.fighter_is_fallen[0] = 1u;
    input.fighter_can_get_up[0] = 1u;
    advance_active(&state, &input, &result);
    require(state.round_result == REK_G1_ROUND_WON_BY_KO,
        "expiry refresh can recover makes ko");

    state = single_count_state(0u);
    input = advance_input(3.0f, 0.0f);
    input.fighter_is_fallen[0] = 1u;
    advance_active(&state, &input, &result);
    require(result.score_delta[1] == 5, "zero-time no recovery points");
    require((result.signals & REK_G1_FIGHT_SIGNAL_ROUND_ENDED) != 0u,
        "zero-time no recovery ends");
    require(state.round_result == REK_G1_ROUND_WON_BY_POINTS,
        "zero-time no recovery point result");
    require(!state.knockout_occurred, "no recovery is not ko result");
}

static void test_double_count_expiry(void) {
    RekG1FightStepResult result;
    RekG1FightState state = double_count_state(1u, 0u);
    RekG1FightAdvanceInput input = advance_input(20.0f, 100.0f);
    input.fighter_is_fallen[0] = 1u;
    input.fighter_is_fallen[1] = 1u;
    input.fighter_can_get_up[0] = 1u;
    advance_active(&state, &input, &result);
    require(result.score_delta[0] == 5 && result.score_delta[1] == 5,
        "double ko five each");
    require(result.referee_calls == REK_G1_REFEREE_DOUBLE_KNOCKOUT,
        "double ko call");
    require(state.round_result == REK_G1_ROUND_TIE, "double ko tie");
    require(state.knockout_occurred, "double ko occurred");
    require(state.phase == REK_G1_FIGHT_BETWEEN_ROUNDS,
        "double ko ends round");

    state = double_count_state(0u, 0u);
    input = advance_input(3.0f, 117.0f);
    input.fighter_is_fallen[0] = 1u;
    input.fighter_is_fallen[1] = 1u;
    advance_active(&state, &input, &result);
    require(result.score_delta[0] == 5 && result.score_delta[1] == 5,
        "double no recovery points");
    require((result.signals & REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN)
        != 0u, "double no recovery resets");
    require((result.signals & REK_G1_FIGHT_SIGNAL_ROUND_ENDED) == 0u,
        "double no recovery continues");

    state = active_state();
    RekG1StrikeEvent event = strike_event();
    event.attribution_passed = 0u;
    event.striker_part = REK_G1_BODY_PART_FOOT;
    record_strike(&state, &event, NULL);
    fallen(&state, 0u, 0u, 0u, NULL);
    fallen(&state, 1u, 0u, 0u, NULL);
    input = advance_input(3.0f, 0.0f);
    input.fighter_is_fallen[0] = 1u;
    input.fighter_is_fallen[1] = 1u;
    advance_active(&state, &input, &result);
    require(state.clean_hits[0] == 7 && state.clean_hits[1] == 5,
        "double equal award preserves prior margin");
    require(state.round_result == REK_G1_ROUND_WON_BY_POINTS,
        "double no recovery zero evaluates points");
    require(state.round_winner_index == 0, "double zero winner");

    state = double_count_state(1u, 1u);
    input = advance_input(2.0f, 118.0f);
    input.fighter_is_fallen[0] = 0u;
    input.fighter_is_fallen[1] = 1u;
    input.fighter_can_get_up[1] = 1u;
    advance_active(&state, &input, &result);
    require(!state.count_active[0] && state.count_active[1],
        "one double fighter recovers");
    require(result.score_delta[1] == 3,
        "recovered double slip awards opponent");
    require(state.count_elapsed_seconds == 2.0f,
        "shared count continues");
}

static void test_count_expiry_branch_matrix(void) {
    for (uint32_t active_mask = 1u; active_mask <= 3u; active_mask += 1u) {
        for (uint32_t can_mask = 0u; can_mask <= 3u; can_mask += 1u) {
            for (uint32_t zero_time = 0u; zero_time <= 1u; zero_time += 1u) {
                const uint8_t can0 = (uint8_t)((can_mask & 1u) != 0u);
                const uint8_t can1 = (uint8_t)((can_mask & 2u) != 0u);
                RekG1FightState state = active_state();
                if (active_mask & 1u) {
                    fallen(&state, 0u, can0, can1, NULL);
                }
                if (active_mask & 2u) {
                    fallen(&state, 1u, can0, can1, NULL);
                }
                const float duration = state.count_duration_seconds;
                RekG1FightAdvanceInput input = advance_input(
                    duration, zero_time ? 0.0f : 100.0f);
                input.fighter_is_fallen[0] =
                    (uint8_t)((active_mask & 1u) != 0u);
                input.fighter_is_fallen[1] =
                    (uint8_t)((active_mask & 2u) != 0u);
                input.fighter_can_get_up[0] = can0;
                input.fighter_can_get_up[1] = can1;
                RekG1FightStepResult result;
                advance_active(&state, &input, &result);

                require(!state.count_active[0] && !state.count_active[1],
                    "matrix expiry clears counts");
                require((result.signals & REK_G1_FIGHT_SIGNAL_SCORE_CHANGED)
                    != 0u, "matrix expiry changes score");
                if (active_mask == 3u) {
                    require(result.score_delta[0] == 5
                        && result.score_delta[1] == 5,
                        "matrix double awards five each");
                    require(result.referee_calls
                        == REK_G1_REFEREE_DOUBLE_KNOCKOUT,
                        "matrix double call");
                    const int any_can = can0 || can1;
                    if (any_can || zero_time) {
                        require((result.signals
                            & REK_G1_FIGHT_SIGNAL_ROUND_ENDED) != 0u,
                            "matrix double terminal branch");
                        require(state.round_result == REK_G1_ROUND_TIE,
                            "matrix double terminal tie");
                        require(state.knockout_occurred == (uint8_t)any_can,
                            "matrix double knockout flag");
                    } else {
                        require((result.signals
                            & REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN) != 0u,
                            "matrix double continue reset");
                        require(state.phase == REK_G1_FIGHT_ROUND_ACTIVE,
                            "matrix double continue phase");
                    }
                } else {
                    const uint32_t fallen_fighter = active_mask == 1u
                        ? 0u : 1u;
                    const uint32_t winner = 1u - fallen_fighter;
                    const uint8_t fallen_can_get_up = fallen_fighter == 0u
                        ? can0 : can1;
                    require(result.score_delta[winner] == 5,
                        "matrix single winner five");
                    require(result.score_delta[fallen_fighter] == 0,
                        "matrix single loser zero");
                    require(result.referee_calls == REK_G1_REFEREE_KNOCKOUT,
                        "matrix single call");
                    if (fallen_can_get_up || zero_time) {
                        require((result.signals
                            & REK_G1_FIGHT_SIGNAL_ROUND_ENDED) != 0u,
                            "matrix single terminal branch");
                        require(state.round_winner_index == (int32_t)winner,
                            "matrix single terminal winner");
                        require(state.round_result
                            == (fallen_can_get_up
                                ? REK_G1_ROUND_WON_BY_KO
                                : REK_G1_ROUND_WON_BY_POINTS),
                            "matrix single result type");
                    } else {
                        require((result.signals
                            & REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN) != 0u,
                            "matrix single continue reset");
                        require(state.phase == REK_G1_FIGHT_ROUND_ACTIVE,
                            "matrix single continue phase");
                    }
                }
            }
        }
    }
}

static void end_current_round(
    RekG1FightState* state,
    int32_t winner
) {
    if (winner >= 0) {
        RekG1StrikeEvent event = strike_event();
        event.attribution_passed = 0u;
        event.attacker_fighter_index = (uint32_t)winner;
        event.victim_fighter_index = 1u - (uint32_t)winner;
        record_strike(state, &event, NULL);
    }
    RekG1FightAdvanceInput input = advance_input(0.25f, 0.0f);
    RekG1FightStepResult result;
    advance_active(state, &input, &result);
    require((result.signals & REK_G1_FIGHT_SIGNAL_ROUND_ENDED) != 0u,
        "end current round signal");
}

static void prepare_next_and_activate(
    RekG1FightState* state,
    float expected_duration
) {
    RekG1FightStepResult result;
    require(advance_transition(state, 5.0f, &result)
        == (REK_G1_FIGHT_SIGNAL_ROUND_PREPARED
            | REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN),
        "next round prepared signals");
    require(state->phase == REK_G1_FIGHT_ROUND_COUNTDOWN,
        "next round countdown");
    require(state->round_duration_seconds == expected_duration,
        "next round expected duration");
    activate(state);
}

static void test_round_series_and_transitions(void) {
    RekG1FightState state = active_state();
    end_current_round(&state, -1);
    require(state.round_result == REK_G1_ROUND_TIE, "round one tie");
    require(state.phase == REK_G1_FIGHT_BETWEEN_ROUNDS,
        "tie between rounds");
    RekG1FightStepResult result;
    require(advance_transition(&state, 4.0f, &result) == 0u,
        "between partial no signal");
    require(state.transition_remaining_seconds == 1.0f,
        "between partial timer");
    prepare_next_and_activate(&state, 30.0f);
    require(state.current_round_number == 2u, "redo round number two");
    require(state.current_round_is_redo, "tie makes redo");

    end_current_round(&state, 1);
    require(state.rounds_won[1] == 1u, "round two fighter one win");
    prepare_next_and_activate(&state, 120.0f);
    require(state.current_round_number == 3u, "normal round three");
    require(!state.current_round_is_redo,
        "non-tied preceding round makes normal");

    end_current_round(&state, 0);
    require(state.rounds_won[0] == 1u && state.rounds_won[1] == 1u,
        "three round equal series");
    require(state.phase == REK_G1_FIGHT_BETWEEN_ROUNDS,
        "equal series unresolved");
    prepare_next_and_activate(&state, 120.0f);
    require(state.current_round_number == 4u, "extra round number four");
    require(!state.current_round_is_redo,
        "equal series after win remains normal");

    end_current_round(&state, 0);
    require(state.phase == REK_G1_FIGHT_OVER, "two wins fight over");
    require(state.fight_result == REK_G1_FIGHT_WON_BY_ROUNDS,
        "fight result by rounds");
    require(state.fight_winner_index == 0, "fight winner zero");
    require(advance_transition(&state, 4.0f, &result) == 0u,
        "fight over partial no signal");
    require(state.phase == REK_G1_FIGHT_OVER, "fight over retained");
    require(advance_transition(&state, 1.0f, &result)
        == REK_G1_FIGHT_SIGNAL_FIGHT_EXITED, "fight exit signal");
    require(state.phase == REK_G1_FIGHT_IDLE, "fight exit idle");
    require(rek_g1_fight_prepare_first_round(&state, &result)
        == REK_G1_FIGHT_WRONG_PHASE,
        "completed fight requires explicit reinit");
}

static void test_clock_and_phase_guards(void) {
    RekG1FightState state = single_count_state(1u);
    RekG1FightAdvanceInput input = advance_input(1.0f, 0.0f);
    input.fighter_is_fallen[0] = 1u;
    RekG1FightStepResult result;
    advance_active(&state, &input, &result);
    require(state.phase == REK_G1_FIGHT_ROUND_ACTIVE,
        "zero clock waits for active count");
    require((result.signals & REK_G1_FIGHT_SIGNAL_ROUND_ENDED) == 0u,
        "zero clock no early end");

    state = active_state();
    input = advance_input(0.25f, 0.0f);
    advance_active(&state, &input, &result);
    require(state.round_result == REK_G1_ROUND_TIE,
        "ordinary zero clock tie");
    require((result.signals & REK_G1_FIGHT_SIGNAL_ROUND_ENDED) != 0u,
        "ordinary zero clock ends");

    state = active_state();
    input = advance_input(0.0f, 120.0f);
    require(rek_g1_fight_advance_active(&state, &input, &result)
        == REK_G1_FIGHT_INPUT_INVALID, "zero delta rejected");
    input = advance_input(0.1f, 121.0f);
    require(rek_g1_fight_advance_active(&state, &input, &result)
        == REK_G1_FIGHT_INPUT_INVALID, "increasing round clock rejected");
    input = advance_input(NAN, 120.0f);
    require(rek_g1_fight_advance_active(&state, &input, &result)
        == REK_G1_FIGHT_NON_FINITE, "nan delta rejected");
    input = advance_input(0.1f, 119.9f);
    input.fighter_can_get_up[0] = 2u;
    require(rek_g1_fight_advance_active(&state, &input, &result)
        == REK_G1_FIGHT_INPUT_INVALID, "invalid observed flag rejected");

    require(rek_g1_fight_advance_transition(&state, 1.0f, &result)
        == REK_G1_FIGHT_WRONG_PHASE, "active transition rejected");
    require(rek_g1_fight_activate_round(&state, &result)
        == REK_G1_FIGHT_WRONG_PHASE, "active activation rejected");
    require(rek_g1_fight_on_reset_due(&state, 0u, &result)
        == REK_G1_FIGHT_OK, "active reset due accepted");
    require(result.signals == 0u, "active reset due deliberately ignored");

    RekG1FightState invalid = state;
    invalid.count_active[0] = 1u;
    invalid.count_duration_seconds = 0.0f;
    require(rek_g1_fight_on_falling(&invalid, 0u, 0u, &result)
        == REK_G1_FIGHT_STATE_INVALID, "invalid state rejected");
    invalid = state;
    invalid.last_struck_age_seconds[0] = INFINITY;
    require(rek_g1_fight_on_falling(&invalid, 0u, 0u, &result)
        == REK_G1_FIGHT_NON_FINITE, "nonfinite state rejected");
}

static RekG1FightResolveInput resolve_input(
    uint8_t fallen0,
    uint8_t fallen1
) {
    RekG1FightResolveInput input;
    memset(&input, 0, sizeof(input));
    input.fighter_is_fallen[0] = fallen0;
    input.fighter_is_fallen[1] = fallen1;
    return input;
}

static void test_preactive_attribution_and_spawn_reset(void) {
    RekG1FightState state = prepared_state();
    RekG1StrikeEvent event = strike_event();
    event.zone = REK_G1_BODY_ZONE_LEFT_WRIST;
    event.score_accepted = 0u;
    RekG1FightStepResult result;
    require(rek_g1_fight_record_strike(&state, &event, &result)
        == REK_G1_FIGHT_OK, "countdown attribution accepted");
    state = result.next_state;
    require(state.last_struck_valid[1]
            && state.last_struck_speed[1] == 1.75f,
        "countdown attribution latched");
    event.zone = REK_G1_BODY_ZONE_HEAD;
    event.score_accepted = 1u;
    require(rek_g1_fight_record_strike(&state, &event, &result)
        == REK_G1_FIGHT_WRONG_PHASE,
        "countdown score remains rejected");
    require(rek_g1_fight_apply_spawn_reset(&state, &result)
        == REK_G1_FIGHT_OK, "spawn reset accepted");
    require(!result.next_state.last_struck_valid[0]
            && !result.next_state.last_struck_valid[1]
            && result.next_state.last_struck_age_seconds[1] == 0.0f
            && result.next_state.last_struck_speed[1] == 0.0f,
        "spawn reset clears transient attribution");
}

static void test_two_phase_post_step_order(void) {
    RekG1FightState state = active_state();
    RekG1FightAdvanceInput begin_input = advance_input(0.002f, 119.998f);
    RekG1FightStepResult result;
    require(rek_g1_fight_begin_active_step(
            &state, &begin_input, &result) == REK_G1_FIGHT_OK,
        "two phase begin accepted");
    state = result.next_state;
    require(state.time_remaining_seconds == 119.998f,
        "two phase begin stages round clock");

    require(rek_g1_fight_on_falling(&state, 0u, 0u, &result)
        == REK_G1_FIGHT_OK, "same state falling accepted");
    state = result.next_state;
    const uint8_t can_get_up[2] = {0u, 0u};
    require(rek_g1_fight_on_fallen(
            &state, 0u, can_get_up, &result) == REK_G1_FIGHT_OK,
        "same state fallen accepted");
    state = result.next_state;
    require(state.count_elapsed_seconds == 0.0f,
        "new count starts at zero after preceding physics interval");

    RekG1FightResolveInput resolve = resolve_input(1u, 0u);
    require(rek_g1_fight_resolve_active_step(
            &state, &resolve, &result) == REK_G1_FIGHT_OK,
        "same state resolve accepted");
    state = result.next_state;
    require(state.count_elapsed_seconds == 0.0f
            && result.signals == 0u,
        "same state resolve does not age new count");

    begin_input = advance_input(3.0f, 116.998f);
    begin_input.fighter_is_fallen[0] = 1u;
    require(rek_g1_fight_begin_active_step(
            &state, &begin_input, &result) == REK_G1_FIGHT_OK,
        "deadline begin accepted");
    state = result.next_state;
    require(state.count_elapsed_seconds == 3.0f,
        "deadline staged before resolution");
    require((result.signals & REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN)
            == 0u,
        "begin phase does not resolve deadline");
    require(rek_g1_fight_resolve_active_step(
            &state, &resolve, &result) == REK_G1_FIGHT_OK,
        "deadline resolve accepted");
    require(result.score_delta[1] == 5
            && (result.signals
                & REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN) != 0u,
        "deadline resolve awards five and requests reset");

    state = active_state();
    begin_input = advance_input(0.02f, 0.0f);
    require(rek_g1_fight_begin_active_step(
            &state, &begin_input, &result) == REK_G1_FIGHT_OK,
        "zero clock begin accepted");
    state = result.next_state;
    require(rek_g1_fight_on_falling(&state, 0u, 0u, &result)
        == REK_G1_FIGHT_OK, "deadline falling accepted");
    state = result.next_state;
    require(rek_g1_fight_on_fallen(
            &state, 0u, can_get_up, &result) == REK_G1_FIGHT_OK,
        "deadline fallen accepted");
    state = result.next_state;
    resolve = resolve_input(1u, 0u);
    require(rek_g1_fight_resolve_active_step(
            &state, &resolve, &result) == REK_G1_FIGHT_OK,
        "deadline fall resolve accepted");
    require(result.next_state.phase == REK_G1_FIGHT_ROUND_ACTIVE
            && result.next_state.count_active[0]
            && result.next_state.count_elapsed_seconds == 0.0f,
        "same state fall takes precedence over round deadline");
}

int main(void) {
    test_exact_config();
    test_initial_and_countdown_flow();
    test_strike_scoring_and_validation();
    test_fall_classification_boundaries();
    test_count_start_and_restart();
    test_recovery_flow();
    test_single_count_expiry();
    test_double_count_expiry();
    test_count_expiry_branch_matrix();
    test_round_series_and_transitions();
    test_clock_and_phase_guards();
    test_preactive_attribution_and_spawn_reset();
    test_two_phase_post_step_order();
    printf("g1_fight_state tests passed: %d assertions\n", assertions);
    return 0;
}
