#include "g1_fight_state.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

REK_G1_CONSTANT const RekG1FightConfig REK_G1_FIGHT_CONFIG_F84F1874 = {
    .normal_round_seconds = 120.0f,
    .redo_round_seconds = 30.0f,
    .between_round_seconds = 5.0f,
    .fight_over_seconds = 5.0f,
    .hit_speed_threshold = 1.75f,
    .knockdown_weak_speed = 1.75f,
    .knockdown_strong_speed = 6.0f,
    .knockdown_window_min_seconds = 1.5f,
    .knockdown_window_max_seconds = 5.0f,
    .ko_count_seconds = 10.0f,
    .double_knockdown_count_seconds = 20.0f,
    .no_recovery_count_seconds = 3.0f,
    .hand_hit_points = 1,
    .kick_hit_points = 2,
    .slip_points_to_opponent = 3,
    .ko_points = 5,
    .regular_round_limit = 3u,
    .rounds_to_win = 2u,
};

static REK_G1_FN int valid_flag(uint8_t value) {
    return value == 0u || value == 1u;
}

static REK_G1_FN int valid_fighter_index(uint32_t value) {
    return value < 2u;
}

static REK_G1_FN int valid_round_result(RekG1RoundResult value) {
    return value >= REK_G1_ROUND_IN_PROGRESS
        && value <= REK_G1_ROUND_TIE;
}

static REK_G1_FN int valid_fight_result(RekG1FightResult value) {
    return value >= REK_G1_FIGHT_IN_PROGRESS
        && value <= REK_G1_FIGHT_WON_BY_TKO;
}

static REK_G1_FN int valid_fall_classification(RekG1FallClassification value) {
    return value >= REK_G1_FALL_UNCLASSIFIED
        && value <= REK_G1_FALL_KNOCKDOWN;
}

static REK_G1_FN int valid_winner(int32_t value) {
    return value == -1 || value == 0 || value == 1;
}

static REK_G1_FN int any_count_active(const RekG1FightState* state) {
    return state->count_active[0] || state->count_active[1];
}

static REK_G1_FN RekG1FightStatus validate_state(const RekG1FightState* state) {
    if (state->phase < REK_G1_FIGHT_IDLE
            || state->phase > REK_G1_FIGHT_OVER
            || !valid_round_result(state->round_result)
            || !valid_fight_result(state->fight_result)
            || !valid_winner(state->round_winner_index)
            || !valid_winner(state->fight_winner_index)
            || !valid_flag(state->current_round_is_redo)
            || !valid_flag(state->knockout_occurred)) {
        return REK_G1_FIGHT_STATE_INVALID;
    }
    if (!isfinite(state->round_duration_seconds)
            || !isfinite(state->time_remaining_seconds)
            || !isfinite(state->last_struck_age_seconds[0])
            || !isfinite(state->last_struck_age_seconds[1])
            || !isfinite(state->last_struck_speed[0])
            || !isfinite(state->last_struck_speed[1])
            || !isfinite(state->count_elapsed_seconds)
            || !isfinite(state->count_duration_seconds)
            || !isfinite(state->transition_remaining_seconds)) {
        return REK_G1_FIGHT_NON_FINITE;
    }
    if (state->round_duration_seconds < 0.0f
            || state->time_remaining_seconds < 0.0f
            || state->time_remaining_seconds > state->round_duration_seconds
            || state->last_struck_age_seconds[0] < 0.0f
            || state->last_struck_age_seconds[1] < 0.0f
            || state->last_struck_speed[0] < 0.0f
            || state->last_struck_speed[1] < 0.0f
            || state->count_elapsed_seconds < 0.0f
            || state->count_duration_seconds < 0.0f
            || state->transition_remaining_seconds < 0.0f
            || state->clean_hits[0] < 0
            || state->clean_hits[1] < 0) {
        return REK_G1_FIGHT_STATE_INVALID;
    }
    for (uint32_t fighter = 0u; fighter < 2u; fighter += 1u) {
        if (!valid_flag(state->last_struck_valid[fighter])
                || !valid_flag(state->fall_forced_by_estop[fighter])
                || !valid_flag(state->count_active[fighter])
                || !valid_flag(state->count_is_slip[fighter])
                || !valid_fall_classification(
                    state->fall_classification[fighter])) {
            return REK_G1_FIGHT_STATE_INVALID;
        }
        if (!state->last_struck_valid[fighter]
                && (state->last_struck_age_seconds[fighter] != 0.0f
                    || state->last_struck_speed[fighter] != 0.0f)) {
            return REK_G1_FIGHT_STATE_INVALID;
        }
        if (state->fall_classification[fighter] == REK_G1_FALL_UNCLASSIFIED
                && state->fall_forced_by_estop[fighter]) {
            return REK_G1_FIGHT_STATE_INVALID;
        }
        if (!state->count_active[fighter] && state->count_is_slip[fighter]) {
            return REK_G1_FIGHT_STATE_INVALID;
        }
    }

    const int has_count = any_count_active(state);
    if (has_count) {
        if (state->phase != REK_G1_FIGHT_ROUND_ACTIVE
                || state->count_duration_seconds <= 0.0f
                || state->count_elapsed_seconds >
                    state->count_duration_seconds) {
            return REK_G1_FIGHT_STATE_INVALID;
        }
    } else if (state->count_duration_seconds != 0.0f
            || state->count_elapsed_seconds != 0.0f) {
        return REK_G1_FIGHT_STATE_INVALID;
    }

    if (state->phase == REK_G1_FIGHT_IDLE) {
        if (state->transition_remaining_seconds != 0.0f) {
            return REK_G1_FIGHT_STATE_INVALID;
        }
    } else {
        if (state->current_round_number == 0u
                || state->round_duration_seconds <= 0.0f) {
            return REK_G1_FIGHT_STATE_INVALID;
        }
    }
    if (state->phase == REK_G1_FIGHT_ROUND_COUNTDOWN
            || state->phase == REK_G1_FIGHT_ROUND_ACTIVE) {
        if (state->round_result != REK_G1_ROUND_IN_PROGRESS
                || state->round_winner_index != -1
                || state->knockout_occurred
                || state->fight_result != REK_G1_FIGHT_IN_PROGRESS
                || state->fight_winner_index != -1
                || state->transition_remaining_seconds != 0.0f) {
            return REK_G1_FIGHT_STATE_INVALID;
        }
    } else if (state->phase == REK_G1_FIGHT_BETWEEN_ROUNDS) {
        if (state->round_result == REK_G1_ROUND_IN_PROGRESS
                || state->fight_result != REK_G1_FIGHT_IN_PROGRESS
                || state->fight_winner_index != -1
                || state->transition_remaining_seconds <= 0.0f) {
            return REK_G1_FIGHT_STATE_INVALID;
        }
    } else if (state->phase == REK_G1_FIGHT_OVER) {
        if (state->round_result == REK_G1_ROUND_IN_PROGRESS
                || state->fight_result != REK_G1_FIGHT_WON_BY_ROUNDS
                || !valid_fighter_index((uint32_t)state->fight_winner_index)
                || state->transition_remaining_seconds <= 0.0f) {
            return REK_G1_FIGHT_STATE_INVALID;
        }
    }
    return REK_G1_FIGHT_OK;
}

static REK_G1_FN void initialize_result(
    const RekG1FightState* state,
    RekG1FightStepResult* result
) {
    memset(result, 0, sizeof(*result));
    result->next_state = *state;
    result->subject_fighter_index = -1;
    result->fall_classification = REK_G1_FALL_UNCLASSIFIED;
}

static REK_G1_FN int add_score(
    RekG1FightStepResult* result,
    uint32_t fighter,
    int32_t points
) {
    if (points < 0
            || result->next_state.clean_hits[fighter] > INT32_MAX - points
            || result->score_delta[fighter] > INT32_MAX - points) {
        return 0;
    }
    result->next_state.clean_hits[fighter] += points;
    result->score_delta[fighter] += points;
    result->signals |= REK_G1_FIGHT_SIGNAL_SCORE_CHANGED;
    return 1;
}

static REK_G1_FN int add_round_win(
    RekG1FightStepResult* result,
    uint32_t fighter
) {
    if (result->next_state.rounds_won[fighter] == UINT32_MAX
            || result->rounds_won_delta[fighter] == INT32_MAX) {
        return 0;
    }
    result->next_state.rounds_won[fighter] += 1u;
    result->rounds_won_delta[fighter] += 1;
    return 1;
}

static REK_G1_FN void clear_referee_state(RekG1FightState* state) {
    state->fall_classification[0] = REK_G1_FALL_UNCLASSIFIED;
    state->fall_classification[1] = REK_G1_FALL_UNCLASSIFIED;
    state->fall_forced_by_estop[0] = 0u;
    state->fall_forced_by_estop[1] = 0u;
    state->count_active[0] = 0u;
    state->count_active[1] = 0u;
    state->count_is_slip[0] = 0u;
    state->count_is_slip[1] = 0u;
    state->count_elapsed_seconds = 0.0f;
    state->count_duration_seconds = 0.0f;
}

static REK_G1_FN RekG1FightStatus advance_last_strikes(
    RekG1FightState* state,
    float delta_seconds
) {
    for (uint32_t fighter = 0u; fighter < 2u; fighter += 1u) {
        if (!state->last_struck_valid[fighter]) {
            continue;
        }
        const float next = state->last_struck_age_seconds[fighter]
            + delta_seconds;
        if (!isfinite(next)) {
            return REK_G1_FIGHT_OVERFLOW;
        }
        state->last_struck_age_seconds[fighter] = next;
    }
    return REK_G1_FIGHT_OK;
}

static REK_G1_FN void prepare_round(
    RekG1FightStepResult* result,
    uint32_t round_number,
    uint8_t is_redo
) {
    RekG1FightState* next = &result->next_state;
    next->phase = REK_G1_FIGHT_ROUND_COUNTDOWN;
    next->current_round_number = round_number;
    next->current_round_is_redo = is_redo;
    next->round_duration_seconds = is_redo
        ? REK_G1_FIGHT_CONFIG_F84F1874.redo_round_seconds
        : REK_G1_FIGHT_CONFIG_F84F1874.normal_round_seconds;
    next->time_remaining_seconds = next->round_duration_seconds;
    next->clean_hits[0] = 0;
    next->clean_hits[1] = 0;
    next->falls[0] = 0u;
    next->falls[1] = 0u;
    next->round_result = REK_G1_ROUND_IN_PROGRESS;
    next->round_winner_index = -1;
    next->knockout_occurred = 0u;
    next->transition_remaining_seconds = 0.0f;
    clear_referee_state(next);
    result->signals |= REK_G1_FIGHT_SIGNAL_ROUND_PREPARED
        | REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN;
}

REK_G1_FN RekG1FightStatus rek_g1_fight_state_init(RekG1FightState* state) {
    if (state == NULL) {
        return REK_G1_FIGHT_NULL_ARGUMENT;
    }
    RekG1FightState initialized;
    memset(&initialized, 0, sizeof(initialized));
    initialized.phase = REK_G1_FIGHT_IDLE;
    initialized.round_result = REK_G1_ROUND_IN_PROGRESS;
    initialized.round_winner_index = -1;
    initialized.fight_result = REK_G1_FIGHT_IN_PROGRESS;
    initialized.fight_winner_index = -1;
    initialized.fall_classification[0] = REK_G1_FALL_UNCLASSIFIED;
    initialized.fall_classification[1] = REK_G1_FALL_UNCLASSIFIED;
    *state = initialized;
    return REK_G1_FIGHT_OK;
}

REK_G1_FN RekG1FightStatus rek_g1_fight_prepare_first_round(
    const RekG1FightState* state,
    RekG1FightStepResult* result
) {
    if (state == NULL || result == NULL) {
        return REK_G1_FIGHT_NULL_ARGUMENT;
    }
    RekG1FightStatus status = validate_state(state);
    if (status != REK_G1_FIGHT_OK) {
        return status;
    }
    if (state->phase != REK_G1_FIGHT_IDLE
            || state->current_round_number != 0u
            || state->fight_result != REK_G1_FIGHT_IN_PROGRESS) {
        return REK_G1_FIGHT_WRONG_PHASE;
    }
    initialize_result(state, result);
    prepare_round(result, 1u, 0u);
    return REK_G1_FIGHT_OK;
}

REK_G1_FN RekG1FightStatus rek_g1_fight_activate_round(
    const RekG1FightState* state,
    RekG1FightStepResult* result
) {
    if (state == NULL || result == NULL) {
        return REK_G1_FIGHT_NULL_ARGUMENT;
    }
    RekG1FightStatus status = validate_state(state);
    if (status != REK_G1_FIGHT_OK) {
        return status;
    }
    if (state->phase != REK_G1_FIGHT_ROUND_COUNTDOWN) {
        return REK_G1_FIGHT_WRONG_PHASE;
    }
    initialize_result(state, result);
    result->next_state.phase = REK_G1_FIGHT_ROUND_ACTIVE;
    result->signals = REK_G1_FIGHT_SIGNAL_ROUND_STARTED;
    return REK_G1_FIGHT_OK;
}

static REK_G1_FN int valid_striker_part(RekG1BodyPartType part) {
    return part == REK_G1_BODY_PART_HAND
        || part == REK_G1_BODY_PART_FOOT
        || part == REK_G1_BODY_PART_SHIN;
}

static REK_G1_FN int valid_body_zone(RekG1BodyZone zone) {
    return zone >= REK_G1_BODY_ZONE_UNKNOWN
        && zone <= REK_G1_BODY_ZONE_RIGHT_ANKLE;
}

static REK_G1_FN int scoring_body_zone(RekG1BodyZone zone) {
    return zone == REK_G1_BODY_ZONE_HEAD
        || zone == REK_G1_BODY_ZONE_TORSO
        || zone == REK_G1_BODY_ZONE_PELVIS
        || zone == REK_G1_BODY_ZONE_LEFT_HIP
        || zone == REK_G1_BODY_ZONE_RIGHT_HIP;
}

REK_G1_FN RekG1FightStatus rek_g1_fight_record_strike(
    const RekG1FightState* state,
    const RekG1StrikeEvent* event,
    RekG1FightStepResult* result
) {
    if (state == NULL || event == NULL || result == NULL) {
        return REK_G1_FIGHT_NULL_ARGUMENT;
    }
    RekG1FightStatus status = validate_state(state);
    if (status != REK_G1_FIGHT_OK) {
        return status;
    }
    if (!isfinite(event->relative_speed)) {
        return REK_G1_FIGHT_NON_FINITE;
    }
    if (!valid_fighter_index(event->attacker_fighter_index)
            || !valid_fighter_index(event->victim_fighter_index)
            || event->attacker_fighter_index == event->victim_fighter_index
            || (event->hand_side != REK_G1_HAND_LEFT
                && event->hand_side != REK_G1_HAND_RIGHT)
            || !valid_striker_part(event->striker_part)
            || !valid_body_zone(event->zone)
            || event->relative_speed <
                REK_G1_FIGHT_CONFIG_F84F1874.hit_speed_threshold
            || !valid_flag(event->attribution_passed)
            || !valid_flag(event->score_accepted)
            || (!event->attribution_passed && !event->score_accepted)
            || (event->score_accepted && !scoring_body_zone(event->zone))) {
        return REK_G1_FIGHT_INPUT_INVALID;
    }
    if (event->score_accepted
            && state->phase != REK_G1_FIGHT_ROUND_ACTIVE) {
        return REK_G1_FIGHT_WRONG_PHASE;
    }

    initialize_result(state, result);
    if (event->attribution_passed) {
        const uint32_t victim = event->victim_fighter_index;
        result->next_state.last_struck_valid[victim] = 1u;
        result->next_state.last_struck_age_seconds[victim] = 0.0f;
        result->next_state.last_struck_speed[victim] = event->relative_speed;
    }
    if (event->score_accepted) {
        const int32_t points = event->striker_part == REK_G1_BODY_PART_HAND
            ? REK_G1_FIGHT_CONFIG_F84F1874.hand_hit_points
            : REK_G1_FIGHT_CONFIG_F84F1874.kick_hit_points;
        if (!add_score(result, event->attacker_fighter_index, points)) {
            return REK_G1_FIGHT_OVERFLOW;
        }
    }
    return REK_G1_FIGHT_OK;
}

static REK_G1_FN float knockdown_window(float strike_speed) {
    const RekG1FightConfig* config = &REK_G1_FIGHT_CONFIG_F84F1874;
    float t = (strike_speed - config->knockdown_weak_speed)
        / (config->knockdown_strong_speed - config->knockdown_weak_speed);
    if (t < 0.0f) {
        t = 0.0f;
    } else if (t > 1.0f) {
        t = 1.0f;
    }
    return config->knockdown_window_min_seconds
        + t * (config->knockdown_window_max_seconds
            - config->knockdown_window_min_seconds);
}

REK_G1_FN RekG1FightStatus rek_g1_fight_on_falling(
    const RekG1FightState* state,
    uint32_t fighter_index,
    uint8_t force_slip_estop,
    RekG1FightStepResult* result
) {
    if (state == NULL || result == NULL) {
        return REK_G1_FIGHT_NULL_ARGUMENT;
    }
    RekG1FightStatus status = validate_state(state);
    if (status != REK_G1_FIGHT_OK) {
        return status;
    }
    if (!valid_fighter_index(fighter_index)
            || !valid_flag(force_slip_estop)) {
        return REK_G1_FIGHT_INPUT_INVALID;
    }
    initialize_result(state, result);
    if (state->phase != REK_G1_FIGHT_ROUND_ACTIVE) {
        return REK_G1_FIGHT_OK;
    }
    if (state->count_active[fighter_index]
            || state->fall_classification[fighter_index]
                != REK_G1_FALL_UNCLASSIFIED) {
        return REK_G1_FIGHT_OK;
    }

    RekG1FallClassification classification = REK_G1_FALL_SLIP;
    if (!force_slip_estop && state->last_struck_valid[fighter_index]) {
        const float window = knockdown_window(
            state->last_struck_speed[fighter_index]);
        if (state->last_struck_age_seconds[fighter_index] <= window) {
            classification = REK_G1_FALL_KNOCKDOWN;
        }
    }
    result->next_state.fall_classification[fighter_index] = classification;
    result->next_state.fall_forced_by_estop[fighter_index] =
        force_slip_estop;
    result->signals |= REK_G1_FIGHT_SIGNAL_FALL_CLASSIFIED;
    result->subject_fighter_index = (int32_t)fighter_index;
    result->fall_classification = classification;
    return REK_G1_FIGHT_OK;
}

REK_G1_FN RekG1FightStatus rek_g1_fight_on_fallen(
    const RekG1FightState* state,
    uint32_t fighter_index,
    const uint8_t fighter_can_get_up[2],
    RekG1FightStepResult* result
) {
    if (state == NULL || fighter_can_get_up == NULL || result == NULL) {
        return REK_G1_FIGHT_NULL_ARGUMENT;
    }
    RekG1FightStatus status = validate_state(state);
    if (status != REK_G1_FIGHT_OK) {
        return status;
    }
    if (!valid_fighter_index(fighter_index)
            || !valid_flag(fighter_can_get_up[0])
            || !valid_flag(fighter_can_get_up[1])) {
        return REK_G1_FIGHT_INPUT_INVALID;
    }
    initialize_result(state, result);
    if (state->phase != REK_G1_FIGHT_ROUND_ACTIVE) {
        return REK_G1_FIGHT_OK;
    }
    if (state->count_active[fighter_index]) {
        return REK_G1_FIGHT_INPUT_INVALID;
    }
    if (state->falls[fighter_index] == UINT32_MAX) {
        return REK_G1_FIGHT_OVERFLOW;
    }

    const int had_active_count = any_count_active(state);
    RekG1FallClassification classification =
        state->fall_classification[fighter_index];
    if (classification == REK_G1_FALL_UNCLASSIFIED) {
        classification = REK_G1_FALL_SLIP;
    }
    const uint8_t forced_estop =
        state->fall_forced_by_estop[fighter_index];

    result->next_state.falls[fighter_index] += 1u;
    result->next_state.fall_classification[fighter_index] =
        REK_G1_FALL_UNCLASSIFIED;
    result->next_state.fall_forced_by_estop[fighter_index] = 0u;
    result->next_state.count_active[fighter_index] = 1u;
    result->next_state.count_is_slip[fighter_index] =
        classification == REK_G1_FALL_SLIP;
    result->next_state.count_elapsed_seconds = 0.0f;
    result->subject_fighter_index = (int32_t)fighter_index;
    result->fall_classification = classification;

    if (classification == REK_G1_FALL_KNOCKDOWN) {
        result->referee_calls |= REK_G1_REFEREE_KNOCKDOWN;
    } else if (forced_estop) {
        result->referee_calls |= REK_G1_REFEREE_SLIP_ESTOP;
    } else {
        result->referee_calls |= REK_G1_REFEREE_SLIP;
    }

    if (!had_active_count) {
        result->next_state.count_duration_seconds =
            fighter_can_get_up[fighter_index]
                ? REK_G1_FIGHT_CONFIG_F84F1874.ko_count_seconds
                : REK_G1_FIGHT_CONFIG_F84F1874.no_recovery_count_seconds;
        result->signals |= REK_G1_FIGHT_SIGNAL_COUNT_STARTED;
    } else {
        result->next_state.count_duration_seconds =
            fighter_can_get_up[0] || fighter_can_get_up[1]
                ? REK_G1_FIGHT_CONFIG_F84F1874
                    .double_knockdown_count_seconds
                : REK_G1_FIGHT_CONFIG_F84F1874
                    .no_recovery_count_seconds;
        result->signals |= REK_G1_FIGHT_SIGNAL_COUNT_RESTARTED;
        result->referee_calls |= REK_G1_REFEREE_DOUBLE_KNOCKDOWN;
    }
    return REK_G1_FIGHT_OK;
}

REK_G1_FN RekG1FightStatus rek_g1_fight_on_reset_due(
    const RekG1FightState* state,
    uint32_t fighter_index,
    RekG1FightStepResult* result
) {
    if (state == NULL || result == NULL) {
        return REK_G1_FIGHT_NULL_ARGUMENT;
    }
    RekG1FightStatus status = validate_state(state);
    if (status != REK_G1_FIGHT_OK) {
        return status;
    }
    if (!valid_fighter_index(fighter_index)) {
        return REK_G1_FIGHT_INPUT_INVALID;
    }
    initialize_result(state, result);
    result->subject_fighter_index = (int32_t)fighter_index;
    if (state->phase == REK_G1_FIGHT_ROUND_COUNTDOWN) {
        result->signals |= REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN;
    }
    return REK_G1_FIGHT_OK;
}

REK_G1_FN RekG1FightStatus rek_g1_fight_apply_spawn_reset(
    const RekG1FightState* state,
    RekG1FightStepResult* result
) {
    if (state == NULL || result == NULL) {
        return REK_G1_FIGHT_NULL_ARGUMENT;
    }
    const RekG1FightStatus status = validate_state(state);
    if (status != REK_G1_FIGHT_OK) {
        return status;
    }
    initialize_result(state, result);
    for (uint32_t fighter = 0u; fighter < 2u; fighter += 1u) {
        result->next_state.last_struck_valid[fighter] = 0u;
        result->next_state.last_struck_age_seconds[fighter] = 0.0f;
        result->next_state.last_struck_speed[fighter] = 0.0f;
    }
    return REK_G1_FIGHT_OK;
}

static REK_G1_FN RekG1FightStatus end_round(RekG1FightStepResult* result) {
    RekG1FightState* next = &result->next_state;
    if (!next->knockout_occurred) {
        if (next->clean_hits[0] > next->clean_hits[1]) {
            next->round_result = REK_G1_ROUND_WON_BY_POINTS;
            next->round_winner_index = 0;
            if (!add_round_win(result, 0u)) {
                return REK_G1_FIGHT_OVERFLOW;
            }
        } else if (next->clean_hits[1] > next->clean_hits[0]) {
            next->round_result = REK_G1_ROUND_WON_BY_POINTS;
            next->round_winner_index = 1;
            if (!add_round_win(result, 1u)) {
                return REK_G1_FIGHT_OVERFLOW;
            }
        } else {
            next->round_result = REK_G1_ROUND_TIE;
            next->round_winner_index = -1;
        }
    }

    if (next->rounds_won[0] >=
            REK_G1_FIGHT_CONFIG_F84F1874.rounds_to_win
            && next->rounds_won[1] >=
                REK_G1_FIGHT_CONFIG_F84F1874.rounds_to_win) {
        return REK_G1_FIGHT_STATE_INVALID;
    }

    int32_t fight_winner = -1;
    if (next->rounds_won[0] >=
            REK_G1_FIGHT_CONFIG_F84F1874.rounds_to_win) {
        fight_winner = 0;
    } else if (next->rounds_won[1] >=
            REK_G1_FIGHT_CONFIG_F84F1874.rounds_to_win) {
        fight_winner = 1;
    } else if (next->current_round_number >=
            REK_G1_FIGHT_CONFIG_F84F1874.regular_round_limit) {
        if (next->rounds_won[0] > next->rounds_won[1]) {
            fight_winner = 0;
        } else if (next->rounds_won[1] > next->rounds_won[0]) {
            fight_winner = 1;
        }
    }

    clear_referee_state(next);
    result->signals |= REK_G1_FIGHT_SIGNAL_ROUND_ENDED;
    if (fight_winner >= 0) {
        next->fight_result = REK_G1_FIGHT_WON_BY_ROUNDS;
        next->fight_winner_index = fight_winner;
        next->phase = REK_G1_FIGHT_OVER;
        next->transition_remaining_seconds =
            REK_G1_FIGHT_CONFIG_F84F1874.fight_over_seconds;
        result->signals |= REK_G1_FIGHT_SIGNAL_FIGHT_ENDED;
    } else {
        next->phase = REK_G1_FIGHT_BETWEEN_ROUNDS;
        next->transition_remaining_seconds =
            REK_G1_FIGHT_CONFIG_F84F1874.between_round_seconds;
    }
    return REK_G1_FIGHT_OK;
}

static REK_G1_FN RekG1FightStatus resolve_count_expiry(
    const RekG1FightResolveInput* input,
    RekG1FightStepResult* result
) {
    RekG1FightState* next = &result->next_state;
    const uint8_t active0 = next->count_active[0];
    const uint8_t active1 = next->count_active[1];
    const int is_double = active0 && active1;
    clear_referee_state(next);

    if (is_double) {
        if (!add_score(result, 0u,
                REK_G1_FIGHT_CONFIG_F84F1874.ko_points)
                || !add_score(result, 1u,
                    REK_G1_FIGHT_CONFIG_F84F1874.ko_points)) {
            return REK_G1_FIGHT_OVERFLOW;
        }
        result->referee_calls |= REK_G1_REFEREE_DOUBLE_KNOCKOUT;
        if (input->fighter_can_get_up[0]
                || input->fighter_can_get_up[1]) {
            next->knockout_occurred = 1u;
            next->round_result = REK_G1_ROUND_TIE;
            next->round_winner_index = -1;
            return end_round(result);
        }
        if (next->time_remaining_seconds > 0.0f) {
            result->signals |= REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN;
            return REK_G1_FIGHT_OK;
        }
        return end_round(result);
    }

    const uint32_t fallen = active0 ? 0u : 1u;
    const uint32_t winner = 1u - fallen;
    result->subject_fighter_index = (int32_t)fallen;
    if (!add_score(result, winner,
            REK_G1_FIGHT_CONFIG_F84F1874.ko_points)) {
        return REK_G1_FIGHT_OVERFLOW;
    }
    result->referee_calls |= REK_G1_REFEREE_KNOCKOUT;
    if (input->fighter_can_get_up[fallen]) {
        next->knockout_occurred = 1u;
        next->round_result = REK_G1_ROUND_WON_BY_KO;
        next->round_winner_index = (int32_t)winner;
        if (!add_round_win(result, winner)) {
            return REK_G1_FIGHT_OVERFLOW;
        }
        return end_round(result);
    }
    if (next->time_remaining_seconds > 0.0f) {
        result->signals |= REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN;
        return REK_G1_FIGHT_OK;
    }
    return end_round(result);
}

static REK_G1_FN int valid_fighter_resolution(
    const uint8_t fighter_is_fallen[2],
    const uint8_t fighter_is_recovering[2],
    const uint8_t fighter_can_get_up[2]
) {
    if (fighter_is_fallen == NULL || fighter_is_recovering == NULL
            || fighter_can_get_up == NULL) {
        return 0;
    }
    for (uint32_t fighter = 0u; fighter < 2u; fighter += 1u) {
        if (!valid_flag(fighter_is_fallen[fighter])
                || !valid_flag(fighter_is_recovering[fighter])
                || !valid_flag(fighter_can_get_up[fighter])) {
            return 0;
        }
    }
    return 1;
}

REK_G1_FN RekG1FightStatus rek_g1_fight_begin_active_step(
    const RekG1FightState* state,
    const RekG1FightAdvanceInput* input,
    RekG1FightStepResult* result
) {
    if (state == NULL || input == NULL || result == NULL) {
        return REK_G1_FIGHT_NULL_ARGUMENT;
    }
    RekG1FightStatus status = validate_state(state);
    if (status != REK_G1_FIGHT_OK) {
        return status;
    }
    if (state->phase != REK_G1_FIGHT_ROUND_ACTIVE) {
        return REK_G1_FIGHT_WRONG_PHASE;
    }
    if (!isfinite(input->delta_seconds)
            || !isfinite(input->time_remaining_seconds)) {
        return REK_G1_FIGHT_NON_FINITE;
    }
    if (input->delta_seconds <= 0.0f
            || input->time_remaining_seconds < 0.0f
            || input->time_remaining_seconds > state->time_remaining_seconds) {
        return REK_G1_FIGHT_INPUT_INVALID;
    }
    if (!valid_fighter_resolution(
            input->fighter_is_fallen,
            input->fighter_is_recovering,
            input->fighter_can_get_up)) {
        return REK_G1_FIGHT_INPUT_INVALID;
    }

    initialize_result(state, result);
    result->next_state.time_remaining_seconds =
        input->time_remaining_seconds;
    status = advance_last_strikes(&result->next_state,
        input->delta_seconds);
    if (status != REK_G1_FIGHT_OK) {
        return status;
    }

    for (uint32_t fighter = 0u; fighter < 2u; fighter += 1u) {
        if (!result->next_state.count_active[fighter]
                || input->fighter_is_fallen[fighter]
                || input->fighter_is_recovering[fighter]) {
            continue;
        }
        result->next_state.count_active[fighter] = 0u;
        result->signals |= REK_G1_FIGHT_SIGNAL_COUNT_CLEARED;
        result->referee_calls |= REK_G1_REFEREE_BEAT_COUNT;
        if (result->next_state.count_is_slip[fighter]) {
            if (!add_score(result, 1u - fighter,
                    REK_G1_FIGHT_CONFIG_F84F1874
                        .slip_points_to_opponent)) {
                return REK_G1_FIGHT_OVERFLOW;
            }
        }
        result->next_state.count_is_slip[fighter] = 0u;
    }

    if (!any_count_active(&result->next_state)) {
        result->next_state.count_elapsed_seconds = 0.0f;
        result->next_state.count_duration_seconds = 0.0f;
        return REK_G1_FIGHT_OK;
    }

    const float elapsed = result->next_state.count_elapsed_seconds
        + input->delta_seconds;
    if (!isfinite(elapsed)) {
        return REK_G1_FIGHT_OVERFLOW;
    }
    if (elapsed >= result->next_state.count_duration_seconds) {
        result->next_state.count_elapsed_seconds =
            result->next_state.count_duration_seconds;
        return REK_G1_FIGHT_OK;
    }
    result->next_state.count_elapsed_seconds = elapsed;
    return REK_G1_FIGHT_OK;
}

REK_G1_FN RekG1FightStatus rek_g1_fight_resolve_active_step(
    const RekG1FightState* state,
    const RekG1FightResolveInput* input,
    RekG1FightStepResult* result
) {
    if (state == NULL || input == NULL || result == NULL) {
        return REK_G1_FIGHT_NULL_ARGUMENT;
    }
    const RekG1FightStatus status = validate_state(state);
    if (status != REK_G1_FIGHT_OK) {
        return status;
    }
    if (state->phase != REK_G1_FIGHT_ROUND_ACTIVE) {
        return REK_G1_FIGHT_WRONG_PHASE;
    }
    if (!valid_fighter_resolution(
            input->fighter_is_fallen,
            input->fighter_is_recovering,
            input->fighter_can_get_up)) {
        return REK_G1_FIGHT_INPUT_INVALID;
    }
    initialize_result(state, result);
    if (any_count_active(&result->next_state)) {
        if (result->next_state.count_elapsed_seconds
                >= result->next_state.count_duration_seconds) {
            return resolve_count_expiry(input, result);
        }
        return REK_G1_FIGHT_OK;
    }
    if (result->next_state.time_remaining_seconds <= 0.0f) {
        return end_round(result);
    }
    return REK_G1_FIGHT_OK;
}

static REK_G1_FN int merge_step_results(
    const RekG1FightStepResult* first,
    RekG1FightStepResult* second
) {
    if (first == NULL || second == NULL) return 0;
    for (uint32_t fighter = 0u; fighter < 2u; fighter += 1u) {
        if ((first->score_delta[fighter] > 0
                    && second->score_delta[fighter]
                        > INT32_MAX - first->score_delta[fighter])
                || (first->rounds_won_delta[fighter] > 0
                    && second->rounds_won_delta[fighter]
                        > INT32_MAX - first->rounds_won_delta[fighter])) {
            return 0;
        }
        second->score_delta[fighter] += first->score_delta[fighter];
        second->rounds_won_delta[fighter] +=
            first->rounds_won_delta[fighter];
    }
    second->signals |= first->signals;
    second->referee_calls |= first->referee_calls;
    if (second->subject_fighter_index < 0
            && first->subject_fighter_index >= 0) {
        second->subject_fighter_index = first->subject_fighter_index;
        second->fall_classification = first->fall_classification;
    }
    return 1;
}

REK_G1_FN RekG1FightStatus rek_g1_fight_advance_active(
    const RekG1FightState* state,
    const RekG1FightAdvanceInput* input,
    RekG1FightStepResult* result
) {
    if (state == NULL || input == NULL || result == NULL) {
        return REK_G1_FIGHT_NULL_ARGUMENT;
    }
    RekG1FightStepResult begun = REK_G1_ZERO_INIT;
    RekG1FightStatus status = rek_g1_fight_begin_active_step(
        state, input, &begun);
    if (status != REK_G1_FIGHT_OK) return status;
    const RekG1FightResolveInput resolve_input = {
        .fighter_is_fallen = {
            input->fighter_is_fallen[0], input->fighter_is_fallen[1]},
        .fighter_is_recovering = {
            input->fighter_is_recovering[0],
            input->fighter_is_recovering[1]},
        .fighter_can_get_up = {
            input->fighter_can_get_up[0], input->fighter_can_get_up[1]},
    };
    status = rek_g1_fight_resolve_active_step(
        &begun.next_state, &resolve_input, result);
    if (status != REK_G1_FIGHT_OK) return status;
    return merge_step_results(&begun, result)
        ? REK_G1_FIGHT_OK : REK_G1_FIGHT_OVERFLOW;
}

REK_G1_FN RekG1FightStatus rek_g1_fight_advance_transition(
    const RekG1FightState* state,
    float delta_seconds,
    RekG1FightStepResult* result
) {
    if (state == NULL || result == NULL) {
        return REK_G1_FIGHT_NULL_ARGUMENT;
    }
    RekG1FightStatus status = validate_state(state);
    if (status != REK_G1_FIGHT_OK) {
        return status;
    }
    if (!isfinite(delta_seconds)) {
        return REK_G1_FIGHT_NON_FINITE;
    }
    if (delta_seconds <= 0.0f) {
        return REK_G1_FIGHT_INPUT_INVALID;
    }
    if (state->phase != REK_G1_FIGHT_ROUND_COUNTDOWN
            && state->phase != REK_G1_FIGHT_BETWEEN_ROUNDS
            && state->phase != REK_G1_FIGHT_OVER) {
        return REK_G1_FIGHT_WRONG_PHASE;
    }

    initialize_result(state, result);
    status = advance_last_strikes(&result->next_state, delta_seconds);
    if (status != REK_G1_FIGHT_OK) {
        return status;
    }
    if (state->phase == REK_G1_FIGHT_ROUND_COUNTDOWN) {
        return REK_G1_FIGHT_OK;
    }

    if (delta_seconds < state->transition_remaining_seconds) {
        result->next_state.transition_remaining_seconds -= delta_seconds;
        return REK_G1_FIGHT_OK;
    }
    result->next_state.transition_remaining_seconds = 0.0f;
    if (state->phase == REK_G1_FIGHT_OVER) {
        result->next_state.phase = REK_G1_FIGHT_IDLE;
        result->signals |= REK_G1_FIGHT_SIGNAL_FIGHT_EXITED;
        return REK_G1_FIGHT_OK;
    }
    if (state->current_round_number == UINT32_MAX) {
        return REK_G1_FIGHT_OVERFLOW;
    }
    const uint8_t is_redo = state->round_result == REK_G1_ROUND_TIE;
    prepare_round(result, state->current_round_number + 1u, is_redo);
    return REK_G1_FIGHT_OK;
}
