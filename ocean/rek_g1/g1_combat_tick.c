#include "g1_combat_tick.h"

#include <limits.h>
#include <math.h>
#include <string.h>

static int binary_flag(uint8_t value) {
    return value == 0u || value == 1u;
}

static int valid_phase(RekG1FallPhase phase) {
    return phase == REK_G1_FALL_UPRIGHT
        || phase == REK_G1_FALL_FALLING
        || phase == REK_G1_FALL_FALLEN;
}

static int initialized_state(const RekG1CombatArenaState* state) {
    return state != NULL && state->initialized == 1u;
}

const char* rek_g1_combat_tick_status_string(RekG1CombatTickStatus status) {
    switch (status) {
        case REK_G1_COMBAT_TICK_OK: return "ok";
        case REK_G1_COMBAT_TICK_NULL_ARGUMENT: return "null argument";
        case REK_G1_COMBAT_TICK_NOT_READY: return "not ready";
        case REK_G1_COMBAT_TICK_INPUT_INVALID: return "invalid input";
        case REK_G1_COMBAT_TICK_HIT_REJECTED: return "hit rejected";
        case REK_G1_COMBAT_TICK_FIGHT_REJECTED: return "fight rejected";
        case REK_G1_COMBAT_TICK_OVERFLOW: return "accumulator overflow";
        default: return "unknown combat tick status";
    }
}

RekG1CombatTickStatus rek_g1_combat_arena_init(
        RekG1CombatArenaState* state) {
    if (state == NULL) return REK_G1_COMBAT_TICK_NULL_ARGUMENT;
    RekG1CombatArenaState local;
    memset(&local, 0, sizeof(local));
    RekG1FightStepResult step = {0};
    if (rek_g1_fight_state_init(&local.fight) != REK_G1_FIGHT_OK
            || rek_g1_fight_prepare_first_round(
                &local.fight, &step) != REK_G1_FIGHT_OK) {
        return REK_G1_COMBAT_TICK_FIGHT_REJECTED;
    }
    local.fight = step.next_state;
    if (rek_g1_fight_activate_round(
            &local.fight, &step) != REK_G1_FIGHT_OK) {
        return REK_G1_COMBAT_TICK_FIGHT_REJECTED;
    }
    local.fight = step.next_state;
    rek_g1_hit_detector_reset(&local.hit_detector);
    local.initialized = 1u;
    *state = local;
    return REK_G1_COMBAT_TICK_OK;
}

static int input_valid(
        const RekG1CombatArenaState* state,
        const RekG1CombatSubstepInput* input) {
    if (!initialized_state(state) || input == NULL
            || !isfinite(input->delta_seconds)
            || !isfinite(input->time_remaining_seconds)
            || input->delta_seconds <= 0.0f
            || input->time_remaining_seconds < 0.0f
            || input->time_remaining_seconds
                > state->fight.time_remaining_seconds
            || (input->contact_count > 0u && input->contacts == NULL)) {
        return 0;
    }
    const uint32_t valid_fall_events = REK_G1_FALL_EVENT_FALLING_STARTED
        | REK_G1_FALL_EVENT_FALLING_CLEARED
        | REK_G1_FALL_EVENT_BECAME_FALLEN
        | REK_G1_FALL_EVENT_RESET_TIMEOUT_DUE;
    for (uint32_t fighter = 0u; fighter < 2u; fighter++) {
        if (!valid_phase(input->fall_phase[fighter])
                || (input->fall_events[fighter] & ~valid_fall_events) != 0u
                || !binary_flag(input->fighter_is_recovering[fighter])
                || !binary_flag(input->fighter_can_get_up[fighter])
                || !binary_flag(input->force_slip_estop[fighter])) {
            return 0;
        }
    }
    return 1;
}

static int add_positive_i32(int32_t* target, int32_t value) {
    if (target == NULL || value < 0 || *target > INT32_MAX - value) return 0;
    *target += value;
    return 1;
}

static int merge_fight_result(
        RekG1CombatSubstepResult* output,
        const RekG1FightStepResult* step) {
    if (output == NULL || step == NULL) return 0;
    for (uint32_t fighter = 0u; fighter < 2u; fighter++) {
        if (!add_positive_i32(
                &output->score_delta[fighter], step->score_delta[fighter])
                || !add_positive_i32(
                    &output->rounds_won_delta[fighter],
                    step->rounds_won_delta[fighter])) {
            return 0;
        }
    }
    output->signals |= step->signals;
    output->referee_calls |= step->referee_calls;
    output->next_state.fight = step->next_state;
    return 1;
}

static RekG1CombatTickStatus apply_fight_step(
        RekG1CombatSubstepResult* output,
        RekG1FightStatus status,
        const RekG1FightStepResult* step) {
    if (status != REK_G1_FIGHT_OK) {
        return REK_G1_COMBAT_TICK_FIGHT_REJECTED;
    }
    return merge_fight_result(output, step)
        ? REK_G1_COMBAT_TICK_OK : REK_G1_COMBAT_TICK_OVERFLOW;
}

RekG1CombatTickStatus rek_g1_combat_arena_substep(
        const RekG1CombatArenaState* state,
        const RekG1HitDetectorConfig* hit_config,
        const RekG1CombatSubstepInput* input,
        RekG1CombatSubstepResult* result) {
    if (state == NULL || hit_config == NULL || input == NULL
            || result == NULL) {
        return REK_G1_COMBAT_TICK_NULL_ARGUMENT;
    }
    if (!initialized_state(state)) return REK_G1_COMBAT_TICK_NOT_READY;
    if (!input_valid(state, input)) return REK_G1_COMBAT_TICK_INPUT_INVALID;

    RekG1CombatSubstepResult local;
    memset(&local, 0, sizeof(local));
    local.next_state = *state;
    const uint8_t round_active =
        local.next_state.fight.phase == REK_G1_FIGHT_ROUND_ACTIVE;
    RekG1FightAdvanceInput begin_input = {
        .delta_seconds = input->delta_seconds,
        .time_remaining_seconds = input->time_remaining_seconds,
        .fighter_is_fallen = {
            input->fall_phase[0] == REK_G1_FALL_FALLEN,
            input->fall_phase[1] == REK_G1_FALL_FALLEN,
        },
        .fighter_is_recovering = {
            input->fighter_is_recovering[0],
            input->fighter_is_recovering[1],
        },
        .fighter_can_get_up = {
            input->fighter_can_get_up[0],
            input->fighter_can_get_up[1],
        },
    };
    RekG1FightStepResult fight_step = {0};
    RekG1CombatTickStatus status = REK_G1_COMBAT_TICK_OK;
    if (round_active) {
        status = apply_fight_step(
            &local,
            rek_g1_fight_begin_active_step(
                &local.next_state.fight, &begin_input, &fight_step),
            &fight_step);
        if (status != REK_G1_COMBAT_TICK_OK) return status;
    }
    for (size_t index = 0u; index < input->contact_count; index++) {
        const RekG1HitContact* contact = &input->contacts[index];
        if (contact->round_active != round_active) {
            return REK_G1_COMBAT_TICK_INPUT_INVALID;
        }
        RekG1HitResult hit = {0};
        if (!rek_g1_hit_detector_process(
                &local.next_state.hit_detector,
                hit_config,
                contact,
                &hit)) {
            return REK_G1_COMBAT_TICK_HIT_REJECTED;
        }
        if (hit.attribution_accepted) {
            if (local.attributed_contact_count == UINT32_MAX) {
                return REK_G1_COMBAT_TICK_OVERFLOW;
            }
            local.attributed_contact_count++;
        }
        if (hit.score_accepted) {
            if (local.scored_contact_count == UINT32_MAX) {
                return REK_G1_COMBAT_TICK_OVERFLOW;
            }
            local.scored_contact_count++;
        }
        if (!hit.attribution_accepted && !hit.score_accepted) continue;
        const RekG1StrikeEvent strike = {
            .attacker_fighter_index = contact->striker_fighter,
            .victim_fighter_index = contact->target_fighter,
            .hand_side = contact->striker_side,
            .striker_part = contact->striker_part,
            .zone = contact->target_zone,
            .relative_speed = contact->relative_speed_mps,
            .attribution_passed = hit.attribution_accepted,
            .score_accepted = hit.score_accepted,
        };
        status = apply_fight_step(
            &local,
            rek_g1_fight_record_strike(
                &local.next_state.fight, &strike, &fight_step),
            &fight_step);
        if (status != REK_G1_COMBAT_TICK_OK) return status;
    }

    if (!round_active) {
        if (input->fall_events[0] != REK_G1_FALL_EVENT_NONE
                || input->fall_events[1] != REK_G1_FALL_EVENT_NONE) {
            return REK_G1_COMBAT_TICK_INPUT_INVALID;
        }
        *result = local;
        return REK_G1_COMBAT_TICK_OK;
    }

    for (uint32_t fighter = 0u; fighter < 2u; fighter++) {
        const uint32_t events = input->fall_events[fighter];
        if ((events & REK_G1_FALL_EVENT_FALLING_STARTED) != 0u) {
            status = apply_fight_step(
                &local,
                rek_g1_fight_on_falling(
                    &local.next_state.fight,
                    fighter,
                    input->force_slip_estop[fighter],
                    &fight_step),
                &fight_step);
            if (status != REK_G1_COMBAT_TICK_OK) return status;
        }
        if ((events & REK_G1_FALL_EVENT_BECAME_FALLEN) != 0u) {
            status = apply_fight_step(
                &local,
                rek_g1_fight_on_fallen(
                    &local.next_state.fight,
                    fighter,
                    input->fighter_can_get_up,
                    &fight_step),
                &fight_step);
            if (status != REK_G1_COMBAT_TICK_OK) return status;
        }
        if ((events & REK_G1_FALL_EVENT_RESET_TIMEOUT_DUE) != 0u) {
            status = apply_fight_step(
                &local,
                rek_g1_fight_on_reset_due(
                    &local.next_state.fight, fighter, &fight_step),
                &fight_step);
            if (status != REK_G1_COMBAT_TICK_OK) return status;
        }
    }

    const RekG1FightResolveInput resolve_input = {
        .fighter_is_fallen = {
            input->fall_phase[0] == REK_G1_FALL_FALLEN,
            input->fall_phase[1] == REK_G1_FALL_FALLEN,
        },
        .fighter_is_recovering = {
            input->fighter_is_recovering[0],
            input->fighter_is_recovering[1],
        },
        .fighter_can_get_up = {
            input->fighter_can_get_up[0],
            input->fighter_can_get_up[1],
        },
    };
    status = apply_fight_step(
        &local,
        rek_g1_fight_resolve_active_step(
            &local.next_state.fight, &resolve_input, &fight_step),
        &fight_step);
    if (status != REK_G1_COMBAT_TICK_OK) return status;
    *result = local;
    return REK_G1_COMBAT_TICK_OK;
}

RekG1CombatTickStatus rek_g1_combat_arena_apply_spawn_reset(
        const RekG1CombatArenaState* state,
        RekG1CombatArenaState* next_state) {
    if (state == NULL || next_state == NULL) {
        return REK_G1_COMBAT_TICK_NULL_ARGUMENT;
    }
    if (!initialized_state(state)) return REK_G1_COMBAT_TICK_NOT_READY;
    RekG1CombatArenaState local = *state;
    RekG1FightStepResult fight_step = {0};
    if (rek_g1_fight_apply_spawn_reset(
            &local.fight, &fight_step) != REK_G1_FIGHT_OK) {
        return REK_G1_COMBAT_TICK_FIGHT_REJECTED;
    }
    local.fight = fight_step.next_state;
    rek_g1_hit_detector_reset(&local.hit_detector);
    *next_state = local;
    return REK_G1_COMBAT_TICK_OK;
}
