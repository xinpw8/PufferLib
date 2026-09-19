#pragma once
#include "recovered_balance.cuh"
#include <string.h>

namespace rek5_balance_test {
struct Outcome {
    uint64_t digest;
    uint32_t checks;
    int failure_line;
};

static REK_G1_FN inline void mix(uint64_t& hash, uint32_t value) {
    hash = (hash ^ value) * UINT64_C(1099511628211);
}
static REK_G1_FN inline void mix_float(uint64_t& hash, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    mix(hash, bits);
}
static REK_G1_FN inline uint64_t state_digest(const rek5_balance::State& state) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned i = 0; i < 2; ++i) {
        const auto& f = state.fall[i];
        mix(hash, f.phase); mix_float(hash, f.fallen_hold_seconds);
        mix_float(hash, f.fallen_elapsed_seconds); mix_float(hash, f.fallen_timer_seconds);
        mix_float(hash, f.reset_grace_remaining_seconds); mix(hash, f.recovery_armed);
        const auto& h = state.combat.hit_detector;
        for (unsigned slot = 0; slot < REK_G1_HIT_STRIKER_BODY_SLOTS; ++slot) {
            mix_float(hash, h.last_score_time_seconds[i][slot]);
            mix(hash, h.cooldown_seen[i][slot]);
        }
        mix(hash, h.scored_move_id[i]); mix(hash, h.scored_apex_mask[i]);
        mix(hash, h.scored_move_seen[i]);
    }
    const auto& f = state.combat.fight;
    mix(hash, f.phase); mix(hash, f.current_round_number); mix(hash, f.current_round_is_redo);
    mix_float(hash, f.round_duration_seconds); mix_float(hash, f.time_remaining_seconds);
    mix(hash, f.round_result); mix(hash, f.round_winner_index); mix(hash, f.knockout_occurred);
    mix(hash, f.fight_result); mix(hash, f.fight_winner_index);
    for (unsigned i = 0; i < 2; ++i) {
        mix(hash, f.clean_hits[i]); mix(hash, f.falls[i]); mix(hash, f.rounds_won[i]);
        mix(hash, f.last_struck_valid[i]); mix_float(hash, f.last_struck_age_seconds[i]);
        mix_float(hash, f.last_struck_speed[i]); mix(hash, f.fall_classification[i]);
        mix(hash, f.fall_forced_by_estop[i]); mix(hash, f.count_active[i]); mix(hash, f.count_is_slip[i]);
    }
    mix_float(hash, f.count_elapsed_seconds); mix_float(hash, f.count_duration_seconds);
    mix_float(hash, f.transition_remaining_seconds);
    mix(hash, state.combat.initialized); mix(hash, state.physical_reset_pending);
    return hash;
}

static REK_G1_FN inline rek5_balance::Input neutral_input(
        const rek5_balance::State& state, float dt = 0.125f) {
    rek5_balance::Input input{};
    input.provenance.modeled = rek5_balance::AllClockFields;
    input.delta_seconds = dt;
    input.time_remaining_seconds = fmaxf(0.0f, state.combat.fight.time_remaining_seconds - dt);
    for (auto& fighter : input.fighters) {
        fighter.provenance.modeled = rek5_balance::AllFighterFields;
        fighter.dynamics.tracking_active = 1;
        fighter.dynamics.pelvis_height_ratio = 1.0f;
        fighter.dynamics.has_foot_body_contact = 1;
        fighter.dynamics.fixed_delta_seconds = dt;
        fighter.detector_tick_enabled = 1;
    }
    return input;
}

// Independent composition of the pre-existing native CPU oracle. This does
// not call the adapter for classification, events, referee or reset logic.
static REK_G1_FN inline int direct_step(const rek5_balance::State& state,
        const rek5_balance::Input& input, rek5_balance::Result& result) {
    result = {};
    result.next_state = state;
    RekG1CombatSubstepInput facts{};
    facts.delta_seconds = input.delta_seconds;
    facts.time_remaining_seconds = input.time_remaining_seconds;
    facts.contacts = input.contacts;
    facts.contact_count = input.contact_count;
    for (unsigned i = 0; i < 2; ++i) {
        RekG1FallStepResult fall{};
        if (input.fighters[i].detector_tick_enabled) {
            if (rek_g1_fall_state_step(&REK_G1_FALL_CONFIG_F84F1874, &state.fall[i],
                    &input.fighters[i].dynamics, &fall) != REK_G1_FALL_OK) return 0;
            result.next_state.fall[i] = fall.next_state;
            result.fall_events[i] = fall.events;
        }
        facts.fall_phase[i] = result.next_state.fall[i].phase;
        facts.fall_events[i] = result.fall_events[i];
        facts.fighter_can_get_up[i] = input.fighters[i].dynamics.can_get_up;
        facts.fighter_is_recovering[i] = input.fighters[i].is_recovering;
        facts.force_slip_estop[i] = input.fighters[i].force_slip_estop;
    }
    const auto config = rek_g1_current_build_hit_detector_config();
    if (rek_g1_combat_arena_substep(&state.combat, &config, &facts, &result.combat)
            != REK_G1_COMBAT_TICK_OK) return 0;
    result.next_state.combat = result.combat.next_state;
    result.next_state.physical_reset_pending =
        (result.combat.signals & REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN) != 0;
    return 1;
}

#define BALANCE_CHECK(condition) do { ++out.checks; if (!(condition)) { \
    out.failure_line = __LINE__; return out; } } while (0)

// Sixteen deterministic dynamics fixtures. The values are synthetic probes
// of recovered rules, not fitted live-game trajectories.
static REK_G1_FN inline Outcome run_case(unsigned case_id) {
    using namespace rek5_balance;
    Outcome out{UINT64_C(14695981039346656037), 0, 0};
    const unsigned scenario = case_id % 16;
    State state{};
    BALANCE_CHECK(init_active(&state) == Ok);
    State reference = state;
    const bool double_fall = scenario == 2 || scenario == 4;
    const bool recovery = scenario == 3 || scenario == 4;
    const bool expect_fallen = scenario < 8 || scenario == 14 || scenario == 15;
    const bool slow = scenario == 6 || scenario == 7 || scenario == 14;
    const unsigned expected_fall_tick = (slow ? 4u : 2u) + (scenario == 5 ? 8u : 0u);
    unsigned first_fallen_tick = 999;
    bool resolved = false;
    // This is an attribution-only contact on a non-scoring wrist zone. It
    // must still classify a same-tick fall as a knockdown.
    RekG1HitContact contact{};
    contact.striker_body_position_world[2] = 1;
    contact.target_body_position_world[0] = 1;
    contact.target_body_position_world[2] = 1;
    contact.striker_body_linear_velocity_world[0] = 2.5f;
    contact.relative_speed_mps = 2.5f;
    contact.striker_part = REK_G1_BODY_PART_FOOT;
    contact.striker_side = REK_G1_HAND_LEFT;
    contact.target_zone = REK_G1_BODY_ZONE_LEFT_WRIST;
    contact.striker_fighter = 0; contact.target_fighter = 1;
    contact.striker_body_slot = 2;
    contact.is_enter = contact.round_active = contact.striker_upright = 1;
    contact.target_upright = contact.target_standing = 1;

    for (unsigned tick = 0; tick < (expect_fallen ? 190u : 16u); ++tick) {
        Input input = neutral_input(state);
        for (unsigned fighter = double_fall ? 0u : 1u; fighter < 2; ++fighter) {
            auto& sample = input.fighters[fighter].dynamics;
            sample.tilt_degrees = 75;
            sample.pelvis_height_ratio = 0.3f;
            sample.both_feet_off_floor = 1;
            sample.has_foot_body_contact = scenario == 14;
            sample.distinct_nonfoot_body_contact_count = scenario == 6 ? 1 : 3;
            sample.can_get_up = recovery && fighter == 1;
            if (scenario == 5 && tick < 8) input.fighters[fighter].detector_tick_enabled = 0;
            if (scenario == 7) {
                sample.tracking_active = 0;
                sample.pelvis_height_ratio = 1;
                sample.distinct_nonfoot_body_contact_count = 0;
            }
            if (scenario == 8) sample.distinct_nonfoot_body_contact_count = 0;
            if (scenario == 9) {
                sample.tilt_degrees = REK_G1_FALL_CONFIG_F84F1874.falling_tilt_degrees;
                sample.pelvis_height_ratio = 1;
                sample.both_feet_off_floor = 0;
            }
            if (scenario == 10) {
                sample.tilt_degrees = tick == 0 ? 43.0f : 0.0f;
                sample.pelvis_height_ratio = 1;
                sample.both_feet_off_floor = 0;
            }
            if (scenario == 11) sample.tilt_degrees = 0;
            if (scenario == 12) sample.tilt_degrees = REK_G1_FALL_CONFIG_F84F1874.fallen_tilt_degrees;
            if (scenario == 13) sample.pelvis_height_ratio = REK_G1_FALL_CONFIG_F84F1874.fallen_height_ratio;
            if (scenario == 15) input.fighters[fighter].force_slip_estop = 1;
        }
        if ((scenario == 1 || scenario == 15) && tick == 0) {
            contact.time_seconds = input.delta_seconds;
            input.contacts = &contact;
            input.contact_count = 1;
        }
        // Equivalent mixed measured/model provenance is accepted explicitly.
        if (case_id & 16) {
            input.fighters[1].provenance.measured = Tilt;
            input.fighters[1].provenance.modeled &= ~Tilt;
        }
        Result result{}, oracle{};
        BALANCE_CHECK(step(&state, &input, &result) == Ok);
        BALANCE_CHECK(direct_step(reference, input, oracle));
        BALANCE_CHECK(state_digest(result.next_state) == state_digest(oracle.next_state));
        BALANCE_CHECK(result.combat.signals == oracle.combat.signals);
        BALANCE_CHECK(result.combat.referee_calls == oracle.combat.referee_calls);
        BALANCE_CHECK(result.combat.attributed_contact_count == oracle.combat.attributed_contact_count);
        BALANCE_CHECK(result.combat.scored_contact_count == oracle.combat.scored_contact_count);
        for (unsigned i = 0; i < 2; ++i) {
            BALANCE_CHECK(result.fall_events[i] == oracle.fall_events[i]);
            BALANCE_CHECK(result.combat.score_delta[i] == oracle.combat.score_delta[i]);
            BALANCE_CHECK(result.combat.rounds_won_delta[i] == oracle.combat.rounds_won_delta[i]);
        }
        state = result.next_state;
        reference = oracle.next_state;
        const uint64_t digest = state_digest(state);
        mix(out.digest, uint32_t(digest)); mix(out.digest, uint32_t(digest >> 32));
        mix(out.digest, result.combat.referee_calls); mix(out.digest, result.combat.signals);
        if (result.fall_events[1] & REK_G1_FALL_EVENT_BECAME_FALLEN) {
            first_fallen_tick = tick;
            BALANCE_CHECK(expect_fallen);
            BALANCE_CHECK(state.combat.fight.count_elapsed_seconds == 0);
            BALANCE_CHECK(state.combat.fight.count_duration_seconds ==
                (recovery ? (double_fall ? 20.0f : 10.0f) : 3.0f));
            BALANCE_CHECK(result.combat.referee_calls & (scenario == 1
                ? REK_G1_REFEREE_KNOCKDOWN : scenario == 15
                ? REK_G1_REFEREE_SLIP_ESTOP : REK_G1_REFEREE_SLIP));
            BALANCE_CHECK(state.combat.fight.falls[1] == 1);
        }
        if (state.physical_reset_pending || (result.combat.signals & REK_G1_FIGHT_SIGNAL_ROUND_ENDED)) {
            BALANCE_CHECK(expect_fallen);
            BALANCE_CHECK(tick == first_fallen_tick + (recovery ? (double_fall ? 160u : 80u) : 24u));
            BALANCE_CHECK(state.combat.fight.clean_hits[0] == 5);
            BALANCE_CHECK(state.combat.fight.clean_hits[1] == (double_fall ? 5 : 0));
            BALANCE_CHECK(result.combat.referee_calls & (double_fall
                ? REK_G1_REFEREE_DOUBLE_KNOCKOUT : REK_G1_REFEREE_KNOCKOUT));
            BALANCE_CHECK(state.physical_reset_pending == !recovery);
            if (!recovery) {
                Result untouched = result;
                BALANCE_CHECK(step(&state, &input, &untouched) == PhysicalResetPending);
                State reset{};
                BALANCE_CHECK(acknowledge_spawn_reset(&state, 0, &reset) == ResetNotConfirmed);
                BALANCE_CHECK(acknowledge_spawn_reset(&state, 1, &reset) == Ok);
                State native_reset = state;
                BALANCE_CHECK(rek_g1_combat_arena_apply_spawn_reset(&state.combat,
                    &native_reset.combat) == REK_G1_COMBAT_TICK_OK);
                for (unsigned i = 0; i < 2; ++i) {
                    BALANCE_CHECK(rek_g1_fall_state_apply_fight_spawn_reset(&REK_G1_FALL_CONFIG_F84F1874,
                        &state.fall[i], &native_reset.fall[i]) == REK_G1_FALL_OK);
                    BALANCE_CHECK(reset.fall[i].phase == REK_G1_FALL_UPRIGHT);
                    BALANCE_CHECK(reset.fall[i].reset_grace_remaining_seconds == 2);
                    BALANCE_CHECK(reset.combat.fight.falls[i] == state.combat.fight.falls[i]);
                    BALANCE_CHECK(!reset.combat.fight.last_struck_valid[i]);
                }
                native_reset.physical_reset_pending = 0;
                BALANCE_CHECK(state_digest(reset) == state_digest(native_reset));
                BALANCE_CHECK(reset.combat.fight.time_remaining_seconds == state.combat.fight.time_remaining_seconds);
                // Exactly 16 ticks consume the 2 s gate; the 17th can fall.
                for (unsigned gate_tick = 0; gate_tick < 17; ++gate_tick) {
                    Input gated = neutral_input(reset);
                    gated.fighters[1].dynamics.tilt_degrees = 75;
                    BALANCE_CHECK(step(&reset, &gated, &result) == Ok);
                    BALANCE_CHECK(result.next_state.fall[1].phase ==
                        (gate_tick < 16 ? REK_G1_FALL_UPRIGHT : REK_G1_FALL_FALLING));
                    reset = result.next_state;
                }
            } else {
                BALANCE_CHECK(state.combat.fight.phase == REK_G1_FIGHT_BETWEEN_ROUNDS);
                BALANCE_CHECK(state.combat.fight.round_result ==
                    (double_fall ? REK_G1_ROUND_TIE : REK_G1_ROUND_WON_BY_KO));
            }
            resolved = true;
            break;
        }
    }
    BALANCE_CHECK(resolved == expect_fallen);
    BALANCE_CHECK(first_fallen_tick == (expect_fallen ? expected_fall_tick : 999u));
    if (scenario == 9 || scenario == 10) BALANCE_CHECK(state.fall[1].phase == REK_G1_FALL_UPRIGHT);
    return out;
}
#undef BALANCE_CHECK
} // namespace rek5_balance_test
