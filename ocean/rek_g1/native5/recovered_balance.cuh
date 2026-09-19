#pragma once

#if !defined(REK_G1_CUDA_DEVICE)
extern "C" {
#endif
#include "../g1_combat_tick.h"
#if !defined(REK_G1_CUDA_DEVICE)
}
#endif
#include <math.h>

// No dynamics or recovered rules are approximated here. Compile the original
// g1_{fall_state,fight_state,combat_tick,hit_detector}.c for the same target.
// Define REK_G1_CUDA_DEVICE before any recovered header in device translation
// units; g1_cuda_device.cu already provides the corresponding source adapter.
namespace rek5_balance {

enum Field : uint32_t {
    Tracking = 1u << 0, Tilt = 1u << 1, PelvisHeightRatio = 1u << 2,
    FeetOffFloor = 1u << 3, FootBodyContact = 1u << 4,
    NonfootBodyContacts = 1u << 5, CanGetUp = 1u << 6,
    DetectorTickEnabled = 1u << 7, IsRecovering = 1u << 8,
    ForceSlipEstop = 1u << 9,
    AllFighterFields = (1u << 10) - 1u,
};
enum ClockField : uint32_t {
    FixedDelta = 1u << 0, RoundTimeRemaining = 1u << 1,
    CompleteContactStream = 1u << 2,
    AllClockFields = (1u << 3) - 1u,
};
enum Status {
    Ok, NullArgument, MissingInput, InvalidInput, NotReady, WrongPhase,
    FallRejected, CombatRejected, PhysicalResetPending, ResetNotConfirmed,
};

struct Provenance {
    uint32_t measured;
    uint32_t modeled;
};
struct FighterInput {
    Provenance provenance;
    RekG1FallSample dynamics;
    // Caller resolves native isResetting/fallDetectionSuppressed/
    // motorShutdownHold early gates from measured or modeled state.
    uint8_t detector_tick_enabled;
    uint8_t is_recovering;
    uint8_t force_slip_estop;
};
struct Input {
    Provenance provenance;
    float delta_seconds;
    float time_remaining_seconds;
    FighterInput fighters[2];
    // This bit certifies completeness and validity of ALL fields in contacts
    // and their nested strike intents, including a known empty stream. It
    // cannot be set just because a compact renderer reports no contacts.
    const RekG1HitContact* contacts;
    size_t contact_count;
};
struct State {
    RekG1FallState fall[2];
    RekG1CombatArenaState combat;
    uint8_t physical_reset_pending;
};
struct Result {
    State next_state;
    uint32_t fall_events[2];
    RekG1CombatSubstepResult combat;
};

static REK_G1_FN inline Status provenance_status(
        const Provenance& value, uint32_t required) {
    if ((value.measured & value.modeled) != 0u
            || ((value.measured | value.modeled) & ~required) != 0u)
        return InvalidInput;
    return (value.measured | value.modeled) == required ? Ok : MissingInput;
}

// Same immediate-active episode boundary as rek_g1_combat_arena_init. This
// does not infer the unknown external countdown or a physical spawn/reset.
static REK_G1_FN inline Status init_active(State* state) {
    if (!state) return NullArgument;
    State local{};
    if (rek_g1_combat_arena_init(&local.combat) != REK_G1_COMBAT_TICK_OK)
        return CombatRejected;
    for (unsigned i = 0; i < 2; ++i)
        if (rek_g1_fall_state_init(&REK_G1_FALL_CONFIG_F84F1874,
                &local.fall[i]) != REK_G1_FALL_OK) return FallRejected;
    *state = local;
    return Ok;
}

// Atomic active-round step. Unknown inputs, mismatched clocks and native
// errors leave both input state and output result unchanged. The explicit
// modeled bits identify simulation assumptions; they confer no game parity.
static REK_G1_FN inline Status step(
        const State* state, const Input* input, Result* result) {
    if (!state || !input || !result) return NullArgument;
    if (state->combat.initialized != 1u) return NotReady;
    if (state->physical_reset_pending > 1u) return InvalidInput;
    if (state->physical_reset_pending) return PhysicalResetPending;
    if (state->combat.fight.phase != REK_G1_FIGHT_ROUND_ACTIVE)
        return WrongPhase;
    Status status = provenance_status(input->provenance, AllClockFields);
    if (status != Ok) return status;
    if (!isfinite(input->delta_seconds) || input->delta_seconds <= 0.0f
            || !isfinite(input->time_remaining_seconds)
            || input->time_remaining_seconds < 0.0f
            || input->time_remaining_seconds > state->combat.fight.time_remaining_seconds
            || (input->contact_count && !input->contacts)) return InvalidInput;

    Result local{};
    local.next_state = *state;
    RekG1CombatSubstepInput combat_input{};
    combat_input.delta_seconds = input->delta_seconds;
    combat_input.time_remaining_seconds = input->time_remaining_seconds;
    combat_input.contacts = input->contacts;
    combat_input.contact_count = input->contact_count;
    for (unsigned i = 0; i < 2; ++i) {
        const FighterInput& fighter = input->fighters[i];
        status = provenance_status(fighter.provenance, AllFighterFields);
        if (status != Ok) return status;
        if (fighter.detector_tick_enabled > 1u || fighter.is_recovering > 1u
                || fighter.force_slip_estop > 1u
                || fighter.dynamics.fixed_delta_seconds != input->delta_seconds)
            return InvalidInput;
        // Even a gated tick validates the complete dynamics packet through
        // the native validator; its candidate state is then discarded.
        RekG1FallStepResult fall{};
        if (rek_g1_fall_state_step(&REK_G1_FALL_CONFIG_F84F1874,
                &state->fall[i], &fighter.dynamics, &fall) != REK_G1_FALL_OK)
            return FallRejected;
        if (fighter.detector_tick_enabled) {
            local.next_state.fall[i] = fall.next_state;
            local.fall_events[i] = fall.events;
        }
        combat_input.fall_phase[i] = local.next_state.fall[i].phase;
        combat_input.fall_events[i] = local.fall_events[i];
        combat_input.fighter_is_recovering[i] = fighter.is_recovering;
        combat_input.fighter_can_get_up[i] = fighter.dynamics.can_get_up;
        combat_input.force_slip_estop[i] = fighter.force_slip_estop;
    }
    const RekG1HitDetectorConfig config = rek_g1_current_build_hit_detector_config();
    if (rek_g1_combat_arena_substep(&state->combat, &config, &combat_input,
            &local.combat) != REK_G1_COMBAT_TICK_OK) return CombatRejected;
    local.next_state.combat = local.combat.next_state;
    local.next_state.physical_reset_pending =
        (local.combat.signals & REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN) != 0u;
    *result = local;
    return Ok;
}

// Caller first completes physical ResetBothToSpawn, then acknowledges it.
// The result retains scores/falls/time and resets both detector states with
// native 2 s grace, plus native contact and transient attribution history.
static REK_G1_FN inline Status acknowledge_spawn_reset(const State* state,
        uint8_t physical_reset_confirmed, State* result) {
    if (!state || !result) return NullArgument;
    if (physical_reset_confirmed != 1u) return ResetNotConfirmed;
    if (state->physical_reset_pending != 1u) return InvalidInput;
    State local = *state;
    if (rek_g1_combat_arena_apply_spawn_reset(&state->combat,
            &local.combat) != REK_G1_COMBAT_TICK_OK) return CombatRejected;
    for (unsigned i = 0; i < 2; ++i)
        if (rek_g1_fall_state_apply_fight_spawn_reset(&REK_G1_FALL_CONFIG_F84F1874,
                &state->fall[i], &local.fall[i]) != REK_G1_FALL_OK) return FallRejected;
    local.physical_reset_pending = 0u;
    *result = local;
    return Ok;
}

} // namespace rek5_balance
