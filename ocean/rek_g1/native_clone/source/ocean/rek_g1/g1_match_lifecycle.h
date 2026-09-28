#pragma once

#include "g1_native_combat_cuda.h"

/* Host/device shared transition used by the additive CUDA entrypoint.
 * Bodies and controller histories are owned by the caller. The single reset
 * pulse occurs at ROUND_PREPARED, never merely because ROUND_ENDED occurred.
 * The 20 ms clock matches one caller controller tick. Timeline/countdown
 * completion is explicit; the clone may omit that presentation interval.
 */
static REK_G1_FN inline int32_t rek_g1_native_match_begin(
        RekG1CudaNativeCombatState* state,
        uint8_t countdown_complete,
        uint32_t* signals,
        uint8_t* physical_reset) {
    if (!state || !signals || !physical_reset || countdown_complete > 1)
        return REK_G1_CUDA_NATIVE_COMBAT_INPUT_INVALID;
    RekG1CudaNativeCombatState next = *state;
    const RekG1FightPhase phase = next.combat.fight.phase;
    uint32_t emitted = 0;
    uint8_t reset = 0;
    if (!next.combat.initialized)
        return REK_G1_CUDA_NATIVE_COMBAT_TICK_REJECTED;
    if (phase == REK_G1_FIGHT_BETWEEN_ROUNDS || phase == REK_G1_FIGHT_OVER
            || phase == REK_G1_FIGHT_ROUND_COUNTDOWN) {
        RekG1FightStepResult result = {};
        const RekG1FightStatus status = phase == REK_G1_FIGHT_ROUND_COUNTDOWN
                && countdown_complete
            ? rek_g1_fight_activate_round(&next.combat.fight, &result)
            : rek_g1_fight_advance_transition(&next.combat.fight, 0.02f, &result);
        if (status != REK_G1_FIGHT_OK)
            return REK_G1_CUDA_NATIVE_COMBAT_TICK_REJECTED;
        next.combat.fight = result.next_state;
        emitted = result.signals;
        if (emitted & REK_G1_FIGHT_SIGNAL_ROUND_PREPARED) {
            RekG1CombatArenaState reset_combat = {};
            RekG1FallState reset_fall[2] = {};
            if (rek_g1_combat_arena_apply_spawn_reset(&next.combat, &reset_combat)
                    != REK_G1_COMBAT_TICK_OK
                    || rek_g1_fall_state_apply_fight_spawn_reset(
                        &REK_G1_FALL_CONFIG_F84F1874, &next.fall[0], &reset_fall[0])
                        != REK_G1_FALL_OK
                    || rek_g1_fall_state_apply_fight_spawn_reset(
                        &REK_G1_FALL_CONFIG_F84F1874, &next.fall[1], &reset_fall[1])
                        != REK_G1_FALL_OK)
                return REK_G1_CUDA_NATIVE_COMBAT_RESET_REJECTED;
            next.combat = reset_combat;
            next.fall[0] = reset_fall[0];
            next.fall[1] = reset_fall[1];
            next.reset_pending = 0;
            next.reset_complete_not_before_seconds = 0.0f;
            reset = 1;
        }
    } else if (phase != REK_G1_FIGHT_ROUND_ACTIVE && phase != REK_G1_FIGHT_IDLE) {
        return REK_G1_CUDA_NATIVE_COMBAT_TICK_REJECTED;
    }
    // This is the legacy episode request, not a command to clear a match.
    next.pending_episode_reset = 0;
    *state = next;
    *signals = emitted;
    *physical_reset = reset;
    return REK_G1_CUDA_NATIVE_COMBAT_OK;
}
