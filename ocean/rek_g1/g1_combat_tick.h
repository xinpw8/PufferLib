#pragma once

#include <stddef.h>
#include <stdint.h>

#include "g1_fall_state.h"
#include "g1_fight_state.h"
#include "g1_hit_detector.h"

/*
 * Allocation-free combat coordinator for one two-fighter arena. MuJoCo
 * measurement and physical spawn resets remain caller-owned.
 */

typedef enum RekG1CombatTickStatus {
    REK_G1_COMBAT_TICK_OK = 0,
    REK_G1_COMBAT_TICK_NULL_ARGUMENT = 1,
    REK_G1_COMBAT_TICK_NOT_READY = 2,
    REK_G1_COMBAT_TICK_INPUT_INVALID = 3,
    REK_G1_COMBAT_TICK_HIT_REJECTED = 4,
    REK_G1_COMBAT_TICK_FIGHT_REJECTED = 5,
    REK_G1_COMBAT_TICK_OVERFLOW = 6,
} RekG1CombatTickStatus;

typedef struct RekG1CombatArenaState {
    RekG1HitDetectorState hit_detector;
    RekG1FightState fight;
    uint8_t initialized;
} RekG1CombatArenaState;

typedef struct RekG1CombatSubstepInput {
    float delta_seconds;
    float time_remaining_seconds;
    const RekG1HitContact* contacts;
    size_t contact_count;
    RekG1FallPhase fall_phase[2];
    uint32_t fall_events[2];
    uint8_t fighter_is_recovering[2];
    uint8_t fighter_can_get_up[2];
    uint8_t force_slip_estop[2];
} RekG1CombatSubstepInput;

typedef struct RekG1CombatSubstepResult {
    RekG1CombatArenaState next_state;
    uint32_t signals;
    uint32_t referee_calls;
    int32_t score_delta[2];
    int32_t rounds_won_delta[2];
    uint32_t attributed_contact_count;
    uint32_t scored_contact_count;
} RekG1CombatSubstepResult;

/*
 * Initializes the exact current-build fight state, prepares round one, and
 * activates it immediately. Immediate activation is the Puffer active-round
 * episode boundary. The external REK countdown Timeline duration is unknown
 * and is not inferred here.
 */
RekG1CombatTickStatus rek_g1_combat_arena_init(
    RekG1CombatArenaState* state
);

/*
 * Applies one post-mj_step state in recovered order. On failure neither state
 * nor result is modified.
 */
RekG1CombatTickStatus rek_g1_combat_arena_substep(
    const RekG1CombatArenaState* state,
    const RekG1HitDetectorConfig* hit_config,
    const RekG1CombatSubstepInput* input,
    RekG1CombatSubstepResult* result
);

/* Commit the caller-owned physical ResetBothToSpawn boundary. */
RekG1CombatTickStatus rek_g1_combat_arena_apply_spawn_reset(
    const RekG1CombatArenaState* state,
    RekG1CombatArenaState* next_state
);

const char* rek_g1_combat_tick_status_string(RekG1CombatTickStatus status);
