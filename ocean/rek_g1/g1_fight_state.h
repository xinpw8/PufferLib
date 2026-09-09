#pragma once

#include <stdint.h>

#include "g1_combat_types.h"

/*
 * Pure current-build REK fight state machine.
 *
 * Contact geometry, strike-apex acceptance, robot fall geometry, physical
 * reset, get-up execution, and RL reward shaping remain caller-owned. This
 * module consumes the measured callback-level facts those systems produce.
 */

#define REK_G1_FIGHT_BUILD_FINGERPRINT \
    "f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659"

typedef enum RekG1FightStatus {
    REK_G1_FIGHT_OK = 0,
    REK_G1_FIGHT_NULL_ARGUMENT = 1,
    REK_G1_FIGHT_NON_FINITE = 2,
    REK_G1_FIGHT_STATE_INVALID = 3,
    REK_G1_FIGHT_INPUT_INVALID = 4,
    REK_G1_FIGHT_WRONG_PHASE = 5,
    REK_G1_FIGHT_OVERFLOW = 6,
} RekG1FightStatus;

/* Module phase. ROUND_COUNTDOWN maps to native RoundActive + !IsActive. */
typedef enum RekG1FightPhase {
    REK_G1_FIGHT_IDLE = 0,
    REK_G1_FIGHT_ROUND_COUNTDOWN = 1,
    REK_G1_FIGHT_ROUND_ACTIVE = 2,
    REK_G1_FIGHT_BETWEEN_ROUNDS = 3,
    REK_G1_FIGHT_OVER = 4,
} RekG1FightPhase;

/* Native REKApp values retained for serialized-state compatibility. */
typedef enum RekG1RoundResult {
    REK_G1_ROUND_IN_PROGRESS = 0,
    REK_G1_ROUND_WON_BY_POINTS = 1,
    REK_G1_ROUND_WON_BY_KO = 2,
    REK_G1_ROUND_TIE = 3,
    REK_G1_ROUND_REDO = 4,
} RekG1RoundResult;

typedef enum RekG1FightResult {
    REK_G1_FIGHT_IN_PROGRESS = 0,
    REK_G1_FIGHT_WON_BY_ROUNDS = 1,
    REK_G1_FIGHT_WON_BY_TKO = 2,
} RekG1FightResult;

typedef enum RekG1FallClassification {
    REK_G1_FALL_UNCLASSIFIED = 0,
    REK_G1_FALL_SLIP = 1,
    REK_G1_FALL_KNOCKDOWN = 2,
} RekG1FallClassification;

/* Native RefereeCall value converted to a bit position. */
typedef enum RekG1RefereeCallBit {
    REK_G1_REFEREE_SLIP = 1u << 0,
    REK_G1_REFEREE_SLIP_ESTOP = 1u << 1,
    REK_G1_REFEREE_KNOCKDOWN = 1u << 2,
    REK_G1_REFEREE_BEAT_COUNT = 1u << 3,
    REK_G1_REFEREE_KNOCKOUT = 1u << 4,
    REK_G1_REFEREE_DOUBLE_KNOCKDOWN = 1u << 5,
    REK_G1_REFEREE_DOUBLE_KNOCKOUT = 1u << 6,
} RekG1RefereeCallBit;

typedef enum RekG1FightSignal {
    REK_G1_FIGHT_SIGNAL_NONE = 0,
    REK_G1_FIGHT_SIGNAL_SCORE_CHANGED = 1u << 0,
    REK_G1_FIGHT_SIGNAL_FALL_CLASSIFIED = 1u << 1,
    REK_G1_FIGHT_SIGNAL_COUNT_STARTED = 1u << 2,
    REK_G1_FIGHT_SIGNAL_COUNT_RESTARTED = 1u << 3,
    REK_G1_FIGHT_SIGNAL_COUNT_CLEARED = 1u << 4,
    REK_G1_FIGHT_SIGNAL_RESET_BOTH_TO_SPAWN = 1u << 5,
    REK_G1_FIGHT_SIGNAL_ROUND_PREPARED = 1u << 6,
    REK_G1_FIGHT_SIGNAL_ROUND_STARTED = 1u << 7,
    REK_G1_FIGHT_SIGNAL_ROUND_ENDED = 1u << 8,
    REK_G1_FIGHT_SIGNAL_FIGHT_ENDED = 1u << 9,
    REK_G1_FIGHT_SIGNAL_FIGHT_EXITED = 1u << 10,
} RekG1FightSignal;

typedef struct RekG1FightConfig {
    float normal_round_seconds;
    float redo_round_seconds;
    float between_round_seconds;
    float fight_over_seconds;
    float hit_speed_threshold;
    float knockdown_weak_speed;
    float knockdown_strong_speed;
    float knockdown_window_min_seconds;
    float knockdown_window_max_seconds;
    float ko_count_seconds;
    float double_knockdown_count_seconds;
    float no_recovery_count_seconds;
    int32_t hand_hit_points;
    int32_t kick_hit_points;
    int32_t slip_points_to_opponent;
    int32_t ko_points;
    uint32_t regular_round_limit;
    uint32_t rounds_to_win;
} RekG1FightConfig;

/* Exact level1/level2/level3 current-build values. */
extern const RekG1FightConfig REK_G1_FIGHT_CONFIG_F84F1874;

typedef struct RekG1FightState {
    RekG1FightPhase phase;
    uint32_t current_round_number;
    uint8_t current_round_is_redo;
    float round_duration_seconds;
    float time_remaining_seconds;
    int32_t clean_hits[2];
    uint32_t falls[2];
    RekG1RoundResult round_result;
    int32_t round_winner_index;
    uint8_t knockout_occurred;

    uint32_t rounds_won[2];
    RekG1FightResult fight_result;
    int32_t fight_winner_index;

    uint8_t last_struck_valid[2];
    float last_struck_age_seconds[2];
    float last_struck_speed[2];
    RekG1FallClassification fall_classification[2];
    uint8_t fall_forced_by_estop[2];
    uint8_t count_active[2];
    uint8_t count_is_slip[2];
    float count_elapsed_seconds;
    float count_duration_seconds;
    float transition_remaining_seconds;
} RekG1FightState;

/*
 * The upstream hit detector supplies callback-level facts. attribution_passed
 * means its IsAggressorStrike gate passed. score_accepted means every scoring
 * gate passed and PointTracker.RecordHit accepted the event. These flags are
 * deliberately separate because current REK updates fall attribution before
 * its scoring-zone and apex gates.
 */
typedef struct RekG1StrikeEvent {
    uint32_t attacker_fighter_index;
    uint32_t victim_fighter_index;
    RekG1HandSide hand_side;
    RekG1BodyPartType striker_part;
    RekG1BodyZone zone;
    float relative_speed;
    uint8_t attribution_passed;
    uint8_t score_accepted;
} RekG1StrikeEvent;

/* Current observations sampled on one referee-coroutine clock advance. */
typedef struct RekG1FightAdvanceInput {
    float delta_seconds;
    float time_remaining_seconds;
    uint8_t fighter_is_fallen[2];
    uint8_t fighter_is_recovering[2];
    uint8_t fighter_can_get_up[2];
} RekG1FightAdvanceInput;

typedef struct RekG1FightResolveInput {
    uint8_t fighter_is_fallen[2];
    uint8_t fighter_is_recovering[2];
    uint8_t fighter_can_get_up[2];
} RekG1FightResolveInput;

typedef struct RekG1FightStepResult {
    RekG1FightState next_state;
    uint32_t signals;
    uint32_t referee_calls;
    int32_t score_delta[2];
    int32_t rounds_won_delta[2];
    int32_t subject_fighter_index;
    RekG1FallClassification fall_classification;
} RekG1FightStepResult;

/* Initialize an idle BestOf3 fight. On error, state is not modified. */
RekG1FightStatus rek_g1_fight_state_init(RekG1FightState* state);

/*
 * Prepare round 1 from IDLE. Between-round preparation is automatic after the
 * measured 5 s transition. Preparation requests a two-fighter spawn reset and
 * enters ROUND_COUNTDOWN. Countdown duration is caller-owned because that
 * external Timeline duration remains outside this fight-state contract.
 */
RekG1FightStatus rek_g1_fight_prepare_first_round(
    const RekG1FightState* state,
    RekG1FightStepResult* result
);

/* Activate a prepared round at the caller-measured countdown completion. */
RekG1FightStatus rek_g1_fight_activate_round(
    const RekG1FightState* state,
    RekG1FightStepResult* result
);

/* Apply one measured HitDetector/PointTracker strike event. */
RekG1FightStatus rek_g1_fight_record_strike(
    const RekG1FightState* state,
    const RekG1StrikeEvent* event,
    RekG1FightStepResult* result
);

/* Latch slip versus knockdown at Robot.OnFalling. */
RekG1FightStatus rek_g1_fight_on_falling(
    const RekG1FightState* state,
    uint32_t fighter_index,
    uint8_t force_slip_estop,
    RekG1FightStepResult* result
);

/* Increment Falls and start or restart the shared count at Robot.OnFallen. */
RekG1FightStatus rek_g1_fight_on_fallen(
    const RekG1FightState* state,
    uint32_t fighter_index,
    const uint8_t fighter_can_get_up[2],
    RekG1FightStepResult* result
);

/*
 * Apply Robot.OnResetDue routing. The active-round path deliberately emits no
 * reset. The pre-round countdown path requests a two-fighter spawn reset.
 */
RekG1FightStatus rek_g1_fight_on_reset_due(
    const RekG1FightState* state,
    uint32_t fighter_index,
    RekG1FightStepResult* result
);

/*
 * Clear transient strike attribution after the caller has physically reset
 * both fighters to spawn. Score, falls, round time, and match progress remain
 * unchanged. This is separate from Robot.ResetAfterFall state.
 */
RekG1FightStatus rek_g1_fight_apply_spawn_reset(
    const RekG1FightState* state,
    RekG1FightStepResult* result
);

/*
 * Two-phase active-round clock boundary for a physics engine that observes
 * hits and falls after each mj_step. Begin ages only state that existed before
 * the new physics state, clears counts beaten at that state, and stages
 * deadlines without resolving them. The caller then applies hit/fall events
 * from that same state and calls resolve. A count created between these calls
 * begins at zero and is not aged by the preceding physics interval.
 */
RekG1FightStatus rek_g1_fight_begin_active_step(
    const RekG1FightState* state,
    const RekG1FightAdvanceInput* input,
    RekG1FightStepResult* result
);

RekG1FightStatus rek_g1_fight_resolve_active_step(
    const RekG1FightState* state,
    const RekG1FightResolveInput* input,
    RekG1FightStepResult* result
);

/*
 * Advance an active round. Recovery checks run before deadline expiry, matching
 * RefereeCountRoutine. time_remaining_seconds is the caller-measured clamped
 * round clock. On a reset signal the caller performs the physical reset.
 */
RekG1FightStatus rek_g1_fight_advance_active(
    const RekG1FightState* state,
    const RekG1FightAdvanceInput* input,
    RekG1FightStepResult* result
);

/*
 * Advance countdown, between-round, or fight-over wall time. Countdown never
 * activates itself. Between-round expiry prepares the next 120 s or 30 s round.
 */
RekG1FightStatus rek_g1_fight_advance_transition(
    const RekG1FightState* state,
    float delta_seconds,
    RekG1FightStepResult* result
);
