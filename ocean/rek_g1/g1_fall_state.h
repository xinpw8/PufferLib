#pragma once

#include <stdint.h>

/*
 * Build-pinned fall detector recovered from REKApp.Robot in the current G1
 * build.  Geometry is deliberately outside this module.  Every value in a
 * RekG1FallSample must be measured by the caller from the simulated state.
 *
 * This module owns the upright -> falling -> fallen detector, the unarmed
 * fallen reset countdown, and the post-reset grace gate.  It does not choose
 * FightCoordinator behavior, execute a teleport, or execute a get-up policy.
 */

#define REK_G1_FALL_BUILD_FINGERPRINT \
    "f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659"

typedef enum RekG1FallStatus {
    REK_G1_FALL_OK = 0,
    REK_G1_FALL_NULL_ARGUMENT = 1,
    REK_G1_FALL_NON_FINITE = 2,
    REK_G1_FALL_CONFIG_INVALID = 3,
    REK_G1_FALL_STATE_INVALID = 4,
    REK_G1_FALL_INPUT_INVALID = 5,
} RekG1FallStatus;

typedef enum RekG1FallPhase {
    REK_G1_FALL_UPRIGHT = 0,
    REK_G1_FALL_FALLING = 1,
    REK_G1_FALL_FALLEN = 2,
} RekG1FallPhase;

typedef enum RekG1FallEvent {
    REK_G1_FALL_EVENT_NONE = 0,
    REK_G1_FALL_EVENT_FALLING_STARTED = 1u << 0,
    REK_G1_FALL_EVENT_FALLING_CLEARED = 1u << 1,
    REK_G1_FALL_EVENT_BECAME_FALLEN = 1u << 2,
    REK_G1_FALL_EVENT_RESET_TIMEOUT_DUE = 1u << 3,
} RekG1FallEvent;

typedef struct RekG1FallConfig {
    float falling_tilt_degrees;
    float falling_height_ratio;
    float fallen_tilt_degrees;
    float fallen_height_ratio;
    uint32_t fallen_contact_points;
    float fallen_hold_fast_seconds;
    float fallen_hold_slow_seconds;
    float fallen_reset_timeout_seconds;
    float reset_grace_seconds;
    float fight_spawn_reset_grace_seconds;
} RekG1FallConfig;

/* Serialized Robot values for build fingerprint f84f1874... */
extern const RekG1FallConfig REK_G1_FALL_CONFIG_F84F1874;

typedef struct RekG1FallState {
    RekG1FallPhase phase;
    float fallen_hold_seconds;
    float fallen_elapsed_seconds;
    float fallen_timer_seconds;
    float reset_grace_remaining_seconds;
    /* Sampled from IPolicyRunner.CanGetUp only when BECAME_FALLEN occurs. */
    uint8_t recovery_armed;
} RekG1FallState;

typedef struct RekG1FallSample {
    uint8_t tracking_active;
    float tilt_degrees;
    float pelvis_height_ratio;
    uint8_t both_feet_off_floor;
    uint8_t has_foot_body_contact;
    uint32_t distinct_nonfoot_body_contact_count;
    float fixed_delta_seconds;
    /* Caller-measured IPolicyRunner.CanGetUp at this fixed tick. */
    uint8_t can_get_up;
} RekG1FallSample;

typedef struct RekG1FallStepResult {
    RekG1FallState next_state;
    uint32_t events;
} RekG1FallStepResult;

/* Initialize an upright state.  On error, state is not modified. */
RekG1FallStatus rek_g1_fall_state_init(
    const RekG1FallConfig* config,
    RekG1FallState* state
);

/*
 * Advance one active Robot.FixedUpdate fall-detection tick.  The caller must
 * retain responsibility for the native early gates not represented here,
 * including isResetting, fallDetectionSuppressed, and motorShutdownHold.  The
 * reset-grace gate is represented and consumes one fixed tick whenever it is
 * positive.
 *
 * If recovery_armed is true, the native reset countdown is bypassed.  Get-up
 * settle and handoff require additional measured inputs and are intentionally
 * outside this API.  On RESET_TIMEOUT_DUE, the timer is rearmed to the pinned
 * timeout just as the OnResetDue callback path does.  The caller decides
 * whether that event resets an arena, resets a robot, or is otherwise handled.
 *
 * On error, result is not modified.
 */
RekG1FallStatus rek_g1_fall_state_step(
    const RekG1FallConfig* config,
    const RekG1FallState* state,
    const RekG1FallSample* sample,
    RekG1FallStepResult* result
);

/*
 * Apply the state changes at entry to Robot.ResetAfterFall and
 * TeleportAndResetJoints.  This does not perform the teleport.  It starts the
 * pinned 0.5 s grace gate and returns the detector to upright.  The input state
 * must be FALLEN.  On error, next_state is not modified.
 */
RekG1FallStatus rek_g1_fall_state_apply_reset_after_fall(
    const RekG1FallConfig* config,
    const RekG1FallState* state,
    RekG1FallState* next_state
);

/*
 * Apply Robot.ResetToSpawn as invoked by FightCoordinator.ResetBothToSpawn.
 * Unlike the robot-local ResetAfterFall path above, this applies to both
 * fighters regardless of their current fall phase and starts the recovered
 * 2.0 s fall-detection suppression interval.  Physical root and joint reset
 * timing remains caller-owned.
 */
RekG1FallStatus rek_g1_fall_state_apply_fight_spawn_reset(
    const RekG1FallConfig* config,
    const RekG1FallState* state,
    RekG1FallState* next_state
);
