#pragma once

#include <stdint.h>

#include "native_motion_routes.h"

/*
 * Build-pinned scalar command path recovered from the current
 * RobotInputController and SonicPolicyRunner native methods.  This layer
 * selects a locomotion route and computes the two scalar updates consumed by
 * SonicMotionComposer.  It does not choose transition entry frames, sample a
 * clip, or claim REK trajectory parity.
 */

#define REK_G1_NATIVE_COMMAND_BUILD_FINGERPRINT \
    REK_G1_STATIC_ROUTE_BUILD_FINGERPRINT

/* Binary32 value at GameAssembly VA 0x183D57BA8 in the pinned build. */
#define REK_G1_NATIVE_COMMAND_EPSILON 0.001f

/* Binary32 value at GameAssembly VA 0x183D57BB4 in the pinned build. */
#define REK_G1_NATIVE_STOP_BRAKE_DONE_MAGNITUDE 0.05f

typedef enum RekG1NativeCommandStatus {
    REK_G1_NATIVE_COMMAND_OK = 0,
    REK_G1_NATIVE_COMMAND_NULL_ARGUMENT = 1,
    REK_G1_NATIVE_COMMAND_NON_FINITE = 2,
    REK_G1_NATIVE_COMMAND_CONFIG_INVALID = 3,
    REK_G1_NATIVE_COMMAND_HEADING_OWNERSHIP_INVALID = 4,
    REK_G1_NATIVE_COMMAND_STATE_INVALID = 5,
    REK_G1_NATIVE_COMMAND_TIMING_INVALID = 6,
    REK_G1_NATIVE_COMMAND_INPUT_INVALID = 7,
    REK_G1_NATIVE_COMMAND_MEASUREMENT_UNAVAILABLE = 8,
} RekG1NativeCommandStatus;

typedef struct RekG1NativeVelocityCommand {
    float forward;
    float strafe;
    float yaw;
} RekG1NativeVelocityCommand;

typedef struct RekG1NativeCommandConfig {
    float locomotion_speed_scale;
    float command_yaw_rate_scale;
    float heading_yaw_rate_scale;
    uint32_t controller_rate_hz;
} RekG1NativeCommandConfig;

typedef struct RekG1NativeRouteSelection {
    RekG1NativeRouteId route_id;
    uint8_t locomotion_active;
} RekG1NativeRouteSelection;

typedef struct RekG1NativePlaybackUpdate {
    float command_magnitude;
    float scale;
    /*
     * SonicPolicyRunner skips SetLocomotionSpeed when its configured scale is
     * exactly zero.  In that case scale is zero and apply is false.
     */
    uint8_t apply;
} RekG1NativePlaybackUpdate;

typedef struct RekG1NativeHeadingUpdate {
    float clip_delta_radians;
    float command_delta_radians;
    float forgiveness_delta_radians;
    float total_delta_radians;
} RekG1NativeHeadingUpdate;

typedef enum RekG1NativeLocomotionEvent {
    REK_G1_NATIVE_LOCOMOTION_EVENT_NONE = 0,
    REK_G1_NATIVE_LOCOMOTION_EVENT_PLAY_IDLE = 1,
    REK_G1_NATIVE_LOCOMOTION_EVENT_PLAY_ROUTE = 2,
} RekG1NativeLocomotionEvent;

/*
 * Exact scalar fields consumed by UpdateLocomotionClip,
 * TransitionSettled, and UpdateStopBrake.  Callers must supply measured
 * values from the selected RobotConfig.  This API has no built-in defaults.
 */
typedef struct RekG1NativeLocomotionConfig {
    float settle_linear_speed;
    float settle_yaw_rate;
    float stop_brake_rate;
    uint8_t transition_settle;
} RekG1NativeLocomotionConfig;

typedef struct RekG1NativeBaseVelocitySample {
    RekG1NativeVelocityCommand angular_velocity_local;
    RekG1NativeVelocityCommand linear_velocity_local;
    uint8_t available;
} RekG1NativeBaseVelocitySample;

/*
 * Route ids FORWARD through TURN_RIGHT map one-to-one to the build's
 * LocomotionDir values 0 through 5.  IDLE is permitted only in direction
 * fields whose associated state flag is false.
 */
typedef struct RekG1NativeLocomotionState {
    RekG1NativeVelocityCommand stop_brake_command;
    RekG1NativeVelocityCommand last_driven_command;
    RekG1NativeRouteId current_route_id;
    RekG1NativeRouteId transition_from_route_id;
    RekG1NativeRouteId momentum_route_id;
    uint8_t locomotion_active;
    uint8_t transition_settling;
    uint8_t stop_braking;
    uint8_t has_momentum;
} RekG1NativeLocomotionState;

typedef struct RekG1NativeLocomotionStepInput {
    RekG1NativeVelocityCommand command;
    RekG1NativeBaseVelocitySample base_velocity;
    float delta_seconds;
    uint8_t restrict_yaw;
    uint8_t composer_action_playing;
    uint8_t composer_busy;
    /*
     * Explicit result of resolving the selected direction's
     * MocapClipConfig and its motion asset.  Zero reproduces the native
     * early return without inventing a replacement route.
     */
    uint8_t selected_route_playable;
} RekG1NativeLocomotionStepInput;

typedef struct RekG1NativeLocomotionStepResult {
    RekG1NativeLocomotionState next_state;
    RekG1NativeVelocityCommand effective_velocity;
    RekG1NativeRouteId selected_route_id;
    RekG1NativeRouteId event_route_id;
    RekG1NativeLocomotionEvent event;
    /* The pinned method called set_VelocityCommand during this step. */
    uint8_t velocity_write;
    uint8_t transition_check_performed;
    uint8_t transition_settled;
} RekG1NativeLocomotionStepResult;

RekG1NativeCommandStatus rek_g1_native_select_locomotion_route(
    RekG1NativeVelocityCommand command,
    RekG1NativeRouteSelection* selection
);

RekG1NativeCommandStatus rek_g1_native_playback_update(
    RekG1NativeVelocityCommand command,
    const RekG1NativeCommandConfig* config,
    RekG1NativePlaybackUpdate* update
);

RekG1NativeCommandStatus rek_g1_native_heading_update(
    RekG1NativeVelocityCommand command,
    const RekG1NativeCommandConfig* config,
    float heading_clip_ownership,
    float consumed_clip_heading_delta_radians,
    float forgiveness_delta_radians,
    RekG1NativeHeadingUpdate* update
);

RekG1NativeCommandStatus rek_g1_native_transition_settled(
    RekG1NativeRouteId outgoing_route_id,
    const RekG1NativeLocomotionConfig* config,
    const RekG1NativeBaseVelocitySample* base_velocity,
    uint8_t* settled
);

/*
 * Heap-free, transactional state transition recovered from the pinned
 * RobotInputController methods.  On error, result is not modified.
 * The game method treats an unavailable base-velocity measurement as
 * settled.  This API instead returns MEASUREMENT_UNAVAILABLE so a training
 * runtime cannot silently replace a missing measurement with that fallback.
 * PLAY_IDLE represents a call to PlayIdle.  PLAY_ROUTE is emitted only when
 * selected_route_playable is true, matching the native clip and motion
 * presence checks.  This component does not establish trajectory parity.
 */
RekG1NativeCommandStatus rek_g1_native_locomotion_step(
    const RekG1NativeLocomotionState* state,
    const RekG1NativeLocomotionConfig* config,
    const RekG1NativeLocomotionStepInput* input,
    RekG1NativeLocomotionStepResult* result
);
