#pragma once

#include <stddef.h>
#include <stdint.h>

#include "g1_combat_tick.h"
#include "g1_fall_mujoco.h"
#include "g1_hit_mujoco.h"
#include "g1_strike_catalog.h"
#include "gear_sonic_native_duel.h"
#include "native_locomotion_command.h"
#include "native_puffer_vector.h"
#include "sonic_motion_composer_native.h"

/*
 * Build-pinned semantic G1 runtime. One Puffer environment is one physical
 * robot. Rows are arena-major and player-then-opponent, so environment_count
 * is exactly 2 * arena_count. Both rows in an arena are applied before the
 * single shared-contact MuJoCo step.
 *
 * The runtime supplies factual physical, fall, hit, score, and referee state.
 * Its zero-sum score-delta reward is a labeled training contract, not a REK
 * measurement. A Puffer episode ends at the recovered round-end event. The
 * terminal observation is the final state of that episode. Only the ended
 * arena is reset before its next outer step. The current REK server policy
 * identity and delayed spawn-reset parity remain gated.
 */

#define REK_G1_SEMANTIC_DUEL_SCHEMA_VERSION 4u
#define REK_G1_SEMANTIC_DUEL_CONTROLLER_RATE_HZ 50u
#define REK_G1_SEMANTIC_DUEL_CONTROL_DELTA_SECONDS 0.02f
#define REK_G1_SEMANTIC_DUEL_PHYSICS_DELTA_SECONDS 0.002f
#define REK_G1_SEMANTIC_DUEL_YAW_RAMP_SECONDS 0.5f
#define REK_G1_SEMANTIC_DUEL_SETTLE_LINEAR_SPEED 0.03f
#define REK_G1_SEMANTIC_DUEL_SETTLE_YAW_RATE 0.03f
#define REK_G1_SEMANTIC_DUEL_STOP_BRAKE_RATE 2.0f
#define REK_G1_SEMANTIC_DUEL_REFERENCE_ROWS \
    SONIC_MOTION_COMPOSER_NATIVE_REFERENCE_ROWS

enum {
    REK_G1_SEMANTIC_DUEL_REWARDS_IMPLEMENTED = 1,
    REK_G1_SEMANTIC_DUEL_TERMINALS_IMPLEMENTED = 1,
    /*
     * Candidate selection. Build-pinned f84f1874 static evidence has null
     * prone and supine get-up clips on SonicPolicyRunner path 3188. In the
     * user-observed authentic L100 session, a down awarded five points and
     * reset both fighters instead of invoking a get-up. Runtime injection in
     * other builds remains unknown, and this observation alone cannot support
     * a parity claim.
     */
    REK_G1_SEMANTIC_DUEL_PROVISIONAL_CAN_GET_UP = 0,
};

typedef enum RekG1SemanticDuelStatus {
    REK_G1_SEMANTIC_DUEL_OK = 0,
    REK_G1_SEMANTIC_DUEL_NULL_ARGUMENT = 1,
    REK_G1_SEMANTIC_DUEL_INVALID_DUEL = 2,
    REK_G1_SEMANTIC_DUEL_INVALID_CONFIG = 3,
    REK_G1_SEMANTIC_DUEL_INVALID_ROUTE_ASSET = 4,
    REK_G1_SEMANTIC_DUEL_INVALID_FIXED_IDLE = 5,
    REK_G1_SEMANTIC_DUEL_ALLOCATION_FAILED = 6,
    REK_G1_SEMANTIC_DUEL_NOT_READY = 7,
    REK_G1_SEMANTIC_DUEL_RECOVERY_REQUIRED = 8,
    REK_G1_SEMANTIC_DUEL_FORGIVENESS_UNAVAILABLE = 9,
    REK_G1_SEMANTIC_DUEL_COMPOSER_FAILED = 10,
    REK_G1_SEMANTIC_DUEL_LOCOMOTION_FAILED = 11,
    REK_G1_SEMANTIC_DUEL_PHYSICS_FAILED = 12,
    REK_G1_SEMANTIC_DUEL_OBSERVATION_FAILED = 13,
    REK_G1_SEMANTIC_DUEL_PROTOCOL_INVALID = 14,
    REK_G1_SEMANTIC_DUEL_FALL_MEASUREMENT_FAILED = 15,
    REK_G1_SEMANTIC_DUEL_HIT_MEASUREMENT_FAILED = 16,
    REK_G1_SEMANTIC_DUEL_COMBAT_FAILED = 17,
    REK_G1_SEMANTIC_DUEL_ARENA_RESET_FAILED = 18,
} RekG1SemanticDuelStatus;

/*
 * Clip sample arrays remain caller-owned for the runtime lifetime. Move
 * duration is a separately configured compositor traversal length in
 * controller ticks. It is never inferred from asset frame count. It must be
 * zero for locomotion and nonzero for discrete moves, then match the generated
 * Puffer action table at reset. It is not a measured physical completion time.
 */
typedef struct RekG1SemanticDuelRouteAsset {
    RekG1NativeRouteId route_id;
    SonicMotionComposerNativeClip clip;
    uint32_t configured_compositor_duration_ticks;
} RekG1SemanticDuelRouteAsset;

typedef struct RekG1SemanticDuelFallObservation {
    float tracking_active;
    float tilt_degrees;
    float pelvis_height_ratio;
    float both_feet_off_floor;
    float left_foot_body_contact;
    float right_foot_body_contact;
    float distinct_nonfoot_body_contact_count;
    float build_pinned_can_get_up;
    float phase;
    float fallen_hold_seconds;
    float fallen_elapsed_seconds;
    float fallen_timer_seconds;
    float reset_grace_remaining_seconds;
    float recovery_armed;
    float events;
} RekG1SemanticDuelFallObservation;

typedef struct RekG1SemanticDuelEntityObservation {
    float root_position_world[3];
    float root_quaternion_wxyz[4];
    float linear_velocity_local[3];
    float angular_velocity_local[3];
    float joint_position_mujoco[GEAR_SONIC_ACTION_DIM];
    float joint_velocity_mujoco[GEAR_SONIC_ACTION_DIM];
    RekG1SemanticDuelFallObservation fall;
} RekG1SemanticDuelEntityObservation;

typedef struct RekG1SemanticDuelFightObservation {
    float self_fighter_index;
    float phase;
    float current_round_number;
    float current_round_is_redo;
    float round_duration_seconds;
    float time_remaining_seconds;
    float self_clean_hits;
    float opponent_clean_hits;
    float self_falls;
    float opponent_falls;
    float self_rounds_won;
    float opponent_rounds_won;
    float self_last_struck_valid;
    float opponent_last_struck_valid;
    float self_last_struck_age_seconds;
    float opponent_last_struck_age_seconds;
    float self_last_struck_speed;
    float opponent_last_struck_speed;
    float self_fall_classification;
    float opponent_fall_classification;
    float self_count_active;
    float opponent_count_active;
    float self_count_is_slip;
    float opponent_count_is_slip;
    float count_elapsed_seconds;
    float count_duration_seconds;
    float round_result;
    float round_winner_index;
    float knockout_occurred;
    float fight_result;
    float fight_winner_index;
    float tick_signals;
    float tick_referee_calls;
    float tick_self_score_delta;
    float tick_opponent_score_delta;
    float tick_self_fall_events;
    float tick_opponent_fall_events;
    float tick_attributed_contact_count;
    float tick_scored_contact_count;
} RekG1SemanticDuelFightObservation;

/* All fields are binary32 so the struct is a direct Puffer observation row. */
typedef struct RekG1SemanticDuelObservation {
    RekG1SemanticDuelEntityObservation self;
    RekG1SemanticDuelEntityObservation opponent;
    float self_heading_delta_wxyz[4];
    float effective_forward;
    float effective_strafe;
    float effective_yaw;
    float active_route_id;
    float locomotion_active;
    float transition_settling;
    float action_playing;
    float composer_busy;
    RekG1SemanticDuelFightObservation fight;
} RekG1SemanticDuelObservation;

enum {
    REK_G1_SEMANTIC_DUEL_FALL_OBSERVATION_FLOATS = 15,
    REK_G1_SEMANTIC_DUEL_ENTITY_OBSERVATION_FLOATS = 86,
    REK_G1_SEMANTIC_DUEL_FIGHT_OBSERVATION_FLOATS = 39,
    REK_G1_SEMANTIC_DUEL_OBSERVATION_FLOATS = 223,
};

/*
 * The forgiveness callback must return the exact heading-forgiveness delta in
 * radians for this row and tick. Supplying an explicit measured zero is valid;
 * omitting the callback is not. The runtime never assumes zero.
 */
typedef int (*RekG1SemanticDuelForgivenessFn)(
    void* context,
    const GearSonicNativeDuelVector* duel,
    size_t robot_row,
    float* delta_radians_out
);

typedef struct RekG1SemanticDuelConfig {
    RekG1InputTiming input_timing;
    RekG1NativeCommandConfig command;
    RekG1NativeLocomotionConfig locomotion;
    SonicMotionComposerNativeBackends composer_backends;
    SonicMotionComposerNativeMirrorTable mirror_table;
    RekG1SemanticDuelForgivenessFn forgiveness_delta;
    void* forgiveness_context;
} RekG1SemanticDuelConfig;

/*
 * The forgiveness callback context and mirror-table arrays remain
 * caller-owned and must outlive the runtime. Callback execution is serial in
 * physical-row order.
 */

typedef struct RekG1SemanticDuelRuntime {
    GearSonicNativeDuelVector* duel;
    const RekG1NativeMotionRouteTable* motion_routes;
    const RekG1SemanticDuelRouteAsset* route_assets;
    size_t route_asset_count;
    size_t robot_count;
    RekG1SemanticDuelConfig config;
    SonicMotionComposerNativeConfig
        route_configs[REK_G1_STATIC_ROUTE_COUNT];
    SonicMotionComposerNative* composers;
    SonicMotionComposerNative* scratch_composers;
    RekG1NativeLocomotionState* locomotion_states;
    RekG1NativeLocomotionState* scratch_locomotion_states;
    RekG1NativeVelocityCommand* effective_velocity;
    RekG1NativeVelocityCommand* scratch_effective_velocity;
    RekG1NativeRouteId* active_route_ids;
    RekG1NativeRouteId* scratch_active_route_ids;
    RekG1NativeBaseVelocitySample* velocity_samples;
    RekG1SemanticDuelEntityObservation* entity_observations;
    RekG1FallMujocoAdapter fall_adapter;
    RekG1HitMujocoAdapter hit_adapter;
    RekG1CombatArenaState* combat_states;
    RekG1CombatArenaState* scratch_combat_states;
    RekG1HitMujocoCandidate* hit_candidates;
    RekG1HitContact* hit_contacts;
    size_t hit_candidate_capacity;
    RekG1FallState* fall_states;
    RekG1FallState* scratch_fall_states;
    RekG1FallMujocoMeasurement* fall_measurements;
    RekG1FallMujocoMeasurement* scratch_fall_measurements;
    uint32_t* fall_events;
    uint32_t* scratch_fall_events;
    uint32_t* substep_fall_events;
    uint32_t* fight_signals;
    uint32_t* referee_calls;
    int32_t* score_deltas;
    uint32_t* attributed_contact_counts;
    uint32_t* scored_contact_counts;
    uint8_t* arena_reset_events;
    uint8_t* arena_input_reset_events;
    uint8_t* arena_terminals;
    uint8_t* pending_episode_resets;
    float* forgiveness_deltas;
    double* scratch_heading_delta_wxyz;
    float* reference_dof_position;
    float* reference_dof_next_position;
    float* reference_root_rotation_xyzw;
    uint8_t initialized;
    uint8_t fall_adapter_open;
    uint8_t hit_adapter_open;
    uint8_t ready;
    uint8_t failed;
    RekG1SemanticDuelStatus last_status;
} RekG1SemanticDuelRuntime;

const char* rek_g1_semantic_duel_status_string(
    RekG1SemanticDuelStatus status);

/*
 * Allocates all scratch/state storage. The caller must pass a zero-initialized
 * object and close a successful instance before opening it again. After this
 * call succeeds, reset and advance do not allocate. The duel and route assets
 * are not owned.
 */
RekG1SemanticDuelStatus rek_g1_semantic_duel_open(
    RekG1SemanticDuelRuntime* runtime,
    GearSonicNativeDuelVector* duel,
    const RekG1NativeMotionRouteTable* motion_routes,
    const RekG1SemanticDuelRouteAsset* route_assets,
    size_t route_asset_count,
    const RekG1SemanticDuelConfig* config,
    char* error,
    size_t error_capacity
);

int rek_g1_semantic_duel_reset_batch(
    void* context,
    const RekG1NativeMotionRouteTable* motion_routes,
    const RekG1PufferActionTable* action_table,
    size_t environment_count,
    RekG1RuntimeFacts* facts_out,
    void* observations,
    size_t observation_stride_bytes,
    float* rewards,
    float* terminals,
    char* error,
    size_t error_capacity
);

int rek_g1_semantic_duel_advance_batch(
    void* context,
    const RekG1NativeMotionRouteTable* motion_routes,
    const RekG1PufferActionTable* action_table,
    const RekG1SemanticTick* semantics,
    size_t environment_count,
    RekG1RuntimeFacts* next_facts_out,
    void* observations,
    size_t observation_stride_bytes,
    float* rewards,
    float* terminals,
    char* error,
    size_t error_capacity
);

RekG1NativeBatchOps rek_g1_semantic_duel_batch_ops(void);

/* Frees runtime-owned storage. It does not close the caller-owned duel. */
void rek_g1_semantic_duel_close(void* context);
