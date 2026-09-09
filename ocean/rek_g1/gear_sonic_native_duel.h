#pragma once

#include <stddef.h>
#include <stdint.h>

#include <mujoco/mujoco.h>

#include "gear_sonic_native_batch.h"

#define GEAR_SONIC_DUEL_FIGHTERS 2
#define GEAR_SONIC_DUEL_QPOS_DIM 72
#define GEAR_SONIC_DUEL_QVEL_DIM 70
#define GEAR_SONIC_DUEL_CONTROL_DIM 58
#define GEAR_SONIC_DUEL_DAMPEN_RETENTION 0.10000000149011612f

typedef enum GearSonicDuelFighter {
    GEAR_SONIC_DUEL_PLAYER = 0,
    GEAR_SONIC_DUEL_OPPONENT = 1,
} GearSonicDuelFighter;

typedef struct GearSonicDuelFighterMap {
    int joint_ids[GEAR_SONIC_ACTION_DIM];
    int qpos_addresses[GEAR_SONIC_ACTION_DIM];
    int qvel_addresses[GEAR_SONIC_ACTION_DIM];
    int actuator_ids[GEAR_SONIC_ACTION_DIM];
    int root_qpos_address;
    int root_qvel_address;
    int root_body_id;
} GearSonicDuelFighterMap;

typedef struct GearSonicNativeDuelVector {
    uint64_t lifecycle_token;
    mjModel* model;
    mjData** data;
    size_t arena_count;
    size_t robot_count;
    int physics_workers;
    GearSonicDuelFighterMap fighters[GEAR_SONIC_DUEL_FIGHTERS];
    GearSonicNativeMotion fixed_motion;
    GearSonicOrtBatch ort;
    GearSonicNativeBatch controller;
    double* base_quaternion_wxyz;
    double* base_angular_velocity_local;
    double* joint_position_mujoco;
    double* joint_velocity_mujoco;
    double* heading_delta_wxyz;
    size_t* reference_frames;
    /*
     * policy_ticks counts accepted native 50 Hz controller evaluations. It is
     * not SonicPolicyRunner.stepCounter, which advances on unsuspended 2 ms
     * FixedUpdate calls. Both counters freeze while Robot.IsPolicySuspended is
     * true. motion_ticks drives the reference frame.
     */
    uint64_t* policy_ticks;
    uint64_t* motion_ticks;
    float* command_lpf_state_mujoco;
    uint8_t* command_lpf_initialized;
    uint8_t* dampened_rows;
    float* dampened_control_target_mujoco;
    float* dampened_kp_mujoco;
    float* dampened_kd_mujoco;
    float* dampened_force_limit_mujoco;
    void* dampened_controller_snapshots;
    uint8_t* resetting_rows;
    uint8_t* reset_pending_arenas;
    /* Set by any successful reset completion; outer steps clear it on entry. */
    uint8_t* reset_completed_in_step_arenas;
    double* reset_complete_not_before_time;
    int spawn_prefixes_verified;
    int failed;
} GearSonicNativeDuelVector;

typedef struct GearSonicNativeDuelPostStepObservation {
    const GearSonicNativeDuelVector* vector;
    const mjModel* model;
    const mjData* data;
    size_t arena_index;
    uint32_t physics_substep_index;
    double physics_dt_seconds;
} GearSonicNativeDuelPostStepObservation;

typedef enum GearSonicNativeDuelPostStepDirective {
    GEAR_SONIC_DUEL_POST_STEP_CONTINUE = 0,
    GEAR_SONIC_DUEL_POST_STEP_RESET_ARENA_IMMEDIATE = 1,
} GearSonicNativeDuelPostStepDirective;

/* Return zero to reject the just-completed physics substep. */
typedef int (*GearSonicNativeDuelPostStepObserver)(
    void* context,
    const GearSonicNativeDuelPostStepObservation* observation
);

/*
 * Return zero to reject the just-completed substep. A successful callback
 * must write one GearSonicNativeDuelPostStepDirective. The reset directive is
 * applied serially to observation->arena_index before another callback or
 * physics substep is entered.
 */
typedef int (*GearSonicNativeDuelPostStepDirectiveCallback)(
    void* context,
    const GearSonicNativeDuelPostStepObservation* observation,
    GearSonicNativeDuelPostStepDirective* directive
);

/*
 * The initial runtime deliberately has one caller-supplied fixed reference
 * motion shared by every physical robot.  Its controller batch contains
 * exactly 2 * arena_count rows in arena-major, player-then-opponent order.
 * No semantic commands are accepted by gear_sonic_native_duel_step_fixed. The
 * caller must zero-initialize the vector. Opening an active vector is rejected
 * without changing it.
 */
int gear_sonic_native_duel_open(
    GearSonicNativeDuelVector* vector,
    const char* model_path,
    const char* encoder_path,
    const char* decoder_path,
    GearSonicNativeMotion fixed_motion,
    size_t arena_count,
    int physics_workers,
    char* error,
    size_t error_capacity
);

/*
 * Equivalent to gear_sonic_native_duel_open except that the ONNX Runtime
 * sessions are constructed directly from the supplied model arrays. The
 * arrays need only remain valid through this call.
 */
int gear_sonic_native_duel_open_from_memory(
    GearSonicNativeDuelVector* vector,
    const char* model_path,
    const void* encoder_data,
    size_t encoder_byte_count,
    const void* decoder_data,
    size_t decoder_byte_count,
    GearSonicNativeMotion fixed_motion,
    size_t arena_count,
    int physics_workers,
    char* error,
    size_t error_capacity
);

int gear_sonic_native_duel_reset(
    GearSonicNativeDuelVector* vector,
    char* error,
    size_t error_capacity
);

/*
 * Immediately restore the physical state and the two controller rows of one
 * arena without touching another arena. The selected mjData time is
 * preserved. This is only the immediate physical reset primitive required by
 * the vector runtime. It does not implement REK's recovered
 * ResetBothToSpawn sequence, whose reset is applied one FixedUpdate tick after
 * the round-reset request.
 */
int gear_sonic_native_duel_reset_arena_immediate(
    GearSonicNativeDuelVector* vector,
    size_t arena_index,
    char* error,
    size_t error_capacity
);

/*
 * Enter or leave recovered Robot dampening for one controller row. Entering
 * snapshots the row's last applied joint target and the live staged actuator
 * gains, then applies the exact recovered float32 retention value to kp, kd,
 * and force limit without modifying the shared mjModel. While dampened, the
 * controller history, policy output, reference frame, and command LPF remain
 * frozen. The arena's MuJoCo physics continues advancing.
 *
 * This surface implements recovered fall-time execution semantics. The live
 * native actuator values it retains are staging gains, not evidence that the
 * current Steam build uses those particular numeric gains.
 */
int gear_sonic_native_duel_set_row_dampened(
    GearSonicNativeDuelVector* vector,
    size_t row,
    int dampened,
    char* error,
    size_t error_capacity
);

/*
 * Begin the recovered in-round ResetBothToSpawn physical sequence for one
 * arena. Both root poses are restored immediately. Articulated joint qpos and
 * every qvel are retained, and both controller rows become policy-suspended.
 * Completion is rejected until mjData.time reaches the 2 ms deadline. This
 * public primitive cannot prove that mj_step caused the time advance, so its
 * caller must invoke completion only after a real fixed step. The semantic
 * runtime satisfies that requirement.
 */
int gear_sonic_native_duel_begin_arena_reset(
    GearSonicNativeDuelVector* vector,
    size_t arena_index,
    char* error,
    size_t error_capacity
);

/*
 * Complete a pending in-round reset after its one-boundary delay. Articulated
 * joint qpos and qvel are set to zero, root poses are restored again, and the
 * arena's two controller rows, counters, LPF state, and suspension state are
 * reset. Arena time and root qvel are preserved. Composer and input-controller
 * reset handlers remain the semantic runtime caller's responsibility.
 */
int gear_sonic_native_duel_complete_arena_reset(
    GearSonicNativeDuelVector* vector,
    size_t arena_index,
    char* error,
    size_t error_capacity
);

int gear_sonic_native_duel_step_fixed(
    GearSonicNativeDuelVector* vector,
    char* error,
    size_t error_capacity
);

/*
 * Invoke observer serially in ascending arena order after each parallel
 * mj_step barrier. physics_substep_index is zero-based within the ten 2 ms
 * substeps of one policy tick. A zero callback result latches vector failure,
 * reports the arena and substep, and prevents all later substeps.
 */
int gear_sonic_native_duel_step_fixed_with_post_step_observer(
    GearSonicNativeDuelVector* vector,
    GearSonicNativeDuelPostStepObserver observer,
    void* observer_context,
    char* error,
    size_t error_capacity
);

/*
 * Advance from externally composed reference windows.  Each reference array
 * is arena-major, player-then-opponent and therefore contains exactly
 * 2 * arena_count * GEAR_SONIC_HISTORY_FRAMES rows.  The caller must resolve
 * semantic commands to measured motion samples before calling this boundary.
 */
int gear_sonic_native_duel_step_references(
    GearSonicNativeDuelVector* vector,
    const GearSonicNativeReferenceInput* references,
    char* error,
    size_t error_capacity
);

int gear_sonic_native_duel_step_references_with_post_step_observer(
    GearSonicNativeDuelVector* vector,
    const GearSonicNativeReferenceInput* references,
    GearSonicNativeDuelPostStepObserver observer,
    void* observer_context,
    char* error,
    size_t error_capacity
);

/*
 * Apply serial callback directives after each arena's completed 2 ms
 * substep. RESET_ARENA_IMMEDIATE uses the immediate physical primitive above,
 * preserves that arena's mjData time, and completes before any later callback
 * or substep. It is not the recovered one-FixedUpdate delayed REK reset.
 */
int gear_sonic_native_duel_step_references_with_post_step_directive(
    GearSonicNativeDuelVector* vector,
    const GearSonicNativeReferenceInput* references,
    GearSonicNativeDuelPostStepDirectiveCallback callback,
    void* callback_context,
    char* error,
    size_t error_capacity
);

/*
 * This explicit fail-closed surface reserves semantic routing without
 * pretending command identifiers have been recovered or wired.  Calling it
 * never advances physics and always reports the missing router.
 */
int gear_sonic_native_duel_step_semantic_unavailable(
    GearSonicNativeDuelVector* vector,
    const uint32_t* opaque_command_ids,
    size_t command_count,
    char* error,
    size_t error_capacity
);

void gear_sonic_native_duel_close(GearSonicNativeDuelVector* vector);
