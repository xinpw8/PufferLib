#pragma once

#include <stddef.h>
#include <stdint.h>

#include "g1_hit_detector.h"
#include "gear_sonic_native_duel.h"

#define REK_G1_HIT_MUJOCO_PHYSICS_SUBSTEPS 10u
#define REK_G1_HIT_MUJOCO_MODEL_BODY_COUNT 63u
#define REK_G1_HIT_MUJOCO_MODEL_GEOM_COUNT 91u
#define REK_G1_HIT_MUJOCO_FIGHTER_BODY_COUNT 30u

typedef enum RekG1HitMujocoStatus {
    REK_G1_HIT_MUJOCO_OK = 0,
    REK_G1_HIT_MUJOCO_NULL_ARGUMENT = 1,
    REK_G1_HIT_MUJOCO_INVALID_DUEL = 2,
    REK_G1_HIT_MUJOCO_MAPPING_MISSING = 3,
    REK_G1_HIT_MUJOCO_MAPPING_MISMATCH = 4,
    REK_G1_HIT_MUJOCO_ALLOCATION_FAILED = 5,
    REK_G1_HIT_MUJOCO_OBSERVATION_INVALID = 6,
    REK_G1_HIT_MUJOCO_SUBSTEP_SEQUENCE_INVALID = 7,
    REK_G1_HIT_MUJOCO_NON_FINITE = 8,
    REK_G1_HIT_MUJOCO_CAPACITY_INSUFFICIENT = 9,
    REK_G1_HIT_MUJOCO_CALLER_FACTS_INVALID = 10,
} RekG1HitMujocoStatus;

typedef enum RekG1HitMujocoStrikerSlot {
    REK_G1_HIT_MUJOCO_LEFT_HAND_SLOT = 0,
    REK_G1_HIT_MUJOCO_RIGHT_HAND_SLOT = 1,
    REK_G1_HIT_MUJOCO_LEFT_FOOT_SLOT = 2,
    REK_G1_HIT_MUJOCO_RIGHT_FOOT_SLOT = 3,
    REK_G1_HIT_MUJOCO_LEFT_SHIN_SLOT = 4,
    REK_G1_HIT_MUJOCO_RIGHT_SHIN_SLOT = 5,
} RekG1HitMujocoStrikerSlot;

typedef enum RekG1HitMujocoFighterBody {
    REK_G1_HIT_MUJOCO_PELVIS_BODY = 0,
    REK_G1_HIT_MUJOCO_LEFT_HIP_PITCH_BODY = 1,
    REK_G1_HIT_MUJOCO_LEFT_HIP_ROLL_BODY = 2,
    REK_G1_HIT_MUJOCO_LEFT_HIP_YAW_BODY = 3,
    REK_G1_HIT_MUJOCO_LEFT_KNEE_BODY = 4,
    REK_G1_HIT_MUJOCO_LEFT_ANKLE_PITCH_BODY = 5,
    REK_G1_HIT_MUJOCO_LEFT_ANKLE_ROLL_BODY = 6,
    REK_G1_HIT_MUJOCO_RIGHT_HIP_PITCH_BODY = 7,
    REK_G1_HIT_MUJOCO_RIGHT_HIP_ROLL_BODY = 8,
    REK_G1_HIT_MUJOCO_RIGHT_HIP_YAW_BODY = 9,
    REK_G1_HIT_MUJOCO_RIGHT_KNEE_BODY = 10,
    REK_G1_HIT_MUJOCO_RIGHT_ANKLE_PITCH_BODY = 11,
    REK_G1_HIT_MUJOCO_RIGHT_ANKLE_ROLL_BODY = 12,
    REK_G1_HIT_MUJOCO_WAIST_YAW_BODY = 13,
    REK_G1_HIT_MUJOCO_WAIST_ROLL_BODY = 14,
    REK_G1_HIT_MUJOCO_TORSO_BODY = 15,
    REK_G1_HIT_MUJOCO_LEFT_SHOULDER_PITCH_BODY = 16,
    REK_G1_HIT_MUJOCO_LEFT_SHOULDER_ROLL_BODY = 17,
    REK_G1_HIT_MUJOCO_LEFT_SHOULDER_YAW_BODY = 18,
    REK_G1_HIT_MUJOCO_LEFT_ELBOW_BODY = 19,
    REK_G1_HIT_MUJOCO_LEFT_WRIST_ROLL_BODY = 20,
    REK_G1_HIT_MUJOCO_LEFT_WRIST_PITCH_BODY = 21,
    REK_G1_HIT_MUJOCO_LEFT_WRIST_YAW_BODY = 22,
    REK_G1_HIT_MUJOCO_RIGHT_SHOULDER_PITCH_BODY = 23,
    REK_G1_HIT_MUJOCO_RIGHT_SHOULDER_ROLL_BODY = 24,
    REK_G1_HIT_MUJOCO_RIGHT_SHOULDER_YAW_BODY = 25,
    REK_G1_HIT_MUJOCO_RIGHT_ELBOW_BODY = 26,
    REK_G1_HIT_MUJOCO_RIGHT_WRIST_ROLL_BODY = 27,
    REK_G1_HIT_MUJOCO_RIGHT_WRIST_PITCH_BODY = 28,
    REK_G1_HIT_MUJOCO_RIGHT_WRIST_YAW_BODY = 29,
} RekG1HitMujocoFighterBody;

/*
 * Pure MuJoCo measurements for one directed contact-enter candidate. The
 * left/right values stored in striker_side are HandSide values 0/1 after
 * BodyPartTag.TryGetSide conversion, not serialized BodySide values 1/2.
 */
typedef struct RekG1HitMujocoCandidate {
    size_t arena_index;
    uint32_t physics_substep_index;
    int striker_geom_id;
    int target_geom_id;
    int striker_body_id;
    int target_body_id;
    uint32_t striker_fighter;
    uint32_t target_fighter;
    uint32_t striker_body_slot;
    RekG1BodyPartType striker_part;
    RekG1HandSide striker_side;
    RekG1BodyZone target_zone;
    float striker_body_position_world[3];
    float target_body_position_world[3];
    float striker_body_linear_velocity_world[3];
    float target_body_linear_velocity_world[3];
    float relative_speed_mps;
} RekG1HitMujocoCandidate;

/* Facts measured by other runtime owners. This adapter only validates/copies. */
typedef struct RekG1HitMujocoCallerFacts {
    RekG1StrikeIntent strike_intent;
    float time_seconds;
    uint8_t round_active;
    uint8_t fighter_upright[REK_G1_HIT_FIGHTERS];
    uint8_t fighter_standing[REK_G1_HIT_FIGHTERS];
} RekG1HitMujocoCallerFacts;

typedef struct RekG1HitMujocoAdapter {
    GearSonicNativeDuelVector* duel;
    int body_ids[GEAR_SONIC_DUEL_FIGHTERS]
        [REK_G1_HIT_MUJOCO_FIGHTER_BODY_COUNT];
    int8_t* body_owner;
    RekG1BodyZone* body_zone;
    RekG1BodyZone* geom_zone;
    RekG1BodyPartType* striker_part;
    int8_t* striker_side;
    int8_t* striker_slot;
    uint8_t* previous_pairs;
    uint8_t* current_pairs;
    uint8_t* expected_substep;
    RekG1HitMujocoCandidate* candidate_scratch;
    size_t arena_count;
    size_t body_count;
    size_t geom_count;
    size_t pair_span;
    size_t candidate_scratch_capacity;
    RekG1HitMujocoStatus last_status;
    uint8_t initialized;
    uint8_t ready;
} RekG1HitMujocoAdapter;

const char* rek_g1_hit_mujoco_status_string(RekG1HitMujocoStatus status);

/* The duel/model/mjData remain caller-owned for the adapter lifetime. */
RekG1HitMujocoStatus rek_g1_hit_mujoco_open(
    RekG1HitMujocoAdapter* adapter,
    GearSonicNativeDuelVector* duel,
    char* error,
    size_t error_capacity);

/* Clear every arena's geom-pair history after a caller-owned duel reset. */
RekG1HitMujocoStatus rek_g1_hit_mujoco_reset(
    RekG1HitMujocoAdapter* adapter,
    char* error,
    size_t error_capacity);

/*
 * Clear one arena's contact-enter membership after a physical spawn reset.
 * The physics-substep sequence is intentionally preserved so this is safe
 * between substeps of the same 20 ms controller tick.
 */
RekG1HitMujocoStatus rek_g1_hit_mujoco_clear_arena_contacts(
    RekG1HitMujocoAdapter* adapter,
    size_t arena_index,
    char* error,
    size_t error_capacity);

/*
 * Scan one required post-step observation. Unique pair identity uses sorted
 * geom IDs, while candidates retain raw MuJoCo contact and direction order.
 * Failure changes neither pair history, expected substep, output, nor count.
 */
RekG1HitMujocoStatus rek_g1_hit_mujoco_scan_substep(
    RekG1HitMujocoAdapter* adapter,
    const GearSonicNativeDuelPostStepObservation* observation,
    RekG1HitMujocoCandidate* candidates,
    size_t candidate_capacity,
    size_t* candidate_count,
    char* error,
    size_t error_capacity);

/* Assemble detector input from one candidate and explicit external facts. */
RekG1HitMujocoStatus rek_g1_hit_mujoco_candidate_to_contact(
    const RekG1HitMujocoCandidate* candidate,
    const RekG1HitMujocoCallerFacts* facts,
    RekG1HitContact* contact,
    char* error,
    size_t error_capacity);

void rek_g1_hit_mujoco_close(RekG1HitMujocoAdapter* adapter);
