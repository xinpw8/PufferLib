#pragma once

#include <stddef.h>
#include <stdint.h>

#include "g1_fall_state.h"
#include "gear_sonic_native_duel.h"

/*
 * MuJoCo measurement boundary for the build-pinned REK G1 fall detector.
 * Names are resolved and validated once at open. Sampling uses only cached
 * model body/geom IDs and caller-owned live mjData.
 */

typedef enum RekG1FallMujocoStatus {
    REK_G1_FALL_MUJOCO_OK = 0,
    REK_G1_FALL_MUJOCO_NULL_ARGUMENT = 1,
    REK_G1_FALL_MUJOCO_INVALID_DUEL = 2,
    REK_G1_FALL_MUJOCO_MAPPING_MISSING = 3,
    REK_G1_FALL_MUJOCO_MAPPING_MISMATCH = 4,
    REK_G1_FALL_MUJOCO_ALLOCATION_FAILED = 5,
    REK_G1_FALL_MUJOCO_NOT_CALIBRATED = 6,
    REK_G1_FALL_MUJOCO_NON_FINITE = 7,
    REK_G1_FALL_MUJOCO_MEASUREMENT_INVALID = 8,
} RekG1FallMujocoStatus;

typedef struct RekG1FallMujocoCalibration {
    double upright_up_local[3];
    double standing_pelvis_height;
    double reset_floor_height;
    uint8_t calibrated;
} RekG1FallMujocoCalibration;

typedef struct RekG1FallMujocoMeasurement {
    RekG1FallSample fall_sample;
    uint8_t left_foot_body_contact;
    uint8_t right_foot_body_contact;
    float floor_height;
    float standing_pelvis_height;
} RekG1FallMujocoMeasurement;

typedef struct RekG1FallMujocoAdapter {
    GearSonicNativeDuelVector* duel;
    int floor_geom_id;
    int root_body_ids[GEAR_SONIC_DUEL_FIGHTERS];
    int left_foot_body_ids[GEAR_SONIC_DUEL_FIGHTERS];
    int right_foot_body_ids[GEAR_SONIC_DUEL_FIGHTERS];
    int8_t* body_owner;
    uint8_t* contact_seen;
    RekG1FallMujocoCalibration* calibrations;
    size_t body_count;
    size_t robot_count;
    RekG1FallMujocoStatus last_status;
    uint8_t initialized;
    uint8_t ready;
} RekG1FallMujocoAdapter;

const char* rek_g1_fall_mujoco_status_string(
    RekG1FallMujocoStatus status);

/*
 * Validate the exact two-fighter arena mapping and allocate fixed scratch
 * storage. The duel and its model/data remain caller-owned.
 */
RekG1FallMujocoStatus rek_g1_fall_mujoco_open(
    RekG1FallMujocoAdapter* adapter,
    GearSonicNativeDuelVector* duel,
    char* error,
    size_t error_capacity);

/*
 * Calibrate uprightUpLocal and standing pelvis height from every reset row.
 * The floor height is measured from the exact fixed floor box. Failure clears
 * every row's calibrated flag so later sampling fails closed.
 */
RekG1FallMujocoStatus rek_g1_fall_mujoco_calibrate_reset(
    RekG1FallMujocoAdapter* adapter,
    char* error,
    size_t error_capacity);

/*
 * Sample one arena-major fighter row. can_get_up is an explicit caller fact;
 * the adapter does not assume a policy implementation. On failure, output is
 * not modified.
 */
RekG1FallMujocoStatus rek_g1_fall_mujoco_sample(
    RekG1FallMujocoAdapter* adapter,
    size_t robot_row,
    float fixed_delta_seconds,
    uint8_t can_get_up,
    RekG1FallMujocoMeasurement* output,
    char* error,
    size_t error_capacity);

void rek_g1_fall_mujoco_close(RekG1FallMujocoAdapter* adapter);
