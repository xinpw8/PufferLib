#pragma once

#include <stddef.h>
#include <stdint.h>

#include "g1_combat_types.h"

#define REK_G1_HIT_FIGHTERS 2u
#define REK_G1_HIT_STRIKER_BODY_SLOTS 6u
#define REK_G1_HIT_MAX_APEX_BIT 30u

typedef enum RekG1AimLimb {
    REK_G1_AIM_LIMB_NONE = 0,
    REK_G1_AIM_LIMB_LEFT_UPPER_BODY = 1,
    REK_G1_AIM_LIMB_RIGHT_UPPER_BODY = 2,
    REK_G1_AIM_LIMB_LEFT_LOWER_BODY = 3,
    REK_G1_AIM_LIMB_RIGHT_LOWER_BODY = 4,
} RekG1AimLimb;

typedef struct RekG1ImpactEvent {
    float impact_time_seconds;
    float lead_time_seconds;
    float release_time_seconds;
    /* Retained build metadata. Its current-build control effect is unknown. */
    float gain_boost;
    RekG1AimLimb limb;
} RekG1ImpactEvent;

typedef struct RekG1StrikeIntent {
    const RekG1ImpactEvent* impact_events;
    size_t impact_event_count;
    float clip_cursor_frames;
    float clip_fps;
    int32_t move_id;
    uint8_t action_playing;
    uint8_t layer_active;
    uint8_t layer_loop;
} RekG1StrikeIntent;

typedef struct RekG1HitDetectorConfig {
    float speed_threshold_mps;
    float knockdown_strike_approach_mps;
    float per_body_cooldown_seconds;
    float apex_min_ramp;
} RekG1HitDetectorConfig;

typedef struct RekG1HitDetectorState {
    float last_score_time_seconds
        [REK_G1_HIT_FIGHTERS][REK_G1_HIT_STRIKER_BODY_SLOTS];
    int32_t scored_move_id[REK_G1_HIT_FIGHTERS];
    uint32_t scored_apex_mask[REK_G1_HIT_FIGHTERS];
    uint8_t cooldown_seen
        [REK_G1_HIT_FIGHTERS][REK_G1_HIT_STRIKER_BODY_SLOTS];
    uint8_t scored_move_seen[REK_G1_HIT_FIGHTERS];
} RekG1HitDetectorState;

typedef struct RekG1HitContact {
    RekG1StrikeIntent strike_intent;
    float striker_body_position_world[3];
    float target_body_position_world[3];
    float striker_body_linear_velocity_world[3];
    float target_body_linear_velocity_world[3];
    float relative_speed_mps;
    float time_seconds;
    RekG1BodyPartType striker_part;
    RekG1HandSide striker_side;
    RekG1BodyZone target_zone;
    uint32_t striker_fighter;
    uint32_t target_fighter;
    uint32_t striker_body_slot;
    uint8_t is_enter;
    uint8_t round_active;
    uint8_t striker_upright;
    uint8_t target_upright;
    uint8_t target_standing;
} RekG1HitContact;

typedef struct RekG1HitResult {
    float points_awarded;
    int32_t apex_event_index;
    float apex_ramp;
    uint8_t attribution_accepted;
    uint8_t score_accepted;
} RekG1HitResult;

RekG1HitDetectorConfig rek_g1_current_build_hit_detector_config(void);

void rek_g1_hit_detector_reset(RekG1HitDetectorState* state);

float rek_g1_impact_event_ramp_at(
    const RekG1ImpactEvent* event,
    float clip_time_seconds
);

int rek_g1_strike_intent_apex(
    const RekG1StrikeIntent* intent,
    RekG1BodyPartType striker_part,
    RekG1HandSide striker_side,
    float minimum_ramp,
    int32_t* apex_event_index_out,
    float* apex_ramp_out
);

/*
 * Applies the pinned f84f1874 HitDetector acceptance order to one directed
 * contact-enter candidate. Geometry identities and velocities are measured by
 * the caller. On success, state is changed only for an accepted score.
 */
int rek_g1_hit_detector_process(
    RekG1HitDetectorState* state,
    const RekG1HitDetectorConfig* config,
    const RekG1HitContact* contact,
    RekG1HitResult* result
);
