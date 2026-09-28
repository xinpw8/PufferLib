#include "g1_hit_detector.h"

#include <math.h>
#include <string.h>

static REK_G1_FN int finite_vector3(const float value[3]) {
    return value != NULL && isfinite(value[0])
        && isfinite(value[1]) && isfinite(value[2]);
}

static REK_G1_FN int binary_flag(uint8_t value) {
    return value == 0u || value == 1u;
}

static REK_G1_FN int valid_side(RekG1HandSide side) {
    return side == REK_G1_HAND_LEFT || side == REK_G1_HAND_RIGHT;
}

static REK_G1_FN int kick_striker(RekG1BodyPartType part) {
    return part == REK_G1_BODY_PART_FOOT || part == REK_G1_BODY_PART_SHIN;
}

static REK_G1_FN int valid_striker(RekG1BodyPartType part) {
    return part == REK_G1_BODY_PART_HAND || kick_striker(part);
}

static REK_G1_FN int scoring_zone(RekG1BodyZone zone) {
    return zone == REK_G1_BODY_ZONE_HEAD
        || zone == REK_G1_BODY_ZONE_TORSO
        || zone == REK_G1_BODY_ZONE_PELVIS
        || zone == REK_G1_BODY_ZONE_LEFT_HIP
        || zone == REK_G1_BODY_ZONE_RIGHT_HIP;
}

static REK_G1_FN int limb_matches(
        RekG1AimLimb limb,
        RekG1BodyPartType part,
        RekG1HandSide side) {
    if (!valid_side(side) || limb == REK_G1_AIM_LIMB_NONE) return 0;
    const int limb_is_kick = limb == REK_G1_AIM_LIMB_LEFT_LOWER_BODY
        || limb == REK_G1_AIM_LIMB_RIGHT_LOWER_BODY;
    const RekG1HandSide limb_side =
        limb == REK_G1_AIM_LIMB_LEFT_UPPER_BODY
            || limb == REK_G1_AIM_LIMB_LEFT_LOWER_BODY
        ? REK_G1_HAND_LEFT
        : limb == REK_G1_AIM_LIMB_RIGHT_UPPER_BODY
            || limb == REK_G1_AIM_LIMB_RIGHT_LOWER_BODY
        ? REK_G1_HAND_RIGHT
        : (RekG1HandSide)-1;
    return limb_side == side && limb_is_kick == kick_striker(part);
}

static REK_G1_FN int valid_event(const RekG1ImpactEvent* event) {
    return event != NULL
        && isfinite(event->impact_time_seconds)
        && isfinite(event->lead_time_seconds)
        && isfinite(event->release_time_seconds)
        && event->impact_time_seconds >= 0.0f
        && event->lead_time_seconds >= 0.0f
        && event->release_time_seconds >= 0.0f
        && event->limb >= REK_G1_AIM_LIMB_NONE
        && event->limb <= REK_G1_AIM_LIMB_RIGHT_LOWER_BODY;
}

REK_G1_FN RekG1HitDetectorConfig rek_g1_current_build_hit_detector_config(void) {
    return (RekG1HitDetectorConfig){
        .speed_threshold_mps = 1.75f,
        .knockdown_strike_approach_mps = 2.0f,
        .per_body_cooldown_seconds = 0.30000001192092896f,
        .apex_min_ramp = 0.20000000298023224f,
    };
}

REK_G1_FN void rek_g1_hit_detector_reset(RekG1HitDetectorState* state) {
    if (state != NULL) memset(state, 0, sizeof(*state));
}

REK_G1_FN float rek_g1_impact_event_ramp_at(
        const RekG1ImpactEvent* event,
        float clip_time_seconds) {
    if (!valid_event(event) || !isfinite(clip_time_seconds)) return 0.0f;
    const float delta = clip_time_seconds - event->impact_time_seconds;
    float ramp = 0.0f;
    if (delta > 0.0f) {
        if (event->release_time_seconds <= 0.0f
                || delta > event->release_time_seconds) {
            return 0.0f;
        }
        ramp = 1.0f - delta / event->release_time_seconds;
    } else {
        const float lead = fmaxf(0.0001f, event->lead_time_seconds);
        if (delta < -lead) return 0.0f;
        ramp = delta / lead + 1.0f;
    }
    volatile float rounded_ramp = ramp;
    volatile float doubled = rounded_ramp + rounded_ramp;
    volatile float squared = rounded_ramp * rounded_ramp;
    volatile float factor = 3.0f - doubled;
    return squared * factor;
}

REK_G1_FN int rek_g1_strike_intent_apex(
        const RekG1StrikeIntent* intent,
        RekG1BodyPartType striker_part,
        RekG1HandSide striker_side,
        float minimum_ramp,
        int32_t* apex_event_index_out,
        float* apex_ramp_out) {
    if (apex_event_index_out == NULL || apex_ramp_out == NULL) return 0;
    *apex_event_index_out = -1;
    *apex_ramp_out = 0.0f;
    if (intent == NULL || !isfinite(minimum_ramp) || minimum_ramp < 0.0f
            || !valid_striker(striker_part) || !valid_side(striker_side)
            || !binary_flag(intent->action_playing)
            || !binary_flag(intent->layer_active)
            || !binary_flag(intent->layer_loop)
            || !intent->action_playing || !intent->layer_active
            || intent->layer_loop || intent->impact_events == NULL
            || intent->impact_event_count == 0u
            || !isfinite(intent->clip_cursor_frames)
            || !isfinite(intent->clip_fps)
            || intent->clip_fps <= 0.0f) {
        return 0;
    }
    if (intent->clip_cursor_frames < 0.0f) return 0;
    const double cursor = (double)intent->clip_cursor_frames;
    const double cursor_floor = floor(cursor);
    const double fraction = cursor - cursor_floor;
    double rounded_cursor = cursor_floor;
    if (fraction > 0.5
            || (fraction == 0.5 && fmod(cursor_floor, 2.0) != 0.0)) {
        rounded_cursor += 1.0;
    }
    if (rounded_cursor > (double)INT32_MAX) return 0;
    const int32_t cursor_frame = (int32_t)rounded_cursor;
    const float clip_time_seconds = (float)cursor_frame / intent->clip_fps;
    for (size_t index = 0; index < intent->impact_event_count; index++) {
        const RekG1ImpactEvent* event = &intent->impact_events[index];
        if (!valid_event(event)
                || event->limb == REK_G1_AIM_LIMB_NONE
                || !limb_matches(event->limb, striker_part, striker_side)) {
            continue;
        }
        const float ramp = rek_g1_impact_event_ramp_at(
            event, clip_time_seconds);
        if (ramp >= minimum_ramp) {
            if (index > (size_t)INT32_MAX) return 0;
            *apex_event_index_out = (int32_t)index;
            *apex_ramp_out = ramp;
            return 1;
        }
    }
    return 0;
}

static REK_G1_FN int attribution_accepted(
        const RekG1HitDetectorConfig* config,
        const RekG1HitContact* contact) {
    if (!contact->target_standing) return 0;
    float direction[3];
    float norm_squared = 0.0f;
    for (size_t axis = 0; axis < 3u; axis++) {
        direction[axis] = contact->target_body_position_world[axis]
            - contact->striker_body_position_world[axis];
        norm_squared += direction[axis] * direction[axis];
    }
    if (!(norm_squared > 0.0f) || !isfinite(norm_squared)) return 0;
    const float inverse_norm = 1.0f / sqrtf(norm_squared);
    float striker_approach = 0.0f;
    float target_approach = 0.0f;
    for (size_t axis = 0; axis < 3u; axis++) {
        direction[axis] *= inverse_norm;
        striker_approach +=
            contact->striker_body_linear_velocity_world[axis] * direction[axis];
        target_approach -=
            contact->target_body_linear_velocity_world[axis] * direction[axis];
    }
    return striker_approach >= config->knockdown_strike_approach_mps
        && striker_approach > target_approach;
}

static REK_G1_FN int valid_config(const RekG1HitDetectorConfig* config) {
    return config != NULL
        && isfinite(config->speed_threshold_mps)
        && isfinite(config->knockdown_strike_approach_mps)
        && isfinite(config->per_body_cooldown_seconds)
        && isfinite(config->apex_min_ramp)
        && config->speed_threshold_mps >= 0.0f
        && config->knockdown_strike_approach_mps >= 0.0f
        && config->per_body_cooldown_seconds >= 0.0f
        && config->apex_min_ramp >= 0.0f;
}

static REK_G1_FN int valid_contact(const RekG1HitContact* contact) {
    return contact != NULL
        && contact->striker_fighter < REK_G1_HIT_FIGHTERS
        && contact->target_fighter < REK_G1_HIT_FIGHTERS
        && contact->striker_fighter != contact->target_fighter
        && contact->striker_body_slot < REK_G1_HIT_STRIKER_BODY_SLOTS
        && valid_striker(contact->striker_part)
        && valid_side(contact->striker_side)
        && contact->target_zone >= REK_G1_BODY_ZONE_UNKNOWN
        && contact->target_zone <= REK_G1_BODY_ZONE_RIGHT_ANKLE
        && binary_flag(contact->is_enter)
        && binary_flag(contact->round_active)
        && binary_flag(contact->striker_upright)
        && binary_flag(contact->target_upright)
        && binary_flag(contact->target_standing)
        && isfinite(contact->relative_speed_mps)
        && contact->relative_speed_mps >= 0.0f
        && isfinite(contact->time_seconds)
        && contact->time_seconds >= 0.0f
        && finite_vector3(contact->striker_body_position_world)
        && finite_vector3(contact->target_body_position_world)
        && finite_vector3(contact->striker_body_linear_velocity_world)
        && finite_vector3(contact->target_body_linear_velocity_world);
}

REK_G1_FN int rek_g1_hit_detector_process(
        RekG1HitDetectorState* state,
        const RekG1HitDetectorConfig* config,
        const RekG1HitContact* contact,
        RekG1HitResult* result) {
    if (state == NULL || result == NULL
            || !valid_config(config) || !valid_contact(contact)) {
        return 0;
    }
    *result = (RekG1HitResult){.apex_event_index = -1};
    if (!contact->is_enter
            || contact->relative_speed_mps < config->speed_threshold_mps) {
        return 1;
    }

    result->attribution_accepted = (uint8_t)attribution_accepted(config, contact);
    if (!scoring_zone(contact->target_zone)
            || !contact->round_active
            || !contact->striker_upright
            || !contact->target_upright) {
        return 1;
    }

    int32_t apex_index = -1;
    float apex_ramp = 0.0f;
    if (!rek_g1_strike_intent_apex(
            &contact->strike_intent,
            contact->striker_part,
            contact->striker_side,
            config->apex_min_ramp,
            &apex_index,
            &apex_ramp)) {
        return 1;
    }
    result->apex_event_index = apex_index;
    result->apex_ramp = apex_ramp;

    const uint32_t fighter = contact->striker_fighter;
    const uint32_t slot = contact->striker_body_slot;
    if (state->cooldown_seen[fighter][slot]
            && contact->time_seconds
                - state->last_score_time_seconds[fighter][slot]
                    < config->per_body_cooldown_seconds) {
        return 1;
    }
    const uint32_t bit_index = apex_index > (int32_t)REK_G1_HIT_MAX_APEX_BIT
        ? REK_G1_HIT_MAX_APEX_BIT : (uint32_t)apex_index;
    const uint32_t apex_bit = UINT32_C(1) << bit_index;
    if (state->scored_move_seen[fighter]
            && state->scored_move_id[fighter] == contact->strike_intent.move_id
            && (state->scored_apex_mask[fighter] & apex_bit) != 0u) {
        return 1;
    }

    state->last_score_time_seconds[fighter][slot] = contact->time_seconds;
    state->cooldown_seen[fighter][slot] = 1u;
    if (!state->scored_move_seen[fighter]
            || state->scored_move_id[fighter] != contact->strike_intent.move_id) {
        state->scored_move_id[fighter] = contact->strike_intent.move_id;
        state->scored_apex_mask[fighter] = 0u;
        state->scored_move_seen[fighter] = 1u;
    }
    state->scored_apex_mask[fighter] |= apex_bit;
    result->points_awarded = kick_striker(contact->striker_part) ? 2.0f : 1.0f;
    result->score_accepted = 1u;
    return 1;
}
