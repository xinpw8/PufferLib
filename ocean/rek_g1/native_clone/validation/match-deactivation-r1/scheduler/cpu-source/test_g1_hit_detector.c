#include "g1_hit_detector.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int checks = 0;

static void require(int condition, const char* message) {
    checks++;
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        exit(1);
    }
}

static void near(float actual, float expected, float tolerance, const char* message) {
    require(isfinite(actual) && fabsf(actual - expected) <= tolerance, message);
}

static RekG1HitContact base_contact(
        const RekG1ImpactEvent* event,
        RekG1BodyPartType part,
        RekG1HandSide side) {
    return (RekG1HitContact){
        .strike_intent = {
            .impact_events = event,
            .impact_event_count = 1u,
            .clip_cursor_frames = 50.0f,
            .clip_fps = 50.0f,
            .move_id = 7,
            .action_playing = 1u,
            .layer_active = 1u,
            .layer_loop = 0u,
        },
        .striker_body_position_world = {0.0f, 0.0f, 0.0f},
        .target_body_position_world = {1.0f, 0.0f, 0.0f},
        .striker_body_linear_velocity_world = {3.0f, 0.0f, 0.0f},
        .target_body_linear_velocity_world = {0.0f, 0.0f, 0.0f},
        .relative_speed_mps = 3.0f,
        .time_seconds = 1.0f,
        .striker_part = part,
        .striker_side = side,
        .target_zone = REK_G1_BODY_ZONE_TORSO,
        .striker_fighter = 0u,
        .target_fighter = 1u,
        .striker_body_slot = 0u,
        .is_enter = 1u,
        .round_active = 1u,
        .striker_upright = 1u,
        .target_upright = 1u,
        .target_standing = 1u,
    };
}

static void test_current_config(void) {
    const RekG1HitDetectorConfig config =
        rek_g1_current_build_hit_detector_config();
    near(config.speed_threshold_mps, 1.75f, 0.0f, "speed threshold");
    near(config.knockdown_strike_approach_mps, 2.0f, 0.0f, "approach threshold");
    near(config.per_body_cooldown_seconds, 0.30000001192092896f, 0.0f, "cooldown");
    near(config.apex_min_ramp, 0.20000000298023224f, 0.0f, "apex ramp");
}

static void test_ramp(void) {
    const RekG1ImpactEvent event = {
        .impact_time_seconds = 1.0f,
        .lead_time_seconds = 0.2f,
        .release_time_seconds = 0.5f,
        .limb = REK_G1_AIM_LIMB_LEFT_LOWER_BODY,
    };
    near(rek_g1_impact_event_ramp_at(&event, 0.79f), 0.0f, 0.0f, "before lead");
    near(rek_g1_impact_event_ramp_at(&event, 0.8f), 0.0f, 1e-6f, "lead start");
    near(rek_g1_impact_event_ramp_at(&event, 0.9f), 0.5f, 1e-6f, "lead midpoint");
    near(rek_g1_impact_event_ramp_at(&event, 1.0f), 1.0f, 0.0f, "at impact");
    near(rek_g1_impact_event_ramp_at(&event, 1.25f), 0.5f, 1e-6f, "release midpoint");
    near(rek_g1_impact_event_ramp_at(&event, 1.5f), 0.0f, 1e-6f, "release end");
    near(rek_g1_impact_event_ramp_at(&event, 1.51f), 0.0f, 0.0f, "after release");

    RekG1ImpactEvent zero_lead = event;
    zero_lead.lead_time_seconds = 0.0f;
    near(rek_g1_impact_event_ramp_at(&zero_lead, 0.999f), 0.0f, 0.0f, "zero lead before");
    near(rek_g1_impact_event_ramp_at(&zero_lead, 1.0f), 1.0f, 0.0f, "zero lead apex");
}

static void test_intent_rounding_and_limb(void) {
    const RekG1ImpactEvent events[] = {
        {.impact_time_seconds = 1.0f, .lead_time_seconds = 0.2f,
         .release_time_seconds = 0.5f, .limb = REK_G1_AIM_LIMB_LEFT_LOWER_BODY},
        {.impact_time_seconds = 1.0f, .lead_time_seconds = 0.2f,
         .release_time_seconds = 0.5f, .limb = REK_G1_AIM_LIMB_RIGHT_UPPER_BODY},
    };
    RekG1StrikeIntent intent = {
        .impact_events = events,
        .impact_event_count = 2u,
        .clip_cursor_frames = 49.51f,
        .clip_fps = 50.0f,
        .move_id = 8,
        .action_playing = 1u,
        .layer_active = 1u,
    };
    int32_t index = -1;
    float ramp = 0.0f;
    require(rek_g1_strike_intent_apex(
        &intent, REK_G1_BODY_PART_SHIN, REK_G1_HAND_LEFT,
        1.0f, &index, &ramp), "rounded cursor apex");
    require(index == 0, "left kick event index");
    near(ramp, 1.0f, 0.0f, "rounded cursor ramp");
    require(rek_g1_strike_intent_apex(
        &intent, REK_G1_BODY_PART_HAND, REK_G1_HAND_RIGHT,
        1.0f, &index, &ramp), "right hand match");
    require(index == 1, "right hand event index");
    require(!rek_g1_strike_intent_apex(
        &intent, REK_G1_BODY_PART_FOOT, REK_G1_HAND_RIGHT,
        0.2f, &index, &ramp), "wrong side rejected");
    intent.clip_cursor_frames = 50.5f;
    require(rek_g1_strike_intent_apex(
        &intent, REK_G1_BODY_PART_SHIN, REK_G1_HAND_LEFT,
        1.0f, &index, &ramp), "half-even rounds even frame down");
    intent.clip_cursor_frames = 51.5f;
    require(!rek_g1_strike_intent_apex(
        &intent, REK_G1_BODY_PART_SHIN, REK_G1_HAND_LEFT,
        1.0f, &index, &ramp), "half-even rounds odd frame up");
    intent.clip_cursor_frames = 49.51f;
    intent.layer_loop = 1u;
    require(!rek_g1_strike_intent_apex(
        &intent, REK_G1_BODY_PART_SHIN, REK_G1_HAND_LEFT,
        0.2f, &index, &ramp), "loop rejected");
}

static void test_move_7_frame_window(void) {
    const RekG1HitDetectorConfig config =
        rek_g1_current_build_hit_detector_config();
    const RekG1ImpactEvent event = {
        .impact_time_seconds = 1.0f,
        .lead_time_seconds = 0.2f,
        .release_time_seconds = 0.5f,
        .limb = REK_G1_AIM_LIMB_LEFT_LOWER_BODY,
    };
    const struct {
        float clip_frame;
        uint8_t score_expected;
        const char* process_message;
        const char* score_message;
    } cases[] = {
        {42.0f, 0u, "move 7 frame 42 process", "move 7 frame 42 rejected"},
        {43.0f, 1u, "move 7 frame 43 process", "move 7 frame 43 accepted"},
        {67.0f, 1u, "move 7 frame 67 process", "move 7 frame 67 accepted"},
        {68.0f, 0u, "move 7 frame 68 process", "move 7 frame 68 rejected"},
    };
    for (size_t index = 0u; index < sizeof(cases) / sizeof(cases[0]); index++) {
        RekG1HitDetectorState state;
        rek_g1_hit_detector_reset(&state);
        RekG1HitContact contact = base_contact(
            &event, REK_G1_BODY_PART_FOOT, REK_G1_HAND_LEFT);
        contact.strike_intent.clip_cursor_frames = cases[index].clip_frame;
        contact.time_seconds = cases[index].clip_frame / 50.0f;
        RekG1HitResult result;
        require(rek_g1_hit_detector_process(
            &state, &config, &contact, &result), cases[index].process_message);
        require(result.score_accepted == cases[index].score_expected,
            cases[index].score_message);
    }
}

static void test_acceptance_order_and_state(void) {
    const RekG1HitDetectorConfig config =
        rek_g1_current_build_hit_detector_config();
    const RekG1ImpactEvent event = {
        .impact_time_seconds = 1.0f,
        .lead_time_seconds = 0.2f,
        .release_time_seconds = 0.5f,
        .limb = REK_G1_AIM_LIMB_LEFT_LOWER_BODY,
    };
    RekG1HitDetectorState state;
    rek_g1_hit_detector_reset(&state);
    RekG1HitContact contact = base_contact(
        &event, REK_G1_BODY_PART_FOOT, REK_G1_HAND_LEFT);
    RekG1HitResult result;

    require(rek_g1_hit_detector_process(&state, &config, &contact, &result), "first process");
    require(result.attribution_accepted, "attribution accepted");
    require(result.score_accepted, "kick score accepted");
    near(result.points_awarded, 2.0f, 0.0f, "kick points");
    require(result.apex_event_index == 0, "kick apex index");

    contact.time_seconds = 1.4f;
    require(rek_g1_hit_detector_process(&state, &config, &contact, &result), "once process");
    require(!result.score_accepted, "same move apex once");
    require(result.attribution_accepted, "attribution survives once rejection");

    contact.strike_intent.move_id = 8;
    contact.time_seconds = 1.299f;
    require(rek_g1_hit_detector_process(&state, &config, &contact, &result), "cooldown process");
    require(!result.score_accepted, "cooldown rejects before equality");
    contact.time_seconds = 1.30000001192092896f;
    require(rek_g1_hit_detector_process(&state, &config, &contact, &result), "cooldown boundary process");
    require(result.score_accepted, "cooldown equality accepts");

    RekG1HitDetectorState untouched = state;
    contact.target_zone = REK_G1_BODY_ZONE_LEFT_KNEE;
    contact.strike_intent.move_id = 9;
    contact.time_seconds = 2.0f;
    require(rek_g1_hit_detector_process(&state, &config, &contact, &result), "nonscoring zone process");
    require(result.attribution_accepted, "nonscoring contact attributes fall");
    require(!result.score_accepted, "nonscoring zone rejected");
    require(state.scored_move_id[0] == untouched.scored_move_id[0], "nonscore state unchanged");

    contact.target_zone = REK_G1_BODY_ZONE_TORSO;
    contact.relative_speed_mps = 1.749f;
    require(rek_g1_hit_detector_process(&state, &config, &contact, &result), "slow process");
    require(!result.attribution_accepted && !result.score_accepted, "slow rejected before attribution");

    contact.relative_speed_mps = 3.0f;
    contact.target_body_position_world[0] = 0.0f;
    require(rek_g1_hit_detector_process(&state, &config, &contact, &result), "zero separation process");
    require(!result.attribution_accepted, "zero separation attribution rejected");
    require(result.score_accepted, "zero separation does not reject score");
}

static void test_hand_points_and_qualifiers(void) {
    const RekG1HitDetectorConfig config =
        rek_g1_current_build_hit_detector_config();
    const RekG1ImpactEvent event = {
        .impact_time_seconds = 1.0f,
        .lead_time_seconds = 0.2f,
        .release_time_seconds = 0.5f,
        .limb = REK_G1_AIM_LIMB_RIGHT_UPPER_BODY,
    };
    RekG1HitDetectorState state;
    rek_g1_hit_detector_reset(&state);
    RekG1HitContact contact = base_contact(
        &event, REK_G1_BODY_PART_HAND, REK_G1_HAND_RIGHT);
    contact.striker_body_slot = 5u;
    RekG1HitResult result;
    require(rek_g1_hit_detector_process(&state, &config, &contact, &result), "hand process");
    require(result.score_accepted, "hand accepted");
    near(result.points_awarded, 1.0f, 0.0f, "hand points");

    rek_g1_hit_detector_reset(&state);
    contact.round_active = 0u;
    require(rek_g1_hit_detector_process(&state, &config, &contact, &result), "inactive process");
    require(result.attribution_accepted, "inactive round still attribution candidate");
    require(!result.score_accepted, "inactive round no score");
    contact.round_active = 1u;
    contact.target_upright = 0u;
    require(rek_g1_hit_detector_process(&state, &config, &contact, &result), "fallen target process");
    require(!result.score_accepted, "both upright required");
    contact.target_upright = 1u;
    contact.target_standing = 0u;
    require(rek_g1_hit_detector_process(&state, &config, &contact, &result), "not standing process");
    require(!result.attribution_accepted && result.score_accepted,
        "standing attribution and upright score are separate");
}

static void test_invalid_input(void) {
    RekG1HitDetectorState state;
    rek_g1_hit_detector_reset(&state);
    RekG1HitDetectorConfig config = rek_g1_current_build_hit_detector_config();
    const RekG1ImpactEvent event = {
        .impact_time_seconds = 1.0f,
        .lead_time_seconds = 0.2f,
        .release_time_seconds = 0.5f,
        .limb = REK_G1_AIM_LIMB_LEFT_LOWER_BODY,
    };
    RekG1HitContact contact = base_contact(
        &event, REK_G1_BODY_PART_FOOT, REK_G1_HAND_LEFT);
    RekG1HitResult result;
    config.speed_threshold_mps = NAN;
    require(!rek_g1_hit_detector_process(&state, &config, &contact, &result), "nan config rejected");
    config = rek_g1_current_build_hit_detector_config();
    contact.striker_body_slot = REK_G1_HIT_STRIKER_BODY_SLOTS;
    require(!rek_g1_hit_detector_process(&state, &config, &contact, &result), "bad slot rejected");
    contact.striker_body_slot = 0u;
    contact.relative_speed_mps = NAN;
    require(!rek_g1_hit_detector_process(&state, &config, &contact, &result), "nan contact rejected");
}

int main(void) {
    test_current_config();
    test_ramp();
    test_intent_rounding_and_limb();
    test_move_7_frame_window();
    test_acceptance_order_and_state();
    test_hand_points_and_qualifiers();
    test_invalid_input();
    printf("PASS g1_hit_detector checks=%d\n", checks);
    return 0;
}
