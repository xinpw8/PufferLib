#include "g1_strike_catalog.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t checks = 0u;

static void require(int condition, const char* message) {
    checks++;
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        exit(1);
    }
}

static uint32_t f32_bits(float value) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static RekG1StrikeComposerSnapshot valid_snapshot(
        RekG1NativeRouteId route_id,
        int32_t action_move_id) {
    return (RekG1StrikeComposerSnapshot){
        .active_route_id = route_id,
        .clip_cursor_frames = 50.0f,
        .clip_fps = 50.0f,
        .action_move_id = action_move_id,
        .action_playing = 1u,
        .current_layer_has_clip = 1u,
        .current_layer_has_config = 1u,
        .current_layer_active = 1u,
        .current_layer_loop = 0u,
    };
}

static void require_cleared(
        const RekG1StrikeIntent* intent,
        const char* message) {
    RekG1StrikeIntent zero;
    memset(&zero, 0, sizeof(zero));
    require(memcmp(intent, &zero, sizeof(zero)) == 0, message);
}

static void expect_snapshot_rejected(
        const RekG1StrikeCatalog* catalog,
        const RekG1StrikeComposerSnapshot* snapshot,
        const char* message) {
    RekG1StrikeIntent intent;
    memset(&intent, 0xA5, sizeof(intent));
    require(
        !rek_g1_strike_intent_from_snapshot(catalog, snapshot, &intent),
        message);
    require_cleared(&intent, "rejected snapshot clears intent");
}

static const RekG1NativeRouteId EXPECTED_ROUTES[] = {
    REK_G1_NATIVE_MOVE_6_LEFT_SIDE,
    REK_G1_NATIVE_MOVE_7_LEFT_FRONT,
    REK_G1_NATIVE_MOVE_8_RIGHT_SIDE,
    REK_G1_NATIVE_MOVE_9_RIGHT_KNEE,
    REK_G1_NATIVE_MOVE_0_LEFT_HOOK,
    REK_G1_NATIVE_MOVE_1_LEFT_JAB,
    REK_G1_NATIVE_MOVE_2_DOUBLE_UPPERCUT,
    REK_G1_NATIVE_MOVE_3_RIGHT_HOOK,
    REK_G1_NATIVE_MOVE_4_RIGHT_JAB,
    REK_G1_NATIVE_MOVE_5_LEFT_JAB_RIGHT_UPPERCUT,
    REK_G1_NATIVE_MOVE_10_SIX_PUNCH,
    REK_G1_NATIVE_MOVE_11_RUN_AND_PUNCH,
    REK_G1_NATIVE_MOVE_12_LEFT_RIGHT_JAB,
    REK_G1_NATIVE_MOVE_13_LEFT_RIGHT_HOOK,
    REK_G1_NATIVE_MOVE_14_LEFT_HOOK_RIGHT_JAB,
    REK_G1_NATIVE_MOVE_15_DOUBLE_HOOK,
    REK_G1_NATIVE_MOVE_16_BUTT_SMACK_EMOTE,
};

static const uint16_t EXPECTED_MOVE_INDICES[] = {
    6u, 7u, 8u, 9u, 0u, 1u, 2u, 3u, 4u,
    5u, 10u, 11u, 12u, 13u, 14u, 15u, 16u,
};

static const uint16_t EXPECTED_EVENT_OFFSETS[] = {
    0u, 1u, 2u, 3u, 4u, 5u, 6u, 8u, 9u,
    10u, 12u, 22u, 23u, 24u, 25u, 26u, 28u,
};

static const uint16_t EXPECTED_EVENT_COUNTS[] = {
    1u, 1u, 1u, 1u, 1u, 1u, 2u, 1u, 1u,
    2u, 10u, 1u, 1u, 1u, 1u, 2u, 1u,
};

static const RekG1ImpactEvent EXPECTED_EVENTS[] = {
    {1.100000023841858f, 0.30000001192092896f, 0.5f, 2.0f,
     REK_G1_AIM_LIMB_LEFT_LOWER_BODY},
    {1.0f, 0.20000000298023224f, 0.5f, 2.0f,
     REK_G1_AIM_LIMB_LEFT_LOWER_BODY},
    {1.1399999856948853f, 0.4000000059604645f, 0.15000000596046448f,
     2.0f, REK_G1_AIM_LIMB_RIGHT_LOWER_BODY},
    {0.6499999761581421f, 0.4000000059604645f, 0.15000000596046448f,
     3.0f, REK_G1_AIM_LIMB_RIGHT_LOWER_BODY},
    {0.3799999952316284f, 0.25f, 0.25f, 8.0f,
     REK_G1_AIM_LIMB_LEFT_UPPER_BODY},
    {0.1850000023841858f, 0.20000000298023224f, 0.33000001311302185f,
     8.0f, REK_G1_AIM_LIMB_LEFT_UPPER_BODY},
    {0.3959999978542328f, 0.20000000298023224f, 0.4000000059604645f,
     8.0f, REK_G1_AIM_LIMB_LEFT_UPPER_BODY},
    {0.7350000143051147f, 0.20000000298023224f, 0.4000000059604645f,
     8.0f, REK_G1_AIM_LIMB_RIGHT_UPPER_BODY},
    {0.49000000953674316f, 0.30000001192092896f, 0.30000001192092896f,
     8.0f, REK_G1_AIM_LIMB_RIGHT_UPPER_BODY},
    {0.20000000298023224f, 0.20000000298023224f, 0.30000001192092896f,
     8.0f, REK_G1_AIM_LIMB_RIGHT_UPPER_BODY},
    {0.30000001192092896f, 0.20000000298023224f, 0.0f, 4.0f,
     REK_G1_AIM_LIMB_LEFT_UPPER_BODY},
    {0.5799999833106995f, 0.2199999988079071f, 0.6899999976158142f,
     8.0f, REK_G1_AIM_LIMB_RIGHT_UPPER_BODY},
    {0.16099999845027924f, 0.20000000298023224f, 0.10000000149011612f,
     8.0f, REK_G1_AIM_LIMB_LEFT_UPPER_BODY},
    {0.3529999852180481f, 0.20000000298023224f, 0.10000000149011612f,
     8.0f, REK_G1_AIM_LIMB_RIGHT_UPPER_BODY},
    {0.5879999995231628f, 0.20000000298023224f, 0.10000000149011612f,
     8.0f, REK_G1_AIM_LIMB_LEFT_UPPER_BODY},
    {0.8169999718666077f, 0.20000000298023224f, 0.10000000149011612f,
     8.0f, REK_G1_AIM_LIMB_RIGHT_UPPER_BODY},
    {1.0f, 0.20000000298023224f, 0.10000000149011612f, 8.0f,
     REK_G1_AIM_LIMB_LEFT_UPPER_BODY},
    {1.2599999904632568f, 0.20000000298023224f, 0.10000000149011612f,
     8.0f, REK_G1_AIM_LIMB_RIGHT_UPPER_BODY},
    {1.5700000524520874f, 0.20000000298023224f, 0.10000000149011612f,
     8.0f, REK_G1_AIM_LIMB_LEFT_UPPER_BODY},
    {1.7000000476837158f, 0.20000000298023224f, 0.10000000149011612f,
     8.0f, REK_G1_AIM_LIMB_RIGHT_UPPER_BODY},
    {2.0f, 0.20000000298023224f, 0.10000000149011612f, 8.0f,
     REK_G1_AIM_LIMB_LEFT_UPPER_BODY},
    {2.3429999351501465f, 0.20000000298023224f, 0.10000000149011612f,
     8.0f, REK_G1_AIM_LIMB_RIGHT_UPPER_BODY},
    {1.7999999523162842f, 0.4000000059604645f, 0.15000000596046448f,
     8.0f, REK_G1_AIM_LIMB_RIGHT_UPPER_BODY},
    {0.30000001192092896f, 0.4000000059604645f, 0.15000000596046448f,
     2.0f, REK_G1_AIM_LIMB_LEFT_UPPER_BODY},
    {0.30000001192092896f, 0.4000000059604645f, 0.15000000596046448f,
     2.0f, REK_G1_AIM_LIMB_LEFT_UPPER_BODY},
    {0.30000001192092896f, 0.4000000059604645f, 0.15000000596046448f,
     2.0f, REK_G1_AIM_LIMB_LEFT_UPPER_BODY},
    {0.5960000157356262f, 0.4000000059604645f, 0.15000000596046448f,
     8.0f, REK_G1_AIM_LIMB_LEFT_UPPER_BODY},
    {0.8429999947547913f, 0.4000000059604645f, 0.15000000596046448f,
     8.0f, REK_G1_AIM_LIMB_RIGHT_UPPER_BODY},
    {0.14000000059604645f, 0.4300000071525574f, 11.600000381469727f,
     5.0f, REK_G1_AIM_LIMB_LEFT_UPPER_BODY},
};

static void test_catalog_identity_and_values(void) {
    const RekG1StrikeCatalog* catalog =
        rek_g1_current_build_strike_catalog();
    require(catalog != NULL, "current-build catalog exists");
    require(rek_g1_validate_strike_catalog(catalog), "catalog validates");
    require(catalog->count == REK_G1_STRIKE_CATALOG_ENTRY_COUNT,
        "all 17 discrete moves are cataloged");
    require(catalog->impact_event_count
        == REK_G1_STRIKE_CATALOG_IMPACT_EVENT_COUNT,
        "all 29 recovered impact events are cataloged");
    require(strcmp(
        catalog->build_fingerprint,
        REK_G1_STRIKE_CATALOG_BUILD_FINGERPRINT) == 0,
        "catalog build fingerprint");
    require(strcmp(
        catalog->source_sha256,
        REK_G1_STRIKE_CATALOG_SOURCE_SHA256) == 0,
        "catalog source hash");

    for (size_t index = 0; index < REK_G1_STRIKE_CATALOG_ENTRY_COUNT;
            index++) {
        const RekG1StrikeCatalogEntry* entry =
            rek_g1_strike_catalog_entry_by_route(
                catalog, EXPECTED_ROUTES[index]);
        require(entry == &catalog->entries[index],
            "route lookup returns exact catalog row");
        require(entry->runtime_move_index == EXPECTED_MOVE_INDICES[index],
            "runtime move index matches recovered route");
        require(entry->impact_event_offset == EXPECTED_EVENT_OFFSETS[index],
            "impact-event offset matches recovered route");
        require(entry->impact_event_count == EXPECTED_EVENT_COUNTS[index],
            "impact-event count matches recovered route");
    }

    for (size_t index = 0; index < REK_G1_STRIKE_CATALOG_IMPACT_EVENT_COUNT;
            index++) {
        const RekG1ImpactEvent* actual = &catalog->impact_events[index];
        const RekG1ImpactEvent* expected = &EXPECTED_EVENTS[index];
        require(f32_bits(actual->impact_time_seconds)
            == f32_bits(expected->impact_time_seconds),
            "impact time exact binary32 value");
        require(f32_bits(actual->lead_time_seconds)
            == f32_bits(expected->lead_time_seconds),
            "lead time exact binary32 value");
        require(f32_bits(actual->release_time_seconds)
            == f32_bits(expected->release_time_seconds),
            "release time exact binary32 value");
        require(f32_bits(actual->gain_boost)
            == f32_bits(expected->gain_boost),
            "gain boost exact binary32 value");
        require(actual->limb == expected->limb,
            "impact limb exact recovered value");
    }

    for (int32_t route = (int32_t)REK_G1_NATIVE_IDLE;
            route <= (int32_t)REK_G1_NATIVE_TURN_RIGHT; route++) {
        require(rek_g1_strike_catalog_entry_by_route(
            catalog, (RekG1NativeRouteId)route) == NULL,
            "locomotion route fails closed");
    }
    require(rek_g1_strike_catalog_entry_by_route(
        catalog, (RekG1NativeRouteId)-1) == NULL,
        "negative route fails closed");
    require(rek_g1_strike_catalog_entry_by_route(
        catalog, (RekG1NativeRouteId)REK_G1_STATIC_ROUTE_COUNT) == NULL,
        "out-of-range route fails closed");
}

static void reset_mutable_catalog(
        RekG1StrikeCatalog* catalog,
        RekG1StrikeCatalogEntry* entries,
        RekG1ImpactEvent* events) {
    const RekG1StrikeCatalog* current =
        rek_g1_current_build_strike_catalog();
    *catalog = *current;
    memcpy(entries, current->entries,
        sizeof(*entries) * REK_G1_STRIKE_CATALOG_ENTRY_COUNT);
    memcpy(events, current->impact_events,
        sizeof(*events) * REK_G1_STRIKE_CATALOG_IMPACT_EVENT_COUNT);
    catalog->entries = entries;
    catalog->impact_events = events;
}

static void test_catalog_validation_rejects_mutation(void) {
    RekG1StrikeCatalog catalog;
    RekG1StrikeCatalogEntry entries[REK_G1_STRIKE_CATALOG_ENTRY_COUNT];
    RekG1ImpactEvent events[REK_G1_STRIKE_CATALOG_IMPACT_EVENT_COUNT];
    reset_mutable_catalog(&catalog, entries, events);
    require(rek_g1_validate_strike_catalog(&catalog),
        "copied catalog validates");

    catalog.build_fingerprint = "different";
    require(!rek_g1_validate_strike_catalog(&catalog),
        "wrong build rejected");
    reset_mutable_catalog(&catalog, entries, events);
    catalog.source_sha256 = "different";
    require(!rek_g1_validate_strike_catalog(&catalog),
        "wrong source rejected");
    reset_mutable_catalog(&catalog, entries, events);
    catalog.count--;
    require(!rek_g1_validate_strike_catalog(&catalog),
        "short entry catalog rejected");
    reset_mutable_catalog(&catalog, entries, events);
    catalog.impact_event_count--;
    require(!rek_g1_validate_strike_catalog(&catalog),
        "short event catalog rejected");
    reset_mutable_catalog(&catalog, entries, events);
    catalog.entries = NULL;
    require(!rek_g1_validate_strike_catalog(&catalog),
        "null entries rejected");
    reset_mutable_catalog(&catalog, entries, events);
    catalog.impact_events = NULL;
    require(!rek_g1_validate_strike_catalog(&catalog),
        "null events rejected");
    require(!rek_g1_validate_strike_catalog(NULL), "null catalog rejected");

    reset_mutable_catalog(&catalog, entries, events);
    entries[0].route_id = REK_G1_NATIVE_IDLE;
    require(!rek_g1_validate_strike_catalog(&catalog),
        "mutated route rejected");
    reset_mutable_catalog(&catalog, entries, events);
    entries[1].runtime_move_index = 9u;
    require(!rek_g1_validate_strike_catalog(&catalog),
        "mutated move index rejected");
    reset_mutable_catalog(&catalog, entries, events);
    entries[2].impact_event_offset++;
    require(!rek_g1_validate_strike_catalog(&catalog),
        "noncontiguous event offset rejected");
    reset_mutable_catalog(&catalog, entries, events);
    entries[3].impact_event_count = 0u;
    require(!rek_g1_validate_strike_catalog(&catalog),
        "empty move event list rejected");
    reset_mutable_catalog(&catalog, entries, events);
    entries[10].impact_event_count++;
    require(!rek_g1_validate_strike_catalog(&catalog),
        "oversized move event list rejected");

    reset_mutable_catalog(&catalog, entries, events);
    events[2].impact_time_seconds = 1.0f;
    require(!rek_g1_validate_strike_catalog(&catalog),
        "mutated impact time rejected");
    reset_mutable_catalog(&catalog, entries, events);
    events[3].lead_time_seconds = NAN;
    require(!rek_g1_validate_strike_catalog(&catalog),
        "non-finite lead rejected");
    reset_mutable_catalog(&catalog, entries, events);
    events[0].release_time_seconds = -0.5f;
    require(!rek_g1_validate_strike_catalog(&catalog),
        "negative release rejected");
    reset_mutable_catalog(&catalog, entries, events);
    events[10].release_time_seconds = -0.0f;
    require(!rek_g1_validate_strike_catalog(&catalog),
        "signed-zero release mutation rejected");
    reset_mutable_catalog(&catalog, entries, events);
    events[0].gain_boost = NAN;
    require(!rek_g1_validate_strike_catalog(&catalog),
        "non-finite gain boost rejected");
    reset_mutable_catalog(&catalog, entries, events);
    events[0].limb = REK_G1_AIM_LIMB_RIGHT_LOWER_BODY;
    require(!rek_g1_validate_strike_catalog(&catalog),
        "mutated limb rejected");
}

static void test_intent_assembly(void) {
    const RekG1StrikeCatalog* catalog =
        rek_g1_current_build_strike_catalog();
    for (size_t index = 0; index < REK_G1_STRIKE_CATALOG_ENTRY_COUNT;
            index++) {
        const int32_t invocation_id = 101 + (int32_t)index;
        const RekG1StrikeComposerSnapshot snapshot = valid_snapshot(
            EXPECTED_ROUTES[index], invocation_id);
        RekG1StrikeIntent intent;
        memset(&intent, 0, sizeof(intent));
        require(rek_g1_strike_intent_from_snapshot(
            catalog, &snapshot, &intent),
            "active discrete move assembles intent");
        const RekG1StrikeCatalogEntry* entry = &catalog->entries[index];
        require(intent.impact_events
            == catalog->impact_events + entry->impact_event_offset,
            "intent points to exact immutable event slice");
        require(intent.impact_event_count == entry->impact_event_count,
            "intent exposes the complete move event slice");
        require(intent.clip_cursor_frames == snapshot.clip_cursor_frames,
            "cursor copied exactly");
        require(intent.clip_fps == snapshot.clip_fps,
            "fps copied exactly");
        require(intent.move_id == invocation_id,
            "composer invocation identity copied exactly");
        require(intent.action_playing == 1u, "playing flag copied");
        require(intent.layer_active == 1u, "active flag copied");
        require(intent.layer_loop == 0u, "non-loop flag copied");
    }

    RekG1StrikeComposerSnapshot snapshot = valid_snapshot(
        REK_G1_NATIVE_MOVE_7_LEFT_FRONT, INT32_MAX);
    RekG1StrikeIntent intent;
    require(rek_g1_strike_intent_from_snapshot(
        catalog, &snapshot, &intent),
        "positive invocation counter is independent of move index");
    int32_t apex = -1;
    float ramp = 0.0f;
    require(rek_g1_strike_intent_apex(
        &intent, REK_G1_BODY_PART_FOOT, REK_G1_HAND_LEFT,
        1.0f, &apex, &ramp),
        "assembled kick intent reaches exact apex");
    require(apex == 0, "single-event kick has apex index zero");
    require(ramp == 1.0f, "assembled kick apex ramp is one");

    static const int32_t wrapped_invocation_ids[] = {0, -1, INT32_MIN};
    for (size_t index = 0;
            index < sizeof(wrapped_invocation_ids)
                / sizeof(wrapped_invocation_ids[0]);
            index++) {
        snapshot = valid_snapshot(
            REK_G1_NATIVE_MOVE_6_LEFT_SIDE,
            wrapped_invocation_ids[index]);
        require(rek_g1_strike_intent_from_snapshot(
            catalog, &snapshot, &intent),
            "wrapped invocation counter remains valid");
        require(intent.move_id == wrapped_invocation_ids[index],
            "wrapped invocation identity copied exactly");
    }
}

static void test_intent_fail_closed(void) {
    const RekG1StrikeCatalog* catalog =
        rek_g1_current_build_strike_catalog();
    RekG1StrikeComposerSnapshot snapshot = valid_snapshot(
        REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    expect_snapshot_rejected(catalog, NULL, "missing snapshot rejected");
    expect_snapshot_rejected(NULL, &snapshot, "missing catalog rejected");
    for (int32_t route = (int32_t)REK_G1_NATIVE_IDLE;
            route <= (int32_t)REK_G1_NATIVE_TURN_RIGHT; route++) {
        snapshot = valid_snapshot((RekG1NativeRouteId)route, 1);
        expect_snapshot_rejected(catalog, &snapshot,
            "locomotion snapshot rejected");
    }
    snapshot = valid_snapshot(
        (RekG1NativeRouteId)REK_G1_STATIC_ROUTE_COUNT, 1);
    expect_snapshot_rejected(catalog, &snapshot,
        "out-of-range route rejected");

    snapshot = valid_snapshot(REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    snapshot.action_playing = 0u;
    expect_snapshot_rejected(catalog, &snapshot,
        "non-playing action rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_has_clip = 0u;
    expect_snapshot_rejected(catalog, &snapshot, "missing clip rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_has_config = 0u;
    expect_snapshot_rejected(catalog, &snapshot, "missing config rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_active = 0u;
    expect_snapshot_rejected(catalog, &snapshot, "inactive layer rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_loop = 1u;
    expect_snapshot_rejected(catalog, &snapshot, "looping layer rejected");

    snapshot = valid_snapshot(REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    snapshot.action_playing = 2u;
    expect_snapshot_rejected(catalog, &snapshot,
        "non-binary action flag rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_has_clip = 2u;
    expect_snapshot_rejected(catalog, &snapshot,
        "non-binary clip flag rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_has_config = 2u;
    expect_snapshot_rejected(catalog, &snapshot,
        "non-binary config flag rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_active = 2u;
    expect_snapshot_rejected(catalog, &snapshot,
        "non-binary active flag rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_loop = 2u;
    expect_snapshot_rejected(catalog, &snapshot,
        "non-binary loop flag rejected");

    snapshot = valid_snapshot(REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    snapshot.clip_cursor_frames = -1.0f;
    expect_snapshot_rejected(catalog, &snapshot, "negative cursor rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    snapshot.clip_cursor_frames = NAN;
    expect_snapshot_rejected(catalog, &snapshot,
        "non-finite cursor rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    snapshot.clip_fps = 49.0f;
    expect_snapshot_rejected(catalog, &snapshot,
        "wrong clip rate rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    snapshot.clip_fps = NAN;
    expect_snapshot_rejected(catalog, &snapshot,
        "non-finite clip rate rejected");

    snapshot = valid_snapshot(REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 1);
    RekG1StrikeCatalog invalid = *catalog;
    invalid.build_fingerprint = "different";
    expect_snapshot_rejected(&invalid, &snapshot,
        "unmatched catalog rejected");
    require(!rek_g1_strike_intent_from_snapshot(
        catalog, &snapshot, NULL), "null output rejected");
}

int main(void) {
    test_catalog_identity_and_values();
    test_catalog_validation_rejects_mutation();
    test_intent_assembly();
    test_intent_fail_closed();
    printf("g1 strike catalog: %zu checks passed\n", checks);
    return 0;
}
