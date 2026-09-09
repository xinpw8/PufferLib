#include "g1_strike_catalog.h"

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

static void test_catalog_identity_and_values(void) {
    const RekG1StrikeCatalog* catalog =
        rek_g1_current_build_strike_catalog();
    require(catalog != NULL, "current-build catalog exists");
    require(rek_g1_validate_strike_catalog(catalog), "catalog validates");
    require(catalog->count == 4u, "catalog has only four kick entries");
    require(strcmp(
        catalog->build_fingerprint,
        REK_G1_STRIKE_CATALOG_BUILD_FINGERPRINT) == 0,
        "catalog build fingerprint");
    require(strcmp(
        catalog->source_sha256,
        REK_G1_STRIKE_CATALOG_SOURCE_SHA256) == 0,
        "catalog source hash");

    static const RekG1NativeRouteId route_ids[] = {
        REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE,
        REK_G1_NATIVE_KICK_MOVE_7_LEFT_FRONT,
        REK_G1_NATIVE_KICK_MOVE_8_RIGHT_SIDE,
        REK_G1_NATIVE_KICK_MOVE_9_RIGHT_KNEE,
    };
    static const uint16_t move_indices[] = {6u, 7u, 8u, 9u};
    static const uint32_t impact_bits[] = {
        UINT32_C(0x3f8ccccd), UINT32_C(0x3f800000),
        UINT32_C(0x3f91eb85), UINT32_C(0x3f266666),
    };
    static const uint32_t lead_bits[] = {
        UINT32_C(0x3e99999a), UINT32_C(0x3e4ccccd),
        UINT32_C(0x3ecccccd), UINT32_C(0x3ecccccd),
    };
    static const uint32_t release_bits[] = {
        UINT32_C(0x3f000000), UINT32_C(0x3f000000),
        UINT32_C(0x3e19999a), UINT32_C(0x3e19999a),
    };
    static const RekG1AimLimb limbs[] = {
        REK_G1_AIM_LIMB_LEFT_LOWER_BODY, REK_G1_AIM_LIMB_LEFT_LOWER_BODY,
        REK_G1_AIM_LIMB_RIGHT_LOWER_BODY, REK_G1_AIM_LIMB_RIGHT_LOWER_BODY,
    };
    for (size_t index = 0; index < 4u; index++) {
        const RekG1StrikeCatalogEntry* entry =
            rek_g1_strike_catalog_entry_by_route(catalog, route_ids[index]);
        require(entry != NULL, "kick route lookup succeeds");
        require(entry == &catalog->entries[index], "lookup returns catalog row");
        require(entry->runtime_move_index == move_indices[index],
            "runtime move index matches recovered route");
        require(f32_bits(entry->impact_event.impact_time_seconds)
            == impact_bits[index], "impact time binary32 value");
        require(f32_bits(entry->impact_event.lead_time_seconds)
            == lead_bits[index], "lead time binary32 value");
        require(f32_bits(entry->impact_event.release_time_seconds)
            == release_bits[index], "release time binary32 value");
        require(entry->impact_event.limb == limbs[index],
            "impact limb matches recovered route");
    }

    for (int32_t route = (int32_t)REK_G1_NATIVE_IDLE;
            route <= (int32_t)REK_G1_NATIVE_TURN_RIGHT; route++) {
        require(rek_g1_strike_catalog_entry_by_route(
            catalog, (RekG1NativeRouteId)route) == NULL,
            "non-kick route fails closed");
    }
    require(rek_g1_strike_catalog_entry_by_route(
        catalog, (RekG1NativeRouteId)-1) == NULL,
        "negative route fails closed");
    require(rek_g1_strike_catalog_entry_by_route(
        catalog, (RekG1NativeRouteId)11) == NULL,
        "unknown route fails closed");
}

static void test_catalog_validation_rejects_mutation(void) {
    const RekG1StrikeCatalog* current =
        rek_g1_current_build_strike_catalog();
    RekG1StrikeCatalog catalog = *current;
    RekG1StrikeCatalogEntry entries[REK_G1_STRIKE_CATALOG_ENTRY_COUNT];
    memcpy(entries, current->entries, sizeof(entries));
    catalog.entries = entries;
    require(rek_g1_validate_strike_catalog(&catalog), "copied catalog validates");

    catalog.build_fingerprint = "different";
    require(!rek_g1_validate_strike_catalog(&catalog), "wrong build rejected");
    catalog = *current;
    catalog.entries = entries;
    catalog.source_sha256 = "different";
    require(!rek_g1_validate_strike_catalog(&catalog), "wrong source rejected");
    catalog = *current;
    catalog.entries = entries;
    catalog.count = 3u;
    require(!rek_g1_validate_strike_catalog(&catalog), "short catalog rejected");
    catalog = *current;
    catalog.entries = NULL;
    require(!rek_g1_validate_strike_catalog(&catalog), "null entries rejected");
    require(!rek_g1_validate_strike_catalog(NULL), "null catalog rejected");

    catalog = *current;
    catalog.entries = entries;
    entries[0].route_id = REK_G1_NATIVE_IDLE;
    require(!rek_g1_validate_strike_catalog(&catalog), "mutated route rejected");
    memcpy(entries, current->entries, sizeof(entries));
    entries[1].runtime_move_index = 9u;
    require(!rek_g1_validate_strike_catalog(&catalog), "mutated move rejected");
    memcpy(entries, current->entries, sizeof(entries));
    entries[2].impact_event.impact_time_seconds = 1.0f;
    require(!rek_g1_validate_strike_catalog(&catalog), "mutated impact rejected");
    memcpy(entries, current->entries, sizeof(entries));
    entries[3].impact_event.lead_time_seconds = NAN;
    require(!rek_g1_validate_strike_catalog(&catalog), "non-finite lead rejected");
    memcpy(entries, current->entries, sizeof(entries));
    entries[0].impact_event.release_time_seconds = -0.5f;
    require(!rek_g1_validate_strike_catalog(&catalog), "negative release rejected");
    memcpy(entries, current->entries, sizeof(entries));
    entries[0].impact_event.limb = REK_G1_AIM_LIMB_RIGHT_LOWER_BODY;
    require(!rek_g1_validate_strike_catalog(&catalog), "mutated limb rejected");
}

static void test_intent_assembly(void) {
    const RekG1StrikeCatalog* catalog =
        rek_g1_current_build_strike_catalog();
    for (int32_t route = (int32_t)REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE;
            route <= (int32_t)REK_G1_NATIVE_KICK_MOVE_9_RIGHT_KNEE;
            route++) {
        const RekG1StrikeComposerSnapshot snapshot = valid_snapshot(
            (RekG1NativeRouteId)route, 101 + route);
        RekG1StrikeIntent intent;
        memset(&intent, 0, sizeof(intent));
        require(rek_g1_strike_intent_from_snapshot(
            catalog, &snapshot, &intent), "active kick assembles intent");
        const RekG1StrikeCatalogEntry* entry =
            rek_g1_strike_catalog_entry_by_route(
                catalog, (RekG1NativeRouteId)route);
        require(intent.impact_events == &entry->impact_event,
            "intent points to immutable catalog event");
        require(intent.impact_event_count == 1u, "one recovered impact event");
        require(intent.clip_cursor_frames == snapshot.clip_cursor_frames,
            "cursor copied exactly");
        require(intent.clip_fps == snapshot.clip_fps, "fps copied exactly");
        require(intent.move_id == snapshot.action_move_id,
            "composer action identity copied exactly");
        require(intent.action_playing == 1u, "playing flag copied");
        require(intent.layer_active == 1u, "active flag copied");
        require(intent.layer_loop == 0u, "non-loop flag copied");
    }

    RekG1StrikeComposerSnapshot snapshot = valid_snapshot(
        REK_G1_NATIVE_KICK_MOVE_7_LEFT_FRONT, 37);
    RekG1StrikeIntent intent;
    require(rek_g1_strike_intent_from_snapshot(
        catalog, &snapshot, &intent), "left-front intent assembled");
    int32_t apex = -1;
    float ramp = 0.0f;
    require(rek_g1_strike_intent_apex(
        &intent, REK_G1_BODY_PART_FOOT, REK_G1_HAND_LEFT,
        1.0f, &apex, &ramp), "assembled intent reaches exact apex");
    require(apex == 0, "single event has apex index zero");
    require(ramp == 1.0f, "assembled intent apex ramp is one");
}

static void test_intent_fail_closed(void) {
    const RekG1StrikeCatalog* catalog =
        rek_g1_current_build_strike_catalog();
    RekG1StrikeComposerSnapshot snapshot = valid_snapshot(
        REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    expect_snapshot_rejected(catalog, NULL, "missing snapshot rejected");
    for (int32_t route = (int32_t)REK_G1_NATIVE_IDLE;
            route <= (int32_t)REK_G1_NATIVE_TURN_RIGHT; route++) {
        snapshot = valid_snapshot((RekG1NativeRouteId)route, 1);
        expect_snapshot_rejected(catalog, &snapshot, "non-kick snapshot rejected");
    }
    snapshot = valid_snapshot((RekG1NativeRouteId)11, 1);
    expect_snapshot_rejected(catalog, &snapshot, "unknown route rejected");

    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    snapshot.action_playing = 0u;
    expect_snapshot_rejected(catalog, &snapshot, "non-playing action rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_has_clip = 0u;
    expect_snapshot_rejected(catalog, &snapshot, "missing clip rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_has_config = 0u;
    expect_snapshot_rejected(catalog, &snapshot, "missing config rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_active = 0u;
    expect_snapshot_rejected(catalog, &snapshot, "inactive layer rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_loop = 1u;
    expect_snapshot_rejected(catalog, &snapshot, "looping layer rejected");

    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    snapshot.action_playing = 2u;
    expect_snapshot_rejected(catalog, &snapshot, "non-binary action flag rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_has_clip = 2u;
    expect_snapshot_rejected(catalog, &snapshot, "non-binary clip flag rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_has_config = 2u;
    expect_snapshot_rejected(catalog, &snapshot, "non-binary config flag rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_active = 2u;
    expect_snapshot_rejected(catalog, &snapshot, "non-binary active flag rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    snapshot.current_layer_loop = 2u;
    expect_snapshot_rejected(catalog, &snapshot, "non-binary loop flag rejected");

    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 0);
    expect_snapshot_rejected(catalog, &snapshot, "missing action identity rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, -1);
    expect_snapshot_rejected(catalog, &snapshot, "negative action identity rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    snapshot.clip_cursor_frames = -1.0f;
    expect_snapshot_rejected(catalog, &snapshot, "negative cursor rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    snapshot.clip_cursor_frames = NAN;
    expect_snapshot_rejected(catalog, &snapshot, "non-finite cursor rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    snapshot.clip_fps = 49.0f;
    expect_snapshot_rejected(catalog, &snapshot, "wrong clip rate rejected");
    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    snapshot.clip_fps = NAN;
    expect_snapshot_rejected(catalog, &snapshot, "missing clip rate rejected");

    snapshot = valid_snapshot(REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE, 1);
    RekG1StrikeCatalog invalid = *catalog;
    invalid.build_fingerprint = "different";
    expect_snapshot_rejected(&invalid, &snapshot, "unmatched catalog rejected");
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
