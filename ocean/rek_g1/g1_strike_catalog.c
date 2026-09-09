#include "g1_strike_catalog.h"

#include <math.h>
#include <string.h>

static const RekG1StrikeCatalogEntry CURRENT_BUILD_ENTRIES[
        REK_G1_STRIKE_CATALOG_ENTRY_COUNT] = {
    {
        .route_id = REK_G1_NATIVE_KICK_MOVE_6_LEFT_SIDE,
        .runtime_move_index = 6u,
        .impact_event = {
            .impact_time_seconds = 1.100000023841858f,
            .lead_time_seconds = 0.30000001192092896f,
            .release_time_seconds = 0.5f,
            .limb = REK_G1_AIM_LIMB_LEFT_LOWER_BODY,
        },
    },
    {
        .route_id = REK_G1_NATIVE_KICK_MOVE_7_LEFT_FRONT,
        .runtime_move_index = 7u,
        .impact_event = {
            .impact_time_seconds = 1.0f,
            .lead_time_seconds = 0.20000000298023224f,
            .release_time_seconds = 0.5f,
            .limb = REK_G1_AIM_LIMB_LEFT_LOWER_BODY,
        },
    },
    {
        .route_id = REK_G1_NATIVE_KICK_MOVE_8_RIGHT_SIDE,
        .runtime_move_index = 8u,
        .impact_event = {
            .impact_time_seconds = 1.1399999856948853f,
            .lead_time_seconds = 0.4000000059604645f,
            .release_time_seconds = 0.15000000596046448f,
            .limb = REK_G1_AIM_LIMB_RIGHT_LOWER_BODY,
        },
    },
    {
        .route_id = REK_G1_NATIVE_KICK_MOVE_9_RIGHT_KNEE,
        .runtime_move_index = 9u,
        .impact_event = {
            .impact_time_seconds = 0.6499999761581421f,
            .lead_time_seconds = 0.4000000059604645f,
            .release_time_seconds = 0.15000000596046448f,
            .limb = REK_G1_AIM_LIMB_RIGHT_LOWER_BODY,
        },
    },
};

static const RekG1StrikeCatalog CURRENT_BUILD_CATALOG = {
    .build_fingerprint = REK_G1_STRIKE_CATALOG_BUILD_FINGERPRINT,
    .source_sha256 = REK_G1_STRIKE_CATALOG_SOURCE_SHA256,
    .entries = CURRENT_BUILD_ENTRIES,
    .count = REK_G1_STRIKE_CATALOG_ENTRY_COUNT,
};

static int binary_flag(uint8_t value) {
    return value == 0u || value == 1u;
}

static int impact_event_equal(
        const RekG1ImpactEvent* actual,
        const RekG1ImpactEvent* expected) {
    return actual != NULL && expected != NULL
        && actual->impact_time_seconds == expected->impact_time_seconds
        && actual->lead_time_seconds == expected->lead_time_seconds
        && actual->release_time_seconds == expected->release_time_seconds
        && actual->limb == expected->limb;
}

const RekG1StrikeCatalog* rek_g1_current_build_strike_catalog(void) {
    return &CURRENT_BUILD_CATALOG;
}

int rek_g1_validate_strike_catalog(const RekG1StrikeCatalog* catalog) {
    if (catalog == NULL || catalog->build_fingerprint == NULL
            || catalog->source_sha256 == NULL || catalog->entries == NULL
            || catalog->count != REK_G1_STRIKE_CATALOG_ENTRY_COUNT
            || strcmp(
                catalog->build_fingerprint,
                REK_G1_STRIKE_CATALOG_BUILD_FINGERPRINT) != 0
            || strcmp(
                catalog->source_sha256,
                REK_G1_STRIKE_CATALOG_SOURCE_SHA256) != 0) {
        return 0;
    }
    for (size_t index = 0; index < REK_G1_STRIKE_CATALOG_ENTRY_COUNT;
            index++) {
        const RekG1StrikeCatalogEntry* actual = &catalog->entries[index];
        const RekG1StrikeCatalogEntry* expected =
            &CURRENT_BUILD_ENTRIES[index];
        if (actual->route_id != expected->route_id
                || actual->runtime_move_index !=
                    expected->runtime_move_index
                || !impact_event_equal(
                    &actual->impact_event, &expected->impact_event)
                || !isfinite(actual->impact_event.impact_time_seconds)
                || !isfinite(actual->impact_event.lead_time_seconds)
                || !isfinite(actual->impact_event.release_time_seconds)
                || actual->impact_event.impact_time_seconds < 0.0f
                || actual->impact_event.lead_time_seconds < 0.0f
                || actual->impact_event.release_time_seconds < 0.0f) {
            return 0;
        }
    }
    return 1;
}

const RekG1StrikeCatalogEntry* rek_g1_strike_catalog_entry_by_route(
        const RekG1StrikeCatalog* catalog,
        RekG1NativeRouteId route_id) {
    if (!rek_g1_validate_strike_catalog(catalog)) return NULL;
    for (size_t index = 0; index < catalog->count; index++) {
        if (catalog->entries[index].route_id == route_id) {
            return &catalog->entries[index];
        }
    }
    return NULL;
}

int rek_g1_strike_intent_from_snapshot(
        const RekG1StrikeCatalog* catalog,
        const RekG1StrikeComposerSnapshot* snapshot,
        RekG1StrikeIntent* output) {
    if (output == NULL) return 0;
    RekG1StrikeComposerSnapshot local = {0};
    if (snapshot != NULL) local = *snapshot;
    memset(output, 0, sizeof(*output));
    if (snapshot == NULL || !binary_flag(local.action_playing)
            || !binary_flag(local.current_layer_has_clip)
            || !binary_flag(local.current_layer_has_config)
            || !binary_flag(local.current_layer_active)
            || !binary_flag(local.current_layer_loop)
            || !local.action_playing || !local.current_layer_has_clip
            || !local.current_layer_has_config || !local.current_layer_active
            || local.current_layer_loop || local.action_move_id <= 0
            || !isfinite(local.clip_cursor_frames)
            || local.clip_cursor_frames < 0.0f
            || local.clip_fps != 50.0f) {
        return 0;
    }
    const RekG1StrikeCatalogEntry* entry =
        rek_g1_strike_catalog_entry_by_route(
            catalog, local.active_route_id);
    if (entry == NULL) return 0;
    *output = (RekG1StrikeIntent){
        .impact_events = &entry->impact_event,
        .impact_event_count = 1u,
        .clip_cursor_frames = local.clip_cursor_frames,
        .clip_fps = local.clip_fps,
        .move_id = local.action_move_id,
        .action_playing = local.action_playing,
        .layer_active = local.current_layer_active,
        .layer_loop = local.current_layer_loop,
    };
    return 1;
}
