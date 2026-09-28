#pragma once

#include <stddef.h>
#include <stdint.h>

#include "g1_hit_detector.h"
#include "native_motion_routes.h"

/*
 * This catalog is static evidence for one recovered public-family G1 build.
 * It does not assert that any current REK server selected the same build.
 */
#define REK_G1_STRIKE_CATALOG_BUILD_FINGERPRINT \
    "f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659"
#define REK_G1_STRIKE_CATALOG_SOURCE_SHA256 \
    "b566d1064452558f4cddeeb9953007154f3dd8e3b4eb6de73bf8deebcacf7bb7"

enum {
    REK_G1_STRIKE_CATALOG_ENTRY_COUNT = 17,
    REK_G1_STRIKE_CATALOG_IMPACT_EVENT_COUNT = 29,
    REK_G1_STRIKE_CATALOG_MAX_EVENTS_PER_MOVE = 10,
};

typedef struct RekG1StrikeCatalogEntry {
    RekG1NativeRouteId route_id;
    uint16_t runtime_move_index;
    uint16_t impact_event_offset;
    uint16_t impact_event_count;
} RekG1StrikeCatalogEntry;

typedef struct RekG1StrikeCatalog {
    const char* build_fingerprint;
    const char* source_sha256;
    const RekG1StrikeCatalogEntry* entries;
    size_t count;
    const RekG1ImpactEvent* impact_events;
    size_t impact_event_count;
} RekG1StrikeCatalog;

/*
 * The semantic runtime copies these fields from one composer state at a
 * defined instant. Keeping the snapshot explicit prevents this catalog from
 * reading mutable composer state at contact-processing time.
 */
typedef struct RekG1StrikeComposerSnapshot {
    RekG1NativeRouteId active_route_id;
    float clip_cursor_frames;
    float clip_fps;
    /* Wrapping composer invocation identity, not the zero-based move index. */
    int32_t action_move_id;
    uint8_t action_playing;
    uint8_t current_layer_has_clip;
    uint8_t current_layer_has_config;
    uint8_t current_layer_active;
    uint8_t current_layer_loop;
} RekG1StrikeComposerSnapshot;

const RekG1StrikeCatalog* rek_g1_current_build_strike_catalog(void);

int rek_g1_validate_strike_catalog(const RekG1StrikeCatalog* catalog);

const RekG1StrikeCatalogEntry* rek_g1_strike_catalog_entry_by_route(
    const RekG1StrikeCatalog* catalog,
    RekG1NativeRouteId route_id
);

/*
 * Returns one only for a complete, active, non-looping discrete-move snapshot whose
 * route and 50 Hz clip identity are present in the pinned catalog. The output
 * is cleared on every rejected input.
 */
int rek_g1_strike_intent_from_snapshot(
    const RekG1StrikeCatalog* catalog,
    const RekG1StrikeComposerSnapshot* snapshot,
    RekG1StrikeIntent* output
);
