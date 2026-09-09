#include "g1_strike_catalog.h"

#include <math.h>
#include <string.h>

static const RekG1ImpactEvent CURRENT_BUILD_IMPACT_EVENTS[
        REK_G1_STRIKE_CATALOG_IMPACT_EVENT_COUNT] = {
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

static const RekG1StrikeCatalogEntry CURRENT_BUILD_ENTRIES[
        REK_G1_STRIKE_CATALOG_ENTRY_COUNT] = {
    {REK_G1_NATIVE_MOVE_6_LEFT_SIDE, 6u, 0u, 1u},
    {REK_G1_NATIVE_MOVE_7_LEFT_FRONT, 7u, 1u, 1u},
    {REK_G1_NATIVE_MOVE_8_RIGHT_SIDE, 8u, 2u, 1u},
    {REK_G1_NATIVE_MOVE_9_RIGHT_KNEE, 9u, 3u, 1u},
    {REK_G1_NATIVE_MOVE_0_LEFT_HOOK, 0u, 4u, 1u},
    {REK_G1_NATIVE_MOVE_1_LEFT_JAB, 1u, 5u, 1u},
    {REK_G1_NATIVE_MOVE_2_DOUBLE_UPPERCUT, 2u, 6u, 2u},
    {REK_G1_NATIVE_MOVE_3_RIGHT_HOOK, 3u, 8u, 1u},
    {REK_G1_NATIVE_MOVE_4_RIGHT_JAB, 4u, 9u, 1u},
    {REK_G1_NATIVE_MOVE_5_LEFT_JAB_RIGHT_UPPERCUT, 5u, 10u, 2u},
    {REK_G1_NATIVE_MOVE_10_SIX_PUNCH, 10u, 12u, 10u},
    {REK_G1_NATIVE_MOVE_11_RUN_AND_PUNCH, 11u, 22u, 1u},
    {REK_G1_NATIVE_MOVE_12_LEFT_RIGHT_JAB, 12u, 23u, 1u},
    {REK_G1_NATIVE_MOVE_13_LEFT_RIGHT_HOOK, 13u, 24u, 1u},
    {REK_G1_NATIVE_MOVE_14_LEFT_HOOK_RIGHT_JAB, 14u, 25u, 1u},
    {REK_G1_NATIVE_MOVE_15_DOUBLE_HOOK, 15u, 26u, 2u},
    {REK_G1_NATIVE_MOVE_16_BUTT_SMACK_EMOTE, 16u, 28u, 1u},
};

_Static_assert(
    REK_G1_NATIVE_MOVE_16_BUTT_SMACK_EMOTE
        - REK_G1_NATIVE_MOVE_6_LEFT_SIDE + 1
        == REK_G1_STRIKE_CATALOG_ENTRY_COUNT,
    "discrete move route ids must remain dense");

static const RekG1StrikeCatalog CURRENT_BUILD_CATALOG = {
    .build_fingerprint = REK_G1_STRIKE_CATALOG_BUILD_FINGERPRINT,
    .source_sha256 = REK_G1_STRIKE_CATALOG_SOURCE_SHA256,
    .entries = CURRENT_BUILD_ENTRIES,
    .count = REK_G1_STRIKE_CATALOG_ENTRY_COUNT,
    .impact_events = CURRENT_BUILD_IMPACT_EVENTS,
    .impact_event_count = REK_G1_STRIKE_CATALOG_IMPACT_EVENT_COUNT,
};

static int binary_flag(uint8_t value) {
    return value == 0u || value == 1u;
}

static int f32_equal_bits(float left, float right) {
    uint32_t left_bits = 0u;
    uint32_t right_bits = 0u;
    memcpy(&left_bits, &left, sizeof(left_bits));
    memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

static int impact_event_equal(
        const RekG1ImpactEvent* actual,
        const RekG1ImpactEvent* expected) {
    return actual != NULL && expected != NULL
        && f32_equal_bits(
            actual->impact_time_seconds, expected->impact_time_seconds)
        && f32_equal_bits(
            actual->lead_time_seconds, expected->lead_time_seconds)
        && f32_equal_bits(
            actual->release_time_seconds, expected->release_time_seconds)
        && f32_equal_bits(actual->gain_boost, expected->gain_boost)
        && actual->limb == expected->limb;
}

const RekG1StrikeCatalog* rek_g1_current_build_strike_catalog(void) {
    return &CURRENT_BUILD_CATALOG;
}

int rek_g1_validate_strike_catalog(const RekG1StrikeCatalog* catalog) {
    if (catalog == NULL || catalog->build_fingerprint == NULL
            || catalog->source_sha256 == NULL || catalog->entries == NULL
            || catalog->impact_events == NULL
            || catalog->count != REK_G1_STRIKE_CATALOG_ENTRY_COUNT
            || catalog->impact_event_count
                != REK_G1_STRIKE_CATALOG_IMPACT_EVENT_COUNT
            || strcmp(
                catalog->build_fingerprint,
                REK_G1_STRIKE_CATALOG_BUILD_FINGERPRINT) != 0
            || strcmp(
                catalog->source_sha256,
                REK_G1_STRIKE_CATALOG_SOURCE_SHA256) != 0) {
        return 0;
    }
    uint8_t route_seen[REK_G1_STATIC_ROUTE_COUNT] = {0};
    uint8_t move_seen[REK_G1_STRIKE_CATALOG_ENTRY_COUNT] = {0};
    size_t next_event = 0u;
    for (size_t index = 0; index < REK_G1_STRIKE_CATALOG_ENTRY_COUNT;
            index++) {
        const RekG1StrikeCatalogEntry* actual = &catalog->entries[index];
        const RekG1StrikeCatalogEntry* expected =
            &CURRENT_BUILD_ENTRIES[index];
        const size_t route_index = (size_t)actual->route_id;
        if (actual->route_id != expected->route_id
                || actual->runtime_move_index != expected->runtime_move_index
                || actual->impact_event_offset
                    != expected->impact_event_offset
                || actual->impact_event_count != expected->impact_event_count
                || route_index >= REK_G1_STATIC_ROUTE_COUNT
                || route_seen[route_index]
                || actual->runtime_move_index
                    >= REK_G1_STRIKE_CATALOG_ENTRY_COUNT
                || move_seen[actual->runtime_move_index]
                || actual->impact_event_count == 0u
                || actual->impact_event_count
                    > REK_G1_STRIKE_CATALOG_MAX_EVENTS_PER_MOVE
                || actual->impact_event_offset != next_event
                || next_event + actual->impact_event_count
                    > catalog->impact_event_count) {
            return 0;
        }
        const RekG1NativeMotionRoute* route =
            rek_g1_native_discrete_move_route(
                rek_g1_native_static_motion_routes(),
                actual->runtime_move_index);
        if (route == NULL || route->id != actual->route_id) return 0;
        route_seen[route_index] = 1u;
        move_seen[actual->runtime_move_index] = 1u;
        for (size_t event_index = 0;
                event_index < actual->impact_event_count;
                event_index++) {
            const size_t pool_index = next_event + event_index;
            const RekG1ImpactEvent* event = &catalog->impact_events[pool_index];
            if (!impact_event_equal(
                    event, &CURRENT_BUILD_IMPACT_EVENTS[pool_index])
                    || !isfinite(event->impact_time_seconds)
                    || !isfinite(event->lead_time_seconds)
                    || !isfinite(event->release_time_seconds)
                    || !isfinite(event->gain_boost)
                    || event->impact_time_seconds < 0.0f
                    || event->lead_time_seconds < 0.0f
                    || event->release_time_seconds < 0.0f
                    || event->gain_boost < 0.0f
                    || event->limb < REK_G1_AIM_LIMB_LEFT_UPPER_BODY
                    || event->limb > REK_G1_AIM_LIMB_RIGHT_LOWER_BODY) {
                return 0;
            }
        }
        next_event += actual->impact_event_count;
    }
    if (next_event != catalog->impact_event_count) return 0;
    for (size_t move = 0; move < REK_G1_STRIKE_CATALOG_ENTRY_COUNT; move++) {
        if (!move_seen[move]) return 0;
    }
    return 1;
}

const RekG1StrikeCatalogEntry* rek_g1_strike_catalog_entry_by_route(
        const RekG1StrikeCatalog* catalog,
        RekG1NativeRouteId route_id) {
    if (catalog == &CURRENT_BUILD_CATALOG) {
        const int32_t index = (int32_t)route_id
            - (int32_t)REK_G1_NATIVE_MOVE_6_LEFT_SIDE;
        if (index < 0 || index >= REK_G1_STRIKE_CATALOG_ENTRY_COUNT) {
            return NULL;
        }
        return &CURRENT_BUILD_ENTRIES[(size_t)index];
    }
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
            || local.current_layer_loop
            || !isfinite(local.clip_cursor_frames)
            || local.clip_cursor_frames < 0.0f
            || local.clip_fps != 50.0f) {
        return 0;
    }
    const RekG1StrikeCatalogEntry* entry =
        rek_g1_strike_catalog_entry_by_route(
            catalog, local.active_route_id);
    if (entry == NULL) {
        return 0;
    }
    *output = (RekG1StrikeIntent){
        .impact_events = catalog->impact_events + entry->impact_event_offset,
        .impact_event_count = entry->impact_event_count,
        .clip_cursor_frames = local.clip_cursor_frames,
        .clip_fps = local.clip_fps,
        .move_id = local.action_move_id,
        .action_playing = local.action_playing,
        .layer_active = local.current_layer_active,
        .layer_loop = local.current_layer_loop,
    };
    return 1;
}
