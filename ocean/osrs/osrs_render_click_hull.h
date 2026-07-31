#ifndef OSRS_RENDER_CLICK_HULL_H
#define OSRS_RENDER_CLICK_HULL_H

#if __has_include("raylib.h")
#include "raylib.h"
#elif __has_include("raylib-5.5_macos/include/raylib.h")
#include "raylib-5.5_macos/include/raylib.h"
#else
#error "raylib.h not found"
#endif
#include "osrs_encounter.h"

#define RENDER_CLICKBOX_PRISM_SIDE_COUNT 6
#define RENDER_CLICKBOX_PRISM_POINT_COUNT (RENDER_CLICKBOX_PRISM_SIDE_COUNT * 2)

static inline int render_entity_clickbox_size(const RenderEntity* entity) {
    return entity->npc_size > 1 ? entity->npc_size : 1;
}

static inline float render_entity_clickbox_radius_tiles(const RenderEntity* entity) {
    return (float)render_entity_clickbox_size(entity) * 0.4f;
}

static inline float render_entity_clickbox_height_tiles(
    const RenderEntity* entity, float visual_height_tiles
) {
    float min_height = (float)render_entity_clickbox_size(entity);
    float model_height = visual_height_tiles * 0.4f;
    return model_height > min_height ? model_height : min_height;
}

static inline int render_build_entity_clickbox_prism_points(
    const RenderEntity* entity,
    float center_x,
    float center_z,
    float ground_y,
    float visual_height_tiles,
    Vector3* out,
    int out_capacity
) {
    if (entity->entity_type != ENTITY_NPC) return 0;
    if (out_capacity < RENDER_CLICKBOX_PRISM_POINT_COUNT) return 0;

    float radius = render_entity_clickbox_radius_tiles(entity);
    float height = render_entity_clickbox_height_tiles(entity, visual_height_tiles);
    static const float hex[RENDER_CLICKBOX_PRISM_SIDE_COUNT][2] = {
        { 1.0f, 0.0f },
        { 0.5f, 0.8660254038f },
        { -0.5f, 0.8660254038f },
        { -1.0f, 0.0f },
        { -0.5f, -0.8660254038f },
        { 0.5f, -0.8660254038f },
    };

    int n = 0;
    for (int i = 0; i < RENDER_CLICKBOX_PRISM_SIDE_COUNT; i++) {
        float x = center_x + hex[i][0] * radius;
        float z = center_z + hex[i][1] * radius;
        out[n++] = (Vector3){ x, ground_y, z };
        out[n++] = (Vector3){ x, ground_y + height, z };
    }
    return n;
}

#endif
