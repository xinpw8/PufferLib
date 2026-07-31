#ifndef OSRS_SCENE_ASSETS_H
#define OSRS_SCENE_ASSETS_H

#include "osrs_assets.h"
#include "osrs_collision.h"
#include "osrs_objects.h"
#include "osrs_render.h"
#include "osrs_terrain.h"

typedef struct {
    OsrsAssetGroupKind required_groups[4];

    const char* terrain_path;
    const char* objects_path;
    const char* objects_secondary_path;
    const char* cmap_path;
    const char* npc_models_path;
    const char* npc_anims_path;

    int world_origin_x;
    int world_origin_y;
} EncounterSceneConfig;

static inline CollisionMap* encounter_load_scene_assets(
    RenderClient* rc, const EncounterSceneConfig* cfg
) {
    if (!rc || !cfg) {
        fprintf(stderr, "encounter_load_scene_assets: null rc or cfg\n");
        abort();
    }

    for (int i = 0; i < (int)(sizeof(cfg->required_groups) / sizeof(cfg->required_groups[0])); i++) {
        OsrsAssetGroupKind group = cfg->required_groups[i];
        if ((int)group < 0) break;
        osrs_asset_require_group(group);
    }

    rc->model_cache = model_cache_load(OSRS_ASSET("equipment.models"));
    if (rc->model_cache) rc->show_models = 1;
    rc->anim_cache = anim_cache_load(OSRS_ASSET("equipment.anims"));
    render_load_projectile_assets(rc);
    render_init_overlay_models(rc);

    if (cfg->terrain_path) {
        rc->terrain = terrain_load(cfg->terrain_path);
        if (rc->terrain && (cfg->world_origin_x || cfg->world_origin_y))
            terrain_offset(rc->terrain, cfg->world_origin_x, cfg->world_origin_y);
    }

    if (cfg->objects_path) {
        rc->objects = objects_load(cfg->objects_path);
        if (rc->objects && (cfg->world_origin_x || cfg->world_origin_y))
            objects_offset(rc->objects, cfg->world_origin_x, cfg->world_origin_y);
    }

    if (cfg->objects_secondary_path) {
        rc->objects_zuk = objects_load(cfg->objects_secondary_path);
        if (rc->objects_zuk && (cfg->world_origin_x || cfg->world_origin_y))
            objects_offset(rc->objects_zuk, cfg->world_origin_x, cfg->world_origin_y);
    }

    if (cfg->npc_models_path)
        rc->npc_model_cache = model_cache_load(cfg->npc_models_path);
    if (cfg->npc_anims_path)
        rc->npc_anim_cache = anim_cache_load(cfg->npc_anims_path);

    CollisionMap* cmap = NULL;
    if (cfg->cmap_path) {
        cmap = collision_map_load(cfg->cmap_path);
        if (cmap) {
            rc->collision_map = cmap;
            rc->collision_world_offset_x = cfg->world_origin_x;
            rc->collision_world_offset_y = cfg->world_origin_y;
        }
    }

    return cmap;
}

#endif
