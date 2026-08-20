#define OSRS_VISUAL

#include <stdlib.h>

#include "osrs_puffer_render.h"
#include "osrs_render_scene.h"

typedef struct {
    OsrsEnv env;
} OsrsPufferRenderer;

void* osrs_puffer_render_create(
    const void* encounter_def,
    void* encounter_state,
    void* encounter_context
) {
    const EncounterDef* def = (const EncounterDef*)encounter_def;
    OsrsPufferRenderer* renderer = (OsrsPufferRenderer*)calloc(1, sizeof(*renderer));
    if (renderer == NULL) abort();

    renderer->env.encounter_def = def;
    renderer->env.encounter_state = encounter_state;
    renderer->env.encounter_context = encounter_context;
    visual_load_encounter_collision_map(def, &renderer->env, def->name);
    visual_init_render_scene(&renderer->env, def->name, NULL);
    return renderer;
}

void osrs_puffer_render_draw(void* opaque_renderer) {
    OsrsPufferRenderer* renderer = (OsrsPufferRenderer*)opaque_renderer;
    pvp_render(&renderer->env);
}

void osrs_puffer_render_destroy(void* opaque_renderer) {
    OsrsPufferRenderer* renderer = (OsrsPufferRenderer*)opaque_renderer;
    render_destroy_client((RenderClient*)renderer->env.client);
    collision_map_free((CollisionMap*)renderer->env.collision_map);
    free(renderer);
}
