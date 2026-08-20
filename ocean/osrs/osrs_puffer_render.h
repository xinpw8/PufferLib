#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void* osrs_puffer_render_create(
    const void* encounter_def,
    void* encounter_state,
    void* encounter_context);
void osrs_puffer_render_draw(void* renderer);
void osrs_puffer_render_destroy(void* renderer);

#ifdef __cplusplus
}
#endif
