/* binding.h – Static env interface for pfr_native
 * Used by pufferlib/extensions/env_binding.c to build _C with pfr_native linked.
 */

#define PFR_STATIC_ENV 1
#include "pfr_native_env.h"

#define OBS_SIZE PFR_OBS_SIZE   /* 129 */
#define NUM_ATNS 1
#define ACT_SIZES {PFR_NUM_ACTIONS}
#define OBS_TYPE UNSIGNED_CHAR
#define ACT_TYPE DOUBLE

#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    pfr_engine_init(&env->core);
    pfr_heatmap_ensure_alloc();
}

/* Expose heatmap pointer + dimensions via my_get so Python can read it */
#define MY_GET
void* my_get(void* env_void, Dict* out) {
    dict_set(out, "heatmap_h", (double)PFR_HEATMAP_H);
    dict_set(out, "heatmap_w", (double)PFR_HEATMAP_W);
    /* Store pointer as uint64 cast to double so Python can recover it */
    union { float* p; uint64_t u; } ptr_cast;
    ptr_cast.p = g_pfr_heatmap;
    dict_set(out, "heatmap_ptr", (double)ptr_cast.u);
    return NULL;
}

/* Non-static accessors for _C bindings */
float* pfr_get_heatmap_ptr(void) { return g_pfr_heatmap; }
int pfr_get_heatmap_h(void) { return PFR_HEATMAP_H; }
int pfr_get_heatmap_w(void) { return PFR_HEATMAP_W; }

void my_log(Log* log, Dict* out) {
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "unique_tiles", log->unique_tiles);
    dict_set(out, "unique_maps", log->unique_maps);
    dict_set(out, "warps_taken", log->warps_taken);
}
