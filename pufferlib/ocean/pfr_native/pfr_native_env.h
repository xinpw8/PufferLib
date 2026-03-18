/*
 * pfr_native.h -- PufferLib ocean env for pokefirered-native runtime
 *
 * Zero Python overhead: c_step extracts obs + computes reward entirely in C.
 * Each Env is ~1.6KB (545B core + 1KB visit hash). No global state.
 *
 * Observation (129 bytes uint8):
 *   [0:8]     scalars: player_x(2), player_y(2), map(2), dir(1), mode(1)
 *   [8:89]    9x9 tile grid: behavior | (collision << 7)
 *   [89:129]  8 NPCs x 5 bytes: dx, dy, graphics_id, facing, active
 *
 * Action: Discrete(9)
 *   0=none, 1=up, 2=down, 3=left, 4=right, 5=A, 6=B, 7=start, 8=select
 */

#ifndef PFR_NATIVE_ENV_H
#define PFR_NATIVE_ENV_H

#include <stdint.h>
#include <stdbool.h>

/* Rename the engine's c_* functions to avoid conflict with env_binding.h */
#define c_init pfr_engine_init
#define c_reset pfr_engine_reset
#define c_step pfr_engine_step
#define c_close pfr_engine_close
#define c_render pfr_engine_render
#define c_save_snapshot pfr_engine_save_snapshot
#include "pfr_native.h"
#undef c_init
#undef c_reset
#undef c_step
#undef c_close
#undef c_render
#undef c_save_snapshot

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "pfr_heatmap_lut.h"

/* ---- Observation layout ---- */

#define PFR_OBS_TILE_RADIUS  4
#define PFR_OBS_TILE_DIM     (2 * PFR_OBS_TILE_RADIUS + 1)  /* 9 */
#define PFR_OBS_TILE_SIZE    (PFR_OBS_TILE_DIM * PFR_OBS_TILE_DIM)  /* 81 */
#define PFR_OBS_SCALAR_SIZE  8
#define PFR_OBS_NPC_COUNT    8
#define PFR_OBS_NPC_FEAT     5
#define PFR_OBS_NPC_SIZE     (PFR_OBS_NPC_COUNT * PFR_OBS_NPC_FEAT)  /* 40 */
#define PFR_OBS_SIZE         (PFR_OBS_SCALAR_SIZE + PFR_OBS_TILE_SIZE + PFR_OBS_NPC_SIZE)  /* 129 */

#define PFR_NUM_ACTIONS      9
#define PFR_VISIT_HASH_SIZE  8192

/* Map visit bitset — track which distinct maps the agent has entered */
#define PFR_MAP_VISIT_MAX    512  /* covers all 425 maps */

/* Episode truncation: reset exploration state every N steps to bound returns.
 * This prevents value function divergence while preserving the heatmap accumulation.
 * 4096 steps ≈ 68 seconds at 60fps — enough time to explore a few rooms.
 * Obviously this is stupid but I've left the clanker note to illustrate the point that
 * it has absolutely 0 idea about how this works. */
#define PFR_TRUNCATION_HORIZON  65536

/* ---- Global shared heatmap (allocated once, written atomically by all envs) ---- */
static float *g_pfr_heatmap = NULL;

static void pfr_heatmap_ensure_alloc(void)
{
    if (g_pfr_heatmap == NULL) {
        g_pfr_heatmap = (float *)calloc(PFR_HEATMAP_SIZE, sizeof(float));
    }
}

static inline void pfr_heatmap_record(uint16_t map_id, int16_t px, int16_t py)
{
    if (g_pfr_heatmap == NULL) return;
    if (map_id > PFR_HEATMAP_MAX_MAP_ID) return;
    const PfrMapOffset *off = &pfr_map_offsets[map_id];
    if (off->gx < 0) return;
    int gx = (int)px + (int)off->gx;
    int gy = (int)py + (int)off->gy;
    if (gx < 0 || gx >= PFR_HEATMAP_W || gy < 0 || gy >= PFR_HEATMAP_H) return;
    /* OMP-safe: multiple threads may write different cells concurrently.
     * Same-cell races just lose a count — acceptable for visualization. */
    g_pfr_heatmap[gy * PFR_HEATMAP_W + gx] += 1.0f;
}

/* ---- Log (all floats, aggregated by env_binding.h vec_log) ---- */
/* IMPORTANT: n must be the LAST field. vec_log sums all floats, divides by n. */

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float unique_tiles;
    float unique_maps;
    float warps_taken;
    float n;
};

/* ---- Env struct ---- */

typedef struct Env Env;
struct Env {
    Log log;
    unsigned char *observations;
#ifdef PFR_STATIC_ENV
    double *actions;
#else
    float *actions;
#endif
    float *rewards;
    float *terminals;
    float *truncations;
    int num_agents;

    PfrNativeCore core;

    /* Tile visit bloom filter for exploration reward */
    uint32_t visit_bits[PFR_VISIT_HASH_SIZE / 32];
    uint32_t visit_count;

    /* Map visit bitset — which distinct maps entered */
    uint32_t map_bits[PFR_MAP_VISIT_MAX / 32];
    uint32_t map_count;

    /* Counters */
    uint32_t step_count;
    uint32_t warp_count;
    float episode_return;

    /* Position tracking for movement reward */
    int16_t last_x;
    int16_t last_y;
    uint16_t last_map;
};

/* ---- Tile hash for exploration ---- */

static uint32_t pfr_tile_hash(uint16_t map_id, int16_t x, int16_t y)
{
    uint32_t h = (uint32_t)map_id * 2654435761u;
    h += (uint32_t)(uint16_t)x * 2246822519u;
    h += (uint32_t)(uint16_t)y;
    return h % PFR_VISIT_HASH_SIZE;
}

static int pfr_visit_check_and_set(Env *env, uint16_t map_id, int16_t x, int16_t y)
{
    uint32_t idx = pfr_tile_hash(map_id, x, y);
    uint32_t word = idx / 32;
    uint32_t bit = 1u << (idx % 32);
    if (env->visit_bits[word] & bit)
        return 0;
    env->visit_bits[word] |= bit;
    env->visit_count++;
    return 1;
}

/* ---- Map visit tracking ---- */

static int pfr_map_visit_check_and_set(Env *env, uint16_t map_id)
{
    if (map_id >= PFR_MAP_VISIT_MAX)
        return 0;
    uint32_t word = map_id / 32;
    uint32_t bit = 1u << (map_id % 32);
    if (env->map_bits[word] & bit)
        return 0;
    env->map_bits[word] |= bit;
    env->map_count++;
    return 1;
}

/* ---- Snapshot log (called every step so trainer logs stay live) ---- */

static void pfr_snapshot_log(Env *env)
{
    env->log.episode_return = env->episode_return;
    env->log.episode_length = (float)env->step_count;
    env->log.unique_tiles   = (float)env->visit_count;
    env->log.unique_maps    = (float)env->map_count;
    env->log.warps_taken    = (float)env->warp_count;
    env->log.n              = 1.0f;
}

/* ---- Observation extraction (writes directly to numpy buffer) ---- */

static void pfr_extract_obs(Env *env)
{
    const PfrNativeState *state = pfr_native_state(&env->core);
    const PfrNativeMap *map = pfr_native_get_map(state->current_map);
    unsigned char *obs = env->observations;
    int dx, dy, tx, ty, i;
    int best_npcs[PFR_OBS_NPC_COUNT];
    int best_dist[PFR_OBS_NPC_COUNT];

    memset(obs, 0, PFR_OBS_SIZE);

    /* Scalars */
    obs[0] = (unsigned char)(state->player_x & 0xFF);
    obs[1] = (unsigned char)((state->player_x >> 8) & 0xFF);
    obs[2] = (unsigned char)(state->player_y & 0xFF);
    obs[3] = (unsigned char)((state->player_y >> 8) & 0xFF);
    obs[4] = (unsigned char)(state->current_map & 0xFF);
    obs[5] = (unsigned char)((state->current_map >> 8) & 0xFF);
    obs[6] = state->player_direction;
    obs[7] = state->mode;

    /* 9x9 tile grid centered on player */
    if (map != NULL)
    {
        unsigned char *tile_obs = obs + PFR_OBS_SCALAR_SIZE;
        for (dy = -PFR_OBS_TILE_RADIUS; dy <= PFR_OBS_TILE_RADIUS; dy++)
        {
            for (dx = -PFR_OBS_TILE_RADIUS; dx <= PFR_OBS_TILE_RADIUS; dx++)
            {
                tx = state->player_x + dx;
                ty = state->player_y + dy;
                if (tx >= 0 && ty >= 0 && tx < (int)map->width && ty < (int)map->height)
                {
                    const PfrNativeTile *tile = &map->tiles[(size_t)ty * map->width + tx];
                    *tile_obs = (unsigned char)((tile->behavior & 0x7F) | (tile->collision ? 0x80 : 0));
                }
                else
                {
                    *tile_obs = 0xFF;  /* out of bounds = impassable */
                }
                tile_obs++;
            }
        }
    }

    /* 8 nearest NPCs */
    for (i = 0; i < PFR_OBS_NPC_COUNT; i++)
    {
        best_npcs[i] = -1;
        best_dist[i] = 0x7FFFFFFF;
    }

    for (i = 0; i < (int)state->object_count; i++)
    {
        const PfrNativeObjectState *obj = &state->objects[i];
        int odx, ody, dist, j, worst, worst_dist;
        if (!obj->active)
            continue;
        odx = obj->x - state->player_x;
        ody = obj->y - state->player_y;
        dist = odx * odx + ody * ody;

        worst = 0;
        worst_dist = best_dist[0];
        for (j = 1; j < PFR_OBS_NPC_COUNT; j++)
        {
            if (best_dist[j] > worst_dist)
            {
                worst = j;
                worst_dist = best_dist[j];
            }
        }
        if (dist < worst_dist)
        {
            best_npcs[worst] = i;
            best_dist[worst] = dist;
        }
    }

    {
        unsigned char *npc_obs = obs + PFR_OBS_SCALAR_SIZE + PFR_OBS_TILE_SIZE;
        for (i = 0; i < PFR_OBS_NPC_COUNT; i++)
        {
            if (best_npcs[i] >= 0)
            {
                const PfrNativeObjectState *obj = &state->objects[best_npcs[i]];
                int odx = obj->x - state->player_x;
                int ody = obj->y - state->player_y;
                /* Clamp to [-127, 127] for int8 */
                if (odx < -127) odx = -127;
                if (odx > 127) odx = 127;
                if (ody < -127) ody = -127;
                if (ody > 127) ody = 127;
                npc_obs[0] = (unsigned char)(int8_t)odx;
                npc_obs[1] = (unsigned char)(int8_t)ody;
                npc_obs[2] = obj->graphics_id;
                npc_obs[3] = obj->facing;
                npc_obs[4] = 1;  /* active */
            }
            npc_obs += PFR_OBS_NPC_FEAT;
        }
    }
}

/* ---- c_reset ---- */

static void c_reset(Env *env)
{
    memset(env->visit_bits, 0, sizeof(env->visit_bits));
    memset(env->map_bits, 0, sizeof(env->map_bits));
    env->visit_count = 0;
    env->map_count = 0;
    env->step_count = 0;
    env->warp_count = 0;
    env->episode_return = 0.0f;

    pfr_engine_reset(&env->core, PFR_NATIVE_BOOTSTRAP_PALLET_TOWN, NULL);
    pfr_extract_obs(env);

    /* Mark starting tile and map visited */
    {
        const PfrNativeState *state = pfr_native_state(&env->core);
        pfr_visit_check_and_set(env, state->current_map, state->player_x, state->player_y);
        pfr_map_visit_check_and_set(env, state->current_map);
    }

    /* Init position tracking */
    {
        const PfrNativeState *s = pfr_native_state(&env->core);
        env->last_x = s->player_x;
        env->last_y = s->player_y;
        env->last_map = s->current_map;
    }

    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
}

/* ---- c_step ---- */

static void c_step(Env *env)
{
    const PfrNativeState *state;
    PfrNativeStepResult result;
    float reward = 0.0f;
    int action;

    /* 1. Read action */
    action = (int)env->actions[0];
    if (action < 0 || action >= PFR_NUM_ACTIONS)
        action = 0;

    /* 2. Step engine */
    result = pfr_engine_step(&env->core, (PfrNativeAction)action);


    /* 3. Extract observation */
    pfr_extract_obs(env);

    /* 4. Get current state */
    state = pfr_native_state(&env->core);

    /* 5. Record position in global heatmap */
    pfr_heatmap_record(state->current_map, state->player_x, state->player_y);

    /* 5b. Movement reward: small reward for changing position */
    if (state->player_x != env->last_x || state->player_y != env->last_y ||
        state->current_map != env->last_map) {
        reward += 0.01f;
        env->last_x = state->player_x;
        env->last_y = state->player_y;
        env->last_map = state->current_map;
    }

    /* 6. Exploration reward: new tile */
    if (pfr_visit_check_and_set(env, state->current_map, state->player_x, state->player_y))
        reward += 0.02f;

    /* 7. Track map visits + warp reward */
    if (pfr_map_visit_check_and_set(env, state->current_map))
        reward += 0.1f;  /* reward for discovering a new map */

    /* 7. Track warps */
    if (result.event == PFR_NATIVE_EVENT_WARPED)
        env->warp_count++;

    env->rewards[0] = reward;
    env->episode_return += reward;
    env->step_count++;

    /* 8. Publish current episode stats continuously for dashboard/wandb. */
    pfr_snapshot_log(env);

    /* 9. Periodic episode reset to keep exploration reward flowing.
     * Full reset: clear visits, warp back to Pallet Town, fresh episode.
     * Heatmap accumulates across resets for visualization. */
    if (env->step_count >= PFR_TRUNCATION_HORIZON) {
        pfr_snapshot_log(env);
        env->terminals[0] = 1;
        c_reset(env);
    } else {
        env->terminals[0] = 0;
    }
}

/* ---- allocate / free (standalone binary only) ---- */

#ifdef PFR_STATIC_ENV
#include "raylib.h"

static void allocate(Env *env)
{
    env->observations = (unsigned char *)calloc(PFR_OBS_SIZE, sizeof(unsigned char));
    env->actions = (double *)calloc(1, sizeof(double));
    env->rewards = (float *)calloc(1, sizeof(float));
    env->terminals = (float *)calloc(1, sizeof(float));
    env->truncations = (float *)calloc(1, sizeof(float));
    env->num_agents = 1;
    pfr_engine_init(&env->core);
}

static void free_allocated(Env *env)
{
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->truncations);
}
#endif /* PFR_STATIC_ENV */

/* ---- c_render ---- */

static void c_render(Env *env)
{
#ifdef PFR_STATIC_ENV
    #define PFR_WINDOW_W  660
    #define PFR_WINDOW_H  660
    #define PFR_TILE_SIZE 50
    #define PFR_GRID_X    ((PFR_WINDOW_W - PFR_OBS_TILE_DIM * PFR_TILE_SIZE) / 2)
    #define PFR_GRID_Y    80

    /* Window must be initialized by main() before entering the game loop.
     * Lazy init here was the cause of the CloseWindow SEGV — see pfr_native.c. */
    if (!IsWindowReady()) return;

    BeginDrawing();
    ClearBackground(BLACK);

    const unsigned char *obs = env->observations;

    /* --- Decode scalars from obs --- */
    int player_x = obs[0] | (obs[1] << 8);
    int player_y = obs[2] | (obs[3] << 8);
    int map_id   = obs[4] | (obs[5] << 8);
    int direction = obs[6];
    int mode      = obs[7];

    /* --- Draw 9x9 tile grid --- */
    const unsigned char *tiles = obs + PFR_OBS_SCALAR_SIZE;
    for (int dy = 0; dy < PFR_OBS_TILE_DIM; dy++) {
        for (int dx = 0; dx < PFR_OBS_TILE_DIM; dx++) {
            int idx = dy * PFR_OBS_TILE_DIM + dx;
            unsigned char tile = tiles[idx];
            int px = PFR_GRID_X + dx * PFR_TILE_SIZE;
            int py = PFR_GRID_Y + dy * PFR_TILE_SIZE;

            Color col;
            if (tile == 0xFF) {
                col = (Color){20, 20, 20, 255};  /* OOB */
            } else {
                unsigned char behavior = tile & 0x7F;
                switch (behavior) {
                    case 0x00: col = (Color){200, 200, 200, 255}; break; /* normal ground */
                    case 0x04: col = (Color){60, 120, 220, 255};  break; /* surfable water */
                    case 0x0C: col = (Color){30, 140, 40, 255};   break; /* tall grass */
                    case 0x3B: col = (Color){160, 100, 50, 255};  break; /* ledge */
                    case 0x01: col = (Color){180, 180, 180, 255}; break; /* wall/impassable */
                    case 0x02: col = (Color){220, 200, 160, 255}; break; /* sand */
                    case 0x08: col = (Color){100, 160, 100, 255}; break; /* short grass */
                    case 0x09: col = (Color){70, 70, 70, 255};    break; /* cave floor */
                    case 0x10: col = (Color){180, 180, 220, 255}; break; /* indoor floor */
                    case 0x0A: col = (Color){50, 100, 180, 255};  break; /* deep water */
                    default:   col = (Color){140, 140, 140, 255}; break; /* unknown */
                }
            }

            DrawRectangle(px, py, PFR_TILE_SIZE - 1, PFR_TILE_SIZE - 1, col);

            /* Collision border (red) */
            if (tile != 0xFF && (tile & 0x80)) {
                DrawRectangleLines(px, py, PFR_TILE_SIZE - 1, PFR_TILE_SIZE - 1, RED);
            }

            /* Player highlight (center tile) */
            if (dx == PFR_OBS_TILE_RADIUS && dy == PFR_OBS_TILE_RADIUS) {
                DrawRectangleLines(px + 2, py + 2, PFR_TILE_SIZE - 5, PFR_TILE_SIZE - 5, YELLOW);
                DrawRectangleLines(px + 3, py + 3, PFR_TILE_SIZE - 7, PFR_TILE_SIZE - 7, YELLOW);
            }
        }
    }

    /* --- Draw NPC markers --- */
    const unsigned char *npcs = obs + PFR_OBS_SCALAR_SIZE + PFR_OBS_TILE_SIZE;
    for (int i = 0; i < PFR_OBS_NPC_COUNT; i++) {
        int ndx    = (int)(int8_t)npcs[i * PFR_OBS_NPC_FEAT + 0];
        int ndy    = (int)(int8_t)npcs[i * PFR_OBS_NPC_FEAT + 1];
        int active = npcs[i * PFR_OBS_NPC_FEAT + 4];
        if (!active) continue;

        /* Map NPC relative position onto grid (centered on player tile) */
        float nx = PFR_GRID_X + (PFR_OBS_TILE_RADIUS + ndx + 0.5f) * PFR_TILE_SIZE;
        float ny = PFR_GRID_Y + (PFR_OBS_TILE_RADIUS + ndy + 0.5f) * PFR_TILE_SIZE;

        /* Only draw if within the grid area */
        if (nx >= PFR_GRID_X && nx < PFR_GRID_X + PFR_OBS_TILE_DIM * PFR_TILE_SIZE &&
            ny >= PFR_GRID_Y && ny < PFR_GRID_Y + PFR_OBS_TILE_DIM * PFR_TILE_SIZE) {
            DrawCircle((int)nx, (int)ny, 8, MAGENTA);
            DrawCircleLines((int)nx, (int)ny, 8, WHITE);
        }
    }

    /* --- HUD --- */
    const char *dir_names[] = {"Down", "Up", "Left", "Right"};
    const char *dir_str = (direction < 4) ? dir_names[direction] : "?";

    DrawText(TextFormat("Map: %d  Pos: (%d, %d)  Dir: %s  Mode: %d",
        map_id, player_x, player_y, dir_str, mode),
        10, 10, 20, RAYWHITE);

    DrawText(TextFormat("Step: %u  Tiles: %u  Maps: %u  Warps: %u",
        env->step_count, env->visit_count, env->map_count, env->warp_count),
        10, 35, 20, RAYWHITE);

    DrawText(TextFormat("Reward: %.2f  Episode Return: %.3f",
        env->rewards[0], env->episode_return),
        10, 60, 20, GREEN);

    /* Direction indicator on player tile */
    int cx = PFR_GRID_X + PFR_OBS_TILE_RADIUS * PFR_TILE_SIZE + PFR_TILE_SIZE / 2;
    int cy = PFR_GRID_Y + PFR_OBS_TILE_RADIUS * PFR_TILE_SIZE + PFR_TILE_SIZE / 2;
    int arrow_len = 12;
    switch (direction) {
        case 0: DrawLine(cx, cy, cx, cy + arrow_len, YELLOW); break; /* Down */
        case 1: DrawLine(cx, cy, cx, cy - arrow_len, YELLOW); break; /* Up */
        case 2: DrawLine(cx, cy, cx - arrow_len, cy, YELLOW); break; /* Left */
        case 3: DrawLine(cx, cy, cx + arrow_len, cy, YELLOW); break; /* Right */
    }

    /* Controls help */
    DrawText("Arrows: Move  Z: A  X: B  Enter: Start  Backspace: Select",
        10, PFR_WINDOW_H - 25, 16, GRAY);

    EndDrawing();
#else
    (void)env;
#endif /* PFR_STATIC_ENV */
}

/* ---- c_close ---- */

static void c_close(Env *env)
{
    (void)env;
}

#endif /* PFR_NATIVE_ENV_H */
