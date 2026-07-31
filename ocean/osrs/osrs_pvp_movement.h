#ifndef OSRS_PVP_MOVEMENT_H
#define OSRS_PVP_MOVEMENT_H

#include "osrs_types.h"
#include "osrs_collision.h"
#include "osrs_encounter.h"
#include "osrs_encounter_player.h"
#include "osrs_pvp_gear.h"

static int select_closest_candidate_tile(
    Player* p, const int candidates[4][2], int target_x, int target_y,
    int* out_x, int* out_y, const CollisionMap* cmap
) {
    int has_best = 0;
    int best_x = 0;
    int best_y = 0;
    int best_dist_agent = 0;
    int best_dist_target = 0;
    int best_hash = 0;

    for (int i = 0; i < 4; i++) {
        int cx = candidates[i][0];
        int cy = candidates[i][1];
        if (!is_in_wilderness(cx, cy)) {
            continue;
        }
        if (!collision_tile_walkable(cmap, 0, cx, cy)) {
            continue;
        }
        int dist_agent = chebyshev_distance(p->x, p->y, cx, cy);
        int dist_target = chebyshev_distance(cx, cy, target_x, target_y);
        int hash = tile_hash(cx, cy);
        if (!has_best ||
            dist_agent < best_dist_agent ||
            (dist_agent == best_dist_agent &&
             (dist_target < best_dist_target ||
              (dist_target == best_dist_target && hash < best_hash)))) {
            has_best = 1;
            best_x = cx;
            best_y = cy;
            best_dist_agent = dist_agent;
            best_dist_target = dist_target;
            best_hash = hash;
        }
    }

    if (!has_best) {
        return 0;
    }
    *out_x = best_x;
    *out_y = best_y;
    return 1;
}

static int select_closest_adjacent_tile(Player* p, int target_x, int target_y, int* out_x, int* out_y, const CollisionMap* cmap) {
    const int candidates[4][2] = {
        {target_x, target_y + 1},
        {target_x + 1, target_y},
        {target_x, target_y - 1},
        {target_x - 1, target_y}
    };
    return select_closest_candidate_tile(p, candidates, target_x, target_y, out_x, out_y, cmap);
}

static int select_closest_diagonal_tile(Player* p, int target_x, int target_y, int* out_x, int* out_y, const CollisionMap* cmap) {
    const int candidates[4][2] = {
        {target_x + 1, target_y + 1},
        {target_x + 1, target_y - 1},
        {target_x - 1, target_y - 1},
        {target_x - 1, target_y + 1}
    };
    return select_closest_candidate_tile(p, candidates, target_x, target_y, out_x, out_y, cmap);
}

static int select_farcast_tile(Player* p, int target_x, int target_y, int distance, int* out_x, int* out_y, const CollisionMap* cmap) {
    int raw_dx = p->x - target_x;
    int raw_dy = p->y - target_y;
    int d = distance;

    int dx = raw_dx < -d ? -d : (raw_dx > d ? d : raw_dx);
    int dy = raw_dy < -d ? -d : (raw_dy > d ? d : raw_dy);

    int adx = abs_int(dx);
    int ady = abs_int(dy);
    if (adx < d && ady < d) {
        if (adx >= ady) {
            dx = (raw_dx >= 0) ? d : -d;
        } else {
            dy = (raw_dy >= 0) ? d : -d;
        }
    }

    int cx = target_x + dx;
    int cy = target_y + dy;

    if (is_in_wilderness(cx, cy) && collision_tile_walkable(cmap, 0, cx, cy)) {
        *out_x = cx;
        *out_y = cy;
        return 1;
    }

    cx = cx < WILD_MIN_X ? WILD_MIN_X : (cx > WILD_MAX_X ? WILD_MAX_X : cx);
    cy = cy < WILD_MIN_Y ? WILD_MIN_Y : (cy > WILD_MAX_Y ? WILD_MAX_Y : cy);
    if (chebyshev_distance(cx, cy, target_x, target_y) == distance
        && collision_tile_walkable(cmap, 0, cx, cy)) {
        *out_x = cx;
        *out_y = cy;
        return 1;
    }

    return 0;
}

static int step_toward_destination(Player* p, const CollisionMap* cmap) {
    int dx = p->dest_x - p->x;
    int dy = p->dest_y - p->y;
    if (dx == 0 && dy == 0) {
        return 0;
    }

    int step_x = (dx > 0) ? 1 : (dx < 0 ? -1 : 0);
    int step_y = (dy > 0) ? 1 : (dy < 0 ? -1 : 0);

    if (step_x != 0 && step_y != 0) {
        if (collision_traversable_step(cmap, 0, p->x, p->y, step_x, step_y)) {
            p->x += step_x;
            p->y += step_y;
            return 1;
        }
        if (collision_traversable_step(cmap, 0, p->x, p->y, step_x, 0)) {
            p->x += step_x;
            return 1;
        }
        if (collision_traversable_step(cmap, 0, p->x, p->y, 0, step_y)) {
            p->y += step_y;
            return 1;
        }
        return 0;
    }

    if (collision_traversable_step(cmap, 0, p->x, p->y, step_x, step_y)) {
        p->x += step_x;
        p->y += step_y;
        return 1;
    }

    return 0;
}

static void set_destination(Player* p, int dest_x, int dest_y, const CollisionMap* cmap) {
    p->dest_x = dest_x;
    p->dest_y = dest_y;
    if (p->x == dest_x && p->y == dest_y) {
        p->is_moving = 0;
        return;
    }
    if (!step_toward_destination(p, cmap)) {
        p->is_moving = 0;
        return;
    }
    if (p->x != dest_x || p->y != dest_y) {
        step_toward_destination(p, cmap);
    }
    p->is_moving = (p->x != dest_x || p->y != dest_y) ? 1 : 0;
}

static int pvp_tile_walkable(void* ctx, int x, int y) {
    const CollisionMap* cmap = (const CollisionMap*)ctx;
    return is_in_wilderness(x, y) && collision_tile_walkable(cmap, 0, x, y);
}

static void move_toward_target(
    Player* p,
    Player* target,
    int attack_range,
    const CollisionMap* cmap
) {
    if (p->frozen_ticks > 0) {
        return;
    }
    int moved = encounter_chase_attack_target(
        p,
        target->x,
        target->y,
        1,
        attack_range,
        cmap,
        0,
        0,
        pvp_tile_walkable,
        (void*)cmap,
        NULL,
        NULL,
        osrs_los_open_query(),
        0,
        0,
        0,
        0);
    p->is_moving = moved;
}

static void step_out_from_same_tile(Player* p, Player* target, const CollisionMap* cmap) {
    if (p->frozen_ticks > 0) {
        return;
    }

    int dest_x = target->x - 1;
    int dest_y = target->y;
    if (is_in_wilderness(dest_x, dest_y) && collision_tile_walkable(cmap, 0, dest_x, dest_y)) {
        set_destination(p, dest_x, dest_y, cmap);
        return;
    }
    dest_x = target->x + 1;
    if (is_in_wilderness(dest_x, dest_y) && collision_tile_walkable(cmap, 0, dest_x, dest_y)) {
        set_destination(p, dest_x, dest_y, cmap);
        return;
    }
    dest_x = target->x;
    dest_y = target->y - 1;
    if (is_in_wilderness(dest_x, dest_y) && collision_tile_walkable(cmap, 0, dest_x, dest_y)) {
        set_destination(p, dest_x, dest_y, cmap);
        return;
    }
    dest_y = target->y + 1;
    if (is_in_wilderness(dest_x, dest_y) && collision_tile_walkable(cmap, 0, dest_x, dest_y)) {
        set_destination(p, dest_x, dest_y, cmap);
        return;
    }
}

/* skipped when either player is frozen: walking under a frozen opponent is a
   legal, intentional position in OSRS PvP */
static void resolve_same_tile(Player* mover, Player* blocker, const CollisionMap* cmap) {
    if (blocker->frozen_ticks > 0) {
        return;
    }
    if (mover->frozen_ticks > 0) {
        return;
    }

    static const int OFFSETS[8][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1},
        {-1, -1}, {1, -1}, {-1, 1}, {1, 1}
    };

    for (int i = 0; i < 8; i++) {
        int nx = mover->x + OFFSETS[i][0];
        int ny = mover->y + OFFSETS[i][1];
        if (is_in_wilderness(nx, ny)
            && collision_tile_walkable(cmap, 0, nx, ny)
            && !(nx == blocker->x && ny == blocker->y)) {
            mover->x = nx;
            mover->y = ny;
            mover->dest_x = nx;
            mover->dest_y = ny;
            mover->is_moving = 0;
            return;
        }
    }
}

static int pvp_lookup_attack_target(void* ctx, int target_slot, OsrsAttackTarget* out) {
    if (target_slot < 0 || target_slot >= NUM_AGENTS) return 0;
    OsrsEnv* env = (OsrsEnv*)ctx;
    Player* target = &env->players[target_slot];
    Player* self = &env->players[1 - target_slot];
    AttackStyle style = get_slot_weapon_attack_style(self);
    int range;
    if (style == ATTACK_STYLE_MELEE || style == ATTACK_STYLE_NONE) {
        range = 1;
    } else {
        range = get_attack_range(self, style);
    }
    out->slot = target_slot;
    out->x = target->x;
    out->y = target->y;
    out->size = 1;
    out->attack_range = range;
    return 1;
}

static inline OsrsEncounterArena pvp_build_arena(OsrsEnv* env) {
    OsrsEncounterArena arena;
    arena.collision_map = (const CollisionMap*)env->collision_map;
    arena.world_offset_x = 0;
    arena.world_offset_y = 0;
    arena.is_walkable = pvp_tile_walkable;
    arena.walkable_ctx = (void*)arena.collision_map;
    arena.extra_blocked = NULL;
    arena.blocked_ctx = NULL;
    arena.los_query = osrs_los_open_query();
    arena.arena_base_x = 0;
    arena.arena_base_y = 0;
    arena.arena_w = 0;
    arena.arena_h = 0;
    return arena;
}

static inline OsrsPlayerStepResult pvp_step_player_movement(OsrsEnv* env, int agent_idx) {
    OsrsPlayerStepResult result = {.target_slot = -1};
    int* dest_x = &env->pvp_runtime.walk_dest_x[agent_idx];
    int* dest_y = &env->pvp_runtime.walk_dest_y[agent_idx];

    if (*dest_x < 0 || *dest_y < 0) return result;

    Player* p = &env->players[agent_idx];
    OsrsEncounterArena arena = pvp_build_arena(env);
    OsrsPlayerStepInput input = {
        .player = p,
        .interaction = &p->interaction,
        .target_lookup = pvp_lookup_attack_target,
        .target_ctx = env,
        .has_new_target = 0,
        .new_target_slot = -1,
        .move_kind = OSRS_PLAYER_MOVE_DESTINATION,
        .target_move_policy = OSRS_PLAYER_TARGET_MOVE_EXPLICIT_FIRST,
        .move_action = 0,
        .dest_x = dest_x,
        .dest_y = dest_y,
        .blocked_ticks = p->frozen_ticks,
        .arena = arena,
    };
    result = osrs_encounter_player_step(&input);
    p->is_moving = (*dest_x >= 0) ? 1 : 0;
    return result;
}

static inline void pvp_set_walk_dest_from_head_move(OsrsEnv* env, int agent_idx, int move_action) {
    Player* p = &env->players[agent_idx];
    if (move_action <= 0 || move_action >= MOVE_DIM) return;
    env->pvp_runtime.walk_dest_x[agent_idx] = p->x + ENCOUNTER_MOVE_TARGET_DX[move_action];
    env->pvp_runtime.walk_dest_y[agent_idx] = p->y + ENCOUNTER_MOVE_TARGET_DY[move_action];
}

#endif // OSRS_PVP_MOVEMENT_H
