#ifndef OSRS_PVP_MOVEMENT_H
#define OSRS_PVP_MOVEMENT_H

#include "osrs_types.h"
#include "osrs_collision.h"
#include "osrs_encounter.h"
#include "osrs_encounter_player.h"
#include "osrs_pvp_gear.h"

static inline int pvp_topology_destination_selectable(
    const EncounterArenaTopology* topology,
    int x,
    int y
) {
    if (!is_in_wilderness(x, y)) return 0;
    if (encounter_arena_topology_contains(topology, x, y))
        return !encounter_arena_topology_tile_blocked(topology, x, y);

    int min_x = topology->origin_x;
    int max_x = min_x + topology->width - 1;
    int min_y = topology->origin_y;
    int max_y = min_y + topology->height - 1;
    int fallback_x = x < min_x ? min_x : (x > max_x ? max_x : x);
    int fallback_y = y < min_y ? min_y : (y > max_y ? max_y : y);
    return !encounter_arena_topology_tile_blocked(
        topology, fallback_x, fallback_y);
}

static int select_closest_candidate_tile(
    Player* p,
    const int candidates[4][2],
    int target_x,
    int target_y,
    int* out_x,
    int* out_y,
    const EncounterArenaTopology* topology
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
        if (!pvp_topology_destination_selectable(topology, cx, cy))
            continue;
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

    if (!has_best) return 0;
    *out_x = best_x;
    *out_y = best_y;
    return 1;
}

static int select_closest_adjacent_tile(
    Player* p,
    int target_x,
    int target_y,
    int* out_x,
    int* out_y,
    const EncounterArenaTopology* topology
) {
    const int candidates[4][2] = {
        {target_x, target_y + 1},
        {target_x + 1, target_y},
        {target_x, target_y - 1},
        {target_x - 1, target_y}
    };
    return select_closest_candidate_tile(
        p, candidates, target_x, target_y, out_x, out_y, topology);
}

static int select_closest_diagonal_tile(
    Player* p,
    int target_x,
    int target_y,
    int* out_x,
    int* out_y,
    const EncounterArenaTopology* topology
) {
    const int candidates[4][2] = {
        {target_x + 1, target_y + 1},
        {target_x + 1, target_y - 1},
        {target_x - 1, target_y - 1},
        {target_x - 1, target_y + 1}
    };
    return select_closest_candidate_tile(
        p, candidates, target_x, target_y, out_x, out_y, topology);
}

static int select_farcast_tile(
    Player* p,
    int target_x,
    int target_y,
    int distance,
    int* out_x,
    int* out_y,
    const EncounterArenaTopology* topology
) {
    int raw_dx = p->x - target_x;
    int raw_dy = p->y - target_y;
    int d = distance;
    int dx = raw_dx < -d ? -d : (raw_dx > d ? d : raw_dx);
    int dy = raw_dy < -d ? -d : (raw_dy > d ? d : raw_dy);
    int adx = abs_int(dx);
    int ady = abs_int(dy);
    if (adx < d && ady < d) {
        if (adx >= ady)
            dx = raw_dx >= 0 ? d : -d;
        else
            dy = raw_dy >= 0 ? d : -d;
    }

    int cx = target_x + dx;
    int cy = target_y + dy;
    if (pvp_topology_destination_selectable(topology, cx, cy)) {
        *out_x = cx;
        *out_y = cy;
        return 1;
    }

    return 0;
}


typedef struct {
    EncounterArenaTopology* topology;
} PvpRouteTopologyOwner;

static PvpRouteTopologyOwner pvp_route_topology_owner;

static uint32_t pvp_route_topology_flags(void* data, int x, int y) {
    const CollisionMap* collision_map = (const CollisionMap*)data;
    if (!is_in_wilderness(x, y)) return COLLISION_BLOCKED | LOS_FULL_MASK;
    return collision_map
        ? (uint32_t)collision_get_flags(collision_map, 0, x, y)
        : 0;
}


static inline int pvp_topology_tile_walkable(
    const EncounterArenaTopology* topology,
    int x,
    int y
) {
    return !encounter_arena_topology_tile_blocked(topology, x, y);
}

static const EncounterArenaTopology* pvp_route_topology_finalize(
    const CollisionMap* collision_map
) {
    EncounterArenaTopologyBuildSpec spec = {
        .origin_x = FIGHT_AREA_BASE_X,
        .origin_y = FIGHT_AREA_BASE_Y,
        .width = FIGHT_AREA_WIDTH,
        .height = FIGHT_AREA_HEIGHT,
        .max_footprint_size = 1,
        .revision = UINT64_C(0x4e48505650000004),
        .tile_flags = pvp_route_topology_flags,
        .tile_flags_ctx = (void*)collision_map,
        .los_build_mode = ENCOUNTER_ARENA_TOPOLOGY_LOS_BUILD_OPEN,
    };
    if (!pvp_route_topology_owner.topology) {
        pvp_route_topology_owner.topology =
            encounter_arena_topology_build(&spec);
        encounter_arena_topology_finalize(pvp_route_topology_owner.topology);
    } else {
        encounter_arena_topology_require_spec(
            pvp_route_topology_owner.topology,
            &spec,
            "nh_pvp");
    }
    return pvp_route_topology_owner.topology;
}



/* skipped when either player is frozen: walking under a frozen opponent is a
   legal, intentional position in OSRS PvP */
static void resolve_same_tile(
    Player* mover,
    Player* blocker,
    const EncounterArenaTopology* topology
) {
    if (blocker->frozen_ticks > 0 || mover->frozen_ticks > 0) return;

    static const int OFFSETS[8][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1},
        {-1, -1}, {1, -1}, {-1, 1}, {1, 1}
    };

    for (int i = 0; i < 8; i++) {
        int nx = mover->x + OFFSETS[i][0];
        int ny = mover->y + OFFSETS[i][1];
        if (pvp_topology_tile_walkable(topology, nx, ny) &&
                !(nx == blocker->x && ny == blocker->y)) {
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

static inline OsrsEncounterArena pvp_build_arena(
    const EncounterArenaTopology* topology
) {
    return (OsrsEncounterArena){
        .topology = topology,
        .blockers = {0},
        .movement_mode = ENCOUNTER_ROUTE_MOVEMENT_RUN,
        .cost_policy = ENCOUNTER_ROUTE_COST_OSRS_TARGET_BFS,
        .destination_cost_policy = ENCOUNTER_ROUTE_COST_SOUTH_FIRST_BFS,
        .attack_geometry = ENCOUNTER_ROUTE_ATTACK_GEOMETRY_TOPOLOGY,
    };
}

static inline OsrsPlayerStepResult pvp_step_player_movement(
    OsrsEnv* env,
    int agent_idx,
    const EncounterArenaTopology* topology,
    OsrsActorRouteCache* route_cache
) {
    OsrsPlayerStepResult result = {.target_slot = -1};
    int* dest_x = &env->pvp_runtime.walk_dest_x[agent_idx];
    int* dest_y = &env->pvp_runtime.walk_dest_y[agent_idx];

    if (*dest_x < 0 || *dest_y < 0) return result;

    Player* p = &env->players[agent_idx];
    OsrsEncounterArena arena = pvp_build_arena(topology);
    OsrsPlayerStepInput input = {
        .player = p,
        .interaction = &p->interaction,
        .route_cache = route_cache,
        .target_lookup = pvp_lookup_attack_target,
        .target_ctx = env,
        .command = {
            .kind = OSRS_PLAYER_CMD_MOVE,
            .move_kind = OSRS_PLAYER_MOVE_DESTINATION,
        },
        .dest_x = dest_x,
        .dest_y = dest_y,
        .blocked_ticks = p->frozen_ticks,
        .arena = arena,
    };
    result = osrs_encounter_player_step(&input);
    p->is_moving = (*dest_x >= 0) ? 1 : 0;
    return result;
}

static inline int pvp_step_player_melee_chase(
    OsrsEnv* env,
    int agent_idx,
    const EncounterArenaTopology* topology,
    OsrsActorRouteCache* route_cache
) {
    Player* player = &env->players[agent_idx];
    Player* target = &env->players[1 - agent_idx];
    int destination_x = 0;
    int destination_y = 0;
    if (!select_closest_adjacent_tile(
            player,
            target->x,
            target->y,
            &destination_x,
            &destination_y,
            topology))
        return 0;

    EncounterRouteInput route_input = {
        .topology = topology,
        .source_x = player->x,
        .source_y = player->y,
        .actor_size = 1,
        .target_x = destination_x,
        .target_y = destination_y,
        .target_size = 1,
        .target_kind = ENCOUNTER_ROUTE_TARGET_TILE,
        .movement_mode = ENCOUNTER_ROUTE_MOVEMENT_RUN,
        .cost_policy = ENCOUNTER_ROUTE_COST_DIRECT,
    };
    EncounterRouteResult route =
        encounter_route_greedy_direct(&route_input);
    int moved = osrs_player_step_apply_route(player, &route);
    player->dest_x = destination_x;
    player->dest_y = destination_y;
    player->is_moving =
        player->x != destination_x || player->y != destination_y;
    osrs_actor_route_cache_clear(route_cache);
    return moved;
}

static inline int pvp_step_player_ranged_chase(
    OsrsEnv* env,
    int agent_idx,
    int attack_range,
    const EncounterArenaTopology* topology,
    OsrsActorRouteCache* route_cache
) {
    Player* player = &env->players[agent_idx];
    Player* target = &env->players[1 - agent_idx];
    EncounterRouteInput route_input = {
        .topology = topology,
        .source_x = player->x,
        .source_y = player->y,
        .actor_size = 1,
        .target_x = target->x,
        .target_y = target->y,
        .target_size = 1,
        .target_kind = ENCOUNTER_ROUTE_TARGET_CARDINAL_ADJACENCY,
        .movement_mode = ENCOUNTER_ROUTE_MOVEMENT_RUN,
        .cost_policy = ENCOUNTER_ROUTE_COST_OSRS,
    };
    EncounterRouteResult route = encounter_route_solve(&route_input);
    if (route.outcome != ROUTE_REACHED_TARGET &&
            route.outcome != ROUTE_REACHED_FALLBACK) {
        player->is_moving = 0;
        return 0;
    }

    int moved = route.first_dx != 0 || route.first_dy != 0;
    player->x += route.first_dx;
    player->y += route.first_dy;
    if ((route.run_dx != 0 || route.run_dy != 0) &&
            !encounter_arena_topology_player_can_attack(
                topology,
                player->x,
                player->y,
                target->x,
                target->y,
                1,
                attack_range)) {
        player->x += route.run_dx;
        player->y += route.run_dy;
        moved = 1;
    }
    player->is_moving = moved;
    osrs_actor_route_cache_clear(route_cache);
    return moved;
}


static inline void pvp_set_walk_dest_from_head_move(OsrsEnv* env, int agent_idx, int move_action) {
    Player* p = &env->players[agent_idx];
    if (move_action <= 0 || move_action >= MOVE_DIM) return;
    env->pvp_runtime.walk_dest_x[agent_idx] = p->x + ENCOUNTER_MOVE_TARGET_DX[move_action];
    env->pvp_runtime.walk_dest_y[agent_idx] = p->y + ENCOUNTER_MOVE_TARGET_DY[move_action];
}

#endif // OSRS_PVP_MOVEMENT_H
