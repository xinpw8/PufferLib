#ifndef OSRS_PATHFINDING_H
#define OSRS_PATHFINDING_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "osrs_collision.h"

static inline int encounter_attack_rect_distance(
    int ax,
    int ay,
    int asize,
    int bx,
    int by,
    int bsize
) {
    int amax_x = ax + asize - 1;
    int amax_y = ay + asize - 1;
    int bmax_x = bx + bsize - 1;
    int bmax_y = by + bsize - 1;
    int dx = amax_x < bx ? bx - amax_x : (bmax_x < ax ? ax - bmax_x : 0);
    int dy = amax_y < by ? by - amax_y : (bmax_y < ay ? ay - bmax_y : 0);
    return dx > dy ? dx : dy;
}

static inline int encounter_entity_footprint_cardinal_reachable(
    const CollisionMap* cmap,
    int world_offset_x,
    int world_offset_y,
    int player_x,
    int player_y,
    int target_x,
    int target_y,
    int target_size
) {
    int target_max_x = target_x + target_size - 1;
    int target_max_y = target_y + target_size - 1;
    int flags = collision_get_flags(
        cmap, 0, player_x + world_offset_x, player_y + world_offset_y);

    if (player_x + 1 == target_x &&
            player_y >= target_y && player_y <= target_max_y)
        return (flags & COLLISION_WALL_EAST) == 0;
    if (player_x == target_max_x + 1 &&
            player_y >= target_y && player_y <= target_max_y)
        return (flags & COLLISION_WALL_WEST) == 0;
    if (player_y + 1 == target_y &&
            player_x >= target_x && player_x <= target_max_x)
        return (flags & COLLISION_WALL_NORTH) == 0;
    if (player_y == target_max_y + 1 &&
            player_x >= target_x && player_x <= target_max_x)
        return (flags & COLLISION_WALL_SOUTH) == 0;
    return 0;
}

static inline int encounter_entity_footprints_overlap(
    int ax, int ay, int a_size,
    int bx, int by, int b_size
) {
    return !(ax + a_size <= bx || bx + b_size <= ax ||
             ay + a_size <= by || by + b_size <= ay);
}

typedef enum {
    OSRS_LOS_OPEN = 0,
    OSRS_LOS_BLOCKERS,
    OSRS_LOS_TILE,
    OSRS_LOS_FLAGS,
} OsrsLosKind;

typedef struct {
    OsrsLosKind kind;
    const LOSBlocker* blockers;
    int blocker_count;
    los_tile_blocked_fn tile_blocked;
    los_tile_flags_fn tile_flags;
    void* tile_ctx;
} OsrsLosQuery;

static inline OsrsLosQuery osrs_los_open(void) {
    return (OsrsLosQuery){.kind = OSRS_LOS_OPEN};
}

static inline OsrsLosQuery osrs_los_blockers(
    const LOSBlocker* blockers,
    int blocker_count
) {
    return (OsrsLosQuery){
        .kind = OSRS_LOS_BLOCKERS,
        .blockers = blockers,
        .blocker_count = blocker_count,
    };
}

static inline OsrsLosQuery osrs_los_tile(
    los_tile_blocked_fn tile_blocked,
    void* tile_ctx
) {
    return (OsrsLosQuery){
        .kind = OSRS_LOS_TILE,
        .tile_blocked = tile_blocked,
        .tile_ctx = tile_ctx,
    };
}

static inline OsrsLosQuery osrs_los_flags(
    los_tile_flags_fn tile_flags,
    void* tile_ctx
) {
    return (OsrsLosQuery){
        .kind = OSRS_LOS_FLAGS,
        .tile_flags = tile_flags,
        .tile_ctx = tile_ctx,
    };
}

static inline const OsrsLosQuery* osrs_los_open_query(void) {
    static const OsrsLosQuery query = {.kind = OSRS_LOS_OPEN};
    return &query;
}

static inline int osrs_los_query_valid(
    const OsrsLosQuery* query,
    int attack_range
) {
    if (attack_range <= 1) return 1;
    if (!query || query->kind < OSRS_LOS_OPEN ||
            query->kind > OSRS_LOS_FLAGS)
        return 0;
    if (query->kind == OSRS_LOS_BLOCKERS &&
            (query->blocker_count < 0 ||
             (query->blocker_count > 0 && !query->blockers)))
        return 0;
    if (query->kind == OSRS_LOS_TILE && !query->tile_blocked)
        return 0;
    return query->kind != OSRS_LOS_FLAGS || query->tile_flags;
}

static inline void osrs_los_require_query(
    const OsrsLosQuery* query,
    int attack_range
) {
    if (osrs_los_query_valid(query, attack_range)) return;
    fprintf(stderr, "invalid OSRS LoS query for attack range %d\n", attack_range);
    abort();
}


static inline int osrs_los_clear(
    const OsrsLosQuery* query,
    int px, int py, int psize,
    int tx, int ty, int tsize,
    int attack_range
) {
    osrs_los_require_query(query, attack_range);
    if (attack_range <= 1) return 1;

    switch (query->kind) {
        case OSRS_LOS_OPEN:
            return 1;

        case OSRS_LOS_BLOCKERS:
            return entity_has_line_of_sight(
                query->blockers,
                query->blocker_count,
                px,
                py,
                psize,
                tx,
                ty,
                tsize,
                attack_range);

        case OSRS_LOS_TILE: {
            int p_los_x = tx;
            if (p_los_x < px) p_los_x = px;
            if (p_los_x >= px + psize) p_los_x = px + psize - 1;
            int p_los_y = ty;
            if (p_los_y < py) p_los_y = py;
            if (p_los_y >= py + psize) p_los_y = py + psize - 1;

            int t_los_x = px;
            if (t_los_x < tx) t_los_x = tx;
            if (t_los_x >= tx + tsize) t_los_x = tx + tsize - 1;
            int t_los_y = py;
            if (t_los_y < ty) t_los_y = ty;
            if (t_los_y >= ty + tsize) t_los_y = ty + tsize - 1;

            return los_tile_ray_clear(
                query->tile_blocked,
                query->tile_ctx,
                t_los_x,
                t_los_y,
                p_los_x,
                p_los_y);
        }

        case OSRS_LOS_FLAGS:
            return entity_has_line_of_sight_with_flags(
                query->tile_flags,
                query->tile_ctx,
                px,
                py,
                psize,
                tx,
                ty,
                tsize,
                attack_range);
    }

    fprintf(stderr, "unhandled OSRS LoS query kind: %d\n", (int)query->kind);
    abort();
}

static inline int encounter_attack_position_valid(
    int player_x,
    int player_y,
    int target_x,
    int target_y,
    int target_size,
    int attack_range,
    const CollisionMap* cmap,
    int world_offset_x,
    int world_offset_y,
    const OsrsLosQuery* los_query
) {
    int distance = encounter_attack_rect_distance(
        player_x, player_y, 1, target_x, target_y, target_size);
    if (distance < 1 || distance > attack_range) return 0;
    if (attack_range == 1)
        return encounter_entity_footprint_cardinal_reachable(
            cmap, world_offset_x, world_offset_y,
            player_x, player_y, target_x, target_y, target_size);
    return osrs_los_clear(
        los_query,
        player_x, player_y, 1,
        target_x, target_y, target_size,
        attack_range);
}

static inline int encounter_player_can_attack(
    int player_x, int player_y,
    int target_x, int target_y, int target_size, int attack_range,
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    const OsrsLosQuery* los_query
) {
    return encounter_attack_position_valid(
        player_x, player_y,
        target_x, target_y, target_size, attack_range,
        cmap, world_offset_x, world_offset_y, los_query);
}
#ifdef __cplusplus
#define OSRS_THREAD_LOCAL thread_local
#else
#define OSRS_THREAD_LOCAL _Thread_local
#endif

#define PATHFIND_MAX_FALLBACK_RADIUS 10

#if defined(__GNUC__) || defined(__clang__)
#define OSRS_ROUTE_NOINLINE __attribute__((noinline))
#else
#define OSRS_ROUTE_NOINLINE
#endif
#define VIA_NONE  0
#define VIA_S     1
#define VIA_W     2
#define VIA_SW    3
#define VIA_N     4
#define VIA_NW    6
#define VIA_E     8
#define VIA_SE    9
#define VIA_NE    12
#define VIA_START 99
static const int8_t encounter_route_osrs_dx[8] =
    {-1, 1, 0, 0, -1, 1, -1, 1};
static const int8_t encounter_route_osrs_dy[8] =
    {0, 0, -1, 1, -1, -1, 1, 1};
static const int8_t encounter_route_osrs_via[8] =
    {VIA_W, VIA_E, VIA_S, VIA_N, VIA_SW, VIA_SE, VIA_NW, VIA_NE};
static const uint8_t encounter_route_osrs_step_mask[8] =
    {8, 16, 2, 64, 1, 4, 32, 128};
static const int8_t encounter_route_south_dx[8] =
    {0, -1, 0, 1, -1, -1, 1, 1};
static const int8_t encounter_route_south_dy[8] =
    {-1, 0, 1, 0, -1, 1, -1, 1};
static const int8_t encounter_route_south_via[8] =
    {VIA_S, VIA_W, VIA_N, VIA_E, VIA_SW, VIA_NW, VIA_SE, VIA_NE};
static const uint8_t encounter_route_south_step_mask[8] =
    {2, 8, 64, 16, 1, 32, 4, 128};


#define ENCOUNTER_ROUTE_MAX_WAYPOINTS 25

typedef enum {
    ROUTE_REACHED_TARGET = 0,
    ROUTE_REACHED_FALLBACK,
    ROUTE_UNREACHABLE,
    ROUTE_INVALID_INPUT,
} EncounterRouteOutcome;

typedef enum {
    ENCOUNTER_ROUTE_TARGET_TILE = 0,
    ENCOUNTER_ROUTE_TARGET_CARDINAL_ADJACENCY,
    ENCOUNTER_ROUTE_TARGET_ATTACK_RANGE,
} EncounterRouteTargetKind;
typedef enum {
    ENCOUNTER_ROUTE_MOVEMENT_WALK = 0,
    ENCOUNTER_ROUTE_MOVEMENT_RUN,
} EncounterRouteMovementMode;
typedef enum {
    ENCOUNTER_ROUTE_RESULT_FULL = 0,
    ENCOUNTER_ROUTE_RESULT_NEXT_STEPS,
} EncounterRouteResultDetail;


typedef enum {
    ENCOUNTER_ROUTE_ATTACK_GEOMETRY_QUERY = 0,
    ENCOUNTER_ROUTE_ATTACK_GEOMETRY_TOPOLOGY,
} EncounterRouteAttackGeometry;

typedef enum {
    ENCOUNTER_ROUTE_COST_OSRS = 0,
    ENCOUNTER_ROUTE_COST_SOUTH_FIRST,
    ENCOUNTER_ROUTE_COST_DIRECT,
    ENCOUNTER_ROUTE_COST_SOUTH_FIRST_BFS,
    ENCOUNTER_ROUTE_COST_SOUTH_FIRST_REVERSE,
    ENCOUNTER_ROUTE_COST_OSRS_TARGET_BFS,
} EncounterRouteCostPolicy;
static inline int encounter_route_cost_is_osrs(
    EncounterRouteCostPolicy policy
) {
    return policy == ENCOUNTER_ROUTE_COST_OSRS ||
        policy == ENCOUNTER_ROUTE_COST_OSRS_TARGET_BFS;
}

typedef int (*encounter_route_blocked_fn)(
    void* ctx,
    int x,
    int y,
    int size);

typedef struct {
    encounter_route_blocked_fn is_blocked;
    void* ctx;
    uint64_t revision;
} EncounterRouteBlockers;

typedef struct {
    const EncounterArenaTopology* topology;
    EncounterRouteBlockers blockers;
    int source_x;
    int source_y;
    int actor_size;
    int target_x;
    int target_y;
    int target_size;
    EncounterRouteTargetKind target_kind;
    int attack_range;
    EncounterRouteAttackGeometry attack_geometry;
    const CollisionMap* collision_map;
    int world_offset_x;
    int world_offset_y;
    const OsrsLosQuery* los_query;
    EncounterRouteMovementMode movement_mode;
    EncounterRouteCostPolicy cost_policy;
    EncounterRouteResultDetail result_detail;
} EncounterRouteInput;

typedef struct {
    EncounterRouteOutcome outcome;
    int destination_x;
    int destination_y;
    int first_dx;
    int first_dy;
    int run_dx;
    int run_dy;
    uint16_t distance;
    uint8_t waypoint_count;
    int waypoint_x[ENCOUNTER_ROUTE_MAX_WAYPOINTS];
    int waypoint_y[ENCOUNTER_ROUTE_MAX_WAYPOINTS];
} EncounterRouteResult;

typedef struct {
    const EncounterArenaTopology* topology;
    void* blocker_ctx;
    encounter_route_blocked_fn blocker;
    uint64_t topology_revision;
    uint64_t blocker_revision;
    int source_x;
    int source_y;
    uint16_t visited_count;
    uint16_t expanded_count;
    uint16_t depth[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint16_t queue[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    int8_t via[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint64_t visited[ENCOUNTER_ARENA_TOPOLOGY_MAX_DIMENSION];
    uint64_t blocker_known[ENCOUNTER_ARENA_TOPOLOGY_MAX_DIMENSION];
    uint64_t blocker_value[ENCOUNTER_ARENA_TOPOLOGY_MAX_DIMENSION];
    uint8_t actor_size;
    uint8_t valid;
} EncounterSourceRouteField;
#define ENCOUNTER_SOURCE_ROUTE_CACHE_SETS 32
#define ENCOUNTER_SOURCE_ROUTE_CACHE_WAYS 8
#define ENCOUNTER_SOURCE_ROUTE_CACHE_SLOTS \
    (ENCOUNTER_SOURCE_ROUTE_CACHE_SETS * ENCOUNTER_SOURCE_ROUTE_CACHE_WAYS)


#define ENCOUNTER_REVERSE_ROUTE_CACHE_SLOTS 4

typedef struct {
    const EncounterArenaTopology* topology;
    void* blocker_ctx;
    encounter_route_blocked_fn blocker;
    uint64_t topology_revision;
    uint64_t blocker_revision;
    int target_x;
    int target_y;
    uint32_t depth_generation[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint16_t queue[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint16_t head;
    uint16_t tail;
    uint16_t generation;
    uint8_t actor_size;
    uint8_t target_size;
    uint8_t outcome;
    uint8_t valid;
} EncounterReverseRouteField;

typedef struct {
    uint16_t generation[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint16_t target_generation[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint16_t queue[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint16_t depth[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    int8_t via[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint16_t blocker_generation[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint8_t blocker_value[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint16_t current_generation;
    EncounterSourceRouteField
        source_fields[ENCOUNTER_SOURCE_ROUTE_CACHE_SLOTS];
    EncounterReverseRouteField
        reverse_fields[ENCOUNTER_REVERSE_ROUTE_CACHE_SLOTS];
    uint8_t next_reverse_field;
    uint8_t next_source_field[ENCOUNTER_SOURCE_ROUTE_CACHE_SETS];
} EncounterRouteScratch;

static OSRS_THREAD_LOCAL EncounterRouteScratch encounter_route_scratch;
static inline uint16_t encounter_route_next_generation(
    EncounterRouteScratch* scratch
) {
    scratch->current_generation++;
    if (scratch->current_generation != 0)
        return scratch->current_generation;
    memset(scratch->generation, 0, sizeof(scratch->generation));
    memset(scratch->target_generation, 0, sizeof(scratch->target_generation));
    memset(scratch->blocker_generation, 0, sizeof(scratch->blocker_generation));
    return ++scratch->current_generation;
}
static inline int encounter_route_is_target(
    const EncounterRouteInput* input,
    int x,
    int y
);
static inline void encounter_route_mark_targets(
    const EncounterRouteInput* input,
    EncounterRouteScratch* scratch,
    uint16_t generation
) {
    const EncounterArenaTopology* topology = input->topology;
    if (input->target_kind == ENCOUNTER_ROUTE_TARGET_TILE) {
        int local_x = input->target_x - topology->origin_x;
        int local_y = input->target_y - topology->origin_y;
        scratch->target_generation[
            local_x * topology->height + local_y] = generation;
        return;
    }
    int margin = input->target_kind == ENCOUNTER_ROUTE_TARGET_ATTACK_RANGE
        ? input->attack_range
        : 1;
    int min_x = input->target_x - margin;
    int min_y = input->target_y - margin;
    int max_x = input->target_x + input->target_size - 1 + margin;
    int max_y = input->target_y + input->target_size - 1 + margin;
    if (min_x < topology->origin_x) min_x = topology->origin_x;
    if (min_y < topology->origin_y) min_y = topology->origin_y;
    int topology_max_x = topology->origin_x + topology->width - 1;
    int topology_max_y = topology->origin_y + topology->height - 1;
    if (max_x > topology_max_x) max_x = topology_max_x;
    if (max_y > topology_max_y) max_y = topology_max_y;
    for (int x = min_x; x <= max_x; x++) {
        for (int y = min_y; y <= max_y; y++) {
            if (!encounter_route_is_target(input, x, y)) continue;
            int local_x = x - topology->origin_x;
            int local_y = y - topology->origin_y;
            scratch->target_generation[
                local_x * topology->height + local_y] = generation;
        }
    }
}

static inline int encounter_route_abs(int value) {
    return value < 0 ? -value : value;
}

static inline int encounter_route_dynamic_blocked(
    const EncounterRouteInput* input,
    int x,
    int y
) {
    return input->blockers.is_blocked &&
        input->blockers.is_blocked(
            input->blockers.ctx, x, y, input->actor_size);
}

static inline int encounter_route_step_allowed(
    const EncounterRouteInput* input,
    int x,
    int y,
    int dx,
    int dy
) {
    if (!encounter_arena_topology_step_allowed_assume_finalized_size_in_range(
            input->topology, x, y, input->actor_size, dx, dy))
        return 0;
    if (encounter_route_dynamic_blocked(input, x + dx, y + dy))
        return 0;
    if (dx != 0 && dy != 0 &&
            (encounter_route_dynamic_blocked(input, x + dx, y) ||
             encounter_route_dynamic_blocked(input, x, y + dy)))
        return 0;
    return 1;
}
static inline int encounter_route_dynamic_blocked_cached_at_index(
    const EncounterRouteInput* input,
    EncounterRouteScratch* scratch,
    uint16_t generation,
    int x,
    int y,
    int index
) {
    if (!input->blockers.is_blocked) return 0;
    if (scratch->blocker_generation[index] != generation) {
        scratch->blocker_generation[index] = generation;
        scratch->blocker_value[index] =
            (uint8_t)encounter_route_dynamic_blocked(input, x, y);
    }
    return scratch->blocker_value[index];
}





static inline int encounter_route_footprints_cardinal_adjacent(
    int actor_x,
    int actor_y,
    int actor_size,
    int target_x,
    int target_y,
    int target_size
) {
    int64_t actor_max_x = (int64_t)actor_x + actor_size - 1;
    int64_t actor_max_y = (int64_t)actor_y + actor_size - 1;
    int64_t target_max_x = (int64_t)target_x + target_size - 1;
    int64_t target_max_y = (int64_t)target_y + target_size - 1;
    int x_overlap =
        (int64_t)actor_x <= target_max_x &&
        (int64_t)target_x <= actor_max_x;
    int y_overlap =
        (int64_t)actor_y <= target_max_y &&
        (int64_t)target_y <= actor_max_y;
    return
        (actor_max_x + 1 == target_x && y_overlap) ||
        (target_max_x + 1 == actor_x && y_overlap) ||
        (actor_max_y + 1 == target_y && x_overlap) ||
        (target_max_y + 1 == actor_y && x_overlap);
}

static inline int encounter_route_cardinal_edge_open(
    const EncounterRouteInput* input,
    int actor_x,
    int actor_y
) {
    const EncounterArenaTopology* topology = input->topology;
    int actor_max_x = actor_x + input->actor_size - 1;
    int actor_max_y = actor_y + input->actor_size - 1;
    int target_max_x = input->target_x + input->target_size - 1;
    int target_max_y = input->target_y + input->target_size - 1;
    if (actor_max_x + 1 == input->target_x) {
        int min_y = actor_y > input->target_y ? actor_y : input->target_y;
        int max_y = actor_max_y < target_max_y ? actor_max_y : target_max_y;
        for (int y = min_y; y <= max_y; y++) {
            int index = encounter_arena_topology_index_raw(topology, actor_max_x, y);
            if ((topology->static_collision_flags[index] & COLLISION_WALL_EAST) == 0)
                return 1;
        }
        return 0;
    }
    if (target_max_x + 1 == actor_x) {
        int min_y = actor_y > input->target_y ? actor_y : input->target_y;
        int max_y = actor_max_y < target_max_y ? actor_max_y : target_max_y;
        for (int y = min_y; y <= max_y; y++) {
            int index = encounter_arena_topology_index_raw(topology, actor_x, y);
            if ((topology->static_collision_flags[index] & COLLISION_WALL_WEST) == 0)
                return 1;
        }
        return 0;
    }
    if (actor_max_y + 1 == input->target_y) {
        int min_x = actor_x > input->target_x ? actor_x : input->target_x;
        int max_x = actor_max_x < target_max_x ? actor_max_x : target_max_x;
        for (int x = min_x; x <= max_x; x++) {
            int index = encounter_arena_topology_index_raw(topology, x, actor_max_y);
            if ((topology->static_collision_flags[index] & COLLISION_WALL_NORTH) == 0)
                return 1;
        }
        return 0;
    }
    if (target_max_y + 1 == actor_y) {
        int min_x = actor_x > input->target_x ? actor_x : input->target_x;
        int max_x = actor_max_x < target_max_x ? actor_max_x : target_max_x;
        for (int x = min_x; x <= max_x; x++) {
            int index = encounter_arena_topology_index_raw(topology, x, actor_y);
            if ((topology->static_collision_flags[index] & COLLISION_WALL_SOUTH) == 0)
                return 1;
        }
        return 0;
    }
    return 0;
}


static inline int encounter_route_is_target(
    const EncounterRouteInput* input,
    int x,
    int y
) {
    if (input->target_kind == ENCOUNTER_ROUTE_TARGET_TILE)
        return x == input->target_x && y == input->target_y;
    if (input->target_kind == ENCOUNTER_ROUTE_TARGET_ATTACK_RANGE) {
        if (input->attack_geometry ==
                ENCOUNTER_ROUTE_ATTACK_GEOMETRY_TOPOLOGY) {
            return encounter_arena_topology_player_can_attack_trusted(
                input->topology,
                x,
                y,
                input->target_x,
                input->target_y,
                input->target_size,
                input->attack_range);
        }
        return encounter_attack_position_valid(
            x, y,
            input->target_x, input->target_y, input->target_size,
            input->attack_range,
            input->collision_map,
            input->world_offset_x,
            input->world_offset_y,
            input->los_query);
    }
    return encounter_route_footprints_cardinal_adjacent(
            x, y, input->actor_size,
            input->target_x, input->target_y, input->target_size) &&
        encounter_route_cardinal_edge_open(input, x, y);
}

static inline int encounter_route_target_distance_squared(
    const EncounterRouteInput* input,
    int x,
    int y
) {
    int64_t target_max_x = (int64_t)input->target_x + input->target_size - 1;
    int64_t target_max_y = (int64_t)input->target_y + input->target_size - 1;
    int64_t actor_max_x = (int64_t)x + input->actor_size - 1;
    int64_t actor_max_y = (int64_t)y + input->actor_size - 1;
    int64_t dx = 0;
    int64_t dy = 0;
    if (actor_max_x < input->target_x) dx = input->target_x - actor_max_x;
    else if (target_max_x < x) dx = (int64_t)x - target_max_x;
    if (actor_max_y < input->target_y) dy = input->target_y - actor_max_y;
    else if (target_max_y < y) dy = (int64_t)y - target_max_y;
    int64_t squared = dx * dx + dy * dy;
    return squared > INT_MAX ? INT_MAX : (int)squared;
}

static inline int encounter_route_input_valid(
    const EncounterRouteInput* input
) {
    if (!input || !input->topology || !input->topology->finalized) return 0;
    if (input->actor_size < 1 ||
            input->actor_size > input->topology->max_footprint_size ||
            input->target_size < 1)
        return 0;
    if (input->target_kind < ENCOUNTER_ROUTE_TARGET_TILE ||
            input->target_kind > ENCOUNTER_ROUTE_TARGET_ATTACK_RANGE)
        return 0;
    if (input->target_kind == ENCOUNTER_ROUTE_TARGET_ATTACK_RANGE) {
        if (input->actor_size != 1 || input->attack_range < 1)
            return 0;
        if (input->attack_geometry ==
                ENCOUNTER_ROUTE_ATTACK_GEOMETRY_QUERY &&
                !osrs_los_query_valid(input->los_query, input->attack_range))
            return 0;
        if (input->attack_geometry <
                ENCOUNTER_ROUTE_ATTACK_GEOMETRY_QUERY ||
                input->attack_geometry >
                ENCOUNTER_ROUTE_ATTACK_GEOMETRY_TOPOLOGY)
            return 0;
        if (input->attack_geometry ==
                ENCOUNTER_ROUTE_ATTACK_GEOMETRY_TOPOLOGY) {
            if (input->target_size >
                    input->topology->max_footprint_size)
                return 0;
            if (!encounter_arena_topology_footprint_in_bounds_assume_finalized_size_in_range(
                    input->topology,
                    input->target_x,
                    input->target_y,
                    input->target_size))
                return 0;
        }
    }
    if (input->movement_mode < ENCOUNTER_ROUTE_MOVEMENT_WALK ||
            input->movement_mode > ENCOUNTER_ROUTE_MOVEMENT_RUN)
        return 0;
    if (input->cost_policy < ENCOUNTER_ROUTE_COST_OSRS ||
            input->cost_policy > ENCOUNTER_ROUTE_COST_OSRS_TARGET_BFS)
        return 0;
    if (input->blockers.is_blocked && input->blockers.revision == 0) return 0;
    if (encounter_arena_topology_footprint_blocked(
            input->topology,
            input->source_x,
            input->source_y,
            input->actor_size))
        return 0;
    return 1;
}

static inline void encounter_route_parent(
    const EncounterArenaTopology* topology,
    int via,
    int* x,
    int* y
) {
    if (via == VIA_NONE || via == VIA_START) {
        fprintf(stderr, "broken OSRS route parent at (%d,%d)\n", *x, *y);
        abort();
    }
    if (via & VIA_W) (*x)++;
    else if (via & VIA_E) (*x)--;
    if (via & VIA_S) (*y)++;
    else if (via & VIA_N) (*y)--;
    if (*x < 0 || *x >= topology->width ||
            *y < 0 || *y >= topology->height) {
        fprintf(stderr, "OSRS route parent left topology\n");
        abort();
    }
}

static inline void encounter_route_build_result_path(
    const EncounterRouteInput* input,
    EncounterRouteResult* result,
    int destination_x,
    int destination_y,
    uint16_t generation
) {
    EncounterRouteScratch* scratch = &encounter_route_scratch;
    const EncounterArenaTopology* topology = input->topology;
    int source_x = input->source_x - topology->origin_x;
    int source_y = input->source_y - topology->origin_y;
    int current_x = destination_x;
    int current_y = destination_y;
    int destination_index = current_x * topology->height + current_y;
    uint16_t distance = scratch->depth[destination_index];
    result->distance = distance;
    result->destination_x = topology->origin_x + destination_x;
    result->destination_y = topology->origin_y + destination_y;

    int first_x = source_x;
    int first_y = source_y;
    int second_x = source_x;
    int second_y = source_y;
    int direction = -1;
    while (current_x != source_x || current_y != source_y) {
        int index = current_x * topology->height + current_y;
        if (scratch->generation[index] != generation) {
            fprintf(stderr, "OSRS route read unstamped parent\n");
            abort();
        }
        uint16_t depth = scratch->depth[index];
        if (depth == 1) {
            first_x = current_x;
            first_y = current_y;
        } else if (depth == 2) {
            second_x = current_x;
            second_y = current_y;
        }
        int next_direction = scratch->via[index];
        if (direction != next_direction) {
            direction = next_direction;
            int count = result->waypoint_count;
            if (count == ENCOUNTER_ROUTE_MAX_WAYPOINTS) count--;
            memmove(
                &result->waypoint_x[1],
                &result->waypoint_x[0],
                (size_t)count * sizeof(result->waypoint_x[0]));
            memmove(
                &result->waypoint_y[1],
                &result->waypoint_y[0],
                (size_t)count * sizeof(result->waypoint_y[0]));
            result->waypoint_x[0] = topology->origin_x + current_x;
            result->waypoint_y[0] = topology->origin_y + current_y;
            result->waypoint_count = (uint8_t)(count + 1);
        }

        encounter_route_parent(topology, next_direction, &current_x, &current_y);
    }
    if (distance == 0) return;
    result->first_dx = first_x - source_x;
    result->first_dy = first_y - source_y;
    if (distance >= 2 &&
            input->movement_mode == ENCOUNTER_ROUTE_MOVEMENT_RUN) {
        result->run_dx = second_x - first_x;
        result->run_dy = second_y - first_y;
    }
}
static inline int encounter_route_destination_allowed(
    const EncounterRouteInput* input,
    int x,
    int y
) {
    return
        !encounter_arena_topology_footprint_blocked(
            input->topology, x, y, input->actor_size) &&
        !encounter_route_dynamic_blocked(input, x, y);
}

static inline void encounter_route_direct_parts(
    const EncounterRouteInput* input,
    int destination_x,
    int destination_y,
    int* cardinal_x,
    int* cardinal_y,
    int* cardinal_count,
    int* diagonal_x,
    int* diagonal_y,
    int* diagonal_count
) {
    int dx = destination_x - input->source_x;
    int dy = destination_y - input->source_y;
    int abs_dx = encounter_route_abs(dx);
    int abs_dy = encounter_route_abs(dy);
    *diagonal_x = (dx > 0) - (dx < 0);
    *diagonal_y = (dy > 0) - (dy < 0);
    *cardinal_x = abs_dx > abs_dy ? *diagonal_x : 0;
    *cardinal_y = abs_dy > abs_dx ? *diagonal_y : 0;
    *cardinal_count = encounter_route_abs(abs_dx - abs_dy);
    *diagonal_count = abs_dx < abs_dy ? abs_dx : abs_dy;
}

static inline int encounter_route_direction_rank(
    const EncounterRouteInput* input,
    int dx,
    int dy
) {
    const int8_t* rank_dx = encounter_route_cost_is_osrs(input->cost_policy)
        ? encounter_route_osrs_dx
        : encounter_route_south_dx;
    const int8_t* rank_dy = encounter_route_cost_is_osrs(input->cost_policy)
        ? encounter_route_osrs_dy
        : encounter_route_south_dy;
    for (int rank = 0; rank < 8; rank++) {
        if (rank_dx[rank] == dx && rank_dy[rank] == dy) return rank;
    }
    abort();
}

static inline int encounter_route_direct_candidate_before(
    const EncounterRouteInput* input,
    int candidate_x,
    int candidate_y,
    int selected_x,
    int selected_y
) {
    if (selected_x == INT_MIN) return 1;
    int candidate_cardinal_x, candidate_cardinal_y, candidate_cardinal_count;
    int candidate_diagonal_x, candidate_diagonal_y, candidate_diagonal_count;
    int selected_cardinal_x, selected_cardinal_y, selected_cardinal_count;
    int selected_diagonal_x, selected_diagonal_y, selected_diagonal_count;
    encounter_route_direct_parts(
        input, candidate_x, candidate_y,
        &candidate_cardinal_x, &candidate_cardinal_y,
        &candidate_cardinal_count,
        &candidate_diagonal_x, &candidate_diagonal_y,
        &candidate_diagonal_count);
    encounter_route_direct_parts(
        input, selected_x, selected_y,
        &selected_cardinal_x, &selected_cardinal_y,
        &selected_cardinal_count,
        &selected_diagonal_x, &selected_diagonal_y,
        &selected_diagonal_count);
    int candidate_distance = candidate_cardinal_count + candidate_diagonal_count;
    int selected_distance = selected_cardinal_count + selected_diagonal_count;
    if (candidate_distance != selected_distance)
        return candidate_distance < selected_distance;
    for (int step = 0; step < candidate_distance; step++) {
        int candidate_rank = step < candidate_cardinal_count
            ? encounter_route_direction_rank(
                input, candidate_cardinal_x, candidate_cardinal_y)
            : encounter_route_direction_rank(
                input, candidate_diagonal_x, candidate_diagonal_y);
        int selected_rank = step < selected_cardinal_count
            ? encounter_route_direction_rank(
                input, selected_cardinal_x, selected_cardinal_y)
            : encounter_route_direction_rank(
                input, selected_diagonal_x, selected_diagonal_y);
        if (candidate_rank != selected_rank)
            return candidate_rank < selected_rank;
    }
    return 0;
}

static inline int encounter_route_try_direct_destination(
    const EncounterRouteInput* input,
    int destination_x,
    int destination_y,
    EncounterRouteResult* result
) {
    int cardinal_x, cardinal_y, cardinal_count;
    int diagonal_x, diagonal_y, diagonal_count;
    encounter_route_direct_parts(
        input, destination_x, destination_y,
        &cardinal_x, &cardinal_y, &cardinal_count,
        &diagonal_x, &diagonal_y, &diagonal_count);
    int x = input->source_x;
    int y = input->source_y;
    for (int i = 0; i < cardinal_count; i++) {
        if (!encounter_route_step_allowed(
                input, x, y, cardinal_x, cardinal_y))
            return 0;
        x += cardinal_x;
        y += cardinal_y;
    }
    for (int i = 0; i < diagonal_count; i++) {
        if (!encounter_route_step_allowed(
                input, x, y, diagonal_x, diagonal_y))
            return 0;
        x += diagonal_x;
        y += diagonal_y;
    }
    result->outcome = ROUTE_REACHED_TARGET;
    result->destination_x = destination_x;
    result->destination_y = destination_y;
    result->distance = (uint16_t)(cardinal_count + diagonal_count);
    if (result->distance == 0) return 1;
    result->first_dx = cardinal_count > 0 ? cardinal_x : diagonal_x;
    result->first_dy = cardinal_count > 0 ? cardinal_y : diagonal_y;
    if (result->distance >= 2 &&
            input->movement_mode == ENCOUNTER_ROUTE_MOVEMENT_RUN) {
        result->run_dx = cardinal_count >= 2 ? cardinal_x : diagonal_x;
        result->run_dy = cardinal_count >= 2 ? cardinal_y : diagonal_y;
    }
    if (cardinal_count > 0 && diagonal_count > 0) {
        result->waypoint_x[result->waypoint_count] =
            input->source_x + cardinal_x * cardinal_count;
        result->waypoint_y[result->waypoint_count] =
            input->source_y + cardinal_y * cardinal_count;
        result->waypoint_count++;
    }
    result->waypoint_x[result->waypoint_count] = destination_x;
    result->waypoint_y[result->waypoint_count] = destination_y;
    result->waypoint_count++;
    return 1;
}

static inline void encounter_route_consider_direct_candidate(
    const EncounterRouteInput* input,
    int candidate_x,
    int candidate_y,
    int* selected_x,
    int* selected_y
) {
    if (!encounter_route_destination_allowed(input, candidate_x, candidate_y) ||
            !encounter_route_is_target(input, candidate_x, candidate_y))
        return;
    if (encounter_route_direct_candidate_before(
            input, candidate_x, candidate_y, *selected_x, *selected_y)) {
        *selected_x = candidate_x;
        *selected_y = candidate_y;
    }
}

static inline int encounter_route_try_direct(
    const EncounterRouteInput* input,
    EncounterRouteResult* result
) {
    if (input->target_kind == ENCOUNTER_ROUTE_TARGET_TILE &&
            input->target_size == 1) {
        return encounter_route_try_direct_destination(
            input, input->target_x, input->target_y, result);
    }
    if (input->target_kind == ENCOUNTER_ROUTE_TARGET_ATTACK_RANGE) {
        if (encounter_route_is_target(
                input, input->source_x, input->source_y))
            return encounter_route_try_direct_destination(
                input, input->source_x, input->source_y, result);
        int selected_x = INT_MIN;
        int selected_y = INT_MIN;
        int target_max_x = input->target_x + input->target_size - 1;
        int target_max_y = input->target_y + input->target_size - 1;
        int min_x = input->target_x - input->attack_range;
        int max_x = target_max_x + input->attack_range;
        int min_y = input->target_y - input->attack_range;
        int max_y = target_max_y + input->attack_range;
        int distance_x = input->source_x < min_x
            ? min_x - input->source_x
            : (input->source_x > max_x
                ? input->source_x - max_x
                : 0);
        int distance_y = input->source_y < min_y
            ? min_y - input->source_y
            : (input->source_y > max_y
                ? input->source_y - max_y
                : 0);
        int nearest_distance =
            distance_x > distance_y ? distance_x : distance_y;
        int open_los =
            input->attack_geometry ==
                ENCOUNTER_ROUTE_ATTACK_GEOMETRY_TOPOLOGY
            ? input->topology->static_los_mode ==
                ENCOUNTER_ARENA_TOPOLOGY_LOS_OPEN
            : input->los_query->kind == OSRS_LOS_OPEN;
        if (open_los) {
            for (int x = min_x; x <= max_x; x++) {
                int min_y_distance = encounter_route_abs(x - input->source_x);
                int min_y_delta =
                    encounter_route_abs(min_y - input->source_y);
                if ((min_y_distance > min_y_delta
                        ? min_y_distance
                        : min_y_delta) == nearest_distance)
                    encounter_route_consider_direct_candidate(
                        input, x, min_y, &selected_x, &selected_y);
                int max_y_delta =
                    encounter_route_abs(max_y - input->source_y);
                if ((min_y_distance > max_y_delta
                        ? min_y_distance
                        : max_y_delta) == nearest_distance)
                    encounter_route_consider_direct_candidate(
                        input, x, max_y, &selected_x, &selected_y);
            }
            for (int y = min_y + 1; y < max_y; y++) {
                int y_distance = encounter_route_abs(y - input->source_y);
                int min_x_delta =
                    encounter_route_abs(min_x - input->source_x);
                if ((min_x_delta > y_distance
                        ? min_x_delta
                        : y_distance) == nearest_distance)
                    encounter_route_consider_direct_candidate(
                        input, min_x, y, &selected_x, &selected_y);
                int max_x_delta =
                    encounter_route_abs(max_x - input->source_x);
                if ((max_x_delta > y_distance
                        ? max_x_delta
                        : y_distance) == nearest_distance)
                    encounter_route_consider_direct_candidate(
                        input, max_x, y, &selected_x, &selected_y);
            }
        } else {
            for (int x = min_x; x <= max_x; x++) {
                for (int y = min_y; y <= max_y; y++) {
                    encounter_route_consider_direct_candidate(
                        input, x, y, &selected_x, &selected_y);
                }
            }
        }
        if (selected_x == INT_MIN) return 0;
        return encounter_route_try_direct_destination(
            input, selected_x, selected_y, result);
    }
    if (input->target_kind != ENCOUNTER_ROUTE_TARGET_CARDINAL_ADJACENCY ||
            input->actor_size != 1)
        return 0;
    int selected_x = INT_MIN;
    int selected_y = INT_MIN;
    int target_max_x = input->target_x + input->target_size - 1;
    int target_max_y = input->target_y + input->target_size - 1;
    for (int y = input->target_y; y <= target_max_y; y++) {
        int candidate_x[2] = {input->target_x - 1, target_max_x + 1};
        for (int i = 0; i < 2; i++) {
            if (!encounter_route_destination_allowed(
                    input, candidate_x[i], y) ||
                    !encounter_route_is_target(input, candidate_x[i], y))
                continue;
            if (encounter_route_direct_candidate_before(
                    input, candidate_x[i], y, selected_x, selected_y)) {
                selected_x = candidate_x[i];
                selected_y = y;
            }
        }
    }
    for (int x = input->target_x; x <= target_max_x; x++) {
        int candidate_y[2] = {input->target_y - 1, target_max_y + 1};
        for (int i = 0; i < 2; i++) {
            if (!encounter_route_destination_allowed(
                    input, x, candidate_y[i]) ||
                    !encounter_route_is_target(input, x, candidate_y[i]))
                continue;
            if (encounter_route_direct_candidate_before(
                    input, x, candidate_y[i], selected_x, selected_y)) {
                selected_x = x;
                selected_y = candidate_y[i];
            }
        }
    }

    if (selected_x == INT_MIN) return 0;
    return encounter_route_try_direct_destination(
        input, selected_x, selected_y, result);
}
static inline EncounterRouteResult encounter_route_greedy_direct(
    const EncounterRouteInput* input
) {
    EncounterRouteResult result;
    memset(&result, 0, sizeof(result));
    result.outcome = ROUTE_UNREACHABLE;
    if (input->target_kind != ENCOUNTER_ROUTE_TARGET_TILE)
        return result;
    int x = input->source_x;
    int y = input->source_y;
    int max_steps =
        input->movement_mode == ENCOUNTER_ROUTE_MOVEMENT_RUN ? 2 : 1;
    for (int step = 0; step < max_steps; step++) {
        if (x == input->target_x && y == input->target_y) break;
        int dx = (input->target_x > x) - (input->target_x < x);
        int dy = (input->target_y > y) - (input->target_y < y);
        int moved_dx = 0;
        int moved_dy = 0;
        if (dx != 0 && dy != 0 &&
                encounter_route_step_allowed(input, x, y, dx, dy)) {
            moved_dx = dx;
            moved_dy = dy;
        } else if (dx != 0 &&
                encounter_route_step_allowed(input, x, y, dx, 0)) {
            moved_dx = dx;
        } else if (dy != 0 &&
                encounter_route_step_allowed(input, x, y, 0, dy)) {
            moved_dy = dy;
        } else {
            break;
        }
        x += moved_dx;
        y += moved_dy;
        if (result.distance == 0) {
            result.first_dx = moved_dx;
            result.first_dy = moved_dy;
        } else {
            result.run_dx = moved_dx;
            result.run_dy = moved_dy;
        }
        result.distance++;
    }
    result.destination_x = x;
    result.destination_y = y;
    if (result.distance > 0) {
        result.waypoint_x[0] = x;
        result.waypoint_y[0] = y;
        result.waypoint_count = 1;
    }
    if (x == input->target_x && y == input->target_y)
        result.outcome = ROUTE_REACHED_TARGET;
    else if (result.distance > 0)
        result.outcome = ROUTE_REACHED_FALLBACK;
    return result;
}

static inline EncounterRouteResult encounter_route_solve(
    const EncounterRouteInput* input);

static inline EncounterRouteResult encounter_route_escape_overlap(
    const EncounterRouteInput* input
) {
    EncounterRouteResult result;
    memset(&result, 0, sizeof(result));
    result.outcome = ROUTE_UNREACHABLE;
    int max_radius = (input->target_size + 1) / 2 + 1;
    int best_distance = INT_MAX;
    int candidate_x = -1;
    int candidate_y = -1;
    for (int dy = -max_radius; dy <= max_radius; dy++) {
        for (int dx = -max_radius; dx <= max_radius; dx++) {
            if (dx == 0 && dy == 0) continue;
            int x = input->source_x + dx;
            int y = input->source_y + dy;
            if (!encounter_route_destination_allowed(input, x, y)) continue;
            if (encounter_entity_footprints_overlap(
                    x, y, input->actor_size,
                    input->target_x, input->target_y, input->target_size))
                continue;
            int distance = dx * dx + dy * dy;
            if (distance < best_distance) {
                best_distance = distance;
                candidate_x = x;
                candidate_y = y;
            }
        }
    }
    if (candidate_x < 0) return result;

    EncounterRouteInput escape_input = *input;
    escape_input.target_x = candidate_x;
    escape_input.target_y = candidate_y;
    escape_input.target_size = 1;
    escape_input.target_kind = ENCOUNTER_ROUTE_TARGET_TILE;
    escape_input.movement_mode = ENCOUNTER_ROUTE_MOVEMENT_RUN;
    escape_input.cost_policy = ENCOUNTER_ROUTE_COST_SOUTH_FIRST;
    EncounterRouteResult escape = encounter_route_solve(&escape_input);
    if (escape.outcome == ROUTE_INVALID_INPUT ||
            escape.outcome == ROUTE_UNREACHABLE)
        return escape;
    if (escape.outcome == ROUTE_REACHED_TARGET &&
            escape.distance == 1)
        return escape;
    int landing_x = input->source_x + escape.first_dx + escape.run_dx;
    int landing_y = input->source_y + escape.first_dy + escape.run_dy;

    EncounterRouteInput landing_input = *input;
    landing_input.target_x = landing_x;
    landing_input.target_y = landing_y;
    landing_input.target_size = 1;
    landing_input.target_kind = ENCOUNTER_ROUTE_TARGET_TILE;
    return encounter_route_solve(&landing_input);
}

static inline int encounter_route_reverse_field_contains(
    const EncounterReverseRouteField* field,
    int index
) {
    return (uint16_t)(field->depth_generation[index] >> 16) ==
        field->generation;
}

static inline uint16_t encounter_route_reverse_field_depth(
    const EncounterReverseRouteField* field,
    int index
) {
    return (uint16_t)field->depth_generation[index];
}

static inline void encounter_route_reverse_field_set_depth(
    EncounterReverseRouteField* field,
    int index,
    uint16_t depth
) {
    field->depth_generation[index] =
        ((uint32_t)field->generation << 16) | depth;
}

static inline int encounter_route_reverse_seed(
    const EncounterRouteInput* input,
    EncounterReverseRouteField* field,
    int x,
    int y
) {
    if (!encounter_route_destination_allowed(input, x, y)) return 0;
    const EncounterArenaTopology* topology = input->topology;
    int local_x = x - topology->origin_x;
    int local_y = y - topology->origin_y;
    if (local_x < 0 || local_x >= topology->width ||
            local_y < 0 || local_y >= topology->height)
        return 0;
    int index = local_x * topology->height + local_y;
    if (encounter_route_reverse_field_contains(field, index)) return 0;
    encounter_route_reverse_field_set_depth(field, index, 0);
    field->queue[field->tail++] =
        (uint16_t)((local_x << 6) | local_y);
    return 1;
}

static inline void encounter_route_reverse_seed_cardinal_edges(
    const EncounterRouteInput* input,
    EncounterReverseRouteField* field,
    int require_target
) {
    int target_max_x = input->target_x + input->target_size - 1;
    int target_max_y = input->target_y + input->target_size - 1;
    int x_edges[2] = {
        input->target_x - input->actor_size,
        target_max_x + 1,
    };
    int y_edges[2] = {
        input->target_y - input->actor_size,
        target_max_y + 1,
    };
    for (int edge = 0; edge < 2; edge++) {
        int x = x_edges[edge];
        for (int y = input->target_y - input->actor_size + 1;
                y <= target_max_y;
                y++) {
            if (require_target &&
                    (!encounter_route_destination_allowed(input, x, y) ||
                     !encounter_route_is_target(input, x, y)))
                continue;
            encounter_route_reverse_seed(input, field, x, y);
        }
        int y = y_edges[edge];
        for (int x_scan = input->target_x - input->actor_size + 1;
                x_scan <= target_max_x;
                x_scan++) {
            if (require_target &&
                    (!encounter_route_destination_allowed(input, x_scan, y) ||
                     !encounter_route_is_target(input, x_scan, y)))
                continue;
            encounter_route_reverse_seed(input, field, x_scan, y);
        }
    }
}

static inline void encounter_route_reverse_build_result(
    const EncounterRouteInput* input,
    EncounterReverseRouteField* field,
    EncounterRouteResult* result
) {
    const int8_t* direction_dx =
        encounter_route_cost_is_osrs(input->cost_policy)
            ? encounter_route_osrs_dx
            : encounter_route_south_dx;
    const int8_t* direction_dy =
        encounter_route_cost_is_osrs(input->cost_policy)
            ? encounter_route_osrs_dy
            : encounter_route_south_dy;
    const EncounterArenaTopology* topology = input->topology;
    int current_x = input->source_x;
    int current_y = input->source_y;
    int source_index =
        (current_x - topology->origin_x) * topology->height +
        current_y - topology->origin_y;
    result->outcome = (EncounterRouteOutcome)field->outcome;
    result->distance =
        encounter_route_reverse_field_depth(field, source_index);
    int segment_dx = 0;
    int segment_dy = 0;
    int previous_x = current_x;
    int previous_y = current_y;
    uint16_t trace_distance =
        input->result_detail == ENCOUNTER_ROUTE_RESULT_NEXT_STEPS &&
            field->outcome == ROUTE_REACHED_TARGET &&
            result->distance > 2
        ? 2
        : result->distance;
    for (uint16_t step = 0; step < trace_distance; step++) {
        int current_index =
            (current_x - topology->origin_x) * topology->height +
            current_y - topology->origin_y;
        uint16_t depth =
            encounter_route_reverse_field_depth(field, current_index);
        int next_x = current_x;
        int next_y = current_y;
        int next_dx = 0;
        int next_dy = 0;
        for (int direction = 0; direction < 8; direction++) {
            int candidate_x = current_x + direction_dx[direction];
            int candidate_y = current_y + direction_dy[direction];
            if (!encounter_arena_topology_contains(
                    topology, candidate_x, candidate_y))
                continue;
            int candidate_index =
                (candidate_x - topology->origin_x) * topology->height +
                candidate_y - topology->origin_y;
            if (!encounter_route_reverse_field_contains(
                    field, candidate_index) ||
                    encounter_route_reverse_field_depth(
                        field, candidate_index) + 1 != depth ||
                    !encounter_route_step_allowed(
                        input, current_x, current_y,
                        direction_dx[direction], direction_dy[direction]))
                continue;
            next_x = candidate_x;
            next_y = candidate_y;
            next_dx = direction_dx[direction];
            next_dy = direction_dy[direction];
            break;
        }
        if (next_dx == 0 && next_dy == 0) {
            fprintf(stderr, "broken OSRS reverse route\n");
            abort();
        }
        if (step == 0) {
            result->first_dx = next_dx;
            result->first_dy = next_dy;
        } else if (step == 1 &&
                input->movement_mode == ENCOUNTER_ROUTE_MOVEMENT_RUN) {
            result->run_dx = next_dx;
            result->run_dy = next_dy;
        }
        if (trace_distance == result->distance &&
                step > 0 &&
                (next_dx != segment_dx || next_dy != segment_dy) &&
                result->waypoint_count < ENCOUNTER_ROUTE_MAX_WAYPOINTS) {
            result->waypoint_x[result->waypoint_count] = previous_x;
            result->waypoint_y[result->waypoint_count] = previous_y;
            result->waypoint_count++;
        }
        segment_dx = next_dx;
        segment_dy = next_dy;
        previous_x = next_x;
        previous_y = next_y;
        current_x = next_x;
        current_y = next_y;
    }
    if (trace_distance != result->distance) {
        result->destination_x = input->target_x;
        result->destination_y = input->target_y;
        return;
    }
    result->destination_x = current_x;
    result->destination_y = current_y;
    if (result->distance > 0 &&
            result->waypoint_count < ENCOUNTER_ROUTE_MAX_WAYPOINTS) {
        result->waypoint_x[result->waypoint_count] = current_x;
        result->waypoint_y[result->waypoint_count] = current_y;
        result->waypoint_count++;
    }
}

static inline int encounter_route_reverse_field_matches(
    const EncounterReverseRouteField* field,
    const EncounterRouteInput* input
) {
    return field->valid &&
        field->topology == input->topology &&
        field->blocker_ctx == input->blockers.ctx &&
        field->blocker == input->blockers.is_blocked &&
        field->topology_revision == input->topology->revision &&
        field->blocker_revision == input->blockers.revision &&
        field->target_x == input->target_x &&
        field->target_y == input->target_y &&
        field->target_size == input->target_size &&
        field->actor_size == input->actor_size;
}

static inline void encounter_route_reverse_field_init(
    EncounterReverseRouteField* field,
    const EncounterRouteInput* input
) {
    field->generation++;
    if (field->generation == 0) {
        memset(
            field->depth_generation, 0, sizeof(field->depth_generation));
        field->generation = 1;
    }
    field->topology = input->topology;
    field->blocker_ctx = input->blockers.ctx;
    field->blocker = input->blockers.is_blocked;
    field->topology_revision = input->topology->revision;
    field->blocker_revision = input->blockers.revision;
    field->target_x = input->target_x;
    field->target_y = input->target_y;
    field->head = 0;
    field->tail = 0;
    field->actor_size = (uint8_t)input->actor_size;
    field->target_size = (uint8_t)input->target_size;
    field->outcome = ROUTE_REACHED_TARGET;
    field->valid = 1;
    if (!encounter_route_reverse_seed(
            input, field, input->target_x, input->target_y)) {
        encounter_route_reverse_seed_cardinal_edges(
            input, field, 0);
        field->outcome = ROUTE_REACHED_FALLBACK;
    }
}

static inline int encounter_route_try_reverse(
    const EncounterRouteInput* input,
    EncounterRouteResult* result
) {
    if (input->target_kind != ENCOUNTER_ROUTE_TARGET_TILE ||
            input->cost_policy != ENCOUNTER_ROUTE_COST_SOUTH_FIRST_REVERSE)
        return 0;
    EncounterRouteScratch* scratch = &encounter_route_scratch;
    EncounterReverseRouteField* field = NULL;
    for (int cache_idx = 0;
            cache_idx < ENCOUNTER_REVERSE_ROUTE_CACHE_SLOTS;
            cache_idx++) {
        if (encounter_route_reverse_field_matches(
                &scratch->reverse_fields[cache_idx], input)) {
            field = &scratch->reverse_fields[cache_idx];
            break;
        }
    }
    if (!field) {
        field = &scratch->reverse_fields[scratch->next_reverse_field];
        scratch->next_reverse_field =
            (uint8_t)((scratch->next_reverse_field + 1) %
                ENCOUNTER_REVERSE_ROUTE_CACHE_SLOTS);
        encounter_route_reverse_field_init(field, input);
    }
    if (field->tail == 0) return 0;

    const EncounterArenaTopology* topology = input->topology;
    int source_local_x = input->source_x - topology->origin_x;
    int source_local_y = input->source_y - topology->origin_y;
    int source_index =
        source_local_x * topology->height + source_local_y;
    if (!encounter_route_reverse_field_contains(field, source_index) &&
            field->head < field->tail) {
        uint16_t blocker_generation =
            encounter_route_next_generation(scratch);
        while (field->head < field->tail &&
                !encounter_route_reverse_field_contains(field, source_index)) {
            int current_packed = field->queue[field->head++];
            int current_x = current_packed >> 6;
            int current_y = current_packed & 63;
            int current_index = current_x * topology->height + current_y;
            int current_abs_x = topology->origin_x + current_x;
            int current_abs_y = topology->origin_y + current_y;
            uint16_t next_depth = (uint16_t)(
                encounter_route_reverse_field_depth(field, current_index) + 1);
            const int8_t* dx = encounter_route_osrs_dx;
            const int8_t* dy = encounter_route_osrs_dy;
            const uint8_t* step_mask = encounter_route_osrs_step_mask;
            for (int direction = 0; direction < 8; direction++) {
                int predecessor_x = current_x - dx[direction];
                int predecessor_y = current_y - dy[direction];
                if (predecessor_x < 0 ||
                        predecessor_x >= topology->width ||
                        predecessor_y < 0 ||
                        predecessor_y >= topology->height)
                    continue;
                int predecessor_index =
                    predecessor_x * topology->height + predecessor_y;
                if (encounter_route_reverse_field_contains(
                        field, predecessor_index))
                    continue;
                int predecessor_abs_x =
                    current_abs_x - dx[direction];
                int predecessor_abs_y =
                    current_abs_y - dy[direction];
                if (encounter_route_dynamic_blocked_cached_at_index(
                        input,
                        scratch,
                        blocker_generation,
                        current_abs_x,
                        current_abs_y,
                        current_index))
                    continue;
                if (dx[direction] != 0 && dy[direction] != 0) {
                    int y_side_index =
                        predecessor_x * topology->height + current_y;
                    if (encounter_route_dynamic_blocked_cached_at_index(
                            input,
                            scratch,
                            blocker_generation,
                            predecessor_abs_x,
                            current_abs_y,
                            y_side_index))
                        continue;
                    int x_side_index =
                        current_x * topology->height + predecessor_y;
                    if (encounter_route_dynamic_blocked_cached_at_index(
                            input,
                            scratch,
                            blocker_generation,
                            current_abs_x,
                            predecessor_abs_y,
                            x_side_index))
                        continue;
                }
                if ((topology->legal_step_masks[input->actor_size - 1]
                        [predecessor_index] & step_mask[direction]) == 0)
                    continue;
                if (field->tail >= topology->tile_count) {
                    fprintf(stderr,
                        "OSRS reverse route queue overflow: %u\n",
                        field->tail);
                    abort();
                }
                encounter_route_reverse_field_set_depth(
                    field, predecessor_index, next_depth);
                field->queue[field->tail++] =
                    (uint16_t)((predecessor_x << 6) | predecessor_y);
            }
        }
    }
    if (!encounter_route_reverse_field_contains(field, source_index)) return 0;
    encounter_route_reverse_build_result(input, field, result);
    return 1;
}
#ifdef OSRS_ROUTE_PROBE
static uint64_t osrs_route_probe_source_builds;
static uint64_t osrs_route_probe_source_nodes;
#endif
static inline int encounter_route_source_field_matches(
    const EncounterSourceRouteField* field,
    const EncounterRouteInput* input
) {
    return field->valid &&
        field->topology == input->topology &&
        field->blocker_ctx == input->blockers.ctx &&
        field->blocker == input->blockers.is_blocked &&
        field->topology_revision == input->topology->revision &&
        field->blocker_revision == input->blockers.revision &&
        field->source_x == input->source_x &&
        field->source_y == input->source_y &&
        field->actor_size == input->actor_size;
}
static inline int encounter_route_source_field_set(
    const EncounterRouteInput* input
) {
    uint64_t key =
        input->topology->revision * UINT64_C(0x165667b19e3779f9);
    key ^= (uint64_t)(uint32_t)input->source_x *
        UINT64_C(0x9e3779b185ebca87);
    key ^= (uint64_t)(uint32_t)input->source_y *
        UINT64_C(0xc2b2ae3d27d4eb4f);
    key ^= input->blockers.revision * UINT64_C(0x85ebca77c2b2ae63);
    key ^= (uint64_t)(uint32_t)input->actor_size << 56;
    key ^= key >> 33;
    key *= UINT64_C(0xff51afd7ed558ccd);
    key ^= key >> 33;
    return (int)(key & (ENCOUNTER_SOURCE_ROUTE_CACHE_SETS - 1));
}


static inline void encounter_route_build_source_field(
    const EncounterRouteInput* input,
    EncounterSourceRouteField* field
) {
    const EncounterArenaTopology* topology = input->topology;
#ifdef OSRS_ROUTE_PROBE
    osrs_route_probe_source_builds++;
#endif
    memset(field->visited, 0, sizeof(field->visited));
    memset(field->blocker_known, 0, sizeof(field->blocker_known));
    memset(field->blocker_value, 0, sizeof(field->blocker_value));
    field->valid = 0;
    int source_local_x = input->source_x - topology->origin_x;
    int source_local_y = input->source_y - topology->origin_y;
    int source_index =
        source_local_x * topology->height + source_local_y;
    field->depth[source_index] = 0;
    field->visited[source_local_x] |= UINT64_C(1) << source_local_y;
    field->via[source_index] = VIA_START;
    field->queue[0] =
        (uint16_t)((source_local_x << 6) | source_local_y);
    field->topology = topology;
    field->blocker_ctx = input->blockers.ctx;
    field->blocker = input->blockers.is_blocked;
    field->topology_revision = topology->revision;
    field->blocker_revision = input->blockers.revision;
    field->source_x = input->source_x;
    field->source_y = input->source_y;
    field->visited_count = 1;
    field->expanded_count = 0;
    field->actor_size = (uint8_t)input->actor_size;
    field->valid = 1;
}

static OSRS_ROUTE_NOINLINE int encounter_route_expand_source_field(
    const EncounterRouteInput* input,
    EncounterSourceRouteField* field,
    const uint64_t* target_edges
) {
    const EncounterArenaTopology* topology = input->topology;
    for (int i = 0; i < field->visited_count; i++) {
        int packed = field->queue[i];
        int local_x = packed >> 6;
        int local_y = packed & 63;
        if (target_edges[local_x] & (UINT64_C(1) << local_y))
            return local_x * topology->height + local_y;
    }
    const int8_t* direction_dx = encounter_route_osrs_dx;
    const int8_t* direction_dy = encounter_route_osrs_dy;
    const int8_t* direction_via = encounter_route_osrs_via;
    const uint8_t* direction_step_mask = encounter_route_osrs_step_mask;
    const uint8_t* legal_step_masks =
        topology->legal_step_masks[input->actor_size - 1];
    while (field->expanded_count < field->visited_count) {
        int current_packed = field->queue[field->expanded_count++];
#ifdef OSRS_ROUTE_PROBE
        osrs_route_probe_source_nodes++;
#endif
        int current_x = current_packed >> 6;
        int current_y = current_packed & 63;
        int current_index = current_x * topology->height + current_y;
        int x = topology->origin_x + current_x;
        int y = topology->origin_y + current_y;
        uint16_t next_depth = (uint16_t)(field->depth[current_index] + 1);
        int first_new = field->visited_count;
        for (int direction = 0; direction < 8; direction++) {
            int next_x = current_x + direction_dx[direction];
            int next_y = current_y + direction_dy[direction];
            if (next_x < 0 || next_x >= topology->width ||
                    next_y < 0 || next_y >= topology->height)
                continue;
            uint64_t next_bit = UINT64_C(1) << next_y;
            if (field->visited[next_x] & next_bit) continue;
            int next_index = next_x * topology->height + next_y;
            int dx = direction_dx[direction];
            int dy = direction_dy[direction];
            if (input->blockers.is_blocked) {
                if ((field->blocker_known[next_x] & next_bit) == 0) {
                    field->blocker_known[next_x] |= next_bit;
                    if (input->blockers.is_blocked(
                            input->blockers.ctx,
                            topology->origin_x + next_x,
                            topology->origin_y + next_y,
                            input->actor_size))
                        field->blocker_value[next_x] |= next_bit;
                }
                if (field->blocker_value[next_x] & next_bit) continue;
                if (dx != 0 && dy != 0) {
                    uint64_t side_bit = UINT64_C(1) << next_y;
                    if ((field->blocker_known[current_x] & side_bit) == 0) {
                        field->blocker_known[current_x] |= side_bit;
                        if (input->blockers.is_blocked(
                                input->blockers.ctx,
                                x,
                                topology->origin_y + next_y,
                                input->actor_size))
                            field->blocker_value[current_x] |= side_bit;
                    }
                    if (field->blocker_value[current_x] & side_bit) continue;
                    side_bit = UINT64_C(1) << current_y;
                    if ((field->blocker_known[next_x] & side_bit) == 0) {
                        field->blocker_known[next_x] |= side_bit;
                        if (input->blockers.is_blocked(
                                input->blockers.ctx,
                                topology->origin_x + next_x,
                                y,
                                input->actor_size))
                            field->blocker_value[next_x] |= side_bit;
                    }
                    if (field->blocker_value[next_x] & side_bit) continue;
                }
            }
            if ((legal_step_masks[current_index] &
                    direction_step_mask[direction]) == 0)
                continue;
            if (field->visited_count >= topology->tile_count) {
                fprintf(stderr, "OSRS source field queue overflow: %u\n",
                    field->visited_count);
                abort();
            }
            field->visited[next_x] |= next_bit;
            field->depth[next_index] = next_depth;
            field->via[next_index] = direction_via[direction];
            field->queue[field->visited_count++] =
                (uint16_t)((next_x << 6) | next_y);
        }
        for (int i = first_new; i < field->visited_count; i++) {
            int packed = field->queue[i];
            int local_x = packed >> 6;
            int local_y = packed & 63;
            if (target_edges[local_x] & (UINT64_C(1) << local_y))
                return local_x * topology->height + local_y;
        }
    }
    return -1;
}

static inline void encounter_route_build_source_field_result(
    const EncounterRouteInput* input,
    const EncounterSourceRouteField* field,
    EncounterRouteResult* result,
    int destination_x,
    int destination_y
) {
    const EncounterArenaTopology* topology = input->topology;
    int source_x = input->source_x - topology->origin_x;
    int source_y = input->source_y - topology->origin_y;
    int current_x = destination_x;
    int current_y = destination_y;
    int destination_index = current_x * topology->height + current_y;
    uint16_t distance = field->depth[destination_index];
    result->distance = distance;
    result->destination_x = topology->origin_x + destination_x;
    result->destination_y = topology->origin_y + destination_y;
    int first_x = source_x;
    int first_y = source_y;
    int second_x = source_x;
    int second_y = source_y;
    int direction = -1;
    while (current_x != source_x || current_y != source_y) {
        int index = current_x * topology->height + current_y;
        uint16_t depth = field->depth[index];
        if (depth == 1) {
            first_x = current_x;
            first_y = current_y;
        } else if (depth == 2) {
            second_x = current_x;
            second_y = current_y;
        }
        int next_direction = field->via[index];
        if (direction != next_direction) {
            direction = next_direction;
            int count = result->waypoint_count;
            if (count == ENCOUNTER_ROUTE_MAX_WAYPOINTS) count--;
            memmove(
                &result->waypoint_x[1],
                &result->waypoint_x[0],
                (size_t)count * sizeof(result->waypoint_x[0]));
            memmove(
                &result->waypoint_y[1],
                &result->waypoint_y[0],
                (size_t)count * sizeof(result->waypoint_y[0]));
            result->waypoint_x[0] = topology->origin_x + current_x;
            result->waypoint_y[0] = topology->origin_y + current_y;
            result->waypoint_count = (uint8_t)(count + 1);
        }
        encounter_route_parent(topology, next_direction, &current_x, &current_y);
    }
    if (distance == 0) return;
    result->first_dx = first_x - source_x;
    result->first_dy = first_y - source_y;
    if (distance >= 2 &&
            input->movement_mode == ENCOUNTER_ROUTE_MOVEMENT_RUN) {
        result->run_dx = second_x - first_x;
        result->run_dy = second_y - first_y;
    }
}

static inline int encounter_route_try_source_field(
    const EncounterRouteInput* input,
    EncounterRouteResult* result
) {
    if ((input->target_kind != ENCOUNTER_ROUTE_TARGET_CARDINAL_ADJACENCY &&
         input->target_kind != ENCOUNTER_ROUTE_TARGET_ATTACK_RANGE) ||
            input->cost_policy != ENCOUNTER_ROUTE_COST_OSRS)
        return 0;
    EncounterRouteScratch* scratch = &encounter_route_scratch;
    int cache_set = encounter_route_source_field_set(input);
    int first_cache_idx = cache_set * ENCOUNTER_SOURCE_ROUTE_CACHE_WAYS;
    EncounterSourceRouteField* field = NULL;
    for (int way = 0; way < ENCOUNTER_SOURCE_ROUTE_CACHE_WAYS; way++) {
        EncounterSourceRouteField* candidate =
            &scratch->source_fields[first_cache_idx + way];
        if (encounter_route_source_field_matches(candidate, input)) {
            field = candidate;
            break;
        }
    }
    if (!field) {
        uint8_t way = scratch->next_source_field[cache_set];
        field = &scratch->source_fields[first_cache_idx + way];
        scratch->next_source_field[cache_set] =
            (uint8_t)((way + 1) % ENCOUNTER_SOURCE_ROUTE_CACHE_WAYS);
        encounter_route_build_source_field(input, field);
    }
    const EncounterArenaTopology* topology = input->topology;
    uint64_t target_edges[ENCOUNTER_ARENA_TOPOLOGY_MAX_DIMENSION] = {0};
    int target_max_x = input->target_x + input->target_size - 1;
    int target_max_y = input->target_y + input->target_size - 1;
    int x_edges[2] = {
        input->target_x - input->actor_size,
        target_max_x + 1,
    };
    int y_edges[2] = {
        input->target_y - input->actor_size,
        target_max_y + 1,
    };
    for (int edge = 0; edge < 2; edge++) {
        int x = x_edges[edge];
        for (int y = input->target_y - input->actor_size + 1;
                y <= target_max_y;
                y++) {
            if (!encounter_route_destination_allowed(input, x, y) ||
                    !encounter_route_is_target(input, x, y))
                continue;
            target_edges[x - topology->origin_x] |=
                UINT64_C(1) << (y - topology->origin_y);
        }
        int y = y_edges[edge];
        for (int x_scan = input->target_x - input->actor_size + 1;
                x_scan <= target_max_x;
                x_scan++) {
            if (!encounter_route_destination_allowed(input, x_scan, y) ||
                    !encounter_route_is_target(input, x_scan, y))
                continue;
            target_edges[x_scan - topology->origin_x] |=
                UINT64_C(1) << (y - topology->origin_y);
        }
    }
    int selected_index = encounter_route_expand_source_field(
        input, field, target_edges);
    if (selected_index >= 0) {
        result->outcome = ROUTE_REACHED_TARGET;
        encounter_route_build_source_field_result(
            input, field, result,
            selected_index / topology->height,
            selected_index % topology->height);
        return 1;
    }
    int best_distance = INT_MAX;
    uint16_t best_depth = UINT16_MAX;
    for (int local_x = 0; local_x < topology->width; local_x++) {
        for (int local_y = 0; local_y < topology->height; local_y++) {
            int index = local_x * topology->height + local_y;
            if ((field->visited[local_x] &
                    (UINT64_C(1) << local_y)) == 0)
                continue;
            int x = topology->origin_x + local_x;
            int y = topology->origin_y + local_y;
            int64_t min_target_x =
                (int64_t)input->target_x - PATHFIND_MAX_FALLBACK_RADIUS;
            int64_t max_target_x =
                (int64_t)input->target_x + input->target_size - 1 +
                PATHFIND_MAX_FALLBACK_RADIUS;
            int64_t min_target_y =
                (int64_t)input->target_y - PATHFIND_MAX_FALLBACK_RADIUS;
            int64_t max_target_y =
                (int64_t)input->target_y + input->target_size - 1 +
                PATHFIND_MAX_FALLBACK_RADIUS;
            if ((int64_t)x < min_target_x || (int64_t)x > max_target_x ||
                    (int64_t)y < min_target_y || (int64_t)y > max_target_y)
                continue;
            int target_distance =
                encounter_route_target_distance_squared(input, x, y);
            uint16_t depth = field->depth[index];
            if (target_distance < best_distance ||
                    (target_distance == best_distance && depth < best_depth)) {
                selected_index = index;
                best_distance = target_distance;
                best_depth = depth;
            }
        }
    }
    int source_index =
        (input->source_x - topology->origin_x) * topology->height +
        input->source_y - topology->origin_y;
    if (selected_index < 0 ||
            (selected_index == source_index &&
             encounter_route_dynamic_blocked(
                input, input->source_x, input->source_y))) {
        result->outcome = ROUTE_UNREACHABLE;
        return 1;
    }
    result->outcome = ROUTE_REACHED_FALLBACK;
    encounter_route_build_source_field_result(
        input, field, result,
        selected_index / topology->height,
        selected_index % topology->height);
    return 1;
}


#ifdef OSRS_ROUTE_PROBE
static uint64_t osrs_route_probe_calls[3];
static uint64_t osrs_route_probe_direct;
static uint64_t osrs_route_probe_overlap;
static uint64_t osrs_route_probe_try_direct;
static uint64_t osrs_route_probe_source;
static uint64_t osrs_route_probe_reverse;
static uint64_t osrs_route_probe_bfs;
static uint64_t osrs_route_probe_nodes;
static uint64_t osrs_route_probe_cost_calls[6];
static uint64_t osrs_route_probe_cost_bfs[6];
#endif
static inline void encounter_route_enqueue_unblocked(
    const EncounterArenaTopology* topology,
    EncounterRouteScratch* scratch,
    uint16_t generation,
    int current_index,
    uint16_t next_depth,
    uint8_t legal_mask,
    uint8_t step_mask,
    int direction_delta,
    int8_t direction_via,
    int* tail
) {
    if ((legal_mask & step_mask) == 0) return;
    int next_index = current_index + direction_delta;
    if (scratch->generation[next_index] == generation) return;
    if (*tail >= topology->tile_count) {
        fprintf(stderr, "OSRS route queue overflow: %d\n", *tail);
        abort();
    }
    scratch->generation[next_index] = generation;
    scratch->depth[next_index] = next_depth;
    scratch->via[next_index] = direction_via;
    scratch->queue[(*tail)++] = (uint16_t)next_index;
}
static OSRS_ROUTE_NOINLINE int encounter_route_expand_unblocked_south_first(
    const EncounterArenaTopology* topology,
    EncounterRouteScratch* scratch,
    uint16_t generation,
    int target_index,
    const uint8_t* legal_step_masks
) {
    int head = 0;
    int tail = 1;
    int height = topology->height;
    while (head < tail) {
        int current_index = scratch->queue[head++];
#ifdef OSRS_ROUTE_PROBE
        osrs_route_probe_nodes++;
#endif
        if (target_index >= 0
                ? current_index == target_index
                : scratch->target_generation[current_index] == generation)
            return current_index;
        uint16_t next_depth = (uint16_t)(scratch->depth[current_index] + 1);
        uint8_t legal_mask = legal_step_masks[current_index];
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 2, -1, VIA_S, &tail);
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 8, -height, VIA_W, &tail);
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 64, 1, VIA_N, &tail);
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 16, height, VIA_E, &tail);
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 1, -height - 1, VIA_SW, &tail);
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 32, -height + 1, VIA_NW, &tail);
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 4, height - 1, VIA_SE, &tail);
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 128, height + 1, VIA_NE, &tail);
    }
    return -1;
}


static OSRS_ROUTE_NOINLINE int encounter_route_expand_unblocked_osrs(
    const EncounterArenaTopology* topology,
    EncounterRouteScratch* scratch,
    uint16_t generation,
    int target_index,
    const uint8_t* legal_step_masks
) {
    int head = 0;
    int tail = 1;
    int height = topology->height;
    while (head < tail) {
        int current_index = scratch->queue[head++];
#ifdef OSRS_ROUTE_PROBE
        osrs_route_probe_nodes++;
#endif
        if (target_index >= 0
                ? current_index == target_index
                : scratch->target_generation[current_index] == generation)
            return current_index;
        uint16_t next_depth = (uint16_t)(scratch->depth[current_index] + 1);
        uint8_t legal_mask = legal_step_masks[current_index];
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 8, -height, VIA_W, &tail);
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 16, height, VIA_E, &tail);
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 2, -1, VIA_S, &tail);
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 64, 1, VIA_N, &tail);
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 1, -height - 1, VIA_SW, &tail);
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 4, height - 1, VIA_SE, &tail);
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 32, -height + 1, VIA_NW, &tail);
        encounter_route_enqueue_unblocked(
            topology, scratch, generation, current_index, next_depth,
            legal_mask, 128, height + 1, VIA_NE, &tail);
    }
    return -1;
}


static inline EncounterRouteResult encounter_route_solve(
    const EncounterRouteInput* input
) {
    EncounterRouteResult result;
    memset(&result, 0, sizeof(result));
    result.outcome = ROUTE_INVALID_INPUT;
    if (!encounter_route_input_valid(input)) return result;
#ifdef OSRS_ROUTE_PROBE
    osrs_route_probe_calls[input->target_kind]++;
    osrs_route_probe_cost_calls[input->cost_policy]++;
#endif
    if (input->cost_policy == ENCOUNTER_ROUTE_COST_DIRECT) {
#ifdef OSRS_ROUTE_PROBE
        osrs_route_probe_direct++;
#endif
        return encounter_route_greedy_direct(input);
    }
    if (input->target_kind == ENCOUNTER_ROUTE_TARGET_TILE &&
            !encounter_arena_topology_footprint_in_bounds_assume_finalized_size_in_range(
                input->topology,
                input->target_x,
                input->target_y,
                input->actor_size)) {
        int min_x = input->topology->origin_x;
        int min_y = input->topology->origin_y;
        int max_x =
            min_x + input->topology->width - input->actor_size;
        int max_y =
            min_y + input->topology->height - input->actor_size;
        EncounterRouteInput fallback_input = *input;
        fallback_input.target_x =
            input->target_x < min_x ? min_x :
            input->target_x > max_x ? max_x :
            input->target_x;
        fallback_input.target_y =
            input->target_y < min_y ? min_y :
            input->target_y > max_y ? max_y :
            input->target_y;
        if (encounter_route_destination_allowed(
                &fallback_input,
                fallback_input.target_x,
                fallback_input.target_y)) {
            EncounterRouteResult fallback =
                encounter_route_solve(&fallback_input);
            if (fallback.outcome == ROUTE_REACHED_TARGET) {
                fallback.outcome = ROUTE_REACHED_FALLBACK;
                return fallback;
            }
        }
    }
    if ((input->target_kind == ENCOUNTER_ROUTE_TARGET_CARDINAL_ADJACENCY ||
         input->target_kind == ENCOUNTER_ROUTE_TARGET_ATTACK_RANGE) &&
            encounter_entity_footprints_overlap(
                input->source_x, input->source_y, input->actor_size,
                input->target_x, input->target_y, input->target_size)) {
#ifdef OSRS_ROUTE_PROBE
        osrs_route_probe_overlap++;
#endif
        return encounter_route_escape_overlap(input);
    }
    if (input->cost_policy != ENCOUNTER_ROUTE_COST_SOUTH_FIRST_BFS &&
            !(input->cost_policy == ENCOUNTER_ROUTE_COST_OSRS &&
              input->target_kind == ENCOUNTER_ROUTE_TARGET_ATTACK_RANGE) &&
            encounter_route_try_direct(input, &result)) {
#ifdef OSRS_ROUTE_PROBE
        osrs_route_probe_try_direct++;
#endif
        return result;
    }
    if (encounter_route_try_source_field(input, &result)) {
#ifdef OSRS_ROUTE_PROBE
        osrs_route_probe_source++;
#endif
        return result;
    }
    if (encounter_route_try_reverse(input, &result)) {
#ifdef OSRS_ROUTE_PROBE
        osrs_route_probe_reverse++;
#endif
        return result;
    }
#ifdef OSRS_ROUTE_PROBE
    osrs_route_probe_bfs++;
    osrs_route_probe_cost_bfs[input->cost_policy]++;
#endif
    const EncounterArenaTopology* topology = input->topology;
    EncounterRouteScratch* scratch = &encounter_route_scratch;
    uint16_t generation = encounter_route_next_generation(scratch);
    int target_index = -1;
    if (input->target_kind == ENCOUNTER_ROUTE_TARGET_TILE) {
        target_index =
            (input->target_x - topology->origin_x) * topology->height +
            input->target_y - topology->origin_y;
    } else {
        encounter_route_mark_targets(input, scratch, generation);
    }
    int blocked_tile_fallback =
        input->target_kind == ENCOUNTER_ROUTE_TARGET_TILE &&
        input->target_size == 1 &&
        input->cost_policy == ENCOUNTER_ROUTE_COST_SOUTH_FIRST_BFS &&
        input->blockers.is_blocked &&
        encounter_route_dynamic_blocked(
            input, input->target_x, input->target_y);
    uint16_t blocked_tile_fallback_depth = UINT16_MAX;
    int blocked_tile_fallback_index = -1;
    int selected_blocked_tile_fallback = 0;
    int source_local_x = input->source_x - topology->origin_x;
    int source_local_y = input->source_y - topology->origin_y;
    int source_index = source_local_x * topology->height + source_local_y;
    scratch->generation[source_index] = generation;
    scratch->depth[source_index] = 0;
    scratch->via[source_index] = VIA_START;
    scratch->queue[0] = (uint16_t)source_index;
    if (blocked_tile_fallback) {
        scratch->blocker_generation[target_index] = generation;
        scratch->blocker_value[target_index] = 1;
        int source_target_distance =
            encounter_route_abs(input->source_x - input->target_x) +
            encounter_route_abs(input->source_y - input->target_y);
        if (source_target_distance == 1) {
            blocked_tile_fallback_depth = 0;
            blocked_tile_fallback_index = source_index;
            if (encounter_route_dynamic_blocked(
                    input, input->source_x, input->source_y)) {
                result.outcome = ROUTE_UNREACHABLE;
                return result;
            }
        }
    }
    int head = 0;
    int tail = 1;
    int selected_index = -1;
    int osrs_cost = encounter_route_cost_is_osrs(input->cost_policy);
    const int8_t* direction_dx = osrs_cost
        ? encounter_route_osrs_dx
        : encounter_route_south_dx;
    const int8_t* direction_dy = osrs_cost
        ? encounter_route_osrs_dy
        : encounter_route_south_dy;
    const int8_t* direction_via = osrs_cost
        ? encounter_route_osrs_via
        : encounter_route_south_via;
    const uint8_t* direction_step_mask = osrs_cost
        ? encounter_route_osrs_step_mask
        : encounter_route_south_step_mask;
    const uint8_t* legal_step_masks =
        topology->legal_step_masks[input->actor_size - 1];
    if (!input->blockers.is_blocked) {
        selected_index = osrs_cost
            ? encounter_route_expand_unblocked_osrs(
                topology,
                scratch,
                generation,
                target_index,
                legal_step_masks)
            : encounter_route_expand_unblocked_south_first(
                topology,
                scratch,
                generation,
                target_index,
                legal_step_masks);
    } else {
        scratch->queue[0] =
            (uint16_t)((source_local_x << 6) | source_local_y);
        while (head < tail) {
            int current_packed = scratch->queue[head++];
            int current_x = current_packed >> 6;
            int current_y = current_packed & 63;
            int current_index = current_x * topology->height + current_y;
            if (blocked_tile_fallback_index >= 0 &&
                    scratch->depth[current_index] >=
                        blocked_tile_fallback_depth) {
                selected_index = blocked_tile_fallback_index;
                selected_blocked_tile_fallback = 1;
                break;
            }
#ifdef OSRS_ROUTE_PROBE
            osrs_route_probe_nodes++;
#endif
            if (target_index >= 0
                    ? current_index == target_index
                    : scratch->target_generation[current_index] == generation) {
                selected_index = current_index;
                break;
            }
            uint16_t next_depth =
                (uint16_t)(scratch->depth[current_index] + 1);
            uint8_t legal_mask = legal_step_masks[current_index];
            int x = topology->origin_x + current_x;
            int y = topology->origin_y + current_y;
            for (int i = 0; i < 8; i++) {
                if ((legal_mask & direction_step_mask[i]) == 0) continue;
                int next_x = current_x + direction_dx[i];
                int next_y = current_y + direction_dy[i];
                int next_index = next_x * topology->height + next_y;
                if (scratch->generation[next_index] == generation) continue;
                int dx = direction_dx[i];
                int dy = direction_dy[i];
                if (encounter_route_dynamic_blocked_cached_at_index(
                        input,
                        scratch,
                        generation,
                        topology->origin_x + next_x,
                        topology->origin_y + next_y,
                        next_index))
                    continue;
                if (dx != 0 && dy != 0) {
                    int side_index =
                        current_x * topology->height + next_y;
                    if (encounter_route_dynamic_blocked_cached_at_index(
                            input,
                            scratch,
                            generation,
                            x,
                            topology->origin_y + next_y,
                            side_index))
                        continue;
                    side_index = next_x * topology->height + current_y;
                    if (encounter_route_dynamic_blocked_cached_at_index(
                            input,
                            scratch,
                            generation,
                            topology->origin_x + next_x,
                            y,
                            side_index))
                        continue;
                }
                if (tail >= topology->tile_count) {
                    fprintf(stderr, "OSRS route queue overflow: %d\n", tail);
                    abort();
                }
                scratch->generation[next_index] = generation;
                scratch->depth[next_index] = next_depth;
                scratch->via[next_index] = (int8_t)direction_via[i];
                scratch->queue[tail++] =
                    (uint16_t)((next_x << 6) | next_y);
                if (blocked_tile_fallback) {
                    int target_distance =
                        encounter_route_abs(
                            topology->origin_x + next_x -
                            input->target_x) +
                        encounter_route_abs(
                            topology->origin_y + next_y -
                            input->target_y);
                    if (target_distance == 1 &&
                            (next_depth < blocked_tile_fallback_depth ||
                             (next_depth == blocked_tile_fallback_depth &&
                              next_index < blocked_tile_fallback_index))) {
                        blocked_tile_fallback_depth = next_depth;
                        blocked_tile_fallback_index = next_index;
                    }
                }
            }
        }
    }

    if (selected_index >= 0) {
        result.outcome = selected_blocked_tile_fallback
            ? ROUTE_REACHED_FALLBACK
            : ROUTE_REACHED_TARGET;
        encounter_route_build_result_path(
            input,
            &result,
            selected_index / topology->height,
            selected_index % topology->height,
            generation);
        return result;
    }
    int best_distance = INT_MAX;
    uint16_t best_depth = UINT16_MAX;
    for (int local_x = 0; local_x < topology->width; local_x++) {
        for (int local_y = 0; local_y < topology->height; local_y++) {
            int index = local_x * topology->height + local_y;
            if (scratch->generation[index] != generation) continue;
            int x = topology->origin_x + local_x;
            int y = topology->origin_y + local_y;
            if (encounter_route_cost_is_osrs(input->cost_policy)) {
                int64_t min_target_x =
                    (int64_t)input->target_x - PATHFIND_MAX_FALLBACK_RADIUS;
                int64_t max_target_x =
                    (int64_t)input->target_x + input->target_size - 1 +
                    PATHFIND_MAX_FALLBACK_RADIUS;
                int64_t min_target_y =
                    (int64_t)input->target_y - PATHFIND_MAX_FALLBACK_RADIUS;
                int64_t max_target_y =
                    (int64_t)input->target_y + input->target_size - 1 +
                    PATHFIND_MAX_FALLBACK_RADIUS;
                if ((int64_t)x < min_target_x || (int64_t)x > max_target_x ||
                        (int64_t)y < min_target_y || (int64_t)y > max_target_y)
                    continue;
            }
            int target_distance;
            if (input->cost_policy == ENCOUNTER_ROUTE_COST_SOUTH_FIRST ||
                    input->cost_policy == ENCOUNTER_ROUTE_COST_SOUTH_FIRST_BFS) {
                target_distance =
                    encounter_route_abs(x - input->target_x) +
                    encounter_route_abs(y - input->target_y);
            } else {
                target_distance =
                    encounter_route_target_distance_squared(input, x, y);
            }
            uint16_t depth = scratch->depth[index];
            if (target_distance < best_distance ||
                    (target_distance == best_distance && depth < best_depth)) {
                selected_index = index;
                best_distance = target_distance;
                best_depth = depth;
            }
        }
    }
    if (selected_index < 0 ||
            (selected_index == source_index &&
             encounter_route_dynamic_blocked(
                input, input->source_x, input->source_y))) {
        result.outcome = ROUTE_UNREACHABLE;
        return result;
    }
    result.outcome = ROUTE_REACHED_FALLBACK;
    encounter_route_build_result_path(
        input,
        &result,
        selected_index / topology->height,
        selected_index % topology->height,
        generation);
    return result;
}

#endif
