#ifndef OSRS_ROUTE_REFERENCE_H
#define OSRS_ROUTE_REFERENCE_H

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

typedef struct {
    int found;
    int next_dx;
    int next_dy;
    int dest_x;
    int dest_y;
    int run_dx;
    int run_dy;
} PathResult;

typedef int (*pathfind_blocked_fn)(void* ctx, int abs_x, int abs_y);

static inline void pathfind_enqueue_or_abort(
    int* queue_x, int* queue_y, int* tail, int capacity, int x, int y
) {
    if (*tail >= capacity) {
        fprintf(stderr, "pathfind queue overflow: capacity=%d\n", capacity);
        abort();
    }

    queue_x[*tail] = x;
    queue_y[*tail] = y;
    (*tail)++;
}

static const int pathfind_dir_dx[8] = {0, -1, 0, 1, -1, -1, 1, 1};
static const int pathfind_dir_dy[8] = {-1, 0, 1, 0, -1, 1, -1, 1};
static const int pathfind_dir_via[8] = {
    VIA_S, VIA_W, VIA_N, VIA_E, VIA_SW, VIA_NW, VIA_SE, VIA_NE
};

static inline PathResult pathfind_step(const CollisionMap* map, int height,
                                       int src_x, int src_y, int dest_x, int dest_y,
                                       pathfind_blocked_fn extra_blocked, void* blocked_ctx) {
    PathResult result = {0, 0, 0, dest_x, dest_y};

    if (src_x == dest_x && src_y == dest_y) {
        result.found = 1;
        return result;
    }

    int dist = abs(src_x - dest_x);
    int dy_abs = abs(src_y - dest_y);
    if (dy_abs > dist) dist = dy_abs;
    if (dist > 64) {
        return result;
    }

    int origin_x = ((src_x >> 3) - 6) << 3;
    int origin_y = ((src_y >> 3) - 6) << 3;

    int local_src_x = src_x - origin_x;
    int local_src_y = src_y - origin_y;
    int local_dest_x = dest_x - origin_x;
    int local_dest_y = dest_y - origin_y;

    if (local_dest_x < 0 || local_dest_x >= PATHFIND_GRID_SIZE ||
        local_dest_y < 0 || local_dest_y >= PATHFIND_GRID_SIZE) {
        return result;
    }

    int via[PATHFIND_GRID_SIZE][PATHFIND_GRID_SIZE];
    int cost[PATHFIND_GRID_SIZE][PATHFIND_GRID_SIZE];
    memset(via, 0, sizeof(via));
    memset(cost, 0, sizeof(cost));

    int queue_x[PATHFIND_MAX_QUEUE_FULL];
    int queue_y[PATHFIND_MAX_QUEUE_FULL];
    int head = 0;
    int tail = 0;

    via[local_src_x][local_src_y] = VIA_START;
    cost[local_src_x][local_src_y] = 1;
    pathfind_enqueue_or_abort(
        queue_x, queue_y, &tail, PATHFIND_MAX_QUEUE_FULL, local_src_x, local_src_y);

    int found_path = 0;
    int cur_x, cur_y;

    while (head < tail) {
        cur_x = queue_x[head];
        cur_y = queue_y[head];
        head++;

        if (cur_x == local_dest_x && cur_y == local_dest_y) {
            found_path = 1;
            break;
        }

        int abs_x = origin_x + cur_x;
        int abs_y = origin_y + cur_y;
        int next_cost = cost[cur_x][cur_y] + 1;

        #define EB(ax, ay) (extra_blocked && extra_blocked(blocked_ctx, (ax), (ay)))

        for (int i = 0; i < 8; i++) {
            int dx = pathfind_dir_dx[i];
            int dy = pathfind_dir_dy[i];
            int next_x = cur_x + dx;
            int next_y = cur_y + dy;
            if (next_x < 0 || next_x >= PATHFIND_GRID_SIZE ||
                next_y < 0 || next_y >= PATHFIND_GRID_SIZE)
                continue;
            if (via[next_x][next_y] != 0) continue;
            if (!collision_traversable_step(map, height, abs_x, abs_y, dx, dy))
                continue;
            if (dx != 0 && dy != 0) {
                if (!collision_traversable_step(map, height, abs_x, abs_y, 0, dy))
                    continue;
                if (!collision_traversable_step(map, height, abs_x, abs_y, dx, 0))
                    continue;
            }
            if (EB(abs_x + dx, abs_y + dy)) continue;
            if (dx != 0 && dy != 0) {
                if (EB(abs_x, abs_y + dy)) continue;
                if (EB(abs_x + dx, abs_y)) continue;
            }
            pathfind_enqueue_or_abort(
                queue_x, queue_y, &tail, PATHFIND_MAX_QUEUE_FULL, next_x, next_y);
            via[next_x][next_y] = pathfind_dir_via[i];
            cost[next_x][next_y] = next_cost;
        }

        #undef EB
    }

    if (!found_path) {
        int best_manhattan = PATHFIND_GRID_SIZE * 2;
        int best_cost = 999999;
        int best_x = -1, best_y = -1;

        for (int fx = 0; fx < PATHFIND_GRID_SIZE; fx++) {
            for (int fy = 0; fy < PATHFIND_GRID_SIZE; fy++) {
                if (cost[fx][fy] == 0) continue;

                int ddx = fx - local_dest_x;
                int ddy = fy - local_dest_y;
                int manhattan = (ddx < 0 ? -ddx : ddx) + (ddy < 0 ? -ddy : ddy);

                if (manhattan < best_manhattan ||
                    (manhattan == best_manhattan && cost[fx][fy] < best_cost)) {
                    best_manhattan = manhattan;
                    best_cost = cost[fx][fy];
                    best_x = fx;
                    best_y = fy;
                }
            }
        }

        if (best_x == -1) {
            return result;
        }

        cur_x = best_x;
        cur_y = best_y;
        found_path = 1;
        result.dest_x = origin_x + best_x;
        result.dest_y = origin_y + best_y;
    }

    while (1) {
        int v = via[cur_x][cur_y];
        int prev_x = cur_x;
        int prev_y = cur_y;

        if (v & VIA_W) prev_x++;
        else if (v & VIA_E) prev_x--;

        if (v & VIA_S) prev_y++;
        else if (v & VIA_N) prev_y--;

        if (prev_x == local_src_x && prev_y == local_src_y) {
            result.found = 1;
            result.next_dx = cur_x - local_src_x;
            result.next_dy = cur_y - local_src_y;
            return result;
        }

        cur_x = prev_x;
        cur_y = prev_y;

        if (via[cur_x][cur_y] == VIA_NONE || via[cur_x][cur_y] == VIA_START) {
            break;
        }
    }

    return result;
}

static inline PathResult pathfind_step_arena(
    const CollisionMap* map, int height,
    int src_x, int src_y, int dest_x, int dest_y,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx,
    int arena_origin_x, int arena_origin_y, int arena_w, int arena_h
) {
    PathResult result = {0, 0, 0, dest_x, dest_y};

    if (arena_w <= 0 || arena_w > PATHFIND_ARENA_MAX ||
        arena_h <= 0 || arena_h > PATHFIND_ARENA_MAX) {
        fprintf(stderr, "pathfind arena dimensions out of bounds: %dx%d\n", arena_w, arena_h);
        abort();
    }

    if (src_x == dest_x && src_y == dest_y) {
        result.found = 1;
        return result;
    }

    int local_src_x = src_x - arena_origin_x;
    int local_src_y = src_y - arena_origin_y;
    int local_dest_x = dest_x - arena_origin_x;
    int local_dest_y = dest_y - arena_origin_y;

    if (local_src_x < 0 || local_src_x >= arena_w ||
        local_src_y < 0 || local_src_y >= arena_h ||
        local_dest_x < 0 || local_dest_x >= arena_w ||
        local_dest_y < 0 || local_dest_y >= arena_h) {
        return result;
    }

    static OSRS_THREAD_LOCAL uint16_t bfs_gen[PATHFIND_ARENA_MAX][PATHFIND_ARENA_MAX];
    static OSRS_THREAD_LOCAL int8_t   bfs_via[PATHFIND_ARENA_MAX][PATHFIND_ARENA_MAX];
    static OSRS_THREAD_LOCAL int16_t  bfs_cost[PATHFIND_ARENA_MAX][PATHFIND_ARENA_MAX];
    static OSRS_THREAD_LOCAL uint16_t bfs_gen_counter = 0;
    bfs_gen_counter++;
    if (bfs_gen_counter == 0) {
        memset(bfs_gen, 0, sizeof(bfs_gen));
        bfs_gen_counter = 1;
    }
    uint16_t gen = bfs_gen_counter;
    #define BFS_VISITED(x, y) (bfs_gen[(x)][(y)] == gen)
    #define BFS_VISIT(x, y, v, c) do { \
        bfs_gen[(x)][(y)] = gen; bfs_via[(x)][(y)] = (v); bfs_cost[(x)][(y)] = (c); \
    } while(0)
    #define BFS_VIA(x, y)  bfs_via[(x)][(y)]
    #define BFS_COST(x, y) bfs_cost[(x)][(y)]

    int queue_x[PATHFIND_MAX_QUEUE_ARENA];
    int queue_y[PATHFIND_MAX_QUEUE_ARENA];
    int head = 0, tail = 0;

    BFS_VISIT(local_src_x, local_src_y, VIA_START, 1);
    pathfind_enqueue_or_abort(
        queue_x, queue_y, &tail, PATHFIND_MAX_QUEUE_ARENA, local_src_x, local_src_y);

    int found_path = 0;
    int cur_x, cur_y;

    while (head < tail) {
        cur_x = queue_x[head];
        cur_y = queue_y[head];
        head++;

        if (cur_x == local_dest_x && cur_y == local_dest_y) {
            found_path = 1;
            break;
        }

        int abs_x = arena_origin_x + cur_x;
        int abs_y = arena_origin_y + cur_y;
        int next_cost = BFS_COST(cur_x, cur_y) + 1;

        #define EB(ax, ay) (extra_blocked && extra_blocked(blocked_ctx, (ax), (ay)))

        for (int i = 0; i < 8; i++) {
            int dx = pathfind_dir_dx[i];
            int dy = pathfind_dir_dy[i];
            int next_x = cur_x + dx;
            int next_y = cur_y + dy;
            if (next_x < 0 || next_x >= arena_w ||
                next_y < 0 || next_y >= arena_h)
                continue;
            if (BFS_VISITED(next_x, next_y)) continue;
            if (!collision_traversable_step(map, height, abs_x, abs_y, dx, dy))
                continue;
            if (dx != 0 && dy != 0) {
                if (!collision_traversable_step(map, height, abs_x, abs_y, 0, dy))
                    continue;
                if (!collision_traversable_step(map, height, abs_x, abs_y, dx, 0))
                    continue;
            }
            if (EB(abs_x + dx, abs_y + dy)) continue;
            if (dx != 0 && dy != 0) {
                if (EB(abs_x, abs_y + dy)) continue;
                if (EB(abs_x + dx, abs_y)) continue;
            }
            pathfind_enqueue_or_abort(
                queue_x, queue_y, &tail, PATHFIND_MAX_QUEUE_ARENA, next_x, next_y);
            BFS_VISIT(next_x, next_y, pathfind_dir_via[i], next_cost);
        }

        #undef EB
    }

    if (!found_path) {
        int best_manhattan = arena_w + arena_h;
        int best_cost = 999999;
        int best_x = -1, best_y = -1;

        for (int fx = 0; fx < arena_w; fx++) {
            for (int fy = 0; fy < arena_h; fy++) {
                if (!BFS_VISITED(fx, fy) || BFS_COST(fx, fy) == 0) continue;
                int ddx = fx - local_dest_x, ddy = fy - local_dest_y;
                int manhattan = (ddx < 0 ? -ddx : ddx) + (ddy < 0 ? -ddy : ddy);
                if (manhattan < best_manhattan ||
                    (manhattan == best_manhattan && BFS_COST(fx, fy) < best_cost)) {
                    best_manhattan = manhattan;
                    best_cost = BFS_COST(fx, fy);
                    best_x = fx; best_y = fy;
                }
            }
        }

        if (best_x == -1) return result;
        cur_x = best_x; cur_y = best_y;
        found_path = 1;
        result.dest_x = arena_origin_x + best_x;
        result.dest_y = arena_origin_y + best_y;
    }

    while (1) {
        if (!BFS_VISITED(cur_x, cur_y)) break;
        int v = BFS_VIA(cur_x, cur_y);
        if (v == VIA_NONE || v == VIA_START) break;
        int prev_x = cur_x, prev_y = cur_y;
        if (v & VIA_W) prev_x++; else if (v & VIA_E) prev_x--;
        if (v & VIA_S) prev_y++; else if (v & VIA_N) prev_y--;
        if (prev_x == local_src_x && prev_y == local_src_y) {
            result.found = 1;
            result.next_dx = cur_x - local_src_x;
            result.next_dy = cur_y - local_src_y;
            return result;
        }
        if (prev_x < 0 || prev_x >= arena_w || prev_y < 0 || prev_y >= arena_h) break;
        cur_x = prev_x; cur_y = prev_y;
    }

    return result;
}


static inline PathResult encounter_pathfind(
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    int src_x, int src_y, int dst_x, int dst_y,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx
) {
    return pathfind_step(cmap, 0,
        src_x + world_offset_x, src_y + world_offset_y,
        dst_x + world_offset_x, dst_y + world_offset_y,
        extra_blocked, blocked_ctx);
}

static inline PathResult encounter_pathfind_arena(
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    int src_x, int src_y, int dst_x, int dst_y,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx,
    int arena_base_x, int arena_base_y, int arena_w, int arena_h
) {
    return pathfind_step_arena(cmap, 0,
        src_x + world_offset_x, src_y + world_offset_y,
        dst_x + world_offset_x, dst_y + world_offset_y,
        extra_blocked, blocked_ctx,
        arena_base_x + world_offset_x, arena_base_y + world_offset_y,
        arena_w, arena_h);
}

static inline int encounter_walk_toward(
    Player* p, int tx, int ty,
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    encounter_walkable_fn is_walkable, void* ctx,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx,
    int arena_base_x, int arena_base_y, int arena_w, int arena_h
) {
    int steps = 0;
    for (int step = 0; step < 2; step++) {
        if (p->x == tx && p->y == ty) break;
        PathResult pr = (arena_w > 0)
            ? encounter_pathfind_arena(cmap, world_offset_x, world_offset_y,
                                       p->x, p->y, tx, ty,
                                       extra_blocked, blocked_ctx,
                                       arena_base_x, arena_base_y, arena_w, arena_h)
            : encounter_pathfind(cmap, world_offset_x, world_offset_y,
                                  p->x, p->y, tx, ty,
                                  extra_blocked, blocked_ctx);
        if (!pr.found || (pr.next_dx == 0 && pr.next_dy == 0)) break;
        int nx = p->x + pr.next_dx, ny = p->y + pr.next_dy;
        if (!is_walkable(ctx, nx, ny)) break;
        p->x = nx; p->y = ny;
        steps++;
    }
    p->is_running = (steps == 2);
    p->dest_x = p->x; p->dest_y = p->y;
    return steps;
}

static inline int encounter_move_toward_dest(
    Player* p, int* dest_x, int* dest_y,
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    encounter_walkable_fn is_walkable, void* ctx,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx,
    int arena_base_x, int arena_base_y, int arena_w, int arena_h
) {
    if (*dest_x < 0 || *dest_y < 0) return 0;
    if (p->x == *dest_x && p->y == *dest_y) {
        *dest_x = -1; *dest_y = -1;
        return 0;
    }
    return encounter_walk_toward(p, *dest_x, *dest_y,
        cmap, world_offset_x, world_offset_y,
        is_walkable, ctx, extra_blocked, blocked_ctx,
        arena_base_x, arena_base_y, arena_w, arena_h);
}

static inline int encounter_target_rect_distance_squared(
    int x,
    int y,
    int target_x,
    int target_y,
    int target_size
) {
    int target_max_x = target_x + target_size - 1;
    int target_max_y = target_y + target_size - 1;
    int dx = x < target_x
        ? target_x - x
        : (x > target_max_x ? x - target_max_x : 0);
    int dy = y < target_y
        ? target_y - y
        : (y > target_max_y ? y - target_max_y : 0);
    return dx * dx + dy * dy;
}


typedef struct {
    const CollisionMap* collision_map;
    encounter_walkable_fn is_walkable;
    void* walkable_ctx;
    pathfind_blocked_fn extra_blocked;
    void* blocked_ctx;
    int world_offset_x;
    int world_offset_y;
    int arena_base_x;
    int arena_base_y;
    int arena_w;
    int arena_h;
    int source_x;
    int source_y;
    int target_edge_local_x;
    int target_edge_local_y;
    int min_explored_x;
    int min_explored_y;
    int max_explored_x;
    int max_explored_y;
    uint64_t visited[PATHFIND_ARENA_MAX];
    uint16_t depth[PATHFIND_ARENA_MAX][PATHFIND_ARENA_MAX];
    int8_t via[PATHFIND_ARENA_MAX][PATHFIND_ARENA_MAX];
    int visited_count;
} EncounterArenaAttackRouteField;

typedef struct {
    int land_x;
    int land_y;
    int route_found;
    int route_x;
    int route_y;
} EncounterAttackRouteLanding;


static inline int encounter_attack_route_step_traversable(
    const CollisionMap* cmap,
    int world_offset_x,
    int world_offset_y,
    int x,
    int y,
    int dx,
    int dy,
    encounter_walkable_fn is_walkable,
    void* walkable_ctx,
    pathfind_blocked_fn extra_blocked,
    void* blocked_ctx
) {
    int next_x = x + dx;
    int next_y = y + dy;
    if (!is_walkable(walkable_ctx, next_x, next_y)) return 0;
    if (dx != 0 && dy != 0 &&
            (!is_walkable(walkable_ctx, next_x, y) ||
             !is_walkable(walkable_ctx, x, next_y))) {
        return 0;
    }
    if (!collision_traversable_step(
            cmap, 0, x + world_offset_x, y + world_offset_y, dx, dy)) {
        return 0;
    }
    if (extra_blocked &&
            extra_blocked(
                blocked_ctx,
                next_x + world_offset_x,
                next_y + world_offset_y)) {
        return 0;
    }
    if (dx == 0 || dy == 0) return 1;
    if (extra_blocked &&
            (extra_blocked(
                blocked_ctx,
                next_x + world_offset_x,
                y + world_offset_y) ||
             extra_blocked(
                blocked_ctx,
                x + world_offset_x,
                next_y + world_offset_y))) {
        return 0;
    }
    return 1;
}
typedef struct {
    encounter_walkable_fn is_walkable;
    void* walkable_ctx;
    int arena_base_x;
    int arena_base_y;
    int arena_w;
    int arena_h;
    uint64_t known[PATHFIND_ARENA_MAX];
    uint64_t walkable[PATHFIND_ARENA_MAX];
} EncounterAttackRouteWalkableCache;

static inline int encounter_attack_route_cached_walkable(
    void* data,
    int x,
    int y
) {
    EncounterAttackRouteWalkableCache* cache =
        (EncounterAttackRouteWalkableCache*)data;
    int local_x = x - cache->arena_base_x;
    int local_y = y - cache->arena_base_y;
    if (local_x < 0 || local_x >= cache->arena_w ||
            local_y < 0 || local_y >= cache->arena_h) {
        return cache->is_walkable(cache->walkable_ctx, x, y);
    }
    uint64_t bit = 1ULL << local_y;
    if ((cache->known[local_x] & bit) == 0) {
        cache->known[local_x] |= bit;
        if (cache->is_walkable(cache->walkable_ctx, x, y))
            cache->walkable[local_x] |= bit;
    }
    return (cache->walkable[local_x] & bit) != 0;
}
static inline void encounter_attack_route_mark_target_edge(
    uint64_t target_edges[PATHFIND_ARENA_MAX],
    const CollisionMap* cmap,
    int world_offset_x,
    int world_offset_y,
    int edge_x,
    int edge_y,
    int target_x,
    int target_y,
    int target_size,
    int arena_base_x,
    int arena_base_y,
    int arena_w,
    int arena_h
) {
    int local_x = edge_x - arena_base_x;
    int local_y = edge_y - arena_base_y;
    if (local_x < 0 || local_x >= arena_w ||
            local_y < 0 || local_y >= arena_h) {
        return;
    }
    if (encounter_entity_footprint_cardinal_reachable(
            cmap,
            world_offset_x,
            world_offset_y,
            edge_x,
            edge_y,
            target_x,
            target_y,
            target_size)) {
        target_edges[local_x] |= 1ULL << local_y;
    }
}




static inline void encounter_build_arena_attack_route_field(
    EncounterArenaAttackRouteField* field,
    const CollisionMap* cmap,
    int world_offset_x,
    int world_offset_y,
    int source_x,
    int source_y,
    int target_x,
    int target_y,
    int target_size,
    encounter_walkable_fn is_walkable,
    void* walkable_ctx,
    pathfind_blocked_fn extra_blocked,
    void* blocked_ctx,
    int arena_base_x,
    int arena_base_y,
    int arena_w,
    int arena_h
) {
    if (!field || !is_walkable || target_size <= 0) {
        fprintf(stderr, "attack route field is missing required input\n");
        abort();
    }
    if (arena_w <= 0 || arena_w > PATHFIND_ARENA_MAX ||
            arena_h <= 0 || arena_h > PATHFIND_ARENA_MAX) {
        fprintf(stderr, "attack route arena dimensions out of bounds: %dx%d\n",
            arena_w, arena_h);
        abort();
    }

    int local_source_x = source_x - arena_base_x;
    int local_source_y = source_y - arena_base_y;
    if (local_source_x < 0 || local_source_x >= arena_w ||
            local_source_y < 0 || local_source_y >= arena_h) {
        fprintf(stderr,
            "attack route source out of arena: source=(%d,%d) arena=(%d,%d,%d,%d)\n",
            source_x, source_y, arena_base_x, arena_base_y, arena_w, arena_h);
        abort();
    }

    memset(
        field->visited,
        0,
        (size_t)arena_w * sizeof(field->visited[0]));
    field->target_edge_local_x = -1;
    field->target_edge_local_y = -1;
    field->collision_map = cmap;
    field->is_walkable = is_walkable;
    field->walkable_ctx = walkable_ctx;
    field->extra_blocked = extra_blocked;
    field->blocked_ctx = blocked_ctx;
    field->world_offset_x = world_offset_x;
    field->world_offset_y = world_offset_y;
    field->arena_base_x = arena_base_x;
    field->arena_base_y = arena_base_y;
    field->arena_w = arena_w;
    field->arena_h = arena_h;
    field->source_x = source_x;
    field->source_y = source_y;
    field->min_explored_x = local_source_x;
    field->min_explored_y = local_source_y;
    field->max_explored_x = local_source_x;
    field->max_explored_y = local_source_y;
    EncounterAttackRouteWalkableCache walkable_cache = {
        .is_walkable = is_walkable,
        .walkable_ctx = walkable_ctx,
        .arena_base_x = arena_base_x,
        .arena_base_y = arena_base_y,
        .arena_w = arena_w,
        .arena_h = arena_h,
    };
    uint64_t target_edges[PATHFIND_ARENA_MAX] = {0};
    for (int offset = 0; offset < target_size; offset++) {
        encounter_attack_route_mark_target_edge(
            target_edges, cmap, world_offset_x, world_offset_y,
            target_x - 1, target_y + offset,
            target_x, target_y, target_size,
            arena_base_x, arena_base_y, arena_w, arena_h);
        encounter_attack_route_mark_target_edge(
            target_edges, cmap, world_offset_x, world_offset_y,
            target_x + target_size, target_y + offset,
            target_x, target_y, target_size,
            arena_base_x, arena_base_y, arena_w, arena_h);
        encounter_attack_route_mark_target_edge(
            target_edges, cmap, world_offset_x, world_offset_y,
            target_x + offset, target_y - 1,
            target_x, target_y, target_size,
            arena_base_x, arena_base_y, arena_w, arena_h);
        encounter_attack_route_mark_target_edge(
            target_edges, cmap, world_offset_x, world_offset_y,
            target_x + offset, target_y + target_size,
            target_x, target_y, target_size,
            arena_base_x, arena_base_y, arena_w, arena_h);
    }




    uint16_t queue[PATHFIND_MAX_QUEUE_ARENA];
    int head = 0;
    field->visited_count = 1;
    field->visited[local_source_x] = 1ULL << local_source_y;
    field->depth[local_source_x][local_source_y] = 0;
    field->via[local_source_x][local_source_y] = VIA_START;
    queue[0] = (uint16_t)((local_source_x << 8) | local_source_y);

    static const int route_dx[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
    static const int route_dy[8] = {0, 0, -1, 1, -1, -1, 1, 1};
    static const int route_via[8] = {
        VIA_W, VIA_E, VIA_S, VIA_N, VIA_SW, VIA_SE, VIA_NW, VIA_NE
    };

    while (head < field->visited_count) {
        uint16_t packed_tile = queue[head++];
        int cur_x = packed_tile >> 8;
        int cur_y = packed_tile & 0xff;

        int tile_x = arena_base_x + cur_x;
        int tile_y = arena_base_y + cur_y;
        if (target_edges[cur_x] & (1ULL << cur_y)) {
            field->target_edge_local_x = cur_x;
            field->target_edge_local_y = cur_y;
            break;
        }
        uint16_t next_depth = field->depth[cur_x][cur_y] + 1;

        for (int i = 0; i < 8; i++) {
            int dx = route_dx[i];
            int dy = route_dy[i];
            int next_x = cur_x + dx;
            int next_y = cur_y + dy;
            if (next_x < 0 || next_x >= arena_w ||
                    next_y < 0 || next_y >= arena_h) {
                continue;
            }
            if (field->visited[next_x] & (1ULL << next_y)) continue;
            if (!encounter_attack_route_step_traversable(
                    cmap,
                    world_offset_x,
                    world_offset_y,
                    tile_x,
                    tile_y,
                    dx,
                    dy,
                    encounter_attack_route_cached_walkable,
                    &walkable_cache,
                    extra_blocked,
                    blocked_ctx)) {
                continue;
            }

            queue[field->visited_count] =
                (uint16_t)((next_x << 8) | next_y);
            field->visited_count++;
            field->visited[next_x] |= 1ULL << next_y;
            field->depth[next_x][next_y] = next_depth;
            field->via[next_x][next_y] = (int8_t)route_via[i];
            if (next_x < field->min_explored_x) field->min_explored_x = next_x;
            if (next_y < field->min_explored_y) field->min_explored_y = next_y;
            if (next_x > field->max_explored_x) field->max_explored_x = next_x;
            if (next_y > field->max_explored_y) field->max_explored_y = next_y;
        }
    }
}

static inline EncounterAttackRouteLanding
encounter_attack_route_overlap_landing(
    const EncounterArenaAttackRouteField* field,
    int target_x,
    int target_y,
    int target_size
) {
    Player player = {
        .x = field->source_x,
        .y = field->source_y,
    };
    int max_r = (target_size + 1) / 2 + 1;
    int best_dsq = 9999;
    int best_x = -1;
    int best_y = -1;
    for (int dy = -max_r; dy <= max_r; dy++) {
        for (int dx = -max_r; dx <= max_r; dx++) {
            if (dx == 0 && dy == 0) continue;
            int x = player.x + dx;
            int y = player.y + dy;
            if (!field->is_walkable(field->walkable_ctx, x, y)) continue;
            if (encounter_entity_footprints_overlap(
                    x, y, 1, target_x, target_y, target_size)) {
                continue;
            }
            int dsq = dx * dx + dy * dy;
            if (dsq < best_dsq) {
                best_dsq = dsq;
                best_x = x;
                best_y = y;
            }
        }
    }
    if (best_x >= 0) {
        encounter_walk_toward(
            &player, best_x, best_y,
            field->collision_map,
            field->world_offset_x,
            field->world_offset_y,
            field->is_walkable,
            field->walkable_ctx,
            field->extra_blocked,
            field->blocked_ctx,
            field->arena_base_x,
            field->arena_base_y,
            field->arena_w,
            field->arena_h);
    }
    return (EncounterAttackRouteLanding){
        .land_x = player.x,
        .land_y = player.y,
        .route_found = player.x != field->source_x ||
            player.y != field->source_y,
        .route_x = player.x,
        .route_y = player.y,
    };
}

static inline EncounterAttackRouteLanding encounter_arena_attack_route_landing(
    const EncounterArenaAttackRouteField* field,
    int target_x,
    int target_y,
    int target_size,
    int attack_range,
    const OsrsLosQuery* los_query
) {
    if (!field || !field->is_walkable ||
            field->arena_w <= 0 || field->arena_w > PATHFIND_ARENA_MAX ||
            field->arena_h <= 0 || field->arena_h > PATHFIND_ARENA_MAX ||
            field->visited_count <= 0 ||
            target_size <= 0 || attack_range <= 0) {
        fprintf(stderr, "invalid attack route landing query\n");
        abort();
    }

    EncounterAttackRouteLanding landing = {
        .land_x = field->source_x,
        .land_y = field->source_y,
        .route_x = field->source_x,
        .route_y = field->source_y,
    };
    int dist = encounter_rect_distance(
        field->source_x, field->source_y, 1,
        target_x, target_y, target_size);
    if (dist == 0) {
        return encounter_attack_route_overlap_landing(
            field, target_x, target_y, target_size);
    }
    if (encounter_player_can_attack(
            field->source_x, field->source_y,
            target_x, target_y, target_size, attack_range,
            field->collision_map, field->world_offset_x, field->world_offset_y,
            los_query)) {
        landing.route_found = 1;
        return landing;
    }


    int selected_x = field->target_edge_local_x;
    int selected_y = field->target_edge_local_y;

    if (selected_x < 0) {
        int target_local_x = target_x - field->arena_base_x;
        int target_local_y = target_y - field->arena_base_y;
        int scan_min_x = target_local_x - PATHFIND_MAX_FALLBACK_RADIUS;
        int scan_min_y = target_local_y - PATHFIND_MAX_FALLBACK_RADIUS;
        int scan_max_x =
            target_local_x + target_size - 1 + PATHFIND_MAX_FALLBACK_RADIUS;
        int scan_max_y =
            target_local_y + target_size - 1 + PATHFIND_MAX_FALLBACK_RADIUS;
        if (scan_min_x < field->min_explored_x)
            scan_min_x = field->min_explored_x;
        if (scan_min_y < field->min_explored_y)
            scan_min_y = field->min_explored_y;
        if (scan_max_x > field->max_explored_x)
            scan_max_x = field->max_explored_x;
        if (scan_max_y > field->max_explored_y)
            scan_max_y = field->max_explored_y;

        int best_dsq = 0x3fffffff;
        int best_depth = 100;
        for (int x = scan_min_x; x <= scan_max_x; x++) {
            for (int y = scan_min_y; y <= scan_max_y; y++) {
                if ((field->visited[x] & (1ULL << y)) == 0) continue;
                int tile_x = field->arena_base_x + x;
                int tile_y = field->arena_base_y + y;
                int depth = field->depth[x][y];
                if (depth >= 100) continue;
                int dsq = encounter_target_rect_distance_squared(
                    tile_x, tile_y, target_x, target_y, target_size);
                if (dsq < best_dsq ||
                        (dsq == best_dsq && depth < best_depth)) {
                    selected_x = x;
                    selected_y = y;
                    best_dsq = dsq;
                    best_depth = depth;
                }
            }
        }
    }
    if (selected_x < 0) return landing;
    landing.route_found = 1;
    landing.route_x = field->arena_base_x + selected_x;
    landing.route_y = field->arena_base_y + selected_y;

    int source_local_x = field->source_x - field->arena_base_x;
    int source_local_y = field->source_y - field->arena_base_y;
    int first_x = source_local_x;
    int first_y = source_local_y;
    int second_x = source_local_x;
    int second_y = source_local_y;
    int cur_x = selected_x;
    int cur_y = selected_y;
    uint16_t selected_depth = field->depth[selected_x][selected_y];
    while (field->depth[cur_x][cur_y] > 0) {
        uint16_t depth = field->depth[cur_x][cur_y];
        if (depth == 1) {
            first_x = cur_x;
            first_y = cur_y;
        } else if (depth == 2) {
            second_x = cur_x;
            second_y = cur_y;
        }

        int via = field->via[cur_x][cur_y];
        if (via == VIA_NONE || via == VIA_START) {
            fprintf(stderr,
                "broken attack route parent at local tile (%d,%d)\n",
                cur_x, cur_y);
            abort();
        }
        if (via & VIA_W) cur_x++;
        else if (via & VIA_E) cur_x--;
        if (via & VIA_S) cur_y++;
        else if (via & VIA_N) cur_y--;
        if (cur_x < 0 || cur_x >= field->arena_w ||
                cur_y < 0 || cur_y >= field->arena_h) {
            fprintf(stderr, "attack route parent left arena\n");
            abort();
        }
    }
    if (cur_x != source_local_x || cur_y != source_local_y) {
        fprintf(stderr, "attack route did not terminate at source\n");
        abort();
    }
    if (selected_depth == 0) return landing;

    landing.land_x = field->arena_base_x + first_x;
    landing.land_y = field->arena_base_y + first_y;
    if (encounter_player_can_attack(
            landing.land_x, landing.land_y,
            target_x, target_y, target_size, attack_range,
            field->collision_map, field->world_offset_x, field->world_offset_y,
            los_query)) {
        return landing;
    }
    if (selected_depth >= 2) {
        landing.land_x = field->arena_base_x + second_x;
        landing.land_y = field->arena_base_y + second_y;
    }
    return landing;
}

#define ENCOUNTER_ATTACK_ROUTE_MAX_WAYPOINTS 25

static inline int encounter_attack_route_compress_waypoints(
    const EncounterArenaAttackRouteField* field,
    const EncounterAttackRouteLanding* landing,
    int waypoint_x[ENCOUNTER_ATTACK_ROUTE_MAX_WAYPOINTS],
    int waypoint_y[ENCOUNTER_ATTACK_ROUTE_MAX_WAYPOINTS]
) {
    if (!landing->route_found ||
            (landing->route_x == field->source_x &&
             landing->route_y == field->source_y)) {
        return 0;
    }

    int source_x = field->source_x - field->arena_base_x;
    int source_y = field->source_y - field->arena_base_y;
    int current_x = landing->route_x - field->arena_base_x;
    int current_y = landing->route_y - field->arena_base_y;
    int current_direction = -1;
    int waypoint_count = 0;

    while (current_x != source_x || current_y != source_y) {
        int next_direction = field->via[current_x][current_y];
        if (next_direction == VIA_NONE || next_direction == VIA_START) {
            fprintf(stderr, "broken attack route during compression\n");
            abort();
        }
        if (current_direction != next_direction) {
            current_direction = next_direction;
            if (waypoint_count == ENCOUNTER_ATTACK_ROUTE_MAX_WAYPOINTS)
                waypoint_count--;
            memmove(
                &waypoint_x[1],
                &waypoint_x[0],
                (size_t)waypoint_count * sizeof(waypoint_x[0]));
            memmove(
                &waypoint_y[1],
                &waypoint_y[0],
                (size_t)waypoint_count * sizeof(waypoint_y[0]));
            waypoint_x[0] = field->arena_base_x + current_x;
            waypoint_y[0] = field->arena_base_y + current_y;
            waypoint_count++;
        }

        if (current_direction & VIA_W) current_x++;
        else if (current_direction & VIA_E) current_x--;
        if (current_direction & VIA_S) current_y++;
        else if (current_direction & VIA_N) current_y--;
    }
    return waypoint_count;
}

static inline PathResult encounter_pathfind_arena_attack_approach(
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    int src_x, int src_y,
    int target_x, int target_y, int target_size, int attack_range,
    encounter_walkable_fn is_walkable, void* ctx,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx,
    const OsrsLosQuery* los_query,
    int arena_base_x, int arena_base_y, int arena_w, int arena_h
) {
    PathResult result = {0, 0, 0, src_x, src_y};
    int local_src_x = src_x - arena_base_x;
    int local_src_y = src_y - arena_base_y;
    if (local_src_x < 0 || local_src_x >= arena_w ||
            local_src_y < 0 || local_src_y >= arena_h) {
        return result;
    }

    EncounterArenaAttackRouteField field;
    encounter_build_arena_attack_route_field(
        &field,
        cmap,
        world_offset_x,
        world_offset_y,
        src_x,
        src_y,
        target_x,
        target_y,
        target_size,
        is_walkable,
        ctx,
        extra_blocked,
        blocked_ctx,
        arena_base_x,
        arena_base_y,
        arena_w,
        arena_h);
    EncounterAttackRouteLanding landing =
        encounter_arena_attack_route_landing(
            &field,
            target_x,
            target_y,
            target_size,
            attack_range,
            los_query);

    int landing_x = landing.land_x - arena_base_x;
    int landing_y = landing.land_y - arena_base_y;
    if (landing_x < 0 || landing_x >= arena_w ||
            landing_y < 0 || landing_y >= arena_h) {
        return result;
    }
    uint16_t landing_depth = field.depth[landing_x][landing_y];
    if (landing_depth == 0) {
        result.found = encounter_player_can_attack(
            src_x, src_y,
            target_x, target_y, target_size, attack_range,
            cmap, world_offset_x, world_offset_y,
            los_query);
        return result;
    }
    if (landing_depth > 2) {
        fprintf(stderr, "attack route landing exceeded one run tick: %u\n",
            (unsigned)landing_depth);
        abort();
    }

    result.found = 1;
    result.dest_x = landing.land_x;
    result.dest_y = landing.land_y;
    if (landing_depth == 1) {
        result.next_dx = landing.land_x - src_x;
        result.next_dy = landing.land_y - src_y;
        return result;
    }

    int first_x = landing_x;
    int first_y = landing_y;
    int via = field.via[landing_x][landing_y];
    if (via & VIA_W) first_x++;
    else if (via & VIA_E) first_x--;
    if (via & VIA_S) first_y++;
    else if (via & VIA_N) first_y--;
    result.next_dx = arena_base_x + first_x - src_x;
    result.next_dy = arena_base_y + first_y - src_y;
    result.run_dx = landing_x - first_x;
    result.run_dy = landing_y - first_y;
    return result;
}

static inline int encounter_chase_attack_target(
    Player* p, int target_x, int target_y, int target_size, int attack_range,
    const CollisionMap* cmap, int world_offset_x, int world_offset_y,
    encounter_walkable_fn is_walkable, void* ctx,
    pathfind_blocked_fn extra_blocked, void* blocked_ctx,
    const OsrsLosQuery* los_query,
    int arena_base_x, int arena_base_y, int arena_w, int arena_h
) {
    int dist = encounter_rect_distance(p->x, p->y, 1,
                                                   target_x, target_y, target_size);

    if (dist == 0) {
        int max_r = (target_size + 1) / 2 + 1;
        int best_dsq = 9999, bx = -1, by = -1;
        for (int dy = -max_r; dy <= max_r; dy++) {
            for (int dx = -max_r; dx <= max_r; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = p->x + dx, ny = p->y + dy;
                if (!is_walkable(ctx, nx, ny)) continue;
                if (encounter_entity_footprints_overlap(nx, ny, 1,
                                                        target_x, target_y, target_size))
                    continue;
                int d = dx * dx + dy * dy;
                if (d < best_dsq) { best_dsq = d; bx = nx; by = ny; }
            }
        }
        if (bx < 0) return 0;
        int steps = encounter_walk_toward(p, bx, by,
            cmap, world_offset_x, world_offset_y,
            is_walkable, ctx, extra_blocked, blocked_ctx,
            arena_base_x, arena_base_y, arena_w, arena_h);
        return steps > 0 ? 1 : 0;
    }
    if (encounter_player_can_attack(
            p->x, p->y,
            target_x, target_y, target_size, attack_range,
            cmap, world_offset_x, world_offset_y,
            los_query))
        return 0;


    int cx, cy;
    cx = -1;
    cy = -1;

    if (arena_w <= 0) {
        int scan_min_x = target_x - attack_range;
        int scan_max_x = target_x + target_size - 1 + attack_range;
        int scan_min_y = target_y - attack_range;
        int scan_max_y = target_y + target_size - 1 + attack_range;

        cx = -1;
        cy = -1;
        int best_player_dsq = 0x3fffffff;
        int best_target_dist = 0x3fffffff;
        if (scan_min_x <= scan_max_x && scan_min_y <= scan_max_y) {
            for (int yy = scan_min_y; yy <= scan_max_y; yy++) {
                for (int xx = scan_min_x; xx <= scan_max_x; xx++) {
                    if (!is_walkable(ctx, xx, yy)) continue;
                    if (!encounter_player_can_attack(xx, yy, target_x, target_y,
                            target_size, attack_range,
                            cmap, world_offset_x, world_offset_y,
                            los_query))
                        continue;
                    int dx = xx - p->x;
                    int dy = yy - p->y;
                    int player_dsq = dx * dx + dy * dy;
                    int target_dist = encounter_rect_distance(
                        xx, yy, 1, target_x, target_y, target_size);
                    if (player_dsq < best_player_dsq ||
                            (player_dsq == best_player_dsq &&
                             target_dist < best_target_dist)) {
                        best_player_dsq = player_dsq;
                        best_target_dist = target_dist;
                        cx = xx;
                        cy = yy;
                    }
                }
            }
        }

        if (cx < 0) {
            cx = p->x < target_x ? target_x :
                 (p->x > target_x + target_size - 1 ? target_x + target_size - 1 : p->x);
            cy = p->y < target_y ? target_y :
                 (p->y > target_y + target_size - 1 ? target_y + target_size - 1 : p->y);
        }
    }

    int cached_run_dx = 0;
    int cached_run_dy = 0;
    int steps = 0;
    for (int step = 0; step < 2; step++) {
        if (encounter_player_can_attack(p->x, p->y, target_x, target_y,
                                         target_size, attack_range,
                                         cmap, world_offset_x, world_offset_y,
                                         los_query))
            break;
        int next_dx;
        int next_dy;
        int use_cached_run = step == 1 && arena_w > 0 &&
            (cached_run_dx != 0 || cached_run_dy != 0);
        if (use_cached_run) {
            int next_x = p->x + cached_run_dx;
            int next_y = p->y + cached_run_dy;
            if (!is_walkable(ctx, next_x, next_y) ||
                    (extra_blocked && extra_blocked(
                        blocked_ctx,
                        next_x + world_offset_x,
                        next_y + world_offset_y))) {
                use_cached_run = 0;
            } else if (cached_run_dx != 0 && cached_run_dy != 0 &&
                    (!is_walkable(ctx, next_x, p->y) ||
                     !is_walkable(ctx, p->x, next_y) ||
                     (extra_blocked &&
                        (extra_blocked(
                            blocked_ctx,
                            next_x + world_offset_x,
                            p->y + world_offset_y) ||
                         extra_blocked(
                            blocked_ctx,
                            p->x + world_offset_x,
                            next_y + world_offset_y))))) {
                use_cached_run = 0;
            }
        }
        if (use_cached_run) {
            next_dx = cached_run_dx;
            next_dy = cached_run_dy;
        } else {
            PathResult pr = (arena_w > 0)
                ? encounter_pathfind_arena_attack_approach(
                    cmap, world_offset_x, world_offset_y,
                    p->x, p->y,
                    target_x, target_y, target_size, attack_range,
                    is_walkable, ctx,
                    extra_blocked, blocked_ctx,
                    los_query,
                    arena_base_x, arena_base_y, arena_w, arena_h)
                : encounter_pathfind(cmap, world_offset_x, world_offset_y,
                    p->x, p->y, cx, cy,
                    extra_blocked, blocked_ctx);
            if (!pr.found || (pr.next_dx == 0 && pr.next_dy == 0)) break;
            next_dx = pr.next_dx;
            next_dy = pr.next_dy;
            if (step == 0) {
                cached_run_dx = pr.run_dx;
                cached_run_dy = pr.run_dy;
            }
        }
        int nx = p->x + next_dx, ny = p->y + next_dy;
        if (!is_walkable(ctx, nx, ny)) break;
        p->x = nx; p->y = ny;
        steps++;
    }
    p->is_running = (steps == 2);
    p->dest_x = p->x; p->dest_y = p->y;
    return steps > 0 ? 1 : 0;
}


typedef struct {
    uint64_t visited[ENCOUNTER_ARENA_TOPOLOGY_MAX_DIMENSION];
    uint16_t depth[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint16_t queue[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    int8_t via[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint16_t count;
} OsrsReferenceRouteField;

typedef int (*osrs_reference_can_attack_fn)(
    void* ctx,
    int player_x,
    int player_y,
    int target_x,
    int target_y,
    int target_size,
    int attack_range);

static inline int osrs_reference_route_blocked(
    const EncounterRouteInput* input,
    int x,
    int y
) {
    return input->blockers.is_blocked &&
        input->blockers.is_blocked(
            input->blockers.ctx, x, y, input->actor_size);
}

static inline int osrs_reference_route_step_allowed(
    const EncounterRouteInput* input,
    int local_x,
    int local_y,
    int direction
) {
    static const int8_t dx[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
    static const int8_t dy[8] = {0, 0, -1, 1, -1, -1, 1, 1};
    static const uint8_t step_mask[8] = {8, 16, 2, 64, 1, 4, 32, 128};
    const EncounterArenaTopology* topology = input->topology;
    int next_x = local_x + dx[direction];
    int next_y = local_y + dy[direction];
    if (next_x < 0 || next_x >= topology->width ||
            next_y < 0 || next_y >= topology->height)
        return 0;
    int x = topology->origin_x + local_x;
    int y = topology->origin_y + local_y;
    if (osrs_reference_route_blocked(
            input, x + dx[direction], y + dy[direction]))
        return 0;
    if (dx[direction] != 0 && dy[direction] != 0 &&
            (osrs_reference_route_blocked(
                input, x + dx[direction], y) ||
             osrs_reference_route_blocked(
                input, x, y + dy[direction])))
        return 0;
    int index = local_x * topology->height + local_y;
    return (topology->legal_step_masks[input->actor_size - 1][index] &
        step_mask[direction]) != 0;
}

static inline void osrs_reference_route_build(
    const EncounterRouteInput* input,
    OsrsReferenceRouteField* field
) {
    static const int8_t dx[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
    static const int8_t dy[8] = {0, 0, -1, 1, -1, -1, 1, 1};
    static const int8_t via[8] = {
        VIA_W, VIA_E, VIA_S, VIA_N, VIA_SW, VIA_SE, VIA_NW, VIA_NE
    };
    const EncounterArenaTopology* topology = input->topology;
    memset(field, 0, sizeof(*field));
    int source_x = input->source_x - topology->origin_x;
    int source_y = input->source_y - topology->origin_y;
    int source_index = source_x * topology->height + source_y;
    field->visited[source_x] = UINT64_C(1) << source_y;
    field->depth[source_index] = 0;
    field->via[source_index] = VIA_START;
    field->queue[0] = (uint16_t)((source_x << 6) | source_y);
    field->count = 1;
    for (int head = 0; head < field->count; head++) {
        int packed = field->queue[head];
        int x = packed >> 6;
        int y = packed & 63;
        int index = x * topology->height + y;
        uint16_t next_depth = (uint16_t)(field->depth[index] + 1);
        for (int direction = 0; direction < 8; direction++) {
            int next_x = x + dx[direction];
            int next_y = y + dy[direction];
            if (next_x < 0 || next_x >= topology->width ||
                    next_y < 0 || next_y >= topology->height)
                continue;
            uint64_t bit = UINT64_C(1) << next_y;
            if (field->visited[next_x] & bit) continue;
            if (!osrs_reference_route_step_allowed(
                    input, x, y, direction))
                continue;
            if (field->count >= topology->tile_count) abort();
            int next_index = next_x * topology->height + next_y;
            field->visited[next_x] |= bit;
            field->depth[next_index] = next_depth;
            field->via[next_index] = via[direction];
            field->queue[field->count++] =
                (uint16_t)((next_x << 6) | next_y);
        }
    }
}

static inline int osrs_reference_route_is_target(
    const EncounterRouteInput* input,
    int x,
    int y
) {
    int64_t actor_max_x = (int64_t)x + input->actor_size - 1;
    int64_t actor_max_y = (int64_t)y + input->actor_size - 1;
    int64_t target_max_x =
        (int64_t)input->target_x + input->target_size - 1;
    int64_t target_max_y =
        (int64_t)input->target_y + input->target_size - 1;
    int x_overlap =
        (int64_t)x <= target_max_x && (int64_t)input->target_x <= actor_max_x;
    int y_overlap =
        (int64_t)y <= target_max_y && (int64_t)input->target_y <= actor_max_y;
    if (x_overlap && y_overlap) return 0;
    return (y_overlap &&
            (actor_max_x + 1 == input->target_x ||
             target_max_x + 1 == x)) ||
        (x_overlap &&
         (actor_max_y + 1 == input->target_y ||
          target_max_y + 1 == y));
}

static inline int osrs_reference_target_distance_squared(
    const EncounterRouteInput* input,
    int x,
    int y
) {
    int64_t target_max_x =
        (int64_t)input->target_x + input->target_size - 1;
    int64_t target_max_y =
        (int64_t)input->target_y + input->target_size - 1;
    int64_t dx = 0;
    int64_t dy = 0;
    if ((int64_t)x < input->target_x) dx = input->target_x - (int64_t)x;
    else if (target_max_x < x) dx = (int64_t)x - target_max_x;
    if ((int64_t)y < input->target_y) dy = input->target_y - (int64_t)y;
    else if (target_max_y < y) dy = (int64_t)y - target_max_y;
    int64_t squared = dx * dx + dy * dy;
    return squared > INT_MAX ? INT_MAX : (int)squared;
}

static inline void osrs_reference_route_parent(
    int via,
    int* x,
    int* y
) {
    if (via & VIA_W) (*x)++;
    else if (via & VIA_E) (*x)--;
    if (via & VIA_S) (*y)++;
    else if (via & VIA_N) (*y)--;
}

static inline EncounterRouteResult osrs_reference_route_query(
    const EncounterRouteInput* input,
    const OsrsReferenceRouteField* field
) {
    EncounterRouteResult result;
    memset(&result, 0, sizeof(result));
    const EncounterArenaTopology* topology = input->topology;
    int selected_index = -1;
    for (int i = 0; i < field->count; i++) {
        int packed = field->queue[i];
        int local_x = packed >> 6;
        int local_y = packed & 63;
        int x = topology->origin_x + local_x;
        int y = topology->origin_y + local_y;
        if (osrs_reference_route_is_target(input, x, y)) {
            selected_index = local_x * topology->height + local_y;
            break;
        }
    }
    if (selected_index >= 0) {
        result.outcome = ROUTE_REACHED_TARGET;
    } else {
        int best_distance = INT_MAX;
        uint16_t best_depth = UINT16_MAX;
        for (int local_x = 0; local_x < topology->width; local_x++) {
            for (int local_y = 0; local_y < topology->height; local_y++) {
                if ((field->visited[local_x] &
                        (UINT64_C(1) << local_y)) == 0)
                    continue;
                int x = topology->origin_x + local_x;
                int y = topology->origin_y + local_y;
                int min_x =
                    input->target_x - PATHFIND_MAX_FALLBACK_RADIUS;
                int max_x = input->target_x + input->target_size - 1 +
                    PATHFIND_MAX_FALLBACK_RADIUS;
                int min_y =
                    input->target_y - PATHFIND_MAX_FALLBACK_RADIUS;
                int max_y = input->target_y + input->target_size - 1 +
                    PATHFIND_MAX_FALLBACK_RADIUS;
                if (x < min_x || x > max_x || y < min_y || y > max_y)
                    continue;
                int index = local_x * topology->height + local_y;
                int distance =
                    osrs_reference_target_distance_squared(input, x, y);
                uint16_t depth = field->depth[index];
                if (distance < best_distance ||
                        (distance == best_distance && depth < best_depth)) {
                    selected_index = index;
                    best_distance = distance;
                    best_depth = depth;
                }
            }
        }
        if (selected_index < 0) {
            result.outcome = ROUTE_UNREACHABLE;
            return result;
        }
        result.outcome = ROUTE_REACHED_FALLBACK;
    }
    int local_x = selected_index / topology->height;
    int local_y = selected_index % topology->height;
    result.destination_x = topology->origin_x + local_x;
    result.destination_y = topology->origin_y + local_y;
    result.distance = field->depth[selected_index];
    int source_x = input->source_x - topology->origin_x;
    int source_y = input->source_y - topology->origin_y;
    int first_x = source_x;
    int first_y = source_y;
    int second_x = source_x;
    int second_y = source_y;
    while (local_x != source_x || local_y != source_y) {
        int index = local_x * topology->height + local_y;
        uint16_t depth = field->depth[index];
        if (depth == 1) {
            first_x = local_x;
            first_y = local_y;
        } else if (depth == 2) {
            second_x = local_x;
            second_y = local_y;
        }
        osrs_reference_route_parent(field->via[index], &local_x, &local_y);
    }
    result.first_dx = first_x - source_x;
    result.first_dy = first_y - source_y;
    if (result.distance >= 2) {
        result.run_dx = second_x - first_x;
        result.run_dy = second_y - first_y;
    }
    return result;
}

static inline uint64_t osrs_reference_route_exhaustive(
    const char* scenario,
    const EncounterArenaTopology* topology,
    EncounterRouteBlockers blockers,
    int maximum_target_size,
    const int* attack_ranges,
    int attack_range_count,
    osrs_reference_can_attack_fn can_attack,
    void* attack_ctx
) {
    uint64_t checks = 0;
    for (int source_x = topology->origin_x;
            source_x < topology->origin_x + topology->width;
            source_x++) {
        for (int source_y = topology->origin_y;
                source_y < topology->origin_y + topology->height;
                source_y++) {
            if (encounter_arena_topology_footprint_blocked(
                    topology, source_x, source_y, 1) ||
                    (blockers.is_blocked &&
                     blockers.is_blocked(
                        blockers.ctx, source_x, source_y, 1)))
                continue;
            EncounterRouteInput input = {
                .topology = topology,
                .blockers = blockers,
                .source_x = source_x,
                .source_y = source_y,
                .actor_size = 1,
                .target_kind = ENCOUNTER_ROUTE_TARGET_CARDINAL_ADJACENCY,
                .movement_mode = ENCOUNTER_ROUTE_MOVEMENT_RUN,
                .cost_policy = ENCOUNTER_ROUTE_COST_OSRS,
            };
            OsrsReferenceRouteField field;
            osrs_reference_route_build(&input, &field);
            for (int target_size = 1;
                    target_size <= maximum_target_size;
                    target_size++) {
                input.target_size = target_size;
                for (int target_x = topology->origin_x;
                        target_x + target_size <=
                            topology->origin_x + topology->width;
                        target_x++) {
                    input.target_x = target_x;
                    for (int target_y = topology->origin_y;
                            target_y + target_size <=
                                topology->origin_y + topology->height;
                            target_y++) {
                        if (encounter_arena_topology_footprint_blocked(
                                topology,
                                target_x,
                                target_y,
                                target_size))
                            continue;
                        if (source_x >= target_x &&
                                source_x < target_x + target_size &&
                                source_y >= target_y &&
                                source_y < target_y + target_size)
                            continue;
                        input.target_y = target_y;
                        EncounterRouteResult expected =
                            osrs_reference_route_query(&input, &field);
                        EncounterRouteResult actual =
                            encounter_route_solve(&input);
                        if (actual.outcome != expected.outcome ||
                                actual.destination_x != expected.destination_x ||
                                actual.destination_y != expected.destination_y ||
                                actual.distance != expected.distance ||
                                actual.first_dx != expected.first_dx ||
                                actual.first_dy != expected.first_dy) {
                            fprintf(stderr,
                                "route equivalence mismatch scenario=%s source=(%d,%d) target=(%d,%d,%d) expected=(%d,%d,%d,%d,%d,%u) actual=(%d,%d,%d,%d,%d,%u)\n",
                                scenario,
                                source_x,
                                source_y,
                                target_x,
                                target_y,
                                target_size,
                                expected.outcome,
                                expected.destination_x,
                                expected.destination_y,
                                expected.first_dx,
                                expected.first_dy,
                                expected.distance,
                                actual.outcome,
                                actual.destination_x,
                                actual.destination_y,
                                actual.first_dx,
                                actual.first_dy,
                                actual.distance);
                            abort();
                        }
                        for (int range_index = 0;
                                range_index < attack_range_count;
                                range_index++) {
                            int range = attack_ranges[range_index];
                            int expected_x = source_x;
                            int expected_y = source_y;
                            int actual_x = source_x;
                            int actual_y = source_y;
                            if (!can_attack(
                                    attack_ctx,
                                    source_x,
                                    source_y,
                                    target_x,
                                    target_y,
                                    target_size,
                                    range)) {
                                expected_x += expected.first_dx;
                                expected_y += expected.first_dy;
                                actual_x += actual.first_dx;
                                actual_y += actual.first_dy;
                                if (!can_attack(
                                        attack_ctx,
                                        expected_x,
                                        expected_y,
                                        target_x,
                                        target_y,
                                        target_size,
                                        range)) {
                                    expected_x += expected.run_dx;
                                    expected_y += expected.run_dy;
                                }
                                if (!can_attack(
                                        attack_ctx,
                                        actual_x,
                                        actual_y,
                                        target_x,
                                        target_y,
                                        target_size,
                                        range)) {
                                    actual_x += actual.run_dx;
                                    actual_y += actual.run_dy;
                                }
                            }
                            if (actual_x != expected_x ||
                                    actual_y != expected_y) {
                                fprintf(stderr,
                                    "route landing mismatch scenario=%s source=(%d,%d) target=(%d,%d,%d) range=%d expected=(%d,%d) actual=(%d,%d)\n",
                                    scenario,
                                    source_x,
                                    source_y,
                                    target_x,
                                    target_y,
                                    target_size,
                                    range,
                                    expected_x,
                                    expected_y,
                                    actual_x,
                                    actual_y);
                                abort();
                            }
                            checks++;
                        }
                    }
                }
            }
        }
    }
    return checks;
}

#endif
