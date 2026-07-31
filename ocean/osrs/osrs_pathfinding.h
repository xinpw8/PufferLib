#ifndef OSRS_PATHFINDING_H
#define OSRS_PATHFINDING_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
#define OSRS_THREAD_LOCAL thread_local
#else
#define OSRS_THREAD_LOCAL _Thread_local
#endif

#include "osrs_collision.h"

#define PATHFIND_GRID_SIZE 104
#define PATHFIND_ARENA_MAX 48
#define PATHFIND_MAX_QUEUE_FULL (PATHFIND_GRID_SIZE * PATHFIND_GRID_SIZE)
#define PATHFIND_MAX_QUEUE_ARENA (PATHFIND_ARENA_MAX * PATHFIND_ARENA_MAX)
#define PATHFIND_MAX_FALLBACK_RADIUS 10

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
        int v = BFS_VIA(cur_x, cur_y);
        int prev_x = cur_x, prev_y = cur_y;
        if (v & VIA_W) prev_x++; else if (v & VIA_E) prev_x--;
        if (v & VIA_S) prev_y++; else if (v & VIA_N) prev_y--;
        if (prev_x == local_src_x && prev_y == local_src_y) {
            result.found = 1;
            result.next_dx = cur_x - local_src_x;
            result.next_dy = cur_y - local_src_y;
            return result;
        }
        cur_x = prev_x; cur_y = prev_y;
        if (BFS_VIA(cur_x, cur_y) == VIA_NONE || BFS_VIA(cur_x, cur_y) == VIA_START) break;
    }

    return result;
}

#endif
