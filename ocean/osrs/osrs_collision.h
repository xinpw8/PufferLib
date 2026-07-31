#ifndef OSRS_COLLISION_H
#define OSRS_COLLISION_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "osrs_assets.h"

#define COLLISION_NONE                   0x000000
#define COLLISION_WALL_NORTH_WEST        0x000001
#define COLLISION_WALL_NORTH             0x000002
#define COLLISION_WALL_NORTH_EAST        0x000004
#define COLLISION_WALL_EAST              0x000008
#define COLLISION_WALL_SOUTH_EAST        0x000010
#define COLLISION_WALL_SOUTH             0x000020
#define COLLISION_WALL_SOUTH_WEST        0x000040
#define COLLISION_WALL_WEST              0x000080

#define COLLISION_IMPENETRABLE_WALL_NORTH_WEST  0x000200
#define COLLISION_IMPENETRABLE_WALL_NORTH       0x000400
#define COLLISION_IMPENETRABLE_WALL_NORTH_EAST  0x000800
#define COLLISION_IMPENETRABLE_WALL_EAST        0x001000
#define COLLISION_IMPENETRABLE_WALL_SOUTH_EAST  0x002000
#define COLLISION_IMPENETRABLE_WALL_SOUTH       0x004000
#define COLLISION_IMPENETRABLE_WALL_SOUTH_WEST  0x008000
#define COLLISION_IMPENETRABLE_WALL_WEST        0x010000

#define COLLISION_IMPENETRABLE_BLOCKED   0x020000
#define COLLISION_BRIDGE                 0x040000
#define COLLISION_BLOCKED                0x200000

#define REGION_SIZE      64
#define REGION_HEIGHT_LEVELS 4

typedef struct {
    int flags[REGION_HEIGHT_LEVELS][REGION_SIZE][REGION_SIZE];
} CollisionRegion;

#define REGION_MAP_CAPACITY 256

typedef struct {
    int key;
    CollisionRegion* region;
} RegionMapEntry;

typedef struct {
    RegionMapEntry entries[REGION_MAP_CAPACITY];
    int count;
} CollisionMap;

static inline int collision_region_hash(int x, int y) {
    int region_x = x >> 6;
    int region_y = y >> 6;
    return region_x * 256 + region_y;
}

static inline int collision_local(int coord) {
    return coord & 0x3F;
}

static inline void collision_map_init(CollisionMap* map) {
    map->count = 0;
    for (int i = 0; i < REGION_MAP_CAPACITY; i++) {
        map->entries[i].key = -1;
        map->entries[i].region = NULL;
    }
}

static inline CollisionMap* collision_map_create(void) {
    CollisionMap* map = (CollisionMap*)malloc(sizeof(CollisionMap));
    collision_map_init(map);
    return map;
}

static inline CollisionRegion* collision_map_get(const CollisionMap* map, int key) {
    int idx = key & (REGION_MAP_CAPACITY - 1);
    for (int i = 0; i < REGION_MAP_CAPACITY; i++) {
        int slot = (idx + i) & (REGION_MAP_CAPACITY - 1);
        if (map->entries[slot].key == key) {
            return map->entries[slot].region;
        }
        if (map->entries[slot].key == -1) {
            return NULL;
        }
    }
    return NULL;
}

static inline void collision_map_put(CollisionMap* map, int key, CollisionRegion* region) {
    int idx = key & (REGION_MAP_CAPACITY - 1);
    for (int i = 0; i < REGION_MAP_CAPACITY; i++) {
        int slot = (idx + i) & (REGION_MAP_CAPACITY - 1);
        if (map->entries[slot].key == key) {
            map->entries[slot].region = region;
            return;
        }
        if (map->entries[slot].key == -1) {
            map->entries[slot].key = key;
            map->entries[slot].region = region;
            map->count++;
            return;
        }
    }
    fprintf(stderr, "collision_map_put: map full (capacity %d)\n", REGION_MAP_CAPACITY);
}

static inline void collision_map_free(CollisionMap* map) {
    if (map == NULL) return;
    for (int i = 0; i < REGION_MAP_CAPACITY; i++) {
        if (map->entries[i].region != NULL) {
            free(map->entries[i].region);
        }
    }
    free(map);
}

static inline int collision_get_flags(const CollisionMap* map, int height, int x, int y) {
    if (map == NULL) return COLLISION_NONE;
    int key = collision_region_hash(x, y);
    const CollisionRegion* region = collision_map_get(map, key);
    if (region == NULL) return COLLISION_NONE;
    int lx = collision_local(x);
    int ly = collision_local(y);
    int h = height < 0 ? 0 : (height >= REGION_HEIGHT_LEVELS ? REGION_HEIGHT_LEVELS - 1 : height);
    return region->flags[h][lx][ly];
}

static inline int collision_is_inactive(const CollisionMap* map, int height, int x, int y, int flag) {
    return (collision_get_flags(map, height, x, y) & flag) == 0;
}

static inline int collision_traversable_north(const CollisionMap* map, int height, int x, int y) {
    if (map == NULL) return 1;
    return collision_is_inactive(map, height, x, y + 1,
        COLLISION_WALL_SOUTH | COLLISION_BLOCKED);
}

static inline int collision_traversable_south(const CollisionMap* map, int height, int x, int y) {
    if (map == NULL) return 1;
    return collision_is_inactive(map, height, x, y - 1,
        COLLISION_WALL_NORTH | COLLISION_BLOCKED);
}

static inline int collision_traversable_east(const CollisionMap* map, int height, int x, int y) {
    if (map == NULL) return 1;
    return collision_is_inactive(map, height, x + 1, y,
        COLLISION_WALL_WEST | COLLISION_BLOCKED);
}

static inline int collision_traversable_west(const CollisionMap* map, int height, int x, int y) {
    if (map == NULL) return 1;
    return collision_is_inactive(map, height, x - 1, y,
        COLLISION_WALL_EAST | COLLISION_BLOCKED);
}

static inline int collision_traversable_north_east(const CollisionMap* map, int height, int x, int y) {
    if (map == NULL) return 1;
    return collision_is_inactive(map, height, x + 1, y + 1,
               COLLISION_WALL_WEST | COLLISION_WALL_SOUTH | COLLISION_WALL_SOUTH_WEST | COLLISION_BLOCKED)
        && collision_is_inactive(map, height, x + 1, y,
               COLLISION_WALL_WEST | COLLISION_BLOCKED)
        && collision_is_inactive(map, height, x, y + 1,
               COLLISION_WALL_SOUTH | COLLISION_BLOCKED);
}

static inline int collision_traversable_north_west(const CollisionMap* map, int height, int x, int y) {
    if (map == NULL) return 1;
    return collision_is_inactive(map, height, x - 1, y + 1,
               COLLISION_WALL_EAST | COLLISION_WALL_SOUTH | COLLISION_WALL_SOUTH_EAST | COLLISION_BLOCKED)
        && collision_is_inactive(map, height, x - 1, y,
               COLLISION_WALL_EAST | COLLISION_BLOCKED)
        && collision_is_inactive(map, height, x, y + 1,
               COLLISION_WALL_SOUTH | COLLISION_BLOCKED);
}

static inline int collision_traversable_south_east(const CollisionMap* map, int height, int x, int y) {
    if (map == NULL) return 1;
    return collision_is_inactive(map, height, x + 1, y - 1,
               COLLISION_WALL_WEST | COLLISION_WALL_NORTH | COLLISION_WALL_NORTH_WEST | COLLISION_BLOCKED)
        && collision_is_inactive(map, height, x + 1, y,
               COLLISION_WALL_WEST | COLLISION_BLOCKED)
        && collision_is_inactive(map, height, x, y - 1,
               COLLISION_WALL_NORTH | COLLISION_BLOCKED);
}

static inline int collision_traversable_south_west(const CollisionMap* map, int height, int x, int y) {
    if (map == NULL) return 1;
    return collision_is_inactive(map, height, x - 1, y - 1,
               COLLISION_WALL_EAST | COLLISION_WALL_NORTH | COLLISION_WALL_NORTH_EAST | COLLISION_BLOCKED)
        && collision_is_inactive(map, height, x - 1, y,
               COLLISION_WALL_EAST | COLLISION_BLOCKED)
        && collision_is_inactive(map, height, x, y - 1,
               COLLISION_WALL_NORTH | COLLISION_BLOCKED);
}

static inline int collision_tile_walkable(const CollisionMap* map, int height, int x, int y) {
    if (map == NULL) return 1;
    return (collision_get_flags(map, height, x, y) & COLLISION_BLOCKED) == 0;
}

static inline int collision_traversable_step(const CollisionMap* map, int height,
                                             int x, int y, int dx, int dy) {
    if (map == NULL) return 1;

    if (dx == 0 && dy == 1)  return collision_traversable_north(map, height, x, y);
    if (dx == 0 && dy == -1) return collision_traversable_south(map, height, x, y);
    if (dx == 1 && dy == 0)  return collision_traversable_east(map, height, x, y);
    if (dx == -1 && dy == 0) return collision_traversable_west(map, height, x, y);
    if (dx == 1 && dy == 1)  return collision_traversable_north_east(map, height, x, y);
    if (dx == -1 && dy == 1) return collision_traversable_north_west(map, height, x, y);
    if (dx == 1 && dy == -1) return collision_traversable_south_east(map, height, x, y);
    if (dx == -1 && dy == -1) return collision_traversable_south_west(map, height, x, y);

    return 1;
}

#define COLLISION_MAP_MAGIC 0x50414D43
#define COLLISION_MAP_VERSION 1

static inline CollisionMap* collision_map_load(const char* path) {
    FILE* f = osrs_asset_fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "collision_map_load: cannot open %s\n", path);
        return NULL;
    }

    uint32_t magic, version, region_count;
    if (fread(&magic, 4, 1, f) != 1 || magic != COLLISION_MAP_MAGIC) {
        fprintf(stderr, "collision_map_load: bad magic in %s\n", path);
        fclose(f);
        return NULL;
    }
    if (fread(&version, 4, 1, f) != 1 || version != COLLISION_MAP_VERSION) {
        fprintf(stderr, "collision_map_load: unsupported version %u in %s\n", version, path);
        fclose(f);
        return NULL;
    }
    if (fread(&region_count, 4, 1, f) != 1) {
        fprintf(stderr, "collision_map_load: truncated header in %s\n", path);
        fclose(f);
        return NULL;
    }

    CollisionMap* map = collision_map_create();

    for (uint32_t i = 0; i < region_count; i++) {
        int32_t key;
        if (fread(&key, 4, 1, f) != 1) {
            fprintf(stderr, "collision_map_load: truncated at region %u in %s\n", i, path);
            collision_map_free(map);
            fclose(f);
            return NULL;
        }

        CollisionRegion* region = (CollisionRegion*)calloc(1, sizeof(CollisionRegion));
        size_t flags_size = sizeof(region->flags);
        if (fread(region->flags, 1, flags_size, f) != flags_size) {
            fprintf(stderr, "collision_map_load: truncated flags at region %u in %s\n", i, path);
            free(region);
            collision_map_free(map);
            fclose(f);
            return NULL;
        }

        collision_map_put(map, key, region);
    }

    fclose(f);
    return map;
}

#define LOS_FULL_MASK   0x20000
#define LOS_EAST_MASK   0x01000
#define LOS_WEST_MASK   0x10000
#define LOS_NORTH_MASK  0x00400
#define LOS_SOUTH_MASK  0x04000
#define LOS_FP_SCALE    65536
#define LOS_FP_HALF     32768

typedef struct {
    int x, y;
    int size;
    uint32_t los_mask;
} LOSBlocker;

static uint32_t los_check_tile(const LOSBlocker* blockers, int count,
                                int px, int py) {
    for (int i = 0; i < count; i++) {
        const LOSBlocker* b = &blockers[i];
        if (px >= b->x && px < b->x + b->size &&
            py >= b->y && py < b->y + b->size) {
            return b->los_mask;
        }
    }
    return 0;
}

static int los_aabb_overlap(int x1, int y1, int s1, int x2, int y2, int s2) {
    return !(x1 >= x2 + s2 || x1 + s1 <= x2 || y1 >= y2 + s2 || y1 + s1 <= y2);
}

static int has_line_of_sight(const LOSBlocker* blockers, int blocker_count,
                              int x1, int y1, int x2, int y2,
                              int src_size, int range) {
    int dx = x2 - x1;
    int dy = y2 - y1;

    if (los_check_tile(blockers, blocker_count, x1, y1)) return 0;
    if (los_check_tile(blockers, blocker_count, x2, y2)) return 0;

    if (los_aabb_overlap(x1, y1, src_size, x2, y2, 1)) return 0;

    int adx = dx < 0 ? -dx : dx;
    int ady = dy < 0 ? -dy : dy;

    if (range > 0 && (adx > range || ady > range)) return 0;

    if (adx > ady) {
        int x_tile = x1;
        int y_fp = y1 * LOS_FP_SCALE + LOS_FP_HALF;
        int slope = (dy * LOS_FP_SCALE) / adx;
        int x_inc = (dx > 0) ? 1 : -1;
        uint32_t x_mask = (dx > 0) ? (LOS_WEST_MASK | LOS_FULL_MASK)
                                    : (LOS_EAST_MASK | LOS_FULL_MASK);
        uint32_t y_mask = (dy < 0) ? (LOS_NORTH_MASK | LOS_FULL_MASK)
                                    : (LOS_SOUTH_MASK | LOS_FULL_MASK);
        if (dy < 0) y_fp -= 1;

        while (x_tile != x2) {
            x_tile += x_inc;
            int y_tile = y_fp >> 16;
            if (los_check_tile(blockers, blocker_count, x_tile, y_tile) & x_mask)
                return 0;
            y_fp += slope;
            int new_y = y_fp >> 16;
            if (new_y != y_tile) {
                if (los_check_tile(blockers, blocker_count, x_tile, new_y) & y_mask)
                    return 0;
            }
        }
    } else if (ady > 0) {
        int y_tile = y1;
        int x_fp = x1 * LOS_FP_SCALE + LOS_FP_HALF;
        int slope = (dx * LOS_FP_SCALE) / ady;
        int y_inc = (dy > 0) ? 1 : -1;
        uint32_t y_mask = (dy > 0) ? (LOS_SOUTH_MASK | LOS_FULL_MASK)
                                    : (LOS_NORTH_MASK | LOS_FULL_MASK);
        uint32_t x_mask = (dx < 0) ? (LOS_EAST_MASK | LOS_FULL_MASK)
                                    : (LOS_WEST_MASK | LOS_FULL_MASK);
        if (dx < 0) x_fp -= 1;

        while (y_tile != y2) {
            y_tile += y_inc;
            int x_tile = x_fp >> 16;
            if (los_check_tile(blockers, blocker_count, x_tile, y_tile) & y_mask)
                return 0;
            x_fp += slope;
            int new_x = x_fp >> 16;
            if (new_x != x_tile) {
                if (los_check_tile(blockers, blocker_count, new_x, y_tile) & x_mask)
                    return 0;
            }
        }
    }

    return 1;
}

static inline int los_intervals_overlap(int a0, int a1, int b0, int b1) {
    return !(a1 < b0 || b1 < a0);
}

static inline int entity_has_line_of_sight(
    const LOSBlocker* blockers, int blocker_count,
    int ax, int ay, int a_size,
    int tx, int ty, int t_size,
    int range
) {
    if (range == 1) {
        if (los_aabb_overlap(ax, ay, a_size, tx, ty, t_size)) return 0;

        int a_x0 = ax;
        int a_x1 = ax + a_size - 1;
        int a_y0 = ay;
        int a_y1 = ay + a_size - 1;
        int t_x0 = tx;
        int t_x1 = tx + t_size - 1;
        int t_y0 = ty;
        int t_y1 = ty + t_size - 1;

        return (a_x1 + 1 == t_x0 && los_intervals_overlap(a_y0, a_y1, t_y0, t_y1)) ||
               (t_x1 + 1 == a_x0 && los_intervals_overlap(a_y0, a_y1, t_y0, t_y1)) ||
               (a_y1 + 1 == t_y0 && los_intervals_overlap(a_x0, a_x1, t_x0, t_x1)) ||
               (t_y1 + 1 == a_y0 && los_intervals_overlap(a_x0, a_x1, t_x0, t_x1));
    }

    int a_px = tx;
    if (a_px < ax) a_px = ax;
    if (a_px >= ax + a_size) a_px = ax + a_size - 1;
    int a_py = ty;
    if (a_py < ay) a_py = ay;
    if (a_py >= ay + a_size) a_py = ay + a_size - 1;

    int t_px = ax;
    if (t_px < tx) t_px = tx;
    if (t_px >= tx + t_size) t_px = tx + t_size - 1;
    int t_py = ay;
    if (t_py < ty) t_py = ty;
    if (t_py >= ty + t_size) t_py = ty + t_size - 1;

    return has_line_of_sight(blockers, blocker_count, a_px, a_py, t_px, t_py, 1, range);
}

#endif
