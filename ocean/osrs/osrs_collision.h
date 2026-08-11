#ifndef OSRS_COLLISION_H
#define OSRS_COLLISION_H

#include <limits.h>
#include <stdint.h>
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

static inline int collision_flags_traversable_step(
    uint32_t destination_flags,
    uint32_t horizontal_side_flags,
    uint32_t vertical_side_flags,
    int dx,
    int dy
) {
    if (dx == 0 && dy == 1)
        return (destination_flags & (COLLISION_WALL_SOUTH | COLLISION_BLOCKED)) == 0;
    if (dx == 0 && dy == -1)
        return (destination_flags & (COLLISION_WALL_NORTH | COLLISION_BLOCKED)) == 0;
    if (dx == 1 && dy == 0)
        return (destination_flags & (COLLISION_WALL_WEST | COLLISION_BLOCKED)) == 0;
    if (dx == -1 && dy == 0)
        return (destination_flags & (COLLISION_WALL_EAST | COLLISION_BLOCKED)) == 0;
    if (dx == 1 && dy == 1)
        return (destination_flags &
                (COLLISION_WALL_WEST | COLLISION_WALL_SOUTH |
                 COLLISION_WALL_SOUTH_WEST | COLLISION_BLOCKED)) == 0 &&
            (horizontal_side_flags &
                (COLLISION_WALL_WEST | COLLISION_BLOCKED)) == 0 &&
            (vertical_side_flags &
                (COLLISION_WALL_SOUTH | COLLISION_BLOCKED)) == 0;
    if (dx == -1 && dy == 1)
        return (destination_flags &
                (COLLISION_WALL_EAST | COLLISION_WALL_SOUTH |
                 COLLISION_WALL_SOUTH_EAST | COLLISION_BLOCKED)) == 0 &&
            (horizontal_side_flags &
                (COLLISION_WALL_EAST | COLLISION_BLOCKED)) == 0 &&
            (vertical_side_flags &
                (COLLISION_WALL_SOUTH | COLLISION_BLOCKED)) == 0;
    if (dx == 1 && dy == -1)
        return (destination_flags &
                (COLLISION_WALL_WEST | COLLISION_WALL_NORTH |
                 COLLISION_WALL_NORTH_WEST | COLLISION_BLOCKED)) == 0 &&
            (horizontal_side_flags &
                (COLLISION_WALL_WEST | COLLISION_BLOCKED)) == 0 &&
            (vertical_side_flags &
                (COLLISION_WALL_NORTH | COLLISION_BLOCKED)) == 0;
    if (dx == -1 && dy == -1)
        return (destination_flags &
                (COLLISION_WALL_EAST | COLLISION_WALL_NORTH |
                 COLLISION_WALL_NORTH_EAST | COLLISION_BLOCKED)) == 0 &&
            (horizontal_side_flags &
                (COLLISION_WALL_EAST | COLLISION_BLOCKED)) == 0 &&
            (vertical_side_flags &
                (COLLISION_WALL_NORTH | COLLISION_BLOCKED)) == 0;
    return 1;
}

static inline int collision_traversable_step(
    const CollisionMap* map,
    int height,
    int x,
    int y,
    int dx,
    int dy
) {
    if (map == NULL) return 1;
    uint32_t horizontal_side_flags = 0;
    uint32_t vertical_side_flags = 0;
    if (dx != 0 && dy != 0) {
        horizontal_side_flags =
            (uint32_t)collision_get_flags(map, height, x + dx, y);
        vertical_side_flags =
            (uint32_t)collision_get_flags(map, height, x, y + dy);
    }
    return collision_flags_traversable_step(
        (uint32_t)collision_get_flags(map, height, x + dx, y + dy),
        horizontal_side_flags,
        vertical_side_flags,
        dx,
        dy);
}

static inline int collision_traversable_north(
    const CollisionMap* map, int height, int x, int y
) {
    return collision_traversable_step(map, height, x, y, 0, 1);
}

static inline int collision_traversable_south(
    const CollisionMap* map, int height, int x, int y
) {
    return collision_traversable_step(map, height, x, y, 0, -1);
}

static inline int collision_traversable_east(
    const CollisionMap* map, int height, int x, int y
) {
    return collision_traversable_step(map, height, x, y, 1, 0);
}

static inline int collision_traversable_west(
    const CollisionMap* map, int height, int x, int y
) {
    return collision_traversable_step(map, height, x, y, -1, 0);
}

static inline int collision_traversable_north_east(
    const CollisionMap* map, int height, int x, int y
) {
    return collision_traversable_step(map, height, x, y, 1, 1);
}

static inline int collision_traversable_north_west(
    const CollisionMap* map, int height, int x, int y
) {
    return collision_traversable_step(map, height, x, y, -1, 1);
}

static inline int collision_traversable_south_east(
    const CollisionMap* map, int height, int x, int y
) {
    return collision_traversable_step(map, height, x, y, 1, -1);
}

static inline int collision_traversable_south_west(
    const CollisionMap* map, int height, int x, int y
) {
    return collision_traversable_step(map, height, x, y, -1, -1);
}

static inline int collision_tile_walkable(const CollisionMap* map, int height, int x, int y) {
    if (map == NULL) return 1;
    return (collision_get_flags(map, height, x, y) & COLLISION_BLOCKED) == 0;
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
static inline int los_aabb_overlap(
    int x1, int y1, int s1, int x2, int y2, int s2
) {
    return !(x1 >= x2 + s2 || x1 + s1 <= x2 ||
        y1 >= y2 + s2 || y1 + s1 <= y2);
}


typedef uint32_t (*los_tile_flags_fn)(void* ctx, int x, int y);

typedef struct {
    const LOSBlocker* blockers;
    int count;
} LOSBlockerFlagsContext;

static inline uint32_t los_check_tile(
    const LOSBlocker* blockers, int count, int px, int py
) {
    for (int i = 0; i < count; i++) {
        const LOSBlocker* b = &blockers[i];
        if (px >= b->x && px < b->x + b->size &&
                py >= b->y && py < b->y + b->size)
            return b->los_mask;
    }
    return 0;
}

static inline uint32_t los_blocker_tile_flags(void* ctx, int x, int y) {
    const LOSBlockerFlagsContext* blockers =
        (const LOSBlockerFlagsContext*)ctx;
    return los_check_tile(blockers->blockers, blockers->count, x, y);
}

static inline int los_has_line_of_sight_with_flags(
    los_tile_flags_fn tile_flags,
    void* tile_flags_ctx,
    int x1,
    int y1,
    int x2,
    int y2,
    int src_size,
    int range
) {
    int dx = x2 - x1;
    int dy = y2 - y1;

    if (tile_flags(tile_flags_ctx, x1, y1)) return 0;
    if (tile_flags(tile_flags_ctx, x2, y2)) return 0;
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
            if (tile_flags(tile_flags_ctx, x_tile, y_tile) & x_mask)
                return 0;
            y_fp += slope;
            int new_y = y_fp >> 16;
            if (new_y != y_tile &&
                    (tile_flags(tile_flags_ctx, x_tile, new_y) & y_mask))
                return 0;
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
            if (tile_flags(tile_flags_ctx, x_tile, y_tile) & y_mask)
                return 0;
            x_fp += slope;
            int new_x = x_fp >> 16;
            if (new_x != x_tile &&
                    (tile_flags(tile_flags_ctx, new_x, y_tile) & x_mask))
                return 0;
        }
    }

    return 1;
}

static int has_line_of_sight(
    const LOSBlocker* blockers,
    int blocker_count,
    int x1,
    int y1,
    int x2,
    int y2,
    int src_size,
    int range
) {
    LOSBlockerFlagsContext ctx = {
        .blockers = blockers,
        .count = blocker_count,
    };
    return los_has_line_of_sight_with_flags(
        los_blocker_tile_flags, &ctx,
        x1, y1, x2, y2, src_size, range);
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

#define ENCOUNTER_ARENA_TOPOLOGY_MAX_DIMENSION 64
#define ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES \
    (ENCOUNTER_ARENA_TOPOLOGY_MAX_DIMENSION * \
     ENCOUNTER_ARENA_TOPOLOGY_MAX_DIMENSION)
#define ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE 5
#define ENCOUNTER_ARENA_TOPOLOGY_LOS_WORDS \
    ((ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES * \
      ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES + 63) / 64)

typedef uint32_t (*encounter_arena_tile_flags_fn)(void* ctx, int x, int y);

typedef struct {
    int origin_x;
    int origin_y;
    int width;
    int height;
    int max_footprint_size;
    uint64_t revision;
    encounter_arena_tile_flags_fn tile_flags;
    void* tile_flags_ctx;
} EncounterArenaTopologyBuildSpec;

typedef struct EncounterArenaTopology {
    int origin_x;
    int origin_y;
    int width;
    int height;
    int tile_count;
    int max_footprint_size;
    uint64_t revision;
    uint8_t finalized;
    uint32_t static_collision_flags[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint8_t static_blocked[ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint8_t footprint_blocked
        [ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE]
        [ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint8_t legal_step_masks
        [ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE]
        [ENCOUNTER_ARENA_TOPOLOGY_MAX_TILES];
    uint64_t static_los_bits[ENCOUNTER_ARENA_TOPOLOGY_LOS_WORDS];
} EncounterArenaTopology;

static inline void encounter_arena_topology_abort(
    const char* reason,
    int value
) {
    fprintf(stderr, "invalid OSRS arena topology %s: %d\n", reason, value);
    abort();
}

static inline int encounter_arena_topology_contains_raw(
    const EncounterArenaTopology* topology,
    int x,
    int y
) {
    return x >= topology->origin_x &&
        x < topology->origin_x + topology->width &&
        y >= topology->origin_y &&
        y < topology->origin_y + topology->height;
}

static inline int encounter_arena_topology_index_raw(
    const EncounterArenaTopology* topology,
    int x,
    int y
) {
    return (x - topology->origin_x) * topology->height +
        (y - topology->origin_y);
}

static inline uint32_t encounter_arena_topology_build_flags(
    const EncounterArenaTopologyBuildSpec* spec,
    int x,
    int y
) {
    if (!spec->tile_flags) return 0;
    return spec->tile_flags(spec->tile_flags_ctx, x, y);
}

static inline int encounter_arena_topology_footprint_blocked_raw(
    const EncounterArenaTopology* topology,
    int x,
    int y,
    int size
) {
    if (x < topology->origin_x ||
            y < topology->origin_y ||
            x > topology->origin_x + topology->width - size ||
            y > topology->origin_y + topology->height - size)
        return 1;
    int index = encounter_arena_topology_index_raw(topology, x, y);
    return topology->footprint_blocked[size - 1][index] != 0;
}

static inline int encounter_arena_topology_build_step_allowed(
    const EncounterArenaTopologyBuildSpec* spec,
    const EncounterArenaTopology* topology,
    int x,
    int y,
    int size,
    int dx,
    int dy
) {
    if (encounter_arena_topology_footprint_blocked_raw(
            topology, x, y, size) ||
            encounter_arena_topology_footprint_blocked_raw(
                topology, x + dx, y + dy, size))
        return 0;

    if (dx != 0) {
        int leading_x = dx > 0 ? x + size - 1 : x;
        for (int offset = 0; offset < size; offset++) {
            int source_y = y + offset;
            if (!collision_flags_traversable_step(
                    encounter_arena_topology_build_flags(
                        spec, leading_x + dx, source_y),
                    encounter_arena_topology_build_flags(
                        spec, leading_x + dx, source_y),
                    encounter_arena_topology_build_flags(
                        spec, leading_x, source_y),
                    dx,
                    0))
                return 0;
        }
    }

    if (dy != 0) {
        int leading_y = dy > 0 ? y + size - 1 : y;
        for (int offset = 0; offset < size; offset++) {
            int source_x = x + offset;
            if (!collision_flags_traversable_step(
                    encounter_arena_topology_build_flags(
                        spec, source_x, leading_y + dy),
                    encounter_arena_topology_build_flags(
                        spec, source_x, leading_y),
                    encounter_arena_topology_build_flags(
                        spec, source_x, leading_y + dy),
                    0,
                    dy))
                return 0;
        }
    }

    if (dx != 0 && dy != 0) {
        int corner_x = dx > 0 ? x + size - 1 : x;
        int corner_y = dy > 0 ? y + size - 1 : y;
        if (!collision_flags_traversable_step(
                encounter_arena_topology_build_flags(
                    spec, corner_x + dx, corner_y + dy),
                encounter_arena_topology_build_flags(
                    spec, corner_x + dx, corner_y),
                encounter_arena_topology_build_flags(
                    spec, corner_x, corner_y + dy),
                dx,
                dy))
            return 0;
    }

    return 1;
}

static uint32_t encounter_arena_topology_los_flags(
    void* ctx,
    int x,
    int y
) {
    const EncounterArenaTopology* topology =
        (const EncounterArenaTopology*)ctx;
    if (!encounter_arena_topology_contains_raw(topology, x, y))
        return LOS_FULL_MASK;
    uint32_t flags = topology->static_collision_flags[
        encounter_arena_topology_index_raw(topology, x, y)];
    return flags &
        (LOS_FULL_MASK | LOS_EAST_MASK | LOS_WEST_MASK |
         LOS_NORTH_MASK | LOS_SOUTH_MASK);
}

static inline void encounter_arena_topology_set_los(
    EncounterArenaTopology* topology,
    int source_index,
    int target_index
) {
    size_t bit_index =
        (size_t)source_index * (size_t)topology->tile_count +
        (size_t)target_index;
    topology->static_los_bits[bit_index >> 6] |=
        UINT64_C(1) << (bit_index & 63);
}

static inline EncounterArenaTopology* encounter_arena_topology_build(
    const EncounterArenaTopologyBuildSpec* spec
) {
    if (!spec) encounter_arena_topology_abort("build spec", 0);
    if (spec->width < 1 ||
            spec->width > ENCOUNTER_ARENA_TOPOLOGY_MAX_DIMENSION)
        encounter_arena_topology_abort("width", spec->width);
    if (spec->height < 1 ||
            spec->height > ENCOUNTER_ARENA_TOPOLOGY_MAX_DIMENSION)
        encounter_arena_topology_abort("height", spec->height);
    if (spec->origin_x > INT_MAX - (spec->width - 1))
        encounter_arena_topology_abort("origin x", spec->origin_x);
    if (spec->origin_y > INT_MAX - (spec->height - 1))
        encounter_arena_topology_abort("origin y", spec->origin_y);
    if (spec->max_footprint_size < 1 ||
            spec->max_footprint_size >
                ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE)
        encounter_arena_topology_abort(
            "footprint size", spec->max_footprint_size);
    if (spec->revision == 0)
        encounter_arena_topology_abort("revision", 0);

    EncounterArenaTopology* topology =
        (EncounterArenaTopology*)calloc(1, sizeof(*topology));
    if (!topology) {
        fprintf(stderr, "failed to allocate OSRS arena topology\n");
        abort();
    }

    topology->origin_x = spec->origin_x;
    topology->origin_y = spec->origin_y;
    topology->width = spec->width;
    topology->height = spec->height;
    topology->tile_count = spec->width * spec->height;
    topology->max_footprint_size = spec->max_footprint_size;
    topology->revision = spec->revision;

    for (int local_x = 0; local_x < topology->width; local_x++) {
        for (int local_y = 0; local_y < topology->height; local_y++) {
            int x = topology->origin_x + local_x;
            int y = topology->origin_y + local_y;
            int index = encounter_arena_topology_index_raw(topology, x, y);
            uint32_t flags =
                encounter_arena_topology_build_flags(spec, x, y);
            topology->static_collision_flags[index] = flags;
            topology->static_blocked[index] =
                (uint8_t)((flags & COLLISION_BLOCKED) != 0);
        }
    }

    for (int size = 1; size <= topology->max_footprint_size; size++) {
        for (int local_x = 0; local_x < topology->width; local_x++) {
            for (int local_y = 0; local_y < topology->height; local_y++) {
                int x = topology->origin_x + local_x;
                int y = topology->origin_y + local_y;
                int index =
                    encounter_arena_topology_index_raw(topology, x, y);
                int blocked =
                    local_x + size > topology->width ||
                    local_y + size > topology->height;
                for (int footprint_x = 0;
                        footprint_x < size && !blocked;
                        footprint_x++) {
                    for (int footprint_y = 0;
                            footprint_y < size;
                            footprint_y++) {
                        int footprint_index =
                            encounter_arena_topology_index_raw(
                                topology,
                                x + footprint_x,
                                y + footprint_y);
                        if (topology->static_blocked[footprint_index]) {
                            blocked = 1;
                            break;
                        }
                    }
                }
                topology->footprint_blocked[size - 1][index] =
                    (uint8_t)blocked;
            }
        }
    }

    for (int size = 1; size <= topology->max_footprint_size; size++) {
        for (int local_x = 0; local_x < topology->width; local_x++) {
            for (int local_y = 0; local_y < topology->height; local_y++) {
                int x = topology->origin_x + local_x;
                int y = topology->origin_y + local_y;
                int index =
                    encounter_arena_topology_index_raw(topology, x, y);
                uint8_t mask = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        int bit = (dy + 1) * 3 + (dx + 1);
                        if (bit > 4) bit--;
                        if (encounter_arena_topology_build_step_allowed(
                                spec, topology, x, y, size, dx, dy))
                            mask |= (uint8_t)(1u << bit);
                    }
                }
                topology->legal_step_masks[size - 1][index] = mask;
            }
        }
    }

    for (int source = 0; source < topology->tile_count; source++) {
        int source_x = topology->origin_x + source / topology->height;
        int source_y = topology->origin_y + source % topology->height;
        if (encounter_arena_topology_los_flags(
                topology, source_x, source_y) == 0)
            encounter_arena_topology_set_los(topology, source, source);

        for (int target = source + 1;
                target < topology->tile_count;
                target++) {
            int target_x =
                topology->origin_x + target / topology->height;
            int target_y =
                topology->origin_y + target % topology->height;
            int forward = los_has_line_of_sight_with_flags(
                encounter_arena_topology_los_flags,
                topology,
                source_x,
                source_y,
                target_x,
                target_y,
                1,
                0);
            int reverse = los_has_line_of_sight_with_flags(
                encounter_arena_topology_los_flags,
                topology,
                target_x,
                target_y,
                source_x,
                source_y,
                1,
                0);
            if (forward && reverse) {
                encounter_arena_topology_set_los(
                    topology, source, target);
                encounter_arena_topology_set_los(
                    topology, target, source);
            }
        }
    }

    return topology;
}

static inline void encounter_arena_topology_finalize(
    EncounterArenaTopology* topology
) {
    if (!topology) encounter_arena_topology_abort("finalize", 0);
    if (topology->finalized)
        encounter_arena_topology_abort("double finalization", 1);
    topology->finalized = 1;
}

static inline void encounter_arena_topology_require_finalized(
    const EncounterArenaTopology* topology
) {
    if (!topology || !topology->finalized)
        encounter_arena_topology_abort("unfinalized query", 0);
}

static inline const EncounterArenaTopology*
encounter_arena_topology_require_revision(
    const EncounterArenaTopology* topology,
    uint64_t revision
) {
    encounter_arena_topology_require_finalized(topology);
    if (revision == 0 || topology->revision != revision) {
        fprintf(stderr,
            "stale OSRS arena topology revision: expected %llu got %llu\n",
            (unsigned long long)topology->revision,
            (unsigned long long)revision);
        abort();
    }
    return topology;
}

static inline int encounter_arena_topology_contains(
    const EncounterArenaTopology* topology,
    int x,
    int y
) {
    encounter_arena_topology_require_finalized(topology);
    return encounter_arena_topology_contains_raw(topology, x, y);
}

static inline int encounter_arena_topology_tile_blocked(
    const EncounterArenaTopology* topology,
    int x,
    int y
) {
    encounter_arena_topology_require_finalized(topology);
    if (!encounter_arena_topology_contains_raw(topology, x, y)) return 1;
    return topology->static_blocked[
        encounter_arena_topology_index_raw(topology, x, y)] != 0;
}

static inline void encounter_arena_topology_require_footprint_size(
    const EncounterArenaTopology* topology,
    int size
) {
    if (size < 1 || size > topology->max_footprint_size)
        encounter_arena_topology_abort("query footprint size", size);
}

static inline int encounter_arena_topology_footprint_blocked(
    const EncounterArenaTopology* topology,
    int x,
    int y,
    int size
) {
    encounter_arena_topology_require_finalized(topology);
    encounter_arena_topology_require_footprint_size(topology, size);
    return encounter_arena_topology_footprint_blocked_raw(
        topology, x, y, size);
}

static inline int encounter_arena_topology_step_allowed(
    const EncounterArenaTopology* topology,
    int x,
    int y,
    int size,
    int dx,
    int dy
) {
    encounter_arena_topology_require_finalized(topology);
    encounter_arena_topology_require_footprint_size(topology, size);
    if (dx < -1 || dx > 1 || dy < -1 || dy > 1 ||
            (dx == 0 && dy == 0))
        encounter_arena_topology_abort("step direction", 0);
    if (!encounter_arena_topology_contains_raw(topology, x, y)) return 0;
    int bit = (dy + 1) * 3 + (dx + 1);
    if (bit > 4) bit--;
    int index = encounter_arena_topology_index_raw(topology, x, y);
    return (topology->legal_step_masks[size - 1][index] &
        (uint8_t)(1u << bit)) != 0;
}

static inline int encounter_arena_topology_los_clear(
    const EncounterArenaTopology* topology,
    int actor_x,
    int actor_y,
    int actor_size,
    int target_x,
    int target_y,
    int target_size,
    int attack_range
) {
    encounter_arena_topology_require_finalized(topology);
    encounter_arena_topology_require_footprint_size(
        topology, actor_size);
    encounter_arena_topology_require_footprint_size(
        topology, target_size);
    if (attack_range <= 1) return 1;
    if (encounter_arena_topology_footprint_blocked_raw(
            topology, actor_x, actor_y, actor_size) ||
            encounter_arena_topology_footprint_blocked_raw(
                topology, target_x, target_y, target_size))
        return 0;

    int actor_los_x = target_x;
    if (actor_los_x < actor_x) actor_los_x = actor_x;
    if (actor_los_x >= actor_x + actor_size)
        actor_los_x = actor_x + actor_size - 1;
    int actor_los_y = target_y;
    if (actor_los_y < actor_y) actor_los_y = actor_y;
    if (actor_los_y >= actor_y + actor_size)
        actor_los_y = actor_y + actor_size - 1;

    int target_los_x = actor_x;
    if (target_los_x < target_x) target_los_x = target_x;
    if (target_los_x >= target_x + target_size)
        target_los_x = target_x + target_size - 1;
    int target_los_y = actor_y;
    if (target_los_y < target_y) target_los_y = target_y;
    if (target_los_y >= target_y + target_size)
        target_los_y = target_y + target_size - 1;

    int actor_index = encounter_arena_topology_index_raw(
        topology, actor_los_x, actor_los_y);
    int target_index = encounter_arena_topology_index_raw(
        topology, target_los_x, target_los_y);
    size_t bit_index =
        (size_t)actor_index * (size_t)topology->tile_count +
        (size_t)target_index;
    return (int)((topology->static_los_bits[bit_index >> 6] >>
        (bit_index & 63)) & UINT64_C(1));
}

#endif
