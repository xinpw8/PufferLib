#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(TEST_ROUTE_TOPOLOGY_INFERNO)
#include "ocean/osrs/encounters/encounter_inferno.h"
#define TEST_DEF ENCOUNTER_INFERNO
#define TEST_MAP_PATH "ocean/osrs/data/inferno.cmap"
#define TEST_OFFSET_X 2246
#define TEST_OFFSET_Y 5315
#define TEST_ORIGIN_X INF_TOPOLOGY_MIN_X
#define TEST_ORIGIN_Y INF_TOPOLOGY_MIN_Y
#define TEST_WIDTH INF_TOPOLOGY_WIDTH
#define TEST_HEIGHT INF_TOPOLOGY_HEIGHT
typedef InfernoContext TestContext;
#define TEST_TOPOLOGY(ctx) ((ctx)->route_topology)
static uint32_t expected_flags(const CollisionMap* map, int x, int y) {
    if (x < INF_ARENA_MIN_X || x > INF_ARENA_MAX_X ||
            y < INF_ARENA_MIN_Y || y > INF_ARENA_MAX_Y)
        return COLLISION_BLOCKED;
    return (uint32_t)collision_get_flags(
        map, 0, x + TEST_OFFSET_X, y + TEST_OFFSET_Y);
}
#elif defined(TEST_ROUTE_TOPOLOGY_COLOSSEUM)
#include "ocean/osrs/encounters/encounter_colosseum.h"
#define TEST_DEF ENCOUNTER_COLOSSEUM
#define TEST_MAP_PATH "ocean/osrs/data/colosseum.cmap"
#define TEST_OFFSET_X 1808
#define TEST_OFFSET_Y 3090
#define TEST_ORIGIN_X COLO_ARENA_MIN_X
#define TEST_ORIGIN_Y COLO_ARENA_MIN_Y
#define TEST_WIDTH COLO_ARENA_WIDTH
#define TEST_HEIGHT COLO_ARENA_HEIGHT
typedef ColosseumContext TestContext;
#define TEST_TOPOLOGY(ctx) ((ctx)->route_topology)
static uint32_t expected_flags(const CollisionMap* map, int x, int y) {
    (void)map;
    return col_route_topology_flags(NULL, x, y);
}
#elif defined(TEST_ROUTE_TOPOLOGY_ZULRAH)
#include "ocean/osrs/encounters/encounter_zulrah.h"
#define TEST_DEF ENCOUNTER_ZULRAH
#define TEST_MAP_PATH "ocean/osrs/data/zulrah.cmap"
#define TEST_OFFSET_X 2256
#define TEST_OFFSET_Y 3061
#define TEST_ORIGIN_X 0
#define TEST_ORIGIN_Y 0
#define TEST_WIDTH ZUL_ARENA_SIZE
#define TEST_HEIGHT ZUL_ARENA_SIZE
typedef ZulrahContext TestContext;
#define TEST_TOPOLOGY(ctx) ((ctx)->route_topology)
static uint32_t expected_flags(const CollisionMap* map, int x, int y) {
    return (uint32_t)collision_get_flags(
        map, 0, x + TEST_OFFSET_X, y + TEST_OFFSET_Y);
}
#elif defined(TEST_ROUTE_TOPOLOGY_NH_PVP)
#include "ocean/osrs/encounters/encounter_nh_pvp.h"
#define TEST_DEF ENCOUNTER_NH_PVP
#define TEST_MAP_PATH "ocean/osrs/data/wilderness.cmap"
#define TEST_OFFSET_X 0
#define TEST_OFFSET_Y 0
#define TEST_ORIGIN_X FIGHT_AREA_BASE_X
#define TEST_ORIGIN_Y FIGHT_AREA_BASE_Y
#define TEST_WIDTH FIGHT_AREA_WIDTH
#define TEST_HEIGHT FIGHT_AREA_HEIGHT
typedef NhPvpContext TestContext;
#define TEST_TOPOLOGY(ctx) ((ctx)->route_topology)
static uint32_t expected_flags(const CollisionMap* map, int x, int y) {
    if (!is_in_wilderness(x, y)) return COLLISION_BLOCKED | LOS_FULL_MASK;
    return (uint32_t)collision_get_flags(map, 0, x, y);
}
#else
#error "define one TEST_ROUTE_TOPOLOGY_* encounter"
#endif

static void put_geometry(
    EncounterState* state,
    EncounterContext* context,
    CollisionMap* map,
    int map_first
) {
    if (map_first)
        TEST_DEF.put_ptr(state, context, "collision_map", map);
#if !defined(TEST_ROUTE_TOPOLOGY_NH_PVP)
    TEST_DEF.put_int(state, context, "world_offset_x", TEST_OFFSET_X);
    TEST_DEF.put_int(state, context, "world_offset_y", TEST_OFFSET_Y);
#endif
    if (!map_first)
        TEST_DEF.put_ptr(state, context, "collision_map", map);
}

static const EncounterArenaTopology* finalize_with_map(
    CollisionMap* map,
    int map_first
) {
    TestContext* context = calloc(1, sizeof(*context));
    if (!context) abort();
    TEST_DEF.init_context((EncounterContext*)context);
    if (!TEST_DEF.init_state) {
        fprintf(stderr, "%s has no embedded-state initializer\n", TEST_DEF.name);
        abort();
    }
    EncounterState* state = TEST_DEF.create();
    if (!state) abort();
    put_geometry(state, (EncounterContext*)context, map, map_first);
    if (TEST_TOPOLOGY(context) != NULL) {
        fprintf(stderr, "%s topology initialized before explicit finalization\n",
            TEST_DEF.name);
        abort();
    }
    TEST_DEF.finalize_context(state, (EncounterContext*)context);
    const EncounterArenaTopology* topology = TEST_TOPOLOGY(context);
    if (!topology || !topology->finalized) abort();
    TEST_DEF.destroy(state);
    free(context);
    return topology;
}

int main(void) {
    CollisionMap* first_map = collision_map_load(TEST_MAP_PATH);
    CollisionMap* second_map = collision_map_load(TEST_MAP_PATH);
    if (!first_map || !second_map || first_map == second_map) abort();

    const EncounterArenaTopology* first = finalize_with_map(first_map, 1);
    int open_tiles = 0;
    int blocked_tiles = 0;
    for (int x = TEST_ORIGIN_X; x < TEST_ORIGIN_X + TEST_WIDTH; x++) {
        for (int y = TEST_ORIGIN_Y; y < TEST_ORIGIN_Y + TEST_HEIGHT; y++) {
            int index = (x - TEST_ORIGIN_X) * TEST_HEIGHT +
                (y - TEST_ORIGIN_Y);
            uint32_t expected = expected_flags(first_map, x, y);
            if (first->static_collision_flags[index] != expected) {
                fprintf(stderr,
                    "%s topology mismatch at (%d,%d): expected %u got %u\n",
                    TEST_DEF.name,
                    x,
                    y,
                    expected,
                    first->static_collision_flags[index]);
                abort();
            }
            if (first->static_blocked[index]) blocked_tiles++;
            else open_tiles++;
        }
    }
#if defined(TEST_ROUTE_TOPOLOGY_INFERNO)
    if (first->static_los_mode != ENCOUNTER_ARENA_TOPOLOGY_LOS_OPEN) {
        fprintf(stderr,
            "inferno collision map incorrectly contributed static LOS blockers\n");
        abort();
    }
    int blocked_x = -1;
    int blocked_y = -1;
    int open_x = -1;
    int open_y = -1;
    for (int x = TEST_ORIGIN_X; x < TEST_ORIGIN_X + TEST_WIDTH; x++) {
        for (int y = TEST_ORIGIN_Y; y < TEST_ORIGIN_Y + TEST_HEIGHT; y++) {
            int index = (x - TEST_ORIGIN_X) * TEST_HEIGHT +
                (y - TEST_ORIGIN_Y);
            if (first->static_blocked[index] && blocked_x < 0) {
                blocked_x = x;
                blocked_y = y;
            } else if (!first->static_blocked[index] && open_x < 0) {
                open_x = x;
                open_y = y;
            }
        }
    }
    if (blocked_x < 0 || open_x < 0 ||
            !encounter_arena_topology_los_clear(
                first, blocked_x, blocked_y, 1, open_x, open_y, 1, 0) ||
            !encounter_arena_topology_los_clear(
                first, open_x, open_y, 1, blocked_x, blocked_y, 1, 0)) {
        fprintf(stderr,
            "inferno movement collision flags leaked into static LOS queries\n");
        abort();
    }
#endif
#if defined(TEST_ROUTE_TOPOLOGY_ZULRAH)
    if (open_tiles != 69 || blocked_tiles != 715) {
        fprintf(stderr, "zulrah topology count mismatch: %d open %d blocked\n",
            open_tiles, blocked_tiles);
        abort();
    }
#endif

    const EncounterArenaTopology* second = finalize_with_map(second_map, 0);
    if (second != first) {
        fprintf(stderr, "%s did not reuse process topology\n", TEST_DEF.name);
        abort();
    }
    printf("%s route topology setup PASS: %d open %d blocked, order-independent, identical-map reuse\n",
        TEST_DEF.name, open_tiles, blocked_tiles);
    collision_map_free(first_map);
    collision_map_free(second_map);
    return 0;
}
