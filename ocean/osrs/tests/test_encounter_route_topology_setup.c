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
typedef struct {
    int count;
    int x[16];
    int y[16];
} NhPvpTestBlockers;

static int nh_pvp_test_blocked(void* data, int x, int y, int size) {
    (void)size;
    const NhPvpTestBlockers* blockers = (const NhPvpTestBlockers*)data;
    for (int i = 0; i < blockers->count; i++)
        if (blockers->x[i] == x && blockers->y[i] == y) return 1;
    return 0;
}

static int nh_pvp_route_results_equal(
    const EncounterRouteResult* source_field,
    const EncounterRouteResult* target_bfs
) {
    if (source_field->outcome != target_bfs->outcome ||
            source_field->destination_x != target_bfs->destination_x ||
            source_field->destination_y != target_bfs->destination_y ||
            source_field->first_dx != target_bfs->first_dx ||
            source_field->first_dy != target_bfs->first_dy ||
            source_field->run_dx != target_bfs->run_dx ||
            source_field->run_dy != target_bfs->run_dy ||
            source_field->distance != target_bfs->distance ||
            source_field->waypoint_count != target_bfs->waypoint_count)
        return 0;
    for (int i = 0; i < source_field->waypoint_count; i++) {
        if (source_field->waypoint_x[i] != target_bfs->waypoint_x[i] ||
                source_field->waypoint_y[i] != target_bfs->waypoint_y[i])
            return 0;
    }
    return 1;
}

static uint64_t nh_pvp_target_bfs_equivalence(
    const EncounterArenaTopology* topology,
    NhPvpTestBlockers* blockers,
    uint64_t blocker_revision
) {
    uint64_t checks = 0;
    EncounterRouteInput input = {
        .topology = topology,
        .blockers = {
            .is_blocked = blockers->count ? nh_pvp_test_blocked : NULL,
            .ctx = blockers,
            .revision = blocker_revision,
        },
        .actor_size = 1,
        .target_size = 1,
        .target_kind = ENCOUNTER_ROUTE_TARGET_CARDINAL_ADJACENCY,
        .movement_mode = ENCOUNTER_ROUTE_MOVEMENT_RUN,
    };
    for (int source_x = topology->origin_x;
            source_x < topology->origin_x + topology->width;
            source_x++) {
        for (int source_y = topology->origin_y;
                source_y < topology->origin_y + topology->height;
                source_y++) {
            if (encounter_arena_topology_footprint_blocked(
                    topology, source_x, source_y, 1) ||
                    nh_pvp_test_blocked(blockers, source_x, source_y, 1))
                continue;
            input.source_x = source_x;
            input.source_y = source_y;
            for (int target_x = topology->origin_x;
                    target_x < topology->origin_x + topology->width;
                    target_x++) {
                for (int target_y = topology->origin_y;
                        target_y < topology->origin_y + topology->height;
                        target_y++) {
                    if (encounter_arena_topology_footprint_blocked(
                            topology, target_x, target_y, 1))
                        continue;
                    input.target_x = target_x;
                    input.target_y = target_y;
                    input.cost_policy = ENCOUNTER_ROUTE_COST_OSRS;
                    EncounterRouteResult source_field =
                        encounter_route_solve(&input);
                    input.cost_policy = ENCOUNTER_ROUTE_COST_OSRS_TARGET_BFS;
                    EncounterRouteResult target_bfs =
                        encounter_route_solve(&input);
                    if (!nh_pvp_route_results_equal(&source_field, &target_bfs)) {
                        fprintf(
                            stderr,
                            "nh_pvp target BFS mismatch source=(%d,%d) target=(%d,%d) blockers=%d\n",
                            source_x,
                            source_y,
                            target_x,
                            target_y,
                            blockers->count);
                        abort();
                    }
                    checks++;
                }
            }
        }
    }
    return checks;
}

static void nh_pvp_target_bfs_selftest(
    const EncounterArenaTopology* topology
) {
    NhPvpTestBlockers blockers = {0};
    uint64_t checks =
        nh_pvp_target_bfs_equivalence(topology, &blockers, 1);
    blockers = (NhPvpTestBlockers){
        .count = 8,
        .x = {
            FIGHT_AREA_BASE_X + 17,
            FIGHT_AREA_BASE_X + 17,
            FIGHT_AREA_BASE_X + 17,
            FIGHT_AREA_BASE_X + 17,
            FIGHT_AREA_BASE_X + 29,
            FIGHT_AREA_BASE_X + 29,
            FIGHT_AREA_BASE_X + 30,
            FIGHT_AREA_BASE_X + 30,
        },
        .y = {
            FIGHT_AREA_BASE_Y + 8,
            FIGHT_AREA_BASE_Y + 9,
            FIGHT_AREA_BASE_Y + 11,
            FIGHT_AREA_BASE_Y + 12,
            FIGHT_AREA_BASE_Y + 18,
            FIGHT_AREA_BASE_Y + 19,
            FIGHT_AREA_BASE_Y + 18,
            FIGHT_AREA_BASE_Y + 19,
        },
    };
    checks += nh_pvp_target_bfs_equivalence(topology, &blockers, 2);
    printf(
        "nh_pvp target-directed BFS equivalence PASS: %llu source-target queries across 2 blocker fields\n",
        (unsigned long long)checks);
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
#if defined(TEST_ROUTE_TOPOLOGY_NH_PVP)
    nh_pvp_target_bfs_selftest(first);
#endif
    printf("%s route topology setup PASS: %d open %d blocked, order-independent, identical-map reuse\n",
        TEST_DEF.name, open_tiles, blocked_tiles);
    collision_map_free(first_map);
    collision_map_free(second_map);
    return 0;
}
