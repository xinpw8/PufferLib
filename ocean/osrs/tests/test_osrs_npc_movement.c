#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "ocean/osrs/osrs_encounter.h"

typedef struct {
    uint8_t blocked[16][16];
    int hold_overlap;
} TestNpcMoveGrid;

#include "ocean/osrs/tests/osrs_test_check.h"

static int test_npc_move_blocked(void* ctx, int x, int y, int size) {
    TestNpcMoveGrid* grid = (TestNpcMoveGrid*)ctx;
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int tx = x + dx;
            int ty = y + dy;
            if (tx < 0 || tx >= 16 || ty < 0 || ty >= 16) return 1;
            if (grid->blocked[tx][ty]) return 1;
        }
    }
    return 0;
}

static int test_npc_move_hold_overlap(void* ctx) {
    TestNpcMoveGrid* grid = (TestNpcMoveGrid*)ctx;
    return grid->hold_overlap;
}

static void test_shared_diagonal_edge_clearance_size_one(void) {
    printf("test_shared_diagonal_edge_clearance_size_one\n");

    TestNpcMoveGrid grid;
    memset(&grid, 0, sizeof(grid));
    grid.blocked[6][5] = 1;
    int x = 5;
    int y = 5;
    int moved = encounter_npc_step_toward_policy(
        &x, &y, 7, 7, 1, 1, ENCOUNTER_NPC_STEP_TRAVEL_TARGET,
        test_npc_move_blocked, &grid, NULL, NULL);
    CHECK("east side block prevents 1x1 diagonal corner cut",
        moved == 1 && x == 5 && y == 6);

    memset(&grid, 0, sizeof(grid));
    grid.blocked[5][6] = 1;
    x = 5;
    y = 5;
    moved = encounter_npc_step_toward_policy(
        &x, &y, 7, 7, 1, 1, ENCOUNTER_NPC_STEP_TRAVEL_TARGET,
        test_npc_move_blocked, &grid, NULL, NULL);
    CHECK("north side block prevents 1x1 diagonal corner cut",
        moved == 1 && x == 6 && y == 5);

    memset(&grid, 0, sizeof(grid));
    x = 5;
    y = 5;
    moved = encounter_npc_step_toward_policy(
        &x, &y, 7, 7, 1, 1, ENCOUNTER_NPC_STEP_TRAVEL_TARGET,
        test_npc_move_blocked, &grid, NULL, NULL);
    CHECK("clear side edges allow 1x1 diagonal move",
        moved == 1 && x == 6 && y == 6);
}

static void test_shared_aggro_target_overlap_rewrite(void) {
    printf("test_shared_aggro_target_overlap_rewrite\n");

    TestNpcMoveGrid grid;
    memset(&grid, 0, sizeof(grid));
    uint32_t rng = 0x12345678u;
    int x = 5;
    int y = 5;
    int moved = encounter_npc_step_toward_policy(
        &x, &y, 6, 6, 1, 1, ENCOUNTER_NPC_STEP_OSRS_AGGRO_TARGET,
        test_npc_move_blocked, &grid, NULL, &rng);
    CHECK("proposed diagonal into target overlap rewrites to x-only",
        moved == 1 && x == 6 && y == 5);

    memset(&grid, 0, sizeof(grid));
    grid.blocked[6][5] = 1;
    rng = 0x12345678u;
    x = 5;
    y = 5;
    moved = encounter_npc_step_toward_policy(
        &x, &y, 6, 6, 1, 1, ENCOUNTER_NPC_STEP_OSRS_AGGRO_TARGET,
        test_npc_move_blocked, &grid, NULL, &rng);
    CHECK("blocked x-only rewrite holds and does not try y-only",
        moved == 0 && x == 5 && y == 5);
}

static void test_shared_melee_policy(void) {
    printf("test_shared_melee_policy\n");

    TestNpcMoveGrid grid;
    memset(&grid, 0, sizeof(grid));
    int x = 5;
    int y = 5;
    int moved = encounter_npc_step_toward_policy(
        &x, &y, 6, 5, 1, 1, ENCOUNTER_NPC_STEP_STOP_AT_MELEE,
        test_npc_move_blocked, &grid, NULL, NULL);
    CHECK("cardinal melee contact with stop-at-melee holds",
        moved == 0 && x == 5 && y == 5);

    x = 5;
    y = 5;
    moved = encounter_npc_step_toward_policy(
        &x, &y, 6, 6, 1, 1, ENCOUNTER_NPC_STEP_STOP_AT_MELEE,
        test_npc_move_blocked, &grid, NULL, NULL);
    CHECK("diagonal melee contact with stop-at-melee tries x-only",
        moved == 1 && x == 6 && y == 5);
}

static void test_shared_current_overlap(void) {
    printf("test_shared_current_overlap\n");

    TestNpcMoveGrid grid;
    memset(&grid, 0, sizeof(grid));
    uint32_t rng = 0x12345678u;
    int x = 5;
    int y = 5;
    int moved = encounter_npc_step_toward_policy(
        &x, &y, 5, 5, 1, 1, ENCOUNTER_NPC_STEP_OSRS_AGGRO_TARGET,
        test_npc_move_blocked, &grid, NULL, &rng);
    int dist = (x > 5 ? x - 5 : 5 - x) + (y > 5 ? y - 5 : 5 - y);
    CHECK("current overlap shuffles one cardinal tile",
        moved == 1 && dist == 1);

    memset(&grid, 0, sizeof(grid));
    grid.blocked[4][5] = 1;
    rng = 12345u;
    x = 5;
    y = 5;
    moved = encounter_npc_step_toward_policy(
        &x, &y, 5, 5, 1, 1, ENCOUNTER_NPC_STEP_OSRS_AGGRO_TARGET,
        test_npc_move_blocked, &grid, NULL, &rng);
    CHECK("blocked sampled overlap shuffle stays under target",
        moved == 0 && x == 5 && y == 5);

    memset(&grid, 0, sizeof(grid));
    grid.hold_overlap = 1;
    rng = 0x12345678u;
    x = 5;
    y = 5;
    moved = encounter_npc_step_toward_policy(
        &x, &y, 5, 5, 1, 1, ENCOUNTER_NPC_STEP_OSRS_AGGRO_TARGET,
        test_npc_move_blocked, &grid, test_npc_move_hold_overlap, &rng);
    CHECK("current overlap plus just-clicked target holds current tile",
        moved == 0 && x == 5 && y == 5);
}

typedef struct {
    uint32_t flags[12][12];
} NpcTopologyGeometry;

static uint32_t npc_topology_flags(void* ctx, int x, int y) {
    const NpcTopologyGeometry* geometry = (const NpcTopologyGeometry*)ctx;
    if (x < 0 || x >= 12 || y < 0 || y >= 12)
        return COLLISION_BLOCKED | LOS_FULL_MASK;
    return geometry->flags[x][y];
}

static int npc_topology_aborts(void (*operation)(void)) {
    fflush(NULL);
    pid_t pid = fork();
    if (pid == 0) {
        operation();
        _exit(0);
    }
    int status = 0;
    if (pid < 0 || waitpid(pid, &status, 0) != pid) return 0;
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

static void npc_topology_rejects_unsupported_footprint(void) {
    NpcTopologyGeometry geometry;
    memset(&geometry, 0, sizeof(geometry));
    EncounterArenaTopologyBuildSpec spec = {
        .origin_x = 0,
        .origin_y = 0,
        .width = 12,
        .height = 12,
        .max_footprint_size =
            ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE + 1,
        .revision = 9,
        .tile_flags = npc_topology_flags,
        .tile_flags_ctx = &geometry,
    };
    (void)encounter_arena_topology_build(&spec);
}

static void test_arena_topology_footprint_masks(void) {
    printf("test_arena_topology_footprint_masks\n");

    NpcTopologyGeometry geometry;
    memset(&geometry, 0, sizeof(geometry));
    geometry.flags[8][8] = COLLISION_BLOCKED;
    EncounterArenaTopologyBuildSpec spec = {
        .origin_x = 0,
        .origin_y = 0,
        .width = 12,
        .height = 12,
        .max_footprint_size =
            ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE,
        .revision = 9,
        .tile_flags = npc_topology_flags,
        .tile_flags_ctx = &geometry,
    };
    EncounterArenaTopology* topology = encounter_arena_topology_build(&spec);
    encounter_arena_topology_finalize(topology);

    for (int size = 1;
            size <= ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE;
            size++) {
        char label[96];
        snprintf(label, sizeof(label),
            "size %d mask rejects footprint covering static blocker", size);
        CHECK(label, encounter_arena_topology_footprint_blocked(
            topology, 9 - size, 9 - size, size));

        snprintf(label, sizeof(label),
            "size %d mask accepts clear in-bounds footprint", size);
        CHECK(label, !encounter_arena_topology_footprint_blocked(
            topology, 1, 1, size));

        snprintf(label, sizeof(label),
            "size %d mask rejects footprint crossing arena edge", size);
        CHECK(label, encounter_arena_topology_footprint_blocked(
            topology, 13 - size, 1, size));
    }

    CHECK("large footprint clear cardinal step uses prebuilt mask",
        encounter_arena_topology_step_allowed(topology, 1, 1, 5, 1, 0));
    CHECK("large footprint clear diagonal step uses prebuilt mask",
        encounter_arena_topology_step_allowed(topology, 1, 1, 5, 1, 1));
    CHECK("large footprint step rejects destination static blocker",
        !encounter_arena_topology_step_allowed(topology, 3, 3, 5, 1, 1));

    free(topology);

    CHECK("topology build rejects unsupported footprint size",
        npc_topology_aborts(npc_topology_rejects_unsupported_footprint));
}

int main(void) {
    test_arena_topology_footprint_masks();
    test_shared_diagonal_edge_clearance_size_one();
    test_shared_aggro_target_overlap_rewrite();
    test_shared_melee_policy();
    test_shared_current_overlap();

    return osrs_test_summary();
}
