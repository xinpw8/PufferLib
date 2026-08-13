#include <stdio.h>
#include <string.h>

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

int main(void) {
    test_shared_diagonal_edge_clearance_size_one();
    test_shared_aggro_target_overlap_rewrite();
    test_shared_melee_policy();
    test_shared_current_overlap();

    return osrs_test_summary();
}
