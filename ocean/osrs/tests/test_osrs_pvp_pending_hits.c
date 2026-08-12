#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "ocean/osrs/encounters/encounter_nh_pvp.h"


#include "ocean/osrs/tests/osrs_test_check.h"

static void assert_child_aborts(const char* label, void (*fn)(void)) {
    fflush(NULL);
    pid_t pid = fork();
    if (pid == 0) {
        fn();
        _exit(0);
    }

    int status = 0;
    waitpid(pid, &status, 0);
    tests_run++;
    if (WIFSIGNALED(status) || (WIFEXITED(status) && WEXITSTATUS(status) != 0)) {
        tests_passed++;
    } else {
        tests_failed++;
        printf("  FAIL: %s - child returned successfully\n", label);
    }
}

static void queue_test_hit(Player* attacker, Player* defender, int damage) {
    queue_hit(123, 0, 1, attacker, defender, damage, ATTACK_STYLE_MAGIC,
        4, 0, damage > 0, 0, 0, 0, 0, 0);
}

static void test_pvp_queue_accepts_capacity(void) {
    printf("--- pvp queue accepts capacity ---\n");

    Player attacker = {0};
    Player defender = {0};
    defender.prayer = PRAYER_NONE;

    for (int i = 0; i < MAX_PENDING_HITS; i++)
        queue_test_hit(&attacker, &defender, i + 1);

    ASSERT_INT_EQ("queue reaches capacity", attacker.num_pending_hits, MAX_PENDING_HITS);
    ASSERT_INT_EQ("first damage kept", attacker.pending_hits[0].damage, 1);
    ASSERT_INT_EQ("last damage kept", attacker.pending_hits[MAX_PENDING_HITS - 1].damage,
        MAX_PENDING_HITS);
}

static void child_pvp_queue_overflow(void) {
    Player attacker = {0};
    Player defender = {0};
    defender.prayer = PRAYER_NONE;

    for (int i = 0; i < MAX_PENDING_HITS; i++)
        queue_test_hit(&attacker, &defender, i + 1);
    queue_test_hit(&attacker, &defender, 99);
}

static void test_pvp_queue_overflow_aborts(void) {
    printf("--- pvp queue overflow aborts ---\n");

    assert_child_aborts("pvp pending-hit overflow aborts", child_pvp_queue_overflow);
}

static void test_pvp_remove_compacts_and_clears_tail(void) {
    printf("--- pvp remove compacts and clears tail ---\n");

    Player attacker = {0};
    Player defender = {0};
    defender.prayer = PRAYER_NONE;
    queue_test_hit(&attacker, &defender, 3);
    queue_test_hit(&attacker, &defender, 7);
    queue_test_hit(&attacker, &defender, 11);

    pvp_remove_pending_hit(&attacker, 1);

    ASSERT_INT_EQ("queue count after remove", attacker.num_pending_hits, 2);
    ASSERT_INT_EQ("first hit remains", attacker.pending_hits[0].damage, 3);
    ASSERT_INT_EQ("third hit compacted", attacker.pending_hits[1].damage, 11);
    ASSERT_INT_EQ("tail damage cleared", attacker.pending_hits[2].damage, 0);
    ASSERT_INT_EQ("tail timer cleared", attacker.pending_hits[2].ticks_until_hit, 0);
}

static CollisionMap* nh_map;
static NhPvpContext nh_context;
static NhPvpState* nh_state;

static void init_nh_fixture(void) {
    EncounterState* state = ENCOUNTER_NH_PVP.create();
    if (!state) abort();
    nh_state = (NhPvpState*)state;
    ENCOUNTER_NH_PVP.init_context((EncounterContext*)&nh_context);
    nh_map = collision_map_load("ocean/osrs/data/wilderness.cmap");
    if (!nh_map) abort();
    ENCOUNTER_NH_PVP.put_ptr(
        state, (EncounterContext*)&nh_context, "collision_map", nh_map);
    ENCOUNTER_NH_PVP.put_int(
        state, (EncounterContext*)&nh_context, "seed", 1);
    ENCOUNTER_NH_PVP.finalize_context(
        state, (EncounterContext*)&nh_context);
    ENCOUNTER_NH_PVP.reset(
        state, (EncounterContext*)&nh_context, 1);
    nh_state->env.auto_reset = 0;
}

static void test_nh_topology_geometry_parity(void) {
    printf("--- NH PvP topology geometry parity ---\n");
    const EncounterArenaTopology* topology = nh_context.route_topology;
    CHECK("NH PvP topology covers the 61 by 28 fight arena",
        topology->origin_x == FIGHT_AREA_BASE_X &&
        topology->origin_y == FIGHT_AREA_BASE_Y &&
        topology->width == 61 && topology->height == 28);
    CHECK("NH PvP keeps the existing open static LOS combat rule",
        topology->static_los_mode == ENCOUNTER_ARENA_TOPOLOGY_LOS_OPEN);

    int diagonal_wall_steps = 0;
    for (int x = FIGHT_AREA_BASE_X;
            x < FIGHT_AREA_BASE_X + FIGHT_AREA_WIDTH;
            x++) {
        for (int y = FIGHT_AREA_BASE_Y;
                y < FIGHT_AREA_BASE_Y + FIGHT_AREA_HEIGHT;
                y++) {
            int expected_walkable =
                is_in_wilderness(x, y) &&
                collision_tile_walkable(nh_map, 0, x, y);
            CHECK("NH PvP static player tile parity",
                pvp_topology_tile_walkable(topology, x, y) ==
                    expected_walkable);
            if (!expected_walkable) continue;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int destination_x = x + dx;
                    int destination_y = y + dy;
                    int expected_step =
                        encounter_arena_topology_contains(
                            topology, destination_x, destination_y) &&
                        collision_traversable_step(
                            nh_map, 0, x, y, dx, dy);
                    int actual_step =
                        encounter_arena_topology_step_allowed(
                            topology, x, y, 1, dx, dy);
                    CHECK("NH PvP cardinal and diagonal wall parity",
                        actual_step == expected_step);
                    if (dx != 0 && dy != 0 &&
                            !expected_step &&
                            pvp_topology_tile_walkable(
                                topology, x + dx, y + dy))
                        diagonal_wall_steps++;
                }
            }
        }
    }
    CHECK("NH PvP collision map contains diagonal wall-side blocking",
        diagonal_wall_steps > 0);
}

static void test_nh_dynamic_player_occupancy(void) {
    printf("--- NH PvP player occupancy stays dynamic ---\n");
    const EncounterArenaTopology* topology = nh_context.route_topology;
    Player mover = {0};
    Player blocker = {0};
    mover.x = blocker.x = FIGHT_AREA_BASE_X + 8;
    mover.y = blocker.y = FIGHT_AREA_BASE_Y + 8;
    CHECK("opponent tile is not baked into static topology",
        pvp_topology_tile_walkable(topology, blocker.x, blocker.y));
    resolve_same_tile(&mover, &blocker, topology);
    ASSERT_INT_EQ("same-tile resolution keeps west-first priority",
        mover.x, blocker.x - 1);
    ASSERT_INT_EQ("same-tile resolution keeps the y coordinate",
        mover.y, blocker.y);
    CHECK("static topology remains unchanged after occupancy resolution",
        pvp_topology_tile_walkable(topology, blocker.x, blocker.y));
}

static void run_nh_chase_case(
    int loadout,
    int combat,
    int expected_x,
    int expected_y,
    const char* x_label,
    const char* y_label
) {
    ENCOUNTER_NH_PVP.reset(
        (EncounterState*)nh_state,
        (EncounterContext*)&nh_context,
        1);
    nh_state->env.auto_reset = 0;
    Player* player = &nh_state->env.players[0];
    Player* target = &nh_state->env.players[1];
    player->x = FIGHT_AREA_BASE_X;
    player->y = FIGHT_AREA_BASE_Y;
    player->dest_x = player->x;
    player->dest_y = player->y;
    target->x = FIGHT_AREA_BASE_X + 10;
    target->y = FIGHT_AREA_BASE_Y + 4;
    target->dest_x = target->x;
    target->dest_y = target->y;
    int actions[NUM_ACTION_HEADS] = {0};
    actions[HEAD_LOADOUT] = loadout;
    actions[HEAD_COMBAT] = combat;
    ENCOUNTER_NH_PVP.step(
        (EncounterState*)nh_state,
        (EncounterContext*)&nh_context,
        actions);
    ASSERT_INT_EQ(x_label, player->x, expected_x);
    ASSERT_INT_EQ(y_label, player->y, expected_y);
    ASSERT_INT_EQ("attack click cancels queued walk x",
        nh_state->env.pvp_runtime.walk_dest_x[0], -1);
    ASSERT_INT_EQ("attack click cancels queued walk y",
        nh_state->env.pvp_runtime.walk_dest_y[0], -1);
}

static void test_nh_attack_chase_destinations(void) {
    printf("--- NH PvP attack chase destinations ---\n");
    run_nh_chase_case(
        LOADOUT_MELEE, ATTACK_ATK,
        FIGHT_AREA_BASE_X + 2, FIGHT_AREA_BASE_Y + 2,
        "melee chase keeps x destination",
        "melee chase keeps y destination");
    run_nh_chase_case(
        LOADOUT_RANGE, ATTACK_ATK,
        FIGHT_AREA_BASE_X + 2, FIGHT_AREA_BASE_Y,
        "ranged chase keeps x destination",
        "ranged chase keeps y destination");
    run_nh_chase_case(
        LOADOUT_MAGE, ATTACK_ICE,
        FIGHT_AREA_BASE_X, FIGHT_AREA_BASE_Y,
        "in-range magic attack does not chase on x",
        "in-range magic attack does not chase on y");
}

static void test_nh_out_of_arena_destination_fallback(void) {
    printf("--- NH PvP out-of-arena destination fallback ---\n");
    Player player = {0};
    player.x = FIGHT_AREA_BASE_X + 1;
    player.y = FIGHT_AREA_BASE_Y + 1;
    int destination_x = -1;
    int destination_y = -1;
    CHECK("farcast click beyond the arena keeps a route fallback target",
        select_farcast_tile(
            &player,
            FIGHT_AREA_BASE_X + 5,
            FIGHT_AREA_BASE_Y,
            6,
            &destination_x,
            &destination_y,
            nh_context.route_topology));
    ASSERT_INT_EQ("farcast keeps the original x destination",
        destination_x, FIGHT_AREA_BASE_X - 1);
    ASSERT_INT_EQ("farcast keeps the original y destination",
        destination_y, FIGHT_AREA_BASE_Y + 1);
    EncounterRouteInput route_input = {
        .topology = nh_context.route_topology,
        .source_x = player.x,
        .source_y = player.y,
        .actor_size = 1,
        .target_x = destination_x,
        .target_y = destination_y,
        .target_size = 1,
        .target_kind = ENCOUNTER_ROUTE_TARGET_TILE,
        .movement_mode = ENCOUNTER_ROUTE_MOVEMENT_RUN,
        .cost_policy = ENCOUNTER_ROUTE_COST_SOUTH_FIRST_BFS,
    };
    EncounterRouteResult route = encounter_route_solve(&route_input);
    ASSERT_INT_EQ("out-of-arena route reaches the arena edge",
        route.outcome, ROUTE_REACHED_FALLBACK);
    ASSERT_INT_EQ("out-of-arena route clamps x to the arena edge",
        route.destination_x, FIGHT_AREA_BASE_X);
    ASSERT_INT_EQ("out-of-arena route keeps the selected y",
        route.destination_y, FIGHT_AREA_BASE_Y + 1);
    ASSERT_INT_EQ("out-of-arena route steps toward the edge on x",
        route.first_dx, -1);
    ASSERT_INT_EQ("out-of-arena route keeps y on the first step",
        route.first_dy, 0);

    route_input.source_x = FIGHT_AREA_BASE_X + 19;
    route_input.source_y = FIGHT_AREA_BASE_Y;
    route_input.target_x = route_input.source_x;
    route_input.target_y = FIGHT_AREA_BASE_Y - 1;
    route = encounter_route_solve(&route_input);
    ASSERT_INT_EQ("boundary source reaches the clamped destination",
        route.outcome, ROUTE_REACHED_FALLBACK);
    ASSERT_INT_EQ("boundary source keeps x",
        route.destination_x, route_input.source_x);
    ASSERT_INT_EQ("boundary source does not route across the arena",
        route.destination_y, route_input.source_y);
    ASSERT_INT_EQ("boundary source has zero route distance",
        route.distance, 0);
}

int main(void) {
    init_nh_fixture();
    test_nh_topology_geometry_parity();
    test_nh_dynamic_player_occupancy();
    test_nh_attack_chase_destinations();
    test_nh_out_of_arena_destination_fallback();

    test_pvp_queue_accepts_capacity();
    test_pvp_queue_overflow_aborts();
    test_pvp_remove_compacts_and_clears_tail();

    return osrs_test_summary();
}
