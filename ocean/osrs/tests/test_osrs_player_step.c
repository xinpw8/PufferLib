/**
 * @file test_osrs_player_step.c
 * @brief Shared player-step command contract: one click per tick.
 *
 * The engine cannot both start an attack and walk to a chosen tile on the same
 * tick. A ground click cancels the entity interaction; an entity click cancels
 * the walk. These pin that contract at the shared layer, where every encounter
 * now routes through OsrsPlayerCommand.
 */

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static size_t topology_allocation_count;
static void* topology_counted_calloc(size_t count, size_t size);

#define calloc topology_counted_calloc
#include "ocean/osrs/osrs_encounter_player.h"
#undef calloc

static void* topology_counted_calloc(size_t count, size_t size) {
    topology_allocation_count++;
    return calloc(count, size);
}
static int tests_run = 0;
static int tests_failed = 0;

#define CHECK(label, cond) do { \
    tests_run++; \
    if (!(cond)) { \
        tests_failed++; \
        printf("  FAIL: %s\n", (label)); \
    } \
} while (0)

#define STEP_GRID 24

typedef struct {
    OsrsAttackTarget target;
    int target_valid;
} StepTargetCtx;
typedef struct {
    int calls;
} StepWalkableCountCtx;


static int step_tile_walkable(void* ctx, int x, int y) {
    (void)ctx;
    return x >= 0 && x < STEP_GRID && y >= 0 && y < STEP_GRID;
}
static int step_counted_tile_walkable(void* ctx, int x, int y) {
    StepWalkableCountCtx* count = (StepWalkableCountCtx*)ctx;
    count->calls++;
    return x >= 0 && x < STEP_GRID && y >= 0 && y < STEP_GRID;
}


static int step_vertical_wall(void* ctx, int x, int y) {
    (void)ctx;
    return x == 5 && y <= 4;
}

static int step_lookup_target(void* ctx, int target_slot, OsrsAttackTarget* out) {
    StepTargetCtx* c = (StepTargetCtx*)ctx;
    if (!c->target_valid || target_slot != c->target.slot) return 0;
    *out = c->target;
    return 1;
}

static OsrsEncounterArena step_arena(void) {
    OsrsEncounterArena arena;
    memset(&arena, 0, sizeof(arena));
    arena.is_walkable = step_tile_walkable;
    arena.los_query = osrs_los_open_query();
    arena.arena_w = STEP_GRID;
    arena.arena_h = STEP_GRID;
    return arena;
}

static void test_attack_route_rejects_cardinal_adjacency_through_wall(void) {
    CollisionMap map;
    CollisionRegion region;
    collision_map_init(&map);
    memset(&region, 0, sizeof(region));
    collision_map_put(&map, collision_region_hash(9, 10), &region);
    region.flags[0][9][10] = COLLISION_WALL_EAST;
    region.flags[0][10][10] = COLLISION_WALL_WEST;

    const OsrsLosQuery* los = osrs_los_open_query();
    PathResult route = encounter_pathfind_arena_attack_approach(
        &map, 0, 0,
        9, 10,
        10, 10, 1, 1,
        step_tile_walkable, NULL,
        NULL, NULL,
        los,
        0, 0, STEP_GRID, STEP_GRID);

    CHECK("wall-adjacent route moves around the wall",
        route.found && (route.next_dx != 0 || route.next_dy != 0));
}

static void test_attack_route_fallback_stays_within_ten_tiles(void) {
    const OsrsLosQuery* los = osrs_los_open_query();
    PathResult route = encounter_pathfind_arena_attack_approach(
        NULL, 0, 0,
        10, 10,
        -20, 10, 1, 1,
        step_tile_walkable, NULL,
        NULL, NULL,
        los,
        0, 0, STEP_GRID, STEP_GRID);

    CHECK("fallback outside the ten-tile target radius fails", !route.found);
}

static void test_attack_route_uses_rsmod_cardinal_first_order(void) {
    PathResult route = encounter_pathfind_arena_attack_approach(
        NULL, 0, 0,
        10, 10,
        8, 8, 1, 1,
        step_tile_walkable, NULL,
        NULL, NULL,
        osrs_los_open_query(),
        0, 0, STEP_GRID, STEP_GRID);

    CHECK("RSMod route takes west before southwest",
        route.found &&
        route.next_dx == -1 && route.next_dy == 0 &&
        route.run_dx == -1 && route.run_dy == -1);
}

static void test_melee_attack_does_not_cross_cardinal_wall(void) {
    CollisionMap map;
    CollisionRegion region;
    collision_map_init(&map);
    memset(&region, 0, sizeof(region));
    collision_map_put(&map, collision_region_hash(9, 10), &region);
    region.flags[0][9][10] = COLLISION_WALL_EAST;
    region.flags[0][10][10] = COLLISION_WALL_WEST;

    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 9;
    player.y = 10;

    OsrsInteraction interaction;
    osrs_interaction_init(&interaction);
    osrs_interaction_set(&interaction, 3);

    StepTargetCtx tctx = {
        .target = { .slot = 3, .x = 10, .y = 10, .size = 1, .attack_range = 1 },
        .target_valid = 1,
    };
    int dest_x = -1;
    int dest_y = -1;
    OsrsEncounterArena arena = step_arena();
    arena.collision_map = &map;

    OsrsPlayerStepInput input;
    memset(&input, 0, sizeof(input));
    input.player = &player;
    input.interaction = &interaction;
    input.target_lookup = step_lookup_target;
    input.target_ctx = &tctx;
    input.dest_x = &dest_x;
    input.dest_y = &dest_y;
    input.arena = arena;

    OsrsPlayerStepResult result = osrs_encounter_player_step(&input);

    CHECK("melee does not attack through cardinal wall",
        !result.can_attack || player.x != 9 || player.y != 10);
    CHECK("melee routes around cardinal wall", result.chased_target);
}

/* an entity click cancels a walk already in flight: the player closes on the
   target instead of continuing to the clicked tile. this is the nh_pvp/zulrah
   regression — EXPLICIT_FIRST used to honour both, giving free damage while
   repositioning. */
static void test_target_command_cancels_walk_in_flight(void) {
    printf("--- entity click cancels a walk in flight ---\n");

    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 5; player.y = 5;
    player.run_energy = 10000;

    OsrsInteraction interaction;
    osrs_interaction_init(&interaction);

    StepTargetCtx tctx = {
        .target = { .slot = 3, .x = 5, .y = 12, .size = 1, .attack_range = 1 },
        .target_valid = 1,
    };

    int dest_x = 5, dest_y = 1;   /* walking south, target is north */
    OsrsEncounterArena arena = step_arena();

    OsrsPlayerStepInput input;
    memset(&input, 0, sizeof(input));
    input.player = &player;
    input.interaction = &interaction;
    input.target_lookup = step_lookup_target;
    input.target_ctx = &tctx;
    input.command.kind = OSRS_PLAYER_CMD_TARGET;
    input.command.target_slot = 3;
    input.dest_x = &dest_x;
    input.dest_y = &dest_y;
    input.arena = arena;

    OsrsPlayerStepResult r = osrs_encounter_player_step(&input);

    CHECK("interaction is set to the clicked entity", interaction.target_slot == 3);
    CHECK("walk destination is cancelled", dest_x == -1 && dest_y == -1);
    CHECK("no explicit move ran", r.explicit_moved == 0);
    CHECK("player closed on the target, not the clicked tile", player.y > 5);
}

/* a ground click cancels the interaction and walks. */
static void test_move_command_cancels_interaction(void) {
    printf("--- ground click cancels the interaction ---\n");

    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 5; player.y = 5;
    player.run_energy = 10000;

    OsrsInteraction interaction;
    osrs_interaction_init(&interaction);
    osrs_interaction_set(&interaction, 3);

    StepTargetCtx tctx = {
        .target = { .slot = 3, .x = 5, .y = 12, .size = 1, .attack_range = 1 },
        .target_valid = 1,
    };

    int dest_x = 5, dest_y = 1;
    OsrsEncounterArena arena = step_arena();

    OsrsPlayerStepInput input;
    memset(&input, 0, sizeof(input));
    input.player = &player;
    input.interaction = &interaction;
    input.target_lookup = step_lookup_target;
    input.target_ctx = &tctx;
    input.command.kind = OSRS_PLAYER_CMD_MOVE;
    input.command.move_kind = OSRS_PLAYER_MOVE_DESTINATION;
    input.dest_x = &dest_x;
    input.dest_y = &dest_y;
    input.arena = arena;

    OsrsPlayerStepResult r = osrs_encounter_player_step(&input);

    CHECK("interaction is cancelled", !osrs_interaction_active(&interaction));
    CHECK("explicit move ran", r.explicit_moved == 1);
    CHECK("player walked toward the clicked tile", player.y < 5);
}

/* no click: an active interaction keeps auto-chasing. */
static void test_none_command_chases_active_interaction(void) {
    printf("--- idle tick auto-chases the standing interaction ---\n");

    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 5; player.y = 5;
    player.run_energy = 10000;

    OsrsInteraction interaction;
    osrs_interaction_init(&interaction);
    osrs_interaction_set(&interaction, 3);

    StepTargetCtx tctx = {
        .target = { .slot = 3, .x = 5, .y = 12, .size = 1, .attack_range = 1 },
        .target_valid = 1,
    };

    int dest_x = -1, dest_y = -1;
    OsrsEncounterArena arena = step_arena();

    OsrsPlayerStepInput input;
    memset(&input, 0, sizeof(input));
    input.player = &player;
    input.interaction = &interaction;
    input.target_lookup = step_lookup_target;
    input.target_ctx = &tctx;
    input.command.kind = OSRS_PLAYER_CMD_NONE;
    input.dest_x = &dest_x;
    input.dest_y = &dest_y;
    input.arena = arena;

    OsrsPlayerStepResult r = osrs_encounter_player_step(&input);

    CHECK("interaction survives an idle tick", osrs_interaction_active(&interaction));
    CHECK("chase ran", r.chased_target == 1);
    CHECK("no explicit move ran", r.explicit_moved == 0);
}

static void test_attack_route_persists_and_invalidates_canonically(void) {
    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 2;
    player.y = 2;

    OsrsInteraction interaction;
    osrs_interaction_init(&interaction);
    osrs_interaction_set(&interaction, 3);

    StepTargetCtx target_ctx = {
        .target = { .slot = 3, .x = 10, .y = 2, .size = 1, .attack_range = 1 },
        .target_valid = 1,
    };
    int dest_x = -1;
    int dest_y = -1;
    OsrsEncounterArena arena = step_arena();
    arena.extra_blocked = step_vertical_wall;

    OsrsPlayerStepInput input;
    memset(&input, 0, sizeof(input));
    input.player = &player;
    input.interaction = &interaction;
    input.target_lookup = step_lookup_target;
    input.target_ctx = &target_ctx;
    input.dest_x = &dest_x;
    input.dest_y = &dest_y;
    input.arena = arena;

    OsrsPlayerStepResult first = osrs_encounter_player_step(&input);
    CHECK("obstacle chase builds a multi-checkpoint route",
        first.chased_target && interaction.route.waypoint_count > 1);
    CHECK("route records its original source",
        interaction.route.planned_source_x == 2 &&
        interaction.route.planned_source_y == 2);

    osrs_encounter_player_step(&input);
    CHECK("route persists while more than one checkpoint remains",
        interaction.route.planned_source_x == 2 &&
        interaction.route.planned_source_y == 2);

    target_ctx.target.y = 3;
    int moved_target_source_x = player.x;
    int moved_target_source_y = player.y;
    osrs_encounter_player_step(&input);
    CHECK("target geometry change reroutes from the current tile",
        interaction.route.target_y == 3 &&
        interaction.route.planned_source_x == moved_target_source_x &&
        interaction.route.planned_source_y == moved_target_source_y);

    player.x = 1;
    player.y = 10;
    osrs_encounter_player_step(&input);
    CHECK("player route divergence reroutes from the observed tile",
        interaction.route.planned_source_x == 1 &&
        interaction.route.planned_source_y == 10);
}

static void test_single_checkpoint_route_reroutes_each_tick(void) {
    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 2;
    player.y = 2;

    OsrsInteraction interaction;
    osrs_interaction_init(&interaction);
    osrs_interaction_set(&interaction, 3);

    StepTargetCtx target_ctx = {
        .target = { .slot = 3, .x = 12, .y = 2, .size = 1, .attack_range = 1 },
        .target_valid = 1,
    };
    int dest_x = -1;
    int dest_y = -1;
    OsrsEncounterArena arena = step_arena();
    OsrsPlayerStepInput input;
    memset(&input, 0, sizeof(input));
    input.player = &player;
    input.interaction = &interaction;
    input.target_lookup = step_lookup_target;
    input.target_ctx = &target_ctx;
    input.dest_x = &dest_x;
    input.dest_y = &dest_y;
    input.arena = arena;

    osrs_encounter_player_step(&input);
    CHECK("straight chase compresses to one checkpoint",
        interaction.route.waypoint_count == 1 &&
        interaction.route.planned_source_x == 2);
    osrs_encounter_player_step(&input);
    CHECK("one-checkpoint entity route reroutes from the current tile",
        interaction.route.planned_source_x == 4 &&
        interaction.route.planned_source_y == 2);
}

static void test_same_target_selection_preserves_route(void) {
    OsrsInteraction interaction = {0};
    osrs_interaction_init(&interaction);
    osrs_interaction_set(&interaction, 3);
    interaction.route.state = OSRS_INTERACTION_ROUTE_READY;
    interaction.route.target_x = 12;
    interaction.route.target_y = 7;
    interaction.route.waypoint_count = 2;
    interaction.route.waypoint_index = 1;
    interaction.route.waypoint_x[1] = 11;
    interaction.route.waypoint_y[1] = 7;

    OsrsInteractionRoute preserved = interaction.route;
    osrs_interaction_set(&interaction, 3);
    CHECK("same target preserves its route",
        memcmp(&interaction.route, &preserved, sizeof(preserved)) == 0);

    osrs_interaction_set(&interaction, 4);
    CHECK("different target clears the route",
        interaction.target_slot == 4 &&
        interaction.route.state == OSRS_INTERACTION_ROUTE_EMPTY &&
        interaction.route.waypoint_count == 0 &&
        interaction.route.waypoint_index == 0);
}

static void test_attack_route_field_stops_at_first_reachable_target_edge(void) {
    EncounterArenaAttackRouteField field;
    encounter_build_arena_attack_route_field(
        &field,
        NULL,
        0,
        0,
        2,
        2,
        12,
        2,
        1,
        step_tile_walkable,
        NULL,
        NULL,
        NULL,
        0,
        0,
        STEP_GRID,
        STEP_GRID);

    EncounterAttackRouteLanding landing = encounter_arena_attack_route_landing(
        &field,
        12,
        2,
        1,
        1,
        osrs_los_open_query());

    CHECK("goal-directed field stops before exhausting the arena",
        field.visited_count < STEP_GRID * STEP_GRID);
    CHECK("goal-directed field keeps the first FIFO target edge",
        landing.route_found && landing.route_x == 11 && landing.route_y == 2);
}

static void test_unreachable_attack_route_field_exhausts_for_fallback(void) {
    EncounterArenaAttackRouteField field;
    encounter_build_arena_attack_route_field(
        &field,
        NULL,
        0,
        0,
        10,
        10,
        -20,
        10,
        1,
        step_tile_walkable,
        NULL,
        NULL,
        NULL,
        0,
        0,
        STEP_GRID,
        STEP_GRID);

    CHECK("unreachable target exhausts the field for fallback",
        field.visited_count == STEP_GRID * STEP_GRID);
}

static void test_attack_route_caches_walkability_per_tile(void) {
    EncounterArenaAttackRouteField field;
    StepWalkableCountCtx walkable_count = {0};
    encounter_build_arena_attack_route_field(
        &field,
        NULL,
        0,
        0,
        1,
        1,
        22,
        22,
        1,
        step_counted_tile_walkable,
        &walkable_count,
        NULL,
        NULL,
        0,
        0,
        STEP_GRID,
        STEP_GRID);

    CHECK("route field checks each arena tile at most once",
        walkable_count.calls <= STEP_GRID * STEP_GRID);
}

static void test_attackable_target_skips_route_scan(void) {
    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 10;
    player.y = 10;
    player.dest_x = 7;
    player.dest_y = 8;
    StepWalkableCountCtx walkable_count = {0};

    int moved = encounter_chase_attack_target(
        &player,
        12,
        10,
        1,
        2,
        NULL,
        0,
        0,
        step_counted_tile_walkable,
        &walkable_count,
        NULL,
        NULL,
        osrs_los_open_query(),
        0,
        0,
        0,
        0);

    CHECK("attackable target does not move", moved == 0);
    CHECK("attackable target skips walkability scan", walkable_count.calls == 0);
    CHECK("attackable target preserves destination",
        player.dest_x == 7 && player.dest_y == 8);
}


typedef struct {
    int origin_x;
    int origin_y;
    int width;
    int height;
    uint32_t flags[8][8];
} TopologyTestGeometry;

static uint32_t topology_test_flags(void* ctx, int x, int y) {
    const TopologyTestGeometry* geometry = (const TopologyTestGeometry*)ctx;
    int local_x = x - geometry->origin_x;
    int local_y = y - geometry->origin_y;
    if (local_x < 0 || local_x >= geometry->width ||
            local_y < 0 || local_y >= geometry->height)
        return COLLISION_BLOCKED | LOS_FULL_MASK;
    return geometry->flags[local_x][local_y];
}

static EncounterArenaTopologyBuildSpec topology_test_spec(
    TopologyTestGeometry* geometry,
    uint64_t revision
) {
    return (EncounterArenaTopologyBuildSpec){
        .origin_x = geometry->origin_x,
        .origin_y = geometry->origin_y,
        .width = geometry->width,
        .height = geometry->height,
        .max_footprint_size = ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE,
        .revision = revision,
        .tile_flags = topology_test_flags,
        .tile_flags_ctx = geometry,
    };
}

static int topology_test_aborts(void (*operation)(void)) {
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

static void topology_build_rejects_zero_width(void) {
    TopologyTestGeometry geometry = {
        .origin_x = 10,
        .origin_y = 20,
        .width = 0,
        .height = 8,
    };
    EncounterArenaTopologyBuildSpec spec = topology_test_spec(&geometry, 1);
    (void)encounter_arena_topology_build(&spec);
}

static void topology_build_rejects_oversized_height(void) {
    TopologyTestGeometry geometry = {
        .origin_x = 10,
        .origin_y = 20,
        .width = 8,
        .height = ENCOUNTER_ARENA_TOPOLOGY_MAX_DIMENSION + 1,
    };
    EncounterArenaTopologyBuildSpec spec = topology_test_spec(&geometry, 1);
    (void)encounter_arena_topology_build(&spec);
}

static void topology_build_rejects_origin_overflow(void) {
    TopologyTestGeometry geometry = {
        .origin_x = INT_MAX,
        .origin_y = 20,
        .width = 2,
        .height = 8,
    };
    EncounterArenaTopologyBuildSpec spec = topology_test_spec(&geometry, 1);
    (void)encounter_arena_topology_build(&spec);
}

static const EncounterArenaTopology* topology_stale_revision_target;

static void topology_rejects_stale_revision(void) {
    (void)encounter_arena_topology_require_revision(
        topology_stale_revision_target,
        topology_stale_revision_target->revision + 1);
}

static void test_arena_topology_bounds_collision_los_and_revision(void) {
    TopologyTestGeometry geometry = {
        .origin_x = 10,
        .origin_y = 20,
        .width = 8,
        .height = 8,
    };
    geometry.flags[0][3] = COLLISION_BLOCKED;
    geometry.flags[7][4] = COLLISION_BLOCKED;
    geometry.flags[3][0] = COLLISION_BLOCKED;
    geometry.flags[4][7] = COLLISION_BLOCKED;
    geometry.flags[2][2] = COLLISION_WALL_EAST;
    geometry.flags[3][2] = COLLISION_WALL_WEST;
    geometry.flags[2][3] = COLLISION_BLOCKED;
    geometry.flags[4][4] = LOS_FULL_MASK;

    size_t allocations_before_build = topology_allocation_count;
    EncounterArenaTopologyBuildSpec spec = topology_test_spec(&geometry, 41);
    EncounterArenaTopology* topology = encounter_arena_topology_build(&spec);

    CHECK("topology build uses one explicitly owned allocation",
        topology_allocation_count == allocations_before_build + 1);
    CHECK("topology stores validated arena identity",
        topology->origin_x == 10 &&
        topology->origin_y == 20 &&
        topology->width == 8 &&
        topology->height == 8 &&
        topology->revision == 41);

    encounter_arena_topology_finalize(topology);

    CHECK("topology bounds include both arena corners",
        encounter_arena_topology_contains(topology, 10, 20) &&
        encounter_arena_topology_contains(topology, 17, 27));
    CHECK("topology bounds reject every outside edge",
        !encounter_arena_topology_contains(topology, 9, 20) &&
        !encounter_arena_topology_contains(topology, 18, 20) &&
        !encounter_arena_topology_contains(topology, 10, 19) &&
        !encounter_arena_topology_contains(topology, 10, 28));
    CHECK("static blocked queries read all four arena edges",
        encounter_arena_topology_tile_blocked(topology, 10, 23) &&
        encounter_arena_topology_tile_blocked(topology, 17, 24) &&
        encounter_arena_topology_tile_blocked(topology, 13, 20) &&
        encounter_arena_topology_tile_blocked(topology, 14, 27));
    CHECK("static blocked queries treat outside bounds as blocked",
        encounter_arena_topology_tile_blocked(topology, 9, 20) &&
        encounter_arena_topology_tile_blocked(topology, 18, 20) &&
        encounter_arena_topology_tile_blocked(topology, 10, 19) &&
        encounter_arena_topology_tile_blocked(topology, 10, 28));

    CHECK("cardinal traversal rejects reciprocal wall edge",
        !encounter_arena_topology_step_allowed(topology, 12, 22, 1, 1, 0) &&
        !encounter_arena_topology_step_allowed(topology, 13, 22, 1, -1, 0));
    CHECK("diagonal traversal rejects blocked cardinal side",
        !encounter_arena_topology_step_allowed(topology, 12, 22, 1, 1, 1));
    CHECK("clear diagonal traversal remains allowed",
        encounter_arena_topology_step_allowed(topology, 15, 21, 1, 1, 1));

    int blocked_forward = encounter_arena_topology_los_clear(
        topology, 11, 24, 1, 16, 24, 1, 10);
    int blocked_reverse = encounter_arena_topology_los_clear(
        topology, 16, 24, 1, 11, 24, 1, 10);
    int clear_forward = encounter_arena_topology_los_clear(
        topology, 11, 21, 1, 16, 21, 1, 10);
    int clear_reverse = encounter_arena_topology_los_clear(
        topology, 16, 21, 1, 11, 21, 1, 10);
    CHECK("static LOS is symmetric through blocking geometry",
        !blocked_forward && blocked_forward == blocked_reverse);
    CHECK("static LOS is symmetric through open geometry",
        clear_forward && clear_forward == clear_reverse);
    CHECK("large-target LOS uses the closest occupied target tile",
        encounter_arena_topology_los_clear(
            topology, 11, 24, 1, 15, 23, 1, 10) &&
        !encounter_arena_topology_los_clear(
            topology, 11, 24, 1, 15, 23, 2, 10));

    CHECK("matching topology revision returns the same immutable object",
        encounter_arena_topology_require_revision(topology, 41) == topology);
    topology_stale_revision_target = topology;
    CHECK("stale topology revision aborts",
        topology_test_aborts(topology_rejects_stale_revision));

    static unsigned char snapshot[sizeof(EncounterArenaTopology)];
    memcpy(snapshot, topology, sizeof(*topology));
    size_t allocations_after_finalize = topology_allocation_count;
    (void)encounter_arena_topology_contains(topology, 12, 21);
    (void)encounter_arena_topology_tile_blocked(topology, 12, 21);
    (void)encounter_arena_topology_footprint_blocked(topology, 12, 21, 2);
    (void)encounter_arena_topology_step_allowed(topology, 15, 21, 1, 1, 1);
    (void)encounter_arena_topology_los_clear(
        topology, 11, 21, 1, 16, 21, 1, 10);
    (void)encounter_arena_topology_require_revision(topology, 41);
    CHECK("finalized topology queries do not allocate",
        topology_allocation_count == allocations_after_finalize);
    CHECK("finalized topology queries do not mutate",
        memcmp(snapshot, topology, sizeof(*topology)) == 0);

    free(topology);
    topology_stale_revision_target = NULL;
}

static void test_arena_topology_rejects_invalid_bounds(void) {
    CHECK("topology build rejects zero width",
        topology_test_aborts(topology_build_rejects_zero_width));
    CHECK("topology build rejects oversized height",
        topology_test_aborts(topology_build_rejects_oversized_height));
    CHECK("topology build rejects overflowing origin plus width",
        topology_test_aborts(topology_build_rejects_origin_overflow));
}

int main(void) {
    test_arena_topology_rejects_invalid_bounds();
    test_arena_topology_bounds_collision_los_and_revision();
    test_same_target_selection_preserves_route();
    test_attack_route_field_stops_at_first_reachable_target_edge();
    test_unreachable_attack_route_field_exhausts_for_fallback();
    test_attack_route_caches_walkability_per_tile();
    test_attackable_target_skips_route_scan();
    test_target_command_cancels_walk_in_flight();
    test_move_command_cancels_interaction();
    test_none_command_chases_active_interaction();
    test_attack_route_rejects_cardinal_adjacency_through_wall();
    test_attack_route_fallback_stays_within_ten_tiles();
    test_attack_route_uses_rsmod_cardinal_first_order();
    test_melee_attack_does_not_cross_cardinal_wall();
    test_attack_route_persists_and_invalidates_canonically();
    test_single_checkpoint_route_reroutes_each_tick();

    printf("\n%d/%d tests passed\n", tests_run - tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}
