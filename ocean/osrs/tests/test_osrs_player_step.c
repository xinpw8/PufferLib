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
    int open_x;
    int open_y;
} StepRouteBlockedCountCtx;





static int step_vertical_wall(void* ctx, int x, int y) {
    (void)ctx;
    return x == 5 && y <= 4;
}

static int step_vertical_wall_route(void* ctx, int x, int y, int size) {
    (void)size;
    return step_vertical_wall(ctx, x, y);
}
static int step_counted_isolated_route(void* ctx, int x, int y, int size) {
    (void)size;
    StepRouteBlockedCountCtx* count = (StepRouteBlockedCountCtx*)ctx;
    count->calls++;
    return x != count->open_x || y != count->open_y;
}


static uint32_t step_open_flags(void* ctx, int x, int y) {
    (void)ctx;
    (void)x;
    (void)y;
    return 0;
}

static const EncounterArenaTopology* step_route_topology(void) {
    static EncounterArenaTopology* topology;
    if (!topology) {
        EncounterArenaTopologyBuildSpec spec = {
            .origin_x = 0,
            .origin_y = 0,
            .width = STEP_GRID,
            .height = STEP_GRID,
            .max_footprint_size = 2,
            .revision = 19,
            .tile_flags = step_open_flags,
            .tile_flags_ctx = NULL,
        };
        topology = encounter_arena_topology_build(&spec);
        encounter_arena_topology_finalize(topology);
    }
    return topology;
}


static uint32_t step_collision_flags(void* ctx, int x, int y) {
    return (uint32_t)collision_get_flags((const CollisionMap*)ctx, 0, x, y);
}

static EncounterArenaTopology* step_collision_topology(
    const CollisionMap* collision_map,
    uint64_t revision
) {
    EncounterArenaTopologyBuildSpec spec = {
        .origin_x = 0,
        .origin_y = 0,
        .width = STEP_GRID,
        .height = STEP_GRID,
        .max_footprint_size = 2,
        .revision = revision,
        .tile_flags = step_collision_flags,
        .tile_flags_ctx = (void*)collision_map,
    };
    EncounterArenaTopology* topology = encounter_arena_topology_build(&spec);
    encounter_arena_topology_finalize(topology);
    return topology;
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
    arena.los_query = osrs_los_open_query();
    arena.topology = step_route_topology();
    arena.blockers.revision = 1;
    arena.movement_mode = ENCOUNTER_ROUTE_MOVEMENT_RUN;
    arena.cost_policy = ENCOUNTER_ROUTE_COST_OSRS;
    arena.destination_cost_policy = ENCOUNTER_ROUTE_COST_SOUTH_FIRST_BFS;
    return arena;
}
static void test_ranged_chase_targets_nearest_attack_position(void) {
    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 2;
    player.y = 10;
    OsrsInteraction interaction;
    osrs_interaction_init(&interaction);
    OsrsActorRouteCache route_cache = {0};
    StepTargetCtx target_ctx = {
        .target = {
            .slot = 3,
            .x = 10,
            .y = 10,
            .size = 1,
            .attack_range = 4,
        },
        .target_valid = 1,
    };
    int dest_x = -1;
    int dest_y = -1;
    OsrsPlayerStepInput input;
    memset(&input, 0, sizeof(input));
    input.player = &player;
    input.interaction = &interaction;
    input.route_cache = &route_cache;
    input.target_lookup = step_lookup_target;
    input.target_ctx = &target_ctx;
    input.command.kind = OSRS_PLAYER_CMD_TARGET;
    input.command.target_slot = 3;
    input.arena = step_arena();
    input.arena.cost_policy = ENCOUNTER_ROUTE_COST_OSRS_TARGET_BFS;
    input.dest_x = &dest_x;
    input.dest_y = &dest_y;

    OsrsPlayerStepResult result = osrs_encounter_player_step(&input);
    CHECK("range-four chase moves two steps toward attack range",
        result.chased_target && player.x == 4 && player.y == 10);
    CHECK("range-four chase targets the nearest valid attack tile",
        route_cache.waypoint_count == 1 &&
        route_cache.waypoint_x[0] == 6 &&
        route_cache.waypoint_y[0] == 10);
    CHECK("range-four chase does not route to cardinal adjacency",
        route_cache.waypoint_x[0] != 9);
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
    OsrsActorRouteCache route_cache = {0};
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
    EncounterArenaTopology* wall_topology =
        step_collision_topology(&map, 20);
    arena.topology = wall_topology;

    OsrsPlayerStepInput input;
    memset(&input, 0, sizeof(input));
    input.player = &player;
    input.interaction = &interaction;
    input.route_cache = &route_cache;
    input.target_lookup = step_lookup_target;
    input.target_ctx = &tctx;
    input.dest_x = &dest_x;
    input.dest_y = &dest_y;
    input.arena = arena;

    OsrsPlayerStepResult result = osrs_encounter_player_step(&input);

    CHECK("melee does not attack through cardinal wall",
        !result.can_attack || player.x != 9 || player.y != 10);
    CHECK("melee routes around cardinal wall", result.chased_target);
    free(wall_topology);
}

static void test_target_command_cancels_walk_in_flight(void) {
    printf("--- entity click cancels a walk in flight ---\n");

    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 5; player.y = 5;
    player.run_energy = 10000;

    OsrsInteraction interaction;
    OsrsActorRouteCache route_cache = {0};
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
    input.route_cache = &route_cache;
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
    OsrsActorRouteCache route_cache = {0};
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
    input.route_cache = &route_cache;
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
static void test_destination_click_routes_around_wall(void) {
    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 2;
    player.y = 2;
    player.run_energy = 10000;

    OsrsInteraction interaction;
    OsrsActorRouteCache route_cache = {0};
    osrs_interaction_init(&interaction);
    StepTargetCtx target_ctx = {0};
    int dest_x = 10;
    int dest_y = 2;
    OsrsEncounterArena arena = step_arena();
    arena.blockers = (EncounterRouteBlockers){
        .is_blocked = step_vertical_wall_route,
        .revision = 2,
    };

    OsrsPlayerStepInput input;
    memset(&input, 0, sizeof(input));
    input.player = &player;
    input.interaction = &interaction;
    input.route_cache = &route_cache;
    input.target_lookup = step_lookup_target;
    input.target_ctx = &target_ctx;
    input.dest_x = &dest_x;
    input.dest_y = &dest_y;
    input.command.kind = OSRS_PLAYER_CMD_MOVE;
    input.command.move_kind = OSRS_PLAYER_MOVE_DESTINATION;
    input.arena = arena;

    for (int tick = 0; tick < 20 && dest_x >= 0; tick++) {
        osrs_encounter_player_step(&input);
    }
    CHECK("destination click routes around a wall to the clicked tile",
        player.x == 10 && player.y == 2 && dest_x == -1 && dest_y == -1);
}


/* no click: an active interaction keeps auto-chasing. */
static void test_none_command_chases_active_interaction(void) {
    printf("--- idle tick auto-chases the standing interaction ---\n");

    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 5; player.y = 5;
    player.run_energy = 10000;

    OsrsInteraction interaction;
    OsrsActorRouteCache route_cache = {0};
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
    input.route_cache = &route_cache;
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
    OsrsActorRouteCache route_cache = {0};
    osrs_interaction_init(&interaction);
    osrs_interaction_set(&interaction, 3);

    StepTargetCtx target_ctx = {
        .target = { .slot = 3, .x = 10, .y = 2, .size = 1, .attack_range = 1 },
        .target_valid = 1,
    };
    int dest_x = -1;
    int dest_y = -1;
    OsrsEncounterArena arena = step_arena();
    arena.blockers.is_blocked = step_vertical_wall_route;
    arena.blockers.revision = 7;

    OsrsPlayerStepInput input;
    memset(&input, 0, sizeof(input));
    input.player = &player;
    input.interaction = &interaction;
    input.route_cache = &route_cache;
    input.target_lookup = step_lookup_target;
    input.target_ctx = &target_ctx;
    input.dest_x = &dest_x;
    input.dest_y = &dest_y;
    input.arena = arena;

    OsrsPlayerStepResult first = osrs_encounter_player_step(&input);
    CHECK("obstacle chase builds a multi-checkpoint route",
        first.chased_target && route_cache.waypoint_count > 1);
    CHECK("route records its original source",
        route_cache.planned_source_x == 2 &&
        route_cache.planned_source_y == 2);

    osrs_encounter_player_step(&input);
    CHECK("route persists while more than one checkpoint remains",
        route_cache.planned_source_x == 2 &&
        route_cache.planned_source_y == 2);
    int blocker_changed_source_x = player.x;
    int blocker_changed_source_y = player.y;
    input.arena.blockers.revision++;
    osrs_encounter_player_step(&input);
    CHECK("dynamic blocker revision invalidates actor-local route cache",
        route_cache.blocker_revision == input.arena.blockers.revision &&
        route_cache.planned_source_x == blocker_changed_source_x &&
        route_cache.planned_source_y == blocker_changed_source_y);

    target_ctx.target.y = 3;
    int moved_target_source_x = player.x;
    int moved_target_source_y = player.y;
    osrs_encounter_player_step(&input);
    CHECK("target geometry change reroutes from the current tile",
        route_cache.target_y == 3 &&
        route_cache.planned_source_x == moved_target_source_x &&
        route_cache.planned_source_y == moved_target_source_y);

    player.x = 1;
    player.y = 10;
    osrs_encounter_player_step(&input);
    CHECK("player route divergence reroutes from the observed tile",
        route_cache.planned_source_x == 1 &&
        route_cache.planned_source_y == 10);
}
static void test_failed_attack_route_persists_and_invalidates(void) {
    Player player = {
        .x = 2,
        .y = 2,
    };
    OsrsInteraction interaction;
    OsrsActorRouteCache route_cache = {0};
    osrs_interaction_init(&interaction);
    osrs_interaction_set(&interaction, 3);

    StepTargetCtx target_ctx = {
        .target = {
            .slot = 3,
            .x = 20,
            .y = 20,
            .size = 1,
            .attack_range = 1,
        },
        .target_valid = 1,
    };
    StepRouteBlockedCountCtx blocker_ctx = {
        .open_x = player.x,
        .open_y = player.y,
    };
    int dest_x = -1;
    int dest_y = -1;
    OsrsEncounterArena arena = step_arena();
    arena.blockers.is_blocked = step_counted_isolated_route;
    arena.blockers.ctx = &blocker_ctx;
    arena.blockers.revision = 7;

    OsrsPlayerStepInput input = {
        .player = &player,
        .interaction = &interaction,
        .route_cache = &route_cache,
        .target_lookup = step_lookup_target,
        .target_ctx = &target_ctx,
        .dest_x = &dest_x,
        .dest_y = &dest_y,
        .arena = arena,
    };

    osrs_encounter_player_step(&input);
    int calls_after_first_route = blocker_ctx.calls;
    CHECK("isolated player produces a failed attack route",
        route_cache.state == OSRS_INTERACTION_ROUTE_FAILED &&
        calls_after_first_route > 0);

    osrs_encounter_player_step(&input);
    CHECK("unchanged failed attack route is cached",
        blocker_ctx.calls == calls_after_first_route);

    input.arena.blockers.revision++;
    osrs_encounter_player_step(&input);
    CHECK("blocker revision invalidates a failed attack route",
        blocker_ctx.calls > calls_after_first_route);
    int calls_after_blocker_change = blocker_ctx.calls;

    target_ctx.target.x--;
    osrs_encounter_player_step(&input);
    CHECK("target geometry invalidates a failed attack route",
        blocker_ctx.calls > calls_after_blocker_change);
    int calls_after_target_change = blocker_ctx.calls;

    player.x++;
    blocker_ctx.open_x = player.x;
    osrs_encounter_player_step(&input);
    CHECK("player position invalidates a failed attack route",
        blocker_ctx.calls > calls_after_target_change);
}


static void test_single_checkpoint_route_persists(void) {
    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 2;
    player.y = 2;

    OsrsInteraction interaction;
    OsrsActorRouteCache route_cache = {0};
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
    input.route_cache = &route_cache;
    input.target_lookup = step_lookup_target;
    input.target_ctx = &target_ctx;
    input.dest_x = &dest_x;
    input.dest_y = &dest_y;
    input.arena = arena;

    osrs_encounter_player_step(&input);
    CHECK("straight chase compresses to one checkpoint",
        route_cache.waypoint_count == 1 &&
        route_cache.planned_source_x == 2);
    osrs_encounter_player_step(&input);
    CHECK("one-checkpoint entity route persists while traversable",
        route_cache.planned_source_x == 2 &&
        route_cache.planned_source_y == 2 &&
        player.x == 6 && player.y == 2);
}

static void test_target_selection_does_not_own_route_cache(void) {
    OsrsInteraction interaction = {0};
    OsrsActorRouteCache route_cache = {
        .state = OSRS_INTERACTION_ROUTE_READY,
        .target_x = 12,
        .target_y = 7,
        .waypoint_count = 2,
        .waypoint_index = 1,
    };
    route_cache.waypoint_x[1] = 11;
    route_cache.waypoint_y[1] = 7;
    osrs_interaction_init(&interaction);
    osrs_interaction_set(&interaction, 3);

    OsrsActorRouteCache preserved = route_cache;
    osrs_interaction_set(&interaction, 4);
    CHECK("target selection leaves context-owned route cache unchanged",
        interaction.target_slot == 4 &&
        memcmp(&route_cache, &preserved, sizeof(preserved)) == 0);
}






typedef struct {
    int origin_x;
    int origin_y;
    int width;
    int height;
    int outside_reads;
    uint32_t flags[8][8];
} TopologyTestGeometry;

static uint32_t topology_test_flags(void* ctx, int x, int y) {
    TopologyTestGeometry* geometry = (TopologyTestGeometry*)ctx;
    int64_t local_x = (int64_t)x - geometry->origin_x;
    int64_t local_y = (int64_t)y - geometry->origin_y;
    if (local_x < 0 || local_x >= geometry->width ||
            local_y < 0 || local_y >= geometry->height) {
        geometry->outside_reads++;
        return COLLISION_BLOCKED | LOS_FULL_MASK;
    }
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
    CHECK("topology detects flagged static LOS",
        topology->static_los_mode == ENCOUNTER_ARENA_TOPOLOGY_LOS_FLAGGED);

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
    CHECK("blocked movement tile also blocks static LOS",
        !encounter_arena_topology_los_clear(
            topology, 11, 23, 1, 13, 23, 1, 10));
    CHECK("LOS rejects overlapping footprints",
        !encounter_arena_topology_los_clear(
            topology, 11, 21, 1, 11, 21, 1, 10));
    CHECK("range-one LOS accepts only cardinal adjacency",
        encounter_arena_topology_los_clear(
            topology, 11, 21, 1, 12, 21, 1, 1) &&
        !encounter_arena_topology_los_clear(
            topology, 11, 21, 1, 12, 22, 1, 1) &&
        !encounter_arena_topology_los_clear(
            topology, 11, 21, 1, 13, 21, 1, 1));
    CHECK("ranged LOS rejects pairs outside attack range",
        !encounter_arena_topology_los_clear(
            topology, 11, 21, 1, 17, 21, 1, 5));
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
    TopologyTestGeometry open_geometry = {
        .origin_x = 10,
        .origin_y = 20,
        .width = 8,
        .height = 8,
    };
    EncounterArenaTopologyBuildSpec open_spec =
        topology_test_spec(&open_geometry, 42);
    EncounterArenaTopology* open_topology =
        encounter_arena_topology_build(&open_spec);
    encounter_arena_topology_finalize(open_topology);
    CHECK("topology detects open static LOS",
        open_topology->static_los_mode ==
            ENCOUNTER_ARENA_TOPOLOGY_LOS_OPEN);
    for (int actor_size = 1; actor_size <= 3; actor_size++) {
        for (int target_size = 1; target_size <= 3; target_size++) {
            for (int range = 1; range <= 10; range++) {
                int expected = entity_has_line_of_sight(
                    NULL, 0,
                    11, 21, actor_size,
                    15, 24, target_size,
                    range);
                int actual = encounter_arena_topology_los_clear(
                    open_topology,
                    11, 21, actor_size,
                    15, 24, target_size,
                    range);
                CHECK("open topology LOS matches open geometry reference",
                    actual == expected);
            }
        }
    }
    free(open_topology);
    topology_stale_revision_target = NULL;
}

static void test_arena_topology_extreme_origins_and_reader_bounds(void) {
    TopologyTestGeometry geometry = {
        .origin_x = INT_MAX,
        .origin_y = INT_MIN,
        .width = 1,
        .height = 1,
    };
    EncounterArenaTopologyBuildSpec spec = topology_test_spec(&geometry, 52);
    EncounterArenaTopology* topology = encounter_arena_topology_build(&spec);

    CHECK("topology build never reads geometry outside arena bounds",
        geometry.outside_reads == 0);

    encounter_arena_topology_finalize(topology);
    CHECK("INT_MAX width-one and INT_MIN height-one origin is contained",
        encounter_arena_topology_contains(topology, INT_MAX, INT_MIN));
    CHECK("extreme origin tile stays walkable",
        !encounter_arena_topology_tile_blocked(
            topology, INT_MAX, INT_MIN) &&
        !encounter_arena_topology_footprint_blocked(
            topology, INT_MAX, INT_MIN, 1));
    CHECK("INT_MIN north edge queries stay out of bounds",
        !encounter_arena_topology_contains(
            topology, INT_MAX, INT_MIN + 1) &&
        encounter_arena_topology_tile_blocked(
            topology, INT_MAX, INT_MIN + 1) &&
        encounter_arena_topology_footprint_blocked(
            topology, INT_MAX, INT_MIN + 1, 1));
    CHECK("INT_MAX west edge queries stay out of bounds",
        !encounter_arena_topology_contains(
            topology, INT_MAX - 1, INT_MIN) &&
        encounter_arena_topology_footprint_blocked(
            topology, INT_MAX - 1, INT_MIN, 1));
    CHECK("all steps from extreme one-tile arena reject without reader access",
        !encounter_arena_topology_step_allowed(
            topology, INT_MAX, INT_MIN, 1, -1, 0) &&
        !encounter_arena_topology_step_allowed(
            topology, INT_MAX, INT_MIN, 1, 1, 0) &&
        !encounter_arena_topology_step_allowed(
            topology, INT_MAX, INT_MIN, 1, 0, -1) &&
        !encounter_arena_topology_step_allowed(
            topology, INT_MAX, INT_MIN, 1, 0, 1) &&
        geometry.outside_reads == 0);

    free(topology);
}

static void test_arena_topology_rejects_invalid_bounds(void) {
    CHECK("topology build rejects zero width",
        topology_test_aborts(topology_build_rejects_zero_width));
    CHECK("topology build rejects oversized height",
        topology_test_aborts(topology_build_rejects_oversized_height));
    CHECK("topology build rejects overflowing origin plus width",
        topology_test_aborts(topology_build_rejects_origin_overflow));
}

typedef struct {
    uint32_t flags[8][8];
} RouteTestGeometry;

typedef struct {
    uint8_t blocked[8][8];
    int calls;
} RouteTestBlockers;

static uint32_t route_test_flags(void* data, int x, int y) {
    RouteTestGeometry* geometry = (RouteTestGeometry*)data;
    if (x < 0 || x >= 8 || y < 0 || y >= 8)
        return COLLISION_BLOCKED | LOS_FULL_MASK;
    return geometry->flags[x][y];
}

static int route_test_blocked(void* data, int x, int y, int size) {
    RouteTestBlockers* blockers = (RouteTestBlockers*)data;
    blockers->calls++;
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int tile_x = x + dx;
            int tile_y = y + dy;
            if (tile_x < 0 || tile_x >= 8 || tile_y < 0 || tile_y >= 8)
                return 1;
            if (blockers->blocked[tile_x][tile_y]) return 1;
        }
    }
    return 0;
}

static EncounterArenaTopology* route_test_topology(
    RouteTestGeometry* geometry
) {
    EncounterArenaTopologyBuildSpec spec = {
        .origin_x = 0,
        .origin_y = 0,
        .width = 8,
        .height = 8,
        .max_footprint_size = 2,
        .revision = 71,
        .tile_flags = route_test_flags,
        .tile_flags_ctx = geometry,
    };
    EncounterArenaTopology* topology = encounter_arena_topology_build(&spec);
    encounter_arena_topology_finalize(topology);
    return topology;
}

static EncounterRouteInput route_test_input(
    const EncounterArenaTopology* topology,
    RouteTestBlockers* blockers
) {
    return (EncounterRouteInput){
        .topology = topology,
        .blockers = {
            .is_blocked = route_test_blocked,
            .ctx = blockers,
            .revision = 1,
        },
        .source_x = 1,
        .source_y = 1,
        .actor_size = 1,
        .target_x = 4,
        .target_y = 1,
        .target_size = 1,
        .target_kind = ENCOUNTER_ROUTE_TARGET_TILE,
        .movement_mode = ENCOUNTER_ROUTE_MOVEMENT_RUN,
        .cost_policy = ENCOUNTER_ROUTE_COST_OSRS,
    };
}

static void test_tagged_route_outcomes_and_payload(void) {
    RouteTestGeometry geometry = {0};
    RouteTestBlockers blockers = {0};
    EncounterArenaTopology* topology = route_test_topology(&geometry);
    EncounterRouteInput input = route_test_input(topology, &blockers);

    EncounterRouteResult reached = encounter_route_solve(&input);
    CHECK("exact target returns tagged reached-target outcome",
        reached.outcome == ROUTE_REACHED_TARGET);
    CHECK("reached route pins destination, shortest distance, and run steps",
        reached.destination_x == 4 && reached.destination_y == 1 &&
        reached.distance == 3 &&
        reached.first_dx == 1 && reached.first_dy == 0 &&
        reached.run_dx == 1 && reached.run_dy == 0);
    CHECK("straight route compresses to its exact waypoint",
        reached.waypoint_count == 1 &&
        reached.waypoint_x[0] == 4 && reached.waypoint_y[0] == 1);

    for (int y = 0; y < 8; y++) blockers.blocked[3][y] = 1;
    input.blockers.revision++;
    EncounterRouteResult fallback = encounter_route_solve(&input);
    CHECK("dynamic wall selects tagged fallback without topology mutation",
        fallback.outcome == ROUTE_REACHED_FALLBACK &&
        topology->revision == 71 &&
        fallback.destination_x == 2 && fallback.destination_y == 1 &&
        fallback.distance == 1);

    memset(&blockers, 1, sizeof(blockers));
    blockers.blocked[1][1] = 0;
    input.blockers.revision++;
    input.target_x = 20;
    EncounterRouteResult unreachable = encounter_route_solve(&input);
    CHECK("no target or fallback returns tagged unreachable",
        unreachable.outcome == ROUTE_UNREACHABLE);

    input.topology = NULL;
    EncounterRouteResult invalid = encounter_route_solve(&input);
    CHECK("bad route contract returns tagged invalid input",
        invalid.outcome == ROUTE_INVALID_INPUT);
    free(topology);
}

static void test_route_cost_order_is_deterministic(void) {
    RouteTestGeometry geometry = {0};
    RouteTestBlockers blockers = {0};
    EncounterArenaTopology* topology = route_test_topology(&geometry);
    EncounterRouteInput input = route_test_input(topology, &blockers);
    input.source_x = 4;
    input.source_y = 4;
    input.target_x = 2;
    input.target_y = 2;
    input.target_kind = ENCOUNTER_ROUTE_TARGET_CARDINAL_ADJACENCY;

    EncounterRouteResult route = encounter_route_solve(&input);
    CHECK("OSRS attack route takes west before southwest",
        route.outcome == ROUTE_REACHED_TARGET &&
        route.first_dx == -1 && route.first_dy == 0 &&
        route.run_dx == -1 && route.run_dy == -1);
    CHECK("attack route selects the first equal-cost FIFO target edge",
        route.destination_x == 2 && route.destination_y == 3 &&
        route.distance == 2);
    free(topology);
}
static void test_attack_range_overlap_uses_deterministic_escape(void) {
    RouteTestGeometry geometry = {0};
    RouteTestBlockers blockers = {0};
    EncounterArenaTopology* topology = route_test_topology(&geometry);
    EncounterRouteInput input = route_test_input(topology, &blockers);
    input.source_x = 2;
    input.source_y = 2;
    input.target_x = 2;
    input.target_y = 2;
    input.target_size = 2;
    input.target_kind = ENCOUNTER_ROUTE_TARGET_ATTACK_RANGE;
    input.attack_range = 4;
    input.los_query = osrs_los_open_query();

    EncounterRouteResult route = encounter_route_solve(&input);
    CHECK("range overlap uses north-first deterministic escape",
        route.outcome == ROUTE_REACHED_TARGET &&
        route.first_dx == 0 && route.first_dy == -1 &&
        route.destination_x == 2 && route.destination_y == 1);
    free(topology);
}

static void test_destination_south_first_tie_order(void) {
    RouteTestGeometry geometry = {0};
    RouteTestBlockers blockers = {0};
    blockers.blocked[3][3] = 1;
    EncounterArenaTopology* topology = route_test_topology(&geometry);
    EncounterRouteInput input = route_test_input(topology, &blockers);
    input.source_x = 4;
    input.source_y = 4;
    input.target_x = 2;
    input.target_y = 2;

    input.cost_policy = ENCOUNTER_ROUTE_COST_OSRS;
    EncounterRouteResult osrs = encounter_route_solve(&input);
    input.cost_policy = ENCOUNTER_ROUTE_COST_SOUTH_FIRST_BFS;
    EncounterRouteResult destination = encounter_route_solve(&input);
    CHECK("symmetric obstacle exposes distinct equal-cost orders",
        osrs.outcome == ROUTE_REACHED_TARGET &&
        destination.outcome == ROUTE_REACHED_TARGET &&
        osrs.distance == destination.distance &&
        osrs.first_dx == -1 && osrs.first_dy == 0 &&
        destination.first_dx == 0 && destination.first_dy == -1);

    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 4;
    player.y = 4;
    int dest_x = 2;
    int dest_y = 2;
    OsrsPlayerStepInput step_input;
    memset(&step_input, 0, sizeof(step_input));
    step_input.player = &player;
    step_input.dest_x = &dest_x;
    step_input.dest_y = &dest_y;
    step_input.arena = step_arena();
    step_input.arena.topology = topology;
    step_input.arena.blockers = input.blockers;
    int steps = osrs_player_step_apply_explicit_move(
        &step_input, OSRS_PLAYER_MOVE_DESTINATION);
    CHECK("destination movement uses south-first equal-cost order",
        steps == 2 && player.x == 4 && player.y == 2);
    free(topology);
}
static void test_reverse_route_field_reuses_matching_target(void) {
    RouteTestGeometry geometry = {0};
    RouteTestBlockers blockers = {0};
    blockers.blocked[3][3] = 1;
    EncounterArenaTopology* topology = route_test_topology(&geometry);
    EncounterRouteInput input = route_test_input(topology, &blockers);
    input.source_x = 1;
    input.source_y = 1;
    input.target_x = 6;
    input.target_y = 6;
    input.cost_policy = ENCOUNTER_ROUTE_COST_SOUTH_FIRST_REVERSE;

    EncounterRouteResult first = encounter_route_solve(&input);
    int first_calls = blockers.calls;
    blockers.calls = 0;
    EncounterRouteResult second = encounter_route_solve(&input);
    int second_calls = blockers.calls;

    CHECK("matching reverse route field preserves its route",
        first.outcome == second.outcome &&
        first.destination_x == second.destination_x &&
        first.destination_y == second.destination_y &&
        first.distance == second.distance &&
        first.first_dx == second.first_dx &&
        first.first_dy == second.first_dy &&
        first.run_dx == second.run_dx &&
        first.run_dy == second.run_dy);
    CHECK("matching reverse route field reuses prior expansion",
        second_calls < first_calls);
    input.result_detail = ENCOUNTER_ROUTE_RESULT_NEXT_STEPS;
    EncounterRouteResult next_steps = encounter_route_solve(&input);
    CHECK("next-step reverse result preserves movement",
        next_steps.outcome == first.outcome &&
        next_steps.destination_x == first.destination_x &&
        next_steps.destination_y == first.destination_y &&
        next_steps.distance == first.distance &&
        next_steps.first_dx == first.first_dx &&
        next_steps.first_dy == first.first_dy &&
        next_steps.run_dx == first.run_dx &&
        next_steps.run_dy == first.run_dy &&
        next_steps.waypoint_count == 0);
    input.result_detail = ENCOUNTER_ROUTE_RESULT_FULL;
    blockers.blocked
        [input.source_x + first.first_dx]
        [input.source_y + first.first_dy] = 1;
    blockers.calls = 0;
    input.blockers.revision++;
    EncounterRouteResult after_revision =
        encounter_route_solve(&input);
    CHECK("changed blocker revision invalidates reverse route field",
        blockers.calls > 0 &&
        (after_revision.first_dx != first.first_dx ||
         after_revision.first_dy != first.first_dy));
    free(topology);
}
static void test_blocked_destination_stops_at_nearest_fallback(void) {
    RouteTestGeometry geometry = {0};
    RouteTestBlockers blockers = {0};
    blockers.blocked[3][3] = 1;
    EncounterArenaTopology* topology = route_test_topology(&geometry);
    EncounterRouteInput input = route_test_input(topology, &blockers);
    input.source_x = 1;
    input.source_y = 1;
    input.target_x = 3;
    input.target_y = 3;
    input.cost_policy = ENCOUNTER_ROUTE_COST_SOUTH_FIRST_BFS;

    EncounterRouteResult route = encounter_route_solve(&input);
    CHECK("blocked destination selects the canonical nearest fallback",
        route.outcome == ROUTE_REACHED_FALLBACK &&
        route.destination_x == 2 && route.destination_y == 3 &&
        route.distance == 2 &&
        route.first_dx == 0 && route.first_dy == 1 &&
        route.run_dx == 1 && route.run_dy == 1);
    CHECK("blocked destination stops after its nearest fallback depth",
        blockers.calls < 32);
    input.source_x = 2;
    input.source_y = 3;
    blockers.blocked[2][3] = 1;
    input.blockers.revision++;
    route = encounter_route_solve(&input);
    CHECK("blocked source adjacent to blocked destination is unreachable",
        route.outcome == ROUTE_UNREACHABLE);
    free(topology);
}

static void test_source_field_reuses_traversal_across_targets(void) {
    RouteTestGeometry geometry = {0};
    RouteTestBlockers blockers = {0};
    blockers.blocked[5][6] = 1;
    blockers.blocked[7][6] = 1;
    blockers.blocked[6][5] = 1;
    blockers.blocked[6][7] = 1;
    EncounterArenaTopology* topology = route_test_topology(&geometry);
    EncounterRouteInput input = route_test_input(topology, &blockers);
    input.target_x = 6;
    input.target_y = 6;
    input.target_kind = ENCOUNTER_ROUTE_TARGET_ATTACK_RANGE;
    input.attack_range = 1;

    EncounterRouteResult unreachable = encounter_route_solve(&input);
    int calls_after_exhaustion = blockers.calls;
    CHECK("enclosed target exhausts source traversal",
        unreachable.outcome == ROUTE_REACHED_FALLBACK &&
        calls_after_exhaustion > 4);

    input.target_y = 1;
    EncounterRouteResult reached = encounter_route_solve(&input);
    CHECK("new target reuses exhausted source traversal",
        reached.outcome == ROUTE_REACHED_TARGET &&
        reached.destination_x == 5 && reached.destination_y == 1 &&
        blockers.calls == calls_after_exhaustion + 4);
    free(topology);
}

static void test_direct_route_policy_preserves_greedy_step_order(void) {
    RouteTestGeometry geometry = {0};
    RouteTestBlockers blockers = {0};
    EncounterArenaTopology* topology = route_test_topology(&geometry);
    EncounterRouteInput input = route_test_input(topology, &blockers);
    input.source_x = 1;
    input.source_y = 1;
    input.target_x = 4;
    input.target_y = 4;
    input.cost_policy = ENCOUNTER_ROUTE_COST_DIRECT;
    blockers.blocked[2][2] = 1;

    EncounterRouteResult route = encounter_route_solve(&input);
    CHECK("direct route falls from blocked diagonal to x cardinal",
        route.outcome == ROUTE_REACHED_FALLBACK &&
        route.first_dx == 1 && route.first_dy == 0 &&
        route.run_dx == 1 && route.run_dy == 0 &&
        route.destination_x == 3 && route.destination_y == 1);
    free(topology);
}

static void route_generation_wrap_operation(void) {
    RouteTestGeometry geometry;
    memset(&geometry, 0, sizeof(geometry));
    for (int x = 0; x < 8; x++)
        for (int y = 0; y < 8; y++)
            geometry.flags[x][y] = COLLISION_BLOCKED;
    for (int x = 1; x <= 4; x++) geometry.flags[x][1] = 0;

    RouteTestBlockers blockers = {0};
    EncounterArenaTopology* topology = route_test_topology(&geometry);
    EncounterRouteInput input = route_test_input(topology, &blockers);
    int stale_index = 2 * topology->height + 1;
    encounter_route_scratch.current_generation = UINT16_MAX;
    encounter_route_scratch.generation[stale_index] = 2;
    encounter_route_scratch.depth[stale_index] = 0;
    encounter_route_scratch.via[stale_index] = VIA_START;

    input.cost_policy = ENCOUNTER_ROUTE_COST_SOUTH_FIRST_REVERSE;
    EncounterRouteResult reverse = {0};
    if (!encounter_route_try_reverse(&input, &reverse) ||
            reverse.outcome != ROUTE_REACHED_TARGET)
        abort();

    input.cost_policy = ENCOUNTER_ROUTE_COST_SOUTH_FIRST_BFS;
    EncounterRouteResult bfs = encounter_route_solve(&input);
    if (bfs.outcome != ROUTE_REACHED_TARGET ||
            bfs.distance != 3 ||
            bfs.first_dx != 1 ||
            bfs.first_dy != 0)
        abort();
    free(topology);
}

static void test_route_generation_wrap_clears_stale_bfs_roots(void) {
    CHECK("route generation wrap clears stale BFS roots",
        !topology_test_aborts(route_generation_wrap_operation));
}



int main(void) {
    test_tagged_route_outcomes_and_payload();
    test_route_cost_order_is_deterministic();
    test_attack_range_overlap_uses_deterministic_escape();
    test_destination_south_first_tie_order();
    test_reverse_route_field_reuses_matching_target();
    test_blocked_destination_stops_at_nearest_fallback();
    test_source_field_reuses_traversal_across_targets();
    test_direct_route_policy_preserves_greedy_step_order();
    test_route_generation_wrap_clears_stale_bfs_roots();
    test_arena_topology_rejects_invalid_bounds();
    test_arena_topology_extreme_origins_and_reader_bounds();
    test_arena_topology_bounds_collision_los_and_revision();
    test_target_selection_does_not_own_route_cache();
    test_ranged_chase_targets_nearest_attack_position();
    test_target_command_cancels_walk_in_flight();
    test_move_command_cancels_interaction();
    test_destination_click_routes_around_wall();
    test_none_command_chases_active_interaction();
    test_melee_attack_does_not_cross_cardinal_wall();
    test_attack_route_persists_and_invalidates_canonically();
    test_failed_attack_route_persists_and_invalidates();
    test_single_checkpoint_route_persists();

    printf("\n%d/%d tests passed\n", tests_run - tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}
