/**
 * @file test_osrs_player_step.c
 * @brief Shared player-step command contract: one click per tick.
 *
 * The engine cannot both start an attack and walk to a chosen tile on the same
 * tick. A ground click cancels the entity interaction; an entity click cancels
 * the walk. These pin that contract at the shared layer, where every encounter
 * now routes through OsrsPlayerCommand.
 */

#include <stdio.h>
#include <string.h>

#include "ocean/osrs/osrs_encounter_player.h"

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


int main(void) {
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
