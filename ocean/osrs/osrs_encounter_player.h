#ifndef OSRS_ENCOUNTER_PLAYER_H
#define OSRS_ENCOUNTER_PLAYER_H

#include "osrs_encounter.h"
#include "osrs_interaction.h"
typedef struct {
    int16_t destination_dx;
    int16_t destination_dy;
    int8_t first_dx;
    int8_t first_dy;
    int8_t run_dx;
    int8_t run_dy;
    uint16_t distance;
    uint8_t outcome;
} OsrsLocalMoveRoute;

typedef enum {
    OSRS_PLAYER_MOVE_NONE = 0,
    OSRS_PLAYER_MOVE_ACTION,
    OSRS_PLAYER_MOVE_DESTINATION,
} OsrsPlayerMoveKind;

/* One click per tick. */
typedef enum {
    OSRS_PLAYER_CMD_NONE = 0,
    OSRS_PLAYER_CMD_TARGET,
    OSRS_PLAYER_CMD_MOVE,
} OsrsPlayerCommandKind;

typedef struct {
    OsrsPlayerCommandKind kind;
    int target_slot;
    OsrsPlayerMoveKind move_kind;
    int move_action;
} OsrsPlayerCommand;


typedef struct {
    const EncounterArenaTopology* topology;
    EncounterRouteBlockers blockers;
    EncounterRouteMovementMode movement_mode;
    EncounterRouteCostPolicy cost_policy;
    EncounterRouteCostPolicy destination_cost_policy;
    EncounterRouteAttackGeometry attack_geometry;
    const CollisionMap* collision_map;
    int world_offset_x;
    int world_offset_y;
    const OsrsLosQuery* los_query;
} OsrsEncounterArena;

typedef struct {
    int slot;
    int x;
    int y;
    int size;
    int attack_range;
} OsrsAttackTarget;

typedef int (*OsrsAttackTargetLookupFn)(
    void* ctx,
    int target_slot,
    OsrsAttackTarget* out);

typedef struct {
    Player* player;
    OsrsInteraction* interaction;
    OsrsActorRouteCache* route_cache;
    OsrsAttackTargetLookupFn target_lookup;
    void* target_ctx;
    OsrsPlayerCommand command;
    int* dest_x;
    int* dest_y;
    int blocked_ticks;
    OsrsEncounterArena arena;
} OsrsPlayerStepInput;

typedef struct {
    int moved;
    int explicit_moved;
    int chased_target;
    int interaction_active;
    int target_slot;
    int can_attack;
} OsrsPlayerStepResult;

static inline void osrs_player_step_require_input(const OsrsPlayerStepInput* input) {
    if (!input || !input->player || !input->interaction ||
            !input->arena.topology) {
        fprintf(stderr, "osrs player step input is missing required fields\n");
        abort();
    }
    encounter_arena_topology_require_finalized(input->arena.topology);
    if (input->command.kind == OSRS_PLAYER_CMD_MOVE &&
            (input->command.move_kind == OSRS_PLAYER_MOVE_DESTINATION ||
             input->command.move_kind == OSRS_PLAYER_MOVE_ACTION) &&
            (!input->dest_x || !input->dest_y)) {
        fprintf(stderr, "osrs player step move input is missing destination storage\n");
        abort();
    }
}

static inline int osrs_player_step_lookup_target(
    const OsrsPlayerStepInput* input,
    int target_slot,
    OsrsAttackTarget* target
) {
    if (!input->target_lookup) return 0;
    return input->target_lookup(input->target_ctx, target_slot, target);
}

static OSRS_ROUTE_NOINLINE int osrs_player_step_can_attack_target(
    const OsrsPlayerStepInput* input,
    const OsrsAttackTarget* target
) {
    if (input->arena.attack_geometry ==
            ENCOUNTER_ROUTE_ATTACK_GEOMETRY_TOPOLOGY) {
        return encounter_arena_topology_player_can_attack_trusted(
            input->arena.topology,
            input->player->x,
            input->player->y,
            target->x,
            target->y,
            target->size,
            target->attack_range);
    }
    return encounter_player_can_attack(
        input->player->x,
        input->player->y,
        target->x,
        target->y,
        target->size,
        target->attack_range,
        input->arena.collision_map,
        input->arena.world_offset_x,
        input->arena.world_offset_y,
        input->arena.los_query);
}

static inline int osrs_player_step_apply_route(
    Player* player,
    const EncounterRouteResult* route
) {
    if ((route->outcome != ROUTE_REACHED_TARGET &&
            route->outcome != ROUTE_REACHED_FALLBACK) ||
            route->distance == 0) {
        player->is_running = 0;
        player->dest_x = player->x;
        player->dest_y = player->y;
        return 0;
    }
    player->x += route->first_dx;
    player->y += route->first_dy;
    int steps = 1;
    if (route->run_dx != 0 || route->run_dy != 0) {
        player->x += route->run_dx;
        player->y += route->run_dy;
        steps++;
    }
    player->is_running = steps == 2;
    player->dest_x = player->x;
    player->dest_y = player->y;
    return steps;
}

static inline int osrs_player_step_apply_explicit_move(
    const OsrsPlayerStepInput* input,
    OsrsPlayerMoveKind move_kind
) {
    Player* player = input->player;
    int target_x;
    int target_y;
    if (move_kind == OSRS_PLAYER_MOVE_ACTION) {
        int move_action = input->command.move_action;
        if (move_action <= 0 || move_action >= ENCOUNTER_MOVE_ACTIONS)
            return 0;
        target_x = player->x + ENCOUNTER_MOVE_TARGET_DX[move_action];
        target_y = player->y + ENCOUNTER_MOVE_TARGET_DY[move_action];
    } else if (move_kind == OSRS_PLAYER_MOVE_DESTINATION) {
        target_x = *input->dest_x;
        target_y = *input->dest_y;
        if (target_x < 0 || target_y < 0) return 0;
        if (player->x == target_x && player->y == target_y) {
            *input->dest_x = -1;
            *input->dest_y = -1;
            return 0;
        }
    } else {
        return 0;
    }
    EncounterRouteInput route_input = {
        .topology = input->arena.topology,
        .blockers = input->arena.blockers,
        .source_x = player->x,
        .source_y = player->y,
        .actor_size = 1,
        .target_x = target_x,
        .target_y = target_y,
        .target_size = 1,
        .target_kind = ENCOUNTER_ROUTE_TARGET_TILE,
        .movement_mode = input->arena.movement_mode,
        .cost_policy = move_kind == OSRS_PLAYER_MOVE_ACTION
            ? ENCOUNTER_ROUTE_COST_DIRECT
            : input->arena.destination_cost_policy,
    };
    EncounterRouteResult route = encounter_route_solve(&route_input);
    return osrs_player_step_apply_route(player, &route);
}

static inline int osrs_player_step_route_matches(
    const OsrsActorRouteCache* route,
    const Player* player,
    const OsrsAttackTarget* target,
    const OsrsEncounterArena* arena
) {
    return route->state != OSRS_INTERACTION_ROUTE_EMPTY &&
        route->topology_revision == arena->topology->revision &&
        route->blocker_revision == arena->blockers.revision &&
        route->actor_size == 1 &&
        route->movement_mode == arena->movement_mode &&
        route->cost_policy == arena->cost_policy &&
        route->target_x == target->x &&
        route->target_y == target->y &&
        route->target_size == target->size &&
        route->attack_range == target->attack_range &&
        route->expected_player_x == player->x &&
        route->expected_player_y == player->y;
}

static inline void osrs_player_step_build_attack_route(
    const OsrsPlayerStepInput* input,
    const OsrsAttackTarget* target
) {
    OsrsActorRouteCache* route = input->route_cache;
    route->topology_revision = input->arena.topology->revision;
    route->blocker_revision = input->arena.blockers.revision;
    route->actor_size = 1;
    route->movement_mode = input->arena.movement_mode;
    route->cost_policy = input->arena.cost_policy;
    route->target_x = target->x;
    route->target_y = target->y;
    route->target_size = target->size;
    route->attack_range = target->attack_range;
    route->expected_player_x = input->player->x;
    route->expected_player_y = input->player->y;
    route->planned_source_x = input->player->x;
    route->planned_source_y = input->player->y;
    route->waypoint_count = 0;
    route->waypoint_index = 0;

    EncounterRouteInput route_input = {
        .topology = input->arena.topology,
        .blockers = input->arena.blockers,
        .source_x = input->player->x,
        .source_y = input->player->y,
        .actor_size = 1,
        .target_x = target->x,
        .target_y = target->y,
        .target_size = target->size,
        .target_kind = ENCOUNTER_ROUTE_TARGET_ATTACK_RANGE,
        .attack_range = target->attack_range,
        .attack_geometry = input->arena.attack_geometry,
        .collision_map = input->arena.collision_map,
        .world_offset_x = input->arena.world_offset_x,
        .world_offset_y = input->arena.world_offset_y,
        .los_query = input->arena.los_query,
        .movement_mode = input->arena.movement_mode,
        .cost_policy = input->arena.cost_policy,
    };
    EncounterRouteResult result = encounter_route_solve(&route_input);
    if (result.outcome == ROUTE_REACHED_TARGET ||
            result.outcome == ROUTE_REACHED_FALLBACK) {
        route->waypoint_count = result.waypoint_count;
        memcpy(
            route->waypoint_x,
            result.waypoint_x,
            (size_t)result.waypoint_count * sizeof(route->waypoint_x[0]));
        memcpy(
            route->waypoint_y,
            result.waypoint_y,
            (size_t)result.waypoint_count * sizeof(route->waypoint_y[0]));
    }
    route->state = route->waypoint_count > 0
        ? OSRS_INTERACTION_ROUTE_READY
        : OSRS_INTERACTION_ROUTE_FAILED;
}

static inline int osrs_player_step_route_next_traversable(
    const OsrsPlayerStepInput* input
) {
    const OsrsActorRouteCache* route = input->route_cache;
    if (route->state != OSRS_INTERACTION_ROUTE_READY ||
            route->waypoint_index >= route->waypoint_count) {
        return 0;
    }
    int waypoint_x = route->waypoint_x[route->waypoint_index];
    int waypoint_y = route->waypoint_y[route->waypoint_index];
    int dx = (waypoint_x > input->player->x) -
        (waypoint_x < input->player->x);
    int dy = (waypoint_y > input->player->y) -
        (waypoint_y < input->player->y);
    EncounterRouteInput route_input = {
        .topology = input->arena.topology,
        .blockers = input->arena.blockers,
        .source_x = input->player->x,
        .source_y = input->player->y,
        .actor_size = 1,
        .target_x = waypoint_x,
        .target_y = waypoint_y,
        .target_size = 1,
        .target_kind = ENCOUNTER_ROUTE_TARGET_TILE,
        .movement_mode = input->arena.movement_mode,
        .cost_policy = input->arena.cost_policy,
    };
    return encounter_route_step_allowed(
        &route_input, input->player->x, input->player->y, dx, dy);
}

static inline int osrs_player_step_chase_target(
    const OsrsPlayerStepInput* input,
    const OsrsAttackTarget* target
) {
    Player* player = input->player;
    OsrsActorRouteCache* route = input->route_cache;
    int distance = encounter_rect_distance(
        player->x, player->y, 1,
        target->x, target->y, target->size);
    if (distance == 0) {
        osrs_actor_route_cache_clear(route);
        osrs_player_step_build_attack_route(input, target);
    }
    if (osrs_player_step_can_attack_target(input, target)) {
        osrs_actor_route_cache_clear(route);
        return 0;
    }

    int reroute =
        !osrs_player_step_route_matches(route, player, target, &input->arena);
    if (!reroute &&
            route->state == OSRS_INTERACTION_ROUTE_READY &&
            !osrs_player_step_route_next_traversable(input))
        reroute = 1;
    if (reroute)
        osrs_player_step_build_attack_route(input, target);
    if (route->state != OSRS_INTERACTION_ROUTE_READY ||
            !osrs_player_step_route_next_traversable(input)) {
        route->expected_player_x = player->x;
        route->expected_player_y = player->y;
        return 0;
    }

    int steps = 0;
    while (steps < 2 &&
            route->waypoint_index < route->waypoint_count &&
            !osrs_player_step_can_attack_target(input, target)) {
        int waypoint_x = route->waypoint_x[route->waypoint_index];
        int waypoint_y = route->waypoint_y[route->waypoint_index];
        if (player->x == waypoint_x && player->y == waypoint_y) {
            route->waypoint_index++;
            continue;
        }
        if (!osrs_player_step_route_next_traversable(input)) {
            osrs_actor_route_cache_clear(route);
            break;
        }
        player->x += (waypoint_x > player->x) - (waypoint_x < player->x);
        player->y += (waypoint_y > player->y) - (waypoint_y < player->y);
        steps++;
        if (player->x == waypoint_x && player->y == waypoint_y)
            route->waypoint_index++;
    }
    player->is_running = steps == 2;
    player->dest_x = player->x;
    player->dest_y = player->y;
    route->expected_player_x = player->x;
    route->expected_player_y = player->y;
    return steps > 0;
}

static inline OsrsPlayerStepResult osrs_encounter_player_step(
    const OsrsPlayerStepInput* input
) {
    osrs_player_step_require_input(input);
    if (!input->route_cache) {
        fprintf(stderr, "OSRS player step missing actor route cache\n");
        abort();
    }

    OsrsPlayerStepResult result = {
        .target_slot = -1,
    };
    Player* player = input->player;
    OsrsInteraction* interaction = input->interaction;

    if (input->blocked_ticks > 0) {
        result.interaction_active = osrs_interaction_active(interaction);
        result.target_slot = result.interaction_active ? interaction->target_slot : -1;
        return result;
    }

    if (input->command.kind == OSRS_PLAYER_CMD_TARGET) {
        OsrsAttackTarget target;
        if (osrs_player_step_lookup_target(input, input->command.target_slot, &target)) {
            osrs_interaction_set(interaction, input->command.target_slot);
        } else {
            osrs_interaction_clear(interaction);
        }
        if (input->dest_x) *input->dest_x = -1;
        if (input->dest_y) *input->dest_y = -1;
    } else if (input->command.kind == OSRS_PLAYER_CMD_MOVE) {
        osrs_interaction_check_interrupt(interaction, OSRS_IACT_MOVE);
    }

    OsrsAttackTarget target;
    int has_target = 0;
    if (osrs_interaction_active(interaction)) {
        has_target = osrs_player_step_lookup_target(
            input, interaction->target_slot, &target);
        if (!has_target) {
            osrs_interaction_clear(interaction);
        }
    }

    if (input->command.kind == OSRS_PLAYER_CMD_MOVE &&
            !osrs_interaction_active(interaction)) {
        result.moved =
            osrs_player_step_apply_explicit_move(input, input->command.move_kind) > 0;
        result.explicit_moved = result.moved;
    } else if (osrs_interaction_active(interaction) && has_target) {
        result.moved = osrs_player_step_chase_target(input, &target) > 0;
        result.chased_target = result.moved;
    }

    if (osrs_interaction_active(interaction)) {
        has_target = osrs_player_step_lookup_target(
            input, interaction->target_slot, &target);
        if (has_target) {
            result.interaction_active = 1;
            result.target_slot = interaction->target_slot;
            result.can_attack = osrs_player_step_can_attack_target(input, &target);
        } else {
            osrs_interaction_clear(interaction);
        }
    }

    player->dest_x = player->x;
    player->dest_y = player->y;
    return result;
}

#endif
