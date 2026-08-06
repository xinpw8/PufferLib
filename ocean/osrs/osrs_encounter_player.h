#ifndef OSRS_ENCOUNTER_PLAYER_H
#define OSRS_ENCOUNTER_PLAYER_H

#include "osrs_encounter.h"
#include "osrs_interaction.h"

typedef enum {
    OSRS_PLAYER_MOVE_NONE = 0,
    OSRS_PLAYER_MOVE_ACTION,
    OSRS_PLAYER_MOVE_DESTINATION,
} OsrsPlayerMoveKind;

/**
 * One click per tick. The engine cannot both start an attack and walk somewhere
 * else on the same tick: a ground click cancels the entity interaction, an entity
 * click cancels the walk. TARGET and MOVE are therefore mutually exclusive by
 * construction rather than by caller discipline.
 *
 * A walk already in flight is state, not a command. It continues under CMD_NONE
 * and is cancelled by either new command.
 */
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
    const CollisionMap* collision_map;
    int world_offset_x;
    int world_offset_y;
    encounter_walkable_fn is_walkable;
    void* walkable_ctx;
    pathfind_blocked_fn extra_blocked;
    void* blocked_ctx;
    const OsrsLosQuery* los_query;
    int arena_base_x;
    int arena_base_y;
    int arena_w;
    int arena_h;
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
    if (!input || !input->player || !input->interaction || !input->arena.is_walkable) {
        fprintf(stderr, "osrs player step input is missing required fields\n");
        abort();
    }
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

static inline int osrs_player_step_can_attack_target(
    const OsrsPlayerStepInput* input,
    const OsrsAttackTarget* target
) {
    return encounter_player_can_attack(
        input->player->x,
        input->player->y,
        target->x,
        target->y,
        target->size,
        target->attack_range,
        input->arena.los_query);
}

static inline int osrs_player_step_apply_explicit_move(
    const OsrsPlayerStepInput* input,
    OsrsPlayerMoveKind move_kind
) {
    Player* player = input->player;
    if (move_kind == OSRS_PLAYER_MOVE_ACTION) {
        int move_action = input->command.move_action;
        if (move_action <= 0 || move_action >= ENCOUNTER_MOVE_ACTIONS)
            return 0;
        return encounter_move_to_target(
            player,
            ENCOUNTER_MOVE_TARGET_DX[move_action],
            ENCOUNTER_MOVE_TARGET_DY[move_action],
            input->arena.is_walkable,
            input->arena.walkable_ctx);
    }

    if (move_kind != OSRS_PLAYER_MOVE_DESTINATION)
        return 0;
    return encounter_move_toward_dest(
        player,
        input->dest_x,
        input->dest_y,
        input->arena.collision_map,
        input->arena.world_offset_x,
        input->arena.world_offset_y,
        input->arena.is_walkable,
        input->arena.walkable_ctx,
        input->arena.extra_blocked,
        input->arena.blocked_ctx,
        input->arena.arena_base_x,
        input->arena.arena_base_y,
        input->arena.arena_w,
        input->arena.arena_h);
}

static inline int osrs_player_step_chase_target(
    const OsrsPlayerStepInput* input,
    const OsrsAttackTarget* target
) {
    return encounter_chase_attack_target(
        input->player,
        target->x,
        target->y,
        target->size,
        target->attack_range,
        input->arena.collision_map,
        input->arena.world_offset_x,
        input->arena.world_offset_y,
        input->arena.is_walkable,
        input->arena.walkable_ctx,
        input->arena.extra_blocked,
        input->arena.blocked_ctx,
        input->arena.los_query,
        input->arena.arena_base_x,
        input->arena.arena_base_y,
        input->arena.arena_w,
        input->arena.arena_h);
}

static inline OsrsPlayerStepResult osrs_encounter_player_step(
    const OsrsPlayerStepInput* input
) {
    osrs_player_step_require_input(input);

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
        /* an entity click cancels any walk already in flight */
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

    /* an active interaction always chases. the only way to walk elsewhere is a
       MOVE command, and that cleared the interaction above, so the two are
       mutually exclusive by construction rather than by a tie-break policy. */
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
