#ifndef OSRS_ENCOUNTER_PLAYER_H
#define OSRS_ENCOUNTER_PLAYER_H

#include "osrs_encounter.h"
#include "osrs_interaction.h"

typedef enum {
    OSRS_PLAYER_MOVE_NONE = 0,
    OSRS_PLAYER_MOVE_ACTION,
    OSRS_PLAYER_MOVE_DESTINATION,
} OsrsPlayerMoveKind;

typedef enum {
    OSRS_PLAYER_TARGET_MOVE_CHASE = 0,
    OSRS_PLAYER_TARGET_MOVE_EXPLICIT_FIRST,
} OsrsPlayerTargetMovePolicy;

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
    int has_new_target;
    int new_target_slot;
    OsrsPlayerMoveKind move_kind;
    OsrsPlayerTargetMovePolicy target_move_policy;
    int move_action;
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
    if ((input->move_kind == OSRS_PLAYER_MOVE_DESTINATION ||
            input->move_kind == OSRS_PLAYER_MOVE_ACTION) &&
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
    const OsrsPlayerStepInput* input
) {
    Player* player = input->player;
    if (input->move_kind == OSRS_PLAYER_MOVE_ACTION) {
        if (input->move_action <= 0 || input->move_action >= ENCOUNTER_MOVE_ACTIONS)
            return 0;
        return encounter_move_to_target(
            player,
            ENCOUNTER_MOVE_TARGET_DX[input->move_action],
            ENCOUNTER_MOVE_TARGET_DY[input->move_action],
            input->arena.is_walkable,
            input->arena.walkable_ctx);
    }

    if (input->move_kind != OSRS_PLAYER_MOVE_DESTINATION)
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

    if (input->has_new_target) {
        OsrsAttackTarget target;
        if (osrs_player_step_lookup_target(input, input->new_target_slot, &target)) {
            osrs_interaction_set(interaction, input->new_target_slot);
        } else {
            osrs_interaction_clear(interaction);
        }
    } else if (input->move_kind != OSRS_PLAYER_MOVE_NONE) {
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

    if (input->move_kind != OSRS_PLAYER_MOVE_NONE &&
            (!osrs_interaction_active(interaction) ||
             input->target_move_policy == OSRS_PLAYER_TARGET_MOVE_EXPLICIT_FIRST)) {
        result.moved = osrs_player_step_apply_explicit_move(input) > 0;
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
