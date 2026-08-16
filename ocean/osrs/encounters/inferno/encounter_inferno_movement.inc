#line 2173 "encounter_inferno.h"

typedef struct {
    InfernoState* s;
    const InfernoContext* ctx;
} InfWalkCtx;


static int inf_npc_collision_flags_blocked(
    const InfernoState* s,
    int x,
    int y,
    int size
) {
    if (size == 1) {
        int grid_x;
        int grid_y;
        return inf_grid_index(x, y, &grid_x, &grid_y) &&
            s->npc_collision_flags[grid_x][grid_y] != 0;
    }
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int grid_x;
            int grid_y;
            if (inf_grid_index(
                    x + dx, y + dy, &grid_x, &grid_y) &&
                    s->npc_collision_flags[grid_x][grid_y] != 0)
                return 1;
        }
    }
    return 0;
}

typedef struct {
    InfernoState* s;
    const InfernoContext* ctx;
    int self_idx;
} InfMoveCtx;

static int inf_npc_environment_blocked_ctx(
    InfernoState* s,
    const InfernoContext* ctx,
    int x,
    int y,
    int size
) {
    return inf_footprint_blocked_ctx(s, ctx, x, y, size);
}

static int inf_npc_blocked(void* ctx, int x, int y, int size) {
    InfMoveCtx* mc = (InfMoveCtx*)ctx;
    InfernoState* s = mc->s;
    if (inf_npc_environment_blocked_ctx(
            s, mc->ctx, x, y, size))
        return 1;
    if (los_aabb_overlap(
            x, y, size, s->player.x, s->player.y, 1))
        return 1;
    return inf_npc_collision_flags_blocked(s, x, y, size);
}

static int inf_npc_blocked_ignore_player(void* ctx, int x, int y, int size) {
    InfMoveCtx* mc = (InfMoveCtx*)ctx;
    InfernoState* s = mc->s;
    if (inf_npc_environment_blocked_ctx(
            s, mc->ctx, x, y, size))
        return 1;
    return inf_npc_collision_flags_blocked(s, x, y, size);
}

static int inf_npc_overlap_hold(void* ctx) {
    const InfMoveCtx* mc = (const InfMoveCtx*)ctx;
    const InfernoState* s = mc->s;
    return s->player_last_interaction_age == 0 &&
           s->player_last_interaction_target_slot == mc->self_idx;
}

static int inf_tile_walkable(void* ctx, int x, int y);


static void inf_npc_move_ctx(
    InfernoState* s,
    const InfernoContext* ctx,
    int idx
) {
    InfNPC* npc = &s->npcs[idx];
    if (!npc->active) return;
    if (npc->stun_timer > 0) return;
    if (npc->dig_freeze_timer > 0) return;
    if (npc->frozen_ticks > 0) return;

    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    if (!stats->can_move) return;
    int uses_collision_flag = inf_npc_sets_collision_flag(npc->type);
    if (uses_collision_flag)
        inf_unstamp_npc_collision_footprint(
            s, npc->x, npc->y, npc->size);

    do {
        if (npc->type != INF_NPC_NIBBLER) {
            InfMoveCtx move_ctx = {s, ctx, idx};
            int stepped = encounter_npc_step_out_from_under(
                &npc->x,
                &npc->y,
                npc->size,
                s->player.x,
                s->player.y,
                inf_npc_blocked_ignore_player,
                &move_ctx,
                inf_npc_overlap_hold,
                &s->rng_state);
            if (stepped == ENCOUNTER_NPC_UNDER_PLAYER_MOVED) {
                npc->moved_this_tick = 1;
                break;
            }
            if (stepped == ENCOUNTER_NPC_UNDER_PLAYER_HELD ||
                    stepped == ENCOUNTER_NPC_UNDER_PLAYER_BLOCKED)
                break;
        }

        InfTargetArea target = inf_npc_current_target_area(s, npc);
        if (target.is_player && npc->aggro_target >= 0)
            npc->aggro_target = -1;
        int target_x = target.x;
        int target_y = target.y;
        int target_size = target.size;
        npc->target_x = target_x;
        npc->target_y = target_y;

        if (npc->type != INF_NPC_NIBBLER &&
                inf_npc_has_los_direct_ctx(s, ctx, idx))
            break;
        if (npc->attack_timer > stats->attack_speed)
            break;

        int old_x = npc->x;
        int old_y = npc->y;
        InfMoveCtx move_ctx = {s, ctx, idx};
        EncounterNpcStepPolicy step_policy;
        encounter_npc_blocked_fn blocked;
        if (target.is_player) {
            step_policy = stats->attack_range == 1
                ? ENCOUNTER_NPC_STEP_OSRS_AGGRO_STOP_AT_MELEE
                : ENCOUNTER_NPC_STEP_OSRS_AGGRO_TARGET;
            blocked = inf_npc_blocked_ignore_player;
        } else {
            step_policy = stats->attack_range == 1
                ? ENCOUNTER_NPC_STEP_STOP_AT_MELEE
                : ENCOUNTER_NPC_STEP_TRAVEL_TARGET;
            blocked = inf_npc_blocked;
        }
        encounter_npc_step_toward_policy(
            &npc->x,
            &npc->y,
            target_x,
            target_y,
            npc->size,
            target_size,
            step_policy,
            blocked,
            &move_ctx,
            inf_npc_overlap_hold,
            &s->rng_state);
        if (npc->x != old_x || npc->y != old_y)
            npc->moved_this_tick = 1;
    } while (0);

    if (uses_collision_flag)
        inf_stamp_npc_collision_footprint(
            s, npc->x, npc->y, npc->size);
}

static void inf_npc_move(InfernoState* s, int idx) {
    inf_npc_move_ctx(s, inf_legacy_context(), idx);
}

static void inf_meleer_dig_check_ctx(InfernoState* s, const InfernoContext* ctx, int idx) {
    InfNPC* npc = &s->npcs[idx];
    if (npc->type != INF_NPC_MELEER || !npc->active) return;
    if (npc->dig_freeze_timer > 0) {
        npc->dig_freeze_timer--;
        if (npc->dig_freeze_timer == 0 && npc->dig_attack_delay == 0) {
            int old_x = npc->x;
            int old_y = npc->y;
            if (inf_npc_sets_collision_flag(npc->type))
                inf_unstamp_npc_collision_footprint(
                    s, old_x, old_y, npc->size);
            int candidates[5][2] = {
                { s->player.x - npc->size + 1, s->player.y - npc->size + 1 },
                { s->player.x,                 s->player.y                 },
                { s->player.x - npc->size + 1, s->player.y                 },
                { s->player.x,                 s->player.y - npc->size + 1 },
                { s->player.x - 1,             s->player.y - 1             },
            };
            int landing_x = candidates[4][0];
            int landing_y = candidates[4][1];
            for (int i = 0; i < 4; i++) {
                if (inf_npc_environment_blocked_ctx(
                        s, ctx, candidates[i][0], candidates[i][1], npc->size))
                    continue;
                landing_x = candidates[i][0];
                landing_y = candidates[i][1];
                break;
            }
            npc->x = landing_x;
            npc->y = landing_y;
            if (inf_npc_sets_collision_flag(npc->type))
                inf_stamp_npc_collision_footprint(
                    s, npc->x, npc->y, npc->size);
            npc->stun_timer = 2;
            npc->dig_attack_delay = 6;
            npc->no_los_ticks = 0;
        }
        return;
    }
    if (npc->dig_attack_delay > 0) {
        npc->dig_attack_delay--;
        return;
    }

    if (!inf_npc_has_los_ctx(s, ctx, idx)) {
        npc->no_los_ticks++;
    } else {
        npc->no_los_ticks = 0;
        return;
    }

    if (npc->no_los_ticks >= 50) {
        npc->dig_freeze_timer = 6;
    } else if (npc->no_los_ticks >= 38) {
        if (encounter_rand_int(&s->rng_state, 10) == 0) {
            npc->dig_freeze_timer = 6;
        }
    }
}

static void inf_meleer_dig_check(InfernoState* s, int idx) {
    inf_meleer_dig_check_ctx(s, inf_legacy_context(), idx);
}
