#line 2173 "encounter_inferno.h"
static int inf_in_arena(int x, int y) {
    return x >= INF_ARENA_MIN_X && x <= INF_ARENA_MAX_X &&
           y >= INF_ARENA_MIN_Y && y <= INF_ARENA_MAX_Y;
}

static int inf_footprint_in_arena(int x, int y, int size) {
    return x >= INF_ARENA_MIN_X && x + size - 1 <= INF_ARENA_MAX_X &&
           y >= INF_ARENA_MIN_Y && y + size - 1 <= INF_ARENA_MAX_Y;
}

static int inf_blocked_by_pillar_size1(InfernoState* s, int x, int y) {
    for (int p = 0; p < INF_NUM_PILLARS; p++) {
        if (!s->pillars[p].active) continue;
        int px = s->pillars[p].x;
        int py = s->pillars[p].y;
        if (x >= px && x < px + INF_PILLAR_SIZE &&
                y >= py && y < py + INF_PILLAR_SIZE)
            return 1;
    }
    return 0;
}

static int inf_blocked_by_pillar(InfernoState* s, int x, int y, int size) {
    if (size == 1) return inf_blocked_by_pillar_size1(s, x, y);
    for (int p = 0; p < INF_NUM_PILLARS; p++) {
        if (!s->pillars[p].active) continue;
        if (los_aabb_overlap(x, y, size,
                             s->pillars[p].x, s->pillars[p].y, INF_PILLAR_SIZE))
            return 1;
    }
    return 0;
}

typedef struct {
    InfernoState* s;
    const InfernoContext* ctx;
} InfWalkCtx;

static int inf_pathfind_blocked(void* data, int abs_x, int abs_y) {
    InfWalkCtx* wc = (InfWalkCtx*)data;
    InfernoState* s = wc->s;
    const InfernoContext* ctx = wc->ctx;
    int lx = abs_x - ctx->world_offset_x;
    int ly = abs_y - ctx->world_offset_y;
    return inf_blocked_by_pillar(s, lx, ly, 1);
}

static int inf_collision_flags_blocked(
    uint8_t flags[INF_ARENA_WIDTH][INF_ARENA_HEIGHT], int x, int y, int size
) {
    if (size == 1) {
        int gx, gy;
        return inf_grid_index(x, y, &gx, &gy) && flags[gx][gy];
    }
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int gx, gy;
            if (inf_grid_index(x + dx, y + dy, &gx, &gy) && flags[gx][gy])
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
    if (size == 1) {
        if (!inf_in_arena(x, y)) return 1;
        if (inf_blocked_by_pillar_size1(s, x, y)) return 1;
        if (ctx->collision_map &&
                !collision_tile_walkable(
                    ctx->collision_map, 0,
                    x + ctx->world_offset_x,
                    y + ctx->world_offset_y))
            return 1;
        return 0;
    }
    if (!inf_footprint_in_arena(x, y, size)) return 1;
    if (inf_blocked_by_pillar(s, x, y, size)) return 1;
    if (ctx->collision_map) {
        for (int dx = 0; dx < size; dx++) {
            for (int dy = 0; dy < size; dy++) {
                if (!collision_tile_walkable(
                        ctx->collision_map, 0,
                        x + dx + ctx->world_offset_x,
                        y + dy + ctx->world_offset_y))
                    return 1;
            }
        }
    }
    return 0;
}

static int inf_npc_blocked(void* ctx, int x, int y, int size) {
    InfMoveCtx* mc = (InfMoveCtx*)ctx;
    InfernoState* s = mc->s;
    (void)mc->self_idx;
    if (inf_npc_environment_blocked_ctx(s, mc->ctx, x, y, size)) return 1;
    if (inf_collision_flags_blocked(s->player_collision_flags, x, y, size)) return 1;
    return inf_collision_flags_blocked(s->npc_collision_flags, x, y, size);
}

static int inf_npc_blocked_ignore_player(void* ctx, int x, int y, int size) {
    InfMoveCtx* mc = (InfMoveCtx*)ctx;
    InfernoState* s = mc->s;
    (void)mc->self_idx;
    if (inf_npc_environment_blocked_ctx(s, mc->ctx, x, y, size)) return 1;
    return inf_collision_flags_blocked(s->npc_collision_flags, x, y, size);
}

static int inf_npc_overlap_hold(void* ctx) {
    const InfMoveCtx* mc = (const InfMoveCtx*)ctx;
    const InfernoState* s = mc->s;
    return s->player_last_interaction_age == 0 &&
           s->player_last_interaction_target_slot == mc->self_idx;
}

static int inf_tile_walkable(void* ctx, int x, int y);

static int inf_npc_environment_blocked(InfernoState* s, int x, int y, int size) {
    return inf_npc_environment_blocked_ctx(s, inf_legacy_context(), x, y, size);
}

static int inf_npc_terrain_blocked(InfernoState* s, int x, int y, int size) {
    return inf_npc_environment_blocked_ctx(s, inf_legacy_context(), x, y, size);
}

static void inf_npc_move_ctx(InfernoState* s, const InfernoContext* ctx, int idx) {
    InfNPC* npc = &s->npcs[idx];
    if (!npc->active) return;
    if (npc->stun_timer > 0) return;
    if (npc->dig_freeze_timer > 0) return;
    if (npc->frozen_ticks > 0) return;

    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    if (!stats->can_move) return;
    int uses_collision_flag = inf_npc_sets_collision_flag(npc->type);
    if (uses_collision_flag)
        inf_unstamp_npc_collision_footprint(s, npc->x, npc->y, npc->size);

    if (npc->type != INF_NPC_NIBBLER) {
        InfMoveCtx mc = { s, ctx, idx };
        int stepped = encounter_npc_step_out_from_under(
            &npc->x, &npc->y, npc->size,
            s->player.x, s->player.y,
            inf_npc_blocked_ignore_player, &mc, inf_npc_overlap_hold, &s->rng_state);
        if (stepped == ENCOUNTER_NPC_UNDER_PLAYER_MOVED) {
            npc->moved_this_tick = 1;
            inf_invalidate_npc_los_cache(s, idx);
            if (uses_collision_flag)
                inf_stamp_npc_collision_footprint(s, npc->x, npc->y, npc->size);
            return;
        }
        if (stepped == ENCOUNTER_NPC_UNDER_PLAYER_HELD ||
                stepped == ENCOUNTER_NPC_UNDER_PLAYER_BLOCKED) {
            if (uses_collision_flag)
                inf_stamp_npc_collision_footprint(s, npc->x, npc->y, npc->size);
            return;
        }
    }

    InfTargetArea target = inf_npc_current_target_area(s, npc);
    if (target.is_player && npc->aggro_target >= 0) npc->aggro_target = -1;
    int tx = target.x;
    int ty = target.y;
    int target_size = target.size;
    npc->target_x = tx;
    npc->target_y = ty;

    if (npc->type != INF_NPC_NIBBLER) {
        int has_los = target.is_player ?
            inf_npc_has_los(s, idx) :
            inf_npc_has_los_direct(s, idx);
        if (!target.is_player)
            s->npc_los_cache[idx] = (int8_t)has_los;
        if (has_los) {
            if (uses_collision_flag)
                inf_stamp_npc_collision_footprint(s, npc->x, npc->y, npc->size);
            return;
        }
    }
    if (npc->attack_timer > stats->attack_speed) {
        if (uses_collision_flag)
            inf_stamp_npc_collision_footprint(s, npc->x, npc->y, npc->size);
        return;
    }

    int ox = npc->x, oy = npc->y;
    InfMoveCtx mc = { s, ctx, idx };
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
        &npc->x, &npc->y, tx, ty, npc->size, target_size,
        step_policy, blocked, &mc, inf_npc_overlap_hold, &s->rng_state);
    if (npc->x != ox || npc->y != oy) {
        npc->moved_this_tick = 1;
        inf_invalidate_npc_los_cache(s, idx);
    }
    if (uses_collision_flag)
        inf_stamp_npc_collision_footprint(s, npc->x, npc->y, npc->size);
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
            int ox = npc->x, oy = npc->y;
            if (inf_npc_sets_collision_flag(npc->type))
                inf_unstamp_npc_collision_footprint(s, ox, oy, npc->size);
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
                inf_stamp_npc_collision_footprint(s, npc->x, npc->y, npc->size);
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

    if (!inf_npc_has_los(s, idx)) {
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
