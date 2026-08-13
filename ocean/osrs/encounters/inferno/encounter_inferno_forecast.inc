#line 5216 "encounter_inferno.h"
#define INF_PLAYER_OBS_SIZE 75
#define INF_PILLAR_OBS_SIZE (INF_NUM_PILLARS * 5)
#define INF_BASE_NPC_OBS_SIZE 896
#define INF_TOTAL_NPC_OBS_SIZE INF_BASE_NPC_OBS_SIZE
#define INF_STEP_OUT_FORECAST_HORIZON 4
#define INF_STEP_OUT_FORECAST_ACTION_FEATURES 8
#define INF_STEP_OUT_FORECAST_OBS_SIZE (ENCOUNTER_MOVE_ACTIONS * INF_STEP_OUT_FORECAST_ACTION_FEATURES)
#define INF_FEATURES_PER_HIT 5
#define INF_SPARK_OBS_SLOTS INF_MAX_PENDING_SPARKS
#define INF_FEATURES_PER_SPARK 7
#define INF_PENDING_HIT_OBS_SIZE (INF_FEATURES_PER_HIT * ENCOUNTER_MAX_PENDING_HITS)
#define INF_PENDING_SPARK_OBS_SIZE (INF_FEATURES_PER_SPARK * INF_SPARK_OBS_SLOTS)
#define INF_NUM_OBS (INF_PLAYER_OBS_SIZE + INF_PILLAR_OBS_SIZE + INF_TOTAL_NPC_OBS_SIZE + INF_STEP_OUT_FORECAST_OBS_SIZE + INF_PENDING_HIT_OBS_SIZE + INF_PENDING_SPARK_OBS_SIZE)

typedef struct {
    int melee_count;
    int ranged_count;
    int magic_count;
    int blob_scan_count;
    int ranger_count;
    int mager_count;
    int max_hit;
} InfStepOutForecastTick;

typedef struct {
    int valid;
    int land_x;
    int land_y;
    InfStepOutForecastTick ticks[INF_STEP_OUT_FORECAST_HORIZON];
    int same_tick_mixed_style_conflict;
    int ranger_mager_offtick_opportunity;
    int melee_fallback_exposure;
} InfStepOutForecastAction;

typedef struct {
    InfStepOutForecastAction actions[ENCOUNTER_MOVE_ACTIONS];
} InfStepOutForecast;

typedef struct {
    int action_feature_mismatches;
    int tick_feature_mismatches;
    int dangerous_false_negatives;
    int dangerous_false_positives;
    int exact_safe_fast_dangerous;
    int exact_dangerous_actions;
    int fast_dangerous_actions;
    int sampled_actions;
    int max_hit_abs_error_sum;
    int max_hit_abs_error_max;
} InfStepOutForecastOracleDiff;

typedef struct {
    int active;
    InfNPCType type;
    int x;
    int y;
    int size;
    int hp;
    int attack_timer;
    int stun_timer;
    int frozen_ticks;
    int dig_freeze_timer;
    int dig_attack_delay;
    int blob_scanned_prayer;
    int had_los_last_tick;
    int jad_attack_style;
    int attack_style;
    int aggro_target;
    int los_to_player_cache;
} InfForecastNpcLocal;

typedef struct {
    const InfernoState* state;
    const InfernoContext* ctx;
    uint8_t (*npc_flags)[INF_ARENA_HEIGHT];
    int player_x;
    int player_y;
} InfForecastMoveCtx;

static float inf_zuk_attack_timer_obs(const InfernoState* s) {
    if (!inf_is_final_wave(s)) return 0.0f;

    int min_timer = 999;
    int zuk_idx = inf_find_live_zuk_idx(s);
    if (zuk_idx >= 0) {
        int timer = s->npcs[zuk_idx].attack_timer;
        if (timer < 0) timer = 0;
        min_timer = timer;
    }

    for (int h = 0; h < s->player_pending_hits.count; h++) {
        const EncounterPendingHit* hit = &s->player_pending_hits.hits[h];
        if (hit->source_npc_type != INF_NPC_ZUK) continue;
        if (hit->ticks_remaining < min_timer)
            min_timer = hit->ticks_remaining;
    }

    return (min_timer < 999) ? (float)min_timer / 10.0f : 0.0f;
}

static int inf_npc_targets_player_for_obs(const InfernoState* s, const InfNPC* npc) {
    InfTargetArea target = inf_npc_current_target_area(s, npc);
    return target.is_player;
}

static int inf_step_out_forecast_action_valid(InfernoState* s, int action) {
    if (action == 0) return 1;
    int nx = s->player.x + ENCOUNTER_MOVE_TARGET_DX[action];
    int ny = s->player.y + ENCOUNTER_MOVE_TARGET_DY[action];
    return inf_in_arena(nx, ny) && !inf_blocked_by_pillar(s, nx, ny, 1);
}

static void inf_step_out_forecast_action_landing_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    int action_idx,
    Player* moved,
    int* valid
) {
    *moved = s->player;
    *valid = inf_step_out_forecast_action_valid((InfernoState*)s, action_idx);
    if (action_idx == 0) return;
    InfWalkCtx walk_ctx = { (InfernoState*)s, ctx };
    encounter_move_to_target(
        moved,
        ENCOUNTER_MOVE_TARGET_DX[action_idx],
        ENCOUNTER_MOVE_TARGET_DY[action_idx],
        inf_tile_walkable,
        &walk_ctx);
}

static Player inf_step_out_forecast_set_action_landing_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    int action_idx,
    InfStepOutForecastAction* action
) {
    Player moved;
    int valid;
    inf_step_out_forecast_action_landing_ctx(
        s, ctx, action_idx, &moved, &valid);
    action->valid = valid;
    action->land_x = moved.x;
    action->land_y = moved.y;
    return moved;
}

static int inf_step_out_forecast_duplicate_landing(
    const InfStepOutForecast* forecast,
    int action_idx
) {
    const InfStepOutForecastAction* action = &forecast->actions[action_idx];
    for (int prior_idx = 0; prior_idx < action_idx; prior_idx++) {
        const InfStepOutForecastAction* prior = &forecast->actions[prior_idx];
        if (prior->valid &&
                prior->land_x == action->land_x &&
                prior->land_y == action->land_y) {
            return prior_idx;
        }
    }
    return -1;
}

static int inf_step_out_forecast_prepare_unique_action_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    InfStepOutForecast* out,
    int action_idx,
    int forecast_slot_count
) {
    InfStepOutForecastAction* action = &out->actions[action_idx];
    Player moved = inf_step_out_forecast_set_action_landing_ctx(
        s, ctx, action_idx, action);
    if (!action->valid || forecast_slot_count == 0) return 0;

    int duplicate_idx = inf_step_out_forecast_duplicate_landing(
        out, action_idx);
    if (duplicate_idx >= 0) {
        int valid = action->valid;
        *action = out->actions[duplicate_idx];
        action->valid = valid;
        action->land_x = moved.x;
        action->land_y = moved.y;
        return 0;
    }
    return 1;
}

static void inf_step_out_forecast_record_style_mask(
    InfStepOutForecastAction* action,
    int tick_idx,
    InfNPCType type,
    const InfNPCStats* stats,
    int style_mask
) {
    if (tick_idx < 0 || tick_idx >= INF_STEP_OUT_FORECAST_HORIZON) return;
    InfStepOutForecastTick* tick = &action->ticks[tick_idx];
    int styles = 0;
    if (style_mask & INF_STYLE_MASK_MELEE) {
        tick->melee_count++;
        int max_hit = inf_npc_style_max_hit(stats, ATTACK_STYLE_MELEE);
        if (max_hit > tick->max_hit) tick->max_hit = max_hit;
        styles++;
    }
    if (style_mask & INF_STYLE_MASK_RANGED) {
        tick->ranged_count++;
        int max_hit = inf_npc_style_max_hit(stats, ATTACK_STYLE_RANGED);
        if (max_hit > tick->max_hit) tick->max_hit = max_hit;
        styles++;
    }
    if (style_mask & INF_STYLE_MASK_MAGIC) {
        tick->magic_count++;
        int max_hit = inf_npc_style_max_hit(stats, ATTACK_STYLE_MAGIC);
        if (max_hit > tick->max_hit) tick->max_hit = max_hit;
        styles++;
    }
    if (type == INF_NPC_RANGER) tick->ranger_count++;
    if (type == INF_NPC_MAGER) tick->mager_count++;
    if (styles >= 2 || ((tick->melee_count > 0) +
            (tick->ranged_count > 0) + (tick->magic_count > 0)) >= 2) {
        action->same_tick_mixed_style_conflict = 1;
    }
    if ((style_mask & INF_STYLE_MASK_MELEE) &&
            (type == INF_NPC_RANGER || type == INF_NPC_MAGER)) {
        action->melee_fallback_exposure = 1;
    }
}

static void inf_step_out_forecast_record_blob_scan(
    InfStepOutForecastAction* action, int tick_idx
) {
    if (tick_idx < 0 || tick_idx >= INF_STEP_OUT_FORECAST_HORIZON) return;
    action->ticks[tick_idx].blob_scan_count++;
}

static int inf_forecast_jad_unknown_style_mask(const InfNPC* npc) {
    if (npc->type != INF_NPC_JAD) return 0;
    if (inf_npc_jad_const(npc)->attack_style != ATTACK_STYLE_NONE) return 0;
    return INF_STYLE_MASK_RANGED | INF_STYLE_MASK_MAGIC;
}

static AttackStyle inf_forecast_blob_scan_style(OverheadPrayer prayer) {
    if (prayer == PRAYER_PROTECT_MAGIC) return ATTACK_STYLE_RANGED;
    if (prayer == PRAYER_PROTECT_RANGED) return ATTACK_STYLE_MAGIC;
    return ATTACK_STYLE_RANGED;
}

static void inf_step_out_forecast_npc_attack(
    InfernoState* s,
    int idx,
    InfStepOutForecastAction* action,
    int tick_idx
) {
    InfNPC* npc = &s->npcs[idx];
    if (!npc->active || npc->death_ticks > 0 || npc->hp <= 0) return;
    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    if (npc->attack_timer > 0) npc->attack_timer--;
    if (npc->stun_timer > 0) { npc->stun_timer--; return; }
    if (npc->dig_freeze_timer > 0 || npc->dig_attack_delay > 0) return;
    if (npc->type == INF_NPC_ZUK_SHIELD || npc->type == INF_NPC_NIBBLER)
        return;
    if (npc->aggro_target >= 0) return;
    if (!inf_npc_targets_player_for_obs(s, npc)) return;

    int has_los_now = 0;
    if (npc->type == INF_NPC_BLOB && stats->attack_range > 1) {
        has_los_now = inf_npc_has_los(s, idx);
        if (npc->blob_scanned_prayer < 0 &&
            has_los_now &&
            !npc->had_los_last_tick) {
            npc->blob_scanned_prayer = (int)s->player.prayer;
            npc->attack_style = inf_forecast_blob_scan_style(s->player.prayer);
            npc->had_los_last_tick = has_los_now;
            npc->attack_timer = stats->attack_speed;
            inf_step_out_forecast_record_blob_scan(action, tick_idx);
            return;
        }
        npc->had_los_last_tick = has_los_now;
    }
    if (npc->attack_timer > 0) return;

    if (npc->type != INF_NPC_BLOB && stats->attack_range > 1)
        has_los_now = inf_npc_has_los(s, idx);

    if (stats->attack_range > 1 &&
        !(npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer >= 0) &&
        !has_los_now) return;

    int dist = encounter_dist_to_npc(
        s->player.x, s->player.y, npc->x, npc->y, npc->size);
    if (dist == 0 || dist > stats->attack_range) return;
    if (stats->attack_range == 1 &&
            !inf_npc_melee_contact_allows_attack_at_tile(
                s->player.x, s->player.y, npc->type,
                npc->x, npc->y, npc->size, dist))
        return;

    if (npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer < 0) {
        npc->blob_scanned_prayer = (int)s->player.prayer;
        npc->attack_style = inf_forecast_blob_scan_style(s->player.prayer);
        npc->attack_timer = stats->attack_speed;
        inf_step_out_forecast_record_blob_scan(action, tick_idx);
        return;
    }

    int style_mask = inf_forecast_jad_unknown_style_mask(npc);
    if (style_mask == 0) {
        int planned_style = inf_npc_planned_style_for_current_state(npc);
        if (planned_style == ATTACK_STYLE_NONE) return;
        style_mask = inf_attack_style_options_mask(
            s, npc, stats, planned_style, dist);
    }
    if (style_mask == 0) return;

    inf_step_out_forecast_record_style_mask(
        action, tick_idx, npc->type, stats, style_mask);

    if (npc->type == INF_NPC_BLOB)
        npc->blob_scanned_prayer = -1;
    if (npc->type == INF_NPC_JAD)
        inf_npc_jad(npc)->attack_style = ATTACK_STYLE_NONE;
    npc->attack_timer = stats->attack_speed;
    if (npc->type == INF_NPC_JAD)
        npc->attack_timer = inf_jad_attack_timer(s->wave);
}

static int inf_collect_step_out_forecast_slots(
    const InfernoState* s,
    int slots[INF_MAX_NPCS]
) {
    int count = 0;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        const InfNPC* npc = &s->npcs[i];
        if (!npc->active || npc->death_ticks > 0) continue;
        slots[count++] = i;
    }
    return count;
}

static void inf_step_out_forecast_tick(
    InfernoState* sim,
    const InfernoContext* ctx,
    InfStepOutForecastAction* action,
    int tick_idx,
    const int slots[INF_MAX_NPCS],
    int slot_count
) {
    for (int slot_idx = 0; slot_idx < slot_count; slot_idx++) {
        int i = slots[slot_idx];
        InfNPC* npc = &sim->npcs[i];
        if (npc->frozen_ticks > 0) npc->frozen_ticks--;
        if (npc->type == INF_NPC_MAGER && npc->resurrect_cooldown > 0)
            npc->resurrect_cooldown--;
        inf_npc_move_ctx(sim, ctx, i);
        inf_step_out_forecast_npc_attack(sim, i, action, tick_idx);
    }
}

#ifdef INF_PROFILE_ENABLED
static void inf_step_out_forecast_tick_profiled(
    InfernoState* sim,
    const InfernoContext* ctx,
    InfStepOutForecastAction* action,
    int tick_idx,
    const int slots[INF_MAX_NPCS],
    int slot_count
) {
    for (int slot_idx = 0; slot_idx < slot_count; slot_idx++) {
        int i = slots[slot_idx];
        InfNPC* npc = &sim->npcs[i];
        if (npc->frozen_ticks > 0) npc->frozen_ticks--;
        if (npc->type == INF_NPC_MAGER && npc->resurrect_cooldown > 0)
            npc->resurrect_cooldown--;
        double t0 = INF_PROFILE_NOW_MS();
        inf_npc_move_ctx(sim, ctx, i);
        double t1 = INF_PROFILE_NOW_MS();
        INF_PROFILE_ADD(INF_PROF_FORECAST_NPC_MOVE, t1 - t0);
        inf_step_out_forecast_npc_attack(sim, i, action, tick_idx);
        INF_PROFILE_ADD(
            INF_PROF_FORECAST_NPC_ATTACK,
            INF_PROFILE_NOW_MS() - t1);
    }
}
#endif

static void inf_restore_step_out_forecast_player_collision_flags(
    InfernoState* sim,
    int prev_player_x,
    int prev_player_y
) {
    int gx, gy;
    if (inf_grid_index(prev_player_x, prev_player_y, &gx, &gy))
        sim->player_collision_flags[gx][gy] = 0;
    if (inf_grid_index(sim->player.x, sim->player.y, &gx, &gy))
        sim->player_collision_flags[gx][gy] = 1;
}

static void inf_restore_step_out_forecast_action_state(
    InfernoState* sim,
    const InfernoState* s,
    const int slots[INF_MAX_NPCS],
    int slot_count,
    const Player* moved
) {
    int prev_player_x = sim->player.x;
    int prev_player_y = sim->player.y;
    sim->player.x = moved->x;
    sim->player.y = moved->y;
    sim->player.is_running = moved->is_running;
    for (int slot_idx = 0; slot_idx < slot_count; slot_idx++) {
        int i = slots[slot_idx];
        sim->npcs[i] = s->npcs[i];
    }
    sim->rng_state = s->rng_state;
    inf_invalidate_los_cache(sim);
    memcpy(
        sim->npc_collision_flags,
        s->npc_collision_flags,
        sizeof(sim->npc_collision_flags));
    inf_restore_step_out_forecast_player_collision_flags(
        sim, prev_player_x, prev_player_y);
}

#ifdef INF_POISON_FORECAST
static void inf_poison_step_out_forecast_restore_source(
    InfernoState* restore_source,
    const InfernoState* s,
    const int slots[INF_MAX_NPCS],
    int slot_count
) {
    memset(restore_source, 0xAA, sizeof(*restore_source));
    for (int slot_idx = 0; slot_idx < slot_count; slot_idx++) {
        int i = slots[slot_idx];
        restore_source->npcs[i] = s->npcs[i];
    }
    restore_source->rng_state = s->rng_state;
    memcpy(
        restore_source->npc_collision_flags,
        s->npc_collision_flags,
        sizeof(restore_source->npc_collision_flags));
}

static void inf_poison_step_out_forecast_check_player_flags(
    const InfernoState* sim,
    const InfernoState* s,
    const Player* moved
) {
    InfernoState oracle = *s;
    memset(
        oracle.player_collision_flags,
        0xAA,
        sizeof(oracle.player_collision_flags));
    oracle.player.x = moved->x;
    oracle.player.y = moved->y;
    oracle.player.is_running = moved->is_running;
    inf_rebuild_player_collision_flags(&oracle);
    if (memcmp(
            sim->player_collision_flags,
            oracle.player_collision_flags,
            sizeof(sim->player_collision_flags)) != 0) {
        fprintf(stderr, "inferno forecast player collision restore mismatch\n");
        abort();
    }
}
#endif

static void inf_step_out_forecast_finalize_action(
    InfStepOutForecastAction* action
) {
    for (int t = 0; t < INF_STEP_OUT_FORECAST_HORIZON - 1; t++) {
        int ranger_then_mager = action->ticks[t].ranger_count > 0 &&
            action->ticks[t + 1].mager_count > 0;
        int mager_then_ranger = action->ticks[t].mager_count > 0 &&
            action->ticks[t + 1].ranger_count > 0;
        if (ranger_then_mager || mager_then_ranger)
            action->ranger_mager_offtick_opportunity = 1;
    }
}

static int inf_step_out_forecast_tick_style_mask(
    const InfStepOutForecastTick* tick
) {
    int mask = 0;
    if (tick->melee_count > 0) mask |= INF_STYLE_MASK_MELEE;
    if (tick->ranged_count > 0) mask |= INF_STYLE_MASK_RANGED;
    if (tick->magic_count > 0) mask |= INF_STYLE_MASK_MAGIC;
    return mask;
}

static int inf_step_out_forecast_tick_has_event(
    const InfStepOutForecastTick* tick
) {
    return tick->melee_count > 0 ||
        tick->ranged_count > 0 ||
        tick->magic_count > 0 ||
        tick->blob_scan_count > 0;
}

static int inf_step_out_forecast_npc_fire_tick_idx(const InfNPC* npc) {
    int tick_idx = npc->attack_timer - 1;
    return tick_idx > 0 ? tick_idx : 0;
}

static void inf_step_out_forecast_fast_static_npc(
    const InfernoState* s,
    const InfNPC* npc,
    InfStepOutForecastAction* action
) {
    if (!npc->active || npc->death_ticks > 0 || npc->hp <= 0) return;
    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    if (npc->stun_timer > 0) return;
    if (npc->dig_freeze_timer > 0 || npc->dig_attack_delay > 0) return;
    if (npc->type == INF_NPC_ZUK_SHIELD || npc->type == INF_NPC_NIBBLER)
        return;
    if (npc->aggro_target >= 0) return;

    int dist = encounter_dist_to_npc(
        action->land_x, action->land_y, npc->x, npc->y, npc->size);
    if (dist == 0 || dist > stats->attack_range) return;
    if (stats->attack_range == 1 &&
            !inf_npc_melee_contact_allows_attack_at_tile(
                action->land_x, action->land_y, npc->type,
                npc->x, npc->y, npc->size, dist))
        return;

    int has_los_now = 1;
    if (stats->attack_range > 1)
        has_los_now = inf_npc_has_los_to_tile(s, npc, action->land_x, action->land_y);

    if (npc->type == INF_NPC_BLOB &&
            npc->blob_scanned_prayer < 0 &&
            has_los_now &&
            !npc->had_los_last_tick) {
        inf_step_out_forecast_record_blob_scan(action, 0);
        return;
    }

    int tick_idx = inf_step_out_forecast_npc_fire_tick_idx(npc);
    if (tick_idx >= INF_STEP_OUT_FORECAST_HORIZON) return;

    if (stats->attack_range > 1 &&
        !(npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer >= 0) &&
        !has_los_now) return;

    if (npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer < 0) {
        inf_step_out_forecast_record_blob_scan(action, tick_idx);
        return;
    }

    int style_mask = inf_forecast_jad_unknown_style_mask(npc);
    if (style_mask == 0) {
        int planned_style = inf_npc_planned_style_for_current_state(npc);
        if (planned_style == ATTACK_STYLE_NONE) return;
        style_mask = inf_attack_style_options_mask_at_tile(
            action->land_x, action->land_y, npc, stats, planned_style, dist);
    }
    if (style_mask == 0) return;

    inf_step_out_forecast_record_style_mask(
        action, tick_idx, npc->type, stats, style_mask);
}

static void inf_forecast_local_stamp_npc(
    uint8_t npc_flags[INF_ARENA_WIDTH][INF_ARENA_HEIGHT],
    int x,
    int y,
    int size
) {
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int gx, gy;
            if (inf_grid_index(x + dx, y + dy, &gx, &gy)) {
                assert(npc_flags[gx][gy] < UINT8_MAX);
                npc_flags[gx][gy]++;
            }
        }
    }
}

static void inf_forecast_local_unstamp_npc(
    uint8_t npc_flags[INF_ARENA_WIDTH][INF_ARENA_HEIGHT],
    int x,
    int y,
    int size
) {
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int gx, gy;
            if (inf_grid_index(x + dx, y + dy, &gx, &gy) &&
                    npc_flags[gx][gy] > 0) {
                npc_flags[gx][gy]--;
            }
        }
    }
}

static int inf_forecast_local_npc_flags_blocked(
    uint8_t npc_flags[INF_ARENA_WIDTH][INF_ARENA_HEIGHT],
    int x,
    int y,
    int size
) {
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int gx, gy;
            if (inf_grid_index(x + dx, y + dy, &gx, &gy) &&
                    npc_flags[gx][gy] > 0) {
                return 1;
            }
        }
    }
    return 0;
}

static int inf_forecast_local_overlaps_player(
    int x, int y, int size, int player_x, int player_y
) {
    return !(x >= player_x + 1 || x + size <= player_x ||
             y >= player_y + 1 || y + size <= player_y);
}

static int inf_forecast_local_npc_blocked(void* ctx, int x, int y, int size) {
    InfForecastMoveCtx* mc = (InfForecastMoveCtx*)ctx;
    if (inf_npc_environment_blocked_ctx(
            (InfernoState*)mc->state, mc->ctx, x, y, size))
        return 1;
    if (inf_forecast_local_overlaps_player(
            x, y, size, mc->player_x, mc->player_y))
        return 1;
    return inf_forecast_local_npc_flags_blocked(mc->npc_flags, x, y, size);
}

static int inf_forecast_local_npc_blocked_ignore_player(
    void* ctx,
    int x,
    int y,
    int size
) {
    InfForecastMoveCtx* mc = (InfForecastMoveCtx*)ctx;
    if (inf_npc_environment_blocked_ctx(
            (InfernoState*)mc->state, mc->ctx, x, y, size))
        return 1;
    return inf_forecast_local_npc_flags_blocked(mc->npc_flags, x, y, size);
}

static void inf_forecast_local_copy_npcs(
    const InfernoState* s,
    InfForecastNpcLocal npcs[INF_MAX_NPCS]
) {
    memset(npcs, 0, sizeof(InfForecastNpcLocal) * INF_MAX_NPCS);
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        const InfNPC* npc = &s->npcs[i];
        if (!npc->active || npc->death_ticks > 0) continue;
        npcs[i] = (InfForecastNpcLocal){
            .active = 1,
            .type = npc->type,
            .x = npc->x,
            .y = npc->y,
            .size = inf_npc_effective_size(npc),
            .hp = npc->hp,
            .attack_timer = npc->attack_timer,
            .stun_timer = npc->stun_timer,
            .frozen_ticks = npc->frozen_ticks,
            .dig_freeze_timer = npc->dig_freeze_timer,
            .dig_attack_delay = npc->dig_attack_delay,
            .blob_scanned_prayer = npc->blob_scanned_prayer,
            .had_los_last_tick = npc->had_los_last_tick,
            .jad_attack_style = (npc->type == INF_NPC_JAD)
                ? inf_npc_jad_const(npc)->attack_style
                : ATTACK_STYLE_NONE,
            .attack_style = npc->attack_style,
            .aggro_target = npc->aggro_target,
            .los_to_player_cache = -1,
        };
    }
}

static void inf_forecast_local_rebuild_npc_flags(
    const InfForecastNpcLocal npcs[INF_MAX_NPCS],
    uint8_t npc_flags[INF_ARENA_WIDTH][INF_ARENA_HEIGHT]
) {
    memset(npc_flags, 0, INF_ARENA_WIDTH * INF_ARENA_HEIGHT * sizeof(uint8_t));
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        const InfForecastNpcLocal* npc = &npcs[i];
        if (!npc->active || npc->hp <= 0) continue;
        if (!inf_npc_sets_collision_flag(npc->type)) continue;
        inf_forecast_local_stamp_npc(npc_flags, npc->x, npc->y, npc->size);
    }
}

static int inf_forecast_local_has_los_to_area(
    const InfernoState* s,
    const InfForecastNpcLocal* npc,
    int target_x,
    int target_y,
    int target_size
) {
    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    return entity_has_line_of_sight(s->los_blockers, s->los_blocker_count,
        npc->x, npc->y, npc->size,
        target_x, target_y, target_size,
        stats->attack_range);
}

static int inf_forecast_local_has_los_to_tile(
    const InfernoState* s,
    const InfForecastNpcLocal* npc,
    int target_x,
    int target_y
) {
    return inf_forecast_local_has_los_to_area(s, npc, target_x, target_y, 1);
}

static int inf_forecast_local_has_los_to_player(
    const InfernoState* s,
    InfForecastNpcLocal* npc,
    int target_x,
    int target_y
) {
    if (npc->los_to_player_cache >= 0)
        return npc->los_to_player_cache;
    npc->los_to_player_cache = inf_forecast_local_has_los_to_tile(
        s, npc, target_x, target_y);
    return npc->los_to_player_cache;
}

static InfTargetArea inf_forecast_local_target_area(
    const InfernoState* s,
    const InfForecastNpcLocal npcs[INF_MAX_NPCS],
    const InfForecastNpcLocal* npc,
    int player_x,
    int player_y
) {
    if (npc->aggro_target >= 0 && npc->aggro_target < INF_MAX_NPCS &&
            npcs[npc->aggro_target].active) {
        const InfForecastNpcLocal* target = &npcs[npc->aggro_target];
        return (InfTargetArea){
            .x = target->x,
            .y = target->y,
            .size = target->size,
            .is_player = 0,
        };
    }

    if (npc->type == INF_NPC_NIBBLER) {
        int pillar_idx = s->nibbler_target_pillar;
        if (pillar_idx >= 0 && pillar_idx < INF_NUM_PILLARS &&
                s->pillars[pillar_idx].active) {
            return (InfTargetArea){
                .x = s->pillars[pillar_idx].x,
                .y = s->pillars[pillar_idx].y,
                .size = INF_PILLAR_SIZE,
                .is_player = 0,
            };
        }
    }

    return (InfTargetArea){
        .x = player_x,
        .y = player_y,
        .size = 1,
        .is_player = 1,
    };
}

static int inf_forecast_local_planned_style(
    const InfForecastNpcLocal* npc
) {
    if (npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer >= 0) {
        OverheadPrayer scanned = (OverheadPrayer)npc->blob_scanned_prayer;
        if (scanned == PRAYER_PROTECT_MAGIC) return ATTACK_STYLE_RANGED;
        if (scanned == PRAYER_PROTECT_RANGED) return ATTACK_STYLE_MAGIC;
    }
    if (npc->type == INF_NPC_JAD)
        return npc->jad_attack_style;
    return npc->attack_style;
}

static int inf_forecast_local_melee_fallback_possible_at_tile(
    int player_x,
    int player_y,
    const InfForecastNpcLocal* npc,
    const InfNPCStats* stats,
    int planned_style,
    int dist
) {
    if (!stats->can_melee || planned_style == ATTACK_STYLE_MELEE || dist != 1)
        return 0;

    return inf_npc_melee_contact_allows_attack_at_tile(
        player_x, player_y, npc->type, npc->x, npc->y, npc->size, dist);
}

static int inf_forecast_local_attack_style_options_mask_at_tile(
    int player_x,
    int player_y,
    const InfForecastNpcLocal* npc,
    const InfNPCStats* stats,
    int planned_style,
    int dist
) {
    int mask = inf_attack_style_mask_bit(planned_style);
    if (inf_forecast_local_melee_fallback_possible_at_tile(
            player_x, player_y, npc, stats, planned_style, dist))
        mask |= INF_STYLE_MASK_MELEE;
    return mask;
}

static int inf_forecast_local_jad_unknown_style_mask(
    const InfForecastNpcLocal* npc
) {
    if (npc->type != INF_NPC_JAD) return 0;
    if (npc->jad_attack_style != ATTACK_STYLE_NONE) return 0;
    return INF_STYLE_MASK_RANGED | INF_STYLE_MASK_MAGIC;
}

static void inf_forecast_local_record_overlap_danger(
    const InfernoState* s,
    InfForecastNpcLocal* npc,
    InfStepOutForecastAction* action,
    int tick_idx
) {
    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    if (npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer < 0) {
        npc->blob_scanned_prayer = (int)s->player.prayer;
        inf_step_out_forecast_record_blob_scan(action, tick_idx);
    }

    int style_mask = inf_forecast_local_jad_unknown_style_mask(npc);
    if (style_mask == 0) {
        int planned_style = inf_forecast_local_planned_style(npc);
        style_mask = inf_attack_style_mask_bit(planned_style);
    }
    if (stats->can_melee)
        style_mask |= INF_STYLE_MASK_MELEE;
    if (style_mask == 0) return;
    inf_step_out_forecast_record_style_mask(
        action, tick_idx, npc->type, stats, style_mask);
}

static void inf_forecast_local_move_npc(
    const InfernoState* s,
    const InfernoContext* ctx,
    InfForecastNpcLocal npcs[INF_MAX_NPCS],
    uint8_t npc_flags[INF_ARENA_WIDTH][INF_ARENA_HEIGHT],
    int idx,
    InfStepOutForecastAction* action,
    int tick_idx
) {
    InfForecastNpcLocal* npc = &npcs[idx];
    if (!npc->active || npc->hp <= 0) return;
    if (npc->stun_timer > 0) return;
    if (npc->dig_freeze_timer > 0) return;
    if (npc->frozen_ticks > 0) return;
    if (npc->type == INF_NPC_NIBBLER || npc->type == INF_NPC_ZUK_SHIELD)
        return;

    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    if (!stats->can_move) return;
    int uses_collision_flag = inf_npc_sets_collision_flag(npc->type);
    if (uses_collision_flag)
        inf_forecast_local_unstamp_npc(npc_flags, npc->x, npc->y, npc->size);

    if (inf_forecast_local_overlaps_player(
            npc->x, npc->y, npc->size, action->land_x, action->land_y)) {
        inf_forecast_local_record_overlap_danger(s, npc, action, tick_idx);
        if (uses_collision_flag)
            inf_forecast_local_stamp_npc(npc_flags, npc->x, npc->y, npc->size);
        return;
    }

    InfTargetArea target = inf_forecast_local_target_area(
        s, npcs, npc, action->land_x, action->land_y);
    if (target.is_player && npc->aggro_target >= 0)
        npc->aggro_target = -1;

    int has_los = inf_forecast_local_has_los_to_area(
        s, npc, target.x, target.y, target.size);
    if (target.is_player)
        npc->los_to_player_cache = has_los;
    if (has_los) {
        if (uses_collision_flag)
            inf_forecast_local_stamp_npc(npc_flags, npc->x, npc->y, npc->size);
        return;
    }
    if (npc->attack_timer > stats->attack_speed) {
        if (uses_collision_flag)
            inf_forecast_local_stamp_npc(npc_flags, npc->x, npc->y, npc->size);
        return;
    }

    InfForecastMoveCtx move_ctx = {
        .state = s,
        .ctx = ctx,
        .npc_flags = npc_flags,
        .player_x = action->land_x,
        .player_y = action->land_y,
    };
    EncounterNpcStepPolicy step_policy;
    encounter_npc_blocked_fn blocked;
    if (target.is_player) {
        step_policy = stats->attack_range == 1
            ? ENCOUNTER_NPC_STEP_OSRS_AGGRO_STOP_AT_MELEE
            : ENCOUNTER_NPC_STEP_OSRS_AGGRO_TARGET;
        blocked = inf_forecast_local_npc_blocked_ignore_player;
    } else {
        step_policy = stats->attack_range == 1
            ? ENCOUNTER_NPC_STEP_STOP_AT_MELEE
            : ENCOUNTER_NPC_STEP_TRAVEL_TARGET;
        blocked = inf_forecast_local_npc_blocked;
    }
    uint32_t rng = s->rng_state;
    encounter_npc_step_toward_policy(
        &npc->x, &npc->y, target.x, target.y, npc->size,
        target.size, step_policy, blocked, &move_ctx, NULL, &rng);
    npc->los_to_player_cache = -1;

    if (uses_collision_flag)
        inf_forecast_local_stamp_npc(npc_flags, npc->x, npc->y, npc->size);
}

static void inf_forecast_local_attack_npc(
    const InfernoState* s,
    InfForecastNpcLocal* npc,
    InfStepOutForecastAction* action,
    int tick_idx
) {
    if (!npc->active || npc->hp <= 0) return;
    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    if (npc->attack_timer > 0) npc->attack_timer--;
    if (npc->stun_timer > 0) { npc->stun_timer--; return; }
    if (npc->dig_freeze_timer > 0 || npc->dig_attack_delay > 0) return;
    if (npc->type == INF_NPC_ZUK_SHIELD || npc->type == INF_NPC_NIBBLER)
        return;
    if (npc->aggro_target >= 0) return;

    if (inf_forecast_local_overlaps_player(
            npc->x, npc->y, npc->size, action->land_x, action->land_y)) {
        inf_forecast_local_record_overlap_danger(s, npc, action, tick_idx);
        return;
    }

    int has_los_now = 0;
    if (npc->type == INF_NPC_BLOB && stats->attack_range > 1) {
        has_los_now = inf_forecast_local_has_los_to_player(
            s, npc, action->land_x, action->land_y);
        if (npc->blob_scanned_prayer < 0 &&
            has_los_now &&
            !npc->had_los_last_tick) {
            npc->blob_scanned_prayer = (int)s->player.prayer;
            npc->attack_style = inf_forecast_blob_scan_style(s->player.prayer);
            npc->had_los_last_tick = has_los_now;
            npc->attack_timer = stats->attack_speed;
            inf_step_out_forecast_record_blob_scan(action, tick_idx);
            return;
        }
        npc->had_los_last_tick = has_los_now;
    }
    if (npc->attack_timer > 0) return;

    if (npc->type != INF_NPC_BLOB && stats->attack_range > 1) {
        has_los_now = inf_forecast_local_has_los_to_player(
            s, npc, action->land_x, action->land_y);
    }

    if (stats->attack_range > 1 &&
        !(npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer >= 0) &&
        !has_los_now) return;

    int dist = encounter_dist_to_npc(
        action->land_x, action->land_y, npc->x, npc->y, npc->size);
    if (dist == 0 || dist > stats->attack_range) return;
    if (stats->attack_range == 1 &&
            !inf_npc_melee_contact_allows_attack_at_tile(
                action->land_x, action->land_y, (InfNPCType)npc->type,
                npc->x, npc->y, npc->size, dist))
        return;

    if (npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer < 0) {
        npc->blob_scanned_prayer = (int)s->player.prayer;
        npc->attack_style = inf_forecast_blob_scan_style(s->player.prayer);
        npc->attack_timer = stats->attack_speed;
        inf_step_out_forecast_record_blob_scan(action, tick_idx);
        return;
    }

    int style_mask = inf_forecast_local_jad_unknown_style_mask(npc);
    if (style_mask == 0) {
        int planned_style = inf_forecast_local_planned_style(npc);
        if (planned_style == ATTACK_STYLE_NONE) return;
        style_mask = inf_forecast_local_attack_style_options_mask_at_tile(
            action->land_x, action->land_y, npc,
            stats, planned_style, dist);
    }
    if (style_mask == 0) return;

    inf_step_out_forecast_record_style_mask(
        action, tick_idx, npc->type, stats, style_mask);

    if (npc->type == INF_NPC_BLOB)
        npc->blob_scanned_prayer = -1;
    if (npc->type == INF_NPC_JAD)
        npc->jad_attack_style = ATTACK_STYLE_NONE;
    npc->attack_timer = stats->attack_speed;
    if (npc->type == INF_NPC_JAD)
        npc->attack_timer = inf_jad_attack_timer(s->wave);
}

static int inf_forecast_npc_uses_readonly_sim(const InfNPC* npc) {
    switch (npc->type) {
        case INF_NPC_RANGER:
        case INF_NPC_MAGER:
        case INF_NPC_JAD:
            return 1;
        default:
            return 0;
    }
}

static void inf_step_out_forecast_conservative_meleer_npc(
    const InfNPC* npc,
    InfStepOutForecastAction* action
) {
    if (npc->type != INF_NPC_MELEER) return;
    if (!npc->active || npc->death_ticks > 0 || npc->hp <= 0) return;
    if (npc->dig_freeze_timer > 0 || npc->dig_attack_delay > 0) return;

    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    int dist = encounter_dist_to_npc(
        action->land_x, action->land_y, npc->x, npc->y, npc->size);
    if (dist > INF_STEP_OUT_FORECAST_HORIZON + stats->attack_range) return;

    int attack_ready_tick = npc->attack_timer > 0 ? npc->attack_timer - 1 : 0;
    int frozen_move_delay = npc->frozen_ticks > 0 ? npc->frozen_ticks - 1 : 0;
    int move_delay = npc->stun_timer > frozen_move_delay ?
        npc->stun_timer : frozen_move_delay;
    int contact_tick = dist <= stats->attack_range ?
        0 : move_delay + dist - stats->attack_range - 1;
    int tick_idx = contact_tick > attack_ready_tick ?
        contact_tick : attack_ready_tick;
    if (tick_idx < 0 || tick_idx >= INF_STEP_OUT_FORECAST_HORIZON) return;

    inf_step_out_forecast_record_style_mask(
        action, tick_idx, npc->type, stats, INF_STYLE_MASK_MELEE);
}

static void inf_step_out_forecast_readonly_tick(
    const InfernoState* s,
    const InfernoContext* ctx,
    InfForecastNpcLocal npcs[INF_MAX_NPCS],
    uint8_t npc_flags[INF_ARENA_WIDTH][INF_ARENA_HEIGHT],
    InfStepOutForecastAction* action,
    int tick_idx,
    const int slots[INF_MAX_NPCS],
    int slot_count
) {
    for (int slot_idx = 0; slot_idx < slot_count; slot_idx++) {
        int i = slots[slot_idx];
        InfForecastNpcLocal* npc = &npcs[i];
        if (npc->frozen_ticks > 0) npc->frozen_ticks--;
        inf_forecast_local_move_npc(s, ctx, npcs, npc_flags, i, action, tick_idx);
        inf_forecast_local_attack_npc(s, npc, action, tick_idx);
    }
}

static void inf_build_step_out_forecast_fast_static_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    InfStepOutForecast* out
) {
    memset(out, 0, sizeof(*out));
    int forecast_slots[INF_MAX_NPCS];
    int forecast_slot_count = inf_collect_step_out_forecast_slots(
        s, forecast_slots);
    for (int action_idx = 0; action_idx < ENCOUNTER_MOVE_ACTIONS; action_idx++) {
        if (!inf_step_out_forecast_prepare_unique_action_ctx(
                s, ctx, out, action_idx, forecast_slot_count))
            continue;
        InfStepOutForecastAction* action = &out->actions[action_idx];

        for (int slot_idx = 0; slot_idx < forecast_slot_count; slot_idx++) {
            int i = forecast_slots[slot_idx];
            inf_step_out_forecast_fast_static_npc(s, &s->npcs[i], action);
        }
        inf_step_out_forecast_finalize_action(action);
    }
}

static void inf_build_step_out_forecast_fast_readonly_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    InfStepOutForecast* out
) {
    memset(out, 0, sizeof(*out));
    int forecast_slots[INF_MAX_NPCS];
    int forecast_slot_count = inf_collect_step_out_forecast_slots(
        s, forecast_slots);
    int readonly_slots[INF_MAX_NPCS];
    int readonly_slot_count = 0;
    for (int slot_idx = 0; slot_idx < forecast_slot_count; slot_idx++) {
        int i = forecast_slots[slot_idx];
        if (inf_forecast_npc_uses_readonly_sim(&s->npcs[i]))
            readonly_slots[readonly_slot_count++] = i;
    }
    InfForecastNpcLocal base_npcs[INF_MAX_NPCS];
    uint8_t base_npc_flags[INF_ARENA_WIDTH][INF_ARENA_HEIGHT];
    if (readonly_slot_count > 0) {
        inf_forecast_local_copy_npcs(s, base_npcs);
        inf_forecast_local_rebuild_npc_flags(base_npcs, base_npc_flags);
    }
    for (int action_idx = 0; action_idx < ENCOUNTER_MOVE_ACTIONS; action_idx++) {
        if (!inf_step_out_forecast_prepare_unique_action_ctx(
                s, ctx, out, action_idx, forecast_slot_count))
            continue;
        InfStepOutForecastAction* action = &out->actions[action_idx];

        for (int slot_idx = 0; slot_idx < forecast_slot_count; slot_idx++) {
            int i = forecast_slots[slot_idx];
            if (inf_forecast_npc_uses_readonly_sim(&s->npcs[i])) continue;
            inf_step_out_forecast_fast_static_npc(s, &s->npcs[i], action);
            inf_step_out_forecast_conservative_meleer_npc(&s->npcs[i], action);
        }

        if (readonly_slot_count > 0) {
            InfForecastNpcLocal npcs[INF_MAX_NPCS];
            uint8_t npc_flags[INF_ARENA_WIDTH][INF_ARENA_HEIGHT];
            memcpy(npcs, base_npcs, sizeof(base_npcs));
            memcpy(npc_flags, base_npc_flags, sizeof(base_npc_flags));

            for (int tick_idx = 0; tick_idx < INF_STEP_OUT_FORECAST_HORIZON; tick_idx++) {
                inf_step_out_forecast_readonly_tick(
                    s, ctx, npcs, npc_flags, action, tick_idx,
                    readonly_slots, readonly_slot_count);
            }
        }
        inf_step_out_forecast_finalize_action(action);
    }
}

static void inf_build_step_out_forecast_exact_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    InfStepOutForecast* out
) {
#ifdef INF_PROFILE_ENABLED
    int inf_prof_enabled = INF_PROFILE_ENABLED();
    int prof_valid_actions = 0;
    int prof_distinct_landings = 0;
    if (inf_prof_enabled)
        INF_PROFILE_ADD(INF_PROF_FORECAST_CALLS, 1.0);
#endif
    memset(out, 0, sizeof(*out));
    int forecast_slots[INF_MAX_NPCS];
    int forecast_slot_count = inf_collect_step_out_forecast_slots(
        s, forecast_slots);

    int forecast_has_active_pillar = 0;
    for (int p = 0; p < INF_NUM_PILLARS; p++) {
        if (s->pillars[p].active) { forecast_has_active_pillar = 1; break; }
    }
#ifdef INF_PROFILE_ENABLED
    double base_copy_t0 = inf_prof_enabled ? INF_PROFILE_NOW_MS() : 0.0;
#endif
    InfernoState sim = *s;
#ifdef INF_PROFILE_ENABLED
    if (inf_prof_enabled) {
        INF_PROFILE_ADD(
            INF_PROF_FORECAST_STATE_COPY,
            INF_PROFILE_NOW_MS() - base_copy_t0);
    }
#endif
    const InfernoState* restore_source = s;
#ifdef INF_POISON_FORECAST
    InfernoState poison_restore_source;
    inf_poison_step_out_forecast_restore_source(
        &poison_restore_source, s, forecast_slots, forecast_slot_count);
    restore_source = &poison_restore_source;
#endif
    for (int action_idx = 0; action_idx < ENCOUNTER_MOVE_ACTIONS; action_idx++) {
        InfStepOutForecastAction* action = &out->actions[action_idx];
#ifdef INF_PROFILE_ENABLED
        double landing_t0 = inf_prof_enabled ? INF_PROFILE_NOW_MS() : 0.0;
#endif
        Player moved = inf_step_out_forecast_set_action_landing_ctx(
            s, ctx, action_idx, action);
#ifdef INF_PROFILE_ENABLED
        if (inf_prof_enabled) {
            INF_PROFILE_ADD(
                INF_PROF_FORECAST_LANDING,
                INF_PROFILE_NOW_MS() - landing_t0);
            if (action->valid) {
                prof_valid_actions++;
                int duplicate_landing = 0;
                for (int prior_idx = 0; prior_idx < action_idx; prior_idx++) {
                    const InfStepOutForecastAction* prior =
                        &out->actions[prior_idx];
                    if (prior->valid &&
                            prior->land_x == action->land_x &&
                            prior->land_y == action->land_y) {
                        duplicate_landing = 1;
                        break;
                    }
                }
                if (!duplicate_landing)
                    prof_distinct_landings++;
            }
        }
#endif
        if (!action->valid || forecast_slot_count == 0) continue;
        if (!forecast_has_active_pillar) continue;

#ifdef INF_PROFILE_ENABLED
        double state_copy_t0 = inf_prof_enabled ? INF_PROFILE_NOW_MS() : 0.0;
#endif
        inf_restore_step_out_forecast_action_state(
            &sim, restore_source, forecast_slots, forecast_slot_count, &moved);
#ifdef INF_POISON_FORECAST
        inf_poison_step_out_forecast_check_player_flags(&sim, s, &moved);
#endif
#ifdef INF_PROFILE_ENABLED
        if (inf_prof_enabled) {
            INF_PROFILE_ADD(
                INF_PROF_FORECAST_STATE_COPY,
                INF_PROFILE_NOW_MS() - state_copy_t0);
        }
#endif

        for (int tick_idx = 0; tick_idx < INF_STEP_OUT_FORECAST_HORIZON; tick_idx++) {
#ifdef INF_PROFILE_ENABLED
            if (inf_prof_enabled) {
                inf_step_out_forecast_tick_profiled(
                    &sim, ctx, action, tick_idx, forecast_slots, forecast_slot_count);
                continue;
            }
#endif
            inf_step_out_forecast_tick(
                &sim, ctx, action, tick_idx, forecast_slots, forecast_slot_count);
        }
        inf_step_out_forecast_finalize_action(action);
    }
#ifdef INF_PROFILE_ENABLED
    if (inf_prof_enabled) {
        INF_PROFILE_ADD(
            INF_PROF_FORECAST_VALID_ACTIONS,
            (double)prof_valid_actions);
        INF_PROFILE_ADD(
            INF_PROF_FORECAST_DISTINCT_LANDINGS,
            (double)prof_distinct_landings);
    }
#endif
}

static void inf_build_step_out_forecast_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    InfStepOutForecast* out
) {
    inf_build_step_out_forecast_exact_ctx(s, ctx, out);
}

static void inf_build_step_out_forecast_mode_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    InfStepOutForecast* out
) {
    switch (ctx->config.step_out_forecast_obs_mode) {
        case INF_STEP_OUT_FORECAST_MODE_EXACT_ROLLOUT:
            inf_build_step_out_forecast_exact_ctx(s, ctx, out);
            return;
        case INF_STEP_OUT_FORECAST_MODE_FAST_STATIC_TILE:
            inf_build_step_out_forecast_fast_static_ctx(s, ctx, out);
            return;
        case INF_STEP_OUT_FORECAST_MODE_FAST_READONLY_MOVE:
            inf_build_step_out_forecast_fast_readonly_ctx(s, ctx, out);
            return;
        default:
            fprintf(stderr,
                "inferno: step_out_forecast_obs_mode must be 1, 2, or 3 for forecast build, got %d\n",
                ctx->config.step_out_forecast_obs_mode);
            abort();
    }
}

static int inf_step_out_forecast_action_dangerous(
    const InfStepOutForecastAction* action
) {
    if (action->same_tick_mixed_style_conflict) return 1;
    for (int tick_idx = 0; tick_idx < INF_STEP_OUT_FORECAST_HORIZON; tick_idx++) {
        const InfStepOutForecastTick* tick = &action->ticks[tick_idx];
        if (tick->max_hit >= 46) return 1;
    }
    return 0;
}

static void inf_compare_step_out_forecasts(
    const InfStepOutForecast* exact,
    const InfStepOutForecast* fast,
    InfStepOutForecastOracleDiff* out
) {
    memset(out, 0, sizeof(*out));
    for (int action_idx = 0; action_idx < ENCOUNTER_MOVE_ACTIONS; action_idx++) {
        const InfStepOutForecastAction* a = &exact->actions[action_idx];
        const InfStepOutForecastAction* b = &fast->actions[action_idx];
        out->sampled_actions++;
        if (a->valid != b->valid ||
                a->land_x != b->land_x ||
                a->land_y != b->land_y ||
                a->same_tick_mixed_style_conflict != b->same_tick_mixed_style_conflict ||
                a->ranger_mager_offtick_opportunity != b->ranger_mager_offtick_opportunity ||
                a->melee_fallback_exposure != b->melee_fallback_exposure) {
            out->action_feature_mismatches++;
        }
        int exact_danger = inf_step_out_forecast_action_dangerous(a);
        int fast_danger = inf_step_out_forecast_action_dangerous(b);
        out->exact_dangerous_actions += exact_danger;
        out->fast_dangerous_actions += fast_danger;
        if (exact_danger && !fast_danger)
            out->dangerous_false_negatives++;
        if (!exact_danger && fast_danger) {
            out->dangerous_false_positives++;
            out->exact_safe_fast_dangerous++;
        }
        for (int tick_idx = 0; tick_idx < INF_STEP_OUT_FORECAST_HORIZON; tick_idx++) {
            const InfStepOutForecastTick* at = &a->ticks[tick_idx];
            const InfStepOutForecastTick* bt = &b->ticks[tick_idx];
            if (at->melee_count != bt->melee_count ||
                    at->ranged_count != bt->ranged_count ||
                    at->magic_count != bt->magic_count ||
                    at->blob_scan_count != bt->blob_scan_count ||
                    at->ranger_count != bt->ranger_count ||
                    at->mager_count != bt->mager_count ||
                    at->max_hit != bt->max_hit) {
                out->tick_feature_mismatches++;
            }
            int max_hit_error = at->max_hit - bt->max_hit;
            if (max_hit_error < 0) max_hit_error = -max_hit_error;
            out->max_hit_abs_error_sum += max_hit_error;
            if (max_hit_error > out->max_hit_abs_error_max)
                out->max_hit_abs_error_max = max_hit_error;
        }
    }
}

static void inf_build_step_out_forecast(
    const InfernoState* s,
    InfStepOutForecast* out
) {
    inf_build_step_out_forecast_ctx(s, inf_legacy_context(), out);
}
