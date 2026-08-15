#line 5216 "encounter_inferno.h"
#define INF_STEP_OUT_FORECAST_HORIZON 4

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

static int inf_step_out_forecast_action_valid(
    const InfernoState* s,
    const InfernoContext* ctx,
    int action
) {
    if (action == 0) return 1;
    int nx = s->player.x + ENCOUNTER_MOVE_TARGET_DX[action];
    int ny = s->player.y + ENCOUNTER_MOVE_TARGET_DY[action];
    return !inf_footprint_blocked_ctx(s, ctx, nx, ny, 1);
}

static void inf_step_out_forecast_action_landing_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    int action_idx,
    Player* moved,
    int* valid
) {
    *moved = s->player;
    *valid = inf_step_out_forecast_action_valid(s, ctx, action_idx);
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
    const InfernoContext* ctx,
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
        has_los_now = inf_npc_has_los_ctx(s, ctx, idx);
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
        has_los_now = inf_npc_has_los_ctx(s, ctx, idx);

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
#ifdef INF_PROFILE_ENABLED
        int profile_enabled = INF_PROFILE_ENABLED();
        double move_start = profile_enabled ? INF_PROFILE_NOW_MS() : 0.0;
#endif
        inf_npc_move_ctx(sim, ctx, i);
#ifdef INF_PROFILE_ENABLED
        double attack_start = profile_enabled ? INF_PROFILE_NOW_MS() : 0.0;
        if (profile_enabled)
            INF_PROFILE_ADD(INF_PROF_FORECAST_NPC_MOVE, attack_start - move_start);
#endif
        inf_step_out_forecast_npc_attack(
            sim, ctx, i, action, tick_idx);
#ifdef INF_PROFILE_ENABLED
        if (profile_enabled)
            INF_PROFILE_ADD(
                INF_PROF_FORECAST_NPC_ATTACK,
                INF_PROFILE_NOW_MS() - attack_start);
#endif
    }
}



static void inf_restore_step_out_forecast_action_state(
    InfernoState* sim,
    const InfernoState* s,
    const int slots[INF_MAX_NPCS],
    int slot_count,
    const Player* moved
) {
    sim->player.x = moved->x;
    sim->player.y = moved->y;
    sim->player.is_running = moved->is_running;
    for (int slot_idx = 0; slot_idx < slot_count; slot_idx++) {
        int i = slots[slot_idx];
        sim->npcs[i] = s->npcs[i];
    }
    sim->rng_state = s->rng_state;
    memcpy(
        sim->npc_collision_flags,
        s->npc_collision_flags,
        sizeof(sim->npc_collision_flags));
}

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


static void inf_build_step_out_forecast_ctx(
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
    InfernoContext forecast_ctx = *ctx;
#ifdef INF_PROFILE_ENABLED
    if (inf_prof_enabled) {
        INF_PROFILE_ADD(
            INF_PROF_FORECAST_STATE_COPY,
            INF_PROFILE_NOW_MS() - base_copy_t0);
    }
#endif
    for (int action_idx = 0; action_idx < ENCOUNTER_MOVE_ACTIONS; action_idx++) {
        InfStepOutForecastAction* action = &out->actions[action_idx];
#ifdef INF_PROFILE_ENABLED
        double landing_t0 = inf_prof_enabled ? INF_PROFILE_NOW_MS() : 0.0;
#endif
        inf_reset_npc_player_los_frame(&forecast_ctx);
        Player moved = inf_step_out_forecast_set_action_landing_ctx(
            s, &forecast_ctx, action_idx, action);
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
            &sim, s, forecast_slots, forecast_slot_count, &moved);
#ifdef INF_PROFILE_ENABLED
        if (inf_prof_enabled) {
            INF_PROFILE_ADD(
                INF_PROF_FORECAST_STATE_COPY,
                INF_PROFILE_NOW_MS() - state_copy_t0);
        }
#endif

        for (int tick_idx = 0; tick_idx < INF_STEP_OUT_FORECAST_HORIZON; tick_idx++)
            inf_step_out_forecast_tick(
                &sim, &forecast_ctx, action, tick_idx,
                forecast_slots, forecast_slot_count);
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
