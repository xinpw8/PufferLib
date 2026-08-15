#line 76 "encounter_colosseum.h"

static const ColoBestGear (*col_get_best_gear_table(ColosseumState* s))[COLO_NUM_NPC_TYPES];
static float col_expected_dpt_for_equipment_vs_npc(
    ColosseumState* s, const uint8_t equipment[NUM_GEAR_SLOTS],
    const ColoNPC* npc, int use_style_matched_prayer);

static void col_record_reward_clamp_headroom(ColosseumState* s) {
    float mag = fabsf(s->reward);
    s->log.reward_steps += 1.0f;
    if (mag > s->log.reward_raw_peak) s->log.reward_raw_peak = mag;
    if (mag <= 1.0f) return;
    float over = mag - 1.0f;
    s->log.reward_clamped_steps += 1.0f;
    s->log.reward_clamp_loss += over;
    if (s->tick_scratch.wave_completed) s->log.clamp_loss_wave_clear += over;
    if (s->episode_over && s->winner == COLO_OUTCOME_PLAYER_WON)
        s->log.clamp_loss_win += over;
}

static float col_terminal_death_reward(ColosseumState* s, const ColosseumContext* ctx) {
    s->log.rew_death -= ctx->config.death_penalty_coeff;
    return -ctx->config.death_penalty_coeff;
}


static void col_update_min_sol_hp_seen(ColosseumState* s) {
    if (!s->sol.started) return;
    int idx = col_sol_find_idx(s);
    if (idx < 0) return;
    int hp = s->npcs[idx].hp;
    if (hp < 0) hp = 0;
    if (hp < s->min_sol_hp_seen) s->min_sol_hp_seen = hp;
}

static int col_sol_score_phase_for_min_hp(int hp) {
    if (hp > 1350) return 0;
    if (hp > 1125) return 1;
    if (hp > 750) return 2;
    if (hp > 375) return 3;
    if (hp > 150) return 4;
    return 5;
}

static float col_sol_score_progress(const ColosseumState* s) {
    if (!s->sol.started) return 0.0f;
    int hp = s->min_sol_hp_seen;
    if (hp < 0) hp = 0;
    if (hp > COLO_SOL_HP_MAX) hp = COLO_SOL_HP_MAX;

    static const int phase_hi_hp[COLO_SOL_NUM_PHASES] = {
        1500, 1350, 1125, 750, 375, 150,
    };
    static const int phase_lo_hp[COLO_SOL_NUM_PHASES] = {
        1350, 1125, 750, 375, 150, 0,
    };
    int phase = col_sol_score_phase_for_min_hp(hp);
    float denom = (float)(phase_hi_hp[phase] - phase_lo_hp[phase]);
    float within_phase = ((float)phase_hi_hp[phase] - (float)hp) / denom;
    if (within_phase < 0.0f) within_phase = 0.0f;
    if (within_phase > 1.0f) within_phase = 1.0f;
    return ((float)phase + within_phase) / (float)COLO_SOL_NUM_PHASES;
}

static float col_current_wave_score_progress(const ColosseumState* s) {
    if (s->wave == COLO_WAVE_BOSS || s->sol.started) return 0.0f;
    if (s->current_wave_hp_pool <= 0) return 0.0f;
    float progress = s->current_wave_fresh_damage / (float)s->current_wave_hp_pool;
    if (progress < 0.0f) return 0.0f;

    if (progress >= 1.0f) return 0.0f;
    return progress;
}

#define COLO_SCORE_LOSS_CEILING 0.5f

static float col_episode_outcome_score(const ColosseumState* s) {
    if (s->episode_over && s->winner == COLO_OUTCOME_PLAYER_WON) return 1.0f;
    int waves_cleared = s->log.waves_cleared;
    if (waves_cleared > COLO_WAVE_BOSS) waves_cleared = COLO_WAVE_BOSS;
    float depth = (float)waves_cleared +
        col_current_wave_score_progress(s) +
        col_sol_score_progress(s);
    float ratio = depth / (float)COLO_NUM_WAVES;
    return COLO_SCORE_LOSS_CEILING * ratio * ratio;
}

static void col_record_terminal_outcome_score(ColosseumState* s) {
    s->log.outcome_score = col_episode_outcome_score(s);
}

static float col_episode_max_wave_depth(const ColosseumState* s) {
    if (s->episode_over && s->winner == COLO_OUTCOME_PLAYER_WON)
        return (float)COLO_NUM_WAVES;
    int waves_cleared = s->log.waves_cleared;
    if (waves_cleared > COLO_WAVE_BOSS) waves_cleared = COLO_WAVE_BOSS;
    return (float)waves_cleared +
        col_current_wave_score_progress(s) +
        col_sol_score_progress(s);
}

typedef struct {
    float total;
    float best;
    float achieved;
} ColoTickAvoidance;

static ColoTickAvoidance col_tick_avoidance(const ColoTickScratch* t) {
    ColoTickAvoidance a = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 4; i++) {
        float v = t->avoidable_threat_by_style[i];
        a.total += v;
        if (v > a.best) a.best = v;
    }
    a.achieved = a.total - t->offpray_damage_this_tick;
    if (a.achieved < 0.0f) a.achieved = 0.0f;
    if (a.achieved > a.best) a.achieved = a.best;
    return a;
}

static void col_accumulate_tick_stats(ColosseumState* s, const ColosseumContext* ctx) {
    const ColConfig* cfg = &ctx->config;
    ColoTickScratch* t = &s->tick_scratch;

    assert(cfg->player_damage_received_scale != 1.0f ||
        fabsf(t->landed_offpray_damage + t->landed_prayed_damage +
              t->landed_unprayable_damage + t->landed_self_damage
              - t->damage_received) < 0.5f);

    s->log.total_damage_dealt += t->damage_dealt;
    s->log.total_boss_damage += t->damage_boss;
    s->log.total_damage_received += t->damage_received;
    s->log.total_prayer_correct += t->prayer_correct;
    s->log.total_npc_attacks += t->attacks_fired;
    col_update_min_sol_hp_seen(s);
    if (s->sol.phase > s->log.boss_phase_reached) s->log.boss_phase_reached = s->sol.phase;
    if (s->wave < cfg->farm_cap_waves)
        s->log.farm_damage += t->fresh_damage_reinforcement;

    ColoTickAvoidance av = col_tick_avoidance(t);
    s->log.avoid_total += av.total;
    s->log.avoid_achieved += av.achieved;
    s->log.avoid_missed += av.best - av.achieved;
    s->log.avoid_impossible += av.total - av.best;
    s->log.dmg_unprayable += t->landed_unprayable_damage;
}

typedef struct {
    float damage;
    float boss_phase;
} ColoShapedReward;

static ColoShapedReward col_shaped_reward(ColosseumState* s, const ColosseumContext* ctx) {
    const ColConfig* cfg = &ctx->config;
    ColoTickScratch* t = &s->tick_scratch;
    ColoShapedReward out = {0.0f, 0.0f};

    float rewarded_fresh = t->fresh_damage_dealt;
    if (s->wave < cfg->farm_cap_waves && cfg->farm_safe_damage_cap)
        rewarded_fresh -= t->fresh_damage_reinforcement;

    out.damage = cfg->damage_reward_coeff * rewarded_fresh
        + cfg->boss_damage_reward_coeff * t->fresh_damage_boss;
    if (s->sol.started && s->sol.phase > s->rewarded_boss_phase) {
        out.boss_phase =
            cfg->boss_phase_bonus * (float)(s->sol.phase - s->rewarded_boss_phase);
        s->rewarded_boss_phase = s->sol.phase;
    }
    return out;
}


static float col_inventory_heal_remaining(const ColosseumState* s) {
    float total = 0.0f;
    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++) {
        const OsrsInventoryCell* cell = &s->player.inventory_cells[i];
        if (osrs_inventory_cell_is_empty(cell)) continue;
        OsrsInventoryClickResolution r =
            osrs_inventory_cell_click_interpret(cell, OSRS_CLICK_TICK_FIRST);
        int per_dose = osrs_consumable_hp_heal_amount(
            r.consumable_kind, s->player.base_hitpoints);
        if (per_dose <= 0) continue;
        int doses = r.dose_count > 0 ? (int)r.dose_count : 1;
        total += (float)(per_dose * doses);
    }
    return total;
}

static void col_record_death_attribution(ColosseumState* s) {
    int kt = s->last_hit_by_type;
    if (kt >= 0 && kt < COLO_NUM_NPC_TYPES) s->log.death_by_type[kt] += 1.0f;
    if (s->last_damage_source == COLO_DAMAGE_NPC_ATTACK &&
            kt >= 0 &&
            kt < COLO_NUM_NPC_TYPES) {
        s->log.npc_attack_death_by_type[kt] += 1.0f;
    }
    if (s->last_damage_source >= 0 &&
            s->last_damage_source < COLO_NUM_DAMAGE_SOURCES)
        s->log.death_by_source[s->last_damage_source] += 1.0f;
    if (col_mod_doom_lethal(s) &&
            s->last_doom_damage_source >= 0 &&
            s->last_doom_damage_source < COLO_NUM_DAMAGE_SOURCES)
        s->log.doom_death_by_source[s->last_doom_damage_source] += 1.0f;
    s->log.death_fatal_damage += s->tick_scratch.damage_received;
    if (__builtin_popcount(s->tick_scratch.styles_faced_mask) >= 2)
        s->log.death_on_conflict_tick += 1.0f;
    s->log.death_dmg_unprayable += s->tick_scratch.landed_unprayable_damage;
    s->log.death_dmg_offpray += s->tick_scratch.landed_offpray_damage;
    s->log.death_dmg_prayed += s->tick_scratch.landed_prayed_damage;
    s->log.death_dmg_self += s->tick_scratch.landed_self_damage;
    s->log.death_heal_remaining += col_inventory_heal_remaining(s);
}

static void col_classify_tick_offpray(ColosseumState* s) {
    float d = s->tick_scratch.offpray_damage_this_tick;
    if (d <= 0.0f) return;
    if (__builtin_popcount(s->tick_scratch.styles_faced_mask) >= 2)
        s->log.offpray_damage_conflict += d;
    else
        s->log.offpray_damage_solo += d;
}

typedef struct {
    int player_phase_reached;
    int spawn_wave_now;
    int in_wave_gap;
    int in_ready_gap;
} ColoPrePlayerPhase;

static ColoPrePlayerPhase col_advance_pre_player_phase_ctx(
    ColosseumState* s,
    ColosseumContext* ctx,
    const int* actions
) {
    ColoPrePlayerPhase phase = {0};
#ifdef COLO_PROFILE_ENABLED
    int col_prof_enabled =
        COLO_PROFILE_ENABLED() && !s->profiling_suppressed;
    double col_prof_t0 = col_prof_enabled ? COLO_PROFILE_NOW_MS() : 0.0;
#endif
    s->reward = 0.0f;
    s->tick_scratch = (ColoTickScratch){0};

    s->active_player_damage_received_scale = ctx->config.player_damage_received_scale;
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_PRE_RESET_SCRATCH);
#endif
    col_context_clear_render_events(ctx);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_PRE_RESET_RENDER);
#endif
    s->player_attack_timing = (EncounterProjectileTiming){0};
    encounter_clear_tick_flags(&s->player);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_PRE_RESET_PLAYER);
#endif
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        ColoNPC* npc = &s->npcs[i];
        if (!npc->active) continue;
        npc->attacked_this_tick = 0;
        npc->attack_style_this_tick = ATTACK_STYLE_NONE;
        npc->moved_last_tick = npc->moved_this_tick;
        npc->moved_this_tick = 0;
        npc->hit_landed_this_tick = 0;
        npc->hit_damage = 0;
        npc->hit_was_successful_this_tick = 0;
    }
    s->tick++;
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_PRE_RESET_NPCS);
#endif

    if (s->tick >= s->episode_max_ticks) {
        col_accumulate_tick_stats(s, ctx);
        s->episode_over = 1;
        s->time_limit_truncated = 1;
        s->winner = COLO_OUTCOME_PLAYER_DIED;
        s->log.rew_timeout -= ctx->config.timeout_penalty;
        s->reward = -ctx->config.timeout_penalty;
        col_record_terminal_outcome_score(s);
        s->log.max_wave_depth = col_episode_max_wave_depth(s);
        col_record_reward_clamp_headroom(s);
        s->episode_return += s->reward;
        return phase;
    }
    col_player_pretick(s, ctx, actions);
    s->player.prayer_display = s->player.prayer;
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_PRE_PLAYER_PRETICK);
#endif

    if (s->wave_ready_delay > 0) s->wave_ready_delay--;
    if (s->wave_attack_delay > 0) s->wave_attack_delay--;

    if (s->wave_spawn_delay > 0) {
        s->wave_spawn_delay--;
        if (s->wave_spawn_delay == 0) phase.spawn_wave_now = 1;
    }

    if (phase.spawn_wave_now) {
        s->wave = s->wave_spawn_target;
        col_spawn_wave(s, ctx);
    }

    phase.in_wave_gap =
        s->wave_spawn_delay > 0 ||
        phase.spawn_wave_now ||
        s->modifiers.draft_pending;
    phase.in_ready_gap = s->wave_ready_delay > 0;
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_PRE_SPAWN);
#endif

    col_resolve_player_projectiles_on_npcs_ctx(s, ctx);
    col_resolve_player_pending_hits_ctx(s, ctx);
    col_tick_npc_death_lingers(s, ctx);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_PRE_RESOLVE_HITS);
#endif

    if (!phase.in_wave_gap && !phase.in_ready_gap) {

        if (s->wave_attack_delay == 0 && s->warband_cycle_anchor < 0)
            s->warband_cycle_anchor = s->tick - 1;
        col_tick_live_stat_drift_and_divines(s);
        col_rebuild_player_collision_flags(s);
#ifdef COLO_PROFILE_ENABLED
        COLO_PROFILE_MARK(COLO_PROF_PRE_COLLISION);
#endif
        col_tick_npcs_ctx(s, ctx);
        if (s->wave_attack_delay == 0) col_warband_attack_phase(s, ctx);
        col_modifiers_tick(s, ctx->route_topology);

        if (s->surge_cooldown > 0) s->surge_cooldown--;
    }
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_PRE_NPC_PHASE);
#endif

    col_classify_tick_offpray(s);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_PRE_OFFPRAY);
#endif

    if (s->player.current_hitpoints <= 0 || col_mod_doom_lethal(s)) {
        col_accumulate_tick_stats(s, ctx);
        col_record_death_attribution(s);
        s->episode_over = 1;
        s->winner = COLO_OUTCOME_PLAYER_DIED;
        s->reward = col_terminal_death_reward(s, ctx);
        col_record_terminal_outcome_score(s);
        s->log.max_wave_depth = col_episode_max_wave_depth(s);
        col_record_reward_clamp_headroom(s);
        s->episode_return += s->reward;
        return phase;
    }
    phase.player_phase_reached = 1;
    return phase;
}

static void col_step_ctx(EncounterState* state, EncounterContext* context, const int* actions) {
    ColosseumState* s = (ColosseumState*)state;
    ColosseumContext* ctx = col_require_context(context);
    encounter_arena_topology_require_finalized(ctx->route_topology);
    if (s->episode_over) return;
#ifdef COLO_PROFILE_ENABLED
    int col_prof_enabled =
        COLO_PROFILE_ENABLED() && !s->profiling_suppressed;
    double col_prof_t0 = col_prof_enabled ? COLO_PROFILE_NOW_MS() : 0.0;
#endif

    ColoPrePlayerPhase phase =
        col_advance_pre_player_phase_ctx(s, ctx, actions);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_STEP_PRE_PLAYER);
#endif
    if (!phase.player_phase_reached) return;

    int can_player_attack = !phase.in_wave_gap;
    int px = s->player.x, py = s->player.y;
    col_tick_player_ctx(s, ctx, actions, can_player_attack);
    s->tick_scratch.player_moved = (s->player.x != px || s->player.y != py) ? 1 : 0;
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_STEP_PLAYER);
#endif

    col_accumulate_tick_stats(s, ctx);
    ColoShapedReward shaped = col_shaped_reward(s, ctx);
    s->reward = shaped.damage + shaped.boss_phase;
    int shaped_paid = 1;
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_STEP_REWARD);
#endif

    int all_dead = 1;
    int wave_cleared = 0;

    if (s->player.current_hitpoints <= 0 || col_mod_doom_lethal(s)) {
        col_record_death_attribution(s);
        s->episode_over = 1;
        s->winner = COLO_OUTCOME_PLAYER_DIED;
        s->reward = col_terminal_death_reward(s, ctx);
        shaped_paid = 0;
        goto finish_step;
    }

    if (phase.spawn_wave_now ||
            s->wave_spawn_delay > 0 ||
            s->modifiers.draft_pending) {
        goto finish_step;
    }

    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        if (col_npc_is_live_enemy(&s->npcs[i])) {
            all_dead = 0;
            break;
        }
    }
    wave_cleared = all_dead;
    if (s->wave == COLO_WAVE_BOSS && s->sol.started)
        wave_cleared = col_sol_find_idx(s) < 0;

    if (!all_dead && s->reinforcement_timer > 0) {
        s->reinforcement_timer--;
        if (s->reinforcement_timer == 0)
            col_spawn_reinforcements(s, ctx);
    }

    if (wave_cleared) {
        s->tick_scratch.wave_completed = 1;
        col_modifiers_on_wave_cleared(s);
        s->log.waves_cleared++;
        if (s->wave + 1 >= COLO_NUM_WAVES) {
            s->episode_over = 1;
            s->winner = COLO_OUTCOME_PLAYER_WON;
            s->reward = ctx->config.win_bonus;
            s->log.rew_win += ctx->config.win_bonus;
            shaped_paid = 0;
        } else {
            s->reward += ctx->config.wave_clear_bonus;
            s->log.rew_wave_clear += ctx->config.wave_clear_bonus;

            s->wave_spawn_target = s->wave + 1;
            col_record_wave_entry(s, s->wave + 1);
            col_modifier_open_draft(s, s->wave + 1);
            if (!s->modifiers.draft_pending)
                s->wave_spawn_delay = col_wave_entry_delay_ticks(s->wave_spawn_target);
            s->log.wave_reached = s->wave + 1;
        }
    }

finish_step:
    if (shaped_paid) {
        s->log.rew_damage += shaped.damage;
        s->log.rew_boss_phase += shaped.boss_phase;
    }
    if (s->episode_over) {
        col_record_terminal_outcome_score(s);
        s->log.max_wave_depth = col_episode_max_wave_depth(s);
    }
    col_record_reward_clamp_headroom(s);
    s->episode_return += s->reward;
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_STEP_WAVE_LOGIC);
#endif
}
