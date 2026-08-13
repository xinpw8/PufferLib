#line 76 "encounter_colosseum.h"

static const ColoBestGear (*col_get_best_gear_table(ColosseumState* s))[COLO_NUM_NPC_TYPES];
static float col_expected_dpt_for_equipment_vs_npc(
    ColosseumState* s, const uint8_t equipment[NUM_GEAR_SLOTS],
    const ColoNPC* npc, int use_style_matched_prayer);

static float col_terminal_death_reward(const ColosseumContext* ctx) {
    return -ctx->config.death_penalty_coeff;
}

static float col_terminal_timeout_reward(const ColosseumContext* ctx) {
    return -ctx->config.timeout_penalty;
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

static float col_episode_outcome_score(const ColosseumState* s) {
    if (s->episode_over && s->winner == COLO_OUTCOME_PLAYER_WON) return 1.0f;
    int waves_cleared = s->log.waves_cleared;
    if (waves_cleared > COLO_WAVE_BOSS) waves_cleared = COLO_WAVE_BOSS;
    float depth = (float)waves_cleared +
        col_current_wave_score_progress(s) +
        col_sol_score_progress(s);
    float ratio = depth / (float)COLO_NUM_WAVES;
    return 0.99f * ratio * ratio;
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

static int col_distinct_live_threat_styles(const ColosseumState* s) {
    int magic, ranged, melee;
    col_live_threat_style_counts(s, &magic, &ranged, &melee);
    return (magic > 0) + (ranged > 0) + (melee > 0);
}

static float col_rewardable_avoided_damage(ColosseumState* s) {
    ColoTickScratch* t = &s->tick_scratch;
    if (__builtin_popcount((unsigned)t->styles_faced_mask) != 1) return 0.0f;
    int style = __builtin_ctz((unsigned)t->styles_faced_mask);
    float raw_avoided = t->avoidable_threat_by_style[style] - t->offpray_damage_this_tick;
    if (raw_avoided <= 0.0f) return 0.0f;
    float progress_budget = s->log.total_damage_dealt - s->avoided_damage_already_rewarded;
    if (progress_budget <= 0.0f) return 0.0f;
    return raw_avoided < progress_budget ? raw_avoided : progress_budget;
}

static float col_attacked_gear_quality_ratio(ColosseumState* s) {
    if (!s->tick_scratch.player_attacked) return -1.0f;
    int npc_slot = s->player_attack_npc_idx;
    if (npc_slot < 0 || npc_slot >= COLO_MAX_NPCS) return -1.0f;
    const ColoNPC* npc = &s->npcs[npc_slot];
    if (!col_npc_is_live_target(npc) || col_type_is_hazard_entity(npc->type))
        return -1.0f;
    const ColoBestGear (*best)[COLO_NUM_NPC_TYPES] = col_get_best_gear_table(s);
    float best_dpt = 0.0f;
    for (int set = 0; set < COLO_NUM_WEAPON_SETS; set++) {
        float d = best[set][npc->type].dpt;
        if (d > best_dpt) best_dpt = d;
    }
    if (best_dpt <= 0.0f) return -1.0f;

    float actual = col_expected_dpt_for_equipment_vs_npc(s, s->player.equipped, npc, 1);
    float ratio = actual / best_dpt;
    if (ratio < 0.0f) ratio = 0.0f;
    if (ratio > 1.0f) ratio = 1.0f;
    return ratio;
}

static int col_attacked_with_offensive_boost(ColosseumState* s) {
    if (!s->tick_scratch.player_attacked) return -1;
    int npc_slot = s->player_attack_npc_idx;
    if (npc_slot < 0 || npc_slot >= COLO_MAX_NPCS) return -1;
    const ColoNPC* npc = &s->npcs[npc_slot];
    if (!col_npc_is_live_target(npc) || col_type_is_hazard_entity(npc->type))
        return -1;
    AttackStyle style = (AttackStyle)s->player_attack_style_id;
    int cur_att = 0, cur_str = 0;
    col_current_style_levels(&s->player, style, &cur_att, &cur_str);
    int base_att, base_str;
    if (style == ATTACK_STYLE_RANGED) {
        base_att = base_str = s->player.base_ranged;
    } else if (style == ATTACK_STYLE_MAGIC) {
        base_att = base_str = s->player.base_magic;
    } else {
        base_att = s->player.base_attack;
        base_str = s->player.base_strength;
    }
    return (cur_att > base_att || cur_str > base_str) ? 1 : 0;
}

static float col_compute_reward_ctx(ColosseumState* s, const ColosseumContext* ctx) {
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

    float reward = 0.0f;
    float rewarded_fresh = t->fresh_damage_dealt;
    if (s->wave < cfg->farm_cap_waves) {
        s->log.farm_damage += t->fresh_damage_reinforcement;
        if (cfg->farm_safe_damage_cap)
            rewarded_fresh -= t->fresh_damage_reinforcement;
    }
    reward += cfg->damage_reward_coeff * rewarded_fresh;
    reward += cfg->boss_damage_reward_coeff * t->fresh_damage_boss;
    reward += cfg->prayer_correct_reward * (float)t->prayer_correct;
    reward -= cfg->offpray_damage_penalty_coeff * t->offpray_damage_this_tick;
    if (cfg->multistyle_exposure_penalty_coeff != 0.0f) {
        int excess_styles = col_distinct_live_threat_styles(s) - 1;
        if (excess_styles > 0)
            reward -= cfg->multistyle_exposure_penalty_coeff * (float)excess_styles;
    }
    if (cfg->argmax_gear_reward_coeff != 0.0f) {
        float gear_quality = col_attacked_gear_quality_ratio(s);
        if (gear_quality >= 0.0f)
            reward += cfg->argmax_gear_reward_coeff * gear_quality;
    }
    if (cfg->offensive_boost_reward_coeff != 0.0f) {
        if (col_attacked_with_offensive_boost(s) > 0)
            reward += cfg->offensive_boost_reward_coeff;
    }
    float rewardable_avoided = col_rewardable_avoided_damage(s);
    reward += cfg->avoided_damage_coeff * rewardable_avoided;
    s->avoided_damage_already_rewarded += rewardable_avoided;

    if (s->sol.started && s->sol.phase > s->rewarded_boss_phase) {
        reward += cfg->boss_phase_bonus * (float)(s->sol.phase - s->rewarded_boss_phase);
        s->rewarded_boss_phase = s->sol.phase;
    }
    return reward;
}

static float col_inventory_heal_remaining(const ColosseumState* s) {
    float total = 0.0f;
    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++) {
        const OsrsInventoryCell* cell = &s->inventory_cells[i];
        if (cell->raw_osrs_id == 0) continue;
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

static void col_step_ctx(EncounterState* state, EncounterContext* context, const int* actions) {
    ColosseumState* s = (ColosseumState*)state;
    ColosseumContext* ctx = context ? (ColosseumContext*)context : col_legacy_context();
    if (s->episode_over) return;

    s->reward = 0.0f;
    s->tick_scratch = (ColoTickScratch){0};

    s->active_player_damage_received_scale = ctx->config.player_damage_received_scale;
    col_context_clear_render_events(ctx);
    s->player_attack_timing = (EncounterProjectileTiming){0};
    encounter_clear_tick_flags(&s->player);
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        s->npcs[i].attacked_this_tick = 0;
        s->npcs[i].attack_style_this_tick = ATTACK_STYLE_NONE;
        s->npcs[i].moved_last_tick = s->npcs[i].moved_this_tick;
        s->npcs[i].moved_this_tick = 0;
        s->npcs[i].hit_landed_this_tick = 0;
        s->npcs[i].hit_damage = 0;
        s->npcs[i].hit_was_successful_this_tick = 0;
    }
    s->tick++;

    if (s->tick >= s->episode_max_ticks) {
        (void)col_compute_reward_ctx(s, ctx);
        s->episode_over = 1;
        s->time_limit_truncated = 1;
        s->winner = COLO_OUTCOME_PLAYER_DIED;
        s->reward = col_terminal_timeout_reward(ctx);
        col_record_terminal_outcome_score(s);
        s->log.max_wave_depth = col_episode_max_wave_depth(s);
        s->episode_return += s->reward;
        return;
    }
    col_player_pretick(s, ctx, actions);
    if (ctx->config.prayer_oracle_mode) col_apply_prayer_oracle(s);
    s->player.prayer_display = s->player.prayer;

    if (s->wave_ready_delay > 0) s->wave_ready_delay--;
    if (s->wave_attack_delay > 0) s->wave_attack_delay--;

    int spawn_wave_now = 0;
    if (s->wave_spawn_delay > 0) {
        s->wave_spawn_delay--;
        if (s->wave_spawn_delay == 0) spawn_wave_now = 1;
    }

    if (spawn_wave_now) {
        s->wave = s->wave_spawn_target;
        col_spawn_wave(s);
    }

    int in_wave_gap = (s->wave_spawn_delay > 0 || spawn_wave_now ||
                       s->modifiers.draft_pending);
    int in_ready_gap = s->wave_ready_delay > 0;

    col_resolve_player_projectiles_on_npcs_ctx(s, ctx);
    col_resolve_player_pending_hits_ctx(s, ctx);
    col_tick_npc_death_lingers(s);

    if (!in_wave_gap && !in_ready_gap) {

        if (s->wave_attack_delay == 0 && s->warband_cycle_anchor < 0)
            s->warband_cycle_anchor = s->tick - 1;
        col_tick_live_stat_drift_and_divines(s);
        col_rebuild_player_collision_flags(s);
        col_tick_npcs_ctx(s, ctx);
        col_modifiers_tick(s);

        if (s->surge_cooldown > 0) s->surge_cooldown--;
    }

    col_classify_tick_offpray(s);

    if (ctx->config.invuln_mode) {
        s->player.current_hitpoints = s->player.base_hitpoints;
    } else if (s->player.current_hitpoints <= 0 || col_mod_doom_lethal(s)) {
        (void)col_compute_reward_ctx(s, ctx);
        col_record_death_attribution(s);
        s->episode_over = 1;
        s->winner = COLO_OUTCOME_PLAYER_DIED;
        s->reward = col_terminal_death_reward(ctx);
        col_record_terminal_outcome_score(s);
        s->log.max_wave_depth = col_episode_max_wave_depth(s);
        s->episode_return += s->reward;
        return;
    }

    int can_player_attack = !in_wave_gap;
    int px = s->player.x, py = s->player.y;
    col_tick_player_ctx(s, ctx, actions, can_player_attack);
    s->tick_scratch.player_moved = (s->player.x != px || s->player.y != py) ? 1 : 0;

    if (!in_wave_gap && !in_ready_gap && s->wave_attack_delay == 0)
        col_warband_attack_phase(s, ctx);

    s->reward = col_compute_reward_ctx(s, ctx);

    if (can_player_attack && ctx->config.stall_penalty_coeff != 0.0f)
        s->reward -= ctx->config.stall_penalty_coeff;
    int all_dead = 1;
    int wave_cleared = 0;

    if (ctx->config.invuln_mode) {
        s->player.current_hitpoints = s->player.base_hitpoints;
    } else if (s->player.current_hitpoints <= 0 || col_mod_doom_lethal(s)) {
        col_record_death_attribution(s);
        s->episode_over = 1;
        s->winner = COLO_OUTCOME_PLAYER_DIED;
        s->reward = col_terminal_death_reward(ctx);
        goto finish_step;
    }

    if (spawn_wave_now || s->wave_spawn_delay > 0 || s->modifiers.draft_pending)
        goto finish_step;

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
        if (s->reinforcement_timer == 0) col_spawn_reinforcements(s);
    }

    if (wave_cleared) {
        s->tick_scratch.wave_completed = 1;
        s->log.waves_cleared++;
        s->reward += ctx->config.wave_clear_bonus;
        if (s->wave + 1 >= COLO_NUM_WAVES) {
            s->episode_over = 1;
            s->winner = COLO_OUTCOME_PLAYER_WON;
            s->reward += ctx->config.win_bonus;
        } else {

            s->wave_spawn_target = s->wave + 1;
            col_record_wave_entry(s, s->wave + 1);
            col_modifier_open_draft(s, s->wave + 1);
            if (!s->modifiers.draft_pending)
                s->wave_spawn_delay = col_wave_entry_delay_ticks(s->wave_spawn_target);
            s->log.wave_reached = s->wave + 1;
        }
    }

finish_step:
    if (s->episode_over) {
        col_record_terminal_outcome_score(s);
        s->log.max_wave_depth = col_episode_max_wave_depth(s);
    }
    s->episode_return += s->reward;
}

static void col_step(EncounterState* state, const int* actions) {
    col_step_ctx(state, (EncounterContext*)col_legacy_context(), actions);
}
