#line 4619 "encounter_inferno.h"
static int inf_healer_is_actively_healing(const InfernoState* s, const InfNPC* npc) {
    if (!npc->active || npc->death_ticks > 0) return 0;
    if (npc->type != INF_NPC_HEALER_JAD && npc->type != INF_NPC_HEALER_ZUK) return 0;
    if (npc->aggro_target < 0 || npc->aggro_target >= INF_MAX_NPCS) return 0;
    return s->npcs[npc->aggro_target].active;
}

static int inf_npc_type_is_actively_healed(
    const InfernoState* s,
    InfNPCType target_type
) {
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        const InfNPC* healer = &s->npcs[i];
        if (!inf_healer_is_actively_healing(s, healer)) continue;
        const InfNPC* target = &s->npcs[healer->aggro_target];
        if (target->type == target_type) return 1;
    }
    return 0;
}

static int inf_count_untagged_zuk_healers(const InfernoState* s) {
    int count = 0;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        const InfNPC* healer = &s->npcs[i];
        if (!inf_healer_is_actively_healing(s, healer)) continue;
        if (healer->type != INF_NPC_HEALER_ZUK) continue;

        const InfNPC* target = &s->npcs[healer->aggro_target];
        if (target->type == INF_NPC_ZUK) count++;
    }
    return count;
}

static int inf_count_post_healer_set_pressure(const InfernoState* s) {
    int count = 0;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        const InfNPC* npc = &s->npcs[i];
        if (!npc->active || npc->death_ticks != 0 || npc->hp <= 0)
            continue;
        if (inf_npc_type_is_set_pressure(npc->type))
            count++;
    }
    return count;
}

static inline float inf_dmg_below_threshold(float old_hp, float new_hp, float t) {
    if (new_hp >= old_hp) return 0.0f;
    float a = old_hp < t ? old_hp : t;
    float b = new_hp < t ? new_hp : t;
    float d = a - b;
    return d > 0.0f ? d : 0.0f;
}

typedef enum {
    INF_ZUK_LOWWATERMARK_TRACK_ONLY,
    INF_ZUK_LOWWATERMARK_REWARD_ALLOWED,
    INF_ZUK_LOWWATERMARK_REWARD_BLOCKED_EXCEPT_THRESHOLD,
} InfZukLowWatermarkMode;

static float inf_zuk_low_watermark_reward(
    InfernoState* s,
    const InfernoContext* ctx,
    InfZukLowWatermarkMode mode
) {
    int zuk_idx = inf_find_live_zuk_idx(s);
    if (zuk_idx < 0) return 0.0f;

    float zuk_hp = (float)s->npcs[zuk_idx].hp;
    if (zuk_hp >= s->min_zuk_hp_seen) return 0.0f;

    float old_min = s->min_zuk_hp_seen;
    float dmg = old_min - zuk_hp;
    int crosses_zuk_healer_threshold = old_min > 240.0f && zuk_hp <= 240.0f;
    int should_pay_zuk_damage =
        mode == INF_ZUK_LOWWATERMARK_REWARD_ALLOWED ||
        (mode == INF_ZUK_LOWWATERMARK_REWARD_BLOCKED_EXCEPT_THRESHOLD &&
            crosses_zuk_healer_threshold);
    float reward = should_pay_zuk_damage ? ctx->config.damage_reward_coeff * dmg : 0.0f;
    if (s->jad_killed_this_episode) {
        reward *= ctx->config.post_jad_zuk_multiplier;
    } else if (s->zuk.jad_spawned) {
        reward *= ctx->config.jad_alive_zuk_multiplier;
    }
    s->min_zuk_hp_seen = zuk_hp;

    if (s->tick_at_le_300 < 0 && zuk_hp <= 300.0f) s->tick_at_le_300 = s->tick;
    if (s->tick_at_le_240 < 0 && zuk_hp <= 240.0f) s->tick_at_le_240 = s->tick;
    if (s->tick_at_le_150 < 0 && zuk_hp <= 150.0f) s->tick_at_le_150 = s->tick;
    s->damage_after_300 += inf_dmg_below_threshold(old_min, zuk_hp, 300.0f);
    s->damage_after_240 += inf_dmg_below_threshold(old_min, zuk_hp, 240.0f);
    s->damage_after_150 += inf_dmg_below_threshold(old_min, zuk_hp, 150.0f);

    return reward;
}

static float inf_joseph_heal_damped_damage(float damage, float max_hp, float total_healed) {
    if (damage <= 0.0f) return 0.0f;
    float denom = max_hp + 4.0f * total_healed;
    if (denom <= 0.0f) return damage;
    return damage * max_hp / denom;
}

static float inf_zuk_healer_attack_shape_reward(
    const InfernoState* s,
    const InfernoContext* ctx
) {
    return
        ctx->config.zuk_untagged_healer_nonmagic_attack_bonus_coeff *
            (float)s->tick_scratch.zuk_untagged_healer_nonmagic_attacks -
        ctx->config.zuk_healer_mage_attack_penalty_coeff *
            (float)s->tick_scratch.zuk_healer_mage_attack_fires;
}

static float inf_offensive_prayer_damage_reward(
    const InfernoState* s,
    const InfernoContext* ctx
) {
    return ctx->config.offensive_prayer_reward_coeff *
        ctx->config.damage_reward_coeff *
        s->tick_scratch.offensive_prayer_correct_damage_roll;
}

static float inf_compute_joseph_reward(
    const InfernoState* s,
    const InfernoContext* ctx,
    int healer_is_actively_healing
) {
    float reward = 0.0f;
    if (healer_is_actively_healing) {
        reward = ctx->config.tag_reward_coeff * (float)s->tick_scratch.healer_tags;
    } else {
        float rewarded_other_damage = fmaxf(
            0.0f,
            s->tick_scratch.damage_dealt -
                s->tick_scratch.damage_jad -
                s->tick_scratch.damage_zuk -
                s->tick_scratch.damage_jad_healers);
        reward = ctx->config.damage_reward_coeff * (
            rewarded_other_damage +
            inf_joseph_heal_damped_damage(
                s->tick_scratch.damage_jad,
                (float)INF_NPC_STATS[INF_NPC_JAD].hp,
                s->total_hp_restored_jad) +
            inf_joseph_heal_damped_damage(
                s->tick_scratch.damage_zuk,
                (float)INF_NPC_STATS[INF_NPC_ZUK].hp,
                s->total_hp_restored_zuk));
        reward += inf_offensive_prayer_damage_reward(s, ctx);
    }
    reward -= ctx->config.shield_penalty_coeff * s->tick_scratch.shield_damage;
    reward += inf_zuk_healer_attack_shape_reward(s, ctx);
    return reward;
}

static float inf_compute_reward_ctx(InfernoState* s, const InfernoContext* ctx) {
    s->total_damage_dealt += s->tick_scratch.damage_dealt;
    s->total_zuk_healer_damage += s->tick_scratch.damage_zuk_healers;
    s->total_damage_received += s->tick_scratch.damage_received;
    s->total_hp_restored += s->tick_scratch.hp_restored;
    s->total_hp_restored_jad += s->tick_scratch.hp_restored_jad;
    s->total_hp_restored_zuk += s->tick_scratch.hp_restored_zuk;

    if (s->tick_scratch.kill_jad > 0) s->jad_killed_this_episode = 1;

    if (s->episode_over) {
        if (s->winner == INF_OUTCOME_PLAYER_WON) return 1.0f;
        if (ctx->config.terminal_penalty_enabled) return -1.0f;
        return -ctx->config.death_penalty_coeff;
    }

    int jad_is_actively_healed =
        inf_npc_type_is_actively_healed(s, INF_NPC_JAD);
    int zuk_is_actively_healed =
        inf_npc_type_is_actively_healed(s, INF_NPC_ZUK);
    int healer_is_actively_healing =
        jad_is_actively_healed || zuk_is_actively_healed;
    int tags_first_gate = inf_zuk_healer_tags_first_reward_gate_active(s, ctx);

    if (ctx->config.joseph_reward_mode == INF_JOSEPH_REWARD_MODE_ON) {
        if (inf_is_final_wave(s))
            inf_zuk_low_watermark_reward(s, ctx, INF_ZUK_LOWWATERMARK_TRACK_ONLY);
        return inf_compute_joseph_reward(s, ctx, healer_is_actively_healing);
    }

    float reward = 0.0f;
    if (inf_is_final_wave(s)) {
        if (tags_first_gate) {
            reward = inf_zuk_low_watermark_reward(
                s, ctx, INF_ZUK_LOWWATERMARK_TRACK_ONLY);
            reward += ctx->config.tag_reward_coeff * (float)s->tick_scratch.healer_tags;
            reward += inf_zuk_healer_attack_shape_reward(s, ctx);
            return reward;
        }
        int use_zuk_healer_phase_hp_delta =
            s->zuk.healer_spawned &&
            ctx->config.zuk_healer_phase_hp_delta_coeff > 0.0f;
        InfZukLowWatermarkMode watermark_mode =
            INF_ZUK_LOWWATERMARK_REWARD_ALLOWED;
        if (use_zuk_healer_phase_hp_delta) {
            watermark_mode = INF_ZUK_LOWWATERMARK_TRACK_ONLY;
        } else if (zuk_is_actively_healed) {
            watermark_mode = INF_ZUK_LOWWATERMARK_REWARD_BLOCKED_EXCEPT_THRESHOLD;
        }
        reward = inf_zuk_low_watermark_reward(s, ctx, watermark_mode);
        if (use_zuk_healer_phase_hp_delta) {
            reward += ctx->config.zuk_healer_phase_hp_delta_coeff *
                (s->tick_scratch.damage_zuk - s->tick_scratch.hp_restored_zuk);
        }
        reward += ctx->config.damage_reward_coeff * s->tick_scratch.damage_zuk_healers;
        if (healer_is_actively_healing) {
            float generic_heal_cost = s->tick_scratch.hp_restored;
            if (use_zuk_healer_phase_hp_delta)
                generic_heal_cost -= s->tick_scratch.hp_restored_zuk;
            if (generic_heal_cost < 0.0f) generic_heal_cost = 0.0f;
            reward -= ctx->config.damage_reward_coeff * generic_heal_cost;
        }
        if (s->zuk.healer_spawned &&
                ctx->config.zuk_untagged_healer_tick_penalty_coeff > 0.0f) {
            reward -= ctx->config.zuk_untagged_healer_tick_penalty_coeff *
                (float)inf_count_untagged_zuk_healers(s);
        }
    } else {
        float rewardable_damage = s->tick_scratch.damage_dealt -
            s->tick_scratch.damage_jad_healers -
            s->tick_scratch.damage_resurrected;
        if (jad_is_actively_healed)
            rewardable_damage -= s->tick_scratch.damage_jad;
        reward = ctx->config.damage_reward_coeff *
            fmaxf(0.0f, rewardable_damage - s->tick_scratch.hp_restored);
    }

    reward += inf_offensive_prayer_damage_reward(s, ctx);
    reward += ctx->config.tag_reward_coeff * (float)s->tick_scratch.healer_tags;
    reward += inf_zuk_healer_attack_shape_reward(s, ctx);
    reward += ctx->config.shield_tag_reward_coeff * (float)s->tick_scratch.shield_tags;
    reward += ctx->config.zuk_untagged_healer_target_bonus_coeff *
        (float)s->tick_scratch.zuk_untagged_healer_targets;
    reward += ctx->config.zuk_safe_untagged_healer_target_bonus_coeff *
        (float)s->tick_scratch.zuk_safe_untagged_healer_targets;

    int zuk_idx = inf_find_live_zuk_idx(s);
    int zuk_hp_now = (zuk_idx >= 0) ? s->npcs[zuk_idx].hp : 1200;
    int late_add_reward_active = s->zuk.jad_spawned || zuk_hp_now <= 600;
    if (late_add_reward_active) {
        float rewardable_jad_damage =
            jad_is_actively_healed ? 0.0f : s->tick_scratch.damage_jad;
        reward += ctx->config.jad_damage_reward_coeff * rewardable_jad_damage;
        reward += ctx->config.zuk_healer_damage_reward_coeff * s->tick_scratch.damage_zuk_healers;
        reward += ctx->config.set_damage_reward_coeff * s->tick_scratch.damage_set;
        if (s->tick_at_all_zuk_healers_dead >= 0) {
            reward += ctx->config.post_healer_zuk_damage_coeff * s->tick_scratch.damage_zuk;
            float post_healer_set_damage_reward =
                ctx->config.post_healer_set_damage_reward_coeff * s->tick_scratch.damage_set;
            float post_healer_set_kill_bonus =
                (float)s->tick_scratch.kill_set * ctx->config.post_healer_set_kill_bonus;
            reward += post_healer_set_damage_reward;
            s->post_healer_set_damage_reward_total +=
                post_healer_set_damage_reward;
            s->post_healer_set_kill_bonus_total += post_healer_set_kill_bonus;
            s->pending_set_kill_bonus += post_healer_set_kill_bonus;
            if (ctx->config.post_healer_set_alive_tick_penalty_coeff > 0.0f) {
                int set_pressure = inf_count_post_healer_set_pressure(s);
                s->post_healer_set_pressure_total += (float)set_pressure;
                float penalty = ctx->config.post_healer_set_alive_tick_penalty_coeff *
                    (float)set_pressure;
                if (ctx->config.post_healer_set_alive_penalty_cap > 0.0f) {
                    float remaining = ctx->config.post_healer_set_alive_penalty_cap -
                        s->post_healer_set_alive_penalty_total;
                    if (remaining < 0.0f) remaining = 0.0f;
                    if (penalty > remaining) penalty = remaining;
                }
                s->post_healer_set_alive_penalty_total += penalty;
                reward -= penalty;
            }
        }
        s->pending_jad_kill_bonus +=
            (float)s->tick_scratch.kill_jad * ctx->config.jad_kill_bonus;
        s->pending_zuk_healer_kill_bonus +=
            (float)s->tick_scratch.kill_zuk_healer * ctx->config.zuk_healer_kill_bonus;
        s->pending_set_kill_bonus +=
            (float)s->tick_scratch.kill_set * ctx->config.set_kill_bonus;
    }
    if (s->pending_jad_kill_bonus > 0.0f) {
        float emit = fminf(0.07f, s->pending_jad_kill_bonus);
        reward += emit;
        s->pending_jad_kill_bonus -= emit;
    }
    if (s->pending_zuk_healer_kill_bonus > 0.0f) {
        float emit = fminf(0.05f, s->pending_zuk_healer_kill_bonus);
        reward += emit;
        s->pending_zuk_healer_kill_bonus -= emit;
    }
    if (s->pending_set_kill_bonus > 0.0f) {
        float emit = fminf(0.03f, s->pending_set_kill_bonus);
        reward += emit;
        s->pending_set_kill_bonus -= emit;
    }

    float shield_penalty = ctx->config.shield_penalty_coeff * s->tick_scratch.shield_damage;
    if (ctx->config.shield_penalty_episode_cap > 0.0f) {
        float remaining = ctx->config.shield_penalty_episode_cap -
            s->shield_penalty_episode_total;
        if (remaining < 0.0f) remaining = 0.0f;
        if (shield_penalty > remaining) shield_penalty = remaining;
    }
    s->shield_penalty_episode_total += shield_penalty;
    reward -= shield_penalty;

    if (ctx->config.phase_900_bonus > 0.0f && !s->phase_900_fired
            && s->min_zuk_hp_seen <= 900.0f && s->min_zuk_hp_seen > 0.0f) {
        reward += ctx->config.phase_900_bonus;
        s->phase_900_fired = 1;
    }
    if (ctx->config.phase_600_bonus > 0.0f && !s->phase_600_fired
            && s->min_zuk_hp_seen <= 600.0f && s->min_zuk_hp_seen > 0.0f) {
        reward += ctx->config.phase_600_bonus;
        s->phase_600_fired = 1;
    }
    if (ctx->config.phase_300_bonus > 0.0f && !s->phase_300_fired
            && s->min_zuk_hp_seen <= 300.0f && s->min_zuk_hp_seen > 0.0f) {
        reward += ctx->config.phase_300_bonus;
        s->phase_300_fired = 1;
    }
    return reward;
}

static float inf_compute_reward(InfernoState* s) {
    return inf_compute_reward_ctx(s, inf_legacy_context());
}

static float inf_terminal_loss_reward(
    const InfernoState* s,
    const InfernoContext* ctx
) {
    (void)s;
    return ctx->config.terminal_penalty_enabled ? -1.0f : 0.0f;
}

static void inf_update_healer_transition_stats(InfernoState* s) {
    s->total_shield_tags += s->tick_scratch.shield_tags;

    if (!s->zuk.healer_spawned) return;

    if (s->tick_at_zuk_healer_spawn < 0)
        s->tick_at_zuk_healer_spawn = s->tick;

    int zuk_idx = inf_find_live_zuk_idx(s);
    if (zuk_idx >= 0) {
        float hp = (float)s->npcs[zuk_idx].hp;
        if (s->tick_at_le_240 < 0 && hp <= 240.0f)
            s->tick_at_le_240 = s->tick;
        if (hp > s->zuk_hp_max_after_healer_spawn)
            s->zuk_hp_max_after_healer_spawn = hp;
    }

    s->total_zuk_healer_tags += s->tick_scratch.zuk_healer_tags;
    if (s->total_zuk_healer_tags > 4) s->total_zuk_healer_tags = 4;
    s->total_zuk_healer_kills += s->tick_scratch.kill_zuk_healer;
    if (s->total_zuk_healer_kills > 4) s->total_zuk_healer_kills = 4;

    if (s->total_zuk_healer_tags > 0 && s->tick_at_first_zuk_healer_tag < 0)
        s->tick_at_first_zuk_healer_tag = s->tick;
    if (s->total_zuk_healer_tags >= 4 && s->tick_at_all_zuk_healers_tagged < 0)
        s->tick_at_all_zuk_healers_tagged = s->tick;

    int live_zuk_healers = 0;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (!s->npcs[i].active || s->npcs[i].hp <= 0) continue;
        if (s->npcs[i].type == INF_NPC_HEALER_ZUK)
            live_zuk_healers++;
    }
    if (live_zuk_healers == 0 && s->tick_at_all_zuk_healers_dead < 0) {
        s->tick_at_all_zuk_healers_dead = s->tick;
        s->zuk_hp_at_all_zuk_healers_dead =
            zuk_idx >= 0 ? (float)s->npcs[zuk_idx].hp : 0.0f;
    }

    if (s->tick_at_all_zuk_healers_dead >= 0 && s->tick_scratch.damage_zuk > 0.0f) {
        s->damage_after_all_zuk_healers_dead += s->tick_scratch.damage_zuk;
        if (s->tick_at_first_zuk_hit_after_all_healers_dead < 0)
            s->tick_at_first_zuk_hit_after_all_healers_dead = s->tick;
    }

    if (s->tick_at_le_240 >= 0) {
        s->hp_restored_after_240 += s->tick_scratch.hp_restored;
        s->spark_damage_after_240 += s->tick_scratch.spark_damage;
    }
}

static int inf_has_live_player_target(const InfernoState* s) {
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (inf_npc_is_live_player_target(s, i)) return 1;
    }
    return 0;
}

static int inf_has_attackable_player_target(const InfernoState* s) {
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (inf_npc_is_live_player_target(s, i) &&
                inf_player_can_attack_npc_from_current_tile(s, i))
            return 1;
    }
    return 0;
}

static int inf_has_live_npc_type(const InfernoState* s, InfNPCType type) {
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        const InfNPC* npc = &s->npcs[i];
        if (npc->active && npc->death_ticks == 0 && npc->hp > 0 &&
                npc->type == type)
            return 1;
    }
    return 0;
}

static InfIdleDiagnosticPhase inf_idle_diagnostic_phase(const InfernoState* s) {
    if (!inf_is_final_wave(s))
        return inf_has_live_npc_type(s, INF_NPC_JAD)
            ? INF_IDLE_PHASE_JAD : INF_IDLE_PHASE_SET;
    if (inf_has_live_npc_type(s, INF_NPC_HEALER_ZUK))
        return INF_IDLE_PHASE_ZUK_HEALERS;
    if (inf_has_live_npc_type(s, INF_NPC_JAD))
        return INF_IDLE_PHASE_ZUK_JAD;
    if (s->tick_at_all_zuk_healers_dead >= 0)
        return INF_IDLE_PHASE_ZUK_POST_HEALERS;
    return INF_IDLE_PHASE_ZUK_PRE_JAD;
}

static int inf_player_has_immediate_threat(const InfernoState* s) {
    if (s->tick_scratch.attacks_fired > 0) return 1;
    for (int h = 0; h < s->player_pending_hits.count; h++) {
        const EncounterPendingHit* ph = &s->player_pending_hits.hits[h];
        if (ph->check_prayer && inf_pending_hit_obs_timer(ph) <= 1)
            return 1;
    }
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        const InfNPC* npc = &s->npcs[i];
        InfNpcPlayerThreat threat = inf_npc_player_threat(s, npc);
        if (threat.telegraph_style_mask && threat.can_attack_this_tick)
            return 1;
    }
    return 0;
}

static int inf_progress_happened_this_tick(const InfernoState* s) {
    return s->tick_scratch.damage_dealt > 0.0f ||
        s->tick_scratch.healer_tags > 0 ||
        s->tick_scratch.shield_tags > 0 ||
        s->tick_scratch.kill_jad > 0 ||
        s->tick_scratch.kill_zuk_healer > 0 ||
        s->tick_scratch.kill_set > 0 ||
        s->tick_scratch.wave_completed > 0;
}

static void inf_record_idle_diagnostics(
    InfernoState* s,
    int can_player_attack,
    int attack_ready_no_attack,
    int target_available_no_attack,
    int safe_attack_opportunity_missed
) {
    if (!can_player_attack || !inf_has_live_player_target(s))
        return;

    InfIdleDiagnosticPhase phase = inf_idle_diagnostic_phase(s);
    assert((int)phase >= 0 && (int)phase < OSRS_INFERNO_IDLE_PHASE_COUNT);

    if (attack_ready_no_attack) {
        s->total_attack_ready_no_attack_ticks++;
        s->attack_ready_no_attack_ticks_by_phase[phase]++;
    }
    if (target_available_no_attack) {
        s->total_target_available_no_attack_ticks++;
        s->target_available_no_attack_ticks_by_phase[phase]++;
    }
    if (safe_attack_opportunity_missed) {
        s->total_safe_attack_opportunity_missed_ticks++;
        s->safe_attack_opportunity_missed_ticks_by_phase[phase]++;
    }
    if (!inf_progress_happened_this_tick(s)) {
        s->total_progressless_ticks++;
        s->progressless_ticks_by_phase[phase]++;
    }
}

static void inf_step_ctx(EncounterState* state, EncounterContext* context, const int* actions) {
    InfernoState* s = (InfernoState*)state;
    InfernoContext* ctx = context ? (InfernoContext*)context : inf_legacy_context();
    if (s->episode_over) return;

    s->reward = 0.0f;
    s->player_moved_last_tick = s->tick_scratch.player_moved;
    s->tick_scratch = (InfTickScratch){ .pillar_lost = -1 };
    s->player_attack_timing = (EncounterProjectileTiming){0};
    encounter_clear_tick_flags(&s->player);
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        s->npcs[i].attacked_this_tick = 0;
        s->npcs[i].attack_style_this_tick = ATTACK_STYLE_NONE;
        s->npcs[i].attack_visual_target = -1;
        s->npcs[i].resurrecting_this_tick = 0;
        s->npcs[i].resurrection_visual_target = -1;
        s->npcs[i].moved_this_tick = 0;
        s->npcs[i].hit_landed_this_tick = 0;
        s->npcs[i].hit_damage = 0;
        s->npcs[i].hit_was_successful_this_tick = 0;
        s->npcs[i].hit_spell_type = 0;
    }
    s->tick++;
    inf_player_pretick(s, ctx, actions);

    int spawn_wave_now = 0;
    if (s->wave_spawn_delay > 0) {
        s->wave_spawn_delay--;
        if (s->wave_spawn_delay == 0)
            spawn_wave_now = 1;
    }
    int in_wave_gap = (s->wave_spawn_delay > 0 || spawn_wave_now);
    if (s->wave_ready_delay > 0)
        s->wave_ready_delay--;
    int in_ready_gap = s->wave_ready_delay > 0;

    inf_resolve_player_projectiles_on_npcs(s);
    inf_resolve_player_pending_hits(s);
    inf_resolve_pending_sparks(s);

    if (!in_wave_gap && !in_ready_gap) {
        inf_rebuild_player_collision_flags(s);
        inf_invalidate_los_cache(s);
        inf_tick_npcs_ctx(s, ctx);
    }
    inf_update_healer_transition_stats(s);

    if (s->player.current_hitpoints <= 0) {
        (void)inf_compute_reward_ctx(s, ctx);
        if (s->last_hit_by_type >= 0 && s->last_hit_by_type < INF_NUM_NPC_TYPES)
            s->killed_by_type[s->last_hit_by_type]++;
        s->episode_over = 1;
        s->winner = INF_OUTCOME_PLAYER_DIED;
        s->reward = inf_terminal_loss_reward(s, ctx);
        s->episode_return += s->reward;
        return;
    }

    int can_player_attack = !in_wave_gap && !in_ready_gap;
    int attack_ready_no_attack = 0;
    int target_available_no_attack = 0;
    int safe_attack_opportunity_missed = 0;
    int player_x_before_tick_player = s->player.x;
    int player_y_before_tick_player = s->player.y;
    inf_tick_player_ctx(s, ctx, actions, can_player_attack);
    s->tick_scratch.player_moved =
        (s->player.x != player_x_before_tick_player ||
         s->player.y != player_y_before_tick_player) ? 1 : 0;
    if (s->tick_scratch.player_moved)
        inf_invalidate_los_cache(s);
    inf_resolve_jad_prayer_checks_after_player(s, ctx);

    {
        int has_alive_npc = can_player_attack && inf_has_live_player_target(s);
        if (has_alive_npc && s->player.attack_timer == 0 && !s->tick_scratch.player_attacked)
            s->ticks_without_action++;
        else
            s->ticks_without_action = 0;
        attack_ready_no_attack = s->ticks_without_action > 0;
        if (attack_ready_no_attack) {
            int target_available = inf_has_live_player_target(s);
            int attackable_target = inf_has_attackable_player_target(s);
            target_available_no_attack = target_available;
            safe_attack_opportunity_missed =
                attackable_target && !inf_player_has_immediate_threat(s);
        }
    }

    s->total_prayer_correct += s->tick_scratch.prayer_correct;
    int ranger_attacked_this_tick = 0;
    int mager_attacked_this_tick = 0;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (s->npcs[i].attacked_this_tick &&
                s->npcs[i].aggro_target < 0 &&
                s->npcs[i].type != INF_NPC_NIBBLER &&
                !(s->npcs[i].type == INF_NPC_BLOB &&
                    s->npcs[i].blob_scanned_prayer >= 0)) {
            s->total_npc_attacks++;
            if (s->npcs[i].type == INF_NPC_RANGER)
                ranger_attacked_this_tick = 1;
            if (s->npcs[i].type == INF_NPC_MAGER)
                mager_attacked_this_tick = 1;
        }
    }
    if (ranger_attacked_this_tick && mager_attacked_this_tick) {
        s->total_ranger_mager_same_tick_attacks++;
        if (s->player_moved_last_tick)
            s->total_step_out_ranger_mager_same_tick_attacks++;
    }
    if (s->tick_scratch.attacks_fired > 0) {
        int styles = s->tick_scratch.styles_fired;
        int n_styles = ((styles >> 0) & 1) + ((styles >> 1) & 1) + ((styles >> 2) & 1);
        if (n_styles >= 2 && s->tick_scratch.prayer_correct > 0) {
            int off_prayer = s->tick_scratch.attacks_fired - s->tick_scratch.prayer_correct;
            s->total_unavoidable_off += off_prayer;
        }
    }
    if (s->ticks_without_action > 0) s->total_idle_ticks++;
    s->total_brews_used += s->tick_scratch.brewed;
    s->total_blood_healed += s->tick_scratch.blood_heal;
    {
        InfNpcPressureSummary pressure = inf_npc_pressure_summary(s);
        s->total_npc_pressure_if_ready_count +=
            (float)pressure.if_ready_total;
        s->total_npc_pressure_this_tick_count +=
            (float)pressure.this_tick_total;
        s->total_npc_pressure_if_ready_max_hit +=
            (float)pressure.if_ready_max_hit;
        s->total_npc_pressure_this_tick_max_hit +=
            (float)pressure.this_tick_max_hit;
        int max_incoming_hit = pressure.if_ready_max_hit >
            pressure.this_tick_max_hit ? pressure.if_ready_max_hit :
            pressure.this_tick_max_hit;
        if ((float)max_incoming_hit > s->max_npc_pressure_incoming_hit)
            s->max_npc_pressure_incoming_hit = (float)max_incoming_hit;
    }

    if (can_player_attack && s->wave == 68) {
        s->total_zuk_ticks++;
        int is_behind_shield = inf_player_behind_zuk_shield_now(s);
        if (is_behind_shield) {
            s->behind_shield_ticks++;
            s->tick_scratch.behind_shield = 1;
        }
        if (!is_behind_shield && s->tick_at_le_240 >= 0)
            s->offshield_ticks_after_240++;
        if (!is_behind_shield && s->tick_at_all_zuk_healers_dead >= 0)
            s->offshield_ticks_after_all_zuk_healers_dead++;
    }

    s->reward = inf_compute_reward_ctx(s, ctx);
    int all_dead = 1;

    if (s->player.current_hitpoints <= 0) {
        if (s->last_hit_by_type >= 0 && s->last_hit_by_type < INF_NUM_NPC_TYPES)
            s->killed_by_type[s->last_hit_by_type]++;
        s->episode_over = 1;
        s->winner = INF_OUTCOME_PLAYER_DIED;
        s->reward = inf_terminal_loss_reward(s, ctx);
        goto finish_step;
    }

    if (spawn_wave_now) {
        s->wave = s->wave_spawn_target;
        inf_spawn_wave(s);
        inf_invalidate_los_cache(s);
        goto finish_step;
    }
    if (s->wave_spawn_delay > 0) goto finish_step;

    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (s->npcs[i].active) { all_dead = 0; break; }
    }
    if (all_dead) {
        s->tick_scratch.wave_completed = 1;
        int next_public_wave = s->wave + 2;
        float wave_clear_reward = 1.0f +
            inf_supply_milestone_surplus_reward(s, ctx, next_public_wave);
        s->reward = wave_clear_reward;
        if (s->wave + 1 >= INF_NUM_WAVES) {
            s->episode_over = 1;
            s->winner = INF_OUTCOME_PLAYER_WON;
        } else {
            s->wave_spawn_target = s->wave + 1;
            s->wave_spawn_delay = 9;
        }
    }

    if (s->tick >= INF_MAX_TICKS) {
        s->episode_over = 1;
        s->winner = INF_OUTCOME_PLAYER_DIED;
        s->reward = inf_terminal_loss_reward(s, ctx);
    }

finish_step:
    inf_record_idle_diagnostics(
        s,
        can_player_attack,
        attack_ready_no_attack,
        target_available_no_attack,
        safe_attack_opportunity_missed);
    s->episode_return += s->reward;
}

static void inf_step(EncounterState* state, const int* actions) {
    inf_step_ctx(state, (EncounterContext*)inf_legacy_context(), actions);
}
