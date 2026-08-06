#line 6591 "encounter_inferno.h"
typedef enum {
    INF_TARGET_CATEGORY_NONE = 0,
    INF_TARGET_CATEGORY_PLAYER,
    INF_TARGET_CATEGORY_ZUK,
    INF_TARGET_CATEGORY_SHIELD,
    INF_TARGET_CATEGORY_OTHER_NPC,
} InfTargetCategory;

static InfTargetCategory inf_npc_target_category(
    const InfernoState* s,
    const InfNPC* npc
) {
    if (!npc || !npc->active) return INF_TARGET_CATEGORY_NONE;
    if (npc->aggro_target < 0) return INF_TARGET_CATEGORY_PLAYER;
    if (npc->aggro_target >= INF_MAX_NPCS) return INF_TARGET_CATEGORY_NONE;

    const InfNPC* target = &s->npcs[npc->aggro_target];
    if (!target->active) return INF_TARGET_CATEGORY_NONE;
    if (target->type == INF_NPC_ZUK) return INF_TARGET_CATEGORY_ZUK;
    if (target->type == INF_NPC_ZUK_SHIELD) return INF_TARGET_CATEGORY_SHIELD;
    return INF_TARGET_CATEGORY_OTHER_NPC;
}

static int inf_wave_phase_index(int wave_idx) {
    int wave = wave_idx + 1;
    if (wave <= 17) return 0;
    if (wave <= 34) return 1;
    if (wave <= 49) return 2;
    if (wave <= 66) return 3;
    if (wave <= 68) return 4;
    return 5;
}

static int inf_npc_is_phantom_barrage_targetable_now(
    InfernoState* s,
    const InfernoContext* ctx,
    int npc_idx
) {
    if (s->player.attack_timer != 0) return 0;
    if (npc_idx < 0 || npc_idx >= INF_MAX_NPCS) return 0;
    return s->npcs[npc_idx].death_ticks == 1 &&
        inf_player_can_phantom_barrage_npc(s, ctx, npc_idx);
}

static int inf_pending_spark_obs_less(
    const InfernoState* s,
    const InfPendingSpark* a,
    int ai,
    const InfPendingSpark* b,
    int bi
) {
    if (a->ticks_remaining != b->ticks_remaining)
        return a->ticks_remaining < b->ticks_remaining;
    int adx = abs(a->x - s->player.x);
    int ady = abs(a->y - s->player.y);
    int bdx = abs(b->x - s->player.x);
    int bdy = abs(b->y - s->player.y);
    int ad = adx > ady ? adx : ady;
    int bd = bdx > bdy ? bdx : bdy;
    if (ad != bd) return ad < bd;
    if (a->x != b->x) return a->x < b->x;
    if (a->y != b->y) return a->y < b->y;
    return ai < bi;
}

static inline float inf_obs_norm_count(int count) {
    return (float)count / (float)INF_OBS_NPCS;
}

static inline float inf_obs_norm_damage_sum(int damage) {
    float value = (float)damage / 300.0f;
    return value > 1.0f ? 1.0f : value;
}

static inline float inf_obs_norm_range_deficit(int deficit) {
    float value = (float)deficit / 15.0f;
    return value > 1.0f ? 1.0f : value;
}

static inline float inf_obs_norm_npc_size(int size) {
    return (float)size / 7.0f;
}

static inline float inf_obs_norm_pillar_size(const InfPillar* pillar) {
    return pillar->active ? (float)INF_PILLAR_SIZE / 7.0f : 0.0f;
}

static const struct { InfNPCType type; int count; }
INF_OBS_SLOT_LAYOUT[INF_NUM_NPC_TYPES] = {
    { INF_NPC_MAGER, 2 },      { INF_NPC_RANGER, 2 },
    { INF_NPC_MELEER, 2 },     { INF_NPC_BLOB, 2 },
    { INF_NPC_BAT, 2 },        { INF_NPC_BLOB_MAGE, 2 },
    { INF_NPC_BLOB_RANGE, 2 }, { INF_NPC_BLOB_MELEE, 2 },
    { INF_NPC_NIBBLER, 6 },    { INF_NPC_JAD, 3 },
    { INF_NPC_ZUK, 1 },        { INF_NPC_ZUK_SHIELD, 1 },
    { INF_NPC_HEALER_JAD, 6 }, { INF_NPC_HEALER_ZUK, 4 },
};

static void inf_refresh_current_obs_slots_ctx(InfernoState* s, const InfernoContext* ctx) {
    int obs_slots[INF_OBS_NPCS];
    for (int j = 0; j < INF_OBS_NPCS; j++) obs_slots[j] = -1;

    int slot_counts[INF_NUM_NPC_TYPES] = {0};
    int slot_offsets[INF_NUM_NPC_TYPES];
    int slot_max[INF_NUM_NPC_TYPES];

    int offset = 0;
    for (int t = 0; t < INF_NUM_NPC_TYPES; t++) {
        slot_offsets[INF_OBS_SLOT_LAYOUT[t].type] = offset;
        slot_max[INF_OBS_SLOT_LAYOUT[t].type] = INF_OBS_SLOT_LAYOUT[t].count;
        offset += INF_OBS_SLOT_LAYOUT[t].count;
    }

    for (int n = 0; n < INF_MAX_NPCS; n++) {
        InfNPC* npc = &s->npcs[n];
        if (npc->active && npc->death_ticks == 0) {
            int t = npc->type;
            if (slot_counts[t] < slot_max[t]) {
                obs_slots[slot_offsets[t] + slot_counts[t]] = n;
                slot_counts[t]++;
            }
        }
    }

    for (int n = 0; n < INF_MAX_NPCS; n++) {
        InfNPC* npc = &s->npcs[n];
        if (npc->active && npc->death_ticks > 0 &&
                inf_npc_is_phantom_barrage_obs_candidate(s, ctx, n)) {
            int t = npc->type;
            if (slot_counts[t] < slot_max[t]) {
                obs_slots[slot_offsets[t] + slot_counts[t]] = n;
                slot_counts[t]++;
            }
        }
    }

    for (int j = 0; j < INF_OBS_NPCS; j++) {
        s->current_obs_slots[j] = obs_slots[j];
    }
}

static void inf_refresh_current_obs_slots(InfernoState* s) {
    inf_refresh_current_obs_slots_ctx(s, inf_legacy_context());
}

static void inf_write_obs_ctx(
    EncounterState* state,
    EncounterContext* context,
    float* obs
) {
    InfernoState* s = (InfernoState*)state;
    InfernoContext* ctx = context ? (InfernoContext*)context : inf_legacy_context();
#ifdef INF_PROFILE_ENABLED
    int inf_prof_enabled = INF_PROFILE_ENABLED();
    double inf_prof_t0 = inf_prof_enabled ? INF_PROFILE_NOW_MS() : 0.0;
#endif
    memset(obs, 0, INF_NUM_OBS * sizeof(float));
    int i = 0;
    int px = s->player.x, py = s->player.y;
    const EncounterLoadoutStats* ls = inf_current_loadout_stats(s, ctx);
    InfSupplyDoses full_supplies = inf_full_starting_supplies();

    obs[i++] = (float)s->player.current_hitpoints / 99.0f;
    obs[i++] = (float)(px - INF_ARENA_MIN_X) / (float)INF_ARENA_WIDTH;
    obs[i++] = (float)(INF_ARENA_MAX_X - px) / (float)INF_ARENA_WIDTH;
    obs[i++] = (float)(py - INF_ARENA_MIN_Y) / (float)INF_ARENA_HEIGHT;
    obs[i++] = (float)(INF_ARENA_MAX_Y - py) / (float)INF_ARENA_HEIGHT;
    obs[i++] = (s->player.prayer == PRAYER_PROTECT_MELEE) ? 1.0f : 0.0f;
    obs[i++] = (s->player.prayer == PRAYER_PROTECT_RANGED) ? 1.0f : 0.0f;
    obs[i++] = (s->player.prayer == PRAYER_PROTECT_MAGIC) ? 1.0f : 0.0f;
    obs[i++] = (s->player.offensive_prayer == OFFENSIVE_PRAYER_PIETY) ? 1.0f : 0.0f;
    obs[i++] = (s->player.offensive_prayer == OFFENSIVE_PRAYER_RIGOUR) ? 1.0f : 0.0f;
    obs[i++] = (s->player.offensive_prayer == OFFENSIVE_PRAYER_AUGURY) ? 1.0f : 0.0f;
    obs[i++] = (float)s->player.brew_doses / (float)full_supplies.brew_doses;
    obs[i++] = (float)s->player.restore_doses / (float)full_supplies.restore_doses;
    obs[i++] = (float)s->player.current_prayer / 99.0f;
    obs[i++] = (float)s->wave / (float)INF_NUM_WAVES;
    {
        int phase = inf_wave_phase_index(s->wave);
        for (int p = 0; p < 6; p++)
            obs[i++] = (p == phase) ? 1.0f : 0.0f;
    }
    obs[i++] = inf_zuk_attack_timer_obs(s);
    obs[i++] = (s->weapon_set == INF_GEAR_MAGE) ? 1.0f : 0.0f;
    obs[i++] = (s->weapon_set == INF_GEAR_LONG_RANGE) ? 1.0f : 0.0f;
    obs[i++] = (s->weapon_set == INF_GEAR_BP) ? 1.0f : 0.0f;
    obs[i++] = (float)s->player.bastion_doses / (float)full_supplies.bastion_doses;
    obs[i++] = (float)s->player.stamina_doses / (float)full_supplies.stamina_doses;
    obs[i++] = (s->stamina_active_ticks > 0) ? 1.0f : 0.0f;
    obs[i++] = (float)s->player.potion_timer / 3.0f;
    obs[i++] = (float)s->player.attack_timer / 8.0f;
    obs[i++] = (float)s->player.current_defence / 99.0f;
    obs[i++] = (float)s->player.current_ranged / 99.0f;
    obs[i++] = (float)s->player.current_magic / 99.0f;
    obs[i++] = osrs_interaction_active(&s->interaction) ? 1.0f : 0.0f;
    obs[i++] = (float)ls->attack_range / 15.0f;
    obs[i++] = (float)s->dead_mob_count / (float)INF_MAX_DEAD_MOBS;
    obs[i++] = (float)ls->max_hit / 80.0f;
    obs[i++] = (float)ls->attack_speed / 6.0f;
    obs[i++] = (float)ls->def_stab / 300.0f;
    obs[i++] = (float)ls->def_magic / 300.0f;
    obs[i++] = (float)ls->def_ranged / 300.0f;
    obs[i++] = (float)s->player.special_energy / 100.0f;

    {
        int min_timer = 999;
        int min_style = 0;
        int has_melee_2 = 0, has_ranged_2 = 0, has_magic_2 = 0;

        for (int h = 0; h < s->player_pending_hits.count; h++) {
            EncounterPendingHit* ph = &s->player_pending_hits.hits[h];
            if (ph->check_prayer) {
                int t = inf_pending_hit_obs_timer(ph);
                if (t < min_timer) {
                    min_timer = t;
                    min_style = ph->attack_style;
                }
                if (t <= 2) {
                    if (ph->attack_style == ATTACK_STYLE_MELEE) has_melee_2 = 1;
                    if (ph->attack_style == ATTACK_STYLE_RANGED) has_ranged_2 = 1;
                    if (ph->attack_style == ATTACK_STYLE_MAGIC) has_magic_2 = 1;
                }
            }
        }

        for (int n = 0; n < INF_MAX_NPCS; n++) {
            InfNPC* npc = &s->npcs[n];
            if (!npc->active || npc->death_ticks > 0) continue;
            InfNpcPlayerThreat threat = inf_npc_player_threat(s, npc);
            int style_mask = threat.telegraph_style_mask;
            if (!style_mask) continue;
            if (npc->stun_timer > 0 ||
                    npc->dig_freeze_timer > 0 ||
                    npc->dig_attack_delay > 0)
                continue;
            int preview_style = inf_attack_style_obs_preview(style_mask);

            int t = npc->attack_timer;
            if (t == 0) t = 1;

            if (t < min_timer) {
                min_timer = t;
                min_style = preview_style;
            }
            if (t <= 2) {
                if (style_mask & INF_STYLE_MASK_MELEE) has_melee_2 = 1;
                if (style_mask & INF_STYLE_MASK_RANGED) has_ranged_2 = 1;
                if (style_mask & INF_STYLE_MASK_MAGIC) has_magic_2 = 1;
            }
        }

        int conflict_count = has_melee_2 + has_ranged_2 + has_magic_2;
        obs[i++] = (min_timer < 999) ? (float)min_timer / 10.0f : 1.0f;
        obs[i++] = (min_style == ATTACK_STYLE_MELEE) ? 1.0f : 0.0f;
        obs[i++] = (min_style == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
        obs[i++] = (min_style == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
        obs[i++] = (float)conflict_count / 3.0f;
    }

    {
        InfNpcPressureSummary pressure = inf_npc_pressure_summary(s);
        obs[i++] = inf_obs_norm_count(pressure.if_ready_total);
        obs[i++] = inf_obs_norm_count(pressure.if_ready_melee);
        obs[i++] = inf_obs_norm_count(pressure.if_ready_ranged);
        obs[i++] = inf_obs_norm_count(pressure.if_ready_magic);
        obs[i++] = (float)pressure.if_ready_max_hit / 150.0f;
        obs[i++] = inf_obs_norm_damage_sum(pressure.if_ready_sum_max_hit);
        obs[i++] = inf_obs_norm_count(pressure.this_tick_total);
        obs[i++] = inf_obs_norm_count(pressure.this_tick_melee);
        obs[i++] = inf_obs_norm_count(pressure.this_tick_ranged);
        obs[i++] = inf_obs_norm_count(pressure.this_tick_magic);
        obs[i++] = (float)pressure.this_tick_max_hit / 150.0f;
        obs[i++] = inf_obs_norm_damage_sum(pressure.this_tick_sum_max_hit);
        obs[i++] = inf_obs_norm_count(pressure.timer_lte_1);
        obs[i++] = inf_obs_norm_count(pressure.timer_lte_2);
        obs[i++] = inf_obs_norm_count(pressure.timer_lte_4);
        obs[i++] = (float)pressure.if_ready_style_count / 3.0f;
        obs[i++] = (float)pressure.this_tick_style_count / 3.0f;
        obs[i++] = (pressure.nearest_timer < 999)
            ? (float)pressure.nearest_timer / 10.0f
            : 1.0f;
    }

    {
        int is_zuk = (s->wave == 68);
        obs[i++] = is_zuk ? 1.0f : 0.0f;

        if (is_zuk) {
            int si = s->zuk.shield_idx;
            int shield_active = (si >= 0 && s->npcs[si].active);

            obs[i++] = shield_active
                ? ((s->zuk.shield_freeze > 0) ? 0.0f : (float)s->zuk.shield_dir)
                : 0.0f;
            obs[i++] = shield_active ? (float)s->zuk.shield_freeze / 5.0f : 0.0f;
            int behind = 0;
            float shield_offset = 0.0f;
            if (shield_active) {
                int sx = s->npcs[si].x;
                int sz = INF_NPC_STATS[INF_NPC_ZUK_SHIELD].size;
                int shield_center = sx + sz / 2;
                shield_offset = (float)(px - shield_center) / 15.0f;
                behind = (px >= sx && px < sx + sz && py >= 41);
            }
            obs[i++] = behind ? 1.0f : 0.0f;
            obs[i++] = shield_offset;
            obs[i++] = s->zuk.enraged ? 1.0f : 0.0f;
            obs[i++] = (float)s->zuk.set_timer / 350.0f;
            obs[i++] = s->zuk.timer_paused ? 1.0f : 0.0f;
            obs[i++] = s->zuk.jad_spawned ? 1.0f : 0.0f;
            obs[i++] = s->zuk.healer_spawned ? 1.0f : 0.0f;
        } else {
            for (int z = 0; z < 9; z++) obs[i++] = 0.0f;
        }
    }

    for (int p = 0; p < INF_NUM_PILLARS; p++) {
        obs[i++] = s->pillars[p].active ? 1.0f : 0.0f;
        obs[i++] = (float)s->pillars[p].hp / (float)INF_PILLAR_HP;
        obs[i++] = (float)(s->pillars[p].x - px) / (float)INF_ARENA_WIDTH;
        obs[i++] = (float)(s->pillars[p].y - py) / (float)INF_ARENA_HEIGHT;
        obs[i++] = inf_obs_norm_pillar_size(&s->pillars[p]);
    }

#ifdef INF_PROFILE_ENABLED
    INF_PROFILE_MARK(INF_PROF_OBS_PREFIX);
#endif
    inf_refresh_current_obs_slots_ctx(s, ctx);
#ifdef INF_PROFILE_ENABLED
    INF_PROFILE_MARK(INF_PROF_OBS_REFRESH_SLOTS);
#endif

    int slot_types[INF_OBS_NPCS];
    int st_idx = 0;
    for (int t = 0; t < INF_NUM_NPC_TYPES; t++) {
        for (int j = 0; j < INF_OBS_SLOT_LAYOUT[t].count; j++)
            slot_types[st_idx++] = INF_OBS_SLOT_LAYOUT[t].type;
    }

    for (int k = 0; k < INF_OBS_NPCS; k++) {
        int n = s->current_obs_slots[k];
        int type = slot_types[k];

        int has_style = (type == INF_NPC_BLOB || type == INF_NPC_JAD);
        int has_scan = (type == INF_NPC_BLOB);
        int has_los = (type != INF_NPC_NIBBLER && type != INF_NPC_MELEER && type != INF_NPC_HEALER_JAD && type != INF_NPC_ZUK_SHIELD);
        int has_target_category = (type != INF_NPC_NIBBLER && type != INF_NPC_ZUK_SHIELD);
        int has_timer = (type != INF_NPC_NIBBLER && type != INF_NPC_HEALER_JAD && type != INF_NPC_ZUK_SHIELD);
        int has_meleer_dig = (type == INF_NPC_MELEER);

        if (n >= 0) {
            InfNPC* npc = &s->npcs[n];
            for (int ty = 0; ty < INF_NPC_TYPE_ONEHOT; ty++)
                obs[i++] = (ty == type) ? 1.0f : 0.0f;
            obs[i++] = (float)npc->hp / (float)npc->max_hp;
            obs[i++] = (float)(npc->x - px) / (float)INF_ARENA_WIDTH;
            obs[i++] = (float)(npc->y - py) / (float)INF_ARENA_HEIGHT;
            obs[i++] = inf_obs_norm_npc_size(npc->size);
            obs[i++] = has_timer ? (float)npc->attack_timer / 10.0f : 0.0f;

            if (has_style) {
                int style = (npc->type == INF_NPC_JAD) ? inf_npc_jad(npc)->attack_style : npc->attack_style;
                obs[i++] = (style == ATTACK_STYLE_MELEE) ? 1.0f : 0.0f;
                obs[i++] = (style == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
                obs[i++] = (style == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
            } else {
                obs[i++] = 0.0f; obs[i++] = 0.0f; obs[i++] = 0.0f;
            }

            obs[i++] = (has_los && inf_npc_has_los(s, n)) ? 1.0f : 0.0f;

            if (has_scan && npc->blob_scanned_prayer >= 0) {
                OverheadPrayer scanned = (OverheadPrayer)npc->blob_scanned_prayer;
                obs[i++] = (scanned == PRAYER_PROTECT_MAGIC) ? 1.0f : 0.0f;
                obs[i++] = (scanned == PRAYER_PROTECT_RANGED) ? 1.0f : 0.0f;
                obs[i++] = (scanned != PRAYER_PROTECT_MAGIC && scanned != PRAYER_PROTECT_RANGED) ? 1.0f : 0.0f;
            } else {
                obs[i++] = 0.0f; obs[i++] = 0.0f; obs[i++] = 0.0f;
            }

            obs[i++] = (float)inf_barrage_total_target_count(s, n) /
                (float)BARRAGE_MAX_HITS;

            {
                InfNpcPlayerThreat threat = inf_npc_player_threat(s, npc);
                obs[i++] = (float)threat.edge_distance / (float)INF_ARENA_HEIGHT;
                obs[i++] = threat.can_attack_if_ready ? 1.0f : 0.0f;
                obs[i++] = threat.can_attack_this_tick ? 1.0f : 0.0f;
                obs[i++] = (float)npc->frozen_ticks / (float)BARRAGE_FREEZE_TICKS;
                obs[i++] = inf_player_can_attack_npc_from_current_tile(s, n) ? 1.0f : 0.0f;
                obs[i++] = inf_player_has_los_to_npc_from_current_tile(s, n) ? 1.0f : 0.0f;
                InfPlayerNpcLoadoutReach mage =
                    inf_player_loadout_reach_to_npc(s, n, INF_GEAR_MAGE);
                InfPlayerNpcLoadoutReach long_range =
                    inf_player_loadout_reach_to_npc(s, n, INF_GEAR_LONG_RANGE);
                InfPlayerNpcLoadoutReach blowpipe =
                    inf_player_loadout_reach_to_npc(s, n, INF_GEAR_BP);
                obs[i++] = mage.can_attack_now ? 1.0f : 0.0f;
                obs[i++] = long_range.can_attack_now ? 1.0f : 0.0f;
                obs[i++] = blowpipe.can_attack_now ? 1.0f : 0.0f;
                obs[i++] = inf_obs_norm_range_deficit(mage.range_deficit);
                obs[i++] = inf_obs_norm_range_deficit(long_range.range_deficit);
                obs[i++] = inf_obs_norm_range_deficit(blowpipe.range_deficit);
            }

            if (has_target_category) {
                InfTargetCategory category = inf_npc_target_category(s, npc);
                obs[i++] = (category == INF_TARGET_CATEGORY_PLAYER) ? 1.0f : 0.0f;
                obs[i++] = (category == INF_TARGET_CATEGORY_ZUK) ? 1.0f : 0.0f;
                obs[i++] = (category == INF_TARGET_CATEGORY_SHIELD) ? 1.0f : 0.0f;
                obs[i++] = (category == INF_TARGET_CATEGORY_OTHER_NPC) ? 1.0f : 0.0f;
            } else {
                obs[i++] = 0.0f; obs[i++] = 0.0f; obs[i++] = 0.0f; obs[i++] = 0.0f;
            }
            obs[i++] = (osrs_interaction_active(&s->interaction) && s->interaction.target_slot == n) ? 1.0f : 0.0f;
            obs[i++] = inf_npc_is_phantom_barrage_targetable_now(s, ctx, n) ? 1.0f : 0.0f;
            if (has_meleer_dig) {
                float no_los = (float)npc->no_los_ticks / 50.0f;
                if (no_los > 1.0f) no_los = 1.0f;
                obs[i++] = no_los;
                obs[i++] = (float)npc->dig_freeze_timer / 6.0f;
                obs[i++] = (float)npc->dig_attack_delay / 6.0f;
            } else {
                obs[i++] = 0.0f; obs[i++] = 0.0f; obs[i++] = 0.0f;
            }
        } else {
            for (int j = 0; j < INF_NPC_SLOT_FEATURES; j++) obs[i++] = 0.0f;
        }
    }
#ifdef INF_PROFILE_ENABLED
    INF_PROFILE_MARK(INF_PROF_OBS_NPC_SLOTS);
#endif

    {
        if (i != INF_OBS_AFTER_NPCS) {
            fprintf(stderr, "FATAL: obs misaligned after NPC section: i=%d expected=%d\n",
                    i, INF_OBS_AFTER_NPCS);
            abort();
        }
    }

    {
        if (ctx->config.step_out_forecast_obs_mode != INF_STEP_OUT_FORECAST_MODE_OFF) {
            InfStepOutForecast forecast;
            inf_build_step_out_forecast_mode_ctx(s, ctx, &forecast);
            for (int action_idx = 0; action_idx < ENCOUNTER_MOVE_ACTIONS; action_idx++) {
                const InfStepOutForecastAction* action = &forecast.actions[action_idx];
                int first_attack_tick = 0;
                int first_style_mask = 0;
                int max_hit = 0;
                int ranger_mager_same_tick = 0;
                for (int tick_idx = 0; tick_idx < INF_STEP_OUT_FORECAST_HORIZON; tick_idx++) {
                    const InfStepOutForecastTick* tick = &action->ticks[tick_idx];
                    int style_mask = inf_step_out_forecast_tick_style_mask(tick);
                    if (first_attack_tick == 0 &&
                            inf_step_out_forecast_tick_has_event(tick)) {
                        first_attack_tick = tick_idx + 1;
                        first_style_mask = style_mask;
                    }
                    if (tick->max_hit > max_hit) max_hit = tick->max_hit;
                    if (tick->ranger_count > 0 && tick->mager_count > 0)
                        ranger_mager_same_tick = 1;
                }
                obs[i++] = action->valid ? 1.0f : 0.0f;
                obs[i++] = (float)first_attack_tick / (float)INF_STEP_OUT_FORECAST_HORIZON;
                obs[i++] = (float)first_style_mask / 7.0f;
                obs[i++] = (float)max_hit / 150.0f;
                obs[i++] = action->same_tick_mixed_style_conflict ? 1.0f : 0.0f;
                obs[i++] = ranger_mager_same_tick ? 1.0f : 0.0f;
                obs[i++] = action->ranger_mager_offtick_opportunity ? 1.0f : 0.0f;
                obs[i++] = action->melee_fallback_exposure ? 1.0f : 0.0f;
            }
        } else {
            i += INF_STEP_OUT_FORECAST_OBS_SIZE;
        }
    }
#ifdef INF_PROFILE_ENABLED
    INF_PROFILE_MARK(INF_PROF_OBS_FORECAST);
#endif

    {
        if (i != INF_OBS_AFTER_FORECAST) {
            fprintf(stderr, "FATAL: obs misaligned after step-out forecast: i=%d expected=%d\n",
                    i, INF_OBS_AFTER_FORECAST);
            abort();
        }
    }

    for (int h = 0; h < ENCOUNTER_MAX_PENDING_HITS; h++) {
        if (h < s->player_pending_hits.count) {
            EncounterPendingHit* ph = &s->player_pending_hits.hits[h];
            obs[i++] = 1.0f;
            obs[i++] = (ph->attack_style == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
            obs[i++] = (ph->attack_style == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
            obs[i++] = (float)inf_pending_hit_obs_timer(ph) / 10.0f;
            obs[i++] = (float)ph->damage / 150.0f;
        } else {
            for (int j = 0; j < INF_FEATURES_PER_HIT; j++) obs[i++] = 0.0f;
        }
    }
#ifdef INF_PROFILE_ENABLED
    INF_PROFILE_MARK(INF_PROF_OBS_PENDING_HITS);
#endif

    if (i != INF_OBS_AFTER_PENDING_HITS) {
        fprintf(stderr, "FATAL: obs misaligned after pending hits: i=%d expected=%d\n",
                i, INF_OBS_AFTER_PENDING_HITS);
        abort();
    }

    int used_sparks[INF_MAX_PENDING_SPARKS] = {0};
    for (int slot = 0; slot < INF_SPARK_OBS_SLOTS; slot++) {
        int best = -1;
        for (int sp = 0; sp < INF_MAX_PENDING_SPARKS; sp++) {
            if (used_sparks[sp] || !s->pending_sparks[sp].active) continue;
            if (best < 0 ||
                inf_pending_spark_obs_less(
                    s, &s->pending_sparks[sp], sp,
                    &s->pending_sparks[best], best)) {
                best = sp;
            }
        }

        if (best >= 0) {
            const InfPendingSpark* spark = &s->pending_sparks[best];
            used_sparks[best] = 1;
            obs[i++] = 1.0f;
            obs[i++] = (float)(spark->x - px) / (float)INF_ARENA_WIDTH;
            obs[i++] = (float)(spark->y - py) / (float)INF_ARENA_HEIGHT;
            obs[i++] = (float)(spark->src_x - px) / (float)INF_ARENA_WIDTH;
            obs[i++] = (float)(spark->src_y - py) / (float)INF_ARENA_HEIGHT;
            obs[i++] = (float)spark->ticks_remaining / 10.0f;
            obs[i++] = (float)spark->damage / 10.0f;
        } else {
            for (int j = 0; j < INF_FEATURES_PER_SPARK; j++) obs[i++] = 0.0f;
        }
    }
#ifdef INF_PROFILE_ENABLED
    INF_PROFILE_MARK(INF_PROF_OBS_SPARKS);
#endif

    if (i != INF_OBS_AFTER_SPARKS) {
        fprintf(stderr, "FATAL: obs misaligned before inventory block: i=%d expected=%d\n",
                i, INF_OBS_AFTER_SPARKS);
        abort();
    }

    {
        static const float inf_zero_post_use_deltas[6] = {0};
        for (int c = 0; c < OSRS_INVENTORY_SIZE; c++) {
            const OsrsInventoryCell* cell = &s->inventory_cells[c];
            int is_equipped = osrs_inventory_cell_holds_equipped_item(
                &s->player, s->inventory_cells, c);
            osrs_write_inventory_cell_affordance_features(
                &obs[i], cell->item_idx, cell->raw_osrs_id, cell->dose,
                is_equipped, inf_zero_post_use_deltas,
                s->player.base_hitpoints, s->player.base_prayer,
                s->player.base_ranged);
            i += OSRS_INVENTORY_CELL_OBS_FEATURES;
        }
        for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
            osrs_write_equipped_self_features(&obs[i], s->player.equipped[slot]);
            i += OSRS_EQUIPPED_SELF_OBS_FEATURES;
        }
    }

    if (i != INF_NUM_OBS) {
        fprintf(stderr, "BUG: inf_write_obs wrote %d features, expected %d\n", i, INF_NUM_OBS);
        abort();
    }
}

static int inf_find_target_obs_slot(const InfernoState* s, int npc_slot) {
    if (npc_slot < 0 || npc_slot >= INF_MAX_NPCS) return -1;
    return encounter_find_observed_target_slot(
        s->current_obs_slots, INF_OBS_NPCS, npc_slot);
}

static void inf_write_obs(EncounterState* state, float* obs) {
    inf_write_obs_ctx(state, (EncounterContext*)inf_legacy_context(), obs);
}

static int inf_obs_slot_is_targetable(
    InfernoState* s,
    const InfernoContext* ctx,
    int obs_slot
) {
    if (obs_slot < 0 || obs_slot >= INF_OBS_NPCS) return 0;
    int npc_slot = s->current_obs_slots[obs_slot];
    if (npc_slot < 0 || npc_slot >= INF_MAX_NPCS) return 0;
    if (s->npcs[npc_slot].active &&
            s->npcs[npc_slot].death_ticks == 0 &&
            s->npcs[npc_slot].type != INF_NPC_ZUK_SHIELD)
        return 1;
    return s->player.attack_timer <= 1 &&
        inf_npc_is_phantom_barrage_obs_candidate(s, ctx, npc_slot);
}

static int inf_is_human_targetable_npc_slot_ctx(
    EncounterState* state,
    EncounterContext* context,
    int npc_slot
) {
    InfernoState* s = (InfernoState*)state;
    InfernoContext* ctx = context ? (InfernoContext*)context : inf_legacy_context();
    inf_refresh_current_obs_slots_ctx(s, ctx);
    return inf_obs_slot_is_targetable(s, ctx, inf_find_target_obs_slot(s, npc_slot));
}

static int inf_is_human_targetable_npc_slot(EncounterState* state, int npc_slot) {
    return inf_is_human_targetable_npc_slot_ctx(
        state, (EncounterContext*)inf_legacy_context(), npc_slot);
}

static int inf_drink_has_effect(const InfernoState* s, OsrsConsumableKind kind) {
    const Player* p = &s->player;
    switch (kind) {
        case OSRS_CONSUMABLE_BREW:
            return p->current_hitpoints < p->base_hitpoints;
        case OSRS_CONSUMABLE_SUPER_RESTORE: {
            int pray_missing = p->base_prayer - p->current_prayer;
            int stats_drained = p->current_attack < 99 || p->current_strength < 99 ||
                                p->current_defence < 99 || p->current_ranged < 99 ||
                                p->current_magic < 99;
            return stats_drained || pray_missing >= (INF_RESTORE_AMOUNT + 1) / 2;
        }
        case OSRS_CONSUMABLE_BASTION:
            return p->current_ranged >= 99 && p->current_ranged <= 105;
        case OSRS_CONSUMABLE_STAMINA:
            return s->stamina_active_ticks == 0;
        default:
            return 0;
    }
}

static void inf_write_mask_ctx(
    EncounterState* state,
    EncounterContext* context,
    float* mask
) {
    InfernoState* s = (InfernoState*)state;
    InfernoContext* ctx = context ? (InfernoContext*)context : inf_legacy_context();
    inf_refresh_current_obs_slots_ctx(s, ctx);
    int offset = 0;
    if (s->total_action_mask_checks == 0) {
        for (int h = 0; h < INF_NUM_ACTION_HEADS; h++) {
            if (s->min_valid_action_count_by_head[h] == 0)
                s->min_valid_action_count_by_head[h] = 1000000;
        }
    }

    mask[offset++] = 1.0f;
    for (int d = 1; d < ENCOUNTER_MOVE_ACTIONS; d++) {
        int nx = s->player.x + ENCOUNTER_MOVE_TARGET_DX[d];
        int ny = s->player.y + ENCOUNTER_MOVE_TARGET_DY[d];
        mask[offset++] = (inf_in_arena(nx, ny) && !inf_blocked_by_pillar(s, nx, ny, 1))
                         ? 1.0f : 0.0f;
    }

    mask[offset++] = 1.0f;
    mask[offset++] = s->player.prayer != PRAYER_NONE ? 1.0f : 0.0f;
    mask[offset++] = s->player.current_prayer > 0 ? 1.0f : 0.0f;
    mask[offset++] = s->player.current_prayer > 0 ? 1.0f : 0.0f;
    mask[offset++] = s->player.current_prayer > 0 ? 1.0f : 0.0f;
    mask[offset++] = s->player.current_prayer > 0 ? 1.0f : 0.0f;

    int has_forced_safe_healer_target =
        ctx->config.zuk_force_safe_untagged_healer_target_mask &&
        inf_has_safe_untagged_zuk_healer_target_now(s, ctx);

    mask[offset++] = has_forced_safe_healer_target ? 0.0f : 1.0f;
    for (int n = 0; n < INF_OBS_NPCS; n++) {
        int npc_idx = s->current_obs_slots[n];
        int is_targetable = inf_obs_slot_is_targetable(s, ctx, n);
        if (is_targetable &&
                (ctx->config.zuk_safe_untagged_healer_target_mask ||
                    ctx->config.zuk_force_safe_untagged_healer_target_mask) &&
                !inf_untagged_zuk_healer_target_is_safe_now(s, ctx, npc_idx)) {
            is_targetable = 0;
        }
        if (is_targetable && has_forced_safe_healer_target &&
                !inf_is_safe_untagged_zuk_healer_target_now(s, ctx, npc_idx)) {
            is_targetable = 0;
        }
        mask[offset++] = is_targetable ? 1.0f : 0.0f;
    }

    {
        int cell_equip_slot[OSRS_INVENTORY_SIZE];
        int cell_can_eat[OSRS_INVENTORY_SIZE];
        int cell_can_drink[OSRS_INVENTORY_SIZE];
        for (int c = 0; c < OSRS_INVENTORY_SIZE; c++) {
            cell_equip_slot[c] = -1;
            cell_can_eat[c] = 0;
            cell_can_drink[c] = 0;
            const OsrsInventoryCell* cell = &s->inventory_cells[c];
            OsrsInventoryClickResolution resolution =
                osrs_inventory_cell_click_interpret(cell, OSRS_CLICK_TICK_FIRST);
            switch (resolution.click_action) {
                case OSRS_CLICK_EQUIP:
                    if (osrs_can_equip_from_cell(&s->player, s->inventory_cells, c))
                        cell_equip_slot[c] = osrs_item_gear_slot(cell->item_idx);
                    break;
                case OSRS_CLICK_EAT:
                    cell_can_eat[c] = osrs_can_eat_consumable_kind(
                        &s->player, resolution.consumable_kind);
                    break;
                case OSRS_CLICK_DRINK:
                    cell_can_drink[c] = cell->dose > 0 &&
                        s->player.potion_timer == 0 &&
                        inf_drink_has_effect(s, resolution.consumable_kind);
                    break;
                case OSRS_CLICK_NONE:
                    break;
            }
        }
        for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
            mask[offset++] = 1.0f;
            for (int c = 0; c < OSRS_INVENTORY_SIZE; c++)
                mask[offset++] = cell_equip_slot[c] == slot ? 1.0f : 0.0f;
        }
        mask[offset++] = 1.0f;
        for (int c = 0; c < OSRS_INVENTORY_SIZE; c++)
            mask[offset++] = cell_can_eat[c] ? 1.0f : 0.0f;
        mask[offset++] = 1.0f;
        for (int c = 0; c < OSRS_INVENTORY_SIZE; c++)
            mask[offset++] = cell_can_drink[c] ? 1.0f : 0.0f;
    }

    mask[offset++] = 1.0f;
    mask[offset++] = s->player.current_magic >= BLOOD_BARRAGE_LEVEL ? 1.0f : 0.0f;
    mask[offset++] = s->player.current_magic >= ICE_BARRAGE_LEVEL ? 1.0f : 0.0f;

    mask[offset++] = 1.0f;
    mask[offset++] = (!s->player.spec_armed &&
                      s->player.equipped[GEAR_SLOT_WEAPON] == ITEM_TOXIC_BLOWPIPE &&
                      s->player.special_energy >= BLOWPIPE_SPEC_COST)
                     ? 1.0f : 0.0f;
    mask[offset++] = s->player.spec_armed ? 1.0f : 0.0f;

    mask[offset++] = 1.0f;
    mask[offset++] = s->player.offensive_prayer != OFFENSIVE_PRAYER_NONE ? 1.0f : 0.0f;
    mask[offset++] = s->player.current_prayer > 0 ? 1.0f : 0.0f;
    mask[offset++] = s->player.current_prayer > 0 ? 1.0f : 0.0f;
    mask[offset++] = s->player.current_prayer > 0 ? 1.0f : 0.0f;

    int mask_offset = 0;
    for (int h = 0; h < INF_NUM_ACTION_HEADS; h++) {
        int valid = 0;
        for (int a = 0; a < INF_ACTION_DIMS[h]; a++) {
            if (mask[mask_offset + a] > 0.0f)
                valid++;
        }
        if (valid == 0)
            s->zero_valid_action_head_count[h]++;
        if (valid < s->min_valid_action_count_by_head[h])
            s->min_valid_action_count_by_head[h] = valid;
        mask_offset += INF_ACTION_DIMS[h];
    }
    int target_offset = inf_action_head_mask_offset(INF_HEAD_TARGET) + 1;
    for (int n = 0; n < INF_OBS_NPCS; n++) {
        if (mask[target_offset + n] <= 0.0f)
            continue;
        int npc_idx = s->current_obs_slots[n];
        if (npc_idx < 0 || npc_idx >= INF_MAX_NPCS)
            continue;
        int type = s->npcs[npc_idx].type;
        if (type == INF_NPC_HEALER_ZUK)
            s->target_head_valid_healer_count++;
        else if (type == INF_NPC_ZUK)
            s->target_head_valid_zuk_count++;
        else if (type != INF_NPC_ZUK_SHIELD)
            s->target_head_valid_set_count++;
    }
    s->total_action_mask_checks++;
}

static void inf_write_mask(EncounterState* state, float* mask) {
    inf_write_mask_ctx(state, (EncounterContext*)inf_legacy_context(), mask);
}

static float inf_get_reward_ctx(EncounterState* state, EncounterContext* context) {
    (void)context;
    return ((InfernoState*)state)->reward;
}

static int inf_is_terminal_ctx(EncounterState* state, EncounterContext* context) {
    (void)context;
    return ((InfernoState*)state)->episode_over;
}

static int inf_get_entity_count_ctx(EncounterState* state, EncounterContext* context) {
    (void)context;
    InfernoState* s = (InfernoState*)state;
    int count = 1;
    for (int i = 0; i < INF_MAX_NPCS; i++)
        if (s->npcs[i].active) count++;
    return count;
}

static void* inf_get_entity_ctx(EncounterState* state, EncounterContext* context, int index) {
    (void)context;
    InfernoState* s = (InfernoState*)state;
    if (index == 0) return &s->player;
    return NULL;
}

static void inf_fill_render_entities_ctx(
    EncounterState* state,
    EncounterContext* context,
    RenderEntity* out,
    int max_entities,
    int* count
) {
    InfernoState* s = (InfernoState*)state;
    InfernoContext* ctx = context ? (InfernoContext*)context : inf_legacy_context();
    int n = 0;

    {
        const EncounterLoadoutStats* ls = inf_current_loadout_stats(s, ctx);
        s->player.gui_max_hit = ls->max_hit;
        s->player.gui_attack_speed = ls->attack_speed;
        s->player.gui_attack_range = ls->attack_range;
        s->player.gui_strength_bonus = ls->strength_bonus;
    }

    if (n < max_entities) {
        osrs_render_entity_from_player_entity(&s->player, &out[n++]);
    }

    for (int i = 0; i < INF_MAX_NPCS && n < max_entities; i++) {
        InfNPC* npc = &s->npcs[i];
        if (!npc->active) continue;

        int attack_target_entity_idx = 0;
        if (npc->type == INF_NPC_NIBBLER || npc->resurrecting_this_tick) {
            attack_target_entity_idx = -1;
        } else if (npc->aggro_target >= 0 && npc->aggro_target < INF_MAX_NPCS &&
                   s->npcs[npc->aggro_target].active) {
            attack_target_entity_idx = -1;
        }

        int npc_anim_id = -1;
        const NpcModelMapping* nm = npc_model_lookup(INF_NPC_DEF_IDS[npc->type]);
        if (npc->death_ticks > 0) {
            npc_anim_id = inf_npc_death_anim_id(npc, nm);
        } else if (npc->type == INF_NPC_MELEER &&
                   npc->dig_freeze_timer == 6) {
            npc_anim_id = INF_GEN_ANIM_MELEER_DIG_DOWN;
        } else if (npc->type == INF_NPC_MELEER &&
                   npc->dig_attack_delay == 6) {
            npc_anim_id = INF_GEN_ANIM_MELEER_DIG_UP;
        } else if (npc->attacked_this_tick) {
            npc_anim_id = inf_npc_attack_anim_id(npc, nm);
        }

        RenderMovementKind render_movement_kind =
            (npc->type == INF_NPC_MELEER && npc->dig_attack_delay == 6)
                ? RENDER_MOVEMENT_TELEPORT
                : RENDER_MOVEMENT_NORMAL;

        int dest_x = npc->target_x;
        int dest_y = npc->target_y;
        if (npc->type == INF_NPC_NIBBLER) {
            int tp = s->nibbler_target_pillar;
            if (tp >= 0 && tp < INF_NUM_PILLARS && s->pillars[tp].active) {
                dest_x = s->pillars[tp].x + INF_PILLAR_SIZE / 2;
                dest_y = s->pillars[tp].y + INF_PILLAR_SIZE / 2;
            } else {
                dest_x = npc->x;
                dest_y = npc->y;
            }
        } else if (npc->resurrection_visual_target >= 0 &&
                   npc->resurrection_visual_target < INF_MAX_NPCS &&
                   s->npcs[npc->resurrection_visual_target].active) {
            InfNPC* at = &s->npcs[npc->resurrection_visual_target];
            dest_x = at->x + at->size / 2;
            dest_y = at->y + at->size / 2;
        } else if (npc->aggro_target >= 0 && npc->aggro_target < INF_MAX_NPCS &&
                   s->npcs[npc->aggro_target].active) {
            InfNPC* at = &s->npcs[npc->aggro_target];
            dest_x = at->x + at->size / 2;
            dest_y = at->y + at->size / 2;
        }

        RenderEntity* re = &out[n++];
        OsrsNpcRenderEntitySpec npc_spec = {
            .npc_def_id = INF_NPC_DEF_IDS[npc->type],
            .npc_slot = i,
            .npc_instance_id = npc->render_id,
            .npc_visible = npc->active,
            .npc_size = npc->size,
            .npc_anim_id = npc_anim_id,
            .x = npc->x,
            .y = npc->y,
            .dest_x = dest_x,
            .dest_y = dest_y,
            .render_movement_kind = render_movement_kind,
            .current_hitpoints = npc->hp,
            .base_hitpoints = npc->max_hp,
            .attack_style_this_tick = npc->attacked_this_tick
                ? npc->attack_style_this_tick
                : ATTACK_STYLE_NONE,
            .hit_landed_this_tick = npc->hit_landed_this_tick,
            .hit_damage = npc->hit_damage,
            .hit_was_successful = npc->hit_was_successful_this_tick,
            .hit_spell_type = npc->hit_spell_type,
            .attack_target_entity_idx = attack_target_entity_idx,
        };
        osrs_render_entity_from_npc_spec(&npc_spec, re);
    }

    encounter_resolve_attack_target(out, n, s->interaction.target_slot);
    *count = n;
}

static void inf_fill_render_entities(
    EncounterState* state,
    RenderEntity* out,
    int max_entities,
    int* count
) {
    inf_fill_render_entities_ctx(
        state, (EncounterContext*)inf_legacy_context(), out, max_entities, count);
}
