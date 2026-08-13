#line 7460 "encounter_inferno.h"
static void inf_put_int_ctx(
    EncounterState* state,
    EncounterContext* context,
    const char* key,
    int value
) {
    InfernoState* s = (InfernoState*)state;
    InfernoContext* ctx = (InfernoContext*)context;
    if (strcmp(key, "start_wave") == 0) {
        inf_require_valid_public_wave(value);
        s->start_wave = value - 1;
        ctx->config.start_wave = value - 1;
    }
    else if (strcmp(key, "seed") == 0) s->rng_state = (uint32_t)value;
    else if (strcmp(key, "world_offset_x") == 0) ctx->world_offset_x = value;
    else if (strcmp(key, "world_offset_y") == 0) ctx->world_offset_y = value;
    else if (strcmp(key, "player_dest_x") == 0) s->player_dest_x = value;
    else if (strcmp(key, "player_dest_y") == 0) s->player_dest_y = value;
    else if (strcmp(key, "human_command_mode") == 0)
        ctx->human_command_mode = encounter_require_binary_config("inferno", key, value);
    else if (strcmp(key, "terminal_penalty_enabled") == 0)
        ctx->config.terminal_penalty_enabled =
            encounter_require_binary_config("inferno", key, value);
    else if (strcmp(key, "curriculum_agent") == 0)
        ctx->config.curriculum_agent =
            encounter_require_binary_config("inferno", key, value);
    else if (strcmp(key, "curriculum_supply_jitter_mode") == 0) {
        inf_require_valid_curriculum_supply_mode(key, value);
        ctx->config.curriculum_supply_jitter_mode = value;
    }
    else if (strcmp(key, "curriculum_no_brew_mode") == 0) {
        inf_require_valid_curriculum_supply_mode(key, value);
        ctx->config.curriculum_no_brew_mode = value;
    }
    else if (strcmp(key, "loadout_profile_mode") == 0) {
        inf_require_valid_loadout_profile_mode(value);
        ctx->config.loadout_profile_mode = (InfLoadoutProfileMode)value;
    }
    else if (strcmp(key, "zuk_healer_reward_mode") == 0) {
        if (value < INF_ZUK_HEALER_REWARD_MODE_BASELINE ||
                value > INF_ZUK_HEALER_REWARD_MODE_TAGS_FIRST) {
            fprintf(stderr,
                "inferno: zuk_healer_reward_mode must be in [0,1], got %d\n",
                value);
            abort();
        }
        ctx->config.zuk_healer_reward_mode = value;
    }
    else if (strcmp(key, "joseph_reward_mode") == 0) {
        if (value < INF_JOSEPH_REWARD_MODE_OFF ||
                value > INF_JOSEPH_REWARD_MODE_ON) {
            fprintf(stderr,
                "inferno: joseph_reward_mode must be in [0,1], got %d\n",
                value);
            abort();
        }
        ctx->config.joseph_reward_mode = value;
    }
    else if (strcmp(key, "zuk_safe_untagged_healer_target_mask") == 0) {
        ctx->config.zuk_safe_untagged_healer_target_mask =
            encounter_require_binary_config("inferno", key, value);
    }
    else if (strcmp(key, "zuk_force_safe_untagged_healer_target_mask") == 0) {
        ctx->config.zuk_force_safe_untagged_healer_target_mask =
            encounter_require_binary_config("inferno", key, value);
    }
    else encounter_abort_unknown_config("inferno", "int", key);
}

static void inf_put_int(EncounterState* state, const char* key, int value) {
    inf_put_int_ctx(state, (EncounterContext*)inf_legacy_context(), key, value);
}

static void inf_put_float_ctx(
    EncounterState* state,
    EncounterContext* context,
    const char* key,
    float value
) {
    (void)state;
    InfernoContext* ctx = (InfernoContext*)context;
    if (strcmp(key, "damage_reward_coeff") == 0) ctx->config.damage_reward_coeff = value;
    else if (strcmp(key, "offensive_prayer_reward_coeff") == 0) {
        inf_require_nonnegative_float_config(key, value);
        ctx->config.offensive_prayer_reward_coeff = value;
    }
    else if (strcmp(key, "shield_penalty_coeff") == 0) ctx->config.shield_penalty_coeff = value;
    else if (strcmp(key, "tag_reward_coeff") == 0) ctx->config.tag_reward_coeff = value;
    else if (strcmp(key, "shield_tag_reward_coeff") == 0) ctx->config.shield_tag_reward_coeff = value;
    else if (strcmp(key, "death_penalty_coeff") == 0) {
        inf_require_nonnegative_float_config(key, value);
        ctx->config.death_penalty_coeff = value;
    }
    else if (strcmp(key, "phase_900_bonus") == 0) ctx->config.phase_900_bonus = value;
    else if (strcmp(key, "phase_600_bonus") == 0) ctx->config.phase_600_bonus = value;
    else if (strcmp(key, "phase_300_bonus") == 0) ctx->config.phase_300_bonus = value;
    else if (strcmp(key, "shield_penalty_episode_cap") == 0) ctx->config.shield_penalty_episode_cap = value;
    else if (strcmp(key, "late_start_supply_profile_scale") == 0) {
        inf_require_valid_supply_scale(value);
        ctx->config.late_start_supply_profile_scale = value;
    }
    else if (strcmp(key, "curriculum_supply_shared_jitter") == 0) {
        inf_require_unit_interval_float_config(key, value);
        ctx->config.curriculum_supply_shared_jitter = value;
    }
    else if (strcmp(key, "curriculum_supply_brew_jitter") == 0) {
        inf_require_unit_interval_float_config(key, value);
        ctx->config.curriculum_supply_brew_jitter = value;
    }
    else if (strcmp(key, "curriculum_supply_restore_jitter") == 0) {
        inf_require_unit_interval_float_config(key, value);
        ctx->config.curriculum_supply_restore_jitter = value;
    }
    else if (strcmp(key, "curriculum_no_brew_frac") == 0) {
        inf_require_unit_interval_float_config(key, value);
        ctx->config.curriculum_no_brew_frac = value;
    }
    else if (strcmp(key, "budget_loadout_fraction") == 0) {
        inf_require_valid_budget_loadout_fraction(value);
        ctx->config.budget_loadout_fraction = value;
    }
    else if (strcmp(key, "supply_milestone_brew_reward_coeff") == 0) {
        inf_require_nonnegative_float_config(key, value);
        ctx->config.supply_milestone_brew_reward_coeff = value;
    }
    else if (strcmp(key, "supply_milestone_restore_reward_coeff") == 0) {
        inf_require_nonnegative_float_config(key, value);
        ctx->config.supply_milestone_restore_reward_coeff = value;
    }
    else if (strcmp(key, "jad_damage_reward_coeff") == 0) ctx->config.jad_damage_reward_coeff = value;
    else if (strcmp(key, "zuk_healer_damage_reward_coeff") == 0) ctx->config.zuk_healer_damage_reward_coeff = value;
    else if (strcmp(key, "set_damage_reward_coeff") == 0) ctx->config.set_damage_reward_coeff = value;
    else if (strcmp(key, "jad_kill_bonus") == 0) ctx->config.jad_kill_bonus = value;
    else if (strcmp(key, "zuk_healer_kill_bonus") == 0) ctx->config.zuk_healer_kill_bonus = value;
    else if (strcmp(key, "set_kill_bonus") == 0) ctx->config.set_kill_bonus = value;
    else if (strcmp(key, "post_healer_zuk_damage_coeff") == 0) ctx->config.post_healer_zuk_damage_coeff = value;
    else if (strcmp(key, "post_healer_set_damage_reward_coeff") == 0) {
        ctx->config.post_healer_set_damage_reward_coeff = value;
    }
    else if (strcmp(key, "post_healer_set_kill_bonus") == 0) {
        ctx->config.post_healer_set_kill_bonus = value;
    }
    else if (strcmp(key, "post_healer_set_alive_tick_penalty_coeff") == 0) {
        ctx->config.post_healer_set_alive_tick_penalty_coeff = value;
    }
    else if (strcmp(key, "post_healer_set_alive_penalty_cap") == 0) {
        ctx->config.post_healer_set_alive_penalty_cap = value;
    }
    else if (strcmp(key, "zuk_healer_phase_hp_delta_coeff") == 0) ctx->config.zuk_healer_phase_hp_delta_coeff = value;
    else if (strcmp(key, "zuk_untagged_healer_tick_penalty_coeff") == 0) {
        ctx->config.zuk_untagged_healer_tick_penalty_coeff = value;
    }
    else if (strcmp(key, "zuk_untagged_healer_target_bonus_coeff") == 0) {
        ctx->config.zuk_untagged_healer_target_bonus_coeff = value;
    }
    else if (strcmp(key, "zuk_safe_untagged_healer_target_bonus_coeff") == 0) {
        ctx->config.zuk_safe_untagged_healer_target_bonus_coeff = value;
    }
    else if (strcmp(key, "zuk_untagged_healer_nonmagic_attack_bonus_coeff") == 0) {
        ctx->config.zuk_untagged_healer_nonmagic_attack_bonus_coeff = value;
    }
    else if (strcmp(key, "zuk_healer_mage_attack_penalty_coeff") == 0) {
        ctx->config.zuk_healer_mage_attack_penalty_coeff = value;
    }
    else if (strcmp(key, "post_jad_zuk_multiplier") == 0) ctx->config.post_jad_zuk_multiplier = value;
    else if (strcmp(key, "jad_alive_zuk_multiplier") == 0) ctx->config.jad_alive_zuk_multiplier = value;
    else encounter_abort_unknown_config("inferno", "float", key);
}

static void inf_put_float(EncounterState* state, const char* key, float value) {
    inf_put_float_ctx(state, (EncounterContext*)inf_legacy_context(), key, value);
}

static void inf_put_ptr_ctx(
    EncounterState* state,
    EncounterContext* context,
    const char* key,
    void* value
) {
    (void)state;
    InfernoContext* ctx = (InfernoContext*)context;
    if (strcmp(key, "collision_map") == 0)
        ctx->collision_map = (const CollisionMap*)value;
    else
        encounter_abort_unknown_config("inferno", "ptr", key);
}

static int inf_get_tick_ctx(EncounterState* state, EncounterContext* context) {
    (void)context;
    return ((InfernoState*)state)->tick;
}

static int inf_get_winner_ctx(EncounterState* state, EncounterContext* context) {
    (void)context;
    return ((InfernoState*)state)->winner;
}

static void* inf_get_log_ctx(EncounterState* state, EncounterContext* context) {
    InfernoState* s = (InfernoState*)state;
    InfernoContext* ctx = (InfernoContext*)context;
    if (s->episode_over && ctx->log) {
        ctx->log->episode_return += s->episode_return;
        ctx->log->episode_length += (float)s->tick;
        ctx->log->wins += (s->winner == INF_OUTCOME_PLAYER_WON) ? 1.0f : 0.0f;
        ctx->log->damage_dealt += s->total_damage_dealt;
        ctx->log->zuk_healer_damage += s->total_zuk_healer_damage;
        ctx->log->damage_received += s->total_damage_received;
        ctx->log->wave += (float)s->wave;
        ctx->log->prayer_correct += (float)s->total_prayer_correct;
        ctx->log->prayer_total += (float)s->total_npc_attacks;
        ctx->log->offensive_prayer_attacks +=
            (float)s->total_offensive_prayer_attacks;
        ctx->log->offensive_prayer_correct +=
            (float)s->total_offensive_prayer_correct;
        for (int i = 0; i < 4; i++) {
            ctx->log->offensive_prayer_attacks_by_style[i] +=
                (float)s->offensive_prayer_attacks_by_style[i];
            ctx->log->offensive_prayer_correct_by_style[i] +=
                (float)s->offensive_prayer_correct_by_style[i];
        }
        ctx->log->idle_ticks += (float)s->total_idle_ticks;
        ctx->log->attack_ready_no_attack_ticks +=
            (float)s->total_attack_ready_no_attack_ticks;
        ctx->log->target_available_no_attack_ticks +=
            (float)s->total_target_available_no_attack_ticks;
        ctx->log->safe_attack_opportunity_missed_ticks +=
            (float)s->total_safe_attack_opportunity_missed_ticks;
        ctx->log->progressless_ticks += (float)s->total_progressless_ticks;
        for (int i = 0; i < OSRS_INFERNO_IDLE_PHASE_COUNT; i++) {
            ctx->log->attack_ready_no_attack_ticks_by_phase[i] +=
                (float)s->attack_ready_no_attack_ticks_by_phase[i];
            ctx->log->target_available_no_attack_ticks_by_phase[i] +=
                (float)s->target_available_no_attack_ticks_by_phase[i];
            ctx->log->safe_attack_opportunity_missed_ticks_by_phase[i] +=
                (float)s->safe_attack_opportunity_missed_ticks_by_phase[i];
            ctx->log->progressless_ticks_by_phase[i] +=
                (float)s->progressless_ticks_by_phase[i];
        }
        ctx->log->brews_used += (float)s->total_brews_used;
        ctx->log->n += 1.0f;
        ctx->log->npc_kills += (float)s->total_npc_kills;
        ctx->log->min_zuk_hp_seen += (s->winner == INF_OUTCOME_PLAYER_WON)
            ? 0.0f
            : (s->min_zuk_hp_seen > 0.0f ? s->min_zuk_hp_seen : 1200.0f);
    }
    return ctx->log;
}

static const OsrsCombatVisualRow* inf_npc_combat_visual_row(
    InfNPCType type, AttackStyle style
) {
    assert(type >= 0 && type < INF_NUM_NPC_TYPES);
    return osrs_combat_visual_find_npc_id(
        (uint16_t)INF_NPC_DEF_IDS[type], style);
}

static void inf_npc_projectile_fallback(
    InfNPCType type,
    AttackStyle style,
    uint32_t* model_id,
    int* anim_id
) {
    assert(model_id && anim_id);
    switch (type) {
        case INF_NPC_BAT:
            *model_id = INF_GFX_1374_MODEL;
            *anim_id = INF_GFX_1374_ANIM;
            return;
        case INF_NPC_BLOB:
            if (style == ATTACK_STYLE_RANGED) {
                *model_id = INF_GFX_1383_MODEL;
                *anim_id = INF_GFX_1383_ANIM;
            } else if (style == ATTACK_STYLE_MAGIC) {
                *model_id = INF_GFX_1384_MODEL;
                *anim_id = OSRS_COMBAT_PROJECTILE_MISSING;
            }
            return;
        case INF_NPC_BLOB_RANGE:
            *model_id = INF_GFX_1379_MODEL;
            *anim_id = INF_GFX_1379_ANIM;
            return;
        case INF_NPC_BLOB_MAGE:
            *model_id = INF_GFX_1381_MODEL;
            *anim_id = INF_GFX_1381_ANIM;
            return;
        default:
            return;
    }
}

static void inf_render_post_tick_ctx(
    EncounterState* state,
    EncounterContext* context,
    EncounterOverlay* ov
) {
    (void)context;
    InfernoState* s = (InfernoState*)state;
    ov->tile_shadow_count = 0;
    ov->projectile_count = 0;
    encounter_write_terminal_status_text(
        s->episode_over,
        s->winner,
        INF_OUTCOME_PLAYER_WON,
        "Inferno cleared",
        inf_npc_type_name(s->last_hit_by_type),
        ov->status_text,
        sizeof(ov->status_text));
    ov->status_text_active = ov->status_text[0] != '\0';

    for (int i = 0; i < INF_MAX_NPCS; i++) {
        InfNPC* npc = &s->npcs[i];
        if (!npc->active || !npc->attacked_this_tick) continue;

        if (npc->type == INF_NPC_NIBBLER) continue;
        if (npc->resurrecting_this_tick) continue;

        if (npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer >= 0) continue;

        const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
        AttackStyle actual_style = npc->attack_style_this_tick;

        if (actual_style == ATTACK_STYLE_NONE && npc->type == INF_NPC_ZUK)
            actual_style = ATTACK_STYLE_MAGIC;

        if (npc->type == INF_NPC_HEALER_ZUK && npc->attack_visual_target < 0)
            continue;

        if (actual_style == ATTACK_STYLE_NONE) continue;

        if (actual_style == ATTACK_STYLE_MELEE) continue;

        int proj_style = encounter_attack_style_to_proj_style(actual_style);
        int npc_size = stats->size;
        int start_h = (int)(npc_size * 0.75f * 128);
        int end_h = 64;
        int curve = 16;
        float arc = 0.0f;

        int target_tile_x = s->player.x;
        int target_tile_y = s->player.y;
        int target_size = 1;
        int target_x = s->player.x, target_y = s->player.y;
        int target_npc_slot = -1;
        if (npc->attack_visual_target >= 0 && npc->attack_visual_target < INF_MAX_NPCS) {
            InfNPC* vt = &s->npcs[npc->attack_visual_target];
            target_tile_x = vt->x;
            target_tile_y = vt->y;
            target_size = vt->size;
            target_x = vt->x + vt->size / 2;
            target_y = vt->y + vt->size / 2;
            end_h = (int)(vt->size * 0.5f * 128);
            target_npc_slot = npc->attack_visual_target;
        }
        int dist = encounter_projectile_distance(
            npc->x, npc->y, npc_size, target_tile_x, target_tile_y, target_size,
            ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
        EncounterProjectileTiming timing =
            inf_npc_projectile_timing(npc->type, actual_style, dist);
        int duration = timing.visual_duration_ticks * 30;
        int start_delay = timing.visual_start_delay_ticks * 30;

        const OsrsCombatVisualRow* npc_visual =
            inf_npc_combat_visual_row(npc->type, actual_style);
        uint32_t proj_model_id = 0;
        int proj_anim_id = OSRS_COMBAT_PROJECTILE_MISSING;
        int impact_gfx_id = 0;
        if (npc_visual) {
            if (npc_visual->projectile.projectile_model_id > 0)
                proj_model_id = (uint32_t)npc_visual->projectile.projectile_model_id;
            if (npc_visual->projectile.projectile_anim_id > 0)
                proj_anim_id = npc_visual->projectile.projectile_anim_id;
            if (npc_visual->projectile.impact_spotanim_id > 0)
                impact_gfx_id = npc_visual->projectile.impact_spotanim_id;
        }

        if (proj_model_id == 0)
            inf_npc_projectile_fallback(
                npc->type, actual_style, &proj_model_id, &proj_anim_id);

        switch (npc->type) {
            case INF_NPC_JAD:
                if (actual_style == ATTACK_STYLE_MAGIC) {
                    arc = 1.0f;
                } else {
                    start_h = end_h;
                }
                start_delay = INF_JAD_PROJECTILE_DELAY * 30;
                break;
            case INF_NPC_HEALER_ZUK:
                arc = 3.0f;
                duration = (npc->attack_visual_target >= 0) ? 3 * 30 : 4 * 30;
                start_delay = 0;
                break;
            default: break;
        }

        if (npc->type == INF_NPC_JAD && actual_style == ATTACK_STYLE_MAGIC) {
            encounter_require_projectile_slots(ov, 3);
            uint32_t model_ids[3] = {
                INF_GFX_448_MODEL, INF_GFX_449_MODEL, INF_GFX_450_MODEL
            };
            int anim_ids[3] = {
                INF_GFX_448_ANIM, INF_GFX_449_ANIM, INF_GFX_450_ANIM
            };
            float offsets[3] = {1.0f, 0.5f, 0.0f};
            for (int j = 0; j < 3; j++) {
                OsrsProjectileEventSpec triple_spec = {
                    .src_x = npc->x,
                    .src_y = npc->y,
                    .dst_x = target_x,
                    .dst_y = target_y,
                    .style = proj_style,
                    .damage = (int)s->tick_scratch.damage_received,
                    .duration_ticks = duration,
                    .start_h = start_h,
                    .end_h = end_h,
                    .curve = curve,
                    .arc_height = arc,
                    .src_size = npc_size,
                    .dst_size = target_size,
                    .model_id = model_ids[j],
                    .anim_id = anim_ids[j],
                    .start_delay = start_delay,
                };
                int pi = target_npc_slot >= 0
                    ? osrs_emit_projectile_npc_to_npc(ov, &triple_spec, i, target_npc_slot)
                    : osrs_emit_projectile_npc_to_player(ov, &triple_spec, i);
                if (target_npc_slot >= 0)
                    encounter_set_projectile_target_npc_slot(ov, pi, target_npc_slot);
                encounter_set_projectile_offset(ov, pi, 0.0f, offsets[j], 0.0f);
            }
            continue;
        }

        if (target_npc_slot < 0)
            impact_gfx_id = 0;

        if (npc->type == INF_NPC_HEALER_ZUK && impact_gfx_id <= 0)
            impact_gfx_id = INF_GFX_659_ID;
        OsrsProjectileEventSpec attack_spec = {
            .src_x = npc->x,
            .src_y = npc->y,
            .dst_x = target_x,
            .dst_y = target_y,
            .style = proj_style,
            .damage = (int)s->tick_scratch.damage_received,
            .duration_ticks = duration,
            .start_h = start_h,
            .end_h = end_h,
            .curve = curve,
            .arc_height = arc,
            .src_size = npc_size,
            .dst_size = target_size,
            .model_id = proj_model_id,
            .anim_id = proj_anim_id,
            .impact_gfx_id = impact_gfx_id,
            .start_delay = start_delay,
        };
        int pi = target_npc_slot >= 0
            ? osrs_emit_projectile_npc_to_npc(ov, &attack_spec, i, target_npc_slot)
            : osrs_emit_projectile_npc_to_player(ov, &attack_spec, i);

        if (npc->type == INF_NPC_JAD && actual_style == ATTACK_STYLE_RANGED)
            encounter_set_projectile_motion_mode(
                ov, pi, ENCOUNTER_PROJECTILE_MOTION_TARGET_ANCHORED);

    }

    for (int i = 0; i < INF_MAX_PENDING_SPARKS; i++) {
        InfPendingSpark* spark = &s->pending_sparks[i];
        if (!spark->active || spark->visual_emitted) continue;

        const OsrsCombatVisualRow* spark_visual =
            inf_npc_combat_visual_row(INF_NPC_HEALER_ZUK, ATTACK_STYLE_MAGIC);
        uint32_t spark_model_id = INF_GFX_660_MODEL;
        int spark_anim_id = OSRS_COMBAT_PROJECTILE_MISSING;
        int spark_impact_gfx_id = INF_GFX_659_ID;
        if (spark_visual) {
            if (spark_visual->projectile.projectile_model_id > 0)
                spark_model_id = (uint32_t)spark_visual->projectile.projectile_model_id;
            if (spark_visual->projectile.projectile_anim_id > 0)
                spark_anim_id = spark_visual->projectile.projectile_anim_id;
            if (spark_visual->projectile.impact_spotanim_id > 0)
                spark_impact_gfx_id = spark_visual->projectile.impact_spotanim_id;
        }

        OsrsProjectileEventSpec spark_spec = {
            .src_x = spark->src_x,
            .src_y = spark->src_y,
            .dst_x = spark->x,
            .dst_y = spark->y,
            .style = encounter_attack_style_to_proj_style(ATTACK_STYLE_MAGIC),
            .damage = spark->damage,
            .duration_ticks = 4 * 30,
            .start_h = 96,
            .end_h = 64,
            .curve = 16,
            .arc_height = 3.0f,
            .src_size = 1,
            .dst_size = 1,
            .model_id = spark_model_id,
            .anim_id = spark_anim_id,
            .impact_gfx_id = spark_impact_gfx_id,
        };
        (void)osrs_emit_projectile_with_spec(ov, &spark_spec, 0);
        spark->visual_emitted = 1;
    }

    if (s->tick_scratch.player_attacked &&
        s->player_attack_style_id != ATTACK_STYLE_MELEE) {
        int target_idx = s->player_attack_npc_idx;
        if (target_idx >= 0 && target_idx < INF_MAX_NPCS) {
            InfNPC* target = &s->npcs[target_idx];
            int target_size = INF_NPC_STATS[target->type].size;
            int p_start_h = 64;
            int p_end_h = (int)(target_size * 0.5f * 128);
            int p_duration = s->player_attack_timing.visual_duration_ticks * 30;
            int p_start_delay = s->player_attack_timing.visual_start_delay_ticks * 30;
            uint8_t weapon = s->player.equipped[GEAR_SLOT_WEAPON];

            if (s->player_attack_style_id != ATTACK_STYLE_MAGIC) {
                const OsrsCombatProjectileProfile* base_profile =
                    osrs_combat_visual_ranged_projectile_profile(
                        weapon, OSRS_COMBAT_PROJECTILE_NONE);
                const OsrsCombatVisualRow* effect = NULL;
                if (s->player.used_special_this_tick && weapon < NUM_ITEMS) {
                    effect = osrs_combat_visual_find_special_projectile_item_id(
                        ITEM_DATABASE[weapon].item_id, ATTACK_STYLE_RANGED);
                }
                OsrsCombatProjectileSequencePart parts[OSRS_COMBAT_PROJECTILE_SEQUENCE_MAX];
                int part_count = osrs_combat_visual_build_projectile_sequence(
                    base_profile, effect, parts, OSRS_COMBAT_PROJECTILE_SEQUENCE_MAX);
                if (part_count < 0) abort();
                for (int part_i = 0; part_i < part_count; part_i++) {
                    const OsrsCombatProjectileProfile* profile = &parts[part_i].projectile;
                    if (profile->projectile_model_id <= 0) continue;
                    int visual_damage = part_i == part_count - 1
                        ? s->player_attack_dmg
                        : 0;
                    OsrsCombatProjectileEmitSpec emit_spec = {
                        .src_x = s->player.x,
                        .src_y = s->player.y,
                        .dst_x = target->x,
                        .dst_y = target->y,
                        .src_size = 1,
                        .dst_size = target_size,
                        .target_npc_slot = target_idx,
                        .attack_style = s->player_attack_style_id,
                        .damage = visual_damage,
                        .duration_ticks = p_duration,
                        .start_delay = p_start_delay,
                        .fallback_start_h = p_start_h,
                        .fallback_end_h = p_end_h,
                        .curve = 16,
                    };
                    (void)osrs_emit_combat_projectile_profile_player_to_npc(
                        ov, profile, &emit_spec);
                }
            }
        }
    }
}

static void inf_render_post_tick(EncounterState* state, EncounterOverlay* ov) {
    inf_render_post_tick_ctx(state, (EncounterContext*)inf_legacy_context(), ov);
}

static void* inf_get_player_for_input(void* state, int idx) {
    InfernoState* s = (InfernoState*)state;
    return (idx == 0) ? (void*)&s->player : NULL;
}

static int inf_first_cell_with_kind(const InfernoState* s, OsrsConsumableKind kind) {
    for (int c = 0; c < OSRS_INVENTORY_SIZE; c++) {
        const OsrsInventoryCell* cell = &s->player.inventory_cells[c];
        const OsrsItemContentMetadata* metadata =
            osrs_inventory_cell_metadata(cell);
        if (metadata->raw_osrs_id == 0) continue;
        if ((OsrsConsumableKind)metadata->consumable_kind != kind) continue;
        if (metadata->click_action == OSRS_CLICK_DRINK &&
                metadata->dose_count == 0) continue;
        return c;
    }
    return -1;
}

static void inf_translate_drink(const InfernoState* s, int* actions, OsrsConsumableKind kind) {
    int cell = inf_first_cell_with_kind(s, kind);
    if (cell >= 0) actions[INF_HEAD_DRINK] = cell + 1;
}

static void inf_translate_gear_swap(const InfernoState* s, int* actions, InfWeaponSet want) {
    const uint8_t* const* loadouts =
        inf_loadouts_for_profile(s->active_loadout_profile);
    const uint8_t* want_loadout = loadouts[want];
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        uint8_t item = want_loadout[slot];
        if (item == ITEM_NONE || s->player.equipped[slot] == item) continue;
        for (int c = 0; c < OSRS_INVENTORY_SIZE; c++) {
            if (osrs_inventory_cell_item_index(
                    &s->player.inventory_cells[c]) == item) {
                actions[INF_HEAD_EQUIP_SLOT(slot)] = c + 1;
                break;
            }
        }
    }
}

static void inf_translate_human_input_ctx(
    HumanInput* hi,
    int* actions,
    EncounterState* state,
    EncounterContext* context
) {
    for (int h = 0; h < INF_NUM_ACTION_HEADS; h++) actions[h] = 0;

    encounter_translate_movement(
        hi, actions, INF_HEAD_PRIMARY, inf_get_player_for_input, state);
    encounter_translate_prayer(hi, actions, INF_HEAD_PRAYER);
    encounter_translate_offensive_prayer(hi, actions, INF_HEAD_OFFENSIVE);
    InfernoState* s = (InfernoState*)state;
    InfernoContext* ctx = (InfernoContext*)context;
    inf_ensure_current_obs_slots_ctx(s, ctx);
    if (hi->pending_target_idx >= 0) {
        int found_slot = inf_find_target_obs_slot(s, hi->pending_target_idx);
        if (inf_obs_slot_is_targetable(s, ctx, found_slot)) {
            actions[INF_HEAD_PRIMARY] =
                inf_primary_attack_action_for_obs_slot(found_slot);
        }
    }

    if (hi->pending_gear >= 1 && hi->pending_gear <= INF_NUM_WEAPON_SETS)
        inf_translate_gear_swap(s, actions, (InfWeaponSet)(hi->pending_gear - 1));

    if (hi->pending_food) {
        int cell = inf_first_cell_with_kind(s, OSRS_CONSUMABLE_SHARK_FOOD);
        if (cell < 0) cell = inf_first_cell_with_kind(s, OSRS_CONSUMABLE_KARAMBWAN);
        if (cell >= 0) actions[INF_HEAD_EAT] = cell + 1;
    }

    if (hi->pending_potion == POTION_BREW)
        inf_translate_drink(s, actions, OSRS_CONSUMABLE_BREW);
    else if (hi->pending_potion == POTION_RESTORE)
        inf_translate_drink(s, actions, OSRS_CONSUMABLE_SUPER_RESTORE);
    else if (hi->pending_potion == POTION_BASTION)
        inf_translate_drink(s, actions, OSRS_CONSUMABLE_BASTION);
    else if (hi->pending_potion == POTION_STAMINA)
        inf_translate_drink(s, actions, OSRS_CONSUMABLE_STAMINA);

    if (hi->pending_spell == OSRS_SPELL_BLOOD_BARRAGE)
        actions[INF_HEAD_SPELL] = OSRS_SPELL_BLOOD_BARRAGE;
    else if (hi->pending_spell == OSRS_SPELL_ICE_BARRAGE)
        actions[INF_HEAD_SPELL] = OSRS_SPELL_ICE_BARRAGE;

    if (hi->pending_spec) actions[INF_HEAD_SPEC] = s->player.spec_armed ? 2 : 1;
}

static void inf_translate_human_input(HumanInput* hi, int* actions, EncounterState* state) {
    inf_translate_human_input_ctx(
        hi, actions, state, (EncounterContext*)inf_legacy_context());
}

static void inf_translate_human_commands_ctx(
    HumanInput* hi,
    int* actions,
    InfernoState* s,
    InfernoContext* ctx
) {
    for (int h = 0; h < INF_NUM_ACTION_HEADS; h++) actions[h] = 0;
    inf_ensure_current_obs_slots_ctx(s, ctx);

    for (int i = 0; i < hi->commands.count; i++) {
        const HumanCommand* cmd = &hi->commands.items[i];
        switch (cmd->kind) {
            case HUMAN_COMMAND_WALK:
                s->player_dest_x = cmd->world_x;
                s->player_dest_y = cmd->world_y;
                actions[INF_HEAD_PRIMARY] = 0;
                actions[INF_HEAD_SPELL] = 0;
                break;
            case HUMAN_COMMAND_ATTACK_NPC: {
                int found_slot = inf_find_target_obs_slot(s, cmd->npc_slot);
                actions[INF_HEAD_PRIMARY] = inf_obs_slot_is_targetable(
                    s, ctx, found_slot)
                        ? inf_primary_attack_action_for_obs_slot(found_slot) : 0;
                s->player_dest_x = -1;
                s->player_dest_y = -1;
                break;
            }
            case HUMAN_COMMAND_SPELL_TARGET: {
                int found_slot = inf_find_target_obs_slot(s, cmd->npc_slot);
                actions[INF_HEAD_PRIMARY] = inf_obs_slot_is_targetable(
                    s, ctx, found_slot)
                        ? inf_primary_attack_action_for_obs_slot(found_slot) : 0;
                if (cmd->spell == OSRS_SPELL_BLOOD_BARRAGE)
                    actions[INF_HEAD_SPELL] = OSRS_SPELL_BLOOD_BARRAGE;
                else if (cmd->spell == OSRS_SPELL_ICE_BARRAGE)
                    actions[INF_HEAD_SPELL] = OSRS_SPELL_ICE_BARRAGE;
                s->player_dest_x = -1;
                s->player_dest_y = -1;
                break;
            }
            case HUMAN_COMMAND_OVERHEAD_PRAYER:
                actions[INF_HEAD_PRAYER] = cmd->overhead_prayer;
                break;
            case HUMAN_COMMAND_OFFENSIVE_PRAYER:
                actions[INF_HEAD_OFFENSIVE] = cmd->offensive_prayer;
                break;
            case HUMAN_COMMAND_EAT: {
                int cell = inf_first_cell_with_kind(s, OSRS_CONSUMABLE_SHARK_FOOD);
                if (cell < 0)
                    cell = inf_first_cell_with_kind(s, OSRS_CONSUMABLE_KARAMBWAN);
                if (cell >= 0) actions[INF_HEAD_EAT] = cell + 1;
                break;
            }
            case HUMAN_COMMAND_DRINK:
                if (cmd->potion == POTION_BREW)
                    inf_translate_drink(s, actions, OSRS_CONSUMABLE_BREW);
                else if (cmd->potion == POTION_RESTORE)
                    inf_translate_drink(s, actions, OSRS_CONSUMABLE_SUPER_RESTORE);
                else if (cmd->potion == POTION_BASTION)
                    inf_translate_drink(s, actions, OSRS_CONSUMABLE_BASTION);
                else if (cmd->potion == POTION_STAMINA)
                    inf_translate_drink(s, actions, OSRS_CONSUMABLE_STAMINA);
                break;
            case HUMAN_COMMAND_SPEC_TOGGLE:
                actions[INF_HEAD_SPEC] = s->player.spec_armed ? 2 : 1;
                break;
            case HUMAN_COMMAND_EQUIP_INVENTORY_ITEM:
            case HUMAN_COMMAND_FIGHT_STYLE:
            case HUMAN_COMMAND_SET_AUTOCAST:
            case HUMAN_COMMAND_ITEM_ON_ITEM:
            case HUMAN_COMMAND_ITEM_ON_WIDGET:
            case HUMAN_COMMAND_SPELL_ON_WIDGET:
            case HUMAN_COMMAND_INVENTORY_PRIMARY_CLICK:
            case HUMAN_COMMAND_NONE:
                break;
        }
    }
}

static void inf_translate_human_commands(HumanInput* hi, int* actions, InfernoState* s) {
    inf_translate_human_commands_ctx(hi, actions, s, inf_legacy_context());
}

static void inf_step_human_commands_ctx(
    EncounterState* state,
    EncounterContext* context,
    HumanInput* hi
) {
    InfernoState* s = (InfernoState*)state;
    InfernoContext* ctx = (InfernoContext*)context;
    int actions[INF_NUM_ACTION_HEADS];
    ctx->human_command_mode = 1;
    ctx->human_commands = hi->commands.items;
    ctx->human_command_count = hi->commands.count;
    inf_refresh_human_loadout_stats(s);
    inf_translate_human_commands_ctx(hi, actions, s, ctx);
    inf_step_ctx(state, (EncounterContext*)ctx, actions);
    ctx->human_commands = NULL;
    ctx->human_command_count = 0;
    human_input_clear_pending(hi);
}

static void inf_step_human_commands(EncounterState* state, HumanInput* hi) {
    inf_step_human_commands_ctx(
        state, (EncounterContext*)inf_legacy_context(), hi);
}

#define INF_SNAPSHOT_MAGIC 0x1FE00001u
#define INF_SNAPSHOT_VERSION 22u

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t state_size;
    uint32_t reserved;
    uint64_t config_fingerprint;
    InfernoState state;
} InfSnapshot;

static void inf_hash_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = (const uint8_t*)ptr;
    for (size_t i = 0; i < n; i++) {
        *h ^= (uint64_t)p[i];
        *h *= 1099511628211ull;
    }
}

#define INF_HASH_CONFIG_FIELD(config, h, field) \
    inf_hash_bytes((h), &(config)->field, sizeof((config)->field))

static uint64_t inf_config_fingerprint(const InfConfig* config) {
    uint64_t h = 1469598103934665603ull;
    INF_HASH_CONFIG_FIELD(config, &h, start_wave);
    INF_HASH_CONFIG_FIELD(config, &h, loadout_profile_mode);
    INF_HASH_CONFIG_FIELD(config, &h, budget_loadout_fraction);
    INF_HASH_CONFIG_FIELD(config, &h, damage_reward_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, offensive_prayer_reward_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, shield_penalty_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, tag_reward_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, late_start_supply_profile_scale);
    INF_HASH_CONFIG_FIELD(config, &h, curriculum_agent);
    INF_HASH_CONFIG_FIELD(config, &h, curriculum_supply_jitter_mode);
    INF_HASH_CONFIG_FIELD(config, &h, curriculum_supply_shared_jitter);
    INF_HASH_CONFIG_FIELD(config, &h, curriculum_supply_brew_jitter);
    INF_HASH_CONFIG_FIELD(config, &h, curriculum_supply_restore_jitter);
    INF_HASH_CONFIG_FIELD(config, &h, curriculum_no_brew_mode);
    INF_HASH_CONFIG_FIELD(config, &h, curriculum_no_brew_frac);
    INF_HASH_CONFIG_FIELD(config, &h, supply_milestone_brew_reward_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, supply_milestone_restore_reward_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, death_penalty_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, terminal_penalty_enabled);
    INF_HASH_CONFIG_FIELD(config, &h, phase_900_bonus);
    INF_HASH_CONFIG_FIELD(config, &h, phase_600_bonus);
    INF_HASH_CONFIG_FIELD(config, &h, phase_300_bonus);
    INF_HASH_CONFIG_FIELD(config, &h, shield_penalty_episode_cap);
    INF_HASH_CONFIG_FIELD(config, &h, jad_damage_reward_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, zuk_healer_damage_reward_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, set_damage_reward_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, jad_kill_bonus);
    INF_HASH_CONFIG_FIELD(config, &h, zuk_healer_kill_bonus);
    INF_HASH_CONFIG_FIELD(config, &h, set_kill_bonus);
    INF_HASH_CONFIG_FIELD(config, &h, post_healer_zuk_damage_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, post_healer_set_damage_reward_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, post_healer_set_kill_bonus);
    INF_HASH_CONFIG_FIELD(config, &h, post_healer_set_alive_tick_penalty_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, post_healer_set_alive_penalty_cap);
    INF_HASH_CONFIG_FIELD(config, &h, zuk_healer_phase_hp_delta_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, zuk_untagged_healer_tick_penalty_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, zuk_untagged_healer_target_bonus_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, zuk_safe_untagged_healer_target_bonus_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, zuk_untagged_healer_nonmagic_attack_bonus_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, zuk_healer_mage_attack_penalty_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, zuk_safe_untagged_healer_target_mask);
    INF_HASH_CONFIG_FIELD(config, &h, zuk_force_safe_untagged_healer_target_mask);
    INF_HASH_CONFIG_FIELD(config, &h, zuk_healer_reward_mode);
    INF_HASH_CONFIG_FIELD(config, &h, post_jad_zuk_multiplier);
    INF_HASH_CONFIG_FIELD(config, &h, jad_alive_zuk_multiplier);
    INF_HASH_CONFIG_FIELD(config, &h, shield_tag_reward_coeff);
    INF_HASH_CONFIG_FIELD(config, &h, joseph_reward_mode);
    return h;
}

#undef INF_HASH_CONFIG_FIELD

static void inf_refresh_after_state_load(InfernoState* s, InfernoContext* ctx) {
    osrs_interaction_zero_serialized_route_padding(&s->player.interaction);
    osrs_interaction_zero_serialized_route_padding(&s->interaction);
    osrs_actor_route_cache_clear(&ctx->player_route_cache);
    inf_build_npc_stats();
    const uint8_t* const* loadouts =
        inf_loadouts_for_profile(s->active_loadout_profile);
    encounter_compute_loadout_stats(loadouts[INF_GEAR_MAGE], ATTACK_STYLE_MAGIC,
        s->player.offensive_prayer, s->player.current_magic,
        inf_player_spell_fight_style(&s->player),
        inf_player_magic_base_damage(&s->player),
        &s->loadout_stats[INF_GEAR_MAGE]);
    encounter_compute_loadout_stats(loadouts[INF_GEAR_LONG_RANGE], ATTACK_STYLE_RANGED,
        s->player.offensive_prayer, s->player.current_ranged,
        FIGHT_STYLE_RAPID, 0, &s->loadout_stats[INF_GEAR_LONG_RANGE]);
    encounter_compute_loadout_stats(loadouts[INF_GEAR_BP], ATTACK_STYLE_RANGED,
        s->player.offensive_prayer, s->player.current_ranged,
        FIGHT_STYLE_RAPID, 0, &s->loadout_stats[INF_GEAR_BP]);
    if (ctx->human_command_mode)
        inf_refresh_human_loadout_stats(s);
    inf_rebuild_npc_collision_flags(s);
    inf_refresh_current_obs_slots_ctx(s, ctx);
}

static size_t inf_snapshot_size_ctx(EncounterState* state, EncounterContext* context) {
    (void)state;
    (void)context;
    return sizeof(InfSnapshot);
}

static size_t inf_snapshot_size(EncounterState* state) {
    return inf_snapshot_size_ctx(state, (EncounterContext*)inf_legacy_context());
}

static void inf_snapshot_ctx(EncounterState* state, EncounterContext* context, void* out) {
    InfernoContext* ctx = (InfernoContext*)context;
    InfSnapshot* snap = (InfSnapshot*)out;
    encounter_snapshot_write_frame(
        snap, sizeof(*snap), INF_SNAPSHOT_MAGIC, INF_SNAPSHOT_VERSION,
        sizeof(InfernoState));
    snap->config_fingerprint = inf_config_fingerprint(&ctx->config);
    encounter_snapshot_copy_state_to(
        snap, offsetof(InfSnapshot, state), state, sizeof(InfernoState));
    osrs_interaction_zero_serialized_route_padding(&snap->state.player.interaction);
    osrs_interaction_zero_serialized_route_padding(&snap->state.interaction);
}

static void inf_snapshot(EncounterState* state, void* out) {
    inf_snapshot_ctx(state, (EncounterContext*)inf_legacy_context(), out);
}

static void inf_restore_ctx(
    EncounterState* state,
    EncounterContext* context,
    const void* data,
    size_t n
) {
    InfernoContext* ctx = (InfernoContext*)context;
    const InfSnapshot* snap = (const InfSnapshot*)data;
    encounter_snapshot_validate_frame(
        "inf_restore", data, n, sizeof(InfSnapshot),
        INF_SNAPSHOT_MAGIC, INF_SNAPSHOT_VERSION, sizeof(InfernoState));
    uint64_t live_fingerprint = inf_config_fingerprint(&ctx->config);
    if (snap->config_fingerprint != live_fingerprint) {
        fprintf(stderr,
            "inf_restore: config fingerprint mismatch (got 0x%016llx, want 0x%016llx)\n",
            (unsigned long long)snap->config_fingerprint,
            (unsigned long long)live_fingerprint);
        abort();
    }

    InfernoState* dst = (InfernoState*)state;
    encounter_snapshot_copy_state_from(
        dst, data, offsetof(InfSnapshot, state), sizeof(InfernoState));
    inf_reset_npc_player_los_frame(ctx);
    inf_refresh_after_state_load(dst, ctx);
}

static void inf_restore(EncounterState* state, const void* data, size_t n) {
    inf_restore_ctx(state, (EncounterContext*)inf_legacy_context(), data, n);
}

typedef struct {
    InfernoState* state;
    InfernoContext* context;
} InfLabInvocation;

static int inf_lab_apply_script_line_alloc_json_void(
    void* data, const char* line, char** out_json
) {
    InfLabInvocation* invocation = (InfLabInvocation*)data;
    if (!out_json) inf_lab_abort("json output pointer is required");
    *out_json = NULL;
    return inf_lab_apply_script_line_impl_ctx(
        invocation->state, invocation->context, line, out_json);
}

static int inf_apply_lab_command_ctx(
    EncounterState* state,
    EncounterContext* context,
    const char* line
) {
    InfLabInvocation invocation = {
        .state = (InfernoState*)state,
        .context = (InfernoContext*)context,
    };
    return encounter_apply_lab_command_dump_wrapper(
        &invocation,
        line,
        INF_LAB_LINE_DUMP,
        inf_lab_apply_script_line_alloc_json_void);
}

static const EncounterDef ENCOUNTER_INFERNO = {
    .name = "inferno",
    .obs_size = INF_NUM_OBS,
    .num_action_heads = INF_NUM_ACTION_HEADS,
    .action_head_dims = INF_ACTION_DIMS,
    .mask_size = INF_ACTION_MASK_SIZE,

    .state_size = sizeof(InfernoState),
    .context_size = sizeof(InfernoContext),
    .init_context = inf_init_context,
    .destroy_context = inf_destroy_context,
    .init_state = inf_init_state,
    .finalize_context = inf_finalize_context,

    .create = inf_create,
    .destroy = inf_destroy,
    .reset = inf_reset_ctx,
    .step = inf_step_ctx,
    .step_human_commands = inf_step_human_commands_ctx,

    .snapshot_size = inf_snapshot_size_ctx,
    .snapshot = inf_snapshot_ctx,
    .restore = inf_restore_ctx,

    .write_obs = inf_write_obs_ctx,
    .write_mask = inf_write_mask_ctx,
    .get_reward = inf_get_reward_ctx,
    .is_terminal = inf_is_terminal_ctx,

    .get_entity_count = inf_get_entity_count_ctx,
    .get_entity = inf_get_entity_ctx,
    .fill_render_entities = inf_fill_render_entities_ctx,

    .put_int = inf_put_int_ctx,
    .put_float = inf_put_float_ctx,
    .put_ptr = inf_put_ptr_ctx,

    .arena_base_x = INF_ARENA_MIN_X,
    .arena_base_y = INF_ARENA_MIN_Y,
    .arena_width = INF_ARENA_WIDTH,
    .arena_height = INF_ARENA_HEIGHT,

    .translate_human_input = inf_translate_human_input_ctx,
    .is_human_targetable_npc_slot = inf_is_human_targetable_npc_slot_ctx,
    .apply_lab_command = inf_apply_lab_command_ctx,
    .head_move = INF_HEAD_PRIMARY,
    .head_prayer = INF_HEAD_PRAYER,
    .head_target = INF_HEAD_PRIMARY,

    .render_post_tick = inf_render_post_tick_ctx,
    .get_log = inf_get_log_ctx,
    .get_tick = inf_get_tick_ctx,
    .get_winner = inf_get_winner_ctx,
};

__attribute__((constructor))
static void inf_register(void) {
    encounter_register(&ENCOUNTER_INFERNO);
}
