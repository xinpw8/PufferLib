#line 76 "encounter_colosseum.h"

static void col_put_int_ctx(
    EncounterState* state,
    EncounterContext* context,
    const char* key,
    int value
) {
    ColosseumState* s = (ColosseumState*)state;
    ColosseumContext* ctx = col_require_context(context);
    if (strcmp(key, "start_wave") == 0) {
        int internal = value - 1;
        if (internal < 0) internal = 0;
        if (internal >= COLO_NUM_WAVES) internal = COLO_NUM_WAVES - 1;
        s->start_wave = internal;
        ctx->config.start_wave = internal;
    }
    else if (strcmp(key, "seed") == 0) s->rng_state = (uint32_t)value;
    else if (strcmp(key, "world_offset_x") == 0) ctx->world_offset_x = value;
    else if (strcmp(key, "world_offset_y") == 0) ctx->world_offset_y = value;
    else if (strcmp(key, "player_dest_x") == 0) s->player_dest_x = value;
    else if (strcmp(key, "player_dest_y") == 0) s->player_dest_y = value;
    else if (strcmp(key, "human_command_mode") == 0)
        ctx->human_command_mode = encounter_require_binary_config("colosseum", key, value);
    else if (strcmp(key, "curriculum_agent") == 0)
        ctx->config.curriculum_agent =
            encounter_require_binary_config("colosseum", key, value);
    else if (strcmp(key, "mask_inventory_heads") == 0)
        ctx->config.mask_inventory_heads =
            encounter_require_binary_config("colosseum", key, value);
    else if (strcmp(key, "laser_obs_mode") == 0)
        ctx->config.laser_obs_mode =
            encounter_require_binary_config("colosseum", key, value);
    else if (strcmp(key, "farm_safe_damage_cap") == 0)
        ctx->config.farm_safe_damage_cap =
            encounter_require_binary_config("colosseum", key, value);
    else if (strcmp(key, "farm_cap_waves") == 0) {
        if (value < 0 || value > COLO_NUM_WAVES) {
            fprintf(stderr, "colosseum: farm_cap_waves must be 0..%d, got %d\n",
                COLO_NUM_WAVES, value);
            abort();
        }
        ctx->config.farm_cap_waves = value;
    }
    else if (strcmp(key, "late_start_state_mode") == 0) {
        if (value < 0 || value > 2) {
            fprintf(stderr, "colosseum: late_start_state_mode must be 0..2, got %d\n",
                value);
            abort();
        }
        ctx->config.late_start_state_mode = value;
    }
    else if (strcmp(key, "bis_gear_oracle_mode") == 0)
        ctx->config.bis_gear_oracle_mode =
            encounter_require_binary_config("colosseum", key, value);
    else if (strcmp(key, "episode_max_ticks_override") == 0) {
        if (value < 0) {
            fprintf(stderr, "colosseum: episode_max_ticks_override must be >= 0, got %d\n",
                value);
            abort();
        }
        ctx->config.episode_max_ticks_override = value;
    }
    else if (strcmp(key, "damage_scale_anneal_ticks") == 0) {
        if (value < 0) {
            fprintf(stderr, "colosseum: damage_scale_anneal_ticks must be >= 0, got %d\n",
                value);
            abort();
        }
        ctx->config.damage_scale_anneal_ticks = value;
    }
    else if (strcmp(key, "loadout_profile_mode") == 0) {
        if (value < 0 || value > COLO_LOADOUT_PROFILE_MODE_MIXED) {
            fprintf(stderr, "colosseum: loadout_profile_mode must be 0..2, got %d\n", value);
            abort();
        }
        ctx->config.loadout_profile_mode = value;
    }
    else encounter_abort_unknown_config("colosseum", "int", key);
}

static void col_put_float_ctx(
    EncounterState* state,
    EncounterContext* context,
    const char* key,
    float value
) {
    (void)state;
    ColosseumContext* ctx = col_require_context(context);
    if (strcmp(key, "damage_reward_coeff") == 0) ctx->config.damage_reward_coeff = value;
    else if (strcmp(key, "late_start_supply_fraction_per_wave") == 0) {
        if (value < 0.0f || value > 0.125f) {
            fprintf(stderr,
                "colosseum: late_start_supply_fraction_per_wave must be 0..0.125, got %f\n",
                (double)value);
            abort();
        }
        ctx->config.late_start_supply_fraction_per_wave = value;
    }
    else if (strcmp(key, "boss_damage_reward_coeff") == 0)
        ctx->config.boss_damage_reward_coeff = value;
    else if (strcmp(key, "wave_clear_bonus") == 0) ctx->config.wave_clear_bonus = value;
    else if (strcmp(key, "boss_phase_bonus") == 0) ctx->config.boss_phase_bonus = value;
    else if (strcmp(key, "win_bonus") == 0) ctx->config.win_bonus = value;
    else if (strcmp(key, "death_penalty_coeff") == 0) ctx->config.death_penalty_coeff = value;
    else if (strcmp(key, "timeout_penalty") == 0) ctx->config.timeout_penalty = value;
    else if (strcmp(key, "beginner_loadout_fraction") == 0) {
        if (!(value >= 0.0f && value <= 1.0f)) {
            fprintf(stderr, "colosseum: beginner_loadout_fraction must be in [0,1], got %.6f\n",
                    value);
            abort();
        }
        ctx->config.beginner_loadout_fraction = value;
    }
    else if (strcmp(key, "prayer_switch_fail_prob") == 0) {
        if (!(value >= 0.0f && value <= 1.0f)) {
            fprintf(stderr, "colosseum: prayer_switch_fail_prob must be in [0,1], got %.6f\n",
                    value);
            abort();
        }
        ctx->config.prayer_switch_fail_prob = value;
    }
    else if (strcmp(key, "player_damage_received_scale") == 0) {
        if (!(value >= 0.0f && value <= 1.0f)) {
            fprintf(stderr,
                "colosseum: player_damage_received_scale must be in [0,1], got %.6f\n",
                value);
            abort();
        }
        ctx->config.player_damage_received_scale = value;
    }
    else if (strcmp(key, "damage_scale_anneal_start") == 0) {
        if (!(value >= 0.0f && value <= 1.0f)) {
            fprintf(stderr,
                "colosseum: damage_scale_anneal_start must be in [0,1], got %.6f\n",
                value);
            abort();
        }
        ctx->config.damage_scale_anneal_start = value;
    }
    else encounter_abort_unknown_config("colosseum", "float", key);
}

static void col_put_ptr_ctx(
    EncounterState* state,
    EncounterContext* context,
    const char* key,
    void* value
) {
    (void)state;
    ColosseumContext* ctx = col_require_context(context);
    if (strcmp(key, "collision_map") == 0)
        ctx->collision_map = (const CollisionMap*)value;
    else
        encounter_abort_unknown_config("colosseum", "ptr", key);
}

static int col_get_tick_ctx(EncounterState* state, EncounterContext* context) {
    (void)context;
    return ((ColosseumState*)state)->tick;
}

static int col_get_winner_ctx(EncounterState* state, EncounterContext* context) {
    (void)context;
    return ((ColosseumState*)state)->winner;
}

static void* col_get_log_ctx(EncounterState* state, EncounterContext* context) {
    ColosseumState* s = (ColosseumState*)state;
    (void)context;
    s->log.episode_return = s->episode_return;
    s->log.episode_length = s->tick;
    s->log.wave_reached = s->wave;
    s->log.win = (s->episode_over && s->winner == COLO_OUTCOME_PLAYER_WON) ? 1 : 0;
    s->log.timed_out = (s->episode_over && s->time_limit_truncated) ? 1 : 0;
    s->log.died = (s->episode_over && s->winner == COLO_OUTCOME_PLAYER_DIED
                   && !s->time_limit_truncated) ? 1 : 0;
    s->log.boss_phase_reached = s->sol.phase;
    s->log.active_tier_sum = col_modifier_active_tier_sum(s);
    return &s->log;
}

static const char* col_attack_style_debug_name(AttackStyle style) {
    switch (style) {
        case ATTACK_STYLE_NONE: return "none";
        case ATTACK_STYLE_MELEE: return "melee";
        case ATTACK_STYLE_RANGED: return "ranged";
        case ATTACK_STYLE_MAGIC: return "magic";
    }
    abort();
}

static const OsrsCombatVisualRow* col_require_npc_combat_visual_projectile_row(
    ColoNpcType type,
    AttackStyle style
) {
    assert(type >= 0 && type < COLO_NUM_NPC_TYPES);
    int npc_def_id = COLO_NPC_DEF_IDS[type];
    const OsrsCombatVisualRow* row = osrs_combat_visual_find_npc_id(
        (uint16_t)npc_def_id, style);
    if (!row || (row->projectile.travel_spotanim_id <= 0 &&
            row->projectile.projectile_model_id <= 0)) {
        fprintf(stderr,
            "colosseum: missing real combat visual projectile for npc def id %d style %s\n",
            npc_def_id, col_attack_style_debug_name(style));
        abort();
    }
    return row;
}

enum {
    COLO_JAVELIN_SKYFALL_LAUNCH_TRAVEL_GFX_ID = 2674,
    COLO_JAVELIN_SKYFALL_DROP_TRAVEL_GFX_ID = 2675,
    COLO_JAVELIN_SKYFALL_IMPACT_GFX_ID = 2676,
    COLO_JAVELIN_SKYFALL_SKY_HEIGHT = 1536,
    COLO_JAVELIN_SKYFALL_GROUND_HEIGHT = 16,
    COLO_JAVELIN_SKYFALL_LAUNCH_CURVE = -48,
    COLO_JAVELIN_SKYFALL_DROP_CURVE = 48,

    COLO_JAVELIN_SKYFALL_DROP_GAME_TICKS = COLO_JAVELIN_SKYFALL_DELAY / 2,
    COLO_JAVELIN_SKYFALL_APEX_GAME_TICKS = 1,
    COLO_JAVELIN_PROJECTILE_RELEASE_DELAY_TICKS = 1,

    COLO_MAGIC_CAST_RELEASE_DELAY_TICKS = 1,
};

static float col_javelin_skyfall_shadow_scale(const ColoJavelinState* jv) {
    if (!jv->skyfall_pending || jv->skyfall_timer <= 0 ||
            jv->skyfall_timer > COLO_JAVELIN_SKYFALL_DELAY) {
        fprintf(stderr, "colosseum: invalid skyfall shadow timer %d pending %d\n",
            jv->skyfall_timer, jv->skyfall_pending);
        abort();
    }
    float progress = (float)(COLO_JAVELIN_SKYFALL_DELAY - jv->skyfall_timer) /
        (float)(COLO_JAVELIN_SKYFALL_DELAY - 1);
    return 0.35f + 0.65f * progress;
}

static void col_emit_javelin_skyfall_shadow(EncounterOverlay* ov, const ColoJavelinState* jv) {
    encounter_emit_tile_shadow(
        ov,
        jv->skyfall_tile_x,
        jv->skyfall_tile_y,
        col_javelin_skyfall_shadow_scale(jv));
}

typedef struct {
    int release_delay_ticks;
    int set_delay;
} ColoProjectileReleaseProfile;

static ColoProjectileReleaseProfile col_npc_projectile_release_profile(
    ColoNpcType type, AttackStyle style
) {
    switch (type) {
        case COLO_JAVELIN_COLOSSUS:
            if (style == ATTACK_STYLE_RANGED)
                return (ColoProjectileReleaseProfile){
                    .release_delay_ticks = COLO_JAVELIN_PROJECTILE_RELEASE_DELAY_TICKS };
            break;
        case COLO_SERPENT_SHAMAN:
        case COLO_SHOCKWAVE_COLOSSUS:
            if (style == ATTACK_STYLE_MAGIC)
                return (ColoProjectileReleaseProfile){
                    .release_delay_ticks = COLO_MAGIC_CAST_RELEASE_DELAY_TICKS };
            break;
        case COLO_MANTICORE:

            return (ColoProjectileReleaseProfile){ .set_delay = 1 };
        default:
            break;
    }
    return (ColoProjectileReleaseProfile){ 0 };
}

static EncounterProjectileTiming col_npc_projectile_timing(
    ColoNpcType type,
    AttackStyle style,
    int distance
) {
    EncounterProjectileDelayKind kind = encounter_projectile_delay_kind_for_style(style);
    ColoProjectileReleaseProfile profile = col_npc_projectile_release_profile(type, style);

    EncounterProjectileDelayOptions options = {0};
    options.set_delay = profile.set_delay;
    int damage_delay = encounter_projectile_hit_delay(distance, 0, kind, options);

    int release = profile.release_delay_ticks;
    int max_release = damage_delay - 1;
    if (release > max_release) release = max_release;
    if (release < 0) release = 0;
    options.start_delay = release;

    EncounterProjectileTiming timing =
        encounter_projectile_timing(distance, 0, kind, options);

    if (timing.visual_duration_ticks < 1 ||
            timing.visual_start_delay_ticks + timing.visual_duration_ticks !=
                timing.damage_delay_ticks) {
        fprintf(stderr,
            "colosseum: projectile timing invariant violated (type %d style %d dist %d: "
            "release %d + flight %d != landing %d)\n",
            (int)type, (int)style, distance,
            timing.visual_start_delay_ticks, timing.visual_duration_ticks,
            timing.damage_delay_ticks);
        abort();
    }
    return timing;
}

static uint32_t col_manticore_orb_model_id(AttackStyle style) {
    switch (style) {
        case ATTACK_STYLE_RANGED: return 51221u;
        case ATTACK_STYLE_MAGIC: return 51215u;
        case ATTACK_STYLE_MELEE: return 51213u;
        case ATTACK_STYLE_NONE: break;
    }
    fprintf(stderr, "colosseum: invalid manticore orb style %d for telegraph model\n",
        (int)style);
    abort();
}

static int col_manticore_orb_anim_id(AttackStyle style) {
    switch (style) {
        case ATTACK_STYLE_RANGED: return 10327;
        case ATTACK_STYLE_MAGIC: return 10329;
        case ATTACK_STYLE_MELEE: return 10328;
        case ATTACK_STYLE_NONE: break;
    }
    fprintf(stderr, "colosseum: invalid manticore orb style %d for telegraph anim\n",
        (int)style);
    abort();
}

static void col_emit_javelin_skyfall_launch_projectile(
    ColosseumState* s,
    EncounterOverlay* ov,
    int npc_slot
) {
    ColoNPC* npc = &s->npcs[npc_slot];
    ColoJavelinState* jv = colo_npc_javelin(npc);
    if (!jv->skyfall_pending) {
        fprintf(stderr, "colosseum: skyfall launch render event without pending skyfall\n");
        abort();
    }

    int npc_size = col_npc_effective_size(npc);
    encounter_require_projectile_slots(ov, 1);

    OsrsProjectileEventSpec projectile_spec = {
        .src_x = npc->x,
        .src_y = npc->y,
        .dst_x = npc->x,
        .dst_y = npc->y,
        .style = encounter_attack_style_to_proj_style(ATTACK_STYLE_RANGED),
        .damage = 0,
        .duration_ticks = (COLO_JAVELIN_SKYFALL_DELAY -
            COLO_JAVELIN_SKYFALL_DROP_GAME_TICKS -
            COLO_JAVELIN_SKYFALL_APEX_GAME_TICKS) * 30,
        .start_h = (int)(npc_size * 0.75f * 128),
        .end_h = COLO_JAVELIN_SKYFALL_SKY_HEIGHT,
        .curve = COLO_JAVELIN_SKYFALL_LAUNCH_CURVE,
        .arc_height = 0.0f,
        .src_size = npc_size,
        .dst_size = npc_size,
        .model_id = 0u,
        .anim_id = -1,
        .travel_gfx_id = COLO_JAVELIN_SKYFALL_LAUNCH_TRAVEL_GFX_ID,
        .impact_gfx_id = 0,
        .start_delay = 0,
    };
    int projectile_idx = osrs_emit_projectile_with_spec(
        ov,
        &projectile_spec,
        0);
    encounter_set_projectile_source_npc_slot(ov, projectile_idx, npc_slot);
}

static void col_emit_manticore_orb_telegraphs(
    EncounterOverlay* ov,
    const ColoNPC* npc,
    int npc_slot
) {
    const ColoManticoreState* mc = &npc->type_state.manticore;
    if (mc->cycle_step < 0) return;
    if (mc->phase == COLO_MANTICORE_PHASE_CHARGING &&
            npc->attack_timer >
                COLO_MANTICORE_CHARGE_TICKS - COLO_MANTICORE_ARM_ANIMATION_TICKS)
        return;
    if (mc->cycle_step >= 3) {
        fprintf(stderr, "colosseum: invalid manticore cycle_step %d\n", mc->cycle_step);
        abort();
    }
    int npc_size = col_npc_effective_size(npc);
    for (int orb = mc->cycle_step; orb < 3; orb++) {
        AttackStyle style = mc->orb_style[orb];
        int stack_index = orb - mc->cycle_step;
        encounter_emit_floating_model(
            ov,
            ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT,
            npc_slot,
            npc->x,
            npc->y,
            col_manticore_orb_model_id(style),
            col_manticore_orb_anim_id(style),
            2.35f + (float)stack_index * 0.62f,
            0.0f,
            0.72f + 0.05f * (float)(npc_size - 1));
    }
}

static void col_emit_javelin_skyfall_drop_projectile(
    EncounterOverlay* ov,
    const ColoJavelinState* jv
) {
    encounter_require_projectile_slots(ov, 1);
    OsrsProjectileEventSpec projectile_spec = {
        .src_x = jv->skyfall_tile_x,
        .src_y = jv->skyfall_tile_y,
        .dst_x = jv->skyfall_tile_x,
        .dst_y = jv->skyfall_tile_y,
        .style = encounter_attack_style_to_proj_style(ATTACK_STYLE_RANGED),
        .damage = 0,
        .duration_ticks = COLO_JAVELIN_SKYFALL_DROP_GAME_TICKS * 30,
        .start_h = COLO_JAVELIN_SKYFALL_SKY_HEIGHT,
        .end_h = COLO_JAVELIN_SKYFALL_GROUND_HEIGHT,
        .curve = COLO_JAVELIN_SKYFALL_DROP_CURVE,
        .arc_height = 0.0f,
        .src_size = 1,
        .dst_size = 1,
        .model_id = 0u,
        .anim_id = -1,
        .travel_gfx_id = COLO_JAVELIN_SKYFALL_DROP_TRAVEL_GFX_ID,
        .impact_gfx_id = COLO_JAVELIN_SKYFALL_IMPACT_GFX_ID,
        .start_delay = 0,
    };
    (void)osrs_emit_projectile_with_spec(
        ov,
        &projectile_spec,
        0);
}

static void col_emit_npc_attack_projectile(
    ColosseumState* s,
    ColosseumContext* ctx,
    EncounterOverlay* ov,
    int npc_slot
) {
    ColoNPC* npc = &s->npcs[npc_slot];
    AttackStyle actual_style = npc->attack_style_this_tick;
    if (actual_style == ATTACK_STYLE_NONE)
        return;
    if (actual_style == ATTACK_STYLE_MELEE && npc->type != COLO_MANTICORE)
        return;
    if (col_type_is_hazard_entity(npc->type))
        return;
    if (npc->type == COLO_JAVELIN_COLOSSUS &&
            ctx->npc_render_attack_anim[npc_slot] ==
                COLO_NPC_RENDER_ATTACK_JAVELIN_SKYFALL) {
        col_emit_javelin_skyfall_launch_projectile(s, ov, npc_slot);
        return;
    }

    int npc_size = col_npc_effective_size(npc);
    int start_h = (int)(npc_size * 0.75f * 128);
    int end_h = 64;
    int target_tile_x = s->player.x;
    int target_tile_y = s->player.y;
    int target_x = s->player.x;
    int target_y = s->player.y;
    int target_size = 1;
    int target_npc_slot = -1;

    int dist = encounter_projectile_distance(
        npc->x, npc->y, npc_size, target_tile_x, target_tile_y, target_size,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    EncounterProjectileTiming timing =
        col_npc_projectile_timing(npc->type, actual_style, dist);
    int duration = timing.visual_duration_ticks * 30;
    int start_delay = timing.visual_start_delay_ticks * 30;

    const OsrsCombatVisualRow* npc_visual =
        col_require_npc_combat_visual_projectile_row(npc->type, actual_style);
    const OsrsCombatProjectileProfile* projectile = &npc_visual->projectile;
    uint32_t model_id = projectile->projectile_model_id > 0
        ? (uint32_t)projectile->projectile_model_id
        : 0u;
    int anim_id = projectile->projectile_model_id > 0
        ? projectile->projectile_anim_id
        : OSRS_COMBAT_PROJECTILE_MISSING;
    int travel_gfx_id = osrs_combat_projectile_value_or(
        projectile->travel_spotanim_id, 0);
    int launch_gfx_id = osrs_combat_projectile_value_or(
        projectile->launch_spotanim_id, 0);
    int impact_gfx_id = osrs_combat_projectile_value_or(
        projectile->impact_spotanim_id, 0);
    int curve = osrs_combat_projectile_value_or(projectile->projectile_angle, 16);
    float arc = osrs_combat_projectile_profile_arc_height(projectile, actual_style);

    if (actual_style == ATTACK_STYLE_MAGIC && arc <= 0.0f)
        arc = 1.0f;
    encounter_require_projectile_slots(ov, 1);
    OsrsProjectileEventSpec spec = {
        .src_x = npc->x,
        .src_y = npc->y,
        .dst_x = target_x,
        .dst_y = target_y,
        .style = encounter_attack_style_to_proj_style(actual_style),
        .damage = (int)s->tick_scratch.damage_received,
        .duration_ticks = duration,
        .start_h = start_h,
        .end_h = end_h,
        .curve = curve,
        .arc_height = arc,
        .src_size = npc_size,
        .dst_size = target_size,
        .model_id = model_id,
        .anim_id = anim_id,
        .travel_gfx_id = travel_gfx_id,
        .launch_gfx_id = launch_gfx_id,
        .impact_gfx_id = impact_gfx_id,
        .start_delay = start_delay,
    };

    if (target_npc_slot >= 0) {
        (void)osrs_emit_projectile_npc_to_npc(
            ov, &spec, npc_slot, target_npc_slot);
    } else {
        (void)osrs_emit_projectile_npc_to_player(ov, &spec, npc_slot);
    }
}

static void col_emit_player_venator_bounce_projectiles(
    ColosseumState* s,
    ColosseumContext* ctx,
    EncounterOverlay* ov,
    const OsrsCombatProjectileProfile* profile,
    int start_delay,
    int duration
) {
    if (ctx->player_venator_chain_count <= 1) return;
    if (ctx->player_venator_chain_count > OSRS_VENATOR_MAX_CHAIN_HITS) abort();
    if (!profile || profile->projectile_model_id <= 0) {
        fprintf(stderr, "colosseum: missing Venator bounce projectile model\n");
        abort();
    }

    int bounce_count = ctx->player_venator_chain_count - 1;
    encounter_require_projectile_slots(ov, bounce_count);
    for (int hop = 1; hop < ctx->player_venator_chain_count; hop++) {
        int source_slot = ctx->player_venator_chain_slots[hop - 1];
        int target_slot = ctx->player_venator_chain_slots[hop];
        if (source_slot < 0 || source_slot >= COLO_MAX_NPCS ||
                target_slot < 0 || target_slot >= COLO_MAX_NPCS) {
            fprintf(stderr, "colosseum: invalid Venator chain slot\n");
            abort();
        }
        ColoNPC* source = &s->npcs[source_slot];
        ColoNPC* target = &s->npcs[target_slot];
        if (!source->active || !target->active) {
            fprintf(stderr, "colosseum: Venator chain target inactive\n");
            abort();
        }
        int source_size = col_npc_effective_size(source);
        int target_size = col_npc_effective_size(target);
        int source_h = (int)(source_size * 0.5f * 128);
        int target_h = (int)(target_size * 0.5f * 128);
        OsrsProjectileEventSpec spec = {
            .src_x = source->x,
            .src_y = source->y,
            .dst_x = target->x,
            .dst_y = target->y,
            .style = encounter_attack_style_to_proj_style(ATTACK_STYLE_RANGED),
            .damage = 0,
            .duration_ticks = duration,
            .start_h = source_h,
            .end_h = target_h,
            .curve = osrs_combat_projectile_value_or(profile->projectile_angle, 16),
            .arc_height = osrs_combat_projectile_profile_arc_height(
                profile, ATTACK_STYLE_RANGED),
            .src_size = source_size,
            .dst_size = target_size,
            .model_id = (uint32_t)profile->projectile_model_id,
            .anim_id = profile->projectile_anim_id,
            .travel_gfx_id = 0,
            .launch_gfx_id = 0,
            .impact_gfx_id = osrs_combat_projectile_value_or(
                profile->impact_spotanim_id, 0),
            .start_delay = start_delay + (hop - 1) * duration,
        };
        (void)osrs_emit_projectile_npc_to_npc(
            ov, &spec, source_slot, target_slot);
    }
}

static void col_emit_player_attack_projectiles(
    ColosseumState* s,
    ColosseumContext* ctx,
    EncounterOverlay* ov
) {
    if (!s->tick_scratch.player_attacked ||
            s->player_attack_style_id == ATTACK_STYLE_MELEE) {
        return;
    }

    int target_idx = s->player_attack_npc_idx;
    if (target_idx < 0 || target_idx >= COLO_MAX_NPCS)
        return;
    ColoNPC* target = &s->npcs[target_idx];
    if (!target->active)
        return;

    uint8_t weapon = s->player.equipped[GEAR_SLOT_WEAPON];
    int target_size = col_npc_effective_size(target);
    int p_start_h = 64;
    int p_end_h = (int)(target_size * 0.5f * 128);
    EncounterProjectileTiming timing = s->player_attack_timing;
    if (timing.visual_duration_ticks <= 0) {
        int dist = encounter_projectile_distance(
            s->player.x, s->player.y, 1, target->x, target->y, target_size,
            ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
        timing = col_player_projectile_timing(
            s->player_attack_style_id, weapon, dist);
    }
    int p_duration = timing.visual_duration_ticks * 30;
    int p_start_delay = timing.visual_start_delay_ticks * 30;

    if (s->player_attack_style_id == ATTACK_STYLE_MAGIC) {
        const OsrsCombatProjectileProfile* profile =
            osrs_combat_visual_magic_projectile_profile(weapon);
        if (!profile || profile->projectile_model_id <= 0)
            return;
        OsrsCombatProjectileEmitSpec emit_spec = {
            .src_x = s->player.x,
            .src_y = s->player.y,
            .dst_x = target->x,
            .dst_y = target->y,
            .src_size = 1,
            .dst_size = target_size,
            .target_npc_slot = target_idx,
            .attack_style = s->player_attack_style_id,
            .damage = s->player_attack_dmg,
            .duration_ticks = p_duration,
            .start_delay = p_start_delay,
            .fallback_start_h = p_start_h,
            .fallback_end_h = p_end_h,
            .curve = 16,
        };
        (void)osrs_emit_combat_projectile_profile_player_to_npc(
            ov,
            profile,
            &emit_spec);
        return;
    }

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
    if (part_count < 0)
        abort();
    encounter_require_projectile_slots(ov, part_count);
    for (int part_i = 0; part_i < part_count; part_i++) {
        const OsrsCombatProjectileProfile* profile = &parts[part_i].projectile;
        if (profile->projectile_model_id <= 0)
            continue;
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
            ov,
            profile,
            &emit_spec);
    }
    col_emit_player_venator_bounce_projectiles(
        s, ctx, ov, base_profile, p_start_delay + p_duration, 30);
}

static void col_emit_thrall_projectile(
    ColosseumState* s,
    const ColosseumContext* ctx,
    EncounterOverlay* ov
) {
    int slot = s->thrall_target_slot;
    if (!col_npc_is_thrall_target(s, slot))
        return;
    ColoNPC* target = &s->npcs[slot];
    int target_size = col_npc_effective_size(target);

    int src_x, src_y;
    col_thrall_render_tile(s, ctx, &src_x, &src_y);

    int dist = encounter_projectile_distance(
        src_x, src_y, 1, target->x, target->y, target_size,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    EncounterProjectileTiming timing =
        col_player_projectile_timing(ATTACK_STYLE_MAGIC, 0, dist);

    encounter_require_projectile_slots(ov, 1);
    OsrsProjectileEventSpec spec = {
        .src_x = src_x,
        .src_y = src_y,
        .dst_x = target->x + target_size / 2,
        .dst_y = target->y + target_size / 2,
        .style = encounter_attack_style_to_proj_style(ATTACK_STYLE_MAGIC),
        .damage = 0,
        .duration_ticks = timing.visual_duration_ticks * 30,
        .start_h = 64,
        .end_h = (int)(target_size * 0.5f * 128),
        .curve = 16,
        .arc_height = 1.0f,
        .src_size = 1,
        .dst_size = target_size,
        .model_id = 0,
        .anim_id = -1,
        .travel_gfx_id = 1907,
        .launch_gfx_id = 0,
        .impact_gfx_id = 1908,
        .start_delay = timing.visual_start_delay_ticks * 30,
    };
    int idx = osrs_emit_projectile_with_spec(ov, &spec, 1);
    encounter_set_projectile_target_npc_slot(ov, idx, slot);
}

enum {
    COLO_SOL_CRYSTAL_CHARGE_MODEL_ID = 51245,
    COLO_SOL_CRYSTAL_CHARGE_ANIM_ID  = 10803,
    COLO_SOL_SPHERE_TRAVEL_GFX_ID    = 2693,
    COLO_SOL_SPHERE_IMPACT_GFX_ID    = 2697,
    COLO_SOL_SAND_PILLAR_MODEL_ID    = 51243,
    COLO_SOL_SAND_PILLAR_ANIM_ID     = 10812,
};

enum {
    COLO_VOLATILITY_HUMAN_EXPLOSION_GFX_ID = 2713,
    COLO_VOLATILITY_MANTICORE_EXPLOSION_GFX_ID = 2721,
    COLO_VOLATILITY_COLOSSUS_EXPLOSION_GFX_ID = 2722,
    COLO_VOLATILITY_MINOTAUR_EXPLOSION_GFX_ID = 2723,
    COLO_VOLATILITY_SOL_EXPLOSION_GFX_ID = 2724,
};

static int col_volatility_explosion_gfx_id(ColoNpcType npc_type) {
    switch (npc_type) {
        case COLO_MANTICORE:
            return COLO_VOLATILITY_MANTICORE_EXPLOSION_GFX_ID;
        case COLO_JAVELIN_COLOSSUS:
        case COLO_SHOCKWAVE_COLOSSUS:
            return COLO_VOLATILITY_COLOSSUS_EXPLOSION_GFX_ID;
        case COLO_MINOTAUR:
            return COLO_VOLATILITY_MINOTAUR_EXPLOSION_GFX_ID;
        case COLO_SOL_HEREDIT:
            return COLO_VOLATILITY_SOL_EXPLOSION_GFX_ID;
        default:
            return COLO_VOLATILITY_HUMAN_EXPLOSION_GFX_ID;
    }
}

static void col_emit_volatility_explosions(
    const ColosseumContext* ctx,
    EncounterOverlay* ov
) {
    encounter_require_projectile_slots(ov, ctx->volatility_explosion_count);
    for (int i = 0; i < ctx->volatility_explosion_count; i++) {
        int x = ctx->volatility_explosion_x[i];
        int y = ctx->volatility_explosion_y[i];
        OsrsProjectileEventSpec spec = {
            .src_x = x,
            .src_y = y,
            .dst_x = x,
            .dst_y = y,
            .style = encounter_attack_style_to_proj_style(ATTACK_STYLE_MAGIC),
            .damage = 0,
            .duration_ticks = 1,
            .start_h = 0,
            .end_h = 0,
            .curve = 0,
            .arc_height = 0.0f,
            .src_size = 1,
            .dst_size = 1,
            .model_id = 0,
            .anim_id = -1,
            .travel_gfx_id = 0,
            .launch_gfx_id =
                col_volatility_explosion_gfx_id(ctx->volatility_explosion_type[i]),
            .impact_gfx_id = 0,
            .start_delay = 0,
        };
        (void)osrs_emit_projectile_with_spec(ov, &spec, 0);
    }
}

static void col_emit_sol_hazard_vfx(ColosseumState* s, EncounterOverlay* ov) {
    if (s->wave != COLO_WAVE_BOSS || !s->sol.started) return;
    const SolHereditState* sol = &s->sol;

    for (int c = 0; c < sol->crystal_count; c++) {
        const ColoSolCrystal* cr = &sol->crystals[c];
        if (!cr->active) continue;

        if (cr->firing_freeze > COLO_SOL_LASER_BEAM_SHOW_MAX) {
            encounter_emit_floating_model(
                ov, ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT,
                COLO_NPC_SLOT_BEAM_CRYSTAL + c, cr->x, cr->y,
                COLO_SOL_CRYSTAL_CHARGE_MODEL_ID, COLO_SOL_CRYSTAL_CHARGE_ANIM_ID,
                1.1f, 0.0f, 1.0f);
        }

        if (cr->firing_freeze == COLO_SOL_LASER_BEAM_SHOW_MIN) {
            int dst_x = cr->x, dst_y = cr->y;
            switch (cr->edge) {
                case COLO_SOL_EDGE_NORTH: dst_y = sol->boss_arena_min_y + 1; break;
                case COLO_SOL_EDGE_SOUTH: dst_y = sol->boss_arena_max_y - 1; break;
                case COLO_SOL_EDGE_EAST:  dst_x = sol->boss_arena_min_x + 1; break;
                case COLO_SOL_EDGE_WEST:  dst_x = sol->boss_arena_max_x - 1; break;
                default: break;
            }
            encounter_require_projectile_slots(ov, 1);
            OsrsProjectileEventSpec sphere_spec = {
                .src_x = cr->x,
                .src_y = cr->y,
                .dst_x = dst_x,
                .dst_y = dst_y,
                .style = encounter_attack_style_to_proj_style(ATTACK_STYLE_MAGIC),
                .damage = 0,
                .duration_ticks = COLO_SOL_LASER_BEAM_SHOW_MIN * 30,
                .start_h = 20,
                .end_h = 20,
                .curve = 0,
                .arc_height = 0.0f,
                .src_size = 1,
                .dst_size = 1,
                .model_id = 0u,
                .anim_id = -1,
                .travel_gfx_id = COLO_SOL_SPHERE_TRAVEL_GFX_ID,
                .impact_gfx_id = COLO_SOL_SPHERE_IMPACT_GFX_ID,
                .start_delay = 0,
            };
            (void)osrs_emit_projectile_with_spec(ov, &sphere_spec, 0);
        }
    }

    for (int b = 0; b < COLO_SOL_BEAM_MAX; b++) {
        if (!sol->beams[b].active) continue;
        encounter_emit_floating_model(
            ov, ENCOUNTER_PROJECTILE_TARGET_NPC_SLOT,
            COLO_NPC_SLOT_BEAM_CRYSTAL + COLO_SOL_MAX_CRYSTALS + b,
            sol->beams[b].x, sol->beams[b].y,
            COLO_SOL_SAND_PILLAR_MODEL_ID, COLO_SOL_SAND_PILLAR_ANIM_ID,
            0.05f, 0.0f, 1.0f);
    }
}

static void col_render_post_tick_ctx(
    EncounterState* state,
    EncounterContext* context,
    EncounterOverlay* ov
) {
    ColosseumState* s = (ColosseumState*)state;
    ColosseumContext* ctx = col_require_context(context);
    ov->tile_shadow_count = 0;
    ov->floating_model_count = 0;
    ov->active_modifier_count = 0;
    ov->projectile_count = 0;
    encounter_write_terminal_status_text(
        s->episode_over,
        s->winner,
        COLO_OUTCOME_PLAYER_WON,
        "Colosseum cleared",
        colo_npc_type_name(s->last_hit_by_type),
        ov->status_text,
        sizeof(ov->status_text));
    ov->status_text_active = ov->status_text[0] != '\0';

    for (int mod = 0; mod < COLO_NUM_REAL_MODIFIERS; mod++) {
        int tier = col_mod_tier(s, (ColoModifier)mod);
        if (tier <= 0) continue;
        if (tier > COLO_MODIFIER_MAX_TIER[mod] || tier > 3) {
            fprintf(stderr, "colosseum: active modifier %d has invalid tier %d\n",
                mod, tier);
            abort();
        }
        encounter_emit_active_modifier(ov, mod, tier);
    }

    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        ColoNPC* npc = &s->npcs[i];
        if (!npc->active || npc->type != COLO_JAVELIN_COLOSSUS)
            continue;
        ColoJavelinState* jv = colo_npc_javelin(npc);
        if (!jv->skyfall_pending)
            continue;
        col_emit_javelin_skyfall_shadow(ov, jv);
        if (jv->skyfall_timer == COLO_JAVELIN_SKYFALL_DROP_GAME_TICKS)
            col_emit_javelin_skyfall_drop_projectile(ov, jv);
    }

    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        ColoNPC* npc = &s->npcs[i];
        if (!npc->active || npc->type != COLO_MANTICORE)
            continue;
        col_emit_manticore_orb_telegraphs(ov, npc, i);
    }

    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        ColoNPC* npc = &s->npcs[i];
        if (!npc->active || !npc->attacked_this_tick)
            continue;
        col_emit_npc_attack_projectile(s, ctx, ov, i);
    }
    col_emit_sol_hazard_vfx(s, ov);
    col_emit_volatility_explosions(ctx, ov);
    col_emit_player_attack_projectiles(s, ctx, ov);
    if (ctx->thrall_attacked_this_tick)
        col_emit_thrall_projectile(s, ctx, ov);
}

static void* col_get_player_for_input(void* state, int idx) {
    ColosseumState* s = (ColosseumState*)state;
    return (idx == 0) ? (void*)&s->player : NULL;
}

static int col_find_target_obs_slot(const ColosseumState* s, int npc_slot) {
    return encounter_find_observed_target_slot(
        s->current_obs_slots, COLO_OBS_NPCS, npc_slot);
}

static int col_is_human_targetable_npc_slot_ctx(
    EncounterState* state,
    EncounterContext* context,
    int npc_slot
) {
    ColosseumState* s = (ColosseumState*)state;
    col_require_context(context);
    col_refresh_current_obs_slots(s);
    int obs_slot = col_find_target_obs_slot(s, npc_slot);
    if (obs_slot < 0) return 0;
    return col_obs_slot_is_targetable(s, obs_slot);
}

static int col_queue_inventory_click_action(
    const ColosseumState* s,
    int* actions,
    int inventory_slot
) {
    if (inventory_slot < 0 || inventory_slot >= COLO_INVENTORY_DISPLAY_SLOTS) return -1;
    OsrsInventoryClickResolution r =
        osrs_inventory_cell_click_interpret(
            &s->player.inventory_cells[inventory_slot], OSRS_CLICK_TICK_FIRST);
    int action_head;
    if (r.click_action == OSRS_CLICK_EQUIP) {
        int gear_slot = osrs_inventory_cell_metadata(
            &s->player.inventory_cells[inventory_slot])->gear_slot;
        if (gear_slot < 0 || gear_slot >= NUM_GEAR_SLOTS) return -1;
        action_head = COLO_HEAD_EQUIP_SLOT(gear_slot);
    } else if (r.click_action == OSRS_CLICK_EAT) {
        action_head = COLO_HEAD_EAT;
    } else if (r.click_action == OSRS_CLICK_DRINK) {
        action_head = COLO_HEAD_DRINK;
    } else {
        return -1;
    }
    actions[action_head] = inventory_slot + 1;
    return action_head;
}

static void col_apply_pending_draft_grapple_heads(HumanInput* hi, int* actions) {
    if (hi->pending_modifier_select > 0 &&
            hi->pending_modifier_select <= COLO_MODIFIER_DRAFT_OPTIONS)
        actions[COLO_HEAD_MODIFIER_SELECT] = hi->pending_modifier_select;
    if (hi->pending_grapple_slot > 0 && hi->pending_grapple_slot <= COLO_NUM_GRAPPLE_SLOTS)
        actions[COLO_HEAD_GRAPPLE_PARRY] = hi->pending_grapple_slot;
}

static void col_translate_human_input_ctx(
    HumanInput* hi,
    int* actions,
    EncounterState* state,
    EncounterContext* context
) {
    for (int h = 0; h < COLO_NUM_ACTION_HEADS; h++) actions[h] = 0;

    encounter_translate_movement(hi, actions, COLO_HEAD_PRIMARY, col_get_player_for_input, state);
    encounter_translate_prayer(hi, actions, COLO_HEAD_PRAYER);

    ColosseumState* s = (ColosseumState*)state;
    col_require_context(context);
    col_refresh_current_obs_slots(s);

    if (hi->pending_target_idx >= 0) {
        int found_slot = col_find_target_obs_slot(s, hi->pending_target_idx);
        actions[COLO_HEAD_PRIMARY] =
            (found_slot >= 0 && col_obs_slot_is_targetable(s, found_slot))
                ? col_primary_attack_action_for_obs_slot(found_slot) : 0;
    } else {
        actions[COLO_HEAD_PRIMARY] = 0;
    }

    if (hi->pending_spec) actions[COLO_HEAD_SPEC] = 1;
    encounter_translate_offensive_prayer(hi, actions, COLO_HEAD_OFFENSIVE);

    col_apply_pending_draft_grapple_heads(hi, actions);
}

static void col_translate_human_commands(
    HumanInput* hi,
    int* actions,
    ColosseumState* s
) {
    for (int h = 0; h < COLO_NUM_ACTION_HEADS; h++) actions[h] = 0;
    col_refresh_current_obs_slots(s);

    for (int i = 0; i < hi->commands.count; i++) {
        const HumanCommand* cmd = &hi->commands.items[i];
        switch (cmd->kind) {
            case HUMAN_COMMAND_WALK:
                s->player_dest_x = cmd->world_x;
                s->player_dest_y = cmd->world_y;
                actions[COLO_HEAD_PRIMARY] = 0;
                break;
            case HUMAN_COMMAND_ATTACK_NPC: {
                int found_slot = col_find_target_obs_slot(s, cmd->npc_slot);
                actions[COLO_HEAD_PRIMARY] =
                    (found_slot >= 0 && col_obs_slot_is_targetable(s, found_slot))
                        ? col_primary_attack_action_for_obs_slot(found_slot) : 0;
                s->player_dest_x = -1;
                s->player_dest_y = -1;
                break;
            }
            case HUMAN_COMMAND_OVERHEAD_PRAYER:
                actions[COLO_HEAD_PRAYER] = cmd->overhead_prayer;
                break;
            case HUMAN_COMMAND_EAT:
            case HUMAN_COMMAND_DRINK:
            case HUMAN_COMMAND_EQUIP_INVENTORY_ITEM:
                col_queue_inventory_click_action(s, actions, cmd->inventory_slot);
                break;
            case HUMAN_COMMAND_SPEC_TOGGLE:
                actions[COLO_HEAD_SPEC] = 1;
                break;
            case HUMAN_COMMAND_INVENTORY_PRIMARY_CLICK:
                col_queue_inventory_click_action(s, actions, cmd->inventory_slot);
                break;
            case HUMAN_COMMAND_ITEM_ON_ITEM:
                osrs_inventory_swap_cells(
                    s->player.inventory_cells,
                    cmd->inventory_slot,
                    cmd->target_inventory_slot);
                break;
            case HUMAN_COMMAND_OFFENSIVE_PRAYER:
                actions[COLO_HEAD_OFFENSIVE] = cmd->offensive_prayer;
                break;
            case HUMAN_COMMAND_SPELL_TARGET:
            case HUMAN_COMMAND_FIGHT_STYLE:
            case HUMAN_COMMAND_SET_AUTOCAST:
            case HUMAN_COMMAND_ITEM_ON_WIDGET:
            case HUMAN_COMMAND_SPELL_ON_WIDGET:
            case HUMAN_COMMAND_NONE:
                break;
        }
    }

    col_apply_pending_draft_grapple_heads(hi, actions);
}

static void col_step_human_commands_ctx(
    EncounterState* state,
    EncounterContext* context,
    HumanInput* hi
) {
    ColosseumState* s = (ColosseumState*)state;
    ColosseumContext* ctx = col_require_context(context);
    int actions[COLO_NUM_ACTION_HEADS];
    ctx->human_command_mode = 1;
    ctx->human_commands = hi->commands.items;
    ctx->human_command_count = hi->commands.count;
    col_translate_human_commands(hi, actions, s);
    col_step_ctx(state, (EncounterContext*)ctx, actions);
    ctx->human_commands = NULL;
    ctx->human_command_count = 0;
    human_input_clear_pending(hi);
}

#define COLO_SNAPSHOT_MAGIC 0xC010C001u
#define COLO_SNAPSHOT_VERSION 26u

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t state_size;
    uint32_t reserved;
    ColosseumState state;
} ColoSnapshot;

static void col_refresh_after_state_load(ColosseumState* s, ColosseumContext* ctx) {
    osrs_interaction_zero_serialized_route_padding(&s->player.interaction);
    osrs_interaction_zero_serialized_route_padding(&s->interaction);
    osrs_actor_route_cache_clear(&ctx->player_route_cache);
    col_build_npc_stats();
    col_mark_live_loadout_dirty(s);
    col_ensure_live_loadout_stats(s);
    col_rebuild_player_collision_flags(s);
    col_refresh_current_obs_slots(s);
}

static size_t col_snapshot_size_ctx(EncounterState* state, EncounterContext* context) {
    (void)state;
    (void)context;
    return sizeof(ColoSnapshot);
}

static void col_snapshot_ctx(EncounterState* state, EncounterContext* context, void* out) {
    (void)context;
    ColoSnapshot* snap = (ColoSnapshot*)out;
    encounter_snapshot_write_frame(
        snap, sizeof(*snap), COLO_SNAPSHOT_MAGIC, COLO_SNAPSHOT_VERSION,
        sizeof(ColosseumState));
    encounter_snapshot_copy_state_to(
        snap, offsetof(ColoSnapshot, state), state, sizeof(ColosseumState));
    osrs_interaction_zero_serialized_route_padding(&snap->state.player.interaction);
    osrs_interaction_zero_serialized_route_padding(&snap->state.interaction);
}

static void col_restore_ctx(
    EncounterState* state,
    EncounterContext* context,
    const void* data,
    size_t n
) {
    ColosseumContext* ctx = col_require_context(context);
    ColosseumState* dst = (ColosseumState*)state;
    encounter_snapshot_validate_frame(
        "col_restore", data, n, sizeof(ColoSnapshot),
        COLO_SNAPSHOT_MAGIC, COLO_SNAPSHOT_VERSION, sizeof(ColosseumState));
    encounter_snapshot_copy_state_from(
        dst, data, offsetof(ColoSnapshot, state), sizeof(ColosseumState));
    col_context_clear_render_events(ctx);
    col_refresh_after_state_load(dst, ctx);
}

typedef struct {
    ColosseumState* state;
    ColosseumContext* context;
} ColoLabInvocation;

static int col_lab_apply_script_line_alloc_json_void(
    void* data, const char* line, char** out_json
) {
    ColoLabInvocation* invocation = (ColoLabInvocation*)data;
    return col_lab_apply_script_line_alloc_json(
        invocation->state, invocation->context, line, out_json);
}

static int col_apply_lab_command_ctx(
    EncounterState* state,
    EncounterContext* context,
    const char* line
) {
    ColosseumContext* ctx = col_require_context(context);
    encounter_arena_topology_require_finalized(ctx->route_topology);
    col_context_clear_render_events(ctx);
    ColoLabInvocation invocation = {
        .state = (ColosseumState*)state,
        .context = ctx,
    };
    return encounter_apply_lab_command_dump_wrapper(
        &invocation, line, COLO_LAB_LINE_DUMP,
        col_lab_apply_script_line_alloc_json_void);
}

static const EncounterDef ENCOUNTER_COLOSSEUM = {
    .name = "colosseum",
    .obs_size = COLO_NUM_OBS,
    .num_action_heads = COLO_NUM_ACTION_HEADS,
    .action_head_dims = COLO_ACTION_DIMS,
    .mask_size = COLO_ACTION_MASK_SIZE,

    .state_size = sizeof(ColosseumState),
    .context_size = sizeof(ColosseumContext),
    .init_context = col_init_context,
    .destroy_context = col_destroy_context,
    .init_state = col_init_state,
    .finalize_context = col_finalize_context,

    .create = col_create,
    .destroy = col_destroy,
    .reset = col_reset_ctx,
    .step = col_step_ctx,
    .step_human_commands = col_step_human_commands_ctx,

    .snapshot_size = col_snapshot_size_ctx,
    .snapshot = col_snapshot_ctx,
    .restore = col_restore_ctx,

    .write_obs = col_write_obs_ctx,
    .write_mask = col_write_mask_ctx,
    .get_reward = col_get_reward_ctx,
    .is_terminal = col_is_terminal_ctx,

    .get_entity_count = col_get_entity_count_ctx,
    .get_entity = col_get_entity_ctx,
    .fill_render_entities = col_fill_render_entities_ctx,

    .put_int = col_put_int_ctx,
    .put_float = col_put_float_ctx,
    .put_ptr = col_put_ptr_ctx,

    .arena_base_x = COLO_ARENA_MIN_X,
    .arena_base_y = COLO_ARENA_MIN_Y,
    .arena_width = COLO_ARENA_WIDTH,
    .arena_height = COLO_ARENA_HEIGHT,

    .translate_human_input = col_translate_human_input_ctx,
    .is_human_targetable_npc_slot = col_is_human_targetable_npc_slot_ctx,
    .apply_lab_command = col_apply_lab_command_ctx,
    .head_move = COLO_HEAD_PRIMARY,
    .head_prayer = COLO_HEAD_PRAYER,
    .head_target = COLO_HEAD_PRIMARY,

    .render_post_tick = col_render_post_tick_ctx,
    .get_log = col_get_log_ctx,
    .get_tick = col_get_tick_ctx,
    .get_winner = col_get_winner_ctx,
};

__attribute__((constructor))
static void col_register(void) {
    encounter_register(&ENCOUNTER_COLOSSEUM);
}
