#line 1436 "encounter_inferno.h"
static void inf_sync_consumable_counters_from_cells(InfernoState* s);
static void inf_refresh_live_stats(InfernoState* s);

static InfSupplyDoses inf_full_starting_supplies(void) {
    return (InfSupplyDoses){
        .brew_doses = 24,
        .restore_doses = 36,
        .bastion_doses = 16,
        .stamina_doses = 4,
    };
}


static void inf_seed_potion_cells(
    InfernoState* s,
    int* next_cell,
    OsrsConsumableKind kind,
    int doses
) {
    while (doses > 0) {
        int dose = doses >= 4 ? 4 : doses;
        if (*next_cell >= OSRS_INVENTORY_SIZE) {
            fprintf(stderr, "inferno: inventory kit overflows %d cells\n",
                OSRS_INVENTORY_SIZE);
            abort();
        }
        s->player.inventory_cells[(*next_cell)++] =
            osrs_inventory_cell_from_content_code(
                osrs_inventory_content_code_from_consumable(kind, dose));
        doses -= dose;
    }
}

static void inf_seed_inventory_cells(InfernoState* s) {
    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++)
        s->player.inventory_cells[i] = osrs_inventory_cell_empty();

    int cell = 0;
    const uint8_t* const* loadouts =
        inf_loadouts_for_profile(s->active_loadout_profile);
    for (int g = 0; g < INF_NUM_WEAPON_SETS; g++) {
        for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
            uint8_t item = loadouts[g][slot];
            if (item == ITEM_NONE) continue;
            if (s->player.equipped[slot] == item) continue;
            int duplicate = 0;
            for (int c = 0; c < cell; c++) {
                if (osrs_inventory_cell_item_index(
                        &s->player.inventory_cells[c]) == item) {
                    duplicate = 1;
                    break;
                }
            }
            if (duplicate) continue;
            if (cell >= OSRS_INVENTORY_SIZE) {
                fprintf(stderr, "inferno: gear kit overflows %d cells\n",
                    OSRS_INVENTORY_SIZE);
                abort();
            }
            s->player.inventory_cells[cell++] =
                osrs_inventory_cell_from_item(item);
        }
    }

    inf_seed_potion_cells(s, &cell, OSRS_CONSUMABLE_BREW, s->player.brew_doses);
    inf_seed_potion_cells(s, &cell, OSRS_CONSUMABLE_SUPER_RESTORE,
        s->player.restore_doses);
    inf_seed_potion_cells(s, &cell, OSRS_CONSUMABLE_BASTION, s->player.bastion_doses);
    inf_seed_potion_cells(s, &cell, OSRS_CONSUMABLE_STAMINA, s->player.stamina_doses);
}

static const InfSupplyProfileAnchor INF_SUPPLY_PROFILE_ANCHORS[] = {
    { 1,  { 1.0000f, 1.0000f, 1.0000f, 1.0000f } },
    { 20, { 1.0000f, 0.9500f, 1.0000f, 1.0000f } },
    { 40, { 0.9167f, 0.8750f, 1.0000f, 1.0000f } },
    { 61, { 0.8333f, 0.7500f, 1.0000f, 1.0000f } },
    { 64, { 0.5833f, 0.5000f, 0.7500f, 1.0000f } },
    { 68, { 0.5833f, 0.4250f, 0.6250f, 1.0000f } },
    { 69, { 0.5000f, 0.3000f, 0.3750f, 1.0000f } },
    { 70, { 0.4375f, 0.2250f, 0.3750f, 1.0000f } },
    { 71, { 0.3750f, 0.1500f, 0.3750f, 1.0000f } },
};

static float inf_lerp_float(float a, float b, float t) {
    return a + (b - a) * t;
}

static void inf_require_unit_interval_float_config(const char* key, float value) {
    if (value < 0.0f || value > 1.0f) {
        fprintf(stderr, "inferno %s must be in [0, 1], got %.6f\n", key, value);
        abort();
    }
}

static void inf_require_valid_supply_scale(float scale) {
    inf_require_unit_interval_float_config("late_start_supply_profile_scale", scale);
}

static void inf_require_valid_curriculum_supply_mode(const char* key, int mode) {
    if (mode < INF_CURRICULUM_SUPPLY_MODE_OFF ||
            mode > INF_CURRICULUM_SUPPLY_MODE_PRE_ZUK) {
        fprintf(stderr, "inferno %s must be in [0, 3], got %d\n", key, mode);
        abort();
    }
}

static void inf_require_valid_budget_loadout_fraction(float fraction) {
    inf_require_unit_interval_float_config("budget_loadout_fraction", fraction);
}

static void inf_require_valid_loadout_profile_mode(int mode) {
    if (mode < INF_LOADOUT_PROFILE_MODE_MAX_ONLY ||
            mode > INF_LOADOUT_PROFILE_MODE_MIXED_MAX_BUDGET) {
        fprintf(stderr, "inferno loadout_profile_mode must be in [0, 2], got %d\n",
            mode);
        abort();
    }
}

static InfLoadoutProfile inf_sample_loadout_profile(
    InfLoadoutProfileMode mode,
    float budget_fraction,
    uint32_t* rng_state
) {
    inf_require_valid_loadout_profile_mode(mode);
    inf_require_valid_budget_loadout_fraction(budget_fraction);

    if (mode == INF_LOADOUT_PROFILE_MODE_MAX_ONLY)
        return INF_LOADOUT_PROFILE_MAX;
    if (mode == INF_LOADOUT_PROFILE_MODE_BUDGET_ONLY)
        return INF_LOADOUT_PROFILE_BUDGET;
    return encounter_rand_float(rng_state) < budget_fraction
        ? INF_LOADOUT_PROFILE_BUDGET
        : INF_LOADOUT_PROFILE_MAX;
}

static void inf_require_nonnegative_float_config(const char* key, float value) {
    if (!(value >= 0.0f)) {
        fprintf(stderr, "inferno %s must be >= 0, got %.6f\n", key, value);
        abort();
    }
}

static void inf_require_valid_public_wave(int public_wave) {
    if (public_wave < 1 || public_wave > INF_MAX_PUBLIC_START_WAVE) {
        fprintf(stderr, "inferno start_wave must be in [1, %d], got %d\n",
            INF_MAX_PUBLIC_START_WAVE, public_wave);
        abort();
    }
}

static int inf_runtime_wave_from_start_wave(int internal_start_wave) {
    return internal_start_wave >= INF_WAVE_ZUK ? INF_WAVE_ZUK : internal_start_wave;
}

static int inf_initial_attack_timer_after_stun(int attack_speed, int stun_timer) {
    return attack_speed + stun_timer;
}

static InfSupplyFractions inf_supply_profile_fractions(int public_wave) {
    inf_require_valid_public_wave(public_wave);

    const int n = (int)(sizeof(INF_SUPPLY_PROFILE_ANCHORS) /
        sizeof(INF_SUPPLY_PROFILE_ANCHORS[0]));
    for (int i = 1; i < n; i++) {
        const InfSupplyProfileAnchor* lo = &INF_SUPPLY_PROFILE_ANCHORS[i - 1];
        const InfSupplyProfileAnchor* hi = &INF_SUPPLY_PROFILE_ANCHORS[i];
        if (public_wave <= hi->public_wave) {
            float t = (float)(public_wave - lo->public_wave) /
                (float)(hi->public_wave - lo->public_wave);
            return (InfSupplyFractions){
                .brew_fraction = inf_lerp_float(lo->fractions.brew_fraction,
                    hi->fractions.brew_fraction, t),
                .restore_fraction = inf_lerp_float(lo->fractions.restore_fraction,
                    hi->fractions.restore_fraction, t),
                .bastion_fraction = inf_lerp_float(lo->fractions.bastion_fraction,
                    hi->fractions.bastion_fraction, t),
                .stamina_fraction = inf_lerp_float(lo->fractions.stamina_fraction,
                    hi->fractions.stamina_fraction, t),
            };
        }
    }

    fprintf(stderr, "inferno supply profile has no anchor for wave %d\n", public_wave);
    abort();
}

static int inf_supply_profile_anchor_index(int public_wave) {
    int n = (int)(sizeof(INF_SUPPLY_PROFILE_ANCHORS) /
        sizeof(INF_SUPPLY_PROFILE_ANCHORS[0]));
    for (int i = 0; i < n; i++) {
        if (INF_SUPPLY_PROFILE_ANCHORS[i].public_wave == public_wave)
            return i;
    }
    return -1;
}

static int inf_profiled_supply_count(int full_doses, float profile_fraction, float scale) {
    assert(full_doses >= 0);

    float effective_fraction = 1.0f - scale * (1.0f - profile_fraction);
    int doses = (int)((float)full_doses * effective_fraction + 0.5f);
    if (doses < 0) doses = 0;
    if (doses > full_doses) doses = full_doses;
    return doses;
}

static int inf_clamp_supply_doses(int doses, int full_doses) {
    if (doses < 0) return 0;
    if (doses > full_doses) return full_doses;
    return doses;
}

static int inf_round_signed_float_to_int(float value) {
    return value >= 0.0f ? (int)(value + 0.5f) : (int)(value - 0.5f);
}

static int inf_curriculum_supply_mode_applies(int mode, int public_wave) {
    inf_require_valid_curriculum_supply_mode("curriculum supply mode", mode);
    switch (mode) {
        case INF_CURRICULUM_SUPPLY_MODE_OFF:
            return 0;
        case INF_CURRICULUM_SUPPLY_MODE_ALL:
            return 1;
        case INF_CURRICULUM_SUPPLY_MODE_ZUK:
            return public_wave >= INF_NUM_WAVES;
        case INF_CURRICULUM_SUPPLY_MODE_PRE_ZUK:
            return public_wave < INF_NUM_WAVES;
        default:
            abort();
    }
}

static void inf_apply_curriculum_supply_variation(
    InfernoState* s,
    const InfernoContext* ctx,
    InfSupplyDoses full
) {
    if (!ctx->config.curriculum_agent) return;

    int public_wave = s->start_wave + 1;
    int jitter_applies = inf_curriculum_supply_mode_applies(
        ctx->config.curriculum_supply_jitter_mode, public_wave);
    float shared_jitter = ctx->config.curriculum_supply_shared_jitter;
    float brew_jitter = ctx->config.curriculum_supply_brew_jitter;
    float restore_jitter = ctx->config.curriculum_supply_restore_jitter;

    if (jitter_applies &&
            (shared_jitter > 0.0f || brew_jitter > 0.0f || restore_jitter > 0.0f)) {
        float shared = (2.0f * encounter_rand_float(&s->rng_state) - 1.0f) *
            shared_jitter;
        float brew_delta = shared;
        float restore_delta = shared;
        if (brew_jitter > 0.0f) {
            brew_delta += (2.0f * encounter_rand_float(&s->rng_state) - 1.0f) *
                brew_jitter;
        }
        if (restore_jitter > 0.0f) {
            restore_delta += (2.0f * encounter_rand_float(&s->rng_state) - 1.0f) *
                restore_jitter;
        }

        s->player.brew_doses = inf_clamp_supply_doses(
            s->player.brew_doses +
                inf_round_signed_float_to_int((float)full.brew_doses * brew_delta),
            full.brew_doses);
        s->player.restore_doses = inf_clamp_supply_doses(
            s->player.restore_doses +
                inf_round_signed_float_to_int((float)full.restore_doses * restore_delta),
            full.restore_doses);
    }

    int no_brew_applies = inf_curriculum_supply_mode_applies(
        ctx->config.curriculum_no_brew_mode, public_wave);
    if (no_brew_applies &&
            ctx->config.curriculum_no_brew_frac > 0.0f &&
            encounter_rand_float(&s->rng_state) < ctx->config.curriculum_no_brew_frac) {
        s->player.brew_doses = 0;
    }
}

static InfSupplyDoses inf_supplies_for_start_wave(InfSupplyDoses full,
                                                  int internal_start_wave,
                                                  float scale) {
    inf_require_valid_supply_scale(scale);

    int public_wave = internal_start_wave + 1;

    InfSupplyFractions fractions = inf_supply_profile_fractions(public_wave);
    return (InfSupplyDoses){
        .brew_doses = inf_profiled_supply_count(full.brew_doses,
            fractions.brew_fraction, scale),
        .restore_doses = inf_profiled_supply_count(full.restore_doses,
            fractions.restore_fraction, scale),
        .bastion_doses = inf_profiled_supply_count(full.bastion_doses,
            fractions.bastion_fraction, scale),
        .stamina_doses = inf_profiled_supply_count(full.stamina_doses,
            fractions.stamina_fraction, scale),
    };
}

static float inf_supply_milestone_surplus_reward(
    InfernoState* s,
    const InfernoContext* ctx,
    int public_wave
) {
    if (ctx->config.supply_milestone_brew_reward_coeff <= 0.0f &&
            ctx->config.supply_milestone_restore_reward_coeff <= 0.0f) {
        return 0.0f;
    }

    int anchor_idx = inf_supply_profile_anchor_index(public_wave);
    if (anchor_idx <= 0) return 0.0f;

    uint32_t anchor_bit = 1u << (uint32_t)anchor_idx;
    if ((s->supply_milestone_rewarded_mask & anchor_bit) != 0)
        return 0.0f;
    s->supply_milestone_rewarded_mask |= anchor_bit;

    InfSupplyDoses full = inf_full_starting_supplies();
    InfSupplyDoses expected = inf_supplies_for_start_wave(
        full, public_wave - 1, ctx->config.late_start_supply_profile_scale);

    int brew_surplus = s->player.brew_doses - expected.brew_doses;
    int restore_surplus = s->player.restore_doses - expected.restore_doses;
    if (brew_surplus < 0) brew_surplus = 0;
    if (restore_surplus < 0) restore_surplus = 0;

    return
        ctx->config.supply_milestone_brew_reward_coeff *
            (float)brew_surplus / (float)full.brew_doses +
        ctx->config.supply_milestone_restore_reward_coeff *
            (float)restore_surplus / (float)full.restore_doses;
}

static void inf_reset_ctx(EncounterState* state, EncounterContext* context, uint32_t seed) {
    inf_build_npc_stats();
    InfernoState* s = (InfernoState*)state;
    InfernoContext* ctx = (InfernoContext*)context;
    encounter_arena_topology_require_finalized(ctx->route_topology);
    osrs_actor_route_cache_clear(&ctx->player_route_cache);
    inf_reset_npc_player_los_frame(ctx);
    int saved_start = ctx->config.start_wave;
    uint32_t saved_rng = s->rng_state;
    memset(s, 0, sizeof(InfernoState));
    s->start_wave = saved_start;
    s->rng_state = encounter_resolve_seed(saved_rng, seed);

    s->player_dest_x = -1;
    s->player_dest_y = -1;
    s->player_last_interaction_target_slot = -1;
    s->player_last_interaction_age = 1;

    s->tick_at_le_300 = -1;
    s->tick_at_le_240 = -1;
    s->tick_at_le_150 = -1;
    s->tick_at_zuk_healer_spawn = -1;
    s->tick_at_first_zuk_healer_tag = -1;
    s->tick_at_all_zuk_healers_tagged = -1;
    s->tick_at_all_zuk_healers_dead = -1;
    s->tick_at_first_zuk_hit_after_all_healers_dead = -1;
    s->tick_at_first_zuk_healer_target = -1;
    s->tick_at_first_zuk_healer_attack = -1;
    s->total_zuk_healer_target_ticks = 0;
    s->total_zuk_healer_attack_fires = 0;
    s->total_zuk_healer_cannot_attack_ticks = 0;
    s->total_zuk_healer_cooldown_ticks = 0;
    s->total_zuk_healer_out_of_range_ticks = 0;
    s->total_zuk_healer_attackable_ticks = 0;
    s->total_action_mask_checks = 0;

    s->player.entity_type = ENTITY_PLAYER;
    encounter_init_maxed_player_combat_stats(&s->player, 99);
    osrs_item_effect_state_init(&s->player.item_effect_state);
    s->weapon_set = INF_GEAR_MAGE;
    s->active_loadout_profile = inf_sample_loadout_profile(
        ctx->config.loadout_profile_mode,
        ctx->config.budget_loadout_fraction,
        &s->rng_state);
    const uint8_t* const* loadouts =
        inf_loadouts_for_profile(s->active_loadout_profile);
    encounter_apply_loadout(&s->player, loadouts[INF_GEAR_MAGE], GEAR_MAGE);
    InfSupplyDoses full_supplies = inf_full_starting_supplies();
    InfSupplyDoses start_supplies = inf_supplies_for_start_wave(
        full_supplies, s->start_wave, ctx->config.late_start_supply_profile_scale);
    s->player.brew_doses = start_supplies.brew_doses;
    s->player.restore_doses = start_supplies.restore_doses;
    s->player.bastion_doses = start_supplies.bastion_doses;
    s->player.stamina_doses = start_supplies.stamina_doses;
    inf_apply_curriculum_supply_variation(s, ctx, full_supplies);
    inf_seed_inventory_cells(s);
    inf_sync_consumable_counters_from_cells(s);
    s->stamina_active_ticks = 0;
    s->player.prayer = PRAYER_NONE;
    s->player.autocast_enabled = 1;
    s->player.autocast_defensive = 0;
    s->player.autocast_spell = ENCOUNTER_SPELL_BLOOD;
    osrs_interaction_init(&s->interaction);
    s->player.spec_armed = 0;
    s->player.special_energy = 100;
    s->player.run_energy = 10000;
    s->last_hit_by_type = -1;

    encounter_compute_loadout_stats(loadouts[INF_GEAR_MAGE], ATTACK_STYLE_MAGIC,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_AUTOCAST,
        inf_player_magic_base_damage(&s->player), &s->loadout_stats[INF_GEAR_MAGE]);
    encounter_compute_loadout_stats(loadouts[INF_GEAR_LONG_RANGE], ATTACK_STYLE_RANGED,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_RAPID, 0, &s->loadout_stats[INF_GEAR_LONG_RANGE]);
    encounter_compute_loadout_stats(loadouts[INF_GEAR_BP], ATTACK_STYLE_RANGED,
        OFFENSIVE_PRAYER_NONE, 99, FIGHT_STYLE_RAPID, 0, &s->loadout_stats[INF_GEAR_BP]);
    inf_refresh_live_stats(s);

    int effective_start = inf_runtime_wave_from_start_wave(s->start_wave);
    int is_zuk_wave = (effective_start >= INF_WAVE_ZUK);
    s->player.x = is_zuk_wave ? INF_ZUK_PLAYER_START_X : INF_PLAYER_START_X;
    s->player.y = is_zuk_wave ? INF_ZUK_PLAYER_START_Y : INF_PLAYER_START_Y;

    for (int i = 0; i < INF_NUM_PILLARS; i++) {
        s->pillars[i].x = INF_PILLAR_POS[i][0];
        s->pillars[i].y = INF_PILLAR_POS[i][1];
        if (effective_start >= 66) {
            s->pillars[i].hp = 0;
            s->pillars[i].active = 0;
        } else {
            s->pillars[i].hp = INF_PILLAR_HP;
            s->pillars[i].active = 1;
        }
    }

    s->dead_mob_count = 0;

    s->wave = effective_start;
    s->wave_spawn_target = effective_start;
    s->wave_spawn_delay = 0;
    s->wave_ready_delay = INF_START_READY_TICKS;
    inf_spawn_wave(s);
}


static int inf_find_free_npc(InfernoState* s) {
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (!s->npcs[i].active) return i;
    }
    return -1;
}
static int inf_grid_index(int x, int y, int* grid_x, int* grid_y) {
    *grid_x = x - INF_ARENA_MIN_X;
    *grid_y = y - INF_ARENA_MIN_Y;
    return *grid_x >= 0 && *grid_x < INF_ARENA_WIDTH &&
        *grid_y >= 0 && *grid_y < INF_ARENA_HEIGHT;
}



static int inf_npc_sets_collision_flag(InfNPCType type) {
    return type != INF_NPC_NIBBLER && type != INF_NPC_ZUK_SHIELD;
}

static int inf_npc_effective_size(const InfNPC* npc) {
    return npc->size > 0 ? npc->size : INF_NPC_STATS[npc->type].size;
}
static void inf_unstamp_npc_collision_footprint(
    InfernoState* s,
    int x,
    int y,
    int size
) {
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int grid_x;
            int grid_y;
            if (!inf_grid_index(
                    x + dx, y + dy, &grid_x, &grid_y))
                continue;
            if (s->npc_collision_flags[grid_x][grid_y] > 0)
                s->npc_collision_flags[grid_x][grid_y]--;
        }
    }
}

static void inf_stamp_npc_collision_footprint(
    InfernoState* s,
    int x,
    int y,
    int size
) {
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int grid_x;
            int grid_y;
            if (!inf_grid_index(
                    x + dx, y + dy, &grid_x, &grid_y))
                continue;
            assert(s->npc_collision_flags[grid_x][grid_y] < UINT8_MAX);
            s->npc_collision_flags[grid_x][grid_y]++;
        }
    }
}

static void inf_rebuild_npc_collision_flags(InfernoState* s) {
    memset(s->npc_collision_flags, 0, sizeof(s->npc_collision_flags));
    for (int npc_idx = 0; npc_idx < INF_MAX_NPCS; npc_idx++) {
        InfNPC* npc = &s->npcs[npc_idx];
        if (!npc->active ||
                !inf_npc_sets_collision_flag(npc->type))
            continue;
        inf_stamp_npc_collision_footprint(
            s, npc->x, npc->y, inf_npc_effective_size(npc));
    }
}



static void inf_deactivate_npc(InfernoState* s, int idx) {
    if (idx < 0 || idx >= INF_MAX_NPCS) return;
    InfNPC* npc = &s->npcs[idx];
    if (npc->active && inf_npc_sets_collision_flag(npc->type))
        inf_unstamp_npc_collision_footprint(
            s, npc->x, npc->y, inf_npc_effective_size(npc));
    encounter_pending_hit_queue_clear(&npc->pending_hits);
    npc->active = 0;
}

static void inf_init_npc(InfernoState* s, int idx, InfNPCType type, int x, int y) {
    InfNPC* npc = &s->npcs[idx];
    const InfNPCStats* stats = &INF_NPC_STATS[type];
    memset(npc, 0, sizeof(InfNPC));

    s->next_npc_render_id++;
    if (s->next_npc_render_id == 0)
        s->next_npc_render_id = 1;

    npc->type = type;
    npc->hp = stats->hp;
    npc->max_hp = stats->hp;
    npc->size = stats->size;
    npc->render_id = s->next_npc_render_id;
    npc->attack_timer = stats->attack_speed;
    npc->attack_style = stats->default_style;
    npc->attack_style_this_tick = ATTACK_STYLE_NONE;
    npc->active = 1;
    npc->x = x;
    npc->y = y;
    npc->target_x = x;
    npc->target_y = y;
    npc->attack_visual_target = -1;
    npc->resurrection_visual_target = -1;
    npc->aggro_target = -1;
    npc->blob_scanned_prayer = -1;
    npc->had_los_last_tick = 0;
    npc->stun_timer = stats->stun_on_spawn;
    inf_npc_init_type_state(npc);
    if (inf_npc_sets_collision_flag(type))
        inf_stamp_npc_collision_footprint(s, x, y, stats->size);

}

static int inf_find_first_active_npc_of_type(const InfernoState* s, InfNPCType type) {
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (s->npcs[i].active && s->npcs[i].type == type) return i;
    }
    return -1;
}

static void inf_spawn_zuk_jad(InfernoState* s, int shield_idx) {
    int j_slot = inf_find_free_npc(s);
    if (j_slot >= 0) {
        inf_init_npc(s, j_slot, INF_NPC_JAD, 24, 32);
        s->npcs[j_slot].aggro_target = shield_idx;
        s->npcs[j_slot].stun_timer = 7;
    }
}

static void inf_spawn_zuk_healers(InfernoState* s, int zuk_idx) {
    static const int healer_pos[4][2] = {
        {16, 48}, {20, 48}, {30, 48}, {34, 48}
    };
    for (int h = 0; h < 4; h++) {
        int slot = inf_find_free_npc(s);
        if (slot >= 0) {
            inf_init_npc(s, slot, INF_NPC_HEALER_ZUK,
                healer_pos[h][0], healer_pos[h][1]);
            s->npcs[slot].aggro_target = zuk_idx;
        }
    }
}

static void inf_seed_joseph_zuk_checkpoint(InfernoState* s) {
    if (s->start_wave < INF_START_WAVE_ZUK_JAD) return;

    int zuk_idx = inf_find_first_active_npc_of_type(s, INF_NPC_ZUK);
    int shield_idx = s->zuk.shield_idx;
    if (zuk_idx < 0 || shield_idx < 0 || shield_idx >= INF_MAX_NPCS ||
            !s->npcs[shield_idx].active)
        return;

    InfNPC* zuk = &s->npcs[zuk_idx];
    zuk->stun_timer = 0;
    s->zuk.has_paused = 1;
    s->zuk.timer_paused = 0;
    s->zuk.jad_spawned = 1;
    s->zuk.shield_freeze = 0;

    if (s->start_wave == INF_START_WAVE_ZUK_JAD) {
        s->zuk.set_timer = 247;
        s->zuk.enraged = 0;
        s->zuk.healer_spawned = 0;
        zuk->hp = 479;
        zuk->attack_timer = INF_NPC_STATS[INF_NPC_ZUK].attack_speed;
        s->min_zuk_hp_seen = (float)zuk->hp;
        inf_spawn_zuk_jad(s, shield_idx);
        return;
    }

    s->zuk.set_timer = s->zuk.set_interval;
    s->zuk.enraged = 1;
    s->zuk.healer_spawned = 1;
    zuk->hp = 239;
    zuk->attack_timer = 7;
    s->min_zuk_hp_seen = (float)zuk->hp;
    inf_spawn_zuk_healers(s, zuk_idx);
}

static void inf_spawn_wave(InfernoState* s) {
    if (s->wave >= INF_NUM_WAVES) return;

    const InfWaveDef* w = &INF_WAVES[s->wave];

    for (int i = 0; i < INF_MAX_NPCS; i++) s->npcs[i].active = 0;
    memset(s->npc_collision_flags, 0, sizeof(s->npc_collision_flags));
    encounter_pending_hit_queue_clear(&s->player_pending_hits);

    s->dead_mob_count = 0;

    inf_shuffle_spawns(s);

    {
        int active_pillars[INF_NUM_PILLARS];
        int num_active = 0;
        for (int p = 0; p < INF_NUM_PILLARS; p++) {
            if (s->pillars[p].active) active_pillars[num_active++] = p;
        }
        s->nibbler_target_pillar = (num_active > 0)
            ? active_pillars[encounter_rand_int(&s->rng_state, num_active)] : -1;
    }

    if (s->wave >= 66) {
        for (int p = 0; p < INF_NUM_PILLARS; p++) {
            if (s->pillars[p].active) {
                s->pillars[p].active = 0;
                s->pillars[p].hp = 0;
            }
        }
    }

    if (s->wave == 66) {
        s->player.x = 18;
        s->player.y = 32;
        int slot = inf_find_free_npc(s);
        if (slot >= 0) {
            inf_init_npc(s, slot, INF_NPC_JAD, 23, 30);
            s->npcs[slot].stun_timer = 1;
            s->npcs[slot].attack_timer = 8;
        }
        return;
    }

    if (s->wave == 67) {
        s->player.x = 25;
        s->player.y = 30;
        int stuns[3] = { 1, 4, 7 };
        for (int i = 2; i > 0; i--) {
            int j = encounter_rand_int(&s->rng_state, i + 1);
            int tmp = stuns[i]; stuns[i] = stuns[j]; stuns[j] = tmp;
        }
        static const int JAD_POS[3][2] = {
            {18, 33},
            {28, 33},
            {23, 22},
        };
        for (int i = 0; i < 3; i++) {
            int slot = inf_find_free_npc(s);
            if (slot < 0) break;
            inf_init_npc(s, slot, INF_NPC_JAD, JAD_POS[i][0], JAD_POS[i][1]);
            s->npcs[slot].stun_timer = stuns[i];
            s->npcs[slot].attack_timer =
                inf_initial_attack_timer_after_stun(9, stuns[i]);
        }
        return;
    }

    if (s->wave == 68) {
        int zuk_idx = inf_find_free_npc(s);
        if (zuk_idx >= 0) {
            inf_init_npc(s, zuk_idx, INF_NPC_ZUK, INF_ZUK_X, INF_ZUK_Y);
            s->min_zuk_hp_seen = (float)s->npcs[zuk_idx].hp;
            s->npcs[zuk_idx].stun_timer = 8;
            s->npcs[zuk_idx].attack_timer = 14;
        }

        int shield_idx = inf_find_free_npc(s);
        if (shield_idx >= 0) {
            inf_init_npc(s, shield_idx, INF_NPC_ZUK_SHIELD, INF_ZUK_SHIELD_X, INF_ZUK_SHIELD_Y);
            s->zuk.shield_idx = shield_idx;
            s->zuk.shield_dir = (encounter_rand_int(&s->rng_state, 2) == 0) ? 1 : -1;
            s->zuk.shield_freeze = 1;
        }

        s->zuk.set_timer = 72;
        s->zuk.set_interval = 350;
        s->zuk.enraged = 0;
        s->zuk.healer_spawned = 0;
        s->zuk.jad_spawned = 0;
        s->zuk.timer_paused = 0;
        s->zuk.has_paused = 0;

        s->player.x = INF_ZUK_PLAYER_START_X;
        s->player.y = INF_ZUK_PLAYER_START_Y;
        inf_seed_joseph_zuk_checkpoint(s);
        return;
    }

    int spawn_idx = 0;
    for (int i = 0; i < w->count && i < INF_MAX_NPCS; i++) {
        InfNPCType type = (InfNPCType)w->types[i];
        int slot = inf_find_free_npc(s);
        if (slot < 0) break;

        int sx, sy;
        if (type == INF_NPC_NIBBLER) {
            sx = INF_NIBBLER_SPAWN_X + encounter_rand_int(&s->rng_state, 3) - 1;
            sy = INF_NIBBLER_SPAWN_Y + encounter_rand_int(&s->rng_state, 3) - 1;
        } else {
            int pi = s->spawn_order[spawn_idx % INF_NUM_SPAWN_POS];
            sx = INF_SPAWN_POS[pi][0];
            sy = INF_SPAWN_POS[pi][1];
            spawn_idx++;
        }

        inf_init_npc(s, slot, type, sx, sy);
    }
}
