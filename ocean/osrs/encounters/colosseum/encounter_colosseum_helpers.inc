#line 76 "encounter_colosseum.h"

static void col_accumulate_tick_stats(ColosseumState* s, const ColosseumContext* ctx);
static void col_spawn_wave(ColosseumState* s);
static void col_tick_npcs_ctx(ColosseumState* s, ColosseumContext* ctx);
static void col_tick_player_ctx(
    ColosseumState* s, ColosseumContext* ctx, const int* actions, int can_attack);
static void col_player_pretick(ColosseumState* s, ColosseumContext* ctx, const int* actions);
static void col_resolve_player_projectiles_on_npcs_ctx(
    ColosseumState* s, ColosseumContext* ctx);
static void col_resolve_player_pending_hits_ctx(ColosseumState* s, ColosseumContext* ctx);
static void col_resolve_player_pending_hits(ColosseumState* s);
static void col_rebuild_player_collision_flags(ColosseumState* s);
static void col_refresh_current_obs_slots_ctx(ColosseumState* s, const ColosseumContext* ctx);

static void col_sol_boss_tick(ColosseumState* s);
static void col_sol_begin_boss_arena(ColosseumState* s);
static int col_sol_clamp_active(const ColosseumState* s);
static int col_in_boss_arena(const ColosseumState* s, int x, int y);

static inline int col_mod_active(const ColosseumState* s, ColoModifier mod);
static inline int col_mod_tier(const ColosseumState* s, ColoModifier mod);
static void col_modifiers_on_wave_spawn(ColosseumState* s);
static void col_modifier_open_draft(ColosseumState* s, int next_wave);
static void col_apply_late_start_entry_state(
    ColosseumState* s, const ColosseumContext* ctx, int start_wave);

static void col_abort_invalid_weapon_set(ColoWeaponSet set) {
    fprintf(stderr, "invalid Colosseum weapon set %d\n", (int)set);
    abort();
}

static AttackStyle col_weapon_set_attack_style(ColoWeaponSet set) {
    switch (set) {
        case COLO_GEAR_MELEE:
            return ATTACK_STYLE_MELEE;
        case COLO_GEAR_RANGED:
            return ATTACK_STYLE_RANGED;
        case COLO_GEAR_MAGIC:
            return ATTACK_STYLE_MAGIC;
        case COLO_NUM_WEAPON_SETS:
            break;
    }
    col_abort_invalid_weapon_set(set);
    return ATTACK_STYLE_MELEE;
}

static FightStyle col_weapon_set_fight_style(ColoWeaponSet set) {
    switch (set) {
        case COLO_GEAR_MELEE:
            return FIGHT_STYLE_AGGRESSIVE;
        case COLO_GEAR_RANGED:
            return FIGHT_STYLE_RAPID;
        case COLO_GEAR_MAGIC:
            return FIGHT_STYLE_ACCURATE;
        case COLO_NUM_WEAPON_SETS:
            break;
    }
    col_abort_invalid_weapon_set(set);
    return FIGHT_STYLE_AGGRESSIVE;
}

static int col_weapon_set_spell_base_damage(const ColosseumState* s, ColoWeaponSet set) {
    if (set != COLO_GEAR_MAGIC) return 0;
    if (s->active_loadout_profile < 0 ||
            s->active_loadout_profile >= COLO_NUM_LOADOUT_PROFILES) {
        fprintf(stderr, "invalid Colosseum loadout profile %d\n",
            (int)s->active_loadout_profile);
        abort();
    }
    return COLO_MAGIC_SPELL_BASE_DAMAGE[s->active_loadout_profile];
}

static AttackStyle col_equipped_weapon_attack_style(const Player* p) {
    int style = get_item_attack_style(p->equipped[GEAR_SLOT_WEAPON]);
    if (style == 1) return ATTACK_STYLE_MELEE;
    if (style == 2) return ATTACK_STYLE_RANGED;
    if (style == 3) return ATTACK_STYLE_MAGIC;
    return ATTACK_STYLE_MELEE;
}

static FightStyle col_equipped_weapon_fight_style(const Player* p) {
    AttackStyle style = col_equipped_weapon_attack_style(p);
    if (style == ATTACK_STYLE_RANGED)
        return FIGHT_STYLE_RAPID;
    if (style == ATTACK_STYLE_MAGIC)
        return FIGHT_STYLE_ACCURATE;
    return FIGHT_STYLE_AGGRESSIVE;
}

static GearSet col_gear_set_for_attack_style(AttackStyle style) {
    if (style == ATTACK_STYLE_RANGED) return GEAR_RANGED;
    if (style == ATTACK_STYLE_MAGIC) return GEAR_MAGE;
    return GEAR_MELEE;
}

static int col_equipped_spell_base_damage(const ColosseumState* s) {
    if (col_equipped_weapon_attack_style(&s->player) != ATTACK_STYLE_MAGIC) return 0;
    return COLO_MAGIC_SPELL_BASE_DAMAGE[s->active_loadout_profile];
}

static void col_current_style_levels(
    const Player* p,
    AttackStyle style,
    int* current_att_level,
    int* current_str_level
) {
    if (style == ATTACK_STYLE_RANGED) {
        *current_att_level = p->current_ranged;
        *current_str_level = p->current_ranged;
        return;
    }
    if (style == ATTACK_STYLE_MAGIC) {
        *current_att_level = p->current_magic;
        *current_str_level = p->current_magic;
        return;
    }
    *current_att_level = p->current_attack;
    *current_str_level = p->current_strength;
}

static void col_mark_live_loadout_dirty(ColosseumState* s) {
    s->loadout_stats_dirty = 1;
}

static void col_recompute_live_loadout_stats(ColosseumState* s) {
    AttackStyle style = col_equipped_weapon_attack_style(&s->player);
    FightStyle fight_style = col_equipped_weapon_fight_style(&s->player);
    int current_att_level;
    int current_str_level;
    col_current_style_levels(&s->player, style, &current_att_level, &current_str_level);
    s->player.current_gear = col_gear_set_for_attack_style(style);
    s->player.visible_gear = s->player.current_gear;
    encounter_compute_loadout_stats(
        s->player.equipped,
        style,
        s->player.offensive_prayer,
        current_att_level,
        fight_style,
        col_equipped_spell_base_damage(s),
        &s->loadout_stats_live);
    encounter_update_loadout_level(
        &s->loadout_stats_live,
        s->player.offensive_prayer,
        current_att_level,
        current_str_level);
    encounter_derive_loadout_effect_profile(s->player.equipped, &s->effects_live);
    s->loadout_stats_dirty = 0;
}

static void col_ensure_live_loadout_stats(ColosseumState* s) {
    if (s->loadout_stats_dirty)
        col_recompute_live_loadout_stats(s);
}

static const EncounterLoadoutStats* col_live_loadout_stats(ColosseumState* s) {
    col_ensure_live_loadout_stats(s);
    return &s->loadout_stats_live;
}

static const OsrsEquipmentEffectProfile* col_live_effects(ColosseumState* s) {
    col_ensure_live_loadout_stats(s);
    return &s->effects_live;
}

static inline EncounterProjectileTiming col_player_projectile_timing(
    AttackStyle style,
    uint8_t weapon,
    int distance
) {
    EncounterProjectileDelayKind kind = encounter_projectile_delay_kind_for_style(style);
    EncounterProjectileDelayOptions options = {0};
    if (style == ATTACK_STYLE_RANGED && weapon == ITEM_TWISTED_BOW)
        options.visual_delay_ticks = 1;
    return encounter_projectile_timing(distance, 1, kind, options);
}

static inline int col_npc_attack_anim_id(int npc_def_id) {
    const NpcModelMapping* nm = npc_model_lookup((uint16_t)npc_def_id);
    if (!nm)
        return -1;
    if (nm->attack_anim != 65535)
        return (int)nm->attack_anim;
    if (nm->walk_anim != 65535 && nm->walk_anim != nm->idle_anim)
        return (int)nm->walk_anim;
    return -1;
}

typedef struct {
    int anim_id;
    int client_ticks;
    int linger_ticks;
} ColoNpcDeathVisual;

static const ColoNpcDeathVisual COLO_NPC_DEATH_VISUALS[COLO_NUM_NPC_TYPES] = {
    [COLO_FREMENNIK_BERSERKER] = { .anim_id = 10857, .client_ticks = 63,  .linger_ticks = 3 },
    [COLO_FREMENNIK_ARCHER]    = { .anim_id = 10851, .client_ticks = 63,  .linger_ticks = 3 },
    [COLO_FREMENNIK_SEER]      = { .anim_id = 10854, .client_ticks = 63,  .linger_ticks = 3 },
    [COLO_SERPENT_SHAMAN]      = { .anim_id = 10860, .client_ticks = 47,  .linger_ticks = 2 },
    [COLO_JAGUAR_WARRIOR]      = { .anim_id = 10848, .client_ticks = 63,  .linger_ticks = 3 },
    [COLO_JAVELIN_COLOSSUS]    = { .anim_id = 10894, .client_ticks = 60,  .linger_ticks = 2 },
    [COLO_SHOCKWAVE_COLOSSUS]  = { .anim_id = 10895, .client_ticks = 60,  .linger_ticks = 2 },
    [COLO_MINOTAUR]            = { .anim_id = 10845, .client_ticks = 150, .linger_ticks = 5 },
    [COLO_MANTICORE]           = { .anim_id = 10866, .client_ticks = 80,  .linger_ticks = 3 },
    [COLO_SOL_HEREDIT]         = { .anim_id = 10888, .client_ticks = 230, .linger_ticks = 8 },
    [COLO_HEALING_TOTEM]       = { .anim_id = -1,    .client_ticks = 0,   .linger_ticks = 0 },
    [COLO_BEE_SWARM]           = { .anim_id = -1,    .client_ticks = 0,   .linger_ticks = 0 },
};

static inline const ColoNpcDeathVisual* col_npc_death_visual(ColoNpcType type) {
    assert(type >= 0 && type < COLO_NUM_NPC_TYPES);
    return &COLO_NPC_DEATH_VISUALS[type];
}

static inline int col_npc_death_anim_id(ColoNpcType type) {
    return col_npc_death_visual(type)->anim_id;
}

static inline int col_npc_death_linger_ticks(ColoNpcType type) {
    return col_npc_death_visual(type)->linger_ticks;
}

static inline int col_npc_is_live_target(const ColoNPC* npc) {
    return npc && npc->active && npc->death_ticks == 0 && npc->hp > 0;
}

static inline int col_npc_is_live_enemy(const ColoNPC* npc) {
    return col_npc_is_live_target(npc) && !col_type_is_hazard_entity(npc->type);
}

static void col_queue_npc_pending_hit(
    ColosseumState* s,
    int npc_slot,
    int damage,
    int ticks,
    AttackStyle style,
    int spell_type
) {
    ColoNPC* npc = &s->npcs[npc_slot];
    EncounterPendingHit hit = {
        .active = 1,
        .ticks_remaining = (int8_t)ticks,
        .attack_style = (int8_t)style,
        .check_prayer = 0,
        .spell_type = (int8_t)spell_type,
        .source_npc_slot = -1,
        .damage = (int16_t)damage,
    };
    encounter_pending_hit_queue_push(
        &npc->pending_hits, hit, "colosseum-npc", s->tick, npc_slot, npc->type);
}

static inline int col_scale_incoming_damage(const ColosseumState* s, int dmg) {
    float scale = s->active_player_damage_received_scale;
    if (scale >= 1.0f) return dmg;
    if (scale <= 0.0f) return 0;
    int scaled = (int)((float)dmg * scale + 0.5f);
    return scaled < 0 ? 0 : scaled;
}

static void col_push_player_pending_hit(ColosseumState* s, EncounterPendingHit hit) {
    hit.damage = (int16_t)col_scale_incoming_damage(s, hit.damage);
    encounter_pending_hit_queue_push(
        &s->player_pending_hits, hit, "colosseum-player", s->tick,
        hit.source_npc_slot, hit.source_npc_type);
}

static ColConfig col_default_config(void) {
    return (ColConfig){
        .start_wave = 0,
        .damage_reward_coeff = 0.0f,
        .boss_damage_reward_coeff = 0.0f,
        .wave_clear_bonus = 1.0f,
        .boss_phase_bonus = 0.0f,
        .win_bonus = 0.0f,
        .farm_safe_damage_cap = 0,
        .farm_cap_waves = COLO_FARM_CAP_WAVES,
        .death_penalty_coeff = 0.0f,
        .timeout_penalty = 0.0f,
        .curriculum_agent = 0,
        .mask_inventory_heads = 0,
        .late_start_state_mode = 2,
        .late_start_supply_fraction_per_wave = 0.07f,
        .prayer_switch_fail_prob = 0.0f,
        .player_damage_received_scale = 1.0f,
        .damage_scale_anneal_start = 1.0f,
        .damage_scale_anneal_ticks = 0,
    };
}

static void col_init_context_typed(ColosseumContext* ctx) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->config = col_default_config();
}

static void col_init_context(EncounterContext* context) {
    col_init_context_typed((ColosseumContext*)context);
}

static void col_destroy_context(EncounterContext* context) {
    (void)context;
}

static ColosseumContext* col_legacy_context(void) {
    static ColosseumContext ctx;
    static int initialized = 0;
    if (!initialized) {
        col_init_context_typed(&ctx);
        initialized = 1;
    }
    return &ctx;
}

static void col_init_state_typed(ColosseumState* s, ColosseumContext* ctx) {
    col_build_npc_stats();
    memset(s, 0, sizeof(*s));
    s->rng_state = 12345;
    s->start_wave = ctx ? ctx->config.start_wave : 0;
    s->episode_max_ticks = COLO_MAX_TICKS;
    s->min_sol_hp_seen = COLO_SOL_HP_MAX;
}

static void col_init_state(EncounterState* state, EncounterContext* context) {
    col_init_state_typed((ColosseumState*)state, (ColosseumContext*)context);
}

static EncounterState* col_create(void) {
    ColosseumState* s = (ColosseumState*)calloc(1, sizeof(ColosseumState));
    col_init_state_typed(s, col_legacy_context());
    return (EncounterState*)s;
}

static void col_destroy(EncounterState* state) {
    free(state);
}
