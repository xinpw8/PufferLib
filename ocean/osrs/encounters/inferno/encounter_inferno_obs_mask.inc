#line 6591 "encounter_inferno.h"
#define INF_PLAYER_OBS_SIZE 45
#define INF_PILLAR_FEATURES 3
#define INF_PILLAR_OBS_SIZE (INF_NUM_PILLARS * INF_PILLAR_FEATURES)
#define INF_NPC_SLOT_FEATURES 13
#define INF_TOTAL_NPC_OBS_SIZE (INF_OBS_NPCS * INF_NPC_SLOT_FEATURES)
#define INF_FEATURES_PER_HIT 3
#define INF_SPARK_OBS_SLOTS INF_MAX_PENDING_SPARKS
#define INF_FEATURES_PER_SPARK 4
#define INF_PENDING_HIT_OBS_SIZE \
    (ENCOUNTER_MAX_PENDING_HITS * INF_FEATURES_PER_HIT)
#define INF_PENDING_SPARK_OBS_SIZE \
    (INF_SPARK_OBS_SLOTS * INF_FEATURES_PER_SPARK)
#define INF_INVENTORY_OBS_SIZE OSRS_INVENTORY_SIZE
#define INF_EQUIPPED_OBS_SIZE OSRS_EQUIPMENT_EFFECT_AGGREGATE_FEATURES

#define INF_OBS_AFTER_PLAYER INF_PLAYER_OBS_SIZE
#define INF_OBS_AFTER_PILLARS (INF_OBS_AFTER_PLAYER + INF_PILLAR_OBS_SIZE)
#define INF_OBS_AFTER_NPCS (INF_OBS_AFTER_PILLARS + INF_TOTAL_NPC_OBS_SIZE)
#define INF_OBS_AFTER_PENDING_HITS \
    (INF_OBS_AFTER_NPCS + INF_PENDING_HIT_OBS_SIZE)
#define INF_OBS_AFTER_SPARKS \
    (INF_OBS_AFTER_PENDING_HITS + INF_PENDING_SPARK_OBS_SIZE)
#define INF_OBS_AFTER_INVENTORY \
    (INF_OBS_AFTER_SPARKS + INF_INVENTORY_OBS_SIZE)
#define INF_OBS_AFTER_EQUIPMENT \
    (INF_OBS_AFTER_INVENTORY + INF_EQUIPPED_OBS_SIZE)
#define INF_NUM_OBS INF_OBS_AFTER_EQUIPMENT

static_assert(INF_OBS_AFTER_PLAYER == 45, "Inferno player observation boundary");
static_assert(INF_OBS_AFTER_PILLARS == 54, "Inferno pillar observation boundary");
static_assert(INF_OBS_AFTER_NPCS == 236, "Inferno NPC observation boundary");
static_assert(INF_OBS_AFTER_PENDING_HITS == 332,
    "Inferno pending-hit observation boundary");
static_assert(INF_OBS_AFTER_SPARKS == 460, "Inferno spark observation boundary");
static_assert(INF_OBS_AFTER_INVENTORY == 488,
    "Inferno inventory observation boundary");
static_assert(INF_OBS_AFTER_EQUIPMENT == 498,
    "Inferno equipment observation boundary");
static_assert(INF_NUM_OBS == 498, "Inferno observation width");

#define INF_PROTECTION_PRAYER_CODE_SCALE 8.0f
#define INF_OFFENSIVE_PRAYER_CODE_SCALE 4.0f
#define INF_WAVE_PHASE_CODE_SCALE 8.0f
#define INF_WEAPON_SET_CODE_SCALE 4.0f
#define INF_NPC_TYPE_CODE_SCALE 16.0f
#define INF_ATTACK_STYLE_CODE_SCALE 4.0f
#define INF_TARGET_CATEGORY_CODE_SCALE 8.0f

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
    for (int slot = 0; slot < INF_OBS_NPCS; slot++) obs_slots[slot] = -1;

    int type_base[INF_NUM_NPC_TYPES] = {0};
    int type_cap[INF_NUM_NPC_TYPES] = {0};
    int slot_counts[INF_NUM_NPC_TYPES] = {0};
    int bucket_capacity = 0;
    for (int layout_idx = 0; layout_idx < INF_NUM_NPC_TYPES; layout_idx++) {
        InfNPCType type = INF_OBS_SLOT_LAYOUT[layout_idx].type;
        type_base[type] = bucket_capacity;
        type_cap[type] = INF_OBS_SLOT_LAYOUT[layout_idx].count;
        bucket_capacity += type_cap[type];
    }
    assert(bucket_capacity <= INF_MAX_NPCS);

    int next_slot = 0;
    for (int pass = 0; pass < 2; pass++) {
        int pass_counts[INF_NUM_NPC_TYPES] = {0};
        int pass_slots[INF_MAX_NPCS];
        for (int npc_idx = 0; npc_idx < INF_MAX_NPCS; npc_idx++) {
            InfNPC* npc = &s->npcs[npc_idx];
            if (!npc->active || npc->type < 0 || npc->type >= INF_NUM_NPC_TYPES)
                continue;

            InfNPCType type = npc->type;
            int eligible = pass == 0
                ? npc->death_ticks == 0
                : npc->death_ticks > 0 &&
                    inf_npc_is_phantom_barrage_obs_candidate(s, ctx, npc_idx);
            if (!eligible || slot_counts[type] >= type_cap[type]) continue;

            pass_slots[type_base[type] + pass_counts[type]++] = npc_idx;
            slot_counts[type]++;
        }

        for (int layout_idx = 0; layout_idx < INF_NUM_NPC_TYPES; layout_idx++) {
            InfNPCType type = INF_OBS_SLOT_LAYOUT[layout_idx].type;
            for (int type_slot = 0; type_slot < pass_counts[type]; type_slot++) {
                if (next_slot >= INF_OBS_NPCS) {
                    fprintf(stderr,
                        "inferno: observation target capacity exceeded at %d slots\n",
                        INF_OBS_NPCS);
                    abort();
                }
                obs_slots[next_slot++] = pass_slots[type_base[type] + type_slot];
            }
        }
    }

    for (int slot = 0; slot < INF_OBS_NPCS; slot++)
        s->current_obs_slots[slot] = obs_slots[slot];
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
    int px = s->player.x;
    int py = s->player.y;
    const EncounterLoadoutStats* ls = inf_current_loadout_stats(s, ctx);
    InfSupplyDoses full_supplies = inf_full_starting_supplies();

    int offensive_prayer_code = 0;
    if (s->player.offensive_prayer == OFFENSIVE_PRAYER_PIETY)
        offensive_prayer_code = 1;
    else if (s->player.offensive_prayer == OFFENSIVE_PRAYER_RIGOUR)
        offensive_prayer_code = 2;
    else if (s->player.offensive_prayer == OFFENSIVE_PRAYER_AUGURY)
        offensive_prayer_code = 3;

    obs[i++] = (float)s->player.current_hitpoints / 99.0f;
    obs[i++] = (float)(px - INF_ARENA_MIN_X) / (float)INF_ARENA_WIDTH;
    obs[i++] = (float)(py - INF_ARENA_MIN_Y) / (float)INF_ARENA_HEIGHT;
    obs[i++] = (float)s->player.prayer / INF_PROTECTION_PRAYER_CODE_SCALE;
    obs[i++] = (float)offensive_prayer_code / INF_OFFENSIVE_PRAYER_CODE_SCALE;
    obs[i++] = (float)s->player.brew_doses / (float)full_supplies.brew_doses;
    obs[i++] = (float)s->player.restore_doses / (float)full_supplies.restore_doses;
    obs[i++] = (float)s->player.current_prayer / 99.0f;
    obs[i++] = (float)s->player.bastion_doses / (float)full_supplies.bastion_doses;
    obs[i++] = (float)s->player.stamina_doses / (float)full_supplies.stamina_doses;
    obs[i++] = (float)s->wave / (float)INF_NUM_WAVES;
    obs[i++] = (float)(inf_wave_phase_index(s->wave) + 1) /
        INF_WAVE_PHASE_CODE_SCALE;
    obs[i++] = inf_zuk_attack_timer_obs(s);
    obs[i++] = (float)(s->weapon_set + 1) / INF_WEAPON_SET_CODE_SCALE;
    obs[i++] = s->stamina_active_ticks > 0 ? 1.0f : 0.0f;
    obs[i++] = (float)s->player.potion_timer / 3.0f;
    obs[i++] = (float)s->player.attack_timer / 8.0f;
    obs[i++] = osrs_interaction_active(&s->interaction) ? 1.0f : 0.0f;
    obs[i++] = (float)s->dead_mob_count / (float)INF_MAX_DEAD_MOBS;
    obs[i++] = (float)s->player.special_energy / 100.0f;
    obs[i++] = (float)s->player.current_defence / 99.0f;
    obs[i++] = (float)s->player.current_ranged / 99.0f;
    obs[i++] = (float)s->player.current_magic / 99.0f;
    obs[i++] = (float)ls->attack_range / 15.0f;
    obs[i++] = (float)ls->max_hit / 80.0f;
    obs[i++] = (float)ls->attack_speed / 6.0f;
    obs[i++] = (float)ls->def_stab / 300.0f;
    obs[i++] = (float)ls->def_slash / 300.0f;
    obs[i++] = (float)ls->def_crush / 300.0f;
    obs[i++] = (float)ls->def_magic / 300.0f;
    obs[i++] = (float)ls->def_ranged / 300.0f;
    obs[i++] = (float)ls->eff_level / 165.0f;
    obs[i++] = (float)ls->attack_bonus / 200.0f;
    obs[i++] = (float)ls->strength_bonus / 160.0f;
    obs[i++] = (float)osrs_spec_cost(
        s->player.equipped[GEAR_SLOT_WEAPON]) / 100.0f;
    obs[i++] = (float)ls->spell_base_damage / 40.0f;

    if (s->wave == INF_WAVE_ZUK) {
        int shield_idx = s->zuk.shield_idx;
        int shield_active = shield_idx >= 0 &&
            shield_idx < INF_MAX_NPCS && s->npcs[shield_idx].active;
        obs[i++] = shield_active
            ? (s->zuk.shield_freeze > 0 ? 0.0f : (float)s->zuk.shield_dir)
            : 0.0f;
        obs[i++] = shield_active ? (float)s->zuk.shield_freeze / 5.0f : 0.0f;
        int behind_shield = 0;
        float shield_offset = 0.0f;
        if (shield_active) {
            int shield_x = s->npcs[shield_idx].x;
            int shield_size = INF_NPC_STATS[INF_NPC_ZUK_SHIELD].size;
            int shield_center = shield_x + shield_size / 2;
            shield_offset = (float)(px - shield_center) / 15.0f;
            behind_shield =
                px >= shield_x && px < shield_x + shield_size && py >= 41;
        }
        obs[i++] = behind_shield ? 1.0f : 0.0f;
        obs[i++] = shield_offset;
        obs[i++] = s->zuk.enraged ? 1.0f : 0.0f;
        obs[i++] = (float)s->zuk.set_timer / 350.0f;
        obs[i++] = s->zuk.timer_paused ? 1.0f : 0.0f;
        obs[i++] = s->zuk.jad_spawned ? 1.0f : 0.0f;
        obs[i++] = s->zuk.healer_spawned ? 1.0f : 0.0f;
    } else {
        i += 9;
    }
    assert(i == INF_OBS_AFTER_PLAYER);

    for (int pillar_idx = 0; pillar_idx < INF_NUM_PILLARS; pillar_idx++) {
        const InfPillar* pillar = &s->pillars[pillar_idx];
        obs[i++] = (float)pillar->hp / (float)INF_PILLAR_HP;
        obs[i++] = (float)(pillar->x - px) / (float)INF_ARENA_WIDTH;
        obs[i++] = (float)(pillar->y - py) / (float)INF_ARENA_HEIGHT;
    }
    assert(i == INF_OBS_AFTER_PILLARS);
#ifdef INF_PROFILE_ENABLED
    INF_PROFILE_MARK(INF_PROF_OBS_PREFIX);
#endif

    inf_refresh_current_obs_slots_ctx(s, ctx);
#ifdef INF_PROFILE_ENABLED
    INF_PROFILE_MARK(INF_PROF_OBS_REFRESH_SLOTS);
#endif

    for (int slot_idx = 0; slot_idx < INF_OBS_NPCS; slot_idx++) {
        int npc_idx = s->current_obs_slots[slot_idx];
        if (npc_idx < 0) {
            i += INF_NPC_SLOT_FEATURES;
            continue;
        }

        InfNPC* npc = &s->npcs[npc_idx];
        AttackStyle style = npc->type == INF_NPC_JAD
            ? inf_npc_jad(npc)->attack_style
            : (AttackStyle)npc->attack_style;
        InfTargetCategory category = inf_npc_target_category(s, npc);
        float no_los_ticks = 0.0f;
        float dig_freeze_timer = 0.0f;
        float dig_attack_delay = 0.0f;
        if (npc->type == INF_NPC_MELEER) {
            no_los_ticks = (float)npc->no_los_ticks / 50.0f;
            if (no_los_ticks > 1.0f) no_los_ticks = 1.0f;
            dig_freeze_timer = (float)npc->dig_freeze_timer / 6.0f;
            dig_attack_delay = (float)npc->dig_attack_delay / 6.0f;
        }

        obs[i++] = (float)(npc->type + 1) / INF_NPC_TYPE_CODE_SCALE;
        obs[i++] = npc->max_hp > 0
            ? (float)npc->hp / (float)npc->max_hp
            : 0.0f;
        obs[i++] = (float)(npc->x - px) / (float)INF_ARENA_WIDTH;
        obs[i++] = (float)(npc->y - py) / (float)INF_ARENA_HEIGHT;
        obs[i++] = (float)npc->attack_timer / 10.0f;
        obs[i++] = (float)style / INF_ATTACK_STYLE_CODE_SCALE;
        obs[i++] = inf_npc_has_los(s, npc_idx) ? 1.0f : 0.0f;
        obs[i++] = (float)npc->frozen_ticks / (float)BARRAGE_FREEZE_TICKS;
        obs[i++] = (float)category / INF_TARGET_CATEGORY_CODE_SCALE;
        obs[i++] = osrs_interaction_active(&s->interaction) &&
            s->interaction.target_slot == npc_idx ? 1.0f : 0.0f;
        obs[i++] = no_los_ticks;
        obs[i++] = dig_freeze_timer;
        obs[i++] = dig_attack_delay;
    }
    assert(i == INF_OBS_AFTER_NPCS);
#ifdef INF_PROFILE_ENABLED
    INF_PROFILE_MARK(INF_PROF_OBS_NPC_SLOTS);
#endif

    for (int hit_idx = 0; hit_idx < ENCOUNTER_MAX_PENDING_HITS; hit_idx++) {
        if (hit_idx < s->player_pending_hits.count) {
            EncounterPendingHit* hit = &s->player_pending_hits.hits[hit_idx];
            obs[i++] = (float)hit->attack_style / INF_ATTACK_STYLE_CODE_SCALE;
            obs[i++] = (float)inf_pending_hit_obs_timer(hit) / 10.0f;
            obs[i++] = (float)hit->damage / 150.0f;
        } else {
            i += INF_FEATURES_PER_HIT;
        }
    }
    assert(i == INF_OBS_AFTER_PENDING_HITS);
#ifdef INF_PROFILE_ENABLED
    INF_PROFILE_MARK(INF_PROF_OBS_PENDING_HITS);
#endif

    int spark_order[INF_MAX_PENDING_SPARKS];
    int spark_count = 0;
    for (int spark_idx = 0; spark_idx < INF_MAX_PENDING_SPARKS; spark_idx++) {
        if (!s->pending_sparks[spark_idx].active) continue;
        int order_idx = spark_count;
        while (order_idx > 0) {
            int previous_idx = spark_order[order_idx - 1];
            if (!inf_pending_spark_obs_less(
                    s, &s->pending_sparks[spark_idx], spark_idx,
                    &s->pending_sparks[previous_idx], previous_idx))
                break;
            spark_order[order_idx] = previous_idx;
            order_idx--;
        }
        spark_order[order_idx] = spark_idx;
        spark_count++;
    }

    for (int slot_idx = 0; slot_idx < INF_SPARK_OBS_SLOTS; slot_idx++) {
        if (slot_idx >= spark_count) {
            i += INF_FEATURES_PER_SPARK;
            continue;
        }
        const InfPendingSpark* spark = &s->pending_sparks[spark_order[slot_idx]];
        obs[i++] = (float)(spark->x - px) / (float)INF_ARENA_WIDTH;
        obs[i++] = (float)(spark->y - py) / (float)INF_ARENA_HEIGHT;
        obs[i++] = (float)spark->ticks_remaining / 10.0f;
        obs[i++] = (float)spark->damage / 10.0f;
    }
    assert(i == INF_OBS_AFTER_SPARKS);
#ifdef INF_PROFILE_ENABLED
    INF_PROFILE_MARK(INF_PROF_OBS_SPARKS);
#endif

    for (int cell_idx = 0; cell_idx < OSRS_INVENTORY_SIZE; cell_idx++) {
        const OsrsInventoryCell* cell = &s->inventory_cells[cell_idx];
        obs[i++] = osrs_inventory_cell_obs_code_encode(
            osrs_inventory_cell_obs_code(cell->item_idx, cell->raw_osrs_id));
    }
    assert(i == INF_OBS_AFTER_INVENTORY);

    osrs_write_equipment_effect_aggregate(
        &obs[i], &s->player.equipment_effect_profile);
    i += OSRS_EQUIPMENT_EFFECT_AGGREGATE_FEATURES;
    assert(i == INF_OBS_AFTER_EQUIPMENT);
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
                osrs_inventory_cell_click_classify(cell);
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
