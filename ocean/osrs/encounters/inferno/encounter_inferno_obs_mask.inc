#line 6591 "encounter_inferno.h"
#define INF_ENCOUNTER_OBS_SIZE 14
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
#define INF_INVENTORY_OBS_SIZE OSRS_SHARED_INVENTORY_OBS_SIZE
#define INF_EQUIPPED_OBS_SIZE OSRS_SHARED_EQUIPPED_OBS_SIZE

#define INF_OBS_AFTER_SHARED OSRS_SHARED_OBS_SIZE
#define INF_OBS_AFTER_ENCOUNTER \
    (INF_OBS_AFTER_SHARED + INF_ENCOUNTER_OBS_SIZE)
#define INF_OBS_AFTER_PILLARS \
    (INF_OBS_AFTER_ENCOUNTER + INF_PILLAR_OBS_SIZE)
#define INF_OBS_AFTER_NPCS \
    (INF_OBS_AFTER_PILLARS + INF_TOTAL_NPC_OBS_SIZE)
#define INF_OBS_AFTER_PENDING_HITS \
    (INF_OBS_AFTER_NPCS + INF_PENDING_HIT_OBS_SIZE)
#define INF_OBS_AFTER_SPARKS \
    (INF_OBS_AFTER_PENDING_HITS + INF_PENDING_SPARK_OBS_SIZE)
#define INF_NUM_OBS INF_OBS_AFTER_SPARKS

#define INF_OBS_WAVE_NORM INF_OBS_AFTER_SHARED
#define INF_OBS_WAVE_PHASE (INF_OBS_WAVE_NORM + 1)
#define INF_OBS_ZUK_ATTACK_TIMER (INF_OBS_WAVE_PHASE + 1)
#define INF_OBS_STAMINA_ACTIVE (INF_OBS_ZUK_ATTACK_TIMER + 1)
#define INF_OBS_DEAD_MOB_COUNT (INF_OBS_STAMINA_ACTIVE + 1)
#define INF_OBS_ZUK_PHASE_START (INF_OBS_DEAD_MOB_COUNT + 1)
#define INF_OBS_ZUK_SHIELD_DIR INF_OBS_ZUK_PHASE_START
#define INF_OBS_ZUK_SHIELD_FREEZE (INF_OBS_ZUK_PHASE_START + 1)

static_assert(INF_OBS_AFTER_SHARED == 101,
    "Inferno shared observation boundary");
static_assert(INF_OBS_AFTER_ENCOUNTER == 115,
    "Inferno encounter observation boundary");
static_assert(INF_OBS_AFTER_PILLARS == 124,
    "Inferno pillar observation boundary");
static_assert(INF_OBS_AFTER_NPCS == 306,
    "Inferno NPC observation boundary");
static_assert(INF_OBS_AFTER_PENDING_HITS == 402,
    "Inferno pending-hit observation boundary");
static_assert(INF_OBS_AFTER_SPARKS == 530,
    "Inferno spark observation boundary");
static_assert(INF_NUM_OBS == 530, "Inferno observation width");

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


static const InfNPCType INF_OBS_SLOT_ORDER[INF_NUM_NPC_TYPES] = {
    INF_NPC_MAGER, INF_NPC_RANGER,
    INF_NPC_MELEER, INF_NPC_BLOB,
    INF_NPC_BAT, INF_NPC_BLOB_MAGE,
    INF_NPC_BLOB_RANGE, INF_NPC_BLOB_MELEE,
    INF_NPC_NIBBLER, INF_NPC_JAD,
    INF_NPC_ZUK, INF_NPC_ZUK_SHIELD,
    INF_NPC_HEALER_JAD, INF_NPC_HEALER_ZUK,
};
static const uint8_t INF_OBS_SLOT_CAP[INF_NUM_NPC_TYPES] = {
    [INF_NPC_NIBBLER] = 6,
    [INF_NPC_BAT] = 2,
    [INF_NPC_BLOB] = 2,
    [INF_NPC_BLOB_MELEE] = 2,
    [INF_NPC_BLOB_RANGE] = 2,
    [INF_NPC_BLOB_MAGE] = 2,
    [INF_NPC_MELEER] = 2,
    [INF_NPC_RANGER] = 2,
    [INF_NPC_MAGER] = 2,
    [INF_NPC_JAD] = 3,
    [INF_NPC_ZUK] = 1,
    [INF_NPC_HEALER_JAD] = 6,
    [INF_NPC_HEALER_ZUK] = 4,
    [INF_NPC_ZUK_SHIELD] = 1,
};
#define INF_OBS_SLOT_MAX_TYPE_CAP 6
static_assert(INF_MAX_NPCS <= INT8_MAX,
    "Inferno observation slots fit signed bytes");

static void inf_invalidate_current_obs_slots(InfernoState* s) {
    s->current_obs_slots_valid = 0;
}

static void inf_refresh_current_obs_slots_ctx(
    InfernoState* s,
    const InfernoContext* ctx
) {
    int8_t live_slots[INF_NUM_NPC_TYPES][INF_OBS_SLOT_MAX_TYPE_CAP];
    int8_t phantom_slots[INF_NUM_NPC_TYPES][INF_OBS_SLOT_MAX_TYPE_CAP];
    uint8_t live_counts[INF_NUM_NPC_TYPES] = {0};
    uint8_t phantom_counts[INF_NUM_NPC_TYPES] = {0};

    for (int npc_idx = 0; npc_idx < INF_MAX_NPCS; npc_idx++) {
        const InfNPC* npc = &s->npcs[npc_idx];
        if (!npc->active || npc->type < 0 || npc->type >= INF_NUM_NPC_TYPES)
            continue;
        InfNPCType type = npc->type;
        uint8_t cap = INF_OBS_SLOT_CAP[type];
        if (npc->death_ticks == 0) {
            if (live_counts[type] < cap)
                live_slots[type][live_counts[type]++] = (int8_t)npc_idx;
        } else if (phantom_counts[type] < cap &&
                inf_npc_is_phantom_barrage_obs_candidate(s, ctx, npc_idx)) {
            phantom_slots[type][phantom_counts[type]++] = (int8_t)npc_idx;
        }
    }

    int next_slot = 0;
    for (int layout_idx = 0; layout_idx < INF_NUM_NPC_TYPES; layout_idx++) {
        InfNPCType type = INF_OBS_SLOT_ORDER[layout_idx];
        for (int type_slot = 0; type_slot < live_counts[type]; type_slot++) {
            if (next_slot >= INF_OBS_NPCS) {
                fprintf(stderr,
                    "inferno: observation target capacity exceeded at %d slots\n",
                    INF_OBS_NPCS);
                abort();
            }
            s->current_obs_slots[next_slot++] = live_slots[type][type_slot];
        }
    }
    for (int layout_idx = 0; layout_idx < INF_NUM_NPC_TYPES; layout_idx++) {
        InfNPCType type = INF_OBS_SLOT_ORDER[layout_idx];
        int available = INF_OBS_SLOT_CAP[type] - live_counts[type];
        int count = phantom_counts[type] < available
            ? phantom_counts[type] : available;
        for (int type_slot = 0; type_slot < count; type_slot++) {
            if (next_slot >= INF_OBS_NPCS) {
                fprintf(stderr,
                    "inferno: observation target capacity exceeded at %d slots\n",
                    INF_OBS_NPCS);
                abort();
            }
            s->current_obs_slots[next_slot++] = phantom_slots[type][type_slot];
        }
    }
    for (int slot = next_slot; slot < INF_OBS_NPCS; slot++)
        s->current_obs_slots[slot] = -1;
    s->current_obs_slots_valid = 1;
}

static void inf_ensure_current_obs_slots_ctx(
    InfernoState* s,
    const InfernoContext* ctx
) {
    if (!s->current_obs_slots_valid)
        inf_refresh_current_obs_slots_ctx(s, ctx);
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
    InfernoContext* ctx = (InfernoContext*)context;
#ifdef INF_PROFILE_ENABLED
    int inf_prof_enabled = INF_PROFILE_ENABLED();
    double inf_prof_t0 = inf_prof_enabled ? INF_PROFILE_NOW_MS() : 0.0;
#endif
    int i = 0;
    int px = s->player.x;
    int py = s->player.y;
    const EncounterLoadoutStats* ls = inf_current_loadout_stats(s, ctx);
    OsrsSharedObservationInput shared_input = {
        .player = &s->player,
        .interaction = &s->interaction,
        .arena_min_x = INF_ARENA_MIN_X,
        .arena_max_x = INF_ARENA_MAX_X,
        .arena_min_y = INF_ARENA_MIN_Y,
        .arena_max_y = INF_ARENA_MAX_Y,
        .attack_style = ls->style,
        .attack_range = ls->attack_range,
        .max_hit = ls->max_hit,
        .attack_speed = ls->attack_speed,
        .defence_stab = ls->def_stab,
        .defence_slash = ls->def_slash,
        .defence_crush = ls->def_crush,
        .defence_magic = ls->def_magic,
        .defence_ranged = ls->def_ranged,
        .effective_level = ls->eff_level,
        .attack_bonus = ls->attack_bonus,
        .strength_bonus = ls->strength_bonus,
        .spell_base_damage = ls->spell_base_damage,
        .special_attack_cost =
            osrs_spec_cost(s->player.equipped[GEAR_SLOT_WEAPON]),
    };
    i += osrs_write_shared_observations(&obs[i], &shared_input);
    assert(i == INF_OBS_AFTER_SHARED);

    obs[i++] = (float)s->wave / (float)INF_NUM_WAVES;
    obs[i++] = (float)(inf_wave_phase_index(s->wave) + 1) /
        INF_WAVE_PHASE_CODE_SCALE;
    obs[i++] = inf_zuk_attack_timer_obs(s);
    obs[i++] = s->stamina_active_ticks > 0 ? 1.0f : 0.0f;
    obs[i++] = (float)s->dead_mob_count / (float)INF_MAX_DEAD_MOBS;

    if (s->wave == INF_WAVE_ZUK) {
        int shield_idx = s->zuk.shield_idx;
        int shield_active = shield_idx >= 0 &&
            shield_idx < INF_MAX_NPCS && s->npcs[shield_idx].active;
        obs[i++] = shield_active
            ? (s->zuk.shield_freeze > 0 ? 0.0f : (float)s->zuk.shield_dir)
            : 0.0f;
        obs[i++] = shield_active
            ? (float)s->zuk.shield_freeze / 5.0f : 0.0f;
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
        memset(&obs[i], 0, 9 * sizeof(*obs));
        i += 9;
    }
    assert(i == INF_OBS_AFTER_ENCOUNTER);

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

    inf_ensure_current_obs_slots_ctx(s, ctx);
#ifdef INF_PROFILE_ENABLED
    INF_PROFILE_MARK(INF_PROF_OBS_REFRESH_SLOTS);
#endif

    for (int slot_idx = 0; slot_idx < INF_OBS_NPCS; slot_idx++) {
        int npc_idx = s->current_obs_slots[slot_idx];
        if (npc_idx < 0) {
            memset(&obs[i], 0, INF_NPC_SLOT_FEATURES * sizeof(*obs));
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
        obs[i++] =
            inf_npc_has_los_ctx(s, ctx, npc_idx) ? 1.0f : 0.0f;
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
            memset(&obs[i], 0, INF_FEATURES_PER_HIT * sizeof(*obs));
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
            memset(&obs[i], 0, INF_FEATURES_PER_SPARK * sizeof(*obs));
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

    assert(i == INF_NUM_OBS);
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
    InfernoContext* ctx = (InfernoContext*)context;
    inf_ensure_current_obs_slots_ctx(s, ctx);
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
static inline InfInventoryClickMemo* inf_inventory_click_memo(
    InfernoState* s,
    int cell_idx
) {
    const OsrsInventoryCell* cell = &s->player.inventory_cells[cell_idx];
    InfInventoryClickMemo* memo = &s->inventory_click_memos[cell_idx];
    if (memo->content_code != cell->content_code) {
        OsrsInventoryClickResolution resolution =
            osrs_inventory_cell_click_classify(cell);
        *memo = (InfInventoryClickMemo){
            .content_code = cell->content_code,
            .click_action = (uint8_t)resolution.click_action,
            .consumable_kind = (uint8_t)resolution.consumable_kind,
        };
    }
    return memo;
}


static void inf_write_mask_ctx(
    EncounterState* state,
    EncounterContext* context,
    float* mask
) {
    InfernoState* s = (InfernoState*)state;
    InfernoContext* ctx = (InfernoContext*)context;
    inf_ensure_current_obs_slots_ctx(s, ctx);
    memset(mask, 0, INF_ACTION_MASK_SIZE * sizeof(*mask));

    int primary_offset = inf_action_head_mask_offset(INF_HEAD_PRIMARY);
    mask[primary_offset] = 1.0f;
    int pillars_at_canonical_positions =
        inf_pillars_at_canonical_positions(s);
    uint32_t static_footprint_mask =
        encounter_arena_topology_nearby_unit_footprint_mask(
            inf_route_topology_for_state(ctx, s),
            s->player.x,
            s->player.y);
    for (int action = 1; action < OSRS_PRIMARY_MOVE_ACTIONS; action++) {
        int dx = ENCOUNTER_MOVE_TARGET_DX[action];
        int dy = ENCOUNTER_MOVE_TARGET_DY[action];
        int step_bit = (dy + 2) * 5 + (dx + 2);
        if (step_bit > 12) step_bit--;
        int nx = s->player.x + dx;
        int ny = s->player.y + dy;
        int blocked =
            !(static_footprint_mask & (UINT32_C(1) << step_bit)) ||
            (!pillars_at_canonical_positions &&
             inf_pillar_footprint_blocked(s, nx, ny, 1));
        mask[primary_offset + action] = blocked ? 0.0f : 1.0f;
    }

    int has_forced_safe_healer_target =
        ctx->config.zuk_force_safe_untagged_healer_target_mask &&
        inf_has_safe_untagged_zuk_healer_target_now(s, ctx);
    for (int obs_slot = 0; obs_slot < INF_OBS_NPCS; obs_slot++) {
        int npc_idx = s->current_obs_slots[obs_slot];
        int is_targetable = inf_obs_slot_is_targetable(s, ctx, obs_slot);
        if (is_targetable &&
                (ctx->config.zuk_safe_untagged_healer_target_mask ||
                    ctx->config.zuk_force_safe_untagged_healer_target_mask) &&
                !inf_untagged_zuk_healer_target_is_safe_now(
                    s, ctx, npc_idx)) {
            is_targetable = 0;
        }
        if (is_targetable && has_forced_safe_healer_target &&
                !inf_is_safe_untagged_zuk_healer_target_now(
                    s, ctx, npc_idx)) {
            is_targetable = 0;
        }
        mask[primary_offset +
            inf_primary_attack_action_for_obs_slot(obs_slot)] =
                is_targetable ? 1.0f : 0.0f;
    }
    if (has_forced_safe_healer_target) mask[primary_offset] = 0.0f;

    int overhead_offset = inf_action_head_mask_offset(INF_HEAD_PRAYER);
    mask[overhead_offset + ENCOUNTER_OVERHEAD_NO_CHANGE] = 1.0f;
    mask[overhead_offset + ENCOUNTER_OVERHEAD_OFF] =
        s->player.prayer != PRAYER_NONE ? 1.0f : 0.0f;
    mask[overhead_offset + ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE] =
        s->player.current_prayer > 0 ? 1.0f : 0.0f;
    mask[overhead_offset + ENCOUNTER_OVERHEAD_SET_REFRESH_RANGED] =
        s->player.current_prayer > 0 ? 1.0f : 0.0f;
    mask[overhead_offset + ENCOUNTER_OVERHEAD_SET_REFRESH_MAGIC] =
        s->player.current_prayer > 0 ? 1.0f : 0.0f;
    mask[overhead_offset + ENCOUNTER_OVERHEAD_SET_REFRESH_REDEMPTION] =
        s->player.current_prayer > 0 ? 1.0f : 0.0f;

    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        int head_offset =
            inf_action_head_mask_offset(INF_HEAD_EQUIP_SLOT(slot));
        mask[head_offset] = 1.0f;
    }
    int eat_offset = inf_action_head_mask_offset(INF_HEAD_EAT);
    int drink_offset = inf_action_head_mask_offset(INF_HEAD_DRINK);
    mask[eat_offset] = 1.0f;
    mask[drink_offset] = 1.0f;
    for (int cell_idx = 0; cell_idx < OSRS_INVENTORY_SIZE; cell_idx++) {
        const OsrsInventoryCell* cell =
            &s->player.inventory_cells[cell_idx];
        const InfInventoryClickMemo* memo =
            inf_inventory_click_memo(s, cell_idx);
        switch (memo->click_action) {
            case OSRS_CLICK_EQUIP: {
                if (!osrs_can_equip_from_cell(
                        &s->player, s->player.inventory_cells, cell_idx)) {
                    break;
                }
                int gear_slot =
                    osrs_inventory_cell_metadata(cell)->gear_slot;
                if (gear_slot < 0 || gear_slot >= NUM_GEAR_SLOTS) {
                    fprintf(stderr,
                        "inferno: equippable inventory cell %d has invalid gear slot %d\n",
                        cell_idx, gear_slot);
                    abort();
                }
                int head_offset = inf_action_head_mask_offset(
                    INF_HEAD_EQUIP_SLOT(gear_slot));
                mask[head_offset + cell_idx + 1] = 1.0f;
                break;
            }
            case OSRS_CLICK_EAT:
                if (osrs_can_eat_consumable_kind(
                        &s->player,
                        (OsrsConsumableKind)memo->consumable_kind)) {
                    mask[eat_offset + cell_idx + 1] = 1.0f;
                }
                break;
            case OSRS_CLICK_DRINK:
                if (osrs_inventory_cell_dose_count(cell) > 0 &&
                        s->player.potion_timer == 0 &&
                        inf_drink_has_effect(
                            s,
                            (OsrsConsumableKind)memo->consumable_kind)) {
                    mask[drink_offset + cell_idx + 1] = 1.0f;
                }
                break;
            case OSRS_CLICK_NONE:
                break;
        }
    }

    int spell_offset = inf_action_head_mask_offset(INF_HEAD_SPELL);
    mask[spell_offset + OSRS_SPELL_NONE] = 1.0f;
    mask[spell_offset + OSRS_SPELL_BLOOD_BARRAGE] =
        s->player.current_magic >= BLOOD_BARRAGE_LEVEL ? 1.0f : 0.0f;
    mask[spell_offset + OSRS_SPELL_ICE_BARRAGE] =
        s->player.current_magic >= ICE_BARRAGE_LEVEL ? 1.0f : 0.0f;

    int spec_offset = inf_action_head_mask_offset(INF_HEAD_SPEC);
    mask[spec_offset] = 1.0f;
    mask[spec_offset + 1] =
        !s->player.spec_armed &&
        s->player.equipped[GEAR_SLOT_WEAPON] == ITEM_TOXIC_BLOWPIPE &&
        s->player.special_energy >= BLOWPIPE_SPEC_COST ? 1.0f : 0.0f;
    mask[spec_offset + 2] = s->player.spec_armed ? 1.0f : 0.0f;

    int offensive_offset =
        inf_action_head_mask_offset(INF_HEAD_OFFENSIVE);
    mask[offensive_offset] = 1.0f;
    mask[offensive_offset + 1] =
        s->player.offensive_prayer != OFFENSIVE_PRAYER_NONE ? 1.0f : 0.0f;
    mask[offensive_offset + 2] =
        s->player.current_prayer > 0 ? 1.0f : 0.0f;
    mask[offensive_offset + 3] =
        s->player.current_prayer > 0 ? 1.0f : 0.0f;
    mask[offensive_offset + 4] =
        s->player.current_prayer > 0 ? 1.0f : 0.0f;

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
    InfernoContext* ctx = (InfernoContext*)context;
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
