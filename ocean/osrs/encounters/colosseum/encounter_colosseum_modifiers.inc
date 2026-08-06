#line 76 "encounter_colosseum.h"

static void col_mod_apply_frailty_hp(ColosseumState* s);

static inline int col_mod_active(const ColosseumState* s, ColoModifier mod) {
    return (s->modifiers.active_mask & (1u << (unsigned)mod)) ? 1 : 0;
}

static inline int col_mod_tier(const ColosseumState* s, ColoModifier mod) {
    return col_mod_active(s, mod) ? s->modifiers.tier[mod] : 0;
}

static const ColoModifier COLO_FIRST_DRAFT_OPTIONS[COLO_MODIFIER_DRAFT_OPTIONS] = {
    COLO_MOD_RELENTLESS,
    COLO_MOD_BLASPHEMY,
    COLO_MOD_FRAILTY,
};

static int col_modifier_eligible(const ColosseumState* s, ColoModifier mod, int next_wave) {
    if (s->modifiers.tier[mod] >= COLO_MODIFIER_MAX_TIER[mod]) return 0;
    if (COLO_MODIFIER_PRE_BOSS_ONLY[mod] && next_wave >= COLO_WAVE_BOSS) return 0;
    if ((mod == COLO_MOD_RED_FLAG || mod == COLO_MOD_DYNAMIC_DUO) &&
        next_wave < COLO_RFDD_FIRST_DRAFT_WAVE) return 0;
    return 1;
}

static int col_modifier_draft_weight(const ColosseumState* s, ColoModifier mod) {
    int tier = s->modifiers.tier[mod];
    if (col_mod_active(s, mod) && tier >= 1 && tier < COLO_MODIFIER_MAX_TIER[mod])
        return COLO_DRAFT_UPGRADE_WEIGHT;
    return 1;
}

static void col_modifier_open_first_draft(ColosseumState* s) {
    for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++)
        s->modifiers.draft_options[o] = (int)COLO_FIRST_DRAFT_OPTIONS[o];
    for (int o = COLO_MODIFIER_DRAFT_OPTIONS - 1; o > 0; o--) {
        int j = encounter_rand_int(&s->rng_state, o + 1);
        int tmp = s->modifiers.draft_options[o];
        s->modifiers.draft_options[o] = s->modifiers.draft_options[j];
        s->modifiers.draft_options[j] = tmp;
    }
    s->modifiers.draft_pending = 1;
    s->modifiers.draft_gates_spawn = 1;
}

static void col_modifier_open_draft(ColosseumState* s, int next_wave) {
    for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++) s->modifiers.draft_options[o] = -1;

    if (next_wave == 0) {
        col_modifier_open_first_draft(s);
        return;
    }

    int pool[COLO_NUM_REAL_MODIFIERS];
    int weight[COLO_NUM_REAL_MODIFIERS];
    int pool_n = 0;
    int total_weight = 0;
    for (int m = 0; m < COLO_NUM_REAL_MODIFIERS; m++) {
        if (!col_modifier_eligible(s, (ColoModifier)m, next_wave)) continue;
        pool[pool_n] = m;
        weight[pool_n] = col_modifier_draft_weight(s, (ColoModifier)m);
        total_weight += weight[pool_n];
        pool_n++;
    }

    int offered = 0;
    while (offered < COLO_MODIFIER_DRAFT_OPTIONS && pool_n > 0) {
        int roll = encounter_rand_int(&s->rng_state, total_weight);
        int pick = 0;
        while (roll >= weight[pick]) roll -= weight[pick++];
        s->modifiers.draft_options[offered++] = pool[pick];
        total_weight -= weight[pick];
        pool[pick] = pool[pool_n - 1];
        weight[pick] = weight[pool_n - 1];
        pool_n--;
    }

    if (offered == 0) return;
    s->modifiers.draft_pending = 1;
    s->modifiers.draft_gates_spawn = 1;
}

static void col_modifier_apply_selection(ColosseumState* s, int option) {
    if (!s->modifiers.draft_pending) return;
    if (option < 0 || option >= COLO_MODIFIER_DRAFT_OPTIONS) return;
    int mod = s->modifiers.draft_options[option];
    if (mod < 0 || mod >= COLO_NUM_REAL_MODIFIERS) return;

    s->modifiers.active_mask |= (1u << (unsigned)mod);
    if (s->modifiers.tier[mod] < COLO_MODIFIER_MAX_TIER[mod]) s->modifiers.tier[mod]++;
    s->log.modifiers_picked++;

    s->modifiers.draft_pending = 0;
    s->modifiers.draft_free_movement = 0;
    for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++) s->modifiers.draft_options[o] = -1;
    if (s->modifiers.draft_gates_spawn) {
        s->modifiers.draft_gates_spawn = 0;
        s->wave_spawn_delay = col_wave_entry_delay_ticks(s->wave_spawn_target);
    }

    if ((ColoModifier)mod == COLO_MOD_FRAILTY) col_mod_apply_frailty_hp(s);
}

static int col_modifier_pick_score(const ColosseumState* s, ColoModifier mod, int next_wave) {
    int t = s->modifiers.tier[mod] + 1;
    int ext = next_wave + 1;
    switch (mod) {
        case COLO_MOD_BLASPHEMY:   return t == 1 ? 90 : t == 2 ? 85 : 80;
        case COLO_MOD_FRAILTY:     return t == 1 ? 88 : t == 2 ? 75 : 10;
        case COLO_MOD_RELENTLESS:  return t == 1 ? 82 : t == 2 ? 55 : 45;
        case COLO_MOD_DOOM:        return t == 1 ? 80 : t == 2 ? 50 : 8;
        case COLO_MOD_MYOPIA:      return t == 1 ? 78 : t == 2 ? 70 : 44;
        case COLO_MOD_MANTIMAYHEM: return t == 1 ? 56 : t == 2 ? 72 : 12;
        case COLO_MOD_SOLARFLARE:  return t == 1 ? 52 : t == 2 ? 48 : 40;
        case COLO_MOD_REENTRY:     return t == 1 ? 50 : t == 2 ? (ext >= 10 ? 42 : 14) : 68;
        case COLO_MOD_VOLATILITY:
            return (t == 1 ? 51 : t == 2 ? 46 : 65) + (ext == 12 ? 30 : 0);
        case COLO_MOD_DYNAMIC_DUO: return ext >= 11 ? 35 : 20;
        case COLO_MOD_QUARTET:     return 18;
        case COLO_MOD_TOTEMIC:     return ext == 12 ? 55 : 16;
        case COLO_MOD_RED_FLAG:    return ext >= 11 ? 19 : 11;
        case COLO_MOD_BEES:        return t == 1 ? 5 : t == 2 ? 4 : 3;
        default:                   return 0;
    }
}

static void col_synthesize_draft_history(ColosseumState* s, int start_wave) {
    for (int w = 0; w < start_wave; w++) {
        col_modifier_open_draft(s, w);
        if (!s->modifiers.draft_pending) continue;
        int best_opt = -1;
        int best_score = -1;
        for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++) {
            int mod = s->modifiers.draft_options[o];
            if (mod < 0) continue;
            int score = col_modifier_pick_score(s, (ColoModifier)mod, w);
            if (score > best_score) { best_score = score; best_opt = o; }
        }
        s->modifiers.draft_gates_spawn = 0;
        if (best_opt >= 0) col_modifier_apply_selection(s, best_opt);
        else s->modifiers.draft_pending = 0;
    }
}

static void col_late_start_drink_noop(void* ctx, OsrsConsumableKind kind) {
    (void)ctx;
    (void)kind;
}

static void col_late_start_drain_supplies(
    ColosseumState* s, const ColosseumContext* ctx, int start_wave
) {
    float per_wave = ctx->config.late_start_supply_fraction_per_wave;
    if (per_wave <= 0.0f) return;
    float f = 1.0f - per_wave * (float)start_wave
            + 0.01f * (float)(encounter_rand_int(&s->rng_state, 11) - 5);
    if (f < 0.10f) f = 0.10f;
    if (f > 1.0f) f = 1.0f;

    int done[COLO_INVENTORY_DISPLAY_SLOTS] = {0};
    for (int c = 0; c < COLO_INVENTORY_DISPLAY_SLOTS; c++) {
        if (done[c] || s->inventory_cells[c].dose == 0) continue;
        int item = s->inventory_cells[c].item_idx;
        int total = 0;
        for (int k = c; k < COLO_INVENTORY_DISPLAY_SLOTS; k++) {
            if (s->inventory_cells[k].item_idx == item &&
                    s->inventory_cells[k].dose > 0) {
                done[k] = 1;
                total += s->inventory_cells[k].dose;
            }
        }
        int drain = total - (int)((float)total * f + 0.5f);
        for (int k = COLO_INVENTORY_DISPLAY_SLOTS - 1; k >= c && drain > 0; k--) {
            if (s->inventory_cells[k].item_idx != item) continue;
            while (s->inventory_cells[k].dose > 0 && drain > 0) {
                OsrsInventoryClickResolution res = osrs_inventory_cell_click_interpret(
                    &s->inventory_cells[k], OSRS_CLICK_TICK_FIRST);
                if (res.click_action != OSRS_CLICK_DRINK) break;
                int scratch_timer = 0;
                OsrsInventoryDrinkConsumeResult consumed =
                    osrs_inventory_cell_consume_drink_one_dose(
                        &s->inventory_cells[k], res, &scratch_timer,
                        col_late_start_drink_noop, s);
                if (!consumed.consumed) break;
                drain--;
            }
        }
    }
    col_sync_consumable_counters_from_inventory(s);
}

#define COLO_WAVE_ENTRY_RESERVOIR_CAP 64

typedef struct {
    OsrsInventoryCell cells[COLO_INVENTORY_DISPLAY_SLOTS];
    uint32_t active_mask;
    uint8_t tier[COLO_NUM_REAL_MODIFIERS];
    int16_t hitpoints;
    int16_t prayer_points;
    uint8_t special_energy;
} ColoWaveEntrySample;

typedef struct {
    int seen;
    int stored;
    ColoWaveEntrySample samples[COLO_WAVE_ENTRY_RESERVOIR_CAP];
} ColoWaveEntryReservoir;

/* Per worker thread, not global. A shared reservoir is order dependent by
   construction: res->seen and the slot draw both advance in whichever order
   threads happen to clear waves, and locking fixes tearing but not order. Envs
   are partitioned schedule(static) (src/pufferl.cu:1040,1334), so each thread
   sees a fixed env subset in a fixed order and its reservoir is reproducible.
   Donors are spread across threads because start_wave is drawn per env, so a
   thread's ~256 envs still hold ~80 wave-1 donors to sample from. */
static OSRS_THREAD_LOCAL ColoWaveEntryReservoir g_colo_wave_entry_res[COLO_NUM_WAVES];
static OSRS_THREAD_LOCAL uint32_t g_colo_wave_entry_rng = 0x9e3779b9u;

static void col_record_wave_entry(const ColosseumState* s, int entered_wave) {
    if (s->start_wave != 0) return;
    if (entered_wave <= 0 || entered_wave >= COLO_NUM_WAVES) return;
    ColoWaveEntryReservoir* res = &g_colo_wave_entry_res[entered_wave];
    res->seen++;
    int slot = -1;
    if (res->stored < COLO_WAVE_ENTRY_RESERVOIR_CAP) {
        slot = res->stored++;
    } else {
        int j = (int)(xorshift32(&g_colo_wave_entry_rng) % (uint32_t)res->seen);
        if (j < COLO_WAVE_ENTRY_RESERVOIR_CAP) slot = j;
    }
    if (slot >= 0) {
        ColoWaveEntrySample* out = &res->samples[slot];
        memcpy(out->cells, s->inventory_cells, sizeof(out->cells));
        out->active_mask = s->modifiers.active_mask;
        for (int m = 0; m < COLO_NUM_REAL_MODIFIERS; m++)
            out->tier[m] = (uint8_t)s->modifiers.tier[m];
        out->hitpoints = (int16_t)s->player.current_hitpoints;
        out->prayer_points = (int16_t)s->player.current_prayer;
        out->special_energy = (uint8_t)s->player.special_energy;
    }
}

static int col_apply_wave_entry_sample(ColosseumState* s, int start_wave) {
    if (start_wave <= 0 || start_wave >= COLO_NUM_WAVES) return 0;
    ColoWaveEntrySample sample;
    int have = 0;
    ColoWaveEntryReservoir* res = &g_colo_wave_entry_res[start_wave];
    if (res->stored > 0) {
        uint32_t r = xorshift32(&g_colo_wave_entry_rng);
        sample = res->samples[r % (uint32_t)res->stored];
        have = 1;
    }
    if (!have) return 0;

    memcpy(s->inventory_cells, sample.cells, sizeof(sample.cells));
    col_sync_consumable_counters_from_inventory(s);
    s->modifiers.active_mask = sample.active_mask;
    for (int m = 0; m < COLO_NUM_REAL_MODIFIERS; m++)
        s->modifiers.tier[m] = sample.tier[m];
    if (col_mod_active(s, COLO_MOD_FRAILTY)) col_mod_apply_frailty_hp(s);
    if (sample.hitpoints > 0 && sample.hitpoints < s->player.current_hitpoints)
        s->player.current_hitpoints = sample.hitpoints;
    if (sample.prayer_points >= 0 && sample.prayer_points < s->player.current_prayer)
        s->player.current_prayer = sample.prayer_points;
    s->player.special_energy = sample.special_energy;
    return 1;
}

static void col_apply_late_start_entry_state(
    ColosseumState* s, const ColosseumContext* ctx, int start_wave
) {
    int mode = ctx->config.late_start_state_mode;
    if (mode <= 0 || start_wave <= 0) return;
    if (mode >= 2 && col_apply_wave_entry_sample(s, start_wave)) return;
    col_synthesize_draft_history(s, start_wave);
    col_late_start_drain_supplies(s, ctx, start_wave);
}

static void col_mod_apply_frailty_hp(ColosseumState* s) {
    int tier = col_mod_tier(s, COLO_MOD_FRAILTY);
    int max_hp = 99 - (99 * COLO_FRAILTY_HP_CUT_PCT[tier]) / 100;
    if (max_hp < 1) max_hp = 1;
    s->player.base_hitpoints = max_hp;
    if (s->player.current_hitpoints > max_hp) s->player.current_hitpoints = max_hp;
}

static int col_mod_relentless_def_level(const ColosseumState* s, int def_level) {
    int tier = col_mod_tier(s, COLO_MOD_RELENTLESS);
    if (tier == 0) return def_level;
    return (def_level * (100 - COLO_RELENTLESS_DEF_BYPASS_PCT[tier])) / 100;
}

static int col_mod_relentless_max_hit_bonus(const ColosseumState* s, int* force_hit) {
    int tier = col_mod_tier(s, COLO_MOD_RELENTLESS);
    *force_hit = (tier >= 3) ? 1 : 0;
    return COLO_RELENTLESS_MAX_HIT_BONUS[tier];
}

static int col_mod_myopia_range(const ColosseumState* s, int base_range) {
    int cut = COLO_MYOPIA_RANGE_CUT[col_mod_tier(s, COLO_MOD_MYOPIA)];
    int r = base_range - cut;
    return r < 1 ? 1 : r;
}

static void col_mod_add_doom_stack(
    ColosseumState* s,
    ColoDamageSource source
) {
    assert(source >= 0 && source < COLO_NUM_DAMAGE_SOURCES);
    int tier = col_mod_tier(s, COLO_MOD_DOOM);
    if (tier <= 0) return;
    int cap = COLO_DOOM_CAP[tier];
    if (s->doom_stacks < cap && s->doom_stacks + 1 >= cap)
        s->last_doom_damage_source = source;
    s->doom_stacks++;
}

static void col_mod_on_player_damaged(
    ColosseumState* s,
    int damage,
    ColoDamageSource source
) {
    if (damage <= 0) return;
    assert(source >= 0 && source < COLO_NUM_DAMAGE_SOURCES);
    s->last_damage_source = source;

    int blas = col_mod_tier(s, COLO_MOD_BLASPHEMY);
    if (blas > 0 && s->player.current_prayer > 0) {
        int drain = (damage * COLO_BLASPHEMY_DRAIN_PCT[blas]) / 100;
        s->player.current_prayer -= drain;
        if (s->player.current_prayer < 0) s->player.current_prayer = 0;
    }

    col_mod_add_doom_stack(s, source);
}

static int col_mod_doom_lethal(const ColosseumState* s) {
    int tier = col_mod_tier(s, COLO_MOD_DOOM);
    return tier > 0 && s->doom_stacks >= COLO_DOOM_CAP[tier];
}

static void col_mod_manticore_apply_venom(ColosseumState* s, int hit_success) {

    if (osrs_effect_profile_has(col_live_effects(s), OSRS_ITEM_EFFECT_VENOM_IMMUNE))
        return;
    if (col_mod_tier(s, COLO_MOD_MANTIMAYHEM) < 2) return;
    if (!hit_success) return;
    s->player_poison = 0;
    s->player_poison_timer = 0;
    if (s->player_venom == 0) {
        s->player_venom = COLO_VENOM_START;
        s->player_venom_timer = COLO_VENOM_INTERVAL;
    } else {
        s->player_venom += COLO_VENOM_STEP;
        if (s->player_venom > COLO_VENOM_CAP) s->player_venom = COLO_VENOM_CAP;
    }
}

static void col_mod_tick_venom(ColosseumState* s) {
    if (s->player_venom <= 0) return;
    if (s->player_venom_timer > 0) s->player_venom_timer--;
    if (s->player_venom_timer > 0) return;
    int dmg = s->player_venom;
    int applied =
        col_damage_player_from(s, dmg, COLO_MANTICORE, COLO_DMG_UNPRAYABLE);
    if (applied > 0) {
        s->last_hit_by_type = COLO_MANTICORE;
        col_mod_on_player_damaged(
            s, applied, COLO_DAMAGE_MANTICORE_VENOM);
    }
    s->player_venom += COLO_VENOM_STEP;
    if (s->player_venom > COLO_VENOM_CAP) s->player_venom = COLO_VENOM_CAP;
    s->player_venom_timer = COLO_VENOM_INTERVAL;
}

static void col_mod_apply_bee_poison(ColosseumState* s) {
    if (osrs_effect_profile_has(col_live_effects(s), OSRS_ITEM_EFFECT_VENOM_IMMUNE))
        return;
    if (s->player_venom > 0) return;
    if (s->player_poison <= 0) s->player_poison_timer = COLO_POISON_INTERVAL;
    if (s->player_poison < COLO_POISON_BEE_CONTACT_SEVERITY)
        s->player_poison = COLO_POISON_BEE_CONTACT_SEVERITY;
}

static void col_mod_tick_poison(ColosseumState* s) {
    if (s->player_poison <= 0) return;
    if (s->player_poison_timer > 0) s->player_poison_timer--;
    if (s->player_poison_timer > 0) return;
    int dmg = (s->player_poison + 4) / 5;
    int applied =
        col_damage_player_from(s, dmg, COLO_BEE_SWARM, COLO_DMG_UNPRAYABLE);
    if (applied > 0) {
        s->last_hit_by_type = COLO_BEE_SWARM;
        col_mod_on_player_damaged(
            s, applied, COLO_DAMAGE_BEE_POISON);
    }
    s->player_poison--;
    if (s->player_poison <= 0) {
        s->player_poison = 0;
        s->player_poison_timer = 0;
    } else {
        s->player_poison_timer = COLO_POISON_INTERVAL;
    }
}

static int col_hazard_tile_free(const ColosseumState* s, int x, int y) {
    if (col_static_blocked(x, y)) return 0;
    if (col_sol_clamp_active(s) && !col_in_boss_arena(s, x, y)) return 0;
    return 1;
}

static int col_spawn_hazard_npc_beside(
    ColosseumState* s, ColoNpcType type, const ColoNPC* owner
) {
    if (col_find_free_npc(s) < 0) return -1;
    int size = col_npc_effective_size(owner);
    if (col_hazard_tile_free(s, owner->x - 1, owner->y - 1))
        return col_spawn_npc_at(s, type, owner->x - 1, owner->y - 1);

    for (int ring = 0; ring < 2; ring++) {
        int lo_x = owner->x - 1 - ring, hi_x = owner->x + size + ring;
        int lo_y = owner->y - 1 - ring, hi_y = owner->y + size + ring;
        for (int y = lo_y; y <= hi_y; y++) {
            for (int x = lo_x; x <= hi_x; x++) {
                int on_shell = (x == lo_x || x == hi_x || y == lo_y || y == hi_y);
                if (!on_shell) continue;
                if (!col_hazard_tile_free(s, x, y)) continue;
                return col_spawn_npc_at(s, type, x, y);
            }
        }
    }

    return -1;
}

static ColoTotem* col_totem_for_owner(ColosseumState* s, int owner_slot) {
    for (int t = 0; t < COLO_MAX_TOTEMS; t++) {
        ColoTotem* tot = &s->totems[t];
        if (tot->phase != COLO_HAZARD_NONE && tot->owner_slot == owner_slot) return tot;
    }
    return NULL;
}

static ColoTotem* col_totem_alloc(ColosseumState* s, int owner_slot) {
    for (int t = 0; t < COLO_MAX_TOTEMS; t++) {
        ColoTotem* tot = &s->totems[t];
        if (tot->phase != COLO_HAZARD_NONE) continue;
        *tot = (ColoTotem){ .phase = COLO_HAZARD_NONE, .owner_slot = owner_slot, .npc_slot = -1 };
        return tot;
    }
    return NULL;
}

static int col_find_sol_slot(const ColosseumState* s) {
    for (int i = 0; i < COLO_MAX_NPCS; i++)
        if (s->npcs[i].active && s->npcs[i].hp > 0 && s->npcs[i].type == COLO_SOL_HEREDIT)
            return i;
    return -1;
}

static void col_hazard_on_npc_death(ColosseumState* s, int npc_slot) {
    const ColoNPC* npc = &s->npcs[npc_slot];
    if (npc->type == COLO_HEALING_TOTEM) {
        for (int o = 0; o < COLO_MAX_TOTEMS; o++) {
            ColoTotem* tot = &s->totems[o];
            if (tot->phase != COLO_HAZARD_ALIVE || tot->npc_slot != npc_slot) continue;
            tot->phase = COLO_HAZARD_RESPAWNING;
            tot->npc_slot = -1;
            tot->respawn_timer = COLO_TOTEM_RESPAWN_TICKS;
            tot->projectile_timer = 0;
            return;
        }
        return;
    }
    if (npc->type == COLO_BEE_SWARM) {
        for (int b = 0; b < COLO_MAX_BEE_SWARMS; b++) {
            ColoBeeSwarm* bee = &s->bees[b];
            if (bee->phase != COLO_HAZARD_ALIVE || bee->npc_slot != npc_slot) continue;
            bee->phase = COLO_HAZARD_RESPAWNING;
            bee->npc_slot = -1;
            bee->respawn_timer = COLO_BEE_RESPAWN_TICKS;
            return;
        }
        return;
    }
}

static void col_totem_release(ColosseumState* s, ColoTotem* tot) {
    if (tot->phase == COLO_HAZARD_ALIVE && tot->npc_slot >= 0 &&
        s->npcs[tot->npc_slot].active &&
        s->npcs[tot->npc_slot].type == COLO_HEALING_TOTEM)
        col_deactivate_npc(s, tot->npc_slot);
    *tot = (ColoTotem){ .phase = COLO_HAZARD_NONE };
}

static void col_mod_on_owner_death(ColosseumState* s, int owner_slot) {
    for (int t = 0; t < COLO_MAX_TOTEMS; t++) {
        ColoTotem* tot = &s->totems[t];
        if (tot->phase == COLO_HAZARD_NONE || tot->owner_slot != owner_slot) continue;
        col_totem_release(s, tot);
    }
}

static void col_totem_place(ColosseumState* s, ColoTotem* tot) {
    int slot = col_spawn_hazard_npc_beside(
        s, COLO_HEALING_TOTEM, &s->npcs[tot->owner_slot]);
    if (slot < 0) {
        tot->phase = COLO_HAZARD_RESPAWNING;
        tot->npc_slot = -1;
        tot->respawn_timer = 1;
        tot->projectile_timer = 0;
        return;
    }
    tot->phase = COLO_HAZARD_ALIVE;
    tot->npc_slot = slot;
    tot->heal_timer = COLO_TOTEM_SPAWN_HEAL_DELAY;
    tot->projectile_timer = 0;
}

static int col_totem_spawn_for_owner(ColosseumState* s, int owner_slot) {
    ColoTotem* tot = col_totem_alloc(s, owner_slot);
    if (!tot) return 0;
    col_totem_place(s, tot);
    return 1;
}

static int col_totem_heal_amount(const ColoNPC* owner) {
    return owner->type == COLO_SOL_HEREDIT
        ? COLO_TOTEM_SOL_HEAL
        : (owner->max_hp * COLO_TOTEM_HEAL_PCT) / 100;
}

static void col_totem_apply_heal(ColoNPC* owner) {
    owner->hp += col_totem_heal_amount(owner);
    if (owner->hp > owner->max_hp) owner->hp = owner->max_hp;
}

static const ColoTotem* col_totem_for_npc_slot(const ColosseumState* s, int npc_slot) {
    for (int t = 0; t < COLO_MAX_TOTEMS; t++) {
        const ColoTotem* tot = &s->totems[t];
        if (tot->phase == COLO_HAZARD_ALIVE && tot->npc_slot == npc_slot) return tot;
    }
    return NULL;
}

static int col_live_totem_count(const ColosseumState* s) {
    int n = 0;
    for (int t = 0; t < COLO_MAX_TOTEMS; t++)
        if (s->totems[t].phase == COLO_HAZARD_ALIVE) n++;
    return n;
}

static void col_mod_on_npc_hp_changed(ColosseumState* s, int npc_slot) {
    if (!col_mod_active(s, COLO_MOD_TOTEMIC)) return;
    const ColoNPC* npc = &s->npcs[npc_slot];
    if (!npc->active || npc->hp <= 0 || npc->max_hp <= 0) return;
    if (col_type_is_hazard_entity(npc->type)) return;
    if (npc->hp * 2 > npc->max_hp) return;
    if (col_totem_for_owner(s, npc_slot)) return;
    if (!col_totem_spawn_for_owner(s, npc_slot)) return;
    if (npc->type == COLO_SOL_HEREDIT)
        s->sol_totem_spawn_timer = COLO_TOTEM_SOL_EXTRA_INTERVAL;
}

static void col_mod_tick_sol_extra_totems(ColosseumState* s) {
    if (s->sol_totem_spawn_timer <= 0) return;
    int sol_slot = col_find_sol_slot(s);
    if (sol_slot < 0) {
        s->sol_totem_spawn_timer = 0;
        return;
    }
    s->sol_totem_spawn_timer--;
    if (s->sol_totem_spawn_timer > 0) return;
    s->sol_totem_spawn_timer = COLO_TOTEM_SOL_EXTRA_INTERVAL;
    col_totem_spawn_for_owner(s, sol_slot);
}

static void col_mod_tick_totems(ColosseumState* s) {
    col_mod_tick_sol_extra_totems(s);

    for (int t = 0; t < COLO_MAX_TOTEMS; t++) {
        ColoTotem* tot = &s->totems[t];
        if (tot->phase == COLO_HAZARD_NONE) continue;

        ColoNPC* owner = &s->npcs[tot->owner_slot];
        if (!owner->active || owner->hp <= 0 ||
            col_type_is_hazard_entity(owner->type)) {
            col_totem_release(s, tot);
            continue;
        }

        if (tot->phase == COLO_HAZARD_RESPAWNING) {
            if (tot->respawn_timer > 0) tot->respawn_timer--;
            if (tot->respawn_timer == 0) col_totem_place(s, tot);
            continue;
        }

        ColoNPC* totem_npc = (tot->npc_slot >= 0) ? &s->npcs[tot->npc_slot] : NULL;
        if (!totem_npc || !totem_npc->active || totem_npc->type != COLO_HEALING_TOTEM) {
            *tot = (ColoTotem){ .phase = COLO_HAZARD_NONE };
            continue;
        }

        if (tot->projectile_timer > 0) {
            tot->projectile_timer--;
            if (tot->projectile_timer == 0) {
                col_totem_apply_heal(owner);
                tot->heal_timer =
                    COLO_TOTEM_HEAL_INTERVAL - COLO_TOTEM_PROJECTILE_TICKS;
            }
            continue;
        }

        if (owner->hp >= owner->max_hp) continue;

        if (tot->heal_timer > 0) tot->heal_timer--;
        if (tot->heal_timer > 0) continue;
        tot->projectile_timer = COLO_TOTEM_PROJECTILE_TICKS;
    }
}

#define COLO_REENTRY_MOLTEN_MAX_HIT 15
#define COLO_REENTRY_MOLTEN_BURN_INTERVAL 2

typedef struct {
    int x;
    int y;
} ColoMoltenPoolTile;

static int col_mod_molten_pool_covered(const ColosseumState* s, int x, int y) {
    for (int i = 0; i < s->molten_count; i++)
        if (s->molten_x[i] == x && s->molten_y[i] == y) return 1;
    return 0;
}

static int col_mod_molten_pool_available(const ColosseumState* s, int x, int y) {
    return !col_static_blocked(x, y) && !col_mod_molten_pool_covered(s, x, y);
}

static ColoMoltenPoolTile col_mod_molten_pool_random_fallback(ColosseumState* s) {
    ColoMoltenPoolTile candidates[COLO_ARENA_TILE_COUNT];
    int count = 0;
    for (int x = COLO_ARENA_MIN_X; x <= COLO_ARENA_MAX_X; x++) {
        for (int y = COLO_ARENA_MIN_Y; y <= COLO_ARENA_MAX_Y; y++) {
            if (!col_mod_molten_pool_available(s, x, y)) continue;
            candidates[count++] = (ColoMoltenPoolTile){ .x = x, .y = y };
        }
    }
    assert(count > 0);
    return candidates[encounter_rand_int(&s->rng_state, count)];
}

static ColoMoltenPoolTile col_mod_molten_pool_tile_or_fallback(
    ColosseumState* s,
    int x,
    int y
) {
    if (col_mod_molten_pool_available(s, x, y))
        return (ColoMoltenPoolTile){ .x = x, .y = y };
    return col_mod_molten_pool_random_fallback(s);
}

static void col_mod_add_molten_pool(
    ColosseumState* s,
    int x,
    int y,
    ColoMoltenPoolKind kind,
    ColoMoltenPoolLifetime lifetime
) {
    if (col_static_blocked(x, y)) return;
    for (int i = 0; i < s->molten_count; i++) {
        if (s->molten_x[i] != x || s->molten_y[i] != y) continue;
        s->molten_kind[i] |= kind;
        if (lifetime == COLO_POOL_PERMANENT) s->molten_lifetime[i] = lifetime;
        s->molten_burn_timer[i] = 0;
        return;
    }
    int slot;
    if (s->molten_count < COLO_SOL_HAZARD_TILES_MAX) slot = s->molten_count++;
    else slot = encounter_rand_int(&s->rng_state, COLO_SOL_HAZARD_TILES_MAX);
    s->molten_x[slot] = x;
    s->molten_y[slot] = y;
    s->molten_kind[slot] = kind;
    s->molten_lifetime[slot] = lifetime;
    s->molten_burn_timer[slot] = 0;
}

static void col_mod_clear_wave_end_pools(ColosseumState* s) {
    int out = 0;
    for (int i = 0; i < s->molten_count; i++) {
        if (s->molten_lifetime[i] != COLO_POOL_PERMANENT) continue;
        if (out != i) {
            s->molten_x[out] = s->molten_x[i];
            s->molten_y[out] = s->molten_y[i];
            s->molten_kind[out] = s->molten_kind[i];
            s->molten_lifetime[out] = s->molten_lifetime[i];
            s->molten_burn_timer[out] = s->molten_burn_timer[i];
        }
        out++;
    }
    s->molten_count = out;
}

static int col_mod_molten_pool_damage(ColosseumState* s) {
    return 1 + encounter_rand_int(
        &s->rng_state, COLO_REENTRY_MOLTEN_MAX_HIT);
}

static void col_mod_tick_molten_pools(ColosseumState* s) {
    int burned = 0;
    for (int i = 0; i < s->molten_count; i++) {
        int reentry = s->molten_kind[i] & COLO_POOL_REENTRY;
        int volatility = s->molten_kind[i] & COLO_POOL_VOLATILITY;
        assert(reentry || volatility);
        int player_on_pool = s->player.x == s->molten_x[i] && s->player.y == s->molten_y[i];
        if (!burned && player_on_pool && s->molten_burn_timer[i] == 0) {
            int dmg = col_mod_molten_pool_damage(s);
            s->molten_burn_timer[i] =
                COLO_REENTRY_MOLTEN_BURN_INTERVAL - 1;
            int hitpoints_before = s->player.current_hitpoints;
            int applied = col_damage_player_from(
                s,
                dmg,
                reentry && !volatility ? COLO_JAVELIN_COLOSSUS : -1,
                COLO_DMG_UNPRAYABLE);
            if (reentry && volatility) {
                s->log.javelin_damage_by_source[
                    COLO_JAVELIN_DAMAGE_REENTRY_VOLATILITY_POOL] +=
                    (float)(hitpoints_before - s->player.current_hitpoints);
                if (applied > 0) {
                    s->last_hit_by_type = -1;
                    col_mod_on_player_damaged(
                        s, applied, COLO_DAMAGE_REENTRY_VOLATILITY_POOL);
                }
            } else if (reentry) {
                s->log.javelin_damage_by_source[COLO_JAVELIN_DAMAGE_REENTRY_POOL] +=
                    (float)(hitpoints_before - s->player.current_hitpoints);
                if (applied > 0) {
                    s->last_hit_by_type = COLO_JAVELIN_COLOSSUS;
                    col_mod_on_player_damaged(
                        s, applied, COLO_DAMAGE_REENTRY_POOL);
                }
            } else {
                if (applied > 0) {
                    s->last_hit_by_type = -1;
                    col_mod_on_player_damaged(
                        s, applied, COLO_DAMAGE_VOLATILITY_POOL);
                }
            }
            burned = 1;
            continue;
        }
        if (s->molten_burn_timer[i] > 0)
            s->molten_burn_timer[i]--;
    }
}

static void col_mod_reentry_on_skyfall(ColosseumState* s, int tile_x, int tile_y) {
    int tier = col_mod_tier(s, COLO_MOD_REENTRY);
    if (tier == 0) return;
    ColoMoltenPoolLifetime lifetime = tier >= 2 ? COLO_POOL_PERMANENT : COLO_POOL_TEMPORARY;
    col_mod_add_molten_pool(s, tile_x, tile_y, COLO_POOL_REENTRY, lifetime);
    if (tier >= 2) {
        ColoMoltenPoolTile sw = col_mod_molten_pool_tile_or_fallback(
            s, tile_x - 1, tile_y - 1);
        col_mod_add_molten_pool(s, sw.x, sw.y, COLO_POOL_REENTRY, COLO_POOL_PERMANENT);
    }
    if (tier >= 3) {
        ColoMoltenPoolTile west = col_mod_molten_pool_tile_or_fallback(
            s, tile_x - 1, tile_y);
        col_mod_add_molten_pool(s, west.x, west.y, COLO_POOL_REENTRY, COLO_POOL_PERMANENT);
    }
}

static int col_volatility_reach_for_tier(int tier) {
    if (tier <= 0) return 0;
    return tier >= 2 ? 2 : 1;
}

static int col_volatility_tile_in_blast(
    int x,
    int y,
    int size,
    int tier,
    int player_x,
    int player_y
) {
    int reach = col_volatility_reach_for_tier(tier);
    if (reach <= 0) return 0;
    int span = size + 2 * reach;
    return encounter_entity_footprints_overlap(
        x - reach, y - reach, span, player_x, player_y, 1);
}

static int col_volatility_player_in_blast(
    const ColosseumState* s,
    int x,
    int y,
    int size,
    int tier
) {
    return col_volatility_tile_in_blast(
        x, y, size, tier, s->player.x, s->player.y);
}

static void col_mod_volatility_on_death(ColosseumState* s, int x, int y, int size) {
    int tier = col_mod_tier(s, COLO_MOD_VOLATILITY);
    if (tier == 0) return;
    if (col_volatility_player_in_blast(s, x, y, size, tier)) {
        int applied = col_damage_player_from(
            s, COLO_VOLATILITY_MAX_HIT, -1, COLO_DMG_UNPRAYABLE);
        if (applied > 0) {
            s->last_hit_by_type = -1;
            col_mod_on_player_damaged(
                s, applied, COLO_DAMAGE_VOLATILITY_EXPLOSION);
        }
    }
    if (tier >= 3) {
        ColoMoltenPoolLifetime lifetime = col_mod_tier(s, COLO_MOD_REENTRY) >= 2
            ? COLO_POOL_PERMANENT : COLO_POOL_TEMPORARY;
        col_mod_add_molten_pool(
            s, x + size / 2, y + size / 2, COLO_POOL_VOLATILITY, lifetime);
    }
}

static void col_clamp_bee_swarm_to_arena(ColoNPC* npc) {
    int size = col_npc_effective_size(npc);
    int max_x = COLO_ARENA_MAX_X - size + 1;
    int max_y = COLO_ARENA_MAX_Y - size + 1;
    if (npc->x < COLO_ARENA_MIN_X) npc->x = COLO_ARENA_MIN_X;
    if (npc->y < COLO_ARENA_MIN_Y) npc->y = COLO_ARENA_MIN_Y;
    if (npc->x > max_x) npc->x = max_x;
    if (npc->y > max_y) npc->y = max_y;
}

static void col_bee_spawn(ColosseumState* s, int b) {
    int pos = encounter_rand_int(&s->rng_state, COLO_NUM_SPAWN_ANCHORS);
    int slot = col_spawn_npc_at(s, COLO_BEE_SWARM,
        COLO_SPAWN_ANCHORS[pos][0], COLO_SPAWN_ANCHORS[pos][1]);
    ColoNPC* npc = &s->npcs[slot];
    col_clamp_bee_swarm_to_arena(npc);
    npc->target_x = npc->x;
    npc->target_y = npc->y;
    s->bees[b] = (ColoBeeSwarm){
        .phase = COLO_HAZARD_ALIVE,
        .npc_slot = slot,
        .move_timer = COLO_BEE_MOVE_INTERVAL,
    };
}

static void col_mod_sync_bees(ColosseumState* s) {
    int want = col_mod_tier(s, COLO_MOD_BEES);
    for (int b = 0; b < COLO_MAX_BEE_SWARMS; b++) {
        ColoBeeSwarm* bee = &s->bees[b];
        int npc_live = bee->phase == COLO_HAZARD_ALIVE && bee->npc_slot >= 0 &&
            s->npcs[bee->npc_slot].active &&
            s->npcs[bee->npc_slot].type == COLO_BEE_SWARM;
        if (b < want) {
            if (npc_live || bee->phase == COLO_HAZARD_RESPAWNING) continue;
            col_bee_spawn(s, b);
        } else {
            if (npc_live) col_deactivate_npc(s, bee->npc_slot);
            *bee = (ColoBeeSwarm){ .phase = COLO_HAZARD_NONE };
        }
    }
}

static int col_bee_swarm_contacts_player(const ColosseumState* s, const ColoNPC* npc) {
    int size = col_npc_effective_size(npc);
    return encounter_entity_footprints_overlap(npc->x, npc->y, size, s->player.x, s->player.y, 1);
}

static void col_bee_swarm_next_tile_toward(
    int player_x,
    int player_y,
    int x,
    int y,
    int size,
    int* out_x,
    int* out_y
) {
    if (!out_x || !out_y || size <= 0) abort();
    int hi_x = x + size - 1;
    int hi_y = y + size - 1;
    if (player_x < x) x--;
    else if (player_x > hi_x) x++;
    if (player_y < y) y--;
    else if (player_y > hi_y) y++;
    int max_x = COLO_ARENA_MAX_X - size + 1;
    int max_y = COLO_ARENA_MAX_Y - size + 1;
    if (x < COLO_ARENA_MIN_X) x = COLO_ARENA_MIN_X;
    if (y < COLO_ARENA_MIN_Y) y = COLO_ARENA_MIN_Y;
    if (x > max_x) x = max_x;
    if (y > max_y) y = max_y;
    *out_x = x;
    *out_y = y;
}

static int col_bee_swarm_contact_damage(ColosseumState* s) {
    int span = COLO_BEE_MAX_DAMAGE - COLO_BEE_MIN_DAMAGE + 1;
    int dmg = COLO_BEE_MIN_DAMAGE + encounter_rand_int(&s->rng_state, span);
    if (osrs_effect_profile_has(col_live_effects(s), OSRS_ITEM_EFFECT_VENOM_IMMUNE))
        dmg /= 2;
    return dmg;
}

static void col_bee_swarm_step_toward_player(ColosseumState* s, ColoNPC* npc) {
    int next_x, next_y;
    col_bee_swarm_next_tile_toward(
        s->player.x,
        s->player.y,
        npc->x,
        npc->y,
        col_npc_effective_size(npc),
        &next_x,
        &next_y);
    npc->x = next_x;
    npc->y = next_y;
    npc->target_x = s->player.x;
    npc->target_y = s->player.y;
}

static void col_mod_tick_bees(ColosseumState* s) {
    for (int b = 0; b < COLO_MAX_BEE_SWARMS; b++) {
        ColoBeeSwarm* bee = &s->bees[b];
        if (bee->phase == COLO_HAZARD_NONE) continue;

        if (bee->phase == COLO_HAZARD_RESPAWNING) {
            if (bee->respawn_timer > 0) bee->respawn_timer--;
            if (bee->respawn_timer == 0) col_bee_spawn(s, b);
            continue;
        }

        ColoNPC* npc = (bee->npc_slot >= 0) ? &s->npcs[bee->npc_slot] : NULL;
        if (!npc || !npc->active || npc->type != COLO_BEE_SWARM) {
            *bee = (ColoBeeSwarm){ .phase = COLO_HAZARD_NONE };
            continue;
        }

        if (bee->move_timer > 0) bee->move_timer--;
        if (bee->move_timer == 0) {
            col_bee_swarm_step_toward_player(s, npc);
            bee->move_timer = COLO_BEE_MOVE_INTERVAL;
        }

        if (col_bee_swarm_contacts_player(s, npc)) {
            col_mod_apply_bee_poison(s);
            int dmg = col_bee_swarm_contact_damage(s);
            int applied = col_damage_player_from(
                s, dmg, COLO_BEE_SWARM, COLO_DMG_UNPRAYABLE);
            if (applied > 0) {
                s->last_hit_by_type = COLO_BEE_SWARM;
                col_mod_on_player_damaged(
                    s, applied, COLO_DAMAGE_BEE_CONTACT);
            }
        }
    }
}

static void col_solarflare_tile(
    const ColosseumState* s, int pillar_index, int step, int* x, int* y
) {
    if (pillar_index < 0 || pillar_index >= COLO_NUM_PILLARS ||
            step < 0 || step >= COLO_SOLARFLARE_RING_STEPS || !x || !y) {
        fprintf(stderr, "invalid solarflare tile request\n");
        abort();
    }

    int px, py;
    if (s->wave == COLO_WAVE_BOSS && s->sol.started) {
        px = (pillar_index & 1) ? s->sol.boss_arena_max_x - 5
                                : s->sol.boss_arena_min_x + 3;
        py = (pillar_index & 2) ? s->sol.boss_arena_max_y - 5
                                : s->sol.boss_arena_min_y + 3;
    } else {
        px = COLO_PILLARS[pillar_index][0];
        py = COLO_PILLARS[pillar_index][1];
    }
    if (step < 4) {
        *x = px - 1 + step;
        *y = py - 1;
    } else if (step < 8) {
        *x = px + COLO_PILLAR_SIZE;
        *y = py - 1 + (step - 4);
    } else if (step < 12) {
        *x = px + COLO_PILLAR_SIZE - (step - 8);
        *y = py + COLO_PILLAR_SIZE;
    } else {
        *x = px - 1;
        *y = py + COLO_PILLAR_SIZE - (step - 12);
    }
}

static int col_solarflare_step_is_corner(int step) {
    if (step < 0 || step >= COLO_SOLARFLARE_RING_STEPS) {
        fprintf(stderr, "invalid solarflare step\n");
        abort();
    }
    return (step & 3) == 0;
}

static void col_mod_sync_solarflare(ColosseumState* s) {
    s->solarflare = (ColoSolarflareOrb){
        .active = col_mod_active(s, COLO_MOD_SOLARFLARE) ? 1 : 0,
    };
}

static int col_mod_solarflare_max_hit(int tier) {
    if (tier >= 3) return COLO_SOLARFLARE_MAX_HIT_T3;
    if (tier == 2) return COLO_SOLARFLARE_MAX_HIT_T2;
    return COLO_SOLARFLARE_MAX_HIT_T1;
}

static void col_mod_apply_solarflare_contact(ColosseumState* s, int tier) {
    for (int p = 0; p < COLO_NUM_PILLARS; p++) {
        int x, y;
        col_solarflare_tile(s, p, s->solarflare.step, &x, &y);
        if (x != s->player.x || y != s->player.y) continue;
        int dmg =
            encounter_rand_int(&s->rng_state, col_mod_solarflare_max_hit(tier) + 1);
        int applied =
            col_damage_player_from(s, dmg, -1, COLO_DMG_UNPRAYABLE);
        if (applied > 0) {
            s->last_hit_by_type = -1;
            col_mod_on_player_damaged(s, applied, COLO_DAMAGE_SOLARFLARE);
        }
        if (tier >= 3) s->player.prayer = PRAYER_NONE;
    }
}

static void col_solarflare_advance_orb(ColoSolarflareOrb* orb, int tier) {
    if (orb->pause_timer > 0) {
        orb->pause_timer--;
        return;
    }
    if (orb->move_timer > 0) orb->move_timer--;
    if (orb->move_timer != 0) return;
    int next_step = orb->step + 1;
    if (next_step == COLO_SOLARFLARE_RING_STEPS) next_step = 0;
    orb->step = next_step;
    orb->move_timer = (tier >= 3)
        ? COLO_SOLARFLARE_MOVE_INTERVAL_T3 : COLO_SOLARFLARE_MOVE_INTERVAL;
    if (col_solarflare_step_is_corner(orb->step)) {
        if (tier == 1) orb->pause_timer = COLO_SOLARFLARE_CORNER_PAUSE;
        if (tier >= 3) orb->pause_timer = COLO_SOLARFLARE_CORNER_PAUSE_T3;
    }
}

static void col_mod_tick_solarflare(ColosseumState* s) {
    if (!s->solarflare.active) return;
    int tier = col_mod_tier(s, COLO_MOD_SOLARFLARE);
    if (tier == 0) {
        s->solarflare = (ColoSolarflareOrb){0};
        return;
    }

    col_solarflare_advance_orb(&s->solarflare, tier);
    col_mod_apply_solarflare_contact(s, tier);
}

static void col_modifiers_tick(ColosseumState* s) {
#ifdef COLO_PROFILE_ENABLED
    int col_prof_enabled =
        COLO_PROFILE_ENABLED() && !s->profiling_suppressed;
    double col_prof_t0 = col_prof_enabled ? COLO_PROFILE_NOW_MS() : 0.0;
#endif
    col_mod_tick_bees(s);
    col_mod_tick_solarflare(s);
    col_mod_tick_molten_pools(s);
    col_mod_tick_venom(s);
    col_mod_tick_poison(s);
    col_mod_tick_totems(s);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_STEP_MODIFIERS_HAZARDS);
#endif
}

static void col_modifiers_on_wave_cleared(ColosseumState* s) {
    s->player_venom = 0;
    s->player_venom_timer = 0;
}

static void col_modifiers_on_wave_spawn(ColosseumState* s) {
    s->doom_stacks = 0;
    s->last_doom_damage_source = -1;
    s->sol_totem_spawn_timer = 0;
    col_mod_clear_wave_end_pools(s);
    col_mod_apply_frailty_hp(s);
    col_mod_sync_bees(s);
    col_mod_sync_solarflare(s);
}

static int col_modifier_active_tier_sum(const ColosseumState* s) {
    int sum = 0;
    for (int m = 0; m < COLO_NUM_REAL_MODIFIERS; m++)
        if (col_mod_active(s, (ColoModifier)m)) sum += s->modifiers.tier[m];
    return sum;
}
