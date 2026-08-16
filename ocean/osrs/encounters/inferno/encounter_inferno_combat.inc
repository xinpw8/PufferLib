#line 2424 "encounter_inferno.h"
static inline int inf_jad_attack_timer(int wave) {
    return wave == 67 ? 9 : 8;
}

static AttackStyle inf_player_equipped_attack_style(const InfernoState* s) {
    uint8_t weapon = s->player.equipped[GEAR_SLOT_WEAPON];
    AttackStyle style = (AttackStyle)get_item_attack_style(weapon);
    if (style == ATTACK_STYLE_MAGIC ||
        style == ATTACK_STYLE_RANGED ||
        style == ATTACK_STYLE_MELEE) {
        return style;
    }
    return ATTACK_STYLE_RANGED;
}

static void inf_refresh_human_loadout_stats(InfernoState* s) {
    AttackStyle style = inf_player_equipped_attack_style(s);
    int spell_base_damage = (style == ATTACK_STYLE_MAGIC)
        ? inf_player_magic_base_damage(&s->player) : 0;
    encounter_compute_player_equipped_stats(
        &s->player, style, s->player.fight_style, spell_base_damage,
        &s->human_loadout_stats);
}

static FightStyle inf_live_fight_style_for(AttackStyle style) {
    if (style == ATTACK_STYLE_MAGIC) return FIGHT_STYLE_AUTOCAST;
    if (style == ATTACK_STYLE_RANGED) return FIGHT_STYLE_RAPID;
    return FIGHT_STYLE_ACCURATE;
}

static void inf_refresh_live_stats(InfernoState* s) {
    AttackStyle style = inf_player_equipped_attack_style(s);
    int spell_base_damage = (style == ATTACK_STYLE_MAGIC)
        ? inf_player_magic_base_damage(&s->player) : 0;
    encounter_compute_player_equipped_stats(
        &s->player, style, inf_live_fight_style_for(style), spell_base_damage,
        &s->live_stats);
}

static const EncounterLoadoutStats* inf_current_loadout_stats(
    InfernoState* s,
    const InfernoContext* ctx
) {
    if (ctx->human_command_mode) {
        inf_refresh_human_loadout_stats(s);
        return &s->human_loadout_stats;
    }
    return &s->live_stats;
}

static int inf_player_weapon_is(const InfernoState* s, uint8_t item) {
    return s->player.equipped[GEAR_SLOT_WEAPON] == item;
}

static void inf_damage_player(InfernoState* s, int damage);

typedef struct {
    EncounterLoadoutStats stats;
    int spell;
    int is_barrage;
    uint8_t weapon;
} InfPlayerAttack;

static int inf_spell_magic_level_requirement(int spell) {
    if (spell == ENCOUNTER_SPELL_ICE) return ICE_BARRAGE_LEVEL;
    if (spell == ENCOUNTER_SPELL_BLOOD) return BLOOD_BARRAGE_LEVEL;
    fprintf(stderr, "BUG: invalid Inferno spell %d\n", spell);
    abort();
}

static int inf_spell_action_to_spell(int action_spell) {
    if (action_spell == 1) return ENCOUNTER_SPELL_BLOOD;
    if (action_spell == 2) return ENCOUNTER_SPELL_ICE;
    return ENCOUNTER_SPELL_NONE;
}

static FightStyle inf_player_spell_fight_style(const Player* p) {
    return (p->autocast_defensive || p->fight_style == FIGHT_STYLE_DEFENSIVE_AUTOCAST)
        ? FIGHT_STYLE_DEFENSIVE_AUTOCAST
        : FIGHT_STYLE_AUTOCAST;
}

static int inf_autocast_is_active(const InfernoState* s) {
    return s->player.autocast_enabled &&
        item_supports_ancient_autocast(s->player.equipped[GEAR_SLOT_WEAPON]);
}

static void inf_compute_manual_spell_stats(
    InfernoState* s,
    int spell,
    EncounterLoadoutStats* out
) {
    encounter_compute_player_equipped_stats(
        &s->player,
        ATTACK_STYLE_MAGIC,
        inf_player_spell_fight_style(&s->player),
        inf_spell_base_damage(spell),
        out);
    inf_apply_spell_cast_stats(out);
}

static int inf_resolve_player_attack_ctx(
    InfernoState* s,
    const InfernoContext* ctx,
    int manual_spell,
    InfPlayerAttack* out
) {
    memset(out, 0, sizeof(*out));
    out->spell = ENCOUNTER_SPELL_NONE;
    out->weapon = s->player.equipped[GEAR_SLOT_WEAPON];

    if (inf_is_barrage_spell(manual_spell)) {
        if (s->player.current_magic < inf_spell_magic_level_requirement(manual_spell))
            return 0;
        inf_compute_manual_spell_stats(s, manual_spell, &out->stats);
        out->spell = manual_spell;
        out->is_barrage = 1;
        return 1;
    }

    const EncounterLoadoutStats* ls = inf_current_loadout_stats(s, ctx);
    out->stats = *ls;

    if (ls->style != ATTACK_STYLE_MAGIC)
        return 1;

    if (!inf_autocast_is_active(s))
        return 0;

    int autocast_spell = inf_player_autocast_spell(&s->player);
    if (s->player.current_magic < inf_spell_magic_level_requirement(autocast_spell))
        return 0;

    out->spell = autocast_spell;
    out->is_barrage = 1;
    inf_apply_spell_cast_stats(&out->stats);
    return 1;
}

static int inf_resolve_player_attack(
    InfernoState* s,
    int manual_spell,
    InfPlayerAttack* out
) {
    return inf_resolve_player_attack_ctx(
        s, inf_legacy_context(), manual_spell, out);
}

static int inf_player_has_any_barrage_spell_available(const InfernoState* s) {
    return s->player.current_magic >= BLOOD_BARRAGE_LEVEL;
}

static int inf_npc_type_can_be_tagged_off_shield(int type) {
    return type == INF_NPC_JAD ||
        type == INF_NPC_RANGER ||
        type == INF_NPC_MAGER;
}

static int inf_is_live_shield_slot(const InfernoState* s, int npc_idx) {
    if (npc_idx < 0 || npc_idx >= INF_MAX_NPCS) return 0;
    const InfNPC* npc = &s->npcs[npc_idx];
    return npc->active &&
        npc->death_ticks == 0 &&
        npc->hp > 0 &&
        npc->type == INF_NPC_ZUK_SHIELD;
}

static int inf_is_shield_taggable_slot(const InfernoState* s, int npc_idx) {
    if (npc_idx < 0 || npc_idx >= INF_MAX_NPCS) return 0;
    const InfNPC* npc = &s->npcs[npc_idx];
    return npc->active &&
        npc->death_ticks == 0 &&
        npc->hp > 0 &&
        inf_npc_type_can_be_tagged_off_shield(npc->type) &&
        inf_is_live_shield_slot(s, npc->aggro_target);
}

static int inf_player_behind_zuk_shield_now(const InfernoState* s) {
    int shield_idx = s->zuk.shield_idx;
    if (shield_idx < 0 || shield_idx >= INF_MAX_NPCS) return 0;

    const InfNPC* shield = &s->npcs[shield_idx];
    if (!shield->active || shield->death_ticks != 0 || shield->hp <= 0)
        return 0;

    int shield_size = INF_NPC_STATS[INF_NPC_ZUK_SHIELD].size;
    return s->player.x >= shield->x &&
        s->player.x < shield->x + shield_size &&
        s->player.y >= 41;
}

static int inf_is_untagged_live_zuk_healer_slot(
    const InfernoState* s,
    int npc_idx
) {
    if (npc_idx < 0 || npc_idx >= INF_MAX_NPCS) return 0;

    const InfNPC* npc = &s->npcs[npc_idx];
    if (!npc->active || npc->death_ticks != 0 || npc->hp <= 0 ||
            npc->type != INF_NPC_HEALER_ZUK)
        return 0;

    if (npc->aggro_target < 0 || npc->aggro_target >= INF_MAX_NPCS)
        return 0;

    const InfNPC* aggro = &s->npcs[npc->aggro_target];
    return aggro->active && aggro->type == INF_NPC_ZUK;
}

static int inf_player_can_attack_npc_from_current_tile_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    int npc_idx
) {
    if (npc_idx < 0 || npc_idx >= INF_MAX_NPCS) return 0;

    const InfNPC* npc = &s->npcs[npc_idx];
    if (!npc->active || npc->death_ticks != 0 || npc->hp <= 0) return 0;

    const EncounterLoadoutStats* ls = &s->live_stats;
    return inf_los_clear_ctx(
        s,
        ctx,
        s->player.x,
        s->player.y,
        1,
        npc->x,
        npc->y,
        npc->size,
        ls->attack_range);
}


static int inf_player_has_los_to_npc_from_current_tile_ctx(
    const InfernoState* s,
    const InfernoContext* ctx,
    int npc_idx
) {
    if (npc_idx < 0 || npc_idx >= INF_MAX_NPCS) return 0;

    const InfNPC* npc = &s->npcs[npc_idx];
    if (!npc->active || npc->death_ticks != 0 || npc->hp <= 0 ||
            npc->type == INF_NPC_ZUK_SHIELD)
        return 0;

    return inf_los_clear_ctx(
        s,
        ctx,
        s->player.x,
        s->player.y,
        1,
        npc->x,
        npc->y,
        npc->size,
        0);
}



static int inf_player_can_phantom_barrage_npc(
    InfernoState* s,
    const InfernoContext* ctx,
    int npc_idx
) {
    if (npc_idx < 0 || npc_idx >= INF_MAX_NPCS) return 0;

    const InfNPC* npc = &s->npcs[npc_idx];
    if (!npc->active || npc->death_ticks <= 0 || npc->hp > 0 ||
            npc->type == INF_NPC_ZUK_SHIELD)
        return 0;

    if (!inf_player_has_any_barrage_spell_available(s)) return 0;

    OsrsLosQuery los_query = inf_player_los_query(s);
    return encounter_player_can_attack(
        s->player.x,
        s->player.y,
        npc->x,
        npc->y,
        npc->size,
        10,
        ctx->collision_map,
        ctx->world_offset_x,
        ctx->world_offset_y,
        &los_query);
}

static int inf_npc_is_phantom_barrage_obs_candidate(
    InfernoState* s,
    const InfernoContext* ctx,
    int npc_idx
) {
    if (npc_idx < 0 || npc_idx >= INF_MAX_NPCS) return 0;
    return s->npcs[npc_idx].death_ticks <= 2 &&
        inf_player_can_phantom_barrage_npc(s, ctx, npc_idx);
}

static int inf_npc_is_phantom_barrage_cast_window(
    InfernoState* s,
    const InfernoContext* ctx,
    int npc_idx
) {
    if (npc_idx < 0 || npc_idx >= INF_MAX_NPCS) return 0;
    return s->player.attack_timer <= 1 &&
        s->npcs[npc_idx].death_ticks == 1 &&
        inf_player_can_phantom_barrage_npc(s, ctx, npc_idx);
}

static int inf_untagged_zuk_healer_target_is_safe_now(
    const InfernoState* s,
    const InfernoContext* ctx,
    int npc_idx
) {
    if (!inf_is_untagged_live_zuk_healer_slot(s, npc_idx)) return 1;
    return inf_player_behind_zuk_shield_now(s) &&
        s->player.attack_timer == 0 &&
        inf_player_can_attack_npc_from_current_tile_ctx(s, ctx, npc_idx);
}

static int inf_is_safe_untagged_zuk_healer_target_now(
    const InfernoState* s,
    const InfernoContext* ctx,
    int npc_idx
) {
    return inf_is_untagged_live_zuk_healer_slot(s, npc_idx) &&
        inf_untagged_zuk_healer_target_is_safe_now(s, ctx, npc_idx);
}

static int inf_has_safe_untagged_zuk_healer_target_now(
    const InfernoState* s,
    const InfernoContext* ctx
) {
    for (int obs_slot = 0; obs_slot < INF_OBS_NPCS; obs_slot++) {
        int npc_idx = s->current_obs_slots[obs_slot];
        if (npc_idx >= 0 && npc_idx < INF_MAX_NPCS &&
                s->npcs[npc_idx].active &&
                s->npcs[npc_idx].death_ticks == 0 &&
                s->npcs[npc_idx].type != INF_NPC_ZUK_SHIELD &&
                inf_is_safe_untagged_zuk_healer_target_now(s, ctx, npc_idx)) {
            return 1;
        }
    }
    return 0;
}

static int inf_npc_target_hit_delay(
    const InfNPC* npc, const InfNPC* target, int style
) {
    int dist = encounter_projectile_distance(
        npc->x, npc->y, npc->size, target->x, target->y, target->size,
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    EncounterProjectileTiming timing = inf_npc_projectile_timing(npc->type, style, dist);
    if (npc->type == INF_NPC_JAD && timing.damage_delay_ticks > 0)
        return INF_JAD_PROJECTILE_DELAY + timing.damage_delay_ticks;
    return timing.damage_delay_ticks;
}

static void inf_apply_npc_target_damage(InfernoState* s, int target_idx, int damage) {
    if (target_idx < 0 || target_idx >= INF_MAX_NPCS) return;
    InfNPC* target = &s->npcs[target_idx];
    if (!target->active) return;

    int target_hp_before = target->hp;
    encounter_damage_npc(&target->hp, &target->hit_landed_this_tick,
                         &target->hit_damage, damage);
    if (target->type == INF_NPC_ZUK_SHIELD)
        s->tick_scratch.shield_damage += (float)(target_hp_before - target->hp);

    if (target->hp <= 0 && target->type == INF_NPC_ZUK_SHIELD) {
        inf_deactivate_npc(s, target_idx);
        s->zuk.shield_idx = -1;
        for (int i = 0; i < INF_MAX_NPCS; i++) {
            if (s->npcs[i].aggro_target == target_idx)
                s->npcs[i].aggro_target = -1;
        }
    }
}

static void inf_queue_npc_target_hit(
    InfernoState* s, int target_idx, int damage, int ticks_remaining
) {
    if (ticks_remaining <= 0) {
        inf_apply_npc_target_damage(s, target_idx, damage);
        return;
    }
    for (int i = 0; i < INF_MAX_NPC_TARGET_HITS; i++) {
        InfNpcTargetHit* hit = &s->npc_target_hits[i];
        if (hit->active) continue;
        hit->active = 1;
        hit->target_idx = target_idx;
        hit->damage = damage;
        hit->ticks_remaining = ticks_remaining;
        return;
    }
    fprintf(stderr, "FATAL: Inferno NPC target hit queue overflow\n");
    abort();
}

static void inf_resolve_npc_target_hits(InfernoState* s) {
    for (int i = 0; i < INF_MAX_NPC_TARGET_HITS; i++) {
        InfNpcTargetHit* hit = &s->npc_target_hits[i];
        if (!hit->active) continue;
        hit->ticks_remaining--;
        if (hit->ticks_remaining > 0) continue;
        inf_apply_npc_target_damage(s, hit->target_idx, hit->damage);
        hit->active = 0;
    }
}

static int inf_apply_elysian_to_player_hit(InfernoState* s, int damage, int* reduced) {
    *reduced = 0;
    if (damage <= 0) return damage;
    osrs_ensure_player_equipment(&s->player);
    if (!osrs_effect_profile_has(
            &s->player.equipment_effect_profile, OSRS_ITEM_EFFECT_ELYSIAN))
        return damage;
    if (encounter_rand_int(&s->rng_state, 10) >= 7)
        return damage;
    *reduced = 1;
    return damage * 75 / 100;
}

static void inf_apply_echo_boots_recoil(InfernoState* s, int attacker_slot, int final_damage) {
    osrs_ensure_player_equipment(&s->player);
    int recoil_damage = osrs_echo_boots_recoil_damage(
        &s->player.equipment_effect_profile,
        &s->player.item_effect_state,
        final_damage);
    if (recoil_damage <= 0) return;
    if (attacker_slot < 0 || attacker_slot >= INF_MAX_NPCS) return;

    InfNPC* npc = &s->npcs[attacker_slot];
    if (!npc->active || npc->death_ticks > 0 || npc->hp <= 0) return;
    if (npc->type == INF_NPC_ZUK || npc->type == INF_NPC_ZUK_SHIELD) return;

    osrs_consume_echo_boots_charge(&s->player);
    int applied = encounter_damage_npc(
        &npc->hp, &npc->hit_landed_this_tick, &npc->hit_damage,
        recoil_damage);
    if (applied <= 0) return;

    s->tick_scratch.damage_dealt += (float)applied;
    if (npc->type == INF_NPC_HEALER_ZUK) {
        s->tick_scratch.damage_zuk_healers += (float)applied;
    } else if (npc->type == INF_NPC_HEALER_JAD) {
        s->tick_scratch.damage_jad_healers += (float)applied;
    } else if (npc->type == INF_NPC_JAD) {
        s->tick_scratch.damage_jad += (float)applied;
    } else {
        s->tick_scratch.damage_set += (float)applied;
    }

    if (npc->hp <= 0) {
        if (npc->type == INF_NPC_JAD) s->tick_scratch.kill_jad++;
        else if (npc->type == INF_NPC_HEALER_ZUK) s->tick_scratch.kill_zuk_healer++;
        else if (npc->type != INF_NPC_HEALER_JAD) s->tick_scratch.kill_set++;
        inf_apply_npc_death(s, attacker_slot);
    }
}

static void inf_npc_attack_ctx(InfernoState* s, InfernoContext* ctx, int idx) {
    InfNPC* npc = &s->npcs[idx];
    if (!npc->active) return;
    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    if (npc->attack_timer > 0) npc->attack_timer--;
    if (npc->stun_timer > 0) { npc->stun_timer--; return; }
    if (npc->dig_freeze_timer > 0) return;
    if (npc->dig_attack_delay > 0) return;

    int has_los_now = 0;
    if (stats->attack_range > 1) {
        inf_sample_npc_player_los_frame(s, ctx, idx);
        has_los_now = inf_npc_has_los_ctx(s, ctx, idx);
    }

    if (npc->type == INF_NPC_BLOB &&
        npc->blob_scanned_prayer < 0 &&
        has_los_now &&
        !npc->had_los_last_tick) {
        npc->blob_scanned_prayer = (int)s->player.prayer;
        if (s->player.prayer == PRAYER_PROTECT_MAGIC) npc->attack_style = ATTACK_STYLE_RANGED;
        else if (s->player.prayer == PRAYER_PROTECT_RANGED) npc->attack_style = ATTACK_STYLE_MAGIC;
        else npc->attack_style = (encounter_rand_int(&s->rng_state, 2) == 0) ? ATTACK_STYLE_MAGIC : ATTACK_STYLE_RANGED;
        npc->had_los_last_tick = has_los_now;
        npc->attacked_this_tick = 1;
        npc->attack_timer = stats->attack_speed;
        return;
    }

    npc->had_los_last_tick = has_los_now;

    if (npc->attack_timer > 0) return;

    if (npc->type == INF_NPC_ZUK_SHIELD) return;

    if (npc->type == INF_NPC_NIBBLER) {
        for (int p = 0; p < INF_NUM_PILLARS; p++) {
            if (!s->pillars[p].active) continue;
            int ddx = npc->x - s->pillars[p].x;
            int ddy = npc->y - s->pillars[p].y;
            if (ddx >= -1 && ddx <= INF_PILLAR_SIZE && ddy >= -1 && ddy <= INF_PILLAR_SIZE) {
                int dmg = encounter_rand_int(&s->rng_state, 5);
                s->pillars[p].hp -= dmg;
                if (s->pillars[p].hp <= 0) {
                    s->pillars[p].active = 0;
                    inf_reset_npc_player_los_frame(ctx);
                    s->tick_scratch.pillar_lost = p;
                    for (int n = 0; n < INF_MAX_NPCS; n++) {
                        if (!s->npcs[n].active) continue;
                        int ndx = s->npcs[n].x - s->pillars[p].x;
                        int ndy = s->npcs[n].y - s->pillars[p].y;
                        if (ndx >= -1 && ndx <= INF_PILLAR_SIZE && ndy >= -1 && ndy <= INF_PILLAR_SIZE) {
                            encounter_damage_npc(&s->npcs[n].hp, &s->npcs[n].hit_landed_this_tick, &s->npcs[n].hit_damage, 12);
                            inf_apply_npc_death(s, n);
                        }
                    }
                    {
                        int pdx = s->player.x - s->pillars[p].x;
                        int pdy = s->player.y - s->pillars[p].y;
                        if (pdx >= -1 && pdx <= INF_PILLAR_SIZE && pdy >= -1 && pdy <= INF_PILLAR_SIZE) {
                            inf_damage_player(s, 49);
                        }
                    }
                }
                npc->attacked_this_tick = 1;
                npc->attack_timer = stats->attack_speed;
                return;
            }
        }
        return;
    }

    if (npc->type == INF_NPC_HEALER_ZUK) {
        InfHealerState* h = inf_npc_healer(npc);
        if (npc->aggro_target >= 0) {
            int ti = npc->aggro_target;
            if (ti >= 0 && ti < INF_MAX_NPCS && s->npcs[ti].active && s->npcs[ti].type == INF_NPC_ZUK) {
                h->target = ti;
                h->timer = 3;
                npc->attack_visual_target = ti;
            }
        } else {
            h->target = -1;
            h->timer = 0;
            npc->attack_visual_target = -1;
            inf_queue_zuk_healer_sparks(s, npc);
        }
        npc->attacked_this_tick = 1;
        npc->attack_style_this_tick = ATTACK_STYLE_MAGIC;
        npc->attack_timer = stats->attack_speed;
        return;
    }

    if (npc->type == INF_NPC_HEALER_JAD) {
        InfHealerState* h = inf_npc_healer(npc);
        int jad_idx = h->owner_idx;
        if (npc->aggro_target >= 0) {
            if (jad_idx >= 0 && s->npcs[jad_idx].active &&
                s->npcs[jad_idx].type == INF_NPC_JAD) {
                h->target = jad_idx;
                h->timer = 3;
                npc->attack_visual_target = jad_idx;
            }
            npc->attacked_this_tick = 1;
            npc->attack_timer = stats->attack_speed;
            return;
        }
        if (inf_los_clear_ctx(
                s, ctx,
                npc->x, npc->y, 1,
                s->player.x, s->player.y, 1,
                1)) {
            int max_hit = osrs_npc_melee_max_hit(stats->str_level, stats->melee_str_bonus);
            int att_roll = osrs_npc_attack_roll(stats->att_level, stats->melee_att_bonus);
            const EncounterLoadoutStats* ls = inf_current_loadout_stats(s, ctx);
            int def_roll = encounter_player_def_roll_from_loadout(
                s->player.current_defence, s->player.current_magic,
                ls->def_stab, ls->def_slash, ls->def_crush, ls->def_magic, ls->def_ranged,
                ATTACK_STYLE_MELEE, MELEE_STYLE_CRUSH);
            int dmg = encounter_npc_roll_attack(att_roll, def_roll, max_hit, &s->rng_state);
            int prayer_matches = (s->player.prayer == PRAYER_PROTECT_MELEE);
              if (prayer_matches) { dmg = 0; s->tick_scratch.prayer_correct = 1; }
              else if (dmg > 0) { s->tick_scratch.off_prayer_hits++; }
            inf_apply_echo_boots_recoil(s, idx, dmg);
            inf_damage_player(s, dmg);
            npc->attack_style_this_tick = ATTACK_STYLE_MELEE;
            npc->attacked_this_tick = 1;
            npc->attack_visual_target = -1;
            npc->attack_timer = stats->attack_speed;
        }
        return;
    }

    if (npc->aggro_target >= 0 && npc->aggro_target < INF_MAX_NPCS) {
        InfNPC* target = &s->npcs[npc->aggro_target];
        if (target->active) {
            AttackStyle actual_style = npc->attack_style;
            if (npc->type == INF_NPC_JAD) {
                actual_style = inf_npc_jad(npc)->attack_style;
                if (actual_style == ATTACK_STYLE_NONE)
                    actual_style = inf_jad_roll_primary_style(&s->rng_state);
            }
            int max_hit = osrs_npc_max_hit(actual_style,
                stats->str_level, stats->range_level,
                stats->melee_str_bonus, stats->ranged_str_bonus,
                stats->magic_base_dmg, stats->magic_dmg_pct);
            if (stats->max_hit_cap > 0 && max_hit > stats->max_hit_cap)
                max_hit = stats->max_hit_cap;
            int dmg = encounter_rand_int(&s->rng_state, max_hit + 1);
            int hit_delay = inf_npc_target_hit_delay(npc, target, actual_style);
            inf_queue_npc_target_hit(s, npc->aggro_target, dmg, hit_delay);
            npc->attacked_this_tick = 1;
            npc->attack_style_this_tick = actual_style;
            npc->attack_visual_target = npc->aggro_target;
            npc->attack_timer = stats->attack_speed;
            if (npc->type == INF_NPC_JAD) {
                inf_npc_jad(npc)->attack_style = ATTACK_STYLE_NONE;
                npc->attack_timer = inf_jad_attack_timer(s->wave);
            }
            return;
        }
        npc->aggro_target = -1;
    }

    if (stats->attack_range > 1 &&
        !(npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer >= 0) &&
        !has_los_now) return;

    int dist = encounter_dist_to_npc(s->player.x, s->player.y,
                                      npc->x, npc->y, npc->size);
    if (dist == 0 || dist > stats->attack_range) return;
    if (stats->attack_range == 1 &&
            !inf_npc_melee_contact_allows_attack_at_tile(
                s->player.x, s->player.y, npc->type,
                npc->x, npc->y, npc->size, dist))
        return;

    if (npc->type == INF_NPC_BLOB) {
        if (npc->blob_scanned_prayer < 0) {
            npc->blob_scanned_prayer = (int)s->player.prayer;
            if (s->player.prayer == PRAYER_PROTECT_MAGIC) npc->attack_style = ATTACK_STYLE_RANGED;
            else if (s->player.prayer == PRAYER_PROTECT_RANGED) npc->attack_style = ATTACK_STYLE_MAGIC;
            else npc->attack_style = (encounter_rand_int(&s->rng_state, 2) == 0) ? ATTACK_STYLE_MAGIC : ATTACK_STYLE_RANGED;
            npc->attacked_this_tick = 1;
            npc->attack_timer = stats->attack_speed;
            return;
        }
        npc->blob_scanned_prayer = -1;
    }

    AttackStyle actual_style = npc->attack_style;

    if (npc->type == INF_NPC_JAD) {
        actual_style = inf_npc_jad(npc)->attack_style;
        if (actual_style == ATTACK_STYLE_NONE)
            actual_style = inf_jad_roll_primary_style(&s->rng_state);
    }

    if (npc->type == INF_NPC_ZUK) {
        int si = s->zuk.shield_idx;
        int player_behind_shield = 0;
        if (si >= 0 && s->npcs[si].active) {
            InfNPC* shield = &s->npcs[si];
            player_behind_shield = (s->player.x >= shield->x &&
                                    s->player.x < shield->x + shield->size &&
                                    s->player.y >= 41);
        }

        if (player_behind_shield) {
            npc->attacked_this_tick = 1;
            npc->attack_style_this_tick = ATTACK_STYLE_MAGIC;
            npc->attack_visual_target = si;
        } else {
            int max_hit = osrs_npc_magic_max_hit(stats->magic_base_dmg, stats->magic_dmg_pct);
            int dmg = encounter_rand_int(&s->rng_state, max_hit + 1);
            EncounterProjectileTiming zuk_timing =
                inf_npc_projectile_timing(npc->type, ATTACK_STYLE_MAGIC, dist);
            inf_queue_player_pending_hit(s, idx, npc->type, (EncounterPendingHit){
                .ticks_remaining = (int8_t)zuk_timing.damage_delay_ticks,
                .attack_style = ATTACK_STYLE_NONE,
                .check_prayer = 0,
                .prayer_check_delay = 0,
                .spell_type = 0,
                .source_npc_type = (int8_t)npc->type,
                .hit_success = 1,
                .elysian_reduced = 0,
                .damage = (int16_t)dmg,
            });
            s->last_hit_by_type = INF_NPC_ZUK;
            npc->attacked_this_tick = 1;
            npc->attack_style_this_tick = ATTACK_STYLE_MAGIC;
        }
        npc->attack_timer = s->zuk.enraged ? 7 : stats->attack_speed;
        return;
    }

    {
        int style_mask = inf_attack_style_options_mask(
            s, npc, stats, actual_style, dist);
        actual_style = inf_choose_attack_style_for_tick(
            &s->rng_state, style_mask);
    }

    if (npc->type == INF_NPC_MAGER &&
        actual_style == ATTACK_STYLE_MAGIC &&
        inf_mager_resurrect_ctx(s, ctx, idx)) {
        npc->attacked_this_tick = 1;
        npc->attack_style_this_tick = ATTACK_STYLE_NONE;
        npc->attack_timer = stats->attack_speed;
        return;
    }

    int max_hit = osrs_npc_max_hit(actual_style,
        stats->str_level, stats->range_level,
        stats->melee_str_bonus, stats->ranged_str_bonus,
        stats->magic_base_dmg, stats->magic_dmg_pct);
    if (stats->max_hit_cap > 0 && max_hit > stats->max_hit_cap)
        max_hit = stats->max_hit_cap;
    int is_delayed_jad = (npc->type == INF_NPC_JAD &&
                          actual_style != ATTACK_STYLE_MELEE);
    int dmg = is_delayed_jad ? max_hit : encounter_rand_int(&s->rng_state, max_hit + 1);
    int accuracy_hit = 1;

    if (!is_delayed_jad) {
        int att_lvl, att_bonus;
        if (actual_style == ATTACK_STYLE_MELEE) {
            att_lvl = stats->att_level; att_bonus = stats->melee_att_bonus;
        } else if (actual_style == ATTACK_STYLE_RANGED) {
            att_lvl = stats->range_level; att_bonus = stats->range_att_bonus;
        } else {
            att_lvl = stats->magic_level; att_bonus = stats->magic_att_bonus;
        }
        int att_roll = osrs_npc_attack_roll(att_lvl, att_bonus);
        const EncounterLoadoutStats* ls = inf_current_loadout_stats(s, ctx);
        int def_roll = encounter_player_def_roll_from_loadout(
            s->player.current_defence, s->player.current_magic,
            ls->def_stab, ls->def_slash, ls->def_crush, ls->def_magic, ls->def_ranged,
            actual_style, stats->melee_style);
        if (!encounter_roll_hit_chance(&s->rng_state, att_roll, def_roll)) {
            dmg = 0;
            accuracy_hit = 0;
        }
    }

    EncounterProjectileTiming hit_timing =
        inf_npc_projectile_timing(npc->type, actual_style, dist);
    int hit_delay = hit_timing.damage_delay_ticks;
    if (is_delayed_jad)
        hit_delay += INF_JAD_PROJECTILE_DELAY;

    if (actual_style == ATTACK_STYLE_MELEE) s->tick_scratch.styles_fired |= 1;
    else if (actual_style == ATTACK_STYLE_RANGED) s->tick_scratch.styles_fired |= 2;
    else if (actual_style == ATTACK_STYLE_MAGIC) s->tick_scratch.styles_fired |= 4;
    s->tick_scratch.attacks_fired++;
    s->attacks_by_type[npc->type]++;

    if (npc->type == INF_NPC_BAT) {
        s->player.run_energy -= 300;
        if (s->player.run_energy < 0) s->player.run_energy = 0;
    }

    if (hit_delay == 0) {
        int prayer_matches = encounter_prayer_correct_for_style(s->player.prayer, actual_style);
        if (prayer_matches) {
            dmg = 0;
            s->tick_scratch.prayer_correct++;
            s->prayer_correct_by_type[npc->type]++;
        } else if (dmg > 0) {
            s->tick_scratch.off_prayer_hits++;
        }
        int elysian_reduced = 0;
        dmg = inf_apply_elysian_to_player_hit(s, dmg, &elysian_reduced);
        if (elysian_reduced) s->player.elysian_proc_this_tick = 1;
        inf_apply_echo_boots_recoil(s, idx, dmg);
        s->dmg_from_type[npc->type] += (float)dmg;
        if (dmg > 0) s->last_hit_by_type = npc->type;
        inf_damage_player(s, dmg);
    } else {
        int is_jad = (npc->type == INF_NPC_JAD);
        if (!is_jad) {
            EncounterProtectResolve pr =
                encounter_resolve_protect_at_throw(dmg, s->player.prayer, actual_style);
            dmg = pr.frozen_damage;
            if (pr.prayed) {
                s->tick_scratch.prayer_correct++;
                s->prayer_correct_by_type[npc->type]++;
            } else if (dmg > 0) {
                s->tick_scratch.off_prayer_hits++;
            }
        }
        if (!is_delayed_jad) {
            s->dmg_from_type[npc->type] += (float)dmg;
            if (dmg > 0) s->last_hit_by_type = npc->type;
        }
        if (npc->type == INF_NPC_BAT && dmg > 0 &&
            encounter_rand_int(&s->rng_state, 2) == 0) {
            if (s->player.current_attack > 0) s->player.current_attack--;
            if (s->player.current_strength > 0) s->player.current_strength--;
            if (s->player.current_defence > 0) s->player.current_defence--;
            if (s->player.current_ranged > 0) s->player.current_ranged--;
            if (s->player.current_magic > 0) s->player.current_magic--;
        }
        EncounterPendingHit hit;
        if (is_jad) {
            hit = (EncounterPendingHit){
                .ticks_remaining = (int8_t)hit_delay,
                .attack_style = (int8_t)actual_style,
                .check_prayer = 1,
                .prayer_check_delay = INF_JAD_PROJECTILE_DELAY + 1,
                .spell_type = ENCOUNTER_SPELL_NONE,
                .source_npc_type = (int8_t)npc->type,
                .hit_success = (int8_t)accuracy_hit,
                .elysian_reduced = 0,
                .damage = (int16_t)dmg,
            };
        } else {
            hit = encounter_pending_hit_resolved_at_throw(
                dmg, hit_delay, actual_style, s->player.prayer,
                npc->type, 0 , accuracy_hit, NULL);
        }
        inf_queue_player_pending_hit(s, idx, npc->type, hit);
    }

    npc->attacked_this_tick = 1;
    npc->attack_style_this_tick = actual_style;
    if (npc->type == INF_NPC_JAD)
        inf_npc_jad(npc)->attack_style = ATTACK_STYLE_NONE;
    npc->attack_timer = stats->attack_speed;

    if (npc->type == INF_NPC_JAD)
        npc->attack_timer = inf_jad_attack_timer(s->wave);
}

static void inf_npc_attack(InfernoState* s, int idx) {
    inf_npc_attack_ctx(s, inf_legacy_context(), idx);
}

static int inf_find_mager_respawn_tile(
    InfernoState* s, const InfernoContext* ctx, int size, int* out_x, int* out_y
) {
    InfMoveCtx mc = { s, ctx, -1 };
    for (int x = 26; x < 33; x++) {
        for (int y = 24; y < 37; y++) {
            if (inf_npc_blocked(&mc, x, y, size)) continue;
            if (encounter_entity_footprints_overlap(x, y, size,
                                                    s->player.x, s->player.y, 1))
                continue;
            *out_x = x;
            *out_y = y;
            return 1;
        }
    }

    *out_x = 21;
    *out_y = 22;
    return !inf_npc_blocked(&mc, *out_x, *out_y, size);
}

static int inf_mager_resurrect_ctx(InfernoState* s, InfernoContext* ctx, int idx) {
    InfNPC* npc = &s->npcs[idx];
    if (npc->type != INF_NPC_MAGER || !npc->active) return 0;
    if (s->wave >= 68) return 0;
    if (npc->resurrect_cooldown > 0) return 0;
    if (s->dead_mob_count == 0) return 0;

    if (encounter_rand_int(&s->rng_state, 10) != 0) return 0;

    int di = encounter_rand_int(&s->rng_state, s->dead_mob_count);
    InfDeadMob* dm = &s->dead_mobs[di];

    int slot = inf_find_free_npc(s);
    if (slot < 0) return 0;

    int rx = 21;
    int ry = 22;
    if (!inf_find_mager_respawn_tile(s, ctx, INF_NPC_STATS[dm->type].size, &rx, &ry))
        return 0;

    inf_init_npc(s, slot, dm->type, rx, ry);
    s->npcs[slot].hp = dm->hp;
    s->npcs[slot].max_hp = dm->max_hp;
    s->npcs[slot].resurrection_count = 1;
    s->npcs[slot].attack_timer = INF_NPC_STATS[dm->type].attack_speed;

    s->dead_mobs[di] = s->dead_mobs[s->dead_mob_count - 1];
    s->dead_mob_count--;

    npc->resurrect_cooldown = 8;
    npc->resurrecting_this_tick = 1;
    npc->resurrection_visual_target = slot;
    npc->attack_visual_target = slot;
    return 1;
}

static int inf_mager_resurrect(InfernoState* s, int idx) {
    return inf_mager_resurrect_ctx(s, inf_legacy_context(), idx);
}

#define INF_JAD_HEALER_MAX_SPAWN_CANDIDATES 165

static int inf_try_sample_jad_healer_spawn(
    InfernoState* s, const InfernoContext* ctx, const InfNPC* jad,
    int min_dx, int max_dx, int min_dy, int max_dy,
    int* out_x, int* out_y
) {
    int offsets[INF_JAD_HEALER_MAX_SPAWN_CANDIDATES][2];
    int order[INF_JAD_HEALER_MAX_SPAWN_CANDIDATES];
    int count = 0;
    for (int dx = min_dx; dx <= max_dx; dx++) {
        for (int dy = min_dy; dy <= max_dy; dy++) {
            assert(count < INF_JAD_HEALER_MAX_SPAWN_CANDIDATES);
            offsets[count][0] = dx;
            offsets[count][1] = dy;
            order[count] = count;
            count++;
        }
    }
    encounter_shuffle(order, count, &s->rng_state);

    for (int i = 0; i < count; i++) {
        int hx = jad->x + offsets[order[i]][0];
        int hy = jad->y + offsets[order[i]][1];
        if (encounter_entity_footprints_overlap(hx, hy, 1, jad->x, jad->y, jad->size))
            continue;
        if (inf_npc_environment_blocked_ctx(s, ctx, hx, hy, 1))
            continue;
        *out_x = hx;
        *out_y = hy;
        return 1;
    }

    return 0;
}

static void inf_sample_jad_healer_spawn(
    InfernoState* s,
    const InfernoContext* ctx,
    const InfNPC* jad,
    int* out_x,
    int* out_y
) {
    if (s->wave == 68 &&
        inf_try_sample_jad_healer_spawn(s, ctx, jad, 0, 5, 5, 8, out_x, out_y))
        return;

    if (inf_try_sample_jad_healer_spawn(s, ctx, jad, -5, 5, -4, 10, out_x, out_y))
        return;

    fprintf(stderr, "FATAL: no valid Jad healer spawn tile for Jad at (%d,%d) on wave %d\n",
            jad->x, jad->y, s->wave + 1);
    abort();
}

static void inf_jad_check_healers_ctx(InfernoState* s, const InfernoContext* ctx, int idx) {
    InfNPC* npc = &s->npcs[idx];
    if (npc->type != INF_NPC_JAD || !npc->active) return;
    if (inf_npc_jad(npc)->healer_spawned) return;

    if (npc->hp >= npc->max_hp / 2) return;
    inf_npc_jad(npc)->healer_spawned = 1;

    int num_healers;
    if (s->wave == 66)      num_healers = 5;
    else if (s->wave == 67) num_healers = 3;
    else                    num_healers = 3;

    for (int h = 0; h < num_healers; h++) {
        int slot = inf_find_free_npc(s);
        if (slot < 0) break;
        int hx = npc->x;
        int hy = npc->y;
        inf_sample_jad_healer_spawn(s, ctx, npc, &hx, &hy);
        inf_init_npc(s, slot, INF_NPC_HEALER_JAD, hx, hy);
        inf_npc_healer(&s->npcs[slot])->owner_idx = idx;
        s->npcs[slot].aggro_target = idx;
    }
}

static void inf_jad_check_healers(InfernoState* s, int idx) {
    inf_jad_check_healers_ctx(s, inf_legacy_context(), idx);
}

static void inf_zuk_tick(InfernoState* s) {
    if (!inf_is_final_wave(s)) return;

    int zuk_idx = inf_find_live_zuk_idx(s);
    if (zuk_idx < 0) return;
    InfNPC* zuk = &s->npcs[zuk_idx];

    int si = s->zuk.shield_idx;
    if (si >= 0 && s->npcs[si].active) {
        InfNPC* shield = &s->npcs[si];
        if (s->zuk.shield_freeze > 0) {
            s->zuk.shield_freeze--;
        } else {
            int old_x = shield->x;
            int old_y = shield->y;
            if (inf_npc_sets_collision_flag(shield->type))
                inf_unstamp_npc_collision_footprint(
                    s, old_x, old_y, shield->size);
            shield->x += s->zuk.shield_dir;
            if (shield->x < 11) {
                s->zuk.shield_freeze = 4;
                s->zuk.shield_dir = 1;
            } else if (shield->x > 35) {
                s->zuk.shield_freeze = 4;
                s->zuk.shield_dir = -1;
            }
            if (inf_npc_sets_collision_flag(shield->type))
                inf_stamp_npc_collision_footprint(
                    s, shield->x, shield->y, shield->size);
        }
    }

    if (!s->zuk.has_paused && zuk->hp < 600) {
        s->zuk.timer_paused = 1;
        s->zuk.has_paused = 1;
    }

    if (!s->zuk.jad_spawned && s->zuk.timer_paused && zuk->hp < 480 &&
        si >= 0 && s->npcs[si].active) {
        s->zuk.jad_spawned = 1;
        s->zuk.timer_paused = 0;
        s->zuk.set_timer += 175;
        inf_spawn_zuk_jad(s, si);
    }

    if (!s->zuk.timer_paused) {
        if (s->zuk.set_timer > 0)
            s->zuk.set_timer--;
        if (s->zuk.set_timer == 0) {
            int m_slot = inf_find_free_npc(s);
            if (m_slot >= 0) {
                inf_init_npc(s, m_slot, INF_NPC_MAGER, 20, 36);
                if (si >= 0) s->npcs[m_slot].aggro_target = si;
                s->npcs[m_slot].stun_timer = 7;
            }
            int r_slot = inf_find_free_npc(s);
            if (r_slot >= 0) {
                inf_init_npc(s, r_slot, INF_NPC_RANGER, 29, 36);
                if (si >= 0) s->npcs[r_slot].aggro_target = si;
                s->npcs[r_slot].stun_timer = 9;
            }
            s->zuk.set_timer = s->zuk.set_interval;
        }
    }

    if (!s->zuk.healer_spawned && zuk->hp < 240) {
        s->zuk.healer_spawned = 1;
        s->zuk.enraged = 1;
        inf_spawn_zuk_healers(s, zuk_idx);
    }

    if (zuk->hp <= 0) {
        for (int i = 0; i < INF_MAX_NPCS; i++) {
            inf_deactivate_npc(s, i);
        }
    }
}

static void inf_healer_apply_landed_heal(InfernoState* s, int idx) {
    InfNPC* npc = &s->npcs[idx];
    InfHealerState* h = inf_npc_healer(npc);
    int target_idx = h->target;
    int heal_cap = (npc->type == INF_NPC_HEALER_ZUK) ? 25 : 20;

    h->target = -1;
    if (target_idx < 0 || target_idx >= INF_MAX_NPCS) return;
    if (!s->npcs[target_idx].active) return;

    int heal = encounter_rand_int(&s->rng_state, heal_cap);
    int before = s->npcs[target_idx].hp;
    s->npcs[target_idx].hp += heal;
    if (s->npcs[target_idx].hp > s->npcs[target_idx].max_hp)
        s->npcs[target_idx].hp = s->npcs[target_idx].max_hp;
    float effective_heal = (float)(s->npcs[target_idx].hp - before);
    s->tick_scratch.hp_restored += effective_heal;
    if (s->npcs[target_idx].type == INF_NPC_JAD)
        s->tick_scratch.hp_restored_jad += effective_heal;
    if (s->npcs[target_idx].type == INF_NPC_ZUK)
        s->tick_scratch.hp_restored_zuk += effective_heal;
}

static int inf_redemption_threshold(const InfernoState* s) {
    return (s->player.base_hitpoints - 1) / 10;
}

static void inf_apply_redemption_if_due(InfernoState* s, int before_hp) {
    Player* p = &s->player;
    if (p->prayer != PRAYER_REDEMPTION || p->current_prayer <= 0)
        return;
    if (p->current_hitpoints <= 0)
        return;

    int threshold = inf_redemption_threshold(s);
    int before_band = before_hp > 0 && before_hp <= threshold;
    int after_band = p->current_hitpoints > 0 && p->current_hitpoints <= threshold;
    if (!before_band && !after_band)
        return;

    int old_offensive = p->offensive_prayer;
    int heal = p->base_prayer / 4;
    p->current_hitpoints += heal;
    if (p->current_hitpoints > p->base_hitpoints)
        p->current_hitpoints = p->base_hitpoints;
    p->current_prayer = 0;
    p->prayer = PRAYER_NONE;
    p->offensive_prayer = OFFENSIVE_PRAYER_NONE;
    p->prayer_drain_counter = 0;
    p->prayer_just_activated = 0;
    p->offensive_prayer_just_activated = 0;
    if (old_offensive != OFFENSIVE_PRAYER_NONE)
        encounter_recompute_loadout_max_hits(s->loadout_stats, INF_NUM_WEAPON_SETS, p);
}

static void inf_damage_player(InfernoState* s, int damage) {
    int before_hp = s->player.current_hitpoints;
    encounter_damage_player(&s->player, damage, &s->tick_scratch.damage_received);
    inf_apply_redemption_if_due(s, before_hp);
}

static void inf_queue_pending_spark(
    InfernoState* s, int src_x, int src_y, int x, int y, int damage, int ticks_remaining
) {
    for (int i = 0; i < INF_MAX_PENDING_SPARKS; i++) {
        if (s->pending_sparks[i].active) continue;
        s->pending_sparks[i].active = 1;
        s->pending_sparks[i].src_x = src_x;
        s->pending_sparks[i].src_y = src_y;
        s->pending_sparks[i].x = x;
        s->pending_sparks[i].y = y;
        s->pending_sparks[i].damage = damage;
        s->pending_sparks[i].ticks_remaining = ticks_remaining;
        s->pending_sparks[i].visual_emitted = 0;
        return;
    }
    fprintf(stderr, "inferno: pending spark queue overflow at %d slots\n",
        INF_MAX_PENDING_SPARKS);
    abort();
}

static void inf_queue_zuk_healer_sparks(InfernoState* s, const InfNPC* npc) {
    int clamped_x = s->player.x;
    if (clamped_x < npc->x - 5) clamped_x = npc->x - 5;
    if (clamped_x > npc->x + 4) clamped_x = npc->x + 4;

    int direct_damage = 5 + encounter_rand_int(&s->rng_state, 6);
    inf_queue_pending_spark(s, npc->x, npc->y, clamped_x, s->player.y,
                            direct_damage, 3);

    int spark1_x = npc->x + encounter_rand_int(&s->rng_state, 11) - 5;
    int spark1_y = 40 + encounter_rand_int(&s->rng_state, 4);
    int spark1_damage = 5 + encounter_rand_int(&s->rng_state, 6);
    inf_queue_pending_spark(s, npc->x, npc->y, spark1_x, spark1_y,
                            spark1_damage, 3);

    int spark2_x = npc->x + encounter_rand_int(&s->rng_state, 11) - 5;
    int spark2_y = 40 + encounter_rand_int(&s->rng_state, 4);
    int spark2_damage = 5 + encounter_rand_int(&s->rng_state, 6);
    inf_queue_pending_spark(s, npc->x, npc->y, spark2_x, spark2_y,
                            spark2_damage, 3);
}

static void inf_resolve_pending_sparks(InfernoState* s) {
    for (int i = 0; i < INF_MAX_PENDING_SPARKS; i++) {
        if (!s->pending_sparks[i].active) continue;
        s->pending_sparks[i].ticks_remaining--;
        if (s->pending_sparks[i].ticks_remaining > 0) continue;

        if (encounter_entity_footprints_overlap(
                s->pending_sparks[i].x - 1, s->pending_sparks[i].y - 1, 3,
                s->player.x, s->player.y, 1)) {
            inf_damage_player(s, s->pending_sparks[i].damage);
            s->tick_scratch.spark_damage += (float)s->pending_sparks[i].damage;
            s->last_hit_by_type = INF_NPC_HEALER_ZUK;
        }
        s->pending_sparks[i].active = 0;
    }
}

static void inf_tick_npcs_ctx(InfernoState* s, InfernoContext* ctx) {

    inf_resolve_npc_target_hits(s);
    inf_zuk_tick(s);

    int blob_split_x[INF_MAX_NPCS];
    int blob_split_y[INF_MAX_NPCS];
    int blob_split_parent[INF_MAX_NPCS];
    int blob_split_count = 0;

    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (!s->npcs[i].active) continue;

        if ((s->npcs[i].type == INF_NPC_HEALER_JAD || s->npcs[i].type == INF_NPC_HEALER_ZUK) &&
            s->npcs[i].death_ticks == 0) {
            InfHealerState* h = inf_npc_healer(&s->npcs[i]);
            if (h->timer > 0) {
                h->timer--;
                if (h->timer == 0)
                    inf_healer_apply_landed_heal(s, i);
            }
        }

        if (s->npcs[i].death_ticks > 0) {
            if (osrs_npc_death_linger_tick(&s->npcs[i].death_ticks)) {
                if (s->npcs[i].spawn_blob_splits_on_removal &&
                    blob_split_count < INF_MAX_NPCS) {
                    blob_split_x[blob_split_count] = s->npcs[i].x;
                    blob_split_y[blob_split_count] = s->npcs[i].y;
                    blob_split_parent[blob_split_count] = i;
                    blob_split_count++;
                }
                inf_deactivate_npc(s, i);
            }
            continue;
        }

        if (s->npcs[i].frozen_ticks > 0) s->npcs[i].frozen_ticks--;

        if (s->npcs[i].type == INF_NPC_MAGER && s->npcs[i].resurrect_cooldown > 0)
            s->npcs[i].resurrect_cooldown--;

        if (s->npcs[i].type == INF_NPC_MELEER)
            inf_meleer_dig_check_ctx(s, ctx, i);

        inf_npc_move_ctx(s, ctx, i);
        inf_npc_attack_ctx(s, ctx, i);

        if (s->npcs[i].type == INF_NPC_JAD)
            inf_jad_check_healers_ctx(s, ctx, i);
    }

    for (int i = 0; i < blob_split_count; i++) {
        InfNPCType split_types[3] = {
            INF_NPC_BLOB_RANGE, INF_NPC_BLOB_MELEE, INF_NPC_BLOB_MAGE
        };
        int split_offsets[3][2] = {
            {1, 1}, {0, 0}, {2, 2}
        };
        for (int sp = 0; sp < 3; sp++) {
            int slot = -1;
            for (int n = 0; n < INF_MAX_NPCS; n++) {
                if (n == blob_split_parent[i] || s->npcs[n].active) continue;
                slot = n;
                break;
            }
            if (slot < 0) break;
            inf_init_npc(
                s, slot, split_types[sp],
                blob_split_x[i] + split_offsets[sp][0],
                blob_split_y[i] + split_offsets[sp][1]);
            s->npcs[slot].attack_timer = 4;
        }
    }
}

static void inf_tick_npcs(InfernoState* s) {
    inf_tick_npcs_ctx(s, inf_legacy_context());
}
