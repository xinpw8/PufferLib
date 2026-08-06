#line 76 "encounter_colosseum.h"

static void col_apply_npc_death(ColosseumState* s, int idx) {
    ColoNPC* npc = &s->npcs[idx];

    if (col_type_is_hazard_entity(npc->type)) {
        col_hazard_on_npc_death(s, idx);
        col_deactivate_npc(s, idx);
        return;
    }
    if (npc->death_ticks > 0) return;
    assert(npc->active && npc->hp <= 0);
    s->log.total_npc_kills++;

    if (s->death_charge_window_left > 0) {
        s->player.special_energy += COLO_DEATH_CHARGE_SPEC_GAIN;
        if (s->player.special_energy > 100) s->player.special_energy = 100;
        s->death_charge_window_left = 0;
        s->death_charge_cd = COLO_DEATH_CHARGE_CD;
    }

    col_mod_volatility_on_death(s, npc->x, npc->y, col_npc_effective_size(npc));
    col_mod_on_owner_death(s, idx);
    if (!osrs_npc_death_linger_start(
            npc->hp, npc->active, &npc->death_ticks,
            col_npc_death_linger_ticks(npc->type)))
        return;
    encounter_pending_hit_queue_clear(&npc->pending_hits);
}

static void col_render_hit_events_add(
    int* count,
    int damage[ENCOUNTER_RENDER_HITS_MAX],
    int hit_damage
) {
    if (!count || !damage || *count < 0 || *count >= ENCOUNTER_RENDER_HITS_MAX) {
        fprintf(stderr, "invalid colosseum render hit event\n");
        abort();
    }
    damage[*count] = hit_damage > 0 ? hit_damage : 0;
    *count += 1;
}

static void col_apply_echo_boots_recoil(
    ColosseumState* s, ColosseumContext* ctx, int attacker_slot, int final_damage
) {
    int recoil_damage = osrs_echo_boots_recoil_damage(
        col_live_effects(s), &s->player.item_effect_state, final_damage);
    if (recoil_damage <= 0) return;
    if (attacker_slot < 0 || attacker_slot >= COLO_MAX_NPCS) return;

    ColoNPC* npc = &s->npcs[attacker_slot];
    if (!col_npc_is_live_enemy(npc)) return;
    if (npc->type == COLO_SOL_HEREDIT) return;

    osrs_consume_echo_boots_charge(&s->player);
    int landed = 0, hit_damage = 0;
    int prev_min_hp = npc->min_hp_seen;
    int applied = encounter_damage_npc(
        &npc->hp, &landed, &hit_damage, recoil_damage);
    if (applied <= 0) return;

    npc->hit_landed_this_tick = landed;
    npc->hit_damage += hit_damage;
    npc->hit_was_successful_this_tick = 1;
    col_render_hit_events_add(
        &ctx->npc_render_hit_count[attacker_slot],
        ctx->npc_render_hit_damage[attacker_slot],
        hit_damage);
    s->tick_scratch.damage_dealt += (float)applied;
    float fresh = (float)(prev_min_hp - npc->hp);
    if (fresh > 0.0f) {
        npc->min_hp_seen = npc->hp;
        s->tick_scratch.fresh_damage_dealt += fresh;
        if (npc->spawned_as_reinforcement)
            s->tick_scratch.fresh_damage_reinforcement += fresh;
        s->current_wave_fresh_damage += fresh;
    }
    if (npc->hp <= 0) {
        col_apply_npc_death(s, attacker_slot);
    } else {
        col_mod_on_npc_hp_changed(s, attacker_slot);
    }
}

static void col_resolve_player_projectiles_on_npcs_ctx(
    ColosseumState* s,
    ColosseumContext* ctx
) {
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        ColoNPC* npc = &s->npcs[i];
        if (!col_npc_is_live_target(npc)) continue;
        for (int h = 0; h < npc->pending_hits.count; h++) {
            EncounterPendingHit* ph = &npc->pending_hits.hits[h];
            int landed = 0, hit_damage = 0;
            float dmg_acc = 0.0f;
            int frozen = npc->frozen_ticks;
            int prev_min_hp = npc->min_hp_seen;
            if (encounter_resolve_npc_pending_hit(
                    ph, &npc->hp, &landed, &hit_damage, &frozen, NULL, &dmg_acc)) {
                npc->hit_landed_this_tick = landed;
                npc->hit_damage = hit_damage;
                npc->hit_was_successful_this_tick = (hit_damage > 0);
                col_render_hit_events_add(
                    &ctx->npc_render_hit_count[i],
                    ctx->npc_render_hit_damage[i],
                    hit_damage);
                s->tick_scratch.damage_dealt += dmg_acc;
                if (npc->type == COLO_SOL_HEREDIT)
                    s->tick_scratch.damage_boss += dmg_acc;
                float fresh = (float)(prev_min_hp - npc->hp);
                if (fresh > 0.0f) {
                    npc->min_hp_seen = npc->hp;
                    s->tick_scratch.fresh_damage_dealt += fresh;
                    if (npc->spawned_as_reinforcement)
                        s->tick_scratch.fresh_damage_reinforcement += fresh;
                    if (npc->type == COLO_SOL_HEREDIT)
                        s->tick_scratch.fresh_damage_boss += fresh;
                    else
                        s->current_wave_fresh_damage += fresh;
                }
                if (npc->hp <= 0) {
                    col_apply_npc_death(s, i);
                    break;
                }
                col_mod_on_npc_hp_changed(s, i);
                encounter_pending_hit_queue_remove(&npc->pending_hits, h, "colosseum-npc");
                h--;
            }
        }
    }
}

static void col_resolve_player_projectiles_on_npcs(ColosseumState* s) {
    col_resolve_player_projectiles_on_npcs_ctx(s, col_legacy_context());
}

static void col_tick_npc_death_lingers(ColosseumState* s) {
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        ColoNPC* npc = &s->npcs[i];
        if (!npc->active || npc->death_ticks <= 0) continue;
        if (osrs_npc_death_linger_tick(&npc->death_ticks))
            col_deactivate_npc(s, i);
    }
}

static void col_log_prayer_event(ColosseumState* s, int npc_type, int prayed, int dmg) {
    assert(npc_type >= 0 && npc_type < COLO_NUM_NPC_TYPES);
    s->log.pray_faced_by_type[npc_type] += 1.0f;
    if (prayed) s->log.pray_correct_by_type[npc_type] += 1.0f;
    else s->log.offpray_damage_by_type[npc_type] += (float)dmg;
}

static inline int col_damage_player_from(
    ColosseumState* s, int dmg, int npc_type, ColoDamageChannel channel
) {
    dmg = col_scale_incoming_damage(s, dmg);
    int hitpoints_before = s->player.current_hitpoints;
    encounter_damage_player(&s->player, dmg, NULL);
    int applied = hitpoints_before - s->player.current_hitpoints;
    s->tick_scratch.damage_received += (float)applied;
    switch (channel) {
        case COLO_DMG_OFFPRAY:
            s->tick_scratch.landed_offpray_damage += (float)applied;
            break;
        case COLO_DMG_UNPRAYABLE:
            s->tick_scratch.landed_unprayable_damage += (float)applied;
            if (applied > 0 && npc_type >= 0 && npc_type < COLO_NUM_NPC_TYPES)
                s->log.typeless_damage_by_type[npc_type] += (float)applied;
            break;
    }
    if (applied > 0 && npc_type >= 0 && npc_type < COLO_NUM_NPC_TYPES)
        s->log.total_damage_by_type[npc_type] += (float)applied;
    return applied;
}

static inline int col_damage_player_from_sol(
    ColosseumState* s, int dmg, ColoSolDamageSource source
) {
    assert(source >= 0 && source < COLO_NUM_SOL_DAMAGE_SOURCES);
    ColoDamageChannel channel = source == COLO_SOL_DAMAGE_TRIPLE_PARRY
        ? COLO_DMG_OFFPRAY
        : COLO_DMG_UNPRAYABLE;
    int hitpoints_before = s->player.current_hitpoints;
    int applied =
        col_damage_player_from(s, dmg, COLO_SOL_HEREDIT, channel);
    s->log.sol_damage_by_source[source] +=
        (float)(hitpoints_before - s->player.current_hitpoints);
    return applied;
}

static ColoDamageSource col_damage_source_from_sol(
    ColoSolDamageSource source
) {
    switch (source) {
        case COLO_SOL_DAMAGE_SPEAR_1:
            return COLO_DAMAGE_SOL_SPEAR_1;
        case COLO_SOL_DAMAGE_SPEAR_2:
            return COLO_DAMAGE_SOL_SPEAR_2;
        case COLO_SOL_DAMAGE_SHIELD_1:
            return COLO_DAMAGE_SOL_SHIELD_1;
        case COLO_SOL_DAMAGE_SHIELD_2:
            return COLO_DAMAGE_SOL_SHIELD_2;
        case COLO_SOL_DAMAGE_TRIPLE_PARRY:
            return COLO_DAMAGE_SOL_TRIPLE_PARRY;
        case COLO_SOL_DAMAGE_GRAPPLE:
            return COLO_DAMAGE_SOL_GRAPPLE;
        case COLO_SOL_DAMAGE_CRYSTAL_LASER:
            return COLO_DAMAGE_SOL_CRYSTAL_LASER;
        case COLO_SOL_DAMAGE_MOLTEN_SAND:
            return COLO_DAMAGE_SOL_MOLTEN_SAND;
        case COLO_NUM_SOL_DAMAGE_SOURCES:
            break;
    }
    abort();
}

typedef struct {
    ColosseumState* state;
    ColosseumContext* context;
} ColPendingHitObserverContext;

static void col_pending_hit_prayer_observer(
    void* user, const EncounterPendingHit* hit, int dmg, int applied,
    int prayed, int checked
) {
    ColPendingHitObserverContext* observer_ctx = (ColPendingHitObserverContext*)user;
    ColosseumState* s = observer_ctx->state;
    ColosseumContext* ctx = observer_ctx->context;
    col_render_hit_events_add(
        &ctx->player_render_hit_count, ctx->player_render_hit_damage, dmg);
    if (applied > 0 &&
            hit->source_npc_type >= 0 &&
            hit->source_npc_type < COLO_NUM_NPC_TYPES) {
        s->log.total_damage_by_type[hit->source_npc_type] += (float)applied;

        s->last_hit_by_type = hit->source_npc_type;
        if (hit->source_npc_type == COLO_JAVELIN_COLOSSUS) {
            s->log.javelin_damage_by_source[COLO_JAVELIN_DAMAGE_BASIC_RANGED] +=
                (float)applied;
        }
        ColoDamageSource source =
            hit->source_npc_type == COLO_JAVELIN_COLOSSUS
                ? COLO_DAMAGE_JAVELIN_BASIC_RANGED
                : COLO_DAMAGE_NPC_ATTACK;
        s->last_damage_source = source;
        col_mod_add_doom_stack(s, source);
    }

    col_apply_echo_boots_recoil(s, ctx, hit->source_npc_slot, dmg);

    if (applied > 0) {
        if (checked && prayed)
            s->tick_scratch.landed_prayed_damage += (float)applied;
        else
            s->tick_scratch.landed_offpray_damage += (float)applied;
    }
    if (!checked && hit->source_npc_type == COLO_MANTICORE) {
        if (hit->hit_success) col_mod_manticore_apply_venom(s, hit->hit_success);
    }
    if (!checked) return;
    s->tick_scratch.styles_faced_mask |= (1 << hit->attack_style);
    s->tick_scratch.avoidable_threat_by_style[hit->attack_style] += (float)hit->damage;
    if (!prayed) s->tick_scratch.offpray_damage_this_tick += (float)dmg;
    col_log_prayer_event(s, hit->source_npc_type, prayed, dmg);
}

static void col_resolve_player_pending_hits_ctx(ColosseumState* s, ColosseumContext* ctx) {
    int prayer_correct = 0;
    float damage_received = 0.0f;
    ColPendingHitObserverContext observer_ctx = {
        .state = s,
        .context = ctx,
    };
    encounter_resolve_player_pending_hits_observed(
        &s->player_pending_hits, &s->player, s->player.prayer,
        &damage_received, &prayer_correct, NULL,
        col_pending_hit_prayer_observer, &observer_ctx);
    int applied = (int)damage_received;
    s->tick_scratch.damage_received += damage_received;
    s->tick_scratch.prayer_correct += prayer_correct;

    int blas = col_mod_tier(s, COLO_MOD_BLASPHEMY);
    if (blas > 0 && applied > 0 && s->player.current_prayer > 0) {
        int drain = (applied * COLO_BLASPHEMY_DRAIN_PCT[blas]) / 100;
        s->player.current_prayer -= drain;
        if (s->player.current_prayer < 0) s->player.current_prayer = 0;
    }
}

static void col_resolve_player_pending_hits(ColosseumState* s) {
    col_resolve_player_pending_hits_ctx(s, col_legacy_context());
}

static int col_npc_dist_to_player(const ColosseumState* s, const ColoNPC* npc) {
    return encounter_dist_to_npc(
        s->player.x, s->player.y, npc->x, npc->y, col_npc_effective_size(npc));
}

static int col_npc_melee_adjacent_to_player(const ColosseumState* s, const ColoNPC* npc) {
    return entity_has_line_of_sight(
        NULL, 0, npc->x, npc->y, col_npc_effective_size(npc),
        s->player.x, s->player.y, 1, 1);
}

static int col_player_def_roll(ColosseumState* s, AttackStyle style, MeleeStyle melee_style) {
    const EncounterLoadoutStats* ls = col_live_loadout_stats(s);
    int def_level = col_mod_relentless_def_level(s, s->player.current_defence);
    return encounter_player_def_roll_from_loadout(
        def_level, s->player.current_magic,
        ls->def_stab, ls->def_slash, ls->def_crush, ls->def_magic, ls->def_ranged,
        style, melee_style);
}

static int col_npc_roll_vs_player(
    ColosseumState* s, const ColoNpcStats* stats, AttackStyle style, int max_hit,
    int* hit_success
) {
    int att_level, att_bonus;
    if (style == ATTACK_STYLE_MELEE) {
        att_level = stats->att_level; att_bonus = stats->melee_att_bonus;
    } else if (style == ATTACK_STYLE_RANGED) {
        att_level = stats->range_level; att_bonus = stats->range_att_bonus;
    } else {
        att_level = stats->magic_level; att_bonus = stats->magic_att_bonus;
    }
    int att_roll = osrs_npc_attack_roll(att_level, att_bonus);
    int def_roll = col_player_def_roll(s, style, stats->melee_style);

    int force_hit = 0;
    int max_hit_bonus = col_mod_relentless_max_hit_bonus(s, &force_hit);
    int effective_max = max_hit + max_hit_bonus;

    return encounter_npc_roll_attack_ex(
        att_roll, def_roll, effective_max, force_hit, &s->rng_state, hit_success);
}

static int col_npc_projectile_delay(AttackStyle style, int distance) {
    EncounterProjectileDelayOptions opts = {0};
    EncounterProjectileTiming timing = encounter_projectile_timing(
        distance, 0, encounter_projectile_delay_kind_for_style(style), opts);
    return timing.damage_delay_ticks;
}

static void col_npc_queue_projectile(
    ColosseumState* s, int idx, const ColoNpcStats* stats, AttackStyle style, int max_hit
) {
    ColoNPC* npc = &s->npcs[idx];
    int dist = col_npc_dist_to_player(s, npc);
    int hit_success = 0;
    int dmg = col_npc_roll_vs_player(s, stats, style, max_hit, &hit_success);
    int rolled_dmg = dmg;
    int delay = col_npc_projectile_delay(style, dist);
    int prayed = 0;
    EncounterPendingHit hit = encounter_pending_hit_resolved_at_throw(
        dmg, delay, style, s->player.prayer, npc->type, idx, hit_success, &prayed);
    dmg = hit.damage;
    if (prayed) s->tick_scratch.prayer_correct++;
    col_log_prayer_event(s, npc->type, prayed, dmg);
    s->tick_scratch.styles_faced_mask |= (1 << style);
    s->tick_scratch.avoidable_threat_by_style[style] += (float)rolled_dmg;
    if (!prayed) s->tick_scratch.offpray_damage_this_tick += (float)dmg;
    col_push_player_pending_hit(s, hit);
    npc->attacked_this_tick = 1;
    npc->attack_style_this_tick = style;
    s->tick_scratch.attacks_fired++;
}

static void col_apply_instant_melee_hit(
    ColosseumState* s, ColosseumContext* ctx, int attacker_slot, int npc_type,
    int dmg, int hit_success
) {
    (void)hit_success;
    int rolled_dmg = dmg;
    EncounterProtectResolve pr = encounter_resolve_protect_at_throw(dmg, s->player.prayer, ATTACK_STYLE_MELEE);
    int prayed = pr.prayed;
    dmg = pr.frozen_damage;
    if (prayed) s->tick_scratch.prayer_correct++;
    col_render_hit_events_add(
        &ctx->player_render_hit_count, ctx->player_render_hit_damage, dmg);
    s->tick_scratch.styles_faced_mask |= (1 << ATTACK_STYLE_MELEE);
    s->tick_scratch.avoidable_threat_by_style[ATTACK_STYLE_MELEE] += (float)rolled_dmg;
    if (!prayed) s->tick_scratch.offpray_damage_this_tick += (float)dmg;
    col_log_prayer_event(s, npc_type, prayed, dmg);

    int applied =
        col_damage_player_from(s, dmg, npc_type, COLO_DMG_OFFPRAY);
    if (applied > 0) {
        col_mod_on_player_damaged(
            s, applied, COLO_DAMAGE_NPC_ATTACK);
        s->last_hit_by_type = npc_type;
    }
    col_apply_echo_boots_recoil(s, ctx, attacker_slot, dmg);
}

static void col_npc_melee_attack(
    ColosseumState* s, ColosseumContext* ctx, int idx, const ColoNpcStats* stats, int max_hit
) {
    ColoNPC* npc = &s->npcs[idx];
    int hit_success = 0;
    int dmg = col_npc_roll_vs_player(s, stats, ATTACK_STYLE_MELEE, max_hit, &hit_success);
    col_apply_instant_melee_hit(s, ctx, idx, npc->type, dmg, hit_success);
    npc->attacked_this_tick = 1;
    npc->attack_style_this_tick = ATTACK_STYLE_MELEE;
    s->tick_scratch.attacks_fired++;
}

static void col_npc_attack_jaguar(
    ColosseumState* s, ColosseumContext* ctx, int idx, const ColoNpcStats* stats
) {
    for (int h = 0; h < 3; h++)
        col_npc_melee_attack(s, ctx, idx, stats, stats->max_hit);
}

static void col_npc_minotaur_heal_ally(ColosseumState* s, int idx) {
    ColoNPC* mino = &s->npcs[idx];
    int mcx = mino->x + 1, mcy = mino->y + 1;
    int best = -1;
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        if (i == idx) continue;
        ColoNPC* other = &s->npcs[i];
        if (!col_npc_is_live_target(other)) continue;
        if (other->type == COLO_MINOTAUR) continue;
        if (other->hp * 4 >= other->max_hp * 3) continue;
        int osz = col_npc_effective_size(other);
        int ocx = other->x + osz / 2, ocy = other->y + osz / 2;
        int dx = ocx > mcx ? ocx - mcx : mcx - ocx;
        int dy = ocy > mcy ? ocy - mcy : mcy - ocy;
        if ((dx > dy ? dx : dy) > COLO_MINOTAUR_HEAL_RANGE) continue;
        if (!col_tiles_have_los(s, mcx, mcy, ocx, ocy)) continue;
        if (best < 0 ||
            other->hp * s->npcs[best].max_hp < s->npcs[best].hp * other->max_hp)
            best = i;
    }
    if (best >= 0) s->npcs[best].hp = s->npcs[best].max_hp;
}

static void col_npc_attack_javelin(
    ColosseumState* s,
    ColosseumContext* ctx,
    int idx,
    const ColoNpcStats* stats
) {
    assert(ctx);
    ctx->npc_render_attack_anim[idx] = COLO_NPC_RENDER_ATTACK_NORMAL;
    ColoNPC* npc = &s->npcs[idx];
    ColoJavelinState* jv = colo_npc_javelin(npc);
    jv->attack_count++;
    if (jv->attack_count % 5 == 0) {
        jv->skyfall_pending = 1;
        jv->skyfall_tile_x = s->player.x;
        jv->skyfall_tile_y = s->player.y;
        jv->skyfall_timer = COLO_JAVELIN_SKYFALL_DELAY;
        jv->skyfall_damage = encounter_rand_int(
            &s->rng_state, COLO_JAVELIN_SKYFALL_MAX_HIT + 1);
        npc->attacked_this_tick = 1;
        npc->attack_style_this_tick = ATTACK_STYLE_RANGED;
        ctx->npc_render_attack_anim[idx] = COLO_NPC_RENDER_ATTACK_JAVELIN_SKYFALL;
        s->tick_scratch.attacks_fired++;
        return;
    }
    col_npc_queue_projectile(s, idx, stats, ATTACK_STYLE_RANGED, stats->max_hit);
}

static void col_npc_resolve_javelin_skyfall(
    ColosseumState* s,
    ColosseumContext* ctx,
    int idx
) {
    ColoNPC* npc = &s->npcs[idx];
    ColoJavelinState* jv = colo_npc_javelin(npc);
    if (!jv->skyfall_pending) return;
    jv->skyfall_timer--;
    if (jv->skyfall_timer > 0) return;
    jv->skyfall_pending = 0;
    int on_tile = (s->player.x == jv->skyfall_tile_x && s->player.y == jv->skyfall_tile_y);
    int dmg = on_tile ? jv->skyfall_damage : 0;
    int hitpoints_before = s->player.current_hitpoints;
    int applied =
        col_damage_player_from(s, dmg, npc->type, COLO_DMG_UNPRAYABLE);
    s->log.javelin_damage_by_source[COLO_JAVELIN_DAMAGE_SKYFALL] +=
        (float)(hitpoints_before - s->player.current_hitpoints);
    if (applied > 0) {
        s->last_hit_by_type = npc->type;
        col_mod_on_player_damaged(
            s, applied, COLO_DAMAGE_JAVELIN_SKYFALL);
    }
    col_mod_reentry_on_skyfall(s, jv->skyfall_tile_x, jv->skyfall_tile_y);
}

static void col_manticore_delay_ready_peers(ColosseumState* s, int idx) {
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        if (i == idx) continue;
        ColoNPC* other = &s->npcs[i];
        if (!col_npc_is_live_target(other) || other->type != COLO_MANTICORE) continue;
        if (other->attack_timer > 1) continue;
        if (other->type_state.manticore.cycle_step > 0) continue;
        if (other->stun_timer > 0 || other->frozen_ticks > 0) continue;
        int dist = col_npc_dist_to_player(s, other);
        if (dist < 1 || dist > COLO_NPC_STATS[COLO_MANTICORE].attack_range) continue;
        if (!col_npc_has_los_to_player(s, other)) continue;
        other->attack_timer = COLO_MANTICORE_STAGGER_TICKS + 1;
    }
}

static int col_manticore_orb_max_hit(AttackStyle style) {
    if (style == ATTACK_STYLE_RANGED) return COLO_MANTICORE_MAX_HIT_RANGED;
    if (style == ATTACK_STYLE_MAGIC) return COLO_MANTICORE_MAX_HIT_MAGIC;
    return COLO_MANTICORE_MAX_HIT_MELEE;
}

static void col_npc_manticore_fire_orb(ColosseumState* s, int idx, const ColoNpcStats* stats) {
#ifdef COLO_PROFILE_ENABLED
    int col_prof_enabled =
        COLO_PROFILE_ENABLED() && !s->profiling_suppressed;
    double col_prof_t0 = col_prof_enabled ? COLO_PROFILE_NOW_MS() : 0.0;
#endif
    ColoNPC* npc = &s->npcs[idx];
    ColoManticoreState* mc = colo_npc_manticore(npc);
    AttackStyle style = mc->orb_style[mc->cycle_step];
    int max_hit = col_manticore_orb_max_hit(style);
    int mayhem = col_mod_tier(s, COLO_MOD_MANTIMAYHEM);
    int shots = (mayhem >= 1) ? 2 : 1;
    for (int shot = 0; shot < shots; shot++) {
        int hit_success = 0;
        int dmg = col_npc_roll_vs_player(s, stats, style, max_hit, &hit_success);
        int rolled_dmg = dmg;
        int prayed = 0;
        EncounterPendingHit hit = encounter_pending_hit_resolved_at_throw(
            dmg, 1, style, s->player.prayer, npc->type, idx, hit_success, &prayed);
        dmg = hit.damage;
        if (prayed) s->tick_scratch.prayer_correct++;
        col_log_prayer_event(s, npc->type, prayed, dmg);
        s->tick_scratch.styles_faced_mask |= (1 << style);
        s->tick_scratch.avoidable_threat_by_style[style] += (float)rolled_dmg;
        if (!prayed) s->tick_scratch.offpray_damage_this_tick += (float)dmg;
        col_push_player_pending_hit(s, hit);
    }
    npc->attacked_this_tick = 1;
    npc->attack_style_this_tick = style;
    s->tick_scratch.attacks_fired++;
    mc->cycle_step++;
    if (mc->cycle_step >= 3) {
        mc->cycle_step = -1;
        mc->orb_style[0] = ATTACK_STYLE_NONE;
        mc->orb_style[1] = ATTACK_STYLE_NONE;
        mc->orb_style[2] = ATTACK_STYLE_NONE;

        npc->attack_timer = stats->attack_speed - 2;
    }
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_STEP_MANTICORE_BARRAGE);
#endif
}

static void col_npc_manticore_advance_barrage(ColosseumState* s, int idx, const ColoNpcStats* stats) {
    ColoNPC* npc = &s->npcs[idx];
    if (npc->type_state.manticore.cycle_step < 0) return;
    if (npc->stun_timer > 0 || npc->frozen_ticks > 0) return;
    col_npc_manticore_fire_orb(s, idx, stats);
}

static void col_npc_manticore_arm(ColosseumState* s, int idx) {
    ColoManticoreState* mc = colo_npc_manticore(&s->npcs[idx]);
    if (mc->cycle_step >= 0) return;
    if (mc->orb_style[0] != ATTACK_STYLE_NONE) return;
    for (int o = 0; o < 3; o++) mc->orb_style[o] = mc->fixed_orb_style[o];
    mc->cycle_step = 0;
}

static void col_npc_attack_manticore(ColosseumState* s, int idx, const ColoNpcStats* stats) {
    ColoNPC* npc = &s->npcs[idx];
    ColoManticoreState* mc = colo_npc_manticore(npc);
    mc->cycle_step = 0;
    col_npc_manticore_fire_orb(s, idx, stats);
    col_manticore_delay_ready_peers(s, idx);
}

static void col_npc_attack_ctx(ColosseumState* s, ColosseumContext* ctx, int idx) {
    assert(ctx);
    ColoNPC* npc = &s->npcs[idx];
    const ColoNpcStats* stats = &COLO_NPC_STATS[npc->type];

    if (npc->type == COLO_SOL_HEREDIT) return;

    if (col_type_is_warbander(npc->type)) return;

    if (s->wave_attack_delay > 0) return;

    if (npc->type == COLO_MANTICORE && npc->type_state.manticore.cycle_step > 0) {
        col_npc_manticore_advance_barrage(s, idx, stats);
        return;
    }

    if (npc->attack_timer > 0) npc->attack_timer--;
    if (npc->stun_timer > 0 || npc->frozen_ticks > 0) return;

    if (npc->type == COLO_MANTICORE) {
        int mdist = col_npc_dist_to_player(s, npc);
        if (mdist < 1 || mdist > stats->attack_range) return;
        if (!col_npc_has_los_to_player(s, npc)) return;
        int cycle_step_before_arm = npc->type_state.manticore.cycle_step;
        col_npc_manticore_arm(s, idx);
        if (cycle_step_before_arm < 0 &&
                npc->type_state.manticore.cycle_step == 0 &&
                npc->attack_timer > 0) {
            ctx->manticore_charge_started[idx] = 1;
        }
        if (npc->attack_timer > 0) return;
        col_npc_attack_manticore(s, idx, stats);
        return;
    }

    if (npc->attack_timer > 0) return;

    AttackStyle style = stats->default_style;
    int dist = col_npc_dist_to_player(s, npc);

    if (npc->type == COLO_MINOTAUR) {
        if (dist == 1) col_npc_melee_attack(s, ctx, idx, stats, stats->max_hit);
        else col_npc_minotaur_heal_ally(s, idx);
        npc->attack_timer = stats->attack_speed;
        return;
    }

    if (stats->attack_range > 1) {
        if (dist < 1 || dist > stats->attack_range) return;
        if (!col_npc_has_los_to_player(s, npc)) return;
    } else {

        if (!col_npc_melee_adjacent_to_player(s, npc)) return;
    }

    switch (npc->type) {
        case COLO_JAGUAR_WARRIOR:
            col_npc_attack_jaguar(s, ctx, idx, stats);
            break;
        case COLO_JAVELIN_COLOSSUS:
            col_npc_attack_javelin(s, ctx, idx, stats);
            break;
        case COLO_SERPENT_SHAMAN:
        case COLO_SHOCKWAVE_COLOSSUS:
            col_npc_queue_projectile(s, idx, stats, style, stats->max_hit);
            break;
        default:
            return;
    }
    npc->attack_timer = stats->attack_speed;
}

static int col_warband_melee_adjacent(const ColosseumState* s, const ColoNPC* npc) {
    int dx = npc->x > s->player.x ? npc->x - s->player.x : s->player.x - npc->x;
    int dy = npc->y > s->player.y ? npc->y - s->player.y : s->player.y - npc->y;
    return dx + dy == 1;
}

static void col_warband_attack_phase(ColosseumState* s, ColosseumContext* ctx) {
#ifdef COLO_PROFILE_ENABLED
    int col_prof_enabled =
        COLO_PROFILE_ENABLED() && !s->profiling_suppressed;
    double col_prof_t0 = col_prof_enabled ? COLO_PROFILE_NOW_MS() : 0.0;
#endif
    if (s->warband_cycle_anchor < 0) {
#ifdef COLO_PROFILE_ENABLED
        COLO_PROFILE_MARK(COLO_PROF_STEP_WARBAND_ATTACK);
#endif
        return;
    }
    int phase = (s->tick - s->warband_cycle_anchor) % COLO_WARBAND_CYCLE_TICKS;
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        ColoNPC* npc = &s->npcs[i];
        if (!col_npc_is_live_target(npc)) continue;
        if (!col_type_is_warbander(npc->type)) continue;
        int wait = (col_warband_window_offset(npc->type) - phase
                    + COLO_WARBAND_CYCLE_TICKS) % COLO_WARBAND_CYCLE_TICKS;
        npc->attack_timer = wait;
        if (wait != 0) continue;
        if (npc->stun_timer > 0 || npc->frozen_ticks > 0) continue;
        if (npc->moved_this_tick || npc->moved_last_tick) continue;
        if (!col_warband_melee_adjacent(s, npc)) continue;
        const ColoNpcStats* stats = &COLO_NPC_STATS[npc->type];
        if (npc->type == COLO_FREMENNIK_BERSERKER)
            col_npc_melee_attack(s, ctx, i, stats, stats->max_hit);
        else
            col_npc_queue_projectile(s, i, stats, stats->default_style, stats->max_hit);
    }
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_STEP_WARBAND_ATTACK);
#endif
}

static void col_live_threat_style_counts(
    const ColosseumState* s, int* magic, int* ranged, int* melee
) {
    *magic = 0;
    *ranged = 0;
    *melee = 0;
    for (int n = 0; n < COLO_MAX_NPCS; n++) {
        const ColoNPC* npc = &s->npcs[n];
        if (!col_npc_is_live_enemy(npc)) continue;
        if (npc->type == COLO_SOL_HEREDIT) continue;
        const ColoNpcStats* ns = &COLO_NPC_STATS[npc->type];
        if (ns->attack_range < 1) continue;
        int dist = col_npc_dist_to_player(s, npc);
        if (dist < 1 || dist > ns->attack_range) continue;
        if (ns->attack_range > 1 && !col_npc_has_los_to_player(s, npc)) continue;
        AttackStyle style = ns->default_style;
        if (npc->type == COLO_MANTICORE) {
            const ColoManticoreState* mc = &npc->type_state.manticore;
            int orb = mc->cycle_step >= 0 ? mc->cycle_step : 0;
            if (orb < 3 && mc->orb_style[orb] != ATTACK_STYLE_NONE) style = mc->orb_style[orb];
        }
        if (style == ATTACK_STYLE_MAGIC) (*magic)++;
        else if (style == ATTACK_STYLE_RANGED) (*ranged)++;
        else if (style == ATTACK_STYLE_MELEE) (*melee)++;
    }
}

static void col_tick_npcs_ctx(ColosseumState* s, ColosseumContext* ctx) {
#ifdef COLO_PROFILE_ENABLED
    int col_prof_enabled =
        COLO_PROFILE_ENABLED() && !s->profiling_suppressed;
    double col_prof_total_t0 = col_prof_enabled ? COLO_PROFILE_NOW_MS() : 0.0;
    double col_prof_t0 = col_prof_total_t0;
#endif
    col_sol_boss_tick(s);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_STEP_SOL_BOSS);
#endif

    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        ColoNPC* npc = &s->npcs[i];
        if (col_npc_is_live_target(npc) && npc->type == COLO_JAVELIN_COLOSSUS)
            col_npc_resolve_javelin_skyfall(s, ctx, i);
    }
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_STEP_JAVELIN_SKYFALL);
#endif

    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        ColoNPC* npc = &s->npcs[i];
        if (!npc->active) continue;

        if (npc->death_ticks > 0) continue;
        if (npc->stun_timer > 0) npc->stun_timer--;
        if (npc->frozen_ticks > 0) npc->frozen_ticks--;

        if (col_type_is_hazard_entity(npc->type)) continue;

#ifdef COLO_PROFILE_ENABLED
        double col_prof_call_t0 = col_prof_enabled ? COLO_PROFILE_NOW_MS() : 0.0;
#endif
        col_npc_move_ctx(s, ctx, i);
#ifdef COLO_PROFILE_ENABLED
        if (col_prof_enabled)
            COLO_PROFILE_ADD(
                COLO_PROF_STEP_NPC_MOVEMENT,
                COLO_PROFILE_NOW_MS() - col_prof_call_t0);
        col_prof_call_t0 = col_prof_enabled ? COLO_PROFILE_NOW_MS() : 0.0;
#endif
        col_npc_attack_ctx(s, ctx, i);
#ifdef COLO_PROFILE_ENABLED
        if (col_prof_enabled)
            COLO_PROFILE_ADD(
                COLO_PROF_STEP_NPC_ATTACK,
                COLO_PROFILE_NOW_MS() - col_prof_call_t0);
#endif
    }
#ifdef COLO_PROFILE_ENABLED
    if (col_prof_enabled)
        COLO_PROFILE_ADD(COLO_PROF_STEP_NPC_TOTAL, COLO_PROFILE_NOW_MS() - col_prof_total_t0);
#endif
}
