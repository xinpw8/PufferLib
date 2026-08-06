#line 862 "encounter_inferno.h"
static inline int inf_is_barrage_spell(int spell) {
    return spell == ENCOUNTER_SPELL_BLOOD || spell == ENCOUNTER_SPELL_ICE;
}

static inline int inf_spell_base_damage(int spell) {
    if (spell == ENCOUNTER_SPELL_ICE) return ICE_BARRAGE_MAX_HIT;
    if (spell == ENCOUNTER_SPELL_BLOOD) return BLOOD_BARRAGE_MAX_HIT;
    fprintf(stderr, "BUG: invalid Inferno spell %d\n", spell);
    abort();
}

static inline int inf_player_autocast_spell(const Player* p) {
    return p->autocast_spell == ENCOUNTER_SPELL_ICE
        ? ENCOUNTER_SPELL_ICE
        : ENCOUNTER_SPELL_BLOOD;
}

static inline int inf_player_magic_base_damage(const Player* p) {
    return inf_spell_base_damage(inf_player_autocast_spell(p));
}

static inline void inf_apply_spell_cast_stats(EncounterLoadoutStats* out) {
    out->attack_speed = COMBAT_SPELL_CAST_SPEED;
    out->attack_range = COMBAT_SPELL_ATTACK_RANGE;
}

static EncounterPendingHit* inf_queue_npc_pending_hit(
    InfernoState* s,
    int npc_idx,
    int ticks_remaining,
    int damage,
    AttackStyle attack_style,
    int spell_type,
    int hit_success
) {
    if (npc_idx < 0 || npc_idx >= INF_MAX_NPCS) {
        fprintf(stderr, "inferno: invalid pending-hit npc slot %d\n", npc_idx);
        abort();
    }

    InfNPC* npc = &s->npcs[npc_idx];
    if (!npc->active) {
        fprintf(stderr,
            "inferno: pending-hit queued for inactive npc tick=%d slot=%d "
            "type=%d delay=%d style=%d spell=%d damage=%d\n",
            s->tick, npc_idx, npc->type, ticks_remaining, attack_style,
            spell_type, damage);
        abort();
    }

    EncounterPendingHit hit = {
        .ticks_remaining = (int8_t)ticks_remaining,
        .attack_style = (int8_t)attack_style,
        .check_prayer = 0,
        .spell_type = (int8_t)spell_type,
        .hit_success = (int8_t)hit_success,
        .elysian_reduced = 0,
        .damage = (int16_t)damage,
    };
    return encounter_pending_hit_queue_push(
        &npc->pending_hits, hit, "inferno npc", s->tick, npc_idx, npc->type);
}

static EncounterPendingHit* inf_queue_player_pending_hit(
    InfernoState* s,
    int npc_idx,
    int npc_type,
    EncounterPendingHit hit
) {
    hit.source_npc_slot = (int8_t)npc_idx;
    return encounter_pending_hit_queue_push(
        &s->player_pending_hits, hit, "inferno player", s->tick, npc_idx, npc_type);
}

static void inf_shuffle_spawns(InfernoState* s) {
    for (int i = 0; i < INF_NUM_SPAWN_POS; i++)
        s->spawn_order[i] = i;
    encounter_shuffle(s->spawn_order, INF_NUM_SPAWN_POS, &s->rng_state);
}

static void inf_rebuild_los(InfernoState* s) {
    s->los_blocker_count = 0;
    for (int i = 0; i < INF_NUM_PILLARS; i++) {
        if (s->pillars[i].active) {
            LOSBlocker* b = &s->los_blockers[s->los_blocker_count++];
            b->x = s->pillars[i].x;
            b->y = s->pillars[i].y;
            b->size = INF_PILLAR_SIZE;
            b->los_mask = LOS_FULL_MASK;
        }
    }
}

static OsrsLosQuery inf_player_los_query(const InfernoState* s) {
    return osrs_los_blockers(s->los_blockers, s->los_blocker_count);
}

typedef struct {
    int x;
    int y;
    int size;
    int is_player;
} InfTargetArea;

static InfTargetArea inf_npc_current_target_area(const InfernoState* s, const InfNPC* npc) {
    if (npc->type == INF_NPC_NIBBLER) {
        int pillar_idx = s->nibbler_target_pillar;
        if (pillar_idx >= 0 && pillar_idx < INF_NUM_PILLARS &&
            s->pillars[pillar_idx].active) {
            return (InfTargetArea){
                .x = s->pillars[pillar_idx].x,
                .y = s->pillars[pillar_idx].y,
                .size = INF_PILLAR_SIZE,
                .is_player = 0,
            };
        }
        for (int i = 0; i < INF_NUM_PILLARS; i++) {
            if (s->pillars[i].active) {
                return (InfTargetArea){
                    .x = s->pillars[i].x,
                    .y = s->pillars[i].y,
                    .size = INF_PILLAR_SIZE,
                    .is_player = 0,
                };
            }
        }
    }

    if (npc->aggro_target >= 0 && npc->aggro_target < INF_MAX_NPCS &&
        s->npcs[npc->aggro_target].active) {
        const InfNPC* target = &s->npcs[npc->aggro_target];
        return (InfTargetArea){
            .x = target->x,
            .y = target->y,
            .size = target->size,
            .is_player = 0,
        };
    }

    return (InfTargetArea){
        .x = s->player.x,
        .y = s->player.y,
        .size = 1,
        .is_player = 1,
    };
}

static int inf_npc_has_los_to_area(
    const InfernoState* s,
    const InfNPC* npc,
    int target_x,
    int target_y,
    int target_size
) {
    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    return entity_has_line_of_sight(s->los_blockers, s->los_blocker_count,
        npc->x, npc->y, npc->size,
        target_x, target_y, target_size,
        stats->attack_range);
}

static int inf_npc_has_los_to_tile(
    const InfernoState* s, const InfNPC* npc, int target_x, int target_y
) {
    return inf_npc_has_los_to_area(s, npc, target_x, target_y, 1);
}

static int inf_npc_has_los_direct(const InfernoState* s, int i) {
    const InfNPC* npc = &s->npcs[i];
    InfTargetArea target = inf_npc_current_target_area(s, npc);
    return inf_npc_has_los_to_area(s, npc, target.x, target.y, target.size);
}

static int inf_npc_has_los(InfernoState* s, int i) {
    if (s->npc_los_cache[i] >= 0)
        return s->npc_los_cache[i];
    int result = inf_npc_has_los_direct(s, i);
    s->npc_los_cache[i] = (int8_t)result;
    return result;
}

static inline void inf_invalidate_los_cache(InfernoState* s) {
    memset(s->npc_los_cache, -1, sizeof(s->npc_los_cache));
}

static inline void inf_invalidate_npc_los_cache(InfernoState* s, int i) {
    s->npc_los_cache[i] = -1;
}

static inline int inf_is_final_wave(const InfernoState* s) {
    return s->wave == INF_NUM_WAVES - 1;
}

static inline int inf_zuk_healer_tags_first_reward_gate_active(
    const InfernoState* s,
    const InfernoContext* ctx
) {
    return ctx->config.zuk_healer_reward_mode == INF_ZUK_HEALER_REWARD_MODE_TAGS_FIRST &&
        s->zuk.healer_spawned && s->total_zuk_healer_tags < 4;
}

static int inf_find_live_zuk_idx(const InfernoState* s) {
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (s->npcs[i].active && s->npcs[i].type == INF_NPC_ZUK) return i;
    }
    return -1;
}

static int inf_is_live_zuk_healer_slot(const InfernoState* s, int npc_slot) {
    if (npc_slot < 0 || npc_slot >= INF_MAX_NPCS) return 0;
    const InfNPC* npc = &s->npcs[npc_slot];
    return npc->active &&
        npc->hp > 0 &&
        npc->death_ticks == 0 &&
        npc->type == INF_NPC_HEALER_ZUK;
}

static int inf_npc_type_is_set_pressure(int type) {
    return type == INF_NPC_MAGER ||
        type == INF_NPC_MELEER ||
        type == INF_NPC_RANGER;
}

enum {
    INF_STYLE_MASK_MELEE = 1 << 0,
    INF_STYLE_MASK_RANGED = 1 << 1,
    INF_STYLE_MASK_MAGIC = 1 << 2,
};

static inline int inf_attack_style_mask_bit(int style) {
    if (style == ATTACK_STYLE_MELEE) return INF_STYLE_MASK_MELEE;
    if (style == ATTACK_STYLE_RANGED) return INF_STYLE_MASK_RANGED;
    if (style == ATTACK_STYLE_MAGIC) return INF_STYLE_MASK_MAGIC;
    return 0;
}

static inline int inf_cardinal_contact_with_npc(int px, int py, int nx, int ny, int npc_size) {
    int cx = px < nx ? nx : (px > nx + npc_size - 1 ? nx + npc_size - 1 : px);
    int cy = py < ny ? ny : (py > ny + npc_size - 1 ? ny + npc_size - 1 : py);
    int dx = px - cx; if (dx < 0) dx = -dx;
    int dy = py - cy; if (dy < 0) dy = -dy;
    return dx + dy == 1;
}

static inline int inf_npc_melee_contact_allows_attack_at_tile(
    int player_x,
    int player_y,
    InfNPCType npc_type,
    int npc_x,
    int npc_y,
    int npc_size,
    int dist
) {
    if (dist != 1) return 0;
    if (npc_type == INF_NPC_RANGER || npc_type == INF_NPC_MAGER)
        return 1;
    return inf_cardinal_contact_with_npc(
        player_x, player_y, npc_x, npc_y, npc_size);
}

static inline int inf_melee_fallback_possible_at_tile(
    int player_x, int player_y, const InfNPC* npc, const InfNPCStats* stats,
    int planned_style, int dist
) {
    if (!stats->can_melee || planned_style == ATTACK_STYLE_MELEE || dist != 1)
        return 0;

    return inf_npc_melee_contact_allows_attack_at_tile(
        player_x, player_y, npc->type, npc->x, npc->y, npc->size, dist);
}

static inline int inf_melee_fallback_possible(
    const InfernoState* s, const InfNPC* npc, const InfNPCStats* stats,
    int planned_style, int dist
) {
    return inf_melee_fallback_possible_at_tile(
        s->player.x, s->player.y, npc, stats, planned_style, dist);
}

static inline int inf_attack_style_options_mask_at_tile(
    int player_x, int player_y, const InfNPC* npc, const InfNPCStats* stats,
    int planned_style, int dist
) {
    int mask = inf_attack_style_mask_bit(planned_style);
    if (inf_melee_fallback_possible_at_tile(
            player_x, player_y, npc, stats, planned_style, dist))
        mask |= INF_STYLE_MASK_MELEE;
    return mask;
}

static inline int inf_attack_style_options_mask(
    const InfernoState* s, const InfNPC* npc, const InfNPCStats* stats,
    int planned_style, int dist
) {
    return inf_attack_style_options_mask_at_tile(
        s->player.x, s->player.y, npc, stats, planned_style, dist);
}

static inline int inf_attack_style_from_mask(int style_mask) {
    if (style_mask == INF_STYLE_MASK_MELEE) return ATTACK_STYLE_MELEE;
    if (style_mask == INF_STYLE_MASK_RANGED) return ATTACK_STYLE_RANGED;
    if (style_mask == INF_STYLE_MASK_MAGIC) return ATTACK_STYLE_MAGIC;
    return ATTACK_STYLE_NONE;
}

static inline int inf_style_count_from_mask(int style_mask) {
    int mask = style_mask & (INF_STYLE_MASK_MELEE |
        INF_STYLE_MASK_RANGED | INF_STYLE_MASK_MAGIC);
    return ((mask & INF_STYLE_MASK_MELEE) != 0) +
        ((mask & INF_STYLE_MASK_RANGED) != 0) +
        ((mask & INF_STYLE_MASK_MAGIC) != 0);
}

static inline int inf_npc_style_max_hit(
    const InfNPCStats* stats, int style
) {
    if (style == ATTACK_STYLE_NONE) return 0;
    int max_hit = osrs_npc_max_hit(style,
        stats->str_level, stats->range_level,
        stats->melee_str_bonus, stats->ranged_str_bonus,
        stats->magic_base_dmg, stats->magic_dmg_pct);
    if (stats->max_hit_cap > 0 && max_hit > stats->max_hit_cap)
        max_hit = stats->max_hit_cap;
    return max_hit;
}

static inline int inf_npc_style_mask_max_hit(
    const InfNPCStats* stats, int style_mask
) {
    int max_hit = 0;
    if (style_mask & INF_STYLE_MASK_MELEE) {
        int style_max = inf_npc_style_max_hit(stats, ATTACK_STYLE_MELEE);
        if (style_max > max_hit) max_hit = style_max;
    }
    if (style_mask & INF_STYLE_MASK_RANGED) {
        int style_max = inf_npc_style_max_hit(stats, ATTACK_STYLE_RANGED);
        if (style_max > max_hit) max_hit = style_max;
    }
    if (style_mask & INF_STYLE_MASK_MAGIC) {
        int style_max = inf_npc_style_max_hit(stats, ATTACK_STYLE_MAGIC);
        if (style_max > max_hit) max_hit = style_max;
    }
    return max_hit;
}

static inline int inf_attack_style_obs_preview(int style_mask) {
    int primary_mask = style_mask & (INF_STYLE_MASK_RANGED | INF_STYLE_MASK_MAGIC);
    int primary_style = inf_attack_style_from_mask(primary_mask);
    if (primary_style != ATTACK_STYLE_NONE)
        return primary_style;
    return inf_attack_style_from_mask(style_mask);
}

static inline int inf_attack_style_telegraph_mask(
    const InfernoState* s, const InfNPC* npc, const InfNPCStats* stats,
    int planned_style, int dist
) {
    if (npc->type == INF_NPC_JAD)
        return inf_attack_style_mask_bit(planned_style);
    return inf_attack_style_options_mask(s, npc, stats, planned_style, dist);
}

typedef struct {
    int edge_distance;
    int style_mask;
    int telegraph_style_mask;
    int can_attack_if_ready;
    int can_attack_this_tick;
} InfNpcPlayerThreat;

static inline int inf_npc_planned_style_for_current_state(const InfNPC* npc) {
    if (npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer >= 0) {
        OverheadPrayer scanned = (OverheadPrayer)npc->blob_scanned_prayer;
        if (scanned == PRAYER_PROTECT_MAGIC) return ATTACK_STYLE_RANGED;
        if (scanned == PRAYER_PROTECT_RANGED) return ATTACK_STYLE_MAGIC;
    }
    if (npc->type == INF_NPC_JAD)
        return inf_npc_jad_const(npc)->attack_style;
    return npc->attack_style;
}

static inline int inf_npc_type_has_player_pressure(InfNPCType type) {
    return type != INF_NPC_NIBBLER &&
        type != INF_NPC_ZUK &&
        type != INF_NPC_ZUK_SHIELD &&
        type != INF_NPC_HEALER_ZUK;
}

static inline InfNpcPlayerThreat inf_npc_player_threat(
    const InfernoState* s,
    const InfNPC* npc
) {
    InfNpcPlayerThreat threat = {0};
    if (!npc->active || npc->death_ticks > 0)
        return threat;
    if (!inf_npc_type_has_player_pressure(npc->type))
        return threat;

    InfTargetArea target = inf_npc_current_target_area(s, npc);
    if (!target.is_player)
        return threat;

    const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
    int dist = encounter_dist_to_npc(
        s->player.x, s->player.y, npc->x, npc->y, npc->size);
    threat.edge_distance = dist;
    if (dist == 0 || dist > stats->attack_range)
        return threat;
    if (stats->attack_range == 1 &&
            !inf_npc_melee_contact_allows_attack_at_tile(
                s->player.x, s->player.y, npc->type,
                npc->x, npc->y, npc->size, dist))
        return threat;

    if (stats->attack_range > 1 &&
            !(npc->type == INF_NPC_BLOB && npc->blob_scanned_prayer >= 0) &&
            !inf_npc_has_los_to_tile(s, npc, s->player.x, s->player.y))
        return threat;

    threat.can_attack_if_ready = 1;

    int planned_style = inf_npc_planned_style_for_current_state(npc);
    if (planned_style != ATTACK_STYLE_NONE) {
        threat.style_mask = inf_attack_style_options_mask(
            s, npc, stats, planned_style, dist);
        threat.telegraph_style_mask = inf_attack_style_telegraph_mask(
            s, npc, stats, planned_style, dist);
    }

    threat.can_attack_this_tick = npc->attack_timer <= 1 &&
        npc->stun_timer == 0 &&
        npc->dig_freeze_timer == 0 &&
        npc->dig_attack_delay == 0;
    return threat;
}

static inline int inf_npc_is_live_player_target(
    const InfernoState* s,
    int npc_idx
) {
    if (npc_idx < 0 || npc_idx >= INF_MAX_NPCS) return 0;
    const InfNPC* npc = &s->npcs[npc_idx];
    return npc->active && npc->death_ticks == 0 && npc->hp > 0 &&
        npc->type != INF_NPC_ZUK_SHIELD;
}

typedef struct {
    int if_ready_total;
    int if_ready_melee;
    int if_ready_ranged;
    int if_ready_magic;
    int if_ready_max_hit;
    int if_ready_sum_max_hit;
    int this_tick_total;
    int this_tick_melee;
    int this_tick_ranged;
    int this_tick_magic;
    int this_tick_max_hit;
    int this_tick_sum_max_hit;
    int timer_lte_1;
    int timer_lte_2;
    int timer_lte_4;
    int if_ready_style_count;
    int this_tick_style_count;
    int nearest_timer;
} InfNpcPressureSummary;

static inline int inf_npc_pressure_timer(const InfNPC* npc) {
    return npc->attack_timer > 0 ? npc->attack_timer : 1;
}

static inline int inf_npc_attack_unblocked_for_pressure(const InfNPC* npc) {
    return npc->stun_timer == 0 &&
        npc->dig_freeze_timer == 0 &&
        npc->dig_attack_delay == 0;
}

static inline InfNpcPressureSummary inf_npc_pressure_summary(
    const InfernoState* s
) {
    InfNpcPressureSummary out = { .nearest_timer = 999 };
    int if_ready_style_mask = 0;
    int this_tick_style_mask = 0;

    for (int i = 0; i < INF_MAX_NPCS; i++) {
        const InfNPC* npc = &s->npcs[i];
        InfNpcPlayerThreat threat = inf_npc_player_threat(s, npc);
        if (!threat.can_attack_if_ready || !threat.style_mask)
            continue;

        const InfNPCStats* stats = &INF_NPC_STATS[npc->type];
        int max_hit = inf_npc_style_mask_max_hit(stats, threat.style_mask);
        out.if_ready_total++;
        if (threat.style_mask & INF_STYLE_MASK_MELEE) out.if_ready_melee++;
        if (threat.style_mask & INF_STYLE_MASK_RANGED) out.if_ready_ranged++;
        if (threat.style_mask & INF_STYLE_MASK_MAGIC) out.if_ready_magic++;
        if (max_hit > out.if_ready_max_hit) out.if_ready_max_hit = max_hit;
        out.if_ready_sum_max_hit += max_hit;
        if_ready_style_mask |= threat.style_mask;

        if (inf_npc_attack_unblocked_for_pressure(npc)) {
            int timer = inf_npc_pressure_timer(npc);
            if (timer < out.nearest_timer) out.nearest_timer = timer;
            if (timer <= 1) out.timer_lte_1++;
            if (timer <= 2) out.timer_lte_2++;
            if (timer <= 4) out.timer_lte_4++;
        }

        if (!threat.can_attack_this_tick)
            continue;

        out.this_tick_total++;
        if (threat.style_mask & INF_STYLE_MASK_MELEE) out.this_tick_melee++;
        if (threat.style_mask & INF_STYLE_MASK_RANGED) out.this_tick_ranged++;
        if (threat.style_mask & INF_STYLE_MASK_MAGIC) out.this_tick_magic++;
        if (max_hit > out.this_tick_max_hit) out.this_tick_max_hit = max_hit;
        out.this_tick_sum_max_hit += max_hit;
        this_tick_style_mask |= threat.style_mask;
    }

    out.if_ready_style_count = inf_style_count_from_mask(if_ready_style_mask);
    out.this_tick_style_count = inf_style_count_from_mask(this_tick_style_mask);
    return out;
}

static inline int inf_barrage_total_target_count(
    const InfernoState* s,
    int primary_idx
) {
    if (primary_idx < 0 || primary_idx >= INF_MAX_NPCS) return 0;
    const InfNPC* primary = &s->npcs[primary_idx];
    if (!primary->active || primary->death_ticks > 0 ||
            primary->type == INF_NPC_ZUK_SHIELD)
        return 0;

    int count = 1;
    for (int i = 0; i < INF_MAX_NPCS && count < BARRAGE_MAX_HITS; i++) {
        if (i == primary_idx) continue;
        const InfNPC* npc = &s->npcs[i];
        if (!npc->active || npc->death_ticks > 0 ||
                npc->type == INF_NPC_ZUK_SHIELD)
            continue;

        int dx = npc->x - primary->x;
        int dy = npc->y - primary->y;
        if (dx >= -1 && dx <= 1 && dy >= -1 && dy <= 1)
            count++;
    }
    return count;
}

static inline int inf_pending_hit_obs_timer(const EncounterPendingHit* ph) {
    if (ph->check_prayer && ph->prayer_check_delay > 0)
        return ph->prayer_check_delay;
    return ph->ticks_remaining;
}

#define INF_JAD_PROJECTILE_DELAY 3

static inline EncounterProjectileDelayOptions inf_npc_projectile_options(
    InfNPCType type, int style
) {
    EncounterProjectileDelayOptions options = {0};
    if (type == INF_NPC_MAGER && style == ATTACK_STYLE_MAGIC) {
        options.visual_delay_ticks = 2;
        options.visual_hit_early_ticks = -1;
    } else if (type == INF_NPC_RANGER && style == ATTACK_STYLE_RANGED) {
        options.reduce_delay = -2;
        options.visual_delay_ticks = 3;
    } else if (type == INF_NPC_JAD && style != ATTACK_STYLE_MELEE) {
        options.reduce_delay = INF_JAD_PROJECTILE_DELAY;
        options.visual_hit_early_ticks = -1;
    } else if (type == INF_NPC_ZUK) {
        options.set_delay = 4;
        options.visual_delay_ticks = 2;
    }
    return options;
}

static inline EncounterProjectileTiming inf_npc_projectile_timing(
    InfNPCType type, int style, int distance
) {
    return encounter_projectile_timing(
        distance, 0, encounter_projectile_delay_kind_for_style((AttackStyle)style),
        inf_npc_projectile_options(type, style));
}

static inline EncounterProjectileTiming inf_player_projectile_timing(
    int style, uint8_t weapon, int is_special, int distance
) {
    EncounterProjectileDelayKind kind = encounter_projectile_delay_kind_for_style((AttackStyle)style);
    EncounterProjectileDelayOptions options = {0};
    if (style == ATTACK_STYLE_RANGED) {
        if (weapon == ITEM_TOXIC_BLOWPIPE) {
            kind = ENCOUNTER_PROJECTILE_DELAY_THROWN;
            options.visual_delay_ticks = 1;
            if (is_special) {
                options.reduce_delay = -1;
                options.visual_hit_early_ticks = 1;
            }
        } else if (weapon == ITEM_TWISTED_BOW) {
            options.visual_delay_ticks = 1;
        }
    }
    return encounter_projectile_timing(distance, 1, kind, options);
}

static inline AttackStyle inf_jad_roll_primary_style(uint32_t* rng_state) {
    return (encounter_rand_int(rng_state, 2) == 0)
        ? ATTACK_STYLE_RANGED
        : ATTACK_STYLE_MAGIC;
}

static inline int inf_npc_attack_anim_id(const InfNPC* npc, const NpcModelMapping* nm) {
    if (npc->resurrecting_this_tick)
        return INF_GEN_ANIM_MAGER_RESURRECT;
    if (npc->type == INF_NPC_RANGER &&
        npc->attack_style_this_tick == ATTACK_STYLE_MELEE)
        return INF_GEN_ANIM_RANGER_ATTACK_MELEE;
    if (npc->type == INF_NPC_MAGER &&
        npc->attack_style_this_tick == ATTACK_STYLE_MELEE)
        return INF_GEN_ANIM_MAGER_ATTACK_MELEE;

    if (npc->type == INF_NPC_JAD) {
        switch (npc->attack_style_this_tick) {
            case ATTACK_STYLE_MAGIC:
                return INF_GEN_ANIM_JALTOK_JAD_ATTACK_MAGIC;
            case ATTACK_STYLE_RANGED:
                return INF_GEN_ANIM_JALTOK_JAD_ATTACK_RANGED;
            case ATTACK_STYLE_MELEE:
                return INF_GEN_ANIM_JALTOK_JAD_ATTACK_MELEE;
            default:
                return -1;
        }
    }

    if (!nm || nm->attack_anim == 65535)
        return -1;
    return (int)nm->attack_anim;
}

static inline int inf_npc_death_anim_id(const InfNPC* npc, const NpcModelMapping* nm) {
    switch (npc->type) {
        case INF_NPC_NIBBLER:
            return INF_GEN_ANIM_NIBBLER_DEATH;
        case INF_NPC_BAT:
            return INF_GEN_ANIM_BAT_DEATH;
        case INF_NPC_BLOB:
        case INF_NPC_BLOB_MELEE:
        case INF_NPC_BLOB_RANGE:
        case INF_NPC_BLOB_MAGE:
            return INF_GEN_ANIM_BLOB_DEATH;
        case INF_NPC_MELEER:
            return INF_GEN_ANIM_MELEER_DEATH;
        case INF_NPC_RANGER:
            return INF_GEN_ANIM_RANGER_DEATH;
        case INF_NPC_MAGER:
            return INF_GEN_ANIM_MAGER_DEATH;
        case INF_NPC_JAD:
            return INF_GEN_ANIM_JALTOK_JAD_DEATH;
        case INF_NPC_ZUK:
            return INF_GEN_ANIM_TZKAL_ZUK_DEATH;
        case INF_NPC_ZUK_SHIELD:
            return INF_GEN_ANIM_ZUK_SHIELD_DEATH;
        default:
            return nm ? (int)nm->idle_anim : -1;
    }
}

static inline AttackStyle inf_choose_attack_style_for_tick(
    uint32_t* rng_state, int style_mask
) {
    AttackStyle primary_style = (AttackStyle)inf_attack_style_from_mask(
        style_mask & ~INF_STYLE_MASK_MELEE);
    if ((style_mask & INF_STYLE_MASK_MELEE) && primary_style != ATTACK_STYLE_NONE)
        return (encounter_rand_int(rng_state, 2) == 0) ? ATTACK_STYLE_MELEE : primary_style;
    return (AttackStyle)inf_attack_style_from_mask(style_mask);
}

static inline int inf_dead_mob_is_resurrectable(InfNPCType type) {
    switch (type) {
        case INF_NPC_BAT:
        case INF_NPC_BLOB:
        case INF_NPC_MELEER:
        case INF_NPC_RANGER:
        case INF_NPC_MAGER:
            return 1;
        default:
            return 0;
    }
}

static void inf_store_dead_mob(InfernoState* s, InfNPC* npc) {
    if (s->dead_mob_count >= INF_MAX_DEAD_MOBS) return;
    if (!inf_dead_mob_is_resurrectable(npc->type)) return;
    if (npc->resurrection_count != 0) return;

    InfDeadMob* dm = &s->dead_mobs[s->dead_mob_count++];
    dm->type = npc->type;
    dm->x = npc->x;
    dm->y = npc->y;
    dm->hp = npc->max_hp / 2;
    dm->max_hp = npc->max_hp;
}

static float inf_compute_reward_ctx(InfernoState* s, const InfernoContext* ctx);
static float inf_compute_reward(InfernoState* s);
static void inf_spawn_wave(InfernoState* s);
static void inf_tick_npcs_ctx(InfernoState* s, InfernoContext* ctx);
static void inf_tick_npcs(InfernoState* s);
static void inf_tick_player_ctx(
    InfernoState* s,
    InfernoContext* ctx,
    const int* actions,
    int can_attack);
static void inf_tick_player(InfernoState* s, const int* actions, int can_attack);
static void inf_apply_npc_death(InfernoState* s, int npc_idx);
static int inf_mager_resurrect_ctx(InfernoState* s, InfernoContext* ctx, int idx);
static void inf_queue_zuk_healer_sparks(InfernoState* s, const InfNPC* npc);
static void inf_resolve_pending_sparks(InfernoState* s);
static void inf_rebuild_player_collision_flags(InfernoState* s);
static void inf_refresh_current_obs_slots_ctx(InfernoState* s, const InfernoContext* ctx);
static void inf_refresh_current_obs_slots(InfernoState* s);

static InfConfig inf_default_config(void) {
    return (InfConfig){
        .start_wave = 0,
        .loadout_profile_mode = INF_LOADOUT_PROFILE_MODE_MAX_ONLY,
        .budget_loadout_fraction = 0.0f,
        .damage_reward_coeff = 0.0f,
        .offensive_prayer_reward_coeff = 0.0f,
        .shield_penalty_coeff = 0.0f,
        .tag_reward_coeff = 0.0f,
        .late_start_supply_profile_scale = 1.0f,
        .curriculum_agent = 0,
        .curriculum_supply_jitter_mode = INF_CURRICULUM_SUPPLY_MODE_OFF,
        .curriculum_supply_shared_jitter = 0.0f,
        .curriculum_supply_brew_jitter = 0.0f,
        .curriculum_supply_restore_jitter = 0.0f,
        .curriculum_no_brew_mode = INF_CURRICULUM_SUPPLY_MODE_OFF,
        .curriculum_no_brew_frac = 0.0f,
        .supply_milestone_brew_reward_coeff = 0.0f,
        .supply_milestone_restore_reward_coeff = 0.0f,
        .death_penalty_coeff = 0.0f,
        .terminal_penalty_enabled = 0,
        .step_out_forecast_obs_enabled = 1,
        .step_out_forecast_obs_mode = INF_STEP_OUT_FORECAST_MODE_EXACT_ROLLOUT,
        .phase_900_bonus = 0.0f,
        .phase_600_bonus = 0.0f,
        .phase_300_bonus = 0.0f,
        .shield_penalty_episode_cap = 0.0f,
        .jad_damage_reward_coeff = 0.0f,
        .zuk_healer_damage_reward_coeff = 0.0f,
        .set_damage_reward_coeff = 0.0f,
        .jad_kill_bonus = 0.0f,
        .zuk_healer_kill_bonus = 0.0f,
        .set_kill_bonus = 0.0f,
        .post_healer_zuk_damage_coeff = 0.0f,
        .post_healer_set_damage_reward_coeff = 0.0f,
        .post_healer_set_kill_bonus = 0.0f,
        .post_healer_set_alive_tick_penalty_coeff = 0.0f,
        .post_healer_set_alive_penalty_cap = 0.0f,
        .zuk_healer_phase_hp_delta_coeff = 0.0f,
        .zuk_untagged_healer_tick_penalty_coeff = 0.0f,
        .zuk_untagged_healer_target_bonus_coeff = 0.0f,
        .zuk_safe_untagged_healer_target_bonus_coeff = 0.0f,
        .zuk_untagged_healer_nonmagic_attack_bonus_coeff = 0.0f,
        .zuk_healer_mage_attack_penalty_coeff = 0.0f,
        .zuk_safe_untagged_healer_target_mask = 0,
        .zuk_force_safe_untagged_healer_target_mask = 0,
        .zuk_healer_reward_mode = INF_ZUK_HEALER_REWARD_MODE_BASELINE,
        .post_jad_zuk_multiplier = 1.0f,
        .jad_alive_zuk_multiplier = 1.0f,
        .shield_tag_reward_coeff = 0.0f,
        .joseph_reward_mode = INF_JOSEPH_REWARD_MODE_OFF,
    };
}

static void inf_init_context_typed(InfernoContext* ctx) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->config = inf_default_config();
}

static void inf_init_context(EncounterContext* context) {
    inf_init_context_typed((InfernoContext*)context);
}

static void inf_destroy_context(EncounterContext* context) {
    (void)context;
}

static InfernoContext* inf_legacy_context(void) {
    static InfernoContext ctx;
    static Log log;
    static int initialized = 0;
    if (!initialized) {
        inf_init_context_typed(&ctx);
        ctx.log = &log;
        initialized = 1;
    }
    return &ctx;
}

static void inf_init_state_typed(InfernoState* s, InfernoContext* ctx) {
    inf_build_npc_stats();
    memset(s, 0, sizeof(*s));
    s->rng_state = 12345;
    s->start_wave = ctx ? ctx->config.start_wave : 0;
    s->active_loadout_profile = INF_LOADOUT_PROFILE_MAX;
    memset(s->npc_los_cache, -1, sizeof(s->npc_los_cache));
}

static void inf_init_state(EncounterState* state, EncounterContext* context) {
    inf_init_state_typed((InfernoState*)state, (InfernoContext*)context);
}

static EncounterState* inf_create(void) {
    InfernoState* s = (InfernoState*)calloc(1, sizeof(InfernoState));
    inf_init_state_typed(s, inf_legacy_context());
    return (EncounterState*)s;
}

static void inf_destroy(EncounterState* state) {
    free(state);
}
