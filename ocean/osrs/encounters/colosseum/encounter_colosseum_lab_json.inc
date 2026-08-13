#line 76 "encounter_colosseum.h"

static void col_lab_append_npcs_json(ColosseumState* s, ColoLabString* out) {
    encounter_lab_string_append(out, "\"npcs\":[");
    int emitted = 0;
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        ColoNPC* npc = &s->npcs[i];
        if (!npc->active) continue;
        if (emitted > 0) encounter_lab_string_append(out, ",");
        const EncounterPendingHit* pending =
            encounter_pending_hit_queue_earliest(&npc->pending_hits);
        int pending_ticks = pending ? pending->ticks_remaining : 0;
        encounter_lab_string_append(out,
            "{\"slot\":%d,\"type\":\"%s\",\"type_id\":%d,\"def_id\":%d,"
            "\"x\":%d,\"y\":%d,\"size\":%d,\"hp\":%d,\"max_hp\":%d,"
            "\"timer\":%d,\"stun\":%d,\"frozen\":%d,\"death_ticks\":%d,"
            "\"pending_count\":%d,\"pending_earliest_ticks\":%d}",
            i, col_lab_npc_type_name(npc->type), npc->type,
            COLO_NPC_DEF_IDS[npc->type], npc->x, npc->y,
            col_npc_effective_size(npc), npc->hp, npc->max_hp,
            npc->attack_timer, npc->stun_timer, npc->frozen_ticks,
            npc->death_ticks, npc->pending_hits.count, pending_ticks);
        emitted++;
    }
    encounter_lab_string_append(out, "]");
}

static void col_lab_append_modifiers_json(const ColosseumState* s, ColoLabString* out) {
    encounter_lab_string_append(out, "\"modifiers\":{\"active\":[");
    int emitted = 0;
    for (int m = 0; m < COLO_NUM_REAL_MODIFIERS; m++) {
        if (!(s->modifiers.active_mask & (1u << (unsigned)m))) continue;
        if (emitted > 0) encounter_lab_string_append(out, ",");
        encounter_lab_string_append(out, "{\"id\":%d,\"name\":\"%s\",\"tier\":%d}",
            m, col_lab_modifier_name(m), s->modifiers.tier[m]);
        emitted++;
    }
    encounter_lab_string_append(out, "],\"draft_pending\":%d,\"draft_options\":[",
        s->modifiers.draft_pending);
    for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++) {
        if (o > 0) encounter_lab_string_append(out, ",");
        encounter_lab_string_append(out, "%d", s->modifiers.draft_options[o]);
    }
    encounter_lab_string_append(out, "]}");
}

static char* col_lab_alloc_json(ColosseumState* s) {
    ColoLabString out;
    encounter_lab_string_init(&out, "colosseum lab");
    encounter_lab_string_append(&out,
        "{\"tick\":%d,\"wave\":%d,"
        "\"player\":{\"x\":%d,\"y\":%d,\"hp\":%d,\"max_hp\":%d,\"prayer\":\"%s\","
        "\"attack_timer\":%d,\"weapon_set\":%d,\"attacked\":%d,"
        "\"attack_target\":%d,\"attack_damage\":%d,"
        "\"loadout_profile\":%d,\"brew_doses\":%d,\"restore_doses\":%d,"
        "\"combat_doses\":%d,\"ranged_doses\":%d,\"surge_doses\":%d,"
        "\"surge_cooldown\":%d,\"spec_energy\":%d,\"spec_armed\":%d,"
        "\"thrall_active\":%d,\"thrall_target\":%d,\"thrall_lifetime\":%d,"
        "\"thrall_recast_cd\":%d,\"death_charge_window\":%d,\"death_charge_cd\":%d},",
        s->tick, s->wave + 1,
        s->player.x, s->player.y, s->player.current_hitpoints,
        s->player.base_hitpoints, col_lab_overhead_name(s->player.prayer),
        s->player.attack_timer, s->weapon_set, s->tick_scratch.player_attacked,
        s->player_attack_npc_idx, s->player_attack_dmg,
        s->active_loadout_profile, s->player.brew_doses, s->player.restore_doses,
        s->player.combat_potion_doses, s->player.ranged_potion_doses, s->surge_doses,
        s->surge_cooldown, s->player.special_energy, s->player.spec_armed,
        s->thrall_active, s->thrall_target_slot, s->thrall_lifetime_left,
        s->thrall_recast_cd, s->death_charge_window_left, s->death_charge_cd);
    col_lab_append_modifiers_json(s, &out);
    encounter_lab_string_append(&out, ",");
    encounter_lab_string_append(&out,
        "\"boss\":{\"started\":%d,\"phase\":%d,\"grapple_active\":%d,"
        "\"grapple_slot\":%d,\"parry_active\":%d},",
        s->sol.started, s->sol.phase, s->sol.grapple_active,
        s->sol.grapple_body_slot, s->sol.parry_hits_left > 0 ? 1 : 0);
    col_lab_append_npcs_json(s, &out);
    encounter_lab_string_append(&out, "}");
    return out.data;
}
