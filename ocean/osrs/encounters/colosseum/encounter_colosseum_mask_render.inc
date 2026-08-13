#line 76 "encounter_colosseum.h"

static void col_write_mask_ctx(
    EncounterState* state,
    EncounterContext* context,
    float* mask
) {
    ColosseumState* s = (ColosseumState*)state;
    ColosseumContext* ctx = col_require_context(context);
    encounter_arena_topology_require_finalized(ctx->route_topology);
    col_refresh_current_obs_slots_ctx(s, ctx);

    int primary_offset = col_action_head_mask_offset(COLO_HEAD_PRIMARY);
    for (int action = 0; action < COLO_PRIMARY_DIM; action++) {
        mask[primary_offset + action] =
            col_primary_action_valid(s, ctx, action) ? 1.0f : 0.0f;
    }

    int overhead_offset = col_action_head_mask_offset(COLO_HEAD_PRAYER);
    mask[overhead_offset + ENCOUNTER_OVERHEAD_NO_CHANGE] = 1.0f;
    mask[overhead_offset + ENCOUNTER_OVERHEAD_OFF] =
        s->player.prayer != PRAYER_NONE ? 1.0f : 0.0f;
    for (int action = ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE;
            action < OSRS_OVERHEAD_DIM;
            action++) {
        mask[overhead_offset + action] =
            s->player.current_prayer > 0 ? 1.0f : 0.0f;
    }

    int inventory_heads_enabled = !ctx->config.mask_inventory_heads;
    int cell_equip_slot[COLO_INVENTORY_DISPLAY_SLOTS];
    int cell_can_eat[COLO_INVENTORY_DISPLAY_SLOTS];
    int cell_can_drink[COLO_INVENTORY_DISPLAY_SLOTS];
    int inventory_has_empty_cell =
        osrs_first_empty_inventory_cell(s->player.inventory_cells, -1) >= 0;
    for (int cell = 0; cell < COLO_INVENTORY_DISPLAY_SLOTS; cell++) {
        cell_equip_slot[cell] = -1;
        cell_can_eat[cell] = 0;
        cell_can_drink[cell] = 0;
        OsrsInventoryClickResolution resolution =
            osrs_inventory_cell_click_classify(
                &s->player.inventory_cells[cell]);
        if (resolution.click_action == OSRS_CLICK_EQUIP) {
            const OsrsItemContentMetadata* metadata =
                osrs_inventory_cell_metadata(
                    &s->player.inventory_cells[cell]);
            if (osrs_can_equip_metadata(
                    &s->player, metadata, inventory_has_empty_cell)) {
                cell_equip_slot[cell] = metadata->gear_slot;
            }
        } else if (resolution.click_action == OSRS_CLICK_EAT) {
            cell_can_eat[cell] =
                col_can_eat_consumable_kind(s, resolution.consumable_kind);
        } else if (resolution.click_action == OSRS_CLICK_DRINK) {
            cell_can_drink[cell] = osrs_inventory_cell_dose_count(
                &s->player.inventory_cells[cell]) > 0 &&
                s->player.potion_timer == 0 &&
                col_drink_has_effect(s, resolution.consumable_kind);
        }
    }
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        int head_offset =
            col_action_head_mask_offset(COLO_HEAD_EQUIP_SLOT(slot));
        mask[head_offset] = 1.0f;
        for (int cell = 0; cell < COLO_INVENTORY_DISPLAY_SLOTS; cell++) {
            mask[head_offset + cell + 1] =
                inventory_heads_enabled && cell_equip_slot[cell] == slot
                    ? 1.0f : 0.0f;
        }
    }
    int eat_offset = col_action_head_mask_offset(COLO_HEAD_EAT);
    int drink_offset = col_action_head_mask_offset(COLO_HEAD_DRINK);
    mask[eat_offset] = 1.0f;
    mask[drink_offset] = 1.0f;
    for (int cell = 0; cell < COLO_INVENTORY_DISPLAY_SLOTS; cell++) {
        mask[eat_offset + cell + 1] =
            inventory_heads_enabled && cell_can_eat[cell] ? 1.0f : 0.0f;
        mask[drink_offset + cell + 1] =
            inventory_heads_enabled && cell_can_drink[cell] ? 1.0f : 0.0f;
    }

    int spell_offset = col_action_head_mask_offset(COLO_HEAD_SPELL);
    mask[spell_offset + OSRS_SPELL_NONE] = 1.0f;
    for (int spell = OSRS_SPELL_BLOOD_BARRAGE;
            spell < OSRS_SPELL_DEATH_CHARGE;
            spell++)
        mask[spell_offset + spell] = 0.0f;
    int frozen =
        s->modifiers.draft_pending && !s->modifiers.draft_free_movement;
    mask[spell_offset + OSRS_SPELL_DEATH_CHARGE] =
        !frozen && s->death_charge_cd == 0 &&
        s->death_charge_window_left == 0 ? 1.0f : 0.0f;

    int spec_offset = col_action_head_mask_offset(COLO_HEAD_SPEC);
    mask[spec_offset] = 1.0f;
    int spec_cost = osrs_spec_cost(s->player.equipped[GEAR_SLOT_WEAPON]);
    mask[spec_offset + 1] =
        spec_cost > 0 &&
        (s->player.special_energy >= spec_cost || s->player.spec_armed)
            ? 1.0f : 0.0f;
    mask[spec_offset + 2] = s->player.spec_armed ? 1.0f : 0.0f;

    int offensive_offset = col_action_head_mask_offset(COLO_HEAD_OFFENSIVE);
    mask[offensive_offset] = 1.0f;
    mask[offensive_offset + 1] =
        s->player.offensive_prayer != OFFENSIVE_PRAYER_NONE ? 1.0f : 0.0f;
    mask[offensive_offset + 2] =
        s->player.current_prayer > 0 &&
        s->player.offensive_prayer != OFFENSIVE_PRAYER_PIETY ? 1.0f : 0.0f;
    mask[offensive_offset + 3] =
        s->player.current_prayer > 0 &&
        s->player.offensive_prayer != OFFENSIVE_PRAYER_RIGOUR ? 1.0f : 0.0f;
    mask[offensive_offset + 4] =
        s->player.current_prayer > 0 &&
        s->player.offensive_prayer != OFFENSIVE_PRAYER_AUGURY ? 1.0f : 0.0f;

    int modifier_offset =
        col_action_head_mask_offset(COLO_HEAD_MODIFIER_SELECT);
    mask[modifier_offset] = s->modifiers.draft_pending ? 0.0f : 1.0f;
    for (int option = 0; option < COLO_MODIFIER_DRAFT_OPTIONS; option++) {
        mask[modifier_offset + option + 1] =
            s->modifiers.draft_pending &&
            s->modifiers.draft_options[option] >= 0 ? 1.0f : 0.0f;
    }

    int grapple_offset =
        col_action_head_mask_offset(COLO_HEAD_GRAPPLE_PARRY);
    mask[grapple_offset] = 1.0f;
    for (int grapple = 0; grapple < COLO_NUM_GRAPPLE_SLOTS; grapple++) {
        mask[grapple_offset + grapple + 1] =
            s->sol.grapple_active ? 1.0f : 0.0f;
    }
}



static float col_get_reward_ctx(EncounterState* state, EncounterContext* context) {
    (void)context;
    return ((ColosseumState*)state)->reward;
}

static int col_is_terminal_ctx(EncounterState* state, EncounterContext* context) {
    (void)context;
    return ((ColosseumState*)state)->episode_over;
}

static int col_get_entity_count_ctx(EncounterState* state, EncounterContext* context) {
    (void)context;
    ColosseumState* s = (ColosseumState*)state;
    int count = 1;
    for (int i = 0; i < COLO_MAX_NPCS; i++)
        if (s->npcs[i].active) count++;
    return count;
}

static void* col_get_entity_ctx(EncounterState* state, EncounterContext* context, int index) {
    (void)context;
    ColosseumState* s = (ColosseumState*)state;
    if (index == 0) return &s->player;
    return NULL;
}

static void col_prepare_render_player(const ColosseumState* s, int spec_fired_kind, Player* out) {
    *out = s->player;
    GearSet gear = col_gear_set_for_attack_style(col_equipped_weapon_attack_style(&s->player));
    out->current_gear = gear;
    out->visible_gear = gear;
    if (spec_fired_kind > 0 && spec_fired_kind < NUM_ITEMS) {
        out->equipped[GEAR_SLOT_WEAPON] = (uint8_t)spec_fired_kind;
        out->equipped[GEAR_SLOT_SHIELD] = osrs_suppress_shield_for_two_handed_weapon(
            out->equipped[GEAR_SLOT_WEAPON], out->equipped[GEAR_SLOT_SHIELD]);
        out->current_gear = GEAR_MELEE;
        out->visible_gear = GEAR_MELEE;
    }
}

static int col_sol_render_attack_anim_id(const SolHereditState* sol) {
    switch (sol->last_attack_kind) {
        case COLO_SOL_ATTACK_SPEAR:   return 10883;
        case COLO_SOL_ATTACK_SHIELD:  return 10885;
        case COLO_SOL_ATTACK_GRAPPLE: return 10884;
        case COLO_SOL_ATTACK_TRIPLE:  return sol->phase >= 3 ? 10886 : 10887;
        default: break;
    }
    return 10883;
}

static int col_npc_render_attack_anim_id(
    const ColosseumState* s,
    const ColosseumContext* ctx,
    int npc_slot,
    const ColoNPC* npc,
    int def_id
) {
    assert(ctx);
    if (npc->type == COLO_SOL_HEREDIT)
        return col_sol_render_attack_anim_id(&s->sol);
    if (npc->type == COLO_MANTICORE && ctx->manticore_charge_started[npc_slot])
        return 10868;
    switch (ctx->npc_render_attack_anim[npc_slot]) {
        case COLO_NPC_RENDER_ATTACK_NORMAL:
            return col_npc_attack_anim_id(def_id);
        case COLO_NPC_RENDER_ATTACK_JAVELIN_SKYFALL:
            assert(npc->type == COLO_JAVELIN_COLOSSUS);
            return COLO_JAVELIN_SKYFALL_ANIM_ID;
    }
    abort();
}

static void col_thrall_render_tile(
    ColosseumState* s,
    const ColosseumContext* ctx,
    int* out_x,
    int* out_y
) {
    static const int RING[16][2] = {
        {0, 1}, {1, 0}, {-1, 0}, {0, -1},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
        {0, 2}, {2, 0}, {-2, 0}, {0, -2},
        {1, 2}, {2, 1}, {-1, 2}, {-2, 1} };
    ColoGeometryContext geometry = {
        .state = s,
        .context = ctx,
    };
    for (int a = 0; a < 16; a++) {
        int x = s->player.x + RING[a][0];
        int y = s->player.y + RING[a][1];
        if (!col_npc_blocked(&geometry, x, y, 1)) {
            *out_x = x;
            *out_y = y;
            return;
        }
    }
    *out_x = s->player.x;
    *out_y = s->player.y + 1;
}

static void col_emit_decoration_entity(
    RenderEntity* out, int* n, int max_entities, int def_id, int npc_slot,
    uint32_t instance_id, int anim_id, int x, int y, const char* debug_name
) {
    if (*n >= max_entities) return;
    RenderEntity* re = &out[(*n)++];
    OsrsNpcRenderEntitySpec spec = {
        .npc_def_id = def_id,
        .npc_slot = npc_slot,
        .npc_instance_id = instance_id,
        .npc_visible = 1,
        .npc_size = 1,
        .npc_anim_id = anim_id,
        .x = x,
        .y = y,
        .dest_x = x,
        .dest_y = y,
        .current_hitpoints = 1,
        .base_hitpoints = 1,
        .attack_style_this_tick = ATTACK_STYLE_NONE,
        .hit_landed_this_tick = 0,
        .hit_damage = 0,
        .hit_was_successful = 0,
        .hit_spell_type = ENCOUNTER_SPELL_NONE,
        .attack_target_entity_idx = 0,
    };
    osrs_render_entity_from_npc_spec(&spec, re);
    re->debug_npc_type_name = debug_name;
}

static void col_fill_render_entities_ctx(
    EncounterState* state,
    EncounterContext* context,
    RenderEntity* out,
    int max_entities,
    int* count
) {
    ColosseumContext* ctx = col_require_context(context);
    ColosseumState* s = (ColosseumState*)state;
    const EncounterLoadoutStats* ls = col_live_loadout_stats(s);
    s->player.gui_max_hit = ls->max_hit;
    s->player.gui_attack_speed = ls->attack_speed;
    s->player.gui_attack_range = ls->attack_range;
    s->player.gui_strength_bonus = ls->strength_bonus;

    int n = 0;
    if (n < max_entities) {
        Player render_player;
        col_prepare_render_player(s, ctx->player_spec_fired_kind, &render_player);
        RenderEntity* re = &out[n++];
        osrs_render_entity_from_player_entity(&render_player, re);
        if (ctx->player_render_hit_count < 0 ||
                ctx->player_render_hit_count > ENCOUNTER_RENDER_HITS_MAX) {
            fprintf(stderr, "invalid player render hit count\n");
            abort();
        }
        re->render_hit_count = ctx->player_render_hit_count;
        for (int h = 0; h < ctx->player_render_hit_count; h++)
            re->render_hit_damage[h] = ctx->player_render_hit_damage[h];
    }

    for (int i = 0; i < COLO_MAX_NPCS && n < max_entities; i++) {
        ColoNPC* npc = &s->npcs[i];
        if (!npc->active) continue;
        int def_id = COLO_NPC_DEF_IDS[npc->type];
        if (npc->type == COLO_MINOTAUR && col_mod_active(s, COLO_MOD_RED_FLAG))
            def_id = COLO_NPC_DEF_ID_MINOTAUR_ROUTEFIND;
        int npc_anim_id = -1;
        if (npc->death_ticks > 0)
            npc_anim_id = col_npc_death_anim_id(npc->type);
        else if ((npc->type == COLO_MANTICORE && ctx->manticore_charge_started[i]) ||
                 npc->attacked_this_tick)
            npc_anim_id = col_npc_render_attack_anim_id(s, ctx, i, npc, def_id);
        RenderEntity* re = &out[n++];
        OsrsNpcRenderEntitySpec npc_spec = {
            .npc_def_id = def_id,
            .npc_slot = i,
            .npc_instance_id = npc->render_id,
            .npc_visible = npc->active,
            .npc_size = npc->size,
            .npc_anim_id = npc_anim_id,
            .x = npc->x,
            .y = npc->y,
            .dest_x = npc->target_x,
            .dest_y = npc->target_y,
            .current_hitpoints = npc->hp,
            .base_hitpoints = npc->max_hp,
            .attack_style_this_tick = npc->attacked_this_tick
                ? npc->attack_style_this_tick : ATTACK_STYLE_NONE,
            .hit_landed_this_tick = npc->hit_landed_this_tick,
            .hit_damage = npc->hit_damage,
            .hit_was_successful = npc->hit_was_successful_this_tick,
            .hit_spell_type = ENCOUNTER_SPELL_NONE,
            .attack_target_entity_idx = 0,
        };
        osrs_render_entity_from_npc_spec(&npc_spec, re);
        if (ctx->npc_render_hit_count[i] < 0 ||
                ctx->npc_render_hit_count[i] > ENCOUNTER_RENDER_HITS_MAX) {
            fprintf(stderr, "invalid npc render hit count\n");
            abort();
        }
        re->render_hit_count = ctx->npc_render_hit_count[i];
        for (int h = 0; h < ctx->npc_render_hit_count[i]; h++)
            re->render_hit_damage[h] = ctx->npc_render_hit_damage[i][h];
        re->is_running = col_type_is_warbander(npc->type) ? 1 : 0;
        re->frozen_ticks = npc->frozen_ticks;
        re->debug_npc_type_name = colo_npc_type_name(npc->type);
        re->debug_attack_timer = npc->attack_timer;
        re->debug_attack_style = npc->attack_style;
        if (npc->type == COLO_MANTICORE) {
            const ColoManticoreState* mc = &npc->type_state.manticore;
            re->debug_manticore_state_active = 1;
            re->debug_manticore_cycle_step = mc->cycle_step;
            for (int orb = 0; orb < 3; orb++)
                re->debug_manticore_orb_style[orb] = mc->orb_style[orb];
        }
    }

    if (n < max_entities) {
        int tx, ty;
        col_thrall_render_tile(s, ctx, &tx, &ty);
        col_emit_decoration_entity(out, &n, max_entities,
            COLO_NPC_DEF_ID_THRALL, COLO_NPC_SLOT_THRALL, COLO_THRALL_RENDER_INSTANCE_ID,
            ctx->thrall_attacked_this_tick ? 11101 : -1, tx, ty, "Arceuus Thrall");
    }

    if (s->solarflare.active) {
        for (int p = 0; p < COLO_NUM_PILLARS && n < max_entities; p++) {
            int x, y;
            col_solarflare_tile(s, p, s->solarflare.step, &x, &y);
            col_emit_decoration_entity(out, &n, max_entities,
                COLO_NPC_DEF_ID_SOLARFLARE, COLO_NPC_SLOT_SOLARFLARE + p,
                COLO_SOLARFLARE_RENDER_INSTANCE_ID + (uint32_t)p,
                COLO_SOLARFLARE_IDLE_ANIM_ID, x, y, "Solar flare");
        }
    }

    if (s->wave == COLO_WAVE_BOSS && s->sol.started) {
        for (int c = 0; c < s->sol.crystal_count && n < max_entities; c++) {
            const ColoSolCrystal* cr = &s->sol.crystals[c];
            if (!cr->active) continue;
            col_emit_decoration_entity(out, &n, max_entities,
                COLO_NPC_DEF_ID_BEAM_CRYSTAL, COLO_NPC_SLOT_BEAM_CRYSTAL + c,
                COLO_BEAM_CRYSTAL_RENDER_INSTANCE_ID + (uint32_t)c,
                COLO_BEAM_CRYSTAL_IDLE_ANIM_ID, cr->x, cr->y, "Beam crystal");
        }
    }

    encounter_resolve_attack_target(out, n, s->interaction.target_slot);
    *count = n;
}
