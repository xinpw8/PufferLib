#line 76 "encounter_colosseum.h"

static int col_grid_index(int x, int y, int* gx, int* gy) {
    *gx = x - COLO_ARENA_MIN_X;
    *gy = y - COLO_ARENA_MIN_Y;
    return *gx >= 0 && *gx < COLO_ARENA_WIDTH && *gy >= 0 && *gy < COLO_ARENA_HEIGHT;
}

static int col_npc_effective_size(const ColoNPC* npc) {
    return npc->size > 0 ? npc->size : COLO_NPC_STATS[npc->type].size;
}

static void col_stamp_npc_collision_footprint(ColosseumState* s, int x, int y, int size, uint8_t v) {
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int gx, gy;
            if (col_grid_index(x + dx, y + dy, &gx, &gy))
                s->npc_collision_flags[gx][gy] = v;
        }
    }
}

static void col_rebuild_player_collision_flags(ColosseumState* s) {
    memset(s->player_collision_flags, 0, sizeof(s->player_collision_flags));
    int gx, gy;
    if (col_grid_index(s->player.x, s->player.y, &gx, &gy))
        s->player_collision_flags[gx][gy] = 1;
}

static int col_find_free_npc(ColosseumState* s) {
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        if (!s->npcs[i].active) return i;
    }
    return -1;
}

static void col_reset_current_wave_score_progress(ColosseumState* s) {
    s->current_wave_total_killable = 0;
    s->current_wave_hp_pool = 0;
    s->current_wave_fresh_damage = 0.0f;
}

static void col_record_score_spawn(ColosseumState* s, ColoNpcType type) {
    if (s->wave == COLO_WAVE_BOSS) return;
    if (col_type_is_hazard_entity(type)) return;
    s->current_wave_total_killable++;
    s->current_wave_hp_pool += COLO_NPC_STATS[type].hp;
}

static void col_deactivate_npc(ColosseumState* s, int idx) {
    ColoNPC* npc = &s->npcs[idx];

    if (!col_type_is_hazard_entity(npc->type) && !col_type_is_warbander(npc->type)) {
        int size = col_npc_effective_size(npc);
        col_stamp_npc_collision_footprint(s, npc->x, npc->y, size, 0);
    }
    npc->active = 0;
}

static void col_roll_manticore_orb_cycle(ColosseumState* s, AttackStyle out[3]) {
    int mantimayhem = col_mod_tier(s, COLO_MOD_MANTIMAYHEM);
    if (mantimayhem >= 3) {
        out[0] = ATTACK_STYLE_RANGED;
        out[1] = ATTACK_STYLE_MAGIC;
        out[2] = ATTACK_STYLE_MELEE;
        for (int o = 2; o > 0; o--) {
            int j = encounter_rand_int(&s->rng_state, o + 1);
            AttackStyle tmp = out[o];
            out[o] = out[j];
            out[j] = tmp;
        }
    } else if (encounter_rand_int(&s->rng_state, 2) == 0) {
        out[0] = ATTACK_STYLE_RANGED;
        out[1] = ATTACK_STYLE_MAGIC;
        out[2] = ATTACK_STYLE_MELEE;
    } else {
        out[0] = ATTACK_STYLE_MAGIC;
        out[1] = ATTACK_STYLE_RANGED;
        out[2] = ATTACK_STYLE_MELEE;
    }
}

static const AttackStyle* col_wave_manticore_orb_cycle(ColosseumState* s) {
    if (!s->wave_manticore_pattern_rolled) {
        col_roll_manticore_orb_cycle(s, s->wave_manticore_orb_style);
        s->wave_manticore_pattern_rolled = 1;
    }
    return s->wave_manticore_orb_style;
}

static void col_init_npc(ColosseumState* s, int idx, ColoNpcType type, int x, int y) {
    ColoNPC* npc = &s->npcs[idx];
    memset(npc, 0, sizeof(*npc));
    npc->type = type;
    colo_npc_init_type_state(npc);
    if (type == COLO_MANTICORE) {
        const AttackStyle* cycle = col_wave_manticore_orb_cycle(s);
        ColoManticoreState* mc = colo_npc_manticore(npc);
        for (int o = 0; o < 3; o++) mc->fixed_orb_style[o] = cycle[o];
    }
    const ColoNpcStats* st = &COLO_NPC_STATS[type];
    npc->x = x;
    npc->y = y;
    npc->hp = st->hp;
    npc->max_hp = st->hp;
    npc->min_hp_seen = st->hp;
    npc->size = st->size;
    npc->render_id = ++s->next_npc_render_id;
    npc->attack_timer = st->attack_speed;
    npc->attack_style = st->default_style;
    npc->active = 1;
    npc->target_x = x;
    npc->target_y = y;
    npc->stun_timer = 0;

    if (!col_type_is_hazard_entity(type) && !col_type_is_warbander(type) &&
            col_npc_effective_size(npc) > 0)
        col_stamp_npc_collision_footprint(s, x, y, col_npc_effective_size(npc), 1);
}

static int col_spawn_npc_at(ColosseumState* s, ColoNpcType type, int x, int y) {
    int slot = col_find_free_npc(s);
    assert(slot >= 0 && "colosseum NPC slots exhausted");
    col_init_npc(s, slot, type, x, y);
    col_record_score_spawn(s, type);
    return slot;
}

static int col_spawn_excluded_near_player(const ColosseumState* s, int x, int y, int size) {
    return encounter_rect_distance(
        s->player.x, s->player.y, 1, x, y, size) <= COLO_SPAWN_EXCLUSION_CHEB;
}

static int col_spawn_footprint_free(const ColosseumState* s, int x, int y, int size) {
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int tx = x + dx, ty = y + dy;
            if (col_static_blocked(tx, ty)) return 0;
            if (col_sol_clamp_active(s) && !col_in_boss_arena(s, tx, ty)) return 0;
            int gx, gy;
            if (!col_grid_index(tx, ty, &gx, &gy)) return 0;
            if (s->npc_collision_flags[gx][gy]) return 0;
            if (tx == s->player.x && ty == s->player.y) return 0;
        }
    }
    return 1;
}

static int col_spawn_anchor_candidates(const ColosseumState* s, int* out) {
    int n = 0;
    for (int a = 0; a < COLO_NUM_SPAWN_ANCHORS; a++) {
        int ax = COLO_SPAWN_ANCHORS[a][0], ay = COLO_SPAWN_ANCHORS[a][1];
        if (col_spawn_excluded_near_player(s, ax, ay, COLO_SPAWN_ZONE_SIZE)) continue;
        if (!col_spawn_footprint_free(s, ax, ay, COLO_SPAWN_ZONE_SIZE)) continue;
        out[n++] = a;
    }
    return n;
}

static int col_warband_spawn_tile_ok(const ColosseumState* s, int x, int y) {
    if (col_static_blocked(x, y)) return 0;
    if (col_sol_clamp_active(s) && !col_in_boss_arena(s, x, y)) return 0;
    return !(x == s->player.x && y == s->player.y);
}

static void col_draw_warband_box_tile(ColosseumState* s, int* ox, int* oy) {
    int span_x = COLO_WARBAND_BOX_MAX_X - COLO_WARBAND_BOX_MIN_X + 1;
    int span_y = COLO_WARBAND_BOX_MAX_Y - COLO_WARBAND_BOX_MIN_Y + 1;
    for (int t = 0; t < 32; t++) {
        int x = COLO_WARBAND_BOX_MIN_X + encounter_rand_int(&s->rng_state, span_x);
        int y = COLO_WARBAND_BOX_MIN_Y + encounter_rand_int(&s->rng_state, span_y);
        if (col_warband_spawn_tile_ok(s, x, y)) { *ox = x; *oy = y; return; }
    }
    for (int y = COLO_WARBAND_BOX_MIN_Y; y <= COLO_WARBAND_BOX_MAX_Y; y++)
        for (int x = COLO_WARBAND_BOX_MIN_X; x <= COLO_WARBAND_BOX_MAX_X; x++)
            if (col_warband_spawn_tile_ok(s, x, y)) { *ox = x; *oy = y; return; }
    assert(0 && "warband centre box full");
}

static void col_spawn_warbander_formation(
    ColosseumState* s, ColoNpcType type, int ax, int ay,
    int dx, int dy, int dir) {
    int x = ax + dx, y = ay + dy;
    if (!col_warband_spawn_tile_ok(s, x, y)) {
        int found = 0;
        for (int r = 1; r <= 3 && !found; r++) {
            for (int oy = -r; oy <= r && !found; oy++) {
                for (int ox = -r; ox <= r && !found; ox++) {
                    if (ox > -r && ox < r && oy > -r && oy < r) continue;
                    if (col_warband_spawn_tile_ok(s, x + ox, y + oy)) {
                        x += ox; y += oy; found = 1;
                    }
                }
            }
        }
        if (!found) col_draw_warband_box_tile(s, &x, &y);
    }
    int slot = col_spawn_npc_at(s, type, x, y);
    colo_npc_warband(&s->npcs[slot])->formation_dir = dir;
}

static void col_spawn_pair_adjacent(ColosseumState* s, ColoNpcType type, int px, int py) {
    int size = COLO_NPC_STATS[type].size;
    const int off[8][2] = {
        {size, 0}, {-size, 0}, {0, size}, {0, -size},
        {size, size}, {-size, -size}, {size, -size}, {-size, size} };
    for (int o = 0; o < 8; o++) {
        int x = px + off[o][0], y = py + off[o][1];
        if (!col_spawn_footprint_free(s, x, y, size)) continue;
        col_spawn_npc_at(s, type, x, y);
        return;
    }
    for (int r = 1; r <= size + 2; r++) {
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                if (dx > -r && dx < r && dy > -r && dy < r) continue;
                int x = px + dx, y = py + dy;
                if (!col_spawn_footprint_free(s, x, y, size)) continue;
                col_spawn_npc_at(s, type, x, y);
                return;
            }
        }
    }
    int best = -1, best_dist = 1 << 30;
    for (int a = 0; a < COLO_NUM_SPAWN_ANCHORS; a++) {
        int ax = COLO_SPAWN_ANCHORS[a][0], ay = COLO_SPAWN_ANCHORS[a][1];
        if (col_spawn_excluded_near_player(s, ax, ay, COLO_SPAWN_ZONE_SIZE)) continue;
        if (!col_spawn_footprint_free(s, ax, ay, COLO_SPAWN_ZONE_SIZE)) continue;
        int dx = ax > px ? ax - px : px - ax;
        int dy = ay > py ? ay - py : py - ay;
        int cheb = dx > dy ? dx : dy;
        if (cheb < best_dist) { best_dist = cheb; best = a; }
    }
    assert(best >= 0 && "no spawn spot for the Dynamic Duo pair");
    col_spawn_npc_at(s, type, COLO_SPAWN_ANCHORS[best][0], COLO_SPAWN_ANCHORS[best][1]);
}

static void col_spawn_quartet_boss_wave(ColosseumState* s, ColoNpcType type) {
    int span = COLO_BOSS_ARENA_SIZE;
    int qx = -1, qy = -1;
    for (int t = 0; t < 64 && qx < 0; t++) {
        int x = COLO_BOSS_ARENA_MIN_X + encounter_rand_int(&s->rng_state, span);
        int y = COLO_BOSS_ARENA_MIN_Y + encounter_rand_int(&s->rng_state, span);
        if (col_spawn_excluded_near_player(s, x, y, 1)) continue;
        if (!col_spawn_footprint_free(s, x, y, 1)) continue;
        qx = x; qy = y;
    }
    for (int y = COLO_BOSS_ARENA_MIN_Y; y <= COLO_BOSS_ARENA_MAX_Y && qx < 0; y++) {
        for (int x = COLO_BOSS_ARENA_MIN_X; x <= COLO_BOSS_ARENA_MAX_X && qx < 0; x++) {
            if (col_spawn_excluded_near_player(s, x, y, 1)) continue;
            if (!col_spawn_footprint_free(s, x, y, 1)) continue;
            qx = x; qy = y;
        }
    }
    assert(qx >= 0 && "no spawn tile for the wave-12 Quartet warbander");
    int slot = col_spawn_npc_at(s, type, qx, qy);
    colo_npc_warband(&s->npcs[slot])->formation_dir = COLO_WARBAND_FORM_SOUTH;
}

static const ColoNpcType COLO_WARBAND_TYPES[3] = {
    COLO_FREMENNIK_BERSERKER, COLO_FREMENNIK_ARCHER, COLO_FREMENNIK_SEER };

static void col_spawn_wave(ColosseumState* s) {
    col_reset_current_wave_score_progress(s);

    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        if (!s->npcs[i].active) continue;
        if (s->npcs[i].type == COLO_BEE_SWARM) continue;
        col_deactivate_npc(s, i);
    }
    memset(s->totems, 0, sizeof(s->totems));
    memset(s->npc_collision_flags, 0, sizeof(s->npc_collision_flags));

    s->warband_cycle_anchor = -1;
    s->wave_manticore_pattern_rolled = 0;

    if (s->wave < 0 || s->wave >= COLO_NUM_WAVES) return;
    const ColoWaveDef* def = &COLO_WAVES[s->wave];

    if (s->wave == COLO_WAVE_BOSS) {
        col_sol_begin_boss_arena(s);
        for (int i = 0; i < def->count; i++) {
            assert((ColoNpcType)def->types[i] == COLO_SOL_HEREDIT);
            col_spawn_npc_at(s, COLO_SOL_HEREDIT, COLO_SOL_SPAWN_X, COLO_SOL_SPAWN_Y);
        }
        if (col_mod_active(s, COLO_MOD_QUARTET)) {
            ColoNpcType extra = COLO_WARBAND_TYPES[encounter_rand_int(&s->rng_state, 3)];
            col_spawn_quartet_boss_wave(s, extra);
        }
    } else {
        int archer_x = -1, archer_y = -1;
        for (int i = 0; i < def->count; i++) {
            if ((ColoNpcType)def->types[i] != COLO_FREMENNIK_ARCHER) continue;
            col_draw_warband_box_tile(s, &archer_x, &archer_y);
            int slot = col_spawn_npc_at(s, COLO_FREMENNIK_ARCHER, archer_x, archer_y);
            colo_npc_warband(&s->npcs[slot])->formation_dir = COLO_WARBAND_FORM_WEST;
        }
        for (int i = 0; i < def->count; i++) {
            ColoNpcType type = (ColoNpcType)def->types[i];
            if (!col_type_is_warbander(type) || type == COLO_FREMENNIK_ARCHER) continue;
            assert(archer_x >= 0 && "warband def lists berserker/seer without an archer");
            if (type == COLO_FREMENNIK_SEER)
                col_spawn_warbander_formation(s, type, archer_x, archer_y,
                    2, 0, COLO_WARBAND_FORM_EAST);
            else
                col_spawn_warbander_formation(s, type, archer_x, archer_y,
                    1, 1, COLO_WARBAND_FORM_NORTH);
        }

        int candidates[COLO_NUM_SPAWN_ANCHORS];
        int ncand = col_spawn_anchor_candidates(s, candidates);
        encounter_shuffle(candidates, ncand, &s->rng_state);
        int cursor = 0;
        for (int i = 0; i < def->count; i++) {
            ColoNpcType type = (ColoNpcType)def->types[i];
            if (col_type_is_warbander(type)) continue;
            int anchor = -1;
            while (cursor < ncand) {
                int a = candidates[cursor++];
                if (col_spawn_footprint_free(
                        s, COLO_SPAWN_ANCHORS[a][0], COLO_SPAWN_ANCHORS[a][1],
                        COLO_SPAWN_ZONE_SIZE)) { anchor = a; break; }
            }
            assert(anchor >= 0 && "primary spawn ran out of anchor candidates");
            int ax = COLO_SPAWN_ANCHORS[anchor][0], ay = COLO_SPAWN_ANCHORS[anchor][1];
            col_spawn_npc_at(s, type, ax, ay);
            if (type == COLO_SHOCKWAVE_COLOSSUS && col_mod_active(s, COLO_MOD_DYNAMIC_DUO))
                col_spawn_pair_adjacent(s, type, ax, ay);
        }

        if (col_mod_active(s, COLO_MOD_QUARTET)) {
            ColoNpcType extra = COLO_WARBAND_TYPES[encounter_rand_int(&s->rng_state, 3)];
            assert(archer_x >= 0 && "quartet without an archer spawn");
            col_spawn_warbander_formation(s, extra, archer_x, archer_y,
                1, -1, COLO_WARBAND_FORM_SOUTH);
        }
    }

    s->reinforcement_timer = COLO_REINFORCEMENT_TICKS;

    s->wave_ready_delay = COLO_WAVE_READY_TICKS;
    s->wave_attack_delay = COLO_WAVE_ATTACK_GATE_TICKS;
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        ColoNPC* npc = &s->npcs[i];
        if (!npc->active) continue;
        if (npc->type == COLO_BEE_SWARM || npc->type == COLO_SOL_HEREDIT) continue;
        npc->attack_timer = 0;
    }
    col_modifiers_on_wave_spawn(s);
}

#define COLO_REINFORCE_FIRED (-1)

static void col_spawn_reinforcements(ColosseumState* s) {
    if (s->wave < 0 || s->wave >= COLO_NUM_WAVES) return;
    const ColoWaveDef* def = &COLO_WAVES[s->wave];
    if (def->reinforce_count <= 0) {
        s->reinforcement_timer = COLO_REINFORCE_FIRED;
        return;
    }

    int north = s->player.y >= COLO_GATE_LINE_NORTH_MIN_Y;
    int base_x = COLO_GATE_MIN_X;
    for (int i = 0; i < def->reinforce_count; i++) {
        ColoNpcType type = (ColoNpcType)def->reinforce[i];
        int size = COLO_NPC_STATS[type].size;
        int y = north ? COLO_GATE_NORTH_SPAWN_ROW - (size - 1) : COLO_GATE_SOUTH_SPAWN_ROW;
        int x = base_x;
        for (int try_x = base_x; try_x + size - 1 <= COLO_GATE_MAX_X; try_x++) {
            if (col_spawn_footprint_free(s, try_x, y, size)) { x = try_x; break; }
        }
        int slot = col_spawn_npc_at(s, type, x, y);
        s->npcs[slot].spawned_as_reinforcement = 1;
        base_x = x + size;
    }

    s->reinforcement_timer = COLO_REINFORCE_FIRED;
}

static void col_init_empty_inventory_cells(ColosseumState* s) {
    for (int i = 0; i < COLO_INVENTORY_DISPLAY_SLOTS; i++) {
        s->inventory_cells[i] = osrs_inventory_cell_empty();
    }
}

static void col_seed_inventory_cells_from_gameplay_loadout(ColosseumState* s) {
    col_init_empty_inventory_cells(s);
    const int* kit = COLO_GAMEPLAY_INVENTORY_LOADOUT[s->active_loadout_profile];
    for (int i = 0; i < COLO_INVENTORY_DISPLAY_SLOTS; i++) {
        uint16_t raw_id = (uint16_t)kit[i];
        if (raw_id == 0) continue;

        s->inventory_cells[i] = osrs_inventory_cell_from_raw_osrs_id(raw_id);
        if (s->inventory_cells[i].item_idx == ITEM_NONE &&
                s->inventory_cells[i].dose == 0 &&
                raw_id != 27281) {
            fprintf(stderr, "colosseum: gameplay inventory raw id %u has no cell mapping\n",
                raw_id);
            abort();
        }
    }
}

static void col_sync_consumable_counters_from_inventory(ColosseumState* s) {
    int brew_doses = 0;
    int restore_doses = 0;
    int combat_doses = 0;
    int ranged_doses = 0;
    int surge_doses = 0;
    int antivenom_doses = 0;
    for (int i = 0; i < COLO_INVENTORY_DISPLAY_SLOTS; i++) {
        const ColoInvCell* cell = &s->inventory_cells[i];
        if (cell->dose == 0) continue;
        OsrsInventoryClickResolution resolution =
            osrs_inventory_cell_click_interpret(cell, OSRS_CLICK_TICK_FIRST);
        if (resolution.click_action != OSRS_CLICK_DRINK) continue;
        switch (resolution.consumable_kind) {
            case OSRS_CONSUMABLE_BREW:
                brew_doses += cell->dose;
                break;
            case OSRS_CONSUMABLE_SUPER_RESTORE:
            case OSRS_CONSUMABLE_SANFEW:
                restore_doses += cell->dose;
                break;
            case OSRS_CONSUMABLE_SUPER_COMBAT:
            case OSRS_CONSUMABLE_DIVINE_COMBAT:
                combat_doses += cell->dose;
                break;
            case OSRS_CONSUMABLE_RANGING:
            case OSRS_CONSUMABLE_DIVINE_RANGING:
                ranged_doses += cell->dose;
                break;
            case OSRS_CONSUMABLE_SURGE:
                surge_doses += cell->dose;
                break;
            case OSRS_CONSUMABLE_ANTIVENOM_PLUS:
                antivenom_doses += cell->dose;
                break;
            case OSRS_CONSUMABLE_GUTHIX_REST:
            case OSRS_CONSUMABLE_SATURATED_HEART:
            case OSRS_CONSUMABLE_NONE:
            case OSRS_CONSUMABLE_SHARK_FOOD:
            case OSRS_CONSUMABLE_KARAMBWAN:
                break;
        }
    }
    s->player.brew_doses = brew_doses;
    s->player.restore_doses = restore_doses;
    s->player.combat_potion_doses = combat_doses;
    s->player.ranged_potion_doses = ranged_doses;
    s->player.antivenom_doses = antivenom_doses;
    s->surge_doses = surge_doses;
}

static void col_strip_brews_from_inventory(ColosseumState* s) {
    for (int i = 0; i < COLO_INVENTORY_DISPLAY_SLOTS; i++) {
        ColoInvCell* cell = &s->inventory_cells[i];
        if (cell->dose == 0) continue;
        OsrsInventoryClickResolution resolution =
            osrs_inventory_cell_click_interpret(cell, OSRS_CLICK_TICK_FIRST);
        if (resolution.consumable_kind == OSRS_CONSUMABLE_BREW)
            *cell = osrs_inventory_cell_empty();
    }
}

static void col_reset_ctx(EncounterState* state, EncounterContext* context, uint32_t seed) {
    col_build_npc_stats();
    ColosseumState* s = (ColosseumState*)state;
    ColosseumContext* ctx = context ? (ColosseumContext*)context : col_legacy_context();
    int saved_start = ctx->config.start_wave;
    uint32_t saved_rng = s->rng_state;
    col_context_clear_render_events(ctx);

    ColoObsMemoCaches saved_obs_memos = s->obs_memos;
    memset(s, 0, sizeof(ColosseumState));
    s->obs_memos = saved_obs_memos;
    s->start_wave = saved_start;
    s->rng_state = encounter_resolve_seed(saved_rng, seed);

    s->player_dest_x = -1;
    s->player_dest_y = -1;
    s->last_hit_by_type = -1;
    s->min_sol_hp_seen = COLO_SOL_HP_MAX;

    s->active_player_damage_received_scale = ctx->config.player_damage_received_scale;
    for (int i = 0; i < COLO_OBS_NPCS; i++) s->current_obs_slots[i] = -1;

    s->player.entity_type = ENTITY_PLAYER;
    encounter_init_maxed_player_combat_stats(&s->player, 99);
    osrs_item_effect_state_init(&s->player.item_effect_state);

    s->weapon_set = COLO_GEAR_MELEE;
    s->active_loadout_profile = col_sample_loadout_profile(
        ctx->config.loadout_profile_mode,
        ctx->config.beginner_loadout_fraction,
        &s->rng_state);
    switch (s->active_loadout_profile) {
        case COLO_LOADOUT_PROFILE_SPEEDRUN:
            s->episode_max_ticks = COLO_SPEEDRUN_MAX_TICKS;
            break;
        case COLO_LOADOUT_PROFILE_BEGINNER:
            s->episode_max_ticks = COLO_BEGINNER_MAX_TICKS;
            break;
        default:
            fprintf(stderr, "colosseum: invalid active_loadout_profile %d\n",
                (int)s->active_loadout_profile);
            abort();
    }
    if (ctx->config.episode_max_ticks_override > 0)
        s->episode_max_ticks = ctx->config.episode_max_ticks_override;
    const uint8_t* const* loadouts = col_loadouts_for_profile(s->active_loadout_profile);
    encounter_apply_loadout(&s->player, loadouts[COLO_GEAR_MELEE], GEAR_MELEE);
    memset(s->player.inventory, ITEM_NONE, sizeof(s->player.inventory));
    memset(s->player.num_items_in_slot, 0, sizeof(s->player.num_items_in_slot));
    col_seed_inventory_cells_from_gameplay_loadout(s);
    if (ctx->config.remove_brews) col_strip_brews_from_inventory(s);

    const ColoSupplyLoadout* supplies = &COLO_SUPPLY_LOADOUTS[s->active_loadout_profile];
    s->full_supplies = *supplies;
    col_sync_consumable_counters_from_inventory(s);
    s->surge_cooldown = 0;

    s->player.prayer = PRAYER_NONE;
    s->player.offensive_prayer = OFFENSIVE_PRAYER_NONE;
    osrs_interaction_init(&s->interaction);
    s->player_last_interaction_target_slot = -1;
    s->player_last_interaction_age = 1;
    s->player.spec_armed = 0;
    s->player.special_energy = 100;
    s->player.run_energy = 10000;

    col_mark_live_loadout_dirty(s);
    col_ensure_live_loadout_stats(s);

    s->thrall_target_slot = -1;

    s->player.x = COLO_PLAYER_START_X;
    s->player.y = COLO_PLAYER_START_Y;
    col_rebuild_player_collision_flags(s);

    int effective_start = s->start_wave;
    if (effective_start < 0) effective_start = 0;
    if (effective_start >= COLO_NUM_WAVES) effective_start = COLO_NUM_WAVES - 1;
    s->wave = effective_start;
    s->wave_spawn_target = effective_start;
    s->wave_spawn_delay = 0;
    s->wave_ready_delay = 0;
    s->wave_attack_delay = 0;
    s->log.wave_reached = effective_start;

    if (effective_start == 0) {
        col_modifier_open_draft(s, 0);
        if (s->modifiers.draft_pending) s->modifiers.draft_free_movement = 1;
        else s->wave_spawn_delay = col_wave_entry_delay_ticks(0);
    } else if (ctx->config.late_start_state_mode > 0) {
        col_apply_late_start_entry_state(s, ctx, effective_start);
        col_modifier_open_draft(s, effective_start);
        if (!s->modifiers.draft_pending)
            s->wave_spawn_delay = col_wave_entry_delay_ticks(effective_start);
    } else {
        s->wave_spawn_delay = col_wave_entry_delay_ticks(effective_start);
    }
}

static void col_reset(EncounterState* state, uint32_t seed) {
    col_reset_ctx(state, (EncounterContext*)col_legacy_context(), seed);
}
