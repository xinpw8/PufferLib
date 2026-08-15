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
    s->log.npc_stamp_tiles += (float)(size * size);
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int gx, gy;
            if (col_grid_index(x + dx, y + dy, &gx, &gy))
                s->npc_collision_flags[gx][gy] = v;
        }
    }
}

static int col_npc_blocks_other_npcs(const ColosseumState* s, ColoNpcType type) {
    return !(type == COLO_MINOTAUR && col_mod_active(s, COLO_MOD_RED_FLAG));
}

static void col_stamp_npc_collision(ColosseumState* s, const ColoNPC* npc, uint8_t v) {
    if (!col_npc_blocks_other_npcs(s, npc->type)) return;
    col_stamp_npc_collision_footprint(
        s, npc->x, npc->y, col_npc_effective_size(npc), v);
}

static void col_rebuild_player_collision_flags(ColosseumState* s) {
    if (!col_grid_index(s->player.x, s->player.y, &s->player_grid_x, &s->player_grid_y)) {
        s->player_grid_x = -1;
        s->player_grid_y = -1;
    }
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

    if (!col_type_is_hazard_entity(npc->type) && !col_type_is_warbander(npc->type))
        col_stamp_npc_collision(s, npc, 0);
    npc->active = 0;
    npc->attacked_this_tick = 0;
    npc->attack_style_this_tick = ATTACK_STYLE_NONE;
    npc->moved_this_tick = 0;
    npc->moved_last_tick = 0;
    npc->hit_landed_this_tick = 0;
    npc->hit_damage = 0;
    npc->hit_was_successful_this_tick = 0;
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
    npc->attack_timer = type == COLO_MANTICORE
        ? COLO_MANTICORE_ACTIVATION_TICKS
        : st->attack_speed;
    npc->attack_style = st->default_style;
    npc->active = 1;
    npc->target_x = x;
    npc->target_y = y;
    npc->stun_timer = 0;

    if (!col_type_is_hazard_entity(type) && !col_type_is_warbander(type) &&
            col_npc_effective_size(npc) > 0)
        col_stamp_npc_collision(s, npc, 1);
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

static int col_spawn_footprint_free(
    const ColosseumState* s,
    const EncounterArenaTopology* topology,
    int x,
    int y,
    int size
) {
    if (encounter_arena_topology_footprint_blocked(
            topology, x, y, size))
        return 0;
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int tx = x + dx;
            int ty = y + dy;
            if (col_sol_clamp_active(s) &&
                    !col_in_boss_arena(s, tx, ty))
                return 0;
            int gx, gy;
            if (!col_grid_index(tx, ty, &gx, &gy)) return 0;
            if (s->npc_collision_flags[gx][gy]) return 0;
            if (tx == s->player.x && ty == s->player.y) return 0;
        }
    }
    return 1;
}

static int col_spawn_anchor_candidates(
    const ColosseumState* s,
    const EncounterArenaTopology* topology,
    int* out
) {
    int n = 0;
    for (int a = 0; a < COLO_NUM_SPAWN_ANCHORS; a++) {
        int ax = COLO_SPAWN_ANCHORS[a][0];
        int ay = COLO_SPAWN_ANCHORS[a][1];
        if (col_spawn_excluded_near_player(
                s, ax, ay, COLO_SPAWN_ZONE_SIZE))
            continue;
        if (!col_spawn_footprint_free(
                s, topology, ax, ay, COLO_SPAWN_ZONE_SIZE))
            continue;
        out[n++] = a;
    }
    return n;
}

static int col_warband_spawn_tile_ok(
    const ColosseumState* s,
    const EncounterArenaTopology* topology,
    int x,
    int y
) {
    if (encounter_arena_topology_tile_blocked(topology, x, y)) return 0;
    if (col_sol_clamp_active(s) && !col_in_boss_arena(s, x, y)) return 0;
    return !(x == s->player.x && y == s->player.y);
}

static void col_draw_warband_box_tile(
    ColosseumState* s,
    const EncounterArenaTopology* topology,
    int* out_x,
    int* out_y
) {
    int span_x = COLO_WARBAND_BOX_MAX_X - COLO_WARBAND_BOX_MIN_X + 1;
    int span_y = COLO_WARBAND_BOX_MAX_Y - COLO_WARBAND_BOX_MIN_Y + 1;
    for (int attempt = 0; attempt < 32; attempt++) {
        int x = COLO_WARBAND_BOX_MIN_X +
            encounter_rand_int(&s->rng_state, span_x);
        int y = COLO_WARBAND_BOX_MIN_Y +
            encounter_rand_int(&s->rng_state, span_y);
        if (col_warband_spawn_tile_ok(s, topology, x, y)) {
            *out_x = x;
            *out_y = y;
            return;
        }
    }
    for (int y = COLO_WARBAND_BOX_MIN_Y; y <= COLO_WARBAND_BOX_MAX_Y; y++) {
        for (int x = COLO_WARBAND_BOX_MIN_X; x <= COLO_WARBAND_BOX_MAX_X; x++) {
            if (!col_warband_spawn_tile_ok(s, topology, x, y)) continue;
            *out_x = x;
            *out_y = y;
            return;
        }
    }
    assert(0 && "warband centre box full");
}

static void col_spawn_warbander_formation(
    ColosseumState* s,
    const EncounterArenaTopology* topology,
    ColoNpcType type,
    int anchor_x,
    int anchor_y,
    int dx,
    int dy,
    int direction
) {
    int x = anchor_x + dx;
    int y = anchor_y + dy;
    if (!col_warband_spawn_tile_ok(s, topology, x, y)) {
        int found = 0;
        for (int radius = 1; radius <= 3 && !found; radius++) {
            for (int offset_y = -radius;
                    offset_y <= radius && !found;
                    offset_y++) {
                for (int offset_x = -radius;
                        offset_x <= radius && !found;
                        offset_x++) {
                    if (offset_x > -radius && offset_x < radius &&
                            offset_y > -radius && offset_y < radius)
                        continue;
                    if (col_warband_spawn_tile_ok(
                            s, topology, x + offset_x, y + offset_y)) {
                        x += offset_x;
                        y += offset_y;
                        found = 1;
                    }
                }
            }
        }
        if (!found)
            col_draw_warband_box_tile(s, topology, &x, &y);
    }
    int slot = col_spawn_npc_at(s, type, x, y);
    colo_npc_warband(&s->npcs[slot])->formation_dir = direction;
}

static void col_spawn_pair_adjacent(
    ColosseumState* s,
    const EncounterArenaTopology* topology,
    ColoNpcType type,
    int parent_x,
    int parent_y
) {
    int size = COLO_NPC_STATS[type].size;
    const int offsets[8][2] = {
        {size, 0}, {-size, 0}, {0, size}, {0, -size},
        {size, size}, {-size, -size}, {size, -size}, {-size, size},
    };
    for (int offset = 0; offset < 8; offset++) {
        int x = parent_x + offsets[offset][0];
        int y = parent_y + offsets[offset][1];
        if (!col_spawn_footprint_free(s, topology, x, y, size)) continue;
        col_spawn_npc_at(s, type, x, y);
        return;
    }
    for (int radius = 1; radius <= size + 2; radius++) {
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                if (dx > -radius && dx < radius &&
                        dy > -radius && dy < radius)
                    continue;
                int x = parent_x + dx;
                int y = parent_y + dy;
                if (!col_spawn_footprint_free(
                        s, topology, x, y, size))
                    continue;
                col_spawn_npc_at(s, type, x, y);
                return;
            }
        }
    }
    int best = -1;
    int best_dist = 1 << 30;
    for (int anchor = 0; anchor < COLO_NUM_SPAWN_ANCHORS; anchor++) {
        int x = COLO_SPAWN_ANCHORS[anchor][0];
        int y = COLO_SPAWN_ANCHORS[anchor][1];
        if (col_spawn_excluded_near_player(
                s, x, y, COLO_SPAWN_ZONE_SIZE))
            continue;
        if (!col_spawn_footprint_free(
                s, topology, x, y, COLO_SPAWN_ZONE_SIZE))
            continue;
        int dx = x > parent_x ? x - parent_x : parent_x - x;
        int dy = y > parent_y ? y - parent_y : parent_y - y;
        int distance = dx > dy ? dx : dy;
        if (distance < best_dist) {
            best_dist = distance;
            best = anchor;
        }
    }
    assert(best >= 0 && "no spawn spot for the Dynamic Duo pair");
    col_spawn_npc_at(
        s,
        type,
        COLO_SPAWN_ANCHORS[best][0],
        COLO_SPAWN_ANCHORS[best][1]);
}

static void col_spawn_quartet_boss_wave(
    ColosseumState* s,
    const EncounterArenaTopology* topology,
    ColoNpcType type
) {
    int span = COLO_BOSS_ARENA_SIZE;
    int quartet_x = -1;
    int quartet_y = -1;
    for (int attempt = 0; attempt < 64 && quartet_x < 0; attempt++) {
        int x = COLO_BOSS_ARENA_MIN_X +
            encounter_rand_int(&s->rng_state, span);
        int y = COLO_BOSS_ARENA_MIN_Y +
            encounter_rand_int(&s->rng_state, span);
        if (col_spawn_excluded_near_player(s, x, y, 1)) continue;
        if (!col_spawn_footprint_free(s, topology, x, y, 1)) continue;
        quartet_x = x;
        quartet_y = y;
    }
    for (int y = COLO_BOSS_ARENA_MIN_Y;
            y <= COLO_BOSS_ARENA_MAX_Y && quartet_x < 0;
            y++) {
        for (int x = COLO_BOSS_ARENA_MIN_X;
                x <= COLO_BOSS_ARENA_MAX_X && quartet_x < 0;
                x++) {
            if (col_spawn_excluded_near_player(s, x, y, 1)) continue;
            if (!col_spawn_footprint_free(s, topology, x, y, 1)) continue;
            quartet_x = x;
            quartet_y = y;
        }
    }
    assert(quartet_x >= 0 &&
        "no spawn tile for the wave-12 Quartet warbander");
    int slot = col_spawn_npc_at(s, type, quartet_x, quartet_y);
    colo_npc_warband(&s->npcs[slot])->formation_dir =
        COLO_WARBAND_FORM_SOUTH;
}

static const ColoNpcType COLO_WARBAND_TYPES[3] = {
    COLO_FREMENNIK_BERSERKER, COLO_FREMENNIK_ARCHER, COLO_FREMENNIK_SEER };

static void col_spawn_wave(
    ColosseumState* s,
    ColosseumContext* context
) {
    const EncounterArenaTopology* topology = context->route_topology;
    encounter_arena_topology_require_finalized(topology);
    col_reset_current_wave_score_progress(s);

    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        if (!s->npcs[i].active) continue;
        if (s->npcs[i].type == COLO_BEE_SWARM) continue;
        if (s->npcs[i].death_ticks > 0)
            col_remove_npc_corpse(s, context, i);
        else
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
            col_spawn_quartet_boss_wave(s, topology, extra);
        }
    } else {
        int archer_x = -1, archer_y = -1;
        for (int i = 0; i < def->count; i++) {
            if ((ColoNpcType)def->types[i] != COLO_FREMENNIK_ARCHER) continue;
            col_draw_warband_box_tile(
                s, topology, &archer_x, &archer_y);
            int slot = col_spawn_npc_at(s, COLO_FREMENNIK_ARCHER, archer_x, archer_y);
            colo_npc_warband(&s->npcs[slot])->formation_dir = COLO_WARBAND_FORM_WEST;
        }
        for (int i = 0; i < def->count; i++) {
            ColoNpcType type = (ColoNpcType)def->types[i];
            if (!col_type_is_warbander(type) || type == COLO_FREMENNIK_ARCHER) continue;
            assert(archer_x >= 0 && "warband def lists berserker/seer without an archer");
            if (type == COLO_FREMENNIK_SEER)
                col_spawn_warbander_formation(
                    s, topology, type, archer_x, archer_y,
                    2, 0, COLO_WARBAND_FORM_EAST);
            else
                col_spawn_warbander_formation(
                    s, topology, type, archer_x, archer_y,
                    1, 1, COLO_WARBAND_FORM_NORTH);
        }

        int candidates[COLO_NUM_SPAWN_ANCHORS];
        int ncand =
            col_spawn_anchor_candidates(s, topology, candidates);
        encounter_shuffle(candidates, ncand, &s->rng_state);
        int cursor = 0;
        for (int i = 0; i < def->count; i++) {
            ColoNpcType type = (ColoNpcType)def->types[i];
            if (col_type_is_warbander(type)) continue;
            int anchor = -1;
            while (cursor < ncand) {
                int a = candidates[cursor++];
                if (col_spawn_footprint_free(
                        s, topology,
                        COLO_SPAWN_ANCHORS[a][0],
                        COLO_SPAWN_ANCHORS[a][1],
                        COLO_SPAWN_ZONE_SIZE)) {
                    anchor = a;
                    break;
                }
            }
            assert(anchor >= 0 && "primary spawn ran out of anchor candidates");
            int ax = COLO_SPAWN_ANCHORS[anchor][0], ay = COLO_SPAWN_ANCHORS[anchor][1];
            col_spawn_npc_at(s, type, ax, ay);
            if (type == COLO_SHOCKWAVE_COLOSSUS && col_mod_active(s, COLO_MOD_DYNAMIC_DUO))
                col_spawn_pair_adjacent(s, topology, type, ax, ay);
        }

        if (col_mod_active(s, COLO_MOD_QUARTET)) {
            ColoNpcType extra = COLO_WARBAND_TYPES[encounter_rand_int(&s->rng_state, 3)];
            assert(archer_x >= 0 && "quartet without an archer spawn");
            col_spawn_warbander_formation(
                s, topology, extra, archer_x, archer_y,
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

static void col_spawn_reinforcements(
    ColosseumState* s,
    const ColosseumContext* context
) {
    const EncounterArenaTopology* topology = context->route_topology;
    encounter_arena_topology_require_finalized(topology);
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
            if (col_spawn_footprint_free(
                    s, topology, try_x, y, size)) {
                x = try_x;
                break;
            }
        }
        int slot = col_spawn_npc_at(s, type, x, y);
        s->npcs[slot].spawned_as_reinforcement = 1;
        base_x = x + size;
    }

    s->reinforcement_timer = COLO_REINFORCE_FIRED;
}

static void col_init_empty_inventory_cells(ColosseumState* s) {
    for (int i = 0; i < COLO_INVENTORY_DISPLAY_SLOTS; i++) {
        s->player.inventory_cells[i] = osrs_inventory_cell_empty();
    }
}

static void col_seed_inventory_cells_from_gameplay_loadout(ColosseumState* s) {
    col_init_empty_inventory_cells(s);
    const int* kit = COLO_GAMEPLAY_INVENTORY_LOADOUT[s->active_loadout_profile];
    for (int i = 0; i < COLO_INVENTORY_DISPLAY_SLOTS; i++) {
        uint16_t raw_id = (uint16_t)kit[i];
        if (raw_id == 0) continue;

        s->player.inventory_cells[i] =
            osrs_inventory_cell_from_raw_osrs_id(raw_id);
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
        const ColoInvCell* cell = &s->player.inventory_cells[i];
        uint8_t dose_count = osrs_inventory_cell_dose_count(cell);
        if (dose_count == 0) continue;
        OsrsInventoryClickResolution resolution =
            osrs_inventory_cell_click_interpret(cell, OSRS_CLICK_TICK_FIRST);
        if (resolution.click_action != OSRS_CLICK_DRINK) continue;
        switch (resolution.consumable_kind) {
            case OSRS_CONSUMABLE_BREW:
                brew_doses += dose_count;
                break;
            case OSRS_CONSUMABLE_SUPER_RESTORE:
            case OSRS_CONSUMABLE_SANFEW:
            case OSRS_CONSUMABLE_PRAYER_RESTORE:
                restore_doses += dose_count;
                break;
            case OSRS_CONSUMABLE_SUPER_COMBAT:
            case OSRS_CONSUMABLE_DIVINE_COMBAT:
                combat_doses += dose_count;
                break;
            case OSRS_CONSUMABLE_RANGING:
            case OSRS_CONSUMABLE_DIVINE_RANGING:
                ranged_doses += dose_count;
                break;
            case OSRS_CONSUMABLE_SURGE:
                surge_doses += dose_count;
                break;
            case OSRS_CONSUMABLE_ANTIVENOM_PLUS:
                antivenom_doses += dose_count;
                break;
            case OSRS_CONSUMABLE_GUTHIX_REST:
            case OSRS_CONSUMABLE_SATURATED_HEART:
            case OSRS_CONSUMABLE_NONE:
            case OSRS_CONSUMABLE_SHARK_FOOD:
            case OSRS_CONSUMABLE_KARAMBWAN:
                break;
            case OSRS_CONSUMABLE_BASTION:
            case OSRS_CONSUMABLE_STAMINA:
            case OSRS_CONSUMABLE_COUNT:
                fprintf(stderr, "colosseum: consumable kind %d not in colosseum kits\n",
                    (int)resolution.consumable_kind);
                abort();
        }
    }
    s->player.brew_doses = brew_doses;
    s->player.restore_doses = restore_doses;
    s->player.combat_potion_doses = combat_doses;
    s->player.ranged_potion_doses = ranged_doses;
    s->player.antivenom_doses = antivenom_doses;
    s->surge_doses = surge_doses;
}

static void col_reset_ctx(EncounterState* state, EncounterContext* context, uint32_t seed) {
    col_build_npc_stats();
    ColosseumState* s = (ColosseumState*)state;
    ColosseumContext* ctx = col_require_context(context);
    encounter_arena_topology_require_finalized(ctx->route_topology);
    osrs_actor_route_cache_clear(&ctx->player_route_cache);
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
    s->last_damage_source = -1;
    s->last_doom_damage_source = -1;
    s->sol.boss_idx = -1;
    s->min_sol_hp_seen = COLO_SOL_HP_MAX;

    s->active_player_damage_received_scale = ctx->config.player_damage_received_scale;
    for (int i = 0; i < COLO_OBS_NPCS; i++) s->current_obs_slots[i] = -1;

    s->player.entity_type = ENTITY_PLAYER;
    encounter_init_maxed_player_combat_stats(&s->player, COLO_PLAYER_PRAYER_LEVEL);
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
    col_seed_inventory_cells_from_gameplay_loadout(s);

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

    col_resummon_thrall(s);

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
