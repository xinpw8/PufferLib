#line 76 "encounter_colosseum.h"

static int col_sol_clamp_active(const ColosseumState* s) {
    return s->wave == COLO_WAVE_BOSS && s->sol.started;
}

static int col_in_boss_arena(const ColosseumState* s, int x, int y) {
    return x > s->sol.boss_arena_min_x && x < s->sol.boss_arena_max_x &&
           y > s->sol.boss_arena_min_y && y < s->sol.boss_arena_max_y;
}

static int col_sol_phase_for_hp(int hp) {
    int max = COLO_SOL_HP_MAX;
    if (hp > (max * 90) / 100) return 0;
    if (hp > (max * 75) / 100) return 1;
    if (hp > (max * 50) / 100) return 2;
    if (hp > (max * 25) / 100) return 3;
    if (hp > (max * 10) / 100) return 4;
    return 5;
}

static int col_sol_find_idx(const ColosseumState* s) {
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        const ColoNPC* npc = &s->npcs[i];
        if (col_npc_is_live_target(npc) && npc->type == COLO_SOL_HEREDIT)
            return i;
    }
    return -1;
}

static int col_player_overhead_protect_on(const ColosseumState* s) {
    return s->player.prayer == PRAYER_PROTECT_MELEE ||
           s->player.prayer == PRAYER_PROTECT_RANGED ||
           s->player.prayer == PRAYER_PROTECT_MAGIC;
}

static void col_sol_begin_boss_arena(ColosseumState* s) {
    s->sol = (SolHereditState){0};
    s->sol.started = 1;
    s->sol.boss_idx = -1;
    s->sol.force_spear = 1;
    s->sol.laser_cooldown = COLO_SOL_CRYSTAL_COOLDOWN_MIN;
    s->sol.laser_volley_active = 0;
    s->sol.laser_show_seen = 0;
    s->sol.laser_aligned_show = 0;
    s->sol.laser_n_aligned_show = 0;
    s->sol.laser_n_active_at_fire = 0;
    s->sol.boss_arena_min_x = COLO_BOSS_ARENA_MIN_X;
    s->sol.boss_arena_min_y = COLO_BOSS_ARENA_MIN_Y;
    s->sol.boss_arena_max_x = COLO_BOSS_ARENA_MAX_X;
    s->sol.boss_arena_max_y = COLO_BOSS_ARENA_MAX_Y;

    s->player.x = COLO_BOSS_PLAYER_START_X;
    s->player.y = COLO_BOSS_PLAYER_START_Y;
    s->player_dest_x = -1;
    s->player_dest_y = -1;
    osrs_interaction_clear(&s->interaction);
    col_rebuild_player_collision_flags(s);
}

static void col_sol_cast_aoe(ColosseumState* s, int idx, int kind, int variant);
static void col_sol_start_triple_parry(ColosseumState* s, int idx);
static void col_sol_start_grapple(ColosseumState* s);
static void col_sol_fire_lasers(ColosseumState* s);

static int col_sol_attack_delay(int kind, int phase) {
    switch (kind) {
        case COLO_SOL_ATTACK_SPEAR:
            return phase < 2 ? COLO_SOL_SPEAR_DELAY : COLO_SOL_SPEAR_DELAY_FAST;
        case COLO_SOL_ATTACK_SHIELD:
            return phase < 2 ? COLO_SOL_SHIELD_DELAY : COLO_SOL_SHIELD_DELAY_FAST;
        case COLO_SOL_ATTACK_TRIPLE:
            return phase == 2 ? COLO_SOL_TRIPLE_DELAY_MID : COLO_SOL_TRIPLE_DELAY;
        case COLO_SOL_ATTACK_GRAPPLE:
            return COLO_SOL_GRAPPLE_DELAY;
        default:
            assert(0 && "next delay queried for a non-attack");
            return COLO_SOL_SPEAR_DELAY;
    }
}

static int col_sol_attack_freeze(int kind, int phase) {
    switch (kind) {
        case COLO_SOL_ATTACK_SPEAR:   return COLO_SOL_SPEAR_FREEZE;
        case COLO_SOL_ATTACK_SHIELD:  return COLO_SOL_SHIELD_FREEZE;
        case COLO_SOL_ATTACK_GRAPPLE: return COLO_SOL_GRAPPLE_FREEZE;
        case COLO_SOL_ATTACK_TRIPLE:
            return 2 * COLO_SOL_TP_HIT_GAP +
                   (phase >= 3 ? COLO_SOL_TP_HIT3_GAP_LOW : COLO_SOL_TP_HIT_GAP);
        default:
            assert(0 && "freeze queried for a non-attack");
            return COLO_SOL_SPEAR_FREEZE;
    }
}

static int col_sol_select_attack(ColosseumState* s) {
    SolHereditState* sol = &s->sol;
    if (sol->force_spear) {
        sol->force_spear = 0;
        if (sol->special_cooldown > 0) sol->special_cooldown--;
        return COLO_SOL_ATTACK_SPEAR;
    }

    int pool[6];
    int n = 0;
    pool[n++] = COLO_SOL_ATTACK_SPEAR;
    pool[n++] = COLO_SOL_ATTACK_SPEAR;
    pool[n++] = COLO_SOL_ATTACK_SHIELD;
    pool[n++] = COLO_SOL_ATTACK_SHIELD;
    if (sol->special_cooldown <= 0) {
        if (sol->phase >= 1) pool[n++] = COLO_SOL_ATTACK_TRIPLE;
        if (sol->phase >= 2) pool[n++] = COLO_SOL_ATTACK_GRAPPLE;
    }

    int kind = pool[encounter_rand_int(&s->rng_state, n)];
    if (kind == COLO_SOL_ATTACK_TRIPLE || kind == COLO_SOL_ATTACK_GRAPPLE)
        sol->special_cooldown = COLO_SOL_SPECIAL_COOLDOWN;
    else if (sol->special_cooldown > 0)
        sol->special_cooldown--;
    return kind;
}

static int col_sol_pick_variant(SolHereditState* sol, int kind) {
    int variant = (sol->last_attack_kind == kind && sol->last_variant == 1) ? 2 : 1;
    sol->last_attack_kind = kind;
    sol->last_variant = variant;
    return variant;
}

static int col_sol_adjacent_to_player(const ColosseumState* s, int idx) {
    const ColoNPC* boss = &s->npcs[idx];
    return encounter_dist_to_npc(s->player.x, s->player.y, boss->x, boss->y,
        col_npc_effective_size(boss)) == 1;
}

static void col_sol_tick_attack_engine(ColosseumState* s, int idx) {
    SolHereditState* sol = &s->sol;
    if (sol->immobile_ticks > 0) sol->immobile_ticks--;
    if (sol->attack_delay > 0) sol->attack_delay--;
    if (sol->immobile_ticks > 0 || sol->attack_delay > 0) return;
    if (!col_sol_adjacent_to_player(s, idx)) return;

    int kind = col_sol_select_attack(s);
    int variant = col_sol_pick_variant(sol, kind);
    switch (kind) {
        case COLO_SOL_ATTACK_SPEAR:
        case COLO_SOL_ATTACK_SHIELD:
            col_sol_cast_aoe(s, idx, kind, variant);
            break;
        case COLO_SOL_ATTACK_TRIPLE:
            col_sol_start_triple_parry(s, idx);
            break;
        case COLO_SOL_ATTACK_GRAPPLE:
            col_sol_start_grapple(s);
            break;
        default:
            assert(0 && "selected a non-attack");
            break;
    }
    sol->immobile_ticks = col_sol_attack_freeze(kind, sol->phase);
    sol->attack_delay = col_sol_attack_delay(kind, sol->phase);
    s->npcs[idx].attacked_this_tick = 1;
    s->npcs[idx].attack_style_this_tick = ATTACK_STYLE_MELEE;

    if (sol->crystal_count > 0 && sol->laser_cooldown < 0)
        col_sol_fire_lasers(s);
}

static void col_sol_move(ColosseumState* s, ColosseumContext* ctx, int idx) {
    SolHereditState* sol = &s->sol;
    ColoNPC* boss = &s->npcs[idx];
    boss->target_x = s->player.x;
    boss->target_y = s->player.y;
    if (sol->immobile_ticks > 0) return;
    int size = col_npc_effective_size(boss);
    if (encounter_dist_to_npc(s->player.x, s->player.y, boss->x, boss->y, size) <= 1)
        return;
    col_stamp_npc_collision_footprint(s, boss->x, boss->y, size, 0);
    int moved = col_npc_path_step(
        s, ctx, &boss->x, &boss->y, size,
        s->player.x - size / 2, s->player.y - size / 2);
    col_stamp_npc_collision_footprint(s, boss->x, boss->y, size, 1);
    boss->moved_this_tick = moved ? 1 : 0;
}

static void col_sol_aoe_direction(int px, int py, int anchor_x, int anchor_y,
                                  int* dx, int* dy) {
    int closest_x = px, closest_y = py;
    if (closest_x < anchor_x) closest_x = anchor_x;
    if (closest_x > anchor_x + 4) closest_x = anchor_x + 4;
    if (closest_y < anchor_y) closest_y = anchor_y;
    if (closest_y > anchor_y + 4) closest_y = anchor_y + 4;
    int ox = px - closest_x;
    int oy = py - closest_y;
    *dx = ox > 0 ? 1 : ox < 0 ? -1 : 0;
    *dy = oy > 0 ? 1 : oy < 0 ? -1 : 0;
    if (*dx == 0 && *dy == 0) { *dx = -1; *dy = -1; }
}

static void col_sol_aoe_effective_dir(const SolHereditState* sol, int x, int y,
                                      int* dx, int* dy) {
    if (sol->aoe_dir_x != 0 || sol->aoe_dir_y != 0) {
        *dx = sol->aoe_dir_x;
        *dy = sol->aoe_dir_y;
        return;
    }
    col_sol_aoe_direction(x, y, sol->aoe_x, sol->aoe_y, dx, dy);
}

static int col_sol_aoe_tile_is_hazard(const SolHereditState* sol, int x, int y) {
    int cx = sol->aoe_x + 2, cy = sol->aoe_y + 2;

    if (sol->aoe_attack == COLO_SOL_AOE_SHIELD1 ||
        sol->aoe_attack == COLO_SOL_AOE_SHIELD2) {
        int dx = abs(x - cx), dy = abs(y - cy);
        int cheb = dx > dy ? dx : dy;
        int ring = sol->aoe_attack == COLO_SOL_AOE_SHIELD1
            ? COLO_SOL_SHIELD1_RING : COLO_SOL_SHIELD2_RING;
        return cheb != ring;
    }
    if (sol->aoe_attack != COLO_SOL_AOE_SPEAR1 &&
        sol->aoe_attack != COLO_SOL_AOE_SPEAR2) return 0;

    int dx, dy;
    col_sol_aoe_effective_dir(sol, x, y, &dx, &dy);
    if (dx != 0 && dy != 0) {
        int u = (x - (cx + 2 * dx)) * dx;
        int v = (y - (cy + 2 * dy)) * dy;
        if (sol->aoe_attack == COLO_SOL_AOE_SPEAR1) {
            if (abs(x - cx) <= 2 && abs(y - cy) <= 2) return 1;
            if (u == v - 1 && v >= 1 && v <= 1 + COLO_SOL_SPEAR_LINE_LEN) return 1;
            if (v == u - 1 && u >= 1 && u <= 1 + COLO_SOL_SPEAR_LINE_LEN) return 1;
            return 0;
        }
        if (abs(x - cx) <= 3 && abs(y - cy) <= 3) return 1;
        if (u == v && u >= 2 && u <= 2 + COLO_SOL_SPEAR_LINE_LEN) return 1;
        if (v == u + 3 && u >= -1 && u <= -1 + COLO_SOL_SPEAR_LINE_LEN) return 1;
        if (u == v + 3 && v >= -1 && v <= -1 + COLO_SOL_SPEAR_LINE_LEN) return 1;
        return 0;
    }

    int forward = dx != 0 ? (x - cx) * dx : (y - cy) * dy;
    int lateral = dx != 0 ? (y - cy) : (x - cx);
    if (sol->aoe_attack == COLO_SOL_AOE_SPEAR1) {
        if (lateral >= -2 && lateral <= 2 &&
            forward >= -2 && forward <= 3) return 1;
        if (forward >= 4 && forward <= 4 + COLO_SOL_SPEAR_LINE_LEN)
            return lateral == -1 || lateral == 1;
        return 0;
    }
    if (lateral >= -3 && lateral <= 3 &&
        forward >= -3 && forward <= 3) return 1;
    if (forward >= 4 && forward <= 4 + COLO_SOL_SPEAR_LINE_LEN)
        return lateral == -2 || lateral == 0 || lateral == 2;
    return 0;
}

static ColoSolDamageSource col_sol_aoe_damage_source(ColoSolAoeAttack attack) {
    switch (attack) {
        case COLO_SOL_AOE_SPEAR1:
            return COLO_SOL_DAMAGE_SPEAR_1;
        case COLO_SOL_AOE_SPEAR2:
            return COLO_SOL_DAMAGE_SPEAR_2;
        case COLO_SOL_AOE_SHIELD1:
            return COLO_SOL_DAMAGE_SHIELD_1;
        case COLO_SOL_AOE_SHIELD2:
            return COLO_SOL_DAMAGE_SHIELD_2;
        case COLO_SOL_AOE_NONE:
            break;
    }
    fprintf(stderr, "invalid Sol AOE damage source %d\n", (int)attack);
    abort();
}

static void col_sol_cast_aoe(ColosseumState* s, int idx, int kind, int variant) {
    SolHereditState* sol = &s->sol;
    const ColoNPC* boss = &s->npcs[idx];
    if (kind == COLO_SOL_ATTACK_SPEAR)
        sol->aoe_attack = variant == 1 ? COLO_SOL_AOE_SPEAR1 : COLO_SOL_AOE_SPEAR2;
    else
        sol->aoe_attack = variant == 1 ? COLO_SOL_AOE_SHIELD1 : COLO_SOL_AOE_SHIELD2;
    sol->aoe_age = 0;
    sol->aoe_x = boss->x;
    sol->aoe_y = boss->y;
    sol->aoe_dir_x = 0;
    sol->aoe_dir_y = 0;
}

static void col_sol_tick_aoe(ColosseumState* s) {
    SolHereditState* sol = &s->sol;
    if (sol->aoe_attack == COLO_SOL_AOE_NONE) return;
    sol->aoe_age++;
    if (sol->aoe_age == COLO_SOL_AOE_DAMAGE_AGE &&
        (sol->aoe_attack == COLO_SOL_AOE_SPEAR1 ||
         sol->aoe_attack == COLO_SOL_AOE_SPEAR2))
        col_sol_aoe_direction(s->player.x, s->player.y, sol->aoe_x, sol->aoe_y,
                              &sol->aoe_dir_x, &sol->aoe_dir_y);
    if (sol->aoe_age == COLO_SOL_AOE_DAMAGE_AGE &&
        col_sol_aoe_tile_is_hazard(sol, s->player.x, s->player.y)) {
        int dmg = COLO_SOL_SLAM_MIN_HIT
            + encounter_rand_int(&s->rng_state, COLO_SOL_SLAM_RAND);
        ColoSolDamageSource source =
            col_sol_aoe_damage_source(sol->aoe_attack);
        int applied = col_damage_player_from_sol(s, dmg, source);
        if (applied > 0) {
            s->last_hit_by_type = COLO_SOL_HEREDIT;
            col_mod_on_player_damaged(
                s, applied, col_damage_source_from_sol(source));
        }
    }
    if (sol->aoe_age >= COLO_SOL_AOE_LIFETIME) sol->aoe_attack = COLO_SOL_AOE_NONE;
}

static void col_sol_crystal_patrol(
    const SolHereditState* sol, int edge, int* fixed, int* lo, int* hi
) {
    switch (edge) {
        case COLO_SOL_EDGE_NORTH:
            *fixed = sol->boss_arena_max_y - 1;
            *lo = sol->boss_arena_min_x + 2;
            *hi = sol->boss_arena_max_x - 2;
            return;
        case COLO_SOL_EDGE_SOUTH:
            *fixed = sol->boss_arena_min_y + 1;
            *lo = sol->boss_arena_min_x + 2;
            *hi = sol->boss_arena_max_x - 2;
            return;
        case COLO_SOL_EDGE_EAST:
            *fixed = sol->boss_arena_max_x - 1;
            *lo = sol->boss_arena_min_y + 2;
            *hi = sol->boss_arena_max_y - 2;
            return;
        case COLO_SOL_EDGE_WEST:
            *fixed = sol->boss_arena_min_x + 1;
            *lo = sol->boss_arena_min_y + 2;
            *hi = sol->boss_arena_max_y - 2;
            return;
        default:
            fprintf(stderr, "invalid crystal edge %d\n", edge);
            abort();
    }
}

static int col_sol_crystal_hits_tile(const ColoSolCrystal* c, int x, int y) {
    if (c->edge == COLO_SOL_EDGE_NORTH || c->edge == COLO_SOL_EDGE_SOUTH)
        return x == c->x;
    return y == c->y;
}

static int col_sol_crystal_in_line(const ColosseumState* s, const ColoSolCrystal* c) {
    return col_sol_crystal_hits_tile(c, s->player.x, s->player.y);
}

static void col_sol_spawn_crystal(ColosseumState* s) {
    SolHereditState* sol = &s->sol;
    if (sol->crystal_count >= COLO_SOL_MAX_CRYSTALS) return;
    ColoSolCrystal* c = &sol->crystals[sol->crystal_count];
    c->active = 1;
    c->edge = sol->crystal_count;
    int fixed, lo, hi;
    col_sol_crystal_patrol(sol, c->edge, &fixed, &lo, &hi);
    int along = lo + encounter_rand_int(&s->rng_state, hi - lo);
    if (c->edge == COLO_SOL_EDGE_NORTH || c->edge == COLO_SOL_EDGE_SOUTH) {
        c->x = along;
        c->y = fixed;
    } else {
        c->x = fixed;
        c->y = along;
    }
    c->dir = encounter_rand_int(&s->rng_state, 2) ? 1 : -1;
    if (along == lo) c->dir = 1;
    if (along == hi) c->dir = -1;
    c->move_timer = COLO_SOL_CRYSTAL_MOVE_TICKS_SPAWN;
    c->firing_freeze = 0;
    sol->crystal_count++;
}

static void col_sol_fire_lasers(ColosseumState* s) {
    SolHereditState* sol = &s->sol;
    int active = 0;
    int aligned = 0;
    for (int i = 0; i < sol->crystal_count; i++) {
        if (!sol->crystals[i].active) continue;
        sol->crystals[i].firing_freeze = COLO_SOL_LASER_FREEZE;
        active++;
        if (col_sol_crystal_in_line(s, &sol->crystals[i])) aligned++;
    }
    sol->laser_volley_active = 1;
    sol->laser_show_seen = 0;
    sol->laser_aligned_show = 0;
    sol->laser_n_aligned_show = 0;
    sol->laser_n_active_at_fire = active;
    s->log.laser_volleys += 1.0f;
    if (aligned > 0) s->log.laser_aligned_at_fire += 1.0f;
    sol->laser_cooldown = sol->phase >= 5
        ? COLO_SOL_CRYSTAL_COOLDOWN_ENRAGE
        : COLO_SOL_CRYSTAL_COOLDOWN_MIN
            + encounter_rand_int(&s->rng_state, COLO_SOL_CRYSTAL_COOLDOWN_RAND);
}

static void col_sol_tick_crystals(ColosseumState* s) {
    SolHereditState* sol = &s->sol;
    int aligned_show = 0;
    int aligned_pre = 0;
    int aligned_damage = 0;
    int showing = 0;
    int prefire = 0;
    int damaging = 0;
    int firing = 0;
    for (int i = 0; i < sol->crystal_count; i++) {
        ColoSolCrystal* c = &sol->crystals[i];
        if (!c->active) continue;

        if (c->firing_freeze <= 0 && c->move_timer <= 0) {
            int fixed, lo, hi;
            col_sol_crystal_patrol(sol, c->edge, &fixed, &lo, &hi);
            int* along = (c->edge == COLO_SOL_EDGE_NORTH ||
                          c->edge == COLO_SOL_EDGE_SOUTH) ? &c->x : &c->y;
            *along += c->dir;
            if (*along <= lo || *along >= hi) c->dir = -c->dir;
            c->move_timer = COLO_SOL_CRYSTAL_MOVE_TICKS;
        }

        if (c->firing_freeze > 0) firing = 1;
        if (c->firing_freeze == COLO_SOL_LASER_BEAM_SHOW_MAX) {
            showing = 1;
            if (col_sol_crystal_in_line(s, c)) aligned_show++;
        }
        if (c->firing_freeze == COLO_SOL_LASER_DAMAGE_FREEZE + 1) {
            prefire = 1;
            if (col_sol_crystal_in_line(s, c)) aligned_pre++;
        }
        if (c->firing_freeze == COLO_SOL_LASER_DAMAGE_FREEZE) {
            damaging = 1;
            if (col_sol_crystal_in_line(s, c)) {
                aligned_damage++;
                int dmg = COLO_SOL_LASER_MIN_HIT
                    + encounter_rand_int(&s->rng_state, COLO_SOL_LASER_RAND);
                int applied = col_damage_player_from_sol(
                    s, dmg, COLO_SOL_DAMAGE_CRYSTAL_LASER);
                if (applied > 0) {
                    s->last_hit_by_type = COLO_SOL_HEREDIT;
                    col_mod_on_player_damaged(
                        s, applied, COLO_DAMAGE_SOL_CRYSTAL_LASER);
                    s->log.laser_hits += 1.0f;
                    s->log.laser_dmg += (float)applied;
                }
            }
        }

        c->move_timer--;
        c->firing_freeze--;
    }
    if (showing) {
        sol->laser_show_seen = 1;
        sol->laser_n_aligned_show = aligned_show;
        if (aligned_show > 0) {
            sol->laser_aligned_show = 1;
            s->log.laser_aligned_at_show += 1.0f;
        }
    }
    if (prefire && aligned_pre > 0)
        s->log.laser_aligned_at_pre += 1.0f;
    if (damaging) {
        if (aligned_damage > 0) s->log.laser_aligned_at_damage += 1.0f;
        if (sol->laser_aligned_show && aligned_damage == 0)
            s->log.laser_react_ok += 1.0f;
        if (sol->laser_aligned_show && aligned_damage > 0)
            s->log.laser_react_fail += 1.0f;
    }
    if (sol->laser_volley_active && !firing) {
        sol->laser_volley_active = 0;
        sol->laser_show_seen = 0;
        sol->laser_aligned_show = 0;
        sol->laser_n_aligned_show = 0;
        sol->laser_n_active_at_fire = 0;
    }
}

static int col_sol_molten_sand_at(const SolHereditState* sol, int x, int y) {
    for (int i = 0; i < sol->hazard_tile_count; i++)
        if (sol->hazard_tile_x[i] == x && sol->hazard_tile_y[i] == y) return 1;
    return 0;
}

static void col_sol_add_pool(
    ColosseumState* s,
    const ColosseumContext* ctx,
    int x,
    int y
) {
    if (col_topology_tile_blocked(ctx, x, y)) return;
    SolHereditState* sol = &s->sol;
    if (col_sol_molten_sand_at(sol, x, y)) return;
    int slot = sol->hazard_tile_count < COLO_SOL_HAZARD_TILES_MAX
        ? sol->hazard_tile_count++
        : encounter_rand_int(&s->rng_state, COLO_SOL_HAZARD_TILES_MAX);
    sol->hazard_tile_x[slot] = x;
    sol->hazard_tile_y[slot] = y;
}

typedef struct {
    int x, y;
} ColSolMoltenCandidate;

static int col_sol_sample_molten_candidates(
    ColosseumState* s,
    const ColosseumContext* ctx,
    int spread,
    int count,
    int force_player_tile,
    ColSolMoltenCandidate* out
) {
    ColSolMoltenCandidate pool[81];
    int pool_count = 0;
    int player_idx = -1;
    for (int x = s->player.x - spread; x <= s->player.x + spread; x++) {
        for (int y = s->player.y - spread; y <= s->player.y + spread; y++) {
            if (!col_in_boss_arena(s, x, y)) continue;
            if (col_topology_tile_blocked(ctx, x, y)) continue;
            if (pool_count >= (int)(sizeof(pool) / sizeof(pool[0]))) break;
            if (x == s->player.x && y == s->player.y) player_idx = pool_count;
            pool[pool_count++] = (ColSolMoltenCandidate){ .x = x, .y = y };
        }
    }

    int out_count = 0;
    if (force_player_tile && player_idx >= 0 && out_count < count) {
        out[out_count++] = pool[player_idx];
        pool[player_idx] = pool[--pool_count];
    }

    while (out_count < count && pool_count > 0) {
        int idx = encounter_rand_int(&s->rng_state, pool_count);
        out[out_count++] = pool[idx];
        pool[idx] = pool[--pool_count];
    }

    return out_count;
}

static void col_sol_drop_beam_at(ColosseumState* s, int x, int y) {
    SolHereditState* sol = &s->sol;
    for (int b = 0; b < COLO_SOL_BEAM_MAX; b++) {
        if (sol->beams[b].active) continue;
        sol->beams[b] = (ColoSolBeam){
            .active = 1, .x = x, .y = y,
            .ticks_to_pool = COLO_SOL_BEAM_TO_POOL_TICKS };
        return;
    }
}

static void col_sol_drop_beams(
    ColosseumState* s,
    const ColosseumContext* ctx
) {
    ColSolMoltenCandidate tiles[COLO_SOL_BEAM_COUNT];
    int ntiles = col_sol_sample_molten_candidates(
        s, ctx, COLO_SOL_BEAM_SPREAD, COLO_SOL_BEAM_COUNT, 1, tiles);
    for (int n = 0; n < ntiles; n++)
        col_sol_drop_beam_at(s, tiles[n].x, tiles[n].y);
}

static void col_sol_tick_molten(
    ColosseumState* s,
    const ColosseumContext* ctx
) {
    SolHereditState* sol = &s->sol;
    if (sol->phase >= 5) {
        if (sol->enrage_pool_timer > 0) sol->enrage_pool_timer--;
        if (sol->enrage_pool_timer == 0) {
            ColSolMoltenCandidate tile[1];
            int ntiles = col_sol_sample_molten_candidates(
                s, ctx, COLO_SOL_ENRAGE_POOL_SPREAD, 1, 0, tile);
            if (ntiles > 0) col_sol_drop_beam_at(s, tile[0].x, tile[0].y);
            sol->enrage_pool_timer = COLO_SOL_MOLTEN_INTERVAL;
        }
    }

    for (int b = 0; b < COLO_SOL_BEAM_MAX; b++) {
        if (!sol->beams[b].active) continue;
        sol->beams[b].ticks_to_pool--;
        if (sol->beams[b].ticks_to_pool > 0) continue;
        col_sol_add_pool(
            s, ctx, sol->beams[b].x, sol->beams[b].y);
        sol->beams[b].active = 0;
    }

    if (col_sol_molten_sand_at(sol, s->player.x, s->player.y)) {
        int dmg = COLO_MOLTEN_SAND_MIN_HIT
            + encounter_rand_int(&s->rng_state, COLO_MOLTEN_SAND_RAND);
        int applied =
            col_damage_player_from_sol(s, dmg, COLO_SOL_DAMAGE_MOLTEN_SAND);
        if (applied > 0) {
            s->last_hit_by_type = COLO_SOL_HEREDIT;
            col_mod_on_player_damaged(
                s, applied, COLO_DAMAGE_SOL_MOLTEN_SAND);
        }
    }
}

static const int COLO_SOL_PARRY_LOOKBACK[COLO_SOL_PARRY_HITS] = { 3, 2, 2 };

static int col_sol_overhead_on_in_window(const ColosseumState* s, int lookback) {
    for (int k = 1; k <= lookback; k++) {
        int t = s->tick - k;
        if (t < 0) continue;
        if (s->sol.overhead_history[t % COLO_SOL_PRAYER_HISTORY]) return 1;
    }
    return 0;
}

static void col_sol_start_triple_parry(ColosseumState* s, int idx) {
    (void)idx;
    SolHereditState* sol = &s->sol;
    static const int SHORT_DMG[COLO_SOL_PARRY_HITS] = { 15, 25, 35 };
    static const int LONG_DMG[COLO_SOL_PARRY_HITS]  = { 15, 30, 45 };
    int low = sol->phase >= 3;
    const int* dmg = low ? LONG_DMG : SHORT_DMG;
    int gap3 = low ? COLO_SOL_TP_HIT3_GAP_LOW : COLO_SOL_TP_HIT_GAP;

    sol->parry_land_in[0] = COLO_SOL_TP_HIT_GAP;
    sol->parry_land_in[1] = 2 * COLO_SOL_TP_HIT_GAP;
    sol->parry_land_in[2] = 2 * COLO_SOL_TP_HIT_GAP + gap3;
    for (int h = 0; h < COLO_SOL_PARRY_HITS; h++) sol->parry_damage[h] = dmg[h];
    sol->parry_hits_left = COLO_SOL_PARRY_HITS;
}

static void col_sol_tick_triple_parry(ColosseumState* s) {
    SolHereditState* sol = &s->sol;
    if (sol->parry_hits_left <= 0) return;
    for (int h = 0; h < COLO_SOL_PARRY_HITS; h++) {
        if (sol->parry_land_in[h] <= 0) continue;
        sol->parry_land_in[h]--;
        if (sol->parry_land_in[h] > 0) continue;
        sol->parry_hits_left--;

        int early = col_sol_overhead_on_in_window(s, COLO_SOL_PARRY_LOOKBACK[h]);
        int blocked = !early && s->player.prayer == PRAYER_PROTECT_MELEE;
        s->tick_scratch.styles_faced_mask |= (1 << ATTACK_STYLE_MELEE);
        s->tick_scratch.avoidable_threat_by_style[ATTACK_STYLE_MELEE] +=
            (float)sol->parry_damage[h];
        if (blocked) {
            s->tick_scratch.prayer_correct++;
        } else {
            s->tick_scratch.offpray_damage_this_tick += (float)sol->parry_damage[h];
            int applied = col_damage_player_from_sol(
                s, sol->parry_damage[h], COLO_SOL_DAMAGE_TRIPLE_PARRY);
            if (applied > 0) {
                s->last_hit_by_type = COLO_SOL_HEREDIT;
                col_mod_on_player_damaged(
                    s, applied, COLO_DAMAGE_SOL_TRIPLE_PARRY);
            }
        }
        col_log_prayer_event(s, COLO_SOL_HEREDIT, blocked, blocked ? 0 : sol->parry_damage[h]);
        s->player.prayer = PRAYER_NONE;
    }
}

static void col_sol_start_grapple(ColosseumState* s) {
    SolHereditState* sol = &s->sol;
    sol->grapple_active = 1;
    sol->grapple_resolved = 0;
    sol->grapple_body_slot = encounter_rand_int(&s->rng_state, COLO_NUM_GRAPPLE_SLOTS);
    sol->grapple_timer = COLO_SOL_GRAPPLE_WINDOW;
}

static int col_sol_grapple_parry_attempt(ColosseumState* s, int slot) {
    SolHereditState* sol = &s->sol;
    if (!sol->grapple_active || sol->grapple_resolved) return 0;
    if (slot != sol->grapple_body_slot) return 0;

    int perfect = sol->grapple_timer <= 1;
    sol->grapple_resolved = 1;
    sol->grapple_active = 0;
    sol->grapple_timer = 0;
    sol->grapple_outcome = perfect
        ? COLO_GRAPPLE_OUTCOME_PERFECT : COLO_GRAPPLE_OUTCOME_DEFENDED;
    sol->grapple_outcome_tick = s->tick;
    if (perfect) {
        sol->next_attack_guaranteed_max = 1;
        sol->guaranteed_max_ticks = COLO_SOL_PERFECT_MAX_TICKS;
    }
    return 1;
}

static void col_sol_tick_grapple(ColosseumState* s) {
    SolHereditState* sol = &s->sol;
    if (!sol->grapple_active) return;
    if (sol->grapple_timer > 0) sol->grapple_timer--;
    if (sol->grapple_timer == 0 && !sol->grapple_resolved) {
        int dmg = COLO_SOL_SLAM_MIN_HIT
            + encounter_rand_int(&s->rng_state, COLO_SOL_SLAM_RAND);
        int applied =
            col_damage_player_from_sol(s, dmg, COLO_SOL_DAMAGE_GRAPPLE);
        if (applied > 0) {
            s->last_hit_by_type = COLO_SOL_HEREDIT;
            col_mod_on_player_damaged(
                s, applied, COLO_DAMAGE_SOL_GRAPPLE);
        }
        sol->grapple_active = 0;
        sol->grapple_resolved = 1;
        sol->grapple_outcome = COLO_GRAPPLE_OUTCOME_FAILED;
        sol->grapple_outcome_tick = s->tick;
    }
}

static void col_sol_enter_phase(
    ColosseumState* s,
    const ColosseumContext* ctx,
    int new_phase
) {
    SolHereditState* sol = &s->sol;
    sol->phase = new_phase;
    col_sol_drop_beams(s, ctx);
    if (new_phase >= 1 && new_phase <= 4) col_sol_spawn_crystal(s);
    if (new_phase >= 5) sol->laser_cooldown = COLO_SOL_CRYSTAL_COOLDOWN_ENRAGE;
    if (new_phase >= 5) {

        ColSolMoltenCandidate tiles[COLO_SOL_ENRAGE_OPEN_POOLS];
        int ntiles = col_sol_sample_molten_candidates(
            s, ctx, COLO_SOL_ENRAGE_POOL_SPREAD,
            COLO_SOL_ENRAGE_OPEN_POOLS, 0, tiles);
        for (int p = 0; p < ntiles; p++)
            col_sol_drop_beam_at(s, tiles[p].x, tiles[p].y);
        sol->enrage_pool_timer = COLO_SOL_MOLTEN_INTERVAL;
    }
    if (new_phase > s->log.boss_phase_reached) s->log.boss_phase_reached = new_phase;
}

static void col_sol_boss_tick(ColosseumState* s, ColosseumContext* ctx) {
    SolHereditState* sol = &s->sol;
    if (s->wave != COLO_WAVE_BOSS || !sol->started) return;

    int idx = col_sol_find_idx(s);
    sol->boss_idx = idx;
    if (idx < 0) return;

    sol->overhead_history[s->tick % COLO_SOL_PRAYER_HISTORY] =
        (uint8_t)col_player_overhead_protect_on(s);

    int target_phase = col_sol_phase_for_hp(s->npcs[idx].hp);
    if (target_phase > sol->phase) {
        while (sol->phase < target_phase)
            col_sol_enter_phase(s, ctx, sol->phase + 1);
        sol->immobile_ticks = COLO_SOL_TRANSITION_FREEZE;
        sol->force_spear = 1;
    }

    if (sol->guaranteed_max_ticks > 0) {
        sol->guaranteed_max_ticks--;
        if (sol->guaranteed_max_ticks == 0) sol->next_attack_guaranteed_max = 0;
    }
    sol->laser_cooldown--;

    col_sol_tick_aoe(s);
    col_sol_tick_crystals(s);
    col_sol_tick_molten(s, ctx);
    col_sol_tick_triple_parry(s);
    col_sol_tick_grapple(s);

    col_sol_tick_attack_engine(s, idx);
    col_sol_move(s, ctx, idx);
}
