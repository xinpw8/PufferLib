#line 76 "encounter_colosseum.h"

static int col_player_walkable(void* ctx, int x, int y) {
    ColosseumState* s = (ColosseumState*)ctx;
    if (col_static_blocked(x, y)) return 0;
    if (col_sol_clamp_active(s) && !col_in_boss_arena(s, x, y)) return 0;
    return 1;
}

static int col_los_tile_blocked(void* ctx, int x, int y) {
    (void)ctx;
    return col_static_blocked(x, y);
}

static int col_tiles_have_los(const ColosseumState* s, int x0, int y0, int x1, int y1) {
    (void)s;
    if (!COLO_STATIC_LOOKUP_TABLES_BUILT) col_build_static_arena();
    if (col_static_los_endpoint_in_bounds(x0, y0) &&
            col_static_los_endpoint_in_bounds(x1, y1))
        return col_static_los_get(x0, y0, x1, y1);
    return col_tiles_have_los_reference_slowpath(x0, y0, x1, y1);
}

static OsrsLosQuery col_player_los_query(ColosseumState* s) {
    return osrs_los_tile(col_los_tile_blocked, s);
}

static int col_npc_has_los_to_player(const ColosseumState* s, const ColoNPC* npc) {
    int size = npc->size > 0 ? npc->size : COLO_NPC_STATS[npc->type].size;
    int sx = s->player.x, sy = s->player.y;
    if (sx < npc->x) sx = npc->x;
    if (sx > npc->x + size - 1) sx = npc->x + size - 1;
    if (sy < npc->y) sy = npc->y;
    if (sy > npc->y + size - 1) sy = npc->y + size - 1;
    return col_tiles_have_los(s, sx, sy, s->player.x, s->player.y);
}

static int col_npc_blocked_ignore_player(void* ctx, int x, int y, int size) {
    ColosseumState* s = (ColosseumState*)ctx;
    if (col_static_footprint_blocked_lookup(x, y, size)) return 1;
    int sol_clamp_active = col_sol_clamp_active(s);
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            if (sol_clamp_active && !col_in_boss_arena(s, x + dx, y + dy)) return 1;
            int gx, gy;
            if (!col_grid_index(x + dx, y + dy, &gx, &gy)) return 1;
            if (s->npc_collision_flags[gx][gy]) return 1;
        }
    }
    return 0;
}

static int col_npc_blocked(void* ctx, int x, int y, int size) {
    ColosseumState* s = (ColosseumState*)ctx;
    if (col_npc_blocked_ignore_player(ctx, x, y, size)) return 1;
    for (int dx = 0; dx < size; dx++) {
        for (int dy = 0; dy < size; dy++) {
            int gx, gy;
            if (!col_grid_index(x + dx, y + dy, &gx, &gy)) return 1;
            if (s->player_collision_flags[gx][gy]) return 1;
        }
    }
    return 0;
}

typedef struct {
    ColosseumState* s;
    int self_idx;
} ColoNpcMoveCtx;

static int col_npc_move_blocked_ignore_player(void* ctx, int x, int y, int size) {
    ColoNpcMoveCtx* move_ctx = (ColoNpcMoveCtx*)ctx;
    return col_npc_blocked_ignore_player(move_ctx->s, x, y, size);
}

static int col_npc_move_overlap_hold(void* ctx) {
    ColoNpcMoveCtx* move_ctx = (ColoNpcMoveCtx*)ctx;
    ColosseumState* s = move_ctx->s;
    return s->player_last_interaction_age == 0 &&
           s->player_last_interaction_target_slot == move_ctx->self_idx;
}

typedef struct {
    ColosseumState* s;
    int size;
    uint16_t gen;

    int ignore_npcs;
} ColoNpcPathCtx;

typedef struct {
    uint16_t blocked_gen[COLO_ARENA_WIDTH][COLO_ARENA_HEIGHT];
    uint8_t blocked_value[COLO_ARENA_WIDTH][COLO_ARENA_HEIGHT];
    uint16_t counter;
} ColoNpcPathCache;

static OSRS_THREAD_LOCAL ColoNpcPathCache col_npc_path_cache;

static ColoNpcPathCtx col_npc_path_ctx_begin(ColosseumState* s, int size) {
    col_npc_path_cache.counter++;
    if (col_npc_path_cache.counter == 0) {
        memset(col_npc_path_cache.blocked_gen, 0,
            sizeof(col_npc_path_cache.blocked_gen));
        col_npc_path_cache.counter = 1;
    }
    return (ColoNpcPathCtx){ .s = s, .size = size, .gen = col_npc_path_cache.counter };
}

static int col_npc_path_blocked_uncached(ColoNpcPathCtx* pc, int x, int y) {
    ColosseumState* s = pc->s;
    if (s->player.x >= x && s->player.x < x + pc->size &&
        s->player.y >= y && s->player.y < y + pc->size) return 1;
    if (pc->ignore_npcs) {
        for (int dx = 0; dx < pc->size; dx++) {
            for (int dy = 0; dy < pc->size; dy++) {
                int tx = x + dx, ty = y + dy;
                if (col_static_blocked(tx, ty)) return 1;
                if (col_sol_clamp_active(s) && !col_in_boss_arena(s, tx, ty)) return 1;
            }
        }
        return 0;
    }
    return col_npc_blocked_ignore_player(s, x, y, pc->size);
}

static int col_npc_path_blocked(void* ctx, int x, int y) {
    ColoNpcPathCtx* pc = (ColoNpcPathCtx*)ctx;
    int gx, gy;
    if (!col_grid_index(x, y, &gx, &gy))
        return col_npc_path_blocked_uncached(pc, x, y);
    if (col_npc_path_cache.blocked_gen[gx][gy] != pc->gen) {
        col_npc_path_cache.blocked_gen[gx][gy] = pc->gen;
        col_npc_path_cache.blocked_value[gx][gy] =
            (uint8_t)col_npc_path_blocked_uncached(pc, x, y);
    }
    return col_npc_path_cache.blocked_value[gx][gy];
}

#define COL_WARBAND_BFS_MEMO_BITS 16
#define COL_WARBAND_BFS_MEMO_SIZE (1 << COL_WARBAND_BFS_MEMO_BITS)
static OSRS_THREAD_LOCAL uint64_t col_warband_bfs_memo_key[COL_WARBAND_BFS_MEMO_SIZE];
static OSRS_THREAD_LOCAL uint8_t col_warband_bfs_memo_result[COL_WARBAND_BFS_MEMO_SIZE];

#ifdef COLO_BFS_INSTR
static long col_warband_bfs_memo_instr_hits;
static long col_warband_bfs_memo_instr_misses;
static int col_warband_bfs_memo_instr_registered;
static void col_warband_bfs_memo_instr_report(void) {
    long total = col_warband_bfs_memo_instr_hits + col_warband_bfs_memo_instr_misses;
    fprintf(stderr, "COLO_BFS_INSTR: memo hits=%ld misses=%ld (%.1f%% hit)\n",
        col_warband_bfs_memo_instr_hits, col_warband_bfs_memo_instr_misses,
        total ? 100.0 * col_warband_bfs_memo_instr_hits / total : 0.0);
}
#endif

static uint64_t col_warband_bfs_memo_pack(
    const ColosseumState* s, int x, int y, int tx, int ty) {
    uint64_t key = 1;
    key = key << 6 | (uint64_t)x;
    key = key << 6 | (uint64_t)y;
    key = key << 6 | (uint64_t)tx;
    key = key << 6 | (uint64_t)ty;
    key = key << 6 | (uint64_t)s->player.x;
    key = key << 6 | (uint64_t)s->player.y;
    if (col_sol_clamp_active(s)) {
        key = key << 1 | 1;
        key = key << 6 | (uint64_t)s->sol.boss_arena_min_x;
        key = key << 6 | (uint64_t)s->sol.boss_arena_max_x;
        key = key << 6 | (uint64_t)s->sol.boss_arena_min_y;
        key = key << 6 | (uint64_t)s->sol.boss_arena_max_y;
    } else {
        key <<= 1;
    }
    return key;
}

static uint32_t col_warband_bfs_memo_slot(uint64_t key) {
    return (uint32_t)((key * 0x9E3779B97F4A7C15ULL) >> (64 - COL_WARBAND_BFS_MEMO_BITS));
}

static uint8_t col_warband_bfs_memo_encode(PathResult pr) {
    return (uint8_t)((pr.found ? 1 : 0) |
        (uint32_t)(pr.next_dx + 1) << 1 |
        (uint32_t)(pr.next_dy + 1) << 3);
}

static int col_npc_path_step_with_ctx(ColoNpcPathCtx* pc, int* x, int* y, int tx, int ty) {
    if (tx < COLO_ARENA_MIN_X) tx = COLO_ARENA_MIN_X;
    if (tx > COLO_ARENA_MAX_X) tx = COLO_ARENA_MAX_X;
    if (ty < COLO_ARENA_MIN_Y) ty = COLO_ARENA_MIN_Y;
    if (ty > COLO_ARENA_MAX_Y) ty = COLO_ARENA_MAX_Y;
    if (*x == tx && *y == ty) return 0;
    int direct_dx = (tx > *x) - (tx < *x);
    int direct_dy = (ty > *y) - (ty < *y);
    int direct_abs_dx = tx > *x ? tx - *x : *x - tx;
    int direct_abs_dy = ty > *y ? ty - *y : *y - ty;
    if (direct_abs_dx <= 1 && direct_abs_dy <= 1 &&
            !col_npc_path_blocked(pc, tx, ty) &&
            (direct_dx == 0 || direct_dy == 0 ||
             (!col_npc_path_blocked(pc, *x + direct_dx, *y) &&
              !col_npc_path_blocked(pc, *x, *y + direct_dy)))) {
        *x = tx;
        *y = ty;
        return 1;
    }
    int memoable = pc->ignore_npcs && pc->size == 1 &&
        *x >= COLO_ARENA_MIN_X && *x <= COLO_ARENA_MAX_X &&
        *y >= COLO_ARENA_MIN_Y && *y <= COLO_ARENA_MAX_Y &&
        pc->s->player.x >= COLO_ARENA_MIN_X && pc->s->player.x <= COLO_ARENA_MAX_X &&
        pc->s->player.y >= COLO_ARENA_MIN_Y && pc->s->player.y <= COLO_ARENA_MAX_Y;
    uint64_t memo_key = 0;
    uint32_t memo_slot = 0;
    if (memoable) {
#ifdef COLO_BFS_INSTR
        if (!col_warband_bfs_memo_instr_registered) {
            col_warband_bfs_memo_instr_registered = 1;
            atexit(col_warband_bfs_memo_instr_report);
        }
#endif
        memo_key = col_warband_bfs_memo_pack(pc->s, *x, *y, tx, ty);
        memo_slot = col_warband_bfs_memo_slot(memo_key);
        if (col_warband_bfs_memo_key[memo_slot] == memo_key) {
#ifdef COLO_BFS_INSTR
            col_warband_bfs_memo_instr_hits++;
#endif
            uint8_t r = col_warband_bfs_memo_result[memo_slot];
            int found = r & 1;
            int next_dx = (int)((r >> 1) & 3) - 1;
            int next_dy = (int)((r >> 3) & 3) - 1;
            if (!found || (next_dx == 0 && next_dy == 0)) return 0;
            *x += next_dx;
            *y += next_dy;
            return 1;
        }
#ifdef COLO_BFS_INSTR
        col_warband_bfs_memo_instr_misses++;
#endif
    }
#ifdef COLO_PROFILE_ENABLED
    int col_prof_enabled = COLO_PROFILE_ENABLED();
    double col_prof_t0 = col_prof_enabled ? COLO_PROFILE_NOW_MS() : 0.0;
#endif
    PathResult pr = pathfind_step_arena(NULL, 0, *x, *y, tx, ty,
        col_npc_path_blocked, pc,
        COLO_ARENA_MIN_X, COLO_ARENA_MIN_Y, COLO_ARENA_WIDTH, COLO_ARENA_HEIGHT);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_STEP_NPC_PATHFINDING);
#endif
    if (memoable) {
        col_warband_bfs_memo_key[memo_slot] = memo_key;
        col_warband_bfs_memo_result[memo_slot] = col_warband_bfs_memo_encode(pr);
    }
    if (!pr.found || (pr.next_dx == 0 && pr.next_dy == 0)) return 0;
    *x += pr.next_dx;
    *y += pr.next_dy;
    return 1;
}

static int col_npc_path_step(ColosseumState* s, int* x, int* y, int size, int tx, int ty) {
    ColoNpcPathCtx pc = col_npc_path_ctx_begin(s, size);
    return col_npc_path_step_with_ctx(&pc, x, y, tx, ty);
}

static void col_warband_pick_target(
    ColosseumState* s, ColoNpcPathCtx* pc, ColoNPC* npc, int* tx, int* ty) {
    int px = s->player.x, py = s->player.y;
    int pref = colo_npc_warband(npc)->formation_dir;
    int fx = px + COLO_WARBAND_FORM_OFFSET[pref][0];
    int fy = py + COLO_WARBAND_FORM_OFFSET[pref][1];
    *tx = fx;
    *ty = fy;
    if (!col_npc_path_blocked(pc, fx, fy)) return;

    static const int ADJ[8][2] = {
        {0, 1}, {1, 0}, {-1, 0}, {0, -1},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1} };
    int best_key = 1 << 30;
    for (int a = 0; a < 8; a++) {
        int x = px + ADJ[a][0], y = py + ADJ[a][1];
        if (col_npc_path_blocked(pc, x, y)) continue;
        int dx = npc->x > x ? npc->x - x : x - npc->x;
        int dy = npc->y > y ? npc->y - y : y - npc->y;
        int cheb = dx > dy ? dx : dy;
        int key = (a >= 4 ? 4096 : 0) + cheb * 16 + a;
        if (key < best_key) { best_key = key; *tx = x; *ty = y; }
    }
}

static void col_warband_move(ColosseumState* s, int idx) {
    ColoNPC* npc = &s->npcs[idx];
    npc->target_x = s->player.x;
    npc->target_y = s->player.y;

    ColoNpcPathCtx pc = col_npc_path_ctx_begin(s, 1);
    pc.ignore_npcs = 1;
    int tx, ty;
    col_warband_pick_target(s, &pc, npc, &tx, &ty);
    int steps = 0;
    for (int k = 0; k < COLO_WARBAND_TILES_PER_TICK; k++) {
        if (npc->x == tx && npc->y == ty) break;
        if (!col_npc_path_step_with_ctx(&pc, &npc->x, &npc->y, tx, ty)) break;
        steps++;
    }
    npc->moved_this_tick = steps > 0 ? 1 : 0;
}

static void col_npc_move_ctx(ColosseumState* s, ColosseumContext* ctx, int idx) {
    (void)ctx;
    ColoNPC* npc = &s->npcs[idx];
    const ColoNpcStats* st = &COLO_NPC_STATS[npc->type];
    if (npc->type == COLO_SOL_HEREDIT) return;
    if (!st->can_move) return;
    if (npc->stun_timer > 0 || npc->frozen_ticks > 0) return;

    if (col_type_is_warbander(npc->type)) {
        col_warband_move(s, idx);
        return;
    }

    int size = col_npc_effective_size(npc);
    int tx = s->player.x, ty = s->player.y;
    npc->target_x = tx;
    npc->target_y = ty;

    col_stamp_npc_collision_footprint(s, npc->x, npc->y, size, 0);
    ColoNpcMoveCtx move_ctx = { s, idx };

    int is_ranged = st->attack_range > 1;
    if (is_ranged) {
        int dist = encounter_dist_to_npc(s->player.x, s->player.y, npc->x, npc->y, size);
        if (dist >= 1 && dist <= st->attack_range && col_npc_has_los_to_player(s, npc)) {
            col_stamp_npc_collision_footprint(s, npc->x, npc->y, size, 1);
            npc->moved_this_tick = 0;
            return;
        }
    }
    if (npc->attack_timer > st->attack_speed) {
        col_stamp_npc_collision_footprint(s, npc->x, npc->y, size, 1);
        npc->moved_this_tick = 0;
        return;
    }

    int moved;
    if (npc->type == COLO_MINOTAUR && col_mod_active(s, COLO_MOD_RED_FLAG)) {

        moved = 0;
        if (encounter_dist_to_npc(tx, ty, npc->x, npc->y, size) > 1)
            moved = col_npc_path_step(s, &npc->x, &npc->y, size, tx - 1, ty - 1);
    } else {
        EncounterNpcStepPolicy step_policy = is_ranged
            ? ENCOUNTER_NPC_STEP_OSRS_AGGRO_TARGET
            : ENCOUNTER_NPC_STEP_OSRS_AGGRO_STOP_AT_MELEE;
        moved = encounter_npc_step_toward_policy(
            &npc->x, &npc->y, tx, ty, size, 1, step_policy,
            col_npc_move_blocked_ignore_player, &move_ctx,
            col_npc_move_overlap_hold, &s->rng_state);
    }
    col_stamp_npc_collision_footprint(s, npc->x, npc->y, size, 1);
    npc->moved_this_tick = moved ? 1 : 0;
}
