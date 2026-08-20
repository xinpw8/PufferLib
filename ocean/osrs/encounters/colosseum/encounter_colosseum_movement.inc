#line 76 "encounter_colosseum.h"

typedef struct {
    ColosseumState* state;
    const ColosseumContext* context;
} ColoGeometryContext;

static int col_player_walkable_ctx(
    const ColosseumState* state,
    const ColosseumContext* context,
    int x,
    int y
) {
    if (col_topology_tile_blocked(context, x, y)) return 0;
    if (col_sol_clamp_active(state) && !col_in_boss_arena(state, x, y))
        return 0;
    return 1;
}


static int col_topology_los_tile_blocked(void* data, int x, int y) {
    const ColosseumContext* context = (const ColosseumContext*)data;
    return col_topology_tile_blocked(context, x, y);
}

static OsrsLosQuery col_player_los_query(const ColosseumContext* context) {
    return osrs_los_tile(
        col_topology_los_tile_blocked, (void*)context);
}

static int col_npc_has_los_to_player(
    const ColosseumState* state,
    const ColosseumContext* context,
    const ColoNPC* npc
) {
    int size = npc->size > 0 ? npc->size : COLO_NPC_STATS[npc->type].size;
    return col_topology_los_clear(
        context,
        npc->x,
        npc->y,
        size,
        state->player.x,
        state->player.y,
        1,
        0);
}

static inline int
col_npc_blocked_ignore_player_assume_finalized_size_in_range(
    ColoGeometryContext* geometry,
    int x,
    int y,
    int size
) {
    ColosseumState* state = geometry->state;
    state->log.npc_blocked_calls += 1.0f;
    state->log.npc_blocked_tiles += (float)(size * size);
    if (encounter_arena_topology_footprint_blocked_assume_finalized_size_in_range(
            geometry->context->route_topology, x, y, size))
        return 1;
    if (col_sol_clamp_active(state) &&
            (x <= state->sol.boss_arena_min_x ||
             y <= state->sol.boss_arena_min_y ||
             x + size - 1 >= state->sol.boss_arena_max_x ||
             y + size - 1 >= state->sol.boss_arena_max_y))
        return 1;
    int grid_x = x - COLO_ARENA_MIN_X;
    int grid_y = y - COLO_ARENA_MIN_Y;
    for (int dx = 0; dx < size; dx++)
        for (int dy = 0; dy < size; dy++)
            if (state->npc_collision_flags[grid_x + dx][grid_y + dy])
                return 1;
    return 0;
}
static int col_npc_blocked_ignore_player(
    void* data,
    int x,
    int y,
    int size
) {
    ColoGeometryContext* geometry = (ColoGeometryContext*)data;
    const EncounterArenaTopology* topology =
        geometry->context->route_topology;
    encounter_arena_topology_require_finalized(topology);
    encounter_arena_topology_require_footprint_size(topology, size);
    return col_npc_blocked_ignore_player_assume_finalized_size_in_range(
        geometry, x, y, size);
}


static int col_npc_blocked(void* data, int x, int y, int size) {
    ColoGeometryContext* geometry = (ColoGeometryContext*)data;
    ColosseumState* state = geometry->state;
    if (col_npc_blocked_ignore_player(data, x, y, size)) return 1;
    return state->player.x >= x && state->player.x < x + size &&
        state->player.y >= y && state->player.y < y + size;
}

typedef struct {
    ColoGeometryContext geometry;
    int self_idx;
} ColoNpcMoveCtx;

static int col_npc_move_blocked_ignore_player(
    void* data,
    int x,
    int y,
    int size
) {
    ColoNpcMoveCtx* move = (ColoNpcMoveCtx*)data;
    return col_npc_blocked_ignore_player_assume_finalized_size_in_range(
        &move->geometry, x, y, size);
}

static int col_npc_move_overlap_hold(void* data) {
    ColoNpcMoveCtx* move = (ColoNpcMoveCtx*)data;
    ColosseumState* state = move->geometry.state;
    return state->player_last_interaction_age == 0 &&
        state->player_last_interaction_target_slot == move->self_idx;
}

typedef struct {
    ColoGeometryContext geometry;
    const uint16_t (*npc_collision_prefix)[COLO_ARENA_HEIGHT + 1];
    int size;
    int ignore_npcs;
} ColoNpcPathCtx;

static ColoNpcPathCtx col_npc_path_ctx_begin(
    ColosseumState* state,
    const ColosseumContext* context,
    int size
) {
    encounter_arena_topology_require_finalized(context->route_topology);
    encounter_arena_topology_require_footprint_size(
        context->route_topology, size);
    return (ColoNpcPathCtx){
        .geometry = {
            .state = state,
            .context = context,
        },
        .size = size,
    };
}
static uint64_t col_npc_path_blocker_revision(
    const ColoNpcPathCtx* path
) {
    const Player* player = &path->geometry.state->player;
    uint64_t tile =
        (uint64_t)(player->x - COLO_ARENA_MIN_X) *
            COLO_ARENA_HEIGHT +
        (uint64_t)(player->y - COLO_ARENA_MIN_Y);
    return 1 + tile +
        (uint64_t)col_sol_clamp_active(path->geometry.state) *
            COLO_ARENA_WIDTH * COLO_ARENA_HEIGHT;
}

static void col_npc_collision_prefix_build(
    const ColosseumState* state,
    uint16_t prefix[COLO_ARENA_WIDTH + 1][COLO_ARENA_HEIGHT + 1]
) {
    memset(prefix, 0, sizeof(uint16_t) *
        (COLO_ARENA_WIDTH + 1) * (COLO_ARENA_HEIGHT + 1));
    for (int x = 0; x < COLO_ARENA_WIDTH; x++) {
        uint16_t column_sum = 0;
        for (int y = 0; y < COLO_ARENA_HEIGHT; y++) {
            column_sum += state->npc_collision_flags[x][y] != 0;
            prefix[x + 1][y + 1] = prefix[x][y + 1] + column_sum;
        }
    }
}

static int col_npc_path_blocked_uncached(
    ColoNpcPathCtx* path,
    int x,
    int y
) {
    ColosseumState* state = path->geometry.state;
    if (state->player.x >= x && state->player.x < x + path->size &&
            state->player.y >= y && state->player.y < y + path->size)
        return 1;
    if (path->ignore_npcs) {
        if (encounter_arena_topology_footprint_blocked_assume_finalized_size_in_range(
                path->geometry.context->route_topology,
                x,
                y,
                path->size))
            return 1;
        if (!col_sol_clamp_active(state)) return 0;
        for (int dx = 0; dx < path->size; dx++)
            for (int dy = 0; dy < path->size; dy++)
                if (!col_in_boss_arena(state, x + dx, y + dy))
                    return 1;
        return 0;
    }
    if (!path->npc_collision_prefix)
        return col_npc_blocked_ignore_player_assume_finalized_size_in_range(
            &path->geometry, x, y, path->size);
    if (encounter_arena_topology_footprint_blocked_assume_finalized_size_in_range(
            path->geometry.context->route_topology, x, y, path->size))
        return 1;
    if (col_sol_clamp_active(state) &&
            (x <= state->sol.boss_arena_min_x ||
             y <= state->sol.boss_arena_min_y ||
             x + path->size - 1 >= state->sol.boss_arena_max_x ||
             y + path->size - 1 >= state->sol.boss_arena_max_y))
        return 1;
    int x0 = x - COLO_ARENA_MIN_X;
    int y0 = y - COLO_ARENA_MIN_Y;
    int x1 = x0 + path->size;
    int y1 = y0 + path->size;
    uint16_t occupied =
        path->npc_collision_prefix[x1][y1] -
        path->npc_collision_prefix[x0][y1] -
        path->npc_collision_prefix[x1][y0] +
        path->npc_collision_prefix[x0][y0];
    return occupied != 0;
}

static int col_npc_path_blocked(void* data, int x, int y, int size) {
    ColoNpcPathCtx* path = (ColoNpcPathCtx*)data;
    if (size != path->size) {
        fprintf(stderr, "colosseum NPC route actor size changed\n");
        abort();
    }
    return col_npc_path_blocked_uncached(path, x, y);
}

static int col_npc_path_steps_with_ctx(
    ColoNpcPathCtx* pc,
    int* x,
    int* y,
    int tx,
    int ty,
    int maximum_steps
) {
    if (tx < COLO_ARENA_MIN_X) tx = COLO_ARENA_MIN_X;
    if (tx > COLO_ARENA_MAX_X) tx = COLO_ARENA_MAX_X;
    if (ty < COLO_ARENA_MIN_Y) ty = COLO_ARENA_MIN_Y;
    if (ty > COLO_ARENA_MAX_Y) ty = COLO_ARENA_MAX_Y;
    if (*x == tx && *y == ty) return 0;
#ifdef COLO_PROFILE_ENABLED
    int col_prof_enabled =
        COLO_PROFILE_ENABLED() &&
        !pc->geometry.state->profiling_suppressed;
    double col_prof_t0 = col_prof_enabled ? COLO_PROFILE_NOW_MS() : 0.0;
#endif
    EncounterRouteInput input = {
        .topology = pc->geometry.context->route_topology,
        .blockers = {
            .is_blocked = col_npc_path_blocked,
            .ctx = pc,
            .revision = col_npc_path_blocker_revision(pc),
        },
        .source_x = *x,
        .source_y = *y,
        .actor_size = pc->size,
        .target_x = tx,
        .target_y = ty,
        .target_size = 1,
        .target_kind = ENCOUNTER_ROUTE_TARGET_TILE,
        .movement_mode = maximum_steps > 1
            ? ENCOUNTER_ROUTE_MOVEMENT_RUN
            : ENCOUNTER_ROUTE_MOVEMENT_WALK,
        .cost_policy = pc->ignore_npcs
            ? ENCOUNTER_ROUTE_COST_SOUTH_FIRST_REVERSE
            : ENCOUNTER_ROUTE_COST_SOUTH_FIRST_BFS,
        .result_detail = ENCOUNTER_ROUTE_RESULT_NEXT_STEPS,
    };
    EncounterRouteResult route = encounter_route_solve(&input);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_STEP_NPC_PATHFINDING);
#endif
    if ((route.outcome != ROUTE_REACHED_TARGET &&
            route.outcome != ROUTE_REACHED_FALLBACK) ||
            route.distance == 0) {
        return 0;
    }
    *x += route.first_dx;
    *y += route.first_dy;
    if (maximum_steps > 1 && route.distance > 1) {
        *x += route.run_dx;
        *y += route.run_dy;
        return 2;
    }
    return 1;
}

static OSRS_ROUTE_NOINLINE int col_npc_path_step_with_prefix(
    ColosseumState* state,
    const ColosseumContext* context,
    int* x,
    int* y,
    int size,
    int tx,
    int ty
) {
    uint16_t prefix[COLO_ARENA_WIDTH + 1][COLO_ARENA_HEIGHT + 1];
    col_npc_collision_prefix_build(state, prefix);
    ColoNpcPathCtx path = col_npc_path_ctx_begin(state, context, size);
    path.npc_collision_prefix = prefix;
    return col_npc_path_steps_with_ctx(&path, x, y, tx, ty, 1);
}

static int col_npc_path_step(
    ColosseumState* s,
    const ColosseumContext* context,
    int* x,
    int* y,
    int size,
    int tx,
    int ty
) {
    if (size >= COLO_NPC_STATS[COLO_SOL_HEREDIT].size)
        return col_npc_path_step_with_prefix(
            s, context, x, y, size, tx, ty);
    ColoNpcPathCtx pc = col_npc_path_ctx_begin(s, context, size);
    return col_npc_path_steps_with_ctx(&pc, x, y, tx, ty, 1);
}



static void col_warband_pick_target(
    ColosseumState* s, ColoNpcPathCtx* pc, ColoNPC* npc, int* tx, int* ty) {
    int px = s->player.x, py = s->player.y;
    int pref = colo_npc_warband(npc)->formation_dir;
    int fx = px + COLO_WARBAND_FORM_OFFSET[pref][0];
    int fy = py + COLO_WARBAND_FORM_OFFSET[pref][1];
    *tx = fx;
    *ty = fy;
    if (!col_npc_path_blocked(pc, fx, fy, pc->size)) return;

    static const int ADJ[8][2] = {
        {0, 1}, {1, 0}, {-1, 0}, {0, -1},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1} };
    int best_key = 1 << 30;
    for (int a = 0; a < 8; a++) {
        int x = px + ADJ[a][0], y = py + ADJ[a][1];
        if (col_npc_path_blocked(pc, x, y, pc->size)) continue;
        int dx = npc->x > x ? npc->x - x : x - npc->x;
        int dy = npc->y > y ? npc->y - y : y - npc->y;
        int cheb = dx > dy ? dx : dy;
        int key = (a >= 4 ? 4096 : 0) + cheb * 16 + a;
        if (key < best_key) { best_key = key; *tx = x; *ty = y; }
    }
}

static void col_warband_move(
    ColosseumState* s,
    const ColosseumContext* ctx,
    int idx
) {
    ColoNPC* npc = &s->npcs[idx];
    npc->target_x = s->player.x;
    npc->target_y = s->player.y;

    ColoNpcPathCtx pc = col_npc_path_ctx_begin(s, ctx, 1);
    pc.ignore_npcs = 1;
    int tx, ty;
    col_warband_pick_target(s, &pc, npc, &tx, &ty);
    int steps = col_npc_path_steps_with_ctx(
        &pc, &npc->x, &npc->y, tx, ty, COLO_WARBAND_TILES_PER_TICK);
    npc->moved_this_tick = steps > 0 ? 1 : 0;
}

static void col_npc_move_ctx(ColosseumState* s, ColosseumContext* ctx, int idx) {
    if (!ctx || !ctx->route_topology) {
        fprintf(stderr, "colosseum route context is not bound\n");
        abort();
    }
    ColoNPC* npc = &s->npcs[idx];
    const ColoNpcStats* st = &COLO_NPC_STATS[npc->type];
    if (npc->type == COLO_SOL_HEREDIT) return;
    if (!st->can_move) return;
    if (npc->stun_timer > 0 || npc->frozen_ticks > 0) return;

    if (col_type_is_warbander(npc->type)) {
        col_warband_move(s, ctx, idx);
        return;
    }

    int size = col_npc_effective_size(npc);
    int tx = s->player.x, ty = s->player.y;
    npc->target_x = tx;
    npc->target_y = ty;

    int is_ranged = st->attack_range > 1;
    if (is_ranged) {
        int dist = encounter_dist_to_npc(s->player.x, s->player.y, npc->x, npc->y, size);
        if (dist >= 1 && dist <= st->attack_range &&
                col_npc_has_los_to_player(s, ctx, npc)) {
            npc->moved_this_tick = 0;
            return;
        }
    }
    if (npc->attack_timer > st->attack_speed) {
        npc->moved_this_tick = 0;
        return;
    }

    col_stamp_npc_collision(s, npc, 0);
    ColoNpcMoveCtx move_ctx = {
        .geometry = {
            .state = s,
            .context = ctx,
        },
        .self_idx = idx,
    };

    int moved;
    if (npc->type == COLO_MINOTAUR && col_mod_active(s, COLO_MOD_RED_FLAG)) {

        moved = 0;
        if (encounter_dist_to_npc(tx, ty, npc->x, npc->y, size) > 1)
            moved = col_npc_path_step(
                s, ctx, &npc->x, &npc->y, size, tx - 1, ty - 1);
    } else {
        EncounterNpcStepPolicy step_policy = is_ranged
            ? ENCOUNTER_NPC_STEP_OSRS_AGGRO_TARGET
            : ENCOUNTER_NPC_STEP_OSRS_AGGRO_STOP_AT_MELEE;
        moved = encounter_npc_step_toward_policy(
            &npc->x, &npc->y, tx, ty, size, 1, step_policy,
            col_npc_move_blocked_ignore_player, &move_ctx,
            col_npc_move_overlap_hold, &s->rng_state);
    }
    col_stamp_npc_collision(s, npc, 1);
    npc->moved_this_tick = moved ? 1 : 0;
}
