#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "ocean/osrs/encounters/encounter_colosseum.h"
#ifdef OSRS_INTERACTION_SERIALIZED_ROUTE_BYTES
#include "ocean/osrs/tests/osrs_route_reference.h"
#endif

static void col_step_out_forecast_landing_selftest_one_state(
    ColosseumState* s,
    ColosseumContext* ctx
) {
    for (int x = COLO_ARENA_MIN_X - 2; x <= COLO_ARENA_MAX_X + 2; x++) {
        for (int y = COLO_ARENA_MIN_Y - 2; y <= COLO_ARENA_MAX_Y + 2; y++) {
            s->player.x = x;
            s->player.y = y;
            for (int action_idx = 0; action_idx < ENCOUNTER_MOVE_ACTIONS; action_idx++) {
                int valid =
                    col_step_out_forecast_action_valid(s, ctx, action_idx);
                Player moved = s->player;
                ColoGeometryContext geometry = {
                    .state = s,
                    .context = ctx,
                };
                if (action_idx != 0) {
                    encounter_move_to_target(
                        &moved,
                        ENCOUNTER_MOVE_TARGET_DX[action_idx],
                        ENCOUNTER_MOVE_TARGET_DY[action_idx],
                        col_player_walkable,
                        &geometry);
                }
                ColoForecastLanding landing =
                    col_step_out_forecast_action_landing_ctx(
                        s, ctx, action_idx);
                if (landing.valid != valid ||
                        landing.land_x != moved.x ||
                        landing.land_y != moved.y) {
                    fprintf(stderr,
                        "colosseum landing mismatch player=(%d,%d) action=%d: helper=(%d,%d,%d) full=(%d,%d,%d)\n",
                        x, y, action_idx,
                        landing.valid, landing.land_x, landing.land_y,
                        valid, moved.x, moved.y);
                    abort();
                }
            }
        }
    }
}

static void col_step_out_forecast_landing_selftest(void) {
    ColosseumContext ctx;
    ColosseumState s;
    col_init_context_typed(&ctx);
    memset(&s, 0, sizeof(s));
    col_finalize_route_topology(&ctx);
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 0x51A7u);
    memset(s.npcs, 0, sizeof(s.npcs));
    memset(s.npc_collision_flags, 0, sizeof(s.npc_collision_flags));
    col_rebuild_player_collision_flags(&s);
    col_step_out_forecast_landing_selftest_one_state(&s, &ctx);

    col_spawn_npc_at(&s, COLO_SERPENT_SHAMAN, 12, 9);
    col_spawn_npc_at(&s, COLO_JAVELIN_COLOSSUS, 18, 18);
    col_rebuild_player_collision_flags(&s);
    col_step_out_forecast_landing_selftest_one_state(&s, &ctx);

    s.wave = COLO_WAVE_BOSS;
    s.sol.started = 1;
    s.sol.boss_arena_min_x = COLO_BOSS_ARENA_MIN_X;
    s.sol.boss_arena_min_y = COLO_BOSS_ARENA_MIN_Y;
    s.sol.boss_arena_max_x = COLO_BOSS_ARENA_MAX_X;
    s.sol.boss_arena_max_y = COLO_BOSS_ARENA_MAX_Y;
    col_step_out_forecast_landing_selftest_one_state(&s, &ctx);

    printf("colosseum landing helper selftest PASS: %d actions across 3 states\n",
        ENCOUNTER_MOVE_ACTIONS);
}

static int exact_static_tile_blocked_reference(int x, int y) {
    if (x < COLO_ARENA_MIN_X || x > COLO_ARENA_MAX_X ||
            y < COLO_ARENA_MIN_Y || y > COLO_ARENA_MAX_Y)
        return 1;
    for (int span = 0; span < COLO_WALL_SPANS_PER_ROW; span++) {
        ColoWallSpan wall = COLO_WALL_SPANS[y][span];
        if (x >= wall.lo && x < wall.hi) return 1;
    }
    for (int pillar = 0; pillar < COLO_NUM_PILLARS; pillar++) {
        int pillar_x = COLO_PILLARS[pillar][0];
        int pillar_y = COLO_PILLARS[pillar][1];
        if (x >= pillar_x && x < pillar_x + COLO_PILLAR_SIZE &&
                y >= pillar_y && y < pillar_y + COLO_PILLAR_SIZE)
            return 1;
    }
    return 0;
}

static int exact_static_los_blocked(void* data, int x, int y) {
    (void)data;
    return exact_static_tile_blocked_reference(x, y);
}

static int exact_static_footprint_blocked_reference(int x, int y, int size) {
    for (int dx = 0; dx < size; dx++)
        for (int dy = 0; dy < size; dy++)
            if (exact_static_tile_blocked_reference(x + dx, y + dy))
                return 1;
    return 0;
}

static void col_static_los_table_selftest(void) {
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    col_finalize_route_topology(&ctx);
    OsrsLosQuery reference =
        osrs_los_tile(exact_static_los_blocked, NULL);
    int checks = 0;
    for (int x0 = COLO_ARENA_MIN_X; x0 <= COLO_ARENA_MAX_X; x0++) {
        for (int y0 = COLO_ARENA_MIN_Y; y0 <= COLO_ARENA_MAX_Y; y0++) {
            for (int x1 = COLO_ARENA_MIN_X; x1 <= COLO_ARENA_MAX_X; x1++) {
                for (int y1 = COLO_ARENA_MIN_Y; y1 <= COLO_ARENA_MAX_Y; y1++) {
                    if (x0 == x1 && y0 == y1) continue;
                    int topology = col_topology_los_clear(
                        &ctx, x0, y0, 1, x1, y1, 1, 0);
                    int slow =
                        osrs_los_tile_ray_clear(&reference, x0, y0, x1, y1);
                    if (topology != slow) {
                        fprintf(stderr,
                            "colosseum topology LoS mismatch (%d,%d)->(%d,%d): topology=%d reference=%d\n",
                            x0, y0, x1, y1, topology, slow);
                        abort();
                    }
                    checks++;
                }
            }
        }
    }
    printf("colosseum topology LoS parity PASS: %d directed pairs\n", checks);
}
static void col_static_los_endpoint_semantics_selftest(void) {
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    col_finalize_route_topology(&ctx);
    OsrsLosQuery reference =
        osrs_los_tile(exact_static_los_blocked, NULL);
    uint64_t raw_clear[2][2] = {{0}};
    uint64_t ranged_clear[2][2] = {{0}};
    uint64_t total[2][2] = {{0}};

    for (int source_x = COLO_ARENA_MIN_X;
            source_x <= COLO_ARENA_MAX_X;
            source_x++) {
        for (int source_y = COLO_ARENA_MIN_Y;
                source_y <= COLO_ARENA_MAX_Y;
                source_y++) {
            int source_blocked =
                exact_static_tile_blocked_reference(source_x, source_y);
            for (int target_x = COLO_ARENA_MIN_X;
                    target_x <= COLO_ARENA_MAX_X;
                    target_x++) {
                for (int target_y = COLO_ARENA_MIN_Y;
                        target_y <= COLO_ARENA_MAX_Y;
                        target_y++) {
                    int target_blocked =
                        exact_static_tile_blocked_reference(
                            target_x, target_y);
                    int raw = osrs_los_tile_ray_clear(
                        &reference,
                        source_x,
                        source_y,
                        target_x,
                        target_y);
                    int topology_raw = col_topology_los_clear(
                        &ctx,
                        source_x,
                        source_y,
                        1,
                        target_x,
                        target_y,
                        1,
                        0);
                    if (raw != topology_raw) {
                        fprintf(stderr,
                            "colosseum raw endpoint LoS mismatch (%d,%d)->(%d,%d): topology=%d reference=%d\n",
                            source_x,
                            source_y,
                            target_x,
                            target_y,
                            topology_raw,
                            raw);
                        abort();
                    }

                    int range_one = osrs_los_clear(
                        &reference,
                        source_x,
                        source_y,
                        1,
                        target_x,
                        target_y,
                        1,
                        1);
                    if (!range_one) {
                        fprintf(stderr,
                            "colosseum tile LoS range-one law changed (%d,%d)->(%d,%d)\n",
                            source_x,
                            source_y,
                            target_x,
                            target_y);
                        abort();
                    }

                    int ranged = osrs_los_clear(
                        &reference,
                        source_x,
                        source_y,
                        1,
                        target_x,
                        target_y,
                        1,
                        COLO_ARENA_WIDTH);
                    int topology_ranged = col_topology_los_clear(
                        &ctx,
                        target_x,
                        target_y,
                        1,
                        source_x,
                        source_y,
                        1,
                        0);
                    if (ranged != topology_ranged) {
                        fprintf(stderr,
                            "colosseum ranged endpoint LoS mismatch actor=(%d,%d) target=(%d,%d): topology=%d reference=%d\n",
                            source_x,
                            source_y,
                            target_x,
                            target_y,
                            topology_ranged,
                            ranged);
                        abort();
                    }

                    total[source_blocked][target_blocked]++;
                    raw_clear[source_blocked][target_blocked] +=
                        (uint64_t)raw;
                    ranged_clear[source_blocked][target_blocked] +=
                        (uint64_t)ranged;
                }
            }
        }
    }

    printf(
        "colosseum endpoint LoS parity PASS: raw clear/total OO=%llu/%llu OB=%llu/%llu BO=%llu/%llu BB=%llu/%llu; ranged OO=%llu OB=%llu BO=%llu BB=%llu; range-one=all-clear\n",
        (unsigned long long)raw_clear[0][0],
        (unsigned long long)total[0][0],
        (unsigned long long)raw_clear[0][1],
        (unsigned long long)total[0][1],
        (unsigned long long)raw_clear[1][0],
        (unsigned long long)total[1][0],
        (unsigned long long)raw_clear[1][1],
        (unsigned long long)total[1][1],
        (unsigned long long)ranged_clear[0][0],
        (unsigned long long)ranged_clear[0][1],
        (unsigned long long)ranged_clear[1][0],
        (unsigned long long)ranged_clear[1][1]);
}


static void col_static_footprint_table_selftest(void) {
    ColosseumContext ctx;
    col_init_context_typed(&ctx);
    col_finalize_route_topology(&ctx);
    col_build_npc_stats();
    int sizes[ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE + 1] = {0};
    for (int type = 0; type < COLO_NUM_NPC_TYPES; type++)
        sizes[COLO_NPC_STATS[type].size] = 1;
    int checks = 0;
    for (int size = 1; size <= ENCOUNTER_ARENA_TOPOLOGY_MAX_FOOTPRINT_SIZE; size++) {
        if (!sizes[size]) continue;
        for (int x = COLO_ARENA_MIN_X - size; x <= COLO_ARENA_MAX_X + 1; x++) {
            for (int y = COLO_ARENA_MIN_Y - size; y <= COLO_ARENA_MAX_Y + 1; y++) {
                int topology =
                    col_topology_footprint_blocked(&ctx, x, y, size);
                int slow =
                    exact_static_footprint_blocked_reference(x, y, size);
                if (topology != slow) {
                    fprintf(stderr,
                        "colosseum topology footprint mismatch (%d,%d) size=%d: topology=%d reference=%d\n",
                        x, y, size, topology, slow);
                    abort();
                }
                checks++;
            }
        }
    }
    printf("colosseum topology footprint parity PASS: %d NPC-size checks\n",
        checks);
}

#define col_init_context_typed(ctx_ptr) do { \
    col_init_context_typed(ctx_ptr); \
    (ctx_ptr)->config.late_start_state_mode = 0; \
} while (0)

#define EXACT_MAGIC "COLOEXACTv1"
#define EXACT_VERSION 5u
#define EXACT_CHUNK_BYTES 65536

typedef struct {
    char magic[16];
    uint32_t version;
    uint32_t state_size;
    uint32_t obs_size;
    uint32_t action_mask_size;
    uint32_t record_count;
} ColoExactFileHeader;

typedef struct {
    uint32_t scenario_id;
    uint32_t step_index;
    uint32_t tick;
    uint32_t wave;
    uint32_t terminal;
    uint32_t winner;
    uint32_t state_size;
    uint32_t obs_size;
    uint32_t action_mask_size;
    uint64_t state_hash;
    uint64_t obs_hash;
    uint64_t action_mask_hash;
    float reward;
} ColoExactRecordHeader;

typedef struct {
    FILE* file;
    uint32_t record_count;
} ColoExactWriter;

static uint64_t exact_fnv_bytes(uint64_t h, const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < size; i++) {
        h ^= bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t exact_hash_bytes(const void* data, size_t size) {
    return exact_fnv_bytes(1469598103934665603ULL, data, size);
}

static void exact_write_all(FILE* file, const void* data, size_t size) {
    if (fwrite(data, 1, size, file) != size) {
        perror("write colosseum exact fixture");
        abort();
    }
}

static void exact_readable_path(
    char* out,
    size_t out_size,
    const char* dir,
    const char* file_name
) {
    int n = snprintf(out, out_size, "%s/%s", dir, file_name);
    if (n < 0 || (size_t)n >= out_size) {
        fprintf(stderr, "fixture path too long: %s/%s\n", dir, file_name);
        abort();
    }
}

static void exact_mkdir_if_needed(const char* dir) {
    if (mkdir(dir, 0777) == 0) return;
    if (errno == EEXIST) return;
    perror("mkdir colosseum exact fixture dir");
    abort();
}

static void exact_writer_open(ColoExactWriter* writer, const char* path) {
    memset(writer, 0, sizeof(*writer));
    writer->file = fopen(path, "wb");
    if (!writer->file) {
        perror("open colosseum exact fixture");
        abort();
    }

    ColoExactFileHeader header = {0};
    memcpy(header.magic, EXACT_MAGIC, sizeof(EXACT_MAGIC));
    header.version = EXACT_VERSION;
    header.state_size = (uint32_t)sizeof(ColosseumState);
    header.obs_size = COLO_NUM_OBS;
    header.action_mask_size = COLO_ACTION_MASK_SIZE;
    exact_write_all(writer->file, &header, sizeof(header));
}

static void exact_writer_close(ColoExactWriter* writer) {
    ColoExactFileHeader header = {0};
    memcpy(header.magic, EXACT_MAGIC, sizeof(EXACT_MAGIC));
    header.version = EXACT_VERSION;
    header.state_size = (uint32_t)sizeof(ColosseumState);
    header.obs_size = COLO_NUM_OBS;
    header.action_mask_size = COLO_ACTION_MASK_SIZE;
    header.record_count = writer->record_count;
    if (fseek(writer->file, 0, SEEK_SET) != 0) {
        perror("seek colosseum exact fixture");
        abort();
    }
    exact_write_all(writer->file, &header, sizeof(header));
    if (fclose(writer->file) != 0) {
        perror("close colosseum exact fixture");
        abort();
    }
    writer->file = NULL;
}

static void exact_zero_serialized_route_storage(ColosseumState* state) {
#ifdef OSRS_INTERACTION_SERIALIZED_ROUTE_BYTES
    osrs_interaction_zero_serialized_route_padding(&state->player.interaction);
    osrs_interaction_zero_serialized_route_padding(&state->interaction);
#else
    memset(&state->player.interaction.route, 0, sizeof(state->player.interaction.route));
    memset(&state->interaction.route, 0, sizeof(state->interaction.route));
#endif
    state->log.npc_blocked_calls = 0.0f;
    state->log.npc_blocked_tiles = 0.0f;
}

static void exact_capture(
    ColoExactWriter* writer,
    uint32_t scenario_id,
    uint32_t step_index,
    ColosseumState* s,
    ColosseumContext* ctx
) {
    float obs[COLO_NUM_OBS];
    float action_mask[COLO_ACTION_MASK_SIZE];

    col_write_obs_ctx((EncounterState*)s, (EncounterContext*)ctx, obs);
    col_write_mask_ctx((EncounterState*)s, (EncounterContext*)ctx, action_mask);
    ColosseumState canonical_state = *s;
    exact_zero_serialized_route_storage(&canonical_state);

    ColoExactRecordHeader record = {0};
    record.scenario_id = scenario_id;
    record.step_index = step_index;
    record.tick = (uint32_t)s->tick;
    record.wave = (uint32_t)s->wave;
    record.terminal = (uint32_t)col_is_terminal_ctx(
        (EncounterState*)s, (EncounterContext*)ctx);
    record.winner = (uint32_t)s->winner;
    record.state_size = (uint32_t)sizeof(*s);
    record.obs_size = COLO_NUM_OBS;
    record.action_mask_size = COLO_ACTION_MASK_SIZE;
    record.state_hash = exact_hash_bytes(&canonical_state, sizeof(canonical_state));
    record.obs_hash = exact_hash_bytes(obs, sizeof(obs));
    record.action_mask_hash = exact_hash_bytes(action_mask, sizeof(action_mask));
    record.reward = col_get_reward_ctx((EncounterState*)s, (EncounterContext*)ctx);

    exact_write_all(writer->file, &record, sizeof(record));
    exact_write_all(writer->file, obs, sizeof(obs));
    exact_write_all(writer->file, action_mask, sizeof(action_mask));
    exact_write_all(writer->file, &canonical_state, sizeof(canonical_state));
    writer->record_count++;
}

static uint64_t exact_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void exact_zero_actions(int actions[COLO_NUM_ACTION_HEADS]) {
    memset(actions, 0, sizeof(int) * COLO_NUM_ACTION_HEADS);
}

static void exact_trace_actions(
    ColosseumState* s,
    uint64_t* rng,
    int actions[COLO_NUM_ACTION_HEADS]
) {
    for (int head = 0; head < COLO_NUM_ACTION_HEADS; head++) {
        actions[head] = (int)(exact_splitmix64(rng) % (uint64_t)COLO_ACTION_DIMS[head]);
    }
    if (s->modifiers.draft_pending) {
        actions[COLO_HEAD_PRIMARY] = 0;
        actions[COLO_HEAD_MODIFIER_SELECT] = 1 +
            (int)(exact_splitmix64(rng) % COLO_MODIFIER_DRAFT_OPTIONS);
    }
}

static void exact_init_state(
    ColosseumState* s,
    ColosseumContext* ctx,
    int start_wave,
    uint32_t seed
) {
    col_init_context_typed(ctx);
    ctx->config.start_wave = start_wave;
    memset(s, 0, sizeof(*s));
    col_finalize_route_topology(ctx);
    col_reset_ctx((EncounterState*)s, (EncounterContext*)ctx, seed);
}

static void exact_refresh_geometry(ColosseumState* s, ColosseumContext* ctx) {
    col_rebuild_player_collision_flags(s);
    col_refresh_current_obs_slots_ctx(s, ctx);
}

static void exact_clear_npcs(ColosseumState* s, ColosseumContext* ctx) {
    memset(s->npcs, 0, sizeof(s->npcs));
    memset(s->npc_collision_flags, 0, sizeof(s->npc_collision_flags));
    memset(s->totems, 0, sizeof(s->totems));
    memset(s->bees, 0, sizeof(s->bees));
    exact_refresh_geometry(s, ctx);
}

static void exact_prepare_custom(
    ColosseumState* s,
    ColosseumContext* ctx,
    uint32_t seed,
    int player_x,
    int player_y
) {
    exact_init_state(s, ctx, 1, seed);
    exact_clear_npcs(s, ctx);
    s->modifiers.draft_pending = 0;
    s->modifiers.draft_gates_spawn = 0;
    s->wave_ready_delay = 0;
    s->wave_spawn_delay = 0;
    s->reinforcement_timer = COLO_REINFORCE_FIRED;
    s->warband_cycle_anchor = s->tick;
    s->player.x = player_x;
    s->player.y = player_y;
    exact_refresh_geometry(s, ctx);
}

#ifdef OSRS_INTERACTION_SERIALIZED_ROUTE_BYTES
typedef struct {
    ColoWalkCtx walk;
    int blocker_count;
    int blocker_x[16];
    int blocker_y[16];
} ExactAttackRouteContext;

static int exact_attack_route_walkable(void* ctx, int x, int y) {
    ExactAttackRouteContext* route_ctx = (ExactAttackRouteContext*)ctx;
    return col_tile_walkable(&route_ctx->walk, x, y);
}

static int exact_attack_route_blocked(void* ctx, int abs_x, int abs_y) {
    ExactAttackRouteContext* route_ctx = (ExactAttackRouteContext*)ctx;
    int x = abs_x - route_ctx->walk.ctx->world_offset_x;
    int y = abs_y - route_ctx->walk.ctx->world_offset_y;
    for (int i = 0; i < route_ctx->blocker_count; i++) {
        if (route_ctx->blocker_x[i] == x &&
                route_ctx->blocker_y[i] == y) {
            return 1;
        }
    }
    return 0;
}

static int exact_optimized_route_blocked(void* data, int x, int y, int size) {
    (void)size;
    ExactAttackRouteContext* route_ctx = (ExactAttackRouteContext*)data;
    if (!exact_attack_route_walkable(route_ctx, x, y)) return 1;
    for (int i = 0; i < route_ctx->blocker_count; i++) {
        if (route_ctx->blocker_x[i] == x &&
                route_ctx->blocker_y[i] == y)
            return 1;
    }
    return 0;
}
static int exact_attack_route_can_attack(
    void* data,
    int player_x,
    int player_y,
    int target_x,
    int target_y,
    int target_size,
    int attack_range
) {
    ExactAttackRouteContext* route_ctx = (ExactAttackRouteContext*)data;
    OsrsLosQuery query =
        col_player_los_query(route_ctx->walk.ctx);
    return encounter_player_can_attack(
        player_x,
        player_y,
        target_x,
        target_y,
        target_size,
        attack_range,
        route_ctx->walk.ctx->collision_map,
        route_ctx->walk.ctx->world_offset_x,
        route_ctx->walk.ctx->world_offset_y,
        &query);
}

static EncounterAttackRouteLanding exact_attack_route_optimized_landing(
    const ColosseumState* s,
    const ColosseumContext* ctx,
    ExactAttackRouteContext* route_ctx,
    int target_x,
    int target_y,
    int target_size,
    int attack_range,
    const OsrsLosQuery* los_query
) {
    Player player = s->player;
    if (encounter_player_can_attack(
            player.x, player.y,
            target_x, target_y, target_size, attack_range,
            ctx->collision_map,
            ctx->world_offset_x,
            ctx->world_offset_y,
            los_query))
        return (EncounterAttackRouteLanding){
            .land_x = player.x,
            .land_y = player.y,
        };
    EncounterRouteInput input = {
        .topology = ctx->route_topology,
        .blockers = {
            .is_blocked = exact_optimized_route_blocked,
            .ctx = route_ctx,
            .revision = (uint64_t)route_ctx->blocker_count + 1,
        },
        .source_x = player.x,
        .source_y = player.y,
        .actor_size = 1,
        .target_x = target_x,
        .target_y = target_y,
        .target_size = target_size,
        .target_kind = ENCOUNTER_ROUTE_TARGET_CARDINAL_ADJACENCY,
        .movement_mode = ENCOUNTER_ROUTE_MOVEMENT_RUN,
        .cost_policy = ENCOUNTER_ROUTE_COST_OSRS,
    };
    EncounterRouteResult route = encounter_route_solve(&input);
    if (route.outcome == ROUTE_REACHED_TARGET ||
            route.outcome == ROUTE_REACHED_FALLBACK) {
        player.x += route.first_dx;
        player.y += route.first_dy;
        if ((route.run_dx != 0 || route.run_dy != 0) &&
                !encounter_player_can_attack(
                    player.x, player.y,
                    target_x, target_y, target_size, attack_range,
                    ctx->collision_map,
                    ctx->world_offset_x,
                    ctx->world_offset_y,
                    los_query)) {
            player.x += route.run_dx;
            player.y += route.run_dy;
        }
    }
    return (EncounterAttackRouteLanding){
        .land_x = player.x,
        .land_y = player.y,
    };
}

static EncounterAttackRouteLanding exact_attack_route_runtime_landing(
    const ColosseumState* s,
    const ColosseumContext* ctx,
    ExactAttackRouteContext* route_ctx,
    int target_x,
    int target_y,
    int target_size,
    int attack_range,
    const OsrsLosQuery* los_query
) {
    Player player = s->player;
    encounter_chase_attack_target(
        &player,
        target_x,
        target_y,
        target_size,
        attack_range,
        ctx->collision_map,
        ctx->world_offset_x,
        ctx->world_offset_y,
        exact_attack_route_walkable,
        route_ctx,
        exact_attack_route_blocked,
        route_ctx,
        los_query,
        COLO_ARENA_MIN_X,
        COLO_ARENA_MIN_Y,
        COLO_ARENA_WIDTH,
        COLO_ARENA_HEIGHT);
    return (EncounterAttackRouteLanding){
        .land_x = player.x,
        .land_y = player.y,
    };
}

static int exact_attack_route_property_scenario(
    const char* scenario,
    ColosseumState* s,
    ColosseumContext* ctx,
    ExactAttackRouteContext* route_ctx
) {
    OsrsLosQuery los_query = col_player_los_query(ctx);
    static const int target_sizes[] = {1, 2, 5};
    static const int attack_ranges[] = {1, 3, 6, 10};
    int checks = 0;

    for (int target_y = COLO_ARENA_MIN_Y - 1;
            target_y <= COLO_ARENA_MAX_Y + 1;
            target_y += 5) {
        for (int target_x = COLO_ARENA_MIN_X - 1;
                target_x <= COLO_ARENA_MAX_X + 1;
                target_x += 4) {
            for (size_t size_idx = 0;
                    size_idx < sizeof(target_sizes) / sizeof(target_sizes[0]);
                    size_idx++) {
                int target_size = target_sizes[size_idx];
                EncounterArenaAttackRouteField field;
                if (target_x < COLO_ARENA_MIN_X ||
                        target_y < COLO_ARENA_MIN_Y ||
                        target_x + target_size - 1 > COLO_ARENA_MAX_X ||
                        target_y + target_size - 1 > COLO_ARENA_MAX_Y)
                    continue;
                encounter_build_arena_attack_route_field(
                    &field,
                    ctx->collision_map,
                    ctx->world_offset_x,
                    ctx->world_offset_y,
                    s->player.x,
                    s->player.y,
                    target_x,
                    target_y,
                    target_size,
                    exact_attack_route_walkable,
                    route_ctx,
                    exact_attack_route_blocked,
                    route_ctx,
                    COLO_ARENA_MIN_X,
                    COLO_ARENA_MIN_Y,
                    COLO_ARENA_WIDTH,
                    COLO_ARENA_HEIGHT);
                for (size_t range_idx = 0;
                        range_idx < sizeof(attack_ranges) / sizeof(attack_ranges[0]);
                        range_idx++) {
                    int attack_range = attack_ranges[range_idx];
                    EncounterAttackRouteLanding expected =
                        exact_attack_route_runtime_landing(
                            s, ctx, route_ctx,
                            target_x, target_y, target_size, attack_range,
                            &los_query);
                    EncounterAttackRouteLanding actual =
                        exact_attack_route_optimized_landing(
                            s, ctx, route_ctx,
                            target_x, target_y, target_size, attack_range,
                            &los_query);
                    if (actual.land_x != expected.land_x ||
                            actual.land_y != expected.land_y) {
                        fprintf(stderr,
                            "attack route mismatch scenario=%s player=(%d,%d) target=(%d,%d,%d) range=%d field=(%d,%d) runtime=(%d,%d)\n",
                            scenario,
                            s->player.x,
                            s->player.y,
                            target_x,
                            target_y,
                            target_size,
                            attack_range,
                            actual.land_x,
                            actual.land_y,
                            expected.land_x,
                            expected.land_y);
                        abort();
                    }
                    checks++;
                }
            }
        }
    }
    return checks;
}
static void exact_attack_route_property_selftest(void) {
    ColosseumState s;
    ColosseumContext ctx;
    exact_prepare_custom(
        &s, &ctx, 0xA771u, COLO_PLAYER_START_X, COLO_PLAYER_START_Y);
    ExactAttackRouteContext route_ctx = {
        .walk = {&s, &ctx},
    };
    int checks = exact_attack_route_property_scenario(
        "open-arena", &s, &ctx, &route_ctx);

    route_ctx.blocker_count = 8;
    int blocker_x[] = {5, 6, 7, 8, 9, 8, 7, 6};
    int blocker_y[] = {18, 17, 16, 17, 18, 19, 20, 19};
    memcpy(route_ctx.blocker_x, blocker_x, sizeof(blocker_x));
    memcpy(route_ctx.blocker_y, blocker_y, sizeof(blocker_y));
    checks += exact_attack_route_property_scenario(
        "dynamic-blockers", &s, &ctx, &route_ctx);

    route_ctx.blocker_count = 0;
    s.wave = COLO_WAVE_BOSS;
    s.sol.started = 1;
    s.sol.boss_arena_min_x = COLO_BOSS_ARENA_MIN_X;
    s.sol.boss_arena_min_y = COLO_BOSS_ARENA_MIN_Y;
    s.sol.boss_arena_max_x = COLO_BOSS_ARENA_MAX_X;
    s.sol.boss_arena_max_y = COLO_BOSS_ARENA_MAX_Y;
    s.player.x = -1;
    s.player.y = -1;
    for (int x = s.sol.boss_arena_min_x;
            x <= s.sol.boss_arena_max_x && s.player.x < 0;
            x++) {
        for (int y = s.sol.boss_arena_min_y;
                y <= s.sol.boss_arena_max_y;
                y++) {
            if (!encounter_arena_topology_tile_blocked(
                    ctx.route_topology, x, y)) {
                s.player.x = x;
                s.player.y = y;
                break;
            }
        }
    }
    exact_refresh_geometry(&s, &ctx);
    checks += exact_attack_route_property_scenario(
        "sol-clamp", &s, &ctx, &route_ctx);
    static const int exhaustive_attack_ranges[] = {1, 3, 6, 10};
    exact_prepare_custom(
        &s, &ctx, 0xA771u, COLO_PLAYER_START_X, COLO_PLAYER_START_Y);
    route_ctx.walk.s = &s;
    route_ctx.walk.ctx = &ctx;
    route_ctx.blocker_count = 0;
    uint64_t exhaustive_checks = osrs_reference_route_exhaustive(
        "open-arena",
        ctx.route_topology,
        (EncounterRouteBlockers){
            .is_blocked = exact_optimized_route_blocked,
            .ctx = &route_ctx,
            .revision = 1,
        },
        5,
        exhaustive_attack_ranges,
        4,
        exact_attack_route_can_attack,
        &route_ctx);
    route_ctx.blocker_count = 8;
    memcpy(route_ctx.blocker_x, blocker_x, sizeof(blocker_x));
    memcpy(route_ctx.blocker_y, blocker_y, sizeof(blocker_y));
    exhaustive_checks += osrs_reference_route_exhaustive(
        "dynamic-blockers",
        ctx.route_topology,
        (EncounterRouteBlockers){
            .is_blocked = exact_optimized_route_blocked,
            .ctx = &route_ctx,
            .revision = 9,
        },
        5,
        exhaustive_attack_ranges,
        4,
        exact_attack_route_can_attack,
        &route_ctx);
    printf(
        "colosseum exhaustive route equivalence PASS: %llu source-target-range queries across 2 blocker fields\n",
        (unsigned long long)exhaustive_checks);
    printf(
        "colosseum attack route property selftest PASS: %d target queries across 3 fields\n",
        checks);
}
#endif

static void exact_run_steps(
    ColoExactWriter* writer,
    uint32_t scenario_id,
    ColosseumState* s,
    ColosseumContext* ctx,
    int steps,
    uint64_t action_seed
) {
    uint32_t capture_idx = 0;
    int actions[COLO_NUM_ACTION_HEADS];
    uint64_t rng = action_seed;
    exact_capture(writer, scenario_id, capture_idx++, s, ctx);
    for (int step = 0; step < steps; step++) {
        exact_trace_actions(s, &rng, actions);
        col_step_ctx((EncounterState*)s, (EncounterContext*)ctx, actions);
        exact_capture(writer, scenario_id, capture_idx++, s, ctx);
        if (s->episode_over) break;
    }
}

static void exact_run_idle_steps(
    ColoExactWriter* writer,
    uint32_t scenario_id,
    ColosseumState* s,
    ColosseumContext* ctx,
    int steps
) {
    uint32_t capture_idx = 0;
    int actions[COLO_NUM_ACTION_HEADS];
    exact_zero_actions(actions);
    exact_capture(writer, scenario_id, capture_idx++, s, ctx);
    for (int step = 0; step < steps; step++) {
        col_step_ctx((EncounterState*)s, (EncounterContext*)ctx, actions);
        exact_capture(writer, scenario_id, capture_idx++, s, ctx);
        if (s->episode_over) break;
    }
}

static void exact_scenario_wave_rollout(
    ColoExactWriter* writer,
    uint32_t scenario_id,
    int start_wave,
    uint32_t seed,
    int steps
) {
    ColosseumState s;
    ColosseumContext ctx;
    exact_init_state(&s, &ctx, start_wave, seed);
    exact_run_steps(writer, scenario_id, &s, &ctx, steps,
        ((uint64_t)scenario_id << 40) ^ seed ^ 0x53A91C4D12ULL);
}

static void exact_scenario_pillar_safespot(ColoExactWriter* writer) {
    ColosseumState s;
    ColosseumContext ctx;
    exact_prepare_custom(&s, &ctx, 0x1101u, 7, 9);
    col_spawn_npc_at(&s, COLO_SERPENT_SHAMAN, 12, 9);
    exact_refresh_geometry(&s, &ctx);
    exact_run_steps(writer, 100u, &s, &ctx, 8, 0x991001u);
}

static void exact_scenario_manticore_mid_barrage(ColoExactWriter* writer) {
    ColosseumState s;
    ColosseumContext ctx;
    exact_prepare_custom(&s, &ctx, 0x1102u, 17, 16);
    int slot = col_spawn_npc_at(&s, COLO_MANTICORE, 16, 19);
    ColoManticoreState* mc = colo_npc_manticore(&s.npcs[slot]);
    mc->cycle_step = 1;
    mc->orb_style[0] = ATTACK_STYLE_MAGIC;
    mc->orb_style[1] = ATTACK_STYLE_RANGED;
    mc->orb_style[2] = ATTACK_STYLE_MELEE;
    s.npcs[slot].attack_timer = 0;
    exact_refresh_geometry(&s, &ctx);
    exact_run_idle_steps(writer, 101u, &s, &ctx, 5);
}

static void exact_scenario_javelin_boundary(ColoExactWriter* writer) {
    ColosseumState s;
    ColosseumContext ctx;
    exact_prepare_custom(&s, &ctx, 0x1103u, 16, 16);
    int slot = col_spawn_npc_at(&s, COLO_JAVELIN_COLOSSUS, 18, 18);
    s.npcs[slot].attack_timer = 0;
    colo_npc_javelin(&s.npcs[slot])->attack_count = 4;
    exact_refresh_geometry(&s, &ctx);
    exact_run_idle_steps(writer, 102u, &s, &ctx, 6);
}

static void exact_scenario_warband_phase(ColoExactWriter* writer) {
    ColosseumState s;
    ColosseumContext ctx;
    exact_prepare_custom(&s, &ctx, 0x1104u, 17, 16);
    int slot = col_spawn_npc_at(&s, COLO_FREMENNIK_BERSERKER, 17, 17);
    colo_npc_warband(&s.npcs[slot])->formation_dir = COLO_WARBAND_FORM_NORTH;
    s.warband_cycle_anchor = s.tick;
    exact_refresh_geometry(&s, &ctx);
    exact_run_idle_steps(writer, 103u, &s, &ctx, 8);
}

static void exact_scenario_red_flag_minotaur(ColoExactWriter* writer) {
    ColosseumState s;
    ColosseumContext ctx;
    exact_prepare_custom(&s, &ctx, 0x1105u, 7, 9);
    s.modifiers.active_mask |= 1u << COLO_MOD_RED_FLAG;
    s.modifiers.tier[COLO_MOD_RED_FLAG] = 1;
    col_spawn_npc_at(&s, COLO_MINOTAUR, 12, 9);
    exact_refresh_geometry(&s, &ctx);
    exact_run_idle_steps(writer, 104u, &s, &ctx, 10);
}

static void exact_scenario_frozen_npc(ColoExactWriter* writer) {
    ColosseumState s;
    ColosseumContext ctx;
    exact_prepare_custom(&s, &ctx, 0x1106u, 17, 16);
    int slot = col_spawn_npc_at(&s, COLO_SERPENT_SHAMAN, 20, 16);
    s.npcs[slot].attack_timer = 0;
    s.npcs[slot].frozen_ticks = 3;
    exact_refresh_geometry(&s, &ctx);
    exact_run_idle_steps(writer, 105u, &s, &ctx, 6);
}

static void exact_scenario_stunned_npc(ColoExactWriter* writer) {
    ColosseumState s;
    ColosseumContext ctx;
    exact_prepare_custom(&s, &ctx, 0x1107u, 17, 16);
    int slot = col_spawn_npc_at(&s, COLO_SHOCKWAVE_COLOSSUS, 20, 16);
    s.npcs[slot].attack_timer = 0;
    s.npcs[slot].stun_timer = 3;
    exact_refresh_geometry(&s, &ctx);
    exact_run_idle_steps(writer, 106u, &s, &ctx, 6);
}

static void exact_scenario_perimeter_los(ColoExactWriter* writer) {
    ColosseumState s;
    ColosseumContext ctx;
    exact_prepare_custom(&s, &ctx, 0x1108u, 0, 15);
    int edge = col_spawn_npc_at(&s, COLO_SERPENT_SHAMAN, COLO_ARENA_MIN_X, 15);
    s.npcs[edge].attack_timer = 0;
    int sol = col_spawn_npc_at(&s, COLO_SOL_HEREDIT, COLO_SOL_SPAWN_X, COLO_SOL_SPAWN_Y);
    s.wave = COLO_WAVE_BOSS;
    s.sol.started = 1;
    s.sol.boss_idx = sol;
    s.sol.boss_arena_min_x = COLO_BOSS_ARENA_MIN_X;
    s.sol.boss_arena_min_y = COLO_BOSS_ARENA_MIN_Y;
    s.sol.boss_arena_max_x = COLO_BOSS_ARENA_MAX_X;
    s.sol.boss_arena_max_y = COLO_BOSS_ARENA_MAX_Y;
    s.sol.phase = 1;
    s.sol.crystal_count = 1;
    s.sol.crystals[0].active = 1;
    s.sol.crystals[0].edge = COLO_SOL_EDGE_NORTH;
    s.sol.crystals[0].x = 16;
    s.sol.crystals[0].y = s.sol.boss_arena_max_y - 1;
    s.sol.crystals[0].dir = 1;
    s.sol.crystals[0].move_timer = COLO_SOL_CRYSTAL_MOVE_TICKS;
    s.sol.crystals[0].firing_freeze = COLO_SOL_LASER_FREEZE;
    s.sol.laser_cooldown = COLO_SOL_CRYSTAL_COOLDOWN_MIN;
    s.player.x = 16;
    s.player.y = 14;
    exact_refresh_geometry(&s, &ctx);
    exact_run_idle_steps(writer, 107u, &s, &ctx, 5);
}

static void exact_generate_fixture(const char* path) {
    ColoExactWriter writer;
    exact_writer_open(&writer, path);

    exact_scenario_wave_rollout(&writer, 1u, 0, 0xC010001u, 28);
    exact_scenario_wave_rollout(&writer, 4u, 3, 0xC010004u, 32);
    exact_scenario_wave_rollout(&writer, 8u, 7, 0xC010008u, 34);
    exact_scenario_wave_rollout(&writer, 12u, 11, 0xC010012u, 34);
    exact_scenario_pillar_safespot(&writer);
    exact_scenario_manticore_mid_barrage(&writer);
    exact_scenario_javelin_boundary(&writer);
    exact_scenario_warband_phase(&writer);
    exact_scenario_red_flag_minotaur(&writer);
    exact_scenario_frozen_npc(&writer);
    exact_scenario_stunned_npc(&writer);
    exact_scenario_perimeter_los(&writer);

    exact_writer_close(&writer);
}

static void exact_read_all(FILE* file, void* data, size_t size) {
    if (fread(data, 1, size, file) != size) {
        fprintf(stderr, "truncated colosseum exact fixture\n");
        abort();
    }
}

static int exact_compare_files(const char* expected_path, const char* actual_path) {
    FILE* expected = fopen(expected_path, "rb");
    if (!expected) {
        perror("open expected colosseum exact fixture");
        abort();
    }
    FILE* actual = fopen(actual_path, "rb");
    if (!actual) {
        perror("open actual colosseum exact fixture");
        abort();
    }

    ColoExactFileHeader expected_file;
    ColoExactFileHeader actual_file;
    exact_read_all(expected, &expected_file, sizeof(expected_file));
    exact_read_all(actual, &actual_file, sizeof(actual_file));
    if (memcmp(&expected_file, &actual_file, sizeof(expected_file)) != 0) {
        printf("colosseum exact mismatch: file header\n");
        fclose(expected);
        fclose(actual);
        return 1;
    }

    for (uint32_t index = 0; index < expected_file.record_count; index++) {
        ColoExactRecordHeader expected_record;
        ColoExactRecordHeader actual_record;
        float expected_obs[COLO_NUM_OBS];
        float actual_obs[COLO_NUM_OBS];
        float expected_mask[COLO_ACTION_MASK_SIZE];
        float actual_mask[COLO_ACTION_MASK_SIZE];
        ColosseumState expected_state;
        ColosseumState actual_state;
        exact_read_all(expected, &expected_record, sizeof(expected_record));
        exact_read_all(actual, &actual_record, sizeof(actual_record));
        exact_read_all(expected, expected_obs, sizeof(expected_obs));
        exact_read_all(actual, actual_obs, sizeof(actual_obs));
        exact_read_all(expected, expected_mask, sizeof(expected_mask));
        exact_read_all(actual, actual_mask, sizeof(actual_mask));
        exact_read_all(expected, &expected_state, sizeof(expected_state));
        exact_read_all(actual, &actual_state, sizeof(actual_state));

        if (exact_hash_bytes(&expected_state, sizeof(expected_state)) !=
                    expected_record.state_hash ||
                exact_hash_bytes(&actual_state, sizeof(actual_state)) !=
                    actual_record.state_hash) {
            printf("colosseum exact fixture state hash mismatch at record %u\n", index);
            fclose(expected);
            fclose(actual);
            return 1;
        }
        exact_zero_serialized_route_storage(&expected_state);
        exact_zero_serialized_route_storage(&actual_state);
        expected_record.state_hash =
            exact_hash_bytes(&expected_state, sizeof(expected_state));
        actual_record.state_hash =
            exact_hash_bytes(&actual_state, sizeof(actual_state));
        if (memcmp(&expected_record, &actual_record, sizeof(expected_record)) != 0 ||
                memcmp(expected_obs, actual_obs, sizeof(expected_obs)) != 0 ||
                memcmp(expected_mask, actual_mask, sizeof(expected_mask)) != 0 ||
                memcmp(&expected_state, &actual_state, sizeof(expected_state)) != 0) {
            printf(
                "colosseum exact mismatch at record %u scenario %u step %u\n",
                index,
                expected_record.scenario_id,
                expected_record.step_index);
            printf(
                "expected player=(%d,%d) dest=(%d,%d) target=%d actual player=(%d,%d) dest=(%d,%d) target=%d\n",
                expected_state.player.x,
                expected_state.player.y,
                expected_state.player.dest_x,
                expected_state.player.dest_y,
                expected_state.player.interaction.target_slot,
                actual_state.player.x,
                actual_state.player.y,
                actual_state.player.dest_x,
                actual_state.player.dest_y,
                actual_state.player.interaction.target_slot);
            const uint8_t* expected_bytes = (const uint8_t*)&expected_state;
            const uint8_t* actual_bytes = (const uint8_t*)&actual_state;
            for (size_t byte = 0; byte < sizeof(expected_state); byte++) {
                if (expected_bytes[byte] == actual_bytes[byte]) continue;
                printf(
                    "first state byte mismatch offset=%zu expected=%u actual=%u\n",
                    byte,
                    (unsigned)expected_bytes[byte],
                    (unsigned)actual_bytes[byte]);
                break;
            }
            for (int obs = 0; obs < COLO_NUM_OBS; obs++) {
                if (expected_obs[obs] == actual_obs[obs]) continue;
                printf(
                    "first obs mismatch index=%d expected=%g actual=%g\n",
                    obs,
                    (double)expected_obs[obs],
                    (double)actual_obs[obs]);
                break;
            }
            for (int mask = 0; mask < COLO_ACTION_MASK_SIZE; mask++) {
                if (expected_mask[mask] == actual_mask[mask]) continue;
                printf(
                    "first mask mismatch index=%d expected=%g actual=%g\n",
                    mask,
                    (double)expected_mask[mask],
                    (double)actual_mask[mask]);
                break;
            }
            fclose(expected);
            fclose(actual);
            return 1;
        }
    }

    int expected_tail = fgetc(expected);
    int actual_tail = fgetc(actual);
    fclose(expected);
    fclose(actual);
    if (expected_tail != EOF || actual_tail != EOF) {
        printf("colosseum exact mismatch: trailing data\n");
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
#ifdef OSRS_INTERACTION_SERIALIZED_ROUTE_BYTES
    if (argc == 2 && strcmp(argv[1], "--attack-route-selftest") == 0) {
        col_static_los_endpoint_semantics_selftest();
        exact_attack_route_property_selftest();
        return 0;
    }
#endif
    if (argc != 3 ||
            (strcmp(argv[1], "--write-golden") != 0 &&
             strcmp(argv[1], "--compare") != 0)) {
        fprintf(stderr,
            "usage: %s --attack-route-selftest | --write-golden DIR | --compare DIR\n",
            argv[0]);
        return 2;
    }

    col_static_los_table_selftest();
    col_static_los_endpoint_semantics_selftest();
    col_static_footprint_table_selftest();
    col_step_out_forecast_landing_selftest();
#ifdef OSRS_INTERACTION_SERIALIZED_ROUTE_BYTES
    exact_attack_route_property_selftest();
#endif

    char fixture_path[1024];
    char current_path[1024];
    exact_mkdir_if_needed(argv[2]);
    exact_readable_path(
        fixture_path, sizeof(fixture_path), argv[2],
        "colosseum_forecast_exact.bin");

    if (strcmp(argv[1], "--write-golden") == 0) {
        exact_generate_fixture(fixture_path);
        printf("colosseum exact golden wrote %s\n", fixture_path);
        return 0;
    }

    exact_readable_path(
        current_path, sizeof(current_path), argv[2],
        "colosseum_forecast_exact.current.bin");
    exact_generate_fixture(current_path);
    int failed = exact_compare_files(fixture_path, current_path);
    if (failed) return 1;
    printf("colosseum exact golden compare PASS: %s\n", fixture_path);
    return 0;
}
