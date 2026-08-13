#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "ocean/osrs/encounters/encounter_colosseum.h"

static void col_step_out_forecast_landing_selftest_one_state(ColosseumState* s) {
    for (int x = COLO_ARENA_MIN_X - 2; x <= COLO_ARENA_MAX_X + 2; x++) {
        for (int y = COLO_ARENA_MIN_Y - 2; y <= COLO_ARENA_MAX_Y + 2; y++) {
            s->player.x = x;
            s->player.y = y;
            for (int action_idx = 0; action_idx < ENCOUNTER_MOVE_ACTIONS; action_idx++) {
                int valid = col_step_out_forecast_action_valid(s, action_idx);
                Player moved = s->player;
                if (action_idx != 0) {
                    encounter_move_to_target(
                        &moved,
                        ENCOUNTER_MOVE_TARGET_DX[action_idx],
                        ENCOUNTER_MOVE_TARGET_DY[action_idx],
                        col_player_walkable,
                        (void*)s);
                }
                ColoForecastLanding landing =
                    col_step_out_forecast_action_landing_ctx(s, action_idx);
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
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 0x51A7u);
    memset(s.npcs, 0, sizeof(s.npcs));
    memset(s.npc_collision_flags, 0, sizeof(s.npc_collision_flags));
    col_rebuild_player_collision_flags(&s);
    col_step_out_forecast_landing_selftest_one_state(&s);

    col_spawn_npc_at(&s, COLO_SERPENT_SHAMAN, 12, 9);
    col_spawn_npc_at(&s, COLO_JAVELIN_COLOSSUS, 18, 18);
    col_rebuild_player_collision_flags(&s);
    col_step_out_forecast_landing_selftest_one_state(&s);

    s.wave = COLO_WAVE_BOSS;
    s.sol.started = 1;
    s.sol.boss_arena_min_x = COLO_BOSS_ARENA_MIN_X;
    s.sol.boss_arena_min_y = COLO_BOSS_ARENA_MIN_Y;
    s.sol.boss_arena_max_x = COLO_BOSS_ARENA_MAX_X;
    s.sol.boss_arena_max_y = COLO_BOSS_ARENA_MAX_Y;
    col_step_out_forecast_landing_selftest_one_state(&s);

    printf("colosseum landing helper selftest PASS: %d actions across 3 states\n",
        ENCOUNTER_MOVE_ACTIONS);
}

#define col_init_context_typed(ctx_ptr) do { \
    col_init_context_typed(ctx_ptr); \
    (ctx_ptr)->config.late_start_state_mode = 0; \
} while (0)

#define EXACT_MAGIC "COLOEXACTv1"
#define EXACT_VERSION 2u
#define EXACT_CHUNK_BYTES 65536

typedef struct {
    char magic[16];
    uint32_t version;
    uint32_t state_size;
    uint32_t forecast_size;
    uint32_t forecast_obs_size;
    uint32_t obs_size;
    uint32_t action_mask_size;
    uint32_t action_features;
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
    uint32_t forecast_size;
    uint64_t state_hash;
    uint64_t forecast_hash;
    uint64_t forecast_obs_hash;
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
    header.forecast_size = (uint32_t)sizeof(ColoStepOutForecast);
    header.forecast_obs_size = COLO_STEP_OUT_FORECAST_OBS_SIZE;
    header.obs_size = COLO_NUM_OBS;
    header.action_mask_size = COLO_ACTION_MASK_SIZE;
    header.action_features = COLO_STEP_OUT_FORECAST_ACTION_FEATURES;
    exact_write_all(writer->file, &header, sizeof(header));
}

static void exact_writer_close(ColoExactWriter* writer) {
    ColoExactFileHeader header = {0};
    memcpy(header.magic, EXACT_MAGIC, sizeof(EXACT_MAGIC));
    header.version = EXACT_VERSION;
    header.state_size = (uint32_t)sizeof(ColosseumState);
    header.forecast_size = (uint32_t)sizeof(ColoStepOutForecast);
    header.forecast_obs_size = COLO_STEP_OUT_FORECAST_OBS_SIZE;
    header.obs_size = COLO_NUM_OBS;
    header.action_mask_size = COLO_ACTION_MASK_SIZE;
    header.action_features = COLO_STEP_OUT_FORECAST_ACTION_FEATURES;
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

static void exact_capture(
    ColoExactWriter* writer,
    uint32_t scenario_id,
    uint32_t step_index,
    ColosseumState* s,
    ColosseumContext* ctx
) {
    ColoStepOutForecast forecast;
    float forecast_obs[COLO_STEP_OUT_FORECAST_OBS_SIZE];
    float obs[COLO_NUM_OBS];
    float action_mask[COLO_ACTION_MASK_SIZE];

    ColoForecastObsSummary summaries[ENCOUNTER_MOVE_ACTIONS];
    col_build_step_out_forecast_horizon_mode_summary(
        s, &forecast, summaries, COLO_STEP_OUT_FORECAST_HORIZON,
        ctx->config.forecast_run_tile_mode);
    int fi = col_write_step_out_forecast_obs_summary(
        &forecast, summaries, COLO_STEP_OUT_FORECAST_HORIZON, forecast_obs, 0);
    assert(fi == COLO_STEP_OUT_FORECAST_OBS_SIZE);
    col_write_obs_ctx((EncounterState*)s, (EncounterContext*)ctx, obs);
    col_write_mask_ctx((EncounterState*)s, (EncounterContext*)ctx, action_mask);

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
    record.forecast_size = (uint32_t)sizeof(forecast);
    record.state_hash = exact_hash_bytes(s, sizeof(*s));
    record.forecast_hash = exact_hash_bytes(&forecast, sizeof(forecast));
    record.forecast_obs_hash = exact_hash_bytes(forecast_obs, sizeof(forecast_obs));
    record.obs_hash = exact_hash_bytes(obs, sizeof(obs));
    record.action_mask_hash = exact_hash_bytes(action_mask, sizeof(action_mask));
    record.reward = col_get_reward_ctx((EncounterState*)s, (EncounterContext*)ctx);

    exact_write_all(writer->file, &record, sizeof(record));
    exact_write_all(writer->file, &forecast, sizeof(forecast));
    exact_write_all(writer->file, forecast_obs, sizeof(forecast_obs));
    exact_write_all(writer->file, obs, sizeof(obs));
    exact_write_all(writer->file, action_mask, sizeof(action_mask));
    exact_write_all(writer->file, s, sizeof(*s));
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
    ctx->config.step_out_forecast_obs_enabled = 1;
    memset(s, 0, sizeof(*s));
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

    uint8_t expected_buf[EXACT_CHUNK_BYTES];
    uint8_t actual_buf[EXACT_CHUNK_BYTES];
    uint64_t offset = 0;
    for (;;) {
        size_t ne = fread(expected_buf, 1, sizeof(expected_buf), expected);
        size_t na = fread(actual_buf, 1, sizeof(actual_buf), actual);
        if (ne != na) {
            printf("colosseum exact mismatch: size differs at byte %llu\n",
                (unsigned long long)offset);
            fclose(expected);
            fclose(actual);
            return 1;
        }
        if (ne == 0) break;
        if (memcmp(expected_buf, actual_buf, ne) != 0) {
            for (size_t i = 0; i < ne; i++) {
                if (expected_buf[i] == actual_buf[i]) continue;
                printf("colosseum exact mismatch at byte %llu: expected %u got %u\n",
                    (unsigned long long)(offset + i),
                    (unsigned)expected_buf[i],
                    (unsigned)actual_buf[i]);
                fclose(expected);
                fclose(actual);
                return 1;
            }
        }
        offset += (uint64_t)ne;
    }

    fclose(expected);
    fclose(actual);
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 3 ||
            (strcmp(argv[1], "--write-golden") != 0 &&
             strcmp(argv[1], "--compare") != 0)) {
        fprintf(stderr,
            "usage: %s --write-golden DIR | --compare DIR\n", argv[0]);
        return 2;
    }

    col_static_los_table_selftest();
    col_static_footprint_table_selftest();
    col_step_out_forecast_landing_selftest();

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
