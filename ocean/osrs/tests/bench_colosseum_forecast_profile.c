#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

#include "ocean/osrs/encounters/encounter_colosseum.h"

#define BENCH_CORPUS_STATES 512

static volatile double bench_sink = 0.0;

typedef enum {
    COLO_BENCH_SETUP_CLEAR = 0,
    COLO_BENCH_SETUP_COLLECT,
    COLO_BENCH_SETUP_BASE_COPY,
    COLO_BENCH_SETUP_FLAGS,
    COLO_BENCH_SETUP_PRECOMP,
    COLO_BENCH_ACTION_PREP,
    COLO_BENCH_RESTORE_NPCS,
    COLO_BENCH_RESTORE_FLAGS,
    COLO_BENCH_TICK_ADMIN,
    COLO_BENCH_MOVEMENT,
    COLO_BENCH_ATTACK_AI,
    COLO_BENCH_FINALIZE,
    COLO_BENCH_OBS_SCORE,
    COLO_BENCH_COUNT,
} ColoBenchBucket;

typedef struct {
    uint64_t ns[COLO_BENCH_COUNT];
    uint64_t total_ns;
    uint64_t fanout_ns;
    uint64_t forecasts;
    uint64_t actions_seen;
    uint64_t valid_actions;
    uint64_t rollout_actions;
    uint64_t invalid_or_duplicate_actions;
    uint64_t static_threat_actions;
    uint64_t rollout_ticks;
    uint64_t npc_tick_visits;
    uint64_t slot_count_sum;
    uint64_t forecast_obs_checksum;
} ColoForecastBenchProfile;

typedef struct {
    ColosseumState states[BENCH_CORPUS_STATES];
    int count;
} ColoBenchCorpus;

static const char* const COLO_BENCH_BUCKET_NAMES[COLO_BENCH_COUNT] = {
    "setup_clear",
    "setup_collect_slots",
    "setup_base_npc_copy",
    "setup_base_flags",
    "setup_precompute",
    "action_landing_dedup",
    "restore_npc_slots",
    "restore_npc_flags",
    "rollout_tick_admin",
    "rollout_movement",
    "rollout_attack_ai",
    "action_finalize",
    "obs_score_encode",
};

static uint64_t now_ns(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    uint64_t ticks = mach_absolute_time();
    return ticks * (uint64_t)timebase.numer / (uint64_t)timebase.denom;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static void bench_add_ns(ColoForecastBenchProfile* profile, int bucket, uint64_t start_ns) {
    profile->ns[bucket] += now_ns() - start_ns;
}

static uint64_t bench_hash_bytes(uint64_t h, const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < size; i++) {
        h ^= bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t bench_splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void bench_trace_actions(
    ColosseumState* s,
    uint64_t* rng,
    int actions[COLO_NUM_ACTION_HEADS]
) {
    for (int head = 0; head < COLO_NUM_ACTION_HEADS; head++) {
        actions[head] = (int)(bench_splitmix64(rng) % (uint64_t)COLO_ACTION_DIMS[head]);
    }
    if (s->modifiers.draft_pending) {
        actions[COLO_HEAD_PRIMARY] = 0;
        actions[COLO_HEAD_MODIFIER_SELECT] = 1 +
            (int)(bench_splitmix64(rng) % COLO_MODIFIER_DRAFT_OPTIONS);
    }
}

static void bench_init_context(ColosseumContext* ctx, int start_wave) {
    col_init_context_typed(ctx);
    ctx->config.start_wave = start_wave;
    ctx->config.step_out_forecast_obs_enabled = 1;
    ctx->config.forecast_horizon = COLO_STEP_OUT_FORECAST_HORIZON;
    ctx->config.forecast_run_tile_mode = COLO_FORECAST_RUN_TILE_FULL;
}

static void bench_capture_state(
    ColoBenchCorpus* corpus,
    const ColosseumState* s
) {
    if (corpus->count >= BENCH_CORPUS_STATES) return;
    corpus->states[corpus->count++] = *s;
}

static void bench_add_wave_rollout(
    ColoBenchCorpus* corpus,
    int start_wave,
    uint32_t seed,
    uint64_t action_seed
) {
    ColosseumContext ctx;
    ColosseumState s;
    int actions[COLO_NUM_ACTION_HEADS];
    uint64_t rng = action_seed;
    bench_init_context(&ctx, start_wave);
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, seed);
    for (int step = 0; step < 160 && corpus->count < BENCH_CORPUS_STATES; step++) {
        bench_capture_state(corpus, &s);
        bench_trace_actions(&s, &rng, actions);
        col_step_ctx((EncounterState*)&s, (EncounterContext*)&ctx, actions);
        if (s.episode_over) {
            col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, seed + (uint32_t)step + 1u);
        }
    }
}

static void bench_prepare_custom(
    ColosseumState* s,
    ColosseumContext* ctx,
    uint32_t seed,
    int player_x,
    int player_y
) {
    bench_init_context(ctx, 1);
    memset(s, 0, sizeof(*s));
    col_reset_ctx((EncounterState*)s, (EncounterContext*)ctx, seed);
    memset(s->npcs, 0, sizeof(s->npcs));
    memset(s->npc_collision_flags, 0, sizeof(s->npc_collision_flags));
    memset(s->totems, 0, sizeof(s->totems));
    memset(s->bees, 0, sizeof(s->bees));
    s->modifiers.draft_pending = 0;
    s->modifiers.draft_gates_spawn = 0;
    s->wave_ready_delay = 0;
    s->wave_spawn_delay = 0;
    s->reinforcement_timer = COLO_REINFORCE_FIRED;
    s->warband_cycle_anchor = s->tick;
    s->player.x = player_x;
    s->player.y = player_y;
    col_rebuild_player_collision_flags(s);
    col_refresh_current_obs_slots_ctx(s, ctx);
}

static void bench_add_custom_dense(ColoBenchCorpus* corpus) {
    ColosseumState s;
    ColosseumContext ctx;
    bench_prepare_custom(&s, &ctx, 0xC010D375u, 17, 16);
    int slot = col_spawn_npc_at(&s, COLO_MANTICORE, 16, 20);
    ColoManticoreState* mc = colo_npc_manticore(&s.npcs[slot]);
    mc->cycle_step = 1;
    mc->orb_style[0] = ATTACK_STYLE_MAGIC;
    mc->orb_style[1] = ATTACK_STYLE_RANGED;
    mc->orb_style[2] = ATTACK_STYLE_MELEE;
    s.npcs[slot].attack_timer = 0;
    slot = col_spawn_npc_at(&s, COLO_JAVELIN_COLOSSUS, 21, 17);
    s.npcs[slot].attack_timer = 0;
    colo_npc_javelin(&s.npcs[slot])->attack_count = 4;
    slot = col_spawn_npc_at(&s, COLO_SERPENT_SHAMAN, 12, 16);
    s.npcs[slot].attack_timer = 0;
    slot = col_spawn_npc_at(&s, COLO_SHOCKWAVE_COLOSSUS, 19, 12);
    s.npcs[slot].attack_timer = 0;
    slot = col_spawn_npc_at(&s, COLO_MINOTAUR, 14, 16);
    s.npcs[slot].attack_timer = 0;
    slot = col_spawn_npc_at(&s, COLO_JAGUAR_WARRIOR, 17, 19);
    s.npcs[slot].attack_timer = 0;
    slot = col_spawn_npc_at(&s, COLO_FREMENNIK_BERSERKER, 17, 17);
    colo_npc_warband(&s.npcs[slot])->formation_dir = COLO_WARBAND_FORM_NORTH;
    slot = col_spawn_npc_at(&s, COLO_FREMENNIK_ARCHER, 18, 17);
    colo_npc_warband(&s.npcs[slot])->formation_dir = COLO_WARBAND_FORM_EAST;
    col_rebuild_player_collision_flags(&s);
    col_refresh_current_obs_slots_ctx(&s, &ctx);
    bench_capture_state(corpus, &s);
}

static void bench_build_corpus(ColoBenchCorpus* corpus) {
    memset(corpus, 0, sizeof(*corpus));
    bench_add_custom_dense(corpus);
    bench_add_wave_rollout(corpus, 0, 0xC010001u, 0xA1001u);
    bench_add_wave_rollout(corpus, 3, 0xC010004u, 0xA1004u);
    bench_add_wave_rollout(corpus, 7, 0xC010008u, 0xA1008u);
    bench_add_wave_rollout(corpus, 11, 0xC010012u, 0xA1012u);
    if (corpus->count <= 0) abort();
}

static void bench_encode_step_out_forecast_obs(
    const ColoStepOutForecast* forecast,
    int horizon,
    float out[COLO_STEP_OUT_FORECAST_OBS_SIZE]
) {
    int i = 0;
    for (int action_idx = 0; action_idx < ENCOUNTER_MOVE_ACTIONS; action_idx++) {
        const ColoStepOutForecastAction* action = &forecast->actions[action_idx];
        int first_attack_tick = 0;
        int first_style_mask = 0;
        int max_hit = 0;
        int ranged_magic_same_tick = 0;
        for (int tick_idx = 0; tick_idx < horizon; tick_idx++) {
            const ColoStepOutForecastTick* tick = &action->ticks[tick_idx];
            int style_mask = col_step_out_forecast_tick_style_mask(tick);
            if (first_attack_tick == 0 &&
                    col_step_out_forecast_tick_has_event(tick)) {
                first_attack_tick = tick_idx + 1;
                first_style_mask = style_mask;
            }
            if (tick->max_hit > max_hit) max_hit = tick->max_hit;
            if (tick->ranged_count > 0 && tick->magic_count > 0)
                ranged_magic_same_tick = 1;
        }
        out[i++] = action->valid ? 1.0f : 0.0f;
        out[i++] = (float)first_attack_tick / (float)horizon;
        out[i++] = (float)first_style_mask / 7.0f;
        out[i++] = (float)max_hit / 150.0f;
        out[i++] = action->same_tick_mixed_style_conflict ? 1.0f : 0.0f;
        out[i++] = ranged_magic_same_tick ? 1.0f : 0.0f;
        out[i++] = action->ranged_magic_offtick_opportunity ? 1.0f : 0.0f;
        out[i++] = action->melee_fallback_exposure ? 1.0f : 0.0f;
    }
    assert(i == COLO_STEP_OUT_FORECAST_OBS_SIZE);
}

static void bench_profiled_tick(
    const ColosseumState* s,
    ColoForecastNpcLocal npcs[COLO_MAX_NPCS],
    uint8_t npc_flags[COLO_ARENA_WIDTH][COLO_ARENA_HEIGHT],
    ColoStepOutForecastAction* action,
    ColoForecastObsSummary* summary,
    int tick_idx,
    const int slots[COLO_MAX_NPCS],
    int slot_count,
    const ColoForecastPrecomp* pre,
    ColoForecastBenchProfile* profile
) {
    for (int slot_idx = 0; slot_idx < slot_count; slot_idx++) {
        int i = slots[slot_idx];
        ColoForecastNpcLocal* npc = &npcs[i];
        uint64_t start = now_ns();
        if (npc->stun_timer > 0) npc->stun_timer--;
        if (npc->frozen_ticks > 0) npc->frozen_ticks--;
        if (npc->type == COLO_SOL_HEREDIT) {
            if (npc->sol_immobile_ticks > 0) npc->sol_immobile_ticks--;
            if (npc->sol_attack_delay > 0) npc->sol_attack_delay--;
        }
        bench_add_ns(profile, COLO_BENCH_TICK_ADMIN, start);

        start = now_ns();
        ColoForecastMoveResult move_result =
            col_forecast_local_move_npc(
                s, npcs, npc_flags, i, action, pre->sol_clamp_active);
        bench_add_ns(profile, COLO_BENCH_MOVEMENT, start);

        start = now_ns();
        col_forecast_local_attack_npc(
            s, npcs, slots, slot_count, i, action, summary, tick_idx,
            &move_result, pre);
        bench_add_ns(profile, COLO_BENCH_ATTACK_AI, start);
        profile->npc_tick_visits++;
    }
}

static void bench_profiled_forecast_build(
    const ColosseumState* s,
    ColoStepOutForecast* out,
    ColoForecastObsSummary summaries[ENCOUNTER_MOVE_ACTIONS],
    int horizon,
    int run_tile_mode,
    ColoForecastBenchProfile* profile
) {
    if (horizon < 1) horizon = 1;
    if (horizon > COLO_STEP_OUT_FORECAST_HORIZON)
        horizon = COLO_STEP_OUT_FORECAST_HORIZON;

    uint64_t total_start = now_ns();
    uint64_t start = now_ns();
    memset(out, 0, sizeof(*out));
    if (summaries)
        memset(summaries, 0, sizeof(ColoForecastObsSummary) * ENCOUNTER_MOVE_ACTIONS);
    bench_add_ns(profile, COLO_BENCH_SETUP_CLEAR, start);

    int forecast_slots[COLO_MAX_NPCS];
    start = now_ns();
    int forecast_slot_count = col_collect_step_out_forecast_slots(s, forecast_slots);
    bench_add_ns(profile, COLO_BENCH_SETUP_COLLECT, start);
    profile->slot_count_sum += (uint64_t)forecast_slot_count;

    ColoForecastNpcLocal base_npcs[COLO_MAX_NPCS];
    uint8_t base_npc_flags[COLO_ARENA_WIDTH][COLO_ARENA_HEIGHT];
    ColoForecastPrecomp pre = {0};
    if (forecast_slot_count > 0) {
        start = now_ns();
        col_forecast_local_copy_npc_slots(
            s, base_npcs, forecast_slots, forecast_slot_count);
        bench_add_ns(profile, COLO_BENCH_SETUP_BASE_COPY, start);

        start = now_ns();
        col_forecast_local_rebuild_npc_flags(
            base_npcs, base_npc_flags, forecast_slots, forecast_slot_count);
        bench_add_ns(profile, COLO_BENCH_SETUP_FLAGS, start);

        start = now_ns();
        pre = col_forecast_precompute(
            s, base_npcs, forecast_slots, forecast_slot_count);
        bench_add_ns(profile, COLO_BENCH_SETUP_PRECOMP, start);
    }

    uint64_t fanout_start = now_ns();
    for (int action_idx = 0; action_idx < ENCOUNTER_MOVE_ACTIONS; action_idx++) {
        profile->actions_seen++;
        start = now_ns();
        int unique_action = col_step_out_forecast_prepare_unique_action_ctx(
            s, out, summaries, action_idx, forecast_slot_count);
        bench_add_ns(profile, COLO_BENCH_ACTION_PREP, start);
        if (!unique_action) {
            profile->invalid_or_duplicate_actions++;
            continue;
        }
        profile->valid_actions++;
        ColoStepOutForecastAction* action = &out->actions[action_idx];
        ColoForecastObsSummary* summary = summaries ? &summaries[action_idx] : NULL;
        if (run_tile_mode == COLO_FORECAST_RUN_TILE_STATIC_THREAT &&
                col_forecast_action_is_run_tile(action_idx)) {
            col_forecast_static_threat_action(
                s, base_npcs, forecast_slots, forecast_slot_count, action, summary);
            profile->static_threat_actions++;
            start = now_ns();
            col_step_out_forecast_finalize_action(action, horizon);
            bench_add_ns(profile, COLO_BENCH_FINALIZE, start);
            continue;
        }

        ColoForecastNpcLocal npcs[COLO_MAX_NPCS];
        uint8_t npc_flags[COLO_ARENA_WIDTH][COLO_ARENA_HEIGHT];
        start = now_ns();
        col_forecast_local_copy_slot_set(
            npcs, base_npcs, forecast_slots, forecast_slot_count);
        bench_add_ns(profile, COLO_BENCH_RESTORE_NPCS, start);

        start = now_ns();
        memcpy(npc_flags, base_npc_flags, sizeof(base_npc_flags));
        bench_add_ns(profile, COLO_BENCH_RESTORE_FLAGS, start);

        profile->rollout_actions++;
        for (int tick_idx = 0; tick_idx < horizon; tick_idx++) {
            bench_profiled_tick(
                s, npcs, npc_flags, action, summary, tick_idx,
                forecast_slots, forecast_slot_count, &pre, profile);
            profile->rollout_ticks++;
        }

        start = now_ns();
        col_step_out_forecast_finalize_action(action, horizon);
        bench_add_ns(profile, COLO_BENCH_FINALIZE, start);
    }
    profile->fanout_ns += now_ns() - fanout_start;
    profile->total_ns += now_ns() - total_start;
    profile->forecasts++;
}

static void bench_profiled_forecast_obs(
    const ColosseumState* s,
    float out[COLO_STEP_OUT_FORECAST_OBS_SIZE],
    ColoForecastBenchProfile* profile
) {
    ColoStepOutForecast forecast;
    ColoForecastObsSummary summaries[ENCOUNTER_MOVE_ACTIONS];
    bench_profiled_forecast_build(
        s, &forecast, summaries, COLO_STEP_OUT_FORECAST_HORIZON,
        COLO_FORECAST_RUN_TILE_FULL, profile);
    uint64_t start = now_ns();
    col_write_step_out_forecast_obs_summary(
        &forecast, summaries, COLO_STEP_OUT_FORECAST_HORIZON, out, 0);
    profile->forecast_obs_checksum = bench_hash_bytes(
        profile->forecast_obs_checksum, out, sizeof(float) * COLO_STEP_OUT_FORECAST_OBS_SIZE);
    bench_add_ns(profile, COLO_BENCH_OBS_SCORE, start);
}

static void bench_raw_forecast_obs(
    const ColosseumState* s,
    float out[COLO_STEP_OUT_FORECAST_OBS_SIZE]
) {
    ColoStepOutForecast forecast;
    ColoForecastObsSummary summaries[ENCOUNTER_MOVE_ACTIONS];
    col_build_step_out_forecast_horizon_mode_summary(
        s, &forecast, summaries, COLO_STEP_OUT_FORECAST_HORIZON,
        COLO_FORECAST_RUN_TILE_FULL);
    col_write_step_out_forecast_obs_summary(
        &forecast, summaries, COLO_STEP_OUT_FORECAST_HORIZON, out, 0);
    bench_sink += out[0];
}

static void bench_check_profiled_exact(const ColoBenchCorpus* corpus) {
    float obs_a[COLO_STEP_OUT_FORECAST_OBS_SIZE];
    float obs_b[COLO_STEP_OUT_FORECAST_OBS_SIZE];
    for (int i = 0; i < corpus->count; i++) {
        ColoStepOutForecast expected;
        ColoStepOutForecast actual;
        ColoForecastBenchProfile profile = {0};
        col_build_step_out_forecast_horizon_mode(
            &corpus->states[i], &expected, COLO_STEP_OUT_FORECAST_HORIZON,
            COLO_FORECAST_RUN_TILE_FULL);
        bench_profiled_forecast_build(
            &corpus->states[i], &actual, NULL, COLO_STEP_OUT_FORECAST_HORIZON,
            COLO_FORECAST_RUN_TILE_FULL, &profile);
        if (memcmp(&expected, &actual, sizeof(expected)) != 0) {
            fprintf(stderr, "profiled forecast mismatch on corpus state %d\n", i);
            abort();
        }
        bench_encode_step_out_forecast_obs(
            &expected, COLO_STEP_OUT_FORECAST_HORIZON, obs_a);
        bench_encode_step_out_forecast_obs(
            &actual, COLO_STEP_OUT_FORECAST_HORIZON, obs_b);
        if (memcmp(obs_a, obs_b, sizeof(obs_a)) != 0) {
            fprintf(stderr, "profiled obs mismatch on corpus state %d\n", i);
            abort();
        }
    }
}

static double bench_raw_forecast_seconds(
    const ColoBenchCorpus* corpus,
    int iters
) {
    float obs[COLO_STEP_OUT_FORECAST_OBS_SIZE];
    uint64_t start = now_ns();
    for (int i = 0; i < iters; i++) {
        bench_raw_forecast_obs(&corpus->states[i % corpus->count], obs);
    }
    return (double)(now_ns() - start) / 1000000000.0;
}

static double bench_state_copy_seconds(
    const ColoBenchCorpus* corpus,
    int iters
) {
    uint64_t start = now_ns();
    for (int i = 0; i < iters; i++) {
        ColosseumState s = corpus->states[i % corpus->count];
        bench_sink += (double)s.player.x;
    }
    return (double)(now_ns() - start) / 1000000000.0;
}

static double bench_env_step_seconds(
    const ColoBenchCorpus* corpus,
    const ColosseumContext* template_ctx,
    int iters
) {
    float obs[COLO_NUM_OBS + COLO_ACTION_MASK_SIZE];
    int actions[COLO_NUM_ACTION_HEADS] = {0};
    uint64_t start = now_ns();
    for (int i = 0; i < iters; i++) {
        ColosseumState s = corpus->states[i % corpus->count];
        ColosseumContext ctx = *template_ctx;
        col_step_ctx((EncounterState*)&s, (EncounterContext*)&ctx, actions);
        col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
        col_write_mask_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs + COLO_NUM_OBS);
        bench_sink += obs[0];
        bench_sink += col_get_reward_ctx((EncounterState*)&s, (EncounterContext*)&ctx);
        bench_sink += col_is_terminal_ctx((EncounterState*)&s, (EncounterContext*)&ctx);
    }
    return (double)(now_ns() - start) / 1000000000.0;
}

static void bench_collect_profile(
    const ColoBenchCorpus* corpus,
    int iters,
    ColoForecastBenchProfile* profile
) {
    float obs[COLO_STEP_OUT_FORECAST_OBS_SIZE];
    memset(profile, 0, sizeof(*profile));
    for (int i = 0; i < iters; i++) {
        bench_profiled_forecast_obs(&corpus->states[i % corpus->count], obs, profile);
    }
}

static uint64_t bench_profile_exclusive_sum(const ColoForecastBenchProfile* profile) {
    uint64_t sum = 0;
    for (int i = 0; i < COLO_BENCH_COUNT; i++) sum += profile->ns[i];
    return sum;
}

static void bench_print_profile(
    const ColoForecastBenchProfile* profile,
    double raw_us_per_forecast,
    double env_us_per_step,
    double env_no_forecast_us_per_step
) {
    double measured_us = (double)profile->total_ns / (double)profile->forecasts / 1000.0;
    double obs_score_us =
        (double)profile->ns[COLO_BENCH_OBS_SCORE] / (double)profile->forecasts / 1000.0;
    uint64_t exclusive_sum = bench_profile_exclusive_sum(profile);
    uint64_t unattributed_ns =
        profile->total_ns + profile->ns[COLO_BENCH_OBS_SCORE] > exclusive_sum
            ? profile->total_ns + profile->ns[COLO_BENCH_OBS_SCORE] - exclusive_sum
            : 0;
    double total_for_pct_us = measured_us + obs_score_us;

    printf("corpus_forecasts=%llu actions=%llu valid=%llu rollout_actions=%llu skipped=%llu avg_slots=%.2f\n",
        (unsigned long long)profile->forecasts,
        (unsigned long long)profile->actions_seen,
        (unsigned long long)profile->valid_actions,
        (unsigned long long)profile->rollout_actions,
        (unsigned long long)profile->invalid_or_duplicate_actions,
        (double)profile->slot_count_sum / (double)profile->forecasts);
    double accounted_us = (double)exclusive_sum / (double)profile->forecasts / 1000.0;
    double forecast_delta_us = env_us_per_step - env_no_forecast_us_per_step;

    printf("rollout_ticks=%llu npc_tick_visits=%llu fanout_us=%.3f measured_forecast_plus_obs_us=%.3f raw_forecast_plus_obs_us=%.3f env_step_us=%.3f env_step_no_forecast_us=%.3f forecast_delta_us=%.3f forecast_pct_of_env_step=%.2f delta_pct_of_env_step=%.2f\n",
        (unsigned long long)profile->rollout_ticks,
        (unsigned long long)profile->npc_tick_visits,
        (double)profile->fanout_ns / (double)profile->forecasts / 1000.0,
        total_for_pct_us,
        raw_us_per_forecast,
        env_us_per_step,
        env_no_forecast_us_per_step,
        forecast_delta_us,
        raw_us_per_forecast * 100.0 / env_us_per_step,
        forecast_delta_us * 100.0 / env_us_per_step);
    printf("internal bucket breakdown, accounted-normalized and raw-scaled:\n");
    for (int i = 0; i < COLO_BENCH_COUNT; i++) {
        double us = (double)profile->ns[i] / (double)profile->forecasts / 1000.0;
        double pct = accounted_us > 0.0 ? us * 100.0 / accounted_us : 0.0;
        printf("  %-24s %9.3f measured_us %9.3f raw_scaled_us %6.2f%%\n",
            COLO_BENCH_BUCKET_NAMES[i], us, raw_us_per_forecast * pct / 100.0, pct);
    }
    printf("timer_unattributed_loop_overhead %.3f us\n",
        (double)unattributed_ns / (double)profile->forecasts / 1000.0);
    printf("forecast_obs_checksum=%llu bench_sink=%.3f\n",
        (unsigned long long)profile->forecast_obs_checksum, bench_sink);
}

int main(void) {
    ColosseumContext ctx;
    ColosseumContext no_forecast_ctx;
    ColoForecastBenchProfile profile;
    ColoBenchCorpus* corpus = calloc(1, sizeof(*corpus));
    if (!corpus) abort();
    bench_init_context(&ctx, 0);
    bench_init_context(&no_forecast_ctx, 0);
    no_forecast_ctx.config.step_out_forecast_obs_enabled = 0;
    bench_build_corpus(corpus);
    bench_check_profiled_exact(corpus);

    int raw_iters = 50000;
    int env_iters = 20000;
    int profiled_iters = 3000;
    double raw_seconds = bench_raw_forecast_seconds(corpus, raw_iters);
    double copy_seconds = bench_state_copy_seconds(corpus, env_iters);
    double env_seconds = bench_env_step_seconds(corpus, &ctx, env_iters);
    double env_no_forecast_seconds =
        bench_env_step_seconds(corpus, &no_forecast_ctx, env_iters);
    bench_collect_profile(corpus, profiled_iters, &profile);

    double raw_us = raw_seconds * 1000000.0 / (double)raw_iters;
    double copy_us = copy_seconds * 1000000.0 / (double)env_iters;
    double env_us = env_seconds * 1000000.0 / (double)env_iters - copy_us;
    double env_no_forecast_us =
        env_no_forecast_seconds * 1000000.0 / (double)env_iters - copy_us;

    printf("sizeof(ColosseumState)=%zu COLO_NUM_OBS=%d COLO_STEP_OUT_FORECAST_OBS_SIZE=%d actions=%d horizon=%d corpus_states=%d\n",
        sizeof(ColosseumState),
        COLO_NUM_OBS,
        COLO_STEP_OUT_FORECAST_OBS_SIZE,
        ENCOUNTER_MOVE_ACTIONS,
        COLO_STEP_OUT_FORECAST_HORIZON,
        corpus->count);
    printf("raw_iters=%d env_iters=%d profiled_iters=%d state_copy_us=%.3f\n",
        raw_iters, env_iters, profiled_iters, copy_us);
    bench_print_profile(&profile, raw_us, env_us, env_no_forecast_us);
    free(corpus);
    return 0;
}
