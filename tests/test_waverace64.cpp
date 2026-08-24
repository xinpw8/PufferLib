#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __linux__
#include <sys/stat.h>
#endif

#include "waverace64.h"

extern "C" void func_8004D30C(uint8_t* rdram, recomp_context* ctx);

#ifndef PUFFER_ENV_UNCLIPPED_REWARDS
#error "Wave Race potential shaping requires unclipped learner rewards"
#endif

#define CONFIG_LAPS_ADDR 0x801CE618u
#define LAP_TARGET_ADDR  0x801CE728u
#define SUNNY_ROUTE_TOTAL 29078.811f

static void wr32(WaveRace64* env, uint32_t va, uint32_t value) {
    wr_wr32(env->machine.rdram, va, value);
}

static uint64_t hash_byte(uint64_t hash, uint8_t value) {
    return (hash ^ value) * UINT64_C(1099511628211);
}

static uint64_t hash_u32(uint64_t hash, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        hash = hash_byte(hash, (uint8_t)(value >> shift));
    }
    return hash;
}

typedef struct ActionDigest {
    uint64_t trajectory;
    uint64_t rdram;
} ActionDigest;

static uint64_t hash_authoritative_state(WaveRace64* env, uint64_t hash) {
    uint32_t physics = wr64_physics_addr(env, 0);
    uint32_t rider = wr64_rider_addr(env, 0);
    hash = hash_u32(hash, wr64_u(env, WR_ADDR_GAMESTATE));
    for (uint32_t offset = WR64_PHYSICS_POS;
            offset <= WR64_PHYSICS_POS + 8; offset += 4) {
        hash = hash_u32(hash, wr64_u(env, physics + offset));
    }
    for (uint32_t offset = WR_PHYSICS_BASIS_0_X;
            offset <= WR_PHYSICS_BASIS_2_Z; offset += 4) {
        hash = hash_u32(hash, wr64_u(env, physics + offset));
    }
    hash = hash_u32(hash,
        wr64_u(env, physics + WR64_PHYSICS_FORWARD_X));
    hash = hash_u32(hash,
        wr64_u(env, physics + WR64_PHYSICS_FORWARD_Z));
    hash = hash_u32(hash, wr64_u(env, physics + WR64_PHYSICS_STATE));
    hash = hash_u32(hash, wr64_u(env, physics + WR64_PHYSICS_STATE_FRAME));
    hash = hash_u32(hash, (uint32_t)wr64_recovery(env));
    hash = hash_u32(hash, wr64_u(env, rider + WR_RIDER_LAP));
    hash = hash_u32(hash, wr64_u(env, rider + WR_RIDER_NODE));
    hash = hash_u32(hash, wr64_u(env, rider + WR64_RIDER_POWER));
    hash = hash_u32(hash, wr64_u(env, rider + WR64_RIDER_MISSES));
    hash = hash_u32(hash, wr64_u(env, rider + WR64_RIDER_LAP_TIME));
    hash = hash_u32(hash, wr64_u(env, rider + WR64_RIDER_TOTAL_TIME));
    hash = hash_u32(hash, wr64_u(env, rider + WR64_RIDER_DQ));
    hash = hash_u32(hash, wr64_u(env, rider + WR64_RIDER_ENDED));
    hash = hash_u32(hash, wr64_u(env, rider + WR64_RIDER_FINISHED));
    return hash;
}

static uint64_t hash_rdram(WaveRace64* env) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < WR_RDRAM_SIZE; i++) {
        hash = hash_byte(hash, env->machine.rdram[i]);
    }
    return hash;
}

static uint64_t hash_wave_grid(WaveRace64* env) {
    size_t offset = WR64_WATER_GRID & UINT32_C(0x1FFFFFFF);
    size_t size = (size_t)WR64_WATER_ROWS * WR64_WATER_COLS * 4u;
    return wr64_hash_bytes(env->machine.rdram + offset, size);
}

static void set_action(WaveRace64* env, int x, int y,
        int a, int b, int r);

typedef struct HostStepDigest {
    uint64_t rdram;
    uint64_t authoritative;
    uint64_t machine_ticks;
    State state;
    uint32_t observations[WR64_OBS_SIZE];
    uint32_t reward;
    uint32_t terminal;
} HostStepDigest;

static HostStepDigest current_host_step_digest(WaveRace64* env) {
    HostStepDigest digest = {};
    digest.rdram = hash_rdram(env);
    digest.authoritative = hash_authoritative_state(
        env, UINT64_C(14695981039346656037));
    digest.machine_ticks = env->machine.ticks;
    digest.state = env->state;
    memcpy(digest.observations, env->agents[0].observations,
        sizeof(digest.observations));
    memcpy(&digest.reward, env->agents[0].rewards, sizeof(digest.reward));
    memcpy(&digest.terminal, env->agents[0].terminals,
        sizeof(digest.terminal));
    return digest;
}

typedef struct WaveResetDigest {
    uint64_t rdram;
    uint64_t grid;
    uint64_t authoritative;
    uint32_t active_variant;
    uint32_t observations[12];
    uint32_t vertical_origin;
} WaveResetDigest;

static WaveResetDigest current_wave_digest(WaveRace64* env) {
    assert(wr64_reset_contract_valid(env, env->curriculum_laps));
    WaveResetDigest digest = {};
    digest.rdram = hash_rdram(env);
    digest.grid = hash_wave_grid(env);
    digest.authoritative = hash_authoritative_state(
        env, UINT64_C(14695981039346656037));
    digest.active_variant = env->active_wave_variant;
    memcpy(digest.observations, env->agents[0].observations + 43,
        sizeof(digest.observations));
    memcpy(&digest.vertical_origin, &env->vertical_origin,
        sizeof(digest.vertical_origin));
    return digest;
}

static WaveResetDigest wave_reset_digest(WaveRace64* env) {
    puf_reset(env);
    return current_wave_digest(env);
}

static int wave_reset_digest_equal(
        const WaveResetDigest* a, const WaveResetDigest* b) {
    return a->rdram == b->rdram
        && a->grid == b->grid
        && a->authoritative == b->authoritative
        && a->active_variant == b->active_variant
        && a->vertical_origin == b->vertical_origin
        && memcmp(a->observations, b->observations,
            sizeof(a->observations)) == 0;
}

#if defined(__aarch64__)
static uint64_t test_read_fpcr() {
    uint64_t value;
    __asm__ volatile("mrs %0, fpcr" : "=r"(value));
    return value;
}

static void test_write_fpcr(uint64_t value) {
    __asm__ volatile("msr fpcr, %0\nisb" : : "r"(value) : "memory");
}
#endif

static void test_wave_fp_guard(WaveRace64* env, Dict* kwargs) {
#if defined(__aarch64__)
    const uint64_t fpcr_fz = UINT64_C(1) << 24;
    const uint64_t fpcr_fz16 = UINT64_C(1) << 19;
    const uint64_t fpcr_rmode = UINT64_C(3) << 22;
    const uint64_t fpcr_dn = UINT64_C(1) << 25;
    const uint64_t fpcr_all_controls = (UINT64_C(1) << 26)
        | fpcr_dn | fpcr_fz | fpcr_rmode | fpcr_fz16
        | (UINT64_C(1) << 15) | (UINT64_C(0x1F) << 8)
        | (UINT64_C(1) << 2) | (UINT64_C(1) << 1)
        | UINT64_C(1);

    puf_reset(env);
    uint64_t expected_init_rdram = hash_rdram(env);
    float expected_route_total = env->route_total;
    float expected_route_arc[WR64_MAX_COURSE_NODES];
    int32_t expected_route_pred[WR64_MAX_COURSE_NODES];
    memcpy(expected_route_arc, env->route_arc, sizeof(expected_route_arc));
    memcpy(expected_route_pred, env->route_pred, sizeof(expected_route_pred));

    uint64_t saved_fpcr = test_read_fpcr();
    uint64_t requested_hostile = saved_fpcr | fpcr_all_controls;
    test_write_fpcr(requested_hostile);
    uint64_t hostile_fpcr = test_read_fpcr();
    test_write_fpcr(saved_fpcr);

    test_write_fpcr(hostile_fpcr);
    uint64_t outer_scope = wr_env_fp_enter();
    uint64_t canonical_fpcr = test_read_fpcr();
    uint64_t inner_scope = wr_env_fp_enter();
    test_write_fpcr(canonical_fpcr ^ fpcr_dn);
    uint64_t mutated_inner_fpcr = test_read_fpcr();
    wr_env_fp_leave(inner_scope);
    uint64_t after_inner_fpcr = test_read_fpcr();
    wr_env_fp_leave(outer_scope);
    uint64_t after_outer_fpcr = test_read_fpcr();
    test_write_fpcr(saved_fpcr);
    assert(canonical_fpcr == 0);
    if (mutated_inner_fpcr != canonical_fpcr) {
        assert(after_inner_fpcr == canonical_fpcr);
    }
    assert(after_outer_fpcr == hostile_fpcr);

    WaveRace64 hostile_init = {};
    test_write_fpcr(hostile_fpcr);
    puf_init(&hostile_init, kwargs);
    uint64_t init_restored_fpcr = test_read_fpcr();
    test_write_fpcr(saved_fpcr);
    assert(init_restored_fpcr == hostile_fpcr);
    assert(hash_rdram(&hostile_init) == expected_init_rdram);
    assert(hostile_init.route_total == expected_route_total);
    assert(memcmp(hostile_init.route_arc,
        expected_route_arc, sizeof(expected_route_arc)) == 0);
    assert(memcmp(hostile_init.route_pred,
        expected_route_pred, sizeof(expected_route_pred)) == 0);
    puf_close(&hostile_init);

    WaveResetDigest expected_reset = wave_reset_digest(env);
    test_write_fpcr(hostile_fpcr);
    puf_reset(env);
    uint64_t reset_restored_fpcr = test_read_fpcr();
    test_write_fpcr(saved_fpcr);
    WaveResetDigest hostile_reset = current_wave_digest(env);
    assert(reset_restored_fpcr == hostile_fpcr);
    assert(wave_reset_digest_equal(&expected_reset, &hostile_reset));

    puf_reset(env);
    puffer_state_refresh(env);
    uint64_t expected_refresh_rdram = hash_rdram(env);
    uint32_t expected_refresh_obs[WR64_OBS_SIZE];
    memcpy(expected_refresh_obs, env->agents[0].observations,
        sizeof(expected_refresh_obs));
    memset(env->agents[0].observations, 0xA5,
        sizeof(expected_refresh_obs));
    test_write_fpcr(hostile_fpcr);
    puffer_state_refresh(env);
    uint64_t refresh_restored_fpcr = test_read_fpcr();
    test_write_fpcr(saved_fpcr);
    assert(refresh_restored_fpcr == hostile_fpcr);
    assert(hash_rdram(env) == expected_refresh_rdram);
    assert(memcmp(env->agents[0].observations,
        expected_refresh_obs, sizeof(expected_refresh_obs)) == 0);

    WR64RenderState expected_render = {};
    WR64RenderState hostile_render = {};
    wr64_capture_render_state(env, &expected_render);
    test_write_fpcr(hostile_fpcr);
    wr64_capture_render_state(env, &hostile_render);
    uint64_t render_restored_fpcr = test_read_fpcr();
    test_write_fpcr(saved_fpcr);
    assert(render_restored_fpcr == hostile_fpcr);
    assert(memcmp(&expected_render,
        &hostile_render, sizeof(expected_render)) == 0);

    puf_reset(env);
    set_action(env, 7, 8, 1, 0, 0);
    puf_step(env);
    HostStepDigest expected_step = current_host_step_digest(env);
    puf_reset(env);
    set_action(env, 7, 8, 1, 0, 0);
    test_write_fpcr(hostile_fpcr);
    puf_step(env);
    uint64_t step_restored_fpcr = test_read_fpcr();
    test_write_fpcr(saved_fpcr);
    HostStepDigest hostile_step = current_host_step_digest(env);
    assert(step_restored_fpcr == hostile_fpcr);
    assert(memcmp(&expected_step, &hostile_step, sizeof(expected_step)) == 0);

    puf_reset(env);
    env->state.tick = WR64_MAX_STEPS;
    set_action(env, 7, 8, 1, 0, 0);
    puf_step(env);
    HostStepDigest expected_terminal = current_host_step_digest(env);
    assert(expected_terminal.terminal != 0);
    puf_reset(env);
    env->state.tick = WR64_MAX_STEPS;
    set_action(env, 7, 8, 1, 0, 0);
    test_write_fpcr(hostile_fpcr);
    puf_step(env);
    uint64_t terminal_restored_fpcr = test_read_fpcr();
    test_write_fpcr(saved_fpcr);
    HostStepDigest hostile_terminal = current_host_step_digest(env);
    assert(terminal_restored_fpcr == hostile_fpcr);
    assert(memcmp(&expected_terminal,
        &hostile_terminal, sizeof(expected_terminal)) == 0);

    puf_reset(env);
    printf("PASS FPCR init/reset/refresh/render/step/terminal "
        "rdram=%016llx authoritative=%016llx hostile=%016llx\n",
        (unsigned long long)expected_step.rdram,
        (unsigned long long)expected_step.authoritative,
        (unsigned long long)hostile_fpcr);
#else
    (void)env;
    (void)kwargs;
    puts("SKIP FPCR host-scope replay (requires aarch64)");
#endif
}

static uint64_t wave_physics_trajectory(WaveRace64* env) {
    const uint32_t primary_rng_addr = UINT32_C(0x800D4640);
    wr_wr32(env->machine.rdram,
        primary_rng_addr, UINT32_C(0xA6A1F097));
    WRPad pad = {};
    pad.stick_y = 80;
    pad.a = 1;
    uint32_t wave_episode = env->wave_episode;
    uint64_t hash = UINT64_C(14695981039346656037);
    uint64_t first_grid = hash_wave_grid(env);
    for (int frame = 0; frame < 120; frame++) {
        (void)wr_env_step(&env->machine, &pad, 1);
        hash = hash_authoritative_state(env, hash);
        assert(env->wave_episode == wave_episode);
    }
    assert(hash_wave_grid(env) != first_grid);
    return hash;
}

static void test_wave_selection_permutations() {
    const uint32_t count = 128;
    uint8_t first_seen[count] = {};
    for (uint32_t env_index = 0; env_index < count; env_index++) {
        WaveRace64 probe = {};
        probe.wave_seed = 42;
        probe.rng = env_index;
        probe.wave_variants = (int32_t)count;
        probe.wave_rng_state = wr64_wave_stream_seed(
            probe.wave_seed, env_index);
        uint8_t cycle_seen[count] = {};
        for (uint32_t episode = 0; episode < count; episode++) {
            uint32_t variant = wr64_wave_next_variant(&probe);
            assert(variant < count);
            assert(!cycle_seen[variant]);
            cycle_seen[variant] = 1;
            if (episode == 0) {
                assert(!first_seen[variant]);
                first_seen[variant] = 1;
            }
        }
    }
    for (uint32_t variant = 0; variant < count; variant++) {
        assert(first_seen[variant]);
    }
    puts("PASS K=128 stratified first reset and full-cycle permutations");
}

static void bind_test_agent(WaveRace64* env,
        float* observations, float* actions,
        float* reward, float* terminal) {
    env->agents[0].observations = observations;
    env->agents[0].actions = actions;
    env->agents[0].rewards = reward;
    env->agents[0].terminals = terminal;
}

static void test_wave_episode_randomization(Dict* base_kwargs) {
    Dict kwargs = {};
    dict_copy(&kwargs, base_kwargs);
    dict_set(&kwargs, "randomize_waves", 1);
    dict_set(&kwargs, "wave_seed", 777);
    dict_set(&kwargs, "wave_variants", 4);

    WaveRace64 env = {};
    env.rng = 0;
    puf_init(&env, &kwargs);
    float observations[WR64_OBS_SIZE] = {};
    float actions[NUM_ATNS] = {};
    float reward = 0.f;
    float terminal = 0.f;
    bind_test_agent(&env, observations, actions, &reward, &terminal);
    assert(env.wave_pool != NULL);
    assert(env.wave_pool->count == 4);

    WaveResetDigest expected[4] = {};
    uint32_t episode_for_variant[4] = {};
    uint32_t seen = 0;
    env.wave_episode = 0;
    uint32_t selection_seed = env.wave_rng_state;
    for (uint32_t episode = 0; episode < 4; episode++) {
        expected[episode] = wave_reset_digest(&env);
        uint32_t variant = expected[episode].active_variant;
        assert(variant < 4);
        assert((seen & (1u << variant)) == 0);
        seen |= 1u << variant;
        episode_for_variant[variant] = episode;
        assert(expected[episode].rdram
            == env.wave_pool->variants[variant].rdram_hash);
        assert(expected[episode].grid
            == env.wave_pool->variants[variant].water_hash);
        float x, y, z;
        wr64_position(&env, &x, &y, &z);
        uint32_t y_bits;
        memcpy(&y_bits, &y, sizeof(y_bits));
        assert(y_bits == expected[episode].vertical_origin);
        assert(env.agents[0].observations[6] == 0.f);
    }
    assert(seen == 0xFu);
    assert(env.wave_rng_state == selection_seed);
    for (int a = 0; a < 4; a++) {
        for (int b = a + 1; b < 4; b++) {
            assert(expected[a].rdram != expected[b].rdram);
            assert(expected[a].grid != expected[b].grid);
        }
    }

    env.wave_episode = 0;
    for (uint32_t episode = 0; episode < 4; episode++) {
        WaveResetDigest replay = wave_reset_digest(&env);
        assert(wave_reset_digest_equal(&expected[episode], &replay));
    }

    env.wave_episode = episode_for_variant[0];
    WaveResetDigest variant_zero = wave_reset_digest(&env);
    uint64_t first_physics = wave_physics_trajectory(&env);
    env.wave_episode = episode_for_variant[1];
    WaveResetDigest variant_one = wave_reset_digest(&env);
    uint64_t second_physics = wave_physics_trajectory(&env);
    assert(variant_zero.active_variant == 0);
    assert(variant_one.active_variant == 1);
    assert(first_physics != second_physics);
    env.wave_episode = episode_for_variant[0];
    (void)wave_reset_digest(&env);
    uint64_t first_physics_replay = wave_physics_trajectory(&env);
    assert(first_physics == first_physics_replay);

    env.wave_episode = 0;
    puf_eval_reset(&env);
    uint32_t eval_variant = env.active_wave_variant;
    assert(env.wave_episode == 1);
    env.wave_episode = 91;
    puf_eval_reset(&env);
    assert(env.wave_episode == 1);
    assert(env.active_wave_variant == eval_variant);

    uint32_t target_variant = expected[1].active_variant;
    WaveRace64 native = {};
    native.rng = target_variant;
    wr64_puf_init_core(&native, &kwargs);
    float native_observations[WR64_OBS_SIZE] = {};
    float native_actions[NUM_ATNS] = {};
    float native_reward = 0.f;
    float native_terminal = 0.f;
    bind_test_agent(&native, native_observations, native_actions,
        &native_reward, &native_terminal);
    native.randomize_waves = 0;
    puf_reset(&native);
    env.wave_episode = episode_for_variant[target_variant];
    puf_reset(&env);
    assert(env.active_wave_variant == target_variant);
    assert(memcmp(env.machine.rdram,
        native.machine.rdram, WR_RDRAM_SIZE) == 0);
    assert(memcmp(&env.state, &native.state, sizeof(env.state)) == 0);
    assert(memcmp(env.agents[0].observations,
        native.agents[0].observations,
        sizeof(native_observations)) == 0);
    assert(env.vertical_origin == native.vertical_origin);
    WR64RenderState transplanted_render = {};
    WR64RenderState native_render = {};
    wr64_capture_render_state(&env, &transplanted_render);
    wr64_capture_render_state(&native, &native_render);
    assert(memcmp(&transplanted_render,
        &native_render, sizeof(native_render)) == 0);

    set_action(&env, 8, 8, 1, 0, 0);
    set_action(&native, 8, 8, 1, 0, 0);
    puf_step(&env);
    puf_step(&native);
    HostStepDigest transplanted_step = current_host_step_digest(&env);
    HostStepDigest native_step = current_host_step_digest(&native);
    assert(memcmp(&transplanted_step,
        &native_step, sizeof(native_step)) == 0);
    wr64_capture_render_state(&env, &transplanted_render);
    wr64_capture_render_state(&native, &native_render);
    assert(memcmp(&transplanted_render,
        &native_render, sizeof(native_render)) == 0);

    puf_close(&native);
    size_t delta_pages = env.wave_pool->total_pages;
    puf_close(&env);
    dict_clear(&kwargs);
    printf("PASS authentic reset pool variants=4 delta_pages=%zu "
        "cycle=unique replay=exact physics=%016llx/%016llx\n",
        delta_pages,
        (unsigned long long)first_physics,
        (unsigned long long)second_physics);
}

static void test_water_sampler(WaveRace64* env) {
    uint8_t* scratch = (uint8_t*)malloc(WR_RDRAM_SIZE);
    assert(scratch != NULL);
    memcpy(scratch, env->machine.rdram, WR_RDRAM_SIZE);
    uint64_t before = hash_rdram(env);
    uint32_t rng = UINT32_C(0xA17E5EED);
    int branch_a = 0;
    int branch_b = 0;
    for (int sample = 0; sample < 4096; sample++) {
        rng = rng * UINT32_C(1664525) + UINT32_C(1013904223);
        float x = (float)((int32_t)(rng % 40001u) - 20000);
        rng = rng * UINT32_C(1664525) + UINT32_C(1013904223);
        float z = (float)((int32_t)(rng % 40001u) - 20000);

        recomp_context context;
        wr_ctx_init(&context);
        context.r29 = (uint64_t)(int64_t)(int32_t)UINT32_C(0x807FFF00);
        context.f12.fl = x;
        context.f14.fl = z;
        func_8004D30C(scratch, &context);
        float actual = wr64_water_height(env, x, z);
        uint32_t expected_bits;
        uint32_t actual_bits;
        memcpy(&expected_bits, &context.f0.fl, sizeof(expected_bits));
        memcpy(&actual_bits, &actual, sizeof(actual_bits));
        assert(actual_bits == expected_bits);

        const float k0 = wr64_f32_bits(UINT32_C(0x3F93CD3A));
        const float k1 = wr64_f32_bits(UINT32_C(0x3F13CD3A));
        volatile float vzf = k0*z;
        volatile float u0 = k1*z;
        volatile float uf = u0 + x;
        int32_t v = ((int32_t)vzf) % 24576;
        int32_t u = ((int32_t)uf) % 24576;
        int32_t fv = wr64_sub32(wr64_shl32(wr64_asr6(v), 6), v);
        int32_t fu = wr64_sub32(wr64_shl32(wr64_asr6(u), 6), u);
        if (fv < fu) branch_a++;
        else branch_b++;
    }
    assert(branch_a > 0 && branch_b > 0);
    assert(hash_rdram(env) == before);
    free(scratch);
    printf("PASS exact-water-query samples=4096 branches=%d/%d pure=YES\n",
        branch_a, branch_b);
}

static void test_render_state_capture(WaveRace64* env) {
    WR64RenderState a;
    WR64RenderState b;
    memset(&a, 0xA5, sizeof(a));
    memset(&b, 0x5A, sizeof(b));
    uint64_t before = hash_rdram(env);
    wr64_capture_render_state(env, &a);
    wr64_capture_render_state(env, &b);
    assert(hash_rdram(env) == before);
    assert(memcmp(&a, &b, sizeof(a)) == 0);
    assert(a.version == 2);
    assert(a.game_state == wr64_u(env, WR_ADDR_GAMESTATE));
    assert(a.course_id == WR_COURSE_SUNNY_BEACH);
    assert(a.target_node == wr64_node(env));
    assert(a.target_laps == wr64_target_laps(env));
    assert(a.race_time_ms == wr64_race_time_ms(env));
    assert(a.lap_time_ms == wr64_lap_time_ms(env));
    for (int lap = 0; lap < 3; lap++) {
        assert(a.lap_splits_ms[lap] == wr64_lap_split_ms(env, lap));
    }
    assert(a.speed_kmh == wr64_speed_kmh(env));
    assert(a.power == wr64_power(env));
    assert(a.misses == wr64_misses(env));
    assert(a.node_count == wr64_node_count(env, WR64_COURSE_PRIMARY));
    assert(a.node_count > 0);
    for (int i = 0; i < a.node_count; i++) {
        const WR64RenderNode* node = &a.nodes[i];
        uint32_t address = wr64_course_addr(WR64_COURSE_PRIMARY, i, 0);
        assert(node->index == i);
        assert(node->next == wr64_next_node(env, WR64_COURSE_PRIMARY, i));
        assert(node->type == (int32_t)wr64_u(
            env, address + WR64_COURSE_NODE_TYPE));
        float pass_x;
        float pass_z;
        assert(wr64_pass_point(env, i, &pass_x, &pass_z));
        assert(memcmp(&node->pass_x, &pass_x, sizeof(float)) == 0);
        assert(memcmp(&node->pass_z, &pass_z, sizeof(float)) == 0);
    }
    for (int row = 0; row < WR64_RENDER_WATER_DIM; row++) {
        for (int col = 0; col < WR64_RENDER_WATER_DIM; col++) {
            float x = a.water_origin_x + (float)col*a.water_spacing;
            float z = a.water_origin_z + (float)row*a.water_spacing;
            float expected = wr64_water_height(env, x, z);
            float actual = a.water[row*WR64_RENDER_WATER_DIM + col];
            assert(memcmp(&expected, &actual, sizeof(float)) == 0);
        }
    }
    assert(wr64_render_state_hash(&a) == wr64_render_state_hash(&b));
    assert(env->client == NULL);
    puts("PASS render-state authoritative and read-only");
}

static ActionDigest action_digest(WaveRace64* env,
        int a, int b, int z, int r) {
    puf_reset(env);
    WRPad pad = {};
    pad.stick_y = 80;
    pad.a = (uint8_t)a;
    pad.b = (uint8_t)b;
    pad.z = (uint8_t)z;
    pad.r = (uint8_t)r;

    uint64_t trajectory = UINT64_C(14695981039346656037);
    for (int frame = 0; frame < 240; frame++) {
        (void)wr_env_step(&env->machine, &pad, 1);
        trajectory = hash_authoritative_state(env, trajectory);
    }

    uint64_t rdram = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < WR_RDRAM_SIZE; i++) {
        rdram = hash_byte(rdram, env->machine.rdram[i]);
    }
    return {trajectory, rdram};
}

static void test_action_effects(WaveRace64* env) {
    ActionDigest a = action_digest(env, 1, 0, 0, 0);
    ActionDigest a_repeat = action_digest(env, 1, 0, 0, 0);
    ActionDigest z = action_digest(env, 0, 0, 1, 0);
    ActionDigest ab = action_digest(env, 1, 1, 0, 0);
    ActionDigest ar = action_digest(env, 1, 0, 0, 1);
    int b_trajectory = ab.trajectory != a.trajectory;
    int r_trajectory = ar.trajectory != a.trajectory;
    int b_rdram = ab.rdram != a.rdram;
    int r_rdram = ar.rdram != a.rdram;
    printf("PASS action-effects B=%d/%d R=%d/%d A/Z=%d/%d "
        "(trajectory/RDRAM)\n", b_trajectory, b_rdram,
        r_trajectory, r_rdram, a.trajectory == z.trajectory,
        a.rdram == z.rdram);
    assert(a_repeat.trajectory == a.trajectory && a_repeat.rdram == a.rdram);
    // A and Z are documented throttle aliases. Controller bookkeeping may
    // differ, but their authoritative gameplay trajectory must not.
    assert(a.trajectory == z.trajectory);
    // This straight launch is only a characterization. B is retained because
    // the intervention suite below requires authoritative effects in diverse
    // mid-race and recovery states.
    assert(b_rdram);
    assert(r_trajectory && r_rdram);
    puf_reset(env);
}

static int nearest_stick(float desired) {
    int best = 0;
    float best_error = 1000.f;
    for (int i = 0; i < 15; i++) {
        float error = fabsf(desired - (float)WR64_STICK_X[i]);
        if (error < best_error) {
            best = i;
            best_error = error;
        }
    }
    return best;
}

static void set_rewards(WaveRace64* env, float speed, float progress,
        float slip, float checkpoint, float miss, float finish, float fail) {
    env->reward_speed = speed;
    env->reward_progress = progress;
    env->reward_slip = slip;
    env->reward_checkpoint = checkpoint;
    env->reward_miss = miss;
    env->reward_finish = finish;
    env->reward_fail = fail;
}

static void set_action(WaveRace64* env, int x, int y,
        int a, int b, int r) {
    float* actions = env->agents[0].actions;
    actions[0] = (float)x;
    actions[1] = (float)y;
    actions[2] = (float)a;
    actions[3] = (float)b;
    actions[4] = (float)r;
}

static void test_native_speed_conversion() {
    assert(wr64_speed_to_kmh(54.999f) == 97);
    assert(wr64_speed_to_kmh(55.000f) == 99);
    assert(wr64_speed_to_kmh(55.999f) == 99);
    assert(wr64_speed_to_kmh(63.999f) == 113);
    assert(wr64_speed_to_kmh(10000.f) == 999);
    assert(wr64_speed_to_kmh(NAN) == 0);
    puts("PASS native double-truncation speed conversion");
}

static void test_action_contract(WaveRace64* env) {
    int sizes[] = ACT_SIZES;
    int expected_sizes[] = {15, 9, 2, 2, 2};
    assert(OBS_SIZE == 57);
    assert(NUM_ATNS == 5);
    for (int i = 0; i < NUM_ATNS; i++) assert(sizes[i] == expected_sizes[i]);

    env->agents[0].rewards[0] = 17.f;
    env->agents[0].terminals[0] = 0.75f;
    puf_reset(env);
    // Reset belongs to the new state. The vector backend owns transition-buffer
    // initialization and must preserve a terminal through autoreset.
    assert(env->agents[0].rewards[0] == 17.f);
    assert(env->agents[0].terminals[0] == 0.75f);
    env->agents[0].rewards[0] = 0.f;
    env->agents[0].terminals[0] = 0.f;

    puf_reset(env);
    set_action(env, 0, 0, 0, 0, 0);
    puf_step(env);
    float expected_low[] = {0.f, 0.f, 0.f, 0.f, -1.f, -1.f};
    for (int i = 0; i < 6; i++) {
        assert(fabsf(env->agents[0].observations[9 + i] - expected_low[i]) < 1e-6f);
    }

    set_action(env, 14, 8, 1, 1, 1);
    puf_step(env);
    float expected_high[] = {1.f, 1.f, 0.f, 1.f, 1.f, 1.f};
    for (int i = 0; i < 6; i++) {
        assert(fabsf(env->agents[0].observations[9 + i]
            - expected_high[i]) < 1e-6f);
    }
    for (int i = 0; i < OBS_SIZE; i++) {
        assert(isfinite(env->agents[0].observations[i]));
    }
    float expected_internal_speed = wr64_physics_speed(env) / WR64_SPEED_SCALE;
    float expected_power = (float)wr64_power(env) * 0.2f;
    assert(memcmp(&env->agents[0].observations[55],
        &expected_internal_speed, sizeof(float)) == 0);
    assert(memcmp(&env->agents[0].observations[56],
        &expected_power, sizeof(float)) == 0);
    float x;
    float y;
    float z;
    wr64_position(env, &x, &y, &z);
    float hx;
    float hz;
    wr64_heading(env, env->state.velocity_x, env->state.velocity_z, &hx, &hz);
    static const float lateral_offsets[3] = {-96.f, 0.f, 96.f};
    static const float forward_offsets[4] = {-64.f, 64.f, 192.f, 384.f};
    int index = 43;
    for (int forward_i = 0; forward_i < 4; forward_i++) {
        for (int lateral_i = 0; lateral_i < 3; lateral_i++) {
            float forward = forward_offsets[forward_i];
            float lateral = lateral_offsets[lateral_i];
            float sample_x = x + forward*hx + lateral*hz;
            float sample_z = z + forward*hz - lateral*hx;
            float expected = (wr64_water_height(env, sample_x, sample_z) - y)
                * 0.01f;
            assert(memcmp(&env->agents[0].observations[index],
                &expected, sizeof(float)) == 0);
            index++;
        }
    }
}

static void test_internal_frameskip(WaveRace64* env) {
    set_rewards(env, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
    env->frameskip = 4;
    puf_reset(env);
    set_action(env, 7, 8, 1, 0, 0);
    puf_step(env);
    assert(env->state.tick == 4);
    assert(fabsf(env->agents[0].observations[15]
        - 4.f / (float)WR64_MAX_STEPS) < 1e-7f);
    env->frameskip = 1;
    puf_reset(env);
    puts("PASS internal-frameskip policy-transition parity");
}

static void characterize_body_basis(WaveRace64* env) {
    static const uint32_t offsets[9] = {
        WR_PHYSICS_BASIS_0_X, WR_PHYSICS_BASIS_0_Y,
        WR_PHYSICS_BASIS_0_Z, WR_PHYSICS_BASIS_1_X,
        WR_PHYSICS_BASIS_1_Y, WR_PHYSICS_BASIS_1_Z,
        WR_PHYSICS_BASIS_2_X, WR_PHYSICS_BASIS_2_Y,
        WR_PHYSICS_BASIS_2_Z,
    };
    puf_reset(env);
    uint32_t physics = wr64_physics_addr(env, 0);
    printf("body-basis reset");
    for (int i = 0; i < 9; i++) {
        printf(" %+.6f", wr64_f(env, physics + offsets[i]));
    }
    putchar('\n');
    for (int row = 0; row < 3; row++) {
        float x = wr64_f(env, physics + offsets[3*row]);
        float y = wr64_f(env, physics + offsets[3*row + 1]);
        float z = wr64_f(env, physics + offsets[3*row + 2]);
        assert(fabsf(x*x + y*y + z*z - 1.f) < 1e-3f);
    }
    float hx;
    float hz;
    wr64_heading(env, 0.f, 0.f, &hx, &hz);
    float basis0_dot = hx*wr64_f(env, physics + WR_PHYSICS_BASIS_0_X)
        + hz*wr64_f(env, physics + WR_PHYSICS_BASIS_0_Z);
    float basis2_dot = hx*wr64_f(env, physics + WR_PHYSICS_BASIS_2_X)
        + hz*wr64_f(env, physics + WR_PHYSICS_BASIS_2_Z);
    printf(" heading=(%+.6f,%+.6f) heading-dot-basis0=%+.6f "
        "heading-dot-basis2=%+.6f\n", hx, hz, basis0_dot, basis2_dot);
}

static void assert_body_basis_observation(WaveRace64* env) {
    const float* o = env->agents[0].observations;
    for (int row = 0; row < 3; row++) {
        float x = o[34 + 3*row];
        float y = o[35 + 3*row];
        float z = o[36 + 3*row];
        assert(fabsf(x*x + y*y + z*z - 1.f) < 2e-3f);
    }
    for (int a = 0; a < 3; a++) {
        for (int b = a + 1; b < 3; b++) {
            float dot = o[34 + 3*a] * o[34 + 3*b]
                + o[35 + 3*a] * o[35 + 3*b]
                + o[36 + 3*a] * o[36 + 3*b];
            assert(fabsf(dot) < 2e-3f);
        }
    }
}

static void test_strict_contract_and_recovery_lane(WaveRace64* env) {
    puf_reset(env);
    assert(wr64_reset_contract_valid(env, 3));
    assert(wr64_u(env, WR_ADDR_MODE_STATE) == 2);
    assert(wr64_race_time_ms(env) == 0);
    assert(wr64_lap_time_ms(env) == 0);
    assert(wr64_environment_fault(env, WR_STATE_RACING, 1) == 0);
    assert(wr64_environment_fault(env, 0xFFFFFFFFu, 1) == 1);

    set_action(env, 7, 4, 1, 0, 0);
    for (int update = 0; update < 16; update++) puf_step(env);
    assert(wr64_race_time_ms(env) > 0);
    assert(wr64_lap_time_ms(env) > 0);
    assert(wr64_physics_speed(env) > 0.f);
    assert(!wr64_reset_contract_valid(env, 3));
    puf_reset(env);
    assert(wr64_reset_contract_valid(env, 3));

    uint32_t active_rider = wr64_u(env, WR64_ACTIVE_RIDER_ADDR);
    wr32(env, WR64_ACTIVE_RIDER_ADDR, 1);
    assert(wr64_active_rider(env) == -1);
    assert(!wr64_race_identity_valid(env));
    wr32(env, WR64_ACTIVE_RIDER_ADDR, active_rider);

    uint32_t rider_count = wr64_u(env, WR64_RIDER_COUNT_ADDR);
    wr32(env, WR64_RIDER_COUNT_ADDR, 2);
    assert(wr64_active_rider(env) == -1);
    assert(!wr64_race_identity_valid(env));
    wr32(env, WR64_RIDER_COUNT_ADDR, rider_count);

    uint32_t node_address = wr64_rider_addr(env, WR_RIDER_NODE);
    uint32_t node = wr64_u(env, node_address);
    wr32(env, node_address, WR64_MAX_COURSE_NODES);
    assert(!wr64_race_identity_valid(env));
    wr32(env, node_address, node);
    assert(wr64_reset_contract_valid(env, 3));

    uint32_t recovery_address = wr64_physics_addr(env, WR64_PHYSICS_RECOVERY);
    uint16_t guest_value = wr_rd16(env->machine.rdram, recovery_address);
    uint8_t* wrong_lane = env->machine.rdram
        + (recovery_address & 0x1FFFFFFFu);
    uint16_t wrong_lane_value;
    uint16_t zero = 0;
    memcpy(&wrong_lane_value, wrong_lane, sizeof(wrong_lane_value));
    memcpy(wrong_lane, &zero, sizeof(zero));
    wr_wr16(env->machine.rdram, recovery_address, 1);
    assert(wr64_h(env, recovery_address) == 1);
    assert(wr64_recovery(env) == 2);
    wr_wr16(env->machine.rdram, recovery_address, guest_value);
    memcpy(wrong_lane, &wrong_lane_value, sizeof(wrong_lane_value));
    puf_reset(env);
    assert(wr64_recovery(env) == 0);
    puts("PASS strict-contract and recovery-halfword");
}

static void test_buoy_observation_and_wrap(WaveRace64* env) {
    set_rewards(env, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
    puf_reset(env);

    int saw_negative = 0;
    int saw_positive = 0;
    int saw_neutral = 0;
    int32_t count = wr64_node_count(env, WR64_COURSE_PRIMARY);
    for (int32_t node = 0; node < count; node++) {
        int32_t type = (int32_t)wr64_u(env,
            wr64_course_addr(WR64_COURSE_PRIMARY, node, WR64_COURSE_NODE_TYPE));
        float side = wr64_buoy_side(env, node);
        assert(side == (type == 0 ? -1.f : (type == 1 ? 1.f : 0.f)));
        saw_negative |= side < 0.f;
        saw_positive |= side > 0.f;
        saw_neutral |= side == 0.f;
    }
    assert(saw_negative && saw_positive && saw_neutral);
    assert(env->agents[0].observations[16]
        == wr64_buoy_side(env, wr64_node(env)));
    assert(fabsf(env->agents[0].observations[6]) < 1e-6f);

    float previous_absolute = env->state.prev_course_progress;
    float previous_fraction = env->agents[0].observations[20];
    int saw_wrap = 0;
    set_action(env, 7, 8, 1, 0, 0);
    for (int step = 1; step <= 128; step++) {
        puf_step(env);
        assert(env->agents[0].terminals[0] == 0.f);
        assert_body_basis_observation(env);
        float absolute = env->state.prev_course_progress;
        float fraction = env->agents[0].observations[20];
        float delta = absolute - previous_absolute;
        assert(isfinite(delta));
        assert(fabsf(delta) <= 500.f);
        if (previous_fraction > 0.8f && fraction < 0.2f) {
            assert(delta >= 0.f);
            saw_wrap = 1;
        }
        float* obs = env->agents[0].observations;
        assert(fabsf(obs[17]*obs[17] + obs[18]*obs[18] - 1.f) < 1e-4f);
        assert(obs[19] >= 0.f && obs[19] <= 1.f);
        assert(fabsf(obs[24]*obs[24] + obs[25]*obs[25] - 1.f) < 1e-4f);
        assert(obs[26] >= 0.f && obs[26] <= 1.f);
        assert(fabsf(obs[27]*obs[27] + obs[28]*obs[28] - 1.f) < 1e-4f);
        assert(obs[29] >= 0.f && obs[29] <= 1.f);
        assert(obs[30] == 0.f || obs[30] == 1.f);
        assert(obs[31] >= 0.f && obs[31] <= 1.f);
        assert(obs[32] >= 0.f && obs[32] <= 1.f);
        assert(fabsf(obs[2] - env->state.velocity_x / WR64_SPEED_SCALE) < 1e-6f);
        assert(fabsf(obs[3] - env->state.velocity_z / WR64_SPEED_SCALE) < 1e-6f);
        assert(fabsf(obs[33] - env->state.velocity_y / WR64_SPEED_SCALE) < 1e-6f);
        assert(fabsf(obs[8] - hypotf(env->state.velocity_x,
            env->state.velocity_z) / WR64_SPEED_SCALE) < 1e-6f);
        previous_absolute = absolute;
        previous_fraction = fraction;
    }
    assert(saw_wrap);
    puts("PASS buoy-observation and lap-wrap continuity");
}

static void test_missed_buoy(WaveRace64* env) {
    memset(&env->log, 0, sizeof(env->log));
    set_rewards(env, 0.f, 0.f, 0.f, 0.1f, 0.5f, 0.f, 0.f);
    float production_discount = env->discount;
    // Isolate event attribution from potential time discount: a missed buoy
    // must receive the miss penalty and never a successful-checkpoint reward.
    env->discount = 1.f;
    puf_reset(env);
    int first_miss = -1;
    for (int step = 1; step < 400; step++) {
        float* obs = env->agents[0].observations;
        assert(obs[21] == 0.f);
        float angle = atan2f(-obs[18], obs[17]);
        float desired = 50.f * angle;
        if (desired > 80.f) desired = 80.f;
        if (desired < -80.f) desired = -80.f;
        set_action(env, nearest_stick(desired), 8, 1, 0, 0);
        puf_step(env);
        if (env->agents[0].observations[21] > 0.f) {
            first_miss = step;
            break;
        }
    }

    assert(first_miss == 145);
    assert(env->agents[0].terminals[0] == 0.f);
    assert(env->agents[0].observations[21] == 0.2f);
    assert(env->agents[0].rewards[0] == -0.5f);
    printf("PASS missed-buoy frame=%d reward=%.1f\n",
        first_miss, env->agents[0].rewards[0]);

    int terminal_step = -1;
    for (int step = first_miss + 1; step <= 2000; step++) {
        float* obs = env->agents[0].observations;
        float angle = atan2f(-obs[18], obs[17]);
        float desired = 50.f * angle;
        if (desired > 80.f) desired = 80.f;
        if (desired < -80.f) desired = -80.f;
        set_action(env, nearest_stick(desired), 8, 1, 0, 0);
        puf_step(env);
        if (env->agents[0].terminals[0] == 1.f) {
            terminal_step = step;
            break;
        }
    }
    assert(terminal_step == 569);
    assert(env->log.misses == 5.f);
    assert(env->log.disqualification_rate == 1.f);
    assert(env->log.failure_rate == 0.f);
    assert(env->log.success_rate == 0.f);
    assert(env->log.safety_timeout_rate == 0.f);
    printf("PASS official-disqualification frame=%d misses=%.0f\n",
        terminal_step, env->log.misses);
    env->discount = production_discount;
}

static void test_failed_shaping(WaveRace64* env) {
    memset(&env->log, 0, sizeof(env->log));
    set_rewards(env, 0.f, 1.f, 0.f, 0.1f, 0.f, 10.f, 2.f);
    puf_reset(env);
    set_action(env, 7, 8, 1, 0, 0);

    float reward_sum = 0.f;
    double discounted_return = 0.0;
    double discount_power = 1.0;
    int terminal_step = -1;
    for (int step = 1; step <= 1000; step++) {
        puf_step(env);
        reward_sum += env->agents[0].rewards[0];
        discounted_return += discount_power * env->agents[0].rewards[0];
        discount_power *= env->discount;
        if (env->agents[0].terminals[0] == 1.f) {
            terminal_step = step;
            break;
        }
    }

    assert(terminal_step > 0);
    assert(env->log.n == 1.f);
    assert(env->log.score > 1000.f);
    assert(env->log.checkpoints > 0.f);
    assert(env->log.failure_rate == 1.f);
    assert(env->log.success_rate == 0.f);
    assert(fabsf(env->log.episode_return - reward_sum) < 1e-4f);
    assert(reward_sum <= -2.f + 1e-4f);
    double expected = -2.0;
    assert(fabs(discounted_return - expected) < 2e-3);
    assert(env->agents[0].rewards[0] < -2.f);
    printf("PASS discounted-failure frame=%d return=%.3f expected=%.3f\n",
        terminal_step, discounted_return, expected);
}

static void test_terminal_progress_credit(WaveRace64* env) {
    memset(&env->log, 0, sizeof(env->log));
    set_rewards(env, 0.f, 1.f, 0.f, 0.1f, 0.f, 10.f, 2.f);
    env->reward_mode = 1;
    puf_reset(env);
    set_action(env, 7, 8, 1, 0, 0);

    double discounted_return = 0.0;
    double discount_power = 1.0;
    int terminal_step = -1;
    for (int step = 1; step <= 1000; step++) {
        puf_step(env);
        discounted_return += discount_power * env->agents[0].rewards[0];
        discount_power *= env->discount;
        if (env->agents[0].terminals[0] == 1.f) {
            terminal_step = step;
            break;
        }
    }

    assert(terminal_step > 0);
    float terminal_potential = env->log.score / env->route_total
        + env->reward_checkpoint * env->log.checkpoints;
    double expected = -env->reward_fail
        + pow((double)env->discount, terminal_step) * terminal_potential;
    assert(terminal_potential > 0.f);
    assert(fabs(discounted_return - expected) < 2e-3);
    assert(discounted_return > -env->reward_fail);
    printf("PASS terminal-progress-credit frame=%d return=%.3f expected=%.3f\n",
        terminal_step, discounted_return, expected);
    env->reward_mode = 0;
}

static void test_frontier_reward(WaveRace64* env) {
    memset(&env->log, 0, sizeof(env->log));
    set_rewards(env, 0.f, 1.f, 0.f, 0.1f, 0.f, 10.f, 2.f);
    env->reward_mode = 2;
    puf_reset(env);
    set_action(env, 7, 8, 1, 0, 0);

    int terminal_step = -1;
    for (int step = 1; step <= 1000; step++) {
        puf_step(env);
        if (env->agents[0].terminals[0] == 1.f) {
            terminal_step = step;
            break;
        }
    }

    assert(terminal_step > 0);
    float progress_reward = 3.f * env->log.perf;
    float checkpoint_reward = env->reward_checkpoint * env->log.checkpoints;
    float time_cost = env->reward_fail * (1.f - env->discount)
        * (float)(terminal_step - 1);
    float expected = progress_reward + checkpoint_reward
        - time_cost - env->reward_fail;
    assert(progress_reward > 0.f && checkpoint_reward > 0.f);
    assert(fabsf(env->log.episode_return - expected) < 2e-3f);
    printf("PASS frontier-reward frame=%d return=%.3f expected=%.3f\n",
        terminal_step, env->log.episode_return, expected);
    env->reward_mode = 0;
}

static void test_lap_curriculum(WaveRace64* env) {
    env->curriculum_start_laps = 1;
    env->curriculum_max_laps = 3;
    env->curriculum_successes_per_lap = 2;
    env->curriculum_laps = 1;
    env->curriculum_successes = 0;
    puf_reset(env);
    assert(wr64_target_laps(env) == 1);
    assert(wr64_u(env, CONFIG_LAPS_ADDR) == 1);

    wr64_record_curriculum_success(env);
    assert(env->curriculum_laps == 1 && env->curriculum_successes == 1);
    puf_reset(env);
    assert(wr64_target_laps(env) == 1);
    wr64_record_curriculum_success(env);
    assert(env->curriculum_laps == 2 && env->curriculum_successes == 0);
    puf_reset(env);
    assert(wr64_target_laps(env) == 2);
    wr64_record_curriculum_success(env);
    wr64_record_curriculum_success(env);
    assert(env->curriculum_laps == 3 && env->curriculum_successes == 0);
    puf_reset(env);
    assert(wr64_target_laps(env) == 3);

    env->curriculum_laps = 1;
    env->curriculum_successes = 1;
    puf_reset(env);
    assert(wr64_target_laps(env) == 1);
    assert(wr64_u(env, CONFIG_LAPS_ADDR) == 1);
    assert(wr64_u(env, LAP_TARGET_ADDR) == 1);

    puf_eval_reset(env);
    assert(env->curriculum_laps == 3);
    assert(env->curriculum_successes == 0);
    assert(wr64_target_laps(env) == 3);
    assert(wr64_u(env, CONFIG_LAPS_ADDR) == 3);
    assert(wr64_u(env, LAP_TARGET_ADDR) == 3);
    assert(wr64_reset_contract_valid(env, 3));
    assert(env->agents[0].observations[22] == 0.f);
    assert(env->agents[0].observations[23] == 0.f);
    assert(env->agents[0].observations[32] == 0.f);
    puf_reset(env);
    assert(wr64_target_laps(env) == 3);

    env->curriculum_start_laps = 3;
    env->curriculum_max_laps = 3;
    env->curriculum_successes_per_lap = 1;
    env->curriculum_laps = 3;
    env->curriculum_successes = 0;
    puts("PASS lap-curriculum 1->2->3");
}

typedef struct RouteController {
    float steer_gain;
    float throttle_angle;
    float dampen_angle;
    float high_throttle_angle;
    float pass_scale;
    float curve_near_blend;
    float curve_far_blend;
    float curve_distance;
    float slide_angle;
    int stick_y;
} RouteController;

static void route_action_config(WaveRace64* env,
        const RouteController* controller) {
    const float* obs = env->agents[0].observations;
    // Reconstruct both target vectors in the rider-local frame. Distances share
    // the same route-total denominator, so the common scale cancels in atan2.
    float center_x = obs[17] * obs[19];
    float center_z = -obs[18] * obs[19];
    float pass_x = obs[24] * obs[26];
    float pass_z = -obs[25] * obs[26];
    float dx = center_x
        + controller->pass_scale * (pass_x - center_x);
    float dz = center_z
        + controller->pass_scale * (pass_z - center_z);
    if (obs[30] > 0.5f) {
        float nx = obs[27] * obs[29];
        float nz = -obs[28] * obs[29];
        float blend = obs[26]
                < controller->curve_distance / SUNNY_ROUTE_TOTAL
            ? controller->curve_near_blend : controller->curve_far_blend;
        dx = dx * (1.f - blend) + nx * blend;
        dz = dz * (1.f - blend) + nz * blend;
    }
    float angle = atan2f(dz, dx);

    int steer = (int)lrintf(angle * controller->steer_gain);
    if (steer > 80) steer = 80;
    if (steer < -80) steer = -80;
    int a = fabsf(angle) <= controller->throttle_angle;
    int b = fabsf(angle) > controller->dampen_angle;
    int z_button = obs[15] * (float)WR64_MAX_STEPS >= 59.5f
        && controller->high_throttle_angle > 0.f
        && fabsf(angle) > controller->high_throttle_angle;
    int r = controller->slide_angle > 0.f
        && fabsf(angle) > controller->slide_angle;
    // Z is a documented throttle alias, so fold the oracle's Z choice into A.
    set_action(env, nearest_stick((float)steer), controller->stick_y,
        a || z_button, b, r);
}

static void route_action_from_observation(WaveRace64* env) {
    static const RouteController controller = {
        134.347687f, 0.228220314f, 0.242280304f, 2.70581746f,
        1.75507069f, 0.427550882f, 0.f, 484.638611f,
        0.245567679f, 1,
    };
    route_action_config(env, &controller);
}

static WRPad current_agent_pad(WaveRace64* env) {
    WRPad pad = {};
    int x = (int)env->agents[0].actions[0];
    int y = (int)env->agents[0].actions[1];
    if (x < 0) x = 0;
    if (x > 14) x = 14;
    if (y < 0) y = 0;
    if (y > 8) y = 8;
    pad.stick_x = WR64_STICK_X[x];
    pad.stick_y = WR64_STICK_Y[y];
    pad.a = (uint8_t)((int)env->agents[0].actions[2] & 1);
    pad.b = (uint8_t)((int)env->agents[0].actions[3] & 1);
    pad.r = (uint8_t)((int)env->agents[0].actions[4] & 1);
    return pad;
}

static uint64_t intervention_trace(WaveRace64* env,
        const WRSnapshot* checkpoint, const WRPad* pad) {
    wr_current = &env->machine;
    wr_snapshot_restore(checkpoint, &env->machine);
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int update = 0; update < 240; update++) {
        (void)wr_env_step(&env->machine, pad, 1);
        hash = hash_authoritative_state(env, hash);
    }
    return hash;
}

static void compare_interventions(WaveRace64* env, int* b_effects,
        int* stick_y_effects) {
    WRSnapshot checkpoint = {};
    assert(wr_snapshot_capture(&checkpoint, &env->machine) == 0);
    WRPad base = current_agent_pad(env);
    WRPad dampen = base;
    dampen.b ^= 1;
    WRPad stick_y = base;
    stick_y.stick_y = base.stick_y <= 0 ? 80 : -80;
    uint64_t base_hash = intervention_trace(env, &checkpoint, &base);
    uint64_t b_hash = intervention_trace(env, &checkpoint, &dampen);
    uint64_t y_hash = intervention_trace(env, &checkpoint, &stick_y);
    *b_effects += base_hash != b_hash;
    *stick_y_effects += base_hash != y_hash;
    wr_snapshot_free(&checkpoint);
}

static const RouteController PRODUCTION_CONTROLLER = {
    134.347687f, 0.228220314f, 0.242280304f, 2.70581746f,
    1.75507069f, 0.427550882f, 0.f, 484.638611f,
    0.245567679f, 1,
};

static void characterize_b_and_stick_y(WaveRace64* env) {
    static const int decisions[] = {50, 300, 600, 900, 1250};
    int b_effects = 0;
    int stick_y_effects = 0;
    for (size_t probe = 0;
            probe < sizeof(decisions)/sizeof(decisions[0]); probe++) {
        env->frameskip = 4;
        puf_reset(env);
        for (int decision = 1; decision <= decisions[probe]; decision++) {
            route_action_config(env, &PRODUCTION_CONTROLLER);
            puf_step(env);
            assert(env->agents[0].terminals[0] == 0.f);
        }
        compare_interventions(env, &b_effects, &stick_y_effects);
    }
    int midrace_b_effects = b_effects;
    int midrace_stick_y_effects = stick_y_effects;

    env->frameskip = 1;
    puf_reset(env);
    uint32_t rng = UINT32_C(0x12345678);
    int saw_recovery = 0;
    for (int update = 0; update < 8192 && !saw_recovery; update++) {
        rng = rng * UINT32_C(1664525) + UINT32_C(1013904223);
        int x = (int)(rng % 15u);
        rng = rng * UINT32_C(1664525) + UINT32_C(1013904223);
        int y = (int)(rng % 9u);
        rng = rng * UINT32_C(1664525) + UINT32_C(1013904223);
        set_action(env, x, y, (rng >> 0) & 1u, (rng >> 1) & 1u,
            (rng >> 2) & 1u);
        puf_step(env);
        if (env->agents[0].terminals[0] == 1.f) {
            puf_reset(env);
        } else if (wr64_recovery(env) != 0) {
            saw_recovery = 1;
            compare_interventions(env, &b_effects, &stick_y_effects);
        }
    }
    assert(saw_recovery);
    int recovery_b_effect = b_effects - midrace_b_effects;
    int recovery_stick_y_effect = stick_y_effects - midrace_stick_y_effects;
    assert(midrace_b_effects == 5);
    assert(recovery_b_effect == 1);
    assert(midrace_stick_y_effects == 5);
    assert(recovery_stick_y_effect == 1);
    printf("PASS interventions midrace-B=%d/5 recovery-B=%d/1 "
        "midrace-stick-Y=%d/5 recovery-stick-Y=%d/1\n",
        midrace_b_effects, recovery_b_effect,
        midrace_stick_y_effects, recovery_stick_y_effect);
    puf_reset(env);
}

static void test_official_finish(WaveRace64* env) {
    memset(&env->log, 0, sizeof(env->log));
    set_rewards(env, 0.f, 0.f, 0.f, 0.f, 0.f, 10.f, 2.f);
    puf_reset(env);
    wr32(env, CONFIG_LAPS_ADDR, 1);
    wr32(env, LAP_TARGET_ADDR, 1);
    assert(wr64_u(env, WR_ADDR_GAMESTATE) == WR_STATE_RACING);
    assert(wr64_finished(env) == 0);
    assert(wr64_ended(env) == 0);
    assert(wr64_disqualified(env) == 0);

    uint64_t action_hash = UINT64_C(14695981039346656037);
    int terminal_frame = -1;
    for (int frame = 1; frame <= 2500; frame++) {
        assert(wr64_u(env, WR_ADDR_GAMESTATE) == WR_STATE_RACING);
        assert(wr64_finished(env) == 0);
        assert(wr64_ended(env) == 0);
        assert(wr64_disqualified(env) == 0);

        route_action_from_observation(env);
        for (int i = 0; i < NUM_ATNS; i++) {
            action_hash = hash_byte(action_hash,
                (uint8_t)(int)env->agents[0].actions[i]);
        }
        puf_step(env);
        if (env->agents[0].terminals[0] == 0.f) {
            float time_cost = -env->reward_fail * (1.f - env->discount);
            assert(fabsf(env->agents[0].rewards[0] - time_cost) < 1e-7f);
            assert(env->log.n == 0.f);
            continue;
        }

        terminal_frame = frame;
        assert(env->agents[0].rewards[0] == 10.f);
        assert(env->log.n == 1.f);
        assert(env->log.success_rate == 1.f);
        assert(env->log.target_laps == 1.f);
        assert(env->log.three_lap_success_rate == 0.f);
        assert(env->log.failure_rate == 0.f);
        assert(env->log.successful_race_time_ms > 0.f);
        assert(env->log.successful_lap_1_ms > 0.f);
        assert(env->log.successful_lap_2_ms == 0.f);
        assert(env->log.successful_lap_3_ms == 0.f);
        assert(env->log.successful_race_time_ms
            == env->log.successful_lap_1_ms);
        break;
    }

    assert(terminal_frame == 1758);
    assert(env->log.episode_length == 1758.f);
    assert(action_hash == UINT64_C(0x64157B7EA07F2A23));
    int expected_final[] = {4, 1, 0, 1, 1};
    for (int i = 0; i < NUM_ATNS; i++) {
        assert((int)env->agents[0].actions[i] == expected_final[i]);
    }
    printf("PASS official-finish frame=%d hash=%016llx reward=%.1f "
        "time=%.0f lap1=%.0f\n",
        terminal_frame, (unsigned long long)action_hash,
        env->agents[0].rewards[0], env->log.successful_race_time_ms,
        env->log.successful_lap_1_ms);
}

typedef struct ObsStats {
    float min[OBS_SIZE];
    float max[OBS_SIZE];
} ObsStats;

static void obs_stats_init(ObsStats* stats) {
    for (int i = 0; i < OBS_SIZE; i++) {
        stats->min[i] = INFINITY;
        stats->max[i] = -INFINITY;
    }
}

static void obs_stats_add(ObsStats* stats, const float* observations) {
    for (int i = 0; i < OBS_SIZE; i++) {
        assert(isfinite(observations[i]));
        stats->min[i] = fminf(stats->min[i], observations[i]);
        stats->max[i] = fmaxf(stats->max[i], observations[i]);
    }
}

static void assert_observation_ranges(const ObsStats* stats) {
    for (int i = 0; i < OBS_SIZE; i++) {
        assert(isfinite(stats->min[i]) && isfinite(stats->max[i]));
        assert(stats->min[i] >= -64.f && stats->max[i] <= 64.f);
    }
    assert(stats->min[6] >= -2.f && stats->max[6] <= 2.f);
    int unit_features[] = {4, 5, 7, 9, 10, 11, 12, 13, 14, 16,
        17, 18, 24, 25, 27, 28, 30, 31, 34, 35, 36, 37, 38,
        39, 40, 41, 42};
    for (size_t i = 0; i < sizeof(unit_features)/sizeof(unit_features[0]); i++) {
        int feature = unit_features[i];
        assert(stats->min[feature] >= -1.001f);
        assert(stats->max[feature] <= 1.001f);
    }
    int fractions[] = {15, 19, 20, 21, 22, 26, 29, 32, 56};
    for (size_t i = 0; i < sizeof(fractions)/sizeof(fractions[0]); i++) {
        int feature = fractions[i];
        assert(stats->min[feature] >= -1e-5f);
        assert(stats->max[feature] <= 1.00001f);
    }
    for (int feature = 43; feature < 55; feature++) {
        assert(stats->min[feature] >= -4.f);
        assert(stats->max[feature] <= 4.f);
    }
}

static void test_production_three_lap_finish(WaveRace64* env) {
    // Deterministic policy selected by a bounded parameter search. It reads
    // only the production observation and completes all three laps without a
    // miss while actions remain fixed across each four-update transition.
    ObsStats stats;
    int terminal_frame = -1;
    float min_reward = INFINITY;
    float max_reward = -INFINITY;
    double discounted_return = 0.0;
    double discount_power = 1.0;
    uint64_t action_hash = UINT64_C(14695981039346656037);
    memset(&env->log, 0, sizeof(env->log));
    set_rewards(env, 0.f, 1.f, 0.f, 0.1f, 0.5f, 10.f, 2.f);
    env->frameskip = 4;
    puf_reset(env);
    assert(wr64_target_laps(env) == 3);
    obs_stats_init(&stats);
    for (int frame = 1; frame <= 4000; frame++) {
        obs_stats_add(&stats, env->agents[0].observations);
        assert_body_basis_observation(env);
        route_action_config(env, &PRODUCTION_CONTROLLER);
        for (int i = 0; i < NUM_ATNS; i++) {
            action_hash = hash_byte(action_hash,
                (uint8_t)(int)env->agents[0].actions[i]);
        }
        puf_step(env);
        min_reward = fminf(min_reward, env->agents[0].rewards[0]);
        max_reward = fmaxf(max_reward, env->agents[0].rewards[0]);
        discounted_return += discount_power * env->agents[0].rewards[0];
        discount_power *= env->discount;
        if (env->agents[0].terminals[0] == 1.f) {
            terminal_frame = frame;
            break;
        }
    }
    assert(terminal_frame == 1300);
    assert(env->log.episode_length == 5200.f);
    assert(action_hash == UINT64_C(0x6C5A285EE76CE8E0));
    assert(terminal_frame > 0);
    assert(env->log.n == 1.f);
    assert_observation_ranges(&stats);
    assert(env->log.success_rate == 1.f);
    assert(env->log.target_laps == 3.f);
    assert(env->log.three_lap_success_rate == 1.f);
    assert(env->log.failure_rate == 0.f);
    assert(env->log.disqualification_rate == 0.f);
    assert(env->log.safety_timeout_rate == 0.f);
    assert(env->log.misses == 0.f);
    assert(env->log.successful_race_time_ms > 0.f);
    assert(env->log.successful_lap_1_ms > 0.f);
    assert(env->log.successful_lap_2_ms > 0.f);
    assert(env->log.successful_lap_3_ms > 0.f);
    assert(env->log.successful_race_time_ms
        == env->log.successful_lap_1_ms
            + env->log.successful_lap_2_ms
            + env->log.successful_lap_3_ms);
    assert(min_reward < max_reward && max_reward > 1.f);
    double expected = -env->reward_fail
        + (env->reward_fail + env->reward_finish)
            * pow((double)env->discount, terminal_frame - 1);
    assert(fabs(discounted_return - expected) < 3e-3);
    printf("PASS production-three-lap decisions=%d updates=%.0f "
        "hash=%016llx score=%.1f reward=[%.3f,%.3f] "
        "return=%.3f discounted=%.3f "
        "y=[%.3f,%.3f] time=%.0f laps=[%.0f,%.0f,%.0f]\n",
        terminal_frame, env->log.episode_length,
        (unsigned long long)action_hash, env->log.score,
        min_reward, max_reward, env->log.episode_return, discounted_return,
        stats.min[6], stats.max[6], env->log.successful_race_time_ms,
        env->log.successful_lap_1_ms, env->log.successful_lap_2_ms,
        env->log.successful_lap_3_ms);
    env->frameskip = 1;
}

static void test_random_observation_ranges(WaveRace64* env) {
    set_rewards(env, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
    puf_reset(env);
    ObsStats stats;
    obs_stats_init(&stats);
    uint32_t rng = UINT32_C(0x12345678);
    int saw_recovery = 0;
    for (int step = 0; step < 8192; step++) {
        obs_stats_add(&stats, env->agents[0].observations);
        assert_body_basis_observation(env);
        saw_recovery |= env->agents[0].observations[31] > 0.f;
        rng = rng * UINT32_C(1664525) + UINT32_C(1013904223);
        int x = (int)(rng % 15u);
        rng = rng * UINT32_C(1664525) + UINT32_C(1013904223);
        int y = (int)(rng % 9u);
        rng = rng * UINT32_C(1664525) + UINT32_C(1013904223);
        set_action(env, x, y, (rng >> 0) & 1u, (rng >> 1) & 1u,
            (rng >> 2) & 1u);
        puf_step(env);
        if (env->agents[0].terminals[0] == 1.f) puf_reset(env);
    }
    assert(saw_recovery);
    assert_observation_ranges(&stats);
    printf("PASS random-observation-ranges y=[%.3f,%.3f] speed=[%.3f,%.3f]\n",
        stats.min[6], stats.max[6], stats.min[8], stats.max[8]);
}

static int run_production_baseline(WaveRace64* env, uint32_t* rng,
        double* discounted_out) {
    memset(&env->log, 0, sizeof(env->log));
    set_rewards(env, 0.f, 1.f, 0.f, 0.1f, 0.5f, 10.f, 2.f);
    env->frameskip = 4;
    puf_reset(env);
    double discounted_return = 0.0;
    double discount_power = 1.0;
    for (int decision = 1; decision <= 4000; decision++) {
        if (rng) {
            *rng = *rng * UINT32_C(1664525) + UINT32_C(1013904223);
            int x = (int)(*rng % 15u);
            *rng = *rng * UINT32_C(1664525) + UINT32_C(1013904223);
            int y = (int)(*rng % 9u);
            *rng = *rng * UINT32_C(1664525) + UINT32_C(1013904223);
            set_action(env, x, y, (*rng >> 0) & 1u, (*rng >> 1) & 1u,
                (*rng >> 2) & 1u);
        } else {
            set_action(env, 7, 4, 0, 0, 0);
        }
        puf_step(env);
        discounted_return += discount_power * env->agents[0].rewards[0];
        discount_power *= env->discount;
        if (env->agents[0].terminals[0] == 1.f) {
            if (discounted_out) *discounted_out = discounted_return;
            return decision;
        }
    }
    return -1;
}

static void assert_baseline_result(WaveRace64* env, int decisions) {
    assert(decisions > 0);
    assert(env->log.n == 1.f);
    assert(env->log.success_rate == 0.f);
    assert(isfinite(env->log.perf));
    assert(env->log.perf >= 0.f && env->log.perf <= 1.f);
    float causes = env->log.failure_rate + env->log.disqualification_rate
        + env->log.safety_timeout_rate;
    assert(causes == 1.f);
}

static void test_production_baselines(WaveRace64* env) {
    double noop_discounted = 0.0;
    int noop_decisions = run_production_baseline(
        env, NULL, &noop_discounted);
    assert_baseline_result(env, noop_decisions);
    assert(env->log.misses == 0.f);
    assert(fabs(noop_discounted + env->reward_fail) < 2e-3);
    printf("PASS no-op-baseline decisions=%d perf=%.4f discounted=%.3f "
        "cause=%.0f/%.0f/%.0f\n", noop_decisions, env->log.perf,
        noop_discounted, env->log.failure_rate,
        env->log.disqualification_rate, env->log.safety_timeout_rate);

    uint32_t rng = UINT32_C(0xC001D00D);
    for (int episode = 0; episode < 3; episode++) {
        int decisions = run_production_baseline(env, &rng, NULL);
        assert_baseline_result(env, decisions);
        printf("PASS random-baseline episode=%d decisions=%d perf=%.4f "
            "cause=%.0f/%.0f/%.0f\n",
            episode, decisions, env->log.perf, env->log.failure_rate,
            env->log.disqualification_rate, env->log.safety_timeout_rate);
    }
    env->frameskip = 1;
    puf_reset(env);
}

#ifdef __linux__
static int affinity_equal(const cpu_set_t* a, const cpu_set_t* b) {
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (CPU_ISSET(cpu, a) != CPU_ISSET(cpu, b)) return 0;
    }
    return 1;
}

static void test_vec_affinity_and_ownership(Dict* env_kwargs) {
    Dict vec_kwargs = {};
    dict_set(&vec_kwargs, "total_agents", 2);
    dict_set(&vec_kwargs, "num_buffers", 1);
    dict_set(&vec_kwargs, "num_threads", 2);

    cpu_set_t before;
    cpu_set_t after;
    assert(sched_getaffinity(0, sizeof(before), &before) == 0);
    int num_envs = 0;
    int starts[1] = {};
    int counts[1] = {};
    Env* envs = my_vec_init(
        &num_envs, starts, counts, &vec_kwargs, env_kwargs);
    assert(sched_getaffinity(0, sizeof(after), &after) == 0);
    assert(affinity_equal(&before, &after));
    assert(num_envs == 2 && starts[0] == 0 && counts[0] == 2);
    for (int i = 0; i < num_envs; i++) {
        assert(envs[i].snap.owner == &envs[i].machine);
        puf_close(&envs[i]);
    }
    my_vec_close(envs);
    dict_clear(&vec_kwargs);
    puts("PASS vec-affinity and address-stable ownership");
}

static void test_random_pool_vec_shapes(Dict* base_kwargs) {
    const int shapes[] = {2, 4, 8};
    Dict env_kwargs = {};
    dict_copy(&env_kwargs, base_kwargs);
    dict_set(&env_kwargs, "randomize_waves", 1);
    dict_set(&env_kwargs, "wave_seed", 991);
    dict_set(&env_kwargs, "wave_variants", 4);

    for (size_t shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); shape++) {
        int requested = shapes[shape];
        Dict vec_kwargs = {};
        dict_set(&vec_kwargs, "total_agents", requested);
        dict_set(&vec_kwargs, "num_buffers", 1);
        dict_set(&vec_kwargs, "num_threads", requested < 4 ? requested : 4);
        int num_envs = 0;
        int starts[1] = {};
        int counts[1] = {};
        Env* envs = my_vec_init(
            &num_envs, starts, counts, &vec_kwargs, &env_kwargs);
        assert(num_envs == requested);
        assert(starts[0] == 0 && counts[0] == requested);
        WR64WaveVariantPool* pool = envs[0].wave_pool;
        assert(pool != NULL && pool->count == 4);
        assert(pool->references == (uint32_t)requested);
#ifdef __linux__
        struct stat canonical_backing = {};
        assert(fstat(envs[0].snap.memfd, &canonical_backing) == 0);
        for (int i = 1; i < requested; i++) {
            struct stat rebased_backing = {};
            assert(fstat(envs[i].snap.memfd, &rebased_backing) == 0);
            assert(rebased_backing.st_dev == canonical_backing.st_dev);
            assert(rebased_backing.st_ino == canonical_backing.st_ino);
        }
#endif

        float* observations = (float*)calloc(
            (size_t)requested * WR64_OBS_SIZE, sizeof(float));
        float* actions = (float*)calloc(
            (size_t)requested * NUM_ATNS, sizeof(float));
        float* rewards = (float*)calloc((size_t)requested, sizeof(float));
        float* terminals = (float*)calloc((size_t)requested, sizeof(float));
        assert(observations && actions && rewards && terminals);
        uint32_t first_seen = 0;
        for (int i = 0; i < requested; i++) {
            assert(envs[i].wave_pool == pool);
            assert(envs[i].snap.owner == &envs[i].machine);
            assert(envs[i].snap.rdram == NULL);
            bind_test_agent(&envs[i],
                observations + (size_t)i * WR64_OBS_SIZE,
                actions + (size_t)i * NUM_ATNS,
                &rewards[i], &terminals[i]);
            envs[i].curriculum_laps = 1;
            puf_reset(&envs[i]);
            assert(wr64_reset_contract_valid(
                &envs[i], envs[i].curriculum_laps));
            assert(wr64_target_laps(&envs[i]) == 1);
            if (i < 4) {
                uint32_t bit = 1u << envs[i].active_wave_variant;
                assert((first_seen & bit) == 0);
                first_seen |= bit;
            }
        }
        assert(__builtin_popcount(first_seen)
            == (requested < 4 ? requested : 4));
        uint64_t untouched = hash_rdram(&envs[1]);
        set_action(&envs[0], 8, 8, 1, 0, 0);
        puf_step(&envs[0]);
        assert(hash_rdram(&envs[1]) == untouched);
        for (int i = 1; i < requested; i++) {
            set_action(&envs[i], 8, 8, 1, 0, 0);
            puf_step(&envs[i]);
        }

        uint32_t replay_episode = envs[1].wave_episode;
        puf_reset(&envs[1]);
        set_action(&envs[1], 11, 8, 1, 0, 1);
        puf_step(&envs[1]);
        HostStepDigest owner_alive = current_host_step_digest(&envs[1]);
        envs[1].wave_episode = replay_episode;
        puf_close(&envs[0]);
        assert(pool->references == (uint32_t)requested - 1u);
        puf_reset(&envs[1]);
        set_action(&envs[1], 11, 8, 1, 0, 1);
        puf_step(&envs[1]);
        HostStepDigest owner_closed = current_host_step_digest(&envs[1]);
        assert(memcmp(&owner_alive, &owner_closed,
            sizeof(owner_alive)) == 0);
        for (int i = 1; i < requested; i++) puf_close(&envs[i]);
        my_vec_close(envs);
        free(terminals);
        free(rewards);
        free(actions);
        free(observations);
        dict_clear(&vec_kwargs);
        printf("PASS random pool vec N=%d K=4 ownership/isolation\n",
            requested);
    }
    dict_clear(&env_kwargs);
}
#endif

int main(int argc, char** argv) {
    assert(argc == 2);
    Dict kwargs = {};
    dict_set_str(&kwargs, "rom_path", argv[1]);
    dict_set(&kwargs, "frameskip", 1);
    dict_set(&kwargs, "randomize_waves", 0);
    dict_set(&kwargs, "wave_seed", 42);
    dict_set(&kwargs, "wave_variants", 1);
    dict_set(&kwargs, "reward_speed", 0);
    dict_set(&kwargs, "reward_progress", 1);
    dict_set(&kwargs, "reward_slip", 0);
    dict_set(&kwargs, "reward_checkpoint", 0.1);
    dict_set(&kwargs, "reward_miss", 0.5);
    dict_set(&kwargs, "reward_finish", 10);
    dict_set(&kwargs, "reward_fail", 2);
    dict_set(&kwargs, "discount", 0.9995);
    dict_set(&kwargs, "reward_mode", 0);
    dict_set(&kwargs, "curriculum_start_laps", 3);
    dict_set(&kwargs, "curriculum_max_laps", 3);
    dict_set(&kwargs, "curriculum_successes_per_lap", 1);

    WaveRace64 env = {};
    puf_init(&env, &kwargs);
    assert(env.num_agents == 1);
    assert(env.snap.owner == &env.machine);

    float observations[OBS_SIZE] = {};
    float actions[NUM_ATNS] = {};
    float reward = 0.f;
    float terminal = 0.f;
    env.agents[0].observations = observations;
    env.agents[0].actions = actions;
    env.agents[0].rewards = &reward;
    env.agents[0].terminals = &terminal;

    test_water_sampler(&env);
    test_render_state_capture(&env);
    test_native_speed_conversion();
    test_action_contract(&env);
    test_internal_frameskip(&env);
    characterize_body_basis(&env);
    test_action_effects(&env);
    test_strict_contract_and_recovery_lane(&env);
    test_buoy_observation_and_wrap(&env);
    test_missed_buoy(&env);
    test_failed_shaping(&env);
    test_terminal_progress_credit(&env);
    test_frontier_reward(&env);
    test_lap_curriculum(&env);
    characterize_b_and_stick_y(&env);
    test_official_finish(&env);
    test_production_three_lap_finish(&env);
    test_random_observation_ranges(&env);
    test_production_baselines(&env);
    test_wave_fp_guard(&env, &kwargs);
    test_wave_selection_permutations();
    test_wave_episode_randomization(&kwargs);

    puf_close(&env);
#ifdef __linux__
    test_vec_affinity_and_ownership(&kwargs);
    test_random_pool_vec_shapes(&kwargs);
#endif
    dict_clear(&kwargs);
    puts("PASS waverace64 deterministic regressions");
    return 0;
}
