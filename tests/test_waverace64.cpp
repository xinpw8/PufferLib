#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "waverace64.h"

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
    hash = hash_u32(hash, wr64_u(env, rider + WR64_RIDER_MISSES));
    hash = hash_u32(hash, wr64_u(env, rider + WR64_RIDER_DQ));
    hash = hash_u32(hash, wr64_u(env, rider + WR64_RIDER_ENDED));
    hash = hash_u32(hash, wr64_u(env, rider + WR64_RIDER_FINISHED));
    return hash;
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

static void test_action_contract(WaveRace64* env) {
    int sizes[] = ACT_SIZES;
    int expected_sizes[] = {15, 9, 2, 2, 2};
    assert(OBS_SIZE == 43);
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
    assert(wr64_reset_contract_valid(env));
    assert(wr64_environment_fault(env, WR_STATE_RACING, 1) == 0);
    assert(wr64_environment_fault(env, 0xFFFFFFFFu, 1) == 1);

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
    assert(wr64_reset_contract_valid(env));

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

    assert(first_miss == 161);
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
    assert(terminal_step == 743);
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
    double expected = -2.0 * pow((double)env->discount, terminal_step - 1);
    assert(fabs(discounted_return - expected) < 2e-3);
    assert(env->agents[0].rewards[0] < -2.f);
    printf("PASS discounted-failure frame=%d return=%.3f expected=%.3f\n",
        terminal_step, discounted_return, expected);
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
        130.f, 0.50f, 0.95f, 0.60f, 1.f,
        0.85f, 0.45f, 1400.f, 0.f, 4,
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
    75.848f, 0.14112f, 0.49904f, 1.3445f, 1.22524f,
    0.990414f, 0.0293148f, 1025.64f, 0.708171f, 2,
};

static void characterize_b_and_stick_y(WaveRace64* env) {
    static const int decisions[] = {50, 200, 600, 1200, 2000};
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
    assert(recovery_b_effect == 0);
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
    for (int frame = 1; frame <= 1500; frame++) {
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
            assert(env->agents[0].rewards[0] == 0.f);
            assert(env->log.n == 0.f);
            continue;
        }

        terminal_frame = frame;
        assert(env->agents[0].rewards[0] == 10.f);
        assert(env->log.n == 1.f);
        assert(env->log.success_rate == 1.f);
        assert(env->log.failure_rate == 0.f);
        break;
    }

    assert(terminal_frame == 1070);
    assert(action_hash == UINT64_C(0xC6AE00920FD86802));
    int expected_final[] = {14, 4, 1, 0, 0};
    for (int i = 0; i < NUM_ATNS; i++) {
        assert((int)env->agents[0].actions[i] == expected_final[i]);
    }
    printf("PASS official-finish frame=%d hash=%016llx reward=%.1f\n",
        terminal_frame, (unsigned long long)action_hash,
        env->agents[0].rewards[0]);
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
    int fractions[] = {15, 19, 20, 21, 22, 26, 29, 32};
    for (size_t i = 0; i < sizeof(fractions)/sizeof(fractions[0]); i++) {
        int feature = fractions[i];
        assert(stats->min[feature] >= -1e-5f);
        assert(stats->max[feature] <= 1.00001f);
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
        puf_step(env);
        min_reward = fminf(min_reward, env->agents[0].rewards[0]);
        max_reward = fmaxf(max_reward, env->agents[0].rewards[0]);
        if (env->agents[0].terminals[0] == 1.f) {
            terminal_frame = frame;
            break;
        }
    }
    assert(terminal_frame > 0);
    assert(terminal_frame == 2334);
    assert(env->log.n == 1.f);
    assert_observation_ranges(&stats);
    assert(env->log.success_rate == 1.f);
    assert(env->log.failure_rate == 0.f);
    assert(env->log.disqualification_rate == 0.f);
    assert(env->log.safety_timeout_rate == 0.f);
    assert(env->log.misses == 0.f);
    assert(min_reward < max_reward && max_reward > 1.f);
    printf("PASS production-three-lap decisions=%d updates=%.0f "
        "score=%.1f reward=[%.3f,%.3f] return=%.3f y=[%.3f,%.3f]\n",
        terminal_frame, env->log.episode_length, env->log.score,
        min_reward, max_reward, env->log.episode_return,
        stats.min[6], stats.max[6]);
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

static int run_production_baseline(WaveRace64* env, uint32_t* rng) {
    memset(&env->log, 0, sizeof(env->log));
    set_rewards(env, 0.f, 1.f, 0.f, 0.1f, 0.5f, 10.f, 2.f);
    env->frameskip = 4;
    puf_reset(env);
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
        if (env->agents[0].terminals[0] == 1.f) return decision;
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
    int noop_decisions = run_production_baseline(env, NULL);
    assert_baseline_result(env, noop_decisions);
    printf("PASS no-op-baseline decisions=%d perf=%.4f cause=%.0f/%.0f/%.0f\n",
        noop_decisions, env->log.perf, env->log.failure_rate,
        env->log.disqualification_rate, env->log.safety_timeout_rate);

    uint32_t rng = UINT32_C(0xC001D00D);
    for (int episode = 0; episode < 3; episode++) {
        int decisions = run_production_baseline(env, &rng);
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
#endif

int main(int argc, char** argv) {
    assert(argc == 2);
    Dict kwargs = {};
    dict_set_str(&kwargs, "rom_path", argv[1]);
    dict_set(&kwargs, "frameskip", 1);
    dict_set(&kwargs, "reward_speed", 0);
    dict_set(&kwargs, "reward_progress", 1);
    dict_set(&kwargs, "reward_slip", 0);
    dict_set(&kwargs, "reward_checkpoint", 0.1);
    dict_set(&kwargs, "reward_miss", 0.5);
    dict_set(&kwargs, "reward_finish", 10);
    dict_set(&kwargs, "reward_fail", 2);
    dict_set(&kwargs, "discount", 0.999);

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

    test_action_contract(&env);
    test_internal_frameskip(&env);
    characterize_body_basis(&env);
    test_action_effects(&env);
    test_strict_contract_and_recovery_lane(&env);
    test_buoy_observation_and_wrap(&env);
    test_missed_buoy(&env);
    test_failed_shaping(&env);
    characterize_b_and_stick_y(&env);
    test_official_finish(&env);
    test_production_three_lap_finish(&env);
    test_random_observation_ranges(&env);
    test_production_baselines(&env);

    puf_close(&env);
#ifdef __linux__
    test_vec_affinity_and_ownership(&kwargs);
#endif
    dict_clear(&kwargs);
    puts("PASS waverace64 deterministic regressions");
    return 0;
}
