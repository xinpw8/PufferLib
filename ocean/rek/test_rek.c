#define _POSIX_C_SOURCE 200809L

#include <time.h>

#include "rek.h"

#define REK_TEST_GUARD_LEFT 123456.0f
#define REK_TEST_GUARD_RIGHT -654321.0f

static double monotonic_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec / 1e9;
}

static void test_fail(Rek* env, const char* message) {
    fprintf(stderr, "rek plant test: %s\n", message);
    c_close(env);
    exit(1);
}

static int guards_intact(
        const float* observations,
        const float* actions,
        const float* rewards,
        const float* terminals) {
    return observations[0] == REK_TEST_GUARD_LEFT
        && observations[REK_OBS_SIZE + 1] == REK_TEST_GUARD_RIGHT
        && actions[0] == REK_TEST_GUARD_LEFT
        && actions[REK_NUM_ACTUATORS + 1] == REK_TEST_GUARD_RIGHT
        && rewards[0] == REK_TEST_GUARD_LEFT
        && rewards[2] == REK_TEST_GUARD_RIGHT
        && terminals[0] == REK_TEST_GUARD_LEFT
        && terminals[2] == REK_TEST_GUARD_RIGHT;
}

int main(int argc, char** argv) {
    int steps = 100;
    if (argc == 2 && strcmp(argv[1], "--bench") == 0) steps = 10000;

    float observation_storage[REK_OBS_SIZE + 2] = {0};
    float action_storage[REK_NUM_ACTUATORS + 2] = {0};
    float reward_storage[3] = {REK_TEST_GUARD_LEFT, 0.0f, REK_TEST_GUARD_RIGHT};
    float terminal_storage[3] = {REK_TEST_GUARD_LEFT, 0.0f, REK_TEST_GUARD_RIGHT};
    observation_storage[0] = REK_TEST_GUARD_LEFT;
    observation_storage[REK_OBS_SIZE + 1] = REK_TEST_GUARD_RIGHT;
    action_storage[0] = REK_TEST_GUARD_LEFT;
    action_storage[REK_NUM_ACTUATORS + 1] = REK_TEST_GUARD_RIGHT;

    Rek env = {
        .observations = &observation_storage[1],
        .actions = &action_storage[1],
        .rewards = &reward_storage[1],
        .terminals = &terminal_storage[1],
        .num_agents = 1,
        .max_steps = steps + 1,
    };

    double init_started = monotonic_seconds();
    rek_init(&env);
    c_reset(&env);
    double init_seconds = monotonic_seconds() - init_started;

    char verified_mjcf_sha256[65];
    const char* model_path = getenv("REK_MJCF_PATH");
    if (model_path == NULL
            || !rek_sha256_file(model_path, verified_mjcf_sha256)
            || strcmp(verified_mjcf_sha256, REK_EXPECTED_MJCF_SHA256) != 0) {
        test_fail(&env, "MJCF hash was not verified");
    }

    float initial_observations[REK_OBS_SIZE];
    memcpy(initial_observations, env.observations, sizeof(initial_observations));
    if (!rek_state_is_finite(&env)) test_fail(&env, "initial state is not finite");

    for (int i = 0; i < REK_NUM_ACTUATORS; i++) {
        env.actions[i] = (i % 2 == 0) ? 0.5f : -0.25f;
    }
    rek_apply_actions(&env);
    for (int i = 0; i < REK_NUM_ACTUATORS; i++) {
        double lower = env.model->actuator_ctrlrange[2 * i];
        double upper = env.model->actuator_ctrlrange[2 * i + 1];
        double expected = 0.5 * (lower + upper)
            + 0.5 * tanh((double)env.actions[i]) * (upper - lower);
        if (fabs(env.data->ctrl[i] - expected) > 1e-12) {
            test_fail(&env, "action-to-control mapping mismatch");
        }
    }

    env.actions[0] = NAN;
    rek_apply_actions(&env);
    if (env.data->ctrl[0] != 0.0) {
        test_fail(&env, "non-finite action did not map to neutral control");
    }
    env.actions[0] = 1000.0f;
    rek_apply_actions(&env);
    if (fabs(env.data->ctrl[0] - env.model->actuator_ctrlrange[1]) > 1e-12) {
        test_fail(&env, "positive extreme action did not approach upper range");
    }
    env.actions[0] = -1000.0f;
    rek_apply_actions(&env);
    if (fabs(env.data->ctrl[0] - env.model->actuator_ctrlrange[0]) > 1e-12) {
        test_fail(&env, "negative extreme action did not approach lower range");
    }

    memset(env.actions, 0, REK_NUM_ACTUATORS * sizeof(float));
    c_reset(&env);
    for (int i = 0; i < 20; i++) c_step(&env);
    float zero_action_observations[REK_OBS_SIZE];
    memcpy(zero_action_observations, env.observations, sizeof(zero_action_observations));

    c_reset(&env);
    env.actions[0] = 0.5f;
    for (int i = 0; i < 20; i++) c_step(&env);
    double action_effect_max_abs = 0.0;
    for (int i = 0; i < REK_OBS_SIZE; i++) {
        double difference = fabs(
            (double)env.observations[i] - (double)zero_action_observations[i]
        );
        if (difference > action_effect_max_abs) action_effect_max_abs = difference;
    }
    if (!(action_effect_max_abs > 0.0)) {
        test_fail(&env, "non-zero action did not change the trajectory");
    }

    memset(env.actions, 0, REK_NUM_ACTUATORS * sizeof(float));
    env.max_steps = 5;
    c_reset(&env);
    for (int i = 0; i < 5; i++) {
        c_step(&env);
        if (i < 4 && env.terminals[0] != 0.0f) {
            test_fail(&env, "terminal was raised before timeout");
        }
    }
    if (env.terminals[0] != 1.0f
            || env.log.n != 1.0f
            || env.log.timeout != 1.0f) {
        test_fail(&env, "timeout did not produce one recorded terminal boundary");
    }
    if (memcmp(initial_observations, env.observations, sizeof(initial_observations)) != 0) {
        test_fail(&env, "timeout did not publish the deterministic reset state");
    }

    float log_boundaries_before_invalid = env.log.n;
    float invalid_count_before = env.log.invalid_termination;
    float timeout_count_before_invalid = env.log.timeout;
    env.max_steps = steps + 1;
    c_reset(&env);
    env.data->time = NAN;
    c_step(&env);
    if (env.terminals[0] != 1.0f
            || env.log.n != log_boundaries_before_invalid + 1.0f
            || env.log.invalid_termination != invalid_count_before + 1.0f
            || env.log.timeout != timeout_count_before_invalid
            || !rek_state_is_finite(&env)
            || memcmp(initial_observations, env.observations,
                      sizeof(initial_observations)) != 0
            || !isfinite(env.log.episode_return)
            || !isfinite(env.log.episode_length)
            || !isfinite(env.log.root_height)
            || !isfinite(env.log.max_abs_qvel)) {
        test_fail(&env, "invalid state did not produce one finite reset boundary");
    }

    env.max_steps = steps + 1;
    c_reset(&env);
    double started = monotonic_seconds();
    for (int step = 0; step < steps; step++) {
        c_step(&env);
        if (!rek_state_is_finite(&env)) {
            test_fail(&env, "non-finite state in benchmark trajectory");
        }
        if (env.rewards[0] != 0.0f || env.terminals[0] != 0.0f) {
            test_fail(&env, "unexpected diagnostic reward or terminal");
        }
    }
    double elapsed = monotonic_seconds() - started;

    if (!guards_intact(
            observation_storage,
            action_storage,
            reward_storage,
            terminal_storage)) {
        test_fail(&env, "buffer guard was modified");
    }

    printf(
        "{\"schema\":\"rek.puffer_plant_test.v2\","
        "\"plant_id\":\"%s\",\"verified_mjcf_sha256\":\"%s\","
        "\"nbody\":%lld,\"njnt\":%lld,\"ngeom\":%lld,"
        "\"nq\":%lld,\"nv\":%lld,\"nu\":%lld,"
        "\"steps\":%d,\"sim_time\":%.9g,\"finite\":true,"
        "\"zero_reward\":true,\"timeout_terminal_verified\":true,"
        "\"invalid_terminal_verified\":true,\"mjcf_hash_verified\":true,"
        "\"buffer_guards_verified\":true,\"all_actuators_verified\":true,"
        "\"action_effect_max_abs\":%.9g,\"init_seconds\":%.9g,"
        "\"elapsed_seconds\":%.9g,\"steps_per_second\":%.9g}\n",
        REK_PLANT_ID,
        verified_mjcf_sha256,
        (long long)env.model->nbody,
        (long long)env.model->njnt,
        (long long)env.model->ngeom,
        (long long)env.model->nq,
        (long long)env.model->nv,
        (long long)env.model->nu,
        steps,
        env.data->time,
        action_effect_max_abs,
        init_seconds,
        elapsed,
        steps / elapsed
    );
    c_close(&env);
    return 0;
}
