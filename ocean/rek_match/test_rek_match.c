#define _POSIX_C_SOURCE 200809L

#include <time.h>

#include "rek_match.h"

#define REK_MATCH_TEST_GUARD_LEFT 123456.0f
#define REK_MATCH_TEST_GUARD_RIGHT -654321.0f
#define REK_MATCH_TEST_STEPS 100

static double rek_match_test_monotonic_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec / 1e9;
}

static void rek_match_test_fail(RekMatch* env, const char* message) {
    fprintf(stderr, "rek match diagnostic test: %s\n", message);
    if (env != NULL) c_close(env);
    exit(1);
}

static double rek_match_test_expected_control(
        const mjModel* model,
        int actuator,
        float action) {
    double lower = model->actuator_ctrlrange[2 * actuator];
    double upper = model->actuator_ctrlrange[2 * actuator + 1];
    return 0.5 * (lower + upper)
        + 0.5 * rek_match_normalize_action(action) * (upper - lower);
}

static int rek_match_test_guards_intact(
        const float* observations,
        const float* actions,
        const float* rewards,
        const float* terminals) {
    int observation_values = REK_MATCH_NUM_AGENTS * REK_MATCH_OBS_SIZE;
    int action_values = REK_MATCH_NUM_AGENTS * REK_MATCH_ACTIONS_PER_AGENT;
    return observations[0] == REK_MATCH_TEST_GUARD_LEFT
        && observations[observation_values + 1] == REK_MATCH_TEST_GUARD_RIGHT
        && actions[0] == REK_MATCH_TEST_GUARD_LEFT
        && actions[action_values + 1] == REK_MATCH_TEST_GUARD_RIGHT
        && rewards[0] == REK_MATCH_TEST_GUARD_LEFT
        && rewards[REK_MATCH_NUM_AGENTS + 1] == REK_MATCH_TEST_GUARD_RIGHT
        && terminals[0] == REK_MATCH_TEST_GUARD_LEFT
        && terminals[REK_MATCH_NUM_AGENTS + 1] == REK_MATCH_TEST_GUARD_RIGHT;
}

static int rek_match_test_observations_finite(const RekMatch* env) {
    int values = REK_MATCH_NUM_AGENTS * REK_MATCH_OBS_SIZE;
    for (int i = 0; i < values; i++) {
        if (!isfinite(env->observations[i])) return 0;
    }
    return 1;
}

static int rek_match_test_ego_layout(const RekMatch* env, int observer) {
    int opponent = 1 - observer;
    const float* observation = env->observations
        + observer * REK_MATCH_OBS_SIZE;
    int cursor = 0;
    for (int i = 0; i < REK_MATCH_QPOS_PER_AGENT; i++) {
        if (observation[cursor++]
                != (float)env->data->qpos[
                    observer * REK_MATCH_QPOS_PER_AGENT + i]) {
            return 0;
        }
    }
    for (int i = 0; i < REK_MATCH_QVEL_PER_AGENT; i++) {
        if (observation[cursor++]
                != (float)env->data->qvel[
                    observer * REK_MATCH_QVEL_PER_AGENT + i]) {
            return 0;
        }
    }
    for (int i = 0; i < REK_MATCH_QPOS_PER_AGENT; i++) {
        if (observation[cursor++]
                != (float)env->data->qpos[
                    opponent * REK_MATCH_QPOS_PER_AGENT + i]) {
            return 0;
        }
    }
    for (int i = 0; i < REK_MATCH_QVEL_PER_AGENT; i++) {
        if (observation[cursor++]
                != (float)env->data->qvel[
                    opponent * REK_MATCH_QVEL_PER_AGENT + i]) {
            return 0;
        }
    }
    return cursor == REK_MATCH_OBS_SIZE;
}

static void rek_match_test_partition_guards(RekMatch* env) {
    int physical_envs = 0;
    int envs_per_buffer = 0;
    if (!rek_match_partition_layout(
                8, 2, &physical_envs, &envs_per_buffer)
            || physical_envs != 4 || envs_per_buffer != 2
            || !rek_match_partition_layout(
                6, 3, &physical_envs, &envs_per_buffer)
            || physical_envs != 3 || envs_per_buffer != 1
            || rek_match_partition_layout(7, 1, NULL, NULL)
            || rek_match_partition_layout(8, 3, NULL, NULL)
            || rek_match_partition_layout(0, 1, NULL, NULL)
            || rek_match_partition_layout(8, 0, NULL, NULL)) {
        rek_match_test_fail(env, "physical environment partition guard failed");
    }
}

static void rek_match_test_keyframe_reset(
        RekMatch* env,
        float initial_observations[
            REK_MATCH_NUM_AGENTS * REK_MATCH_OBS_SIZE]) {
    c_reset(env);
    memcpy(
        initial_observations,
        env->observations,
        REK_MATCH_NUM_AGENTS * REK_MATCH_OBS_SIZE * sizeof(float)
    );

    const mjtNum* key_qpos = env->model->key_qpos
        + REK_MATCH_KEYFRAME_ID * REK_MATCH_NQ;
    const mjtNum* key_qvel = env->model->key_qvel
        + REK_MATCH_KEYFRAME_ID * REK_MATCH_NV;
    if (memcmp(env->data->qpos, key_qpos, REK_MATCH_NQ * sizeof(mjtNum)) != 0
            || memcmp(
                env->data->qvel,
                key_qvel,
                REK_MATCH_NV * sizeof(mjtNum)) != 0
            || env->data->time
                != env->model->key_time[REK_MATCH_KEYFRAME_ID]) {
        rek_match_test_fail(env, "reset did not load the exact keyframe state");
    }

    env->data->qpos[0] += 0.25;
    env->data->qvel[0] = 1.0;
    env->data->time += 1.0;
    c_reset(env);
    if (memcmp(
                initial_observations,
                env->observations,
                REK_MATCH_NUM_AGENTS * REK_MATCH_OBS_SIZE * sizeof(float)) != 0
            || memcmp(
                env->data->qpos,
                key_qpos,
                REK_MATCH_NQ * sizeof(mjtNum)) != 0
            || memcmp(
                env->data->qvel,
                key_qvel,
                REK_MATCH_NV * sizeof(mjtNum)) != 0) {
        rek_match_test_fail(env, "keyframe reset was not deterministic");
    }
}

static void rek_match_test_ego_swap(RekMatch* env) {
    for (int i = 0; i < REK_MATCH_NQ; i++) {
        env->data->qpos[i] = (mjtNum)(1000 + i);
    }
    for (int i = 0; i < REK_MATCH_NV; i++) {
        env->data->qvel[i] = (mjtNum)(2000 + i);
    }
    rek_match_compute_observations(env);

    int side_values = REK_MATCH_QPOS_PER_AGENT + REK_MATCH_QVEL_PER_AGENT;
    const float* agent_0 = env->observations;
    const float* agent_1 = env->observations + REK_MATCH_OBS_SIZE;
    if (!rek_match_test_ego_layout(env, 0)
            || !rek_match_test_ego_layout(env, 1)
            || memcmp(agent_0, agent_1 + side_values,
                      (size_t)side_values * sizeof(float)) != 0
            || memcmp(agent_1, agent_0 + side_values,
                      (size_t)side_values * sizeof(float)) != 0) {
        rek_match_test_fail(env, "ego-first observation swap failed");
    }
    c_reset(env);
}

static void rek_match_test_action_isolation(RekMatch* env) {
    const mjtNum sentinel = 12345.0;
    for (int isolated_agent = 0;
            isolated_agent < REK_MATCH_NUM_AGENTS;
            isolated_agent++) {
        for (int i = 0; i < REK_MATCH_NUM_ACTUATORS; i++) {
            env->data->ctrl[i] = sentinel + (mjtNum)i;
        }
        memset(
            env->actions,
            0,
            REK_MATCH_NUM_AGENTS
                * REK_MATCH_ACTIONS_PER_AGENT * sizeof(float)
        );
        for (int action = 0;
                action < REK_MATCH_ACTIONS_PER_AGENT;
                action++) {
            int index = isolated_agent * REK_MATCH_ACTIONS_PER_AGENT + action;
            env->actions[index] = (action % 2 == 0)
                ? 0.02f * (float)(action + 1)
                : -0.015f * (float)(action + 1);
        }

        rek_match_apply_agent_actions(env, isolated_agent);
        for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
            for (int action = 0;
                    action < REK_MATCH_ACTIONS_PER_AGENT;
                    action++) {
                int actuator = agent * REK_MATCH_ACTIONS_PER_AGENT + action;
                if (agent == isolated_agent) {
                    double expected = rek_match_test_expected_control(
                        env->model, actuator, env->actions[actuator]
                    );
                    if (fabs(env->data->ctrl[actuator] - expected) > 1e-12) {
                        rek_match_test_fail(
                            env, "action-to-control mapping mismatch"
                        );
                    }
                } else if (env->data->ctrl[actuator]
                        != sentinel + (mjtNum)actuator) {
                    rek_match_test_fail(
                        env, "agent action modified the other control block"
                    );
                }
            }
        }
    }

    memset(
        env->actions,
        0,
        REK_MATCH_NUM_AGENTS * REK_MATCH_ACTIONS_PER_AGENT * sizeof(float)
    );
    env->actions[0] = NAN;
    env->actions[REK_MATCH_ACTIONS_PER_AGENT] = INFINITY;
    rek_match_apply_actions(env);
    if (env->data->ctrl[0] != rek_match_test_expected_control(
                env->model, 0, 0.0f)
            || env->data->ctrl[REK_MATCH_ACTIONS_PER_AGENT]
                != rek_match_test_expected_control(
                    env->model, REK_MATCH_ACTIONS_PER_AGENT, 0.0f)) {
        rek_match_test_fail(env, "non-finite action did not map to neutral");
    }

    env->actions[0] = 1000.0f;
    env->actions[REK_MATCH_ACTIONS_PER_AGENT] = -1000.0f;
    rek_match_apply_actions(env);
    if (fabs(env->data->ctrl[0] - env->model->actuator_ctrlrange[1]) > 1e-12
            || fabs(
                env->data->ctrl[REK_MATCH_ACTIONS_PER_AGENT]
                - env->model->actuator_ctrlrange[
                    2 * REK_MATCH_ACTIONS_PER_AGENT]) > 1e-12) {
        rek_match_test_fail(env, "extreme action range mapping failed");
    }

    memset(
        env->actions,
        0,
        REK_MATCH_NUM_AGENTS * REK_MATCH_ACTIONS_PER_AGENT * sizeof(float)
    );
    c_reset(env);
}

static void rek_match_test_simultaneous_terminals(
        RekMatch* env,
        const float initial_observations[
            REK_MATCH_NUM_AGENTS * REK_MATCH_OBS_SIZE]) {
    env->max_steps = 3;
    c_reset(env);
    for (int step = 0; step < 3; step++) {
        c_step(env);
        for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
            float expected_terminal = step == 2 ? 1.0f : 0.0f;
            if (env->rewards[agent] != 0.0f
                    || env->terminals[agent] != expected_terminal) {
                rek_match_test_fail(
                    env, "timeout terminal was not shared simultaneously"
                );
            }
        }
    }
    if (env->log.n != 1.0f || env->log.timeout != 1.0f
            || env->log.invalid_termination != 0.0f
            || memcmp(
                initial_observations,
                env->observations,
                REK_MATCH_NUM_AGENTS
                    * REK_MATCH_OBS_SIZE * sizeof(float)) != 0) {
        rek_match_test_fail(env, "timeout boundary did not reset once");
    }

    float boundaries_before = env->log.n;
    float invalid_before = env->log.invalid_termination;
    float timeouts_before = env->log.timeout;
    env->max_steps = REK_MATCH_TEST_STEPS + 1;
    c_reset(env);
    env->data->time = NAN;
    c_step(env);
    for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
        if (env->rewards[agent] != 0.0f || env->terminals[agent] != 1.0f) {
            rek_match_test_fail(
                env, "invalid terminal was not shared simultaneously"
            );
        }
    }
    if (env->log.n != boundaries_before + 1.0f
            || env->log.invalid_termination != invalid_before + 1.0f
            || env->log.timeout != timeouts_before
            || !rek_match_state_is_finite(env)
            || memcmp(
                initial_observations,
                env->observations,
                REK_MATCH_NUM_AGENTS
                    * REK_MATCH_OBS_SIZE * sizeof(float)) != 0
            || !isfinite(env->log.episode_return)
            || !isfinite(env->log.episode_length)
            || !isfinite(env->log.mean_root_height)
            || !isfinite(env->log.max_abs_qvel)) {
        rek_match_test_fail(env, "invalid boundary did not reset once");
    }
}

static double rek_match_test_finite_steps(RekMatch* env) {
    env->max_steps = REK_MATCH_TEST_STEPS + 1;
    c_reset(env);
    double started = rek_match_test_monotonic_seconds();
    for (int step = 0; step < REK_MATCH_TEST_STEPS; step++) {
        for (int action = 0;
                action < REK_MATCH_NUM_AGENTS
                    * REK_MATCH_ACTIONS_PER_AGENT;
                action++) {
            env->actions[action] = 0.05f * (float)sin(
                0.013 * (double)(step + 1) * (double)(action + 3)
            );
        }
        c_step(env);
        if (!rek_match_state_is_finite(env)
                || !rek_match_test_observations_finite(env)) {
            rek_match_test_fail(env, "non-finite 100-step trajectory");
        }
        for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
            if (env->rewards[agent] != 0.0f
                    || env->terminals[agent] != 0.0f) {
                rek_match_test_fail(
                    env, "unexpected diagnostic reward or terminal"
                );
            }
        }
    }
    return rek_match_test_monotonic_seconds() - started;
}

static void rek_match_test_shared_lifecycle(RekMatch* primary) {
    mjModel* shared_model = rek_match_load_model();
    RekMatch shared[2] = {0};
    float observations[2][REK_MATCH_NUM_AGENTS * REK_MATCH_OBS_SIZE] = {0};
    float actions[2][REK_MATCH_NUM_AGENTS * REK_MATCH_ACTIONS_PER_AGENT] = {0};
    float rewards[2][REK_MATCH_NUM_AGENTS] = {0};
    float terminals[2][REK_MATCH_NUM_AGENTS] = {0};

    for (int i = 0; i < 2; i++) {
        shared[i].observations = observations[i];
        shared[i].actions = actions[i];
        shared[i].rewards = rewards[i];
        shared[i].terminals = terminals[i];
        shared[i].num_agents = REK_MATCH_NUM_AGENTS;
        shared[i].max_steps = REK_MATCH_TEST_STEPS + 1;
        rek_match_init_with_model(&shared[i], shared_model, 0);
        shared[i].shared_model_env_count = 2;
        c_reset(&shared[i]);
    }

    if (shared[0].model != shared[1].model
            || shared[0].data == shared[1].data) {
        rek_match_test_fail(primary, "model was not shared with separate data");
    }
    mjtNum other_qpos = shared[1].data->qpos[0];
    shared[0].data->qpos[0] += 1.0;
    if (shared[1].data->qpos[0] != other_qpos) {
        rek_match_test_fail(primary, "physical environment data was shared");
    }

    c_close(&shared[0]);
    c_close(&shared[1]);
    if (shared[0].data != NULL || shared[1].data != NULL
            || shared[0].model != shared_model
            || shared[1].model != shared_model) {
        rek_match_test_fail(primary, "shared data close order failed");
    }
    rek_match_close_shared_model(shared);
    if (shared[0].model != NULL || shared[1].model != NULL) {
        rek_match_test_fail(primary, "shared model was not released once");
    }
}

int main(void) {
    float observation_storage[
        REK_MATCH_NUM_AGENTS * REK_MATCH_OBS_SIZE + 2] = {0};
    float action_storage[
        REK_MATCH_NUM_AGENTS * REK_MATCH_ACTIONS_PER_AGENT + 2] = {0};
    float reward_storage[REK_MATCH_NUM_AGENTS + 2] = {0};
    float terminal_storage[REK_MATCH_NUM_AGENTS + 2] = {0};
    observation_storage[0] = REK_MATCH_TEST_GUARD_LEFT;
    observation_storage[
        REK_MATCH_NUM_AGENTS * REK_MATCH_OBS_SIZE + 1]
        = REK_MATCH_TEST_GUARD_RIGHT;
    action_storage[0] = REK_MATCH_TEST_GUARD_LEFT;
    action_storage[
        REK_MATCH_NUM_AGENTS * REK_MATCH_ACTIONS_PER_AGENT + 1]
        = REK_MATCH_TEST_GUARD_RIGHT;
    reward_storage[0] = REK_MATCH_TEST_GUARD_LEFT;
    reward_storage[REK_MATCH_NUM_AGENTS + 1] = REK_MATCH_TEST_GUARD_RIGHT;
    terminal_storage[0] = REK_MATCH_TEST_GUARD_LEFT;
    terminal_storage[REK_MATCH_NUM_AGENTS + 1] = REK_MATCH_TEST_GUARD_RIGHT;

    RekMatch env = {
        .observations = &observation_storage[1],
        .actions = &action_storage[1],
        .rewards = &reward_storage[1],
        .terminals = &terminal_storage[1],
        .num_agents = REK_MATCH_NUM_AGENTS,
        .max_steps = REK_MATCH_TEST_STEPS + 1,
    };

    double init_started = rek_match_test_monotonic_seconds();
    rek_match_init(&env);
    double init_seconds = rek_match_test_monotonic_seconds() - init_started;
    if (!rek_match_model_identity_matches(env.model)) {
        rek_match_test_fail(&env, "strict model identity check failed");
    }

    char verified_mjcf_sha256[65];
    const char* model_path = getenv("REK_MJCF_PATH");
    if (model_path == NULL
            || !rek_sha256_file(model_path, verified_mjcf_sha256)
            || strcmp(
                verified_mjcf_sha256,
                REK_MATCH_EXPECTED_MJCF_SHA256) != 0) {
        rek_match_test_fail(&env, "MJCF hash was not verified");
    }

    rek_match_test_partition_guards(&env);
    float initial_observations[
        REK_MATCH_NUM_AGENTS * REK_MATCH_OBS_SIZE];
    rek_match_test_keyframe_reset(&env, initial_observations);
    if (!rek_match_state_is_finite(&env)
            || !rek_match_test_observations_finite(&env)) {
        rek_match_test_fail(&env, "initial keyframe state is not finite");
    }

    rek_match_test_ego_swap(&env);
    rek_match_test_action_isolation(&env);
    rek_match_test_simultaneous_terminals(&env, initial_observations);
    double elapsed = rek_match_test_finite_steps(&env);
    rek_match_test_shared_lifecycle(&env);

    if (!rek_match_test_guards_intact(
            observation_storage,
            action_storage,
            reward_storage,
            terminal_storage)) {
        rek_match_test_fail(&env, "buffer guard was modified");
    }

    double steps_per_second = elapsed > 0.0
        ? (double)REK_MATCH_TEST_STEPS / elapsed : 0.0;
    printf(
        "{\"schema\":\"rek.puffer_match_diagnostic_test.v1\","
        "\"model\":\"%s\",\"verified_mjcf_sha256\":\"%s\","
        "\"nbody\":%lld,\"njnt\":%lld,\"ngeom\":%lld,"
        "\"nq\":%lld,\"nv\":%lld,\"nu\":%lld,"
        "\"agents_per_physical_env\":%d,\"observation_size\":%d,"
        "\"actions_per_agent\":%d,\"steps\":%d,"
        "\"finite\":true,\"zero_rewards\":true,"
        "\"model_identity_verified\":true,\"names_verified\":true,"
        "\"ranges_verified\":true,\"addresses_verified\":true,"
        "\"keyframe_reset_verified\":true,"
        "\"action_isolation_verified\":true,\"ego_swap_verified\":true,"
        "\"simultaneous_terminals_verified\":true,"
        "\"partition_guards_verified\":true,"
        "\"buffer_guards_verified\":true,"
        "\"shared_lifecycle_verified\":true,"
        "\"init_seconds\":%.9g,\"elapsed_seconds\":%.9g,"
        "\"steps_per_second\":%.9g}\n",
        REK_MATCH_MODEL_NAME,
        verified_mjcf_sha256,
        (long long)env.model->nbody,
        (long long)env.model->njnt,
        (long long)env.model->ngeom,
        (long long)env.model->nq,
        (long long)env.model->nv,
        (long long)env.model->nu,
        REK_MATCH_NUM_AGENTS,
        REK_MATCH_OBS_SIZE,
        REK_MATCH_ACTIONS_PER_AGENT,
        REK_MATCH_TEST_STEPS,
        init_seconds,
        elapsed,
        steps_per_second
    );
    c_close(&env);
    return 0;
}
