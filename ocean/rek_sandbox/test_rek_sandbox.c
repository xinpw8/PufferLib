#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "rek_sandbox.h"

static void configure_env(RekSandbox* env, int max_steps) {
    env->num_agents = REK_SANDBOX_NUM_AGENTS;
    env->max_steps = max_steps;
    env->action_scale = 0.35f;
    env->action_clip = 1.0f;
    env->dummy_amplitude = 0.15f;
    env->dummy_frequency_hz = 0.5f;
    env->fall_height = 0.5f;
    env->fall_up_z = 0.5f;
    env->upright_reward_weight = 0.5f;
    env->height_reward_weight = 0.5f;
    env->action_cost_weight = 0.002f;
    env->fall_penalty = 5.0f;
    env->root_stabilizer_scale = 1.0f;
}

static void attach_buffers(
        RekSandbox* env,
        float observations[REK_SANDBOX_OBS_SIZE],
        float actions[REK_SANDBOX_NUM_ACTIONS],
        float rewards[REK_SANDBOX_NUM_AGENTS],
        float terminals[REK_SANDBOX_NUM_AGENTS]) {
    env->observations = observations;
    env->actions = actions;
    env->rewards = rewards;
    env->terminals = terminals;
}

static void assert_finite_vector(const float* values, int count) {
    for (int i = 0; i < count; i++) assert(isfinite(values[i]));
}

static void test_actuator_configuration(const RekSandbox* env) {
    for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
        for (int action = 0; action < REK_SANDBOX_NUM_ACTIONS; action++) {
            int actuator = agent * REK_SANDBOX_NUM_ACTIONS + action;
            const mjtNum* gain = env->model->actuator_gainprm
                + actuator * mjNGAIN;
            const mjtNum* bias = env->model->actuator_biasprm
                + actuator * mjNBIAS;
            double force_limit = REK_MATCH_CTRL_RANGE_MAGNITUDES[action];
            assert(env->model->actuator_gaintype[actuator] == mjGAIN_FIXED);
            assert(env->model->actuator_biastype[actuator] == mjBIAS_AFFINE);
            assert(gain[0] == REK_SANDBOX_PD_KP[action]);
            assert(bias[1] == -REK_SANDBOX_PD_KP[action]);
            assert(bias[2] == -REK_SANDBOX_PD_KD[action]);
            assert(env->model->actuator_ctrllimited[actuator] == 0);
            assert(env->model->actuator_forcelimited[actuator] == 1);
            assert(env->model->actuator_forcerange[2 * actuator] == -force_limit);
            assert(env->model->actuator_forcerange[2 * actuator + 1] == force_limit);
        }
    }
}

static void test_action_and_dummy(RekSandbox* env) {
    c_reset(env);
    double baseline = rek_sandbox_keyframe_joint_target(env, 0, 13);
    env->actions[13] = 1.0f;
    c_step(env);
    double expected = rek_sandbox_limit_joint_target(
        env, 0, 13, baseline + env->action_scale
    );
    assert(fabs(env->data->ctrl[13] - expected) < 1e-12);
    assert(env->terminals[0] == 0.0f);
    assert(isfinite(env->rewards[0]));
    assert(env->rewards[0] != 0.0f);

    double dummy_baseline = rek_sandbox_keyframe_joint_target(env, 1, 13);
    c_step(env);
    assert(fabs(env->data->ctrl[REK_SANDBOX_NUM_ACTIONS + 13]
            - dummy_baseline) > 1e-8);
    assert_finite_vector(env->observations, REK_SANDBOX_OBS_SIZE);
}

static void test_dummy_reset_without_terminal(RekSandbox* env) {
    memset(env->actions, 0, REK_SANDBOX_NUM_ACTIONS * sizeof(float));
    c_reset(env);
    env->data->qpos[REK_MATCH_QPOS_PER_AGENT + 2] = 0.1;
    mj_forward(env->model, env->data);
    c_step(env);
    assert(env->terminals[0] == 0.0f);
    assert(env->episode_dummy_resets == 1);
    double expected_height = env->model->key_qpos[
        REK_MATCH_KEYFRAME_ID * env->model->nq
        + REK_MATCH_QPOS_PER_AGENT + 2
    ];
    assert(fabs(env->data->qpos[REK_MATCH_QPOS_PER_AGENT + 2]
            - expected_height) < 1e-12);
}

static void test_learner_fall_terminal(RekSandbox* env) {
    c_reset(env);
    float log_count = env->log.n;
    env->data->qpos[2] = 0.1;
    mj_forward(env->model, env->data);
    c_step(env);
    assert(env->terminals[0] == 1.0f);
    assert(env->rewards[0] < 0.0f);
    assert(env->log.n == log_count + 1.0f);
    assert(env->log.learner_fall >= 1.0f);
    assert(fabs(env->data->qpos[2] - env->model->key_qpos[2]) < 1e-12);
}

static int test_autonomous_rollout(RekSandbox* env) {
    memset(env->actions, 0, REK_SANDBOX_NUM_ACTIONS * sizeof(float));
    env->max_steps = 500;
    c_reset(env);
    int terminals = 0;
    int nonzero_rewards = 0;
    for (int step = 0; step < 2000; step++) {
        for (int action = 0; action < REK_SANDBOX_NUM_ACTIONS; action++) {
            env->actions[action] = 0.15f * sinf(
                0.013f * step + 0.17f * action
            );
        }
        c_step(env);
        terminals += env->terminals[0] == 1.0f;
        nonzero_rewards += env->rewards[0] != 0.0f;
        assert(isfinite(env->rewards[0]));
        assert(rek_sandbox_state_is_finite(env));
        assert_finite_vector(env->observations, REK_SANDBOX_OBS_SIZE);
    }
    assert(terminals > 0);
    assert(nonzero_rewards == 2000);
    return terminals;
}

static void test_zero_action_300_seconds(
        RekSandbox* env,
        double* minimum_height_out,
        double* minimum_up_z_out) {
    memset(env->actions, 0, REK_SANDBOX_NUM_ACTIONS * sizeof(float));
    env->max_steps = 20000;
    c_reset(env);
    double minimum_height = INFINITY;
    double minimum_up_z = INFINITY;
    for (int step = 0; step < 15000; step++) {
        c_step(env);
        assert(env->terminals[0] == 0.0f);
        minimum_height = fmin(
            minimum_height,
            rek_sandbox_root_height(env, 0)
        );
        minimum_up_z = fmin(
            minimum_up_z,
            rek_sandbox_root_up_z(env, 0)
        );
    }
    assert(minimum_height >= env->fall_height);
    assert(minimum_up_z >= env->fall_up_z);
    *minimum_height_out = minimum_height;
    *minimum_up_z_out = minimum_up_z;
}

static void test_timeout(RekSandbox* env) {
    memset(env->actions, 0, REK_SANDBOX_NUM_ACTIONS * sizeof(float));
    env->max_steps = 2;
    c_reset(env);
    c_step(env);
    assert(env->terminals[0] == 0.0f);
    c_step(env);
    assert(env->terminals[0] == 1.0f);
    assert(env->log.timeout >= 1.0f);
}

static void test_shared_model_lifecycle(void) {
    RekSandbox envs[2] = {0};
    float observations[2][REK_SANDBOX_OBS_SIZE] = {{0}};
    float actions[2][REK_SANDBOX_NUM_ACTIONS] = {{0}};
    float rewards[2][REK_SANDBOX_NUM_AGENTS] = {{0}};
    float terminals[2][REK_SANDBOX_NUM_AGENTS] = {{0}};
    mjModel* shared_model = rek_sandbox_load_model();

    for (int i = 0; i < 2; i++) {
        configure_env(&envs[i], 100);
        attach_buffers(
            &envs[i], observations[i], actions[i], rewards[i], terminals[i]
        );
        rek_sandbox_init_with_model(&envs[i], shared_model, 0);
        envs[i].shared_model_env_count = 2;
        c_reset(&envs[i]);
        c_step(&envs[i]);
        c_close(&envs[i]);
    }
    rek_sandbox_close_shared_model(envs);
    assert(envs[0].model == NULL);
    assert(envs[1].model == NULL);
}

int main(void) {
    RekSandbox env = {0};
    float observations[REK_SANDBOX_OBS_SIZE] = {0};
    float actions[REK_SANDBOX_NUM_ACTIONS] = {0};
    float rewards[REK_SANDBOX_NUM_AGENTS] = {0};
    float terminals[REK_SANDBOX_NUM_AGENTS] = {0};

    configure_env(&env, 500);
    attach_buffers(&env, observations, actions, rewards, terminals);
    rek_sandbox_init(&env);
    assert(rek_match_model_dimensions_match(env.model));
    assert(rek_sandbox_config_is_valid(&env));
    test_actuator_configuration(&env);
    c_reset(&env);
    assert_finite_vector(observations, REK_SANDBOX_OBS_SIZE);
    test_action_and_dummy(&env);
    test_dummy_reset_without_terminal(&env);
    test_learner_fall_terminal(&env);
    double zero_action_minimum_height = 0.0;
    double zero_action_minimum_up_z = 0.0;
    test_zero_action_300_seconds(
        &env,
        &zero_action_minimum_height,
        &zero_action_minimum_up_z
    );
    int rollout_terminals = test_autonomous_rollout(&env);
    test_timeout(&env);

    printf(
        "{\"status\":\"pass\",\"rollout_steps\":2000,"
        "\"rollout_terminals\":%d,\"log_n\":%.0f,"
        "\"zero_action_steps\":15000,\"zero_action_seconds\":%.6f,"
        "\"zero_action_minimum_height\":%.9f,"
        "\"zero_action_minimum_up_z\":%.9f,"
        "\"model_sha256\":\"%s\",\"controller_sha256\":\"%s\"," 
        "\"root_controller_sha256\":\"%s\",\"mujoco_version\":%d}\n",
        rollout_terminals,
        env.log.n,
        15000.0 * env.model->opt.timestep,
        zero_action_minimum_height,
        zero_action_minimum_up_z,
        REK_MATCH_EXPECTED_MJCF_SHA256,
        REK_SANDBOX_EXPECTED_CONTROLLER_SHA256,
        REK_SANDBOX_ROOT_CONTROLLER_ARTIFACT_SHA256,
        mj_version()
    );
    c_close(&env);
    test_shared_model_lifecycle();
    return 0;
}
