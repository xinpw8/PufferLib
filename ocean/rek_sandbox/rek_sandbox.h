// Server-free T800 balance curriculum on the hash-bound two-fighter plant.
//
// This environment is deliberately separate from rek_match. The MuJoCo plant,
// actuator force limits, and PdStand gain arrays are evidence-bound. The action
// scale, target baseline, dummy motion, reward, and fall gate are provisional
// curriculum choices. They are not REK behavior or parity claims.

#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mujoco/mujoco.h>

// Reuse the diagnostic's hash and model-identity validation without exporting
// its generic environment symbols into this binding.
#define Log RekMatchDiagnosticLog
#define c_reset rek_match_diagnostic_c_reset
#define c_step rek_match_diagnostic_c_step
#define c_render rek_match_diagnostic_c_render
#define c_close rek_match_diagnostic_c_close
#include "../rek_match/rek_match.h"
#undef c_close
#undef c_render
#undef c_step
#undef c_reset
#undef Log

static void rek_sandbox_reference_diagnostic_helpers(void) {
    (void)rek_match_partition_layout;
    (void)rek_match_init;
    (void)rek_match_close_shared_model;
}

#define REK_SANDBOX_NUM_AGENTS 1
#define REK_SANDBOX_OBS_SIZE REK_MATCH_OBS_SIZE
#define REK_SANDBOX_NUM_ACTIONS REK_MATCH_ACTIONS_PER_AGENT
#define REK_SANDBOX_ACT_SIZES REK_MATCH_ACT_SIZES
#define REK_SANDBOX_EXPECTED_CONTROLLER_SHA256 \
    "5b262c83fa0db89804007ec176e4aefa72bb123090e1b81391b77176d78e28d7"
#define REK_SANDBOX_ROOT_CONTROLLER_ARTIFACT_SHA256 \
    "31f74cf7cb3b416760880b9ca439ffbd39323cc2540c88fc5be9a927ce687761"
#define REK_SANDBOX_EXPECTED_MUJOCO_VERSION 3007000
#define REK_SANDBOX_ROOT_LEG_ACTIONS 12
#define REK_SANDBOX_ROOT_DIMS 6
#define REK_SANDBOX_ROOT_FD_EPSILON 1e-4
#define REK_SANDBOX_ROOT_DAMPING 8.613707589105229e-7
#define REK_SANDBOX_ROOT_TARGET_DELTA_LIMIT 0.06425133234876036

// Serialized EngineAIPolicyRunner PdStand arrays. Static analysis establishes
// their use by the straighten routine only. Using them for continuous residual
// position control is a provisional low-level controller hypothesis.
static const double REK_SANDBOX_PD_KP[REK_SANDBOX_NUM_ACTIONS] = {
    1080.0, 480.0, 880.0, 1000.0, 800.0, 100.0,
    1080.0, 480.0, 880.0, 1000.0, 800.0, 100.0,
    200.0,
    120.0, 120.0, 120.0, 120.0, 120.0,
    120.0, 120.0, 120.0, 120.0, 120.0, 100.0, 100.0,
};

static const double REK_SANDBOX_PD_KD[REK_SANDBOX_NUM_ACTIONS] = {
    25.0, 25.0, 25.0, 25.0, 2.0, 2.0,
    25.0, 25.0, 25.0, 25.0, 2.0, 2.0,
    1.0,
    1.0, 1.0, 1.0, 1.0, 0.200000002980232,
    1.0, 1.0, 1.0, 1.0, 0.200000002980232, 1.0, 1.0,
};

// Unitless joint weights for a deterministic repeated dummy pose. These values
// are a curriculum choice, not recovered move trajectories.
static const double REK_SANDBOX_DUMMY_WAVE[REK_SANDBOX_NUM_ACTIONS] = {
    0.20, 0.10, 0.10, 0.15, 0.05, 0.05,
    -0.20, -0.10, -0.10, -0.15, -0.05, -0.05,
    0.30,
    1.00, 0.25, 0.30, -0.75, 0.10,
    -1.00, -0.25, -0.30, 0.75, -0.10, 0.10, 0.20,
};

static const double REK_SANDBOX_ROOT_KP[REK_SANDBOX_ROOT_DIMS] = {
    120.721216799542,
    120.721216799542,
    44.972098219558,
    388.103557643136,
    388.103557643136,
    0.0,
};

static const double REK_SANDBOX_ROOT_KD[REK_SANDBOX_ROOT_DIMS] = {
    20.256756785599,
    20.256756785599,
    1.553386758450,
    27.141501179662,
    27.141501179662,
    0.127017625339,
};

typedef struct Log {
    float score;
    float episode_return;
    float episode_length;
    float learner_root_height;
    float learner_root_up_z;
    float action_rms;
    float dummy_resets;
    float learner_fall;
    float invalid_termination;
    float timeout;
    float n;
} Log;

typedef struct RekSandbox {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    Log log;

    const mjModel* model;
    mjData* data;
    int owns_model;
    int shared_model_env_count;

    int tick;
    int max_steps;
    float episode_return;
    float episode_action_rms_sum;
    int episode_dummy_resets;

    float action_scale;
    float action_clip;
    float dummy_amplitude;
    float dummy_frequency_hz;
    float fall_height;
    float fall_up_z;
    float upright_reward_weight;
    float height_reward_weight;
    float action_cost_weight;
    float fall_penalty;
    float root_stabilizer_scale;

    double root_pinv[REK_MATCH_NUM_AGENTS]
        [REK_SANDBOX_ROOT_LEG_ACTIONS][REK_SANDBOX_ROOT_DIMS];
} RekSandbox;

static void rek_sandbox_fail(const char* message) {
    fprintf(stderr, "rek sandbox: %s\n", message);
    abort();
}

static double rek_sandbox_clip(double value, double lower, double upper) {
    if (value < lower) return lower;
    if (value > upper) return upper;
    return value;
}

static void rek_sandbox_verify_artifact(
        const char* environment_variable,
        const char* expected_sha256,
        const char* label) {
    const char* path = getenv(environment_variable);
    if (path == NULL || path[0] == '\0') {
        fprintf(stderr, "rek sandbox: %s is required\n", environment_variable);
        abort();
    }

    char actual_sha256[65];
    if (!rek_sha256_file(path, actual_sha256)) {
        fprintf(stderr, "rek sandbox: could not hash %s\n", environment_variable);
        abort();
    }
    if (strcmp(actual_sha256, expected_sha256) != 0) {
        fprintf(
            stderr,
            "rek sandbox: %s SHA-256 mismatch expected=%s actual=%s\n",
            label,
            expected_sha256,
            actual_sha256
        );
        abort();
    }
}

static void rek_sandbox_verify_runtime(void) {
    if (mjVERSION_HEADER != REK_SANDBOX_EXPECTED_MUJOCO_VERSION
            || mj_version() != REK_SANDBOX_EXPECTED_MUJOCO_VERSION) {
        fprintf(
            stderr,
            "rek sandbox: MuJoCo version mismatch expected=%d header=%d runtime=%d\n",
            REK_SANDBOX_EXPECTED_MUJOCO_VERSION,
            mjVERSION_HEADER,
            mj_version()
        );
        abort();
    }
}

static void rek_sandbox_configure_implicit_position_actuators(mjModel* model) {
    for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
        for (int action = 0; action < REK_SANDBOX_NUM_ACTIONS; action++) {
            int actuator = agent * REK_SANDBOX_NUM_ACTIONS + action;
            double force_limit = REK_MATCH_CTRL_RANGE_MAGNITUDES[action];
            mjtNum* gain = model->actuator_gainprm + actuator * mjNGAIN;
            mjtNum* bias = model->actuator_biasprm + actuator * mjNBIAS;

            mju_zero(gain, mjNGAIN);
            mju_zero(bias, mjNBIAS);
            model->actuator_gaintype[actuator] = mjGAIN_FIXED;
            model->actuator_biastype[actuator] = mjBIAS_AFFINE;
            gain[0] = REK_SANDBOX_PD_KP[action];
            bias[1] = -REK_SANDBOX_PD_KP[action];
            bias[2] = -REK_SANDBOX_PD_KD[action];
            model->actuator_ctrllimited[actuator] = 0;
            model->actuator_forcelimited[actuator] = 1;
            model->actuator_forcerange[2 * actuator] = -force_limit;
            model->actuator_forcerange[2 * actuator + 1] = force_limit;
        }
    }
}

static mjModel* rek_sandbox_load_model(void) {
    rek_sandbox_reference_diagnostic_helpers();
    rek_sandbox_verify_runtime();
    rek_sandbox_verify_artifact(
        "REK_CONTROLLER_PATH",
        REK_SANDBOX_EXPECTED_CONTROLLER_SHA256,
        "controller artifact"
    );
    rek_sandbox_verify_artifact(
        "REK_ROOT_CONTROLLER_PATH",
        REK_SANDBOX_ROOT_CONTROLLER_ARTIFACT_SHA256,
        "root-controller artifact"
    );
    mjModel* model = rek_match_load_model();
    rek_sandbox_configure_implicit_position_actuators(model);
    return model;
}

static int rek_sandbox_config_is_valid(const RekSandbox* env) {
    double initial_height = env->model->key_qpos[2];
    return env->max_steps > 0
        && isfinite(env->action_scale) && env->action_scale >= 0.0f
        && isfinite(env->action_clip) && env->action_clip > 0.0f
        && isfinite(env->dummy_amplitude) && env->dummy_amplitude >= 0.0f
        && isfinite(env->dummy_frequency_hz) && env->dummy_frequency_hz >= 0.0f
        && isfinite(env->fall_height) && env->fall_height >= 0.0f
        && env->fall_height < initial_height
        && isfinite(env->fall_up_z) && env->fall_up_z > -1.0f
        && env->fall_up_z < 1.0f
        && isfinite(env->upright_reward_weight)
        && env->upright_reward_weight >= 0.0f
        && isfinite(env->height_reward_weight)
        && env->height_reward_weight >= 0.0f
        && isfinite(env->action_cost_weight) && env->action_cost_weight >= 0.0f
        && isfinite(env->fall_penalty) && env->fall_penalty >= 0.0f
        && isfinite(env->root_stabilizer_scale)
        && env->root_stabilizer_scale >= 0.0f
        && env->root_stabilizer_scale <= 1.0f;
}

static void rek_sandbox_init_root_controller(RekSandbox* env);

static void rek_sandbox_init_with_model(
        RekSandbox* env,
        const mjModel* model,
        int owns_model) {
    if (env == NULL || model == NULL) {
        rek_sandbox_fail("environment and model are required");
    }
    env->model = model;
    env->owns_model = owns_model;
    env->shared_model_env_count = 1;
    env->data = mj_makeData(model);
    if (env->data == NULL) rek_sandbox_fail("mj_makeData failed");
    rek_sandbox_init_root_controller(env);
}

static void rek_sandbox_init(RekSandbox* env) {
    rek_sandbox_init_with_model(env, rek_sandbox_load_model(), 1);
}

static int rek_sandbox_state_is_finite(const RekSandbox* env) {
    for (int i = 0; i < env->model->nq; i++) {
        if (!isfinite(env->data->qpos[i])) return 0;
    }
    for (int i = 0; i < env->model->nv; i++) {
        if (!isfinite(env->data->qvel[i])
                || !isfinite(env->data->qacc[i])) {
            return 0;
        }
    }
    for (int i = 0; i < env->model->nu; i++) {
        if (!isfinite(env->data->actuator_force[i])) return 0;
    }
    return isfinite(env->data->time);
}

static int rek_sandbox_root_body(int agent) {
    return 1 + agent * 30;
}

static double rek_sandbox_root_height(const RekSandbox* env, int agent) {
    return env->data->xpos[3 * rek_sandbox_root_body(agent) + 2];
}

static double rek_sandbox_root_up_z(const RekSandbox* env, int agent) {
    return env->data->xmat[9 * rek_sandbox_root_body(agent) + 8];
}

static int rek_sandbox_fallen(const RekSandbox* env, int agent) {
    return rek_sandbox_root_height(env, agent) < env->fall_height
        || rek_sandbox_root_up_z(env, agent) < env->fall_up_z;
}

static void rek_sandbox_compute_observation(RekSandbox* env) {
    int cursor = 0;
    for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
        int qpos_start = agent * REK_MATCH_QPOS_PER_AGENT;
        int qvel_start = agent * REK_MATCH_QVEL_PER_AGENT;
        for (int i = 0; i < REK_MATCH_QPOS_PER_AGENT; i++) {
            env->observations[cursor++] = (float)env->data->qpos[qpos_start + i];
        }
        for (int i = 0; i < REK_MATCH_QVEL_PER_AGENT; i++) {
            env->observations[cursor++] = (float)env->data->qvel[qvel_start + i];
        }
    }
}

static double rek_sandbox_keyframe_joint_target(
        const RekSandbox* env,
        int agent,
        int action) {
    int qpos = agent * REK_MATCH_QPOS_PER_AGENT
        + REK_MATCH_LOCAL_JOINT_QPOS_ADDRESSES[action];
    return env->model->key_qpos[REK_MATCH_KEYFRAME_ID * env->model->nq + qpos];
}

static double rek_sandbox_limit_joint_target(
        const RekSandbox* env,
        int agent,
        int action,
        double target) {
    int joint = agent * 26 + REK_MATCH_LOCAL_JOINT_IDS[action];
    if (env->model->jnt_limited[joint]) {
        target = rek_sandbox_clip(
            target,
            env->model->jnt_range[2 * joint],
            env->model->jnt_range[2 * joint + 1]
        );
    }
    return target;
}

static int rek_sandbox_inverse_6x6(const double input[36], double output[36]) {
    double augmented[REK_SANDBOX_ROOT_DIMS][2 * REK_SANDBOX_ROOT_DIMS];
    for (int row = 0; row < REK_SANDBOX_ROOT_DIMS; row++) {
        for (int column = 0; column < REK_SANDBOX_ROOT_DIMS; column++) {
            augmented[row][column] = input[row * REK_SANDBOX_ROOT_DIMS + column];
            augmented[row][REK_SANDBOX_ROOT_DIMS + column]
                = row == column ? 1.0 : 0.0;
        }
    }

    for (int column = 0; column < REK_SANDBOX_ROOT_DIMS; column++) {
        int pivot_row = column;
        double pivot_magnitude = fabs(augmented[pivot_row][column]);
        for (int row = column + 1; row < REK_SANDBOX_ROOT_DIMS; row++) {
            double magnitude = fabs(augmented[row][column]);
            if (magnitude > pivot_magnitude) {
                pivot_row = row;
                pivot_magnitude = magnitude;
            }
        }
        if (!(pivot_magnitude > 1e-18) || !isfinite(pivot_magnitude)) return 0;
        if (pivot_row != column) {
            for (int item = 0; item < 2 * REK_SANDBOX_ROOT_DIMS; item++) {
                double temporary = augmented[column][item];
                augmented[column][item] = augmented[pivot_row][item];
                augmented[pivot_row][item] = temporary;
            }
        }

        double pivot = augmented[column][column];
        for (int item = 0; item < 2 * REK_SANDBOX_ROOT_DIMS; item++) {
            augmented[column][item] /= pivot;
        }
        for (int row = 0; row < REK_SANDBOX_ROOT_DIMS; row++) {
            if (row == column) continue;
            double factor = augmented[row][column];
            for (int item = 0; item < 2 * REK_SANDBOX_ROOT_DIMS; item++) {
                augmented[row][item] -= factor * augmented[column][item];
            }
        }
    }

    for (int row = 0; row < REK_SANDBOX_ROOT_DIMS; row++) {
        for (int column = 0; column < REK_SANDBOX_ROOT_DIMS; column++) {
            output[row * REK_SANDBOX_ROOT_DIMS + column]
                = augmented[row][REK_SANDBOX_ROOT_DIMS + column];
        }
    }
    return 1;
}

static void rek_sandbox_set_keyframe_controls(RekSandbox* env) {
    for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
        for (int action = 0; action < REK_SANDBOX_NUM_ACTIONS; action++) {
            int actuator = agent * REK_SANDBOX_NUM_ACTIONS + action;
            env->data->ctrl[actuator] = rek_sandbox_keyframe_joint_target(
                env, agent, action
            );
        }
    }
}

static void rek_sandbox_init_root_controller(RekSandbox* env) {
    double response[REK_SANDBOX_ROOT_DIMS][REK_SANDBOX_ROOT_LEG_ACTIONS];
    mjtNum base_qacc[REK_MATCH_NV];
    mj_resetDataKeyframe(env->model, env->data, REK_MATCH_KEYFRAME_ID);
    rek_sandbox_set_keyframe_controls(env);
    mj_forward(env->model, env->data);
    mju_copy(base_qacc, env->data->qacc, REK_MATCH_NV);

    for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
        int root_dof = agent * REK_MATCH_QVEL_PER_AGENT;
        for (int action = 0; action < REK_SANDBOX_ROOT_LEG_ACTIONS; action++) {
            int actuator = agent * REK_SANDBOX_NUM_ACTIONS + action;
            env->data->ctrl[actuator] += REK_SANDBOX_ROOT_FD_EPSILON;
            mj_forward(env->model, env->data);
            for (int root_axis = 0; root_axis < REK_SANDBOX_ROOT_DIMS; root_axis++) {
                response[root_axis][action] = (
                    env->data->qacc[root_dof + root_axis]
                    - base_qacc[root_dof + root_axis]
                ) / REK_SANDBOX_ROOT_FD_EPSILON;
            }
            env->data->ctrl[actuator] -= REK_SANDBOX_ROOT_FD_EPSILON;
        }
        mj_forward(env->model, env->data);

        double gram[36] = {0};
        double trace = 0.0;
        for (int row = 0; row < REK_SANDBOX_ROOT_DIMS; row++) {
            for (int column = 0; column < REK_SANDBOX_ROOT_DIMS; column++) {
                for (int action = 0; action < REK_SANDBOX_ROOT_LEG_ACTIONS; action++) {
                    gram[row * REK_SANDBOX_ROOT_DIMS + column]
                        += response[row][action] * response[column][action];
                }
            }
            trace += gram[row * REK_SANDBOX_ROOT_DIMS + row];
        }
        double scale = fmax(trace / REK_SANDBOX_ROOT_DIMS, 1e-12);
        for (int axis = 0; axis < REK_SANDBOX_ROOT_DIMS; axis++) {
            gram[axis * REK_SANDBOX_ROOT_DIMS + axis]
                += REK_SANDBOX_ROOT_DAMPING * scale;
        }

        double inverse[36];
        if (!rek_sandbox_inverse_6x6(gram, inverse)) {
            rek_sandbox_fail("root controller inverse failed");
        }
        for (int action = 0; action < REK_SANDBOX_ROOT_LEG_ACTIONS; action++) {
            for (int output_axis = 0;
                    output_axis < REK_SANDBOX_ROOT_DIMS;
                    output_axis++) {
                double value = 0.0;
                for (int input_axis = 0;
                        input_axis < REK_SANDBOX_ROOT_DIMS;
                        input_axis++) {
                    value += response[input_axis][action]
                        * inverse[input_axis * REK_SANDBOX_ROOT_DIMS + output_axis];
                }
                env->root_pinv[agent][action][output_axis] = value;
            }
        }
    }
}

static void rek_sandbox_apply_root_stabilizer(RekSandbox* env) {
    if (env->root_stabilizer_scale <= 0.0f) return;
    mjtNum tangent_error[REK_MATCH_NV];
    const mjtNum* key_qpos = env->model->key_qpos
        + REK_MATCH_KEYFRAME_ID * env->model->nq;
    mj_differentiatePos(
        env->model,
        tangent_error,
        1.0,
        env->data->qpos,
        key_qpos
    );

    for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
        int root_dof = agent * REK_MATCH_QVEL_PER_AGENT;
        double residual[REK_SANDBOX_ROOT_DIMS];
        for (int axis = 0; axis < REK_SANDBOX_ROOT_DIMS; axis++) {
            double desired_acceleration = REK_SANDBOX_ROOT_KP[axis]
                    * tangent_error[root_dof + axis]
                - REK_SANDBOX_ROOT_KD[axis]
                    * env->data->qvel[root_dof + axis];
            residual[axis] = desired_acceleration
                - env->data->qacc[root_dof + axis];
        }

        for (int action = 0; action < REK_SANDBOX_ROOT_LEG_ACTIONS; action++) {
            double target_delta = 0.0;
            for (int axis = 0; axis < REK_SANDBOX_ROOT_DIMS; axis++) {
                target_delta += env->root_pinv[agent][action][axis]
                    * residual[axis];
            }
            double limit = REK_SANDBOX_ROOT_TARGET_DELTA_LIMIT
                * env->root_stabilizer_scale;
            target_delta = rek_sandbox_clip(
                target_delta * env->root_stabilizer_scale,
                -limit,
                limit
            );
            int actuator = agent * REK_SANDBOX_NUM_ACTIONS + action;
            env->data->ctrl[actuator] = rek_sandbox_limit_joint_target(
                env,
                agent,
                action,
                env->data->ctrl[actuator] + target_delta
            );
        }
    }
}

static float rek_sandbox_apply_learner_action(RekSandbox* env) {
    double square_sum = 0.0;
    for (int action = 0; action < REK_SANDBOX_NUM_ACTIONS; action++) {
        double value = env->actions[action];
        if (!isfinite(value)) value = 0.0;
        value = rek_sandbox_clip(value, -env->action_clip, env->action_clip);
        double target = rek_sandbox_keyframe_joint_target(env, 0, action)
            + env->action_scale * value;
        target = rek_sandbox_limit_joint_target(env, 0, action, target);
        env->data->ctrl[action] = target;
        square_sum += value * value;
    }
    return (float)sqrt(square_sum / REK_SANDBOX_NUM_ACTIONS);
}

static void rek_sandbox_apply_dummy_action(RekSandbox* env) {
    const double two_pi = 6.2831853071795864769;
    double phase = two_pi * env->dummy_frequency_hz * env->data->time;
    double wave = sin(phase);
    for (int action = 0; action < REK_SANDBOX_NUM_ACTIONS; action++) {
        int actuator = REK_SANDBOX_NUM_ACTIONS + action;
        double target = rek_sandbox_keyframe_joint_target(env, 1, action)
            + env->dummy_amplitude * REK_SANDBOX_DUMMY_WAVE[action] * wave;
        target = rek_sandbox_limit_joint_target(env, 1, action, target);
        env->data->ctrl[actuator] = target;
    }
}

static void rek_sandbox_reset_dummy(RekSandbox* env) {
    int qpos_start = REK_MATCH_QPOS_PER_AGENT;
    int qvel_start = REK_MATCH_QVEL_PER_AGENT;
    const mjtNum* key_qpos = env->model->key_qpos
        + REK_MATCH_KEYFRAME_ID * env->model->nq;
    const mjtNum* key_qvel = env->model->key_qvel
        + REK_MATCH_KEYFRAME_ID * env->model->nv;
    mju_copy(
        env->data->qpos + qpos_start,
        key_qpos + qpos_start,
        REK_MATCH_QPOS_PER_AGENT
    );
    mju_copy(
        env->data->qvel + qvel_start,
        key_qvel + qvel_start,
        REK_MATCH_QVEL_PER_AGENT
    );
    mju_zero(env->data->qacc + qvel_start, REK_MATCH_QVEL_PER_AGENT);
    rek_sandbox_apply_dummy_action(env);
    mj_forward(env->model, env->data);
    env->episode_dummy_resets += 1;
}

static void rek_sandbox_reset_state(RekSandbox* env) {
    mj_resetDataKeyframe(env->model, env->data, REK_MATCH_KEYFRAME_ID);
    env->tick = 0;
    env->episode_return = 0.0f;
    env->episode_action_rms_sum = 0.0f;
    env->episode_dummy_resets = 0;
    rek_sandbox_apply_dummy_action(env);
    for (int action = 0; action < REK_SANDBOX_NUM_ACTIONS; action++) {
        env->data->ctrl[action] = rek_sandbox_keyframe_joint_target(
            env, 0, action
        );
    }
    mj_forward(env->model, env->data);
    rek_sandbox_compute_observation(env);
}

void c_reset(RekSandbox* env) {
    if (!rek_sandbox_config_is_valid(env)) {
        rek_sandbox_fail("invalid curriculum configuration");
    }
    rek_sandbox_reset_state(env);
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
}

static float rek_sandbox_reward(
        const RekSandbox* env,
        float action_rms,
        int learner_fall,
        int invalid) {
    if (invalid) return -env->fall_penalty;
    double height = rek_sandbox_root_height(env, 0);
    double up_z = rek_sandbox_root_up_z(env, 0);
    double initial_height = env->model->key_qpos[2];
    double height_score = rek_sandbox_clip(
        (height - env->fall_height) / (initial_height - env->fall_height),
        0.0,
        1.0
    );
    double upright_score = rek_sandbox_clip(
        (up_z - env->fall_up_z) / (1.0 - env->fall_up_z),
        0.0,
        1.0
    );
    double reward = env->height_reward_weight * height_score
        + env->upright_reward_weight * upright_score
        - env->action_cost_weight * action_rms * action_rms;
    if (learner_fall) reward -= env->fall_penalty;
    return (float)reward;
}

static void rek_sandbox_add_log(
        RekSandbox* env,
        float action_rms,
        int learner_fall,
        int invalid,
        int timeout) {
    env->log.episode_return += env->episode_return;
    env->log.score += env->episode_return;
    env->log.episode_length += (float)env->tick;
    if (!invalid) {
        env->log.learner_root_height += (float)rek_sandbox_root_height(env, 0);
        env->log.learner_root_up_z += (float)rek_sandbox_root_up_z(env, 0);
    }
    env->log.action_rms += env->tick > 0
        ? env->episode_action_rms_sum / env->tick
        : action_rms;
    env->log.dummy_resets += (float)env->episode_dummy_resets;
    env->log.learner_fall += learner_fall ? 1.0f : 0.0f;
    env->log.invalid_termination += invalid ? 1.0f : 0.0f;
    env->log.timeout += timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

void c_step(RekSandbox* env) {
    float action_rms = rek_sandbox_apply_learner_action(env);
    rek_sandbox_apply_dummy_action(env);
    mj_forward(env->model, env->data);
    rek_sandbox_apply_root_stabilizer(env);
    mj_step(env->model, env->data);
    env->tick += 1;

    int invalid = !rek_sandbox_state_is_finite(env);
    int learner_fall = !invalid && rek_sandbox_fallen(env, 0);
    int timeout = env->tick >= env->max_steps;
    float reward = rek_sandbox_reward(env, action_rms, learner_fall, invalid);
    env->episode_return += reward;
    env->episode_action_rms_sum += action_rms;

    if (!invalid && !learner_fall && rek_sandbox_fallen(env, 1)) {
        rek_sandbox_reset_dummy(env);
    }

    if (invalid || learner_fall || timeout) {
        rek_sandbox_add_log(env, action_rms, learner_fall, invalid, timeout);
        rek_sandbox_reset_state(env);
        env->rewards[0] = reward;
        env->terminals[0] = 1.0f;
        return;
    }

    env->rewards[0] = reward;
    env->terminals[0] = 0.0f;
    rek_sandbox_compute_observation(env);
}

void c_render(RekSandbox* env) {
    (void)env;
}

void c_close(RekSandbox* env) {
    if (env->data != NULL) {
        mj_deleteData(env->data);
        env->data = NULL;
    }
    if (env->owns_model && env->model != NULL) {
        mj_deleteModel((mjModel*)env->model);
        env->model = NULL;
    }
}

static void rek_sandbox_close_shared_model(RekSandbox* envs) {
    if (envs == NULL || envs[0].model == NULL) return;
    const mjModel* shared_model = envs[0].model;
    int count = envs[0].shared_model_env_count;
    if (count <= 0) rek_sandbox_fail("invalid shared model environment count");

    for (int i = 0; i < count; i++) {
        if (envs[i].data != NULL || envs[i].owns_model
                || envs[i].model != shared_model) {
            rek_sandbox_fail("shared model lifecycle mismatch");
        }
    }
    mj_deleteModel((mjModel*)shared_model);
    for (int i = 0; i < count; i++) envs[i].model = NULL;
}
