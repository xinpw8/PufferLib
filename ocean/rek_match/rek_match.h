// Two-agent MuJoCo diagnostic for the hash-bound T800 arena artifact.
//
// Each physical environment contains two policy agents and one mjData. The
// model is immutable and may be shared by every physical environment. Each
// agent receives an ego-first raw qpos/qvel observation and controls only its
// own 25-actuator block. Rewards are identically zero.

#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mujoco/mujoco.h>

#include "../rek/rek_sha256.h"

#define REK_MATCH_NUM_AGENTS 2
#define REK_MATCH_NUM_BODIES 61
#define REK_MATCH_NUM_JOINTS 52
#define REK_MATCH_NUM_GEOMS 91
#define REK_MATCH_NQ 64
#define REK_MATCH_NV 62
#define REK_MATCH_NUM_ACTUATORS 50
#define REK_MATCH_QPOS_PER_AGENT 32
#define REK_MATCH_QVEL_PER_AGENT 31
#define REK_MATCH_ACTIONS_PER_AGENT 25
#define REK_MATCH_OBS_SIZE \
    (2 * (REK_MATCH_QPOS_PER_AGENT + REK_MATCH_QVEL_PER_AGENT))
#define REK_MATCH_MODEL_NAME "rek_t800_t800_arena_diagnostic"
#define REK_MATCH_KEYFRAME_NAME "client_observed_round2_first_active"
#define REK_MATCH_KEYFRAME_ID 0
#define REK_MATCH_EXPECTED_MJCF_SHA256 \
    "01caa6ed90277a90fc71b4c16d7959fb70f7702fa15bea7c68d3eaa9d5f27b2c"
#define REK_MATCH_ACT_SIZES { \
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
    1, 1, 1, 1, 1 \
}

static const char* const REK_MATCH_FIGHTER_PREFIXES[REK_MATCH_NUM_AGENTS] = {
    "fighter_0__",
    "fighter_1__",
};

static const char* const REK_MATCH_ROOT_BODY_SUFFIX = "LINK_BASE_3177";
static const char* const REK_MATCH_FREE_JOINT_SUFFIX =
    "joint__LINK_BASE_freejoint_3101";

static const char* const
REK_MATCH_ACTUATOR_SUFFIXES[REK_MATCH_ACTIONS_PER_AGENT] = {
    "motor_J00_HIP_PITCH_L_3349",
    "motor_J01_HIP_ROLL_L_2893",
    "motor_J02_HIP_YAW_L_3046",
    "motor_J03_KNEE_PITCH_L_3145",
    "motor_J04_ANKLE_PITCH_L_3070",
    "motor_J05_ANKLE_ROLL_L_3237",
    "motor_J06_HIP_PITCH_R_3163",
    "motor_J07_HIP_ROLL_R_3295",
    "motor_J08_HIP_YAW_R_3108",
    "motor_J09_KNEE_PITCH_R_2993",
    "motor_J10_ANKLE_PITCH_R_3180",
    "motor_J11_ANKLE_ROLL_R_3034",
    "motor_J12_TORSO_YAW_3129",
    "motor_J13_SHOULDER_PITCH_L_3205",
    "motor_J14_SHOULDER_ROLL_L_3178",
    "motor_J15_SHOULDER_YAW_L_3321",
    "motor_J16_ELBOW_PITCH_L_3225",
    "motor_J17_ELBOW_YAW_L_3216",
    "motor_J18_SHOULDER_PITCH_R_3402",
    "motor_J19_SHOULDER_ROLL_R_3276",
    "motor_J20_SHOULDER_YAW_R_3238",
    "motor_J21_ELBOW_PITCH_R_3165",
    "motor_J22_ELBOW_YAW_R_3195",
    "motor_J23_HEAD_PITCH_3147",
    "motor_J24_HEAD_YAW_3477",
};

static const char* const
REK_MATCH_JOINT_SUFFIXES[REK_MATCH_ACTIONS_PER_AGENT] = {
    "joint__J00_HIP_PITCH_L_3202",
    "joint__J01_HIP_ROLL_L_3156",
    "joint__J02_HIP_YAW_L_3403",
    "joint__J03_KNEE_PITCH_L_3448",
    "joint__J04_ANKLE_PITCH_L_3132",
    "joint__J05_ANKLE_ROLL_L_3431",
    "joint__J06_HIP_PITCH_R_3167",
    "joint__J07_HIP_ROLL_R_2962",
    "joint__J08_HIP_YAW_R_3013",
    "joint__J09_KNEE_PITCH_R_3252",
    "joint__J10_ANKLE_PITCH_R_3006",
    "joint__J11_ANKLE_ROLL_R_3208",
    "joint__J12_TORSO_YAW_3458",
    "joint__J13_SHOULDER_PITCH_L_3494",
    "joint__J14_SHOULDER_ROLL_L_3297",
    "joint__J15_SHOULDER_YAW_L_3493",
    "joint__J16_ELBOW_PITCH_L_3166",
    "joint__J17_ELBOW_YAW_L_2903",
    "joint__J18_SHOULDER_PITCH_R_2922",
    "joint__J19_SHOULDER_ROLL_R_3116",
    "joint__J20_SHOULDER_YAW_R_3028",
    "joint__J21_ELBOW_PITCH_R_3018",
    "joint__J22_ELBOW_YAW_R_3481",
    "joint__J23_HEAD_PITCH_3223",
    "joint__J24_HEAD_YAW_3294",
};

static const double
REK_MATCH_CTRL_RANGE_MAGNITUDES[REK_MATCH_ACTIONS_PER_AGENT] = {
    415.0, 370.0, 222.0, 415.0, 160.0, 160.0,
    415.0, 370.0, 222.0, 415.0, 160.0, 160.0,
    222.0,
    160.0, 160.0, 160.0, 160.0, 52.0,
    160.0, 160.0, 160.0, 160.0, 52.0, 52.0, 52.0,
};

// These tables are indexed by diagnostic action order, not compiled joint ID.
static const int
REK_MATCH_LOCAL_JOINT_IDS[REK_MATCH_ACTIONS_PER_AGENT] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 14, 15,
};

static const int
REK_MATCH_LOCAL_JOINT_QPOS_ADDRESSES[REK_MATCH_ACTIONS_PER_AGENT] = {
    7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 20, 21,
};

static const int
REK_MATCH_LOCAL_JOINT_DOF_ADDRESSES[REK_MATCH_ACTIONS_PER_AGENT] = {
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 19, 20,
};

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float mean_root_height;
    float max_abs_qvel;
    float invalid_termination;
    float timeout;
    float n;
} Log;

typedef struct RekMatch {
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
} RekMatch;

static void rek_match_fail(const char* message) {
    fprintf(stderr, "rek match diagnostic: %s\n", message);
    abort();
}

static int rek_match_partition_layout(
        int total_agents,
        int num_buffers,
        int* physical_envs_out,
        int* envs_per_buffer_out) {
    if (total_agents <= 0 || (total_agents % REK_MATCH_NUM_AGENTS) != 0
            || num_buffers <= 0) {
        return 0;
    }

    int physical_envs = total_agents / REK_MATCH_NUM_AGENTS;
    if ((physical_envs % num_buffers) != 0) return 0;

    if (physical_envs_out != NULL) *physical_envs_out = physical_envs;
    if (envs_per_buffer_out != NULL) {
        *envs_per_buffer_out = physical_envs / num_buffers;
    }
    return 1;
}

static int rek_match_model_dimensions_match(const mjModel* model) {
    return model != NULL
        && model->nbody == REK_MATCH_NUM_BODIES
        && model->njnt == REK_MATCH_NUM_JOINTS
        && model->ngeom == REK_MATCH_NUM_GEOMS
        && model->nq == REK_MATCH_NQ
        && model->nv == REK_MATCH_NV
        && model->nu == REK_MATCH_NUM_ACTUATORS;
}

static int rek_match_has_prefix(const char* value, const char* prefix) {
    return value != NULL
        && strncmp(value, prefix, strlen(prefix)) == 0;
}

static int rek_match_prefixed_name_matches(
        const char* actual,
        int agent,
        const char* suffix) {
    char expected[128];
    int count = snprintf(
        expected,
        sizeof(expected),
        "%s%s",
        REK_MATCH_FIGHTER_PREFIXES[agent],
        suffix
    );
    return count >= 0 && (size_t)count < sizeof(expected)
        && actual != NULL && strcmp(actual, expected) == 0;
}

static int rek_match_object_names_match(const mjModel* model) {
    for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
        int first_body = 1 + agent * 30;
        int last_body = first_body + 30;
        for (int body = first_body; body < last_body; body++) {
            const char* name = mj_id2name(model, mjOBJ_BODY, body);
            if (!rek_match_has_prefix(name, REK_MATCH_FIGHTER_PREFIXES[agent])
                    || mj_name2id(model, mjOBJ_BODY, name) != body) {
                return 0;
            }
        }

        int first_joint = agent * 26;
        int last_joint = first_joint + 26;
        for (int joint = first_joint; joint < last_joint; joint++) {
            const char* name = mj_id2name(model, mjOBJ_JOINT, joint);
            if (!rek_match_has_prefix(name, REK_MATCH_FIGHTER_PREFIXES[agent])
                    || mj_name2id(model, mjOBJ_JOINT, name) != joint) {
                return 0;
            }
        }

        for (int geom = 0; geom < model->ngeom; geom++) {
            int body = model->geom_bodyid[geom];
            if (body >= first_body && body < last_body) {
                const char* name = mj_id2name(model, mjOBJ_GEOM, geom);
                if (!rek_match_has_prefix(
                            name, REK_MATCH_FIGHTER_PREFIXES[agent])
                        || mj_name2id(model, mjOBJ_GEOM, name) != geom) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

static int rek_match_model_identity_matches(const mjModel* model) {
    const double tolerance = 1e-14;
    if (!rek_match_model_dimensions_match(model)) return 0;

    if (model->names == NULL
            || strcmp(model->names, REK_MATCH_MODEL_NAME) != 0
            || fabs(model->opt.timestep - 0.019999992913832199) > tolerance
            || fabs(model->opt.gravity[0]) > tolerance
            || fabs(model->opt.gravity[1]) > tolerance
            || fabs(model->opt.gravity[2] - (-9.8100004196166992)) > tolerance
            || model->opt.integrator != mjINT_IMPLICITFAST
            || model->opt.solver != mjSOL_NEWTON
            || model->opt.iterations != 100
            || model->nkey != 1
            || mj_name2id(model, mjOBJ_KEY, REK_MATCH_KEYFRAME_NAME)
                != REK_MATCH_KEYFRAME_ID) {
        return 0;
    }

    const char* keyframe_name = mj_id2name(
        model, mjOBJ_KEY, REK_MATCH_KEYFRAME_ID
    );
    if (keyframe_name == NULL
            || strcmp(keyframe_name, REK_MATCH_KEYFRAME_NAME) != 0
            || !rek_match_object_names_match(model)) {
        return 0;
    }

    for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
        int free_joint = agent * 26;
        int root_body = 1 + agent * 30;
        if (!rek_match_prefixed_name_matches(
                    mj_id2name(model, mjOBJ_JOINT, free_joint),
                    agent,
                    REK_MATCH_FREE_JOINT_SUFFIX)
                || !rek_match_prefixed_name_matches(
                    mj_id2name(model, mjOBJ_BODY, root_body),
                    agent,
                    REK_MATCH_ROOT_BODY_SUFFIX)
                || model->jnt_type[free_joint] != mjJNT_FREE
                || model->jnt_bodyid[free_joint] != root_body
                || model->jnt_qposadr[free_joint]
                    != agent * REK_MATCH_QPOS_PER_AGENT
                || model->jnt_dofadr[free_joint]
                    != agent * REK_MATCH_QVEL_PER_AGENT) {
            return 0;
        }

        for (int action = 0; action < REK_MATCH_ACTIONS_PER_AGENT; action++) {
            int actuator = agent * REK_MATCH_ACTIONS_PER_AGENT + action;
            int joint = agent * 26 + REK_MATCH_LOCAL_JOINT_IDS[action];
            int expected_qpos = agent * REK_MATCH_QPOS_PER_AGENT
                + REK_MATCH_LOCAL_JOINT_QPOS_ADDRESSES[action];
            int expected_dof = agent * REK_MATCH_QVEL_PER_AGENT
                + REK_MATCH_LOCAL_JOINT_DOF_ADDRESSES[action];
            double expected_limit = REK_MATCH_CTRL_RANGE_MAGNITUDES[action];

            if (!rek_match_prefixed_name_matches(
                        mj_id2name(model, mjOBJ_ACTUATOR, actuator),
                        agent,
                        REK_MATCH_ACTUATOR_SUFFIXES[action])
                    || !rek_match_prefixed_name_matches(
                        mj_id2name(model, mjOBJ_JOINT, joint),
                        agent,
                        REK_MATCH_JOINT_SUFFIXES[action])
                    || model->actuator_trntype[actuator] != mjTRN_JOINT
                    || model->actuator_trnid[2 * actuator] != joint
                    || model->jnt_type[joint] != mjJNT_HINGE
                    || model->jnt_qposadr[joint] != expected_qpos
                    || model->jnt_dofadr[joint] != expected_dof
                    || fabs(model->actuator_ctrlrange[2 * actuator]
                            + expected_limit) > tolerance
                    || fabs(model->actuator_ctrlrange[2 * actuator + 1]
                            - expected_limit) > tolerance
                    || model->actuator_ctrllimited[actuator] != 1) {
                return 0;
            }
        }
    }
    return 1;
}

static mjModel* rek_match_load_model(void) {
    const char* model_path = getenv("REK_MJCF_PATH");
    if (model_path == NULL || model_path[0] == '\0') {
        rek_match_fail("REK_MJCF_PATH is required");
    }

    char actual_sha256[65];
    if (!rek_sha256_file(model_path, actual_sha256)) {
        rek_match_fail("could not hash REK_MJCF_PATH");
    }
    if (strcmp(actual_sha256, REK_MATCH_EXPECTED_MJCF_SHA256) != 0) {
        fprintf(
            stderr,
            "rek match diagnostic: MJCF SHA-256 mismatch expected=%s actual=%s\n",
            REK_MATCH_EXPECTED_MJCF_SHA256,
            actual_sha256
        );
        abort();
    }

    char error[1024] = {0};
    mjModel* model = mj_loadXML(model_path, NULL, error, (int)sizeof(error));
    if (model == NULL) {
        fprintf(stderr, "rek match diagnostic: mj_loadXML failed: %s\n", error);
        abort();
    }
    if (!rek_match_model_identity_matches(model)) {
        mj_deleteModel(model);
        rek_match_fail("model identity validation failed");
    }
    return model;
}

static void rek_match_init_with_model(
        RekMatch* env,
        const mjModel* model,
        int owns_model) {
    if (env == NULL || model == NULL) {
        rek_match_fail("environment and model are required");
    }
    env->model = model;
    env->owns_model = owns_model;
    env->shared_model_env_count = 1;
    env->data = mj_makeData(model);
    if (env->data == NULL) rek_match_fail("mj_makeData failed");
}

static void rek_match_init(RekMatch* env) {
    rek_match_init_with_model(env, rek_match_load_model(), 1);
}

static int rek_match_state_is_finite(const RekMatch* env) {
    for (int i = 0; i < env->model->nq; i++) {
        if (!isfinite(env->data->qpos[i])) return 0;
    }
    for (int i = 0; i < env->model->nv; i++) {
        if (!isfinite(env->data->qvel[i])
                || !isfinite(env->data->qacc[i])) {
            return 0;
        }
    }
    return isfinite(env->data->time);
}

static float rek_match_max_abs_qvel(const RekMatch* env) {
    float result = 0.0f;
    for (int i = 0; i < env->model->nv; i++) {
        float value = (float)fabs(env->data->qvel[i]);
        if (value > result) result = value;
    }
    return result;
}

static void rek_match_compute_agent_observation(RekMatch* env, int observer) {
    int opponent = 1 - observer;
    int qpos_offsets[2] = {
        observer * REK_MATCH_QPOS_PER_AGENT,
        opponent * REK_MATCH_QPOS_PER_AGENT,
    };
    int qvel_offsets[2] = {
        observer * REK_MATCH_QVEL_PER_AGENT,
        opponent * REK_MATCH_QVEL_PER_AGENT,
    };
    float* output = env->observations + observer * REK_MATCH_OBS_SIZE;
    int cursor = 0;
    for (int side = 0; side < 2; side++) {
        for (int i = 0; i < REK_MATCH_QPOS_PER_AGENT; i++) {
            output[cursor++] = (float)env->data->qpos[qpos_offsets[side] + i];
        }
        for (int i = 0; i < REK_MATCH_QVEL_PER_AGENT; i++) {
            output[cursor++] = (float)env->data->qvel[qvel_offsets[side] + i];
        }
    }
}

static void rek_match_compute_observations(RekMatch* env) {
    for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
        rek_match_compute_agent_observation(env, agent);
    }
}

static void rek_match_reset_state(RekMatch* env) {
    mj_resetDataKeyframe(env->model, env->data, REK_MATCH_KEYFRAME_ID);
    mj_forward(env->model, env->data);
    env->tick = 0;
    env->episode_return = 0.0f;
    rek_match_compute_observations(env);
}

void c_reset(RekMatch* env) {
    rek_match_reset_state(env);
    for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
        env->rewards[agent] = 0.0f;
        env->terminals[agent] = 0.0f;
    }
}

static double rek_match_normalize_action(float value) {
    if (!isfinite(value)) return 0.0;
    return tanh((double)value);
}

static void rek_match_apply_agent_actions(RekMatch* env, int agent) {
    if (agent < 0 || agent >= REK_MATCH_NUM_AGENTS) return;
    int action_start = agent * REK_MATCH_ACTIONS_PER_AGENT;
    int control_start = agent * REK_MATCH_ACTIONS_PER_AGENT;
    for (int i = 0; i < REK_MATCH_ACTIONS_PER_AGENT; i++) {
        int actuator = control_start + i;
        double normalized = rek_match_normalize_action(
            env->actions[action_start + i]
        );
        double lower = env->model->actuator_ctrlrange[2 * actuator];
        double upper = env->model->actuator_ctrlrange[2 * actuator + 1];
        env->data->ctrl[actuator] = 0.5 * (lower + upper)
            + 0.5 * normalized * (upper - lower);
    }
}

static void rek_match_apply_actions(RekMatch* env) {
    for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
        rek_match_apply_agent_actions(env, agent);
    }
}

static void rek_match_add_log(RekMatch* env, int invalid, int timeout) {
    env->log.perf += 0.0f;
    env->log.score += 0.0f;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->tick;
    env->log.mean_root_height += invalid ? 0.0f
        : 0.5f * (float)(env->data->qpos[2] + env->data->qpos[34]);
    env->log.max_abs_qvel += invalid ? 0.0f
        : rek_match_max_abs_qvel(env);
    env->log.invalid_termination += invalid ? 1.0f : 0.0f;
    env->log.timeout += timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

void c_step(RekMatch* env) {
    rek_match_apply_actions(env);
    mj_step(env->model, env->data);
    env->tick += 1;

    int invalid = !rek_match_state_is_finite(env);
    int timeout = env->max_steps > 0 && env->tick >= env->max_steps;
    float terminal = (invalid || timeout) ? 1.0f : 0.0f;
    for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
        env->rewards[agent] = 0.0f;
        env->terminals[agent] = terminal;
    }

    if (invalid || timeout) {
        rek_match_add_log(env, invalid, timeout);
        rek_match_reset_state(env);
        for (int agent = 0; agent < REK_MATCH_NUM_AGENTS; agent++) {
            env->rewards[agent] = 0.0f;
            env->terminals[agent] = 1.0f;
        }
        return;
    }
    rek_match_compute_observations(env);
}

void c_render(RekMatch* env) {
    (void)env;
}

void c_close(RekMatch* env) {
    if (env->data != NULL) {
        mj_deleteData(env->data);
        env->data = NULL;
    }
    if (env->owns_model && env->model != NULL) {
        mj_deleteModel((mjModel*)env->model);
        env->model = NULL;
    }
}

static void rek_match_close_shared_model(RekMatch* envs) {
    if (envs == NULL || envs[0].model == NULL) return;
    const mjModel* shared_model = envs[0].model;
    int count = envs[0].shared_model_env_count;
    if (count <= 0) rek_match_fail("invalid shared model environment count");

    for (int i = 0; i < count; i++) {
        if (envs[i].data != NULL || envs[i].owns_model
                || envs[i].model != shared_model) {
            rek_match_fail("shared model lifecycle mismatch");
        }
    }
    mj_deleteModel((mjModel*)shared_model);
    for (int i = 0; i < count; i++) envs[i].model = NULL;
}
