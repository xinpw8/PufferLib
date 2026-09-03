// Dynamics-only Puffer environment for measured REK MuJoCo plants.
//
// This deliberately contains no inferred combat reward, damage, move, network,
// opponent, or termination mechanics. It only exposes MuJoCo qpos/qvel and
// accepts latent direct-motor controls for plant integration tests. A tanh
// transform maps each finite policy action to the measured actuator range.

#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mujoco/mujoco.h>

#include "rek_sha256.h"

#ifdef REK_PROFILE_G1
#define REK_NUM_BODIES 31
#define REK_NUM_JOINTS 30
#define REK_NUM_GEOMS 37
#define REK_NQ 36
#define REK_NV 35
#define REK_NUM_ACTUATORS 29
#define REK_PLANT_ID "g1_29dof.recovered.v7"
#define REK_EXPECTED_MJCF_SHA256 \
    "811fdc1e5bee74026b780974207cbcd628cdd83a249d3f76b75a668d71aad835"
#define REK_EXPECTED_CTRL_LIMITED 0
#define REK_ACT_SIZES { \
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
    1, 1, 1, 1, 1, 1, 1, 1, 1 \
}

static const char* const REK_EXPECTED_ACTUATOR_NAMES[REK_NUM_ACTUATORS] = {
    "left_hip_pitch_3206",
    "left_hip_roll_3164",
    "left_hip_yaw_2984",
    "left_knee_3031",
    "left_ankle_pitch_3234",
    "left_ankle_roll_2975",
    "right_hip_pitch_3479",
    "right_hip_roll_3421",
    "right_hip_yaw_2949",
    "right_knee_3114",
    "right_ankle_pitch_3119",
    "right_ankle_roll_3089",
    "waist_yaw_3343",
    "waist_roll_3272",
    "waist_pitch_3466",
    "left_shoulder_pitch_3146",
    "left_shoulder_roll_3230",
    "left_shoulder_yaw_3150",
    "left_elbow_3483",
    "left_wrist_roll_3213",
    "left_wrist_pitch_3488",
    "left_wrist_yaw_3009",
    "right_shoulder_pitch_2971",
    "right_shoulder_roll_3289",
    "right_shoulder_yaw_2966",
    "right_elbow_3486",
    "right_wrist_roll_3478",
    "right_wrist_pitch_2990",
    "right_wrist_yaw_2973",
};

static const double REK_EXPECTED_CTRL_RANGE_MAGNITUDES[REK_NUM_ACTUATORS] = {
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 50.0, 50.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
};
#else
#define REK_NUM_BODIES 31
#define REK_NUM_JOINTS 26
#define REK_NUM_GEOMS 54
#define REK_NQ 32
#define REK_NV 31
#define REK_NUM_ACTUATORS 25
#define REK_PLANT_ID "t800_factory_arena.recovered.v1"
#define REK_EXPECTED_MJCF_SHA256 \
    "0a5fb688156fb57474056470c78e2209ebfff4e09e3935b73e3375c28d33ba93"
#define REK_EXPECTED_CTRL_LIMITED 1
#define REK_ACT_SIZES { \
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
    1, 1, 1, 1, 1 \
}

static const char* const REK_EXPECTED_ACTUATOR_NAMES[REK_NUM_ACTUATORS] = {
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

static const double REK_EXPECTED_CTRL_RANGE_MAGNITUDES[REK_NUM_ACTUATORS] = {
    415.0, 370.0, 222.0, 415.0, 160.0, 160.0,
    415.0, 370.0, 222.0, 415.0, 160.0, 160.0,
    222.0,
    160.0, 160.0, 160.0, 160.0, 52.0,
    160.0, 160.0, 160.0, 160.0, 52.0, 52.0, 52.0,
};
#endif

#define REK_OBS_SIZE (REK_NQ + REK_NV)

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float root_height;
    float max_abs_qvel;
    float invalid_termination;
    float timeout;
    float n;
} Log;

typedef struct Rek {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    Log log;

    mjModel* model;
    mjData* data;
    int owns_model;
    int tick;
    int max_steps;
    float episode_return;
} Rek;

static void rek_fail(const char* message) {
    fprintf(stderr, "rek plant environment: %s\n", message);
    abort();
}

static int rek_model_dimensions_match(const mjModel* model) {
    return model->nbody == REK_NUM_BODIES
        && model->njnt == REK_NUM_JOINTS
        && model->ngeom == REK_NUM_GEOMS
        && model->nq == REK_NQ
        && model->nv == REK_NV
        && model->nu == REK_NUM_ACTUATORS;
}

static int rek_model_identity_matches(const mjModel* model) {
    const double tolerance = 1e-14;
    if (fabs(model->opt.timestep - 0.019999992913832199) > tolerance
            || fabs(model->opt.gravity[0]) > tolerance
            || fabs(model->opt.gravity[1]) > tolerance
            || fabs(model->opt.gravity[2] - (-9.8100004196166992)) > tolerance
            || model->opt.integrator != mjINT_IMPLICITFAST
            || model->opt.solver != mjSOL_NEWTON
            || model->opt.iterations != 100) {
        fprintf(stderr, "rek plant environment: global MuJoCo settings mismatch\n");
        return 0;
    }

    for (int i = 0; i < REK_NUM_ACTUATORS; i++) {
        const char* actual_name = mj_id2name(model, mjOBJ_ACTUATOR, i);
        if (actual_name == NULL
                || strcmp(actual_name, REK_EXPECTED_ACTUATOR_NAMES[i]) != 0) {
            fprintf(
                stderr,
                "rek plant environment: actuator %d name mismatch: expected=%s actual=%s\n",
                i,
                REK_EXPECTED_ACTUATOR_NAMES[i],
                actual_name == NULL ? "<null>" : actual_name
            );
            return 0;
        }

        double expected_limit = REK_EXPECTED_CTRL_RANGE_MAGNITUDES[i];
        double actual_lower = model->actuator_ctrlrange[2 * i];
        double actual_upper = model->actuator_ctrlrange[2 * i + 1];
        if (fabs(actual_lower + expected_limit) > tolerance
                || fabs(actual_upper - expected_limit) > tolerance) {
            fprintf(
                stderr,
                "rek plant environment: actuator %d control range mismatch: "
                "expected=[%.17g,%.17g] actual=[%.17g,%.17g]\n",
                i,
                -expected_limit,
                expected_limit,
                actual_lower,
                actual_upper
            );
            return 0;
        }
        if ((int)model->actuator_ctrllimited[i] != REK_EXPECTED_CTRL_LIMITED) {
            fprintf(
                stderr,
                "rek plant environment: actuator %d ctrllimited mismatch: "
                "expected=%d actual=%d\n",
                i,
                REK_EXPECTED_CTRL_LIMITED,
                (int)model->actuator_ctrllimited[i]
            );
            return 0;
        }
    }
    return 1;
}

static mjModel* rek_load_model(void) {
    const char* model_path = getenv("REK_MJCF_PATH");
    if (model_path == NULL || model_path[0] == '\0') {
        rek_fail("REK_MJCF_PATH is required");
    }

    char actual_sha256[65];
    if (!rek_sha256_file(model_path, actual_sha256)) {
        rek_fail("could not hash REK_MJCF_PATH");
    }
    if (strcmp(actual_sha256, REK_EXPECTED_MJCF_SHA256) != 0) {
        fprintf(
            stderr,
            "rek plant environment: MJCF SHA-256 mismatch expected=%s actual=%s\n",
            REK_EXPECTED_MJCF_SHA256,
            actual_sha256
        );
        abort();
    }

    char error[1024] = {0};
    mjModel* model = mj_loadXML(model_path, NULL, error, (int)sizeof(error));
    if (model == NULL) {
        fprintf(stderr, "rek plant environment: mj_loadXML failed: %s\n", error);
        abort();
    }
    if (!rek_model_dimensions_match(model)) {
        fprintf(
            stderr,
            "rek plant environment: unexpected model dimensions "
            "nbody=%lld njnt=%lld ngeom=%lld nq=%lld nv=%lld nu=%lld\n",
            (long long)model->nbody,
            (long long)model->njnt,
            (long long)model->ngeom,
            (long long)model->nq,
            (long long)model->nv,
            (long long)model->nu
        );
        mj_deleteModel(model);
        abort();
    }
    if (!rek_model_identity_matches(model)) {
        mj_deleteModel(model);
        rek_fail("model identity validation failed");
    }
    return model;
}

static void rek_init_with_model(Rek* env, mjModel* model, int owns_model) {
    env->model = model;
    env->owns_model = owns_model;
    env->data = mj_makeData(env->model);
    if (env->data == NULL) {
        rek_fail("mj_makeData failed");
    }
}

static void rek_init(Rek* env) {
    rek_init_with_model(env, rek_load_model(), 1);
}

static int rek_state_is_finite(const Rek* env) {
    for (int i = 0; i < env->model->nq; i++) {
        if (!isfinite(env->data->qpos[i])) return 0;
    }
    for (int i = 0; i < env->model->nv; i++) {
        if (!isfinite(env->data->qvel[i]) || !isfinite(env->data->qacc[i])) return 0;
    }
    return isfinite(env->data->time);
}

static float rek_max_abs_qvel(const Rek* env) {
    float result = 0.0f;
    for (int i = 0; i < env->model->nv; i++) {
        float value = (float)fabs(env->data->qvel[i]);
        if (value > result) result = value;
    }
    return result;
}

static void rek_compute_observations(Rek* env) {
    for (int i = 0; i < env->model->nq; i++) {
        env->observations[i] = (float)env->data->qpos[i];
    }
    for (int i = 0; i < env->model->nv; i++) {
        env->observations[env->model->nq + i] = (float)env->data->qvel[i];
    }
}

static void rek_reset_state(Rek* env) {
    mj_resetData(env->model, env->data);
    mj_forward(env->model, env->data);
    env->tick = 0;
    env->episode_return = 0.0f;
    rek_compute_observations(env);
}

void c_reset(Rek* env) {
    rek_reset_state(env);
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
}

static double rek_normalize_action(float value) {
    if (!isfinite(value)) return 0.0;
    return tanh((double)value);
}

static void rek_apply_actions(Rek* env) {
    for (int i = 0; i < env->model->nu; i++) {
        double normalized = rek_normalize_action(env->actions[i]);
        double lower = env->model->actuator_ctrlrange[2 * i];
        double upper = env->model->actuator_ctrlrange[2 * i + 1];
        env->data->ctrl[i] = 0.5 * (lower + upper)
            + 0.5 * normalized * (upper - lower);
    }
}

static void rek_add_log(Rek* env, int invalid, int timeout) {
    env->log.perf += 0.0f;
    env->log.score += 0.0f;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->tick;
    env->log.root_height += invalid ? 0.0f : (float)env->data->qpos[2];
    env->log.max_abs_qvel += invalid ? 0.0f : rek_max_abs_qvel(env);
    env->log.invalid_termination += invalid ? 1.0f : 0.0f;
    env->log.timeout += timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

void c_step(Rek* env) {
    rek_apply_actions(env);
    mj_step(env->model, env->data);
    env->tick += 1;

    int invalid = !rek_state_is_finite(env);
    int timeout = env->max_steps > 0 && env->tick >= env->max_steps;
    env->rewards[0] = 0.0f;
    env->terminals[0] = (invalid || timeout) ? 1.0f : 0.0f;
    env->episode_return += env->rewards[0];

    if (invalid || timeout) {
        float terminal = env->terminals[0];
        float reward = env->rewards[0];
        rek_add_log(env, invalid, timeout);
        rek_reset_state(env);
        env->terminals[0] = terminal;
        env->rewards[0] = reward;
        return;
    }
    rek_compute_observations(env);
}

void c_render(Rek* env) {
    (void)env;
}

void c_close(Rek* env) {
    if (env->data != NULL) {
        mj_deleteData(env->data);
        env->data = NULL;
    }
    if (env->owns_model && env->model != NULL) {
        mj_deleteModel(env->model);
        env->model = NULL;
    }
}
