// High-level T800 fight environment for PufferLib CUDA training.
//
// Physics, actuator limits, PdStand gains, keyboard action encoding, move
// slots, impact times, and limb ids come from the pinned Steam client.
// T800 ONNX/trajectory payloads are absent in that client, so locomotion is a
// reduced-order executor: keyframe PdStand plus measured walk/strafe/yaw
// speeds. Canned moves use measured timing and limb ids; joint motion is a
// distal-limb reach, not a recovered FactoryPolicy clip. Replace that reach
// with fitted visual trajectories when six-move windows exist.
//
// PROVISIONAL: the root tracker, distal-limb reach, contact-to-hit rule,
// fall/recovery transition timing, rewards, terminal/reset semantics, and the
// second policy-controlled opponent have not passed the REK variance gate.

#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mujoco/mujoco.h>

#include "../rek_strategy/strategy_router.h"

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

#define REK_FIGHT_NUM_AGENTS 2
#define REK_FIGHT_NUM_ACTIONS 4
#define REK_FIGHT_ACT_SIZES REK_STRATEGY_ACT_SIZES
#define REK_FIGHT_MOVE_MASK_SIZE REK_STRATEGY_MOVE_CATEGORIES
#define REK_FIGHT_AGENT_STATE_OBS_SIZE 22
#define REK_FIGHT_GLOBAL_STATE_OBS_SIZE 3
#define REK_FIGHT_OBS_SIZE (REK_MATCH_OBS_SIZE \
    + 2 * REK_FIGHT_AGENT_STATE_OBS_SIZE + REK_FIGHT_GLOBAL_STATE_OBS_SIZE)
#define REK_FIGHT_JOINTS 25
#define REK_FIGHT_LIMBS 4
#define REK_FIGHT_MAX_IMPACTS 3
#define REK_FIGHT_MAX_LIMB_GEOMS 16
#define REK_FIGHT_ROOT_LEGS 12
#define REK_FIGHT_ROOT_DIMS 6
#define REK_FIGHT_MUJOCO_VERSION 3007000
#define REK_FIGHT_DT 0.019999992913832199
#define REK_FIGHT_FORWARD_SPEED 0.800000011920929
#define REK_FIGHT_STRAFE_SPEED 0.4339999854564667
#define REK_FIGHT_YAW_SPEED 1.5
#define REK_FIGHT_COOLDOWN_S 0.5
#define REK_FIGHT_RECOVERY_S 1.257999986410141
#define REK_FIGHT_HIT_REWARD 1.0f
#define REK_FIGHT_FALL_REWARD 5.0f
#define REK_FIGHT_TIME_COST 0.001f

static const double REK_FIGHT_PD_KP[REK_FIGHT_JOINTS] = {
    1080.0, 480.0, 880.0, 1000.0, 800.0, 100.0,
    1080.0, 480.0, 880.0, 1000.0, 800.0, 100.0,
    200.0,
    120.0, 120.0, 120.0, 120.0, 120.0,
    120.0, 120.0, 120.0, 120.0, 120.0, 100.0, 100.0,
};

static const double REK_FIGHT_PD_KD[REK_FIGHT_JOINTS] = {
    25.0, 25.0, 25.0, 25.0, 2.0, 2.0,
    25.0, 25.0, 25.0, 25.0, 2.0, 2.0,
    1.0,
    1.0, 1.0, 1.0, 1.0, 0.200000002980232,
    1.0, 1.0, 1.0, 1.0, 0.200000002980232, 1.0, 1.0,
};

static const double REK_FIGHT_ROOT_KP[REK_FIGHT_ROOT_DIMS] = {
    120.721216799542, 120.721216799542, 44.972098219558,
    388.103557643136, 388.103557643136, 0.0,
};

static const double REK_FIGHT_ROOT_KD[REK_FIGHT_ROOT_DIMS] = {
    20.256756785599, 20.256756785599, 1.553386758450,
    27.141501179662, 27.141501179662, 0.127017625339,
};

// Distal joint indices in actuator order for limbs 1..4 (L arm, R arm, L leg, R leg).
static const int REK_FIGHT_LIMB_JOINTS[REK_FIGHT_LIMBS][5] = {
    {13, 14, 15, 16, 17},
    {18, 19, 20, 21, 22},
    {0, 1, 2, 3, 4},
    {6, 7, 8, 9, 10},
};

static const int REK_FIGHT_LIMB_JOINT_COUNTS[REK_FIGHT_LIMBS] = {5, 5, 5, 5};

typedef struct RekFightMove {
    int slot;
    int limb;
    int impact_count;
    float duration_s;
    float impact_s[REK_FIGHT_MAX_IMPACTS];
    float lead_s[REK_FIGHT_MAX_IMPACTS];
    float release_s[REK_FIGHT_MAX_IMPACTS];
} RekFightMove;

// Measured from T800 RobotConfig move objects. duration = last impact + release + blend_out.
static const RekFightMove REK_FIGHT_MOVES[6] = {
    {2, 1, 3, 2.0099999904632568f,
        {0.7599999904632568f, 1.149999976158142f, 1.809999942779541f},
        {0.10000000149011612f, 0.10000000149011612f, 0.10000000149011612f},
        {0.1899999976158142f, 0.10000000149011612f, 0.10000000149011612f}},
    {3, 4, 1, 1.5100000143051148f,
        {1.1100000143051147f, 0, 0},
        {0.25f, 0, 0},
        {0.30000001192092896f, 0, 0}},
    {4, 1, 1, 0.5699999928474426f,
        {0.38999998569488525f, 0, 0},
        {0.11999999731779099f, 0, 0},
        {0.07999999821186066f, 0, 0}},
    {5, 2, 1, 0.5200000107288361f,
        {0.2199999988079071f, 0, 0},
        {0.11999999731779099f, 0, 0},
        {0.20000000298023224f, 0, 0}},
    {9, 2, 1, 0.20000000298023224f,
        {0.0f, 0, 0},
        {0.0f, 0, 0},
        {0.0f, 0, 0}},
    {10, 3, 1, 1.350000023841858f,
        {1.100000023841858f, 0, 0},
        {0.20000000298023224f, 0, 0},
        {0.15000000596046448f, 0, 0}},
};

typedef struct Log {
    float score;
    float episode_return;
    float episode_length;
    float hits;
    float falls;
    float invalid_termination;
    float timeout;
    float n;
} Log;

typedef struct RekFightAgent {
    RekStrategyRouter router;
    int recovering;
    int move_in_progress;
    int cooldown_active;
    int move_slot;
    int move_ticks;
    int move_duration_ticks;
    int cooldown_ticks;
    int recovery_ticks;
    int scored_impacts;
    int hits;
    int fallen;
} RekFightAgent;

typedef struct RekFight {
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
    float fall_height;
    float fall_up_z;
    float root_stabilizer_scale;
    float episode_return[REK_FIGHT_NUM_AGENTS];

    RekFightAgent agent[REK_FIGHT_NUM_AGENTS];
    double root_pinv[REK_FIGHT_NUM_AGENTS][REK_FIGHT_ROOT_LEGS][REK_FIGHT_ROOT_DIMS];
    int limb_geom_count[REK_FIGHT_NUM_AGENTS][REK_FIGHT_LIMBS];
    int limb_geoms[REK_FIGHT_NUM_AGENTS][REK_FIGHT_LIMBS][REK_FIGHT_MAX_LIMB_GEOMS];
} RekFight;

static void rek_fight_fail(const char* message) {
    fprintf(stderr, "rek_fight: %s\n", message);
    abort();
}

static double rek_fight_clip(double value, double lower, double upper) {
    if (value < lower) return lower;
    if (value > upper) return upper;
    return value;
}

static int rek_fight_bin(float value) {
    int bin = (int)lrintf(value);
    if (bin < 0) bin = 0;
    if (bin > 2) bin = 2;
    return bin;
}

static const RekFightMove* rek_fight_move_by_slot(int slot) {
    for (int i = 0; i < 6; i++) {
        if (REK_FIGHT_MOVES[i].slot == slot) return &REK_FIGHT_MOVES[i];
    }
    return NULL;
}

static double rek_fight_keyframe_joint(const RekFight* env, int agent, int action) {
    int qpos = agent * REK_MATCH_QPOS_PER_AGENT
        + REK_MATCH_LOCAL_JOINT_QPOS_ADDRESSES[action];
    return env->model->key_qpos[REK_MATCH_KEYFRAME_ID * env->model->nq + qpos];
}

static double rek_fight_limit_joint(
        const RekFight* env, int agent, int action, double target) {
    int joint = agent * 26 + REK_MATCH_LOCAL_JOINT_IDS[action];
    if (env->model->jnt_limited[joint]) {
        target = rek_fight_clip(
            target,
            env->model->jnt_range[2 * joint],
            env->model->jnt_range[2 * joint + 1]
        );
    }
    return target;
}

static int rek_fight_root_body(int agent) {
    return 1 + agent * 30;
}

static double rek_fight_root_height(const RekFight* env, int agent) {
    return env->data->xpos[3 * rek_fight_root_body(agent) + 2];
}

static double rek_fight_root_up_z(const RekFight* env, int agent) {
    return env->data->xmat[9 * rek_fight_root_body(agent) + 8];
}

static int rek_fight_fallen(const RekFight* env, int agent) {
    return rek_fight_root_height(env, agent) < env->fall_height
        || rek_fight_root_up_z(env, agent) < env->fall_up_z;
}

static int rek_fight_move_action_available(
        const RekFightAgent* state, int move_category) {
    if (move_category == 0) return 1;
    return !state->recovering
        && !state->move_in_progress
        && !state->cooldown_active
        && move_category != state->router.last_emitted_move_category;
}

static int rek_fight_inverse_6x6(const double input[36], double output[36]) {
    double augmented[REK_FIGHT_ROOT_DIMS][2 * REK_FIGHT_ROOT_DIMS];
    for (int row = 0; row < REK_FIGHT_ROOT_DIMS; row++) {
        for (int column = 0; column < REK_FIGHT_ROOT_DIMS; column++) {
            augmented[row][column] = input[row * REK_FIGHT_ROOT_DIMS + column];
            augmented[row][REK_FIGHT_ROOT_DIMS + column] = row == column ? 1.0 : 0.0;
        }
    }
    for (int column = 0; column < REK_FIGHT_ROOT_DIMS; column++) {
        int pivot_row = column;
        double pivot_magnitude = fabs(augmented[pivot_row][column]);
        for (int row = column + 1; row < REK_FIGHT_ROOT_DIMS; row++) {
            double magnitude = fabs(augmented[row][column]);
            if (magnitude > pivot_magnitude) {
                pivot_row = row;
                pivot_magnitude = magnitude;
            }
        }
        if (!(pivot_magnitude > 1e-18) || !isfinite(pivot_magnitude)) return 0;
        if (pivot_row != column) {
            for (int item = 0; item < 2 * REK_FIGHT_ROOT_DIMS; item++) {
                double temporary = augmented[column][item];
                augmented[column][item] = augmented[pivot_row][item];
                augmented[pivot_row][item] = temporary;
            }
        }
        double pivot = augmented[column][column];
        for (int item = 0; item < 2 * REK_FIGHT_ROOT_DIMS; item++) {
            augmented[column][item] /= pivot;
        }
        for (int row = 0; row < REK_FIGHT_ROOT_DIMS; row++) {
            if (row == column) continue;
            double factor = augmented[row][column];
            for (int item = 0; item < 2 * REK_FIGHT_ROOT_DIMS; item++) {
                augmented[row][item] -= factor * augmented[column][item];
            }
        }
    }
    for (int row = 0; row < REK_FIGHT_ROOT_DIMS; row++) {
        for (int column = 0; column < REK_FIGHT_ROOT_DIMS; column++) {
            output[row * REK_FIGHT_ROOT_DIMS + column]
                = augmented[row][REK_FIGHT_ROOT_DIMS + column];
        }
    }
    return 1;
}

static void rek_fight_configure_actuators(mjModel* model) {
    for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
        for (int action = 0; action < REK_FIGHT_JOINTS; action++) {
            int actuator = agent * REK_FIGHT_JOINTS + action;
            double force_limit = REK_MATCH_CTRL_RANGE_MAGNITUDES[action];
            mjtNum* gain = model->actuator_gainprm + actuator * mjNGAIN;
            mjtNum* bias = model->actuator_biasprm + actuator * mjNBIAS;
            mju_zero(gain, mjNGAIN);
            mju_zero(bias, mjNBIAS);
            model->actuator_gaintype[actuator] = mjGAIN_FIXED;
            model->actuator_biastype[actuator] = mjBIAS_AFFINE;
            gain[0] = REK_FIGHT_PD_KP[action];
            bias[1] = -REK_FIGHT_PD_KP[action];
            bias[2] = -REK_FIGHT_PD_KD[action];
            model->actuator_ctrllimited[actuator] = 0;
            model->actuator_forcelimited[actuator] = 1;
            model->actuator_forcerange[2 * actuator] = -force_limit;
            model->actuator_forcerange[2 * actuator + 1] = force_limit;
        }
    }
}

static int rek_fight_body_descends_from_any(
        const mjModel* model,
        int body,
        const int* ancestors,
        int ancestor_count) {
    while (body > 0) {
        for (int item = 0; item < ancestor_count; item++) {
            if (body == ancestors[item]) return 1;
        }
        body = model->body_parentid[body];
    }
    return 0;
}

static void rek_fight_collect_limb_geoms(RekFight* env) {
    for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
        for (int limb = 0; limb < REK_FIGHT_LIMBS; limb++) {
            int bodies[8];
            int body_count = 0;
            for (int item = 0; item < REK_FIGHT_LIMB_JOINT_COUNTS[limb]; item++) {
                int action = REK_FIGHT_LIMB_JOINTS[limb][item];
                int joint = agent * 26 + REK_MATCH_LOCAL_JOINT_IDS[action];
                bodies[body_count++] = env->model->jnt_bodyid[joint];
            }
            int geom_count = 0;
            for (int geom = 0; geom < env->model->ngeom; geom++) {
                int body = env->model->geom_bodyid[geom];
                if (rek_fight_body_descends_from_any(
                            env->model, body, bodies, body_count)) {
                    if (geom_count >= REK_FIGHT_MAX_LIMB_GEOMS) {
                        rek_fight_fail("limb geometry capacity is too small");
                    }
                    env->limb_geoms[agent][limb][geom_count++] = geom;
                }
            }
            env->limb_geom_count[agent][limb] = geom_count;
        }
    }
}

static void rek_fight_set_agent_keyframe_controls(RekFight* env, int agent) {
    for (int action = 0; action < REK_FIGHT_JOINTS; action++) {
        env->data->ctrl[agent * REK_FIGHT_JOINTS + action]
            = rek_fight_keyframe_joint(env, agent, action);
    }
}

static void rek_fight_set_keyframe_controls(RekFight* env) {
    for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
        rek_fight_set_agent_keyframe_controls(env, agent);
    }
}

static void rek_fight_init_root_controller(RekFight* env) {
    double response[REK_FIGHT_ROOT_DIMS][REK_FIGHT_ROOT_LEGS];
    mjtNum base_qacc[REK_MATCH_NV];
    mj_resetDataKeyframe(env->model, env->data, REK_MATCH_KEYFRAME_ID);
    rek_fight_set_keyframe_controls(env);
    mj_forward(env->model, env->data);
    mju_copy(base_qacc, env->data->qacc, REK_MATCH_NV);

    for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
        int root_dof = agent * REK_MATCH_QVEL_PER_AGENT;
        for (int action = 0; action < REK_FIGHT_ROOT_LEGS; action++) {
            int actuator = agent * REK_FIGHT_JOINTS + action;
            env->data->ctrl[actuator] += 1e-4;
            mj_forward(env->model, env->data);
            for (int axis = 0; axis < REK_FIGHT_ROOT_DIMS; axis++) {
                response[axis][action] = (
                    env->data->qacc[root_dof + axis] - base_qacc[root_dof + axis]
                ) / 1e-4;
            }
            env->data->ctrl[actuator] -= 1e-4;
        }
        mj_forward(env->model, env->data);

        double gram[36] = {0};
        double trace = 0.0;
        for (int row = 0; row < REK_FIGHT_ROOT_DIMS; row++) {
            for (int column = 0; column < REK_FIGHT_ROOT_DIMS; column++) {
                for (int action = 0; action < REK_FIGHT_ROOT_LEGS; action++) {
                    gram[row * REK_FIGHT_ROOT_DIMS + column]
                        += response[row][action] * response[column][action];
                }
            }
            trace += gram[row * REK_FIGHT_ROOT_DIMS + row];
        }
        double scale = fmax(trace / REK_FIGHT_ROOT_DIMS, 1e-12);
        for (int axis = 0; axis < REK_FIGHT_ROOT_DIMS; axis++) {
            gram[axis * REK_FIGHT_ROOT_DIMS + axis] += 8.613707589105229e-7 * scale;
        }
        double inverse[36];
        if (!rek_fight_inverse_6x6(gram, inverse)) {
            rek_fight_fail("root controller inverse failed");
        }
        for (int action = 0; action < REK_FIGHT_ROOT_LEGS; action++) {
            for (int output_axis = 0; output_axis < REK_FIGHT_ROOT_DIMS; output_axis++) {
                double value = 0.0;
                for (int input_axis = 0; input_axis < REK_FIGHT_ROOT_DIMS; input_axis++) {
                    value += response[input_axis][action]
                        * inverse[input_axis * REK_FIGHT_ROOT_DIMS + output_axis];
                }
                env->root_pinv[agent][action][output_axis] = value;
            }
        }
    }
}

static mjModel* rek_fight_load_model(void) {
    if (mjVERSION_HEADER != REK_FIGHT_MUJOCO_VERSION
            || mj_version() != REK_FIGHT_MUJOCO_VERSION) {
        rek_fight_fail("MuJoCo 3.7.0 required");
    }
    mjModel* model = rek_match_load_model();
    rek_fight_configure_actuators(model);
    return model;
}

static void rek_fight_init_with_model(
        RekFight* env, const mjModel* model, int owns_model) {
    if (env == NULL || model == NULL) rek_fight_fail("environment and model are required");
    env->model = model;
    env->owns_model = owns_model;
    env->shared_model_env_count = 1;
    env->num_agents = REK_FIGHT_NUM_AGENTS;
    env->data = mj_makeData(model);
    if (env->data == NULL) rek_fight_fail("mj_makeData failed");
    RekStrategyCapabilities capabilities = {1, 1, 1};
    for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
        if (!rek_strategy_router_init(&env->agent[agent].router, capabilities)) {
            rek_fight_fail("strategy router init failed");
        }
    }
    rek_fight_collect_limb_geoms(env);
    rek_fight_init_root_controller(env);
}

static void rek_fight_init(RekFight* env) {
    rek_fight_init_with_model(env, rek_fight_load_model(), 1);
}

static int rek_fight_state_is_finite(const RekFight* env) {
    for (int i = 0; i < env->model->nq; i++) {
        if (!isfinite(env->data->qpos[i])) return 0;
    }
    for (int i = 0; i < env->model->nv; i++) {
        if (!isfinite(env->data->qvel[i]) || !isfinite(env->data->qacc[i])) return 0;
    }
    return isfinite(env->data->time);
}

static int rek_fight_write_agent_state_observation(
        float* output, int cursor, const RekFightAgent* state) {
    output[cursor++] = (float)state->recovering;
    output[cursor++] = (float)state->move_in_progress;
    output[cursor++] = (float)state->cooldown_active;
    output[cursor++] = (float)state->fallen;
    output[cursor++] = (float)state->move_slot;
    output[cursor++] = (float)state->move_ticks;
    output[cursor++] = (float)state->move_duration_ticks;
    output[cursor++] = (float)state->cooldown_ticks;
    output[cursor++] = (float)state->recovery_ticks;
    output[cursor++] = (float)state->scored_impacts;
    output[cursor++] = (float)state->hits;
    output[cursor++] = (float)state->router.last_emitted_move_category;
    for (int axis = 0; axis < REK_STRATEGY_VELOCITY_DIMS; axis++) {
        output[cursor++] = state->router.held_velocity[axis];
    }
    for (int category = 0; category < REK_FIGHT_MOVE_MASK_SIZE; category++) {
        output[cursor++] = (float)rek_fight_move_action_available(state, category);
    }
    return cursor;
}

static void rek_fight_compute_agent_observation(RekFight* env, int observer) {
    int opponent = 1 - observer;
    int qpos_offsets[2] = {
        observer * REK_MATCH_QPOS_PER_AGENT,
        opponent * REK_MATCH_QPOS_PER_AGENT,
    };
    int qvel_offsets[2] = {
        observer * REK_MATCH_QVEL_PER_AGENT,
        opponent * REK_MATCH_QVEL_PER_AGENT,
    };
    float* output = env->observations + observer * REK_FIGHT_OBS_SIZE;
    int cursor = 0;
    for (int side = 0; side < 2; side++) {
        for (int i = 0; i < REK_MATCH_QPOS_PER_AGENT; i++) {
            output[cursor++] = (float)env->data->qpos[qpos_offsets[side] + i];
        }
        for (int i = 0; i < REK_MATCH_QVEL_PER_AGENT; i++) {
            output[cursor++] = (float)env->data->qvel[qvel_offsets[side] + i];
        }
    }
    cursor = rek_fight_write_agent_state_observation(
        output, cursor, &env->agent[observer]
    );
    cursor = rek_fight_write_agent_state_observation(
        output, cursor, &env->agent[opponent]
    );
    output[cursor++] = (float)env->tick;
    output[cursor++] = (float)env->data->time;
    output[cursor++] = env->max_steps > 0
        ? (float)(env->max_steps - env->tick)
        : -1.0f;
    if (cursor != REK_FIGHT_OBS_SIZE) {
        rek_fight_fail("observation schema size mismatch");
    }
}

static void rek_fight_compute_observations(RekFight* env) {
    for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
        rek_fight_compute_agent_observation(env, agent);
    }
}

static void rek_fight_reset_agent(RekFightAgent* agent) {
    agent->recovering = 0;
    agent->move_in_progress = 0;
    agent->cooldown_active = 0;
    agent->move_slot = -1;
    agent->move_ticks = 0;
    agent->move_duration_ticks = 0;
    agent->cooldown_ticks = 0;
    agent->recovery_ticks = 0;
    agent->scored_impacts = 0;
    agent->hits = 0;
    agent->fallen = 0;
    agent->router.last_emitted_move_category = 0;
    agent->router.held_velocity[0] = 0.0f;
    agent->router.held_velocity[1] = 0.0f;
    agent->router.held_velocity[2] = 0.0f;
}

static void rek_fight_reset_state(RekFight* env) {
    mj_resetDataKeyframe(env->model, env->data, REK_MATCH_KEYFRAME_ID);
    rek_fight_set_keyframe_controls(env);
    mj_forward(env->model, env->data);
    env->tick = 0;
    for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
        rek_fight_reset_agent(&env->agent[agent]);
        env->episode_return[agent] = 0.0f;
    }
    rek_fight_compute_observations(env);
}

void c_reset(RekFight* env) {
    rek_fight_reset_state(env);
    for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
        env->rewards[agent] = 0.0f;
        env->terminals[agent] = 0.0f;
    }
}

static void rek_fight_read_action(
        const RekFight* env, int agent, RekStrategyAction* action) {
    const float* input = env->actions + agent * REK_FIGHT_NUM_ACTIONS;
    action->velocity_bins[0] = rek_fight_bin(input[0]);
    action->velocity_bins[1] = rek_fight_bin(input[1]);
    action->velocity_bins[2] = rek_fight_bin(input[2]);
    int move = (int)lrintf(input[3]);
    if (move < 0) move = 0;
    if (move >= REK_STRATEGY_MOVE_CATEGORIES) move = 0;
    action->move_category = move;
}

static void rek_fight_apply_reach(
        RekFight* env, int agent, const RekFightMove* move, float phase) {
    if (move == NULL || phase <= 0.0f) return;
    int limb = move->limb - 1;
    if (limb < 0 || limb >= REK_FIGHT_LIMBS) return;
    double envelope = sin((double)phase * 3.14159265358979323846);
    if (envelope < 0.0) envelope = 0.0;
    for (int item = 0; item < REK_FIGHT_LIMB_JOINT_COUNTS[limb]; item++) {
        int action = REK_FIGHT_LIMB_JOINTS[limb][item];
        int joint = agent * 26 + REK_MATCH_LOCAL_JOINT_IDS[action];
        double lower = env->model->jnt_range[2 * joint];
        double upper = env->model->jnt_range[2 * joint + 1];
        double stand = rek_fight_keyframe_joint(env, agent, action);
        double extreme = (limb == 0 || limb == 2) ? upper : lower;
        double target = stand + 0.65 * envelope * (extreme - stand);
        env->data->ctrl[agent * REK_FIGHT_JOINTS + action]
            = rek_fight_limit_joint(env, agent, action, target);
    }
}

static void rek_fight_apply_root_command(
        RekFight* env, int agent, const RekStrategyOutput* command) {
    if (env->root_stabilizer_scale <= 0.0f) return;
    mjtNum tangent_error[REK_MATCH_NV];
    const mjtNum* key_qpos = env->model->key_qpos
        + REK_MATCH_KEYFRAME_ID * env->model->nq;
    mj_differentiatePos(env->model, tangent_error, 1.0, env->data->qpos, key_qpos);

    int root_qpos = agent * REK_MATCH_QPOS_PER_AGENT;
    int root_dof = agent * REK_MATCH_QVEL_PER_AGENT;
    double qw = env->data->qpos[root_qpos + 3];
    double qx = env->data->qpos[root_qpos + 4];
    double qy = env->data->qpos[root_qpos + 5];
    double qz = env->data->qpos[root_qpos + 6];
    double yaw = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
    double cosine = cos(yaw);
    double sine = sin(yaw);
    double forward = command->velocity[0] * REK_FIGHT_FORWARD_SPEED;
    double strafe = command->velocity[1] * REK_FIGHT_STRAFE_SPEED;
    double desired_x = cosine * forward - sine * strafe;
    double desired_y = sine * forward + cosine * strafe;
    double desired_yaw = command->velocity[2] * REK_FIGHT_YAW_SPEED;

    double residual[REK_FIGHT_ROOT_DIMS];
    double desired_vel[REK_FIGHT_ROOT_DIMS] = {
        desired_x, desired_y, 0.0, 0.0, 0.0, desired_yaw,
    };
    for (int axis = 0; axis < REK_FIGHT_ROOT_DIMS; axis++) {
        double position_term = REK_FIGHT_ROOT_KP[axis] * tangent_error[root_dof + axis];
        if (axis == 0 || axis == 1 || axis == 5) {
            position_term = 0.0;
        }
        double desired_acceleration = position_term
            + REK_FIGHT_ROOT_KD[axis] * (
                desired_vel[axis] - env->data->qvel[root_dof + axis]
            );
        residual[axis] = desired_acceleration - env->data->qacc[root_dof + axis];
    }
    for (int action = 0; action < REK_FIGHT_ROOT_LEGS; action++) {
        double target_delta = 0.0;
        for (int axis = 0; axis < REK_FIGHT_ROOT_DIMS; axis++) {
            target_delta += env->root_pinv[agent][action][axis] * residual[axis];
        }
        double limit = 0.06425133234876036 * env->root_stabilizer_scale;
        target_delta = rek_fight_clip(
            target_delta * env->root_stabilizer_scale, -limit, limit
        );
        int actuator = agent * REK_FIGHT_JOINTS + action;
        env->data->ctrl[actuator] = rek_fight_limit_joint(
            env, agent, action, env->data->ctrl[actuator] + target_delta
        );
    }
}

static int rek_fight_geom_in_limb(const RekFight* env, int agent, int limb, int geom) {
    for (int i = 0; i < env->limb_geom_count[agent][limb]; i++) {
        if (env->limb_geoms[agent][limb][i] == geom) return 1;
    }
    return 0;
}

static int rek_fight_body_agent(const mjModel* model, int body) {
    if (body <= 0) return -1;
    if (body <= 30) return 0;
    if (body <= 60) return 1;
    (void)model;
    return -1;
}

static int rek_fight_contact_hit(const RekFight* env, int attacker, int limb) {
    int defender = 1 - attacker;
    for (int contact = 0; contact < env->data->ncon; contact++) {
        int geom1 = env->data->contact[contact].geom1;
        int geom2 = env->data->contact[contact].geom2;
        int body1 = env->model->geom_bodyid[geom1];
        int body2 = env->model->geom_bodyid[geom2];
        int agent1 = rek_fight_body_agent(env->model, body1);
        int agent2 = rek_fight_body_agent(env->model, body2);
        if (agent1 != attacker && agent2 != attacker) continue;
        if (agent1 != defender && agent2 != defender) continue;
        int attacker_geom = agent1 == attacker ? geom1 : geom2;
        if (rek_fight_geom_in_limb(env, attacker, limb, attacker_geom)) return 1;
    }
    return 0;
}

static RekStrategyOutput rek_fight_plan_agent(RekFight* env, int agent) {
    RekFightAgent* state = &env->agent[agent];
    RekStrategyAction action;
    rek_fight_read_action(env, agent, &action);
    RekStrategyGates gates = {
        state->recovering,
        state->move_in_progress,
        state->cooldown_active,
    };
    RekStrategyOutput command;
    if (!rek_strategy_route(&state->router, &action, &gates, &command)) {
        memset(&command, 0, sizeof(command));
        command.move_slot = -1;
    }

    if (state->recovering) {
        state->recovery_ticks += 1;
        if (state->recovery_ticks * REK_FIGHT_DT >= REK_FIGHT_RECOVERY_S
                && !rek_fight_fallen(env, agent)) {
            state->recovering = 0;
            state->recovery_ticks = 0;
            state->fallen = 0;
        }
    } else if (command.request_move && command.move_slot >= 0) {
        const RekFightMove* move = rek_fight_move_by_slot(command.move_slot);
        if (move != NULL) {
            state->move_in_progress = 1;
            state->move_slot = command.move_slot;
            state->move_ticks = 0;
            state->scored_impacts = 0;
            state->move_duration_ticks = (int)ceil(move->duration_s / REK_FIGHT_DT);
            if (state->move_duration_ticks < 1) state->move_duration_ticks = 1;
        }
    }

    rek_fight_set_agent_keyframe_controls(env, agent);
    if (state->move_in_progress) {
        const RekFightMove* active = rek_fight_move_by_slot(state->move_slot);
        float phase = 0.0f;
        if (active != NULL && active->duration_s > 0.0f) {
            phase = (float)(state->move_ticks * REK_FIGHT_DT / active->duration_s);
            if (phase > 1.0f) phase = 1.0f;
        }
        rek_fight_apply_reach(env, agent, active, phase);
    }
    return command;
}

static float rek_fight_score_hits(RekFight* env, int agent) {
    RekFightAgent* state = &env->agent[agent];
    if (!state->move_in_progress) return 0.0f;
    const RekFightMove* move = rek_fight_move_by_slot(state->move_slot);
    if (move == NULL) return 0.0f;
    float time_s = (float)(state->move_ticks * REK_FIGHT_DT);
    float reward = 0.0f;
    for (int impact = 0; impact < move->impact_count; impact++) {
        int mask = 1 << impact;
        if (state->scored_impacts & mask) continue;
        float start = move->impact_s[impact] - move->lead_s[impact];
        float end = move->impact_s[impact] + move->release_s[impact];
        if (time_s < start || time_s > end) continue;
        if (rek_fight_contact_hit(env, agent, move->limb - 1)) {
            state->scored_impacts |= mask;
            state->hits += 1;
            reward += REK_FIGHT_HIT_REWARD;
        }
    }
    return reward;
}

static void rek_fight_advance_agent_timers(RekFightAgent* state) {
    if (state->move_in_progress) {
        state->move_ticks += 1;
        if (state->move_ticks >= state->move_duration_ticks) {
            state->move_in_progress = 0;
            state->move_slot = -1;
            state->cooldown_active = 1;
            state->cooldown_ticks = (int)ceil(
                REK_FIGHT_COOLDOWN_S / REK_FIGHT_DT
            );
        }
    } else if (state->cooldown_active) {
        state->cooldown_ticks -= 1;
        if (state->cooldown_ticks <= 0) {
            state->cooldown_active = 0;
            state->cooldown_ticks = 0;
        }
    }
}

static void rek_fight_add_log(RekFight* env, int invalid, int timeout) {
    env->log.episode_return += env->episode_return[0];
    env->log.score += env->episode_return[0];
    env->log.episode_length += (float)env->tick;
    env->log.hits += (float)(env->agent[0].hits + env->agent[1].hits);
    env->log.falls += (float)(env->agent[0].fallen + env->agent[1].fallen);
    env->log.invalid_termination += invalid ? 1.0f : 0.0f;
    env->log.timeout += timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

void c_step(RekFight* env) {
    RekStrategyOutput commands[REK_FIGHT_NUM_AGENTS];
    for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
        commands[agent] = rek_fight_plan_agent(env, agent);
    }
    mj_forward(env->model, env->data);
    for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
        if (!env->agent[agent].recovering) {
            rek_fight_apply_root_command(env, agent, &commands[agent]);
        }
    }
    mj_step(env->model, env->data);
    env->tick += 1;

    float reward[REK_FIGHT_NUM_AGENTS] = {
        -REK_FIGHT_TIME_COST, -REK_FIGHT_TIME_COST,
    };
    for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
        reward[agent] += rek_fight_score_hits(env, agent);
        rek_fight_advance_agent_timers(&env->agent[agent]);
        int fallen = rek_fight_fallen(env, agent);
        if (fallen && !env->agent[agent].fallen) {
            env->agent[agent].fallen = 1;
            env->agent[agent].recovering = 1;
            env->agent[agent].recovery_ticks = 0;
            env->agent[agent].move_in_progress = 0;
            reward[agent] -= REK_FIGHT_FALL_REWARD;
            reward[1 - agent] += REK_FIGHT_FALL_REWARD;
        }
    }

    int invalid = !rek_fight_state_is_finite(env);
    int timeout = env->max_steps > 0 && env->tick >= env->max_steps;
    int round_over = timeout || invalid
        || (env->agent[0].fallen && env->agent[1].fallen);
    for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
        env->episode_return[agent] += reward[agent];
        env->rewards[agent] = reward[agent];
        env->terminals[agent] = round_over ? 1.0f : 0.0f;
    }
    if (round_over) {
        rek_fight_add_log(env, invalid, timeout);
        rek_fight_reset_state(env);
        for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
            env->terminals[agent] = 1.0f;
        }
        return;
    }
    rek_fight_compute_observations(env);
}

void c_render(RekFight* env) {
    (void)env;
}

void c_close(RekFight* env) {
    if (env->data != NULL) {
        mj_deleteData(env->data);
        env->data = NULL;
    }
    if (env->owns_model && env->model != NULL) {
        mj_deleteModel((mjModel*)env->model);
        env->model = NULL;
    }
}

static void rek_fight_close_shared_model(RekFight* envs) {
    if (envs == NULL || envs[0].model == NULL) return;
    const mjModel* shared_model = envs[0].model;
    int count = envs[0].shared_model_env_count;
    if (count <= 0) rek_fight_fail("invalid shared model environment count");
    for (int i = 0; i < count; i++) {
        if (envs[i].model == shared_model) {
            envs[i].owns_model = 0;
            c_close(&envs[i]);
        }
    }
    mj_deleteModel((mjModel*)shared_model);
}

static int rek_fight_partition_layout(
        int total_agents,
        int num_buffers,
        int* physical_envs_out,
        int* envs_per_buffer_out) {
    return rek_match_partition_layout(
        total_agents, num_buffers, physical_envs_out, envs_per_buffer_out
    );
}
