#include "rek_fight.h"

#if defined(_WIN32)
#define REK_HUMAN_API __declspec(dllexport)
#else
#define REK_HUMAN_API __attribute__((visibility("default")))
#endif

typedef struct RekHumanEval {
    RekFight env;
    float observations[REK_FIGHT_NUM_AGENTS * REK_FIGHT_OBS_SIZE];
    float actions[REK_FIGHT_NUM_AGENTS * REK_FIGHT_NUM_ACTIONS];
    float rewards[REK_FIGHT_NUM_AGENTS];
    float terminals[REK_FIGHT_NUM_AGENTS];
} RekHumanEval;

static int rek_human_action_bin(int value) {
    if (value < 0) return 0;
    if (value > 2) return 2;
    return value;
}

static int rek_human_move_category(int value) {
    if (value < 0 || value >= REK_STRATEGY_MOVE_CATEGORIES) return 0;
    return value;
}

REK_HUMAN_API RekHumanEval* rek_human_create(void) {
    RekHumanEval* session = (RekHumanEval*)calloc(1, sizeof(RekHumanEval));
    if (session == NULL) return NULL;

    session->env.observations = session->observations;
    session->env.actions = session->actions;
    session->env.rewards = session->rewards;
    session->env.terminals = session->terminals;
    session->env.max_steps = 0;
    session->env.fall_height = 0.5f;
    session->env.fall_up_z = 0.5f;
    session->env.root_stabilizer_scale = 1.0f;
    rek_fight_init(&session->env);
    c_reset(&session->env);
    return session;
}

REK_HUMAN_API void rek_human_destroy(RekHumanEval* session) {
    if (session == NULL) return;
    c_close(&session->env);
    free(session);
}

REK_HUMAN_API void rek_human_reset(RekHumanEval* session) {
    if (session == NULL) return;
    c_reset(&session->env);
}

REK_HUMAN_API void rek_human_step(
        RekHumanEval* session,
        int forward,
        int strafe,
        int yaw,
        int move_category) {
    if (session == NULL) return;

    session->actions[0] = (float)rek_human_action_bin(forward);
    session->actions[1] = (float)rek_human_action_bin(strafe);
    session->actions[2] = (float)rek_human_action_bin(yaw);
    session->actions[3] = (float)rek_human_move_category(move_category);

    double dx = session->env.data->qpos[0] - session->env.data->qpos[32];
    double dy = session->env.data->qpos[1] - session->env.data->qpos[33];
    double distance = sqrt(dx * dx + dy * dy);
    double qw = session->env.data->qpos[35];
    double qx = session->env.data->qpos[36];
    double qy = session->env.data->qpos[37];
    double qz = session->env.data->qpos[38];
    double heading = atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz)
    );
    double heading_error = atan2(sin(atan2(dy, dx) - heading),
                                 cos(atan2(dy, dx) - heading));
    session->actions[4] = distance > 1.1 && fabs(heading_error) < 0.6
        ? 2.0f : 1.0f;
    session->actions[5] = 1.0f;
    session->actions[6] = heading_error > 0.06
        ? 2.0f : heading_error < -0.06 ? 0.0f : 1.0f;
    session->actions[7] = 0.0f;
    if (distance < 1.4 && session->env.tick > 0
            && session->env.tick % 90 == 0) {
        static const int attack_cycle[4] = {3, 4, 2, 6};
        session->actions[7] = (float)attack_cycle[(session->env.tick / 90) % 4];
    }

    c_step(&session->env);
}

REK_HUMAN_API int rek_human_nq(const RekHumanEval* session) {
    return session == NULL ? 0 : session->env.model->nq;
}

REK_HUMAN_API int rek_human_nv(const RekHumanEval* session) {
    return session == NULL ? 0 : session->env.model->nv;
}

REK_HUMAN_API int rek_human_copy_state(
        const RekHumanEval* session,
        double* qpos,
        int qpos_capacity,
        double* qvel,
        int qvel_capacity) {
    if (session == NULL || qpos == NULL || qvel == NULL) return 0;
    if (qpos_capacity < session->env.model->nq
            || qvel_capacity < session->env.model->nv) return 0;
    memcpy(qpos, session->env.data->qpos,
        (size_t)session->env.model->nq * sizeof(double));
    memcpy(qvel, session->env.data->qvel,
        (size_t)session->env.model->nv * sizeof(double));
    return 1;
}

REK_HUMAN_API int rek_human_tick(const RekHumanEval* session) {
    return session == NULL ? 0 : session->env.tick;
}

REK_HUMAN_API int rek_human_hits(const RekHumanEval* session, int agent) {
    if (session == NULL || agent < 0 || agent >= REK_FIGHT_NUM_AGENTS) return 0;
    return session->env.agent[agent].hits;
}

REK_HUMAN_API int rek_human_fallen(const RekHumanEval* session, int agent) {
    if (session == NULL || agent < 0 || agent >= REK_FIGHT_NUM_AGENTS) return 0;
    return session->env.agent[agent].fallen;
}

REK_HUMAN_API int rek_human_move_slot(const RekHumanEval* session, int agent) {
    if (session == NULL || agent < 0 || agent >= REK_FIGHT_NUM_AGENTS) return -1;
    return session->env.agent[agent].move_slot;
}

REK_HUMAN_API float rek_human_reward(const RekHumanEval* session, int agent) {
    if (session == NULL || agent < 0 || agent >= REK_FIGHT_NUM_AGENTS) return 0.0f;
    return session->env.rewards[agent];
}

REK_HUMAN_API float rek_human_episode_return(
        const RekHumanEval* session, int agent) {
    if (session == NULL || agent < 0 || agent >= REK_FIGHT_NUM_AGENTS) return 0.0f;
    return session->env.episode_return[agent];
}
