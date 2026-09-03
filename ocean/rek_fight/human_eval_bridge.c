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

    session->actions[4] = 1.0f;
    session->actions[5] = 1.0f;
    session->actions[6] = 1.0f;
    session->actions[7] = 0.0f;
    if (session->env.tick > 0 && session->env.tick % 125 == 0) {
        session->actions[7] = (float)(1 + (session->env.tick / 125) % 6);
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
