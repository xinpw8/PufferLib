#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "rek_fight.h"

static void configure_env(RekFight* env, int max_steps) {
    env->num_agents = REK_FIGHT_NUM_AGENTS;
    env->max_steps = max_steps;
    env->fall_height = 0.5f;
    env->fall_up_z = 0.5f;
    env->root_stabilizer_scale = 1.0f;
}

static void attach_buffers(
        RekFight* env,
        float observations[REK_FIGHT_NUM_AGENTS * REK_FIGHT_OBS_SIZE],
        float actions[REK_FIGHT_NUM_AGENTS * REK_FIGHT_NUM_ACTIONS],
        float rewards[REK_FIGHT_NUM_AGENTS],
        float terminals[REK_FIGHT_NUM_AGENTS]) {
    env->observations = observations;
    env->actions = actions;
    env->rewards = rewards;
    env->terminals = terminals;
}

static void idle_actions(float* actions) {
    for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
        float* row = actions + agent * REK_FIGHT_NUM_ACTIONS;
        row[0] = 1.0f;
        row[1] = 1.0f;
        row[2] = 1.0f;
        row[3] = 0.0f;
    }
}

static void test_router_and_idle(RekFight* env) {
    float observations[REK_FIGHT_NUM_AGENTS * REK_FIGHT_OBS_SIZE];
    float actions[REK_FIGHT_NUM_AGENTS * REK_FIGHT_NUM_ACTIONS];
    float rewards[REK_FIGHT_NUM_AGENTS];
    float terminals[REK_FIGHT_NUM_AGENTS];
    attach_buffers(env, observations, actions, rewards, terminals);
    idle_actions(actions);
    c_reset(env);
    assert(env->num_agents == 2);
    assert(env->model->nq == 64);
    assert(env->model->nu == 50);
    for (int step = 0; step < 25; step++) {
        c_step(env);
        assert(isfinite(rewards[0]) && isfinite(rewards[1]));
        assert(terminals[0] == 0.0f);
        assert(!env->agent[0].move_in_progress);
    }
}

static void test_move_request(RekFight* env) {
    float observations[REK_FIGHT_NUM_AGENTS * REK_FIGHT_OBS_SIZE];
    float actions[REK_FIGHT_NUM_AGENTS * REK_FIGHT_NUM_ACTIONS];
    float rewards[REK_FIGHT_NUM_AGENTS];
    float terminals[REK_FIGHT_NUM_AGENTS];
    attach_buffers(env, observations, actions, rewards, terminals);
    idle_actions(actions);
    c_reset(env);
    actions[3] = 6.0f;
    c_step(env);
    assert(env->agent[0].move_in_progress == 1);
    assert(env->agent[0].move_slot == 10);
    idle_actions(actions);
    int guarded = 0;
    for (int step = 0; step < 5; step++) {
        actions[3] = 6.0f;
        c_step(env);
        if (env->agent[0].move_in_progress) guarded += 1;
    }
    assert(guarded == 5);
}

static void test_scripted_approach(RekFight* env) {
    float observations[REK_FIGHT_NUM_AGENTS * REK_FIGHT_OBS_SIZE];
    float actions[REK_FIGHT_NUM_AGENTS * REK_FIGHT_NUM_ACTIONS];
    float rewards[REK_FIGHT_NUM_AGENTS];
    float terminals[REK_FIGHT_NUM_AGENTS];
    attach_buffers(env, observations, actions, rewards, terminals);
    idle_actions(actions);
    c_reset(env);
    double start = env->data->qpos[0];
    for (int step = 0; step < 40; step++) {
        actions[0] = 2.0f;
        actions[1] = 1.0f;
        actions[2] = 1.0f;
        actions[3] = 0.0f;
        actions[4] = 0.0f;
        actions[5] = 1.0f;
        actions[6] = 1.0f;
        actions[7] = 0.0f;
        c_step(env);
    }
    printf(
        "rek_fight demo: x0 %.3f -> %.3f hits %d/%d fallen %d/%d return %.3f\n",
        start,
        env->data->qpos[0],
        env->agent[0].hits,
        env->agent[1].hits,
        env->agent[0].fallen,
        env->agent[1].fallen,
        env->episode_return[0]
    );
    assert(isfinite(env->data->qpos[0]));
}

int main(void) {
    RekFight env;
    memset(&env, 0, sizeof(env));
    configure_env(&env, 1500);
    rek_fight_init(&env);
    test_router_and_idle(&env);
    test_move_request(&env);
    test_scripted_approach(&env);
    c_close(&env);
    printf("rek_fight tests passed\n");
    return 0;
}
