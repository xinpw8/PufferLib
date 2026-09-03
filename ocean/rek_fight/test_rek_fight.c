#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static int first_geom_for_agent(const RekFight* env, int agent) {
    for (int geom = 0; geom < env->model->ngeom; geom++) {
        int body = env->model->geom_bodyid[geom];
        if (rek_fight_body_agent(env->model, body) == agent) return geom;
    }
    return -1;
}

static void test_descendant_limb_geometries(const RekFight* env) {
    const char* terminal_names[2] = {"LINK_WRIST_END_L", "LINK_FOOT_L"};
    const int terminal_limbs[2] = {0, 2};
    for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
        for (int terminal = 0; terminal < 2; terminal++) {
            int matched_body = 0;
            int matched_geom = 0;
            for (int body = 1; body < env->model->nbody; body++) {
                const char* name = mj_id2name(env->model, mjOBJ_BODY, body);
                if (rek_fight_body_agent(env->model, body) != agent
                        || name == NULL
                        || strstr(name, terminal_names[terminal]) == NULL) {
                    continue;
                }
                matched_body += 1;
                for (int geom = 0; geom < env->model->ngeom; geom++) {
                    if (env->model->geom_bodyid[geom] != body) continue;
                    matched_geom += 1;
                    assert(rek_fight_geom_in_limb(
                        env, agent, terminal_limbs[terminal], geom
                    ));
                }
            }
            assert(matched_body == 1);
            assert(matched_geom > 0);
        }
    }
}

static void test_complete_state_observation(RekFight* env) {
    float observations[REK_FIGHT_NUM_AGENTS * REK_FIGHT_OBS_SIZE];
    float actions[REK_FIGHT_NUM_AGENTS * REK_FIGHT_NUM_ACTIONS];
    float rewards[REK_FIGHT_NUM_AGENTS];
    float terminals[REK_FIGHT_NUM_AGENTS];
    attach_buffers(env, observations, actions, rewards, terminals);
    idle_actions(actions);
    c_reset(env);

    RekFightAgent* state = &env->agent[0];
    state->recovering = 1;
    state->move_in_progress = 1;
    state->cooldown_active = 1;
    state->fallen = 1;
    state->move_slot = 4;
    state->move_ticks = 7;
    state->move_duration_ticks = 29;
    state->cooldown_ticks = 11;
    state->recovery_ticks = 13;
    state->scored_impacts = 5;
    state->hits = 17;
    state->router.last_emitted_move_category = 3;
    state->router.held_velocity[0] = -1.0f;
    state->router.held_velocity[1] = 0.0f;
    state->router.held_velocity[2] = 1.0f;
    env->tick = 19;
    env->data->time = 0.38;
    rek_fight_compute_observations(env);

    const int state_start = REK_MATCH_OBS_SIZE;
    const float expected_prefix[15] = {
        1, 1, 1, 1, 4, 7, 29, 11, 13, 5, 17, 3, -1, 0, 1,
    };
    for (int i = 0; i < 15; i++) {
        assert(observations[state_start + i] == expected_prefix[i]);
    }
    assert(observations[state_start + 15] == 1.0f);
    for (int category = 1; category < REK_FIGHT_MOVE_MASK_SIZE; category++) {
        assert(observations[state_start + 15 + category] == 0.0f);
    }
    int global_start = REK_MATCH_OBS_SIZE
        + 2 * REK_FIGHT_AGENT_STATE_OBS_SIZE;
    assert(observations[global_start] == 19.0f);
    assert(fabsf(observations[global_start + 1] - 0.38f) < 1e-6f);
    assert(observations[global_start + 2] == (float)(env->max_steps - 19));

    const float* opponent_view = observations + REK_FIGHT_OBS_SIZE;
    int other_state_start = REK_MATCH_OBS_SIZE + REK_FIGHT_AGENT_STATE_OBS_SIZE;
    for (int i = 0; i < 15; i++) {
        assert(opponent_view[other_state_start + i] == expected_prefix[i]);
    }
}

static void test_move_action_mask(void) {
    RekFightAgent state;
    memset(&state, 0, sizeof(state));
    state.router.last_emitted_move_category = 3;
    assert(rek_fight_move_action_available(&state, 0));
    for (int category = 1; category < REK_FIGHT_MOVE_MASK_SIZE; category++) {
        assert(rek_fight_move_action_available(&state, category) == (category != 3));
    }
    state.cooldown_active = 1;
    for (int category = 1; category < REK_FIGHT_MOVE_MASK_SIZE; category++) {
        assert(!rek_fight_move_action_available(&state, category));
    }
}

static void test_zero_time_impact(RekFight* env) {
    float observations[REK_FIGHT_NUM_AGENTS * REK_FIGHT_OBS_SIZE];
    float actions[REK_FIGHT_NUM_AGENTS * REK_FIGHT_NUM_ACTIONS];
    float rewards[REK_FIGHT_NUM_AGENTS];
    float terminals[REK_FIGHT_NUM_AGENTS];
    attach_buffers(env, observations, actions, rewards, terminals);
    idle_actions(actions);
    c_reset(env);

    RekFightAgent* state = &env->agent[0];
    state->move_in_progress = 1;
    state->move_slot = 9;
    state->move_ticks = 0;
    state->move_duration_ticks = 10;
    int attacker_geom = env->limb_geoms[0][1][0];
    int defender_geom = first_geom_for_agent(env, 1);
    assert(attacker_geom >= 0 && defender_geom >= 0);
    assert(env->data->contact != NULL);
    env->data->ncon = 1;
    env->data->contact[0].geom1 = attacker_geom;
    env->data->contact[0].geom2 = defender_geom;

    assert(rek_fight_score_hits(env, 0) == REK_FIGHT_HIT_REWARD);
    assert(state->scored_impacts == 1);
    assert(state->hits == 1);
    assert(state->move_ticks == 0);
    rek_fight_advance_agent_timers(state);
    assert(state->move_ticks == 1);
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

static void assert_upright_for_steps(
        RekFight* env,
        float actions[REK_FIGHT_NUM_AGENTS * REK_FIGHT_NUM_ACTIONS],
        int steps) {
    for (int step = 0; step < steps; step++) {
        c_step(env);
        for (int agent = 0; agent < REK_FIGHT_NUM_AGENTS; agent++) {
            assert(!env->agent[agent].fallen);
            assert(!env->agent[agent].recovering);
            assert(rek_fight_root_height(env, agent) > env->fall_height);
            assert(rek_fight_root_up_z(env, agent) > env->fall_up_z);
        }
    }
    idle_actions(actions);
}

static void test_framework_controller(RekFight* env) {
    float observations[REK_FIGHT_NUM_AGENTS * REK_FIGHT_OBS_SIZE];
    float actions[REK_FIGHT_NUM_AGENTS * REK_FIGHT_NUM_ACTIONS];
    float rewards[REK_FIGHT_NUM_AGENTS];
    float terminals[REK_FIGHT_NUM_AGENTS];
    attach_buffers(env, observations, actions, rewards, terminals);
    idle_actions(actions);
    c_reset(env);

    double idle_x = env->data->qpos[0];
    double idle_y = env->data->qpos[1];
    assert_upright_for_steps(env, actions, 500);
    assert(fabs(env->data->qpos[0] - idle_x) < 1e-12);
    assert(fabs(env->data->qpos[1] - idle_y) < 1e-12);

    c_reset(env);
    double forward_x = env->data->qpos[0];
    actions[0] = 2.0f;
    assert_upright_for_steps(env, actions, 100);
    assert(env->data->qpos[0] - forward_x > 1.5);

    c_reset(env);
    double backward_x = env->data->qpos[0];
    actions[0] = 0.0f;
    assert_upright_for_steps(env, actions, 100);
    assert(env->data->qpos[0] - backward_x < -1.5);

    c_reset(env);
    double strafe_y = env->data->qpos[1];
    actions[1] = 2.0f;
    assert_upright_for_steps(env, actions, 100);
    assert(env->data->qpos[1] - strafe_y > 0.8);

    c_reset(env);
    double start_qpos[REK_FIGHT_JOINTS];
    for (int action = 0; action < REK_FIGHT_JOINTS; action++) {
        int qpos = REK_MATCH_LOCAL_JOINT_QPOS_ADDRESSES[action];
        start_qpos[action] = env->data->qpos[qpos];
    }
    actions[3] = 2.0f;
    c_step(env);
    actions[3] = 0.0f;
    for (int step = 0; step < 27; step++) c_step(env);
    int moved_joint = 0;
    for (int action = 0; action < REK_FIGHT_JOINTS; action++) {
        int qpos = REK_MATCH_LOCAL_JOINT_QPOS_ADDRESSES[action];
        if (fabs(env->data->qpos[qpos] - start_qpos[action]) > 0.01) {
            moved_joint = 1;
        }
    }
    assert(moved_joint);
    assert_upright_for_steps(env, actions, 100);

    c_reset(env);
    env->data->qpos[2] = 0.2;
    env->data->qpos[3] = 0.7071067811865476;
    env->data->qpos[4] = 0.7071067811865476;
    env->data->qpos[5] = 0.0;
    env->data->qpos[6] = 0.0;
    mju_zero(env->data->qvel, env->model->nv);
    mj_forward(env->model, env->data);
    c_step(env);
    assert(env->agent[0].recovering);
    for (int step = 0; step < 80; step++) c_step(env);
    assert(!env->agent[0].recovering);
    assert(!env->agent[0].fallen);
    assert(rek_fight_root_height(env, 0) > env->fall_height);
    assert(rek_fight_root_up_z(env, 0) > env->fall_up_z);
}

static void run_benchmark(RekFight* env, int steps) {
    float observations[REK_FIGHT_NUM_AGENTS * REK_FIGHT_OBS_SIZE];
    float actions[REK_FIGHT_NUM_AGENTS * REK_FIGHT_NUM_ACTIONS];
    float rewards[REK_FIGHT_NUM_AGENTS];
    float terminals[REK_FIGHT_NUM_AGENTS];
    attach_buffers(env, observations, actions, rewards, terminals);
    idle_actions(actions);
    c_reset(env);
    clock_t start = clock();
    for (int step = 0; step < steps; step++) {
        actions[0] = (float)((step / 100) % 3);
        actions[2] = (float)((step / 70) % 3);
        actions[3] = step % 160 == 0 ? (float)(1 + (step / 160) % 6) : 0.0f;
        actions[4] = (float)(2 - (step / 90) % 3);
        actions[6] = (float)((step / 60) % 3);
        actions[7] = step % 190 == 0 ? (float)(1 + (step / 190) % 6) : 0.0f;
        c_step(env);
    }
    double seconds = (double)(clock() - start) / (double)CLOCKS_PER_SEC;
    double steps_per_second = seconds > 0.0 ? steps / seconds : 0.0;
    printf(
        "{\"schema\":\"rek_fight.smoke_benchmark.v1\","
        "\"steps\":%d,\"seconds\":%.9g,\"steps_per_second\":%.9g,"
        "\"observation_size\":%d,\"finite\":%s}\n",
        steps,
        seconds,
        steps_per_second,
        REK_FIGHT_OBS_SIZE,
        rek_fight_state_is_finite(env) ? "true" : "false"
    );
}

static void run_control_diagnostic_case(
        RekFight* env,
        const char* name,
        int forward,
        int strafe,
        int yaw,
        int move,
        int steps) {
    float observations[REK_FIGHT_NUM_AGENTS * REK_FIGHT_OBS_SIZE];
    float actions[REK_FIGHT_NUM_AGENTS * REK_FIGHT_NUM_ACTIONS];
    float rewards[REK_FIGHT_NUM_AGENTS];
    float terminals[REK_FIGHT_NUM_AGENTS];
    attach_buffers(env, observations, actions, rewards, terminals);
    idle_actions(actions);
    env->max_steps = 0;
    c_reset(env);

    double start_x = env->data->qpos[0];
    double start_y = env->data->qpos[1];
    double minimum_height[2] = {
        rek_fight_root_height(env, 0),
        rek_fight_root_height(env, 1),
    };
    double minimum_up_z[2] = {
        rek_fight_root_up_z(env, 0),
        rek_fight_root_up_z(env, 1),
    };
    int first_fallen[2] = {-1, -1};
    int terminal_tick = -1;
    actions[0] = (float)forward;
    actions[1] = (float)strafe;
    actions[2] = (float)yaw;
    actions[3] = (float)move;
    for (int step = 1; step <= steps; step++) {
        c_step(env);
        for (int agent = 0; agent < 2; agent++) {
            double height = rek_fight_root_height(env, agent);
            double up_z = rek_fight_root_up_z(env, agent);
            if (height < minimum_height[agent]) minimum_height[agent] = height;
            if (up_z < minimum_up_z[agent]) minimum_up_z[agent] = up_z;
            if (env->agent[agent].fallen && first_fallen[agent] < 0) {
                first_fallen[agent] = step;
            }
        }
        if (terminals[0] != 0.0f && terminal_tick < 0) terminal_tick = step;
        actions[3] = 0.0f;
    }
    printf(
        "{\"schema\":\"rek_fight.control_diagnostic.v1\","
        "\"case\":\"%s\",\"steps\":%d,\"first_fallen\":[%d,%d],"
        "\"terminal_tick\":%d,\"minimum_height\":[%.9g,%.9g],"
        "\"minimum_up_z\":[%.9g,%.9g],\"root_delta_xy\":[%.9g,%.9g]}\n",
        name,
        steps,
        first_fallen[0], first_fallen[1],
        terminal_tick,
        minimum_height[0], minimum_height[1],
        minimum_up_z[0], minimum_up_z[1],
        env->data->qpos[0] - start_x,
        env->data->qpos[1] - start_y
    );
}

static void run_control_diagnostics(RekFight* env) {
    run_control_diagnostic_case(env, "idle", 1, 1, 1, 0, 500);
    run_control_diagnostic_case(env, "forward", 2, 1, 1, 0, 500);
    run_control_diagnostic_case(env, "backward", 0, 1, 1, 0, 500);
    run_control_diagnostic_case(env, "strafe_positive", 1, 2, 1, 0, 500);
    run_control_diagnostic_case(env, "strafe_negative", 1, 0, 1, 0, 500);
    run_control_diagnostic_case(env, "yaw_positive", 1, 1, 2, 0, 500);
    run_control_diagnostic_case(env, "yaw_negative", 1, 1, 0, 0, 500);
    for (int move = 1; move <= 6; move++) {
        char name[32];
        snprintf(name, sizeof(name), "move_%d", move);
        run_control_diagnostic_case(env, name, 1, 1, 1, move, 500);
    }
}

int main(int argc, char** argv) {
    RekFight env;
    memset(&env, 0, sizeof(env));
    configure_env(&env, 1500);
    rek_fight_init(&env);
    if (argc == 3 && strcmp(argv[1], "--benchmark") == 0) {
        int steps = (int)strtol(argv[2], NULL, 10);
        assert(steps > 0);
        run_benchmark(&env, steps);
        c_close(&env);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--diagnose-controls") == 0) {
        run_control_diagnostics(&env);
        c_close(&env);
        return 0;
    }
    assert(REK_FIGHT_OBS_SIZE == 173);
    test_descendant_limb_geometries(&env);
    test_complete_state_observation(&env);
    test_move_action_mask();
    test_zero_time_impact(&env);
    test_router_and_idle(&env);
    test_move_request(&env);
    test_scripted_approach(&env);
    test_framework_controller(&env);
    c_close(&env);
    printf("rek_fight tests passed\n");
    return 0;
}
