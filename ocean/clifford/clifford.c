#include <stdio.h>
#include "clifford.h"

int main(void) {
    Clifford env = {0};
    env.num_agents = 1;
    env.rng = 1;
    set_difficulty_level(&env, 0.0);
    env.max_steps = 32;
    env.single_qubit_cost = 0.01f;
    env.cz_cost = 1.0f;
    env.goal_bonus = 25.0f;
    env.failure_penalty = 0.0f;
    env.hamming_scale = 1.0f;
    rng_seed(&env.xor_rng, 1);
    init(&env);
    env.agents[0].observations = (unsigned char*)calloc(OBS_SIZE, sizeof(unsigned char));
    env.agents[0].actions = (float*)calloc(1, sizeof(float));
    env.agents[0].rewards = (float*)calloc(1, sizeof(float));
    env.agents[0].terminals = (float*)calloc(1, sizeof(float));
    puf_reset(&env);
    assert(is_identity(&env));
    assert(tableau_hamming(&env) == 0);
    env.agents[0].actions[0] = 0.0f;
    puf_step(&env);
    assert(!is_identity(&env));
    assert(tableau_hamming(&env) > 0);
    env.agents[0].actions[0] = 0.0f;
    puf_step(&env);
    assert(is_identity(&env));
    assert(tableau_hamming(&env) == 0);

    copy_identity_cols(&env);
    int path[8];
    for (int i = 0; i < 8; ++i) {
        path[i] = sample_action(&env);
        apply_action(&env, path[i]);
    }
    assert(!is_identity(&env));
    for (int i = 7; i >= 0; --i) {
        apply_action(&env, path[i]);
    }
    assert(is_identity(&env));
    assert(tableau_hamming(&env) == 0);

    set_difficulty_level(&env, 8.0);
    puf_reset(&env);
    assert(!is_identity(&env));
    for (int step = 0; step < 1000; ++step) {
        env.agents[0].actions[0] = (float)sample_action(&env);
        puf_step(&env);
    }

    printf("clifford smoke complete: n=%d actions=%d obs=%d reward=%f terminal=%f\n",
        CLIFFORD_N_QUBITS, CLIFFORD_NUM_ACTIONS, OBS_SIZE,
        env.agents[0].rewards[0], env.agents[0].terminals[0]);
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    puf_close(&env);
    return 0;
}
