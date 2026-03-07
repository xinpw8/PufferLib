#include <stdlib.h>
#include <string.h>

// Minimal Rock-Paper-Scissors selfplay environment
// Obs: [one-hot last opponent move (3) | one-hot last own move (3) | one-hot last opponent move (3) | one-hot last own move (3)]
// First half is learner's view, second half is opponent's view (mirrored)
// Actions: [learner_action, opponent_action] each in {0=rock, 1=paper, 2=scissors}
// Reward: +1 win, -1 loss, 0 draw (from learner perspective)

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct Client Client;
typedef struct CRPS CRPS;
struct CRPS {
    float* observations;
    double* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    Log log;
    Client* client;

    int last_p1_action;
    int last_p2_action;
    int tick;
    int max_ticks;
    float cumulative_reward;
};

void allocate_crps(CRPS* env) {
    env->observations = (float*)calloc(12, sizeof(float));
    env->actions = (double*)calloc(2, sizeof(double));
    env->terminals = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
}

void free_allocated_crps(CRPS* env) {
    free(env->observations);
    free(env->actions);
    free(env->terminals);
    free(env->rewards);
}

void c_close(CRPS* env) {}

static void write_obs(CRPS* env) {
    memset(env->observations, 0, 12 * sizeof(float));
    // Learner view: [opponent's last move one-hot(3), own last move one-hot(3)]
    if (env->last_p2_action >= 0) env->observations[env->last_p2_action] = 1.0f;
    if (env->last_p1_action >= 0) env->observations[3 + env->last_p1_action] = 1.0f;
    // Opponent view (mirrored): [learner's last move one-hot(3), own last move one-hot(3)]
    if (env->last_p1_action >= 0) env->observations[6 + env->last_p1_action] = 1.0f;
    if (env->last_p2_action >= 0) env->observations[9 + env->last_p2_action] = 1.0f;
}

void init(CRPS* env) {
    env->last_p1_action = -1;
    env->last_p2_action = -1;
    env->tick = 0;
    env->cumulative_reward = 0.0f;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    write_obs(env);
}

void c_reset(CRPS* env) {
    env->log.episode_return += env->cumulative_reward;
    env->log.episode_length += env->tick;
    env->log.n += 1;
    init(env);
}

void c_step(CRPS* env) {
    int p1 = ((int)env->actions[0]) % 3;  // learner
    int p2 = ((int)env->actions[1]) % 3;  // opponent

    // RPS: 0=rock, 1=paper, 2=scissors
    // Winner: paper>rock, scissors>paper, rock>scissors
    float reward = 0.0f;
    if (p1 == p2) {
        reward = 0.0f;
    } else if ((p1 == 0 && p2 == 2) || (p1 == 1 && p2 == 0) || (p1 == 2 && p2 == 1)) {
        reward = 1.0f;
        env->log.perf += 1.0f;
        env->log.score += 1.0f;
    } else {
        reward = -1.0f;
        env->log.score -= 1.0f;
    }

    env->rewards[0] = reward;
    env->cumulative_reward += reward;
    env->last_p1_action = p1;
    env->last_p2_action = p2;
    env->tick++;

    if (env->tick >= env->max_ticks) {
        env->terminals[0] = 1.0f;
        c_reset(env);
    } else {
        env->terminals[0] = 0.0f;
        write_obs(env);
    }
}
