/*
 * WEF gprof / throughput harness.
 *
 * Builds a pool of envs totaling 8192 fish, then steps:
 *   for t in 0..T:
 *     for e in 0..num_envs:   // num_envs = 8192 / fish_per_env
 *       step(envs[e])
 *
 * Outer-t / inner-env order avoids overcounting L1/L2 cache hits from
 * hammering the same env repeatedly.
 *
 * Build + gprof:
 *   ./ocean/wef/run_profile.sh --gprof
 * Throughput only:
 *   ./ocean/wef/run_profile.sh
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "wef.h"

#define DEFAULT_TOTAL_FISH 8192
#define DEFAULT_NUM_FISH 4
#define DEFAULT_STEPS 64
#define DEFAULT_WARMUP 2

typedef struct {
    int total_fish;
    int num_fish;
    int steps;
    int warmup;
} Options;

static double timespec_seconds(struct timespec t0, struct timespec t1) {
    return (double)(t1.tv_sec - t0.tv_sec) +
        (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

static float random_action(unsigned int* rng) {
    return 2.0f * (float)rand_r(rng) / (float)RAND_MAX - 1.0f;
}

static void configure_env(Wef* env, unsigned int seed, int num_fish) {
    Dict kwargs = {0};
    dict_set(&kwargs, "num_agents", num_fish);
    dict_set(&kwargs, "min_arena_width", 70);
    dict_set(&kwargs, "min_arena_height", 70);
    dict_set(&kwargs, "max_arena_width", 70);
    dict_set(&kwargs, "max_arena_height", 70);
    dict_set(&kwargs, "food_distribution", FOOD_RANDOM);
    dict_set(&kwargs, "num_food", 64);
    dict_set(&kwargs, "patch_radius", 6);
    dict_set(&kwargs, "patch_radius_std", 1.5);
    dict_set(&kwargs, "patch_density", 0.001);
    dict_set(&kwargs, "electric_field_radius", 15);
    dict_set(&kwargs, "reflection_wall_range", 100);
    dict_set(&kwargs, "field_fish_range", 100);
    dict_set(&kwargs, "field_food_range", 5);
    dict_set(&kwargs, "episode_length", 4096);
    memset(env, 0, sizeof(*env));
    env->rng = seed ? seed : 1u;
    puf_init(env, &kwargs);
    dict_clear(&kwargs);
}

static void bind_agents(
    Wef* env, obs_t* obs, float* act, float* rew, float* term
) {
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].observations = obs + i * OBS_SIZE;
        env->agents[i].actions = act + i * NUM_ATNS;
        env->agents[i].rewards = rew + i;
        env->agents[i].terminals = term + i;
        env->agents[i].action_mask = NULL;
        env->agents[i].policy = 0;
    }
}

static void fill_random_actions(Wef* env, unsigned int* rng) {
    for (int a = 0; a < env->num_agents; a++) {
        float* action = env->agents[a].actions;
        for (int k = 0; k < ACTION_SIZE; k++) {
            action[k] = random_action(rng);
        }
    }
}

static void print_usage(const char* argv0) {
    printf(
        "Usage: %s [options]\n"
        "  --total-fish N   Total fish across all envs (default %d)\n"
        "  --num-fish N     Agents per env (default %d)\n"
        "  --steps N        Timed outer timesteps (default %d)\n"
        "  --warmup N       Untimed outer timesteps (default %d)\n"
        "  -h, --help\n",
        argv0,
        DEFAULT_TOTAL_FISH,
        DEFAULT_NUM_FISH,
        DEFAULT_STEPS,
        DEFAULT_WARMUP
    );
}

static bool parse_options(int argc, char** argv, Options* opt) {
    *opt = (Options){
        .total_fish = DEFAULT_TOTAL_FISH,
        .num_fish = DEFAULT_NUM_FISH,
        .steps = DEFAULT_STEPS,
        .warmup = DEFAULT_WARMUP,
    };
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            print_usage(argv[0]);
            return false;
        } else if (!strcmp(argv[i], "--total-fish") && i + 1 < argc) {
            opt->total_fish = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--num-fish") && i + 1 < argc) {
            opt->num_fish = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
            opt->steps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
            opt->warmup = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return false;
        }
    }
    if (opt->total_fish <= 0 || opt->num_fish <= 0 || opt->steps <= 0) {
        fprintf(stderr, "total-fish, num-fish, steps must be positive\n");
        return false;
    }
    if (opt->total_fish % opt->num_fish != 0) {
        fprintf(stderr, "total-fish (%d) must be divisible by num-fish (%d)\n",
            opt->total_fish, opt->num_fish);
        return false;
    }
    if (opt->num_fish > MAX_AGENTS) {
        fprintf(stderr, "num-fish (%d) > MAX_AGENTS (%d)\n",
            opt->num_fish, MAX_AGENTS);
        return false;
    }
    return true;
}

/* One outer timestep over the whole env pool (avoids cache overcount). */
static void step_pool(Wef* envs, unsigned int* rngs, int num_envs) {
    for (int e = 0; e < num_envs; e++) {
        fill_random_actions(&envs[e], &rngs[e]);
        puf_step(&envs[e]);
    }
}

int main(int argc, char** argv) {
    Options opt;
    if (!parse_options(argc, argv, &opt)) {
        return 1;
    }

    int num_envs = opt.total_fish / opt.num_fish;
    Wef* envs = (Wef*)calloc((size_t)num_envs, sizeof(Wef));
    unsigned int* rngs =
        (unsigned int*)calloc((size_t)num_envs, sizeof(unsigned int));
    if (!envs || !rngs) {
        fprintf(stderr, "allocation failed (num_envs=%d sizeof(Wef)=%zu)\n",
            num_envs, sizeof(Wef));
        free(envs);
        free(rngs);
        return 1;
    }

    printf("wef profile\n");
    printf("  total_fish=%d  num_fish_per_env=%d  num_envs=%d\n",
        opt.total_fish, opt.num_fish, num_envs);
    printf("  warmup=%d  steps=%d  sizeof(Wef)=%zu\n",
        opt.warmup, opt.steps, sizeof(Wef));
    printf("  loop: for t: for e in 0..%d: step(envs[e])\n", num_envs);
    fflush(stdout);

    size_t per_env_obs = (size_t)opt.num_fish * OBS_SIZE;
    size_t per_env_act = (size_t)opt.num_fish * NUM_ATNS;
    obs_t* all_obs = (obs_t*)calloc((size_t)num_envs * per_env_obs, sizeof(obs_t));
    float* all_act = (float*)calloc((size_t)num_envs * per_env_act, sizeof(float));
    float* all_rew = (float*)calloc((size_t)num_envs * opt.num_fish, sizeof(float));
    float* all_term = (float*)calloc((size_t)num_envs * opt.num_fish, sizeof(float));
    if (!all_obs || !all_act || !all_rew || !all_term) {
        fprintf(stderr, "agent buffer allocation failed\n");
        return 1;
    }
    for (int e = 0; e < num_envs; e++) {
        unsigned int seed = (unsigned int)(e + 1);
        configure_env(&envs[e], seed, opt.num_fish);
        bind_agents(
            &envs[e],
            all_obs + (size_t)e * per_env_obs,
            all_act + (size_t)e * per_env_act,
            all_rew + (size_t)e * opt.num_fish,
            all_term + (size_t)e * opt.num_fish
        );
        rngs[e] = seed * 0x9e3779b9u + 1u;
        puf_reset(&envs[e]);
    }

    /* Warmup: same outer-t / inner-e order, not timed. */
    for (int t = 0; t < opt.warmup; t++) {
        step_pool(envs, rngs, num_envs);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int t = 0; t < opt.steps; t++) {
        step_pool(envs, rngs, num_envs);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = timespec_seconds(t0, t1);
    long long env_steps = (long long)opt.steps * (long long)num_envs;
    long long agent_steps = env_steps * (long long)opt.num_fish;
    double env_sps = elapsed > 0.0 ? (double)env_steps / elapsed : 0.0;
    double agent_sps = elapsed > 0.0 ? (double)agent_steps / elapsed : 0.0;

    printf("\n  results:\n");
    printf("    elapsed=%.4fs\n", elapsed);
    printf("    env_steps=%lld  agent_steps=%lld\n", env_steps, agent_steps);
    printf("    env_SPS=%.1f  agent_SPS=%.1f\n", env_sps, agent_sps);

    free(all_obs);
    free(all_act);
    free(all_rew);
    free(all_term);
    free(envs);
    free(rngs);
    return 0;
}
