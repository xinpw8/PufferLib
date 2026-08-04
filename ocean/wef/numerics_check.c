/*
 * Deterministic wef numerics fingerprint for refactor checks.
 *
 * Build:
 *   clang -O2 -mavx2 -DNDEBUG -I./raylib-5.5_linux_amd64/include -I./src \
 *     -I./vendor -I./ocean/wef ocean/wef/numerics_check.c \
 *     raylib-5.5_linux_amd64/lib/libraylib.a -lGL -lm -lpthread -ldl -lrt \
 *     -DPLATFORM_DESKTOP -o wef_numerics_check
 *
 * Run:
 *   ./wef_numerics_check            # print fingerprints
 *   ./wef_numerics_check --selftest  # bit-identical double-run
 *   ./wef_numerics_check --compare path.txt
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wef.h"

#define FNV_OFFSET 14695981039346656037ULL
#define FNV_PRIME 1099511628211ULL
#define NUM_FISH 4
#define NUM_STEPS 100
#define NUM_FOOD 16
#define NUM_SEEDS 3

static const unsigned int SEEDS[NUM_SEEDS] = {0u, 1u, 2u};

typedef struct SeedMetrics {
    unsigned int seed;
    double reward_sum;
    double final_obs_sum;
    int food_eaten;
    float episode_return;
    unsigned int rng_end;
    uint64_t obs_fnv1a;
    uint64_t rewards_fnv1a;
    uint64_t state_fnv1a;
} SeedMetrics;

static float random_action(unsigned int* rng) {
    return 2.0f * (float)rand_r(rng) / (float)RAND_MAX - 1.0f;
}

static uint64_t fnv1a_update(uint64_t hash, const void* data, size_t nbytes) {
    const unsigned char* bytes = (const unsigned char*)data;
    for (size_t i = 0; i < nbytes; i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

static unsigned int env_rng_from_seed(unsigned int seed) {
    return seed == 0u ? 0xA341316Cu : seed;
}

static void configure_env(Wef* env, unsigned int seed) {
    Dict kwargs = {0};
    dict_set(&kwargs, "num_agents", NUM_FISH);
    dict_set(&kwargs, "min_arena_width", 70);
    dict_set(&kwargs, "min_arena_height", 70);
    dict_set(&kwargs, "max_arena_width", 70);
    dict_set(&kwargs, "max_arena_height", 70);
    dict_set(&kwargs, "food_distribution", FOOD_UNIFORM);
    dict_set(&kwargs, "num_food", NUM_FOOD);
    dict_set(&kwargs, "patch_radius", 6);
    dict_set(&kwargs, "patch_radius_std", 1.5);
    dict_set(&kwargs, "patch_density", 0.001);
    dict_set(&kwargs, "electric_field_radius", 15);
    dict_set(&kwargs, "reflection_wall_range", 100);
    dict_set(&kwargs, "field_fish_range", 100);
    dict_set(&kwargs, "field_food_range", 5);
    dict_set(&kwargs, "episode_length", 4096);
    memset(env, 0, sizeof(*env));
    env->rng = env_rng_from_seed(seed);
    puf_init(env, &kwargs);
    dict_clear(&kwargs);
}

static SeedMetrics run_seed(unsigned int seed) {
    Wef env;
    configure_env(&env, seed);
    obs_t observations[NUM_FISH * OBS_SIZE];
    float actions[NUM_FISH * NUM_ATNS];
    float rewards[NUM_FISH];
    float terminals[NUM_FISH];
    memset(observations, 0, sizeof(observations));
    memset(actions, 0, sizeof(actions));
    memset(rewards, 0, sizeof(rewards));
    memset(terminals, 0, sizeof(terminals));
    for (int i = 0; i < NUM_FISH; i++) {
        env.agents[i].observations = observations + i * OBS_SIZE;
        env.agents[i].actions = actions + i * NUM_ATNS;
        env.agents[i].rewards = rewards + i;
        env.agents[i].terminals = terminals + i;
        env.agents[i].action_mask = NULL;
        env.agents[i].policy = 0;
    }
    puf_reset(&env);

    SeedMetrics m = {0};
    m.seed = seed;
    uint64_t obs_h = FNV_OFFSET;
    uint64_t rew_h = FNV_OFFSET;
    uint64_t state_h = FNV_OFFSET;
    unsigned int action_rng = seed;

    for (int t = 0; t < NUM_STEPS; t++) {
        for (int i = 0; i < NUM_FISH; i++) {
            float* action = env.agents[i].actions;
            for (int a = 0; a < ACTION_SIZE; a++) {
                action[a] = random_action(&action_rng);
            }
        }
        puf_step(&env);

        for (int i = 0; i < NUM_FISH; i++) {
            float r = env.agents[i].rewards[0];
            m.reward_sum += (double)r;
            rew_h = fnv1a_update(rew_h, &r, sizeof(r));
            obs_h = fnv1a_update(
                obs_h, env.agents[i].observations, OBS_SIZE * sizeof(float)
            );
            state_h = fnv1a_update(state_h, &env.fish[i].pos, sizeof(Vec2));
            state_h = fnv1a_update(
                state_h, &env.fish[i].orientation, sizeof(float)
            );
        }
        state_h = fnv1a_update(state_h, &env.tick, sizeof(env.tick));
        state_h = fnv1a_update(state_h, &env.food_eaten, sizeof(env.food_eaten));
        state_h = fnv1a_update(state_h, &env.rng, sizeof(env.rng));
    }

    m.food_eaten = env.food_eaten;
    m.episode_return = env.episode_return;
    m.rng_end = env.rng;
    m.obs_fnv1a = obs_h;
    m.rewards_fnv1a = rew_h;
    m.state_fnv1a = state_h;
    for (int i = 0; i < NUM_FISH; i++) {
        float* obs = (float*)env.agents[i].observations;
        for (int k = 0; k < OBS_SIZE; k++) {
            m.final_obs_sum += (double)obs[k];
        }
    }

    return m;
}

static void print_metrics(const SeedMetrics* m) {
    printf(
        "seed=%u reward_sum=%.17g final_obs_sum=%.17g food_eaten=%d "
        "episode_return=%.9g rng_end=%u obs_fnv=0x%016llx rew_fnv=0x%016llx "
        "state_fnv=0x%016llx\n",
        m->seed, m->reward_sum, m->final_obs_sum, m->food_eaten,
        m->episode_return, m->rng_end,
        (unsigned long long)m->obs_fnv1a,
        (unsigned long long)m->rewards_fnv1a,
        (unsigned long long)m->state_fnv1a
    );
}

static bool metrics_equal(const SeedMetrics* a, const SeedMetrics* b) {
    return a->seed == b->seed
        && a->food_eaten == b->food_eaten
        && a->rng_end == b->rng_end
        && a->obs_fnv1a == b->obs_fnv1a
        && a->rewards_fnv1a == b->rewards_fnv1a
        && a->state_fnv1a == b->state_fnv1a
        && memcmp(&a->reward_sum, &b->reward_sum, sizeof(double)) == 0
        && memcmp(&a->final_obs_sum, &b->final_obs_sum, sizeof(double)) == 0
        && memcmp(&a->episode_return, &b->episode_return, sizeof(float)) == 0;
}

static int parse_line(const char* line, SeedMetrics* m) {
    unsigned long long obs, rew, state;
    int n = sscanf(
        line,
        "seed=%u reward_sum=%lf final_obs_sum=%lf food_eaten=%d "
        "episode_return=%f rng_end=%u obs_fnv=0x%llx rew_fnv=0x%llx "
        "state_fnv=0x%llx",
        &m->seed, &m->reward_sum, &m->final_obs_sum, &m->food_eaten,
        &m->episode_return, &m->rng_end, &obs, &rew, &state
    );
    if (n != 9) return 0;
    m->obs_fnv1a = (uint64_t)obs;
    m->rewards_fnv1a = (uint64_t)rew;
    m->state_fnv1a = (uint64_t)state;
    return 1;
}

int main(int argc, char** argv) {
    bool selftest = false;
    const char* compare_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--selftest")) {
            selftest = true;
        } else if (!strcmp(argv[i], "--compare") && i + 1 < argc) {
            compare_path = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--selftest] [--compare path]\n", argv[0]);
            return 1;
        }
    }

    SeedMetrics metrics[NUM_SEEDS];
    for (int i = 0; i < NUM_SEEDS; i++) {
        metrics[i] = run_seed(SEEDS[i]);
        if (selftest) {
            SeedMetrics again = run_seed(SEEDS[i]);
            if (!metrics_equal(&metrics[i], &again)) {
                fprintf(stderr, "FAIL selftest seed=%u not bit-identical\n",
                    SEEDS[i]);
                return 1;
            }
        }
        print_metrics(&metrics[i]);
    }

    if (selftest) {
        printf("SELFTEST OK\n");
    }

    if (compare_path) {
        FILE* f = fopen(compare_path, "r");
        if (!f) {
            perror(compare_path);
            return 1;
        }
        char line[512];
        SeedMetrics baseline[NUM_SEEDS];
        int nbase = 0;
        while (nbase < NUM_SEEDS && fgets(line, sizeof(line), f)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            if (!parse_line(line, &baseline[nbase])) continue;
            nbase++;
        }
        fclose(f);
        if (nbase != NUM_SEEDS) {
            fprintf(stderr, "FAIL expected %d baseline seeds, got %d\n",
                NUM_SEEDS, nbase);
            return 1;
        }
        int mismatches = 0;
        for (int i = 0; i < NUM_SEEDS; i++) {
            if (!metrics_equal(&metrics[i], &baseline[i])) {
                fprintf(stderr, "MISMATCH seed=%u\n", metrics[i].seed);
                fprintf(stderr, "  got:  ");
                print_metrics(&metrics[i]);
                fprintf(stderr, "  base: ");
                print_metrics(&baseline[i]);
                mismatches++;
            }
        }
        if (mismatches) {
            fprintf(stderr, "FAIL %d/%d seeds differ from %s\n",
                mismatches, NUM_SEEDS, compare_path);
            return 1;
        }
        printf("COMPARE OK vs %s\n", compare_path);
    }
    return 0;
}
