#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#define SELF_OBS 27
#define MAP_OBS 8

static int new_agent_obs[SELF_OBS];
static int new_map_obs[MAP_OBS];

typedef struct test_struct test_struct;
struct test_struct {
    int* observations;
    int num_agents;
};

void print_obs(test_struct* env, int agent_idx) {
    printf("agent %d obs: ", agent_idx);
    for (int i = 0; i < (SELF_OBS); i++) {
        printf("%d ", env->observations[agent_idx*(SELF_OBS) + i]);
    }
    printf("\n");
}

void print_map_obs(test_struct* env) {
    printf("map obs: ");
    for (int i = 0; i < (MAP_OBS); i++) {
        printf("%d ", env->observations[env->num_agents*(SELF_OBS) + i]);
    }
    printf("\n");
}

void add_obs(test_struct* env) {
    int obs_idx = 0;
    for (int i = 0; i < env->num_agents * (SELF_OBS); i++) {
        if (i % (SELF_OBS) == 0 ) {
            obs_idx++;
        }
        env->observations[i] = obs_idx;
    }
    for (int i = 0; i < (MAP_OBS); i++) {
        env->observations[env->num_agents*(SELF_OBS) + i] = rand() % 100;
    }
}

void init(test_struct* env) {
    env->observations = (int*)calloc(env->num_agents*(SELF_OBS) + MAP_OBS, sizeof(int));
    print_obs(env, 0);
    print_obs(env, 1);
    print_map_obs(env);
    add_obs(env);
    print_obs(env, 0);
    print_obs(env, 1);
    print_map_obs(env);
}

void free_initialized(test_struct* env) {
    free(env->observations);
}

void step(test_struct* env) {
    int rand_agents[5] = {rand() % env->num_agents, rand() % env->num_agents, rand() % env->num_agents, rand() % env->num_agents, rand() % env->num_agents};
    for(int i=0;i<5;i++) {
        for (int j=0;j<SELF_OBS;j++) {
            new_agent_obs[j] = rand() % 100;
        }
        // printf("selected agents %d\n", rand_agents[i]);
        // printf("old obs\n");
        // print_obs(env, rand_agents[i]);
        memcpy(env->observations + rand_agents[i]*(SELF_OBS), new_agent_obs, (SELF_OBS)*sizeof(int));
        // printf("new obs\n");
        // print_obs(env, rand_agents[i]);
    }
    for (int j=0;j<MAP_OBS;j++) {
        new_map_obs[j] = rand() % 100;
    }
    memcpy(env->observations + env->num_agents*(SELF_OBS), new_map_obs, (MAP_OBS)*sizeof(int));

    // print_obs(env, rand_agents[0]);
    // print_obs(env, rand_agents[1]);
    // print_map_obs(env);
}
