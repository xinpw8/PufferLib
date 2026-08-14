#include "onlyfish.h"
#include "puffercpu.h"
#include <dirent.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const char* dirpath = "resources/onlyfish/";
    char fish[32][64];
    int num_agents = 0;
    const char* hosted[] = {"danadvantage.bin", "lonelypuff.bin"};
    for (int i = 0; i < 2; i++) {
        snprintf(fish[num_agents++], 64, "%s", hosted[i]);
    }

    DIR* dir = opendir(dirpath);
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir))) {
            size_t len = strlen(entry->d_name);
            if (len <= 4 || strcmp(entry->d_name + len - 4, ".bin") != 0) {
                continue;
            }
            int seen = 0;
            for (int i = 0; i < num_agents; i++) {
                if (strcmp(fish[i], entry->d_name) == 0) {
                    seen = 1;
                }
            }
            if (!seen && num_agents < 32) {
                snprintf(fish[num_agents++], 64, "%s", entry->d_name);
            }
        }
        closedir(dir);
    }

    LinearLSTM** nets = calloc(num_agents, sizeof(LinearLSTM*));
    char** names = calloc(num_agents, sizeof(char*));
    int logit_sizes[2] = {9, 5};
    for (int i = 0; i < num_agents; i++) {
        char fullpath[256];
        snprintf(fullpath, sizeof(fullpath), "%s%s", dirpath, fish[i]);
        Weights* weights = load_weights(fullpath);
        nets[i] = make_linearlstm(weights, 1, 21, logit_sizes, 2);
        names[i] = strdup(fish[i]);
        char* dot = strrchr(names[i], '.');
        if (dot) {
            *dot = '\0';
        }
    }

    int num_goals = 4;
    int num_obs = 21;

    OnlyFish env = {
        .width = 1280,
        .height = 720,
        .num_agents = num_agents,
        .num_goals = num_goals,
        .names = names
    };
    init(&env);

    env.agents[0].observations = calloc(env.num_agents * num_obs, sizeof(float));
    env.agents[0].actions = calloc(2 * env.num_agents, sizeof(int));
    env.agents[0].rewards = calloc(env.num_agents, sizeof(float));
    env.agents[0].terminals = calloc(env.num_agents, sizeof(unsigned char));

    puf_reset(&env);
    puf_render(&env);

    while (!WindowShouldClose()) {
        for (int i = 0; i < num_agents; i++) {
            forward_linearlstm(nets[i], env.agents[0].observations + i * num_obs, env.agents[0].actions + 2 * i);
        }
        puf_step(&env);
        puf_render(&env);
    }

    for (int i = 0; i < num_agents; i++) {
        free_linearlstm(nets[i]);
        free(names[i]);
    }
    free(nets);
    free(names);
    free(env.agents[0].observations);
    free(env.agents[0].actions);
    free(env.agents[0].rewards);
    free(env.agents[0].terminals);
    puf_close(&env);
}
