#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>
#include "config.h"

#define PUFFER_VECENV_INCLUDE
#include ENV_HEADER
#undef PUFFER_VECENV_INCLUDE

#include "puffernet.h"

static int has_suffix(const char* s, const char* suffix) {
    size_t n = strlen(s);
    size_t m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

static void find_latest_checkpoint(const char* dir,
        char* out, size_t out_size, time_t* best_time) {
    DIR* dp = opendir(dir);
    if (!dp) {
        return;
    }

    struct dirent* ent = NULL;
    while ((ent = readdir(dp))) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }

        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);

        struct stat st;
        if (stat(path, &st) != 0) {
            continue;
        }

        if (S_ISDIR(st.st_mode)) {
            find_latest_checkpoint(path, out, out_size, best_time);
        } else if (S_ISREG(st.st_mode) && has_suffix(path, ".bin") &&
                st.st_ctime >= *best_time) {
            *best_time = st.st_ctime;
            snprintf(out, out_size, "%s", path);
        }
    }

    closedir(dp);
}

static const char* model_path(Config* cfg, const char* env_name,
        char* out, size_t out_size) {
    const char* path = puf_config_str(cfg, "base", "load_model_path");
    if (path && strcmp(path, "None") != 0) {
        if (strcmp(path, "latest") != 0) {
            return path;
        }

        char root[2048];
        snprintf(root, sizeof(root), "%s/%s",
            puf_config_str(cfg, "base", "checkpoint_dir"), env_name);
        out[0] = 0;
        time_t best_time = 0;
        find_latest_checkpoint(root, out, out_size, &best_time);
        if (!out[0]) {
            fprintf(stderr, "no .bin checkpoints found in %s\n", root);
            exit(1);
        }
        return out;
    }

    snprintf(out, out_size, "resources/%s/%s_weights.bin", env_name, env_name);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s ENV [section.key=value ...]\n", argv[0]);
        return 1;
    }

    const char* env_name = argv[1];
    Config cfg = {0};
    puf_config_load_env(&cfg, env_name, argc - 2, argv + 2);

    if (sizeof(obs_t) != sizeof(float)) {
        fprintf(stderr, "cpu eval currently requires float observations\n");
        return 1;
    }

    char path_buf[1024];
    const char* path = model_path(&cfg, env_name, path_buf, sizeof(path_buf));
    Weights* weights = load_weights(path);
    if (!weights) {
        return 1;
    }

    int act_sizes[] = ACT_SIZES;
    int num_actions = (int)(sizeof(act_sizes) / sizeof(act_sizes[0]));

    int hidden_size = (int)puf_config_get(&cfg, "policy", "hidden_size");
    int num_layers = (int)puf_config_get(&cfg, "policy", "num_layers");

    Env env = {0};
    env.rng = 0;
    puf_init(&env, &cfg.env);

    obs_t observations[env.num_agents * OBS_SIZE];
    float actions[env.num_agents * NUM_ATNS];
    float rewards[env.num_agents];
    float terminals[env.num_agents];
    memset(observations, 0, sizeof(observations));
    memset(actions, 0, sizeof(actions));
    memset(rewards, 0, sizeof(rewards));
    memset(terminals, 0, sizeof(terminals));
    for (int i = 0; i < env.num_agents; i++) {
        env.agents[i].observations = observations + i * OBS_SIZE;
        env.agents[i].actions = actions + i * NUM_ATNS;
        env.agents[i].rewards = rewards + i;
        env.agents[i].terminals = terminals + i;
        env.agents[i].action_mask = NULL;
        env.agents[i].policy = 0;
    }
    puf_reset(&env);

    PufferNet* net = make_puffernet(weights, env.num_agents, OBS_SIZE,
        hidden_size, num_layers, act_sizes, num_actions);

    int frame = 0;
    puf_render(&env);
    while (!WindowShouldClose()) {
        if (frame % 4 == 0) {
            forward_puffernet(net, observations, actions);
        }
        frame = (frame + 1) % 4;
        puf_step(&env);
        puf_render(&env);
    }

    puf_close(&env);
    free_puffernet(net);
    free(weights);
    puf_config_free(&cfg);
    return 0;
}
