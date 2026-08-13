#include "env.h"

int b2InternalAssertFcn(const char* condition, const char* fileName, int lineNumber) {
    fprintf(stderr, "box2d assert %s at %s:%d\n", condition, fileName, lineNumber);
    return 1;
}

void puf_init(Env* env, Dict* kwargs) {
    uint8_t num_drones = (uint8_t)dict_get(kwargs, "num_drones");
    uint8_t num_agents = (uint8_t)dict_get(kwargs, "num_agents");
    int8_t map_idx = (int8_t)dict_get(kwargs, "map_idx");
    uint64_t seed = (uint64_t)dict_get(kwargs, "seed");
    bool enable_teams = (bool)dict_get(kwargs, "enable_teams");
    bool sitting_duck = (bool)dict_get(kwargs, "sitting_duck");
    bool is_training = (bool)dict_get(kwargs, "is_training");
    bool continuous = (bool)dict_get(kwargs, "continuous");
    initEnv(env, num_drones, num_agents, map_idx, seed, enable_teams,
            sitting_duck, is_training, continuous);
    static int maps_ready = 0;
    if (!maps_ready) {
        initMaps(env);
        maps_ready = 1;
    }
    setRewards(
        env,
        (float)dict_get(kwargs, "reward_win"),
        (float)dict_get(kwargs, "reward_self_kill"),
        (float)dict_get(kwargs, "reward_enemy_death"),
        (float)dict_get(kwargs, "reward_enemy_kill"),
        0.0f,
        0.0f,
        (float)dict_get(kwargs, "reward_death"),
        (float)dict_get(kwargs, "reward_energy_emptied"),
        (float)dict_get(kwargs, "reward_weapon_pickup"),
        (float)dict_get(kwargs, "reward_shield_break"),
        (float)dict_get(kwargs, "reward_shot_hit_coef"),
        (float)dict_get(kwargs, "reward_explosion_hit_coef"));
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "episode_length", log->length);
    dict_set(out, "ties", log->ties);
    dict_set(out, "perf", log->stats[0].wins);
    dict_set(out, "score", log->stats[0].wins);
}
