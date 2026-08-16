#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "pufferenv.h"

#define Log OsrsSharedLog
#include "../osrs/encounters/encounter_zulrah.h"
#undef Log

#define OBS_SIZE ZUL_NUM_OBS
#define NUM_ATNS ZUL_NUM_ACTION_HEADS
#define ACT_SIZES ZUL_ACTION_DIMS_INIT
typedef float obs_t;

#define ZUL_ENV_STATE(env) ((EncounterState*)&(env)->state)
#define ZUL_ENV_CONTEXT(env) ((EncounterContext*)&(env)->context)

struct Log {
    float episode_return;
    float episode_length;
    float wins;
    float kills;
    float score;
    float damage_dealt;
    float damage_received;
    float cloud_occupancy_frac;
    float cloud_damage_received;
    float tier_n[ZUL_NUM_GEAR_TIERS];
    float tier_wins[ZUL_NUM_GEAR_TIERS];
    float tier_score[ZUL_NUM_GEAR_TIERS];
    float n;
};

struct Env {
    Log log;
    int num_agents;
    unsigned int rng;
    Agent agents[1];
    int tag;
    int boundary_reached;

    ZulrahState state;
    ZulrahContext context;
    int acts_staging[ZUL_NUM_ACTION_HEADS];
};

static inline uint32_t zul_lowbias32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static void zul_write_native_action_mask(Env* env, unsigned char* mask_out) {
    if (!mask_out) return;
    zul_write_mask_bytes(
        ZUL_ENV_STATE(env), ZUL_ENV_CONTEXT(env), mask_out);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->agents[0].policy = 0;
    ENCOUNTER_ZULRAH.init_context(ZUL_ENV_CONTEXT(env));
    ENCOUNTER_ZULRAH.init_state(ZUL_ENV_STATE(env), ZUL_ENV_CONTEXT(env));

    uint32_t seed_offset = 0;
    const char* seed_offset_str = getenv("PUFFER_ENV_SEED_OFFSET");
    if (seed_offset_str) {
        seed_offset = (uint32_t)strtoul(seed_offset_str, NULL, 10);
    }
    uint32_t env_seed = zul_lowbias32(env->rng + seed_offset);
    if (env_seed == 0) {
        env_seed = 1;
    }
    env->state.rng_state = env_seed;

    memset(&env->log, 0, sizeof(Log));

    static const char* const int_keys[] = {
        "gear_tier",
        "gear_tier_mode",
        "episode_mode",
    };
    for (size_t k = 0; k < sizeof(int_keys) / sizeof(*int_keys); k++) {
        ENCOUNTER_ZULRAH.put_int(
            ZUL_ENV_STATE(env), ZUL_ENV_CONTEXT(env),
            int_keys[k], (int)dict_get(kwargs, int_keys[k]));
    }

    static const char* const float_keys[] = {
        "gear_tier_weight_0",
        "gear_tier_weight_1",
        "gear_tier_weight_2",
        "reward_win",
        "reward_loss_penalty",
        "reward_damage_dealt",
        "reward_correct_style",
        "reward_damage_received_penalty",
        "reward_cloud_occupancy_penalty",
    };
    for (size_t k = 0; k < sizeof(float_keys) / sizeof(*float_keys); k++) {
        ENCOUNTER_ZULRAH.put_float(
            ZUL_ENV_STATE(env), ZUL_ENV_CONTEXT(env),
            float_keys[k], (float)dict_get(kwargs, float_keys[k]));
    }

    ENCOUNTER_ZULRAH.finalize_context(ZUL_ENV_STATE(env), ZUL_ENV_CONTEXT(env));
}

void puf_reset(Env* env) {
    Agent* agent = &env->agents[0];
    ENCOUNTER_ZULRAH.reset(ZUL_ENV_STATE(env), ZUL_ENV_CONTEXT(env), 0);
    ENCOUNTER_ZULRAH.write_obs(
        ZUL_ENV_STATE(env), ZUL_ENV_CONTEXT(env), (float*)agent->observations);
    zul_write_native_action_mask(env, agent->action_mask);
    agent->rewards[0] = 0.0f;
    agent->terminals[0] = 0.0f;
}

void puf_step(Env* env) {
    Agent* agent = &env->agents[0];
    for (int i = 0; i < NUM_ATNS; i++) {
        env->acts_staging[i] = (int)agent->actions[i];
    }

    ENCOUNTER_ZULRAH.step(ZUL_ENV_STATE(env), ZUL_ENV_CONTEXT(env), env->acts_staging);

    float* obs = (float*)agent->observations;
    ENCOUNTER_ZULRAH.write_obs(ZUL_ENV_STATE(env), ZUL_ENV_CONTEXT(env), obs);
    zul_write_native_action_mask(env, agent->action_mask);

    agent->rewards[0] = ENCOUNTER_ZULRAH.get_reward(ZUL_ENV_STATE(env), ZUL_ENV_CONTEXT(env));
    int is_terminal = ENCOUNTER_ZULRAH.is_terminal(ZUL_ENV_STATE(env), ZUL_ENV_CONTEXT(env));
    agent->terminals[0] = (float)is_terminal;

    if (is_terminal) {
        ZulrahState* s = &env->state;
        ZulEpisodeOutcome outcome = zul_episode_outcome(s);
        int tier = s->gear_tier;
        if (tier < 0 || tier >= ZUL_NUM_GEAR_TIERS) {
            fprintf(stderr, "zulrah invalid sampled gear tier %d\n", tier);
            abort();
        }
        env->log.episode_return += s->episode_return;
        env->log.episode_length += (float)s->tick;
        env->log.wins += outcome.win;
        env->log.kills += (float)s->kills_this_episode;
        env->log.score += outcome.score;
        env->log.damage_dealt += s->total_damage_dealt;
        env->log.damage_received += s->total_damage_received;
        env->log.cloud_occupancy_frac += s->tick > 0
            ? (float)s->total_cloud_occupancy_ticks / (float)s->tick
            : 0.0f;
        env->log.cloud_damage_received += s->total_cloud_damage_received;
        env->log.tier_n[tier] += 1.0f;
        env->log.tier_wins[tier] += outcome.win;
        env->log.tier_score[tier] += outcome.score;
        env->log.n += 1.0f;

        ENCOUNTER_ZULRAH.reset(ZUL_ENV_STATE(env), ZUL_ENV_CONTEXT(env), 0);
        ENCOUNTER_ZULRAH.write_obs(ZUL_ENV_STATE(env), ZUL_ENV_CONTEXT(env), obs);
        zul_write_native_action_mask(env, agent->action_mask);
    }
}

void puf_render(Env* env) {
    (void)env;
}

void puf_close(Env* env) {
    ENCOUNTER_ZULRAH.destroy_context(ZUL_ENV_CONTEXT(env));
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "wins", log->wins);
    dict_set(out, "kills", log->kills);
    dict_set(out, "score", log->score);
    dict_set(out, "damage_dealt", log->damage_dealt);
    dict_set(out, "damage_received", log->damage_received);
    dict_set(out, "cloud_occupancy_frac", log->cloud_occupancy_frac);
    dict_set(out, "cloud_damage_received", log->cloud_damage_received);

    static const char* const TIER_WIN_KEYS[ZUL_NUM_GEAR_TIERS] = {
        "tier0_win_rate", "tier1_win_rate", "tier2_win_rate"};
    static const char* const TIER_SCORE_KEYS[ZUL_NUM_GEAR_TIERS] = {
        "tier0_score", "tier1_score", "tier2_score"};
    for (int t = 0; t < ZUL_NUM_GEAR_TIERS; t++) {
        float tn = log->tier_n[t];
        dict_set(out, TIER_WIN_KEYS[t],
            tn > 0.0f ? log->tier_wins[t] / tn : 0.0f);
        dict_set(out, TIER_SCORE_KEYS[t],
            tn > 0.0f ? log->tier_score[t] / tn : 0.0f);
    }
}
