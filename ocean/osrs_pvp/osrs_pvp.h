#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "pufferenv.h"

#define Log OsrsSharedLog
#include "../osrs/encounters/encounter_nh_pvp.h"
#undef Log

#define OBS_SIZE NH_PVP_NUM_OBS
#define NUM_ATNS OSRS_BASE_NUM_ACTION_HEADS
#define ACT_SIZES NH_PVP_ACTION_DIMS_INIT
typedef float obs_t;

#define NH_PVP_ENV_STATE(env) ((EncounterState*)&(env)->state)
#define NH_PVP_ENV_CONTEXT(env) ((EncounterContext*)&(env)->context)

struct Log {
    float episode_return;
    float episode_length;
    float wins;
    float damage_dealt;
    float damage_received;
    float prayer_correct;
    float prayer_total;
    float food_remaining;
    float karambwan_remaining;
    float brews_remaining;
    float spec_energy_remaining;
    float attacks_landed;
    float off_prayer_hits;
    float n;
};

struct Env {
    Log log;
    int num_agents;
    unsigned int rng;
    Agent agents[1];
    int tag;
    int boundary_reached;

    NhPvpState state;
    NhPvpContext context;
    int acts_staging[OSRS_BASE_NUM_ACTION_HEADS];
};

static inline uint32_t nh_pvp_lowbias32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static void nh_pvp_write_native_action_mask(Env* env, unsigned char* mask_out) {
    pvp_write_action_mask_bytes(
        mask_out, &env->state.env, 0, env->context.route_topology);
}

static void nh_pvp_apply_shaping(Env* env, int enabled, float scale) {
    RewardShapingConfig* shaping = &env->state.env.shaping;
    shaping->enabled = enabled;
    shaping->shaping_scale = scale;
    shaping->damage_dealt_coef = 0.005f;
    shaping->damage_received_coef = -0.005f;
    shaping->correct_prayer_bonus = 0.03f;
    shaping->wrong_prayer_penalty = -0.02f;
    shaping->prayer_switch_no_attack_penalty = -0.01f;
    shaping->off_prayer_hit_bonus = 0.03f;
    shaping->melee_frozen_penalty = -0.05f;
    shaping->wasted_eat_penalty = -0.001f;
    shaping->premature_eat_penalty = -0.02f;
    shaping->magic_no_staff_penalty = -0.05f;
    shaping->gear_mismatch_penalty = -0.05f;
    shaping->spec_off_prayer_bonus = 0.02f;
    shaping->spec_low_defence_bonus = 0.01f;
    shaping->spec_low_hp_bonus = 0.02f;
    shaping->smart_triple_eat_bonus = 0.05f;
    shaping->wasted_triple_eat_penalty = -0.0005f;
    shaping->damage_burst_bonus = 0.002f;
    shaping->damage_burst_threshold = 30;
    shaping->premature_eat_threshold = 0.7071f;
    shaping->ko_bonus = 0.15f;
    shaping->wasted_resources_penalty = -0.07f;
    shaping->prayer_penalty_enabled = 1;
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->agents[0].policy = 0;
    ENCOUNTER_NH_PVP.init_context(NH_PVP_ENV_CONTEXT(env));
    ENCOUNTER_NH_PVP.init_state(NH_PVP_ENV_STATE(env), NH_PVP_ENV_CONTEXT(env));

    uint32_t seed_offset = 0;
    const char* seed_offset_str = getenv("PUFFER_ENV_SEED_OFFSET");
    if (seed_offset_str) {
        seed_offset = (uint32_t)strtoul(seed_offset_str, NULL, 10);
    }
    uint32_t env_seed = nh_pvp_lowbias32(env->rng + seed_offset);
    if (env_seed == 0) env_seed = 1;

    static const char* const int_keys[] = {
        "opponent_type",
        "gear_tier",
        "is_lms",
        "use_c_opponent",
        "auto_reset",
    };
    for (size_t k = 0; k < sizeof(int_keys) / sizeof(*int_keys); k++) {
        ENCOUNTER_NH_PVP.put_int(
            NH_PVP_ENV_STATE(env), NH_PVP_ENV_CONTEXT(env),
            int_keys[k], (int)dict_get(kwargs, int_keys[k]));
    }
    ENCOUNTER_NH_PVP.put_int(
        NH_PVP_ENV_STATE(env), NH_PVP_ENV_CONTEXT(env), "seed", (int)env_seed);
    nh_pvp_apply_shaping(
        env,
        (int)dict_get(kwargs, "shaping_enabled"),
        (float)dict_get(kwargs, "shaping_scale"));

    memset(&env->log, 0, sizeof(env->log));
    ENCOUNTER_NH_PVP.finalize_context(
        NH_PVP_ENV_STATE(env), NH_PVP_ENV_CONTEXT(env));
}

void puf_reset(Env* env) {
    Agent* agent = &env->agents[0];
    ENCOUNTER_NH_PVP.reset(NH_PVP_ENV_STATE(env), NH_PVP_ENV_CONTEXT(env), 0);
    ENCOUNTER_NH_PVP.write_obs(
        NH_PVP_ENV_STATE(env), NH_PVP_ENV_CONTEXT(env),
        (float*)agent->observations);
    nh_pvp_write_native_action_mask(env, agent->action_mask);
    agent->rewards[0] = 0.0f;
    agent->terminals[0] = 0.0f;
}

void puf_step(Env* env) {
    Agent* agent = &env->agents[0];
    for (int i = 0; i < NUM_ATNS; i++) {
        env->acts_staging[i] = (int)agent->actions[i];
    }

    ENCOUNTER_NH_PVP.step(
        NH_PVP_ENV_STATE(env), NH_PVP_ENV_CONTEXT(env), env->acts_staging);
    ENCOUNTER_NH_PVP.write_obs(
        NH_PVP_ENV_STATE(env), NH_PVP_ENV_CONTEXT(env),
        (float*)agent->observations);
    nh_pvp_write_native_action_mask(env, agent->action_mask);

    agent->rewards[0] = ENCOUNTER_NH_PVP.get_reward(
        NH_PVP_ENV_STATE(env), NH_PVP_ENV_CONTEXT(env));
    int is_terminal = ENCOUNTER_NH_PVP.is_terminal(
        NH_PVP_ENV_STATE(env), NH_PVP_ENV_CONTEXT(env));
    agent->terminals[0] = (float)is_terminal;

    if (is_terminal) {
        OsrsSharedLog* episode = &env->state.env.log;
        env->log.episode_return += episode->episode_return;
        env->log.episode_length += episode->episode_length;
        env->log.wins += episode->wins;
        env->log.damage_dealt += episode->damage_dealt;
        env->log.damage_received += episode->damage_received;
        env->log.prayer_correct += episode->prayer_correct;
        env->log.prayer_total += episode->prayer_total;
        env->log.food_remaining += episode->food_remaining;
        env->log.karambwan_remaining += episode->karambwan_remaining;
        env->log.brews_remaining += episode->brews_remaining;
        env->log.spec_energy_remaining += episode->spec_energy_remaining;
        env->log.attacks_landed += episode->attacks_landed;
        env->log.off_prayer_hits += episode->off_prayer_hits;
        env->log.n += episode->n;
        memset(episode, 0, sizeof(*episode));

        ENCOUNTER_NH_PVP.reset(
            NH_PVP_ENV_STATE(env), NH_PVP_ENV_CONTEXT(env), 0);
        ENCOUNTER_NH_PVP.write_obs(
            NH_PVP_ENV_STATE(env), NH_PVP_ENV_CONTEXT(env),
            (float*)agent->observations);
        nh_pvp_write_native_action_mask(env, agent->action_mask);
    }
}

void puf_render(Env* env) {
    (void)env;
}

void puf_close(Env* env) {
    pvp_close(&env->state.env);
    ENCOUNTER_NH_PVP.destroy_context(NH_PVP_ENV_CONTEXT(env));
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "wins", log->wins);
    dict_set(out, "damage_dealt", log->damage_dealt);
    dict_set(out, "damage_received", log->damage_received);
    dict_set(out, "prayer_correct_rate",
        log->prayer_total > 0.0f ? log->prayer_correct / log->prayer_total : 0.0f);
    dict_set(out, "food_remaining", log->food_remaining);
    dict_set(out, "karambwan_remaining", log->karambwan_remaining);
    dict_set(out, "brews_remaining", log->brews_remaining);
    dict_set(out, "spec_remaining", log->spec_energy_remaining);
    dict_set(out, "attacks_landed", log->attacks_landed);
    dict_set(out, "off_prayer_hits", log->off_prayer_hits);
    dict_set(out, "damage_per_hit",
        log->attacks_landed > 0.0f ? log->damage_dealt / log->attacks_landed : 0.0f);
    float damage_fraction = log->damage_dealt / 99.0f;
    dict_set(out, "score",
        log->wins + (1.0f - log->wins) * damage_fraction * 0.5f);
}
