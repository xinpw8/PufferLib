#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "pufferenv.h"

#define Log OsrsSharedLog
#include "../osrs/encounters/encounter_colosseum.h"
#undef Log

#define OBS_SIZE COLO_NUM_OBS
#define NUM_ATNS COLO_NUM_ACTION_HEADS
#define ACT_SIZES COLO_ACTION_DIMS_INIT
typedef float obs_t;

#define COLO_ENV_STATE(env) ((EncounterState*)&(env)->state)
#define COLO_ENV_CONTEXT(env) ((EncounterContext*)&(env)->context)
#define COLO_MAX_CURRICULUM_TIERS 8

struct Log {
    float episode_return;
    float episode_length;
    float damage_dealt;
    float damage_received;
    float wins;
    float wave;
    float npc_kills;
    float prayer_correct;
    float prayer_total;
    float score;
    float sol_min_hp;
    float max_depth_reached;
    float current_set_argmax_dpt_hit;
    float current_set_argmax_dpt_n;
    float attacked_argmax_set_hit;
    float attacked_argmax_set_n;
    float pray_faced_by_type[COLO_NUM_NPC_TYPES];
    float pray_correct_by_type[COLO_NUM_NPC_TYPES];
    float offpray_damage_by_type[COLO_NUM_NPC_TYPES];
    float total_damage_by_type[COLO_NUM_NPC_TYPES];
    float death_by_type[COLO_NUM_NPC_TYPES];
    float typeless_damage_by_type[COLO_NUM_NPC_TYPES];
    float death_fatal_damage;
    float offpray_damage_conflict;
    float offpray_damage_solo;
    float death_on_conflict_tick;
    float death_dmg_unprayable;
    float death_dmg_offpray;
    float death_dmg_prayed;
    float death_dmg_self;
    float death_heal_remaining;
    float farm_damage;
    float n;
};

struct Env {
    Log log;
    int num_agents;
    unsigned int rng;
    Agent agents[1];
    int tag;
    int boundary_reached;

    ColosseumState state;
    ColosseumContext context;
    int config_start_wave;
    int acts_staging[COLO_NUM_ACTION_HEADS];
    uint64_t damage_scale_anneal_step;
    float max_episode_depth_seen;
};

static inline uint32_t col_lowbias32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static inline float col_curriculum_uniform(uint32_t env_index) {
    uint32_t h = col_lowbias32(env_index ^ 0x9e3779b9U);
    return (float)(h >> 8) * (1.0f / 16777216.0f);
}

static void col_write_action_mask_bytes(Env* env, unsigned char* mask_out) {
    if (!mask_out) return;
    float mask_f[COLO_ACTION_MASK_SIZE];
    ENCOUNTER_COLOSSEUM.write_mask(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), mask_f);
    for (int i = 0; i < COLO_ACTION_MASK_SIZE; i++) {
        mask_out[i] = mask_f[i] != 0.0f ? 1 : 0;
    }
}

static inline void col_log_dpt_sample(float* hit_acc, float* n_acc, int sample) {
    if (sample < 0) {
        return;
    }
    *hit_acc += sample ? 1.0f : 0.0f;
    *n_acc += 1.0f;
}

static void col_assign_curriculum_wave(Env* env, Dict* kwargs) {
    int classic_curriculum_mode = (int)dict_get(kwargs, "classic_curriculum_mode");
    if (classic_curriculum_mode < 0 || classic_curriculum_mode > 1) {
        fprintf(stderr, "colosseum: classic_curriculum_mode must be 0 or 1, got %d\n",
            classic_curriculum_mode);
        abort();
    }
    int num_tiers_config = (int)dict_get(kwargs, "curriculum_num_tiers");
    if (num_tiers_config < 0 || num_tiers_config > COLO_MAX_CURRICULUM_TIERS) {
        fprintf(stderr, "colosseum: curriculum_num_tiers must be 0..%d, got %d\n",
            COLO_MAX_CURRICULUM_TIERS, num_tiers_config);
        abort();
    }

    static const char* const wave_keys[COLO_MAX_CURRICULUM_TIERS] = {
        "curriculum_wave_1", "curriculum_wave_2", "curriculum_wave_3", "curriculum_wave_4",
        "curriculum_wave_5", "curriculum_wave_6", "curriculum_wave_7", "curriculum_wave_8",
    };
    static const char* const frac_keys[COLO_MAX_CURRICULUM_TIERS] = {
        "curriculum_frac_1", "curriculum_frac_2", "curriculum_frac_3", "curriculum_frac_4",
        "curriculum_frac_5", "curriculum_frac_6", "curriculum_frac_7", "curriculum_frac_8",
    };
    int waves[COLO_MAX_CURRICULUM_TIERS];
    float fracs[COLO_MAX_CURRICULUM_TIERS];
    int num_tiers = 0;
    for (int i = 0; i < num_tiers_config; i++) {
        int wave = (int)dict_get(kwargs, wave_keys[i]);
        float frac = (float)dict_get(kwargs, frac_keys[i]);
        if (frac > 0.0f) {
            waves[num_tiers] = wave;
            fracs[num_tiers] = frac;
            num_tiers++;
        }
    }

    if (classic_curriculum_mode == 0 || num_tiers == 0) {
        return;
    }
    float draw = col_curriculum_uniform(env->rng);
    float cumulative = 0.0f;
    for (int t = 0; t < num_tiers; t++) {
        cumulative += fracs[t];
        if (draw < cumulative) {
            ENCOUNTER_COLOSSEUM.put_int(
                COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), "start_wave", waves[t]);
            ENCOUNTER_COLOSSEUM.put_int(
                COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), "curriculum_agent", 1);
            return;
        }
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->agents[0].policy = 0;
    ENCOUNTER_COLOSSEUM.init_context(COLO_ENV_CONTEXT(env));
    ENCOUNTER_COLOSSEUM.init_state(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env));

    uint32_t seed_offset = 0;
    const char* seed_offset_str = getenv("PUFFER_ENV_SEED_OFFSET");
    if (seed_offset_str) {
        seed_offset = (uint32_t)strtoul(seed_offset_str, NULL, 10);
    }
    uint32_t env_seed = col_lowbias32(env->rng + seed_offset);
    if (env_seed == 0) {
        env_seed = 1;
    }
    env->state.rng_state = env_seed;

    memset(&env->log, 0, sizeof(Log));

    int start_wave = (int)dict_get(kwargs, "start_wave");
    ENCOUNTER_COLOSSEUM.put_int(
        COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), "start_wave", start_wave);
    env->config_start_wave = start_wave - 1;

    static const char* const float_keys[] = {
        "damage_reward_coeff",
        "wave_clear_bonus",
        "win_bonus",
        "death_penalty_coeff",
        "timeout_penalty",
        "boss_damage_reward_coeff",
        "boss_phase_bonus",
        "argmax_gear_reward_coeff",
        "offensive_boost_reward_coeff",
        "beginner_loadout_fraction",
        "late_start_supply_fraction_per_wave",
        "prayer_switch_fail_prob",
        "player_damage_received_scale",
        "damage_scale_anneal_start",
    };
    for (size_t k = 0; k < sizeof(float_keys) / sizeof(*float_keys); k++) {
        ENCOUNTER_COLOSSEUM.put_float(
            COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env),
            float_keys[k], (float)dict_get(kwargs, float_keys[k]));
    }

    static const char* const int_keys[] = {
        "farm_safe_damage_cap",
        "farm_cap_waves",
        "loadout_profile_mode",
        "step_out_forecast_obs_enabled",
        "threat_field_obs_enabled",
        "forecast_horizon",
        "mask_inventory_heads",
        "prayer_oracle_mode",
        "late_start_state_mode",
        "bis_gear_oracle_mode",
        "invuln_mode",
        "episode_max_ticks_override",
        "remove_brews",
        "damage_scale_anneal_ticks",
    };
    for (size_t k = 0; k < sizeof(int_keys) / sizeof(*int_keys); k++) {
        ENCOUNTER_COLOSSEUM.put_int(
            COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env),
            int_keys[k], (int)dict_get(kwargs, int_keys[k]));
    }

    col_assign_curriculum_wave(env, kwargs);
}

void puf_reset(Env* env) {
    Agent* agent = &env->agents[0];
    ENCOUNTER_COLOSSEUM.reset(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), 0);
    ENCOUNTER_COLOSSEUM.write_obs(
        COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), (float*)agent->observations);
    col_write_action_mask_bytes(env, agent->action_mask);
}

void puf_step(Env* env) {
    Agent* agent = &env->agents[0];
    for (int i = 0; i < NUM_ATNS; i++) {
        env->acts_staging[i] = (int)agent->actions[i];
    }

    int anneal_ticks = env->context.config.damage_scale_anneal_ticks;
    float anneal_start = env->context.config.damage_scale_anneal_start;
    env->damage_scale_anneal_step++;
    if (anneal_ticks > 0 && anneal_start < 1.0f) {
        float frac = (float)env->damage_scale_anneal_step / (float)anneal_ticks;
        if (frac > 1.0f) frac = 1.0f;
        env->context.config.player_damage_received_scale =
            anneal_start + (1.0f - anneal_start) * frac;
    }

    ENCOUNTER_COLOSSEUM.step(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), env->acts_staging);

    float* obs = (float*)agent->observations;
    ENCOUNTER_COLOSSEUM.write_obs(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), obs);
    col_write_action_mask_bytes(env, agent->action_mask);

    agent->rewards[0] = ENCOUNTER_COLOSSEUM.get_reward(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env));
    int is_terminal = ENCOUNTER_COLOSSEUM.is_terminal(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env));
    agent->terminals[0] = (float)is_terminal;

    if (env->state.start_wave == env->config_start_wave) {
        col_log_dpt_sample(
            &env->log.current_set_argmax_dpt_hit,
            &env->log.current_set_argmax_dpt_n,
            col_current_set_is_argmax_dpt_for_target(&env->state));
        col_log_dpt_sample(
            &env->log.attacked_argmax_set_hit,
            &env->log.attacked_argmax_set_n,
            col_attacked_with_argmax_set(&env->state));
    }

    if (is_terminal) {
        ColosseumState* s = &env->state;
        ColosseumLog* clog = (ColosseumLog*)ENCOUNTER_COLOSSEUM.get_log(
            COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env));
        if (s->start_wave == env->config_start_wave) {
            if (clog->max_wave_depth > env->max_episode_depth_seen) {
                env->max_episode_depth_seen = clog->max_wave_depth;
            }
            env->log.max_depth_reached += env->max_episode_depth_seen;
            env->log.n += 1.0f;
            env->log.episode_return += clog->episode_return;
            env->log.episode_length += (float)clog->episode_length;
            env->log.wins += (float)clog->win;
            env->log.wave += (float)clog->wave_reached;
            env->log.damage_dealt += clog->total_damage_dealt;
            env->log.damage_received += clog->total_damage_received;
            env->log.npc_kills += (float)clog->total_npc_kills;
            env->log.score += clog->outcome_score;
            env->log.sol_min_hp += (float)s->min_sol_hp_seen;
            env->log.prayer_correct += (float)clog->total_prayer_correct;
            env->log.prayer_total += (float)clog->total_npc_attacks;
            for (int t = 0; t < COLO_NUM_NPC_TYPES; t++) {
                env->log.pray_faced_by_type[t] += clog->pray_faced_by_type[t];
                env->log.pray_correct_by_type[t] += clog->pray_correct_by_type[t];
                env->log.offpray_damage_by_type[t] += clog->offpray_damage_by_type[t];
                env->log.total_damage_by_type[t] += clog->total_damage_by_type[t];
                env->log.death_by_type[t] += clog->death_by_type[t];
                env->log.typeless_damage_by_type[t] += clog->typeless_damage_by_type[t];
            }
            env->log.death_fatal_damage += clog->death_fatal_damage;
            env->log.offpray_damage_conflict += clog->offpray_damage_conflict;
            env->log.offpray_damage_solo += clog->offpray_damage_solo;
            env->log.death_on_conflict_tick += clog->death_on_conflict_tick;
            env->log.death_dmg_unprayable += clog->death_dmg_unprayable;
            env->log.death_dmg_offpray += clog->death_dmg_offpray;
            env->log.death_dmg_prayed += clog->death_dmg_prayed;
            env->log.death_dmg_self += clog->death_dmg_self;
            env->log.death_heal_remaining += clog->death_heal_remaining;
            env->log.farm_damage += clog->farm_damage;
        }
        ENCOUNTER_COLOSSEUM.reset(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), 0);
        ENCOUNTER_COLOSSEUM.write_obs(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), obs);
        col_write_action_mask_bytes(env, agent->action_mask);
    }
}

void puf_render(Env* env) {
    (void)env;
}

void puf_close(Env* env) {
    ENCOUNTER_COLOSSEUM.destroy_context(COLO_ENV_CONTEXT(env));
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "damage_dealt", log->damage_dealt);
    dict_set(out, "damage_received", log->damage_received);
    dict_set(out, "wins", log->wins);
    dict_set(out, "wave", log->wave);
    dict_set(out, "npc_kills", log->npc_kills);

    float prayer_rate = log->prayer_total > 0.0f
        ? log->prayer_correct / log->prayer_total : 0.0f;
    dict_set(out, "prayer_correct_rate", prayer_rate);

    dict_set(out, "score", log->score);
    dict_set(out, "sol_min_hp", log->sol_min_hp);
    dict_set(out, "max_depth_reached", log->max_depth_reached);
    dict_set(out, "current_set_is_argmax_dpt_for_target",
        log->current_set_argmax_dpt_n > 0.0f
            ? log->current_set_argmax_dpt_hit / log->current_set_argmax_dpt_n : 0.0f);
    dict_set(out, "attacked_with_argmax_set",
        log->attacked_argmax_set_n > 0.0f
            ? log->attacked_argmax_set_hit / log->attacked_argmax_set_n : 0.0f);

    static const char* const OFFPRAY_RATE_KEYS[COLO_NUM_NPC_TYPES] = {
        "offpray_rate_berserker", "offpray_rate_archer", "offpray_rate_seer",
        "offpray_rate_serpent", "offpray_rate_jaguar", "offpray_rate_javelin",
        "offpray_rate_shockwave", "offpray_rate_minotaur", "offpray_rate_manticore",
        "offpray_rate_sol", "offpray_rate_totem", "offpray_rate_bee"};
    static const char* const OFFPRAY_DMG_KEYS[COLO_NUM_NPC_TYPES] = {
        "offpray_dmg_berserker", "offpray_dmg_archer", "offpray_dmg_seer",
        "offpray_dmg_serpent", "offpray_dmg_jaguar", "offpray_dmg_javelin",
        "offpray_dmg_shockwave", "offpray_dmg_minotaur", "offpray_dmg_manticore",
        "offpray_dmg_sol", "offpray_dmg_totem", "offpray_dmg_bee"};
    static const char* const TOTAL_DMG_KEYS[COLO_NUM_NPC_TYPES] = {
        "total_dmg_berserker", "total_dmg_archer", "total_dmg_seer",
        "total_dmg_serpent", "total_dmg_jaguar", "total_dmg_javelin",
        "total_dmg_shockwave", "total_dmg_minotaur", "total_dmg_manticore",
        "total_dmg_sol", "total_dmg_totem", "total_dmg_bee"};
    for (int t = 0; t < COLO_NUM_NPC_TYPES; t++) {
        float faced = log->pray_faced_by_type[t];
        float off_rate = faced > 0.0f
            ? (faced - log->pray_correct_by_type[t]) / faced : 0.0f;
        dict_set(out, OFFPRAY_RATE_KEYS[t], off_rate);
        dict_set(out, OFFPRAY_DMG_KEYS[t], log->offpray_damage_by_type[t]);
        dict_set(out, TOTAL_DMG_KEYS[t], log->total_damage_by_type[t]);
    }

    static const char* const DEATH_BY_KEYS[COLO_NUM_NPC_TYPES] = {
        "death_by_berserker", "death_by_archer", "death_by_seer",
        "death_by_serpent", "death_by_jaguar", "death_by_javelin",
        "death_by_shockwave", "death_by_minotaur", "death_by_manticore",
        "death_by_sol", "death_by_totem", "death_by_bee"};
    for (int t = 0; t < COLO_NUM_NPC_TYPES; t++) {
        dict_set(out, DEATH_BY_KEYS[t], log->death_by_type[t]);
    }
    dict_set(out, "death_fatal_damage", log->death_fatal_damage);
    dict_set(out, "offpray_dmg_conflict", log->offpray_damage_conflict);
    dict_set(out, "offpray_dmg_solo", log->offpray_damage_solo);
    dict_set(out, "death_on_conflict_tick", log->death_on_conflict_tick);

    static const char* const TYPELESS_DMG_KEYS[COLO_NUM_NPC_TYPES] = {
        "typeless_dmg_berserker", "typeless_dmg_archer", "typeless_dmg_seer",
        "typeless_dmg_serpent", "typeless_dmg_jaguar", "typeless_dmg_javelin",
        "typeless_dmg_shockwave", "typeless_dmg_minotaur", "typeless_dmg_manticore",
        "typeless_dmg_sol", "typeless_dmg_totem", "typeless_dmg_bee"};
    for (int t = 0; t < COLO_NUM_NPC_TYPES; t++) {
        dict_set(out, TYPELESS_DMG_KEYS[t], log->typeless_damage_by_type[t]);
    }
    dict_set(out, "death_dmg_unprayable", log->death_dmg_unprayable);
    dict_set(out, "death_dmg_offpray", log->death_dmg_offpray);
    dict_set(out, "death_dmg_prayed", log->death_dmg_prayed);
    dict_set(out, "death_dmg_self", log->death_dmg_self);
    dict_set(out, "death_heal_remaining", log->death_heal_remaining);
    dict_set(out, "farm_damage", log->farm_damage);
}
