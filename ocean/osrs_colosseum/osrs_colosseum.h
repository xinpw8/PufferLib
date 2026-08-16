#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "pufferenv.h"

/* Before the encounter header, so the COLO_PROFILE_MARK sites inside it compile in.
 * Every mark is a branch on a static int and does no work until
 * PUFFER_COLOSSEUM_PROFILE is set, so the training path pays a predictable branch. */
#include "colosseum_profile.h"

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
#define COLO_DPT_SAMPLE_INTERVAL 64

struct Log {
    float episode_return;
    float episode_length;
    float damage_dealt;
    float damage_received;
    float wins;
    float deaths;
    float timeouts;
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
    float npc_attack_death_by_type[COLO_NUM_NPC_TYPES];
    float typeless_damage_by_type[COLO_NUM_NPC_TYPES];
    float sol_damage_by_source[COLO_NUM_SOL_DAMAGE_SOURCES];
    float javelin_damage_by_source[COLO_NUM_JAVELIN_DAMAGE_SOURCES];
    float death_by_source[COLO_NUM_DAMAGE_SOURCES];
    float doom_death_by_source[COLO_NUM_DAMAGE_SOURCES];
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
    float reward_steps;
    float reward_clamped_steps;
    float reward_clamp_loss;
    float reward_raw_peak;
    float clamp_loss_wave_clear;
    float clamp_loss_win;
    float rew_damage;
    float rew_boss_phase;
    float rew_wave_clear;
    float rew_win;
    float rew_death;
    float rew_timeout;
    float laser_volleys;
    float laser_hits;
    float laser_dmg;
    float laser_aligned_at_fire;
    float laser_aligned_at_show;
    float laser_aligned_at_pre;
    float laser_aligned_at_damage;
    float laser_react_ok;
    float laser_react_fail;
    float avoid_total;
    float avoid_achieved;
    float avoid_missed;
    float avoid_impossible;
    float dmg_unprayable;
    float inv_memo_hits;
    float inv_memo_misses;
    float npc_blocked_calls;
    float npc_blocked_tiles;
    float npc_stamp_tiles;
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
    uint64_t dpt_sample_step;
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
    ENCOUNTER_COLOSSEUM.write_mask(
        COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), mask_f);
    for (int i = 0; i < COLO_ACTION_MASK_SIZE; i++)
        mask_out[i] = mask_f[i] != 0.0f ? 1 : 0;
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
        DictItem* wave_item = dict_find(kwargs, wave_keys[i]);
        DictItem* frac_item = dict_find(kwargs, frac_keys[i]);
        if (!wave_item || !frac_item) {
            fprintf(stderr,
                "colosseum: curriculum_num_tiers=%d requires %s and %s, which are not set\n",
                num_tiers_config, wave_keys[i], frac_keys[i]);
            abort();
        }
        int wave = (int)wave_item->value;
        float frac = (float)frac_item->value;
        if (frac > 0.0f) {
            waves[num_tiers] = wave;
            fracs[num_tiers] = frac;
            num_tiers++;
        }
    }

    if (classic_curriculum_mode == 0 || num_tiers == 0) {
        return;
    }
    float total_frac = 0.0f;
    for (int t = 0; t < num_tiers; t++) {
        total_frac += fracs[t];
    }
    if (total_frac > 0.9f) {
        float rescale = 0.9f / total_frac;
        for (int t = 0; t < num_tiers; t++) {
            fracs[t] *= rescale;
        }
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
        "mask_inventory_heads",
        "late_start_state_mode",
        "bis_gear_oracle_mode",
        "laser_obs_mode",
        "episode_max_ticks_override",
        "damage_scale_anneal_ticks",
    };
    for (size_t k = 0; k < sizeof(int_keys) / sizeof(*int_keys); k++) {
        ENCOUNTER_COLOSSEUM.put_int(
            COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env),
            int_keys[k], (int)dict_get(kwargs, int_keys[k]));
    }

    col_assign_curriculum_wave(env, kwargs);
    ENCOUNTER_COLOSSEUM.finalize_context(
        COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env));
}

void puf_reset(Env* env) {
    Agent* agent = &env->agents[0];
    ENCOUNTER_COLOSSEUM.reset(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), 0);
    ENCOUNTER_COLOSSEUM.write_obs(
        COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), (float*)agent->observations);
    col_write_action_mask_bytes(env, agent->action_mask);
}

void puf_step(Env* env) {
#ifdef COLO_PROFILE_ENABLED
    int col_prof_enabled = COLO_PROFILE_ENABLED();
    double col_prof_step_t0 = col_prof_enabled ? COLO_PROFILE_NOW_MS() : 0.0;
    double col_prof_t0 = col_prof_step_t0;
#endif
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

#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_C_ACTIONS);
#endif

    ENCOUNTER_COLOSSEUM.step(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), env->acts_staging);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_C_ENCOUNTER_STEP);
#endif

    float* obs = (float*)agent->observations;
    ENCOUNTER_COLOSSEUM.write_obs(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), obs);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_C_WRITE_OBS);
#endif
    col_write_action_mask_bytes(env, agent->action_mask);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_C_WRITE_MASK);
#endif

    agent->rewards[0] = ENCOUNTER_COLOSSEUM.get_reward(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env));
    int is_terminal = ENCOUNTER_COLOSSEUM.is_terminal(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env));
    agent->terminals[0] = (float)is_terminal;
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_C_REWARD_TERMINAL);
#endif

    /* Both readings run the best-gear oracle, which is the single most expensive thing
     * left in the env step. They are means over millions of steps, so a 1-in-N sample is
     * just as precise and costs N times less. */
    if (env->state.start_wave == env->config_start_wave &&
            (env->dpt_sample_step++ % COLO_DPT_SAMPLE_INTERVAL) == 0) {
        col_log_dpt_sample(
            &env->log.current_set_argmax_dpt_hit,
            &env->log.current_set_argmax_dpt_n,
            col_current_set_is_argmax_dpt_for_target(&env->state));
        col_log_dpt_sample(
            &env->log.attacked_argmax_set_hit,
            &env->log.attacked_argmax_set_n,
            col_attacked_with_argmax_set(&env->state));
#ifdef COLO_PROFILE_ENABLED
        COLO_PROFILE_MARK(COLO_PROF_C_LOG_DPT);
#endif
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
            env->log.deaths += (float)clog->died;
            env->log.timeouts += (float)clog->timed_out;
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
                env->log.npc_attack_death_by_type[t] +=
                    clog->npc_attack_death_by_type[t];
                env->log.typeless_damage_by_type[t] += clog->typeless_damage_by_type[t];
            }
            for (int source = 0; source < COLO_NUM_SOL_DAMAGE_SOURCES; source++)
                env->log.sol_damage_by_source[source] +=
                    clog->sol_damage_by_source[source];
            for (int source = 0;
                    source < COLO_NUM_JAVELIN_DAMAGE_SOURCES;
                    source++)
                env->log.javelin_damage_by_source[source] +=
                    clog->javelin_damage_by_source[source];
            for (int source = 0; source < COLO_NUM_DAMAGE_SOURCES; source++) {
                env->log.death_by_source[source] +=
                    clog->death_by_source[source];
                env->log.doom_death_by_source[source] +=
                    clog->doom_death_by_source[source];
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
            env->log.reward_steps += clog->reward_steps;
            env->log.reward_clamped_steps += clog->reward_clamped_steps;
            env->log.reward_clamp_loss += clog->reward_clamp_loss;
            env->log.clamp_loss_wave_clear += clog->clamp_loss_wave_clear;
            env->log.clamp_loss_win += clog->clamp_loss_win;
            env->log.reward_raw_peak += clog->reward_raw_peak;
            env->log.rew_damage += clog->rew_damage;
            env->log.rew_boss_phase += clog->rew_boss_phase;
            env->log.rew_wave_clear += clog->rew_wave_clear;
            env->log.rew_win += clog->rew_win;
            env->log.rew_death += clog->rew_death;
            env->log.rew_timeout += clog->rew_timeout;
            env->log.laser_volleys += clog->laser_volleys;
            env->log.laser_hits += clog->laser_hits;
            env->log.laser_dmg += clog->laser_dmg;
            env->log.laser_aligned_at_fire += clog->laser_aligned_at_fire;
            env->log.laser_aligned_at_show += clog->laser_aligned_at_show;
            env->log.laser_aligned_at_pre += clog->laser_aligned_at_pre;
            env->log.laser_aligned_at_damage += clog->laser_aligned_at_damage;
            env->log.laser_react_ok += clog->laser_react_ok;
            env->log.laser_react_fail += clog->laser_react_fail;
            env->log.avoid_total += clog->avoid_total;
            env->log.avoid_achieved += clog->avoid_achieved;
            env->log.avoid_missed += clog->avoid_missed;
            env->log.avoid_impossible += clog->avoid_impossible;
            env->log.dmg_unprayable += clog->dmg_unprayable;
            env->log.inv_memo_hits += clog->inv_memo_hits;
            env->log.inv_memo_misses += clog->inv_memo_misses;
            env->log.npc_blocked_calls += clog->npc_blocked_calls;
            env->log.npc_blocked_tiles += clog->npc_blocked_tiles;
            env->log.npc_stamp_tiles += clog->npc_stamp_tiles;
        }
#ifdef COLO_PROFILE_ENABLED
        COLO_PROFILE_MARK(COLO_PROF_C_TERMINAL_LOG);
#endif
        ENCOUNTER_COLOSSEUM.reset(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), 0);
        ENCOUNTER_COLOSSEUM.write_obs(COLO_ENV_STATE(env), COLO_ENV_CONTEXT(env), obs);
        col_write_action_mask_bytes(env, agent->action_mask);
#ifdef COLO_PROFILE_ENABLED
        COLO_PROFILE_MARK(COLO_PROF_C_RESET);
#endif
    }
#ifdef COLO_PROFILE_ENABLED
    if (col_prof_enabled)
        COLO_PROFILE_ADD(COLO_PROF_C_STEP_TOTAL, COLO_PROFILE_NOW_MS() - col_prof_step_t0);
        COLO_PROFILE_ADD(COLO_PROF_ENV_STEPS, 1.0);
#endif
}

#ifdef COLO_PROFILE_ENABLED
/* best_gear_* accumulate event counts, not milliseconds, so a %-of-step column on them
 * is meaningless. */
static int col_profile_slot_is_counter(int slot) {
    return slot == COLO_PROF_ENV_STEPS ||
        slot == COLO_PROF_BEST_GEAR_REQUESTS ||
        slot == COLO_PROF_BEST_GEAR_HITS ||
        slot == COLO_PROF_BEST_GEAR_BUILDS;
}

#define PUF_ENV_PROFILE_REPORT puf_env_profile_report
void puf_env_profile_report(void) {
    int n = colosseum_env_profile_count();
    if (n <= 0) return;

    double v[COLO_PROF_COUNT];
    for (int i = 0; i < n; i++) v[i] = colosseum_env_profile_read_reset_ms(i);
    double total = v[COLO_PROF_C_STEP_TOTAL];
    if (total <= 0.0) return;

    /* Absolute ns per env step is the number worth judging: a share only says how a slot
     * compares to its siblings, never whether the work itself is justified. */
    double steps = v[COLO_PROF_ENV_STEPS];
    if (steps <= 0.0) return;
    printf("\nenv profile: %.0f env-steps, %.0f ns/env-step total\n",
        steps, total * 1e6 / steps);
    printf("  %-24s %12s %8s\n", "", "ns/env-step", "share");
    for (int i = 0; i < n; i++) {
        if (v[i] <= 0.0 || col_profile_slot_is_counter(i)) continue;
        printf("  %-24s %12.1f %7.1f%%\n",
            colosseum_env_profile_name(i), v[i] * 1e6 / steps, 100.0 * v[i] / total);
    }
    for (int i = 0; i < n; i++) {
        if (v[i] <= 0.0 || !col_profile_slot_is_counter(i)) continue;
        printf("  %-24s %10.0f  (count)\n", colosseum_env_profile_name(i), v[i]);
    }
    fflush(stdout);
}
#endif

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
    dict_set(out, "deaths", log->deaths);
    dict_set(out, "timeouts", log->timeouts);
    dict_set(out, "wave", log->wave);
    dict_set(out, "npc_kills", log->npc_kills);

    float prayer_rate = log->prayer_total > 0.0f
        ? log->prayer_correct / log->prayer_total : 0.0f;
    dict_set(out, "prayer_correct_rate", prayer_rate);

    dict_set(out, "score", log->score);
    dict_set(out, "sol_min_hp", log->sol_min_hp);
    dict_set(out, "max_depth_reached", log->max_depth_reached);
    dict_set(out, "avoid_total", log->avoid_total);
    dict_set(out, "avoid_achieved", log->avoid_achieved);
    dict_set(out, "avoid_missed", log->avoid_missed);
    dict_set(out, "avoid_impossible", log->avoid_impossible);
    float inv_lookups = log->inv_memo_hits + log->inv_memo_misses;
    dict_set(out, "inv_memo_hit_rate", inv_lookups > 0.0f
        ? log->inv_memo_hits / inv_lookups : 0.0f);
    dict_set(out, "inv_memo_misses", log->inv_memo_misses);
    dict_set(out, "npc_blocked_tiles", log->npc_blocked_tiles);
    dict_set(out, "npc_stamp_tiles", log->npc_stamp_tiles);

    /* Ahead of the per-type breakdowns on purpose: the dashboard renders only the first
     * PUF_DASH_MAX_USER_ROWS*2 keys, and how much reward the [-1,1] clamp discards is a
     * health check on the whole reward design, not a per-NPC detail. */
    dict_set(out, "reward_clamp_frac", log->reward_steps > 0.0f
        ? log->reward_clamped_steps / log->reward_steps : 0.0f);
    dict_set(out, "reward_clamp_loss", log->reward_clamp_loss);
    dict_set(out, "reward_raw_peak", log->reward_raw_peak);
    dict_set(out, "laser_volleys", log->laser_volleys);
    dict_set(out, "laser_hits", log->laser_hits);
    dict_set(out, "laser_dmg", log->laser_dmg);
    dict_set(out, "laser_aligned_at_fire", log->laser_aligned_at_fire);
    dict_set(out, "laser_aligned_at_show", log->laser_aligned_at_show);
    dict_set(out, "laser_aligned_at_pre", log->laser_aligned_at_pre);
    dict_set(out, "laser_aligned_at_damage", log->laser_aligned_at_damage);
    dict_set(out, "laser_react_ok", log->laser_react_ok);
    dict_set(out, "laser_react_fail", log->laser_react_fail);
    dict_set(out, "clamp_loss_wave_clear", log->clamp_loss_wave_clear);
    dict_set(out, "clamp_loss_win", log->clamp_loss_win);
    dict_set(out, "rew_damage", log->rew_damage);
    dict_set(out, "rew_boss_phase", log->rew_boss_phase);
    dict_set(out, "rew_wave_clear", log->rew_wave_clear);
    dict_set(out, "rew_win", log->rew_win);
    dict_set(out, "rew_death", log->rew_death);
    dict_set(out, "rew_timeout", log->rew_timeout);
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
    static const char* const NPC_ATTACK_DEATH_BY_KEYS[COLO_NUM_NPC_TYPES] = {
        "npc_attack_death_by_berserker",
        "npc_attack_death_by_archer",
        "npc_attack_death_by_seer",
        "npc_attack_death_by_serpent",
        "npc_attack_death_by_jaguar",
        "npc_attack_death_by_javelin",
        "npc_attack_death_by_shockwave",
        "npc_attack_death_by_minotaur",
        "npc_attack_death_by_manticore",
        "npc_attack_death_by_sol",
        "npc_attack_death_by_totem",
        "npc_attack_death_by_bee",
    };
    for (int t = 0; t < COLO_NUM_NPC_TYPES; t++) {
        dict_set(
            out,
            NPC_ATTACK_DEATH_BY_KEYS[t],
            log->npc_attack_death_by_type[t]);
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
    static const char* const SOL_DAMAGE_SOURCE_KEYS[COLO_NUM_SOL_DAMAGE_SOURCES] = {
        "sol_dmg_spear_1",
        "sol_dmg_spear_2",
        "sol_dmg_shield_1",
        "sol_dmg_shield_2",
        "sol_dmg_triple_parry",
        "sol_dmg_grapple",
        "sol_dmg_crystal_laser",
        "sol_dmg_molten_sand",
    };
    for (int source = 0; source < COLO_NUM_SOL_DAMAGE_SOURCES; source++)
        dict_set(
            out,
            SOL_DAMAGE_SOURCE_KEYS[source],
            log->sol_damage_by_source[source]);
    static const char* const JAVELIN_DAMAGE_SOURCE_KEYS[
        COLO_NUM_JAVELIN_DAMAGE_SOURCES
    ] = {
        "javelin_dmg_basic_ranged",
        "javelin_dmg_skyfall",
        "javelin_dmg_reentry_pool",
        "javelin_dmg_reentry_volatility_pool",
    };
    for (int source = 0;
            source < COLO_NUM_JAVELIN_DAMAGE_SOURCES;
            source++)
        dict_set(
            out,
            JAVELIN_DAMAGE_SOURCE_KEYS[source],
            log->javelin_damage_by_source[source]);
    static const char* const DEATH_SOURCE_KEYS[COLO_NUM_DAMAGE_SOURCES] = {
        "death_source_npc_attack",
        "death_source_javelin_basic_ranged",
        "death_source_manticore_venom",
        "death_source_bee_poison",
        "death_source_bee_contact",
        "death_source_javelin_skyfall",
        "death_source_reentry_pool",
        "death_source_volatility_explosion",
        "death_source_volatility_pool",
        "death_source_reentry_volatility_pool",
        "death_source_solarflare",
        "death_source_self",
        "death_source_sol_spear_1",
        "death_source_sol_spear_2",
        "death_source_sol_shield_1",
        "death_source_sol_shield_2",
        "death_source_sol_triple_parry",
        "death_source_sol_grapple",
        "death_source_sol_crystal_laser",
        "death_source_sol_molten_sand",
    };
    static const char* const DOOM_DEATH_SOURCE_KEYS[COLO_NUM_DAMAGE_SOURCES] = {
        "doom_death_source_npc_attack",
        "doom_death_source_javelin_basic_ranged",
        "doom_death_source_manticore_venom",
        "doom_death_source_bee_poison",
        "doom_death_source_bee_contact",
        "doom_death_source_javelin_skyfall",
        "doom_death_source_reentry_pool",
        "doom_death_source_volatility_explosion",
        "doom_death_source_volatility_pool",
        "doom_death_source_reentry_volatility_pool",
        "doom_death_source_solarflare",
        "doom_death_source_self",
        "doom_death_source_sol_spear_1",
        "doom_death_source_sol_spear_2",
        "doom_death_source_sol_shield_1",
        "doom_death_source_sol_shield_2",
        "doom_death_source_sol_triple_parry",
        "doom_death_source_sol_grapple",
        "doom_death_source_sol_crystal_laser",
        "doom_death_source_sol_molten_sand",
    };
    for (int source = 0; source < COLO_NUM_DAMAGE_SOURCES; source++) {
        dict_set(out, DEATH_SOURCE_KEYS[source], log->death_by_source[source]);
        dict_set(
            out,
            DOOM_DEATH_SOURCE_KEYS[source],
            log->doom_death_by_source[source]);
    }
    dict_set(out, "death_dmg_unprayable", log->death_dmg_unprayable);
    dict_set(out, "death_dmg_offpray", log->death_dmg_offpray);
    dict_set(out, "death_dmg_prayed", log->death_dmg_prayed);
    dict_set(out, "death_dmg_self", log->death_dmg_self);
    dict_set(out, "death_heal_remaining", log->death_heal_remaining);
    dict_set(out, "farm_damage", log->farm_damage);
}
