#include "robocode.h"
#define OBS_SIZE (EGO_FEATURES + OTHER_FEATURES)
#define NUM_ATNS 5
#define ACT_SIZES {4, 9, 11, 11, 6}
#define OBS_TENSOR_T FloatTensor

#define MY_USES_PERM
#define MY_USES_TAGS
#define Env Robocode
#include "vecenv.h"

// Selfplay-pool routing: write per-slot pointers into the global vec buffers,
// respecting agent_perm if set (selfplay re-routes logical slots into specific
// physical rows so banks own contiguous ranges). Identity perm = adjacent
// slot_base + s layout — matches single-agent / bot-mode runs.
void my_setup_perm(StaticVec* vec, Env* env, int slot_base) {
    for (int s = 0; s < env->num_agents; s++) {
        int phys = vec->agent_perm ? vec->agent_perm[slot_base + s] : (slot_base + s);
        env->obs_ptr[s]      = (float*)vec->observations + (size_t)phys * OBS_SIZE;
        env->action_ptr[s]   = vec->actions + (size_t)phys * NUM_ATNS;
        env->reward_ptr[s]   = vec->rewards + phys;
        env->terminal_ptr[s] = vec->terminals + phys;
    }
}


static inline float dict_get_float_default(Dict* kwargs, const char* key, float default_value) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item ? (float)item->value : default_value;
}

void my_init(Env* env, Dict* kwargs) {
    env->width = dict_get(kwargs, "width")->value;
    env->height = dict_get(kwargs, "height")->value;
    env->num_agents = dict_get(kwargs, "num_agents")->value;
    env->num_bots = dict_get(kwargs, "num_bots")->value;
    env->max_ticks = (int)dict_get(kwargs, "max_ticks")->value;
    env->reward_damage = dict_get_float_default(kwargs, "reward_damage", 0.0f);
    env->reward_spot = dict_get_float_default(kwargs, "reward_spot", 0.0f);
    env->reward_melee_damage_inflicted = dict_get_float_default(kwargs, "reward_melee_damage_inflicted", 0.0f);
    env->reward_damage_taken = dict_get_float_default(kwargs, "reward_damage_taken", 0.0f);
    env->reward_range_damage_inflicted = dict_get_float_default(kwargs, "reward_range_damage_inflicted", 0.0f);
    env->reward_melee_damage_inflicted_slot_0 = dict_get_float_default(kwargs,
        "reward_melee_damage_inflicted_slot_0", env->reward_melee_damage_inflicted);
    env->reward_damage_taken_slot_0 = dict_get_float_default(kwargs,
        "reward_damage_taken_slot_0", env->reward_damage_taken);
    env->reward_range_damage_inflicted_slot_0 = dict_get_float_default(kwargs,
        "reward_range_damage_inflicted_slot_0", env->reward_range_damage_inflicted);
    env->reward_melee_damage_inflicted_slot_1 = dict_get_float_default(kwargs,
        "reward_melee_damage_inflicted_slot_1", env->reward_melee_damage_inflicted);
    env->reward_damage_taken_slot_1 = dict_get_float_default(kwargs,
        "reward_damage_taken_slot_1", env->reward_damage_taken);
    env->reward_range_damage_inflicted_slot_1 = dict_get_float_default(kwargs,
        "reward_range_damage_inflicted_slot_1", env->reward_range_damage_inflicted);
    DictItem* dr_item = dict_get_unsafe(kwargs, "dr");
    env->dr = dr_item ? (float)dr_item->value : 0.0f;
    env->bot_policy = dict_get(kwargs, "bot_policy")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "damage_received", log->damage_received);
    dict_set(out, "melee_damage_inflicted", log->melee_damage_inflicted);
    dict_set(out, "damage_taken", log->damage_taken);
    dict_set(out, "range_damage_inflicted", log->range_damage_inflicted);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    // Historical-pool stats. selfplay.py reads hist_score_bank_<b> /
    // hist_n_bank_<b> per bank to drive swap decisions. Legacy aggregate
    // hist_score / hist_n sum across all banks for backward-compat dashboards.
    dict_set(out, "hist_score", log->hist_score);
    dict_set(out, "hist_n", log->hist_n);
    dict_set(out, "hist_score_bank_0", log->hist_score_bank[0]);
    dict_set(out, "hist_score_bank_1", log->hist_score_bank[1]);
    dict_set(out, "hist_n_bank_0", log->hist_n_bank[0]);
    dict_set(out, "hist_n_bank_1", log->hist_n_bank[1]);
    // Per-slot scores — match() reads slot_0_score / slot_1_score as A/B win rates.
    dict_set(out, "slot_0_score", log->slot_0_score);
    dict_set(out, "slot_1_score", log->slot_1_score);
    dict_set(out, "draw_rate", log->draw_rate);
    dict_set(out, "n", log->n);
}
