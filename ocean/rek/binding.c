#include "rek.h"
#include "render.h"

#define OBS_SIZE REK_OBS_SIZE
#define NUM_ATNS REK_NUM_ATNS
// Three discrete heads: locomotion (neutral + 8 directions), set move
// (neutral + one entry per bound key), guard on/off. NUM_MOVE_DEFS already
// counts the neutral entry, so the move head needs no +1.
#define ACT_SIZES {NUM_MOVE_DIRS, NUM_MOVE_DEFS, 2}
#define OBS_TENSOR_T FloatTensor

#define MY_USES_PERM
#define MY_USES_TAGS
#define Env Rek
#include "vecenv.h"

// Selfplay-pool routing: write per-slot pointers into the global vec buffers,
// honouring agent_perm when the pool has rerouted logical slots into specific
// physical rows (primary vs frozen bank). Identity perm gives the adjacent
// slot_base + s layout that single-agent and bot-mode runs expect.
void my_setup_perm(StaticVec* vec, Env* env, int slot_base) {
    for (int s = 0; s < env->num_agents; s++) {
        int phys = vec->agent_perm ? vec->agent_perm[slot_base + s] : (slot_base + s);
        env->obs_ptr[s]      = (float*)vec->observations + (size_t)phys * OBS_SIZE;
        env->action_ptr[s]   = vec->actions + (size_t)phys * NUM_ATNS;
        env->reward_ptr[s]   = vec->rewards + phys;
        env->terminal_ptr[s] = vec->terminals + phys;
    }
}

static inline float kwarg_float(Dict* kwargs, const char* key, float fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item ? (float)item->value : fallback;
}

static inline int kwarg_int(Dict* kwargs, const char* key, int fallback) {
    DictItem* item = dict_get_unsafe(kwargs, key);
    return item ? (int)item->value : fallback;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = kwarg_int(kwargs, "num_agents", 2);
    env->num_bots   = kwarg_int(kwargs, "num_bots", 0);
    env->bot_policy = kwarg_int(kwargs, "bot_policy", 0);

    // Round length in seconds on the config side; frames everywhere internally.
    float round_seconds = kwarg_float(kwargs, "round_seconds", 60.0f);
    env->round_frames = (int)(round_seconds * REK_TICK_HZ);
    if (env->round_frames < 1) env->round_frames = 1;

    env->arena_radius = kwarg_float(kwargs, "arena_radius", 3.0f);
    env->body_radius  = kwarg_float(kwargs, "body_radius", 0.28f);

    env->move_speed        = kwarg_float(kwargs, "move_speed", 1.4f);
    env->guard_speed_mult  = kwarg_float(kwargs, "guard_speed_mult", 0.5f);
    env->accel             = kwarg_float(kwargs, "accel", 0.35f);
    env->friction          = kwarg_float(kwargs, "friction", 0.82f);
    env->turn_rate         = kwarg_float(kwargs, "turn_rate", 0.18f);
    env->balance_decay     = kwarg_float(kwargs, "balance_decay", 0.02f);
    env->guard_balance_mult = kwarg_float(kwargs, "guard_balance_mult", 0.35f);
    env->hitstun_frames    = kwarg_int(kwargs, "hitstun_frames", 6);
    env->getup_frames      = kwarg_int(kwargs, "getup_frames", 45);

    env->reward_hit        = kwarg_float(kwargs, "reward_hit", 0.1f);
    env->reward_hit_taken  = kwarg_float(kwargs, "reward_hit_taken", -0.1f);
    env->reward_down       = kwarg_float(kwargs, "reward_down", -0.3f);
    env->reward_down_dealt = kwarg_float(kwargs, "reward_down_dealt", 0.3f);
    env->reward_win        = kwarg_float(kwargs, "reward_win", 1.0f);
    env->reward_guard      = kwarg_float(kwargs, "reward_guard", 0.0f);

    env->dr = kwarg_float(kwargs, "dr", 1.0f);

    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "hits_landed", log->hits_landed);
    dict_set(out, "hits_taken", log->hits_taken);
    dict_set(out, "downs", log->downs);
    dict_set(out, "knockouts", log->knockouts);
    dict_set(out, "guard_uptime", log->guard_uptime);
    dict_set(out, "whiff_rate", log->whiff_rate);
    dict_set(out, "hist_score", log->hist_score);
    dict_set(out, "hist_n", log->hist_n);
    dict_set(out, "hist_score_bank_0", log->hist_score_bank[0]);
    dict_set(out, "hist_n_bank_0", log->hist_n_bank[0]);
    dict_set(out, "hist_score_bank_1", log->hist_score_bank[1]);
    dict_set(out, "hist_n_bank_1", log->hist_n_bank[1]);
    dict_set(out, "slot_0_score", log->slot_0_score);
    dict_set(out, "slot_1_score", log->slot_1_score);
    dict_set(out, "draw_rate", log->draw_rate);
}
