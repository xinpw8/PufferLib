#ifndef PUFFER_INFERNO_PROFILE_H
#define PUFFER_INFERNO_PROFILE_H

#ifndef INFERNO_ENV_EXPORT
#define INFERNO_ENV_EXPORT __attribute__((visibility("default")))
#endif

#define OSRS_ENV_PROFILE_SLOTS(X) \
    X(INF_PROF_C_STEP_TOTAL,             "c_step_total") \
    X(INF_PROF_C_ACTIONS,                "c_actions") \
    X(INF_PROF_C_ENCOUNTER_STEP,         "c_encounter_step") \
    X(INF_PROF_C_WRITE_OBS,              "c_write_obs") \
    X(INF_PROF_C_WRITE_MASK,             "c_write_mask") \
    X(INF_PROF_C_REWARD_TERMINAL,        "c_reward_terminal") \
    X(INF_PROF_C_TERMINAL_LOG,           "c_terminal_log") \
    X(INF_PROF_C_RESET,                  "c_reset") \
    X(INF_PROF_OBS_PREFIX,               "obs_prefix") \
    X(INF_PROF_OBS_REFRESH_SLOTS,        "obs_refresh_slots") \
    X(INF_PROF_OBS_NPC_SLOTS,            "obs_npc_slots") \
    X(INF_PROF_OBS_FORECAST,             "obs_forecast") \
    X(INF_PROF_OBS_PENDING_HITS,         "obs_pending_hits") \
    X(INF_PROF_OBS_SPARKS,               "obs_sparks") \
    X(INF_PROF_FORECAST_LANDING,         "forecast_landing") \
    X(INF_PROF_FORECAST_STATE_COPY,      "forecast_state_copy") \
    X(INF_PROF_FORECAST_NPC_MOVE,        "forecast_npc_move") \
    X(INF_PROF_FORECAST_NPC_ATTACK,      "forecast_npc_attack") \
    X(INF_PROF_FORECAST_CALLS,           "forecast_calls") \
    X(INF_PROF_FORECAST_VALID_ACTIONS,   "forecast_valid_actions") \
    X(INF_PROF_FORECAST_DISTINCT_LANDINGS, "forecast_distinct_landings")

#define OSRS_ENV_PROFILE_PREFIX     inferno
#define OSRS_ENV_PROFILE_COUNT      INF_PROF_COUNT
#define OSRS_ENV_PROFILE_SLOT_TYPE  InfernoProfileSlot
#define OSRS_ENV_PROFILE_ENV_VAR    "PUFFER_INFERNO_PROFILE"
#define OSRS_ENV_PROFILE_EXPORT     INFERNO_ENV_EXPORT
#include "../osrs/osrs_env_profile.h"

#define INF_PROFILE_ENABLED() inferno_profile_enabled()
#define INF_PROFILE_NOW_MS() inferno_profile_now_ms()
#define INF_PROFILE_ADD(slot, ms) inferno_profile_add((slot), (ms))
#define INF_PROFILE_MARK(slot) inferno_profile_mark(inf_prof_enabled, &inf_prof_t0, (slot))

#endif
