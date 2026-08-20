#ifndef PUFFER_COLOSSEUM_PROFILE_H
#define PUFFER_COLOSSEUM_PROFILE_H

#ifndef COLOSSEUM_ENV_EXPORT
#define COLOSSEUM_ENV_EXPORT __attribute__((visibility("default")))
#endif

#define OSRS_ENV_PROFILE_SLOTS(X) \
    X(COLO_PROF_C_STEP_TOTAL,           "c_step_total") \
    X(COLO_PROF_C_ACTIONS,              "c_actions") \
    X(COLO_PROF_C_ENCOUNTER_STEP,       "c_encounter_step") \
    X(COLO_PROF_C_WRITE_OBS,            "c_write_obs") \
    X(COLO_PROF_C_WRITE_MASK,           "c_write_mask") \
    X(COLO_PROF_C_REWARD_TERMINAL,      "c_reward_terminal") \
    X(COLO_PROF_C_TERMINAL_LOG,         "c_terminal_log") \
    X(COLO_PROF_C_RESET,                "c_reset") \
    X(COLO_PROF_C_LOG_DPT,              "c_log_dpt") \
    X(COLO_PROF_OBS_REFRESH_SLOTS,      "obs_refresh_slots") \
    X(COLO_PROF_OBS_PREFIX,             "obs_prefix") \
    X(COLO_PROF_OBS_INVENTORY,          "obs_inventory") \
    X(COLO_PROF_OBS_VENATOR,            "obs_venator") \
    X(COLO_PROF_OBS_NPC_SLOTS,          "obs_npc_slots") \
    X(COLO_PROF_OBS_MODIFIERS,          "obs_modifiers") \
    X(COLO_PROF_OBS_BOSS,               "obs_boss") \
    X(COLO_PROF_OBS_PENDING_HITS,       "obs_pending_hits") \
    X(COLO_PROF_OBS_THREAT_LOS,         "obs_threat_los") \
    X(COLO_PROF_OBS_THRALL_DC,          "obs_thrall_dc") \
    X(COLO_PROF_OBS_SPAWN,              "obs_spawn") \
    X(COLO_PROF_STEP_PRE_PLAYER,        "step_pre_player") \
    X(COLO_PROF_PRE_RESET_SCRATCH,      "pre_reset_scratch") \
    X(COLO_PROF_PRE_RESET_RENDER,       "pre_reset_render") \
    X(COLO_PROF_PRE_RESET_PLAYER,       "pre_reset_player") \
    X(COLO_PROF_PRE_RESET_NPCS,         "pre_reset_npcs") \
    X(COLO_PROF_PRE_PLAYER_PRETICK,     "pre_player_pretick") \
    X(COLO_PROF_PRE_SPAWN,              "pre_spawn") \
    X(COLO_PROF_PRE_RESOLVE_HITS,       "pre_resolve_hits") \
    X(COLO_PROF_PRE_COLLISION,          "pre_collision") \
    X(COLO_PROF_PRE_NPC_PHASE,          "pre_npc_phase") \
    X(COLO_PROF_PRE_OFFPRAY,            "pre_offpray") \
    X(COLO_PROF_STEP_PLAYER,            "step_player") \
    X(COLO_PROF_PLAYER_TIMERS,          "player_timers") \
    X(COLO_PROF_PLAYER_INVENTORY,       "player_inventory") \
    X(COLO_PROF_PLAYER_INTENT,          "player_intent") \
    X(COLO_PROF_PLAYER_MOVE,            "player_move") \
    X(COLO_PROF_PLAYER_ATTACK,          "player_attack") \
    X(COLO_PROF_STEP_REWARD,            "step_reward") \
    X(COLO_PROF_STEP_WAVE_LOGIC,        "step_wave_logic") \
    X(COLO_PROF_STEP_NPC_TOTAL,         "step_npc_total") \
    X(COLO_PROF_STEP_SOL_BOSS,          "step_sol_boss") \
    X(COLO_PROF_STEP_JAVELIN_SKYFALL,   "step_javelin_skyfall") \
    X(COLO_PROF_STEP_NPC_MOVEMENT,      "step_npc_movement") \
    X(COLO_PROF_STEP_NPC_PATHFINDING,   "step_npc_pathfinding") \
    X(COLO_PROF_STEP_NPC_ATTACK,        "step_npc_attack") \
    X(COLO_PROF_STEP_MANTICORE_BARRAGE, "step_manticore_barrage") \
    X(COLO_PROF_STEP_WARBAND_ATTACK,    "step_warband_attack") \
    X(COLO_PROF_STEP_MODIFIERS_HAZARDS, "step_modifiers_hazards") \
    X(COLO_PROF_ENV_STEPS,              "env_steps") \
    X(COLO_PROF_BEST_GEAR_REQUESTS,     "best_gear_requests") \
    X(COLO_PROF_BEST_GEAR_HITS,         "best_gear_hits") \
    X(COLO_PROF_BEST_GEAR_BUILDS,       "best_gear_builds") \
    X(COLO_PROF_VENATOR_REQUESTS,       "venator_requests") \
    X(COLO_PROF_VENATOR_HITS,           "venator_hits") \
    X(COLO_PROF_VENATOR_REFRESHES,       "venator_refreshes")

#define OSRS_ENV_PROFILE_PREFIX     colosseum
#define OSRS_ENV_PROFILE_COUNT      COLO_PROF_COUNT
#define OSRS_ENV_PROFILE_SLOT_TYPE  ColosseumProfileSlot
#define OSRS_ENV_PROFILE_ENV_VAR    "PUFFER_COLOSSEUM_PROFILE"
#define OSRS_ENV_PROFILE_EXPORT     COLOSSEUM_ENV_EXPORT
#include "../osrs/osrs_env_profile.h"

#define COLO_PROFILE_ENABLED() colosseum_profile_enabled()
#define COLO_PROFILE_NOW_MS() colosseum_profile_now_ms()
#define COLO_PROFILE_ADD(slot, ms) colosseum_profile_add((slot), (ms))
#define COLO_PROFILE_MARK(slot) colosseum_profile_mark(col_prof_enabled, &col_prof_t0, (slot))

#endif
