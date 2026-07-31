#ifndef ENCOUNTER_COLOSSEUM_H
#define ENCOUNTER_COLOSSEUM_H

#include "../osrs_types.h"
#include "../osrs_items.h"
#include "../osrs_monsters_generated.h"
#include "../osrs_collision.h"
#include "../osrs_combat.h"
#include "../osrs_combat_visuals.h"
#include "../osrs_special_attacks.h"
#include "../osrs_pvp_gear.h"
#include "../osrs_encounter.h"
#include "../osrs_encounter_player.h"
#include "../osrs_encounter_visual_events.h"
#include "../osrs_player_consumables.h"
#include "../osrs_inventory_clicks.h"
#include "../osrs_interaction.h"
#include "../data/npc_models.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <ctype.h>
#include <errno.h>

typedef enum {
    COLO_FREMENNIK_BERSERKER = 0,
    COLO_FREMENNIK_ARCHER,
    COLO_FREMENNIK_SEER,
    COLO_SERPENT_SHAMAN,
    COLO_JAGUAR_WARRIOR,
    COLO_JAVELIN_COLOSSUS,
    COLO_SHOCKWAVE_COLOSSUS,
    COLO_MINOTAUR,
    COLO_MANTICORE,
    COLO_SOL_HEREDIT,
    COLO_HEALING_TOTEM,
    COLO_BEE_SWARM,
    COLO_NUM_NPC_TYPES
} ColoNpcType;

typedef enum {
    COLO_OUTCOME_PLAYER_WON = 0,
    COLO_OUTCOME_PLAYER_DIED = 1,
} ColoOutcome;

typedef enum {
    COLO_MOD_BEES = 0,
    COLO_MOD_BLASPHEMY,
    COLO_MOD_DOOM,
    COLO_MOD_DYNAMIC_DUO,
    COLO_MOD_FRAILTY,
    COLO_MOD_MANTIMAYHEM,
    COLO_MOD_MYOPIA,
    COLO_MOD_REENTRY,
    COLO_MOD_RED_FLAG,
    COLO_MOD_RELENTLESS,
    COLO_MOD_SOLARFLARE,
    COLO_MOD_QUARTET,
    COLO_MOD_TOTEMIC,
    COLO_MOD_VOLATILITY,
    COLO_NUM_REAL_MODIFIERS
} ColoModifier;

#include "colosseum/encounter_colosseum_model.inc"
#include "colosseum/encounter_colosseum_helpers.inc"
#include "colosseum/encounter_colosseum_reset_spawn.inc"
#include "colosseum/encounter_colosseum_movement.inc"
#include "colosseum/encounter_colosseum_modifiers.inc"
#include "colosseum/encounter_colosseum_combat.inc"
#include "colosseum/encounter_colosseum_boss.inc"
#include "colosseum/encounter_colosseum_player_actions.inc"
#include "colosseum/encounter_colosseum_reward_step.inc"
#include "colosseum/encounter_colosseum_forecast.inc"
#include "colosseum/encounter_colosseum_obs_mask.inc"
#include "colosseum/encounter_colosseum_mask_render.inc"
#include "colosseum/encounter_colosseum_lab.inc"
#include "colosseum/encounter_colosseum_lab_parse.inc"
#include "colosseum/encounter_colosseum_lab_json.inc"
#include "colosseum/encounter_colosseum_render_snapshot.inc"

#endif
