#ifndef ENCOUNTER_INFERNO_H
#define ENCOUNTER_INFERNO_H

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
#include "../osrs_inventory_actions.h"
#include "../osrs_policy.h"
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

#include "inferno/encounter_inferno_model.inc"
#include "inferno/encounter_inferno_helpers.inc"
#include "inferno/encounter_inferno_reset_spawn.inc"
#include "inferno/encounter_inferno_movement.inc"
#include "inferno/encounter_inferno_combat.inc"
#include "inferno/encounter_inferno_player_actions.inc"
#include "inferno/encounter_inferno_reward_step.inc"
#include "inferno/encounter_inferno_forecast.inc"
#include "inferno/encounter_inferno_lab.inc"
#include "inferno/encounter_inferno_obs_mask.inc"
#include "inferno/encounter_inferno_render_snapshot.inc"

#endif
