#ifndef OSRS_POLICY_H
#define OSRS_POLICY_H

#include "osrs_inventory_actions.h"

#define OSRS_SHARED_SELF_OBS_SIZE 52
#define OSRS_SHARED_INVENTORY_CELL_OBS_FEATURES 1
#define OSRS_SHARED_INVENTORY_OBS_SIZE \
    (OSRS_INVENTORY_SIZE * OSRS_SHARED_INVENTORY_CELL_OBS_FEATURES)
#define OSRS_SHARED_EQUIPPED_OBS_SIZE NUM_GEAR_SLOTS
#define OSRS_SHARED_ITEM_OBS_SIZE \
    (OSRS_SHARED_INVENTORY_OBS_SIZE + OSRS_SHARED_EQUIPPED_OBS_SIZE)
#define OSRS_SHARED_EFFECT_OBS_SIZE OSRS_EQUIPMENT_EFFECT_AGGREGATE_FEATURES
#define OSRS_SHARED_OBS_SIZE \
    (OSRS_SHARED_SELF_OBS_SIZE + OSRS_SHARED_ITEM_OBS_SIZE + \
     OSRS_SHARED_EFFECT_OBS_SIZE)
#define OSRS_SHARED_OBS_MAX_HIT 40
#define OSRS_SHARED_OBS_INVENTORY_START OSRS_SHARED_SELF_OBS_SIZE
#define OSRS_SHARED_OBS_EQUIPPED_START \
    (OSRS_SHARED_OBS_INVENTORY_START + OSRS_SHARED_INVENTORY_OBS_SIZE)
#define OSRS_SHARED_OBS_EFFECT_START \
    (OSRS_SHARED_OBS_EQUIPPED_START + OSRS_SHARED_EQUIPPED_OBS_SIZE)

#define OSRS_HEAD_PRIMARY 0
#define OSRS_HEAD_OVERHEAD 1
#define OSRS_HEAD_EQUIP_BASE 2
#define OSRS_HEAD_EQUIP_SLOT(slot) (OSRS_HEAD_EQUIP_BASE + (slot))
#define OSRS_HEAD_EAT (OSRS_HEAD_EQUIP_BASE + NUM_GEAR_SLOTS)
#define OSRS_HEAD_DRINK (OSRS_HEAD_EAT + 1)
#define OSRS_HEAD_SPELL (OSRS_HEAD_DRINK + 1)
#define OSRS_HEAD_SPECIAL (OSRS_HEAD_SPELL + 1)
#define OSRS_HEAD_OFFENSIVE (OSRS_HEAD_SPECIAL + 1)
#define OSRS_BASE_NUM_ACTION_HEADS (OSRS_HEAD_OFFENSIVE + 1)

#define OSRS_PRIMARY_MOVE_ACTIONS 25
#define OSRS_PRIMARY_DIM(target_slots) (OSRS_PRIMARY_MOVE_ACTIONS + (target_slots))
#define OSRS_OVERHEAD_DIM 7
#define OSRS_INVENTORY_CLICK_DIM (OSRS_INVENTORY_SIZE + 1)
#define OSRS_SPELL_DIM 5
#define OSRS_SPECIAL_DIM 3
#define OSRS_OFFENSIVE_DIM 5
#define OSRS_BASE_ACTION_MASK_SIZE(target_slots) \
    (OSRS_PRIMARY_DIM(target_slots) + OSRS_OVERHEAD_DIM + \
     (NUM_GEAR_SLOTS + 2) * OSRS_INVENTORY_CLICK_DIM + \
     OSRS_SPELL_DIM + OSRS_SPECIAL_DIM + OSRS_OFFENSIVE_DIM)
#define OSRS_BASE_ACTION_DIMS(target_slots) \
    OSRS_PRIMARY_DIM(target_slots), OSRS_OVERHEAD_DIM, \
    OSRS_INVENTORY_CLICK_DIM, OSRS_INVENTORY_CLICK_DIM, \
    OSRS_INVENTORY_CLICK_DIM, OSRS_INVENTORY_CLICK_DIM, \
    OSRS_INVENTORY_CLICK_DIM, OSRS_INVENTORY_CLICK_DIM, \
    OSRS_INVENTORY_CLICK_DIM, OSRS_INVENTORY_CLICK_DIM, \
    OSRS_INVENTORY_CLICK_DIM, OSRS_INVENTORY_CLICK_DIM, \
    OSRS_INVENTORY_CLICK_DIM, OSRS_INVENTORY_CLICK_DIM, \
    OSRS_INVENTORY_CLICK_DIM, OSRS_SPELL_DIM, \
    OSRS_SPECIAL_DIM, OSRS_OFFENSIVE_DIM
#define OSRS_BASE_ACTION_DIMS_INIT(target_slots) { \
    OSRS_BASE_ACTION_DIMS(target_slots) \
}

typedef enum {
    OSRS_SPELL_NONE = 0,
    OSRS_SPELL_BLOOD_BARRAGE,
    OSRS_SPELL_ICE_BARRAGE,
    OSRS_SPELL_VENGEANCE,
    OSRS_SPELL_DEATH_CHARGE,
} OsrsSpellAction;

typedef struct {
    const Player* player;
    const OsrsInteraction* interaction;
    int arena_min_x;
    int arena_max_x;
    int arena_min_y;
    int arena_max_y;
    AttackStyle attack_style;
    int attack_range;
    int max_hit;
    int attack_speed;
    int defence_stab;
    int defence_slash;
    int defence_crush;
    int defence_magic;
    int defence_ranged;
    int effective_level;
    int attack_bonus;
    int strength_bonus;
    int spell_base_damage;
    int special_attack_cost;
} OsrsSharedObservationInput;

static inline float osrs_policy_ratio(int value, int scale) {
    return scale > 0 ? (float)value / (float)scale : 0.0f;
}

static inline int osrs_base_action_head_mask_offset(int target_slots, int head) {
    if (target_slots < 0 || head < 0 || head >= OSRS_BASE_NUM_ACTION_HEADS) {
        fprintf(stderr, "osrs policy action offset: invalid targets=%d head=%d\n",
            target_slots, head);
        abort();
    }
    int offset = 0;
    for (int current = 0; current < head; current++) {
        if (current == OSRS_HEAD_PRIMARY) {
            offset += OSRS_PRIMARY_DIM(target_slots);
        } else if (current == OSRS_HEAD_OVERHEAD) {
            offset += OSRS_OVERHEAD_DIM;
        } else if (current >= OSRS_HEAD_EQUIP_BASE && current <= OSRS_HEAD_DRINK) {
            offset += OSRS_INVENTORY_CLICK_DIM;
        } else if (current == OSRS_HEAD_SPELL) {
            offset += OSRS_SPELL_DIM;
        } else if (current == OSRS_HEAD_SPECIAL) {
            offset += OSRS_SPECIAL_DIM;
        } else {
            offset += OSRS_OFFENSIVE_DIM;
        }
    }
    return offset;
}

static inline int osrs_write_shared_observations(
    float* obs,
    const OsrsSharedObservationInput* input
) {
    if (!obs || !input || !input->player || !input->interaction ||
            input->arena_max_x <= input->arena_min_x ||
            input->arena_max_y <= input->arena_min_y) {
        fprintf(stderr, "osrs shared observation: invalid input\n");
        abort();
    }

    const Player* player = input->player;
    int width = input->arena_max_x - input->arena_min_x;
    int height = input->arena_max_y - input->arena_min_y;
    int i = 0;

    obs[i++] = osrs_policy_ratio(player->current_hitpoints, player->base_hitpoints);
    obs[i++] = osrs_policy_ratio(player->current_prayer, player->base_prayer);
    obs[i++] = osrs_policy_ratio(player->x - input->arena_min_x, width);
    obs[i++] = osrs_policy_ratio(input->arena_max_x - player->x, width);
    obs[i++] = osrs_policy_ratio(player->y - input->arena_min_y, height);
    obs[i++] = osrs_policy_ratio(input->arena_max_y - player->y, height);
    obs[i++] = player->prayer == PRAYER_PROTECT_MELEE ? 1.0f : 0.0f;
    obs[i++] = player->prayer == PRAYER_PROTECT_RANGED ? 1.0f : 0.0f;
    obs[i++] = player->prayer == PRAYER_PROTECT_MAGIC ? 1.0f : 0.0f;
    obs[i++] = player->prayer == PRAYER_SMITE ? 1.0f : 0.0f;
    obs[i++] = player->prayer == PRAYER_REDEMPTION ? 1.0f : 0.0f;
    obs[i++] = player->offensive_prayer == OFFENSIVE_PRAYER_PIETY ? 1.0f : 0.0f;
    obs[i++] = player->offensive_prayer == OFFENSIVE_PRAYER_RIGOUR ? 1.0f : 0.0f;
    obs[i++] = player->offensive_prayer == OFFENSIVE_PRAYER_AUGURY ? 1.0f : 0.0f;
    obs[i++] = osrs_policy_ratio(player->run_energy, 10000);
    obs[i++] = osrs_policy_ratio(player->special_energy, 100);
    obs[i++] = player->spec_armed ? 1.0f : 0.0f;
    obs[i++] = osrs_policy_ratio(player->attack_timer, 8);
    obs[i++] = osrs_policy_ratio(player->food_timer, 3);
    obs[i++] = osrs_policy_ratio(player->potion_timer, 3);
    obs[i++] = osrs_policy_ratio(player->karambwan_timer, 2);
    obs[i++] = osrs_policy_ratio(player->frozen_ticks, 32);
    obs[i++] = osrs_policy_ratio(player->freeze_immunity_ticks, 5);
    obs[i++] = osrs_interaction_active(input->interaction) ? 1.0f : 0.0f;
    obs[i++] = osrs_policy_ratio(player->current_attack, 150);
    obs[i++] = osrs_policy_ratio(player->current_strength, 150);
    obs[i++] = osrs_policy_ratio(player->current_defence, 150);
    obs[i++] = osrs_policy_ratio(player->current_ranged, 150);
    obs[i++] = osrs_policy_ratio(player->current_magic, 150);
    obs[i++] = osrs_policy_ratio(player->base_attack, 99);
    obs[i++] = osrs_policy_ratio(player->base_strength, 99);
    obs[i++] = osrs_policy_ratio(player->base_defence, 99);
    obs[i++] = osrs_policy_ratio(player->base_ranged, 99);
    obs[i++] = osrs_policy_ratio(player->base_magic, 99);
    obs[i++] = osrs_policy_ratio(player->base_prayer, 99);
    obs[i++] = osrs_policy_ratio(player->base_hitpoints, 99);
    obs[i++] = input->attack_style == ATTACK_STYLE_MELEE ? 1.0f : 0.0f;
    obs[i++] = input->attack_style == ATTACK_STYLE_RANGED ? 1.0f : 0.0f;
    obs[i++] = input->attack_style == ATTACK_STYLE_MAGIC ? 1.0f : 0.0f;
    obs[i++] = osrs_policy_ratio(input->attack_range, 15);
    obs[i++] = osrs_policy_ratio(input->max_hit, 80);
    obs[i++] = osrs_policy_ratio(input->attack_speed, 8);
    obs[i++] = osrs_policy_ratio(input->defence_stab, 300);
    obs[i++] = osrs_policy_ratio(input->defence_slash, 300);
    obs[i++] = osrs_policy_ratio(input->defence_crush, 300);
    obs[i++] = osrs_policy_ratio(input->defence_magic, 300);
    obs[i++] = osrs_policy_ratio(input->defence_ranged, 300);
    obs[i++] = osrs_policy_ratio(input->effective_level, 165);
    obs[i++] = osrs_policy_ratio(input->attack_bonus, 200);
    obs[i++] = osrs_policy_ratio(input->strength_bonus, 160);
    obs[i++] = osrs_policy_ratio(input->spell_base_damage, 40);
    obs[i++] = osrs_policy_ratio(input->special_attack_cost, 100);

    if (i != OSRS_SHARED_SELF_OBS_SIZE) {
        fprintf(stderr, "osrs shared observation: self wrote %d expected %d\n",
            i, OSRS_SHARED_SELF_OBS_SIZE);
        abort();
    }

    const OsrsInventoryCell* inventory_cells =
        osrs_player_inventory_cells_const(player);
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
        osrs_write_inventory_cell_obs_code(&obs[i], &inventory_cells[cell]);
        i += OSRS_SHARED_INVENTORY_CELL_OBS_FEATURES;
    }

    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        uint8_t item = player->equipped[slot];
        uint16_t content_code = item == ITEM_NONE
            ? 0 : osrs_inventory_content_code_from_item(item);
        obs[i++] = osrs_inventory_cell_obs_code_encode(content_code);
    }

    osrs_write_equipment_effect_aggregate(
        &obs[i], &player->equipment_effect_profile);
    i += OSRS_EQUIPMENT_EFFECT_AGGREGATE_FEATURES;

    if (i != OSRS_SHARED_OBS_SIZE) {
        fprintf(stderr, "osrs shared observation: wrote %d expected %d\n",
            i, OSRS_SHARED_OBS_SIZE);
        abort();
    }
    return i;
}

#endif
