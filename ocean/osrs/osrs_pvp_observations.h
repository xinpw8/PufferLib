#ifndef OSRS_PVP_OBSERVATIONS_H
#define OSRS_PVP_OBSERVATIONS_H

#include <string.h>

#include "osrs_policy.h"
#include "osrs_player_consumables.h"
#include "osrs_pvp_combat.h"
#include "osrs_pvp_gear.h"
#include "osrs_pvp_movement.h"

#define NH_PVP_SPECIFIC_OBS_SIZE 32
#define NH_PVP_NUM_OBS (OSRS_SHARED_OBS_SIZE + NH_PVP_SPECIFIC_OBS_SIZE)
#define NH_PVP_TARGET_SLOTS 1
#define NH_PVP_ACTION_MASK_SIZE OSRS_BASE_ACTION_MASK_SIZE(NH_PVP_TARGET_SLOTS)

static inline int can_use_brew_boost(Player* p) {
    int def_boost = osrs_brew_defence_boost_amount(p->base_defence);
    int def_cap = p->is_lms ? p->base_defence : p->base_defence + def_boost;
    return p->current_defence < def_cap - 1 ||
        p->current_hitpoints <= p->base_hitpoints;
}

static inline int can_restore_stats(Player* p) {
    int stats_drained = p->current_attack < p->base_attack ||
        p->current_defence < p->base_defence ||
        p->current_strength < p->base_strength ||
        p->current_ranged < p->base_ranged ||
        p->current_magic < p->base_magic;
    int prayer_low = p->current_prayer < (int)(p->base_prayer * 0.9f);
    return stats_drained || prayer_low;
}

static inline int can_boost_combat_skills(Player* p) {
    int max_att = p->base_attack + osrs_super_combat_boost_amount(p->base_attack);
    int max_str = p->base_strength + osrs_super_combat_boost_amount(p->base_strength);
    int def_boost = osrs_super_combat_boost_amount(p->base_defence);
    int max_def = p->is_lms ? p->base_defence : p->base_defence + def_boost;
    return max_att > p->current_attack + 1 ||
        max_def > p->current_defence + 1 ||
        max_str > p->current_strength + 1;
}

static inline int can_boost_ranged(Player* p) {
    int max_ranged = p->base_ranged + osrs_ranging_boost_amount(p->base_ranged);
    return max_ranged > p->current_ranged + 1;
}

static inline int can_use_potion(Player* p, int potion_type) {
    if (remaining_ticks(p->potion_timer) > 0) return 0;
    switch (potion_type) {
        case POTION_BREW: return p->brew_doses > 0;
        case POTION_RESTORE: return p->restore_doses > 0;
        case POTION_COMBAT: return p->combat_potion_doses > 0;
        case POTION_RANGED: return p->ranged_potion_doses > 0;
        default: return 0;
    }
}

static inline int pvp_drink_kind_available(
    Player* p,
    OsrsConsumableKind kind
) {
    switch (kind) {
        case OSRS_CONSUMABLE_BREW:
            return can_use_potion(p, POTION_BREW) && can_use_brew_boost(p);
        case OSRS_CONSUMABLE_SUPER_RESTORE:
            return can_use_potion(p, POTION_RESTORE) && can_restore_stats(p);
        case OSRS_CONSUMABLE_SUPER_COMBAT:
            return can_use_potion(p, POTION_COMBAT) && can_boost_combat_skills(p);
        case OSRS_CONSUMABLE_RANGING:
            return can_use_potion(p, POTION_RANGED) && can_boost_ranged(p);
        default:
            return 0;
    }
}

static inline void pvp_shared_observation_input(
    Player* p,
    OsrsSharedObservationInput* out
) {
    AttackStyle style = get_slot_weapon_attack_style(p);
    GearBonuses* gear = get_slot_gear_bonuses(p);
    int spell_base_damage = style == ATTACK_STYLE_MAGIC
        ? get_ice_base_hit(p->current_magic) : 0;
    *out = (OsrsSharedObservationInput){
        .player = p,
        .interaction = &p->interaction,
        .arena_min_x = FIGHT_AREA_BASE_X,
        .arena_max_x = FIGHT_AREA_BASE_X + FIGHT_AREA_WIDTH,
        .arena_min_y = FIGHT_AREA_BASE_Y,
        .arena_max_y = FIGHT_AREA_BASE_Y + FIGHT_AREA_HEIGHT,
        .attack_style = style,
        .attack_range = get_attack_range(p, style),
        .max_hit = calculate_max_hit(p, style, 1.0f, spell_base_damage),
        .attack_speed = gear->attack_speed,
        .defence_stab = gear->stab_defence,
        .defence_slash = gear->slash_defence,
        .defence_crush = gear->crush_defence,
        .defence_magic = gear->magic_defence,
        .defence_ranged = gear->ranged_defence,
        .effective_level = calculate_effective_attack(p, style),
        .attack_bonus = get_attack_bonus(p, style),
        .strength_bonus = get_strength_bonus(p, style),
        .spell_base_damage = spell_base_damage,
        .special_attack_cost = osrs_spec_cost(p->equipped[GEAR_SLOT_WEAPON]),
    };
}

static inline void pvp_write_observations(
    float* obs,
    OsrsEnv* env,
    int agent_idx
) {
    Player* p = &env->players[agent_idx];
    Player* target = &env->players[1 - agent_idx];
    p->last_obs_target_x = target->x;
    p->last_obs_target_y = target->y;

    OsrsSharedObservationInput shared_input;
    pvp_shared_observation_input(p, &shared_input);
    int i = osrs_write_shared_observations(obs, &shared_input);
    if (i != OSRS_SHARED_OBS_SIZE) abort();

    obs[i++] = osrs_policy_ratio(target->current_hitpoints, target->base_hitpoints);
    obs[i++] = osrs_policy_ratio(target->current_prayer, target->base_prayer);
    obs[i++] = osrs_policy_ratio(target->x - p->x, FIGHT_AREA_WIDTH);
    obs[i++] = osrs_policy_ratio(target->y - p->y, FIGHT_AREA_HEIGHT);
    obs[i++] = target->prayer == PRAYER_PROTECT_MELEE ? 1.0f : 0.0f;
    obs[i++] = target->prayer == PRAYER_PROTECT_RANGED ? 1.0f : 0.0f;
    obs[i++] = target->prayer == PRAYER_PROTECT_MAGIC ? 1.0f : 0.0f;
    obs[i++] = target->prayer == PRAYER_SMITE ? 1.0f : 0.0f;
    obs[i++] = target->prayer == PRAYER_REDEMPTION ? 1.0f : 0.0f;
    obs[i++] = target->offensive_prayer == OFFENSIVE_PRAYER_PIETY ? 1.0f : 0.0f;
    obs[i++] = target->offensive_prayer == OFFENSIVE_PRAYER_RIGOUR ? 1.0f : 0.0f;
    obs[i++] = target->offensive_prayer == OFFENSIVE_PRAYER_AUGURY ? 1.0f : 0.0f;
    obs[i++] = osrs_policy_ratio(target->special_energy, 100);
    obs[i++] = osrs_policy_ratio(target->attack_timer, 8);
    obs[i++] = osrs_policy_ratio(target->food_timer, 3);
    obs[i++] = osrs_policy_ratio(target->potion_timer, 3);
    obs[i++] = osrs_policy_ratio(target->frozen_ticks, 32);
    obs[i++] = osrs_policy_ratio(target->freeze_immunity_ticks, 5);
    obs[i++] = target->veng_active ? 1.0f : 0.0f;
    obs[i++] = osrs_policy_ratio(target->veng_cooldown, 50);
    obs[i++] = p->observed_target_lunar_spellbook ? 1.0f : 0.0f;
    obs[i++] = target->last_attack_style == ATTACK_STYLE_MELEE ? 1.0f : 0.0f;
    obs[i++] = target->last_attack_style == ATTACK_STYLE_RANGED ? 1.0f : 0.0f;
    obs[i++] = target->last_attack_style == ATTACK_STYLE_MAGIC ? 1.0f : 0.0f;
    obs[i++] = target->is_moving ? 1.0f : 0.0f;
    int pending_damage = 0;
    for (int hit = 0; hit < p->num_pending_hits; hit++)
        pending_damage += p->pending_hits[hit].damage;
    obs[i++] = osrs_policy_ratio(pending_damage, p->base_hitpoints);
    obs[i++] = osrs_policy_ratio(get_ticks_until_next_hit(target), 6);
    obs[i++] = agent_idx == env->pid_holder ? 1.0f : 0.0f;
    AttackStyle target_style = get_slot_weapon_attack_style(target);
    obs[i++] = target_style == ATTACK_STYLE_MELEE ? 1.0f : 0.0f;
    obs[i++] = target_style == ATTACK_STYLE_RANGED ? 1.0f : 0.0f;
    obs[i++] = target_style == ATTACK_STYLE_MAGIC ? 1.0f : 0.0f;
    obs[i++] = osrs_policy_ratio(
        chebyshev_distance(p->x, p->y, target->x, target->y), 10);
    if (i != NH_PVP_NUM_OBS) abort();
}

static inline void pvp_write_action_mask(
    float* mask,
    OsrsEnv* env,
    int agent_idx,
    const EncounterArenaTopology* topology
) {
    Player* p = &env->players[agent_idx];
    Player* target = &env->players[1 - agent_idx];

    int offset = osrs_base_action_head_mask_offset(
        NH_PVP_TARGET_SLOTS, OSRS_HEAD_PRIMARY);
    mask[offset] = 1.0f;
    for (int action = 1; action < OSRS_PRIMARY_MOVE_ACTIONS; action++) {
        int nx = p->x + ENCOUNTER_MOVE_TARGET_DX[action];
        int ny = p->y + ENCOUNTER_MOVE_TARGET_DY[action];
        mask[offset + action] = can_move(p) &&
            pvp_topology_tile_walkable(topology, nx, ny) ? 1.0f : 0.0f;
    }
    mask[offset + OSRS_PRIMARY_MOVE_ACTIONS] =
        target->current_hitpoints > 0 ? 1.0f : 0.0f;

    offset = osrs_base_action_head_mask_offset(
        NH_PVP_TARGET_SLOTS, OSRS_HEAD_OVERHEAD);
    int has_prayer = p->current_prayer > 0;
    mask[offset + ENCOUNTER_OVERHEAD_NO_CHANGE] = 1.0f;
    mask[offset + ENCOUNTER_OVERHEAD_OFF] = p->prayer != PRAYER_NONE;
    mask[offset + ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE] = has_prayer;
    mask[offset + ENCOUNTER_OVERHEAD_SET_REFRESH_RANGED] = has_prayer;
    mask[offset + ENCOUNTER_OVERHEAD_SET_REFRESH_MAGIC] = has_prayer;
    mask[offset + ENCOUNTER_OVERHEAD_SET_REFRESH_SMITE] = has_prayer && !env->is_lms;
    mask[offset + ENCOUNTER_OVERHEAD_SET_REFRESH_REDEMPTION] = has_prayer && !env->is_lms;

    int inventory_has_empty_cell =
        osrs_first_empty_inventory_cell(p->inventory_cells, -1) >= 0;
    int cell_equip_slot[OSRS_INVENTORY_SIZE];
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
        const OsrsItemContentMetadata* metadata =
            osrs_inventory_cell_metadata(&p->inventory_cells[cell]);
        cell_equip_slot[cell] =
            osrs_can_equip_metadata(
                p, metadata, inventory_has_empty_cell)
                    ? metadata->gear_slot : -1;
    }
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        offset = osrs_base_action_head_mask_offset(
            NH_PVP_TARGET_SLOTS, OSRS_HEAD_EQUIP_SLOT(slot));
        mask[offset] = 1.0f;
        for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++)
            mask[offset + cell + 1] =
                cell_equip_slot[cell] == slot ? 1.0f : 0.0f;
    }

    offset = osrs_base_action_head_mask_offset(
        NH_PVP_TARGET_SLOTS, OSRS_HEAD_EAT);
    mask[offset] = 1.0f;
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
        OsrsInventoryClickResolution resolution =
            osrs_inventory_cell_click_classify(&p->inventory_cells[cell]);
        mask[offset + cell + 1] = resolution.click_action == OSRS_CLICK_EAT &&
            osrs_can_eat_consumable_kind(p, resolution.consumable_kind)
                ? 1.0f : 0.0f;
    }

    offset = osrs_base_action_head_mask_offset(
        NH_PVP_TARGET_SLOTS, OSRS_HEAD_DRINK);
    mask[offset] = 1.0f;
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
        OsrsInventoryClickResolution resolution =
            osrs_inventory_cell_click_classify(&p->inventory_cells[cell]);
        mask[offset + cell + 1] = resolution.click_action == OSRS_CLICK_DRINK &&
            pvp_drink_kind_available(p, resolution.consumable_kind)
                ? 1.0f : 0.0f;
    }

    offset = osrs_base_action_head_mask_offset(
        NH_PVP_TARGET_SLOTS, OSRS_HEAD_SPELL);
    mask[offset + OSRS_SPELL_NONE] = 1.0f;
    mask[offset + OSRS_SPELL_BLOOD_BARRAGE] = can_cast_blood_spell(p);
    mask[offset + OSRS_SPELL_ICE_BARRAGE] = can_cast_ice_spell(p);
    mask[offset + OSRS_SPELL_VENGEANCE] = !env->is_lms &&
        p->is_lunar_spellbook && !p->veng_active &&
        remaining_ticks(p->veng_cooldown) == 0 && p->current_magic >= 94;
    mask[offset + OSRS_SPELL_DEATH_CHARGE] = 0.0f;

    offset = osrs_base_action_head_mask_offset(
        NH_PVP_TARGET_SLOTS, OSRS_HEAD_SPECIAL);
    mask[offset] = 1.0f;
    mask[offset + 1] = can_toggle_spec(p);
    mask[offset + 2] = p->spec_armed;

    offset = osrs_base_action_head_mask_offset(
        NH_PVP_TARGET_SLOTS, OSRS_HEAD_OFFENSIVE);
    mask[offset + ENCOUNTER_OFFENSIVE_NO_CHANGE] = 1.0f;
    mask[offset + ENCOUNTER_OFFENSIVE_OFF] =
        p->offensive_prayer != OFFENSIVE_PRAYER_NONE;
    mask[offset + ENCOUNTER_OFFENSIVE_SET_REFRESH_PIETY] = has_prayer;
    mask[offset + ENCOUNTER_OFFENSIVE_SET_REFRESH_RIGOUR] = has_prayer;
    mask[offset + ENCOUNTER_OFFENSIVE_SET_REFRESH_AUGURY] = has_prayer;
}

static inline void pvp_write_action_mask_bytes(
    unsigned char* mask,
    OsrsEnv* env,
    int agent_idx,
    const EncounterArenaTopology* topology
) {
    Player* p = &env->players[agent_idx];
    Player* target = &env->players[1 - agent_idx];
    memset(mask, 0, NH_PVP_ACTION_MASK_SIZE);

    int offset = osrs_base_action_head_mask_offset(
        NH_PVP_TARGET_SLOTS, OSRS_HEAD_PRIMARY);
    mask[offset] = 1;
    if (can_move(p)) {
        for (int action = 1; action < OSRS_PRIMARY_MOVE_ACTIONS; action++) {
            int nx = p->x + ENCOUNTER_MOVE_TARGET_DX[action];
            int ny = p->y + ENCOUNTER_MOVE_TARGET_DY[action];
            mask[offset + action] =
                pvp_topology_tile_walkable(topology, nx, ny);
        }
    }
    mask[offset + OSRS_PRIMARY_MOVE_ACTIONS] =
        target->current_hitpoints > 0;

    offset = osrs_base_action_head_mask_offset(
        NH_PVP_TARGET_SLOTS, OSRS_HEAD_OVERHEAD);
    int has_prayer = p->current_prayer > 0;
    mask[offset + ENCOUNTER_OVERHEAD_NO_CHANGE] = 1;
    mask[offset + ENCOUNTER_OVERHEAD_OFF] = p->prayer != PRAYER_NONE;
    mask[offset + ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE] = has_prayer;
    mask[offset + ENCOUNTER_OVERHEAD_SET_REFRESH_RANGED] = has_prayer;
    mask[offset + ENCOUNTER_OVERHEAD_SET_REFRESH_MAGIC] = has_prayer;
    mask[offset + ENCOUNTER_OVERHEAD_SET_REFRESH_SMITE] =
        has_prayer && !env->is_lms;
    mask[offset + ENCOUNTER_OVERHEAD_SET_REFRESH_REDEMPTION] =
        has_prayer && !env->is_lms;

    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        offset = osrs_base_action_head_mask_offset(
            NH_PVP_TARGET_SLOTS, OSRS_HEAD_EQUIP_SLOT(slot));
        mask[offset] = 1;
    }
    int eat_offset = osrs_base_action_head_mask_offset(
        NH_PVP_TARGET_SLOTS, OSRS_HEAD_EAT);
    int drink_offset = osrs_base_action_head_mask_offset(
        NH_PVP_TARGET_SLOTS, OSRS_HEAD_DRINK);
    mask[eat_offset] = 1;
    mask[drink_offset] = 1;
    int inventory_has_empty_cell =
        osrs_first_empty_inventory_cell(p->inventory_cells, -1) >= 0;
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
        OsrsInventoryClickResolution resolution =
            osrs_inventory_cell_click_classify(&p->inventory_cells[cell]);
        if (resolution.click_action == OSRS_CLICK_EQUIP) {
            const OsrsItemContentMetadata* metadata =
                osrs_inventory_cell_metadata(
                    &p->inventory_cells[cell]);
            if (!osrs_can_equip_metadata(
                    p, metadata, inventory_has_empty_cell)) {
                continue;
            }
            int gear_slot = metadata->gear_slot;
            offset = osrs_base_action_head_mask_offset(
                NH_PVP_TARGET_SLOTS, OSRS_HEAD_EQUIP_SLOT(gear_slot));
            mask[offset + cell + 1] = 1;
        } else if (resolution.click_action == OSRS_CLICK_EAT &&
                osrs_can_eat_consumable_kind(
                    p, resolution.consumable_kind)) {
            mask[eat_offset + cell + 1] = 1;
        } else if (resolution.click_action == OSRS_CLICK_DRINK &&
                pvp_drink_kind_available(
                    p, resolution.consumable_kind)) {
            mask[drink_offset + cell + 1] = 1;
        }
    }

    offset = osrs_base_action_head_mask_offset(
        NH_PVP_TARGET_SLOTS, OSRS_HEAD_SPELL);
    mask[offset + OSRS_SPELL_NONE] = 1;
    mask[offset + OSRS_SPELL_BLOOD_BARRAGE] = can_cast_blood_spell(p);
    mask[offset + OSRS_SPELL_ICE_BARRAGE] = can_cast_ice_spell(p);
    mask[offset + OSRS_SPELL_VENGEANCE] = !env->is_lms &&
        p->is_lunar_spellbook && !p->veng_active &&
        remaining_ticks(p->veng_cooldown) == 0 && p->current_magic >= 94;

    offset = osrs_base_action_head_mask_offset(
        NH_PVP_TARGET_SLOTS, OSRS_HEAD_SPECIAL);
    mask[offset] = 1;
    mask[offset + 1] = can_toggle_spec(p);
    mask[offset + 2] = p->spec_armed;

    offset = osrs_base_action_head_mask_offset(
        NH_PVP_TARGET_SLOTS, OSRS_HEAD_OFFENSIVE);
    mask[offset + ENCOUNTER_OFFENSIVE_NO_CHANGE] = 1;
    mask[offset + ENCOUNTER_OFFENSIVE_OFF] =
        p->offensive_prayer != OFFENSIVE_PRAYER_NONE;
    mask[offset + ENCOUNTER_OFFENSIVE_SET_REFRESH_PIETY] = has_prayer;
    mask[offset + ENCOUNTER_OFFENSIVE_SET_REFRESH_RIGOUR] = has_prayer;
    mask[offset + ENCOUNTER_OFFENSIVE_SET_REFRESH_AUGURY] = has_prayer;
}

#endif
