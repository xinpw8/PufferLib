#ifndef OSRS_PVP_ACTIONS_H
#define OSRS_PVP_ACTIONS_H

#include "osrs_types.h"
#include "osrs_items.h"
#include "osrs_consumables.h"
#include "osrs_player_consumables.h"
#include "osrs_pvp_gear.h"
#include "osrs_pvp_combat.h"
#include "osrs_pvp_movement.h"
#include "osrs_pvp_observations.h"
#include "osrs_encounter.h"

/* fury +3 + neitiznot +3, worn in every loadout */
#define PRAYER_BONUS 6

static void eat_food(Player* p, int is_karambwan) {
    osrs_player_eat_food_type(p, is_karambwan ? FOOD_KARAMBWAN : FOOD_SHARK);
}

static void drink_potion(Player* p, int potion_type) {
    if (p->potion_timer > 0) return;

    switch (potion_type) {
        case POTION_BREW: {
            if (p->brew_doses <= 0) return;
            p->brew_doses--;
            BrewResult br = osrs_brew_effect(p->base_hitpoints, p->base_defence,
                                             p->current_attack, p->current_strength,
                                             p->current_ranged, p->current_magic);
            int hp_before = p->current_hitpoints;
            int max_hp = p->base_hitpoints + br.hp_healed;
            int actual_heal = max_int(0, min_int(br.hp_healed, max_hp - hp_before));
            int waste = br.hp_healed - actual_heal;
            int def_before = p->current_defence;
            int max_def = p->is_lms ? p->base_defence : p->base_defence + br.def_boost;
            p->current_defence = clamp(def_before + br.def_boost, 0, max_def);
            p->current_hitpoints = clamp(hp_before + br.hp_healed, 0, max_hp);
            p->last_brew_heal = actual_heal;
            p->last_brew_waste = waste;
            p->current_attack = clamp(p->current_attack - br.att_drain, 0, 255);
            p->current_strength = clamp(p->current_strength - br.str_drain, 0, 255);
            p->current_magic = clamp(p->current_magic - br.magic_drain, 0, 255);
            p->current_ranged = clamp(p->current_ranged - br.range_drain, 0, 255);
            p->last_potion_type = potion_type;
            p->ate_brew_this_tick = 1;
            break;
        }

        case POTION_RESTORE: {
            if (p->restore_doses <= 0) return;
            p->restore_doses--;
            int had_restore_need = (
                p->current_attack < p->base_attack ||
                p->current_strength < p->base_strength ||
                p->current_defence < p->base_defence ||
                p->current_ranged < p->base_ranged ||
                p->current_magic < p->base_magic ||
                p->current_prayer < p->base_prayer
            );
            p->current_prayer = clamp(
                p->current_prayer + osrs_super_restore_amount(p->base_prayer),
                0, p->base_prayer);
            int atk_restore = osrs_super_restore_amount(p->base_attack);
            int str_restore = osrs_super_restore_amount(p->base_strength);
            int def_restore = osrs_super_restore_amount(p->base_defence);
            int rng_restore = osrs_super_restore_amount(p->base_ranged);
            int mag_restore = osrs_super_restore_amount(p->base_magic);
            if (p->current_attack < p->base_attack) {
                p->current_attack = clamp(p->current_attack + atk_restore, 0, p->base_attack);
            }
            if (p->current_strength < p->base_strength) {
                p->current_strength = clamp(p->current_strength + str_restore, 0, p->base_strength);
            }
            if (p->current_defence < p->base_defence) {
                p->current_defence = clamp(p->current_defence + def_restore, 0, p->base_defence);
            }
            if (p->current_ranged < p->base_ranged) {
                p->current_ranged = clamp(p->current_ranged + rng_restore, 0, p->base_ranged);
            }
            if (p->current_magic < p->base_magic) {
                p->current_magic = clamp(p->current_magic + mag_restore, 0, p->base_magic);
            }
            p->last_potion_type = potion_type;
            p->last_potion_was_waste = had_restore_need ? 0 : 1;
            break;
        }

        case POTION_COMBAT: {
            if (p->combat_potion_doses <= 0) return;
            p->combat_potion_doses--;
            int atk_boost = osrs_super_combat_boost_amount(p->base_attack);
            int str_boost = osrs_super_combat_boost_amount(p->base_strength);
            int def_boost = osrs_super_combat_boost_amount(p->base_defence);
            int atk_cap = p->base_attack + atk_boost;
            int str_cap = p->base_strength + str_boost;
            int def_cap = p->is_lms ? p->base_defence : p->base_defence + def_boost;
            int had_boost_need = (
                p->current_attack < atk_cap ||
                p->current_strength < str_cap ||
                p->current_defence < def_cap
            );
            if (p->current_attack < atk_cap) {
                p->current_attack = clamp(p->current_attack + atk_boost, 0, atk_cap);
            }
            if (p->current_strength < str_cap) {
                p->current_strength = clamp(p->current_strength + str_boost, 0, str_cap);
            }
            if (p->current_defence < def_cap) {
                p->current_defence = clamp(p->current_defence + def_boost, 0, def_cap);
            }
            p->last_potion_type = potion_type;
            p->last_potion_was_waste = had_boost_need ? 0 : 1;
            break;
        }

        case POTION_RANGED: {
            if (p->ranged_potion_doses <= 0) return;
            p->ranged_potion_doses--;
            int rng_boost = osrs_ranging_boost_amount(p->base_ranged);
            int rng_cap = p->base_ranged + rng_boost;
            int had_boost_need = p->current_ranged < rng_cap;
            if (p->current_ranged < rng_cap) {
                p->current_ranged = clamp(p->current_ranged + rng_boost, 0, rng_cap);
            }
            p->last_potion_type = potion_type;
            p->last_potion_was_waste = had_boost_need ? 0 : 1;
            break;
        }
    }

    p->potion_timer = 3;
    p->food_timer = 3;
}

/* consumable timers are NOT decremented here: they tick after execute_switches
   in pvp_step so observations show the post-use countdown */
static void update_timers(Player* p) {
    p->damage_applied_this_tick = 0;

    if (p->has_attack_timer) {
        p->attack_timer_uncapped -= 1;
        if (p->attack_timer >= 0) {
            p->attack_timer -= 1;
        }
    }
    if (p->frozen_ticks > 0) p->frozen_ticks--;
    if (p->freeze_immunity_ticks > 0) p->freeze_immunity_ticks--;
    if (p->veng_cooldown > 0) p->veng_cooldown--;

    if (!p->is_lms) {
        encounter_drain_all_prayers(p, PRAYER_BONUS);
    } else {
        p->prayer_just_activated = 0;
        p->offensive_prayer_just_activated = 0;
    }

    if (p->run_energy < OSRS_RUN_ENERGY_FULL && (!p->is_moving || !p->is_running)) {
        p->run_recovery_ticks += 1;
        if (p->run_recovery_ticks >= RUN_ENERGY_RECOVER_TICKS) {
            p->run_energy = clamp(
                p->run_energy + OSRS_RUN_ENERGY_UNITS_PER_PERCENT,
                0,
                OSRS_RUN_ENERGY_FULL
            );
            p->run_recovery_ticks = 0;
        }
    } else {
        p->run_recovery_ticks = 0;
    }

    if (p->spec_regen_active && p->special_energy < 100) {
        osrs_tick_special_regen(p);
    } else if (p->spec_regen_active) {
        p->item_effect_state.special_regen_ticks = 0;
    }
}

static void reset_tick_flags(Player* p) {
    p->just_attacked = 0;
    p->last_queued_hit_damage = 0;
    p->attack_was_on_prayer = 0;
    p->player_prayed_correct = 0;
    p->target_prayed_correct = 0;
    p->tick_damage_scale = 0.0f;
    p->damage_dealt_scale = 0.0f;
    p->damage_received_scale = 0.0f;
    p->last_food_heal = 0;
    p->last_food_waste = 0;
    p->last_karambwan_heal = 0;
    p->last_karambwan_waste = 0;
    p->last_brew_heal = 0;
    p->last_brew_waste = 0;
    p->last_potion_type = 0;
    p->last_potion_was_waste = 0;
    p->attack_click_canceled = 0;
    p->attack_click_ready = 0;
    p->attack_style_this_tick = ATTACK_STYLE_NONE;
    p->magic_type_this_tick = 0;
    p->used_special_this_tick = 0;
    p->ate_food_this_tick = 0;
    p->ate_karambwan_this_tick = 0;
    p->ate_brew_this_tick = 0;
    p->cast_veng_this_tick = 0;
    p->clicks_this_tick = 0;
}

static void execute_switches(OsrsEnv* env, int agent_idx, int* actions) {
    Player* p = &env->players[agent_idx];
    const CollisionMap* cmap = (const CollisionMap*)env->collision_map;

    p->consumable_used_this_tick = 0;

    int overhead_action = actions[HEAD_OVERHEAD];
    int offensive_action = actions[HEAD_OFFENSIVE];

    if (env->is_lms &&
        (overhead_action == ENCOUNTER_OVERHEAD_SET_REFRESH_SMITE ||
         overhead_action == ENCOUNTER_OVERHEAD_SET_REFRESH_REDEMPTION)) {
        overhead_action = ENCOUNTER_OVERHEAD_NO_CHANGE;
    }
    if (p->current_prayer <= 0) {
        if (overhead_action >= ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE)
            overhead_action = ENCOUNTER_OVERHEAD_NO_CHANGE;
        if (offensive_action >= ENCOUNTER_OFFENSIVE_SET_REFRESH_PIETY)
            offensive_action = ENCOUNTER_OFFENSIVE_NO_CHANGE;
    }

    OverheadPrayer prev_prayer = p->prayer;
    OffensivePrayer prev_offensive = p->offensive_prayer;
    int prayer_commanded =
        overhead_action != ENCOUNTER_OVERHEAD_NO_CHANGE ||
        offensive_action != ENCOUNTER_OFFENSIVE_NO_CHANGE;
    if (encounter_apply_overhead_action(&p->prayer, overhead_action)) {
        p->prayer_just_activated = 1;
    }
    if (encounter_apply_offensive_action(&p->offensive_prayer, offensive_action)) {
        p->offensive_prayer_just_activated = 1;
    }
    if (prayer_commanded || p->prayer != prev_prayer || p->offensive_prayer != prev_offensive)
        p->clicks_this_tick++;
    int loadout_action = actions[HEAD_LOADOUT];
    int loadout_switches = apply_loadout(p, loadout_action);
    p->clicks_this_tick += loadout_switches;
    if (loadout_switches > 0)
        osrs_interaction_check_interrupt(&p->interaction, OSRS_IACT_EQUIP);

    if (loadout_action == LOADOUT_SPEC_MELEE || loadout_action == LOADOUT_SPEC_RANGE ||
        loadout_action == LOADOUT_SPEC_MAGIC || loadout_action == LOADOUT_GMAUL) {
        p->spec_armed = 1;
    }
    int food_action = actions[HEAD_FOOD];
    if (food_action == FOOD_EAT && can_eat_food(p)) {
        eat_food(p, 0);
        p->consumable_used_this_tick = 1;
        p->clicks_this_tick++;
        osrs_interaction_check_interrupt(&p->interaction, OSRS_IACT_EAT);
    }

    int potion_action = actions[HEAD_POTION];
    int potion_ok = 0;
    switch (potion_action) {
        case POTION_BREW:
            potion_ok = can_use_potion(p, POTION_BREW) && can_use_brew_boost(p);
            break;
        case POTION_RESTORE:
            potion_ok = can_use_potion(p, POTION_RESTORE) && can_restore_stats(p);
            break;
        case POTION_COMBAT:
            potion_ok = can_use_potion(p, POTION_COMBAT) && can_boost_combat_skills(p);
            break;
        case POTION_RANGED:
            potion_ok = can_use_potion(p, POTION_RANGED) && can_boost_ranged(p);
            break;
        default:
            break;
    }
    if (potion_ok) {
        drink_potion(p, potion_action);
        p->consumable_used_this_tick = 1;
        p->clicks_this_tick++;
        osrs_interaction_check_interrupt(&p->interaction, OSRS_IACT_DRINK);
    }

    int karam_action = actions[HEAD_KARAMBWAN];
    if (karam_action == KARAM_EAT && can_eat_karambwan(p)) {
        eat_food(p, 1);
        p->consumable_used_this_tick = 1;
        p->clicks_this_tick++;
        osrs_interaction_check_interrupt(&p->interaction, OSRS_IACT_EAT);
    }
    int combat_action = actions[HEAD_COMBAT];
    int head_move = actions[HEAD_MOVE];
    int is_spec_loadout = (loadout_action == LOADOUT_SPEC_MELEE ||
                           loadout_action == LOADOUT_SPEC_RANGE ||
                           loadout_action == LOADOUT_SPEC_MAGIC ||
                           loadout_action == LOADOUT_GMAUL);

    int attack_claims_tick = is_attack_action(combat_action);
    if (attack_claims_tick) {
        env->pvp_runtime.walk_dest_x[agent_idx] = -1;
        env->pvp_runtime.walk_dest_y[agent_idx] = -1;
    }

    int command_issued = 0;
    if (!is_spec_loadout && !attack_claims_tick && head_move > 0 && head_move < MOVE_DIM) {
        pvp_set_walk_dest_from_head_move(env, agent_idx, head_move);
        command_issued = 1;
    } else if (!is_spec_loadout && is_move_action(combat_action)) {
        int tx = p->last_obs_target_x;
        int ty = p->last_obs_target_y;
        int dest_x = -1, dest_y = -1;
        switch (combat_action) {
            case MOVE_ADJACENT:
                if (!select_closest_adjacent_tile(p, tx, ty, &dest_x, &dest_y, cmap)) {
                    dest_x = -1; dest_y = -1;
                }
                break;
            case MOVE_UNDER:
                if (is_in_wilderness(tx, ty) && collision_tile_walkable(cmap, 0, tx, ty)) {
                    dest_x = tx; dest_y = ty;
                }
                break;
            case MOVE_DIAGONAL:
                if (!select_closest_diagonal_tile(p, tx, ty, &dest_x, &dest_y, cmap)) {
                    dest_x = -1; dest_y = -1;
                }
                break;
            case MOVE_FARCAST_2:
            case MOVE_FARCAST_3:
            case MOVE_FARCAST_4:
            case MOVE_FARCAST_5:
            case MOVE_FARCAST_6:
            case MOVE_FARCAST_7: {
                int fd = combat_action - MOVE_FARCAST_2 + 2;
                if (!select_farcast_tile(p, tx, ty, fd, &dest_x, &dest_y, cmap)) {
                    dest_x = -1; dest_y = -1;
                }
                break;
            }
            default:
                break;
        }
        env->pvp_runtime.walk_dest_x[agent_idx] = dest_x;
        env->pvp_runtime.walk_dest_y[agent_idx] = dest_y;
        command_issued = (dest_x >= 0);
    }
    /* no clearing else: walk_dest persists until arrival, matching OSRS click
       semantics (the SDK sets it back to -1 when the player gets there) */
    if (command_issued) {
        p->clicks_this_tick++;
        osrs_interaction_check_interrupt(&p->interaction, OSRS_IACT_MOVE);
    }
    int veng_action = actions[HEAD_VENG];
    if (veng_action == VENG_CAST && p->is_lunar_spellbook &&
        !p->veng_active && remaining_ticks(p->veng_cooldown) == 0 &&
        p->current_magic >= 94) {
        p->veng_active = 1;
        p->veng_cooldown = 50;
        p->cast_veng_this_tick = 1;
        p->clicks_this_tick++;
    }
}

typedef struct {
    int attack_action;
    int move_action;
    int explicit_move_in_progress;
    int is_gmaul;
} PvpAttackDecode;

static PvpAttackDecode pvp_decode_attack_actions(
    OsrsEnv* env, int agent_idx, Player* p, const int* actions
) {
    int loadout_action = actions[HEAD_LOADOUT];
    int combat_action = actions[HEAD_COMBAT];
    int attack_action = is_attack_action(combat_action) ? combat_action : ATTACK_NONE;
    int move_action = is_move_action(combat_action) ? combat_action : MOVE_NONE;
    int explicit_move_in_progress = (actions[HEAD_MOVE] > 0 && actions[HEAD_MOVE] < MOVE_DIM)
        || env->pvp_runtime.walk_dest_x[agent_idx] >= 0;

    int is_gmaul = (loadout_action == LOADOUT_GMAUL);
    if (is_gmaul) {
        attack_action = ATTACK_ATK;
        move_action = MOVE_NONE;
    }

    int current_loadout = get_current_loadout(p);
    int in_mage_loadout = (current_loadout == LOADOUT_MAGE);
    int in_tank_loadout = (current_loadout == LOADOUT_TANK);
    if (attack_action == ATTACK_ATK && (in_mage_loadout || in_tank_loadout) && !is_gmaul) {
        attack_action = ATTACK_NONE;
    }

    return (PvpAttackDecode){
        .attack_action = attack_action,
        .move_action = move_action,
        .explicit_move_in_progress = explicit_move_in_progress,
        .is_gmaul = is_gmaul,
    };
}

static void execute_attack_movement(OsrsEnv* env, int agent_idx, int* actions) {
    Player* p = &env->players[agent_idx];
    Player* t = &env->players[1 - agent_idx];
    const CollisionMap* cmap = (const CollisionMap*)env->collision_map;

    PvpAttackDecode decode = pvp_decode_attack_actions(env, agent_idx, p, actions);

    if (decode.attack_action != ATTACK_NONE)
        osrs_interaction_set(&p->interaction, 1 - agent_idx);

    int has_attack = (decode.attack_action != ATTACK_NONE) || osrs_interaction_active(&p->interaction);
    int dist = chebyshev_distance(p->x, p->y, t->x, t->y);

    AttackStyle attack_style = ATTACK_STYLE_NONE;
    if (decode.attack_action != ATTACK_NONE) {
        switch (decode.attack_action) {
            case ATTACK_ATK:
                attack_style = get_slot_weapon_attack_style(p);
                break;
            case ATTACK_ICE:
                attack_style = ATTACK_STYLE_MAGIC;
                break;
            case ATTACK_BLOOD:
                attack_style = ATTACK_STYLE_MAGIC;
                break;
            default:
                break;
        }
    } else if (osrs_interaction_active(&p->interaction)) {
        attack_style = get_slot_weapon_attack_style(p);
    }
    if (decode.attack_action == ATTACK_ICE && !can_cast_ice_spell(p)) {
        attack_style = ATTACK_STYLE_NONE;
    }
    if (decode.attack_action == ATTACK_BLOOD && !can_cast_blood_spell(p)) {
        attack_style = ATTACK_STYLE_NONE;
    }

    p->did_attack_auto_move = 0;

    if (has_attack && decode.move_action == MOVE_NONE && !decode.explicit_move_in_progress && can_move(p)) {
        if (attack_style == ATTACK_STYLE_MELEE && !is_in_melee_range(p, t)) {
            int adj_x, adj_y;
            if (select_closest_adjacent_tile(p, t->x, t->y, &adj_x, &adj_y, cmap)) {
                set_destination(p, adj_x, adj_y, cmap);
            }
            p->did_attack_auto_move = 1;
            dist = chebyshev_distance(p->x, p->y, t->x, t->y);
        }
    }

    if (has_attack && dist == 0 && can_move(p)) {
        step_out_from_same_tile(p, t, cmap);
    }
}

/* runs after BOTH players' attack movement so range checks use final positions;
   checking ranges in the movement phase reintroduces the PID-dependent same-tile bug */
static void execute_attack_combat(OsrsEnv* env, int agent_idx, int* actions) {
    Player* p = &env->players[agent_idx];
    Player* t = &env->players[1 - agent_idx];
    const CollisionMap* cmap = (const CollisionMap*)env->collision_map;

    PvpAttackDecode decode = pvp_decode_attack_actions(env, agent_idx, p, actions);

    if (decode.attack_action == ATTACK_NONE && osrs_interaction_active(&p->interaction)) {
        AttackStyle weapon_style = get_slot_weapon_attack_style(p);
        if (weapon_style != ATTACK_STYLE_MAGIC) {
            decode.attack_action = ATTACK_ATK;
        }
    }

    int attack_ready = can_attack_now(p);
    int has_attack = (decode.attack_action != ATTACK_NONE);
    int dist = chebyshev_distance(p->x, p->y, t->x, t->y);

    AttackStyle attack_style = ATTACK_STYLE_NONE;
    int magic_type = 0;

    switch (decode.attack_action) {
        case ATTACK_ATK:
            attack_style = get_slot_weapon_attack_style(p);
            break;
        case ATTACK_ICE:
            attack_style = ATTACK_STYLE_MAGIC;
            magic_type = 1;
            break;
        case ATTACK_BLOOD:
            attack_style = ATTACK_STYLE_MAGIC;
            magic_type = 2;
            break;
        default:
            break;
    }
    if (decode.attack_action == ATTACK_ICE && !can_cast_ice_spell(p)) {
        attack_style = ATTACK_STYLE_NONE;
    }
    if (decode.attack_action == ATTACK_BLOOD && !can_cast_blood_spell(p)) {
        attack_style = ATTACK_STYLE_NONE;
    }

    int can_attack = attack_ready || (decode.is_gmaul && is_granite_maul_attack_available(p));

    switch (decode.attack_action) {
        case ATTACK_ATK:
            if (can_attack && attack_style != ATTACK_STYLE_NONE) {
                AttackStyle actual_style = (attack_style == ATTACK_STYLE_MAGIC)
                    ? ATTACK_STYLE_MELEE
                    : attack_style;
                int in_attack_range = 0;
                if (actual_style == ATTACK_STYLE_MELEE) {
                    in_attack_range = is_in_melee_range(p, t);
                } else {
                    int range = get_attack_range(p, actual_style);
                    in_attack_range = (dist > 0 && dist <= range);
                }
                if (in_attack_range) {
                    int is_special = p->spec_armed && is_special_ready(p, actual_style);
                    perform_attack(env, agent_idx, 1 - agent_idx, actual_style, is_special, 0, dist);
                    if (is_special)
                        osrs_spec_disarm(&p->spec_armed);
                    p->clicks_this_tick++;
                }
            }
            break;
        case ATTACK_ICE:
        case ATTACK_BLOOD:
            if (attack_ready && attack_style == ATTACK_STYLE_MAGIC) {
                int range = get_attack_range(p, ATTACK_STYLE_MAGIC);
                if (dist > 0 && dist <= range) {
                    perform_attack(env, agent_idx, 1 - agent_idx, ATTACK_STYLE_MAGIC, 0, magic_type, dist);
                    p->clicks_this_tick++;
                }
            }
            break;
        default:
            break;
    }

    if (has_attack && decode.move_action == MOVE_NONE && !decode.explicit_move_in_progress
            && can_move(p) && !p->did_attack_auto_move) {
        int in_range = 0;
        int auto_walk_range = 1;
        switch (attack_style) {
            case ATTACK_STYLE_MELEE:
                in_range = is_in_melee_range(p, t);
                break;
            case ATTACK_STYLE_RANGED: {
                int range = get_attack_range(p, ATTACK_STYLE_RANGED);
                auto_walk_range = range;
                in_range = (dist <= range);
                break;
            }
            case ATTACK_STYLE_MAGIC: {
                int range = get_attack_range(p, ATTACK_STYLE_MAGIC);
                auto_walk_range = range;
                in_range = (dist <= range);
                break;
            }
            default:
                in_range = 1;
                break;
        }
        if (!in_range) {
            if (attack_style == ATTACK_STYLE_MELEE) {
                int adj_x, adj_y;
                if (select_closest_adjacent_tile(p, t->x, t->y, &adj_x, &adj_y, cmap)) {
                    set_destination(p, adj_x, adj_y, cmap);
                }
            } else {
                move_toward_target(p, t, auto_walk_range, cmap);
            }
        }
    }
}

static float calculate_reward(OsrsEnv* env, int agent_idx) {
    float reward = 0.0f;
    Player* p = &env->players[agent_idx];
    Player* t = &env->players[1 - agent_idx];
    const RewardShapingConfig* cfg = &env->shaping;

    if (env->episode_over) {
        if (env->winner == agent_idx) {
            reward += 1.0f;
        }
    }

    if (cfg->prayer_penalty_enabled && !t->just_attacked) {
        int overhead = env->last_executed_actions[agent_idx * NUM_ACTION_HEADS + HEAD_OVERHEAD];
        if (overhead == OVERHEAD_MAGE || overhead == OVERHEAD_RANGED || overhead == OVERHEAD_MELEE) {
            reward += cfg->prayer_switch_no_attack_penalty;
        }
    }

    if (cfg->click_penalty_enabled && p->clicks_this_tick > cfg->click_penalty_threshold) {
        int excess = p->clicks_this_tick - cfg->click_penalty_threshold;
        reward += cfg->click_penalty_coef * (float)excess;
    }

    float base_hp = (float)p->base_hitpoints;
    if (p->damage_dealt_scale > 0.0f) {
        reward += p->damage_dealt_scale * base_hp * 0.005f;
    }
    if (t->just_attacked && p->player_prayed_correct) {
        reward += 0.01f;
    }

    if (!cfg->enabled) {
        return reward;
    }

    if (env->episode_over) {
        if (env->winner == agent_idx) {
            if (t->food_count > 0 || t->karambwan_count > 0 || t->brew_doses > 0) {
                reward += cfg->ko_bonus;
            }
            float opp_total = (float)(t->food_count + t->karambwan_count
                                       + t->brew_doses + t->restore_doses
                                       + t->combat_potion_doses
                                       + t->ranged_potion_doses);
            float max_total = (float)(MAXED_FOOD_COUNT + MAXED_KARAMBWAN_COUNT
                                      + MAXED_BREW_DOSES + MAXED_RESTORE_DOSES
                                      + MAXED_COMBAT_POTION_DOSES
                                      + MAXED_RANGED_POTION_DOSES);
            if (max_total > 0.0f) {
                reward += cfg->ko_supplies_bonus_coef * (opp_total / max_total);
            }
        } else if (env->winner == (1 - agent_idx)) {
            if (p->food_count > 0 || p->karambwan_count > 0 || p->brew_doses > 0) {
                reward += cfg->wasted_resources_penalty;
            }
        }
    }
    float tick_shaping = 0.0f;

    if (p->damage_dealt_scale > 0.0f) {
        float damage_hp = p->damage_dealt_scale * base_hp;
        tick_shaping += damage_hp * cfg->damage_dealt_coef;
        if (damage_hp >= (float)cfg->damage_burst_threshold) {
            tick_shaping += (damage_hp - (float)cfg->damage_burst_threshold + 1.0f)
                          * cfg->damage_burst_bonus;
        }
    }

    if (p->damage_received_scale > 0.0f) {
        tick_shaping += p->damage_received_scale * base_hp * cfg->damage_received_coef;
    }

    if (t->just_attacked) {
        if (p->player_prayed_correct) {
            tick_shaping += cfg->correct_prayer_bonus;
        } else {
            tick_shaping += cfg->wrong_prayer_penalty;
        }
    }

    if (p->just_attacked) {
        if (!p->target_prayed_correct) {
            tick_shaping += cfg->off_prayer_hit_bonus;
        }

        if (p->attack_style_this_tick == ATTACK_STYLE_MELEE
            && p->frozen_ticks > 0 && !is_in_melee_range(p, t)) {
            tick_shaping += cfg->melee_frozen_penalty;
        }

        if (p->used_special_this_tick) {
            if (t->prayer != PRAYER_PROTECT_MELEE) {
                tick_shaping += cfg->spec_off_prayer_bonus;
            }
            AttackStyle target_style = get_slot_weapon_attack_style(t);
            if (target_style == ATTACK_STYLE_MAGIC) {
                tick_shaping += cfg->spec_low_defence_bonus;
            }
            float target_hp_pct = (float)t->current_hitpoints / (float)t->base_hitpoints;
            if (target_hp_pct < 0.5f) {
                tick_shaping += cfg->spec_low_hp_bonus;
            }
        }

        if (p->attack_style_this_tick == ATTACK_STYLE_MAGIC) {
            AttackStyle weapon_style = get_slot_weapon_attack_style(p);
            if (weapon_style != ATTACK_STYLE_MAGIC) {
                tick_shaping += cfg->magic_no_staff_penalty;
            }
        }

        GearBonuses* gear = get_slot_gear_bonuses(p);
        int attack_bonus = 0;
        switch (p->attack_style_this_tick) {
            case ATTACK_STYLE_MAGIC:
                attack_bonus = gear->magic_attack;
                break;
            case ATTACK_STYLE_RANGED:
                attack_bonus = gear->ranged_attack;
                break;
            case ATTACK_STYLE_MELEE:
                attack_bonus = gear->slash_attack;
                if (gear->stab_attack > attack_bonus) attack_bonus = gear->stab_attack;
                if (gear->crush_attack > attack_bonus) attack_bonus = gear->crush_attack;
                break;
            default:
                break;
        }
        if (attack_bonus < 0) {
            tick_shaping += cfg->gear_mismatch_penalty;
        }
    }

    int ate_food = p->ate_food_this_tick;
    int ate_karam = p->ate_karambwan_this_tick;
    int ate_brew = p->ate_brew_this_tick;

    if (ate_food || ate_karam) {
        float hp_before = p->prev_hp_percent;
        if (hp_before > cfg->premature_eat_threshold) {
            tick_shaping += cfg->premature_eat_penalty;
        }
        float max_heal;
        if (ate_food) {
            max_heal = 20.0f / base_hp;
        } else {
            max_heal = 18.0f / base_hp;
        }
        float wasted = hp_before + max_heal - 1.0f;
        if (wasted > 0.0f) {
            float wasted_hp = wasted * base_hp;
            tick_shaping += cfg->wasted_eat_penalty * wasted_hp;
        }
    }

    if (ate_food && ate_brew && ate_karam) {
        float hp_before = p->prev_hp_percent;
        float hp_threshold = 45.0f / base_hp;
        if (hp_before <= hp_threshold) {
            tick_shaping += cfg->smart_triple_eat_bonus;
        } else {
            float food_brew_heal = (20.0f + 16.0f) / base_hp;
            float hp_after_food_brew = hp_before + food_brew_heal;
            if (hp_after_food_brew > 1.0f) hp_after_food_brew = 1.0f;
            float missing_after = 1.0f - hp_after_food_brew;
            float karam_heal_norm = 18.0f / base_hp;
            float wasted_karam = karam_heal_norm - missing_after;
            if (wasted_karam > 0.0f) {
                float wasted_karam_hp = wasted_karam * base_hp;
                tick_shaping += cfg->wasted_triple_eat_penalty * wasted_karam_hp;
            }
        }
    }

    reward += tick_shaping * cfg->shaping_scale;

    return reward;
}

#endif // OSRS_PVP_ACTIONS_H
