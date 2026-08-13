#ifndef OSRS_PVP_OBSERVATIONS_H
#define OSRS_PVP_OBSERVATIONS_H

#include <string.h>
#include "osrs_types.h"
#include "osrs_player_consumables.h"
#include "osrs_pvp_gear.h"
#include "osrs_pvp_combat.h"
#include "osrs_encounter.h"
#include "osrs_pvp_movement.h"

static inline float get_relative_level_combat(int current, int base) {
    int max_level = base + osrs_super_combat_boost_amount(base);
    return (float)current / (float)max_level;
}

static inline float get_relative_level_ranged(int current, int base) {
    int max_level = base + osrs_ranging_boost_amount(base);
    return (float)current / (float)max_level;
}

static inline float get_relative_level_magic(int current, int base) {
    return (float)current / (float)base;
}

static inline int can_use_brew_boost(Player* p) {
    int def_boost = osrs_brew_defence_boost_amount(p->base_defence);
    int def_cap = p->is_lms ? p->base_defence : p->base_defence + def_boost;
    if (p->current_defence < def_cap - 1) {
        return 1;
    }
    return p->current_hitpoints <= p->base_hitpoints;
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
    if (remaining_ticks(p->potion_timer) > 0) {
        return 0;
    }
    switch (potion_type) {
        case 1: return p->brew_doses > 0;
        case 2: return p->restore_doses > 0;
        case 3: return p->combat_potion_doses > 0;
        case 4: return p->ranged_potion_doses > 0;
        default: return 0;
    }
}

static inline int can_eat_food(Player* p) {
    return osrs_player_can_eat_food_type(p, FOOD_SHARK);
}

static inline int can_eat_karambwan(Player* p) {
    return osrs_player_can_eat_food_type(p, FOOD_KARAMBWAN);
}

static inline int can_move_adjacent(Player* p, const CollisionMap* cmap) {
    int dest_x = 0;
    int dest_y = 0;
    if (!select_closest_adjacent_tile(p, p->last_obs_target_x, p->last_obs_target_y, &dest_x, &dest_y, cmap)) {
        return 0;
    }
    return !(dest_x == p->x && dest_y == p->y);
}

static inline int can_move_under(Player* p, Player* target) {
    int dist = chebyshev_distance(p->x, p->y, p->last_obs_target_x, p->last_obs_target_y);
    return remaining_ticks(target->frozen_ticks) > 0 && dist != 0;
}

static inline int can_move_to_farcast(Player* p, int distance, const CollisionMap* cmap) {
    int dest_x = 0;
    int dest_y = 0;
    if (!select_farcast_tile(p, p->last_obs_target_x, p->last_obs_target_y, distance, &dest_x, &dest_y, cmap)) {
        return 0;
    }
    return !(dest_x == p->x && dest_y == p->y);
}

static inline int can_move_diagonal(Player* p, const CollisionMap* cmap) {
    int dest_x = 0;
    int dest_y = 0;
    if (!select_closest_diagonal_tile(p, p->last_obs_target_x, p->last_obs_target_y, &dest_x, &dest_y, cmap)) {
        return 0;
    }
    return !(dest_x == p->x && dest_y == p->y);
}

static void init_obs_norm_divisors(float* d) {
    for (int i = 0; i < SLOT_NUM_OBSERVATIONS; i++) d[i] = 1.0f;

    d[4] = 100.0f;    d[21] = 100.0f;
    d[22] = 10.0f;    d[23] = 10.0f;    d[24] = 16.0f;    d[25] = 20.0f;    d[26] = 15.0f;    d[27] = 4.0f;
    d[29] = 32.0f;    d[30] = 32.0f;    d[31] = 32.0f;    d[32] = 32.0f;
    d[39] = 6.0f;    d[40] = 3.0f;    d[41] = 3.0f;    d[42] = 3.0f;    d[43] = 4.0f;    d[44] = 6.0f;    d[45] = 3.0f;
    d[46] = 2.0f;

    d[47] = 6.0f;
    d[48] = 6.0f;

    d[65] = 2.0f;

    d[60] = 7.0f;
    d[61] = 7.0f;
    d[62] = 7.0f;

    for (int i = 96; i <= 102; i++) d[i] = 99.0f;

    d[106] = 4.0f;

    for (int i = 119; i <= 132; i++) d[i] = 170.0f;
    d[123] = 6.0f;    d[124] = 10.0f;
    d[139] = 50.0f;
    d[140] = 50.0f;
}

static float OBS_NORM_DIVISORS[SLOT_NUM_OBSERVATIONS];
static int _obs_norm_initialized = 0;

static void ensure_obs_norm_initialized(void) {
    if (!_obs_norm_initialized) {
        init_obs_norm_divisors(OBS_NORM_DIVISORS);
        _obs_norm_initialized = 1;
    }
}

static void ocean_write_obs_agent(OsrsEnv* env, float* dst, int agent_idx) {
    ensure_obs_norm_initialized();
    float* src = env->observations + agent_idx * SLOT_NUM_OBSERVATIONS;
    for (int i = 0; i < SLOT_NUM_OBSERVATIONS; i++) {
        dst[i] = src[i] / OBS_NORM_DIVISORS[i];
    }

    unsigned char* mask = env->action_masks + agent_idx * ACTION_MASK_SIZE;
    for (int i = 0; i < ACTION_MASK_SIZE; i++) {
        dst[SLOT_NUM_OBSERVATIONS + i] = (float)mask[i];
    }
}

/** Binding-facing output layout: [normalized obs, action mask as float]. */
static void ocean_write_obs(OsrsEnv* env) {
    ocean_write_obs_agent(env, env->ocean_io.agent_obs, 0);
}

static void ocean_write_obs_p1(OsrsEnv* env) {
    ocean_write_obs_agent(env, env->ocean_io.agent_obs_p1, 1);
}

static void generate_slot_observations(OsrsEnv* env, int agent_idx) {
    Player* p = &env->players[agent_idx];
    Player* t = &env->players[1 - agent_idx];

    float* obs = env->observations + agent_idx * SLOT_NUM_OBSERVATIONS;

    p->last_obs_target_x = t->x;
    p->last_obs_target_y = t->y;

    obs[0] = (p->visible_gear == GEAR_MELEE) ? 1.0f : 0.0f;
    obs[1] = (p->visible_gear == GEAR_RANGED) ? 1.0f : 0.0f;
    obs[2] = (p->visible_gear == GEAR_MAGE) ? 1.0f : 0.0f;
    obs[3] = (float)p->spec_armed;
    obs[4] = (float)p->special_energy;

    obs[5] = (p->prayer == PRAYER_PROTECT_MELEE) ? 1.0f : 0.0f;
    obs[6] = (p->prayer == PRAYER_PROTECT_RANGED) ? 1.0f : 0.0f;
    obs[7] = (p->prayer == PRAYER_PROTECT_MAGIC) ? 1.0f : 0.0f;
    obs[8] = (p->prayer == PRAYER_SMITE) ? 1.0f : 0.0f;
    obs[9] = (p->prayer == PRAYER_REDEMPTION) ? 1.0f : 0.0f;

    obs[10] = (float)p->current_hitpoints / (float)p->base_hitpoints;
    obs[11] = p->last_target_health_percent;

    obs[12] = (t->last_attack_style == ATTACK_STYLE_MELEE) ? 1.0f : 0.0f;
    obs[13] = (t->last_attack_style == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
    obs[14] = (t->last_attack_style == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
    obs[15] = (t->last_attack_style == ATTACK_STYLE_NONE) ? 1.0f : 0.0f;

    obs[16] = (t->prayer == PRAYER_PROTECT_MELEE) ? 1.0f : 0.0f;
    obs[17] = (t->prayer == PRAYER_PROTECT_RANGED) ? 1.0f : 0.0f;
    obs[18] = (t->prayer == PRAYER_PROTECT_MAGIC) ? 1.0f : 0.0f;
    obs[19] = (t->prayer == PRAYER_SMITE) ? 1.0f : 0.0f;
    obs[20] = (t->prayer == PRAYER_REDEMPTION) ? 1.0f : 0.0f;
    obs[21] = (float)t->special_energy;

    obs[22] = (float)p->ranged_potion_doses;
    obs[23] = (float)p->combat_potion_doses;
    obs[24] = (float)p->restore_doses;
    obs[25] = (float)p->brew_doses;
    obs[26] = (float)p->food_count;
    obs[27] = (float)p->karambwan_count;
    obs[28] = (float)p->current_prayer / (float)p->base_prayer;

    obs[29] = (float)remaining_ticks(p->frozen_ticks);
    obs[30] = (float)remaining_ticks(t->frozen_ticks);
    obs[31] = (float)remaining_ticks(p->freeze_immunity_ticks);
    obs[32] = (float)remaining_ticks(t->freeze_immunity_ticks);

    obs[33] = is_in_melee_range(p, t) ? 1.0f : 0.0f;

    obs[34] = get_relative_level_combat(p->current_strength, p->base_strength);
    obs[35] = get_relative_level_combat(p->current_attack, p->base_attack);
    obs[36] = get_relative_level_combat(p->current_defence, p->base_defence);
    obs[37] = get_relative_level_ranged(p->current_ranged, p->base_ranged);
    obs[38] = get_relative_level_magic(p->current_magic, p->base_magic);

    obs[39] = (float)p->attack_timer;
    obs[40] = (float)remaining_ticks(p->food_timer);
    obs[41] = (float)remaining_ticks(p->potion_timer);
    obs[42] = (float)remaining_ticks(p->karambwan_timer);

    int attack_delay = get_attack_timer_uncapped(p) - 1;
    if (attack_delay < -3) attack_delay = -3;
    else if (attack_delay > 0) attack_delay = 0;
    obs[43] = (float)(attack_delay + 3);

    obs[44] = (float)remaining_ticks(t->attack_timer);
    obs[45] = (float)remaining_ticks(t->food_timer);

    int pending_damage = 0;
    for (int i = 0; i < p->num_pending_hits; i++) {
        pending_damage += p->pending_hits[i].damage;
    }
    obs[46] = (float)pending_damage / (float)t->base_hitpoints;

    int ticks_until_hit_on_target = get_ticks_until_next_hit(p);
    int ticks_until_hit_on_player = get_ticks_until_next_hit(t);
    obs[47] = (float)ticks_until_hit_on_target;
    obs[48] = (float)ticks_until_hit_on_player;

    obs[49] = p->just_attacked ? 1.0f : 0.0f;
    obs[50] = t->just_attacked ? 1.0f : 0.0f;

    obs[51] = p->tick_damage_scale;
    obs[52] = p->damage_received_scale;
    obs[53] = p->damage_dealt_scale;

    obs[54] = (p->last_attack_style != ATTACK_STYLE_NONE) ? 1.0f : 0.0f;
    obs[55] = p->is_moving ? 1.0f : 0.0f;
    obs[56] = t->is_moving ? 1.0f : 0.0f;

    obs[57] = (agent_idx == env->pid_holder) ? 1.0f : 0.0f;

    obs[58] = (!p->is_lunar_spellbook && p->current_magic >= 94) ? 1.0f : 0.0f;
    obs[59] = (!p->is_lunar_spellbook && p->current_magic >= 92) ? 1.0f : 0.0f;

    int dist = chebyshev_distance(p->x, p->y, t->x, t->y);
    int destination_distance = p->is_moving
        ? chebyshev_distance(p->dest_x, p->dest_y, t->x, t->y) : dist;
    int distance_to_destination = p->is_moving
        ? chebyshev_distance(p->x, p->y, p->dest_x, p->dest_y) : 0;

    if (destination_distance > 7) destination_distance = 7;
    if (distance_to_destination > 7) distance_to_destination = 7;
    if (dist > 7) dist = 7;

    obs[60] = (float)destination_distance;
    obs[61] = (float)distance_to_destination;
    obs[62] = (float)dist;

    obs[63] = p->player_prayed_correct ? 1.0f : 0.0f;
    obs[64] = p->target_prayed_correct ? 1.0f : 0.0f;

    float damage_scale = (p->total_damage_dealt + 1.0f) / (p->total_damage_received + 1.0f);
    obs[65] = clampf(damage_scale, 0.5f, 2.0f);

    obs[66] = confidence_scale(p->total_target_hit_count);
    obs[67] = ratio_or_zero(p->target_hit_melee_count, p->total_target_hit_count);
    obs[68] = ratio_or_zero(p->target_hit_magic_count, p->total_target_hit_count);
    obs[69] = ratio_or_zero(p->target_hit_ranged_count, p->total_target_hit_count);
    obs[70] = ratio_or_zero(p->player_hit_melee_count, p->total_target_pray_count);
    obs[71] = ratio_or_zero(p->player_hit_magic_count, p->total_target_pray_count);
    obs[72] = ratio_or_zero(p->player_hit_ranged_count, p->total_target_pray_count);
    obs[73] = ratio_or_zero(p->target_hit_correct_count, p->total_target_hit_count);
    obs[74] = confidence_scale(p->total_target_pray_count);
    obs[75] = ratio_or_zero(p->target_pray_magic_count, p->total_target_pray_count);
    obs[76] = ratio_or_zero(p->target_pray_ranged_count, p->total_target_pray_count);
    obs[77] = ratio_or_zero(p->target_pray_melee_count, p->total_target_pray_count);
    obs[78] = ratio_or_zero(p->player_pray_magic_count, p->total_target_hit_count);
    obs[79] = ratio_or_zero(p->player_pray_ranged_count, p->total_target_hit_count);
    obs[80] = ratio_or_zero(p->player_pray_melee_count, p->total_target_hit_count);
    obs[81] = ratio_or_zero(p->target_pray_correct_count, p->total_target_pray_count);

    int recent_target_hit_melee = 0, recent_target_hit_magic = 0, recent_target_hit_ranged = 0;
    int recent_player_hit_melee = 0, recent_player_hit_magic = 0, recent_player_hit_ranged = 0;
    int recent_target_pray_magic = 0, recent_target_pray_ranged = 0, recent_target_pray_melee = 0;
    int recent_player_pray_magic = 0, recent_player_pray_ranged = 0, recent_player_pray_melee = 0;
    int recent_target_hit_correct = 0, recent_target_pray_correct = 0;

    for (int i = 0; i < HISTORY_SIZE; i++) {
        if (p->recent_target_attack_styles[i] == ATTACK_STYLE_MELEE) recent_target_hit_melee++;
        else if (p->recent_target_attack_styles[i] == ATTACK_STYLE_MAGIC) recent_target_hit_magic++;
        else if (p->recent_target_attack_styles[i] == ATTACK_STYLE_RANGED) recent_target_hit_ranged++;

        if (p->recent_player_attack_styles[i] == ATTACK_STYLE_MELEE) recent_player_hit_melee++;
        else if (p->recent_player_attack_styles[i] == ATTACK_STYLE_MAGIC) recent_player_hit_magic++;
        else if (p->recent_player_attack_styles[i] == ATTACK_STYLE_RANGED) recent_player_hit_ranged++;

        if (p->recent_target_prayer_styles[i] == ATTACK_STYLE_MAGIC) recent_target_pray_magic++;
        else if (p->recent_target_prayer_styles[i] == ATTACK_STYLE_RANGED) recent_target_pray_ranged++;
        else if (p->recent_target_prayer_styles[i] == ATTACK_STYLE_MELEE) recent_target_pray_melee++;

        if (p->recent_player_prayer_styles[i] == ATTACK_STYLE_MAGIC) recent_player_pray_magic++;
        else if (p->recent_player_prayer_styles[i] == ATTACK_STYLE_RANGED) recent_player_pray_ranged++;
        else if (p->recent_player_prayer_styles[i] == ATTACK_STYLE_MELEE) recent_player_pray_melee++;

        if (p->recent_target_hit_correct[i]) recent_target_hit_correct++;
        if (p->recent_target_prayer_correct[i]) recent_target_pray_correct++;
    }

    obs[82] = (float)recent_target_hit_melee / (float)HISTORY_SIZE;
    obs[83] = (float)recent_target_hit_magic / (float)HISTORY_SIZE;
    obs[84] = (float)recent_target_hit_ranged / (float)HISTORY_SIZE;
    obs[85] = (float)recent_player_hit_melee / (float)HISTORY_SIZE;
    obs[86] = (float)recent_player_hit_magic / (float)HISTORY_SIZE;
    obs[87] = (float)recent_player_hit_ranged / (float)HISTORY_SIZE;
    obs[88] = (float)recent_target_hit_correct / (float)HISTORY_SIZE;
    obs[89] = (float)recent_target_pray_magic / (float)HISTORY_SIZE;
    obs[90] = (float)recent_target_pray_ranged / (float)HISTORY_SIZE;
    obs[91] = (float)recent_target_pray_melee / (float)HISTORY_SIZE;
    obs[92] = (float)recent_player_pray_magic / (float)HISTORY_SIZE;
    obs[93] = (float)recent_player_pray_ranged / (float)HISTORY_SIZE;
    obs[94] = (float)recent_player_pray_melee / (float)HISTORY_SIZE;
    obs[95] = (float)recent_target_pray_correct / (float)HISTORY_SIZE;

    obs[96] = (float)p->base_attack;
    obs[97] = (float)p->base_strength;
    obs[98] = (float)p->base_defence;
    obs[99] = (float)p->base_ranged;
    obs[100] = (float)p->base_magic;
    obs[101] = (float)p->base_prayer;
    obs[102] = (float)p->base_hitpoints;

    int melee_spec_cost = get_melee_spec_cost(p->melee_spec_weapon);
    obs[103] = (p->melee_spec_weapon == MELEE_SPEC_NONE) ? 0.5f : (float)melee_spec_cost / 100.0f;
    obs[104] = get_melee_spec_str_mult(p->melee_spec_weapon);
    obs[105] = get_melee_spec_acc_mult(p->melee_spec_weapon);

    int melee_hit_count = (p->melee_spec_weapon == MELEE_SPEC_DRAGON_CLAWS) ? 4 :
                          (p->melee_spec_weapon == MELEE_SPEC_DRAGON_DAGGER ||
                           p->melee_spec_weapon == MELEE_SPEC_ABYSSAL_DAGGER) ? 2 : 1;
    obs[106] = (float)melee_hit_count;
    obs[107] = (p->melee_spec_weapon == MELEE_SPEC_VOIDWAKER) ? 1.0f : 0.0f;
    obs[108] = (p->melee_spec_weapon == MELEE_SPEC_DWH ||
                p->melee_spec_weapon == MELEE_SPEC_BGS) ? 1.0f : 0.0f;
    obs[109] = (p->melee_spec_weapon == MELEE_SPEC_GRANITE_MAUL) ? 1.0f : 0.0f;

    int ranged_spec_cost = get_ranged_spec_cost(p->ranged_spec_weapon);
    obs[110] = (p->ranged_spec_weapon == RANGED_SPEC_NONE) ? 0.5f : (float)ranged_spec_cost / 100.0f;
    obs[111] = get_ranged_spec_str_mult(p->ranged_spec_weapon);
    obs[112] = get_ranged_spec_acc_mult(p->ranged_spec_weapon);
    obs[113] = p->bolt_proc_damage;
    obs[114] = p->bolt_ignores_defense ? 1.0f : 0.0f;

    obs[115] = (p->magic_spec_weapon != MAGIC_SPEC_NONE) ? 1.0f : 0.0f;
    obs[116] = (p->ranged_spec_weapon != RANGED_SPEC_NONE) ? 1.0f : 0.0f;
    obs[117] = p->has_blood_fury ? 1.0f : 0.0f;
    osrs_ensure_player_equipment(p);
    obs[118] = (p->equipment_effect_profile.dharok_piece_count >= 4) ? 1.0f : 0.0f;

    GearBonuses* slot_bonuses = get_slot_gear_bonuses(p);
    obs[119] = (float)slot_bonuses->magic_attack;
    obs[120] = (float)slot_bonuses->magic_strength;
    obs[121] = (float)slot_bonuses->ranged_attack;
    obs[122] = (float)slot_bonuses->ranged_strength;
    obs[123] = (float)slot_bonuses->attack_speed;
    obs[124] = (float)slot_bonuses->attack_range;
    obs[125] = (float)slot_bonuses->slash_attack;
    obs[126] = (float)slot_bonuses->melee_strength;
    obs[127] = (float)slot_bonuses->ranged_defence;
    obs[128] = (float)slot_bonuses->magic_defence;
    obs[129] = (float)slot_bonuses->slash_defence;

    GearBonuses* target_bonuses = get_slot_gear_bonuses(t);
    obs[130] = (float)target_bonuses->ranged_defence;
    obs[131] = (float)target_bonuses->magic_defence;
    obs[132] = (float)target_bonuses->slash_defence;

    obs[133] = env->is_lms ? 1.0f : 0.0f;
    obs[134] = env->pvp_runtime.is_pvp_arena ? 1.0f : 0.0f;
    obs[135] = p->veng_active ? 1.0f : 0.0f;
    obs[136] = t->veng_active ? 1.0f : 0.0f;
    obs[137] = p->is_lunar_spellbook ? 1.0f : 0.0f;
    obs[138] = p->observed_target_lunar_spellbook ? 1.0f : 0.0f;
    obs[139] = (float)remaining_ticks(p->veng_cooldown);
    obs[140] = (float)remaining_ticks(t->veng_cooldown);
    obs[141] = is_blood_attack_available(p) ? 1.0f : 0.0f;
    obs[142] = is_ice_attack_available(p) ? 1.0f : 0.0f;
    obs[143] = can_toggle_spec(p) ? 1.0f : 0.0f;
    obs[144] = is_ranged_attack_available(p) ? 1.0f : 0.0f;
    obs[145] = is_ranged_spec_attack_available(p) ? 1.0f : 0.0f;
    obs[146] = is_melee_attack_available(p, t) ? 1.0f : 0.0f;
    obs[147] = is_melee_spec_attack_available(p, t) ? 1.0f : 0.0f;
    obs[148] = (p->brew_doses > 0) ? 0.8f : 0.0f;

    obs[149] = (p->attack_timer <= 0) ? 1.0f : 0.0f;

    obs[150] = (float)p->equipped[GEAR_SLOT_WEAPON] / 63.0f;
    obs[151] = (p->attack_style_this_tick == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
    obs[152] = (p->attack_style_this_tick == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
    obs[153] = (p->attack_style_this_tick == ATTACK_STYLE_MELEE) ? 1.0f : 0.0f;

    AttackStyle target_style = get_slot_weapon_attack_style(t);
    obs[154] = (target_style == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
    obs[155] = (target_style == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
    obs[156] = (target_style == ATTACK_STYLE_MELEE) ? 1.0f : 0.0f;

    obs[157] = (p->offensive_prayer == OFFENSIVE_PRAYER_PIETY) ? 1.0f : 0.0f;
    obs[158] = (p->offensive_prayer == OFFENSIVE_PRAYER_RIGOUR) ? 1.0f : 0.0f;
    obs[159] = (p->offensive_prayer == OFFENSIVE_PRAYER_AUGURY) ? 1.0f : 0.0f;

    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        obs[160 + slot] = (float)p->equipped[slot] / 63.0f;
    }

    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        obs[171 + slot] = (float)t->equipped[slot] / 63.0f;
    }

    uint8_t best_mspec = find_best_melee_spec(p);
    obs[182] = (best_mspec == ITEM_VOIDWAKER) ? 1.0f : 0.0f;

    obs[183] = p->used_special_this_tick ? 1.0f : 0.0f;
    obs[184] = p->ate_food_this_tick ? 1.0f : 0.0f;
    obs[185] = p->ate_karambwan_this_tick ? 1.0f : 0.0f;
    AttackStyle current_weapon_style = get_slot_weapon_attack_style(p);
    obs[186] = (current_weapon_style == ATTACK_STYLE_MAGIC) ? 1.0f : 0.0f;
    obs[187] = (current_weapon_style == ATTACK_STYLE_RANGED) ? 1.0f : 0.0f;
    obs[188] = (current_weapon_style == ATTACK_STYLE_MELEE) ? 1.0f : 0.0f;
    obs[189] = p->ate_brew_this_tick ? 1.0f : 0.0f;

    float wild_w = (float)(WILD_MAX_X - WILD_MIN_X);
    float wild_h = (float)(WILD_MAX_Y - WILD_MIN_Y);
    obs[190] = (float)(p->x - WILD_MIN_X) / wild_w;
    obs[191] = (float)(WILD_MAX_X - p->x) / wild_w;
    obs[192] = (float)(p->y - WILD_MIN_Y) / wild_h;
    obs[193] = (float)(WILD_MAX_Y - p->y) / wild_h;
    float scale = (wild_w > wild_h ? wild_h : wild_w) * 0.5f;
    if (scale < 1.0f) scale = 1.0f;
    obs[194] = clampf((float)(t->x - p->x) / scale, -1.0f, 1.0f);
    obs[195] = clampf((float)(t->y - p->y) / scale, -1.0f, 1.0f);
    const CollisionMap* cmap_obs = (const CollisionMap*)env->collision_map;
    for (int m = 0; m < MOVE_DIM; m++) {
        if (m == 0) {
            obs[196 + m] = 1.0f;
            continue;
        }
        int nx = p->x + ENCOUNTER_MOVE_TARGET_DX[m];
        int ny = p->y + ENCOUNTER_MOVE_TARGET_DY[m];
        obs[196 + m] = pvp_tile_walkable((void*)cmap_obs, nx, ny) ? 1.0f : 0.0f;
    }
}

static void compute_action_masks(OsrsEnv* env, int agent_idx) {
    Player* p = &env->players[agent_idx];
    Player* t = &env->players[1 - agent_idx];

    unsigned char* mask = env->action_masks + agent_idx * ACTION_MASK_SIZE;
    int offset = 0;

    mask[offset + LOADOUT_KEEP] = 1;
    for (int l = LOADOUT_MELEE; l <= LOADOUT_TANK; l++) {
        mask[offset + l] = is_loadout_active(p, l) ? 0 : 1;
    }

    int frozen_no_melee = !can_move(p) && !is_in_melee_range(p, t);

    uint8_t best_melee_spec = find_best_melee_spec(p);
    int melee_spec_cost = 25;
    if (best_melee_spec == ITEM_AGS || best_melee_spec == ITEM_ANCIENT_GS) melee_spec_cost = 50;
    if (best_melee_spec == ITEM_STATIUS_WARHAMMER) melee_spec_cost = 35;
    mask[offset + LOADOUT_SPEC_MELEE] = (best_melee_spec != ITEM_NONE) &&
        (p->special_energy >= melee_spec_cost) && !frozen_no_melee;

    uint8_t best_range_spec = find_best_ranged_spec(p);
    int range_spec_cost = 50;
    mask[offset + LOADOUT_SPEC_RANGE] = (best_range_spec != ITEM_NONE) &&
        (p->special_energy >= range_spec_cost);

    uint8_t best_magic_spec = find_best_magic_spec(p);
    mask[offset + LOADOUT_SPEC_MAGIC] = (best_magic_spec != ITEM_NONE) &&
        (p->special_energy >= 55);

    mask[offset + LOADOUT_GMAUL] = player_has_gmaul(p) &&
        (p->special_energy >= 50) && !frozen_no_melee;

    if (frozen_no_melee) {
        mask[offset + LOADOUT_MELEE] = 0;
    }
    offset += LOADOUT_DIM;

    int attack_ready = remaining_ticks(p->attack_timer) == 0;
    int current_loadout = get_current_loadout(p);
    int in_mage_loadout = (current_loadout == LOADOUT_MAGE);
    int in_tank_loadout = (current_loadout == LOADOUT_TANK);
    int weapon_style = get_slot_weapon_attack_style(p);
    int melee_reachable = (weapon_style == ATTACK_STYLE_MELEE)
        ? (is_in_melee_range(p, t) || can_move(p))
        : 1;
    int can_move_now = can_move(p);
    mask[offset + ATTACK_NONE] = 1;
    mask[offset + ATTACK_ATK] = attack_ready && !in_mage_loadout && !in_tank_loadout &&
                                 weapon_style != ATTACK_STYLE_NONE &&
                                 melee_reachable;
    mask[offset + ATTACK_ICE] = attack_ready && can_cast_ice_spell(p);
    mask[offset + ATTACK_BLOOD] = attack_ready && can_cast_blood_spell(p);
    const CollisionMap* cmap = (const CollisionMap*)env->collision_map;
    mask[offset + MOVE_ADJACENT] = can_move_now && can_move_adjacent(p, cmap);
    mask[offset + MOVE_UNDER] = can_move_now && can_move_under(p, t);
    mask[offset + MOVE_DIAGONAL] = can_move_now && can_move_diagonal(p, cmap);
    mask[offset + MOVE_FARCAST_2] = can_move_now && can_move_to_farcast(p, 2, cmap);
    mask[offset + MOVE_FARCAST_3] = can_move_now && can_move_to_farcast(p, 3, cmap);
    mask[offset + MOVE_FARCAST_4] = can_move_now && can_move_to_farcast(p, 4, cmap);
    mask[offset + MOVE_FARCAST_5] = can_move_now && can_move_to_farcast(p, 5, cmap);
    mask[offset + MOVE_FARCAST_6] = can_move_now && can_move_to_farcast(p, 6, cmap);
    mask[offset + MOVE_FARCAST_7] = can_move_now && can_move_to_farcast(p, 7, cmap);
    offset += COMBAT_DIM;

    int has_prayer = p->current_prayer > 0;
    mask[offset + ENCOUNTER_OVERHEAD_NO_CHANGE] = 1;
    mask[offset + ENCOUNTER_OVERHEAD_OFF] = p->prayer != PRAYER_NONE;
    mask[offset + ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE]      = has_prayer;
    mask[offset + ENCOUNTER_OVERHEAD_SET_REFRESH_RANGED]     = has_prayer;
    mask[offset + ENCOUNTER_OVERHEAD_SET_REFRESH_MAGIC]      = has_prayer;
    mask[offset + ENCOUNTER_OVERHEAD_SET_REFRESH_SMITE]      = has_prayer && !env->is_lms;
    mask[offset + ENCOUNTER_OVERHEAD_SET_REFRESH_REDEMPTION] = has_prayer && !env->is_lms;
    offset += OVERHEAD_DIM;

    mask[offset + FOOD_NONE] = 1;
    mask[offset + FOOD_EAT] = can_eat_food(p);
    offset += FOOD_DIM;

    mask[offset + POTION_NONE] = 1;
    mask[offset + POTION_BREW] = can_use_potion(p, 1) && can_use_brew_boost(p);
    mask[offset + POTION_RESTORE] = can_use_potion(p, 2) && can_restore_stats(p);
    mask[offset + POTION_COMBAT] = can_use_potion(p, 3) && can_boost_combat_skills(p);
    mask[offset + POTION_RANGED] = can_use_potion(p, 4) && can_boost_ranged(p);
    offset += POTION_DIM;

    mask[offset + KARAM_NONE] = 1;
    mask[offset + KARAM_EAT] = can_eat_karambwan(p);
    offset += KARAMBWAN_DIM;

    mask[offset + VENG_NONE] = 1;
    mask[offset + VENG_CAST] = !env->is_lms && p->is_lunar_spellbook && !p->veng_active &&
                                (remaining_ticks(p->veng_cooldown) == 0) && p->current_magic >= 94;
    offset += VENG_DIM;

    mask[offset + ENCOUNTER_OFFENSIVE_NO_CHANGE] = 1;
    mask[offset + ENCOUNTER_OFFENSIVE_OFF] = p->offensive_prayer != OFFENSIVE_PRAYER_NONE;
    mask[offset + ENCOUNTER_OFFENSIVE_SET_REFRESH_PIETY]  = has_prayer;
    mask[offset + ENCOUNTER_OFFENSIVE_SET_REFRESH_RIGOUR] = has_prayer;
    mask[offset + ENCOUNTER_OFFENSIVE_SET_REFRESH_AUGURY] = has_prayer;
    offset += OFFENSIVE_DIM;

    mask[offset + 0] = 1;
    int can_move_for_move_head = can_move(p);
    for (int m = 1; m < MOVE_DIM; m++) {
        if (!can_move_for_move_head) {
            mask[offset + m] = 0;
            continue;
        }
        int nx = p->x + ENCOUNTER_MOVE_TARGET_DX[m];
        int ny = p->y + ENCOUNTER_MOVE_TARGET_DY[m];
        mask[offset + m] = pvp_tile_walkable((void*)cmap, nx, ny) ? 1 : 0;
    }
    offset += MOVE_DIM;
}

#endif // OSRS_PVP_OBSERVATIONS_H
