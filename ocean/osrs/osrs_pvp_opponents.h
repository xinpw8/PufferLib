#ifndef OSRS_PVP_OPPONENTS_H
#define OSRS_PVP_OPPONENTS_H

#include "osrs_policy.h"
#include "osrs_pvp_actions.h"

#define OPP_STYLE_MAGE    0
#define OPP_STYLE_RANGED  1
#define OPP_STYLE_MELEE   2
#define OPP_STYLE_SPEC    3

static inline PvpEquipmentPlan opp_equipment_plan(int style) {
    if (style == OPP_STYLE_MAGE) return PVP_EQUIPMENT_MAGIC;
    if (style == OPP_STYLE_RANGED) return PVP_EQUIPMENT_RANGED;
    if (style == OPP_STYLE_SPEC) return PVP_EQUIPMENT_SPEC_MELEE;
    return PVP_EQUIPMENT_MELEE;
}

static inline void opp_apply_equipment_plan(
    int* actions,
    const Player* self,
    PvpEquipmentPlan plan
) {
    pvp_emit_equipment_plan_actions(actions, self, plan);
    if (plan == PVP_EQUIPMENT_SPEC_MELEE ||
            plan == PVP_EQUIPMENT_SPEC_RANGED ||
            plan == PVP_EQUIPMENT_SPEC_MAGIC ||
            plan == PVP_EQUIPMENT_GMAUL)
        actions[OSRS_HEAD_SPECIAL] = 1;
}

static inline void opp_apply_gear_switch(
    int* actions,
    const Player* self,
    int style
) {
    opp_apply_equipment_plan(actions, self, opp_equipment_plan(style));
}

static inline int opp_find_consumable_cell(
    const Player* player,
    OsrsConsumableKind kind
) {
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
        const OsrsItemContentMetadata* metadata =
            osrs_inventory_cell_metadata(&player->inventory_cells[cell]);
        if (metadata->consumable_kind == kind) return cell;
    }
    return -1;
}

static inline void opp_emit_consumable(
    int* actions,
    const Player* player,
    OsrsConsumableKind kind
) {
    int cell = opp_find_consumable_cell(player, kind);
    if (cell < 0) return;
    OsrsClickAction click = (OsrsClickAction)osrs_inventory_cell_metadata(
        &player->inventory_cells[cell])->click_action;
    if (click == OSRS_CLICK_EAT) actions[OSRS_HEAD_EAT] = cell + 1;
    else if (click == OSRS_CLICK_DRINK) actions[OSRS_HEAD_DRINK] = cell + 1;
}

static inline void opp_emit_move_toward(
    int* actions,
    const Player* self,
    int destination_x,
    int destination_y
) {
    int dx = clamp(destination_x - self->x, -2, 2);
    int dy = clamp(destination_y - self->y, -2, 2);
    for (int action = 1; action < OSRS_PRIMARY_MOVE_ACTIONS; action++) {
        if (ENCOUNTER_MOVE_TARGET_DX[action] == dx &&
                ENCOUNTER_MOVE_TARGET_DY[action] == dy) {
            actions[OSRS_HEAD_PRIMARY] = action;
            return;
        }
    }
}

static inline void opp_emit_farcast_move(
    int* actions,
    const Player* self,
    const Player* target,
    int distance
) {
    int raw_dx = self->x - target->x;
    int raw_dy = self->y - target->y;
    int dx = clamp(raw_dx, -distance, distance);
    int dy = clamp(raw_dy, -distance, distance);
    int adx = abs_int(dx);
    int ady = abs_int(dy);
    if (adx < distance && ady < distance) {
        if (adx >= ady) dx = raw_dx >= 0 ? distance : -distance;
        else dy = raw_dy >= 0 ? distance : -distance;
    }
    opp_emit_move_toward(actions, self, target->x + dx, target->y + dy);
}

typedef struct {
    int can_food;
    int can_brew;
    int can_karambwan;
    int can_restore;
} OppConsumables;

static inline void opp_tick_cooldowns(OpponentState* opp) {
    if (opp->food_cooldown > 0) opp->food_cooldown--;
    if (opp->potion_cooldown > 0) opp->potion_cooldown--;
    if (opp->karambwan_cooldown > 0) opp->karambwan_cooldown--;
}

static inline OppConsumables opp_get_consumables(OpponentState* opp, Player* self) {
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    OppConsumables c;
    c.can_food = opp->food_cooldown <= 0 && can_eat_food(self) && hp_pct < 1.0f;
    c.can_brew = opp->potion_cooldown <= 0 &&
        pvp_drink_kind_available(self, OSRS_CONSUMABLE_BREW);
    c.can_karambwan = opp->karambwan_cooldown <= 0 &&
        can_eat_karambwan(self) && hp_pct < 1.0f;
    c.can_restore = opp->potion_cooldown <= 0 &&
        pvp_drink_kind_available(self, OSRS_CONSUMABLE_SUPER_RESTORE);
    return c;
}

static inline void opp_emit_preferred_food(
    OpponentState* opp,
    int* actions,
    const Player* self,
    OppConsumables consumables
) {
    if (consumables.can_food) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SHARK_FOOD);
        opp->food_cooldown = 3;
    } else if (consumables.can_karambwan) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_KARAMBWAN);
        opp->karambwan_cooldown = 2;
    }
}

static inline AttackStyle opp_get_gear_style(Player* p) {
    int s = get_item_attack_style(p->equipped[GEAR_SLOT_WEAPON]);
    if (s == 3) return ATTACK_STYLE_MAGIC;
    if (s == 2) return ATTACK_STYLE_RANGED;
    if (s == 1) return ATTACK_STYLE_MELEE;
    return ATTACK_STYLE_MAGIC;
}

static inline int opp_get_defensive_prayer(Player* target) {
    AttackStyle target_style = opp_get_gear_style(target);
    if (target_style == ATTACK_STYLE_MAGIC)  return OVERHEAD_MAGE;
    if (target_style == ATTACK_STYLE_RANGED) return OVERHEAD_RANGED;
    if (target_style == ATTACK_STYLE_MELEE)  return OVERHEAD_MELEE;
    return OVERHEAD_MAGE;
}

static inline int opp_has_prayer_active(Player* self, int prayer_action) {
    if (prayer_action == OVERHEAD_MELEE)  return self->prayer == PRAYER_PROTECT_MELEE;
    if (prayer_action == OVERHEAD_RANGED) return self->prayer == PRAYER_PROTECT_RANGED;
    if (prayer_action == OVERHEAD_MAGE)   return self->prayer == PRAYER_PROTECT_MAGIC;
    return 0;
}

static inline int opp_attack_ready(Player* self) {
    return self->attack_timer <= 0;
}

static inline int opp_can_reach_melee(Player* self, Player* target) {
    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
    return dist <= 1 || (self->frozen_ticks == 0 && dist <= 5);
}

static inline int opp_get_off_prayer_mask(Player* self, Player* target) {
    int mask = 0;
    if (target->prayer != PRAYER_PROTECT_MAGIC)   mask |= (1 << OPP_STYLE_MAGE);
    if (target->prayer != PRAYER_PROTECT_RANGED)  mask |= (1 << OPP_STYLE_RANGED);
    if (target->prayer != PRAYER_PROTECT_MELEE && opp_can_reach_melee(self, target))
        mask |= (1 << OPP_STYLE_MELEE);
    if (mask == 0) mask = (1 << OPP_STYLE_MAGE);
    return mask;
}

static inline int opp_pick_from_mask(OsrsEnv* env, int mask) {
    int choices[3];
    int count = 0;
    for (int i = 0; i < 3; i++) {
        if (mask & (1 << i)) choices[count++] = i;
    }
    return choices[rand_int(env, count)];
}

static inline int opp_is_drained(Player* self) {
    return self->current_strength < self->base_strength ||
           self->current_attack < self->base_attack ||
           self->current_defence < self->base_defence ||
           self->current_ranged < self->base_ranged ||
           self->current_magic < self->base_magic;
}

static inline int opp_should_fc3(Player* self, Player* target) {
    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
    return target->freeze_immunity_ticks > 1 &&
           self->frozen_ticks == 0 &&
           self->attack_timer <= 2 &&
           dist > 3;
}

static inline void opp_update_flee_tracking(OpponentState* opp, Player* self, Player* target) {
    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
    if (dist > opp->prev_dist_to_target && dist > 1) {
        opp->target_fleeing_ticks++;
    } else {
        opp->target_fleeing_ticks = 0;
    }
    opp->prev_dist_to_target = dist;
}

typedef struct { float base; float variance; } RandRange;

typedef struct {
    RandRange prayer_accuracy;
    RandRange off_prayer_rate;
    RandRange offensive_prayer_rate;
    RandRange action_delay_chance;
    RandRange mistake_rate;
    RandRange offensive_prayer_miss;
} OpponentRandRanges;

#define RR(b, v) {(b), (v)}

static const OpponentRandRanges OPP_RAND_RANGES[OPP_RANGE_KITER + 1] = {
    [OPP_NONE]                  = { RR(0,0),      RR(0,0),      RR(0,0),      RR(0,0),       RR(0,0),       RR(0,0) },
    [OPP_TRUE_RANDOM]           = { RR(0.33,0),   RR(0.33,0),   RR(0,0),      RR(0,0),       RR(0,0),       RR(0,0) },
    [OPP_PANICKING]             = { RR(0.33,0.1), RR(0.33,0),   RR(0,0),      RR(0.10,0.05), RR(0,0),       RR(0,0) },
    [OPP_WEAK_RANDOM]           = { RR(0.40,0.1), RR(0.33,0.1), RR(0,0),      RR(0.10,0.05), RR(0.05,0.03), RR(0,0) },
    [OPP_SEMI_RANDOM]           = { RR(0.50,0.1), RR(0.40,0.1), RR(0.05,0.03),RR(0.08,0.04), RR(0.05,0.03), RR(0,0) },
    [OPP_STICKY_PRAYER]         = { RR(0.33,0),   RR(0.33,0),   RR(0,0),      RR(0.10,0.05), RR(0,0),       RR(0,0) },
    [OPP_RANDOM_EATER]          = { RR(0.40,0.1), RR(0.33,0.1), RR(0,0),      RR(0.08,0.04), RR(0.05,0.03), RR(0,0) },
    [OPP_PRAYER_ROOKIE]         = { RR(0.30,0.1), RR(0.20,0.1), RR(0,0),      RR(0.12,0.05), RR(0.08,0.04), RR(0,0) },
    [OPP_IMPROVED]              = { RR(0.95,0.05),RR(0.95,0.05),RR(0.80,0.10),RR(0.05,0.03), RR(0.03,0.02), RR(0.05,0.03) },
    [OPP_MIXED_EASY]            = { RR(0,0),      RR(0,0),      RR(0,0),      RR(0,0),       RR(0,0),       RR(0,0) },
    [OPP_MIXED_MEDIUM]          = { RR(0,0),      RR(0,0),      RR(0,0),      RR(0,0),       RR(0,0),       RR(0,0) },
    [OPP_ONETICK]               = { RR(0.97,0.03),RR(0.97,0.03),RR(0.90,0.05),RR(0.03,0.02), RR(0.02,0.01), RR(0.03,0.02) },
    [OPP_UNPREDICTABLE_IMPROVED]= { RR(0.92,0.05),RR(0.90,0.05),RR(0.75,0.10),RR(0.08,0.04), RR(0.05,0.03), RR(0.08,0.04) },
    [OPP_UNPREDICTABLE_ONETICK] = { RR(0.95,0.03),RR(0.95,0.03),RR(0.85,0.08),RR(0.05,0.03), RR(0.03,0.02), RR(0.05,0.03) },
    [OPP_MIXED_HARD]            = { RR(0,0),      RR(0,0),      RR(0,0),      RR(0,0),       RR(0,0),       RR(0,0) },
    [OPP_MIXED_HARD_BALANCED]   = { RR(0,0),      RR(0,0),      RR(0,0),      RR(0,0),       RR(0,0),       RR(0,0) },
    [OPP_PFSP]                  = { RR(0,0),      RR(0,0),      RR(0,0),      RR(0,0),       RR(0,0),       RR(0,0) },
    [OPP_NOVICE_NH]             = { RR(0.60,0.10),RR(0.10,0.05),RR(0.10,0.05),RR(0.15,0.05), RR(0.10,0.05), RR(0.30,0.10) },
    [OPP_APPRENTICE_NH]         = { RR(0.60,0.10),RR(0.20,0.08),RR(0.20,0.08),RR(0.12,0.05), RR(0.08,0.04), RR(0.30,0.10) },
    [OPP_COMPETENT_NH]          = { RR(0.75,0.08),RR(0.25,0.08),RR(0.25,0.08),RR(0.10,0.04), RR(0.06,0.03), RR(0.20,0.08) },
    [OPP_INTERMEDIATE_NH]       = { RR(0.85,0.05),RR(0.70,0.08),RR(0.50,0.10),RR(0.08,0.04), RR(0.05,0.03), RR(0.20,0.08) },
    [OPP_ADVANCED_NH]           = { RR(0.95,0.05),RR(0.90,0.05),RR(0.75,0.08),RR(0.05,0.03), RR(0.03,0.02), RR(0.10,0.05) },
    [OPP_PROFICIENT_NH]         = { RR(0.95,0.03),RR(0.92,0.04),RR(0.80,0.08),RR(0.04,0.02), RR(0.03,0.02), RR(0.10,0.05) },
    [OPP_EXPERT_NH]             = { RR(0.97,0.03),RR(0.95,0.03),RR(0.85,0.05),RR(0.03,0.02), RR(0.02,0.01), RR(0.10,0.05) },
    [OPP_MASTER_NH]             = { RR(0.98,0.02),RR(0.97,0.03),RR(0.90,0.05),RR(0.02,0.01), RR(0.01,0.01), RR(0.01,0.01) },
    [OPP_SAVANT_NH]             = { RR(0.98,0.02),RR(0.97,0.03),RR(0.90,0.05),RR(0.02,0.01), RR(0.01,0.01), RR(0.01,0.01) },
    [OPP_NIGHTMARE_NH]          = { RR(0.99,0.01),RR(0.98,0.02),RR(0.95,0.03),RR(0.01,0.01), RR(0.005,0.005),RR(0.01,0.01) },
    [OPP_VENG_FIGHTER]          = { RR(0.92,0.05),RR(0.90,0.05),RR(0.85,0.10),RR(0.03,0.02), RR(0.02,0.01), RR(0.05,0.03) },
    [OPP_BLOOD_HEALER]          = { RR(0.90,0.05),RR(0.88,0.05),RR(0.80,0.10),RR(0.05,0.03), RR(0.04,0.02), RR(0.05,0.03) },
    [OPP_GMAUL_COMBO]           = { RR(0.96,0.03),RR(0.95,0.03),RR(0.90,0.05),RR(0.03,0.02), RR(0.02,0.01), RR(0.02,0.01) },
    [OPP_RANGE_KITER]           = { RR(0.93,0.04),RR(0.93,0.04),RR(0.85,0.08),RR(0.04,0.02), RR(0.03,0.02), RR(0.04,0.02) },
};

#undef RR

static inline float rand_range(OsrsEnv* env, RandRange r) {
    float v = r.base + (rand_float(env) * 2.0f - 1.0f) * r.variance;
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

static inline int opp_should_skip_offensive(OsrsEnv* env, OpponentState* opp) {
    return rand_float(env) < opp->action_delay_chance;
}

static inline int opp_pick_off_prayer_style_biased(OsrsEnv* env, OpponentState* opp,
                                                    Player* self, Player* target) {
    int off_mask = opp_get_off_prayer_mask(self, target);
    float weights[3] = {0};
    float total = 0;
    for (int i = 0; i < 3; i++) {
        if (off_mask & (1 << i)) {
            weights[i] = opp->style_bias[i];
            total += weights[i];
        }
    }
    if (total <= 0) return opp_pick_from_mask(env, off_mask);

    float r = rand_float(env) * total;
    float cum = 0;
    for (int i = 0; i < 3; i++) {
        cum += weights[i];
        if (r < cum) return i;
    }
    return opp_pick_from_mask(env, off_mask);
}

static inline int opp_apply_prayer_mistake(OsrsEnv* env, OpponentState* opp, int correct_prayer) {
    if (rand_float(env) < opp->mistake_rate) {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        return prayers[rand_int(env, 3)];
    }
    return correct_prayer;
}

static const float UNPREDICTABLE_IMP_PRAYER_CUM[] = {0.70f, 0.90f, 0.98f, 1.00f};
#define UNPREDICTABLE_IMP_PRAYER_CUM_LEN 4

static const float UNPREDICTABLE_IMP_ACTION_CUM[] = {0.85f, 0.97f, 1.00f};
#define UNPREDICTABLE_IMP_ACTION_CUM_LEN 3

static const float UNPREDICTABLE_OT_PRAYER_CUM[] = {0.80f, 0.95f, 0.99f, 1.00f};
#define UNPREDICTABLE_OT_PRAYER_CUM_LEN 4

static const float UNPREDICTABLE_OT_ACTION_CUM[] = {0.90f, 0.98f, 1.00f};
#define UNPREDICTABLE_OT_ACTION_CUM_LEN 3

#define UNPREDICTABLE_IMP_WRONG_PRAYER      0.05f
#define UNPREDICTABLE_IMP_SUBOPTIMAL_ATTACK 0.03f
#define UNPREDICTABLE_OT_FAKE_FAIL          0.12f
#define UNPREDICTABLE_OT_WRONG_PREDICT      0.08f

static inline int opp_sample_delay(OsrsEnv* env, const float* cum_weights, int num_weights) {
    float r = rand_float(env);
    for (int i = 0; i < num_weights; i++) {
        if (r < cum_weights[i]) return i;
    }
    return num_weights - 1;
}

static inline int opp_get_defensive_prayer_with_spec(Player* target) {
    if (target->visible_gear == GEAR_MELEE)  return OVERHEAD_MELEE;
    if (target->visible_gear == GEAR_RANGED) return OVERHEAD_RANGED;
    if (target->visible_gear == GEAR_MAGE)   return OVERHEAD_MAGE;
    return opp_get_defensive_prayer(target);
}

static inline int opp_get_opponent_prayer_style(Player* target) {
    if (target->prayer == PRAYER_PROTECT_MAGIC)  return OPP_STYLE_MAGE;
    if (target->prayer == PRAYER_PROTECT_RANGED) return OPP_STYLE_RANGED;
    if (target->prayer == PRAYER_PROTECT_MELEE)  return OPP_STYLE_MELEE;
    return -1;
}

static inline int opp_get_target_gear_style(Player* target) {
    return (int)target->visible_gear;
}

static inline int opp_get_mage_attack(Player* self, Player* target) {
    int can_freeze = target->freeze_immunity_ticks <= 1 && target->frozen_ticks == 0;
    if (can_freeze) return 0;
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    return hp_pct > 0.98f ? 0 : 1;
}

static void opp_apply_boost_potion(OsrsEnv* env, OpponentState* opp, int* actions,
                                    Player* self, int attack_style, int potion_used) {
    (void)env;
    if (potion_used) return;
    if (opp->potion_cooldown > 0) return;
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    if (opp_is_drained(self) && hp_pct > 0.90f &&
            pvp_drink_kind_available(self, OSRS_CONSUMABLE_SUPER_RESTORE)) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SUPER_RESTORE);
        opp->potion_cooldown = 3;
        return;
    }

    if (hp_pct <= 0.90f) return;

    if (attack_style == OPP_STYLE_MELEE || attack_style == OPP_STYLE_SPEC) {
        if (self->current_strength <= self->base_strength &&
                pvp_drink_kind_available(self, OSRS_CONSUMABLE_SUPER_COMBAT)) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SUPER_COMBAT);
            opp->potion_cooldown = 3;
        }
    } else if (attack_style == OPP_STYLE_RANGED) {
        if (self->current_ranged <= self->base_ranged &&
                pvp_drink_kind_available(self, OSRS_CONSUMABLE_RANGING)) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_RANGING);
            opp->potion_cooldown = 3;
        }
    }
}

static inline int opp_check_eating_queued(int* actions) {
    return actions[OSRS_HEAD_EAT] != 0;
}

static int opp_apply_consumables(OsrsEnv* env, OpponentState* opp, int* actions,
                                  Player* self, int include_drained_restore) {
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;
    OppConsumables cons = opp_get_consumables(opp, self);
    int potion_used = 0;

    if (hp_pct < opp->eat_triple_threshold && cons.can_brew && (cons.can_food || cons.can_karambwan)) {
        opp_emit_preferred_food(opp, actions, self, cons);
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
        opp->potion_cooldown = 3;
        potion_used = 1;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_brew) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SHARK_FOOD);
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
        opp->food_cooldown = 3;
        opp->potion_cooldown = 3;
        potion_used = 1;
    } else if (hp_pct < opp->eat_double_threshold && (cons.can_food || cons.can_karambwan)) {
        opp_emit_preferred_food(opp, actions, self, cons);
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
        opp->potion_cooldown = 3;
        potion_used = 1;
    } else if (hp_pct < 0.60f && cons.can_food) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SHARK_FOOD);
        opp->food_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_karambwan) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_KARAMBWAN);
        opp->karambwan_cooldown = 2;
    } else if (opp_is_drained(self) && hp_pct < 0.90f && cons.can_brew) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
        opp->potion_cooldown = 3;
        potion_used = 1;
    } else if (prayer_pct < 0.30f && cons.can_restore) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SUPER_RESTORE);
        opp->potion_cooldown = 3;
    } else if (include_drained_restore && opp_is_drained(self) && cons.can_restore) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SUPER_RESTORE);
        opp->potion_cooldown = 3;
    }

    (void)env;
    return potion_used;
}

static inline int opp_set_refresh_for_prayer(OverheadPrayer p) {
    switch (p) {
        case PRAYER_PROTECT_MAGIC:  return ENCOUNTER_OVERHEAD_SET_REFRESH_MAGIC;
        case PRAYER_PROTECT_RANGED: return ENCOUNTER_OVERHEAD_SET_REFRESH_RANGED;
        case PRAYER_PROTECT_MELEE:  return ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE;
        case PRAYER_SMITE:          return ENCOUNTER_OVERHEAD_SET_REFRESH_SMITE;
        case PRAYER_REDEMPTION:     return ENCOUNTER_OVERHEAD_SET_REFRESH_REDEMPTION;
        default:                    return ENCOUNTER_OVERHEAD_NO_CHANGE;
    }
}

static inline void opp_emit_prayer(int* actions, Player* self, int target_overhead_action) {
    OverheadPrayer target_prayer;
    switch (target_overhead_action) {
        case OVERHEAD_NONE:       target_prayer = PRAYER_NONE;           break;
        case OVERHEAD_MAGE:       target_prayer = PRAYER_PROTECT_MAGIC;  break;
        case OVERHEAD_RANGED:     target_prayer = PRAYER_PROTECT_RANGED; break;
        case OVERHEAD_MELEE:      target_prayer = PRAYER_PROTECT_MELEE;  break;
        case OVERHEAD_SMITE:      target_prayer = PRAYER_SMITE;          break;
        case OVERHEAD_REDEMPTION: target_prayer = PRAYER_REDEMPTION;     break;
        default: return;
    }
    if (self->prayer == target_prayer) return;
    actions[OSRS_HEAD_OVERHEAD] = (target_prayer == PRAYER_NONE)
        ? ENCOUNTER_OVERHEAD_OFF
        : opp_set_refresh_for_prayer(target_prayer);
}

static inline int opp_process_pending_prayer(OpponentState* opp, int* actions, Player* self) {
    if (opp->pending_prayer_value == 0) return 0;
    if (opp->pending_prayer_delay > 0) {
        opp->pending_prayer_delay--;
        if (opp->pending_prayer_delay > 0) return 0;
    }
    OverheadPrayer target_prayer = PRAYER_NONE;
    int action = ENCOUNTER_OVERHEAD_NO_CHANGE;
    switch (opp->pending_prayer_value) {
        case OVERHEAD_MAGE:       target_prayer = PRAYER_PROTECT_MAGIC;  action = ENCOUNTER_OVERHEAD_SET_REFRESH_MAGIC; break;
        case OVERHEAD_RANGED:     target_prayer = PRAYER_PROTECT_RANGED; action = ENCOUNTER_OVERHEAD_SET_REFRESH_RANGED; break;
        case OVERHEAD_MELEE:      target_prayer = PRAYER_PROTECT_MELEE;  action = ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE; break;
        case OVERHEAD_SMITE:      target_prayer = PRAYER_SMITE;          action = ENCOUNTER_OVERHEAD_SET_REFRESH_SMITE; break;
        case OVERHEAD_REDEMPTION: target_prayer = PRAYER_REDEMPTION;     action = ENCOUNTER_OVERHEAD_SET_REFRESH_REDEMPTION; break;
        default: break;
    }
    if (self->prayer != target_prayer) actions[OSRS_HEAD_OVERHEAD] = action;
    opp->pending_prayer_value = 0;
    return 1;
}

static void opp_handle_delayed_prayer(OsrsEnv* env, OpponentState* opp, int* actions,
                                       Player* self, Player* target,
                                       const float* cum_weights, int cum_len,
                                       float wrong_prayer_prob, int include_spec) {
    int target_style = opp_get_target_gear_style(target);
    if (target_style != opp->last_target_gear_style) {
        opp->last_target_gear_style = target_style;

        int needed_prayer = include_spec
            ? opp_get_defensive_prayer_with_spec(target)
            : opp_get_defensive_prayer(target);

        int needs_switch = !opp_has_prayer_active(self, needed_prayer);

        if (needs_switch) {
            if (rand_float(env) < wrong_prayer_prob) {
                int wrong_options[2];
                int wcount = 0;
                int all_prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
                for (int i = 0; i < 3; i++) {
                    if (all_prayers[i] != needed_prayer)
                        wrong_options[wcount++] = all_prayers[i];
                }
                needed_prayer = wrong_options[rand_int(env, wcount)];
            }

            int delay = opp_sample_delay(env, cum_weights, cum_len);
            opp->pending_prayer_value = needed_prayer;
            opp->pending_prayer_delay = delay;
        }
    }

    opp_process_pending_prayer(opp, actions, self);
}

static void opp_apply_defensive_prayer(OsrsEnv* env, OpponentState* opp, int* actions,
                                        Player* self, Player* target, int use_spec_detect) {
    int def_prayer;
    if (rand_float(env) < opp->prayer_accuracy) {
        def_prayer = use_spec_detect
            ? opp_get_defensive_prayer_with_spec(target)
            : opp_get_defensive_prayer(target);
    } else {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        def_prayer = prayers[rand_int(env, 3)];
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }
}

static inline void opp_offensive_prayer_noop_roll(OsrsEnv* env) {
    /* dead roll from an unimplemented offensive-prayer branch: removing it shifts
       every later draw in the opponent RNG stream */
    (void)rand_float(env);
}

static inline void opp_emit_attack(int* actions, int actual_attack) {
    actions[OSRS_HEAD_PRIMARY] = OSRS_PRIMARY_MOVE_ACTIONS;
    if (actual_attack == 0)
        actions[OSRS_HEAD_SPELL] = OSRS_SPELL_ICE_BARRAGE;
    else if (actual_attack == 1)
        actions[OSRS_HEAD_SPELL] = OSRS_SPELL_BLOOD_BARRAGE;
    else if (actual_attack == 3)
        actions[OSRS_HEAD_SPECIAL] = 1;
}

static void opp_move_when_waiting(OsrsEnv* env, OpponentState* opp, int* actions,
                                   Player* self, Player* target, float under_prob) {
    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
    if (target->frozen_ticks > 0 && self->frozen_ticks == 0 && dist > 0 &&
        (under_prob >= 1.0f || (under_prob > 0.0f && rand_float(env) < under_prob))) {
        opp_emit_move_toward(actions, self, target->x, target->y);
    } else if (opp->target_fleeing_ticks >= 2 && dist > 3 && self->frozen_ticks == 0) {
        opp_emit_farcast_move(actions, self, target, 3);
    } else if (opp_should_fc3(self, target) && target->prayer != PRAYER_PROTECT_MELEE) {
        opp_emit_farcast_move(actions, self, target, 3);
    }
}

static void opp_attack_random_style(OsrsEnv* env, int* actions) {
    Player* self = &env->players[1];
    int style = rand_int(env, 3);
    opp_apply_gear_switch(actions, self, style);
    if (style == OPP_STYLE_MAGE) {
        opp_emit_attack(actions, rand_int(env, 2) == 0 ? 0 : 1);
    } else {
        opp_emit_attack(actions, 2);
    }
}

static void opp_attack_random_style_with_spec(OsrsEnv* env, Player* self, int* actions) {
    int style = rand_int(env, 3);
    if (self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
        rand_float(env) < 0.30f) {
        opp_apply_gear_switch(actions, self, OPP_STYLE_SPEC);
        opp_emit_attack(actions, 2);
    } else {
        opp_apply_gear_switch(actions, self, style);
        if (style == OPP_STYLE_MAGE) {
            opp_emit_attack(actions, rand_int(env, 2) == 0 ? 0 : 1);
        } else {
            opp_emit_attack(actions, 2);
        }
    }
}

typedef struct { int melee; int ranged; int magic; } OppSpecPlan;

static OppSpecPlan opp_plan_specs(Player* self, Player* target, int dist) {
    int can_melee_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
    float target_hp_pct = (float)target->current_hitpoints / (float)target->base_hitpoints;
    uint8_t ranged_spec = find_best_ranged_spec(self);
    uint8_t magic_spec = find_best_magic_spec(self);
    int has_ranged_or_magic = (ranged_spec != ITEM_NONE || magic_spec != ITEM_NONE);

    OppSpecPlan plan;
    plan.melee = opp_attack_ready(self) &&
        self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
        target->prayer != PRAYER_PROTECT_MELEE &&
        can_melee_range &&
        (!has_ranged_or_magic || target_hp_pct < 0.55f);
    plan.ranged = opp_attack_ready(self) && ranged_spec != ITEM_NONE &&
        self->special_energy >= get_ranged_spec_cost(self->ranged_spec_weapon) &&
        target->prayer != PRAYER_PROTECT_RANGED &&
        target_hp_pct < 0.55f;
    plan.magic = opp_attack_ready(self) && magic_spec != ITEM_NONE &&
        self->special_energy >= get_magic_spec_cost(self->magic_spec_weapon) &&
        target->prayer != PRAYER_PROTECT_MAGIC &&
        target_hp_pct < 0.55f;
    return plan;
}

static int opp_try_fake_switch(OsrsEnv* env, OpponentState* opp, int* actions,
                                Player* self, Player* target, int off_mask,
                                float fail_prob) {
    if (opp->fake_switch_pending && opp_attack_ready(self)) {
        opp->fake_switch_pending = 0;
        opp->fake_switch_style = -1;
        if (fail_prob >= 0.0f) opp->fake_switch_failed = 0;
        return 0;
    }
    if (opp_attack_ready(self) || opp->fake_switch_pending) return 0;
    if (rand_float(env) >= 0.30f) return 0;

    int current_style = (int)self->current_gear;
    int can_fake_melee = self->frozen_ticks <= 10 ||
                         chebyshev_distance(self->x, self->y, target->x, target->y) <= 1;

    int fake_options[3];
    int fake_count = 0;
    for (int s = 0; s < 3; s++) {
        if (!(off_mask & (1 << s))) continue;
        if (s == current_style) continue;
        if (s == OPP_STYLE_MELEE && !can_fake_melee) continue;
        fake_options[fake_count++] = s;
    }

    if (fake_count == 0) return 0;

    opp->fake_switch_pending = 1;
    opp->fake_switch_style = fake_options[rand_int(env, fake_count)];
    opp->opponent_prayer_at_fake = opp_get_opponent_prayer_style(target);
    if (fail_prob >= 0.0f)
        opp->fake_switch_failed = (rand_float(env) < fail_prob) ? 1 : 0;

    opp_apply_gear_switch(actions, self, opp->fake_switch_style);

    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
    if (target->frozen_ticks > 0 && self->frozen_ticks == 0 && dist > 0) {
        opp_emit_move_toward(actions, self, target->x, target->y);
    }
    return 1;
}

static void opp_true_random(OsrsEnv* env, int* actions) {
    static const int action_head_dims[OSRS_BASE_NUM_ACTION_HEADS] =
        OSRS_BASE_ACTION_DIMS_INIT(1);
    for (int head = 0; head < OSRS_BASE_NUM_ACTION_HEADS; head++)
        actions[head] = rand_int(env, action_head_dims[head]);
}

static void opp_panicking(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    if (!opp_has_prayer_active(self, opp->chosen_prayer)) {
        opp_emit_prayer(actions, self, opp->chosen_prayer);
    }

    int eating = 0;
    if (hp_pct < 0.25f) {
        if (cons.can_food) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SHARK_FOOD);
            opp->food_cooldown = 3;
            eating = 1;
        }
        if (cons.can_brew) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
            opp->potion_cooldown = 3;
        }
    }

    if (opp_should_skip_offensive(env, opp)) return;

    if (opp_attack_ready(self) && !eating && rand_float(env) < 0.30f) {
        opp_apply_gear_switch(actions, self, opp->chosen_style);

        if (opp->chosen_style == OPP_STYLE_MAGE) {
            int spell = rand_int(env, 2) == 0 ? 0 : 1;
            opp_emit_attack(actions, spell);
        } else {
            opp_emit_attack(actions, 2);
        }
    }
}

static void opp_weak_random(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    int prayers[] = {OVERHEAD_NONE, OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
    opp_emit_prayer(actions, self, prayers[rand_int(env, 4)]);

    int eating = 0;
    if (hp_pct < 0.30f && rand_float(env) > 0.50f) {
        if (cons.can_food) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SHARK_FOOD);
            opp->food_cooldown = 3;
            eating = 1;
        } else if (cons.can_brew) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
            opp->potion_cooldown = 3;
            eating = 1;
        }
    }

    if (opp_should_skip_offensive(env, opp)) return;

    if (opp_attack_ready(self) && !eating) {
        opp_attack_random_style(env, actions);
    }
}

static void opp_semi_random(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
    opp_emit_prayer(actions, self, prayers[rand_int(env, 3)]);

    int eating = 0;
    if (hp_pct < 0.30f) {
        if (cons.can_food) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SHARK_FOOD);
            opp->food_cooldown = 3;
            eating = 1;
        } else if (cons.can_brew) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
            opp->potion_cooldown = 3;
            eating = 1;
        }
    }

    if (opp_should_skip_offensive(env, opp)) return;

    if (opp_attack_ready(self) && !eating) {
        opp_attack_random_style(env, actions);
    }
}

static void opp_sticky_prayer(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    if (!opp->current_prayer_set || rand_float(env) < 0.08f) {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        opp->current_prayer = prayers[rand_int(env, 3)];
        opp->current_prayer_set = 1;
    }
    if (!opp_has_prayer_active(self, opp->current_prayer)) {
        opp_emit_prayer(actions, self, opp->current_prayer);
    }

    int eating = 0;
    if (hp_pct < 0.30f) {
        if (cons.can_food) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SHARK_FOOD);
            opp->food_cooldown = 3;
            eating = 1;
        } else if (cons.can_brew) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
            opp->potion_cooldown = 3;
            eating = 1;
        }
    }

    if (opp_should_skip_offensive(env, opp)) return;

    if (opp_attack_ready(self) && !eating) {
        opp_attack_random_style(env, actions);
    }
}

static void opp_random_eater(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    if (!opp->current_prayer_set || rand_float(env) < 0.08f) {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        opp->current_prayer = prayers[rand_int(env, 3)];
        opp->current_prayer_set = 1;
    }
    if (!opp_has_prayer_active(self, opp->current_prayer)) {
        opp_emit_prayer(actions, self, opp->current_prayer);
    }

    int potion_used = 0;
    if (hp_pct < 0.35f) {
        opp_emit_preferred_food(opp, actions, self, cons);
        if (cons.can_brew) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
            opp->potion_cooldown = 3;
            potion_used = 1;
        }
    } else if (hp_pct < 0.55f) {
        if (cons.can_food) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SHARK_FOOD);
            opp->food_cooldown = 3;
        } else if (cons.can_brew) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
            opp->potion_cooldown = 3;
            potion_used = 1;
        }
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
        opp->potion_cooldown = 3;
        potion_used = 1;
    }

    if (!potion_used && prayer_pct < 0.30f && cons.can_restore) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SUPER_RESTORE);
        opp->potion_cooldown = 3;
    }

    int eating = opp_check_eating_queued(actions);

    if (opp_should_skip_offensive(env, opp)) return;

    if (opp_attack_ready(self) && !eating) {
        opp_attack_random_style_with_spec(env, self, actions);
    }
}

static void opp_prayer_rookie(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    opp_apply_defensive_prayer(env, opp, actions, self, target, 0);

    if (hp_pct < 0.35f) {
        opp_emit_preferred_food(opp, actions, self, cons);
        if (cons.can_brew) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
            opp->potion_cooldown = 3;
        }
    } else if (hp_pct < 0.55f) {
        if (cons.can_food) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SHARK_FOOD);
            opp->food_cooldown = 3;
        } else if (cons.can_brew) {
            opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
            opp->potion_cooldown = 3;
        }
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
        opp->potion_cooldown = 3;
    }

    int eating = opp_check_eating_queued(actions);

    if (opp_should_skip_offensive(env, opp)) return;

    if (opp_attack_ready(self) && !eating) {
        opp_attack_random_style_with_spec(env, self, actions);
    }
}

typedef struct {
    int biased_style_pick;
    float spec_target_hp_gate;
    float spec_trigger_prob;
    int offensive_noop_roll;
    int has_move_block;
    float move_under_prob;
} OppNhTier;

static const OppNhTier NH_TIER_IMPROVED     = {1, 2.0f,  1.0f,  0, 1, 2.0f};
static const OppNhTier NH_TIER_COMPETENT    = {0, 0.60f, 0.50f, 1, 0, 0.0f};
static const OppNhTier NH_TIER_INTERMEDIATE = {0, 0.60f, 1.0f,  1, 0, 0.0f};
static const OppNhTier NH_TIER_ADVANCED     = {1, 2.0f,  1.0f,  1, 1, 0.0f};
static const OppNhTier NH_TIER_PROFICIENT   = {1, 2.0f,  1.0f,  1, 1, 0.25f};
static const OppNhTier NH_TIER_EXPERT       = {1, 2.0f,  1.0f,  1, 1, 0.50f};

static void opp_nh_tier(OsrsEnv* env, OpponentState* opp, int* actions, const OppNhTier* tier) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);

    opp_apply_defensive_prayer(env, opp, actions, self, target, 0);
    opp_apply_consumables(env, opp, actions, self, 1);

    int eating = opp_check_eating_queued(actions);

    if (opp_should_skip_offensive(env, opp)) return;

    if (opp_attack_ready(self) && !eating) {
        int attack_style;
        if (rand_float(env) < opp->off_prayer_rate) {
            attack_style = tier->biased_style_pick
                ? opp_pick_off_prayer_style_biased(env, opp, self, target)
                : opp_pick_from_mask(env, opp_get_off_prayer_mask(self, target));
        } else {
            attack_style = rand_int(env, 3);
        }

        opp_apply_boost_potion(env, opp, actions, self, attack_style, 0);

        float target_hp_pct = (float)target->current_hitpoints / (float)target->base_hitpoints;
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        int can_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
        int should_spec =
            self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
            target_hp_pct < tier->spec_target_hp_gate &&
            target->prayer != PRAYER_PROTECT_MELEE &&
            can_spec_range &&
            (tier->spec_trigger_prob >= 1.0f ||
             rand_float(env) < tier->spec_trigger_prob);

        if (should_spec && opp->target_fleeing_ticks >= 2 && dist > 1) {
            should_spec = 0;
            attack_style = OPP_STYLE_MAGE;
        }

        int actual_style;
        int actual_attack;
        if (should_spec) {
            actual_style = OPP_STYLE_SPEC;
            actual_attack = 3;
        } else if (attack_style == OPP_STYLE_MAGE) {
            actual_style = OPP_STYLE_MAGE;
            actual_attack = (hp_pct < 0.30f) ? 1 : 0;
        } else {
            actual_style = attack_style;
            actual_attack = 2;
        }

        if (actual_attack == 3 ||
                rand_float(env) >= opp->offensive_prayer_miss)
            opp_apply_gear_switch(actions, self, actual_style);

        if (tier->offensive_noop_roll) opp_offensive_prayer_noop_roll(env);

        opp_emit_attack(actions, actual_attack);
    } else if (tier->has_move_block && !opp_attack_ready(self)) {
        opp_move_when_waiting(env, opp, actions, self, target, tier->move_under_prob);
    }
}

static void opp_nh_basic(OsrsEnv* env, OpponentState* opp, int* actions,
                          float spec_prob, int coin_flip_spell, int drained_restore_tail) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);

    opp_apply_defensive_prayer(env, opp, actions, self, target, 0);
    opp_apply_consumables(env, opp, actions, self, drained_restore_tail);

    int eating = opp_check_eating_queued(actions);

    if (opp_should_skip_offensive(env, opp)) return;

    if (opp_attack_ready(self) && !eating) {
        int style;
        if (rand_float(env) < opp->off_prayer_rate) {
            style = opp_pick_from_mask(env, opp_get_off_prayer_mask(self, target));
        } else {
            style = rand_int(env, 3);
        }

        opp_apply_boost_potion(env, opp, actions, self, style, 0);

        if (self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
            rand_float(env) < spec_prob) {
            int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
            if (opp->target_fleeing_ticks >= 2 && dist > 1) {
                opp_apply_gear_switch(actions, self, OPP_STYLE_MAGE);
                opp_emit_attack(actions, 0);
            } else {
                opp_apply_gear_switch(actions, self, OPP_STYLE_SPEC);
                opp_emit_attack(actions, 2);
            }
        } else {
            if (rand_float(env) >= opp->offensive_prayer_miss) {
                opp_apply_gear_switch(actions, self, style);
            }

            opp_offensive_prayer_noop_roll(env);

            if (style == OPP_STYLE_MAGE) {
                int spell = coin_flip_spell
                    ? (rand_int(env, 2) == 0 ? 0 : 1)
                    : (hp_pct < 0.30f ? 1 : 0);
                opp_emit_attack(actions, spell);
            } else {
                opp_emit_attack(actions, 2);
            }
        }
    }
}

static void opp_onetick(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);

    if (!opp_attack_ready(self)) {
        opp_apply_equipment_plan(actions, self, PVP_EQUIPMENT_TANK);
    }

    opp_apply_defensive_prayer(env, opp, actions, self, target, 1);

    int potion_used = opp_apply_consumables(env, opp, actions, self, 1);

    int eating_queued = opp_check_eating_queued(actions);

    int off_mask = opp_get_off_prayer_mask(self, target);

    if (opp_try_fake_switch(env, opp, actions, self, target, off_mask, -1.0f)) return;

    int preferred_style = -1;
    if (opp->opponent_prayer_at_fake >= 0) {
        preferred_style = opp->opponent_prayer_at_fake;
        opp->opponent_prayer_at_fake = -1;
    }

    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
    OppSpecPlan spec = opp_plan_specs(self, target, dist);

    if (spec.melee && opp->target_fleeing_ticks >= 2 && dist > 1) {
        spec.melee = 0;
    }

    int actual_style;
    int actual_attack;
    PvpEquipmentPlan spec_plan = PVP_EQUIPMENT_SPEC_MELEE;

    if (spec.ranged && (dist >= 3 || target->frozen_ticks > 0)) {
        actual_style = OPP_STYLE_RANGED;
        actual_attack = 3;
        spec_plan = PVP_EQUIPMENT_SPEC_RANGED;
    } else if (spec.magic) {
        actual_style = OPP_STYLE_MAGE;
        actual_attack = 3;
        spec_plan = PVP_EQUIPMENT_SPEC_MAGIC;
    } else if (spec.melee) {
        actual_style = OPP_STYLE_SPEC;
        actual_attack = 3;
    } else if (target->frozen_ticks == 0 && (off_mask & (1 << OPP_STYLE_MAGE))) {
        actual_style = OPP_STYLE_MAGE;
        actual_attack = opp_get_mage_attack(self, target) == 0 ? 0 : 1;
    } else {
        int can_use_preferred = preferred_style >= 0 &&
            (preferred_style != OPP_STYLE_MELEE || self->frozen_ticks <= 10 || dist <= 1);

        if (can_use_preferred) {
            actual_style = preferred_style;
            if (preferred_style == OPP_STYLE_MAGE) {
                actual_attack = (hp_pct < 0.98f) ? 1 : 0;
            } else {
                actual_attack = 2;
            }
        } else if (off_mask & (1 << OPP_STYLE_MAGE)) {
            actual_style = OPP_STYLE_MAGE;
            actual_attack = (hp_pct < 0.98f) ? 1 : 0;
        } else {
            int non_mage[2];
            int nm_count = 0;
            for (int s = 1; s < 3; s++) {
                if (off_mask & (1 << s)) non_mage[nm_count++] = s;
            }
            if (nm_count == 0) {
                actual_style = OPP_STYLE_RANGED;
            } else {
                actual_style = non_mage[rand_int(env, nm_count)];
            }
            actual_attack = 2;
        }
    }

    opp_apply_boost_potion(env, opp, actions, self, actual_style, potion_used);

    if (opp_should_skip_offensive(env, opp)) return;

    if (opp_attack_ready(self) && !eating_queued) {
        if (actual_attack == 3) {
            opp_apply_equipment_plan(actions, self, spec_plan);
        } else if (rand_float(env) >= opp->offensive_prayer_miss) {
            opp_apply_gear_switch(actions, self, actual_style);
        }

        opp_emit_attack(actions, actual_attack);
    } else if (!opp_attack_ready(self)) {
        opp_move_when_waiting(env, opp, actions, self, target, 2.0f);
    }
}

static void opp_unpredictable_improved(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);

    opp_handle_delayed_prayer(env, opp, actions, self, target,
                               UNPREDICTABLE_IMP_PRAYER_CUM, UNPREDICTABLE_IMP_PRAYER_CUM_LEN,
                               UNPREDICTABLE_IMP_WRONG_PRAYER, 0);

    int potion_used = opp_apply_consumables(env, opp, actions, self, 1);

    int eating_queued = opp_check_eating_queued(actions);

    if (opp_should_skip_offensive(env, opp)) return;

    int attack_style;

    if (rand_float(env) < UNPREDICTABLE_IMP_SUBOPTIMAL_ATTACK) {
        attack_style = rand_int(env, 3);
    } else {
        attack_style = opp_pick_off_prayer_style_biased(env, opp, self, target);
    }

    opp_apply_boost_potion(env, opp, actions, self, attack_style, potion_used);

    if (opp_attack_ready(self) && !eating_queued) {
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        int can_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
        int should_spec = self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
                          target->prayer != PRAYER_PROTECT_MELEE &&
                          can_spec_range;

        if (should_spec && opp->target_fleeing_ticks >= 2 && dist > 1) {
            should_spec = 0;
            attack_style = OPP_STYLE_MAGE;
        }

        int actual_style;
        int actual_attack;

        if (should_spec) {
            actual_style = OPP_STYLE_SPEC;
            actual_attack = 3;
        } else if (attack_style == OPP_STYLE_MAGE) {
            actual_style = OPP_STYLE_MAGE;
            actual_attack = (hp_pct < 0.30f) ? 1 : 0;
        } else {
            actual_style = attack_style;
            actual_attack = 2;
        }

        if (actual_attack == 3 ||
                rand_float(env) >= opp->offensive_prayer_miss)
            opp_apply_gear_switch(actions, self, actual_style);

        int action_delay = opp_sample_delay(env, UNPREDICTABLE_IMP_ACTION_CUM, UNPREDICTABLE_IMP_ACTION_CUM_LEN);
        if (action_delay == 0) {
            opp_emit_attack(actions, actual_attack);
        }
    } else if (!opp_attack_ready(self)) {
        opp_move_when_waiting(env, opp, actions, self, target, 2.0f);
    }
}

static void opp_unpredictable_onetick(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);

    if (!opp_attack_ready(self)) {
        opp_apply_equipment_plan(actions, self, PVP_EQUIPMENT_TANK);
    }

    opp_handle_delayed_prayer(env, opp, actions, self, target,
                               UNPREDICTABLE_OT_PRAYER_CUM, UNPREDICTABLE_OT_PRAYER_CUM_LEN,
                               0.0f, 1);

    int potion_used = opp_apply_consumables(env, opp, actions, self, 1);

    int eating_queued = opp_check_eating_queued(actions);

    int off_mask = opp_get_off_prayer_mask(self, target);

    if (opp_try_fake_switch(env, opp, actions, self, target, off_mask,
                            UNPREDICTABLE_OT_FAKE_FAIL)) return;

    int preferred_style = -1;

    if (opp->opponent_prayer_at_fake >= 0 && !opp->fake_switch_failed) {
        if (rand_float(env) < UNPREDICTABLE_OT_WRONG_PREDICT) {
            preferred_style = rand_int(env, 3);
        } else {
            preferred_style = opp->opponent_prayer_at_fake;
        }
        opp->opponent_prayer_at_fake = -1;
    } else if (opp->fake_switch_failed) {
        opp->opponent_prayer_at_fake = -1;
        opp->fake_switch_failed = 0;
    }

    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
    OppSpecPlan spec = opp_plan_specs(self, target, dist);

    if (spec.melee && opp->target_fleeing_ticks >= 2 && dist > 1) {
        spec.melee = 0;
    }

    int actual_style;
    int actual_attack;
    PvpEquipmentPlan spec_plan = PVP_EQUIPMENT_SPEC_MELEE;

    if (spec.ranged && (dist >= 3 || target->frozen_ticks > 0)) {
        actual_style = OPP_STYLE_RANGED;
        actual_attack = 3;
        spec_plan = PVP_EQUIPMENT_SPEC_RANGED;
    } else if (spec.magic) {
        actual_style = OPP_STYLE_MAGE;
        actual_attack = 3;
        spec_plan = PVP_EQUIPMENT_SPEC_MAGIC;
    } else if (spec.melee) {
        actual_style = OPP_STYLE_SPEC;
        actual_attack = 3;
    } else if (target->frozen_ticks == 0 && (off_mask & (1 << OPP_STYLE_MAGE))) {
        actual_style = OPP_STYLE_MAGE;
        actual_attack = opp_get_mage_attack(self, target) == 0 ? 0 : 1;
    } else {
        int can_use_preferred = preferred_style >= 0 &&
            (preferred_style != OPP_STYLE_MELEE || self->frozen_ticks <= 10 || dist <= 1);

        if (can_use_preferred) {
            actual_style = preferred_style;
            actual_attack = (preferred_style == OPP_STYLE_MAGE)
                ? ((hp_pct < 0.98f) ? 1 : 0)
                : 2;
        } else if (off_mask & (1 << OPP_STYLE_MAGE)) {
            actual_style = OPP_STYLE_MAGE;
            actual_attack = (hp_pct < 0.98f) ? 1 : 0;
        } else {
            int non_mage[2];
            int nm_count = 0;
            for (int s = 1; s < 3; s++) {
                if (off_mask & (1 << s)) non_mage[nm_count++] = s;
            }
            actual_style = (nm_count > 0) ? non_mage[rand_int(env, nm_count)] : OPP_STYLE_RANGED;
            actual_attack = 2;
        }
    }

    opp_apply_boost_potion(env, opp, actions, self, actual_style, potion_used);

    if (opp_should_skip_offensive(env, opp)) return;

    if (opp_attack_ready(self) && !eating_queued) {
        int action_delay = opp_sample_delay(env, UNPREDICTABLE_OT_ACTION_CUM, UNPREDICTABLE_OT_ACTION_CUM_LEN);
        if (action_delay == 0) {
            if (actual_attack == 3) {
                opp_apply_equipment_plan(actions, self, spec_plan);
            } else if (rand_float(env) >= opp->offensive_prayer_miss) {
                opp_apply_gear_switch(actions, self, actual_style);
            }

            opp_emit_attack(actions, actual_attack);
        }
    } else if (!opp_attack_ready(self)) {
        opp_move_when_waiting(env, opp, actions, self, target, 2.0f);
    }
}

static void opp_read_agent_action(OsrsEnv* env, OpponentState* opp) {
    opp->has_read_this_tick = 0;
    opp->read_agent_style = ATTACK_STYLE_NONE;
    opp->read_agent_prayer = PRAYER_NONE;
    opp->read_agent_moving = 0;

    if (opp->read_chance <= 0.0f || rand_float(env) >= opp->read_chance) {
        return;
    }

    /* env->actions holds THIS tick's agent actions; pending_actions is last tick's */
    int* agent_actions = &env->actions[0];

    int primary = agent_actions[OSRS_HEAD_PRIMARY];
    int spell = agent_actions[OSRS_HEAD_SPELL];
    int weapon_cell_action = agent_actions[OSRS_HEAD_EQUIP_SLOT(GEAR_SLOT_WEAPON)];

    if (weapon_cell_action > 0 && weapon_cell_action <= OSRS_INVENTORY_SIZE) {
        const OsrsItemContentMetadata* metadata = osrs_inventory_cell_metadata(
            &env->players[0].inventory_cells[weapon_cell_action - 1]);
        if (metadata->click_action == OSRS_CLICK_EQUIP &&
                metadata->gear_slot == GEAR_SLOT_WEAPON) {
            opp->read_agent_style = (AttackStyle)metadata->attack_style;
            opp->has_read_this_tick = 1;
        }
    }
    if (!opp->has_read_this_tick &&
            (spell == OSRS_SPELL_ICE_BARRAGE ||
             spell == OSRS_SPELL_BLOOD_BARRAGE)) {
        opp->read_agent_style = ATTACK_STYLE_MAGIC;
        opp->has_read_this_tick = 1;
    } else if (!opp->has_read_this_tick &&
            primary >= OSRS_PRIMARY_MOVE_ACTIONS &&
            primary < OSRS_PRIMARY_DIM(1)) {
        opp->read_agent_style = get_slot_weapon_attack_style(&env->players[0]);
        opp->has_read_this_tick = 1;
    }

    int overhead = agent_actions[OSRS_HEAD_OVERHEAD];
    if (overhead == ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE)       opp->read_agent_prayer = PRAYER_PROTECT_MELEE;
    else if (overhead == ENCOUNTER_OVERHEAD_SET_REFRESH_RANGED) opp->read_agent_prayer = PRAYER_PROTECT_RANGED;
    else if (overhead == ENCOUNTER_OVERHEAD_SET_REFRESH_MAGIC)  opp->read_agent_prayer = PRAYER_PROTECT_MAGIC;
    else if (overhead == ENCOUNTER_OVERHEAD_SET_REFRESH_SMITE)  opp->read_agent_prayer = PRAYER_SMITE;
    else if (overhead == ENCOUNTER_OVERHEAD_SET_REFRESH_REDEMPTION) opp->read_agent_prayer = PRAYER_REDEMPTION;

    opp->read_agent_moving = primary > 0 && primary < OSRS_PRIMARY_MOVE_ACTIONS;
}

static inline int opp_get_read_defensive_prayer(OpponentState* opp) {
    if (opp->read_agent_style == ATTACK_STYLE_MAGIC) return OVERHEAD_MAGE;
    if (opp->read_agent_style == ATTACK_STYLE_RANGED) return OVERHEAD_RANGED;
    if (opp->read_agent_style == ATTACK_STYLE_MELEE) return OVERHEAD_MELEE;
    return -1;
}

static inline int opp_style_off_read_prayer(OpponentState* opp, int style) {
    if (opp->read_agent_prayer == PRAYER_NONE) return 1;
    if (style == OPP_STYLE_MAGE && opp->read_agent_prayer != PRAYER_PROTECT_MAGIC) return 1;
    if (style == OPP_STYLE_RANGED && opp->read_agent_prayer != PRAYER_PROTECT_RANGED) return 1;
    if (style == OPP_STYLE_MELEE && opp->read_agent_prayer != PRAYER_PROTECT_MELEE) return 1;
    return 0;
}

static void opp_master_nh(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];

    opp_tick_cooldowns(opp);

    opp_read_agent_action(env, opp);

    if (!opp_attack_ready(self)) {
        opp_apply_equipment_plan(actions, self, PVP_EQUIPMENT_TANK);
    }

    int def_prayer = -1;
    if (opp->has_read_this_tick && opp->read_agent_style != ATTACK_STYLE_NONE) {
        def_prayer = opp_get_read_defensive_prayer(opp);
    }
    if (def_prayer < 0) {
        if (rand_float(env) < opp->prayer_accuracy) {
            def_prayer = opp_get_defensive_prayer_with_spec(target);
        } else {
            int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
            def_prayer = prayers[rand_int(env, 3)];
        }
    }
    def_prayer = opp_apply_prayer_mistake(env, opp, def_prayer);
    if (!opp_has_prayer_active(self, def_prayer)) {
        opp_emit_prayer(actions, self, def_prayer);
    }

    int potion_used = opp_apply_consumables(env, opp, actions, self, 1);
    int eating_queued = opp_check_eating_queued(actions);

    int off_mask = opp_get_off_prayer_mask(self, target);

    if (opp_try_fake_switch(env, opp, actions, self, target, off_mask, -1.0f)) return;

    int preferred_style = -1;
    if (opp->opponent_prayer_at_fake >= 0) {
        preferred_style = opp->opponent_prayer_at_fake;
        opp->opponent_prayer_at_fake = -1;
    }

    if (opp->has_read_this_tick && opp->read_agent_prayer != PRAYER_NONE) {
        int read_off_styles[3];
        int read_off_count = 0;
        for (int s = 0; s < 3; s++) {
            if (!(off_mask & (1 << s))) continue;
            if (opp_style_off_read_prayer(opp, s)) {
                read_off_styles[read_off_count++] = s;
            }
        }
        if (read_off_count > 0) {
            preferred_style = read_off_styles[rand_int(env, read_off_count)];
        }
    }

    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
    OppSpecPlan spec = opp_plan_specs(self, target, dist);

    if (opp->has_read_this_tick) {
        if (spec.melee && opp->read_agent_prayer == PRAYER_PROTECT_MELEE)
            spec.melee = 0;
        if (spec.ranged && opp->read_agent_prayer == PRAYER_PROTECT_RANGED)
            spec.ranged = 0;
        if (spec.magic && opp->read_agent_prayer == PRAYER_PROTECT_MAGIC)
            spec.magic = 0;
    }

    if (spec.melee && opp->target_fleeing_ticks >= 2 && dist > 1) {
        spec.melee = 0;
    }

    if (spec.melee && opp->has_read_this_tick && opp->read_agent_moving && dist > 1) {
        spec.melee = 0;
    }

    int actual_style;
    int actual_attack;
    PvpEquipmentPlan spec_plan = PVP_EQUIPMENT_SPEC_MELEE;

    if (spec.ranged && (dist >= 3 || target->frozen_ticks > 0)) {
        actual_style = OPP_STYLE_RANGED;
        actual_attack = 3;
        spec_plan = PVP_EQUIPMENT_SPEC_RANGED;
    } else if (spec.magic) {
        actual_style = OPP_STYLE_MAGE;
        actual_attack = 3;
        spec_plan = PVP_EQUIPMENT_SPEC_MAGIC;
    } else if (spec.melee) {
        actual_style = OPP_STYLE_SPEC;
        actual_attack = 3;
    } else if (preferred_style >= 0) {
        actual_style = preferred_style;
        actual_attack = (preferred_style == OPP_STYLE_MAGE)
            ? (opp_get_mage_attack(self, target) == 0 ? 0 : 1)
            : 2;
    } else if (target->frozen_ticks == 0 && (off_mask & (1 << OPP_STYLE_MAGE))) {
        actual_style = OPP_STYLE_MAGE;
        actual_attack = opp_get_mage_attack(self, target) == 0 ? 0 : 1;
    } else {
        actual_style = opp_pick_from_mask(env, off_mask);
        actual_attack = (actual_style == OPP_STYLE_MAGE)
            ? (opp_get_mage_attack(self, target) == 0 ? 0 : 1)
            : 2;
    }

    opp_apply_boost_potion(env, opp, actions, self, actual_style, potion_used);

    if (opp_should_skip_offensive(env, opp)) return;

    if (opp_attack_ready(self) && !eating_queued) {
        if (actual_attack == 3) {
            opp_apply_equipment_plan(actions, self, spec_plan);
        } else if (rand_float(env) >= opp->offensive_prayer_miss) {
            opp_apply_gear_switch(actions, self, actual_style);
        }

        opp_emit_attack(actions, actual_attack);
    } else if (!opp_attack_ready(self)) {
        opp_move_when_waiting(env, opp, actions, self, target, 2.0f);
    }
}

static void opp_veng_fighter(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    opp_apply_defensive_prayer(env, opp, actions, self, target, 0);

    if (hp_pct < opp->eat_triple_threshold && cons.can_brew && (cons.can_food || cons.can_karambwan)) {
        opp_emit_preferred_food(opp, actions, self, cons);
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
        opp->potion_cooldown = 3;
    } else if (hp_pct < opp->eat_double_threshold && cons.can_food && cons.can_brew) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SHARK_FOOD);
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
        opp->food_cooldown = 3; opp->potion_cooldown = 3;
    } else if (hp_pct < opp->eat_double_threshold && (cons.can_food || cons.can_karambwan)) {
        opp_emit_preferred_food(opp, actions, self, cons);
    } else if (hp_pct < opp->eat_brew_threshold && cons.can_brew) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
        opp->potion_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_food) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SHARK_FOOD);
        opp->food_cooldown = 3;
    } else if (hp_pct < 0.60f && cons.can_karambwan) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_KARAMBWAN);
        opp->karambwan_cooldown = 2;
    } else if (opp_is_drained(self) && hp_pct < 0.90f && cons.can_brew) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && cons.can_restore) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SUPER_RESTORE);
        opp->potion_cooldown = 3;
    }

    int eating = opp_check_eating_queued(actions);

    if (!self->veng_active && remaining_ticks(self->veng_cooldown) == 0) {
        actions[OSRS_HEAD_SPELL] = OSRS_SPELL_VENGEANCE;
    }

    if (opp_should_skip_offensive(env, opp)) return;

    if (opp_attack_ready(self) && !eating) {
        int attack_style;
        if (rand_float(env) < opp->off_prayer_rate) {
            int off_mask = opp_get_off_prayer_mask(self, target);
            off_mask &= ~(1 << OPP_STYLE_MAGE);
            if (off_mask == 0) off_mask = (1 << OPP_STYLE_MELEE) | (1 << OPP_STYLE_RANGED);
            attack_style = opp_pick_from_mask(env, off_mask);
        } else {
            attack_style = rand_int(env, 2) + 1;
        }

        opp_apply_boost_potion(env, opp, actions, self, attack_style, 0);

        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        int can_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
        int should_spec = (self->special_energy >= get_melee_spec_cost(self->melee_spec_weapon) &&
                          target->prayer != PRAYER_PROTECT_MELEE &&
                          can_spec_range);

        if (should_spec && opp->target_fleeing_ticks >= 2 && dist > 1) {
            should_spec = 0;
        }

        if (should_spec) {
            opp_apply_gear_switch(actions, self, OPP_STYLE_SPEC);
            opp_emit_attack(actions, 2);
        } else {
            if (rand_float(env) >= opp->offensive_prayer_miss) {
                opp_apply_gear_switch(actions, self, attack_style);
            }
            opp_emit_attack(actions, 2);
        }
    } else if (!opp_attack_ready(self)) {
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        if (target->frozen_ticks > 0 && self->frozen_ticks == 0 && dist > 0 &&
            rand_float(env) < 0.40f) {
            opp_emit_move_toward(actions, self, target->x, target->y);
        } else if (opp->target_fleeing_ticks >= 2 && dist > 3 && self->frozen_ticks == 0) {
            opp_emit_farcast_move(actions, self, target, 3);
        }
    }
}

static void opp_blood_healer(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float prayer_pct = (float)self->current_prayer / (float)self->base_prayer;

    opp_tick_cooldowns(opp);
    OppConsumables cons = opp_get_consumables(opp, self);

    opp_apply_defensive_prayer(env, opp, actions, self, target, 0);

    if (hp_pct < 0.25f && cons.can_brew && (cons.can_food || cons.can_karambwan)) {
        opp_emit_preferred_food(opp, actions, self, cons);
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
        opp->potion_cooldown = 3;
    } else if (hp_pct < 0.35f && cons.can_food && cons.can_brew) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SHARK_FOOD);
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
        opp->food_cooldown = 3; opp->potion_cooldown = 3;
    } else if (hp_pct < 0.35f && (cons.can_food || cons.can_karambwan)) {
        opp_emit_preferred_food(opp, actions, self, cons);
    } else if (hp_pct < 0.35f && cons.can_brew) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && hp_pct < 0.50f && cons.can_brew) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_BREW);
        opp->potion_cooldown = 3;
    } else if (prayer_pct < 0.30f && cons.can_restore) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SUPER_RESTORE);
        opp->potion_cooldown = 3;
    } else if (opp_is_drained(self) && cons.can_restore) {
        opp_emit_consumable(actions, self, OSRS_CONSUMABLE_SUPER_RESTORE);
        opp->potion_cooldown = 3;
    }

    int eating = opp_check_eating_queued(actions);

    if (opp_should_skip_offensive(env, opp)) return;

    if (opp_attack_ready(self) && !eating) {
        int attack_style;
        int actual_attack;

        if (hp_pct < 0.40f) {
            attack_style = OPP_STYLE_MAGE;
            actual_attack = 1;
        } else if (hp_pct < 0.70f) {
            if (rand_float(env) < 0.80f) {
                attack_style = OPP_STYLE_MAGE;
                actual_attack = 1;
            } else {
                if (rand_float(env) < opp->off_prayer_rate) {
                    attack_style = opp_pick_off_prayer_style_biased(env, opp, self, target);
                } else {
                    attack_style = rand_int(env, 3);
                }
                if (attack_style == OPP_STYLE_MAGE) {
                    actual_attack = (target->frozen_ticks == 0 && target->freeze_immunity_ticks == 0)
                                    ? 0 : 1;
                } else {
                    actual_attack = 2;
                }
            }
        } else {
            if (rand_float(env) < opp->off_prayer_rate) {
                attack_style = opp_pick_off_prayer_style_biased(env, opp, self, target);
            } else {
                attack_style = rand_int(env, 3);
            }
            if (attack_style == OPP_STYLE_MAGE) {
                actual_attack = (target->frozen_ticks == 0 && target->freeze_immunity_ticks == 0)
                                ? 0 : 1;
            } else {
                actual_attack = 2;
            }
        }

        opp_apply_boost_potion(env, opp, actions, self, attack_style, 0);

        if (rand_float(env) >= opp->offensive_prayer_miss) {
            opp_apply_gear_switch(actions, self, attack_style);
        }

        if (hp_pct < 0.35f && actual_attack != 1) {
            opp_apply_equipment_plan(actions, self, PVP_EQUIPMENT_TANK);
            if (actual_attack == 0) opp_emit_attack(actions, actual_attack);
        } else {
            opp_emit_attack(actions, actual_attack);
        }
    } else if (!opp_attack_ready(self)) {
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);
        if (self->frozen_ticks == 0) {
            if (target->frozen_ticks > 0 && dist < 5) {
                opp_emit_farcast_move(actions, self, target, 5);
            } else if (dist < 4 && target->frozen_ticks == 0) {
                opp_emit_farcast_move(actions, self, target, 5);
            } else if (opp->target_fleeing_ticks >= 2 && dist > 5) {
                opp_emit_farcast_move(actions, self, target, 5);
            }
        }
    }
}

#define COMBO_IDLE       0
#define COMBO_SPEC_FIRED 1

static void opp_gmaul_combo(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float target_hp_pct = (float)target->current_hitpoints / (float)target->base_hitpoints;
    int has_gmaul = player_has_gmaul(self);

    opp_tick_cooldowns(opp);

    opp_apply_defensive_prayer(env, opp, actions, self, target, 0);
    opp_apply_consumables(env, opp, actions, self, 1);

    int eating = opp_check_eating_queued(actions);

    if (opp_should_skip_offensive(env, opp)) return;

    if (opp->combo_state == COMBO_SPEC_FIRED && has_gmaul && !eating) {
        opp_apply_equipment_plan(actions, self, PVP_EQUIPMENT_GMAUL);
        opp_emit_attack(actions, 2);
        opp->combo_state = COMBO_IDLE;
        return;
    }
    opp->combo_state = COMBO_IDLE;

    if (opp_attack_ready(self) && !eating) {
        int dist = chebyshev_distance(self->x, self->y, target->x, target->y);

        int melee_spec_cost = get_melee_spec_cost(self->melee_spec_weapon);
        int gmaul_cost = 50;
        int can_spec_range = (self->frozen_ticks > 0) ? (dist <= 1) : (dist <= 3);
        int should_combo = (has_gmaul &&
                           target_hp_pct < opp->ko_threshold &&
                           self->special_energy >= melee_spec_cost + gmaul_cost &&
                           target->prayer != PRAYER_PROTECT_MELEE &&
                           can_spec_range);

        uint8_t ranged_spec = find_best_ranged_spec(self);
        int should_ranged_spec = (ranged_spec != 0 &&
                                 target_hp_pct < opp->ko_threshold &&
                                 self->special_energy >= get_ranged_spec_cost(self->ranged_spec_weapon) &&
                                 target->prayer != PRAYER_PROTECT_RANGED &&
                                 rand_float(env) < 0.25f);

        if ((should_combo || should_ranged_spec) &&
            opp->target_fleeing_ticks >= 2 && dist > 1) {
            should_combo = 0;
            should_ranged_spec = 0;
        }

        if (should_combo) {
            opp_apply_gear_switch(actions, self, OPP_STYLE_SPEC);
            opp_emit_attack(actions, 2);
            opp->combo_state = COMBO_SPEC_FIRED;
        } else if (should_ranged_spec) {
            opp_apply_equipment_plan(actions, self, PVP_EQUIPMENT_SPEC_RANGED);
            opp_emit_attack(actions, 2);
        } else {
            int attack_style;
            if (rand_float(env) < opp->off_prayer_rate) {
                attack_style = opp_pick_off_prayer_style_biased(env, opp, self, target);
            } else {
                attack_style = rand_int(env, 3);
            }

            opp_apply_boost_potion(env, opp, actions, self, attack_style, 0);

            int should_regular_spec = (!has_gmaul &&
                                      self->special_energy >= melee_spec_cost &&
                                      target->prayer != PRAYER_PROTECT_MELEE &&
                                      target_hp_pct < 0.50f &&
                                      can_spec_range);
            if (should_regular_spec && opp->target_fleeing_ticks < 2) {
                opp_apply_gear_switch(actions, self, OPP_STYLE_SPEC);
                opp_emit_attack(actions, 2);
            } else {
                if (rand_float(env) >= opp->offensive_prayer_miss) {
                    opp_apply_gear_switch(actions, self, attack_style);
                }

                if (attack_style == OPP_STYLE_MAGE) {
                    int spell = target->frozen_ticks == 0 &&
                        target->freeze_immunity_ticks == 0 ? 0 : 1;
                    opp_emit_attack(actions, spell);
                } else {
                    opp_emit_attack(actions, 2);
                }
            }
        }
    } else if (!opp_attack_ready(self)) {
        opp_move_when_waiting(env, opp, actions, self, target, 2.0f);
    }
}

static void opp_range_kiter(OsrsEnv* env, OpponentState* opp, int* actions) {
    Player* self = &env->players[1];
    Player* target = &env->players[0];
    float hp_pct = (float)self->current_hitpoints / (float)self->base_hitpoints;
    float target_hp_pct = (float)target->current_hitpoints / (float)target->base_hitpoints;
    int dist = chebyshev_distance(self->x, self->y, target->x, target->y);

    opp_tick_cooldowns(opp);

    opp_apply_defensive_prayer(env, opp, actions, self, target, 0);
    opp_apply_consumables(env, opp, actions, self, 1);

    int eating = opp_check_eating_queued(actions);

    if (opp_should_skip_offensive(env, opp)) return;

    if (opp_attack_ready(self) && !eating) {
        uint8_t ranged_spec = find_best_ranged_spec(self);
        int has_ranged_spec = (ranged_spec != 0);
        int ranged_spec_cost = has_ranged_spec
                               ? get_ranged_spec_cost(self->ranged_spec_weapon) : 100;

        int should_ranged_spec = (has_ranged_spec &&
                                 self->special_energy >= ranged_spec_cost &&
                                 target->prayer != PRAYER_PROTECT_RANGED &&
                                 target_hp_pct < 0.55f);

        if (should_ranged_spec && (target->frozen_ticks > 0 || dist >= 3)) {
            opp_apply_equipment_plan(actions, self, PVP_EQUIPMENT_SPEC_RANGED);
            opp_emit_attack(actions, 2);
        } else {
            int attack_style;
            int force_melee = (self->frozen_ticks > 0 && dist <= 1);
            int prefer_ranged = (dist >= 3 || target->frozen_ticks > 0);

            if (force_melee) {
                attack_style = OPP_STYLE_MELEE;
            } else if (prefer_ranged && rand_float(env) < 0.80f) {
                attack_style = OPP_STYLE_RANGED;
            } else if (rand_float(env) < opp->off_prayer_rate) {
                attack_style = opp_pick_off_prayer_style_biased(env, opp, self, target);
            } else {
                attack_style = rand_int(env, 3);
            }

            int actual_attack;
            if (hp_pct < 0.30f && attack_style == OPP_STYLE_MAGE) {
                actual_attack = 1;
            } else if (attack_style == OPP_STYLE_MAGE) {
                actual_attack = (target->frozen_ticks == 0 &&
                                target->freeze_immunity_ticks == 0)
                               ? 0 : 2;
                if (actual_attack == 2) attack_style = OPP_STYLE_RANGED;
            } else {
                actual_attack = 2;
            }

            opp_apply_boost_potion(env, opp, actions, self, attack_style, 0);

            int melee_spec_cost = get_melee_spec_cost(self->melee_spec_weapon);
            int can_melee_spec = (self->special_energy >= melee_spec_cost &&
                                 target->prayer != PRAYER_PROTECT_MELEE &&
                                 dist <= 1 && self->frozen_ticks == 0);
            if (can_melee_spec && target_hp_pct < 0.40f && !has_ranged_spec) {
                opp_apply_gear_switch(actions, self, OPP_STYLE_SPEC);
                opp_emit_attack(actions, 2);
            } else {
                if (rand_float(env) >= opp->offensive_prayer_miss) {
                    opp_apply_gear_switch(actions, self, attack_style);
                }

                opp_emit_attack(actions, actual_attack);
            }
        }
    } else if (!opp_attack_ready(self)) {
        if (self->frozen_ticks == 0) {
            if (target->frozen_ticks > 0 && dist < 5) {
                opp_emit_farcast_move(actions, self, target, 5);
            } else if (dist < 4) {
                opp_emit_farcast_move(actions, self, target, 5);
            } else if (dist > 7) {
                opp_emit_farcast_move(actions, self, target, 5);
            }
        }
    }
}

static const OpponentType MIXED_EASY_POOL[] = {
    OPP_PANICKING, OPP_TRUE_RANDOM, OPP_WEAK_RANDOM, OPP_SEMI_RANDOM,
    OPP_STICKY_PRAYER, OPP_RANDOM_EATER, OPP_PRAYER_ROOKIE, OPP_IMPROVED,
};
static const int MIXED_EASY_CUM_WEIGHTS[] = {18, 36, 54, 69, 79, 89, 95, 100};
#define MIXED_EASY_POOL_SIZE 8

static const OpponentType MIXED_MEDIUM_POOL[] = {
    OPP_RANDOM_EATER, OPP_PRAYER_ROOKIE, OPP_STICKY_PRAYER,
    OPP_SEMI_RANDOM, OPP_IMPROVED,
};
static const int MIXED_MEDIUM_CUM_WEIGHTS[] = {25, 45, 65, 80, 100};
#define MIXED_MEDIUM_POOL_SIZE 5

static const OpponentType MIXED_HARD_POOL[] = {
    OPP_IMPROVED, OPP_ONETICK, OPP_UNPREDICTABLE_IMPROVED,
    OPP_UNPREDICTABLE_ONETICK, OPP_RANDOM_EATER,
};
static const int MIXED_HARD_CUM_WEIGHTS[] = {20, 40, 60, 80, 100};
#define MIXED_HARD_POOL_SIZE 5

static const OpponentType MIXED_HARD_BALANCED_POOL[] = {
    OPP_RANDOM_EATER, OPP_IMPROVED, OPP_UNPREDICTABLE_IMPROVED,
    OPP_ONETICK, OPP_UNPREDICTABLE_ONETICK,
};
static const int MIXED_HARD_BALANCED_CUM_WEIGHTS[] = {25, 55, 75, 90, 100};
#define MIXED_HARD_BALANCED_POOL_SIZE 5

static OpponentType opp_select_from_pool(
    OsrsEnv* env, const OpponentType* pool, const int* cum_weights, int pool_size
) {
    int r = rand_int(env, 100);
    for (int i = 0; i < pool_size; i++) {
        if (r < cum_weights[i]) return pool[i];
    }
    return pool[pool_size - 1];
}

static void opponent_reset(OsrsEnv* env, OpponentState* opp) {
    opp->food_cooldown = 0;
    opp->potion_cooldown = 0;
    opp->karambwan_cooldown = 0;
    opp->current_prayer_set = 0;

    opp->fake_switch_pending = 0;
    opp->fake_switch_style = -1;
    opp->opponent_prayer_at_fake = -1;
    opp->fake_switch_failed = 0;
    opp->pending_prayer_value = 0;
    opp->pending_prayer_delay = 0;
    opp->last_target_gear_style = -1;

    opp->eat_triple_threshold = 0.30f + (rand_float(env) * 0.10f - 0.05f);
    opp->eat_double_threshold = 0.50f + (rand_float(env) * 0.10f - 0.05f);
    opp->eat_brew_threshold   = 0.70f + (rand_float(env) * 0.10f - 0.05f);

    opp->has_read_this_tick = 0;
    opp->read_agent_style = ATTACK_STYLE_NONE;
    opp->read_agent_prayer = PRAYER_NONE;
    opp->read_chance = 0.0f;
    opp->read_agent_moving = 0;
    opp->prev_dist_to_target = 0;
    opp->target_fleeing_ticks = 0;

    if (opp->type == OPP_PANICKING) {
        int prayers[] = {OVERHEAD_MELEE, OVERHEAD_RANGED, OVERHEAD_MAGE};
        opp->chosen_prayer = prayers[rand_int(env, 3)];
        opp->chosen_style = rand_int(env, 3);
    }

    if (opp->type == OPP_MIXED_EASY) {
        opp->active_sub_policy = opp_select_from_pool(
            env, MIXED_EASY_POOL, MIXED_EASY_CUM_WEIGHTS, MIXED_EASY_POOL_SIZE);
    } else if (opp->type == OPP_MIXED_MEDIUM) {
        opp->active_sub_policy = opp_select_from_pool(
            env, MIXED_MEDIUM_POOL, MIXED_MEDIUM_CUM_WEIGHTS, MIXED_MEDIUM_POOL_SIZE);
    } else if (opp->type == OPP_MIXED_HARD) {
        opp->active_sub_policy = opp_select_from_pool(
            env, MIXED_HARD_POOL, MIXED_HARD_CUM_WEIGHTS, MIXED_HARD_POOL_SIZE);
    } else if (opp->type == OPP_MIXED_HARD_BALANCED) {
        opp->active_sub_policy = opp_select_from_pool(
            env, MIXED_HARD_BALANCED_POOL, MIXED_HARD_BALANCED_CUM_WEIGHTS,
            MIXED_HARD_BALANCED_POOL_SIZE);
    } else if (opp->type == OPP_PFSP && env->pvp_runtime.pfsp.pool_size > 0) {
        int idx = 0;
        int r = rand_int(env, 1000);
        for (int i = 0; i < env->pvp_runtime.pfsp.pool_size; i++) {
            if (r < env->pvp_runtime.pfsp.cum_weights[i]) { idx = i; break; }
        }
        env->pvp_runtime.pfsp.active_pool_idx = idx;
        opp->active_sub_policy = env->pvp_runtime.pfsp.pool[idx];

        if (opp->active_sub_policy == OPP_SELFPLAY) {
            env->pvp_runtime.use_c_opponent = 0;
            env->pvp_runtime.use_external_opponent_actions = 1;
            if (env->ocean_io.selfplay_mask) *env->ocean_io.selfplay_mask = 1;
        } else {
            env->pvp_runtime.use_c_opponent = 1;
            env->pvp_runtime.use_external_opponent_actions = 0;
            if (env->ocean_io.selfplay_mask) *env->ocean_io.selfplay_mask = 0;
        }
    } else if (opp->type == OPP_PFSP) {
        opp->active_sub_policy = OPP_IMPROVED;
        env->pvp_runtime.pfsp.active_pool_idx = -1;
    }

    OpponentType resolved = opp->active_sub_policy ? opp->active_sub_policy : opp->type;
    if (resolved > 0 && resolved <= OPP_RANGE_KITER) {
        const OpponentRandRanges* r = &OPP_RAND_RANGES[resolved];
        opp->prayer_accuracy = rand_range(env, r->prayer_accuracy);
        opp->off_prayer_rate = rand_range(env, r->off_prayer_rate);
        opp->offensive_prayer_rate = rand_range(env, r->offensive_prayer_rate);
        opp->action_delay_chance = rand_range(env, r->action_delay_chance);
        opp->mistake_rate = rand_range(env, r->mistake_rate);
        opp->offensive_prayer_miss = rand_range(env, r->offensive_prayer_miss);
    }

    if (resolved == OPP_MASTER_NH) {
        opp->read_chance = 0.10f;
    } else if (resolved == OPP_SAVANT_NH) {
        opp->read_chance = 0.25f;
    } else if (resolved == OPP_NIGHTMARE_NH) {
        opp->read_chance = 0.50f;
    }

    if (resolved == OPP_VENG_FIGHTER) {
        env->players[1].is_lunar_spellbook = 1;
    }

    if (resolved == OPP_IMPROVED || resolved == OPP_ONETICK ||
        resolved == OPP_UNPREDICTABLE_IMPROVED || resolved == OPP_UNPREDICTABLE_ONETICK ||
        (resolved >= OPP_ADVANCED_NH && resolved <= OPP_NIGHTMARE_NH) ||
        resolved == OPP_BLOOD_HEALER || resolved == OPP_GMAUL_COMBO ||
        resolved == OPP_RANGE_KITER) {
        float raw[3];
        for (int i = 0; i < 3; i++) raw[i] = 0.33f + (rand_float(env) - 0.5f) * 0.4f;
        float sum = raw[0] + raw[1] + raw[2];
        for (int i = 0; i < 3; i++) opp->style_bias[i] = raw[i] / sum;
    } else {
        opp->style_bias[0] = opp->style_bias[1] = opp->style_bias[2] = 0.333f;
    }

    if (resolved == OPP_GMAUL_COMBO) {
        opp->combo_state = 0;
        opp->ko_threshold = 0.45f + rand_float(env) * 0.15f;
    }
}

static void generate_opponent_action(OsrsEnv* env, OpponentState* opp) {
    int* actions = env->pending_actions + OSRS_BASE_NUM_ACTION_HEADS;

    memset(actions, 0, OSRS_BASE_NUM_ACTION_HEADS * sizeof(int));

    opp_update_flee_tracking(opp, &env->players[1], &env->players[0]);

    OpponentType active = opp->type;
    if (active == OPP_MIXED_EASY || active == OPP_MIXED_MEDIUM ||
        active == OPP_MIXED_HARD || active == OPP_MIXED_HARD_BALANCED ||
        active == OPP_PFSP) {
        active = opp->active_sub_policy;
    }

    switch (active) {
        case OPP_TRUE_RANDOM:
            opp_true_random(env, actions);
            break;
        case OPP_PANICKING:
            opp_panicking(env, opp, actions);
            break;
        case OPP_WEAK_RANDOM:
            opp_weak_random(env, opp, actions);
            break;
        case OPP_SEMI_RANDOM:
            opp_semi_random(env, opp, actions);
            break;
        case OPP_STICKY_PRAYER:
            opp_sticky_prayer(env, opp, actions);
            break;
        case OPP_RANDOM_EATER:
            opp_random_eater(env, opp, actions);
            break;
        case OPP_PRAYER_ROOKIE:
            opp_prayer_rookie(env, opp, actions);
            break;
        case OPP_IMPROVED:
            opp_nh_tier(env, opp, actions, &NH_TIER_IMPROVED);
            break;
        case OPP_ONETICK:
            opp_onetick(env, opp, actions);
            break;
        case OPP_UNPREDICTABLE_IMPROVED:
            opp_unpredictable_improved(env, opp, actions);
            break;
        case OPP_UNPREDICTABLE_ONETICK:
            opp_unpredictable_onetick(env, opp, actions);
            break;
        case OPP_NOVICE_NH:
            opp_nh_basic(env, opp, actions, 0.15f, 1, 0);
            break;
        case OPP_APPRENTICE_NH:
            opp_nh_basic(env, opp, actions, 0.30f, 0, 1);
            break;
        case OPP_COMPETENT_NH:
            opp_nh_tier(env, opp, actions, &NH_TIER_COMPETENT);
            break;
        case OPP_INTERMEDIATE_NH:
            opp_nh_tier(env, opp, actions, &NH_TIER_INTERMEDIATE);
            break;
        case OPP_ADVANCED_NH:
            opp_nh_tier(env, opp, actions, &NH_TIER_ADVANCED);
            break;
        case OPP_PROFICIENT_NH:
            opp_nh_tier(env, opp, actions, &NH_TIER_PROFICIENT);
            break;
        case OPP_EXPERT_NH:
            opp_nh_tier(env, opp, actions, &NH_TIER_EXPERT);
            break;
        case OPP_MASTER_NH:
        case OPP_SAVANT_NH:
        case OPP_NIGHTMARE_NH:
            opp_master_nh(env, opp, actions);
            break;
        case OPP_VENG_FIGHTER:
            opp_veng_fighter(env, opp, actions);
            break;
        case OPP_BLOOD_HEALER:
            opp_blood_healer(env, opp, actions);
            break;
        case OPP_GMAUL_COMBO:
            opp_gmaul_combo(env, opp, actions);
            break;
        case OPP_RANGE_KITER:
            opp_range_kiter(env, opp, actions);
            break;
        default:
            break;
    }
}

static inline void pvp_swap_players_and_actions(OsrsEnv* env) {
    Player player = env->players[0];
    env->players[0] = env->players[1];
    env->players[1] = player;
    for (int head = 0; head < OSRS_BASE_NUM_ACTION_HEADS; head++) {
        int action = env->pending_actions[head];
        env->pending_actions[head] =
            env->pending_actions[OSRS_BASE_NUM_ACTION_HEADS + head];
        env->pending_actions[OSRS_BASE_NUM_ACTION_HEADS + head] = action;
    }
}

static inline void generate_opponent_action_for_player0(
    OsrsEnv* env,
    OpponentState* opp
) {
    pvp_swap_players_and_actions(env);
    generate_opponent_action(env, opp);
    pvp_swap_players_and_actions(env);
}

#endif /* OSRS_PVP_OPPONENTS_H */
