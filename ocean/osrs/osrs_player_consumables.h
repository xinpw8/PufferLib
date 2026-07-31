#ifndef OSRS_PLAYER_CONSUMABLES_H
#define OSRS_PLAYER_CONSUMABLES_H

#include <stdio.h>
#include <stdlib.h>

#include "osrs_types.h"
#include "osrs_consumables.h"

typedef struct {
    int hp_healed;
    int hp_wasted;
    int attack_delay_ticks;
    int consumed;
} OsrsPlayerEatResult;

static inline void osrs_require_player_food_action(FoodType type) {
    switch (type) {
        case FOOD_SHARK:
        case FOOD_KARAMBWAN:
        case FOOD_MANTA_RAY:
        case FOOD_ANGLERFISH:
            return;
        default:
            fprintf(stderr, "unsupported player food action type: %d\n", (int)type);
            abort();
    }
}

static inline int osrs_player_food_timer(const Player* p, FoodType type) {
    osrs_require_player_food_action(type);
    return type == FOOD_KARAMBWAN ? p->karambwan_timer : p->food_timer;
}

static inline int osrs_player_food_count(const Player* p, FoodType type) {
    osrs_require_player_food_action(type);
    return type == FOOD_KARAMBWAN ? p->karambwan_count : p->food_count;
}

static inline int osrs_player_can_eat_food_type(const Player* p, FoodType type) {
    osrs_require_player_food_action(type);
    if (osrs_player_food_count(p, type) <= 0) return 0;
    EatResult r = osrs_eat_food(type, p->current_hitpoints,
        p->base_hitpoints, osrs_player_food_timer(p, type));
    return r.consumed;
}

static inline int osrs_player_food_wasted_hp(const Player* p, FoodType type) {
    osrs_require_player_food_action(type);
    EatResult r = osrs_eat_food(type, p->current_hitpoints,
        p->base_hitpoints, osrs_player_food_timer(p, type));
    if (!r.consumed) return 0;
    return osrs_food_heal_amount(type) - r.hp_healed;
}

static inline OsrsPlayerEatResult osrs_player_eat_food_type(Player* p, FoodType type) {
    osrs_require_player_food_action(type);
    OsrsPlayerEatResult out = {0, 0, 0, 0};
    if (osrs_player_food_count(p, type) <= 0) return out;

    EatResult r = osrs_eat_food(type, p->current_hitpoints,
        p->base_hitpoints, osrs_player_food_timer(p, type));
    if (!r.consumed) return out;

    int heal_amount = osrs_food_heal_amount(type);
    out.consumed = 1;
    out.hp_healed = r.hp_healed;
    out.hp_wasted = heal_amount - r.hp_healed;

    if (type == FOOD_KARAMBWAN) {
        p->karambwan_count--;
        p->karambwan_timer = 2;
        p->food_timer = 3;
        p->potion_timer = 3;
        p->ate_karambwan_this_tick = 1;
        p->last_karambwan_heal = out.hp_healed;
        p->last_karambwan_waste = out.hp_wasted;
        out.attack_delay_ticks = 2;
    } else {
        p->food_count--;
        p->food_timer = 3;
        p->ate_food_this_tick = 1;
        p->last_food_heal = out.hp_healed;
        p->last_food_waste = out.hp_wasted;
        out.attack_delay_ticks = 3;
    }

    p->current_hitpoints += out.hp_healed;
    if (type != FOOD_ANGLERFISH && p->current_hitpoints > p->base_hitpoints)
        p->current_hitpoints = p->base_hitpoints;

    int combat_ticks = 0;
    if (p->has_attack_timer) {
        combat_ticks = p->attack_timer > 0 ? p->attack_timer : 0;
    }
    p->attack_timer = combat_ticks + out.attack_delay_ticks;
    p->attack_timer_uncapped = combat_ticks + out.attack_delay_ticks;
    p->has_attack_timer = 1;

    return out;
}

#endif
