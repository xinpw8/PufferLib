#ifndef OSRS_CONSUMABLES_H
#define OSRS_CONSUMABLES_H

#include <stdint.h>

typedef enum {
    FOOD_SHARK = 0,
    FOOD_KARAMBWAN,
    FOOD_MANTA_RAY,
    FOOD_ANGLERFISH,
    NUM_FOOD_TYPES
} FoodType;

typedef enum {
    POTION_PRAYER_RESTORE = 0,
    POTION_SUPER_RESTORE,
    POTION_ANTIVENOM_PLUS,
    POTION_RANGING,
    POTION_SUPER_COMBAT,
    POTION_SATURATED_HEART,
    POTION_SANFEW,
    NUM_POTION_TYPES
} PotionType;

typedef struct {
    int hp_healed;
    int consumed;
} EatResult;

typedef struct {
    int prayer_restored;
    int level_boost;
    int venom_cured;
    int antivenom_ticks;
    int consumed;
} DrinkResult;

typedef struct {
    int hp_healed;
    int def_boost;
    int att_drain;
    int str_drain;
    int range_drain;
    int magic_drain;
} BrewResult;

static inline int osrs_food_heal_amount(FoodType type) {
    switch (type) {
        case FOOD_SHARK:       return 20;
        case FOOD_KARAMBWAN:   return 18;
        case FOOD_MANTA_RAY:   return 22;
        case FOOD_ANGLERFISH:  return 22;
        default: return 0;
    }
}

static inline int osrs_saturated_heart_magic_boost(int base_magic) {
    return 4 + base_magic / 10;
}

static inline int osrs_super_restore_amount(int level) {
    return 8 + level / 4;
}

static inline int osrs_sanfew_restore_amount(int level) {
    return 4 + level * 30 / 100;
}

static inline int osrs_super_combat_boost_amount(int level) {
    return 5 + level * 15 / 100;
}

static inline int osrs_ranging_boost_amount(int level) {
    return 4 + level / 10;
}

static inline int osrs_brew_heal_amount(int base_hp) {
    return base_hp * 15 / 100 + 2;
}

static inline int osrs_brew_defence_boost_amount(int base_defence) {
    return base_defence * 20 / 100 + 2;
}

static inline EatResult osrs_eat_food(FoodType type, int current_hp, int max_hp, int food_timer) {
    EatResult r = {0, 0};
    if (food_timer > 0) return r;

    int heal = osrs_food_heal_amount(type);
    if (heal <= 0) return r;

    if (type == FOOD_ANGLERFISH) {
        r.consumed = 1;
        r.hp_healed = heal;
        return r;
    }

    if (current_hp >= max_hp) return r;

    r.consumed = 1;
    r.hp_healed = heal;
    if (current_hp + heal > max_hp) r.hp_healed = max_hp - current_hp;
    return r;
}

static inline DrinkResult osrs_drink_potion(PotionType type, int current_prayer,
                                             int prayer_level, int potion_timer) {
    DrinkResult r = {0, 0, 0, 0, 0};
    if (potion_timer > 0) return r;

    switch (type) {
        case POTION_PRAYER_RESTORE:
            if (current_prayer >= prayer_level) return r;
            r.consumed = 1;
            r.prayer_restored = 7 + prayer_level / 4;
            break;
        case POTION_SUPER_RESTORE:
            if (current_prayer >= prayer_level) return r;
            r.consumed = 1;
            r.prayer_restored = osrs_super_restore_amount(prayer_level);
            break;
        case POTION_SANFEW:
            r.consumed = 1;
            r.prayer_restored = osrs_sanfew_restore_amount(prayer_level);
            r.venom_cured = 1;
            break;
        case POTION_ANTIVENOM_PLUS:
            r.consumed = 1;
            r.venom_cured = 1;
            r.antivenom_ticks = 300;
            break;
        case POTION_RANGING:
            r.consumed = 1;
            r.level_boost = osrs_ranging_boost_amount(prayer_level);
            break;
        case POTION_SUPER_COMBAT:
            r.consumed = 1;
            r.level_boost = osrs_super_combat_boost_amount(prayer_level);
            break;
        case POTION_SATURATED_HEART:
            r.consumed = 1;
            r.level_boost = osrs_saturated_heart_magic_boost(prayer_level);
            break;
        default:
            break;
    }
    return r;
}

static inline BrewResult osrs_brew_effect(int base_hp, int base_def,
                                           int current_att, int current_str,
                                           int current_range, int current_magic) {
    BrewResult r;
    r.hp_healed = osrs_brew_heal_amount(base_hp);
    r.def_boost = osrs_brew_defence_boost_amount(base_def);
    r.att_drain = current_att * 10 / 100 + 2;
    r.str_drain = current_str * 10 / 100 + 2;
    r.range_drain = current_range * 10 / 100 + 2;
    r.magic_drain = current_magic * 10 / 100 + 2;
    return r;
}

#endif
