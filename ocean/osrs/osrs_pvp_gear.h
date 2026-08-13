#ifndef OSRS_PVP_GEAR_H
#define OSRS_PVP_GEAR_H

#include "osrs_types.h"
#include "osrs_items.h"
#include "osrs_inventory.h"
#include "osrs_combat.h"
#include "osrs_item_effects.h"

static const MeleeBonusType MELEE_SPEC_BONUS_TYPES[] = {
    [MELEE_SPEC_NONE] = MELEE_BONUS_SLASH,
    [MELEE_SPEC_AGS] = MELEE_BONUS_SLASH,
    [MELEE_SPEC_DRAGON_CLAWS] = MELEE_BONUS_SLASH,
    [MELEE_SPEC_GRANITE_MAUL] = MELEE_BONUS_CRUSH,
    [MELEE_SPEC_DRAGON_DAGGER] = MELEE_BONUS_STAB,
    [MELEE_SPEC_VOIDWAKER] = MELEE_BONUS_SLASH,
    [MELEE_SPEC_DWH] = MELEE_BONUS_CRUSH,
    [MELEE_SPEC_BGS] = MELEE_BONUS_SLASH,
    [MELEE_SPEC_ZGS] = MELEE_BONUS_SLASH,
    [MELEE_SPEC_SGS] = MELEE_BONUS_SLASH,
    [MELEE_SPEC_ANCIENT_GS] = MELEE_BONUS_SLASH,
    [MELEE_SPEC_VESTAS] = MELEE_BONUS_SLASH,
    [MELEE_SPEC_ABYSSAL_DAGGER] = MELEE_BONUS_STAB,
    [MELEE_SPEC_DRAGON_LONGSWORD] = MELEE_BONUS_SLASH,
    [MELEE_SPEC_DRAGON_MACE] = MELEE_BONUS_CRUSH,
    [MELEE_SPEC_ABYSSAL_BLUDGEON] = MELEE_BONUS_CRUSH,
};

static const uint8_t MELEE_WEAPON_PRIORITY[] = {
    ITEM_VESTAS, ITEM_GHRAZI_RAPIER, ITEM_INQUISITORS_MACE, ITEM_ELDER_MAUL,
    ITEM_VOIDWAKER, ITEM_ANCIENT_GS, ITEM_AGS, ITEM_STATIUS_WARHAMMER, ITEM_WHIP
};

static const uint8_t RANGE_WEAPON_PRIORITY[] = {
    ITEM_MORRIGANS_JAVELIN, ITEM_ZARYTE_CROSSBOW, ITEM_ARMADYL_CROSSBOW, ITEM_RUNE_CROSSBOW
};

static const uint8_t MAGE_WEAPON_PRIORITY[] = {
    ITEM_ZURIELS_STAFF, ITEM_KODAI_WAND, ITEM_VOLATILE_STAFF,
    ITEM_STAFF_OF_DEAD, ITEM_AHRIM_STAFF
};

static const uint8_t MELEE_SPEC_PRIORITY[] = {
    ITEM_VESTAS, ITEM_ANCIENT_GS, ITEM_AGS, ITEM_DRAGON_CLAWS,
    ITEM_VOIDWAKER, ITEM_STATIUS_WARHAMMER, ITEM_DRAGON_DAGGER
};
#define MELEE_SPEC_PRIORITY_LEN (sizeof(MELEE_SPEC_PRIORITY) / sizeof(MELEE_SPEC_PRIORITY[0]))

static const uint8_t RANGE_SPEC_PRIORITY[] = {
    ITEM_MORRIGANS_JAVELIN, ITEM_ZARYTE_CROSSBOW, ITEM_ARMADYL_CROSSBOW,
    ITEM_DARK_BOW, ITEM_HEAVY_BALLISTA
};
#define RANGE_SPEC_PRIORITY_LEN (sizeof(RANGE_SPEC_PRIORITY) / sizeof(RANGE_SPEC_PRIORITY[0]))

static const uint8_t MAGIC_SPEC_PRIORITY[] = {
    ITEM_VOLATILE_STAFF
};
#define MAGIC_SPEC_PRIORITY_LEN (sizeof(MAGIC_SPEC_PRIORITY) / sizeof(MAGIC_SPEC_PRIORITY[0]))

static const uint8_t TANK_BODY_PRIORITY[] = {
    ITEM_KARILS_TOP, ITEM_BLACK_DHIDE_BODY
};

static const uint8_t MAGE_BODY_PRIORITY[] = {
    ITEM_ANCESTRAL_TOP, ITEM_AHRIMS_ROBETOP, ITEM_MYSTIC_TOP
};

static const uint8_t TANK_LEGS_PRIORITY[] = {
    ITEM_BANDOS_TASSETS, ITEM_TORAGS_PLATELEGS, ITEM_DHAROKS_PLATELEGS,
    ITEM_VERACS_PLATESKIRT, ITEM_RUNE_PLATELEGS
};

static const uint8_t MAGE_LEGS_PRIORITY[] = {
    ITEM_ANCESTRAL_BOTTOM, ITEM_AHRIMS_ROBESKIRT, ITEM_MYSTIC_BOTTOM
};

static const uint8_t MELEE_SHIELD_PRIORITY[] = {
    ITEM_DRAGON_DEFENDER
};

static const uint8_t TANK_SHIELD_PRIORITY[] = {
    ITEM_BLESSED_SPIRIT_SHIELD, ITEM_SPIRIT_SHIELD
};

static const uint8_t MAGE_SHIELD_PRIORITY[] = {
    ITEM_MAGES_BOOK, ITEM_BLESSED_SPIRIT_SHIELD, ITEM_SPIRIT_SHIELD
};

static const uint8_t TANK_HEAD_PRIORITY[] = {
    ITEM_TORAGS_HELM, ITEM_GUTHANS_HELM, ITEM_VERACS_HELM,
    ITEM_DHAROKS_HELM, ITEM_HELM_NEITIZNOT
};

static const uint8_t MAGE_HEAD_PRIORITY[] = {ITEM_ANCESTRAL_HAT, ITEM_HELM_NEITIZNOT};

static const uint8_t MELEE_CAPE_PRIORITY[] = {ITEM_INFERNAL_CAPE, ITEM_GOD_CAPE};

static const uint8_t MAGE_CAPE_PRIORITY[] = {ITEM_GOD_CAPE};

static const uint8_t MELEE_NECK_PRIORITY[] = {ITEM_FURY, ITEM_GLORY};

static const uint8_t MAGE_NECK_PRIORITY[] = {ITEM_OCCULT_NECKLACE, ITEM_GLORY};

static const uint8_t MELEE_RING_PRIORITY[] = {ITEM_BERSERKER_RING};

static const uint8_t MAGE_RING_PRIORITY[] = {ITEM_LIGHTBEARER, ITEM_SEERS_RING_I, ITEM_BERSERKER_RING};

static inline GearBonuses* get_slot_gear_bonuses(Player* p) {
    osrs_ensure_player_equipment(p);
    return &p->slot_cached_bonuses;
}

static inline void update_spec_weapons_for_weapon(Player* p, uint8_t weapon_item) {
    p->melee_spec_weapon = MELEE_SPEC_NONE;
    p->ranged_spec_weapon = RANGED_SPEC_NONE;
    p->magic_spec_weapon = MAGIC_SPEC_NONE;

    switch (weapon_item) {
        case ITEM_DRAGON_DAGGER:
            p->melee_spec_weapon = MELEE_SPEC_DRAGON_DAGGER; break;
        case ITEM_DRAGON_CLAWS:
            p->melee_spec_weapon = MELEE_SPEC_DRAGON_CLAWS; break;
        case ITEM_AGS:
            p->melee_spec_weapon = MELEE_SPEC_AGS; break;
        case ITEM_ANCIENT_GS:
            p->melee_spec_weapon = MELEE_SPEC_ANCIENT_GS; break;
        case ITEM_GRANITE_MAUL:
            p->melee_spec_weapon = MELEE_SPEC_GRANITE_MAUL; break;
        case ITEM_VESTAS:
            p->melee_spec_weapon = MELEE_SPEC_VESTAS; break;
        case ITEM_VOIDWAKER:
            p->melee_spec_weapon = MELEE_SPEC_VOIDWAKER; break;
        case ITEM_STATIUS_WARHAMMER:
            p->melee_spec_weapon = MELEE_SPEC_DWH; break;
        case ITEM_ELDER_MAUL:
            break;
        case ITEM_DARK_BOW:
            p->ranged_spec_weapon = RANGED_SPEC_DARK_BOW; break;
        case ITEM_HEAVY_BALLISTA:
            p->ranged_spec_weapon = RANGED_SPEC_BALLISTA; break;
        case ITEM_ARMADYL_CROSSBOW:
            p->ranged_spec_weapon = RANGED_SPEC_ACB; break;
        case ITEM_ZARYTE_CROSSBOW:
            p->ranged_spec_weapon = RANGED_SPEC_ZCB; break;
        case ITEM_MORRIGANS_JAVELIN:
            p->ranged_spec_weapon = RANGED_SPEC_MORRIGANS; break;
        case ITEM_VOLATILE_STAFF:
            p->magic_spec_weapon = MAGIC_SPEC_VOLATILE_STAFF; break;
        default:
            break;
    }
}

static inline int item_is_spec_weapon(uint8_t weapon_item) {
    switch (weapon_item) {
        case ITEM_DRAGON_DAGGER:
        case ITEM_DRAGON_CLAWS:
        case ITEM_AGS:
        case ITEM_ANCIENT_GS:
        case ITEM_GRANITE_MAUL:
        case ITEM_VESTAS:
        case ITEM_VOIDWAKER:
        case ITEM_STATIUS_WARHAMMER:
        case ITEM_DARK_BOW:
        case ITEM_HEAVY_BALLISTA:
        case ITEM_ARMADYL_CROSSBOW:
        case ITEM_ZARYTE_CROSSBOW:
        case ITEM_MORRIGANS_JAVELIN:
        case ITEM_VOLATILE_STAFF:
            return 1;
        default:
            return 0;
    }
}

/* PvP inventories are per-slot LMS upgrade pools, not the flat 28-slot bag, so
   osrs_equip_from_inventory does not apply here */
static inline int slot_equip_item(Player* p, int gear_slot, uint8_t item_idx) {
    if (gear_slot < 0 || gear_slot >= NUM_GEAR_SLOTS) return 0;
    if (p->equipped[gear_slot] == item_idx) return 0;

    p->equipped[gear_slot] = item_idx;
    p->slot_gear_dirty = 1;

    if (gear_slot == GEAR_SLOT_WEAPON && item_idx < NUM_ITEMS) {
        update_spec_weapons_for_weapon(p, item_idx);
        int style = get_item_attack_style(item_idx);

        if (item_is_spec_weapon(item_idx)) {
            p->current_gear = GEAR_SPEC;
        } else if (style == ATTACK_STYLE_MELEE) {
            p->current_gear = GEAR_MELEE;
        } else if (style == ATTACK_STYLE_RANGED) {
            p->current_gear = GEAR_RANGED;
        } else if (style == ATTACK_STYLE_MAGIC) {
            p->current_gear = GEAR_MAGE;
        }

        if (item_idx == ITEM_VOIDWAKER) {
            p->visible_gear = GEAR_MAGE;
        } else if (style == ATTACK_STYLE_MELEE) {
            p->visible_gear = GEAR_MELEE;
        } else if (style == ATTACK_STYLE_RANGED) {
            p->visible_gear = GEAR_RANGED;
        } else if (style == ATTACK_STYLE_MAGIC) {
            p->visible_gear = GEAR_MAGE;
        }
    }

    if (gear_slot == GEAR_SLOT_WEAPON) {
        p->equipped[GEAR_SLOT_SHIELD] = osrs_suppress_shield_for_two_handed_weapon(
            item_idx, p->equipped[GEAR_SLOT_SHIELD]);
    }

    osrs_refresh_player_equipment(p);
    return 1;
}

static inline int player_has_item_in_slot(Player* p, int gear_slot, uint8_t item_idx) {
    for (int i = 0; i < p->num_items_in_slot[gear_slot]; i++) {
        if (p->inventory[gear_slot][i] == item_idx) return 1;
    }
    return 0;
}

static inline uint8_t find_best_available(
    Player* p, int gear_slot,
    const uint8_t* priority, int priority_len
) {
    for (int i = 0; i < priority_len; i++) {
        if (player_has_item_in_slot(p, gear_slot, priority[i])) {
            return priority[i];
        }
    }
    return ITEM_NONE;
}

static inline uint8_t find_best_melee_spec(Player* p) {
    return find_best_available(p, GEAR_SLOT_WEAPON, MELEE_SPEC_PRIORITY, MELEE_SPEC_PRIORITY_LEN);
}

static inline uint8_t find_best_ranged_spec(Player* p) {
    return find_best_available(p, GEAR_SLOT_WEAPON, RANGE_SPEC_PRIORITY, RANGE_SPEC_PRIORITY_LEN);
}

static inline uint8_t find_best_magic_spec(Player* p) {
    return find_best_available(p, GEAR_SLOT_WEAPON, MAGIC_SPEC_PRIORITY, MAGIC_SPEC_PRIORITY_LEN);
}

static inline int player_has_gmaul(Player* p) {
    return player_has_item_in_slot(p, GEAR_SLOT_WEAPON, ITEM_GRANITE_MAUL);
}

typedef struct {
    const uint8_t* items;
    int len;
} GearPriorityList;

typedef struct {
    GearPriorityList weapon;
    GearPriorityList shield;
    GearPriorityList body;
    GearPriorityList legs;
    GearPriorityList head;
    GearPriorityList cape;
    GearPriorityList neck;
    GearPriorityList ring;
    int shield_two_handed_aware;
} LoadoutPriorities;

#define GEAR_LIST(arr) { arr, (int)(sizeof(arr) / sizeof((arr)[0])) }

/* rows are positional in LOADOUT_* enum order (KEEP hole first, then MELEE,
   RANGE, MAGE, TANK, SPEC_MELEE, SPEC_RANGE, SPEC_MAGIC); columns are weapon,
   shield, body, legs, head, cape, neck, ring, shield_two_handed_aware.
   g++ on the trainer include path cannot compile designated initializers here. */
static const LoadoutPriorities LOADOUT_PRIORITIES[LOADOUT_GMAUL] = {
    { {NULL, 0}, {NULL, 0}, {NULL, 0}, {NULL, 0},
      {NULL, 0}, {NULL, 0}, {NULL, 0}, {NULL, 0}, 0 },
    { GEAR_LIST(MELEE_WEAPON_PRIORITY), GEAR_LIST(MELEE_SHIELD_PRIORITY),
      GEAR_LIST(TANK_BODY_PRIORITY), GEAR_LIST(TANK_LEGS_PRIORITY),
      GEAR_LIST(TANK_HEAD_PRIORITY), GEAR_LIST(MELEE_CAPE_PRIORITY),
      GEAR_LIST(MELEE_NECK_PRIORITY), GEAR_LIST(MELEE_RING_PRIORITY), 1 },
    { GEAR_LIST(RANGE_WEAPON_PRIORITY), GEAR_LIST(TANK_SHIELD_PRIORITY),
      GEAR_LIST(TANK_BODY_PRIORITY), GEAR_LIST(TANK_LEGS_PRIORITY),
      GEAR_LIST(TANK_HEAD_PRIORITY), GEAR_LIST(MAGE_CAPE_PRIORITY),
      GEAR_LIST(MELEE_NECK_PRIORITY), GEAR_LIST(MAGE_RING_PRIORITY), 0 },
    { GEAR_LIST(MAGE_WEAPON_PRIORITY), GEAR_LIST(MAGE_SHIELD_PRIORITY),
      GEAR_LIST(MAGE_BODY_PRIORITY), GEAR_LIST(MAGE_LEGS_PRIORITY),
      GEAR_LIST(MAGE_HEAD_PRIORITY), GEAR_LIST(MAGE_CAPE_PRIORITY),
      GEAR_LIST(MAGE_NECK_PRIORITY), GEAR_LIST(MAGE_RING_PRIORITY), 0 },
    { GEAR_LIST(MAGE_WEAPON_PRIORITY), GEAR_LIST(TANK_SHIELD_PRIORITY),
      GEAR_LIST(TANK_BODY_PRIORITY), GEAR_LIST(TANK_LEGS_PRIORITY),
      GEAR_LIST(TANK_HEAD_PRIORITY), GEAR_LIST(MAGE_CAPE_PRIORITY),
      GEAR_LIST(MELEE_NECK_PRIORITY), GEAR_LIST(MAGE_RING_PRIORITY), 0 },
    { GEAR_LIST(MELEE_SPEC_PRIORITY), GEAR_LIST(MELEE_SHIELD_PRIORITY),
      GEAR_LIST(TANK_BODY_PRIORITY), GEAR_LIST(TANK_LEGS_PRIORITY),
      GEAR_LIST(TANK_HEAD_PRIORITY), GEAR_LIST(MELEE_CAPE_PRIORITY),
      GEAR_LIST(MELEE_NECK_PRIORITY), GEAR_LIST(MELEE_RING_PRIORITY), 1 },
    { GEAR_LIST(RANGE_SPEC_PRIORITY), GEAR_LIST(TANK_SHIELD_PRIORITY),
      GEAR_LIST(TANK_BODY_PRIORITY), GEAR_LIST(TANK_LEGS_PRIORITY),
      GEAR_LIST(TANK_HEAD_PRIORITY), GEAR_LIST(MAGE_CAPE_PRIORITY),
      GEAR_LIST(MELEE_NECK_PRIORITY), GEAR_LIST(MELEE_RING_PRIORITY), 1 },
    { GEAR_LIST(MAGIC_SPEC_PRIORITY), GEAR_LIST(MAGE_SHIELD_PRIORITY),
      GEAR_LIST(MAGE_BODY_PRIORITY), GEAR_LIST(MAGE_LEGS_PRIORITY),
      GEAR_LIST(MAGE_HEAD_PRIORITY), GEAR_LIST(MAGE_CAPE_PRIORITY),
      GEAR_LIST(MAGE_NECK_PRIORITY), GEAR_LIST(MAGE_RING_PRIORITY), 0 },
};

/* out[] order must match DYNAMIC_GEAR_SLOTS: weapon, shield, body, legs, head,
   cape, neck, ring */
static inline void resolve_loadout(Player* p, int loadout, uint8_t out[NUM_DYNAMIC_GEAR_SLOTS]) {
    for (int i = 0; i < NUM_DYNAMIC_GEAR_SLOTS; i++) {
        out[i] = p->equipped[DYNAMIC_GEAR_SLOTS[i]];
    }

    if (loadout == LOADOUT_GMAUL) {
        out[0] = ITEM_GRANITE_MAUL;
        out[1] = osrs_suppress_shield_for_two_handed_weapon(out[0], out[1]);
        return;
    }
    if (loadout < LOADOUT_MELEE || loadout > LOADOUT_SPEC_MAGIC) {
        return;
    }

    const LoadoutPriorities* lp = &LOADOUT_PRIORITIES[loadout];

    uint8_t weapon = find_best_available(p, GEAR_SLOT_WEAPON, lp->weapon.items, lp->weapon.len);
    if (weapon != ITEM_NONE) out[0] = weapon;

    if (lp->shield_two_handed_aware && item_is_two_handed(out[0])) {
        out[1] = osrs_suppress_shield_for_two_handed_weapon(out[0], out[1]);
    } else {
        uint8_t shield = find_best_available(p, GEAR_SLOT_SHIELD, lp->shield.items, lp->shield.len);
        if (shield != ITEM_NONE) out[1] = shield;
    }

    uint8_t body = find_best_available(p, GEAR_SLOT_BODY, lp->body.items, lp->body.len);
    if (body != ITEM_NONE) out[2] = body;
    uint8_t legs = find_best_available(p, GEAR_SLOT_LEGS, lp->legs.items, lp->legs.len);
    if (legs != ITEM_NONE) out[3] = legs;
    uint8_t head = find_best_available(p, GEAR_SLOT_HEAD, lp->head.items, lp->head.len);
    if (head != ITEM_NONE) out[4] = head;
    uint8_t cape = find_best_available(p, GEAR_SLOT_CAPE, lp->cape.items, lp->cape.len);
    if (cape != ITEM_NONE) out[5] = cape;
    uint8_t neck = find_best_available(p, GEAR_SLOT_NECK, lp->neck.items, lp->neck.len);
    if (neck != ITEM_NONE) out[6] = neck;
    uint8_t ring = find_best_available(p, GEAR_SLOT_RING, lp->ring.items, lp->ring.len);
    if (ring != ITEM_NONE) out[7] = ring;
}

static inline int apply_loadout(Player* p, int loadout) {
    if (loadout <= LOADOUT_KEEP || loadout > LOADOUT_GMAUL) return 0;

    uint8_t resolved[NUM_DYNAMIC_GEAR_SLOTS];
    resolve_loadout(p, loadout, resolved);

    int changed = 0;
    for (int i = 0; i < NUM_DYNAMIC_GEAR_SLOTS; i++) {
        int gear_slot = DYNAMIC_GEAR_SLOTS[i];
        changed += slot_equip_item(p, gear_slot, resolved[i]);
    }

    return changed;
}

static inline int is_loadout_active(Player* p, int loadout) {
    if (loadout <= LOADOUT_KEEP || loadout > LOADOUT_GMAUL) return 0;

    uint8_t resolved[NUM_DYNAMIC_GEAR_SLOTS];
    resolve_loadout(p, loadout, resolved);

    for (int i = 0; i < NUM_DYNAMIC_GEAR_SLOTS; i++) {
        int gear_slot = DYNAMIC_GEAR_SLOTS[i];
        if (p->equipped[gear_slot] != resolved[i]) return 0;
    }
    return 1;
}

static inline int get_current_loadout(Player* p) {
    for (int l = 1; l <= LOADOUT_GMAUL; l++) {
        if (is_loadout_active(p, l)) return l;
    }
    return 0;
}

static inline AttackStyle get_slot_weapon_attack_style(Player* p) {
    uint8_t weapon = p->equipped[GEAR_SLOT_WEAPON];
    if (weapon >= NUM_ITEMS) return ATTACK_STYLE_NONE;
    return (AttackStyle)get_item_attack_style(weapon);
}

static inline void init_slot_equipment_lms(Player* p) {
    memset(p->inventory, ITEM_NONE, sizeof(p->inventory));
    memset(p->num_items_in_slot, 0, sizeof(p->num_items_in_slot));

    p->equipped[GEAR_SLOT_HEAD] = ITEM_HELM_NEITIZNOT;
    p->equipped[GEAR_SLOT_CAPE] = ITEM_GOD_CAPE;
    p->equipped[GEAR_SLOT_NECK] = ITEM_GLORY;
    p->equipped[GEAR_SLOT_AMMO] = ITEM_DIAMOND_BOLTS_E;
    p->equipped[GEAR_SLOT_WEAPON] = ITEM_WHIP;
    p->equipped[GEAR_SLOT_SHIELD] = ITEM_DRAGON_DEFENDER;
    p->equipped[GEAR_SLOT_BODY] = ITEM_BLACK_DHIDE_BODY;
    p->equipped[GEAR_SLOT_LEGS] = ITEM_RUNE_PLATELEGS;
    p->equipped[GEAR_SLOT_HANDS] = ITEM_BARROWS_GLOVES;
    p->equipped[GEAR_SLOT_FEET] = ITEM_CLIMBING_BOOTS;
    p->equipped[GEAR_SLOT_RING] = ITEM_BERSERKER_RING;
    update_spec_weapons_for_weapon(p, p->equipped[GEAR_SLOT_WEAPON]);

    p->inventory[GEAR_SLOT_HEAD][0] = ITEM_HELM_NEITIZNOT;
    p->num_items_in_slot[GEAR_SLOT_HEAD] = 1;

    p->inventory[GEAR_SLOT_CAPE][0] = ITEM_GOD_CAPE;
    p->num_items_in_slot[GEAR_SLOT_CAPE] = 1;

    p->inventory[GEAR_SLOT_NECK][0] = ITEM_GLORY;
    p->num_items_in_slot[GEAR_SLOT_NECK] = 1;

    p->inventory[GEAR_SLOT_AMMO][0] = ITEM_DIAMOND_BOLTS_E;
    p->num_items_in_slot[GEAR_SLOT_AMMO] = 1;

    p->inventory[GEAR_SLOT_WEAPON][0] = ITEM_WHIP;
    p->inventory[GEAR_SLOT_WEAPON][1] = ITEM_RUNE_CROSSBOW;
    p->inventory[GEAR_SLOT_WEAPON][2] = ITEM_AHRIM_STAFF;
    p->inventory[GEAR_SLOT_WEAPON][3] = ITEM_DRAGON_DAGGER;
    p->num_items_in_slot[GEAR_SLOT_WEAPON] = 4;

    p->inventory[GEAR_SLOT_SHIELD][0] = ITEM_DRAGON_DEFENDER;
    p->inventory[GEAR_SLOT_SHIELD][1] = ITEM_SPIRIT_SHIELD;
    p->num_items_in_slot[GEAR_SLOT_SHIELD] = 2;

    p->inventory[GEAR_SLOT_BODY][0] = ITEM_BLACK_DHIDE_BODY;
    p->inventory[GEAR_SLOT_BODY][1] = ITEM_MYSTIC_TOP;
    p->num_items_in_slot[GEAR_SLOT_BODY] = 2;

    p->inventory[GEAR_SLOT_LEGS][0] = ITEM_RUNE_PLATELEGS;
    p->inventory[GEAR_SLOT_LEGS][1] = ITEM_MYSTIC_BOTTOM;
    p->num_items_in_slot[GEAR_SLOT_LEGS] = 2;

    p->inventory[GEAR_SLOT_HANDS][0] = ITEM_BARROWS_GLOVES;
    p->num_items_in_slot[GEAR_SLOT_HANDS] = 1;

    p->inventory[GEAR_SLOT_FEET][0] = ITEM_CLIMBING_BOOTS;
    p->num_items_in_slot[GEAR_SLOT_FEET] = 1;

    p->inventory[GEAR_SLOT_RING][0] = ITEM_BERSERKER_RING;
    p->num_items_in_slot[GEAR_SLOT_RING] = 1;

    osrs_refresh_player_equipment(p);
    p->current_gear = GEAR_MELEE;
}

static inline int add_item_to_inventory(Player* p, int gear_slot, uint8_t item_idx) {
    if (gear_slot < 0 || gear_slot >= NUM_GEAR_SLOTS) return 0;
    if (p->num_items_in_slot[gear_slot] >= MAX_ITEMS_PER_SLOT) return 0;

    for (int i = 0; i < p->num_items_in_slot[gear_slot]; i++) {
        if (p->inventory[gear_slot][i] == item_idx) return 0;
    }

    p->inventory[gear_slot][p->num_items_in_slot[gear_slot]] = item_idx;
    p->num_items_in_slot[gear_slot]++;
    return 1;
}

static const uint8_t UPGRADE_REPLACES[NUM_ITEMS] = {
    [ITEM_HELM_NEITIZNOT]       = ITEM_NONE,
    [ITEM_GOD_CAPE]             = ITEM_NONE,
    [ITEM_GLORY]                = ITEM_NONE,
    [ITEM_BLACK_DHIDE_BODY]     = ITEM_NONE,
    [ITEM_MYSTIC_TOP]           = ITEM_NONE,
    [ITEM_RUNE_PLATELEGS]       = ITEM_NONE,
    [ITEM_MYSTIC_BOTTOM]        = ITEM_NONE,
    [ITEM_WHIP]                 = ITEM_NONE,
    [ITEM_RUNE_CROSSBOW]        = ITEM_NONE,
    [ITEM_AHRIM_STAFF]          = ITEM_NONE,
    [ITEM_DRAGON_DAGGER]        = ITEM_NONE,
    [ITEM_DRAGON_DEFENDER]      = ITEM_NONE,
    [ITEM_SPIRIT_SHIELD]        = ITEM_NONE,
    [ITEM_BARROWS_GLOVES]       = ITEM_NONE,
    [ITEM_CLIMBING_BOOTS]       = ITEM_NONE,
    [ITEM_BERSERKER_RING]       = ITEM_NONE,
    [ITEM_DIAMOND_BOLTS_E]      = ITEM_NONE,
    [ITEM_GHRAZI_RAPIER]        = ITEM_WHIP,
    [ITEM_INQUISITORS_MACE]     = ITEM_WHIP,
    [ITEM_STAFF_OF_DEAD]        = ITEM_AHRIM_STAFF,
    [ITEM_KODAI_WAND]           = ITEM_AHRIM_STAFF,
    [ITEM_VOLATILE_STAFF]       = ITEM_AHRIM_STAFF,
    [ITEM_ZURIELS_STAFF]        = ITEM_AHRIM_STAFF,
    [ITEM_ARMADYL_CROSSBOW]     = ITEM_RUNE_CROSSBOW,
    [ITEM_ZARYTE_CROSSBOW]      = ITEM_RUNE_CROSSBOW,
    [ITEM_DRAGON_CLAWS]         = ITEM_DRAGON_DAGGER,
    [ITEM_AGS]                  = ITEM_DRAGON_DAGGER,
    [ITEM_ANCIENT_GS]           = ITEM_DRAGON_DAGGER,
    [ITEM_GRANITE_MAUL]         = ITEM_NONE,
    [ITEM_ELDER_MAUL]           = ITEM_WHIP,
    [ITEM_DARK_BOW]             = ITEM_NONE,
    [ITEM_HEAVY_BALLISTA]       = ITEM_NONE,
    [ITEM_VESTAS]               = ITEM_DRAGON_DAGGER,
    [ITEM_VOIDWAKER]            = ITEM_DRAGON_DAGGER,
    [ITEM_STATIUS_WARHAMMER]    = ITEM_DRAGON_DAGGER,
    [ITEM_MORRIGANS_JAVELIN]    = ITEM_RUNE_CROSSBOW,
    [ITEM_ANCESTRAL_HAT]        = ITEM_NONE,
    [ITEM_ANCESTRAL_TOP]        = ITEM_MYSTIC_TOP,
    [ITEM_ANCESTRAL_BOTTOM]     = ITEM_MYSTIC_BOTTOM,
    [ITEM_AHRIMS_ROBETOP]       = ITEM_MYSTIC_TOP,
    [ITEM_AHRIMS_ROBESKIRT]     = ITEM_MYSTIC_BOTTOM,
    [ITEM_KARILS_TOP]           = ITEM_BLACK_DHIDE_BODY,
    [ITEM_BANDOS_TASSETS]       = ITEM_RUNE_PLATELEGS,
    [ITEM_BLESSED_SPIRIT_SHIELD]= ITEM_SPIRIT_SHIELD,
    [ITEM_FURY]                 = ITEM_GLORY,
    [ITEM_OCCULT_NECKLACE]      = ITEM_NONE,
    [ITEM_INFERNAL_CAPE]        = ITEM_NONE,
    [ITEM_ETERNAL_BOOTS]        = ITEM_CLIMBING_BOOTS,
    [ITEM_SEERS_RING_I]         = ITEM_NONE,
    [ITEM_LIGHTBEARER]          = ITEM_NONE,
    [ITEM_MAGES_BOOK]           = ITEM_NONE,
    [ITEM_DRAGON_ARROWS]        = ITEM_NONE,
    [ITEM_TORAGS_PLATELEGS]     = ITEM_RUNE_PLATELEGS,
    [ITEM_DHAROKS_PLATELEGS]    = ITEM_RUNE_PLATELEGS,
    [ITEM_VERACS_PLATESKIRT]    = ITEM_RUNE_PLATELEGS,
    [ITEM_TORAGS_HELM]          = ITEM_HELM_NEITIZNOT,
    [ITEM_DHAROKS_HELM]         = ITEM_HELM_NEITIZNOT,
    [ITEM_VERACS_HELM]          = ITEM_HELM_NEITIZNOT,
    [ITEM_GUTHANS_HELM]         = ITEM_HELM_NEITIZNOT,
    [ITEM_OPAL_DRAGON_BOLTS]    = ITEM_NONE,
};

static inline int remove_item_from_inventory(Player* p, int gear_slot, uint8_t item_idx) {
    for (int i = 0; i < p->num_items_in_slot[gear_slot]; i++) {
        if (p->inventory[gear_slot][i] == item_idx) {
            for (int j = i; j < p->num_items_in_slot[gear_slot] - 1; j++) {
                p->inventory[gear_slot][j] = p->inventory[gear_slot][j + 1];
            }
            p->num_items_in_slot[gear_slot]--;
            p->inventory[gear_slot][p->num_items_in_slot[gear_slot]] = ITEM_NONE;
            return 1;
        }
    }
    return 0;
}

static inline int item_to_gear_slot(uint8_t item_idx) {
    return osrs_item_gear_slot(item_idx);
}

static const uint8_t CHAIN_REPLACES[][2] = {
    { ITEM_VESTAS, ITEM_WHIP },
    { ITEM_ZURIELS_STAFF, ITEM_STAFF_OF_DEAD },
    { ITEM_ZURIELS_STAFF, ITEM_VOLATILE_STAFF },
    { ITEM_KODAI_WAND, ITEM_STAFF_OF_DEAD },
    { ITEM_KODAI_WAND, ITEM_VOLATILE_STAFF },
    { ITEM_KODAI_WAND, ITEM_ZURIELS_STAFF },
    { ITEM_VOLATILE_STAFF, ITEM_STAFF_OF_DEAD },
    { ITEM_ZARYTE_CROSSBOW, ITEM_ARMADYL_CROSSBOW },
    { ITEM_MORRIGANS_JAVELIN, ITEM_ZARYTE_CROSSBOW },
    { ITEM_MORRIGANS_JAVELIN, ITEM_ARMADYL_CROSSBOW },
    { ITEM_MORRIGANS_JAVELIN, ITEM_HEAVY_BALLISTA },
    { ITEM_MORRIGANS_JAVELIN, ITEM_DARK_BOW },
    { ITEM_ZARYTE_CROSSBOW, ITEM_HEAVY_BALLISTA },
    { ITEM_ZARYTE_CROSSBOW, ITEM_DARK_BOW },
    { ITEM_ARMADYL_CROSSBOW, ITEM_HEAVY_BALLISTA },
    { ITEM_ARMADYL_CROSSBOW, ITEM_DARK_BOW },
    { ITEM_ANCESTRAL_TOP, ITEM_AHRIMS_ROBETOP },
    { ITEM_ANCESTRAL_BOTTOM, ITEM_AHRIMS_ROBESKIRT },
    { ITEM_BANDOS_TASSETS, ITEM_TORAGS_PLATELEGS },
    { ITEM_BANDOS_TASSETS, ITEM_DHAROKS_PLATELEGS },
    { ITEM_BANDOS_TASSETS, ITEM_VERACS_PLATESKIRT },
    { ITEM_GHRAZI_RAPIER, ITEM_INQUISITORS_MACE },
    { ITEM_GHRAZI_RAPIER, ITEM_WHIP },
    { ITEM_INQUISITORS_MACE, ITEM_WHIP },
    { ITEM_ELDER_MAUL, ITEM_WHIP },
    { ITEM_GHRAZI_RAPIER, ITEM_ELDER_MAUL },
    { ITEM_INQUISITORS_MACE, ITEM_ELDER_MAUL },
    { ITEM_VESTAS, ITEM_ELDER_MAUL },
    { ITEM_VESTAS, ITEM_GHRAZI_RAPIER },
    { ITEM_VESTAS, ITEM_INQUISITORS_MACE },
    { ITEM_VOIDWAKER, ITEM_WHIP },
    { ITEM_VOIDWAKER, ITEM_GHRAZI_RAPIER },
    { ITEM_VOIDWAKER, ITEM_INQUISITORS_MACE },
    { ITEM_VOIDWAKER, ITEM_ELDER_MAUL },
    { ITEM_STATIUS_WARHAMMER, ITEM_WHIP },
    { ITEM_STATIUS_WARHAMMER, ITEM_GHRAZI_RAPIER },
    { ITEM_STATIUS_WARHAMMER, ITEM_INQUISITORS_MACE },
    { ITEM_STATIUS_WARHAMMER, ITEM_ELDER_MAUL },
    { ITEM_STATIUS_WARHAMMER, ITEM_AGS },
    { ITEM_STATIUS_WARHAMMER, ITEM_ANCIENT_GS },
    { ITEM_STATIUS_WARHAMMER, ITEM_DRAGON_CLAWS },
    { ITEM_AGS, ITEM_WHIP },
    { ITEM_ANCIENT_GS, ITEM_WHIP },
    { ITEM_ANCIENT_GS, ITEM_AGS },
    { ITEM_ANCIENT_GS, ITEM_DRAGON_CLAWS },
    { ITEM_AGS, ITEM_DRAGON_CLAWS },
    { ITEM_LIGHTBEARER, ITEM_SEERS_RING_I },
    { ITEM_TORAGS_HELM, ITEM_GUTHANS_HELM },
    { ITEM_TORAGS_HELM, ITEM_VERACS_HELM },
    { ITEM_TORAGS_HELM, ITEM_DHAROKS_HELM },
    { ITEM_GUTHANS_HELM, ITEM_VERACS_HELM },
    { ITEM_GUTHANS_HELM, ITEM_DHAROKS_HELM },
    { ITEM_VERACS_HELM, ITEM_DHAROKS_HELM },
};
#define CHAIN_REPLACES_LEN (sizeof(CHAIN_REPLACES) / sizeof(CHAIN_REPLACES[0]))

static inline void add_loot_item(Player* p, uint8_t item_idx) {
    int gear_slot = item_to_gear_slot(item_idx);
    if (gear_slot < 0) return;

    for (int i = 0; i < (int)CHAIN_REPLACES_LEN; i++) {
        if (CHAIN_REPLACES[i][1] == item_idx) {
            uint8_t better = CHAIN_REPLACES[i][0];
            int better_slot = item_to_gear_slot(better);
            if (better_slot >= 0 && player_has_item_in_slot(p, better_slot, better)) {
                return;
            }
        }
    }

    uint8_t replaces = UPGRADE_REPLACES[item_idx];
    if (replaces != ITEM_NONE) {
        int replace_slot = item_to_gear_slot(replaces);
        if (replace_slot >= 0) {
            remove_item_from_inventory(p, replace_slot, replaces);
        }
    }

    for (int i = 0; i < (int)CHAIN_REPLACES_LEN; i++) {
        if (CHAIN_REPLACES[i][0] == item_idx) {
            uint8_t obsolete = CHAIN_REPLACES[i][1];
            int obs_slot = item_to_gear_slot(obsolete);
            if (obs_slot >= 0) {
                remove_item_from_inventory(p, obs_slot, obsolete);
            }
        }
    }

    add_item_to_inventory(p, gear_slot, item_idx);

    if ((item_idx == ITEM_ARMADYL_CROSSBOW || item_idx == ITEM_ZARYTE_CROSSBOW)
        && player_has_item_in_slot(p, GEAR_SLOT_AMMO, ITEM_OPAL_DRAGON_BOLTS)) {
        remove_item_from_inventory(p, GEAR_SLOT_AMMO, ITEM_DIAMOND_BOLTS_E);
        p->equipped[GEAR_SLOT_AMMO] = ITEM_OPAL_DRAGON_BOLTS;
    }

}

/* 4 brew + 2 restore + 1 combat + 1 ranged + 2 karambwan + 1 rune pouch */
#define FIXED_INVENTORY_SLOTS 11

static inline int count_switch_items(Player* p) {
    int switches = 0;
    for (int s = 0; s < NUM_GEAR_SLOTS; s++) {
        if (p->num_items_in_slot[s] > 1) {
            switches += p->num_items_in_slot[s] - 1;
        }
    }
    return switches;
}

static inline int compute_food_count(Player* p) {
    int switches = count_switch_items(p);
    int food = 28 - FIXED_INVENTORY_SLOTS - switches;
    return food > 1 ? food : 1;
}

static const uint8_t CHEST_LOOT[] = {
    ITEM_DRAGON_CLAWS, ITEM_AGS, ITEM_ANCIENT_GS, ITEM_GRANITE_MAUL,
    ITEM_VOLATILE_STAFF, ITEM_ZARYTE_CROSSBOW, ITEM_ARMADYL_CROSSBOW,
    ITEM_DARK_BOW, ITEM_GHRAZI_RAPIER, ITEM_INQUISITORS_MACE,
    ITEM_KODAI_WAND, ITEM_STAFF_OF_DEAD, ITEM_ELDER_MAUL,
    ITEM_HEAVY_BALLISTA, ITEM_OCCULT_NECKLACE, ITEM_INFERNAL_CAPE,
    ITEM_SEERS_RING_I, ITEM_MAGES_BOOK,
    ITEM_ANCESTRAL_HAT, ITEM_ANCESTRAL_TOP, ITEM_ANCESTRAL_BOTTOM,
    ITEM_AHRIMS_ROBETOP, ITEM_AHRIMS_ROBESKIRT, ITEM_KARILS_TOP,
    ITEM_BANDOS_TASSETS, ITEM_BLESSED_SPIRIT_SHIELD,
    ITEM_FURY, ITEM_ETERNAL_BOOTS,
    ITEM_TORAGS_PLATELEGS, ITEM_DHAROKS_PLATELEGS, ITEM_VERACS_PLATESKIRT,
    ITEM_TORAGS_HELM, ITEM_DHAROKS_HELM, ITEM_VERACS_HELM, ITEM_GUTHANS_HELM,
    ITEM_OPAL_DRAGON_BOLTS,
};
#define CHEST_LOOT_LEN (sizeof(CHEST_LOOT) / sizeof(CHEST_LOOT[0]))

static const uint8_t BLOODIER_LOOT[] = {
    ITEM_VESTAS, ITEM_VOIDWAKER, ITEM_STATIUS_WARHAMMER,
    ITEM_MORRIGANS_JAVELIN, ITEM_ZURIELS_STAFF, ITEM_LIGHTBEARER
};
#define BLOODIER_LOOT_LEN (sizeof(BLOODIER_LOOT) / sizeof(BLOODIER_LOOT[0]))

/* roll counts per tier pin the seeded RNG call order: tier 1 adds 2 chest rolls,
   tier 2 adds 4 more, tier 3 adds 2 more plus 1 bloodier roll */
static inline void init_player_gear_randomized(Player* p, int tier, uint32_t* rng) {
    init_slot_equipment_lms(p);

    if (tier <= 0) return;

    #define ADD_RANDOM_LOOT(table, len) do { \
        uint32_t _r = xorshift32(rng); \
        uint8_t _item = (table)[_r % (len)]; \
        add_loot_item(p, _item); \
    } while(0)

    if (tier >= 1) {
        ADD_RANDOM_LOOT(CHEST_LOOT, CHEST_LOOT_LEN);
        ADD_RANDOM_LOOT(CHEST_LOOT, CHEST_LOOT_LEN);
    }

    if (tier >= 2) {
        ADD_RANDOM_LOOT(CHEST_LOOT, CHEST_LOOT_LEN);
        ADD_RANDOM_LOOT(CHEST_LOOT, CHEST_LOOT_LEN);
        ADD_RANDOM_LOOT(CHEST_LOOT, CHEST_LOOT_LEN);
        ADD_RANDOM_LOOT(CHEST_LOOT, CHEST_LOOT_LEN);
    }

    if (tier >= 3) {
        ADD_RANDOM_LOOT(CHEST_LOOT, CHEST_LOOT_LEN);
        ADD_RANDOM_LOOT(CHEST_LOOT, CHEST_LOOT_LEN);
        ADD_RANDOM_LOOT(BLOODIER_LOOT, BLOODIER_LOOT_LEN);
    }

    #undef ADD_RANDOM_LOOT

    if (tier >= 3 && player_has_item_in_slot(p, GEAR_SLOT_SHIELD, ITEM_DRAGON_DEFENDER)) {
        int has_1h_melee = 0;
        for (int i = 0; i < p->num_items_in_slot[GEAR_SLOT_WEAPON]; i++) {
            uint8_t w = p->inventory[GEAR_SLOT_WEAPON][i];
            if (get_item_attack_style(w) == ATTACK_STYLE_MELEE && !item_is_two_handed(w)) {
                has_1h_melee = 1;
                break;
            }
        }
        if (!has_1h_melee) {
            remove_item_from_inventory(p, GEAR_SLOT_SHIELD, ITEM_DRAGON_DEFENDER);
        }
    }

    uint8_t resolved[NUM_DYNAMIC_GEAR_SLOTS];
    resolve_loadout(p, LOADOUT_MELEE, resolved);
    for (int i = 0; i < NUM_DYNAMIC_GEAR_SLOTS; i++) {
        slot_equip_item(p, DYNAMIC_GEAR_SLOTS[i], resolved[i]);
    }

    osrs_refresh_player_equipment(p);
    p->current_gear = GEAR_MELEE;
}

static inline int sample_gear_tier(float weights[4], uint32_t* rng) {
    float r = (float)xorshift32(rng) / (float)UINT32_MAX;
    float cumulative = 0.0f;
    for (int i = 0; i < 4; i++) {
        cumulative += weights[i];
        if (r < cumulative) return i;
    }
    return 0;
}

#endif // OSRS_PVP_GEAR_H
