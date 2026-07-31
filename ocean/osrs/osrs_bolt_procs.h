#ifndef OSRS_BOLT_PROCS_H
#define OSRS_BOLT_PROCS_H

#include "osrs_combat.h"
#include "osrs_items.h"

typedef struct {
    int proc_triggered;
    int modified_damage;
} BoltProcResult;

static inline BoltProcResult osrs_resolve_bolt_proc(
    int bolt_item_idx, int base_damage, int hit_accurate,
    int max_hit, int ranged_level, int target_current_hp,
    int is_zcb_spec, uint32_t* rng_state
) {
    BoltProcResult r = { 0, base_damage };

    switch (bolt_item_idx) {

    case ITEM_DIAMOND_BOLTS_E:
    case ITEM_DIAMOND_DRAGON_BOLTS_E: {
        if (!hit_accurate && !is_zcb_spec) break;
        if (is_zcb_spec || encounter_roll_ratio_u16(rng_state, 11, 100)) {
            int effect_max = max_hit * (is_zcb_spec ? 126 : 115) / 100;
            r.proc_triggered = 1;
            r.modified_damage = encounter_rand_int(rng_state, effect_max + 1);
        }
        break;
    }

    case ITEM_OPAL_DRAGON_BOLTS: {
        if (is_zcb_spec || encounter_roll_ratio_u16(rng_state, 11, 200)) {
            int bonus = ranged_level / (is_zcb_spec ? 9 : 10);
            r.proc_triggered = 1;
            r.modified_damage = base_damage + bonus;
        }
        break;
    }

    case ITEM_RUBY_DRAGON_BOLTS_E: {
        if (!hit_accurate) break;
        if (is_zcb_spec || encounter_roll_ratio_u16(rng_state, 33, 500)) {
            int cap = is_zcb_spec ? 110 : 100;
            int effect_dmg = target_current_hp * (is_zcb_spec ? 22 : 20) / 100;
            if (effect_dmg > cap) effect_dmg = cap;
            r.proc_triggered = 1;
            r.modified_damage = effect_dmg;
        }
        break;
    }

    default:
        break;
    }

    return r;
}

#endif
