#ifndef OSRS_COMBAT_H
#define OSRS_COMBAT_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "osrs_types.h"
#include "osrs_items.h"

/** Standard OSRS accuracy formula over pre-computed rolls
    (eff_level * (bonus + 64)); returns hit probability in [0, 1]. */
static inline float osrs_hit_chance(int att_roll, int def_roll) {
    if (att_roll > def_roll)
        return 1.0f - (float)(def_roll + 2) / (2.0f * (float)(att_roll + 1));
    else
        return (float)att_roll / (2.0f * (float)(def_roll + 1));
}

/** Twisted bow accuracy multiplier; target_magic =
    min(max(npc_magic_level, npc_magic_attack_bonus), 250). */
static inline float osrs_tbow_acc_mult(int target_magic) {
    int m = target_magic < 250 ? target_magic : 250;
    float lin = (float)(3 * m);
    float quad = lin / 10.0f;
    float mult = (140.0f + (lin - 10.0f) / 100.0f - (quad - 100.0f) * (quad - 100.0f) / 100.0f) / 100.0f;
    if (mult > 1.4f) mult = 1.4f;
    if (mult < 0.0f) mult = 0.0f;
    return mult;
}

/** Twisted bow damage multiplier; same input as osrs_tbow_acc_mult. */
static inline float osrs_tbow_dmg_mult(int target_magic) {
    int m = target_magic < 250 ? target_magic : 250;
    float lin = (float)(3 * m);
    float quad = lin / 10.0f;
    float mult = (250.0f + (lin - 14.0f) / 100.0f - (quad - 140.0f) * (quad - 140.0f) / 100.0f) / 100.0f;
    if (mult > 2.5f) mult = 2.5f;
    if (mult < 0.0f) mult = 0.0f;
    return mult;
}

static inline int encounter_rand_int(uint32_t* rng_state, int max) {
    if (max <= 0) return 0;
    return (int)(xorshift32(rng_state) % (unsigned)max);
}

static inline float encounter_rand_float(uint32_t* rng_state) {
    return (float)(xorshift32(rng_state) & 0xFFFF) / 65536.0f;
}

static inline int encounter_roll_ratio_u16(
    uint32_t* rng_state,
    uint64_t numerator,
    uint64_t denominator
) {
    uint32_t roll = xorshift32(rng_state) & 0xFFFFu;
    if (denominator == 0 || numerator == 0) return 0;
    if (numerator >= denominator) return 1;
    return (uint64_t)roll * denominator < numerator * 65536ull;
}

static inline void osrs_hit_chance_fraction(
    int att_roll,
    int def_roll,
    uint64_t* numerator,
    uint64_t* denominator
) {
    if (att_roll > def_roll) {
        *numerator = (uint64_t)(2 * att_roll - def_roll);
        *denominator = (uint64_t)(2 * (att_roll + 1));
    } else {
        *numerator = (uint64_t)att_roll;
        *denominator = (uint64_t)(2 * (def_roll + 1));
    }
}

static inline int encounter_roll_hit_chance(
    uint32_t* rng_state,
    int att_roll,
    int def_roll
) {
    uint64_t numerator, denominator;
    osrs_hit_chance_fraction(att_roll, def_roll, &numerator, &denominator);
    return encounter_roll_ratio_u16(rng_state, numerator, denominator);
}

static inline void osrs_hit_chance_double_fraction(
    int att_roll,
    int def_roll,
    uint64_t* numerator,
    uint64_t* denominator
) {
    uint64_t a = (uint64_t)att_roll;
    uint64_t d = (uint64_t)def_roll;
    if (att_roll >= def_roll) {
        uint64_t miss_num = (d + 2ull) * (2ull * d + 3ull);
        uint64_t den = 6ull * (a + 1ull) * (a + 1ull);
        *numerator = den > miss_num ? den - miss_num : 0;
        *denominator = den;
    } else {
        *numerator = a * (4ull * a + 5ull);
        *denominator = 6ull * (a + 1ull) * (d + 1ull);
    }
}

static inline int encounter_roll_hit_chance_double(
    uint32_t* rng_state,
    int att_roll,
    int def_roll
) {
    uint64_t numerator, denominator;
    osrs_hit_chance_double_fraction(att_roll, def_roll, &numerator, &denominator);
    return encounter_roll_ratio_u16(rng_state, numerator, denominator);
}

#define ENCOUNTER_SPELL_NONE  0
#define ENCOUNTER_SPELL_ICE   1
#define ENCOUNTER_SPELL_BLOOD 2

#define BARRAGE_MAX_HITS 9
#define BARRAGE_FREEZE_TICKS 32

typedef struct {
    int active;          /* in: 1 if this target slot is valid */
    int x, y;            /* in: NPC SW corner tile */
    int magic_level;     /* in: magic rolls vs magic level, not defence */
    int magic_def_bonus; /* in */
    int npc_idx;         /* in: index into the caller's NPC array */
    int* frozen_ticks;   /* in: NULL = no freeze tracking */
    int rolled;          /* out */
    int hit;             /* out */
    int damage;          /* out: 0 if splashed */
} BarrageTarget;

typedef struct {
    int total_damage;
    int num_hits;
    int num_successful;
} BarrageResult;

/** Barrage vs targets[0] plus AoE over active targets within 1 tile of it.
    ICE freezes at cast time through each target's frozen_ticks pointer; the
    caller still queues the returned damage as delayed pending hits. */
static inline BarrageResult osrs_barrage_resolve(
    BarrageTarget* targets, int max_targets,
    int att_roll, int max_hit, uint32_t* rng_state,
    int spell_type,
    int primary_use_double_accuracy
) {
    BarrageResult result = { 0, 0, 0 };

    if (max_targets < 1 || !targets[0].active) return result;

    int px = targets[0].x, py = targets[0].y;
    {
        int def_roll = (targets[0].magic_level + 9) * (targets[0].magic_def_bonus + 64);
        targets[0].rolled = 1;
        targets[0].hit = primary_use_double_accuracy
            ? encounter_roll_hit_chance_double(rng_state, att_roll, def_roll)
            : encounter_roll_hit_chance(rng_state, att_roll, def_roll);
        targets[0].damage = targets[0].hit ? encounter_rand_int(rng_state, max_hit + 1) : 0;
        result.total_damage += targets[0].damage;
        result.num_hits++;
        if (targets[0].hit) {
            result.num_successful++;
            if (spell_type == ENCOUNTER_SPELL_ICE && targets[0].frozen_ticks)
                *targets[0].frozen_ticks = BARRAGE_FREEZE_TICKS;
        }
    }

    for (int i = 1; i < max_targets && result.num_hits < BARRAGE_MAX_HITS; i++) {
        if (!targets[i].active) continue;
        int dx = targets[i].x - px;
        int dy = targets[i].y - py;
        if (dx < -1 || dx > 1 || dy < -1 || dy > 1) continue;

        int def_roll = (targets[i].magic_level + 9) * (targets[i].magic_def_bonus + 64);
        targets[i].rolled = 1;
        targets[i].hit = encounter_roll_hit_chance(rng_state, att_roll, def_roll);
        targets[i].damage = targets[i].hit ? encounter_rand_int(rng_state, max_hit + 1) : 0;
        result.total_damage += targets[i].damage;
        result.num_hits++;
        if (targets[i].hit) {
            result.num_successful++;
            if (spell_type == ENCOUNTER_SPELL_ICE && targets[i].frozen_ticks)
                *targets[i].frozen_ticks = BARRAGE_FREEZE_TICKS;
        }
    }

    return result;
}

static inline int osrs_npc_melee_max_hit(int str_level, int melee_str_bonus) {
    return ((str_level + 9) * (melee_str_bonus + 64) + 320) / 640;
}

static inline int osrs_npc_ranged_max_hit(int range_level, int ranged_str_bonus) {
    return (int)(0.5 + (double)(range_level + 9) * (ranged_str_bonus + 64) / 640.0);
}

/** magic_dmg_pct is percent: 100 = 1.0x, 175 = 1.75x. */
static inline int osrs_npc_magic_max_hit(int base_spell_dmg, int magic_dmg_pct) {
    return base_spell_dmg * magic_dmg_pct / 100;
}

static inline int osrs_npc_attack_roll(int att_level, int att_bonus) {
    return (att_level + 9) * (att_bonus + 64);
}

/** NPC defence roll the player attacks into; the caller passes the level the
    style rolls against (drain-adjusted Defence for melee/ranged, Magic for magic). */
static inline int osrs_npc_def_roll(int def_level, int def_bonus) {
    return (def_level + 9) * (def_bonus + 64);
}

static inline int encounter_npc_melee_def_bonus(
    int stab_def, int slash_def, int crush_def, int melee_style
) {
    if (melee_style == 1) return slash_def;
    if (melee_style == 2) return crush_def;
    return stab_def;
}

/** Style-dispatched NPC defence roll: a monster's MAGIC defence rolls off its
    Magic level, not its Defence level, so the caller passes both. */
static inline int encounter_npc_target_def_roll(
    int melee_ranged_def_level,
    int magic_level,
    int stab_def,
    int slash_def,
    int crush_def,
    int magic_def_bonus,
    int ranged_def_bonus,
    int attack_style,
    int melee_style
) {
    if (attack_style == 3)
        return osrs_npc_def_roll(magic_level, magic_def_bonus);
    if (attack_style == 2)
        return osrs_npc_def_roll(melee_ranged_def_level, ranged_def_bonus);
    return osrs_npc_def_roll(
        melee_ranged_def_level,
        encounter_npc_melee_def_bonus(stab_def, slash_def, crush_def, melee_style));
}

/** Player defence roll vs an NPC attack: players get +8, not the NPC +9. */
static inline int osrs_player_def_roll_vs_npc(
    int def_level, int magic_level, int def_bonus, int attack_style
) {
    int eff_def;
    if (attack_style == 3) {
        eff_def = (int)(magic_level * 0.7 + def_level * 0.3) + 8;
    } else {
        eff_def = def_level + 8;
    }
    return eff_def * (def_bonus + 64);
}

static inline int encounter_player_def_bonus(
    int def_stab, int def_slash, int def_crush, int def_magic, int def_ranged,
    int attack_style, int melee_style
) {
    if (attack_style == 2) return def_ranged;
    if (attack_style == 3) return def_magic;
    if (melee_style == 1) return def_slash;
    if (melee_style == 2) return def_crush;
    return def_stab;
}

/** Loadout defence bonus for the incoming style, then the OSRS defence roll;
    callers apply any encounter-specific Defence adjustment before this. */
static inline int encounter_player_def_roll_from_loadout(
    int def_level,
    int magic_level,
    int def_stab,
    int def_slash,
    int def_crush,
    int def_magic,
    int def_ranged,
    int attack_style,
    int melee_style
) {
    int def_bonus = encounter_player_def_bonus(
        def_stab, def_slash, def_crush, def_magic, def_ranged,
        attack_style, melee_style);
    return osrs_player_def_roll_vs_npc(
        def_level, magic_level, def_bonus, attack_style);
}

static inline int osrs_npc_max_hit(
    int attack_style,
    int str_level, int range_level,
    int melee_str_bonus, int ranged_str_bonus,
    int magic_base_dmg, int magic_dmg_pct
) {
    if (attack_style == 1)
        return osrs_npc_melee_max_hit(str_level, melee_str_bonus);
    if (attack_style == 2)
        return osrs_npc_ranged_max_hit(range_level, ranged_str_bonus);
    if (attack_style == 3)
        return osrs_npc_magic_max_hit(magic_base_dmg, magic_dmg_pct);
    return 0;
}

/** Damage roll FIRST (0..max_hit), THEN the accuracy roll; this draw order is
    canonical for every encounter and goldens depend on it. force_hit SKIPS the
    accuracy draw entirely rather than ignoring it, for the same reason. */
static inline int encounter_npc_roll_attack_ex(
    int att_roll, int def_roll, int max_hit, int force_hit,
    uint32_t* rng_state, int* hit_success
) {
    int dmg = encounter_rand_int(rng_state, max_hit + 1);
    int hit = force_hit ? 1 : encounter_roll_hit_chance(rng_state, att_roll, def_roll);
    if (hit_success) *hit_success = hit;
    return hit ? dmg : 0;
}

static inline int encounter_npc_roll_attack(
    int att_roll, int def_roll, int max_hit, uint32_t* rng_state
) {
    return encounter_npc_roll_attack_ex(att_roll, def_roll, max_hit, 0, rng_state, NULL);
}

/** The prayer and style enums run in OPPOSITE orders (prayer MAGIC=1..MELEE=3,
    style MELEE=1..MAGIC=3), hence the crossed constants. */
static inline int encounter_prayer_correct_for_style(int prayer, int attack_style) {
    return (attack_style == 1 && prayer == 3) ||
           (attack_style == 2 && prayer == 2) ||
           (attack_style == 3 && prayer == 1);
}

/** Protect-prayer outcome and damage lock on the THROW tick: flicking after the
    throw cannot change a hit already in flight. Jad-style deferred checks are
    the exception (prayer_check_delay). */
typedef struct {
    int frozen_damage;
    int prayed;
} EncounterProtectResolve;

static inline EncounterProtectResolve encounter_resolve_protect_at_throw(
    int raw_damage, int overhead_prayer, int attack_style
) {
    int prayed = encounter_prayer_correct_for_style(overhead_prayer, attack_style);
    return (EncounterProtectResolve){ .frozen_damage = prayed ? 0 : raw_damage, .prayed = prayed };
}

static inline int encounter_magic_hit_delay(int distance, int is_player) {
    return (1 + distance) / 3 + 1 + (is_player ? 1 : 0);
}

static inline int encounter_ranged_hit_delay(int distance, int is_player) {
    return (3 + distance) / 6 + 1 + (is_player ? 1 : 0);
}

static inline int encounter_thrown_hit_delay(int distance, int is_player) {
    return distance / 6 + 1 + (is_player ? 1 : 0);
}

static inline int encounter_blowpipe_hit_delay(int distance, int is_player) {
    return encounter_thrown_hit_delay(distance, is_player);
}

static inline int encounter_ballista_hit_delay(int distance, int is_player) {
    return 2 + (1 + distance) / 6 + (is_player ? 1 : 0);
}

static inline int encounter_dark_bow_second_hit_delay(int distance, int is_player) {
    return 1 + (2 + distance) / 3 + (is_player ? 1 : 0);
}

static inline int encounter_eye_of_ayak_hit_delay(int distance) {
    return distance <= 2 ? 1 : 2;
}

typedef enum {
    ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE,
    ENCOUNTER_PROJECTILE_DISTANCE_TARGET_SW_TILE,
} EncounterProjectileDistanceMode;

typedef enum {
    ENCOUNTER_PROJECTILE_DELAY_MELEE,
    ENCOUNTER_PROJECTILE_DELAY_MAGIC,
    ENCOUNTER_PROJECTILE_DELAY_RANGED,
    ENCOUNTER_PROJECTILE_DELAY_THROWN,
    ENCOUNTER_PROJECTILE_DELAY_BALLISTA,
    ENCOUNTER_PROJECTILE_DELAY_DARK_BOW_SECOND,
    ENCOUNTER_PROJECTILE_DELAY_EYE_OF_AYAK,
} EncounterProjectileDelayKind;

typedef struct {
    int set_delay;
    int reduce_delay;
    int start_delay;
    int visual_delay_ticks;
    int visual_hit_early_ticks;
} EncounterProjectileDelayOptions;

typedef struct {
    int damage_delay_ticks;
    int visual_start_delay_ticks;
    int visual_duration_ticks;
} EncounterProjectileTiming;

static inline int encounter_rect_distance(
    int ax, int ay, int asize, int bx, int by, int bsize
) {
    int amax_x = ax + asize - 1;
    int amax_y = ay + asize - 1;
    int bmax_x = bx + bsize - 1;
    int bmax_y = by + bsize - 1;
    int dx = 0;
    int dy = 0;
    if (amax_x < bx) dx = bx - amax_x;
    else if (bmax_x < ax) dx = ax - bmax_x;
    if (amax_y < by) dy = by - amax_y;
    else if (bmax_y < ay) dy = ay - bmax_y;
    return dx > dy ? dx : dy;
}

static inline int encounter_projectile_distance(
    int source_x, int source_y, int source_size,
    int target_x, int target_y, int target_size,
    EncounterProjectileDistanceMode mode
) {
    switch (mode) {
        case ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE:
            return encounter_rect_distance(
                source_x, source_y, source_size, target_x, target_y, target_size);
        case ENCOUNTER_PROJECTILE_DISTANCE_TARGET_SW_TILE:
            return chebyshev_distance(source_x, source_y, target_x, target_y);
    }
    abort();
}

static inline int encounter_projectile_base_hit_delay(
    int distance, int is_player, EncounterProjectileDelayKind kind
) {
    switch (kind) {
        case ENCOUNTER_PROJECTILE_DELAY_MELEE:
            return 0;
        case ENCOUNTER_PROJECTILE_DELAY_MAGIC:
            return encounter_magic_hit_delay(distance, is_player);
        case ENCOUNTER_PROJECTILE_DELAY_RANGED:
            return encounter_ranged_hit_delay(distance, is_player);
        case ENCOUNTER_PROJECTILE_DELAY_THROWN:
            return encounter_thrown_hit_delay(distance, is_player);
        case ENCOUNTER_PROJECTILE_DELAY_BALLISTA:
            return encounter_ballista_hit_delay(distance, is_player);
        case ENCOUNTER_PROJECTILE_DELAY_DARK_BOW_SECOND:
            return encounter_dark_bow_second_hit_delay(distance, is_player);
        case ENCOUNTER_PROJECTILE_DELAY_EYE_OF_AYAK:
            return encounter_eye_of_ayak_hit_delay(distance);
    }
    abort();
}

static inline int encounter_projectile_hit_delay(
    int distance, int is_player, EncounterProjectileDelayKind kind,
    EncounterProjectileDelayOptions options
) {
    int delay = encounter_projectile_base_hit_delay(distance, is_player, kind);
    if (delay > 0) {
        delay -= options.reduce_delay;
        if (delay < 1) delay = 1;
    }
    if (options.set_delay > 0)
        delay = options.set_delay;
    return delay;
}

static inline EncounterProjectileTiming encounter_projectile_timing(
    int distance, int is_player, EncounterProjectileDelayKind kind,
    EncounterProjectileDelayOptions options
) {
    int delay = encounter_projectile_hit_delay(distance, is_player, kind, options);
    int start_delay = options.start_delay > 0
        ? options.start_delay
        : options.visual_delay_ticks;
    int duration = delay - start_delay - options.visual_hit_early_ticks;
    if (duration < 1) duration = 1;
    return (EncounterProjectileTiming){
        .damage_delay_ticks = delay,
        .visual_start_delay_ticks = start_delay,
        .visual_duration_ticks = duration,
    };
}

static inline int encounter_dist_to_npc(int px, int py, int nx, int ny, int npc_size) {
    return encounter_rect_distance(px, py, 1, nx, ny, npc_size);
}

static inline void encounter_shuffle(int* arr, int n, uint32_t* rng) {
    for (int i = n - 1; i > 0; i--) {
        int j = encounter_rand_int(rng, i + 1);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}
/** Standard effective level: floor(base * prayer_mult) + style_bonus + 8. Magic
    accuracy is the exception; use osrs_magic_effective_attack_level for it. */
static inline int osrs_player_eff_level(int base_level, float prayer_mult, int style_bonus) {
    return (int)(base_level * prayer_mult) + style_bonus + 8;
}

static inline int osrs_stance_att_bonus(FightStyle fs, AttackStyle atk) {
    switch (fs) {
        case FIGHT_STYLE_ACCURATE:   return atk == ATTACK_STYLE_MAGIC ? 2 : 3;
        case FIGHT_STYLE_CONTROLLED: return atk == ATTACK_STYLE_MELEE ? 1 : 0;
        case FIGHT_STYLE_LONGRANGE:  return 0;
        default:                     return 0;
    }
}

/** Magic accuracy folds in +9 instead of the melee/ranged +8. */
static inline int osrs_magic_effective_attack_level(
    int magic_level, float prayer_mult, FightStyle fight_style
) {
    return (int)(magic_level * prayer_mult) +
        osrs_stance_att_bonus(fight_style, ATTACK_STYLE_MAGIC) + 9;
}

static inline float osrs_offensive_magic_dmg_mult(OffensivePrayer op) {
    return (op == OFFENSIVE_PRAYER_AUGURY) ? 1.04f : 1.0f;
}

static inline int osrs_stance_str_bonus(FightStyle fs) {
    switch (fs) {
        case FIGHT_STYLE_AGGRESSIVE: return 3;
        case FIGHT_STYLE_CONTROLLED: return 1;
        default:                     return 0;
    }
}

static inline int osrs_stance_def_bonus(FightStyle fs) {
    switch (fs) {
        case FIGHT_STYLE_DEFENSIVE:
        case FIGHT_STYLE_LONGRANGE:  return 3;
        case FIGHT_STYLE_CONTROLLED: return 1;
        default:                     return 0;
    }
}

static inline int osrs_stance_speed_mod(FightStyle fs) {
    return fs == FIGHT_STYLE_RAPID ? -1 : 0;
}

static inline int osrs_stance_range_mod(FightStyle fs) {
    return fs == FIGHT_STYLE_LONGRANGE ? 2 : 0;
}

static inline int osrs_player_att_roll(int eff_level, int equipment_bonus) {
    return eff_level * (equipment_bonus + 64);
}

static inline int osrs_player_melee_max_hit(int eff_str_level, int str_bonus) {
    return (eff_str_level * (str_bonus + 64) + 320) / 640;
}

static inline int osrs_player_ranged_max_hit(int eff_range_level, int ranged_str_bonus) {
    return (eff_range_level * (ranged_str_bonus + 64) + 320) / 640;
}

/** magic_dmg_pct is the total gear bonus in percent (30 = +30%). */
static inline int osrs_player_magic_max_hit(int spell_base_dmg, int magic_dmg_pct) {
    return spell_base_dmg * (100 + magic_dmg_pct) / 100;
}

/** Correct overhead prayer blocks 100% of damage in PvE but only 40% in PvP. */
static inline int osrs_prayer_reduce_damage(int damage, int prayer, int attack_style, int is_pvp) {
    if (damage <= 0) return 0;
    if (!encounter_prayer_correct_for_style(prayer, attack_style)) return damage;
    if (is_pvp) return (int)(damage * 0.6f);
    return 0;
}

/** Closed form of rolling accuracy twice and hitting if EITHER succeeds
    (osmumten's fang, confliction gauntlets). */
static inline float osrs_hit_chance_double(int att_roll, int def_roll) {
    float fa = (float)att_roll, fd = (float)def_roll;
    if (att_roll >= def_roll) {
        float num = (fd + 2.0f) * (2.0f * fd + 3.0f);
        float den = 6.0f * (fa + 1.0f) * (fa + 1.0f);
        return 1.0f - num / den;
    }
    return fa * (4.0f * fa + 5.0f) / (6.0f * (fa + 1.0f) * (fd + 1.0f));
}

typedef struct {
    int attack_stab, attack_slash, attack_crush, attack_magic, attack_ranged;
    int defence_stab, defence_slash, defence_crush, defence_magic, defence_ranged;
    int melee_strength, ranged_strength, magic_damage, prayer;
    int attack_speed, attack_range;
} EquipmentBonuses;

/** Sum ITEM_DATABASE bonuses over a loadout; attack_speed and attack_range come
    from the weapon slot only. */
static inline void osrs_sum_equipment_bonuses(const uint8_t loadout[NUM_GEAR_SLOTS],
                                               EquipmentBonuses* out) {
    memset(out, 0, sizeof(*out));
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        uint8_t idx = loadout[slot];
        if (idx == 255) continue;
        const Item* item = &ITEM_DATABASE[idx];
        out->attack_stab += item->attack_stab;
        out->attack_slash += item->attack_slash;
        out->attack_crush += item->attack_crush;
        out->attack_magic += item->attack_magic;
        out->attack_ranged += item->attack_ranged;
        out->defence_stab += item->defence_stab;
        out->defence_slash += item->defence_slash;
        out->defence_crush += item->defence_crush;
        out->defence_magic += item->defence_magic;
        out->defence_ranged += item->defence_ranged;
        out->melee_strength += item->melee_strength;
        out->ranged_strength += item->ranged_strength;
        out->magic_damage += item->magic_damage;
        out->prayer += item->prayer;
    }
    uint8_t weapon = loadout[GEAR_SLOT_WEAPON];
    if (weapon != 255) {
        out->attack_speed = ITEM_DATABASE[weapon].attack_speed;
        out->attack_range = ITEM_DATABASE[weapon].attack_range;
    }
}

#include "osrs_venator.h"

#endif /* OSRS_COMBAT_H */
