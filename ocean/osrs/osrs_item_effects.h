#ifndef OSRS_ITEM_EFFECTS_H
#define OSRS_ITEM_EFFECTS_H

#include <assert.h>
#include <string.h>

#include "osrs_damage.h"

#define OSRS_SPEC_REGEN_INTERVAL 50
#define OSRS_SPEC_REGEN_LIGHTBEARER 25
#define OSRS_SPEC_REGEN_AMOUNT 10
#define OSRS_ECHO_BOOTS_MAX_CHARGES 60000

typedef struct {
    int attack_roll;
    int max_hit;
    int min_hit;
    int use_double_accuracy;
    int use_fang_accuracy;
} OsrsPreparedAttackEffects;

typedef struct {
    int heal_amount;
} OsrsPostAttackEffects;

static inline OsrsTargetRef osrs_target_ref_none(void) {
    OsrsTargetRef target = { .kind = OSRS_TARGET_NONE, .id = -1 };
    return target;
}

static inline int osrs_target_ref_equal(OsrsTargetRef lhs, OsrsTargetRef rhs) {
    return lhs.kind == rhs.kind && lhs.id == rhs.id;
}

static inline OsrsTargetEffectContext osrs_target_effect_context_none(void) {
    return (OsrsTargetEffectContext){0};
}

static inline OsrsTargetEffectContext osrs_target_effect_context_magic(
    int magic_level,
    int magic_attack_bonus
) {
    return (OsrsTargetEffectContext){
        .magic_level = magic_level,
        .magic_attack_bonus = magic_attack_bonus,
        .target_class = OSRS_TARGET_CLASS_STANDARD,
    };
}

static inline int osrs_target_effect_context_is_dragon(
    OsrsTargetEffectContext target_context
) {
    return target_context.target_class == OSRS_TARGET_CLASS_DRAGON;
}

static inline int osrs_magic_attack_is_ancient(OsrsMagicAttackKind kind) {
    return kind == OSRS_MAGIC_ATTACK_ANCIENT_ICE ||
           kind == OSRS_MAGIC_ATTACK_ANCIENT_BLOOD;
}

static inline int osrs_effect_profile_has(
    const OsrsEquipmentEffectProfile* profile, uint32_t effect_mask
) {
    return (profile->effect_mask & effect_mask) != 0;
}

#define OSRS_EQUIPMENT_EFFECT_AGGREGATE_FEATURES 10

static inline void osrs_item_effect_class4(uint32_t effect_mask, float out[4]) {
    uint32_t lifesteal = OSRS_ITEM_EFFECT_BLOOD_FURY | OSRS_ITEM_EFFECT_SANG_HEAL;
    uint32_t damage_amp = OSRS_ITEM_EFFECT_TWISTED_BOW | OSRS_ITEM_EFFECT_FANG |
        OSRS_ITEM_EFFECT_TUMEKENS_SHADOW | OSRS_ITEM_EFFECT_DHAROK_PIECE |
        OSRS_ITEM_EFFECT_DRAGON_HUNTER_WAND | OSRS_ITEM_EFFECT_VENATOR_BOUNCE;
    uint32_t defensive = OSRS_ITEM_EFFECT_ELYSIAN | OSRS_ITEM_EFFECT_CRYSTAL_ARMOUR |
        OSRS_ITEM_EFFECT_RECOIL_RING | OSRS_ITEM_EFFECT_VENOM_IMMUNE |
        OSRS_ITEM_EFFECT_ECHO_BOOTS | OSRS_ITEM_EFFECT_CONFLICTION |
        OSRS_ITEM_EFFECT_VIRTUS_PIECE;
    uint32_t util = OSRS_ITEM_EFFECT_LIGHTBEARER;
    out[0] = (effect_mask & lifesteal)  ? 1.0f : 0.0f;
    out[1] = (effect_mask & damage_amp) ? 1.0f : 0.0f;
    out[2] = (effect_mask & defensive)  ? 1.0f : 0.0f;
    out[3] = (effect_mask & util)       ? 1.0f : 0.0f;
}

static inline void osrs_write_equipment_effect_aggregate(
    float* out,
    const OsrsEquipmentEffectProfile* profile
) {
    osrs_item_effect_class4(profile->effect_mask, out);
    out[4] = (float)profile->virtus_piece_count / 3.0f;
    out[5] = (float)profile->dharok_piece_count / 4.0f;
    out[6] = (float)profile->crystal_armour_points / 6.0f;
    out[7] = profile->recoil_source != OSRS_RECOIL_SOURCE_NONE ? 1.0f : 0.0f;
    out[8] = profile->spec_regen_mode == OSRS_SPEC_REGEN_MODE_LIGHTBEARER ? 1.0f : 0.0f;
    out[9] = profile->shield_item != ITEM_NONE ? 1.0f : 0.0f;
}

static inline OsrsRecoilSource osrs_recoil_source_from_ring(uint8_t ring_item) {
    if (ring_item == ITEM_RING_OF_RECOIL) {
        return OSRS_RECOIL_SOURCE_RING_OF_RECOIL;
    }
    if (ring_item == ITEM_RING_OF_SUFFERING_RI) {
        return OSRS_RECOIL_SOURCE_RING_OF_SUFFERING_RI;
    }
    return OSRS_RECOIL_SOURCE_NONE;
}

static inline OsrsSpecRegenMode osrs_spec_regen_mode_from_ring(uint8_t ring_item) {
    if (ring_item == ITEM_LIGHTBEARER) {
        return OSRS_SPEC_REGEN_MODE_LIGHTBEARER;
    }
    return OSRS_SPEC_REGEN_MODE_NORMAL;
}

static inline int osrs_scythe_splats_for_target_size(int target_size) {
    return target_size >= 3 ? 3 : target_size == 2 ? 2 : 1;
}

static inline uint8_t osrs_crystal_armour_points(uint8_t item_index) {
    switch (item_index) {
        case ITEM_CRYSTAL_HELM: return 1;
        case ITEM_CRYSTAL_LEGS: return 2;
        case ITEM_CRYSTAL_BODY: return 3;
        default: return 0;
    }
}

static inline void osrs_item_effect_state_init(OsrsItemEffectState* state) {
    memset(state, 0, sizeof(*state));
    state->confliction_weapon_item = ITEM_NONE;
    state->confliction_target = osrs_target_ref_none();
}

static inline void osrs_clear_confliction_state(OsrsItemEffectState* state) {
    state->confliction_is_primed = 0;
    state->confliction_weapon_item = ITEM_NONE;
    state->confliction_magic_kind = OSRS_MAGIC_ATTACK_NONE;
    state->confliction_target = osrs_target_ref_none();
}

static inline GearBonuses osrs_gear_bonuses_from_equipment_bonuses(
    const EquipmentBonuses* equipment_bonuses
) {
    GearBonuses total = {0};
    total.stab_attack = equipment_bonuses->attack_stab;
    total.slash_attack = equipment_bonuses->attack_slash;
    total.crush_attack = equipment_bonuses->attack_crush;
    total.magic_attack = equipment_bonuses->attack_magic;
    total.ranged_attack = equipment_bonuses->attack_ranged;
    total.stab_defence = equipment_bonuses->defence_stab;
    total.slash_defence = equipment_bonuses->defence_slash;
    total.crush_defence = equipment_bonuses->defence_crush;
    total.magic_defence = equipment_bonuses->defence_magic;
    total.ranged_defence = equipment_bonuses->defence_ranged;
    total.melee_strength = equipment_bonuses->melee_strength;
    total.ranged_strength = equipment_bonuses->ranged_strength;
    total.magic_strength = equipment_bonuses->magic_damage;
    total.attack_speed = equipment_bonuses->attack_speed;
    total.attack_range = equipment_bonuses->attack_range;
    return total;
}

static inline void osrs_derive_equipment_effect_profile(
    const uint8_t equipped[NUM_GEAR_SLOTS],
    OsrsEquipmentEffectProfile* out
) {
    memset(out, 0, sizeof(*out));
    out->weapon_item = equipped[GEAR_SLOT_WEAPON];
    out->ring_item = equipped[GEAR_SLOT_RING];
    out->shield_item = equipped[GEAR_SLOT_SHIELD];
    out->recoil_source = osrs_recoil_source_from_ring(out->ring_item);
    out->spec_regen_mode = osrs_spec_regen_mode_from_ring(out->ring_item);

    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
        uint8_t item_index = equipped[slot];
        if (item_index >= NUM_ITEMS) {
            continue;
        }

        uint32_t effect_mask = ITEM_DATABASE[item_index].effect_mask;
        out->effect_mask |= effect_mask;

        if (effect_mask & OSRS_ITEM_EFFECT_VIRTUS_PIECE) {
            out->virtus_piece_count += 1;
        }
        if (effect_mask & OSRS_ITEM_EFFECT_DHAROK_PIECE) {
            out->dharok_piece_count += 1;
        }
        if (effect_mask & OSRS_ITEM_EFFECT_CRYSTAL_ARMOUR) {
            out->crystal_armour_points += osrs_crystal_armour_points(item_index);
        }
    }
}

static inline void osrs_sync_item_effect_state(
    Player* player, const OsrsEquipmentEffectProfile* previous_profile
) {
    const OsrsEquipmentEffectProfile* current_profile = &player->equipment_effect_profile;

    if (previous_profile->spec_regen_mode != current_profile->spec_regen_mode) {
        if (current_profile->spec_regen_mode == OSRS_SPEC_REGEN_MODE_LIGHTBEARER) {
            if (player->item_effect_state.special_regen_ticks > OSRS_SPEC_REGEN_LIGHTBEARER) {
                player->item_effect_state.special_regen_ticks = 0;
            }
        } else {
            player->item_effect_state.special_regen_ticks = 0;
        }
    }

    if (previous_profile->recoil_source != current_profile->recoil_source) {
        if (current_profile->recoil_source == OSRS_RECOIL_SOURCE_RING_OF_RECOIL) {
            player->item_effect_state.recoil_charges = RECOIL_MAX_CHARGES;
        } else if (current_profile->recoil_source == OSRS_RECOIL_SOURCE_RING_OF_SUFFERING_RI) {
            player->item_effect_state.recoil_charges = RECOIL_MAX_CHARGES;
        } else {
            player->item_effect_state.recoil_charges = 0;
        }
    }

    if (!osrs_effect_profile_has(current_profile, OSRS_ITEM_EFFECT_CONFLICTION)) {
        osrs_clear_confliction_state(&player->item_effect_state);
    }

    int had_echo = osrs_effect_profile_has(previous_profile, OSRS_ITEM_EFFECT_ECHO_BOOTS);
    int has_echo = osrs_effect_profile_has(current_profile, OSRS_ITEM_EFFECT_ECHO_BOOTS);
    if (has_echo && !had_echo) {
        player->item_effect_state.echo_boot_charges = OSRS_ECHO_BOOTS_MAX_CHARGES;
    } else if (!has_echo) {
        player->item_effect_state.echo_boot_charges = 0;
    }
}

static inline void osrs_refresh_player_equipment(Player* player) {
    OsrsEquipmentEffectProfile previous_profile = player->equipment_effect_profile;
    EquipmentBonuses equipment_bonuses;
    osrs_sum_equipment_bonuses(player->equipped, &equipment_bonuses);
    player->slot_cached_bonuses = osrs_gear_bonuses_from_equipment_bonuses(&equipment_bonuses);
    osrs_derive_equipment_effect_profile(player->equipped, &player->equipment_effect_profile);
    osrs_sync_item_effect_state(player, &previous_profile);
    player->slot_gear_dirty = 0;
}

static inline void osrs_ensure_player_equipment(Player* player) {
    if (player->slot_gear_dirty) {
        osrs_refresh_player_equipment(player);
    }
}

static inline int osrs_confliction_can_apply(
    const OsrsEquipmentEffectProfile* profile,
    AttackStyle style,
    uint8_t weapon_item,
    int is_primary_target
) {
    return style == ATTACK_STYLE_MAGIC &&
           is_primary_target &&
           osrs_effect_profile_has(profile, OSRS_ITEM_EFFECT_CONFLICTION) &&
           !item_is_two_handed(weapon_item);
}

static inline int osrs_confliction_is_match(
    const OsrsItemEffectState* state,
    uint8_t weapon_item,
    OsrsMagicAttackKind magic_kind,
    OsrsTargetRef target_ref
) {
    return state->confliction_is_primed &&
           state->confliction_weapon_item == weapon_item &&
           state->confliction_magic_kind == magic_kind &&
           osrs_target_ref_equal(state->confliction_target, target_ref);
}

static inline int osrs_fang_hit_bound_shrink(int max_hit) {
    return max_hit * 3 / 20;
}

static inline OsrsPreparedAttackEffects osrs_prepare_attack_effects_for_melee_style(
    const OsrsEquipmentEffectProfile* profile,
    const OsrsItemEffectState* state,
    uint8_t weapon_item,
    AttackStyle style,
    MeleeStyle melee_style,
    OsrsMagicAttackKind magic_kind,
    OsrsTargetRef target_ref,
    int is_primary_target,
    int base_attack_roll,
    int base_max_hit,
    OsrsTargetEffectContext target_context,
    int attacker_current_hitpoints,
    int attacker_base_hitpoints
) {
    OsrsPreparedAttackEffects result = {
        .attack_roll = base_attack_roll,
        .max_hit = base_max_hit,
        .min_hit = 0,
        .use_double_accuracy = 0,
        .use_fang_accuracy = 0,
    };

    if (style == ATTACK_STYLE_RANGED &&
        weapon_item == ITEM_TWISTED_BOW &&
        osrs_effect_profile_has(profile, OSRS_ITEM_EFFECT_TWISTED_BOW)) {
        int target_magic = max_int(
            target_context.magic_level,
            target_context.magic_attack_bonus);
        result.attack_roll = (int)(result.attack_roll * osrs_tbow_acc_mult(target_magic));
        result.max_hit = (int)(result.max_hit * osrs_tbow_dmg_mult(target_magic));
    }

    if (style == ATTACK_STYLE_RANGED &&
        weapon_item == ITEM_BOW_OF_FAERDHINEN &&
        profile->crystal_armour_points > 0) {
        result.attack_roll =
            result.attack_roll * (20 + profile->crystal_armour_points) / 20;
        result.max_hit =
            result.max_hit * (40 + profile->crystal_armour_points) / 40;
    }

    if (osrs_target_effect_context_is_dragon(target_context) &&
        weapon_item == ITEM_DRAGON_HUNTER_WAND &&
        osrs_effect_profile_has(profile, OSRS_ITEM_EFFECT_DRAGON_HUNTER_WAND) &&
        (style == ATTACK_STYLE_MAGIC || style == ATTACK_STYLE_MELEE)) {
        result.attack_roll = result.attack_roll * 7 / 4;
        result.max_hit = result.max_hit * 7 / 5;
    }

    if (style == ATTACK_STYLE_MAGIC && osrs_magic_attack_is_ancient(magic_kind) &&
        profile->virtus_piece_count > 0) {
        result.max_hit = result.max_hit * (100 + 3 * profile->virtus_piece_count) / 100;
    }

    if (style == ATTACK_STYLE_MELEE && profile->dharok_piece_count >= 4) {
        float hp_ratio = 1.0f - ((float)attacker_current_hitpoints / (float)attacker_base_hitpoints);
        result.max_hit = (int)(result.max_hit * (1.0f + hp_ratio * hp_ratio));
    }

    if (style == ATTACK_STYLE_MELEE &&
        weapon_item == ITEM_OSMUMTENS_FANG &&
        osrs_effect_profile_has(profile, OSRS_ITEM_EFFECT_FANG)) {
        int fang_shrink = osrs_fang_hit_bound_shrink(result.max_hit);
        result.min_hit = fang_shrink;
        result.max_hit -= fang_shrink;
        if (melee_style == MELEE_STYLE_STAB) {
            result.use_fang_accuracy = 1;
        }
    }

    if (osrs_confliction_can_apply(profile, style, weapon_item, is_primary_target) &&
        osrs_confliction_is_match(state, weapon_item, magic_kind, target_ref)) {
        result.use_double_accuracy = 1;
    }

    return result;
}

static inline OsrsPreparedAttackEffects osrs_prepare_attack_effects(
    const OsrsEquipmentEffectProfile* profile,
    const OsrsItemEffectState* state,
    uint8_t weapon_item,
    AttackStyle style,
    OsrsMagicAttackKind magic_kind,
    OsrsTargetRef target_ref,
    int is_primary_target,
    int base_attack_roll,
    int base_max_hit,
    OsrsTargetEffectContext target_context,
    int attacker_current_hitpoints,
    int attacker_base_hitpoints
) {
    return osrs_prepare_attack_effects_for_melee_style(
        profile, state, weapon_item, style, MELEE_STYLE_STAB, magic_kind,
        target_ref, is_primary_target, base_attack_roll, base_max_hit,
        target_context, attacker_current_hitpoints, attacker_base_hitpoints);
}

static inline int osrs_roll_prepared_attack_damage(
    const OsrsPreparedAttackEffects* prepared,
    int def_roll,
    int splat_max_hit,
    uint32_t* rng_state
) {
    assert(prepared->min_hit <= splat_max_hit);
    int damage = prepared->min_hit +
                 encounter_rand_int(rng_state, splat_max_hit - prepared->min_hit + 1);
    int hit = (prepared->use_fang_accuracy || prepared->use_double_accuracy)
        ? encounter_roll_hit_chance_double(rng_state, prepared->attack_roll, def_roll)
        : encounter_roll_hit_chance(rng_state, prepared->attack_roll, def_roll);
    return hit ? damage : 0;
}

static inline int osrs_blood_fury_heal_amount(
    const OsrsEquipmentEffectProfile* profile,
    AttackStyle style,
    int damage_dealt,
    uint32_t* rng_state
) {
    if (damage_dealt > 0 &&
        style == ATTACK_STYLE_MELEE &&
        osrs_effect_profile_has(profile, OSRS_ITEM_EFFECT_BLOOD_FURY) &&
        encounter_rand_int(rng_state, 5) == 0) {
        return damage_dealt * 30 / 100;
    }
    return 0;
}

static inline OsrsPostAttackEffects osrs_finalize_attack_effects(
    const OsrsEquipmentEffectProfile* profile,
    OsrsItemEffectState* state,
    uint8_t weapon_item,
    AttackStyle style,
    OsrsMagicAttackKind magic_kind,
    OsrsTargetRef target_ref,
    int is_primary_target,
    int used_double_accuracy,
    int hit_landed,
    int damage_dealt,
    uint32_t* rng_state
) {
    OsrsPostAttackEffects result = { .heal_amount = 0 };

    if (osrs_confliction_can_apply(profile, style, weapon_item, is_primary_target)) {
        if (used_double_accuracy) {
            osrs_clear_confliction_state(state);
        } else if (hit_landed) {
            osrs_clear_confliction_state(state);
        } else {
            state->confliction_is_primed = 1;
            state->confliction_weapon_item = weapon_item;
            state->confliction_magic_kind = magic_kind;
            state->confliction_target = target_ref;
        }
    }

    if (damage_dealt > 0 &&
        style == ATTACK_STYLE_MAGIC &&
        weapon_item == ITEM_SANGUINESTI_STAFF &&
        osrs_effect_profile_has(profile, OSRS_ITEM_EFFECT_SANG_HEAL) &&
        encounter_rand_int(rng_state, 6) == 0) {
        result.heal_amount = damage_dealt / 2;
    }

    return result;
}

static inline int osrs_has_recoil_available(
    const OsrsEquipmentEffectProfile* profile,
    const OsrsItemEffectState* state
) {
    if (profile->recoil_source == OSRS_RECOIL_SOURCE_RING_OF_SUFFERING_RI) {
        return 1;
    }
    if (profile->recoil_source == OSRS_RECOIL_SOURCE_RING_OF_RECOIL) {
        return state->recoil_charges > 0;
    }
    return 0;
}

static inline int osrs_has_echo_boots_recoil_available(
    const OsrsEquipmentEffectProfile* profile,
    const OsrsItemEffectState* state
) {
    return osrs_effect_profile_has(profile, OSRS_ITEM_EFFECT_ECHO_BOOTS) &&
        state->echo_boot_charges > 0;
}

static inline int osrs_echo_boots_recoil_damage(
    const OsrsEquipmentEffectProfile* profile,
    const OsrsItemEffectState* state,
    int final_damage
) {
    if (final_damage <= 0) return 0;
    if (!osrs_has_echo_boots_recoil_available(profile, state)) return 0;
    return 1;
}

static inline DamageResult osrs_apply_passive_damage_pipeline(
    int raw_damage,
    int attack_style,
    int target_prayer,
    int is_pvp,
    int target_veng_active,
    int attacker_smite_active,
    const OsrsEquipmentEffectProfile* defender_profile,
    const OsrsItemEffectState* defender_state,
    uint32_t* rng_state
) {
    int prayer_correct = encounter_prayer_correct_for_style(target_prayer, attack_style);
    int final_damage = osrs_prayer_reduce_damage(raw_damage, target_prayer, attack_style, is_pvp);
    int elysian_reduced = 0;

    if (final_damage > 0 &&
        osrs_effect_profile_has(defender_profile, OSRS_ITEM_EFFECT_ELYSIAN) &&
        encounter_rand_int(rng_state, 10) < 7) {
        final_damage = final_damage * 75 / 100;
        elysian_reduced = 1;
    }

    DamageResult result = osrs_apply_post_mitigation_pipeline(
        final_damage,
        prayer_correct,
        target_veng_active,
        osrs_has_recoil_available(defender_profile, defender_state),
        attacker_smite_active
    );
    result.elysian_reduced = elysian_reduced;
    return result;
}

static inline void osrs_consume_echo_boots_charge(Player* defender) {
    osrs_ensure_player_equipment(defender);
    if (!osrs_effect_profile_has(
            &defender->equipment_effect_profile, OSRS_ITEM_EFFECT_ECHO_BOOTS)) {
        return;
    }
    if (defender->item_effect_state.echo_boot_charges <= 0)
        return;
    defender->item_effect_state.echo_boot_charges--;
}

static inline void osrs_consume_recoil_charges(Player* defender, int recoil_damage) {
    osrs_ensure_player_equipment(defender);
    if (defender->equipment_effect_profile.recoil_source != OSRS_RECOIL_SOURCE_RING_OF_RECOIL) {
        return;
    }

    defender->item_effect_state.recoil_charges -= recoil_damage;
    if (defender->item_effect_state.recoil_charges > 0) {
        return;
    }

    defender->item_effect_state.recoil_charges = 0;
    defender->equipped[GEAR_SLOT_RING] = ITEM_NONE;
    osrs_refresh_player_equipment(defender);
}

static inline void osrs_tick_special_regen(Player* player) {
    osrs_ensure_player_equipment(player);
    if (player->special_energy >= 100) {
        player->item_effect_state.special_regen_ticks = 0;
        return;
    }

    int regen_interval = player->equipment_effect_profile.spec_regen_mode ==
        OSRS_SPEC_REGEN_MODE_LIGHTBEARER ? OSRS_SPEC_REGEN_LIGHTBEARER : OSRS_SPEC_REGEN_INTERVAL;
    player->item_effect_state.special_regen_ticks += 1;
    if (player->item_effect_state.special_regen_ticks >= regen_interval) {
        player->special_energy = clamp(player->special_energy + OSRS_SPEC_REGEN_AMOUNT, 0, 100);
        player->item_effect_state.special_regen_ticks = 0;
    }
}

#endif
