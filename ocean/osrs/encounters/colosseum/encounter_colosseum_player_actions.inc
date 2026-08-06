#line 76 "encounter_colosseum.h"

#define COLO_HEAD_PRIMARY         0
#define COLO_HEAD_PRAYER          1

#define COLO_HEAD_EQUIP_BASE      2
#define COLO_HEAD_EQUIP_SLOT(slot) (COLO_HEAD_EQUIP_BASE + (slot))
#define COLO_HEAD_EAT            (COLO_HEAD_EQUIP_BASE + NUM_GEAR_SLOTS)
#define COLO_HEAD_DRINK          (COLO_HEAD_EAT + 1)
#define COLO_HEAD_SPEC           (COLO_HEAD_DRINK + 1)
#define COLO_HEAD_MODIFIER_SELECT (COLO_HEAD_SPEC + 1)
#define COLO_HEAD_GRAPPLE_PARRY  (COLO_HEAD_MODIFIER_SELECT + 1)
#define COLO_HEAD_OFFENSIVE      (COLO_HEAD_GRAPPLE_PARRY + 1)
#define COLO_HEAD_SPELL          (COLO_HEAD_OFFENSIVE + 1)

#define COLO_INV_CLICK_HEADS (NUM_GEAR_SLOTS + 2)
#define COLO_INV_CLICK_DIM 29
#define COLO_PRIMARY_ATTACK_BASE ENCOUNTER_MOVE_ACTIONS
#define COLO_PRIMARY_DIM (ENCOUNTER_MOVE_ACTIONS + COLO_OBS_NPCS)
#define COLO_OVERHEAD_DIM ENCOUNTER_OVERHEAD_DIM_PVE
#define COLO_SPEC_DIM 3
#define COLO_SPELL_DIM 3

#define COLO_OVERHEAD_NO_CHANGE ENCOUNTER_OVERHEAD_NO_CHANGE
#define COLO_OVERHEAD_OFF ENCOUNTER_OVERHEAD_OFF
#define COLO_OVERHEAD_MELEE ENCOUNTER_OVERHEAD_SET_REFRESH_MELEE
#define COLO_OVERHEAD_RANGED ENCOUNTER_OVERHEAD_SET_REFRESH_RANGED
#define COLO_OVERHEAD_MAGIC ENCOUNTER_OVERHEAD_SET_REFRESH_MAGIC

#define COLO_SPELL_SUMMON_THRALL 1
#define COLO_SPELL_DEATH_CHARGE  2

#define COLO_MODIFIER_SELECT_DIM (1 + COLO_MODIFIER_DRAFT_OPTIONS)

#define COLO_GRAPPLE_PARRY_DIM (1 + COLO_NUM_GRAPPLE_SLOTS)

#define COLO_ACTION_DIMS_INIT { \
    COLO_PRIMARY_DIM, \
    COLO_OVERHEAD_DIM, \
    COLO_INV_CLICK_DIM, COLO_INV_CLICK_DIM, COLO_INV_CLICK_DIM, COLO_INV_CLICK_DIM, \
    COLO_INV_CLICK_DIM, COLO_INV_CLICK_DIM, COLO_INV_CLICK_DIM, COLO_INV_CLICK_DIM, \
    COLO_INV_CLICK_DIM, COLO_INV_CLICK_DIM, COLO_INV_CLICK_DIM, \
    COLO_INV_CLICK_DIM, \
    COLO_INV_CLICK_DIM, \
    COLO_SPEC_DIM, \
    COLO_MODIFIER_SELECT_DIM, \
    COLO_GRAPPLE_PARRY_DIM, \
    ENCOUNTER_OFFENSIVE_DIM, \
    COLO_SPELL_DIM \
}
static const int COLO_ACTION_DIMS[COLO_NUM_ACTION_HEADS] = COLO_ACTION_DIMS_INIT;
static_assert(COLO_NUM_ACTION_HEADS == 2 + NUM_GEAR_SLOTS + 2 + 5,
    "colosseum action head count must match the per-category layout");
static_assert(COLO_HEAD_SPELL == COLO_NUM_ACTION_HEADS - 1,
    "SPELL must be the last colosseum action head");
#define COLO_ACTION_MASK_SIZE (ENCOUNTER_MOVE_ACTIONS + COLO_OVERHEAD_DIM + \
    COLO_OBS_NPCS + COLO_INV_CLICK_HEADS * COLO_INV_CLICK_DIM + COLO_SPEC_DIM + \
    COLO_MODIFIER_SELECT_DIM + COLO_GRAPPLE_PARRY_DIM + ENCOUNTER_OFFENSIVE_DIM + \
    COLO_SPELL_DIM)

static int col_action_head_mask_offset(int head) {
    int offset = 0;
    for (int h = 0; h < head; h++)
        offset += COLO_ACTION_DIMS[h];
    return offset;
}

static int col_primary_attack_action_for_obs_slot(int obs_slot) {
    if (obs_slot < 0 || obs_slot >= COLO_OBS_NPCS) {
        fprintf(stderr, "colosseum primary attack obs slot out of range: %d\n", obs_slot);
        abort();
    }
    return COLO_PRIMARY_ATTACK_BASE + obs_slot;
}

static int col_primary_obs_slot_from_action(int primary_action) {
    if (primary_action < COLO_PRIMARY_ATTACK_BASE ||
            primary_action >= COLO_PRIMARY_DIM) {
        fprintf(stderr, "colosseum primary attack action out of range: %d\n",
            primary_action);
        abort();
    }
    return primary_action - COLO_PRIMARY_ATTACK_BASE;
}

static EncounterStatDriftPins col_divine_stat_drift_pins(const ColosseumState* s) {
    EncounterStatDriftPins pins = encounter_stat_drift_no_pins();
    if (s->divine_combat_timer > 0) {
        pins = encounter_merge_stat_drift_pins(
            pins, encounter_divine_super_combat_pins(&s->player));
    }
    if (s->divine_ranged_timer > 0) {
        pins = encounter_merge_stat_drift_pins(
            pins, encounter_divine_ranging_pins(&s->player));
    }
    return pins;
}

static int col_enforce_divine_stat_floors(ColosseumState* s) {
    return encounter_enforce_stat_drift_pins(&s->player, col_divine_stat_drift_pins(s));
}

static void col_finish_stat_moving_consumable(ColosseumState* s) {
    col_enforce_divine_stat_floors(s);
    col_mark_live_loadout_dirty(s);
}

static int col_tick_divine_combat_timer(ColosseumState* s) {
    if (s->divine_combat_timer <= 0) return 0;
    s->divine_combat_timer--;
    if (s->divine_combat_timer > 0) return 0;
    s->player.current_attack = s->player.base_attack;
    s->player.current_strength = s->player.base_strength;
    s->player.current_defence = s->player.base_defence;
    return 1;
}

static int col_tick_divine_ranged_timer(ColosseumState* s) {
    if (s->divine_ranged_timer <= 0) return 0;
    s->divine_ranged_timer--;
    if (s->divine_ranged_timer > 0) return 0;
    s->player.current_ranged = s->player.base_ranged;
    return 1;
}

static void col_tick_live_stat_drift_and_divines(ColosseumState* s) {
    int changed = 0;
    changed |= col_tick_divine_combat_timer(s);
    changed |= col_tick_divine_ranged_timer(s);
    changed |= encounter_tick_stat_drift(
        &s->player, &s->stat_drift_timer, col_divine_stat_drift_pins(s));
    if (changed) col_mark_live_loadout_dirty(s);
}

static void col_apply_divine_combat_potion_effect(ColosseumState* s) {
    int hitpoints_before = s->player.current_hitpoints;
    encounter_damage_player(&s->player, 10, NULL);
    int applied = hitpoints_before - s->player.current_hitpoints;
    s->tick_scratch.damage_received += (float)applied;
    s->tick_scratch.landed_self_damage += (float)applied;
    if (applied > 0) {
        s->last_damage_source = COLO_DAMAGE_SELF;
        col_mod_add_doom_stack(s, COLO_DAMAGE_SELF);
    }
    s->divine_combat_timer = ENCOUNTER_DIVINE_POTION_TICKS;
}

static void col_apply_divine_ranged_potion_effect(ColosseumState* s) {
    int hitpoints_before = s->player.current_hitpoints;
    encounter_damage_player(&s->player, 10, NULL);
    int applied = hitpoints_before - s->player.current_hitpoints;
    s->tick_scratch.damage_received += (float)applied;
    s->tick_scratch.landed_self_damage += (float)applied;
    if (applied > 0) {
        s->last_damage_source = COLO_DAMAGE_SELF;
        col_mod_add_doom_stack(s, COLO_DAMAGE_SELF);
    }
    s->divine_ranged_timer = ENCOUNTER_DIVINE_POTION_TICKS;
}

static int col_overhead_switch_slips(uint32_t* rng_state, float fail_prob) {
    uint64_t numerator = (uint64_t)(fail_prob * 65536.0f);
    return encounter_roll_ratio_u16(rng_state, numerator, 65536);
}

static void col_player_pretick(ColosseumState* s, ColosseumContext* ctx, const int* actions) {
    OverheadPrayer prev_prayer = s->player.prayer;
    OffensivePrayer prev_offensive = s->player.offensive_prayer;
    int overhead_activated = encounter_apply_overhead_action(&s->player.prayer,
        actions[COLO_HEAD_PRAYER]);
    if (s->player.prayer != prev_prayer &&
            ctx->config.prayer_switch_fail_prob > 0.0f &&
            col_overhead_switch_slips(&s->rng_state, ctx->config.prayer_switch_fail_prob)) {
        s->player.prayer = prev_prayer;
        overhead_activated = 0;
    }
    if (overhead_activated)
        s->player.prayer_just_activated = 1;
    if (encounter_apply_offensive_action(&s->player.offensive_prayer,
            actions[COLO_HEAD_OFFENSIVE]))
        s->player.offensive_prayer_just_activated = 1;
    encounter_drain_all_prayers(&s->player, encounter_player_prayer_bonus(&s->player));
    if (s->player.offensive_prayer != prev_offensive)
        col_mark_live_loadout_dirty(s);
}

static void col_apply_weapon_set(ColosseumState* s, ColoWeaponSet set) {
    if (set < 0 || set >= COLO_NUM_WEAPON_SETS)
        col_abort_invalid_weapon_set(set);
    if (s->weapon_set == set) return;
    GearSet gear = (set == COLO_GEAR_RANGED) ? GEAR_RANGED
                 : (set == COLO_GEAR_MAGIC) ? GEAR_MAGE
                 : GEAR_MELEE;
    encounter_apply_loadout(&s->player,
        col_loadouts_for_profile(s->active_loadout_profile)[set], gear);
    s->weapon_set = set;
    s->player.spec_armed = 0;
    col_mark_live_loadout_dirty(s);
}

static void col_sync_weapon_set_from_equipped_weapon(ColosseumState* s) {
    AttackStyle style = col_equipped_weapon_attack_style(&s->player);
    s->weapon_set = style == ATTACK_STYLE_RANGED ? COLO_GEAR_RANGED
        : style == ATTACK_STYLE_MAGIC ? COLO_GEAR_MAGIC
        : COLO_GEAR_MELEE;
    s->player.current_gear = col_gear_set_for_attack_style(style);
    s->player.visible_gear = s->player.current_gear;
}

static int col_can_equip_from_cell(const ColosseumState* s, int cell_idx) {
    return osrs_can_equip_from_cell(&s->player, s->inventory_cells, cell_idx);
}

static void col_equip_from_cell(ColosseumState* s, int cell_idx) {
    if (osrs_equip_from_cell(&s->player, s->inventory_cells, cell_idx) < 0) return;
    col_sync_weapon_set_from_equipped_weapon(s);
    col_mark_live_loadout_dirty(s);
}

static int col_can_eat_consumable_kind(
    const ColosseumState* s,
    OsrsConsumableKind kind
) {
    return osrs_can_eat_consumable_kind(&s->player, kind);
}

#define COLO_BOOST_REDRINK_LEVEL 105

static int col_restore_worth_drinking(const ColosseumState* s) {
    const Player* p = &s->player;
    int stats_drained = p->current_attack   < p->base_attack   ||
                        p->current_strength < p->base_strength ||
                        p->current_defence  < p->base_defence  ||
                        p->current_ranged   < p->base_ranged   ||
                        p->current_magic    < p->base_magic;
    int restore_amount = osrs_super_restore_amount(p->base_prayer);
    int pray_worth = (p->base_prayer - p->current_prayer) >= (restore_amount + 1) / 2;
    return stats_drained || pray_worth;
}

static int col_drink_has_effect(const ColosseumState* s, OsrsConsumableKind kind) {
    const Player* p = &s->player;
    int afflicted = s->player_venom > 0 || s->player_poison > 0;
    switch (kind) {
        case OSRS_CONSUMABLE_BREW:
            return p->current_hitpoints < p->base_hitpoints;
        case OSRS_CONSUMABLE_SUPER_COMBAT:
        case OSRS_CONSUMABLE_DIVINE_COMBAT:
            return p->current_attack   < COLO_BOOST_REDRINK_LEVEL ||
                   p->current_strength < COLO_BOOST_REDRINK_LEVEL ||
                   p->current_defence  < COLO_BOOST_REDRINK_LEVEL;
        case OSRS_CONSUMABLE_RANGING:
        case OSRS_CONSUMABLE_DIVINE_RANGING:
            return p->current_ranged < COLO_BOOST_REDRINK_LEVEL;
        case OSRS_CONSUMABLE_SATURATED_HEART:
            return p->current_magic < COLO_BOOST_REDRINK_LEVEL;
        case OSRS_CONSUMABLE_SUPER_RESTORE:
            return col_restore_worth_drinking(s);
        case OSRS_CONSUMABLE_PRAYER_RESTORE:
            return p->current_prayer < p->base_prayer;
        case OSRS_CONSUMABLE_SANFEW:
            return afflicted || col_restore_worth_drinking(s);
        default:
            return 1;
    }
}

static int col_inventory_cell_actionable(const ColosseumState* s, int cell_idx) {
    const ColoInvCell* cell = &s->inventory_cells[cell_idx];
    OsrsInventoryClickResolution r =
        osrs_inventory_cell_click_interpret(cell, OSRS_CLICK_TICK_FIRST);
    if (r.click_action == OSRS_CLICK_EQUIP) return col_can_equip_from_cell(s, cell_idx);
    if (r.click_action == OSRS_CLICK_DRINK)
        return cell->dose > 0 && s->player.potion_timer == 0 &&
               col_drink_has_effect(s, r.consumable_kind);
    if (r.click_action == OSRS_CLICK_EAT)
        return col_can_eat_consumable_kind(s, r.consumable_kind);
    return 0;
}

static void col_empty_inventory_cell(ColosseumState* s, int cell_idx) {
    s->inventory_cells[cell_idx] = osrs_inventory_cell_empty();
}

static void col_heal_player_capped(ColosseumState* s, int amount) {
    s->player.current_hitpoints += amount;
    if (s->player.current_hitpoints > s->player.base_hitpoints)
        s->player.current_hitpoints = s->player.base_hitpoints;
}

static int col_apply_food_cell(ColosseumState* s, OsrsConsumableKind kind) {
    FoodType type;
    if (kind == OSRS_CONSUMABLE_SHARK_FOOD) type = FOOD_SHARK;
    else if (kind == OSRS_CONSUMABLE_KARAMBWAN) type = FOOD_KARAMBWAN;
    else {
        fprintf(stderr, "colosseum: consumable kind %d is not food\n", (int)kind);
        abort();
    }
    return osrs_player_eat_food_effects(&s->player, type).consumed;
}

static void col_apply_drink_one_dose_effect(void* ctx, OsrsConsumableKind kind) {
    ColosseumState* s = (ColosseumState*)ctx;
    switch (kind) {
        case OSRS_CONSUMABLE_BREW: {
            int brew_heal = osrs_brew_heal_amount(s->player.base_hitpoints);
            int cap = col_mod_active(s, COLO_MOD_FRAILTY)
                ? s->player.base_hitpoints
                : s->player.base_hitpoints + brew_heal;
            encounter_apply_brew_heal_capped(&s->player, brew_heal, cap);
            encounter_apply_consumable_stat_effect(
                &s->player, ENCOUNTER_CONSUMABLE_STAT_EFFECT_BREW_DRAIN);
            col_finish_stat_moving_consumable(s);
            return;
        }
        case OSRS_CONSUMABLE_SUPER_RESTORE:
            encounter_add_prayer_restore(
                &s->player, osrs_super_restore_amount(s->player.base_prayer));
            encounter_apply_consumable_stat_effect(
                &s->player, ENCOUNTER_CONSUMABLE_STAT_EFFECT_RESTORE);
            encounter_cap_prayer_restore(&s->player);
            col_finish_stat_moving_consumable(s);
            return;
        case OSRS_CONSUMABLE_PRAYER_RESTORE:
            encounter_add_prayer_restore(
                &s->player, osrs_prayer_potion_restore_amount(s->player.base_prayer));
            encounter_cap_prayer_restore(&s->player);
            return;
        case OSRS_CONSUMABLE_SANFEW:
            encounter_add_prayer_restore(
                &s->player, osrs_sanfew_restore_amount(s->player.base_prayer));
            encounter_apply_consumable_stat_effect(
                &s->player, ENCOUNTER_CONSUMABLE_STAT_EFFECT_SANFEW);
            s->player_venom = 0;
            s->player_venom_timer = 0;
            s->player_poison = 0;
            s->player_poison_timer = 0;
            encounter_cap_prayer_restore(&s->player);
            col_finish_stat_moving_consumable(s);
            return;
        case OSRS_CONSUMABLE_SUPER_COMBAT:
            encounter_super_combat_boost(&s->player);
            col_enforce_divine_stat_floors(s);
            col_mark_live_loadout_dirty(s);
            return;
        case OSRS_CONSUMABLE_DIVINE_COMBAT:
            encounter_super_combat_boost(&s->player);
            col_apply_divine_combat_potion_effect(s);
            col_enforce_divine_stat_floors(s);
            col_mark_live_loadout_dirty(s);
            return;
        case OSRS_CONSUMABLE_RANGING:
            encounter_ranging_boost(&s->player);
            col_enforce_divine_stat_floors(s);
            col_mark_live_loadout_dirty(s);
            return;
        case OSRS_CONSUMABLE_DIVINE_RANGING:
            encounter_ranging_boost(&s->player);
            col_apply_divine_ranged_potion_effect(s);
            col_enforce_divine_stat_floors(s);
            col_mark_live_loadout_dirty(s);
            return;
        case OSRS_CONSUMABLE_SURGE:
            s->player.special_energy += COLO_SURGE_RESTORE_AMOUNT;
            if (s->player.special_energy > 100) s->player.special_energy = 100;
            s->surge_cooldown = COLO_SURGE_COOLDOWN_TICKS;
            return;
        case OSRS_CONSUMABLE_GUTHIX_REST:
            col_heal_player_capped(s, 5);
            s->player_venom = 0;
            s->player_venom_timer = 0;
            s->player_poison = 0;
            s->player_poison_timer = 0;
            return;
        case OSRS_CONSUMABLE_SATURATED_HEART:
            encounter_apply_saturated_heart_boost(&s->player);
            col_mark_live_loadout_dirty(s);
            return;
        case OSRS_CONSUMABLE_ANTIVENOM_PLUS:
            s->player_venom = 0;
            s->player_venom_timer = 0;
            s->player_poison = 0;
            s->player_poison_timer = 0;
            return;
        case OSRS_CONSUMABLE_NONE:
        case OSRS_CONSUMABLE_SHARK_FOOD:
        case OSRS_CONSUMABLE_KARAMBWAN:
            fprintf(stderr, "colosseum: consumable kind %d is not drinkable\n", (int)kind);
            abort();
        case OSRS_CONSUMABLE_BASTION:
        case OSRS_CONSUMABLE_STAMINA:
            fprintf(stderr, "colosseum: consumable kind %d not in colosseum kits\n", (int)kind);
            abort();
    }
    abort();
}

static int col_apply_drink_cell(
    ColosseumState* s,
    int cell_idx,
    OsrsInventoryClickResolution resolution
) {
    if (cell_idx < 0 || cell_idx >= COLO_INVENTORY_DISPLAY_SLOTS) return 0;
    ColoInvCell* cell = &s->inventory_cells[cell_idx];
    if (resolution.click_action != OSRS_CLICK_DRINK) return 0;
    OsrsInventoryDrinkConsumeResult consume =
        osrs_inventory_cell_consume_drink_one_dose(
            cell,
            resolution,
            &s->player.potion_timer,
            col_apply_drink_one_dose_effect,
            s);
    if (!consume.consumed) return 0;
    col_sync_consumable_counters_from_inventory(s);
    return 1;
}

static OsrsInventoryTickIntent col_resolve_inventory_tick_intent(
    const ColosseumState* s,
    const int* actions
) {
    OsrsInventoryClickActions clicks;
    for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++)
        clicks.equip_by_slot[slot] = actions[COLO_HEAD_EQUIP_SLOT(slot)];
    clicks.eat = actions[COLO_HEAD_EAT];
    clicks.drink = actions[COLO_HEAD_DRINK];
    return osrs_resolve_inventory_tick_intent(&s->player, s->inventory_cells, &clicks);
}

static void col_apply_inventory_tick_intent(
    ColosseumState* s,
    OsrsInventoryTickIntent* intent
) {
    if (osrs_inventory_tick_intent_has_effect(intent))
        osrs_interaction_check_interrupt(&s->interaction, OSRS_IACT_EQUIP);

    OsrsInventoryApplyStep step;
    while (osrs_inventory_intent_next(intent, &step)) {
        switch (step.kind) {
            case OSRS_INVENTORY_APPLY_EQUIP:
                col_equip_from_cell(s, step.cell_idx);
                break;
            case OSRS_INVENTORY_APPLY_EAT:
                if (col_apply_food_cell(s, step.resolution.consumable_kind))
                    col_empty_inventory_cell(s, step.cell_idx);
                break;
            case OSRS_INVENTORY_APPLY_DRINK:
                (void)col_apply_drink_cell(s, step.cell_idx, step.resolution);
                break;
        }
    }
}

static int col_npc_is_thrall_target(const ColosseumState* s, int slot) {
    if (slot < 0 || slot >= COLO_MAX_NPCS) return 0;
    const ColoNPC* npc = &s->npcs[slot];
    return col_npc_is_live_enemy(npc) && npc->type != COLO_SOL_HEREDIT;
}

static int col_player_target_slot(const ColosseumState* s) {
    if (!osrs_interaction_active(&s->interaction)) return -1;
    return s->interaction.target_slot;
}

static void col_summon_thrall(ColosseumState* s, int target_slot) {
    s->thrall_active = 1;
    s->thrall_target_slot = target_slot;
    s->thrall_lifetime_left = COLO_THRALL_LIFETIME_TICKS[s->active_loadout_profile];
    s->thrall_attack_timer = COLO_THRALL_TICK;
    s->thrall_recast_cd = COLO_THRALL_RECAST_CD;
}

static void col_tick_thrall_and_death_charge(ColosseumState* s, ColosseumContext* ctx) {
    if (s->thrall_recast_cd > 0) s->thrall_recast_cd--;
    if (s->death_charge_cd > 0) s->death_charge_cd--;
    if (s->death_charge_window_left > 0) s->death_charge_window_left--;

    if (s->thrall_active) {
        s->thrall_lifetime_left--;
        if (s->thrall_lifetime_left <= 0) {
            s->thrall_active = 0;
            s->thrall_target_slot = -1;
            s->thrall_lifetime_left = 0;
            s->thrall_attack_timer = 0;
            return;
        }
        int player_target = col_player_target_slot(s);
        if (col_npc_is_thrall_target(s, player_target))
            s->thrall_target_slot = player_target;
        else if (!col_npc_is_thrall_target(s, s->thrall_target_slot))
            s->thrall_target_slot = -1;

        if (s->thrall_attack_timer > 0) s->thrall_attack_timer--;
        if (s->thrall_attack_timer == 0) {
            s->thrall_attack_timer = COLO_THRALL_TICK;
            if (col_npc_is_thrall_target(s, s->thrall_target_slot)) {
                int dmg = encounter_rand_int(&s->rng_state, COLO_THRALL_MAX_HIT + 1);
                col_queue_npc_pending_hit(s, s->thrall_target_slot, dmg, 2,
                    ATTACK_STYLE_MAGIC, ENCOUNTER_SPELL_NONE);
                ctx->thrall_attacked_this_tick = 1;
            }
        }
    }
}

typedef struct {
    ColosseumState* s;
    const ColosseumContext* ctx;
} ColoWalkCtx;

static int col_tile_walkable(void* ctx, int x, int y) {
    ColoWalkCtx* wc = (ColoWalkCtx*)ctx;
    return col_player_walkable(wc->s, x, y);
}

static int col_pathfind_blocked(void* ctx, int abs_x, int abs_y) {
    ColoWalkCtx* wc = (ColoWalkCtx*)ctx;
    int x = abs_x - wc->ctx->world_offset_x;
    int y = abs_y - wc->ctx->world_offset_y;
    if (col_static_blocked(x, y)) return 1;
    if (col_sol_clamp_active(wc->s) && !col_in_boss_arena(wc->s, x, y)) return 1;
    return 0;
}

static int col_npc_collision_flag_at(const ColosseumState* s, int x, int y) {
    int gx, gy;
    if (!col_grid_index(x, y, &gx, &gy)) return 0;
    return s->npc_collision_flags[gx][gy] != 0;
}

static int col_player_attack_range(ColosseumState* s) {
    const EncounterLoadoutStats* ls = col_live_loadout_stats(s);
    int base = ls->attack_range > 0 ? ls->attack_range : 1;
    return col_mod_myopia_range(s, base);
}

static int col_lookup_player_attack_target(void* ctx, int target_slot, OsrsAttackTarget* out) {
    ColosseumState* s = (ColosseumState*)ctx;
    if (target_slot < 0 || target_slot >= COLO_MAX_NPCS) return 0;
    ColoNPC* npc = &s->npcs[target_slot];
    if (!col_npc_is_live_target(npc)) return 0;
    out->slot = target_slot;
    out->x = npc->x;
    out->y = npc->y;
    out->size = col_npc_effective_size(npc);
    out->attack_range = col_player_attack_range(s);
    return 1;
}

static int col_obs_slot_to_npc(const ColosseumState* s, int obs_slot) {
    if (obs_slot < 0 || obs_slot >= COLO_OBS_NPCS) return -1;
    return s->current_obs_slots[obs_slot];
}

static MeleeStyle col_item_melee_style(uint8_t weapon_item) {
    const Item* weapon = get_item(weapon_item);
    if (!weapon) return MELEE_STYLE_STAB;
    MeleeStyle style = MELEE_STYLE_STAB;
    int best = weapon->attack_stab;
    if (weapon->attack_slash > best) { best = weapon->attack_slash; style = MELEE_STYLE_SLASH; }
    if (weapon->attack_crush > best) { style = MELEE_STYLE_CRUSH; }
    return style;
}

static int col_weapon_has_item_effect(uint8_t weapon_item, uint32_t effect_mask) {
    const Item* weapon = get_item(weapon_item);
    return weapon && (weapon->effect_mask & effect_mask) != 0;
}

static int col_npc_effective_def_level(const ColoNPC* npc, const ColoNpcStats* ns) {
    int def = ns->def_level - npc->def_drained;
    return def < 0 ? 0 : def;
}

static void col_player_spec_attack(ColosseumState* s, int npc_slot) {
    ColoNPC* npc = &s->npcs[npc_slot];
    const ColoNpcStats* ns = &COLO_NPC_STATS[npc->type];
    uint8_t item = s->player.equipped[GEAR_SLOT_WEAPON];
    assert(osrs_spec_cost(item) > 0 &&
           "spec attack dispatched with a non-special weapon equipped");
    const EncounterLoadoutStats* ls = col_live_loadout_stats(s);
    const OsrsEquipmentEffectProfile* effects = col_live_effects(s);

    s->player.spec_armed = 0;
    if (!encounter_use_spec(&s->player, osrs_spec_cost(item))) return;

    int eff_def = col_npc_effective_def_level(npc, ns);
    int def_bonus = encounter_npc_melee_def_bonus(
        ns->stab_def, ns->slash_def, ns->crush_def, col_item_melee_style(item));
    int def_roll = osrs_npc_def_roll(eff_def, def_bonus);
    int att_roll = osrs_player_att_roll(ls->eff_level, ls->attack_bonus);

    SpecResult r = osrs_resolve_spec(item, att_roll, ls->max_hit, def_roll,
                                     eff_def, &s->rng_state);
    if (s->wave == COLO_WAVE_BOSS && s->sol.started && s->sol.next_attack_guaranteed_max) {
        osrs_spec_result_force_max(&r, item, ls->max_hit, eff_def);
        s->sol.next_attack_guaranteed_max = 0;
        s->sol.guaranteed_max_ticks = 0;
    }

    int total = 0;
    int bf_heal = 0;
    int num_hits = r.num_hits > 0 ? r.num_hits : 1;
    for (int i = 0; i < num_hits; i++) {
        col_queue_npc_pending_hit(s, npc_slot, r.damage[i], 0,
                                  ATTACK_STYLE_MELEE, ENCOUNTER_SPELL_NONE);
        total += r.damage[i];
        bf_heal += osrs_blood_fury_heal_amount(effects, ATTACK_STYLE_MELEE,
                                               r.damage[i], &s->rng_state);
    }

    if (r.def_drain > 0) {
        npc->def_drained += r.def_drain;
        if (npc->def_drained > ns->def_level) npc->def_drained = ns->def_level;
    }
    if (r.heal > 0) {
        col_heal_player_capped(s, r.heal);
    }
    if (r.prayer_restore > 0) {
        s->player.current_prayer += r.prayer_restore;
        if (s->player.current_prayer > s->player.base_prayer)
            s->player.current_prayer = s->player.base_prayer;
    }

    OsrsPostAttackEffects post = osrs_finalize_attack_effects(
        effects, &s->player.item_effect_state, item,
        ATTACK_STYLE_MELEE, OSRS_MAGIC_ATTACK_NONE, osrs_target_ref_none(), 1,
        0, total > 0, total, &s->rng_state);
    int spec_heal = post.heal_amount + bf_heal;
    if (spec_heal > 0) {
        col_heal_player_capped(s, spec_heal);
    }

    s->player.attack_timer = ls->attack_speed > 0 ? ls->attack_speed : 4;
    s->player_attack_npc_idx = npc_slot;
    s->player_attack_dmg = total;
    s->player_attack_style_id = ATTACK_STYLE_MELEE;
    s->player_attack_timing = (EncounterProjectileTiming){
        .damage_delay_ticks = 0,
        .visual_start_delay_ticks = 0,
        .visual_duration_ticks = 1,
    };
    s->player.attack_style_this_tick = ATTACK_STYLE_MELEE;
    s->player.used_special_this_tick = 1;
    s->tick_scratch.player_attacked = 1;
    s->tick_scratch.attacks_fired++;
}

#define COLO_SCYTHE_MAX_HITS 5

typedef struct {
    int npc_slot;
    int splat_rank;
} ColScytheResolvedHit;

static int col_signum(int value) {
    return (value > 0) - (value < 0);
}

static int col_npc_occupies_tile(const ColoNPC* npc, int x, int y) {
    int size = col_npc_effective_size(npc);
    return x >= npc->x && x < npc->x + size &&
           y >= npc->y && y < npc->y + size;
}

static int col_scythe_tile_target(
    const ColosseumState* s,
    int primary_slot,
    int x,
    int y
) {
    const ColoNPC* primary = &s->npcs[primary_slot];
    if (col_npc_is_live_target(primary) && col_npc_occupies_tile(primary, x, y))
        return primary_slot;

    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        if (i == primary_slot) continue;
        const ColoNPC* npc = &s->npcs[i];
        if (!col_npc_is_live_enemy(npc)) continue;
        if (col_npc_occupies_tile(npc, x, y))
            return i;
    }
    return -1;
}

static void col_scythe_front_arc_tiles(
    const ColosseumState* s,
    const ColoNPC* primary,
    int tile_x[3],
    int tile_y[3]
) {
    int size = col_npc_effective_size(primary);
    int closest_x = s->player.x;
    int closest_y = s->player.y;
    if (closest_x < primary->x) closest_x = primary->x;
    else if (closest_x >= primary->x + size) closest_x = primary->x + size - 1;
    if (closest_y < primary->y) closest_y = primary->y;
    else if (closest_y >= primary->y + size) closest_y = primary->y + size - 1;

    int dx = col_signum(closest_x - s->player.x);
    int dy = col_signum(closest_y - s->player.y);
    if (dx == 0 && dy == 0) {
        fprintf(stderr, "invalid scythe target direction\n");
        abort();
    }

    tile_x[0] = s->player.x + dx;
    tile_y[0] = s->player.y + dy;
    if (dx != 0 && dy != 0) {
        tile_x[1] = s->player.x + dx;
        tile_y[1] = s->player.y;
        tile_x[2] = s->player.x;
        tile_y[2] = s->player.y + dy;
        return;
    }

    int left_x = -dy;
    int left_y = dx;
    tile_x[1] = tile_x[0] + left_x;
    tile_y[1] = tile_y[0] + left_y;
    tile_x[2] = tile_x[0] - left_x;
    tile_y[2] = tile_y[0] - left_y;
}

static int col_resolve_scythe_hits(
    const ColosseumState* s,
    int primary_slot,
    ColScytheResolvedHit out[COLO_SCYTHE_MAX_HITS]
) {
    const ColoNPC* primary = &s->npcs[primary_slot];
    int count = 0;

    int primary_splats = osrs_scythe_splats_for_target_size(col_npc_effective_size(primary));
    for (int rank = 0; rank < primary_splats; rank++) {
        out[count++] = (ColScytheResolvedHit){
            .npc_slot = primary_slot,
            .splat_rank = rank,
        };
    }

    if (col_npc_occupies_tile(primary, s->player.x, s->player.y))
        return count;

    int tile_x[3];
    int tile_y[3];
    col_scythe_front_arc_tiles(s, primary, tile_x, tile_y);
    for (int t = 0; t < 3 && count < COLO_SCYTHE_MAX_HITS; t++) {
        int target_slot = col_scythe_tile_target(s, primary_slot, tile_x[t], tile_y[t]);
        if (target_slot < 0 || target_slot == primary_slot) continue;
        int already = 0;
        for (int j = 0; j < count; j++)
            if (out[j].npc_slot == target_slot) { already = 1; break; }
        if (already) continue;
        out[count++] = (ColScytheResolvedHit){
            .npc_slot = target_slot,
            .splat_rank = 0,
        };
    }
    return count;
}

static OsrsVenatorMonster col_venator_monster_from_npc(
    const ColoNPC* npc,
    int slot
) {
    if (!npc || !col_npc_is_live_enemy(npc)) abort();
    return osrs_venator_monster(
        slot,
        npc->x,
        npc->y,
        col_npc_effective_size(npc),
        OSRS_VENATOR_MONSTER_ALIVE);
}

static OsrsVenatorChain col_resolve_venator_chain(
    const ColosseumState* s,
    int primary_slot
) {
    if (!s || primary_slot < 0 || primary_slot >= COLO_MAX_NPCS) abort();
    const ColoNPC* primary_npc = &s->npcs[primary_slot];
    OsrsVenatorMonster primary =
        col_venator_monster_from_npc(primary_npc, primary_slot);
    OsrsVenatorMonster candidates[COLO_MAX_NPCS];
    int candidate_count = 0;
    for (int slot = 0; slot < COLO_MAX_NPCS; slot++) {
        const ColoNPC* npc = &s->npcs[slot];
        if (!col_npc_is_live_enemy(npc)) continue;
        candidates[candidate_count++] = col_venator_monster_from_npc(npc, slot);
    }
    return osrs_venator_resolve_chain(primary, candidates, candidate_count);
}

static int col_npc_target_def_roll(
    const ColoNPC* npc,
    const ColoNpcStats* ns,
    AttackStyle style,
    MeleeStyle weapon_melee_style
) {
    return encounter_npc_target_def_roll(
        col_npc_effective_def_level(npc, ns), ns->magic_level,
        ns->stab_def, ns->slash_def, ns->crush_def,
        ns->magic_def_bonus, ns->ranged_def_bonus,
        style, weapon_melee_style);
}

static float col_expected_dpt_vs_npc(
    const EncounterLoadoutStats* ls,
    const OsrsEquipmentEffectProfile* effects,
    uint8_t weapon_item,
    AttackStyle style,
    MeleeStyle melee_style,
    const ColoNPC* npc,
    const ColoNpcStats* ns
) {
    if (!ls || !effects || !npc || !ns) abort();
    if (!col_npc_is_live_target(npc) || col_type_is_hazard_entity(npc->type))
        return 0.0f;
    if (ls->attack_speed < 0 || ls->max_hit < 0) {
        fprintf(stderr, "invalid Colosseum expected-DPT loadout\n");
        abort();
    }
    if (style == ATTACK_STYLE_NONE || ls->attack_speed == 0 || ls->max_hit == 0)
        return 0.0f;

    OsrsItemEffectState tmp_effect_state = {0};
    int base_att_roll = osrs_player_att_roll(ls->eff_level, ls->attack_bonus);
    OsrsPreparedAttackEffects prepared =
        osrs_prepare_attack_effects_for_melee_style(
            effects,
            &tmp_effect_state,
            weapon_item,
            style,
            melee_style,
            OSRS_MAGIC_ATTACK_NONE,
            osrs_target_ref_none(),
            1,
            base_att_roll,
            ls->max_hit,
            osrs_target_effect_context_magic(ns->magic_level, ns->magic_att_bonus),
            MAXED_BASE_HITPOINTS,
            MAXED_BASE_HITPOINTS);

    int target_def_roll = col_npc_target_def_roll(npc, ns, style, melee_style);
    float hit_prob;
    if (osrs_confliction_can_apply(effects, style, weapon_item, 1)) {

        float single = osrs_hit_chance(prepared.attack_roll, target_def_roll);
        float dbl = osrs_hit_chance_double(prepared.attack_roll, target_def_roll);
        float denom = 1.0f + dbl - single;
        hit_prob = denom > 0.0f ? dbl / denom : dbl;
    } else if (prepared.use_fang_accuracy || prepared.use_double_accuracy) {
        hit_prob = osrs_hit_chance_double(prepared.attack_roll, target_def_roll);
    } else {
        hit_prob = osrs_hit_chance(prepared.attack_roll, target_def_roll);
    }

    int splat_count = 1;
    if (weapon_item == ITEM_SCYTHE_OF_VITUR)
        splat_count = osrs_scythe_splats_for_target_size(col_npc_effective_size(npc));

    float expected_damage = 0.0f;
    int forced_max_hit = ns->player_style_that_max_hits != ATTACK_STYLE_NONE &&
                         ns->player_style_that_max_hits == style;
    for (int splat = 0; splat < splat_count; splat++) {
        int splat_max = prepared.max_hit >> splat;
        if (forced_max_hit) {
            expected_damage += (float)splat_max;
        } else {
            expected_damage +=
                hit_prob * ((float)prepared.min_hit + (float)splat_max) * 0.5f;
        }
    }

    return expected_damage / (float)ls->attack_speed;
}

static void col_player_attack_target_ctx(
    ColosseumState* s,
    ColosseumContext* ctx,
    int npc_slot
) {
    ColoNPC* npc = &s->npcs[npc_slot];
    const ColoNpcStats* ns = &COLO_NPC_STATS[npc->type];
    const EncounterLoadoutStats* ls = col_live_loadout_stats(s);

    if (!col_type_is_hazard_entity(npc->type) && s->player.spec_armed) {
        col_player_spec_attack(s, npc_slot);
        return;
    }

    const OsrsEquipmentEffectProfile* effects = col_live_effects(s);
    uint8_t weapon_item = s->player.equipped[GEAR_SLOT_WEAPON];
    int projectile_dist = encounter_projectile_distance(
        s->player.x, s->player.y, 1,
        npc->x, npc->y, col_npc_effective_size(npc),
        ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE);
    EncounterProjectileTiming hit_timing =
        col_player_projectile_timing(ls->style, weapon_item, projectile_dist);
    int base_att_roll = osrs_player_att_roll(ls->eff_level, ls->attack_bonus);
    MeleeStyle weapon_melee_style = col_item_melee_style(weapon_item);
    OsrsPreparedAttackEffects prepared = osrs_prepare_attack_effects_for_melee_style(
        effects, &s->player.item_effect_state, weapon_item, ls->style,
        weapon_melee_style, OSRS_MAGIC_ATTACK_NONE, osrs_target_ref_none(), 1,
        base_att_roll, ls->max_hit,
        osrs_target_effect_context_magic(ns->magic_level, ns->magic_att_bonus),
        s->player.current_hitpoints, s->player.base_hitpoints);

    ColScytheResolvedHit resolved_hits[COLO_SCYTHE_MAX_HITS];
    OsrsVenatorChain venator_chain = { OSRS_VENATOR_CHAIN_LENGTH_ONE };
    int resolved_count = 1;
    resolved_hits[0] = (ColScytheResolvedHit){ .npc_slot = npc_slot, .splat_rank = 0 };
    int venator_attack =
        !col_type_is_hazard_entity(npc->type) &&
        col_weapon_has_item_effect(weapon_item, OSRS_ITEM_EFFECT_VENATOR_BOUNCE);
    if (venator_attack) {
        venator_chain = col_resolve_venator_chain(s, npc_slot);
        resolved_count = (int)venator_chain.length;
        col_context_record_venator_chain(ctx, &venator_chain);
    } else if (weapon_item == ITEM_SCYTHE_OF_VITUR &&
            !col_type_is_hazard_entity(npc->type)) {
        resolved_count = col_resolve_scythe_hits(s, npc_slot, resolved_hits);
    }
    if (resolved_count <= 0) {
        fprintf(stderr, "player attack resolved no targets\n");
        abort();
    }

    int ticks = hit_timing.damage_delay_ticks;
    int damage = 0;
    int bf_heal = 0;
    for (int splat = 0; splat < resolved_count; splat++) {
        int hit_slot = venator_attack
            ? venator_chain.hits[splat].slot
            : resolved_hits[splat].npc_slot;
        ColoNPC* hit_npc = &s->npcs[hit_slot];
        const ColoNpcStats* hit_ns = &COLO_NPC_STATS[hit_npc->type];
        int rps_max_hit = hit_ns->player_style_that_max_hits != ATTACK_STYLE_NONE &&
                          hit_ns->player_style_that_max_hits == ls->style;
        int target_def_roll = col_npc_target_def_roll(
            hit_npc, hit_ns, ls->style, weapon_melee_style);
        int splat_max = venator_attack
            ? (splat == 0
                ? prepared.max_hit
                : osrs_venator_bounce_max_hit(prepared.max_hit))
            : (prepared.max_hit >> resolved_hits[splat].splat_rank);
        int splat_damage;
        if (col_type_is_hazard_entity(hit_npc->type)) {
            splat_damage = hit_npc->hp;
        } else if (s->sol.next_attack_guaranteed_max) {
            splat_damage = splat_max;
            s->sol.next_attack_guaranteed_max = 0;
            s->sol.guaranteed_max_ticks = 0;
        } else if (rps_max_hit) {
            splat_damage = splat_max;
        } else {
            splat_damage = osrs_roll_prepared_attack_damage(
                &prepared, target_def_roll, splat_max, &s->rng_state);
        }
        int hit_ticks = venator_attack ? ticks + splat : ticks;
        col_queue_npc_pending_hit(s, hit_slot, splat_damage, hit_ticks,
                                  ls->style, ENCOUNTER_SPELL_NONE);
        damage += splat_damage;
        bf_heal += osrs_blood_fury_heal_amount(effects, ls->style, splat_damage,
                                               &s->rng_state);
    }

    OsrsPostAttackEffects post = osrs_finalize_attack_effects(
        effects, &s->player.item_effect_state, weapon_item, ls->style,
        OSRS_MAGIC_ATTACK_NONE, osrs_target_ref_none(), 1, 0,
        damage > 0, damage, &s->rng_state);
    int swing_heal = post.heal_amount + bf_heal;
    if (swing_heal > 0) {
        col_heal_player_capped(s, swing_heal);
    }

    s->player.attack_timer = ls->attack_speed > 0 ? ls->attack_speed : 4;
    s->player_attack_npc_idx = npc_slot;
    s->player_attack_dmg = damage;
    s->player_attack_style_id = ls->style;
    s->player_attack_timing = hit_timing;
    s->player.attack_style_this_tick = ls->style;
    s->tick_scratch.player_attacked = 1;
    s->tick_scratch.attacks_fired++;
}

static void col_player_attack_target(ColosseumState* s, int npc_slot) {
    col_player_attack_target_ctx(s, col_legacy_context(), npc_slot);
}

typedef struct {
    OsrsPlayerStepInput input;
    int move_action;
} ColoPrimaryStepPlan;

static ColoPrimaryStepPlan col_primary_step_plan(
    ColosseumState* s,
    ColosseumContext* ctx,
    Player* player,
    OsrsInteraction* interaction,
    int primary_action,
    int* player_dest_x,
    int* player_dest_y,
    ColoWalkCtx* walk_ctx,
    OsrsLosQuery* los_query
) {
    if (primary_action < 0 || primary_action >= COLO_PRIMARY_DIM) {
        fprintf(stderr, "colosseum primary action out of range: %d\n", primary_action);
        abort();
    }

    int has_new_target = 0;
    int new_target_slot = -1;
    int move_action = 0;
    if (primary_action > 0 && primary_action < ENCOUNTER_MOVE_ACTIONS) {
        move_action = primary_action;
    } else if (primary_action >= COLO_PRIMARY_ATTACK_BASE) {
        int obs_slot = col_primary_obs_slot_from_action(primary_action);
        int npc_slot = col_obs_slot_to_npc(s, obs_slot);
        if (npc_slot >= 0 && s->npcs[npc_slot].active) {
            has_new_target = 1;
            new_target_slot = npc_slot;
        }
    }

    OsrsPlayerCommand command = { .kind = OSRS_PLAYER_CMD_NONE };
    if (has_new_target) {
        command.kind = OSRS_PLAYER_CMD_TARGET;
        command.target_slot = new_target_slot;
    } else if (*player_dest_x >= 0) {
        command.kind = OSRS_PLAYER_CMD_MOVE;
        command.move_kind = OSRS_PLAYER_MOVE_DESTINATION;
    } else if (move_action > 0 && move_action < ENCOUNTER_MOVE_ACTIONS) {
        command.kind = OSRS_PLAYER_CMD_MOVE;
        command.move_kind = OSRS_PLAYER_MOVE_ACTION;
        command.move_action = move_action;
    }

    *walk_ctx = (ColoWalkCtx){s, ctx};
    *los_query = col_player_los_query(s);
    return (ColoPrimaryStepPlan){
        .input = {
            .player = player,
            .interaction = interaction,
            .target_lookup = col_lookup_player_attack_target,
            .target_ctx = s,
            .command = command,
            .dest_x = player_dest_x,
            .dest_y = player_dest_y,
            .arena = {
                .collision_map = ctx->collision_map,
                .world_offset_x = ctx->world_offset_x,
                .world_offset_y = ctx->world_offset_y,
                .is_walkable = col_tile_walkable,
                .walkable_ctx = walk_ctx,
                .extra_blocked = col_pathfind_blocked,
                .blocked_ctx = walk_ctx,
                .los_query = los_query,
                .arena_base_x = COLO_ARENA_MIN_X,
                .arena_base_y = COLO_ARENA_MIN_Y,
                .arena_w = COLO_ARENA_WIDTH,
                .arena_h = COLO_ARENA_HEIGHT,
            },
        },
        .move_action = move_action,
    };
}

static void col_tick_player_ctx(
    ColosseumState* s,
    ColosseumContext* ctx,
    const int* actions,
    int can_attack
) {
#ifdef COLO_PROFILE_ENABLED
    int col_prof_enabled =
        COLO_PROFILE_ENABLED() && !s->profiling_suppressed;
    double col_prof_t0 = col_prof_enabled ? COLO_PROFILE_NOW_MS() : 0.0;
#endif
    if (s->player_last_interaction_age == 0)
        s->player_last_interaction_age = 1;

    osrs_tick_special_regen(&s->player);

    if (s->player.potion_timer > 0) s->player.potion_timer--;
    if (s->player.food_timer > 0) s->player.food_timer--;
    if (s->player.karambwan_timer > 0) s->player.karambwan_timer--;
    if (encounter_tick_saturated_heart(&s->player))
        col_mark_live_loadout_dirty(s);

#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_PLAYER_TIMERS);
#endif

    if (!ctx->config.mask_inventory_heads) {
        OsrsInventoryTickIntent intent = col_resolve_inventory_tick_intent(s, actions);
        col_apply_inventory_tick_intent(s, &intent);
    }
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_PLAYER_INVENTORY);
#endif

    int spec_act = actions[COLO_HEAD_SPEC];
    uint8_t equipped_weapon = s->player.equipped[GEAR_SLOT_WEAPON];
    int spec_cost = osrs_spec_cost(equipped_weapon);
    if (spec_act == 1 && spec_cost > 0 && s->player.special_energy >= spec_cost) {
        s->player.spec_armed = 1;
    } else if (spec_act == 2) {
        s->player.spec_armed = 0;
    }

    int spell_act = s->modifiers.draft_pending ? 0 : actions[COLO_HEAD_SPELL];
    if (spell_act == COLO_SPELL_SUMMON_THRALL && s->thrall_recast_cd == 0) {
        int target = col_player_target_slot(s);
        if (col_npc_is_thrall_target(s, target)) col_summon_thrall(s, target);
    } else if (spell_act == COLO_SPELL_DEATH_CHARGE &&
            s->death_charge_cd == 0 && s->death_charge_window_left == 0) {
        s->death_charge_window_left = COLO_DEATH_CHARGE_WINDOW;
    }

    int modifier_act = actions[COLO_HEAD_MODIFIER_SELECT];
    if (modifier_act > 0 && s->modifiers.draft_pending)
        col_modifier_apply_selection(s, modifier_act - 1);

    if (s->modifiers.draft_pending && !s->modifiers.draft_free_movement) return;

    int grapple_act = actions[COLO_HEAD_GRAPPLE_PARRY];
    if (grapple_act > 0 && grapple_act <= COLO_NUM_GRAPPLE_SLOTS)
        col_sol_grapple_parry_attempt(s, grapple_act - 1);

#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_PLAYER_INTENT);
#endif

    int primary_act = actions[COLO_HEAD_PRIMARY];
    ColoWalkCtx walk_ctx;
    OsrsLosQuery los_query;
    ColoPrimaryStepPlan primary_plan = col_primary_step_plan(
        s, ctx, &s->player, &s->interaction, primary_act,
        &s->player_dest_x, &s->player_dest_y, &walk_ctx, &los_query);
    OsrsPlayerStepResult step_result =
        osrs_encounter_player_step(&primary_plan.input);
    if (primary_plan.input.command.kind == OSRS_PLAYER_CMD_TARGET &&
            step_result.target_slot == primary_plan.input.command.target_slot) {
        s->player_last_interaction_target_slot = primary_plan.input.command.target_slot;
        s->player_last_interaction_age = 0;
    }
    col_rebuild_player_collision_flags(s);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_PLAYER_MOVE);
#endif

    if (ctx->config.bis_gear_oracle_mode) col_apply_bis_gear_oracle(s);

    if (can_attack && s->player.attack_timer > 0) s->player.attack_timer--;
    if (can_attack && osrs_interaction_active(&s->interaction) && s->player.attack_timer == 0) {
        int slot = s->interaction.target_slot;
        ColoNPC* npc = &s->npcs[slot];
        if (npc->active) {
            int player_range = col_player_attack_range(s);
            if (encounter_player_can_attack(s->player.x, s->player.y,
                    npc->x, npc->y, col_npc_effective_size(npc),
                    player_range, &los_query)) {

                int armed_kind = (!col_type_is_hazard_entity(npc->type) &&
                                  s->player.spec_armed)
                    ? s->player.equipped[GEAR_SLOT_WEAPON] : 0;
                col_player_attack_target_ctx(s, ctx, slot);
                if (armed_kind > 0 && s->player.used_special_this_tick)
                    ctx->player_spec_fired_kind = armed_kind;
            }
        }
    }

    if (can_attack) col_tick_thrall_and_death_charge(s, ctx);
#ifdef COLO_PROFILE_ENABLED
    COLO_PROFILE_MARK(COLO_PROF_PLAYER_ATTACK);
#endif
}
