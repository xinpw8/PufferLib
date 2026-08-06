#line 3605 "encounter_inferno.h"
#define INF_HEAD_MOVE       0
#define INF_HEAD_PRAYER     1
#define INF_HEAD_TARGET     2
#define INF_HEAD_EQUIP_BASE 3
#define INF_HEAD_EQUIP_SLOT(slot) (INF_HEAD_EQUIP_BASE + (slot))
#define INF_HEAD_EAT        (INF_HEAD_EQUIP_BASE + NUM_GEAR_SLOTS)
#define INF_HEAD_DRINK      (INF_HEAD_EAT + 1)
#define INF_HEAD_SPELL      (INF_HEAD_DRINK + 1)
#define INF_HEAD_SPEC       (INF_HEAD_SPELL + 1)
#define INF_HEAD_OFFENSIVE  (INF_HEAD_SPEC + 1)
#define INF_INV_CLICK_HEADS (NUM_GEAR_SLOTS + 2)
#define INF_INV_CLICK_DIM   (OSRS_INVENTORY_SIZE + 1)
#define INF_SPEC_DIM        3
#define INF_ACTION_DIMS_INIT { \
    ENCOUNTER_MOVE_ACTIONS, \
    ENCOUNTER_OVERHEAD_DIM_PVE_REDEMPTION, \
    INF_OBS_NPCS+1, \
    INF_INV_CLICK_DIM, INF_INV_CLICK_DIM, INF_INV_CLICK_DIM, \
    INF_INV_CLICK_DIM, INF_INV_CLICK_DIM, INF_INV_CLICK_DIM, \
    INF_INV_CLICK_DIM, INF_INV_CLICK_DIM, INF_INV_CLICK_DIM, \
    INF_INV_CLICK_DIM, INF_INV_CLICK_DIM, INF_INV_CLICK_DIM, \
    INF_INV_CLICK_DIM, \
    3, \
    INF_SPEC_DIM, \
    ENCOUNTER_OFFENSIVE_DIM \
}
static const int INF_ACTION_DIMS[INF_NUM_ACTION_HEADS] = INF_ACTION_DIMS_INIT;
#define INF_ACTION_MASK_SIZE (ENCOUNTER_MOVE_ACTIONS + ENCOUNTER_OVERHEAD_DIM_PVE_REDEMPTION + INF_OBS_NPCS+1 + INF_INV_CLICK_HEADS*INF_INV_CLICK_DIM + 3 + INF_SPEC_DIM + ENCOUNTER_OFFENSIVE_DIM)
#define INF_OVERHEAD_SET_REFRESH_REDEMPTION ENCOUNTER_OVERHEAD_SET_REFRESH_SMITE

static int inf_action_head_mask_offset(int head) {
    int offset = 0;
    for (int h = 0; h < head; h++)
        offset += INF_ACTION_DIMS[h];
    return offset;
}

static int inf_canonical_overhead_action(int action) {
    return action == INF_OVERHEAD_SET_REFRESH_REDEMPTION
        ? ENCOUNTER_OVERHEAD_SET_REFRESH_REDEMPTION
        : action;
}

static int inf_local_overhead_action(int action) {
    if (action == ENCOUNTER_OVERHEAD_SET_REFRESH_REDEMPTION)
        return INF_OVERHEAD_SET_REFRESH_REDEMPTION;
    if (action == ENCOUNTER_OVERHEAD_SET_REFRESH_SMITE)
        return ENCOUNTER_OVERHEAD_NO_CHANGE;
    return action;
}

static void inf_record_offensive_prayer_attack(
    InfernoState* s,
    AttackStyle style,
    float damage_roll
) {
    OffensivePrayer required = encounter_offensive_prayer_for_style(style);
    if (required == OFFENSIVE_PRAYER_NONE) return;

    int style_idx = (int)style;
    assert(style_idx > ATTACK_STYLE_NONE && style_idx <= ATTACK_STYLE_MAGIC);
    int correct = s->player.offensive_prayer == required;
    s->tick_scratch.offensive_prayer_correct = correct;
    s->total_offensive_prayer_attacks++;
    s->total_offensive_prayer_correct += correct;
    s->offensive_prayer_attacks_by_style[style_idx]++;
    s->offensive_prayer_correct_by_style[style_idx] += correct;
    if (correct && damage_roll > 0.0f)
        s->tick_scratch.offensive_prayer_correct_damage_roll += damage_roll;
}

static float inf_record_player_reward_damage(
    InfernoState* s,
    int npc_idx,
    int damage_roll
) {
    if (npc_idx < 0 || npc_idx >= INF_MAX_NPCS || damage_roll <= 0) return 0.0f;

    InfNPC* npc = &s->npcs[npc_idx];
    if (!npc->active || npc->death_ticks > 0 || npc->hp <= 0) return 0.0f;

    float damage = (float)(damage_roll > npc->hp ? npc->hp : damage_roll);
    s->tick_scratch.damage_dealt += damage;
    if (npc->resurrection_count != 0) {
        s->tick_scratch.damage_resurrected += damage;
        return damage;
    }

    switch (npc->type) {
        case INF_NPC_ZUK:
            s->tick_scratch.damage_zuk += damage;
            break;
        case INF_NPC_HEALER_ZUK:
            s->tick_scratch.damage_zuk_healers += damage;
            break;
        case INF_NPC_HEALER_JAD:
            s->tick_scratch.damage_jad_healers += damage;
            break;
        case INF_NPC_JAD:
            s->tick_scratch.damage_jad += damage;
            break;
        case INF_NPC_ZUK_SHIELD:
            break;
        default:
            s->tick_scratch.damage_set += damage;
            break;
    }
    return damage;
}

static int inf_tile_walkable(void* data, int x, int y) {
    InfWalkCtx* wc = (InfWalkCtx*)data;
    InfernoState* s = wc->s;
    const InfernoContext* ctx = wc->ctx;
    if (!inf_in_arena(x, y)) return 0;
    if (inf_blocked_by_pillar(s, x, y, 1)) return 0;
    if (ctx->collision_map)
        if (!collision_tile_walkable(ctx->collision_map, 0,
                x + ctx->world_offset_x, y + ctx->world_offset_y))
            return 0;
    return 1;
}

typedef struct {
    InfernoState* s;
    InfernoContext* ctx;
    int manual_spell;
} InfPlayerStepTargetCtx;

static int inf_lookup_player_attack_target(
    void* data,
    int target_slot,
    OsrsAttackTarget* out
) {
    InfPlayerStepTargetCtx* tc = (InfPlayerStepTargetCtx*)data;
    InfernoState* s = tc->s;
    InfernoContext* ctx = tc->ctx;
    if (target_slot < 0 || target_slot >= INF_MAX_NPCS)
        return 0;

    InfNPC* npc = &s->npcs[target_slot];
    int target_can_manual_phantom =
        inf_is_barrage_spell(tc->manual_spell) &&
        inf_npc_is_phantom_barrage_cast_window(s, ctx, target_slot);
    if (!npc->active ||
            (npc->death_ticks > 0 && !target_can_manual_phantom))
        return 0;

    InfPlayerAttack attack;
    int has_attack = inf_resolve_player_attack_ctx(
        s, ctx, tc->manual_spell, &attack);
    const EncounterLoadoutStats* ls = has_attack
        ? &attack.stats
        : inf_current_loadout_stats(s, ctx);
    *out = (OsrsAttackTarget){
        .slot = target_slot,
        .x = npc->x,
        .y = npc->y,
        .size = npc->size,
        .attack_range = ls->attack_range,
    };
    return 1;
}

#define INF_BREW_HEAL       osrs_brew_heal_amount(99)
#define INF_RESTORE_AMOUNT  osrs_super_restore_amount(99)
#define INF_NPC_DEATH_LINGER_TICKS 3

static void inf_apply_npc_death(InfernoState* s, int npc_idx) {
    InfNPC* npc = &s->npcs[npc_idx];
    if (!osrs_npc_death_linger_start(
            npc->hp, npc->active, &npc->death_ticks, INF_NPC_DEATH_LINGER_TICKS))
        return;
    encounter_pending_hit_queue_clear(&npc->pending_hits);
    s->total_npc_kills++;

    if (npc->type == INF_NPC_BLOB)
        npc->spawn_blob_splits_on_removal = 1;
    inf_store_dead_mob(s, npc);

    if (npc->type == INF_NPC_JAD) {
        for (int j = 0; j < INF_MAX_NPCS; j++) {
            if (s->npcs[j].active &&
                s->npcs[j].type == INF_NPC_HEALER_JAD &&
                inf_npc_healer(&s->npcs[j])->owner_idx == npc_idx) {
                inf_deactivate_npc(s, j);
            }
        }
    }
}

static void inf_apply_blood_barrage_heal_damage(InfernoState* s, int damage) {
    if (damage <= 0) return;
    int hp_before = s->player.current_hitpoints;
    s->player.current_hitpoints += damage * BLOOD_BARRAGE_HEAL_PERCENT / 100;
    if (s->player.current_hitpoints > s->player.base_hitpoints)
        s->player.current_hitpoints = s->player.base_hitpoints;
    s->tick_scratch.blood_heal += s->player.current_hitpoints - hp_before;
}

static void inf_player_pretick(InfernoState* s, InfernoContext* ctx, const int* actions) {
    OffensivePrayer prev_offensive = s->player.offensive_prayer;
    int prayer_act = inf_canonical_overhead_action(actions[INF_HEAD_PRAYER]);
    int offensive_act = actions[INF_HEAD_OFFENSIVE];
    if (encounter_apply_overhead_action(&s->player.prayer, prayer_act)) {
        s->player.prayer_just_activated = 1;
    }
    if (encounter_apply_offensive_action(&s->player.offensive_prayer, offensive_act)) {
        s->player.offensive_prayer_just_activated = 1;
    }
    encounter_drain_all_prayers(
        &s->player, encounter_player_prayer_bonus(&s->player));
    if (s->player.offensive_prayer != prev_offensive) {
        encounter_recompute_loadout_max_hits(s->loadout_stats, INF_NUM_WEAPON_SETS, &s->player);
        if (ctx->human_command_mode)
            inf_refresh_human_loadout_stats(s);
    }
}

static void inf_note_human_weapon_set(InfernoState* s) {
    uint8_t weapon = s->player.equipped[GEAR_SLOT_WEAPON];
    const uint8_t* const* loadouts =
        inf_loadouts_for_profile(s->active_loadout_profile);
    for (int g = 0; g < INF_NUM_WEAPON_SETS; g++) {
        if (loadouts[g][GEAR_SLOT_WEAPON] == weapon) {
            s->weapon_set = (InfWeaponSet)g;
            return;
        }
    }
}

static void inf_sync_consumable_counters_from_cells(InfernoState* s) {
    Player* p = &s->player;
    p->brew_doses = 0;
    p->restore_doses = 0;
    p->bastion_doses = 0;
    p->stamina_doses = 0;
    for (int i = 0; i < OSRS_INVENTORY_SIZE; i++) {
        const OsrsInventoryCell* cell = &s->inventory_cells[i];
        if (cell->raw_osrs_id == 0) continue;
        OsrsConsumableClick click =
            osrs_consumable_click_lookup_raw_osrs_id(cell->raw_osrs_id);
        switch (click.consumable_kind) {
            case OSRS_CONSUMABLE_BREW:          p->brew_doses += cell->dose; break;
            case OSRS_CONSUMABLE_SUPER_RESTORE: p->restore_doses += cell->dose; break;
            case OSRS_CONSUMABLE_BASTION:       p->bastion_doses += cell->dose; break;
            case OSRS_CONSUMABLE_STAMINA:       p->stamina_doses += cell->dose; break;
            default: break;
        }
    }
}

static void inf_apply_drink_one_dose_effect(void* ctx, OsrsConsumableKind kind) {
    InfernoState* s = (InfernoState*)ctx;
    switch (kind) {
        case OSRS_CONSUMABLE_BREW:
            encounter_apply_brew_heal(&s->player, INF_BREW_HEAL);
            s->tick_scratch.brewed = 1;
            encounter_apply_consumable_stat_effect(
                &s->player, ENCOUNTER_CONSUMABLE_STAT_EFFECT_BREW_DRAIN);
            break;
        case OSRS_CONSUMABLE_SUPER_RESTORE:
            encounter_add_prayer_restore(&s->player, INF_RESTORE_AMOUNT);
            encounter_apply_consumable_stat_effect(
                &s->player, ENCOUNTER_CONSUMABLE_STAT_EFFECT_RESTORE);
            encounter_cap_prayer_restore(&s->player);
            break;
        case OSRS_CONSUMABLE_BASTION:
            encounter_bastion_boost(&s->player);
            break;
        case OSRS_CONSUMABLE_STAMINA:
            s->stamina_active_ticks = 200;
            break;
        default:
            fprintf(stderr, "inferno: consumable kind %d not in inferno kit\n", (int)kind);
            abort();
    }
    encounter_recompute_loadout_max_hits(s->loadout_stats, INF_NUM_WEAPON_SETS, &s->player);
    inf_refresh_live_stats(s);
}

static void inf_apply_eat_cell(InfernoState* s, int cell_idx, OsrsConsumableKind kind) {
    FoodType food;
    switch (kind) {
        case OSRS_CONSUMABLE_SHARK_FOOD: food = FOOD_SHARK; break;
        case OSRS_CONSUMABLE_KARAMBWAN:  food = FOOD_KARAMBWAN; break;
        default:
            fprintf(stderr, "inferno: consumable kind %d is not food\n", (int)kind);
            abort();
    }
    osrs_player_eat_food_effects(&s->player, food);
    osrs_inventory_cell_consume_eat(&s->inventory_cells[cell_idx]);
}

static void inf_apply_human_player_commands_ctx(InfernoState* s, InfernoContext* ctx) {
    int did_change_equipment = 0;
    for (int i = 0; i < ctx->human_command_count; i++) {
        const HumanCommand* cmd = &ctx->human_commands[i];
        if (cmd->kind == HUMAN_COMMAND_EQUIP_INVENTORY_ITEM) {
            if (cmd->gear_slot >= 0 && cmd->gear_slot < NUM_GEAR_SLOTS &&
                cmd->item_db_idx >= 0 && cmd->item_db_idx < NUM_ITEMS) {
                int changed = slot_equip_item(&s->player, cmd->gear_slot, (uint8_t)cmd->item_db_idx);
                if (changed) {
                    s->total_gear_switches++;
                    did_change_equipment = 1;
                    if (cmd->gear_slot == GEAR_SLOT_WEAPON) {
                        AttackStyle style = inf_player_equipped_attack_style(s);
                        if (item_supports_ancient_autocast(s->player.equipped[GEAR_SLOT_WEAPON])) {
                            s->player.fight_style = s->player.autocast_defensive
                                ? FIGHT_STYLE_DEFENSIVE_AUTOCAST
                                : FIGHT_STYLE_AUTOCAST;
                        } else {
                            s->player.fight_style = inf_live_fight_style_for(style);
                        }
                        inf_note_human_weapon_set(s);
                    }
                }
            }
        } else if (cmd->kind == HUMAN_COMMAND_FIGHT_STYLE) {
            if (cmd->fight_style >= FIGHT_STYLE_ACCURATE &&
                cmd->fight_style <= FIGHT_STYLE_DEFENSIVE_AUTOCAST) {
                s->player.fight_style = (FightStyle)cmd->fight_style;
                if (cmd->fight_style == FIGHT_STYLE_AUTOCAST ||
                        cmd->fight_style == FIGHT_STYLE_DEFENSIVE_AUTOCAST) {
                    s->player.autocast_enabled = 1;
                    s->player.autocast_defensive =
                        cmd->fight_style == FIGHT_STYLE_DEFENSIVE_AUTOCAST;
                }
                did_change_equipment = 1;
            }
        } else if (cmd->kind == HUMAN_COMMAND_SET_AUTOCAST) {
            if (inf_is_barrage_spell(cmd->autocast_spell)) {
                s->player.autocast_enabled = 1;
                s->player.autocast_defensive = cmd->autocast_defensive != 0;
                s->player.autocast_spell = cmd->autocast_spell;
                if (item_supports_ancient_autocast(s->player.equipped[GEAR_SLOT_WEAPON])) {
                    s->player.fight_style = s->player.autocast_defensive
                        ? FIGHT_STYLE_DEFENSIVE_AUTOCAST
                        : FIGHT_STYLE_AUTOCAST;
                }
                did_change_equipment = 1;
            }
        }
    }
    if (did_change_equipment)
        inf_refresh_human_loadout_stats(s);
}

static void inf_apply_human_player_commands(InfernoState* s) {
    inf_apply_human_player_commands_ctx(s, inf_legacy_context());
}

static void inf_tick_player_ctx(
    InfernoState* s,
    InfernoContext* ctx,
    const int* actions,
    int can_attack
) {
    s->tick_scratch.offensive_prayer_correct = 0;
    if (s->player_last_interaction_age == 0)
        s->player_last_interaction_age = 1;

    if (ctx->human_command_mode) {
        inf_apply_human_player_commands_ctx(s, ctx);
    }

    int manual_spell = inf_spell_action_to_spell(actions[INF_HEAD_SPELL]);

    osrs_tick_special_regen(&s->player);

    if (actions[INF_HEAD_SPEC] == 1)
        s->player.spec_armed = 1;
    else if (actions[INF_HEAD_SPEC] == 2)
        s->player.spec_armed = 0;

    if (s->tick > 0 && s->tick % 60 == 0) {
        int* stats[] = { &s->player.current_ranged, &s->player.current_magic,
                         &s->player.current_attack, &s->player.current_strength,
                         &s->player.current_defence };
        for (int si = 0; si < 5; si++) {
            if (*stats[si] > 99) (*stats[si])--;
            else if (*stats[si] < 99) (*stats[si])++;
        }
        encounter_recompute_loadout_max_hits(s->loadout_stats, INF_NUM_WEAPON_SETS, &s->player);
        inf_refresh_live_stats(s);
        if (ctx->human_command_mode)
            inf_refresh_human_loadout_stats(s);
    }

    if (s->player.potion_timer > 0) s->player.potion_timer--;
    if (s->stamina_active_ticks > 0) s->stamina_active_ticks--;

    if (!ctx->human_command_mode) {
        OsrsInventoryClickActions clicks;
        for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++)
            clicks.equip_by_slot[slot] = actions[INF_HEAD_EQUIP_SLOT(slot)];
        clicks.eat = actions[INF_HEAD_EAT];
        clicks.drink = actions[INF_HEAD_DRINK];
        OsrsInventoryTickIntent intent =
            osrs_resolve_inventory_tick_intent(&s->player, s->inventory_cells, &clicks);
        if (intent.eat_cell >= 0)
            osrs_interaction_check_interrupt(&s->interaction, OSRS_IACT_EAT);
        if (intent.drink_cell >= 0)
            osrs_interaction_check_interrupt(&s->interaction, OSRS_IACT_DRINK);
        for (int slot = 0; slot < NUM_GEAR_SLOTS; slot++) {
            if (intent.equip_cell_by_slot[slot] >= 0) {
                osrs_interaction_check_interrupt(&s->interaction, OSRS_IACT_EQUIP);
                break;
            }
        }
        OsrsInventoryApplyStep click_step;
        while (osrs_inventory_intent_next(&intent, &click_step)) {
            switch (click_step.kind) {
                case OSRS_INVENTORY_APPLY_EQUIP:
                    if (osrs_equip_from_cell(
                            &s->player, s->inventory_cells, click_step.cell_idx) >= 0) {
                        s->total_gear_switches++;
                        inf_note_human_weapon_set(s);
                        inf_refresh_live_stats(s);
                    }
                    break;
                case OSRS_INVENTORY_APPLY_EAT:
                    inf_apply_eat_cell(
                        s, click_step.cell_idx, click_step.resolution.consumable_kind);
                    break;
                case OSRS_INVENTORY_APPLY_DRINK: {
                    OsrsInventoryDrinkConsumeResult drink =
                        osrs_inventory_cell_consume_drink_one_dose(
                            &s->inventory_cells[click_step.cell_idx],
                            click_step.resolution,
                            &s->player.potion_timer,
                            inf_apply_drink_one_dose_effect, s);
                    if (drink.consumed)
                        inf_sync_consumable_counters_from_cells(s);
                    break;
                }
            }
        }
    }

    inf_refresh_current_obs_slots_ctx(s, ctx);

    int target = actions[INF_HEAD_TARGET];
    int has_new_target = 0;
    if (target > 0 && target <= INF_OBS_NPCS) {
        int obs_idx = target - 1;
        int npc_idx = s->current_obs_slots[obs_idx];
        int is_live_target =
            npc_idx >= 0 && npc_idx < INF_MAX_NPCS &&
            s->npcs[npc_idx].active &&
            s->npcs[npc_idx].death_ticks == 0 &&
            s->npcs[npc_idx].type != INF_NPC_ZUK_SHIELD;
        int is_phantom_target =
            npc_idx >= 0 && npc_idx < INF_MAX_NPCS &&
            inf_is_barrage_spell(manual_spell) &&
            inf_npc_is_phantom_barrage_obs_candidate(s, ctx, npc_idx);
        if (is_live_target || is_phantom_target) {
            if (npc_idx < 32 && inf_is_untagged_live_zuk_healer_slot(s, npc_idx)) {
                int is_safe_target =
                    inf_player_behind_zuk_shield_now(s) &&
                    s->player.attack_timer == 0 &&
                    inf_player_can_attack_npc_from_current_tile(s, npc_idx);
                s->total_zuk_untagged_healer_targets++;
                if (is_safe_target)
                    s->total_zuk_safe_untagged_healer_targets++;
                else
                    s->total_zuk_unsafe_untagged_healer_targets++;
                if ((s->rewarded_zuk_healer_target_mask & (1u << npc_idx)) == 0) {
                    if (ctx->config.zuk_untagged_healer_target_bonus_coeff > 0.0f) {
                        s->tick_scratch.zuk_untagged_healer_targets++;
                        s->total_zuk_untagged_healer_target_rewards++;
                        s->rewarded_zuk_healer_target_mask |= 1u << npc_idx;
                    } else if (
                            ctx->config.zuk_safe_untagged_healer_target_bonus_coeff > 0.0f &&
                            is_safe_target) {
                        s->tick_scratch.zuk_safe_untagged_healer_targets++;
                        s->total_zuk_safe_untagged_healer_target_rewards++;
                        s->rewarded_zuk_healer_target_mask |= 1u << npc_idx;
                    }
                }
            }
            osrs_interaction_set(&s->interaction, npc_idx);
            s->player_last_interaction_target_slot = npc_idx;
            s->player_last_interaction_age = 0;
            has_new_target = 1;
        }
    }
    if (ctx->config.zuk_safe_untagged_healer_target_mask &&
            osrs_interaction_active(&s->interaction) &&
            !inf_untagged_zuk_healer_target_is_safe_now(
                s, ctx,
                s->interaction.target_slot)) {
        osrs_interaction_clear(&s->interaction);
    }
    if (ctx->config.zuk_force_safe_untagged_healer_target_mask &&
            osrs_interaction_active(&s->interaction) &&
            inf_has_safe_untagged_zuk_healer_target_now(s, ctx) &&
            !inf_is_safe_untagged_zuk_healer_target_now(
                s, ctx,
                s->interaction.target_slot)) {
        osrs_interaction_clear(&s->interaction);
    }
    int has_explicit_move = (actions[INF_HEAD_MOVE] > 0 || s->player_dest_x >= 0);
    if (osrs_interaction_active(&s->interaction)) {
        int target_slot = s->interaction.target_slot;
        int target_can_manual_phantom =
            inf_is_barrage_spell(manual_spell) &&
            inf_npc_is_phantom_barrage_cast_window(s, ctx, target_slot);
        if (!s->npcs[target_slot].active ||
                (s->npcs[target_slot].death_ticks > 0 &&
                 !target_can_manual_phantom)) {
            osrs_interaction_clear(&s->interaction);
        }
    }

    if (osrs_interaction_active(&s->interaction) &&
            inf_is_live_zuk_healer_slot(s, s->interaction.target_slot)) {
        s->total_zuk_healer_target_ticks++;
        if (s->tick_at_first_zuk_healer_target < 0)
            s->tick_at_first_zuk_healer_target = s->tick;
    }

    int claimed_target_standing =
        has_new_target && osrs_interaction_active(&s->interaction);
    OsrsPlayerCommand command = { .kind = OSRS_PLAYER_CMD_NONE };
    int move_act = actions[INF_HEAD_MOVE];
    if (!claimed_target_standing && has_explicit_move) {
        if (s->player_dest_x >= 0) {
            command.kind = OSRS_PLAYER_CMD_MOVE;
            command.move_kind = OSRS_PLAYER_MOVE_DESTINATION;
        } else if (move_act > 0 && move_act < ENCOUNTER_MOVE_ACTIONS) {
            command.kind = OSRS_PLAYER_CMD_MOVE;
            command.move_kind = OSRS_PLAYER_MOVE_ACTION;
            command.move_action = move_act;
        }
    }
    InfWalkCtx wc = { s, ctx };
    InfPlayerStepTargetCtx target_ctx = {
        .s = s,
        .ctx = ctx,
        .manual_spell = manual_spell,
    };
    OsrsLosQuery los_query = inf_player_los_query(s);
    OsrsPlayerStepInput player_step_input = {
        .player = &s->player,
        .interaction = &s->interaction,
        .target_lookup = inf_lookup_player_attack_target,
        .target_ctx = &target_ctx,
        .command = command,
        .dest_x = &s->player_dest_x,
        .dest_y = &s->player_dest_y,
        .arena = {
            .collision_map = ctx->collision_map,
            .world_offset_x = ctx->world_offset_x,
            .world_offset_y = ctx->world_offset_y,
            .is_walkable = inf_tile_walkable,
            .walkable_ctx = &wc,
            .extra_blocked = inf_pathfind_blocked,
            .blocked_ctx = &wc,
            .los_query = &los_query,
            .arena_base_x = INF_ARENA_MIN_X,
            .arena_base_y = INF_ARENA_MIN_Y,
            .arena_w = INF_ARENA_WIDTH,
            .arena_h = INF_ARENA_HEIGHT,
        },
    };
    (void)osrs_encounter_player_step(&player_step_input);
    inf_rebuild_player_collision_flags(s);

    if (can_attack && s->player.attack_timer > 0) s->player.attack_timer--;
    int has_zuk_healer_target = osrs_interaction_active(&s->interaction) &&
        inf_is_live_zuk_healer_slot(s, s->interaction.target_slot);
    if (has_zuk_healer_target) {
        if (!can_attack) {
            s->total_zuk_healer_cannot_attack_ticks++;
        } else if (s->player.attack_timer > 0) {
            s->total_zuk_healer_cooldown_ticks++;
        } else {
            InfNPC* target_npc = &s->npcs[s->interaction.target_slot];
            InfPlayerAttack healer_attack;
            int has_healer_attack = inf_resolve_player_attack_ctx(s, ctx, manual_spell, &healer_attack);
            if (has_healer_attack && encounter_player_can_attack(s->player.x, s->player.y,
                    target_npc->x, target_npc->y, target_npc->size,
                    healer_attack.stats.attack_range, &los_query)) {
                s->total_zuk_healer_attackable_ticks++;
            } else {
                s->total_zuk_healer_out_of_range_ticks++;
            }
        }
    }
    if (can_attack && osrs_interaction_active(&s->interaction) && s->player.attack_timer == 0) {
        InfNPC* target_npc = &s->npcs[s->interaction.target_slot];
        if (target_npc->active) {
            InfPlayerAttack attack;
            int has_attack = inf_resolve_player_attack_ctx(s, ctx, manual_spell, &attack);
            const EncounterLoadoutStats* ls = &attack.stats;
            int is_magic_attack = attack.is_barrage;
            int weapon_is_blowpipe = inf_player_weapon_is(s, ITEM_TOXIC_BLOWPIPE);
            uint8_t player_weapon = attack.weapon;
            int active_spell = attack.spell;
            int is_manual_dying_barrage =
                target_npc->death_ticks > 0 && inf_is_barrage_spell(manual_spell);

            if (target_npc->death_ticks > 0 && !is_manual_dying_barrage) {
                osrs_interaction_clear(&s->interaction);
            } else if (has_attack && encounter_player_can_attack(s->player.x, s->player.y,
                    target_npc->x, target_npc->y, target_npc->size,
                    ls->attack_range, &los_query)) {
                int is_blowpipe_spec_attack = weapon_is_blowpipe &&
                    s->player.spec_armed && s->player.special_energy >= BLOWPIPE_SPEC_COST;
                EncounterProjectileDistanceMode distance_mode = is_magic_attack
                    ? ENCOUNTER_PROJECTILE_DISTANCE_TARGET_SW_TILE
                    : ENCOUNTER_PROJECTILE_DISTANCE_CLOSEST_TILE;
                int projectile_dist = encounter_projectile_distance(
                    s->player.x, s->player.y, 1,
                    target_npc->x, target_npc->y, target_npc->size,
                    distance_mode);
                EncounterProjectileTiming hit_timing = inf_player_projectile_timing(
                    ls->style, player_weapon, is_blowpipe_spec_attack, projectile_dist);
                int hit_delay = hit_timing.damage_delay_ticks;

                int total_dmg = 0;

                if (is_magic_attack) {
                    OsrsTargetRef target_ref = {
                        .kind = OSRS_TARGET_NPC,
                        .id = s->interaction.target_slot,
                    };
                    OsrsMagicAttackKind magic_kind = (active_spell == ENCOUNTER_SPELL_ICE)
                        ? OSRS_MAGIC_ATTACK_ANCIENT_ICE
                        : OSRS_MAGIC_ATTACK_ANCIENT_BLOOD;
                    OsrsPreparedAttackEffects attack_effects = osrs_prepare_attack_effects(
                        &s->player.equipment_effect_profile,
                        &s->player.item_effect_state,
                        s->player.equipped[GEAR_SLOT_WEAPON],
                        ATTACK_STYLE_MAGIC,
                        magic_kind,
                        target_ref,
                        1,
                        ls->eff_level * (ls->attack_bonus + 64),
                        ls->max_hit,
                        osrs_target_effect_context_none(),
                        s->player.current_hitpoints,
                        s->player.base_hitpoints
                    );

                    BarrageTarget btargets[INF_MAX_NPCS + 1];
                    int bt_count = 0;
                    {
                        const InfNPCStats* ns = &INF_NPC_STATS[target_npc->type];
                        btargets[bt_count++] = (BarrageTarget){
                            .active = 1, .x = target_npc->x, .y = target_npc->y,
                            .magic_level = ns->magic_level, .magic_def_bonus = ns->magic_def_bonus,
                            .npc_idx = s->interaction.target_slot,
                            .frozen_ticks = &s->npcs[s->interaction.target_slot].frozen_ticks,
                            .hit = 0, .damage = 0
                        };
                    }
                    for (int i = 0; i < INF_MAX_NPCS; i++) {
                        if (i == s->interaction.target_slot || !s->npcs[i].active) continue;
                        if (s->npcs[i].type == INF_NPC_ZUK_SHIELD) continue;
                        if (s->npcs[i].death_ticks > 0) continue;
                        const InfNPCStats* ns2 = &INF_NPC_STATS[s->npcs[i].type];
                        btargets[bt_count++] = (BarrageTarget){
                            .active = 1, .x = s->npcs[i].x, .y = s->npcs[i].y,
                            .magic_level = ns2->magic_level, .magic_def_bonus = ns2->magic_def_bonus,
                            .npc_idx = i,
                            .frozen_ticks = &s->npcs[i].frozen_ticks,
                            .hit = 0, .damage = 0
                        };
                    }

                    BarrageResult br = osrs_barrage_resolve(
                        btargets, bt_count, attack_effects.attack_roll, attack_effects.max_hit,
                        &s->rng_state, active_spell, attack_effects.use_double_accuracy);
                    total_dmg = br.total_damage;
                    float reward_damage = 0.0f;
                    int phantom_primary_blood_damage = 0;
                    if (target_npc->death_ticks > 0) {
                        if (active_spell == ENCOUNTER_SPELL_BLOOD && btargets[0].hit)
                            phantom_primary_blood_damage = btargets[0].damage;
                        total_dmg -= btargets[0].damage;
                        btargets[0].hit = 0;
                        btargets[0].damage = 0;
                    }
                    inf_apply_blood_barrage_heal_damage(
                        s, phantom_primary_blood_damage);
                    osrs_finalize_attack_effects(
                        &s->player.equipment_effect_profile,
                        &s->player.item_effect_state,
                        s->player.equipped[GEAR_SLOT_WEAPON],
                        ATTACK_STYLE_MAGIC,
                        magic_kind,
                        target_ref,
                        1,
                        attack_effects.use_double_accuracy,
                        btargets[0].hit,
                        btargets[0].damage,
                        &s->rng_state
                    );

                    for (int i = 0; i < bt_count; i++) {
                        if (!btargets[i].active || !btargets[i].rolled) continue;
                        int nidx = btargets[i].npc_idx;
                        if (s->npcs[nidx].death_ticks > 0) continue;
                        reward_damage += inf_record_player_reward_damage(
                            s, nidx, btargets[i].damage);
                        inf_queue_npc_pending_hit(
                            s, nidx, hit_delay, btargets[i].damage,
                            ATTACK_STYLE_MAGIC, active_spell, btargets[i].hit);
                    }
                    total_dmg = (int)reward_damage;

                } else if (is_blowpipe_spec_attack) {
                    if (!encounter_use_spec(&s->player, BLOWPIPE_SPEC_COST)) abort();
                    osrs_spec_disarm(&s->player.spec_armed);
                    s->player.used_special_this_tick = 1;
                    const InfNPCStats* ns = &INF_NPC_STATS[target_npc->type];
                    int base_att_roll = ls->eff_level * (ls->attack_bonus + 64);
                    total_dmg = osrs_blowpipe_spec_resolve(
                        base_att_roll, ls->max_hit,
                        ns->def_level, ns->ranged_def_bonus, &s->rng_state);
                    int heal = total_dmg * BLOWPIPE_SPEC_HEAL_PCT / 100;
                    s->player.current_hitpoints += heal;
                    if (s->player.current_hitpoints > s->player.base_hitpoints)
                        s->player.current_hitpoints = s->player.base_hitpoints;
                    int reward_damage = (int)inf_record_player_reward_damage(
                        s, s->interaction.target_slot, total_dmg);
                    total_dmg = reward_damage;
                    inf_queue_npc_pending_hit(
                        s, s->interaction.target_slot, hit_delay, total_dmg,
                        ATTACK_STYLE_RANGED, ENCOUNTER_SPELL_NONE, total_dmg > 0);

                } else if (ls->style == ATTACK_STYLE_RANGED) {
                    const InfNPCStats* ns = &INF_NPC_STATS[target_npc->type];
                    OsrsPreparedAttackEffects attack_effects = osrs_prepare_attack_effects(
                        &s->player.equipment_effect_profile,
                        &s->player.item_effect_state,
                        s->player.equipped[GEAR_SLOT_WEAPON],
                        ATTACK_STYLE_RANGED,
                        OSRS_MAGIC_ATTACK_NONE,
                        (OsrsTargetRef){ .kind = OSRS_TARGET_NPC, .id = s->interaction.target_slot },
                        1,
                        ls->eff_level * (ls->attack_bonus + 64),
                        ls->max_hit,
                        osrs_target_effect_context_magic(
                            ns->magic_level,
                            ns->magic_att_bonus),
                        s->player.current_hitpoints,
                        s->player.base_hitpoints
                    );
                    int def_roll = (ns->def_level + 9) * (ns->ranged_def_bonus + 64);
                    int hit_success = encounter_roll_hit_chance(
                        &s->rng_state, attack_effects.attack_roll, def_roll);
                    if (hit_success) {
                        total_dmg = encounter_rand_int(&s->rng_state, attack_effects.max_hit + 1);
                    }
                    int reward_damage = (int)inf_record_player_reward_damage(
                        s, s->interaction.target_slot, total_dmg);
                    total_dmg = reward_damage;
                    inf_queue_npc_pending_hit(
                        s, s->interaction.target_slot, hit_delay, total_dmg,
                        ATTACK_STYLE_RANGED, ENCOUNTER_SPELL_NONE, hit_success);
                }

                s->player.attack_timer = ls->attack_speed;
                inf_record_offensive_prayer_attack(s, ls->style, (float)total_dmg);
                if (target_npc->type == INF_NPC_HEALER_ZUK) {
                    int target_was_untagged = target_npc->aggro_target >= 0;
                    s->total_zuk_healer_attack_fires++;
                    if (target_was_untagged && !is_magic_attack)
                        s->tick_scratch.zuk_untagged_healer_nonmagic_attacks++;
                    if (is_magic_attack)
                        s->tick_scratch.zuk_healer_mage_attack_fires++;
                    if (s->tick_at_first_zuk_healer_attack < 0)
                        s->tick_at_first_zuk_healer_attack = s->tick;
                }

                s->tick_scratch.player_attacked = 1;
                s->player_attack_npc_idx = s->interaction.target_slot;
                s->player_attack_dmg = total_dmg;
                s->player_attack_style_id = ls->style;
                s->player_attack_timing = hit_timing;

                s->player.attack_style_this_tick = ls->style;
                if (ls->style == ATTACK_STYLE_MAGIC) {
                    s->player.magic_type_this_tick = (active_spell == ENCOUNTER_SPELL_ICE) ? 1 : 2;
                }
                if (target_npc->death_ticks > 0)
                    osrs_interaction_clear(&s->interaction);
            }
        }
    }
}

static void inf_tick_player(InfernoState* s, const int* actions, int can_attack) {
    inf_tick_player_ctx(s, inf_legacy_context(), actions, can_attack);
}

static int inf_roll_delayed_jad_damage(
    InfernoState* s,
    const InfernoContext* ctx,
    int attack_style
) {
    const InfNPCStats* stats = &INF_NPC_STATS[INF_NPC_JAD];
    int max_hit = osrs_npc_max_hit(attack_style,
        stats->str_level, stats->range_level,
        stats->melee_str_bonus, stats->ranged_str_bonus,
        stats->magic_base_dmg, stats->magic_dmg_pct);
    if (stats->max_hit_cap > 0 && max_hit > stats->max_hit_cap)
        max_hit = stats->max_hit_cap;

    int dmg = encounter_rand_int(&s->rng_state, max_hit + 1);
    int att_lvl, att_bonus;
    if (attack_style == ATTACK_STYLE_RANGED) {
        att_lvl = stats->range_level;
        att_bonus = stats->range_att_bonus;
    } else {
        att_lvl = stats->magic_level;
        att_bonus = stats->magic_att_bonus;
    }

    int att_roll = osrs_npc_attack_roll(att_lvl, att_bonus);
    const EncounterLoadoutStats* ls = inf_current_loadout_stats(s, ctx);
    int def_roll = encounter_player_def_roll_from_loadout(
        s->player.current_defence, s->player.current_magic,
        ls->def_stab, ls->def_slash, ls->def_crush, ls->def_magic, ls->def_ranged,
        attack_style, stats->melee_style);
    if (!encounter_roll_hit_chance(&s->rng_state, att_roll, def_roll))
        dmg = 0;
    return dmg;
}

static void inf_apply_delayed_prayer_check(
    InfernoState* s,
    const InfernoContext* ctx,
    EncounterPendingHit* hit
) {
    if (encounter_prayer_correct_for_style(s->player.prayer, hit->attack_style)) {
        hit->damage = 0;
        s->tick_scratch.prayer_correct++;
        if (hit->source_npc_type >= 0 && hit->source_npc_type < INF_NUM_NPC_TYPES)
            s->prayer_correct_by_type[hit->source_npc_type]++;
    } else if (hit->source_npc_type == INF_NPC_JAD) {
        hit->damage = inf_roll_delayed_jad_damage(s, ctx, hit->attack_style);
        s->dmg_from_type[INF_NPC_JAD] += (float)hit->damage;
        if (hit->damage > 0) {
            s->last_hit_by_type = INF_NPC_JAD;
            s->tick_scratch.off_prayer_hits++;
        }
    } else if (hit->damage > 0 && hit->attack_style != ATTACK_STYLE_NONE) {
        s->tick_scratch.off_prayer_hits++;
    }
    hit->check_prayer = 0;
}

static void inf_resolve_player_projectiles_on_npcs(InfernoState* s) {
    int blood_heal_acc = 0;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        if (!s->npcs[i].active || s->npcs[i].death_ticks > 0) continue;
        InfNPC* npc = &s->npcs[i];
        for (int h = 0; h < npc->pending_hits.count;) {
            EncounterPendingHit* ph = &npc->pending_hits.hits[h];
            int spell = ph->spell_type;
            int hit_success = ph->hit_success;
            int old_hit_landed = npc->hit_landed_this_tick;
            int old_hit_damage = npc->hit_damage;
            int hp_before = npc->hp;
            int landed = encounter_resolve_npc_pending_hit(
                ph, &npc->hp, &npc->hit_landed_this_tick, &npc->hit_damage,
                &npc->frozen_ticks, &blood_heal_acc, NULL);
            if (!landed) {
                if (!ph->active) {
                    encounter_pending_hit_queue_remove(&npc->pending_hits, h, "inferno npc");
                    continue;
                }
                h++;
                continue;
            }

            if (old_hit_landed)
                npc->hit_damage += old_hit_damage;

            int t = npc->type;
            int hp_after = npc->hp;
            if (hp_before > 0 && hp_after <= 0) {
                if (t == INF_NPC_JAD) s->tick_scratch.kill_jad++;
                else if (t == INF_NPC_HEALER_ZUK) s->tick_scratch.kill_zuk_healer++;
                else if (t != INF_NPC_ZUK && t != INF_NPC_ZUK_SHIELD &&
                         t != INF_NPC_HEALER_JAD) s->tick_scratch.kill_set++;
            }
            npc->hit_spell_type = spell;
            if (hit_success)
                npc->hit_was_successful_this_tick = 1;
            int shield_taggable = inf_is_shield_taggable_slot(s, i);
            if (npc->aggro_target != -1) {
                if (npc->type == INF_NPC_HEALER_ZUK ||
                    npc->type == INF_NPC_HEALER_JAD) {
                    s->tick_scratch.healer_tags++;
                    if (npc->type == INF_NPC_HEALER_ZUK)
                        s->tick_scratch.zuk_healer_tags++;
                }
                if (shield_taggable)
                    s->tick_scratch.shield_tags++;
                npc->aggro_target = -1;
                npc->stun_timer = 2;
            }
            encounter_pending_hit_queue_remove(&npc->pending_hits, h, "inferno npc");
            inf_apply_npc_death(s, i);
            if (!npc->active || npc->death_ticks > 0)
                break;
        }
    }
    inf_apply_blood_barrage_heal_damage(s, blood_heal_acc);
}

static void inf_resolve_player_pending_hits(InfernoState* s) {
    for (int i = 0; i < s->player_pending_hits.count; i++) {
        EncounterPendingHit* hit = &s->player_pending_hits.hits[i];

        hit->ticks_remaining--;
        if (hit->ticks_remaining <= 0) {
            int dmg = hit->damage;
            if (hit->check_prayer) {
                if (encounter_prayer_correct_for_style(s->player.prayer, hit->attack_style)) {
                    dmg = 0;
                    s->tick_scratch.prayer_correct++;
                    if (hit->source_npc_type >= 0 && hit->source_npc_type < INF_NUM_NPC_TYPES)
                        s->prayer_correct_by_type[hit->source_npc_type]++;
                } else if (dmg > 0 && hit->attack_style != ATTACK_STYLE_NONE) {
                    s->tick_scratch.off_prayer_hits++;
                }
            } else if (dmg > 0 && hit->attack_style != ATTACK_STYLE_NONE &&
                       hit->source_npc_type != INF_NPC_JAD) {
                s->tick_scratch.off_prayer_hits++;
            }

            if (dmg > 0 &&
                hit->source_npc_type >= 0 &&
                hit->source_npc_type < INF_NUM_NPC_TYPES) {
                s->last_hit_by_type = hit->source_npc_type;
            }
            if (hit->elysian_reduced) {
                s->player.elysian_proc_this_tick = 1;
            } else {
                int elysian_reduced = 0;
                dmg = inf_apply_elysian_to_player_hit(s, dmg, &elysian_reduced);
                if (elysian_reduced) s->player.elysian_proc_this_tick = 1;
            }
            inf_apply_echo_boots_recoil(s, hit->source_npc_slot, dmg);
            inf_damage_player(s, dmg);
            encounter_pending_hit_queue_remove(&s->player_pending_hits, i, "inferno player");
            i--;
        }
    }
}

static void inf_resolve_jad_prayer_checks_after_player(
    InfernoState* s,
    const InfernoContext* ctx
) {
    for (int i = 0; i < s->player_pending_hits.count; i++) {
        EncounterPendingHit* hit = &s->player_pending_hits.hits[i];
        if (!hit->check_prayer ||
            hit->source_npc_type != INF_NPC_JAD ||
            hit->prayer_check_delay <= 0) {
            continue;
        }
        hit->prayer_check_delay--;
        if (hit->prayer_check_delay == 0)
            inf_apply_delayed_prayer_check(s, ctx, hit);
    }
}
