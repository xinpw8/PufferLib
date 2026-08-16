#line 76 "encounter_colosseum.h"

typedef enum {
    COLO_LAB_COMMAND_NONE = 0,
    COLO_LAB_COMMAND_RESET,
    COLO_LAB_COMMAND_SET_PLAYER,
    COLO_LAB_COMMAND_SPAWN_NPC,
    COLO_LAB_COMMAND_MOVE_NPC,
    COLO_LAB_COMMAND_DELETE_NPC,
    COLO_LAB_COMMAND_KILL_NPC,
    COLO_LAB_COMMAND_SET_NPC_HP,
    COLO_LAB_COMMAND_SET_NPC_TIMER,
    COLO_LAB_COMMAND_SPAWN_WAVE,
    COLO_LAB_COMMAND_SET_BOSS_PHASE,
    COLO_LAB_COMMAND_START_GRAPPLE,
    COLO_LAB_COMMAND_SET_MODIFIER,
    COLO_LAB_COMMAND_OPEN_DRAFT,
    COLO_LAB_COMMAND_CLEAR_NPCS,
    COLO_LAB_COMMAND_STEP_TICKS,
} ColoLabCommandKind;

typedef enum {
    COLO_LAB_LINE_NONE = 0,
    COLO_LAB_LINE_DUMP,
} ColoLabLineResult;

typedef EncounterLabOptionalInt ColoLabOptionalInt;
#define COLO_LAB_OPTIONAL_INT_UNSET ENCOUNTER_LAB_OPTIONAL_INT_UNSET
#define COLO_LAB_OPTIONAL_INT_SET ENCOUNTER_LAB_OPTIONAL_INT_SET

typedef struct {
    int x;
    int y;
    ColoLabOptionalInt hp;
    ColoLabOptionalInt prayer;
} ColoLabPlayerCommand;

typedef struct {
    int slot;
    ColoNpcType type;
    int x;
    int y;
    ColoLabOptionalInt hp;
    ColoLabOptionalInt timer;
} ColoLabSpawnNpcCommand;

typedef struct {
    int slot;
    int x;
    int y;
} ColoLabMoveNpcCommand;

typedef struct {
    int slot;
} ColoLabNpcSlotCommand;

typedef struct {
    int slot;
    int hp;
} ColoLabNpcHpCommand;

typedef struct {
    int slot;
    int timer;
} ColoLabNpcTimerCommand;

typedef struct {
    int wave;
} ColoLabWaveCommand;

typedef struct {
    int phase;
} ColoLabBossPhaseCommand;

typedef struct {
    int modifier;
    int tier;
} ColoLabModifierCommand;

typedef struct {
    int option[COLO_MODIFIER_DRAFT_OPTIONS];
} ColoLabDraftCommand;

typedef struct {
    int ticks;
} ColoLabStepTicksCommand;

typedef struct {
    uint32_t seed;
} ColoLabResetCommand;

typedef struct {
    ColoLabCommandKind kind;
    union {
        ColoLabResetCommand reset;
        ColoLabPlayerCommand player;
        ColoLabSpawnNpcCommand spawn_npc;
        ColoLabMoveNpcCommand move_npc;
        ColoLabNpcSlotCommand npc_slot;
        ColoLabNpcHpCommand npc_hp;
        ColoLabNpcTimerCommand npc_timer;
        ColoLabWaveCommand wave;
        ColoLabBossPhaseCommand boss_phase;
        ColoLabModifierCommand modifier;
        ColoLabDraftCommand draft;
        ColoLabStepTicksCommand step_ticks;
    } as;
} ColosseumLabCommand;

typedef EncounterLabString ColoLabString;

static void col_lab_abort(const char* fmt, ...) {
    va_list args;
    fprintf(stderr, "colosseum lab: ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}

static void col_lab_require_slot(int slot) {
    if (slot < 0 || slot >= COLO_MAX_NPCS)
        col_lab_abort("npc slot must be in [0,%d], got %d", COLO_MAX_NPCS - 1, slot);
}

static void col_lab_require_type(int type) {
    if (type < 0 || type >= COLO_NUM_NPC_TYPES)
        col_lab_abort("npc type must be in [0,%d], got %d", COLO_NUM_NPC_TYPES - 1, type);
}

static int col_lab_in_arena(
    const ColosseumContext* ctx,
    int x,
    int y
) {
    return encounter_arena_topology_contains(ctx->route_topology, x, y);
}

static void col_lab_clear_transient(
    ColosseumState* s,
    ColosseumContext* ctx
) {
    col_context_clear_render_events(ctx);
    osrs_interaction_init(&s->interaction);
    s->player_last_interaction_target_slot = -1;
    s->player_last_interaction_age = 1;
    s->player_dest_x = -1;
    s->player_dest_y = -1;
    encounter_pending_hit_queue_clear(&s->player_pending_hits);
    s->tick_scratch = (ColoTickScratch){0};
    s->player_attack_npc_idx = -1;
    s->player_attack_dmg = 0;
    s->player_attack_style_id = ATTACK_STYLE_NONE;
    for (int i = 0; i < COLO_MAX_NPCS; i++) {
        ColoNPC* npc = &s->npcs[i];
        npc->attacked_this_tick = 0;
        npc->attack_style_this_tick = ATTACK_STYLE_NONE;
        npc->moved_this_tick = 0;
        npc->hit_landed_this_tick = 0;
        npc->hit_damage = 0;
        npc->hit_was_successful_this_tick = 0;
        encounter_pending_hit_queue_clear(&npc->pending_hits);
    }
}

static void col_lab_refresh_geometry(
    ColosseumState* s,
    const ColosseumContext* ctx
) {
    col_rebuild_player_collision_flags(s);
    col_refresh_current_obs_slots_ctx(s, ctx);
}

static void col_lab_remove_all_npcs(ColosseumState* s) {
    memset(s->npcs, 0, sizeof(s->npcs));
    memset(s->totems, 0, sizeof(s->totems));
    memset(s->bees, 0, sizeof(s->bees));
}

static void col_lab_clear_npcs(
    ColosseumState* s,
    ColosseumContext* ctx
) {
    col_lab_remove_all_npcs(s);
    col_lab_clear_transient(s, ctx);
    col_lab_refresh_geometry(s, ctx);
}

static const char* col_lab_modifier_name(int mod);

static void col_lab_apply_command(
    ColosseumState* s,
    ColosseumContext* ctx,
    const ColosseumLabCommand* cmd
) {
    if (!s || !ctx || !cmd) col_lab_abort("null command");
    encounter_arena_topology_require_finalized(ctx->route_topology);

    switch (cmd->kind) {
        case COLO_LAB_COMMAND_NONE:
            return;

        case COLO_LAB_COMMAND_RESET:
            col_reset_ctx(
                (EncounterState*)s, (EncounterContext*)ctx,
                cmd->as.reset.seed);
            return;

        case COLO_LAB_COMMAND_SET_PLAYER: {
            const ColoLabPlayerCommand* p = &cmd->as.player;
            if (!col_lab_in_arena(ctx, p->x, p->y))
                col_lab_abort("player tile out of arena: (%d,%d)", p->x, p->y);
            s->player.x = p->x;
            s->player.y = p->y;
            if (p->hp.kind == ENCOUNTER_LAB_OPTIONAL_INT_SET) {
                int hp = p->hp.value;
                if (hp < 0) col_lab_abort("player hp must be nonnegative");
                if (hp > s->player.base_hitpoints)
                    col_lab_abort("player hp %d exceeds max %d", hp, s->player.base_hitpoints);
                s->player.current_hitpoints = hp;
            }
            if (p->prayer.kind == ENCOUNTER_LAB_OPTIONAL_INT_SET)
                s->player.prayer = (OverheadPrayer)p->prayer.value;
            col_lab_clear_transient(s, ctx);
            col_lab_refresh_geometry(s, ctx);
            return;
        }

        case COLO_LAB_COMMAND_SPAWN_NPC: {
            ColoLabSpawnNpcCommand spawn_local = cmd->as.spawn_npc;
            ColoLabSpawnNpcCommand* spawn = &spawn_local;
            if (spawn->slot < 0) {
                spawn->slot = col_find_free_npc(s);
                if (spawn->slot < 0)
                    col_lab_abort("no free npc slot (all %d in use)", COLO_MAX_NPCS);
            }
            col_lab_require_slot(spawn->slot);
            col_lab_require_type(spawn->type);
            if (!col_lab_in_arena(ctx, spawn->x, spawn->y))
                col_lab_abort("npc tile out of arena: (%d,%d)", spawn->x, spawn->y);
            if (s->npcs[spawn->slot].active)
                col_deactivate_npc(s, spawn->slot);
            col_init_npc(s, spawn->slot, spawn->type, spawn->x, spawn->y);
            if (spawn->hp.kind == ENCOUNTER_LAB_OPTIONAL_INT_SET) {
                int hp = spawn->hp.value;
                if (hp < 0) col_lab_abort("npc hp must be nonnegative");
                if (hp > s->npcs[spawn->slot].max_hp)
                    col_lab_abort("npc hp %d exceeds max hp %d",
                        hp, s->npcs[spawn->slot].max_hp);
                s->npcs[spawn->slot].hp = hp;
            }
            if (spawn->timer.kind == ENCOUNTER_LAB_OPTIONAL_INT_SET) {
                if (spawn->timer.value < 0)
                    col_lab_abort("npc timer must be nonnegative");
                s->npcs[spawn->slot].attack_timer = spawn->timer.value;
            }
            col_lab_clear_transient(s, ctx);
            col_lab_refresh_geometry(s, ctx);
            return;
        }

        case COLO_LAB_COMMAND_MOVE_NPC: {
            const ColoLabMoveNpcCommand* move = &cmd->as.move_npc;
            col_lab_require_slot(move->slot);
            if (!s->npcs[move->slot].active)
                col_lab_abort("cannot move inactive npc slot %d", move->slot);
            if (!col_lab_in_arena(ctx, move->x, move->y))
                col_lab_abort("npc tile out of arena: (%d,%d)", move->x, move->y);
            col_stamp_npc_collision(s, &s->npcs[move->slot], 0);
            s->npcs[move->slot].x = move->x;
            s->npcs[move->slot].y = move->y;
            s->npcs[move->slot].target_x = move->x;
            s->npcs[move->slot].target_y = move->y;
            col_stamp_npc_collision(s, &s->npcs[move->slot], 1);
            col_lab_clear_transient(s, ctx);
            col_lab_refresh_geometry(s, ctx);
            return;
        }

        case COLO_LAB_COMMAND_DELETE_NPC: {
            int slot = cmd->as.npc_slot.slot;
            col_lab_require_slot(slot);
            col_deactivate_npc(s, slot);
            col_lab_clear_transient(s, ctx);
            col_lab_refresh_geometry(s, ctx);
            return;
        }

        case COLO_LAB_COMMAND_KILL_NPC: {
            int slot = cmd->as.npc_slot.slot;
            col_lab_require_slot(slot);
            if (!s->npcs[slot].active)
                col_lab_abort("cannot kill inactive npc slot %d", slot);
            s->npcs[slot].hp = 0;
            col_deactivate_npc(s, slot);
            col_lab_clear_transient(s, ctx);
            col_lab_refresh_geometry(s, ctx);
            return;
        }

        case COLO_LAB_COMMAND_SET_NPC_HP: {
            int slot = cmd->as.npc_hp.slot;
            int hp = cmd->as.npc_hp.hp;
            col_lab_require_slot(slot);
            if (!s->npcs[slot].active)
                col_lab_abort("cannot set hp for inactive npc slot %d", slot);
            if (hp < 0) col_lab_abort("npc hp must be nonnegative");
            if (hp > s->npcs[slot].max_hp)
                col_lab_abort("npc hp %d exceeds max hp %d", hp, s->npcs[slot].max_hp);
            s->npcs[slot].hp = hp;
            col_lab_refresh_geometry(s, ctx);
            return;
        }

        case COLO_LAB_COMMAND_SET_NPC_TIMER: {
            int slot = cmd->as.npc_timer.slot;
            int timer = cmd->as.npc_timer.timer;
            col_lab_require_slot(slot);
            if (!s->npcs[slot].active)
                col_lab_abort("cannot set timer for inactive npc slot %d", slot);
            if (timer < 0) col_lab_abort("npc timer must be nonnegative");
            s->npcs[slot].attack_timer = timer;
            col_lab_refresh_geometry(s, ctx);
            return;
        }

        case COLO_LAB_COMMAND_SPAWN_WAVE: {
            int wave = cmd->as.wave.wave;
            if (wave < 1 || wave > COLO_NUM_WAVES)
                col_lab_abort("wave must be in [1,%d], got %d", COLO_NUM_WAVES, wave);
            col_lab_remove_all_npcs(s);
            s->modifiers.draft_pending = 0;
            s->modifiers.draft_gates_spawn = 0;
            for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++)
                s->modifiers.draft_options[o] = -1;
            s->wave = wave - 1;
            s->wave_spawn_target = s->wave;
            s->wave_spawn_delay = 0;

            col_spawn_wave(s, ctx);
            col_lab_clear_transient(s, ctx);
            col_lab_refresh_geometry(s, ctx);
            return;
        }

        case COLO_LAB_COMMAND_SET_BOSS_PHASE: {
            int phase = cmd->as.boss_phase.phase;
            if (phase < 0 || phase >= COLO_SOL_NUM_PHASES)
                col_lab_abort("boss phase must be in [0,%d], got %d",
                    COLO_SOL_NUM_PHASES - 1, phase);
            int idx = col_sol_find_idx(s);
            if (idx < 0)
                col_lab_abort("no live Sol Heredit; spawn one or 'wave wave=12' first");

            static const int PHASE_HP[COLO_SOL_NUM_PHASES] = {
                COLO_SOL_HP_MAX,
                (COLO_SOL_HP_MAX * 90) / 100,
                (COLO_SOL_HP_MAX * 75) / 100,
                (COLO_SOL_HP_MAX * 50) / 100,
                (COLO_SOL_HP_MAX * 25) / 100,
                (COLO_SOL_HP_MAX * 10) / 100,
            };
            s->npcs[idx].hp = PHASE_HP[phase];
            if (!s->sol.started) col_sol_begin_boss_arena(s);
            s->sol.boss_idx = idx;

            if (s->sol.phase < phase) {
                while (s->sol.phase < phase)
                    col_sol_enter_phase(s, ctx, s->sol.phase + 1);
                s->sol.immobile_ticks = COLO_SOL_TRANSITION_FREEZE;
                s->sol.force_spear = 1;
            }
            s->sol.phase = phase;
            col_lab_refresh_geometry(s, ctx);
            return;
        }

        case COLO_LAB_COMMAND_START_GRAPPLE: {
            int idx = col_sol_find_idx(s);
            if (idx < 0)
                col_lab_abort("no live Sol Heredit to grapple; 'wave wave=12' first");
            if (!s->sol.started) col_sol_begin_boss_arena(s);
            s->sol.boss_idx = idx;
            col_sol_start_grapple(s);
            return;
        }

        case COLO_LAB_COMMAND_SET_MODIFIER: {
            int mod = cmd->as.modifier.modifier;
            int tier = cmd->as.modifier.tier;
            if (mod < 0 || mod >= COLO_NUM_REAL_MODIFIERS)
                col_lab_abort("modifier id must be in [0,%d], got %d",
                    COLO_NUM_REAL_MODIFIERS - 1, mod);
            int max_tier = COLO_MODIFIER_MAX_TIER[mod];
            if (tier < 0 || tier > max_tier)
                col_lab_abort("tier for %s must be in [0,%d], got %d",
                    col_lab_modifier_name(mod), max_tier, tier);
            if (tier == 0) {
                s->modifiers.active_mask &= ~(1u << (unsigned)mod);
                s->modifiers.tier[mod] = 0;
            } else {
                s->modifiers.active_mask |= (1u << (unsigned)mod);
                s->modifiers.tier[mod] = (uint8_t)tier;
            }
            if ((ColoModifier)mod == COLO_MOD_FRAILTY) col_mod_apply_frailty_hp(s);
            col_lab_refresh_geometry(s, ctx);
            return;
        }

        case COLO_LAB_COMMAND_OPEN_DRAFT: {
            const ColoLabDraftCommand* draft = &cmd->as.draft;
            int offered = 0;
            for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++) {
                s->modifiers.draft_options[o] = draft->option[o];
                if (draft->option[o] >= 0) offered++;
            }
            if (offered == 0) col_lab_abort("draft needs at least one option");

            s->modifiers.draft_pending = 1;
            s->modifiers.draft_gates_spawn = 0;
            return;
        }

        case COLO_LAB_COMMAND_CLEAR_NPCS:
            col_lab_clear_npcs(s, ctx);
            return;

        case COLO_LAB_COMMAND_STEP_TICKS: {
            int ticks = cmd->as.step_ticks.ticks;
            if (ticks < 0) col_lab_abort("step_ticks must be nonnegative");
            int actions[COLO_NUM_ACTION_HEADS] = {0};
            for (int t = 0; t < ticks; t++)
                col_step_ctx(
                    (EncounterState*)s, (EncounterContext*)ctx, actions);
            col_lab_refresh_geometry(s, ctx);
            return;
        }
    }

    col_lab_abort("unknown command kind %d", cmd->kind);
}
