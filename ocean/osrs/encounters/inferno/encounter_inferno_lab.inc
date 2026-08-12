#line 5567 "encounter_inferno.h"
#define INF_LAB_PILLAR_CONTEXT_RADIUS (INF_PILLAR_SIZE + 2)

typedef enum {
    INF_LAB_COMMAND_NONE = 0,
    INF_LAB_COMMAND_RESET,
    INF_LAB_COMMAND_SET_PLAYER,
    INF_LAB_COMMAND_SPAWN_NPC,
    INF_LAB_COMMAND_MOVE_NPC,
    INF_LAB_COMMAND_DELETE_NPC,
    INF_LAB_COMMAND_KILL_NPC,
    INF_LAB_COMMAND_SET_NPC_HP,
    INF_LAB_COMMAND_SET_NPC_TIMER,
    INF_LAB_COMMAND_SET_PILLAR,
    INF_LAB_COMMAND_SPAWN_WAVE,
    INF_LAB_COMMAND_CLEAR_NPCS,
    INF_LAB_COMMAND_STEP_TICKS,
} InfLabCommandKind;

typedef enum {
    INF_LAB_LINE_NONE = 0,
    INF_LAB_LINE_FORECAST,
    INF_LAB_LINE_DUMP,
} InfLabLineResult;

typedef enum {
    INF_LAB_PILLAR_REMOVED = 0,
    INF_LAB_PILLAR_ACTIVE,
} InfLabPillarState;

typedef EncounterLabOptionalInt InfLabOptionalInt;
#define INF_LAB_OPTIONAL_INT_UNSET ENCOUNTER_LAB_OPTIONAL_INT_UNSET
#define INF_LAB_OPTIONAL_INT_SET ENCOUNTER_LAB_OPTIONAL_INT_SET

typedef struct {
    int x;
    int y;
} InfLabTileCommand;

typedef struct {
    int slot;
    InfNPCType type;
    int x;
    int y;
    InfLabOptionalInt hp;
    InfLabOptionalInt timer;
} InfLabSpawnNpcCommand;

typedef struct {
    int slot;
    int x;
    int y;
} InfLabMoveNpcCommand;

typedef struct {
    int slot;
} InfLabNpcSlotCommand;

typedef struct {
    int slot;
    int hp;
} InfLabNpcHpCommand;

typedef struct {
    int slot;
    int timer;
} InfLabNpcTimerCommand;

typedef struct {
    int pillar_idx;
    InfLabPillarState state;
    InfLabOptionalInt hp;
} InfLabPillarCommand;

typedef struct {
    int wave;
} InfLabWaveCommand;

typedef struct {
    int ticks;
} InfLabStepTicksCommand;

typedef struct {
    uint32_t seed;
} InfLabResetCommand;

typedef struct {
    InfLabCommandKind kind;
    union {
        InfLabResetCommand reset;
        InfLabTileCommand tile;
        InfLabSpawnNpcCommand spawn_npc;
        InfLabMoveNpcCommand move_npc;
        InfLabNpcSlotCommand npc_slot;
        InfLabNpcHpCommand npc_hp;
        InfLabNpcTimerCommand npc_timer;
        InfLabPillarCommand pillar;
        InfLabWaveCommand wave;
        InfLabStepTicksCommand step_ticks;
    } as;
} InfernoLabCommand;

typedef EncounterLabString InfLabString;

static void inf_lab_abort(const char* fmt, ...) {
    va_list args;
    fprintf(stderr, "inferno lab: ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}

static void inf_lab_require_slot(int slot) {
    if (slot < 0 || slot >= INF_MAX_NPCS)
        inf_lab_abort("npc slot must be in [0,%d], got %d", INF_MAX_NPCS - 1, slot);
}

static void inf_lab_require_type(int type) {
    if (type < 0 || type >= INF_NUM_NPC_TYPES)
        inf_lab_abort("npc type must be in [0,%d], got %d", INF_NUM_NPC_TYPES - 1, type);
}

static void inf_lab_require_player_tile(
    InfernoState* s,
    const InfernoContext* ctx,
    int x,
    int y
) {
    if (inf_footprint_blocked_ctx(s, ctx, x, y, 1))
        inf_lab_abort("player tile is blocked: (%d,%d)", x, y);
}

static void inf_lab_clear_transient(InfernoState* s) {
    osrs_interaction_init(&s->interaction);
    s->player_last_interaction_target_slot = -1;
    s->player_last_interaction_age = 1;
    s->player_dest_x = -1;
    s->player_dest_y = -1;
    encounter_pending_hit_queue_clear(&s->player_pending_hits);
    memset(s->pending_sparks, 0, sizeof(s->pending_sparks));
    memset(s->npc_target_hits, 0, sizeof(s->npc_target_hits));
    s->tick_scratch.offensive_prayer_correct_damage_roll = 0.0f;
    s->tick_scratch.offensive_prayer_correct = 0;
    s->tick_scratch.player_attacked = 0;
    s->player_attack_npc_idx = -1;
    s->player_attack_dmg = 0;
    s->player_attack_style_id = ATTACK_STYLE_NONE;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        InfNPC* npc = &s->npcs[i];
        npc->attacked_this_tick = 0;
        npc->attack_style_this_tick = ATTACK_STYLE_NONE;
        npc->attack_visual_target = -1;
        npc->moved_this_tick = 0;
        npc->hit_landed_this_tick = 0;
        npc->hit_damage = 0;
        npc->hit_was_successful_this_tick = 0;
        npc->hit_spell_type = ENCOUNTER_SPELL_NONE;
        encounter_pending_hit_queue_clear(&npc->pending_hits);
    }
}

static void inf_lab_refresh_geometry(
    InfernoState* s,
    const InfernoContext* ctx
) {
    inf_rebuild_npc_collision_flags(s);
    inf_refresh_current_obs_slots_ctx(s, ctx);
}

static void inf_lab_remove_all_npcs(InfernoState* s) {
    memset(s->npcs, 0, sizeof(s->npcs));
    s->dead_mob_count = 0;
}

static void inf_lab_clear_npcs(
    InfernoState* s,
    const InfernoContext* ctx
) {
    inf_lab_remove_all_npcs(s);
    inf_lab_clear_transient(s);
    inf_lab_refresh_geometry(s, ctx);
}

static void inf_lab_apply_command_ctx(
    InfernoState* s,
    InfernoContext* ctx,
    const InfernoLabCommand* cmd
) {
    if (!s || !cmd) inf_lab_abort("null command");

    switch (cmd->kind) {
        case INF_LAB_COMMAND_NONE:
            return;

        case INF_LAB_COMMAND_RESET:
            inf_reset_ctx(
                (EncounterState*)s, (EncounterContext*)ctx, cmd->as.reset.seed);
            return;

        case INF_LAB_COMMAND_SET_PLAYER:
            inf_lab_require_player_tile(s, ctx, cmd->as.tile.x, cmd->as.tile.y);
            s->player.x = cmd->as.tile.x;
            s->player.y = cmd->as.tile.y;
            inf_lab_clear_transient(s);
            inf_lab_refresh_geometry(s, ctx);
            return;

        case INF_LAB_COMMAND_SPAWN_NPC: {
            InfLabSpawnNpcCommand spawn_local = cmd->as.spawn_npc;
            InfLabSpawnNpcCommand* spawn = &spawn_local;
            if (spawn->slot < 0) {
                spawn->slot = inf_find_free_npc(s);
                if (spawn->slot < 0)
                    inf_lab_abort("no free npc slot (all %d in use)", INF_MAX_NPCS);
            }
            inf_lab_require_slot(spawn->slot);
            inf_lab_require_type(spawn->type);
            if (s->npcs[spawn->slot].active)
                inf_deactivate_npc(s, spawn->slot);
            inf_init_npc(s, spawn->slot, spawn->type, spawn->x, spawn->y);
            if (spawn->hp.kind == ENCOUNTER_LAB_OPTIONAL_INT_SET) {
                int hp = spawn->hp.value;
                if (hp < 0) inf_lab_abort("npc hp must be nonnegative");
                if (hp > s->npcs[spawn->slot].max_hp)
                    inf_lab_abort("npc hp %d exceeds max hp %d",
                        hp, s->npcs[spawn->slot].max_hp);
                s->npcs[spawn->slot].hp = hp;
            }
            if (spawn->timer.kind == ENCOUNTER_LAB_OPTIONAL_INT_SET) {
                if (spawn->timer.value < 0)
                    inf_lab_abort("npc timer must be nonnegative");
                s->npcs[spawn->slot].attack_timer = spawn->timer.value;
            }
            inf_lab_clear_transient(s);
            inf_lab_refresh_geometry(s, ctx);
            return;
        }

        case INF_LAB_COMMAND_MOVE_NPC: {
            const InfLabMoveNpcCommand* move = &cmd->as.move_npc;
            inf_lab_require_slot(move->slot);
            if (!s->npcs[move->slot].active)
                inf_lab_abort("cannot move inactive npc slot %d", move->slot);
            s->npcs[move->slot].x = move->x;
            s->npcs[move->slot].y = move->y;
            s->npcs[move->slot].target_x = move->x;
            s->npcs[move->slot].target_y = move->y;
            inf_lab_clear_transient(s);
            inf_lab_refresh_geometry(s, ctx);
            return;
        }

        case INF_LAB_COMMAND_DELETE_NPC: {
            int slot = cmd->as.npc_slot.slot;
            inf_lab_require_slot(slot);
            inf_deactivate_npc(s, slot);
            inf_lab_clear_transient(s);
            inf_lab_refresh_geometry(s, ctx);
            return;
        }

        case INF_LAB_COMMAND_KILL_NPC: {
            int slot = cmd->as.npc_slot.slot;
            inf_lab_require_slot(slot);
            if (!s->npcs[slot].active)
                inf_lab_abort("cannot kill inactive npc slot %d", slot);
            s->npcs[slot].hp = 0;
            inf_apply_npc_death(s, slot);
            inf_lab_refresh_geometry(s, ctx);
            return;
        }

        case INF_LAB_COMMAND_SET_NPC_HP: {
            int slot = cmd->as.npc_hp.slot;
            int hp = cmd->as.npc_hp.hp;
            inf_lab_require_slot(slot);
            if (!s->npcs[slot].active)
                inf_lab_abort("cannot set hp for inactive npc slot %d", slot);
            if (hp < 0) inf_lab_abort("npc hp must be nonnegative");
            if (hp > s->npcs[slot].max_hp)
                inf_lab_abort("npc hp %d exceeds max hp %d",
                    hp, s->npcs[slot].max_hp);
            s->npcs[slot].hp = hp;
            inf_lab_refresh_geometry(s, ctx);
            return;
        }

        case INF_LAB_COMMAND_SET_NPC_TIMER: {
            int slot = cmd->as.npc_timer.slot;
            int timer = cmd->as.npc_timer.timer;
            inf_lab_require_slot(slot);
            if (!s->npcs[slot].active)
                inf_lab_abort("cannot set timer for inactive npc slot %d", slot);
            if (timer < 0) inf_lab_abort("npc timer must be nonnegative");
            s->npcs[slot].attack_timer = timer;
            inf_lab_refresh_geometry(s, ctx);
            return;
        }

        case INF_LAB_COMMAND_SET_PILLAR: {
            const InfLabPillarCommand* pillar = &cmd->as.pillar;
            if (pillar->pillar_idx < 0 || pillar->pillar_idx >= INF_NUM_PILLARS)
                inf_lab_abort("pillar idx must be in [0,%d], got %d",
                    INF_NUM_PILLARS - 1, pillar->pillar_idx);
            s->pillars[pillar->pillar_idx].active =
                pillar->state == INF_LAB_PILLAR_ACTIVE;
            if (pillar->state == INF_LAB_PILLAR_ACTIVE) {
                if (pillar->hp.kind == ENCOUNTER_LAB_OPTIONAL_INT_UNSET) {
                    s->pillars[pillar->pillar_idx].hp = INF_PILLAR_HP;
                } else if (pillar->hp.value <= 0 ||
                        pillar->hp.value > INF_PILLAR_HP) {
                    inf_lab_abort("pillar hp must be in [1,%d], got %d",
                        INF_PILLAR_HP, pillar->hp.value);
                } else {
                    s->pillars[pillar->pillar_idx].hp = pillar->hp.value;
                }
            } else {
                s->pillars[pillar->pillar_idx].hp = 0;
            }
            inf_reset_npc_player_los_frame(ctx);
            inf_lab_clear_transient(s);
            inf_lab_refresh_geometry(s, ctx);
            return;
        }

        case INF_LAB_COMMAND_SPAWN_WAVE:
            inf_require_valid_public_wave(cmd->as.wave.wave);
            inf_lab_remove_all_npcs(s);
            s->wave = cmd->as.wave.wave - 1;
            s->wave_spawn_target = s->wave;
            s->wave_spawn_delay = 0;
            s->wave_ready_delay = INF_START_READY_TICKS;
            inf_spawn_wave(s);
            inf_lab_clear_transient(s);
            inf_lab_refresh_geometry(s, ctx);
            return;

        case INF_LAB_COMMAND_CLEAR_NPCS:
            inf_lab_clear_npcs(s, ctx);
            return;

        case INF_LAB_COMMAND_STEP_TICKS: {
            int ticks = cmd->as.step_ticks.ticks;
            if (ticks < 0) inf_lab_abort("step_ticks must be nonnegative");
            int actions[INF_NUM_ACTION_HEADS] = {0};
            for (int t = 0; t < ticks; t++)
                inf_step_ctx((EncounterState*)s, (EncounterContext*)ctx, actions);
            inf_lab_refresh_geometry(s, ctx);
            return;
        }
    }

    inf_lab_abort("unknown command kind %d", cmd->kind);
}


static int inf_lab_nearest_pillar_idx(const InfernoState* s, int x, int y) {
    int best_idx = -1;
    int best_dist = INT32_MAX;
    for (int p = 0; p < INF_NUM_PILLARS; p++) {
        int cx = s->pillars[p].x + INF_PILLAR_SIZE / 2;
        int cy = s->pillars[p].y + INF_PILLAR_SIZE / 2;
        int dist = abs(cx - x) + abs(cy - y);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = p;
        }
    }
    return best_dist <= INF_LAB_PILLAR_CONTEXT_RADIUS ? best_idx : -1;
}

static const char* inf_lab_npc_type_name(int type) {
    switch (type) {
        case INF_NPC_NIBBLER: return "nibbler";
        case INF_NPC_BAT: return "bat";
        case INF_NPC_BLOB: return "blob";
        case INF_NPC_BLOB_MELEE: return "blob_melee";
        case INF_NPC_BLOB_RANGE: return "blob_range";
        case INF_NPC_BLOB_MAGE: return "blob_mage";
        case INF_NPC_MELEER: return "meleer";
        case INF_NPC_RANGER: return "ranger";
        case INF_NPC_MAGER: return "mager";
        case INF_NPC_JAD: return "jad";
        case INF_NPC_ZUK: return "zuk";
        case INF_NPC_HEALER_JAD: return "healer_jad";
        case INF_NPC_HEALER_ZUK: return "healer_zuk";
        case INF_NPC_ZUK_SHIELD: return "zuk_shield";
        default: return "unknown";
    }
}

static int inf_lab_parse_npc_type(const char* value) {
    if (strcmp(value, "nibbler") == 0) return INF_NPC_NIBBLER;
    if (strcmp(value, "bat") == 0) return INF_NPC_BAT;
    if (strcmp(value, "blob") == 0) return INF_NPC_BLOB;
    if (strcmp(value, "blob_melee") == 0) return INF_NPC_BLOB_MELEE;
    if (strcmp(value, "blob_range") == 0) return INF_NPC_BLOB_RANGE;
    if (strcmp(value, "blob_mage") == 0) return INF_NPC_BLOB_MAGE;
    if (strcmp(value, "meleer") == 0 || strcmp(value, "melee") == 0)
        return INF_NPC_MELEER;
    if (strcmp(value, "ranger") == 0 || strcmp(value, "range") == 0)
        return INF_NPC_RANGER;
    if (strcmp(value, "mager") == 0 || strcmp(value, "mage") == 0)
        return INF_NPC_MAGER;
    if (strcmp(value, "jad") == 0) return INF_NPC_JAD;
    if (strcmp(value, "zuk") == 0) return INF_NPC_ZUK;
    if (strcmp(value, "healer_jad") == 0 || strcmp(value, "jad_healer") == 0)
        return INF_NPC_HEALER_JAD;
    if (strcmp(value, "healer_zuk") == 0 || strcmp(value, "zuk_healer") == 0)
        return INF_NPC_HEALER_ZUK;
    if (strcmp(value, "zuk_shield") == 0 || strcmp(value, "shield") == 0)
        return INF_NPC_ZUK_SHIELD;

    char* end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno == 0 && end && *end == '\0') {
        inf_lab_require_type((int)parsed);
        return (int)parsed;
    }
    inf_lab_abort("unknown npc type %s", value);
    return 0;
}

static InfLabOptionalInt inf_lab_parse_optional_hp_value(const char* value) {
    return encounter_lab_parse_optional_full_int("inferno lab", value);
}

typedef struct {
    int slot;
    int type;
    int x;
    int y;
    InfLabOptionalInt hp;
    InfLabOptionalInt timer;
    int pillar_idx;
    InfLabPillarState pillar_state;
    int wave;
    int ticks;
    uint32_t seed;
} InfLabParsedArgs;

static InfLabParsedArgs inf_lab_parsed_args_default(void) {
    return (InfLabParsedArgs){
        .slot = -1,
        .type = -1,
        .x = INT32_MIN,
        .y = INT32_MIN,
        .hp = { .kind = ENCOUNTER_LAB_OPTIONAL_INT_UNSET },
        .timer = { .kind = ENCOUNTER_LAB_OPTIONAL_INT_UNSET },
        .pillar_idx = -1,
        .pillar_state = INF_LAB_PILLAR_ACTIVE,
        .wave = -1,
        .ticks = 0,
        .seed = 0,
    };
}

static void inf_lab_apply_script_arg(
    InfLabParsedArgs* args,
    InfLabCommandKind kind,
    const char* command,
    const char* key,
    const char* value
) {
    switch (kind) {
        case INF_LAB_COMMAND_RESET:
            if (strcmp(key, "seed") == 0) {
                args->seed = encounter_lab_parse_seed_value("inferno lab", value);
                return;
            }
            break;

        case INF_LAB_COMMAND_SET_PLAYER:
            if (strcmp(key, "x") == 0) {
                args->x = encounter_lab_parse_int_value("inferno lab", value);
                return;
            }
            if (strcmp(key, "y") == 0) {
                args->y = encounter_lab_parse_int_value("inferno lab", value);
                return;
            }
            break;

        case INF_LAB_COMMAND_SPAWN_NPC:
            if (strcmp(key, "slot") == 0) {
                args->slot = encounter_lab_parse_int_value("inferno lab", value);
                return;
            }
            if (strcmp(key, "type") == 0) {
                args->type = inf_lab_parse_npc_type(value);
                return;
            }
            if (strcmp(key, "x") == 0) {
                args->x = encounter_lab_parse_int_value("inferno lab", value);
                return;
            }
            if (strcmp(key, "y") == 0) {
                args->y = encounter_lab_parse_int_value("inferno lab", value);
                return;
            }
            if (strcmp(key, "hp") == 0) {
                args->hp = inf_lab_parse_optional_hp_value(value);
                return;
            }
            if (strcmp(key, "timer") == 0) {
                args->timer = encounter_lab_optional_int_set(
                    encounter_lab_parse_int_value("inferno lab", value));
                return;
            }
            break;

        case INF_LAB_COMMAND_MOVE_NPC:
            if (strcmp(key, "slot") == 0) {
                args->slot = encounter_lab_parse_int_value("inferno lab", value);
                return;
            }
            if (strcmp(key, "x") == 0) {
                args->x = encounter_lab_parse_int_value("inferno lab", value);
                return;
            }
            if (strcmp(key, "y") == 0) {
                args->y = encounter_lab_parse_int_value("inferno lab", value);
                return;
            }
            break;

        case INF_LAB_COMMAND_DELETE_NPC:
        case INF_LAB_COMMAND_KILL_NPC:
            if (strcmp(key, "slot") == 0) {
                args->slot = encounter_lab_parse_int_value("inferno lab", value);
                return;
            }
            break;

        case INF_LAB_COMMAND_SET_NPC_HP:
            if (strcmp(key, "slot") == 0) {
                args->slot = encounter_lab_parse_int_value("inferno lab", value);
                return;
            }
            if (strcmp(key, "hp") == 0) {
                args->hp = encounter_lab_optional_int_set(
                    encounter_lab_parse_int_value("inferno lab", value));
                return;
            }
            break;

        case INF_LAB_COMMAND_SET_NPC_TIMER:
            if (strcmp(key, "slot") == 0) {
                args->slot = encounter_lab_parse_int_value("inferno lab", value);
                return;
            }
            if (strcmp(key, "timer") == 0) {
                args->timer = encounter_lab_optional_int_set(
                    encounter_lab_parse_int_value("inferno lab", value));
                return;
            }
            break;

        case INF_LAB_COMMAND_SET_PILLAR:
            if (strcmp(key, "idx") == 0 || strcmp(key, "pillar_idx") == 0) {
                args->pillar_idx = encounter_lab_parse_int_value("inferno lab", value);
                return;
            }
            if (strcmp(key, "active") == 0) {
                int active = encounter_lab_parse_int_value("inferno lab", value);
                if (active != 0 && active != 1)
                    inf_lab_abort("pillar active must be 0 or 1, got %d", active);
                args->pillar_state = active
                    ? INF_LAB_PILLAR_ACTIVE
                    : INF_LAB_PILLAR_REMOVED;
                return;
            }
            if (strcmp(key, "hp") == 0) {
                args->hp = inf_lab_parse_optional_hp_value(value);
                return;
            }
            break;

        case INF_LAB_COMMAND_SPAWN_WAVE:
            if (strcmp(key, "wave") == 0 || strcmp(key, "public") == 0) {
                args->wave = encounter_lab_parse_int_value("inferno lab", value);
                return;
            }
            break;

        case INF_LAB_COMMAND_STEP_TICKS:
            if (strcmp(key, "ticks") == 0) {
                args->ticks = encounter_lab_parse_int_value("inferno lab", value);
                return;
            }
            break;

        case INF_LAB_COMMAND_NONE:
        case INF_LAB_COMMAND_CLEAR_NPCS:
            break;
    }

    inf_lab_abort("unknown key %s for command %s", key, command);
}

static InfernoLabCommand inf_lab_build_script_command(
    InfLabCommandKind kind, const InfLabParsedArgs* args, const char* command
) {
    InfernoLabCommand out;
    memset(&out, 0, sizeof(out));

    switch (kind) {
        case INF_LAB_COMMAND_NONE:
            out.kind = INF_LAB_COMMAND_NONE;
            return out;

        case INF_LAB_COMMAND_RESET:
            out.kind = INF_LAB_COMMAND_RESET;
            out.as.reset.seed = args->seed;
            return out;

        case INF_LAB_COMMAND_SET_PLAYER:
            if (args->x == INT32_MIN || args->y == INT32_MIN)
                inf_lab_abort("command %s requires x and y", command);
            out.kind = INF_LAB_COMMAND_SET_PLAYER;
            out.as.tile.x = args->x;
            out.as.tile.y = args->y;
            return out;

        case INF_LAB_COMMAND_SPAWN_NPC:
            if (args->x == INT32_MIN || args->y == INT32_MIN)
                inf_lab_abort("command %s requires x and y", command);
            if (args->type < 0)
                inf_lab_abort("npc command requires type");
            out.kind = INF_LAB_COMMAND_SPAWN_NPC;
            out.as.spawn_npc.slot = args->slot;
            out.as.spawn_npc.type = (InfNPCType)args->type;
            out.as.spawn_npc.x = args->x;
            out.as.spawn_npc.y = args->y;
            out.as.spawn_npc.hp = args->hp;
            out.as.spawn_npc.timer = args->timer;
            return out;

        case INF_LAB_COMMAND_MOVE_NPC:
            if (args->x == INT32_MIN || args->y == INT32_MIN)
                inf_lab_abort("command %s requires x and y", command);
            if (args->slot < 0)
                inf_lab_abort("command %s requires slot", command);
            out.kind = INF_LAB_COMMAND_MOVE_NPC;
            out.as.move_npc.slot = args->slot;
            out.as.move_npc.x = args->x;
            out.as.move_npc.y = args->y;
            return out;

        case INF_LAB_COMMAND_DELETE_NPC:
        case INF_LAB_COMMAND_KILL_NPC:
            if (args->slot < 0)
                inf_lab_abort("command %s requires slot", command);
            out.kind = kind;
            out.as.npc_slot.slot = args->slot;
            return out;

        case INF_LAB_COMMAND_SET_NPC_HP:
            if (args->slot < 0)
                inf_lab_abort("command %s requires slot", command);
            if (args->hp.kind != ENCOUNTER_LAB_OPTIONAL_INT_SET)
                inf_lab_abort("set_npc_hp requires hp");
            out.kind = INF_LAB_COMMAND_SET_NPC_HP;
            out.as.npc_hp.slot = args->slot;
            out.as.npc_hp.hp = args->hp.value;
            return out;

        case INF_LAB_COMMAND_SET_NPC_TIMER:
            if (args->slot < 0)
                inf_lab_abort("command %s requires slot", command);
            if (args->timer.kind != ENCOUNTER_LAB_OPTIONAL_INT_SET)
                inf_lab_abort("set_npc_timer requires timer");
            out.kind = INF_LAB_COMMAND_SET_NPC_TIMER;
            out.as.npc_timer.slot = args->slot;
            out.as.npc_timer.timer = args->timer.value;
            return out;

        case INF_LAB_COMMAND_SET_PILLAR:
            if (args->pillar_idx < 0)
                inf_lab_abort("set_pillar requires idx");
            out.kind = INF_LAB_COMMAND_SET_PILLAR;
            out.as.pillar.pillar_idx = args->pillar_idx;
            out.as.pillar.state = args->pillar_state;
            out.as.pillar.hp = args->hp;
            return out;

        case INF_LAB_COMMAND_SPAWN_WAVE:
            if (args->wave < 0)
                inf_lab_abort("spawn_wave requires wave");
            out.kind = INF_LAB_COMMAND_SPAWN_WAVE;
            out.as.wave.wave = args->wave;
            return out;

        case INF_LAB_COMMAND_CLEAR_NPCS:
            out.kind = INF_LAB_COMMAND_CLEAR_NPCS;
            return out;

        case INF_LAB_COMMAND_STEP_TICKS:
            out.kind = INF_LAB_COMMAND_STEP_TICKS;
            out.as.step_ticks.ticks = args->ticks;
            return out;
    }

    inf_lab_abort("unknown command kind %d", kind);
    out.kind = INF_LAB_COMMAND_NONE;
    return out;
}

static void inf_lab_append_forecast_tick_json(
    InfLabString* out, const InfStepOutForecastTick* tick, int tick_idx
) {
    int style_mask = inf_step_out_forecast_tick_style_mask(tick);
    encounter_lab_string_append(out,
        "{\"t\":%d,\"style_mask\":%d,\"melee\":%d,\"ranged\":%d,"
        "\"magic\":%d,\"blob_scan\":%d,\"ranger\":%d,\"mager\":%d,"
        "\"max_hit\":%d}",
        tick_idx + 1, style_mask, tick->melee_count, tick->ranged_count,
        tick->magic_count, tick->blob_scan_count, tick->ranger_count,
        tick->mager_count, tick->max_hit);
}

static int inf_lab_forecast_action_has_ranger_mager_same_tick(
    const InfStepOutForecastAction* action
) {
    for (int tick_idx = 0; tick_idx < INF_STEP_OUT_FORECAST_HORIZON; tick_idx++) {
        if (action->ticks[tick_idx].ranger_count > 0 &&
                action->ticks[tick_idx].mager_count > 0) {
            return 1;
        }
    }
    return 0;
}

static void inf_lab_append_forecast_json(
    InfernoState* s,
    const InfernoContext* ctx,
    InfLabString* out
) {
    InfStepOutForecast forecast;
    inf_build_step_out_forecast_ctx(s, ctx, &forecast);
    encounter_lab_string_append(out, "\"forecast\":{\"horizon\":%d,\"actions\":[",
        INF_STEP_OUT_FORECAST_HORIZON);
    for (int action_idx = 0; action_idx < ENCOUNTER_MOVE_ACTIONS; action_idx++) {
        const InfStepOutForecastAction* action = &forecast.actions[action_idx];
        if (action_idx > 0) encounter_lab_string_append(out, ",");
        encounter_lab_string_append(out,
            "{\"action\":%d,\"dx\":%d,\"dy\":%d,\"valid\":%d,"
            "\"land_x\":%d,\"land_y\":%d,\"same_tick_mixed\":%d,"
            "\"ranger_mager_same_tick\":%d,\"ranger_mager_offtick\":%d,"
            "\"melee_fallback\":%d,\"ticks\":[",
            action_idx,
            ENCOUNTER_MOVE_TARGET_DX[action_idx],
            ENCOUNTER_MOVE_TARGET_DY[action_idx],
            action->valid, action->land_x, action->land_y,
            action->same_tick_mixed_style_conflict,
            inf_lab_forecast_action_has_ranger_mager_same_tick(action),
            action->ranger_mager_offtick_opportunity,
            action->melee_fallback_exposure);
        for (int tick_idx = 0; tick_idx < INF_STEP_OUT_FORECAST_HORIZON; tick_idx++) {
            if (tick_idx > 0) encounter_lab_string_append(out, ",");
            inf_lab_append_forecast_tick_json(out, &action->ticks[tick_idx], tick_idx);
        }
        encounter_lab_string_append(out, "]}");
    }
    encounter_lab_string_append(out, "]}");
}

static void inf_lab_append_npcs_json(
    InfernoState* s,
    const InfernoContext* ctx,
    InfLabString* out
) {
    encounter_lab_string_append(out, "\"npcs\":[");
    int emitted = 0;
    for (int i = 0; i < INF_MAX_NPCS; i++) {
        InfNPC* npc = &s->npcs[i];
        if (!npc->active) continue;
        if (emitted > 0) encounter_lab_string_append(out, ",");
        int los = npc->death_ticks > 0 ? 0 : inf_npc_has_los_ctx(s, ctx, i);
        const EncounterPendingHit* pending =
            encounter_pending_hit_queue_earliest(&npc->pending_hits);
        int pending_ticks = pending ? pending->ticks_remaining : 0;
        int pending_style = pending ? pending->attack_style : ATTACK_STYLE_NONE;
        int pending_spell = pending ? pending->spell_type : ENCOUNTER_SPELL_NONE;
        int pending_hit_success = pending ? pending->hit_success : 0;
        int pending_damage = encounter_pending_hit_queue_damage_sum(&npc->pending_hits);
        encounter_lab_string_append(out,
            "{\"slot\":%d,\"type\":\"%s\",\"type_id\":%d,\"def_id\":%d,"
            "\"x\":%d,\"y\":%d,\"size\":%d,\"hp\":%d,\"max_hp\":%d,"
            "\"timer\":%d,\"stun\":%d,\"frozen\":%d,\"death_ticks\":%d,"
            "\"los_to_player\":%d,\"pending_count\":%d,"
            "\"pending_earliest_ticks\":%d,\"pending_style\":%d,"
            "\"pending_spell\":%d,\"pending_hit_success\":%d,"
            "\"pending_damage\":%d}",
            i, inf_lab_npc_type_name(npc->type), npc->type,
            INF_NPC_DEF_IDS[npc->type], npc->x, npc->y,
            inf_npc_effective_size(npc), npc->hp, npc->max_hp,
            npc->attack_timer, npc->stun_timer, npc->frozen_ticks,
            npc->death_ticks, los, npc->pending_hits.count,
            pending_ticks, pending_style, pending_spell,
            pending_hit_success, pending_damage);
        emitted++;
    }
    encounter_lab_string_append(out, "]");
}

static void inf_lab_append_pillars_json(const InfernoState* s, InfLabString* out) {
    encounter_lab_string_append(out, "\"pillars\":[");
    for (int p = 0; p < INF_NUM_PILLARS; p++) {
        if (p > 0) encounter_lab_string_append(out, ",");
        encounter_lab_string_append(out,
            "{\"idx\":%d,\"x\":%d,\"y\":%d,\"size\":%d,"
            "\"active\":%d,\"hp\":%d}",
            p, s->pillars[p].x, s->pillars[p].y, INF_PILLAR_SIZE,
            s->pillars[p].active, s->pillars[p].hp);
    }
    encounter_lab_string_append(out, "]");
}

static char* inf_lab_alloc_json_ctx(
    InfernoState* s,
    const InfernoContext* ctx
) {
    InfLabString out;
    encounter_lab_string_init(&out, "inferno lab");
    encounter_lab_string_append(&out,
        "{\"tick\":%d,\"wave\":%d,"
        "\"player\":{\"x\":%d,\"y\":%d,\"hp\":%d,\"prayer\":%d,"
        "\"attack_timer\":%d,\"attacked\":%d,\"attack_target\":%d,"
        "\"attack_style\":%d,\"attack_damage\":%d},",
        s->tick, s->wave + 1,
        s->player.x, s->player.y,
        s->player.current_hitpoints, s->player.current_prayer,
        s->player.attack_timer, s->tick_scratch.player_attacked,
        s->player_attack_npc_idx, s->player_attack_style_id,
        s->player_attack_dmg);
    inf_lab_append_pillars_json(s, &out);
    encounter_lab_string_append(&out, ",");
    inf_lab_append_npcs_json(s, ctx, &out);
    encounter_lab_string_append(&out, ",");
    inf_lab_append_forecast_json(s, ctx, &out);
    encounter_lab_string_append(&out, "}");
    return out.data;
}


static InfLabCommandKind inf_lab_command_kind_for(const char* command) {
    static const EncounterLabCommandAlias aliases[] = {
        { "reset", INF_LAB_COMMAND_RESET },
        { "player", INF_LAB_COMMAND_SET_PLAYER },
        { "set_player", INF_LAB_COMMAND_SET_PLAYER },
        { "npc", INF_LAB_COMMAND_SPAWN_NPC },
        { "spawn_npc", INF_LAB_COMMAND_SPAWN_NPC },
        { "move_npc", INF_LAB_COMMAND_MOVE_NPC },
        { "delete_npc", INF_LAB_COMMAND_DELETE_NPC },
        { "kill_npc", INF_LAB_COMMAND_KILL_NPC },
        { "set_npc_hp", INF_LAB_COMMAND_SET_NPC_HP },
        { "set_npc_timer", INF_LAB_COMMAND_SET_NPC_TIMER },
        { "pillar", INF_LAB_COMMAND_SET_PILLAR },
        { "set_pillar", INF_LAB_COMMAND_SET_PILLAR },
        { "wave", INF_LAB_COMMAND_SPAWN_WAVE },
        { "spawn_wave", INF_LAB_COMMAND_SPAWN_WAVE },
        { "clear_npcs", INF_LAB_COMMAND_CLEAR_NPCS },
        { "step", INF_LAB_COMMAND_STEP_TICKS },
        { "step_ticks", INF_LAB_COMMAND_STEP_TICKS },
    };
    return (InfLabCommandKind)encounter_lab_lookup_command_kind(
        "inferno lab", command, aliases, sizeof(aliases) / sizeof(aliases[0]));
}

static InfLabLineResult inf_lab_apply_script_line_impl_ctx(
    InfernoState* s,
    InfernoContext* ctx,
    const char* line,
    char** out_json
) {
    EncounterLabLine lex = encounter_lab_line_begin("inferno lab", line);
    if (!lex.command) {
        free(lex.buffer);
        return INF_LAB_LINE_NONE;
    }

    if (strcmp(lex.command, "forecast") == 0) {
        InfStepOutForecast forecast;
        inf_build_step_out_forecast_ctx(s, ctx, &forecast);
        (void)forecast;
        free(lex.buffer);
        return INF_LAB_LINE_FORECAST;
    }
    if (strcmp(lex.command, "dump") == 0 || strcmp(lex.command, "dump_json") == 0) {
        if (out_json) *out_json = inf_lab_alloc_json_ctx(s, ctx);
        free(lex.buffer);
        return INF_LAB_LINE_DUMP;
    }

    InfLabCommandKind kind = inf_lab_command_kind_for(lex.command);
    InfLabParsedArgs args = inf_lab_parsed_args_default();

    for (char* token = encounter_lab_next_token("inferno lab", &lex.cursor);
            token != NULL;
            token = encounter_lab_next_token("inferno lab", &lex.cursor)) {
        const char* key = NULL;
        const char* value = NULL;
        encounter_lab_parse_key_value("inferno lab", token, &key, &value);
        inf_lab_apply_script_arg(&args, kind, lex.command, key, value);
    }

    InfernoLabCommand cmd = inf_lab_build_script_command(kind, &args, lex.command);
    inf_lab_apply_command_ctx(s, ctx, &cmd);
    free(lex.buffer);
    return INF_LAB_LINE_NONE;
}
