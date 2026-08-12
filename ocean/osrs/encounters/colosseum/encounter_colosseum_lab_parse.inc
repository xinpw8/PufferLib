#line 76 "encounter_colosseum.h"

static const char* col_lab_npc_type_name(int type) {
    switch (type) {
        case COLO_FREMENNIK_BERSERKER: return "berserker";
        case COLO_FREMENNIK_ARCHER:    return "archer";
        case COLO_FREMENNIK_SEER:      return "seer";
        case COLO_SERPENT_SHAMAN:      return "shaman";
        case COLO_JAGUAR_WARRIOR:      return "jaguar";
        case COLO_JAVELIN_COLOSSUS:    return "javelin";
        case COLO_SHOCKWAVE_COLOSSUS:  return "shockwave";
        case COLO_MINOTAUR:            return "minotaur";
        case COLO_MANTICORE:           return "manticore";
        case COLO_SOL_HEREDIT:         return "sol";
        case COLO_HEALING_TOTEM:       return "totem";
        case COLO_BEE_SWARM:           return "bee";
        default:                       return "unknown";
    }
}

static int col_lab_parse_npc_type(const char* value) {
    if (strcmp(value, "berserker") == 0) return COLO_FREMENNIK_BERSERKER;
    if (strcmp(value, "archer") == 0)    return COLO_FREMENNIK_ARCHER;
    if (strcmp(value, "seer") == 0)      return COLO_FREMENNIK_SEER;
    if (strcmp(value, "shaman") == 0 || strcmp(value, "serpent") == 0)
        return COLO_SERPENT_SHAMAN;
    if (strcmp(value, "jaguar") == 0)    return COLO_JAGUAR_WARRIOR;
    if (strcmp(value, "javelin") == 0)   return COLO_JAVELIN_COLOSSUS;
    if (strcmp(value, "shockwave") == 0) return COLO_SHOCKWAVE_COLOSSUS;
    if (strcmp(value, "minotaur") == 0)  return COLO_MINOTAUR;
    if (strcmp(value, "manticore") == 0) return COLO_MANTICORE;
    if (strcmp(value, "sol") == 0 || strcmp(value, "heredit") == 0)
        return COLO_SOL_HEREDIT;
    if (strcmp(value, "totem") == 0)     return COLO_HEALING_TOTEM;
    if (strcmp(value, "bee") == 0 || strcmp(value, "bees") == 0)
        return COLO_BEE_SWARM;

    char* end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno == 0 && end && *end == '\0') {
        col_lab_require_type((int)parsed);
        return (int)parsed;
    }
    col_lab_abort("unknown npc type %s", value);
    return 0;
}

static const char* col_lab_modifier_name(int mod) {
    switch (mod) {
        case COLO_MOD_BEES:        return "bees";
        case COLO_MOD_BLASPHEMY:   return "blasphemy";
        case COLO_MOD_DOOM:        return "doom";
        case COLO_MOD_DYNAMIC_DUO: return "dynamic_duo";
        case COLO_MOD_FRAILTY:     return "frailty";
        case COLO_MOD_MANTIMAYHEM: return "mantimayhem";
        case COLO_MOD_MYOPIA:      return "myopia";
        case COLO_MOD_REENTRY:     return "reentry";
        case COLO_MOD_RED_FLAG:    return "red_flag";
        case COLO_MOD_RELENTLESS:  return "relentless";
        case COLO_MOD_SOLARFLARE:  return "solarflare";
        case COLO_MOD_QUARTET:     return "quartet";
        case COLO_MOD_TOTEMIC:     return "totemic";
        case COLO_MOD_VOLATILITY:  return "volatility";
        default:                   return "unknown";
    }
}

static int col_lab_parse_modifier(const char* value) {
    for (int m = 0; m < COLO_NUM_REAL_MODIFIERS; m++)
        if (strcmp(value, col_lab_modifier_name(m)) == 0) return m;

    char* end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno == 0 && end && *end == '\0' &&
            parsed >= 0 && parsed < COLO_NUM_REAL_MODIFIERS) {
        return (int)parsed;
    }
    col_lab_abort("unknown modifier %s", value);
    return 0;
}

static const char* col_lab_overhead_name(int prayer) {
    switch (prayer) {
        case PRAYER_PROTECT_MELEE:  return "melee";
        case PRAYER_PROTECT_RANGED: return "ranged";
        case PRAYER_PROTECT_MAGIC:  return "magic";
        default:                    return "none";
    }
}

static int col_lab_overhead_for_name(const char* value) {
    if (strcmp(value, "melee") == 0)  return PRAYER_PROTECT_MELEE;
    if (strcmp(value, "ranged") == 0) return PRAYER_PROTECT_RANGED;
    if (strcmp(value, "magic") == 0)  return PRAYER_PROTECT_MAGIC;
    if (strcmp(value, "none") == 0 || strcmp(value, "off") == 0) return PRAYER_NONE;
    col_lab_abort("unknown prayer %s (want melee|ranged|magic|none)", value);
    return PRAYER_NONE;
}

static ColoLabOptionalInt col_lab_parse_optional_hp_value(const char* value) {
    return encounter_lab_parse_optional_full_int("colosseum lab", value);
}

typedef struct {
    int slot;
    int type;
    int x;
    int y;
    ColoLabOptionalInt hp;
    ColoLabOptionalInt timer;
    ColoLabOptionalInt prayer;
    int wave;
    int phase;
    int modifier;
    int tier;
    int draft[COLO_MODIFIER_DRAFT_OPTIONS];
    int ticks;
    uint32_t seed;
} ColoLabParsedArgs;

static ColoLabParsedArgs col_lab_parsed_args_default(void) {
    ColoLabParsedArgs a = {
        .slot = -1,
        .type = -1,
        .x = INT32_MIN,
        .y = INT32_MIN,
        .hp = { .kind = ENCOUNTER_LAB_OPTIONAL_INT_UNSET },
        .timer = { .kind = ENCOUNTER_LAB_OPTIONAL_INT_UNSET },
        .prayer = { .kind = ENCOUNTER_LAB_OPTIONAL_INT_UNSET },
        .wave = -1,
        .phase = -1,
        .modifier = -1,
        .tier = 1,
        .ticks = 1,
        .seed = 0,
    };
    for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++) a.draft[o] = -1;
    return a;
}

static void col_lab_apply_script_arg(
    ColoLabParsedArgs* args,
    ColoLabCommandKind kind,
    const char* command,
    const char* key,
    const char* value
) {
    switch (kind) {
        case COLO_LAB_COMMAND_RESET:
            if (strcmp(key, "seed") == 0) {
                args->seed = encounter_lab_parse_seed_value("colosseum lab", value);
                return;
            }
            break;

        case COLO_LAB_COMMAND_SET_PLAYER:
            if (strcmp(key, "x") == 0) {
                args->x = encounter_lab_parse_int_value("colosseum lab", value);
                return;
            }
            if (strcmp(key, "y") == 0) {
                args->y = encounter_lab_parse_int_value("colosseum lab", value);
                return;
            }
            if (strcmp(key, "hp") == 0) { args->hp = col_lab_parse_optional_hp_value(value); return; }
            if (strcmp(key, "prayer") == 0) {
                args->prayer = encounter_lab_optional_int_set(col_lab_overhead_for_name(value));
                return;
            }
            break;

        case COLO_LAB_COMMAND_SPAWN_NPC:
            if (strcmp(key, "slot") == 0) {
                args->slot = encounter_lab_parse_int_value("colosseum lab", value);
                return;
            }
            if (strcmp(key, "type") == 0) { args->type = col_lab_parse_npc_type(value); return; }
            if (strcmp(key, "x") == 0) {
                args->x = encounter_lab_parse_int_value("colosseum lab", value);
                return;
            }
            if (strcmp(key, "y") == 0) {
                args->y = encounter_lab_parse_int_value("colosseum lab", value);
                return;
            }
            if (strcmp(key, "hp") == 0) { args->hp = col_lab_parse_optional_hp_value(value); return; }
            if (strcmp(key, "timer") == 0) {
                args->timer = encounter_lab_optional_int_set(
                    encounter_lab_parse_int_value("colosseum lab", value));
                return;
            }
            break;

        case COLO_LAB_COMMAND_MOVE_NPC:
            if (strcmp(key, "slot") == 0) {
                args->slot = encounter_lab_parse_int_value("colosseum lab", value);
                return;
            }
            if (strcmp(key, "x") == 0) {
                args->x = encounter_lab_parse_int_value("colosseum lab", value);
                return;
            }
            if (strcmp(key, "y") == 0) {
                args->y = encounter_lab_parse_int_value("colosseum lab", value);
                return;
            }
            break;

        case COLO_LAB_COMMAND_DELETE_NPC:
        case COLO_LAB_COMMAND_KILL_NPC:
            if (strcmp(key, "slot") == 0) {
                args->slot = encounter_lab_parse_int_value("colosseum lab", value);
                return;
            }
            break;

        case COLO_LAB_COMMAND_SET_NPC_HP:
            if (strcmp(key, "slot") == 0) {
                args->slot = encounter_lab_parse_int_value("colosseum lab", value);
                return;
            }
            if (strcmp(key, "hp") == 0) {
                args->hp = encounter_lab_optional_int_set(
                    encounter_lab_parse_int_value("colosseum lab", value));
                return;
            }
            break;

        case COLO_LAB_COMMAND_SET_NPC_TIMER:
            if (strcmp(key, "slot") == 0) {
                args->slot = encounter_lab_parse_int_value("colosseum lab", value);
                return;
            }
            if (strcmp(key, "timer") == 0) {
                args->timer = encounter_lab_optional_int_set(
                    encounter_lab_parse_int_value("colosseum lab", value));
                return;
            }
            break;

        case COLO_LAB_COMMAND_SPAWN_WAVE:
            if (strcmp(key, "wave") == 0) {
                args->wave = encounter_lab_parse_int_value("colosseum lab", value);
                return;
            }
            break;

        case COLO_LAB_COMMAND_SET_BOSS_PHASE:
            if (strcmp(key, "phase") == 0) {
                args->phase = encounter_lab_parse_int_value("colosseum lab", value);
                return;
            }
            break;

        case COLO_LAB_COMMAND_SET_MODIFIER:
            if (strcmp(key, "mod") == 0 || strcmp(key, "modifier") == 0) {
                args->modifier = col_lab_parse_modifier(value);
                return;
            }
            if (strcmp(key, "tier") == 0) {
                args->tier = encounter_lab_parse_int_value("colosseum lab", value);
                return;
            }
            break;

        case COLO_LAB_COMMAND_OPEN_DRAFT:
            if (strcmp(key, "opt0") == 0) { args->draft[0] = col_lab_parse_modifier(value); return; }
            if (strcmp(key, "opt1") == 0) { args->draft[1] = col_lab_parse_modifier(value); return; }
            if (strcmp(key, "opt2") == 0) { args->draft[2] = col_lab_parse_modifier(value); return; }
            break;

        case COLO_LAB_COMMAND_STEP_TICKS:
            if (strcmp(key, "ticks") == 0) {
                args->ticks = encounter_lab_parse_int_value("colosseum lab", value);
                return;
            }
            break;

        case COLO_LAB_COMMAND_START_GRAPPLE:
        case COLO_LAB_COMMAND_CLEAR_NPCS:
        case COLO_LAB_COMMAND_NONE:
            break;
    }

    col_lab_abort("unknown key %s for command %s", key, command);
}

static ColosseumLabCommand col_lab_build_script_command(
    ColoLabCommandKind kind, const ColoLabParsedArgs* args, const char* command
) {
    ColosseumLabCommand out;
    memset(&out, 0, sizeof(out));
    out.kind = kind;

    switch (kind) {
        case COLO_LAB_COMMAND_NONE:
            return out;

        case COLO_LAB_COMMAND_RESET:
            out.as.reset.seed = args->seed;
            return out;

        case COLO_LAB_COMMAND_SET_PLAYER:
            if (args->x == INT32_MIN || args->y == INT32_MIN)
                col_lab_abort("command %s requires x and y", command);
            out.as.player.x = args->x;
            out.as.player.y = args->y;
            out.as.player.hp = args->hp;
            out.as.player.prayer = args->prayer;
            return out;

        case COLO_LAB_COMMAND_SPAWN_NPC:
            if (args->x == INT32_MIN || args->y == INT32_MIN)
                col_lab_abort("command %s requires x and y", command);
            if (args->type < 0)
                col_lab_abort("npc command requires type");
            out.as.spawn_npc.slot = args->slot;
            out.as.spawn_npc.type = (ColoNpcType)args->type;
            out.as.spawn_npc.x = args->x;
            out.as.spawn_npc.y = args->y;
            out.as.spawn_npc.hp = args->hp;
            out.as.spawn_npc.timer = args->timer;
            return out;

        case COLO_LAB_COMMAND_MOVE_NPC:
            if (args->x == INT32_MIN || args->y == INT32_MIN)
                col_lab_abort("command %s requires x and y", command);
            if (args->slot < 0) col_lab_abort("command %s requires slot", command);
            out.as.move_npc.slot = args->slot;
            out.as.move_npc.x = args->x;
            out.as.move_npc.y = args->y;
            return out;

        case COLO_LAB_COMMAND_DELETE_NPC:
        case COLO_LAB_COMMAND_KILL_NPC:
            if (args->slot < 0) col_lab_abort("command %s requires slot", command);
            out.as.npc_slot.slot = args->slot;
            return out;

        case COLO_LAB_COMMAND_SET_NPC_HP:
            if (args->slot < 0) col_lab_abort("command %s requires slot", command);
            if (args->hp.kind != ENCOUNTER_LAB_OPTIONAL_INT_SET)
                col_lab_abort("set_npc_hp requires hp");
            out.as.npc_hp.slot = args->slot;
            out.as.npc_hp.hp = args->hp.value;
            return out;

        case COLO_LAB_COMMAND_SET_NPC_TIMER:
            if (args->slot < 0) col_lab_abort("command %s requires slot", command);
            if (args->timer.kind != ENCOUNTER_LAB_OPTIONAL_INT_SET)
                col_lab_abort("set_npc_timer requires timer");
            out.as.npc_timer.slot = args->slot;
            out.as.npc_timer.timer = args->timer.value;
            return out;

        case COLO_LAB_COMMAND_SPAWN_WAVE:
            if (args->wave < 0) col_lab_abort("wave requires wave=N");
            out.as.wave.wave = args->wave;
            return out;

        case COLO_LAB_COMMAND_SET_BOSS_PHASE:
            if (args->phase < 0) col_lab_abort("boss_phase requires phase=N");
            out.as.boss_phase.phase = args->phase;
            return out;

        case COLO_LAB_COMMAND_START_GRAPPLE:
            return out;

        case COLO_LAB_COMMAND_SET_MODIFIER:
            if (args->modifier < 0) col_lab_abort("modifier requires mod=NAME");
            out.as.modifier.modifier = args->modifier;
            out.as.modifier.tier = args->tier;
            return out;

        case COLO_LAB_COMMAND_OPEN_DRAFT:
            for (int o = 0; o < COLO_MODIFIER_DRAFT_OPTIONS; o++)
                out.as.draft.option[o] = args->draft[o];
            return out;

        case COLO_LAB_COMMAND_CLEAR_NPCS:
            return out;

        case COLO_LAB_COMMAND_STEP_TICKS:
            out.as.step_ticks.ticks = args->ticks;
            return out;
    }

    col_lab_abort("unknown command kind %d", kind);
    out.kind = COLO_LAB_COMMAND_NONE;
    return out;
}

static char* col_lab_alloc_json(ColosseumState* s);

static ColoLabCommandKind col_lab_command_kind_for(const char* command) {
    static const EncounterLabCommandAlias aliases[] = {
        { "reset", COLO_LAB_COMMAND_RESET },
        { "player", COLO_LAB_COMMAND_SET_PLAYER },
        { "set_player", COLO_LAB_COMMAND_SET_PLAYER },
        { "npc", COLO_LAB_COMMAND_SPAWN_NPC },
        { "spawn_npc", COLO_LAB_COMMAND_SPAWN_NPC },
        { "move_npc", COLO_LAB_COMMAND_MOVE_NPC },
        { "delete_npc", COLO_LAB_COMMAND_DELETE_NPC },
        { "kill_npc", COLO_LAB_COMMAND_KILL_NPC },
        { "set_npc_hp", COLO_LAB_COMMAND_SET_NPC_HP },
        { "set_npc_timer", COLO_LAB_COMMAND_SET_NPC_TIMER },
        { "wave", COLO_LAB_COMMAND_SPAWN_WAVE },
        { "spawn_wave", COLO_LAB_COMMAND_SPAWN_WAVE },
        { "boss_phase", COLO_LAB_COMMAND_SET_BOSS_PHASE },
        { "phase", COLO_LAB_COMMAND_SET_BOSS_PHASE },
        { "grapple", COLO_LAB_COMMAND_START_GRAPPLE },
        { "start_grapple", COLO_LAB_COMMAND_START_GRAPPLE },
        { "modifier", COLO_LAB_COMMAND_SET_MODIFIER },
        { "mod", COLO_LAB_COMMAND_SET_MODIFIER },
        { "draft", COLO_LAB_COMMAND_OPEN_DRAFT },
        { "clear_npcs", COLO_LAB_COMMAND_CLEAR_NPCS },
        { "step", COLO_LAB_COMMAND_STEP_TICKS },
        { "step_ticks", COLO_LAB_COMMAND_STEP_TICKS },
    };
    return (ColoLabCommandKind)encounter_lab_lookup_command_kind(
        "colosseum lab", command, aliases, sizeof(aliases) / sizeof(aliases[0]));
}

static ColoLabLineResult col_lab_apply_script_line_alloc_json(
    ColosseumState* s,
    ColosseumContext* ctx,
    const char* line,
    char** out_json
) {
    if (!out_json) col_lab_abort("json output pointer is required");
    *out_json = NULL;
    EncounterLabLine lex = encounter_lab_line_begin("colosseum lab", line);
    if (!lex.command) { free(lex.buffer); return COLO_LAB_LINE_NONE; }

    if (strcmp(lex.command, "dump") == 0 || strcmp(lex.command, "dump_json") == 0) {
        if (out_json) *out_json = col_lab_alloc_json(s);
        free(lex.buffer);
        return COLO_LAB_LINE_DUMP;
    }

    ColoLabCommandKind kind = col_lab_command_kind_for(lex.command);
    ColoLabParsedArgs args = col_lab_parsed_args_default();

    for (char* token = encounter_lab_next_token("colosseum lab", &lex.cursor);
            token != NULL;
            token = encounter_lab_next_token("colosseum lab", &lex.cursor)) {
        const char* key = NULL;
        const char* value = NULL;
        encounter_lab_parse_key_value("colosseum lab", token, &key, &value);
        col_lab_apply_script_arg(&args, kind, lex.command, key, value);
    }

    ColosseumLabCommand cmd = col_lab_build_script_command(kind, &args, lex.command);
    col_lab_apply_command(s, ctx, &cmd);
    free(lex.buffer);
    return COLO_LAB_LINE_NONE;
}
