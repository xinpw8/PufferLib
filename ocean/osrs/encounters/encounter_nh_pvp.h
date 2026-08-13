#ifndef ENCOUNTER_NH_PVP_H
#define ENCOUNTER_NH_PVP_H

#include "../osrs_encounter.h"
#include "../osrs_encounter_visual_events.h"
#include "../osrs_env.h"

#define NH_PVP_TARGET_SLOTS 1
#define NH_PVP_ACTION_MASK_SIZE \
    OSRS_BASE_ACTION_MASK_SIZE(NH_PVP_TARGET_SLOTS)
#define NH_PVP_ACTION_DIMS_INIT OSRS_BASE_ACTION_DIMS_INIT(NH_PVP_TARGET_SLOTS)
static const int NH_PVP_ACTION_DIMS[OSRS_BASE_NUM_ACTION_HEADS] =
    NH_PVP_ACTION_DIMS_INIT;

typedef struct {
    OsrsEnv env;
} NhPvpState;

typedef struct {
    const CollisionMap* collision_map;
    const EncounterArenaTopology* route_topology;
    OsrsActorRouteCache player_route_cache[NUM_AGENTS];
} NhPvpContext;

static int nh_pvp_find_consumable_cell(
    const Player* player,
    OsrsConsumableKind kind
) {
    for (int cell = 0; cell < OSRS_INVENTORY_SIZE; cell++) {
        const OsrsItemContentMetadata* metadata =
            osrs_inventory_cell_metadata(&player->inventory_cells[cell]);
        if (metadata->consumable_kind == kind) return cell;
    }
    return -1;
}

static void nh_pvp_translate_inventory_cell(
    int* actions,
    const Player* player,
    int cell
) {
    if (cell < 0 || cell >= OSRS_INVENTORY_SIZE) return;
    const OsrsItemContentMetadata* metadata =
        osrs_inventory_cell_metadata(&player->inventory_cells[cell]);
    int click_action = cell + 1;
    if (metadata->click_action == OSRS_CLICK_EQUIP) {
        if (metadata->gear_slot >= 0 && metadata->gear_slot < NUM_GEAR_SLOTS)
            actions[OSRS_HEAD_EQUIP_SLOT(metadata->gear_slot)] = click_action;
    } else if (metadata->click_action == OSRS_CLICK_EAT) {
        actions[OSRS_HEAD_EAT] = click_action;
    } else if (metadata->click_action == OSRS_CLICK_DRINK) {
        actions[OSRS_HEAD_DRINK] = click_action;
    }
}

static void nh_pvp_translate_human_input(
    HumanInput* hi,
    int* actions,
    Player* agent,
    Player* target
) {
    memset(actions, 0, OSRS_BASE_NUM_ACTION_HEADS * sizeof(int));
    if (hi->pending_move_x >= 0 && hi->pending_move_y >= 0) {
        int dx = hi->pending_move_x - agent->x;
        int dy = hi->pending_move_y - agent->y;
        if (dx < -2) dx = -2;
        if (dx > 2) dx = 2;
        if (dy < -2) dy = -2;
        if (dy > 2) dy = 2;
        for (int action = 1; action < OSRS_PRIMARY_MOVE_ACTIONS; action++) {
            if (ENCOUNTER_MOVE_TARGET_DX[action] == dx &&
                    ENCOUNTER_MOVE_TARGET_DY[action] == dy) {
                actions[OSRS_HEAD_PRIMARY] = action;
                break;
            }
        }
    }
    encounter_translate_prayer(hi, actions, OSRS_HEAD_OVERHEAD);
    encounter_translate_offensive_prayer(hi, actions, OSRS_HEAD_OFFENSIVE);

    if (hi->pending_attack) {
        actions[OSRS_HEAD_PRIMARY] = OSRS_PRIMARY_MOVE_ACTIONS;
        if (hi->pending_spell == PVP_ATTACK_ICE)
            actions[OSRS_HEAD_SPELL] = OSRS_SPELL_ICE_BARRAGE;
        else if (hi->pending_spell == PVP_ATTACK_BLOOD)
            actions[OSRS_HEAD_SPELL] = OSRS_SPELL_BLOOD_BARRAGE;
    }
    if (hi->pending_food) {
        int cell = nh_pvp_find_consumable_cell(
            agent, OSRS_CONSUMABLE_SHARK_FOOD);
        if (cell >= 0) actions[OSRS_HEAD_EAT] = cell + 1;
    }
    if (hi->pending_karambwan) {
        int cell = nh_pvp_find_consumable_cell(
            agent, OSRS_CONSUMABLE_KARAMBWAN);
        if (cell >= 0) actions[OSRS_HEAD_EAT] = cell + 1;
    }
    OsrsConsumableKind potion_kind = OSRS_CONSUMABLE_NONE;
    if (hi->pending_potion == POTION_BREW)
        potion_kind = OSRS_CONSUMABLE_BREW;
    else if (hi->pending_potion == POTION_RESTORE)
        potion_kind = OSRS_CONSUMABLE_SUPER_RESTORE;
    else if (hi->pending_potion == POTION_COMBAT)
        potion_kind = OSRS_CONSUMABLE_SUPER_COMBAT;
    else if (hi->pending_potion == POTION_RANGED)
        potion_kind = OSRS_CONSUMABLE_RANGING;
    if (potion_kind != OSRS_CONSUMABLE_NONE) {
        int cell = nh_pvp_find_consumable_cell(agent, potion_kind);
        if (cell >= 0) actions[OSRS_HEAD_DRINK] = cell + 1;
    }
    for (int i = 0; i < hi->commands.count; i++) {
        const HumanCommand* command = &hi->commands.items[i];
        if (command->kind == HUMAN_COMMAND_EQUIP_INVENTORY_ITEM ||
                command->kind == HUMAN_COMMAND_INVENTORY_PRIMARY_CLICK ||
                command->kind == HUMAN_COMMAND_EAT ||
                command->kind == HUMAN_COMMAND_DRINK) {
            nh_pvp_translate_inventory_cell(
                actions, agent, command->inventory_slot);
        }
    }
    if (hi->pending_veng)
        actions[OSRS_HEAD_SPELL] = OSRS_SPELL_VENGEANCE;
    if (hi->pending_spec)
        actions[OSRS_HEAD_SPECIAL] = agent->spec_armed ? 2 : 1;
    (void)target;
}

static void nh_pvp_init_state(
    EncounterState* state,
    EncounterContext* context
) {
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    memset(s, 0, sizeof(*s));
    pvp_init(&s->env);
    s->env.ocean_io.agent_actions = s->env._acts_buf;
    s->env.ocean_io.agent_rewards = s->env._rews_buf;
    s->env.ocean_io.agent_terminals = s->env._terms_buf;
}

static EncounterState* nh_pvp_create(void) {
    NhPvpState* s = (NhPvpState*)malloc(sizeof(NhPvpState));
    if (!s) abort();
    nh_pvp_init_state((EncounterState*)s, NULL);
    return (EncounterState*)s;
}

static void nh_pvp_destroy(EncounterState* state) {
    NhPvpState* s = (NhPvpState*)state;
    pvp_close(&s->env);
    free(s);
}

static void nh_pvp_init_context(EncounterContext* context) {
    memset(context, 0, sizeof(NhPvpContext));
}

static void nh_pvp_destroy_context(EncounterContext* context) {
    (void)context;
}
static void nh_pvp_finalize_context(
    EncounterState* state,
    EncounterContext* context
) {
    (void)state;
    NhPvpContext* ctx = (NhPvpContext*)context;
    if (!ctx || ctx->route_topology) {
        fprintf(stderr, "NH PvP route topology finalized twice\n");
        abort();
    }
    ctx->route_topology =
        pvp_route_topology_finalize(ctx->collision_map);
}


static void nh_pvp_reset(EncounterState* state, EncounterContext* context, uint32_t seed) {
    NhPvpState* s = (NhPvpState*)state;
    NhPvpContext* ctx = (NhPvpContext*)context;
    encounter_arena_topology_require_finalized(ctx->route_topology);
    if (seed != 0) {
        s->env.has_rng_seed = 1;
        s->env.rng_seed = seed;
    }
    pvp_actor_route_caches_clear(ctx->player_route_cache);
    pvp_reset(&s->env, ctx->route_topology);
}

static void nh_pvp_step(EncounterState* state, EncounterContext* context, const int* actions) {
    NhPvpContext* ctx = (NhPvpContext*)context;
    NhPvpState* s = (NhPvpState*)state;
    encounter_arena_topology_require_finalized(ctx->route_topology);
    memcpy(s->env.ocean_io.agent_actions, actions,
        OSRS_BASE_NUM_ACTION_HEADS * sizeof(int));
    pvp_step(&s->env, ctx->route_topology, ctx->player_route_cache);
}

static void nh_pvp_step_human_commands(
    EncounterState* state,
    EncounterContext* context,
    HumanInput* hi
) {
    NhPvpContext* ctx = (NhPvpContext*)context;
    encounter_arena_topology_require_finalized(ctx->route_topology);
    NhPvpState* s = (NhPvpState*)state;
    int saved_use_c_opponent_p0 = s->env.pvp_runtime.use_c_opponent_p0;
    s->env.pvp_runtime.use_c_opponent_p0 = 0;
    if (hi->pending_move_x >= 0 && hi->pending_move_y >= 0) {
        s->env.pvp_runtime.walk_dest_x[0] = hi->pending_move_x;
        s->env.pvp_runtime.walk_dest_y[0] = hi->pending_move_y;
    }
    nh_pvp_translate_human_input(
        hi,
        s->env.ocean_io.agent_actions,
        &s->env.players[0],
        &s->env.players[1]);
    pvp_step(&s->env, ctx->route_topology, ctx->player_route_cache);
    s->env.pvp_runtime.use_c_opponent_p0 = saved_use_c_opponent_p0;
    /* pending_move must survive until arrival so later ticks keep re-arming
       walk_dest for the same click */
    if (s->env.pvp_runtime.walk_dest_x[0] < 0 || s->env.pvp_runtime.walk_dest_y[0] < 0) {
        human_input_clear_move(hi);
    }
    hi->pending_attack = 0;
    hi->pending_spell = 0;
    hi->pending_prayer = 0;
    hi->pending_offensive_prayer = 0;
    hi->pending_food = 0;
    hi->pending_potion = 0;
    hi->pending_karambwan = 0;
    hi->pending_veng = 0;
    hi->pending_spec = 0;
}

static void nh_pvp_write_obs(
    EncounterState* state,
    EncounterContext* context,
    float* obs_out
) {
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    pvp_write_observations(obs_out, &s->env, 0);
}

static void nh_pvp_write_mask(
    EncounterState* state,
    EncounterContext* context,
    float* mask_out
) {
    NhPvpState* s = (NhPvpState*)state;
    NhPvpContext* ctx = (NhPvpContext*)context;
    pvp_write_action_mask(mask_out, &s->env, 0, ctx->route_topology);
}

static float nh_pvp_get_reward(EncounterState* state, EncounterContext* context) {
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    return s->env._rews_buf[0];
}

static int nh_pvp_is_terminal(EncounterState* state, EncounterContext* context) {
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    return s->env.episode_over;
}

static int nh_pvp_get_entity_count(EncounterState* state, EncounterContext* context) {
    (void)state;
    (void)context;
    return NUM_AGENTS;
}

static void* nh_pvp_get_entity(EncounterState* state, EncounterContext* context, int index) {
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    return &s->env.players[index];
}

static void nh_pvp_fill_render_entities(
    EncounterState* state,
    EncounterContext* context,
    RenderEntity* out,
    int max_entities,
    int* count
) {
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    int n = NUM_AGENTS < max_entities ? NUM_AGENTS : max_entities;
    for (int i = 0; i < n; i++) {
        osrs_render_entity_from_player_entity(&s->env.players[i], &out[i]);
        out[i].attack_target_entity_idx = (n >= 2) ? (1 - i) : -1;
    }
    *count = n;
}

static void nh_pvp_put_int(
    EncounterState* state,
    EncounterContext* context,
    const char* key,
    int value
) {
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    if (strcmp(key, "opponent_type") == 0) {
        s->env.pvp_runtime.opponent.type = (OpponentType)value;
    } else if (strcmp(key, "opponent_p0_type") == 0) {
        s->env.pvp_runtime.opponent_p0.type = (OpponentType)value;
    } else if (strcmp(key, "is_lms") == 0) {
        s->env.is_lms = value;
    } else if (strcmp(key, "use_c_opponent") == 0) {
        s->env.pvp_runtime.use_c_opponent = value;
    } else if (strcmp(key, "use_c_opponent_p0") == 0) {
        s->env.pvp_runtime.use_c_opponent_p0 = value;
    } else if (strcmp(key, "auto_reset") == 0) {
        s->env.auto_reset = value;
    } else if (strcmp(key, "gear_tier") == 0) {
        if (value < 0) {
            s->env.pvp_runtime.gear_tier_weights[0] = 0.60f;
            s->env.pvp_runtime.gear_tier_weights[1] = 0.25f;
            s->env.pvp_runtime.gear_tier_weights[2] = 0.10f;
            s->env.pvp_runtime.gear_tier_weights[3] = 0.05f;
        } else {
            if (value > 3) {
                fprintf(stderr, "nh_pvp: invalid gear_tier %d\n", value);
                abort();
            }
            for (int t = 0; t < 4; t++)
                s->env.pvp_runtime.gear_tier_weights[t] = 0.0f;
            s->env.pvp_runtime.gear_tier_weights[value] = 1.0f;
        }
    } else if (strcmp(key, "seed") == 0) {
        s->env.has_rng_seed = 1;
        s->env.rng_seed = (uint32_t)value;
    }
}

static void nh_pvp_put_float(
    EncounterState* state,
    EncounterContext* context,
    const char* key,
    float value
) {
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    if (strcmp(key, "shaping_scale") == 0) {
        s->env.shaping.shaping_scale = value;
    }
}

static void nh_pvp_put_ptr(
    EncounterState* state,
    EncounterContext* context,
    const char* key,
    void* value
) {
    (void)state;
    NhPvpContext* ctx = (NhPvpContext*)context;
    if (strcmp(key, "collision_map") == 0)
        ctx->collision_map = (const CollisionMap*)value;
    else
        encounter_abort_unknown_config("nh_pvp", "ptr", key);
}

static void* nh_pvp_get_log(EncounterState* state, EncounterContext* context) {
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    return &s->env.log;
}

static int nh_pvp_get_tick(EncounterState* state, EncounterContext* context) {
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    return s->env.tick;
}

static int nh_pvp_get_winner(EncounterState* state, EncounterContext* context) {
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    return s->env.winner;
}

static const EncounterDef ENCOUNTER_NH_PVP = {
    .name = "nh_pvp",
    .obs_size = NH_PVP_NUM_OBS,
    .num_action_heads = OSRS_BASE_NUM_ACTION_HEADS,
    .action_head_dims = NH_PVP_ACTION_DIMS,
    .mask_size = NH_PVP_ACTION_MASK_SIZE,
    .state_size = sizeof(NhPvpState),
    .context_size = sizeof(NhPvpContext),
    .init_context = nh_pvp_init_context,
    .destroy_context = nh_pvp_destroy_context,
    .init_state = nh_pvp_init_state,
    .finalize_context = nh_pvp_finalize_context,

    .create = nh_pvp_create,
    .destroy = nh_pvp_destroy,
    .reset = nh_pvp_reset,
    .step = nh_pvp_step,
    .step_human_commands = nh_pvp_step_human_commands,

    .write_obs = nh_pvp_write_obs,
    .write_mask = nh_pvp_write_mask,
    .get_reward = nh_pvp_get_reward,
    .is_terminal = nh_pvp_is_terminal,

    .get_entity_count = nh_pvp_get_entity_count,
    .get_entity = nh_pvp_get_entity,
    .fill_render_entities = nh_pvp_fill_render_entities,

    .put_int = nh_pvp_put_int,
    .put_float = nh_pvp_put_float,
    .put_ptr = nh_pvp_put_ptr,

    .translate_human_input = NULL,
    .head_move = OSRS_HEAD_PRIMARY,
    .head_prayer = OSRS_HEAD_OVERHEAD,
    .head_target = OSRS_HEAD_PRIMARY,

    .render_post_tick = NULL,
    .get_log = nh_pvp_get_log,
    .get_tick = nh_pvp_get_tick,
    .get_winner = nh_pvp_get_winner,
};

__attribute__((constructor))
static void nh_pvp_register(void) {
    encounter_register(&ENCOUNTER_NH_PVP);
}

#endif /* ENCOUNTER_NH_PVP_H */
