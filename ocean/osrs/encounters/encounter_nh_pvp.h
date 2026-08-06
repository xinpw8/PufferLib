#ifndef ENCOUNTER_NH_PVP_H
#define ENCOUNTER_NH_PVP_H

#include "../osrs_encounter.h"
#include "../osrs_encounter_visual_events.h"
#include "../osrs_env.h"

/* order must match the HEAD_* indices in osrs_types.h */
static const int NH_PVP_ACTION_DIMS[] = {
    LOADOUT_DIM, COMBAT_DIM, OVERHEAD_DIM,
    FOOD_DIM, POTION_DIM, KARAMBWAN_DIM, VENG_DIM, OFFENSIVE_DIM, MOVE_DIM
};

typedef struct {
    OsrsEnv env;
} NhPvpState;

typedef struct {
    int unused;
} NhPvpContext;

static void nh_pvp_translate_human_input(HumanInput* hi, int* actions, Player* agent, Player* target) {
    for (int h = 0; h < NUM_ACTION_HEADS; h++) actions[h] = 0;
    actions[HEAD_LOADOUT] = LOADOUT_KEEP;

    if (hi->pending_attack) {
        if (hi->pending_spell == ATTACK_ICE) actions[HEAD_COMBAT] = ATTACK_ICE;
        else if (hi->pending_spell == ATTACK_BLOOD) actions[HEAD_COMBAT] = ATTACK_BLOOD;
        else actions[HEAD_COMBAT] = ATTACK_ATK;
    }
    encounter_translate_prayer(hi, actions, HEAD_OVERHEAD);
    encounter_translate_offensive_prayer(hi, actions, HEAD_OFFENSIVE);

    if (hi->pending_food) actions[HEAD_FOOD] = FOOD_EAT;
    if (hi->pending_potion > 0) actions[HEAD_POTION] = hi->pending_potion;
    if (hi->pending_karambwan) actions[HEAD_KARAMBWAN] = KARAM_EAT;
    if (hi->pending_veng) actions[HEAD_VENG] = VENG_CAST;
    if (hi->pending_spec) {
        AttackStyle style = (AttackStyle)get_item_attack_style(agent->equipped[GEAR_SLOT_WEAPON]);
        if (style == ATTACK_STYLE_MELEE) actions[HEAD_LOADOUT] = LOADOUT_SPEC_MELEE;
        else if (style == ATTACK_STYLE_RANGED) actions[HEAD_LOADOUT] = LOADOUT_SPEC_RANGE;
        else if (style == ATTACK_STYLE_MAGIC) actions[HEAD_LOADOUT] = LOADOUT_SPEC_MAGIC;
    }
    (void)target;
}

static EncounterState* nh_pvp_create(void) {
    NhPvpState* s = (NhPvpState*)calloc(1, sizeof(NhPvpState));
    pvp_init(&s->env);
    s->env.ocean_io.agent_obs = s->env._obs_buf;
    s->env.ocean_io.agent_actions = s->env._acts_buf;
    s->env.ocean_io.agent_rewards = s->env._rews_buf;
    s->env.ocean_io.agent_terminals = s->env._terms_buf;
    return (EncounterState*)s;
}

static void nh_pvp_destroy(EncounterState* state) {
    NhPvpState* s = (NhPvpState*)state;
    pvp_close(&s->env);
    free(s);
}

static void nh_pvp_init_context(EncounterContext* context) {
    (void)context;
}

static void nh_pvp_destroy_context(EncounterContext* context) {
    (void)context;
}

static void nh_pvp_reset(EncounterState* state, EncounterContext* context, uint32_t seed) {
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    if (seed != 0) {
        s->env.has_rng_seed = 1;
        s->env.rng_seed = seed;
    }
    pvp_reset(&s->env);
}

static void nh_pvp_step(EncounterState* state, EncounterContext* context, const int* actions) {
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    memcpy(s->env.ocean_io.agent_actions, actions, NUM_ACTION_HEADS * sizeof(int));
    pvp_step(&s->env);
}

static void nh_pvp_step_human_commands(
    EncounterState* state,
    EncounterContext* context,
    HumanInput* hi
) {
    (void)context;
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
    pvp_step(&s->env);
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

static void nh_pvp_write_obs(EncounterState* state, EncounterContext* context, float* obs_out) {
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    memcpy(obs_out, s->env._obs_buf, SLOT_NUM_OBSERVATIONS * sizeof(float));
}

static void nh_pvp_write_mask(
    EncounterState* state,
    EncounterContext* context,
    float* mask_out
) {
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    for (int i = 0; i < ACTION_MASK_SIZE; i++) {
        mask_out[i] = (float)s->env._masks_buf[i];
    }
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
    (void)context;
    NhPvpState* s = (NhPvpState*)state;
    if (strcmp(key, "collision_map") == 0) {
        s->env.collision_map = value;
    }
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
    .obs_size = SLOT_NUM_OBSERVATIONS,
    .num_action_heads = NUM_ACTION_HEADS,
    .action_head_dims = NH_PVP_ACTION_DIMS,
    .mask_size = ACTION_MASK_SIZE,
    .state_size = sizeof(NhPvpState),
    .context_size = sizeof(NhPvpContext),
    .init_context = nh_pvp_init_context,
    .destroy_context = nh_pvp_destroy_context,

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
    .head_move = -1,
    .head_prayer = -1,
    .head_target = -1,

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
