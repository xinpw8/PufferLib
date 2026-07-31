#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ocean/osrs/encounters/encounter_colosseum.h"

#define PROBE_NPC_TELLS_OFFSET 37
#define PROBE_SKYFALL_DAMAGE 38
#define PROBE_PLAYER_X 17
#define PROBE_PLAYER_Y 16
#define PROBE_JAVELIN_X 20
#define PROBE_JAVELIN_Y 16

typedef enum {
    PROBE_TARGET_AFTER_MOVE_NONE = 0,
    PROBE_TARGET_AFTER_MOVE_SAME_TICK,
    PROBE_TARGET_AFTER_MOVE_NEXT_TICK,
} ProbeTargetAfterMoveMode;

typedef struct {
    int visible_tick;
    int visible_timer;
    int marked_x;
    int marked_y;
    int obs_slot;
    float obs_pending;
    float obs_timer;
    float obs_dx;
    float obs_dy;
} ProbeSkyfallObs;

typedef struct {
    int damage_taken;
    int first_visible_tick;
    int first_visible_timer;
    int move_action;
    int move_tick;
    int landing_tick;
    int player_x_after_move;
    int player_y_after_move;
    int interaction_after_move;
    int player_x_before_landing;
    int player_y_before_landing;
} ProbeStepDodgeResult;

static void probe_fail(const char* label) {
    fprintf(stderr, "FAIL %s\n", label);
    abort();
}

static void probe_check(const char* label, int ok) {
    if (!ok) probe_fail(label);
    printf("PASS %s\n", label);
}

static void probe_check_float(const char* label, float got, float expected) {
    float delta = fabsf(got - expected);
    if (delta > 0.000001f) {
        fprintf(stderr, "FAIL %s got=%.9f expected=%.9f delta=%.9f\n",
            label, got, expected, delta);
        abort();
    }
    printf("PASS %s got=%.6f expected=%.6f\n", label, got, expected);
}

static void probe_clear_npcs(ColosseumState* s) {
    memset(s->npcs, 0, sizeof(s->npcs));
    memset(s->npc_collision_flags, 0, sizeof(s->npc_collision_flags));
    memset(s->totems, 0, sizeof(s->totems));
    memset(s->bees, 0, sizeof(s->bees));
    col_rebuild_player_collision_flags(s);
}

static void probe_init_context(ColosseumContext* ctx) {
    col_init_context_typed(ctx);
    ctx->config.loadout_profile_mode = COLO_LOADOUT_PROFILE_MODE_SPEEDRUN_ONLY;
    ctx->config.beginner_loadout_fraction = 0.0f;
    ctx->config.step_out_forecast_obs_enabled = 1;
    ctx->config.action_debug_log = 0;
}

static void probe_init_empty_state(
    ColosseumState* s,
    ColosseumContext* ctx,
    uint32_t seed
) {
    probe_init_context(ctx);
    memset(s, 0, sizeof(*s));
    col_reset_ctx((EncounterState*)s, (EncounterContext*)ctx, seed);
    probe_clear_npcs(s);
    s->wave_spawn_delay = 0;
    s->wave_ready_delay = 0;
    s->modifiers.draft_pending = 0;
    s->warband_cycle_anchor = s->tick;
    s->player.x = PROBE_PLAYER_X;
    s->player.y = PROBE_PLAYER_Y;
    s->player.current_hitpoints = 99;
    s->player.attack_timer = 99;
    s->player_dest_x = -1;
    s->player_dest_y = -1;
    col_apply_weapon_set(s, COLO_GEAR_RANGED);
    encounter_pending_hit_queue_clear(&s->player_pending_hits);
    col_rebuild_player_collision_flags(s);
}

static ColoJavelinState* probe_spawn_javelin(ColosseumState* s) {
    col_init_npc(s, 0, COLO_JAVELIN_COLOSSUS, PROBE_JAVELIN_X, PROBE_JAVELIN_Y);
    ColoNPC* npc = &s->npcs[0];
    npc->stun_timer = 0;
    npc->frozen_ticks = 0;
    return colo_npc_javelin(npc);
}

static int probe_obs_slot_for_npc(const ColosseumState* s, int npc_idx) {
    for (int slot = 0; slot < COLO_OBS_NPCS; slot++) {
        if (s->current_obs_slots[slot] == npc_idx) return slot;
    }
    probe_fail("javelin obs slot found");
    return -1;
}

static float* probe_npc_tells(float* obs, int obs_slot) {
    int base = COLO_OBS_AFTER_EQUIPPED_SELF +
        obs_slot * COLO_FEATURES_PER_NPC +
        PROBE_NPC_TELLS_OFFSET;
    return &obs[base];
}

static ProbeSkyfallObs probe_read_skyfall_obs(
    ColosseumState* s,
    ColosseumContext* ctx,
    float* obs
) {
    col_write_obs_ctx((EncounterState*)s, (EncounterContext*)ctx, obs);
    int obs_slot = probe_obs_slot_for_npc(s, 0);
    float* tells = probe_npc_tells(obs, obs_slot);
    ColoJavelinState* jv = colo_npc_javelin(&s->npcs[0]);
    ProbeSkyfallObs out = {
        .visible_tick = s->tick,
        .visible_timer = jv->skyfall_timer,
        .marked_x = jv->skyfall_tile_x,
        .marked_y = jv->skyfall_tile_y,
        .obs_slot = obs_slot,
        .obs_pending = tells[0],
        .obs_timer = tells[1],
        .obs_dx = tells[2],
        .obs_dy = tells[3],
    };
    return out;
}

static int probe_move_action_ending_off_tile(const ColosseumState* s, int x, int y) {
    for (int action = 1; action < ENCOUNTER_MOVE_ACTIONS; action++) {
        ColosseumState tmp = *s;
        int moved = encounter_move_to_target(
            &tmp.player,
            ENCOUNTER_MOVE_TARGET_DX[action],
            ENCOUNTER_MOVE_TARGET_DY[action],
            col_player_walkable,
            &tmp);
        if (moved > 0 && (tmp.player.x != x || tmp.player.y != y)) return action;
    }
    probe_fail("move action ending off marked tile found");
    return 0;
}

static int probe_primary_attack_action_for_slot(int obs_slot) {
    if (obs_slot < 0 || obs_slot >= COLO_OBS_NPCS) probe_fail("valid obs target slot");
    return col_primary_attack_action_for_obs_slot(obs_slot);
}

static void probe_zero_actions(int* actions) {
    for (int head = 0; head < COLO_NUM_ACTION_HEADS; head++) actions[head] = 0;
}

static void probe_step(ColosseumState* s, ColosseumContext* ctx, int* actions) {
    col_step_ctx((EncounterState*)s, (EncounterContext*)ctx, actions);
}

static void probe_establish_attack_lock(ColosseumState* s, ColosseumContext* ctx) {
    int actions[COLO_NUM_ACTION_HEADS];
    float obs[COLO_NUM_OBS];
    probe_zero_actions(actions);
    col_write_obs_ctx((EncounterState*)s, (EncounterContext*)ctx, obs);
    actions[COLO_HEAD_PRIMARY] =
        probe_primary_attack_action_for_slot(probe_obs_slot_for_npc(s, 0));
    probe_step(s, ctx, actions);
    probe_check("attack lock active before skyfall fire",
        osrs_interaction_active(&s->interaction) &&
        s->interaction.target_slot == 0);
}

static ProbeSkyfallObs probe_fire_skyfall_visible(
    ColosseumState* s,
    ColosseumContext* ctx,
    int attack_timer_before_fire
) {
    int actions[COLO_NUM_ACTION_HEADS];
    static float obs[COLO_NUM_OBS];
    ColoJavelinState* jv = colo_npc_javelin(&s->npcs[0]);
    jv->attack_count = 4;
    s->npcs[0].attack_timer = attack_timer_before_fire;
    probe_zero_actions(actions);
    probe_step(s, ctx, actions);
    probe_check("real step fired skyfall",
        jv->skyfall_pending == 1 &&
        jv->skyfall_timer == COLO_JAVELIN_SKYFALL_DELAY &&
        jv->skyfall_tile_x == s->player.x &&
        jv->skyfall_tile_y == s->player.y);
    int rolled_damage = jv->skyfall_damage;
    jv->skyfall_damage = PROBE_SKYFALL_DAMAGE;
    ProbeSkyfallObs seen = probe_read_skyfall_obs(s, ctx, obs);
    printf(
        "FULL_STEP fire tick=%d marked=(%d,%d) timer=%d rolled_damage=%d forced_damage=%d obs_slot=%d obs=(pending %.1f timer %.3f dx %.3f dy %.3f)\n",
        seen.visible_tick,
        seen.marked_x,
        seen.marked_y,
        seen.visible_timer,
        rolled_damage,
        jv->skyfall_damage,
        seen.obs_slot,
        seen.obs_pending,
        seen.obs_timer,
        seen.obs_dx,
        seen.obs_dy);
    return seen;
}

static ProbeStepDodgeResult probe_run_no_lock_wait_case(int wait_visible_ticks) {
    ColosseumContext ctx;
    ColosseumState s;
    int actions[COLO_NUM_ACTION_HEADS];
    probe_init_empty_state(&s, &ctx, 0x5100u + (uint32_t)wait_visible_ticks);
    probe_spawn_javelin(&s);
    ProbeSkyfallObs first = probe_fire_skyfall_visible(&s, &ctx, 0);
    int move_action = probe_move_action_ending_off_tile(&s, first.marked_x, first.marked_y);
    int hp_before = s.player.current_hitpoints;
    int move_tick = -1;
    int player_x_after_move = s.player.x;
    int player_y_after_move = s.player.y;
    int interaction_after_move = osrs_interaction_active(&s.interaction)
        ? s.interaction.target_slot : -1;

    for (int i = 0; i < wait_visible_ticks; i++) {
        probe_zero_actions(actions);
        probe_step(&s, &ctx, actions);
    }

    if (colo_npc_javelin(&s.npcs[0])->skyfall_pending) {
        probe_zero_actions(actions);
        actions[COLO_HEAD_PRIMARY] = move_action;
        move_tick = s.tick + 1;
        probe_step(&s, &ctx, actions);
        player_x_after_move = s.player.x;
        player_y_after_move = s.player.y;
        interaction_after_move = osrs_interaction_active(&s.interaction)
            ? s.interaction.target_slot : -1;
    }

    int player_x_before_landing = s.player.x;
    int player_y_before_landing = s.player.y;
    while (colo_npc_javelin(&s.npcs[0])->skyfall_pending) {
        player_x_before_landing = s.player.x;
        player_y_before_landing = s.player.y;
        probe_zero_actions(actions);
        probe_step(&s, &ctx, actions);
    }

    ProbeStepDodgeResult result = {
        .damage_taken = hp_before - s.player.current_hitpoints,
        .first_visible_tick = first.visible_tick,
        .first_visible_timer = first.visible_timer,
        .move_action = move_action,
        .move_tick = move_tick,
        .landing_tick = s.tick,
        .player_x_after_move = player_x_after_move,
        .player_y_after_move = player_y_after_move,
        .interaction_after_move = interaction_after_move,
        .player_x_before_landing = player_x_before_landing,
        .player_y_before_landing = player_y_before_landing,
    };
    return result;
}

static ProbeStepDodgeResult probe_run_attack_lock_case(
    const char* label,
    ProbeTargetAfterMoveMode target_mode
) {
    ColosseumContext ctx;
    ColosseumState s;
    int actions[COLO_NUM_ACTION_HEADS];
    static float obs[COLO_NUM_OBS];
    probe_init_empty_state(&s, &ctx, 0x6200u + (uint32_t)target_mode);
    probe_spawn_javelin(&s);
    ColoJavelinState* jv = colo_npc_javelin(&s.npcs[0]);
    jv->attack_count = 4;
    s.npcs[0].attack_timer = 2;
    int player_range = col_player_attack_range(&s);
    int npc_dist = col_npc_dist_to_player(&s, &s.npcs[0]);
    printf(
        "LOCK_SETUP %s player_range=%d npc_dist=%d player=(%d,%d) javelin_sw=(%d,%d)\n",
        label,
        player_range,
        npc_dist,
        s.player.x,
        s.player.y,
        s.npcs[0].x,
        s.npcs[0].y);
    probe_establish_attack_lock(&s, &ctx);
    ProbeSkyfallObs first = probe_fire_skyfall_visible(&s, &ctx, 1);
    int move_action = probe_move_action_ending_off_tile(&s, first.marked_x, first.marked_y);
    int hp_before = s.player.current_hitpoints;

    probe_zero_actions(actions);
    actions[COLO_HEAD_PRIMARY] = move_action;
    if (target_mode == PROBE_TARGET_AFTER_MOVE_SAME_TICK) {
        col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
        actions[COLO_HEAD_PRIMARY] =
            probe_primary_attack_action_for_slot(probe_obs_slot_for_npc(&s, 0));
    }
    int move_tick = s.tick + 1;
    probe_step(&s, &ctx, actions);
    int player_x_after_move = s.player.x;
    int player_y_after_move = s.player.y;
    int interaction_after_move = osrs_interaction_active(&s.interaction)
        ? s.interaction.target_slot : -1;

    if (target_mode == PROBE_TARGET_AFTER_MOVE_NEXT_TICK &&
            colo_npc_javelin(&s.npcs[0])->skyfall_pending) {
        probe_zero_actions(actions);
        col_write_obs_ctx((EncounterState*)&s, (EncounterContext*)&ctx, obs);
        actions[COLO_HEAD_PRIMARY] =
            probe_primary_attack_action_for_slot(probe_obs_slot_for_npc(&s, 0));
        probe_step(&s, &ctx, actions);
    }

    int player_x_before_landing = s.player.x;
    int player_y_before_landing = s.player.y;
    while (colo_npc_javelin(&s.npcs[0])->skyfall_pending) {
        player_x_before_landing = s.player.x;
        player_y_before_landing = s.player.y;
        probe_zero_actions(actions);
        probe_step(&s, &ctx, actions);
    }

    ProbeStepDodgeResult result = {
        .damage_taken = hp_before - s.player.current_hitpoints,
        .first_visible_tick = first.visible_tick,
        .first_visible_timer = first.visible_timer,
        .move_action = move_action,
        .move_tick = move_tick,
        .landing_tick = s.tick,
        .player_x_after_move = player_x_after_move,
        .player_y_after_move = player_y_after_move,
        .interaction_after_move = interaction_after_move,
        .player_x_before_landing = player_x_before_landing,
        .player_y_before_landing = player_y_before_landing,
    };
    return result;
}

static ProbeStepDodgeResult probe_run_attack_lock_idle_case(const char* label) {
    ColosseumContext ctx;
    ColosseumState s;
    int actions[COLO_NUM_ACTION_HEADS];
    probe_init_empty_state(&s, &ctx, 0x6300u);
    probe_spawn_javelin(&s);
    ColoJavelinState* jv = colo_npc_javelin(&s.npcs[0]);
    jv->attack_count = 4;
    s.npcs[0].attack_timer = 2;
    printf("LOCK_SETUP %s player_range=%d npc_dist=%d\n",
        label,
        col_player_attack_range(&s),
        col_npc_dist_to_player(&s, &s.npcs[0]));
    probe_establish_attack_lock(&s, &ctx);
    ProbeSkyfallObs first = probe_fire_skyfall_visible(&s, &ctx, 1);
    int hp_before = s.player.current_hitpoints;
    int player_x_before_landing = s.player.x;
    int player_y_before_landing = s.player.y;

    while (colo_npc_javelin(&s.npcs[0])->skyfall_pending) {
        player_x_before_landing = s.player.x;
        player_y_before_landing = s.player.y;
        probe_zero_actions(actions);
        probe_step(&s, &ctx, actions);
    }

    ProbeStepDodgeResult result = {
        .damage_taken = hp_before - s.player.current_hitpoints,
        .first_visible_tick = first.visible_tick,
        .first_visible_timer = first.visible_timer,
        .move_action = 0,
        .move_tick = -1,
        .landing_tick = s.tick,
        .player_x_after_move = s.player.x,
        .player_y_after_move = s.player.y,
        .interaction_after_move = osrs_interaction_active(&s.interaction)
            ? s.interaction.target_slot : -1,
        .player_x_before_landing = player_x_before_landing,
        .player_y_before_landing = player_y_before_landing,
    };
    return result;
}

static void probe_print_step_result(const char* label, ProbeStepDodgeResult r) {
    printf(
        "%s first_visible_tick=%d first_timer=%d move_action=%d move_tick=%d landing_tick=%d after_move=(%d,%d) before_landing=(%d,%d) interaction_after_move=%d damage=%d\n",
        label,
        r.first_visible_tick,
        r.first_visible_timer,
        r.move_action,
        r.move_tick,
        r.landing_tick,
        r.player_x_after_move,
        r.player_y_after_move,
        r.player_x_before_landing,
        r.player_y_before_landing,
        r.interaction_after_move,
        r.damage_taken);
}

static void probe_obs_honesty(void) {
    printf("\n== OBS HONESTY ==\n");
    ColosseumContext ctx;
    ColosseumState s;
    static float obs[COLO_NUM_OBS];
    probe_init_empty_state(&s, &ctx, 0x1001u);
    ColoJavelinState* jv = probe_spawn_javelin(&s);
    jv->skyfall_pending = 1;
    jv->skyfall_tile_x = PROBE_PLAYER_X;
    jv->skyfall_tile_y = PROBE_PLAYER_Y;
    jv->skyfall_timer = COLO_JAVELIN_SKYFALL_DELAY;
    jv->skyfall_damage = PROBE_SKYFALL_DAMAGE;

    const int player_positions[][2] = {
        {PROBE_PLAYER_X, PROBE_PLAYER_Y},
        {PROBE_PLAYER_X - 1, PROBE_PLAYER_Y},
        {PROBE_PLAYER_X + 1, PROBE_PLAYER_Y + 1},
        {PROBE_PLAYER_X - 2, PROBE_PLAYER_Y - 2},
    };
    int count = (int)(sizeof(player_positions) / sizeof(player_positions[0]));
    for (int i = 0; i < count; i++) {
        s.player.x = player_positions[i][0];
        s.player.y = player_positions[i][1];
        col_rebuild_player_collision_flags(&s);
        ProbeSkyfallObs seen = probe_read_skyfall_obs(&s, &ctx, obs);
        float expected_dx = col_obs_rel_x(jv->skyfall_tile_x, s.player.x);
        float expected_dy = col_obs_rel_y(jv->skyfall_tile_y, s.player.y);
        printf(
            "OBS player=(%d,%d) marked=(%d,%d) raw_delta=(%d,%d) obs_slot=%d pending=%.1f timer=%.3f dx=%.6f dy=%.6f\n",
            s.player.x,
            s.player.y,
            jv->skyfall_tile_x,
            jv->skyfall_tile_y,
            jv->skyfall_tile_x - s.player.x,
            jv->skyfall_tile_y - s.player.y,
            seen.obs_slot,
            seen.obs_pending,
            seen.obs_timer,
            seen.obs_dx,
            seen.obs_dy);
        probe_check_float("skyfall tell dx matches marked tile", seen.obs_dx, expected_dx);
        probe_check_float("skyfall tell dy matches marked tile", seen.obs_dy, expected_dy);
    }
}

static void probe_direct_resolve(void) {
    printf("\n== SIM DODGE LOGIC ==\n");
    ColosseumContext ctx;
    ColosseumState s;
    probe_init_empty_state(&s, &ctx, 0x2002u);
    ColoJavelinState* jv = probe_spawn_javelin(&s);
    jv->skyfall_pending = 1;
    jv->skyfall_timer = 1;
    jv->skyfall_damage = PROBE_SKYFALL_DAMAGE;
    jv->skyfall_tile_x = s.player.x;
    jv->skyfall_tile_y = s.player.y;
    int hp_before = s.player.current_hitpoints;
    col_npc_resolve_javelin_skyfall(&s, &ctx, 0);
    printf("DIRECT on_tile hp=%d->%d damage=%d pending=%d\n",
        hp_before,
        s.player.current_hitpoints,
        hp_before - s.player.current_hitpoints,
        jv->skyfall_pending);
    probe_check("direct on-tile skyfall applies damage",
        hp_before - s.player.current_hitpoints == PROBE_SKYFALL_DAMAGE &&
        jv->skyfall_pending == 0);

    probe_init_empty_state(&s, &ctx, 0x2003u);
    jv = probe_spawn_javelin(&s);
    jv->skyfall_pending = 1;
    jv->skyfall_timer = 1;
    jv->skyfall_damage = PROBE_SKYFALL_DAMAGE;
    jv->skyfall_tile_x = s.player.x;
    jv->skyfall_tile_y = s.player.y;
    s.player.x += 1;
    col_rebuild_player_collision_flags(&s);
    hp_before = s.player.current_hitpoints;
    col_npc_resolve_javelin_skyfall(&s, &ctx, 0);
    printf("DIRECT off_tile player=(%d,%d) marked=(%d,%d) hp=%d->%d damage=%d pending=%d\n",
        s.player.x,
        s.player.y,
        jv->skyfall_tile_x,
        jv->skyfall_tile_y,
        hp_before,
        s.player.current_hitpoints,
        hp_before - s.player.current_hitpoints,
        jv->skyfall_pending);
    probe_check("direct off-tile skyfall applies zero damage",
        hp_before - s.player.current_hitpoints == 0 &&
        jv->skyfall_pending == 0);
}

static void probe_full_step_dodgeability(void) {
    printf("\n== FULL-STEP DODGEABILITY ==\n");
    ProbeStepDodgeResult wait0 = probe_run_no_lock_wait_case(0);
    ProbeStepDodgeResult wait1 = probe_run_no_lock_wait_case(1);
    ProbeStepDodgeResult wait2 = probe_run_no_lock_wait_case(2);
    probe_print_step_result("FULL_STEP wait0_move_from_timer3", wait0);
    probe_print_step_result("FULL_STEP wait1_move_from_timer2", wait1);
    probe_print_step_result("FULL_STEP wait2_move_from_timer1", wait2);
    probe_check("timer 3 visible action dodges skyfall", wait0.damage_taken == 0);
    probe_check("timer 2 visible action dodges skyfall", wait1.damage_taken == 0);
    probe_check("timer 1 visible action is too late", wait2.damage_taken == PROBE_SKYFALL_DAMAGE);
    printf("ACTIONABLE_LEAD move_actions=2 visible_timers=3,2 too_late_timer=1\n");
}

static void probe_attack_lock_interaction(void) {
    printf("\n== ATTACK-LOCK INTERACTION ==\n");
    ProbeStepDodgeResult idle = probe_run_attack_lock_idle_case("locked_idle");
    ProbeStepDodgeResult move_only = probe_run_attack_lock_case(
        "locked_move_only",
        PROBE_TARGET_AFTER_MOVE_NONE);
    ProbeStepDodgeResult target_move = probe_run_attack_lock_case(
        "locked_target_and_move_same_tick",
        PROBE_TARGET_AFTER_MOVE_SAME_TICK);
    ProbeStepDodgeResult move_then_target = probe_run_attack_lock_case(
        "locked_move_then_retarget_next_tick",
        PROBE_TARGET_AFTER_MOVE_NEXT_TICK);
    probe_print_step_result("LOCK idle", idle);
    probe_print_step_result("LOCK move_only", move_only);
    probe_print_step_result("LOCK target_and_move_same_tick", target_move);
    probe_print_step_result("LOCK move_then_retarget_next_tick", move_then_target);
    probe_check("locked idle eats skyfall", idle.damage_taken == PROBE_SKYFALL_DAMAGE);
    probe_check("MOVE-only while locked breaks lock and dodges", move_only.damage_taken == 0);
    probe_check("TARGET+MOVE same tick stays on tile and eats skyfall",
        target_move.damage_taken == PROBE_SKYFALL_DAMAGE);
    probe_check("retargeting one tick after a successful dodge stays safe",
        move_then_target.damage_taken == 0);
}

int main(void) {
    printf("=== Colosseum javelin skyfall dodge probe ===\n");
    printf("step order under test: action pretick, NPC skyfall resolve, NPC attack fire, player movement, obs write\n");
    probe_obs_honesty();
    probe_direct_resolve();
    probe_full_step_dodgeability();
    probe_attack_lock_interaction();
    printf("\nVERDICT_DATA obs_honest=1 off_tile_dodges=1 real_step_actionable_lead=2 move_only_lock_dodges=1 target_plus_move_blocks_dodge=1\n");
    return 0;
}
