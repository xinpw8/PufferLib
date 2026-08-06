/**
 * @file test_osrs_player_step.c
 * @brief Shared player-step command contract: one click per tick.
 *
 * The engine cannot both start an attack and walk to a chosen tile on the same
 * tick. A ground click cancels the entity interaction; an entity click cancels
 * the walk. These pin that contract at the shared layer, where every encounter
 * now routes through OsrsPlayerCommand.
 */

#include <stdio.h>
#include <string.h>

#include "ocean/osrs/osrs_encounter_player.h"

static int tests_run = 0;
static int tests_failed = 0;

#define CHECK(label, cond) do { \
    tests_run++; \
    if (!(cond)) { \
        tests_failed++; \
        printf("  FAIL: %s\n", (label)); \
    } \
} while (0)

#define STEP_GRID 24

typedef struct {
    OsrsAttackTarget target;
    int target_valid;
} StepTargetCtx;

static int step_tile_walkable(void* ctx, int x, int y) {
    (void)ctx;
    return x >= 0 && x < STEP_GRID && y >= 0 && y < STEP_GRID;
}

static int step_lookup_target(void* ctx, int target_slot, OsrsAttackTarget* out) {
    StepTargetCtx* c = (StepTargetCtx*)ctx;
    if (!c->target_valid || target_slot != c->target.slot) return 0;
    *out = c->target;
    return 1;
}

static OsrsEncounterArena step_arena(void) {
    OsrsEncounterArena arena;
    memset(&arena, 0, sizeof(arena));
    arena.is_walkable = step_tile_walkable;
    arena.los_query = osrs_los_open_query();
    arena.arena_w = STEP_GRID;
    arena.arena_h = STEP_GRID;
    return arena;
}

/* an entity click cancels a walk already in flight: the player closes on the
   target instead of continuing to the clicked tile. this is the nh_pvp/zulrah
   regression — EXPLICIT_FIRST used to honour both, giving free damage while
   repositioning. */
static void test_target_command_cancels_walk_in_flight(void) {
    printf("--- entity click cancels a walk in flight ---\n");

    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 5; player.y = 5;
    player.run_energy = 10000;

    OsrsInteraction interaction;
    osrs_interaction_init(&interaction);

    StepTargetCtx tctx = {
        .target = { .slot = 3, .x = 5, .y = 12, .size = 1, .attack_range = 1 },
        .target_valid = 1,
    };

    int dest_x = 5, dest_y = 1;   /* walking south, target is north */
    OsrsEncounterArena arena = step_arena();

    OsrsPlayerStepInput input;
    memset(&input, 0, sizeof(input));
    input.player = &player;
    input.interaction = &interaction;
    input.target_lookup = step_lookup_target;
    input.target_ctx = &tctx;
    input.command.kind = OSRS_PLAYER_CMD_TARGET;
    input.command.target_slot = 3;
    input.dest_x = &dest_x;
    input.dest_y = &dest_y;
    input.arena = arena;

    OsrsPlayerStepResult r = osrs_encounter_player_step(&input);

    CHECK("interaction is set to the clicked entity", interaction.target_slot == 3);
    CHECK("walk destination is cancelled", dest_x == -1 && dest_y == -1);
    CHECK("no explicit move ran", r.explicit_moved == 0);
    CHECK("player closed on the target, not the clicked tile", player.y > 5);
}

/* a ground click cancels the interaction and walks. */
static void test_move_command_cancels_interaction(void) {
    printf("--- ground click cancels the interaction ---\n");

    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 5; player.y = 5;
    player.run_energy = 10000;

    OsrsInteraction interaction;
    osrs_interaction_init(&interaction);
    osrs_interaction_set(&interaction, 3);

    StepTargetCtx tctx = {
        .target = { .slot = 3, .x = 5, .y = 12, .size = 1, .attack_range = 1 },
        .target_valid = 1,
    };

    int dest_x = 5, dest_y = 1;
    OsrsEncounterArena arena = step_arena();

    OsrsPlayerStepInput input;
    memset(&input, 0, sizeof(input));
    input.player = &player;
    input.interaction = &interaction;
    input.target_lookup = step_lookup_target;
    input.target_ctx = &tctx;
    input.command.kind = OSRS_PLAYER_CMD_MOVE;
    input.command.move_kind = OSRS_PLAYER_MOVE_DESTINATION;
    input.dest_x = &dest_x;
    input.dest_y = &dest_y;
    input.arena = arena;

    OsrsPlayerStepResult r = osrs_encounter_player_step(&input);

    CHECK("interaction is cancelled", !osrs_interaction_active(&interaction));
    CHECK("explicit move ran", r.explicit_moved == 1);
    CHECK("player walked toward the clicked tile", player.y < 5);
}

/* no click: an active interaction keeps auto-chasing. */
static void test_none_command_chases_active_interaction(void) {
    printf("--- idle tick auto-chases the standing interaction ---\n");

    Player player;
    memset(&player, 0, sizeof(player));
    player.x = 5; player.y = 5;
    player.run_energy = 10000;

    OsrsInteraction interaction;
    osrs_interaction_init(&interaction);
    osrs_interaction_set(&interaction, 3);

    StepTargetCtx tctx = {
        .target = { .slot = 3, .x = 5, .y = 12, .size = 1, .attack_range = 1 },
        .target_valid = 1,
    };

    int dest_x = -1, dest_y = -1;
    OsrsEncounterArena arena = step_arena();

    OsrsPlayerStepInput input;
    memset(&input, 0, sizeof(input));
    input.player = &player;
    input.interaction = &interaction;
    input.target_lookup = step_lookup_target;
    input.target_ctx = &tctx;
    input.command.kind = OSRS_PLAYER_CMD_NONE;
    input.dest_x = &dest_x;
    input.dest_y = &dest_y;
    input.arena = arena;

    OsrsPlayerStepResult r = osrs_encounter_player_step(&input);

    CHECK("interaction survives an idle tick", osrs_interaction_active(&interaction));
    CHECK("chase ran", r.chased_target == 1);
    CHECK("no explicit move ran", r.explicit_moved == 0);
}

int main(void) {
    test_target_command_cancels_walk_in_flight();
    test_move_command_cancels_interaction();
    test_none_command_chases_active_interaction();

    printf("\n%d/%d tests passed\n", tests_run - tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}
