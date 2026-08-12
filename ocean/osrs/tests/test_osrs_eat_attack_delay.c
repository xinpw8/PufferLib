/**
 * Engine contract (osrs-engine-quirks section 12): food adds its attack delay
 * ONLY to an existing cooldown. From a ready weapon, eating adds nothing.
 * Karambwan adds 2, standard food 3, and they stack onto a live cooldown.
 */
#include <stdio.h>
#include <string.h>
#include "ocean/osrs/encounters/encounter_colosseum.h"
#include "ocean/osrs/tests/osrs_test_check.h"

static Player ready_player(void) {
    Player p;
    memset(&p, 0, sizeof(p));
    p.base_hitpoints = 99;
    p.current_hitpoints = 50;
    p.attack_timer = 0;
    p.attack_timer_uncapped = 0;
    p.has_attack_timer = 0;
    return p;
}

static void test_engine_contract(void) {
    Player p = ready_player();
    osrs_player_eat_food_effects(&p, FOOD_SHARK);
    CHECK("shark from ready adds no attack delay", p.attack_timer == 0);

    p = ready_player();
    osrs_player_eat_food_effects(&p, FOOD_KARAMBWAN);
    CHECK("karambwan from ready adds no attack delay", p.attack_timer == 0);

    p = ready_player();
    p.attack_timer = 4;
    osrs_player_eat_food_effects(&p, FOOD_SHARK);
    CHECK("shark adds 3 to an existing cooldown", p.attack_timer == 7);

    p = ready_player();
    p.attack_timer = 4;
    osrs_player_eat_food_effects(&p, FOOD_KARAMBWAN);
    CHECK("karambwan adds 2 to an existing cooldown", p.attack_timer == 6);

    /* the bug: has_attack_timer unset must not erase a live cooldown */
    p = ready_player();
    p.attack_timer = 5;
    p.has_attack_timer = 0;
    osrs_player_eat_food_effects(&p, FOOD_SHARK);
    CHECK("a live cooldown survives the first eat when has_attack_timer is unset", p.attack_timer == 8);

    /* food then karambwan on a live cooldown stacks 3 + 2 */
    p = ready_player();
    p.attack_timer = 4;
    osrs_player_eat_food_effects(&p, FOOD_SHARK);
    osrs_player_eat_food_effects(&p, FOOD_KARAMBWAN);
    CHECK("shark then karambwan stack onto a live cooldown", p.attack_timer == 9);

    p = ready_player();
    p.current_hitpoints = 50;
    osrs_player_eat_food_effects(&p, FOOD_SHARK);
    CHECK("eating still heals from ready", p.current_hitpoints > 50);
}

static void test_colosseum_charges_the_cooldown(void) {
    ColosseumContext ctx;
    static ColosseumState s;
    col_init_context_typed(&ctx);
    col_finalize_route_topology(&ctx);
    memset(&s, 0, sizeof(s));
    col_reset_ctx((EncounterState*)&s, (EncounterContext*)&ctx, 4242u);

    s.player.current_hitpoints = 20;
    s.player.attack_timer = 4;
    s.player.food_timer = 0;
    s.player.karambwan_timer = 0;
    int before = s.player.current_hitpoints;

    int consumed = col_apply_food_cell(&s, OSRS_CONSUMABLE_SHARK_FOOD);
    CHECK("colosseum shark is consumed when the food timer is clear", consumed == 1);
    CHECK("colosseum shark heals", s.player.current_hitpoints > before);
    CHECK("colosseum eating charges the attack cooldown instead of being free", s.player.attack_timer == 7);

    /* refused eat must report not-consumed so the caller keeps the food */
    s.player.food_timer = 3;
    int refused = col_apply_food_cell(&s, OSRS_CONSUMABLE_SHARK_FOOD);
    CHECK("colosseum eat under an active food timer is refused", refused == 0);
}

int main(void) {
    test_engine_contract();
    test_colosseum_charges_the_cooldown();
    return osrs_test_summary();
}
