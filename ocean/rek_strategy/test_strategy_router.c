#include <assert.h>
#include <stdio.h>

#include "strategy_router.h"

static RekStrategyCapabilities all_capabilities(void) {
    return (RekStrategyCapabilities){
        .locomotion_executor_present = 1,
        .canned_move_executor_present = 1,
        .drive_recovery_present = 1,
    };
}

int main(void) {
    const int expected_slots[REK_STRATEGY_MOVE_CATEGORIES] = {
        -1, 2, 3, 4, 5, 9, 10,
    };
    for (int category = 0; category < REK_STRATEGY_MOVE_CATEGORIES; category++) {
        assert(REK_STRATEGY_MOVE_SLOTS[category] == expected_slots[category]);
    }

    RekStrategyRouter router;
    RekStrategyCapabilities missing = all_capabilities();
    missing.drive_recovery_present = 0;
    assert(!rek_strategy_router_init(&router, missing));
    assert(!router.initialized);
    assert(rek_strategy_router_init(&router, all_capabilities()));
    router.held_velocity[1] = -1.0f;
    assert(rek_strategy_translation_held(&router));
    router.held_velocity[1] = 0.0f;
    router.held_velocity[2] = 1.0f;
    assert(!rek_strategy_translation_held(&router));

    RekStrategyAction action = {
        .velocity_bins = {2, 1, 0},
        .move_category = 2,
    };
    RekStrategyGates clear = {0};
    RekStrategyOutput output;
    assert(rek_strategy_route(&router, &action, &clear, &output));
    assert(output.velocity[0] == 1.0f);
    assert(output.velocity[1] == 0.0f);
    assert(output.velocity[2] == -1.0f);
    assert(!output.request_move && output.move_slot == -1);
    assert(router.last_emitted_move_category == 0);

    // Translation is held and may overlap yaw. The still-held move category
    // starts only after translation is released, and owns all root axes.
    action.velocity_bins[0] = 1;
    RekStrategyGates settling = {.translation_not_settled = 1};
    assert(rek_strategy_route(&router, &action, &settling, &output));
    assert(!output.request_move && output.move_slot == -1);
    assert(output.velocity[2] == -1.0f);
    assert(router.last_emitted_move_category == 0);

    assert(rek_strategy_route(&router, &action, &clear, &output));
    assert(output.request_move && output.move_slot == 3);
    assert(output.velocity[0] == 0.0f);
    assert(output.velocity[1] == 0.0f);
    assert(output.velocity[2] == 0.0f);

    assert(rek_strategy_route(&router, &action, &clear, &output));
    assert(!output.request_move && output.move_slot == -1);
    assert(output.velocity[2] == -1.0f);

    RekStrategyGates busy = {.move_in_progress = 1};
    action.velocity_bins[0] = 2;
    action.velocity_bins[1] = 0;
    action.velocity_bins[2] = 2;
    action.move_category = 0;
    assert(rek_strategy_route(&router, &action, &busy, &output));
    assert(!output.request_move && output.move_slot == -1);
    assert(output.velocity[0] == 0.0f);
    assert(output.velocity[1] == 0.0f);
    assert(output.velocity[2] == 0.0f);
    assert(router.held_velocity[0] == 1.0f);
    assert(router.held_velocity[1] == -1.0f);
    assert(router.held_velocity[2] == 1.0f);

    assert(rek_strategy_route(&router, &action, &clear, &output));
    assert(!output.request_move);
    assert(output.velocity[0] == 1.0f);
    assert(output.velocity[1] == -1.0f);
    assert(output.velocity[2] == 1.0f);

    action.velocity_bins[0] = 1;
    action.velocity_bins[1] = 1;
    action.velocity_bins[2] = 0;
    action.move_category = 2;
    assert(rek_strategy_route(&router, &action, &clear, &output));
    assert(output.request_move && output.move_slot == 3);
    assert(output.velocity[2] == 0.0f);

    action.move_category = 3;
    assert(rek_strategy_route(&router, &action, &busy, &output));
    assert(!output.request_move);
    assert(rek_strategy_route(&router, &action, &clear, &output));
    assert(output.request_move && output.move_slot == 4);

    action.velocity_bins[0] = 0;
    action.velocity_bins[1] = 0;
    action.velocity_bins[2] = 2;
    action.move_category = 4;
    RekStrategyGates recovery = {.recovering = 1};
    assert(rek_strategy_route(&router, &action, &recovery, &output));
    assert(output.recovery_has_priority);
    assert(output.velocity[0] == 0.0f);
    assert(output.velocity[1] == 0.0f);
    assert(output.velocity[2] == 0.0f);
    assert(!output.request_move);
    assert(router.held_velocity[0] == -1.0f);
    assert(router.held_velocity[1] == -1.0f);
    assert(router.held_velocity[2] == 1.0f);

    action.move_category = 0;
    assert(rek_strategy_route(&router, &action, &recovery, &output));
    assert(output.recovery_has_priority);
    assert(router.last_emitted_move_category == 0);

    action.velocity_bins[0] = 3;
    assert(!rek_strategy_route(&router, &action, &clear, &output));
    assert(!output.request_move && output.move_slot == -1);

    puts("strategy router tests: ok");
    return 0;
}
