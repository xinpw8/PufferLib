// Measured T800 strategy action routing. This is an interface layer only.
// It contains no locomotion, canned-move, recovery, contact, or reward model.

#pragma once

#include <math.h>
#include <string.h>

#define REK_STRATEGY_VELOCITY_DIMS 3
#define REK_STRATEGY_MOVE_CATEGORIES 7
#define REK_STRATEGY_NUM_ACTION_HEADS 4
#define REK_STRATEGY_ACT_SIZES {3, 3, 3, 7}
#define REK_STRATEGY_BUILD_FINGERPRINT \
    "f84f187491e3b5cd73493de379ed972c5580b60d63f33956e396e6dec28b1659"
#define REK_STRATEGY_CONTRACT_SHA256 \
    "f5ee07beaa3229270b7a6202f28e7ec886ab65f0ea73145fa9d545a5d3a856d7"

static const int REK_STRATEGY_MOVE_SLOTS[REK_STRATEGY_MOVE_CATEGORIES] = {
    -1, 2, 3, 4, 5, 9, 10,
};

typedef struct RekStrategyCapabilities {
    int locomotion_executor_present;
    int canned_move_executor_present;
    int drive_recovery_present;
} RekStrategyCapabilities;

typedef struct RekStrategyAction {
    // Puffer categories 0, 1, 2 map to measured keyboard values -1, 0, +1.
    int velocity_bins[REK_STRATEGY_VELOCITY_DIMS];
    // 0 emits no move request. Categories 1..6 map through MOVE_SLOTS.
    int move_category;
} RekStrategyAction;

typedef struct RekStrategyGates {
    int recovering;
    int move_in_progress;
    int manual_switch_cooldown_active;
} RekStrategyGates;

typedef struct RekStrategyOutput {
    float velocity[REK_STRATEGY_VELOCITY_DIMS];
    int request_move;
    int move_slot;
    int recovery_has_priority;
} RekStrategyOutput;

typedef struct RekStrategyRouter {
    float held_velocity[REK_STRATEGY_VELOCITY_DIMS];
    int last_emitted_move_category;
    int initialized;
} RekStrategyRouter;

static int rek_strategy_boolean(int value) {
    return value == 0 || value == 1;
}

static int rek_strategy_router_init(
        RekStrategyRouter* router,
        RekStrategyCapabilities capabilities) {
    if (router == NULL) return 0;
    memset(router, 0, sizeof(*router));
    if (capabilities.locomotion_executor_present != 1
            || capabilities.canned_move_executor_present != 1
            || capabilities.drive_recovery_present != 1) {
        return 0;
    }
    router->initialized = 1;
    return 1;
}

static float rek_strategy_decode_velocity_bin(int category) {
    return (float)(category - 1);
}

static int rek_strategy_action_is_valid(const RekStrategyAction* action) {
    if (action == NULL
            || action->move_category < 0
            || action->move_category >= REK_STRATEGY_MOVE_CATEGORIES) {
        return 0;
    }
    for (int axis = 0; axis < REK_STRATEGY_VELOCITY_DIMS; axis++) {
        if (action->velocity_bins[axis] < 0
                || action->velocity_bins[axis] > 2) {
            return 0;
        }
    }
    return 1;
}

static int rek_strategy_gates_are_valid(const RekStrategyGates* gates) {
    return gates != NULL
        && rek_strategy_boolean(gates->recovering)
        && rek_strategy_boolean(gates->move_in_progress)
        && rek_strategy_boolean(gates->manual_switch_cooldown_active);
}

static int rek_strategy_route(
        RekStrategyRouter* router,
        const RekStrategyAction* action,
        const RekStrategyGates* gates,
        RekStrategyOutput* output) {
    if (output == NULL) return 0;
    memset(output, 0, sizeof(*output));
    output->move_slot = -1;
    if (router == NULL || router->initialized != 1
            || !rek_strategy_action_is_valid(action)
            || !rek_strategy_gates_are_valid(gates)) {
        return 0;
    }

    // DriveRecovery is evaluated outside this router and always wins. Learned
    // velocity and move requests are suppressed until its recovering gate clears.
    if (gates->recovering) {
        output->recovery_has_priority = 1;
        return 1;
    }

    for (int axis = 0; axis < REK_STRATEGY_VELOCITY_DIMS; axis++) {
        router->held_velocity[axis] = rek_strategy_decode_velocity_bin(
            action->velocity_bins[axis]
        );
        output->velocity[axis] = router->held_velocity[axis];
    }

    if (action->move_category == 0) {
        router->last_emitted_move_category = 0;
        return 1;
    }
    if (gates->move_in_progress || gates->manual_switch_cooldown_active
            || action->move_category == router->last_emitted_move_category) {
        return 1;
    }

    output->request_move = 1;
    output->move_slot = REK_STRATEGY_MOVE_SLOTS[action->move_category];
    router->last_emitted_move_category = action->move_category;
    return 1;
}

