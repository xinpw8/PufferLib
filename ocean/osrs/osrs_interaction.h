#ifndef OSRS_INTERACTION_H
#define OSRS_INTERACTION_H

#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define OSRS_INTERACTION_ROUTE_MAX_WAYPOINTS 25

typedef enum {
    OSRS_INTERACTION_ROUTE_EMPTY = 0,
    OSRS_INTERACTION_ROUTE_READY,
    OSRS_INTERACTION_ROUTE_FAILED,
} OsrsInteractionRouteState;

typedef struct {
    OsrsInteractionRouteState state;
    uint64_t topology_revision;
    uint64_t blocker_revision;
    int actor_size;
    uint8_t movement_mode;
    uint8_t cost_policy;
    int target_x;
    int target_y;
    int target_size;
    int attack_range;
    int planned_source_x;
    int planned_source_y;
    int expected_player_x;
    int expected_player_y;
    int waypoint_count;
    int waypoint_index;
    int waypoint_x[OSRS_INTERACTION_ROUTE_MAX_WAYPOINTS];
    int waypoint_y[OSRS_INTERACTION_ROUTE_MAX_WAYPOINTS];
} OsrsActorRouteCache;

#define OSRS_INTERACTION_SERIALIZED_ROUTE_BYTES 244

typedef struct {
    int target_slot;
    uint8_t serialized_route_padding[OSRS_INTERACTION_SERIALIZED_ROUTE_BYTES];
} OsrsInteraction;

static_assert(sizeof(OsrsInteraction) == 248, "OsrsInteraction serialized layout");
static_assert(offsetof(OsrsInteraction, target_slot) == 0, "OsrsInteraction target offset");
static_assert(offsetof(OsrsInteraction, serialized_route_padding) == 4,
    "OsrsInteraction reserved route offset");

static inline void osrs_interaction_zero_serialized_route_padding(
    OsrsInteraction* ix
) {
    memset(ix->serialized_route_padding, 0, sizeof(ix->serialized_route_padding));
}

static inline void osrs_actor_route_cache_clear(OsrsActorRouteCache* route) {
    route->state = OSRS_INTERACTION_ROUTE_EMPTY;
    route->waypoint_count = 0;
    route->waypoint_index = 0;
}

static inline void osrs_interaction_set(OsrsInteraction* ix, int target_slot) {
    ix->target_slot = target_slot;
}

static inline void osrs_interaction_clear(OsrsInteraction* ix) {
    ix->target_slot = -1;
}
static inline int osrs_interaction_active(const OsrsInteraction* ix) {
    return ix->target_slot >= 0;
}

static inline void osrs_interaction_init(OsrsInteraction* ix) {
    ix->target_slot = -1;
    osrs_interaction_zero_serialized_route_padding(ix);
}

#define OSRS_IACT_NONE     0
#define OSRS_IACT_MOVE     1
#define OSRS_IACT_EAT      2
#define OSRS_IACT_DRINK    3
#define OSRS_IACT_EQUIP    4
#define OSRS_IACT_PRAYER   5
#define OSRS_IACT_SPEC     6
#define OSRS_IACT_ATTACK   7

static inline int osrs_interaction_check_interrupt(OsrsInteraction* ix, int action_type) {
    switch (action_type) {
        case OSRS_IACT_MOVE:
        case OSRS_IACT_EAT:
        case OSRS_IACT_DRINK:
        case OSRS_IACT_EQUIP:
            osrs_interaction_clear(ix);
            return 1;
        case OSRS_IACT_NONE:
        case OSRS_IACT_PRAYER:
        case OSRS_IACT_SPEC:
        case OSRS_IACT_ATTACK:
        default:
            return 0;
    }
}

static inline void osrs_spec_disarm(int* spec_armed) {
    *spec_armed = 0;
}

#endif
