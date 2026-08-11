#ifndef OSRS_INTERACTION_H
#define OSRS_INTERACTION_H

#define OSRS_INTERACTION_ROUTE_MAX_WAYPOINTS 25

typedef enum {
    OSRS_INTERACTION_ROUTE_EMPTY = 0,
    OSRS_INTERACTION_ROUTE_READY,
    OSRS_INTERACTION_ROUTE_FAILED,
} OsrsInteractionRouteState;

typedef struct {
    OsrsInteractionRouteState state;
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
} OsrsInteractionRoute;

typedef struct {
    int target_slot;
    OsrsInteractionRoute route;
} OsrsInteraction;

static inline void osrs_interaction_route_clear(OsrsInteraction* ix) {
    ix->route.state = OSRS_INTERACTION_ROUTE_EMPTY;
    ix->route.waypoint_count = 0;
    ix->route.waypoint_index = 0;
}

static inline void osrs_interaction_set(OsrsInteraction* ix, int target_slot) {
    if (ix->target_slot == target_slot) return;
    ix->target_slot = target_slot;
    osrs_interaction_route_clear(ix);
}

static inline void osrs_interaction_clear(OsrsInteraction* ix) {
    ix->target_slot = -1;
    osrs_interaction_route_clear(ix);
}

static inline int osrs_interaction_active(const OsrsInteraction* ix) {
    return ix->target_slot >= 0;
}

static inline void osrs_interaction_init(OsrsInteraction* ix) {
    ix->target_slot = -1;
    osrs_interaction_route_clear(ix);
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
