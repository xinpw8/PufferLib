#ifndef OSRS_INTERACTION_H
#define OSRS_INTERACTION_H

typedef struct {
    int target_slot;
} OsrsInteraction;

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

static inline void osrs_spec_toggle(int* spec_armed) {
    *spec_armed = !(*spec_armed);
}

static inline void osrs_spec_disarm(int* spec_armed) {
    *spec_armed = 0;
}

#endif
