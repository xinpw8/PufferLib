#ifndef OSRS_INVENTORY_DRAG_H
#define OSRS_INVENTORY_DRAG_H

#define OSRS_INVENTORY_DRAG_DEAD_ZONE 5
#define OSRS_INVENTORY_DRAG_HOLD_SECONDS 0.180

static inline int osrs_inventory_drag_ready(double held_seconds, int delta_x, int delta_y) {
    return held_seconds >= OSRS_INVENTORY_DRAG_HOLD_SECONDS &&
        (delta_x > OSRS_INVENTORY_DRAG_DEAD_ZONE ||
         delta_x < -OSRS_INVENTORY_DRAG_DEAD_ZONE ||
         delta_y > OSRS_INVENTORY_DRAG_DEAD_ZONE ||
         delta_y < -OSRS_INVENTORY_DRAG_DEAD_ZONE);
}

static inline void osrs_inventory_drag_release(
    int* active,
    int* source_slot,
    int* dim_slot,
    int* dim_timer
) {
    *active = 0;
    *source_slot = -1;
    *dim_slot = -1;
    *dim_timer = 0;
}

#endif
