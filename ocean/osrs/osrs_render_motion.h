#ifndef OSRS_RENDER_MOTION_H
#define OSRS_RENDER_MOTION_H

#include <math.h>
#include <stddef.h>

#define OSRS_RENDER_SUB_UNITS_PER_TILE 128.0f
#define OSRS_RENDER_CLIENT_TICKS_PER_GAME_TICK 30.0f
#define OSRS_RENDER_ENTITY_GROUND_LIFT_OSRS_UNITS 8.0f
#define OSRS_RENDER_ENTITY_GROUND_LIFT \
    (OSRS_RENDER_ENTITY_GROUND_LIFT_OSRS_UNITS / OSRS_RENDER_SUB_UNITS_PER_TILE)
#define OSRS_RENDER_RUN_SPEED_THRESHOLD 8.0f
#define OSRS_RENDER_WAYPOINT_QUEUE_DEPTH 10
#define OSRS_RENDER_AXIS_SNAP_SUB_UNITS 256.0f

typedef enum {
    RENDER_MOVEMENT_NORMAL = 0,
    RENDER_MOVEMENT_TELEPORT = 1,
} RenderMovementKind;

typedef struct {
    float x[OSRS_RENDER_WAYPOINT_QUEUE_DEPTH];
    float y[OSRS_RENDER_WAYPOINT_QUEUE_DEPTH];
    int running[OSRS_RENDER_WAYPOINT_QUEUE_DEPTH];
    int length;
} OsrsRenderWaypointQueue;

static inline float osrs_render_entity_model_ground(float ground) {
    return ground + OSRS_RENDER_ENTITY_GROUND_LIFT;
}

static inline int osrs_render_speed_one_client_tick(
    int path_length,
    int waypoint_running,
    int* stall_debt
) {
    int speed = 4;
    if (path_length > 2) speed = 6;
    if (path_length > 3) speed = 8;
    if (stall_debt != NULL && *stall_debt > 0 && path_length > 1) {
        speed = 8;
        (*stall_debt)--;
    }
    if (waypoint_running) speed <<= 1;
    return speed;
}

static inline int osrs_render_speed_uses_run_pose(float effective_speed) {
    return effective_speed >= OSRS_RENDER_RUN_SPEED_THRESHOLD;
}

static inline int osrs_render_should_seed_visual_position(
    int visible,
    int new_identity,
    int became_visible,
    RenderMovementKind movement_kind
) {
    if (!visible) return 0;
    return new_identity || became_visible ||
        movement_kind == RENDER_MOVEMENT_TELEPORT;
}

static inline float osrs_render_advance_axis_toward(
    float sub, float dest, float speed
) {
    float dx = dest - sub;
    if (dx == 0.0f) return sub;
    if (dx > 0.0f) return sub + (dx > speed ? speed : dx);
    return sub + (dx < -speed ? -speed : dx);
}

static inline void osrs_render_waypoint_queue_clear(OsrsRenderWaypointQueue* q) {
    q->length = 0;
}

static inline void osrs_render_waypoint_push(
    OsrsRenderWaypointQueue* q, float x, float y, int running
) {
    int n = q->length < OSRS_RENDER_WAYPOINT_QUEUE_DEPTH - 1
        ? q->length
        : OSRS_RENDER_WAYPOINT_QUEUE_DEPTH - 1;
    for (int i = n; i > 0; i--) {
        q->x[i] = q->x[i - 1];
        q->y[i] = q->y[i - 1];
        q->running[i] = q->running[i - 1];
    }
    q->x[0] = x;
    q->y[0] = y;
    q->running[0] = running;
    q->length = n + 1;
}

static inline int osrs_render_waypoint_advance_one_client_tick(
    OsrsRenderWaypointQueue* q,
    float* sub_x,
    float* sub_y,
    int* stall_debt,
    int* speed_out,
    float* dir_dx,
    float* dir_dy
) {
    if (q->length == 0) {
        *speed_out = 0;
        *dir_dx = 0.0f;
        *dir_dy = 0.0f;
        return 0;
    }
    int wi = q->length - 1;
    float wx = q->x[wi];
    float wy = q->y[wi];
    *dir_dx = wx - *sub_x;
    *dir_dy = wy - *sub_y;
    if (*dir_dx > OSRS_RENDER_AXIS_SNAP_SUB_UNITS ||
        *dir_dx < -OSRS_RENDER_AXIS_SNAP_SUB_UNITS ||
        *dir_dy > OSRS_RENDER_AXIS_SNAP_SUB_UNITS ||
        *dir_dy < -OSRS_RENDER_AXIS_SNAP_SUB_UNITS) {
        *sub_x = wx;
        *sub_y = wy;
        q->length--;
        *speed_out = 0;
        return 1;
    }
    int speed = osrs_render_speed_one_client_tick(
        q->length, q->running[wi], stall_debt);
    *sub_x = osrs_render_advance_axis_toward(*sub_x, wx, (float)speed);
    *sub_y = osrs_render_advance_axis_toward(*sub_y, wy, (float)speed);
    if (*sub_x == wx && *sub_y == wy) q->length--;
    *speed_out = speed;
    return 1;
}

#endif
