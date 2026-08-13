#ifndef OSRS_PROJECTILE_ORIENTATION_H
#define OSRS_PROJECTILE_ORIENTATION_H

#include <math.h>

#define OSRS_PROJECTILE_ORIENTATION_PI 3.14159265358979323846f

typedef struct {
    float yaw;
    float pitch;
} OsrsProjectileOrientation;

static inline float osrs_projectile_clamp_progress(float progress) {
    if (progress < 0.0f) return 0.0f;
    if (progress > 1.0f) return 1.0f;
    return progress;
}

static inline float osrs_projectile_height_at_progress(
    float progress,
    float start_height,
    float end_height,
    float arc_height,
    float height_vel,
    float height_accel
) {
    float t = osrs_projectile_clamp_progress(progress);
    if (arc_height > 0.0f) {
        return sinf(t * OSRS_PROJECTILE_ORIENTATION_PI) * arc_height
            + start_height + (end_height - start_height) * t;
    }
    return start_height + height_vel * t + 0.5f * height_accel * t * t;
}

static inline float osrs_projectile_anchor_coord_from_subtile(int subtile_coord) {
    return (float)subtile_coord / 128.0f - 0.5f;
}

static inline float osrs_projectile_subtile_from_anchor_coord(float anchor_coord) {
    return (anchor_coord + 0.5f) * 128.0f;
}

static inline OsrsProjectileOrientation osrs_projectile_orientation_from_step(
    float osrs_dx,
    float osrs_dy,
    float height_delta
) {
    float horizontal = sqrtf(osrs_dx * osrs_dx + osrs_dy * osrs_dy);
    float pitch_horizontal = horizontal > 0.0001f ? horizontal : 0.0001f;

    OsrsProjectileOrientation orientation;
    orientation.yaw = horizontal > 0.0001f ? atan2f(-osrs_dx, osrs_dy) : 0.0f;
    orientation.pitch = atan2f(height_delta, pitch_horizontal);
    return orientation;
}

#endif
