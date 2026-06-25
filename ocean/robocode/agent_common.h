#ifndef ROBOCODE_AGENT_COMMON_H
#define ROBOCODE_AGENT_COMMON_H

#include <float.h>
#include <math.h>

#define RB_PI 3.14159265358979323846f
#define RB_D2R (RB_PI / 180.0f)
#define RB_R2D (180.0f / RB_PI)

static inline float rb_clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float rb_norm_deg(float deg) {
    while (deg > 180.0f) deg -= 360.0f;
    while (deg <= -180.0f) deg += 360.0f;
    return deg;
}

static inline float rb_abs_bearing_deg(float x0, float y0, float x1, float y1) {
    float bearing = atan2f(y1 - y0, x1 - x0) * RB_R2D;
    return bearing < 0.0f ? bearing + 360.0f : bearing;
}

static inline float rb_dist2(float x0, float y0, float x1, float y1) {
    float dx = x1 - x0;
    float dy = y1 - y0;
    return dx*dx + dy*dy;
}

static inline float rb_dist(float x0, float y0, float x1, float y1) {
    return sqrtf(rb_dist2(x0, y0, x1, y1));
}

static inline float rb_bullet_speed(float power) {
    return 20.0f - 3.0f * power;
}

static inline int rb_nearest_agent(Robocode* env, Robot* bot) {
    int target = -1;
    float best = FLT_MAX;
    int total = env->num_agents + env->num_bots;
    int limit = env->num_agents > 0 ? env->num_agents : total;
    for (int i = 0; i < limit; i++) {
        Robot* other = &env->robots[i];
        if (other == bot || other->energy < 0) continue;
        float d2 = rb_dist2(bot->x, bot->y, other->x, other->y);
        if (d2 < best) {
            best = d2;
            target = i;
        }
    }
    return target;
}

static inline void rb_clip_to_field(Robocode* env, Robot* bot) {
    bot->x = rb_clampf(bot->x, 16.0f, (float)env->width - 16.0f);
    bot->y = rb_clampf(bot->y, 16.0f, (float)env->height - 16.0f);
}

static inline float rb_turn_body_to(Robot* bot, float heading) {
    float max_turn = 10.0f - 0.75f * fabsf(bot->v);
    if (max_turn < 0.0f) max_turn = 0.0f;
    return turn(&bot->heading, rb_norm_deg(heading - bot->heading), max_turn, 0.0f);
}

static inline float rb_turn_gun_to(Robot* bot, float heading) {
    return turn(&bot->gun_heading, rb_norm_deg(heading - bot->gun_heading), 20.0f, 0.0f);
}

static inline float rb_turn_radar_to(Robot* bot, float heading, float overshoot) {
    bot->radar_heading_prev = bot->radar_heading;
    float delta = rb_norm_deg(heading - bot->radar_heading);
    if (delta >= 0.0f) delta += overshoot;
    else delta -= overshoot;
    return turn(&bot->radar_heading, delta, 45.0f, 0.0f);
}

static inline int rb_scan_target(Robocode* env, Robot* bot, BotMem* m, int target_idx) {
    Robot* target = &env->robots[target_idx];
    rb_turn_radar_to(bot, rb_abs_bearing_deg(bot->x, bot->y, target->x, target->y), 6.0f);
    if (scan_area(env, bot) == target_idx) {
        m->last_x = target->x;
        m->last_y = target->y;
        m->last_heading = target->heading;
        m->last_v = target->v;
        m->last_energy_seen = target->energy;
        m->last_scan_tick = m->tick;
    }
    return m->last_scan_tick != 0;
}


static inline float rb_drive_to(Robocode* env, Robot* bot, float x, float y) {
    float heading = rb_abs_bearing_deg(bot->x, bot->y, x, y);
    float delta = rb_norm_deg(heading - bot->heading);
    float dir = 1.0f;
    if (cosf(delta * RB_D2R) < 0.0f) {
        delta = rb_norm_deg(delta + 180.0f);
        dir = -1.0f;
    }
    float max_turn = 10.0f - 0.75f * fabsf(bot->v);
    if (max_turn < 0.0f) max_turn = 0.0f;
    turn(&bot->heading, delta, max_turn, 0.0f);
    move(env, bot, dir);
    rb_clip_to_field(env, bot);
    return delta;
}

static inline float rb_linear_aim_deg(Robot* bot, Robot* target, float bullet_speed) {
    float dx = target->x - bot->x;
    float dy = target->y - bot->y;
    float dist = sqrtf(dx*dx + dy*dy);
    float dt = dist / fmaxf(bullet_speed, 0.1f);
    float tvx = cos_deg(target->heading) * target->v;
    float tvy = sin_deg(target->heading) * target->v;
    return rb_abs_bearing_deg(bot->x, bot->y, target->x + tvx*dt, target->y + tvy*dt);
}

static inline float rb_select_firepower(Robot* bot, Robot* target, float dist) {
    float power = fminf(bot->energy / 6.0f, 1300.0f / fmaxf(dist, 1.0f));
    power = fminf(power, target->energy / 3.0f);
    return rb_clampf(power, 0.1f, 3.0f);
}

#endif  // ROBOCODE_AGENT_COMMON_H
