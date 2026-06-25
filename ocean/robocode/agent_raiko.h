#ifndef ROBOCODE_AGENT_RAIKO_H
#define ROBOCODE_AGENT_RAIKO_H

#include "agent_common.h"

#ifndef RAIKO_GF_ZERO
#define RAIKO_GF_ZERO 15
#endif
#ifndef RAIKO_GF_BINS
#define RAIKO_GF_BINS 31
#endif
#ifndef RAIKO_DIST_BINS
#define RAIKO_DIST_BINS 8
#endif
#ifndef RAIKO_WAVES
#define RAIKO_WAVES 16
#endif
#ifndef RAIKO_BEST_DISTANCE
#define RAIKO_BEST_DISTANCE 525.0f
#endif

static inline void raiko_update_waves(BotMem* m, float target_x, float target_y) {
    for (int i = 0; i < RAIKO_WAVES; i++) {
        RBRaikoWave* w = &m->raiko_waves[i];
        if (!w->active) continue;
        w->distance += w->speed;
        float target_dist = rb_dist(w->x, w->y, target_x, target_y);
        if (target_dist <= w->distance + w->speed) {
            float bearing = rb_abs_bearing_deg(w->x, w->y, target_x, target_y);
            float step = w->bearing_step;
            if (fabsf(step) < 0.001f) step = step < 0.0f ? -0.001f : 0.001f;
            float gf = rb_norm_deg(bearing - w->abs_bearing) / step;
            int bin = (int)roundf(gf + RAIKO_GF_ZERO);
            if (bin >= 0 && bin < RAIKO_GF_BINS) {
                m->raiko_guess[w->dist_bin][bin] += 1;
            }
            w->active = 0;
        } else if (w->distance > 1400.0f) {
            w->active = 0;
        }
    }
}

static inline int raiko_best_bin(BotMem* m, int dist_bin) {
    int best = RAIKO_GF_ZERO;
    for (int i = RAIKO_GF_BINS - 1; i >= 0; i--) {
        if (m->raiko_guess[dist_bin][i] > m->raiko_guess[dist_bin][best]) {
            best = i;
        }
    }
    return best;
}

static inline void raiko_add_wave(BotMem* m, Robot* bot, float abs_bearing,
                                  float bearing_step, float speed, int dist_bin) {
    RBRaikoWave* w = &m->raiko_waves[m->raiko_wave_head];
    m->raiko_wave_head = (m->raiko_wave_head + 1) % RAIKO_WAVES;
    w->x = bot->x;
    w->y = bot->y;
    w->abs_bearing = abs_bearing;
    w->bearing_step = bearing_step;
    w->speed = speed;
    w->distance = 0.0f;
    w->dist_bin = dist_bin;
    w->active = 1;
}

static void bot_raiko_step(Robocode* env, int bot_idx, BotMem* m) {
    Robot* bot = &env->robots[bot_idx];
    int target_idx = rb_nearest_agent(env, bot);
    if (target_idx < 0) return;
    if (!rb_scan_target(env, bot, m, target_idx)) return;

    if (!m->raiko_initialized) {
        m->raiko_initialized = 1;
        m->raiko_circle_dir = 1.0f;
        m->raiko_bearing_dir = 1.0f;
        m->raiko_enemy_energy = (float)m->last_energy_seen;
        m->raiko_enemy_firepower = 2.0f;
    }

    float abs_bearing = rb_abs_bearing_deg(bot->x, bot->y, m->last_x, m->last_y);
    float distance = rb_dist(bot->x, bot->y, m->last_x, m->last_y);
    float drop = m->raiko_enemy_energy - (float)m->last_energy_seen;
    if (drop >= 0.1f && drop <= 3.0f) m->raiko_enemy_firepower = drop;
    m->raiko_enemy_energy = (float)m->last_energy_seen;

    raiko_update_waves(m, m->last_x, m->last_y);

    float dist_delta = 0.02f + RB_PI * 0.5f;
    dist_delta += distance > RAIKO_BEST_DISTANCE ? -0.1f : 0.5f;
    float dest_x = bot->x;
    float dest_y = bot->y;
    for (int tries = 0; tries < 160; tries++) {
        dist_delta -= 0.02f;
        float angle = abs_bearing * RB_D2R + m->raiko_circle_dir * dist_delta;
        dest_x = bot->x + 170.0f * cosf(angle);
        dest_y = bot->y + 170.0f * sinf(angle);
        if (dest_x > 18.0f && dest_x < env->width - 18.0f &&
                dest_y > 18.0f && dest_y < env->height - 18.0f) {
            break;
        }
    }

    float theta = 0.5952f * rb_bullet_speed(m->raiko_enemy_firepower) / fmaxf(distance, 1.0f);
    if ((rand_unit(env) > powf(fmaxf(theta, 0.001f), fmaxf(theta, 0.001f))) ||
            dist_delta < RB_PI / 5.0f ||
            (dist_delta < RB_PI / 3.5f && distance < 400.0f)) {
        m->raiko_circle_dir = -m->raiko_circle_dir;
        m->raiko_last_reverse_tick = m->tick;
    }
    rb_drive_to(env, bot, dest_x, dest_y);

    float tx = m->last_x - bot->x;
    float ty = m->last_y - bot->y;
    float inv_dist = 1.0f / fmaxf(distance, 1.0f);
    float ux = tx * inv_dist;
    float uy = ty * inv_dist;
    float tvx = cos_deg(m->last_heading) * m->last_v;
    float tvy = sin_deg(m->last_heading) * m->last_v;
    float enemy_lat_vel = -tvx * uy + tvy * ux;
    if (fabsf(enemy_lat_vel) > 0.01f) {
        m->raiko_bearing_dir = enemy_lat_vel > 0.0f ? 1.0f : -1.0f;
    }

    int dist_bin = (int)(distance / 140.0f);
    if (dist_bin < 0) dist_bin = 0;
    if (dist_bin >= RAIKO_DIST_BINS) dist_bin = RAIKO_DIST_BINS - 1;
    float bullet_power = dist_bin == 0 ? 3.0f : 2.0f;
    bullet_power = fminf(bullet_power, bot->energy / 4.0f);
    bullet_power = fminf(bullet_power, (float)m->last_energy_seen / 4.0f);
    bullet_power = rb_clampf(bullet_power, 0.1f, 3.0f);
    float speed = rb_bullet_speed(bullet_power);
    float max_escape = asinf(fminf(8.0f / speed, 1.0f)) * RB_R2D;
    float bearing_step = m->raiko_bearing_dir * max_escape / (float)RAIKO_GF_ZERO;
    int best_bin = raiko_best_bin(m, dist_bin);
    float aim = abs_bearing + (float)(best_bin - RAIKO_GF_ZERO) * bearing_step;
    float gun_delta = rb_turn_gun_to(bot, aim);

    if (fabsf(gun_delta) < 2.0f && bot->gun_heat <= 0.0f && bot->energy > 1) {
        fire(env, bot, bot_idx, bullet_power);
        raiko_add_wave(m, bot, abs_bearing, bearing_step, speed, dist_bin);
    }
    rb_turn_radar_to(bot, abs_bearing, fabsf(rb_norm_deg(abs_bearing - bot->radar_heading)));
}

#endif  // ROBOCODE_AGENT_RAIKO_H
