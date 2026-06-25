#ifndef ROBOCODE_AGENT_HAWK_ON_FIRE_H
#define ROBOCODE_AGENT_HAWK_ON_FIRE_H

#include "agent_common.h"

static inline float hof_position_risk(BotMem* m, Robot* bot,
                                      float x, float y, float add_last) {
    float risk = 0.0f;
    float last_d2 = rb_dist2(x, y, m->dest_last_x, m->dest_last_y);
    if (last_d2 > 1.0f) risk += add_last * 0.08f / last_d2;

    float d2 = rb_dist2(x, y, m->last_x, m->last_y);
    if (d2 <= 1.0f) d2 = 1.0f;
    float move_angle = rb_abs_bearing_deg(x, y, bot->x, bot->y) * RB_D2R;
    float enemy_angle = rb_abs_bearing_deg(x, y, m->last_x, m->last_y) * RB_D2R;
    float my_energy = fmaxf((float)bot->energy, 1.0f);
    float energy_weight = fminf((float)m->last_energy_seen / my_energy, 2.0f);
    risk += energy_weight * (1.0f + fabsf(cosf(move_angle - enemy_angle))) / d2;
    return risk;
}

static void bot_hawk_on_fire_step(Robocode* env, int bot_idx, BotMem* m) {
    Robot* bot = &env->robots[bot_idx];
    int target_idx = rb_nearest_agent(env, bot);
    if (target_idx < 0) return;
    if (!rb_scan_target(env, bot, m, target_idx)) return;

    if (!m->dest_initialized) {
        m->dest_x = bot->x;
        m->dest_y = bot->y;
        m->dest_last_x = bot->x;
        m->dest_last_y = bot->y;
        m->dest_initialized = 1;
    }

    float distance = rb_dist(bot->x, bot->y, m->last_x, m->last_y);
    float aim = rb_abs_bearing_deg(bot->x, bot->y, m->last_x, m->last_y);
    float gun_delta = rb_turn_gun_to(bot, aim);
    if (fabsf(gun_delta) < 2.0f && bot->gun_heat <= 0.0f &&
            bot->energy > 1 && m->last_energy_seen > 0) {
        float power = fminf(bot->energy / 6.0f, 1300.0f / fmaxf(distance, 1.0f));
        power = fminf(power, (float)m->last_energy_seen / 3.0f);
        fire(env, bot, bot_idx, rb_clampf(power, 0.1f, 3.0f));
    }

    if (rb_dist(bot->x, bot->y, m->dest_x, m->dest_y) < 15.0f) {
        float add_last = 1.0f - roundf(rand_unit(env));
        float best_x = m->dest_x;
        float best_y = m->dest_y;
        float best_risk = hof_position_risk(m, bot, best_x, best_y, add_last);

        for (int i = 0; i < 200; i++) {
            float radius = fminf(distance * 0.8f, 100.0f + 200.0f * rand_unit(env));
            float angle = 360.0f * rand_unit(env);
            float x = bot->x + radius * cos_deg(angle);
            float y = bot->y + radius * sin_deg(angle);
            if (x < 30.0f || x > env->width - 30.0f ||
                    y < 30.0f || y > env->height - 30.0f) {
                continue;
            }
            float risk = hof_position_risk(m, bot, x, y, add_last);
            if (risk < best_risk) {
                best_risk = risk;
                best_x = x;
                best_y = y;
            }
        }
        m->dest_last_x = bot->x;
        m->dest_last_y = bot->y;
        m->dest_x = best_x;
        m->dest_y = best_y;
    }

    rb_drive_to(env, bot, m->dest_x, m->dest_y);
}

#endif  // ROBOCODE_AGENT_HAWK_ON_FIRE_H
