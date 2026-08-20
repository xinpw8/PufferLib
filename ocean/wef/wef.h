// Weakly electric fish env. Port of KempnerInstitute/wef biophysics.
// Positions in cm; field measure converts to m and returns V/m.

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

// 5c trainer macros (no binding.c). Float obs; pufferl converts on upload.
#define ACT_SIZES {1, 1, 1, 1}
#define NUM_ATNS 4

// Constants
#define PI_F 3.14159265358979323846f
#define CM_TO_M 0.01f
#define K_COULOMB 8.99e9f
#define EPSILON_0 8.854e-12f
#define FIELD_EPS_M 1e-5f
#define SENSOR_EPS 1e-25f

// Values from upstream cfg.py and Table 1 of the accompanying publication
#define SIMULATION_HZ 83.0f
#define BODY_RADIUS_CM 1.0f
#define FOOD_RADIUS_CM 0.25f
#define CONDUCTOR_CONTRAST -0.5f
#define EOD_CHARGE_C 1.11e-15f
#define EOD_POLE_OFFSET_CM 0.5f
#define INTRINSIC_MOMENT_C_M 1.11e-23f
#define FOOD_INTRINSIC_MOMENT_C_M 1.11e-24f

// Max speeds: 35 cm/s * 2, 3.6 rad/s * 4; then / SIMULATION_HZ.
#define MAX_LINEAR_VELOCITY_CM_S 70.0f
#define MAX_ANGULAR_VELOCITY_RAD_S 14.4f
#define SIZE_SPEED_EXPONENT 1.0f

// Sensors
#define NUM_MORMYROMASTS 36
#define NUM_AMPULLARY 24
#define NUM_KNOLLEN 12
#define MAX_AGENTS 4

#define MAX_FOOD 64
#define OBS_SIZE 110
#define ACTION_SIZE 4
#define EATING_RADIUS_CM 2.0f
#define BITING_RADIUS_CM 3.0f
#define EATING_ANGLE (PI_F / 4.0f)
#define MAX_PATCHES 90

// Reward coefficients
#define EAT_REWARD 1.0f
#define PROXIMITY_SHAPING_REWARD 0.1f
#define BITTEN_REWARD -0.5f
#define BITE_REWARD -0.0001f
#define COLLISION_REWARD -0.05f
#define EFFORT_OVER_REWARD -0.01f
#define PENALIZE_EFFORT_OVER_FRAC 0.5f

#define TRACE_LENGTH 65

#define EAT_COOLDOWN_STEPS 3
#define BITE_COOLDOWN_STEPS 5

#define AMPULLARY_MIN_VM 2e-10f
#define AMPULLARY_MAX_VM 2e-8f
#define MORMYROMAST_MIN_VM 5e-8f
#define MORMYROMAST_MAX_VM 5e-2f
#define KNOLLEN_MIN_VM 2e-7f

// Sensor interaction ranges in cm.
// Food/prey and conspecifics (other agents) have separate cutoffs for morm & amp.
#define MORM_FOOD_RANGE_CM 5.0f       // prey ≤ 5 cm
#define MORM_AGENT_RANGE_CM 10.0f     // agents ≤ 10 cm
#define AMP_FOOD_RANGE_CM 4.0f        // prey ≤ 4 cm
#define AMP_AGENT_RANGE_CM 8.0f       // conspecifics ≤ 8 cm
#define KNOLLEN_AGENT_RANGE_CM 100.0f // conspecific EOD ≤ 100 cm

// Arena wall indices for first-order image charges.
#define WALL_LEFT 0
#define WALL_RIGHT 1
#define WALL_BOTTOM 2
#define WALL_TOP 3
#define NUM_WALLS 4

// Render palette
#define WEF_COLOR_BG            ((Color){6, 24, 24, 255})
#define WEF_COLOR_MIDGRAY       ((Color){120, 120, 120, 255})
#define WEF_COLOR_PANEL         ((Color){10, 10, 10, 200})
#define WEF_COLOR_TEXT          ((Color){240, 240, 240, 255})
#define WEF_COLOR_FISH          ((Color){180, 150, 230, 255})
#define WEF_COLOR_FISH_PULSE    ((Color){180, 150, 230, 140})
#define WEF_COLOR_SENSOR        ((Color){184, 164, 224, 200})
#define WEF_COLOR_FOOD          ((Color){0x7A, 0xEF, 0x9A, 200})  // lighter green
#define WEF_COLOR_EOD_POS       ((Color){220, 60, 50, 255})
#define WEF_COLOR_EOD_NEG       ((Color){60, 120, 255, 255})

// Field arrows: yellow (min) → red (max) log gradient over WEF |E| in V/cm.
#define WEF_FIELD_LOG_LO        (-7.0f)   // 1e-7 V/cm
#define WEF_FIELD_LOG_HI        (-3.0f)   // 1e-3 V/cm
#define WEF_COLOR_FIELD_WEAK    ((Color){255, 255, 40, 200})   // yellow (min |E|)
#define WEF_COLOR_FIELD_STRONG  ((Color){230, 40, 30, 255})    // red (max |E|)

typedef struct { float x, y; } Vec2;

// One electroreceptor site in the fish body frame (AoS).
typedef struct Sensor {
    Vec2 p;  // position on body
    Vec2 n;  // outward normal
} Sensor;

// Per-fish dynamics / EOD / dipoles. Sensor geometry is shared (see g_*).
typedef struct FishAgent {
    Vec2 pos;
    float orientation;
    float max_linear_velocity;
    float max_angular_velocity;
    Vec2 disp_ego;
    float size;
    int bite_cooldown;
    int eat_cooldown;
    bool emits_eod;
    bool bite_action;
    bool was_bitten;
    bool has_previous_food_distance;
    float previous_food_distance;
    float last_action[ACTION_SIZE];
    Vec2 eod_pos[2];
    float eod_charge[2];
    Vec2 intrinsic_moment;
    Vec2 induced_moment;
} FishAgent;

// Shared sensor layouts (body radius fixed → identical for every fish)
Sensor g_morm[NUM_MORMYROMASTS];
Sensor g_amp[NUM_AMPULLARY];
Sensor g_knollen[NUM_KNOLLEN];

float clamp(float value, float minimum, float maximum) {
    return fminf(maximum, fmaxf(minimum, value));
}

float wrap_angle(float angle) {
    return atan2f(sinf(angle), cosf(angle));
}

// Log-scale electroreceptor encoding used by mormyromast + ampullary.
float encode_log_sensor(float reading, float lo, float hi) {
    if (reading == 0.0f) {
        return 0.0f;
    }
    float sign = reading < 0.0f ? -1.0f : 1.0f;
    float mag = fmaxf(clamp(fabsf(reading), lo, hi), SENSOR_EPS);
    float nrm = (log10f(mag) - log10f(lo)) / (log10f(hi) - log10f(lo));
    return sign * clamp(nrm, 0.0f, 1.0f);
}

// Sensor local frame → world pose of fish
Sensor sensor_world(const Sensor* s, const FishAgent* fish) {
    float c = cosf(fish->orientation);
    float sn = sinf(fish->orientation);
    return (Sensor){
        {
            c * s->p.x - sn * s->p.y + fish->pos.x,
            sn * s->p.x + c * s->p.y + fish->pos.y,
        },
        {
            c * s->n.x - sn * s->n.y,
            sn * s->n.x + c * s->n.y,
        },
    };
}

bool in_forward_cone(const FishAgent* fish, Vec2 target,
        float radius_cm, float cone) {
    float dx = target.x - fish->pos.x;
    float dy = target.y - fish->pos.y;
    if (dx * dx + dy * dy >= radius_cm * radius_cm) {
        return false;
    }
    float bearing = atan2f(dy, dx);
    return fabsf(wrap_angle(bearing - fish->orientation)) <= cone * 0.5f;
}

// Induced dipole scale for a conducting sphere of radius_cm (contrast κ).
float conductor_scale(float radius_cm) {
    float r = radius_cm * CM_TO_M;
    return 3.0f * EPSILON_0 * CONDUCTOR_CONTRAST *
        (4.0f / 3.0f) * PI_F * r * r * r;
}

// Caps: 2 EOD poles/agent; agent+food dipoles.
enum { WEF_MAX_MONO = 2 * MAX_AGENTS, WEF_MAX_DIP = MAX_AGENTS + MAX_FOOD };

// Electric sources for measure_field: position in meters, moments/charges SI.
typedef struct { Vec2 p; float q; } Mono;
typedef struct { Vec2 p, m; } Dipole;

Vec2 to_m(Vec2 p_cm) {
    return (Vec2){p_cm.x * CM_TO_M, p_cm.y * CM_TO_M};
}

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float food_eaten_mean;
    float eod_rate;
    float collisions_fish;
    float bites;
    float food_per_fish_area;
    float n;
};

typedef struct Trace {
    Vec2 pos[TRACE_LENGTH];
    int index;
    int count;
} Trace;

typedef struct Client {
    int window_width;
    int window_height;
    int margin;
    bool show_field;
    bool show_sensors;
    Trace traces[MAX_AGENTS];
} Client;

typedef struct FishFood {
    Vec2 pos;
    float orientation;
    bool active;
    Vec2 intrinsic_moment;
    Vec2 induced_moment;
} FishFood;

typedef enum FoodDistribution {
    FOOD_UNIFORM,
    FOOD_PATCHY,
    FOOD_RANDOM,
} FoodDistribution;

struct Env {
    Log log;
    Agent agents[MAX_AGENTS];
    int tag;
    int boundary_reached;
    int num_agents;
    int tick;
    int episode_length;
    unsigned int rng;
    float arena_size_x;
    float arena_size_y;
    float min_arena_size_x;
    float min_arena_size_y;
    float max_arena_size_x;
    float max_arena_size_y;
    float electric_field_radius_cm;
    float reflection_wall_range_cm;
    FoodDistribution food_distribution;
    int configured_num_food;
    float patch_radius_cm;
    float patch_radius_std_cm;
    float patch_density;
    FishAgent fish[MAX_AGENTS];
    FishFood food[MAX_FOOD];
    int num_food;
    int food_eaten;
    int eod_agent_steps;
    int collisions_fish;
    int bites;
    float food_per_fish_area;
    float episode_return;
    float amp_intrinsic_baseline[NUM_AMPULLARY];
    Client* client;
};
typedef Env Wef;

float random_uniform(Wef* env, float low, float high) {
    float unit = (float)rand_r(&env->rng) / (float)RAND_MAX;
    return low + (high - low) * unit;
}

// Probe in cm. Sources (Mono/Dipole.p) already meters.
// Monos are always agent EODs → agent_range. Dipoles: first n_agent_dips are
// agents, the rest food → agent_range / food_range (paper sensor cutoffs).
// wall_range_cm=0 → no image charges.
Vec2 measure_field(Env* env, Vec2 probe_cm, const Mono* mono, int n_mono,
    const Dipole* dip, int n_dip, int n_agent_dips,
    float agent_range_cm, float food_range_cm, float wall_range_cm) {
    float pmx = probe_cm.x * CM_TO_M;
    float pmy = probe_cm.y * CM_TO_M;
    float agent_r = agent_range_cm * CM_TO_M;
    float food_r = food_range_cm * CM_TO_M;
    float agent_range2 = agent_r * agent_r;
    float food_range2 = food_r * food_r;
    float eps_m = FIELD_EPS_M;
    float field_x = 0.0f;
    float field_y = 0.0f;

    // Stage sources only when wall images are possible (skip for induce/knollen).
    int want_walls = wall_range_cm > 0.0f;
    int near_walls[NUM_WALLS];
    int num_near_walls = 0;
    float arena_mx = 0.0f;
    float arena_my = 0.0f;
    if (want_walls) {
        arena_mx = env->arena_size_x * CM_TO_M;
        arena_my = env->arena_size_y * CM_TO_M;
        float wall_range_m = wall_range_cm * CM_TO_M;
        float wall_dist[NUM_WALLS] = {
            pmx, arena_mx - pmx, pmy, arena_my - pmy
        };
        for (int wall = 0; wall < NUM_WALLS; wall++) {
            if (wall_dist[wall] <= wall_range_m) {
                near_walls[num_near_walls++] = wall;
            }
        }
        if (num_near_walls == 0) {
            want_walls = 0;
        }
    }

    Mono mono_in[WEF_MAX_MONO];
    Dipole dip_in[WEF_MAX_DIP];
    int n_mono_in = 0;
    int n_dip_in = 0;

    for (int i = 0; i < n_mono; i++) {
        float sx = mono[i].p.x;
        float sy = mono[i].p.y;
        float dx = pmx - sx;
        float dy = pmy - sy;
        // EOD monos are always conspecific/agent sources.
        if (dx * dx + dy * dy > agent_range2) {
            continue;
        }
        if (want_walls) {
            mono_in[n_mono_in++] = mono[i];
        }
        float dist = sqrtf(dx * dx + dy * dy) + eps_m;
        float inv_d = 1.0f / dist;
        float w = K_COULOMB * mono[i].q * inv_d * inv_d * inv_d;
        field_x += dx * w;
        field_y += dy * w;
    }
    for (int i = 0; i < n_dip; i++) {
        float sx = dip[i].p.x;
        float sy = dip[i].p.y;
        float range2 = (i < n_agent_dips) ? agent_range2 : food_range2;
        float dx = pmx - sx;
        float dy = pmy - sy;
        if (dx * dx + dy * dy > range2) {
            continue;
        }
        if (want_walls) {
            dip_in[n_dip_in++] = dip[i];
        }
        float dist = sqrtf(dx * dx + dy * dy) + eps_m;
        float inv_d2 = 1.0f / (dist * dist);
        float inv_d3 = inv_d2 / dist;
        float mdot = dip[i].m.x * dx + dip[i].m.y * dy;
        float k = K_COULOMB * inv_d3;
        float t = 3.0f * mdot * inv_d2;
        field_x += k * (t * dx - dip[i].m.x);
        field_y += k * (t * dy - dip[i].m.y);
    }

    for (int w = 0; w < num_near_walls; w++) {
        int wall = near_walls[w];
        for (int k = 0; k < n_mono_in; k++) {
            float sx = mono_in[k].p.x;
            float sy = mono_in[k].p.y;
            if (wall <= WALL_RIGHT) {
                sx = wall == WALL_LEFT ? -sx : 2.0f * arena_mx - sx;
            } else {
                sy = wall == WALL_BOTTOM ? -sy : 2.0f * arena_my - sy;
            }
            float dx = pmx - sx;
            float dy = pmy - sy;
            float dist = sqrtf(dx * dx + dy * dy) + eps_m;
            float inv_d = 1.0f / dist;
            float wt = K_COULOMB * mono_in[k].q * inv_d * inv_d * inv_d;
            field_x += dx * wt;
            field_y += dy * wt;
        }
        for (int k = 0; k < n_dip_in; k++) {
            float sx = dip_in[k].p.x;
            float sy = dip_in[k].p.y;
            if (wall <= WALL_RIGHT) {
                sx = wall == WALL_LEFT ? -sx : 2.0f * arena_mx - sx;
            } else {
                sy = wall == WALL_BOTTOM ? -sy : 2.0f * arena_my - sy;
            }
            float dx = pmx - sx;
            float dy = pmy - sy;
            float dist = sqrtf(dx * dx + dy * dy) + eps_m;
            float inv_d2 = 1.0f / (dist * dist);
            float inv_d3 = inv_d2 / dist;
            float mx = dip_in[k].m.x;
            float my = dip_in[k].m.y;
            float mdot = mx * dx + my * dy;
            float kcoef = K_COULOMB * inv_d3;
            float t = 3.0f * mdot * inv_d2;
            field_x += kcoef * (t * dx - mx);
            field_y += kcoef * (t * dy - my);
        }
    }
    return (Vec2){field_x, field_y};
}
void compute_observations(Wef* env) {
    // Build EOD poles + induced/intrinsic moments, then pack obs.
    Mono eod[WEF_MAX_MONO];
    int n_eod = 0;
    float body_scale = conductor_scale(BODY_RADIUS_CM);
    float food_scale = conductor_scale(FOOD_RADIUS_CM);
    float max_moment = EOD_CHARGE_C * BODY_RADIUS_CM;
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->fish[i];
        float c = cosf(agent->orientation);
        float s = sinf(agent->orientation);
        float q = agent->emits_eod ? EOD_CHARGE_C : 0.0f;
        agent->eod_pos[0] = (Vec2){
            c * EOD_POLE_OFFSET_CM + agent->pos.x,
            s * EOD_POLE_OFFSET_CM + agent->pos.y,
        };
        agent->eod_pos[1] = (Vec2){
            -c * EOD_POLE_OFFSET_CM + agent->pos.x,
            -s * EOD_POLE_OFFSET_CM + agent->pos.y,
        };
        agent->eod_charge[0] = q;
        agent->eod_charge[1] = -q;
        agent->intrinsic_moment = (Vec2){c * INTRINSIC_MOMENT_C_M, s * INTRINSIC_MOMENT_C_M};
        eod[n_eod++] = (Mono){to_m(agent->eod_pos[0]), q};
        eod[n_eod++] = (Mono){to_m(agent->eod_pos[1]), -q};
    }
    // EOD induces dipoles on non-emitting bodies; food gets intrinsic + induced.
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->fish[i];
        float moment_x = 0.0f;
        float moment_y = 0.0f;
        if (!agent->emits_eod) {
            // Induced by nearby EODs (conspecific EOD → body, morm agent range).
            Vec2 f = measure_field(
                env, agent->pos, eod, n_eod, NULL, 0, 0,
                MORM_AGENT_RANGE_CM, MORM_FOOD_RANGE_CM, 0.0f
            );
            moment_x = f.x * body_scale;
            moment_y = f.y * body_scale;
            float mag = sqrtf(moment_x * moment_x + moment_y * moment_y);
            if (mag > max_moment) {
                float s = max_moment / mag;
                moment_x *= s;
                moment_y *= s;
            }
        }
        agent->induced_moment = (Vec2){moment_x, moment_y};
    }
    for (int i = 0; i < env->num_food; i++) {
        if (!env->food[i].active) {
            env->food[i].intrinsic_moment = (Vec2){0};
            env->food[i].induced_moment = (Vec2){0};
            continue;
        }
        float fc = cosf(env->food[i].orientation);
        float fs = sinf(env->food[i].orientation);
        env->food[i].intrinsic_moment = (Vec2){
            -fs * FOOD_INTRINSIC_MOMENT_C_M,
            fc * FOOD_INTRINSIC_MOMENT_C_M,
        };
        Vec2 f = measure_field(
            env, env->food[i].pos, eod, n_eod, NULL, 0, 0,
            MORM_AGENT_RANGE_CM, MORM_FOOD_RANGE_CM, 0.0f
        );
        env->food[i].induced_moment = (Vec2){f.x * food_scale, f.y * food_scale};
    }
    // Induced + intrinsic dipoles as AoS for measure_field
    Dipole induced[MAX_AGENTS + MAX_FOOD];
    Dipole intrinsic[MAX_AGENTS + MAX_FOOD];
    int n_induced = 0;
    int n_intrinsic = 0;
    for (int a = 0; a < env->num_agents; a++) {
        induced[n_induced++] = (Dipole){
            to_m(env->fish[a].pos), env->fish[a].induced_moment
        };
        intrinsic[n_intrinsic++] = (Dipole){
            to_m(env->fish[a].pos), env->fish[a].intrinsic_moment
        };
    }
    for (int f = 0; f < env->num_food; f++) {
        if (!env->food[f].active) {
            continue;
        }
        induced[n_induced++] = (Dipole){
            to_m(env->food[f].pos), env->food[f].induced_moment
        };
        intrinsic[n_intrinsic++] = (Dipole){
            to_m(env->food[f].pos), env->food[f].intrinsic_moment
        };
    }
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->fish[i];
        obs_t* obs = env->agents[i].observations;
        int obs_idx = 0;

        bool cons_eod = false;
        for (int other = 0; other < env->num_agents; other++) {
            if (other != i && env->fish[other].emits_eod) {
                cons_eod = true;
                break;
            }
        }
        // Mormyromasts: induced field.
        for (int sensor_idx = 0; sensor_idx < NUM_MORMYROMASTS; sensor_idx++) {
            Sensor w = sensor_world(&g_morm[sensor_idx], agent);
            Vec2 f = measure_field(
                env, w.p, NULL, 0, induced, n_induced, env->num_agents,
                MORM_AGENT_RANGE_CM, MORM_FOOD_RANGE_CM,
                env->reflection_wall_range_cm
            );
            float reading = f.x * w.n.x + f.y * w.n.y;
            if (!agent->emits_eod) {
                reading *= 100.0f;
            }
            reading *= random_uniform(env, 0.95f, 1.05f);
            obs[obs_idx++] = encode_log_sensor(
                reading, MORMYROMAST_MIN_VM, MORMYROMAST_MAX_VM
            );
        }
        // Ampullary: intrinsic.
        for (int sensor_idx = 0; sensor_idx < NUM_AMPULLARY; sensor_idx++) {
            Sensor w = sensor_world(&g_amp[sensor_idx], agent);
            Vec2 f = measure_field(
                env, w.p, NULL, 0, intrinsic, n_intrinsic, env->num_agents,
                AMP_AGENT_RANGE_CM, AMP_FOOD_RANGE_CM,
                env->reflection_wall_range_cm
            );
            float noise = cons_eod ? 0.5f : 0.05f;
            float reading = (f.x * w.n.x + f.y * w.n.y -
                env->amp_intrinsic_baseline[sensor_idx]) *
                random_uniform(env, 1.0f - noise, 1.0f + noise);
            obs[obs_idx++] = encode_log_sensor(
                reading, AMPULLARY_MIN_VM, AMPULLARY_MAX_VM
            );
        }
        // Knollenorgans: conspecific EOD only.
        int metadata_start = NUM_MORMYROMASTS + NUM_AMPULLARY + NUM_KNOLLEN * (MAX_AGENTS - 1);
        int cons_slot = 0;
        for (int other = 0; other < MAX_AGENTS; other++) {
            if (other == i) {
                continue;
            }
            bool valid = other < env->num_agents && env->fish[other].emits_eod;
            for (int sensor_idx = 0; sensor_idx < NUM_KNOLLEN; sensor_idx++) {
                float value = 0.0f;
                if (valid) {
                    Sensor w = sensor_world(&g_knollen[sensor_idx], agent);
                    Mono eod[2] = {
                        {
                            to_m(env->fish[other].eod_pos[0]),
                            env->fish[other].eod_charge[0]
                        },
                        {
                            to_m(env->fish[other].eod_pos[1]),
                            env->fish[other].eod_charge[1]
                        },
                    };
                    Vec2 f = measure_field(
                        env, w.p, eod, 2, NULL, 0, 0,
                        KNOLLEN_AGENT_RANGE_CM, 0.0f, 0.0f
                    );
                    float raw = (f.x * w.n.x + f.y * w.n.y) *
                        random_uniform(env, 0.95f, 1.05f);
                    if (fabsf(raw) > KNOLLEN_MIN_VM) {
                        value = raw < 0.0f ? -1.0f : 1.0f;
                    }
                }
                obs[obs_idx++] = value;
            }
            bool detected = false;
            int block_start = obs_idx - NUM_KNOLLEN;
            for (int k = 0; k < NUM_KNOLLEN; k++) {
                if (obs[block_start + k] != 0.0f) {
                    detected = true;
                    break;
                }
            }
            float metadata = -1.0f;
            if (valid && detected) {
                metadata = agent->size - env->fish[other].size;
                metadata += random_uniform(env, -0.05f, 0.05f);
                metadata = clamp(metadata, -1.0f, 1.0f);
            }
            obs[metadata_start + cons_slot] = metadata;
            cons_slot++;
        }
        obs_idx = metadata_start + MAX_AGENTS - 1;
        for (int action_idx = 0; action_idx < ACTION_SIZE; action_idx++) {
            obs[obs_idx++] = agent->last_action[action_idx];
        }
        obs[obs_idx++] = 0.0f;
        obs[obs_idx++] = agent->was_bitten ? 1.0f : 0.0f;
        obs[obs_idx++] = agent->size;
        obs[obs_idx++] = (float)agent->bite_cooldown / (float)BITE_COOLDOWN_STEPS;
        obs[obs_idx++] = clamp(
            agent->disp_ego.x / agent->max_linear_velocity, -1.0f, 1.0f);
        obs[obs_idx++] = clamp(
            agent->disp_ego.y / agent->max_linear_velocity, -1.0f, 1.0f);
        obs[obs_idx++] = (float)agent->eat_cooldown / (float)EAT_COOLDOWN_STEPS;
    }
}

void puf_reset(Wef* env) {
    // Sample arena size from configured min/max
    env->arena_size_x = random_uniform(env, env->min_arena_size_x, env->max_arena_size_x);
    env->arena_size_y = random_uniform(env, env->min_arena_size_y, env->max_arena_size_y);
    
    // Select food distribution mode; if random, choose uniform or patchy
    FoodDistribution mode = env->food_distribution;
    if (mode == FOOD_RANDOM) {
        mode = (FoodDistribution)(rand_r(&env->rng) % 2);
    }
    env->tick = 0;
    env->food_eaten = 0;
    env->eod_agent_steps = 0;
    env->collisions_fish = 0;
    env->bites = 0;
    env->episode_return = 0.0f;

    // Spawn fish without body overlap; place sensors in local frame
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent agent = {0};
        agent.size = random_uniform(env, 0.0f, 1.0f);
        Vec2 pos = {0};
        for (int attempts = 0; attempts < 1000; attempts++) {
            pos = (Vec2){
                random_uniform(env, 3.0f, env->arena_size_x - 3.0f),
                random_uniform(env, 3.0f, env->arena_size_y - 3.0f),
            };
            bool overlap = false;
            for (int j = 0; j < i; j++) {
                float dx = pos.x - env->fish[j].pos.x;
                float dy = pos.y - env->fish[j].pos.y;
                float diam = 2.0f * BODY_RADIUS_CM;
                if (dx * dx + dy * dy < diam * diam) {
                    overlap = true;
                    break;
                }
            }
            if (!overlap) {
                break;
            }
        }
        agent.pos = pos;
        agent.orientation = random_uniform(env, -PI_F, PI_F);
        // Agent's size determines its max linear/angular velocity (larger fish are faster)
        float size_mult = powf(1.0f + agent.size, SIZE_SPEED_EXPONENT);
        agent.max_linear_velocity = (MAX_LINEAR_VELOCITY_CM_S / SIMULATION_HZ) * size_mult;
        agent.max_angular_velocity = (MAX_ANGULAR_VELOCITY_RAD_S / SIMULATION_HZ) * size_mult;
        agent.emits_eod = true;
        env->fish[i] = agent;
    }

    // Distribute food
    env->num_food = env->configured_num_food;
    if (mode == FOOD_UNIFORM) {
        for (int i = 0; i < env->num_food; i++) {
            env->food[i] = (FishFood){
                .pos = {
                    random_uniform(env, 0.0f, env->arena_size_x),
                    random_uniform(env, 0.0f, env->arena_size_y),
                },
                .orientation = random_uniform(env, 0.0f, 2.0f * PI_F),
                .active = true,
            };
        }
    } else {
        // Patchy: random circular patches, food sampled uniformly in a patch disk
        float centers_x[MAX_PATCHES];
        float centers_y[MAX_PATCHES];
        float radii[MAX_PATCHES];
        float max_radius =
            fminf(env->arena_size_x, env->arena_size_y) * 0.5f;
        int num_patches = (int)clamp(
            ceilf(env->patch_density *
                env->arena_size_x * env->arena_size_y),
            1, MAX_PATCHES
        );
        for (int p = 0; p < num_patches; p++) {
            centers_x[p] = random_uniform(env, 0.0f, env->arena_size_x);
            centers_y[p] = random_uniform(env, 0.0f, env->arena_size_y);
            radii[p] = clamp(
                env->patch_radius_cm + random_uniform(
                    env, -env->patch_radius_std_cm, env->patch_radius_std_cm
                ),
                1.0f, max_radius
            );
        }
        for (int i = 0; i < env->num_food; i++) {
            int p = (int)(rand_r(&env->rng) % (unsigned)num_patches);
            float angle = random_uniform(env, 0.0f, 2.0f * PI_F);
            float r = radii[p] * sqrtf(random_uniform(env, 0.0f, 1.0f));
            env->food[i] = (FishFood){
                .pos = {
                    clamp(centers_x[p] + r * cosf(angle), 0.0f, env->arena_size_x),
                    clamp(centers_y[p] + r * sinf(angle), 0.0f, env->arena_size_y),
                },
                .orientation = random_uniform(env, 0.0f, 2.0f * PI_F),
                .active = true,
            };
        }
    }
    // Ampullary baseline: unit intrinsic dipole at arena center
    Vec2 center = {env->arena_size_x * 0.5f, env->arena_size_y * 0.5f};
    Dipole baseline_dip[1] = {{to_m(center), (Vec2){INTRINSIC_MOMENT_C_M, 0.0f}}};
    for (int i = 0; i < NUM_AMPULLARY; i++) {
        Vec2 probe = {
            center.x + g_amp[i].p.x,
            center.y + g_amp[i].p.y,
        };
        Vec2 f = measure_field(
            env, probe, NULL, 0, baseline_dip, 1, 1,
            AMP_AGENT_RANGE_CM, AMP_FOOD_RANGE_CM,
            env->reflection_wall_range_cm
        );
        env->amp_intrinsic_baseline[i] = f.x * g_amp[i].n.x + f.y * g_amp[i].n.y;
    }
    
    // Record food density per fish area
    float arena_area = env->arena_size_x * env->arena_size_y;
    env->food_per_fish_area = (float)env->num_food / (arena_area * (float)env->num_agents);
    
    compute_observations(env);
}

void puf_step(Wef* env) {
    env->tick++;
    for (int i = 0; i < env->num_agents; i++) {
        env->fish[i].was_bitten = false;
        env->agents[i].rewards[0] = 0.0f;
        env->agents[i].terminals[0] = 0.0f;
    }

    // Actions, eat, first-order motion, collisions (per fish)
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->fish[i];
        float* raw_action = env->agents[i].actions;
        float move = 1.0f / (1.0f + expf(-(float)raw_action[0]));
        float turn = tanhf((float)raw_action[1]);
        agent->emits_eod = raw_action[2] > 0.0f;
        agent->bite_action = raw_action[3] > 0.0f && agent->bite_cooldown <= 0;
        if (agent->bite_action) {
            agent->bite_cooldown = BITE_COOLDOWN_STEPS;
        }
        agent->last_action[0] = move;
        agent->last_action[1] = turn;
        agent->last_action[2] = agent->emits_eod ? 1.0f : 0.0f;
        agent->last_action[3] = agent->bite_action ? 1.0f : 0.0f;
        env->eod_agent_steps += agent->emits_eod ? 1 : 0;

        // Effort penalty
        if (PENALIZE_EFFORT_OVER_FRAC < 1.0f) {
            float move_over = fmaxf(0.0f, fabsf(move) - PENALIZE_EFFORT_OVER_FRAC);
            float turn_over = fmaxf(0.0f, fabsf(turn) - PENALIZE_EFFORT_OVER_FRAC);
            if (move_over > 0.0f || turn_over > 0.0f) {
                env->agents[i].rewards[0] += EFFORT_OVER_REWARD * (move_over + turn_over);
            }
        }

        // Eat first active pellet in forward 45° cone within 2 cm
        if (!agent->bite_action && agent->eat_cooldown <= 0) {
            for (int f = 0; f < env->num_food; f++) {
                if (!env->food[f].active) {
                    continue;
                }
                if (!in_forward_cone(agent, env->food[f].pos, EATING_RADIUS_CM, EATING_ANGLE)) {
                    continue;
                }
                env->food[f].active = false;
                env->food_eaten++;
                agent->eat_cooldown = EAT_COOLDOWN_STEPS;
                env->agents[i].rewards[0] += EAT_REWARD;
                break;
            }
        }

        Vec2 prev = agent->pos;
        float prev_ori = agent->orientation;
        float lin = 0.0f;
        float ang = 0.0f;
        if (agent->eat_cooldown <= 0) {
            lin = move * agent->max_linear_velocity;
            ang = turn * agent->max_angular_velocity;
        }
        agent->orientation = wrap_angle(agent->orientation + ang);
        agent->pos.x += cosf(agent->orientation) * lin;
        agent->pos.y += sinf(agent->orientation) * lin;

        bool collided = false;
        for (int j = 0; j < env->num_agents; j++) {
            if (j == i) {
                continue;
            }
            float dx = agent->pos.x - env->fish[j].pos.x;
            float dy = agent->pos.y - env->fish[j].pos.y;
            float diam = 2.0f * BODY_RADIUS_CM;
            if (dx * dx + dy * dy < diam * diam) {
                collided = true;
                break;
            }
        }
        if (collided) {
            agent->pos = prev;
        }
        agent->pos.x = clamp(
            agent->pos.x, BODY_RADIUS_CM,
            env->arena_size_x - BODY_RADIUS_CM
        );
        agent->pos.y = clamp(
            agent->pos.y, BODY_RADIUS_CM,
            env->arena_size_y - BODY_RADIUS_CM
        );
        float gx = agent->pos.x - prev.x;
        float gy = agent->pos.y - prev.y;
        float c = cosf(prev_ori);
        float s = sinf(prev_ori);
        // rotate ground displacement into ego frame (angle -prev_ori)
        agent->disp_ego = (Vec2){c * gx + s * gy, -s * gx + c * gy};
        env->collisions_fish += collided ? 1 : 0;
        if (collided) {
            env->agents[i].rewards[0] += COLLISION_REWARD;
        }

        // Proximity shaping 
        float nearest_food = INFINITY;
        bool any_food = false;
        for (int f = 0; f < env->num_food; f++) {
            if (!env->food[f].active) {
                continue;
            }
            any_food = true;
            float dx = agent->pos.x - env->food[f].pos.x;
            float dy = agent->pos.y - env->food[f].pos.y;
            float d = sqrtf(dx * dx + dy * dy);
            if (d < nearest_food) {
                nearest_food = d;
            }
        }
        if (any_food) {
            if (agent->has_previous_food_distance) {
                float arena_sum = env->arena_size_x + env->arena_size_y;
                env->agents[i].rewards[0] += PROXIMITY_SHAPING_REWARD
                    * (agent->previous_food_distance - nearest_food)
                    / arena_sum;
            }
            agent->previous_food_distance = nearest_food;
            agent->has_previous_food_distance = true;
        }
    }

    // Bites after all fish have moved
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* attacker = &env->fish[i];
        if (!attacker->bite_action) {
            continue;
        }
        int victim = -1;
        float nearest = INFINITY;
        for (int j = 0; j < env->num_agents; j++) {
            if (i == j) {
                continue;
            }
            if (!in_forward_cone(attacker, env->fish[j].pos, BITING_RADIUS_CM, EATING_ANGLE)) {
                continue;
            }
            float dx = env->fish[j].pos.x - attacker->pos.x;
            float dy = env->fish[j].pos.y - attacker->pos.y;
            float dist2 = dx * dx + dy * dy;
            if (dist2 < nearest) {
                victim = j;
                nearest = dist2;
            }
        }
        if (victim >= 0) {
            env->fish[victim].was_bitten = true;
            env->bites++;
            // Reference: is_bitten * (1 + size_diff), size_diff ∈ [-1, 1] → factor ∈ [0, 2]
            float size_difference = attacker->size - env->fish[victim].size;
            env->agents[victim].rewards[0] += BITTEN_REWARD * (1.0f + size_difference);
            env->agents[i].rewards[0] += BITE_REWARD;
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        env->fish[i].eat_cooldown -= env->fish[i].eat_cooldown > 0;
        env->fish[i].bite_cooldown -= env->fish[i].bite_cooldown > 0;
        env->episode_return += env->agents[i].rewards[0];
    }

    compute_observations(env);

    if (env->tick >= env->episode_length || env->food_eaten == env->num_food) {
        env->log.episode_length += (float)env->tick;
        env->log.episode_return += env->episode_return;
        env->log.score += env->episode_return;
        env->log.perf += (float)env->food_eaten / (float)env->num_food;
        env->log.food_eaten_mean +=(float)env->food_eaten / (float)env->num_agents;
        env->log.eod_rate += (float)env->eod_agent_steps / (float)(env->tick * env->num_agents);
        env->log.collisions_fish += (float)env->collisions_fish;
        env->log.bites += (float)env->bites;
        env->log.food_per_fish_area += env->food_per_fish_area;
        env->log.n += 1.0f;
        puf_reset(env);
        for (int i = 0; i < env->num_agents; i++) {
            env->agents[i].terminals[0] = 1.0f;
        }
    }
}

Vector2 world_to_screen(const Wef* env, Vec2 p) {
    Client* client = env->client;
    float usable_width = client->window_width - 2.0f * client->margin;
    float usable_height = client->window_height - 2.0f * client->margin;
    return (Vector2){
        client->margin + p.x / env->arena_size_x * usable_width,
        client->window_height - client->margin -
            p.y / env->arena_size_y * usable_height,
    };
}

static Color wef_lerp_color(Color a, Color b, float t) {
    t = clamp(t, 0.0f, 1.0f);
    return (Color){
        (unsigned char)((float)a.r + ((float)b.r - (float)a.r) * t),
        (unsigned char)((float)a.g + ((float)b.g - (float)a.g) * t),
        (unsigned char)((float)a.b + ((float)b.b - (float)a.b) * t),
        (unsigned char)((float)a.a + ((float)b.a - (float)a.a) * t),
    };
}

// Map |E| in V/m → color; colormap is defined in V/cm (yellow → red).
static Color wef_color_from_field(float strength_vm) {
    float strength_vcm = strength_vm * 0.01f;  // V/m → V/cm
    float log_s = log10f(fmaxf(strength_vcm, 1e-20f));
    float t = clamp(
        (log_s - WEF_FIELD_LOG_LO) / (WEF_FIELD_LOG_HI - WEF_FIELD_LOG_LO),
        0.0f,
        1.0f
    );
    return wef_lerp_color(WEF_COLOR_FIELD_WEAK, WEF_COLOR_FIELD_STRONG, t);
}

// Unit-direction arrow in screen space.
static void wef_draw_field_arrow(Vector2 base, float ux, float uy, float len, Color color) {
    Vector2 tip = {base.x + ux * len, base.y + uy * len};
    Vector2 wing = {base.x + ux * len * 0.75f, base.y + uy * len * 0.75f};
    float nx = -uy * len * 0.15f;
    float ny = ux * len * 0.15f;
    DrawLineV(base, tip, color);
    DrawLineV((Vector2){wing.x + nx, wing.y + ny}, tip, color);
    DrawLineV((Vector2){wing.x - nx, wing.y - ny}, tip, color);
}

// Compact V/cm color bar for the WEF log range.
static void wef_draw_field_colorbar(int win_w, int win_h) {
    const int bar_h = 120;
    const int bar_x0 = win_w - 18;
    const int bar_x1 = win_w - 10;
    const int bar_y1 = win_h - 14;
    const int bar_y0 = bar_y1 - bar_h;
    for (int i = 0; i < bar_h; i++) {
        float t = (float)i / (float)(bar_h - 1);
        float log_s = WEF_FIELD_LOG_LO + t * (WEF_FIELD_LOG_HI - WEF_FIELD_LOG_LO);
        // Color map expects V/m; convert V/cm → V/m (*100).
        Color c = wef_color_from_field(powf(10.0f, log_s) * 100.0f);
        DrawLine(bar_x0, bar_y1 - i, bar_x1, bar_y1 - i, c);
    }
    DrawRectangleLines(bar_x0 - 1, bar_y0, bar_x1 - bar_x0 + 2, bar_h, WEF_COLOR_MIDGRAY);
    DrawText("V/cm", win_w - 42, bar_y0 - 12, 10, WEF_COLOR_MIDGRAY);
    DrawText("1e-3", win_w - 48, bar_y0 + 2, 10, WEF_COLOR_MIDGRAY);
    DrawText("1e-7", win_w - 48, bar_y1 - 10, 10, WEF_COLOR_MIDGRAY);
}

void puf_render(Wef* env) {
    if (env->client == NULL) {
        Client* client = (Client*)calloc(1, sizeof(Client));
        client->window_width = 900;
        client->window_height = 900;
        client->margin = 55;
        client->show_field = true;
        client->show_sensors = true;
        InitWindow(client->window_width, client->window_height, "Weakly Electric fish");
        SetTargetFPS(60);
        env->client = client;
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }
    // Hold Left Shift + WASD/arrows; Space bites. Skip F/S viz toggles while driving.
    if (IsWindowReady() && IsKeyDown(KEY_LEFT_SHIFT)) {
        float* a = env->agents[0].actions;
        a[0] = 0.0f;
        a[1] = 0.0f;
        a[2] = 1.0f;
        a[3] = -1.0f;
        if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
            a[0] = 1.0f;
        } else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
            a[0] = -1.0f;
        }
        if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
            a[1] = -1.0f;
        } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
            a[1] = 1.0f;
        }
        if (IsKeyDown(KEY_SPACE)) {
            a[3] = 1.0f;
        }
    } else {
        if (IsKeyPressed(KEY_F)) {
            env->client->show_field = !env->client->show_field;
        }
        if (IsKeyPressed(KEY_S)) {
            env->client->show_sensors = !env->client->show_sensors;
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        Trace* trace = &env->client->traces[i];
        if (env->agents[i].terminals[0]) {
            trace->index = 0;
            trace->count = 0;
        }
        trace->pos[trace->index] = env->fish[i].pos;
        trace->index = (trace->index + 1) % TRACE_LENGTH;
        if (trace->count < TRACE_LENGTH) {
            trace->count++;
        }
    }

    BeginDrawing();
    ClearBackground(WEF_COLOR_BG);

    Vector2 arena_min = world_to_screen(env, (Vec2){0.0f, env->arena_size_y});
    Vector2 arena_max = world_to_screen(env, (Vec2){env->arena_size_x, 0.0f});
    DrawRectangleRec(
        (Rectangle){
            arena_min.x, arena_min.y,
            arena_max.x - arena_min.x, arena_max.y - arena_min.y
        },
        WEF_COLOR_BG
    );

    if (env->client->show_field) {
        Mono mono[WEF_MAX_MONO];
        Dipole induced[MAX_AGENTS + MAX_FOOD];
        Dipole intrinsic[MAX_AGENTS + MAX_FOOD];
        int n_mono = 0;
        int n_ind = 0;
        int n_intr = 0;
        for (int a = 0; a < env->num_agents; a++) {
            for (int p = 0; p < 2; p++) {
                mono[n_mono++] = (Mono){
                    to_m(env->fish[a].eod_pos[p]), env->fish[a].eod_charge[p]
                };
            }
            induced[n_ind++] = (Dipole){
                to_m(env->fish[a].pos), env->fish[a].induced_moment
            };
            intrinsic[n_intr++] = (Dipole){
                to_m(env->fish[a].pos), env->fish[a].intrinsic_moment
            };
        }
        for (int f = 0; f < env->num_food; f++) {
            if (!env->food[f].active) {
                continue;
            }
            induced[n_ind++] = (Dipole){
                to_m(env->food[f].pos), env->food[f].induced_moment
            };
            intrinsic[n_intr++] = (Dipole){
                to_m(env->food[f].pos), env->food[f].intrinsic_moment
            };
        }

        // Local vector field around each fish (fixed-length arrows, strength → color).
        const int columns = 36;
        const int rows = 36;
        float radius_squared =
            env->electric_field_radius_cm * env->electric_field_radius_cm;
        const float arrow_len = 10.0f;
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                Vec2 pos = {
                    env->arena_size_x * (column + 0.5f) / columns,
                    env->arena_size_y * (row + 0.5f) / rows,
                };
                bool near_fish = false;
                for (int i = 0; i < env->num_agents; i++) {
                    float dx = pos.x - env->fish[i].pos.x;
                    float dy = pos.y - env->fish[i].pos.y;
                    if (dx * dx + dy * dy <= radius_squared) {
                        near_fish = true;
                        break;
                    }
                }
                if (!near_fish) {
                    continue;
                }

                // Viz: knollen agent range for EODs; morm food range for food dips.
                Vec2 f1 = measure_field(
                    env, pos, mono, n_mono, induced, n_ind, env->num_agents,
                    KNOLLEN_AGENT_RANGE_CM, MORM_FOOD_RANGE_CM,
                    env->reflection_wall_range_cm
                );
                Vec2 f2 = measure_field(
                    env, pos, NULL, 0, intrinsic, n_intr, env->num_agents,
                    KNOLLEN_AGENT_RANGE_CM, AMP_FOOD_RANGE_CM,
                    env->reflection_wall_range_cm
                );
                float fx = f1.x + f2.x;
                float fy = f1.y + f2.y;
                float strength = sqrtf(fx * fx + fy * fy);
                if (strength <= 1e-20f) {
                    continue;
                }
                // Screen-space unit direction
                float ux = fx / strength;
                float uy = -(fy / strength);
                Vector2 base = world_to_screen(env, pos);
                // Center arrow on sample point (vector-field look).
                Vector2 mid = {
                    base.x - ux * arrow_len * 0.5f,
                    base.y - uy * arrow_len * 0.5f,
                };
                wef_draw_field_arrow(
                    mid, ux, uy, arrow_len, wef_color_from_field(strength)
                );
            }
        }
    }

    for (int i = 0; i < env->num_agents; i++) {
        Trace* trace = &env->client->traces[i];
        for (int j = 0; j < trace->count - 1; j++) {
            int current =
                (trace->index - j - 1 + TRACE_LENGTH) % TRACE_LENGTH;
            int previous =
                (trace->index - j - 2 + TRACE_LENGTH) % TRACE_LENGTH;
            float alpha =
                0.55f * (float)(trace->count - j) / (float)trace->count;
            DrawLineEx(
                world_to_screen(env, trace->pos[current]),
                world_to_screen(env, trace->pos[previous]),
                2.0f, ColorAlpha(WEF_COLOR_FISH, alpha)
            );
        }
    }

    Client* client = env->client;
    float scale = fminf(
        (client->window_width - 2.0f * client->margin) / env->arena_size_x,
        (client->window_height - 2.0f * client->margin) / env->arena_size_y
    );
    for (int i = 0; i < env->num_food; i++) {
        if (!env->food[i].active) {
            continue;
        }
        Vector2 position = world_to_screen(env, env->food[i].pos);
        float radius = fmaxf(2.5f, FOOD_RADIUS_CM * scale);
        DrawCircleV(position, radius, WEF_COLOR_FOOD);
    }
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->fish[i];
        Vector2 center = world_to_screen(env, agent->pos);
        float radius = BODY_RADIUS_CM * scale;

        if (agent->emits_eod) {
            float pulse = radius + 5.0f +
                5.0f * sinf((float)env->tick * 0.18f);
            DrawCircleLines((int)center.x, (int)center.y, pulse,
                WEF_COLOR_FISH_PULSE);
        }

        float heading_x = cosf(agent->orientation);
        float heading_y = -sinf(agent->orientation);
        DrawRing(center, radius - 1.5f, radius + 1.5f,
            0.0f, 360.0f, 32, WEF_COLOR_FISH);
        Vector2 nose = {
            center.x + heading_x * radius * 1.35f,
            center.y + heading_y * radius * 1.35f,
        };
        DrawLineEx(center, nose, 3.0f, WEF_COLOR_FISH);

        Vector2 positive = world_to_screen(env, agent->eod_pos[0]);
        Vector2 negative = world_to_screen(env, agent->eod_pos[1]);
        DrawCircleV(positive, 3.0f, WEF_COLOR_EOD_POS);
        DrawCircleV(negative, 3.0f, WEF_COLOR_EOD_NEG);

        if (env->client->show_sensors) {
            for (int sensor_idx = 0; sensor_idx < NUM_KNOLLEN;
                    sensor_idx++) {
                Sensor w = sensor_world(&g_knollen[sensor_idx], agent);
                DrawCircleV(
                    world_to_screen(env, w.p),
                    1.5f, WEF_COLOR_SENSOR
                );
            }
        }
        DrawText(TextFormat("%d", i + 1),
            (int)(center.x + radius + 4), (int)(center.y - radius), 16,
            WEF_COLOR_TEXT);
    }

    DrawRectangleLinesEx(
        (Rectangle){
            arena_min.x, arena_min.y,
            arena_max.x - arena_min.x, arena_max.y - arena_min.y
        },
        2.0f, WEF_COLOR_MIDGRAY
    );
    DrawText("Weakly electric fish", 20, 14, 24, WEF_COLOR_TEXT);
    int active_eods = 0;
    for (int i = 0; i < env->num_agents; i++) {
        active_eods += env->fish[i].emits_eod ? 1 : 0;
    }
    DrawText(TextFormat("step %d   active EODs %d/%d",
        env->tick, active_eods, env->num_agents),
        env->client->window_width - 285, 18, 18, WEF_COLOR_MIDGRAY);
    DrawText(TextFormat("food %d/%d", env->food_eaten, env->num_food),
        20, env->client->window_height - 32, 18, WEF_COLOR_MIDGRAY);
    DrawText(TextFormat("field radius %.0f cm", env->electric_field_radius_cm),
        180, env->client->window_height - 32, 18, WEF_COLOR_MIDGRAY);

    if (env->client->show_field) {
        wef_draw_field_colorbar(
            env->client->window_width, env->client->window_height
        );
    }
    EndDrawing();
    puf_web_vsync();
}

void puf_close(Wef* env) {
    if (env->client) {
        if (IsWindowReady()) {
            CloseWindow();
        }
        free(env->client);
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents");
    assert(env->num_agents > 0 && env->num_agents <= MAX_AGENTS);
    env->min_arena_size_x = dict_get(kwargs, "min_arena_width");
    env->min_arena_size_y = dict_get(kwargs, "min_arena_height");
    env->max_arena_size_x = dict_get(kwargs, "max_arena_width");
    env->max_arena_size_y = dict_get(kwargs, "max_arena_height");
    env->arena_size_x = env->min_arena_size_x;
    env->arena_size_y = env->min_arena_size_y;
    env->food_distribution = (FoodDistribution)dict_get(kwargs, "food_distribution");
    env->configured_num_food = dict_get(kwargs, "num_food");
    assert(env->configured_num_food > 0 && env->configured_num_food <= MAX_FOOD);
    env->patch_radius_cm = dict_get(kwargs, "patch_radius");
    env->patch_radius_std_cm = dict_get(kwargs, "patch_radius_std");
    env->patch_density = dict_get(kwargs, "patch_density");
    env->electric_field_radius_cm = dict_get(kwargs, "electric_field_radius");
    env->reflection_wall_range_cm = dict_get(kwargs, "reflection_wall_range");
    env->episode_length = dict_get(kwargs, "episode_length");
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }

    // Shared body-frame sensor rings (fixed body radius).
    float r = BODY_RADIUS_CM;
    float chin = PI_F / 3.0f;
    int num_chin = 10;
    int num_rest = NUM_MORMYROMASTS - num_chin;
    for (int s = 0; s < num_chin; s++) {
        float a = -0.5f * chin + chin * (float)s / (float)(num_chin - 1);
        float c = cosf(a);
        float sn = sinf(a);
        g_morm[s] = (Sensor){{c * r, sn * r}, {c, sn}};
    }
    for (int s = 0; s < num_rest; s++) {
        float a = 0.5f * chin
            + (2.0f * PI_F - chin) * (float)s / (float)num_rest;
        float c = cosf(a);
        float sn = sinf(a);
        g_morm[num_chin + s] = (Sensor){{c * r, sn * r}, {c, sn}};
    }
    for (int s = 0; s < NUM_AMPULLARY; s++) {
        float a = 2.0f * PI_F * (float)s / (float)NUM_AMPULLARY;
        float c = cosf(a);
        float sn = sinf(a);
        g_amp[s] = (Sensor){{c * r, sn * r}, {c, sn}};
    }
    for (int s = 0; s < NUM_KNOLLEN; s++) {
        float a = 2.0f * PI_F * (float)s / (float)NUM_KNOLLEN;
        float c = cosf(a);
        float sn = sinf(a);
        g_knollen[s] = (Sensor){{c * r, sn * r}, {c, sn}};
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "food_eaten_mean", log->food_eaten_mean);
    dict_set(out, "eod_rate", log->eod_rate);
    dict_set(out, "collisions_fish", log->collisions_fish);
    dict_set(out, "bites", log->bites);
    dict_set(out, "food_per_fish_area", log->food_per_fish_area);
    dict_set(out, "n", log->n);
}
