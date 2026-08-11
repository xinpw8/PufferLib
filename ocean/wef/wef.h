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
#include "pufferenv.h"

// 5c trainer macros (no binding.c). Float obs; pufferl converts on upload.
#define ACT_SIZES {1, 1, 1, 1}
#define NUM_ATNS 4
typedef float obs_t;

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
#define BITTEN_REWARD -0.5f
#define BITE_REWARD -0.0001f
#define COLLISION_REWARD -0.05f

#define TRACE_LENGTH 128

#define EAT_COOLDOWN_STEPS 3
#define BITE_COOLDOWN_STEPS 5

#define AMPULLARY_MIN_VM 2e-10f
#define AMPULLARY_MAX_VM 2e-8f
#define MORMYROMAST_MIN_VM 5e-8f
#define MORMYROMAST_MAX_VM 5e-2f
#define KNOLLEN_MIN_VM 2e-7f

// Arena wall indices for first-order image charges.
#define WALL_LEFT 0
#define WALL_RIGHT 1
#define WALL_BOTTOM 2
#define WALL_TOP 3
#define NUM_WALLS 4

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
    float field_fish_range_cm;
    float field_food_range_cm;
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

// Probe in cm. Sources (Mono/Dipole.p) already meters. wall_range_cm=0 → no images.
Vec2 measure_field(Env* env, Vec2 probe_cm, const Mono* mono, int n_mono,
    const Dipole* dip, int n_dip, int n_agent_dipoles, float wall_range_cm) {
    float pmx = probe_cm.x * CM_TO_M;
    float pmy = probe_cm.y * CM_TO_M;
    float fish_r = env->field_fish_range_cm * CM_TO_M;
    float food_r = env->field_food_range_cm * CM_TO_M;
    float fish_range2 = fish_r * fish_r;
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
        if (dx * dx + dy * dy > fish_range2) {
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
        float range2 = (i < n_agent_dipoles) ? fish_range2 : food_range2;
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
            Vec2 f = measure_field(
                env, agent->pos, eod, n_eod, NULL, 0, 0, 0.0f
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
            env, env->food[i].pos, eod, n_eod, NULL, 0, 0, 0.0f
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
        obs_t* obs = (obs_t*)env->agents[i].observations;
        int obs_idx = 0;

        bool cons_eod = false;
        for (int other = 0; other < env->num_agents; other++) {
            if (other != i && env->fish[other].emits_eod) {
                cons_eod = true;
                break;
            }
        }
        // Mormyromasts: induced image field after direct-EOD subtraction.
        for (int sensor_idx = 0; sensor_idx < NUM_MORMYROMASTS; sensor_idx++) {
            Sensor w = sensor_world(&g_morm[sensor_idx], agent);
            Vec2 f = measure_field(
                env, w.p, NULL, 0, induced, n_induced, env->num_agents,
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
        // Ampullary: intrinsic sources with static self-field removed.
        for (int sensor_idx = 0; sensor_idx < NUM_AMPULLARY; sensor_idx++) {
            Sensor w = sensor_world(&g_amp[sensor_idx], agent);
            Vec2 f = measure_field(
                env, w.p, NULL, 0, intrinsic, n_intrinsic, env->num_agents,
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
        // Knollenorgans: one directional 12-receptor block per conspecific.
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
                        env, w.p, eod, 2, NULL, 0, 0, 0.0f
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
        float size_mult = 1.0f + agent.size;
        agent.max_linear_velocity = (35.0f / SIMULATION_HZ) * size_mult;
        agent.max_angular_velocity = (3.6f / SIMULATION_HZ) * size_mult;
        agent.emits_eod = true;
        env->fish[i] = agent;
        env->agents[i].rewards[0] = 0.0f;
        env->agents[i].terminals[0] = 0.0f;
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
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }
    if (IsKeyPressed(KEY_F)) {
        env->client->show_field = !env->client->show_field;
    }
    if (IsKeyPressed(KEY_S)) {
        env->client->show_sensors = !env->client->show_sensors;
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
    ClearBackground(WHITE);

    Vector2 arena_min = world_to_screen(env, (Vec2){0.0f, env->arena_size_y});
    Vector2 arena_max = world_to_screen(env, (Vec2){env->arena_size_x, 0.0f});
    DrawRectangleRec(
        (Rectangle){
            arena_min.x, arena_min.y,
            arena_max.x - arena_min.x, arena_max.y - arena_min.y
        },
        WHITE
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

        const int columns = 25;
        const int rows = 25;
        float radius_squared =
            env->electric_field_radius_cm * env->electric_field_radius_cm;
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

                Vec2 f1 = measure_field(
                    env, pos, mono, n_mono, induced, n_ind, env->num_agents,
                    env->reflection_wall_range_cm
                );
                Vec2 f2 = measure_field(
                    env, pos, NULL, 0, intrinsic, n_intr, env->num_agents,
                    env->reflection_wall_range_cm
                );
                float fx = f1.x + f2.x;
                float fy = f1.y + f2.y;
                float strength = sqrtf(fx * fx + fy * fy);
                if (strength <= 0.0f) {
                    continue;
                }
                float log_strength = log10f(strength);
                float t = clamp((log_strength + 8.0f) / 7.0f, 0.0f, 1.0f);
                float arrow_length = 3.5f + 8.0f * t;
                Vector2 start = world_to_screen(env, pos);
                Vector2 end = {
                    start.x + (fx / strength) * arrow_length,
                    start.y - (fy / strength) * arrow_length,
                };
                unsigned char shade = (unsigned char)(215.0f - 70.0f * t);
                Color color = (Color){shade, shade, shade, 120};
                DrawCircleV(start, 1.3f, color);
                DrawLineEx(start, end, 0.9f, color);
            }
        }
    }

    const Color color = {157, 122, 216, 255};
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
                2.0f, ColorAlpha(color, alpha)
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
        // Food pellets
        DrawCircleV(position, radius, (Color){0x00, 0x80, 0x00, 140});
    }
    for (int i = 0; i < env->num_agents; i++) {
        FishAgent* agent = &env->fish[i];
        Vector2 center = world_to_screen(env, agent->pos);
        float radius = BODY_RADIUS_CM * scale;

        if (agent->emits_eod) {
            float pulse = radius + 5.0f +
                5.0f * sinf((float)env->tick * 0.18f);
            DrawCircleLines((int)center.x, (int)center.y, pulse,
                (Color){color.r, color.g, color.b, 120});
        }

        float heading_x = cosf(agent->orientation);
        float heading_y = -sinf(agent->orientation);
        DrawRing(center, radius - 1.5f, radius + 1.5f,
            0.0f, 360.0f, 32, color);
        Vector2 nose = {
            center.x + heading_x * radius * 1.35f,
            center.y + heading_y * radius * 1.35f,
        };
        DrawLineEx(center, nose, 3.0f, color);

        Vector2 positive = world_to_screen(env, agent->eod_pos[0]);
        Vector2 negative = world_to_screen(env, agent->eod_pos[1]);
        DrawCircleV(positive, 3.0f, RED);
        DrawCircleV(negative, 3.0f, BLUE);

        if (env->client->show_sensors) {
            for (int sensor_idx = 0; sensor_idx < NUM_KNOLLEN;
                    sensor_idx++) {
                Sensor w = sensor_world(&g_knollen[sensor_idx], agent);
                DrawCircleV(
                    world_to_screen(env, w.p),
                    1.5f, (Color){184, 164, 224, 180}
                );
            }
        }
        DrawText(TextFormat("%d", i + 1),
            (int)(center.x + radius + 4), (int)(center.y - radius), 16, BLACK);
    }

    DrawRectangleLinesEx(
        (Rectangle){
            arena_min.x, arena_min.y,
            arena_max.x - arena_min.x, arena_max.y - arena_min.y
        },
        3.0f, DARKGRAY
    );
    DrawText("Weakly electric fish", 20, 14, 24, BLACK);
    int active_eods = 0;
    for (int i = 0; i < env->num_agents; i++) {
        active_eods += env->fish[i].emits_eod ? 1 : 0;
    }
    DrawText(TextFormat("step %d   active EODs %d/%d",
        env->tick, active_eods, env->num_agents),
        env->client->window_width - 285, 18, 18, DARKGRAY);
    DrawText(TextFormat("food %d/%d", env->food_eaten, env->num_food),
        env->client->window_width - 145,
        env->client->window_height - 32, 18, DARKGRAY);
    DrawText(TextFormat("field radius %.0f cm", env->electric_field_radius_cm),
        430, env->client->window_height - 32, 18, DARKGRAY);
    EndDrawing();
}

void puf_close(Wef* env) {
    if (env->client) {
        if (IsWindowReady()) {
            CloseWindow();
        }
        free(env->client);
        env->client = NULL;
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = (int)dict_get(kwargs, "num_agents");
    assert(env->num_agents > 0 && env->num_agents <= MAX_AGENTS);
    env->min_arena_size_x = (float)dict_get(kwargs, "min_arena_width");
    env->min_arena_size_y = (float)dict_get(kwargs, "min_arena_height");
    env->max_arena_size_x = (float)dict_get(kwargs, "max_arena_width");
    env->max_arena_size_y = (float)dict_get(kwargs, "max_arena_height");
    env->arena_size_x = env->min_arena_size_x;
    env->arena_size_y = env->min_arena_size_y;
    env->food_distribution = (FoodDistribution)(int)dict_get(kwargs, "food_distribution");
    env->configured_num_food = (int)dict_get(kwargs, "num_food");
    assert(env->configured_num_food > 0 && env->configured_num_food <= MAX_FOOD);
    env->patch_radius_cm = (float)dict_get(kwargs, "patch_radius");
    env->patch_radius_std_cm = (float)dict_get(kwargs, "patch_radius_std");
    env->patch_density = (float)dict_get(kwargs, "patch_density");
    env->electric_field_radius_cm = (float)dict_get(kwargs, "electric_field_radius");
    env->reflection_wall_range_cm = (float)dict_get(kwargs, "reflection_wall_range");
    env->field_fish_range_cm = (float)dict_get(kwargs, "field_fish_range");
    env->field_food_range_cm = (float)dict_get(kwargs, "field_food_range");
    env->episode_length = (int)dict_get(kwargs, "episode_length");
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
}
