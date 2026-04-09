#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include "raylib.h"

#define NOOP   0
#define LEFT   1
#define RIGHT  2
#define UP     3
#define DOWN   4

// ─── Physics & game constants 
#define TICK_RATE               (1.0f / 60.0f)
#define GRAVITY                 980.0f          // px/s² slope-gravity scale

#define AGENT_RADIUS            12.0f
#define BOULDER_RADIUS          48.0f
#define GOAL_RADIUS             40.0f

#define AGENT_MASS              1.0f
#define BOULDER_MASS            20.0f

#define AGENT_ACCEL             700.0f          // px/s² per directional action
#define AGENT_DRAG              6.0f            // per-second linear velocity decay
#define BOULDER_LINEAR_DRAG     0.4f
#define BOULDER_ROLLING_DRAG    0.8f            // per-second angular velocity decay

#define RESTITUTION_BODIES      0.05f           // agent ↔ boulder (nearly inelastic)
#define RESTITUTION_WALL        0.15f
#define FRICTION_COEF           0.45f           // Coulomb friction coefficient

#define MAX_SPEED               1200.0f
#define MAX_ANGULAR_VEL         30.0f           // rad/s normalisation for boulder spin
#define MAX_TICKS               600
#define GOAL_CENTER_RADIUS      16.0f
#define GOAL_HOLD_TICKS         10

#define MAX_AGENTS              2
// Per-agent obs layout: self(6) + boulder(8) + goal(5) + (MAX_AGENTS-1) other agents(8 each)
#define OBS_SIZE                (6 + 8 + 5 + (MAX_AGENTS - 1) * 8)

typedef struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct Client {
    float width;
    float height;
    Texture2D sprites;
} Client;

// Circular rigid body — used for both agents and the boulder.
// All physics functions operate on this type.
typedef struct Entity {
    float x, y;            
    float vx, vy;          
    float mass;
    float radius;
    float angle;           // cumulative rotation (rad), increases CCW
    float angular_vel;     // angular velocity   (rad/s), + = CCW
    float inv_inertia;     // 1/I; solid disk: I = ½·mass·radius²  →  inv = 2/(m·r²)
} Entity;

typedef struct Boulder {
    Client* client;
    Log log;
    Log* logs;                    
    float* observations;  
    float* actions;        
    float* rewards;        
    float* terminals;     
    Entity boulder;
    Entity* agents;
    int num_agents;
    float goal_x, goal_y;
    int goal_hold;         // consecutive ticks boulder has been within GOAL_RADIUS
    float* heightmap;      // flat [hmap_rows × hmap_cols] elevation values (px)
    int hmap_cols;
    int hmap_rows;
    int width;
    int height;
    int tick;
    float score;
    unsigned int rng;
    int* moving_boulder;
} Boulder;

// Helper functions 

// Returns a uniform float in [0, 1) using rand_r seeded from env->rng.
static inline float randf(Boulder* env) {
    return (float)rand_r(&env->rng) / ((float)RAND_MAX + 1.0f);
}

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Initialise a circular rigid body as a solid disk at rest.
// Precomputes inv_inertia = 2 / (mass · radius²).
static inline void entity_init(Entity* e,float x, float y,float mass, float radius) {
    *e = (Entity){
        .x = x, .y = y,
        .mass = mass,
        .radius = radius,
        .inv_inertia = 2.0f / (mass * radius * radius),
    };
}

// Semi-implicit Euler integration with exponential velocity decay (drag).
//   ax, ay        : net acceleration (px/s²) this tick.
//   linear_drag   : linear  damping rate (s⁻¹); applied as exp(−k·dt) factor.
//   rolling_drag  : angular damping rate (s⁻¹).
static inline void integrate_entity(Entity* e,float ax, float ay,float linear_drag,float rolling_drag) {
    const float dt = TICK_RATE;
    e->vx += ax * dt;
    e->vy += ay * dt;
    float ld = expf(-linear_drag  * dt);
    float rd = expf(-rolling_drag * dt);
    e->vx *= ld;
    e->vy *= ld;
    e->x  += e->vx * dt;
    e->y  += e->vy * dt;
    e->angular_vel *= rd;
    e->angle += e->angular_vel * dt;
}

// ─── Height-map 

// Bilinear sample of the elevation map at world position (x, y).
// Returns 0 if heightmap is NULL or position is out of range.
static inline float heightmap_sample(const Boulder* env, float x, float y) {
    if (!env->heightmap) return 0.0f;
    float cx = (x / env->width)  * (float)(env->hmap_cols - 1);
    float cy = (y / env->height) * (float)(env->hmap_rows - 1);
    int ix = (int)cx, iy = (int)cy;
    if (ix < 0 || iy < 0 || ix >= env->hmap_cols - 1 || iy >= env->hmap_rows - 1)
        return 0.0f;
    float tx = cx - ix, ty = cy - iy;
    int s = env->hmap_cols;
    float h00 = env->heightmap[ iy    * s + ix    ];
    float h10 = env->heightmap[ iy    * s + ix + 1];
    float h01 = env->heightmap[(iy+1) * s + ix    ];
    float h11 = env->heightmap[(iy+1) * s + ix + 1];
    return h00*(1-tx)*(1-ty) + h10*tx*(1-ty) + h01*(1-tx)*ty + h11*tx*ty;
}

// Computes slope-induced gravitational acceleration (px/s²) at world position
// (x, y) via central-difference gradient of the height field.
// Fills *out_ax and *out_ay with the downhill acceleration components.
static inline void heightmap_gravity(const Boulder* env,
                                     float x, float y,
                                     float* out_ax, float* out_ay) {
    const float h = 4.0f;
    float dh_dx = (heightmap_sample(env, x + h, y) -
                   heightmap_sample(env, x - h, y)) / (2.0f * h);
    float dh_dy = (heightmap_sample(env, x, y + h) -
                   heightmap_sample(env, x, y - h)) / (2.0f * h);
    *out_ax = -GRAVITY * dh_dx;
    *out_ay = -GRAVITY * dh_dy;
}

// ─── Sphere–sphere impulse resolution
//
// Convention: n is the unit normal FROM a TO b.
//   vrel_n > 0  ⟹  contact points approaching  ⟹  apply impulse.
//
// Normal restitution + Coulomb tangential friction.
// Both bodies are modified in place.
//
// Derivation notes (2D rigid-body impulse):
//   • Contact arm:  r_a = +a->radius·n,  r_b = −b->radius·n
//   • Velocity at contact:  v_c = v_cm + ω × r  →  (vx − ω·ry , vy + ω·rx)
//   • r_a × n = 0  ⟹  angular term drops out of the normal-impulse denominator.
//   • cross2d(r_a, t) = +a->radius,  cross2d(r_b, t) = −b->radius
//     where t = (−ny, nx) is the CCW tangent.
static inline int resolve_sphere_sphere(Entity* a, Entity* b,
                                         float restitution, float friction) {
    float dx = b->x - a->x;
    float dy = b->y - a->y;
    float dist2 = dx*dx + dy*dy;
    float rsum  = a->radius + b->radius;
    if (dist2 >= rsum * rsum || dist2 < 1e-8f) return 0;

    float dist = sqrtf(dist2);
    float nx = dx / dist;           // unit normal  a → b
    float ny = dy / dist;
    float tx = -ny;                 // unit tangent (CCW 90° from n)
    float ty =  nx;

    // Positional correction: push apart proportional to inverse mass
    float overlap  = rsum - dist;
    float inv_msum = 1.0f / (a->mass + b->mass);
    a->x -= nx * overlap * (b->mass * inv_msum);
    a->y -= ny * overlap * (b->mass * inv_msum);
    b->x += nx * overlap * (a->mass * inv_msum);
    b->y += ny * overlap * (a->mass * inv_msum);

    // Contact-point arm vectors (centre → contact point)
    float ra_x =  a->radius * nx,  ra_y =  a->radius * ny;
    float rb_x = -b->radius * nx,  rb_y = -b->radius * ny;

    // Velocity at each contact point:  v_c = v_cm + ω × r  (2D: vx−ω·ry, vy+ω·rx)
    float vca_x = a->vx - a->angular_vel * ra_y;
    float vca_y = a->vy + a->angular_vel * ra_x;
    float vcb_x = b->vx - b->angular_vel * rb_y;
    float vcb_y = b->vy + b->angular_vel * rb_x;

    float vrx    = vca_x - vcb_x;
    float vry    = vca_y - vcb_y;
    float vrel_n = vrx * nx + vry * ny;
    if (vrel_n <= 0.0f) return 0;     // separating; no impulse needed

    float vrel_t = vrx * tx + vry * ty;

    float inv_meff = 1.0f/a->mass + 1.0f/b->mass;

    // Normal impulse (angular inertia term is 0 for sphere–sphere)
    float j_n = (1.0f + restitution) * vrel_n / inv_meff;

    // Tangential (friction) impulse including angular inertia in denominator
    // cross2d(ra, t) = +a->radius;  cross2d(rb, t) = −b->radius
    float denom_t = inv_meff
                  + a->radius * a->radius * a->inv_inertia
                  + b->radius * b->radius * b->inv_inertia;
    float j_t     = vrel_t / denom_t;
    float j_t_max = friction * j_n;
    if (j_t >  j_t_max) j_t =  j_t_max;
    if (j_t < -j_t_max) j_t = -j_t_max;

    // Linear impulse:  impulse on a = −J,  impulse on b = +J
    float Jx = j_n * nx + j_t * tx;
    float Jy = j_n * ny + j_t * ty;
    a->vx -= Jx / a->mass;
    a->vy -= Jy / a->mass;
    b->vx += Jx / b->mass;
    b->vy += Jy / b->mass;

    // Angular impulse:  Δω = cross2d(r, F) · inv_I
    //   F on a = −J,  F on b = +J
    a->angular_vel += (ra_x * (-Jy) - ra_y * (-Jx)) * a->inv_inertia;
    b->angular_vel += (rb_x *   Jy  - rb_y *   Jx ) * b->inv_inertia;

    return 1;
}

// ─── Sphere–horizontal-wall impulse resolution
//
// wall_y   : y-coordinate of the wall line.
// normal_y : outward wall normal y-component (screen coords: y increases DOWN).
//            +1  → top / ceiling wall (normal points downward toward entity).
//            −1  → bottom / floor wall (normal points upward toward entity).
//
// Derivation:
//   Contact arm rc = (0, −normal_y · radius)  →  toward wall face.
//   x-velocity at contact: vx_c = vx + ω·(−rc_y) = vx + ω·normal_y·radius.
//   cross2d(rc, t_x) = −rc_y = normal_y·radius.
static inline void resolve_sphere_hwall(Entity* e,
                                        float wall_y, float normal_y,
                                        float restitution, float friction) {
    // Signed distance from wall to entity centre along outward normal
    float sd  = (e->y - wall_y) * normal_y;
    float pen = e->radius - sd;
    if (pen <= 0.0f) return;

    e->y += pen * normal_y;             // push entity off wall

    float vn = e->vy * normal_y;        // velocity along outward normal
    if (vn >= 0.0f) return;             // already leaving

    float j_n = -(1.0f + restitution) * vn * e->mass;  // positive

    float rc_y  = -normal_y * e->radius;                 // arm y-component
    float vx_c  = e->vx + e->angular_vel * (-rc_y);      // x-vel at contact
    float rc_ct = -rc_y;                                  // cross2d(rc, (1,0))
    float denom_t = 1.0f/e->mass + rc_ct * rc_ct * e->inv_inertia;
    float j_t     = -vx_c / denom_t;
    float j_t_max = friction * j_n;
    if (j_t >  j_t_max) j_t =  j_t_max;
    if (j_t < -j_t_max) j_t = -j_t_max;

    e->vy += j_n * normal_y / e->mass;
    e->vx += j_t / e->mass;
    // cross2d((0, rc_y), (j_t, j_n·normal_y)) = −rc_y·j_t
    e->angular_vel += (-rc_y * j_t) * e->inv_inertia;
}

// ─── Sphere–vertical-wall impulse resolution ─────────────────────────────────
//
// wall_x   : x-coordinate of the wall line.
// normal_x : outward wall normal x-component.
//            +1  → left wall  (normal points right toward entity).
//            −1  → right wall (normal points left  toward entity).
//
// Derivation:
//   Contact arm rc = (−normal_x · radius, 0).
//   y-velocity at contact: vy_c = vy + ω·rc_x.
//   cross2d(rc, t_y) = rc_x = −normal_x·radius.
static inline void resolve_sphere_vwall(Entity* e,
                                        float wall_x, float normal_x,
                                        float restitution, float friction) {
    float sd  = (e->x - wall_x) * normal_x;
    float pen = e->radius - sd;
    if (pen <= 0.0f) return;

    e->x += pen * normal_x;

    float vn = e->vx * normal_x;
    if (vn >= 0.0f) return;

    float j_n = -(1.0f + restitution) * vn * e->mass;  // positive

    float rc_x  = -normal_x * e->radius;                 // arm x-component
    float vy_c  = e->vy + e->angular_vel * rc_x;         // y-vel at contact
    float rc_ct = rc_x;                                   // cross2d(rc, (0,1))
    float denom_t = 1.0f/e->mass + rc_ct * rc_ct * e->inv_inertia;
    float j_t     = -vy_c / denom_t;
    float j_t_max = friction * j_n;
    if (j_t >  j_t_max) j_t =  j_t_max;
    if (j_t < -j_t_max) j_t = -j_t_max;

    e->vx += j_n * normal_x / e->mass;
    e->vy += j_t / e->mass;
    // cross2d((rc_x, 0), (j_n·normal_x, j_t)) = rc_x·j_t
    e->angular_vel += (rc_x * j_t) * e->inv_inertia;
}


void init(Boulder* env) {
    env->tick      = 0;
    env->score     = 0.0f;
    env->goal_hold = 0;
    env->agents    = (Entity*)calloc(env->num_agents, sizeof(Entity));
    env->logs = (Log*)calloc(env->num_agents, sizeof(Log));
    env->moving_boulder = (int*)calloc(env->num_agents, sizeof(int));
}

void allocate(Boulder* env) {
    init(env);
    int n = env->num_agents;
    env->observations = (float*)calloc(n * OBS_SIZE, sizeof(float));
    env->actions      = (float*)calloc(n * 2,   sizeof(float));
    env->rewards      = (float*)calloc(n,      sizeof(float));
    env->terminals    = (float*)calloc(n,      sizeof(float));
    env->agents       = (Entity*)calloc(n,     sizeof(Entity));
}

void c_close(Boulder* env) {}

void free_allocated(Boulder* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->agents);
    if (env->heightmap) free(env->heightmap);
    c_close(env);
}

void add_log(Boulder* env) {
    for(int i= 0;i<env->num_agents; i++){
        env->log.episode_length += env->logs[i].episode_length;
        env->log.episode_return += env->logs[i].episode_return;
        env->log.score          += env->score;
        env->log.perf           += env->score;
        env->log.n              += 1;
    }
}

void compute_observations(Boulder* env) {
    float diag = hypotf((float)env->width, (float)env->height);
    float dist_scale = env->width > env->height ? env->width: env->height;
    dist_scale *= 0.5;
    for (int i = 0; i < env->num_agents; i++) {
        float*  obs = env->observations + i * OBS_SIZE;
        Entity* a   = &env->agents[i];
        int idx = 0;

        // Self (4): position relative to walls and velocity
        obs[idx++] = a->x / env->width;
        obs[idx++] = (env->width - a->x) / env->width;
        obs[idx++] = a->y / env->height;
        obs[idx++] = (env->height - a->y) / env->height;
        obs[idx++] = a->vx / MAX_SPEED;
        obs[idx++] = a->vy / MAX_SPEED;

        // Boulder (8): ego-relative pos, dist, angle, velocity, spin
        float bdx   = env->boulder.x - a->x;
        float bdy   = env->boulder.y - a->y;
        float bdist = hypotf(bdx, bdy);
        float bang  = atan2f(bdy, bdx);
        obs[idx++] = clampf(bdx / dist_scale, -1.0f, 1.0f);
        obs[idx++] = clampf(bdy / dist_scale, -1.0f, 1.0f);
        obs[idx++] = clampf(bdist / diag, 0.0f, 1.0f);
        obs[idx++] = sinf(bang);
        obs[idx++] = cosf(bang);
        obs[idx++] = env->boulder.vx / MAX_SPEED;
        obs[idx++] = env->boulder.vy / MAX_SPEED;
        obs[idx++] = env->boulder.angular_vel / MAX_ANGULAR_VEL;

        // Goal (5): ego-relative pos, dist, angle
        float gdx   = env->goal_x - env->boulder.x;
        float gdy   = env->goal_y - env->boulder.y;
        float gdist = hypotf(gdx, gdy);
        float gang  = atan2f(gdy, gdx);
        obs[idx++] = clampf(gdx / dist_scale, -1.0f, 1.0f);
        obs[idx++] = clampf(gdy / dist_scale, -1.0f, 1.0f);
        obs[idx++] = clampf(gdist / diag, 0.0f, 1.0f);
        obs[idx++] = sinf(gang);
        obs[idx++] = cosf(gang);

        // Other agents (8 each, up to MAX_AGENTS-1 slots, zero-padded if fewer)
        int filled = 0;
        for (int j = 0; j < env->num_agents && filled < MAX_AGENTS - 1; j++) {
            if (j == i) continue;
            Entity* o   = &env->agents[j];
            float odx   = o->x - a->x;
            float ody   = o->y - a->y;
            float odist = hypotf(odx, ody);
            float oang  = atan2f(ody, odx);
            obs[idx++] = clampf(odx / dist_scale, -1.0f, 1.0f);
            obs[idx++] = clampf(ody / dist_scale, -1.0f, 1.0f);
            obs[idx++] = clampf(odist / diag, 0.0f, 1.0f);
            obs[idx++] = sinf(oang);
            obs[idx++] = cosf(oang);
            obs[idx++] = (o->vx - a->vx) / MAX_SPEED;
            obs[idx++] = (o->vy - a->vy) / MAX_SPEED;
            obs[idx++] = hypotf(o->vx, o->vy) / MAX_SPEED;
            filled++;
        }
        while (filled++ < MAX_AGENTS - 1) {
            for (int k = 0; k < 8; k++) obs[idx++] = 0.0f;
        }
    }
}

void c_reset(Boulder* env) {
    env->score     = 0.0f;
    env->tick      = 0;
    env->goal_hold = 0;

    entity_init(&env->boulder,env->width  * 0.5f,env->height * 0.5f,BOULDER_MASS, BOULDER_RADIUS);

    float pi2 = 2.0f * 3.14159265f;
    for (int i = 0; i < env->num_agents; i++) {
        env->logs[i] = (Log){0};
        float angle = (float)i * (pi2 / env->num_agents);
        float dist  = BOULDER_RADIUS + AGENT_RADIUS + 24.0f;
        float start_x = env->boulder.x + cosf(angle) * dist;
        float start_y = env->boulder.y + sinf(angle) * dist;
        entity_init(&env->agents[i], start_x, start_y, AGENT_MASS, AGENT_RADIUS);
    }

    // Random goal far enough from the boulder start
    float gx, gy;
    do {
        gx = GOAL_RADIUS + randf(env) * (env->width  - 2.0f * GOAL_RADIUS);
        gy = GOAL_RADIUS + randf(env) * (env->height - 2.0f * GOAL_RADIUS);
    } while (hypotf(gx - env->boulder.x, gy - env->boulder.y) < BOULDER_RADIUS * 3.0f);
    env->goal_x = gx;
    env->goal_y = gy;

    memset(env->moving_boulder, 0, 2*sizeof(int));
    compute_observations(env);
}

void c_step(Boulder* env) {
    float dist_to_goal_before = hypotf(env->boulder.x - env->goal_x, env->boulder.y - env->goal_y);
    // 8-direction unit vectors (screen coords: y increases down)
    // index: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
    static const float DIR_AX[8] = { 1.0f,  0.707f,  0.0f, -0.707f, -1.0f, -0.707f,  0.0f,  0.707f};
    static const float DIR_AY[8] = { 0.0f, -0.707f, -1.0f, -0.707f,  0.0f,  0.707f,  1.0f,  0.707f};

    // Agent acceleration from discrete actions + height-map slope
    for (int i = 0; i < env->num_agents; i++) {
        int throttle = (int)env->actions[i * 2 + 0];   // 0=off, 1=on
        int dir      = (int)env->actions[i * 2 + 1];   // 0..7
        float ax = 0.0f, ay = 0.0f;
        if (throttle && dir >= 0 && dir < 8) {
            ax = DIR_AX[dir] * AGENT_ACCEL;
            ay = DIR_AY[dir] * AGENT_ACCEL;
        }

        float hgx = 0.0f, hgy = 0.0f;
        heightmap_gravity(env, env->agents[i].x, env->agents[i].y, &hgx, &hgy);
        integrate_entity(&env->agents[i], ax + hgx, ay + hgy,
                         AGENT_DRAG, 0.0f);
    }

    // Boulder integration (height-map slope only; agents push via collision)
    {
        float hgx = 0.0f, hgy = 0.0f;
        heightmap_gravity(env, env->boulder.x, env->boulder.y, &hgx, &hgy);
        integrate_entity(&env->boulder, hgx, hgy,BOULDER_LINEAR_DRAG, BOULDER_ROLLING_DRAG);
    }

    // Collision: agents ↔ boulder
    for (int i = 0; i < env->num_agents; i++) {
        int collided = resolve_sphere_sphere(&env->agents[i], &env->boulder,RESTITUTION_BODIES, FRICTION_COEF);
        env->moving_boulder[i] = 1.0f;
    }

    // Collision: agents ↔ agents
    for (int i = 0; i < env->num_agents; i++) {
        for (int j = i + 1; j < env->num_agents; j++) {
            resolve_sphere_sphere(&env->agents[i], &env->agents[j],RESTITUTION_BODIES, FRICTION_COEF);
        }
    }

    // All entities ↔ map boundary walls (sphere-on-line)
    float wx0 = 0.0f;
    float wy0 = 0.0f;
    float wx1 = env->width;
    float wy1 = env->height;

    for (int i = 0; i < env->num_agents; i++) {
        resolve_sphere_vwall(&env->agents[i], wx0, +1.0f, RESTITUTION_WALL, FRICTION_COEF);
        resolve_sphere_vwall(&env->agents[i], wx1, -1.0f, RESTITUTION_WALL, FRICTION_COEF);
        resolve_sphere_hwall(&env->agents[i], wy0, +1.0f, RESTITUTION_WALL, FRICTION_COEF);
        resolve_sphere_hwall(&env->agents[i], wy1, -1.0f, RESTITUTION_WALL, FRICTION_COEF);
    }
    resolve_sphere_vwall(&env->boulder, wx0, +1.0f, RESTITUTION_WALL, FRICTION_COEF);
    resolve_sphere_vwall(&env->boulder, wx1, -1.0f, RESTITUTION_WALL, FRICTION_COEF);
    resolve_sphere_hwall(&env->boulder, wy0, +1.0f, RESTITUTION_WALL, FRICTION_COEF);
    resolve_sphere_hwall(&env->boulder, wy1, -1.0f, RESTITUTION_WALL, FRICTION_COEF);

    // Goal: reward every tick boulder overlaps goal or approaches goal
    float dist_to_goal = hypotf(env->boulder.x - env->goal_x, env->boulder.y - env->goal_y);
    int on_goal = (dist_to_goal < GOAL_CENTER_RADIUS);
    float boulder_move_reward = 0.01f*(dist_to_goal_before - dist_to_goal);

    for (int i = 0; i < env->num_agents; i++){
        float r = on_goal ? 1.0f : env->moving_boulder[i] ? boulder_move_reward : 0.0f;
        env->rewards[i] += r;
        env->logs[i].episode_return += r;

    }
    if (on_goal) {
        env->goal_hold++;
    } else {
        env->goal_hold = 0;
    }

    env->tick++;
    int goal_held_for_full = (env->goal_hold >= GOAL_HOLD_TICKS);
    if(goal_held_for_full){
        env->score += 1.0f;
    }
    int done = goal_held_for_full || (env->tick >= MAX_TICKS);
    if (done) {
        for (int i = 0; i < env->num_agents; i++){
            env->terminals[i] = 1.0f;
            env->logs[i].episode_length = env->tick;
        }
        add_log(env);
        c_reset(env);
    }

    compute_observations(env);
}

// Rendering

Client* make_client(Boulder* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width  = env->width;
    client->height = env->height;
    InitWindow(env->width, env->height, "PufferLib Boulder");
    SetTargetFPS(60);
    client->sprites = LoadTexture("resources/shared/puffers.png");
    return client;
}

void close_client(Client* client) {
    UnloadTexture(client->sprites);
    CloseWindow();
    free(client);
}

void c_render(Boulder* env) {
    if (env->client == NULL) env->client = make_client(env);
    if (IsKeyDown(KEY_ESCAPE))   exit(0);
    if (IsKeyPressed(KEY_TAB))   ToggleFullscreen();

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    // Goal ring
    DrawCircle((int)env->goal_x, (int)env->goal_y, (int)GOAL_RADIUS,
               (Color){200, 200, 0, 60});
    DrawCircleLines((int)env->goal_x, (int)env->goal_y, (int)GOAL_RADIUS, YELLOW);

    // Boulder: large dark sphere with rotation indicator line
    DrawCircle((int)env->boulder.x, (int)env->boulder.y,
               (int)env->boulder.radius, (Color){120, 75, 30, 255});
    DrawCircleLines((int)env->boulder.x, (int)env->boulder.y,
                    (int)env->boulder.radius, (Color){180, 120, 60, 255});
    {
        float lx = env->boulder.x + cosf(env->boulder.angle) * env->boulder.radius * 0.75f;
        float ly = env->boulder.y + sinf(env->boulder.angle) * env->boulder.radius * 0.75f;
        DrawLine((int)env->boulder.x, (int)env->boulder.y,
                 (int)lx, (int)ly, (Color){220, 180, 100, 255});
    }

    // Agents: sprites from puffers.png, tinted per agent index, with velocity arrow
    // Sprite sheet row: y=576 facing left, y=608 facing right (32px rows)
    // Use sprite column 0 for all agents; differentiate via tint color
    for (int i = 0; i < env->num_agents; i++) {
        Entity* a = &env->agents[i];
        int src_x = 32 * i;                          // different column = different color
        int src_y = (a->vx >= 0.0f) ? 576 : 608;   // facing right or left
        DrawTexturePro(
            env->client->sprites,
            (Rectangle){ src_x, src_y, 32, 32 },
            (Rectangle){ a->x - 16, a->y - 16, 32, 32 },
            (Vector2){0, 0},
            0,
            WHITE
        );
        float speed = hypotf(a->vx, a->vy);
        if (speed > 20.0f) {
            float dx = a->vx / speed * (a->radius + 8.0f);
            float dy = a->vy / speed * (a->radius + 8.0f);
            DrawLine((int)a->x, (int)a->y,
                     (int)(a->x + dx), (int)(a->y + dy), WHITE);
        }
    }

    EndDrawing();
}
