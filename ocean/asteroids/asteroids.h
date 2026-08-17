#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef float obs_t;
#include "pufferenv.h"

#define MAX_PARTICLES 10
#define MAX_ASTEROIDS 20

// Entity encoder layout (fixed slots, no distance sort):
// self 4: sin(angle), cos(angle), fwd_vel/VEL_NORM, right_vel/VEL_NORM
// each of MAX_ASTEROIDS (ship frame, Euclidean wrap):
//   rel_fwd/size, rel_right/size, vel_fwd, vel_right, radius/40.
// Empty / off-playfield slots are zeros (radius==0).
#define ACT_SIZES {4}
#define OBS_SIZE (4 + MAX_ASTEROIDS * 5)
#define NUM_ATNS 1

#ifdef PUFFERCPU_EVAL_MAIN
#define PUF_ASTEROIDS_NET 1
#include "asteroids_net.h"
#endif

const unsigned char FORWARD = 0;
const unsigned char TURN_LEFT = 1;
const unsigned char TURN_RIGHT = 2;
const unsigned char SHOOT = 3;

const float FRICTION = 0.95f;
const float SPEED = 0.6f;
const float PARTICLE_SPEED = 7.0f;
const float ROTATION_SPEED = 0.1f;
const float ASTEROID_SPEED = 3.0f;
const int SHOOT_DELAY = 18;

const int MAX_TICK = 3600;

const int DEBUG = 0;

// for render only game over state
static int global_game_over_timer = 0;
static int global_game_over_started = 0;
static int global_render_flag = 0;

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct {
    Vector2 position;
    Vector2 velocity;
} Particle;

typedef struct {
    Vector2 position;
    Vector2 velocity;
    int radius;
    int radius_sq;
    Vector2 shape[12];
    int num_vertices;
} Asteroid;

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    int size;
    Vector2 player_position;
    Vector2 player_vel;
    float player_angle;
    int player_radius;
    int thruster_on;
    Particle particles[MAX_PARTICLES];
    int particle_index;
    Asteroid asteroids[MAX_ASTEROIDS];
    int asteroid_index;
    int last_shot;
    int tick;
    int score;
    float episode_return;
    int frameskip;
    unsigned int rng;
};
typedef Env Asteroids;

static float random_float(unsigned int *rng, float low, float high) {
    return low + (high - low) * ((float)rand_r(rng) / (float)RAND_MAX);
}

void generate_asteroid_shape(Asteroid *as, unsigned int *rng) {
    as->num_vertices = 8 + (as->radius / 10);

    for (int v = 0; v < as->num_vertices; v++) {
        float angle = (2.0f * PI * v) / as->num_vertices;
        float radius_variation =
            as->radius * (0.7f + 0.6f * random_float(rng, 0.0f, 1.0f));
        as->shape[v].x = cosf(angle) * radius_variation;
        as->shape[v].y = sinf(angle) * radius_variation;
    }
}

float clamp(float val, float low, float high) {
    return fmin(fmax(val, low), high);
}

Vector2 rotate_vector(Vector2 point, Vector2 center, float angle) {
    float s = sinf(angle);
    float c = cosf(angle);

    // Translate point back to origin:
    point.x -= center.x;
    point.y -= center.y;

    // Rotate point
    float xnew = point.x * c - point.y * s;
    float ynew = point.x * s + point.y * c;

    // Translate point back:
    point.x = xnew + center.x;
    point.y = ynew + center.y;
    return point;
}

Vector2 get_direction_vector(Asteroids *env) {
    float px = env->player_position.x;
    float py = env->player_position.y;
    Vector2 dir = (Vector2){px, py - 1};
    dir = rotate_vector(dir, env->player_position, env->player_angle);
    return (Vector2){dir.x - px, dir.y - py};
}

static float wrap_pos(float x, float size) {
    if (x < 0) {
        x += size;
    }
    if (x > size) {
        x -= size;
    }
    return x;
}

static float wrap_delta(float d, float size) {
    if (d > size * 0.5f) {
        d -= size;
    }
    if (d < -size * 0.5f) {
        d += size;
    }
    return d;
}

void move_particles(Asteroids *env) {
    Particle p;
    float size = (float)env->size;
    for (int i = 0; i < MAX_PARTICLES; i++) {
        p = env->particles[i];
        if (p.position.x == 0 && p.position.y == 0) {
            continue;
        }
        p.position.x = wrap_pos(
            p.position.x + p.velocity.x * PARTICLE_SPEED, size);
        p.position.y = wrap_pos(
            p.position.y + p.velocity.y * PARTICLE_SPEED, size);
        env->particles[i] = p;
    }
}

void move_asteroids(Asteroids *env) {
    Asteroid *as;
    float size = (float)env->size;
    for (int i = 0; i < MAX_ASTEROIDS; i++) {
        as = &env->asteroids[i];
        if (as->radius == 0) {
            continue;
        }

        as->position.x = wrap_pos(
            as->position.x + as->velocity.x * ASTEROID_SPEED, size);
        as->position.y = wrap_pos(
            as->position.y + as->velocity.y * ASTEROID_SPEED, size);
    }
}

Vector2 angle_to_vector(float angle) {
    Vector2 v;
    v.x = cosf(angle);
    v.y = sinf(angle);
    return v;
}

void spawn_asteroids(Asteroids *env) {
    float px, py;
    float angle;
    if (rand_r(&env->rng) % 10 == 0) {
        switch (rand_r(&env->rng) % 4) {
            case 0:
                // left edge
                px = 0;
                py = rand_r(&env->rng) % env->size;
                angle = random_float(&env->rng, -PI / 2, PI / 2);
                break;
            case 1:
                // right edge
                px = env->size;
                py = rand_r(&env->rng) % env->size;
                angle = random_float(&env->rng, PI / 2, 3 * PI / 2);
                break;
            case 2:
                // top edge
                px = rand_r(&env->rng) % env->size;
                py = 0;
                angle = random_float(&env->rng, PI, 2 * PI);
                break;
            default:
                // bottom edge
                px = rand_r(&env->rng) % env->size;
                py = env->size;
                angle = random_float(&env->rng, 0, PI);
                break;
        }

        Vector2 direction = angle_to_vector(angle);
        Vector2 start_pos = (Vector2){px, py};
        Asteroid as;
        switch (rand_r(&env->rng) % 3) {
            case 0:
                // small
                as = (Asteroid){start_pos, direction, 10, 100};
                break;
            case 1:
                // medium
                as = (Asteroid){start_pos, direction, 20, 400};
                break;
            default:
                // big
                as = (Asteroid){start_pos, direction, 40, 1600};
                break;
        }
        env->asteroid_index = (env->asteroid_index + 1) % MAX_ASTEROIDS;
        env->asteroids[env->asteroid_index] = as;
        if (global_render_flag) {
            generate_asteroid_shape(
                &env->asteroids[env->asteroid_index], &env->rng);
        }
    }
}

int particle_asteroid_collision(Asteroids *env, Particle *p, Asteroid *as) {
    float size = (float)env->size;
    float dx = wrap_delta(p->position.x - as->position.x, size);
    float dy = wrap_delta(p->position.y - as->position.y, size);
    return as->radius_sq > dx * dx + dy * dy;
}

void split_asteroid(Asteroids *env, Asteroid *as) {
    int new_radius = as->radius == 40 ? 20 : 10;

    float original_angle = atan2f(as->velocity.y, as->velocity.x);

    float offset1 = random_float(&env->rng, -PI / 4, PI / 4);
    float offset2 = random_float(&env->rng, -PI / 4, PI / 4);

    float angle1 = original_angle + offset1;
    float angle2 = original_angle + offset2;

    Vector2 direction1 = angle_to_vector(angle1);
    Vector2 direction2 = angle_to_vector(angle2);

    float len1 = sqrtf(
        direction1.x * direction1.x + direction1.y * direction1.y);
    float len2 = sqrtf(
        direction2.x * direction2.x + direction2.y * direction2.y);
    if (len1 > 0) {
        direction1.x /= len1;
        direction1.y /= len1;
    }
    if (len2 > 0) {
        direction2.x /= len2;
        direction2.y /= len2;
    }

    Vector2 start_pos = (Vector2){as->position.x, as->position.y};

    int new_index1 = (env->asteroid_index + 1) % MAX_ASTEROIDS;
    int new_index2 = (new_index1 + 1) % MAX_ASTEROIDS;

    as->position = start_pos;
    as->velocity = direction1;
    as->radius = new_radius;
    as->radius_sq = new_radius * new_radius;
    env->asteroids[new_index1] = (Asteroid){
        start_pos, direction2, new_radius, new_radius * new_radius};
    env->asteroid_index = new_index2;

    generate_asteroid_shape(as, &env->rng);
    generate_asteroid_shape(&env->asteroids[new_index1], &env->rng);
}

void check_particle_asteroid_collision(Asteroids *env) {
    Particle *p;
    Asteroid *as;
    for (int i = 0; i < MAX_PARTICLES; i++) {
        p = &env->particles[i];
        if (p->position.x == 0 && p->position.y == 0) {
            continue;
        }

        for (int j = 0; j < MAX_ASTEROIDS; j++) {
            as = &env->asteroids[j];
            if (as->radius == 0) {
                continue;
            }

            if (particle_asteroid_collision(env, p, as)) {
                memset(p, 0, sizeof(*p));
                env->score += 1;
                env->agents[0].rewards[0] += 1.0f;

                switch (as->radius) {
                    case 10:
                        memset(as, 0, sizeof(*as));
                        break;
                    case 20:
                        split_asteroid(env, as);
                        break;
                    default:
                        split_asteroid(env, as);
                        break;
                }
                break;
            }
        }
    }
}

void check_player_asteroid_collision(Asteroids *env) {
    float min_dist;
    float dx, dy;
    Asteroid *as;
    for (int i = 0; i < MAX_ASTEROIDS; i++) {
        as = &env->asteroids[i];
        if (as->radius == 0) {
            continue;
        }

        min_dist = env->player_radius + as->radius;
        dx = wrap_delta(
            env->player_position.x - as->position.x, (float)env->size);
        dy = wrap_delta(
            env->player_position.y - as->position.y, (float)env->size);
        if (min_dist * min_dist > dx * dx + dy * dy) {
            env->agents[0].terminals[0] = 1;
            env->agents[0].rewards[0] += -1.0f;
            return;
        }
    }
}

void compute_observations(Asteroids *env) {
    float*obs = env->agents[0].observations;
    int idx = 0;
    const float vel_norm = SPEED / (1.0f - FRICTION);
    const float size = (float)env->size;

    // Facing matches get_direction_vector: rotate (0,-1) by player_angle.
    float c = cosf(env->player_angle);
    float s = sinf(env->player_angle);
    float fx = s;
    float fy = -c;
    float rx = c;
    float ry = s;

    obs[idx++] = s;
    obs[idx++] = c;
    float pvx = env->player_vel.x;
    float pvy = env->player_vel.y;
    obs[idx++] = (pvx * fx + pvy * fy) / vel_norm;
    obs[idx++] = (pvx * rx + pvy * ry) / vel_norm;

    float px = env->player_position.x;
    float py = env->player_position.y;
    for (int i = 0; i < MAX_ASTEROIDS; i++) {
        Asteroid *as = &env->asteroids[i];
        if (as->radius == 0) {
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
            continue;
        }
        float dx = wrap_delta(as->position.x - px, size);
        float dy = wrap_delta(as->position.y - py, size);
        obs[idx++] = (dx * fx + dy * fy) / size;
        obs[idx++] = (dx * rx + dy * ry) / size;
        obs[idx++] = as->velocity.x * fx + as->velocity.y * fy;
        obs[idx++] = as->velocity.x * rx + as->velocity.y * ry;
        obs[idx++] = (float)as->radius / 40.0f;
    }
}

void add_log(Asteroids *env) {
    env->log.perf += env->score / 100.0f;
    env->log.score += env->score;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return;
    env->log.n++;
}

void puf_reset(Asteroids *env) {
    env->player_position = (Vector2){env->size / 2.0f, env->size / 2.0f};
    env->player_angle = 0.0f;
    env->player_radius = 12;
    env->player_vel = (Vector2){0, 0};
    env->thruster_on = 0;
    memset(env->particles, 0, sizeof(Particle) * MAX_PARTICLES);
    memset(env->asteroids, 0, sizeof(Asteroid) * MAX_ASTEROIDS);
    env->particle_index = 0;
    env->asteroid_index = 0;
    env->tick = 0;
    env->score = 0;
    env->episode_return = 0;
    env->last_shot = 0;
    compute_observations(env);
}

void step_frame(Asteroids *env, int action) {
    // slow down each step
    env->player_vel.x *= FRICTION;
    env->player_vel.y *= FRICTION;

    Vector2 dir = get_direction_vector(env);

    if (action == TURN_LEFT) {
        env->player_angle -= ROTATION_SPEED;
    }
    if (action == TURN_RIGHT) {
        env->player_angle += ROTATION_SPEED;
    }
    if (action == FORWARD) {
        env->player_vel.x += dir.x * SPEED;
        env->player_vel.y += dir.y * SPEED;
        env->thruster_on = 1;
    }

    int elapsed = env->tick - env->last_shot;

    if (action == SHOOT && elapsed >= SHOOT_DELAY) {
        env->last_shot = env->tick;
        env->particle_index = (env->particle_index + 1) % MAX_PARTICLES;
        Vector2 start_pos = (Vector2){
            env->player_position.x + 20 * dir.x,
            env->player_position.y + 20 * dir.y};
        env->particles[env->particle_index] = (Particle){start_pos, dir};
    }

    // Wrap before collisions so edge hits match wrap obs.
    env->player_position.x = wrap_pos(
        env->player_position.x + env->player_vel.x, (float)env->size);
    env->player_position.y = wrap_pos(
        env->player_position.y + env->player_vel.y, (float)env->size);

    move_particles(env);
    spawn_asteroids(env);
    move_asteroids(env);
    check_particle_asteroid_collision(env);
    check_player_asteroid_collision(env);
}

// Hold Left Shift + WASD/arrows/space. CPU eval is forward→step→render, so
// apply here and at the start of puf_step so the net does not win the frame.
static void asteroids_human_controls(Asteroids *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP)) {
        env->agents[0].actions[0] = FORWARD;
    } else if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) {
        env->agents[0].actions[0] = TURN_LEFT;
    } else if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) {
        env->agents[0].actions[0] = TURN_RIGHT;
    } else if (IsKeyDown(KEY_SPACE)) {
        env->agents[0].actions[0] = SHOOT;
    } else {
        env->agents[0].actions[0] = -1;
    }
}

void puf_step(Asteroids *env) {
    asteroids_human_controls(env);
    env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 0;
    env->thruster_on = 0;

    // only when rendering
    if (global_game_over_timer > 0) {
        return;
    }

    int action = (int)env->agents[0].actions[0];
    for (int i = 0; i < env->frameskip; i++) {
        env->tick += 1;
        step_frame(env, action);
    }

    env->episode_return += env->agents[0].rewards[0];
    if (env->agents[0].terminals[0] == 1 || env->tick > MAX_TICK) {
        env->agents[0].terminals[0] = 1;
        add_log(env);
        puf_reset(env);
        return;
    }

    compute_observations(env);
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

void draw_player(Asteroids *env) {
    if (global_game_over_timer > 0) {
        return;
    }

    float px = env->player_position.x;
    float py = env->player_position.y;

    if (DEBUG) {
        DrawPixel(px, py, RED);
        Vector2 dir = get_direction_vector(env);
        dir = (Vector2){dir.x * 10.0f, dir.y * 10.f};
        Vector2 t = (Vector2){dir.x + px, dir.y + py};
        DrawLineV(env->player_position, t, RED);
        DrawCircleLines(px, py, env->player_radius, RED);
    }

    Vector2 ps[8];

    // ship
    ps[0] = (Vector2){px - 10, py + 10};
    ps[1] = (Vector2){px + 10, py + 10};
    ps[2] = (Vector2){px, py - 20};
    ps[3] = (Vector2){px - 9, py + 6};
    ps[4] = (Vector2){px + 9, py + 6};
    ps[5] = (Vector2){px - 5, py + 6};
    ps[6] = (Vector2){px + 5, py + 6};
    ps[7] = (Vector2){px, py + 14};

    for (int i = 0; i < 8; i++) {
        ps[i] = rotate_vector(ps[i], env->player_position, env->player_angle);
    }

    DrawLineV(ps[0], ps[2], PUFF_RED);
    DrawLineV(ps[1], ps[2], PUFF_RED);

    DrawLineV(ps[3], ps[4], PUFF_RED);

    if (env->thruster_on) {
        DrawLineV(ps[5], ps[7], PUFF_RED);
        DrawLineV(ps[6], ps[7], PUFF_RED);
    }
}

void draw_particles(Asteroids *env) {
    for (int i = 0; i < MAX_PARTICLES; i++) {
        DrawCircle(
            env->particles[i].position.x,
            env->particles[i].position.y,
            2,
            PUFF_RED);
    }
}

void draw_asteroids(Asteroids *env) {
    Asteroid as;
    for (int i = 0; i < MAX_ASTEROIDS; i++) {
        as = env->asteroids[i];
        if (as.radius == 0) {
            continue;
        }

        if (DEBUG) {
            DrawCircleLines(as.position.x, as.position.y, as.radius, RED);
        }

        for (int v = 0; v < as.num_vertices; v++) {
            int next_v = (v + 1) % as.num_vertices;
            Vector2 pos1 = {
                as.position.x + as.shape[v].x,
                as.position.y + as.shape[v].y};
            Vector2 pos2 = {
                as.position.x + as.shape[next_v].x,
                as.position.y + as.shape[next_v].y};
            DrawLineV(pos1, pos2, PUFF_CYAN);
        }
    }
}

void puf_render(Asteroids *env) {
    if (!IsWindowReady()) {
        InitWindow(env->size, env->size, "PufferLib Asteroids");
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        SetTargetFPS(60);
        global_render_flag = 1;
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    asteroids_human_controls(env);

    if (env->agents[0].terminals[0] == 1 && !global_game_over_started) {
        global_game_over_started = 1;
        global_game_over_timer = 120;
    }

    if (global_game_over_timer > 0) {
        global_game_over_timer--;
    } else {
        global_game_over_started = 0;
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    draw_player(env);
    draw_particles(env);
    draw_asteroids(env);

    DrawText(TextFormat("Score: %d", env->score), 10, 10, 20, PUFF_WHITE);
    DrawText(
        TextFormat("%d s", (int)(env->tick / 60)),
        env->size - 40,
        10,
        20,
        PUFF_WHITE);

    if (global_game_over_timer > 0) {
        const char *game_over_text = "GAME OVER";
        int text_width = MeasureText(game_over_text, 40);
        int x = (env->size - text_width) / 2;
        int y = env->size / 2 - 20;

        float alpha = (float)global_game_over_timer / 120.0f;
        int alpha_value = (int)(alpha * 255);

        Color text_color = ColorAlpha(PUFF_RED, alpha_value);
        DrawTextEx(
            GetFontDefault(),
            game_over_text,
            (Vector2){x, y},
            40,
            2,
            text_color);
    }

    EndDrawing();
    puf_web_vsync();
}

void puf_close(Asteroids *env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

void puf_init(Env *env, Dict *kwargs) {
    env->num_agents = 1;
    env->size = dict_get(kwargs, "size");
    env->frameskip = dict_get(kwargs, "frameskip");
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
}

void puf_log(Log *log, Dict *out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}
