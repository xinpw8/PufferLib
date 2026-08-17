#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define ACT_SIZES {3}
#define NUM_ATNS 1

#define MAX_OBSTACLES 9
#define OBS_SIZE (5 + 3 * 9)
#define MAX_TICKS 10000

#define PLAYER_HEIGHT 48
#define PLAYER_WIDTH 32
#define PLAYER_JUMP 10.0f
#define GRAVITY 0.5f

#define CACTUS_HEIGHT 24
#define CACTUS_WIDTH 24
#define CACTUS_Y 0

#define BIRD_HEIGHT 24
#define BIRD_WIDTH 48
#define BIRD_Y 44

#define NOOP 0
#define JUMP 1
#define CROUCH 2

struct Log {
    float perf;
    float score;
    float episode_length;
    float episode_return;
    float n;
};

typedef struct {
    Texture2D dinosaur_up;
    Texture2D dinosaur_down;
    Texture2D cactus;
    Texture2D bird;
} Client;

typedef struct {
    float x;
    float y;
    float y_velocity;
    float jump_strength;
    int ticks;
    float width;
    float height;
    float x_offset;
} DinoAgent;

enum ObstacleType {
    CACTUS,
    BIRD
};

typedef struct{
    float x;
    float y;
    float width;
    float height;
    enum ObstacleType type;
} Obstacle;

struct Env {
    /* Mandatory */
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    unsigned int rng;
    int num_agents;
    /* Not customizable */
    Client* client;
    DinoAgent* agent;
    Obstacle* obstacles;
    int num_obstacles;
    int speed;
    int spawn_rate;
    float gravity;
    int spawn_ticks;
    int max_obstacles;
    /* Customizable */
    int width;
    int height;
    int speed_init;
    int speed_max;
    int spawn_rate_min;
    int spawn_rate_max;
    int rate_increment_rate;
};
typedef Env Dinosaur;

Client* make_client(Dinosaur* env){
    Client* client = (Client*)calloc(1, sizeof(Client));

    InitWindow(env->width, env->height, "Pufferlib Dinosaur");
    SetTargetFPS(60);

    client->cactus = LoadTexture("resources/dino/cactus.png");
    client->bird = LoadTexture("resources/dino/bird.png");
    client->dinosaur_up = LoadTexture("resources/dino/dino.png");
    client->dinosaur_down = LoadTexture("resources/dino/dino_down.png");
    return client;
}

void c_init(Dinosaur* env){
    env->num_agents = 1;
    env->gravity = GRAVITY;
    env->spawn_rate = 1;
    env->spawn_ticks = 0;
    env->max_obstacles = MAX_OBSTACLES;

    env->agent = (DinoAgent*)calloc(1, sizeof(DinoAgent));
    env->agent->x = 0.0f + 2.0f * PLAYER_WIDTH;
    env->agent->y = 0;
    env->agent->jump_strength = PLAYER_JUMP;
    env->agent->width = PLAYER_WIDTH;
    env->agent->height = PLAYER_HEIGHT;
    env->obstacles = (Obstacle*)calloc(MAX_OBSTACLES, sizeof(Obstacle));
    env->num_obstacles = 0;

    env->num_agents = 1;
}

void compute_observations(Dinosaur* env) {
    float* obs = env->agents[0].observations;
    float max_jump = (env->agent->jump_strength * env->agent->jump_strength)
        / (2.0f * env->gravity);
    int i = 0;
    obs[i++] = env->agent->y / max_jump;
    obs[i++] = env->agent->width / (PLAYER_WIDTH * 2.0f);
    obs[i++] = env->agent->height / (float)PLAYER_HEIGHT;
    obs[i++] = (float)env->speed / 10.0f;
    obs[i++] = env->agent->ticks / 100.0f;

    for (int o = 0; o < env->max_obstacles; o++) {
        if (o < env->num_obstacles) {
            Obstacle* obstacle = &env->obstacles[o];
            obs[i++] = obstacle->type == CACTUS ? 0.0f : 1.0f;
            obs[i++] = (obstacle->x - env->agent->x) / env->width;
            obs[i++] = obstacle->y / (env->height / 2.0f);
        } else {
            obs[i++] = -1.0f;
            obs[i++] = 0.0f;
            obs[i++] = 0.0f;
        }
    }
}

void add_log(Dinosaur* env) {
    float score = env->agent->ticks / 100.0f - 1.0f;
    env->log.episode_return += score;
    env->log.episode_length += env->agent->ticks;
    env->log.score += score;
    env->log.perf += score;
    env->log.n += 1;
}

void puf_reset(Dinosaur* env){
    env->speed = env->speed_init;
    env->spawn_rate = 1;
    env->spawn_ticks = 0;

    env->agent->ticks = 0;
    env->agent->y_velocity = 0.0f;
    env->agent->y = 0.0f;
    env->agent->height = PLAYER_HEIGHT;
    env->agent->width = PLAYER_WIDTH;
    env->agent->x_offset = 0.0f;

    env->num_obstacles = 0;
    compute_observations(env);
}

void process_input(Dinosaur* env){
    int action = (int)env->agents[0].actions[0];
    switch(action){
        case NOOP:
            env->agent->y_velocity = -env->agent->jump_strength;
            env->agent->height = PLAYER_HEIGHT;
            env->agent->width = PLAYER_WIDTH;
            env->agent->x_offset = 0.0f;
            break;
        case CROUCH:
            env->agent->y_velocity = -env->agent->jump_strength;
            env->agent->height = PLAYER_HEIGHT / 2.f;
            env->agent->width = PLAYER_WIDTH * 2.0f;
            env->agent->x_offset = PLAYER_WIDTH;
            break;
        case JUMP:
            if(env->agent->y == 0.0f) env->agent->y_velocity = env->agent->jump_strength;
            env->agent->height = PLAYER_HEIGHT;
            env->agent->width = PLAYER_WIDTH;
            env->agent->x_offset = 0.0f;
            break;
    }
}

void process_gravity(Dinosaur* env){
    env->agent->y_velocity -= env->gravity;
    env->agent->y += env->agent->y_velocity;
    if(env->agent->y <= 0){
        env->agent->y = 0;
        env->agent->y_velocity = 0;
    }
}

void process_obstacle_collisions(Dinosaur* env){
    float agent_x_max = env->agent->x + env->agent->x_offset;
    float agent_x_min = agent_x_max - env->agent->width;
    float agent_y_min = env->agent->y;
    float agent_y_max = agent_y_min + env->agent->height;

    for(int o = 0; o < env->num_obstacles; o++){
        Obstacle* obstacle = &env->obstacles[o];
        obstacle->x -= env->speed;

        float obstacle_x_max = obstacle->x;
        float obstacle_x_min = obstacle_x_max - obstacle->width;
        float obstacle_y_min = obstacle->y;
        float obstacle_y_max = obstacle_y_min + obstacle->height;

        bool colliding_x = agent_x_min <= obstacle_x_max
            && agent_x_max >= obstacle_x_min;
        bool colliding_y = agent_y_min <= obstacle_y_max
            && agent_y_max >= obstacle_y_min;

        if (colliding_x && colliding_y) {
            *env->agents[0].terminals = 1.0f;
            *env->agents[0].rewards -= 1.0f;
            add_log(env);
            puf_reset(env);
            return;
        }

        if (obstacle->x < -obstacle->width) {
            for (int j = o; j < env->num_obstacles - 1; j++) {
                env->obstacles[j] = env->obstacles[j + 1];
            }
            env->num_obstacles--;
            o--;
        }
    }
}

void process_obstacle_spawns(Dinosaur* env){
    if (env->spawn_rate <= 0 || env->spawn_ticks % env->spawn_rate != 0) {
        return;
    }

    int spawn_rng = (int)(rand_r(&env->rng) % 4) + 1;
    if (spawn_rng < 4) {
        while (spawn_rng > 0
                && env->num_obstacles + spawn_rng > env->max_obstacles) {
            spawn_rng--;
        }
        for (int i = 0; i < spawn_rng; i++) {
            env->obstacles[env->num_obstacles++] = (Obstacle){
                .x = env->width + i * (CACTUS_WIDTH + 10.0f),
                .y = 0,
                .width = CACTUS_WIDTH,
                .height = CACTUS_HEIGHT,
                .type = CACTUS
            };
        }
    } else if (env->num_obstacles < env->max_obstacles) {
        env->obstacles[env->num_obstacles++] = (Obstacle){
            .x = env->width + BIRD_WIDTH + 10.0f,
            .y = BIRD_Y,
            .width = BIRD_WIDTH,
            .height = BIRD_HEIGHT,
            .type = BIRD
        };
    }

    int span = env->spawn_rate_max - env->spawn_rate_min;
    env->spawn_rate = (int)(rand_r(&env->rng) % (unsigned)span)
        + env->spawn_rate_min;
    env->spawn_rate = (int)(env->spawn_rate
        / (env->speed / (float)env->speed_init));
    if (env->spawn_rate < 1) {
        env->spawn_rate = 1;
    }
    env->spawn_ticks = 0;
}

// Hold Left Shift + up/down.
static void dino_human_controls(Dinosaur *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    env->agents[0].actions[0] = (float)NOOP;
    if (IsKeyDown(KEY_UP)) {
        env->agents[0].actions[0] = (float)JUMP;
    }
    if (IsKeyDown(KEY_DOWN)) {
        env->agents[0].actions[0] = (float)CROUCH;
    }
}

void puf_step(Dinosaur* env){
    dino_human_controls(env);
    env->agent->ticks += 1;
    env->spawn_ticks += 1;
    *env->agents[0].rewards = 0.01f;
    *env->agents[0].terminals = 0.0f;

    process_input(env);
    process_gravity(env);
    process_obstacle_collisions(env);

    if (env->agent->ticks >= MAX_TICKS) {
        *env->agents[0].terminals = 1.0f;
        add_log(env);
        puf_reset(env);
    }

    process_obstacle_spawns(env);

    if (env->agent->ticks > 0
            && env->agent->ticks % env->rate_increment_rate == 0) {
        if (env->speed < env->speed_max) {
            env->speed += 1;
        }
    }

    compute_observations(env);
}

void puf_render(Dinosaur* env){
    if(env->client == NULL) {
        env->client = make_client(env);
    }

    if(IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    dino_human_controls(env);

    BeginDrawing();

    ClearBackground((Color){255, 255, 255, 255});
    DrawRectangle(0, env->height/2.0f, env->width, env->height, (Color){95, 87, 79, 255});

    for(int o = 0; o < env->num_obstacles; o++){
        Obstacle* obstacle = &env->obstacles[o];
        Texture2D tex;
        tex = obstacle->type == CACTUS ? env->client->cactus : env->client->bird;
        DrawTexturePro(
            tex,
            (Rectangle){0, 0, obstacle->width, obstacle->height},
            (Rectangle){
                obstacle->x - obstacle->width,
                env->height/2.0f - obstacle->height - obstacle->y,
                obstacle->width,
                obstacle->height,
            },
            (Vector2){0, 0},
            0.0,
            WHITE
        );
    }

    Texture2D tex;
    int action = (int)env->agents[0].actions[0];
    switch(action){
        case NOOP:
        case JUMP:
            tex = env->client->dinosaur_up;
            break;
        case CROUCH:
            tex = env->client->dinosaur_down;
            break;
    }
    DrawTexturePro(
        tex,
        (Rectangle){0, 0, env->agent->width, env->agent->height},
        (Rectangle){
            env->agent->x - env->agent->width + env->agent->x_offset,
            env->height/2.0f - env->agent->height - env->agent->y,
            env->agent->width,
            env->agent->height
        },
        (Vector2){0, 0},
        0.0f,
        WHITE
    );

    EndDrawing();
    puf_web_vsync();
}

void puf_close(Dinosaur* env){
    free(env->agent);
    free(env->obstacles);
    if(env->client != NULL){
        UnloadTexture(env->client->cactus);
        UnloadTexture(env->client->dinosaur_up);
        UnloadTexture(env->client->dinosaur_down);
        CloseWindow();
        free(env->client);
    }
}

// --- Native trainer (pufferl) API ---
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->width = dict_get(kwargs, "width");
    env->height = dict_get(kwargs, "height");
    env->speed_init = dict_get(kwargs, "speed_init");
    env->speed_max = dict_get(kwargs, "speed_max");
    env->spawn_rate_min = dict_get(kwargs, "spawn_rate_min");
    env->spawn_rate_max = dict_get(kwargs, "spawn_rate_max");
    env->rate_increment_rate = dict_get(kwargs, "rate_increment_rate");
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    c_init(env);
}

