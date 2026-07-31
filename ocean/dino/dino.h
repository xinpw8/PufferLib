#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"
#include "pufferenv.h"

#define ACT_SIZES {3}
#define NUM_ATNS 1
#if defined(from_float) && !defined(PRECISION_FLOAT)
typedef precision_t obs_t;
#else
typedef float obs_t;
#endif

#define MAX_OBSTACLES 9
#define OBS_SIZE (5 + 3 * 9)

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

void allocate(Dinosaur* env) {
	env->agents[0].observations = (obs_t*)calloc(OBS_SIZE, sizeof(obs_t));
	env->agents[0].actions = (float *)calloc(1, sizeof(float));
	env->agents[0].rewards = (float *)calloc(1, sizeof(float));
	env->agents[0].terminals = (float *)calloc(1, sizeof(float));
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    env->num_agents = 1;

}

void c_init(Dinosaur* env){
    allocate(env);
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

    env->num_agents = 1;
}

void compute_observations(Dinosaur* env) {
    int obs_idx = 0;

    memset(((obs_t*)env->agents[0].observations), 0, OBS_SIZE * sizeof(float));

    ((obs_t*)env->agents[0].observations)[obs_idx++] = env->agent->y / (pow(env->agent->jump_strength, 2) / (2 * env->gravity));
    ((obs_t*)env->agents[0].observations)[obs_idx++] = env->agent->width / (PLAYER_WIDTH * 2.0f);
    ((obs_t*)env->agents[0].observations)[obs_idx++] = env->agent->height / (float) PLAYER_HEIGHT;
    ((obs_t*)env->agents[0].observations)[obs_idx++] = (float) env->speed / 10.0f;
    ((obs_t*)env->agents[0].observations)[obs_idx++] = env->agent->ticks/100.0f;

    for(int o = 0; o < env->max_obstacles; o++){
        if (o < env->num_obstacles) {
            Obstacle* obstacle = &env->obstacles[o];
            ((obs_t*)env->agents[0].observations)[obs_idx++] = obstacle->type == CACTUS ? 0.0f : 1.0f;
            ((obs_t*)env->agents[0].observations)[obs_idx++] = ((obstacle->x - env->agent->x) / env->width);
            ((obs_t*)env->agents[0].observations)[obs_idx++] = obstacle->y/(env->height / 2.0f);
        } else {
            ((obs_t*)env->agents[0].observations)[obs_idx++] = -1.0f;
            ((obs_t*)env->agents[0].observations)[obs_idx++] = 0.0f;
            ((obs_t*)env->agents[0].observations)[obs_idx++] = 0.0f;
        }
    }
}

void puf_reset(Dinosaur* env){
    env->speed = env->speed_init;
    env->spawn_rate = 1;
    env->spawn_ticks = 0;

    env->agent->ticks = 0;
    env->agent->y_velocity = 0.0f;
    env->agent->y = 0.0f;

    env->num_obstacles = 0;
    if (env->obstacles != NULL) {
          free(env->obstacles);
          env->obstacles = NULL;
    }

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

        bool colliding_x = ((agent_x_max <= obstacle_x_max && agent_x_max >= obstacle_x_min) || (agent_x_min <= obstacle_x_max && agent_x_min >= obstacle_x_min));
        bool colliding_y = ((agent_y_max <= obstacle_y_max && agent_y_max >= obstacle_y_min) || (agent_y_min <= obstacle_y_max && agent_y_min >= obstacle_y_min));
        
        if(colliding_x && colliding_y){
            *env->agents[0].terminals = 1.0f;
            *env->agents[0].rewards -= 1.0f;

            env->log.episode_return += env->agent->ticks / 100.0f - 1.0f;
            env->log.episode_length += env->agent->ticks;
            env->log.score += env->agent->ticks / 100.0f - 1.0f;
            env->log.perf += env->agent->ticks / 100.0f - 1.0f;
            env->log.n += 1;
            puf_reset(env);
            return;
        }

        bool out_of_bounds = obstacle->x < 0 - obstacle->width;
        if(out_of_bounds){
            for(int j = o; j < env->num_obstacles - 1; j++){
                env->obstacles[j] = env->obstacles[j+1];
            }
            env->num_obstacles--;
            env->obstacles = (Obstacle*)realloc(env->obstacles, env->num_obstacles * sizeof(Obstacle));
            o--;
        }
    }
}

void process_obstacle_spawns(Dinosaur* env){
    bool time_to_spawn = env->spawn_ticks % env->spawn_rate == 0;
    
    if(time_to_spawn){
        int spawn_rng = rand_r(&env->rng) % 4 + 1;

        if(spawn_rng < 4){
            int max_loop = 0;
            while(spawn_rng + env->num_obstacles >= env->max_obstacles && max_loop < 100){
                spawn_rng = rand_r(&env->rng) % 3;
                max_loop++;
            }

            for(int i  = 0; i < spawn_rng; i++){
                env->num_obstacles++;
                env->obstacles = (Obstacle*)realloc(env->obstacles, env->num_obstacles * sizeof(Obstacle));
                env->obstacles[env->num_obstacles-1] = (Obstacle) {
                    .x = env->width + i * (CACTUS_WIDTH + 10.0f),
                    .y = 0,
                    .width = CACTUS_WIDTH,
                    .height = CACTUS_HEIGHT,
                    .type = CACTUS
                };
            }
        } else if (env->num_obstacles <= env->max_obstacles){
            env->num_obstacles++;
            env->obstacles = (Obstacle*)realloc(env->obstacles, env->num_obstacles * sizeof(Obstacle));
            env->obstacles[env->num_obstacles-1] = (Obstacle) {
                .x = env->width + BIRD_WIDTH + 10.0f,
                .y = BIRD_Y,
                .width = BIRD_WIDTH,
                .height = BIRD_HEIGHT,
                .type = BIRD
            };
        }
        env->spawn_rate = rand() % (env->spawn_rate_max - env->spawn_rate_min) + env->spawn_rate_min;
        env->spawn_rate = env->spawn_rate / (env->speed / (float) env->speed_init);
        env->spawn_ticks = 0;
    }
}

void puf_step(Dinosaur* env){
    env->agent->ticks += 1;
    env->spawn_ticks += 1;
    *env->agents[0].rewards += 0.01f;
    *env->agents[0].terminals = 0.0f;

    process_input(env);
    process_gravity(env);
    process_obstacle_collisions(env);
    process_obstacle_spawns(env);

    if(env->agent->ticks > 0 && env->agent->ticks % env->rate_increment_rate == 0){
        if(env->speed <= env->speed_max) env->speed+=1;
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

