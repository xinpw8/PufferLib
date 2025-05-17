#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include "raylib.h"
#include "simplex.h"

const unsigned char NOOP = 0;
const unsigned char DOWN = 1;
const unsigned char UP = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char TARGET = 2;

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};
 
typedef struct Client Client;
typedef struct Terraform {
    Log log;
    Client* client;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int size;
    int tick;
    float* map;
} Terraform;

void init(Terraform* env) {
    env->map = calloc(env->size*env->size, sizeof(float));
}

void allocate(Terraform* env) {
    env->observations = (unsigned char*)calloc(env->size*env->size, sizeof(unsigned char));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    init(env);
}

void free_allocated(Terraform* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}

void add_log(Terraform* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

void perlin_noise(float* map, int width, int height,
        float base_frequency, int octaves, int offset_x, int offset_y) {
    float frequencies[octaves];
    for (int i = 0; i < octaves; i++) {
        frequencies[i] = base_frequency*pow(2, i);
    }

    float min_value = FLT_MAX;
    float max_value = FLT_MIN;
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*width + c;
            for (int oct = 0; oct < octaves; oct++) {
                float freq = frequencies[oct];
                map[adr] = noise2(freq*c + offset_x, freq*r + offset_y);
            }
            float val = map[adr];
            if (val < min_value) {
                min_value = val;
            }
            if (val > max_value) {
                max_value = val;
            }
        }
    }

    float scale = 1.0/(max_value - min_value);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*width + c;
            map[adr] = scale * (map[adr] - min_value);
        }
    }
}

void c_reset(Terraform* env) {
    memset(env->observations, 0, env->size*env->size*sizeof(unsigned char));
    env->tick = 0;

    perlin_noise(env->map, env->size, env->size, 1.0/64.0, 2, 0, 0);
}

void c_step(Terraform* env) {
    env->tick += 1;

    //int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0;

}

void c_close(Terraform* env) {
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};

typedef struct Client Client;
struct Client {
    Texture2D ball;
};

Client* make_client(Terraform* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    int px = 64*env->size;
    InitWindow(px, px, "PufferLib Terraform");
    SetTargetFPS(5);

    //client->agent = LoadTexture("resources/puffers_128.png");
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void c_render(Terraform* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    EndDrawing();
}
