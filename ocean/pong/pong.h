#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define ACT_SIZES {3}
#define OBS_SIZE 8
#define NUM_ATNS 1

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct Client Client;
struct Env {
    Client* client;
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    float paddle_yl;
    float paddle_yr;
    float ball_x;
    float ball_y;
    float ball_vx;
    float ball_vy;
    unsigned int score_l;
    unsigned int score_r;
    float width;
    float height;
    float paddle_width;
    float paddle_height;
    float ball_width;
    float ball_height;
    float paddle_speed;
    float ball_initial_speed_x;
    float ball_initial_speed_y;
    float ball_max_speed_y;
    float ball_speed_y_increment;
    unsigned int max_score;
    float min_paddle_y;
    float max_paddle_y;
    float paddle_dir;
    int tick;
    int n_bounces;
    int win;
    int frameskip;
    int continuous;
    unsigned int rng;
};
typedef Env Pong;

void init(Pong* env) {
    env->tick = 0;
    env->n_bounces = 0;
    env->win = 0;
    env->min_paddle_y = -env->paddle_height / 2;
    env->max_paddle_y = env->height - env->paddle_height / 2;
    env->paddle_dir = 0;
}

void add_log(Pong* env) {
    float score = (float)env->score_r - (float)env->score_l;
    env->log.episode_length += env->tick;
    env->log.episode_return += score;
    env->log.score += score;
    env->log.perf += (float)(env->score_r) / ((float)env->score_l + (float)env->score_r);
    env->log.n += 1;
}

void compute_observations(Pong* env) {
    obs_t* obs = env->agents[0].observations;
    obs[0] = (env->paddle_yl - env->min_paddle_y) / (env->max_paddle_y - env->min_paddle_y);
    obs[1] = (env->paddle_yr - env->min_paddle_y) / (env->max_paddle_y - env->min_paddle_y);
    obs[2] = env->ball_x / env->width;
    obs[3] = env->ball_y / env->height;
    obs[4] = (env->ball_vx + env->ball_initial_speed_x) / (2 * env->ball_initial_speed_x);
    obs[5] = (env->ball_vy + env->ball_max_speed_y) / (2 * env->ball_max_speed_y);
    obs[6] = env->score_l / env->max_score;
    obs[7] = env->score_r / env->max_score;
}

void reset_round(Pong* env) {
    env->paddle_yl = env->height / 2 - env->paddle_height / 2;
    env->paddle_yr = env->height / 2 - env->paddle_height / 2;
    env->ball_x = env->width / 5;
    env->ball_y = env->height / 2 - env->ball_height / 2;
    env->ball_vx = env->ball_initial_speed_x;
    env->ball_vy = ((rand_r(&env->rng) & 1) ? 1.0f : -1.0f)
        * env->ball_initial_speed_y;
    env->tick = 0;
    env->n_bounces = 0;
}

void puf_reset(Pong* env) {
    reset_round(env);
    env->score_l = 0;
    env->score_r = 0;
    compute_observations(env);
}

// Hold Left Shift + W/S or arrows (or wheel if continuous).
static void pong_human_controls(Pong *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    if (env->continuous) {
        float move = GetMouseWheelMove();
        env->agents[0].actions[0] = fmaxf(-1.0f, fminf(1.0f, move));
        return;
    }
    env->agents[0].actions[0] = 0.0f;
    if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
        env->agents[0].actions[0] = 1.0f;
    }
    if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
        env->agents[0].actions[0] = 2.0f;
    }
}

void puf_step(Pong* env) {
    env->tick += 1;
    env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 0;
    float* actions = env->agents[0].actions;
    float* rewards = env->agents[0].rewards;
    float* terminals = env->agents[0].terminals;

    if (env->continuous) {
        env->paddle_dir = actions[0];
    } else {
        float act = actions[0];
        env->paddle_dir = 0;
        if (act == 0.0) {
            env->paddle_dir = 0;
        } else if (act == 1.0) {
            env->paddle_dir = 1;
        } else if (act == 2.0) {
            env->paddle_dir = -1;
        }
    }

    for (int i = 0; i < env->frameskip; i++) {
        env->paddle_yr += env->paddle_speed * env->paddle_dir;

        float opp_paddle_delta = env->ball_y - (env->paddle_yl + env->paddle_height / 2);
        opp_paddle_delta = fminf(fmaxf(opp_paddle_delta, -env->paddle_speed), env->paddle_speed);
        env->paddle_yl += opp_paddle_delta;

        env->paddle_yr = fminf(fmaxf(env->paddle_yr, env->min_paddle_y), env->max_paddle_y);
        env->paddle_yl = fminf(fmaxf(env->paddle_yl, env->min_paddle_y), env->max_paddle_y);

        env->ball_x += env->ball_vx;
        env->ball_y += env->ball_vy;

        if (env->ball_y < 0 || env->ball_y + env->ball_height > env->height) {
            env->ball_vy = -env->ball_vy;
        }

        if (env->ball_x < 0) {
            if (env->ball_y + env->ball_height > env->paddle_yl &&
                env->ball_y < env->paddle_yl + env->paddle_height) {
                env->ball_vx = -env->ball_vx;
                env->n_bounces += 1;
            } else {
                env->win = 1;
                env->score_r += 1;
                rewards[0] = 1;
                if (env->score_r == env->max_score) {
                    terminals[0] = 1;
                    add_log(env);
                    puf_reset(env);
                    return;
                } else {
                    reset_round(env);
                    compute_observations(env);
                    return;
                }
            }
        }

        if (env->ball_x + env->ball_width > env->width) {
            if (env->ball_y + env->ball_height > env->paddle_yr &&
                env->ball_y < env->paddle_yr + env->paddle_height) {
                env->ball_vx = -env->ball_vx;
                env->n_bounces += 1;
                env->ball_vy += env->ball_speed_y_increment * env->paddle_dir;
                env->ball_vy = fminf(fmaxf(env->ball_vy, -env->ball_max_speed_y), env->ball_max_speed_y);
                if (fabsf(env->ball_vy) < 0.01) {
                    env->ball_vy = env->ball_speed_y_increment;
                }
            } else {
                env->win = 0;
                env->score_l += 1;
                rewards[0] = -1.0;
                if (env->score_l == env->max_score) {
                    terminals[0] = 1;
                    add_log(env);
                    puf_reset(env);
                    return;
                } else {
                    reset_round(env);
                    compute_observations(env);
                    return;
                }
            }

            env->ball_x = fminf(fmaxf(env->ball_x, 0), env->width - env->ball_width);
            env->ball_y = fminf(fmaxf(env->ball_y, 0), env->height - env->ball_height);
        }
        compute_observations(env);
    }
}

typedef struct Client Client;
struct Client {
    float width;
    float height;
    float paddle_width;
    float paddle_height;
    float ball_width;
    float ball_height;
    float x_pad;
    Color paddle_left_color;
    Color paddle_right_color;
    Color ball_color;
    Texture2D ball;
};

Client* make_client(Pong* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    client->paddle_width = env->paddle_width;
    client->paddle_height = env->paddle_height;
    client->ball_width = env->ball_width;
    client->ball_height = env->ball_height;
    client->x_pad = 3 * client->paddle_width;
    client->paddle_left_color = (Color){255, 0, 0, 255};
    client->paddle_right_color = (Color){0, 255, 255, 255};
    client->ball_color = (Color){255, 255, 255, 255};

    InitWindow(env->width + 2 * client->x_pad, env->height, "PufferLib Pong");
    SetTargetFPS(60 / (env->frameskip > 0 ? env->frameskip : 1));

    client->ball = LoadTexture("resources/shared/puffers_128.png");
    return client;
}

void close_client(Client* client) {
    UnloadTexture(client->ball);
    CloseWindow();
    free(client);
}

void puf_close(Pong* env) {
    if (env->client) {
        close_client(env->client);
    }
}

void puf_render(Pong* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    Client* client = env->client;

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    pong_human_controls(env);

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    DrawRectangle(
        client->x_pad - client->paddle_width,
        client->height - env->paddle_yl - client->paddle_height,
        client->paddle_width,
        client->paddle_height,
        client->paddle_left_color
    );

    DrawRectangle(
        client->width + client->x_pad,
        client->height - env->paddle_yr - client->paddle_height,
        client->paddle_width,
        client->paddle_height,
        client->paddle_right_color
    );

    DrawTexturePro(
        client->ball,
        (Rectangle){
            (env->ball_vx > 0) ? 0 : 128,
            0, 128, 128,
        },
        (Rectangle){
            client->x_pad + env->ball_x,
            client->height - env->ball_y - client->ball_height,
            client->ball_width,
            client->ball_height
        },
        (Vector2){0, 0},
        0,
        WHITE
    );

    DrawText(
        TextFormat("%i", env->score_l),
        client->width / 2 + client->x_pad - 50 - MeasureText(TextFormat("%i", env->score_l), 30) / 2,
        10, 30, (Color){0, 187, 187, 255}
    );
    DrawText(
        TextFormat("%i", env->score_r),
        client->width / 2 + client->x_pad + 50 - MeasureText(TextFormat("%i", env->score_r), 30) / 2,
        10, 30, (Color){0, 187, 187, 255}
    );

    EndDrawing();
    puf_web_vsync();
}

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
    env->paddle_width = dict_get(kwargs, "paddle_width");
    env->paddle_height = dict_get(kwargs, "paddle_height");
    env->ball_width = dict_get(kwargs, "ball_width");
    env->ball_height = dict_get(kwargs, "ball_height");
    env->paddle_speed = dict_get(kwargs, "paddle_speed");
    env->ball_initial_speed_x = dict_get(kwargs, "ball_initial_speed_x");
    env->ball_initial_speed_y = dict_get(kwargs, "ball_initial_speed_y");
    env->ball_max_speed_y = dict_get(kwargs, "ball_max_speed_y");
    env->ball_speed_y_increment = dict_get(kwargs, "ball_speed_y_increment");
    env->max_score = dict_get(kwargs, "max_score");
    env->frameskip = dict_get(kwargs, "frameskip");
    env->continuous = dict_get(kwargs, "continuous");
    env->agents[0].policy = 0;
    init(env);
}
