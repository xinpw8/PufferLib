#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define ACT_SIZES {6}
#define OBS_SIZE 10
#define NUM_ATNS 1

#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480
#define PLAYER_MAX_LIVES 10
#define ENEMY_SPAWN_Y 50
#define ENEMY_SPAWN_X -30
#define INIT_BULLET_SPEED 3.0f
#define MAX_SCORE (5 * PLAYER_MAX_LIVES)
#define BULLET_SPEED (INIT_BULLET_SPEED * SPEED_SCALE)

// Reward ablations. CLI: env.reward_mode=N
#define REWARD_STOCK 0
#define REWARD_SCORE_ONLY 1
#define REWARD_NO_RIGHT_WIPE 2
#define REWARD_NO_GATES 3
#define REWARD_KILL_PLUS_HIT 4
#define REWARD_NO_FIRE_BONUS 5
#define REWARD_CLOSENESS_ONLY 6

static const float SPEED_SCALE = 4.0f;
static const int ENEMY_WIDTH = 16;
static const int ENEMY_HEIGHT = 17;
static const int PLAYER_WIDTH = 17;
static const int PLAYER_HEIGHT = 17;
static const int PLAYER_BULLET_WIDTH = 17;
static const int PLAYER_BULLET_HEIGHT = 6;

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float lives;
    float vertical_closeness_rew;
    float fired_bullet_rew;
    float kill_streak;
    float hit_enemy_with_bullet_rew;
    float avg_score_difference;
    float n;
};

typedef struct Bullet {
    float x;
    float y;
    bool active;
} Bullet;

typedef struct Enemy {
    float x;
    float y;
    float enemy_speed;
    bool active;
    bool attacking;
    int crossed_screen;
    Bullet bullet;
} Enemy;

typedef struct Player {
    float x;
    float y;
    float player_speed;
    int score;
    int lives;
    Bullet bullet;
    bool bullet_fired;
    bool player_stuck;
} Player;

typedef struct Client {
    Texture2D player_texture;
    Texture2D enemy_texture;
    Texture2D player_bullet_texture;
    Texture2D enemy_bullet_texture;
    Texture2D explosion_texture;
} Client;

struct Env {
    Client* client;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int reset_count;
    int num_obs;
    bool game_over;
    int tick;
    int player_explosion_timer;
    int enemy_explosion_timer;
    int kill_streak;
    int enemy_respawns;
    Player player;
    Enemy enemy;
    int num_agents;
    int reward_mode;
    float episode_return;
    Log log;
    unsigned int rng;
};
typedef Env Blastar;

void add_log(Blastar* env) {
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return;
    env->log.lives += env->player.lives;
    env->log.score += env->player.score;
    env->log.perf += env->player.score / (float)MAX_SCORE;
    env->log.kill_streak += env->kill_streak;
    env->log.n += 1;
}

static inline void scale_speeds(Blastar* env) {
    env->player.player_speed *= SPEED_SCALE;
    env->enemy.enemy_speed *= SPEED_SCALE;
}

void puf_reset(Blastar* env) {
    env->game_over = false;
    env->tick = 0;
    env->episode_return = 0.0f;
    env->player_explosion_timer = 0;
    env->enemy_explosion_timer = 0;
    env->player.player_speed = 2.0f;
    env->enemy.enemy_speed = 1.0f;
    scale_speeds(env);
    env->player.x = (float)(rand() % (SCREEN_WIDTH - PLAYER_WIDTH));
    env->player.y = (float)(rand() % (SCREEN_HEIGHT - PLAYER_HEIGHT));
    env->player.score = 0;
    env->player.lives = PLAYER_MAX_LIVES;
    env->player.bullet_fired = false;
    env->player.player_stuck = false;
    env->player.bullet.active = false;
    env->player.bullet.x = env->player.x;
    env->player.bullet.y = env->player.y;
    env->kill_streak = 0;
    env->enemy.x = ENEMY_SPAWN_X;
    env->enemy.y = ENEMY_SPAWN_Y;
    env->enemy.active = true;
    env->enemy.attacking = false;
    if (env->reset_count < 1) {
        env->enemy_respawns = 0;
        env->enemy.crossed_screen = 0;
    }
    env->enemy.bullet.active = false;
    env->enemy.bullet.x = env->enemy.x;
    env->enemy.bullet.y = env->enemy.y;
    env->reset_count++;
}

void puf_close(Blastar* env) {
}

void init(Blastar* env, int num_obs) {
    env->reset_count = 0;
    env->num_obs = num_obs;
    env->log = (Log){0};
    puf_reset(env);
}

static inline void calculate_center(float x, float y, int width, int height, float* center_x, float* center_y) {
    *center_x = x + width / 2.0f;
    *center_y = y + height / 2.0f;
}

void compute_observations(Blastar* env) {
    float* obs = env->agents[0].observations;
    memset(obs, 0, env->num_obs * sizeof(float));
    obs[0] = env->player.x / SCREEN_WIDTH;
    obs[1] = env->player.y / SCREEN_HEIGHT;
    obs[2] = env->enemy.x / SCREEN_WIDTH;
    obs[3] = env->enemy.y / SCREEN_HEIGHT;
    if (env->player.bullet.active) {
        obs[4] = env->player.bullet.x / SCREEN_WIDTH;
        obs[5] = env->player.bullet.y / SCREEN_HEIGHT;
        obs[6] = 1.0f;
    }
    if (env->enemy.bullet.active) {
        obs[7] = env->enemy.bullet.x / SCREEN_WIDTH;
        obs[8] = env->enemy.bullet.y / SCREEN_HEIGHT;
        obs[9] = 1.0f;
    }
}

bool check_collision(float x1, float y1, float w1, float h1, float x2, float y2, float w2, float h2) {
    if (x1 < x2 + w2 && x1 + w1 > x2 && y1 < y2 + h2 && y1 + h1 > y2) {
        return true;
    }
    return false;
}

// Hold Left Shift + WASD/arrows/space.
static void blastar_human_controls(Blastar *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        env->agents[0].actions[0] = 1;
    } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        env->agents[0].actions[0] = 2;
    } else if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
        env->agents[0].actions[0] = 3;
    } else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
        env->agents[0].actions[0] = 4;
    } else if (IsKeyDown(KEY_SPACE)) {
        env->agents[0].actions[0] = 5;
    } else {
        env->agents[0].actions[0] = 0;
    }
}

void puf_step(Blastar* env) {
    blastar_human_controls(env);
    if (env->game_over) {
        env->agents[0].terminals[0] = 1;
        add_log(env);
        puf_reset(env);
        return;
    }

    env->tick++;
    float rew = 0.0f;
    env->agents[0].rewards[0] = rew;
    float fired_bullet_rew = 0.0f;
    float vertical_closeness_rew = 0.0f;
    float hit_enemy_with_bullet_rew = 0.0f;
    int crossed_screen = 0;
    int scored_kill = 0;
    int action = env->agents[0].actions[0];

    if (env->player_explosion_timer > 0) {
        env->player_explosion_timer--;
        env->kill_streak = 0;
        if (env->player_explosion_timer == 0) {
            env->player.player_stuck = false;
            env->player.bullet.active = false;
        }
        compute_observations(env);
        return;
    }

    if (env->enemy_explosion_timer > 0) {
        env->enemy_explosion_timer--;
        if (env->enemy_explosion_timer == 0) {
            env->enemy.crossed_screen = 0;
            float respawn_bias = 0.1f;
            if ((float)rand() / (float)RAND_MAX > respawn_bias) {
                env->enemy.x = -ENEMY_WIDTH;
                env->enemy.y = rand() % (SCREEN_HEIGHT - ENEMY_HEIGHT);
                env->enemy_respawns += 1;
            }
            env->enemy.active = true;
            env->enemy.attacking = false;
        }
        compute_observations(env);
        return;
    }

    if (env->enemy.y > (SCREEN_HEIGHT - (ENEMY_HEIGHT * 3.5f))) {
        env->enemy.y = (SCREEN_HEIGHT - (ENEMY_HEIGHT * 3.5f));
    }

    if (!env->player.player_stuck) {
        if (action == 1 && env->player.x > 0) env->player.x -= env->player.player_speed;
        if (action == 2 && env->player.x < SCREEN_WIDTH - PLAYER_WIDTH) env->player.x += env->player.player_speed;
        if (action == 3 && env->player.y > 0) env->player.y -= env->player.player_speed;
        if (action == 4 && env->player.y < SCREEN_HEIGHT - PLAYER_HEIGHT) env->player.y += env->player.player_speed;
    }

    if (action == 5 && (!env->enemy.bullet.active)) {
        if (env->player.bullet.active) {
            env->player.bullet.active = false;
        } else if (env->reward_mode != REWARD_NO_FIRE_BONUS) {
            fired_bullet_rew += 0.0005f;
        }
        env->player.bullet.active = true;
        env->player.bullet.x = env->player.x + PLAYER_WIDTH / 2 - PLAYER_BULLET_WIDTH / 2;
        env->player.bullet.y = env->player.y;
    }

    if (env->player.bullet.active) {
        env->player.bullet.y -= BULLET_SPEED;
        if (env->player.bullet.y < 0) {
            env->player.bullet.active = false;
        }
    }

    float player_center_x;
    float enemy_center_x;
    float dummy;
    calculate_center(env->player.x, env->player.y, PLAYER_WIDTH, PLAYER_HEIGHT, &player_center_x, &dummy);
    calculate_center(env->enemy.x, env->enemy.y, ENEMY_WIDTH, ENEMY_HEIGHT, &enemy_center_x, &dummy);

    if (!env->enemy.attacking) {
        env->enemy.x += env->enemy.enemy_speed;
        if (env->enemy.x > SCREEN_WIDTH) {
            env->enemy.x = -ENEMY_WIDTH;
            crossed_screen += 1;
        }
    }

    if (fabs(player_center_x - enemy_center_x) < SPEED_SCALE &&
        !env->enemy.attacking && env->enemy.active &&
        env->enemy.y < env->player.y - (ENEMY_HEIGHT / 2)) {
        if (rand() % 2 == 0) {
            env->enemy.attacking = true;
            if (!env->enemy.bullet.active) {
                env->enemy.bullet.active = true;
                calculate_center(env->enemy.x, env->enemy.y, ENEMY_WIDTH, ENEMY_HEIGHT, &enemy_center_x, &dummy);
                env->enemy.bullet.x = enemy_center_x - 5.0f;
                env->enemy.bullet.y = env->enemy.y + ENEMY_HEIGHT;
                env->player.bullet.active = false;
                env->player.player_stuck = true;
            }
        } else {
            env->enemy.attacking = false;
            env->enemy.x += env->enemy.enemy_speed;
        }
    }

    if (env->enemy.bullet.active) {
        env->enemy.bullet.y += BULLET_SPEED;
        if (env->enemy.bullet.y > SCREEN_HEIGHT) {
            env->enemy.bullet.active = false;
            env->player.player_stuck = false;
            env->enemy.attacking = false;
        }
    }

    if (check_collision(env->player.x, env->player.y, PLAYER_WIDTH, PLAYER_HEIGHT, 
                        env->enemy.x, env->enemy.y, ENEMY_WIDTH, ENEMY_HEIGHT)) {
        env->player.lives--;
        env->enemy.active = false;
        env->enemy_explosion_timer = 30;
        env->enemy.x = -ENEMY_WIDTH;
        env->enemy.y = rand() % (SCREEN_HEIGHT - ENEMY_HEIGHT);
        env->player_explosion_timer = 30;
        env->player.player_stuck = false;

        if (env->player.lives <= 0) {
            env->player.lives = 0;
            env->game_over = true;
            env->agents[0].terminals[0] = 1;
            add_log(env);
            compute_observations(env);
            puf_reset(env);
        }
        compute_observations(env);
        return;
    }

    if (env->player.bullet.active && env->player.y > env->enemy.y + ENEMY_HEIGHT &&
        check_collision(env->player.bullet.x, env->player.bullet.y, PLAYER_BULLET_WIDTH, PLAYER_BULLET_HEIGHT,
                        env->enemy.x, env->enemy.y, ENEMY_WIDTH, ENEMY_HEIGHT) &&
        env->enemy.active) {
        env->player.bullet.active = false;
        env->enemy.active = false;
        env->kill_streak += 1;
        fired_bullet_rew += 1.5f;
        env->player.score += 1;
        scored_kill = 1;
        env->enemy_explosion_timer = 30;
        float enemy_x_normalized = 1.0f - (env->enemy.x / SCREEN_WIDTH);
        hit_enemy_with_bullet_rew += (crossed_screen == 0) ? (4.5f * enemy_x_normalized)
                                                          : (3.5f * enemy_x_normalized);
    }

    if (env->enemy.bullet.active &&
        check_collision(env->enemy.bullet.x, env->enemy.bullet.y, 10, 12, 
                        env->player.x, env->player.y, PLAYER_WIDTH, PLAYER_HEIGHT)) {
        env->enemy.bullet.active = false;
        env->player.lives--;
        env->player_explosion_timer = 30;
        env->player.player_stuck = false;
        env->enemy.attacking = false;
        env->enemy.x = -ENEMY_WIDTH;
        env->enemy.y = rand() % (SCREEN_HEIGHT - ENEMY_HEIGHT);

        if (env->player.lives <= 0) {
            env->player.lives = 0;
            env->game_over = true;
            env->agents[0].terminals[0] = 1;
            compute_observations(env);
            add_log(env);
            puf_reset(env);
        }
    }

    if (!(env->player.y > env->enemy.y + ENEMY_HEIGHT)) {
        vertical_closeness_rew = 0.0f;
        fired_bullet_rew = 0.0f;
        hit_enemy_with_bullet_rew = 0.0f;
    } else {
        float v_delta_distance = env->player.y - env->enemy.y;
        v_delta_distance = 2.0f - (v_delta_distance / SCREEN_HEIGHT);
        vertical_closeness_rew = 0.01f * v_delta_distance;
    }

    float avg_score_difference = 0.0f;
    if (env->player.score > 0) {
        avg_score_difference = (float)env->player.score / (env->tick + 1);
    }

    env->log.avg_score_difference = avg_score_difference;
    env->log.fired_bullet_rew = fired_bullet_rew;
    env->log.kill_streak = env->kill_streak;
    env->log.hit_enemy_with_bullet_rew = hit_enemy_with_bullet_rew;
    env->log.vertical_closeness_rew = vertical_closeness_rew;
    env->enemy.crossed_screen = crossed_screen;

    int mode = env->reward_mode;
    if (mode == REWARD_SCORE_ONLY) {
        rew = scored_kill ? 1.0f : 0.0f;
    } else if (mode == REWARD_KILL_PLUS_HIT) {
        rew = (scored_kill ? 1.0f : 0.0f) + hit_enemy_with_bullet_rew;
    } else if (mode == REWARD_CLOSENESS_ONLY) {
        rew = vertical_closeness_rew;
    } else {
        rew += fired_bullet_rew + vertical_closeness_rew + hit_enemy_with_bullet_rew + avg_score_difference;
        rew *= (1.0f + env->kill_streak * 0.1f);

        int apply_gate = (mode != REWARD_NO_GATES);
        int apply_wipe = (mode != REWARD_NO_RIGHT_WIPE && mode != REWARD_NO_GATES);
        if (apply_gate) {
            if (!(env->player.y > env->enemy.y + ENEMY_HEIGHT &&
                  fabs(player_center_x - enemy_center_x) > ENEMY_WIDTH * 0.3f)) {
                rew = fminf(rew, 0.0f);
            }
        }
        if (apply_wipe && env->player.x > SCREEN_WIDTH / 2.0f) {
            env->episode_return = 0.0f;
            rew = 0.0f;
        }
    }

    env->agents[0].rewards[0] = rew;
    env->episode_return += rew;

    if (env->player.score > MAX_SCORE) {
        env->game_over = true;
        env->agents[0].terminals[0] = 1;
        compute_observations(env);
        add_log(env);
        puf_reset(env);
    }

    compute_observations(env);
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255}; 
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

Client* make_client(Blastar* env) {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Blastar");
    Client* client = (Client*)malloc(sizeof(Client));
    SetTargetFPS(60);
    client->player_texture = LoadTexture("resources/blastar/player_ship.png");
    client->enemy_texture = LoadTexture("resources/blastar/enemy_ship.png");
    client->player_bullet_texture = LoadTexture("resources/blastar/player_bullet.png");
    client->enemy_bullet_texture = LoadTexture("resources/blastar/enemy_bullet.png");
    client->explosion_texture = LoadTexture("resources/blastar/player_death_explosion.png");
    env->client = client;
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void puf_render(Blastar* env) {
    if (env->client == NULL) {
        make_client(env);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    blastar_human_controls(env);

    Client* client = env->client;

    if (WindowShouldClose()) {
        env->game_over = true;
        close_client(client);
        env->client = NULL;
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    if (env->game_over && env->player.lives <=0) {
        DrawText("GAME OVER", SCREEN_WIDTH / 2 - MeasureText("GAME OVER", 30) / 2, SCREEN_HEIGHT / 2 - 15, 30, PUFF_RED);
        DrawText(TextFormat("FINAL SCORE: %d", env->player.score), SCREEN_WIDTH / 2 - MeasureText(TextFormat("FINAL SCORE: %d", env->player.score), 20)/2, SCREEN_HEIGHT / 2 + 25, 20, PUFF_CYAN);
    } else {
        if (env->player_explosion_timer > 0) {
            DrawTexture(client->explosion_texture, env->player.x, env->player.y, WHITE);
        } else if (env->player.lives > 0) {
            DrawTexture(client->player_texture, env->player.x, env->player.y, WHITE);
        }
        if (env->enemy_explosion_timer > 0) {
            DrawTexture(client->explosion_texture, env->enemy.x, env->enemy.y, WHITE);
        } else if (env->enemy.active) {
            DrawTexture(client->enemy_texture, env->enemy.x, env->enemy.y, WHITE);
        }
        if (env->player.bullet.active) {
            DrawTexture(client->player_bullet_texture, env->player.bullet.x, env->player.bullet.y, WHITE);
        }
        if (env->enemy.bullet.active) {
            DrawTexture(client->enemy_bullet_texture, env->enemy.bullet.x, env->enemy.bullet.y, WHITE);
        }
        if (env->player.player_stuck) {
            DrawText("Status Beam", SCREEN_WIDTH - MeasureText("Status Beam", 20) - 10, SCREEN_HEIGHT / 3, 20, PUFF_RED);
        }
        DrawText(TextFormat("SCORE: %d", env->player.score), 10, 10, 20, PUFF_CYAN);
        DrawText(TextFormat("LIVES: %d", env->player.lives), SCREEN_WIDTH - MeasureText(TextFormat("LIVES: %d", env->player.lives), 20) - 10, 10, 20, PUFF_CYAN);
    }
    EndDrawing();
    puf_web_vsync();
}

// --- Native trainer (pufferl) API ---
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "lives", log->lives);
    dict_set(out, "vertical_closeness_rew", log->vertical_closeness_rew);
    dict_set(out, "fired_bullet_rew", log->fired_bullet_rew);
    dict_set(out, "kill_streak", log->kill_streak);
    dict_set(out, "hit_enemy_with_bullet_rew", log->hit_enemy_with_bullet_rew);
    dict_set(out, "avg_score_difference", log->avg_score_difference);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    int num_obs = dict_get(kwargs, "num_obs");
    env->reward_mode = dict_get(kwargs, "reward_mode");
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    init(env, num_obs);
}

