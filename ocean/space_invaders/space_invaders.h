#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define SI_NOOP 0
#define SI_LEFT 1
#define SI_RIGHT 2
#define SI_FIRE 3

#define SI_ROWS 5
#define SI_COLS 11
#define SI_NUM_INVADERS (SI_ROWS * SI_COLS)
#define SI_MAX_ENEMY_BULLETS 3
#define SI_SPR_CELL 64
#define SI_SPR_TYPE0 0
#define SI_SPR_TYPE1 2
#define SI_SPR_TYPE2 4
#define SI_SPR_PLAYER 6
#define SI_SPR_PBULLET 7
#define SI_SPR_EBULLET 8

#define ACT_SIZES {4}
#define OBS_SIZE (9 + SI_NUM_INVADERS + 3 * SI_MAX_ENEMY_BULLETS)
#define NUM_ATNS 1

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct Client {
    int width;
    int height;
    Texture2D sprites;
} Client;

typedef struct Bullet {
    float x;
    float y;
    int active;
} Bullet;

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    Client* client;

    int width;
    int height;
    int frameskip;
    int player_speed;
    int player_bullet_speed;
    int enemy_bullet_speed;
    int formation_dx;
    int formation_dy;
    int formation_start_interval;
    int enemy_fire_interval;
    int invader_w;
    int invader_h;
    int invader_spacing_x;
    int invader_spacing_y;
    int formation_margin_x;
    int formation_margin_y;
    int player_w;
    int player_h;
    int player_y_offset;
    int bullet_w;
    int bullet_h;
    int max_lives;

    int score;
    int lives;
    int tick;
    int formation_dir;
    int formation_tick;
    int fire_cooldown;
    int num_alive;
    float episode_return_accum;
    float player_x;
    float formation_x;
    float formation_y;
    float invaders_alive[SI_NUM_INVADERS];
    Bullet player_bullet;
    Bullet enemy_bullets[SI_MAX_ENEMY_BULLETS];
    int row_alive[SI_ROWS];
    int col_alive[SI_COLS];
    int min_row;
    int max_row;
    int min_col;
    int max_col;
    unsigned int rng;
};
typedef Env SpaceInvaders;

static inline int si_row_points(int row) {
    if (row == 0) {
        return 30;
    }
    if (row <= 2) {
        return 20;
    }
    return 10;
}

static inline float si_invader_x(SpaceInvaders* env, int col) {
    return env->formation_x + col * (env->invader_w + env->invader_spacing_x);
}

static inline float si_invader_y(SpaceInvaders* env, int row) {
    return env->formation_y + row * (env->invader_h + env->invader_spacing_y);
}

static inline int si_player_y(SpaceInvaders* env) {
    return env->height - env->player_y_offset - env->player_h;
}

void init(SpaceInvaders* env) {
    env->tick = 0;
    memset(&env->log, 0, sizeof(Log));
}

void puf_close(SpaceInvaders* env) {
    if (env->client != NULL) {
        if (env->client->sprites.id > 0) {
            UnloadTexture(env->client->sprites);
        }
        if (IsWindowReady()) {
            CloseWindow();
        }
        free(env->client);
    }
}

void add_log(SpaceInvaders* env) {
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return_accum;
    env->log.score += env->score;
    env->log.perf += (float)(SI_NUM_INVADERS - env->num_alive)
        / (float)SI_NUM_INVADERS;
    env->log.n += 1;
}

void compute_observations(SpaceInvaders* env) {
    obs_t* o = env->agents[0].observations;
    int i = 0;
    o[i++] = env->player_x / (float)env->width;
    o[i++] = (float)env->player_bullet.active;
    o[i++] = env->player_bullet.x / (float)env->width;
    o[i++] = env->player_bullet.y / (float)env->height;
    o[i++] = (env->formation_dir > 0) ? 1.0f : 0.0f;
    o[i++] = env->formation_x / (float)env->width;
    o[i++] = env->formation_y / (float)env->height;
    o[i++] = (float)env->num_alive / (float)SI_NUM_INVADERS;
    o[i++] = (float)env->lives / (float)env->max_lives;
    for (int k = 0; k < SI_NUM_INVADERS; k++) {
        o[i++] = env->invaders_alive[k];
    }
    for (int b = 0; b < SI_MAX_ENEMY_BULLETS; b++) {
        o[i++] = (float)env->enemy_bullets[b].active;
        o[i++] = env->enemy_bullets[b].x / (float)env->width;
        o[i++] = env->enemy_bullets[b].y / (float)env->height;
    }
}

void reset_formation(SpaceInvaders* env) {
    env->formation_x = env->formation_margin_x;
    env->formation_y = env->formation_margin_y;
    env->formation_dir = 1;
    env->formation_tick = 0;
    env->num_alive = SI_NUM_INVADERS;
    for (int i = 0; i < SI_NUM_INVADERS; i++) {
        env->invaders_alive[i] = 1.0f;
    }
    for (int r = 0; r < SI_ROWS; r++) {
        env->row_alive[r] = SI_COLS;
    }
    for (int c = 0; c < SI_COLS; c++) {
        env->col_alive[c] = SI_ROWS;
    }
    env->min_row = 0;
    env->max_row = SI_ROWS - 1;
    env->min_col = 0;
    env->max_col = SI_COLS - 1;
    env->player_bullet.active = 0;
    for (int b = 0; b < SI_MAX_ENEMY_BULLETS; b++) {
        env->enemy_bullets[b].active = 0;
    }
    env->fire_cooldown = env->enemy_fire_interval;
}

static inline void recompute_grid_bounds(SpaceInvaders* env) {
    if (env->num_alive == 0) {
        env->min_row = SI_ROWS;
        env->max_row = -1;
        env->min_col = SI_COLS;
        env->max_col = -1;
        return;
    }
    int mr = 0;
    while (mr < SI_ROWS && env->row_alive[mr] == 0) {
        mr++;
    }
    int Mr = SI_ROWS - 1;
    while (Mr >= 0 && env->row_alive[Mr] == 0) {
        Mr--;
    }
    int mc = 0;
    while (mc < SI_COLS && env->col_alive[mc] == 0) {
        mc++;
    }
    int Mc = SI_COLS - 1;
    while (Mc >= 0 && env->col_alive[Mc] == 0) {
        Mc--;
    }
    env->min_row = mr;
    env->max_row = Mr;
    env->min_col = mc;
    env->max_col = Mc;
}

static inline int formation_bounds_cached(SpaceInvaders* env, float* out_min_x,
        float* out_max_x, float* out_max_y) {
    if (env->num_alive == 0) {
        return 0;
    }
    *out_min_x = si_invader_x(env, env->min_col);
    *out_max_x = si_invader_x(env, env->max_col) + env->invader_w;
    *out_max_y = si_invader_y(env, env->max_row) + env->invader_h;
    return 1;
}

void puf_reset(SpaceInvaders* env) {
    env->score = 0;
    env->lives = env->max_lives;
    env->tick = 0;
    env->episode_return_accum = 0.0f;
    // Random start x so "stay still and fire" is not the local optimum.
    int max_px = env->width - env->player_w;
    env->player_x = (float)(rand_r(&env->rng) % (max_px + 1));
    reset_formation(env);
    compute_observations(env);
}

void step_formation(SpaceInvaders* env) {
    int interval = env->formation_start_interval * env->num_alive
        / SI_NUM_INVADERS;
    if (interval < 2) {
        interval = 2;
    }
    env->formation_tick++;
    if (env->formation_tick < interval) {
        return;
    }
    env->formation_tick = 0;

    float min_x, max_x, max_y;
    if (!formation_bounds_cached(env, &min_x, &max_x, &max_y)) {
        return;
    }

    float dx = env->formation_dir * env->formation_dx;
    float new_min = min_x + dx;
    float new_max = max_x + dx;
    if (new_min < 0 || new_max > env->width) {
        env->formation_dir *= -1;
        env->formation_y += env->formation_dy;
        return;
    }
    env->formation_x += dx;
}

void maybe_enemy_fire(SpaceInvaders* env) {
    if (env->fire_cooldown > 0) {
        env->fire_cooldown--;
        return;
    }
    int slot = -1;
    for (int b = 0; b < SI_MAX_ENEMY_BULLETS; b++) {
        if (!env->enemy_bullets[b].active) {
            slot = b;
            break;
        }
    }
    if (slot < 0) {
        return;
    }
    for (int attempt = 0; attempt < 6; attempt++) {
        int col = rand_r(&env->rng) % SI_COLS;
        int r = SI_ROWS - 1;
        while (r >= 0 && env->invaders_alive[r * SI_COLS + col] == 0.0f) {
            r--;
        }
        if (r < 0) {
            continue;
        }
        env->enemy_bullets[slot].x = si_invader_x(env, col)
            + env->invader_w / 2.0f - env->bullet_w / 2.0f;
        env->enemy_bullets[slot].y = si_invader_y(env, r) + env->invader_h;
        env->enemy_bullets[slot].active = 1;
        env->fire_cooldown = env->enemy_fire_interval;
        return;
    }
}

static inline int aabb(float ax, float ay, float aw, float ah,
        float bx, float by, float bw, float bh) {
    return ax < bx + bw && ax + aw > bx && ay < by + bh && ay + ah > by;
}

static int try_player_bullet_hit(SpaceInvaders* env, float* rewards) {
    if (!env->player_bullet.active || env->num_alive == 0) {
        return 0;
    }
    float form_top_y = si_invader_y(env, env->min_row);
    float form_bot_y = si_invader_y(env, env->max_row) + env->invader_h;
    float pb_top = env->player_bullet.y;
    float pb_bot = pb_top + env->bullet_h;
    if (pb_bot <= form_top_y || pb_top >= form_bot_y) {
        return 0;
    }
    float pb_left = env->player_bullet.x;
    float pb_right = pb_left + env->bullet_w;
    int col_pitch = env->invader_w + env->invader_spacing_x;
    float rel_lo = (pb_left - env->formation_x) / (float)col_pitch;
    float rel_hi = (pb_right - env->formation_x) / (float)col_pitch;
    int col_lo = (int)rel_lo - (rel_lo < (int)rel_lo);
    int col_hi = (int)rel_hi - (rel_hi < (int)rel_hi);
    if (col_lo < env->min_col) {
        col_lo = env->min_col;
    }
    if (col_hi > env->max_col) {
        col_hi = env->max_col;
    }
    if (col_lo > col_hi) {
        return 0;
    }
    for (int c = col_lo; c <= col_hi; c++) {
        if (env->col_alive[c] == 0) {
            continue;
        }
        float ix = si_invader_x(env, c);
        if (pb_right <= ix || pb_left >= ix + env->invader_w) {
            continue;
        }
        for (int r = env->min_row; r <= env->max_row; r++) {
            int idx = r * SI_COLS + c;
            if (env->invaders_alive[idx] == 0.0f) {
                continue;
            }
            float iy = si_invader_y(env, r);
            if (pb_bot <= iy || pb_top >= iy + env->invader_h) {
                continue;
            }
            env->invaders_alive[idx] = 0.0f;
            env->num_alive--;
            env->row_alive[r]--;
            env->col_alive[c]--;
            if ((r == env->min_row || r == env->max_row)
                    && env->row_alive[r] == 0) {
                recompute_grid_bounds(env);
            } else if ((c == env->min_col || c == env->max_col)
                    && env->col_alive[c] == 0) {
                recompute_grid_bounds(env);
            }
            env->player_bullet.active = 0;
            int pts = si_row_points(r);
            env->score += pts;
            float r_add = (float)pts * 0.1f;
            rewards[0] += r_add;
            env->episode_return_accum += r_add;
            return 1;
        }
    }
    return 0;
}

void step_frame(SpaceInvaders* env, int action) {
    float* rewards = env->agents[0].rewards;
    float* terminals = env->agents[0].terminals;

    if (action == SI_LEFT) {
        env->player_x -= env->player_speed;
    } else if (action == SI_RIGHT) {
        env->player_x += env->player_speed;
    }
    if (env->player_x < 0) {
        env->player_x = 0;
    }
    float max_px = env->width - env->player_w;
    if (env->player_x > max_px) {
        env->player_x = max_px;
    }

    if (action == SI_FIRE && !env->player_bullet.active) {
        env->player_bullet.x = env->player_x
            + env->player_w / 2.0f - env->bullet_w / 2.0f;
        env->player_bullet.y = si_player_y(env) - env->bullet_h;
        env->player_bullet.active = 1;
    }

    if (env->player_bullet.active) {
        env->player_bullet.y -= env->player_bullet_speed;
        if (env->player_bullet.y + env->bullet_h < 0) {
            env->player_bullet.active = 0;
        }
    }

    for (int b = 0; b < SI_MAX_ENEMY_BULLETS; b++) {
        if (!env->enemy_bullets[b].active) {
            continue;
        }
        env->enemy_bullets[b].y += env->enemy_bullet_speed;
        if (env->enemy_bullets[b].y > env->height) {
            env->enemy_bullets[b].active = 0;
        }
    }

    step_formation(env);
    maybe_enemy_fire(env);
    try_player_bullet_hit(env, rewards);

    int player_y = si_player_y(env);
    int player_hit = 0;
    for (int b = 0; b < SI_MAX_ENEMY_BULLETS; b++) {
        if (!env->enemy_bullets[b].active) {
            continue;
        }
        if (aabb(env->enemy_bullets[b].x, env->enemy_bullets[b].y,
                env->bullet_w, env->bullet_h,
                env->player_x, (float)player_y,
                env->player_w, env->player_h)) {
            env->enemy_bullets[b].active = 0;
            player_hit = 1;
            break;
        }
    }

    int invaded = 0;
    if (env->num_alive > 0) {
        float fmax_y = si_invader_y(env, env->max_row) + env->invader_h;
        invaded = (fmax_y >= player_y);
    }

    if (player_hit) {
        env->lives--;
        rewards[0] -= 1.0f;
        env->episode_return_accum -= 1.0f;
    }

    int cleared = env->num_alive == 0;
    if (cleared) {
        rewards[0] += 10.0f;
        env->episode_return_accum += 10.0f;
        reset_formation(env);
    }

    if (env->lives <= 0 || invaded) {
        terminals[0] = 1;
        add_log(env);
        puf_reset(env);
    }
}

// Hold Left Shift + A/D/space.
static void space_invaders_human_controls(SpaceInvaders *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    env->agents[0].actions[0] = SI_NOOP;
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        env->agents[0].actions[0] = SI_LEFT;
    }
    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        env->agents[0].actions[0] = SI_RIGHT;
    }
    if (IsKeyDown(KEY_SPACE) || IsKeyDown(KEY_UP)) {
        env->agents[0].actions[0] = SI_FIRE;
    }
}

void puf_step(SpaceInvaders* env) {
    env->agents[0].terminals[0] = 0;
    env->agents[0].rewards[0] = 0.0f;
    int action = (int)env->agents[0].actions[0];
    for (int i = 0; i < env->frameskip; i++) {
        env->tick++;
        step_frame(env, action);
        if (env->agents[0].terminals[0]) {
            break;
        }
    }
    compute_observations(env);
}

Client* make_client(SpaceInvaders* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = env->width;
    client->height = env->height;
    InitWindow(env->width, env->height, "PufferLib Space Invaders");
    SetTargetFPS(60);
    client->sprites = LoadTexture("resources/space_invaders/sprites.png");
    return client;
}

static void si_draw_sprite(Texture2D sheet, int cell, float x, float y, int w, int h) {
    if (sheet.id == 0) {
        DrawRectangle((int)x, (int)y, w, h, (Color){0, 187, 187, 255});
        return;
    }
    // Bullet ink is ~8x24 in a 64x64 cell. Stretching the empty cell to
    // dest 4x10 nearest-samples empty texels, so the shot vanishes.
    float sx = (float)(cell * SI_SPR_CELL);
    float sy = 0.0f;
    float sw = (float)SI_SPR_CELL;
    float sh = (float)SI_SPR_CELL;
    if (cell == SI_SPR_PBULLET) {
        sx += 28.0f;
        sy += 8.0f;
        sw = 8.0f;
        sh = 24.0f;
    } else if (cell == SI_SPR_EBULLET) {
        sx += 24.0f;
        sy += 8.0f;
        sw = 8.0f;
        sh = 24.0f;
    }
    Rectangle src = {sx, sy, sw, sh};
    Rectangle dest = {x, y, (float)w, (float)h};
    DrawTexturePro(sheet, src, dest, (Vector2){0, 0}, 0.0f, WHITE);
}

void puf_render(SpaceInvaders* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    space_invaders_human_controls(env);
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int py = si_player_y(env);
    Texture2D sheet = env->client->sprites;
    int walk = (env->tick / 16) % 2;
    si_draw_sprite(sheet, SI_SPR_PLAYER, env->player_x, (float)py,
        env->player_w, env->player_h);

    for (int r = 0; r < SI_ROWS; r++) {
        int type = SI_SPR_TYPE2;
        if (r == 0) {
            type = SI_SPR_TYPE0;
        } else if (r <= 2) {
            type = SI_SPR_TYPE1;
        }
        for (int c = 0; c < SI_COLS; c++) {
            int idx = r * SI_COLS + c;
            if (env->invaders_alive[idx] == 0.0f) {
                continue;
            }
            si_draw_sprite(sheet, type + walk,
                si_invader_x(env, c), si_invader_y(env, r),
                env->invader_w, env->invader_h);
        }
    }

    if (env->player_bullet.active) {
        si_draw_sprite(sheet, SI_SPR_PBULLET,
            env->player_bullet.x, env->player_bullet.y,
            env->bullet_w, env->bullet_h);
    }
    for (int b = 0; b < SI_MAX_ENEMY_BULLETS; b++) {
        if (!env->enemy_bullets[b].active) {
            continue;
        }
        si_draw_sprite(sheet, SI_SPR_EBULLET,
            env->enemy_bullets[b].x, env->enemy_bullets[b].y,
            env->bullet_w, env->bullet_h);
    }

    DrawText(TextFormat("Score: %i", env->score), 10, 10, 20, WHITE);
    DrawText(TextFormat("Lives: %i", env->lives), env->width - 100, 10, 20,
        WHITE);
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
    env->frameskip = dict_get(kwargs, "frameskip");
    env->player_speed = dict_get(kwargs, "player_speed");
    env->player_bullet_speed = dict_get(kwargs, "player_bullet_speed");
    env->enemy_bullet_speed = dict_get(kwargs, "enemy_bullet_speed");
    env->formation_dx = dict_get(kwargs, "formation_dx");
    env->formation_dy = dict_get(kwargs, "formation_dy");
    env->formation_start_interval = dict_get(kwargs, "formation_start_interval");
    env->enemy_fire_interval = dict_get(kwargs, "enemy_fire_interval");
    env->invader_w = dict_get(kwargs, "invader_w");
    env->invader_h = dict_get(kwargs, "invader_h");
    env->invader_spacing_x = dict_get(kwargs, "invader_spacing_x");
    env->invader_spacing_y = dict_get(kwargs, "invader_spacing_y");
    env->formation_margin_x = dict_get(kwargs, "formation_margin_x");
    env->formation_margin_y = dict_get(kwargs, "formation_margin_y");
    env->player_w = dict_get(kwargs, "player_w");
    env->player_h = dict_get(kwargs, "player_h");
    env->player_y_offset = dict_get(kwargs, "player_y_offset");
    env->bullet_w = dict_get(kwargs, "bullet_w");
    env->bullet_h = dict_get(kwargs, "bullet_h");
    env->max_lives = dict_get(kwargs, "max_lives");
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    init(env);
}
