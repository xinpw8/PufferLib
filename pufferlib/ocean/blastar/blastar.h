#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "raylib.h"

#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480
#define LOG_BUFFER_SIZE 4096
#define MAX_EPISODE_STEPS 2800
#define PLAYER_MAX_LIVES 20  // 5
#define ENEMY_SPAWN_Y 50
#define ENEMY_SPAWN_X -30

static const float speed_scale = 4.0f;
static const int enemy_width = 16;
static const int enemy_height = 17;
static const int player_width = 17;
static const int player_height = 17;
static const int player_bullet_width = 17;

// Log structure
typedef struct Log {
    float episode_return;
    float episode_length;
    float score;
    float lives;
    float vertical_closeness_rew;
    float fired_bullet_rew;
    float bullet_distance_to_enemy_rew;
    int kill_streak;
    float flat_below_enemy_rew;
    float danger_zone_penalty_rew;
    float crashing_penalty_rew;
    float hit_enemy_with_bullet_rew;
    float hit_by_enemy_bullet_penalty_rew;
    int enemy_crossed_screen;
    float bad_guy_score;
    float avg_score_difference;
} Log;

// LogBuffer structure
typedef struct LogBuffer {
    Log* logs;
    int length;
    int idx;
} LogBuffer;

typedef struct RewardBuffer {
    float* rewards;
    int size;
    int idx;
} RewardBuffer;

typedef struct Bullet {
    float x, y;
    float last_x, last_y;
    bool active;
    double travel_time;
    float bulletSpeed;
} Bullet;

typedef struct Enemy {
    float x, y;
    float last_x, last_y;
    float enemySpeed;
    bool active;
    bool attacking;
    int direction;
    int crossed_screen;
    Bullet bullet;
} Enemy;

typedef struct Player {
    float x, y;
    float last_x, last_y;
    float playerSpeed;
    int score;
    int lives;
    Bullet bullet;
    bool bulletFired;
    bool playerStuck;
    float explosion_timer;
} Player;

typedef struct Client {
    float screen_width;
    float screen_height;
    float player_width;
    float player_height;
    float enemy_width;
    float enemy_height;

    Texture2D player_texture;
    Texture2D enemy_texture;
    Texture2D player_bullet_texture;
    Texture2D enemy_bullet_texture;
    Texture2D explosion_texture;

    Color player_color;
    Color enemy_color;
    Color bullet_color;
    Color explosion_color;
} Client;

typedef struct BlastarEnv {
    int screen_width;
    int screen_height;
    float player_width;
    float player_height;
    float last_bullet_distance;
    bool game_over;
    int tick;
    int playerExplosionTimer;
    int enemyExplosionTimer;
    int max_score;
    int bullet_travel_time;
    bool bullet_crossed_enemy_y;
    int kill_streak;
    float bad_guy_score;
    int enemy_respawns;
    Player player;
    Enemy enemy;
    Bullet bullet;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    LogBuffer* log_buffer;
    Log log;
} BlastarEnv;

static inline void scale_speeds(BlastarEnv* env) {
    env->player.playerSpeed *= speed_scale;
    env->enemy.enemySpeed *= speed_scale;
    env->player.bullet.bulletSpeed *= speed_scale;
    env->enemy.bullet.bulletSpeed *= speed_scale;
}

LogBuffer* allocate_logbuffer(int size) {
    LogBuffer* buffer = (LogBuffer*)malloc(sizeof(LogBuffer));
    buffer->logs = (Log*)malloc(size * sizeof(Log));
    buffer->length = size;
    buffer->idx = 0;
    return buffer;
}

void free_logbuffer(LogBuffer* buffer) {
    free(buffer->logs);
    free(buffer);
}

void add_log(LogBuffer* logs, Log* log) {
    if (logs->idx == logs->length) {
        return;
    }
    logs->logs[logs->idx] = *log;
    logs->idx += 1;
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log aggregated = {0};
    for (int i = 0; i < logs->length; i++) {
        aggregated.episode_return += logs->logs[i].episode_return /= logs->idx;
        aggregated.episode_length += logs->logs[i].episode_length /= logs->idx;
        aggregated.score += logs->logs[i].score /= logs->idx;
        aggregated.lives += logs->logs[i].lives /= logs->idx;
        aggregated.vertical_closeness_rew +=
            logs->logs[i].vertical_closeness_rew /= logs->idx;
        aggregated.fired_bullet_rew += logs->logs[i].fired_bullet_rew /=
            logs->idx;
        aggregated.bullet_distance_to_enemy_rew +=
            logs->logs[i].bullet_distance_to_enemy_rew /= logs->idx;
        aggregated.kill_streak += logs->logs[i].kill_streak /= logs->idx;
        aggregated.flat_below_enemy_rew += logs->logs[i].flat_below_enemy_rew /=
            logs->idx;
        aggregated.danger_zone_penalty_rew +=
            logs->logs[i].danger_zone_penalty_rew /= logs->idx;
        aggregated.crashing_penalty_rew += logs->logs[i].crashing_penalty_rew /=
            logs->idx;
        aggregated.hit_enemy_with_bullet_rew +=
            logs->logs[i].hit_enemy_with_bullet_rew /= logs->idx;
        aggregated.hit_by_enemy_bullet_penalty_rew +=
            logs->logs[i].hit_by_enemy_bullet_penalty_rew /= logs->idx;
        aggregated.enemy_crossed_screen += logs->logs[i].enemy_crossed_screen /=
            logs->idx;
        aggregated.bad_guy_score += logs->logs[i].bad_guy_score /= logs->idx;
        aggregated.avg_score_difference += logs->logs[i].avg_score_difference /=
            logs->idx;
    }
    logs->idx = 0;
    return aggregated;
}

void init(BlastarEnv* env) {
    env->game_over = false;
    env->tick = 0;
    env->playerExplosionTimer = 0;
    env->enemyExplosionTimer = 0;
    env->max_score = 5 * PLAYER_MAX_LIVES;
    env->player.playerSpeed = 2.0f;
    env->enemy.enemySpeed = 1.0f;
    env->player.bullet.bulletSpeed = 3.0f;
    env->enemy.bullet.bulletSpeed = 3.0f;
    scale_speeds(env);
    // Randomize player x and y position
    env->player.x = (float)(rand() % (SCREEN_WIDTH - 17));
    env->player.y = (float)(rand() % (SCREEN_HEIGHT - 17));
    env->player.last_x = env->player.x;
    env->player.last_y = env->player.y;
    env->player.score = 0;
    env->bad_guy_score = 0.0f;
    env->player.lives = PLAYER_MAX_LIVES;
    env->player.bulletFired = false;
    env->player.playerStuck = false;
    env->player.bullet.active = false;
    env->player.bullet.x = env->player.x;
    env->player.bullet.y = env->player.y;
    env->player.bullet.last_x = env->player.bullet.x;
    env->player.bullet.last_y = env->player.bullet.y;
    env->bullet_travel_time = 0;
    env->last_bullet_distance = 0;
    env->kill_streak = 0;
    env->enemy.x = ENEMY_SPAWN_X;
    env->enemy.y = ENEMY_SPAWN_Y;
    env->enemy.last_x = env->enemy.x;
    env->enemy.last_y = env->enemy.y;
    env->enemy.active = true;
    env->enemy.attacking = false;
    env->enemy.direction = 1;
    env->enemy_respawns = 0;

    env->enemy.bullet.active = false;
    env->enemy.bullet.x = env->enemy.x;
    env->enemy.bullet.y = env->enemy.y;
    env->enemy.bullet.last_x = env->enemy.bullet.x;
    env->enemy.bullet.last_y = env->enemy.bullet.y;
}

void allocate(BlastarEnv* env) {
    init(env);
    env->observations = (float*)calloc(31, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_allocated(BlastarEnv* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free_logbuffer(env->log_buffer);
}

void reset(BlastarEnv* env) {
    init(env);
}

void compute_observations(BlastarEnv* env) {
    env->log.lives = env->player.lives;
    env->log.score = env->player.score;
    env->log.bad_guy_score = env->bad_guy_score;
    env->log.enemy_crossed_screen = env->enemy.crossed_screen;

    // Normalize player and enemy positions
    env->observations[0] = env->player.x / SCREEN_WIDTH;  // Normalized player x
    env->observations[1] =
        env->player.y / SCREEN_HEIGHT;                    // Normalized player y
    env->observations[2] = env->enemy.x / SCREEN_WIDTH;   // Normalized enemy x
    env->observations[3] = env->enemy.y / SCREEN_HEIGHT;  // Normalized enemy y

    // Player bullet location and status
    if (env->player.bullet.active) {
        env->observations[4] = env->player.bullet.x / SCREEN_WIDTH;
        env->observations[5] = env->player.bullet.y / SCREEN_HEIGHT;
        env->observations[6] = 1.0f;  // Player bullet speed normalized
    } else {
        env->observations[4] = 0.0f;
        env->observations[5] = 0.0f;
        env->observations[6] = 0.0f;
    }

    // Enemy bullet location and status
    if (env->enemy.bullet.active) {
        env->observations[7] = env->enemy.bullet.x / SCREEN_WIDTH;
        env->observations[8] = env->enemy.bullet.y / SCREEN_HEIGHT;
        env->observations[9] = 1.0f;  // Enemy bullet speed normalized
    } else {
        env->observations[7] = 0.0f;
        env->observations[8] = 0.0f;
        env->observations[9] = 0.0f;
    }

    // Additional observations for player score and lives
    env->observations[10] = env->player.score / (float)env->max_score;
    env->observations[11] = env->player.lives / (float)PLAYER_MAX_LIVES;

    // Enemy speed
    env->observations[12] =
        1.0f / 2.0f;  // Enemy speed normalized (1.0 is hardcoded speed)

    // Player speed
    env->observations[13] =
        2.0f / 2.0f;  // Player speed normalized (2.0 is hardcoded speed)

    // Enemy last known position
    env->observations[14] =
        env->enemy.last_x / SCREEN_WIDTH;  // Normalized enemy x
    env->observations[15] =
        env->enemy.last_y / SCREEN_HEIGHT;  // Normalized enemy y

    // Player last known position
    env->observations[16] =
        env->player.last_x / SCREEN_WIDTH;  // Normalized player x
    env->observations[17] =
        env->player.last_y / SCREEN_HEIGHT;  // Normalized player y

    // Enemy bullet last location
    env->observations[18] = env->enemy.bullet.active
                                ? env->enemy.bullet.last_x / SCREEN_WIDTH
                                : 0.0f;  // Normalized x
    env->observations[19] = env->enemy.bullet.active
                                ? env->enemy.bullet.last_y / SCREEN_HEIGHT
                                : 0.0f;  // Normalized y

    // Player bullet last location
    env->observations[20] = env->player.bullet.active
                                ? env->player.bullet.last_x / SCREEN_WIDTH
                                : 0.0f;  // Normalized x
    env->observations[21] = env->player.bullet.active
                                ? env->player.bullet.last_y / SCREEN_HEIGHT
                                : 0.0f;  // Normalized y

    // Bullet closeness to enemy (Euclidean distance)
    if (env->player.bullet.active) {
        float dx = env->player.bullet.x - env->enemy.x;
        float dy = env->player.bullet.y - env->enemy.y;
        float distance = sqrtf(dx * dx + dy * dy);
        // Normalize the distance to [0, 1]
        env->observations[22] =
            1.0f - (distance / sqrtf(SCREEN_WIDTH * SCREEN_WIDTH +
                                     SCREEN_HEIGHT * SCREEN_HEIGHT));
    } else {
        env->observations[22] = 0.0f;  // No bullet
    }

    // Danger zone calculations (player-enemy distance)
    float player_center_x = env->player.x + player_width / 2.0f;
    float player_center_y = env->player.y + env->player_height / 2.0f;
    float enemy_center_x = env->enemy.x + enemy_width / 2.0f;
    float enemy_center_y = env->enemy.y + enemy_height / 2.0f;
    float dx = player_center_x - enemy_center_x;
    float dy = player_center_y - enemy_center_y;
    float distance = sqrtf(dx * dx + dy * dy);
    float max_distance =
        sqrtf(SCREEN_WIDTH * SCREEN_WIDTH + SCREEN_HEIGHT * SCREEN_HEIGHT);
    env->observations[23] = 1.0f - (distance / max_distance);
    env->observations[24] =
        (distance < 50.0f) ? 1.0f : 0.0f;  // Danger threshold

    // "Below enemy ship" observation: 1.0 if player is below enemy, 0.0
    env->observations[25] =
        (env->player.y > env->enemy.y + enemy_height) ? 1.0f : 0.0f;

    // Enemy crossed screen observation (count)
    if (env->enemy.crossed_screen > 0 && env->player.score > 0) {
        env->observations[26] =
            (float)env->enemy.crossed_screen / (float)env->player.score;
    } else {
        env->observations[26] = 0.0f;
    }

    // Player vs bad guy score difference
    float total_score = env->player.score + env->bad_guy_score;
    if (total_score > 0.0f) {
        env->observations[27] =
            (env->bad_guy_score - env->player.score) / total_score;
        env->observations[28] = env->player.score / total_score;
        env->observations[29] = env->bad_guy_score / total_score;
        env->observations[30] = env->enemy.crossed_screen / total_score;
    } else {
        env->observations[27] = 0.0f;
        env->observations[28] = 0.0f;
        env->observations[29] = 0.0f;
        env->observations[30] = 0.0f;
    }
}

void c_step(BlastarEnv* env) {
    if (env->game_over) {
        if (env->terminals) env->terminals[0] = 1;
        add_log(env->log_buffer, &env->log);
        reset(env);
        return;
    }

    env->tick++;
    env->log.episode_length += 1;

    float rew = 0.0f;
    env->rewards[0] = rew;
    float score = 0.0f;
    float bad_guy_score = 0.0f;
    float fired_bullet_rew = 0.0f;
    float bullet_distance_to_enemy_rew = 0.0f;
    float flat_below_enemy_rew = 0.0f;
    float vertical_closeness_rew = 0.0f;
    float danger_zone_penalty_rew = 0.0f;
    float crashing_penalty_rew = 0.0f;
    float hit_enemy_with_bullet_rew = 0.0f;
    float hit_by_enemy_bullet_penalty_rew = 0.0f;
    int crossed_screen = 0;
    int action = 0;
    action = env->actions[0];

    // Handle player explosion
    if (env->playerExplosionTimer > 0) {
        env->playerExplosionTimer--;
        env->kill_streak = 0;
        if (env->playerExplosionTimer == 0) {
            env->player.playerStuck = false;
            env->player.bullet.active = false;
        }
        compute_observations(env);
        add_log(env->log_buffer, &env->log);
        return;
    }

    // Handle enemy explosion
    if (env->enemyExplosionTimer > 0) {
        env->enemyExplosionTimer--;
        if (env->enemyExplosionTimer == 0) {
            env->enemy.crossed_screen = 0;
            // Rarely respawn in the same place
            float respawn_bias =
                0.1f;  // 10% chance to respawn in the same place
            if ((float)rand() / (float)RAND_MAX > respawn_bias) {
                // Respawn in a new position
                env->enemy.x = -enemy_width;
                env->enemy.y = rand() % (SCREEN_HEIGHT - enemy_height);
                env->enemy_respawns += 1;
            }
            // Otherwise, respawn in the same place as a rare event
            env->enemy.active = true;
            env->enemy.attacking = false;
        }
        compute_observations(env);
        add_log(env->log_buffer, &env->log);
        return;  // Skip further logic while exploding
    }

    // Keep enemy far enough from bottom of the screen
    if (env->enemy.y > (SCREEN_HEIGHT - (enemy_height * 3.5f))) {
        env->enemy.y = (SCREEN_HEIGHT - (enemy_height * 3.5f));
    }

    // Last enemy and player positions
    env->enemy.last_x = env->enemy.x;
    env->enemy.last_y = env->enemy.y;
    env->player.last_x = env->player.x;
    env->player.last_y = env->player.y;

    // Player movement if not stuck
    if (!env->player.playerStuck) {
        if (action == 1 && env->player.x > 0)
            env->player.x -= env->player.playerSpeed;
        if (action == 2 && env->player.x < SCREEN_WIDTH - 17)
            env->player.x += env->player.playerSpeed;
        if (action == 3 && env->player.y > 0)
            env->player.y -= env->player.playerSpeed;
        if (action == 4 && env->player.y < SCREEN_HEIGHT - 17)
            env->player.y += env->player.playerSpeed;
    }

    // Fire player bullet
    if (action == 5 && (!env->enemy.bullet.active)) {
        // If a bullet is already active, replace it with the new one
        if (env->player.bullet.active) {
            env->player.bullet.active =
                false;  // Deactivate the existing bullet
        } else {
            // Reward for firing a single bullet, if it hits enemy
            fired_bullet_rew += 0.002f;
        }

        // Activate and initialize the new bullet
        env->player.bullet.active = true;
        env->player.bullet.x =
            env->player.x + player_width / 2 - player_bullet_width / 2;
        env->player.bullet.y = env->player.y;
    }

    // Update player bullet
    if (env->player.bullet.active) {
        // Update bullet position
        env->player.bullet.y -= env->player.bullet.bulletSpeed;

        // Deactivate bullet if off-screen
        if (env->player.bullet.y < 0) {
            env->player.bullet.active = false;
            env->bullet_travel_time = 0;
        }
    }

    float playerCenterX = env->player.x + player_width / 2.0f;
    float enemyCenterX = env->enemy.x + enemy_width / 2.0f;

    // Last player bullet location
    env->player.bullet.last_x = env->player.bullet.x;
    env->player.bullet.last_y = env->player.bullet.y;

    // Enemy movement
    if (!env->enemy.attacking) {
        env->enemy.x += env->enemy.enemySpeed;
        if (env->enemy.x > SCREEN_WIDTH) {
            env->enemy.x = -enemy_width;  // Respawn off-screen
            crossed_screen += 1;
        }
    }

    // Enemy attack logic
    if (fabs(playerCenterX - enemyCenterX) < speed_scale &&
        !env->enemy.attacking && env->enemy.active &&
        env->enemy.y < env->player.y - (enemy_height / 2)) {
        // 50% chance of attacking
        if (rand() % 2 == 0) {
            env->enemy.attacking = true;
            if (!env->enemy.bullet.active) {
                env->enemy.bullet.active = true;
                env->enemy.bullet.x = enemyCenterX - 5.0f;
                env->enemy.bullet.y = env->enemy.y + enemy_height;
                // Disable active player bullet
                env->player.bullet.active = false;
                // Player stuck
                env->player.playerStuck = true;
            }
        } else {
            env->enemy.attacking = false;
            env->enemy.x += env->enemy.enemySpeed;  // Avoid attack lock
        }
    }

    // Update enemy bullets
    if (env->enemy.bullet.active) {
        env->enemy.bullet.y += env->enemy.bullet.bulletSpeed;
        if (env->enemy.bullet.y > SCREEN_HEIGHT) {
            env->enemy.bullet.active = false;
            env->player.playerStuck = false;
            env->enemy.attacking = false;
        }
    }

    // Last enemy bullet location
    env->enemy.bullet.last_x = env->enemy.bullet.x;
    env->enemy.bullet.last_y = env->enemy.bullet.y;

    // Collision detection
    Rectangle playerHitbox = {env->player.x, env->player.y, 17, 17};
    Rectangle enemyHitbox = {env->enemy.x, env->enemy.y, enemy_width,
                             enemy_height};

    // Player-Enemy Collision
    if (CheckCollisionRecs(playerHitbox, enemyHitbox)) {
        env->player.lives--;
        env->enemy.active = false;
        env->enemyExplosionTimer = 30;

        // Respawn enemy
        env->enemy.x = -enemy_width;
        env->enemy.y = rand() % (SCREEN_HEIGHT - enemy_height);

        env->playerExplosionTimer = 30;
        env->player.playerStuck = false;

        if (env->player.lives <= 0) {
            env->player.lives = 0;
            env->game_over = true;
            if (env->terminals) env->terminals[0] = 1;
            // env->rewards[0] = rew;
            compute_observations(env);
            add_log(env->log_buffer, &env->log);
            reset(env);
        }
        compute_observations(env);
        return;
    }

    // Player bullet hits enemy
    if (env->player.bullet.active &&
        env->player.y > env->enemy.y + enemy_height) {
        Rectangle bulletHitbox = {env->player.bullet.x, env->player.bullet.y,
                                  17, 6};
        if (CheckCollisionRecs(bulletHitbox, enemyHitbox) &&
            env->enemy.active) {
            env->player.bullet.active = false;
            env->enemy.active = false;
            env->kill_streak += 1;
            fired_bullet_rew += 1.5f;
            env->player.score += 1.0f;
            env->log.score += 1.0f;
            env->enemyExplosionTimer = 30;
            if (crossed_screen == 0) {
                hit_enemy_with_bullet_rew += 2.5f;  // Big reward for quick kill
            } else {
                hit_enemy_with_bullet_rew +=
                    1.5f -
                    (0.1f *
                     env->enemy
                         .crossed_screen);  // Less rew if enemy crossed screen
            }
        } else {
        }
    }

    // Enemy bullet hits player
    if (env->enemy.bullet.active) {
        Rectangle bulletHitbox = {env->enemy.bullet.x, env->enemy.bullet.y, 10,
                                  12};
        if (CheckCollisionRecs(bulletHitbox, playerHitbox)) {
            env->enemy.bullet.active = false;
            env->player.lives--;
            bad_guy_score += 1.0f;
            env->playerExplosionTimer = 30;
            env->player.playerStuck = false;
            env->enemy.attacking = false;
            env->enemy.x = -enemy_width;
            env->enemy.y = rand() % (SCREEN_HEIGHT - enemy_height);

            if (env->player.lives <= 0) {
                env->player.lives = 0;
                env->game_over = true;
                if (env->terminals) env->terminals[0] = 1;
                // env->rewards[0] = rew;
                compute_observations(env);
                add_log(env->log_buffer, &env->log);
                reset(env);
            }
        }
    }

    // Reward computation based on player position relative to enemy
    if (env->player.y > env->enemy.y + enemy_height) {
        // Calculate horizontal distance between player and enemy
        float horizontal_distance = fabs(playerCenterX - enemyCenterX);
        float not_underneath_threshold =
            enemy_width * 0.3f;  // Threshold for "underneath"

        if (horizontal_distance > not_underneath_threshold) {
            // Player is below the enemy and not directly underneath
            flat_below_enemy_rew = 0.01f;  // Base reward for being below
            float vertical_closeness =
                1.0f - ((env->player.y - env->enemy.y) / SCREEN_HEIGHT);
            vertical_closeness_rew =
                0.01f *
                vertical_closeness;  // Additional reward for vertical closeness
        } else {
            // Player is directly underneath the enemy
            flat_below_enemy_rew =
                -0.01f;  // Penalty for being directly underneath
            vertical_closeness_rew = 0.0f;
        }
    } else {
        // Player is not below the enemy
        flat_below_enemy_rew = -0.01f;  // Penalty for being above the enemy
        vertical_closeness_rew = 0.0f;

        // Override all rewards to <= 0
        score = 0.0f;
        fired_bullet_rew = 0.0f;
        bullet_distance_to_enemy_rew = 0.0f;
        hit_enemy_with_bullet_rew = 0.0f;
        rew = -0.01f;  // Minimal penalty for incorrect positioning
    }

    env->log.bad_guy_score += bad_guy_score;
    env->bad_guy_score += bad_guy_score;

    float avg_score_difference = 0.0f;
    if (env->player.score + env->bad_guy_score > 0) {
        int score_difference = env->player.score - env->bad_guy_score;
        avg_score_difference =
            (float)score_difference / (env->player.score + env->bad_guy_score);
    }

    env->log.avg_score_difference = avg_score_difference;            // in-use
    env->log.fired_bullet_rew = fired_bullet_rew;                    // in-use
    env->log.kill_streak = env->kill_streak;                         // in-use
    env->log.hit_enemy_with_bullet_rew = hit_enemy_with_bullet_rew;  // in-use
    env->log.flat_below_enemy_rew = flat_below_enemy_rew;            // in-use
    env->enemy.crossed_screen = crossed_screen;

    // not used
    env->log.vertical_closeness_rew = vertical_closeness_rew;
    env->log.bullet_distance_to_enemy_rew = bullet_distance_to_enemy_rew;
    env->log.danger_zone_penalty_rew = danger_zone_penalty_rew;
    env->log.crashing_penalty_rew = crashing_penalty_rew;
    env->log.hit_by_enemy_bullet_penalty_rew = hit_by_enemy_bullet_penalty_rew;

    if (env->enemy_respawns > 0) {
        crossed_screen =
            (float)env->enemy.crossed_screen / (env->enemy_respawns + 1);
    } else {
        crossed_screen = env->enemy.crossed_screen;  // No normalization needed
    }

    // Combine rewards into the total reward
    rew += score + fired_bullet_rew + bullet_distance_to_enemy_rew +
           flat_below_enemy_rew + vertical_closeness_rew - crossed_screen +
           hit_enemy_with_bullet_rew - danger_zone_penalty_rew;

    rew *= (1.0f +
            env->kill_streak * 0.1f);  // Reward scaling based on kill streak

    // Ensure rewards are <= 0 if the condition fails
    if (!(env->player.y > env->enemy.y + enemy_height &&
          fabs(playerCenterX - enemyCenterX) > enemy_width * 0.3f)) {
        rew = fminf(rew, 0.0f);  // Clamp reward to <= 0
    }

    env->rewards[0] = rew;
    env->log.episode_return += rew;

    if (env->bad_guy_score > 100.0f || env->player.score > env->max_score) {
        // env->player.lives = 0;
        env->game_over = true;
        env->terminals[0] = 1;
        compute_observations(env);
        add_log(env->log_buffer, &env->log);
        reset(env);
    }

    compute_observations(env);
    add_log(env->log_buffer, &env->log);
}

Client* make_client(BlastarEnv* env) {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Blastar");

    Client* client = (Client*)malloc(sizeof(Client));
    client->screen_width = SCREEN_WIDTH;
    client->screen_height = SCREEN_HEIGHT;
    client->player_width = client->player_height = env->player_height;
    client->enemy_width = enemy_width;
    client->enemy_height = enemy_height;
    SetTargetFPS(60);

    client->player_texture =
        LoadTexture("./pufferlib/resources/blastar/player_ship.png");
    client->enemy_texture =
        LoadTexture("./pufferlib/resources/blastar/enemy_ship.png");
    client->player_bullet_texture =
        LoadTexture("./pufferlib/resources/blastar/player_bullet.png");
    client->enemy_bullet_texture =
        LoadTexture("./pufferlib/resources/blastar/enemy_bullet.png");
    client->explosion_texture =
        LoadTexture("./pufferlib/resources/blastar/player_death_explosion.png");

    client->player_color = WHITE;
    client->enemy_color = WHITE;
    client->bullet_color = WHITE;
    client->explosion_color = WHITE;
    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void render(Client* client, BlastarEnv* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(BLACK);

    if (env->game_over) {
        DrawText("GAME OVER", client->screen_width / 2 - 60,
                 client->screen_height / 2 - 10, 30, RED);
        DrawText(TextFormat("FINAL SCORE: %d", env->player.score),
                 client->screen_width / 2 - 80, client->screen_height / 2 + 30,
                 20, GREEN);
        EndDrawing();
        return;
    }

    // Draw player or explosion on player death
    if (env->playerExplosionTimer > 0) {
        DrawTexture(client->explosion_texture, env->player.x, env->player.y,
                    client->explosion_color);
    } else if (env->player.lives > 0) {
        DrawTexture(client->player_texture, env->player.x, env->player.y,
                    client->player_color);
    }

    // Draw enemy or explosion on enemy death
    if (env->enemyExplosionTimer > 0) {
        DrawTexture(client->explosion_texture, env->enemy.x, env->enemy.y,
                    client->explosion_color);
    } else if (env->enemy.active) {
        DrawTexture(client->enemy_texture, env->enemy.x, env->enemy.y,
                    client->enemy_color);
    }

    // Draw player bullet
    if (env->player.bullet.active) {
        DrawTexture(client->player_bullet_texture, env->player.bullet.x,
                    env->player.bullet.y, client->bullet_color);
    }

    // Draw enemy bullet
    if (env->enemy.bullet.active) {
        DrawTexture(client->enemy_bullet_texture, env->enemy.bullet.x,
                    env->enemy.bullet.y, client->bullet_color);
    }

    // Draw status beam indicator
    if (env->player.playerStuck) {
        DrawText("Status Beam", client->screen_width - 150,
                 client->screen_height / 3, 20, RED);
    }

    // Draw score and lives
    DrawText(TextFormat("BAD GUY SCORE %d", (int)env->bad_guy_score), 240, 10,
             20, GREEN);
    DrawText(TextFormat("PLAYER SCORE %d", env->player.score), 10, 10, 20,
             GREEN);
    DrawText(TextFormat("LIVES %d", env->player.lives),
             client->screen_width - 100, 10, 20, GREEN);

    EndDrawing();
}
