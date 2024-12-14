#ifndef BLASTAR_ENV_H
#define BLASTAR_ENV_H

#include <math.h>
#include <stdbool.h>
#include "raylib.h"
#include <stdio.h>

#define MAX_BULLETS 10
#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480

typedef struct {
    float x, y;        // Position
    bool active;       // Bullet state
} Bullet;

typedef struct {
    float x, y;        // Enemy position
    bool active;       // Enemy state
    int direction;     // Movement direction (-1 left, 1 right)
    int width;
    int height;
} Enemy;

typedef struct {
    float x, y;        // Player position
    int score;         // Player score
    int lives;         // Remaining lives
    Bullet bullets[MAX_BULLETS];  // Player bullets
    bool bulletFired;  // Whether a bullet is in the air
} Player;

typedef struct {
    Player player;
    Enemy enemy;
    Bullet bomb;       // Enemy bomb
    bool playerStuck;  // Whether the player is stuck under the enemy bomb
    bool gameOver;
    int tick;
} BlastarEnv;

void init_blastar(BlastarEnv *env);
void reset_blastar(BlastarEnv *env);
void step_blastar(BlastarEnv *env, int action);

void init_blastar(BlastarEnv *env) {
    env->player.x = 320;            // Center of the screen
    env->player.y = 480 - 60;       // 2 ship lengths off the bottom
    env->player.score = 0;
    env->player.lives = 5;
    env->player.bulletFired = false;

    env->enemy.x = -30;             // Start off the screen to the left
    env->enemy.y = 50;              // Top of the screen
    env->enemy.active = true;
    env->enemy.direction = 1;       // Always moves right
    env->enemy.width = 30;          // Actual sprite width
    env->enemy.height = 30;         // Actual sprite height

    env->bomb.active = false;       // Initialize bomb state
    env->gameOver = false;
    env->tick = 0;
}


void reset_blastar(BlastarEnv *env) {
    init_blastar(env);
}

void step_blastar(BlastarEnv *env, int action) {
    env->tick++;


// Define hitboxes for the player and the bomb
Rectangle playerHitbox = {
    env->player.x - 15, // Centered on the player's sprite
    env->player.y - 15,
    30, // Player's width
    30  // Player's height
};

Rectangle bombHitbox = {
    env->bomb.x - 5, // Centered on the bomb's sprite
    env->bomb.y - 5,
    10, // Bomb's width
    10  // Bomb's height
};

// Check collision before updating the bomb's position
if (env->bomb.active && CheckCollisionRecs(playerHitbox, bombHitbox)) {
    printf("Collision detected! Bomb: (%f, %f), Player: (%f, %f)\n",
           env->bomb.x, env->bomb.y, env->player.x, env->player.y);

    // Handle collision effects
    env->bomb.active = false;      // Deactivate bomb
    env->player.lives--;           // Decrease player's lives
    env->player.x = 320;           // Reset player position (example)
    env->player.y = 330;
    return;                        // Exit function early
}

// Update bomb's position after collision check
if (env->bomb.active) {
    env->bomb.y += 2.0; // Bomb's downward movement speed
    if (env->bomb.y > SCREEN_HEIGHT) {
        env->bomb.active = false; // Deactivate bomb if off-screen
    }
}

    // Handle player controls only if not stuck
    if (!env->playerStuck) {
        if (action == 1 && env->player.x > 0) { // Left
            env->player.x -= 6;
        } else if (action == 2 && env->player.x < 640 - 30) { // Right
            env->player.x += 6;
        } else if (action == 3 && env->player.y > 0) { // Up
            env->player.y -= 6;
        } else if (action == 4 && env->player.y < 480 - 30) { // Down
            env->player.y += 6;
        }
    }

    // Fire player bullet
    if (action == 5) {
        if (env->player.bulletFired) {
            env->player.bullets[0].active = false;
            env->player.bulletFired = false;
        }
        env->player.bulletFired = true;
        env->player.bullets[0].x = env->player.x + 15;
        env->player.bullets[0].y = env->player.y;
        env->player.bullets[0].active = true;
    }

    // Update player bullet
    if (env->player.bulletFired) {
        env->player.bullets[0].y -= 10;
        if (env->player.bullets[0].y < 0) {
            env->player.bullets[0].active = false;
            env->player.bulletFired = false;
        }
    }

    // Enemy movement
    if (!env->bomb.active) {
        env->enemy.x += env->enemy.direction;
        if (env->enemy.x > 640) { // Reset enemy when it leaves the screen
            env->enemy.x = -env->enemy.width; 
            env->enemy.y += 30;
            if (env->enemy.y > 450) {
                env->enemy.y = 50; // Reset to top if too low
            }
        }
    }

// Enemy bomb logic: Stop and fire bomb if aligned
if (!env->bomb.active && fabs((env->player.x + 15) - (env->enemy.x + env->enemy.width / 2)) < 5) {
    env->bomb.x = env->enemy.x + env->enemy.width / 2 + 20; // Center bomb under enemy
    env->bomb.y = env->enemy.y + env->enemy.height;       // Place bomb below enemy
    env->bomb.active = true;

    // Stop the enemy ship
    env->playerStuck = true;
}


// Update bomb
if (env->bomb.active) {
    env->bomb.y += 2; // Slow fall
    printf("Bomb: (%f, %f)\n", env->bomb.x, env->bomb.y);

    // Collision check between bomb and player
    Rectangle bombHitbox = {
        env->bomb.x - 5, 
        env->bomb.y - 5,
        10,
        10
    };

    if (CheckCollisionRecs(playerHitbox, bombHitbox)) {
        env->bomb.active = false;
        env->player.lives--;

        // Reset player and enemy positions
        env->player.x = 320;
        env->player.y = 480 - 60; // Spawn 2 ship lengths off the bottom
        env->enemy.x = -env->enemy.width;
        env->enemy.y += 30; // Lower enemy on player death
        if (env->enemy.y > 450) {
            env->enemy.y = 50; // Reset enemy to top if too low
        }
        env->playerStuck = false;
    }

    if (env->bomb.y > 480) { // Deactivate bomb if it leaves the screen
        env->bomb.active = false;
        env->playerStuck = false;
    }
}


    // Update bullet collision with enemy
    Rectangle enemyHitbox = {
        env->enemy.x,
        env->enemy.y,
        env->enemy.width,
        env->enemy.height
    };
    Rectangle bulletHitbox = {
        env->player.bullets[0].x,
        env->player.bullets[0].y,
        5,  // Bullet width
        10  // Bullet height
    };
    if (env->player.bulletFired && CheckCollisionRecs(bulletHitbox, enemyHitbox)) {
        env->player.bulletFired = false;
        env->player.bullets[0].active = false;
        env->player.score++;

        env->enemy.x = -env->enemy.width;
        env->enemy.y += 30;
        if (env->enemy.y > 450) {
            env->enemy.y = 50;
        }
    }

    // Check for game over
    if (env->player.lives <= 0) {
        env->gameOver = true;
    }

    printf("Player: (%f, %f)\n", env->player.x, env->player.y);
    printf("Enemy: (%f, %f)\n", env->enemy.x, env->enemy.y);
    printf("Bomb: (%f, %f)\n", env->bomb.x, env->bomb.y);
    printf("Player Hitbox: x=%f, y=%f, w=%f, h=%f\n",
       playerHitbox.x, playerHitbox.y, playerHitbox.width, playerHitbox.height);
printf("Bomb Hitbox: x=%f, y=%f, w=%f, h=%f\n",
       bombHitbox.x, bombHitbox.y, bombHitbox.width, bombHitbox.height);

}



typedef struct {
    Texture2D playerTexture;
    Texture2D enemyTexture;
    Texture2D bulletTexture;
} BlastarRenderer;

void init_renderer(BlastarRenderer *renderer) {
    InitWindow(640, 480, "Blastar");
    SetTargetFPS(60);

    renderer->playerTexture = LoadTexture("pufferlib/resources/blastar/player.png");
    renderer->enemyTexture = LoadTexture("pufferlib/resources/blastar/enemy.png");
    renderer->bulletTexture = LoadTexture("pufferlib/resources/blastar/bullet.png");
}

void close_renderer(BlastarRenderer *renderer) {
    UnloadTexture(renderer->playerTexture);
    UnloadTexture(renderer->enemyTexture);
    UnloadTexture(renderer->bulletTexture);
    CloseWindow();
}

void render_blastar(BlastarRenderer *renderer, BlastarEnv *env) {
    BeginDrawing();
    ClearBackground(BLACK);

    // Draw player
    DrawTexture(renderer->playerTexture, env->player.x, env->player.y, WHITE);

    // Draw enemy
    DrawTexturePro(
        renderer->enemyTexture,
        (Rectangle){0, 0, env->enemy.width, env->enemy.height}, // Source
        (Rectangle){env->enemy.x, env->enemy.y, env->enemy.width, env->enemy.height}, // Destination
        (Vector2){0, 0}, // Origin
        0,               // Rotation
        WHITE
    );


    // Draw player bullet
    if (env->player.bullets[0].active) {
        DrawTexture(renderer->bulletTexture, env->player.bullets[0].x, env->player.bullets[0].y, WHITE);
    }

    // Draw bomb
    if (env->bomb.active) {
        DrawCircle(env->bomb.x, env->bomb.y, 10, RED); // Represent the bomb as a red circle

        // Draw STATUS BEAM text
        DrawText("STATUS BEAM", 480, 160, 20, RED); // 1/3 from the bottom-right
    }

    // Draw score and lives
    DrawText(TextFormat("SCORE %d", env->player.score), 10, 10, 20, GREEN);
    DrawText(TextFormat("SHIPS %d", env->player.lives), 540, 10, 20, GREEN);

    EndDrawing();
}


#endif // BLASTAR_ENV_H