// blastar_renderer.h
#ifndef BLASTAR_RENDERER_H
#define BLASTAR_RENDERER_H

#include "raylib.h"

// Forward declaration of BlastarEnv
typedef struct BlastarEnv BlastarEnv;

// Define the BlastarRenderer struct
typedef struct {
    Texture2D playerTexture;       // Texture for the player's ship
    Texture2D enemyTexture;        // Texture for the enemy's ship
    Texture2D playerBulletTexture; // Texture for the player's bullets
    Texture2D enemyBulletTexture;  // Texture for the enemy's bullets
    Texture2D playerDeathExplosion;// Texture for the player's death explosion
} BlastarRenderer;

// Define the Client struct
typedef struct Client {
    BlastarRenderer renderer;
} Client;

// Function declarations for renderer
void init_renderer(BlastarRenderer *renderer);
void render_blastar(const BlastarRenderer *renderer, const BlastarEnv *env);
void close_renderer(BlastarRenderer *renderer);

// Client-related function declarations
Client* make_client(BlastarEnv* env);
void close_client(Client* client);
void render_blastar_client(Client* client, BlastarEnv* env);

#endif // BLASTAR_RENDERER_H
