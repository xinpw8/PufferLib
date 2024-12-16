#ifndef BLASTAR_RENDERER_H
#define BLASTAR_RENDERER_H

#include "raylib.h"

// Forward declaration of BlastarEnv
typedef struct BlastarEnv BlastarEnv;

// Define the Client struct
typedef struct Client Client;
struct Client {
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
};

// Function declarations
Client* make_client(BlastarEnv* env);
void close_client(Client* client);
void c_render(Client* client, BlastarEnv* env);

#endif // BLASTAR_RENDERER_H
