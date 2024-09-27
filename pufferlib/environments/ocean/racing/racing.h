#ifndef RACING_H
#define RACING_H

#include <stdio.h>
#include <stdlib.h>
#include "raylib.h"  // Include Raylib for rendering

// Define the Client struct for handling window rendering
typedef struct Client {
    int width;
    int height;
} Client;

// Define the CRacing struct that stores all game state information
typedef struct CRacing {
    float car_x;        // X position of the player's car
    float car_y;        // Y position of the player's car
    float car_speed;    // Speed of the player's car
    float road_taper;   // Used to move the road vertically
    float road_shift;   // Used to shift the road horizontally (added this field)
    int width;          // Width of the game window
    int height;         // Height of the game window
} CRacing;

// Declare the functions for game logic
void allocate(CRacing* env);
void reset(CRacing* env);
void step(CRacing* env);
void render(Client* client, CRacing* env);
void free_allocated(CRacing* env);

Client* make_client(CRacing* env);
void close_client(Client* client);

#endif // RACING_H
