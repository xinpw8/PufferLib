#ifndef RACING_H
#define RACING_H

#include <stdlib.h>
#include "raylib.h"  // Assuming Raylib is used for rendering

// Define the Client struct
typedef struct Client {
    int width;
    int height;
} Client;

typedef struct CRacing {
    float car_x;
    float car_y;
    float car_speed;
    unsigned char* actions;
    float* rewards;
    unsigned char* dones;
    int width;
    int height;
    // Add other necessary members for the racing environment
} CRacing;

void allocate(CRacing* env);
void reset(CRacing* env);
void step(CRacing* env);
void render(Client* client, CRacing* env);
void free_allocated(CRacing* env);

Client* make_client(CRacing* env);
void close_client(Client* client);

#endif // RACING_H
