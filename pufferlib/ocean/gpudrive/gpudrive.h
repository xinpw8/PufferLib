#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <time.h>
#define MAX_AGENTS 64
#define MAX_ROAD_OBJECTS 200
#define ROAD_OBS 13
#define OTHER_AGENT_OBS 6
#define SELF_OBJS 6

// Entity Types
#define NONE 0
#define VEHICLE 1
#define PEDESTRIAN 2
#define CYCLIST 3
#define ROAD_LANE 4
#define ROAD_LINE 5
#define ROAD_EDGE 6
#define STOP_SIGN 7
#define CROSSWALK 8
#define SPEED_BUMP 9
#define DRIVEWAY 10

// Trajectory Length
#define TRAJECTORY_LENGTH 91

#define NOOP 0

typedef struct Entity Entity;
struct Entity {
    int type;
    int road_object_id;
    int road_point_id;
    int array_size;
    float* x;
    float* y;
    float* z;
    float* vx;
    float* vy;
    float* vz;
    float* heading;
    int* valid;
    float width;
    float length;
    float height;
    float goal_position_x;
    float goal_position_y;
    float goal_position_z;
    int collision_state;
};

void free_entity(Entity* entity){
    free(entity->x);
    free(entity->y);
    free(entity->z);
    free(entity->vx);
    free(entity->vy);
    free(entity->vz);
    free(entity->heading);
    free(entity->valid);
}



typedef struct GPUDrive GPUDrive;
struct GPUDrive {
    int num_agents;
    int active_agent_count;
    int* active_agent_indices;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;
    int human_agent_idx;
    Entity* entities;
    int num_entities;
    int timestep;
};

Entity* load_map_binary(const char* filename, GPUDrive* env) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;
    
    fread(&env->num_entities, sizeof(int), 1, file);
    fread(&env->active_agent_count, sizeof(int), 1, file);
    Entity* entities = (Entity*)malloc(env->num_entities * sizeof(Entity));
    env->active_agent_indices = (int*)malloc(env->active_agent_count * sizeof(int));
    fread(env->active_agent_indices, sizeof(int), env->active_agent_count, file);
    
    for (int i = 0; i < env->num_entities; i++) {
        // Read base entity data
        fread(&entities[i].type, sizeof(int), 1, file);
        fread(&entities[i].road_object_id, sizeof(int), 1, file);
        fread(&entities[i].road_point_id, sizeof(int), 1, file);
        fread(&entities[i].array_size, sizeof(int), 1, file);
        // Allocate arrays based on type
        int size = entities[i].array_size;
        entities[i].x = (float*)malloc(size * sizeof(float));
        entities[i].y = (float*)malloc(size * sizeof(float));
        entities[i].z = (float*)malloc(size * sizeof(float));
        if (entities[i].type == 1 || entities[i].type == 2 || entities[i].type == 3) {  // Object type
            // Allocate arrays for object-specific data
            entities[i].vx = (float*)malloc(size * sizeof(float));
            entities[i].vy = (float*)malloc(size * sizeof(float));
            entities[i].vz = (float*)malloc(size * sizeof(float));
            entities[i].heading = (float*)malloc(size * sizeof(float));
            entities[i].valid = (int*)malloc(size * sizeof(int));
        } else {
            // Roads don't use these arrays
            entities[i].vx = NULL;
            entities[i].vy = NULL;
            entities[i].vz = NULL;
            entities[i].heading = NULL;
            entities[i].valid = NULL;
        }
        // Read array data
        fread(entities[i].x, sizeof(float), size, file);
        fread(entities[i].y, sizeof(float), size, file);
        fread(entities[i].z, sizeof(float), size, file);
        if (entities[i].type == 1 || entities[i].type == 2 || entities[i].type == 3) {  // Object type
            fread(entities[i].vx, sizeof(float), size, file);
            fread(entities[i].vy, sizeof(float), size, file);
            fread(entities[i].vz, sizeof(float), size, file);
            fread(entities[i].heading, sizeof(float), size, file);
            fread(entities[i].valid, sizeof(int), size, file);
        }
        // Read remaining scalar fields
        fread(&entities[i].width, sizeof(float), 1, file);
        fread(&entities[i].length, sizeof(float), 1, file);
        fread(&entities[i].height, sizeof(float), 1, file);
        fread(&entities[i].goal_position_x, sizeof(float), 1, file);
        fread(&entities[i].goal_position_y, sizeof(float), 1, file);
        fread(&entities[i].goal_position_z, sizeof(float), 1, file);
    }
    fclose(file);
    return entities;
}
void print_entities(GPUDrive* env, int idx){
    // for(int i = 0; i < env->num_entities; i++){
    //     // if(env->entities[i].type >3){
    //         printf("entity %d type: %d\n", i, env->entities[i].type);
    //         printf("entity %d array_size: %d\n", i, env->entities[i].array_size);
    //     // }
    // }
    for (int i = 0; i < env->num_entities; i++){
        printf("entity %d type: %d\n", i, env->entities[i].type);
        printf("entity %d array_size: %d\n", i, env->entities[i].array_size);
    }
    for(int i = 0; i < env->entities[idx].array_size; i++){
        printf("entity %d x: %f\n", idx, env->entities[idx].x[i]);
        printf("entity %d y: %f\n", idx, env->entities[idx].y[i]);
        printf("entity %d z: %f\n", idx, env->entities[idx].z[i]);
        if(env->entities[idx].type == 1 || env->entities[idx].type == 2 || env->entities[idx].type == 3){
            printf("entity %d heading: %f\n", idx, env->entities[idx].heading[i]);
            printf("entity %d vx: %f\n", idx, env->entities[idx].vx[i]);
            printf("entity %d vy: %f\n", idx, env->entities[idx].vy[i]);
            printf("entity %d vz: %f\n", idx, env->entities[idx].vz[i]);
            printf("entity %d valid: %d\n", idx, env->entities[idx].valid[i]);
        }
    }
    printf("entity %d width: %f\n", idx, env->entities[idx].width);
    printf("entity %d length: %f\n", idx, env->entities[idx].length);
    printf("entity %d height: %f\n", idx, env->entities[idx].height);
    printf("entity %d goal_position_x: %f\n", idx, env->entities[idx].goal_position_x);
    printf("entity %d goal_position_y: %f\n", idx, env->entities[idx].goal_position_y);
    printf("entity %d goal_position_z: %f\n", idx, env->entities[idx].goal_position_z);
}

void init(GPUDrive* env){
    env->human_agent_idx = 0;
    env->timestep = 0;
    env->entities = load_map_binary("map.bin", env);
    printf("num_entities: %d\n", env->num_entities);
    printf("active_agent_count: %d\n", env->active_agent_count);
    printf("active_agent_indices: ");
    for(int i = 0; i < env->active_agent_count; i++){
        printf("%d ", env->active_agent_indices[i]);
    }
    printf("\n");

    print_entities(env, 16);
}

void free_initialized(GPUDrive* env){
    for(int i = 0; i < env->num_entities; i++){
        free_entity(&env->entities[i]);
    }
    free(env->entities);
    free(env->active_agent_indices);
}

void allocate(GPUDrive* env){
    int max_obs = SELF_OBJS + (MAX_AGENTS - 1)*OTHER_AGENT_OBS + MAX_ROAD_OBJECTS*ROAD_OBS;
    printf("MAX_OBS: %d\n", max_obs);
    env->observations = (float*)calloc(env->num_agents * max_obs, sizeof(float));
    env->actions = (int*)calloc(env->num_agents, sizeof(int));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->dones = (unsigned char*)calloc(env->num_agents, sizeof(unsigned char));
    init(env);
}

void free_allocated(GPUDrive* env){
    free_initialized(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->dones);
}

void c_reset(GPUDrive* env){
    env->timestep = 0;
}

int c_step(GPUDrive* env){
    env->timestep++;
    if(env->timestep == 91){
        env->timestep = 0;
        return 1;
    }
    return 0;
}   

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D puffers;
    Vector3 camera_target;
    float camera_zoom;
    Camera3D camera;
};


Client* make_client(GPUDrive* env){
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = 1280;
    client->height = 704;
    InitWindow(client->width, client->height, "PufferLib Ray GPU Drive");
    SetTargetFPS(30);
    client->puffers = LoadTexture("resources/puffers_128.png");
    
    // Get initial target position from first active agent
    Vector3 target_pos = {
        env->entities[env->active_agent_indices[0]].x[0],
        env->entities[env->active_agent_indices[0]].y[0],  // Y is up
        env->entities[env->active_agent_indices[0]].z[0]   // Z is depth
    };
    
    // Set up camera to look at target from above and behind
    client->camera.position = (Vector3){ 
        target_pos.x,           // Same X as target
        target_pos.y + 80.0f,   // 20 units above target
        target_pos.z + 50.0f    // 20 units behind target
    };
    client->camera.target = target_pos;
    client->camera.up = (Vector3){ 0.0f, -1.0f, 0.0f };  // Y is up
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;
    
    client->camera_zoom = 3.5f;
    return client;
}


void c_render(Client* client, GPUDrive* env) {
    BeginDrawing();
    ClearBackground(RAYWHITE);
    
    BeginMode3D(client->camera);
    
    // Draw a grid to help with orientation
    DrawGrid(20, 1.0f);
    
    for(int i = 0; i < env->num_entities; i++) {
        if(env->entities[i].type == 1 || env->entities[i].type == 2 || env->entities[i].type == 3) {
            for(int j = 0; j < env->active_agent_count; j++) {
                if(env->active_agent_indices[j] == i) {
                    // Get current heading
                    float heading = env->entities[i].heading[env->timestep];
                    // Create position vector (Y is up, Z is depth)
                    Vector3 position = {
                        env->entities[i].x[env->timestep],
                        env->entities[i].y[env->timestep],
                        env->entities[i].z[env->timestep]
                    };
                    
                    // Create size vector
                    Vector3 size = {
                        env->entities[i].length,
                        env->entities[i].width,
                        env->entities[i].height
                    };
                    
                    // Save current transform
                    rlPushMatrix();
                    
                    // Translate to position, rotate around Y axis, then draw
                    rlTranslatef(position.x, position.y, position.z);
                    rlRotatef(heading * RAD2DEG, 0.0f, 0.0f, 1.0f);  // Convert radians to degrees
                    
                    // Draw cube centered at origin (will be transformed by matrix)
                    Color object_color;
                    if(env->entities[i].type == 1){
                        object_color = DARKBLUE;
                    }
                    else if(env->entities[i].type == 2){
                        object_color = BEIGE;
                    }
                    else if(env->entities[i].type == 3){
                        object_color = YELLOW;
                    }
                    DrawCube((Vector3){0, 0, 0}, size.x, size.y, size.z, object_color);
                    DrawCubeWires((Vector3){0, 0, 0}, size.x, size.y, size.z, BLACK);
                    
                    // Restore previous transform
                    rlPopMatrix();
                    
                    // Draw goal position
                    DrawSphere((Vector3){
                        env->entities[i].goal_position_x,
                        env->entities[i].goal_position_y,
                        env->entities[i].goal_position_z
                    }, 0.5f, DARKGREEN);
                }
            }
        }
        
        // Draw road elements
        if(env->entities[i].type > 3 && env->entities[i].type < 7) {
            for(int j = 0; j < env->entities[i].array_size - 1; j++) {
                Vector3 start = {
                    env->entities[i].x[j],
                    env->entities[i].y[j],
                    env->entities[i].z[j]
                };
                Vector3 end = {
                    env->entities[i].x[j + 1],
                    env->entities[i].y[j + 1],
                    env->entities[i].z[j + 1]
                };
                
                Color lineColor = GRAY;
                if (env->entities[i].type == ROAD_LINE) lineColor = BLUE;
                else if (env->entities[i].type == ROAD_EDGE) lineColor = BLACK;
                else if (env->entities[i].type == DRIVEWAY) lineColor = RED;
                
                DrawLine3D(start, end, lineColor);
            }
        }
    }
    
    EndMode3D();
    
    // Draw debug info
    DrawText(TextFormat("Camera Position: (%.2f, %.2f, %.2f)", 
        client->camera.position.x, 
        client->camera.position.y, 
        client->camera.position.z), 10, 10, 20, BLACK);
    DrawText(TextFormat("Camera Target: (%.2f, %.2f, %.2f)", 
        client->camera.target.x, 
        client->camera.target.y, 
        client->camera.target.z), 10, 30, 20, BLACK);
    DrawText(TextFormat("Timestep: %d", env->timestep), 10, 50, 20, BLACK);
    
    EndDrawing();
}

void close_client(Client* client){
    CloseWindow();
    free(client);
}

