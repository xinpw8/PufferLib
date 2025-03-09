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

// Actions
#define NOOP 0

// Dynamics Models
#define CLASSIC 0
#define INVERTIBLE_BICYLE 1
#define DELTA_LOCAL 2
#define STATE_DYNAMICS 3

#define LOG_BUFFER_SIZE 1024

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
};


typedef struct LogBuffer LogBuffer;
struct LogBuffer {
    Log* logs;
    int length;
    int idx;
};

LogBuffer* allocate_logbuffer(int size) {
    LogBuffer* logs = (LogBuffer*)calloc(1, sizeof(LogBuffer));
    logs->logs = (Log*)calloc(size, sizeof(Log));
    logs->length = size;
    logs->idx = 0;
    return logs;
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
    //printf("Log: %f, %f, %f\n", log->episode_return, log->episode_length, log->score);
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return += logs->logs[i].episode_return;
        log.episode_length += logs->logs[i].episode_length;
 
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    logs->idx = 0;
    return log;
}

typedef struct Entity Entity;
struct Entity {
    int type;
    int road_object_id;
    int road_point_id;
    int array_size;
    float* traj_x;
    float* traj_y;
    float* traj_z;
    float* traj_vx;
    float* traj_vy;
    float* traj_vz;
    float* traj_heading;
    int* traj_valid;
    float width;
    float length;
    float height;
    float goal_position_x;
    float goal_position_y;
    float goal_position_z;
    int collision_state;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    float heading;
    int valid;
};

void free_entity(Entity* entity){
    // free trajectory arrays
    free(entity->traj_x);
    free(entity->traj_y);
    free(entity->traj_z);
    free(entity->traj_vx);
    free(entity->traj_vy);
    free(entity->traj_vz);
    free(entity->traj_heading);
    free(entity->traj_valid);
}



typedef struct GPUDrive GPUDrive;
struct GPUDrive {
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;
    Log* logs;
    int num_agents;
    int active_agent_count;
    int* active_agent_indices;
    int human_agent_idx;
    Entity* entities;
    int num_entities;
    int timestep;
    int dynamics_model;
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
        entities[i].traj_x = (float*)malloc(size * sizeof(float));
        entities[i].traj_y = (float*)malloc(size * sizeof(float));
        entities[i].traj_z = (float*)malloc(size * sizeof(float));
        if (entities[i].type == 1 || entities[i].type == 2 || entities[i].type == 3) {  // Object type
            // Allocate arrays for object-specific data
            entities[i].traj_vx = (float*)malloc(size * sizeof(float));
            entities[i].traj_vy = (float*)malloc(size * sizeof(float));
            entities[i].traj_vz = (float*)malloc(size * sizeof(float));
            entities[i].traj_heading = (float*)malloc(size * sizeof(float));
            entities[i].traj_valid = (int*)malloc(size * sizeof(int));
        } else {
            // Roads don't use these arrays
            entities[i].traj_vx = NULL;
            entities[i].traj_vy = NULL;
            entities[i].traj_vz = NULL;
            entities[i].traj_heading = NULL;
            entities[i].traj_valid = NULL;
        }
        // Read array data
        fread(entities[i].traj_x, sizeof(float), size, file);
        fread(entities[i].traj_y, sizeof(float), size, file);
        fread(entities[i].traj_z, sizeof(float), size, file);
        if (entities[i].type == 1 || entities[i].type == 2 || entities[i].type == 3) {  // Object type
            fread(entities[i].traj_vx, sizeof(float), size, file);
            fread(entities[i].traj_vy, sizeof(float), size, file);
            fread(entities[i].traj_vz, sizeof(float), size, file);
            fread(entities[i].traj_heading, sizeof(float), size, file);
            fread(entities[i].traj_valid, sizeof(int), size, file);
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
        printf("entity %d x: %f\n", idx, env->entities[idx].traj_x[i]);
        printf("entity %d y: %f\n", idx, env->entities[idx].traj_y[i]);
        printf("entity %d z: %f\n", idx, env->entities[idx].traj_z[i]);
        if(env->entities[idx].type == 1 || env->entities[idx].type == 2 || env->entities[idx].type == 3){
            printf("entity %d heading: %f\n", idx, env->entities[idx].traj_heading[i]);
            printf("entity %d vx: %f\n", idx, env->entities[idx].traj_vx[i]);
            printf("entity %d vy: %f\n", idx, env->entities[idx].traj_vy[i]);
            printf("entity %d vz: %f\n", idx, env->entities[idx].traj_vz[i]);
            printf("entity %d valid: %d\n", idx, env->entities[idx].traj_valid[i]);
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
    env->dynamics_model = CLASSIC;
    env->logs = (Log*)calloc(env->active_agent_count, sizeof(Log));
    printf("num_entities: %d\n", env->num_entities);
    printf("active_agent_count: %d\n", env->active_agent_count);
    printf("active_agent_indices: ");
    for(int i = 0; i < env->active_agent_count; i++){
        printf("%d ", env->active_agent_indices[i]);
        env->entities[env->active_agent_indices[i]].x = env->entities[env->active_agent_indices[i]].traj_x[0];
        env->entities[env->active_agent_indices[i]].y = env->entities[env->active_agent_indices[i]].traj_y[0];
        env->entities[env->active_agent_indices[i]].z = env->entities[env->active_agent_indices[i]].traj_z[0];
        env->entities[env->active_agent_indices[i]].vx = env->entities[env->active_agent_indices[i]].traj_vx[0];
        env->entities[env->active_agent_indices[i]].vy = env->entities[env->active_agent_indices[i]].traj_vy[0];
        env->entities[env->active_agent_indices[i]].vz = env->entities[env->active_agent_indices[i]].traj_vz[0];
        env->entities[env->active_agent_indices[i]].heading = env->entities[env->active_agent_indices[i]].traj_heading[0];
        env->entities[env->active_agent_indices[i]].valid = env->entities[env->active_agent_indices[i]].traj_valid[0];
    }
}

void free_initialized(GPUDrive* env){
    for(int i = 0; i < env->num_entities; i++){
        free_entity(&env->entities[i]);
    }
    free(env->entities);
    free(env->active_agent_indices);
    free(env->logs);
}

void allocate(GPUDrive* env){
    int max_obs = SELF_OBJS + (MAX_AGENTS - 1)*OTHER_AGENT_OBS + MAX_ROAD_OBJECTS*ROAD_OBS;
    printf("MAX_OBS: %d\n", max_obs);
    init(env);
    env->observations = (float*)calloc(env->active_agent_count * max_obs, sizeof(float));
    env->actions = (int*)calloc(env->active_agent_count*2, sizeof(int));
    env->rewards = (float*)calloc(env->active_agent_count, sizeof(float));
    env->dones = (unsigned char*)calloc(env->active_agent_count, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_allocated(GPUDrive* env){
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->dones);
    free_logbuffer(env->log_buffer);
    free_initialized(env);
}

void c_reset(GPUDrive* env){
    env->timestep = 0;
    for(int x = 0;x<env->active_agent_count; x++){
        env->logs[x] = (Log){0};
    }
}

void move_dynamics(GPUDrive* env, int* actions, int action_idx, int agent_idx){
    if(env->dynamics_model == CLASSIC){
        // clip acceleration & steering
        Entity* agent = &env->entities[agent_idx];
        
        // Extract action components directly from the multi-discrete action array
        int (*action_array)[2] = (int(*)[2])env->actions;
        float acceleration = action_array[action_idx][0];
        float steering = action_array[action_idx][1];

        
        // Clip acceleration and steering
        acceleration = fmaxf(-6.0f, fminf(acceleration, 6.0f));
        steering = fmaxf(-3.0f, fminf(steering, 3.0f));
        
        // Current state
        float x = agent->x;
        float y = agent->y;
        float heading = agent->heading;
        float vx = agent->vx;
        float vy = agent->vy;
        
        // Calculate current speed
        float speed = sqrtf(vx*vx + vy*vy);
        
        // Time step (adjust as needed)
        const float dt = 0.1f;
        
        // Update position using current velocity and acceleration
        float new_x = x + vx * dt + 0.5f * acceleration * cosf(heading) * dt * dt;
        float new_y = y + vy * dt + 0.5f * acceleration * sinf(heading) * dt * dt;
        
        // Calculate heading change based on steering and speed
        float delta_heading = steering * (speed * dt + 0.5f * acceleration * dt * dt);
        float new_heading = heading + delta_heading;
        
        // Normalize heading to [-π, π]
        while (new_heading > M_PI) new_heading -= 2.0f * M_PI;
        while (new_heading < -M_PI) new_heading += 2.0f * M_PI;
        
        // Update speed and velocity components
        float new_speed = speed + acceleration * dt;
        float new_vx = new_speed * cosf(new_heading);
        float new_vy = new_speed * sinf(new_heading);
        
        // Update agent state for next timestep
        agent->x = new_x;
        agent->y = new_y;
        agent->z = agent->traj_z[env->timestep];
        agent->heading = new_heading;
        agent->vx = new_vx;
        agent->vy = new_vy;
    }
    else if(env->dynamics_model == INVERTIBLE_BICYLE){
        // Invertible bicycle dynamics model
    }
}

void move_expert(GPUDrive* env, int* actions, int agent_idx){
    Entity* agent = &env->entities[agent_idx];
    agent->x = agent->traj_x[env->timestep];
    agent->y = agent->traj_y[env->timestep];
    agent->z = agent->traj_z[env->timestep];
    agent->heading = agent->traj_heading[env->timestep];
}

void move_random(GPUDrive* env, int* actions, int agent_idx){
    Entity* agent = &env->entities[agent_idx];
    float x_rand = agent->x + (rand() % 100 - 50) / 100.0f;
    float y_rand = agent->y + (rand() % 100 - 50) / 100.0f;
    float z_rand = agent->z + (rand() % 100 - 50) / 100.0f;
    float heading_rand = agent->heading + (rand() % 100 - 50) / 100.0f;
    agent->x = x_rand;
    agent->y = y_rand;
    agent->z = z_rand;
    agent->heading = heading_rand;
}

int c_step(GPUDrive* env){
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    env->timestep++;    
    // Process actions for all active agents
    if(env->timestep == 91){
        c_reset(env);
    }
    for(int i=0; i<env->active_agent_count; i++){
        int agent_idx = env->active_agent_indices[i];
        move_dynamics(env, env->actions, i, agent_idx);
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
        env->entities[env->active_agent_indices[0]].x,
        env->entities[env->active_agent_indices[0]].y,  // Y is up
        env->entities[env->active_agent_indices[0]].z   // Z is depth
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
                if(env->active_agent_indices[j] != i) {
                    continue;
                }
                // Get current heading
                float heading = env->entities[i].heading;
                // Create position vector (Y is up, Z is depth)
                Vector3 position = {
                    env->entities[i].x,
                    env->entities[i].y,
                    env->entities[i].z
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
                Color object_color = PINK;
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
        
        // Draw road elements
        if(env->entities[i].type > 3 && env->entities[i].type < 7) {
            for(int j = 0; j < env->entities[i].array_size - 1; j++) {
                Vector3 start = {
                    env->entities[i].traj_x[j],
                    env->entities[i].traj_y[j],
                    env->entities[i].traj_z[j]
                };
                Vector3 end = {
                    env->entities[i].traj_x[j + 1],
                    env->entities[i].traj_y[j + 1],
                    env->entities[i].traj_z[j + 1]
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
    // acceleration & steering
    int human_idx = env->active_agent_indices[env->human_agent_idx];
    DrawText(TextFormat("Controlling Agent: %d", env->human_agent_idx), 10, 70, 20, BLACK);
    DrawText(TextFormat("Agent Index: %d", human_idx), 10, 90, 20, BLACK);
    // Controls help
    DrawText("Controls: W/S - Accelerate/Brake, A/D - Steer, 1-4 - Switch Agent", 
             10, client->height - 30, 20, BLACK);
    EndDrawing();
}

void close_client(Client* client){
    CloseWindow();
    free(client);
}

