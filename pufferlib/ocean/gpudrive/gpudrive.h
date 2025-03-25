#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
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

// collision state
#define NO_COLLISION 0
#define VEHICLE_COLLISION 1
#define OFFROAD 2

// grid cell size
#define GRID_CELL_SIZE 5.0f
#define MAX_ENTITIES_PER_CELL 10
#define SLOTS_PER_CELL (MAX_ENTITIES_PER_CELL*2 + 1)

// Max road segment observation entities
#define MAX_ROAD_SEGMENT_OBSERVATIONS 200

// Observation Space Constants
#define MAX_SPEED 100
#define MAX_VEH_LEN 30
#define MAX_VEH_WIDTH 15
#define MAX_VEH_HEIGHT 10
#define MIN_REL_GOAL_COORD -1000
#define MAX_REL_GOAL_COORD 1000
#define MIN_REL_AGENT_POS -1000
#define MAX_REL_AGENT_POS 1000
#define MAX_ORIENTATION_RAD 2 * PI
#define MIN_RG_COORD -1000
#define MAX_RG_COORD 1000

// Acceleration Values
static const float ACCELERATION_VALUES[7] = {-4.0000f, -2.6670f, -1.3330f, -0.0000f,  1.3330f,  2.6670f,  4.0000f};
static const float STEERING_VALUES[13] = {-3.1420f, -2.6180f, -2.0940f, -1.5710f, -1.0470f, -0.5240f,  0.0000f,  0.5240f,
         1.0470f,  1.5710f,  2.0940f,  2.6180f,  3.1420f};
static const float offsets[4][2] = {
        {-1, 1},  // top-left
        {1, 1},   // top-right
        {1, -1},  // bottom-right
        {-1, -1}  // bottom-left
    };

static const int collision_offsets[9][2] = {
        {-1, -1}, {0, -1}, {1, -1},  // Top row
        {-1,  0}, {0,  0}, {1,  0},  // Middle row (skip center)
        {-1,  1}, {0,  1}, {1,  1}   // Bottom row
    };
#define LOG_BUFFER_SIZE 1024

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float score;
    float offroad_rate;
    float collision_rate;
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
    //printf("Log: %f, %f,\n", log->episode_return, log->episode_length);
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return += logs->logs[i].episode_return / logs->idx;
        log.episode_length += logs->logs[i].episode_length / logs->idx;
	    log.score += logs->logs[i].score / logs->idx;
	    log.offroad_rate += logs->logs[i].offroad_rate / logs->idx;
	    log.collision_rate += logs->logs[i].collision_rate / logs->idx;
	//printf("length: %f", log.episode_length);
    }
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

float relative_distance(float a, float b){
    float distance = sqrtf(powf(a - b, 2));
    return distance;
}

float relative_distance_2d(float x1, float y1, float x2, float y2){
    float dx = x2 - x1;
    float dy = y2 - y1;
    float distance = sqrtf(dx*dx + dy*dy);
    return distance;
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
    int num_cars;
    int timestep;
    int dynamics_model;
    float* fake_data;
    char* goal_reached;
    float* map_corners;
    int* grid_cells;  // holds entity ids and geometry index per cell
    int grid_cols;
    int grid_rows;
    int vision_range;
    int* neighbor_offsets;
    int* neighbor_cache_entities;
    int* neighbor_cache_indices;
    float reward_vehicle_collision;
    float reward_offroad_collision;
};

Entity* load_map_binary(const char* filename, GPUDrive* env) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;
    
    fread(&env->num_entities, sizeof(int), 1, file);
    Entity* entities = (Entity*)malloc(env->num_entities * sizeof(Entity));
    //printf("Num entities: %d\n", env->num_entities);
    for (int i = 0; i < env->num_entities; i++) {
	// Read base entity data
        fread(&entities[i].type, sizeof(int), 1, file);
        if(entities[i].type < 4){
            fread(&entities[i].road_object_id, sizeof(int), 1, file);
            entities[i].road_point_id = -1;
        }
        else{
            fread(&entities[i].road_point_id, sizeof(int), 1, file);
            entities[i].road_object_id = -1;
        }
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

void set_start_position(GPUDrive* env){
    for(int i = 0; i < env->num_entities; i++){
        env->entities[i].x = env->entities[i].traj_x[0];
        env->entities[i].y = env->entities[i].traj_y[0];
        env->entities[i].z = env->entities[i].traj_z[0];
        if(env->entities[i].type == 1 || env->entities[i].type == 2 || env->entities[i].type == 3){
            env->entities[i].vx = env->entities[i].traj_vx[0];
            env->entities[i].vy = env->entities[i].traj_vy[0];
            env->entities[i].vz = env->entities[i].traj_vz[0];
            env->entities[i].heading = env->entities[i].traj_heading[0];
            env->entities[i].valid = env->entities[i].traj_valid[0];
        }
    }
}

void set_active_agents(GPUDrive* env){
    env->active_agent_count = 0;
    int active_agent_indices[env->num_entities];
    for(int i = 0; i < env->num_entities; i++){
        if(env->entities[i].type == 1){
            env->num_cars++;
            int start_idx=0;
            for(int j = 0; j<env->entities[i].array_size; j++){
                if(env->entities[i].traj_valid[j] == 1){
                    start_idx = j;
                    break;
                }
            }
            if(start_idx !=0) continue;
            float distance = relative_distance_2d(
                env->entities[i].traj_x[start_idx],
                env->entities[i].traj_y[start_idx],
                env->entities[i].goal_position_x,
                env->entities[i].goal_position_y);
            //printf("entity %d distance: %f\n", i, distance);
            if(distance >= 2.0f){
                active_agent_indices[env->active_agent_count] = i;
                env->active_agent_count++;
            }
        }
    }
    env->active_agent_indices = (int*)malloc(env->active_agent_count * sizeof(int));
    for(int i=0;i<env->active_agent_count;i++){
        env->active_agent_indices[i] = active_agent_indices[i];
    };
}

// Function to get grid index from coordinates
int getGridIndex(GPUDrive* env, float x1, float y1) {
    
    // Verify coordinate assumptions
    if (env->map_corners[0] >= env->map_corners[2] || env->map_corners[1] >= env->map_corners[3]) {
        printf("Invalid grid coordinates\n");
        return -1;  // Invalid grid coordinates
    }
    
    // Calculate dimensions of the world in meters
    float worldWidth = env->map_corners[2] - env->map_corners[0];   // Positive value
    float worldHeight = env->map_corners[3] - env->map_corners[1];  // Positive value
    // Calculate number of cells in each dimension
    // Each cell is 5 meters
    int cellsX = (int)ceil(worldWidth / GRID_CELL_SIZE);  // Number of columns
    int cellsY = (int)ceil(worldHeight / GRID_CELL_SIZE); // Number of rows
    // Calculate position relative to top-left corner
    float relativeX = x1 - env->map_corners[0];  // Distance from left
    float relativeY = y1 - env->map_corners[1];  // Distance from top
    // Calculate grid coordinates
    int gridX = (int)(relativeX / GRID_CELL_SIZE);  // Column index
    int gridY = (int)(relativeY / GRID_CELL_SIZE);  // Row index
    // Ensure the coordinates are within bounds
    if (gridX < 0 || gridX >= cellsX || gridY < 0 || gridY >= cellsY) {
        return -1;  // Return -1 for out of bounds
    }
    
    // Calculate 1D array index
    // Index = (row * number_of_columns) + column
    int index = (gridY * cellsX) + gridX;
    
    return index;
}

void add_entity_to_grid(GPUDrive* env, int grid_index, int entity_idx, int geometry_idx){
    if(grid_index == -1){
        return;
    }
    int base_index = grid_index * SLOTS_PER_CELL;
    int count = env->grid_cells[base_index];
    if(count>= MAX_ENTITIES_PER_CELL) return;
    env->grid_cells[base_index + count*2 + 1] = entity_idx;
    env->grid_cells[base_index + count*2 + 2] = geometry_idx;
    env->grid_cells[base_index] = count + 1;
    
}

void init_grid_map(GPUDrive* env){
    // Find top left and bottom right points of the map
    float top_left_x = env->entities[0].traj_x[0];
    float top_left_y = env->entities[0].traj_y[0];
    float bottom_right_x = env->entities[0].traj_x[0];
    float bottom_right_y = env->entities[0].traj_y[0];

    for(int i = 0; i < env->num_entities; i++){
        if(env->entities[i].type > 3 && env->entities[i].type < 7){
            // Check all points in the trajectory for road elements
            for(int j = 0; j < env->entities[i].array_size; j++){
                if(env->entities[i].traj_x[j] < top_left_x) top_left_x = env->entities[i].traj_x[j];
                if(env->entities[i].traj_x[j] > bottom_right_x) bottom_right_x = env->entities[i].traj_x[j];
                if(env->entities[i].traj_y[j] < top_left_y) top_left_y = env->entities[i].traj_y[j];
                if(env->entities[i].traj_y[j] > bottom_right_y) bottom_right_y = env->entities[i].traj_y[j];
            }
        }
    }

    env->map_corners = (float*)calloc(4, sizeof(float));
    env->map_corners[0] = top_left_x;
    env->map_corners[1] = top_left_y;
    env->map_corners[2] = bottom_right_x;
    env->map_corners[3] = bottom_right_y;
    
    // Calculate grid dimensions
    float grid_width = bottom_right_x - top_left_x;
    float grid_height = bottom_right_y - top_left_y;
    env->grid_cols = ceil(grid_width / GRID_CELL_SIZE);
    env->grid_rows = ceil(grid_height / GRID_CELL_SIZE);
    
    int grid_cell_count = env->grid_cols * env->grid_rows;
    env->grid_cells = (int*)calloc(grid_cell_count * SLOTS_PER_CELL, sizeof(int));
    
    // Populate grid cells
    for(int i = 0; i < env->num_entities; i++){
        if(env->entities[i].type == ROAD_EDGE){
            for(int j = 0; j < env->entities[i].array_size - 1; j++){
                float x_center = (env->entities[i].traj_x[j] + env->entities[i].traj_x[j+1]) / 2;
                float y_center = (env->entities[i].traj_y[j] + env->entities[i].traj_y[j+1]) / 2;
                int grid_index = getGridIndex(env, x_center, y_center);
                add_entity_to_grid(env, grid_index, i, j);
            }
        }
    }    
}

void init_neighbor_offsets(GPUDrive* env) {
    // Allocate memory for the offsets
    env->neighbor_offsets = (int*)calloc(env->vision_range * env->vision_range * 2, sizeof(int));
    
    // The spiral moves in this sequence: right, up, left, left, down, down, right, right, right...
    // Direction vectors for moving right, up, left, down in a clockwise spiral
    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    
    int x = 0;    // Current x offset
    int y = 0;    // Current y offset
    int dir = 0;  // Current direction (0: right, 1: up, 2: left, 3: down)
    int steps_to_take = 1; // Number of steps in current direction
    int steps_taken = 0;   // Steps taken in current direction
    int segments_completed = 0; // Count of direction segments completed
    int total = 0; // Total offsets added
    int max_offsets = env->vision_range * env->vision_range;
    
    // Start at center (0,0)
    int curr_idx = 0;
    env->neighbor_offsets[curr_idx++] = 0;  // x offset
    env->neighbor_offsets[curr_idx++] = 0;  // y offset
    total++;
    
    // Generate spiral pattern
    while (total < max_offsets) {
        // Move in current direction
        x += dx[dir];
        y += dy[dir];
        
        // Only add if within vision range bounds
        if (abs(x) <= env->vision_range/2 && abs(y) <= env->vision_range/2) {
            env->neighbor_offsets[curr_idx++] = x;
            env->neighbor_offsets[curr_idx++] = y;
            total++;
        }
        
        steps_taken++;
        
        // Check if we need to change direction
        if (steps_taken == steps_to_take) {
            steps_taken = 0;  // Reset steps taken
            dir = (dir + 1) % 4;  // Change direction (clockwise: right->up->left->down)
            segments_completed++;
            
            // Increase step length every two direction changes
            if (segments_completed % 2 == 0) {
                steps_to_take++;
            }
        }
    }
    
    // Debug output to verify the first few offsets
    // printf("First few spiral offsets:\n");
    // for (int i = 0; i < (env->vision_range * env->vision_range); i++) {
    //     printf("(%d,%d) ", env->neighbor_offsets[i*2], env->neighbor_offsets[i*2+1]);
    // }
    // printf("\n");
}

void cache_neighbor_offsets(GPUDrive* env){
    int count = 0;
    int cell_count = env->grid_cols * env->grid_rows;
    for(int i = 0; i < cell_count; i++){
        int cell_x = i % env->grid_cols;  // Convert to 2D coordinates
        int cell_y = i / env->grid_cols;
        env->neighbor_cache_indices[i] = count;
        for(int j = 0; j< env->vision_range * env->vision_range; j++){
            int x = cell_x + env->neighbor_offsets[j*2];
            int y = cell_y + env->neighbor_offsets[j*2+1];
            int grid_index = env->grid_cols*y + x;
            if(x < 0 || x >= env->grid_cols || y < 0 || y >= env->grid_rows) continue;
            int grid_count = env->grid_cells[grid_index*SLOTS_PER_CELL];
            for(int k = 0; k < grid_count; k++){
                count+=2;
            }
        }
    }
    env->neighbor_cache_indices[cell_count] = count;
    env->neighbor_cache_entities = (int*)calloc(count, sizeof(int));
    for(int i = 0; i< cell_count; i ++){
        int neighbor_cache_base_index = 0;
        int cell_x = i % env->grid_cols;  // Convert to 2D coordinates
        int cell_y = i / env->grid_cols;
        for(int j = 0; j<env->vision_range* env->vision_range; j++){
            int x = cell_x + env->neighbor_offsets[j*2];
            int y = cell_y + env->neighbor_offsets[j*2+1];
            int grid_index = env->grid_cols*y + x;
            if(x < 0 || x >= env->grid_cols || y < 0 || y >= env->grid_rows) continue;
            int grid_count = env->grid_cells[grid_index*SLOTS_PER_CELL];
            for(int k = 0; k < grid_count; k++){
                int entity_idx = env->grid_cells[grid_index*SLOTS_PER_CELL + 1 + k*2];
                int geometry_idx = env->grid_cells[grid_index*SLOTS_PER_CELL + 2 + k*2];
                int base_index = env->neighbor_cache_indices[i];
                env->neighbor_cache_entities[base_index + neighbor_cache_base_index] = entity_idx;
                env->neighbor_cache_entities[base_index + neighbor_cache_base_index + 1] = geometry_idx;
                neighbor_cache_base_index+=2;
            }
        }
    }
}

int get_neighbor_cache_entities(GPUDrive* env, int cell_idx, int* entities, int max_entities) {
    if (cell_idx < 0 || cell_idx >= (env->grid_cols * env->grid_rows)) {
        return 0; // Invalid cell index
    }
    int base_index = env->neighbor_cache_indices[cell_idx];
    int end_index = env->neighbor_cache_indices[cell_idx + 1];
    int count = end_index - base_index;
    int pairs = count / 2;  // Entity ID and geometry ID pairs
    // Limit to available space
    if (pairs > max_entities) {
        pairs = max_entities;
        count = pairs * 2;
    }
    memcpy(entities, env->neighbor_cache_entities + base_index, count * sizeof(int));
    return pairs;
}


void init(GPUDrive* env){
    env->human_agent_idx = 0;
    env->timestep = 0;
    env->entities = load_map_binary("map.bin", env);
    env->dynamics_model = CLASSIC;
    set_active_agents(env);
    set_start_position(env);
    env->logs = (Log*)calloc(env->active_agent_count, sizeof(Log));
    env->goal_reached = (char*)calloc(env->active_agent_count, sizeof(char));
    init_grid_map(env);
    env->vision_range = 21;
    init_neighbor_offsets(env);
    env->neighbor_cache_indices = (int*)calloc((env->grid_cols * env->grid_rows) + 1, sizeof(int));
    cache_neighbor_offsets(env);
    
}

void free_initialized(GPUDrive* env){
    for(int i = 0; i < env->num_entities; i++){
        free_entity(&env->entities[i]);
    }
    free(env->entities);
    free(env->active_agent_indices);
    free(env->logs);
    free(env->fake_data);
    free(env->goal_reached);
    free(env->map_corners);
    free(env->grid_cells);  // Free the grid cells memory
    free(env->neighbor_offsets);
    free(env->neighbor_cache_entities);
    free(env->neighbor_cache_indices);
}

void allocate(GPUDrive* env){
    init(env);
    int max_obs = 6 + 7 * (env->num_cars - 1) + 200 * 5;
    printf("max obs: %d\n", max_obs*env->active_agent_count);
    printf("num cars: %d\n", env->num_cars);
    printf("active agent count: %d\n", env->active_agent_count);
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

float normalize_heading(float heading) {
    float wrapped = fmodf(heading + M_PI, 2.0f * M_PI); // Shift to [0, 2π) then wrap
    if (wrapped < 0) wrapped += 2.0f * M_PI;            // Handle negative remainder
    return wrapped - M_PI;                              // Shift back to [-π, π]
}

void move_dynamics(GPUDrive* env, int action_idx, int agent_idx){
    if(env->dynamics_model == CLASSIC){
        // clip acceleration & steering
        Entity* agent = &env->entities[agent_idx];
        // Extract action components directly from the multi-discrete action array
        int (*action_array)[2] = (int(*)[2])env->actions;
        int acceleration_index = action_array[action_idx][0];
        int steering_index = action_array[action_idx][1];
        float acceleration = ACCELERATION_VALUES[acceleration_index];
        float steering = STEERING_VALUES[steering_index];

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
        // Update speed with acceleration
        speed += acceleration * dt;
        if (speed < 0) speed = 0;  // Prevent going backward
        // compute yaw rate
        float omega = (speed * tanf(steering)) / agent->length;
        heading = heading + omega * dt;
        // Normalize heading to range [-π, π]
        heading = normalize_heading(heading);

        // Compute new velocity components
        vx = speed * cosf(heading);
        vy = speed * sinf(heading);

        // Update position
        x += vx * dt;
        y += vy * dt;

        // Apply updates to the agent's state
        agent->x = x;
        agent->y = y;
        agent->heading = heading;
        agent->vx = vx;
        agent->vy = vy;
    }
    else if(env->dynamics_model == INVERTIBLE_BICYLE){
        // Invertible bicycle dynamics model
    }
    return;
}

void move_expert(GPUDrive* env, int* actions, int agent_idx){
    Entity* agent = &env->entities[agent_idx];
    agent->x = agent->traj_x[env->timestep];
    agent->y = agent->traj_y[env->timestep];
    agent->z = agent->traj_z[env->timestep];
    agent->heading = agent->traj_heading[env->timestep];
    //printf("x,y,z: %f %f %f", agent->x, agent->y, agent->z);
}

bool check_line_intersection(float p1[2], float p2[2], float q1[2], float q2[2]) {
    if (fmax(p1[0], p2[0]) < fmin(q1[0], q2[0]) || fmin(p1[0], p2[0]) > fmax(q1[0], q2[0]) ||
    fmax(p1[1], p2[1]) < fmin(q1[1], q2[1]) || fmin(p1[1], p2[1]) > fmax(q1[1], q2[1]))
    return false;
    float s1_x = p2[0] - p1[0];     
    float s1_y = p2[1] - p1[1];
    float s2_x = q2[0] - q1[0];     
    float s2_y = q2[1] - q1[1];

    float s, t;
    float denom = s1_x * s2_y - s2_x * s1_y;

    if (denom == 0)
        return false; // Collinear

    bool denom_positive = denom > 0;

    float s2_s1_x = p1[0] - q1[0];
    float s2_s1_y = p1[1] - q1[1];
    s = (s1_x * s2_s1_y - s1_y * s2_s1_x);
    if ((s < 0) == denom_positive)
        return false; // No intersection

    t = (s2_x * s2_s1_y - s2_y * s2_s1_x);
    if ((t < 0) == denom_positive)
        return false; // No intersection

    if ((s > denom) == denom_positive || (t > denom) == denom_positive)
        return false; // No intersection

    return true;
}

float point_to_line_distance(float point[2], float line_start[2], float line_end[2]) {
    float x0 = point[0], y0 = point[1];
    float x1 = line_start[0], y1 = line_start[1];
    float x2 = line_end[0], y2 = line_end[1];

    float dx = x2 - x1;
    float dy = y2 - y1;
    float denom = dx * dx + dy * dy;

    if (denom == 0) { // Line segment is a point
        return sqrtf((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
    }

    // Parametric value t for closest point on line
    float t = ((x0 - x1) * dx + (y0 - y1) * dy) / denom;
    t = fmaxf(0.0f, fminf(1.0f, t)); // Clamp t to [0, 1] for line segment

    // Closest point on the line segment
    float closest_x = x1 + t * dx;
    float closest_y = y1 + t * dy;

    // Distance from point to closest point
    return sqrtf((x0 - closest_x) * (x0 - closest_x) + (y0 - closest_y) * (y0 - closest_y));
}

int checkNeighbors(GPUDrive* env, float x, float y, int* entity_list, int max_size, const int (*local_offsets)[2], int offset_size) {
    // Get the grid index for the given position (x, y)
    int index = getGridIndex(env, x, y);
    if (index == -1) return 0;  // Return 0 size if position invalid

    // Calculate 2D grid coordinates
    int cellsX = env->grid_cols;
    int gridX = index % cellsX;
    int gridY = index / cellsX;
    

    int entity_list_count = 0;
    // Fill the provided array
    for (int i = 0; i < offset_size; i++) {
        int nx = gridX + local_offsets[i][0];
        int ny = gridY + local_offsets[i][1];
        // Ensure the neighbor is within grid bounds
        if (nx >= 0 && nx < env->grid_cols && ny >= 0 && ny < env->grid_rows) {
            int neighborIndex = (ny * env->grid_cols + nx) * SLOTS_PER_CELL;
            int count = env->grid_cells[neighborIndex];
            
            // Add entities from this cell to the list
            for (int j = 0; j < count && entity_list_count < max_size; j++) {
                int entityId = env->grid_cells[neighborIndex + 1 + j*2];
                int geometry_idx = env->grid_cells[neighborIndex + 2 + j*2];
                entity_list[entity_list_count] = entityId;
                entity_list[entity_list_count + 1] = geometry_idx;
                entity_list_count += 2;
            }
        }
    }

    return entity_list_count;
}


void collision_check(GPUDrive* env, int agent_idx) {
    Entity* agent = &env->entities[agent_idx];
    float half_length = agent->length / 2.0f;
    float half_width = agent->width / 2.0f;
    float cos_heading = cosf(agent->heading);
    float sin_heading = sinf(agent->heading);
    float corners[4][2];
    for (int i = 0; i < 4; i++) {
        corners[i][0] = agent->x + (offsets[i][0] * half_length * cos_heading - offsets[i][1] * half_width * sin_heading);
        corners[i][1] = agent->y + (offsets[i][0] * half_length * sin_heading + offsets[i][1] * half_width * cos_heading);
    }
    int collided = 0;
    int car_collided_with_index = -1;
    int entity_list[MAX_ENTITIES_PER_CELL*2 * 9];  // Array big enough for all neighboring cells
    int list_size = checkNeighbors(env, agent->x, agent->y, entity_list, MAX_ENTITIES_PER_CELL*2 * 9, collision_offsets, 9);
    // printf("agent: %d, list_size: %d\n", agent_idx, list_size);
    for (int i = 0; i < list_size ; i+=2) {
        if(entity_list[i] == -1) continue;
        if(entity_list[i] == agent_idx) continue;
        Entity* entity;
        entity = &env->entities[entity_list[i]];
        int geometry_idx = entity_list[i + 1];
        float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
        float end[2] = {entity->traj_x[geometry_idx + 1], entity->traj_y[geometry_idx + 1]};
        for (int k = 0; k < 4; k++) { // Check each edge of the bounding box
            int next = (k + 1) % 4;
            if (check_line_intersection(corners[k], corners[next], start, end)) {
                collided = OFFROAD;
                break;
            }
        }
        if (collided == OFFROAD) break;
    }
    for(int i = 0; i < env->num_entities; i++){
        if(i == agent_idx) continue;
        Entity* entity = &env->entities[i];
        if (entity->type > 3) break;
        if (entity->type != VEHICLE) continue;
        float x1 = entity->x;
        float y1 = entity->y;
        float dist = sqrtf((x1 - agent->x) * (x1 - agent->x) + (y1 - agent->y) * (y1 - agent->y));
        if(dist > 5.0f) continue;
        float other_corners[4][2];
        for (int z = 0; z < 4; z++) {
            float other_cos_heading = cosf(entity->traj_heading[0]);
            float other_sin_heading = sinf(entity->traj_heading[0]);
            float other_half_length = entity->length / 2.0f;
            float other_half_width = entity->width / 2.0f;
            other_corners[z][0] = entity->x + (offsets[z][0] * other_half_length * other_cos_heading - offsets[z][1] * other_half_width * other_sin_heading);
            other_corners[z][1] = entity->y + (offsets[z][0] * other_half_length * other_sin_heading + offsets[z][1] * other_half_width * other_cos_heading);
        }
        for (int k = 0; k < 4; k++) { // Check each edge of the bounding box
            int next = (k + 1) % 4;
            for (int l = 0; l < 4; l++) { // Check each edge of the bounding box
                int next_l = (l + 1) % 4;
                if (check_line_intersection(corners[k], corners[next], other_corners[l], other_corners[next_l])) {
                    collided = VEHICLE_COLLISION;
                    car_collided_with_index = i;
                    break;
                }
            }
            if (collided == VEHICLE_COLLISION) break;
        }
        if (collided == VEHICLE_COLLISION) break;
    }
    agent->collision_state = collided;
    if (car_collided_with_index != -1) {
        env->entities[car_collided_with_index].collision_state = VEHICLE_COLLISION;
    }
}

float normalize_value(float value, float min, float max){
    return (value - min) / (max - min);
}

float reverse_normalize_value(float value, float min, float max){
    return value * (max - min) + min;
}

void compute_observations(GPUDrive* env) {
    int max_obs = 6 + 7 * (env->num_cars - 1) + 200 * 5;
    memset(env->observations, 0, max_obs * env->active_agent_count * sizeof(float));
    float (*observations)[max_obs] = (float(*)[max_obs])env->observations; 
    for(int i = 0; i < env->active_agent_count; i++) {
        float* obs = &observations[i][0];
        Entity* ego_entity = &env->entities[env->active_agent_indices[i]];
        if(ego_entity->type > 3) break;
        float cos_heading = cosf(ego_entity->heading);
        float sin_heading = sinf(ego_entity->heading);
        float ego_speed = sqrtf(ego_entity->vx * ego_entity->vx + ego_entity->vy * ego_entity->vy);
        // Set goal distances
        float goal_x = relative_distance(ego_entity->x, ego_entity->goal_position_x);
        float goal_y = relative_distance(ego_entity->y, ego_entity->goal_position_y);
        obs[0] = normalize_value(goal_x, MIN_REL_GOAL_COORD, MAX_REL_GOAL_COORD);
        obs[1] = normalize_value(goal_y, MIN_REL_GOAL_COORD, MAX_REL_GOAL_COORD);
        obs[2] = ego_speed / MAX_SPEED;
        obs[3] = ego_entity->width / MAX_VEH_WIDTH;
        obs[4] = ego_entity->length / MAX_VEH_LEN;
        obs[5] = ego_entity->collision_state;
        
        // Relative Pos of other cars
        int obs_idx = 6;  // Start after goal distances
        int cars_seen = 0;
        for(int j = 0; j < env->num_entities; j++) {
            if(env->entities[j].type > 3) break;
            if(j == i) continue;  // Skip self, but don't increment obs_idx
            
            Entity* other_entity = &env->entities[j];
            
            // Store original relative positions
            float dx = other_entity->x - ego_entity->x;
            float dy = other_entity->y - ego_entity->y;
            float dist = sqrtf(dx*dx + dy*dy);
            if(dist > 50.0f) continue;
            // Rotate to ego vehicle's frame
            float rel_x = dx * cos_heading + dy * sin_heading;
            float rel_y = -dx * sin_heading + dy * cos_heading;
            // Store observations with correct indexing
            obs[obs_idx] = normalize_value(rel_x, MIN_REL_AGENT_POS, MAX_REL_AGENT_POS);
            obs[obs_idx + 1] = normalize_value(rel_y, MIN_REL_AGENT_POS, MAX_REL_AGENT_POS);
            obs[obs_idx + 2] = other_entity->width / MAX_VEH_WIDTH;
            obs[obs_idx + 3] = other_entity->length / MAX_VEH_LEN;
            
            // relative heading
            float rel_heading = normalize_heading(other_entity->heading - ego_entity->heading);
            obs[obs_idx + 4] = cosf(rel_heading) / MAX_ORIENTATION_RAD;
            obs[obs_idx + 5] = sinf(rel_heading) / MAX_ORIENTATION_RAD;
            
            // relative speed
            float other_speed = sqrtf(other_entity->vx * other_entity->vx + other_entity->vy * other_entity->vy);
            obs[obs_idx + 6] = other_speed / MAX_SPEED;
            cars_seen++;
            obs_idx += 7;  // Move to next observation slot
        }
        for(int j = cars_seen; j < env->num_cars - 1; j++){
            obs[obs_idx] = -1.0f;
            obs[obs_idx + 1] = -1.0f;
            obs[obs_idx + 2] = -1.0f;
            obs[obs_idx + 3] = -1.0f;
            obs[obs_idx + 4] = -1.0f;
            obs[obs_idx + 5] = -1.0f;
            obs[obs_idx + 6] = 1.0f;
            obs_idx += 7;
	}

        // map observations
        int entity_list[MAX_ROAD_SEGMENT_OBSERVATIONS*2];  // Array big enough for all neighboring cells
        int grid_idx = getGridIndex(env, ego_entity->x, ego_entity->y);
        // if(env->human_agent_idx == i){
        //     printf("Grid index: %d\n", grid_idx);
        // }
        int list_size = get_neighbor_cache_entities(env, grid_idx, entity_list, MAX_ROAD_SEGMENT_OBSERVATIONS);
        for(int k = 0; k < list_size; k++){
            int entity_idx = entity_list[k*2];
            int geometry_idx = entity_list[k*2+1];
            Entity* entity = &env->entities[entity_idx];
            float start[2] = {entity->traj_x[geometry_idx], entity->traj_y[geometry_idx]};
            float end[2] = {entity->traj_x[geometry_idx+1], entity->traj_y[geometry_idx+1]};
            float rel_x_start = start[0] - ego_entity->x;
            float rel_y_start = start[1] - ego_entity->y;
            float rel_x_end = end[0] - ego_entity->x;
            float rel_y_end = end[1] - ego_entity->y;
            float x_start = rel_x_start * cos_heading + rel_y_start * sin_heading;
            float y_start = -rel_x_start * sin_heading + rel_y_start * cos_heading;
            float x_end = rel_x_end * cos_heading + rel_y_end * sin_heading;
            float y_end = -rel_x_end * sin_heading + rel_y_end * cos_heading;
            // if(env->human_agent_idx == i){
            //     printf("K: %d, Entity index: %d, Geometry index: %d\n", k, entity_list[k*2], entity_list[k*2+1]);
            //     printf("start: %f, %f, end: %f, %f\n", start[0], start[1], end[0], end[1]);
            // }
            obs[obs_idx] = normalize_value(x_start, MIN_RG_COORD, MAX_RG_COORD);
            obs[obs_idx + 1] = normalize_value(y_start, MIN_RG_COORD, MAX_RG_COORD);
            obs[obs_idx + 2] = normalize_value(x_end, MIN_RG_COORD, MAX_RG_COORD);
            obs[obs_idx + 3] = normalize_value(y_end, MIN_RG_COORD, MAX_RG_COORD);
            obs[obs_idx + 4] = entity->type;
            obs_idx += 5;
        }

        for(int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS - list_size; k++){
            obs[obs_idx] = -1.0f;
            obs[obs_idx + 1] = -1.0f;
            obs[obs_idx + 2] = -1.0f;
            obs[obs_idx + 3] = -1.0f;
            obs[obs_idx + 4] = 6.0f;
            obs_idx += 5;
        }
	
    }
}

void c_reset(GPUDrive* env){
    env->timestep = 0;
    set_start_position(env);
    for(int x = 0;x<env->active_agent_count; x++){
        env->logs[x] = (Log){0};
        int agent_idx = env->active_agent_indices[x];
        collision_check(env, agent_idx);
    }
    memset(env->goal_reached, 0, env->active_agent_count*sizeof(char));
    compute_observations(env);
}

void c_step(GPUDrive* env){
    memset(env->rewards, 0, env->active_agent_count * sizeof(float));
    env->timestep++;
    if(env->timestep == 91){
	    for(int i = 0; i < env->active_agent_count; i++){
            if(env->goal_reached[i] == 0){
                env->logs[i].score = 0.0f;
            } 
	    else {
                env->logs[i].score = 1.0f;
                // env->logs[i].episode_return +=1.0f;
            }
            add_log(env->log_buffer, &env->logs[i]);
	    }
	    c_reset(env);
    }// Process actions for all active agents
    for(int i = 0; i < env->active_agent_count; i++){
        env->logs[i].score = 0.0f;
	    env->logs[i].episode_length += 1;
        int agent_idx = env->active_agent_indices[i];
        env->entities[agent_idx].collision_state = 0;
        move_dynamics(env, i, agent_idx);
        // move_random(env, agent_idx);
        // move_expert(env, env->actions, agent_idx);
        collision_check(env, agent_idx);
        if(env->entities[agent_idx].collision_state > 0){
            if(env->entities[agent_idx].collision_state == VEHICLE_COLLISION){
                env->rewards[i] = env->reward_vehicle_collision;
                env->logs[i].collision_rate = 1.0f;
                env->logs[i].episode_return += env->reward_vehicle_collision;
            }
            else if(env->entities[agent_idx].collision_state == OFFROAD){
                env->rewards[i] = env->reward_offroad_collision;
                env->logs[i].offroad_rate = 1.0f;
                env->logs[i].episode_return += env->reward_offroad_collision;
            }
        }

        float distance_to_goal = relative_distance_2d(
                env->entities[agent_idx].x,
                env->entities[agent_idx].y,
                env->entities[agent_idx].goal_position_x,
                env->entities[agent_idx].goal_position_y);
        int reached_goal = distance_to_goal < 2.0f;
        if(reached_goal && env->goal_reached[i] == 0){            
            env->rewards[i] += 1.0f;
	        env->goal_reached[i] = 1;
	        env->logs[i].episode_return += 1.0f;
            continue;
	    }
    }
    
    compute_observations(env);
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
        target_pos.z + 150.0f    // 20 units behind target
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
    DrawLine3D((Vector3){env->map_corners[0], env->map_corners[1], 0}, (Vector3){env->map_corners[2], env->map_corners[1], 0}, BLACK);
    DrawLine3D((Vector3){env->map_corners[0], env->map_corners[1], 0}, (Vector3){env->map_corners[0], env->map_corners[3], 0}, BLACK);
    DrawLine3D((Vector3){env->map_corners[2], env->map_corners[1], 0}, (Vector3){env->map_corners[2], env->map_corners[3], 0}, BLACK);
    DrawLine3D((Vector3){env->map_corners[0], env->map_corners[3], 0}, (Vector3){env->map_corners[2], env->map_corners[3], 0}, BLACK);

    for(int i = 0; i < env->num_entities; i++) {
        if(env->entities[i].type == 1) {  // If entity is a vehicle
            // Check if this vehicle is an active agent
            bool is_active_agent = false;
            int agent_index = -1;
            
            for(int j = 0; j < env->active_agent_count; j++) {
                if(env->active_agent_indices[j] == i) {
                    is_active_agent = true;
                    agent_index = j;
                    break;
                }
            }
            Vector3 position;
            float heading;
             position = (Vector3){
                    env->entities[i].x,
                    env->entities[i].y,
                    1
                };      
            heading = env->entities[i].heading;
            
            
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
            
            // Determine color based on active status and other conditions
            Color object_color = GRAY;  // Default color for non-active vehicles
            
            if(is_active_agent) {
                object_color = DARKBLUE;  // Active agents are blue
                
                if(agent_index == env->human_agent_idx) {
                    object_color = PURPLE;  // Human-controlled agent
                }
                
                if(env->entities[i].collision_state == 1 || env->entities[i].collision_state == 2) {
                    object_color = RED;  // Collided agent
                }
                
                // Only draw goal position for active agents
                if(env->entities[i].valid == 1 && env->goal_reached[agent_index] == 0) {
                    DrawCube((Vector3){0, 0, 0}, size.x, size.y, size.z, object_color);
                    DrawCubeWires((Vector3){0, 0, 0}, size.x, size.y, size.z, BLACK);
                    if( agent_index == env->human_agent_idx){
                        int max_obs = 6 + 7 * (env->num_cars - 1) + 200 * 5;
                        float (*observations)[max_obs] = (float(*)[max_obs])env->observations;
                        float* agent_obs = &observations[agent_index][0];
                        // First draw other agent observations
                        int obs_idx = 6;  // Start after goal distances
                        for(int j = 0; j < env->num_cars - 1; j++) {  // -1 because we skip self
                            if(agent_obs[obs_idx] != -1 && agent_obs[obs_idx + 1] != -1) {
                                // Draw position of other agents
                                float x = reverse_normalize_value(agent_obs[obs_idx], MIN_RG_COORD, MAX_RG_COORD);
                                float y = reverse_normalize_value(agent_obs[obs_idx + 1], MIN_RG_COORD, MAX_RG_COORD);
                                DrawLine3D((Vector3){0, 0, 0}, 
                                        (Vector3){x, 
                                                y, 1}, 
                                        ORANGE);
                            }
                            obs_idx += 7;  // Move to next agent observation (7 values per agent)
                        }

                        // Then draw map observations
                        int map_start_idx = 6 + 7 * (env->num_cars - 1);  // Start after agent observations
                        for(int k = 0; k < MAX_ROAD_SEGMENT_OBSERVATIONS; k++) {  // Loop through potential map entities
                            int entity_idx = map_start_idx + k * 5;
                            if(agent_obs[entity_idx] != -1 && agent_obs[entity_idx + 1] != -1) {
                                Color lineColor = BLUE;  // Default color
                                int entity_type = (int)agent_obs[entity_idx + 4];
                                // Choose color based on entity type
                                if(entity_type == ROAD_EDGE) {
                                    lineColor = BLACK;
                                    // For road segments, draw line between start and end points
                                    if(agent_obs[entity_idx + 2] != -1 && agent_obs[entity_idx + 3] != -1) {
                                        float x_start = reverse_normalize_value(agent_obs[entity_idx], MIN_RG_COORD, MAX_RG_COORD);
                                        float y_start = reverse_normalize_value(agent_obs[entity_idx + 1], MIN_RG_COORD, MAX_RG_COORD);
                                        float x_end = reverse_normalize_value(agent_obs[entity_idx + 2], MIN_RG_COORD, MAX_RG_COORD);
                                        float y_end = reverse_normalize_value(agent_obs[entity_idx + 3], MIN_RG_COORD, MAX_RG_COORD);
                                        DrawLine3D((Vector3){0,0,0}, (Vector3){x_start, y_start, 1}, lineColor);    
                                        DrawLine3D((Vector3){0,0,0}, (Vector3){x_end, y_end, 1}, lineColor);
                                    }
                                }
                            }

                        }
                    }
                    
                }
            } else {
                // Draw non-active vehicles
                DrawCube((Vector3){0, 0, 0}, size.x, size.y, size.z, object_color);
                DrawCubeWires((Vector3){0, 0, 0}, size.x, size.y, size.z, BLACK);
            }
            
            // Restore previous transform
            rlPopMatrix();
            
            // Draw goal position for active agents
            if(is_active_agent && env->entities[i].valid == 1) {
                DrawSphere((Vector3){
                    env->entities[i].goal_position_x,
                    env->entities[i].goal_position_y,
                    1
                }, 0.5f, DARKGREEN);
            }
        }
        
        // Draw road elements
        if(env->entities[i].type > 3 && env->entities[i].type < 7) {
            for(int j = 0; j < env->entities[i].array_size - 1; j++) {
                Vector3 start = {
                    env->entities[i].traj_x[j],
                    env->entities[i].traj_y[j],
                    1
                };
                Vector3 end = {
                    env->entities[i].traj_x[j + 1],
                    env->entities[i].traj_y[j + 1],
                    1
                };
                
                Color lineColor = GRAY;
                if (env->entities[i].type == ROAD_LANE) lineColor = GRAY;
                else if (env->entities[i].type == ROAD_LINE) lineColor = BLUE;
                else if (env->entities[i].type == ROAD_EDGE) lineColor = BLACK;
                else if (env->entities[i].type == DRIVEWAY) lineColor = RED;
                
                if(env->entities[i].type == ROAD_EDGE){
                    DrawLine3D(start, end, lineColor);
                }
                if(env->entities[i].type == ROAD_EDGE){
                    DrawSphere(start, 0.5f, lineColor);
                    DrawSphere(end, 0.5f, lineColor);
                }
            }
        }
    }
    
    // Draw grid cells using the stored bounds
    float grid_start_x = env->map_corners[0];
    float grid_start_y = env->map_corners[1];
    for(int i = 0; i < env->grid_cols; i++) {
        for(int j = 0; j < env->grid_rows; j++) {
            float x = grid_start_x + i * GRID_CELL_SIZE;
            float y = grid_start_y + j * GRID_CELL_SIZE;
            // int index = i * env->grid_rows + j;
            DrawCubeWires(
                (Vector3){x + GRID_CELL_SIZE/2, y + GRID_CELL_SIZE/2, 1}, 
                GRID_CELL_SIZE, GRID_CELL_SIZE, 0.1f, GRAY);
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
    // acceleration & steering
    DrawText(TextFormat("Acceleration: %d", env->actions[env->human_agent_idx * 2]), 10, 110, 20, BLACK);
    DrawText(TextFormat("Steering: %d", env->actions[env->human_agent_idx * 2 + 1]), 10, 130, 20, BLACK);
    DrawText(TextFormat("Grid Rows: %d", env->grid_rows), 10, 150, 20, BLACK);
    DrawText(TextFormat("Grid Cols: %d", env->grid_cols), 10, 170, 20, BLACK);
    EndDrawing();
}

void close_client(Client* client){
    CloseWindow();
    free(client);
}
