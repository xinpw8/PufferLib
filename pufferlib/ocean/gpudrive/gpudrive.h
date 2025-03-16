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

#define LOG_BUFFER_SIZE 1024

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float score;
    float collision_count;
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
        log.episode_return += logs->logs[i].episode_return;
        log.episode_length += logs->logs[i].episode_length;
	    log.score += logs->logs[i].score;
	    log.collision_count += logs->logs[i].collision_count;
	//printf("length: %f", log.episode_length);
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.score /= logs->idx;
    log.collision_count /= logs->idx;
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
    float nearest_line_dist;
    float* nearest_line_start;
    float* nearest_line_end;
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
    free(entity->nearest_line_start);
    free(entity->nearest_line_end);
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
    int timestep;
    int dynamics_model;
    float* fake_data;
    char* goal_reached;
};

Entity* load_map_binary(const char* filename, GPUDrive* env) {
    FILE* file = fopen(filename, "rb");
    if (!file) return NULL;
    
    fread(&env->num_entities, sizeof(int), 1, file);
    Entity* entities = (Entity*)malloc(env->num_entities * sizeof(Entity));
    //printf("Num entities: %d\n", env->num_entities);
    for (int i = 0; i < env->num_entities; i++) {
        entities[i].nearest_line_start = (float*)calloc(2, sizeof(float));
	entities[i].nearest_line_end = (float*)calloc(2, sizeof(float));
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

void set_start_position(GPUDrive* env){
    for(int i = 0; i < env->active_agent_count; i++){
        env->entities[env->active_agent_indices[i]].x = env->entities[env->active_agent_indices[i]].traj_x[0];
        env->entities[env->active_agent_indices[i]].y = env->entities[env->active_agent_indices[i]].traj_y[0];
        env->entities[env->active_agent_indices[i]].z = env->entities[env->active_agent_indices[i]].traj_z[0];
        env->entities[env->active_agent_indices[i]].vx = env->entities[env->active_agent_indices[i]].traj_vx[0];
        env->entities[env->active_agent_indices[i]].vy = env->entities[env->active_agent_indices[i]].traj_vy[0];
        env->entities[env->active_agent_indices[i]].vz = env->entities[env->active_agent_indices[i]].traj_vz[0];
        env->entities[env->active_agent_indices[i]].heading = env->entities[env->active_agent_indices[i]].traj_heading[0];
        // env->entities[env->active_agent_indices[i]].heading = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random float between -1 and 1
        env->entities[env->active_agent_indices[i]].valid = env->entities[env->active_agent_indices[i]].traj_valid[0];
        // printf("agent %d\n", env->active_agent_indices[i]);
        // printf("x , y: %f, %f\n", env->entities[env->active_agent_indices[i]].x, env->entities[env->active_agent_indices[i]].y);
        // printf("goal_x, goal_y: %f, %f\n", env->entities[env->active_agent_indices[i]].goal_position_x, env->entities[env->active_agent_indices[i]].goal_position_y);
    }
}

void set_active_agents(GPUDrive* env){
    env->active_agent_count = 0;
    int active_agent_indices[env->num_entities];
    for(int i = 0; i < env->num_entities; i++){
        if(env->entities[i].type == 1){
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
    for(int i = 0; i < env->active_agent_count; i++){
        int valid_count = 0;
        for(int j=  0; j<91; j++){
            if(env->entities[env->active_agent_indices[i]].traj_valid[j] == 1){
                valid_count++;
            }
        }
        //printf("agent %d valid_count: %d\n", env->active_agent_indices[i], valid_count);
    }
}

void init(GPUDrive* env){
    env->human_agent_idx = 0;
    env->timestep = 0;
    env->entities = load_map_binary("map.bin", env);
    env->dynamics_model = CLASSIC;
    set_active_agents(env);
    set_start_position(env);
    env->logs = (Log*)calloc(env->active_agent_count, sizeof(Log));
    //printf("num_entities: %d\n", env->num_entities);
    //printf("Offset of x: %zu\n", offsetof(struct Entity, x));
    //printf("Offset of y: %zu\n", offsetof(struct Entity, y));
    printf("active_agent_count: %d\n", env->active_agent_count);
    env->fake_data = (float*)calloc(7, sizeof(float));
    for (int i = 0;i<7;i++ ){
	    env->fake_data[i] = (float)(rand() % 5) / 5.0f;
    }
    env->goal_reached = (char*)calloc(env->active_agent_count, sizeof(char));
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
}

void allocate(GPUDrive* env){
    init(env);
    int max_obs = 7;
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
        while (heading > M_PI) heading -= 2.0f * M_PI;
        while (heading < -M_PI) heading += 2.0f * M_PI;

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
    float min_dist = 10000;
    float nearest_start[2], nearest_end[2];

    for (int i = 0; i < env->num_entities; i++) {
        if(i == agent_idx) continue;
        Entity* entity = &env->entities[i];
        if(entity->type == ROAD_EDGE){
            for (int j = 0; j < entity->array_size - 1; j++) {
                float start[2] = {entity->traj_x[j], entity->traj_y[j]};
                float end[2] = {entity->traj_x[j + 1], entity->traj_y[j + 1]};
                for (int k = 0; k < 4; k++) { // Check each edge of the bounding box
                    int next = (k + 1) % 4;
                    if (check_line_intersection(corners[k], corners[next], start, end)) {
                        collided = 1;
                        break;
                    }
                }
                if (collided)break;
                // Distance check
                float agent_center[2] = {agent->x, agent->y};
                float dist = point_to_line_distance(agent_center, start, end);
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_start[0] = start[0] - agent->x;
                    nearest_start[1] = start[1] - agent->y;
                    nearest_end[0] = end[0] - agent->x;
                    nearest_end[1] = end[1] - agent->y;
                }
            }
            if (collided) break;
        }
        if(entity->type == VEHICLE){
            float other_corners[4][2];
            for (int i = 0; i < 4; i++) {
                other_corners[i][0] = entity->x + (offsets[i][0] * half_length * cos_heading - offsets[i][1] * half_width * sin_heading);
                other_corners[i][1] = entity->y + (offsets[i][0] * half_length * sin_heading + offsets[i][1] * half_width * cos_heading);
            }
            for (int k = 0; k < 4; k++) { // Check each edge of the bounding box
                int next = (k + 1) % 4;
                for (int l = 0; l < 4; l++) { // Check each edge of the bounding box
                    int next = (l + 1) % 4;
                    if (check_line_intersection(corners[k], corners[next], other_corners[l], other_corners[next])) {
                        collided = 1;
                        break;
                    }
                }
            }
        }
        
    }
    agent->collision_state = collided;
    if (min_dist != 10000) { // If a road edge was found
        agent->nearest_line_dist = min_dist;
        agent->nearest_line_start[0] = nearest_start[0];
        agent->nearest_line_start[1] = nearest_start[1];
        agent->nearest_line_end[0] = nearest_end[0];
        agent->nearest_line_end[1] = nearest_end[1];
    } else {
        agent->nearest_line_dist = -1.0f; // Indicate no line found
    }
}

void compute_observations(GPUDrive* env){
    int max_obs = 7;
    float (*observations)[max_obs] = (float(*)[max_obs])env->observations;
    for(int i = 0; i < env->active_agent_count; i++){
        float* obs = &observations[i][0];
        obs[0] = relative_distance(
            env->entities[env->active_agent_indices[i]].x,
            env->entities[env->active_agent_indices[i]].goal_position_x);
        obs[1] = relative_distance(
            env->entities[env->active_agent_indices[i]].y,
            env->entities[env->active_agent_indices[i]].goal_position_y);
        obs[2] = env->entities[env->active_agent_indices[i]].nearest_line_dist;
        obs[3] = env->entities[env->active_agent_indices[i]].nearest_line_start[0];
        obs[4] = env->entities[env->active_agent_indices[i]].nearest_line_start[1];
        obs[5] = env->entities[env->active_agent_indices[i]].nearest_line_end[0];
        obs[6] = env->entities[env->active_agent_indices[i]].nearest_line_end[1];
    }
};

void c_reset(GPUDrive* env){
    env->timestep = 0;
    set_start_position(env);
    for(int x = 0;x<env->active_agent_count; x++){
        env->logs[x] = (Log){0};
    }
    memset(env->goal_reached, 0, env->active_agent_count*sizeof(char));
    compute_observations(env);
}

void c_step(GPUDrive* env){
    int (*action_array)[2] = (int(*)[2])env->actions;
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
        if(env->entities[agent_idx].collision_state == 1){
            env->rewards[i] = -0.1f;
	    env->logs[i].episode_return -=0.1f;
	    env->logs[i].collision_count += 1.0f;

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
            if(is_active_agent){
                position = (Vector3){
                    env->entities[i].x,
                    env->entities[i].y,
                    1
                };      
                heading = env->entities[i].heading;
      
            } else {
                position = (Vector3){
                    env->entities[i].traj_x[0],
                    env->entities[i].traj_y[0],
                    1
                };
                heading = env->entities[i].traj_heading[0];
            }
            
            
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
                
                if(env->entities[i].collision_state == 1) {
                    object_color = RED;  // Collided agent
                }
                
                // Only draw goal position for active agents
                if(env->entities[i].valid == 1 && env->goal_reached[agent_index] == 0) {
                    DrawCube((Vector3){0, 0, 0}, size.x, size.y, size.z, object_color);
                    DrawCubeWires((Vector3){0, 0, 0}, size.x, size.y, size.z, BLACK);
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
                
                if(env->entities[i].type == ROAD_LANE || env->entities[i].type == ROAD_EDGE){
                    DrawLine3D(start, end, lineColor);
                }

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
    // acceleration & steering
    DrawText(TextFormat("Acceleration: %d", env->actions[env->human_agent_idx * 2]), 10, 110, 20, BLACK);
    DrawText(TextFormat("Steering: %d", env->actions[env->human_agent_idx * 2 + 1]), 10, 130, 20, BLACK);
    EndDrawing();
}

void close_client(Client* client){
    CloseWindow();
    free(client);
}

