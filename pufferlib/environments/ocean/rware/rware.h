#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "raylib.h"

#define NOOP 0
#define FORWARD 1
#define LEFT 2
#define RIGHT 3
#define TOGGLE_LOAD 4
#define TOGGLE_AGENT 5
#define TICK_RATE 1.0f/60.0f
#define NUM_DIRECTIONS 4
static const int DIRECTIONS[NUM_DIRECTIONS] = {0, 1, 2, 3};
static const int DIRECTION_VECTORS[NUM_DIRECTIONS][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
static const int SURROUNDING_VECTORS[8][2] = {{0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0}, {-1,-1}};
static const int tiny_map[110] = {
    0,0,0,0,0,0,0,0,0,0,  
    0,1,1,0,0,0,0,1,1,0,  
    0,1,1,0,0,0,0,1,1,0, 
    0,1,1,0,0,0,0,1,1,0,  
    0,1,1,0,0,0,0,1,1,0,  
    0,1,1,0,0,0,0,1,1,0,  
    0,1,1,0,0,0,0,1,1,0,  
    0,1,1,0,0,0,0,1,1,0, 
    0,1,1,0,0,0,0,1,1,0,  
    0,0,0,0,0,0,0,0,0,0,  
    0,0,0,0,3,3,0,0,0,0   
};
static const int tiny_shelf_locations[32] = {
    11, 12, 17, 18,  // Row 1
    21, 22, 27, 28,  // Row 2
    31, 32, 37, 38,  // Row 3
    41, 42, 47, 48,  // Row 4
    51, 52, 57, 58,  // Row 5
    61, 62, 67, 68,  // Row 6
    71, 72, 77, 78,  // Row 7
    81, 82, 87, 88   // Row 8
};

static const int small_map[200] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    3,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,
    3,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
};

static const int small_shelf_locations[80] = {
    22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,  // Row 1
    42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,58,  // Row 2
    91,92,93,94,95,96,97,98,  // Row 4 (right side only)
    111,112,113,114,115,116,117,118,  // Row 5 (right side only)
    142,143,144,145,146,147,148,149,151,152,153,154,155,156,157,158,  // Row 7 (both sides)
    162,163,164,165,166,167,168,169,171,172,173,174,175,176,177,178   // Row 8 (both sides)
};

static const int medium_map[320] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    3,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,
    3,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
};

static const int medium_shelf_locations[144]={
    22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,  // Row 1
    42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,58,  // Row 2
    82,83,84,85,86,87,88,89,91,92,93,94,95,96,97,98,  // Row 4
    102,103,104,105,106,107,108,109,111,112,113,114,115,116,117,118,  // Row 5
    151,152,153,154,155,156,157,158,  // Row 7 (right side only)
    171,172,173,174,175,176,177,178,  // Row 8 (right side only)
    202,203,204,205,206,207,208,209,211,212,213,214,215,216,217,218,  // Row 10
    222,223,224,225,226,227,228,229,231,232,233,234,235,236,237,238,  // Row 11
    262,263,264,265,266,267,268,269,271,272,273,274,275,276,277,278,  // Row 13
    282,283,284,285,286,287,288,289,291,292,293,294,295,296,297,298   // Row 14
};

//  LD_LIBRARY_PATH=raylib/lib ./go
#define LOG_BUFFER_SIZE 1024

static inline int max(int a, int b) {
    return (a > b) ? a : b;
}

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    int games_played;
    float score;
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
        log.games_played += logs->logs[i].games_played;
        log.score += logs->logs[i].score;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.score /= logs->idx;
    logs->idx = 0;
    return log;
}

typedef struct MovementGraph MovementGraph;
struct MovementGraph {
    int* target_positions;
    int* cycle_ids;
    int* weights;
    bool* is_position_empty;
    int num_cycles;
};


typedef struct CRware CRware;
struct CRware {
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;
    Log log;
    float score;
    int width;
    int height;
    int map_choice;
    int* warehouse_states;
    int num_agents;
    int num_requested_shelves;
    int* agent_locations;
    int* agent_directions;
    int* agent_states;
    int shelves_delivered;
    int human_agent_idx;
    int grid_square_size;
    int* original_shelve_locations;
    MovementGraph* movement_graph;
};

MovementGraph* init_movement_graph(CRware* env) {
    MovementGraph* graph = (MovementGraph*)calloc(1, sizeof(MovementGraph));
    graph->target_positions = (int*)calloc(env->num_agents, sizeof(int));
    graph->cycle_ids = (int*)calloc(env->num_agents, sizeof(int));
    graph->weights = (int*)calloc(env->num_agents, sizeof(int));
    graph->num_cycles = 0;

    // Initialize arrays
    for (int i = 0; i < env->num_agents; i++) {
        graph->target_positions[i] = -1;   // No target
        graph->cycle_ids[i] = -1;          // Not in cycle
        graph->weights[i] = 0;             // No weight
    }
    return graph;
}

int find_agent_at_position(CRware* env, int position) {
    for (int i = 0; i < env->num_agents; i++) {
        if (env->agent_locations[i] == position) {
            return i;
        }
    }
    return -1;
}

void place_agent(CRware* env, int agent_idx) {
    int map_size = env->map_choice == 1 ? 110 : env->map_choice == 2 ? 200 : 320;
    
    while (1) {
        int random_pos = rand() % map_size;
        
        // Skip if position is not empty
        if (env->warehouse_states[random_pos] != 0) {
            continue;
        }

        // Skip if another agent is already here
        int agent_at_position = find_agent_at_position(env, random_pos);
        if (agent_at_position != -1) {
            continue;
        }

        // Position is valid, place the agent
        env->agent_locations[agent_idx] = random_pos;
        env->agent_directions[agent_idx] = rand() % 4;
        env->agent_states[agent_idx] = 0;
        break;
    }
}

int request_new_shelf(CRware* env) {
    int total_shelves = env->map_choice == 1 ? 32 : env->map_choice == 2 ? 80 : 144;
    int random_index = rand() % total_shelves;
    int shelf_location = env->map_choice == 1 ? tiny_shelf_locations[random_index] : env->map_choice == 2 ? small_shelf_locations[random_index] : medium_shelf_locations[random_index];
    if (env->warehouse_states[shelf_location] == 1) {
        env->warehouse_states[shelf_location] = 2;
        return 1;
    }
    return 0;
}

void generate_map(CRware* env,const int* map) {
    // seed new random
    srand(time(NULL));
    int map_size = env->map_choice == 1 ? 110 : env->map_choice == 2 ? 200 : 320;
    for (int i = 0; i < map_size; i++) {
        env->warehouse_states[i] = map[i];
    }

    int requested_shelves_count = 0;
    while (requested_shelves_count < env->num_requested_shelves) {
        requested_shelves_count += request_new_shelf(env);
    }
    // set agents in center
    // for (int i = 0; i < env->num_agents; i++) {
    //     place_agent(env, i);
    // }

    /* let crate a 4 agent cycle*/
    env->agent_locations[0] = 0;
    env->agent_directions[0] = 0;
    env->agent_states[0] = 0;
    env->agent_locations[1] = 1;
    env->agent_directions[1] = 1;
    env->agent_states[1] = 0;
    env->agent_locations[2] = 10;
    env->agent_directions[2] = 3;
    env->agent_states[2] = 0;
    env->agent_locations[3] = 11;
    env->agent_directions[3] = 2;
    env->agent_states[3] = 0;
}




void init(CRware* env) {
    int map_size = env->map_choice == 1 ? 110 : env->map_choice == 2 ? 200 : 320;
    env->warehouse_states = (int*)calloc(map_size, sizeof(int));
    env->agent_locations = (int*)calloc(env->num_agents, sizeof(int));
    env->agent_directions = (int*)calloc(env->num_agents, sizeof(int));
    env->agent_states = (int*)calloc(env->num_agents, sizeof(int));
    if (env->map_choice == 1) {
        generate_map(env, tiny_map);
    }
    else if (env->map_choice == 2) {
        generate_map(env, small_map);
    }
    else {
        generate_map(env, medium_map);
    }
}

void allocate(CRware* env) {
    init(env);
    env->movement_graph = init_movement_graph(env);
    env->observations = (float*)calloc(env->num_agents*(27), sizeof(float));
    env->actions = (int*)calloc(env->num_agents, sizeof(int));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->dones = (unsigned char*)calloc(env->num_agents, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);
}

void free_initialized(CRware* env) {
    free(env->warehouse_states);
    free(env->agent_locations);
    free(env->agent_directions);
    free(env->agent_states);
}

void free_allocated(CRware* env) {
    free(env->actions);
    free(env->observations);
    free(env->dones);
    free(env->rewards);
    free_logbuffer(env->log_buffer);
    free_initialized(env);
}

void compute_observations(CRware* env) {
    int surround_indices[8];
    int obs_idx =0;
    int grid_size_x = env->map_choice == 1 ? 10 : 20;  // Width of the grid
    int grid_size_y = env->map_choice == 1 ? 11 : env->map_choice == 2 ? 10 : 16;  // Height of the grid
    for (int i = 0; i < env->num_agents; i++) {
        // Agent location, direction, state
        int agent_location = env->agent_locations[i];
        int current_x = agent_location % grid_size_x;
        int current_y = agent_location / grid_size_x;
        env->observations[obs_idx] = env->agent_locations[i];
        env->observations[obs_idx+1] = env->agent_directions[i] + 1;
        env->observations[obs_idx+2] = env->agent_states[i];
        obs_idx+=3;
        for (int j = 0; j < 8; j++) {
            int x_offset = SURROUNDING_VECTORS[j][0];
            int y_offset = SURROUNDING_VECTORS[j][1];
            int new_x = current_x + x_offset;
            int new_y = current_y + y_offset;
            surround_indices[j] = new_x + new_y * grid_size_x;
            // other robots location and rotation if on that spot
            for (int k = 0; k < env->num_agents; k++) {
                if(i==k){
                    continue;
                }
                if(env->agent_locations[k] == surround_indices[j]){
                    env->observations[obs_idx] = 1;
                    env->observations[obs_idx+1] = env->agent_directions[k] + 1;
                    break;
                } else {
                    env->observations[obs_idx] = 0;
                    env->observations[obs_idx+1] = 0;
                    break;
                }
            }
            // boundary check
            if (new_x < 0 || new_x >= grid_size_x || new_y < 0 || new_y >= grid_size_y) {
                env->observations[obs_idx+2] = 0;
            } else {
                env->observations[obs_idx+2] = env->warehouse_states[surround_indices[j]];
            }
            obs_idx+=3;
        }
    }
    // for (int i = 0; i < env->num_agents*27; i++) {
    //     printf("obs[%d]: %f\n", i, env->observations[i]);
    // }
}

void reset(CRware* env) {
    env->log = (Log){0};
    env->dones[0] = 0;
    // set agents in center
    env->shelves_delivered = 0;
    env->human_agent_idx = 0;
    generate_map(env, env->map_choice == 1 ? tiny_map : env->map_choice == 2 ? small_map : medium_map);
    compute_observations(env);
}

void end_game(CRware* env){
    env->log.score = env->score;
    env->log.episode_return += env->rewards[0];
    add_log(env->log_buffer, &env->log);
    reset(env);
}

int get_direction(CRware* env, int action, int agent_idx) {
    // For reference: 
    // 0 = right (initial), 1 = down, 2 = left, 3 = up
    if (action == FORWARD) {
        return env->agent_directions[agent_idx];
    }
    else if (action == LEFT) {
        // Rotate counter-clockwise
        return (env->agent_directions[agent_idx] + 3) % NUM_DIRECTIONS;
    }
    else if (action == RIGHT) {
        // Rotate clockwise
        return (env->agent_directions[agent_idx] + 1) % NUM_DIRECTIONS;
    }
    return env->agent_directions[agent_idx];
}

int get_new_position(CRware* env, int agent_idx) {
    int grid_size_x = env->map_choice == 1 ? 10 : env->map_choice == 2 ? 20 : 20;
    int grid_size_y = env->map_choice == 1 ? 11 : env->map_choice == 2 ? 10 : 16;
    int current_position_x = env->agent_locations[agent_idx] % grid_size_x;
    int current_position_y = env->agent_locations[agent_idx] / grid_size_x;
    int new_position_x = current_position_x + DIRECTION_VECTORS[env->agent_directions[agent_idx]][0];
    int new_position_y = current_position_y + DIRECTION_VECTORS[env->agent_directions[agent_idx]][1];
    int new_position = new_position_x + new_position_y * grid_size_x;
    // check boundary
    if (new_position_x < 0 || new_position_x >= grid_size_x || new_position_y < 0 || new_position_y >= grid_size_y) {
        return -1;
    }
    // check if holding shelf and next position is a shelf
    if ((env->agent_states[agent_idx] == 1 || env->agent_states[agent_idx] == 2) && (env->warehouse_states[new_position] == 1 || env->warehouse_states[new_position] == 2)) {
        return -1;
    }
    // check if agent is trying to move into a goal with empty shelf
    if (env->agent_states[agent_idx] == 2 && env->warehouse_states[new_position] == 3) {
        return -1;
    }
    return new_position;
}

int detect_cycle_for_agent(CRware* env, int start_agent) {
    MovementGraph* graph = env->movement_graph;
    
    // If already processed or no target, skip
    if (graph->cycle_ids[start_agent] != -1 || 
        graph->target_positions[start_agent] == -1) {
        return -1;
    }

    // Initialize tortoise and hare
    int tortoise = find_agent_at_position(env, graph->target_positions[start_agent]);
    if (tortoise == -1) return -1;
    int hare = tortoise;

    // Move hare ahead by one step
    hare = find_agent_at_position(env, graph->target_positions[hare]);
    if (hare == -1) return -1;

    // Loop to detect cycle
    while (tortoise != hare) {
        tortoise = find_agent_at_position(env, graph->target_positions[tortoise]);
        if (tortoise == -1) return -1;

        hare = find_agent_at_position(env, graph->target_positions[hare]);
        if (hare == -1) return -1;
        hare = find_agent_at_position(env, graph->target_positions[hare]);
        if (hare == -1) return -1;
    }

    // Find the start of the cycle
    tortoise = start_agent;
    while (tortoise != hare) {
        tortoise = find_agent_at_position(env, graph->target_positions[tortoise]);
        hare = find_agent_at_position(env, graph->target_positions[hare]);
    }

    int cycle_start = tortoise;

    // Mark all agents in the cycle
    int cycle_id = graph->num_cycles++;
    int current = cycle_start;
    do {
        graph->cycle_ids[current] = cycle_id;
        current = find_agent_at_position(env, graph->target_positions[current]);
    } while (current != cycle_start);

    return cycle_id;
}

void detect_cycles(CRware* env) {
    for (int i = 0; i < env->num_agents; i++) {
        if(env->movement_graph->cycle_ids[i] == -1) {
            detect_cycle_for_agent(env, i);
        }
    }
}

void calculate_weights(CRware* env) {
    MovementGraph* graph = env->movement_graph;
    
    // First pass: identify leaf nodes (agents not targeted by others)
    for (int i = 0; i < env->num_agents; i++) {
        if (graph->cycle_ids[i] != -1) continue;  // Skip agents in cycles
        
        bool is_leaf = true;
        for (int j = 0; j < env->num_agents; j++) {
            if (graph->target_positions[j] == env->agent_locations[i]) {
                is_leaf = false;
                break;
            }
        }
        
        if (is_leaf) {
            graph->weights[i] = 1;  // Leaf node
        }
    }

    bool changed = true;
    while(changed) {
        changed = false;
        for (int i = 0; i < env->num_agents; i++) {
            if (graph->cycle_ids[i] != -1) continue;
            
            // Find agents targeting this agent's position
            int max_child_weight = 0;
            for (int j = 0; j < env->num_agents; j++) {
                if (graph->target_positions[j] == env->agent_locations[i]) {
                    max_child_weight = max(max_child_weight, graph->weights[j]);
                }
            }
            
            if (max_child_weight > 0 && graph->weights[i] != max_child_weight + 1) {
                graph->weights[i] = max_child_weight + 1;
                changed = true;
            }
        }
    }
}

void update_movement_graph(CRware* env, int agent_idx) {
    MovementGraph* graph = env->movement_graph;
    int new_position = get_new_position(env, agent_idx);
    if (new_position == -1) {
        return;
    }
    graph->target_positions[agent_idx] = new_position;

    // reset cycle and weights
    for (int i = 0; i < env->num_agents; i++) {
        graph->cycle_ids[i] = -1;
        graph->weights[i] = 0;
    }
    graph->num_cycles = 0;

    // detect cycles with Floyd algorithm
    detect_cycles(env);

    // calculate weights for tree
    calculate_weights(env);
}

void move_agent(CRware* env, int agent_idx) {
    /* given the direction, move the agent in that direction by 1 square
    the positions are mapped to a 1d array on a 10x11 grid
    */
    int new_position = get_new_position(env, agent_idx);
    // check boundary
    if (new_position == -1) {
        return;
    }

    // if reach goal
    if (env->warehouse_states[new_position] == 3 && env->agent_states[agent_idx]==1) {
        if (env->warehouse_states[env->agent_locations[agent_idx]] != 3) {
            env->warehouse_states[env->agent_locations[agent_idx]] = 0;
        }
        env->agent_locations[agent_idx] = new_position;
        return;
    }
    // if agent is holding requested shelf
    if (env->agent_states[agent_idx] == 1) {
        if (env->warehouse_states[env->agent_locations[agent_idx]] != 3) {
            env->warehouse_states[env->agent_locations[agent_idx]] = 0;
        }
        env->warehouse_states[new_position] = 2;
    }
    // if agent is holding empty shelf
    if (env->agent_states[agent_idx] == 2) {
        if (env->warehouse_states[env->agent_locations[agent_idx]] != 3){
            env->warehouse_states[env->agent_locations[agent_idx]] = 0;
        }
        env->warehouse_states[new_position] = 1;
    }
    env->agent_locations[agent_idx] = new_position;
    env->movement_graph->target_positions[agent_idx] = -1;
}

void pickup_shelf(CRware* env, int agent_idx) {
    // pickup shelf
    const int* map = env->map_choice == 1 ? tiny_map : env->map_choice == 2 ? small_map : medium_map;
    if ((env->warehouse_states[env->agent_locations[agent_idx]] == 2) && (env->agent_states[agent_idx]==0)) {
        env->agent_states[agent_idx]=1;
    }
    // return empty shelf
    else if (env->agent_states[agent_idx] == 2 && env->warehouse_states[env->agent_locations[agent_idx]] == map[env->agent_locations[agent_idx]] 
    && env->warehouse_states[env->agent_locations[agent_idx]] != 3) {
        env->agent_states[agent_idx]=0;
        env->warehouse_states[env->agent_locations[agent_idx]] = 1;
    }
    // drop shelf at goal
    else if (env->agent_states[agent_idx] == 1 && env->warehouse_states[env->agent_locations[agent_idx]] == 3) {
        env->agent_states[agent_idx]=2;
        env->rewards[agent_idx] = 1.0;
        env->log.episode_return += 1.0;
        env->shelves_delivered += 1;
        int shelf_count = 0;
        while (shelf_count < 1) {
            shelf_count += request_new_shelf(env);
        }
    }
}

void process_cycle_movements(CRware* env, MovementGraph* graph) {
    for (int cycle = 0; cycle < graph->num_cycles; cycle++) {
        int cycle_size = 0;
        for (int i = 0; i < env->num_agents; i++) {
            if (graph->cycle_ids[i] == cycle) {
                cycle_size++;
            }
        }
        if (cycle_size == 2) continue;

        bool can_move_cycle = true;
        // Verify all agents in cycle can move
        for (int i = 0; i < env->num_agents; i++) {
            if (graph->cycle_ids[i] != cycle) continue;
            int new_pos = get_new_position(env, i);
            if (new_pos == -1) {
                can_move_cycle = false;
                break;
            }
        }
        
        // Move all agents in cycle if possible
        if (!can_move_cycle) continue;
        for (int i = 0; i < env->num_agents; i++) {
            if (graph->cycle_ids[i] != cycle) continue;
            if (env->actions[i] != FORWARD) continue;            
            move_agent(env, i);
        }
    }
}

void process_tree_movements(CRware* env, MovementGraph* graph) {
    int max_weight = 0;
    for (int i = 0; i < env->num_agents; i++) {
        if (graph->cycle_ids[i] == -1 && graph->weights[i] > max_weight) {
            max_weight = graph->weights[i];
        }
    }
    // Process from highest weight to lowest
    for (int weight = max_weight; weight > 0; weight--) {
        for (int i = 0; i < env->num_agents; i++) {
            if (graph->cycle_ids[i] != -1 || graph->weights[i] != weight) continue;
            if (env->actions[i] != FORWARD) continue;

            int new_pos = get_new_position(env, i);
            if (new_pos == -1) continue;

            int target_agent = find_agent_at_position(env, new_pos);
            if (target_agent != -1) continue;
            move_agent(env, i);
        }
    }
}

void step(CRware* env) {
    env->log.episode_length += 1;
    env->rewards[0] = 0.0;

    // if (env->log.episode_length >= 500) {
    //     env->dones[0] = 1;
    //     end_game(env);
    //     printf("reset!\n");
    //     return;
    // }

    // Process each agent's actions
    // Check if any agent is trying to move forward
    // First handle non-movement actions for all agents
    MovementGraph* graph = env->movement_graph;
    for (int i = 0; i < env->num_agents; i++) {
        int action = (int)env->actions[i];
        
        // Handle direction changes and non-movement actions
        if (action != NOOP && action != TOGGLE_LOAD) {
            env->agent_directions[i] = get_direction(env, action, i);
        }
        if (action == TOGGLE_LOAD) {
            pickup_shelf(env, i);
        }
        if (action == TOGGLE_AGENT) {
            env->human_agent_idx = (env->human_agent_idx + 1) % env->num_agents;
            continue;
        }
        if (env->actions[i] == FORWARD) {
            update_movement_graph(env, i);
        }
    }
    int is_movement=0;
    for(int i=0; i<env->num_agents; i++) {
        if (env->actions[i] == FORWARD) is_movement++;
    }
    if (is_movement>=1) {
        // Process movements in cycles first
        process_cycle_movements(env, graph);
        // process tree movements
        process_tree_movements(env, graph);
    }

    if (env->dones[0] == 1) {
        end_game(env);
    }
    compute_observations(env);
}

const Color STONE_GRAY = (Color){80, 80, 80, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D puffers;
};

Client* make_client(int width, int height) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = width;
    client->height = height;
    InitWindow(width, height, "PufferLib Ray RWare");
    SetTargetFPS(15);
    client->puffers = LoadTexture("resources/puffers_128.png");
    return client;
}

void render(Client* client, CRware* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    // State 0: Empty - white
    // State 1: Shelf - dark blue
    // State 2: Requested Shelf - cyan puffer
    // State 3: Agent on map - puffer vector
    // State 4: Agent with full shelf - red puffer
    // State 5: Agent with empty shelf - dark blue and puffer
    // State 6: Goal - dark gray
    int map_size = env->map_choice == 1 ? 110 : env->map_choice == 2 ? 200 : 320;
    int grid_size_x = env->map_choice == 1 ? 10 : 20;  // Tiny map is 10x11, small map is 20x10
    for (int i = 0; i < map_size; i++) {
        int state = env->warehouse_states[i];
        Color color;
        color = state == 1 ? DARKBLUE : 
                state == 2 ? PUFF_CYAN :
                state == 3 ? STONE_GRAY : 
                PUFF_WHITE;
         DrawRectangle(
            i % grid_size_x * env->grid_square_size,  // x position
            i / grid_size_x * env->grid_square_size,  // y position
            env->grid_square_size, 
            env->grid_square_size, 
            color
        );
        
        DrawRectangleLines(
            i % grid_size_x * env->grid_square_size,
            i / grid_size_x * env->grid_square_size,
            env->grid_square_size,
            env->grid_square_size,
            BLACK
        );

        DrawRectangleLines(
            i % grid_size_x * env->grid_square_size + 1,
            i / grid_size_x * env->grid_square_size + 1,
            env->grid_square_size - 2,
            env->grid_square_size - 2,
            state != 0 ? WHITE : BLANK
        );
        // draw agent
        for (int j = 0; j < env->num_agents; j++) {
            if (env->agent_locations[j] != i) {
                continue;
            }
            
            DrawTexturePro(
                client->puffers,
                (Rectangle){0, 0, 128, 128},
                (Rectangle){
                    i % grid_size_x * env->grid_square_size + env->grid_square_size/2,
                    i / grid_size_x * env->grid_square_size + env->grid_square_size/2,
                    env->grid_square_size,
                    env->grid_square_size
                },
                (Vector2){env->grid_square_size/2, env->grid_square_size/2},
                90*env->agent_directions[j],
                env->agent_states[j] != 0 ? RED : WHITE
            );
            // put a number on top of the agent
            DrawText(
                TextFormat("%d", j),
                i % grid_size_x * env->grid_square_size + env->grid_square_size/2,
                i / grid_size_x * env->grid_square_size + env->grid_square_size/2,
                20,
                WHITE
            );
        }
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    
    EndDrawing();
}
void close_client(Client* client) {
    CloseWindow();
    free(client);
}
