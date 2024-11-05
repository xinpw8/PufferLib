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
#define TICK_RATE 1.0f/60.0f
#define NUM_DIRECTIONS 4
static const int DIRECTIONS[NUM_DIRECTIONS] = {0, 1, 2, 3};
static const int DIRECTION_VECTORS[NUM_DIRECTIONS][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
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
//  LD_LIBRARY_PATH=raylib/lib ./go
#define LOG_BUFFER_SIZE 1024

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
};

void generate_tiny_map(CRware* env) {
    int shelves_count = 0;
    // seed new random
    srand(time(NULL));
    for (int i = 0; i < 110; i++) {
        env->warehouse_states[i] = tiny_map[i];
        if (tiny_map[i] == 1 && shelves_count < env->num_requested_shelves && rand() % 100 < 50) {
            env->warehouse_states[i] = 2;
            shelves_count += 1;
        }
    }
    // set agents in center
    for (int i = 0; i < env->num_agents; i++) {
        env->agent_locations[i] = 54 + i;
        env->agent_directions[i] = 0;
        env->agent_states[i] = 0;
    }
}


void init(CRware* env) {
    env->warehouse_states = (int*)calloc(110, sizeof(int));
    env->agent_locations = (int*)calloc(env->num_agents, sizeof(int));
    env->agent_directions = (int*)calloc(env->num_agents, sizeof(int));
    env->agent_states = (int*)calloc(env->num_agents, sizeof(int));
    if (env->map_choice == 1) {
        generate_tiny_map(env);
    }
}

void allocate(CRware* env) {
    init(env);
    env->observations = (float*)calloc(110, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->dones = (unsigned char*)calloc(1, sizeof(unsigned char));
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
    memcpy(env->observations, env->warehouse_states, 110 * sizeof(float));
}

void reset(CRware* env) {
    env->log = (Log){0};
    env->dones[0] = 0;
    // set agents in center
    env->shelves_delivered = 0;
    generate_tiny_map(env);
    compute_observations(env);
}

void end_game(CRware* env){
    env->log.score = env->score;
    env->log.episode_return += env->rewards[0];
    add_log(env->log_buffer, &env->log);
    reset(env);
}

int get_direction(CRware* env, int action) {
    // For reference: 
    // 0 = right (initial), 1 = down, 2 = left, 3 = up
    if (action == FORWARD) {
        return env->agent_directions[0];
    }
    else if (action == LEFT) {
        // Rotate counter-clockwise
        return (env->agent_directions[0] + 3) % NUM_DIRECTIONS;
    }
    else if (action == RIGHT) {
        // Rotate clockwise
        return (env->agent_directions[0] + 1) % NUM_DIRECTIONS;
    }
    return env->agent_directions[0];
}

void move_agent(CRware* env) {
    /* given the direction, move the agent in that direction by 1 square
    the positions are mapped to a 1d array on a 10x11 grid
    */
    int grid_size_x = 10;
    int grid_size_y = 11;
    int current_position_x = env->agent_locations[0] % grid_size_x      ;
    int current_position_y = env->agent_locations[0] / grid_size_x;
    int new_position_x = current_position_x + DIRECTION_VECTORS[env->agent_directions[0]][0];
    int new_position_y = current_position_y + DIRECTION_VECTORS[env->agent_directions[0]][1];
    int new_position = new_position_x + new_position_y * grid_size_x;
    
    // check boundary
    if (new_position_x < 0 || new_position_x >= grid_size_x || new_position_y < 0 || new_position_y >= grid_size_y) {
        return;
    }
    // check if holding shelf and next position is a shelf
    if ((env->agent_states[0] == 1 || env->agent_states[0] == 2) && (env->warehouse_states[new_position] == 1 || env->warehouse_states[new_position] == 2)) {
        return;
    }

    if (env->agent_states[0] == 2 && env->warehouse_states[new_position] == 3) {
        return;
    }

    // if reach goal
    if (env->warehouse_states[new_position] == 3 && env->agent_states[0]==1) {
        if (env->warehouse_states[env->agent_locations[0]] != 3) {
            env->warehouse_states[env->agent_locations[0]] = 0;
        }
        env->agent_locations[0] = new_position;
        return;
    }
    
    if (env->agent_states[0] == 1) {
        if (env->warehouse_states[env->agent_locations[0]] != 3) {
            env->warehouse_states[env->agent_locations[0]] = 0;
        }
        env->warehouse_states[new_position] = 2;
    }
    if (env->agent_states[0] == 2) {
        if (env->warehouse_states[env->agent_locations[0]] != 3){
            env->warehouse_states[env->agent_locations[0]] = 0;
        }
        env->warehouse_states[new_position] = 1;
    }
    env->agent_locations[0] = new_position;
}

void pickup_shelf(CRware* env) {
    // pickup shelf
    if (env->warehouse_states[env->agent_locations[0]] == 2 & env->agent_states[0]==0) {
        env->agent_states[0]=1;
    }
    // return empty shelf
    else if (env->agent_states[0] == 2 && env->warehouse_states[env->agent_locations[0]] == tiny_map[env->agent_locations[0]]) {
        env->agent_states[0]=0;
        env->warehouse_states[env->agent_locations[0]] = 1;
    }
    // drop shelf at goal
    else if (env->agent_states[0] == 1 && env->warehouse_states[env->agent_locations[0]] == 3) {
        env->agent_states[0]=2;
        env->rewards[0] = 1.0;
        env->log.episode_return += 1.0;
        env->shelves_delivered += 1;
        if (env->shelves_delivered == env->num_requested_shelves) {
            env->dones[0] = 1;
        }
    }
}

void step(CRware* env) {
    env->log.episode_length += 1;
    env->rewards[0] = 0.0;
    int action = (int)env->actions[0];
    if (action != NOOP && action != TOGGLE_LOAD) {
        // Turn agent
        env->agent_directions[0] = get_direction(env, action);
    }
    if (action == FORWARD) {
        move_agent(env);
    }
    if (action == TOGGLE_LOAD) {
        pickup_shelf(env);
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
    for (int i = 0; i < 110; i++) {
        int state = env->warehouse_states[i];
        Color color;
        switch(state) {
            case 0: // Empty
                color = PUFF_WHITE;
                break;
            case 1: // Shelf
                color = DARKBLUE;
                break;
            case 2: // Requested Shelf
                color = PUFF_CYAN;
                break;
            case 3: // Goal
                color = STONE_GRAY;
                break;
            default:
                color = PUFF_WHITE;
        }
        DrawRectangle(i%10*32, i/10*32, 32, 32, color);
        DrawRectangleLines(i%10*32, i/10*32, 32, 32, BLACK);
        // DRAW RECTANGLE LINES INNER OUTLINE WHITE
        DrawRectangleLines(i%10*32+1, i/10*32+1, 30, 30, state != 0 ? WHITE : BLANK);

        // draw agent
        for (int j = 0; j < env->num_agents; j++) {
            if (env->agent_locations[j] != i) {
                continue;
            }
            
            DrawTexturePro(
                client->puffers,
                (Rectangle){0, 0, 128, 128},
                (Rectangle){i%10*32 +16, i/10*32 +16, 32, 32},
                (Vector2){16, 16},
                90*env->agent_directions[j],
                env->agent_states[j] != 0 ? RED : WHITE
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
