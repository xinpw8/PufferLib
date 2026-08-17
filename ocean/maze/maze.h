#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "raylib.h"
typedef unsigned char obs_t;
#include "pufferenv.h"

#define ACT_SIZES {5}
#define MY_VEC_INIT
#define MY_VEC_CLOSE
typedef Env Grid;

#define TWO_PI 2.0*PI

#define ATN_PASS 0
#define ATN_EAST 1
#define ATN_NORTH 2
#define ATN_WEST 3
#define ATN_SOUTH 4
#define EMPTY 0
#define WALL 1
#define AGENT 2
#define GOAL 4

#define VISION 5
#define WINDOW (2*VISION + 1)
#define MAX_SIZE 47
#define OBS_SIZE 121
#define NUM_ATNS 1

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct {
    int cell_size;
    int width;
    int height;
    Texture2D puffer;
    float* overlay;
} Renderer;

typedef struct {
    int width;
    int height;
    int spawn_x;
    int spawn_y;
    int x;
    int y;
    int direction;
    unsigned char maze[MAX_SIZE*MAX_SIZE];
} State;

struct Env {
    Renderer* renderer;
    State* levels;
    State state;
    Log log;
    Agent agents[1];
    int num_levels;
    int num_agents;
    int tag;
    int boundary_reached;
    int owns_levels;
    int tick;
    unsigned int rng;
};

bool in_bounds(State* s, int y, int c) {
    return (y >= 0 && y <= s->height && c >= 0 && c <= s->width);
}

int maze_offset(int y, int x) {
    return y*MAX_SIZE + x;
}

void add_log(Grid* env, int idx) {
    env->log.perf += env->agents[0].rewards[idx];
    env->log.score += env->agents[0].rewards[idx];
    env->log.episode_return += env->agents[0].rewards[idx];
    env->log.episode_length += env->tick;
    env->log.n += 1.0;
}
 
void compute_observations(Grid* env) {
    obs_t* obs = env->agents[0].observations;
    memset(obs, 0, WINDOW*WINDOW*env->num_agents);
    State* s = &env->state;
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        int x = s->x;
        int y = s->y;
        int start_r = y - VISION;
        if (start_r < 0) {
            start_r = 0;
        }

        int start_c = x - VISION;
        if (start_c < 0) {
            start_c = 0;
        }

        int end_r = y + VISION;
        if (end_r >= MAX_SIZE) {
            end_r = MAX_SIZE - 1;
        }

        int end_c = x + VISION;
        if (end_c >= MAX_SIZE) {
            end_c = MAX_SIZE - 1;
        }

        int obs_offset = agent_idx*WINDOW*WINDOW;
        for (int r = start_r; r <= end_r; r++) {
            for (int c = start_c; c <= end_c; c++) {
                int r_idx = r - y + VISION;
                int c_idx = c - x + VISION;
                int obs_adr = obs_offset + r_idx*WINDOW + c_idx;
                int adr = maze_offset(r, c);
                obs[obs_adr] = s->maze[adr];
            }
        }
    }
}

void puf_reset(Env* env) {
env->tick = 0;
    int idx = rand_r(&env->rng) % env->num_levels;
    env->state = env->levels[idx];
    compute_observations(env);
}

int move_to(Grid* env, int agent_idx, float y, float x) {
    if (!in_bounds(&env->state, y, x)) {
        return 1;
    }

    State* s = &env->state;
    int adr = maze_offset(round(y), round(x));
    int dest = s->maze[adr];
    if (dest == WALL) {
        return 1;
    } else if (dest == GOAL) {
        env->agents[0].rewards[agent_idx] = 1.0;
        env->agents[0].terminals[agent_idx] = 1.0f;
        add_log(env, agent_idx);
    }

    int start_adr = maze_offset(s->y, s->x);
    s->maze[start_adr] = EMPTY;
    s->maze[adr] = AGENT;
    s->y = y;
    s->x = x;
    return 0;
}
 
// Hold Left Shift + WASD/arrows.
static void maze_human_controls(Env *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    State* s = &env->state;
    if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
        env->agents[0].actions[0] = ATN_NORTH;
    } else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
        env->agents[0].actions[0] = ATN_SOUTH;
    } else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        s->direction = PI;
        env->agents[0].actions[0] = ATN_WEST;
    } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        s->direction = 0;
        env->agents[0].actions[0] = ATN_EAST;
    } else {
        env->agents[0].actions[0] = ATN_PASS;
    }
}

void puf_step(Env* env) {
    maze_human_controls(env);
    env->agents[0].terminals[0] = 0.0f;
    env->agents[0].rewards[0] = 0.0f;

    State* s = &env->state;
    env->tick++;

    int atn = env->agents[0].actions[0];
    int direction = s->direction;
    if (atn != ATN_PASS) {
        direction = atn;
    }

    int x = s->x;
    int y = s->y;
    int dest_x = x;
    int dest_y = y;
    if (direction == ATN_EAST) {
        dest_x = x + 1;
    } else if (direction == ATN_NORTH) {
        dest_y = y - 1;
    } else if (direction == ATN_WEST) {
        dest_x = x - 1;
    } else if (direction == ATN_SOUTH) {
        dest_y = y + 1;
    }
    if (in_bounds(&env->state, dest_y, dest_x)) {
        int err = move_to(env, 0, dest_y, dest_x);
    }

    compute_observations(env);

    if (env->tick >= 2*s->width*s->height) {
        env->agents[0].terminals[0] = 1.0f;
        add_log(env, 0);
    }

    if (env->agents[0].terminals[0]) {
        puf_reset(env);
        int idx = rand_r(&env->rng) % env->num_levels;
        env->state = env->levels[idx];
        compute_observations(env);
    }
}

Renderer* init_renderer(int cell_size, int width, int height) {
    Renderer* renderer = (Renderer*)calloc(1, sizeof(Renderer));
    renderer->cell_size = cell_size;
    renderer->width = width;
    renderer->height = height;

    renderer->overlay = (float*)calloc(width*height, sizeof(float));

    InitWindow(width*cell_size, height*cell_size, "PufferLib Grid");
    SetTargetFPS(60);

    renderer->puffer = LoadTexture("resources/shared/puffers_128.png");
    return renderer;
}

void clear_overlay(Renderer* renderer) {
    memset(renderer->overlay, 0, renderer->width*renderer->height*sizeof(float));
}

void puf_close(Env* env) {
    if (env->renderer) {
        free(env->renderer->overlay);
        CloseWindow();
        free(env->renderer);
    }
    if (env->owns_levels) {
        free(env->levels);
    }
}

void puf_render(Env* env) {
    float overlay = 0.0;
    if (env->renderer == NULL) {
        env->renderer = init_renderer(16, MAX_SIZE, MAX_SIZE);
    }
    Renderer* renderer = env->renderer;
 
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    maze_human_controls(env);

    State* s = &env->state;
    int r = s->y;
    int c = s->x;
    int adr = maze_offset(r, c);

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int ts = renderer->cell_size;
    for (int r = 0; r < s->height; r++) {
        for (int c = 0; c < s->width; c++){
            adr = maze_offset(r, c);
            int tile = s->maze[adr];
            if (tile == EMPTY) {
                continue;
                overlay = renderer->overlay[adr];
                if (overlay == 0) {
                    continue;
                }
                Color color;
                if (overlay < 0) {
                    overlay = -fmaxf(-1.0, overlay);
                    color = (Color){255.0*overlay, 0, 0, 255};
                } else {
                    overlay = fminf(1.0, overlay);
                    color = (Color){0, 255.0*overlay, 0, 255};
                }
                DrawRectangle(c*ts, r*ts, ts, ts, color);
            }

            Color color;
            if (tile == WALL) {
                color = (Color){128, 128, 128, 255};
            } else if (tile == GOAL) {
                color = GREEN;
            } else {
                continue;
            }
 
            DrawRectangle(c*ts, r*ts, ts, ts, color);
       }
    }

    float y = s->y;
    float x = s->x;
    Rectangle source_rect = (Rectangle){0, 0, 128, 128};
    Rectangle dest_rect = (Rectangle){x*ts, y*ts, ts, ts};
    DrawTexturePro(renderer->puffer, source_rect, dest_rect,
        (Vector2){0, 0}, 0, WHITE);

    EndDrawing();
    puf_web_vsync();
}

void generate_growing_tree_maze(unsigned char* maze,
        int width, int height, int max_size, float difficulty, int seed) {
    unsigned int rng = seed;
    int dx[4] = {-1, 0, 1, 0};
    int dy[4] = {0, 1, 0, -1};
    int dirs[4] = {0, 1, 2, 3};
    int cells[2*width*height];
    int num_cells = 1;

    bool visited[width*height];
    memset(visited, false, width*height);

    memset(maze, WALL, max_size*height);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*max_size + c;
            if (r % 2 == 1 && c % 2 == 1) {
                maze[adr] = EMPTY;
            }
        }
    }

    int x_init = rand_r(&rng) % (width - 1);
    int y_init = rand_r(&rng) % (height - 1);

    if (x_init % 2 == 0) {
        x_init++;
    }
    if (y_init % 2 == 0) {
        y_init++;
    }

    int adr = y_init*height + x_init;
    visited[adr] = true;
    cells[0] = x_init;
    cells[1] = y_init;

    while (num_cells > 0) {
        if (rand_r(&rng) % 1000 > 1000*difficulty) {
            int i = rand_r(&rng) % num_cells;
            int tmp_x = cells[2*num_cells - 2];
            int tmp_y = cells[2*num_cells - 1];
            cells[2*num_cells - 2] = cells[2*i];
            cells[2*num_cells - 1] = cells[2*i + 1];
            cells[2*i] = tmp_x;
            cells[2*i + 1] = tmp_y;
 
        }

        int x = cells[2*num_cells - 2];
        int y = cells[2*num_cells - 1];
 
        int nx, ny;

        // In-place direction shuffle
        for (int i = 0; i < 4; i++) {
            int ii = i + rand_r(&rng) % (4 - i);
            int tmp = dirs[i];
            dirs[i] = dirs[ii];
            dirs[ii] = tmp;
        }

        bool made_path = false;
        for (int dir_i = 0; dir_i < 4; dir_i++) {
            int dir = dirs[dir_i];
            nx = x + 2*dx[dir];
            ny = y + 2*dy[dir];
           
            if (nx <= 0 || nx >= width-1 || ny <= 0 || ny >= height-1) {
                continue;
            }

            int visit_adr = ny*width + nx;
            if (visited[visit_adr]) {
                continue;
            }

            visited[visit_adr] = true;
            cells[2*num_cells] = nx;
            cells[2*num_cells + 1] = ny;

            nx = x + dx[dir];
            ny = y + dy[dir];

            int adr = ny*max_size + nx;
            maze[adr] = EMPTY;
            num_cells++;

            made_path = true;
            break;
        }
        if (!made_path) {
            num_cells--;
        }
    }
}

void make_border(State* s) {
    for (int r = 0; r < s->height; r++) {
        int adr = maze_offset(r, 0);
        s->maze[adr] = WALL;
        adr = maze_offset(r, s->width-1);
        s->maze[adr] = WALL;
    }
    for (int c = 0; c < s->width; c++) {
        int adr = maze_offset(0, c);
        s->maze[adr] = WALL;
        adr = maze_offset(s->height-1, c);
        s->maze[adr] = WALL;
    }
}

void spawn_agent(State* s, int idx, int x, int y) {
    int spawn_y = y;
    int spawn_x = x;
    assert(in_bounds(s, spawn_y, spawn_x));
    int adr = maze_offset(spawn_y, spawn_x);
    assert(s->maze[adr] == EMPTY);
    s->spawn_y = spawn_y;
    s->spawn_x = spawn_x;
    s->y = spawn_y;
    s->x = spawn_x;
    s->maze[adr] = AGENT;
    s->direction = 0;
}

void create_maze_level(State* s, float difficulty, int seed) {
    generate_growing_tree_maze(s->maze, s->width, s->height, MAX_SIZE, difficulty, seed);
    make_border(s);
    spawn_agent(s, 0, 1, 1);
    int goal_adr = maze_offset(s->height - 2, s->width - 2);
    s->maze[goal_adr] = GOAL;
}

State* make_maze_levels(int num_maps, int map_size) {
    State* levels = (State*)calloc(num_maps, sizeof(State));
    unsigned int map_rng = 42;
    for (int i = 0; i < num_maps; i++) {
        int sz = map_size;
        if (map_size == -1) {
            sz = 5 + (rand_r(&map_rng) % (MAX_SIZE - 5));
        }
        if (sz % 2 == 0) {
            sz -= 1;
        }

        State* level = &levels[i];
        level->width = sz;
        level->height = sz;

        float difficulty = (float)rand_r(&map_rng) / (float)(RAND_MAX);
        create_maze_level(level, difficulty, i);
    }
    return levels;
}

void puf_init(Env* env, Dict* kwargs) {
    int num_maps = dict_get(kwargs, "num_maps");
    int map_size = dict_get(kwargs, "map_size");
    env->num_levels = num_maps;
    env->num_agents = 1;
    env->levels = make_maze_levels(num_maps, map_size);
    env->owns_levels = 1;
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
}

Env* my_vec_init(int* num_envs_out, int* env_starts, int* env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = dict_get(vec_kwargs, "total_agents");
    int num_buffers = dict_get(vec_kwargs, "num_buffers");
    int agents_per_buf = total_agents / num_buffers;
    int num_envs = total_agents;

    int num_maps = dict_get(env_kwargs, "num_maps");
    int map_size = dict_get(env_kwargs, "map_size");
    State* levels = make_maze_levels(num_maps, map_size);

    Env* envs = (Env*)calloc(num_envs, sizeof(Env));
    int buf = 0;
    int buf_agents = 0;
    env_starts[0] = 0;
    env_counts[0] = 0;

    unsigned int env_rng = 42;
    for (int i = 0; i < num_envs; i++) {
        Env* env = &envs[i];
        env->num_levels = num_maps;
        env->num_agents = 1;
        env->levels = levels;
        env->rng = rand_r(&env_rng);
        env->agents[0].action_mask = NULL;
        env->agents[0].policy = 0;

        buf_agents += env->num_agents;
        env_counts[buf]++;
        if (buf_agents >= agents_per_buf && buf < num_buffers - 1) {
            buf++;
            env_starts[buf] = i + 1;
            env_counts[buf] = 0;
            buf_agents = 0;
        }
    }

    *num_envs_out = num_envs;
    return envs;
}

void my_vec_close(Env* envs) {
    free(envs[0].levels);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}
