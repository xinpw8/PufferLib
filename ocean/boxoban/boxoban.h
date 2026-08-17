#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "raylib.h"
typedef unsigned char obs_t;
#include "pufferenv.h"

#define BOXOBAN_MAPS_IMPLEMENTATION
#include "boxoban_maps.h"

#define ACT_SIZES {5}
#define OBS_SIZE 400
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 6

const unsigned char NOOP = 0;
const unsigned char DOWN = 1;
const unsigned char UP = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;

const unsigned char AGENT = 0;
const unsigned char WALLS = 1;
const unsigned char BOXES = 2;
const unsigned char TARGET = 3;

// Required struct. Only use floats!
struct Log {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float on_targets; // Number of targets currently boxed
    float n; // Required as the last field 
};

typedef struct {
    Texture2D wall;
    Texture2D box;
    Texture2D target;
    Texture2D floor;
    Texture2D agent;
    Texture2D box_on_target;
} Client;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
struct Env {
    Log log; // Required field. Env binding code uses this to aggregate logs
    Agent agents[1];
    int tag;
    int boundary_reached;
    unsigned int rng;
    int size;
    int num_agents;
    int tick;
    int max_steps;
    int agent_x;
    int agent_y;
    bool initialized;
    unsigned char* intermediate_rewards;
    float int_r_coeff;
    float target_loss_pen_coeff;
    int on_target; //num targets currently boxed
    int n_boxes; //boxes in map
    int n_targets; //targets in map
    int difficulty_id; // 0=basic,1=easy,2=medium,3=hard,4=unfiltered
    Client* client;
    int win;
    float episode_return;
};
typedef Env Boxoban;

void ensure_map_loaded(void);

static int boxoban_configure_maps_from_env(Boxoban* env) {
    if (BOXOBAN_MAP_PATH != NULL) {
        return 0;
    }
    if (env->difficulty_id == -1) {
        return 0;
    }

    if (env->difficulty_id < -1) {
        fprintf(stderr, "Invalid Boxoban difficulty id %d\n", env->difficulty_id);
        return -1;
    }

    const char* difficulty_name = boxoban_difficulty_name_from_id(env->difficulty_id);
    if (difficulty_name == NULL) {
        fprintf(stderr, "Invalid Boxoban difficulty id %d\n", env->difficulty_id);
        return -1;
    }
    char prepared_path[512];
    if (boxoban_prepare_maps_for_difficulty(difficulty_name, prepared_path, sizeof(prepared_path)) != 0) {
        return -1;
    }

    return 0;
}

//Entity,x,y  convention y moves top to bottom

static inline void set_entity(Boxoban *env, int entity, int x, int y, unsigned char value) {
    unsigned char* obs = env->agents[0].observations;
    obs[(entity)*env->size*env->size + (y)*env->size + (x)] = value;
}

static inline unsigned char get_entity(Boxoban *env, int entity, int x, int y) {
    unsigned char* obs = env->agents[0].observations;
    return obs[(entity)*env->size*env->size + (y)*env->size + (x)];
}

static inline void set_intermediate_reward(Boxoban *env, int x, int y, unsigned char value) {
    env->intermediate_rewards[(y)*env->size + (x)] = value;
}

static inline unsigned char get_intermediate_reward_status(Boxoban *env, int x, int y) {
    return env->intermediate_rewards[(y)*env->size + (x)];
}

static inline uint32_t get_random_puzzle_idx(Boxoban *env) {
    int idx = rand_r(&env->rng) % PUZZLE_COUNT;
    return idx;
}

void init (Boxoban* env) {
    static int boxoban_maps_ready = 0;
    if (!boxoban_maps_ready) {
        if (boxoban_configure_maps_from_env(env) != 0) {
            fprintf(stderr, "Failed to configure Boxoban maps\n");
            abort();
        }
        ensure_map_loaded();
        boxoban_maps_ready = 1;
    }
    env->intermediate_rewards = (unsigned char*)calloc(env->size*env->size, sizeof(unsigned char));
    env->win = 0;
    env->initialized = false;
  }

void add_log(Boxoban* env) {
    float denom = (float)env->n_boxes;
    float num = (float)env->on_target;
    float perf = (env->win== 1) ? 1.0f : 0.0f;
    env->log.perf += perf;
    env->log.score += perf;
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return;
    env->log.on_targets += env->on_target;
    env->log.n++;
}

bool clear(Boxoban* env, int x, int y) {
    if (x < 0 || y < 0 || x >= env->size || y >= env->size) {
        return false;
    }
    return (get_entity(env, WALLS, x, y) == 0) && (get_entity(env, BOXES, x, y) == 0);
}

// Required function
void puf_reset(Boxoban* env) {
    unsigned char* obs = env->agents[0].observations;
    const uint32_t i = get_random_puzzle_idx(env);
    const uint8_t* puzzle = MAP_BASE + (size_t)i * PUZZLE_SIZE;
    memcpy(obs, puzzle, PUZZLE_OBS_BYTES);

    const uint8_t* meta = puzzle + PUZZLE_OBS_BYTES;
    env->agent_x = (int)meta[0];
    env->agent_y = (int)meta[1];
    env->n_boxes = (int)meta[2];
    env->n_targets = (int)meta[3];
    env->on_target = (int)meta[4];

    memcpy(env->intermediate_rewards,
            obs + TARGET * env->size * env->size,env->size * env->size);

    env->tick = 0;
    env->win = 0;
    env->episode_return = 0;

    if (!env->initialized) {
        env->tick = rand_r(&env->rng) % env->max_steps;
        env->initialized = true;
    }
}

//Updates OBS for moved entity
void move_entity(Boxoban* env,unsigned char entity,int x, int y, int dx, int dy) {
    set_entity(env, entity, x, y, 0);
    set_entity(env, entity, x + dx, y + dy, 1);
}

// Returns target_delta for the pushed box: +1 onto a target, -1 off a target.
int take_action(Boxoban* env, int action) {

    int dx = 0;
    int dy = 0;
    int target_delta = 0;

    if (action == NOOP) {
        return 0;
    }
    else if (action == DOWN) {
        dy = 1;
    }
    else if (action == UP) {
        dy = -1;
    }
    else if (action == LEFT) {
        dx = -1;
    }
    else if (action == RIGHT) {
        dx = 1;
    }

    //if move space is clear, move agent
    if (clear(env, env->agent_x + dx, env->agent_y + dy)) {
        
        move_entity(env, AGENT, env->agent_x, env->agent_y, dx, dy);
        env->agent_y += dy;
        env->agent_x += dx;
        return 0;
    }

    int box_x = env->agent_x + dx;
    int box_y = env->agent_y + dy;
    int box_dest_x = box_x + dx;
    int box_dest_y = box_y + dy;

    //if its not clear, but its a box and box is clear to move, move both
    if (clear(env, box_dest_x, box_dest_y)
            && get_entity(env, BOXES, box_x, box_y) == 1) {

            if (get_entity(env, TARGET, box_x, box_y) == 1) {
                env->on_target -= 1;
                target_delta -= 1;
            }
            move_entity(env, BOXES, box_x, box_y, dx, dy);
            move_entity(env, AGENT, env->agent_x, env->agent_y, dx, dy);
            env->agent_y += dy;
            env->agent_x += dx;

            if (get_entity(env, TARGET, box_dest_x, box_dest_y) == 1) {
                env->on_target += 1;
                target_delta += 1;
            }
            return target_delta;
    }
    return 0;
}

// Hold Shift + WASD/arrows. Skip the step when no direction this frame.
static int boxoban_human_controls(Boxoban *env) {
    if (!IsWindowReady()
            || (!IsKeyDown(KEY_LEFT_SHIFT) && !IsKeyDown(KEY_RIGHT_SHIFT))) {
        return 0;
    }
    int new_action = -1;
    if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
        new_action = UP;
    }
    if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
        new_action = DOWN;
    }
    if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
        new_action = LEFT;
    }
    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
        new_action = RIGHT;
    }
    if (new_action < 0) {
        return -1;
    }
    env->agents[0].actions[0] = new_action;
    return 1;
}

// Required function
void puf_step(Boxoban* env) {
    if (boxoban_human_controls(env) < 0) {
        return;
    }
    env->tick += 1;
    env->agents[0].terminals[0] = 0;
    env->agents[0].rewards[0] = 0.0;
       
    int action = (int)env->agents[0].actions[0];

    int target_delta = take_action(env, action);
    env->agents[0].rewards[0] += (float)target_delta * env->int_r_coeff;

    //Terminals
    if (env->on_target == env->n_targets) {
        env->agents[0].terminals[0] = 1;
        env->agents[0].rewards[0] += 1.0;
        env->win = 1;
        env->episode_return += env->agents[0].rewards[0];
        add_log(env);
        puf_reset(env);
        return;
    }

    if (env->tick >= env->max_steps) {
        env->agents[0].terminals[0] = 1;
        env->agents[0].rewards[0] -= 1.0; 
        env->episode_return += env->agents[0].rewards[0];
        add_log(env);
        puf_reset(env);
        return;
    }
    env->episode_return += env->agents[0].rewards[0];

}

Client* make_client(Boxoban* env) {
    Client* client = (Client*)calloc(1,sizeof(Client));
    client->wall = LoadTexture("resources/boxoban/Wall_Black.jpg");
    client->box = LoadTexture("resources/boxoban/Crate_Black.jpg");
    client->target = LoadTexture("resources/boxoban/EndPoint_Black.jpg");
    client->floor = LoadTexture("resources/boxoban/GroundGravel_Concrete.jpg");
    client->box_on_target = LoadTexture("resources/boxoban/EndPoint_Blue.jpg");
    client->agent = LoadTexture("resources/shared/puffers_128.png");
    env-> client = client;
    return client;
}

#define TILE 32

Texture2D choose_sprite(Client *c, Boxoban *env, int x, int y) {
    int a = get_entity(env, AGENT, x, y);
    int w = get_entity(env, WALLS, x, y);
    int b = get_entity(env, BOXES, x, y);
    int t = get_entity(env, TARGET, x, y);

    if (w) return c->wall;
    if (b && t) return c->box_on_target;
    if (b) return c->box;
    if (a) return c->agent;
    if (t) return c->target;

    return c->floor;
}

void draw_tile(Boxoban *env, int x, int y) {
      Client *c = env->client;
      Rectangle dest = {x * TILE, y * TILE, TILE, TILE};

      // Always lay down the base tile
      DrawTexturePro(
          c->floor,
          (Rectangle){0, 0, (float)c->floor.width, (float)c->floor.height},
          dest,
          (Vector2){0, 0},
          0.0f,
          WHITE);

      if (get_entity(env, TARGET, x, y)) {
          DrawTexturePro(
              c->target,
              (Rectangle){0, 0, (float)c->target.width, (float)c->target.height},
              dest,
              (Vector2){0, 0},
              0.0f,
              WHITE);
      }
      if (get_entity(env, BOXES, x, y)) {
          Texture2D tex = get_entity(env, TARGET, x, y) ? c->box_on_target : c->box;
          DrawTexturePro(
              tex,
              (Rectangle){0, 0, (float)tex.width, (float)tex.height},
              dest,
              (Vector2){0, 0},
              0.0f,
              WHITE);
      }
      if (get_entity(env, WALLS, x, y)) {
          DrawTexturePro(
              c->wall,
              (Rectangle){0, 0, (float)c->wall.width, (float)c->wall.height},
              dest,
              (Vector2){0, 0},
              0.0f,
              WHITE);
      }
      if (get_entity(env, AGENT, x, y)) {
          Rectangle src = {0, 0, c->agent.width / 2.0f, (float)c->agent.height};
          DrawTexturePro(c->agent, src, dest, (Vector2){0, 0}, 0.0f, WHITE);
      }
  }

// Required function. Should handle creating the client on first call
void puf_render(Boxoban* env) {
    if (!IsWindowReady()) {
        InitWindow(TILE*env->size, TILE*env->size, "PufferLib Boxoban");
        SetTargetFPS(60);
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    boxoban_human_controls(env);

    if (env->client == NULL) {
        env->client = make_client(env);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int y = 0; y < env->size; y++) {
        for (int x = 0; x < env->size; x++) {
            draw_tile(env, x, y);
        }
    }

    EndDrawing();
    puf_web_vsync();
}

// Required function. Should clean up anything you allocated
// Do not free observations, actions, rewards, terminals
void puf_close(Boxoban* env) {
    if (env->intermediate_rewards) {
          free(env->intermediate_rewards);
          env->intermediate_rewards = NULL;
      }
    if (IsWindowReady()) {
        if (env->client) {
            UnloadTexture(env->client->wall);
            UnloadTexture(env->client->box);
            UnloadTexture(env->client->target);
            UnloadTexture(env->client->floor);
            UnloadTexture(env->client->agent);
            free(env->client);
            env->client = NULL;
        }
        CloseWindow();
    }
}

// --- Native trainer (pufferl) API ---
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "targets_hit", log->on_targets);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    env->difficulty_id = dict_get(kwargs, "difficulty");
    env->size = 10;
    env->num_agents = 1;
    env->max_steps = dict_get(kwargs, "max_steps");
    env->int_r_coeff = dict_get(kwargs, "int_r_coeff");
    env->target_loss_pen_coeff = dict_get(kwargs, "target_loss_pen_coeff");
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    DictItem* map_item = dict_find(kwargs, "map_bin");
    if (map_item && map_item->str && map_item->str[0]) {
        if (boxoban_set_map_path(map_item->str) != 0) {
            fprintf(stderr, "Failed to set Boxoban map_bin %s\n", map_item->str);
            abort();
        }
    }
    init(env);
}

