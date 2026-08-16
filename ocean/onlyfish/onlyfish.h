/* Welcome to OnlyFish!
 *
 * This is a community event to celebrate 20k followers on X.
 * Your goal is to train a puffer that is fun to play with.
 * In the evaluation environment, you can click to drop a star
 * (star the github repo btw) to feed the puffer. The puffer can
 * also see the location of your cursor and of the nearest few
 * stars and puffers. You can change the reward signal, training
 * environment, and hyperparameters. Keep the model architecture
 * the same and make sure your obs/atn spaces match the eval env.
 * Usage:
 *   python setup.py build_c --inplace --force # Compile your env
 *   python -m pufferlib.pufferl train puffer_onlyfish
 *   python -m pufferlib.pufferl eval puffer_onlyfish --load-model-path latest
 *   python -m pufferlib.pufferl export puffer_onlyfish --load-model-path latest
 *   cp puffer_onlyfish_weights.bin resources/onlyfish/<your_username>.bin
 *   bash scripts/build_ocean onlyfish local # Test your model in C
 *
 *  PR your .bin to:
 *    https://github.com/PufferAI/PufferLib/tree/3.0/pufferlib/resources/onlyfish/your_name.bin
 *
 *  You can also PR a a link to your fork at pufferai/onlyfish.ai
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define WIDTH 1280
#define HEIGHT 720
#define NUM_GOALS 32 
#define GOAL_SPAWNS 8
#define GOAL_OBS 4
#define AGENT_OBS 4
#define ACT_SIZES {9, 5}
#define OBS_SIZE (2*GOAL_OBS + 2*AGENT_OBS + 5)
#define NUM_ATNS 2
#define MAX_AGENTS 64

typedef Env OnlyFish;

#define STAR_SIZE 128 
#define PUFFER_SIZE 128 
#define MOUSE_SPEED 20

float clip(float val, float min, float max) {
    if (val < min) {
        return min;
    } else if (val > max) {
        return max;
    }
    return val;
}

struct pair {
    float val;
    int idx;
};

static int cmp(const void *a, const void *b) {
    const struct pair *pa = (const struct pair*)a;
    const struct pair *pb = (const struct pair*)b;
    if (pa->val < pb->val) return -1;
    if (pa->val > pb->val) return 1;
    return 0;
}

void argsort(float *dists, int *idxs, int n) {
    struct pair pairs[n];
    for (int i = 0; i < n; i++) {
        pairs[i].val = dists[i];
        pairs[i].idx = i;
    }
    qsort(pairs, n, sizeof(struct pair), cmp);
    for (int i = 0; i < n; i++) {
        idxs[i] = pairs[i].idx;
    }
}

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct {
    Texture2D spritesheet;
} Client;

typedef struct {
    float x;
    float y;
    float heading;
    float speed;
    int ticks_since_reward;
} Fish;

typedef struct {
    float x;
    float y;
    bool active;
} Goal;

struct Env {
    Log log;
    Client* client;
    Agent agents[MAX_AGENTS];
    Fish* fish;
    Goal* goals;
    int width;
    int height;
    int num_agents;
    int tag;
    int boundary_reached;
    int num_goals;
    float mouse_x;
    float mouse_y;
    float mouse_heading;
    char** names;
    unsigned int rng;
};

void init(OnlyFish* env) {
    if (env->names == NULL) {
        env->names = (char**)calloc(env->num_agents, sizeof(char*));
        for (int i=0; i<env->num_agents; i++) {
            char name[100];
            sprintf(name, "Agent %d", i);
            env->names[i] = strdup(name);
        }
    }
    env->num_goals = NUM_GOALS;
    env->width = WIDTH;
    env->height = HEIGHT;
    env->fish = (Fish*)calloc(env->num_agents, sizeof(Fish));
    env->goals = (Goal*)calloc(env->num_goals, sizeof(Goal));
}

void update_goals(OnlyFish* env) {
    for (int a=0; a<env->num_agents; a++) {
        Fish* agent = &env->fish[a];
        for (int g=0; g<env->num_goals; g++) {
            if (!env->goals[g].active) {
                continue;
            }
            Goal* goal = &env->goals[g];
            goal->y += HEIGHT/(float)600;
            goal->y = clip(goal->y, 0, HEIGHT - STAR_SIZE/2);
            goal->x = clip(goal->x, STAR_SIZE/2, env->width - STAR_SIZE/2);

            float dx = (goal->x - agent->x);
            float dy = (goal->y - agent->y);
            float dist = sqrt(dx*dx + dy*dy);
            if (dist > 64) {
                continue;
            }
            goal->active = false;

            // Right now, you just get a reward of 1 for eating a star
            env->agents[a].rewards[0] = 1.0f;
            env->log.perf += 1.0f;
            env->log.score += 1.0f;
            env->log.episode_length += agent->ticks_since_reward;
            agent->ticks_since_reward = 0;
            env->log.episode_return += 1.0f;
            env->log.n++;
        }
    }
}

void spawn_goal(OnlyFish* env, int x, int y) {
    for (int g=0; g<env->num_goals; g++) {
        Goal* goal = &env->goals[g];
        if (goal->active) {
            continue;
        }
        goal->active = true;
        goal->x = x;
        goal->y = y;
        break;
    }
}

void compute_observations(OnlyFish* env) {
    for (int i=0; i<env->num_agents; i++) {
        Fish* agent = &env->fish[i];
        obs_t* obs = env->agents[i].observations;
        int obs_idx = 0;
        float x = agent->x;
        float y = agent->y;

        float goal_dists[env->num_goals];
        int goal_idx[env->num_goals];
        int nearby_goals = 0;
        for (int j=0; j<env->num_goals; j++) {
            Goal* goal = &env->goals[j];
            float dist = FLT_MAX;
            if (goal->active) {
                float dx = goal->x - x;
                float dy = goal->y - y;
                dist = sqrt(dx*dx + dy*dy);
                nearby_goals++;
            }
            goal_dists[j] = dist;
            goal_idx[j] = j;
        }
        argsort((float*)goal_dists, (int*)goal_idx, env->num_goals);
        for (int j=0; j<GOAL_OBS; j++) {
            if (j >= nearby_goals) {
                obs[obs_idx++] = 0;
                obs[obs_idx++] = 0;
            } else {
                Goal* goal = &env->goals[goal_idx[j]];
                obs[obs_idx++] = (goal->x - agent->x)/env->width;
                obs[obs_idx++] = (goal->y - agent->y)/env->height;
            }
        }

        float agent_dists[env->num_agents];
        int agent_idx[env->num_agents];
        for (int j=0; j<env->num_agents; j++) {
            Fish* other = &env->fish[j];
            float dx = other->x - x;
            float dy = other->y - y;
            agent_dists[j] = sqrt(dx*dx + dy*dy);
            agent_idx[j] = j;
        }
        argsort((float*)agent_dists, (int*)agent_idx, env->num_agents);
        for (int j=0; j<AGENT_OBS; j++) {
            if (j >= env->num_agents) {
                obs[obs_idx++] = 0;
                obs[obs_idx++] = 0;
            } else {
                Fish* other = &env->fish[agent_idx[j]];
                obs[obs_idx++] = (other->x - agent->x)/env->width;
                obs[obs_idx++] = (other->y - agent->y)/env->height;
            }
        }

        obs[obs_idx++] = agent->heading/(2*PI);
        obs[obs_idx++] = agent->x/env->width;
        obs[obs_idx++] = agent->y/env->height;
        obs[obs_idx++] = env->mouse_x/env->width;
        obs[obs_idx++] = env->mouse_y/env->height;
    }
}

void puf_reset(OnlyFish* env) {
    for (int i=0; i<env->num_agents; i++) {
        env->fish[i].x = rand() % env->width;
        env->fish[i].y = rand() % env->height;
        env->fish[i].ticks_since_reward = 0;
    }
    for (int i=0; i<env->num_goals; i++) {
        env->goals[i].active = false;
    }
    compute_observations(env);
}

void puf_step(OnlyFish* env) {
    for (int i=0; i<env->num_agents; i++) {
        env->agents[i].rewards[0] = 0;
        env->agents[i].terminals[0] = 0;
        Fish* agent = &env->fish[i];
        agent->ticks_since_reward += 1;

        agent->heading += (env->agents[i].actions[0] - 4.0f)/12.0f;
        agent->heading = clip(agent->heading, 0, 2*PI);

        agent->speed += 0.10f*(env->agents[i].actions[1] - 2.0f);
        agent->speed = clip(agent->speed, -10.0f, 10.0f);

        agent->x += agent->speed*cosf(agent->heading);
        agent->x = clip(agent->x, PUFFER_SIZE/2, env->width - PUFFER_SIZE/2);

        agent->y += agent->speed*sinf(agent->heading);
        agent->y = clip(agent->y, PUFFER_SIZE/2, env->height - PUFFER_SIZE/2);

        if (env->client == NULL) {
            if (agent->ticks_since_reward % 512 == 0) {
                env->fish[i].x = rand() % env->width;
                env->fish[i].y = rand() % env->height;
            }
            if (rand() % 10 == 0) {
                int x = rand() % env->width;
                int y = rand() % env->height;
                spawn_goal(env, x, y);
            }
            env->mouse_x += MOUSE_SPEED*cosf(env->mouse_heading);
            env->mouse_y += MOUSE_SPEED*sinf(env->mouse_heading);
            env->mouse_x = clip(env->mouse_x, 0, env->width);
            env->mouse_y = clip(env->mouse_y, 0, env->height);
            if (rand() % 180 == 0) {
                env->mouse_heading = (rand() % 360)*PI/180.0f;
            }
        }
    }
    update_goals(env);
    compute_observations(env);
}

void puf_render(OnlyFish* env) {
    if (env->client == NULL) {
        InitWindow(env->width, env->height, "PufferLib OnlyFish");
        SetTargetFPS(60);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->spritesheet = LoadTexture("resources/shared/puffers.png");
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    env->mouse_x = GetMouseX();
    env->mouse_y = GetMouseY();
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        spawn_goal(env, env->mouse_x, env->mouse_y);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int i=0; i<env->num_goals; i++) {
        Goal* goal = &env->goals[i];
        if (!goal->active) {
            continue;
        }
        DrawTexturePro(
            env->client->spritesheet,
            (Rectangle){
                384, 256, 128, 128,
            },
            (Rectangle){
                goal->x - STAR_SIZE/2,
                goal->y - STAR_SIZE/2,
                STAR_SIZE,
                STAR_SIZE
            },
            (Vector2){0, 0},
            0,
            WHITE
        );
    }

    for (int i=0; i<env->num_agents; i++) {
        Fish* agent = &env->fish[i];
        float heading = agent->heading;
        DrawTexturePro(
            env->client->spritesheet,
            (Rectangle){
                0, 
                (heading < PI/2 || heading > 3*PI/2) ? 0: 128,
                128, 128,
            },
            (Rectangle){
                agent->x - PUFFER_SIZE/2,
                agent->y - PUFFER_SIZE/2,
                PUFFER_SIZE,
                PUFFER_SIZE
            },
            (Vector2){0, 0},
            0,
            WHITE
        );
        int text_width = MeasureText(env->names[i], 14);
        DrawText(env->names[i], agent->x - text_width/2, agent->y + PUFFER_SIZE/2, 14, WHITE);
    }

    EndDrawing();
}

void puf_close(OnlyFish* env) {
    free(env->fish);
    free(env->goals);
    free(env->names);
    if (env->client != NULL) {
        Client* client = env->client;
        UnloadTexture(client->spritesheet);
        CloseWindow();
        free(client);
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents");
    if (env->num_agents > MAX_AGENTS) {
        fprintf(stderr, "onlyfish: num_agents too large\n");
        exit(1);
    }
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
    init(env);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}

