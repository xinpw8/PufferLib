/* Ants: two colonies compete to forage food with pheromone trails. */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ACT_SIZES {4}
#define OBS_SIZE 27
#define NUM_ATNS 1
#define MAX_AGENTS 64

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720
#define NUM_COLONIES 2
#define MAX_FOOD_SOURCES 20
#define MAX_FOOD_PER_SOURCE 20
#define ANT_SPEED 5.0f
#define ANT_SIZE 4
#define FOOD_SIZE 6
#define COLONY_SIZE 20
#define TURN_ANGLE ((float)M_PI / 12.0f)
#define MIN_FOOD_COLONY_DISTANCE 50.0f
#define ANT_RESET_INTERVAL 2048

#define MAX_PHEROMONES 5000
#define PHEROMONE_DEPOSIT_AMOUNT 1.0f
#define PHEROMONE_EVAPORATION_RATE 0.005f
#define PHEROMONE_SIZE 2
#define PHEROMONE_DROP_INTERVAL 5

#define ANT_VISION_RANGE 75.0f
#define ANT_VISION_ANGLE ((float)M_PI / 3.0f)
#define ANT_PHEROMONE_RANGE 100.0f
#define ANT_PHEROMONE_ANGLE (2.0f * (float)M_PI)

#define ACTION_TURN_LEFT 0
#define ACTION_TURN_RIGHT 1
#define ACTION_MOVE_FORWARD 2
#define ACTION_NOOP 3

#define COLONY1_COLOR (Color){187, 0, 0, 255}
#define COLONY2_COLOR (Color){0, 187, 187, 255}
#define PHEROMONE1_COLOR (Color){187, 80, 80, 100}
#define PHEROMONE2_COLOR (Color){0, 160, 160, 100}
#define FOOD_COLOR (Color){0, 200, 0, 255}
#define BACKGROUND_COLOR (Color){6, 24, 24, 255}

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float avg_delivery_steps;
    float colony1_food;
    float colony2_food;
    float total_deliveries;
    float successful_trips;
    float total_resets;
    float n;
};

typedef struct Client Client;

typedef struct {
    float x, y;
} Vector2D;

typedef struct {
    Vector2D position;
    int amount;
} FoodSource;

typedef struct {
    Vector2D position;
    float strength;
    int colony_id;
    float direction;
} Pheromone;

typedef struct {
    Vector2D position;
    float direction;
    int colony_id;
    bool has_food;
    int steps_alive;
    int steps_since_pheromone;
    int steps_without_food;
} Ant;

typedef struct {
    Vector2D position;
    int food_collected;
} Colony;

struct Client {
    int cell_size;
    int width;
    int height;
    bool show_vision_cones;
    bool show_pheromone_range;
};

struct Env {
    Log log;
    Agent agents[MAX_AGENTS];
    int tag;
    int boundary_reached;
    int num_agents;
    unsigned int rng;
    Client* client;

    Ant ants[MAX_AGENTS];
    Colony colonies[NUM_COLONIES];
    FoodSource food_sources[MAX_FOOD_SOURCES];
    Pheromone pheromones[MAX_PHEROMONES];

    int width;
    int height;
    int num_food_sources;
    int num_pheromones;
    int tick;

    float reward_food_pickup;
    float reward_delivery;
};
typedef Env Ants;

static inline float random_float(unsigned int* rng, float minv, float maxv) {
    return minv + (maxv - minv) * ((float)rand_r(rng) / (float)RAND_MAX);
}

static inline float wrap_angle(float angle) {
    while (angle > (float)M_PI) angle -= 2.0f * (float)M_PI;
    while (angle < -(float)M_PI) angle += 2.0f * (float)M_PI;
    return angle;
}

static inline float distance_squared(Vector2D a, Vector2D b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return (dx * dx) + (dy * dy);
}

static inline float get_angle(Vector2D a, Vector2D b) {
    return atan2f(b.y - a.y, b.x - a.x);
}

static inline bool is_in_vision(Vector2D ant_pos, float ant_dir, Vector2D target) {
    float dx = target.x - ant_pos.x;
    float dy = target.y - ant_pos.y;
    float dist_sq = dx * dx + dy * dy;
    if (dist_sq > ANT_VISION_RANGE * ANT_VISION_RANGE) {
        return false;
    }
    float angle_to_target = atan2f(dy, dx);
    float angle_diff = wrap_angle(angle_to_target - ant_dir);
    return fabsf(angle_diff) <= ANT_VISION_ANGLE / 2.0f;
}

static inline bool is_in_pheromone_range(Vector2D ant_pos, Vector2D target) {
    return distance_squared(ant_pos, target)
        <= ANT_PHEROMONE_RANGE * ANT_PHEROMONE_RANGE;
}

static inline void add_pheromone(
        Ants* env, Vector2D position, int colony_id, float direction) {
    if (env->num_pheromones >= MAX_PHEROMONES) {
        for (int i = 0; i < env->num_pheromones - 1; i++) {
            env->pheromones[i] = env->pheromones[i + 1];
        }
        env->num_pheromones--;
    }
    env->pheromones[env->num_pheromones].position = position;
    env->pheromones[env->num_pheromones].strength = PHEROMONE_DEPOSIT_AMOUNT;
    env->pheromones[env->num_pheromones].colony_id = colony_id;
    env->pheromones[env->num_pheromones].direction = direction;
    env->num_pheromones++;
}

static int ants_steer(float heading, Vector2D from, Vector2D to) {
    float diff = wrap_angle(get_angle(from, to) - heading);
    if (diff > (float)M_PI / 8.0f) {
        return ACTION_TURN_RIGHT;
    }
    if (diff < -(float)M_PI / 8.0f) {
        return ACTION_TURN_LEFT;
    }
    return ACTION_MOVE_FORWARD;
}

// Forage nearest food, or return to colony while carrying. Write one agent's
// action. Call on a subset of agents to use as scripted opponents.
void ants_scripted_act(Ants* env, int agent) {
    Ant* ant = &env->ants[agent];
    if (ant->has_food) {
        env->agents[agent].actions[0] = ants_steer(
            ant->direction, ant->position,
            env->colonies[ant->colony_id].position);
        return;
    }

    float best = (float)env->width * (float)env->width;
    Vector2D food;
    food.x = 0.0f;
    food.y = 0.0f;
    int found = 0;
    for (int f = 0; f < env->num_food_sources; f++) {
        if (env->food_sources[f].amount <= 0) {
            continue;
        }
        float d = distance_squared(ant->position, env->food_sources[f].position);
        if (d >= best) {
            continue;
        }
        best = d;
        food = env->food_sources[f].position;
        found = 1;
    }
    env->agents[agent].actions[0] = found ?
        ants_steer(ant->direction, ant->position, food) : ACTION_MOVE_FORWARD;
}

void spawn_ant(Ants* env, int ant_id) {
    Ant* ant = &env->ants[ant_id];
    Colony* colony = &env->colonies[ant->colony_id];
    ant->position = colony->position;
    ant->direction = wrap_angle(
        (float)(rand_r(&env->rng) % 8) * ((float)M_PI / 4.0f));
    ant->has_food = false;
    ant->steps_alive = 0;
    ant->steps_since_pheromone = 0;
    ant->steps_without_food = 0;
}

void spawn_food(Ants* env) {
    for (int attempts = 0; attempts < 100; attempts++) {
        float x = random_float(&env->rng, 50.0f, (float)env->width - 50.0f);
        float y = random_float(&env->rng, 50.0f, (float)env->height - 50.0f);
        Vector2D pos;
        pos.x = x;
        pos.y = y;
        int near_colony = 0;
        for (int j = 0; j < NUM_COLONIES; j++) {
            if (distance_squared(pos, env->colonies[j].position)
                    < MIN_FOOD_COLONY_DISTANCE * MIN_FOOD_COLONY_DISTANCE) {
                near_colony = 1;
                break;
            }
        }
        if (near_colony) {
            continue;
        }
        for (int i = 0; i < MAX_FOOD_SOURCES; i++) {
            if (env->food_sources[i].amount != 0) {
                continue;
            }
            env->food_sources[i].position = pos;
            env->food_sources[i].amount = MAX_FOOD_PER_SOURCE;
            return;
        }
    }
}

void init(Ants* env) {
    env->tick = 0;
    env->client = NULL;
    env->num_pheromones = 0;
    env->colonies[0].position.x = (float)env->width / 4.0f;
    env->colonies[0].position.y = (float)env->height / 2.0f;
    env->colonies[1].position.x = 3.0f * (float)env->width / 4.0f;
    env->colonies[1].position.y = (float)env->height / 2.0f;
    env->colonies[0].food_collected = 0;
    env->colonies[1].food_collected = 0;
    env->num_food_sources = MAX_FOOD_SOURCES;
    for (int i = 0; i < env->num_food_sources; i++) {
        env->food_sources[i].amount = 0;
    }
}

void compute_observations(Ants* env) {
    typedef struct {
        Vector2D position;
        float strength;
        float direction;
    } PheromoneCandidate;

    float map_diag_sq = (float)env->width * (float)env->width
        + (float)env->height * (float)env->height;

    for (int a = 0; a < env->num_agents; a++) {
        Ant* ant = &env->ants[a];
        Colony* colony = &env->colonies[ant->colony_id];
        float* obs = env->agents[a].observations;
        int obs_idx = 0;

        float closest_food_dist_sq = map_diag_sq;
        Vector2D closest_food_pos;
        closest_food_pos.x = 0.0f;
        closest_food_pos.y = 0.0f;
        bool found_food = false;

        for (int i = 0; i < env->num_food_sources; i++) {
            if (env->food_sources[i].amount <= 0) {
                continue;
            }
            Vector2D food_pos = env->food_sources[i].position;
            if (!is_in_vision(ant->position, ant->direction, food_pos)) {
                continue;
            }
            float dist_sq = distance_squared(ant->position, food_pos);
            if (dist_sq >= closest_food_dist_sq) {
                continue;
            }
            closest_food_dist_sq = dist_sq;
            closest_food_pos = food_pos;
            found_food = true;
        }

        PheromoneCandidate candidates[100];
        int num_candidates = 0;
        for (int i = 0; i < env->num_pheromones; i++) {
            if (num_candidates >= 100) {
                break;
            }
            if (env->pheromones[i].colony_id != ant->colony_id) {
                continue;
            }
            if (!is_in_pheromone_range(ant->position, env->pheromones[i].position)) {
                continue;
            }
            candidates[num_candidates].position = env->pheromones[i].position;
            candidates[num_candidates].strength = env->pheromones[i].strength;
            candidates[num_candidates].direction = env->pheromones[i].direction;
            num_candidates++;
        }

        for (int i = 0; i < num_candidates - 1; i++) {
            for (int j = 0; j < num_candidates - i - 1; j++) {
                if (candidates[j].strength >= candidates[j + 1].strength) {
                    continue;
                }
                PheromoneCandidate temp = candidates[j];
                candidates[j] = candidates[j + 1];
                candidates[j + 1] = temp;
            }
        }

        int top_count = num_candidates < 5 ? num_candidates : 5;

        int friendly_ants_nearby = 0;
        for (int i = 0; i < env->num_agents; i++) {
            if (i == a || env->ants[i].colony_id != ant->colony_id) {
                continue;
            }
            if (!is_in_pheromone_range(ant->position, env->ants[i].position)) {
                continue;
            }
            friendly_ants_nearby++;
        }

        obs[obs_idx++] = (colony->position.x - ant->position.x) / (float)env->width;
        obs[obs_idx++] = (colony->position.y - ant->position.y) / (float)env->height;

        if (found_food) {
            obs[obs_idx++] = (closest_food_pos.x - ant->position.x)
                / (float)env->width;
            obs[obs_idx++] = (closest_food_pos.y - ant->position.y)
                / (float)env->height;
        } else {
            obs[obs_idx++] = 0.0f;
            obs[obs_idx++] = 0.0f;
        }

        for (int p = 0; p < 5; p++) {
            if (p >= top_count) {
                obs[obs_idx++] = 0.0f;
                obs[obs_idx++] = 0.0f;
                obs[obs_idx++] = 0.0f;
                obs[obs_idx++] = 0.0f;
                continue;
            }
            obs[obs_idx++] = (candidates[p].position.x - ant->position.x)
                / (float)env->width;
            obs[obs_idx++] = (candidates[p].position.y - ant->position.y)
                / (float)env->height;
            obs[obs_idx++] = candidates[p].direction / (float)M_PI;
            obs[obs_idx++] = candidates[p].strength / PHEROMONE_DEPOSIT_AMOUNT;
        }

        obs[obs_idx++] = ant->has_food ? 1.0f : 0.0f;
        obs[obs_idx++] = ant->direction / (2.0f * (float)M_PI);
        float max_friendly_ants = (float)(env->num_agents / NUM_COLONIES) - 1.0f;
        obs[obs_idx++] = max_friendly_ants > 0.0f
            ? (float)friendly_ants_nearby / max_friendly_ants : 0.0f;
    }
}

void puf_reset(Ants* env) {
    env->tick = 0;
    env->num_pheromones = 0;
    env->colonies[0].food_collected = 0;
    env->colonies[1].food_collected = 0;

    int ants_per_colony = env->num_agents / NUM_COLONIES;
    if (ants_per_colony < 1) {
        ants_per_colony = 1;
    }
    for (int i = 0; i < env->num_agents; i++) {
        env->ants[i].colony_id = i / ants_per_colony;
        if (env->ants[i].colony_id >= NUM_COLONIES) {
            env->ants[i].colony_id = NUM_COLONIES - 1;
        }
        spawn_ant(env, i);
        env->agents[i].rewards[0] = 0.0f;
        env->agents[i].terminals[0] = 0.0f;
    }

    for (int i = 0; i < env->num_food_sources; i++) {
        env->food_sources[i].amount = 0;
    }
    for (int i = 0; i < env->num_food_sources; i++) {
        spawn_food(env);
    }

    compute_observations(env);
}

void update_food_interactions(Ants* env) {
    float pickup_r = (ANT_SIZE + FOOD_SIZE) * (ANT_SIZE + FOOD_SIZE);
    float deliver_r = (ANT_SIZE + COLONY_SIZE) * (ANT_SIZE + COLONY_SIZE);

    for (int a = 0; a < env->num_agents; a++) {
        Ant* ant = &env->ants[a];
        if (!ant->has_food) {
            for (int f = 0; f < env->num_food_sources; f++) {
                if (env->food_sources[f].amount <= 0) {
                    continue;
                }
                if (distance_squared(ant->position, env->food_sources[f].position)
                        >= pickup_r) {
                    continue;
                }
                ant->has_food = true;
                env->food_sources[f].amount--;
                if (env->food_sources[f].amount <= 0) {
                    spawn_food(env);
                }
                env->agents[a].rewards[0] += env->reward_food_pickup;
                env->log.episode_return += env->reward_food_pickup;
                env->log.successful_trips += 1.0f;
                break;
            }
        }

        if (!ant->has_food) {
            continue;
        }
        Colony* colony = &env->colonies[ant->colony_id];
        if (distance_squared(ant->position, colony->position) >= deliver_r) {
            continue;
        }
        ant->has_food = false;
        colony->food_collected++;

        env->agents[a].rewards[0] += env->reward_delivery;
        env->log.episode_return += env->reward_delivery;
        env->log.episode_length += (float)ant->steps_alive;
        env->log.avg_delivery_steps += (float)ant->steps_alive;
        env->log.total_deliveries += 1.0f;
        if (ant->colony_id == 0) {
            env->log.colony1_food += 1.0f;
        } else {
            env->log.colony2_food += 1.0f;
        }
        float steps = fmaxf(1.0f, (float)ant->steps_alive);
        env->log.score += 1000.0f / steps;
        env->log.perf += fmaxf(0.0f,
            1.0f - (float)ant->steps_alive / (float)ANT_RESET_INTERVAL);
        env->log.n += 1.0f;

        ant->steps_alive = 0;
        ant->steps_without_food = 0;
    }
}

void puf_step(Ants* env) {
    env->tick++;

    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].rewards[0] = 0.0f;
        env->agents[i].terminals[0] = 0.0f;
    }

    for (int i = 0; i < env->num_agents; i++) {
        Ant* ant = &env->ants[i];
        ant->steps_alive++;
        if (!ant->has_food) {
            ant->steps_without_food++;
        }

        int action = (int)env->agents[i].actions[0];

        if (!ant->has_food && ant->steps_without_food > 100
                && (rand_r(&env->rng) % 100) < 5) {
            ant->direction += random_float(&env->rng, -1.0f, 1.0f)
                * TURN_ANGLE * 2.0f;
            ant->direction = wrap_angle(ant->direction);
        }

        switch (action) {
            case ACTION_TURN_LEFT:
                ant->direction = wrap_angle(ant->direction - TURN_ANGLE);
                break;
            case ACTION_TURN_RIGHT:
                ant->direction = wrap_angle(ant->direction + TURN_ANGLE);
                break;
            case ACTION_MOVE_FORWARD:
                ant->position.x += ANT_SPEED * cosf(ant->direction);
                ant->position.y += ANT_SPEED * sinf(ant->direction);
                break;
            case ACTION_NOOP:
            default:
                break;
        }

        if (ant->position.x < 0) ant->position.x = (float)env->width;
        if (ant->position.x > (float)env->width) ant->position.x = 0;
        if (ant->position.y < 0) ant->position.y = (float)env->height;
        if (ant->position.y > (float)env->height) ant->position.y = 0;

        if (ant->has_food) {
            ant->steps_since_pheromone++;
            if (ant->steps_since_pheromone >= PHEROMONE_DROP_INTERVAL) {
                add_pheromone(env, ant->position, ant->colony_id, ant->direction);
                ant->steps_since_pheromone = 0;
            }
        }

        if (ant->steps_alive % ANT_RESET_INTERVAL == 0) {
            spawn_ant(env, i);
            env->agents[i].terminals[0] = 1.0f;
            env->log.total_resets += 1.0f;
        }
    }

    for (int i = 0; i < env->num_pheromones; i++) {
        env->pheromones[i].strength -= PHEROMONE_EVAPORATION_RATE;
        if (env->pheromones[i].strength > 0.0f) {
            continue;
        }
        env->pheromones[i] = env->pheromones[env->num_pheromones - 1];
        env->num_pheromones--;
        i--;
    }

    update_food_interactions(env);
    compute_observations(env);
}

void puf_render(Ants* env) {
    if (env->client == NULL) {
        InitWindow(env->width, env->height, "PufferLib Ants");
        SetTargetFPS(60);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->cell_size = 1;
        env->client->width = env->width;
        env->client->height = env->height;
        env->client->show_vision_cones = true;
        env->client->show_pheromone_range = false;
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }
    if (IsKeyPressed(KEY_V)) {
        env->client->show_vision_cones = !env->client->show_vision_cones;
    }
    if (IsKeyPressed(KEY_P)) {
        env->client->show_pheromone_range = !env->client->show_pheromone_range;
    }

    BeginDrawing();
    ClearBackground(BACKGROUND_COLOR);

    for (int i = 0; i < NUM_COLONIES; i++) {
        Color color = (i == 0) ? COLONY1_COLOR : COLONY2_COLOR;
        DrawCircle((int)env->colonies[i].position.x,
            (int)env->colonies[i].position.y, COLONY_SIZE, color);
    }

    for (int i = 0; i < env->num_pheromones; i++) {
        Color pheromone_color = (env->pheromones[i].colony_id == 0)
            ? PHEROMONE1_COLOR : PHEROMONE2_COLOR;
        pheromone_color.a = (unsigned char)(100.0f * env->pheromones[i].strength);
        DrawCircle((int)env->pheromones[i].position.x,
            (int)env->pheromones[i].position.y, PHEROMONE_SIZE, pheromone_color);
    }

    for (int i = 0; i < env->num_food_sources; i++) {
        if (env->food_sources[i].amount <= 0) {
            continue;
        }
        DrawCircle((int)env->food_sources[i].position.x,
            (int)env->food_sources[i].position.y, FOOD_SIZE, FOOD_COLOR);
        DrawText(TextFormat("%d", env->food_sources[i].amount),
            (int)env->food_sources[i].position.x - 5,
            (int)env->food_sources[i].position.y - 5, 10, RAYWHITE);
    }

    for (int i = 0; i < env->num_agents; i++) {
        Ant* ant = &env->ants[i];
        Color ant_color = (ant->colony_id == 0) ? COLONY1_COLOR : COLONY2_COLOR;

        if (env->client->show_pheromone_range) {
            Color pheromone_range_color = ant_color;
            pheromone_range_color.a = 15;
            DrawCircle((int)ant->position.x, (int)ant->position.y,
                ANT_PHEROMONE_RANGE, pheromone_range_color);
        }

        if (env->client->show_vision_cones) {
            Color vision_color = ant_color;
            vision_color.a = 30;
            float start_angle = (ant->direction - ANT_VISION_ANGLE / 2.0f)
                * 180.0f / (float)M_PI;
            float end_angle = (ant->direction + ANT_VISION_ANGLE / 2.0f)
                * 180.0f / (float)M_PI;
            DrawCircleSector(
                (Vector2){ant->position.x, ant->position.y},
                ANT_VISION_RANGE, start_angle, end_angle, 32, vision_color);
        }

        if (ant->has_food) {
            ant_color = FOOD_COLOR;
        }

        DrawCircle((int)ant->position.x, (int)ant->position.y, ANT_SIZE, ant_color);

        float dir_x = ant->position.x + (ANT_SIZE * 1.5f) * cosf(ant->direction);
        float dir_y = ant->position.y + (ANT_SIZE * 1.5f) * sinf(ant->direction);
        DrawLine((int)ant->position.x, (int)ant->position.y,
            (int)dir_x, (int)dir_y, RAYWHITE);
    }

    DrawText(TextFormat("Colony 1: %d (%.1f%%)",
        env->colonies[0].food_collected,
        env->log.total_deliveries > 0.0f
            ? (env->log.colony1_food / env->log.total_deliveries * 100.0f) : 0.0f),
        20, 20, 20, COLONY1_COLOR);
    DrawText(TextFormat("Colony 2: %d (%.1f%%)",
        env->colonies[1].food_collected,
        env->log.total_deliveries > 0.0f
            ? (env->log.colony2_food / env->log.total_deliveries * 100.0f) : 0.0f),
        20, 50, 20, COLONY2_COLOR);
    DrawText(TextFormat("Efficiency: %.1f steps/food",
        env->log.n > 0.0f ? env->log.avg_delivery_steps / env->log.n : 0.0f),
        20, 80, 18, YELLOW);
    DrawText(TextFormat("Throughput: %.2f food/1000 steps",
        env->log.n > 0.0f ? env->log.score / env->log.n : 0.0f),
        20, 105, 18, YELLOW);

    float success_rate = env->log.total_resets > 0.0f
        ? (env->log.successful_trips / env->log.total_resets * 100.0f)
        : 0.0f;
    DrawText(TextFormat("Success Rate: %.1f%%", success_rate),
        20, 130, 18, GREEN);

    DrawText(TextFormat("Tick: %d", env->tick), env->width - 120, 20, 20, RAYWHITE);
    DrawText(TextFormat("Pheromones: %d", env->num_pheromones),
        env->width - 180, 50, 20, RAYWHITE);
    DrawText(TextFormat("Deliveries: %.0f", env->log.total_deliveries),
        env->width - 180, 75, 18, RAYWHITE);

    const char* vision_status = env->client->show_vision_cones ? "ON" : "OFF";
    const char* pheromone_status = env->client->show_pheromone_range ? "ON" : "OFF";
    DrawText(TextFormat("[V] Vision Cones: %s", vision_status),
        20, env->height - 30, 16, RAYWHITE);
    DrawText(TextFormat("[P] Pheromone Range: %s", pheromone_status),
        20, env->height - 50, 16, RAYWHITE);
    DrawText("[ESC] Exit", 20, env->height - 70, 16, GRAY);

    EndDrawing();
    puf_web_vsync();
}

void puf_close(Ants* env) {
    if (!env->client) {
        return;
    }
    CloseWindow();
    free(env->client);
    env->client = NULL;
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "avg_delivery_steps", log->avg_delivery_steps);
    dict_set(out, "colony1_food", log->colony1_food);
    dict_set(out, "colony2_food", log->colony2_food);
    dict_set(out, "total_deliveries", log->total_deliveries);
    dict_set(out, "successful_trips", log->successful_trips);
    dict_set(out, "total_resets", log->total_resets);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = dict_get(kwargs, "num_agents");
    env->width = dict_get(kwargs, "width");
    env->height = dict_get(kwargs, "height");
    env->reward_food_pickup = dict_get(kwargs, "reward_food_pickup");
    env->reward_delivery = dict_get(kwargs, "reward_delivery");
    if (env->num_agents < 1 || env->num_agents > MAX_AGENTS) {
        fprintf(stderr, "ants: num_agents must be in [1, %d]\n", MAX_AGENTS);
        exit(1);
    }
    if (env->width < 100 || env->height < 100) {
        fprintf(stderr, "ants: width/height too small\n");
        exit(1);
    }
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].policy = 0;
        env->agents[i].action_mask = NULL;
    }
    memset(&env->log, 0, sizeof(Log));
    init(env);
}
