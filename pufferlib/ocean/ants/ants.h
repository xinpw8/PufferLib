#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>
#include "raylib.h"

// Constants for the simulation
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define MAX_ANTS_PER_COLONY 50
#define NUM_COLONIES 2
#define MAX_FOOD_SOURCES 20
#define MAX_FOOD_PER_SOURCE 100
#define ANT_SPEED 3.0f
#define ANT_SIZE 4
#define FOOD_SIZE 6
#define COLONY_SIZE 30
#define PHEROMONE_EVAPORATION_RATE 0.00005f
#define PHEROMONE_DEPOSIT_AMOUNT 1.5f
#define MAX_PHEROMONES 5000
#define PHEROMONE_SIZE 4
#define ANT_VISION_RANGE 50.0f
#define ANT_VISION_ANGLE (M_PI / 2)
#define TURN_ANGLE (M_PI / 10)
#define MIN_FOOD_COLONY_DISTANCE 100.0f
#define MAX_TRACE_WAYPOINTS 500 // Max waypoints for text tracing (Increased from 300)
#define PAINTBRUSH_RENDER_SIZE 75.0f // New: Size for rendering paintbrush

// Define structure for waypoints that includes pen state
typedef struct {
    int id;
    int colony_id; // 0 for main colony (red), 1 for secondary/tracing (cyan)
    Vector2 position;
    float direction; // Radians
    bool has_food;
    bool pen_down; // True if the ant should drop pheromones at this waypoint
    // This could also be used to control "jump" segments in a path by setting pen_down = false
    // for the waypoint *before* the jump, and then the next waypoint is the jump target.
} TraceWaypoint;

// Actions
#define ACTION_MOVE_FORWARD 0
#define ACTION_TURN_LEFT 1
#define ACTION_TURN_RIGHT 2
#define ACTION_HALT 3 // New: Action to make the ant stop for a tick
// #define ACTION_DROP_PHEROMONE 3 // This action is now handled by a separate flag

// Colors
#define PHEROMONE1_COLOR (Color){149, 42, 42, 80}     // Dire pheromone (matching red)
#define PHEROMONE2_COLOR (Color){53, 175, 175, 80}    // Radiant pheromone (matching cyan)
#define COLONY1_COLOR (Color){149, 42, 42, 255}       // Dire Creep (matching red puffer)
#define COLONY2_COLOR (Color){53, 175, 175, 255}      // Radiant Creep (matching cyan puffer)
#define FOOD_COLOR (Color){0, 187, 0, 255}
#define BACKGROUND_COLOR (Color){0, 0, 0, 255}

// Required Log struct for PufferLib - IDENTICAL TO SNAKE PATTERN
typedef struct Log Log;
struct Log {
    float perf;              // Performance metric
    float score;             // Total score
    float episode_return;    // Cumulative rewards
    float episode_length;    // Episode duration
    float n;                 // Episode count - REQUIRED AS LAST FIELD
};

// Forward declarations
typedef struct Client Client;
typedef struct AntsEnv AntsEnv;

// Environment structs
typedef struct {
    Vector2 position;
    int amount;
} FoodSource;

typedef struct {
    Vector2 position;
    float strength;
    int colony_id;
} Pheromone;

typedef struct {
    Vector2 position;
    float direction;
    float visual_direction;  // For smooth rotation rendering
    int colony_id;
    bool has_food;
    int lifetime;            // Track ant lifetime for performance metrics
    bool has_paintbrush;     // New: For paintbrush feature
} Ant;

typedef struct {
    Vector2 position;
    int food_collected;
} Colony;

// Raylib client structure - FOLLOWING SNAKE PATTERN
struct Client {
    int cell_size;
    int width;
    int height;
    Texture2D ant;  // Red/cyan puffer texture
    Texture2D colony_base;  // Colony base texture
    Texture2D paintbrush_texture; // New: For paintbrush item
};

// Main environment struct - RESTRUCTURED FOLLOWING SNAKE PATTERN
struct AntsEnv {
    // Required PufferLib fields - IDENTICAL TO SNAKE
    float* observations;        // Flattened observations for all ants
    int* actions;              // Actions for all ants (move/turn)
    bool* ant_is_dropping_pheromone; // New: Separate flag for pheromone dropping
    float* rewards;            // Rewards for all ants
    unsigned char* terminals;   // Terminal flags
    Log log;                   // Main aggregated log
    Log* ant_logs;             // Individual ant logs - CRITICAL ADDITION

    // Debug
    bool debug_pheromone_printed;
    int debug_pheromone_count;
    
    // Environment state
    Colony colonies[NUM_COLONIES];
    Ant* ants;                 // Dynamic array of all ants
    FoodSource food_sources[MAX_FOOD_SOURCES];
    Pheromone pheromones[MAX_PHEROMONES];
    int num_pheromones;
    int num_food_sources;
    
    // Environment parameters
    int num_ants;              // Total number of ants
    int width;                 // Environment width
    int height;                // Environment height
    int obs_size;              // Observation size per ant
    int tick;                  // Current timestep
    
    // Reward parameters
    float reward_food;
    float reward_delivery;
    float reward_death;
    
    // Rendering
    Client* client;            // Raylib client
    int cell_size;

    // Text Tracing Variables
    bool is_tracing_text;
    int current_trace_waypoint_index;
    TraceWaypoint trace_waypoints[MAX_TRACE_WAYPOINTS];
    int num_trace_waypoints;
    bool tracing_complete; // New: Flag to indicate tracing is done

    // Paintbrush Feature Variables
    FoodSource paintbrush_food;       // New: Location and state of the paintbrush item
    bool paintbrush_spawned;          // New: If the paintbrush is currently in the world
    bool currently_writing_with_paintbrush; // New: If the ant is in the special "Puffer" writing mode
};

/**
 * Add an ant's log to the main log when the ant's episode ends.
 * CRITICAL FUNCTION - COPIED FROM SNAKE PATTERN
 * This should only be called during termination conditions for a specific ant.
 * Accumulates the ant's stats into the main log and resets the ant's individual log.
 */
void add_log(AntsEnv* env, int ant_id) {
    env->log.perf += env->ant_logs[ant_id].perf;
    env->log.score += env->ant_logs[ant_id].score;
    env->log.episode_return += env->ant_logs[ant_id].episode_return;
    env->log.episode_length += env->ant_logs[ant_id].episode_length;
    env->log.n += 1;
    
    // Reset individual ant log
    env->ant_logs[ant_id] = (Log){0};
}

// Memory management functions - FOLLOWING SNAKE PATTERN
void init_ants_env(AntsEnv* env) {
    env->ants = (Ant*)calloc(env->num_ants, sizeof(Ant));
    env->ant_logs = (Log*)calloc(env->num_ants, sizeof(Log));
    env->tick = 0;
    env->client = NULL;
    env->num_pheromones = 0;
    env->debug_pheromone_printed = false;
    env->debug_pheromone_count = 0;
    // Initialize food sources
    env->num_food_sources = MAX_FOOD_SOURCES;
    for (int i = 0; i < env->num_food_sources; i++) {
        env->food_sources[i].amount = 0; // Will be set in reset
    }
    
    // Initialize colonies
    env->colonies[0].position = (Vector2){env->width / 4, env->height / 2};
    env->colonies[1].position = (Vector2){3 * env->width / 4, env->height / 2};
    env->colonies[0].food_collected = 0;
    env->colonies[1].food_collected = 0;

    // Initialize tracing variables
    env->is_tracing_text = false;
    env->current_trace_waypoint_index = 0;
    env->num_trace_waypoints = 0;
    env->tracing_complete = false; // Initialize new flag

    // Initialize paintbrush feature variables
    env->paintbrush_spawned = false; // Will be spawned in c_reset
    env->paintbrush_food.amount = 0;
    env->currently_writing_with_paintbrush = false;
}

void allocate_ants_env(AntsEnv* env) {
    env->obs_size = 12; // Fixed observation size per ant
    env->observations = (float*)calloc(env->num_ants * env->obs_size, sizeof(float));
    env->actions = (int*)calloc(env->num_ants, sizeof(int));
    env->ant_is_dropping_pheromone = (bool*)calloc(env->num_ants, sizeof(bool)); // Allocate new flag array
    env->rewards = (float*)calloc(env->num_ants, sizeof(float));
    env->terminals = (unsigned char*)calloc(env->num_ants, sizeof(unsigned char));
    init_ants_env(env);
}

void c_close(AntsEnv* env) {
    if (env->ants) {
        free(env->ants);
        env->ants = NULL;
    }
    if (env->ant_logs) {
        free(env->ant_logs);
        env->ant_logs = NULL;
    }
}

void free_ants_env(AntsEnv* env) {
    c_close(env);
    if (env->observations) {
        free(env->observations);
        env->observations = NULL;
    }
    if (env->actions) {
        free(env->actions);
        env->actions = NULL;
    }
    if (env->rewards) {
        free(env->rewards);
        env->rewards = NULL;
    }
    if (env->terminals) {
        free(env->terminals);
        env->terminals = NULL;
    }
    if (env->ant_is_dropping_pheromone) { // Free new flag array
        free(env->ant_is_dropping_pheromone);
        env->ant_is_dropping_pheromone = NULL;
    }
}

// Helper function implementations
static inline float random_float(float min, float max) {
    return min + (max - min) * ((float)rand() / (float)RAND_MAX);
}

static inline float wrap_angle(float angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

static inline float distance_squared(Vector2 a, Vector2 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

static inline bool is_in_vision(Vector2 ant_pos, float ant_dir, Vector2 target) {
    float dx = target.x - ant_pos.x;
    float dy = target.y - ant_pos.y;
    float dist_sq = dx * dx + dy * dy;
    
    if (dist_sq > ANT_VISION_RANGE * ANT_VISION_RANGE) {
        return false;
    }
    
    float target_angle = atan2(dy, dx);
    float angle_diff = wrap_angle(target_angle - ant_dir);
    
    return fabs(angle_diff) <= ANT_VISION_ANGLE / 2;
}

static inline void add_pheromone(AntsEnv* env, Vector2 position, int colony_id) {
    if (env->num_pheromones >= MAX_PHEROMONES) {
        // Replace oldest pheromone
        for (int i = 0; i < env->num_pheromones - 1; i++) {
            env->pheromones[i] = env->pheromones[i + 1];
        }
        env->num_pheromones--;
    }
    
    env->pheromones[env->num_pheromones].position = position;
    env->pheromones[env->num_pheromones].strength = PHEROMONE_DEPOSIT_AMOUNT;
    env->pheromones[env->num_pheromones].colony_id = colony_id;
    env->num_pheromones++;
}

void get_observation_for_ant(AntsEnv* env, int ant_idx, float* obs) {
    Ant* ant = &env->ants[ant_idx];
    Colony* colony = &env->colonies[ant->colony_id];
    
    // Observation structure (12 elements):
    // [0-1]: ant position (normalized)
    // [2]: ant direction (normalized to 0-1)
    // [3]: has_food (0 or 1)
    // [4-5]: relative position to colony
    // [6-7]: closest food direction
    // [8-9]: strongest pheromone direction
    // [10]: closest food distance (normalized)
    // [11]: strongest pheromone strength
    
    obs[0] = ant->position.x / env->width;
    obs[1] = ant->position.y / env->height;
    obs[2] = (ant->direction + M_PI) / (2 * M_PI);
    obs[3] = ant->has_food ? 1.0f : 0.0f;
    
    // Relative position to colony
    obs[4] = (colony->position.x - ant->position.x) / env->width;
    obs[5] = (colony->position.y - ant->position.y) / env->height;
    
    // Find closest visible food
    float closest_food_dist_sq = env->width * env->width; // Initialize with a large value
    Vector2 closest_food_dir = {0, 0};
    for (int i = 0; i < env->num_food_sources; i++) {
        if (env->food_sources[i].amount > 0) {
            float dist_sq = distance_squared(ant->position, env->food_sources[i].position);
            if (dist_sq < closest_food_dist_sq && is_in_vision(ant->position, ant->direction, env->food_sources[i].position)) {
                closest_food_dist_sq = dist_sq;
                closest_food_dir.x = env->food_sources[i].position.x - ant->position.x;
                closest_food_dir.y = env->food_sources[i].position.y - ant->position.y;
            }
        }
    }
    
    obs[6] = closest_food_dir.x / env->width;
    obs[7] = closest_food_dir.y / env->height;
    obs[10] = sqrt(closest_food_dist_sq) / sqrt(env->width * env->width + env->height * env->height);
    
    // Find strongest visible pheromone
    float strongest_pheromone = 0;
    Vector2 pheromone_dir = {0, 0};
    for (int i = 0; i < env->num_pheromones; i++) {
        if (env->pheromones[i].colony_id == ant->colony_id) { // Only sense own colony's pheromones (or matching type)
            float dist_sq = distance_squared(ant->position, env->pheromones[i].position);
            if (dist_sq <= ANT_VISION_RANGE * ANT_VISION_RANGE && 
                is_in_vision(ant->position, ant->direction, env->pheromones[i].position)) {
                float strength = env->pheromones[i].strength / (sqrt(dist_sq) + 1);
                if (strength > strongest_pheromone) {
                    strongest_pheromone = strength;
                    pheromone_dir.x = env->pheromones[i].position.x - ant->position.x;
                    pheromone_dir.y = env->pheromones[i].position.y - ant->position.y;
                }
            }
        }
    }
    
    obs[8] = pheromone_dir.x / env->width;
    obs[9] = pheromone_dir.y / env->height;
    obs[11] = strongest_pheromone;
}

void compute_observations(AntsEnv* env) {
    for (int i = 0; i < env->num_ants; i++) {
        get_observation_for_ant(env, i, &env->observations[i * env->obs_size]);
    }
}

void spawn_ant(AntsEnv* env, int ant_id) {
    Ant* ant = &env->ants[ant_id];
    Colony* colony = &env->colonies[ant->colony_id];
    
    ant->position = colony->position;
    ant->direction = random_float(0, 2 * M_PI);
    ant->visual_direction = ant->direction;  // Initialize visual direction
    ant->has_food = false;
    ant->lifetime = 0;
    
    // Reset individual ant log
    env->ant_logs[ant_id] = (Log){0};
}

void spawn_food(AntsEnv* env) {
    int idx;
    bool valid_position;
    int attempts = 0;
    
    do {
        float x = random_float(50, env->width - 50);
        float y = random_float(50, env->height - 50);
        
        valid_position = true;
        for (int j = 0; j < NUM_COLONIES; j++) {
            float dist_sq = distance_squared((Vector2){x, y}, env->colonies[j].position);
            if (dist_sq < MIN_FOOD_COLONY_DISTANCE * MIN_FOOD_COLONY_DISTANCE) {
                valid_position = false;
                break;
            }
        }
        
        if (valid_position) {
            // Find an empty food source slot
            for (idx = 0; idx < env->num_food_sources; idx++) {
                if (env->food_sources[idx].amount == 0) {
                    env->food_sources[idx].position.x = x;
                    env->food_sources[idx].position.y = y;
                    env->food_sources[idx].amount = MAX_FOOD_PER_SOURCE;
                    return;
                }
            }
        }
        attempts++;
    } while (!valid_position && attempts < 100);
}


void spawn_paintbrush_food(AntsEnv* env) {
    printf("spawn_paintbrush_food\n");
    if (env->paintbrush_spawned) return; // Only spawn if not already present

    int attempts = 0;
    bool valid_position;
    float x, y;

    do {
        x = random_float(env->width * 0.2f, env->width * 0.8f);  // More central spawn
        y = random_float(env->height * 0.2f, env->height * 0.8f);
        valid_position = true;

        // Ensure it's not too close to colonies
        for (int j = 0; j < NUM_COLONIES; j++) {
            float dist_sq = distance_squared((Vector2){x, y}, env->colonies[j].position);
            if (dist_sq < MIN_FOOD_COLONY_DISTANCE * MIN_FOOD_COLONY_DISTANCE) {
                valid_position = false;
                break;
            }
        }

        // Ensure it's not on top of existing food sources
        if (valid_position) {
            for (int i = 0; i < env->num_food_sources; i++) {
                if (env->food_sources[i].amount > 0) {
                    float dist_sq = distance_squared((Vector2){x, y}, env->food_sources[i].position);
                    // Approx size of paintbrush for collision check
                    if (dist_sq < (FOOD_SIZE + 10.0f) * (FOOD_SIZE + 10.0f)) { 
                        valid_position = false;
                        break;
                    }
                }
            }
        }
        attempts++;
    } while (!valid_position && attempts < 100);

    if (valid_position) {
        env->paintbrush_food.position = (Vector2){x, y};
        env->paintbrush_food.amount = 1; // Represents a single paintbrush item
        env->paintbrush_spawned = true;
        printf("Paintbrush spawned at (%f, %f)\n", x, y);
        // No need to log here, it's just spawning
    }
}

// Waypoint helper functions for text tracing
void add_waypoint(AntsEnv* env, float x, float y, bool pen_down) {
    if (env->num_trace_waypoints < MAX_TRACE_WAYPOINTS) {
        env->trace_waypoints[env->num_trace_waypoints].position.x = x;
        env->trace_waypoints[env->num_trace_waypoints].position.y = y;
        env->trace_waypoints[env->num_trace_waypoints].pen_down = pen_down;
        env->num_trace_waypoints++;
    }
}

// Defines the path for "Puffer 3.0" - triggered by paintbrush
void setup_puffer_3_0_trace_path(AntsEnv* env) {
    env->num_trace_waypoints = 0; // Reset previous path

    float base_x = env->width * 0.1f;
    float base_y = env->height * 0.3f;
    float char_h = 56.0f;   // Was 80.0f
    float char_w = 28.0f;   // Was 40.0f
    float spacing = 14.0f;  // Was 20.0f
    float current_x = base_x;
    float dot_square_size = 5.6f;  // Was 8.0f

    // P
    add_waypoint(env, current_x, base_y + char_h, false); 
    add_waypoint(env, current_x, base_y + char_h, true);  
    add_waypoint(env, current_x, base_y, true);           
    add_waypoint(env, current_x + char_w * 0.75f, base_y, true); 
    add_waypoint(env, current_x + char_w, base_y + char_h * 0.25f, true);
    add_waypoint(env, current_x + char_w * 0.75f, base_y + char_h * 0.5f, true);
    add_waypoint(env, current_x, base_y + char_h * 0.5f, true); 
    current_x += char_w + spacing;

    // u
    add_waypoint(env, current_x, base_y, false); 
    add_waypoint(env, current_x, base_y, true); 
    add_waypoint(env, current_x, base_y + char_h, true); 
    add_waypoint(env, current_x + char_w, base_y + char_h, true); 
    add_waypoint(env, current_x + char_w, base_y, true); 
    current_x += char_w + spacing;

    // f (lowercase f for Puffer 3.0)
    {
        float fx = current_x + char_w * 0.5f; // stem at center
        float top_bar_left = fx - char_w * 0.4f;
        float top_bar_right = fx + char_w * 0.4f;
        float mid_y = base_y + char_h * 0.5f;

        // Vertical stem (bottom to top)
        add_waypoint(env, fx, base_y + char_h, false);
        add_waypoint(env, fx, base_y, true);

        // Top crossbar (at base_y)
        add_waypoint(env, top_bar_left, base_y, false);
        add_waypoint(env, top_bar_right, base_y, true);
        
        // Middle crossbar (from stem to right)
        add_waypoint(env, fx, mid_y, false); // Move to stem at mid_y
        add_waypoint(env, fx + char_w * 0.4f, mid_y, true); // Draw to the right
    }
    current_x += char_w + spacing;

    // f (second lowercase f for Puffer 3.0)
    {
        float fx = current_x + char_w * 0.5f; // stem at center
        float top_bar_left = fx - char_w * 0.4f;
        float top_bar_right = fx + char_w * 0.4f;
        float mid_y = base_y + char_h * 0.5f;

        // Vertical stem (bottom to top)
        add_waypoint(env, fx, base_y + char_h, false);
        add_waypoint(env, fx, base_y, true);

        // Top crossbar (at base_y)
        add_waypoint(env, top_bar_left, base_y, false);
        add_waypoint(env, top_bar_right, base_y, true);
        
        // Middle crossbar (from stem to right)
        add_waypoint(env, fx, mid_y, false);
        add_waypoint(env, fx + char_w * 0.4f, mid_y, true);
    }
    current_x += char_w + spacing;
    
    // e (Mirrored for Puffer)
    add_waypoint(env, current_x, base_y + char_h * 0.5f, false); 
    add_waypoint(env, current_x, base_y + char_h * 0.5f, true); 
    add_waypoint(env, current_x + char_w, base_y + char_h * 0.5f, true); 
    add_waypoint(env, current_x + char_w, base_y, true); 
    add_waypoint(env, current_x, base_y, true); 
    add_waypoint(env, current_x, base_y + char_h, true); 
    add_waypoint(env, current_x + char_w, base_y + char_h, true); 
    current_x += char_w + spacing;

    // r
    add_waypoint(env, current_x, base_y + char_h, false);
    add_waypoint(env, current_x, base_y + char_h, true); 
    add_waypoint(env, current_x, base_y, true);         
    add_waypoint(env, current_x + char_w * 0.75f, base_y, true); 
    add_waypoint(env, current_x + char_w, base_y + char_h * 0.25f, true); 
    current_x += char_w; // No spacing after r, space is next char

    // Space before 3.0
    current_x += spacing * 1.5f; // Increased spacing

    // 3
    add_waypoint(env, current_x, base_y, false);
    add_waypoint(env, current_x, base_y, true); 
    add_waypoint(env, current_x + char_w, base_y, true); 
    add_waypoint(env, current_x + char_w, base_y + char_h * 0.5f, true); 
    add_waypoint(env, current_x, base_y + char_h * 0.5f, true); 
    add_waypoint(env, current_x + char_w, base_y + char_h * 0.5f, true); // Redundant mid-right movement for '3' shape
    add_waypoint(env, current_x + char_w, base_y + char_h, true); 
    add_waypoint(env, current_x, base_y + char_h, true); 
    current_x += char_w; // Advance for the period character block

    // . (period as a short line connected to 3 and leading to 0)
    float period_y_level = base_y + char_h; // Align with bottom of '3'
    float period_length = dot_square_size * 1.5f; // Make it a short dash

    // No pen up from '3', draw directly to period start
    // The last point of '3' is current_x, base_y + char_h (implicitly)
    // Let's lift pen, move slightly, then draw period, then lift, then move to 0 start

    add_waypoint(env, current_x, period_y_level, false); // Pen up after 3, at its end point positionally
    current_x += spacing * 0.25f; // Small gap before period
    add_waypoint(env, current_x, period_y_level, false); // Move to period start (pen up)
    add_waypoint(env, current_x, period_y_level, true);  // Period start (pen down)
    current_x += period_length;
    add_waypoint(env, current_x, period_y_level, true);  // Period end (pen down)
    add_waypoint(env, current_x, period_y_level, false); // Pen up after period
    current_x += spacing * 0.25f; // Small gap after period before 0
    
    // 0
    add_waypoint(env, current_x + char_w * 0.5f, base_y, false); // Move to start of 0 (pen up)
    add_waypoint(env, current_x + char_w * 0.5f, base_y, true); 
    add_waypoint(env, current_x + char_w, base_y + char_h * 0.5f, true); 
    add_waypoint(env, current_x + char_w * 0.5f, base_y + char_h, true); 
    add_waypoint(env, current_x, base_y + char_h * 0.5f, true); 
    add_waypoint(env, current_x + char_w * 0.5f, base_y, true); 

    // Final pen up after finishing
    if (env->num_trace_waypoints > 0) { 
        TraceWaypoint last_wp = env->trace_waypoints[env->num_trace_waypoints - 1];
        add_waypoint(env, last_wp.position.x, last_wp.position.y, false);
    }
}

// Defines the path for "pufferlib 3.0"
void setup_pufferlib_trace_path(AntsEnv* env) {
    env->num_trace_waypoints = 0; // Reset previous path

    float base_x = env->width * 0.05f;  // Start further left
    float base_y = env->height * 0.15f; // Start a bit higher
    float char_h = 105.0f; // Was 150.0f
    float char_w = 63.0f;  // Was 90.0f
    float spacing = 31.5f; // Was 45.0f
    float current_x = base_x;
    float dot_size = 7.0f; // Was 10.0f

    // P
    add_waypoint(env, current_x, base_y + char_h, false); // Move to start of P (pen up)
    add_waypoint(env, current_x, base_y + char_h, true);  // P stem bottom
    add_waypoint(env, current_x, base_y, true);           // P stem top
    add_waypoint(env, current_x + char_w * 0.75f, base_y, true); // P curve top-right
    add_waypoint(env, current_x + char_w, base_y + char_h * 0.25f, true);
    add_waypoint(env, current_x + char_w * 0.75f, base_y + char_h * 0.5f, true);
    add_waypoint(env, current_x, base_y + char_h * 0.5f, true); // P curve join stem
    current_x += char_w + spacing; // Adjusted from current_x += char_w;

    // u
    add_waypoint(env, current_x, base_y, false); // Move to start of u (top-left, pen up)
    add_waypoint(env, current_x, base_y, true); // Top-left of u
    add_waypoint(env, current_x, base_y + char_h, true); // Bottom-left of u
    add_waypoint(env, current_x + char_w, base_y + char_h, true); // Bottom-right of u
    add_waypoint(env, current_x + char_w, base_y, true); // Top-right of u
    current_x += char_w + spacing;

    // f (lowercase f for Pufferlib 3.0)
    {
        float fx = current_x + char_w * 0.5f; // stem at center
        float top_bar_left = fx - char_w * 0.4f;
        float top_bar_right = fx + char_w * 0.4f;
        float mid_y = base_y + char_h * 0.5f;

        // Vertical stem (bottom to top)
        add_waypoint(env, fx, base_y + char_h, false);
        add_waypoint(env, fx, base_y, true);

        // Top crossbar (at base_y)
        add_waypoint(env, top_bar_left, base_y, false);
        add_waypoint(env, top_bar_right, base_y, true);
        
        // Middle crossbar (from stem to right)
        add_waypoint(env, fx, mid_y, false);
        add_waypoint(env, fx + char_w * 0.4f, mid_y, true);
    }
    current_x += char_w + spacing;

    // f (second lowercase f for Pufferlib 3.0)
    {
        float fx = current_x + char_w * 0.5f; // stem at center
        float top_bar_left = fx - char_w * 0.4f;
        float top_bar_right = fx + char_w * 0.4f;
        float mid_y = base_y + char_h * 0.5f;

        // Vertical stem (bottom to top)
        add_waypoint(env, fx, base_y + char_h, false);
        add_waypoint(env, fx, base_y, true);

        // Top crossbar (at base_y)
        add_waypoint(env, top_bar_left, base_y, false);
        add_waypoint(env, top_bar_right, base_y, true);
        
        // Middle crossbar (from stem to right)
        add_waypoint(env, fx, mid_y, false);
        add_waypoint(env, fx + char_w * 0.4f, mid_y, true);
    }
    current_x += char_w + spacing;

    // e (Mirrored for Pufferlib 3.0)
    add_waypoint(env, current_x, base_y + char_h * 0.5f, false); 
    add_waypoint(env, current_x, base_y + char_h * 0.5f, true); 
    add_waypoint(env, current_x + char_w, base_y + char_h * 0.5f, true); 
    add_waypoint(env, current_x + char_w, base_y, true); 
    add_waypoint(env, current_x, base_y, true); 
    add_waypoint(env, current_x, base_y + char_h, true); 
    add_waypoint(env, current_x + char_w, base_y + char_h, true); 
    current_x += char_w + spacing;

    // r
    add_waypoint(env, current_x, base_y + char_h, false);
    add_waypoint(env, current_x, base_y + char_h, true); // Bottom of r stem
    add_waypoint(env, current_x, base_y, true);         // Top of r stem
    add_waypoint(env, current_x + char_w * 0.75f, base_y, true); // r curve top-right
    add_waypoint(env, current_x + char_w, base_y + char_h * 0.25f, true); // r curve mid-right
    current_x += char_w + spacing;

    // l
    add_waypoint(env, current_x, base_y, false);
    add_waypoint(env, current_x, base_y, true); // Top of l
    add_waypoint(env, current_x, base_y + char_h, true); // Bottom of l
    current_x += char_w * 0.6f + spacing; // l is thinner

    // i
    float i_body_x = current_x;
    add_waypoint(env, i_body_x, base_y + char_h * 0.25f, false);
    add_waypoint(env, i_body_x, base_y + char_h * 0.25f, true); // Top of i body
    add_waypoint(env, i_body_x, base_y + char_h, true);      // Bottom of i body
    // dot for i
    float i_dot_x = i_body_x;
    float i_dot_y = base_y + char_h * 0.1f; // Above body
    add_waypoint(env, i_dot_x - dot_size/2, i_dot_y - dot_size/2, false); // Move to dot area (pen up)
    add_waypoint(env, i_dot_x - dot_size/2, i_dot_y - dot_size/2, true);  // Start dot square
    add_waypoint(env, i_dot_x + dot_size/2, i_dot_y - dot_size/2, true);
    add_waypoint(env, i_dot_x + dot_size/2, i_dot_y + dot_size/2, true);
    add_waypoint(env, i_dot_x - dot_size/2, i_dot_y + dot_size/2, true);
    add_waypoint(env, i_dot_x - dot_size/2, i_dot_y - dot_size/2, true);  // Close dot square
    current_x += char_w * 0.6f + spacing; // i is thinner

    // b
    add_waypoint(env, current_x, base_y, false);
    add_waypoint(env, current_x, base_y, true); // Top of b stem
    add_waypoint(env, current_x, base_y + char_h, true); // Bottom of b stem
    add_waypoint(env, current_x + char_w * 0.75f, base_y + char_h, true); // b curve bottom-right
    add_waypoint(env, current_x + char_w, base_y + char_h * 0.75f, true);
    add_waypoint(env, current_x + char_w * 0.75f, base_y + char_h * 0.5f, true);
    add_waypoint(env, current_x, base_y + char_h * 0.5f, true); // Join b curve to stem
    current_x += char_w + spacing + 20.0f; // Extra space before 3.0

    // 3
    add_waypoint(env, current_x, base_y, false);
    add_waypoint(env, current_x, base_y, true); // Top-left of 3
    add_waypoint(env, current_x + char_w, base_y, true); // 3 top bar
    add_waypoint(env, current_x + char_w, base_y + char_h * 0.5f, true); // 3 mid-right
    add_waypoint(env, current_x, base_y + char_h * 0.5f, true); // 3 mid-left (then back)
    add_waypoint(env, current_x + char_w, base_y + char_h * 0.5f, true); // 3 mid-right again
    add_waypoint(env, current_x + char_w, base_y + char_h, true); // 3 bottom-right
    add_waypoint(env, current_x, base_y + char_h, true); // 3 bottom-left
    current_x += char_w + spacing * 0.5f;
    
    // . (period)
    float period_dot_x = current_x + dot_size; // Position it after the 3
    float period_dot_y = base_y + char_h;    // At the baseline
    add_waypoint(env, period_dot_x - dot_size/2, period_dot_y - dot_size/2, false); // Move to dot area (pen up)
    add_waypoint(env, period_dot_x - dot_size/2, period_dot_y - dot_size/2, true);  // Start dot square
    add_waypoint(env, period_dot_x + dot_size/2, period_dot_y - dot_size/2, true);
    add_waypoint(env, period_dot_x + dot_size/2, period_dot_y + dot_size/2, true);
    add_waypoint(env, period_dot_x - dot_size/2, period_dot_y + dot_size/2, true);
    add_waypoint(env, period_dot_x - dot_size/2, period_dot_y - dot_size/2, true);  // Close dot square
    current_x += dot_size * 2 + spacing * 0.5f; // Space after period

    // 0
    add_waypoint(env, current_x + char_w * 0.5f, base_y, false);
    add_waypoint(env, current_x + char_w * 0.5f, base_y, true); // Top-mid of 0
    add_waypoint(env, current_x + char_w, base_y + char_h * 0.5f, true); // Mid-right of 0
    add_waypoint(env, current_x + char_w * 0.5f, base_y + char_h, true); // Bottom-mid of 0
    add_waypoint(env, current_x, base_y + char_h * 0.5f, true); // Mid-left of 0
    add_waypoint(env, current_x + char_w * 0.5f, base_y, true); // Back to Top-mid of 0 (close 0)
    
    // Final pen up after finishing
    if (env->num_trace_waypoints > 0) { // Add a final pen-up move if path exists
        TraceWaypoint last_wp = env->trace_waypoints[env->num_trace_waypoints - 1];
        add_waypoint(env, last_wp.position.x, last_wp.position.y, false);
    }
}

void c_reset(AntsEnv* env) {
    env->tick = 0;
    env->log = (Log){0};
    env->num_pheromones = 0;
    
    // Reset colonies
    env->colonies[0].food_collected = 0;
    env->colonies[1].food_collected = 0;
    
    // Initialize all ants
    int ant_idx = 0;
    for (int i = 0; i < NUM_COLONIES; i++) {
        for (int j = 0; j < env->num_ants / NUM_COLONIES; j++) {
            if (ant_idx < env->num_ants) { // Ensure we don't write past allocated ants if num_ants is small (e.g. 1)
                 env->ants[ant_idx].colony_id = i;
                 env->ants[ant_idx].has_paintbrush = false;
                 spawn_ant(env, ant_idx);
                 ant_idx++;
            }
        }
    }
    
    // If num_ants is 1 (for single ant debugging/tracing), ensure its colony_id is set
    if (env->num_ants == 1 && ant_idx == 0) {
        env->ants[0].colony_id = 0; // Default to colony 0 (red) initially
        env->ants[0].has_paintbrush = false;
        // Tracing will explicitly set it to 1 (cyan) later
        spawn_ant(env, 0);
    }
    
    // Clear food sources and spawn new ones
    for (int i = 0; i < env->num_food_sources; i++) {
        env->food_sources[i].amount = 0;
    }
    
    for (int i = 0; i < env->num_food_sources; i++) {
        spawn_food(env);
    }
    
    // Clear buffers
    memset(env->rewards, 0, env->num_ants * sizeof(float));
    memset(env->terminals, 0, env->num_ants * sizeof(unsigned char));
    memset(env->ant_is_dropping_pheromone, 0, env->num_ants * sizeof(bool)); // Reset drop flags each step

    // Reset tracing state
    env->is_tracing_text = false;
    env->current_trace_waypoint_index = 0;
    env->num_trace_waypoints = 0; // Path will be set on demand
    env->tracing_complete = false;

    // Reset and spawn paintbrush feature elements
    env->currently_writing_with_paintbrush = false;
    for(int i = 0; i < env->num_ants; ++i) env->ants[i].has_paintbrush = false;
    env->paintbrush_spawned = false; // It will be spawned by spawn_paintbrush_food
    printf("reset line 599\n");
    spawn_paintbrush_food(env); // New function call
    
    // Generate initial observations
    compute_observations(env);
}

// New function to determine action for tracing ant
void determine_action_for_tracing_ant(AntsEnv* env, int ant_id) {
    Ant* ant = &env->ants[ant_id];
    if (env->current_trace_waypoint_index >= env->num_trace_waypoints) {
        env->is_tracing_text = false; // Path complete
        env->tracing_complete = true; // Signal main loop to take back control
        env->actions[ant_id] = ACTION_HALT; // Halt after tracing
        
        if (env->currently_writing_with_paintbrush) {
            ant->has_paintbrush = false;
            env->currently_writing_with_paintbrush = false;
            env->paintbrush_spawned = false; // Mark so a new one can spawn
            spawn_paintbrush_food(env); // Spawn a new one
        }
        return;
    }

    TraceWaypoint* target_wp = &env->trace_waypoints[env->current_trace_waypoint_index];
    Vector2 target_pos = target_wp->position;
    float dx = target_pos.x - ant->position.x;
    float dy = target_pos.y - ant->position.y;
    float distance_to_target = sqrt(dx*dx + dy*dy);

    // Check if ant is close enough to the waypoint
    if (distance_to_target < ANT_SPEED * 1.5f) { // Consider waypoint reached
        env->current_trace_waypoint_index++;
        if (env->current_trace_waypoint_index >= env->num_trace_waypoints) {
            // This block is similar to the one at the start, handles last waypoint
            env->is_tracing_text = false;
            env->tracing_complete = true;
            env->actions[ant_id] = ACTION_HALT; // Halt after tracing
            if (env->currently_writing_with_paintbrush) {
                ant->has_paintbrush = false;
                env->currently_writing_with_paintbrush = false;
                env->paintbrush_spawned = false;
                spawn_paintbrush_food(env); 
            }
            return;
        }
        // Update target to the new waypoint for turning logic below
        target_wp = &env->trace_waypoints[env->current_trace_waypoint_index];
        target_pos = target_wp->position;
        dx = target_pos.x - ant->position.x;
        dy = target_pos.y - ant->position.y;
    }

    // Determine action to reach the current target_wp
    float target_angle = atan2(dy, dx);
    float angle_diff = wrap_angle(target_angle - ant->direction);

    if (fabs(angle_diff) < TURN_ANGLE * 0.5f) { // If facing target, move forward
        env->actions[ant_id] = ACTION_MOVE_FORWARD;
    } else if (angle_diff < 0) {
        env->actions[ant_id] = ACTION_TURN_LEFT;
    } else {
        env->actions[ant_id] = ACTION_TURN_RIGHT;
    }
    // Pheromone dropping logic is handled later based on pen_down state of the target_wp
}

void step_ant(AntsEnv* env, int ant_id) {
    Ant* ant = &env->ants[ant_id];
    env->ant_logs[ant_id].episode_length += 1;
    ant->lifetime++;

    int action = env->actions[ant_id]; // Get action from main loop or previous state

    // If tracing text, determine action based on waypoints
    // This check is vital: only the designated tracing ant (colony_id 1) follows waypoints.
    if (env->is_tracing_text && !env->tracing_complete && ant->colony_id == 1) {
        determine_action_for_tracing_ant(env, ant_id);
        action = env->actions[ant_id]; // Update action if it was changed by tracing logic (e.g., to HALT)
    }
    // If tracing just finished this tick (is_tracing_text became false because tracing_complete became true),
    // determine_action_for_tracing_ant would have set action to HALT.

    // Execute action
    if (action != ACTION_HALT) {
        ant->position.x += ANT_SPEED * cos(ant->direction);
        ant->position.y += ANT_SPEED * sin(ant->direction);
        
        // Wrap around edges
        if (ant->position.x < 0) ant->position.x = env->width;
        if (ant->position.x > env->width) ant->position.x = 0;
        if (ant->position.y < 0) ant->position.y = env->height;
        if (ant->position.y > env->height) ant->position.y = 0;
        
        // Action-specific turning
        // ACTION_MOVE_FORWARD (0) is implicit if not turning or halting.
        if (action == ACTION_TURN_LEFT) { // Replaced switch for clarity with HALT
            ant->direction -= TURN_ANGLE;
            ant->direction = wrap_angle(ant->direction);
        } else if (action == ACTION_TURN_RIGHT) {
            ant->direction += TURN_ANGLE;
            ant->direction = wrap_angle(ant->direction);
        }
    }
    
    // Smoothly interpolate visual direction towards actual direction
    float angle_diff = wrap_angle(ant->direction - ant->visual_direction);
    ant->visual_direction += angle_diff * 0.2f;  // 20% interpolation for smooth turning
    ant->visual_direction = wrap_angle(ant->visual_direction);
    
    // Check for food collection
    if (!ant->has_food && !ant->has_paintbrush) { // Can only pick up food if not holding paintbrush
        for (int j = 0; j < env->num_food_sources; j++) {
            if (env->food_sources[j].amount > 0) {
                float dist_sq = distance_squared(ant->position, env->food_sources[j].position);
                if (dist_sq < (ANT_SIZE + FOOD_SIZE) * (ANT_SIZE + FOOD_SIZE)) {
                    ant->has_food = true;
                    env->food_sources[j].amount--;
                    env->rewards[ant_id] = env->reward_food;
                    env->ant_logs[ant_id].episode_return += env->reward_food;
                    break;
                }
            }
        }
    }
    
    // Check for paintbrush collection
    if (!ant->has_paintbrush && env->paintbrush_spawned && env->paintbrush_food.amount > 0) { // Removed !ant->has_food
        float dist_sq_paintbrush = distance_squared(ant->position, env->paintbrush_food.position);
        // Collision check using PAINTBRUSH_RENDER_SIZE
        if (dist_sq_paintbrush < (ANT_SIZE + PAINTBRUSH_RENDER_SIZE / 2.0f) * (ANT_SIZE + PAINTBRUSH_RENDER_SIZE / 2.0f)) { 
            if (ant->has_food) { // If holding food, drop it
                ant->has_food = false;
                // Optionally, could add a small negative reward or log this event
            }
            ant->has_paintbrush = true;
            env->paintbrush_food.amount = 0; // Mark as collected
            env->paintbrush_spawned = false; // It's picked up, not in world

            // Start "Puffer" tracing routine automatically
            env->is_tracing_text = true;
            env->tracing_complete = false;
            env->currently_writing_with_paintbrush = true;
            // ant->colony_id = 1; // DO NOT CHANGE ANT'S COLONY_ID, retain original sprite
            
            // Ensure only one ant is tracing for now if multiple ants exist.
            // More complex logic would be needed if multiple ants could trace simultaneously.
        }
    }
    
    // Check for food delivery
    if (ant->has_food) {
        Colony* colony = &env->colonies[ant->colony_id];
        float dist_sq = distance_squared(ant->position, colony->position);
        if (dist_sq < (ANT_SIZE + COLONY_SIZE) * (ANT_SIZE + COLONY_SIZE)) {
            ant->has_food = false;
            colony->food_collected++;
            env->rewards[ant_id] = env->reward_delivery;
            env->ant_logs[ant_id].episode_return += env->reward_delivery;
            env->ant_logs[ant_id].score += 1; // Score based on deliveries
        }
    }
    
    // Pheromone dropping logic
    bool should_drop_pheromone_now = false;
    if (env->is_tracing_text && env->current_trace_waypoint_index < env->num_trace_waypoints) {
        // For any tracing (KEY_FIVE or paintbrush), use the waypoint's pen_down state
        should_drop_pheromone_now = env->trace_waypoints[env->current_trace_waypoint_index].pen_down;
    } else if (!env->is_tracing_text) { // Manual or AI control when not tracing
        if (env->ant_is_dropping_pheromone[ant_id]) { 
            should_drop_pheromone_now = true;
        }
    }

    if (should_drop_pheromone_now) {
        // If tracing, always use colony_id 1 (cyan) for pheromones,
        // regardless of the ant's actual current colony_id (sprite).
        // Otherwise, use the ant's own colony_id.
        int pheromone_colony_id_to_use = env->is_tracing_text ? 1 : ant->colony_id;
        add_pheromone(env, ant->position, pheromone_colony_id_to_use);
    }
    
    // MULTIPLE TERMINAL CONDITIONS FOR FREQUENT LOG GENERATION
    bool should_terminate = false;
    
    // Terminal Condition 1: Significantly increased lifetime limit
    if (ant->lifetime > 200000 && !env->is_tracing_text) { // Only terminate if not tracing, and after a very long time
        should_terminate = true;
    }
        
    if (should_terminate) {
        env->ant_logs[ant_id].perf = env->ant_logs[ant_id].episode_length > 0 ? 
                                     env->ant_logs[ant_id].score / env->ant_logs[ant_id].episode_length : 0;
        add_log(env, ant_id);
        spawn_ant(env, ant_id);
        env->terminals[ant_id] = 1;
        
        // Debug output for terminal condition verification
        if (env->tick % 100 == 0) {
            printf("Ant %d terminated at tick %d, lifetime %d, score %.1f\n", 
                   ant_id, env->tick, ant->lifetime, env->ant_logs[ant_id].score);
        }
    }
}

void c_step(AntsEnv* env) {
    env->tick++;

    // Periodically try to spawn paintbrush if conditions are met
    if (env->tick > 10 && env->tick % 200 == 100 && !env->paintbrush_spawned && !env->currently_writing_with_paintbrush) {
        spawn_paintbrush_food(env);
    }
    
    // Clear rewards and terminals
    memset(env->rewards, 0, env->num_ants * sizeof(float));
    memset(env->terminals, 0, env->num_ants * sizeof(unsigned char));
    // env.ant_is_dropping_pheromone is now managed by the main loop in ants.c before c_step is called
    // memset(env->ant_is_dropping_pheromone, 0, env->num_ants * sizeof(bool)); // REMOVED THIS LINE
    
    // Step all ants
    for (int i = 0; i < env->num_ants; i++) {
        step_ant(env, i);
    }
    
    // Update pheromones
    for (int i = 0; i < env->num_pheromones; i++) {
        env->pheromones[i].strength -= PHEROMONE_EVAPORATION_RATE;
        if (env->pheromones[i].strength <= 0) {
            // Remove evaporated pheromone
            env->pheromones[i] = env->pheromones[env->num_pheromones - 1];
            env->num_pheromones--;
            i--;
        }
    }
    
    // Generate new observations
    compute_observations(env);
}

// Helper function to check if file exists
static inline bool file_exists(const char* path) {
    return access(path, F_OK) != -1;
}

// Raylib client functions - FOLLOWING SNAKE PATTERN
Client* make_client(int cell_size, int width, int height) {
    Client* client = (Client*)malloc(sizeof(Client));
    client->cell_size = cell_size;
    client->width = width;
    client->height = height;
    InitWindow(width, height, "PufferLib Ant Colony");
    SetTargetFPS(60);
    
    // Load texture with path resolution logic from breakout.h
    char texturePath[PATH_MAX] = {0};
    char basePath[PATH_MAX] = {0};
    char resolvedPath[PATH_MAX] = {0};

    // Try to find puffers texture
    const char* pufferPaths[] = {
        "./resources/puffers_128_red_cyan.png",
        "./pufferlib/resources/puffers_128_red_cyan.png",
        "./pufferlib/pufferlib/resources/puffers_128_red_cyan.png",
        "/puffertank/release_test_pufferlib/pufferlib/resources/puffers_128_red_cyan.png"
    };

    int found = 0;
    for (size_t i = 0; i < sizeof(pufferPaths)/sizeof(pufferPaths[0]); i++) {
        if (file_exists(pufferPaths[i])) {
            if (realpath(pufferPaths[i], resolvedPath) != NULL) {
                strncpy(texturePath, resolvedPath, PATH_MAX - 1);
                found = 1;
                break;
            }
        }
    }

    if (!found) {
        TraceLog(LOG_ERROR, "Failed to find puffers_128_red_cyan.png from current directory.");
        CloseWindow();
        free(client);
        exit(EXIT_FAILURE);
    }

    client->ant = LoadTexture(texturePath);
    TraceLog(LOG_INFO, "Puffer texture loaded: %s", texturePath);
    
    // Try to find colony base texture
    const char* basePaths[] = {
        "./resources/ant_colony_base.png",
        "./pufferlib/resources/ant_colony_base.png",
        "./pufferlib/pufferlib/resources/ant_colony_base.png",
        "/puffertank/release_test_pufferlib/pufferlib/resources/ant_colony_base.png"
    };

    found = 0;
    for (size_t i = 0; i < sizeof(basePaths)/sizeof(basePaths[0]); i++) {
        if (file_exists(basePaths[i])) {
            if (realpath(basePaths[i], resolvedPath) != NULL) {
                strncpy(basePath, resolvedPath, PATH_MAX - 1);
                found = 1;
                break;
            }
        }
    }

    if (!found) {
        TraceLog(LOG_ERROR, "Failed to find ant_colony_base.png from current directory.");
        UnloadTexture(client->ant);
        CloseWindow();
        free(client);
        exit(EXIT_FAILURE);
    }

    client->colony_base = LoadTexture(basePath);
    TraceLog(LOG_INFO, "Colony base texture loaded: %s", basePath);
    
    // Try to find paintbrush texture
    const char* paintbrushImagePaths[] = {
        "./resources/paintbrush.png",
        "./pufferlib/resources/paintbrush.png",
        "./pufferlib/pufferlib/resources/paintbrush.png",
        "/puffertank/release_test_pufferlib/pufferlib/resources/paintbrush.png"
    };
    char paintbrushTexturePath[PATH_MAX] = {0};
    found = 0;
    for (size_t i = 0; i < sizeof(paintbrushImagePaths)/sizeof(paintbrushImagePaths[0]); i++) {
        if (file_exists(paintbrushImagePaths[i])) {
            if (realpath(paintbrushImagePaths[i], resolvedPath) != NULL) {
                strncpy(paintbrushTexturePath, resolvedPath, PATH_MAX - 1);
                found = 1;
                break;
            }
        }
    }
    if (!found) {
        TraceLog(LOG_WARNING, "Paintbrush texture (paintbrush.png) not found. Feature will be visual-less.");
        client->paintbrush_texture = (Texture2D){0}; // Invalid texture, won't draw
    } else {
        client->paintbrush_texture = LoadTexture(paintbrushTexturePath);
        TraceLog(LOG_INFO, "Paintbrush texture loaded: %s", paintbrushTexturePath);
    }
    
    return client;
}

void close_client(Client* client) {
    UnloadTexture(client->ant);
    UnloadTexture(client->colony_base);
    if (client->paintbrush_texture.id > 0) { // Unload only if loaded
        UnloadTexture(client->paintbrush_texture);
    }
    CloseWindow();
    free(client);
}

void c_render(AntsEnv* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    
    if (env->client == NULL) {
        env->client = make_client(1, env->width, env->height);
    }

    // Define texture properties and pivot points
    #define ANT_TEXTURE_WIDTH 125.0f
    #define ANT_TEXTURE_HEIGHT 89.0f

    // Option 1: Geometric Center (actual middle of the 125x89 texture)
    // #define PIVOT_OFFSET_X (ANT_TEXTURE_WIDTH / 2.0f)
    // #define PIVOT_OFFSET_Y (ANT_TEXTURE_HEIGHT / 2.0f)

    // Option 2: Subjective Center (derived from user's detailed description)
    // This is the point within the 125x89 texture (from its top-left 0,0)
    // that should be considered its logical center/pivot. (Approx. 56, 38)
    #define PIVOT_OFFSET_X 80.0f // User calibrated
    #define PIVOT_OFFSET_Y 55.0f // User calibrated

    BeginDrawing();
    ClearBackground(BACKGROUND_COLOR);
    
    // Draw colonies with texture
    for (int i = 0; i < NUM_COLONIES; i++) {
        Color colony_color = (i == 0) ? COLONY1_COLOR : COLONY2_COLOR;
        Vector2 pos = env->colonies[i].position;
        
        // Draw colony base texture
        float base_scale = (COLONY_SIZE * 2.0f) / 246.0f; // Scale to fit colony size
        Rectangle source_rect = {0, 0, 246, 252}; // Original source rectangle for the texture

        if (i == 1) { // Flip colony base horizontally for the second colony (e.g., Radiant)
            source_rect.x = source_rect.x + source_rect.width; // Adjust x position for flip
            source_rect.width = -source_rect.width;           // Negative width flips the texture
        }
        
        DrawTexturePro(
            env->client->colony_base,
            source_rect, // Use the (potentially flipped) source_rect
            (Rectangle){
                pos.x - (246 * base_scale) / 2,
                pos.y - (252 * base_scale) / 2,
                246 * base_scale,
                252 * base_scale
            },
            (Vector2){0, 0},
            0,
            colony_color
        );
    }
    
    // Draw food sources with better styling
    for (int i = 0; i < env->num_food_sources; i++) {
        if (env->food_sources[i].amount > 0) {
            Vector2 pos = env->food_sources[i].position;
            
            // Food pile effect - multiple small circles
            float pile_size = FOOD_SIZE * (env->food_sources[i].amount / (float)MAX_FOOD_PER_SOURCE);
            
            // Base circle
            Color food_base = FOOD_COLOR;
            food_base.a = 150;
            DrawCircle(pos.x, pos.y, pile_size, food_base);
            
            // Small food particles
            for (int j = 0; j < 5; j++) {
                float angle = j * 2 * M_PI / 5;
                float offset = pile_size * 0.5f;
                float px = pos.x + offset * cos(angle);
                float py = pos.y + offset * sin(angle);
                DrawCircle(px, py, pile_size * 0.3f, FOOD_COLOR);
            }
            
            // Center highlight
            DrawCircle(pos.x, pos.y, pile_size * 0.4f, FOOD_COLOR);
        }
    }
    
    // Draw paintbrush item on the ground
    if (env->paintbrush_spawned && env->paintbrush_food.amount > 0 && env->client->paintbrush_texture.id > 0) {
        DrawTexturePro(
            env->client->paintbrush_texture,
            (Rectangle){0, 0, (float)env->client->paintbrush_texture.width, (float)env->client->paintbrush_texture.height},
            (Rectangle){
                env->paintbrush_food.position.x,
                env->paintbrush_food.position.y,
                PAINTBRUSH_RENDER_SIZE,
                PAINTBRUSH_RENDER_SIZE
            },
            (Vector2){PAINTBRUSH_RENDER_SIZE/2, PAINTBRUSH_RENDER_SIZE/2}, // Pivot at center of rendered brush
            0.0f, // No rotation for ground item
            WHITE
        );
    }

    // Draw pheromones
    for (int i = 0; i < env->num_pheromones; i++) {
        Color pheromone_color = (env->pheromones[i].colony_id == 0) ? PHEROMONE1_COLOR : PHEROMONE2_COLOR;
        pheromone_color.a = (unsigned char)(100 * env->pheromones[i].strength);
        DrawCircle(env->pheromones[i].position.x, env->pheromones[i].position.y, 
                  PHEROMONE_SIZE, pheromone_color);
    }
    
    // Draw ants using texture instead of circles
    for (int i = 0; i < env->num_ants; i++) {
        Ant* ant = &env->ants[i];
        int sprite_x = (ant->colony_id == 0) ? 0 : 125; // Select sprite for colony

        // ant->position is the logical PIVOT point of the ant on the screen.
        
        // --- Start of DrawTexturePro logic for scaling and rotation ---
        // Scaling logic to restore previous visual size (approx. 48px width)
        float target_rendered_width = ANT_SIZE * 4 * 3.0f; // ANT_SIZE=4, so 4*4*3.0 = 48.0f
        float scale_factor = target_rendered_width / ANT_TEXTURE_WIDTH; // Scale factor to get from 125px to 48px width

        float actual_rendered_width = ANT_TEXTURE_WIDTH * scale_factor; // Should be target_rendered_width (e.g., 48.0f)
        float actual_rendered_height = ANT_TEXTURE_HEIGHT * scale_factor; // Maintain aspect ratio

        // Calculate the pivot point on the SCALED texture. This is the origin for DrawTexturePro.
        Vector2 origin_on_scaled_texture = {
            PIVOT_OFFSET_X * scale_factor,
            PIVOT_OFFSET_Y * scale_factor
        };

        Color tint = WHITE;
        if (ant->has_food) {
            tint = (Color){150, 255, 150, 255};
        } else if (ant->has_paintbrush) {
            tint = (Color){200, 150, 255, 255}; // Lavender tint for holding paintbrush
        }
        
        float rotation_degrees = ant->visual_direction * 180.0f / M_PI;

        DrawTexturePro(
            env->client->ant,
            (Rectangle){(float)sprite_x, 0, ANT_TEXTURE_WIDTH, ANT_TEXTURE_HEIGHT}, // Use the direct source rect from texture sheet
            (Rectangle){ // Destination rect on screen
                ant->position.x, // The X of the pivot point on screen
                ant->position.y, // The Y of the pivot point on screen
                actual_rendered_width,  // Full width of the scaled texture
                actual_rendered_height // Full height of the scaled texture
            },
            origin_on_scaled_texture, // Origin of rotation/scaling (the pivot point *within the destination rectangle*)
            rotation_degrees,
            tint
        );
        // --- End of DrawTexturePro logic ---
         
        // Draw a small BLUE circle at the ant's logical PIVOT point for alignment checking
        DrawCircle(ant->position.x, ant->position.y, 3, BLUE); // Smaller, different color

        // Draw paintbrush near ant if holding it
        if (ant->has_paintbrush && env->client->paintbrush_texture.id > 0) {
            // Offset from ant's pivot to simulate holding (e.g., in front/mouth)
            float brush_offset_x = 20.0f; 
            float brush_offset_y = 0.0f;  

            float rotated_offset_x = brush_offset_x * cos(ant->visual_direction) - brush_offset_y * sin(ant->visual_direction);
            float rotated_offset_y = brush_offset_x * sin(ant->visual_direction) + brush_offset_y * cos(ant->visual_direction);

            DrawTexturePro(
                env->client->paintbrush_texture,
                (Rectangle){0, 0, (float)env->client->paintbrush_texture.width, (float)env->client->paintbrush_texture.height},
                (Rectangle){
                    ant->position.x + rotated_offset_x,
                    ant->position.y + rotated_offset_y,
                    PAINTBRUSH_RENDER_SIZE, 
                    PAINTBRUSH_RENDER_SIZE
                },
                (Vector2){PAINTBRUSH_RENDER_SIZE/2, PAINTBRUSH_RENDER_SIZE/2}, // pivot at center of rendered brush
                rotation_degrees, 
                WHITE
            );
        }
 
        // The DrawTextureRec and old green debug line are removed as DrawTexturePro handles this now.
    }
    
    // Draw UI with better colors
    Color dire_text = (Color){255, 100, 100, 255};
    Color radiant_text = (Color){100, 200, 255, 255};
    DrawText(TextFormat("Dire Food: %d", env->colonies[0].food_collected), 20, 20, 20, dire_text);
    DrawText(TextFormat("Radiant Food: %d", env->colonies[1].food_collected), 20, 50, 20, radiant_text);
    DrawText(TextFormat("Tick: %d", env->tick), env->width - 120, 20, 20, RAYWHITE);
    
    EndDrawing();
}
