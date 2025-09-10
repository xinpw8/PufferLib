/* Drone Pick and Place: An RL environment for drone manipulation tasks
 * Training drones to pick up objects and place them at target locations
 * For the REVEL x LycheeAI Isaac Sim Hackathon
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

// Action constants for discrete control (10 actions for quadcopter)
const unsigned char MOVE_FORWARD = 0;
const unsigned char MOVE_BACKWARD = 1;
const unsigned char MOVE_LEFT = 2;
const unsigned char MOVE_RIGHT = 3;
const unsigned char MOVE_UP = 4;
const unsigned char MOVE_DOWN = 5;
const unsigned char ROTATE_LEFT = 6;
const unsigned char ROTATE_RIGHT = 7;
const unsigned char GRIPPER_OPEN = 8;
const unsigned char GRIPPER_CLOSE = 9;

// State constants
const unsigned char STATE_SEARCHING = 0;
const unsigned char STATE_APPROACHING = 1;
const unsigned char STATE_GRASPING = 2;
const unsigned char STATE_TRANSPORTING = 3;
const unsigned char STATE_PLACING = 4;

// Required struct. Only use floats!
typedef struct {
    float perf; // 0-1 normalized performance metric
    float score; // Unnormalized score
    float episode_return; // Sum of agent rewards over episode
    float episode_length; // Number of steps in agent episode
    float grasp_success; // Success rate of grasping attempts
    float placement_success; // Success rate of placements
    float efficiency; // Path efficiency metric
    float n; // Required as the last field 
} Log;

// Internal tracking for success rates
typedef struct {
    int grasp_attempts;
    int grasp_successes;
    int placement_attempts;
    int placement_successes;
} Stats;

typedef struct {
    float x, y, z; // Position
    float vx, vy, vz; // Velocity
    float qw, qx, qy, qz; // Orientation quaternion
    float wx, wy, wz; // Angular velocity
    float yaw, pitch, roll; // Euler angles for convenience
    float gripper_open; // 0 = closed, 1 = open
    unsigned char state;
    int ticks_without_progress;
} Drone;

typedef struct {
    float x, y, z; // Position
    float vx, vy, vz; // Velocity (for physics simulation)
    float radius; // Object size
    unsigned char is_grasped;
    unsigned char is_placed;
} Object;

typedef struct {
    float x, y, z; // Target zone center
    float radius; // Zone radius
    unsigned char has_object; // Whether an object is placed here
} TargetZone;

typedef struct {
    Camera3D camera;
    int initialized;
} Client;

// Main environment struct
typedef struct {
    Log log; // Required field
    Stats stats; // Track attempts and successes
    Client* client;
    Drone* drones; // Support multiple drones
    Object* objects; // Multiple objects to pick
    TargetZone* targets; // Multiple target zones
    float* observations; // Required - continuous observations
    int* actions; // Required - discrete actions
    float* rewards; // Required
    unsigned char* terminals; // Required
    
    // Environment parameters
    int num_drones;
    int num_objects;
    int num_targets;
    float world_size;
    float max_height;
    int max_steps;
    int current_step;
    int debug_mode; // Set to 1 for standalone, 0 for vectorized
    
    // Physics parameters
    float dt;
    float gravity;
    float max_velocity;
    float max_angular_velocity;
    float grip_distance;
    float place_distance;
} DronePickPlace;

// Initialize environment
void init(DronePickPlace* env) {
    env->drones = calloc(env->num_drones, sizeof(Drone));
    env->objects = calloc(env->num_objects, sizeof(Object));
    env->targets = calloc(env->num_targets, sizeof(TargetZone));
    
    // Set physics defaults
    env->dt = 0.02f;
    env->gravity = -9.81f;
    env->max_velocity = 5.0f;
    env->max_angular_velocity = 3.14f;
    env->grip_distance = 0.25f;  // Very generous for easy grasping
    env->place_distance = 0.35f;  // Even more generous for placement
}

// Random float in range
float randf(float min, float max) {
    return min + (max - min) * ((float)rand() / (float)RAND_MAX);
}

// Distance between two 3D points
float distance3d(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

// Update physics for a single drone
void update_drone_physics(DronePickPlace* env, int drone_idx) {
    Drone* drone = &env->drones[drone_idx];
    int action = env->actions[drone_idx];
    
    // Apply control inputs with much stronger force
    float move_force = 10.0f;  // Much stronger for faster movement
    float rotate_speed = 2.5f;
    
    switch(action) {
        case MOVE_FORWARD:  // +Y direction (absolute)
            drone->vy += move_force * env->dt;
            break;
        case MOVE_BACKWARD:  // -Y direction (absolute)
            drone->vy -= move_force * env->dt;
            break;
        case MOVE_LEFT:  // -X direction (absolute)
            drone->vx -= move_force * env->dt;
            break;
        case MOVE_RIGHT:  // +X direction (absolute)
            drone->vx += move_force * env->dt;
            break;
        case MOVE_UP:
            drone->vz += move_force * env->dt;
            break;
        case MOVE_DOWN:
            drone->vz -= move_force * env->dt;
            break;
        case ROTATE_LEFT:
            drone->yaw += rotate_speed * env->dt;
            drone->wz = rotate_speed;
            break;
        case ROTATE_RIGHT:
            drone->yaw -= rotate_speed * env->dt;
            drone->wz = -rotate_speed;
            break;
        case GRIPPER_OPEN:
            drone->gripper_open = 1.0f;
            break;
        case GRIPPER_CLOSE:
            drone->gripper_open = 0.0f;
            break;
    }
    
    // Apply drag (air resistance)
    float drag = 0.98f;  // Much less drag for sustained movement
    drone->vx *= drag;
    drone->vy *= drag;
    drone->vz *= drag;
    
    // Apply gravity with hover compensation
    drone->vz += env->gravity * env->dt * 0.05f; // Very reduced gravity for drone hover
    
    // Clamp velocities
    float speed = sqrtf(drone->vx*drone->vx + drone->vy*drone->vy + drone->vz*drone->vz);
    if (speed > env->max_velocity) {
        float scale = env->max_velocity / speed;
        drone->vx *= scale;
        drone->vy *= scale;
        drone->vz *= scale;
    }
    
    // Update position
    drone->x += drone->vx * env->dt;
    drone->y += drone->vy * env->dt;
    drone->z += drone->vz * env->dt;
    
    // Boundary constraints
    drone->x = fmaxf(0, fminf(env->world_size, drone->x));
    drone->y = fmaxf(0, fminf(env->world_size, drone->y));
    drone->z = fmaxf(0.05f, fminf(env->max_height, drone->z));  // Allow drone to go lower
    
    // Wrap yaw angle
    while (drone->yaw > 3.14159f) drone->yaw -= 2 * 3.14159f;
    while (drone->yaw < -3.14159f) drone->yaw += 2 * 3.14159f;
    
    // Update quaternion from Euler angles (simplified - only yaw for now)
    drone->qw = cosf(drone->yaw / 2.0f);
    drone->qx = 0.0f;
    drone->qy = 0.0f;
    drone->qz = sinf(drone->yaw / 2.0f);
    
    // Update angular velocities with damping
    drone->wx *= 0.9f;
    drone->wy *= 0.9f;
    drone->wz *= 0.9f;
}

// Check and handle grasping
void update_grasping(DronePickPlace* env) {
    for (int d = 0; d < env->num_drones; d++) {
        Drone* drone = &env->drones[d];
        
        // Only process if gripper is closed
        if (drone->gripper_open > 0.5f) {
            continue;
        }
        
        // Check each object
        for (int o = 0; o < env->num_objects; o++) {
            Object* obj = &env->objects[o];
            
            // Skip if already grasped by another drone or placed
            if (obj->is_grasped || obj->is_placed) {
                continue;
            }
            
            // Check distance
            float dist = distance3d(drone->x, drone->y, drone->z,
                                   obj->x, obj->y, obj->z);
            
            // If close enough to attempt grasp
            if (dist < env->grip_distance * 1.5f) {
                if (dist < env->grip_distance && !obj->is_grasped) {
                    // This is the actual grasp attempt and success
                    env->stats.grasp_attempts++;
                    obj->is_grasped = 1;
                    drone->state = STATE_TRANSPORTING;
                    env->rewards[d] = 1.0f; // Max reward for successful grasp (clipped to 1)
                    env->stats.grasp_successes++;
                    
                    // Only log in debug mode (standalone) to avoid spam from parallel envs
                    if (env->debug_mode) {
                        printf("Drone %d grabbed object %d! (dist=%.2f)\n", d, o, dist);
                    }
                    break;
                }
            }
        }
    }
    
    // Update object positions for grasped objects
    for (int o = 0; o < env->num_objects; o++) {
        Object* obj = &env->objects[o];
        if (obj->is_grasped && !obj->is_placed) {
            // Find which drone has it
            for (int d = 0; d < env->num_drones; d++) {
                Drone* drone = &env->drones[d];
                if (drone->gripper_open < 0.5f) {
                    float dist = distance3d(drone->x, drone->y, drone->z,
                                          obj->x, obj->y, obj->z);
                    if (dist < env->grip_distance * 2) {
                        // Move object with drone
                        obj->x = drone->x;
                        obj->y = drone->y;
                        obj->z = drone->z - 0.1f; // Below drone
                    }
                }
            }
        }
    }
}

// Check and handle placement
void update_placement(DronePickPlace* env) {
    for (int o = 0; o < env->num_objects; o++) {
        Object* obj = &env->objects[o];
        
        if (!obj->is_grasped || obj->is_placed) {
            continue;
        }
        
        // Check if near any target zone (we already know object is grasped from check above)
        for (int t = 0; t < env->num_targets; t++) {
                TargetZone* target = &env->targets[t];
                float dist = distance3d(obj->x, obj->y, obj->z,
                                      target->x, target->y, target->z);
                
                if (dist < env->place_distance * 1.5f) {
                    // Check if any drone released it (attempting placement)
                    // But NOT in first 5 steps to avoid accidental placement at start
                    for (int d = 0; d < env->num_drones; d++) {
                        Drone* drone = &env->drones[d];
                        if (drone->gripper_open > 0.5f && env->current_step > 5) {
                            float drone_dist = distance3d(drone->x, drone->y, drone->z,
                                                         obj->x, obj->y, obj->z);
                            if (drone_dist < env->grip_distance * 2) {
                                
                                if (dist < env->place_distance) {
                                    // This is the actual placement attempt and success
                                    env->stats.placement_attempts++;
                                    obj->is_placed = 1;
                                    obj->is_grasped = 0;
                                    obj->vx = obj->vy = obj->vz = 0; // Stop object movement
                                    target->has_object = 1; // Mark target as occupied
                                    drone->state = STATE_SEARCHING;
                                    env->rewards[d] = 1.0f; // Max reward for successful placement
                                    env->stats.placement_successes++;
                                env->log.perf += 1.0f;
                                env->log.score += 50.0f;
                                
                                // Only log in debug mode (standalone)
                                if (env->debug_mode) {
                                    printf("Drone %d placed object %d in target %d!\n", d, o, t);
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
}

// Compute observations for all agents (45 observations per drone)
void compute_observations(DronePickPlace* env) {
    // 45 observations total:
    // Drone state: 14 (pos:3, vel:3, quat:4, ang_vel:3, gripper:1)
    // Objects: 21 (3 objects * 7 features each)
    // Targets: 8 (2 targets * 4 features each)
    // Task info: 2 (time_remaining, task_progress)
    int obs_per_drone = 45;
    
    for (int d = 0; d < env->num_drones; d++) {
        Drone* drone = &env->drones[d];
        int obs_idx = d * obs_per_drone;
        
        // Drone state (14 values)
        // Position (3)
        env->observations[obs_idx++] = drone->x / env->world_size;
        env->observations[obs_idx++] = drone->y / env->world_size;
        env->observations[obs_idx++] = drone->z / env->max_height;
        
        // Velocity (3)
        env->observations[obs_idx++] = drone->vx / env->max_velocity;
        env->observations[obs_idx++] = drone->vy / env->max_velocity;
        env->observations[obs_idx++] = drone->vz / env->max_velocity;
        
        // Orientation quaternion (4)
        env->observations[obs_idx++] = drone->qw;
        env->observations[obs_idx++] = drone->qx;
        env->observations[obs_idx++] = drone->qy;
        env->observations[obs_idx++] = drone->qz;
        
        // Angular velocity (3)
        env->observations[obs_idx++] = drone->wx / env->max_angular_velocity;
        env->observations[obs_idx++] = drone->wy / env->max_angular_velocity;
        env->observations[obs_idx++] = drone->wz / env->max_angular_velocity;
        
        // Gripper state (1)
        env->observations[obs_idx++] = drone->gripper_open;
        
        // Object states (7 per object * 3 objects = 21)
        for (int o = 0; o < env->num_objects; o++) {
            Object* obj = &env->objects[o];
            // Position (3)
            env->observations[obs_idx++] = obj->x / env->world_size;
            env->observations[obs_idx++] = obj->y / env->world_size;
            env->observations[obs_idx++] = obj->z / env->max_height;
            // Velocity (3)
            env->observations[obs_idx++] = obj->vx / env->max_velocity;
            env->observations[obs_idx++] = obj->vy / env->max_velocity;
            env->observations[obs_idx++] = obj->vz / env->max_velocity;
            // Status (1)
            env->observations[obs_idx++] = (float)(obj->is_grasped * 2 + obj->is_placed);
        }
        
        // Target zone states (4 per target * 2 targets = 8)
        for (int t = 0; t < env->num_targets; t++) {
            TargetZone* target = &env->targets[t];
            // Position (3)
            env->observations[obs_idx++] = target->x / env->world_size;
            env->observations[obs_idx++] = target->y / env->world_size;
            env->observations[obs_idx++] = target->z / env->max_height;
            // Has object (1)
            env->observations[obs_idx++] = (float)target->has_object;
        }
        
        // Task info (2)
        env->observations[obs_idx++] = 1.0f - (float)env->current_step / env->max_steps; // Time remaining
        
        // Task progress (ratio of placed objects)
        int placed_count = 0;
        for (int o = 0; o < env->num_objects; o++) {
            if (env->objects[o].is_placed) placed_count++;
        }
        env->observations[obs_idx++] = (float)placed_count / env->num_objects;
    }
}

// Add to log
void add_log(DronePickPlace* env) {
    env->log.episode_length += env->current_step;
    for (int d = 0; d < env->num_drones; d++) {
        env->log.episode_return += env->rewards[d];
    }
    
    // For simplicity, just track binary success for this episode
    // Since we have 1 object, either it was grasped or not, placed or not
    // Check if object was successfully grasped this episode
    env->log.grasp_success += (env->stats.grasp_successes > 0) ? 1.0f : 0.0f;
    
    // Check if object was successfully placed this episode
    env->log.placement_success += (env->stats.placement_successes > 0) ? 1.0f : 0.0f;
    
    env->log.n++;
}

// Required reset function
void c_reset(DronePickPlace* env) {
    env->current_step = 0;
    
    // Reset stats for new episode
    env->stats.grasp_attempts = 0;
    env->stats.grasp_successes = 0;
    env->stats.placement_attempts = 0;
    env->stats.placement_successes = 0;
    
    // Reset object to fixed position (single object)
    for (int o = 0; o < env->num_objects; o++) {
        Object* obj = &env->objects[o];
        // Single object at fixed position, away from target
        // Target is at (1.15, 1.0), so put object on opposite side
        obj->x = 0.5f;  // Left side, away from target at 1.15
        obj->y = 1.0f;  // Center Y
        obj->z = 0.1f; // On ground
        obj->vx = obj->vy = obj->vz = 0;
        obj->radius = 0.1f;  // Larger for easier grasping
        obj->is_grasped = 0;
        obj->is_placed = 0;
    }
    
    // Reset drones - START AWAY FROM TARGET ZONE
    for (int d = 0; d < env->num_drones; d++) {
        Drone* drone = &env->drones[d];
        
        // Start to the LEFT of object, away from target zone
        // Object is at (1.0, 1.0), target is at (1.15, 1.0)
        // So we start at (0.7, 1.0) - must move RIGHT to get to object
        drone->x = 0.7f;  // Left of object (object at 1.0, target at 1.15)
        drone->y = 1.0f;  // Same Y as object  
        drone->z = 0.25f;  // Above ground level
        
        drone->vx = drone->vy = drone->vz = 0;
        drone->wx = drone->wy = drone->wz = 0;
        drone->yaw = 0;  // Fixed orientation facing forward
        drone->pitch = 0;
        drone->roll = 0;
        // Initialize quaternion from yaw
        drone->qw = 1.0f;  // No rotation
        drone->qx = 0.0f;
        drone->qy = 0.0f;
        drone->qz = 0.0f;
        drone->gripper_open = 1.0f;
        drone->state = STATE_SEARCHING;
        drone->ticks_without_progress = 0;
    }
    
    // Reset target zone - VERY close to starting position
    for (int t = 0; t < env->num_targets; t++) {
        TargetZone* target = &env->targets[t];
        // Target EXTREMELY close - almost overlapping
        target->x = 1.15f;  // Only 0.15 units to the right of object
        target->y = 1.0f;  // Same Y as object
        target->z = 0.1f;  // Same height as object for easy placement
        target->radius = 0.3f;  // Very large radius for easy placement
        target->has_object = 0;
    }
    
    // Clear rewards and terminals
    memset(env->rewards, 0, env->num_drones * sizeof(float));
    memset(env->terminals, 0, env->num_drones * sizeof(unsigned char));
    
    // Compute initial observations
    compute_observations(env);
}

// Required step function
void c_step(DronePickPlace* env) {
    env->current_step++;
    
    // Clear rewards (do this every step)
    memset(env->rewards, 0, env->num_drones * sizeof(float));
    
    // Update drone physics
    for (int d = 0; d < env->num_drones; d++) {
        update_drone_physics(env, d);
        
        // Penalty for no progress
        env->drones[d].ticks_without_progress++;
        if (env->drones[d].ticks_without_progress > 500) {
            env->rewards[d] = -0.1f;  // Small negative for no progress
        }
    }
    
    // Update grasping and placement
    update_grasping(env);
    update_placement(env);
    
    // Update physics for objects
    for (int o = 0; o < env->num_objects; o++) {
        Object* obj = &env->objects[o];
        
        if (obj->is_grasped) {
            // Object follows the drone that grasped it
            // Find which drone has it
            for (int d = 0; d < env->num_drones; d++) {
                Drone* drone = &env->drones[d];
                if (drone->state == STATE_TRANSPORTING) {
                    // Move object with drone (slightly below it)
                    obj->x = drone->x;
                    obj->y = drone->y;
                    obj->z = drone->z - 0.15f;  // Carry below drone
                    obj->vx = drone->vx;
                    obj->vy = drone->vy;
                    obj->vz = drone->vz;
                    break;
                }
            }
        } else if (!obj->is_placed) {
            // Apply gravity to free objects
            obj->vz += env->gravity * env->dt;
            
            // Apply drag
            obj->vx *= 0.98f;
            obj->vy *= 0.98f;
            obj->vz *= 0.98f;
            
            // Update position
            obj->x += obj->vx * env->dt;
            obj->y += obj->vy * env->dt;
            obj->z += obj->vz * env->dt;
            
            // Ground collision
            if (obj->z < 0.1f) {
                obj->z = 0.1f;
                obj->vz = 0;
                obj->vx *= 0.8f;  // Friction
                obj->vy *= 0.8f;
            }
            
            // Boundary constraints
            obj->x = fmaxf(obj->radius, fminf(env->world_size - obj->radius, obj->x));
            obj->y = fmaxf(obj->radius, fminf(env->world_size - obj->radius, obj->y));
        }
    }
    
    // Compute rewards based on distance to objectives
    for (int d = 0; d < env->num_drones; d++) {
        Drone* drone = &env->drones[d];
        
        // SPARSE REWARDS - only significant milestones get rewarded
        Object* obj = &env->objects[0];
        TargetZone* target = &env->targets[0];
        
        if (!obj->is_placed) {
            if (!obj->is_grasped) {
                // PHASE 1: Need to pick up object
                float dist_to_obj = distance3d(drone->x, drone->y, drone->z,
                                              obj->x, obj->y, obj->z);
                
                // Small negative reward for time pressure
                float reward = -0.001f;
                
                // Small reward only when VERY close and moving toward object
                if (dist_to_obj < 0.3f) {
                    // Check if drone is moving toward object (dot product of velocity and direction)
                    float dx = obj->x - drone->x;
                    float dy = obj->y - drone->y;
                    float dz = obj->z - drone->z;
                    float dist_sq = dx*dx + dy*dy + dz*dz;
                    if (dist_sq > 0.0001f) {
                        // Normalize direction
                        float inv_dist = 1.0f / sqrtf(dist_sq);
                        dx *= inv_dist;
                        dy *= inv_dist;
                        dz *= inv_dist;
                        
                        // Dot product with velocity
                        float vel_toward = drone->vx * dx + drone->vy * dy + drone->vz * dz;
                        if (vel_toward > 0.01f) {
                            reward = 0.01f;  // Small reward for approaching when close
                        }
                    }
                }
                
                env->rewards[d] = reward;
                
            } else {
                // PHASE 2: Object is grasped - need to transport
                float dist_to_target = distance3d(drone->x, drone->y, drone->z,
                                                 target->x, target->y, target->z);
                
                // Small negative reward for time pressure
                float reward = -0.001f;
                
                // Check if drone is moving toward target
                float dx = target->x - drone->x;
                float dy = target->y - drone->y;
                float dz = target->z - drone->z;
                float dist_sq = dx*dx + dy*dy + dz*dz;
                if (dist_sq > 0.0001f) {
                    // Normalize direction
                    float inv_dist = 1.0f / sqrtf(dist_sq);
                    dx *= inv_dist;
                    dy *= inv_dist;
                    dz *= inv_dist;
                    
                    // Dot product with velocity
                    float vel_toward = drone->vx * dx + drone->vy * dy + drone->vz * dz;
                    if (vel_toward > 0.01f) {
                        reward = 0.02f;  // Slightly higher reward for carrying toward target
                    }
                }
                
                env->rewards[d] = reward;
            }
        } else {
            // Task completed - no additional reward (already got placement reward)
            env->rewards[d] = 0.0f;
        }
    }
    
    // Check termination conditions
    int all_placed = 1;
    for (int o = 0; o < env->num_objects; o++) {
        if (!env->objects[o].is_placed) {
            all_placed = 0;
            break;
        }
    }
    
    if (all_placed || env->current_step >= env->max_steps) {
        for (int d = 0; d < env->num_drones; d++) {
            env->terminals[d] = 1;
            if (all_placed) {
                env->rewards[d] = 1.0f; // Max reward for completing task
            }
        }
        add_log(env);
        c_reset(env);
    }
    
    // Update observations
    compute_observations(env);
}

// Required render function
void c_render(DronePickPlace* env) {
    if (!IsWindowReady()) {
        InitWindow(800, 600, "Drone Pick & Place Environment");
        SetTargetFPS(30);
        
        // Initialize client if needed
        if (env->client == NULL) {
            env->client = calloc(1, sizeof(Client));
            // Better camera angle - isometric view
            env->client->camera.position = (Vector3){env->world_size * 1.5f, -env->world_size * 0.8f, env->world_size * 1.2f};
            env->client->camera.target = (Vector3){env->world_size/2, env->world_size/2, 0.3f};
            env->client->camera.up = (Vector3){0, 0, 1};
            env->client->camera.fovy = 60.0f;
            env->client->camera.projection = CAMERA_PERSPECTIVE;
        }
    }
    
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    
    BeginDrawing();
    ClearBackground((Color){50, 50, 60, 255});
    
    BeginMode3D(env->client->camera);
    
    // Draw ground plane
    DrawCube((Vector3){env->world_size/2, env->world_size/2, -0.01f}, 
             env->world_size, env->world_size, 0.02f, (Color){80, 80, 90, 255});
    
    // Draw grid lines for reference (make them subtle)
    for (int i = 0; i <= 10; i++) {
        float pos = i * env->world_size / 10.0f;
        DrawLine3D((Vector3){pos, 0, 0}, (Vector3){pos, env->world_size, 0}, (Color){100, 100, 110, 150});
        DrawLine3D((Vector3){0, pos, 0}, (Vector3){env->world_size, pos, 0}, (Color){100, 100, 110, 150});
    }
    
    // Draw drones
    for (int d = 0; d < env->num_drones; d++) {
        Drone* drone = &env->drones[d];
        // Drone body color indicates gripper state
        Color drone_color = drone->gripper_open > 0.5f ? (Color){100, 150, 255, 255} : (Color){100, 255, 150, 255};
        DrawCube((Vector3){drone->x, drone->y, drone->z}, 0.15f, 0.15f, 0.08f, drone_color);
        
        // Draw propellers (simplified)
        DrawCube((Vector3){drone->x, drone->y, drone->z + 0.05f}, 0.25f, 0.02f, 0.01f, (Color){50, 50, 50, 200});
        DrawCube((Vector3){drone->x, drone->y, drone->z + 0.05f}, 0.02f, 0.25f, 0.01f, (Color){50, 50, 50, 200});
        
        // Draw heading indicator
        Vector3 front = {drone->x + cosf(drone->yaw) * 0.15f,
                        drone->y + sinf(drone->yaw) * 0.15f,
                        drone->z};
        DrawLine3D((Vector3){drone->x, drone->y, drone->z}, front, RED);
    }
    
    // Draw objects as cubes for better visibility
    for (int o = 0; o < env->num_objects; o++) {
        Object* obj = &env->objects[o];
        Color obj_color = obj->is_placed ? GREEN : (obj->is_grasped ? YELLOW : RED);
        // Draw cube with size = 2 * radius for each dimension
        float cube_size = obj->radius * 2;
        DrawCube((Vector3){obj->x, obj->y, obj->z}, cube_size, cube_size, cube_size, obj_color);
        // Draw cube wires for better visibility
        DrawCubeWires((Vector3){obj->x, obj->y, obj->z}, cube_size, cube_size, cube_size, BLACK);
    }
    
    // Draw target zones
    for (int t = 0; t < env->num_targets; t++) {
        TargetZone* target = &env->targets[t];
        // Draw target zones more visibly
        Color zone_color = target->has_object ? (Color){50, 255, 50, 150} : (Color){255, 200, 50, 100};
        DrawCylinderEx((Vector3){target->x, target->y, 0}, 
                       (Vector3){target->x, target->y, 0.03f},
                       target->radius, target->radius, 12, zone_color);
        // Draw target zone outline
        DrawCylinderWiresEx((Vector3){target->x, target->y, 0}, 
                            (Vector3){target->x, target->y, 0.03f},
                            target->radius, target->radius, 12, (Color){255, 255, 255, 200});
    }
    
    EndMode3D();
    
    // Draw HUD
    DrawText(TextFormat("Step: %d/%d", env->current_step, env->max_steps), 10, 10, 20, WHITE);
    int placed = 0;
    for (int o = 0; o < env->num_objects; o++) {
        if (env->objects[o].is_placed) placed++;
    }
    DrawText(TextFormat("Placed: %d/%d", placed, env->num_objects), 10, 35, 20, WHITE);
    
    // Display reward for first drone
    if (env->num_drones > 0) {
        DrawText(TextFormat("Reward: %.3f", env->rewards[0]), 10, 60, 20, YELLOW);
        
        // Show drone state
        Drone* drone = &env->drones[0];
        const char* state_str = "UNKNOWN";
        switch(drone->state) {
            case STATE_SEARCHING: state_str = "SEARCHING"; break;
            case STATE_APPROACHING: state_str = "APPROACHING"; break;
            case STATE_GRASPING: state_str = "GRASPING"; break;
            case STATE_TRANSPORTING: state_str = "TRANSPORTING"; break;
            case STATE_PLACING: state_str = "PLACING"; break;
        }
        DrawText(TextFormat("State: %s", state_str), 10, 85, 20, SKYBLUE);
        
        // Show object status
        if (env->num_objects > 0) {
            Object* obj = &env->objects[0];
            DrawText(TextFormat("Object: %s", obj->is_grasped ? "GRASPED" : "FREE"), 10, 110, 20, GREEN);
        }
    }
    
    EndDrawing();
}

// Required close function
void c_close(DronePickPlace* env) {
    if (env->drones) {
        free(env->drones);
        env->drones = NULL;
    }
    if (env->objects) {
        free(env->objects);
        env->objects = NULL;
    }
    if (env->targets) {
        free(env->targets);
        env->targets = NULL;
    }
    if (env->client) {
        free(env->client);
        env->client = NULL;
    }
    if (IsWindowReady()) {
        CloseWindow();
    }
}