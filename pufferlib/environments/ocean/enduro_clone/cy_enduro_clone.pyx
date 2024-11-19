cimport numpy as cnp
from libc.stdlib cimport calloc, free
from libc.stdint cimport int32_t
import os

cdef extern from "enduro_clone.h":
    int TARGET_FPS = 60
    int LOG_BUFFER_SIZE = 4096
    int SCREEN_WIDTH = 160
    int SCREEN_HEIGHT = 210
    int PLAYABLE_AREA_TOP = 0
    int PLAYABLE_AREA_BOTTOM = 154
    int PLAYABLE_AREA_LEFT = 8
    int PLAYABLE_AREA_RIGHT = 160
    int ACTION_HEIGHT = (PLAYABLE_AREA_BOTTOM - PLAYABLE_AREA_TOP)
    int CAR_WIDTH = 16
    int CAR_HEIGHT = 11
    int MAX_ENEMIES = 10
    int CRASH_NOOP_DURATION_CAR_VS_CAR = 90
    int CRASH_NOOP_DURATION_CAR_VS_ROAD = 20
    int INITIAL_CARS_TO_PASS = 200
    int VANISHING_POINT_X = 86
    int VANISHING_POINT_Y = 52
    float VANISHING_POINT_TRANSITION_SPEED = 1.0
    float CURVE_TRANSITION_SPEED = 0.05
    float LOGICAL_VANISHING_Y = (VANISHING_POINT_Y + 12)
    float INITIAL_PLAYER_X = 86.0
    float PLAYER_MAX_Y = (ACTION_HEIGHT - CAR_HEIGHT)
    float PLAYER_MIN_Y = (ACTION_HEIGHT - CAR_HEIGHT - 9)
    float ACCELERATION_RATE = 0.2
    float DECELERATION_RATE = 0.1
    float MIN_SPEED = -2.5
    float MAX_SPEED = 7.5
    float ENEMY_CAR_SPEED = 0.1
    int CURVE_STRAIGHT = 0
    int CURVE_LEFT = -1
    int CURVE_RIGHT = 1
    int NUM_LANES = 3
    float PLAYER_MIN_X = 65.5
    float PLAYER_MAX_X = 91.5
    float ROAD_LEFT_OFFSET = 50.0
    float ROAD_RIGHT_OFFSET = 51.0
    float VANISHING_POINT_X_LEFT = 110.0
    float VANISHING_POINT_X_RIGHT = 62.0
    float CURVE_VANISHING_POINT_SHIFT = 55.0
    float CURVE_PLAYER_SHIFT_FACTOR = 0.025
    float WIGGLE_AMPLITUDE = 10.0
    float WIGGLE_SPEED = 10.1
    float WIGGLE_LENGTH = 26.0
    
    ctypedef struct Log:
        float episode_return
        float episode_length
        float score
        
    ctypedef struct LogBuffer:
        Log* logs
        int length
        int idx
        
    ctypedef struct Car:
        int lane
        float y
        int passed
        int colorIndex
        
    ctypedef struct Enduro:
        float* observations
        int* actions
        float* rewards
        unsigned char* terminals
        unsigned char* truncateds
        LogBuffer* log_buffer
        Log log
        float width
        float height
        float car_width
        float car_height
        int max_enemies
        float elapsedTime
        int initial_cars_to_pass
        float min_speed
        float max_speed
        float player_x
        float player_y
        float speed
        int score
        int day
        int lane
        int step_count
        int numEnemies
        int carsToPass
        float collision_cooldown_car_vs_car
        float collision_cooldown_car_vs_road
        float collision_invulnerability_timer
        int drift_direction
        float action_height
        Car enemyCars[10]
        float initial_player_x
        float road_scroll_offset
        int current_curve_direction
        float current_curve_factor
        float target_curve_factor
        float target_vanishing_point_x
        float current_vanishing_point_x
        float base_target_vanishing_point_x
        float vanishing_point_x
        float base_vanishing_point_x
        float t_p
        float wiggle_y
        float wiggle_speed
        float wiggle_length
        float wiggle_amplitude
        unsigned char wiggle_active
        int currentGear
        float gearSpeedThresholds[4]
        float gearAccelerationRates[4]
        float gearTimings[4]
        float gearElapsedTime
        int currentStage
        float enemySpawnTimer
        float enemySpawnInterval
        float last_road_left
        float last_road_right
        int closest_edge_lane
        int last_spawned_lane
        float totalAccelerationTime
        float parallaxFactor
        unsigned char victoryAchieved
        int flagTimer
        unsigned char showLeftFlag
        int victoryDisplayTimer
        float backgroundTransitionTimes[16]
        int backgroundIndex
        int currentBackgroundIndex
        int previousBackgroundIndex
    
    ctypedef enum Action:
        ACTION_NOOP = 0
        ACTION_FIRE = 1
        ACTION_RIGHT = 2
        ACTION_LEFT = 3
        ACTION_DOWN = 4
        ACTION_DOWNRIGHT = 5
        ACTION_DOWNLEFT = 6
        ACTION_RIGHTFIRE = 7
        ACTION_LEFTFIRE = 8

    # Function declarations
    LogBuffer* allocate_logbuffer(int size)
    void free_logbuffer(LogBuffer* buffer)
    void add_log(LogBuffer* logs, Log * log)
    Log aggregate_and_clear(LogBuffer* logs)
    void init(Enduro* env)
    void allocate(Enduro* env)
    void free_allocated(Enduro* env)
    void reset(Enduro* env)
    unsigned char check_collision(Enduro* env, Car* car)
    int get_player_lane(Enduro* env)
    float get_car_scale(float y)
    void add_enemy_car(Enduro* env)
    void update_vanishing_point(Enduro* env, float offset)
    void accelerate(Enduro* env)
    void steppy(Enduro* env)
    void update_road_curve(Enduro* env)
    float quadratic_bezier(float bottom_x, float control_x, float top_x, float t)
    float road_edge_x(Enduro* env, float y, float offset, unsigned char left)
    float car_x_in_lane(Enduro* env, int lane, float y)
    void updateVictoryEffects(Enduro* env)
    void updateBackground(Enduro* env)
    void compute_observations(Enduro* env)


cdef class CyEnduro:
    cdef:
        Enduro* envs
        LogBuffer* logs
        int num_envs
        float width
        float height
        float car_width
        float car_height
        float min_speed
        float max_speed
        int initial_cars_to_pass

    def __init__(self, float[:, :] observations, int[:] actions,
                 float[:] rewards, unsigned char[:] terminals,
                 unsigned char[:] truncateds, int num_envs,
                 float width, float height,
                 float car_width, float car_height, float min_speed,
                 float max_speed, int initial_cars_to_pass):

        self.num_envs = num_envs
        print("Creating", num_envs, "environments")
        self.envs = <Enduro*> calloc(num_envs, sizeof(Enduro))
        print("Allocated memory for", num_envs, "environments")
        self.logs = allocate_logbuffer(num_envs)

        cdef int i
        cdef Enduro* env
        for i in range(num_envs):
            env = &self.envs[i]
            env.observations = &observations[i, 0]
            env.actions = &actions[i]
            env.rewards = &rewards[i]
            env.terminals = &terminals[i]
            env.truncateds = &truncateds[i]
            env.log_buffer = self.logs
            env.width = width
            env.height = height
            env.car_width = car_width
            env.car_height = car_height
            env.min_speed = min_speed
            env.max_speed = max_speed
            env.initial_cars_to_pass = initial_cars_to_pass
            env.max_enemies = MAX_ENEMIES
            init(env)
            
    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])
            
    def step(self):
        cdef int i
        for i in range(self.num_envs):
            steppy(&self.envs[i])
            
    def close(self):
        free_allocated(self.envs)
        free_logbuffer(self.logs)
        free(self.envs)
        
    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log