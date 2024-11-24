# cy_puffer_enduro.pyx

# Declare C dependencies
from libc.stdlib cimport calloc, free
from libc.stdint cimport int32_t, uint8_t
from libc.string cimport memcpy

cimport numpy as cnp
import numpy as np
from numpy cimport ndarray, float32_t, int32_t, uint8_t

# Declare constants
cdef int MAX_ENEMIES = 10

# Import C structs, enums, and functions from the header
cdef extern from "puffer_enduro.h":
    # Constants
    int LOG_BUFFER_SIZE
    int TARGET_FPS
    float INITIAL_PLAYER_X
    float PLAYER_MAX_Y
    float PLAYER_MIN_Y
    float VANISHING_POINT_X
    float VANISHING_POINT_Y
    float WIGGLE_SPEED
    float WIGGLE_LENGTH
    float WIGGLE_AMPLITUDE
    int ACTION_HEIGHT

    # Structs
    ctypedef struct Log:
        float episode_return
        float episode_length
        float score
        float reward
        float passed_cars
        float days_completed
        float days_failed
        float collisions_player_vs_car
        float collisions_player_vs_road

    ctypedef struct LogBuffer:
        Log* logs
        int length
        int idx

    ctypedef struct Car:
        int lane
        float y
        float last_y
        int passed
        int colorIndex

    ctypedef struct GameState:
        pass

    ctypedef struct Client:
        float width
        float height
        GameState gameState

    ctypedef struct Enduro:
        float* observations
        int32_t* actions
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
        float elapsedTimeEnv
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
        float enemySpawnTimer
        float enemySpawnInterval
        float enemySpeed
        unsigned char dayCompleted
        float last_road_left
        float last_road_right
        int closest_edge_lane
        int last_spawned_lane
        float totalAccelerationTime
        float parallaxFactor
        float dayTransitionTimes[16]
        int dayTimeIndex
        int currentDayTimeIndex
        int previousDayTimeIndex

    # Function prototypes
    LogBuffer* allocate_logbuffer(int size)
    void free_logbuffer(LogBuffer* buffer)
    void add_log(LogBuffer* logs, Log* log)
    Log aggregate_and_clear(LogBuffer* logs)
    void allocate(Enduro* env)
    void init(Enduro* env)
    void free_allocated(Enduro* env)
    void reset_round(Enduro* env)
    void reset(Enduro* env)
    unsigned char check_collision(Enduro* env, Car* car)
    int get_player_lane(Enduro* env)
    float get_car_scale(float y)
    void add_enemy_car(Enduro* env)
    void update_vanishing_point(Enduro* env, float offset)
    void accelerate(Enduro* env)
    void c_step(Enduro* env)
    void update_road_curve(Enduro* env)
    float quadratic_bezier(float bottom_x, float control_x, float top_x, float t)
    float road_edge_x(Enduro* env, float y, float offset, unsigned char left)
    float car_x_in_lane(Enduro* env, int lane, float y)
    void compute_observations(Enduro* env)

    # Client functions
    Client* make_client(Client* client)
    void close_client(Client* client, Enduro* env)
    void render_car(Client* client, GameState* gameState)
    void handleEvents(int* running, Enduro* env)
    void initRaylib()
    void loadTextures(GameState* gameState)
    void updateCarAnimation(GameState* gameState)
    void updateScoreboard(GameState* gameState)
    void updateBackground(GameState* gameState)
    void renderBackground(GameState* gameState)
    void renderScoreboard(GameState* gameState)
    void updateMountains(GameState* gameState)
    void renderMountains(GameState* gameState)
    void updateVictoryEffects(GameState* gameState)
    void c_render(Client* client, Enduro* env)
    void cleanup(GameState* gameState)

# Define Cython class to wrap the C Enduro environment

cdef class CyEnduro:
    cdef:
        Enduro* envs
        Log log
        LogBuffer* logs
        int num_envs
        Client* client
        int max_enemies

    def __cinit__(self, 
                float32_t[:, :] observations,
                int32_t[:] actions,
                float32_t[:] rewards,
                uint8_t[:] terminals,
                uint8_t[:] truncateds,
                int num_envs):

        if num_envs <= 0 or num_envs > 1000000:
            raise ValueError("num_envs must be between 1 and 1000000")
        
        # For debugging, set num_envs to a smaller number
        # num_envs = 2  # Uncomment this line for debugging
        self.num_envs = num_envs

        self.envs = <Enduro*> calloc(<size_t>self.num_envs, sizeof(Enduro))
        if not self.envs:
            raise MemoryError("Failed to allocate memory for environments")
            
        self.logs = allocate_logbuffer(self.num_envs)
        if not self.logs:
            free(self.envs)
            raise MemoryError("Failed to allocate memory for logs")

        self.client = <Client*>NULL

        cdef int i
        cdef Enduro* env
        for i in range(self.num_envs):
            env = &self.envs[i]
            env.observations = &observations[i, 0]
            env.actions = <int32_t*>&actions[i]
            env.rewards = &rewards[i]
            env.terminals = &terminals[i]
            env.truncateds = &truncateds[i]
            env.log_buffer = self.logs
            init(env)
                
    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])
            
    def step(self):
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])
            
    def close(self):
        if self.c_envs:
            self.c_envs.close()
        free_logbuffer(self.logs)
        free(self.envs)
    
    def render(self):
        cdef Enduro* env = &self.envs[0]
        if self.client is NULL:
            self.client = make_client(self.client)
        c_render(self.client, env)
        
    def get_log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
