# cy_puffer_enduro.pyx
# cython: language_level=3

cimport numpy as cnp
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset
from libc.time cimport time
from random import SystemRandom

rng = SystemRandom()

cdef extern from "puffer_enduro.h":
    # Structures
    ctypedef struct Log:
        float episode_return
        float episode_length
        float score
        float reward
        float stay_on_road_reward
        float passed_cars
        float passed_by_enemy
        int cars_to_pass
        float days_completed
        float days_failed
        float collisions_player_vs_car
        float collisions_player_vs_road

    ctypedef struct LogBuffer:
        Log* logs
        int length
        int idx

    ctypedef int size_t

    ctypedef struct Car:
        int lane
        float y
        float last_y
        int passed
        int colorIndex
        
    # ctypedef struct Enduro:
    #     float* observations
    #     int* actions
    #     float* rewards
    #     unsigned char* terminals
    #     unsigned char* truncateds
    #     LogBuffer* log_buffer
    #     size_t obs_size
    #     int num_envs

    ctypedef struct Enduro:
        float* observations
        int* actions
        float* rewards
        unsigned char* terminals
        unsigned char* truncateds
        LogBuffer* log_buffer
        Log log
        size_t obs_size
        int num_envs
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
        Car enemyCars[10]  # Adjust MAX_ENEMIES as needed
        float initial_player_x
        float road_scroll_offset
        int current_curve_direction
        float current_curve_factor
        float target_curve_factor
        float current_step_threshold
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
        float dayTransitionTimes[16]  # Adjust NUM_BACKGROUND_TRANSITIONS as needed
        int dayTimeIndex
        int currentDayTimeIndex
        int previousDayTimeIndex
        unsigned int rng_state
        unsigned int index
        int reset_count

    ctypedef struct Client:
        float width
        float height
        Enduro gameState

    # Function prototypes
    LogBuffer* allocate_logbuffer(int size)
    void free_logbuffer(LogBuffer* buffer)
    Log aggregate_and_clear(LogBuffer* logs)
    void init(Enduro* env, int seed, int env_index)
    void reset(Enduro* env)
    void c_step(Enduro* env)
    void c_render(Client* client, Enduro* env)
    Client* make_client(Enduro* env)
    void close_client(Client* client, Enduro* env)

# Define Cython wrapper class
cdef class CyEnduro:
    cdef:
        Enduro* envs
        LogBuffer* logs
        Client* client
        int num_envs

    def __init__(self,
                 float[:, :] observations,
                 int[:] actions,
                 float[:] rewards,
                 cnp.uint8_t[:] terminals,
                 cnp.uint8_t[:] truncateds,
                 int num_envs):
        """
        Initialize the CyEnduro environment wrapper.

        Parameters:
            observations: The observation buffer.
            actions: The action buffer.
            rewards: The reward buffer.
            terminals: The terminal state buffer.
            truncateds: The truncated state buffer.
            num_envs: The number of environments.
        """
        cdef int i
        cdef long t
        self.num_envs = num_envs

        # Allocate memory for environments
        self.envs = <Enduro*>calloc(num_envs, sizeof(Enduro))
        if not self.envs:
            raise MemoryError("Failed to allocate memory for environments")

        # Allocate memory for logs
        self.logs = allocate_logbuffer(num_envs)
        if not self.logs:
            free(self.envs)
            raise MemoryError("Failed to allocate memory for logs")

        # Generate a unique seed using high-resolution time and environment index
        from time import time as py_time  # Python time module for high-resolution time

        for i in range(num_envs):
            unique_seed = rng.randint(0, 2**32 - 1) & 0x7FFFFFFF
            memset(&self.envs[i], 0, sizeof(Enduro))
            self.envs[i].observations = &observations[i, 0]
            self.envs[i].actions = &actions[i]
            self.envs[i].rewards = &rewards[i]
            self.envs[i].terminals = &terminals[i]
            self.envs[i].truncateds = &truncateds[i]
            self.envs[i].log_buffer = self.logs
            self.envs[i].obs_size = observations.shape[1]

            if i % 100 == 0:
                print(f"Initializing environment #{i} with seed {unique_seed}")

            init(&self.envs[i], unique_seed, i)

    def reset(self):
        """
        Reset all environments.
        """
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        """
        Perform one step in all environments.
        """
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self):
        """
        Render the environment.
        """
        if not self.client:
            self.client = make_client(&self.envs[0])
        c_render(self.client, &self.envs[0])

    def close(self):
        """
        Clean up resources.
        """
        if self.client:
            close_client(self.client, &self.envs[0])
        if self.envs:
            free(self.envs)
        if self.logs:
            free_logbuffer(self.logs)

    def log(self):
        """
        Aggregate and return logs.
        """
        cdef Log log = aggregate_and_clear(self.logs)
        return {
            'episode_return': log.episode_return,
            'episode_length': log.episode_length,
            'score': log.score,
            'reward': log.reward,
            'stay_on_road_reward': log.stay_on_road_reward,
            'passed_cars': log.passed_cars,
            'passed_by_enemy': log.passed_by_enemy,
            'cars_to_pass': log.cars_to_pass,
            'days_completed': log.days_completed,
            'days_failed': log.days_failed,
            'collisions_player_vs_car': log.collisions_player_vs_car,
            'collisions_player_vs_road': log.collisions_player_vs_road,
        }
