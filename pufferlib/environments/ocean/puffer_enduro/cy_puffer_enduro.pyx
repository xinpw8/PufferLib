# cy_puffer_enduro.pyx
# cython: language_level=3

cimport numpy as cnp
from libc.stdlib cimport malloc, calloc, free
from libc.stdint cimport int32_t, uint8_t

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

    ctypedef struct Car:
        int lane
        float y
        float last_y
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

    ctypedef struct Client:
        float width
        float height
        Enduro gameState

    # Function prototypes
    LogBuffer* allocate_logbuffer(int size)
    void free_logbuffer(LogBuffer* buffer)
    Log aggregate_and_clear(LogBuffer* logs)

    void allocate(Enduro* env)
    void init(Enduro* env)
    void free_allocated(Enduro* env)
    void reset(Enduro* env)
    void c_step(Enduro* env)
    Client* make_client(Enduro* env)
    void close_client(Client* client, Enduro* env)
    void c_render(Client* client, Enduro* env)

# Define Cython wrapper class
cdef class CyEnduro:
    cdef:
        Enduro* envs
        Client* client
        LogBuffer* logs
        int num_envs

    def __init__(self,
                 float[:, :] observations,
                 int[:] actions,
                 float[:] rewards,
                 uint8_t[:] terminals,
                 uint8_t[:] truncateds,
                 int num_envs):
        self.num_envs = num_envs
        self.client = NULL
        self.envs = <Enduro*>calloc(num_envs, sizeof(Enduro))
        if not self.envs:
            raise MemoryError("Failed to allocate memory for environments")
        self.logs = allocate_logbuffer(num_envs)
        if not self.logs:
            free(self.envs)
            raise MemoryError("Failed to allocate memory for logs")

        cdef int i
        for i in range(num_envs):
            self.envs[i] = Enduro(
                observations=&observations[i, 0],
                actions=&actions[i],
                rewards=&rewards[i],
                terminals=&terminals[i],
                truncateds=&truncateds[i],
                log_buffer=self.logs,
            )
            init(&self.envs[i])

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self):
        cdef Enduro* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(env)

        c_render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client, self.envs)
            self.client = NULL

        free(self.envs)
        free_logbuffer(self.logs)

    def log(self):
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


    # def log(self):
    #     cdef Log log = aggregate_and_clear(self.logs)
    #     return log
    #     # return {
    #     #     'episode_return': log.episode_return,
    #     #     'episode_length': log.episode_length,
    #     #     'score': log.score,
    #     #     'passed_cars': log.passed_cars,
    #     #     'days_completed': log.days_completed,
    #     #     'days_failed': log.days_failed,
    #     #     'collisions_player_vs_car': log.collisions_player_vs_car,
    #     #     'collisions_player_vs_road': log.collisions_player_vs_road
    #     # }
