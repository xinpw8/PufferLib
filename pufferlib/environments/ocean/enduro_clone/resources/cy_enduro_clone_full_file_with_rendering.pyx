# cy_enduro_clone.pyx

from libc.stdlib cimport calloc, free
from libc.stdint cimport int32_t, uint8_t

cimport numpy as cnp
import numpy as np
from numpy cimport ndarray, float32_t, int32_t, uint8_t

cdef int MAX_ENEMIES = 10

cdef extern from "enduro_clone.h":
    int LOG_BUFFER_SIZE

    # Constants and enums from enduro_clone.h
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

    ctypedef enum GameStage:
        GAME_STAGE_DAY_START
        GAME_STAGE_NIGHT
        GAME_STAGE_GRASS_AFTER_SNOW

    ctypedef struct Log:
        float episode_return
        float episode_length
        float score

    ctypedef struct LogBuffer:
        pass

    LogBuffer* allocate_logbuffer(int size)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Car:
        int lane
        float y
        int passed
        int colorIndex

    ctypedef struct GameState:
        pass  # Assuming GameState is not needed in Cython

    ctypedef struct Enduro:
        float* observations
        int32_t actions
        float* rewards
        unsigned char* terminals
        unsigned char* truncateds
        LogBuffer* log_buffer
        Log log

        float width
        float height
        float hud_height
        float car_width
        float car_height
        int max_enemies
        float crash_noop_duration
        float day_length
        int initial_cars_to_pass
        float min_speed
        float max_speed

        float elapsedTime
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

        # Road curve variables
        int current_curve_direction
        float current_curve_factor
        float target_curve_factor
        float target_vanishing_point_x
        float current_vanishing_point_x
        float base_target_vanishing_point_x
        float vanishing_point_x
        float base_vanishing_point_x
        float t_p

        # Roadside wiggle effect
        float wiggle_y
        float wiggle_speed
        float wiggle_length
        float wiggle_amplitude
        unsigned char wiggle_active  # Changed from 'bool'

        # Player car acceleration
        int currentGear
        float gearSpeedThresholds[4]
        float gearAccelerationRates[4]
        float gearTimings[4]
        float gearElapsedTime

        # Enemy spawning
        GameStage currentStage
        float enemySpawnTimer
        float enemySpawnInterval

        # Logging
        float last_road_left
        float last_road_right
        int closest_edge_lane
        int last_spawned_lane
        float totalAccelerationTime

        # Mountain rendering
        float parallaxFactor

        GameState gameState

    ctypedef struct Client:
        pass

    void init(Enduro* env)
    void reset(Enduro* env)
    void step(Enduro* env)

    Client* make_client(Enduro* env)
    void close_client(Client* client)
    void render(Client* client, Enduro* env)

cdef class CyEnduro:
    cdef:
        Enduro* envs
        Client* client
        LogBuffer* logs
        int num_envs
        int obs_size
        object observations
        object actions
        object rewards
        object terminals
        object truncateds

    def __init__(self,
                 ndarray[float32_t, ndim=2] observations,
                 ndarray[int32_t, ndim=1] actions,
                 ndarray[float32_t, ndim=1] rewards,
                 ndarray[uint8_t, ndim=1] terminals,
                 ndarray[uint8_t, ndim=1] truncateds,
                 int num_envs,
                 float width, float height, float hud_height,
                 float car_width, float car_height,
                 int max_enemies,
                 float crash_noop_duration, float day_length,
                 int initial_cars_to_pass, float min_speed, float max_speed):

        cdef int i, j
        cdef float* observations_i_data
        cdef int32_t* actions_i
        cdef float* rewards_i
        cdef unsigned char* terminals_i
        cdef unsigned char* truncateds_i
        cdef Enduro* env

        # Declare variables at the top level
        cdef float totalSpeedRange
        cdef float totalTime
        cdef float cumulativeSpeed
        cdef float gearTime
        cdef float gearSpeedIncrement

        self.num_envs = num_envs
        self.client = NULL
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)
        self.envs = <Enduro*> calloc(num_envs, sizeof(Enduro))

        # Calculate observation size
        self.obs_size = 6 + 2 * max_enemies + 3

        # Assign arrays to class attributes
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.truncateds = truncateds

        # Declare memoryviews for local use
        cdef float32_t[:, :] observations_mv = observations
        cdef int32_t[:] actions_mv = actions
        cdef float32_t[:] rewards_mv = rewards
        cdef uint8_t[:] terminals_mv = terminals
        cdef uint8_t[:] truncateds_mv = truncateds

        for i in range(num_envs):
            observations_i_data = &observations_mv[i, 0]
            actions_i = &actions_mv[i]
            rewards_i = &rewards_mv[i]
            terminals_i = &terminals_mv[i]
            truncateds_i = &truncateds_mv[i]
            env = &self.envs[i]

            # Assign pointers to C arrays
            env.observations = observations_i_data
            env.actions = actions_i[0]
            env.rewards = rewards_i
            env.terminals = terminals_i
            env.truncateds = truncateds_i
            env.log_buffer = self.logs

            # Initialize environment parameters
            env.width = width
            env.height = height
            env.hud_height = hud_height
            env.car_width = car_width
            env.car_height = car_height
            env.max_enemies = max_enemies
            env.initial_cars_to_pass = initial_cars_to_pass
            env.min_speed = min_speed
            env.max_speed = max_speed

            # Initialize player position and speed
            env.elapsedTime = 0.0
            env.player_x = INITIAL_PLAYER_X
            env.player_y = PLAYER_MAX_Y
            env.speed = env.min_speed

            # Initialize game state variables
            env.score = 0
            env.day = 1
            env.lane = 0
            env.step_count = 0
            env.numEnemies = 0
            env.carsToPass = env.initial_cars_to_pass
            env.collision_cooldown_car_vs_car = 0.0
            env.collision_cooldown_car_vs_road = 0.0
            env.collision_invulnerability_timer = 0.0
            env.drift_direction = 0
            env.action_height = ACTION_HEIGHT

            # Initialize enemy cars
            for j in range(max_enemies):
                env.enemyCars[j].lane = 0
                env.enemyCars[j].y = 0.0
                env.enemyCars[j].passed = 0
                env.enemyCars[j].colorIndex = 0

            env.initial_player_x = INITIAL_PLAYER_X
            env.road_scroll_offset = 0.0

            # Road curve variables
            env.current_curve_direction = 0
            env.current_curve_factor = 0.0
            env.target_curve_factor = 0.0
            env.target_vanishing_point_x = VANISHING_POINT_X
            env.current_vanishing_point_x = VANISHING_POINT_X
            env.base_target_vanishing_point_x = VANISHING_POINT_X
            env.vanishing_point_x = VANISHING_POINT_X
            env.base_vanishing_point_x = VANISHING_POINT_X
            env.t_p = 0.0

            # Wiggle effect variables
            env.wiggle_y = VANISHING_POINT_Y
            env.wiggle_speed = WIGGLE_SPEED
            env.wiggle_length = WIGGLE_LENGTH
            env.wiggle_amplitude = WIGGLE_AMPLITUDE
            env.wiggle_active = 1  # Use 1 for True

            # Player car acceleration
            env.currentGear = 0
            env.gearElapsedTime = 0.0

            # Initialize gear timings
            env.gearTimings[0] = 4.0
            env.gearTimings[1] = 2.5
            env.gearTimings[2] = 3.25
            env.gearTimings[3] = 1.5

            # Calculate speed thresholds and acceleration rates
            totalSpeedRange = env.max_speed - env.min_speed
            totalTime = sum(env.gearTimings)

            cumulativeSpeed = env.min_speed
            for j in range(4):
                gearTime = env.gearTimings[j]
                gearSpeedIncrement = totalSpeedRange * (gearTime / totalTime)
                env.gearSpeedThresholds[j] = cumulativeSpeed + gearSpeedIncrement
                env.gearAccelerationRates[j] = gearSpeedIncrement / (gearTime * TARGET_FPS)
                cumulativeSpeed = env.gearSpeedThresholds[j]

            # Enemy spawning
            env.currentStage = GAME_STAGE_DAY_START
            env.enemySpawnTimer = 0.0
            env.enemySpawnInterval = 0.8777
            env.last_spawned_lane = -1
            env.closest_edge_lane = -1

            # Logging
            env.totalAccelerationTime = 0.0

            # Mountain rendering
            env.parallaxFactor = 0.0

            # Initialize GameState if necessary

            # Call init after setting necessary variables
            init(env)

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            # Update actions from the array
            self.envs[i].actions = self.actions[i]
            step(&self.envs[i])

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free(self.envs)
        free_logbuffer(self.logs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return {'episode_return': log.episode_return,
                'episode_length': log.episode_length,
                'score': log.score}
