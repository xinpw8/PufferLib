# cy_racing.pyx
# cython: language_level=3
# distutils: language=c

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport free, malloc

cdef extern from "racing.h":
    ctypedef struct EnemyCar:
        float rear_bumper_y
        int lane
        int active

    ctypedef struct CEnduro:
        float* observations
        unsigned char* actions
        float* rewards
        unsigned char* terminals
        float* player_x_y
        float* other_cars_x_y
        int* other_cars_active
        unsigned int* score_day
        int frameskip
        float width
        float height
        float player_width
        float player_height
        float other_car_width
        float other_car_height
        float player_speed
        float base_car_speed
        float max_player_speed
        float min_player_speed
        float speed_increment
        int num_enemy_cars
        EnemyCar* enemy_cars
        int max_score
        float front_bumper_y
        float left_distance_to_edge
        float right_distance_to_edge
        float speed
        int dayNumber
        int cars_passed
        int carsRemainingLo
        int carsRemainingHi
        int throttleValue
        float reward
        int done

    ctypedef struct Client

    void init_env(CEnduro *env, int num_enemy_cars)
    void allocate(CEnduro *env, int num_enemy_cars)
    void free_initialized(CEnduro *env)
    void free_allocated(CEnduro *env)
    void compute_observations(CEnduro *env, int num_enemy_cars)
    void spawn_enemy_cars(CEnduro *env, int num_enemy_cars)
    void step(CEnduro *env, int action, int num_enemy_cars)
    void apply_weather_conditions(CEnduro *env)
    void reset_env(CEnduro *env, int num_enemy_cars)
    Client *make_client(CEnduro *env)
    void render(Client *client, CEnduro *env, int num_enemy_cars)
    void render_hud(CEnduro *env)
    void close_client(Client *client)

cdef class CyEnduro:
    cdef CEnduro *env
    cdef Client *client
    cdef int num_enemy_cars
    cdef bint render_enabled  # Whether rendering is enabled

    def __cinit__(self, 
                  cnp.ndarray[cnp.float32_t, ndim=2] obs,  # 2D array for observations
                  cnp.ndarray[cnp.uint8_t, ndim=1] actions,  # 1D array for actions
                  cnp.ndarray[cnp.float32_t, ndim=1] rewards,  # 1D array for rewards
                  cnp.ndarray[cnp.uint8_t, ndim=1] terminals,  # 1D array for terminals
                  cnp.ndarray[cnp.float32_t, ndim=1] player_x_y,  # 1D array for player's (x, y)
                  cnp.ndarray[cnp.float32_t, ndim=2] other_cars_x_y,  # 2D array for other cars' positions
                  cnp.ndarray[cnp.int32_t, ndim=1] other_cars_active,  # 1D array for active cars
                  cnp.ndarray[cnp.uint32_t, ndim=1] score_day,  # 1D array for score/day
                  float width, float height, float player_width, float player_height, 
                  float other_car_width, float other_car_height, 
                  float player_speed, float base_car_speed, float max_player_speed, 
                  float min_player_speed, float speed_increment, int max_score, 
                  int num_enemy_cars, bint render_enabled):  # Add the render_enabled flag
        self.render_enabled = render_enabled  # Set whether rendering is enabled
        
        # Allocate memory for CEnduro structure
        self.env = <CEnduro*>malloc(sizeof(CEnduro))
        if self.env is NULL:
            raise MemoryError("Failed to allocate CEnduro")
        
        # Allocate memory for the environment with the number of enemy cars
        allocate(self.env, num_enemy_cars)
        
        # Set the environment parameters
        self.env.observations = <float *>obs.data
        self.env.actions = <unsigned char *>actions.data
        self.env.rewards = <float *>rewards.data
        self.env.terminals = <unsigned char *>terminals.data
        self.env.player_x_y = <float *>player_x_y.data
        self.env.other_cars_x_y = <float *>other_cars_x_y.data
        self.env.other_cars_active = <int *>other_cars_active.data
        self.env.score_day = <unsigned int *>score_day.data
        self.env.width = width
        self.env.height = height
        self.env.player_width = player_width
        self.env.player_height = player_height
        self.env.other_car_width = other_car_width
        self.env.other_car_height = other_car_height
        self.env.player_speed = player_speed
        self.env.base_car_speed = base_car_speed
        self.env.max_player_speed = max_player_speed
        self.env.min_player_speed = min_player_speed
        self.env.speed_increment = speed_increment
        self.env.max_score = max_score

        # Create the client for rendering (optional)
        self.client = NULL
        if render_enabled:
            self.client = make_client(self.env)
            if self.client is NULL:
                free_allocated(self.env)
                free(self.env)
                self.env = NULL
                raise MemoryError("Failed to create client")


    def __dealloc__(self):
        if self.client is not NULL:
            close_client(self.client)
        if self.env is not NULL:
            free_allocated(self.env)
            free(self.env)

    def reset(self):
        reset_env(self.env, self.num_enemy_cars)
        return self.get_observation()

    def step(self, int action):
        step(self.env, action, self.num_enemy_cars)
        cdef bint done = self.env.done != 0
        return self.get_observation(), self.env.reward, done, {}

    def render(self):
        if self.render_enabled:
            render(self.client, self.env, self.num_enemy_cars)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL
        free_initialized(self.env)

    def get_observation(self):
        cdef int obs_size = 5 + 2 * self.num_enemy_cars  # Adjust this based on your actual observation size
        return [self.env.observations[i] for i in range(obs_size)]

    @property
    def score(self):
        return self.env.throttleValue  # Using throttle as a score placeholder

    @property
    def day(self):
        return self.env.dayNumber
