#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
cimport numpy as cnp
from libc.stdlib cimport free

cdef extern from "racing.h":
    ctypedef struct EnemyCar:
        float rear_bumper_y
        int lane
        int active

    int NUM_ENEMY_CARS

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
        int max_score
        float front_bumper_y
        float left_distance_to_edge
        float right_distance_to_edge
        float speed
        int dayNumber
        int carsRemainingLo
        int carsRemainingHi
        int throttleValue
        float reward
        int done

    ctypedef struct Client

    void init_env(CEnduro *env)
    CEnduro* allocate()
    void free_initialized(CEnduro *env)
    void free_allocated(CEnduro *env)
    void compute_observations(CEnduro *env)
    void spawn_enemy_cars(CEnduro *env)
    bint step(CEnduro *env, int action)
    void apply_weather_conditions(CEnduro *env)
    void reset_env(CEnduro *env)
    Client* make_client(CEnduro *env)
    void render(Client *client, CEnduro *env)
    void render_hud(CEnduro *env)
    void close_client(Client *client)

cdef class CyEnduro:
    cdef CEnduro* env
    cdef Client* client

    def __cinit__(self):
        self.env = allocate()
        self.client = make_client(self.env)
    
    def __dealloc__(self):
        if self.client is not NULL:
            close_client(self.client)
        if self.env is not NULL:
            free_allocated(self.env)

    def reset(self):
        reset_env(self.env)
        return self.get_observation()

    def step(self, int action):
        cdef bint done = step(self.env, action)
        return self.get_observation(), self.env.reward, done, {}

    def render(self):
        if self.client == NULL:
            self.client = make_client(self.env)
        render(self.client, self.env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL
        free_initialized(self.env)

    def get_observation(self):
        cdef int obs_size = 10  # Adjust this based on your actual observation size
        return [self.env.observations[i] for i in range(obs_size)]

    @property
    def score(self):
        return self.env.score_day[0]

    @property
    def day(self):
        return self.env.score_day[1]