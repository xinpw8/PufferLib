# cy_racing.pyx
cimport numpy as cnp
from libc.stdlib cimport calloc, free
from libc.stdio cimport printf

cdef extern from "racing.h":
    ctypedef struct CRacingEnv:
        float* observations
        unsigned char* actions
        float* rewards
        unsigned char* dones
        int dayNumber
        int cars_passed
        int tick
        int day_length
        float speed
        int* carsRemainingLo
        int* carsRemainingHi
        float* enemy_car_y
        float* enemy_car_x
        int* enemy_active
        int num_enemy_cars
        int done

    CRacingEnv* init_racing_env(CRacingEnv* env)
    void free_racing_env(CRacingEnv* env)
    void reset(CRacingEnv* env)
    void step(CRacingEnv* env)
    void spawn_enemy_cars(CRacingEnv* env)

cdef class CyRacingEnv:
    cdef:
        CRacingEnv* env

    def __init__(self, cnp.ndarray observations, cnp.ndarray actions,
                 cnp.ndarray rewards, cnp.ndarray terminals):

        self.env = <CRacingEnv*> calloc(1, sizeof(CRacingEnv))

        self.env.observations = <float*> observations.data
        self.env.actions = <unsigned char*> actions.data
        self.env.rewards = <float*> rewards.data
        self.env.dones = <unsigned char*> terminals.data

        init_racing_env(self.env)

    def reset(self):
        reset(self.env)

    def step(self):
        step(self.env)

    def close(self):
        free_racing_env(self.env)
