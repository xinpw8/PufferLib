# cython: language_level=3

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t
import random

cnp.import_array()

cdef struct CRacingEnv:
    float* state  # Pointer to state array (player_x, player_y, speed, day, cars_passed, enemy_car_x, enemy_car_y...)
    int tick
    int num_enemy_cars
    float width, height
    float max_player_speed, min_player_speed

# Initialize the environment
cdef CRacingEnv* init_racing_env(int num_enemy_cars):
    cdef CRacingEnv* env = <CRacingEnv*>malloc(sizeof(CRacingEnv))
    env.state = <float*>malloc((5 + 2 * num_enemy_cars) * sizeof(float))  # Adjust state array size
    env.tick = 0
    env.num_enemy_cars = num_enemy_cars
    env.width = 1.0
    env.height = 1.0
    env.max_player_speed = 0.1
    env.min_player_speed = 0.01
    return env

# Reset environment to initial state
cdef void reset(CRacingEnv* env):
    env.tick = 0
    env.state[0] = 0.5  # player_x
    env.state[1] = 0.9  # player_y
    env.state[2] = 0.0  # speed
    env.state[3] = 0.0  # day
    env.state[4] = 0.0  # cars_passed

    # Set enemy car positions
    for i in range(env.num_enemy_cars):
        env.state[5 + i * 2] = random.random()  # enemy car x
        env.state[6 + i * 2] = random.random()  # enemy car y

# Perform one step in the environment
cdef tuple step(CRacingEnv* env, object action):
    cdef int action_int
    cdef float accel_x = 0.0, accel_y = 0.0

    action_int = <int>action
    if action_int == 1:  # ACCEL
        accel_x = 0.1
    elif action_int == 2:  # DECEL
        accel_x = -0.1
    elif action_int == 3:  # LEFT
        accel_y = -0.1
    elif action_int == 4:  # RIGHT
        accel_y = 0.1

    env.state[2] += accel_x  # Update speed
    env.state[0] += env.state[2]  # Update player_x based on speed
    env.state[1] += accel_y  # Update player_y

    # Update enemy car positions
    for i in range(env.num_enemy_cars):
        env.state[5 + i * 2] -= 0.1  # Move enemy cars down
        if env.state[5 + i * 2] < 0:
            env.state[5 + i * 2] = 1.0  # Reset enemy car position

    reward = 0.0
    done = 0
    env.tick += 1  # Increase tick count
    
    # Create a NumPy array to return the state
    cdef int obs_size = 5 + 2 * env.num_enemy_cars
    cdef cnp.ndarray[cnp.float32_t, ndim=1] state = np.zeros(obs_size, dtype=np.float32)
    
    # Manually copy the C array to the NumPy array
    for i in range(obs_size):
        state[i] = env.state[i]

    return state, reward, done, False, {}

# Free environment memory
cdef void free_racing_env(CRacingEnv* env):
    free(env.state)
    free(env)

cdef class CRacingCy:
    cdef CRacingEnv* env
    cdef int num_enemy_cars

    def __init__(self, int num_enemy_cars):
        self.num_enemy_cars = num_enemy_cars
        self.env = init_racing_env(num_enemy_cars)

    def reset(self):
        reset(self.env)

    def step(self, object action):
        return step(self.env, action)

    def get_state(self):
        cdef int obs_size = 5 + 2 * self.num_enemy_cars
        cdef cnp.ndarray[cnp.float32_t, ndim=1] state = np.zeros(obs_size, dtype=np.float32)
        
        # Manually copy the C array to the NumPy array
        for i in range(obs_size):
            state[i] = self.env.state[i]
        
        return state

    def get_tick(self):
        return self.env.tick

    def __dealloc__(self):
        free_racing_env(self.env)
