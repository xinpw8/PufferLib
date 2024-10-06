# cy_racing_cy.pyx
# cython: language_level=3

import numpy as np
cimport numpy as cnp
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free, rand
from libc.stdint cimport int32_t
import random

cnp.import_array()

# Action definitions
cdef int ACTION_NOOP = 0
cdef int ACTION_ACCEL = 1
cdef int ACTION_DECEL = 2
cdef int ACTION_LEFT = 3
cdef int ACTION_RIGHT = 4

# Define screen dimensions for rendering
cdef int TOTAL_SCREEN_WIDTH = 160
cdef int TOTAL_SCREEN_HEIGHT = 210
cdef int ACTION_SCREEN_X_START = 8
cdef int ACTION_SCREEN_Y_START = 0
cdef int ACTION_SCREEN_WIDTH = 152  # from x=8 to x=160
cdef int ACTION_SCREEN_HEIGHT = 155 # from y=0 to y=155

cdef int SCOREBOARD_X_START = 48
cdef int SCOREBOARD_Y_START = 161
cdef int SCOREBOARD_WIDTH = 64  # from x=48 to x=112
cdef int SCOREBOARD_HEIGHT = 30 # from y=161 to y=191

cdef int CARS_LEFT_X_START = 72
cdef int CARS_LEFT_Y_START = 179
cdef int CARS_LEFT_WIDTH = 32  # from x=72 to x=104
cdef int CARS_LEFT_HEIGHT = 9  # from y=179 to y=188

cdef int DAY_X_START = 56
cdef int DAY_Y_START = 179
cdef int DAY_WIDTH = 8    # from x=56 to x=64
cdef int DAY_HEIGHT = 9   # from y=179 to y=188
cdef int DAY_LENGTH = 300000  # Number of ticks in a day

cdef float ROAD_WIDTH = 90.0
cdef float CAR_WIDTH = 16.0
cdef float PLAYER_CAR_LENGTH = 11.0
cdef float ENEMY_CAR_LENGTH = 11.0
cdef float MAX_SPEED = 100.0
cdef float MIN_SPEED = -10.0
cdef float SPEED_INCREMENT = 5.0
cdef float MAX_Y_POSITION = ACTION_SCREEN_HEIGHT + ENEMY_CAR_LENGTH # Max Y for enemy cars
cdef float MIN_Y_POSITION = 0.0 # Min Y for enemy cars (spawn just above the screen)
cdef float MIN_DISTANCE_BETWEEN_CARS = 40.0  # Minimum Y distance between adjacent enemy cars

cdef float PASS_THRESHOLD = ACTION_SCREEN_HEIGHT  # Distance for passed cars to disappear


cdef struct EnemyCar:
    float rear_bumper_y
    int lane
    int active

cdef struct CRacingEnv:
    float* state  # Pointer to state array (player_x, player_y, speed, day, cars_passed, enemy_car_x, enemy_car_y...)
    float front_bumper_y                # 1
    float left_distance_to_edge         # 2
    float right_distance_to_edge        # 3
    float speed                         # 4
    int dayNumber                       # 5
    int cars_passed                     # 6
    int carsRemainingLo                 # 7
    int carsRemainingHi                 # 8
    int throttleValue                   # 9
    float reward                        # 10
    int done                            # 11
    EnemyCar* enemy_cars                # 12
    float* observations                 # 13
    unsigned char* actions              # 14
    float* rewards                      # 15
    unsigned char* terminals            # 16
    float* player_x_y                   # 17
    float* other_cars_x_y               # 18
    int* other_cars_active              # 19
    unsigned int* score_day             # 20
    int tick                            # Added tick counter
    int day_length                      # Added day_length

# DO NOT MOVE cdef CRacingEnv* OR CHANGE THE STRUCTURE
# FEEL FREE TO ADD THE NECESSARY VARIABLES TO THIS
# Initialize the cython env (does the heavy lifting)
cdef CRacingEnv* init_racing_env():
    cdef CRacingEnv* env = <CRacingEnv*>malloc(sizeof(CRacingEnv))
    env.state = <float*>malloc((7 + 30) * sizeof(float))  # Always for max 15 enemy cars (2 * 15)
    env.enemy_cars = <EnemyCar*>malloc(15 * sizeof(EnemyCar))  # Fixed size of 15 cars
    # printf("Memory allocated for enemy cars.\n")

    env.day_length = DAY_LENGTH
    env.tick = 0
    env.dayNumber = 1
    env.speed = MIN_SPEED
    env.cars_passed = 0
    env.carsRemainingLo = (env.dayNumber + 1) * 100
    env.done = 0

    # Initialize enemy cars, default to inactive
    for i in range(15):
        env.enemy_cars[i].rear_bumper_y = MAX_Y_POSITION + 100.0  # Off-screen start
        env.enemy_cars[i].lane = random.randint(0, 2)
        env.enemy_cars[i].active = 0  # Inactive initially
        # printf("Enemy car %d initialized at lane %d with position y=%f. Active=%d \n", i, env.enemy_cars[i].lane, env.enemy_cars[i].rear_bumper_y, env.enemy_cars[i].active)

    return env



# Modified reset function to no longer depend on num_enemy_cars
cdef void reset(CRacingEnv* env):
    env.tick = 0
    env.dayNumber = 1
    env.speed = MIN_SPEED
    env.cars_passed = 0
    env.carsRemainingLo = (env.dayNumber + 1) * 100
    env.done = 0
    
    # Reset enemy cars
    for i in range(15):
        env.enemy_cars[i].active = 0  # Mark all inactive
    spawn_enemy_cars(env)  # Will activate a subset based on the day and speed

    # printf("Player reset: speed=%f, dayNumber=%d, carsRemainingLo=%d\n", env.speed, env.dayNumber, env.carsRemainingLo)


# Perform one step in the environment
cdef tuple step(CRacingEnv* env, object action):
    # printf("Performing step function. Action: %d\n", <int>action)
    cdef int action_int
    cdef float accel_x = 0.0, lateral_speed = 0.0

    action_int = <int>action
    if action_int == 1:  # ACCEL
        env.speed += SPEED_INCREMENT
        # printf("Accelerating. New speed: %f\n", env.speed)
        if env.speed > MAX_SPEED:
            env.speed = MAX_SPEED
            # printf("Speed capped at MAX_SPEED: %f\n", MAX_SPEED)
    elif action_int == 2:  # DECEL
        env.speed -= SPEED_INCREMENT
        # printf("Decelerating. New speed: %f\n", env.speed)
        if env.speed < MIN_SPEED:
            env.speed = MIN_SPEED
            # printf("Speed capped at MIN_SPEED: %f\n", MIN_SPEED)

    elif action_int == 3:  # LEFT
        lateral_speed = 1 + (env.speed / MAX_SPEED * 5)
        env.left_distance_to_edge -= lateral_speed
        if env.left_distance_to_edge < 0:
            env.left_distance_to_edge = 0
        env.right_distance_to_edge = ROAD_WIDTH - CAR_WIDTH - env.left_distance_to_edge
    elif action_int == 4:  # RIGHT
        lateral_speed = 1 + (env.speed / MAX_SPEED * 5)
        env.right_distance_to_edge -= lateral_speed
        if env.right_distance_to_edge < 0:
            env.right_distance_to_edge = 0
        env.left_distance_to_edge = ROAD_WIDTH - CAR_WIDTH - env.right_distance_to_edge

    # Update enemy car positions for all 15 cars
    for i in range(15):
        if env.enemy_cars[i].active:
            env.enemy_cars[i].rear_bumper_y -= env.speed * 0.1

            if env.enemy_cars[i].rear_bumper_y < env.front_bumper_y - PASS_THRESHOLD:
                env.enemy_cars[i].active = 0  # Car passed
                env.carsRemainingLo -= 1

            if env.enemy_cars[i].rear_bumper_y < MIN_Y_POSITION:
                env.enemy_cars[i].rear_bumper_y = MAX_Y_POSITION + random.random() * 500
                env.enemy_cars[i].lane = rand() % 3

    # Handle collision detection and rewards
    reward = env.speed * 0.01
    for i in range(15):
        if env.enemy_cars[i].active:
            enemy_lane_center = (env.enemy_cars[i].lane + 0.5) * (ROAD_WIDTH / 3.0)
            enemy_left_edge = enemy_lane_center - (CAR_WIDTH / 2)
            enemy_right_edge = enemy_lane_center + (CAR_WIDTH / 2)
            player_left_edge = env.left_distance_to_edge
            player_right_edge = ROAD_WIDTH - env.right_distance_to_edge

            # Check for lateral and longitudinal overlap for collision detection
            lateral_overlap = not (player_right_edge <= enemy_left_edge or player_left_edge >= enemy_right_edge)
            enemy_front_bumper_y = env.enemy_cars[i].rear_bumper_y + ENEMY_CAR_LENGTH
            longitudinal_overlap = not (env.front_bumper_y + PLAYER_CAR_LENGTH <= env.enemy_cars[i].rear_bumper_y or env.front_bumper_y >= enemy_front_bumper_y)

            if lateral_overlap and longitudinal_overlap:
                env.speed -= (env.speed - MIN_SPEED) / 30.0  # Gradually reduce speed on collision
                if env.speed < MIN_SPEED:
                    env.speed = MIN_SPEED
                reward = -10.0
                env.done = 1
                break

    # Handle day progression
    if env.carsRemainingLo <= 0:
        env.dayNumber += 1
        env.carsRemainingLo = 200

    env.tick += 1

    # Apply placeholder weather conditions
    apply_weather_conditions(env)  # Call the placeholder function
    
    cdef cnp.ndarray[cnp.float32_t, ndim=1] np_state = np.zeros((7 + 30,), dtype=np.float32)    
    
    # Copy the C array to the NumPy array (el scuffo)
    for i in range(7 + 30):
        np_state[i] = env.state[i]

    # Spawn enemy cars
    spawn_enemy_cars(env)

    # Day length logic
    if env.tick >= env.day_length:
        if env.carsRemainingLo > 0:
            env.done = 1  # Game over: time ran out
        else:
            env.dayNumber += 1  # New day
            env.carsRemainingLo = (env.dayNumber + 1) * 100
            env.tick = 0  # Reset day timer

    env.tick += 1  # Increment tick count

    # Prepare the observation array to reflect the environment
    compute_observations(env)

    return np_state, reward, env.done, False, {}



# Spawn cars based on player speed and day number    
# We want the code to run if num_active_cars is < the day's max
# but it should only have a probability of successfully spawning
# The probability is higher with higher speeds (which makes sense, seeing as though
# a player going as fast as possible would be encountering the most cars)

cdef void spawn_enemy_cars(CRacingEnv* env):
    cdef int num_active_cars = min(env.dayNumber + 2, 15)  # Limit of dayNumber + 2, capped at 15
    cdef float spawn_probability
    # printf("Spawning enemy cars.\n")
    # printf("Day number: %d, Speed: %f, Active cars: %d\n", env.dayNumber, env.speed, num_active_cars)
    
    # Calculate spawn probability based on speed
    spawn_probability = 0.1 + ((env.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)) * 0.8  # from 0.1 to 0.9
    # printf("Spawn probability: %f\n", spawn_probability)

    for i in range(num_active_cars):
        if random.random() < spawn_probability:
            env.enemy_cars[i].rear_bumper_y = MAX_Y_POSITION + random.random() * 500
            env.enemy_cars[i].lane = random.randint(0, 2)
            env.enemy_cars[i].active = 1
            # printf("Enemy car %d spawned.\n", i)
        else:
            env.enemy_cars[i].active = 0
            # printf("Enemy car %d not spawned.\n", i)



# TODO: Implement apply_weather_conditions
cdef void apply_weather_conditions(CRacingEnv* env):
    pass

cdef void _check_observation(float* obs, int size):
    cdef float low = -100
    cdef float high = 1000
    cdef int i

    for i in range(size):
        if obs[i] < low or obs[i] > high:
            # printf("Warning: Observation out of range: %f\n", obs[i])
            obs[i] = -1


cdef void compute_observations(CRacingEnv* env):
    # printf("Computing observations.\n")
    cdef int obs_index = 0

    # Default initialization of the state
    for i in range(7 + 30):  # This is the size of the observation array
        env.state[i] = -1.0  # Default to a known, safe value
        # printf("State[%d] = %f\n", i, env.state[i])

    # Player data
    env.state[0] = env.left_distance_to_edge
    env.state[1] = env.front_bumper_y
    env.state[2] = env.speed
    env.state[3] = env.dayNumber
    env.state[4] = env.carsRemainingLo

    # Enemy car data, always 15 entries (fixed size)
    obs_index = 5
    for i in range(15):
        if env.enemy_cars[i].active:
            env.state[obs_index] = env.enemy_cars[i].lane
            env.state[obs_index + 1] = env.enemy_cars[i].rear_bumper_y
        else:
            env.state[obs_index] = -1.0  # Safe placeholder for inactive cars
            env.state[obs_index + 1] = -1.0
        obs_index += 2

    # Check the observations
    _check_observation(env.state, 7 + 30)


# Free environment memory
cdef void free_racing_env(CRacingEnv* env):
    free(env.state)
    free(env)

# DO NOT MOVE cdef class CRacingCy OR CHANGE THE STRUCTURE
# FEEL FREE TO ADD THE NECESSARY VARIABLES TO THIS
# The Python wrapper for the environment
cdef class CRacingCy:
    cdef CRacingEnv* env

    def __init__(self):
        self.env = init_racing_env()  # Call without num_enemy_cars

    def reset(self):
        reset(self.env)

    def step(self, object action):
        return step(self.env, action)

    def get_state(self):
        # Observation size now fixed for 15 enemy cars
        cdef int obs_size = 7 + 30  # 15 enemy cars, 2 values per car
        cdef cnp.ndarray[cnp.float32_t, ndim=1] state = np.zeros(obs_size, dtype=np.float32)
        
        for i in range(obs_size):
            state[i] = self.env.state[i]
        
        return state

    def get_tick(self):
        return self.env.tick

    def __dealloc__(self):
        if self.env != NULL:
            free(self.env.state)
            free(self.env.enemy_cars)
            free(self.env)
